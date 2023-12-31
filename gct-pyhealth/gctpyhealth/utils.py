import numpy as np
import os
import sys
import math
import torch
import json
import random
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import argparse

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    auc,
    roc_curve,
    precision_recall_curve)


class eICUPriorDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def priors_collate_fn(batch):
    # now, we want to return a big tensor with (batch_idx, idx_1, idx_2) like in gct code
    new_batch = []
    for i, item in enumerate(batch):
        num_indices = item[0].shape[-1]
        new_indices = torch.cat((torch.tensor([i] * num_indices).reshape(1, -1), item[0]), axis=0)
        new_batch.append((new_indices, item[1]))
    indices = torch.cat([t[0] for t in new_batch], axis=1)
    values = torch.cat([t[1] for t in new_batch], axis=-1)
    return indices, values


# def get_extended_attention_mask(attention_mask):
#     if attention_mask.dim() == 2:
#         extended_attention_mask = attention_mask[:, None, None, :]
#     elif attention_mask.dim() == 3:
#         extended_attention_mask = attention_mask[:, None, :, :]
#     extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
#     return extended_attention_mask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def nested_concat(tensors, new_tensors, dim=0):
    """Concat the `new_tensors` to `tensors` on `dim`. Works for tensors or nested list/tuples of tensors."""
    assert type(tensors) == type(
        new_tensors
    ), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def prepare_data(data, priors_data, device):
    features = {'conditions_hash': data[0],
                'procedures_hash': data[1],
                'conditions_masks': data[2],
                'procedures_masks': data[3],
                'readmission': data[4],
                'expired': data[5]}
    for k, v in features.items():
        features[k] = v.to(device)
    priors = {'indices': priors_data[0].to(device),
              'values': priors_data[1].to(device)}

    return features, priors


def compute_metrics(preds, labels):
    metrics = {}
    preds = np.argmax(preds, axis=1)
    # average precision
    ap = average_precision_score(labels, preds)
    # metrics['AP'] = ap

    # auprc
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    auc_pr = auc(recalls, precisions)
    metrics['AUCPR'] = auc_pr

    # auroc
    fpr, tpr, thresholds = roc_curve(labels, preds)
    auc_roc = auc(fpr, tpr)
    metrics['AUROC'] = auc_roc

    # f1 score, precision, recall
    # precision, recall, fscore, support = precision_recall_fscore_support(labels, preds, average='weighted')
    # metrics['precision'] = precision
    # metrics['recall'] = recall
    # metrics['fscore'] = fscore

    return metrics


class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--learning_rate', type=float, default=1e-3)
        self.add_argument('--eps', type=float, default=1e-8)
        self.add_argument('--max_grad_norm', type=float, default=1.0)
        self.add_argument('--intermediate_size', type=int, default=256)

        self.add_argument('--eval_batch_size', type=int, default=256)

        self.add_argument('--warmup', type=float, default=0.05)
        self.add_argument('--logging_steps', type=int, default=100)
        self.add_argument('--max_steps', type=int, default=100000)
        self.add_argument('--num_train_epochs', type=int, default=0)

        self.add_argument('--seed', type=int, default=42)

        self.add_argument('--do_train', default=False, action='store_true')
        self.add_argument('--do_eval', default=False, action='store_true')
        self.add_argument('--do_test', default=False, action='store_true')

        # integrate cmd line arguments
        self.add_argument('--label_key', type=str, default='expired')
        self.add_argument('--num_stacks', type=int, default=3)
        self.add_argument('--num_heads', type=int, default=1)
        self.add_argument('--post_mlp_dropout', type=float, default=0.2)

        # loading prev model
        self.add_argument('--load_prev_model', default=False, action='store_true')
        self.add_argument('--prev_model_path', type=str, default="eicu_output/model_pyhealth/model.p")

    def parse_args(self):
        args = super().parse_args()
        return args


def prediction_loop(device, label_key, model, dataloader, priors_dataloader, description='Evaluating'):
    from tqdm import tqdm
    batch_size = dataloader.batch_size
    eval_losses = []
    preds = None
    label_ids = None
    model.eval()

    for data, priors_data in tqdm(zip(dataloader, priors_dataloader), desc=description):
        data, priors_data = prepare_data(data, priors_data, device)
        with torch.no_grad():
            outputs = model(data, priors_data)
            loss = outputs['loss'].mean().item()
            logits = outputs['logit']

        labels = data[label_key]

        batch_size = data[list(data.keys())[0]].shape[0]
        eval_losses.extend([loss] * batch_size)
        preds = logits if preds is None else nested_concat(preds, logits, dim=0)
        label_ids = labels if label_ids is None else nested_concat(label_ids, labels, dim=0)

    if preds is not None:
        preds = nested_numpify(preds)
    if label_ids is not None:
        label_ids = nested_numpify(label_ids)
    metrics = compute_metrics(preds, label_ids)

    metrics['eval_loss'] = np.mean(eval_losses)

    for key in list(metrics.keys()):
        if not key.startswith('eval_'):
            metrics['eval_{}'.format(key)] = metrics.pop(key)

    return metrics
