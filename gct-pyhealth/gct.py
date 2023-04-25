import numpy as np
import os
import sys
import math
import torch
from typing import List, Dict, Union, Callable

from torch import nn
from pyhealth.datasets import eICUDataset
from pyhealth.datasets import BaseEHRDataset
from pyhealth.models.base_model import BaseModel

VALID_OPERATION_LEVEL = ["visit", "event", "patient"]


class FeatureEmbedder(nn.Module):
    def __init__(self, feature_keys, vocab_sizes, embedding_dim, hidden_dropout_prob):
        super().__init__()
        self.embeddings = {}
        self.feature_keys = feature_keys
        self.dx_embeddings = nn.Embedding(vocab_sizes['dx_ints'] + 1, embedding_dim,
                                          padding_idx=vocab_sizes['dx_ints'])
        self.proc_embeddings = nn.Embedding(vocab_sizes['proc_ints'] + 1, embedding_dim,
                                            padding_idx=vocab_sizes['proc_ints'])
        self.visit_embeddings = nn.Embedding(1, embedding_dim)

        # stuff to try when everything is done as add-on
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, features):
        batch_size = features[self.feature_keys[0]].shape[0]
        embeddings = {}
        masks = {}

        embeddings['dx_ints'] = self.dx_embeddings(features['dx_ints'])
        embeddings['proc_ints'] = self.proc_embeddings(features['proc_ints'])
        device = features['dx_ints'].device

        embeddings['visit'] = self.visit_embeddings(torch.tensor([0]).to(device))
        embeddings['visit'] = embeddings['visit'].unsqueeze(0).expand(batch_size, -1, -1)
        masks['visit'] = torch.ones(batch_size, 1).to(device)
        for name, embedding in embeddings.items():
            embeddings[name] = self.layernorm(embedding)
            embeddings[name] = self.dropout(embeddings[name])

        return embeddings, masks


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_dim, stack_idx):
        super().__init__()
        self.stack_idx = stack_idx
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embedding_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embedding_dim, self.all_head_size)
        self.key = nn.Linear(embedding_dim, self.all_head_size)
        self.value = nn.Linear(embedding_dim, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        if self.stack_idx == 0 and prior is not None:
            attention_probs = prior[:, None, :, :].expand(-1, self.num_attention_heads, -1, -1)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            query_layer = self.transpose_for_scores(mixed_query_layer)
            key_layer = self.transpose_for_scores(mixed_key_layer)
            # take dot product between query and key to get raw attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)

        mixed_value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # dropping out entire tokens to attend to; extra experiment
        # attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SelfOutput(nn.Module):
    def __init__(self, embedding_dim, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, num_heads, embedding_dim, hidden_dropout_prob, stack_idx):
        super().__init__()
        self.self_attention = SelfAttention(num_heads, embedding_dim, stack_idx)
        self.self_output = SelfOutput(embedding_dim, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.self_attention(hidden_states, attention_mask, guide_mask, prior,
                                                     output_attentions)
        attention_output = self.self_output(self_attention_outputs[0], hidden_states)
        outputs = (attention_output,) + self_attention_outputs[1:]
        return outputs


class GCTLayer(nn.Module):
    def __init__(self, num_heads, embedding_dim, hidden_dropout_prob, stack_idx):
        super().__init__()
        self.attention = Attention(num_heads, embedding_dim, hidden_dropout_prob, stack_idx)

    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.attention(hidden_states, attention_mask, guide_mask, prior, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]
        outputs = (attention_output,) + outputs
        return outputs


class Pooler(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.dense = nn.Linear(embedding_dim, embedding_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GCT(BaseModel):
    """GCT model.

    Paper: Edward Choi et al. Learning the Graphical Structure of Electronic Health Records with
    Graph Convolutional Transformer. AAAI 2020.

    Note:

    Args:
        dataset: the eICU dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["diagnosisString", "admissionDx", "treatment"].
        label_key: key in samples to use as label (e.g., "readmission", "mortality").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters for the GCT layer.
    """

    def __init__(
            self,
            dataset: BaseEHRDataset,
            feature_keys: List[str] = ["dx_ints", "proc_ints"],
            label_key: str = "readmission",
            mode: str = "binary",
            embedding_dim: int = 128,
            max_num_codes: int = 50,
            num_stacks: int = 2,
            batch_size: int = 32,
            reg_coef: float = 0.1,
            prior_scalar: float = 0.5,
            hidden_dropout_prob: float = 0.08,
            num_heads: int = 1,
            **kwargs
    ):

        super(GCT, self).__init__(
            dataset=dataset,  # TODO
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,  # TODO
        )

        self.embedding_dim = embedding_dim
        self.num_labels = 2  # TODO args.num_labels
        self.label_key = label_key  # TODO args.label_key
        self.feature_keys = feature_keys  # TODO
        self.vocab_sizes = {'dx_ints': 3249, 'proc_ints': 2210}  # TODO

        self.num_heads = num_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.batch_size = batch_size
        self.num_stacks = num_stacks
        self.reg_coef = reg_coef
        self.prior_scalar = prior_scalar
        self.max_num_codes = max_num_codes

        self.layers = nn.ModuleList(
            [GCTLayer(self.num_heads, self.embedding_dim, self.hidden_dropout_prob, i) for i in range(self.num_stacks)])
        self.embeddings = FeatureEmbedder(self.feature_keys, self.vocab_sizes, self.embedding_dim,
                                          self.hidden_dropout_prob)
        self.pooler = Pooler(self.embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.embedding_dim, self.num_labels)

    def create_matrix_vdp(self, features, masks, priors):
        batch_size = features['dx_ints'].shape[0]
        device = features['dx_ints'].device
        num_dx_ids = self.max_num_codes
        num_proc_ids = self.max_num_codes
        num_codes = 1 + num_dx_ids + num_proc_ids

        row0 = torch.cat([torch.zeros([1, 1]), torch.ones([1, num_dx_ids]), torch.zeros([1, num_proc_ids])], axis=1)
        row1 = torch.cat([torch.zeros([num_dx_ids, num_dx_ids + 1]), torch.ones([num_dx_ids, num_proc_ids])],
                         axis=1)
        row2 = torch.zeros([num_proc_ids, num_codes])

        guide = torch.cat([row0, row1, row2], axis=0)
        guide = guide + guide.t()
        guide = guide.to(device)

        guide = guide.unsqueeze(0)
        guide = guide.expand(batch_size, -1, -1)
        guide = (guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + torch.eye(num_codes).to(device).unsqueeze(0))

        prior_idx = priors['indices'].t()
        temp_idx = (prior_idx[:, 0] * 100000 + prior_idx[:, 1] * 1000 + prior_idx[:, 2])
        sorted_idx = torch.argsort(temp_idx)
        prior_idx = prior_idx[sorted_idx]

        prior_idx_shape = [batch_size, self.max_num_codes * 2, self.max_num_codes * 2]
        sparse_prior = torch.sparse.FloatTensor(prior_idx.t(), priors['values'], torch.Size(prior_idx_shape))
        prior_guide = sparse_prior.to_dense()

        visit_guide = torch.tensor([self.prior_scalar] * self.max_num_codes + [0.0] * self.max_num_codes * 1,
                                   dtype=torch.float, device=device)
        prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1), prior_guide],
                                axis=1)
        visit_guide = torch.cat([torch.tensor([0.0], device=device), visit_guide], axis=0)
        prior_guide = torch.cat([visit_guide.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1), prior_guide],
                                axis=2)
        prior_guide = (prior_guide * masks.unsqueeze(-1) * masks.unsqueeze(1) + self.prior_scalar * torch.eye(
            num_codes, device=device).unsqueeze(0))
        degrees = torch.sum(prior_guide, axis=2)
        prior_guide = prior_guide / degrees.unsqueeze(-1)

        return guide, prior_guide

    def get_loss(self, logits, labels, attentions):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        kl_terms = []
        for i in range(1, self.num_stacks):
            log_p = torch.log(attentions[i - 1] + 1e-12)
            log_q = torch.log(attentions[i] + 1e-12)
            kl_term = attentions[i - 1] * (log_p - log_q)
            kl_term = torch.sum(kl_term, axis=-1)
            kl_term = torch.mean(kl_term)
            kl_terms.append(kl_term)
        reg_term = torch.mean(torch.tensor(kl_terms))
        loss += self.reg_coef * reg_term
        return loss

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, data, priors_data):
        # TODO
        """ Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
            ------
            patient_emb = torch.cat(patient_emb, dim=1)
            # (patient, label_size)
            logits = self.fc(patient_emb)
            # obtain y_true, loss, y_prob
            y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
            loss = self.get_loss_function()(logits, y_true)
            y_prob = self.prepare_y_prob(logits)
            results = {
                "loss": loss,
                "y_prob": y_prob,
                "y_true": y_true,
                "logit": logits,
            }
            if kwargs.get("embed", False):
                results["embed"] = patient_emb
        """

        embedding_dict, mask_dict = self.embeddings(data)
        mask_dict['dx_ints'] = data['dx_masks']
        mask_dict['proc_ints'] = data['proc_masks']

        keys = ['visit'] + self.feature_keys
        hidden_states = torch.cat([embedding_dict[key] for key in keys], axis=1)
        masks = torch.cat([mask_dict[key] for key in keys], axis=1)

        guide, prior_guide = self.create_matrix_vdp(data, masks, priors_data)

        all_hidden_states = ()
        all_attentions = ()
        # make attention_mask, guide_mask
        extended_attention_mask = self.get_extended_attention_mask(masks)
        extended_guide_mask = self.get_extended_attention_mask(guide)

        # multiple stacked attention layers
        for i, layer_module in enumerate(self.layers):
            all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, extended_attention_mask, extended_guide_mask, prior_guide)
            hidden_states = layer_outputs[0]
            all_attentions = all_attentions + (layer_outputs[1],)
        all_hidden_states = all_hidden_states + (hidden_states,)

        pooled_output = self.pooler(hidden_states)

        # get logits and loss
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.get_loss(logits, data[self.label_key], all_attentions)

        return tuple(v for v in [loss, logits, all_hidden_states, all_attentions] if v is not None)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor representing the predicted probabilities.
                y_true: a tensor representing the true labels.
        """
        patient_emb = []
        for feature_key in self.feature_keys:
            input_info = self.dataset.input_info[feature_key]
            dim_, type_ = input_info["dim"], input_info["type"]

            # for case 1: [code1, code2, code3, ...]
            if (dim_ == 2) and (type_ == str):
                raise NotImplementedError

            # for case 2: [[code1, code2], [code3, ...], ...]
            elif (dim_ == 3) and (type_ == str):
                raise NotImplementedError


            # for case 3: [[1.5, 2.0, 0.0], ...]
            elif (dim_ == 2) and (type_ in [float, int]):
                raise NotImplementedError


            # for case 4: [[[1.5, 2.0, 0.0], [1.8, 2.4, 6.0]], ...]
            elif (dim_ == 3) and (type_ in [float, int]):
                raise NotImplementedError
            else:
                raise NotImplementedError

            # _, x = self.transformer[feature_key](x, mask)
            patient_emb.append(x)

        # computed patient embedding
        patient_emb = torch.cat(patient_emb, dim=1)

        # get logits and loss
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # obtain y_true, loss, y_prob
        y_true = self.prepare_labels(kwargs[self.label_key], self.label_tokenizer)
        loss = self.get_loss(logits, data[self.label_key], all_attentions)
        y_prob = self.prepare_y_prob(logits)
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import eICUDataset

    pass
