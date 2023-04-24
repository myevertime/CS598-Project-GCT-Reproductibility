import numpy as np
import os
import sys
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from pyhealth.datasets import SampleEHRDataset
from pyhealth.models import BaseModel
from pyhealth.tokenizer import Tokenizer


class FeatureEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embeddings = {}
        self.feature_keys = args.feature_keys
        self.dx_embeddings = nn.Embedding(args.vocab_sizes['dx_ints'] + 1, args.hidden_size,
                                          padding_idx=args.vocab_sizes['dx_ints'])
        self.proc_embeddings = nn.Embedding(args.vocab_sizes['proc_ints'] + 1, args.hidden_size,
                                            padding_idx=args.vocab_sizes['proc_ints'])
        self.visit_embeddings = nn.Embedding(1, args.hidden_size)

        ## stuff to try when everything is done as add-on
        self.layernorm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

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
    def __init__(self, args, stack_idx):
        super().__init__()
        self.stack_idx = stack_idx
        self.num_attention_heads = args.num_heads
        self.attention_head_size = int(args.hidden_size / args.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)
        # experiment with dropout after completion
        # self.dropout = nn.Dropout(args.hidden_dropout_prob)

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
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.self_attention = SelfAttention(args, stack_idx)
        self.self_output = SelfOutput(args)

    def forward(self, hidden_states, attention_mask, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.self_attention(hidden_states, attention_mask, guide_mask, prior,
                                                     output_attentions)
        attention_output = self.self_output(self_attention_outputs[0], hidden_states)
        outputs = (attention_output,) + self_attention_outputs[1:]
        return outputs


class IntermediateLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.intermediate_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.intermediate_size, args.hidden_size)
        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.activation(self.dense(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class GCTLayer(nn.Module):
    def __init__(self, args, stack_idx):
        super().__init__()
        self.attention = Attention(args, stack_idx)
        # self.intermediate = IntermediateLayer(args)
        # self.output = OutputLayer(args)

    def forward(self, hidden_states, attention_mask=None, guide_mask=None, prior=None, output_attentions=True):
        self_attention_outputs = self.attention(hidden_states, attention_mask, guide_mask, prior, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        # intermediate_output = self.intermediate(attention_output)
        # layer_output = self.output(intermediate_output, attention_output)

        # outputs = (layer_output,) + outputs
        outputs = (attention_output,) + outputs
        return outputs


class Pooler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
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
            dataset: SampleEHRDataset,
            feature_keys: List[str],
            label_key: str,
            mode: str,
            embedding_dim: int = 128,
            **kwargs
    ):
        super(GCT, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.embedding_dim = embedding_dim

        # Parse arguments
        self.args.label_key = label_key
        self.args.num_labels = 2
        self.args.reg_coef = kwargs['reg_coef']
        self.args.use_guide = True
        self.args.use_prior = True
        self.args.prior_scalar = 0.5
        self.args.batch_size = 64
        self.args.num_stacks = 3
        self.args.max_num_codes = 50
        self.args.output_attentions = True
        self.args.output_hidden_states = True
        self.args.feature_keys = ['dx_ints', 'proc_ints']

        self.layers = nn.ModuleList([GCTLayer(self.args, i) for i in range(self.num_stacks)])
        self.embeddings = FeatureEmbedder(self.args)
        self.pooler = Pooler(self.args)

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
        pass

    def get_loss(self, logits, labels, attentions):
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if self.use_prior:
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

    def create_matrix_vdp(self, features, masks, priors):
        batch_size = features['dx_ints'].shape[0]
        device = features['dx_ints'].device
        num_dx_ids = self.max_num_codes if self.use_prior else features['dx_ints'].shape[-1]
        num_proc_ids = self.max_num_codes if self.use_prior else features['proc_ints'].shape[-1]
        num_codes = 1 + num_dx_ids + num_proc_ids

        guide = None
        prior_guide = None

        if self.use_guide:
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

        if self.use_prior:
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


if __name__ == "__main__":
    pass
