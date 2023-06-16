# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BEiT model. """


import collections.abc
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import zCurve
import hilbert

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from einops import rearrange, repeat

from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging
from models.sparse_config import BeitConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BeitConfig"
_CHECKPOINT_FOR_DOC = "microsoft/beit-base-patch16-224"

BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/beit-base-patch16-224",
    # See all BEiT models at https://huggingface.co/models?filter=beit
]


@dataclass
class BeitModelOutputWithPooling(BaseModelOutputWithPooling):
    """
    Class for outputs of :class:`~transformers.BeitModel`.

    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Average of the last layer hidden states of the patch tokens (excluding the `[CLS]` token) if
            `config.use_mean_pooling` is set to True. If set to False, then the final hidden state of the `[CLS]` token
            will be returned.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    token_idx: Optional[Tuple[torch.LongTensor]] = None


@dataclass
class BeitModelOutput(BaseModelOutput):
    token_idx: Optional[Tuple[torch.LongTensor]] = None


# Inspired by
# https://github.com/rwightman/pytorch-image-models/blob/b9bd960a032c75ca6b808ddeed76bee5f3ed4972/timm/models/layers/helpers.py
# From PyTorch internals
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


# Based on https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class BeitEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        if config.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        else:
            self.mask_token = None
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        if config.use_absolute_position_embeddings:
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        else:
            self.position_embeddings = None
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values, bool_masked_pos=None):

        if pixel_values.ndim == 5:  # video input=
            embeddings = self.patch_embeddings(pixel_values.flatten(0, 1))
            embeddings = rearrange(embeddings, '(b m) n d -> b (m n) d', m=pixel_values.shape[1])
        else:  # image input
            embeddings = self.patch_embeddings(pixel_values)
            
        batch_size, seq_len, _ = embeddings.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1 - w) + mask_tokens * w

        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


# Based on timm implementation, which can be found here:
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_shape = patch_shape

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)

        return x


class BeitSelfAttention(nn.Module):
    def __init__(self, config, window_size=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # sparse params
        self.random_attn = config.sparse_random_attn
        self.local_attn = config.sparse_local_attn
        self.block_size = config.attn_block_size
        self.num_cls_tokens = config.num_cls_tokens
        if self.local_attn is not None and self.random_attn is not None:
            self.num_kv_blocks = self.local_attn + self.random_attn

        if window_size:
            self.relative_position_bias = BeitRelativePositionBias3D(config, window_size=window_size)
        else:
            self.relative_position_bias = None
    
    def split_heads(self, x):
        return rearrange(x, 'b n (h d) -> b h n d', h=self.num_attention_heads)
    
    def join_heads(self, x):
        return rearrange(x, 'b h n d -> b n (h d)')
    
    def blockify(self, x):
        assert x.dim() == 4, f"Unsupported input shape {x.shape}"
        seq_len = x.shape[2]
        if seq_len % self.block_size > 0:  # seq_len not divisible by block_size, zero pad
            pad_len = self.block_size - seq_len % self.block_size
            x = nn.functional.pad(x, (0, 0, 0, pad_len))
        else:
            pad_len = 0
        x = rearrange(x, 'b h (m n) d -> b h m n d', n=self.block_size)
        return x, pad_len
    
    def dense_attention(self, q, k, v, head_mask=None, relative_position_bias=None, q_idx=None, k_idx=None):
        # q, k, v: (bsz, num_heads, seq_len, dims)
        assert k.shape[2] == v.shape[2], "Key and value shapes mismatch"
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim / math.sqrt(self.attention_head_size)

        # Add relative position bias if present.
        if self.relative_position_bias is not None:
            if q_idx is not None and q_idx.ndim == 2:
                assert k_idx is not None and len(q_idx) == len(k_idx)
                bias = torch.stack([
                    self.relative_position_bias(from_idx=q_idx_, to_idx=k_idx_)
                    for q_idx_, k_idx_ in zip(q_idx, k_idx)
                ])
            else:
                bias = self.relative_position_bias(from_idx=q_idx, to_idx=k_idx).unsqueeze(0)
            sim = sim + bias

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            sim = sim + relative_position_bias

        # Normalize the attention scores to probabilities.
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        return out, attn
    
    def _sparse_attn_relative_position_bias(self, q_idx, pad_q, attn_idx, group_len):
        q_idx_blk = nn.functional.pad(q_idx, (0, pad_q)).view(-1, self.block_size)
        attn_idx_flt = rearrange(q_idx_blk[attn_idx], 'm n j -> m (n j)')  # (seq_len, num_kv_blocks * group_len)
        cls_idx = torch.arange(self.num_cls_tokens, device=q_idx.device)
        cls_idx = repeat(cls_idx, 'n -> m n', m=len(attn_idx_flt))
        attn_idx_flt = torch.cat((cls_idx, attn_idx_flt), dim=1)
        attn_idx_flt = repeat(attn_idx_flt, 'm n -> (m i) n', i=group_len)
        if pad_q > 0:
            attn_idx_flt = attn_idx_flt[:-pad_q]
        bias_flt = self.relative_position_bias(from_idx=q_idx, to_idx=attn_idx_flt)
        if pad_q > 0:
            bias_flt = nn.functional.pad(bias_flt, (0, 0, 0, pad_q))
        return rearrange(bias_flt, 'h (m i) n -> h m i n', i=group_len)  # num_heads, seq_len, group_len, (num_kv_blocks * group_len + num_cls_tokens)
    
    def sparse_attention(self, q, k, v, head_mask=None, relative_position_bias=None, q_idx=None, mimic_full=False):
        assert self.local_attn == 0 or self.local_attn % 2 == 1, "Even local window size not supported"
        assert k.shape[2] == v.shape[2], "Key and value shapes mismatch"

        
        if not mimic_full:
            cls_k, k = k[..., :self.num_cls_tokens, :], k[..., self.num_cls_tokens:, :]  # cls_k: (bsz, num_heads, num_cls_tokens, dims)
            cls_v, v = v[..., :self.num_cls_tokens, :], v[..., self.num_cls_tokens:, :]

        # pad token sequence to multiples of block_size
        if mimic_full:
            bsz, num_heads, seq_len, dims = q.shape
        else:
            q, pad_q = self.blockify(q)  # q: (bsz, num_heads, seq_len, group_len, dims)
            k, pad_k = self.blockify(k)
            v, pad_v = self.blockify(v)
            bsz, num_heads, seq_len, group_len, dims = q.shape

            # global attention
            cls_sim = torch.einsum('b h n i d, b h j d -> b h n i j', q, cls_k)  # (bsz, num_heads, seq_len, group_len, num_cls_tokens)

        if mimic_full:
            sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)
            sim = sim / math.sqrt(self.attention_head_size)
            sim = sim + self.relative_position_bias(from_idx=q_idx).unsqueeze(0)
        
        else:
            # initialize empty sim matrix
            sim = torch.empty((bsz, num_heads, seq_len, self.num_kv_blocks, group_len, group_len), device=q.device)
            attn_idx = torch.zeros((seq_len, self.num_kv_blocks), dtype=torch.int64, device=q.device)

            # local window attention
            cnt = 0
            if self.local_attn > 0:
                num_rolls = self.local_attn // 2
                for r in range(-num_rolls, num_rolls + 1):
                    sim[..., cnt, :, :] = torch.einsum('b h n i d, b h n j d -> b h n i j', q, k.roll(-r, dims=2))
                    attn_idx[:, cnt] = torch.arange(seq_len, device=q.device).roll(r)
                    cnt += 1
            
            # random attention
            if self.random_attn > 0:
                # generate random attention pattern
                rand = torch.rand((seq_len, seq_len), device=q.device)
                if self.local_attn > 0:
                    # avoid overlap with local attention
                    for r in range(-num_rolls, num_rolls + 1):
                        tgt_idx = list(i % seq_len for i in range(r, seq_len + r))
                        rand[range(seq_len), tgt_idx] = 0
                _, idx = rand.topk(self.random_attn, dim=-1)  # seq_len, random_attn
                idx, _ = torch.sort(idx, dim=1)
                attn_idx[:, cnt:] = idx

                idx_ = repeat(idx, 'n m -> b h n m i d', b=bsz, h=num_heads, i=group_len, d=dims)

                for r in range(self.random_attn):
                    sim[..., cnt, :, :] = torch.einsum('b h n i d, b h n j d -> b h n i j', q, k.gather(2, idx_[..., r, :, :]))
                    cnt += 1

            sim = rearrange(sim, 'b h m n i j -> b h m i (n j)')  # (bsz, num_heads, seq_len, group_len, num_kv_blocks * group_len)
            sim = torch.cat((cls_sim, sim), -1)
            sim = sim / math.sqrt(self.attention_head_size)

            # Add relative position bias if present.
            # NOTE: we assume q and k (excluding cls) use same token indexing, for relative position embedding
            if self.relative_position_bias is not None:
                assert q_idx is not None, "query index required for relative position bias"
                if q_idx.ndim == 2:
                    # different indices for each sample
                    bias = torch.stack([
                        self._sparse_attn_relative_position_bias(q_idx_, pad_q, attn_idx, group_len)
                        for q_idx_ in q_idx
                    ])
                else:
                    bias = self._sparse_attn_relative_position_bias(q_idx, pad_q, attn_idx, group_len).unsqueeze(0)
                sim = sim + bias

        # Add shared relative position bias if provided.
        if relative_position_bias is not None:
            raise NotImplementedError
            sim = sim + relative_position_bias

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        # block attention
        if mimic_full:
            out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        else:
            out = torch.empty((bsz, num_heads, seq_len, group_len, dims), device=q.device)
            for m in range(seq_len):
                v_row = torch.index_select(v, 2, attn_idx[m])
                v_row = rearrange(v_row, 'b h n j d -> b h (n j) d')  # (bsz, num_heads, num_kv_blocks * group_len, dims)
                v_row = torch.cat((cls_v, v_row), 2)
                out[..., m, :, :] = torch.einsum('b h i j, b h j d -> b h i d', attn[..., m, :, :], v_row)
            out = rearrange(out, 'b h n i d -> b h (n i) d')
            if pad_q > 0:
                out = out[..., :-pad_q, :]

        return out, attn
        
    def forward(self, hidden_states, head_mask=None, output_attentions=False, relative_position_bias=None, token_idx=None):
        # compute qkv
        q = self.split_heads(self.query(hidden_states))
        k = self.split_heads(self.key(hidden_states))
        v = self.split_heads(self.value(hidden_states))
        
        # combine local token_idx with cls tokens
        # NOTE: assume token_idx starts from 0
        cls_q_idx = torch.arange(self.num_cls_tokens, device=q.device)
        if token_idx is not None:
            if token_idx.ndim == 2:
                cls_q_idx = repeat(cls_q_idx, 'n -> b n', b=q.shape[0])
            all_token_idx = torch.cat((cls_q_idx, token_idx + self.num_cls_tokens), dim=-1)
        else:
            all_token_idx = None

        if self.random_attn is None:
            outputs, attention_probs = self.dense_attention(q, k, v, head_mask=head_mask,
                                                            relative_position_bias=relative_position_bias,
                                                            q_idx=all_token_idx,
                                                            k_idx=all_token_idx)
            cls_attention_probs = attention_probs[..., :self.num_cls_tokens, :]

        else:
            cls_q, q = q[..., :self.num_cls_tokens, :], q[..., self.num_cls_tokens:, :]

            # dense global attention (num_cls_tokens, seq_len)
            cls_outputs, cls_attention_probs = self.dense_attention(cls_q, k, v, head_mask=head_mask,
                                                                    relative_position_bias=relative_position_bias,
                                                                    q_idx=cls_q_idx,
                                                                    k_idx=all_token_idx)

            # sparse local attention (local_seq_len, seq_len)
            if token_idx is None:
                token_idx = torch.arange(q.shape[-2], device=q.device)
            outputs, attention_probs = self.sparse_attention(q, k, v, head_mask=head_mask,
                                                             relative_position_bias=relative_position_bias,
                                                             q_idx=token_idx + self.num_cls_tokens)

            outputs = torch.cat((cls_outputs, outputs), dim=2)
        
        outputs = self.join_heads(outputs)

        outputs = (outputs, cls_attention_probs) if output_attentions else (outputs,)

        return outputs


class BeitSelfOutput(nn.Module):
    """
    The residual connection is defined in BeitLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, gamma=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitAttention(nn.Module):
    def __init__(self, config, window_size=None):
        super().__init__()
        self.attention = BeitSelfAttention(config, window_size=window_size)
        self.output = BeitSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, head_mask=None, output_attentions=False, relative_position_bias=None, token_idx=None):
        self_outputs = self.attention(hidden_states, head_mask, output_attentions, relative_position_bias, token_idx)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BeitIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BeitOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class BeitLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, window_size=None, drop_path_rate=0.0, 
                 token_keep_rate=1.0):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BeitAttention(config, window_size=window_size)
        self.intermediate = BeitIntermediate(config)
        self.output = BeitOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # sparse params
        self.token_keep_rate = token_keep_rate
        self.token_keep_strategy = config.token_keep_strategy
        self.num_cls_tokens = config.num_cls_tokens

        init_values = config.layer_scale_init_value
        if init_values > 0:
            self.lambda_1 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
            self.lambda_2 = nn.Parameter(init_values * torch.ones((config.hidden_size)), requires_grad=True)
        else:
            self.lambda_1, self.lambda_2 = None, None
    
    def sparsify(self, x, attn):
        x_cls, x_ = x[:, :self.num_cls_tokens], x[:, self.num_cls_tokens:]
        assert 0 < self.token_keep_rate <= 1, "Expected keep rate in range (0, 1]"
        left_tokens = math.ceil(self.token_keep_rate * x_.size(1))

        if self.token_keep_strategy == 'cls_attn':
            if len(attn.shape) == 4:
                attn = attn.mean(1)  # pool over attention heads
            cls_attn = attn[:, 0, self.num_cls_tokens:]
            _, idx = torch.topk(cls_attn, left_tokens, dim=1)  # [B, left_tokens]

        elif self.token_keep_strategy == 'random':
            rand = torch.rand(x_.shape[:2], device=x_.device)
            _, idx = torch.topk(rand, left_tokens, dim=1)  # [B, left_tokens]

        else:
            raise NotImplementedError(f"Sparse strategy {self.token_keep_strategy} is not implemented")

        idx, _ = torch.sort(idx, dim=1)
        index = idx.unsqueeze(-1).expand(-1, -1, x_.size(-1))  # [B, left_tokens, C]
        outputs = torch.cat((x_cls, x_.gather(1, index)), dim=1).contiguous()
        return outputs, idx

    def forward(self, hidden_states, head_mask=None, output_attentions=False, relative_position_bias=None, token_idx=None):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in BEiT, layernorm is applied before self-attention
            head_mask,
            output_attentions=(output_attentions or self.token_keep_rate < 1),
            relative_position_bias=relative_position_bias,
            token_idx=token_idx
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # apply lambda_1 if present
        if self.lambda_1 is not None:
            attention_output = self.lambda_1 * attention_output

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in BEiT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)

        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output)

        if self.lambda_2 is not None:
            layer_output = self.lambda_2 * layer_output

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        # node sparsification
        if self.token_keep_rate < 1:
            layer_output, token_keep_idx = self.sparsify(layer_output, outputs[0])
            if token_idx is not None:
                if token_idx.ndim == 1:
                    token_idx = repeat(token_idx, 'n -> b n', b=len(token_keep_idx))
                token_keep_idx = token_idx.gather(1, token_keep_idx)
            outputs = outputs + (token_keep_idx,)

        outputs = (layer_output,) + outputs

        return outputs


class BeitRelativePositionBias(nn.Module):
    def __init__(self, config, window_size):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = torch.zeros(
            size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype
        )
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] + 1, self.window_size[0] * self.window_size[1] + 1, -1
        )  # Wh*Ww,Wh*Ww,nH

        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class BeitRelativePositionBias3D(nn.Module):
    """
    3D relative position bias
    """
    def __init__(self, config, window_size, num_cls_tokens=1):
        super().__init__()
        self.window_size = window_size
        self.num_cls_tokens = num_cls_tokens
        
        relative_size = [w * 2 - 1 for w in window_size]
        self.num_relative_distance = np.prod(relative_size) + 2 * num_cls_tokens + num_cls_tokens ** 2

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, config.num_attention_heads)
        )

        # get pair-wise relative position index for each token inside the window
        coords_range = [torch.arange(w) for w in window_size]
        coords_flatten = torch.stack(torch.meshgrid(coords_range)).flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        for i, w in enumerate(window_size):
            relative_coords[:, :, i] += w - 1  # shift to start from 0
        
        for i, r in enumerate(relative_size[1:]):
            relative_coords[:, :, :i + 1] *= r

        self.seq_len = np.prod(window_size) + num_cls_tokens
        relative_position_index = torch.zeros((self.seq_len, self.seq_len), dtype=relative_coords.dtype)
        relative_position_index[num_cls_tokens:, num_cls_tokens:] = relative_coords.sum(-1)
        
        start = np.prod(relative_size)
        cls2loc = torch.arange(num_cls_tokens).unsqueeze(1) + start
        relative_position_index[:num_cls_tokens, num_cls_tokens:] = cls2loc
        start += num_cls_tokens

        loc2cls = torch.arange(num_cls_tokens).unsqueeze(0) + start
        relative_position_index[num_cls_tokens:, :num_cls_tokens] = loc2cls
        start += num_cls_tokens

        cls2cls = torch.arange(num_cls_tokens ** 2).view(num_cls_tokens, num_cls_tokens) + start
        relative_position_index[:num_cls_tokens, :num_cls_tokens] = cls2cls

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, from_idx=None, to_idx=None):
        """
        from_idx: indices of query tokens (1-dim)
        to_idx: indices of key/value tokens (1-dim, or 2-dim w/ one row per query)
        """
        attn_idx = self.relative_position_index

        # query indices
        if from_idx is not None:
            attn_idx = attn_idx[from_idx]

        # key indices
        if to_idx is not None:
            assert to_idx.ndim in (1, 2), "to_idx must be 1- or 2-dimensional tensors"
            if to_idx.ndim == 1:
                attn_idx = attn_idx[:, to_idx]
            else:
                attn_idx = attn_idx.gather(1, to_idx)

        rows, cols = attn_idx.shape
        relative_position_bias = self.relative_position_bias_table[attn_idx.flatten()]
        relative_position_bias = rearrange(relative_position_bias, '(i j) h -> h i j', i=rows, j=cols)
        return relative_position_bias.contiguous()


class BeitEncoder(nn.Module):
    def __init__(self, config, window_size=None):
        super().__init__()
        self.config = config
        if config.use_shared_relative_position_bias:
            self.relative_position_bias = BeitRelativePositionBias3D(config, window_size=window_size)
        else:
            self.relative_position_bias = None

        self._register_token_order(window_size)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]

        # node sparsification
        token_keep_rate = [1] * config.num_hidden_layers
        for loc in config.token_drop_loc:
            token_keep_rate[loc] = config.token_keep_rate
        
        self.layer = nn.ModuleList(
            [
                BeitLayer(
                    config,
                    window_size=window_size if config.use_relative_position_bias else None,
                    drop_path_rate=dpr[i], token_keep_rate=token_keep_rate[i]
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.gradient_checkpointing = False
    
    def _register_token_order(self, shape):
        if self.config.token_3d_order == 'none':
            order = None
        elif self.config.token_3d_order == 'zcurve':
            nbits = max(shape).bit_length()
            coords = list(np.ndindex(*shape))
            order = zCurve.par_interlace(coords, len(shape), nbits)
            order = torch.tensor(np.argsort(order))
        elif self.config.token_3d_order == 'hilbert':
            nbits = max(shape).bit_length()
            coords = list(np.ndindex(*shape))
            order = hilbert.encode(np.stack(coords), len(shape), nbits)
            order = torch.tensor(np.argsort(order))
        else:
            raise NotImplementedError(f"Token ordering {self.config.token_3d_order} not supported")

        if order is not None:
            self.register_buffer('token_order', order, persistent=False)
        else:
            self.token_order = None

    def forward(
        self,
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        output_token_idx=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_token_idx = () if output_token_idx else None

        token_idx = self.token_order
        if token_idx is not None:
            cls_states, local_states = hidden_states[:, :self.config.num_cls_tokens], hidden_states[:, self.config.num_cls_tokens:]
            local_states = torch.index_select(local_states, dim=1, index=token_idx)
            hidden_states = torch.cat((cls_states, local_states), 1)

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                relative_position_bias = (
                    self.relative_position_bias() if self.relative_position_bias is not None else None
                )
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions, relative_position_bias, token_idx)

            hidden_states = layer_outputs[0]

            if layer_module.token_keep_rate < 1:
                token_idx = layer_outputs[-1]

                if output_token_idx:
                    all_token_idx = all_token_idx + (token_idx,)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BeitModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            token_idx=all_token_idx
        )


class BeitPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BeitConfig
    base_model_prefix = "beit"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BeitEncoder):
            module.gradient_checkpointing = value


BEIT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config (:class:`~transformers.BeitConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

BEIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using :class:`~transformers.BeitFeatureExtractor`. See
            :meth:`transformers.BeitFeatureExtractor.__call__` for details.

        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Beit Model transformer outputting raw hidden-states without any specific head on top.",
    BEIT_START_DOCSTRING,
)
class BeitModel(BeitPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True, num_frames=None):
        super().__init__(config)
        self.config = config

        self.embeddings = BeitEmbeddings(config)
        self.window_size = self.embeddings.patch_embeddings.patch_shape
        if num_frames is not None:
            self.window_size = (num_frames,) + self.window_size
        self.encoder = BeitEncoder(config, window_size=self.window_size)

        self.layernorm = (
            nn.Identity() if config.use_mean_pooling else nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        )
        self.pooler = BeitPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BeitModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        output_token_idx=None,
        return_dict=None,
    ):
        r"""
        Returns:

        Examples::

            >>> from transformers import BeitFeatureExtractor, BeitModel
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')
            >>> model = BeitModel.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> last_hidden_states = outputs.last_hidden_state
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(pixel_values, bool_masked_pos)

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_token_idx=output_token_idx,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BeitModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            token_idx=encoder_outputs.token_idx,
        )


class BeitPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.use_mean_pooling else None
        )

    def forward(self, hidden_states):
        if self.layernorm is not None:
            # Mean pool the final hidden states of the patch tokens
            patch_tokens = hidden_states[:, 1:, :]
            pooled_output = self.layernorm(patch_tokens.mean(1))
        else:
            # Pool by simply taking the final hidden state of the [CLS] token
            pooled_output = hidden_states[:, 0]

        return pooled_output


@add_start_docstrings(
    "Beit Model transformer with a 'language' modeling head on top (to predict visual tokens).",
    BEIT_START_DOCSTRING,
)
class BeitForMaskedImageModeling(BeitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # Classifier head
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        bool_masked_pos=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        bool_masked_pos (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import BeitFeatureExtractor, BeitForMaskedImageModeling
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k')
            >>> model = BeitForMaskedImageModeling.from_pretrained('microsoft/beit-base-patch16-224-pt22k')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.layernorm(sequence_output)
        prediction_scores = self.lm_head(sequence_output[:, 1:])

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores[bool_masked_pos], labels)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    Beit Model transformer with an image classification head on top (a linear layer on top of the average of the final
    hidden states of the patch tokens) e.g. for ImageNet.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForImageClassification(BeitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=True)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the image classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import BeitFeatureExtractor, BeitForImageClassification
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224')
            >>> model = BeitForImageClassification.from_pretrained('microsoft/beit-base-patch16-224')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> logits = outputs.logits
            >>> # model predicts one of the 1000 ImageNet classes
            >>> predicted_class_idx = logits.argmax(-1).item()
            >>> print("Predicted class:", model.config.id2label[predicted_class_idx])
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs.pooler_output if return_dict else outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BeitConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation(output)

        return output


class BeitPyramidPoolingModule(nn.ModuleList):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        align_corners (bool): align_corners argument of F.interpolate.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    BeitConvModule(self.in_channels, self.channels, kernel_size=1),
                )
            )

    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class BeitUperHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config):
        super().__init__()

        self.pool_scales = config.pool_scales  # e.g. (1, 2, 3, 6)
        self.in_channels = [config.hidden_size] * 4  # e.g. [768, 768, 768, 768]
        self.channels = config.hidden_size
        self.align_corners = False
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

        # PSP Module
        self.psp_modules = BeitPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = BeitConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = BeitConvModule(in_channels, self.channels, kernel_size=1)
            fpn_conv = BeitConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = BeitConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, encoder_hidden_states):
        # build laterals
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        laterals.append(self.psp_forward(encoder_hidden_states))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        output = self.fpn_bottleneck(fpn_outs)
        output = self.classifier(output)

        return output


class BeitFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is implemented of `FCNNet
    <https://arxiv.org/abs/1411.4038>`_.

    Args:
        config (BeitConfig): Configuration.
        in_channels
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        dilation (int): The dilation rate for convs in the head. Default: 1.


    Based on OpenMMLab's implementation, found in https://github.com/open-mmlab/mmsegmentation.
    """

    def __init__(self, config, in_index=2, kernel_size=3, dilation=1):
        super().__init__()
        self.in_channels = config.hidden_size
        self.channels = config.auxiliary_channels
        self.num_convs = config.auxiliary_num_convs
        self.concat_input = config.auxiliary_concat_input
        self.in_index = in_index

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            BeitConvModule(
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):
            convs.append(
                BeitConvModule(
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = BeitConvModule(
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)

    def forward(self, encoder_hidden_states):
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]
        output = self.convs(hidden_states)
        if self.concat_input:
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))
        output = self.classifier(output)
        return output


@add_start_docstrings(
    """
    Beit Model transformer with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    BEIT_START_DOCSTRING,
)
class BeitForSemanticSegmentation(BeitPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.beit = BeitModel(config, add_pooling_layer=False)

        # FPNs
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU(),
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_size, config.hidden_size, kernel_size=2, stride=2),
        )
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Semantic segmentation head(s)
        self.decode_head = BeitUperHead(config)
        self.auxiliary_head = BeitFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        self.post_init()

    def compute_loss(self, logits, auxiliary_logits, labels):
        # upsample logits to the images' original size
        upsampled_logits = nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        if auxiliary_logits is not None:
            upsampled_auxiliary_logits = nn.functional.interpolate(
                auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
        # compute weighted loss
        loss_fct = CrossEntropyLoss(ignore_index=self.config.semantic_loss_ignore_index)
        main_loss = loss_fct(upsampled_logits, labels)
        auxiliary_loss = loss_fct(upsampled_auxiliary_logits, labels)
        loss = main_loss + self.config.auxiliary_loss_weight * auxiliary_loss

        return loss

    @add_start_docstrings_to_model_forward(BEIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values=None,
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, height, width)`, `optional`):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels > 1`, a classification loss is computed
            (Cross-Entropy).

        Returns:

        Examples::

            >>> from transformers import BeitFeatureExtractor, BeitForSemanticSegmentation
            >>> from PIL import Image
            >>> import requests

            >>> url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
            >>> image = Image.open(requests.get(url, stream=True).raw)

            >>> feature_extractor = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')
            >>> model = BeitForSemanticSegmentation.from_pretrained('microsoft/beit-base-finetuned-ade-640-640')

            >>> inputs = feature_extractor(images=image, return_tensors="pt")
            >>> outputs = model(**inputs)
            >>> # logits are of shape (batch_size, num_labels, height/4, width/4)
            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.beit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=return_dict,
        )

        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[2]

        # only keep certain features, and reshape
        # note that we do +1 as the encoder_hidden_states also includes the initial embeddings
        features = [feature for idx, feature in enumerate(encoder_hidden_states) if idx + 1 in self.config.out_indices]
        batch_size = pixel_values.shape[0]
        patch_resolution = self.config.image_size // self.config.patch_size
        features = [
            x[:, 1:, :].permute(0, 2, 1).reshape(batch_size, -1, patch_resolution, patch_resolution) for x in features
        ]

        # apply FPNs
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        logits = self.decode_head(features)
        auxiliary_logits = None
        if self.auxiliary_head is not None:
            auxiliary_logits = self.auxiliary_head(features)

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                raise ValueError("The number of labels should be greater than one")
            else:
                loss = self.compute_loss(logits, auxiliary_logits, labels)

        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[2:]
            else:
                output = (logits,) + outputs[3:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
        )