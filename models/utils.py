import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy import interpolate
import numpy as np
import logging
from einops import rearrange, repeat

logger = logging.getLogger(__name__)


def _init_transformer_weights(module, initializer_range=0.02):
    """Initialize the weights. Copied from transformers ViT/Bert model init"""
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=initializer_range)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def interpolate_pos_embed(pos_embed_old, pos_embed_new, num_patches_new):
    """
    Args:
        pos_embed_old: (1, L_old, d), pre-trained
        pos_embed_new: (1, L_new, d), newly initialized, to be replaced by interpolated weights
        num_patches_new:
    """
    # interpolate position embedding
    embedding_size = pos_embed_old.shape[-1]
    num_extra_tokens = pos_embed_new.shape[-2] - num_patches_new
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_old.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches_new ** 0.5)

    if orig_size != new_size:
        # class_token and dist_token are kept unchanged
        # the extra tokens seems always at the beginning of the position embedding
        extra_tokens = pos_embed_old[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_old[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(
            -1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        interpolated_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        logger.info(f"reshape position embedding from {orig_size}**2 to {new_size}**2")
        return interpolated_pos_embed
    else:
        return pos_embed_old


def interpolate_pos_relative_bias_beit(state_dict_old, state_dict_new, patch_shape_new):
    """
    Args:
        state_dict_old: loaded state dict
        state_dict_new: state dict for model with new image size
        patch_shape_new: new model patch_shape
    ref: https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py
    """
    all_keys = list(state_dict_old.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict_old.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = state_dict_old[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = state_dict_new[key].size()
            dst_patch_shape = patch_shape_new
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                # logger.info("Position interpolate for %s from %dx%d to %dx%d" % (
                #     key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # logger.info("Original positions = %s" % str(x))
                # logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict_old[key] = new_rel_pos_bias
    return state_dict_old


def interpolate_pos_relative_bias_beit_3d(state_dict_old, state_dict_new, patch_shape_new, src_t_size=1):
    """
    Args:
        state_dict_old: loaded state dict
        state_dict_new: state dict for model with new image size
        patch_shape_new: new model patch_shape
    ref: https://github.com/microsoft/unilm/blob/master/beit/run_class_finetuning.py
    """
    all_keys = list(state_dict_old.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            state_dict_old.pop(key)

        if "relative_position_bias_table" in key:
            src_num_pos, num_attn_heads = state_dict_old[key].size()
            dst_num_pos, _ = state_dict_new[key].size()
            if src_num_pos == dst_num_pos:
                continue

            num_extra_tokens = dst_num_pos - np.prod([w * 2 - 1 for w in patch_shape_new])
                
            src_s_size = int((src_num_pos - num_extra_tokens) / src_t_size)
            src_size = int(src_s_size ** 0.5)
            dst_size = patch_shape_new[-1] * 2 - 1

            if src_size != dst_size:
                # Spatial interpolation
                logger.info(f"Position interpolate for {key} from {src_size}x{src_size} to {dst_size}x{dst_size}")
                
                rel_pos_bias = state_dict_old[key]
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                # logger.info("Original positions = %s" % str(x))
                # logger.info("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict_old[key] = new_rel_pos_bias

            dst_t_size = patch_shape_new[0] * 2 - 1
            if src_t_size != dst_t_size:
                # Temporal interpolation
                logger.info(f"Inflating {key} from {src_t_size}x{src_size}x{src_size} to {dst_t_size}x{dst_size}x{dst_size}")
                
                rel_pos_bias = state_dict_old[key]
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                
                if src_t_size == 1:
                    rel_pos_bias = repeat(rel_pos_bias, 's d -> (t s) d', t=dst_t_size)
                else:
                    rel_pos_bias = rearrange(rel_pos_bias, '(t s) d -> s d t', t=src_t_size)
                    rel_pos_bias = F.interpolate(rel_pos_bias, dst_t_size, mode='nearest')
                    rel_pos_bias = rearrange(rel_pos_bias, 's d t -> (t s) d')
                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                state_dict_old[key] = new_rel_pos_bias

    return state_dict_old


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*repeat_idx)
    order_index = torch.LongTensor(np.concatenate(
        [init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)

