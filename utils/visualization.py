import os
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from einops import rearrange


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))


def save_token_map(img_id, img, token_idx, thw_shape, save_path, bg_val=0.5, layer_id=(0, 1, 2,), texts=None):
    os.makedirs(save_path, exist_ok=True)

    # normalize image
    bsz = len(img_id)
    img_norm = img.detach().clone()
    img_norm = rearrange(img_norm, '(b k) n c h w -> b (k n) c h w', b=bsz)
    for t in img_norm:
        norm_ip(t, float(t.min()), float(t.max()))
    
    with torch.no_grad():
        for k, token_idx_ in enumerate(token_idx):
            if k not in layer_id:
                continue

            num_clips = token_idx_.shape[0]
            mask = torch.full((num_clips, *thw_shape), bg_val, device=img.device)
            mask_flt = rearrange(mask, 'b t h w -> b (t h w)')
            mask_flt[torch.arange(num_clips).unsqueeze(1), token_idx_] = 1
            mask_rsz = F.interpolate(mask, img.shape[-2:], mode='nearest')
            mask_rsz = rearrange(mask_rsz, '(b k) n h w -> b (k n) 1 h w', b=bsz)
            img_norm = mask_rsz * img_norm

    # save images
    for i, id in enumerate(img_id):
        save_fp = os.path.join(save_path, f'vid{id}.jpg')
        save_image(img_norm[i], fp=save_fp, nrow=thw_shape[0], normalize=False)

        if texts is not None:
            save_fp_txt = os.path.join(save_path, f'vid{id}.txt')
            with open(save_fp_txt, 'w') as txt:
                txt.write(texts[i] + '\n')