import torch

def reorder_frames(video, temporal_order='none'):
    bs, n_frames = video.shape[:2]
    if temporal_order == 'none':
        pass
    elif temporal_order == 'reverse':
        video = video.flip(1)
    elif temporal_order == 'shuffle':
        # shuffle each example independently
        idx = torch.argsort(torch.rand(bs, n_frames), dim=1)
        video = video[torch.arange(bs).unsqueeze(1), idx]
    elif temporal_order == 'freeze':
        # take random frame and repeat
        idx = torch.randint(n_frames, [bs])
        video = video[torch.arange(bs), idx].unsqueeze(1).expand_as(video).contiguous()
    else:
        raise ValueError(f"Invalid reordering method: {temporal_order}")
    return video