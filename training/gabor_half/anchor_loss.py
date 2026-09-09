"""Length-masked encoder recovery losses, averaged equally over utterances."""
import torch


def normalized_mse(actual, target, lengths):
    actual, target = actual.float(), target.float()
    mask = torch.arange(actual.shape[2], device=actual.device)[None, :] < lengths[:, None]
    error = ((actual - target).square().mean(dim=1) * mask).sum(dim=1)
    energy = (target.square().mean(dim=1) * mask).sum(dim=1)
    return (error / energy.clamp_min(1e-8)).mean()
