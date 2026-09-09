import torch
from anchor_loss import normalized_mse


def test_masked_reconstruction_and_amplitude():
    target = torch.ones(2, 4, 8)
    lengths = torch.tensor([3, 7])
    actual = target.clone()
    actual[0, :, 3:] = 900
    actual[1, :, 7:] = -900
    assert normalized_mse(actual, target, lengths).item() == 0
    assert normalized_mse(2 * target, target, lengths).item() == 1
    actual.requires_grad_(True)
    normalized_mse(actual, target, lengths).backward()
    assert torch.count_nonzero(actual.grad) == 0


def test_zero_energy_stays_finite():
    target = torch.zeros(1, 4, 8)
    actual = torch.ones_like(target, requires_grad=True)
    loss = normalized_mse(actual, target, torch.tensor([3]))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(actual.grad).all()
