import pytest
import torch
from torch import nn

from layer_anchor import LayerAnchor


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, x):
        return self.norm(x + .1 * self.conv(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(24))

    def forward(self, x, lengths):
        x = x.transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return x.transpose(1, 2), lengths


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = nn.Linear(4, 1)
        self.metrics = {}

    def log(self, name, value, **kwargs):
        self.metrics[name] = value

    def training_step(self, batch, batch_idx):
        encoded, _ = self.encoder(*batch)
        return {'loss': self.decoder(encoded.transpose(1, 2)).square().mean()}


def test_layer_recovery_keeps_teacher_external_and_reaches_all_layers():
    torch.manual_seed(19)
    model = Student().train()
    keys = set(model.state_dict())
    anchor = LayerAnchor(model, 5., 1., .1)
    assert set(model.state_dict()) == keys
    assert all(not p.requires_grad for p in anchor.teacher.parameters())
    batch = (torch.randn(2, 4, 8), torch.tensor([3, 8]))
    model.training_step(batch, 0)
    # Training and no-grad linear kernels can differ at FP32 rounding precision.
    assert model.metrics['layer_anchor_block'] < 1e-12
    assert model.metrics['layer_anchor_conv'] < 1e-12
    with torch.no_grad():
        model.encoder.layers[0].conv.weight.add_(.2)
    result = model.training_step(batch, 1)
    assert model.metrics['layer_anchor_block'] > 0
    assert model.metrics['layer_anchor_conv'] > 0
    result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is None for p in anchor.teacher.parameters())
    assert anchor.student_outputs == anchor.teacher_outputs == {}


def test_layer_recovery_requires_positive_asr_loss():
    with pytest.raises(ValueError):
        LayerAnchor(Student(), 5., 1., 0.)
