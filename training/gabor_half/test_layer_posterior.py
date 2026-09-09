"""Exercise shared teacher forwards, objective composition and gradient ownership."""
import torch
from torch import nn

from layer_anchor import LayerAnchor


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Linear(4, 4)

    def forward(self, inputs):
        return inputs + .1 * self.conv(inputs)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block() for _ in range(24)])
        self.calls = 0

    def forward(self, audio_signal, length):
        self.calls += 1
        value = audio_signal.transpose(1, 2)
        for block in self.layers:
            value = block(value)
        return value.transpose(1, 2), length


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)
        self.calls = 0

    def forward(self, targets, target_length):
        self.calls += 1
        bos = torch.zeros(targets.shape[0], 1, dtype=targets.dtype, device=targets.device)
        return self.embedding(torch.cat((bos, targets), 1)).transpose(1, 2), target_length, None


class Joint(nn.Module):
    masking_prob = 0
    log_softmax = None
    temperature = 1.
    num_classes_with_blank = 8198
    num_extra_outputs = 5

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(4, 4)
        self.pred = nn.Linear(4, 4)
        self.joint_net = nn.Sequential(nn.Tanh(), nn.Linear(4, 8198))

    def is_adapter_available(self):
        return False

    def joint_after_projection(self, encoder, decoder):
        return self.joint_net(encoder.unsqueeze(2) + decoder.unsqueeze(1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.joint = Joint()
        self.logged = {}
        self.training_calls = 0

    @property
    def device(self):
        return next(self.parameters()).device

    def log(self, name, value, **kwargs):
        self.logged[name] = value

    def training_step(self, batch, batch_idx):
        self.training_calls += 1
        audio, lengths, targets, target_lengths = batch
        encoder, _ = self.encoder(audio_signal=audio, length=lengths)
        decoder, _, _ = self.decoder(targets=targets, target_length=target_lengths)
        logits = self.joint.joint_after_projection(self.joint.enc(encoder.transpose(1, 2)),
                                                  self.joint.pred(decoder.transpose(1, 2)))
        loss = logits.square().mean()
        self.last_asr = loss.detach()
        return {'loss': loss}


def batch():
    return torch.randn(2, 4, 5), torch.tensor([5, 3]), torch.tensor([[2, 3, 4], [5, 6, 0]]), torch.tensor([3, 2])


def test_combined_teacher_runs_once_and_never_enters_student_state():
    torch.manual_seed(19)
    model = Model()
    parameters = {id(p) for p in model.parameters()}
    state_keys = set(model.state_dict())
    anchor = LayerAnchor(model, 50., 25., .1, posterior_scale=1., posterior_points=2)
    anchor.on_fit_start(None, model)
    with torch.no_grad():
        model.encoder.layers[0].conv.weight.add_(.03)
        model.decoder.embedding.weight.add_(.02)
        model.joint.joint_net[-1].weight.add_(torch.randn_like(model.joint.joint_net[-1].weight) * .01)
    result = model.training_step(batch(), 0)
    assert model.training_calls == model.encoder.calls == model.decoder.calls == 1
    assert anchor.teacher.calls == anchor.head_teacher['decoder'].calls == 1
    assert anchor.parity_checked and not anchor.posterior_cache
    expected = .1 * model.last_asr + 50 * model.logged['layer_anchor_block'] + 25 * model.logged['layer_anchor_conv']
    expected += model.logged['teacher_token_kl'] + model.logged['teacher_duration_kl']
    torch.testing.assert_close(result['loss'].detach(), expected)
    assert model.logged['teacher_token_kl'] > 0 and model.logged['teacher_duration_kl'] > 0
    result['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(not p.requires_grad and p.grad is None for teacher in (anchor.teacher, anchor.head_teacher)
               for p in teacher.parameters())
    assert parameters == {id(p) for p in model.parameters()}
    assert state_keys == set(model.state_dict())


def test_disabled_posterior_keeps_layer_only_objective_and_rng():
    torch.manual_seed(20)
    model = Model()
    anchor = LayerAnchor(model, 50., 25., .1)
    anchor.on_fit_start(None, model)
    inputs = batch()
    rng = torch.random.get_rng_state().clone()
    result = model.training_step(inputs, 0)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert anchor.head_teacher is None and not anchor.posterior_cache
    assert not any(name.startswith('teacher_') for name in model.logged)
    torch.testing.assert_close(result['loss'].detach(), .1 * model.last_asr)
    assert model.training_calls == anchor.teacher.calls == 1
