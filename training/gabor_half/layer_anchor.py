"""Training-only recovery against every original Conformer block and branch."""
import copy
import types

import lightning.pytorch as pl
import torch
from torch import nn

from anchor_loss import normalized_mse
from posterior_loss import sample_indices, gather_steps, tdt_kl


class LayerAnchor(pl.Callback):
    def __init__(self, model, block_scale, conv_scale, asr_scale, posterior_scale=0., posterior_points=16):
        if min(block_scale, conv_scale, asr_scale) <= 0:
            raise ValueError('Every recovery objective must remain active')
        if posterior_scale < 0 or posterior_points < 1:
            raise ValueError('Invalid optional posterior objective')
        # Copy before attaching hooks or loading a recovery candidate.
        self.teacher = copy.deepcopy(model.encoder).eval().requires_grad_(False)
        self.student_outputs = {}
        self.teacher_outputs = {}
        self.losses = None
        self.handles = []
        self.posterior_cache = {}
        self.head_teacher = None
        self.parity_checked = False
        if posterior_scale:
            joint = model.joint
            if joint.is_adapter_available() or joint.masking_prob > 0:
                raise ValueError('Expected an unadapted TDT joint')
            if joint.log_softmax not in (None, False) or joint.temperature != 1.:
                raise ValueError('Expected raw TDT logits with temperature 1')
            token_count = joint.num_classes_with_blank - joint.num_extra_outputs
            if joint.num_extra_outputs != 5 or token_count != 8193:
                raise ValueError('Unexpected source TDT head')
            self.head_teacher = nn.ModuleDict({name: copy.deepcopy(module) for name, module in {
                'decoder': model.decoder, 'enc': joint.enc, 'pred': joint.pred,
                'joint_net': joint.joint_net}.items()}).eval().requires_grad_(False)

            def decoder_hook(module, args, kwargs, output):
                if not model.training or not torch.is_grad_enabled():
                    return
                with torch.no_grad():
                    target = self.head_teacher['decoder'](*args, **kwargs)
                if output[0].shape != target[0].shape or not torch.equal(output[1], target[1]):
                    raise RuntimeError('Teacher and student decoder layouts differ')
                indices = sample_indices(output[1] + 1, posterior_points)
                self.posterior_cache['decoder'] = (gather_steps(output[0], indices), gather_steps(target[0], indices))

            self.handles.append(model.decoder.register_forward_hook(decoder_hook, with_kwargs=True))
        self.count = len(model.encoder.layers)
        if self.count != 24:
            raise ValueError('This experiment expects 24 Conformer blocks')

        def capture(destination, name, student):
            def hook(module, args, output):
                if not model.training or (student and not torch.is_grad_enabled()):
                    return
                if not isinstance(output, torch.Tensor) or output.ndim != 3:
                    raise ValueError('Expected offline B,T,C Conformer output')
                destination[name] = output
            return hook

        for index, (student, teacher) in enumerate(zip(model.encoder.layers, self.teacher.layers)):
            for kind in ('block', 'conv'):
                left = student if kind == 'block' else student.conv
                right = teacher if kind == 'block' else teacher.conv
                name = (kind, index)
                self.handles.append(left.register_forward_hook(capture(self.student_outputs, name, True)))
                self.handles.append(right.register_forward_hook(capture(self.teacher_outputs, name, False)))

        def encoder_hook(module, args, kwargs, output):
            if not model.training or not torch.is_grad_enabled():
                return
            with torch.no_grad():
                target, target_lengths = self.teacher(*args, **kwargs)
            actual, lengths = output
            if actual.shape != target.shape or not torch.equal(lengths, target_lengths):
                raise RuntimeError('Teacher and student encoder layouts differ')
            if posterior_scale:
                # Reuse this teacher forward for both hidden states and logits.
                indices = sample_indices(lengths, posterior_points)
                self.posterior_cache['encoder'] = (gather_steps(actual, indices), gather_steps(target, indices))
            expected = {(kind, i) for kind in ('block', 'conv') for i in range(self.count)}
            if set(self.student_outputs) != expected or set(self.teacher_outputs) != expected:
                raise RuntimeError('Incomplete layer recovery observations')
            self.losses = {}
            for kind in ('block', 'conv'):
                losses = []
                for index in range(self.count):
                    student = self.student_outputs[kind, index]
                    reference = self.teacher_outputs[kind, index]
                    if student.shape != reference.shape or student.shape[1] != actual.shape[2]:
                        raise RuntimeError('Intermediate sequence layout differs')
                    losses.append(normalized_mse(student.transpose(1, 2), reference.transpose(1, 2), lengths))
                self.losses[kind] = torch.stack(losses).mean()
            # Autograd retains the necessary values; do not keep a second cache.
            self.student_outputs.clear()
            self.teacher_outputs.clear()

        self.handles.append(model.encoder.register_forward_hook(encoder_hook, with_kwargs=True))
        original = model.training_step

        def training_step(module, batch, batch_idx):
            self.student_outputs.clear()
            self.teacher_outputs.clear()
            self.losses = None
            self.posterior_cache.clear()
            result = original(batch, batch_idx)
            if self.losses is None:
                raise RuntimeError('Layer reference did not run')
            asr = result['loss'] if isinstance(result, dict) else result
            total = asr_scale * asr + block_scale * self.losses['block'] + conv_scale * self.losses['conv']
            if posterior_scale:
                if set(self.posterior_cache) != {'encoder', 'decoder'}:
                    raise RuntimeError('Missing shared posterior teacher inputs')
                enc, teacher_enc = self.posterior_cache['encoder']
                pred, teacher_pred = self.posterior_cache['decoder']
                logits = model.joint.joint_net(model.joint.enc(enc).unsqueeze(2) + model.joint.pred(pred).unsqueeze(1))
                if not self.parity_checked:
                    with torch.no_grad():
                        direct = model.joint.joint_after_projection(model.joint.enc(enc), model.joint.pred(pred))
                    if not torch.equal(logits.detach(), direct):
                        raise RuntimeError('Native joint parity failed')
                    self.parity_checked = True
                    print('SHARED_TEACHER_JOINT_PARITY pass', flush=True)
                with torch.no_grad():
                    target = self.head_teacher['joint_net'](
                        self.head_teacher['enc'](teacher_enc).unsqueeze(2)
                        + self.head_teacher['pred'](teacher_pred).unsqueeze(1))
                token_loss, duration_loss = tdt_kl(logits, target, token_count)
                total = total + posterior_scale * (token_loss + duration_loss)
                module.log('teacher_token_kl', token_loss.detach(), on_step=True)
                module.log('teacher_duration_kl', duration_loss.detach(), on_step=True)
                self.posterior_cache.clear()
            for kind, loss in self.losses.items():
                module.log('layer_anchor_' + kind, loss.detach(), on_step=True)
            module.log('recovery_total_loss', total.detach(), on_step=True)
            if isinstance(result, dict):
                result['loss'] = total
                return result
            return total

        model.training_step = types.MethodType(training_step, model)

    def on_fit_start(self, trainer, model):
        self.teacher.to(model.device).eval()
        if self.head_teacher is not None:
            self.head_teacher.to(model.device).eval()
