"""One finite pass over explicitly selected evaluation data with frozen Gabor rows.

Run only after preparation and alignment decisions have been sealed. Checkpoints
contain the finite batch cursor, optimizer, RNG, and exact source identities.
For short runs, --skip-optimizer-checkpoint keeps only the final portable model.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import types

import numpy as np
import soundfile as sf
import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf, open_dict

from batches import make_batches, learning_rate


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


class AudioDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        samples, rate = sf.read(row['audio_filepath'], dtype='float32')
        if rate != 16000 or samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError(f'Invalid training audio: {row["uid"]}')
        if hashlib.sha256(samples.astype('<f4').tobytes()).hexdigest() != row['training_pcm_sha256']:
            raise ValueError(f'Training PCM changed: {row["uid"]}')
        return i, torch.from_numpy(samples), torch.tensor(row['tokens'], dtype=torch.long)


def collate(batch):
    ids, signals, tokens = zip(*batch)
    return (list(ids), torch.nn.utils.rnn.pad_sequence(signals, batch_first=True),
            torch.tensor([len(x) for x in signals], dtype=torch.long),
            torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True),
            torch.tensor([len(x) for x in tokens], dtype=torch.long))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--fits', type=Path, required=True)
    p.add_argument('--gabor-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--stop-after-step', type=int, default=0,
                   help='Save an optimizer-boundary checkpoint and exit for a resume check.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--skip-optimizer-checkpoint', action='store_true')
    a = p.parse_args()
    if a.skip_optimizer_checkpoint and (a.resume or a.stop_after_step):
        p.error('Optimizer checkpoint skipping is incompatible with resume/pause')
    sys.path.insert(0, str(a.gabor_code))
    from frozen import install, verify, materialized_state_dict, tensor_sha
    from nemo.collections.asr.models import ASRModel
    from nemo.core.classes.mixins import AccessMixin

    a.output.mkdir(parents=True, exist_ok=True)
    a.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if (a.output / 'complete.json').exists():
        raise FileExistsError('This run is already complete')
    if (a.output / 'run.json').exists() and not a.resume:
        raise FileExistsError('Existing run requires explicit --resume')
    plan = json.loads(a.plan.read_text())
    preparation = json.loads(a.prepared.read_text())
    assert preparation['complete']
    assert preparation['plan_sha256'] == plan.get('source_plan_sha256', sha(a.plan))
    assert sha(a.parent) == plan['parent_sha256']
    assert sha(a.fits) == plan['fit_sha256']
    if not a.smoke_steps:
        assert plan['status'] == 'sealed_for_training'
    seed = plan['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows, manifests = [], []
    for entry in sorted(preparation['manifests'], key=lambda e: e['split']):
        path = Path(entry['path'])
        assert sha(path) == entry['sha256']
        if entry['alignment_status'] == 'repair_required':
            raise ValueError('Unresolved source alignment: ' + entry['split'])
        part = [json.loads(line) for line in path.open()]
        assert len(part) == entry['rows']
        rows.extend(part)
        manifests.append(entry)
    assert len({r['uid'] for r in rows}) == len(rows)
    assert len(rows) == preparation['rows']
    if not a.smoke_steps:
        assert len(rows) == plan['training_rows']

    model = ASRModel.restore_from(str(a.parent), map_location='cpu')
    parent_hashes = {n: tensor_sha(v) for n, v in model.state_dict().items()}
    receipts = install(model, a.fits)
    dense = materialized_state_dict(model, model.state_dict())
    assert {n: tensor_sha(v) for n, v in dense.items()} == parent_hashes, 'Installation altered the parent'
    del dense
    bn_state = {n: v.clone() for n, v in model.named_buffers()
                if n.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    for row in rows:
        row['tokens'] = model.tokenizer.text_to_ids(row['text'])
        if not row['tokens']:
            raise ValueError('Empty BPE target: ' + row['uid'])
        if not a.smoke_steps:
            assert hashlib.sha256(row['text'].encode()).hexdigest() == row['training_reference_sha256']
            assert model.tokenizer.tokenizer.unk_id() not in row['tokens'], 'Unknown training token: ' + row['uid']
    batches = make_batches(rows, seed,
                           max_seconds=plan['batching']['max_padded_seconds'],
                           max_utterances=plan['batching']['max_utterances'])
    accumulation = plan['batching']['accumulate_batches']
    full_steps = math.ceil(len(batches) / accumulation)
    if a.smoke_steps:
        # Profile the largest padded batches, not just easy short examples.
        batches.sort(key=lambda b: max(rows[i]['duration'] for i in b) * len(b), reverse=True)
        batches = batches[:a.smoke_steps * accumulation]
    steps = math.ceil(len(batches) / accumulation)
    order_hash = hashlib.sha256(json.dumps([[rows[i]['uid'] for i in b] for b in batches],
                                           separators=(',', ':')).encode()).hexdigest()
    source_hashes = {f: sha(Path(__file__).parent / f) for f in ['train.py', 'batches.py']}
    source_hashes.update({'gabor/' + f: sha(a.gabor_code / f) for f in ['frozen.py', 'fit.py', 'identity.py']})

    # Keep normalization buffers; affine parameters remain in the optimizer.
    model.spec_augmentation = None
    model.preprocessor.featurizer.dither = 0.0
    model.train()
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.eval()
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
        if isinstance(module, torch.nn.LSTM):
            module.dropout = 0.0
    model.cuda()
    model.joint.set_fuse_loss_wer(True)
    model.joint.set_fused_batch_size(plan['batching'].get('fused_joint_batch_size', 1))
    model.joint.set_loss(model.loss)
    model.joint.set_wer(model.wer)
    assert model.loss.reduction == 'mean_volume'
    assert not model._optim_normalize_joint_txu
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['proposed_optimizer']['peak_lr'],
                                 betas=tuple(plan['proposed_optimizer']['betas']),
                                 weight_decay=plan['proposed_optimizer']['weight_decay'], fused=True)
    assert {id(v) for v in model.parameters()} == {id(v) for g in optimizer.param_groups for v in g['params']}
    with open_dict(model.cfg):
        model.cfg.optim = OmegaConf.create(dict(name='adamw', lr=plan['proposed_optimizer']['peak_lr'],
                                               betas=plan['proposed_optimizer']['betas'],
                                               weight_decay=plan['proposed_optimizer']['weight_decay']))
        model.cfg.train_ds = None
        # NeMo transcribe reads validation_ds.sample_rate from the saved config.
        assert model.cfg.validation_ds is not None
        model.cfg.validation_ds.manifest_filepath = None
    run = dict(plan_sha256=sha(a.plan), prepared_sha256=sha(a.prepared), parent_sha256=sha(a.parent),
               fit_sha256=sha(a.fits), batch_order_sha256=order_hash, source_sha256=source_hashes,
               steps=steps, full_epoch_steps=full_steps, batches=len(batches),
               rows=sum(map(len, batches)), smoke=bool(a.smoke_steps),
               parameter_tensors=len(list(model.parameters())),
               trainable_scalar_parameters=sum(v.numel() for v in model.parameters()),
               frozen_gabor_rows=12288, loss='NeMo TDT, mean_volume over each accumulated target-token volume',
               augmentation='none; dropout disabled; dither zero',
               batchnorm='running statistics preserved; affine parameters trained',
               precision='BF16 autocast; FP32 parameters and AdamW states',
               cuda=torch.version.cuda, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               cudnn=torch.backends.cudnn.version(), torch_threads=torch.get_num_threads(),
               tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
               versions={name: importlib.metadata.version(name) for name in
                         ['nemo_toolkit', 'numpy', 'soundfile', 'soxr', 'transformers', 'huggingface_hub']},
               all_remaining_parameters_in_optimizer=True,
               evaluation_data_used_for_training=True,
               optimizer_checkpoint_saved=not a.skip_optimizer_checkpoint)
    next_batch, step, seen = 0, 0, Counter()
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        assert state['run'] == run
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        next_batch, step = state['next_batch'], state['step']
        seen.update(state['seen_by_split'])
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        del state
    atomic_json(a.output / 'run.json', run)
    atomic_json(a.output / 'frozen-kernels.json', receipts)
    atomic_json(a.output / 'parent-tensor-hashes.json', parent_hashes)
    initial_trainable_hashes = {n: tensor_sha(v) for n, v in model.named_parameters()} if not a.resume else None
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(AudioDataset(rows), batch_sampler=batches[next_batch:], collate_fn=collate,
                        num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0,
                        generator=generator)
    verify(model, receipts)

    def checkpoint(cursor):
        verify(model, receipts)
        if a.skip_optimizer_checkpoint:
            return
        state = dict(run=run, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     next_batch=cursor, step=step, seen_by_split=dict(seen),
                     rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all())
        target = a.checkpoint_dir / 'resume.pt'
        tmp = target.with_suffix('.tmp')
        torch.save(state, tmp)
        os.replace(tmp, target)
        atomic_json(a.output / 'checkpoint.json', dict(step=step, next_batch=cursor,
                                                     path=str(target), bytes=target.stat().st_size))

    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    tokens_in_group, total_loss, group_rows = 0, 0.0, []
    for batch_index, batch in enumerate(loader, next_batch):
        ids, signal, length, target, target_length = batch
        group_rows.extend(ids)
        volume = int(target_length.sum())
        tokens_in_group += volume
        signal, length, target, target_length = [v.cuda(non_blocking=True) for v in (signal, length, target, target_length)]
        if AccessMixin.is_access_enabled(model.model_guid):
            AccessMixin.reset_registry(model)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            encoded, encoded_len = model(input_signal=signal, input_signal_length=length)
            decoded, _, _ = model.decoder(targets=target, target_length=target_length)
            loss, _, _, _ = model.joint(encoder_outputs=encoded, decoder_outputs=decoded,
                                      encoder_lengths=encoded_len, transcripts=target,
                                      transcript_lengths=target_length, compute_wer=False)
            loss = model.add_auxiliary_losses(loss)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite loss at batch {batch_index}')
        (loss * volume).backward()
        total_loss += float(loss.detach()) * volume
        del signal, length, target, target_length, encoded, encoded_len, decoded, loss
        if AccessMixin.is_access_enabled(model.model_guid):
            AccessMixin.reset_registry(model)
        if (batch_index + 1) % accumulation and batch_index + 1 != len(batches):
            continue
        step += 1
        for parameter in model.parameters():
            if parameter.grad is None:
                raise RuntimeError('A trainable parameter received no gradient')
            parameter.grad.div_(tokens_in_group)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan['proposed_optimizer']['clip_grad_norm'],
                                                   error_if_nonfinite=True)
        if not (a.output / 'gradient-coverage.json').exists():
            coverage = {n: dict(trainable=v.requires_grad, has_gradient=v.grad is not None,
                                finite=bool(torch.isfinite(v.grad).all())) for n, v in model.named_parameters()}
            assert all(all(v.values()) for v in coverage.values())
            atomic_json(a.output / 'gradient-coverage.json', coverage)
        lr = learning_rate(step, full_steps, plan['proposed_optimizer']['peak_lr'],
                           plan['proposed_optimizer']['end_lr'], plan['proposed_optimizer']['warmup_fraction'])
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen.update(rows[i]['split'] for i in group_rows)
        metric = dict(step=step, total_steps=steps, next_batch=batch_index + 1, loss=total_loss / tokens_in_group,
                      lr=lr, gradient_norm=float(grad_norm), seen_rows=sum(seen.values()), seen_by_split=dict(seen),
                      elapsed_seconds=time.monotonic() - started,
                      cuda_peak_bytes=torch.cuda.max_memory_allocated())
        with (a.output / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(metric) + '\n')
        atomic_json(a.output / 'progress.json', metric)
        if step % 10 == 0 or step <= 3:
            print('TRAIN', json.dumps(metric), flush=True)
        if step % 25 == 0:
            verify(model, receipts)
        if step % 250 == 0 and not a.smoke_steps:
            checkpoint(batch_index + 1)
        if a.stop_after_step and step == a.stop_after_step:
            checkpoint(batch_index + 1)
            atomic_json(a.output / 'paused.json', metric)
            print('PAUSED_AT_OPTIMIZER_BOUNDARY', step, flush=True)
            return
        tokens_in_group, total_loss, group_rows = 0, 0.0, []

    verify(model, receipts)
    current_buffers = dict(model.named_buffers())
    for name, value in bn_state.items():
        assert torch.equal(current_buffers[name].cpu(), value), 'BN statistics changed: ' + name
    assert sum(seen.values()) == sum(map(len, batches))
    if not a.smoke_steps:
        assert dict(seen) == dict(Counter(r['split'] for r in rows))
        checkpoint(len(batches))
    final_state = materialized_state_dict(model, model.state_dict())
    assert set(final_state) == set(parent_hashes)
    assert all(torch.isfinite(v).all() for v in final_state.values() if v.is_floating_point())
    if initial_trainable_hashes is not None:
        unchanged = [n for n, v in model.named_parameters() if tensor_sha(v) == initial_trainable_hashes[n]]
        assert not unchanged, 'Trainable tensors did not change: ' + str(unchanged)
    audit = dict(frozen_gabor_rows_unchanged=12288, batchnorm_statistics_unchanged=True,
                 floating_tensors_finite=True, tensor_keys_match_parent=True,
                 changed_tensors=sum(tensor_sha(v) != parent_hashes[n] for n, v in final_state.items()),
                 seen_by_split=dict(seen), seen_rows=sum(seen.values()),
                 exactly_one_pass=not bool(a.smoke_steps), optimizer_steps=step)
    del final_state
    if not a.smoke_steps:
        original = model.state_dict
        def dense_state(module, *args, **kwargs):
            return materialized_state_dict(module, original(*args, **kwargs))
        model.state_dict = types.MethodType(dense_state, model)
        destination = a.checkpoint_dir / 'orukeet-targeted-ft.nemo'
        try:
            model.save_to(str(destination))
        finally:
            del model.state_dict
        audit.update(checkpoint_path=str(destination), checkpoint_sha256=sha(destination))
    atomic_json(a.output / 'complete.json', dict(run=run, audit=audit))
    print('COMPLETE', json.dumps(audit), flush=True)


if __name__ == '__main__':
    main()
