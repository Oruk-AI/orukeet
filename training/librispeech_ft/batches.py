"""Finite, deterministic duration buckets; no resampling or dropped tail."""
import math
import random


def make_batches(rows, seed, max_seconds=120, max_utterances=16, pool_size=2048):
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), pool_size):
        pool = sorted(order[start:start + pool_size], key=lambda i: rows[i]['duration'])
        batch, longest = [], 0
        for i in pool:
            duration = rows[i]['duration']
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f'Invalid duration at row {i}')
            if batch and (len(batch) == max_utterances or max(longest, duration) * (len(batch) + 1) > max_seconds):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(i)
            longest = max(longest, duration)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    flattened = [i for batch in batches for i in batch]
    assert len(flattened) == len(rows) and sorted(flattened) == list(range(len(rows)))
    return batches


def learning_rate(step, total_steps, peak, end, warmup_fraction):
    """One-based optimizer step; last update uses the declared terminal LR."""
    warmup = max(1, math.ceil(total_steps * warmup_fraction))
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return end + (peak - end) * (1 + math.cos(math.pi * progress)) / 2
