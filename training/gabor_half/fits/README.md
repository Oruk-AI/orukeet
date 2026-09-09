# Exact fitted kernels

These numerical artifacts contain fits to the original Orukeet checkpoint,
ranked across all 24,576 temporal depthwise kernels. Exactly 12,288 are selected.
They are the actual artifacts used by every recovery run in this experiment.

`fits.json.gz` contains all per-kernel parameters, errors, and selection flags.
Decompress it before passing the JSON path to the training or audit scripts:

```sh
python -c 'import gzip,pathlib; p=pathlib.Path("training/gabor_half/fits"); (p/"fits.json").write_bytes(gzip.decompress((p/"fits.json.gz").read_bytes()))'
```

`fits.npz` contains the original and fitted arrays used for ranking and plotting.
`manifest.json` records the source identity and hashes. Fit parameters use
float64; training taps use float32. Keep the recorded rounding path when
reconstructing the frozen rows.

These checkpoint-derived arrays inherit the release's **CC BY-SA 4.0** weight
license and attribution to NVIDIA's Parakeet TDT 0.6B v3 and Oruk AI's Orukeet
fine-tune. The surrounding experiment code is MIT-licensed. See [LICENSE-WEIGHTS](../../../LICENSE-WEIGHTS) and
[data provenance](../../../docs/data-and-licenses.md) for the source chain. These artifacts
remain private pending release review.
