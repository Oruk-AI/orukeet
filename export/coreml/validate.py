#!/usr/bin/env python3
"""Compare converted components with the verified NeMo source on real audio."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import coremltools as ct
import numpy as np
import soundfile as sf
import torch

from convert import SOURCE_SHA256, Weights, operations, read_blob, sha256


class FixedPosition(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values)

    def forward(self, _):
        return self.values


def quantized_encoder(model, package, weights):
    """Execute the exported weights in NeMo to separate compression from conversion.

    The shipped graph quantizes the precomputed position projections and folded
    convolutions. Reproduce those representations explicitly in this oracle.
    """
    encoder = copy.deepcopy(model.encoder).eval()
    targets = encoder.state_dict()
    with (package / "Data/com.apple.CoreML/weights/weight.bin").open("rb") as handle:
        for op in operations(package):
            name = op.outputs[0].name
            if op.type == "constexpr_lut_to_dense":
                lut = read_blob(handle, op.attributes["lut"], np.float16)
                packed = read_blob(handle, op.attributes["indices"], np.uint8)
                shape = tuple(d.constant.size for d in op.outputs[0].type.tensorType.dimensions)
                count = int(np.prod(shape))
                bits = np.unpackbits(packed, bitorder="little")[:count * 6].reshape(count, 6)
                indices = (bits * (1 << np.arange(6))).sum(axis=1)
                array = lut[indices].reshape(shape).astype(np.float32)
            elif op.type == "const" and op.attributes["val"].WhichOneof("value") == "blobFileValue":
                shape = tuple(d.constant.size for d in op.outputs[0].type.tensorType.dimensions)
                array = read_blob(handle, op.attributes["val"], np.float16).reshape(shape).astype(np.float32)
            else:
                continue
            name = name.removesuffix("_palettized").removesuffix("_to_fp16")
            tensor = torch.from_numpy(array)
            if name.startswith("module_"):
                key = weights.names["encoder_" + name.removeprefix("module_")].removeprefix("encoder.")
                targets[key].copy_(tensor)
            elif name.startswith("op_"):
                layer = (int(name[3:]) - 273) // 163
                values = tensor.permute(0, 3, 1, 2).reshape(1, 375, 1024).contiguous()
                encoder.layers[layer].self_attn.linear_pos = FixedPosition(values)
            elif name.startswith("const_"):
                layer, is_bias = divmod(int(name[6:]) - 248, 2)
                conv = encoder.layers[layer].conv
                if is_bias:
                    conv.depthwise_conv.bias = torch.nn.Parameter(tensor, requires_grad=False)
                    conv.batch_norm = torch.nn.Identity()
                else:
                    conv.depthwise_conv.weight.data.copy_(tensor)
    return encoder


def difference(expected, actual):
    a = np.asarray(expected, dtype=np.float64).ravel()
    b = np.asarray(actual, dtype=np.float64).ravel()
    if a.shape != b.shape or not np.isfinite(b).all():
        raise ValueError("Non-finite or shape-mismatched output")
    denominator = np.linalg.norm(a)
    return {"relative_l2": float(np.linalg.norm(a - b) / max(denominator, 1e-12)),
            "max_abs": float(np.max(np.abs(a - b))),
            "cosine": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--audio", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("Unexpected source checkpoint")
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    torch.set_num_threads(4)
    model = EncDecRNNTBPEModel.restore_from(str(args.source), map_location="cpu").eval()
    model.preprocessor.featurizer.dither = 0
    model.decoder._rnnt_export = True
    # Verify positional folding against the actual NeMo module, independently of
    # the exporter's formula, before checking quantized inference.
    state = model.state_dict()
    weights = Weights(state)
    _, pe = model.encoder.pos_enc(torch.zeros(1, 188, 1024))
    position_errors = []
    for layer in range(24):
        expected = model.encoder.layers[layer].self_attn.linear_pos(pe).view(1, 375, 8, 128).permute(0, 2, 3, 1)
        actual = weights.tensor("Encoder", f"op_{273 + 163 * layer}_to_fp16_palettized")
        position_errors.append(float((expected - actual).abs().max()))
    if max(position_errors) > 1e-5:
        raise ValueError(f"Incorrect positional folding: {max(position_errors)}")
    with torch.no_grad():
        quantized = quantized_encoder(model, args.models / "Encoder.mlpackage", weights)
    components = {name: ct.models.CompiledMLModel(str(args.models / f"{name}.mlmodelc"), compute_units=ct.ComputeUnit.CPU_ONLY)
                  for name in ["Preprocessor", "Encoder", "Decoder", "JointDecisionv3"]}
    report = {"source_sha256": SOURCE_SHA256, "position_max_abs": max(position_errors), "fixtures": []}
    failures = []
    with torch.inference_mode():
        for audio_path in args.audio:
            data, rate = sf.read(audio_path, dtype="float32")
            if rate != 16000 or data.ndim != 1 or len(data) == 0 or len(data) > 240000:
                raise ValueError("Validation audio must be 16 kHz mono and at most 15 seconds")
            audio = np.pad(data, (0, 240000 - len(data)))[None, :]
            mel = components["Preprocessor"].predict({"audio_signal": audio, "audio_length": np.array([len(data)], np.int32)})
            encoded = components["Encoder"].predict(mel)
            expected, expected_length = model.encoder(audio_signal=torch.from_numpy(mel["mel"]), length=torch.from_numpy(mel["mel_length"]).long())
            length = int(expected_length[0])
            if int(encoded["encoder_length"][0]) != length:
                raise ValueError("Encoder lengths disagree")
            error = difference(expected[:, :, :length], encoded["encoder"][:, :, :length])
            quantized_expected, _ = quantized(audio_signal=torch.from_numpy(mel["mel"]), length=torch.from_numpy(mel["mel_length"]).long())
            conversion_error = difference(quantized_expected[:, :, :length], encoded["encoder"][:, :, :length])
            # Compare the SAME exported weights, executed independently in NeMo.
            # Full-precision drift remains reported separately; WER is a separate gate.
            if conversion_error["relative_l2"] > 0.05 or conversion_error["cosine"] < 0.999:
                failures.append(f"{audio_path.name}: encoder conversion drift")
            record = {"audio": str(audio_path), "sha256": sha256(audio_path), "encoder_vs_fp32": error,
                      "encoder_vs_same_quantized_weights": conversion_error, "decoder_steps": []}
            h = np.zeros((2, 1, 640), np.float32)
            c = np.zeros_like(h)
            target = 8192
            for frame in np.linspace(0, length - 1, min(24, length), dtype=int):
                targets = np.array([[target]], np.int32)
                dec = components["Decoder"].predict({"targets": targets, "target_length": np.array([1], np.int32), "h_in": h, "c_in": c})
                ref_dec, _, ref_state = model.decoder(targets=torch.from_numpy(targets).long(), target_length=torch.tensor([1]), states=[torch.from_numpy(h), torch.from_numpy(c)])
                dec_error = difference(ref_dec, dec["decoder"])
                state_error = difference(torch.stack(ref_state), np.stack([dec["h_out"], dec["c_out"]]))
                if dec_error["relative_l2"] > 0.03 or state_error["relative_l2"] > 0.03:
                    failures.append(f"{audio_path.name}: decoder/state numerical drift")
                enc_step = encoded["encoder"][:, :, frame:frame + 1].copy()
                joint = components["JointDecisionv3"].predict({"encoder_step": enc_step, "decoder_step": dec["decoder"]})
                projected = model.joint.enc(torch.from_numpy(enc_step).transpose(1, 2)).unsqueeze(2)
                projected = projected + model.joint.pred(torch.from_numpy(dec["decoder"]).transpose(1, 2)).unsqueeze(1)
                logits = model.joint.joint_net(projected).numpy()
                expected_token = int(np.argmax(logits[..., :8193]))
                expected_duration = int(np.argmax(logits[..., 8193:]))
                actual_token = int(joint["token_id"].item())
                actual_duration = int(joint["duration"].item())
                record["decoder_steps"].append({"frame": int(frame), "decoder": dec_error, "state": state_error,
                                                 "token_match": expected_token == actual_token,
                                                 "duration_match": expected_duration == actual_duration})
                h, c = dec["h_out"], dec["c_out"]
                target = actual_token
            # NeMo transcription is recorded separately from the same-mel tensor
            # test, so front-end/decoder policy differences remain visible.
            model.decoder._rnnt_export = False
            hypotheses = model.transcribe(audio=[data], batch_size=1, return_hypotheses=True, verbose=False)
            record["source_transcript"] = hypotheses[0].text
            # NeMo 2.3.1 unfreezes submodules after transcribe(), which puts them
            # back into training mode even when the parent began in eval mode.
            model.eval()
            model.decoder._rnnt_export = True
            report["fixtures"].append(record)
            print(audio_path.name, error, record["source_transcript"], flush=True)
    report["failures"] = failures
    report["passed"] = not failures
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit("Conversion validation failed; see report")


if __name__ == "__main__":
    main()
