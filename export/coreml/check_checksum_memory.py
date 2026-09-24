#!/usr/bin/env python3
"""Measure the shipping Swift verifier on a model-sized synthetic file, no weights.

The subprocess isolates peak RSS from compiler/test-runner allocations. The
outer autorelease pool reproduces an installer executor that does not drain
Foundation's FileHandle/NSData objects between reads.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile


HARNESS = r'''
import Darwin
import Foundation

let file = URL(fileURLWithPath: CommandLine.arguments[1])
let count = Int64(CommandLine.arguments[2])!
let bundle = OrukeetBundle(url: file, bytes: count,
                           sha256: CommandLine.arguments[3], archiveRoot: "synthetic")
let start = ProcessInfo.processInfo.systemUptime
try autoreleasepool { try bundle.verifyArchive(at: file) }
var usage = rusage()
guard getrusage(RUSAGE_SELF, &usage) == 0 else { fatalError("getrusage failed") }
let result: [String: Any] = [
    "archive_bytes": count, "peak_rss_bytes": usage.ru_maxrss,
    "verification_seconds": ProcessInfo.processInfo.systemUptime - start,
    "checksum_verified": true,
]
print(String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).parent /
                        "benchmark/Sources/OrukeetCoreML/OrukeetBundle.swift")
    parser.add_argument("--bytes", type=int, default=554_985_744)
    parser.add_argument("--max-rss-mib", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        parser.error("This regression check uses Apple Foundation and Darwin RSS units")
    if args.bytes <= 0:
        parser.error("--bytes must be positive")
    with tempfile.TemporaryDirectory(prefix="orukeet-checksum-") as temporary:
        root = Path(temporary)
        (root / "main.swift").write_text(HARNESS)
        binary = root / "verify"
        subprocess.run(["xcrun", "swiftc", "-O", str(args.source.resolve()),
                        str(root / "main.swift"), "-o", str(binary)], check=True)
        payload = root / "synthetic.bin"
        with payload.open("wb") as stream:
            stream.truncate(args.bytes)  # Sparse zero file; no model data.
        digest = hashlib.sha256()
        chunk = bytes(1_048_576)
        left = args.bytes
        while left:
            size = min(left, len(chunk))
            digest.update(chunk[:size])
            left -= size
        result = json.loads(subprocess.check_output(
            [str(binary), str(payload), str(args.bytes), digest.hexdigest()], text=True))
    result.update({"source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                   "max_rss_bytes": args.max_rss_mib * 1_048_576,
                   "scope": "macOS checksum memory regression; synthetic zeros, no model weights",
                   "system": platform.platform()})
    result["passed"] = result["peak_rss_bytes"] <= result["max_rss_bytes"]
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")
    if not result["passed"]:
        raise SystemExit("SHA-256 verification exceeded bounded-memory threshold")


if __name__ == "__main__":
    main()
