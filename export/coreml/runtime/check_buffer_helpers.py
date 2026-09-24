#!/usr/bin/env python3
"""Run native Core ML safety checks against a checkout's actual array helpers.

Only Swift source and small in-memory arrays are used. No models are loaded or
fetched. This standalone check works with Command Line Tools without XCTest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

SOURCE = Path('Sources/FluidAudio/ASR/Parakeet/SlidingWindow/TDT/Decoder/TdtDecoderState.swift')

CHECKS = r'''
var assertions = 0
var cases: [String] = []
func check(_ condition: @autoclosure () -> Bool, _ message: String) {
    assertions += 1
    precondition(condition(), message)
}
func view(_ memory: UnsafeMutablePointer<Float>, offset: Int = 0,
          shape: [NSNumber], strides: [NSNumber]) throws -> MLMultiArray {
    try MLMultiArray(dataPointer: UnsafeMutableRawPointer(memory.advanced(by: offset)),
        shape: shape, dataType: .float32, strides: strides, deallocator: { _ in })
}
for type: MLMultiArrayDataType in [.float16, .float32, .float64, .int32] {
    let source = try MLMultiArray(shape: [3, 4], dataType: type)
    let destination = try MLMultiArray(shape: [3, 4], dataType: type)
    for fill: NSNumber in [0, 7, -2] {
        source.resetData(to: fill)
        for i in 0..<source.count { check(source[i] == fill, "dense fill \(type) \(i)") }
        destination.copyData(from: source)
        for i in 0..<source.count { check(destination[i] == fill, "dense copy \(type) \(i)") }
    }
    destination.copyData(from: destination)
    for i in 0..<destination.count { check(destination[i] == -2, "self copy") }
    cases.append("dense fill/copy/self-copy \(type)")
}
let backing = UnsafeMutablePointer<Float>.allocate(capacity: 32)
let other = UnsafeMutablePointer<Float>.allocate(capacity: 32)
defer { backing.deallocate(); other.deallocate() }
func seed() { for i in 0..<32 { backing[i] = Float(100+i); other[i] = Float(10+i) } }
let singleton = try view(backing, shape: [1, 4], strides: [16, 1])
let singletonSource = try view(other, shape: [1, 4], strides: [16, 1])
for fill: NSNumber in [0, 7] {
    seed(); singleton.resetData(to: fill)
    for i in 0..<4 { check(backing[i] == fill.floatValue, "singleton logical fill") }
    for i in 4..<32 { check(backing[i] == Float(100+i), "singleton trailing sentinel") }
}
seed(); singleton.copyData(from: singletonSource)
for i in 0..<4 { check(backing[i] == Float(10+i), "singleton logical copy") }
for i in 4..<32 { check(backing[i] == Float(100+i), "singleton copy tail sentinel") }
cases.append("contiguous singleton view preserves trailing storage")
let strided = try view(backing, shape: [2, 2], strides: [4, 1])
let stridedSource = try view(other, shape: [2, 2], strides: [4, 1])
let logical = Set([0, 1, 4, 5])
for fill: NSNumber in [0, 7] {
    seed(); strided.resetData(to: fill)
    for i in 0..<32 {
        check(backing[i] == (logical.contains(i) ? fill.floatValue : Float(100+i)), "strided fill sentinel")
    }
}
seed(); strided.copyData(from: stridedSource)
for i in 0..<32 {
    check(backing[i] == (logical.contains(i) ? Float(10+i) : Float(100+i)), "strided copy sentinel")
}
cases.append("strided fill/copy preserves gaps and tails")
for (sourceOffset, destinationOffset) in [(0, 2), (2, 0)] {
    seed()
    let source = try view(backing, offset: sourceOffset, shape: [1, 8], strides: [8, 1])
    let destination = try view(backing, offset: destinationOffset, shape: [1, 8], strides: [8, 1])
    destination.copyData(from: source)
    for i in 0..<32 {
        let expected = (destinationOffset..<destinationOffset+8).contains(i)
            ? Float(100+sourceOffset+i-destinationOffset) : Float(100+i)
        check(backing[i] == expected, "contiguous overlap \(sourceOffset)->\(destinationOffset)")
    }
}
cases.append("contiguous overlap forward and reverse")
for (sourceOffset, destinationOffset) in [(0, 1), (1, 0)] {
    seed()
    let source = try view(backing, offset: sourceOffset, shape: [2, 2], strides: [4, 1])
    let destination = try view(backing, offset: destinationOffset, shape: [2, 2], strides: [3, 1])
    let expected = [0, 1, 4, 5].map { Float(100+sourceOffset+$0) }
    destination.copyData(from: source)
    let targetIndices = [0, 1, 3, 4].map { destinationOffset+$0 }
    for i in 0..<32 {
        let value = targetIndices.firstIndex(of: i).map { expected[$0] } ?? Float(100+i)
        check(backing[i] == value, "different-layout overlap \(sourceOffset)->\(destinationOffset)")
    }
}
cases.append("different-layout overlap snapshots forward and reverse")
let doubleSource = try MLMultiArray(shape: [2, 3], dataType: .float64)
let intDestination = try MLMultiArray(shape: [2, 3], dataType: .int32)
for i in 0..<6 { doubleSource[i] = NSNumber(value: i+2) }
intDestination.copyData(from: doubleSource)
for i in 0..<6 { check(intDestination[i].intValue == i+2, "cross-type conversion") }
cases.append("different datatype logical conversion")
let large = try MLMultiArray(shape: [1, 240000], dataType: .float32)
large.resetData(to: 7)
large.resetData(to: 0)
for i in 0..<large.count { check(large[i].intValue == 0, "large logical reset") }
cases.append("complete 240000-element preprocessor reset")
let result: [String: Any] = ["status": "passed", "cases": cases, "assertions": assertions]
let data = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    production = (args.checkout / SOURCE).read_bytes()
    text = production.decode()
    marker = 'extension MLMultiArray {'
    if text.count(marker) != 1:
        raise ValueError('Expected exactly one production MLMultiArray helper extension')
    helpers = marker + text.split(marker, 1)[1]
    with tempfile.TemporaryDirectory(prefix='orukeet-buffer-checks-') as temporary:
        root = Path(temporary)
        swift = root / 'main.swift'
        swift.write_text('import Foundation\nimport CoreML\n' + helpers + '\n' + CHECKS)
        executable = root / 'check-buffers'
        subprocess.run(['swiftc', '-O', str(swift), '-o', str(executable)], check=True)
        result = json.loads(subprocess.check_output([str(executable)], text=True))
    result.update({
        'production_source': str(SOURCE),
        'production_source_sha256': hashlib.sha256(production).hexdigest(),
        'scope': 'Native macOS Core ML arrays; no model loading or iOS device execution',
        'swift_version': subprocess.check_output(['swift', '--version'], text=True).strip(),
    })
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
