# Mobile installation fixes

The SDK retains the authenticated portable ZIP inside each complete installation.
An OS update rebuilds from that local archive with no download, then atomically
publishes a new OS-specific compiled cache. A failed or cancelled rebuild leaves
the previous source intact. This layout remains compatible with OpenWhispr's
cleanup that preserves only the directory returned by `installedDirectory()`.

The Swift regression cases cover two successive OS migrations after deleting the
caller's ZIP and the obsolete cache, legacy receipt reuse/backfill, recompilation
failure and cancellation, retained-source corruption, and abandoned staging
directories. Synthetic archives contain no model weights.

## Checksum memory

`python3 export/coreml/check_checksum_memory.py` compiles the shipping verifier
and checks a sparse 554,985,744-byte zero file in a fresh child process. An outer
autorelease pool reproduces the Foundation buffer-lifetime issue independently
of test-runner or compiler allocations. CI requires peak process RSS <=128 MiB.

On this Mac (macOS 26.4.1, arm64):

| Verifier | Peak RSS | Verification time | Regression check |
| --- | ---: | ---: | --- |
| Prior implementation | 563,576,832 B | 0.415 s | Fails |
| Autorelease pool per 1 MiB read | 9,486,336 B | 0.388 s | Passes |

The memory check was mutation-validated against the prior verifier; both hash the
same file successfully, but the old code exceeds the memory ceiling. The JSON
receipts include source hashes and OS identity. These are checksum measurements,
not iPhone inference performance or full-install peak-memory measurements.

## Storage and migration

Retaining source adds 554,985,744 bytes to final installed storage. Extraction is
removed before retaining a copied caller archive, so normal first-install peak
storage is unchanged when extracted packages are larger than the ZIP. OS
recompilation keeps the old complete installation until publication; applications
must allow additional temporary space and only prune old siblings after success.

Legacy caches without a retained or caller-owned ZIP cannot reconstruct portable
source. They remain usable on their original OS and can be backfilled by calling
`install(fromArchive:)` without recompilation. A legacy cache whose source has
already been deleted requires one new download when the OS changes.
