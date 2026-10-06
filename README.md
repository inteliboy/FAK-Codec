# FAK Codec

FAK is a lossless audio codec: `decode(encode(x))` is bit-for-bit identical to the source PCM, and every
file stores a SHA-256 of the audio it was made from so that `fak verify` can prove it. This repository
contains the codec library, the `fak` command-line tool, the test suite and the bitstream specification.

- **Format:** version 21, specification 1.1.0 ([`docs/bitstream-spec.md`](docs/bitstream-spec.md)).
- **Language:** Rust (stable, edition 2021). The only runtime dependency is the optional `mimalloc` allocator.
- **License:** MIT OR Apache-2.0, at your option.
- **foobar2000:** the components (`foo_input_fak` for playback and Converter encoding, `foo_input_fak_adv` with the FAK menu and settings) are in a separate repository, [foo_input_fak](https://github.com/inteliboy/foo_input_fak).

## Compatibility

A decoder accepts exactly one format version (shown by `fak version`). Files written by an earlier
version (for example version 20) are rejected with an error, not misdecoded. Version 21 is the
current format; see the specification for what a version change means and what is guaranteed.

## What it handles

| | |
|---|---|
| Sample formats | 8-, 16-, 24- and 32-bit integer PCM; 32-bit IEEE float (exactly, see the specification) |
| Channels | 1 to 255 (stereo has dedicated decorrelation; other layouts are coded per channel with optional cross-channel prediction) |
| Container | WAV input and output (`WAVE_FORMAT_PCM`, `WAVE_FORMAT_IEEE_FLOAT`, `WAVE_FORMAT_EXTENSIBLE` with channel mask) |
| Metadata | Vorbis-comment style tags, embedded pictures (PNG/JPEG), embedded cue sheet (+ CD disc IDs and AccurateRip/CTDB checksums) |
| Seeking and streaming | Independent chunks (about one second by default): seek granularity is one chunk, and chunks decode in parallel |
| Error handling | Per-chunk CRC; optional whole-file Reed-Solomon parity (`--fec`, `--level archival`) that rebuilds damaged chunks |

## Platforms

Built and tested by the CI in this repository on every push:

| OS | Architecture |
|---|---|
| Linux | x86-64, aarch64 |
| macOS | arm64, x86-64 |
| Windows | x86-64 |

The decoder is deterministic and integer-exact, so a file decodes to the same PCM on every platform. The
single exception to "integer only" is the OLS chunk type of the format (chunk config 3 and 4, produced at
`--level insane`): it uses binary64 arithmetic and the specification requires correctly rounded IEEE
operations without fused multiply-add. In the development tree's CI, files produced on every platform are
decoded on a reference platform and compared with the source PCM. SIMD kernels (AVX2 on x86-64, NEON on aarch64) are selected at
runtime and are tested against a portable scalar implementation (see [Performance and SIMD](#performance-and-simd)).

The encoder is not required to be reproducible: two encodes of the same input may differ in bytes (hardware
acceleration, thread scheduling), and every such file decodes to the same audio.

## Build

```
cargo build --release          # target/release/fak (fak.exe on Windows)
cargo test --release
```

`cargo build --release --no-default-features` uses the system allocator instead of mimalloc.

## Performance and SIMD

**Build it the ordinary way.** `cargo build --release` produces one binary that is fast on every CPU it runs on.
You do not need `-C target-cpu=native`, `RUSTFLAGS`, an AVX2 or AVX-512 build, or a separate binary per CPU
generation; the release binaries are built exactly this way. A `target-cpu=native` binary
is the worse choice for distribution: it requires the build machine's exact instruction set and stops with an
illegal-instruction fault on an older CPU.

How that works: the hot loops are written as separate kernels, each compiled with its own
`#[target_feature(...)]`, and the program picks the best kernel the running CPU supports, once, at first use
(`is_x86_feature_detected!` / `is_aarch64_feature_detected!`). The rest of the binary is compiled for the
baseline target of your Rust toolchain (x86-64 SSE2, aarch64 NEON). `fak cpuinfo` prints what was detected and
which kernel each operation uses on your machine.

| Operation | x86-64 kernels | aarch64 |
|---|---|---|
| LPC residual search, candidate ranking, autocorrelation (encoder) | AVX2, AVX-512F, AVX-512 VNNI (16-bit ranking); AVX-512 is used only where it times faster than AVX2 on that CPU | NEON |
| LPC reconstruction (decoder) | blocked AVX2 | NEON |
| Stage-2 LMS predictor (encoder and decoder) | AVX2 | NEON |
| Carried long filter (`insane`; encoder and decoder) | AVX2 integer dot product (`vpmaddwd`) | portable code (left to the compiler's vectoriser) |
| OLS stage (`insane`, stereo up to 24 bits) | AVX2 builds of the f64 loops, with FMA left off | portable |
| Cross-channel sums, FFT analysis loops | AVX2 | NEON |
| Rice decoding | `lzcnt` + `bmi2` | portable |
| CRC-32 | PCLMULQDQ | CRC32 instructions |
| SHA-256 | SHA-NI | SHA2 instructions |

Every operation also has a portable scalar implementation, which is the reference: it is used on CPUs
without the instruction set and is what each SIMD kernel is tested against, by differential tests that
require identical output. **The choice of kernel never changes the decoded audio**: integer kernels are exact by
construction (wrapping sums are order-independent), and the floating-point kernels never use fused
multiply-add and keep the scalar operation order, so every operation rounds identically. Encoded bytes are not
guaranteed to be identical across machines (`--accel apple` on macOS is the documented exception, see `fak help`),
but every valid file decodes to exactly the source audio. In the measurement below the portable and native builds
did produce identical files.

Measured, so the advice above is not only an argument: the same source built portably and with
`RUSTFLAGS="-C target-cpu=native"` (which enables AVX2, BMI2 and FMA for the whole program on this machine), 12
excerpts, single-threaded, one run each, 48 of 48 decodes bit-exact:

| Level | Portable | `target-cpu=native` | Compressed bytes |
|---|---|---|---|
| `max` | 41x encode, 488x decode | 42x, 489x | identical |
| `insane` | 10.0x encode, 74x decode | 10.2x, 75x | identical |

The differences are 1-3%, within the run-to-run noise of a single timing, with byte-identical output. On this
machine native compilation buys nothing measurable because the kernels that matter are already compiled for
AVX2. Only an x86-64 Intel machine was measured; no aarch64 timing has been made (the aarch64 CI checks exactness, not speed).

For testing and timing, kernels can be switched off with environment variables set to `1`:
`FAK_DISABLE_AVX512`, `FAK_FORCE_AVX512`, `FAK_DISABLE_VNNI`, `FAK_DISABLE_AVX2_CLONES`, `FAK_DISABLE_HW_CRC`,
`FAK_DISABLE_SHA_NI`, `FAK_DISABLE_LZCNT`, `FAK_DISABLE_BLOCKED`. They change speed only. `fak decode -t N` and
`fak encode -t N` set the worker count (default: all logical CPUs); the output does not depend on it.

## Use

```
fak encode album.wav album.fak                       # level normal
fak encode --level max album.wav album.fak
fak encode --level archival album.wav album.fak      # insane + Reed-Solomon parity
fak decode album.fak album.wav
fak verify album.fak                                 # decode and check the stored SHA-256
fak info album.fak                                   # properties, tags, pictures, cue sheet
fak edit song.fak --set-tag TITLE=Better --picture front:cover.jpg
```

`fak help` lists every command and option.

## Compression levels

| Level | Description |
|---|---|
| `fast` | fixed 4096-sample frames |
| `normal` | frame lengths chosen by an estimate (default) |
| `max` | frame lengths chosen by trial encoding, about 4x the encode time of `normal` |
| `insane` | `max` plus adaptive prediction stages (a carried long filter and, for stereo up to 24 bits, an OLS predictor) |
| `archival` | `insane` plus Reed-Solomon parity |

Measured with FAK 1.1.3 on one machine (x86-64 Core Ultra X7 358H, Windows 11), one run, every codec single-threaded, on 20-second excerpts
(starting at 30 s) of 12 real music recordings (5 at 16-bit, 7 at 24-bit; classical, spoken-word, electronic
and vocal recordings; 240 s of audio in total), against FLAC 1.5.0 at `-8`; all 48 decodes were bit-exact:

| Codec | Size vs FLAC `-8` (total bytes) | Encode speed | Decode speed |
|---|---|---|---|
| FLAC 1.5.0 `-8` | baseline | 232x realtime | 612x realtime |
| FAK `max` | -2.03% | 42x | 498x |
| FAK `insane` | -2.82% | 10x | 75x (60x on 24-bit, 119x on 16-bit) |

`insane` was smaller than FLAC on all 12 files (from -1.6% to -6.3%); the gain is larger on the 16-bit
files (-5.1% in total) than on the 24-bit ones (-2.4%). Treat this as an indication, not a benchmark: it is
12 excerpts, a single timing run and a single machine, and the files are mostly classical music. Results on
other material, and on full-length files, will differ.

Ten of those 12 files were used while developing the `insane` predictors, so that figure is optimistic. The same
measurement was then made once on 5 recordings that were never used for tuning (2 piano pieces and a string
quartet at 24-bit, 2 electronic/pop tracks at 16-bit; 100 s of audio), with the same method;
all 20 decodes were bit-exact:

| Codec | Size vs FLAC `-8` (total bytes) | Per-file range | Encode speed | Decode speed |
|---|---|---|---|---|
| FLAC 1.5.0 `-8` | baseline | | 206x realtime | 556x realtime |
| FAK `max` | -1.74% | -1.1% to -4.6% | 34x | 420x |
| FAK `insane` | -2.41% | -1.8% to -5.4% | 9x | 52x |

On this set `insane` is 0.54% smaller than the 1.0.0 release's `insane` (every file smaller, -0.42% to -0.63%),
at about 8 times the decode time (425x realtime for 1.0.0's `insane`, 52x now). The sizes in this table were
measured with 1.1.0 and are unchanged in 1.1.3, which writes the same format and the same audio data; only the
speed columns were measured again (timing only, nothing was tuned on these files). Five files and one timing run
remain a small sample.

## Correctness and robustness

- Every file carries the SHA-256 of its source audio; `fak verify` decodes the whole file and compares.
- The test suite (`cargo test --release`) covers round trips at every compression level, bit depths 8 to 32,
  float input, extreme sample values, chunk and frame boundary lengths, damaged and truncated files, files
  with damaged chunks that parity repairs, the stereo OLS chunk, and decoding of stored files from a
  previous build (`tests/data`) so that a decoder change cannot silently alter what an existing file means.
- The decoder treats its input as untrusted: sizes are bounded before allocation, arithmetic is checked, and
  malformed files return an error. The test suite includes randomized corruption of valid files.

## Specification

[`docs/bitstream-spec.md`](docs/bitstream-spec.md) specifies the container, the chunk and frame structure,
every predictor and entropy-coding step the decoder performs, the checksums and the parity scheme.
Identifiers of the form `D-0xx` and `H-xx` that appear in source comments refer to design decisions and
experiments recorded in the development notes, which are not part of this repository.

## About this repository

The sources here are exported from the development tree; each commit is a snapshot of it. Please open an
issue for bugs, questions and proposals.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at
your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this project shall be dual licensed as above, without any additional terms or conditions.
