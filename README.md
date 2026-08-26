# LaughingMan

> Overlays the Laughing Man logo from *Ghost in the Shell: Stand Alone Complex* onto faces in a
> webcam or video stream — a native Rust app that composites a multi-channel-SDF logo on the GPU.

![Windows Build Status](https://github.com/Snowda/LaughingMan/workflows/Windows/badge.svg)
![Linux Build Status](https://github.com/Snowda/LaughingMan/workflows/Linux/badge.svg)
![License](https://img.shields.io/github/license/Snowda/LaughingMan)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/Snowda/LaughingMan.svg)](http://isitmaintained.com/project/Snowda/LaughingMan "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/Snowda/LaughingMan.svg)](http://isitmaintained.com/project/Snowda/LaughingMan "Percentage of issues still open")

## Overview

A capture → detect → track → composite pipeline: a frame source (webcam, video file, or a synthetic
test pattern) is scanned for faces, and the logo is drawn over each face by a Vulkan compositor. The
logo is rendered from baked **MTSDF** atlases so it stays crisp at any size and its text ring can spin
independently. The overlay stays level with the camera (it does not tilt with the head).

## Build

Requires a recent stable Rust toolchain.

```bash
cargo build --release
```

Optional feature flags (off by default to keep the runtime build light):

| Feature | Enables |
|---------|---------|
| `detect` | The SCRFD ONNX face detector (pulls ONNX Runtime). Without it, a synthetic demo face drives the overlay. |
| `bake` | The `laughing-bake` tool + the `mask` module (pulls fdsm/usvg/resvg). Needed only to bake the logo atlases. |

## Quick start (no logo, no camera)

Runs a synthetic scrolling test pattern with a procedural placeholder logo — nothing to install:

```bash
cargo run --release -- --test-pattern
```

## Preparing the logo

The real Laughing Man art is copyrighted and **not included** — supply your own `laugh.svg`. The logo
is composited as **three layers**, each a separate SVG baked into its own atlas (all in one shared
frame so they stay aligned):

| Layer | Contents | Purpose |
|-------|----------|---------|
| `--static` | Rings, face features, cap, text band — everything except the rotating text | The still logo + the white face backing (derived at load) |
| `--text` | The circular quote only | Spins around the logo's centre |
| `--front` | Features + cap | Its silhouette occludes the ring so the cap reads as *in front* of the spinning text |

Split your `laugh.svg` into these three layer SVGs — each keeping only its own part (a vector editor
works, or delete the paths that don't belong to a layer). Name them `laugh-static.svg`,
`laugh-text.svg`, and `laugh-front.svg`, then bake them (the text layer is the slow one — thousands of
glyph segments):

```bash
cargo run --release --features bake --bin laughing-bake -- \
    --static laugh-static.svg --text laugh-text.svg --front laugh-front.svg
```

This writes `assets/static.mtsdf.png`, `assets/text.mtsdf.png`, and `assets/front.mtsdf.png`
(`--out-dir` defaults to `assets`, `--size` to 1024, `--range` to 8). A single combined SVG can also
be baked with `--svg <file>`.

## Running with the logo

```bash
cargo run --release -- \
    --logo-static assets/static.mtsdf.png \
    --logo-text   assets/text.mtsdf.png \
    --logo-front  assets/front.mtsdf.png
```

- `--logo-front` is optional; omit it and the spinning text draws over the cap instead of behind it.
- With only `--logo-static` (no `--logo-text`), a single combined atlas is composited without a
  spinning ring.
- With no logo flags at all, a procedural placeholder is used.

### Controls

| Key | Action |
|-----|--------|
| `S` | Save a full-resolution screenshot (video + overlay) to `screenshots/laugh_<timestamp>.png` |
| `Esc` / `Space` | Quit |

## Frame sources & options

| Flag | Effect |
|------|--------|
| *(default)* | Capture from webcam device `--source <index>` (default 0) |
| `--video <file>` | Decode an mp4/mkv/… as the source (ffmpeg is auto-downloaded on first use) |
| `--test-pattern` | Synthetic scrolling pattern, no camera |
| `--size WxH` | Requested capture resolution (e.g. `1280x720`); defaults to the camera's highest |
| `--list` | List capture devices and exit |
| `--probe` | Print capture dimensions + measured rate to stdout, no window |

## Face detection

Detection uses an SCRFD `*_kps` ONNX model and the `detect` feature. Without a model, a synthetic demo
face positions the overlay:

```bash
cargo run --release --features detect -- --model scrfd_500m_kps.onnx --video clip.mp4 \
    --logo-static assets/static.mtsdf.png --logo-text assets/text.mtsdf.png --logo-front assets/front.mtsdf.png
```

## Report Issues

[Submit an issue](https://github.com/Snowda/LaughingMan/issues)

## Contribute

[Submit a Pull Request!](https://github.com/Snowda/LaughingMan/pulls)

## License

Licensed under the [MIT](https://github.com/Snowda/LaughingMan/blob/master/LICENSE) License.
