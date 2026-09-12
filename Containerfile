# Containerfile — Podman/OCI build for laughing-man (Linux/x86-64).
#
# laughing-man is a Vulkan + winit GUI app: it opens a window, captures a webcam
# (v4l2 on Linux), and renders on the GPU. A container therefore needs GPU, camera,
# display, and audio *passed through from the host* at run time — see podman-compose.yaml
# and the "Run" notes below. The image itself is just the built binary plus the Vulkan
# loader, an ICD (mesa/lavapipe as a software fallback), and the X11/Wayland/ALSA runtime libs.
#
# Path-dependency note: Cargo.toml depends on the sibling ../Aspire workspace by path
# (dsl, aspire, spirv-reader). Those live OUTSIDE this repo, so the Aspire tree is supplied
# as a named additional build context named `aspire` rather than being part of the main
# context. Build from the repo root with:
#
#   podman build -t laughing-man:latest --build-context aspire=../Aspire -f Containerfile .
#
# (podman-compose.yaml wires the same thing up via `additional_contexts`.)

# ---- Builder ---------------------------------------------------------------
# Pinned to the crate's rust-version (edition 2024). bookworm carries the -dev headers below.
FROM docker.io/library/rust:1.96-bookworm AS builder

# Build-time deps for the default runtime binary:
#   pkg-config + cmake     — used by several -sys crates during the build
#   libvulkan-dev          — ash links the Vulkan loader
#   libxkbcommon-dev, libwayland-dev, xorg dev libs — winit's windowing backends
#   libasound2-dev         — rodio/cpal (ALSA) audio output
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config cmake \
        libvulkan-dev \
        libxkbcommon-dev libwayland-dev \
        libx11-dev libxcb1-dev libxrandr-dev libxi-dev libxcursor-dev libxinerama-dev \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Mirror the on-disk sibling layout so the ../Aspire path deps resolve unchanged.
# The `aspire` build context is the sibling Aspire repo; copy only what the path deps
# and their workspace resolution need (root manifest + crates), not examples/target.
WORKDIR /src/Aspire
COPY --from=aspire Cargo.toml Cargo.lock ./
COPY --from=aspire crates ./crates

WORKDIR /src/LaughingMan
COPY . .

# Release build of the runtime app only (default features: no bake/detect/integration_tests).
# --locked honours the committed Cargo.lock for a reproducible build.
RUN cargo build --release --locked --bin laughing-man

# ---- Runtime ---------------------------------------------------------------
FROM docker.io/library/debian:bookworm-slim AS runtime

# Runtime libraries (the .so counterparts of the -dev packages above):
#   libvulkan1 + mesa-vulkan-drivers — Vulkan loader + ICDs (lavapipe gives a software
#     fallback so the image runs without a passed-through GPU, e.g. for --probe/--list)
#   libxkbcommon0/wayland/xorg libs  — winit dlopens these at window creation
#   libasound2                       — ALSA runtime for audio
#   ca-certificates                  — ffmpeg-sidecar downloads its ffmpeg binary over https
RUN apt-get update && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers \
        libxkbcommon0 libwayland-client0 \
        libx11-6 libxcb1 libxrandr2 libxi6 libxcursor1 libxinerama1 \
        libasound2 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Run as an unprivileged user; add to `video` so a passed-through /dev/dri and /dev/video*
# are accessible (host video GID may differ — override group_add / --group-add if so).
RUN useradd --create-home --uid 1000 app \
    && usermod -aG video app
USER app
WORKDIR /home/app

COPY --from=builder /src/LaughingMan/target/release/laughing-man /usr/local/bin/laughing-man

# ffmpeg-sidecar caches its downloaded binary under $HOME; keep it in a writable, mountable spot.
ENV HOME=/home/app

ENTRYPOINT ["laughing-man"]
# No default CMD: pass flags at run time, e.g. `--list`, `--probe`, `--test-pattern`,
# or `--video /media/clip.mp4`. Bare `podman run …` opens the live webcam window.
