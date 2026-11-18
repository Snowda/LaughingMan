# LaughingMan build environment. It is the builder stage of the Containerfile (`nix-shell --run "cargo build"`)
# and it also works as a plain `nix-shell` on a Linux host. Both pins are fixed revisions with hashes, the
# same as Wander's shell.nix.
let
  # nixos-unstable, 2026-10-20.
  nixpkgs = fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/02c087f3fe673c9e317f206a627cb6b910336dbe.tar.gz";
    sha256 = "1l60ar0j7j7jrrg4r7j4kscrih0qn2n0lslffzflhv86lyq650r2";
  };
  # rust-overlay gives the exact rustc that Cargo.toml's rust-version and the Containerfile's rust:1.100.0 pin.
  rustOverlay = fetchTarball {
    url = "https://github.com/oxalica/rust-overlay/archive/8bbfc858c7b85f5e6ed6d6cc201ffcee1bb08a9b.tar.gz";
    sha256 = "0dk7kbba29zgdn1ydsr59ipfbn1rjwljcs73cclg0zwl2rninv1n";
  };
  pkgs = import nixpkgs { overlays = [ (import rustOverlay) ]; };
  rust = pkgs.rust-bin.stable."1.100.0".default.override {
    extensions = [ "rustfmt" "clippy" ];
  };
  # Runtime libraries: ash and winit dlopen these, so they go on LD_LIBRARY_PATH (and the image's rpath).
  runtimeLibs = with pkgs; [
    vulkan-loader
    libxkbcommon
    wayland
    libx11
    libxcb
    libxrandr
    libxi
    libxcursor
    libxinerama
    alsa-lib
  ];
in
pkgs.mkShell {
  # bindgenHook points v4l2-sys-mit's bindgen at libclang and the C headers (the image's libclang-dev).
  nativeBuildInputs = [ rust pkgs.pkg-config pkgs.cmake pkgs.rustPlatform.bindgenHook ];
  buildInputs = runtimeLibs ++ [ pkgs.vulkan-headers ];
  packages = with pkgs; [
    # `ntc` runs cargo-nextest; deny.toml is checked by cargo-deny.
    cargo-nextest
    cargo-deny
    # The Containerfile reads the binary's interpreter and rpath to copy its /nix/store closure.
    patchelf
    git
    cacert
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
  # Mesa's Vulkan drivers (lavapipe is the software fallback). The Containerfile ships them; the shell
  # does not point the loader at them, so a host keeps its own GPU driver.
  MESA_DRIVERS = pkgs.mesa;
}
