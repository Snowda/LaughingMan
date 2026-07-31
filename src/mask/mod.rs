//! Offline MSDF bake (Phase 2): turn an SVG (the Laughing Man logo, split into a static layer and a
//! rotating text ring) into a multi-channel + true SDF (MTSDF) PNG atlas the shader samples. Two
//! steps: [`shape_from_svg`] parses the SVG into an fdsm shape, and [`bake_shape`] runs the fdsm
//! pipeline (fit → edge-color → generate MTSDF → sign + error correction) into an RGBA8 image.
//!
//! Gated behind the `bake` feature so the runtime app never pulls the fdsm/usvg toolchain.

pub mod bake;
pub mod svg_shape;

pub use bake::{BakeParams, Fill, bake_shape};
pub use svg_shape::{shape_from_svg, svg_fill_rule};
