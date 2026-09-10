pub mod bake;
pub mod svg_shape;

pub use bake::{BakeParams, Fill, bake_layers, bake_shape, bake_shape_in_frame, layers_bounds, shape_bounds};
pub use svg_shape::{shape_from_svg, shapes_from_svg, svg_fill_rule};
