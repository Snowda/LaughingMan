//! Centralized numeric casts: the ONE module re-permitting the crate's denied `as`-cast lints, for
//! the graphics/DSP float↔int narrowing that has no `TryFrom`. Every cast goes through these `to_*`
//! helpers (each exactly `self as T`; float→int saturates, int→int truncates, widening is exact).
#![allow(
    clippy::as_conversions,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::unnecessary_cast
)]

/// `as`-cast conversions to the numeric types the crate uses, behind one auditable trait.
pub trait Cast: Copy {
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
    fn to_u8(self) -> u8;
    fn to_u32(self) -> u32;
    fn to_u64(self) -> u64;
    fn to_usize(self) -> usize;
    fn to_i32(self) -> i32;
    fn to_i64(self) -> i64;
}

macro_rules! impl_cast {
    ($($t:ty),+ $(,)?) => { $(
        impl Cast for $t {
            #[inline]
            fn to_f32(self) -> f32 { self as f32 }
            #[inline]
            fn to_f64(self) -> f64 { self as f64 }
            #[inline]
            fn to_u8(self) -> u8 { self as u8 }
            #[inline]
            fn to_u32(self) -> u32 { self as u32 }
            #[inline]
            fn to_u64(self) -> u64 { self as u64 }
            #[inline]
            fn to_usize(self) -> usize { self as usize }
            #[inline]
            fn to_i32(self) -> i32 { self as i32 }
            #[inline]
            fn to_i64(self) -> i64 { self as i64 }
        }
    )+ };
}

impl_cast!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64);
