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

#[cfg(test)]
mod tests {
    use super::Cast;

    // --- Widening / int→float is exact -------------------------------------------------

    #[test]
    fn widening_int_to_int_is_exact() {
        let x: u8 = 200;
        assert_eq!(x.to_u32(), 200u32);
        assert_eq!(x.to_u64(), 200u64);
        assert_eq!(x.to_usize(), 200usize);
        assert_eq!(x.to_i32(), 200i32);
        assert_eq!(x.to_i64(), 200i64);
        assert_eq!(1_000_000i32.to_i64(), 1_000_000i64);
    }

    #[test]
    fn small_int_to_float_is_exact() {
        assert_eq!(255u8.to_f32(), 255.0f32);
        assert_eq!(255u8.to_f64(), 255.0f64);
        assert_eq!((-3i32).to_f32(), -3.0f32);
        assert_eq!((-3i32).to_f64(), -3.0f64);
    }

    // --- int→int truncates (low bits kept, sign reinterpreted) --------------------------

    #[test]
    fn int_to_int_truncates_high_bits() {
        assert_eq!(300u32.to_u8(), 44u8); // 300 - 256
        assert_eq!(256u32.to_u8(), 0u8);
        assert_eq!(0x1_0000_0001u64.to_u32(), 1u32);
    }

    #[test]
    fn negative_to_unsigned_wraps_via_twos_complement() {
        assert_eq!((-1i32).to_u8(), 255u8);
        assert_eq!((-1i32).to_u32(), u32::MAX);
        assert_eq!((-1i64).to_u64(), u64::MAX);
        assert_eq!((-1i8).to_u8(), 255u8);
    }

    #[test]
    fn wide_unsigned_to_signed_reinterprets_bits() {
        assert_eq!(u32::MAX.to_i32(), -1i32);
        assert_eq!(u64::MAX.to_i64(), -1i64);
    }

    // --- float→int saturates (Rust `as` semantics), truncating toward zero --------------

    #[test]
    fn float_to_int_truncates_toward_zero() {
        assert_eq!(3.9f32.to_i32(), 3i32);
        assert_eq!((-3.9f32).to_i32(), -3i32);
        assert_eq!(3.9f64.to_u32(), 3u32);
    }

    #[test]
    fn float_above_range_saturates_to_max() {
        assert_eq!(300.0f32.to_u8(), 255u8);
        assert_eq!(f32::INFINITY.to_u32(), u32::MAX);
        assert_eq!(1e20f64.to_i32(), i32::MAX);
        assert_eq!(1e30f64.to_u64(), u64::MAX);
    }

    #[test]
    fn float_below_range_saturates_to_min() {
        assert_eq!((-5.0f32).to_u8(), 0u8);
        assert_eq!(f32::NEG_INFINITY.to_i32(), i32::MIN);
        assert_eq!((-1e20f64).to_i64(), i64::MIN);
        assert_eq!((-1.0f32).to_u64(), 0u64);
    }

    #[test]
    fn nan_to_int_is_zero() {
        assert_eq!(f32::NAN.to_i32(), 0i32);
        assert_eq!(f64::NAN.to_u8(), 0u8);
        assert_eq!(f32::NAN.to_usize(), 0usize);
    }

    // --- float→float ------------------------------------------------------------------

    #[test]
    fn float_widening_is_exact() {
        assert_eq!(0.5f32.to_f64(), 0.5f64);
        assert_eq!(1.5f64.to_f32(), 1.5f32); // representable in both
    }

    #[test]
    fn f64_to_f32_overflow_is_infinity() {
        // float→float overflow yields infinity, not a saturated finite max.
        assert_eq!(1e40f64.to_f32(), f32::INFINITY);
        assert_eq!((-1e40f64).to_f32(), f32::NEG_INFINITY);
    }

    #[test]
    fn f64_to_f32_loses_precision_but_stays_finite() {
        // 0.1 is unrepresentable in binary; narrowing changes the value but not its magnitude class.
        let narrowed = 0.1f64.to_f32();
        assert_ne!(narrowed.to_f64(), 0.1f64); // precision was lost
        assert!((narrowed.to_f64() - 0.1f64).abs() < 1e-7); // but only slightly
    }

    // --- every to_* method is exercised (100% function coverage) ------------------------

    #[test]
    fn all_methods_reachable_for_one_type() {
        let x: i16 = 42;
        assert_eq!(x.to_f32(), 42.0f32);
        assert_eq!(x.to_f64(), 42.0f64);
        assert_eq!(x.to_u8(), 42u8);
        assert_eq!(x.to_u32(), 42u32);
        assert_eq!(x.to_u64(), 42u64);
        assert_eq!(x.to_usize(), 42usize);
        assert_eq!(x.to_i32(), 42i32);
        assert_eq!(x.to_i64(), 42i64);
    }
}
