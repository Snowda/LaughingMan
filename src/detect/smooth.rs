//! One Euro filter (Casiez et al., CHI 2012), a low-pass that adapts to speed. The cutoff is
//! `min_cutoff + beta·|ẋ|`. It removes jitter at rest and adds little lag to fast motion.

use std::f32::consts::PI;

// Cutoff (Hz) of the low-pass applied to the derivative estimate.
const DEFAULT_D_CUTOFF: f32 = 1.0;
/// The smallest timestep (seconds) the filters accept. It stops a zero or negative `dt`.
pub const MIN_DT: f32 = 1.0e-4;

/// A scalar One Euro filter.
pub struct OneEuro {
    min_cutoff: f32,
    beta: f32,
    d_cutoff: f32,
    x_prev: Option<f32>,
    dx_prev: f32,
}

impl OneEuro {
    #[must_use]
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff: DEFAULT_D_CUTOFF,
            x_prev: None,
            dx_prev: 0.0,
        }
    }

    /// Filters `x`, seen `dt` seconds after the previous sample. The first sample passes through.
    pub fn filter(&mut self, x: f32, dt: f32) -> f32 {
        let Some(xp) = self.x_prev else {
            self.x_prev = Some(x);
            return x;
        };
        let dt: f32 = dt.max(MIN_DT);
        // Low-pass the derivative, then set the cutoff from its magnitude.
        let dx: f32 = (x - xp) / dt;
        let a_d: f32 = alpha(self.d_cutoff, dt);
        let edx: f32 = a_d * dx + (1.0 - a_d) * self.dx_prev;
        self.dx_prev = edx;
        let cutoff: f32 = self.min_cutoff + self.beta * edx.abs();
        let a: f32 = alpha(cutoff, dt);
        let xf: f32 = a * x + (1.0 - a) * xp;
        self.x_prev = Some(xf);
        xf
    }
}

// The exponential factor of a first-order low-pass at `cutoff` Hz over `dt` seconds.
fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau: f32 = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// A 2D One Euro filter (independent per axis) — for landmark points.
pub struct OneEuroPoint {
    x: OneEuro,
    y: OneEuro,
}

impl OneEuroPoint {
    #[must_use]
    pub fn new(min_cutoff: f32, beta: f32) -> Self {
        Self {
            x: OneEuro::new(min_cutoff, beta),
            y: OneEuro::new(min_cutoff, beta),
        }
    }

    pub fn filter(&mut self, p: (f32, f32), dt: f32) -> (f32, f32) {
        (self.x.filter(p.0, dt), self.y.filter(p.1, dt))
    }
}

#[cfg(test)]
mod tests {
    use super::OneEuro;
    use crate::num::usize_to_f32;

    const DT: f32 = 1.0 / 30.0;

    fn variance(xs: &[f32]) -> f32 {
        let n: f32 = usize_to_f32(xs.len());
        let mean: f32 = xs.iter().sum::<f32>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn suppresses_jitter_on_a_static_signal() {
        // Static 100 ±10 jitter: filtered variance (post-warmup) is far below the input's.
        let mut f: OneEuro = OneEuro::new(1.0, 0.0);
        let inputs: [f32; 60] = std::array::from_fn(|i| if i % 2 == 0 { 110.0 } else { 90.0 });
        let outputs: [f32; 60] = inputs.map(|x| f.filter(x, DT));
        let in_var: f32 = variance(&inputs[30..]);
        let out_var: f32 = variance(&outputs[30..]);
        assert!(out_var < in_var * 0.25, "in={in_var} out={out_var}");
    }

    #[test]
    fn tracks_a_ramp_with_little_lag_unlike_a_fixed_lowpass() {
        // A ramp at constant velocity. One Euro (beta>0) has a small lag. A fixed low-pass lags far more.
        let v: f32 = 300.0; // px/s
        let mut euro: OneEuro = OneEuro::new(1.0, 0.5);
        let mut fixed: OneEuro = OneEuro::new(1.0, 0.0);
        let (mut euro_err, mut fixed_err) = (0.0, 0.0);
        for i in 0..120u16 {
            let x: f32 = v * f32::from(i) * DT;
            euro_err = (x - euro.filter(x, DT)).abs();
            fixed_err = (x - fixed.filter(x, DT)).abs();
        }
        assert!(euro_err < 5.0, "one euro steady-state lag small: {euro_err}");
        assert!(euro_err < fixed_err * 0.2, "speed adaptation beats a fixed low-pass: {euro_err} vs {fixed_err}");
    }

    #[test]
    fn first_sample_passes_through() {
        let mut f: OneEuro = OneEuro::new(1.0, 0.5);
        assert_eq!(f.filter(42.0, DT).to_bits(), 42.0f32.to_bits());
    }
}
