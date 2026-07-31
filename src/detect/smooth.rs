//! One Euro filter (Casiez, Roussel, Vogel — CHI 2012): a speed-adaptive low-pass. Its cutoff rises
//! with the estimated speed (`cutoff = min_cutoff + beta·|ẋ|`), so it removes jitter at rest yet
//! barely lags fast motion — which a fixed-cutoff filter (EMA) cannot. Used to smooth the landmark
//! and box coordinates the overlay's rotation/placement come from.

use std::f32::consts::PI;

// Cutoff (Hz) of the low-pass applied to the derivative estimate.
const DEFAULT_D_CUTOFF: f32 = 1.0;
// Guards against a zero/negative timestep.
const MIN_DT: f32 = 1.0e-4;

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

    /// Filters sample `x` observed `dt` seconds after the previous one. The first sample passes
    /// through unchanged (no history yet).
    pub fn filter(&mut self, x: f32, dt: f32) -> f32 {
        let Some(xp) = self.x_prev else {
            self.x_prev = Some(x);
            return x;
        };
        let dt = dt.max(MIN_DT);
        // Low-pass the derivative, then set the cutoff from its magnitude.
        let dx = (x - xp) / dt;
        let a_d = alpha(self.d_cutoff, dt);
        let edx = a_d * dx + (1.0 - a_d) * self.dx_prev;
        self.dx_prev = edx;
        let cutoff = self.min_cutoff + self.beta * edx.abs();
        let a = alpha(cutoff, dt);
        let xf = a * x + (1.0 - a) * xp;
        self.x_prev = Some(xf);
        xf
    }
}

// The exponential-smoothing factor for a first-order low-pass at `cutoff` Hz over `dt` seconds.
fn alpha(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (2.0 * PI * cutoff);
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
#[allow(clippy::cast_precision_loss, clippy::as_conversions)]
mod tests {
    use super::OneEuro;

    const DT: f32 = 1.0 / 30.0;

    fn variance(xs: &[f32]) -> f32 {
        let n = xs.len() as f32;
        let mean = xs.iter().sum::<f32>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n
    }

    #[test]
    fn suppresses_jitter_on_a_static_signal() {
        // A static 100 with ±10 jitter. The filtered output's variance (after warmup) is far below
        // the input's — the point of the smoothing layer.
        let mut f = OneEuro::new(1.0, 0.0);
        let inputs: Vec<f32> = (0..60).map(|i| if i % 2 == 0 { 110.0 } else { 90.0 }).collect();
        let outputs: Vec<f32> = inputs.iter().map(|&x| f.filter(x, DT)).collect();
        let in_var = variance(&inputs[30..]);
        let out_var = variance(&outputs[30..]);
        assert!(out_var < in_var * 0.25, "in={in_var} out={out_var}");
    }

    #[test]
    fn tracks_a_ramp_with_little_lag_unlike_a_fixed_lowpass() {
        // A constant-velocity ramp. One Euro's speed adaptation (beta > 0) tracks it with small
        // steady-state lag; a fixed low-pass (beta = 0, ≈ EMA) lags far more — the property the plan
        // calls out ("which EMA would fail").
        let v = 300.0; // px/s
        let mut euro = OneEuro::new(1.0, 0.5);
        let mut fixed = OneEuro::new(1.0, 0.0);
        let (mut euro_err, mut fixed_err) = (0.0, 0.0);
        for i in 0..120 {
            let x = v * (i as f32) * DT;
            euro_err = (x - euro.filter(x, DT)).abs();
            fixed_err = (x - fixed.filter(x, DT)).abs();
        }
        assert!(euro_err < 5.0, "one euro steady-state lag small: {euro_err}");
        assert!(euro_err < fixed_err * 0.2, "speed adaptation beats a fixed low-pass: {euro_err} vs {fixed_err}");
    }

    #[test]
    fn first_sample_passes_through() {
        let mut f = OneEuro::new(1.0, 0.5);
        assert_eq!(f.filter(42.0, DT), 42.0);
    }
}
