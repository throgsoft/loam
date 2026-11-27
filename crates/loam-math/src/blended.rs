use glam::{Mat3, Vec3};

use crate::space::Space;

/// `g_ij(p) = f(p)·δ_ij` in the chart `Space::Point` carries, not in some other chart.
pub trait ConformallyFlat: Space {
    /// May return `f32::INFINITY` at the chart boundary.
    fn conformal_factor(&self, p: Vec3) -> f32;

    /// `R = −(4/f)·[∇²φ + |∇φ|²/2]`; the default is finite-difference.
    fn scalar_curvature(&self, p: Vec3) -> f32 {
        const EPS: f32 = 5.0e-3;
        let phi_at = self.conformal_log_half(p);
        let lap = (self.conformal_log_half(p + Vec3::X * EPS)
            + self.conformal_log_half(p - Vec3::X * EPS)
            + self.conformal_log_half(p + Vec3::Y * EPS)
            + self.conformal_log_half(p - Vec3::Y * EPS)
            + self.conformal_log_half(p + Vec3::Z * EPS)
            + self.conformal_log_half(p - Vec3::Z * EPS)
            - 6.0 * phi_at)
            / (EPS * EPS);
        let grad = self.conformal_log_half_gradient(p);
        let grad_sq = grad.length_squared();
        let f_p = self.conformal_factor(p);
        if !f_p.is_finite() || f_p <= 0.0 {
            return 0.0;
        }
        -(4.0 / f_p) * (lap + 0.5 * grad_sq)
    }

    fn conformal_log_half(&self, p: Vec3) -> f32 {
        0.5 * self.conformal_factor(p).ln()
    }

    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.conformal_log_half(p + Vec3::X * EPS)
            - self.conformal_log_half(p - Vec3::X * EPS))
            / (2.0 * EPS);
        let dy = (self.conformal_log_half(p + Vec3::Y * EPS)
            - self.conformal_log_half(p - Vec3::Y * EPS))
            / (2.0 * EPS);
        let dz = (self.conformal_log_half(p + Vec3::Z * EPS)
            - self.conformal_log_half(p - Vec3::Z * EPS))
            / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

impl ConformallyFlat for crate::EuclideanR3 {
    fn conformal_factor(&self, _p: Vec3) -> f32 {
        1.0
    }
    fn conformal_log_half(&self, _p: Vec3) -> f32 {
        0.0
    }
    fn conformal_log_half_gradient(&self, _p: Vec3) -> Vec3 {
        Vec3::ZERO
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        0.0
    }
}

impl ConformallyFlat for crate::HyperbolicH3 {
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        4.0 / (denom * denom)
    }
    fn conformal_log_half(&self, p: Vec3) -> f32 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return f32::INFINITY;
        }
        std::f32::consts::LN_2 - denom.ln()
    }
    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let r2 = p.length_squared();
        let denom = (1.0 - r2).max(0.0);
        if denom <= 0.0 {
            return Vec3::ZERO;
        }
        p * (2.0 / denom)
    }
    fn scalar_curvature(&self, _p: Vec3) -> f32 {
        -6.0
    }
}

/// Metric `g = (1 − α)·g_A + α·g_B`, with a spatially varying [`BlendingField`].
pub struct BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub a: A,
    pub b: B,
    pub field: F,
}

impl<A, B, F> BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3>,
    B: Space<Point = Vec3, Vector = Vec3>,
    F: BlendingField,
{
    pub fn new(a: A, b: B, field: F) -> Self {
        Self { a, b, field }
    }
}

impl<A, B, F> Space for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    type Point = Vec3;
    type Vector = Vec3;
    type Frame = Mat3;

    fn frame_at(&self, at: Vec3) -> Mat3 {
        Mat3::from_diagonal(Vec3::splat(1.0 / self.conformal_factor(at).sqrt()))
    }

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        let log = self.log(a, b);
        let f_a = self.conformal_factor(a);
        f_a.sqrt() * log.length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        rk4_geodesic(self, at, v, GEODESIC_DEFAULT_STEPS).0
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        gauss_newton_log(self, from, to, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        transport_along_geodesic(self, from, to, v)
    }

    fn chart_envelope(&self) -> f32 {
        self.a.chart_envelope().min(self.b.chart_envelope())
    }

    fn valid_point(&self, p: Vec3) -> bool {
        self.a.valid_point(p) && self.b.valid_point(p)
    }
}

pub const GEODESIC_DEFAULT_STEPS: u32 = 32;

/// Maximum accepted heuristic from metric-scaled RK4 step doubling.
pub const GEODESIC_ERROR_BUDGET: f32 = 1.0e-3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicState {
    pub point: Vec3,
    pub velocity: Vec3,
    pub frame: Mat3,
    pub estimated_error: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicLog {
    pub vector: Vec3,
    pub estimated_error: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeodesicError {
    InvalidStepCount,
    InvalidErrorBudget,
    NonFinite,
    ChartBoundary,
    ErrorBudget { estimated: f32, budget: f32 },
    Singular,
    NoConvergence,
}

#[derive(Clone, Copy)]
struct GeodesicDerivative {
    point: Vec3,
    velocity: Vec3,
    frame: Mat3,
}

fn validate_geodesic_point<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    point: Vec3,
) -> Result<(), GeodesicError> {
    if !point.is_finite() {
        return Err(GeodesicError::NonFinite);
    }
    if !space.valid_point(point) {
        return Err(GeodesicError::ChartBoundary);
    }
    Ok(())
}

// Wald, General Relativity, 1984, App. D.
fn transport_rhs(gradient: Vec3, tangent: Vec3, v: Vec3) -> Vec3 {
    -(v * gradient.dot(tangent) + tangent * gradient.dot(v) - gradient * tangent.dot(v))
}

#[inline]
fn geodesic_derivative<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    state: &GeodesicState,
    fault: &mut Option<GeodesicError>,
) -> GeodesicDerivative {
    let gradient = space.conformal_log_half_gradient(state.point);
    let velocity_squared = state.velocity.length_squared();
    let derivative = GeodesicDerivative {
        point: state.velocity,
        velocity: gradient * velocity_squared
            - state.velocity * (2.0 * gradient.dot(state.velocity)),
        frame: Mat3::from_cols(
            transport_rhs(gradient, state.velocity, state.frame.x_axis),
            transport_rhs(gradient, state.velocity, state.frame.y_axis),
            transport_rhs(gradient, state.velocity, state.frame.z_axis),
        ),
    };
    if fault.is_none() {
        *fault = match validate_geodesic_point(space, state.point) {
            Err(found) => Some(found),
            Ok(()) => (!state.velocity.is_finite()
                || !state.frame.is_finite()
                || !gradient.is_finite()
                || !derivative.velocity.is_finite()
                || !derivative.frame.is_finite())
            .then_some(GeodesicError::NonFinite),
        };
    }
    derivative
}

fn advance_geodesic_state(
    state: GeodesicState,
    derivative: GeodesicDerivative,
    scale: f32,
) -> GeodesicState {
    GeodesicState {
        point: state.point + derivative.point * scale,
        velocity: state.velocity + derivative.velocity * scale,
        frame: state.frame + derivative.frame * scale,
        estimated_error: state.estimated_error,
    }
}

fn rk4_geodesic_frame_step<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    state: GeodesicState,
    h: f32,
) -> (GeodesicState, Option<GeodesicError>) {
    let mut fault = None;
    let k1 = geodesic_derivative(space, &state, &mut fault);
    let k2 = geodesic_derivative(
        space,
        &advance_geodesic_state(state, k1, h * 0.5),
        &mut fault,
    );
    let k3 = geodesic_derivative(
        space,
        &advance_geodesic_state(state, k2, h * 0.5),
        &mut fault,
    );
    let k4 = geodesic_derivative(space, &advance_geodesic_state(state, k3, h), &mut fault);
    let point = state.point + (k1.point + k2.point * 2.0 + k3.point * 2.0 + k4.point) * (h / 6.0);
    let velocity = state.velocity
        + (k1.velocity + k2.velocity * 2.0 + k3.velocity * 2.0 + k4.velocity) * (h / 6.0);
    let frame = state.frame + (k1.frame + k2.frame * 2.0 + k3.frame * 2.0 + k4.frame) * (h / 6.0);
    let next = GeodesicState {
        point,
        velocity,
        frame,
        estimated_error: state.estimated_error,
    };
    geodesic_derivative(space, &next, &mut fault);
    (next, fault)
}

fn geodesic_state_error<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    coarse: GeodesicState,
    refined: GeodesicState,
) -> Result<f32, GeodesicError> {
    let factor = space.conformal_factor(refined.point);
    if !factor.is_finite() || factor <= 0.0 {
        return Err(GeodesicError::NonFinite);
    }
    let scale = factor.sqrt() / 15.0;
    let point = (refined.point - coarse.point).length() * scale;
    let velocity = (refined.velocity - coarse.velocity).length() * scale;
    let frame = (refined.frame.x_axis - coarse.frame.x_axis)
        .length()
        .max((refined.frame.y_axis - coarse.frame.y_axis).length())
        .max((refined.frame.z_axis - coarse.frame.z_axis).length())
        * scale;
    let error = point.max(velocity).max(frame);
    if !error.is_finite() {
        return Err(GeodesicError::NonFinite);
    }
    Ok(error)
}

fn integrate_geodesic_frame<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    at: Vec3,
    velocity: Vec3,
    frame: Mat3,
    n_steps: u32,
    error_budget: Option<f32>,
) -> (GeodesicState, Option<GeodesicError>) {
    let mut state = GeodesicState {
        point: at,
        velocity,
        frame,
        estimated_error: 0.0,
    };
    if let Some(budget) = error_budget {
        if n_steps == 0 {
            return (state, Some(GeodesicError::InvalidStepCount));
        }
        if !budget.is_finite() || budget <= 0.0 {
            return (state, Some(GeodesicError::InvalidErrorBudget));
        }
    }
    let h = 1.0 / n_steps as f32;
    for _ in 0..n_steps {
        let (coarse, fault) = rk4_geodesic_frame_step(space, state, h);
        let Some(budget) = error_budget else {
            if !coarse.point.is_finite() || !coarse.velocity.is_finite() {
                tracing::warn!(
                    "rk4_geodesic step produced non-finite state; clamping to previous step"
                );
                break;
            }
            state = coarse;
            continue;
        };
        if let Some(fault) = fault {
            return (state, Some(fault));
        }
        let (half, fault) = rk4_geodesic_frame_step(space, state, h * 0.5);
        if let Some(fault) = fault {
            return (state, Some(fault));
        }
        let (refined, fault) = rk4_geodesic_frame_step(space, half, h * 0.5);
        if let Some(fault) = fault {
            return (state, Some(fault));
        }
        let estimated_error = match geodesic_state_error(space, coarse, refined) {
            Ok(error) => state.estimated_error + error,
            Err(fault) => return (state, Some(fault)),
        };
        if estimated_error > budget {
            return (
                state,
                Some(GeodesicError::ErrorBudget {
                    estimated: estimated_error,
                    budget,
                }),
            );
        }
        state = GeodesicState {
            estimated_error,
            ..refined
        };
    }
    (state, None)
}

pub fn integrate_geodesic_frame_checked<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    at: Vec3,
    velocity: Vec3,
    frame: Mat3,
    n_steps: u32,
    error_budget: f32,
) -> Result<GeodesicState, GeodesicError> {
    match integrate_geodesic_frame(space, at, velocity, frame, n_steps, Some(error_budget)) {
        (state, None) => Ok(state),
        (_, Some(fault)) => Err(fault),
    }
}

// Press et al., Numerical Recipes, 3rd ed., 2007, §18.1; Nocedal and Wright, Numerical Optimization, 2nd ed., 2006, ch. 10.
fn gauss_newton_log_core<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
    error_budget: Option<f32>,
) -> (GeodesicLog, Option<GeodesicError>) {
    let mut log = GeodesicLog {
        vector: Vec3::ZERO,
        estimated_error: 0.0,
    };
    let mut target_scale = 0.0;
    if let Some(budget) = error_budget {
        if n_steps == 0 {
            return (log, Some(GeodesicError::InvalidStepCount));
        }
        if !budget.is_finite() || budget <= 0.0 {
            return (log, Some(GeodesicError::InvalidErrorBudget));
        }
        if let Err(fault) = validate_geodesic_point(space, from) {
            return (log, Some(fault));
        }
        if let Err(fault) = validate_geodesic_point(space, to) {
            return (log, Some(fault));
        }
        if from == to {
            return (log, None);
        }
        let target_factor = space.conformal_factor(to);
        if !target_factor.is_finite() || target_factor <= 0.0 {
            return (log, Some(GeodesicError::NonFinite));
        }
        target_scale = target_factor.sqrt();
    } else if from == to {
        return (log, None);
    }
    log.vector = to - from;
    for _ in 0..max_iters {
        let (state, fault) = integrate_geodesic_frame(
            space,
            from,
            log.vector,
            Mat3::IDENTITY,
            n_steps,
            error_budget,
        );
        if let Some(fault) = fault {
            return (log, Some(fault));
        }
        let residual = to - state.point;
        log.estimated_error = state.estimated_error + residual.length() * target_scale;
        let converged = match error_budget {
            Some(budget) => log.estimated_error <= budget,
            None => residual.length() < LOG_RESIDUAL_TOL,
        };
        if converged {
            return (log, None);
        }
        let two_eps = 2.0 * LOG_JACOBIAN_EPS;
        let mut jacobian = Mat3::ZERO;
        for axis in 0..3 {
            let mut delta = Vec3::ZERO;
            delta[axis] = LOG_JACOBIAN_EPS;
            let (plus, fault) = integrate_geodesic_frame(
                space,
                from,
                log.vector + delta,
                Mat3::IDENTITY,
                n_steps,
                error_budget,
            );
            if let Some(fault) = fault {
                return (log, Some(fault));
            }
            let (minus, fault) = integrate_geodesic_frame(
                space,
                from,
                log.vector - delta,
                Mat3::IDENTITY,
                n_steps,
                error_budget,
            );
            if let Some(fault) = fault {
                return (log, Some(fault));
            }
            *jacobian.col_mut(axis) = (plus.point - minus.point) / two_eps;
        }
        let determinant = jacobian.determinant();
        if !determinant.is_finite() {
            return (log, Some(GeodesicError::NonFinite));
        }
        if determinant.abs() < 1.0e-8 {
            return (log, Some(GeodesicError::Singular));
        }
        let next = log.vector + jacobian.inverse() * residual;
        if !next.is_finite() {
            return (log, Some(GeodesicError::NonFinite));
        }
        log.vector = next;
    }
    (log, Some(GeodesicError::NoConvergence))
}

pub fn geodesic_log_checked<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
    error_budget: f32,
) -> Result<GeodesicLog, GeodesicError> {
    match gauss_newton_log_core(space, from, to, n_steps, max_iters, Some(error_budget)) {
        (log, None) => Ok(log),
        (_, Some(fault)) => Err(fault),
    }
}

pub fn integrate_geodesic_frame_to_checked<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    frame: Mat3,
    n_steps: u32,
    max_iters: u32,
    error_budget: f32,
) -> Result<GeodesicState, GeodesicError> {
    let logarithm = geodesic_log_checked(space, from, to, n_steps, max_iters, error_budget)?;
    let mut state = integrate_geodesic_frame_checked(
        space,
        from,
        logarithm.vector,
        frame,
        n_steps,
        error_budget,
    )?;
    state.estimated_error = state.estimated_error.max(logarithm.estimated_error);
    Ok(state)
}

pub fn rk4_geodesic<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    at: Vec3,
    v: Vec3,
    n_steps: u32,
) -> (Vec3, Vec3) {
    let (state, _) = integrate_geodesic_frame(space, at, v, Mat3::IDENTITY, n_steps, None);
    (state.point, state.velocity)
}

fn transport_along_geodesic<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    v: Vec3,
) -> Vec3 {
    let tangent = gauss_newton_log(space, from, to, GEODESIC_DEFAULT_STEPS, LOG_MAX_ITERS);
    let (state, _) = integrate_geodesic_frame(
        space,
        from,
        tangent,
        Mat3::IDENTITY,
        GEODESIC_DEFAULT_STEPS,
        None,
    );
    state.frame * v
}

pub const LOG_MAX_ITERS: u32 = 12;

/// Euclidean threshold on the residual `|to − exp_from(v)|`.
pub const LOG_RESIDUAL_TOL: f32 = 1.0e-5;

const LOG_JACOBIAN_EPS: f32 = 1.0e-3;

/// Solves `exp_from(v) ≈ to`; failure returns the last finite iterate.
pub fn gauss_newton_log<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
) -> Vec3 {
    gauss_newton_log_checked(space, from, to, n_steps, max_iters).0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogError {
    Singular,
    NoConvergence,
}

/// `gauss_newton_log` with its failure reported instead of hidden; the vector is still the last finite iterate.
pub fn gauss_newton_log_checked<S: ConformallyFlat<Point = Vec3, Vector = Vec3>>(
    space: &S,
    from: Vec3,
    to: Vec3,
    n_steps: u32,
    max_iters: u32,
) -> (Vec3, Option<LogError>) {
    let (log, fault) = gauss_newton_log_core(space, from, to, n_steps, max_iters, None);
    match fault {
        None => (log.vector, None),
        Some(GeodesicError::Singular) => {
            tracing::warn!(
                "gauss_newton_log: singular Jacobian; returning best guess. \
                 `to` may be in the cut locus of `from`."
            );
            (log.vector, Some(LogError::Singular))
        }
        Some(_) => {
            tracing::warn!(
                "gauss_newton_log: did not converge in {max_iters} iters; \
                 residual remained > {LOG_RESIDUAL_TOL}. Returning best guess."
            );
            (log.vector, Some(LogError::NoConvergence))
        }
    }
}

impl<A, B, F> ConformallyFlat for BlendedSpace<A, B, F>
where
    A: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    B: Space<Point = Vec3, Vector = Vec3> + ConformallyFlat,
    F: BlendingField,
{
    fn conformal_factor(&self, p: Vec3) -> f32 {
        let alpha = self.field.weight(p);
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);

        if alpha <= 0.0 {
            return f_a;
        }
        if alpha >= 1.0 {
            return f_b;
        }
        (1.0 - alpha) * f_a + alpha * f_b
    }

    fn conformal_log_half_gradient(&self, p: Vec3) -> Vec3 {
        let alpha = self.field.weight(p);
        if alpha <= 0.0 {
            return self.a.conformal_log_half_gradient(p);
        }
        if alpha >= 1.0 {
            return self.b.conformal_log_half_gradient(p);
        }
        let f_a = self.a.conformal_factor(p);
        let f_b = self.b.conformal_factor(p);
        let f = (1.0 - alpha) * f_a + alpha * f_b;

        if !f.is_finite() || f <= 0.0 {
            return Vec3::NAN;
        }
        let grad_alpha = self.field.gradient(p);
        let grad_phi_a = self.a.conformal_log_half_gradient(p);
        let grad_phi_b = self.b.conformal_log_half_gradient(p);
        let grad_f = grad_alpha * (f_b - f_a)
            + grad_phi_a * (2.0 * (1.0 - alpha) * f_a)
            + grad_phi_b * (2.0 * alpha * f_b);
        grad_f / (2.0 * f)
    }
}

/// `weight` is 0 for pure A and 1 for pure B, and must be C¹: the integrator differentiates it.
pub trait BlendingField: Copy + Send + Sync + 'static {
    /// Implementations must clamp to `[0, 1]`.
    fn weight(&self, p: Vec3) -> f32;

    fn gradient(&self, p: Vec3) -> Vec3 {
        const EPS: f32 = 1.0e-3;
        let dx = (self.weight(p + Vec3::X * EPS) - self.weight(p - Vec3::X * EPS)) / (2.0 * EPS);
        let dy = (self.weight(p + Vec3::Y * EPS) - self.weight(p - Vec3::Y * EPS)) / (2.0 * EPS);
        let dz = (self.weight(p + Vec3::Z * EPS) - self.weight(p - Vec3::Z * EPS)) / (2.0 * EPS);
        Vec3::new(dx, dy, dz)
    }
}

// Perlin, Improving Noise, 2002.
/// C² quintic smootherstep along x.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LinearBlendX {
    start: f32,
    end: f32,
}

impl LinearBlendX {
    /// Orders reversed bounds; rejects nonfinite bounds, widths, and peak gradients.
    pub fn new(start: f32, end: f32) -> Option<Self> {
        if !start.is_finite() || !end.is_finite() {
            return None;
        }
        let (start, end) = (start.min(end), start.max(end));
        let width = end - start;
        if width <= 0.0 || !width.is_finite() || !(1.875 / width).is_finite() {
            return None;
        }
        Some(Self { start, end })
    }

    fn width(&self) -> f32 {
        self.end - self.start
    }
}

impl BlendingField for LinearBlendX {
    fn weight(&self, p: Vec3) -> f32 {
        let w = self.width();
        let t = ((p.x - self.start) / w).clamp(0.0, 1.0);
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn gradient(&self, p: Vec3) -> Vec3 {
        let w = self.width();
        let raw_t = (p.x - self.start) / w;
        if !(0.0..=1.0).contains(&raw_t) {
            return Vec3::ZERO;
        }
        let t = raw_t;
        let one_minus_t = 1.0 - t;
        let dx = 30.0 * t * t * one_minus_t * one_minus_t / w;
        Vec3::new(dx, 0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EuclideanR3, HyperbolicH3};

    fn close(a: f32, b: f32, tol: f32) {
        assert!(
            (a - b).abs() <= tol,
            "expected {a} close to {b} (tol {tol})"
        );
    }

    #[test]
    fn equal_endpoint_weights() {
        #[derive(Clone, Copy)]
        struct Bump;
        impl BlendingField for Bump {
            fn weight(&self, p: Vec3) -> f32 {
                0.1 * (1.0 - (p.x / 0.2).powi(2)).max(0.0).powi(2)
            }
            fn gradient(&self, p: Vec3) -> Vec3 {
                Vec3::X * (-10.0 * p.x * (1.0 - (p.x / 0.2).powi(2)).max(0.0))
            }
        }
        let space = BlendedSpace::new(EuclideanR3, HyperbolicH3, Bump);
        let a = Vec3::X * -0.2;
        let b = Vec3::X * 0.2;
        let length: f32 = (0..1024)
            .map(|i| {
                let x = -0.2 + (i as f32 + 0.5) * (0.4 / 1024.0);
                space.conformal_factor(Vec3::X * x).sqrt() * (0.4 / 1024.0)
            })
            .sum();
        assert!(length > 0.42);
        assert!((space.distance(a, b) - length).abs() < 1e-3);
    }

    #[test]
    fn distinct_nearby_points() {
        let space = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(50.0, 100.0).expect("a finite interval"),
        );
        let from = Vec3::ZERO;
        let to = Vec3::X * (0.1 * LOG_RESIDUAL_TOL);
        let distance = space.distance(from, to);
        assert!(distance > 0.0);
        assert!((distance - to.x).abs() < 1e-10);
    }

    fn assert_log_half_gradient_matches_finite_diff<S>(space: &S, samples: &[Vec3])
    where
        S: ConformallyFlat<Point = Vec3, Vector = Vec3>,
    {
        const EPS: f32 = 1e-3;
        for &p in samples {
            let numeric = Vec3::from_array([Vec3::X, Vec3::Y, Vec3::Z].map(|axis| {
                (space.conformal_log_half(p + axis * EPS)
                    - space.conformal_log_half(p - axis * EPS))
                    / (2.0 * EPS)
            }));
            let analytic = space.conformal_log_half_gradient(p);
            assert!(
                (analytic - numeric).abs().max_element() <= 5e-3,
                "gradient {analytic:?} misses the finite difference {numeric:?} at {p:?}"
            );
        }
    }

    #[test]
    fn log_half_gradient() {
        assert_log_half_gradient_matches_finite_diff(
            &HyperbolicH3,
            &[
                Vec3::new(0.1, 0.0, 0.0),
                Vec3::new(0.3, -0.2, 0.1),
                Vec3::new(-0.4, 0.5, 0.2),
            ],
        );
        assert_log_half_gradient_matches_finite_diff(
            &BlendedSpace::new(
                EuclideanR3,
                HyperbolicH3,
                LinearBlendX::new(-0.5, 0.5).expect("a finite interval"),
            ),
            &[
                Vec3::new(-0.7, 0.05, 0.0),
                Vec3::new(0.0, 0.1, -0.1),
                Vec3::new(0.6, 0.05, 0.0),
            ],
        );
    }

    #[test]
    fn checked_geodesic() {
        let at = Vec3::new(0.25, -0.1, 0.05);
        let velocity = Vec3::new(0.2, 0.15, -0.04);
        let frame = Mat3::from_cols(velocity, Vec3::Y, Vec3::Z);
        let state = integrate_geodesic_frame_checked(
            &HyperbolicH3,
            at,
            velocity,
            frame,
            GEODESIC_DEFAULT_STEPS,
            GEODESIC_ERROR_BUDGET,
        )
        .expect("checked integration");

        close((state.frame.x_axis - state.velocity).length(), 0.0, 1e-6);
        assert!((state.point - at - velocity).length() > 1e-4);

        assert_eq!(
            integrate_geodesic_frame_checked(
                &EuclideanR3,
                Vec3::ZERO,
                Vec3::splat(1.0e20),
                Mat3::IDENTITY,
                GEODESIC_DEFAULT_STEPS,
                GEODESIC_ERROR_BUDGET,
            ),
            Err(GeodesicError::NonFinite)
        );
        assert_eq!(
            integrate_geodesic_frame_checked(
                &HyperbolicH3,
                Vec3::X * 0.9,
                Vec3::X,
                Mat3::IDENTITY,
                GEODESIC_DEFAULT_STEPS,
                GEODESIC_ERROR_BUDGET,
            ),
            Err(GeodesicError::ChartBoundary)
        );
    }

    #[test]
    fn outside_blend_zone() {
        let pure_a = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(50.0, 100.0).expect("a finite interval"),
        );
        let p = Vec3::new(1.0, 2.0, 3.0);
        let q = Vec3::new(4.0, 5.0, 6.0);
        let v = Vec3::new(0.5, -0.3, 0.7);

        close(
            (pure_a.exp(p, v) - EuclideanR3.exp(p, v)).length(),
            0.0,
            1e-5,
        );
        close(
            (pure_a.log(p, q) - EuclideanR3.log(p, q)).length(),
            0.0,
            1e-3,
        );
        close(pure_a.distance(p, q), EuclideanR3.distance(p, q), 1e-3);
        close(
            (pure_a.parallel_transport(p, q, v) - EuclideanR3.parallel_transport(p, q, v)).length(),
            0.0,
            1e-5,
        );
        close(
            pure_a.conformal_factor(p),
            EuclideanR3.conformal_factor(p),
            1e-6,
        );

        let pure_b = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(-100.0, -50.0).expect("a finite interval"),
        );
        let p = Vec3::new(0.1, 0.0, 0.0);
        let q = Vec3::new(0.2, 0.1, 0.0);
        let v = Vec3::new(0.05, 0.0, 0.0);

        close(
            (pure_b.exp(p, v) - HyperbolicH3.exp(p, v)).length(),
            0.0,
            5e-3,
        );
        close(
            (pure_b.log(p, q) - HyperbolicH3.log(p, q)).length(),
            0.0,
            5e-3,
        );
        close(pure_b.distance(p, q), HyperbolicH3.distance(p, q), 5e-3);
        close(
            pure_b.conformal_factor(p),
            HyperbolicH3.conformal_factor(p),
            1e-3,
        );

        let past_the_zone = BlendedSpace::new(
            EuclideanR3,
            HyperbolicH3,
            LinearBlendX::new(-0.15, 0.15).expect("a finite interval"),
        );
        let v = Vec3::new(0.05, -0.03, 0.04);

        let mut worst = 0.0_f32;
        for (a, b) in [
            (Vec3::new(0.3, 0.05, 0.0), Vec3::new(0.45, -0.05, 0.1)),
            (Vec3::new(0.25, 0.0, 0.0), Vec3::new(0.4, 0.1, -0.05)),
            (Vec3::new(0.5, 0.1, 0.05), Vec3::new(0.35, 0.15, 0.0)),
        ] {
            let closed = HyperbolicH3.parallel_transport(a, b, v);
            let integrated = past_the_zone.parallel_transport(a, b, v);
            worst = worst.max((integrated - closed).length());
        }

        assert!(
            worst <= 1.0e-4,
            "the blended transport left the H3 closed form by {worst}"
        );
    }
}
