//! Drone3D Benchmark — MI system learns to hover and navigate a 3D quadrotor.
//!
//! X-configuration quadrotor (4 rotors). Full rigid-body dynamics in world frame.
//! No small-angle approximation — rotation matrix used for thrust projection.
//!
//! State (12D):  x/y/z errors, vx/vy/vz, phi/theta/psi, omega_x/y/z
//! Input:        96 channels  (12 vars × 8 Gaussian tiles)
//! Actions:      9 discrete   (HOVER | ASCEND | DESCEND | FWD | BWD | RGT | LFT | ASC+RL | ASC+RR)
//! Directional actions preserve hover-level total thrust — attitude is the control knob.
//!
//! Profiles:
//!   quick    (200 eps) — hover at (0,0,2), no wind
//!   standard (1000 eps) — altitude waypoints, no wind
//!   extended (3000 eps) — full 3D waypoints + OU wind in x and y
//!
//! Run: cargo run --example drone --release
//! Run: cargo run --example drone --release -- --standard
//! Run: cargo run --example drone --release -- --extended

use morphon_core::developmental::DevelopmentalConfig;
use morphon_core::endoquilibrium::EndoConfig;
use morphon_core::homeostasis::{CompetitionMode, HomeostasisParams};
use morphon_core::learning::LearningParams;
use morphon_core::morphogenesis::MorphogenesisParams;
use morphon_core::morphon::MetabolicConfig;
use morphon_core::scheduler::SchedulerConfig;
use morphon_core::system::{System, SystemConfig};
use morphon_core::types::LifecycleConfig;
use rand::Rng;
use serde_json::json;
use std::fs;

// ─── Physics ──────────────────────────────────────────────────────────────────

const GRAVITY:    f64 = 9.81;
const DRONE_MASS: f64 = 0.5;    // kg
const ARM:        f64 = 0.15;   // m, rotor-to-CoM distance
const I_XX:       f64 = 0.010;  // kg·m², roll inertia
const I_YY:       f64 = 0.010;  // pitch inertia
const I_ZZ:       f64 = 0.020;  // yaw inertia (slightly higher)
const DRAG:       f64 = 0.025;  // yaw drag coefficient (reaction torque per thrust unit)
// At frac=0.5 on all 4 rotors: total = 4×0.5×MAX_T = 2×MAX_T = m×g → MAX_T = m×g/2
const MAX_T:      f64 = DRONE_MASS * GRAVITY / 2.0; // N per rotor at fraction 1.0
const DT:         f64 = 0.02;   // 50 Hz

// ─── Bounds ───────────────────────────────────────────────────────────────────

const X_LIM:     f64 = 3.0;
const Y_LIM:     f64 = 3.0;
const Z_MIN:     f64 = 0.1;   // ground clearance
const Z_MAX:     f64 = 6.0;
const ANGLE_LIM: f64 = std::f64::consts::FRAC_PI_4; // 45° max tilt
const V_MAX:     f64 = 4.0;   // m/s normalization
const OMEGA_MAX: f64 = 5.0;   // rad/s normalization

// ─── Actions ──────────────────────────────────────────────────────────────────
//
// Rotor layout (top view, X-configuration):
//   R1(fl,CCW)  R2(fr,CW)
//        \      /
//         [body]
//        /      \
//   R4(bl,CW)  R3(br,CCW)
//
// Roll  torque = (R1+R4 − R2−R3) × ARM  [left − right rotors]
// Pitch torque = (R1+R2 − R3−R4) × ARM  [front − back rotors]
// Yaw   torque = (R1+R3 − R2−R4) × DRAG [CCW − CW rotors, reaction]
//
// Convention: phi>0 = roll right (+y accel), theta>0 = nose up (−x accel)
//
// Directional actions keep total thrust = 2×MAX_T = m×g (hover-neutral altitude).
// Only ASCEND and DESCEND deviate from this.
const ACTIONS: [(f64, f64, f64, f64); 9] = [
    //               R1(fl) R2(fr) R3(br) R4(bl)
    (0.50, 0.50, 0.50, 0.50), // 0 HOVER    — zero torque, hover thrust
    (0.70, 0.70, 0.70, 0.70), // 1 ASCEND   — all high, +z
    (0.30, 0.30, 0.30, 0.30), // 2 DESCEND  — all low, -z
    (0.40, 0.40, 0.60, 0.60), // 3 FWD (+x) — back high → τ_pitch<0 → nose down → +x
    (0.60, 0.60, 0.40, 0.40), // 4 BWD (-x) — front high → τ_pitch>0 → nose up → -x
    (0.60, 0.40, 0.40, 0.60), // 5 RGT (+y) — left high → τ_roll>0 → roll right → +y
    (0.40, 0.60, 0.60, 0.40), // 6 LFT (-y) — right high → τ_roll<0 → roll left → -y
    (0.55, 0.65, 0.65, 0.55), // 7 ASC+RL   — ascend + left-roll (corrects phi>0 while climbing)
    (0.65, 0.55, 0.55, 0.65), // 8 ASC+RR   — ascend + right-roll (corrects phi<0 while climbing)
];
const ACTION_NAMES: [&str; 9] = ["HOVER  ", "ASCEND ", "DESCEND", "FWD+X  ", "BWD-X  ", "RGT+Y  ", "LFT-Y  ", "ASC+RL ", "ASC+RR "];

// ─── Waypoints ────────────────────────────────────────────────────────────────

// Quick: single hover point
const HOVER_POINT: (f64, f64, f64) = (0.0, 0.0, 2.0);

// Standard: altitude-only waypoints (x=y=0)
const ALT_WAYPOINTS: &[(f64, f64, f64)] = &[
    (0.0, 0.0, 2.0),
    (0.0, 0.0, 1.3),
    (0.0, 0.0, 3.5),
    (0.0, 0.0, 1.8),
    (0.0, 0.0, 2.8),
];

// Extended: full 3D waypoints
const WAYPOINTS_3D: &[(f64, f64, f64)] = &[
    ( 0.0,  0.0, 2.0),
    ( 1.5,  0.0, 2.5),
    ( 0.0,  1.5, 1.5),
    (-1.5,  0.0, 2.0),
    ( 0.0, -1.5, 3.0),
    ( 1.2,  1.2, 2.0),
];

const WAYPOINT_STEPS: usize = 100; // max steps before advancing waypoint
const WAYPOINT_TOL:   f64   = 0.30; // m, proximity threshold

const INTERNAL_STEPS: usize = 4;
const GAMMA:          f64   = 0.97;

// ─── Drone ────────────────────────────────────────────────────────────────────

struct Drone3D {
    x: f64, y: f64, z: f64,
    vx: f64, vy: f64, vz: f64,
    phi: f64, theta: f64, psi: f64,   // roll, pitch, yaw
    omx: f64, omy: f64, omz: f64,     // angular rates
    t: f64,
}

impl Drone3D {
    fn reset(&mut self, tgt: (f64, f64, f64), rng: &mut impl Rng) {
        let (tx, ty, tz) = tgt;
        self.x     = tx + rng.random_range(-0.3..0.3);
        self.y     = ty + rng.random_range(-0.3..0.3);
        // Randomize starting altitude across the full flight envelope so the policy
        // encounters both low-altitude ascents and high-altitude descents consistently.
        self.z     = (tz + rng.random_range(-1.5..1.5)).clamp(0.5, 5.5);
        self.vx    = rng.random_range(-0.2..0.2);
        self.vy    = rng.random_range(-0.2..0.2);
        self.vz    = rng.random_range(-0.1..0.1);
        self.phi   = rng.random_range(-0.08..0.08);
        self.theta = rng.random_range(-0.08..0.08);
        self.psi   = rng.random_range(-0.1..0.1);
        self.omx   = 0.0; self.omy = 0.0; self.omz = 0.0;
        self.t     = 0.0;
    }

    /// 12 state vars × 8 Gaussian tiles = 96 population-coded channels.
    /// All vars encoded relative to current target so the same policy generalises
    /// across waypoints.
    fn observe(&self, tgt: (f64, f64, f64)) -> Vec<f64> {
        let (tx, ty, tz) = tgt;
        let raw: [f64; 12] = [
            ((self.x - tx) / X_LIM).clamp(-1.5, 1.5),
            ((self.y - ty) / Y_LIM).clamp(-1.5, 1.5),
            ((self.z - tz) / Z_MAX).clamp(-1.5, 1.5),
            (self.vx / V_MAX).clamp(-1.5, 1.5),
            (self.vy / V_MAX).clamp(-1.5, 1.5),
            (self.vz / V_MAX).clamp(-1.5, 1.5),
            (self.phi   / ANGLE_LIM).clamp(-1.5, 1.5),
            (self.theta / ANGLE_LIM).clamp(-1.5, 1.5),
            (self.psi   / std::f64::consts::PI).clamp(-1.5, 1.5),
            (self.omx / OMEGA_MAX).clamp(-1.5, 1.5),
            (self.omy / OMEGA_MAX).clamp(-1.5, 1.5),
            (self.omz / OMEGA_MAX).clamp(-1.5, 1.5),
        ];
        let centers: [f64; 8] = [-0.85, -0.60, -0.35, -0.10, 0.10, 0.35, 0.60, 0.85];
        let width = 0.30;
        let amp   = 4.0;
        let mut out = Vec::with_capacity(96);
        for &v in &raw {
            for &c in &centers {
                out.push((-(v - c).powi(2) / (2.0 * width * width)).exp() * amp);
            }
        }
        out
    }

    /// Step physics. Applies full rotation-matrix thrust projection.
    /// `gx`, `gy`: horizontal acceleration disturbances (m/s²).
    /// Returns true while within safe operating bounds.
    fn step(&mut self, action: usize, gx: f64, gy: f64) -> bool {
        let (r1, r2, r3, r4) = ACTIONS[action];
        let total_f = (r1 + r2 + r3 + r4) * MAX_T;

        // Rotation matrix body→world (ZYX Euler: psi → theta → phi)
        let (sp, cp) = (self.phi.sin(),   self.phi.cos());
        let (st, ct) = (self.theta.sin(), self.theta.cos());
        let (sy, cy) = (self.psi.sin(),   self.psi.cos());

        // Body z-axis (thrust direction) in world frame:
        // e_z = R * [0,0,1]^T
        let thrust_x =  (cy * st * cp + sy * sp) * total_f / DRONE_MASS + gx;
        let thrust_y =  (sy * st * cp - cy * sp) * total_f / DRONE_MASS + gy;
        let thrust_z =  (ct * cp)                * total_f / DRONE_MASS - GRAVITY;

        // Torques in body frame
        let tau_roll  = (r1 + r4 - r2 - r3) * ARM  * MAX_T;
        let tau_pitch = (r1 + r2 - r3 - r4) * ARM  * MAX_T;
        let tau_yaw   = (r1 + r3 - r2 - r4) * DRAG * MAX_T;

        // Euler integration
        self.vx  += DT * thrust_x;
        self.vy  += DT * thrust_y;
        self.vz  += DT * thrust_z;
        self.x   += DT * self.vx;
        self.y   += DT * self.vy;
        self.z   += DT * self.vz;

        self.omx += DT * tau_roll  / I_XX;
        self.omy += DT * tau_pitch / I_YY;
        self.omz += DT * tau_yaw   / I_ZZ;
        self.phi   += DT * self.omx;
        self.theta += DT * self.omy;
        self.psi   += DT * self.omz;
        self.t     += DT;

        self.z > Z_MIN && self.z < Z_MAX
            && self.x.abs() < X_LIM
            && self.y.abs() < Y_LIM
            && self.phi.abs()   < ANGLE_LIM
            && self.theta.abs() < ANGLE_LIM
    }
}

// ─── Wind (Ornstein-Uhlenbeck) ────────────────────────────────────────────────

struct Wind { vx: f64, vy: f64, theta: f64, sigma: f64 }

impl Wind {
    fn new(sigma: f64) -> Self { Self { vx: 0.0, vy: 0.0, theta: 0.20, sigma } }

    fn step(&mut self, rng: &mut impl Rng) -> (f64, f64) {
        let nx = normal_sample(rng);
        let ny = normal_sample(rng);
        let sq = DT.sqrt();
        self.vx += -self.theta * self.vx * DT + self.sigma * nx * sq;
        self.vy += -self.theta * self.vy * DT + self.sigma * ny * sq;
        self.vx = self.vx.clamp(-4.0, 4.0);
        self.vy = self.vy.clamp(-4.0, 4.0);
        (self.vx, self.vy)
    }

    fn display(&self) -> String {
        fn arrow(v: f64) -> &'static str {
            match (v * 2.0) as i32 {
                i32::MIN..=-3 => "<<<", -2 => "<< ", -1 => "<  ",
                0             => " - ",  1 => "  >",  2 => " >>",
                3..=i32::MAX  => ">>>",
            }
        }
        format!("x:{} y:{}", arrow(self.vx), arrow(self.vy))
    }
}

fn normal_sample(rng: &mut impl Rng) -> f64 {
    let u1 = rng.random_range(1e-10_f64..1.0);
    let u2 = rng.random_range(0.0_f64..1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

// ─── Heuristic supervisor ─────────────────────────────────────────────────────
//
// Priority: attitude → altitude → x-error → y-error → hover.
// Tilt is always corrected first because an unstable drone can't navigate.
//
// Composite actions 7/8 (ASC+RL / ASC+RR) are used when the drone is
// significantly below target AND rolling off-axis — they ascend while
// simultaneously applying corrective roll torque, avoiding the altitude-bleed
// trap where pure tilt correction stalls the climb.
fn correct_action(d: &Drone3D, tgt: (f64, f64, f64)) -> usize {
    let (tx, ty, tz) = tgt;
    let tilt = d.phi.abs().max(d.theta.abs());
    let below = (tz - d.z) > 0.20; // meaningfully below target altitude

    if tilt > 0.10 {
        if d.phi.abs() > d.theta.abs() {
            // Roll correction: use composite ascend+roll when also below target
            if d.phi > 0.0 {
                if below { 7 } else { 6 } // ASC+RL or pure LFT
            } else {
                if below { 8 } else { 5 } // ASC+RR or pure RGT
            }
        } else {
            if d.theta > 0.0 { 3 } else { 4 } // FWD reduces theta, BWD raises
        }
    } else {
        let proj_z = (tz - d.z) + d.vz * 0.40;
        let proj_x = (tx - d.x) + d.vx * 0.40;
        let proj_y = (ty - d.y) + d.vy * 0.40;
        let az = proj_z.abs();
        let ax = proj_x.abs() * 0.65; // altitude takes priority over horizontal
        let ay = proj_y.abs() * 0.65;

        if az >= ax.max(ay) && az > 0.22 {
            if proj_z > 0.0 { 1 } else { 2 }
        } else if ax >= ay && ax > 0.20 {
            if proj_x > 0.0 { 3 } else { 4 }
        } else if ay > 0.20 {
            if proj_y > 0.0 { 5 } else { 6 }
        } else {
            0 // hover
        }
    }
}

// ─── Critic ───────────────────────────────────────────────────────────────────

struct Critic { weights: [f64; 24], bias: f64, lr: f64 }

impl Critic {
    fn new() -> Self { Self { weights: [0.0; 24], bias: 0.0, lr: 0.06 } }

    fn features(d: &Drone3D, tgt: (f64, f64, f64)) -> [f64; 24] {
        let (tx, ty, tz) = tgt;
        let s = [
            (d.x - tx) / X_LIM, (d.y - ty) / Y_LIM, (d.z - tz) / Z_MAX,
            d.vx / V_MAX, d.vy / V_MAX, d.vz / V_MAX,
            d.phi / ANGLE_LIM, d.theta / ANGLE_LIM,
            d.omx / OMEGA_MAX, d.omy / OMEGA_MAX, d.omz / OMEGA_MAX,
            // distance to target (single composite feature)
            ((d.x-tx).powi(2) + (d.y-ty).powi(2) + (d.z-tz).powi(2)).sqrt() / 5.0,
        ];
        let mut f = [0.0f64; 24];
        for i in 0..12 { f[i] = s[i]; }
        for i in 0..12 { f[12 + i] = s[i] * s[i]; }
        f
    }

    fn predict(&self, d: &Drone3D, tgt: (f64, f64, f64)) -> f64 {
        Self::features(d, tgt).iter()
            .zip(self.weights.iter())
            .map(|(f, w)| f * w)
            .sum::<f64>() + self.bias
    }

    fn update(&mut self, d: &Drone3D, tgt: (f64,f64,f64), reward: f64,
              d2: &Drone3D, tgt2: (f64,f64,f64), done: bool) -> f64 {
        let v  = self.predict(d, tgt);
        let vn = if done { 0.0 } else { self.predict(d2, tgt2) };
        let td = reward + GAMMA * vn - v;
        let f  = Self::features(d, tgt);
        for i in 0..24 {
            self.weights[i] = (self.weights[i] + self.lr * td * f[i]).clamp(-10.0, 10.0);
        }
        self.bias = (self.bias + self.lr * td).clamp(-10.0, 10.0);
        td
    }
}

// ─── Visualization ────────────────────────────────────────────────────────────

struct TrajPoint { x: f64, y: f64, z: f64, phi: f64, theta: f64, action: usize }

/// Two side-by-side projections: xz (side view) and xy (top-down view).
fn visualize(traj: &[TrajPoint], tgt: (f64,f64,f64), ep: usize, wp_idx: usize, wp_list: &[(f64,f64,f64)], wind: &Wind) {
    const W: usize = 30; // columns per panel
    const H: usize = 13; // rows (z: 0..6 in 0.5m steps)

    let (tx, ty, tz) = tgt;

    let x_to_col = |x: f64| -> usize {
        ((x + X_LIM) / (2.0 * X_LIM) * (W - 1) as f64)
            .round().clamp(0.0, (W - 1) as f64) as usize
    };
    let y_to_col = |y: f64| -> usize {
        ((y + Y_LIM) / (2.0 * Y_LIM) * (W - 1) as f64)
            .round().clamp(0.0, (W - 1) as f64) as usize
    };
    let z_to_row = |z: f64| -> usize {
        ((Z_MAX - z) / Z_MAX * (H - 1) as f64)
            .round().clamp(0.0, (H - 1) as f64) as usize
    };

    let mut xz = vec![vec![b' '; W]; H]; // side view
    let mut xy = vec![vec![b' '; W]; H]; // top view

    // Target lines
    let tz_row = z_to_row(tz);
    for c in 0..W { xz[tz_row][c] = b'-'; }
    let tx_col = x_to_col(tx);
    let ty_col = y_to_col(ty);
    for r in 0..H { xy[r][tx_col] = b'|'; }
    // target crosshair on top view
    if tz_row < H { xy[tz_row][ty_col] = b'*'; }
    // target x on side view
    xz[tz_row][tx_col] = b'*';

    // Trajectory
    let n = traj.len();
    for (i, p) in traj.iter().enumerate() {
        let age = n - 1 - i;
        let ch = if age == 0 {
            match p.phi.abs().max(p.theta.abs()) {
                t if t > 0.25 => {
                    if p.phi.abs() > p.theta.abs() {
                        if p.phi > 0.0 { b'>' } else { b'<' }
                    } else {
                        if p.theta > 0.0 { b'^' } else { b'v' }
                    }
                }
                _ => b'O',
            }
        } else if age < 15 { b'.' } else { b'\'' };

        let xz_r = z_to_row(p.z);
        let xz_c = x_to_col(p.x);
        if age > 0 || xz[xz_r][xz_c] == b' ' || xz[xz_r][xz_c] == b'-' {
            xz[xz_r][xz_c] = ch;
        }

        let xy_r = y_to_col(p.y); // repurpose as row in top view (y vertical)
        let xy_c = x_to_col(p.x);
        if xy_r < H && (age > 0 || xy[xy_r][xy_c] == b' ' || xy[xy_r][xy_c] == b'|') {
            xy[xy_r][xy_c] = ch;
        }
    }

    // Header
    println!("  +-- ep {:>4}  wp {}/{} ({:.1},{:.1},{:.1}m) {:->14}+",
        ep, wp_idx + 1, wp_list.len(), tx, ty, tz, "");

    // Side-by-side grid
    let z_labels = [6.0_f64, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0];
    let y_labels = [3.0_f64, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0];
    println!("  | {:^30} | {:^30} |", "SIDE VIEW (xz)", "TOP VIEW (xy)");
    for i in 0..H {
        let zl = if i < z_labels.len() { z_labels[i] } else { 0.0 };
        let yl = if i < y_labels.len() { y_labels[i] } else { 0.0 };
        let xz_line: String = xz[i].iter().map(|&b| b as char).collect();
        let xy_line: String = xy[i].iter().map(|&b| b as char).collect();
        let wp_mark = if (zl - tz).abs() < 0.26 { "<z" } else { "  " };
        println!("  |z{:4.1}|{}|{}  y{:5.1}|{}|",
            zl, xz_line, wp_mark, yl, xy_line);
    }

    // Ground line
    let ground: String = std::iter::repeat('=').take(W).collect();
    println!("  |     |{}|    |     |{}|", ground, ground);
    println!("  | x: {:^6}  ..  {:^6} |    | x: {:^6}  ..  {:^6} |",
        "-3.0", "+3.0", "-3.0", "+3.0");

    // State summary
    if let Some(p) = traj.last() {
        println!("  pos=({:+.2},{:+.2},{:.2})  phi={:+.1}° theta={:+.1}°  action={}  wind [{}]",
            p.x, p.y, p.z,
            p.phi.to_degrees(), p.theta.to_degrees(),
            ACTION_NAMES[p.action], wind.display());
    }
    println!();
}

// ─── System ───────────────────────────────────────────────────────────────────

fn create_system() -> System {
    let config = SystemConfig {
        developmental: DevelopmentalConfig {
            seed_size: 100,
            dimensions: 4,
            initial_connectivity: 0.20,
            proliferation_rounds: 2,
            target_input_size:  Some(96),
            target_output_size: Some(9),
            ..DevelopmentalConfig::cerebellar()
        },
        scheduler: SchedulerConfig {
            medium_period: 1, slow_period: 10,
            glacial_period: 100, homeostasis_period: 10, memory_period: 25,
        },
        learning: LearningParams {
            tau_eligibility:  6.0,
            tau_trace:        10.0,
            a_plus:           1.0,
            a_minus:         -0.5,
            tau_tag:         700.0,
            tag_threshold:    0.3,
            capture_threshold: 0.7,
            capture_rate:     0.2,
            weight_max:       3.0, weight_min: 0.01,
            alpha_reward:     2.0, alpha_novelty: 0.5,
            alpha_arousal:    0.5, alpha_homeostasis: 0.1,
            transmitter_potentiation:  0.002,
            heterosynaptic_depression: 0.003,
            tag_accumulation_rate:     0.3,
            ..Default::default()
        },
        morphogenesis: MorphogenesisParams {
            migration_rate: 0.08, max_morphons: Some(500),
            division_threshold: 1.0, fusion_min_size: 2, apoptosis_min_age: 500,
            ..Default::default()
        },
        homeostasis: HomeostasisParams {
            migration_cooldown_duration: 5.0,
            competition_mode: CompetitionMode::default(),
            ..Default::default()
        },
        lifecycle: LifecycleConfig {
            division: false, differentiation: true, fusion: false,
            apoptosis: false, migration: false, synaptogenesis: true,
        },
        metabolic:      MetabolicConfig::default(),
        endoquilibrium: EndoConfig { enabled: true, ..Default::default() },
        dt: 1.0,
        working_memory_capacity: 7,
        episodic_memory_capacity: 200,
        dream: morphon_core::types::DreamConfig::default(),
        ..Default::default()
    };

    let mut sys = System::new(config);
    sys.limbic.enabled = true;
    sys.enable_analog_readout();
    sys.set_consolidation_gate(60.0);
    let sensory_ids: std::collections::HashSet<u64> = sys.morphons.values()
        .filter(|m| m.cell_type == morphon_core::CellType::Sensory)
        .map(|m| m.id)
        .collect();
    sys.filter_readout_weights(|id| sensory_ids.contains(&id));
    sys
}

// ─── Episode ──────────────────────────────────────────────────────────────────

fn select_action(outputs: &[f64], epsilon: f64, rng: &mut impl Rng) -> usize {
    if outputs.is_empty() || rng.random_range(0.0..1.0) < epsilon {
        return rng.random_range(0..9usize);
    }
    outputs.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn run_episode(
    system:   &mut System,
    drone:    &mut Drone3D,
    critic:   &mut Critic,
    wind:     &mut Wind,
    wps:      &[(f64,f64,f64)],
    max_steps: usize,
    epsilon:   f64,
    gusts:     bool,
    rng:       &mut impl Rng,
) -> (usize, f64, usize, Vec<TrajPoint>, &'static str, f64) {
    let mut wp_idx    = 0usize;
    let mut wp_steps  = 0usize;
    let mut wp_near   = 0usize;
    let mut wps_done  = 0usize;
    let mut tgt       = wps[0];

    drone.reset(tgt, rng);
    system.reset_voltages();
    wind.vx = 0.0; wind.vy = 0.0;

    let mut steps       = 0usize;
    let mut pos_acc     = 0.0f64;
    let mut near_steps  = 0usize; // steps within WAYPOINT_TOL of current target
    let mut crash_cause = "survived";
    let mut traj        = Vec::new();

    for _ in 0..max_steps {
        let (gx, gy) = if gusts { wind.step(rng) } else { (0.0, 0.0) };

        let obs = drone.observe(tgt);
        let pre = Drone3D {
            x: drone.x, y: drone.y, z: drone.z,
            vx: drone.vx, vy: drone.vy, vz: drone.vz,
            phi: drone.phi, theta: drone.theta, psi: drone.psi,
            omx: drone.omx, omy: drone.omy, omz: drone.omz,
            t: drone.t,
        };

        // Limbic: salience eval + arousal injection on first cortical step,
        // then remaining INTERNAL_STEPS-1 steps for temporal integration.
        let mut outputs = system.process_with_limbic(&obs, None);
        if INTERNAL_STEPS > 1 {
            outputs = system.process_steps(&obs, INTERNAL_STEPS - 1);
        }
        let action  = select_action(&outputs, epsilon, rng);
        let alive   = drone.step(action, gx, gy);
        steps += 1;

        let (ex, ey, ez) = (drone.x - tgt.0, drone.y - tgt.1, drone.z - tgt.2);
        let pos_err = (ex*ex + ey*ey + ez*ez).sqrt();
        pos_acc += pos_err;
        if pos_err < WAYPOINT_TOL { near_steps += 1; }

        traj.push(TrajPoint {
            x: drone.x, y: drone.y, z: drone.z,
            phi: drone.phi, theta: drone.theta, action,
        });

        // Waypoint advance
        wp_steps += 1;
        if pos_err < WAYPOINT_TOL { wp_near += 1; } else { wp_near = 0; }
        if (wp_near >= 15 || wp_steps >= WAYPOINT_STEPS) && wp_idx + 1 < wps.len() {
            if wp_near >= 15 { wps_done += 1; }
            wp_idx += 1; tgt = wps[wp_idx]; wp_steps = 0; wp_near = 0;
        }

        // Reward: 3D position error + attitude stability + velocity damping
        let pos_n  = (pos_err / 4.0).min(1.0);
        let att_n  = (drone.phi.abs() + drone.theta.abs()) / (2.0 * ANGLE_LIM);
        let vel_n  = ((drone.vx*drone.vx + drone.vy*drone.vy + drone.vz*drone.vz).sqrt() / V_MAX).min(1.0);
        let reward = if alive { 1.0 - 0.45*pos_n - 0.35*att_n.min(1.0) - 0.20*vel_n }
                     else { -1.0 };

        let _int_td = system.inject_td_error(reward, GAMMA);
        let td_err  = critic.update(&pre, tgt, reward, drone, tgt, !alive);
        let correct = correct_action(&pre, tgt);
        let base_lr = 0.05;

        match system.readout_training_mode() {
            morphon_core::types::ReadoutTrainingMode::Supervised => {
                system.train_readout(correct, base_lr);
                system.reward_contrastive(correct, 0.2, 0.1);
            }
            morphon_core::types::ReadoutTrainingMode::TDOnly => {
                if td_err > 0.0 {
                    system.train_readout(action, td_err.min(1.0) * base_lr);
                    system.reward_contrastive(action, td_err.min(1.0) * 0.2, 0.1);
                } else {
                    system.train_readout(correct, td_err.abs().min(1.0) * base_lr * 0.5);
                }
            }
            _ => unreachable!(),
        }
        // Limbic RPE: deliver reward against per-action expectation.
        // Uses correct action as the class label so motivational drive tracks
        // which actions are rewarding vs penalising over time.
        system.deliver_limbic_reward(&obs, correct, reward);

        let tilt = (drone.phi.abs() + drone.theta.abs()) / (2.0 * ANGLE_LIM);
        if tilt > 0.5 { system.inject_novelty((tilt - 0.5) * 2.0); }

        if !alive {
            // Classify crash cause for post-episode diagnostics
            crash_cause = if drone.z <= Z_MIN { "floor" }
                else if drone.z >= Z_MAX { "ceiling" }
                else if drone.x.abs() >= X_LIM || drone.y.abs() >= Y_LIM { "oob" }
                else { "tilt" };
            system.inject_arousal(0.8);
            system.inject_novelty(0.5);
            break;
        }
    }
    let near_pct = near_steps as f64 / steps.max(1) as f64;
    (steps, pos_acc / steps.max(1) as f64, wps_done, traj, crash_cause, near_pct)
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn parse_profile() -> &'static str {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--extended") { "extended" }
    else if args.iter().any(|a| a == "--standard") { "standard" }
    else { "quick" }
}

fn main() {
    let profile = parse_profile();
    let (num_eps, max_steps, gusts, wps): (usize, usize, bool, &[(f64,f64,f64)]) = match profile {
        "extended" => (3000, 500, true,  WAYPOINTS_3D),
        "standard" => (1000, 300, false, ALT_WAYPOINTS),
        _          => (200,  200, false, &[HOVER_POINT]),
    };

    println!("=== MORPHON Drone3D Benchmark [{}] ===\n", profile);
    println!("Waypoints: {:?}", wps);
    if gusts { println!("Wind: OU turbulence (σ=1.6, θ=0.20) in x and y\n"); }

    let mut system = create_system();
    let mut drone  = Drone3D {
        x: 0.0, y: 0.0, z: 2.0, vx: 0.0, vy: 0.0, vz: 0.0,
        phi: 0.01, theta: 0.0, psi: 0.0,
        omx: 0.0, omy: 0.0, omz: 0.0, t: 0.0,
    };
    let mut critic = Critic::new();
    let mut wind   = Wind::new(1.6);
    let mut rng    = rand::rng();

    let st = system.inspect();
    println!("Network:  {} morphons | {} synapses | {} in | {} out",
        st.total_morphons, st.total_synapses, system.input_size(), system.output_size());
    println!("Physics:  mass={:.1} kg | arm={:.2} m | hover={:.3} N/rotor × 4",
        DRONE_MASS, ARM, MAX_T * 0.5);
    println!("Actions:  [0=HOVER | 1=ASCEND | 2=DESCEND | 3=FWD | 4=BWD | 5=RGT | 6=LFT | 7=ASC+RL | 8=ASC+RR]");
    println!("Bounds:   xyz∈[±{:.0}/±{:.0}/{:.0}..{:.0}] tilt∈[±{:.0}°]\n",
        X_LIM, Y_LIM, Z_MIN, Z_MAX, ANGLE_LIM.to_degrees());

    // Canonical probes: 4 unambiguous test cases
    // - at target, level, still → HOVER (0)
    // - below target             → ASCEND (1)
    // - theta > 0 (nose up)      → FWD (3) to correct
    // - phi > 0 at target height → LFT-Y (6) to correct (tests y-axis blind spot)
    let probe_expected = [0usize, 1, 3, 6];
    let probe_states: &[(f64,f64,f64,f64,f64)] = &[
        (0.0, 0.0, wps[0].2,       0.0,  0.0),
        (0.0, 0.0, wps[0].2 - 0.5, 0.0,  0.0),
        (0.0, 0.0, wps[0].2,       0.0,  0.18),
        (0.0, 0.0, wps[0].2,       0.15, 0.0),  // phi>0 at target z → LFT
    ];

    // Warm-up
    let warmup = drone.observe(wps[0]);
    for _ in 0..20 { system.process_steps(&warmup, INTERNAL_STEPS); }

    let mut best_steps  = 0usize;
    let mut total_wps   = 0usize;
    let mut recent_steps: Vec<usize> = Vec::new();
    let mut recent_errs:  Vec<f64>   = Vec::new();
    let mut all_steps:    Vec<usize> = Vec::new();
    let mut all_errs:     Vec<f64>   = Vec::new();
    let mut last_traj:    Vec<TrajPoint> = Vec::new();
    let mut last_wp_idx   = 0usize;
    let print_every = if profile == "quick" { 50 } else { 100 };

    for ep in 0..num_eps {
        let epsilon = (0.5 * (1.0 - ep as f64 / num_eps as f64)).max(0.05);
        let (steps, avg_err, n_wps, traj, crash_cause, near_pct) = run_episode(
            &mut system, &mut drone, &mut critic, &mut wind,
            wps, max_steps, epsilon, gusts, &mut rng,
        );

        recent_steps.push(steps);
        recent_errs.push(avg_err);
        all_steps.push(steps);
        all_errs.push(avg_err);
        if recent_steps.len() > 100 { recent_steps.remove(0); }
        if recent_errs.len()  > 100 { recent_errs.remove(0); }
        best_steps = best_steps.max(steps);
        total_wps += n_wps;

        // track final waypoint index for viz
        if !traj.is_empty() {
            let mut wi = 0usize; let mut ws = 0usize; let mut wn = 0usize;
            let init_tgt = wps[0];
            let mut cur_tgt = init_tgt;
            for p in &traj {
                ws += 1;
                let pe = ((p.x-cur_tgt.0).powi(2)+(p.y-cur_tgt.1).powi(2)+(p.z-cur_tgt.2).powi(2)).sqrt();
                if pe < WAYPOINT_TOL { wn += 1; } else { wn = 0; }
                if (wn >= 15 || ws >= WAYPOINT_STEPS) && wi + 1 < wps.len() {
                    wi += 1; cur_tgt = wps[wi]; ws = 0; wn = 0;
                }
            }
            last_wp_idx = wi;
            last_traj = traj;
        }

        system.report_performance(steps as f64);
        system.report_episode_end(steps as f64);

        let avg_s   = recent_steps.iter().sum::<usize>() as f64 / recent_steps.len() as f64;
        let avg_e   = recent_errs.iter().sum::<f64>() / recent_errs.len() as f64;

        if (ep + 1) % print_every == 0 || ep == 0 || steps >= max_steps {
            // Policy probes
            let mut correct_probes = 0usize;
            let mut probe_str = String::new();
            for (i, &(px, py, pz, pphi, ptheta)) in probe_states.iter().enumerate() {
                let pd = Drone3D {
                    x: px, y: py, z: pz, vx: 0.0, vy: 0.0, vz: 0.0,
                    phi: pphi, theta: ptheta, psi: 0.0,
                    omx: 0.0, omy: 0.0, omz: 0.0, t: 0.0,
                };
                let out = system.process_steps(&pd.observe(wps[0]), INTERNAL_STEPS);
                let chose = out.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i).unwrap_or(0);
                let ok = chose == probe_expected[i];
                if ok { correct_probes += 1; }
                probe_str.push(if ok { '+' } else { '-' });
            }
            let diag = system.diagnostics();
            println!("Ep {:>4} | {:>3}/{} | avg {:>5.1} | err {:>5.3}m | near={:.0}% | wp={:>3} | probe={}/4 [{}] | crash={} | {}",
                ep + 1, steps, max_steps, avg_s, avg_e, near_pct * 100.0, total_wps,
                correct_probes, probe_str, crash_cause, diag.summary());

            if ep > 0 && !last_traj.is_empty() {
                visualize(&last_traj, wps[last_wp_idx], ep + 1, last_wp_idx, wps, &wind);
            }
        }

        if recent_steps.len() >= 100 && avg_s >= max_steps as f64 * 0.80 {
            println!("\n*** MASTERED at ep {}! avg={:.1}/{} ({:.0}%) ***",
                ep + 1, avg_s, max_steps, avg_s / max_steps as f64 * 100.0);
            break;
        }
    }

    println!("\n=== Final ===");
    let s    = system.inspect();
    let diag = system.diagnostics();
    let avg_100 = recent_steps.iter().sum::<usize>() as f64 / recent_steps.len().max(1) as f64;
    let avg_err = recent_errs.iter().sum::<f64>() / recent_errs.len().max(1) as f64;
    let mastered = recent_steps.len() >= 100 && avg_100 >= max_steps as f64 * 0.80;
    println!("Morphons: {} | Synapses: {} | FR: {:.3}", s.total_morphons, s.total_synapses, s.firing_rate);
    println!("Learning: {}", diag.summary());
    println!("Endo: {}", system.endo.summary());
    println!("Waypoints reached total: {}", total_wps);

    let version = env!("CARGO_PKG_VERSION");
    let results = json!({
        "benchmark": "drone3d", "profile": profile, "version": version,
        "gusts": gusts, "waypoints": wps,
        "timestamp": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
        "episodes": num_eps, "max_steps_per_episode": max_steps,
        "results": {
            "best_steps": best_steps, "avg_last_100": avg_100,
            "avg_3d_position_error_m": avg_err,
            "waypoints_reached_total": total_wps, "mastered": mastered,
            "episode_steps": all_steps, "episode_errors": all_errs,
        },
        "physics": {
            "mass_kg": DRONE_MASS, "arm_m": ARM,
            "I_xx": I_XX, "I_yy": I_YY, "I_zz": I_ZZ,
            "max_thrust_per_rotor_n": MAX_T,
        },
        "system": {
            "morphons": s.total_morphons, "synapses": s.total_synapses,
            "firing_rate": s.firing_rate, "avg_myelination": s.avg_myelination,
        },
        "diagnostics": {
            "weight_mean": diag.weight_mean, "weight_std": diag.weight_std,
            "active_tags": diag.active_tags, "total_captures": diag.total_captures,
        },
    });

    let dir = format!("docs/benchmark_results/v{}", version);
    fs::create_dir_all(&dir).ok();
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let json_str = serde_json::to_string_pretty(&results).unwrap();
    fs::write(format!("{}/drone3d_{}.json", dir, ts), &json_str).unwrap();
    fs::write(format!("{}/drone3d_latest.json", dir), &json_str).unwrap();
    println!("\nResults saved to docs/benchmark_results/v{}/drone3d_{}.json", version, ts);
}
