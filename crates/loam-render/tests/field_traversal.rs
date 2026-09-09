use glam::Vec3;
use loam_math::EuclideanR3;
use loam_runtime::domain::FieldKind;
use loam_runtime::field::{evaluate_counted, FieldCounts, FieldPrimitive, OP_SPHERE, OP_UNION};
use loam_runtime::FieldProgram;
use loam_scene::{Scene, SceneNode};

const RADIUS: f32 = 0.35;
const HIT_EPS: f32 = 1e-3;
const MIN_STEP: f32 = 1e-4;
const MAX_T: f32 = 60.0;

fn centers(count: usize) -> Vec<Vec3> {
    (0..count)
        .map(|i| {
            let t = i as f32;
            Vec3::new(
                (t * 0.37).sin() * 4.0,
                (t * 0.71).cos() * 4.0,
                -3.0 - (t * 0.11).sin().abs() * 8.0,
            )
        })
        .collect()
}

fn identity_frame() -> [[f32; 4]; 4] {
    let mut frame = [[0.0; 4]; 4];
    for (axis, column) in frame.iter_mut().enumerate() {
        column[axis] = 1.0;
    }
    frame
}

fn interpreted(centers: &[Vec3]) -> FieldProgram {
    let mut program = FieldProgram {
        primitives: centers
            .iter()
            .map(|c| FieldPrimitive {
                frame: identity_frame(),
                translation: [c.x, c.y, c.z, 0.0],
                params: [RADIUS, 0.0, 0.0, 0.0],
            })
            .collect(),
        program: Vec::new(),
        stack: 2,
        kind: FieldKind::ExactDistance,
    };
    for index in 0..centers.len() as u32 {
        program.program.extend([OP_SPHERE, index]);
        if index > 0 {
            program.program.extend([OP_UNION, 0]);
        }
    }
    program
}

fn specialized(centers: &[Vec3]) -> Scene {
    let mut root = SceneNode::sphere(centers[0], RADIUS);
    for &c in &centers[1..] {
        root = root.union(SceneNode::sphere(c, RADIUS));
    }
    Scene::new(root)
}

fn march(mut sdf: impl FnMut(Vec3) -> f32, ro: Vec3, rd: Vec3) -> (Option<f32>, u64) {
    let mut t = 0.0f32;
    let mut steps = 0u64;
    for _ in 0..256 {
        steps += 1;
        let d = sdf(ro + rd * t);
        if d < HIT_EPS {
            return (Some(t), steps);
        }
        t += d.max(MIN_STEP);
        if t > MAX_T {
            break;
        }
    }
    (None, steps)
}

#[test]
fn the_interpreter_and_the_specialized_emit_march_the_same_ray_alike() {
    let centers = centers(1000);
    let program = interpreted(&centers);
    let scene = specialized(&centers);
    let ro = Vec3::ZERO;

    let mut agreed = 0;
    let mut interpreter_steps = 0u64;
    let mut interpreter_counts = FieldCounts::default();
    let mut emit_steps = 0u64;
    for pixel in 0..16 {
        let angle = pixel as f32 * 0.19;
        let rd = Vec3::new(angle.sin() * 0.4, angle.cos() * 0.4, -1.0).normalize();

        let mut counts = FieldCounts::default();
        let (interpreted_hit, steps) = march(
            |p| {
                evaluate_counted(
                    &program.program,
                    &program.primitives,
                    [p.x, p.y, p.z, 0.0],
                    &mut counts,
                )
                .expect("well-formed program")
            },
            ro,
            rd,
        );
        interpreter_steps += steps;
        interpreter_counts.primitive_evals += counts.primitive_evals;
        interpreter_counts.instructions += counts.instructions;

        let (emit_hit, steps) = march(|p| scene.eval(&EuclideanR3, p), ro, rd);
        emit_steps += steps;

        match (interpreted_hit, emit_hit) {
            (Some(a), Some(b)) => {
                assert!(
                    (a - b).abs() < 1e-4,
                    "ray {pixel}: interpreter hit at {a}, specialized emit at {b}"
                );
                agreed += 1;
            }
            (None, None) => agreed += 1,
            (a, b) => panic!("ray {pixel}: interpreter {a:?}, specialized emit {b:?}"),
        }
    }
    assert_eq!(agreed, 16);
    assert_eq!(interpreter_steps, emit_steps);
    println!(
        "1000 primitives, 16 rays: steps {interpreter_steps} interpreter, {emit_steps} \
         specialized; primitive evaluations {} interpreter, {} specialized; \
         instructions {}",
        interpreter_counts.primitive_evals,
        emit_steps * centers.len() as u64,
        interpreter_counts.instructions,
    );
}
