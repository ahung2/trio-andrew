import os
os.environ["MUJOCO_GL"] = "egl"
import pathlib, mujoco, imageio, numpy as np, subprocess
from tinympc.interface import TinyMPC

# --- dimensions ---
NX = 37   # qpos (19) + qvel (18)
NU = 12   # joint torques
N  = 25   # horizon steps
DT = 0.01 # timestep

# --- model ---
model_path = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/task_hill.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))
data  = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
print(f"nq={model.nq} nv={model.nv} nu={model.nu}")

# --- reference state: standing upright ---
x_ref = np.zeros(NX)
x_ref[:model.nq] = data.qpos.copy()   # home qpos from keyframe
x_ref[2]         = 0.27               # target trunk height

# --- cost matrices ---
Q = np.diag(np.array(
    [100.0]*3 +   # trunk xyz position
    [200.0]*4 +   # trunk quaternion
    [10.0]*12 +   # joint positions
    [5.0]*3  +   # trunk linear velocity
    [5.0]*3  +   # trunk angular velocity
    [1.0]*12     # joint velocities
))
R = np.diag(np.ones(NU) * 0.001)

# --- control bounds (from a1.xml forcerange) ---
u_min = np.full(NU, -33.5)
u_max = np.full(NU,  33.5)

# --- state bounds ---
x_min = -np.inf * np.ones(NX)
x_max =  np.inf * np.ones(NX)
x_min[2] = 0.05   # trunk height must stay above 5cm

def get_state(data):
    return np.concatenate([data.qpos, data.qvel])

def linearize_dynamics(model, data, u0, eps=1e-4):
    """Finite difference linearization around current state/control."""
    x0 = get_state(data)
    nx, nu = NX, NU

    # save state
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = data.ctrl.copy()

    def step_from(qpos, qvel, ctrl):
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.ctrl[:] = ctrl
        mujoco.mj_step(model, d)
        return np.concatenate([d.qpos, d.qvel])

    # nominal next state
    x1_nom = step_from(qpos0, qvel0, u0)

    # A matrix: perturb state
    A = np.zeros((nx, nx))
    for i in range(model.nq):
        qpos_p = qpos0.copy(); qpos_p[i] += eps
        x1_p = step_from(qpos_p, qvel0, u0)
        A[:, i] = (x1_p - x1_nom) / eps

    for i in range(model.nv):
        qvel_p = qvel0.copy(); qvel_p[i] += eps
        x1_p = step_from(qpos0, qvel_p, u0)
        A[:, model.nq + i] = (x1_p - x1_nom) / eps

    # B matrix: perturb control
    B = np.zeros((nx, nu))
    for i in range(nu):
        u_p = u0.copy(); u_p[i] += eps
        x1_p = step_from(qpos0, qvel0, u_p)
        B[:, i] = (x1_p - x1_nom) / eps

    # affine offset: f = x1_nom - A @ x0 - B @ u0
    f = x1_nom - A @ x0 - B @ u0

    return A, B, f

# --- initialize TinyMPC ---
print("Linearizing dynamics at home pose...")
u0 = np.zeros(NU)
A0, B0, f0 = linearize_dynamics(model, data, u0)
print(f"A: {A0.shape}, B: {B0.shape}, f: {f0.shape}")

prob = TinyMPC()
prob.setup(
    A=A0, B=B0, Q=Q, R=R, N=N,
    fdyn=f0,
    u_min=u_min, u_max=u_max,
    x_min=x_min, x_max=x_max,
    rho=1.0,
    max_iter=100,
    check_termination=10,
    en_state_bound=True,
    en_input_bound=True,
    verbose=False
)
prob.set_x_ref(x_ref)
print("TinyMPC initialized")

# --- renderer ---
renderer = mujoco.Renderer(model, 480, 640)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
camera.trackbodyid = model.body('trunk').id
camera.distance = 1.5
camera.elevation = -20
camera.azimuth = 45

# --- MPC loop ---
frames, fps = [], 30
fell = False
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)
u_prev = np.zeros(NU)

print("Running TinyMPC control loop...")
while data.time < 30.0:
    x0 = get_state(data)

    # re-linearize every 5 steps
    if int(data.time / DT) % 5 == 0:
        A, B, f = linearize_dynamics(model, data, u_prev)
        prob.A = A
        prob.B = B
        prob.compute_cache_terms()

    prob.set_x0(x0)
    output = prob.solve()
    u = np.array(output['controls'])[:, 0] if output['controls'].ndim > 1 else np.array(output['controls'])

    data.ctrl[:] = np.clip(u, u_min, u_max)
    u_prev = u.copy()

    mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all():
        print(f"Unstable at t={data.time:.2f}s")
        break
    if data.qpos[2] < 0.15:
        fell = True
        print(f"Fell at t={data.time:.2f}s")
        break

    if len(frames) < data.time * fps:
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render())

print(f"Final time: {data.time:.2f}s")
print(f"Final height: {data.qpos[2]:.3f}m")
print(f"Fell: {fell}")

imageio.mimsave("mpc_tinympc_quadruped.mp4", frames, fps=fps)
print("Saved mpc_tinympc_quadruped.mp4")

subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "add", "quadruped-hill-mpc/"], check=True)
subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "commit", "-m", "add TinyMPC ADMM Riccati controller"], check=True)
subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "push"], check=True)
print("Pushed to GitHub")
