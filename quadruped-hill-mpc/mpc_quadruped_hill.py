import os
os.environ["MUJOCO_GL"] = "egl"
import pathlib, mujoco, imageio, subprocess, numpy as np
from mujoco_mpc import agent as agent_lib

# --- experiment parameters ---
CONDITION = "stuck_FR_knee"
FRICTIONLOSS = 0.2
DAMPING = 2.0
DAMPING_ABDUCTION = 1.0

# --- patch a1_modified.xml ---
a1_src = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/a1_base.xml')
a1_dst = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/a1_modified.xml')

with open(a1_src) as f:
    xml = f.read()

xml = xml.replace(
    'damping="2" armature="0.01" frictionloss="0.2"',
    f'damping="{DAMPING}" armature="0.01" frictionloss="{FRICTIONLOSS}"'
)
xml = xml.replace(
    'axis="1 0 0" damping="1" range=',
    f'axis="1 0 0" damping="{DAMPING_ABDUCTION}" range='
)
xml = xml.replace(
    '<joint class="knee" name="FR_calf_joint"/>',
    '<joint class="knee" name="FR_calf_joint" damping="50" range="-1.5 -1.4"/>'
)

with open(a1_dst, 'w') as f:
    f.write(xml)
print(f"Patched: frictionloss={FRICTIONLOSS}, damping={DAMPING}, stuck FR knee")

# --- load model ---
model_path = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/task_hill.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
print("nq", model.nq, "nv", model.nv, "nu", model.nu)
renderer = mujoco.Renderer(model, 480, 640)

# --- camera ---
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
camera.trackbodyid = model.body('trunk').id
camera.distance = 1.5
camera.elevation = -20
camera.azimuth = 45

# --- run ---
frames, fps = [], 30
fell = False
unstable = False

with agent_lib.Agent(task_id="Quadruped Hill", model=model) as agent:
    agent.reset()
    while data.time < 30.0:
        agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel,
                        act=data.act, mocap_pos=data.mocap_pos,
                        mocap_quat=data.mocap_quat)
        try:
            agent.planner_step()
        except Exception as e:
            print(f"Planner failed at t={data.time:.2f}s: {e}")
            break
        data.ctrl = agent.get_action()
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            print(f"Simulation unstable at t={data.time:.2f}s")
            unstable = True
            break
        if data.qpos[2] < 0.15:
            fell = True
            print(f"Fell at t={data.time:.2f}s")
            break
        if len(frames) < data.time * fps:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

print(f"Condition: {CONDITION}")
print(f"Final time: {data.time:.2f}s")
print(f"Final torso height: {data.qpos[2]:.3f}m")
print(f"Fell: {fell}, Unstable: {unstable}")
imageio.mimsave(f"mpc_hill_{CONDITION}.mp4", frames, fps=fps)
print(f"Saved mpc_hill_{CONDITION}.mp4")

subprocess.run(["cp", f"mpc_hill_{CONDITION}.mp4", "/home/ahung2/trio-andrew/quadruped-hill-mpc/"], check=True)
subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "add", "quadruped-hill-mpc/"], check=True)
subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "commit", "-m", f"add {CONDITION} result"], check=True)
subprocess.run(["git", "-C", "/home/ahung2/trio-andrew", "push"], check=True)
print(f"Pushed {CONDITION} to GitHub")
