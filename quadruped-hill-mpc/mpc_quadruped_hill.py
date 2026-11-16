import os
os.environ["MUJOCO_GL"] = "egl"
import pathlib, mujoco, imageio, shutil
from mujoco_mpc import agent as agent_lib

# --- experiment parameters, change these each run ---
CONDITION = "tar"
FRICTIONLOSS = 8.0
DAMPING = 20.0
DAMPING_ABDUCTION = 12.0

# --- patch a1_modified.xml with current parameters ---
a1_src = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/a1_base.xml')
a1_dst = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/a1_modified.xml')

with open(a1_src) as f:
    xml = f.read()

# patch default joint (hip/knee)
xml = xml.replace(
    'damping="2" armature="0.01" frictionloss="0.2"',
    f'damping="{DAMPING}" armature="0.01" frictionloss="{FRICTIONLOSS}"'
)
# patch abduction damping
xml = xml.replace(
    'axis="1 0 0" damping="1" range=',
    f'axis="1 0 0" damping="{DAMPING_ABDUCTION}" range='
)

with open(a1_dst, 'w') as f:
    f.write(xml)

print(f"Patched a1_modified.xml: frictionloss={FRICTIONLOSS}, damping={DAMPING}")

# --- load model and run ---
model_path = pathlib.Path('/home/ahung2/mujoco_mpc_new/build/mjpc/tasks/quadruped/task_hill.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
print("nq", model.nq, "nv", model.nv, "nu", model.nu)
renderer = mujoco.Renderer(model, 480, 640)

frames, fps = [], 30
with agent_lib.Agent(task_id="Quadruped Hill", model=model) as agent:
    agent.reset()
    while data.time < 30.0:
        agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel,
                        act=data.act, mocap_pos=data.mocap_pos,
                        mocap_quat=data.mocap_quat)
        agent.planner_step()
        data.ctrl = agent.get_action()
        mujoco.mj_step(model, data)
        if len(frames) < data.time * fps:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

print(f"Condition: {CONDITION}")
print(f"Final time: {data.time:.2f}s")
print(f"Final torso height: {data.qpos[2]:.3f}m")
print(f"Fell: {data.qpos[2] < 0.15}")
imageio.mimsave(f"mpc_hill_{CONDITION}.mp4", frames, fps=fps)
print(f"Saved mpc_hill_{CONDITION}.mp4")
