import os
os.environ["MUJOCO_GL"] = "egl"
import pathlib, mujoco, imageio
from mujoco_mpc import agent as agent_lib

model_path = pathlib.Path('/home/ahung2/mujoco-env/lib/python3.12/site-packages/mujoco_mpc/mjpc/tasks/quadruped/task_hill.xml')
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)
print("nq", model.nq, "nv", model.nv, "nu", model.nu)
renderer = mujoco.Renderer(model, 480, 640)

frames, fps = [], 30
with agent_lib.Agent(task_id="Quadruped Hill", model=model) as agent:
    agent.reset()
    while data.time < 5.0:
        agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel,
                        act=data.act, mocap_pos=data.mocap_pos,
                        mocap_quat=data.mocap_quat)
        agent.planner_step()
        data.ctrl = agent.get_action()
        mujoco.mj_step(model, data)
        if len(frames) < data.time * fps:
            renderer.update_scene(data)
            frames.append(renderer.render())

imageio.mimsave("mpc_quadruped_hill.mp4", frames, fps=fps)
print("Saved mpc_quadruped_hill.mp4")
