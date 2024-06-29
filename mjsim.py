import math
import numpy as np
import mujoco, mujoco_viewer
from tqdm import tqdm
from collections import deque


from robots.robot_config.GR1_mj_config import GR1T1LowerLimbCfg ##lower limb cfg version
import torch

class cmd:
    #type in the command to ctrl the robot movement
    vx = 0.0
    vy = 0.0
    dyaw = 0.0

def get_obs(data):
    '''Extracts an observation from the mujoco data structure
    '''
    q = data.qpos.astype(np.double)
    dq = data.qvel.astype(np.double)
    quat = data.sensor('orientation').astype(np.double)#[w,x,y,z]
    omega = data.sensor('angular-velocity').data.astype(np.double)
    
    return (q, dq, quat, omega)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    '''Calculates torques from position commands
    '''
    return (target_q - q) * kp + (target_dq - dq) * kd

def run_mujoco(policy, cfg):
    model = mujoco.MjModel.from_xml_path(cfg.sim_config.mujoco_model_path)
    model.opt.timestep = cfg.sim_config.dt
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    #initialize
    target_q = np.zeros((cfg.env.num_actions), dtype=np.double)
    action = np.zeros((cfg.env.num_actions), dtype=np.double)
    target_dq = np.zeros((cfg.env.num_actions), dtype=np.double)
    count_lowlevel = 0


    for _ in tqdm(range(int(cfg.sim_config.sim_duration / cfg.sim_config.dt)), desc="Simulating..."):

        # Obtain an observation
        q, dq, quat, omega= get_obs(data)

        q = q[-cfg.env.num_actions:]
        dq = dq[-cfg.env.num_actions:]
        quat_inv = quat_rotate_inverse(quat)
        # 1000hz -> 50hz
        if count_lowlevel % cfg.sim_config.decimation == 0:

            obs = np.zeros([1, cfg.env.num_single_obs], dtype=np.float32)

            obs[0, 0:3] = omega
            obs[0, 3] = cmd.vx 
            obs[0, 4] = cmd.vy 
            obs[0, 5] = cmd.dyaw 
            obs[0, 6:9] = quat_inv
            obs[0, 9:19] = q * cfg.normalization.obs_scales.dof_pos
            obs[0, 19:29] = dq * cfg.normalization.obs_scales.dof_vel
            obs[0, 29:39] = action
            obs = np.clip(obs, -cfg.normalization.clip_observations, cfg.normalization.clip_observations)

            policy_input = np.zeros([1, cfg.env.num_observations], dtype=np.float32)
            
            policy_input[0, :cfg.env.num_single_obs] = obs[0, :cfg.env.num_single_obs]   
                     
            action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
            action = np.clip(action, cfg.normalization.clip_actions_min, cfg.normalization.clip_actions_max)
           
            target_q = action * cfg.control.action_scale
            target_dq = target_q - obs[0, 29:39]

        # Generate PD control
        tau = pd_control(target_q, q, cfg.robot_config.kps,
                        target_dq, dq, cfg.robot_config.kds)  # Calc torques
        tau = np.clip(tau, -cfg.robot_config.tau_limit, cfg.robot_config.tau_limit)  # Clamp torques
        data.ctrl = tau

        mujoco.mj_step(model, data)
        viewer.render()
        count_lowlevel += 1

    viewer.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deployment script.')
    parser.add_argument('--load_model', type=str, required=True,
                        help='Run to load from.')
    parser.add_argument('--terrain', action='store_true', help='terrain or plane')
    args = parser.parse_args()
    mujoco_model_path = GR1T1LowerLimbCfg.MujocoModelPath(path='./robots/gr1t1/scene.xml')
    policy = torch.jit.load(args.load_model)
    run_mujoco(policy, GR1T1LowerLimbCfg)
