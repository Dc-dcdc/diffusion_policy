import time
import click
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import scipy.spatial.transform as st
import zarr

# 引入 ReplayBuffer 用于读取数据
from diffusion_policy.common.replay_buffer import ReplayBuffer

# =============================================================================
# 1. 仿真环境类 (直接复用您提供的 SimEnv)
# =============================================================================
class SimEnv:
    def __init__(self, xml_path, frequency=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.max_pos_speed = 0.5 # 提高限速以匹配录制时的速度
        
        # 渲染设置
        self.render_width = 320 
        self.render_height = 240
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        
        self.mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap_target')
        self.mocap_id = self.model.body_mocapid[self.mocap_body_id]
        
        # 初始化环境
        self.reset_env()

    def reset_env(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        
        # 同步 Mocap 到初始位置
        tool_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[tool_id]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[tool_id]
        mujoco.mj_forward(self.model, self.data)

    def set_robot_pose(self, pose):
        """强制设置机械臂位置 (用于回放初始化)"""
        # pose: [x, y]
        current_z = 0.96 # 固定高度
        self.data.mocap_pos[self.mocap_id] = [pose[0], pose[1], current_z]
        # 让物理引擎运行几步以稳定
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def get_vis_img(self):
        self.renderer.update_scene(self.data, camera="Top_down_Camera")
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def exec_actions(self, action):
        """执行动作: action 是 (2,) 的 XY 坐标"""
        current_pos = self.data.mocap_pos[self.mocap_id][:2].copy()
        
        # 计算 Delta
        target_xy = action
        delta = target_xy - current_pos
        
        # 简单的限速逻辑
        dt = 1.0 / self.frequency
        max_step_dist = self.max_pos_speed * dt
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_step_dist:
            delta = delta / delta_norm * max_step_dist
            
        safe_target_xy = current_pos + delta
        safe_target_xy = np.clip(safe_target_xy, [-0.5, -0.5], [1.0, 0.5])
        
        target_z = 0.96
        self.data.mocap_pos[self.mocap_id] = [safe_target_xy[0], safe_target_xy[1], target_z]
        
        sim_steps = int(1.0 / self.frequency / self.model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)

# =============================================================================
# 2. 主函数 (回放逻辑)
# =============================================================================
@click.command()
@click.option('--dataset', '-d', default='/home/dc/diffusion_policy/data/demo_pusht_mujoco_notback/replay_buffer.zarr', help='Path to Zarr dataset (e.g., data/replay_buffer.zarr)')
@click.option('--xml_path', '-x', default='/home/dc/diffusion_policy/diffusion_policy/env/pusht/mujoco_ur5e/pusht.xml', help="Path to your Mujoco XML.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--start_ep', default=0, type=int, help="Start episode index.")
def main(dataset, xml_path, frequency, start_ep):
    # 1. 加载数据集
    print(f"Loading dataset from: {dataset}")
    # 使用 ReplayBuffer 读取 Zarr
    replay_buffer = ReplayBuffer.copy_from_path(dataset, keys=['action', 'robot_eef_pose'])
    print(f"Total Episodes: {replay_buffer.n_episodes}")
    
    # 2. 初始化环境
    env = SimEnv(xml_path=xml_path, frequency=frequency)
    
    cv2.namedWindow('Replay View', cv2.WINDOW_AUTOSIZE)
    
    print("========================================")
    print("Starting Replay...")
    print("Press 'N' to next episode, 'Q' to quit.")
    print("========================================")

    # 3. 遍历每一个 Episode
    for ep_idx in range(start_ep, replay_buffer.n_episodes):
        print(f"Playing Episode {ep_idx} ...")
        
        # 获取该 Episode 的数据范围
        start_idx = replay_buffer.episode_ends[ep_idx-1] if ep_idx > 0 else 0
        end_idx = replay_buffer.episode_ends[ep_idx]
        
        # 提取当前 episode 的动作和状态
        # 注意: 这里的 action 是您录制时存进去的，理论上是绝对坐标
        episode_actions = replay_buffer['action'][start_idx:end_idx]
        episode_start_pose = replay_buffer['robot_eef_pose'][start_idx]
        
        # === 重置环境并对齐起点 ===
        env.reset_env()
        # [关键] 将机械臂瞬移到录制时的起始位置
        # 否则回放的动作会因为起点不同而产生偏移
        env.set_robot_pose(episode_start_pose) 
        
        # 4. 逐帧执行动作
        for step, action in enumerate(episode_actions):
            # 执行动作
            env.exec_actions(action)
            
            # 渲染
            vis_img = env.get_vis_img()
            
            # 显示信息
            cv2.putText(vis_img, f"Ep: {ep_idx} | Step: {step}/{len(episode_actions)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(vis_img, f"Act: {action[:2]}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow('Replay View', vis_img)
            
            # 键盘控制
            key = cv2.pollKey()
            if key == ord('q'): 
                exit(0)
            elif key == ord('n'): # 跳过当前 Episode
                break
                
            # 保持回放速度
            time.sleep(1/frequency)
            
        # 每个 Episode 结束后暂停一下
        time.sleep(0.5)

if __name__ == '__main__':
    main()