import time
import os
import shutil # 用于删除文件夹
import pathlib #  用于路径处理
import click
import cv2
import numpy as np
import mujoco
import mujoco.viewer
import scipy.spatial.transform as st
import math
from pynput import keyboard
import collections
import torch
import dill
import hydra
from omegaconf import OmegaConf
from datetime import datetime
# 复用 diffusion_policy 现有的工具
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

# 注册 OmegaConf eval 解析器 (加载模型配置必需)
OmegaConf.register_new_resolver("eval", eval, replace=True)

# ==========================================
# 虚拟 SpaceMouse (支持大窗口触控板模式)
# ==========================================
class VirtualSpaceMouse:
    def __init__(self, window_name, sensitivity=1.5):
        self.window_name = window_name
        self.sensitivity = sensitivity
        self.dx = 0
        self.dy = 0
        self.last_x = None
        self.last_y = None
        
        # 状态标志
        self.left_btn = False
        self.right_btn = False
        self.shift = False
        self.ctrl = False
        
        # 注册鼠标回调函数
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        self.shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        self.ctrl = bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        self.left_btn = bool(flags & cv2.EVENT_FLAG_LBUTTON)
        self.right_btn = bool(flags & cv2.EVENT_FLAG_RBUTTON)

        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return

        if event == cv2.EVENT_MOUSEMOVE:
             if self.left_btn: 
                self.dx += (x - self.last_x)
                self.dy += (y - self.last_y)
        
        self.last_x = x
        self.last_y = y

    def get_motion_state_transformed(self):
        cur_dx = self.dx
        cur_dy = self.dy
        self.dx = 0
        self.dy = 0
        
        scale = 0.01 * self.sensitivity
        vx, vy, vz, vr, vp, vyaw = 0, 0, 0, 0, 0, 0
        
        if self.shift:
            vz = -cur_dy * scale 
        elif self.ctrl:
            vyaw = cur_dx * scale * 2.0
        else:
            vx = cur_dy * scale 
            vy = cur_dx * scale 
            
        state = np.array([vx, vy, vz, vr, vp, vyaw])
        return np.clip(state, -1.0, 1.0)
    
    def draw_feedback(self, img):
        h, w = img.shape[:2]
        cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), (50, 50, 50), 2)
        cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), (50, 50, 50), 2)
        
        status = "Mode: Move XY"
        color = (0, 255, 0)
        if self.shift: 
            status = "Mode: Move Z (Height)"
            color = (0, 255, 255)
        elif self.ctrl: 
            status = "Mode: Rotate (Yaw)"
            color = (0, 100, 255)
            
        if self.left_btn:
            cv2.circle(img, (self.last_x, self.last_y), 10, color, -1)
            cv2.putText(img, "DRAGGING", (self.last_x + 15, self.last_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(img, "Hold L-Click to Move", (w//2 - 100, h//2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

        cv2.putText(img, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(img, "Shift: Z-Axis | Ctrl: Rotate", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

# ==========================================
# One Euro Filter (防抖动)
# ==========================================
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=None, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        
        self.x_prev = float(x0) if isinstance(x0, (float, int)) else np.array(x0, dtype=np.float64)
        self.dx_prev = float(dx0) if dx0 is not None else np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self, t0, x0):
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t0

# =============================================================================
# ObsBuffer (用于处理历史帧堆叠和图像Resize，从 Runner 移植)
# =============================================================================
class ObsBuffer:
    def __init__(self, n_obs_steps, shape_meta):
        self.n_obs_steps = n_obs_steps
        self.buffer = collections.deque(maxlen=n_obs_steps)
        self.shape_meta = shape_meta 
    
    def reset(self):
        self.buffer.clear()
        
    def add_obs(self, obs):
        self.buffer.append(obs)
        
    def get_stacked_obs(self):
        # 如果 Buffer 没满，复制第一帧填充
        while len(self.buffer) < self.n_obs_steps:
            self.buffer.appendleft(self.buffer[0])
            
        stacked = {}
        keys = self.buffer[0].keys()
        
        for k in keys:
            if k == 'timestamp': continue
            val = np.stack([x[k] for x in self.buffer]) 
            
            # 图像处理 (Resize + Normalize)
            if 'obs' in self.shape_meta and k in self.shape_meta['obs']:
                meta = self.shape_meta['obs'][k]
                if meta.get('type') == 'rgb':
                    target_shape = tuple(meta['shape']) 
                    current_shape = val.shape[1:] 
                    
                    if current_shape[-2:] != target_shape[-2:]:
                        T, C, H, W = val.shape
                        resized_imgs = []
                        for i in range(T):
                            img_hwc = np.moveaxis(val[i], 0, -1)
                            img_res = cv2.resize(img_hwc, (target_shape[2], target_shape[1]), interpolation=cv2.INTER_AREA)
                            resized_imgs.append(np.moveaxis(img_res, -1, 0))
                        val = np.array(resized_imgs)
                    
                    val = val.astype(np.float32) / 255.0
                    
            stacked[k] = val
        return stacked

# ==========================================
# 仿真环境 SimEnv
# ==========================================
class SimEnv:
    def __init__(self, xml_path, output_dir, frequency=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.render_width = 320
        self.render_height = 240
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self.Tblock_random_init=True 
        
        self.tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pusht_block')
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal_target')
        self.bullseye_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bullseye_target')
        
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap_target')
        if mocap_body_id == -1: raise ValueError("XML Error: mocap_target missing")
        self.mocap_id = self.model.body_mocapid[mocap_body_id]
        
        block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pusht_block')
        if block_body_id != -1:
            joint_addr = self.model.body_jntadr[block_body_id]
            self.block_qpos_addr = self.model.jnt_qposadr[joint_addr]
        else:
            print("Warning: pusht_block not found, randomization disabled.")
            self.block_qpos_addr = -1
        
        self.output_dir = pathlib.Path(output_dir)

        self.video_root_dir = self.output_dir.joinpath('videos')
        self.video_root_dir.mkdir(parents=True, exist_ok=True)
        self.video_writers = {} 

        self.current_episode = {}
        self.is_recording = False 
        self.this_video_dir = None 
        self.t_record_start = 0.0 
        self.reset_env()

        self.max_pos_speed = 0.5  #鼠标到mujoco速度的映射比例
        self.max_rot_speed = 1.0

    def reset_env(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self._sync_mocap_to_robot()
        
        if self.block_qpos_addr != -1 and self.Tblock_random_init:
            rand_x = np.random.uniform(0.40, 0.60)  
            rand_y = np.random.uniform(-0.35, -0.15)
            fixed_z = 0.95 
            rand_yaw = np.random.uniform(0.5, 3.5)
            rot = st.Rotation.from_euler('z', rand_yaw) 
            quat_xyzw = rot.as_quat() 
            mj_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            start_idx = self.block_qpos_addr
            self.data.qpos[start_idx : start_idx+3] = [rand_x, rand_y, fixed_z]
            self.data.qpos[start_idx+3 : start_idx+7] = mj_quat
        else:
            target_pos = [0.50, -0.25, 0.95] 
            target_euler = [0, 0, 2.5] 
            rot = st.Rotation.from_euler('xyz', target_euler)
            quat_xyzw = rot.as_quat()
            mj_quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            start_idx = self.block_qpos_addr
            self.data.qpos[start_idx : start_idx+3] = target_pos
            self.data.qpos[start_idx+3 : start_idx+7] = mj_quat
        
        mujoco.mj_forward(self.model, self.data)
        self._move_robot_to_start_pose()
        print(f">>> Environment Reset Done.")

    def _sync_mocap_to_robot(self):
        tool_pusher = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[tool_pusher]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[tool_pusher]
        mujoco.mj_forward(self.model, self.data)

    def _move_robot_to_start_pose(self):
        target_pos = np.array([0.3, -0.2, 0.96]) 
        rot = st.Rotation.from_euler('xyz', [-180, 0, -90], degrees=True)
        quat_xyzw = rot.as_quat() 
        target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]) 
        self.data.mocap_pos[self.mocap_id] = target_pos
        self.data.mocap_quat[self.mocap_id] = target_quat
        for _ in range(500): 
            mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        obs = {}
        # 强制刷新确保图像最新
        mujoco.mj_forward(self.model, self.data)
        
        self.renderer.update_scene(self.data, camera="Wrist_Camera")
        camera_0 = self.renderer.render() 
        obs['camera_0'] = np.moveaxis(camera_0, -1, 0) 
        
        self.renderer.update_scene(self.data, camera="Front_Camera")
        camera_1 = self.renderer.render() 
        obs['camera_1'] = np.moveaxis(camera_1, -1, 0) 
        
        pos = self.data.mocap_pos[self.mocap_id].copy()
        obs['robot_eef_pose'] = pos[:2] # 匹配训练数据键名

        self.renderer.update_scene(self.data, camera="Top_down_Camera")
        camera_2 = self.renderer.render() 
        obs['camera_2'] = np.moveaxis(camera_2, -1, 0) # 添加这一行，存入 obs

        self.last_vis_img = cv2.cvtColor(camera_2, cv2.COLOR_RGB2BGR)
        return obs

    def get_robot_state(self):
        pos = self.data.mocap_pos[self.mocap_id].copy()
        quat = self.data.mocap_quat[self.mocap_id].copy()
        rot_vec = st.Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_rotvec()
        pose = np.concatenate([pos, rot_vec])
        return {'TargetTCPPose': pose}

    def exec_actions(self, actions):
        target_pose = actions[0]
        self.data.mocap_pos[self.mocap_id] = target_pose[:3]
        rot = st.Rotation.from_rotvec(target_pose[3:])
        q = rot.as_quat() 
        self.data.mocap_quat[self.mocap_id] = [q[3], q[0], q[1], q[2]]
        
        sim_steps = int(1.0 / self.frequency / self.model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
        if self.is_recording:
            obs = self.get_obs() 
            if self.video_writers:
                img_0_rgb = np.moveaxis(obs['camera_0'], 0, -1) 
                img_0_bgr = cv2.cvtColor(img_0_rgb, cv2.COLOR_RGB2BGR)
                self.video_writers['0'].write(img_0_bgr)
                
                img_1_rgb = np.moveaxis(obs['camera_1'], 0, -1)
                img_1_bgr = cv2.cvtColor(img_1_rgb, cv2.COLOR_RGB2BGR)
                self.video_writers['1'].write(img_1_bgr)

                img_2_rgb = np.moveaxis(obs['camera_2'], 0, -1)
                img_2_bgr = cv2.cvtColor(img_2_rgb, cv2.COLOR_RGB2BGR)
                self.video_writers['2'].write(img_2_bgr)
        #     current_joint = self.data.qpos[2:8].copy()
        #     current_joint_vel = self.data.qvel[2:8].copy()
        #     data = {
        #         'action': target_pose[:2].copy(),
        #         'robot_eef_pose': target_pose[:2].copy(),
        #         'camera_0': obs['camera_0'],
        #         'camera_1': obs['camera_1'],
        #         'raw_action': target_pose[:].copy(),
        #         'raw_robot_eef_pose': target_pose[:].copy(),
        #         'robot_joint': current_joint,         
        #         'robot_joint_vel': current_joint_vel, 
        #         'timestamp': np.array([time.time()])
        #     }
        #     for key, value in data.items():
        #         if key not in self.current_episode: self.current_episode[key] = []
        #         self.current_episode[key].append(value)

    def start_episode(self):
        self.is_recording = True
        self.current_episode = {}
        self.t_record_start = time.monotonic() 
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.this_video_dir = self.video_root_dir.joinpath(time_str)
        self.this_video_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        path_0 = str(self.this_video_dir.joinpath('0.mp4'))
        self.video_writers['0'] = cv2.VideoWriter(path_0, fourcc, self.frequency, (self.render_width, self.render_height))
        path_1 = str(self.this_video_dir.joinpath('1.mp4'))
        self.video_writers['1'] = cv2.VideoWriter(path_1, fourcc, self.frequency, (self.render_width, self.render_height))
        path_2 = str(self.this_video_dir.joinpath('2.mp4'))
        self.video_writers['2'] = cv2.VideoWriter(path_2, fourcc, self.frequency, (self.render_width, self.render_height))
        print("Dataset Recording Started...")

    def end_episode(self):
        self.is_recording = False
        current_duration = time.monotonic() - self.t_record_start
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}

        if current_duration > 100.0:
            print(f"\n[Timeout] 录制时长 {current_duration:.1f}s > 100s。删除数据并重置环境...")
            if self.this_video_dir is not None and self.this_video_dir.exists():
                shutil.rmtree(str(self.this_video_dir))
            return False


    def drop_episode(self):
        self.is_recording = False
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}
        if self.this_video_dir is not None and self.this_video_dir.exists():
            shutil.rmtree(str(self.this_video_dir))
            print(f"Episode Dropped. Video folder deleted: {self.this_video_dir}")
        else:
            print("Episode Dropped (No video folder to delete).")
        self.this_video_dir = None

    def check_conditions(self, block_tol=0.05, robot_tol=0.05):
        block_pos = self.data.xpos[self.block_body_id][:2]
        goal_pos = self.data.xpos[self.goal_body_id][:2]
        block_dist = np.linalg.norm(block_pos - goal_pos)
        
        block_quat = self.data.xquat[self.block_body_id]
        goal_quat = self.data.xquat[self.goal_body_id]
        block_yaw = st.Rotation.from_quat([block_quat[1], block_quat[2], block_quat[3], block_quat[0]]).as_euler('zyx')[0]
        goal_yaw = st.Rotation.from_quat([goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]]).as_euler('zyx')[0]
        rot_error = np.abs((block_yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi)
        
        is_success = (block_dist < block_tol) and (rot_error < 0.03) 

        tool_pos = self.data.xpos[self.tool_body_id][:2]
        target_pos = np.array([0, 0]) 
        if self.bullseye_body_id != -1:
            target_pos = self.data.xpos[self.bullseye_body_id][:2]
        
        robot_dist = np.linalg.norm(tool_pos - target_pos)
        is_robot_reset = robot_dist < robot_tol

        return is_success, is_robot_reset

@click.command()
@click.option('--output', '-o', default='/home/dc/diffusion_policy/data/test_pusht_mujoco', help="Directory to save dataset.")
@click.option('--xml_path', '-x', default='/home/dc/diffusion_policy/diffusion_policy/env/pusht/mujoco_ur5e/pusht.xml', help="Path to your Mujoco XML.")
@click.option('--checkpoint', '-c', default='/home/dc/diffusion_policy/data/outputs/2026.02.05/10.26.24_train_diffusion_unet_image_mujoco_image/checkpoints/epoch=0790-train_loss=0.000.ckpt', help='Path to checkpoint (.ckpt)')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
def main(output, xml_path, checkpoint, frequency=10):
    dt = 1/frequency
    
    # ==========================================
    # 1. 加载策略模型 (Policy Load)
    # ==========================================
    print(f"Loading checkpoint: {checkpoint}")
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    policy: BaseImagePolicy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.eval().to(device)
    
    # 设置推理步数
    policy.num_inference_steps = 6 # DDIM
    if hasattr(policy, 'n_action_steps'):
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    
    print("Model Loaded Successfully!")
    
    # 初始化 ObsBuffer
    obs_buffer = ObsBuffer(n_obs_steps=cfg.n_obs_steps, shape_meta=cfg.task.shape_meta)

    # 动作队列，用于存储模型预测出的未来动作序列
    action_queue = collections.deque(maxlen=100)
    # 定义每次预测执行的步数 (Open-Loop Horizon)
    steps_per_inference = 100
    # ==========================================
    # 2. 交互循环
    # ==========================================
    with KeystrokeCounter() as key_counter:
        control_window = 'Large Control Pad'
        control_w, control_h = 1200, 900
        cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(control_window, control_w, control_h) 
        
        robot_window = 'Robot View'
        cv2.namedWindow(robot_window, cv2.WINDOW_AUTOSIZE)
        
        with VirtualSpaceMouse(window_name=control_window, sensitivity=3.0) as sm:
            
            env = SimEnv(xml_path, output, frequency=frequency)

            t_start = time.monotonic()
            pos_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)
            rot_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)

            # [新增 Policy 滤波器] 处理模型绝对位置 (Absolute)
            # min_cutoff: 越小越平滑但延迟越高 (建议 0.1~1.0)
            # beta: 处理高速运动时的响应速度
            policy_pos_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.8, beta=0.1)

            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.lookat = np.array([0.5, 0, 0.8])
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 180 
                viewer.cam.elevation = -90 
                
                print(f"   Ready! 控制方式:")
                print(f"  [Mode]  P: 切换策略控制 (AI) | M: 切换手动控制 (Manual)")
                print(f"  [Move]  鼠标左键拖动 (Manual模式下)")
                print(f"  [Rec]   1:开始 | 2:保存 | 空格:重置")
                print(f"   0:不打断录像重置 | q:退出")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                # 初始化时，立刻将滤波器同步到机器人当前位置！
                policy_pos_filter.reset(t_start, target_pose[:3])
                iter_idx = 0
                stop = False
                t_start = time.monotonic()
                
                control_bg = np.zeros((control_h, control_w, 3), dtype=np.uint8)
                
                # 控制模式状态
                control_mode = "MANUAL" # "MANUAL" or "POLICY"
                
                while not stop and viewer.is_running():
                    current_time = time.monotonic() 
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    
                    # 1. 总是获取最新的观测，用于 Buffer 和 显示
                    obs = env.get_obs()
                    obs_buffer.add_obs(obs)

                    is_success, is_robot_reset = env.check_conditions(block_tol=0.005, robot_tol=0.008)
                    press_events = key_counter.get_press_events()

                    # 自动重置逻辑 (仅录制时或手动模式生效，避免打断Policy连贯性)
                    if is_success and is_robot_reset and control_mode == "MANUAL":
                        env.end_episode()
                        env.reset_env() 
                        obs_buffer.reset()
                        # [新增] 重置后立即补充一帧观测，防止 Buffer 为空
                        first_obs = env.get_obs()
                        obs_buffer.add_obs(first_obs)
                        policy.reset()
                        action_queue.clear() # [新增] 清空队列
                        pos_filter.reset(current_time, np.zeros(3))
                        rot_filter.reset(current_time, np.zeros(3))
                        state = env.get_robot_state()
                        target_pose = state['TargetTCPPose']

                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'): stop = True
                        elif key_stroke == KeyCode(char='p'): 
                            control_mode = "POLICY"
                            print(">>> Switched to POLICY Mode (AI)")
                            policy.reset() # 切换时重置策略状态
                            action_queue.clear() # [新增] 切换模式时清空队列
                            # [修复3] 切换到 AI 模式瞬间，同步滤波器，防止瞬移
                            policy_pos_filter.reset(current_time, target_pose[:3])
                        elif key_stroke == KeyCode(char='m'): 
                            control_mode = "MANUAL"
                            print(">>> Switched to MANUAL Mode")
                            action_queue.clear() # [新增] 切换模式时清空队列
                            # 切换回手动时，重置滤波器状态为当前静止，防止飞车
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3))
                            
                        elif key_stroke == KeyCode(char='1'): env.start_episode()
                        elif key_stroke == KeyCode(char='2'): 
                            env.end_episode()
                            env.reset_env() 
                            obs_buffer.reset()
                            # [新增] 重置后立即补充一帧观测，防止 Buffer 为空
                            first_obs = env.get_obs()
                            obs_buffer.add_obs(first_obs)
                            policy.reset()
                            action_queue.clear() # [新增] 重置时清空队列
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3))
                            # [修复2] 将 Policy 滤波器重置为当前 TCP 位置
                            policy_pos_filter.reset(current_time, target_pose[:3])
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                        elif key_stroke == Key.space: 
                            if env.is_recording: env.drop_episode()
                            env.reset_env()
                            obs_buffer.reset()
                            # [新增] 重置后立即补充一帧观测，防止 Buffer 为空
                            first_obs = env.get_obs()
                            obs_buffer.add_obs(first_obs)
                            policy.reset()
                            action_queue.clear() # [新增] 重置时清空队列
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3))
                            # [修复2] 将 Policy 滤波器重置为当前 TCP 位置
                            policy_pos_filter.reset(current_time, target_pose[:3])
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                        elif key_stroke == KeyCode(char='0'):  #随机初始化继续录像
                            env.reset_env()
                            obs_buffer.reset()
                            # [新增] 重置后立即补充一帧观测，防止 Buffer 为空
                            first_obs = env.get_obs()
                            obs_buffer.add_obs(first_obs)
                            policy.reset()
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3)) 
                            # [修复2] 将 Policy 滤波器重置为当前 TCP 位置
                            policy_pos_filter.reset(current_time, target_pose[:3])
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']

                    # ==========================================
                    # 核心控制逻辑分支
                    # ==========================================
                    if control_mode == "POLICY":
                        # AI 控制逻辑
                        # 如果队列为空，则进行推理并填充队列
                        if len(action_queue) == 0:
                            # 1. 准备数据
                            obs_dict_np = obs_buffer.get_stacked_obs()
                            filter_keys = ['camera_0', 'camera_1', 'robot_eef_pose']
                            obs_dict_np_filtered = {k: v for k, v in obs_dict_np.items() if k in filter_keys}
                            obs_dict = dict_apply(obs_dict_np_filtered, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            # 2. 推理
                            with torch.no_grad():
                                result = policy.predict_action(obs_dict)
                                # [修改] 获取前 steps_per_inference 步动作 (例如前6步)
                                # result['action'] shape: (B, T, D) -> (T, D)
                                action_seq = result['action'][0, :steps_per_inference].detach().cpu().numpy()
                                
                            # 3. 将动作放入队列
                            for act in action_seq:
                                action_queue.append(act)
                                
                        # 2. 执行与平滑
                        if len(action_queue) > 0:
                            # 1. 取出模型预测的下一步动作 (XY 坐标)
                            raw_action_xy = action_queue.popleft() # shape: (2,)
                            
                            # 2. 构造 3D 目标输入 (保持当前 Z 轴高度不变)
                            # 因为 policy_pos_filter 是按 3D 初始化和重置的
                            raw_target_3d = np.array([raw_action_xy[0], raw_action_xy[1], 0.97])
                            
                            # 3. 绝对位置滤波 (OneEuroFilter)
                            # 输入绝对坐标，输出平滑后的绝对坐标
                            smooth_pos_3d = policy_pos_filter(current_time, raw_target_3d)
                            
                            # 4. 速度限幅 (Velocity Limiting / Step Clipping)
                            # 计算“平滑后的新目标”与“当前目标”的距离
                            # 如果变化量超过物理极限（例如 0.05m/帧），则进行截断
                            max_step_dist = 0.02  # 单步最大移动距离 (5cm)
                            delta_pos = smooth_pos_3d[:2] - target_pose[:2]
                            dist = np.linalg.norm(delta_pos)
                            
                            if dist > max_step_dist:
                                delta_pos = delta_pos / dist * max_step_dist
                                smooth_pos_3d[:2] = target_pose[:2] + delta_pos
                            
                            # 5. 空间限幅 (Workspace Clipping)
                            # 根据 PushT 环境的大致范围限制 XY，防止飞出工作区
                            smooth_pos_3d[0] = np.clip(smooth_pos_3d[0], 0, 0.8) # X轴范围
                            smooth_pos_3d[1] = np.clip(smooth_pos_3d[1], -0.6, 0.6) # Y轴范围
                            
                            # 6. 更新 target_pose
                            target_pose[:2] = smooth_pos_3d[:2]

                            # Z 轴单独限幅 (保持在安全高度)
                            target_pose[2] = np.clip(smooth_pos_3d[2], 0.9, 1.0)
                        
                        # 4. 重要：同步滤波器状态，以便切换回手动时衔接自然
                        # 这样手动控制会从 AI 停下的地方开始，而不是从上次鼠标的位置跳过去
                        # 我们这里假设 AI 是瞬移，所以 velocity=0
                        # 或者我们可以不做这一步，但切换回 Manual 的第一帧可能会有跳变
                        # 简单做法：不做处理，但在切换 Manual 的按键事件里 reset 了 filter
                        
                    else:
                        # MANUAL 控制逻辑 (保持原有) 
                        # 获取的是速度值：[vx, vy, vz, vr, vp, vyaw]
                        raw_state = sm.get_motion_state_transformed()
                        raw_dpos = raw_state[:3] * (env.max_pos_speed / frequency) #速度×时间
                        raw_drot = raw_state[3:] * (env.max_rot_speed / frequency)
                        
                        smooth_dpos = pos_filter(current_time, raw_dpos)
                        smooth_drot = rot_filter(current_time, raw_drot)

                        smooth_dpos = np.clip(smooth_dpos, -0.02, 0.02)        

                        if np.linalg.norm(smooth_dpos) < 0.0005: smooth_dpos = np.zeros(3)
                        if np.linalg.norm(smooth_drot) < 0.0005: smooth_drot = np.zeros(3)
                        
                        target_pose[:3] += smooth_dpos
                        drot = st.Rotation.from_euler('xyz', smooth_drot)
                        curr_rot = st.Rotation.from_rotvec(target_pose[3:])
                        target_pose[3:] = (drot * curr_rot).as_rotvec()
                        target_pose[2] = np.clip(target_pose[2], 0.8, 1.1)

                    # 执行动作
                    env.exec_actions([target_pose])
                    viewer.sync()
                    
                    # 渲染窗口
                    vis_img = env.last_vis_img.copy()
                    
                    # 绘制状态信息
                    mode_text = f"MODE: {control_mode}"
                    mode_color = (203, 59, 46) if control_mode == "POLICY" else (255, 0, 0)
                    cv2.putText(vis_img, mode_text, (160, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
                    
                    if is_success :
                        color = (113, 153, 83) 
                        text = "SUCCESS!"
                    else :
                        color = (203, 59, 46) 
                        text = "FAIL!"
                    cv2.putText(vis_img, text, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if env.is_recording:
                        run_time = current_time - env.t_record_start
                        time_text = f"{run_time:.1f} s"
                        color = (203, 59, 46)
                    else:
                        run_time = current_time - t_start
                        time_text = f"{run_time:.1f} s"
                        color = (0, 0, 0)
                    cv2.putText(vis_img, time_text, (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 4)
                
                    if env.is_recording:
                        cv2.circle(vis_img, (20, 20), 10, (0, 0, 255), -1)
                        cv2.putText(vis_img, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    cv2.imshow(robot_window, vis_img)
                    
                    control_img = control_bg.copy()
                    sm.draw_feedback(control_img)
                    cv2.imshow(control_window, control_img)
                    cv2.pollKey() 

                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                env.renderer.close()
                cv2.destroyAllWindows()

if __name__ == '__main__':                       
    main()