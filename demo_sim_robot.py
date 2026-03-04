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

# 复用 diffusion_policy 现有的工具
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.keystroke_counter import KeystrokeCounter, Key, KeyCode

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
        """
        min_cutoff: 最小截止频率 (Hz)。越小越平滑，但在慢速时延迟越大。推荐 1.0 或 0.5。
        beta: 速度系数。越大越灵敏 (减少高速时的延迟)，但可能引入噪音。推荐 0.007 ~ 0.1。
        d_cutoff: 导数截止频率 (Hz)。通常设为 1.0 即可。
        """
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
        """
        t: 当前时间戳 (秒)
        x: 当前观测值 (位置或角度)
        """
        t_e = t - self.t_prev
        
        # 防止时间倒流或重复
        if t_e <= 0: return self.x_prev

        # 计算信号变化率 (速度) 的滤波
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # 动态计算主信号的截止频率
        # 速度越快 (abs(dx_hat)越大)，cutoff 越高，滤波越弱，延迟越小
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        
        # 对主信号滤波
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        # 更新状态
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        
        return x_hat
    
    def reset(self, t0, x0):
        self.x_prev = np.array(x0, dtype=np.float64)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t0

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
        self.Tblock_random_init=True #初始化Tblock位姿
        
        # 获取 body ID，用于检测Tblock是否推成功且复位
        self.tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pusht_block')
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal_target')
        self.bullseye_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'bullseye_target')
        # Mocap ID
        mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap_target')
        if mocap_body_id == -1: raise ValueError("XML Error: mocap_target missing")
        self.mocap_id = self.model.body_mocapid[mocap_body_id]
        
        # [新增] 找到 T-block 的关节地址 (用于随机化)
        block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pusht_block')
        if block_body_id != -1:
            joint_addr = self.model.body_jntadr[block_body_id]
            self.block_qpos_addr = self.model.jnt_qposadr[joint_addr]
        else:
            print("Warning: pusht_block not found, randomization disabled.")
            self.block_qpos_addr = -1
        
        # Buffer
        self.output_dir = output_dir
        if not self.output_dir.endswith('.zarr'): self.output_dir += '.zarr'

        # 视频保存路径配置
        # 视频将保存在 /data/videos/
        self.zarr_path = pathlib.Path(self.output_dir)
        self.video_root_dir = self.zarr_path.parent.joinpath('videos')
        self.video_root_dir.mkdir(parents=True, exist_ok=True)
        self.video_writers = {} # 存储 VideoWriter 对象

        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path=self.output_dir, mode='a')
        self.current_episode = {}
        self.is_recording = False #录制标志位
        self.this_video_dir = None # 初始化该变量，防止未录制直接保存时报错
        self.t_record_start = 0.0 # 用于记录录制开始的时间戳
        # 初始化时随机重置一次
        self.reset_env()

        self.max_pos_speed = 0.5 
        self.max_rot_speed = 1.0

    # [新增] 核心重置函数
    def reset_env(self):
        # 1. 重置物理状态
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        
        # 2. 同步 Mocap 防止瞬移
        self._sync_mocap_to_robot()
        
        # 3. 随机化 T-Block
        if self.block_qpos_addr != -1 and self.Tblock_random_init:
            #Tblock在桌面的位置范围
            rand_x = np.random.uniform(0.40, 0.60)  
            rand_y = np.random.uniform(-0.35, -0.15)
            fixed_z = 0.95 
            
            rand_yaw = np.random.uniform(0.5, 3.5)
            rot = st.Rotation.from_euler('z', rand_yaw) #设置欧拉角
            quat_xyzw = rot.as_quat() #转换为四元数
            mj_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            start_idx = self.block_qpos_addr
            self.data.qpos[start_idx : start_idx+3] = [rand_x, rand_y, fixed_z]
            self.data.qpos[start_idx+3 : start_idx+7] = mj_quat
        else:
            target_pos = [0.50, -0.25, 0.95] 
            
            # 设置欧拉角 (例如 0, 0, 3) -> 转为四元数
            target_euler = [0, 0, 2.5] 
            
            # 转换旋转
            rot = st.Rotation.from_euler('xyz', target_euler)
            quat_xyzw = rot.as_quat()
            # MuJoCo 四元数顺序是 [w, x, y, z]
            mj_quat = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            #固定Tblock位姿
            start_idx = self.block_qpos_addr
            self.data.qpos[start_idx : start_idx+3] = target_pos
            self.data.qpos[start_idx+3 : start_idx+7] = mj_quat
        # 4. 刷新并移动机械臂到初始高空点
        mujoco.mj_forward(self.model, self.data)
        self._move_robot_to_start_pose()
        print(f">>> Environment Reset Done.")

    def _sync_mocap_to_robot(self):
        tool_pusher = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[tool_pusher]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[tool_pusher]
        mujoco.mj_forward(self.model, self.data)

    def _move_robot_to_start_pose(self):
        # 移动到桌子上方圆圈处准备
        target_pos = np.array([0.3, -0.2, 0.96]) 
        rot = st.Rotation.from_euler('xyz', [-180, 0, -90], degrees=True)
        quat_xyzw = rot.as_quat() # Scipy 输出是 [x, y, z, w]
        target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]) # MuJoCo 需要 [w, x, y, z]，所以需要手动重排顺序
        
        self.data.mocap_pos[self.mocap_id] = target_pos
        self.data.mocap_quat[self.mocap_id] = target_quat
        
        for _ in range(500): 
            mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        obs = {}
        # 渲染图像
        self.renderer.update_scene(self.data, camera="Wrist_Camera")
        camera_0 = self.renderer.render() # (H, W, 3) RGB
        obs['camera_0'] = np.moveaxis(camera_0, -1, 0) # 转为 (3, H, W) 给 Zarr
        
        self.renderer.update_scene(self.data, camera="Front_Camera")
        camera_1 = self.renderer.render() # (H, W, 3) RGB
        obs['camera_1'] = np.moveaxis(camera_1, -1, 0) # 转为 (3, H, W) 给 Zarr
        
        # 仅用于可视化窗口显示
        self.renderer.update_scene(self.data, camera="Top_down_Camera")
        camera_2 = self.renderer.render() # (H, W, 3) RGB
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

            # 将 obs 中的 Channel-First (3,H,W) 数据转换回 Channel-Last (H,W,3)
            # 并且从 RGB 转换为 BGR (OpenCV 格式)
            if self.video_writers:
                # 写入 camera_0 (对应 MP4 文件名 0.mp4)
                img_1_rgb = np.moveaxis(obs['camera_0'], 0, -1) # (H,W,3)
                img_1_bgr = cv2.cvtColor(img_1_rgb, cv2.COLOR_RGB2BGR)
                self.video_writers['0'].write(img_1_bgr)
                
                # 写入 camera_1 (对应 MP4 文件名 1.mp4)
                img_3_rgb = np.moveaxis(obs['camera_1'], 0, -1)
                img_3_bgr = cv2.cvtColor(img_3_rgb, cv2.COLOR_RGB2BGR)
                self.video_writers['1'].write(img_3_bgr)

            # 获取机械臂关节信息 (这里 UR5对应2-8关节位置ID)
            current_joint = self.data.qpos[2:8].copy()
            current_joint_vel = self.data.qvel[2:8].copy()
            data = {
                'action': target_pose[:2].copy(),
                'robot_eef_pose': target_pose[:2].copy(),
                'camera_0': obs['camera_0'],
                'camera_1': obs['camera_1'],
                'raw_action': target_pose[:].copy(),
                'raw_robot_eef_pose': target_pose[:].copy(),
                'robot_joint': current_joint,          # (6,) 关节角度
                'robot_joint_vel': current_joint_vel,  # (6,) 关节速度
                'timestamp': np.array([time.time()])
            }
            for key, value in data.items():
                if key not in self.current_episode: self.current_episode[key] = []
                self.current_episode[key].append(value)

    def start_episode(self):
        self.is_recording = True
        self.current_episode = {}
        self.t_record_start = time.monotonic() # 记录按下开始键的瞬间时间

        # 创建当前 Episode 的视频文件夹
        episode_id = self.replay_buffer.n_episodes
        self.this_video_dir = self.video_root_dir.joinpath(str(episode_id))
        self.this_video_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 VideoWriters
        # 对应 real_env，保存 camera_0 和 camera_1
        # fps 使用 self.frequency
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 初始化 camera_0 写入器
        path_1 = str(self.this_video_dir.joinpath('0.mp4'))
        self.video_writers['0'] = cv2.VideoWriter(path_1, fourcc, self.frequency, (self.render_width, self.render_height))
        
        # 初始化 camera_1 写入器
        path_3 = str(self.this_video_dir.joinpath('1.mp4'))
        self.video_writers['1'] = cv2.VideoWriter(path_3, fourcc, self.frequency, (self.render_width, self.render_height))
        print("Dataset Recording Started...")

    def end_episode(self):
        self.is_recording = False

        # 计算时长
        current_duration = time.monotonic() - self.t_record_start
        
        # 关闭视频写入器
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}

        # 大于30秒的录像不保存并返回false
        if current_duration > 30.0:
            print(f"\n[Timeout] 录制时长 {current_duration:.1f}s > 30s。删除数据并重置环境...")
            # 1删除视频文件夹
            if self.this_video_dir is not None and self.this_video_dir.exists():
                shutil.rmtree(str(self.this_video_dir))
            self.current_episode = {} #清空内存中的动作缓存，防止坏数据残留
            # 返回 False，告诉 main 函数这次没存，且环境已重置
            return False

        # ==========================================
        # 正常保存逻辑 (小于等于 30s)
        # ==========================================
        if self.current_episode:
            episode_data = {k: np.array(v) for k, v in self.current_episode.items()}
            self.replay_buffer.add_episode(episode_data)
            # 保存成功后，必须清空内存缓存 (释放 RAM)
            self.current_episode = {}
            # 保存成功后，切断对该文件夹的引用，防止误删
            self.this_video_dir = None
            print(f"Episode Saved! Total Episodes: {self.replay_buffer.n_episodes}")
            print("\n[Auto Save] 任务完成且机器人已复位，自动保存数据且重置环境！")
            return True # 返回成功
        else:
            # 只有当 this_video_dir 存在且不为 None 时才尝试删除
            if self.this_video_dir is not None and self.this_video_dir.exists():
                shutil.rmtree(str(self.this_video_dir))

            self.current_episode = {}
            self.this_video_dir = None
            return False

    def drop_episode(self):
        self.is_recording = False
        self.current_episode = {}
        
        # 关闭并删除视频
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}
        
        # 增加 None 检查
        if self.this_video_dir is not None and self.this_video_dir.exists():
            shutil.rmtree(str(self.this_video_dir))
            print(f"Episode Dropped. Video folder deleted: {self.this_video_dir}")
        else:
            print("Episode Dropped (No video folder to delete).")
        # 清空动作缓存
        self.current_episode = {}
        
        # 切断引用
        self.this_video_dir = None
    #  成功检测函数
    def check_conditions(self, block_tol=0.05, robot_tol=0.05):
        """
        返回两个标志位:
        is_success: T-block 是否到达目标
        is_robot_reset: 机器人是否回到 Bullseye
        """
        # 1. 检查 T-block 是否成功 (距离 + 角度)
        block_pos = self.data.xpos[self.block_body_id][:2]
        goal_pos = self.data.xpos[self.goal_body_id][:2]
        block_dist = np.linalg.norm(block_pos - goal_pos)
        
        # 计算角度误差 (Yaw)
        block_quat = self.data.xquat[self.block_body_id]
        goal_quat = self.data.xquat[self.goal_body_id]
        block_yaw = st.Rotation.from_quat([block_quat[1], block_quat[2], block_quat[3], block_quat[0]]).as_euler('zyx')[0]
        goal_yaw = st.Rotation.from_quat([goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]]).as_euler('zyx')[0]
        rot_error = np.abs((block_yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi)
        
        is_success = (block_dist < block_tol) and (rot_error < 0.03) # 角度容差设宽一点方便操作

        # 2. 检查 Robot 是否复位 (是否到达 Bullseye)
        tool_pos = self.data.xpos[self.tool_body_id][:2]
        
        # 获取 Bullseye 坐标
        target_pos = np.array([0, 0]) # 默认原点
        if self.bullseye_body_id != -1:
            target_pos = self.data.xpos[self.bullseye_body_id][:2]
        elif self.bullseye_site_id != -1:
            target_pos = self.data.site_xpos[self.bullseye_site_id][:2]
            
        robot_dist = np.linalg.norm(tool_pos - target_pos)
        is_robot_reset = robot_dist < robot_tol

        return is_success, is_robot_reset

@click.command()
@click.option('--output', '-o', default='./data/demo_pusht_mujoco_plus/replay_buffer', help="Directory to save dataset.")
@click.option('--xml_path', '-x', default='./diffusion_policy/env/pusht/mujoco_ur5e/pusht.xml', help="Path to your Mujoco XML.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
def main(output, xml_path, frequency=10):
    dt = 1/frequency
    
    with KeystrokeCounter() as key_counter:
        control_window = 'Large Control Pad'
        control_w, control_h = 1200, 900
        cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(control_window, control_w, control_h) 
        
        robot_window = 'Robot View'
        cv2.namedWindow(robot_window, cv2.WINDOW_AUTOSIZE)
        
        with VirtualSpaceMouse(window_name=control_window, sensitivity=3.0) as sm:
            
            env = SimEnv(xml_path, output, frequency=frequency)

            # ==========================================
            # 初始化 One Euro 滤波器
            # ------------------------------------------
            # min_cutoff=0.1: 在你手停住时，过滤掉 0.1Hz 以上的抖动 (超级稳)
            # beta=0.05: 当你快速移动时，灵敏度提升
            # ==========================================
            t_start = time.monotonic()
            # 初始值为 0 (因为我们滤波的是 dpos 增量)
            pos_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)
            rot_filter = OneEuroFilter(t0=t_start, x0=np.zeros(3), min_cutoff=0.5, beta=0.1)

            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                
                # 初始化视角
                viewer.cam.lookat = np.array([0.5, 0, 0.8])
                viewer.cam.distance = 2.0
                viewer.cam.azimuth = 180 #水平转角
                viewer.cam.elevation = -90  #平视0~俯视90
                
                print(f"   Ready! 控制方式:")
                print(f"  [Move] 鼠标左键拖动 (大黑窗口)")
                print(f"  [Rec]  1:开始录制 | 2:保存 | 空格:删除并重置")
                print(f"  [Reset] 0: 不打断录制随机重置环境")
                
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                
                iter_idx = 0
                stop = False
                t_start = time.monotonic()
                
                control_bg = np.zeros((control_h, control_w, 3), dtype=np.uint8)
                
                while not stop and viewer.is_running():
                    current_time = time.monotonic() # 获取当前精确时间
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    
                    # 每一帧都获取 obs，如果 is_recording 为 True，内部会自动写入视频
                    # env.get_obs() # <--- 这一行删掉，移到 exec_actions 内部处理，或者只用于显示

                    # 获取状态标志位，判断推T是否成功
                    is_success, is_robot_reset = env.check_conditions(block_tol=0.005, robot_tol=0.008)
                    press_events = key_counter.get_press_events()

                    if is_success and is_robot_reset:
                        # 调用结束函数，是否成功保存数据
                        env.end_episode()
                        env.reset_env() 
                        # 同步控制机械臂状态
                        pos_filter.reset(current_time, np.zeros(3))
                        rot_filter.reset(current_time, np.zeros(3))
                        state = env.get_robot_state()
                        target_pose = state['TargetTCPPose']
                        # env.start_episode() # 甚至可以直接开始下一条录制
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'): stop = True
                        elif key_stroke == KeyCode(char='1'): env.start_episode()
                        elif key_stroke == KeyCode(char='2'): 
                            env.end_episode()
                            env.reset_env() 
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3))
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                        elif key_stroke == Key.space: 
                            # 只有在“正在录制”时，才执行删除操作
                            if env.is_recording:
                                env.drop_episode()
                            
                            # 无论是否录制，Space键都会重置环境位置
                            env.reset_env()
                            
                            # 重置时，也要重置滤波器状态，否则机械臂会乱飘
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3))
                            
                            # 重置后，必须同步更新 target_pose
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']
                        elif key_stroke == KeyCode(char='0'): #可以不打断录制进行复位
                            env.reset_env()
                            # 重置时，也要重置滤波器状态，否则机械臂会乱飘
                            pos_filter.reset(current_time, np.zeros(3))
                            rot_filter.reset(current_time, np.zeros(3)) 
                            
                            state = env.get_robot_state()
                            target_pose = state['TargetTCPPose']

                    # 获取当前画面用于显示 (如果正在录制，显示 REC)
                    # 我们需要手动调用一次 get_obs 来更新 renderer，但不需要返回数据给 Zarr
                    # 仅为了 last_vis_img 更新
                    if not env.is_recording:
                        env.get_obs()
                    
                    # 渲染窗口
                    vis_img = env.last_vis_img.copy()
                    # ==========================================
                    # 在左下方绘制成功信息
                    # ==========================================
                    if is_success :
                        # 成功时显示绿色文字
                        color = (113, 153, 83) # Green
                        text = "SUCCESS!"
                    else :
                        color = (203, 59, 46) # red
                        text = "FAIL!"
                    # 将文字画在 Robot View 上
                    cv2.putText(vis_img, text, (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # ==========================================
                    # 在右上方绘制时间信息
                    # ==========================================
                    # 计算运行时间
                    if env.is_recording:
                        # 如果正在录制：当前时间 - 录制开始时间
                        run_time = current_time - env.t_record_start
                        time_text = f"{run_time:.1f} s"
                        color = (203, 59, 46)
                    else:
                        # 如果没录制：显示整体时间线
                        run_time = current_time - t_start
                        time_text = f"{run_time:.1f} s"
                        color = (0, 0, 0)
                    # 绘制文字
                    cv2.putText(vis_img, time_text, (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 4)
                
                    if env.is_recording:
                        cv2.circle(vis_img, (20, 20), 10, (0, 0, 255), -1)
                        cv2.putText(vis_img, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imshow(robot_window, vis_img)
                    
                    control_img = control_bg.copy()
                    sm.draw_feedback(control_img)
                    cv2.imshow(control_window, control_img)
                    cv2.pollKey() 

                    # 获取原始鼠标输入 (Raw Input)
                    raw_state = sm.get_motion_state_transformed()
                    raw_dpos = raw_state[:3] * (env.max_pos_speed / frequency)
                    raw_drot = raw_state[3:] * (env.max_rot_speed / frequency)
                    


                    # 注意：OneEuroFilter 需要传入当前时间戳
                    smooth_dpos = pos_filter(current_time, raw_dpos)
                    smooth_drot = rot_filter(current_time, raw_drot)

                    # # 再次添加最大速度截断 (Clamping)
                    # # 防止鼠标甩太快导致物理崩坏 (例如限制单帧最大移动 2cm)
                    smooth_dpos = np.clip(smooth_dpos, -0.02, 0.02)        

                    # 死区截断 (可选): 如果数值太小直接置零，进一步消除静止抖动
                    if np.linalg.norm(smooth_dpos) < 0.0005: smooth_dpos = np.zeros(3)
                    if np.linalg.norm(smooth_drot) < 0.0005: smooth_drot = np.zeros(3)

                    
                    target_pose[:3] += smooth_dpos
                    drot = st.Rotation.from_euler('xyz', smooth_drot)
                    curr_rot = st.Rotation.from_rotvec(target_pose[3:])
                    target_pose[3:] = (drot * curr_rot).as_rotvec()
                    target_pose[2] = np.clip(target_pose[2], 0.8, 1.1)

                    # 执行动作 (内部包含录制逻辑)
                    env.exec_actions([target_pose])
                    viewer.sync()
                    
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                env.renderer.close()
                cv2.destroyAllWindows()

if __name__ == '__main__':                       
    main()