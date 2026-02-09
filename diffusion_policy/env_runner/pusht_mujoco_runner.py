import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import cv2
import mujoco
import scipy.spatial.transform as st
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

# =============================================================================
# 1. 辅助类：ObsBuffer (保持不变)
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
        while len(self.buffer) < self.n_obs_steps:
            self.buffer.appendleft(self.buffer[0])
            
        stacked = {}
        keys = self.buffer[0].keys()
        
        for k in keys:
            if k == 'timestamp': continue
            val = np.stack([x[k] for x in self.buffer]) 
            
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

# =============================================================================
# 2. 仿真环境类 (保持不变)
# =============================================================================
class SimEnv:
    def __init__(self, xml_path, frequency=10):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frequency = frequency
        self.dt = 1 / frequency
        self.max_pos_speed = 0.05
        self.render_width = 320 
        self.render_height = 240
        self.renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self.Tblock_random_init = True
        
        self.mocap_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'mocap_target')
        self.mocap_id = self.model.body_mocapid[self.mocap_body_id]
        self.block_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pusht_block')
        self.goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'goal_target')
        self.tool_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'tool_pusher')
        
        if self.block_body_id != -1:
            joint_addr = self.model.body_jntadr[self.block_body_id]
            self.block_qpos_addr = self.model.jnt_qposadr[joint_addr]
        else:
            self.block_qpos_addr = -1

    def reset_env(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        
        self.data.mocap_pos[self.mocap_id] = self.data.xpos[self.tool_body_id]
        self.data.mocap_quat[self.mocap_id] = self.data.xquat[self.tool_body_id]
        
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

    def _move_robot_to_start_pose(self):
        target_pos = np.array([0.3, -0.2, 0.96]) 
        rot = st.Rotation.from_euler('xyz', [-180, 0, -90], degrees=True)
        quat_xyzw = rot.as_quat()
        target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        self.data.mocap_pos[self.mocap_id] = target_pos
        self.data.mocap_quat[self.mocap_id] = target_quat
        for _ in range(100): mujoco.mj_step(self.model, self.data)

    def get_obs(self):
        obs = {}
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="Wrist_Camera")
        obs['camera_0'] = np.moveaxis(self.renderer.render(), -1, 0)
        self.renderer.update_scene(self.data, camera="Front_Camera")
        obs['camera_1'] = np.moveaxis(self.renderer.render(), -1, 0)
        pos = self.data.mocap_pos[self.mocap_id].copy()
        obs['robot_eef_pose'] = pos[:2]
        return obs

    def exec_actions(self, action):
        current_pos = self.data.mocap_pos[self.mocap_id][:2].copy()
        target_xy = action
        delta = target_xy - current_pos
        dt = 1.0 / self.frequency
        max_step_dist = self.max_pos_speed * dt
        delta_norm = np.linalg.norm(delta)
        if delta_norm > max_step_dist:
            delta = delta / delta_norm * max_step_dist
        safe_target_xy = np.clip(current_pos + delta, [-0.5, -0.5], [1.0, 0.5])
        
        self.data.mocap_pos[self.mocap_id] = [safe_target_xy[0], safe_target_xy[1], 0.96]
        sim_steps = int(1.0 / self.frequency / self.model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)

    def check_success(self, block_tol=0.05):
        block_pos = self.data.xpos[self.block_body_id][:2]
        goal_pos = self.data.xpos[self.goal_body_id][:2]
        dist = np.linalg.norm(block_pos - goal_pos)
        
        block_quat = self.data.xquat[self.block_body_id]
        goal_quat = self.data.xquat[self.goal_body_id]
        block_yaw = st.Rotation.from_quat([block_quat[1], block_quat[2], block_quat[3], block_quat[0]]).as_euler('zyx')[0]
        goal_yaw = st.Rotation.from_quat([goal_quat[1], goal_quat[2], goal_quat[3], goal_quat[0]]).as_euler('zyx')[0]
        rot_error = np.abs((block_yaw - goal_yaw + np.pi) % (2 * np.pi) - np.pi)
        return (dist < block_tol) and (rot_error < 0.1)

    def get_vis_img(self):
        self.renderer.update_scene(self.data, camera="Top_down_Camera")
        return self.renderer.render() # RGB

# =============================================================================
# 3. Runner 主类 (修改部分)
# =============================================================================
class MujocoPushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            xml_path,
            n_train=0, 
            n_train_vis=0,
            n_test=10, 
            n_test_vis=3, 
            max_steps=1000,
            n_obs_steps=2,
            n_action_steps=8,
            fps=10,
            shape_meta=None,
            **kwargs
        ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.xml_path = xml_path
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.shape_meta = shape_meta
        self.iter_count = 0 # 内部计数器，用于记录是第几次 Run
        
        self.env = SimEnv(xml_path=xml_path, frequency=fps)

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        policy.num_inference_steps = 6 # DDIM 步数
        log_data = dict()
        all_rewards = []
        
        # 创建本地保存目录: output_dir/eval_videos/epoch_X/
        eval_epoch = self.iter_count 
        self.iter_count += 1 # 每次调用 run 自动 +1
        
        video_save_dir = pathlib.Path(self.output_dir).joinpath('eval_videos', f'epoch_{eval_epoch}')
        video_save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 循环测试 N 个 Episode
        for i in range(self.n_test):
            # ... 重置环境，运行策略 ...
            seed = 10001 + i
            self.env.reset_env(seed=seed)
            
            obs_buffer = ObsBuffer(n_obs_steps=self.n_obs_steps, shape_meta=self.shape_meta)
            obs = self.env.get_obs()
            obs_buffer.add_obs(obs)
            
            policy.reset()
            
            video_frames = []
            done = False
            success = False
            step_count = 0
            
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval Epoch {eval_epoch} | Ep {i}/{self.n_test}", leave=False)
            while step_count < self.max_steps and not done:
                obs = self.env.get_obs()
                obs_buffer.add_obs(obs)
                
                # 收集视频帧 (仅前 n_test_vis 个)
                if i < self.n_test_vis:
                    video_frames.append(self.env.get_vis_img())

                obs_dict_np = obs_buffer.get_stacked_obs()
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                
                with torch.no_grad():
                    result = policy.predict_action(obs_dict)
                    action_seq = result['action'][0].detach().cpu().numpy()
                
                for action in action_seq:
                    if step_count >= self.max_steps: break
                    self.env.exec_actions(action)
                    step_count += 1
                    
                    if self.env.check_success():
                        success = True
                        done = True 
                        break
                
                pbar.update(len(action_seq))
            pbar.close()
            
            score = 1.0 if success else 0.0
            all_rewards.append(score)
            
            log_key = f'test/sim_max_reward_{seed}'
            log_data[log_key] = score
            
            #  本地保存视频 (使用 OpenCV)
            if i < self.n_test_vis and len(video_frames) > 0:
                video_path = video_save_dir.joinpath(f'seed_{seed}_score_{score:.0f}.mp4')
                
                # 初始化 VideoWriter
                # 注意: SimEnv 返回的是 RGB，OpenCV 需要 BGR
                height, width, _ = video_frames[0].shape
                
                # [关键修改] 将 'mp4v' 改为 'avc1' (H.264编码)，VS Code 才能直接播放
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1') 
                except:
                    # 如果系统缺少 H.264 编码器，回退到 mp4v
                    print("Warning: H.264 encoder not found, falling back to mp4v")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    
                out = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
                
                for frame in video_frames:
                    # RGB -> BGR
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(bgr_frame)
                
                out.release()
                
        # 计算平均分
        log_data['test/mean_score'] = np.mean(all_rewards)
        print(f"Eval Epoch {eval_epoch} Finished. Mean Score: {log_data['test/mean_score']}")
        print(f"Videos saved to: {video_save_dir}")
        
        return log_data