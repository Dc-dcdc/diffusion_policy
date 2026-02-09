from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetMujocoImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict, 
            noise_scheduler: DDPMScheduler,
            horizon=16,          # 预测步数 (PushT 标准为 16)
            n_action_steps=8,    # 执行步数 (PushT 标准为 8)
            n_obs_steps=2,       # 观测历史长度 (PushT 标准为 2)
            num_inference_steps=None, 
            obs_as_global_cond=True, # 关键：将图像特征作为 Global Condition 注入 U-Net
            crop_shape=(216, 288),     # 训练数据增强：如果输入是 240x320，随机裁剪到 [216, 288] 能防止过拟合
            diffusion_step_embed_dim=256, 
            down_dims=(256,512,1024), 
            kernel_size=5, 
            n_groups=8, 
            cond_predict_scale=True, 
            obs_encoder_group_norm=False, 
            eval_fixed_crop=True,    # 评估/推理时是否使用中心裁剪 (建议为 True 以保持一致性)
            **kwargs):
        super().__init__()

        # 1. 解析 shape_meta
        # 您的 shape_meta 包含: camera_0, camera_1 (rgb) 和 robot_eef_pose (low_dim)
        action_shape = shape_meta['action']['shape'] 
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [], 
            'rgb': [], 
            'depth': [], 
            'scan': [] 
        }
        obs_key_shapes = dict() 
        
        # 自动分类 obs 中的 key (rgb vs low_dim)
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key) # 这里会自动加入 'camera_0', 'camera_1'
            elif type == 'low_dim':
                obs_config['low_dim'].append(key) # 这里会自动加入 'robot_eef_pose'
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # 2. 配置 Robomimic 视觉编码器 (ResNet)
        # 即使是 MuJoCo 任务，我们也复用 Robomimic 强大的视觉 Backbone
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square', 
            dataset_type='ph')
        
        with config.unlocked():
            config.observation.modalities.obs = obs_config

            # 配置随机裁剪增强
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # 训练时：随机裁剪
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # 初始化 ObsUtils
        ObsUtils.initialize_obs_utils_with_config(config)

        # 3. 创建编码器模型
        # 这里会自动处理多相机输入：camera_0 和 camera_1 的特征会被提取并拼接
        policy_algo: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )
        obs_encoder = policy_algo.nets['policy'].nets['encoder'].nets['obs']
        
        # 替换 BatchNorm 为 GroupNorm (对于小 Batch Size 训练更稳定)
        if obs_encoder_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )

        # 处理评估时的裁剪逻辑
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # 4. 创建 Diffusion U-Net 模型
        obs_feature_dim = obs_encoder.output_shape()[0] # 视觉特征 + 低维特征的总维度
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None

        if obs_as_global_cond:
            # 标准做法：U-Net 输入只有 Action，观测作为 Global Condition
            input_dim = action_dim 
            # Global Condition = 单帧特征维度 * 观测历史长度
            global_cond_dim = obs_feature_dim * n_obs_steps 

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # 掩码生成器 (Inpainting 模式用，Global Cond 模式下不太重要但需保留接口)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        if 'obs_encoder' in kwargs:
            kwargs.pop('obs_encoder')
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= 推理 (Inference) ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        # 初始化随机高斯噪声轨迹
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # 设置推理时间步 (例如 100 -> 0)
        scheduler.set_timesteps(self.num_inference_steps)
        
        for t in scheduler.timesteps:
            # 1. 强制覆盖已知条件 (Inpainting 模式才有用，Global Cond 模式 condition_mask 全为 False)
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. 预测噪声 residual
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. 去噪一步: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # 最后再次覆盖条件，确保精确性
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        输入: obs_dict (包含 'camera_0', 'camera_1', 'robot_eef_pose')
        输出: action (反归一化后的真实动作)
        """
        assert 'past_action' not in obs_dict
        
        # 1. 归一化输入
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values())) 
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # 2. 准备条件输入
        local_cond = None
        global_cond = None
        
        if self.obs_as_global_cond:
            # 提取 n_obs_steps 帧的特征并展平
            # shape: [B, T, C, H, W] -> [B*T, C, H, W]
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            
            # 通过 ResNet 提取特征 -> [B*T, FeatureDim]
            nobs_features = self.obs_encoder(this_nobs)
            
            # 重塑回 [B, T*FeatureDim] 作为 Global Condition
            global_cond = nobs_features.reshape(B, -1)
            
            # 动作部分的初始输入为空
            cond_data = torch.zeros(size=(B, T, Da), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # Inpainting 模式 (备用)
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=self.device, dtype=self.dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # 3. 运行扩散采样
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # 4. 提取动作并反归一化
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # 5. 截取需要执行的步数
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        return {
            'action': action,
            'action_pred': action_pred
        }

    # ========= 训练 (Training) ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # 1. 归一化输入
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        trajectory = nactions
        cond_data = trajectory

        # 2. 提取图像和低维状态特征
        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # 拼接多帧特征
            global_cond = nobs_features.reshape(batch_size, -1)
            local_cond = None
        else:
            # Inpainting 模式
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, self.horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
            global_cond = None
            local_cond = None

        # 3. 扩散前向过程 (加噪)
        # 生成噪声
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # 随机采样时间步 t
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # 加噪: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * epsilon
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # 4. 处理条件掩码 (Global Cond 模式下，condition_mask 全为 False)
        condition_mask = self.mask_generator(trajectory.shape)
        loss_mask = ~condition_mask
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # 5. 模型预测噪声
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # 6. 计算损失 (MSE Loss)
        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        
        return loss