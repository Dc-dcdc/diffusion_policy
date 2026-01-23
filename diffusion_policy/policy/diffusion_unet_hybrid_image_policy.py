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


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    #这里都是默认值，具体还得看配置文件中的policy部分
    def __init__(self, 
            shape_meta: dict, #形状元数据  {'action': {'shape': [2]}, 'obs': {'agent_pos': {'shape': [2], 'type': 'low_dim'}, 'image': {'shape': [3, 96, 96], 'type': 'rgb'}}}
            noise_scheduler: DDPMScheduler,
            horizon, #预测步数  16
            n_action_steps, #执行步数  8
            n_obs_steps, #观测长度  2
            num_inference_steps=None,  #推理时去噪步数 100
            obs_as_global_cond=True,  #图像作为条件注入
            crop_shape=(76, 76), #裁剪，原图96x96
            diffusion_step_embed_dim=256, #扩散时间步（Time Step）的特征向量维度
            down_dims=(256,512,1024), #下采样每一层通道数
            kernel_size=5, #卷积核，每次卷积前后5个时间步信息
            n_groups=8,  #groupnorm组数
            cond_predict_scale=True, #在 FiLM调节中，是否预测缩放系数（Scale）和偏移量
            obs_encoder_group_norm=False, #图像编码器（ResNet）内部是否使用 GroupNorm
            eval_fixed_crop=False, #True表示固定裁剪  false表示随机裁剪，以模拟相机抖动
            # parameters passed to step
            **kwargs):   #接收所有未显式列出的参数（防止配置文件里多写参数导致报错）
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape'] #获取动作维度  shape_meta = {'action': {'shape': [2]}, 'obs': {'agent_pos': {'shape': [2], 'type': 'low_dim'}, 'image': {'shape': [3, 96, 96], 'type': 'rgb'}}}
        assert len(action_shape) == 1  #要求是一维，action_shape= [2]
        action_dim = action_shape[0]  #获取具体动作维度数值，action_dim=2
        obs_shape_meta = shape_meta['obs']  #obs_shape_meta = {'agent_pos': {'shape': [2], 'type': 'low_dim'}, 'image': {'shape': [3, 96, 96], 'type': 'rgb'}},
        obs_config = {
            'low_dim': [],  #低维状态（如关节角、末端位置、速度） MLP处理
            'rgb': [],   #彩色图像   resnet/CNN处理
            'depth': [], #深度图像
            'scan': []   #激光雷达扫描数据
        }
        obs_key_shapes = dict()  #获取数据形状   {'agent_pos': [2], 'image': [3, 96, 96]}
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim') #获取 type 字段，没有则使用默认的low_dim
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config( #获取一个默认的 robomimic 配置对象
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square', #任务名称
            dataset_type='ph') #数据集类型Proficient Human，专家演示数据
        
        with config.unlocked(): #临时解锁
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None: #配置图像不裁剪
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':  #如果默认配置里开启了 CropRandomizer（随机裁剪），将其关闭
                        modality['obs_randomizer_class'] = None
            else: #裁剪
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer': #找到配置里的 CropRandomizer，并将参数改成裁剪尺寸
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config) #初始化全局观测处理工具

        # load model 
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,  #算法名称，即对应什么模型
                config=config,  #包含模型详细超参数
                obs_key_shapes=obs_key_shapes, #观测规格即输入   {'agent_pos': [2], 'image': [3, 96, 96]}
                ac_dim=action_dim, #动作规格即输出
                device='cpu',  #初始化位置
            )
        #获取观测要用的神经网络模块（视觉编码器）    这里policy，也就是robomimic 库里已经编好了各种网络模型，可以直接获取使用
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            #将批量归一化层替换为组归一化层
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        # 评估时是否固定裁剪
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

        # create diffusion model 
        obs_feature_dim = obs_encoder.output_shape()[0] #获取观测编码输出的特征向量维度  66
        input_dim = action_dim + obs_feature_dim  #老款DP  动作和观测一起输入，一起输出
        global_cond_dim = None  #不需要额外的全局条件输入端口
        if obs_as_global_cond:  #如果观测作为条件，即新款DP
            input_dim = action_dim #输入只有动作
            global_cond_dim = obs_feature_dim * n_obs_steps  #将观测作为条件进行输入，维度=观测维度 x 过去帧数  132

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,  #
            diffusion_step_embed_dim=diffusion_step_embed_dim,   #时间步编码长度
            down_dims=down_dims,  #下采样每一层通道数  [512, 1024, 2048]
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(  #掩码生成，手动遮住动作
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim, #判断观测是条件输入还是input输入
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()  #归一化到[-1,1]
        self.horizon = horizon  #预测步数
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim 
        self.n_action_steps = n_action_steps  #执行步数
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None: #推理步数
            num_inference_steps = noise_scheduler.config.num_train_timesteps #未设置则使用训练时的扩散完整步长 100
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))  #扩散模型部分参数总量  权重+偏置
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters())) #视觉编码部分参数总量 
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        #创建一个完全由随机噪声构成的初始轨迹
        trajectory = torch.randn(
            size=condition_data.shape,  #[B, T, Da] 即[Batch, horizon，action_dim]
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps) #获取推理时间步列表，从大到小    
        for t in scheduler.timesteps: #推理步数
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask] #Global模式不影响  Inpainting对观测部分采样真实值进行覆盖，表示只预测动作部分

            # 2. predict model output # 调用forward，预测的噪声值
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond) #噪声轨迹、时间步、全局条件 -> 预测噪声   output: [B, T, Da]  即[Batch, horizon，action_dim]

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step( #执行单步去噪
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory #[B, T, Da] 即[Batch, horizon，action_dim]


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values())) #获取字典里所有的“值”，排列后取出第一个 nobs = {'image': Tensor(shape=[batch, n_obs_steps, 3, 96, 96]),  'agent_pos': Tensor(shape=[batch, n_obs_steps, 2]) }
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])) #合并To=2帧 观测数据 [B, 2, 3, 96, 96] -> [B*2, 3, 96, 96]
            nobs_features = self.obs_encoder(this_nobs) # [B*T, 3, 96, 96] + [B*2, 2] -> [B*T,66]
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1) #[B*T,66] -> [B,66*T]
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else: #Inpainting (修补)
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(  #采样结果[B, T, Da] 即[Batch, horizon，action_dim]
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]  #提取动作部分   主要是为了取出Inpainting模式下的动作，除去观测
        action_pred = self.normalizer['action'].unnormalize(naction_pred) #反归一化  返回真实动作值

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]  #提取n_action_steps步信息
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    # 将数据集归一化好的参数 加载 到策略中
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch #防止二次加工
        nobs = self.normalizer.normalize(batch['obs']) #归一化
        nactions = self.normalizer['action'].normalize(batch['action']) #归一化  形状为[Batch, horizon, Dim]
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1] #序列长度

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond: #观测作为“全局指令”
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])) #合并n_obs_steps=2帧 观测数据 [B, 2, 3, 96, 96] -> [B*2, 3, 96, 96]
            nobs_features = self.obs_encoder(this_nobs)  #特征提取  [B*2, 66]
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1) #[Batch, (2 * feature_dim)]，把n_obs_steps帧数据连在一起，对齐批次
        else: #观测作为“序列一部分”
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1) #拼接  Action + Obs
            trajectory = cond_data.detach() #对轨迹进行覆盖

        # generate impainting mask
        #生成修补掩码
        condition_mask = self.mask_generator(trajectory.shape) 

        # Sample noise that we'll add to the images
        # 生成一个与 trajectory（轨迹数据）形状完全相同的张量，里面的数值全部是从 标准正态分布N(0,1)中随机抽取的
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]  #提取Batch Size数值，并为每条轨迹随机抽取时间步
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise( #一步加噪公式进行加噪
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask #按位取反，将预测内容作为计算项   

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]  #对于Action + Obs，需要用 干净的obs 对 加噪的obs 进行覆盖
        
        # Predict the noise residual 
        # 调用forward，预测的噪声值
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond) #输入带噪声的轨迹，时间步，全局条件

        pred_type = self.noise_scheduler.config.prediction_type #获取预测类型
        if pred_type == 'epsilon': #预测噪声
            target = noise #目标就是添加的噪声
        elif pred_type == 'sample': #预测动作
            target = trajectory #目标就是原轨迹
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none') #计算均方误差，保留张量，禁止合并
        loss = loss * loss_mask.type(loss.dtype) #剔除已知条件部分
        loss = reduce(loss, 'b ... -> b (...)', 'mean') #计算每条轨迹loss均值
        loss = loss.mean() #计算所有轨迹loss均值
        return loss
