from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class MujocoPushTImageDataset(BaseImageDataset):
    def __init__(self,
            dataset_path, # [修改] 这里必须叫 dataset_path，与 yaml 对应 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            shape_meta=None,  # [新增] 接收 shape_meta 防止报错
            **kwargs          # [新增] 接收所有其他未定义的参数，防止报错
            ):
        
        super().__init__()
        # [修改 1] 对齐 Zarr 中的 Key
        self.replay_buffer = ReplayBuffer.copy_from_path(
            dataset_path, keys=['camera_0', 'camera_1', 'robot_eef_pose', 'action'])
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes,
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # [修改 2] 适配 Normalizer 的 Key
        data = {
            'action': self.replay_buffer['action'],
            'robot_eef_pose': self.replay_buffer['robot_eef_pose']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为两个相机分别设置图像归一化器 (通常是 identity 映射，因为已经在 _sample_to_data 中除以 255 了)
        normalizer['camera_0'] = get_image_range_normalizer()
        normalizer['camera_1'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # [修改 3] 数据处理逻辑对齐
        
        # 1. 处理状态数据
        # Zarr 中 robot_eef_pose 已经是 (T, 2) float64
        robot_eef_pose = sample['robot_eef_pose'].astype(np.float32)
        
        # 2. 处理图像数据
        # Zarr 中 camera_0 是 (T, 3, 240, 320) uint8
        # 不需要 np.moveaxis，因为已经是 Channel First
        camera_0 = sample['camera_0'].astype(np.float32) / 255.0
        camera_1 = sample['camera_1'].astype(np.float32) / 255.0
        
        # 3. 组装 data 字典
        # 这里的 keys 必须与 Config 文件中 shape_meta 定义的 keys 一致
        data = {
            'obs': {
                'camera_0': camera_0,      # T, 3, 240, 320
                'camera_1': camera_1,      # T, 3, 240, 320
                'robot_eef_pose': robot_eef_pose, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data