from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1] #每条数据在总数据中的起始索引
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1): #[-1,episode_length-16+7]
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)  #实际读的起点 - 想要读的起点   想读 idx = -1，实际读了 0  $0 - (-1) = 1$，左边少读了 1 帧，后续在前面需要补一个pad
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx # 想要读的终点 - 实际读的终点
            sample_start_idx = 0 + start_offset #真数据填充起点
            sample_end_idx = sequence_length - end_offset #真数据填充终点
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

# 样本的索引值是累加的   读到样本末尾时，会使用padding填充，不会和下一条样本进行混合读取
class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            #                      在样本中索引起点     在样本中索引终点   真实样本填充起点     真实样本填充终点
            # 生成索引，返回每个样本的[buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]   例[88, 100, 0, 12] 12<16 则后面补pad   [0, 4, 1, 5]，这里1>0,前面要补pad
            indices = create_indices(
                episode_ends,  #每条轨迹的终点，防止跨界 例：[163,305,454,612,751]
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            #低维动作数据、agentpos 16帧全部读取
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            #图像数据，只读前2帧，提升计算速度
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data) #配置长度(2)和理论长度(16)取小的
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data] #取前两帧，剩余为nan
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                # 1. 创建一个全黑的画布 (zeros)
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                # 2. 头部填充 (Pad Before)
                if sample_start_idx > 0:#
                    data[:sample_start_idx] = sample[0] # 复制第一帧
                # 3. 尾部填充 (Pad After)
                if sample_end_idx < self.sequence_length:#数据不够长
                    data[sample_end_idx:] = sample[-1] # 复制最后一帧，填满剩下位置
                # 4. 填入真实数据
                data[sample_start_idx:sample_end_idx] = sample  #填入剩余数据，组成完整数据
            result[key] = data
        return result
