import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply #对字典中元素进行批量处理
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,  #训练集评估次数
            n_train_vis=3, #评估后存下的视频数
            train_start_seed=0,
            n_test=22, #测试集评估次数
            n_test_vis=6,  #存下视频数
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22, #视频压缩质量  数值越小画质越好
            render_size=96, #渲染分辨率
            past_action=False, #是否输入历史动作
            tqdm_interval_sec=5.0, #进度条刷新间隔
            n_envs=None #并行环境数量
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test
        #运行多少步保存一帧视频画面
        steps_per_render = max(10 // fps, 1)
        def env_fn():
                # 序列对齐
            return MultiStepWrapper(
                # 视频录制
                VideoRecordingWrapper(
                    # 物理仿真 也就是env
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis #只录像前n_train_vis次

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop() #以此防止上一轮没关干净
                env.env.file_path = None
                if enable_render:
                    # 生成唯一文件名：media/随机ID.mp4
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn)) #序列号并打包

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)  # 并行数量
        n_inits = len(self.env_init_fn_dills)  # 总数量
        n_chunks = math.ceil(n_inits / n_envs) # 需要轮次数

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs) #防止超出
            this_global_slice = slice(start, end) #获取全局切片
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs) #获取本次切片
            
            this_init_fns = self.env_init_fn_dills[this_global_slice] #根据切片 获取 对应任务
            #如果不够任务并行数，则进行补齐
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns]) #为每个子进程分配对应参数  

            # start rollout
            obs = env.reset()  #环境复位，并返回观测值obs = {'image': ...,   # 图像数据  'agent_pos': ...  # 机器人位置}
            past_action = None  #动作历史清空
            policy.reset() # 清理策略缓存  避免执行上一次任务结束时没跑完的残余动作

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}",  #进度条
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[ 
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict,  #CPU (Numpy) -> GPU (Tensor)
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict) #返回预测动作  8步

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy()) #GPU (Tensor) -> CPU (Numpy)

                action = np_action_dict['action']

                # step env
                #执行动作并返回最后n_obs_steps帧观测值
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice] #获取reward
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i] ##prefix为train 或 test
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():  #prefix为train 或 test
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data
'''
log_data = {
    # --- 训练集 (Train) 的详细成绩 ---
    'train/sim_max_reward_0': 1.0,      # 种子0：完美完成 (覆盖率 100%)
    'train/sim_max_reward_1': 0.95,     # 种子1：完成得很好 (覆盖率 95%)

    # --- 测试集 (Test) 的详细成绩 ---
    'test/sim_max_reward_100': 0.88,    # 种子100：勉强算过关
    'test/sim_max_reward_101': 0.12,    # 种子101：失败 (可能是推飞了，或者没推到)
    'test/sim_max_reward_102': 0.92,    # 种子102：成功

    # ★ 训练集平均分：(1.0 + 0.95) / 2
    # 代表模型对自己见过的情况掌握得如何
    'train/mean_score': 0.975,

    # ★ 测试集平均分：(0.88 + 0.12 + 0.92) / 3
    # 代表模型的泛化能力 (这是论文里最重要的那个数字)
    'test/mean_score': 0.64
}
'''