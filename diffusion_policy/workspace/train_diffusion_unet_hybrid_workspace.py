# 强行把当前的运行环境“搬”回项目根目录，以解决导包报错（ModuleNotFoundError）和文件路径找不到**的问题
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent) #回到最外层根目录
    sys.path.append(ROOT_DIR) #解决import找不到包的问题
    os.chdir(ROOT_DIR) #解决文件路径问题

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

# 向 OmegaConf 配置系统注册一个名为 "eval" 的自定义解析器，允许yaml文件中执行python代码
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetHybridWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch'] #额外保存全局步数和当前轮数

    def __init__(self, cfg: OmegaConf, output_dir=None): #接收一个 OmegaConf 对象，包含了所有的训练配置（来源于 yaml 文件）
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed #配置文件中读取随机数种子
        torch.manual_seed(seed) #设置 PyTorch 的种子（影响神经网络权重的初始化、Dropout 等）
        np.random.seed(seed)  #设置 NumPy 的种子（影响数据预处理、环境随机性）
        random.seed(seed) #设置 Python 标准库 random 的种子

        # configure model
        self.model: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy) #输出的是实例对象   获取对应的类并直接使用参数进行初始化

        self.ema_model: DiffusionUnetHybridImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model) #拷贝模型，对主模型进行平滑滤波   参数更新方式为： ema_model = 0.9999 * ema_model + 0.0001 * model

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters()) #优化器 adamw

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training 恢复训练
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)  #diffusion_policy/dataset/pusht_image_dataset.py
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)  #数据参数配置  batch_size，进程数量，是否打乱 等
        normalizer = dataset.get_normalizer() #归一化

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer) #将数据集归一化好的参数 加载 到策略模型中
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        # WandB 的云端服务器
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        # 将所有的计算负载（模型参数、缓冲区、优化器状态）从CPU转到GPU
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True)) #CPU (Numpy) -> GPU (Tensor)
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every  #梯度累加次数，也就是模型更新频率，不除的话容易梯度爆炸
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            #主模型更新
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model) #参数更新方式为： ema_model = 0.9999 * ema_model + 0.0001 * model


                        # logging
                        # 1. 先构建基础日志字典
                        raw_loss_cpu = raw_loss.item() #将gpu计算的loss值导出到cpu，解放gpu内存
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False) #更新现实的进度条
                        train_losses.append(raw_loss_cpu) #记录损失
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        # 2. 【新增】如果开启了 EMA，计算并记录当前的衰减率
                        if cfg.training.use_ema:
                            # 注意：get_decay 需要传入当前的步数作为参数
                            step_log['ema_decay'] = ema.get_decay(self.global_step)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step) #更新云端服务器
                            json_logger.log(step_log) #更新日志
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss #最后批次的loss 换为 整个epoch的平均loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval() #切换评估模式，保证模型输出稳定   特别是关闭dropout，保证模型完整性

                # run rollout  推演测试，评估 实际部署性能
                if (self.epoch % cfg.training.rollout_every) == 0:  #每50轮推演一次，并录制视频
                    runner_log = env_runner.run(policy) 
                    # log all
                    step_log.update(runner_log)

                # run validation
                # 预测噪声 和 真实噪声 之间的差异
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch) #计算 预测噪声 和 真实噪声 的loss
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                # 去噪生成动作过程 和 真实动作 的差异
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad(): #每一步计算完，中间结果立刻丢弃，只保留最终输出传给下一步    不会积累历史参数   因为扩散模型生成数据需要迭代上百次，不丢弃会导致显存爆炸
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action) #计算 生成动作 和 真实动作 的loss
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint 记录最新的以及最好的K个
                if (self.epoch % cfg.training.checkpoint_every) == 0: #每50个epoch
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint() #训练存档 保存模型权重、优化器状态、调度器状态、训练进度等所有信息
                    if cfg.checkpoint.save_last_snapshot: 
                        self.save_snapshot() #模型快照 只保存模型权重

                    # sanitize metric names 指标键名清洗
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_') #将/换为_  例： val/loss -> val_loss
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict) #记录表现最好的K个模型

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train() #恢复训练模型，如打开dropout 

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step) #更新云端
                json_logger.log(step_log)  #更新日志
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
