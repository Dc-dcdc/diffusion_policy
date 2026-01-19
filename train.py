"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf  #Python 配置管理库
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# 向 OmegaConf 配置系统注册一个名为 "eval" 的自定义解析器，允许yaml文件中执行python代码
OmegaConf.register_new_resolver("eval", eval, replace=True)   #绑定的函数Python内置的eval()函数

# Hydra 会自动创建一个基于时间戳的文件夹（例如 outputs/2023-10-27/10-00-00/），把当前的配置 (.hydra/config.yaml)、日志和代码快照都进行保存
@hydra.main(
    version_base=None,  #不需要向后兼容旧版本
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))   #去 diffusion_policy/config 目录下寻找配置文件  可以在命令行微调实验参数
) #读取完配置文件后，将其解析为一个字典对象 cfg 传给 main 函数

def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)  #计算所有动态变量并固定（如时间戳 ${now:}）

    cls = hydra.utils.get_class(cfg._target_) #Hydra 根据配置中的 _target_ 字段，动态导入该类。
    workspace: BaseWorkspace = cls(cfg) #实例化
    workspace.run()

if __name__ == "__main__":
     main()
