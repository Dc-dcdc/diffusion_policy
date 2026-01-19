import sys
import torch
import numpy as np

def check_r3m_installation():
    print("Step 1: 尝试导入 r3m 包...")
    try:
        from r3m import load_r3m
        print("✅ [成功] r3m 包已导入。")
    except ImportError as e:
        print(f"❌ [失败] 无法导入 r3m。错误信息: {e}")
        print("💡 建议: 请尝试运行 `pip install r3m` 或检查 PYTHONPATH。")
        return

    print("\nStep 2: 尝试加载 r3m 模型 (resnet18)...")
    try:
        # 检测是否有 GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   -> 使用设备: {device}")
        
        # 加载模型 (首次运行会自动下载预训练权重)
        model = load_r3m("resnet18") 
        model.eval()
        model.to(device)
        print("✅ [成功] r3m 模型加载成功。")
    except Exception as e:
        print(f"❌ [失败] 模型加载出错。可能原因：网络问题(无法下载权重)或依赖冲突。")
        print(f"错误信息: {e}")
        return

    print("\nStep 3: 进行一次前向推理测试...")
    try:
        # 创建一个假的随机图片输入 (Batch=1, Channels=3, H=224, W=224)
        # R3M 默认接受 0-255 的输入 (如果是通过其特定的 transforms 处理)
        # 这里我们直接模拟经过预处理后的 tensor
        dummy_input = torch.rand(1, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ [成功] 推理完成！")
        print(f"   -> 输出特征维度: {output.shape}") 
        # resnet18通常输出 512 维特征
        if output.shape[1] == 512:
            print("   -> 维度验证正确 (ResNet18 -> 512)")
            
    except Exception as e:
        print(f"❌ [失败] 推理过程出错。错误信息: {e}")
        return

    print("\n🎉 恭喜！r3m 安装及运行环境完全正常。")

if __name__ == "__main__":
    check_r3m_installation()