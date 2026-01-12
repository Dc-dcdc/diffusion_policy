import torch
import r3m

print("👁️ 正在检查 R3M 库...")

try:
    # 1. 尝试加载最小的模型 (ResNet18)
    # 注意：第一次运行这行代码时，会自动下载约 40MB-100MB 的权重文件
    print("⏳ 正在加载 R3M 模型 (首次运行会下载权重，请耐心等待)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    r3m_model = r3m.load_r3m("resnet18") # 也可以换成 "resnet50"
    r3m_model.to(device)
    r3m_model.eval()

    # 2. 创建一个伪造的图像输入 (Batch=1, Channel=3, Height=224, Width=224)
    # R3M 期望的输入是标准的 ImageNet 尺寸
    dummy_img = torch.rand(1, 3, 224, 224).to(device)

    # 3. 进行一次前向传播
    with torch.no_grad():
        embedding = r3m_model(dummy_img)

    print(f"✅ R3M 安装成功！")
    print(f"   运行设备: {device}")
    print(f"   输出向量维度: {embedding.shape} (应为 [1, 512])")

except ImportError:
    print("❌ 错误: 找不到 r3m 包。请确认是否安装。")
except Exception as e:
    print(f"❌ 运行出错: {e}")
    print("👉 提示: 如果是网络连接错误，可能是因为无法下载预训练权重。")