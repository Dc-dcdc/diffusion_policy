import robosuite as suite
import numpy as np

print("🤖 正在初始化 Robosuite 环境...")

try:
    # 1. 创建环境
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=False,          # 不弹窗
        has_offscreen_renderer=False, # 不渲染后台图像
        use_camera_obs=False,         # 不使用相机观察
    )

    # 2. 重置环境
    env.reset()
    
    # 获取动作维度
    dim = env.action_dim
    print(f"✅ 环境创建成功！动作维度: {dim}")

    # 3. 随机跑 10 步
    for i in range(10):
        action = np.random.randn(dim) # 随机生成动作
        obs, reward, done, info = env.step(action)
        print(f"第 {i+1} 步: 机械臂移动正常 (Reward: {reward:.4f})")

    print("✅✅✅ Robosuite 测试完美通过！你的 Diffusion Policy 环境已就绪！")

except Exception as e:
    print(f"❌ 运行出错: {e}")