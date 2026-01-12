import mujoco
import numpy as np


print("--------------------------------------------------")
print(f"🕵️‍♂️ 真相调查:")
print(f"你导入的 mujoco 来自哪里: {mujoco.__file__}")
print("--------------------------------------------------")
# 1. 定义一个简单的 XML 模型
xml = """
<mujoco>
  <worldbody>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" type="sphere" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

try:
    # 2. 加载模型
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # 3. 运行 100 步模拟
    print("🚀 开始模拟...")
    for i in range(100):
        mujoco.mj_step(model, data)

    print(f"✅ 新版 Mujoco 测试成功！最终位置数据: {data.qpos}")

except Exception as e:
    print(f"❌ 新版 Mujoco 出错: {e}")