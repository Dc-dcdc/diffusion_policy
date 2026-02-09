import time
import numpy as np

import cv2
import scipy.spatial.transform as st
import mujoco
import mujoco.viewer

xml_path = "/home/dc/diffusion_policy/diffusion_policy/env/pusht/mujoco_ur5e/pusht.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 初始化渲染器
renderer = mujoco.Renderer(model, height=240, width=320)

# def reset_and_sync(model, data):
"""
核心修复函数：重置仿真，并强行同步 Mocap 位置，防止物理爆炸
"""
# mujoco.mj_resetData(model, data)
# 1. 重置所有物体到 XML/Keyframe 定义的初始位置 (T块回桌上，机械臂回Home)
mujoco.mj_resetDataKeyframe(model, data, 0)

# print(data)
# 2. 必须刷新一次正向运动学 (计算机械臂手腕此刻的世界坐标)
mujoco.mj_forward(model, data) #
# -------------------打印mujoco所有body的id和name-----------------------------------------------
print("--- Body ID List ---")
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    print(f"ID: {i} -> Name: {name}")
# -----------------------------end------------------------------------------------------------
# 3. 找到 ID
wrist_body_name = 'tool_pusher' # 确保和 XML 里 weld 的 body2 一致
mocap_body_name = 'mocap_target'

wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body_name)
mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
print(data.xpos[wrist_id])
print(data.xpos[mocap_body_id])
if wrist_id != -1 and mocap_body_id != -1:
    mocap_id = model.body_mocapid[mocap_body_id]
    
#     # 4. 【关键瞬移】把 Mocap 搬运到机械臂手腕当前的位置
#     # 这样 weld 约束的误差瞬间变为 0，就不会产生爆炸力了
    target_pos = np.array([0.3, 0.4, 0.96])
    rot = st.Rotation.from_euler('xyz', [-180, 0, -90], degrees=True)
    quat_xyzw = rot.as_quat() # Scipy 输出是 [x, y, z, w]
    target_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    data.mocap_pos[mocap_id] = target_pos
    data.mocap_quat[mocap_id] = target_quat
    for _ in range(500):
        mujoco.mj_step(model, data)
    # data.mocap_pos[mocap_id] = data.xpos[wrist_id]
    # data.mocap_quat[mocap_id] = data.xquat[wrist_id]

#     # time.sleep(10)
#     # 5. 再次刷新，让瞬移生效
    mujoco.mj_forward(model, data)
    print(data.mocap_pos[mocap_id])
    print(data.xpos[wrist_id])
    #     print(">>> 重置成功：Mocap 已吸附到机械臂末端。")
    # else:
    #     print(f"❌ 错误：找不到 Body。Wrist ID: {wrist_id}, Mocap ID: {mocap_body_id}")

# === 程序启动时，先执行一次安全的重置 ===
# reset_and_sync(model, data)

# target_pos = np.array([0.3, 0.4, 0.96])
# data.mocap_pos[mocap_id] = target_pos
# for _ in range(500):
#     mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    
    # set viewing camera
    mujoco.mjv_defaultFreeCamera(model, viewer.cam) ## 重置相机参数
    viewer.cam.distance = 1.2 ## distance: 缩放距离 (1.2米)
    viewer.cam.elevation = -15 #仰角/俯角 (-15度，表示稍微向下看)
    viewer.cam.azimuth = 70 #方位角 (70度，表示水平旋转的角度)
    viewer.cam.lookat = (0.3, 0, .3) #镜头聚焦的中心点坐标 (x=0.3, y=0, z=0.3)
    
    while viewer.is_running():
        step_start = time.time() # 记录这一步开始的时间，用于后面控制帧率
        
        # data.ctrl[0:6] = data.qpos[0:6] #让电机保持该位姿
        # data.ctrl[0:6] =[3.1415, -1.5708, 2, -2, -1.5708, 0]
        
        mujoco.mj_step(model, data)


        viewer.sync() #把最新的 data 状态同步显示到交互窗口里
        
        # 2. 更新场景
        # 告诉渲染器："现在的物理状态(data)变了，请同步一下"
        renderer.update_scene(data, 'Wrist_Camera')
        # 3. 渲染并获取像素
        # 这一步生成 RGB 图像矩阵，并把结果存成 numpy 数组，形状通常是 (480, 640, 3)
        Wrist_Camera = renderer.render()

        renderer.update_scene(data, 'Front_Camera')
        Front_Camera = renderer.render()

        renderer.update_scene(data, 'Top_down_Camera')
        Top_down_Camera = renderer.render()
        # 此时你可以把它喂给神经网络，或者用 cv、plt 显示出来
        Wrist_Camera = cv2.cvtColor(Wrist_Camera, cv2.COLOR_RGB2BGR)  # 颜色转换：MuJoCo 输出 RGB，OpenCV 默认使用 BGR
        Front_Camera = cv2.cvtColor(Front_Camera, cv2.COLOR_RGB2BGR)
        Top_down_Camera = cv2.cvtColor(Top_down_Camera, cv2.COLOR_RGB2BGR)
        # 弹出一个名为 'frame' 的 OpenCV 小窗口显示图像
        cv2.imshow('Wrist_Camera', Wrist_Camera)
        cv2.imshow('Front_Camera', Front_Camera)
        cv2.imshow('Top_down_Camera', Top_down_Camera)
        # 必不可少！等待 1ms 并刷新 OpenCV 窗口。
        cv2.waitKey(1) # 如果没有这行，OpenCV 窗口会假死/不更新。
        
        # 计算刚才那些计算（物理+渲染）花了多久
        # model.opt.timestep 是物理步长（比如 0.002秒）
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        
        # 如果计算太快（比如只要 0.0001秒），就睡一会
        # 强行把运行速度限制在“实时”速度，防止仿真跑得飞快看不清
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)