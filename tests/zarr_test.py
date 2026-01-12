import zarr
import numpy as np

# 打开刚才生成的文件
# mode='r' 表示只读模式
root = zarr.open('/home/dc/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr', mode='r')

print("📂 文件结构树:")
print(root.tree())

print("\n📊 数据详情:")
# 假设里面有一个 data 组
if 'data' in root:
    for key in root['data'].keys():
        arr = root['data'][key]
        print(f"  - {key}: 形状 {arr.shape}, 类型 {arr.dtype}")

print("\n📍 索引信息:")
if 'meta' in root and 'episode_ends' in root['meta']:
    ends = root['meta']['episode_ends'][:]
    print(f"  - 共录制了 {len(ends)} 个回合 (Episodes)")
    print(f"  - 结束帧索引: {ends}")