#!/usr/bin/env python3
import sys
sys.path.append('.')

from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
import numpy as np

print("="*60)
print("POLAR安全性验证 - 调试版")
print("="*60)

config = TD3Config()

print("\n[1/4] 初始化环境...")
try:
    env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
    print("✓ 环境初始化成功")
except Exception as e:
    print(f"✗ 环境初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[2/4] 初始化智能体...")
try:
    agent = TD3Agent(12, 2, 0.5, config)
    print("✓ 智能体初始化成功")
except Exception as e:
    print(f"✗ 智能体初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[3/4] 加载模型...")
model_path = './models/final_20251009_105845'
try:
    agent.load(model_path)
    print(f"✓ 模型加载成功: {model_path}")
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[4/4] 测试环境重置...")
try:
    obs, info = env.reset()
    print(f"✓ 环境重置成功")
    print(f"  观测形状: {obs.shape}")
    print(f"  观测类型: {type(obs)}, {obs.dtype}")
    print(f"  观测范围: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  info内容: {info}")
except Exception as e:
    print(f"✗ 环境重置失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("开始验证（测试3个状态）...")
print("="*60)

safe_count = 0
unsafe_count = 0
error_count = 0

for i in range(3):
    print(f"\n--- 测试 {i+1}/3 ---")
    try:
        obs, info = env.reset()
        print(f"重置成功, obs shape: {obs.shape}")
        
        is_safe, ranges = agent.verify_safety(
            obs,
            observation_error=0.01,
            bern_order=1,
            error_steps=4000
        )
        
        if is_safe:
            safe_count += 1
            print(f"✅ 安全 - 动作范围: {ranges}")
        else:
            unsafe_count += 1
            print(f"⚠️  不安全 - 动作范围: {ranges}")
            
    except Exception as e:
        error_count += 1
        print(f"✗ 验证错误: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("测试结果")
print("="*60)
print(f"安全:   {safe_count}/3")
print(f"不安全: {unsafe_count}/3")
print(f"错误:   {error_count}/3")
print("="*60)

if error_count == 0:
    print("\n✅ 基本测试通过！可以运行完整的100次验证了")
else:
    print("\n⚠️  仍有错误，请检查上面的错误信息")

env.close()
