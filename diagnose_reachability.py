#!/usr/bin/env python3
"""
诊断POLAR可达集计算
检查为什么可达集宽度异常小
"""

import sys
sys.path.append('.')

from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
import numpy as np
import torch

print("="*70)
print("POLAR 可达集诊断")
print("="*70)

# 初始化
config = TD3Config()
env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
agent = TD3Agent(12, 2, 0.5, config)
agent.load('./models/final_20251009_105845')

# 获取一个测试状态
obs, _ = env.reset()
print(f"\n测试状态:")
print(f"  观测: {obs}")
print(f"  距离: {obs[0]*5:.2f}m, 方位: {obs[1]*np.pi:.2f}rad")
print(f"  激光: min={np.min(obs[2:10])*10:.2f}m, max={np.max(obs[2:10])*10:.2f}m")

# 1. 测试网络直接输出
print(f"\n【测试1】网络直接输出（无扰动）")
with torch.no_grad():
    state_tensor = torch.FloatTensor(obs.reshape(1, -1)).to(agent.device)
    action = agent.actor(state_tensor).cpu().numpy().flatten()
    print(f"  动作输出: 线速度={action[0]:.6f}, 角速度={action[1]:.6f}")

# 2. 测试输入扰动的影响
print(f"\n【测试2】输入扰动对输出的影响")
perturbations = [0.001, 0.005, 0.01, 0.02, 0.05]
for perturb in perturbations:
    actions = []
    for _ in range(20):
        # 添加随机扰动
        noise = np.random.uniform(-perturb, perturb, size=obs.shape)
        perturbed_obs = obs + noise
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(perturbed_obs.reshape(1, -1)).to(agent.device)
            action = agent.actor(state_tensor).cpu().numpy().flatten()
            actions.append(action)
    
    actions = np.array(actions)
    linear_range = actions[:, 0].max() - actions[:, 0].min()
    angular_range = actions[:, 1].max() - actions[:, 1].min()
    
    print(f"  扰动±{perturb:.3f}: 线速度范围={linear_range:.6f}, 角速度范围={angular_range:.6f}")

# 3. 检查网络权重
print(f"\n【测试3】网络权重统计")
with torch.no_grad():
    for name, param in agent.actor.named_parameters():
        weights = param.cpu().numpy()
        print(f"  {name:20s}: shape={str(weights.shape):15s} "
              f"mean={weights.mean():8.4f} std={weights.std():8.4f} "
              f"min={weights.min():8.4f} max={weights.max():8.4f}")

# 4. 测试POLAR计算的详细过程
print(f"\n【测试4】POLAR可达集计算（详细）")

# 使用不同的observation_error测试
test_errors = [0.001, 0.005, 0.01, 0.02, 0.05]
print(f"\n观测误差对可达集宽度的影响:")
print(f"{'误差':<8s} {'线速度宽度':<15s} {'角速度宽度':<15s} {'计算时间':<10s}")
print("-" * 60)

import time
for error in test_errors:
    start_time = time.time()
    is_safe, ranges = agent.verify_safety(
        obs,
        observation_error=error,
        bern_order=1,
        error_steps=4000
    )
    elapsed = time.time() - start_time
    
    linear_width = ranges[0][1] - ranges[0][0]
    angular_width = ranges[1][1] - ranges[1][0]
    
    print(f"±{error:.3f}  {linear_width:>14.6f}  {angular_width:>14.6f}  {elapsed:>9.2f}s")
    print(f"        线速度: [{ranges[0][0]:.6f}, {ranges[0][1]:.6f}]")
    print(f"        角速度: [{ranges[1][0]:.6f}, {ranges[1][1]:.6f}]")

# 5. 测试Bernstein多项式阶数的影响
print(f"\n【测试5】Bernstein多项式阶数的影响")
test_orders = [1, 2, 3, 4, 5]
print(f"{'阶数':<6s} {'线速度宽度':<15s} {'角速度宽度':<15s} {'计算时间':<10s}")
print("-" * 60)

for order in test_orders:
    start_time = time.time()
    try:
        is_safe, ranges = agent.verify_safety(
            obs,
            observation_error=0.01,
            bern_order=order,
            error_steps=4000
        )
        elapsed = time.time() - start_time
        
        linear_width = ranges[0][1] - ranges[0][0]
        angular_width = ranges[1][1] - ranges[1][0]
        
        print(f"{order:<6d} {linear_width:>14.6f}  {angular_width:>14.6f}  {elapsed:>9.2f}s")
    except Exception as e:
        print(f"{order:<6d} 计算失败: {e}")

# 6. 比较：直接扰动 vs POLAR计算
print(f"\n【测试6】方法对比")
print(f"\n方法1: 蒙特卡洛采样（10000次随机扰动）")
actions = []
for _ in range(10000):
    noise = np.random.uniform(-0.01, 0.01, size=obs.shape)
    perturbed_obs = obs + noise
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(perturbed_obs.reshape(1, -1)).to(agent.device)
        action = agent.actor(state_tensor).cpu().numpy().flatten()
        actions.append(action)

actions = np.array(actions)
mc_linear_range = [actions[:, 0].min(), actions[:, 0].max()]
mc_angular_range = [actions[:, 1].min(), actions[:, 1].max()]

print(f"  线速度范围: [{mc_linear_range[0]:.6f}, {mc_linear_range[1]:.6f}] "
      f"宽度={mc_linear_range[1]-mc_linear_range[0]:.6f}")
print(f"  角速度范围: [{mc_angular_range[0]:.6f}, {mc_angular_range[1]:.6f}] "
      f"宽度={mc_angular_range[1]-mc_angular_range[0]:.6f}")

print(f"\n方法2: POLAR形式化验证")
is_safe, polar_ranges = agent.verify_safety(obs, observation_error=0.01, bern_order=1, error_steps=4000)
polar_linear_width = polar_ranges[0][1] - polar_ranges[0][0]
polar_angular_width = polar_ranges[1][1] - polar_ranges[1][0]

print(f"  线速度范围: [{polar_ranges[0][0]:.6f}, {polar_ranges[0][1]:.6f}] "
      f"宽度={polar_linear_width:.6f}")
print(f"  角速度范围: [{polar_ranges[1][0]:.6f}, {polar_ranges[1][1]:.6f}] "
      f"宽度={polar_angular_width:.6f}")

print(f"\n对比:")
print(f"  线速度: MC采样={mc_linear_range[1]-mc_linear_range[0]:.6f} vs "
      f"POLAR={polar_linear_width:.6f} (比值={polar_linear_width/(mc_linear_range[1]-mc_linear_range[0]):.2f})")
print(f"  角速度: MC采样={mc_angular_range[1]-mc_angular_range[0]:.6f} vs "
      f"POLAR={polar_angular_width:.6f} (比值={polar_angular_width/(mc_angular_range[1]-mc_angular_range[0]):.2f})")

print(f"\n{'='*70}")
print(f"诊断结论:")
print(f"{'='*70}")
print(f"\n如果POLAR计算的可达集明显小于蒙特卡洛采样，可能的原因:")
print(f"  1. Taylor模型的误差区间估计过于保守（紧）")
print(f"  2. Bernstein多项式近似阶数太低")
print(f"  3. 网络本身对输入扰动不敏感（权重小或已收敛到稳定策略）")
print(f"\n如果蒙特卡洛采样的范围也很小，说明:")
print(f"  - 网络训练得很稳定，对输入噪声鲁棒")
print(f"  - 这是正常现象，模型质量很好")

env.close()