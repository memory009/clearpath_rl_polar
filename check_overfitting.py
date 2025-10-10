#!/usr/bin/env python3
"""
检查模型是否过拟合到简单任务
"""

import sys
sys.path.append('.')

from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("过拟合诊断")
print("="*70)

config = TD3Config()
agent = TD3Agent(12, 2, 0.5, config)
agent.load('./models/final_20251009_105845')

# 测试1: 在原始目标点上测试一致性
print("\n【测试1】在训练目标(2,2)上的一致性")
print("运行10次episode，记录步数和轨迹...")

original_env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
original_steps = []
original_actions = []

for ep in range(10):
    obs, _ = original_env.reset()
    done = False
    step = 0
    actions = []
    
    while not done and step < 256:
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = original_env.step(action)
        actions.append(action.copy())
        step += 1
        
        if done or truncated:
            break
    
    original_steps.append(step)
    original_actions.append(np.array(actions))
    print(f"  Episode {ep+1}: {step}步 {'✓到达' if info.get('goal_reached') else '✗超时'}")

print(f"\n步数统计:")
print(f"  平均: {np.mean(original_steps):.1f}步")
print(f"  标准差: {np.std(original_steps):.2f}步")
print(f"  范围: [{min(original_steps)}, {max(original_steps)}]")

if np.std(original_steps) < 3:
    print(f"  ⚠️  标准差很小 - 网络输出高度一致（可能过拟合）")

# 计算动作的一致性
print(f"\n动作一致性分析:")
min_len = min(len(a) for a in original_actions)
trimmed_actions = [a[:min_len] for a in original_actions]
actions_array = np.array(trimmed_actions)  # shape: (10, min_len, 2)

linear_std = np.mean(np.std(actions_array[:, :, 0], axis=0))
angular_std = np.mean(np.std(actions_array[:, :, 1], axis=0))

print(f"  线速度标准差: {linear_std:.6f}")
print(f"  角速度标准差: {angular_std:.6f}")

if linear_std < 0.01 and angular_std < 0.05:
    print(f"  ⚠️  动作标准差极小 - 几乎完全相同的轨迹")

original_env.close()

# 测试2: 在不同目标点上测试泛化能力
print(f"\n{'='*70}")
print("【测试2】泛化能力测试 - 不同目标点")
print(f"{'='*70}")

test_goals = [
    (1.0, 1.0, "近距离"),
    (2.0, 2.0, "训练目标"),
    (3.0, 3.0, "中距离"),
    (4.0, 4.0, "远距离"),
    (5.0, 5.0, "很远"),
    (2.0, 3.0, "不同方向1"),
    (3.0, 2.0, "不同方向2"),
    (1.5, 2.5, "不同方向3"),
]

results = []

for goal_x, goal_y, desc in test_goals:
    env = ClearpathNavEnv(goal_pos=(goal_x, goal_y))
    
    success_count = 0
    steps_list = []
    
    for _ in range(5):  # 每个目标测试5次
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 256:
            action = agent.select_action(obs)
            obs, reward, done, truncated, info = env.step(action)
            step += 1
            
            if done or truncated:
                break
        
        if info.get('goal_reached'):
            success_count += 1
        steps_list.append(step)
    
    distance = np.sqrt(goal_x**2 + goal_y**2)
    results.append({
        'goal': (goal_x, goal_y),
        'desc': desc,
        'distance': distance,
        'success_rate': success_count / 5,
        'avg_steps': np.mean(steps_list),
        'std_steps': np.std(steps_list)
    })
    
    env.close()
    
    print(f"\n目标 ({goal_x:.1f}, {goal_y:.1f}) - {desc} (距离{distance:.2f}m)")
    print(f"  成功率: {success_count}/5 ({success_count/5*100:.0f}%)")
    print(f"  平均步数: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")

# 分析结果
print(f"\n{'='*70}")
print("诊断结论")
print(f"{'='*70}")

training_goal_result = [r for r in results if r['goal'] == (2.0, 2.0)][0]
other_results = [r for r in results if r['goal'] != (2.0, 2.0)]

print(f"\n1. 训练目标(2,2)性能:")
print(f"   成功率: {training_goal_result['success_rate']*100:.0f}%")
print(f"   步数标准差: {training_goal_result['std_steps']:.2f}")

avg_other_success = np.mean([r['success_rate'] for r in other_results])
print(f"\n2. 其他目标平均性能:")
print(f"   成功率: {avg_other_success*100:.0f}%")

print(f"\n3. 过拟合指标:")

# 指标1: 训练目标vs其他目标的成功率差异
success_gap = training_goal_result['success_rate'] - avg_other_success
print(f"   成功率差距: {success_gap*100:.1f}%")

if success_gap > 0.3:
    print(f"   ⚠️  严重过拟合 - 在训练目标上表现明显更好")
elif success_gap > 0.1:
    print(f"   ⚠️  轻度过拟合")
else:
    print(f"   ✓ 泛化能力良好")

# 指标2: 步数一致性
if training_goal_result['std_steps'] < 3:
    print(f"   ⚠️  步数标准差<3 - 网络记住了固定策略")
else:
    print(f"   ✓ 步数有合理变化")

# 指标3: 距离比例分析
print(f"\n4. 距离-步数关系:")
for r in sorted(results, key=lambda x: x['distance']):
    steps_per_meter = r['avg_steps'] / r['distance'] if r['distance'] > 0 else 0
    print(f"   {r['distance']:.2f}m: {r['avg_steps']:.1f}步 "
          f"({steps_per_meter:.1f}步/米) 成功率{r['success_rate']*100:.0f}%")

# 建议
print(f"\n{'='*70}")
print("改进建议")
print(f"{'='*70}")

print(f"\n如果发现过拟合，建议:")
print(f"  1. 增加任务难度:")
print(f"     - 目标点设置为随机: goal_pos=(random.uniform(2,5), random.uniform(2,5))")
print(f"     - 增加障碍物")
print(f"     - 更复杂的环境")
print(f"")
print(f"  2. 减少训练时间:")
print(f"     - 100k步对于2.83m的简单任务来说太多了")
print(f"     - 建议: 30k-50k步")
print(f"")
print(f"  3. 增加探索:")
print(f"     - 增加动作噪声")
print(f"     - 使用curriculum learning（由易到难）")
print(f"")
print(f"  4. 数据增强:")
print(f"     - 在训练中添加观测噪声")
print(f"     - 随机初始位置和朝向")

# 可视化（如果matplotlib可用）
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图: 距离 vs 成功率
    distances = [r['distance'] for r in results]
    success_rates = [r['success_rate']*100 for r in results]
    colors = ['red' if r['goal'] == (2.0, 2.0) else 'blue' for r in results]
    
    axes[0].scatter(distances, success_rates, c=colors, s=100, alpha=0.6)
    axes[0].axhline(y=100, color='green', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('Distance to Goal (m)')
    axes[0].set_ylabel('Success Rate (%)')
    axes[0].set_title('Generalization Test')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Training goal (2,2)', 'Other goals'], loc='lower left')
    
    # 右图: 距离 vs 平均步数
    avg_steps = [r['avg_steps'] for r in results]
    axes[1].scatter(distances, avg_steps, c=colors, s=100, alpha=0.6)
    axes[1].set_xlabel('Distance to Goal (m)')
    axes[1].set_ylabel('Average Steps')
    axes[1].set_title('Steps vs Distance')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=150)
    print(f"\n📊 可视化已保存到: overfitting_analysis.png")
except Exception as e:
    print(f"\n(可视化跳过: {e})")

print(f"\n{'='*70}")