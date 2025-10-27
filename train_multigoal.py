#!/usr/bin/env python3
"""
Clearpath TD3+POLAR 训练脚本
"""

import sys
import os
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# 添加路径
sys.path.append(os.path.dirname(__file__))

# 导入自定义模块
from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config


def train_td3_polar(config=None):
    """
    主训练函数
    
    Args:
        config: TD3Config对象
    """
    if config is None:
        config = TD3Config()
    
    print("\n" + "="*80)
    print("Clearpath TD3+POLAR 训练")
    print("="*80)
    
    # 1. 创建环境
    print("\n[1/4] 创建环境...")
    env = ClearpathNavEnv(
        robot_name=config.robot_name,
        goal_pos=config.goal_pos,
        max_steps=config.max_steps,
        collision_threshold=config.collision_threshold
    )
    print(f"✓ 环境OK: 观测={env.observation_space.shape}, 动作={env.action_space.shape}")
    
    # 2. 创建智能体
    print("\n[2/4] 创建TD3智能体...")
    agent = TD3Agent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        max_action=config.max_action,
        config=config
    )
    
    # 3. 创建日志目录
    print("\n[3/4] 准备日志...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.log_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"✓ 日志目录: {log_dir}")
    
    # 4. 训练循环
    print("\n[4/4] 开始训练...")
    print(f"  总步数: {config.total_timesteps}")
    print(f"  随机探索: {config.start_timesteps}步")
    print(f"  POLAR验证间隔: {config.verify_interval}步")
    
    # ===== 新增：多目标池 =====
    goal_pool = [
        (3.0, -2.0),   # 原目标（右上）
        (2.0, 2.0),    # 左上
        (-2.0, 2.0),   # 左下
        (-2.0, -2.0),  # 右下
        (3.0, 0.0),    # 上中
        (0.0, 3.0),    # 左中
    ]
    print(f"\n  🎯 多目标训练模式：{len(goal_pool)}个目标点")
    for i, goal in enumerate(goal_pool, 1):
        print(f"    目标{i}: {goal}")
    
    # 目标统计
    goal_stats = {goal: {'success': 0, 'collision': 0, 'timeout': 0, 'total': 0} 
                  for goal in goal_pool}
    current_goal = goal_pool[0]
    env.world_goal = np.array(current_goal, dtype=np.float32)
    
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    
    # 统计变量
    rewards_history = []
    safe_count = 0
    total_verify = 0
    
    pbar = tqdm(total=config.total_timesteps, desc="训练进度")
    
    for t in range(config.total_timesteps):
        # 选择动作
        if t < config.start_timesteps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)
            noise = np.random.normal(0, config.max_action * config.expl_noise, 
                                    size=config.action_dim)
            action = (action + noise).clip(-config.max_action, config.max_action)
        
        # 执行动作
        next_state, reward, done, truncated, info = env.step(action)
        done_bool = float(done) if episode_steps < config.max_steps else 0
        
        # 存储经验
        agent.store_transition(state, action, next_state, reward, done_bool)
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        # 更新网络
        if t >= config.start_timesteps:
            agent.train()
        
        # POLAR验证
        if t > 0 and t % config.verify_interval == 0:
            try:
                is_safe, action_ranges = agent.verify_safety(
                    state,
                    observation_error=config.observation_error,
                    bern_order=config.bern_order,
                    error_steps=config.error_steps
                )
                
                total_verify += 1
                if is_safe:
                    safe_count += 1
                
                pbar.write(f"[验证] 步数{t}: 安全率={safe_count}/{total_verify} ({safe_count/total_verify:.1%})")
            except Exception as e:
                pbar.write(f"[警告] POLAR验证失败: {e}")
        
        # Episode结束
        if done or truncated:
            rewards_history.append(episode_reward)
            
            # ===== 新增：记录当前目标的统计 =====
            goal_stats[current_goal]['total'] += 1
            if info.get('goal_reached'):
                goal_stats[current_goal]['success'] += 1
            elif info.get('collision'):
                goal_stats[current_goal]['collision'] += 1
            elif info.get('timeout'):
                goal_stats[current_goal]['timeout'] += 1
            
            # 构建消息
            msg = f"Episode {episode_num}: R={episode_reward:.1f}, Steps={episode_steps}, 目标{current_goal}"
            if info.get('goal_reached'):
                msg += " 🎯成功"
            elif info.get('collision'):
                msg += " ⚠️碰撞"
            elif info.get('timeout'):
                msg += " ⏱️超时"
            pbar.write(msg)
            
            # ===== 新增：随机选择新目标 =====
            current_goal = goal_pool[np.random.randint(len(goal_pool))]
            env.world_goal = np.array(current_goal, dtype=np.float32)
            
            # 重置
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
            
            # 定期保存
            if episode_num % 50 == 0:
                model_path = os.path.join(config.model_path, timestamp)
                os.makedirs(model_path, exist_ok=True)
                agent.save(f"{model_path}/episode_{episode_num}")
                pbar.write(f"✓ 模型已保存: episode_{episode_num}")
                
                # ===== 新增：打印目标统计 =====
                pbar.write("\n" + "="*70)
                pbar.write(f"Episode {episode_num} - 各目标表现统计:")
                pbar.write("="*70)
                for goal, stats in goal_stats.items():
                    total = stats['total']
                    if total > 0:
                        success_rate = stats['success'] / total * 100
                        collision_rate = stats['collision'] / total * 100
                        pbar.write(f"  {str(goal):<15s}: {total:3d}次 | "
                                  f"成功 {success_rate:5.1f}% ({stats['success']:2d}) | "
                                  f"碰撞 {collision_rate:5.1f}% ({stats['collision']:2d})")
                pbar.write("="*70 + "\n")
        
        pbar.update(1)
        pbar.set_postfix({
            'Ep': episode_num,
            'R': f'{episode_reward:.0f}',
            'Safe': f'{safe_count}/{total_verify}'
        })
    
    pbar.close()
    
    # 训练完成
    print("\n" + "="*80)
    print("训练完成！")
    print("="*80)
    print(f"总Episode: {episode_num}")
    print(f"平均奖励: {np.mean(rewards_history):.2f}")
    print(f"最高奖励: {np.max(rewards_history):.2f}")
    if total_verify > 0:
        print(f"POLAR安全率: {safe_count/total_verify:.2%}")
    
    # ===== 新增：各目标最终表现统计 =====
    print("\n" + "="*80)
    print("各目标最终表现:")
    print("="*80)
    print(f"{'目标位置':<15s} {'训练次数':<10s} {'成功率':<12s} {'碰撞率':<12s}")
    print("-" * 80)
    
    total_success = 0
    total_collision = 0
    total_episodes = 0
    
    for goal, stats in goal_stats.items():
        total = stats['total']
        if total > 0:
            success_rate = stats['success'] / total * 100
            collision_rate = stats['collision'] / total * 100
            print(f"{str(goal):<15s} {total:<10d} "
                  f"{success_rate:>5.1f}% ({stats['success']:3d}) "
                  f"{collision_rate:>5.1f}% ({stats['collision']:3d})")
            
            total_success += stats['success']
            total_collision += stats['collision']
            total_episodes += total
    
    # 总体统计
    print("-" * 80)
    if total_episodes > 0:
        overall_success = total_success / total_episodes * 100
        overall_collision = total_collision / total_episodes * 100
        print(f"{'总体':<15s} {total_episodes:<10d} "
              f"{overall_success:>5.1f}% ({total_success:3d}) "
              f"{overall_collision:>5.1f}% ({total_collision:3d})")
    print("="*80)
    
    # 保存最终模型
    final_path = os.path.join(config.model_path, f"final_{timestamp}")
    os.makedirs(final_path, exist_ok=True)
    agent.save(final_path)
    agent.save_weights(f"{final_path}/weights_for_polar.npz")
    print(f"\n✓ 最终模型: {final_path}")
    print(f"✓ POLAR权重: {final_path}/weights_for_polar.npz")
    
    env.close()
    
    return agent


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--start_timesteps', type=int, default=25000)
    parser.add_argument('--verify_interval', type=int, default=1000)
    
    args = parser.parse_args()
    
    config = TD3Config()
    config.total_timesteps = args.timesteps
    config.start_timesteps = args.start_timesteps
    config.verify_interval = args.verify_interval
    
    agent = train_td3_polar(config)
    
    print("\n✓ 全部完成！")