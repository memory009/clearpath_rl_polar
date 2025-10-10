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
            
            msg = f"Episode {episode_num}: R={episode_reward:.1f}, Steps={episode_steps}"
            if info.get('goal_reached'):
                msg += " 🎯成功"
            elif info.get('collision'):
                msg += " ⚠️碰撞"
            pbar.write(msg)
            
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
        print(f"安全率: {safe_count/total_verify:.2%}")
    
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
