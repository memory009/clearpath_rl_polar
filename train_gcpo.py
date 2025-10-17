#!/usr/bin/env python3
"""
GCPO 完整训练流程
阶段1: BC预训练 → 阶段2: RL Fine-tuning (自适应课程)
"""

import sys
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from algorithms.gcpo_agent import GCPOAgent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config


class GCPOConfig(TD3Config):
    """GCPO训练配置"""
    
    # BC预训练参数
    bc_epochs = 100
    bc_batch_size = 256
    bc_learning_rate = 3e-4
    bc_val_split = 0.1
    
    # RL Fine-tuning参数
    rl_total_timesteps = 100000
    rl_start_timesteps = 5000  # 比原来短,因为有BC初始化
    
    # 课程学习参数
    goal_space_bounds = [(0.5, 5.5), (-3.0, 3.5)]
    capability_window = 100
    target_capability = 0.5  # 目标成功率
    
    # 能力评估
    eval_interval = 5000  # 每隔多少步评估一次
    eval_episodes = 10


def train_gcpo_rl_phase(
    agent, 
    config, 
    start_from_bc=True,
    save_dir='./models/gcpo'
):
    """
    GCPO RL Fine-tuning阶段
    
    Args:
        agent: 预训练的GCPO智能体
        config: 配置
        start_from_bc: 是否从BC预训练开始
        save_dir: 保存目录
    
    Returns:
        agent: 训练后的智能体
    """
    print("\n" + "="*80)
    print("🚀 GCPO RL Fine-tuning")
    print("="*80)
    
    if start_from_bc:
        print("  起点: BC预训练策略")
    else:
        print("  起点: 随机策略")
    
    print(f"  总步数: {config.rl_total_timesteps}")
    print(f"  初始探索: {config.rl_start_timesteps}步")
    print(f"  自适应课程: ✓ 启用")
    print("="*80 + "\n")
    
    # 切换到RL模式
    agent.switch_to_rl_mode()
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('./logs', f'gcpo_rl_{timestamp}')
    model_dir = os.path.join(save_dir, f'gcpo_rl_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练统计
    total_timesteps = 0
    episode_num = 0
    rewards_history = []
    success_history = []
    goal_history = []
    
    # POLAR验证统计
    safe_count = 0
    total_verify = 0
    
    # 创建进度条
    pbar = tqdm(total=config.rl_total_timesteps, desc="RL训练")
    
    # 主训练循环
    while total_timesteps < config.rl_total_timesteps:
        # 1. 从课程中采样目标
        curriculum_goal = agent.sample_curriculum_goal()
        
        # 2. 创建环境(使用课程目标)
        env = ClearpathNavEnv(goal_pos=tuple(curriculum_goal))
        
        # 3. 重置环境
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Episode循环
        for step in range(config.max_steps):
            # 选择动作
            if total_timesteps < config.rl_start_timesteps:
                # 初始探索阶段:随机动作
                action = env.action_space.sample()
            else:
                # 使用策略
                action = agent.select_action(state)
                # 添加探索噪声
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
            total_timesteps += 1
            
            # 训练网络
            if total_timesteps >= config.rl_start_timesteps:
                agent.train()
            
            # POLAR验证(可选)
            if total_timesteps > 0 and total_timesteps % config.verify_interval == 0:
                try:
                    is_safe, _ = agent.verify_safety(
                        state,
                        observation_error=config.observation_error,
                        bern_order=config.bern_order,
                        error_steps=config.error_steps
                    )
                    total_verify += 1
                    if is_safe:
                        safe_count += 1
                except:
                    pass
            
            pbar.update(1)
            
            if done or truncated:
                break
        
        # Episode结束处理
        episode_success = info.get('goal_reached', False)
        
        # 更新能力估计
        agent.update_capability(curriculum_goal, episode_success)
        
        # 记录统计
        rewards_history.append(episode_reward)
        success_history.append(episode_success)
        goal_history.append(curriculum_goal.copy())
        
        # 打印信息
        capability = agent.capability_estimator.estimate_capability(curriculum_goal)
        msg = (f"Ep {episode_num}: "
               f"Goal=({curriculum_goal[0]:.1f},{curriculum_goal[1]:.1f}), "
               f"R={episode_reward:.1f}, "
               f"Steps={episode_steps}, "
               f"Cap={capability:.1%}")
        
        if episode_success:
            msg += " 🎯"
        elif info.get('collision'):
            msg += " ⚠️"
        
        pbar.write(msg)
        
        episode_num += 1
        
        # 定期统计和保存
        if episode_num % 50 == 0:
            recent_success = np.mean(success_history[-50:])
            avg_reward = np.mean(rewards_history[-50:])
            
            pbar.write("\n" + "="*60)
            pbar.write(f"Episode {episode_num} 统计:")
            pbar.write(f"  最近50ep成功率: {recent_success:.1%}")
            pbar.write(f"  最近50ep平均奖励: {avg_reward:.1f}")
            pbar.write(f"  Memory大小: {agent.memory.size}")
            
            if total_verify > 0:
                pbar.write(f"  POLAR安全率: {safe_count/total_verify:.1%}")
            
            # 能力估计统计
            cap_stats = agent.get_capability_stats()
            pbar.write(f"  跟踪目标数: {cap_stats['num_goals_tracked']}")
            
            pbar.write("\n  距离区间成功率:")
            for dist, stats in sorted(cap_stats['distance_bin_stats'].items()):
                pbar.write(f"    <{dist:.1f}m: {stats['success_rate']:.1%} "
                          f"({stats['samples']}样本)")
            
            pbar.write("="*60 + "\n")
            
            # 保存模型
            agent.save_gcpo(f"{model_dir}/episode_{episode_num}")
            pbar.write(f"✓ 模型已保存: episode_{episode_num}\n")
        
        # 更新进度条显示
        pbar.set_postfix({
            'Ep': episode_num,
            'Success': f'{np.mean(success_history[-20:]):.1%}' if len(success_history) >= 20 else 'N/A'
        })
        
        env.close()
    
    pbar.close()
    
    # 训练完成
    print("\n" + "="*80)
    print("训练完成!")
    print("="*80)
    print(f"总Episodes: {episode_num}")
    print(f"总Steps: {total_timesteps}")
    print(f"总成功率: {np.mean(success_history):.1%}")
    print(f"最近50ep成功率: {np.mean(success_history[-50:]):.1%}")
    
    if total_verify > 0:
        print(f"\nPOLAR验证:")
        print(f"  总验证: {total_verify}")
        print(f"  安全率: {safe_count/total_verify:.1%}")
    
    # 能力估计最终统计
    print(f"\n能力估计统计:")
    cap_stats = agent.get_capability_stats()
    print(f"  跟踪目标数: {cap_stats['num_goals_tracked']}")
    print(f"  距离区间成功率:")
    for dist, stats in sorted(cap_stats['distance_bin_stats'].items()):
        print(f"    <{dist:.1f}m: {stats['success_rate']:.1%} ({stats['samples']}样本)")
    
    # 保存最终模型
    final_path = os.path.join(model_dir, "final")
    agent.save_gcpo(final_path)
    agent.save_weights(f"{final_path}/weights_for_polar.npz")
    print(f"\n✓ 最终模型: {final_path}")
    
    # 保存训练曲线
    np.savez(
        f"{log_dir}/training_history.npz",
        rewards=rewards_history,
        successes=success_history,
        goals=np.array(goal_history)
    )
    print(f"✓ 训练数据: {log_dir}/training_history.npz")
    
    return agent


def evaluate_gcpo(agent, test_goals, num_episodes_per_goal=5):
    """
    评估GCPO在不同目标上的表现
    
    Args:
        agent: GCPO智能体
        test_goals: 测试目标列表
        num_episodes_per_goal: 每个目标测试的episodes数
    
    Returns:
        results: 评估结果字典
    """
    print("\n" + "="*60)
    print("评估GCPO策略")
    print("="*60)
    
    results = {}
    
    for goal in test_goals:
        print(f"\n测试目标: ({goal[0]:.2f}, {goal[1]:.2f})")
        
        env = ClearpathNavEnv(goal_pos=goal)
        
        success_count = 0
        rewards = []
        steps_list = []
        
        for ep in range(num_episodes_per_goal):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done and episode_steps < 256:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            success = info.get('goal_reached', False)
            if success:
                success_count += 1
            
            rewards.append(episode_reward)
            steps_list.append(episode_steps)
            
            print(f"  Ep {ep+1}: R={episode_reward:6.1f}, "
                  f"Steps={episode_steps:3d}, "
                  f"{'✅' if success else '❌'}")
        
        env.close()
        
        # 统计
        results[tuple(goal)] = {
            'success_rate': success_count / num_episodes_per_goal,
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps_list)
        }
        
        print(f"  成功率: {results[tuple(goal)]['success_rate']:.1%}")
    
    print("\n" + "="*60)
    print("评估汇总:")
    print("="*60)
    for goal, stats in results.items():
        print(f"  {goal}: 成功率={stats['success_rate']:.1%}, "
              f"平均奖励={stats['avg_reward']:.1f}")
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GCPO训练')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='rl',
                       choices=['rl', 'evaluate'],
                       help='训练模式: rl=RL训练, evaluate=评估')
    
    # RL训练参数
    parser.add_argument('--bc_model', type=str, default='./models/bc_pretrained',
                       help='BC预训练模型路径')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='RL训练总步数')
    parser.add_argument('--start_timesteps', type=int, default=5000,
                       help='初始探索步数')
    parser.add_argument('--save_dir', type=str, default='./models/gcpo',
                       help='模型保存目录')
    
    # 评估参数
    parser.add_argument('--eval_model', type=str, default=None,
                       help='评估模型路径')
    parser.add_argument('--eval_goals', type=str, default='1,1;2,2;5,-2',
                       help='评估目标,格式: x1,y1;x2,y2')
    
    args = parser.parse_args()
    
    try:
        config = GCPOConfig()
        
        if args.mode == 'rl':
            # RL训练模式
            print("="*70)
            print("GCPO RL Fine-tuning 训练")
            print("="*70)
            
            # 创建智能体
            agent = GCPOAgent(
                state_dim=12,
                action_dim=2,
                max_action=0.5,
                config=config
            )
            
            # 加载BC预训练模型
            if os.path.exists(args.bc_model):
                print(f"\n加载BC预训练模型: {args.bc_model}")
                agent.load(args.bc_model)
                start_from_bc = True
            else:
                print(f"\n⚠️  BC模型不存在: {args.bc_model}")
                print("将从随机初始化开始训练")
                start_from_bc = False
            
            # 设置训练参数
            config.rl_total_timesteps = args.timesteps
            config.rl_start_timesteps = args.start_timesteps
            
            # 开始RL训练
            agent = train_gcpo_rl_phase(
                agent, 
                config,
                start_from_bc=start_from_bc,
                save_dir=args.save_dir
            )
            
        elif args.mode == 'evaluate':
            # 评估模式
            if args.eval_model is None:
                print("错误: 评估模式需要指定 --eval_model")
                sys.exit(1)
            
            print("="*70)
            print("GCPO 策略评估")
            print("="*70)
            
            # 加载模型
            agent = GCPOAgent(12, 2, 0.5, config)
            agent.load_gcpo(args.eval_model)
            
            # 解析测试目标
            test_goals = []
            for goal_str in args.eval_goals.split(';'):
                x, y = map(float, goal_str.split(','))
                test_goals.append((x, y))
            
            # 评估
            results = evaluate_gcpo(agent, test_goals, num_episodes_per_goal=5)
        
        print("\n✅ 全部完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()