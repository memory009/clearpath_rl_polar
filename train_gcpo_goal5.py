#!/usr/bin/env python3
"""
GCPO Goal5 专用训练流程
从BC预训练(Goal5) → RL Fine-tuning (专注Goal5及其邻域)
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
from adaptive_max_steps import AdaptiveStepsManager


class GCPOGoal5Config(TD3Config):
    """GCPO Goal5训练配置"""
    
    # BC预训练参数
    bc_epochs = 100
    bc_batch_size = 256
    bc_learning_rate = 3e-4
    bc_val_split = 0.1
    
    # RL Fine-tuning参数
    rl_total_timesteps = 100000
    rl_start_timesteps = 3000  # Goal5已有BC基础,减少随机探索
    
    # Goal5目标区域
    goal5_center = (4.5, -2.0)  # Goal5中心位置
    goal5_radius_small = 0.3    # 小扰动半径
    goal5_radius_medium = 0.6   # 中等扰动半径
    goal5_radius_large = 1.0    # 大扰动半径
    
    # 课程学习参数
    goal_space_bounds = [(0.5, 5.5), (-3.0, 3.5)]
    capability_window = 100
    target_capability = 0.5
    
    # 能力评估
    eval_interval = 5000
    eval_episodes = 10


def train_gcpo_goal5_rl_phase(
    agent, 
    config, 
    start_from_bc=True,
    save_dir='./models/gcpo_goal5'
):
    """
    GCPO Goal5 RL Fine-tuning阶段
    专注于Goal5及其周围区域的训练
    
    Args:
        agent: 预训练的GCPO智能体(Goal5 BC)
        config: 配置
        start_from_bc: 是否从BC预训练开始
        save_dir: 保存目录
    
    Returns:
        agent: 训练后的智能体
    """
    print("\n" + "="*80)
    print("🚀 GCPO Goal5 RL Fine-tuning")
    print("="*80)
    
    if start_from_bc:
        print("  起点: Goal5 BC预训练策略")
    else:
        print("  起点: 随机策略")
    
    print(f"  目标中心: ({config.goal5_center[0]:.1f}, {config.goal5_center[1]:.1f})")
    print(f"  总步数: {config.rl_total_timesteps}")
    print(f"  初始探索: {config.rl_start_timesteps}步")
    print(f"  自适应课程: ✓ 启用 (专注Goal5区域)")
    print(f"  自适应步数: ✓ 启用")
    print("="*80 + "\n")
    
    # 切换到RL模式
    agent.switch_to_rl_mode()
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('./logs', f'gcpo_goal5_rl_{timestamp}')
    model_dir = os.path.join(save_dir, f'gcpo_goal5_rl_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练统计
    total_timesteps = 0
    episode_num = 0
    rewards_history = []
    success_history = []
    goal_history = []
    stage_history = []  # 记录每个episode的课程阶段
    
    # POLAR验证统计
    safe_count = 0
    total_verify = 0
    
    # 🔥 创建自适应步数管理器
    # 优先使用Goal5数据,如果不存在则使用全局数据
    if os.path.exists('./demonstrations_goal5'):
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations_goal5')
        print("✓ 使用Goal5专用演示数据初始化自适应步数管理器")
    else:
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations')
        print("✓ 使用全局演示数据初始化自适应步数管理器")
    
    # 🔥 Goal5渐进式课程设计
    goal5_center = np.array(config.goal5_center, dtype=np.float32)
    
    print(f"\n🎓 Goal5渐进式课程:")
    print(f"  阶段1 (0-25k步):   固定Goal5 - 巩固BC学习")
    print(f"  阶段2 (25k-50k步): 小扰动 - Goal5 ± {config.goal5_radius_small}m")
    print(f"  阶段3 (50k-75k步): 中扰动 - Goal5 ± {config.goal5_radius_medium}m")
    print(f"  阶段4 (75k+步):    大扰动 - Goal5 ± {config.goal5_radius_large}m")
    print()
    
    # 创建进度条
    pbar = tqdm(total=config.rl_total_timesteps, desc="Goal5 RL训练")
    
    # 主训练循环
    while total_timesteps < config.rl_total_timesteps:
        # 🔥 Goal5渐进式课程: 根据训练进度选择目标
        if total_timesteps < 25000:
            # 阶段1: 固定Goal5 (巩固BC学习的知识)
            curriculum_goal = goal5_center.copy()
            stage = "固定"
            
        elif total_timesteps < 50000:
            # 阶段2: Goal5 + 小扰动 (开始局部泛化)
            noise = np.random.uniform(-config.goal5_radius_small, 
                                     config.goal5_radius_small, size=2)
            curriculum_goal = goal5_center + noise
            stage = "小扰动"
            
        elif total_timesteps < 75000:
            # 阶段3: Goal5 + 中等扰动 (进一步泛化)
            noise = np.random.uniform(-config.goal5_radius_medium, 
                                     config.goal5_radius_medium, size=2)
            curriculum_goal = goal5_center + noise
            stage = "中扰动"
            
        else:
            # 阶段4: Goal5 + 大扰动 (充分探索Goal5邻域)
            noise = np.random.uniform(-config.goal5_radius_large, 
                                     config.goal5_radius_large, size=2)
            curriculum_goal = goal5_center + noise
            stage = "大扰动"
        
        # 确保目标在合理范围内
        curriculum_goal = np.clip(
            curriculum_goal,
            [config.goal_space_bounds[0][0], config.goal_space_bounds[1][0]],
            [config.goal_space_bounds[0][1], config.goal_space_bounds[1][1]]
        )
        
        # 🔥 使用自适应步数管理器
        adaptive_max_steps = steps_manager.get_max_steps(curriculum_goal)
        
        # 创建环境
        env = ClearpathNavEnv(
            goal_pos=tuple(curriculum_goal), 
            max_steps=adaptive_max_steps
        )
        
        # 重置环境
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Episode循环
        for step in range(adaptive_max_steps):
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
            done_bool = float(done) if episode_steps < adaptive_max_steps else 0
            
            # 存储经验
            agent.store_transition(state, action, next_state, reward, done_bool)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # 训练网络
            if total_timesteps >= config.rl_start_timesteps:
                agent.train()
            
            # POLAR验证
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
        
        # 🔥 更新自适应步数统计
        steps_manager.update_online(curriculum_goal, episode_steps, episode_success)
        
        # 记录统计
        rewards_history.append(episode_reward)
        success_history.append(episode_success)
        goal_history.append(curriculum_goal.copy())
        stage_history.append(stage)
        
        # 计算到Goal5中心的距离
        distance_to_goal5 = np.linalg.norm(curriculum_goal - goal5_center)
        
        # 打印信息
        capability = agent.capability_estimator.estimate_capability(curriculum_goal)
        msg = (f"Ep {episode_num} [{stage:4s}]: "
               f"Goal=({curriculum_goal[0]:.2f},{curriculum_goal[1]:.2f}), "
               f"Dist2G5={distance_to_goal5:.2f}m, "
               f"R={episode_reward:.1f}, "
               f"Steps={episode_steps}/{adaptive_max_steps}, "
               f"Cap={capability:.1%}")
        
        if episode_success:
            msg += " 🎯"
        elif info.get('collision'):
            msg += " ⚠️"
        elif episode_steps >= adaptive_max_steps:
            msg += " ⏱️"
        
        pbar.write(msg)
        
        episode_num += 1
        
        # 定期统计和保存
        if episode_num % 50 == 0:
            recent_success = np.mean(success_history[-50:])
            avg_reward = np.mean(rewards_history[-50:])
            
            # 统计各阶段的成功率
            recent_stages = stage_history[-50:]
            recent_successes = success_history[-50:]
            stage_stats = {}
            for s, succ in zip(recent_stages, recent_successes):
                if s not in stage_stats:
                    stage_stats[s] = {'success': 0, 'total': 0}
                stage_stats[s]['total'] += 1
                if succ:
                    stage_stats[s]['success'] += 1
            
            pbar.write("\n" + "="*70)
            pbar.write(f"Episode {episode_num} 统计 (总步数: {total_timesteps}):")
            pbar.write(f"  最近50ep成功率: {recent_success:.1%}")
            pbar.write(f"  最近50ep平均奖励: {avg_reward:.1f}")
            pbar.write(f"  Memory大小: {agent.memory.size}")
            
            if total_verify > 0:
                pbar.write(f"  POLAR安全率: {safe_count/total_verify:.1%}")
            
            # 各阶段成功率
            pbar.write(f"\n  各阶段成功率 (最近50ep):")
            for stage_name, stats in stage_stats.items():
                sr = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                pbar.write(f"    {stage_name:6s}: {sr:.1%} ({stats['success']}/{stats['total']})")
            
            # 能力估计统计
            cap_stats = agent.get_capability_stats()
            pbar.write(f"\n  能力估计:")
            pbar.write(f"    跟踪目标数: {cap_stats['num_goals_tracked']}")
            
            # 🔥 自适应步数统计
            steps_stats = steps_manager.get_statistics_summary()
            pbar.write(f"\n  自适应步数统计:")
            pbar.write(f"    跟踪目标数: {steps_stats['num_goals']}")
            pbar.write(f"    平均步数: {steps_stats['avg_steps_overall']:.0f}")
            pbar.write(f"    步数范围: {steps_stats['min_steps_overall']:.0f}-{steps_stats['max_steps_overall']:.0f}")
            
            # Goal5中心区域的能力
            goal5_capability = agent.capability_estimator.estimate_capability(goal5_center)
            pbar.write(f"\n  Goal5中心能力: {goal5_capability:.1%}")
            
            pbar.write("\n  距离区间成功率:")
            for dist, stats in sorted(cap_stats['distance_bin_stats'].items()):
                pbar.write(f"    <{dist:.1f}m: {stats['success_rate']:.1%} "
                          f"({stats['samples']}样本)")
            
            pbar.write("="*70 + "\n")
            
            # 保存模型
            agent.save_gcpo(f"{model_dir}/episode_{episode_num}")
            pbar.write(f"✓ 模型已保存: episode_{episode_num}\n")
        
        # 更新进度条显示
        recent_20_success = np.mean(success_history[-20:]) if len(success_history) >= 20 else 0
        pbar.set_postfix({
            'Ep': episode_num,
            'Stage': stage,
            'Success': f'{recent_20_success:.1%}'
        })
        
        env.close()
    
    pbar.close()
    
    # 训练完成
    print("\n" + "="*80)
    print("Goal5 训练完成!")
    print("="*80)
    print(f"总Episodes: {episode_num}")
    print(f"总Steps: {total_timesteps}")
    print(f"总成功率: {np.mean(success_history):.1%}")
    print(f"最近50ep成功率: {np.mean(success_history[-50:]):.1%}")
    
    # 各阶段统计
    print(f"\n各阶段整体统计:")
    unique_stages = list(dict.fromkeys(stage_history))  # 保持顺序的去重
    for stage_name in unique_stages:
        stage_indices = [i for i, s in enumerate(stage_history) if s == stage_name]
        stage_successes = [success_history[i] for i in stage_indices]
        if len(stage_successes) > 0:
            print(f"  {stage_name:6s}: {np.mean(stage_successes):.1%} "
                  f"({sum(stage_successes)}/{len(stage_successes)})")
    
    if total_verify > 0:
        print(f"\nPOLAR验证:")
        print(f"  总验证: {total_verify}")
        print(f"  安全率: {safe_count/total_verify:.1%}")
    
    # Goal5中心区域的最终能力
    goal5_capability = agent.capability_estimator.estimate_capability(goal5_center)
    print(f"\nGoal5中心最终能力: {goal5_capability:.1%}")
    
    # 能力估计最终统计
    print(f"\n能力估计统计:")
    cap_stats = agent.get_capability_stats()
    print(f"  跟踪目标数: {cap_stats['num_goals_tracked']}")
    print(f"  距离区间成功率:")
    for dist, stats in sorted(cap_stats['distance_bin_stats'].items()):
        print(f"    <{dist:.1f}m: {stats['success_rate']:.1%} ({stats['samples']}样本)")
    
    # 🔥 自适应步数最终统计
    print(f"\n自适应步数最终统计:")
    steps_stats = steps_manager.get_statistics_summary()
    print(f"  跟踪目标数: {steps_stats['num_goals']}")
    print(f"  平均步数: {steps_stats['avg_steps_overall']:.0f}")
    print(f"  步数范围: {steps_stats['min_steps_overall']:.0f}-{steps_stats['max_steps_overall']:.0f}")
    
    # 保存最终模型
    final_path = os.path.join(model_dir, "final")
    agent.save_gcpo(final_path)
    agent.save_weights(f"{final_path}/weights_for_polar.npz")
    print(f"\n✓ 最终模型: {final_path}")
    
    # 🔥 保存自适应步数统计
    steps_manager.save(f"{model_dir}/adaptive_steps_goal5.pkl")
    
    # 保存训练曲线
    np.savez(
        f"{log_dir}/training_history.npz",
        rewards=rewards_history,
        successes=success_history,
        goals=np.array(goal_history),
        stages=stage_history
    )
    print(f"✓ 训练数据: {log_dir}/training_history.npz")
    
    # 保存Goal5专用信息
    with open(f"{model_dir}/goal5_info.txt", 'w') as f:
        f.write(f"Goal5 Training Information\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Goal5 Center: {config.goal5_center}\n")
        f.write(f"Total Episodes: {episode_num}\n")
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"Overall Success Rate: {np.mean(success_history):.1%}\n")
        f.write(f"Recent 50ep Success Rate: {np.mean(success_history[-50:]):.1%}\n")
        f.write(f"Goal5 Center Capability: {goal5_capability:.1%}\n")
        f.write(f"\nStage Statistics:\n")
        for stage_name in unique_stages:
            stage_indices = [i for i, s in enumerate(stage_history) if s == stage_name]
            stage_successes = [success_history[i] for i in stage_indices]
            if len(stage_successes) > 0:
                f.write(f"  {stage_name}: {np.mean(stage_successes):.1%} "
                       f"({sum(stage_successes)}/{len(stage_successes)})\n")
    
    print(f"✓ Goal5信息: {model_dir}/goal5_info.txt")
    
    return agent


def evaluate_gcpo_goal5(agent, config, num_episodes=20):
    """
    评估GCPO在Goal5及其邻域的表现
    
    Args:
        agent: GCPO智能体
        config: 配置对象
        num_episodes: 每个测试点的episodes数
    
    Returns:
        results: 评估结果字典
    """
    print("\n" + "="*70)
    print("评估GCPO Goal5策略")
    print("="*70)
    
    # 🔥 评估时也使用自适应步数
    if os.path.exists('./demonstrations_goal5'):
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations_goal5')
    else:
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations')
    
    goal5_center = np.array(config.goal5_center, dtype=np.float32)
    
    # 测试点: Goal5中心 + 周围不同距离的点
    test_configs = [
        ("Goal5中心", goal5_center),
        ("小扰动+X", goal5_center + [0.3, 0.0]),
        ("小扰动-X", goal5_center + [-0.3, 0.0]),
        ("小扰动+Y", goal5_center + [0.0, 0.3]),
        ("小扰动-Y", goal5_center + [0.0, -0.3]),
        ("中扰动", goal5_center + [0.5, 0.5]),
        ("大扰动", goal5_center + [0.8, -0.8]),
    ]
    
    results = {}
    
    for name, goal in test_configs:
        print(f"\n测试点: {name} - ({goal[0]:.2f}, {goal[1]:.2f})")
        
        # 获取自适应步数
        adaptive_max_steps = steps_manager.get_max_steps(goal)
        print(f"  自适应max_steps: {adaptive_max_steps}")
        
        env = ClearpathNavEnv(goal_pos=tuple(goal), max_steps=adaptive_max_steps)
        
        success_count = 0
        collision_count = 0
        timeout_count = 0
        rewards = []
        steps_list = []
        distances = []
        
        for ep in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            
            done = False
            while not done and episode_steps < adaptive_max_steps:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                state = next_state
                
                if done or truncated:
                    break
            
            success = info.get('goal_reached', False)
            collision = info.get('collision', False)
            
            if success:
                success_count += 1
            if collision:
                collision_count += 1
            if episode_steps >= adaptive_max_steps:
                timeout_count += 1
            
            rewards.append(episode_reward)
            steps_list.append(episode_steps)
            distances.append(info.get('distance', 0))
            
            status = '✅' if success else ('💥' if collision else '⏱️')
            print(f"  Ep {ep+1:2d}: R={episode_reward:7.1f}, "
                  f"Steps={episode_steps:3d}/{adaptive_max_steps}, "
                  f"Dist={distances[-1]:5.2f}m {status}")
        
        env.close()
        
        # 统计
        results[name] = {
            'goal_position': tuple(goal),
            'success_rate': success_count / num_episodes,
            'collision_rate': collision_count / num_episodes,
            'timeout_rate': timeout_count / num_episodes,
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps_list),
            'avg_final_distance': np.mean(distances),
            'min_distance': np.min(distances)
        }
        
        print(f"  统计: 成功={results[name]['success_rate']:.1%}, "
              f"碰撞={results[name]['collision_rate']:.1%}, "
              f"超时={results[name]['timeout_rate']:.1%}")
    
    print("\n" + "="*70)
    print("Goal5 评估汇总:")
    print("="*70)
    for name, stats in results.items():
        print(f"\n  {name}:")
        print(f"    位置: {stats['goal_position']}")
        print(f"    成功率: {stats['success_rate']:.1%}")
        print(f"    平均奖励: {stats['avg_reward']:.1f}")
        print(f"    平均步数: {stats['avg_steps']:.1f}")
        print(f"    最终距离: {stats['avg_final_distance']:.2f}m (最小={stats['min_distance']:.2f}m)")
    print("="*70 + "\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GCPO Goal5专用训练')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='rl',
                       choices=['rl', 'evaluate'],
                       help='训练模式: rl=RL训练, evaluate=评估')
    
    # RL训练参数
    parser.add_argument('--bc_model', type=str, default='./models/bc_pretrained_goal5',
                       help='Goal5 BC预训练模型路径')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='RL训练总步数')
    parser.add_argument('--start_timesteps', type=int, default=3000,
                       help='初始探索步数')
    parser.add_argument('--save_dir', type=str, default='./models/gcpo_goal5',
                       help='模型保存目录')
    
    # Goal5参数
    parser.add_argument('--goal5_x', type=float, default=4.5,
                       help='Goal5 X坐标')
    parser.add_argument('--goal5_y', type=float, default=-2.0,
                       help='Goal5 Y坐标')
    
    # 评估参数
    parser.add_argument('--eval_model', type=str, default=None,
                       help='评估模型路径')
    parser.add_argument('--eval_episodes', type=int, default=20,
                       help='每个测试点的episodes数')
    
    args = parser.parse_args()
    
    try:
        config = GCPOGoal5Config()
        config.goal5_center = (args.goal5_x, args.goal5_y)
        
        if args.mode == 'rl':
            # RL训练模式
            print("="*70)
            print("GCPO Goal5 RL Fine-tuning 训练")
            print("="*70)
            print(f"Goal5位置: ({args.goal5_x}, {args.goal5_y})")
            
            # 创建智能体
            agent = GCPOAgent(
                state_dim=12,
                action_dim=2,
                max_action=0.5,
                config=config
            )
            
            # 加载Goal5 BC预训练模型
            if os.path.exists(args.bc_model):
                print(f"\n加载Goal5 BC预训练模型: {args.bc_model}")
                agent.load(args.bc_model)
                start_from_bc = True
            else:
                print(f"\n⚠️  Goal5 BC模型不存在: {args.bc_model}")
                print("将从随机初始化开始训练")
                start_from_bc = False
            
            # 设置训练参数
            config.rl_total_timesteps = args.timesteps
            config.rl_start_timesteps = args.start_timesteps
            
            # 开始RL训练
            agent = train_gcpo_goal5_rl_phase(
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
            print("GCPO Goal5 策略评估")
            print("="*70)
            
            # 加载模型
            agent = GCPOAgent(12, 2, 0.5, config)
            agent.load_gcpo(args.eval_model)
            
            # 评估
            results = evaluate_gcpo_goal5(agent, config, num_episodes=args.eval_episodes)
            
            # 保存评估结果
            eval_save_path = os.path.join(args.eval_model, 'evaluation_results_goal5.npz')
            
            # 准备保存的数据
            save_data = {}
            for name, stats in results.items():
                for key, value in stats.items():
                    save_data[f"{name}_{key}"] = value
            
            np.savez(eval_save_path, **save_data)
            print(f"✓ 评估结果已保存: {eval_save_path}")
        
        print("\n✅ 全部完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()