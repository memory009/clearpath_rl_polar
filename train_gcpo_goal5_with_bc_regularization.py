#!/usr/bin/env python3
"""
GCPO Goal5 训练 - 带BC正则化防止遗忘
关键改进: 在RL训练时保持BC策略约束
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__))

from algorithms.gcpo_agent import GCPOAgent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
from adaptive_max_steps import AdaptiveStepsManager


class GCPOGoal5ConfigWithBCReg(TD3Config):
    """GCPO Goal5训练配置 - 带BC正则化"""
    
    # BC预训练参数
    bc_epochs = 100
    bc_batch_size = 256
    bc_learning_rate = 3e-4
    bc_val_split = 0.1
    
    # 🔥 BC正则化参数 (防止遗忘的关键!)
    bc_regularization = True
    bc_reg_lambda = 0.1  # BC正则化权重 (可调: 0.05-0.5)
    bc_demo_dir = './demonstrations_goal5'  # BC演示数据目录
    bc_reg_batch_size = 64  # 每次训练时采样的BC数据量
    
    # RL Fine-tuning参数
    rl_total_timesteps = 100000
    rl_start_timesteps = 1000  # 🔥 减少到1000步 (有BC基础不需要太多随机探索)
    
    # 🔥 降低学习率 (防止破坏BC知识)
    actor_lr = 1e-4  # 从3e-4降到1e-4
    critic_lr = 1e-3  # 保持不变
    
    # 🔥 降低探索噪声 (BC已经提供好的基础策略)
    expl_noise = 0.05  # 从0.1降到0.05
    
    # Goal5目标区域
    goal5_center = (4.5, -2.0)
    goal5_radius_small = 0.3
    goal5_radius_medium = 0.6
    goal5_radius_large = 1.0
    
    # 课程学习参数
    goal_space_bounds = [(0.5, 5.5), (-3.0, 3.5)]
    capability_window = 100
    target_capability = 0.5
    
    # 能力评估
    eval_interval = 5000
    eval_episodes = 10


class BCRegularizer:
    """BC正则化器 - 防止灾难性遗忘"""
    
    def __init__(self, demo_dir, goal_idx=5, device='cuda'):
        """
        初始化BC正则化器
        
        Args:
            demo_dir: 演示数据目录
            goal_idx: 目标索引
            device: 设备
        """
        self.device = device
        self.demo_states = []
        self.demo_actions = []
        
        # 加载演示数据
        import glob
        pattern = os.path.join(demo_dir, f'demo_goal{goal_idx}_num*.npz')
        demo_files = sorted(glob.glob(pattern))
        
        if len(demo_files) == 0:
            print(f"⚠️  警告: 未找到BC演示数据，BC正则化将被禁用")
            return
        
        print(f"\n🔥 加载BC正则化数据:")
        print(f"  目录: {demo_dir}")
        print(f"  文件数: {len(demo_files)}")
        
        for demo_file in demo_files:
            data = np.load(demo_file)
            self.demo_states.extend(data['states'])
            self.demo_actions.extend(data['actions'])
        
        self.demo_states = torch.FloatTensor(np.array(self.demo_states)).to(device)
        self.demo_actions = torch.FloatTensor(np.array(self.demo_actions)).to(device)
        
        print(f"  总样本数: {len(self.demo_states)}")
        print(f"  BC正则化已启用 ✓\n")
    
    def compute_bc_loss(self, actor, batch_size=64):
        """
        计算BC正则化损失
        
        Args:
            actor: Actor网络
            batch_size: 批次大小
        
        Returns:
            bc_loss: BC正则化损失
        """
        if len(self.demo_states) == 0:
            return torch.tensor(0.0).to(self.device)
        
        # 随机采样BC数据
        indices = np.random.randint(0, len(self.demo_states), size=batch_size)
        states = self.demo_states[indices]
        actions = self.demo_actions[indices]
        
        # 计算当前策略的动作
        predicted_actions = actor(states)
        
        # MSE损失
        bc_loss = torch.nn.functional.mse_loss(predicted_actions, actions)
        
        return bc_loss


def train_gcpo_goal5_with_bc_reg(
    agent, 
    config, 
    bc_regularizer,
    start_from_bc=True,
    save_dir='./models/gcpo_goal5_bc_reg'
):
    """
    GCPO Goal5 RL Fine-tuning - 带BC正则化
    
    Args:
        agent: 预训练的GCPO智能体
        config: 配置
        bc_regularizer: BC正则化器
        start_from_bc: 是否从BC预训练开始
        save_dir: 保存目录
    """
    print("\n" + "="*80)
    print("🚀 GCPO Goal5 RL Fine-tuning (带BC正则化)")
    print("="*80)
    
    if start_from_bc:
        print("  起点: Goal5 BC预训练策略")
    else:
        print("  起点: 随机策略")
    
    print(f"  目标中心: ({config.goal5_center[0]:.1f}, {config.goal5_center[1]:.1f})")
    print(f"  总步数: {config.rl_total_timesteps}")
    print(f"  初始探索: {config.rl_start_timesteps}步")
    print(f"  🔥 BC正则化: ✓ 启用 (lambda={config.bc_reg_lambda})")
    print(f"  🔥 Actor学习率: {config.actor_lr} (降低以保护BC知识)")
    print(f"  🔥 探索噪声: {config.expl_noise} (降低以利用BC策略)")
    print("="*80 + "\n")
    
    # 🔥 应用降低的学习率
    for param_group in agent.actor_optimizer.param_groups:
        param_group['lr'] = config.actor_lr
    
    # 切换到RL模式
    agent.switch_to_rl_mode()
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join('./logs', f'gcpo_goal5_bc_reg_{timestamp}')
    model_dir = os.path.join(save_dir, f'gcpo_goal5_bc_reg_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 训练统计
    total_timesteps = 0
    episode_num = 0
    rewards_history = []
    success_history = []
    goal_history = []
    stage_history = []
    
    # BC正则化统计
    bc_loss_history = []
    
    # POLAR验证统计
    safe_count = 0
    total_verify = 0
    
    # 自适应步数管理器
    if os.path.exists('./demonstrations_goal5'):
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations_goal5')
    else:
        steps_manager = AdaptiveStepsManager(demo_dir='./demonstrations')
    
    # Goal5渐进式课程
    goal5_center = np.array(config.goal5_center, dtype=np.float32)
    
    print(f"🎓 Goal5渐进式课程:")
    print(f"  阶段1 (0-25k步):   固定Goal5")
    print(f"  阶段2 (25k-50k步): 小扰动 ± {config.goal5_radius_small}m")
    print(f"  阶段3 (50k-75k步): 中扰动 ± {config.goal5_radius_medium}m")
    print(f"  阶段4 (75k+步):    大扰动 ± {config.goal5_radius_large}m")
    print()
    
    pbar = tqdm(total=config.rl_total_timesteps, desc="Goal5 RL训练 (BC正则化)")
    
    # 主训练循环
    while total_timesteps < config.rl_total_timesteps:
        # 渐进式课程
        if total_timesteps < 25000:
            curriculum_goal = goal5_center.copy()
            stage = "固定"
        elif total_timesteps < 50000:
            noise = np.random.uniform(-config.goal5_radius_small, 
                                     config.goal5_radius_small, size=2)
            curriculum_goal = goal5_center + noise
            stage = "小扰动"
        elif total_timesteps < 75000:
            noise = np.random.uniform(-config.goal5_radius_medium, 
                                     config.goal5_radius_medium, size=2)
            curriculum_goal = goal5_center + noise
            stage = "中扰动"
        else:
            noise = np.random.uniform(-config.goal5_radius_large, 
                                     config.goal5_radius_large, size=2)
            curriculum_goal = goal5_center + noise
            stage = "大扰动"
        
        curriculum_goal = np.clip(
            curriculum_goal,
            [config.goal_space_bounds[0][0], config.goal_space_bounds[1][0]],
            [config.goal_space_bounds[0][1], config.goal_space_bounds[1][1]]
        )
        
        adaptive_max_steps = steps_manager.get_max_steps(curriculum_goal)
        
        env = ClearpathNavEnv(
            goal_pos=tuple(curriculum_goal), 
            max_steps=adaptive_max_steps
        )
        
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        # Episode循环
        for step in range(adaptive_max_steps):
            # 选择动作
            if total_timesteps < config.rl_start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                # 🔥 降低的探索噪声
                noise = np.random.normal(0, config.max_action * config.expl_noise, 
                                        size=config.action_dim)
                action = (action + noise).clip(-config.max_action, config.max_action)
            
            next_state, reward, done, truncated, info = env.step(action)
            done_bool = float(done) if episode_steps < adaptive_max_steps else 0
            
            agent.store_transition(state, action, next_state, reward, done_bool)
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_timesteps += 1
            
            # 🔥 训练网络 (带BC正则化)
            if total_timesteps >= config.rl_start_timesteps:
                # 标准RL训练
                agent.train()
                
                # 🔥 BC正则化 (每步都应用)
                if config.bc_regularization and len(bc_regularizer.demo_states) > 0:
                    # 计算BC损失
                    bc_loss = bc_regularizer.compute_bc_loss(
                        agent.actor, 
                        batch_size=config.bc_reg_batch_size
                    )
                    
                    # BC正则化梯度更新
                    agent.actor_optimizer.zero_grad()
                    (config.bc_reg_lambda * bc_loss).backward()
                    agent.actor_optimizer.step()
                    
                    bc_loss_history.append(bc_loss.item())
            
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
        
        # Episode结束
        episode_success = info.get('goal_reached', False)
        agent.update_capability(curriculum_goal, episode_success)
        steps_manager.update_online(curriculum_goal, episode_steps, episode_success)
        
        rewards_history.append(episode_reward)
        success_history.append(episode_success)
        goal_history.append(curriculum_goal.copy())
        stage_history.append(stage)
        
        distance_to_goal5 = np.linalg.norm(curriculum_goal - goal5_center)
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
        
        pbar.write(msg)
        
        episode_num += 1
        
        # 定期统计
        if episode_num % 50 == 0:
            recent_success = np.mean(success_history[-50:])
            avg_reward = np.mean(rewards_history[-50:])
            
            pbar.write("\n" + "="*70)
            pbar.write(f"Episode {episode_num} 统计:")
            pbar.write(f"  最近50ep成功率: {recent_success:.1%}")
            pbar.write(f"  最近50ep平均奖励: {avg_reward:.1f}")
            
            # 🔥 BC正则化损失
            if len(bc_loss_history) > 0:
                recent_bc_loss = np.mean(bc_loss_history[-100:])
                pbar.write(f"  🔥 BC正则化损失: {recent_bc_loss:.6f}")
            
            # 各阶段成功率
            recent_stages = stage_history[-50:]
            recent_successes = success_history[-50:]
            stage_stats = {}
            for s, succ in zip(recent_stages, recent_successes):
                if s not in stage_stats:
                    stage_stats[s] = {'success': 0, 'total': 0}
                stage_stats[s]['total'] += 1
                if succ:
                    stage_stats[s]['success'] += 1
            
            pbar.write(f"\n  各阶段成功率:")
            for stage_name, stats in stage_stats.items():
                sr = stats['success'] / stats['total'] if stats['total'] > 0 else 0
                pbar.write(f"    {stage_name:6s}: {sr:.1%}")
            
            goal5_capability = agent.capability_estimator.estimate_capability(goal5_center)
            pbar.write(f"\n  🎯 Goal5中心能力: {goal5_capability:.1%}")
            
            pbar.write("="*70 + "\n")
            
            agent.save_gcpo(f"{model_dir}/episode_{episode_num}")
            pbar.write(f"✓ 模型已保存\n")
        
        pbar.set_postfix({
            'Ep': episode_num,
            'Stage': stage,
            'Success': f'{np.mean(success_history[-20:]):.1%}' if len(success_history) >= 20 else 'N/A'
        })
        
        env.close()
    
    pbar.close()
    
    # 训练完成统计
    print("\n" + "="*80)
    print("Goal5 训练完成 (带BC正则化)!")
    print("="*80)
    print(f"总Episodes: {episode_num}")
    print(f"总Steps: {total_timesteps}")
    print(f"总成功率: {np.mean(success_history):.1%}")
    print(f"最近50ep成功率: {np.mean(success_history[-50:]):.1%}")
    
    # BC正则化效果
    if len(bc_loss_history) > 0:
        print(f"\n🔥 BC正则化统计:")
        print(f"  平均BC损失: {np.mean(bc_loss_history):.6f}")
        print(f"  最终BC损失: {np.mean(bc_loss_history[-100:]):.6f}")
    
    # 各阶段统计
    print(f"\n各阶段整体统计:")
    unique_stages = list(dict.fromkeys(stage_history))
    for stage_name in unique_stages:
        stage_indices = [i for i, s in enumerate(stage_history) if s == stage_name]
        stage_successes = [success_history[i] for i in stage_indices]
        if len(stage_successes) > 0:
            print(f"  {stage_name:6s}: {np.mean(stage_successes):.1%} "
                  f"({sum(stage_successes)}/{len(stage_successes)})")
    
    goal5_capability = agent.capability_estimator.estimate_capability(goal5_center)
    print(f"\n🎯 Goal5中心最终能力: {goal5_capability:.1%}")
    
    # 保存最终模型
    final_path = os.path.join(model_dir, "final")
    agent.save_gcpo(final_path)
    print(f"\n✓ 最终模型: {final_path}")
    
    # 保存训练数据
    np.savez(
        f"{log_dir}/training_history.npz",
        rewards=rewards_history,
        successes=success_history,
        goals=np.array(goal_history),
        stages=stage_history,
        bc_losses=bc_loss_history if len(bc_loss_history) > 0 else []
    )
    
    return agent


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GCPO Goal5训练 (带BC正则化)')
    parser.add_argument('--bc_model', type=str, default='./models/bc_pretrained_goal5')
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--bc_reg_lambda', type=float, default=0.1,
                       help='BC正则化权重 (推荐0.05-0.5)')
    parser.add_argument('--save_dir', type=str, default='./models/gcpo_goal5_bc_reg')
    
    args = parser.parse_args()
    
    try:
        config = GCPOGoal5ConfigWithBCReg()
        config.bc_reg_lambda = args.bc_reg_lambda
        config.rl_total_timesteps = args.timesteps
        
        # 创建智能体
        agent = GCPOAgent(12, 2, 0.5, config)
        
        # 加载BC模型
        if os.path.exists(args.bc_model):
            print(f"加载BC模型: {args.bc_model}")
            agent.load(args.bc_model)
            start_from_bc = True
        else:
            print(f"⚠️  BC模型不存在，从随机初始化开始")
            start_from_bc = False
        
        # 🔥 创建BC正则化器
        bc_regularizer = BCRegularizer(
            demo_dir=config.bc_demo_dir,
            goal_idx=5,
            device=agent.device
        )
        
        # 训练
        agent = train_gcpo_goal5_with_bc_reg(
            agent,
            config,
            bc_regularizer,
            start_from_bc=start_from_bc,
            save_dir=args.save_dir
        )
        
        print("\n✅ 全部完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()