#!/usr/bin/env python3
"""
Behavioral Cloning 预训练 - 专门针对 Goal5
从Goal5的人工演示中学习初始策略
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
import glob

sys.path.append(os.path.dirname(__file__))
from algorithms.td3_polar import TD3Agent
from algorithms.networks import Actor
from utils.config import TD3Config


class Goal5DemonstrationDataset(Dataset):
    """Goal5演示数据集"""
    
    def __init__(self, demo_dir, goal_idx=5, verbose=True):
        """
        加载Goal5演示数据
        
        Args:
            demo_dir: 演示数据目录
            goal_idx: 目标索引 (默认5)
            verbose: 是否显示详细信息
        """
        self.states = []
        self.actions = []
        self.goals = []
        self.goal_idx = goal_idx
        
        # 加载Goal5的演示文件
        # 支持两种命名格式:
        # 1. demo_goal5_num*.npz (来自 collect_goal5_supplement.py)
        # 2. demo_goal4_num*.npz (来自 collect_demonstrations.py, 如果goal_idx=4)
        pattern = os.path.join(demo_dir, f'demo_goal{goal_idx}_num*.npz')
        demo_files = sorted(glob.glob(pattern))
        
        if len(demo_files) == 0:
            raise ValueError(f"在 {demo_dir} 中没有找到 Goal{goal_idx} 的演示数据文件!\n"
                           f"  查找模式: {pattern}\n"
                           f"  请确认:\n"
                           f"    1. 目录路径正确\n"
                           f"    2. 已经收集了Goal{goal_idx}的演示数据\n"
                           f"    3. 文件命名格式为 demo_goal{goal_idx}_num*.npz")
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"加载 Goal{goal_idx} 演示数据")
            print(f"{'='*70}")
            print(f"  目录: {demo_dir}")
            print(f"  文件数: {len(demo_files)}")
        
        total_steps = 0
        goal_positions = []
        
        for demo_file in demo_files:
            data = np.load(demo_file)
            
            states = data['states']
            actions = data['actions']
            goals = data['goals']
            
            # 验证goal_idx
            file_goal_idx = data.get('goal_idx', None)
            if file_goal_idx is not None and file_goal_idx != goal_idx:
                if verbose:
                    print(f"  ⚠️  跳过文件 (goal_idx不匹配): {os.path.basename(demo_file)}")
                continue
            
            # 添加到数据集
            self.states.extend(states)
            self.actions.extend(actions)
            self.goals.extend(goals)
            
            total_steps += len(states)
            goal_positions.append(goals[0])
        
        # 转换为numpy数组
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.goals = np.array(self.goals, dtype=np.float32)
        
        if verbose:
            print(f"  总episodes: {len(demo_files)}")
            print(f"  总步数: {total_steps}")
            print(f"  平均步数/episode: {total_steps/len(demo_files):.1f}")
            
            # 显示目标位置统计
            if len(goal_positions) > 0:
                goal_mean = np.mean(goal_positions, axis=0)
                goal_std = np.std(goal_positions, axis=0)
                print(f"\n  Goal{goal_idx} 目标位置统计:")
                print(f"    平均: ({goal_mean[0]:.2f}, {goal_mean[1]:.2f})")
                print(f"    标准差: ({goal_std[0]:.3f}, {goal_std[1]:.3f})")
            
            print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        """
        获取一个样本
        
        Returns:
            state: 观测状态
            action: 对应的动作
            goal: 目标位置
        """
        return (
            torch.FloatTensor(self.states[idx]),
            torch.FloatTensor(self.actions[idx]),
            torch.FloatTensor(self.goals[idx])
        )


class BCTrainer:
    """行为克隆训练器"""
    
    def __init__(self, agent, config):
        """
        初始化BC训练器
        
        Args:
            agent: TD3Agent对象
            config: 训练配置
        """
        self.agent = agent
        self.config = config
        self.device = agent.device
        
        # BC专用的优化器
        self.optimizer = optim.Adam(
            agent.actor.parameters(), 
            lr=config.bc_learning_rate
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练统计
        self.train_losses = []
        self.val_losses = []
        
        print(f"✓ BC训练器初始化完成")
        print(f"  学习率: {config.bc_learning_rate}")
        print(f"  Batch大小: {config.bc_batch_size}")
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.agent.actor.train()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for states, actions, goals in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            
            # 前向传播
            predicted_actions = self.agent.actor(states)
            
            # 计算损失
            loss = self.criterion(predicted_actions, actions)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.agent.actor.parameters(), 
                max_norm=1.0
            )
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """验证"""
        self.agent.actor.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, actions, goals in val_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                predicted_actions = self.agent.actor(states)
                loss = self.criterion(predicted_actions, actions)
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_loss = val_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
        
        Returns:
            best_val_loss: 最佳验证损失
        """
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10  # Early stopping耐心值
        
        print(f"\n{'='*70}")
        print(f"开始BC训练 - Goal5专用")
        print(f"{'='*70}")
        print(f"  Epochs: {num_epochs}")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  验证样本: {len(val_loader.dataset)}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 打印进度
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  ✓ 新的最佳模型 (val_loss={val_loss:.6f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⚠️  验证损失在{patience}个epoch内未改善,提前停止")
                break
        
        print(f"\n{'='*70}")
        print(f"BC训练完成")
        print(f"{'='*70}")
        print(f"  最佳验证损失: {best_val_loss:.6f}")
        print(f"  最终训练损失: {self.train_losses[-1]:.6f}")
        print(f"{'='*70}\n")
        
        return best_val_loss


def pretrain_goal5_from_demonstrations(
    demo_dir='./demonstrations_goal5',
    model_save_path='./models/bc_pretrained_goal5',
    goal_idx=5,
    num_epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    val_split=0.1,
    seed=42
):
    """
    从Goal5演示数据进行BC预训练的主函数
    
    Args:
        demo_dir: Goal5演示数据目录
        model_save_path: 模型保存路径
        goal_idx: 目标索引 (默认5)
        num_epochs: 训练轮数
        batch_size: Batch大小
        learning_rate: 学习率
        val_split: 验证集比例
        seed: 随机种子
    
    Returns:
        agent: 预训练后的智能体
        best_val_loss: 最佳验证损失
    """
    print("\n" + "="*70)
    print("🎯 Behavioral Cloning 预训练 - Goal5 专用")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. 加载Goal5演示数据
    dataset = Goal5DemonstrationDataset(demo_dir, goal_idx=goal_idx, verbose=True)
    
    # 2. 划分训练集和验证集
    num_val = int(len(dataset) * val_split)
    num_train = len(dataset) - num_val
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [num_train, num_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"数据划分:")
    print(f"  训练集: {num_train} 样本")
    print(f"  验证集: {num_val} 样本")
    
    # 3. 创建智能体
    config = TD3Config()
    config.bc_learning_rate = learning_rate
    config.bc_batch_size = batch_size
    
    agent = TD3Agent(
        state_dim=12,
        action_dim=2,
        max_action=0.5,
        config=config
    )
    
    # 4. 创建BC训练器
    trainer = BCTrainer(agent, config)
    
    # 5. 训练
    best_val_loss = trainer.train(train_loader, val_loader, num_epochs)
    
    # 6. 保存模型
    os.makedirs(model_save_path, exist_ok=True)
    agent.save(model_save_path)
    
    # 保存训练曲线
    np.savez(
        os.path.join(model_save_path, 'bc_training_curves_goal5.npz'),
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses,
        goal_idx=goal_idx
    )
    
    # 保存训练信息
    with open(os.path.join(model_save_path, 'training_info.txt'), 'w') as f:
        f.write(f"Goal5 BC Pretraining\n")
        f.write(f"{'='*50}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Demo Directory: {demo_dir}\n")
        f.write(f"Goal Index: {goal_idx}\n")
        f.write(f"Total Samples: {len(dataset)}\n")
        f.write(f"Training Samples: {num_train}\n")
        f.write(f"Validation Samples: {num_val}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
        f.write(f"Final Train Loss: {trainer.train_losses[-1]:.6f}\n")
    
    print(f"\n{'='*70}")
    print(f"✅ Goal5 预训练完成")
    print(f"{'='*70}")
    print(f"  模型保存: {model_save_path}")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"  训练信息: {model_save_path}/training_info.txt")
    print(f"{'='*70}\n")
    
    return agent, best_val_loss


def evaluate_goal5_bc_policy(agent, env, num_episodes=10):
    """
    评估Goal5 BC预训练的策略
    
    Args:
        agent: 预训练的智能体
        env: 环境
        num_episodes: 测试episode数
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*70}")
    print(f"评估 Goal5 BC预训练策略")
    print(f"{'='*70}")

    max_steps = env.max_steps
    print(f"评估时的最大步数: {max_steps}")
    print(f"目标位置: ({env.world_goal[0]:.2f}, {env.world_goal[1]:.2f})")
    
    success_count = 0
    total_rewards = []
    total_steps = []
    final_distances = []
    collision_count = 0
    timeout_count = 0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        done = False
        while not done and episode_steps < max_steps:
            # 使用策略选择动作(无噪声)
            action = agent.select_action(state)
            
            # 执行
            next_state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
            
            if done or truncated:
                break
        
        # 统计
        success = info.get('goal_reached', False)
        collision = info.get('collision', False)
        
        if success:
            success_count += 1
        if collision:
            collision_count += 1
        if episode_steps >= max_steps:
            timeout_count += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        final_distances.append(info.get('distance', 0))
        
        status_icon = '✅' if success else ('💥' if collision else '⏱️')
        print(f"  Episode {ep+1:2d}: "
              f"Reward={episode_reward:7.1f}, "
              f"Steps={episode_steps:3d}, "
              f"Dist={final_distances[-1]:5.2f}m "
              f"{status_icon}")
    
    # 结果
    results = {
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'timeout_rate': timeout_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'avg_final_distance': np.mean(final_distances),
        'min_distance': np.min(final_distances),
        'max_distance': np.max(final_distances)
    }
    
    print(f"\n{'='*70}")
    print(f"Goal5 评估结果:")
    print(f"{'='*70}")
    print(f"  成功率: {results['success_rate']:.1%} ({success_count}/{num_episodes})")
    print(f"  碰撞率: {results['collision_rate']:.1%}")
    print(f"  超时率: {results['timeout_rate']:.1%}")
    print(f"  平均奖励: {results['avg_reward']:.1f}")
    print(f"  平均步数: {results['avg_steps']:.1f}")
    print(f"  最终距离: {results['avg_final_distance']:.2f}m (min={results['min_distance']:.2f}, max={results['max_distance']:.2f})")
    print(f"{'='*70}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Goal5 BC预训练')
    parser.add_argument('--demo_dir', type=str, default='./demonstrations_goal5',
                       help='Goal5演示数据目录')
    parser.add_argument('--save_path', type=str, default='./models/bc_pretrained_goal5',
                       help='模型保存路径')
    parser.add_argument('--goal_idx', type=int, default=5,
                       help='目标索引 (默认5)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='验证集比例')
    parser.add_argument('--evaluate', action='store_true',
                       help='训练后评估策略')
    parser.add_argument('--eval_goal', type=str, default='4.5,-2.0',
                       help='评估目标位置 (默认Goal5位置)')
    parser.add_argument('--eval_max_steps', type=int, default=1024,
                       help='评估时的最大步数')
    parser.add_argument('--eval_episodes', type=int, default=10,
                       help='评估episode数')
    
    args = parser.parse_args()
    
    try:
        # Goal5 BC预训练
        agent, best_val_loss = pretrain_goal5_from_demonstrations(
            demo_dir=args.demo_dir,
            model_save_path=args.save_path,
            goal_idx=args.goal_idx,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split
        )
        
        # 可选：评估策略
        if args.evaluate:
            print("\n开始评估Goal5预训练策略...")
            
            from envs.clearpath_nav_env import ClearpathNavEnv
            import rclpy
            
            if not rclpy.ok():
                rclpy.init()
            
            eval_goal = tuple(map(float, args.eval_goal.split(',')))
            env = ClearpathNavEnv(goal_pos=eval_goal, max_steps=args.eval_max_steps)
            
            results = evaluate_goal5_bc_policy(agent, env, num_episodes=args.eval_episodes)
            
            # 保存评估结果
            eval_results_path = os.path.join(args.save_path, 'evaluation_results.npz')
            np.savez(eval_results_path, **results)
            print(f"评估结果已保存: {eval_results_path}")
            
            env.close()
        
        print("\n✅ 全部完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()