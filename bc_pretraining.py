#!/usr/bin/env python3
"""
Behavioral Cloning 预训练
从人工演示中学习初始策略
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


class DemonstrationDataset(Dataset):
    """演示数据集"""
    
    def __init__(self, demo_dir, verbose=True):
        """
        加载演示数据
        
        Args:
            demo_dir: 演示数据目录
            verbose: 是否显示详细信息
        """
        self.states = []
        self.actions = []
        self.goals = []
        
        # 加载所有演示文件
        demo_files = sorted(glob.glob(os.path.join(demo_dir, '*.npz')))
        
        if len(demo_files) == 0:
            raise ValueError(f"在 {demo_dir} 中没有找到演示数据文件!")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"加载演示数据")
            print(f"{'='*60}")
            print(f"  目录: {demo_dir}")
            print(f"  文件数: {len(demo_files)}")
        
        total_steps = 0
        goal_distribution = {}
        
        for demo_file in demo_files:
            data = np.load(demo_file)
            
            states = data['states']
            actions = data['actions']
            goals = data['goals']
            
            # 添加到数据集
            self.states.extend(states)
            self.actions.extend(actions)
            self.goals.extend(goals)
            
            total_steps += len(states)
            
            # 统计目标分布
            goal_key = tuple(np.round(goals[0], 1))
            goal_distribution[goal_key] = goal_distribution.get(goal_key, 0) + 1
        
        # 转换为numpy数组
        self.states = np.array(self.states, dtype=np.float32)
        self.actions = np.array(self.actions, dtype=np.float32)
        self.goals = np.array(self.goals, dtype=np.float32)
        
        if verbose:
            print(f"  总episodes: {len(demo_files)}")
            print(f"  总步数: {total_steps}")
            print(f"  平均步数/episode: {total_steps/len(demo_files):.1f}")
            print(f"\n  目标分布:")
            for goal_pos, count in sorted(goal_distribution.items()):
                print(f"    {goal_pos}: {count} episodes")
            print(f"{'='*60}\n")
    
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
        
        # BC专用的优化器(学习率可能不同)
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
            
            # 梯度裁剪(防止梯度爆炸)
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
        
        print(f"\n{'='*60}")
        print(f"开始BC训练")
        print(f"{'='*60}")
        print(f"  Epochs: {num_epochs}")
        print(f"  训练样本: {len(train_loader.dataset)}")
        print(f"  验证样本: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")
        
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
        
        print(f"\n{'='*60}")
        print(f"BC训练完成")
        print(f"{'='*60}")
        if num_epochs > 0 and len(self.train_losses) > 0:
            print(f"  最佳验证损失: {best_val_loss:.6f}")
            print(f"  最终训练损失: {self.train_losses[-1]:.6f}")
        else:
            print(f"  ⚠️  未进行训练 (epochs=0)")
        print(f"{'='*60}\n")
        
        return best_val_loss


def pretrain_from_demonstrations(
    demo_dir='./demonstrations',
    model_save_path='./models/bc_pretrained',
    num_epochs=100,
    batch_size=256,
    learning_rate=3e-4,
    val_split=0.1,
    seed=42
):
    """
    从演示数据进行BC预训练的主函数
    
    Args:
        demo_dir: 演示数据目录
        model_save_path: 模型保存路径
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
    print("🎓 Behavioral Cloning 预训练")
    print("="*70)
    
    # 设置随机种子
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. 加载演示数据
    dataset = DemonstrationDataset(demo_dir, verbose=True)
    
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
        os.path.join(model_save_path, 'bc_training_curves.npz'),
        train_losses=trainer.train_losses,
        val_losses=trainer.val_losses
    )
    
    print(f"\n{'='*70}")
    print(f"✅ 预训练完成")
    print(f"{'='*70}")
    print(f"  模型保存: {model_save_path}")
    print(f"  最佳验证损失: {best_val_loss:.6f}")
    print(f"{'='*70}\n")
    
    return agent, best_val_loss


def evaluate_bc_policy(agent, env, num_episodes=10):
    """
    评估BC预训练的策略
    
    Args:
        agent: 预训练的智能体
        env: 环境
        num_episodes: 测试episode数
    
    Returns:
        results: 评估结果字典
    """
    print(f"\n{'='*60}")
    print(f"评估BC预训练策略")
    print(f"{'='*60}")

    max_steps = env.max_steps
    print(f"评估时的最大步数: {max_steps}")
    
    success_count = 0
    total_rewards = []
    total_steps = []
    final_distances = []
    
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
        if success:
            success_count += 1
        
        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        final_distances.append(info.get('distance', 0))
        
        print(f"  Episode {ep+1:2d}: "
              f"Reward={episode_reward:6.1f}, "
              f"Steps={episode_steps:3d}, "
              f"Success={'✅' if success else '❌'}")
    
    # 结果
    results = {
        'success_rate': success_count / num_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_steps': np.mean(total_steps),
        'avg_final_distance': np.mean(final_distances)
    }
    
    print(f"\n{'='*60}")
    print(f"评估结果:")
    print(f"{'='*60}")
    print(f"  成功率: {results['success_rate']:.1%}")
    print(f"  平均奖励: {results['avg_reward']:.1f}")
    print(f"  平均步数: {results['avg_steps']:.1f}")
    print(f"  平均最终距离: {results['avg_final_distance']:.2f}m")
    print(f"{'='*60}\n")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BC预训练')
    parser.add_argument('--demo_dir', type=str, default='./demonstrations',
                       help='演示数据目录')
    parser.add_argument('--save_path', type=str, default='./models/bc_pretrained',
                       help='模型保存路径')
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
    parser.add_argument('--eval_goal', type=str, default='5.0,-0.5',
                       help='评估目标位置')
    parser.add_argument('--eval_max_steps', type=int, default=1024,
                   help='评估时的最大步数')
    
    args = parser.parse_args()
    
    try:
        # BC预训练
        agent, best_val_loss = pretrain_from_demonstrations(
            demo_dir=args.demo_dir,
            model_save_path=args.save_path,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_split=args.val_split
        )
        
        # 可选：评估策略
        if args.evaluate:
            from envs.clearpath_nav_env import ClearpathNavEnv
            
            eval_goal = tuple(map(float, args.eval_goal.split(',')))
            env = ClearpathNavEnv(goal_pos=eval_goal, max_steps=args.eval_max_steps)
            
            results = evaluate_bc_policy(agent, env, num_episodes=10)
            
            env.close()
        
        print("\n✅ 全部完成!")
        
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()