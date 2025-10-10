#!/usr/bin/env python3
"""
神经网络定义模块
与论文完全一致的网络结构：26神经元隐藏层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """
    Actor网络（策略网络）
    
    结构：
        输入：12维观测 [距离, 角度, 8激光, 2速度]
        隐藏层1：26神经元 + ReLU
        隐藏层2：26神经元 + ReLU  
        输出层：2维动作 [线速度, 角速度] + Tanh
    """
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 26)
        self.l2 = nn.Linear(26, 26)
        self.l3 = nn.Linear(26, action_dim)
        
        self.max_action = max_action
    
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.max_action * torch.tanh(self.l3(a))
        return a


class Critic(nn.Module):
    """
    Critic网络（价值网络）- 双Q网络
    
    结构（Q1和Q2相同）：
        输入：14维 [12状态 + 2动作]
        隐藏层1：26神经元 + ReLU
        隐藏层2：26神经元 + ReLU
        输出层：1维Q值
    """
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1网络
        self.l1 = nn.Linear(state_dim + action_dim, 26)
        self.l2 = nn.Linear(26, 26)
        self.l3 = nn.Linear(26, 1)
        
        # Q2网络
        self.l4 = nn.Linear(state_dim + action_dim, 26)
        self.l5 = nn.Linear(26, 26)
        self.l6 = nn.Linear(26, 1)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("="*60)
    print("测试神经网络")
    print("="*60)
    
    # 创建网络
    print("\n[1/3] 创建Actor网络...")
    actor = Actor(state_dim=12, action_dim=2, max_action=0.5)
    print(f"✓ Actor创建成功")
    print(f"  结构: 12 → 26 → 26 → 2")
    
    # 测试Actor前向传播
    print("\n[2/3] 测试Actor前向传播...")
    test_state = torch.randn(1, 12)  # batch_size=1, state_dim=12
    test_action = actor(test_state)
    print(f"✓ 前向传播成功")
    print(f"  输入shape: {test_state.shape}")
    print(f"  输出shape: {test_action.shape}")
    print(f"  输出值: {test_action.detach().numpy()}")
    print(f"  动作范围: [{test_action.min().item():.3f}, {test_action.max().item():.3f}]")
    
    # 打印参数量
    actor_params = sum(p.numel() for p in actor.parameters())
    print(f"  Actor参数量: {actor_params:,}")
    
    # 创建Critic网络
    print("\n[3/3] 创建Critic网络...")
    critic = Critic(state_dim=12, action_dim=2)
    print(f"✓ Critic创建成功")
    print(f"  结构: 14 → 26 → 26 → 1 (双Q网络)")
    
    # 测试Critic前向传播
    test_action_for_critic = torch.randn(1, 2)
    q1, q2 = critic(test_state, test_action_for_critic)
    print(f"✓ 前向传播成功")
    print(f"  Q1值: {q1.item():.3f}")
    print(f"  Q2值: {q2.item():.3f}")
    
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"  Critic参数量: {critic_params:,}")
    
    # 打印所有层的详细信息
    print("\n[详细] Actor网络层：")
    for name, param in actor.named_parameters():
        print(f"  {name:20s} shape: {list(param.shape)}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！")
    print("="*60)
