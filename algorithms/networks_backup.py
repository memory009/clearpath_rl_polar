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
