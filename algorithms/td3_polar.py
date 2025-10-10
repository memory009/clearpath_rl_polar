#!/usr/bin/env python3
"""
TD3 + POLAR 完整实现
包含训练和可达性验证功能
latest
"""

import torch
import numpy as np
import copy
import sympy as sym
import sys
import os

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from algorithms.networks import Actor, Critic
from algorithms.memory import ReplayBuffer
from verification.taylor_model import (
    TaylorModel, TaylorArithmetic, BernsteinPolynomial,
    compute_tm_bounds, apply_activation
)


class TD3Agent:
    """
    TD3智能体 + POLAR可达性验证
    
    结合了：
    1. TD3训练算法
    2. POLAR可达性分析
    """
    
    def __init__(self, state_dim, action_dim, max_action, config):
        """
        初始化TD3智能体
        
        Args:
            state_dim: 状态维度（12）
            action_dim: 动作维度（2）
            max_action: 最大动作值
            config: TD3Config配置对象
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = torch.device(config.device)
        
        # 创建Actor网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # 创建Critic网络（双Q网络）
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # 经验回放
        self.memory = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))
        
        # TD3参数
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.policy_noise = config.policy_noise
        self.noise_clip = config.noise_clip
        self.policy_freq = config.policy_freq
        
        # 训练计数器
        self.total_it = 0
        
        print(f"✓ TD3Agent初始化完成")
        print(f"  Actor: {state_dim} → 26 → 26 → {action_dim}")
        print(f"  Critic: {state_dim+action_dim} → 26 → 26 → 1 (×2)")
    
    def select_action(self, state):
        """
        选择动作（用于训练和测试）
        
        Args:
            state: numpy array [state_dim]
        
        Returns:
            action: numpy array [action_dim]
        """
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action
    
    def store_transition(self, state, action, next_state, reward, done):
        """存储经验到回放缓冲区"""
        self.memory.push(state, action, next_state, reward, done)
    
    def train(self):
        """训练TD3网络（一次更新）"""
        if self.memory.size < self.batch_size:
            return None
        
        self.total_it += 1
        
        # 从经验池采样
        state, action, next_state, reward, not_done = self.memory.sample(self.batch_size)
        
        with torch.no_grad():
            # 为目标动作添加噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # 计算目标Q值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q
        
        # 获取当前Q估计
        current_Q1, current_Q2 = self.critic(state, action)
        
        # 计算Critic损失
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + \
                     torch.nn.functional.mse_loss(current_Q2, target_Q)
        
        # 优化Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = None
        
        # 延迟策略更新
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), 
                                          self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for param, target_param in zip(self.actor.parameters(), 
                                          self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None
        }
    
    def verify_safety(self, state, observation_error=0.01, 
                     bern_order=1, error_steps=4000):
        """
        POLAR可达性验证 - 论文核心算法
        
        Args:
            state: 当前观测状态 [12维]
            observation_error: 观测误差范围
            bern_order: Bernstein多项式阶数
            error_steps: 误差估计采样步数
        
        Returns:
            is_safe: bool, 是否安全
            action_ranges: list of [min, max], 每个动作维度的可达集
        """
        # 1. 提取网络权重
        actor_weights = []
        actor_biases = []
        
        with torch.no_grad():
            for name, param in self.actor.named_parameters():
                if 'weight' in name:
                    actor_weights.append(param.cpu().numpy())
                elif 'bias' in name:
                    actor_biases.append(param.cpu().numpy())
        
        # 2. 创建符号变量
        z_symbols = [sym.Symbol(f'z{i}') for i in range(self.state_dim)]
        
        # 3. 构造输入Taylor模型
        TM_state = []
        for i in range(self.state_dim):
            # 重要：在创建Poly时就指定generators
            poly = sym.Poly(observation_error * z_symbols[i] + state[i], *z_symbols)
            TM_state.append(TaylorModel(poly, [-0.0, 0.0]))
        
        # 4. 逐层传播
        TM_input = TM_state
        TA = TaylorArithmetic()
        BP = BernsteinPolynomial(error_steps=error_steps)
        
        for layer_idx in range(len(actor_biases)):
            TM_temp = []
            weights = actor_weights[layer_idx]
            biases = actor_biases[layer_idx]
            
            for neuron_idx in range(len(biases)):
                # 加权求和
                tm_neuron = TA.weighted_sumforall(
                    TM_input, weights[neuron_idx], biases[neuron_idx]
                )
                
                # 应用激活函数
                if layer_idx < 2:  # ReLU层
                    a, b = compute_tm_bounds(tm_neuron)
                    
                    if a >= 0:
                        # 完全在正区域，直接通过
                        TM_after = tm_neuron
                    elif b <= 0:
                        # 完全在负区域，输出为0
                        # 关键修复：创建零多项式时必须指定generators
                        zero_poly = sym.Poly(0, *z_symbols)
                        TM_after = TaylorModel(zero_poly, [0, 0])
                    else:
                        # 跨越零点，需要用Bernstein多项式近似
                        bern_poly = BP.approximate(a, b, bern_order, 'relu')
                        bern_error = BP.compute_error(a, b, 'relu')
                        TM_after = apply_activation(
                            tm_neuron, bern_poly, bern_error, bern_order
                        )
                else:  # Tanh层（输出层）
                    a, b = compute_tm_bounds(tm_neuron)
                    bern_poly = BP.approximate(a, b, bern_order, 'tanh')
                    bern_error = BP.compute_error(a, b, 'tanh')
                    TM_after = apply_activation(
                        tm_neuron, bern_poly, bern_error, bern_order
                    )
                    # 缩放到动作空间
                    TM_after = TA.constant_product(TM_after, self.max_action)
                
                TM_temp.append(TM_after)
            
            TM_input = TM_temp
        
        # 5. 计算动作可达集
        action_ranges = []
        for tm in TM_input:
            a, b = compute_tm_bounds(tm)
            action_ranges.append([a, b])
        
        # 6. 安全性判断
        is_safe = self._check_action_safety(action_ranges, state)
        
        return is_safe, action_ranges
    
    def _check_action_safety(self, action_ranges, state):
        """
        检查动作可达集是否安全
        
        Args:
            action_ranges: [[min_v, max_v], [min_w, max_w]]
            state: 当前观测状态
        
        Returns:
            is_safe: bool
        """
        # 1. 检查可达集宽度（不确定性）
        for i, (min_val, max_val) in enumerate(action_ranges):
            range_width = max_val - min_val
            if range_width > 1.5:  # 可达集太宽，不确定性太大
                return False
        
        # 2. 检查碰撞风险
        laser_readings = state[2:10]  # 8个激光雷达读数
        min_laser = np.min(laser_readings)
        
        # 如果距离障碍物很近
        if min_laser < 0.05:  # 归一化后的值，对应0.5米
            linear_vel_range = action_ranges[0]
            # 如果可能前进，则不安全
            if linear_vel_range[1] > 0.3:
                return False
        
        # 3. 检查动作范围是否合理
        # 线速度应该在[-0.5, 0.5]范围内
        if action_ranges[0][0] < -0.6 or action_ranges[0][1] > 0.6:
            return False
        
        # 角速度应该在[-1.0, 1.0]范围内
        if action_ranges[1][0] < -1.1 or action_ranges[1][1] > 1.1:
            return False
        
        return True
    
    def save(self, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor.pth")
        torch.save(self.critic.state_dict(), f"{path}/critic.pth")
        torch.save(self.actor_target.state_dict(), f"{path}/actor_target.pth")
        torch.save(self.critic_target.state_dict(), f"{path}/critic_target.pth")
        print(f"✓ 模型已保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        self.actor.load_state_dict(torch.load(f"{path}/actor.pth", 
                                              map_location=self.device))
        self.critic.load_state_dict(torch.load(f"{path}/critic.pth",
                                               map_location=self.device))
        self.actor_target.load_state_dict(torch.load(f"{path}/actor_target.pth",
                                                     map_location=self.device))
        self.critic_target.load_state_dict(torch.load(f"{path}/critic_target.pth",
                                                      map_location=self.device))
        print(f"✓ 模型已加载: {path}")
    
    def save_weights(self, path):
        """保存权重为numpy格式（用于POLAR验证或其他工具）"""
        weights_dict = {}
        
        for name, param in self.actor.named_parameters():
            weights_dict[f'actor_{name}'] = param.cpu().detach().numpy()
        
        for name, param in self.critic.named_parameters():
            weights_dict[f'critic_{name}'] = param.cpu().detach().numpy()
        
        np.savez(path, **weights_dict)
        print(f"✓ 权重已保存到: {path}")