#!/usr/bin/env python3
"""
GCPO (Goal-Conditioned On-Policy) Agent
基于NeurIPS 2024论文的实现
"""

import torch
import numpy as np
import copy
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from algorithms.td3_polar import TD3Agent
from algorithms.networks import Actor, Critic


class CapabilityEstimator:
    """
    能力估计器 - GCPO核心组件
    估计智能体达到不同目标的能力
    """
    
    def __init__(self, window_size=100):
        """
        Args:
            window_size: 滑动窗口大小
        """
        self.window_size = window_size
        
        # 存储每个目标的成功历史
        # key: (goal_x, goal_y), value: deque of success/failure
        self.goal_history = {}
        
        # 距离区间统计
        self.distance_bins = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 10.0]
        self.distance_stats = {d: deque(maxlen=50) for d in self.distance_bins}
    
    def update(self, goal, success):
        """
        更新目标的成功历史
        
        Args:
            goal: 目标位置 (x, y)
            success: 是否成功 (bool)
        """
        goal_key = self._discretize_goal(goal)
        
        if goal_key not in self.goal_history:
            self.goal_history[goal_key] = deque(maxlen=self.window_size)
        
        self.goal_history[goal_key].append(1 if success else 0)
        
        # 更新距离统计
        distance = np.linalg.norm(goal)
        bin_idx = self._get_distance_bin(distance)
        if bin_idx is not None:
            self.distance_stats[self.distance_bins[bin_idx]].append(1 if success else 0)
    
    def estimate_capability(self, goal):
        """
        估计达到目标的能力 (成功概率)
        
        Args:
            goal: 目标位置 (x, y)
        
        Returns:
            capability: 估计的成功概率 [0, 1]
        """
        goal_key = self._discretize_goal(goal)
        
        # 1. 如果有该目标的历史记录
        if goal_key in self.goal_history and len(self.goal_history[goal_key]) > 5:
            return np.mean(self.goal_history[goal_key])
        
        # 2. 否则使用距离bin的统计
        distance = np.linalg.norm(goal)
        bin_idx = self._get_distance_bin(distance)
        
        if bin_idx is not None:
            bin_key = self.distance_bins[bin_idx]
            if len(self.distance_stats[bin_key]) > 3:
                return np.mean(self.distance_stats[bin_key])
        
        # 3. 默认值：根据距离给一个先验
        # 近距离目标更容易
        if distance < 1.0:
            return 0.7
        elif distance < 2.0:
            return 0.5
        elif distance < 3.0:
            return 0.3
        else:
            return 0.1
    
    def _discretize_goal(self, goal, resolution=0.5):
        """将目标离散化到网格"""
        x = round(goal[0] / resolution) * resolution
        y = round(goal[1] / resolution) * resolution
        return (x, y)
    
    def _get_distance_bin(self, distance):
        """获取距离所在的bin索引"""
        for i, bin_max in enumerate(self.distance_bins):
            if distance < bin_max:
                return i
        return None
    
    def get_statistics(self):
        """获取统计信息"""
        stats = {
            'num_goals_tracked': len(self.goal_history),
            'distance_bin_stats': {}
        }
        
        for bin_max, history in self.distance_stats.items():
            if len(history) > 0:
                stats['distance_bin_stats'][bin_max] = {
                    'samples': len(history),
                    'success_rate': np.mean(history)
                }
        
        return stats


class SelfCurriculum:
    """
    自适应课程生成器 - GCPO核心组件
    根据能力估计自动选择合适难度的目标
    """
    
    def __init__(self, goal_space_bounds, capability_estimator):
        """
        Args:
            goal_space_bounds: 目标空间边界 [(x_min, x_max), (y_min, y_max)]
            capability_estimator: 能力估计器
        """
        self.bounds = goal_space_bounds
        self.capability_estimator = capability_estimator
        
        # 课程参数
        self.target_capability = 0.5  # 目标成功率 (太简单或太难都不好)
        self.capability_tolerance = 0.2  # 容忍范围
        
        # 采样策略
        self.exploration_prob = 0.2  # 探索概率
    
    def sample_goal(self):
        """
        采样一个合适难度的目标
        
        Returns:
            goal: (x, y)
        """
        # 以一定概率进行探索(随机采样)
        if np.random.random() < self.exploration_prob:
            return self._random_goal()
        
        # 否则采样"恰好合适"的目标
        max_attempts = 50
        best_goal = None
        best_score = -float('inf')
        
        for _ in range(max_attempts):
            candidate = self._random_goal()
            capability = self.capability_estimator.estimate_capability(candidate)
            
            # 评分：越接近target_capability越好
            score = -abs(capability - self.target_capability)
            
            if score > best_score:
                best_score = score
                best_goal = candidate
        
        return best_goal
    
    def _random_goal(self):
        """在目标空间中随机采样"""
        x = np.random.uniform(self.bounds[0][0], self.bounds[0][1])
        y = np.random.uniform(self.bounds[1][0], self.bounds[1][1])
        return np.array([x, y], dtype=np.float32)


class GCPOAgent(TD3Agent):
    """
    GCPO智能体
    继承TD3Agent,添加能力估计和自适应课程
    """
    
    def __init__(self, state_dim, action_dim, max_action, config):
        """
        初始化GCPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_action: 最大动作值
            config: 配置对象
        """
        # 调用父类初始化
        super().__init__(state_dim, action_dim, max_action, config)
        
        # GCPO特有组件
        self.capability_estimator = CapabilityEstimator(window_size=100)
        
        # 目标空间定义
        goal_space_bounds = getattr(config, 'goal_space_bounds', 
                                    [(0.5, 5.5), (-3.0, 3.5)])
        
        self.curriculum = SelfCurriculum(
            goal_space_bounds,
            self.capability_estimator
        )
        
        # 训练模式
        self.training_mode = 'bc'  # 'bc' or 'rl'
        
        print(f"✓ GCPO Agent初始化完成")
        print(f"  能力估计: 滑动窗口={self.capability_estimator.window_size}")
        print(f"  目标空间: x={goal_space_bounds[0]}, y={goal_space_bounds[1]}")
    
    def update_capability(self, goal, success):
        """
        更新能力估计
        
        Args:
            goal: 目标位置 (x, y)
            success: 是否成功
        """
        self.capability_estimator.update(goal, success)
    
    def sample_curriculum_goal(self):
        """
        从课程中采样目标
        
        Returns:
            goal: (x, y)
        """
        return self.curriculum.sample_goal()
    
    def switch_to_rl_mode(self):
        """切换到RL训练模式"""
        self.training_mode = 'rl'
        print("\n" + "="*60)
        print("🔄 切换到RL Fine-tuning模式")
        print("="*60)
        print("  BC预训练 → RL自适应课程")
        print("="*60 + "\n")
    
    def get_capability_stats(self):
        """获取能力估计统计"""
        return self.capability_estimator.get_statistics()
    
    def save_gcpo(self, path):
        """
        保存GCPO特有的组件
        
        Args:
            path: 保存路径
        """
        # 保存基础模型
        self.save(path)
        
        # 保存能力估计器历史
        import pickle
        
        capability_data = {
            'goal_history': dict(self.capability_estimator.goal_history),
            'distance_stats': dict(self.capability_estimator.distance_stats)
        }
        
        with open(os.path.join(path, 'capability_estimator.pkl'), 'wb') as f:
            pickle.dump(capability_data, f)
        
        print(f"✓ GCPO组件已保存到: {path}")
    
    def load_gcpo(self, path):
        """
        加载GCPO特有的组件
        
        Args:
            path: 加载路径
        """
        # 加载基础模型
        self.load(path)
        
        # 加载能力估计器
        import pickle
        
        capability_path = os.path.join(path, 'capability_estimator.pkl')
        if os.path.exists(capability_path):
            with open(capability_path, 'rb') as f:
                capability_data = pickle.load(f)
            
            # 转换回deque
            for key, value in capability_data['goal_history'].items():
                self.capability_estimator.goal_history[key] = deque(
                    value, maxlen=self.capability_estimator.window_size
                )
            
            for key, value in capability_data['distance_stats'].items():
                self.capability_estimator.distance_stats[key] = deque(
                    value, maxlen=50
                )
            
            print(f"✓ GCPO组件已加载: {path}")


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("="*60)
    print("GCPO Agent 测试")
    print("="*60)
    
    from utils.config import TD3Config
    
    # 1. 创建GCPO智能体
    print("\n[1/3] 创建GCPO智能体...")
    config = TD3Config()
    agent = GCPOAgent(
        state_dim=12,
        action_dim=2,
        max_action=0.5,
        config=config
    )
    print("✓ 智能体创建成功")
    
    # 2. 测试能力估计
    print("\n[2/3] 测试能力估计...")
    test_goals = [
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 1.5),
        (5.0, -2.0)
    ]
    
    # 模拟一些成功/失败记录
    for goal in test_goals:
        for _ in range(20):
            # 近距离目标成功率高
            distance = np.linalg.norm(goal)
            success = np.random.random() < (1.0 / (1.0 + distance * 0.3))
            agent.update_capability(goal, success)
    
    print("  各目标的能力估计:")
    for goal in test_goals:
        capability = agent.capability_estimator.estimate_capability(goal)
        print(f"    {goal}: {capability:.2%}")
    
    # 3. 测试课程采样
    print("\n[3/3] 测试课程采样...")
    print("  采样10个课程目标:")
    for i in range(10):
        goal = agent.sample_curriculum_goal()
        capability = agent.capability_estimator.estimate_capability(goal)
        print(f"    {i+1}. ({goal[0]:.2f}, {goal[1]:.2f}) - 估计能力: {capability:.2%}")
    
    # 4. 测试保存和加载
    print("\n[4/4] 测试保存和加载...")
    test_path = "./test_gcpo_model"
    agent.save_gcpo(test_path)
    print("✓ 模型已保存")
    
    # 创建新智能体并加载
    agent2 = GCPOAgent(12, 2, 0.5, config)
    agent2.load_gcpo(test_path)
    print("✓ 模型已加载")
    
    # 清理测试文件
    import shutil
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
        print("✓ 测试文件已清理")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)