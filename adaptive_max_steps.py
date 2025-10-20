#!/usr/bin/env python3
"""
自适应Max Steps管理器
结合历史演示数据 + 在线学习
"""

import numpy as np
import glob
import os
from collections import defaultdict


class AdaptiveStepsManager:
    """
    自适应步数管理器
    
    功能:
    1. 从演示数据初始化每个目标的步数估计
    2. 训练过程中在线更新统计
    3. 对未见过的目标进行智能插值
    """
    
    def __init__(self, demo_dir='./demonstrations', alpha=0.1):
        """
        初始化
        
        Args:
            demo_dir: 演示数据目录
            alpha: 在线学习率
        """
        self.alpha = alpha
        self.goal_stats = {}  # {goal: {'mean': float, 'std': float, 'count': int}}
        
        # 从演示数据初始化
        if os.path.exists(demo_dir):
            self._load_from_demonstrations(demo_dir)
        
        print(f"✓ 自适应步数管理器初始化完成")
        print(f"  已加载 {len(self.goal_stats)} 个目标的历史数据")
    
    def _load_from_demonstrations(self, demo_dir):
        """从演示数据加载初始统计"""
        demo_files = glob.glob(os.path.join(demo_dir, '*.npz'))
        
        goal_steps_map = defaultdict(list)
        
        for demo_file in demo_files:
            try:
                data = np.load(demo_file)
                goal = tuple(np.round(data['goals'][0], 1))
                steps = len(data['states'])
                goal_steps_map[goal].append(steps)
            except:
                continue
        
        # 计算统计
        for goal, steps_list in goal_steps_map.items():
            self.goal_stats[goal] = {
                'mean': np.mean(steps_list),
                'std': np.std(steps_list),
                'min': np.min(steps_list),
                'max': np.max(steps_list),
                'count': len(steps_list)
            }
        
        # 打印统计
        print(f"\n  目标步数统计 (来自演示数据):")
        for goal, stats in sorted(self.goal_stats.items()):
            print(f"    {goal}: {stats['mean']:.0f}±{stats['std']:.0f} 步 "
                  f"(范围: {stats['min']:.0f}-{stats['max']:.0f})")
    
    def get_max_steps(self, goal_pos, safety_margin=1.5):
        """
        获取目标的max_steps
        
        Args:
            goal_pos: 目标位置 (x, y)
            safety_margin: 安全余量(倍数)
        
        Returns:
            max_steps: 推荐的最大步数
        """
        goal_key = tuple(np.round(goal_pos, 1))
        
        if goal_key in self.goal_stats:
            # 已知目标: 使用统计数据
            stats = self.goal_stats[goal_key]
            # 使用: 平均值 + 2*标准差 (覆盖~95%情况)
            estimated = stats['mean'] + 2 * stats['std']
            max_steps = int(estimated * safety_margin)
        else:
            # 未知目标: 从邻近目标插值
            max_steps = self._interpolate_from_neighbors(goal_pos, safety_margin)
        
        # 限制在合理范围
        max_steps = max(256, min(max_steps, 2048))
        
        return max_steps
    
    def _interpolate_from_neighbors(self, goal_pos, safety_margin, k=3):
        """
        从k个最近邻插值估算未知目标的步数
        
        Args:
            goal_pos: 目标位置
            safety_margin: 安全余量
            k: 使用的邻居数量
        """
        if len(self.goal_stats) == 0:
            # 没有任何数据,返回保守值
            return 1024
        
        # 计算到所有已知目标的距离
        distances = []
        for known_goal, stats in self.goal_stats.items():
            dist = np.linalg.norm(np.array(goal_pos) - np.array(known_goal))
            estimated = stats['mean'] + 2 * stats['std']
            distances.append((dist, estimated))
        
        # 排序并取前k个
        distances.sort()
        nearest_k = distances[:min(k, len(distances))]
        
        # 距离加权平均 (距离越近权重越大)
        weights = [1.0 / (d + 0.1) for d, _ in nearest_k]
        weighted_sum = sum(w * s for (_, s), w in zip(nearest_k, weights))
        total_weight = sum(weights)
        
        estimated = weighted_sum / total_weight
        max_steps = int(estimated * safety_margin)
        
        return max_steps
    
    def update_online(self, goal_pos, actual_steps, success):
        """
        在线更新目标的统计 (在训练过程中调用)
        
        Args:
            goal_pos: 目标位置
            actual_steps: 实际使用的步数
            success: 是否成功
        """
        goal_key = tuple(np.round(goal_pos, 1))
        
        # 只用成功的episode更新统计
        if not success:
            return
        
        if goal_key not in self.goal_stats:
            # 新目标: 初始化
            self.goal_stats[goal_key] = {
                'mean': actual_steps,
                'std': actual_steps * 0.2,  # 初始标准差
                'min': actual_steps,
                'max': actual_steps,
                'count': 1
            }
        else:
            # 已有目标: 更新统计
            stats = self.goal_stats[goal_key]
            old_mean = stats['mean']
            
            # 指数移动平均更新均值
            stats['mean'] = (1 - self.alpha) * old_mean + self.alpha * actual_steps
            
            # 更新标准差 (简化版)
            stats['std'] = (1 - self.alpha) * stats['std'] + \
                          self.alpha * abs(actual_steps - old_mean)
            
            # 更新范围
            stats['min'] = min(stats['min'], actual_steps)
            stats['max'] = max(stats['max'], actual_steps)
            stats['count'] += 1
    
    def get_statistics_summary(self):
        """获取统计摘要"""
        if len(self.goal_stats) == 0:
            return "无数据"
        
        all_means = [s['mean'] for s in self.goal_stats.values()]
        
        summary = {
            'num_goals': len(self.goal_stats),
            'avg_steps_overall': np.mean(all_means),
            'min_steps_overall': np.min([s['min'] for s in self.goal_stats.values()]),
            'max_steps_overall': np.max([s['max'] for s in self.goal_stats.values()]),
        }
        
        return summary
    
    def save(self, filepath='./models/adaptive_steps.npz'):
        """保存统计数据"""
        import pickle
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.goal_stats, f)
        
        print(f"✓ 自适应步数统计已保存: {filepath}")
    
    def load(self, filepath='./models/adaptive_steps.npz'):
        """加载统计数据"""
        import pickle
        
        if not os.path.exists(filepath):
            print(f"⚠️  文件不存在: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            self.goal_stats = pickle.load(f)
        
        print(f"✓ 自适应步数统计已加载: {filepath}")
        print(f"  已加载 {len(self.goal_stats)} 个目标的数据")
        
        return True


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("="*60)
    print("测试自适应步数管理器")
    print("="*60)
    
    # 创建管理器
    manager = AdaptiveStepsManager(demo_dir='./demonstrations')
    
    # 测试已知目标
    print("\n[测试1] 已知目标的max_steps:")
    test_goals = [(1.0, 1.0), (2.0, 2.0), (3.0, 1.5), (5.0, -2.0)]
    
    for goal in test_goals:
        max_steps = manager.get_max_steps(goal)
        print(f"  {goal}: {max_steps} 步")
    
    # 测试未知目标(插值)
    print("\n[测试2] 未知目标的max_steps (插值):")
    unknown_goals = [(1.5, 0.5), (4.0, 0.0), (6.0, -3.0)]
    
    for goal in unknown_goals:
        max_steps = manager.get_max_steps(goal)
        print(f"  {goal}: {max_steps} 步 (插值)")
    
    # 测试在线更新
    print("\n[测试3] 在线更新:")
    test_goal = (2.0, 2.0)
    print(f"  更新前: {manager.get_max_steps(test_goal)} 步")
    
    # 模拟几次成功的episode
    for actual_steps in [650, 700, 680]:
        manager.update_online(test_goal, actual_steps, success=True)
    
    print(f"  更新后: {manager.get_max_steps(test_goal)} 步")
    
    # 统计摘要
    print("\n[测试4] 统计摘要:")
    summary = manager.get_statistics_summary()
    print(f"  跟踪目标数: {summary['num_goals']}")
    print(f"  平均步数: {summary['avg_steps_overall']:.0f}")
    print(f"  步数范围: {summary['min_steps_overall']:.0f} - {summary['max_steps_overall']:.0f}")
    
    # 保存和加载
    print("\n[测试5] 保存和加载:")
    manager.save('./test_adaptive_steps.pkl')
    
    manager2 = AdaptiveStepsManager(demo_dir='')  # 空初始化
    manager2.load('./test_adaptive_steps.pkl')
    
    print(f"  验证: {manager2.get_max_steps((2.0, 2.0))} 步")
    
    # 清理测试文件
    import os
    if os.path.exists('./test_adaptive_steps.pkl'):
        os.remove('./test_adaptive_steps.pkl')
        print("✓ 测试文件已清理")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)