#!/usr/bin/env python3
"""
收集人工演示数据 - 用于GCPO (断点续传版 - 修复自动保存)
- 自动检测已有数据,从正确位置继续收集
- 自动在Gazebo中显示目标点标记
- 自动保存到达目标的演示
- 智能的演示收集流程
- 实时反馈和统计
"""

import sys
import os
import numpy as np
from datetime import datetime
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, String
from geometry_msgs.msg import Point
import termios
import tty
import select
import time
import glob
import re

sys.path.append(os.path.dirname(__file__))
from envs.clearpath_nav_env import ClearpathNavEnv


class GoalVisualizer:
    """在Gazebo/RViz中可视化目标点"""
    
    def __init__(self, node):
        self.node = node
        self.marker_pub = node.create_publisher(Marker, '/goal_marker', 10)
        print("✓ 目标可视化已启用")
    
    def publish_goal(self, goal_pos, goal_id=0, color='green'):
        """发布目标点标记"""
        marker = Marker()
        marker.header.frame_id = "j100_0000/base_link"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "goal_markers"
        marker.id = goal_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(goal_pos[0])
        marker.pose.position.y = float(goal_pos[1])
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        
        if color == 'green':
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        elif color == 'red':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        elif color == 'blue':
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        elif color == 'yellow':
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        
        marker.lifetime.sec = 0
        
        for _ in range(5):
            self.marker_pub.publish(marker)
            time.sleep(0.05)
        
        print(f"  🏁 目标标记已发布: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")


class KeyboardTeleop:
    """键盘遥控"""
    
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        self.linear_speed = 0.25
        self.angular_speed = 0.7
        self.speed_increment = 0.05
        self.print_instructions()
    
    def print_instructions(self):
        """打印控制说明"""
        print("\n" + "="*70)
        print("🎮 键盘控制说明")
        print("="*70)
        print("  【移动控制】")
        print("    W/S : 前进/后退")
        print("    A/D : 左转/右转")
        print("    空格 : 停止")
        print()
        print("  【速度调节】")
        print("    Q/E : 增加/减少速度")
        print()
        print("  【演示控制】")
        print("    R : 手动保存当前演示并开始下一个")
        print("    X : 放弃当前演示并重新开始")
        print("    ESC : 退出收集")
        print("="*70)
        print(f"  当前速度: 线={self.linear_speed:.2f}m/s, 角={self.angular_speed:.2f}rad/s")
        print(f"  💡 提示: 到达目标0.3m内会自动保存!")
        print("="*70 + "\n")
    
    def get_key(self):
        """获取按键(非阻塞)"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key
    
    def get_action(self):
        """根据按键返回动作"""
        key = self.get_key()
        
        linear = 0.0
        angular = 0.0
        command = None
        
        if key == 'w':
            linear = self.linear_speed
        elif key == 's':
            linear = -self.linear_speed
        elif key == 'a':
            angular = self.angular_speed
        elif key == 'd':
            angular = -self.angular_speed
        elif key == 'q':
            self.linear_speed = min(0.5, self.linear_speed + self.speed_increment)
            self.angular_speed = min(1.0, self.angular_speed + self.speed_increment)
            print(f"⬆️  速度增加: 线={self.linear_speed:.2f}, 角={self.angular_speed:.2f}")
        elif key == 'e':
            self.linear_speed = max(0.1, self.linear_speed - self.speed_increment)
            self.angular_speed = max(0.2, self.angular_speed - self.speed_increment)
            print(f"⬇️  速度减少: 线={self.linear_speed:.2f}, 角={self.angular_speed:.2f}")
        elif key == ' ':
            linear = 0.0
            angular = 0.0
        elif key == 'r':
            command = 'record'
        elif key == 'x':
            command = 'discard'
        elif key == '\x1b':  # ESC
            command = 'quit'
        
        action = np.array([linear, angular], dtype=np.float32)
        return action, command


class DemonstrationCollectorResume:
    """演示数据收集器 - 带断点续传和自动保存"""
    
    def __init__(self, env, save_dir='./demonstrations_v2', goal_configs=None):
        self.env = env
        self.save_dir = save_dir
        self.goal_configs = goal_configs or []
        os.makedirs(save_dir, exist_ok=True)
        
        # 🔥 检测已有数据
        self.existing_demos = self._detect_existing_demos()
        
        # 当前episode数据
        self.current_episode = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'goals': [],
            'robot_positions': []
        }
        
        # 当前episode统计
        self.episode_start_time = None
        self.episode_steps = 0
        
        # 全局统计
        self.total_episodes = sum(self.existing_demos.values())
        self.total_steps = 0
        self.successful_episodes = self.total_episodes
        
        # 每个目标的统计
        self.goal_stats = self._initialize_goal_stats()
        
        print(f"✓ 演示收集器初始化完成 (断点续传模式)")
        print(f"  保存目录: {save_dir}")
        
        # 🔥 显示已有数据统计
        self._print_existing_data_summary()
    
    def _detect_existing_demos(self):
        """检测每个目标已有的演示数量"""
        pattern = os.path.join(self.save_dir, "demo_goal*_num*.npz")
        files = glob.glob(pattern)
        
        existing = {}
        
        for f in files:
            # 匹配文件名: demo_goal{idx}_num{num}_{timestamp}.npz
            match = re.search(r'demo_goal(\d+)_num(\d+)_', os.path.basename(f))
            if match:
                goal_idx = int(match.group(1))
                demo_num = int(match.group(2))
                
                if goal_idx not in existing:
                    existing[goal_idx] = 0
                existing[goal_idx] = max(existing[goal_idx], demo_num + 1)
        
        return existing
    
    def _initialize_goal_stats(self):
        """初始化目标统计,包含已有数据"""
        stats = {}
        
        for idx, cfg in enumerate(self.goal_configs):
            goal_key = f"goal_{idx}"
            stats[goal_key] = {
                'position': cfg['pos'],
                'target': cfg['num_demos'],
                'existing': self.existing_demos.get(idx, 0),
                'collected_now': 0,
                'attempts': 0,
                'total_steps': 0
            }
        
        return stats
    
    def _print_existing_data_summary(self):
        """打印已有数据摘要"""
        if not self.existing_demos:
            print("\n  📂 未发现已有演示数据,将从头开始收集")
            return
        
        print(f"\n{'='*70}")
        print("📂 检测到已有演示数据:")
        print(f"{'='*70}")
        
        for idx, cfg in enumerate(self.goal_configs):
            existing = self.existing_demos.get(idx, 0)
            target = cfg['num_demos']
            remaining = max(0, target - existing)
            
            status = "✅ 已完成" if existing >= target else f"⏳ 还需{remaining}个"
            
            print(f"  Goal {idx}: ({cfg['pos'][0]:.1f}, {cfg['pos'][1]:.1f})")
            print(f"    已有: {existing}/{target}  {status}")
        
        total_existing = sum(self.existing_demos.values())
        total_target = sum(cfg['num_demos'] for cfg in self.goal_configs)
        total_remaining = total_target - total_existing
        
        print(f"\n  总计: {total_existing}/{total_target} 个演示")
        print(f"  剩余: {total_remaining} 个演示需要收集")
        print(f"{'='*70}\n")
    
    def start_episode(self, goal, goal_idx):
        """开始新的episode"""
        self.current_episode = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'goals': [],
            'robot_positions': []
        }
        
        self.episode_start_time = time.time()
        self.episode_steps = 0
        
        goal_key = f"goal_{goal_idx}"
        self.goal_stats[goal_key]['attempts'] += 1
        
        # 🔥 显示当前进度(包含已有数据)
        existing = self.goal_stats[goal_key]['existing']
        collected = self.goal_stats[goal_key]['collected_now']
        target = self.goal_stats[goal_key]['target']
        total_for_this_goal = existing + collected
        
        print(f"\n{'='*70}")
        print(f"🎬 开始新演示 - Goal {goal_idx}")
        print(f"{'='*70}")
        print(f"  目标位置: ({goal[0]:.2f}, {goal[1]:.2f})")
        print(f"  已有演示: {existing} 个 (之前收集)")
        print(f"  本次已收集: {collected} 个")
        print(f"  总进度: {total_for_this_goal}/{target}")
        print(f"  还需: {max(0, target - total_for_this_goal)} 个")
        print(f"  💡 到达目标<0.3m会自动保存 | 按'R'手动保存 | 按'X'放弃重来")
        print(f"{'='*70}\n")
    
    def add_step(self, state, action, next_state, reward, done, goal, robot_pos):
        """添加一步数据"""
        self.current_episode['states'].append(state)
        self.current_episode['actions'].append(action)
        self.current_episode['next_states'].append(next_state)
        self.current_episode['rewards'].append(reward)
        self.current_episode['dones'].append(done)
        self.current_episode['goals'].append(goal)
        self.current_episode['robot_positions'].append(robot_pos)
        
        self.episode_steps += 1
        self.total_steps += 1
    
    def save_episode(self, goal_idx, success=False):
        """保存当前episode"""
        if len(self.current_episode['states']) == 0:
            print("⚠️  当前episode无数据,不保存")
            return False
        
        if not success:
            print("❌ Episode未成功,不保存")
            return False
        
        # 转换为numpy数组
        episode_data = {
            'states': np.array(self.current_episode['states']),
            'actions': np.array(self.current_episode['actions']),
            'next_states': np.array(self.current_episode['next_states']),
            'rewards': np.array(self.current_episode['rewards']),
            'dones': np.array(self.current_episode['dones']),
            'goals': np.array(self.current_episode['goals']),
            'robot_positions': np.array(self.current_episode['robot_positions']),
            'goal_idx': goal_idx
        }
        
        # 统计
        num_steps = len(episode_data['states'])
        final_distance = np.linalg.norm(
            episode_data['robot_positions'][-1] - episode_data['goals'][-1]
        )
        episode_time = time.time() - self.episode_start_time
        
        # 🔥 保存文件:编号 = 已有数量 + 本次已收集数量
        goal_key = f"goal_{goal_idx}"
        existing = self.goal_stats[goal_key]['existing']
        collected = self.goal_stats[goal_key]['collected_now']
        demo_id = existing + collected  # 续接编号
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_goal{goal_idx}_num{demo_id:02d}_{timestamp}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        np.savez(filepath, **episode_data)
        
        # 更新统计
        self.total_episodes += 1
        self.successful_episodes += 1
        self.goal_stats[goal_key]['collected_now'] += 1
        self.goal_stats[goal_key]['total_steps'] += num_steps
        
        # 🔥 计算总进度
        total_for_this_goal = existing + self.goal_stats[goal_key]['collected_now']
        target = self.goal_stats[goal_key]['target']
        
        print(f"\n{'='*70}")
        print(f"✅ 演示已保存!")
        print(f"{'='*70}")
        print(f"  文件: {filename}")
        print(f"  步数: {num_steps}")
        print(f"  耗时: {episode_time:.1f}秒")
        print(f"  最终距离: {final_distance:.2f}m")
        print(f"\n  Goal {goal_idx} 进度:")
        print(f"    之前已有: {existing}")
        print(f"    本次收集: {self.goal_stats[goal_key]['collected_now']}")
        print(f"    当前总计: {total_for_this_goal}/{target}")
        print(f"    还需收集: {max(0, target - total_for_this_goal)}")
        print(f"{'='*70}\n")
        
        return True
    
    def discard_episode(self):
        """放弃当前episode"""
        steps = len(self.current_episode['states'])
        print(f"\n❌ 已放弃当前演示 ({steps}步)")
        self.current_episode = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
            'goals': [],
            'robot_positions': []
        }
        self.episode_steps = 0
    
    def get_remaining_for_goal(self, goal_idx):
        """获取某个目标还需要收集的数量"""
        goal_key = f"goal_{goal_idx}"
        if goal_key not in self.goal_stats:
            return 0
        
        stats = self.goal_stats[goal_key]
        total_collected = stats['existing'] + stats['collected_now']
        remaining = max(0, stats['target'] - total_collected)
        return remaining
    
    def print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print("📊 收集统计 (包含已有数据)")
        print(f"{'='*70}")
        
        total_existing = sum(s['existing'] for s in self.goal_stats.values())
        total_collected_now = sum(s['collected_now'] for s in self.goal_stats.values())
        total_all = total_existing + total_collected_now
        total_target = sum(s['target'] for s in self.goal_stats.values())
        
        print(f"  总演示数: {total_all}/{total_target}")
        print(f"    之前已有: {total_existing}")
        print(f"    本次收集: {total_collected_now}")
        print(f"    还需收集: {max(0, total_target - total_all)}")
        
        if total_collected_now > 0:
            print(f"\n  本次收集总步数: {self.total_steps}")
            print(f"  本次平均步数/演示: {self.total_steps / total_collected_now:.1f}")
        
        print(f"\n  各目标详细统计:")
        for goal_key, stats in sorted(self.goal_stats.items()):
            goal_pos = stats['position']
            existing = stats['existing']
            collected = stats['collected_now']
            target = stats['target']
            total = existing + collected
            remaining = max(0, target - total)
            
            status = "✅" if total >= target else "⏳"
            
            print(f"\n    {goal_key}: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f}) {status}")
            print(f"      目标数量: {target}")
            print(f"      已有: {existing}")
            print(f"      本次收集: {collected}")
            print(f"      当前总计: {total}/{target}")
            print(f"      还需: {remaining}")
            
            if collected > 0:
                attempts = stats['attempts']
                success_rate = collected / attempts * 100 if attempts > 0 else 0
                avg_steps = stats['total_steps'] / collected
                print(f"      本次成功率: {success_rate:.0f}% ({collected}/{attempts})")
                print(f"      平均步数: {avg_steps:.1f}")
        
        print(f"{'='*70}\n")


def publish_signal(publisher, msg_type, data, repeat=3):
    """通用发布函数"""
    msg = msg_type()
    if msg_type == String:
        msg.data = data
    elif msg_type == Point:
        msg.x, msg.y, msg.z = data[0], data[1], 0.0
    
    for _ in range(repeat):
        publisher.publish(msg)
        time.sleep(0.02)


def collect_demonstrations_resume(goal_configs=None, save_dir='./demonstrations_v2'):
    """
    收集演示数据的主函数 - 支持断点续传和自动保存
    
    Args:
        goal_configs: 目标配置列表 [{'pos': (x,y), 'num_demos': n}, ...]
        save_dir: 保存目录
    """
    print("\n" + "="*70)
    print("🎯 GCPO 演示数据收集系统 (断点续传 + 自动保存)")
    print("="*70)
    
    # 默认目标配置
    if goal_configs is None:
        goal_configs = [
            {'pos': (-2.0, -2.0), 'num_demos': 5},
            {'pos': (2.0, 2.0), 'num_demos': 5},
            {'pos': (-4.0, 1.5), 'num_demos': 10},
            {'pos': (-4.0, -2.0), 'num_demos': 10},
            {'pos': (5.0, -0.5), 'num_demos': 20},
        ]
    
    print(f"\n📋 演示收集计划:")
    print(f"{'='*70}")
    total_demos = sum(cfg['num_demos'] for cfg in goal_configs)
    for i, cfg in enumerate(goal_configs):
        pos = cfg['pos']
        num = cfg['num_demos']
        print(f"  Goal {i}: ({pos[0]:5.1f}, {pos[1]:5.1f}) - 目标 {num:2d} 个演示")
    print(f"{'='*70}")
    print(f"  总目标: {total_demos} 个成功演示")
    print(f"  保存位置: {save_dir}")
    print(f"  💡 特性: 到达目标<0.3m自动保存!")
    print(f"{'='*70}\n")
    
    # 初始化ROS2
    if not rclpy.ok():
        rclpy.init()
    
    # 初始化环境和收集器
    env = ClearpathNavEnv(goal_pos=goal_configs[0]['pos'])
    teleop = KeyboardTeleop()
    collector = DemonstrationCollectorResume(env, save_dir, goal_configs)
    visualizer = GoalVisualizer(env.node)
    
    # 创建发布器
    reset_pub = env.node.create_publisher(String, '/robot_reset_signal', 10)
    goal_pub = env.node.create_publisher(Point, '/current_goal_position', 10)
    progress_pub = env.node.create_publisher(String, '/goal_progress', 10)
    
    time.sleep(0.5)
    
    # 🔥 找到第一个未完成的目标
    current_goal_idx = 0
    for idx in range(len(goal_configs)):
        remaining = collector.get_remaining_for_goal(idx)
        if remaining > 0:
            current_goal_idx = idx
            break
    
    if collector.get_remaining_for_goal(current_goal_idx) == 0:
        print("\n" + "="*70)
        print("🎉 所有目标的演示都已收集完成!")
        print("="*70)
        collector.print_statistics()
        env.close()
        return save_dir
    
    current_goal = goal_configs[current_goal_idx]['pos']
    
    # 🔥 关键修复:更新环境目标位置!
    env.world_goal = np.array(current_goal, dtype=np.float32)
    
    print(f"\n🔄 从 Goal {current_goal_idx} 继续收集...")
    print(f"  位置: ({current_goal[0]:.2f}, {current_goal[1]:.2f})")
    print(f"  还需: {collector.get_remaining_for_goal(current_goal_idx)} 个演示\n")
    
    input("按 Enter 开始收集...")
    
    # 发布初始目标
    visualizer.publish_goal(current_goal, current_goal_idx, 'green')
    publish_signal(goal_pub, Point, current_goal)
    publish_signal(progress_pub, String, f"{current_goal_idx}/{len(goal_configs)}")
    
    # 重置环境
    state, _ = env.reset()
    collector.start_episode(current_goal, current_goal_idx)
    publish_signal(reset_pub, String, 'reset')
    
    running = True
    
    try:
        while running:
            # 🔥 检查当前目标是否完成,自动切换
            while collector.get_remaining_for_goal(current_goal_idx) == 0:
                current_goal_idx += 1
                
                if current_goal_idx >= len(goal_configs):
                    print("\n" + "="*70)
                    print("🎉 所有目标的演示收集完成!")
                    print("="*70)
                    running = False
                    break
                
                # 切换到下一个未完成的目标
                if collector.get_remaining_for_goal(current_goal_idx) > 0:
                    current_goal = goal_configs[current_goal_idx]['pos']
                    
                    print(f"\n{'='*70}")
                    print(f"🔄 切换到 Goal {current_goal_idx}")
                    print(f"{'='*70}")
                    print(f"  位置: ({current_goal[0]:.2f}, {current_goal[1]:.2f})")
                    print(f"  还需: {collector.get_remaining_for_goal(current_goal_idx)} 个演示")
                    print(f"{'='*70}\n")
                    
                    # 更新环境目标
                    env.world_goal = np.array(current_goal, dtype=np.float32)
                    
                    # 发布新目标
                    visualizer.publish_goal(current_goal, current_goal_idx, 'green')
                    publish_signal(goal_pub, Point, current_goal)
                    publish_signal(progress_pub, String, f"{current_goal_idx}/{len(goal_configs)}")
                    
                    # 重置
                    state, _ = env.reset()
                    collector.start_episode(current_goal, current_goal_idx)
                    publish_signal(reset_pub, String, 'reset')
                    break
            
            if not running:
                break
            
            # 获取遥控输入
            action, command = teleop.get_action()
            
            # 处理命令
            if command == 'quit':
                print("\n⚠️  用户请求退出")
                break
            
            elif command == 'record':
                # 手动保存
                if collector.save_episode(current_goal_idx, success=True):
                    remaining = collector.get_remaining_for_goal(current_goal_idx)
                    print(f"剩余: {remaining} 个 (Goal {current_goal_idx})")
                    
                    # 重置开始新演示
                    state, _ = env.reset()
                    collector.start_episode(current_goal, current_goal_idx)
                    publish_signal(reset_pub, String, 'reset')
                continue
            
            elif command == 'discard':
                collector.discard_episode()
                state, _ = env.reset()
                collector.start_episode(current_goal, current_goal_idx)
                publish_signal(reset_pub, String, 'reset')
                continue
            
            # 执行动作
            next_state, reward, done, truncated, info = env.step(action)
            
            # 获取机器人位置
            odom = env.reset_manager.get_relative_odom()
            robot_pos = np.array([odom['x'], odom['y']], dtype=np.float32)
            
            # 记录数据
            collector.add_step(
                state, action, next_state, reward, done or truncated,
                env.goal_relative_to_start, robot_pos
            )
            
            state = next_state
            
            # 显示实时反馈
            distance = np.linalg.norm(robot_pos - env.goal_relative_to_start)
            remaining = collector.get_remaining_for_goal(current_goal_idx)
            print(f"\r🎮 步数:{collector.episode_steps:3d} | "
                  f"距离:{distance:5.2f}m | "
                  f"动作:[{action[0]:5.2f}, {action[1]:5.2f}] | "
                  f"剩余:{remaining}", 
                  end='', flush=True)
            
            # 🔥 Episode自动结束 - 关键修复:手动检测是否到达目标
            if done or truncated:
                print()
                
                # 🔥 手动检测是否到达目标(不依赖info)
                distance = np.linalg.norm(robot_pos - env.goal_relative_to_start)
                goal_reached = distance < 0.3  # 与环境中的判断标准一致
                
                if goal_reached:
                    print(f"🎯 到达目标! (距离: {distance:.2f}m)")
                    # 🔥 自动保存成功的演示
                    if collector.save_episode(current_goal_idx, success=True):
                        remaining = collector.get_remaining_for_goal(current_goal_idx)
                        print(f"剩余: {remaining} 个 (Goal {current_goal_idx})")
                else:
                    reason = "碰撞" if info.get('collision') else f"超时(距离{distance:.2f}m)"
                    print(f"❌ Episode失败: {reason}")
                    collector.discard_episode()
                
                # 重置
                state, _ = env.reset()
                collector.start_episode(current_goal, current_goal_idx)
                publish_signal(reset_pub, String, 'reset')
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    finally:
        # 最终统计
        collector.print_statistics()
        
        print(f"{'='*70}")
        print(f"✅ 演示数据已保存到: {save_dir}")
        print(f"{'='*70}\n")
        
        env.close()
    
    return save_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='收集GCPO演示数据 (断点续传 + 自动保存)')
    parser.add_argument('--save_dir', type=str, default='./demonstrations_v2',
                       help='保存目录 (默认: ./demonstrations_v2)')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式')
    
    args = parser.parse_args()
    
    # 目标配置
    if args.quick:
        goal_configs = [
            {'pos': (1.0, 1.0), 'num_demos': 2},
            {'pos': (2.0, 2.0), 'num_demos': 2},
        ]
    else:
        # 默认配置 - 与原版保持一致
        goal_configs = [
            {'pos': (-2.0, -2.0), 'num_demos': 5},
            {'pos': (2.0, 2.0), 'num_demos': 5},
            {'pos': (-4.0, 1.5), 'num_demos': 10},
            {'pos': (-4.0, -2.0), 'num_demos': 10},
            {'pos': (5.0, -0.5), 'num_demos': 20},
        ]
    
    try:
        collect_demonstrations_resume(goal_configs, args.save_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()