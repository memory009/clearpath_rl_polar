#!/usr/bin/env python3
"""
收集人工演示数据 - 用于GCPO (改进版)
- 自动在Gazebo中显示目标点标记
- 智能的演示收集流程
- 实时反馈和统计
- 修复: 同步目标和重置信号
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

sys.path.append(os.path.dirname(__file__))
from envs.clearpath_nav_env import ClearpathNavEnv


class GoalVisualizer:
    """在Gazebo/RViz中可视化目标点"""
    
    def __init__(self, node):
        self.node = node
        self.marker_pub = node.create_publisher(
            Marker, 
            '/goal_marker', 
            10
        )
        
        print("✓ 目标可视化已启用 - 在RViz中查看 '/goal_marker' topic")
    
    def publish_goal(self, goal_pos, goal_id=0, color='green'):
        """发布目标点标记"""
        marker = Marker()
        marker.header.frame_id = "j100_0000/base_link"
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "goal_markers"
        marker.id = goal_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # 位置
        marker.pose.position.x = float(goal_pos[0])
        marker.pose.position.y = float(goal_pos[1])
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        # 大小
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        
        # 颜色
        if color == 'green':
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
        elif color == 'red':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        elif color == 'blue':
            marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.8)
        elif color == 'yellow':
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        
        marker.lifetime.sec = 0
        
        # 发布
        for _ in range(5):
            self.marker_pub.publish(marker)
            time.sleep(0.05)
        
        print(f"  🏁 目标标记已发布到Gazebo: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")


class KeyboardTeleop:
    """键盘遥控"""
    
    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)
        
        # 控制参数
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
        print("    W : 前进")
        print("    S : 后退")
        print("    A : 左转")
        print("    D : 右转")
        print("    空格 : 停止")
        print()
        print("  【速度调节】")
        print("    Q : 增加速度")
        print("    E : 减少速度")
        print()
        print("  【演示控制】")
        print("    R : 保存当前演示并开始下一个")
        print("    X : 放弃当前演示并重新开始")
        print("    ESC : 退出收集")
        print("="*70)
        print(f"  当前速度: 线速度={self.linear_speed:.2f}m/s, 角速度={self.angular_speed:.2f}rad/s")
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
        """
        根据按键返回动作
        
        Returns:
            action: [linear_vel, angular_vel]
            command: 'record', 'discard', 'quit', or None
        """
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


class DemonstrationCollector:
    """演示数据收集器 - 增强版"""
    
    def __init__(self, env, save_dir='./demonstrations'):
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
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
        self.total_episodes = 0
        self.total_steps = 0
        self.successful_episodes = 0
        
        # 每个目标的统计
        self.goal_stats = {}
        
        print(f"✓ 演示收集器初始化完成")
        print(f"  保存目录: {save_dir}")
    
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
        
        # 初始化目标统计
        goal_key = f"goal_{goal_idx}"
        if goal_key not in self.goal_stats:
            self.goal_stats[goal_key] = {
                'position': goal,
                'attempts': 0,
                'successes': 0,
                'total_steps': 0
            }
        
        self.goal_stats[goal_key]['attempts'] += 1
        
        print(f"\n{'='*70}")
        print(f"🎬 开始新演示")
        print(f"{'='*70}")
        print(f"  目标 #{goal_idx + 1}: ({goal[0]:.2f}, {goal[1]:.2f})")
        print(f"  已收集: {self.goal_stats[goal_key]['successes']} 个成功演示")
        print(f"  按 'R' 保存 | 按 'X' 放弃重来")
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
        
        # 只保存成功的演示
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
        
        # 保存
        goal_key = f"goal_{goal_idx}"
        demo_id = self.goal_stats[goal_key]['successes']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_goal{goal_idx}_num{demo_id:02d}_{timestamp}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        np.savez(filepath, **episode_data)
        
        # 更新统计
        self.total_episodes += 1
        self.successful_episodes += 1
        self.goal_stats[goal_key]['successes'] += 1
        self.goal_stats[goal_key]['total_steps'] += num_steps
        
        print(f"\n{'='*70}")
        print(f"✅ 演示已保存!")
        print(f"{'='*70}")
        print(f"  文件: {filename}")
        print(f"  步数: {num_steps}")
        print(f"  耗时: {episode_time:.1f}秒")
        print(f"  最终距离: {final_distance:.2f}m")
        print(f"  目标#{goal_idx + 1}进度: {self.goal_stats[goal_key]['successes']} 个成功演示")
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
    
    def print_statistics(self):
        """打印统计信息"""
        print(f"\n{'='*70}")
        print("📊 收集统计")
        print(f"{'='*70}")
        print(f"  总演示数: {self.successful_episodes}")
        print(f"  总步数: {self.total_steps}")
        if self.successful_episodes > 0:
            print(f"  平均步数/演示: {self.total_steps / self.successful_episodes:.1f}")
        print()
        print("  各目标统计:")
        for goal_key, stats in sorted(self.goal_stats.items()):
            goal_pos = stats['position']
            attempts = stats['attempts']
            successes = stats['successes']
            success_rate = successes / attempts * 100 if attempts > 0 else 0
            avg_steps = stats['total_steps'] / successes if successes > 0 else 0
            
            print(f"    {goal_key}: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
            print(f"      成功: {successes}/{attempts} ({success_rate:.0f}%)")
            print(f"      平均步数: {avg_steps:.1f}")
        print(f"{'='*70}\n")


def publish_current_goal(goal_pub, goal_pos):
    """发布当前目标位置到监控脚本"""
    goal_msg = Point()
    goal_msg.x = float(goal_pos[0])
    goal_msg.y = float(goal_pos[1])
    goal_msg.z = 0.0
    
    # 多次发送确保收到
    for _ in range(5):
        goal_pub.publish(goal_msg)
        time.sleep(0.02)


def publish_goal_progress(progress_pub, goal_idx, total_goals):
    """发布目标进度"""
    progress_msg = String()
    progress_msg.data = f"{goal_idx}/{total_goals}"
    
    for _ in range(3):
        progress_pub.publish(progress_msg)
        time.sleep(0.02)


def publish_reset_signal(reset_pub):
    """发布重置信号"""
    reset_msg = String()
    reset_msg.data = 'reset'
    
    for _ in range(3):
        reset_pub.publish(reset_msg)
        time.sleep(0.02)


def collect_demonstrations(goal_configs=None, save_dir='./demonstrations'):
    """
    收集演示数据的主函数
    
    Args:
        goal_configs: 目标配置列表 [{'pos': (x,y), 'num_demos': n}, ...]
        save_dir: 保存目录
    """
    print("\n" + "="*70)
    print("🎯 GCPO 演示数据收集系统")
    print("="*70)
    
    # 默认目标配置
    if goal_configs is None:
        goal_configs = [
            {'pos': (1.0, 1.0), 'num_demos': 5},
            {'pos': (2.0, 2.0), 'num_demos': 8},
            {'pos': (3.0, 1.5), 'num_demos': 6},
            {'pos': (4.0, 3.0), 'num_demos': 8},
            {'pos': (5.0, -2.0), 'num_demos': 15},
        ]
    
    print(f"\n📋 演示收集计划:")
    print(f"{'='*70}")
    total_demos = sum(cfg['num_demos'] for cfg in goal_configs)
    for i, cfg in enumerate(goal_configs, 1):
        pos = cfg['pos']
        num = cfg['num_demos']
        print(f"  目标 #{i}: ({pos[0]:5.1f}, {pos[1]:5.1f}) - 需要 {num:2d} 个演示")
    print(f"{'='*70}")
    print(f"  总共需要: {total_demos} 个成功演示")
    print(f"  保存位置: {save_dir}")
    print(f"{'='*70}\n")
    
    input("按 Enter 开始收集...")
    
    # 初始化ROS2
    if not rclpy.ok():
        rclpy.init()
    
    # 初始化
    env = ClearpathNavEnv(goal_pos=goal_configs[0]['pos'])
    teleop = KeyboardTeleop()
    collector = DemonstrationCollector(env, save_dir)
    visualizer = GoalVisualizer(env.node)
    
    # 创建发布器
    reset_pub = env.node.create_publisher(String, '/robot_reset_signal', 10)
    goal_pub = env.node.create_publisher(Point, '/current_goal_position', 10)
    progress_pub = env.node.create_publisher(String, '/goal_progress', 10)
    
    # 等待发布器就绪
    time.sleep(0.5)
    
    # 主循环状态
    current_goal_idx = 0
    demos_collected = 0
    target_demos = goal_configs[0]['num_demos']
    current_goal = goal_configs[0]['pos']
    
    # 发布初始目标标记和位置
    visualizer.publish_goal(current_goal, current_goal_idx, 'green')
    publish_current_goal(goal_pub, current_goal)
    publish_goal_progress(progress_pub, current_goal_idx, len(goal_configs))
    
    # 重置环境
    state, _ = env.reset()
    collector.start_episode(current_goal, current_goal_idx)
    
    # 发送重置信号
    publish_reset_signal(reset_pub)
    
    running = True
    
    while running:
        # 自动切换目标(如果当前目标完成)
        if demos_collected >= target_demos:
            current_goal_idx += 1
            
            if current_goal_idx >= len(goal_configs):
                print("\n" + "="*70)
                print("🎉 所有目标的演示收集完成!")
                print("="*70)
                break
            
            # 切换到下一个目标
            demos_collected = 0
            target_demos = goal_configs[current_goal_idx]['num_demos']
            current_goal = goal_configs[current_goal_idx]['pos']
            
            print(f"\n{'='*70}")
            print(f"🔄 切换到目标 #{current_goal_idx + 1}")
            print(f"{'='*70}")
            print(f"  位置: ({current_goal[0]:.2f}, {current_goal[1]:.2f})")
            print(f"  需要: {target_demos} 个演示")
            print(f"{'='*70}\n")
            
            # 更新环境目标
            env.world_goal = np.array(current_goal, dtype=np.float32)
            
            # 发布新目标标记和位置
            visualizer.publish_goal(current_goal, current_goal_idx, 'green')
            publish_current_goal(goal_pub, current_goal)
            publish_goal_progress(progress_pub, current_goal_idx, len(goal_configs))
            
            # 重置
            state, _ = env.reset()
            collector.start_episode(current_goal, current_goal_idx)
            
            # 发送重置信号
            publish_reset_signal(reset_pub)
        
        # 获取遥控输入
        action, command = teleop.get_action()
        
        # 处理命令
        if command == 'quit':
            print("\n⚠️  用户请求退出")
            break
        
        elif command == 'record':
            # 手动标记成功并保存
            if collector.save_episode(current_goal_idx, success=True):
                demos_collected += 1
                
                # 显示进度
                print(f"进度: {demos_collected}/{target_demos} (目标#{current_goal_idx + 1})")
                
                # 重置开始新演示
                state, _ = env.reset()
                collector.start_episode(current_goal, current_goal_idx)
                
                # 发送重置信号 - 关键修复!
                publish_reset_signal(reset_pub)
            continue
        
        elif command == 'discard':
            # 放弃当前演示
            collector.discard_episode()
            state, _ = env.reset()
            collector.start_episode(current_goal, current_goal_idx)
            
            # 发送重置信号
            publish_reset_signal(reset_pub)
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
        
        # 更新状态
        state = next_state
        
        # 显示实时反馈
        distance = np.linalg.norm(robot_pos - env.goal_relative_to_start)
        print(f"\r🎮 步数:{collector.episode_steps:3d} | "
              f"距离:{distance:5.2f}m | "
              f"动作:[{action[0]:5.2f}, {action[1]:5.2f}] | "
              f"进度:{demos_collected}/{target_demos}", 
              end='', flush=True)
        
        # Episode自动结束
        if done or truncated:
            print()  # 换行
            
            if info.get('goal_reached', False):
                print("🎯 到达目标!")
                # 自动保存成功的演示
                if collector.save_episode(current_goal_idx, success=True):
                    demos_collected += 1
                    print(f"进度: {demos_collected}/{target_demos} (目标#{current_goal_idx + 1})")
            else:
                reason = "碰撞" if info.get('collision') else "超时"
                print(f"❌ Episode失败: {reason}")
                collector.discard_episode()
            
            # 重置
            state, _ = env.reset()
            collector.start_episode(current_goal, current_goal_idx)
            
            # 发送重置信号 - 关键修复!
            publish_reset_signal(reset_pub)
    
    # 最终统计
    collector.print_statistics()
    
    print(f"{'='*70}")
    print(f"✅ 演示数据已保存到: {save_dir}")
    print(f"{'='*70}\n")
    
    env.close()
    
    return save_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='收集GCPO演示数据')
    parser.add_argument('--save_dir', type=str, default='./demonstrations',
                       help='保存目录')
    parser.add_argument('--quick', action='store_true',
                       help='快速测试模式(每个目标只收集2个演示)')
    
    args = parser.parse_args()
    
    # 目标配置
    if args.quick:
        goal_configs = [
            {'pos': (1.0, 1.0), 'num_demos': 2},
            {'pos': (2.0, 2.0), 'num_demos': 2},
        ]
    else:
        goal_configs = [
            {'pos': (1.0, 1.0), 'num_demos': 5},
            {'pos': (2.0, 2.0), 'num_demos': 8},
            {'pos': (3.0, 1.5), 'num_demos': 6},
            {'pos': (4.0, 3.0), 'num_demos': 8},
            {'pos': (5.0, -2.0), 'num_demos': 15},
        ]
    
    try:
        collect_demonstrations(goal_configs, args.save_dir)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()