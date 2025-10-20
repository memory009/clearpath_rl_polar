#!/usr/bin/env python3
"""
收集 Goal5 的演示数据
目标位置: (4.5, -2.0)
保存目录: ./demonstrations_goal5
自动检测已有数据，从正确的编号继续收集
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
        elif color == 'yellow':
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)
        elif color == 'blue':
            marker.color = ColorRGBA(r=0.0, g=0.5, b=1.0, a=0.8)
        
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
        print("    R : 保存当前演示并开始下一个")
        print("    X : 放弃当前演示并重新开始")
        print("    ESC : 退出收集")
        print("="*70)
        print(f"  当前速度: 线={self.linear_speed:.2f}m/s, 角={self.angular_speed:.2f}rad/s")
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


class Goal5Collector:
    """Goal5 数据收集器"""
    
    def __init__(self, env, save_dir='./demonstrations_goal5', goal_idx=5):
        self.env = env
        self.save_dir = save_dir
        self.goal_idx = goal_idx
        os.makedirs(save_dir, exist_ok=True)
        
        # 检测已有的演示数量
        self.starting_demo_num = self._detect_existing_demos()
        self.current_demo_num = self.starting_demo_num
        
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
        
        # 统计
        self.episode_start_time = None
        self.episode_steps = 0
        self.collected_count = 0
        
        print(f"✓ Goal5 收集器初始化完成")
        print(f"  保存目录: {save_dir}")
        print(f"  已有演示: {self.starting_demo_num} 个")
        print(f"  将从 num{self.starting_demo_num:02d} 开始继续收集")
    
    def _detect_existing_demos(self):
        """检测已有的 goal5 演示数量"""
        pattern = os.path.join(self.save_dir, f"demo_goal{self.goal_idx}_num*.npz")
        files = glob.glob(pattern)
        
        if not files:
            print(f"  未找到 goal{self.goal_idx} 的已有演示，将从 num00 开始")
            return 0
        
        # 提取所有编号
        numbers = []
        for f in files:
            match = re.search(r'num(\d+)_', os.path.basename(f))
            if match:
                numbers.append(int(match.group(1)))
        
        if numbers:
            max_num = max(numbers)
            print(f"  找到 {len(numbers)} 个 goal{self.goal_idx} 演示，最大编号: num{max_num:02d}")
            return max_num + 1
        
        return 0
    
    def start_episode(self, goal):
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
        
        print(f"\n{'='*70}")
        print(f"🎬 开始新演示 - Goal5 #{self.current_demo_num:02d}")
        print(f"{'='*70}")
        print(f"  目标位置: ({goal[0]:.2f}, {goal[1]:.2f})")
        print(f"  已收集(本次): {self.collected_count} 个")
        print(f"  总计: {self.starting_demo_num + self.collected_count} 个")
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
    
    def save_episode(self, success=False):
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
            'goal_idx': self.goal_idx
        }
        
        # 统计
        num_steps = len(episode_data['states'])
        final_distance = np.linalg.norm(
            episode_data['robot_positions'][-1] - episode_data['goals'][-1]
        )
        episode_time = time.time() - self.episode_start_time
        
        # 保存文件名格式: demo_goal5_num00_timestamp.npz
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"demo_goal{self.goal_idx}_num{self.current_demo_num:02d}_{timestamp}.npz"
        filepath = os.path.join(self.save_dir, filename)
        
        np.savez(filepath, **episode_data)
        
        # 更新统计
        self.collected_count += 1
        self.current_demo_num += 1
        
        print(f"\n{'='*70}")
        print(f"✅ 演示已保存!")
        print(f"{'='*70}")
        print(f"  文件: {filename}")
        print(f"  步数: {num_steps}")
        print(f"  耗时: {episode_time:.1f}秒")
        print(f"  最终距离: {final_distance:.2f}m")
        print(f"  本次收集: {self.collected_count} 个")
        print(f"  总计: {self.starting_demo_num + self.collected_count} 个")
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
    
    def print_summary(self):
        """打印收集总结"""
        print(f"\n{'='*70}")
        print("📊 收集总结 - Goal5")
        print(f"{'='*70}")
        print(f"  原有演示: {self.starting_demo_num} 个")
        print(f"  本次收集: {self.collected_count} 个")
        print(f"  当前总计: {self.starting_demo_num + self.collected_count} 个")
        if self.collected_count > 0:
            print(f"  编号范围: num{self.starting_demo_num:02d} - num{self.current_demo_num-1:02d}")
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


def collect_goal5_demos(goal_pos=(4.5, -2.0), target_count=None, save_dir='./demonstrations_goal5'):
    """
    收集 Goal5 的演示数据
    
    Args:
        goal_pos: Goal5 的位置坐标 (4.5, -2.0)
        target_count: 目标收集数量（None表示无限制）
        save_dir: 保存目录 (demonstrations_goal5)
    """
    print("\n" + "="*70)
    print("🎯 Goal5 演示数据收集系统")
    print("="*70)
    print(f"\n  目标编号: Goal5")
    print(f"  目标位置: ({goal_pos[0]:.2f}, {goal_pos[1]:.2f})")
    print(f"  保存目录: {save_dir}")
    if target_count:
        print(f"  目标数量: {target_count} 个")
    else:
        print(f"  目标数量: 无限制 (按ESC退出)")
    print(f"\n{'='*70}\n")
    
    input("按 Enter 开始收集...")
    
    # 初始化ROS2
    if not rclpy.ok():
        rclpy.init()
    
    # 初始化环境和组件
    env = ClearpathNavEnv(goal_pos=goal_pos)
    teleop = KeyboardTeleop()
    collector = Goal5Collector(env, save_dir, goal_idx=5)
    visualizer = GoalVisualizer(env.node)
    
    # 创建发布器
    reset_pub = env.node.create_publisher(String, '/robot_reset_signal', 10)
    goal_pub = env.node.create_publisher(Point, '/current_goal_position', 10)
    progress_pub = env.node.create_publisher(String, '/goal_progress', 10)
    
    time.sleep(0.5)
    
    # 发布目标信息
    visualizer.publish_goal(goal_pos, goal_id=5, color='blue')
    publish_signal(goal_pub, Point, goal_pos)
    publish_signal(progress_pub, String, "5/5")
    
    # 重置环境
    state, _ = env.reset()
    collector.start_episode(goal_pos)
    publish_signal(reset_pub, String, 'reset')
    
    running = True
    
    try:
        while running:
            # 检查是否达到目标数量
            if target_count and collector.collected_count >= target_count:
                print(f"\n{'='*70}")
                print(f"🎉 已完成目标数量! 收集了 {collector.collected_count} 个演示")
                print(f"{'='*70}")
                break
            
            # 获取遥控输入
            action, command = teleop.get_action()
            
            # 处理命令
            if command == 'quit':
                print("\n⚠️  用户请求退出")
                break
            
            elif command == 'record':
                if collector.save_episode(success=True):
                    state, _ = env.reset()
                    collector.start_episode(goal_pos)
                    publish_signal(reset_pub, String, 'reset')
                continue
            
            elif command == 'discard':
                collector.discard_episode()
                state, _ = env.reset()
                collector.start_episode(goal_pos)
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
            
            # 实时反馈
            distance = np.linalg.norm(robot_pos - env.goal_relative_to_start)
            print(f"\r🎮 步数:{collector.episode_steps:3d} | "
                  f"距离:{distance:5.2f}m | "
                  f"动作:[{action[0]:5.2f}, {action[1]:5.2f}] | "
                  f"已收集:{collector.collected_count}", 
                  end='', flush=True)
            
            # Episode结束
            if done or truncated:
                print()
                
                if info.get('goal_reached', False):
                    print("🎯 到达目标!")
                    if collector.save_episode(success=True):
                        print(f"已收集: {collector.collected_count}/{target_count if target_count else '∞'}")
                else:
                    reason = "碰撞" if info.get('collision') else "超时"
                    print(f"❌ Episode失败: {reason}")
                    collector.discard_episode()
                
                state, _ = env.reset()
                collector.start_episode(goal_pos)
                publish_signal(reset_pub, String, 'reset')
    
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    
    finally:
        # 打印总结
        collector.print_summary()
        
        print(f"{'='*70}")
        print(f"✅ 演示数据已保存到: {save_dir}")
        print(f"{'='*70}\n")
        
        env.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='收集 Goal5 演示数据')
    parser.add_argument('--goal_x', type=float, default=4.5,
                       help='Goal5 的 X 坐标 (默认: 4.5)')
    parser.add_argument('--goal_y', type=float, default=-2.0,
                       help='Goal5 的 Y 坐标 (默认: -2.0)')
    parser.add_argument('--count', type=int, default=None,
                       help='目标收集数量（不指定则无限制）')
    parser.add_argument('--save_dir', type=str, default='./demonstrations_goal5',
                       help='保存目录 (默认: ./demonstrations_goal5)')
    
    args = parser.parse_args()
    
    goal_pos = (args.goal_x, args.goal_y)
    
    try:
        collect_goal5_demos(
            goal_pos=goal_pos,
            target_count=args.count,
            save_dir=args.save_dir
        )
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()