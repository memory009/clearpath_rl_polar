#!/usr/bin/env python3
"""
实时监控Clearpath机器人位置 - 自动同步目标版
自动接收collect_demonstrations.py的目标更新
"""

import sys
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Point
import numpy as np
import math
import time
from datetime import datetime

# ANSI颜色代码
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class RobotPositionMonitor(Node):
    """机器人位置监控器 - 自动同步目标"""
    
    def __init__(self, robot_name='j100_0000', default_goal=(2.0, 2.0)):
        super().__init__('robot_position_monitor')
        
        self.robot_name = robot_name
        
        # 当前目标(默认)
        self.current_goal = np.array(default_goal, dtype=np.float32)
        self.goal_index = 0
        self.total_goals = 0
        
        # 里程计数据
        self.current_odom = None
        self.start_position = None
        self.start_yaw = None
        self.episode_start_time = None
        
        # 统计
        self.total_distance_traveled = 0.0
        self.last_position = None
        self.episode_count = 0
        self.step_count = 0
        
        # 订阅里程计
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{robot_name}/platform/odom',
            self._odom_callback,
            10
        )
        
        # 订阅重置信号
        self.reset_sub = self.create_subscription(
            String,
            '/robot_reset_signal',
            self._reset_callback,
            10
        )
        
        # 订阅目标更新信号(新增)
        self.goal_sub = self.create_subscription(
            Point,
            '/current_goal_position',
            self._goal_callback,
            10
        )
        
        # 订阅目标进度信号(新增)
        self.progress_sub = self.create_subscription(
            String,
            '/goal_progress',
            self._progress_callback,
            10
        )
        
        self.get_logger().info('✓ 机器人位置监控器启动 (自动同步目标)')
        
        # 等待第一个里程计消息
        self._wait_for_odom()
    
    def _wait_for_odom(self):
        """等待第一个里程计消息"""
        print("等待里程计数据...")
        timeout = time.time() + 5.0
        
        while self.current_odom is None and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        if self.current_odom:
            self._reset_episode()
            print("✓ 里程计数据就绪\n")
        else:
            print("⚠️  里程计数据超时，但继续运行\n")
    
    def _odom_callback(self, msg):
        """里程计回调"""
        self.current_odom = msg
        self.step_count += 1
    
    def _reset_callback(self, msg):
        """重置信号回调"""
        if msg.data == 'reset':
            self._reset_episode()
    
    def _goal_callback(self, msg):
        """目标更新回调(新增)"""
        old_goal = self.current_goal.copy()
        self.current_goal = np.array([msg.x, msg.y], dtype=np.float32)
        
        # 检查是否是目标切换(而不是初始化)
        if np.linalg.norm(old_goal - self.current_goal) > 0.1:
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
            print(f"{Colors.BOLD}{Colors.GREEN}🎯 目标已更新!{Colors.END}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
            print(f"  旧目标: ({old_goal[0]:.2f}, {old_goal[1]:.2f})")
            print(f"  新目标: ({self.current_goal[0]:.2f}, {self.current_goal[1]:.2f})")
            print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
    
    def _progress_callback(self, msg):
        """目标进度回调(新增)"""
        # 格式: "goal_idx/total_goals"
        parts = msg.data.split('/')
        if len(parts) == 2:
            self.goal_index = int(parts[0])
            self.total_goals = int(parts[1])
    
    def _reset_episode(self):
        """重置episode"""
        if self.current_odom is None:
            return
        
        # 记录起点
        self.start_position = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # 记录起始朝向
        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.start_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 重置统计
        self.total_distance_traveled = 0.0
        self.last_position = self.start_position.copy()
        self.episode_start_time = time.time()
        self.step_count = 0
        self.episode_count += 1
    
    def get_relative_position(self):
        """获取相对于起点的位置"""
        if self.current_odom is None or self.start_position is None:
            return None
        
        # 当前world位置
        current_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y
        ])
        
        # World坐标系下的位移
        dx_world = current_pos[0] - self.start_position[0]
        dy_world = current_pos[1] - self.start_position[1]
        
        # 转换到机器人初始坐标系(旋转变换)
        cos_theta = math.cos(-self.start_yaw)
        sin_theta = math.sin(-self.start_yaw)
        
        rel_x = dx_world * cos_theta - dy_world * sin_theta
        rel_y = dx_world * sin_theta + dy_world * cos_theta
        
        # 当前姿态
        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        current_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 相对朝向
        rel_yaw = current_yaw - self.start_yaw
        rel_yaw = math.atan2(math.sin(rel_yaw), math.cos(rel_yaw))
        
        # 速度
        vx = self.current_odom.twist.twist.linear.x
        omega = self.current_odom.twist.twist.angular.z
        
        # 更新总距离
        if self.last_position is not None:
            delta = np.linalg.norm(current_pos - self.last_position)
            self.total_distance_traveled += delta
        self.last_position = current_pos.copy()
        
        return {
            'x': rel_x,
            'y': rel_y,
            'yaw': rel_yaw,
            'vx': vx,
            'omega': omega,
            'world_x': current_pos[0],
            'world_y': current_pos[1]
        }
    
    def get_goal_info(self):
        """获取目标信息"""
        pos = self.get_relative_position()
        if pos is None:
            return None
        
        # 计算到目标的距离和方位
        robot_pos = np.array([pos['x'], pos['y']])
        distance = np.linalg.norm(self.current_goal - robot_pos)
        
        # 目标相对于机器人的向量
        goal_vec = self.current_goal - robot_pos
        
        # 转换到机器人坐标系
        cos_yaw = math.cos(pos['yaw'])
        sin_yaw = math.sin(pos['yaw'])
        
        dx_robot = goal_vec[0] * cos_yaw + goal_vec[1] * sin_yaw
        dy_robot = -goal_vec[0] * sin_yaw + goal_vec[1] * cos_yaw
        
        bearing = math.atan2(dy_robot, dx_robot)
        bearing_deg = math.degrees(bearing)
        
        return {
            'distance': distance,
            'bearing': bearing,
            'bearing_deg': bearing_deg
        }
    
    def display_status(self):
        """显示状态(格式化输出)"""
        pos = self.get_relative_position()
        goal_info = self.get_goal_info()
        
        if pos is None or goal_info is None:
            return
        
        # 标题
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🤖 机器人位置监控 - {datetime.now().strftime('%H:%M:%S')}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")
        
        # Episode信息 + 目标进度
        if self.episode_start_time:
            elapsed = time.time() - self.episode_start_time
            progress_str = ""
            if self.total_goals > 0:
                progress_str = f" | 目标进度: {Colors.YELLOW}{self.goal_index + 1}/{self.total_goals}{Colors.END}"
            
            print(f"{Colors.BOLD}Episode #{self.episode_count}{Colors.END} | "
                  f"步数: {Colors.YELLOW}{self.step_count}{Colors.END} | "
                  f"时间: {Colors.YELLOW}{elapsed:.1f}s{Colors.END}"
                  f"{progress_str}")
        
        print(f"{Colors.CYAN}{'-'*70}{Colors.END}")
        
        # 当前位置(相对于起点)
        print(f"\n{Colors.BOLD}📍 当前位置 (相对于起点):{Colors.END}")
        print(f"  X:     {Colors.GREEN}{pos['x']:7.3f}{Colors.END} m")
        print(f"  Y:     {Colors.GREEN}{pos['y']:7.3f}{Colors.END} m")
        print(f"  朝向:   {Colors.GREEN}{math.degrees(pos['yaw']):7.1f}{Colors.END}°")
        
        # World坐标(绝对位置)
        print(f"\n{Colors.BOLD}🌍 World坐标 (绝对位置):{Colors.END}")
        print(f"  X:     {Colors.BLUE}{pos['world_x']:7.3f}{Colors.END} m")
        print(f"  Y:     {Colors.BLUE}{pos['world_y']:7.3f}{Colors.END} m")
        
        # 目标信息(高亮显示)
        print(f"\n{Colors.BOLD}🎯 当前目标:{Colors.END}")
        
        # 目标序号(如果有)
        if self.total_goals > 0:
            print(f"  {Colors.BOLD}{Colors.YELLOW}目标 #{self.goal_index + 1}/{self.total_goals}{Colors.END}")
        
        print(f"  位置: ({Colors.YELLOW}{Colors.BOLD}{self.current_goal[0]:.2f}{Colors.END}, "
              f"{Colors.YELLOW}{Colors.BOLD}{self.current_goal[1]:.2f}{Colors.END})")
        
        # 距离颜色编码
        distance = goal_info['distance']
        if distance < 0.3:
            dist_color = Colors.GREEN + Colors.BOLD
            dist_status = "✅ 已到达! 按'R'保存"
        elif distance < 0.5:
            dist_color = Colors.GREEN
            dist_status = "🔥 非常近!"
        elif distance < 1.0:
            dist_color = Colors.GREEN
            dist_status = "🔥 很近"
        elif distance < 2.0:
            dist_color = Colors.YELLOW
            dist_status = "👍 接近中"
        elif distance < 4.0:
            dist_color = Colors.YELLOW
            dist_status = "🚶 前进中"
        else:
            dist_color = Colors.RED
            dist_status = "🏃 较远"
        
        print(f"  距离:   {dist_color}{distance:7.3f}{Colors.END} m  {dist_status}")
        
        # 方位显示
        bearing_deg = goal_info['bearing_deg']
        if abs(bearing_deg) < 10:
            bearing_status = "⬆️  正前方"
            bearing_color = Colors.GREEN
        elif abs(bearing_deg) < 45:
            if bearing_deg > 0:
                bearing_status = "↖️  左前方"
            else:
                bearing_status = "↗️  右前方"
            bearing_color = Colors.YELLOW
        elif abs(bearing_deg) < 90:
            if bearing_deg > 0:
                bearing_status = "⬅️  左侧"
            else:
                bearing_status = "➡️  右侧"
            bearing_color = Colors.YELLOW
        else:
            if bearing_deg > 0:
                bearing_status = "↙️  左后方"
            else:
                bearing_status = "↘️  右后方"
            bearing_color = Colors.RED
        
        print(f"  方位角: {bearing_color}{bearing_deg:7.1f}{Colors.END}°  {bearing_status}")
        
        # 速度信息
        print(f"\n{Colors.BOLD}🚗 运动状态:{Colors.END}")
        print(f"  线速度: {Colors.GREEN}{pos['vx']:7.3f}{Colors.END} m/s")
        print(f"  角速度: {Colors.GREEN}{pos['omega']:7.3f}{Colors.END} rad/s")
        print(f"  总行程: {Colors.YELLOW}{self.total_distance_traveled:7.3f}{Colors.END} m")
        
        # 底部分隔线
        print(f"\n{Colors.CYAN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}提示:{Colors.END} 监控会自动同步collect_demonstrations.py的目标")
        print(f"{Colors.CYAN}{'='*70}{Colors.END}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='监控机器人位置(自动同步目标)')
    parser.add_argument('--robot', type=str, default='j100_0000',
                       help='机器人命名空间')
    parser.add_argument('--goal', type=str, default='2.0,2.0',
                       help='初始目标位置,格式: x,y (会被自动覆盖)')
    parser.add_argument('--rate', type=float, default=2.0,
                       help='更新频率(Hz)')
    
    args = parser.parse_args()
    
    # 初始化ROS2
    if not rclpy.ok():
        rclpy.init()
    
    # 解析初始目标位置
    default_goal = tuple(map(float, args.goal.split(',')))
    
    # 创建监控器
    monitor = RobotPositionMonitor(
        robot_name=args.robot,
        default_goal=default_goal
    )
    
    print(f"\n{Colors.BOLD}{Colors.GREEN}✓ 监控器已启动 (自动同步模式){Colors.END}")
    print(f"  机器人: {args.robot}")
    print(f"  初始目标: {default_goal}")
    print(f"  更新频率: {args.rate} Hz")
    print(f"\n{Colors.BOLD}{Colors.CYAN}🔄 目标会自动同步collect_demonstrations.py{Colors.END}")
    print()
    
    # 主循环
    rate = 1.0 / args.rate
    
    try:
        while rclpy.ok():
            # 更新ROS消息
            rclpy.spin_once(monitor, timeout_sec=0.01)
            
            # 显示状态
            monitor.display_status()
            
            # 延迟
            time.sleep(rate)
    
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  用户中断监控{Colors.END}\n")
    
    finally:
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()