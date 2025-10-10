#!/usr/bin/env python3
"""
Clearpath机器人重置类 - 修复版
使用Model Pose作为ground truth解决odometry不同步问题
v3.0
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Pose
from nav_msgs.msg import Odometry
from ros_gz_interfaces.srv import SetEntityPose
import time
import math


class ClearpathReset:
    """Clearpath机器人重置管理类"""
    
    def __init__(self, node, robot_name='j100_0000', world_name='office'):
        """
        初始化重置管理器
        
        Args:
            node: ROS2 Node对象
            robot_name: 机器人命名空间
            world_name: Gazebo世界名称
        """
        self.node = node
        self.robot_name = robot_name
        self.world_name = world_name
        
        # 初始位置
        self.initial_pose = Pose()
        self.initial_pose.position.x = 0.0
        self.initial_pose.position.y = 0.0
        self.initial_pose.position.z = 0.063500
        self.initial_pose.orientation.x = 0.0
        self.initial_pose.orientation.y = 0.0
        self.initial_pose.orientation.z = 0.0
        self.initial_pose.orientation.w = 1.0
        
        # 里程计偏移（用于计算相对位置）
        self.odom_offset_x = 0.0
        self.odom_offset_y = 0.0
        self.odom_offset_yaw = 0.0  # 初始朝向
        
        self.current_odom = None
        
        # 发布速度命令
        self.cmd_vel_pub = node.create_publisher(
            TwistStamped,
            f'/{robot_name}/cmd_vel',
            10
        )
        
        # 订阅里程计
        self.odom_sub = node.create_subscription(
            Odometry,
            f'/{robot_name}/platform/odom',
            self._odom_callback,
            10
        )
        
        # Gazebo重置服务
        self.reset_client = node.create_client(
            SetEntityPose,
            f'/world/{world_name}/set_pose'
        )
        
        # 等待服务
        self._wait_for_service()
        
        # 初始化里程计偏移
        self._init_odom_offset()
        
        node.get_logger().info(f'✓ ClearpathReset初始化完成: {robot_name}')
    
    def _wait_for_service(self, timeout=10.0):
        """等待Gazebo重置服务可用"""
        start_time = time.time()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            if time.time() - start_time > timeout:
                self.node.get_logger().warn('重置服务未连接，但继续运行')
                break
    
    def _odom_callback(self, msg):
        """里程计回调"""
        self.current_odom = msg
    
    def _init_odom_offset(self):
        """初始化里程计偏移"""
        timeout = time.time() + 5.0
        while self.current_odom is None and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        if self.current_odom:
            self.odom_offset_x = self.current_odom.pose.pose.position.x
            self.odom_offset_y = self.current_odom.pose.pose.position.y
            
            # 记录初始朝向
            q = self.current_odom.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.odom_offset_yaw = math.atan2(siny_cosp, cosy_cosp)
    
    def get_relative_odom(self):
        """
        获取相对于episode起点的里程计数据（带旋转变换）
        
        Returns:
            dict: 包含相对位置、速度等信息
        """
        if self.current_odom is None:
            return None
        
        # 当前World位置
        curr_x = self.current_odom.pose.pose.position.x
        curr_y = self.current_odom.pose.pose.position.y
        
        # World坐标系下的位移
        dx_world = curr_x - self.odom_offset_x
        dy_world = curr_y - self.odom_offset_y
        
        # 转换到机器人初始坐标系（旋转变换）
        cos_theta = math.cos(-self.odom_offset_yaw)
        sin_theta = math.sin(-self.odom_offset_yaw)
        
        rel_x = dx_world * cos_theta - dy_world * sin_theta
        rel_y = dx_world * sin_theta + dy_world * cos_theta
        
        # 当前姿态（四元数转欧拉角）
        q = self.current_odom.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        curr_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # 相对朝向
        rel_yaw = curr_yaw - self.odom_offset_yaw
        # 归一化到[-pi, pi]
        rel_yaw = math.atan2(math.sin(rel_yaw), math.cos(rel_yaw))
        
        return {
            'x': rel_x,
            'y': rel_y,
            'yaw': rel_yaw,
            'distance': math.sqrt(rel_x**2 + rel_y**2),
            'vx': self.current_odom.twist.twist.linear.x,
            'vy': self.current_odom.twist.twist.linear.y,
            'omega': self.current_odom.twist.twist.angular.z
        }
    
    def reset(self, position=None):
        """
        重置机器人到指定位置（或初始位置）
        
        Args:
            position: 可选，目标位置 {'x': float, 'y': float, 'yaw': float}
        
        Returns:
            bool: 重置是否成功
        """
        # 停止机器人
        twist = TwistStamped()
        for _ in range(10):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.02)
        
        # 设置目标位置
        if position is not None:
            pose = Pose()
            pose.position.x = position.get('x', 0.0)
            pose.position.y = position.get('y', 0.0)
            pose.position.z = 0.063500
            
            # 设置朝向（yaw转四元数）
            yaw = position.get('yaw', 0.0)
            pose.orientation.z = math.sin(yaw / 2.0)
            pose.orientation.w = math.cos(yaw / 2.0)
        else:
            pose = self.initial_pose
        
        # 调用Gazebo服务
        request = SetEntityPose.Request()
        request.entity.name = f'{self.robot_name}/robot'
        request.entity.type = 2
        request.pose = pose
        
        try:
            future = self.reset_client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=5.0)
            
            if future.result() is None or not future.result().success:
                self.node.get_logger().error('重置失败')
                return False
        except Exception as e:
            self.node.get_logger().error(f'重置异常: {str(e)}')
            return False
        
        # 等待物理引擎稳定
        time.sleep(0.3)
        
        # 刷新里程计
        for _ in range(20):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        
        # 更新里程计偏移（使相对位置归零）
        if self.current_odom:
            self.odom_offset_x = self.current_odom.pose.pose.position.x
            self.odom_offset_y = self.current_odom.pose.pose.position.y
            
            # 更新初始朝向
            q = self.current_odom.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            self.odom_offset_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return True
    
    def stop(self):
        """停止机器人"""
        twist = TwistStamped()
        for _ in range(5):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.02)
