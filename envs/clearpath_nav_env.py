#!/usr/bin/env python3
"""
Clearpath导航环境 - 修复版
修复了Gymnasium兼容性问题
lastest
"""

import rclpy
from rclpy.node import Node
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
import time
import math

# 导入重置类
from .clearpath_reset import ClearpathReset


class ClearpathNavEnv(gym.Env):
    """Clearpath导航环境 - 使用机器人相对极坐标表示目标"""
    
    metadata = {'render_modes': []}
    
    def __init__(self, robot_name='j100_0000', goal_pos=(2.0, 2.0), 
                 max_steps=256, collision_threshold=0.3):
        """
        初始化环境
        
        Args:
            robot_name: 机器人命名空间
            goal_pos: 目标位置 (x, y) - world坐标系
            max_steps: 每个episode最大步数
            collision_threshold: 碰撞阈值（米）
        """
        super().__init__()
        
        # 初始化ROS2
        if not rclpy.ok():
            rclpy.init()
        
        self.node = Node('clearpath_rl_env')
        self.robot_name = robot_name
        self.world_goal = np.array(goal_pos, dtype=np.float32)
        self.max_steps = max_steps
        self.collision_threshold = collision_threshold
        self.max_laser_range = 10.0
        
        # 初始化重置管理器
        self.reset_manager = ClearpathReset(self.node, robot_name)
        
        # Episode管理
        self.goal_relative_to_start = None
        
        # 订阅激光雷达
        self._laser = None
        self.laser_sub = self.node.create_subscription(
            LaserScan,
            f'/{robot_name}/sensors/lidar2d_0/scan',
            self._laser_callback,
            10
        )
        
        # 发布速度命令
        self.cmd_vel_pub = self.node.create_publisher(
            TwistStamped,
            f'/{robot_name}/cmd_vel',
            10
        )
        
        # 观测空间：[距离, 角度, 8个激光, vx, omega] = 12维
        self.observation_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(12,),
            dtype=np.float32
        )
        
        # 动作空间：[线速度, 角速度]
        self.action_space = spaces.Box(
            low=np.array([-0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode状态
        self.step_count = 0
        self.episode_count = 0
        self.prev_distance = 0.0
        
        self.node.get_logger().info('✓ ClearpathNavEnv初始化完成')
        
        # 等待传感器数据
        self._wait_for_sensors()
    
    def _wait_for_sensors(self):
        """等待传感器数据准备"""
        self.node.get_logger().info('等待传感器数据...')
        timeout = time.time() + 10.0
        
        while (self._laser is None or 
               self.reset_manager.current_odom is None) and time.time() < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        if self._laser is None or self.reset_manager.current_odom is None:
            self.node.get_logger().error('传感器数据超时！')
        else:
            self.node.get_logger().info('✓ 传感器数据就绪')
    
    def _laser_callback(self, msg):
        """激光雷达回调"""
        self._laser = msg
    
    def _get_laser_data(self):
        """获取8个方向的激光雷达数据"""
        if self._laser is None:
            return np.ones(8, dtype=np.float32)
        
        ranges = np.array(self._laser.ranges)
        total_points = len(ranges)
        
        # 8个方向：前、右前、右、右后、后、左后、左、左前
        angle_offsets = [0, 45, 90, 135, 180, -135, -90, -45]
        laser_data = []
        
        for angle_offset in angle_offsets:
            index = int((angle_offset + 180) * total_points / 360) % total_points
            distance = ranges[index]
            laser_data.append(distance)
        
        laser_data = np.array(laser_data)
        
        # 处理无效值
        laser_data = np.where(
            np.isfinite(laser_data) & (laser_data > 0),
            laser_data,
            self.max_laser_range
        )
        
        # 归一化到[0, 1]
        laser_data_norm = laser_data / self.max_laser_range
        laser_data_norm = np.clip(laser_data_norm, 0.0, 1.0)
        
        return laser_data_norm
    
    def _compute_goal_in_robot_frame(self):
        """
        计算目标在机器人坐标系下的极坐标表示
        
        Returns:
            distance: 到目标的距离（米）
            bearing: 方位角，相对于机器人前进方向（弧度，[-π, π]）
        """
        odom = self.reset_manager.get_relative_odom()
        if odom is None:
            return 0.0, 0.0
        
        # 当前位置（相对于episode起点）
        current_pos = np.array([odom['x'], odom['y']], dtype=np.float32)
        
        # 目标相对于当前位置的向量
        goal_vec_episode = self.goal_relative_to_start - current_pos
        
        # 转换到机器人坐标系
        robot_yaw = odom['yaw']
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        
        dx_robot = goal_vec_episode[0] * cos_yaw + goal_vec_episode[1] * sin_yaw
        dy_robot = -goal_vec_episode[0] * sin_yaw + goal_vec_episode[1] * cos_yaw
        
        # 转换为极坐标
        distance = math.sqrt(dx_robot**2 + dy_robot**2)
        bearing = math.atan2(dy_robot, dx_robot)
        
        return distance, bearing
    
    def _get_observation(self):
        """获取当前观测 - 使用机器人相对极坐标"""
        odom = self.reset_manager.get_relative_odom()
        if odom is None:
            return np.zeros(12, dtype=np.float32)
        
        # 计算目标在机器人坐标系下的极坐标
        distance, bearing = self._compute_goal_in_robot_frame()
        
        # 获取激光雷达数据
        laser_data = self._get_laser_data()
        
        # 构建观测向量
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = np.clip(distance / 5.0, 0.0, 1.0)  # 距离归一化
        obs[1] = bearing / math.pi  # 角度归一化到[-1, 1]
        obs[2:10] = laser_data  # 8个激光数据
        obs[10] = np.clip(odom['vx'] / 0.5, -1.0, 1.0)  # 线速度
        obs[11] = np.clip(odom['omega'] / 1.0, -1.0, 1.0)  # 角速度
        
        return obs
    
    def _compute_reward(self, obs):
        """计算奖励函数"""
        distance, bearing = self._compute_goal_in_robot_frame()
        
        reward = 0.0
        done = False
        info = {}
        
        # 1. 距离奖励
        distance_reward = (self.prev_distance - distance) * 10.0
        reward += distance_reward
        self.prev_distance = distance
        
        # 2. 朝向奖励
        angle_reward = -abs(bearing) * 0.5
        reward += angle_reward
        
        # 3. 前进速度奖励
        odom = self.reset_manager.get_relative_odom()
        forward_reward = odom['vx'] * 0.2
        reward += forward_reward
        
        # 4. 碰撞检测
        min_laser = np.min(obs[2:10])
        if min_laser < self.collision_threshold / self.max_laser_range:
            reward -= 50.0
            done = True
            info['collision'] = True
        
        # 5. 到达目标
        if distance < 0.3:
            reward += 100.0
            done = True
            info['goal_reached'] = True
        
        # 6. 超时
        if self.step_count >= self.max_steps:
            reward -= 10.0
            done = True
            info['timeout'] = True
        
        # 7. 时间惩罚
        reward -= 0.01
        
        # 记录信息
        info['distance'] = float(distance)
        info['bearing'] = float(bearing)
        info['step'] = int(self.step_count)
        info['reward'] = float(reward)
        
        return reward, done, info
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        # 调用父类reset并处理seed
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
        else:
            super().reset(seed=seed)
        
        # 执行重置
        success = self.reset_manager.reset()
        if not success:
            self.node.get_logger().error('重置失败！')
        
        # 等待稳定
        time.sleep(0.5)
        for _ in range(30):
            rclpy.spin_once(self.node, timeout_sec=0.05)
        
        # 设置目标位置
        self.goal_relative_to_start = self.world_goal.copy()
        
        # 重置状态
        self.step_count = 0
        self.episode_count += 1
        
        # 计算初始距离
        initial_distance, initial_bearing = self._compute_goal_in_robot_frame()
        self.prev_distance = initial_distance
        
        # 获取观测
        obs = self._get_observation()
        
        # 确保观测是正确的numpy数组
        obs = np.array(obs, dtype=np.float32)
        
        # 构造info字典 - 必须是简单的Python类型
        info = {
            'episode': int(self.episode_count),
            'step': 0,
            'distance': float(initial_distance),
            'bearing': float(initial_bearing)
        }
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        # 确保action是numpy array
        action = np.array(action, dtype=np.float32)
        
        # 发送速度命令
        twist = TwistStamped()
        twist.twist.linear.x = float(action[0])
        twist.twist.angular.z = float(action[1])
        self.cmd_vel_pub.publish(twist)
        
        # 等待物理引擎更新
        time.sleep(0.1)
        
        # 刷新传感器数据
        for _ in range(3):
            rclpy.spin_once(self.node, timeout_sec=0.01)
        
        # 获取观测
        obs = self._get_observation()
        obs = np.array(obs, dtype=np.float32)
        
        # 计算奖励
        reward, done, info = self._compute_reward(obs)
        
        self.step_count += 1
        
        truncated = False
        
        return obs, float(reward), bool(done), bool(truncated), info
    
    def close(self):
        """关闭环境"""
        try:
            # 停止机器人
            twist = TwistStamped()
            twist.twist.linear.x = 0.0
            twist.twist.angular.z = 0.0
            self.cmd_vel_pub.publish(twist)
            
            self.reset_manager.stop()
        except Exception as e:
            self.node.get_logger().warn(f'关闭时出错: {e}')
        
        try:
            self.node.destroy_node()
        except:
            pass
        
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except:
                pass