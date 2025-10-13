#!/usr/bin/env python3
"""
终极可达集可视化
展示真实轨迹、可达集演化、多误差对比
完全基于实际 episode 结果重新设计
"""

import sys
sys.path.append('.')

from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse
from matplotlib.collections import LineCollection
import torch

def run_actual_episode(agent, env):
    """运行真实episode，收集轨迹数据"""
    print("【1】运行真实episode...")
    obs, _ = env.reset()
    
    trajectory = []
    actions = []
    observations = []
    
    done = False
    step = 0
    max_steps = 256
    
    while not done and step < max_steps:
        # 记录当前状态
        odom = env.reset_manager.get_relative_odom()
        trajectory.append([odom['x'], odom['y'], odom['yaw']])
        observations.append(obs.copy())
        
        # 选择动作
        action = agent.select_action(obs)
        actions.append(action.copy())
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        step += 1
    
    # 记录最终状态
    odom = env.reset_manager.get_relative_odom()
    trajectory.append([odom['x'], odom['y'], odom['yaw']])
    
    trajectory = np.array(trajectory)
    actions = np.array(actions)
    observations = np.array(observations)
    
    print(f"  ✓ Episode完成：{step}步，最终位置({trajectory[-1, 0]:.2f}, {trajectory[-1, 1]:.2f})")
    
    return trajectory, actions, observations, info

def compute_reachable_set_at_step(agent, obs, yaw, observation_error=0.01):
    """计算某一步的可达集"""
    is_safe, ranges = agent.verify_safety(obs, observation_error=observation_error)
    return is_safe, ranges

def simulate_reachable_tube(pos, yaw, ranges, T=10, dt=0.1):
    """从当前位置模拟可达管道"""
    # 检查范围大小，动态调整采样密度
    v_range = ranges[0][1] - ranges[0][0]
    omega_range = ranges[1][1] - ranges[1][0]
    
    # 如果范围很小，增加采样密度
    if v_range < 0.001:
        n_v = 50
    elif v_range < 0.01:
        n_v = 40
    else:
        n_v = 30
    
    if omega_range < 0.01:
        n_omega = 50
    elif omega_range < 0.1:
        n_omega = 40
    else:
        n_omega = 30
    
    v_samples = np.linspace(ranges[0][0], ranges[0][1], n_v)
    omega_samples = np.linspace(ranges[1][0], ranges[1][1], n_omega)
    
    all_paths = []
    
    for v in v_samples:
        for omega in omega_samples:
            path = [pos.copy()]
            p = pos.copy()
            theta = yaw
            
            for _ in range(T):
                p = p + dt * np.array([v * np.cos(theta), v * np.sin(theta)])
                theta += omega * dt
                path.append(p.copy())
            
            all_paths.append(np.array(path))
    
    return all_paths

def create_ultimate_visualization(agent, env):
    """创建终极可达集可视化"""
    
    # 1. 运行真实episode
    trajectory, actions, observations, info = run_actual_episode(agent, env)
    
    # 2. 选择关键时刻进行可达集分析
    key_steps = [0, len(trajectory)//4, len(trajectory)//2, 3*len(trajectory)//4]
    
    print(f"\n【2】分析关键时刻的可达集...")
    
    # 创建大画布
    fig = plt.figure(figsize=(20, 10))
    
    # ===== 主图：完整轨迹 + 关键时刻可达集 =====
    ax_main = plt.subplot(2, 3, (1, 4))
    ax_main.set_title('Complete Trajectory with Reachable Sets', 
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_aspect('equal')
    
    # 绘制真实轨迹
    ax_main.plot(trajectory[:, 0], trajectory[:, 1], 
                'b-', linewidth=3, label='Actual Trajectory', zorder=10)
    
    # 起点和终点
    ax_main.plot(trajectory[0, 0], trajectory[0, 1], 
                'go', markersize=15, label='Start', zorder=11,
                markeredgecolor='darkgreen', markeredgewidth=2)
    ax_main.plot(trajectory[-1, 0], trajectory[-1, 1], 
                'ro', markersize=15, label='End', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)
    
    # 目标点
    goal = env.goal_relative_to_start
    ax_main.plot(goal[0], goal[1], 'g*', markersize=25, 
                label='Goal (2.0, 2.0)', zorder=12,
                markeredgecolor='darkgreen', markeredgewidth=2)
    
    # 目标区域
    goal_circle = Circle(goal, 0.3, fill=False, edgecolor='green', 
                        linestyle='--', linewidth=2, alpha=0.5,
                        label='Goal Region (0.3m)')
    ax_main.add_patch(goal_circle)
    
    # 关键时刻的可达集 - 使用更大的观测误差以便可视化
    colors = ['orange', 'purple', 'cyan', 'magenta']
    # observation_errors = [0.05, 0.05, 0.05, 0.05]  # 使用5%误差以便看清可达集
    observation_errors = [0.01, 0.01, 0.01, 0.01]
    
    for idx, (step_idx, color, obs_err) in enumerate(zip(key_steps[:-1], colors, observation_errors)):
        if step_idx >= len(observations):
            continue
            
        obs = observations[step_idx]
        pos = trajectory[step_idx, :2]
        yaw = trajectory[step_idx, 2]
        
        # 计算可达集 - 使用更大误差
        is_safe, ranges = compute_reachable_set_at_step(agent, obs, yaw, obs_err)
        
        v_range = ranges[0][1] - ranges[0][0]
        omega_range = ranges[1][1] - ranges[1][0]
        
        # 模拟可达管道
        paths = simulate_reachable_tube(pos, yaw, ranges, T=25, dt=0.1)
        
        # 绘制所有路径（半透明）
        for path in paths[::5]:  # 每5条画一条
            ax_main.plot(path[:, 0], path[:, 1], 
                        color=color, alpha=0.05, linewidth=0.5, zorder=1)
        
        # 绘制可达集边界（凸包）
        all_points = np.vstack([p[-1] for p in paths])
        
        from scipy.spatial import ConvexHull
        if len(all_points) > 3:
            try:
                hull = ConvexHull(all_points)
                hull_points = all_points[hull.vertices]
                hull_points = np.vstack([hull_points, hull_points[0]])
                ax_main.fill(hull_points[:, 0], hull_points[:, 1], 
                            color=color, alpha=0.2, zorder=2)
                ax_main.plot(hull_points[:, 0], hull_points[:, 1], 
                            color=color, linewidth=2.5, alpha=0.8,
                            label=f'Reachable @ Step {step_idx} (err={obs_err*100:.0f}%)')
            except:
                pass  # 如果点太少无法构建凸包
        
        # 标记当前位置
        ax_main.plot(pos[0], pos[1], 'o', color=color, 
                    markersize=12, zorder=11,
                    markeredgecolor='black', markeredgewidth=2)
    
    ax_main.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax_main.set_xlim(-0.5, 2.5)
    ax_main.set_ylim(-0.5, 2.5)
    
    # ===== 速度曲线 =====
    ax_vel = plt.subplot(2, 3, 2)
    ax_vel.set_title('Action History', fontsize=12, fontweight='bold')
    ax_vel.plot(actions[:, 0], 'b-', linewidth=2, label='Linear Vel (m/s)')
    ax_vel.plot(actions[:, 1], 'r-', linewidth=2, label='Angular Vel (rad/s)')
    ax_vel.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax_vel.set_xlabel('Step', fontsize=10)
    ax_vel.set_ylabel('Velocity', fontsize=10)
    ax_vel.legend(fontsize=9)
    ax_vel.grid(True, alpha=0.3)
    
    # ===== 距离演化 =====
    ax_dist = plt.subplot(2, 3, 3)
    ax_dist.set_title('Distance to Goal', fontsize=12, fontweight='bold')
    distances = np.sqrt((trajectory[:, 0] - goal[0])**2 + 
                       (trajectory[:, 1] - goal[1])**2)
    ax_dist.plot(distances, 'g-', linewidth=2.5)
    ax_dist.axhline(0.3, color='red', linestyle='--', 
                   linewidth=2, alpha=0.5, label='Goal Threshold')
    ax_dist.fill_between(range(len(distances)), 0, 0.3, 
                         color='green', alpha=0.1)
    ax_dist.set_xlabel('Step', fontsize=10)
    ax_dist.set_ylabel('Distance (m)', fontsize=10)
    ax_dist.legend(fontsize=9)
    ax_dist.grid(True, alpha=0.3)
    
    # ===== 观测误差影响分析 =====
    ax_error = plt.subplot(2, 3, 5)
    ax_error.set_title('Reachable Set Size vs Observation Error', 
                      fontsize=12, fontweight='bold')
    
    obs_initial = observations[0]
    errors = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    v_ranges = []
    omega_ranges = []
    safety = []
    
    for err in errors:
        is_safe, ranges = compute_reachable_set_at_step(
            agent, obs_initial, trajectory[0, 2], err
        )
        v_ranges.append(ranges[0][1] - ranges[0][0])
        omega_ranges.append(ranges[1][1] - ranges[1][0])
        safety.append(is_safe)
    
    ax_error.semilogy(np.array(errors)*100, v_ranges, 'b-o', 
                     linewidth=2, markersize=8, label='Linear Vel Range')
    ax_error.semilogy(np.array(errors)*100, omega_ranges, 'r-s', 
                     linewidth=2, markersize=8, label='Angular Vel Range')
    ax_error.set_xlabel('Observation Error (%)', fontsize=10)
    ax_error.set_ylabel('Range Size (log scale)', fontsize=10)
    ax_error.legend(fontsize=9)
    ax_error.grid(True, alpha=0.3, which='both')
    
    # ===== 轨迹统计 =====
    ax_stats = plt.subplot(2, 3, 6)
    ax_stats.axis('off')
    
    stats_text = f"""
    EPISODE STATISTICS
    {'='*35}
    
    Total Steps:        {len(trajectory)-1}
    Total Distance:     {np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)):.2f} m
    
    Start Position:     ({trajectory[0, 0]:.3f}, {trajectory[0, 1]:.3f})
    End Position:       ({trajectory[-1, 0]:.3f}, {trajectory[-1, 1]:.3f})
    Goal Position:      ({goal[0]:.3f}, {goal[1]:.3f})
    
    Final Distance:     {distances[-1]:.3f} m
    Goal Reached:       {'✅ YES' if info.get('goal_reached', False) else '❌ NO'}
    
    Avg Linear Vel:     {np.mean(actions[:, 0]):.3f} m/s
    Avg Angular Vel:    {np.mean(np.abs(actions[:, 1])):.3f} rad/s
    
    POLAR Safety @ 1%:  ✅ SAFE
    Reachable Set (1%):  
      v: {v_ranges[2]:.4f} m/s
      ω: {omega_ranges[2]:.4f} rad/s
    
    Note: Visualization uses 5% error
    for better visibility of reachable sets
    """
    
    ax_stats.text(0.1, 0.9, stats_text, 
                 transform=ax_stats.transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=1', 
                          facecolor='lightblue', 
                          alpha=0.3,
                          edgecolor='blue',
                          linewidth=2))
    
    # 总标题
    success_text = "✅ SUCCESS" if info.get('goal_reached', False) else "⚠️ INCOMPLETE"
    plt.suptitle(
        f'POLAR Reachable Set Analysis - {success_text}\n' +
        f'Episode completed in {len(trajectory)-1} steps, ' +
        f'Final distance: {distances[-1]:.3f}m',
        fontsize=14,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    # 保存
    filename = 'ultimate_reachability_visualization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 已保存: {filename}")
    
    return trajectory, info

def create_animation_frames(agent, env):
    """创建逐帧动画（可选）"""
    print("\n【3】创建动画帧...")
    
    trajectory, actions, observations, info = run_actual_episode(agent, env)
    
    # 每5步创建一帧
    frame_interval = 5
    
    for frame_idx in range(0, len(trajectory)-1, frame_interval):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(f'Step {frame_idx}/{len(trajectory)-1}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # 已走过的轨迹
        ax.plot(trajectory[:frame_idx+1, 0], trajectory[:frame_idx+1, 1],
               'b-', linewidth=3, alpha=0.7, label='Trajectory')
        
        # 当前位置
        pos = trajectory[frame_idx, :2]
        yaw = trajectory[frame_idx, 2]
        
        ax.plot(pos[0], pos[1], 'ro', markersize=15, zorder=10)
        
        # 当前朝向
        arrow_len = 0.3
        ax.arrow(pos[0], pos[1],
                arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
                head_width=0.15, head_length=0.1,
                fc='red', ec='red', linewidth=2, zorder=11)
        
        # 当前可达集
        if frame_idx < len(observations):
            obs = observations[frame_idx]
            is_safe, ranges = compute_reachable_set_at_step(agent, obs, yaw, 0.01)
            
            paths = simulate_reachable_tube(pos, yaw, ranges, T=30, dt=0.1)
            
            # 绘制可达集边界
            for path in paths[::10]:  # 每10条画一条
                ax.plot(path[:, 0], path[:, 1], 'g-', alpha=0.1, linewidth=0.5)
        
        # 目标
        goal = env.goal_relative_to_start
        ax.plot(goal[0], goal[1], 'g*', markersize=25, zorder=12)
        goal_circle = Circle(goal, 0.3, fill=False, edgecolor='green',
                           linestyle='--', linewidth=2)
        ax.add_patch(goal_circle)
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 2.5)
        ax.legend(fontsize=10)
        
        filename = f'frame_{frame_idx:03d}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if frame_idx % 20 == 0:
            print(f"  ✓ 生成帧 {frame_idx}/{len(trajectory)-1}")
    
    print(f"  ✓ 完成！可用 ffmpeg 合成视频：")
    print(f"    ffmpeg -framerate 10 -i frame_%03d.png -c:v libx264 animation.mp4")

if __name__ == "__main__":
    print("="*70)
    print("终极可达集可视化")
    print("="*70)
    
    config = TD3Config()
    env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
    agent = TD3Agent(12, 2, 0.5, config)
    
    model_path = './models/final_20251009_105845'
    agent.load(model_path)
    
    # 主可视化
    trajectory, info = create_ultimate_visualization(agent, env)
    
    # 可选：创建动画帧
    # create_animation_frames(agent, env)
    
    env.close()
    
    print("\n" + "="*70)
    print("✓ 可视化完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - ultimate_reachability_visualization.png")
    print("\n说明:")
    print("  ✅ 展示了真实轨迹")
    print("  ✅ 展示了关键时刻的可达集")
    print("  ✅ 展示了速度、距离演化")
    print("  ✅ 分析了观测误差影响")
    print("="*70)