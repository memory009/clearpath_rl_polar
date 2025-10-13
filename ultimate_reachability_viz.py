#!/usr/bin/env python3
"""
终极可达集可视化 - 增强版
展示真实轨迹、密集可达集演化、多误差对比
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

def create_dense_reachability_tube(agent, env, step_interval=5, obs_error=0.01):
    """
    创建密集的可达管道可视化，类似论文图E
    
    Parameters:
    -----------
    step_interval : int
        每隔多少步计算一次可达集（越小越密集，但计算量越大）
    obs_error : float
        观测误差水平
    """
    print(f"\n【2】创建密集可达管道 (error={obs_error*100:.1f}%, interval={step_interval})...")
    
    # 1. 运行真实episode
    trajectory, actions, observations, info = run_actual_episode(agent, env)
    
    # 2. 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    
    # ===== 左图：密集可达管道 =====
    ax_main = axes[0]
    ax_main.set_title(f'Dense Reachable Tube (error={obs_error*100:.1f}%, every {step_interval} steps)', 
                      fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlabel('X Position (m)', fontsize=12)
    ax_main.set_ylabel('Y Position (m)', fontsize=12)
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_aspect('equal')
    
    # 收集所有可达点用于绘制连续管道
    all_reachable_points = []
    step_positions = []
    
    # 在更密集的步长上计算可达集
    total_steps = 0
    for step_idx in range(0, len(trajectory)-1, step_interval):
        if step_idx >= len(observations):
            break
            
        obs = observations[step_idx]
        pos = trajectory[step_idx, :2]
        yaw = trajectory[step_idx, 2]
        
        # 计算可达集
        is_safe, ranges = compute_reachable_set_at_step(agent, obs, yaw, obs_error)
        
        # 使用更短的预测时间但更密集的采样
        paths = simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1)
        
        # 收集这一步的所有可达终点
        step_reachable_points = np.array([p[-1] for p in paths])
        all_reachable_points.append(step_reachable_points)
        step_positions.append(pos)
        
        # 绘制更多的路径样本（减少跳步）
        for path in paths[::2]:  # 每2条画一条，而不是每5条
            ax_main.plot(path[:, 0], path[:, 1], 
                        color='lightgreen', alpha=0.03, linewidth=0.3, zorder=1)
        
        # 标记当前步的位置
        if step_idx % (step_interval * 3) == 0:  # 每隔几个采样点标记一下
            ax_main.plot(pos[0], pos[1], 'o', color='orange', 
                        markersize=6, alpha=0.6, zorder=8)
        
        total_steps += 1
        if total_steps % 5 == 0:
            print(f"  ✓ 已处理 {total_steps} 个时间步...")
    
    # 绘制真实轨迹（在最上层）
    ax_main.plot(trajectory[:, 0], trajectory[:, 1], 
                'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
    
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
                label='Goal', zorder=12,
                markeredgecolor='darkgreen', markeredgewidth=2)
    goal_circle = Circle(goal, 0.3, fill=False, edgecolor='green', 
                        linestyle='--', linewidth=2, alpha=0.5)
    ax_main.add_patch(goal_circle)
    
    ax_main.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_main.set_xlim(-0.5, 2.5)
    ax_main.set_ylim(-0.5, 2.5)
    
    # ===== 右图：累积可达边界（更像论文图E） =====
    ax_tube = axes[1]
    ax_tube.set_title(f'Accumulated Reachable Envelope (error={obs_error*100:.1f}%)', 
                     fontsize=14, fontweight='bold', pad=15)
    ax_tube.set_xlabel('X Position (m)', fontsize=12)
    ax_tube.set_ylabel('Y Position (m)', fontsize=12)
    ax_tube.grid(True, alpha=0.3, linestyle='--')
    ax_tube.set_aspect('equal')
    
    from scipy.spatial import ConvexHull
    
    print(f"  ✓ 绘制累积可达边界...")
    
    # 方法1：绘制整体的可达集外包络（全部点的凸包）
    if len(all_reachable_points) > 0:
        all_points_combined = np.vstack(all_reachable_points)
        if len(all_points_combined) > 3:
            try:
                hull_global = ConvexHull(all_points_combined)
                hull_points_global = all_points_combined[hull_global.vertices]
                hull_points_global = np.vstack([hull_points_global, hull_points_global[0]])
                
                # 绘制全局外包络（半透明）
                ax_tube.fill(hull_points_global[:, 0], hull_points_global[:, 1], 
                           color='lightgreen', alpha=0.2, zorder=1, label='Overall Envelope')
                ax_tube.plot(hull_points_global[:, 0], hull_points_global[:, 1], 
                           color='darkgreen', linewidth=2.5, alpha=0.6, zorder=2)
            except:
                pass
    
    # 方法2：使用散点密度来显示可达管道
    # 收集所有中间点（不仅是终点）来显示管道
    all_tube_points = []
    for step_idx in range(0, len(trajectory)-1, step_interval):
        if step_idx >= len(observations):
            break
        
        obs = observations[step_idx]
        pos = trajectory[step_idx, :2]
        yaw = trajectory[step_idx, 2]
        
        is_safe, ranges = compute_reachable_set_at_step(agent, obs, yaw, obs_error)
        paths = simulate_reachable_tube(pos, yaw, ranges, T=15, dt=0.1)
        
        # 收集每条路径的所有点（不仅是终点）
        for path in paths[::3]:  # 采样一部分路径
            all_tube_points.extend(path[::2])  # 每条路径每隔一个点采样
    
    if len(all_tube_points) > 100:
        all_tube_points = np.array(all_tube_points)
        
        # 使用散点图显示密度
        ax_tube.scatter(all_tube_points[:, 0], all_tube_points[:, 1], 
                       c='green', s=1, alpha=0.02, zorder=1)
        
        # 计算并绘制管道的外边界
        try:
            hull_tube = ConvexHull(all_tube_points)
            hull_points_tube = all_tube_points[hull_tube.vertices]
            hull_points_tube = np.vstack([hull_points_tube, hull_points_tube[0]])
            ax_tube.plot(hull_points_tube[:, 0], hull_points_tube[:, 1], 
                        color='darkgreen', linewidth=2, alpha=0.8, 
                        label='Reachable Tube Boundary', zorder=3)
        except:
            pass
    
    # 绘制真实轨迹
    ax_tube.plot(trajectory[:, 0], trajectory[:, 1], 
                'b-', linewidth=3.5, label='Actual Trajectory', zorder=10, alpha=0.9)
    
    # 起点、终点、目标
    ax_tube.plot(trajectory[0, 0], trajectory[0, 1], 
                'go', markersize=15, label='Start', zorder=11,
                markeredgecolor='darkgreen', markeredgewidth=2)
    ax_tube.plot(trajectory[-1, 0], trajectory[-1, 1], 
                'ro', markersize=15, label='End', zorder=11,
                markeredgecolor='darkred', markeredgewidth=2)
    ax_tube.plot(goal[0], goal[1], 'g*', markersize=25, 
                label='Goal', zorder=12,
                markeredgecolor='darkgreen', markeredgewidth=2)
    goal_circle2 = Circle(goal, 0.3, fill=False, edgecolor='green', 
                         linestyle='--', linewidth=2, alpha=0.5)
    ax_tube.add_patch(goal_circle2)
    
    ax_tube.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax_tube.set_xlim(-0.5, 2.5)
    ax_tube.set_ylim(-0.5, 2.5)
    
    # 总标题
    success_text = "✅ SUCCESS" if info.get('goal_reached', False) else "⚠️ INCOMPLETE"
    plt.suptitle(
        f'Dense Reachability Tube Visualization - {success_text}\n' +
        f'Sampled {len(all_reachable_points)} time steps (every {step_interval} steps) | ' +
        f'Observation Error: {obs_error*100:.1f}%',
        fontsize=15,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout()
    
    # 保存
    filename = f'dense_reachable_tube_err{int(obs_error*100)}_interval{step_interval}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ 已保存: {filename}")
    print(f"  - 采样了 {len(all_reachable_points)} 个时间步")
    
    return trajectory, info

def create_ultimate_visualization(agent, env):
    """创建原始的终极可达集可视化（保留兼容性）"""
    
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
    
    # 关键时刻的可达集
    colors = ['orange', 'purple', 'cyan', 'magenta']
    observation_errors = [0.01, 0.01, 0.01, 0.01]
    
    for idx, (step_idx, color, obs_err) in enumerate(zip(key_steps[:-1], colors, observation_errors)):
        if step_idx >= len(observations):
            continue
            
        obs = observations[step_idx]
        pos = trajectory[step_idx, :2]
        yaw = trajectory[step_idx, 2]
        
        # 计算可达集
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
                pass
        
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

def compare_error_levels(agent, env, step_interval=5):
    """
    比较不同误差水平下的可达管道
    """
    print("\n" + "="*70)
    print(f"比较不同观测误差水平 (采样间隔: {step_interval} steps)")
    print("="*70)
    
    error_levels = [0.01, 0.05]
    
    for err in error_levels:
        print(f"\n处理误差水平: {err*100:.1f}%")
        create_dense_reachability_tube(agent, env, step_interval=step_interval, obs_error=err)

if __name__ == "__main__":
    print("="*70)
    print("终极可达集可视化 - 增强版")
    print("="*70)
    
    config = TD3Config()
    env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
    agent = TD3Agent(12, 2, 0.5, config)
    
    model_path = './models/final_20251009_105845'
    agent.load(model_path)
    
    # 选择可视化模式：
    print("\n选择可视化模式:")
    print("1. 原始可视化（4个关键时刻）")
    print("2. 密集可达管道（推荐，类似论文图E）")
    print("3. 两种都生成")
    
    # 默认选择模式3（生成所有可视化）
    mode = 3
    
    if mode == 1:
        # 原始可视化
        trajectory, info = create_ultimate_visualization(agent, env)
    elif mode == 2:
        # 密集可达管道可视化
        # 参数调整建议：
        # step_interval=5: 平衡（约5分钟）
        # step_interval=3: 密集（约10分钟）
        # step_interval=1: 极致（约30分钟+）
        compare_error_levels(agent, env, step_interval=5)
    else:
        # 生成所有可视化
        print("\n生成原始可视化...")
        trajectory, info = create_ultimate_visualization(agent, env)
        
        print("\n生成密集可达管道可视化...")
        compare_error_levels(agent, env, step_interval=5)
    
    env.close()
    
    print("\n" + "="*70)
    print("✓ 可视化完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  - ultimate_reachability_visualization.png (原始版本)")
    print("  - dense_reachable_tube_err1_interval5.png (密集管道 1%)")
    print("  - dense_reachable_tube_err5_interval5.png (密集管道 5%)")
    print("\n说明:")
    print("  ✅ 左图展示密集采样的可达路径")
    print("  ✅ 右图展示累积的可达管道边界（类似论文图E）")
    print("  ✅ 可以调整 step_interval 参数来控制采样密度")
    print("  ✅ step_interval=5 (当前): 平衡模式")
    print("  ✅ step_interval=3: 更密集")
    print("  ✅ step_interval=1: 极致密集（计算时间长）")
    print("="*70)