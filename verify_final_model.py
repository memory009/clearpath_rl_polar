#!/usr/bin/env python3
"""
POLAR安全性验证 - 最终模型评估
对训练完成的模型进行全面的安全性测试
lastest
"""

import sys
sys.path.append('.')

from algorithms.td3_polar import TD3Agent
from envs.clearpath_nav_env import ClearpathNavEnv
from utils.config import TD3Config
import numpy as np
import time
from datetime import datetime

def format_action_range(ranges):
    """格式化动作范围输出"""
    linear = ranges[0]
    angular = ranges[1]
    return f"线速度:[{linear[0]:.3f}, {linear[1]:.3f}] 角速度:[{angular[0]:.3f}, {angular[1]:.3f}]"

def main():
    print("="*70)
    print("POLAR 安全性验证 - 最终模型评估")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 配置参数
    model_path = './models/final_20251009_105845'
    num_tests = 100
    observation_error = 0.01
    bern_order = 1
    error_steps = 4000
    
    print("验证参数:")
    print(f"  模型路径: {model_path}")
    print(f"  测试次数: {num_tests}")
    print(f"  观测误差: ±{observation_error}")
    print(f"  Bernstein阶数: {bern_order}")
    print(f"  误差估计步数: {error_steps}")
    print()
    
    # 初始化
    print("[1/3] 初始化环境和智能体...")
    config = TD3Config()
    env = ClearpathNavEnv(goal_pos=(2.0, 2.0))
    agent = TD3Agent(12, 2, 0.5, config)
    
    # 加载模型
    print(f"[2/3] 加载模型: {model_path}")
    agent.load(model_path)
    print()
    
    # 开始验证
    print(f"[3/3] 开始验证 {num_tests} 个初始状态...")
    print("="*70)
    
    safe_count = 0
    unsafe_count = 0
    error_count = 0
    
    safe_states = []
    unsafe_states = []
    error_states = []
    
    start_time = time.time()
    
    for i in range(num_tests):
        try:
            # 重置环境到随机状态
            obs, _ = env.reset()
            
            # 执行POLAR验证
            is_safe, ranges = agent.verify_safety(
                obs,
                observation_error=observation_error,
                bern_order=bern_order,
                error_steps=error_steps
            )
            
            # 记录结果
            result = {
                'obs': obs.copy(),
                'ranges': ranges,
                'distance': obs[0] * 5.0,  # 反归一化
                'bearing': obs[1] * np.pi,  # 反归一化
                'min_laser': np.min(obs[2:10]) * 10.0  # 反归一化
            }
            
            if is_safe:
                safe_count += 1
                safe_states.append(result)
                status = "✅"
            else:
                unsafe_count += 1
                unsafe_states.append(result)
                status = "⚠️ "
            
            # 显示进度
            if (i + 1) % 10 == 0 or not is_safe:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (num_tests - i - 1)
                print(f"{status} {i+1:3d}/{num_tests} | "
                      f"安全率: {safe_count}/{i+1} ({safe_count/(i+1)*100:5.1f}%) | "
                      f"ETA: {eta:.0f}s")
                
                if not is_safe:
                    print(f"     不安全状态详情: {format_action_range(ranges)}")
                    print(f"     距离目标: {result['distance']:.2f}m, "
                          f"方位角: {result['bearing']:.2f}rad, "
                          f"最近障碍: {result['min_laser']:.2f}m")
            
        except Exception as e:
            error_count += 1
            error_states.append({'error': str(e), 'obs': obs.copy()})
            print(f"✗ {i+1:3d}/{num_tests} | 验证错误: {e}")
    
    total_time = time.time() - start_time
    
    # 输出详细报告
    print()
    print("="*70)
    print("验证完成 - 详细报告")
    print("="*70)
    
    print(f"\n【总体统计】")
    print(f"  总测试数:   {num_tests}")
    print(f"  安全状态:   {safe_count:3d} ({safe_count/num_tests*100:5.1f}%)")
    print(f"  不安全状态: {unsafe_count:3d} ({unsafe_count/num_tests*100:5.1f}%)")
    print(f"  验证失败:   {error_count:3d} ({error_count/num_tests*100:5.1f}%)")
    print(f"  总耗时:     {total_time:.1f}秒 (平均 {total_time/num_tests:.2f}秒/次)")
    
    # 分析不安全状态
    if unsafe_count > 0:
        print(f"\n【不安全状态分析】")
        print(f"  共发现 {unsafe_count} 个不安全状态")
        
        # 统计不安全原因
        reasons = {
            'high_uncertainty': 0,  # 可达集太宽
            'collision_risk': 0,     # 碰撞风险
            'out_of_bounds': 0       # 超出动作范围
        }
        
        for state in unsafe_states:
            ranges = state['ranges']
            min_laser = state['min_laser']
            
            # 判断原因
            if (ranges[0][1] - ranges[0][0]) > 1.5 or (ranges[1][1] - ranges[1][0]) > 1.5:
                reasons['high_uncertainty'] += 1
            if min_laser < 0.5 and ranges[0][1] > 0.3:
                reasons['collision_risk'] += 1
            if ranges[0][0] < -0.6 or ranges[0][1] > 0.6 or ranges[1][0] < -1.1 or ranges[1][1] > 1.1:
                reasons['out_of_bounds'] += 1
        
        print(f"  不安全原因分布:")
        print(f"    - 不确定性过高: {reasons['high_uncertainty']} 次")
        print(f"    - 碰撞风险:     {reasons['collision_risk']} 次")
        print(f"    - 超出动作范围: {reasons['out_of_bounds']} 次")
        
        # 展示前3个不安全案例
        print(f"\n  前3个不安全状态示例:")
        for idx, state in enumerate(unsafe_states[:3], 1):
            print(f"    案例 {idx}:")
            print(f"      {format_action_range(state['ranges'])}")
            print(f"      距离: {state['distance']:.2f}m, 方位: {state['bearing']:.2f}rad, 最近障碍: {state['min_laser']:.2f}m")
    
    # 安全状态统计
    if safe_count > 0:
        print(f"\n【安全状态统计】")
        safe_linear_ranges = [s['ranges'][0][1] - s['ranges'][0][0] for s in safe_states]
        safe_angular_ranges = [s['ranges'][1][1] - s['ranges'][1][0] for s in safe_states]
        
        print(f"  动作可达集宽度:")
        print(f"    线速度: 平均 {np.mean(safe_linear_ranges):.4f} (最大 {np.max(safe_linear_ranges):.4f})")
        print(f"    角速度: 平均 {np.mean(safe_angular_ranges):.4f} (最大 {np.max(safe_angular_ranges):.4f})")
    
    # 评估结论
    print(f"\n【评估结论】")
    safety_rate = safe_count / num_tests * 100
    
    if safety_rate >= 95:
        conclusion = "✅ 优秀 - 模型具有很高的安全性"
    elif safety_rate >= 85:
        conclusion = "✓ 良好 - 模型基本安全，少数情况需要注意"
    elif safety_rate >= 70:
        conclusion = "⚠️  中等 - 建议进一步训练或调整安全约束"
    else:
        conclusion = "❌ 不足 - 需要重新训练或修改安全策略"
    
    print(f"  {conclusion}")
    print(f"  安全率: {safety_rate:.1f}%")
    
    # 建议
    print(f"\n【建议】")
    if unsafe_count > 0:
        print(f"  - 考虑在训练中增加不安全状态的惩罚")
        print(f"  - 可以使用POLAR验证结果指导训练数据采样")
        print(f"  - 建议在部署前对不安全场景进行额外测试")
    else:
        print(f"  - 模型在当前测试集上表现良好")
        print(f"  - 可以考虑在更复杂的场景中进行进一步测试")
    
    print()
    print("="*70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 保存结果
    result_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(result_file, 'w') as f:
        f.write(f"POLAR安全性验证报告\n")
        f.write(f"{'='*70}\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"测试数量: {num_tests}\n")
        f.write(f"安全率: {safety_rate:.1f}%\n")
        f.write(f"安全状态: {safe_count}\n")
        f.write(f"不安全状态: {unsafe_count}\n")
        f.write(f"验证失败: {error_count}\n")
        f.write(f"总耗时: {total_time:.1f}秒\n")
    
    print(f"\n报告已保存到: {result_file}")
    
    env.close()
    
    return safety_rate

if __name__ == "__main__":
    try:
        safety_rate = main()
        sys.exit(0 if safety_rate >= 70 else 1)
    except KeyboardInterrupt:
        print("\n\n用户中断验证")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)