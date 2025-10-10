#!/usr/bin/env python3
"""
完整的项目文件检测脚本
检查所有文件是否存在并可以正常导入
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.getcwd())

print("=" * 80)
print("Clearpath RL POLAR 项目完整性检查")
print("=" * 80)

# ============ 第1步：检查文件是否存在 ============
print("\n[第1步] 检查文件是否存在...")
print("-" * 80)

required_files = {
    '核心算法': [
        'algorithms/__init__.py',
        'algorithms/networks.py',
        'algorithms/td3_polar.py',
        'algorithms/memory.py',
    ],
    '验证模块': [
        'verification/__init__.py',
        'verification/taylor_model.py',
        'verification/activation_functions.py',
    ],
    '环境模块': [
        'envs/__init__.py',
        'envs/clearpath_nav_env.py',
        'envs/clearpath_reset.py',
    ],
    '配置和脚本': [
        'utils/__init__.py',
        'utils/config.py',
        'train.py',
    ],
}

missing_files = []
all_files_exist = True

for category, files in required_files.items():
    print(f"\n【{category}】")
    for filepath in files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            if size > 0:
                print(f"  ✅ {filepath:40s} ({size:,} bytes)")
            else:
                print(f"  ⚠️  {filepath:40s} (文件为空!)")
                missing_files.append(filepath)
                all_files_exist = False
        else:
            print(f"  ❌ {filepath:40s} (不存在)")
            missing_files.append(filepath)
            all_files_exist = False

# ============ 第2步：测试模块导入 ============
print("\n" + "=" * 80)
print("[第2步] 测试模块导入...")
print("-" * 80)

import_tests = [
    ('algorithms.networks', 'Actor, Critic', '网络定义'),
    ('algorithms.memory', 'ReplayBuffer', '经验回放'),
    ('algorithms.td3_polar', 'TD3Agent', 'TD3智能体'),
    ('verification.taylor_model', 'TaylorModel, compute_tm_bounds', 'Taylor模型'),
    ('verification.activation_functions', 'Activation_functions', '激活函数'),
    ('utils.config', 'TD3Config', '配置'),
]

import_success = []
import_failed = []

for module_name, imports, description in import_tests:
    try:
        exec(f"from {module_name} import {imports}")
        print(f"✅ {module_name:40s} ({description})")
        import_success.append(module_name)
    except ImportError as e:
        print(f"❌ {module_name:40s} 导入失败: {e}")
        import_failed.append((module_name, str(e)))
    except Exception as e:
        print(f"⚠️  {module_name:40s} 错误: {e}")
        import_failed.append((module_name, str(e)))

# ============ 第3步：测试核心功能 ============
print("\n" + "=" * 80)
print("[第3步] 测试核心功能...")
print("-" * 80)

functional_tests_passed = 0
functional_tests_total = 5

# 测试1: 创建Actor网络
print("\n[测试1] 创建Actor网络...")
try:
    from algorithms.networks import Actor
    import torch
    actor = Actor(state_dim=12, action_dim=2, max_action=0.5)
    test_state = torch.randn(1, 12)
    action = actor(test_state)
    assert action.shape == (1, 2), "输出维度错误"
    print("✅ Actor网络工作正常")
    functional_tests_passed += 1
except Exception as e:
    print(f"❌ Actor测试失败: {e}")

# 测试2: 创建Critic网络
print("\n[测试2] 创建Critic网络...")
try:
    from algorithms.networks import Critic
    critic = Critic(state_dim=12, action_dim=2)
    test_action = torch.randn(1, 2)
    q1, q2 = critic(test_state, test_action)
    assert q1.shape == (1, 1), "Q1维度错误"
    assert q2.shape == (1, 1), "Q2维度错误"
    print("✅ Critic网络工作正常")
    functional_tests_passed += 1
except Exception as e:
    print(f"❌ Critic测试失败: {e}")

# 测试3: 测试经验回放
print("\n[测试3] 测试经验回放...")
try:
    from algorithms.memory import ReplayBuffer
    import numpy as np
    buffer = ReplayBuffer(state_dim=12, action_dim=2, max_size=1000)
    
    # 添加一些经验
    for _ in range(10):
        buffer.push(
            state=np.random.randn(12),
            action=np.random.randn(2),
            next_state=np.random.randn(12),
            reward=np.random.randn(),
            done=0.0
        )
    
    assert buffer.size == 10, "经验数量错误"
    print("✅ ReplayBuffer工作正常")
    functional_tests_passed += 1
except Exception as e:
    print(f"❌ ReplayBuffer测试失败: {e}")

# 测试4: 测试Taylor模型
print("\n[测试4] 测试Taylor模型...")
try:
    from verification.taylor_model import TaylorModel, compute_tm_bounds
    import sympy as sym
    
    x = sym.Symbol('x')
    poly = sym.Poly(2*x + 1)
    tm = TaylorModel(poly, [-0.1, 0.1])
    a, b = compute_tm_bounds(tm)
    
    assert isinstance(a, (int, float)), "下界类型错误"
    assert isinstance(b, (int, float)), "上界类型错误"
    assert a < b, "边界关系错误"
    print("✅ Taylor模型工作正常")
    functional_tests_passed += 1
except Exception as e:
    print(f"❌ Taylor模型测试失败: {e}")

# 测试5: 测试TD3Agent创建
print("\n[测试5] 测试TD3Agent创建...")
try:
    from algorithms.td3_polar import TD3Agent
    from utils.config import TD3Config
    
    config = TD3Config()
    agent = TD3Agent(
        state_dim=12,
        action_dim=2,
        max_action=0.5,
        config=config
    )
    
    # 测试动作选择
    test_obs = np.random.randn(12)
    action = agent.select_action(test_obs)
    assert action.shape == (2,), "动作维度错误"
    print("✅ TD3Agent工作正常")
    functional_tests_passed += 1
except Exception as e:
    print(f"❌ TD3Agent测试失败: {e}")

# ============ 第4步：检查环境依赖 ============
print("\n" + "=" * 80)
print("[第4步] 检查环境依赖...")
print("-" * 80)

dependencies = {
    'torch': '2.0.0',
    'numpy': '1.24.0',
    'sympy': '1.12',
    'gymnasium': '0.29.0',
    'tqdm': '4.65.0',
}

deps_ok = True
for package, min_version in dependencies.items():
    try:
        mod = __import__(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✅ {package:20s} {version:20s} (需要 >= {min_version})")
    except ImportError:
        print(f"❌ {package:20s} 未安装 (需要 >= {min_version})")
        deps_ok = False

# ============ 生成总结报告 ============
print("\n" + "=" * 80)
print("检查总结")
print("=" * 80)

print(f"\n【文件检查】")
if all_files_exist:
    print("✅ 所有必需文件都存在")
else:
    print(f"❌ 缺少 {len(missing_files)} 个文件:")
    for f in missing_files:
        print(f"   - {f}")

print(f"\n【模块导入】")
print(f"成功: {len(import_success)}/{len(import_tests)}")
if import_failed:
    print("失败的模块:")
    for module, error in import_failed:
        print(f"   - {module}: {error}")

print(f"\n【功能测试】")
print(f"通过: {functional_tests_passed}/{functional_tests_total}")

print(f"\n【依赖检查】")
if deps_ok:
    print("✅ 所有依赖都已安装")
else:
    print("❌ 部分依赖缺失")

# ============ 给出建议 ============
print("\n" + "=" * 80)
print("🎯 下一步建议")
print("=" * 80)

if all_files_exist and len(import_success) == len(import_tests) and functional_tests_passed == functional_tests_total and deps_ok:
    print("\n🎉 恭喜！所有检查都通过了！")
    print("\n您现在可以：")
    print("\n【选项1】快速测试训练流程（推荐）")
    print("  python3 train.py --timesteps 1000 --start_timesteps 500 --verify_interval 200")
    print("  预计耗时: 10-15分钟")
    print("  目的: 验证整个流程是否正常")
    print("\n【选项2】开始完整训练")
    print("  python3 train.py --timesteps 100000 --start_timesteps 25000")
    print("  预计耗时: 6-8小时")
    print("  目的: 获得论文级别的训练结果")
    print("\n【选项3】查看训练脚本帮助")
    print("  python3 train.py --help")
    
elif not all_files_exist:
    print("\n⚠️  有文件缺失，请先完成以下操作：")
    print("\n1. 检查缺失的文件列表")
    print("2. 从Artifacts或导师代码中复制相应文件")
    print("3. 重新运行此检测脚本")
    
elif import_failed:
    print("\n⚠️  模块导入失败，请检查：")
    print("\n1. 文件内容是否完整（没有复制错误）")
    print("2. Python语法是否正确")
    print("3. 文件编码是否为UTF-8")
    
elif functional_tests_passed < functional_tests_total:
    print("\n⚠️  功能测试未完全通过，请检查：")
    print("\n1. 网络定义是否正确")
    print("2. 依赖是否都已安装")
    print("3. 查看上面的错误信息定位问题")
    
elif not deps_ok:
    print("\n⚠️  依赖缺失，请运行：")
    print("\n  pip install torch numpy sympy gymnasium tqdm")

print("\n" + "=" * 80)
