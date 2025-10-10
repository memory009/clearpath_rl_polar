class TD3Config:
    """TD3训练配置"""
    # 环境参数
    robot_name = 'j100_0000'
    goal_pos = (2.0, 2.0)
    max_steps = 256
    collision_threshold = 0.3
    
    # 网络参数（关键！与论文一致）
    state_dim = 12          # [距离, 角度, 8激光, 2速度]
    action_dim = 2          # [线速度, 角速度]
    max_action = 0.5        # 最大线速度
    hidden_dim = 26         # 隐藏层神经元数（论文使用26）
    
    # 训练参数
    total_timesteps = 100000
    start_timesteps = 25000  # 随机探索步数
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    
    # TD3特定参数
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2          # Actor延迟更新频率
    expl_noise = 0.1         # 探索噪声
    
    # POLAR验证参数
    observation_error = 0.01  # 观测误差范围
    verify_interval = 1000    # 验证间隔
    bern_order = 1           # Bernstein多项式阶数
    error_steps = 4000       # 误差采样步数
    
    # 保存路径
    model_path = './models/'
    log_path = './logs/'
    result_path = './results/'
    
    # 设备
    device = 'cpu'  # 论文使用CPU以避免CUDA警告
