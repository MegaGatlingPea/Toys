# Flow Matching 完整实现指南

## 目录
1. [基本原理](#基本原理)
2. [数学基础](#数学基础)
3. [核心组件](#核心组件)
4. [La-Proteina案例分析](#la-proteina案例分析)
5. [简单实现示例](#简单实现示例)
6. [从高斯到玻尔兹曼分布的实现](#从高斯到玻尔兹曼分布的实现)

---

## 基本原理

### 🌊 Flow Matching 核心思想

Flow Matching是一种生成模型技术，通过学习从简单分布（如高斯噪声）到复杂分布（如数据分布）的**连续变换路径**。

```
简单分布 (t=0) ────flow───→ 复杂分布 (t=1)
    噪声                      目标数据
```

### 关键概念

1. **Flow**: 连续的概率分布变换过程
2. **Vector Field (速度场)**: 描述每个点在每个时刻应该如何移动
3. **Interpolation Path**: 从噪声到数据的具体路径
4. **Flow Matching**: 训练神经网络学习最优的速度场

---

## 数学基础

### 1. 基本微分方程

Flow Matching解决的核心问题是学习以下微分方程：

```
dx_t/dt = v_θ(x_t, t)
```

其中：
- `x_t`: 时刻t的状态
- `v_θ(x_t, t)`: 神经网络预测的速度场
- `t ∈ [0,1]`: 时间参数

### 2. 插值路径 (Interpolation Path)

最简单的线性插值：
```
x_t = (1-t) * x_0 + t * x_1
```

其中：
- `x_0`: 噪声样本 ~ N(0, I)
- `x_1`: 目标数据样本
- `t`: 插值参数

### 3. 目标速度场

理论上的最优速度场：
```
v*(x_t, t) = x_1 - x_0  # 对于线性插值
```

### 4. Flow Matching Loss

训练目标是让神经网络学习这个速度场：
```
L = E[||v_θ(x_t, t) - (x_1 - x_0)||²]
```

### 5. Score-based 扩展

可以添加随机性（SDE形式）：
```
dx_t = v(x_t, t) dt + g(t) s(x_t, t) dt + √(2g(t)) dw_t
```

其中：
- `s(x_t, t)`: Score函数（概率密度梯度）
- `g(t)`: 噪声强度调度
- `dw_t`: 布朗运动

**Score和Velocity的转换关系**：
```python
# Velocity → Score
s(x_t, t) = (t * v(x_t, t) - x_t) / (1 - t)

# Score → Velocity  
v(x_t, t) = (x_t + (1 - t) * s(x_t, t)) / t
```

---

## 核心组件

### 1. 数据加载器 (DataLoader)
```python
class FlowMatchingDataset:
    def __init__(self, data):
        self.data = data  # 目标分布的样本
    
    def __getitem__(self, idx):
        x_1 = self.data[idx]  # 目标样本
        x_0 = torch.randn_like(x_1)  # 噪声样本
        t = torch.rand(1)  # 随机时间
        x_t = (1-t) * x_0 + t * x_1  # 插值
        
        return {
            'x_0': x_0,
            'x_1': x_1, 
            'x_t': x_t,
            't': t,
            'target_v': x_1 - x_0  # 目标速度
        }
```

### 2. 神经网络 (Vector Field Network)
```python
class VectorFieldNetwork(nn.Module):
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x_t, t):
        # x_t: [batch, dim], t: [batch, 1]
        input_tensor = torch.cat([x_t, t], dim=-1)
        return self.net(input_tensor)
```

### 3. Flow Matcher (训练器)
```python
class FlowMatcher:
    def __init__(self, network, device='cpu'):
        self.network = network
        self.device = device
        
    def compute_loss(self, batch):
        x_t = batch['x_t'].to(self.device)
        t = batch['t'].to(self.device)
        target_v = batch['target_v'].to(self.device)
        
        # 网络预测
        pred_v = self.network(x_t, t)
        
        # Flow Matching Loss
        loss = F.mse_loss(pred_v, target_v)
        return loss
```

### 4. 采样器 (Sampler)
```python
class FlowSampler:
    def __init__(self, network, nsteps=100):
        self.network = network
        self.nsteps = nsteps
        
    def sample(self, n_samples, dim):
        # 初始噪声
        x = torch.randn(n_samples, dim)
        dt = 1.0 / self.nsteps
        
        # 数值积分
        for step in range(self.nsteps):
            t = torch.full((n_samples, 1), step * dt)
            v = self.network(x, t)
            x = x + v * dt  # 欧拉方法
            
        return x
```

### 5. 高级采样器 (Score-based)
```python
class AdvancedSampler:
    def __init__(self, network, nsteps=100, use_sde=True):
        self.network = network
        self.nsteps = nsteps
        self.use_sde = use_sde
        
    def vf_to_score(self, x_t, v, t):
        """Velocity Field → Score"""
        return (t * v - x_t) / (1 - t + 1e-5)
    
    def sample(self, n_samples, dim, noise_scale=0.1):
        x = torch.randn(n_samples, dim)
        dt = 1.0 / self.nsteps
        
        for step in range(self.nsteps):
            t_val = step / self.nsteps
            t = torch.full((n_samples, 1), t_val)
            
            # 网络预测
            v = self.network(x, t)
            
            if self.use_sde and t_val < 0.95:
                # SDE模式：添加噪声
                score = self.vf_to_score(x, v, t_val)
                gt = 1.0 / (t_val + 0.01)  # 噪声强度
                
                # 确定性项
                deterministic = v * dt
                
                # 随机项  
                noise = torch.randn_like(x)
                stochastic = torch.sqrt(2 * gt * noise_scale * dt) * noise
                
                x = x + deterministic + stochastic
            else:
                # ODE模式：纯确定性
                x = x + v * dt
                
        return x
```

---

## La-Proteina案例分析

### 多模态Flow Matching

La-Proteina使用**Product Space Flow Matching**，同时处理两个数据模态：

```python
# 两个独立的Flow
bb_ca_flow = RDNFlowMatcher(dim=3)      # 3D坐标
latent_flow = RDNFlowMatcher(dim=8)     # 8D Latent

# 不同的采样策略
sampling_args = {
    "bb_ca": {
        "schedule": {"mode": "log", "p": 2.0},
        "gt": {"mode": "1/t", "p": 1.0},
        "simulation_step_params": {
            "center_every_step": True   # 坐标需要质心居中
        }
    },
    "local_latents": {
        "schedule": {"mode": "power", "p": 2.0}, 
        "gt": {"mode": "tan", "p": 1.0},
        "simulation_step_params": {
            "center_every_step": False  # Latent不需要居中
        }
    }
}
```

### 时间调度策略

```python
def get_schedule(mode, nsteps, p):
    if mode == "uniform":
        return torch.linspace(0, 1, nsteps + 1)
    elif mode == "power":
        t = torch.linspace(0, 1, nsteps + 1)
        return t ** p
    elif mode == "log":
        t = 1.0 - torch.logspace(-p, 0, nsteps + 1).flip(0)
        return (t - t.min()) / (t.max() - t.min())
```

### 噪声强度调度

```python
def get_gt(t, mode, param):
    if mode == "1/t":
        return 1.0 / (t + 0.01)
    elif mode == "tan":
        return (torch.pi/2) * torch.tan((1-t) * torch.pi/2)
    elif mode == "1-t/t":
        return (1-t) / (t + 0.01)
```

---

## 简单实现示例

### 完整的2D示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class Simple2DFlowMatching:
    def __init__(self, hidden_dim=64):
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 2D + time
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, 2)   # 2D output
        )
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        
    def generate_data(self, n_samples=1000):
        """生成目标分布：两个高斯混合"""
        # 第一个高斯
        x1 = torch.randn(n_samples//2, 2) * 0.5 + torch.tensor([2.0, 2.0])
        # 第二个高斯  
        x2 = torch.randn(n_samples//2, 2) * 0.3 + torch.tensor([-2.0, -1.0])
        return torch.cat([x1, x2], dim=0)
        
    def create_batch(self, data, batch_size=128):
        """创建训练批次"""
        idx = torch.randint(0, len(data), (batch_size,))
        x_1 = data[idx]  # 目标样本
        x_0 = torch.randn_like(x_1)  # 噪声样本
        t = torch.rand(batch_size, 1)  # 随机时间
        
        # 线性插值
        x_t = (1 - t) * x_0 + t * x_1
        target_v = x_1 - x_0  # 目标速度场
        
        return x_t, t, target_v
    
    def train_step(self, x_t, t, target_v):
        """单步训练"""
        self.optimizer.zero_grad()
        
        # 网络预测
        input_tensor = torch.cat([x_t, t], dim=1)
        pred_v = self.network(input_tensor)
        
        # Flow Matching Loss
        loss = nn.MSELoss()(pred_v, target_v)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sample(self, n_samples=500, nsteps=100):
        """生成样本"""
        self.network.eval()
        with torch.no_grad():
            # 初始噪声
            x = torch.randn(n_samples, 2)
            dt = 1.0 / nsteps
            
            trajectory = [x.clone()]
            
            for step in range(nsteps):
                t = torch.full((n_samples, 1), step * dt)
                input_tensor = torch.cat([x, t], dim=1)
                v = self.network(input_tensor)
                x = x + v * dt
                trajectory.append(x.clone())
                
        return x, torch.stack(trajectory)
    
    def train(self, n_epochs=1000):
        """训练流程"""
        data = self.generate_data()
        losses = []
        
        for epoch in range(n_epochs):
            x_t, t, target_v = self.create_batch(data)
            loss = self.train_step(x_t, t, target_v)
            losses.append(loss)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
                
        return losses
    
    def visualize(self):
        """可视化结果"""
        # 原始数据
        data = self.generate_data()
        
        # 生成样本
        samples, trajectory = self.sample()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始数据分布
        axes[0].scatter(data[:, 0], data[:, 1], alpha=0.6, s=20)
        axes[0].set_title("Target Distribution")
        axes[0].grid(True)
        
        # 生成样本
        axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.6, s=20, color='red')
        axes[1].set_title("Generated Samples") 
        axes[1].grid(True)
        
        # 生成轨迹
        for i in range(0, len(trajectory), 10):
            traj = trajectory[i]
            axes[2].scatter(traj[:50, 0], traj[:50, 1], alpha=0.3, s=5)
        axes[2].set_title("Generation Trajectory")
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    fm = Simple2DFlowMatching()
    
    print("Training Flow Matching model...")
    losses = fm.train(n_epochs=2000)
    
    print("Generating samples...")
    fm.visualize()
```

---

## 从高斯到玻尔兹曼分布的实现

### 玻尔兹曼分布定义

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class BoltzmannDistribution:
    """多维玻尔兹曼分布"""
    def __init__(self, dim=2, temperature=1.0):
        self.dim = dim
        self.temperature = temperature
        
    def energy(self, x):
        """能量函数：E(x) = 0.5 * x^T A x + b^T x"""
        # 定义一个二次型能量函数
        A = torch.eye(self.dim) + 0.3 * torch.randn(self.dim, self.dim)
        A = A @ A.T  # 确保正定
        b = torch.randn(self.dim)
        
        energy = 0.5 * torch.sum(x @ A * x, dim=-1) + torch.sum(b * x, dim=-1)
        return energy
    
    def log_prob(self, x):
        """对数概率：log p(x) = -E(x)/T - log Z"""
        return -self.energy(x) / self.temperature
    
    def sample_metropolis(self, n_samples=1000, n_steps=1000):
        """Metropolis-Hastings采样（用于生成训练数据）"""
        samples = []
        x = torch.randn(self.dim)  # 初始状态
        
        for _ in range(n_steps):
            # 提议新状态
            x_new = x + torch.randn(self.dim) * 0.1
            
            # 接受概率
            log_alpha = self.log_prob(x_new) - self.log_prob(x)
            alpha = torch.exp(torch.clamp(log_alpha, max=0))
            
            # 接受或拒绝
            if torch.rand(1) < alpha:
                x = x_new
                
            samples.append(x.clone())
            
        return torch.stack(samples[-n_samples:])

class GaussianToBoltzmannFlow:
    """从高斯分布到玻尔兹曼分布的Flow Matching"""
    
    def __init__(self, dim=2, hidden_dim=128):
        self.dim = dim
        self.boltzmann = BoltzmannDistribution(dim)
        
        # 神经网络：输入(x, t)，输出速度场
        self.network = nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
        
    def generate_training_data(self, n_samples=2000):
        """生成训练数据"""
        print("Generating Boltzmann samples with MCMC...")
        return self.boltzmann.sample_metropolis(n_samples, n_steps=5000)
    
    def create_training_batch(self, target_data, batch_size=256):
        """创建训练批次"""
        # 随机选择目标样本
        idx = torch.randint(0, len(target_data), (batch_size,))
        x_1 = target_data[idx]  # 玻尔兹曼分布样本
        
        # 高斯噪声作为起点
        x_0 = torch.randn_like(x_1)
        
        # 随机时间
        t = torch.rand(batch_size, 1)
        
        # 线性插值
        x_t = (1 - t) * x_0 + t * x_1
        
        # 目标速度场
        target_v = x_1 - x_0
        
        return x_t, t, target_v
    
    def train_step(self, x_t, t, target_v):
        """单步训练"""
        self.optimizer.zero_grad()
        
        # 网络输入
        net_input = torch.cat([x_t, t], dim=1)
        pred_v = self.network(net_input)
        
        # Flow Matching Loss
        loss = nn.MSELoss()(pred_v, target_v)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def sample_flow(self, n_samples=1000, nsteps=100, method='ode'):
        """使用训练好的网络进行采样"""
        self.network.eval()
        
        with torch.no_grad():
            # 从标准高斯开始
            x = torch.randn(n_samples, self.dim)
            dt = 1.0 / nsteps
            
            trajectory = [x.clone()]
            
            for step in range(nsteps):
                t_val = step / nsteps
                t = torch.full((n_samples, 1), t_val)
                
                # 网络预测速度
                net_input = torch.cat([x, t], dim=1)
                v = self.network(net_input)
                
                if method == 'ode':
                    # 纯ODE积分
                    x = x + v * dt
                elif method == 'sde':
                    # 添加噪声的SDE积分
                    noise_scale = 0.1 * (1 - t_val)  # 随时间减小的噪声
                    noise = torch.randn_like(x) * noise_scale * torch.sqrt(dt)
                    x = x + v * dt + noise
                    
                trajectory.append(x.clone())
            
        return x, torch.stack(trajectory)
    
    def train(self, n_epochs=3000):
        """训练流程"""
        # 生成目标分布的训练数据
        target_data = self.generate_training_data()
        
        losses = []
        
        print("Training Flow Matching model...")
        for epoch in range(n_epochs):
            x_t, t, target_v = self.create_training_batch(target_data)
            loss = self.train_step(x_t, t, target_v)
            losses.append(loss)
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}/{n_epochs}, Loss: {loss:.6f}")
        
        return losses, target_data
    
    def evaluate(self, target_data):
        """评估和可视化"""
        # 生成样本
        flow_samples_ode, trajectory_ode = self.sample_flow(method='ode')
        flow_samples_sde, _ = self.sample_flow(method='sde')
        
        # 创建可视化
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 第一行：分布对比
        # 目标分布
        axes[0,0].scatter(target_data[:, 0], target_data[:, 1], 
                         alpha=0.6, s=20, c='blue', label='Boltzmann (MCMC)')
        axes[0,0].set_title("Target: Boltzmann Distribution")
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Flow生成（ODE）
        axes[0,1].scatter(flow_samples_ode[:, 0], flow_samples_ode[:, 1], 
                         alpha=0.6, s=20, c='red', label='Flow (ODE)')
        axes[0,1].set_title("Generated: Flow Matching (ODE)")
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Flow生成（SDE）
        axes[0,2].scatter(flow_samples_sde[:, 0], flow_samples_sde[:, 1], 
                         alpha=0.6, s=20, c='green', label='Flow (SDE)')
        axes[0,2].set_title("Generated: Flow Matching (SDE)")
        axes[0,2].legend()
        axes[0,2].grid(True)
        
        # 第二行：生成轨迹和能量分析
        # 生成轨迹
        for i in range(0, len(trajectory_ode), 20):
            traj = trajectory_ode[i]
            if i < 100:  # 只显示前100条轨迹
                axes[1,0].plot(traj[:, 0], traj[:, 1], alpha=0.3, linewidth=0.5)
        axes[1,0].set_title("Generation Trajectories")
        axes[1,0].grid(True)
        
        # 能量分布对比
        target_energy = self.boltzmann.energy(target_data)
        flow_energy_ode = self.boltzmann.energy(flow_samples_ode)
        flow_energy_sde = self.boltzmann.energy(flow_samples_sde)
        
        axes[1,1].hist(target_energy.numpy(), bins=50, alpha=0.5, 
                      label='Target', density=True)
        axes[1,1].hist(flow_energy_ode.numpy(), bins=50, alpha=0.5, 
                      label='Flow (ODE)', density=True)
        axes[1,1].hist(flow_energy_sde.numpy(), bins=50, alpha=0.5, 
                      label='Flow (SDE)', density=True)
        axes[1,1].set_title("Energy Distribution")
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # 二维密度对比
        from scipy.stats import gaussian_kde
        
        # 计算KDE密度
        target_kde = gaussian_kde(target_data.numpy().T)
        flow_kde = gaussian_kde(flow_samples_ode.numpy().T)
        
        # 创建网格
        x_range = np.linspace(-4, 4, 50)
        y_range = np.linspace(-4, 4, 50)
        X, Y = np.meshgrid(x_range, y_range)
        positions = np.vstack([X.ravel(), Y.ravel()])
        
        # 计算密度
        target_density = target_kde(positions).reshape(X.shape)
        flow_density = flow_kde(positions).reshape(X.shape)
        
        # 绘制密度差异
        density_diff = np.abs(target_density - flow_density)
        im = axes[1,2].contourf(X, Y, density_diff, levels=20, cmap='viridis')
        axes[1,2].set_title("Density Difference (|Target - Flow|)")
        plt.colorbar(im, ax=axes[1,2])
        
        plt.tight_layout()
        plt.show()
        
        # 打印统计信息
        print(f"\nEvaluation Results:")
        print(f"Target energy: mean={target_energy.mean():.3f}, std={target_energy.std():.3f}")
        print(f"Flow ODE energy: mean={flow_energy_ode.mean():.3f}, std={flow_energy_ode.std():.3f}")
        print(f"Flow SDE energy: mean={flow_energy_sde.mean():.3f}, std={flow_energy_sde.std():.3f}")

# 完整使用示例
def main():
    # 创建模型
    flow_model = GaussianToBoltzmannFlow(dim=2, hidden_dim=128)
    
    # 训练
    losses, target_data = flow_model.train(n_epochs=4000)
    
    # 绘制训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # 评估结果
    flow_model.evaluate(target_data)

if __name__ == "__main__":
    main()
```

---

## 实践建议

### 1. 从简单开始
- 先实现2D可视化版本
- 理解基本流程和概念
- 逐步增加复杂度

### 2. 关键调试点
- **损失下降**: Flow Matching Loss应该单调下降
- **轨迹可视化**: 观察生成轨迹是否合理
- **分布匹配**: 比较生成分布和目标分布

### 3. 常见问题
- **数值不稳定**: 使用较小的学习率和梯度裁剪
- **模式崩塌**: 增加网络容量或使用SDE采样
- **训练缓慢**: 使用更好的时间调度策略

### 4. 扩展方向
- **条件生成**: 添加条件信息到网络输入
- **多模态**: 像La-Proteina一样处理多个数据空间
- **更复杂的插值**: 使用非线性插值路径
- **最优传输**: 使用最优传输理论改进路径

---

## 参考资源

1. **原始论文**: Flow Matching for Generative Modeling
2. **实现参考**: La-Proteina代码库
3. **理论基础**: 最优传输和随机微分方程
4. **相关技术**: Score-based Models, Diffusion Models

这份文档应该能帮助你理解Flow Matching的完整实现，从理论到实践都有涵盖！
