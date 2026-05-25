# PHYSMOL 开发日志

## 2026-05-21 ~ 2026-05-22：C+Python 混合架构实现

### 背景

原始 PHYSMOL 论文是纯理论架构，无源码。目标硬件为魔塔 AMD 云服务器（23核/200GB RAM/192GB VRAM）。

### 关键决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 语言 | C + Python 混合 | SNN 脉冲传播在 Python 中极慢，C 实现提速 50-100x |
| GPU 框架 | HIP/ROCm | 服务器是 AMD GPU，不支持 CUDA |
| VSA 维度 | 4,096 | 准正交性足够（σ≈0.016），内存省 60% |
| SNN 状态 | 位压缩 uint64 | 4096 神经元 = 512 字节 |
| 训练方案 | 串行两阶段 | 23 核 CPU 是瓶颈，GPU 被闲置；串行方案 CPU/GPU 分别满载 |
| 仿真引擎 | MuJoCo | CPU 核数少时比 Brax(JAX) 更高效 |

### 训练方案变更：双螺旋 → 串行两阶段

**原方案（双螺旋）问题：** 物理螺旋是 CPU-bound 的 MuJoCo 仿真，语言螺旋是 GPU-bound 的矩阵运算，两者并行时互相等待。

**新方案：**
```
Phase 1（物理）: CPU 全力跑 MuJoCo 仿真 + LNN 学习 + SNN 因果图
     ↓ 存储所有物体概念向量
Phase 2（语言）: GPU 全力跑 InfoNCE 语言对齐
```

### 多模态感知设计

MuJoCo 只提供物理量，其他模态的来源：

| 模态 | 数据来源 | 编码方式 |
|------|----------|----------|
| 视觉/颜色 | MuJoCo 渲染或 geom 属性 | HSV 直方图 + 形状描述子 → 稀疏随机投影 |
| 听觉 | 碰撞力 + 材质硬度 + 物体大小 | 从力学量合成声学特征（共振频率、亮度、衰减） |
| 触觉 | 接触力 + 位置 + 法线 | 力分解（法向/切向）+ 滑移比 |
| 嗅觉 | 材质类型查表 | 8维气味轮廓（刺激性、金属感、木质、化学、泥土、花香、烟熏、矿物） |
| 本体感觉 | 关节位置/速度 | 直接编码 |

每个模态独立编码成 VSA 向量，然后捆绑形成多模态概念。

### VSA 概念存储修正

**错误设计：** 直接存整个多模态感知向量作为物体向量。

**正确设计（论文原意）：**
- Codebook 存原子基元：`"红色" → 随机向量`, `"球形" → 随机向量`
- 物体 = 带标签基元的捆绑：`红球 = (color_tag ⊗ red) + (shape_tag ⊗ sphere) + ...`
- 分解 = 解绑标签 + 最近邻搜索：`红球 ⊗ color_tag → 找最近颜色 → "红色"`

### 力场/光场等连续场的表示

场 = FPE 位置编码 ⊗ 场值基元 的空间采样叠加：

```
重力场 = Σ [ H(x,y,z) ⊗ v_gravity_9.8 ]  （均匀场）
光场   = Σ [ H(x,y,z) ⊗ v_brightness_i ]  （近亮远暗）
```

场与物体的区别：
- 物体 = 离散基元捆绑（红色+球形+重...）
- 场 = 空间连续 FPE 采样叠加（位置⊗值 + 位置⊗值 + ...）

### 文件结构

```
src/
├── core/                    # C 核心引擎（SIMD 优化）
│   ├── vsa.c/h             # VSA 绑定/捆绑/FPE
│   ├── snn.c/h             # SNN LIF 脉冲 + STDP + 三因子学习
│   ├── causal.c/h          # 因果图（邻接表）
│   ├── lnn.c/h             # LNN CPU fallback（欧拉-拉格朗日求解）
│   └── memory.c/h          # 阶梯式内存（L1池/L2 hashmap/L3文件）
├── hip/                     # HIP/ROCm GPU 内核模板
│   └── lnn_hip.hip
├── bindings/                # pybind11 绑定（numpy 零拷贝）
│   ├── vsa_py.cpp
│   ├── snn_py.cpp
│   └── lnn_py.cpp
└── python/physmol/          # Python 控制层
    ├── vsa.py              # VSA 封装
    ├── snn.py              # SNN 封装
    ├── lnn.py              # LNN（numpy fallback，无需 C 扩展）
    ├── perception.py       # 多模态感知编码器（5种模态）
    ├── vsa_concepts.py     # 基元 Codebook + 场编码 + 概念组合/分解
    ├── alignment.py        # InfoNCE 语言对齐
    ├── world_model.py      # L1/L2/L3 分层世界模型
    ├── train.py            # 串行两阶段训练主循环
    └── sim_env.py          # MuJoCo 仿真环境（含多模态采集）

tests/
├── test_vsa.c / .py        # VSA 测试
├── test_snn.c / .py        # SNN 测试
└── test_lnn.py             # LNN 测试

config/default.yaml         # 默认配置（4096维）
CMakeLists.txt / setup.py / Makefile  # 构建系统
```

### 硬件资源评估

| 组件 | 内存占用 | 计算瓶颈 |
|------|----------|----------|
| VSA codebook (1万向量) | ~160 MB | 无 |
| SNN 权重 (512×512) | ~1 MB | 无 |
| LNN 参数 | ~5 MB | 无 |
| MuJoCo 4并行环境 | ~100 MB | **CPU（23核）** |
| 词向量 | ~120 MB | 无 |
| PyTorch ROCm | ~2 GB | 无 |
| **总计** | **~13 GB / 200 GB** | **CPU 是唯一瓶颈** |

Phase 1 估算：1000 episodes × 500 steps × 5ms ≈ 42 分钟
Phase 2 估算：50 epochs × 26 词对 ≈ 几秒

### 本地测试状态

| 组件 | 本地（无C扩展） | 云端（有gcc） |
|------|----------------|--------------|
| LNN | numpy fallback 可用 | C 加速可用 |
| 多模态感知 | 纯 numpy 可用 | 同左 |
| 概念系统 | 纯 numpy 可用 | 同左 |
| 世界模型 | 纯 Python 可用 | 同左 |
| VSA/SNN | 不可用（需编译） | C 加速可用 |
