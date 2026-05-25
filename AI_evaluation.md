# PHYSMOL：基于物理同构与符号绑定的具身概念学习系统架构评估、论证与工程优化研究报告

具身智能（Embodied AI）正处于从传统的基于大规模统计映射的"黑盒"模型向具备物理理解力与符号逻辑推理能力的"白盒"架构转型的关键节点。当前的通用大语言模型（LLM）虽然在文本语义关联方面表现出色，但在处理真实物理世界的具身交互时，往往由于缺乏"物理直觉"而产生不可忽视的幻觉（Hallucination），导致其提出的行动建议在现实物理定律约束下难以执行 1。PHYSMOL（Physical Isomorphism and Symbolic Binding）项目正是针对这一核心痛点提出的创新架构，旨在打造一个无需海量被动数据、通过主动物理探索与语言同步、从零开始形成"概念理解"的通用智能系统 3。

本报告将从架构的合理性评价、实现可行性分析以及系统优化完善三个维度，对 PHYSMOL 这一融合了向量符号架构（VSA）、拉格朗日神经网络（LNN）与脉冲神经网络（SNN）的复杂具身智能模型进行详尽的专业论证。

## PHYSMOL 架构设计的合理性深度评价

PHYSMOL 的核心设计理念在于模拟人类婴儿的认知发育过程，即通过主动感知与操纵（Active Perception and Manipulation）来构建关于物体的三维几何、动力学特性及跨感官统一的内在表示 3。这种设计在认知科学与计算神经科学层面具有极高的合理性。它不再将物体视为像素簇，而是将其定义为具有质量（Mass）、弹性（Elasticity）、转动惯量（Moment of Inertia）等本质属性的动态实体 3。

### 向量符号架构（VSA）作为认知底座的科学性

PHYSMOL 选择 10,000 维的超维向量空间作为概念表示的底座，利用了高维空间的"维度诅咒"的反面效应——即"维度祝福"。在高维空间中，随机生成的向量几乎是以指数级概率相互正交的，这为表示海量独立属性基元（Primitives）提供了近乎无限的准正交（Quasi-orthogonal）容量 6。这种分布式表示（Distributed Representation）具有极强的抗噪性：即使向量中的部分维度遭到破坏，其语义核心依然能够通过余弦相似度或汉明距离准确检索 8。

VSA 的合理性还体现在其代数运算上。通过绑定（Binding, ）和捆绑（Bundling, ）操作，PHYSMOL 能够以"一次性学习"（One-shot Learning）的方式构建复杂概念。例如，将"球"的形状向量与"红色"的颜色向量绑定，即可生成一个全新的、且能反向解析的"红球"复合向量 10。这种逻辑运算能力解决了传统神经网络难以处理的结构化关系问题，使系统具备了类似于符号 AI 的组合组合爆炸处理能力，同时保持了神经表示的连续性和容错性 8。

### 物理同构与拉格朗日约束的必然性

PHYSMOL 架构中引入神经物理编码器（Invariant LNN）是对当前纯数据驱动动力学模型的重大修正。传统的神经网络在预测物体运动轨迹时，由于缺乏内在的能量守恒约束，往往会出现随时间累积的漂移错误 13。通过将拉格朗日力学硬编码进网络架构，PHYSMOL 确保了其物理预测在数学上遵循欧拉-拉格朗日方程：

其中  代表动能与势能之差 14。这种"物理信息机器学习"（Physics-Informed Machine Learning）的设计使得系统能够仅凭极少量的交互数据（轨迹序列），即可从观测中蒸馏出物体的本质物理参数  5。这对于需要在动态、不可预测环境下运行的机器人而言，是确保安全性和泛化能力的基石。

### 脉冲因果图与 STDP 的生物合理性

PHYSMOL 使用基于脉冲神经网络（SNN）的因果图来处理事件间的时序关系，这高度契合生物脑的运作机制。通过脉冲时间依赖可塑性（STDP）规则，系统能够自发地将"前因"事件的脉冲与"后果"事件的脉冲关联起来 18。这种事件驱动（Event-driven）的计算模式比传统的连续时间序列处理更为高效，能够自然地捕捉到物理交互中的不连续性（如碰撞瞬间） 20。

## 系统实现的可行性论证

PHYSMOL 的工程实现不仅在理论上可行，而且在现有的软硬件生态系统中已有深厚的技术支撑。

### 物理仿真与多模态数据管道

使用 JAX 驱动的 Brax 或 MuJoCo 作为物理经验场，利用了现代 GPU 的大规模并行加速能力，可以同时运行数千个虚拟环境实例，从而在短时间内积累数百万次的交互经验 3。Brax 对可微分仿真的支持，使得感知编码器能够接收来自物理状态的直接梯度反馈，加速了从像素到物理参数的端到端学习过程 24。

| 仿真引擎 | 硬件加速 | 特色功能 | 适用场景 |
|----------|----------|----------|----------|
| Brax | JAX / GPU | 极致并行，千万级步数/秒 | 大规模强化学习探索 23 |
| MuJoCo | CPU / GPU | 复杂的接触动力学，高保真摩擦力模型 | 精细操纵与抓取实验 22 |
| Isaac Sim | NVIDIA GPU | 光线追踪渲染，真实的视觉反馈 | 视觉-物理联合对齐 25 |

### 超维向量空间（HDC/VSA）的计算效率

关于 VSA 的检索瓶颈问题，现有的近似最近邻搜索算法（如 HNSW 和 PQ）已能支持在千万级向量库中实现亚毫秒级的检索响应 3。研究表明，MAP（Multiply-Add-Permute）模型在深度学习环境中的运行速度比传统的 HRR 快 3-4 倍，且通过线性读出（Linear Readout）可以进一步消除搜索延迟 27。此外，FPGA 和类脑芯片（如 Intel Loihi）已经证明了直接在硬件层级加速 VSA 绑定运算的可行性，大幅降低了具身智能终端的功耗 9。

### 双螺旋训练方案与语言对齐

双螺旋（Dual-Helix）并行训练方案有效解决了"数据饥渴"问题。物理螺旋负责生成具体的感官经验，而语言螺旋负责提取通用的语义模式 3。通过"对齐中枢"（Alignment Hub），系统周期性地将词向量映射到 VSA 概念向量上，这种机制类似于人类幼年时期的"指物认名"过程 3。

在具体的工程实施中，这种对齐可以通过对比学习（Contrastive Learning）损失函数来实现：

通过调节温度参数 ，可以精细化控制物理概念与语言标签之间的绑定强度。这种方法在 CLIP 等跨模态模型中已经得到了验证，但在 PHYSMOL 中，这种对齐是在更具解释性的 VSA 空间进行的，避免了嵌入空间塌陷（Embedding Collapse）的风险 10。

## 针对 PHYSMOL 架构的完善与优化建议

尽管 PHYSMOL 的原始设计已经非常完备，但在深入分析其潜在的工程挑战后，本报告提出以下针对性的完善建议，以提升其在复杂任务中的鲁棒性与泛化能力。

### 引入拉格朗日图神经网络（LGnn）增强可扩展性

原始架构中的不变拉格朗日神经网络（Invariant LNN）在处理单一物体时表现良好，但在处理由多个部分组成的复杂物体（如链条、弹簧连接体）时，其参数效率会迅速下降 33。 建议：将 LNN 升级为拉格朗日图神经网络（LGnn）。在 LGnn 中，复杂的物理系统被表示为一个图 ，其中节点代表质点或刚体，边代表连接约束 5。

节点级预测： 每个节点学习自身的动能函数和局部物理属性（如质量 ）。

边缘级预测： 每条边学习势能函数（如弹性势能 ）。 这种图结构的归纳偏置（Inductive Bias）使得 PHYSMOL 能够实现"跨拓扑泛化"：在一个 3 节链条上训练的模型，可以直接应用于 5 节链条的运动预测，而无需重新训练 5。

### 针对非交换性变换的 VSA 算子改良

传统的 VSA 绑定算子（如 XOR 或元素乘法）通常满足交换律 。然而，在真实的物理空间中，动作的顺序和空间方位具有非交换性（Non-commutative）。例如，"先旋转 90 度再平移"与"先平移再旋转"会导致完全不同的最终位姿 35。 建议：引入分层谐振器网络（Hierarchical Resonator Network, HRN）和分数幂编码（Fractional Power Encoding, FPE）。

FPE 实现： 利用随机频率的复数域变换，将空间坐标  映射为具有连续漂移特性的 hypervectors。

HRN 实现： 将系统分解为笛卡尔参考系和对数极坐标系两个部分，通过矩阵变换进行通信。这使得 VSA 能够准确编码旋转、缩放平移等非交换性变换，并支持复杂的位姿推断 36。

### 三因子学习规则优化长链因果推理

现有的脉冲因果图依赖于简单的 STDP 规则，这在处理即时反馈的物理事件（如撞击）时非常有效，但在处理具有延迟效应的长链因果（如"打开阀门后 10 秒水桶注满"）时，会出现严重的信用分配（Credit Assignment）问题 37。 建议：在 STDP 的基础上引入第三个因子——全局神经调节信号（如模拟多巴胺的奖励信号  或显著性信号）。 新的突触更新规则可表示为：

通过引入这种奖励调测机制，因果图可以过滤掉无关的随机共现事件，仅保留对任务成功或信息增益有显著贡献的因果边。这能显著降低因果图的稀疏度，提高反事实推理的计算效率 37。

### 引入分层世界模型（L1-L2-L3）

具身智能系统需要在不同时间尺度上进行决策。PHYSMOL 需要在保持底层反射式控制（毫秒级）的同时，进行高层战略规划（秒至分钟级） 40。 建议：将 PHYSMOL 的决策层划分为三个层级：

L1（预测器层）： 负责快速、感知的单步预测。此层直接由感知编码器和多模态策略网络驱动，处理如平衡控制、避障等反射性动作 41。

L2（模拟器层）： 当 L1 产生较高的好奇心信号（预测误差）时激活。利用脉冲因果图进行多步前向回溯，运行"虚拟实验"以选择最优行动序列 41。

L3（进化器层）： 专门处理结构性失效。当 L2 的模拟反复失败时，L3 负责触发知识库的结构性更新，如修改 VSA 对象配方或重新校准 LNN 的物理常数 41。

## PHYSMOL 的存储分层与性能加速策略

在 10,000 维的高维空间进行持续学习，内存管理和计算效率是工程化的最大挑战。基于 snippets 中的研究，我们建议实施以下存储和检索优化方案。

### 阶梯式内存布局（Tiered Memory Management）

为了解决"内存墙"瓶颈，PHYSMOL 应当将计算重心放在数据局部性上，利用不同的存储介质处理不同生命周期的知识 3。

| 存储层级 | 技术选型 | 读写特性 | 承载内容 |
|----------|----------|----------|----------|
| 热存储 (L1) | GPU HBM / SRAM | 超高带宽，极低延迟 | 当前活跃节点的 VSA 向量、脉冲轨迹记录 29 |
| 温存储 (L2) | DDR5 / STT-MRAM | 中等带宽，容量较大 | 属性基元库（Codebook）、高频使用的对象配方 3 |
| 冷存储 (L3) | NVMe SSD / 块存储 | 高延迟，海量容量 | 全量历史经验轨迹、冷僻对象向量及符号索引 46 |

### 向量量化与检索加速

针对 VSA 概念检索，应当采用 TurboQuant 等在线向量量化算法。通过随机旋转输入向量，诱导出集中分布，然后在各个坐标上应用标量量化器，可以将 KV 缓存压缩至每个通道 3.5 位，同时保持绝对的质量中性 48。在检索阶段，结合 HNSW 算法并根据局部内在维度（Local Intrinsic Dimensionality, LID）动态调整插入顺序，可以将检索召回率提升 12 个百分点，确保系统在复杂场景下的实时响应 26。

## 验证与评估体系的重构

传统的感知基准测试（如 ImageNet 分类）无法衡量 PHYSMOL 的核心价值。我们需要一套能够量化"物理直觉"和"因果深度"的评估指标。

### 核心评估维度与度量标准

| 评估项目 | 评估方法 | 关键度量标准 (KPI) |
|----------|----------|-------------------|
| 跨模态联想 | 给定"沉闷的碰撞声"，要求系统从冷存储中检索匹配的物体属性向量。 | Top-1 & Top-5 检索召回率 (Recall@k) 3 |
| 零样本轨迹预测 | 展示一个新的斜坡交互场景，预测不同形状物体（如从未见过的五角星块）的下滑轨迹。 | 预测加速度与仿真真实值的 MSE；能量守恒违规率 5 |
| 反事实逻辑验证 | 提问："如果将物体的质量增加一倍，它落地的速度会发生什么变化？" | 脉冲因果图中效应节点的发放强度偏移量 2 |
| 好奇心收敛效率 | 在新环境中，系统通过主动交互消除感知不确定性所需的最少动作数。 | 信息增益曲线的导数（学习进步率） 50 |

### 里程碑式的认知跨越评估

M1（物体恒存性）： 在物体被遮挡物（Occluder）挡住时，系统是否能维持其 VSA 向量的激活状态，并通过 LNN 预测其在遮挡后的轨迹。若系统检测到不连续性（Discontinuity），应触发 Curiosity Block 进行重新扫描 4。

M2（物理定律抽象）： 系统是否能自发从 46 个力学实验数据中，利用符号回归（Symbolic Regression）独立发现  或动量守恒定律，并以物理 DSL（领域专用语言）形式存储 52。

M3（复杂任务规划）： 在机器人抓取任务中，系统是否能根据物体的重心偏移向量（由触觉 VSA 和 LNN 联合推断），动态调整抓取力矩，实现从仿真到真实机器人的零样本迁移 25。

## 风险识别与应对策略

PHYSMOL 作为一个前沿架构，依然面临着若干工程和理论上的风险，需要提前布局。

### LNN 的参数推断漂移风险

当交互轨迹存在高噪声或数据极度稀缺时，Invariant LNN 可能会推断出错误的质量或弹性参数，导致预测失效 16。

对策： 引入"混合优化路径"，交替使用 L-BFGS（快速定位局部极小值）和 Adam（逃离鞍点）进行训练 56。同时，在训练早期引入"特权信息"（Privileged Information），将仿真器的真实物理状态作为监督信号，逐步过渡到纯视觉推断。

### 语言螺旋的语义偏移风险

预训练词向量的语义空间可能与物理 VSA 空间不完全对齐，导致"苹果"对应的词向量偏离了物理上"红色、圆形、有质量"的物体向量 10。

对策： 采用能力增强元模型（Capability-enhanced Meta-model），通过轻量级的正交正则化损失，确保词向量投影层在映射过程中保持语义拓扑结构的一致性 12。

### 长时间运行的内存碎片与垃圾回收

持续的 STDP 学习和 VSA 概念生成会导致温存储（L2）迅速填满，产生大量无效的"幽灵概念"向量 3。

对策： 实施"突触剪枝"机制，定期删除信用分数（Credit Score）低于阈值的因果边和长期未被激活的对象配方，确保系统的轻量化运行 58。

## 结论：通往真正具身 AGI 的必然路径

PHYSMOL 架构设计的合理性得到了认知发展心理学和物理信息机器学习理论的双重支持。它通过 VSA 解决了符号操作与向量表示的鸿沟，通过 LNN 实现了对物理法则的内化，通过 SNN 捕获了物理因果的时序本质，最后通过主动好奇心驱动系统进行自主进化 3。

从实施角度看，随着 Brax、MuJoCo 等高性能模拟器的成熟，以及类脑硬件对脉冲计算和超维运算支持的加强，PHYSMOL 的技术栈已经闭环 9。通过引入本报告提出的 LGnn 图结构优化、非交换性 FPE 编码 以及 三因子学习规则，该架构将具备极强的开放泛化能力。

PHYSMOL 不仅仅是一个模型，它代表了人工智能从"数据驱动"向"知识-物理联合驱动"的范式转移。一旦 M5 阶段（集成 LLM）成功完成，我们将拥有一个具备真实世界物理常识的智能体，它能够"先思而后行"，在复杂多变的现实环境中展现出人类水平的认知灵活性和行动可靠性 2。

#### 引用的著作

1. Neuro-symbolic AI - Wikipedia, 访问时间为 五月 20, 2026， https://en.wikipedia.org/wiki/Neuro-symbolic_AI
2. Browse Preprints - Authorea, 访问时间为 五月 20, 2026， https://faye.authorea.com/browse-all?tags=%5B%22neuro-symbolic+ai%22%5D
3. PHYSMOL.docx
4. Curiosity-driven Intuitive Physics Learning - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/pdf/2105.07426
5. (PDF) Learning the Dynamics of Particle-based Systems with Lagrangian Graph Neural Networks - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/366873656_Learning_the_Dynamics_of_Particle-based_Systems_with_Lagrangian_Graph_Neural_Networks
6. Learning Vector Symbolic Architectures | Research | Automation Technology - TU Chemnitz, 访问时间为 五月 20, 2026， https://www.tu-chemnitz.de/etit/proaut/en/research/vsa.html
7. Vector Symbolic Architectures as a Computing Framework for Emerging Hardware - PMC, 访问时间为 五月 20, 2026， https://pmc.ncbi.nlm.nih.gov/articles/PMC10588678/
8. A comparison of vector symbolic architectures, 访问时间为 五月 20, 2026， https://d-nb.info/1252299222/34
9. Cross-Layer Design of Vector-Symbolic Computing: Bridging Cognition and Brain-Inspired Hardware Acceleration - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2508.14245v1
10. Designing Vector-Symbolic Architectures for Biomedical Applications: Ten Tips and Common Pitfalls - Preprints.org, 访问时间为 五月 20, 2026， https://www.preprints.org/manuscript/202510.0117/v1
11. Designing Vector-Symbolic Architectures for Biomedical Applications: Ten Tips and Common Pitfalls - Preprints.org, 访问时间为 五月 20, 2026， https://www.preprints.org/manuscript/202510.0117/v1/download
12. Why concepts are (probably) vectors - colala, 访问时间为 五月 20, 2026， https://colala.berkeley.edu/papers/piantadosi2024why.pdf
13. From Data to Physics: Physics-Informed Machine Learning Frameworks in Interdisciplinary Applications - MDPI, 访问时间为 五月 20, 2026， https://www.mdpi.com/2673-8716/6/2/16
14. Unsupervised Learning of Lagrangian Dynamics from Images for Prediction and Control, 访问时间为 五月 20, 2026， https://proceedings.neurips.cc/paper/2020/file/79f56e5e3e0e999b3c139f225838d41f-Paper.pdf
15. Lagrangian Neural Networks - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/339840846_Lagrangian_Neural_Networks
16. Lagrangian neural ODEs: Measuring the existence of a Lagrangian with Helmholtz metrics - Machine Learning and the Physical Sciences, 访问时间为 五月 20, 2026， https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_69.pdf
17. Neural Force Field: Few-shot Learning of Generalized Physical Reasoning - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2502.08987v6
18. (PDF) Neuromorphic Computing and Spiking Neural Networks: Bridging Neuroscience with AI Hardware - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/389992461_Neuromorphic_Computing_and_Spiking_Neural_Networks_Bridging_Neuroscience_with_AI_Hardware
19. Models For Neural Spike Computation And Cognition, 访问时间为 五月 20, 2026， https://lan-portal.uob.edu.ly/list/PLAY/174506DI57/models__for-neural_spike-computation-and-cognition.pdf
20. Integrating Causality with Neurochaos Learning: Proposed Approach and Research Agenda - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/pdf/2501.13763
21. The two dragons of cognition: recursive condensation for predictive processing - PMC, 访问时间为 五月 20, 2026， https://pmc.ncbi.nlm.nih.gov/articles/PMC13050847/
22. Where-to-Learn: Analytical Policy Gradient Directed Exploration for On-Policy Robotic Reinforcement Learning | Request PDF - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/403192634_Where-to-Learn_Analytical_Policy_Gradient_Directed_Exploration_for_On-Policy_Robotic_Reinforcement_Learning
23. Flow Matching Policy Gradients - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2507.21053v2
24. gradSim: Differentiable simulation for system identification and visuomotor control, 访问时间为 五月 20, 2026， https://gradsim.github.io/
25. Sim-to-Real and Real-to-Sim: The Engine Behind Capable Physical AI - AWS, 访问时间为 五月 20, 2026， https://aws.amazon.com/blogs/physical-ai/sim-to-real-and-real-to-sim-the-engine-behind-capable-physical-ai/
26. Daily Papers - Hugging Face, 访问时间为 五月 20, 2026， https://huggingface.co/papers?q=minor%20singular%20vectors
27. Practical Lessons on Vector-Symbolic Architectures in Deep Learning-Inspired Environments, 访问时间为 五月 20, 2026， https://proceedings.mlr.press/v284/carzaniga25a.html
28. Paper Abstracts – ASPLOS 2024, 访问时间为 五月 20, 2026， https://www.asplos-conference.org/asplos2024/main-program/abstracts/index.html
29. (PDF) Overmind NSA: A Unified Neuro-Symbolic Computing Architecture with Approximate Nonlinear Activations and Preemptive Memory Bypass - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/403976658_Overmind_NSA_A_Unified_Neuro-Symbolic_Computing_Architecture_with_Approximate_Nonlinear_Activations_and_Preemptive_Memory_Bypass
30. A Dual-Helix Governance Approach Towards Reliable Agentic AI for WebGIS Development, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/401564627_A_Dual-Helix_Governance_Approach_Towards_Reliable_Agentic_AI_for_WebGIS_Development
31. Bridging the Gap: Representation Spaces in Neuro-Symbolic AI - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2411.04393v1
32. A review of embodied intelligence systems: a three-layer framework integrating multimodal perception, world modeling, and structured strategies - PMC, 访问时间为 五月 20, 2026， https://pmc.ncbi.nlm.nih.gov/articles/PMC12631203/
33. Unravelling the Performance of Physics-informed Graph Neural Networks for Dynamical Systems, 访问时间为 五月 20, 2026， https://proceedings.neurips.cc/paper_files/paper/2022/file/17b598fda495256bef6785c2b76c3217-Paper-Datasets_and_Benchmarks.pdf
34. A Riemannian Framework for Learning Reduced-order Lagrangian Dynamics - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2410.18868v1
35. [2202.04771] Orthogonal Matrices for MBAT Vector Symbolic Architectures, and a "Soft" VSA Representation for JSON - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/abs/2202.04771
36. Visual Odometry with Neuromorphic Resonator Networks - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2209.02000v3
37. Three-factor learning in spiking neural networks: An overview of methods and trends from a machine learning perspective - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/397495896_Three-factor_learning_in_spiking_neural_networks_An_overview_of_methods_and_trends_from_a_machine_learning_perspective
38. Addressing Challenges of Spiking Neural Networks in Causal Reinforcement Learning, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/389515954_Addressing_Challenges_of_Spiking_Neural_Networks_in_Causal_Reinforcement_Learning
39. Daily Papers - Hugging Face, 访问时间为 五月 20, 2026， https://huggingface.co/papers?q=global%20sign%20vector
40. The Dual Pillars of Embodied Autonomy: A Technical Deep Dive into Language-Action Models and Vision-Based Control Architectures | by Neel Shah | Towards AI, 访问时间为 五月 20, 2026， https://pub.towardsai.net/the-dual-pillars-of-embodied-autonomy-a-technical-deep-dive-into-language-action-models-and-25d7d24a3d5d
41. Agentic World Modeling: Foundations, Capabilities, Laws, and Beyond - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2604.22748v1
42. Tiered cloud storage via two-stage, latency-aware bidding - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/pdf/1705.02745
43. Improving Neural Network Efficiency via Post-training Quantization with Adaptive Floating-Point | Request PDF - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/358994216_Improving_Neural_Network_Efficiency_via_Post-training_Quantization_with_Adaptive_Floating-Point
44. Paper Title (use style - IEOM Society, 访问时间为 五月 20, 2026， https://ieomsociety.org/proceedings/2024northamerica/285.pdf
45. 2023 Asia and South Pacific Design Automation Conference (ASPDAC) Table of Content, 访问时间为 五月 20, 2026， https://www.sigda.org/publications/aspdac23-toc/
46. InftyDedup: Scalable and Cost-Effective Cloud Tiering with Deduplication - mimuw, 访问时间为 五月 20, 2026， https://www.mimuw.edu.pl/~iwanicki/publications/2023/FAST/kotlarska-FAST2023.pdf
47. Lindorm-UWC: An Ultra-Wide-Column Database for Internet of Vehicles, 访问时间为 五月 20, 2026， https://netman.aiops.org/wp-content/uploads/2025/05/p1920-zheng.pdf
48. cycloarcane/Gathered-Paper-Resources - GitHub, 访问时间为 五月 20, 2026， https://github.com/cycloarcane/Gathered-Paper-Resources
49. (PDF) Causal Inference in Machine Learning: From Counterfactuals to Real-World Decision Systems - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/390032798_Causal_Inference_in_Machine_Learning_From_Counterfactuals_to_Real-World_Decision_Systems
50. Active Learning and Artificial Curiosity in Robots - Pierre-Yves Oudeyer, 访问时间为 五月 20, 2026， https://www.pyoudeyer.com/active-learning-and-artificial-curiosity-in-robots/
51. Active World Model Learning with Progress Curiosity - Harvard Business School, 访问时间为 五月 20, 2026， https://www.hbs.edu/ris/Publication%20Files/Active%20World%20Model%20Learning%20with%20Progress%20Curiosity_a00dbe0e-4834-4935-bc62-63729c93af6b.pdf
52. arxiv.org, 访问时间为 五月 20, 2026， https://arxiv.org/html/2504.01538v2
53. AI-Newton: Autonomous Law Discovery System | PDF - Scribd, 访问时间为 五月 20, 2026， https://www.scribd.com/document/960358697/2504-01538v1
54. PKU research team makes AI system that autonomously detects scientific laws - EurekAlert!, 访问时间为 五月 20, 2026， https://www.eurekalert.org/news-releases/1107868
55. Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2502.20396v1
56. NN-OpInf: an operator inference approach using structure-preserving composable neural networks - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2603.08488v1
57. Daily Papers - Hugging Face, 访问时间为 五月 20, 2026， https://huggingface.co/papers?q=capability%20vectors
58. Agent Harness for Large Language Model Agents: A Survey[v1] | Preprints.org, 访问时间为 五月 20, 2026， https://www.preprints.org/manuscript/202604.0428/v1?ref=observability.how
59. Memory Errors in Modern Systems: The Good, The Bad, and The Ugly - ResearchGate, 访问时间为 五月 20, 2026， https://www.researchgate.net/publication/277917894_Memory_Errors_in_Modern_Systems_The_Good_The_Bad_and_The_Ugly
60. Technical Framework for Building an AGI - Hugging Face, 访问时间为 五月 20, 2026， https://huggingface.co/blog/davehusk/technical-framework-for-building-an-agi
61. Converging Paradigms: The Synergy of Symbolic and Connectionist AI in LLM-Empowered Autonomous Agents - arXiv, 访问时间为 五月 20, 2026， https://arxiv.org/html/2407.08516v5
62. Artificial intelligence as a surrogate brain: bridging neural dynamical models and data - PMC, 访问时间为 五月 20, 2026， https://pmc.ncbi.nlm.nih.gov/articles/PMC12866659/
