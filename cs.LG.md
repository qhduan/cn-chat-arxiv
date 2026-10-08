# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Decoupling Exploration from Optimization in RLVR](https://arxiv.org/abs/2610.10536) | 提出探索-蒸馏框架，将RLVR中的探索与优化解耦：先用新颖性奖励训练探索者策略，再过滤其轨迹并蒸馏到不带新颖性奖励的学生策略中，从而在实现新策略发现的同时避免模型质量退化。 |
| [^2] | [Decentralized SGD under Heavy-Tailed Noise: Optimal Convergence Rates and the Role of Gradient Clipping](https://arxiv.org/abs/2610.10527) | 本文证明了带梯度裁剪的去中心化SGD（DSGD）在重尾噪声下对光滑非凸目标能够达到阶最优收敛速率，肯定地回答了简单的基线去中心化方法结合非线性操作即可实现最优收敛这一问题。 |
| [^3] | [Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models](https://arxiv.org/abs/2610.10526) | 本文揭示了视觉-语言-动作模型对指令措辞的极端敏感性（单词改动可使成功率波动数十个百分点），并提出无需修改策略、由大语言模型将措辞评分证据提炼为十余条改述规则并在部署时应用的方法来缓解该问题。 |
| [^4] | [Distilling Graph Geometry: Knowledge Gap from GNNs to MLPs](https://arxiv.org/abs/2610.10520) | 提出G²MLP，一种由Ollivier-Ricci曲率引导的训练时蒸馏框架，通过识别稀疏图上的谱欠拟合与稠密图上的谱过拟合两种失效模式，使无需图结构的MLP学生模型能够保留GNN教师模型的图诱导几何结构。 |
| [^5] | [Why Forget-Only Unlearning Needs Memorization](https://arxiv.org/abs/2610.10519) | 本文证明仅遗忘式机器遗忘（只用训练模型和待遗忘样本、无保留数据）并非总是可行，其可行性取决于学习方法，且算法必须对训练数据进行足够记忆才能处理任意删除请求。 |
| [^6] | [SciExam for ENSO: Can AI Agents Build Climate Models?](https://arxiv.org/abs/2610.10513) | 该论文提出了SciExam for ENSO基准测试，让AI智能体在六小时内仅凭真实观测数据自主构建ENSO低阶随机气候模型，并用隐藏评分器检验模型能否复现统计特性、恢复隐变量和预测保留年份，结果显示十二个智能体系统中有六个的模型优于已发表的模型。 |
| [^7] | [Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning](https://arxiv.org/abs/2610.10499) | 该论文提出了不可知平滑在线学习领域首个神谕高效且无参数的算法，同时摆脱了对基础测度采样访问能力和完美预测标签这两个限制性假设的依赖。 |
| [^8] | [Evolutionary Architecture Search for Chlorophyll-$a$ Prediction in Lakes using Sentinel-2](https://arxiv.org/abs/2610.10496) | 本研究在固定任务、特征与数据划分的条件下，利用正则化进化的神经架构搜索为Sentinel-2湖泊叶绿素a预测自动找到了一个仅含409个参数的小型网络，将保留集AUC从0.790提升至0.820，并收敛出单窄层加RMS归一化、tanh激活、阶梯衰减RMSprop和权重平均的一致设计配方。 |
| [^9] | [Best Arm Identification for Bandits with Shifting Means](https://arxiv.org/abs/2610.10488) | 该论文针对均值会对抗性漂移但奖励差距保持稳定的新型老虎机环境，提出了重要性加权算法ISM，证明了传统基于广义似然比检验的算法（如Track-and-Stop）在此环境下会失效，而ISM能保持δ-正确性并获得良好的样本复杂度保证。 |
| [^10] | [Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion](https://arxiv.org/abs/2610.10483) | 本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。 |
| [^11] | [Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts](https://arxiv.org/abs/2610.10460) | 提出 Δ-MOPD，通过迁移教师相对其基础模型的 logit 偏移并重新锚定到学生的冻结初始状态，消除端点策略中继承的基础模型偏好干扰，从而在多教师在线策略蒸馏的两种设置中降低教师项范数比和目标-学生 KL，实现更纯粹的教师后训练知识传递。 |
| [^12] | [NeuralBES: A Differentiable, Control-Aware Emulator for Scalable Building Energy Modeling](https://arxiv.org/abs/2610.10459) | NeuralBES提出了一种可微分建筑能耗仿真器，利用共享神经编码器将建筑元数据映射为物理受限的RC热模型参数，并通过对数空间并行扫描高效求解，在保持物理模型可信性的同时实现了跨数百万异构建筑的大规模可扩展能耗建模。 |
| [^13] | [A Good Self-Teacher Meets the Student Where They Are: Joint On-Policy Learning and Teaching](https://arxiv.org/abs/2610.10447) | 该论文指出自蒸馏中特权信息会使教师通过捷径解题、产生与学生当前行为不匹配的监督信号，并提出联合在线策略学习与教学的方法，使教学顺应学生的当前水平。 |
| [^14] | [Seq-Flow: Efficient Probabilistic Forecasting with Self-Rollout Error Control](https://arxiv.org/abs/2610.10440) | 提出 Seq-Flow 条件流模型，通过将样本从先前预测分布直接输运到更新后的分布，并结合自滚动训练控制误差累积，实现仅需少量流评估的高效概率预测更新。 |
| [^15] | [Q-Learning with Scalar Adjoint Matching](https://arxiv.org/abs/2610.10437) | 本文提出标量伴随匹配方法，利用预训练流策略的速度雅可比矩阵集中于对角线这一发现，推导出闭式标量伴随以替代昂贵的逐步向量-雅可比积计算，从而实现更高效的流策略离策略强化学习微调。 |
| [^16] | [Conditional Flow Matching for Generation of 3D Multi-variable Instantaneous Urban Microclimate Fields](https://arxiv.org/abs/2610.10430) | 本文提出基于条件流匹配（CFM）的生成框架，以建筑几何和平均流场为条件，在数秒内生成合理的三维多变量瞬时城市微气候场，既克服了大涡模拟计算成本高的局限，又解决了回归模型无法表征湍流随机性的问题。 |
| [^17] | [Derivative Gaussian Processes on a Two-Direction Budget](https://arxiv.org/abs/2610.10428) | 本文提出一种每个观测梯度仅需两个方向的导数高斯过程，在Vecchia近似下将 $md$ 个梯度坐标压缩为至多 $2m$ 个方向导数，使每个预测目标的计算代价降至 $\mathcal{O}(m^3)$，并给出了近似误差的理论界。 |
| [^18] | [Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL](https://arxiv.org/abs/2610.10422) | 该工作发布了BehaviorTrace开源评估框架，通过植入已知成因的行为实验发现，在线RL中训练数据归因方法的表现很大程度上源于梯度大小和模型流畅度等混淆因素，揭示了现有归因信号的可靠性局限。 |
| [^19] | [Steerspeech: Activation Steering For Emotion Control In Generated Speech](https://arxiv.org/abs/2610.10415) | SteerSpeech是一种轻量级激活引导框架，通过向冻结的TTS模型隐藏激活中注入引导向量，实现推理时的精准情感控制，同时保持说话人身份与语言内容不变。 |
| [^20] | [Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds](https://arxiv.org/abs/2610.10411) | 本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。 |
| [^21] | [RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments](https://arxiv.org/abs/2610.10409) | RobotWorld是一个包含84项任务（涵盖操作、移动操作、运动、驾驶和空中控制）的机器人使用仿真基准，用于评估多模态智能体将指令转化为物理执行的能力，发现现有智能体虽能构建复杂的感知与控制工作流，但难以将这些能力可靠地组合以完成任务。 |
| [^22] | [Rubix: Global Correspondence-Free Point Set Alignment through Assignment Geometry](https://arxiv.org/abs/2610.10408) | 提出Rubix方法，通过刻画置换多边形的几何结构实现无对应点集对齐问题的全局最优求解，证明了多边形顶点数的紧界n(n-1)并解决了Rote的旋转-指派开放问题。 |
| [^23] | [SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions](https://arxiv.org/abs/2610.10407) | SOTA是一个期权交易智能体框架，通过将庞大的期权合约空间抽象为策略层面的决策、由确定性求解器执行投资组合构建，并利用期权隐含收益分布引导大语言模型，使其能够根据市场条件的变化灵活选择和切换期权交易策略。 |
| [^24] | [Cross-Domain Pretraining for Steady-State Neural CFD Surrogates](https://arxiv.org/abs/2610.10398) | 跨域预训练能显著提升神经CFD代理模型在未见数据集上的零样本和少样本泛化能力，在相同样本量下相比从头训练和领域专家迁移可将误差降低2-3倍。 |
| [^25] | [Executing Causal Structure Learning with Linear-Attention Transformers](https://arxiv.org/abs/2610.10395) | 本文显式构造了一个固定权重的线性注意力Transformer，其前向传播可精确复现无环约束下连续因果发现算法的一次迭代更新，并证明在更新间保留算法乘子是实现精确执行的关键。 |
| [^26] | [Kernel Autoresearch for Open-Ended Model Discovery](https://arxiv.org/abs/2610.10394) | Kernaut将核设计视为开放式模型发现，让编程智能体以程序形式自由生成核，同时通过构造契约保证每个核的有效性，并结合质量-多样性档案与新颖性筛选，突破了固定核语法表达能力的限制。 |
| [^27] | [Safe Meta-Policy Design with Risk Control](https://arxiv.org/abs/2610.10393) | 该论文提出一种带风险控制的离线元策略设计方法，在更新退化风险预算约束下通过动态规划规划模型更新时机，并揭示策略改进的信噪比是决定更新频率与风险分配的关键因素。 |
| [^28] | [OrBIT: Structure-Guided Embedding Compression](https://arxiv.org/abs/2610.10385) | OrBIT提出了一种结构引导的嵌入压缩框架，通过从轨道动力学中学习可复用的局部几何来约束共享码字，并利用全局残差分配编码预算，实现嵌入表的高效压缩。 |
| [^29] | [Boosting and the Expressive Power of Simple Weak Learners via the $\gamma$-VC Dimension](https://arxiv.org/abs/2610.10383) | 本文证明 γ-VC 维在相差一个随 γ 缩放的常数因子意义下刻画了从弱到强学习的样本复杂度，锐化了经典 VC 维与 γ-VC 维的一般关系，并针对决策树桩和轴平行矩形给出了 γ-VC 维的改进上下界。 |
| [^30] | [ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals](https://arxiv.org/abs/2610.10381) | 提出ResidualQuant方法，利用循环Transformer中各循环KV状态高度相似的特点，以最后一轮KV状态为参考、用2比特低精度残差表示其余循环，并结合最小二乘缩放、旋转和逐循环混合精度，实现精确的INT2级KV缓存量化。 |
| [^31] | [Continual Learning without Continual Training](https://arxiv.org/abs/2610.10379) | 提出Latent Concept PFN模型，将元训练后冻结的PFN用于持续推断而非持续训练，仅通过在上下文证据集中添加样本并基于潜在概念空间进行贝叶斯后验更新来适应新领域和新类别，全程不改变任何参数，从而显著减少遗忘。 |
| [^32] | [Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation](https://arxiv.org/abs/2610.10368) | 本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。 |
| [^33] | [Temporally Interpretable Differentiable Decision Trees](https://arxiv.org/abs/2610.10367) | 该论文提出了“时间可解释性”这一可解释性的新维度，通过引入两种结合动作分块的新型策略梯度算法，解决了可微决策树的单时间步行为与人类多时间步规划之间的固有不匹配问题，使其更适合序列决策任务。 |
| [^34] | [Koopman Observers for Diffusion Acceleration: Correcting Feature Forecasts with Shallow Measurements](https://arxiv.org/abs/2610.10366) | 提出一种观测修正的库普曼框架，在不改变模型参数的前提下，利用即时计算的浅层特征观测来修正深层特征的库普曼预测，并配合周期性完整评估刷新观测器，从而加速冻结扩散模型的采样。 |
| [^35] | [Pathwise Information Certificates for Decentralized Adaptive Sensing](https://arxiv.org/abs/2610.10362) | 该论文提出一种基于沿实际感知轨迹累积的 Rényi–Chernoff 信息的逐路径证书，为去中心化自适应感知提供了非渐近的 MAP 错误界和随时可用的全网停止规则，并证明信息线性增长可保证误差指数衰减。 |
| [^36] | [ORDERS: An Empirical Study of Norm-Rank Aggregation for Personalized Federated Learning](https://arxiv.org/abs/2610.10361) | 论文提出ORDERS配置，将共享主干、私有残差适配器与按更新范数降序的几何权重聚合相结合，并通过80次运行的完整评估表明其在个性化联邦学习中相比均匀权重对照和FedPer基线仅带来微小但可复现的准确率提升。 |
| [^37] | [Measurement-Efficient Differentiable Quantum Architecture Search for Combinatorial Optimization](https://arxiv.org/abs/2610.10351) | 该论文提出了一种理论推导的测量缩减方案，可在不改变优化目标的前提下将可微分量子架构搜索（DQAS）的梯度测量成本降低约39%至41%，并通过3-SAT和MaxCut基准问题的实验验证了其有效性。 |
| [^38] | [AutoAdapt: Automatic Domain Discovery Enables Low-Cost Extensibility](https://arxiv.org/abs/2610.10349) | AutoAdapt是一个模块化框架，通过自动发现潜在领域并并行训练独立的LoRA适配器与无参数路由，实现了无需全模型重训练的低成本新领域扩展。 |
| [^39] | [Dataset Pruning from First Principles: A Label-Free Linear Programming Approach](https://arxiv.org/abs/2610.10347) | 提出了一种从第一性原理推导的数据集剪枝方法，将无偏子集选择表述为方差最小化的线性规划问题，无需标签和几何邻近假设即可选出具有代表性的训练子集。 |
| [^40] | [Data Reuse in Non-Stationary Learning](https://arxiv.org/abs/2610.10340) | 提出了暴露上限复用（ECR）类算法，通过结合在线变化检测、兼容性测试和污染控制来安全复用历史数据，使非平稳在线学习的遗憾值随不同取值数量而非变化次数增长，类似于偏差-方差权衡。 |
| [^41] | [Average-Reward Reinforcement Learning for Multichain MDPs: A Hierarchical Decomposition Approach](https://arxiv.org/abs/2610.10326) | 该论文提出一种基于异步值迭代与Bather分层分解的强化学习算法，无需模型知识即可求解平均奖励多链MDP，并在有限时间内收敛到最优增益与增益最优策略。 |
| [^42] | [HAN-Mamba: Hierarchical Selective State Space Networks for Multi-Scale Financial Volatility Forecasting](https://arxiv.org/abs/2610.10323) | 本文提出 HAN-Mamba，用选择性状态空间（Mamba）编码器替代层次化架构中的 Transformer 编码器，仅在三标记融合器中保留注意力机制，以高效整合多尺度市场信息并提升金融已实现波动率预测性能。 |
| [^43] | [Estimating Uncoded Crash Factors with Tabular Foundation and System One Models: Kumo Tabular and Jev](https://arxiv.org/abs/2610.10321) | 本研究结合表格基础模型、系统一叙述模型与人工校准，利用警察叙述文字估计编码字段遗漏的车祸因素，发现叙述记录的受伤车祸数量远超编码统计（如手机使用因素为15,074起对7,340起）。 |
| [^44] | [Thinking in Depth: Retrospective Inference for Tabular Foundation Models](https://arxiv.org/abs/2610.10317) | 提出基于回溯推理的表格基础模型Retro，让网络后层能够显式重访并重组前层产生的中间表示，以解决现有TFM中预测精细化集中于深层、分布不均匀的问题。 |
| [^45] | [PoreML: A Data-Driven Framework for Learning Multiphase Flow in Porous Media](https://arxiv.org/abs/2610.10314) | PoreML是一个基于孔隙尺度物理的开源数据驱动框架，统一了多孔介质多相流的数据生成（GPU原生格子Boltzmann求解器）、大规模数据集（3.3 TB、560次模拟、15.8万余个时间步）与模型训练评估流程，填补了该领域机器学习研究中数据稀缺与缺乏统一工作流的关键空白。 |
| [^46] | [Fault-tolerant foundation models](https://arxiv.org/abs/2610.10311) | 该论文发现大型语言模型可被训练以容忍硬件故障，且模型越大错误韧性越强，并由此推测适当训练的LLM可能具有形式上的容错性，从而为在低能耗故障硬件上运行AI推理、实现大幅节能开辟道路。 |
| [^47] | [RSIGym: A Flexible Environment for Recursive Self-Improvement](https://arxiv.org/abs/2610.10310) | 本文提出了RSIGym，一个基于“一切皆服务”理念的智能体原生研究环境，通过可重用服务支持数据、执行框架和联合优化三条改进赛道，实现灵活的递归自我改进研究，并定义了RSI-Index作为跨五个基准测试的统一评估指标。 |
| [^48] | [How Do Transformers Learn to Represent Symmetries?](https://arxiv.org/abs/2610.10305) | 本文研究了原始Transformer通过有限数据增强学习点云对称性的能力，发现不同对称群的可学习性存在递增排序（不保角→保角→基本保角子群），并通过结构分析揭示了模型产生不变性的可解释机制。 |
| [^49] | [SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning](https://arxiv.org/abs/2610.10304) | 提出SemanticFold潜在序列压缩方案，通过在学习的边界折叠前缀隐藏状态来压缩提示前缀，发现压缩对语言建模、可解码性和推理能力的影响是非单调的且各自具有不同的压缩阈值，证明这些能力可以相互分离。 |
| [^50] | [Continual Graph Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2610.10302) | 该论文提出CGMARL框架，通过将任务序列映射为一系列建模任务特定结构的属性图，来促进知识迁移并缓解遗忘，从而解决持续多智能体强化学习中的结构信息利用问题。 |
| [^51] | [Revisiting Explainable AI through Model-Independent Concept Dictionaries](https://arxiv.org/abs/2610.10301) | 提出DictXAI方法，通过在输入域中用包含可解释含义的词典定义概念，将输入的稀疏编码与模型预测归因到具体词典元素，从而实现跨模型、架构无关且可操作的可解释AI解释。 |
| [^52] | [Shared Gaussianization: What Gaussian Regularizers Certify About Contrastive Learning, and What They Miss](https://arxiv.org/abs/2610.10299) | 本文提出共享高斯化（SG）检验，证明该高斯正则化器能以紧的、与维度无关的平方根速率上界总体 InfoNCE 的超出量，并通过单次检验同时检测视图的失配与非均匀性。 |
| [^53] | [Physics-Aligned Electronic Ground-State Learning Improves Generalization](https://arxiv.org/abs/2610.10298) | 该论文通过物理约束将电子基态描述子模型的学习目标与KS-DFT控制方程对齐，其提出的ON-Loss和GROOT方法在尺寸外推任务上相比之前最先进的密度基态模型将能量和力的平均绝对误差分别降低了79.1%和83.4%。 |
| [^54] | [Energy-Efficient Gait Adaptation via Hierarchical Reinforcement Learning for Quadrupedal Locomotion Across Diverse Terrains](https://arxiv.org/abs/2610.10297) | 提出一种分层强化学习框架，将高频关节级运动执行与低频能量最优步态自适应相解耦，实现四足机器人在多样地形与速度范围下能量高效、鲁棒的运动，并可通过零样本方式完成仿真到现实的迁移。 |
| [^55] | [AI Safety Considerations for Agents With Limited Time to Act](https://arxiv.org/abs/2610.10285) | 本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。 |
| [^56] | [Temporal Visuo-Tactile Learning for Dexterous Grasp Stability](https://arxiv.org/abs/2610.10283) | 本文构建了包含 200 个物体、10,000 次抓取试验的视-触-本体感觉多模态数据集，证明了高分辨率动态触觉感知能显著提升灵巧手抓取稳定性的预测能力。 |
| [^57] | [Neural Sampling with Reweighted Normalizing Flows via the Wasserstein--Fisher--Rao JKO Scheme](https://arxiv.org/abs/2610.10278) | 提出了一种基于WFR JKO格式的神经采样算法，首次证明其在任意固定步长下、无需对数凹性等结构性假设即可指数收敛到目标分布，并利用重加权归一化流对其输运与反应分量进行神经参数化实现。 |
| [^58] | [PatchBench: Measuring Collateral Damage in Activation Patching](https://arxiv.org/abs/2610.10276) | 提出PatchBench基准，用于衡量激活修补在修复LLM越狱行为时对无关行为造成的附带损害，从而区分真正的选择性修复与更广泛的局部行为抑制。 |
| [^59] | [Sparse Planning in Visual World Models via Cost Gradients](https://arxiv.org/abs/2610.10274) | 本文提出COSTGRAD，一种无需训练的token选择方法，通过规划代价对每个token的梯度范数筛选出对规划真正重要的空间token，在50%稀疏度下即可保持或超越全token规划的性能，并结合精简CEM搜索实现最高约5倍的规划加速。 |
| [^60] | [A Closed-Loop Non-Asymptotic Convergence Analysis of PPO with Learned Critics and Clipping](https://arxiv.org/abs/2610.10273) | 本文将PPO-Clip建模为闭环演员-评论家系统并首次给出非渐近收敛性分析，统一刻画了策略平稳性与评论家跟踪精度，并显式揭示了对算法参数（包括延迟相关的步长限制）的依赖。 |
| [^61] | [Using Small Language Models to Reverse-Engineer Machine Learning Pipelines Structures](https://arxiv.org/abs/2610.10261) | 该研究验证了小语言模型能够凭借其代码理解与分类能力，从源代码中有效提取机器学习流水线的阶段结构，从而克服人工标注不可扩展及传统分类器难以适应领域多样性的局限。 |
| [^62] | [PairAudit: Guiding Human Review with Graph Tokens under Distribution Shift](https://arxiv.org/abs/2610.10260) | PairAudit利用图Token捕捉相连节点间的预测关系模式，在固定审查预算下发现入侵检测器中被忽略的高置信度错误，在分布偏移（未见攻击）场景下比基于不确定性的审查方法纠正更多错误。 |
| [^63] | [On the Cyclic Assumption of the Cow-Path Search Algorithm](https://arxiv.org/abs/2610.10253) | 本文为牛路搜索问题中“没有任何算法能优于最佳循环顺序算法”这一断言提供了详细的证明，从而完整补足了该随机算法最优性的论证。 |
| [^64] | [Logarithmic Regret via Passive Change Detection in Piecewise-Stationary Self-Tuning Regulation](https://arxiv.org/abs/2610.10250) | 本文提出PIECE-CD算法，利用“系统变化会在旧控制器下提高输出能量”这一被动检测机制，对分段平稳自回归系统的最小方差自校正控制实现了概率至少为1-δ的O((C+1)log((T+1)/δ))对数遗憾。 |
| [^65] | [Stationary Bias and Extrapolation in Nonlinear Two-Timescale Stochastic Approximation](https://arxiv.org/abs/2610.10246) | 本文针对马尔可夫链驱动的非线性双时间尺度随机逼近，推导了在慢步长远小于快步长时仍一致有效的一阶偏差展开，揭示出 ε²/η 的混合偏差项，并表明沿幂律步长路径的偏差指数可以是非整数，因此 Richardson–Romberg 外推必须使用与步长路径相匹配的权重。 |
| [^66] | [MorphCL: Morphological Contrastive Learning for Inertial-based Human Activity Recognition](https://arxiv.org/abs/2610.10245) | MorphCL是一种自监督预训练框架，通过基于运动基元发现和领域特定特征描述符的结构感知分组，将全局结构的显式建模引入基于惯性传感器的人体活动识别自监督学习中，从而解决了现有方法依赖随机采样和局部比较、忽视大规模运动数据全局结构的问题。 |
| [^67] | [ProtocolMatch: Protocol-Dependent Model Selection for Scientific Dynamics Forecasting](https://arxiv.org/abs/2610.10239) | 该论文提出 ProtocolMatch 评估框架，将科学动力学预测的模型选择从单纯的架构比较扩展为“协议依赖”问题，并通过量子自旋动力学实验证明模型优劣排序会随训练规模、观测历史和闭环部署等协议条件而逆转。 |
| [^68] | [From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification](https://arxiv.org/abs/2610.10227) | 本文提出一种三阶段的LLM引导框架，通过提示LLM先生成规则再将其组织成决策树，在少样本表格分类任务中以显著更低的提示开销实现了更优的准确性和可解释性。 |
| [^69] | [Evaluating Sequence Assembly Strategies for Differentially Private Synthetic Time-Series Forecasting](https://arxiv.org/abs/2610.10222) | 该论文首次系统研究了差分隐私合成时间序列在生成后的窗口组装策略（重叠率与窗口加权方案），并通过在四类数据集和五种预测模型上的实验，揭示出下游预测效用由预测模型、重叠率和加权方案共同决定。 |
| [^70] | [Finite-Sample Approximation of Hessian-Guided Perturbed Wasserstein Gradient Flows](https://arxiv.org/abs/2610.10218) | 该论文证明了海森引导扰动Wasserstein梯度流的有限粒子逼近在增长时间尺度上的高概率追踪界，关键在于沿参考路径累积的曲率——负曲率会放大误差而正曲率可抑制误差，从而刻画了暂时不稳定仍可精确追踪的有利情形。 |
| [^71] | [RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations](https://arxiv.org/abs/2610.10214) | RoBART通过为每棵树分配特定的Givens旋转来改进贝叶斯可加回归树，使其能高效逼近与预测变量轴不对齐的边界，并对具有各向异性Hölder光滑性的可加函数证明了后验收缩率。 |
| [^72] | [Edge Accuracy Is Not Enough: Why Dynamics-Learned Structure Fails to Transfer to Inverse Problems](https://arxiv.org/abs/2610.10213) | 本文证明，尽管从动力学中学到的图结构（如NRI）在理论上满足降低样本复杂度的边误差条件，但在迁移到逆问题时仍会系统性失效——边的准确率并不足够，而基于物理的先验（如格林函数）表现明显更优。 |
| [^73] | [OrthoGen: A Generative Orthogonal Learner for Time-Varying Treatments](https://arxiv.org/abs/2610.10210) | 提出OrthoGen框架，通过生成式递归g-计算与正交化学习相结合，在时变混杂下利用灵活的生成模型估计时变处理的条件分布势结果，并增强对冗余参数估计误差的稳健性。 |
| [^74] | [CARES: A Controlled Synthetic Benchmark of Speaker Reactions to Sound](https://arxiv.org/abs/2610.10208) | 本文提出CARES——一个包含10,000个双说话者场景的受控合成基准，通过“说话者对声音产生可听见的反应”这一规则定义声音显著性以固定真实标注，并揭示现有音频-语言模型虽能识别声音本身，却无法理解说话者对声音的反应。 |
| [^75] | [Computations of the slice genus and the unknotting number of links via machine learning](https://arxiv.org/abs/2610.10206) | 该论文利用强化学习和贝叶斯优化，为链环的切片亏格、解结数等难以算法计算的不变量求出新上界，并结合已知下界在许多情形下得到新的精确值，还重现了解结数的非可加性反例。 |
| [^76] | [How to train your model organism](https://arxiv.org/abs/2610.10203) | 该论文提出对齐模式生物的验证不应止于单一目标行为安装，而应从目标行为安装、通用能力保持和输出自然性三个维度进行验证，并证明这些验证指标可以有效预测可解释性方法能否成功恢复已安装的行为。 |
| [^77] | [GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning](https://arxiv.org/abs/2610.10201) | 该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。 |
| [^78] | [Robust Decentralized Fairness Auditing](https://arxiv.org/abs/2610.10199) | 提出了Auditopus——一种无需中央服务器、以多轮方式协同进行、且能够在审计者可能合谋进行“公平洗白”时依然保持鲁棒的LLM去中心化公平性审计方法。 |
| [^79] | [Broadly Applicable Approximate MCMC for Switching Stochastic Differential Equations Using Uniformization and Time-Conditioned Factorized Neural Likelihood Estimation](https://arxiv.org/abs/2610.10194) | 该论文提出了一种结合均匀化与时间条件化因子分解神经似然估计的近似MCMC采样器，突破了现有方法在噪声观测、状态维度、漂移形式和扩散项等方面的限制，实现了对切换随机微分方程广泛适用的贝叶斯推断。 |
| [^80] | [Universal Local Error and Realized Amplification for the First-Order EDM Predictor](https://arxiv.org/abs/2610.10190) | 该论文证明了一阶EDM扩散采样器的单步局部离散化误差具有与数据分布无关的普适二次上界，并通过在高噪声水平利用显式收缩准则、在低噪声水平引入可远小于最坏Lipschitz常数的实际放大效应，最终获得了O(e^{Λ_K}/K)的全局离散化误差保证。 |
| [^81] | [Kinetic Langevin Meets Split Gibbs: Accelerated Posterior Sampling for Imaging Inverse Problems with Diffusion Priors](https://arxiv.org/abs/2610.10187) | 该论文提出RED-KLwSGS方法，将欠阻尼（动力学）朗之万扩散与分裂吉布斯采样框架结合，利用单次去噪得分驱动辅助变量更新，在与Langevin-within-SGS相同的每次迭代成本下实现成像逆问题后验采样的加速，并给出了强对数凹先验下连续和离散时间的非渐近Wasserstein-2收敛保证。 |
| [^82] | [Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization](https://arxiv.org/abs/2610.10186) | 该论文提出了一种通过在高斯过程样本路径上运行贝叶斯优化来最小化期望累积遗憾，从而利用另一个贝叶斯优化过程自动预训练贝叶斯优化算法参数的框架。 |
| [^83] | [Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents](https://arxiv.org/abs/2610.10179) | 该论文系统研究了从中间检索步骤提取学习信号的奖励塑形与信用分配策略，并提出将中间信号与最终结果奖励相结合的训练框架，显著提升了搜索智能体在多跳问题上的学习效率与总体性能。 |
| [^84] | [A Unified Information-Theoretic Approach to Constrained Multi-Fidelity Multi-Objective Bayesian Optimization](https://arxiv.org/abs/2610.10174) | 该论文提出一种统一的信息论方法，通过变分下界近似计算关于最高保真度可行帕累托前沿的信息增益，构建成本感知的采集函数，以联合处理多目标、约束和多保真度选择问题。 |
| [^85] | [Conformal Prediction for Spatially Dependent Data via Sequential Whitening](https://arxiv.org/abs/2610.10168) | 该论文提出一种通过对校准残差进行顺序条件化（顺序白化）的保形预测方法，解决了空间相关数据下可交换性假设失效、且仅由校准残差可预测的空间变异残留导致区间效率下降的问题，在正确的工作协方差与椭圆残差分布下实现精确的有限样本覆盖率，并可借助最近邻近似扩展到大型网络。 |
| [^86] | [Policy Learning with Weak Signals](https://arxiv.org/abs/2610.10167) | 该论文证明在低信噪比的大规模数字实验中最优策略一般不可学习，但当处理效应平滑变化时，基于线性平滑器的极小极大自适应策略可实现趋近于零的福利遗憾，并在Netflix真实实验中验证了个性化平滑策略的优越性。 |
| [^87] | [Progress and Prospect of AI in ARPES Workflow](https://arxiv.org/abs/2610.10140) | 本文系统综述了机器学习方法在ARPES（角分辨光电子能谱）完整工作流程（从样品制备、数据采集到数据分析与理论对比）中的应用现状、优势与局限性，是该领域首篇全面评述机器学习可靠性与应用前景的综述文章。 |
| [^88] | [Attention via Black-Box Vector Search](https://arxiv.org/abs/2610.10135) | 本文通过优先抽样框架统一了基于MIPS的稀疏注意力方法，证明了单个索引下Θ(√n/ε)个检索键的紧致界、多索引下O(log n + 1/ε²)的近最优算法，并展示了通过增强键和查询可突破下界。 |
| [^89] | [m-Set Adversarial Bandits with Winner Feedback](https://arxiv.org/abs/2610.10128) | 本文研究了不同效用和反馈模型下m集对抗性多臂老虎机的遗憾界，其主要技术贡献是遗憾值的信息论下界，揭示了环境设置中的细微变化会对学习速率产生巨大影响。 |
| [^90] | [Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition](https://arxiv.org/abs/2610.10124) | 该论文通过构建三个控制变量的匹配损失函数，首次将“为错失目标添加监督”与“两组目标的概率竞争”这两个效应解耦，发现概率竞争会损害返回物品的排序质量，从而解释了附加错失目标这一训练策略效果难以归因的原因。 |
| [^91] | [What Can a Gaussian Process Design Test](https://arxiv.org/abs/2610.10122) | 该论文揭示仅凭高斯过程的试验设计（无需观测任何响应）就能判断模型假设能否被数据证伪，并将有限特征GP的可检验关系精确刻画为核矩阵的零空间，同时借助Gale对偶性给出“电路”的几何解释，并对一般核函数定义了概率意义上的软性检验关系。 |
| [^92] | [YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory](https://arxiv.org/abs/2610.10118) | YANchor-4B 通过将关键记忆保存为可检索的锚点，以 O(N) 时间和 O(1) 内存的成本实现了高效的长程推理，在数学基准上大幅超越同类模型并具有数倍于 Transformer 的生成吞吐量。 |
| [^93] | [CAFE+FNO: Fourier Kernel Generation via Multiplicative Feature Composition](https://arxiv.org/abs/2610.10105) | 本文提出CAFE+FNO，通过并行仿射分支与Hadamard乘积显式组合傅里叶-切比雪夫特征来生成傅里叶核，从而增强FNO对高频变化的学习能力。 |
| [^94] | [ExperienceIndex: Artifact-Grounded Memory](https://arxiv.org/abs/2610.10091) | 提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。 |
| [^95] | [Multi-Agent Coordination via Support-Preserving Distillation](https://arxiv.org/abs/2610.10087) | 提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。 |
| [^96] | [Activation-Aware Weight Tensorization: A Calibration-Time Preconditioner for Tensor-Network LLM Compression](https://arxiv.org/abs/2610.10085) | AWT是一种免训练的校准期预处理方法，通过激活感知的对角缩放对权重矩阵进行预处理后再执行TT/TTN分解，在2-6倍压缩率下显著缩小了张量网络压缩后的大语言模型与稠密基线之间的困惑度差距。 |
| [^97] | [Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel](https://arxiv.org/abs/2610.10080) | 本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。 |
| [^98] | [Towards Calibrated Probabilistic Forecasts for Events of Interest via Outcome-Conditional Recalibration](https://arxiv.org/abs/2610.10076) | 本文提出了一种简单易实现的事后重校准方法——结果条件重校准，能够在用户定义的结果空间区域（如极端事件）上对概率预测进行重校准，从而确保决策者最关注的事件也能获得校准良好的预测。 |
| [^99] | [Efficient Provably Private Classification with a Tabular Foundation Model](https://arxiv.org/abs/2610.10068) | PrivTab是一种将差分隐私机制内嵌于模型架构的表格基础模型，通过上下文学习将敏感数据行转换为可证明隐私的紧凑摘要，实现了高效且具有正式隐私保证的分类。 |
| [^100] | [TRACK: Telemetry-Based Racing Analysis and Coaching Kit in Sim Racing Games](https://arxiv.org/abs/2610.10061) | 该论文提出了TRACK框架，通过将驾驶环节映射到速度、制动、策略与稳定性构成的四维行为空间并进行无监督聚类来刻画车手驾驶行为画像，同时采用零假设基线校准来严谨评估聚类结果的有效性，为模拟赛车提供数据分析与教练指导能力。 |
| [^101] | [WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting](https://arxiv.org/abs/2610.10057) | 提出WxFM-XL模型，通过引入跨站点误差相关性先验图与自适应动态融合机制，将单变量时间序列基础模型成功适配到多站点天气预报任务中。 |
| [^102] | [Transition Path Sampling Using Koopman Operators and Exit-Time Optimal Control](https://arxiv.org/abs/2610.10054) | 提出一种基于Koopman算子的转移路径采样新方法，利用其线性性质在无需转移路径数据的情况下识别亚稳态集合并估计committor函数，同时将TPS表述为退出时间最优随机控制问题，从而解决了现有神经网络方法的计算开销与性能保证问题。 |
| [^103] | [Evolve on the Host, Predict on the Edge: Deploying Online Neuroevolutionary Architecture Search for Cross-sectional Stock Return Prediction](https://arxiv.org/abs/2610.10038) | 该论文提出ONE-NAS在线神经进化架构搜索方法，通过主机进化、树莓派边缘端预测的分工流水线实现低延迟的日度横截面股票收益预测，在扣除实际交易成本后取得+27.5%的净收益，显著优于在线LSTM、在线GRU等基线模型。 |
| [^104] | [Matching of signal, noise and hardware timescales for filtering and forecasting of correlated noise signals](https://arxiv.org/abs/2610.10037) | 本文利用纳米多孔氧化铌物理储层计算系统揭示了噪声相关时间、储层记忆与预测视界之间的匹配关系决定了相关噪声是被滤波还是被预测，并提出储层记忆视界和预测状态指数两个新指标来区分这两种工作模式。 |
| [^105] | [Gaussian Equivalence for Multi-Head Self-Attention](https://arxiv.org/abs/2610.10033) | 利用随机矩阵理论建立了多头自注意力的高斯等价性，证明用缩放分数加高斯噪声替代softmax注意力可保持中心化输出的极限谱定律，从而分离了头分配与投影宽度的影响。 |
| [^106] | [Structure alone supports efficient visual computation in the Drosophila visual system](https://arxiv.org/abs/2610.10023) | 本研究将固定的果蝇连接组结构与眼睛模型直接耦合，仅通过学习突触增益和神经元阈值，就实现了颜色辨别、形状分类和近似数字辨别等多任务视觉计算，证明了神经连接结构本身即可支撑高效的视觉信息处理。 |
| [^107] | [Controlling Dependence in Implicit Generative Models via Spread Mutual Information](https://arxiv.org/abs/2610.10021) | 提出扩散互信息（SMI），通过对生成变量施加扩散核并跨噪声级别对互信息进行加权积分，克服了隐式生成模型中奇异分布缺少得分函数及密度比估计重叠性差的难题，实现对统计依赖的有效控制。 |
| [^108] | [Oscillatory Neural Dynamics over Sheaves](https://arxiv.org/abs/2610.10018) | ONDA 提出了一种由学习的层传输算子驱动的二阶振荡信息波图学习框架，通过逐茎敏感性分析证明节点间交叉影响永不消失，从而实现有效的长程传播，并在长程传播、图瓶颈、图迁移和异配基准上持续超越标量波方法。 |
| [^109] | [A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks](https://arxiv.org/abs/2610.10014) | 该研究证明，仅以单只果蝇的全脑连接组作为固定拓扑、并为每条神经连接学习一个标量权重，人工网络就能在加法运算和接地关系语言等人类设计的认知任务上显著超越随机重连的对照网络，表明生物神经环路结构本身具有超越其演化目标的通用计算价值。 |
| [^110] | [Beyond Reward Suppression: Near-Optimal Offline Attacks on Warm-Start Bandits with Bounded Rewards](https://arxiv.org/abs/2610.10000) | 该论文首次证明针对热启动老虎机的最优成本攻击不能仅靠压制非目标臂实现，在目标臂奖励接近下边界时必须直接投入成本推广目标臂，并据此设计了成本分配明确的最优次线性攻击方案。 |
| [^111] | [Temporal Predictive Multiplicity: Equally Accurate Time Series Models Yield Different Forecast Trajectories](https://arxiv.org/abs/2610.09994) | 该论文提出“时序预测多样性”框架，揭示预测性能几乎相同的时间序列模型可能在完整预测轨迹上产生显著分歧，且仅在单个时域约束多样性无法消除轨迹层面的差异。 |
| [^112] | [Efficient Patch-Based Anomaly Detection Fused with Diffusion Driven Generative Modeling for Semiconductor Wafer Bin Map Open Set Anomaly Detection](https://arxiv.org/abs/2610.09993) | 该论文提出一种融合基于补丁的学生-教师检测器与去噪扩散概率模型的混合单类异常检测框架，通过百分位校准分数的固定凸组合，仅用700片正常晶圆训练即在晶圆Bin图开集异常检测中达到0.9985的AUROC，并将误分类数从852/1412显著降至618。 |
| [^113] | [Marrying Pricing and Advertising with LLMs](https://arxiv.org/abs/2610.09985) | 提出了一种将LoRA微调的预训练大语言模型与在线actor-critic强化学习相结合的算法，使卖家在需求未知且仅有购买反馈的情况下，联合优化定价与LLM生成的广告以最大化收入。 |
| [^114] | [Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative](https://arxiv.org/abs/2610.09984) | 本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。 |
| [^115] | [TR-PTQ: High-Accuracy Integer-Only Transformer Post Training Quantization via Taylor Region Reformulation](https://arxiv.org/abs/2610.09969) | 该论文发现Transformer量化的精度损失主要源于归一化层的尺度参数和GELU的复合近似而非SoftMax，并提出基于共享泰勒区域指数对数原语的统一纯整数公式TR-PTQ，使除法、平方根等复杂运算均可通过对数域整数运算完成，从而实现高精度的纯整数Transformer推理。 |
| [^116] | [Force without transmission: a depth-induced rank collapse that no loss on the representation reopens](https://arxiv.org/abs/2610.09958) | 该研究发现，深度诱导的 transformer 秩坍缩无法通过任何作用于表示的损失项来修复，因为问题关键在于梯度传播路径被阻断而非矫正力不足，而恢复跳跃连接无需改变任何权重即可立即重开梯度路径、使秩得到恢复。 |
| [^117] | [Possibilistic Radial Transport for Approximate IM Inference](https://arxiv.org/abs/2610.09956) | 提出一种可能性径向传输方法，将参数的可能性轮廓值编码到源点半径中，并结合深度学习算法实现高效的近似可能性推断模型推断，使覆盖率评估、功效分析和新数据预测检验变得切实可行。 |
| [^118] | [Learning Traffic Flow Dynamics with Stochastic Physics-Informed Neural Cellular Automata](https://arxiv.org/abs/2610.09946) | 本文提出一种物理信息神经元胞自动机（PI-NCA），通过设计与道路拓扑物理一致、保证车辆总数守恒的神经架构，并进一步扩展至随机动力学，实现从数据中学习符合物理约束的交通流局部演化规则。 |
| [^119] | [Many Ways to Succeed: Diversity-Driven RL Fine-Tuning for VLA Generalization](https://arxiv.org/abs/2610.09943) | 该论文提出DRIVE方法，将成功行为的多样性作为显式的强化学习目标，通过在匹配任务条件下分组轨迹并进行时间对齐比较、引入成功条件下的内在奖励，从而扩大策略对有效解空间的覆盖，显著提升VLA模型在分布偏移下的泛化能力。 |
| [^120] | [Expected Sample Complexity in Multi-Armed Bandits](https://arxiv.org/abs/2610.09929) | 本文提出了期望近似正确（ACE）新框架来研究多臂老虎机的期望样本复杂度，证明ACE保证蕴含几乎必然收敛到最优期望奖励，并揭示了确定性算法无法获得良好ACE界这一特性，同时针对次优水平ε已知与未知两种情形分析了随机算法。 |
| [^121] | [KGATE : a Knowledge Graph Embedding Training Environment](https://arxiv.org/abs/2610.09927) | 本文提出了KGATE，一个基于PyTorch Geometric和TorchKGE的模块化Python库，通过允许用户灵活组装或自定义编码器、解码器、损失函数等组件，解决了现有知识图谱嵌入库缺乏完整自编码器支持、缺乏维护和结果不可比的问题。 |
| [^122] | [Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking](https://arxiv.org/abs/2610.09919) | 本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。 |
| [^123] | [NeuralZip: Reusable Setup for Fast Lossless Compression](https://arxiv.org/abs/2610.09916) | NeuralZip 通过一次性准备并复用浮点指数的统计结构，使模型检查点的无损压缩速度较基线提升 1.81–21.33 倍，同时保证逐位精确重建。 |
| [^124] | [RollVerify: Bridging Efficiency and Accuracy in Long-Tail Rollout Reinforcement Learning](https://arxiv.org/abs/2610.09914) | 提出 RollVerify 框架，基于部分 rollout 在样本进入训练前主动验证并修复过时的离策略样本，从而在提升强化学习训练效率的同时缩小与完全在线策略训练之间的准确性差距。 |
| [^125] | [Learning joint probabilistic weather forecasts from station observations alone](https://arxiv.org/abs/2610.09898) | CLARA模型仅凭站点观测数据（无需数值天气预报或再分析数据）即可学习五个地面变量的联合高斯概率预报分布，以约2.8万参数的小模型在CPU上实现了优于各类基线的联合概率预报能力。 |
| [^126] | [Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data](https://arxiv.org/abs/2610.09889) | 该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。 |
| [^127] | [Sparsifying Stochasticity, Not Capacity: Partial Stochasticity via Deep Weight Factorization of Prior Scales](https://arxiv.org/abs/2610.09886) | 该论文提出通过对先验尺度进行深度权重因子分解来学习贝叶斯神经网络中哪些参数应保持随机性，使正则化稀疏化随机性而非模型容量，并提供了可线性时间检验的通用条件密度逼近证书，同时证明常见的采样-优化混合方案是II型最大后验目标的随机近似。 |
| [^128] | [Layerwise Error Attribution for Fast and Robust Mixed-Precision Post-Training Quantization](https://arxiv.org/abs/2610.09877) | 提出了一种基于逐层概率误差分析的混合精度训练后量化方法，通过分离传播误差与局部扰动构建可分离评分，实现了无需外部求解器的快速比特分配，并显著增强了对校准数据被污染情况下的鲁棒性。 |
| [^129] | [Think Before You Paint: Recursive Latent Reasoning for Diffusion Models](https://arxiv.org/abs/2610.09876) | 提出PaTh框架，让仅1000万参数的小型递归网络在潜在空间中进行“思考”并通过ControlNet引导冻结的扩散模型，仅靠标准重建损失训练（无需符号目标、求解器或验证器），在困难数独（92.5%）和极端数独（71.2%）视觉推理任务上大幅刷新此前最佳纪录。 |
| [^130] | [An AI-assisted conditioning and geological interpretation workflow for usage in implicit geological modeling](https://arxiv.org/abs/2610.09871) | 该论文提出了一套AI辅助工作流程，利用自监督/半监督对比学习CNN对浅层至深层（300–3500米）陆上地震数据进行降噪与插值，并以极少的人工标注数据自动解释层位与断层，从而加速隐式地质建模。 |
| [^131] | [MUNITE: Unified Multimodal Latent Inference for Any-to-Any Multimodal Generation](https://arxiv.org/abs/2610.09866) | MUNITE提出统一的潜变量框架，将编码与潜变量生成统一为同一条件流推断问题，借助共享潜在表示与基于自蒸馏的条件流匹配，实现从任意模态子集到任意模态的多模态生成。 |
| [^132] | [Global Average Precision for Representation Learning](https://arxiv.org/abs/2610.09863) | 该论文提出全局平均精度（gAP）及其可微分代理损失 gSAP，通过将所有查询-候选对纳入同一排序并联合考虑批次内全部成对比较，弥补了 mAP 和 InfoNCE 等指标与损失不考虑跨查询相似度可比性的缺陷，可作为现有损失的即插即用替代。 |
| [^133] | [DeepTopoClustering: Unsupervised Derivation of Surface Process Taxonomy from 4D Point Clouds for Topographic Monitoring](https://arxiv.org/abs/2610.09860) | 提出了一种无监督深度聚类框架DeepTopoClustering，通过将变化四维对象转化为GeoMorphogram分布序列并利用卷积自编码器与分层聚类目标学习潜在嵌入，从而从永久激光扫描点云中自动推导出地表变化过程的层次分类体系，实现地形变化监测。 |
| [^134] | [Training Advisors for LLM Agents from Task Outcomes](https://arxiv.org/abs/2610.09858) | 提出Caddie方法，通过强化学习仅以智能体最终任务成功与否作为训练信号来训练批评者提供自然语言建议，且训练后的批评者能泛化到不同规模和架构的多个基础模型并显著提升任务成功率。 |
| [^135] | [Stream-Based Active Learning with Cooperative Neural Networks for Data-Efficient Partial Inverse Design: An Automotive Glass Run Channel Case Study](https://arxiv.org/abs/2610.09848) | 该研究提出CoNN-AL框架，将基于数据流的主动学习与协同神经网络-去噪自编码器相结合，利用蒙特卡洛dropout估计预测不确定性并实时筛选最有价值的仿真样本进行标注，在汽车玻璃导槽的部分逆向设计中以远小于90多万总样本的标注量实现了数据高效的逆向设计。 |
| [^136] | [For Those Who Believe in Faithfulness: Optimizing the Area Under Insertion and Deletion Curves for Ranking Relative Feature Importance](https://arxiv.org/abs/2610.09844) | 本文从忠实性概念（插入与删除曲线下面积）出发推导出目标函数并通过随机化高效近似其梯度，同时建立了插入曲线与top-k特征选择之间的联系，从而能够直接优化特征重要性归因的质量。 |
| [^137] | [ORCA: Hunting Compositional Failures in Text-to-Image Diffusion](https://arxiv.org/abs/2610.09841) | 提出ORCA方法，将跨模态组合结构对齐作为低秩辅助损失直接融入扩散模型训练，从而解决文本到图像生成中的属性绑定错误、空间关系颠倒和多对象计数失败等组合性问题。 |
| [^138] | [Fully Interpretable Minimal Transformers: From Geometry to Algorithm](https://arxiv.org/abs/2610.09838) | 通过将Transformer的嵌入维度和注意力头大小限制为2，本文实现了内部表示的完全二维可视化，证明学习到的几何结构可直接解读为逐步执行的算法，并据此完整解析了一个完成“遇到+号输出最近偶数”任务的极简Transformer的每一步计算过程。 |
| [^139] | [Origins of Universal Machine Learning Force-Field Errors in Multicomponent Materials](https://arxiv.org/abs/2610.09837) | 该论文构建了包含7,599个多组分构型的基准测试集，系统评估了11个预训练通用机器学习力场在多组分材料中的表现，并揭示了训练参考覆盖率不足和局部几何异质性增大是力场误差的主要来源。 |
| [^140] | [Dual-QK: Sharp Queries and Flat Keys for Prunable 2-bit KV Caches](https://arxiv.org/abs/2610.09827) | Dual-QK通过成对的非正交查询与键变换，将键能量平坦化以实现2比特量化，同时将查询能量锐化集中以支持动态通道剪枝，解决了传统旋转量化中查询能量分散与剪枝需求之间的冲突。 |
| [^141] | [Homogenization in Multi-Agent Systems](https://arxiv.org/abs/2610.09824) | 本文首次揭示了多智能体系统中的智能体交互会导致行为同质化，并提出三个量化指标，证明这种同质化会在代码生成、招聘和同行评审中引发系统性漏洞、偏见持续影响和评估标准不均等具体风险。 |
| [^142] | [Backdooring Acoustic Foundation Models for Physically Realizable Triggers](https://arxiv.org/abs/2610.09819) | 本文提出FAB后门攻击方法，证明最先进的声学基础模型在实际设置下易被植入物理可实现、隐蔽且无需同步的触发器后门，这些后门能在微调后存活并在激活时显著降低下游任务性能。 |
| [^143] | [BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation](https://arxiv.org/abs/2610.09804) | BoT-GRPO通过长度不变的“词袋”聚合将token级过程奖励高效融入GRPO，无需价值网络即可作为即插即用替代方案，将收敛速度最高提升1.9倍并改善最终生成质量。 |
| [^144] | [DisParQ: Self-Supervised Part Concepts for Interpretable Vision Foundation Models](https://arxiv.org/abs/2610.09802) | DisParQ提出了一种无需类别标签和语言监督的自监督方法，通过可学习原型字典将图像块离散化为空间锚定的部件概念，并结合量化属性建模概念变化，从而实现可解释的视觉基础模型。 |
| [^145] | [A Proof-of-Concept Study of Weakly Supervised Labeling of Fine-Grained EEG Components for Artifact Attenuation](https://arxiv.org/abs/2610.09792) | 提出了一种结合频率感知高维表示与多示例学习的弱监督框架，无需昂贵的成分级专家标注，即可为脑电分离成分的细粒度子成分学习伪迹可能性得分，从而实现EMG伪迹的有效抑制。 |
| [^146] | [AdaPS-LiNGAM: Adaptive Predecessor Selection for Linear Non-Gaussian Acyclic Models under Small-Sample Settings](https://arxiv.org/abs/2610.09782) | 本文揭示了DirectLiNGAM在变量数超过样本量时残差化必然退化的结构性局限，并提出利用由图结构决定的“活动边界”子集进行自适应前驱选择的AdaPS-LiNGAM方法，以实现小样本情形下可靠的因果发现。 |
| [^147] | [Reproducible LLM Inference Benchmarking: A Sequential Isolation Protocol for Regression Testing](https://arxiv.org/abs/2610.09778) | 提出了一种顺序隔离基准测试协议，将大语言模型推理测量的变异系数从15.2%降至2.2%，实现了可复现的推理性能回归测试。 |
| [^148] | [Artificial intelligence pathways from weather to climate](https://arxiv.org/abs/2610.09770) | 该论文回顾了深度学习在天气预报领域的突破性进展，并提出将其拓展至气候预测的两项最低要求：外部强迫因子须显式进入模型以支持干预实验，且必须在极端事件和反事实轨迹等分布外情景中检验模型稳健性。 |
| [^149] | [Fluctuations of Nonlinear Observables in Mean Field Neural Network Training](https://arxiv.org/abs/2610.09768) | 本文通过在加权Sobolev空间中应用仅需普通Fréchet可微性的泛函Delta方法（无需Lions导数），证明了均场神经网络训练中的涨落会传播到有限维非线性观测量，并建立了相应的中心极限定理与显式协方差表示。 |
| [^150] | [Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving](https://arxiv.org/abs/2610.09763) | 该论文发现了自动驾驶离线强化学习中新型的“交互分布偏移”（IDS）问题——即自车候选轨迹在边缘行为分布下支持良好但与周围智能体行为联合考虑时支持不足——并提出交互约束驾驶策略（ICDP）框架来解决这一问题。 |
| [^151] | [Leaner Transformers Can Easily Learn to Cluster](https://arxiv.org/abs/2610.09760) | 本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。 |
| [^152] | [Unrolled Flow Models for Reasoning](https://arxiv.org/abs/2610.09759) | 提出通过模型自身潜在展开进行训练的展开流模型，使得积分步数的增加能够持续提升推理性能，在ProsQA上准确率达97%。 |
| [^153] | [EntroPrefill: Renyi-Guided Context Pruning with Conditional Stability Guarantees for Retrieval-Augmented Generation](https://arxiv.org/abs/2610.09757) | 该论文提出 EntroPrefill，一种由 Renyi 熵引导、带显式注意力质量约束的预填充中期上下文剪枝方法，为检索增强生成提供了可计算的 token 删除上界、自适应剪枝层下依然有效的有限样本观测保证，以及带有显式 Lipschitz 常数的条件 Transformer 扰动稳定性界。 |
| [^154] | [SoftSEEPS improves ML-based precipitation forecasting](https://arxiv.org/abs/2610.09752) | 本文提出了SEEPS评分的可微近似SoftSEEPS，使机器学习降水预报模型能够直接以该评分进行训练，且与RMSE联合优化时几乎不影响两个指标的表现。 |
| [^155] | [A Strength-Monotonic Law for Domain Alignment in Frozen-Embedding Bioacoustic Classification](https://arxiv.org/abs/2610.09737) | 该论文发现了一条强度单调定律：编码器在目标任务上越强，其跨域泛化越依赖分布对齐（MMD）项、也越受域重平衡采样的损害，据此将强编码器的跨域生物声学分类配方简化为冻结嵌入+轻量探针+交叉熵+单个MMD项+数据增强。 |
| [^156] | [Unbounded Characteristic and Universal Kernels](https://arxiv.org/abs/2610.09731) | 本文系统研究了无界核的特征性与万能性等表达能力概念，将针对有界核的成熟理论推广至无界核情形。 |
| [^157] | [Pareto-optimal quantum kernel selection for unsupervised anomaly detection on real malware beaconing data](https://arxiv.org/abs/2610.09717) | 该论文提出一种完全无监督的多目标量子核选择协议，通过同时优化无标签的异常检测质量代理指标（NPD）和与经典核的几何差异（GD），从帕累托前沿中选出量子核，并在真实恶意软件信标检测数据和IQM 20量子比特Garnet处理器上验证了其可击败调优后的经典基线。 |
| [^158] | [EC-EarthFlow: Probabilistic emulation of daily transient global climate model simulations with flow matching](https://arxiv.org/abs/2610.09715) | EC-EarthFlow是一个生成式流匹配模型，能以远低于物理模型的计算成本稳定地进行自回归模拟，准确再现全球气候模型EC-Earth3日温度场的日变率、空间格局、年循环和长期趋势。 |
| [^159] | [Boundary-aware Reinforcement Learning for Hypercube State Spaces via Deterministic Policy Gradient](https://arxiv.org/abs/2610.09712) | 本文针对超立方体上受反射随机微分方程支配的系统，建立了连续时间确定性策略梯度强化学习理论框架，并通过软惩罚或硬结构约束将诺伊曼边界条件嵌入深度确定性策略梯度算法，实现了边界感知的强化学习。 |
| [^160] | [Pretraining Shapes Spectral Structure: Architecture- and Strategy-Conditional Prediction of OOD Robustness in Foundation Models](https://arxiv.org/abs/2610.09709) | 该论文证明基础模型的分布外鲁棒性可仅由预训练权重的谱结构预测——无需任何目标数据，且这一预测由架构与预训练策略共同塑造，其代理指标的方向会随架构族不同而反转。 |
| [^161] | [Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs](https://arxiv.org/abs/2610.09703) | 本文首次全面评估了视觉语言模型中令牌剪枝机制的安全性，揭示了大多数剪枝策略随剪枝比例增加而降低安全性、而基于查询的压缩在极端剪枝下反而提升安全性的对比现象，并识别出“剪枝诱发的恶意放大”这一全新机制。 |
| [^162] | [Few-Shot Learning for Personalised Automated Pain Assessment](https://arxiv.org/abs/2610.09692) | 该研究将小样本学习应用于自动化疼痛评估的个性化，创造性地把从人群级到受试者级的评估转变重新定义为任务域偏移问题，并在三个疼痛数据库上取得了优异的个性化分类性能。 |
| [^163] | [Optimal Regret for Online Market Making with Limit Order Book](https://arxiv.org/abs/2610.09691) | 本文针对限价订单簿反馈模型下的在线做市问题，通过设计基于双耦合网格的买卖价空间离散化方法并结合Hedge算法，将遗憾界从Õ(T^(2/3))改进至最优的Õ(√T)高概率界。 |
| [^164] | [NAViLoss: An Underwater Navigation-Aware Dual-Residual Objective for Physics-Consistent Learning](https://arxiv.org/abs/2610.09690) | 该论文提出NAViLoss——一种面向水下导航的鲁棒且不确定性感知的双残差目标函数，通过联合惩罚速度估计残差并显式兼顾物理一致性与测量不确定性，改进了基于学习的AUV多普勒测速仪速度估计。 |
| [^165] | [The Silhouette Operator: Identifiability of Low-Rank Measures from One-Dimensional Projections](https://arxiv.org/abs/2610.09687) | 本文提出“轮廓算子”框架，证明了适当选取的 2k 个一维投影边缘分布足以唯一识别 $\mathbb{R}^2$ 上任何紧支撑的秩不超过 k 的符号测度，且该数量是最优的、投影方向不能任意选取。 |
| [^166] | [CERO: Where and When to Allocate Rollouts for RL Post-Training](https://arxiv.org/abs/2610.09679) | 该论文提出CERO，一种在线原始-对偶调度器，能够在整个训练周期内协调有限的rollout预算，自适应地决定选择哪些提示词、重访频率以及每轮生成的组数，为强化学习后训练的提示词准入与预算节奏控制提供了带理论保证的高效方案。 |
| [^167] | [A Multi-Source Ultrasound Benchmark Revealing the Limits of Contemporary Self-Supervised Anomaly Detection Methods](https://arxiv.org/abs/2610.09677) | 本文提出SADUSI多源超声基准数据集，涵盖广泛的解剖区域、视图和采集协议，并发现当前自监督异常检测方法（尤其是基于重建的扩散方法）在如此多样化的多源超声数据上表现不佳，从而揭示了现有方法的局限性。 |
| [^168] | [Gauss-Newton Accuracy and Indefinite Hessians: Uniform Coexistence in Low-Cost Sets](https://arxiv.org/abs/2610.09675) | 本文证明在岭正则化非线性最小二乘中，低成本集合内一致共存两种曲率状态：每个全局极小值点的海森矩阵相对误差低于 $(1+\sqrt{2})/8$，而同一集合中也存在海森矩阵不定、相对误差至少为 $15/8$ 的点，并给出逐点证书与尖锐的相对误差界。 |
| [^169] | [DSTNet: Dynamic Spectral Trajectory Network for Causal Multi-Horizon Financial Forecasting](https://arxiv.org/abs/2610.09654) | DSTNet通过构建因果动态频谱轨迹并采用尺度-时间频谱Transformer与CNN-BiLSTM门控融合的架构，在单次前向传递中同时输出1、3、5、10天的多期金融预测，且严格保证不泄露未来信息。 |
| [^170] | [Closed-Form Noise Calibration Against Membership Inference for Random-Allocation DP-SGD](https://arxiv.org/abs/2610.09651) | 本文针对随机分配的DP-SGD提出了一个单行闭式公式，可在微秒级时间内直接计算出抵御任何成员推断攻击所需的噪声量，且所需噪声最多仅为现有最先进闭式界的一半。 |
| [^171] | [When Rank Rises as LLMs Degrade](https://arxiv.org/abs/2610.09647) | 该研究发现LLM后训练中的表示退化（如数据重复）会使RankMe等谱秩指标不降反升，导致单边监控将最差模型误判为最健康，因此指标变化方向取决于具体的退化模式与统计量配对，传统监控假设并不安全。 |
| [^172] | [Q-PhotoMarket: A Design Space Exploration Framework for Photonic Hybrid Quantum Neural Networks in Financial Market Prediction](https://arxiv.org/abs/2610.09641) | 该论文提出了Q-PhotoMarket框架，通过系统性探索超过5,000种光子混合量子神经网络配置，首次全面考察了光子电路设计选择对金融市场预测性能的影响。 |
| [^173] | [Tracing Inputs, Verifying Outputs: Validating Attribution in Music Generation](https://arxiv.org/abs/2610.09637) | 该论文提出通过仅基于音频条件生成、输入追踪以及音乐版本识别模型musicDNA审计记忆化，为AI音乐生成中“哪些音频来源被使用并影响了输出”提供了可验证的归因证据。 |
| [^174] | [Quantum anomaly detection in real scarce data](https://arxiv.org/abs/2610.09635) | 针对小型不平衡数据集上异常检测的难题，本文提出了一种新颖的两步混合经典-量子架构，利用量子机器学习以更少的可训练参数和更小的数据集实现更可解释、更节能的异常检测。 |
| [^175] | [Coding-Agent Benchmarks Should Match Their Users' Task Flows](https://arxiv.org/abs/2610.09633) | 该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。 |
| [^176] | [How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression](https://arxiv.org/abs/2610.09624) | 该研究通过构造仅由单个请求动词即可翻转工具调用决策的最小对比提示对，揭示了智能体大语言模型内部存在一个由抑制机制塑造的“工具调用向量”，它介导了模型在调用工具与直接回答之间的决策。 |
| [^177] | [When does a network's training history predict its future learning better than its current state? Evidence from a response probe and a forecasting screen](https://arxiv.org/abs/2610.09621) | 该研究设计并严格校验了一种“响应探针”方法，用以探究在何种条件下网络的训练历史比其当前状态更能预测其未来的学习能力。 |
| [^178] | [The Identifiability and Observability of Deep Normalized Attention](https://arxiv.org/abs/2610.09620) | 本文证明了对于实解析归一化函数（如softmax），深度归一化注意力的输入-输出函数在一般情形下可确定其有效得分与值映射（至符号差异），并精确刻画了参数坍缩导致得分不可观测的条件及雅可比衰减谱。 |
| [^179] | [Second-order optimization of variable projection SVM models and road abnormality detection](https://arxiv.org/abs/2610.09617) | 本文提出了一种针对变投影泛函的新型二阶优化框架，利用二阶信赖域算法高效训练变投影支持向量机（VP-SVM），并成功应用于基于轮胎传感器一维信号的道路异常检测。 |
| [^180] | [Residual Learning in Empirical Asset Pricing](https://arxiv.org/abs/2610.09613) | 本文提出将残差学习应用于实证资产定价，使神经网络模型在保留浅层模型的基础上得以加深，深度残差模型的样本外夏普比率（2.07）显著优于浅层模型（1.92）和深度前馈模型（0.89），证明模型深度是额外经济价值的来源，并为构建“大型资产定价模型”提供了可行路径。 |
| [^181] | [Sequential Pretraining Favors Large Models](https://arxiv.org/abs/2610.09611) | 该论文定义并量化了“首因偏差”，揭示小模型会将学习容量低效地分配给早期训练数据而大模型对此具有鲁棒性，因此当数据按顺序用于预训练时大模型更具优势，尤其是当后期数据强调代码、数学和推理能力时。 |
| [^182] | [Decoupled Optimization for Teacher-Student Semi-Supervised Learning via a Pioneer Student](https://arxiv.org/abs/2610.09609) | 提出先锋学生（PiS）这一独立参数空间的辅助分支，通过周期性回传知识，解耦了师生强同步与有/无标签损失联合优化导致的两大优化病态。 |
| [^183] | [Lightweight and Versatile Learned Optimization by Recombination of Gradient History](https://arxiv.org/abs/2610.09604) | 本文提出一种轻量级学习优化器，通过动态重组不相交时间跨度的梯度历史平均，仅用37k参数和0.87 GPU小时训练即可零样本泛化到NLP、视觉和图模型等多种任务，显著优于Adam且FLOPs开销低至0.3%。 |
| [^184] | [COPC: Coupled Off-Policy Correction for Asynchronous LLM Reinforcement Learning](https://arxiv.org/abs/2610.09597) | 该论文指出异步强化学习中仅靠策略侧的重要性比率修正无法解决“优势过时性”问题，并提出将策略侧与优势侧修正协同进行的耦合式离线策略修正方法COPC。 |
| [^185] | [Scalable Logistic Gaussian Process Density Regression with Kinetic Langevin Sampling](https://arxiv.org/abs/2610.09591) | 本文提出一种基于逻辑高斯过程的可扩展贝叶斯条件密度估计方法，通过在强对数凹后验上模拟动力学朗之万动力学进行直接采样，并利用Nyström特征支持具有依赖输入参数的非平稳核。 |
| [^186] | [Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation](https://arxiv.org/abs/2610.09587) | 该论文提出协同推理蒸馏框架 CRD，结合教师间交叉反馈、与答案无关的逐步质量评估和连贯性步骤拼接，并通过带预算约束的推理质量优化训练学生模型，使 CRD-4B 仅用 5 万条训练数据便在 MATH-500 和 AIME'25 上超越基线。 |
| [^187] | [Online Resource Allocation with an Endogenous Markov State: Fewer LP Solves Earn More](https://arxiv.org/abs/2610.09577) | 本文揭示了内生马尔可夫状态下在线资源分配中重新求解线性规划频率与遗憾之间的反直觉关系：在非退化条件下低频重新求解即可达到常数遗憾，而在最优解退化时频繁重新求解反而可能导致线性遗憾，即更少的LP求解反而赚得更多。 |
| [^188] | [PEACE: Covariant learning of nonadiabatic manifolds with parity-resolved Hamiltonians](https://arxiv.org/abs/2610.09576) | PEACE通过宇称等变哈密顿量与学习的电子连接相结合，精确再现了非绝热分子动力学中的交叉结构与弛豫动力学，并可扩展至自旋轨道耦合以模拟系间窜越。 |
| [^189] | [EvoSignal: LLM-Guided Evolutionary Design of Modular Traffic Signal Control Programs](https://arxiv.org/abs/2610.09563) | EvoSignal提出将交通信号控制建模为模块化程序设计问题，利用大语言模型引导的进化框架，通过交通知识与性能反馈自动演化出透明、低成本且性能更优的信号控制程序。 |
| [^190] | [UniCSI Towards a Universal Wi-Fi CSI Encoder for Ubiquitous Human Sensing](https://arxiv.org/abs/2610.09559) | 提出了UniCSI统一基础架构，通过物理信息引导的RF分词器等核心创新，直接处理来自不同设备配置的异构Wi-Fi CSI数据并保持原生波形完整性，实现普适的人体感知。 |
| [^191] | [Constitution-Guided Watermarking](https://arxiv.org/abs/2610.09552) | 本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。 |
| [^192] | [A Framework for the Systematic Review of ML Assets in AI Registries](https://arxiv.org/abs/2610.09551) | 本文提出一个框架，将科学文献中的系统性综述方法适配到AI注册库中机器学习资产（预训练模型、数据集、基准等）的检索与选择中，使选择过程透明、可复现且有据可循。 |
| [^193] | [CircuitGate: Logic-Consistent Circuit-Level Functional Modeling for And-Inverter Graphs](https://arxiv.org/abs/2610.09549) | 该论文提出CircuitGate框架，通过显式编码全局主输入支持集与扇入间的再汇聚关系，将AIG表示学习从门级语义提升到电路级功能建模，克服了GNN局部消息传递带来的电路级上下文缺失和拓扑敏感性问题。 |
| [^194] | [Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets](https://arxiv.org/abs/2610.09541) | 该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。 |
| [^195] | [Unpaired Canonical Correlation Analysis](https://arxiv.org/abs/2610.09530) | 提出UCCA方法，通过建立二次分配问题与CCA之间的理论联系，首次实现仅使用非配对数据学习最大化真实潜在配对相关性的线性投影。 |
| [^196] | [Align Before You Combine: Reference Space Calibration for Supervision Without Ground Truth](https://arxiv.org/abs/2610.09525) | 提出一种无需真实标签的校准优先框架，通过合成有序参考空间在融合前对齐各子集评分器，其校准独立于训练集分布，在多个基准上优于未校准平均和最佳单个评分器。 |
| [^197] | [Reflected Anchored Langevin Algorithms](https://arxiv.org/abs/2610.09522) | 本文提出反射锚定朗之万动力学（RALD）及其蒙特卡洛算法 RALMC，通过光滑锚定参考势能与状态相关缩放因子，突破了传统方法要求对数密度可微的限制，实现了约束域上不可微目标分布的高效采样，并给出了显式的收敛界与迭代复杂度。 |
| [^198] | [Differential Refresh Policies for Models Trained on Lagging Data Snapshots: From a Single-Age Equivalence Limit to an Optimal Per-Segment Allocation](https://arxiv.org/abs/2610.09519) | 本文证明任何仅基于单一全局数据年龄的刷新触发器在操作上都等价于简单的均匀计时器，并提出按分段差异化赋予各自年龄与刷新间隔的策略，在刷新成本可分离的条件下实现最优的刷新资源分配。 |
| [^199] | [It Is Not Seeing the Hazard: A Frozen Vision-Language Safety Score Measures Its Caption Bank](https://arxiv.org/abs/2610.09517) | 本文通过受控评估证明，冻结的CLIP安全评分并未真正检测到危险本身，而主要是对其描述文本库以及提示结构、嵌入几何和相机视角等混杂因素作出反应。 |
| [^200] | [Physics-Informed Neural Plasticity: PDE Solvers That Reshape Themselves](https://arxiv.org/abs/2610.09510) | 提出物理信息神经可塑性范式及ReCAP求解器，使PDE求解器在优化过程中根据未解决的物理问题动态重塑自身表示结构，通过局部加密、分裂、剪枝和合并自适应地重新分配模型容量。 |
| [^201] | [TERRA: Learning Transportable Latent Actions through Temporal Effect Representation and Relational Alignment](https://arxiv.org/abs/2610.09509) | 提出TERRA框架，用紧凑的时间效应表征（净特征变化加低阶窗口内动力学）学习连续潜在动作，并通过效应锚定迁移（EAT）使潜在动作能在不同初始状态间迁移复用。 |
| [^202] | [Safe on Average, Unsafe in the Tail: When Is the Episodic-Cost Tail Controllable?](https://arxiv.org/abs/2610.09508) | 本文提出用CVaR₀.₁度量回合成本尾部风险，揭示了仅满足平均成本约束的安全强化学习策略在最差回合中可能不安全，并研究在保持回报的前提下这种尾部违规何时可控。 |
| [^203] | [Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins](https://arxiv.org/abs/2610.09505) | 本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。 |
| [^204] | [SpatialUQ: Post-Hoc Uncertainty Quantification from Spatial Consistency in Black-Box Vision Models](https://arxiv.org/abs/2610.09498) | SpatialUQ通过仅六次前向传播测量全局预测与固定空间裁剪区域之间的Jensen-Shannon散度，在无需访问模型内部的情况下，以MC-Dropout五分之一的计算量将ChestX-ray14上的失败检测AUC从0.664提升至0.784，并提供更优的校准性能。 |
| [^205] | [Sparse Feature Policy Unlearning Mitigates State Hallucination in Vision-Language-Action Models](https://arxiv.org/abs/2610.09496) | 提出SOUL方法，利用稀疏自编码器识别与状态幻觉相关的稀疏特征，并据此选择性地遗忘VLA模型中与幻觉行为相关的策略知识，从而有效减轻视觉-语言-动作模型中的状态幻觉问题 |
| [^206] | [Correspondences as Decisions: JevNexus for Decision-Centric Schema Matching](https://arxiv.org/abs/2610.09487) | JevNexus通过类型化成对决策结合证据门控机制，仅在证据不一致且边际较小时才调用昂贵的列表级精炼，在模式匹配精度与最先进方法相当的同时将延迟降低约7.75倍。 |
| [^207] | [CHASE: Channel-Aligned Structure Exploitation for Geometry-Aware Model Engineering](https://arxiv.org/abs/2610.09476) | 本文提出CHASE框架，将几何与谱对齐（GSA）所刻画的结构特征应用于参数高效微调、剪枝补偿、模型合并、KV共享表示和神经元分组等六类模型工程任务，并开发了CAGA、SAKV、CAPS三种新方法。 |
| [^208] | [MORA: Modeling Observed Changes for Drift-Robust Time-Series Anomaly Detection](https://arxiv.org/abs/2610.09473) | MORA提出了一种漂移鲁棒的时间序列异常检测框架，通过从成对的短期与长期视图重建同一局部目标，利用重建差距进行“时间变化消歧”，并以保守的修正机制调整异常分数，从而区分真实异常与分布漂移。 |
| [^209] | [When Should an In-Context Learner Expand Its Hypothesis Space?](https://arxiv.org/abs/2610.09471) | 该论文将上下文学习中“何时扩展假设空间”的问题形式化为有代价的序贯决策，并通过结构修正环境证明：修正决策是一个由价格、时间范围和查询等价值因素决定的价值边界，而非单纯的证据阈值。 |
| [^210] | [An Invariant Tangent-Angle Descriptor and a Band U-Net for 2D Fragment Adjacency Prediction](https://arxiv.org/abs/2610.09459) | 该论文提出了一种在构造上对碎片旋转和轮廓起始点选择均不变的切线角描述子来替代原有的局部评分，并用带状U-Net替换最终分类器，从而改进了基于轮廓的二维碎片邻接预测的两阶段架构。 |
| [^211] | [DSReg: Provably Recovering Individual World Latents without Reconstruction](https://arxiv.org/abs/2610.09457) | 该论文提出“结构多样性”条件与依赖稀疏正则化方法DSReg，在无需重建、解码器或标签的情况下，可证明地恢复个体的世界潜在变量。 |
| [^212] | [GeoPrior-Mamba: Structured Process Priors with Mamba for Fine-Resolution XCO2 Reconstruction](https://arxiv.org/abs/2610.09456) | 提出GeoPrior-Mamba框架，创新性地利用离线语言模型将生物圈吸收、生态系统呼吸和人为排放的过程知识转化为结构化先验表，并结合多方向Mamba架构，实现从稀疏卫星观测中重建精细分辨率的XCO2场。 |
| [^213] | [An extended deep energy method for thermo-mechanical crack propagation](https://arxiv.org/abs/2610.09433) | 提出一种扩展深度能量方法，用两个神经网络表示温度场与位移场，并通过不连续标量嵌入函数嵌入尖锐折线裂纹、以可训练Williams展开富集裂尖位移，首次将含裂纹域的瞬态热传导与热应力驱动的裂纹扩展统一求解。 |
| [^214] | [What a Reporting Convention Hides: A Matched-Budget Audit of Quantum Natural Gradient with an Exactly Computed Metric](https://arxiv.org/abs/2610.09425) | 该论文通过对量子自然梯度采用精确计算度量并进行等预算的公平审计，揭示了“只计成功运行时间或仅以单一目标读取结论”这两种常见报告惯例会掩盖 Adam、SPSA 与 QNG 之间的真实性能差距，甚至可能颠覆对变分量子优化器的比较结论。 |
| [^215] | [Democratizing MoE inference on commodity GPUs with CoMoE](https://arxiv.org/abs/2610.09424) | CoMoE通过新颖的以主机为中心的路由机制，将主机转变为主动路由枢纽，解决了消费级GPU互连带宽受限的通信瓶颈，使MoE模型推理能够在低成本的商品级GPU上高效部署。 |
| [^216] | [Self-Consuming Generative Models with Co-Evolving Human Preferences](https://arxiv.org/abs/2610.09415) | 该研究首次分析了自消耗生成模型中模型分布与人类偏好共同演化的动力学，证明完全依赖用户筛选数据会放大初始偏差并导致优势实例垄断，而注入足够比例的参考数据则能使系统收敛到唯一的全局平衡。 |
| [^217] | [Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data](https://arxiv.org/abs/2610.09411) | 提出一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化估计单一正交映射，无需任何配对数据即可实现独立训练模型的跨模态表征对齐，并证明标准几何对齐指标可准确预测对齐的可行性。 |
| [^218] | [Stability and Diversity of Networked Self-Consuming Generative Ecosystems](https://arxiv.org/abs/2610.09409) | 本文首次将多个生成模型间的合成数据流动建模为有向加权图，提出了一个理论框架来系统分析网络化自消耗生成生态系统的长期稳定性与多样性。 |
| [^219] | [Noise, Denoise, Correct: MCMC Posterior Sampling with Diffusion Priors in Three Steps](https://arxiv.org/abs/2610.09407) | 该论文提出扩散圆舞曲方法，以SDEdit式加噪-去噪作为提议分布并结合Metropolis-Hastings校正，实现无需先验评估的精确MCMC后验采样，并可通过无梯度集合卡尔曼更新注入观测信息，在非线性、不可微的Navier-Stokes初始条件恢复任务上优于现有基线。 |
| [^220] | [The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs](https://arxiv.org/abs/2610.09384) | 该论文提出人格层级模型，指出大语言模型微调后行为的泛化范围取决于修改的是共享默认人格还是局部人格，且泛化狭窄程度与训练情境人格和默认人格的相似度呈正相关。 |
| [^221] | [From Plausible Hierarchies to Useful Taxonomies: Evaluating Agentic Harnesses on Customer Feedback](https://arxiv.org/abs/2610.09377) | 该研究提出评估AI生成的分类体系何时才真正适用于生产环境，发现尽管所有生成的层级结构都能通过通用的命名与结构检查，但仍需更深层次的产品覆盖评估才能判断其实际实用性。 |
| [^222] | [From Retrieval to Customer Context: Evaluating Frontier-Model Systems for Voice-of-Customer Analysis](https://arxiv.org/abs/2610.09375) | 该论文提出“客户情境图”这一统一建模框架，将客户反馈与业务运营及分析对象通过类型化关系连接，使智能体不仅能分析客户说了什么，还能理解为什么、谁受影响、如何解决等深层情境，并在 9,432 条公开 Cursor 反馈上对比评估了三种前沿模型系统。 |
| [^223] | [Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment](https://arxiv.org/abs/2610.09369) | 提出互不相溶扩散策略，一种无需标签的训练时噪声分配方法，通过保持噪声到动作路径的相对分离，防止扩散策略在低维机器人动作空间中坍缩为单一模态，从而保留多模态动作分布。 |
| [^224] | [Closing the Loop on Contrail Avoidance with Satellite Verification](https://arxiv.org/abs/2610.09363) | 该研究构建了一个仅840万参数、单GPU训练的小型扩散模型，可在卫星图像中检测仅一到两个像素宽的飞机凝结尾迹（PR-AUC达0.476），为“改航规避尾迹—卫星验证”闭环提供了可行的技术路径，并总结出分辨率与简单数据增广比新架构更重要的普适经验。 |
| [^225] | [Global Exponential Convergence of Two-Layer Linear Network Training](https://arxiv.org/abs/2610.09356) | 该论文证明了采用光滑PL预测损失训练的宽两层线性网络的全局指数收敛性，其动力学可精确刻画为神经元协方差的有限维Bures流，并给出显式收敛速率（初始协方差为σ²Id时至少为4σ²κ），且该速率在有限宽度采样下保持稳定。 |
| [^226] | [Multimodal LLMs Can Learn to Read Brain Signals: A Vision--Language Model for Unified Multi-Task EEG Decoding](https://arxiv.org/abs/2610.09355) | BraVista将多通道EEG信号编码为结构化图像，通过对通用视觉-语言模型进行持续后训练，在无需大规模EEG专用预训练的情况下实现了跨数据集的统一多任务脑电解码。 |
| [^227] | [Learning Unknown Constraints without Unsafe Data via Optimality and Counterfactual Regularization](https://arxiv.org/abs/2610.09350) | 提出CF-KKT约束学习框架，通过反事实KKT正则化利用学习到的动力学和局部最优专家演示来恢复未知约束，无需已知动力学、结构化约束表示或存在安全风险的数据探索。 |
| [^228] | [OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models](https://arxiv.org/abs/2610.09346) | OnlineQAT提出了一种两阶段框架，先通过分块QAT获得低比特初始化，再利用冻结的全精度教师模型在学生自身生成的回复上进行同策略蒸馏，从而在2-3比特的超低比特量化下显著恢复大语言模型的精度并超越现有离线QAT方法。 |
| [^229] | [Shared Low-rank Basis Factorization for Data-free Mixture-of-Experts Compression](https://arxiv.org/abs/2610.09342) | 本文提出共享低秩基分解（SLBF），一种无数据的MoE压缩权重重构方法，通过让专家共享秩-k低秩基实现更低的重构误差和更快的收敛，并从理论上证明了剪枝和合并类方法存在不可消除的结构性误差。 |
| [^230] | [Benign Overfitting under Heterogeneous Input Fusion](https://arxiv.org/abs/2610.09340) | 该论文首次研究异构输入融合下的良性过拟合，发现回归中存在一个与截断阈值无关的全谱协方差证书，可保证良性性质在任意一致的联合协方差融合下得以保持，但这种保护是精确的——在证书范围之外，两个良性的边缘输入块融合后可能变得有害。 |
| [^231] | [Ask the Expert: LLM-Guided Reinforcement Learning for Autonomous Cyber Defense](https://arxiv.org/abs/2610.09337) | 该论文提出“询问专家”训练时引导框架，在训练过程中利用LLM将困难防御状态的应对建议转化为分层奖励塑形以提升PPO的样本效率，训练结束后LLM被丢弃，部署时仅需纯强化学习策略。 |
| [^232] | [SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models](https://arxiv.org/abs/2610.09335) | SearchWorld提出了一种将显式空间记忆与价值引导想象相结合的循环状态空间世界模型，通过前瞻性想象推演解决部分可观测性难题，实现无人机在城市环境中的高效目标搜索。 |
| [^233] | [VM-ARRAYDPS: Virtual Microphone Augmented Diffusion Posterior Sampling for Unsupervised Blind Speech Separation](https://arxiv.org/abs/2610.09334) | 本文提出VM-ArrayDPS方法，通过引入虚拟麦克风来增强扩散后验采样中的多通道一致性目标，从而突破实际麦克风阵列数量有限对无监督盲语音分离性能的制约。 |
| [^234] | [Visual Jev Rewards: Reference-Bound Verification for Multi-Subject Image Generation](https://arxiv.org/abs/2610.09328) | 该论文提出了参考绑定的视觉Jev奖励方法，通过二值监督训练的Qwen3.5-4B验证器，仅在请求条件与参考主体身份同时成立时给予正奖励，以此生成GRPO训练信号，显著提升了多主体图像生成中主体交互验证的准确性。 |
| [^235] | [Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization](https://arxiv.org/abs/2610.09311) | 提出分支潜在扩散，通过将1024个词元压缩为64个块潜在表示（16倍压缩），并由并行运行的局部自回归分支进行词元解码，大幅提升扩散语言模型的生成效率。 |
| [^236] | [MovieSTAGE: Scene, Transition, and Global Encoding for Movie-fMRI ADHD Classification](https://arxiv.org/abs/2610.09306) | 提出MovieSTAGE多尺度框架，通过融合场景内超图功能连接、相邻场景间转换差异和整部电影功能连接三种表征，显著提升了基于自然电影fMRI的ADHD分类性能。 |
| [^237] | [Kuration SDK: Addressing the Virtual2Real Gap via Data Curation](https://arxiv.org/abs/2610.09305) | 该论文发现现有基准指标（FVD、LPIPS、JEDi）与世界模型的定性可玩性不对应，即存在“虚拟到真实”差距，并提出在训练前通过数据策展策略和开源的Kuration SDK工具包测量数据诊断属性，以更稳健地弥合这一差距。 |
| [^238] | [emg2face: Expressive Facial Animation with High-Density Surface EMG](https://arxiv.org/abs/2610.09304) | 该论文提出利用高密度表面肌电信号（HD-sEMG）作为非光学替代方案，在被VR头显等设备遮挡面部的情况下，从64通道肌电数据实现表现性的面部动画重建。 |
| [^239] | [Node-level Graph Neural Architecture Search Framework](https://arxiv.org/abs/2610.09297) | 提出节点级图神经架构搜索算法N-GNAS，可在更新节点特征时为每个节点子集自动选择合适的网络架构，并借助对比学习损失缓解过度平滑问题。 |
| [^240] | [RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment](https://arxiv.org/abs/2610.09294) | 提出RT-SAFE，一个在实时约束下评估具身智能体安全性的仿真城市基准，强调在物理世界持续演化的情况下，决策质量与决策延迟同等重要。 |
| [^241] | [LeCuration: A Tiny World Model as a Data Curation Multi-Tool](https://arxiv.org/abs/2610.09285) | 该论文提出LeCuration——一个面向封闭物理世界的微型世界模型，其潜空间嵌入可同时用作异常检测信号与内容聚类依据，并能自回归预测游戏状态以供定性检查，从而为更大的下游物理AI模型提供多功能的数据整理工具。 |
| [^242] | [Self-attention summary networks for subsurface velocity-model building from common-image gathers](https://arxiv.org/abs/2610.09282) | 该论文提出一种多尺度自注意力摘要网络，将高维共成像点道集压缩为保留运动学结构的条件嵌入，用于驱动流匹配模型进行概率性地下速度反演，从而显著改善后验速度推断效果。 |
| [^243] | [Twist Flow for Inverse Problems](https://arxiv.org/abs/2610.09281) | 该论文提出联合扭转流这一增广流匹配方法，通过学习增广源状态与增广终端状态之间的连续输运，克服贝叶斯逆问题中确定性映射导致的后验变异性低估和多峰模式失真问题。 |
| [^244] | [CATune: Structural Constraint-Aware Bayesian Optimization for DBMS Configuration Tuning](https://arxiv.org/abs/2610.09276) | CATune提出了一种结构约束感知的贝叶斯优化框架，将DBMS配置参数间的确定性顺序约束直接建模为搜索域的结构组成部分，在约束一致的子空间内进行优化，从而提升数据库配置自动调优的效率与有效性。 |
| [^245] | [Hardware-aware Calibrated Clustered Attention for Efficient Visual Geometric Transformers](https://arxiv.org/abs/2610.09274) | 提出硬件感知的分块聚类注意力（BC attention），通过块内聚类、哈希超平面校准和基于阈值的误差补偿来加速VGGT的全局注意力层，在长序列场景下实现GPU上实际的延迟提升。 |
| [^246] | [Evaluating Trajectory Features for Routing Final-Layer Attention](https://arxiv.org/abs/2610.09272) | 该论文系统评估了外推误差、曲率等轨迹特征用于最后一层注意力路由的效果，发现这些特征在所有预设比较中均未带来显著增益，甚至不及简单的固定投影基线，表明轨迹特征对注意力路由的预测价值有限。 |
| [^247] | [The Symbol of the Surrogate: Measuring Numerical Provenance in Neural PDE Solvers](https://arxiv.org/abs/2610.09255) | 该论文提出一种基于傅里叶符号的经验诊断方法，用于判定神经PDE代理模型究竟忠实于精确物理演化还是仅仅模仿训练数值求解器的离散化误差，并发现代理模型几乎完全复制（超过99.8%）了训练格式的振幅与相位误差特征。 |
| [^248] | [Efficient Best-of-N policy evaluation for inference-time alignment](https://arxiv.org/abs/2610.09250) | 本文提出了一种无需访问响应似然值的仅样本BoN策略评估框架，利用BoN的顺序统计结构将密度比转化为可由样本估计的得分排名概率，并开发了能跨候选预算高效重用共享辅助样本池的双重稳健估计器BoN-DR，在奖励模型误设下仍保证有效的渐近推断。 |
| [^249] | [An Informational Curse of Horizon in Goal-Conditioned Policy Learning](https://arxiv.org/abs/2610.09247) | 该论文发现目标条件策略学习中存在一种新的“信息性视界诅咒”：训练时目标重标注视界越长，行为克隆策略即使仅面对近处子目标也会出现严重性能退化，而强化学习目标可以缓解这一问题。 |
| [^250] | [Beyond Nominal Equilibria: Risk-Averse Multi-Population Mean-Field Games](https://arxiv.org/abs/2610.09244) | 该论文提出了风险厌恶多群体平均场博弈这一新范式，让每个群体对其他群体平均场流的模糊集进行最坏情况优化，并利用占据测度表述与集值分析工具证明了新型风险厌恶均衡的存在性等理论性质。 |
| [^251] | [The Winner's Curse in LLM Self-Improvement Loops: Selection Noise, Lock-in, and Acceptance Rules](https://arxiv.org/abs/2610.09239) | 该论文将LLM自我改进循环中的“更优则保留”步骤建模为测量噪声下的选择问题，揭示并量化了胜者诅咒现象：首轮之后的大多数自我修改实为有害，且复用小评估集会使选择集得分严重高估真实泛化能力（小选择集时高估达13至20个百分点）。 |
| [^252] | [Symmetry-Informed Causal Partial Identification](https://arxiv.org/abs/2610.09230) | 该论文首次将已知的数据对称性（即因果效应在某些数据变换下的不变性）作为部分识别的新约束来源，通过因果函数形状约束和测度变换两种方法有效收紧了因果效应的识别边界。 |
| [^253] | [Conditional Accuracy Profiles: Diagnosing LLM Judges across Deployment Conditions](https://arxiv.org/abs/2610.09229) | 提出CAP诊断框架，将LLM评判器的准确率分解为内容敏感性、鲁棒性和理由质量三类的八个条件，从而揭示被总体准确率数字掩盖的部署条件差异。 |
| [^254] | [Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders](https://arxiv.org/abs/2610.09227) | 该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。 |
| [^255] | [FLoRa: Flight-Assisted Data Collection from Duty Cycling LoRa Nodes under Energy Constraints](https://arxiv.org/abs/2610.09226) | FLoRa提出了一种融合模拟退火路径规划、CMA-ES悬停定位和POMDP探测决策的多层次无人机数据收集架构，并引入VIP指标，用于在能量约束下从占空比休眠的LoRa物联网设备中高效收集数据。 |
| [^256] | [Consistent Distribution Matching for Data-Free Diffusion Distillation](https://arxiv.org/abs/2610.09221) | 提出一致分布匹配方法，通过单一学生网络统一样本生成与分数估计，仅需一个冻结教师模型和一个可训练学生模型优化单一目标，即可实现无需模拟、无需数据的扩散与流模型加速蒸馏，并证明了学生分布向教师边缘分布的Wasserstein收敛性。 |
| [^257] | [CM-DPO: Constraint-Margin Direct Preference Optimization for LLM Planning](https://arxiv.org/abs/2610.09219) | 本文提出CM-DPO，用基于符号验证器、按违例严重程度缩放的连续约束边际取代DPO的二值偏好信号，并通过字典序目标区分硬软约束，结合SynPlan-R框架生成低偏差偏好数据，显著提升了8B模型在TravelPlanner、NaturalPlan和PlanBench等规划基准上的表现。 |
| [^258] | [CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search](https://arxiv.org/abs/2610.09212) | 提出CurveTQ，将Hessian矩阵LDL分解的对角线权重嵌入Viterbi分支度量，实现无需旋转的曲率加权格状量化，性能可媲美基于随机正交旋转的现有最优二比特量化方法 |
| [^259] | [An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information](https://arxiv.org/abs/2610.09206) | 论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。 |
| [^260] | [Few Bits, One Law: Toward W2A4KV2](https://arxiv.org/abs/2610.09202) | 提出统一的量化感知训练框架CanonQ，通过源规范化与任务感知适配相分离，实现权重2比特、激活4比特、KV缓存2比特（W2A4KV2）的联合极端低比特压缩。 |
| [^261] | [FreeEvolve: Learning to Evolve Beyond Fixed Loops](https://arxiv.org/abs/2610.09197) | FREEEVOLVE 通过元进化将“如何优化”本身变成从经验中学到的能力，让进化器摆脱人工设计的固定搜索循环，自主决定测试什么、收集多少证据、保留哪些候选方案以及何时停止。 |
| [^262] | [Exact Dynamics and Finite-Sample Trajectory Recovery of Linear Recursive Feature Machines](https://arxiv.org/abs/2610.09196) | 本文将线性递归特征机与迭代重加权最小二乘的联系推广到岭正则化含噪多输出回归，并证明了其学习到的特征矩阵在每次迭代中都以 $O(\sqrt{d/n})$ 的误差速率逼近无限数据下的理想结果。 |
| [^263] | [Patient, Place, Prior (P$^3$): What Counts as Personalization in Medical World Models?](https://arxiv.org/abs/2610.09194) | 该论文提出了P³审计框架来检验医学世界模型的预测是否真正实现了患者个性化，并提出Cancer JEPA单步模型，通过在患者条件化降秩回归基线上加入基于遮挡潜在目标训练的病灶约束神经校正，来预测新辅助治疗中未来乳腺MRI检查的冻结表征。 |
| [^264] | [The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning](https://arxiv.org/abs/2610.09186) | 论文提出模式识别与逐步推理是同一光谱的两端：当下一个词元仅依赖少量前文时 LLM 学会逐步推理，任务的推理轨迹构成 De Bruijn 图的有向无环子图，且其边数远小于轨迹数，因此 LLM 可以通过组合少量已学习的边来解决更长的新任务。 |
| [^265] | [Q-PACE: Dynamic Precision Allocation for Quantization-Aware Training](https://arxiv.org/abs/2610.09183) | Q-PACE提出一种基于二阶曲率敏感度的动态混合精度分配方法，在量化感知训练过程中定期测量各层敏感度并重新分配精度，从而在大幅降低计算成本的同时保持模型性能。 |
| [^266] | [LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition](https://arxiv.org/abs/2610.09179) | 该论文发现Transformer隐藏状态范数随深度的增长并非病态，而是一种由归一化权重γ承载的涌现式深度位置编码，并据此提出LayerRoPE，用共享向量与深度条件标量替代逐层γ向量，在减少参数的同时保持性能。 |
| [^267] | [Context-aware Attention-based Gaussian Mixture Models for Vehicular Trajectory Prediction](https://arxiv.org/abs/2610.09174) | 本文提出CAA-GMM模型，通过上下文感知的注意力机制与可解释的高斯混合建模，实现多模态且不确定性感知的车辆轨迹预测，在nuScenes和Argoverse 2数据集上以较低的计算复杂度达到了与最先进方法相当或更优的精度。 |
| [^268] | [Lower Bounds for Parallel Diffusion Sampling](https://arxiv.org/abs/2610.09166) | 本文首次建立了带近似分数的扩散采样的多项式并行轮数下界，证明了对 $R^d$ 中平滑近各向同性高斯混合的采样需要 $\widetilde{\Omega}(d^{1/3})$ 轮、对单位球内各向异性轴对齐盒子的均匀采样需要 $\Omega(d)$ 轮，且这些下界对每轮可进行多项式次查询的任意随机算法均成立。 |
| [^269] | [Sampling SU(N) gauge theory on a 2D lattice from independent plaquettes via holonomies and corner reweighting](https://arxiv.org/abs/2610.09147) | 该论文提出用和乐变量结合角点重加权，把二维 SU(N) 晶格规范理论的采样问题归结为在给定群交换子约束下从独立方格出发的条件采样，从而绕开了从方格映射回链变量这一生成式采样方法的难点。 |
| [^270] | [Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models](https://arxiv.org/abs/2610.09145) | 在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。 |
| [^271] | [A Cognitive-Aware QML-CRL Framework for Detecting Affinity and Romance-Investment Fraud](https://arxiv.org/abs/2610.09141) | 该论文提出一种量子机器学习与经典强化学习混合框架，通过专用量子比特编码诈骗中的认知偏差、操纵性重构的时间顺序和叙事共现信息，并将逐轮对话标记决策建模为最优停止问题，从而实现对亲和诈骗与浪漫投资诈骗的检测。 |
| [^272] | [Directional Evidence Guided Search-Space Reduction for Exact DAG Learning](https://arxiv.org/abs/2610.09136) | 提出了一种名为DECO的非参数混合框架，通过从观测数据中提取依赖性和方向性证据预先构建可采纳父节点集，从而在精确DAG学习中获得搜索空间的指数级缩减。 |
| [^273] | [The Impact of Likelihood Tempering on the Limiting Predictive Moments of Variational Bayesian Linear Neural Networks](https://arxiv.org/abs/2610.09132) | 本文针对宽贝叶斯神经网络中的“先验主导”退化问题，推导了在温度调度T = τ/M^c下变分贝叶斯线性神经网络的极限预测分布，并揭示了似然温度调节与NNGP后验之间的关系。 |
| [^274] | [CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use](https://arxiv.org/abs/2610.09127) | CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。 |
| [^275] | [Domain-informed Adaptive Sampling for Generalizable PINNs in Metal Additive Manufacturing via Conditional Flow Matching](https://arxiv.org/abs/2610.09126) | 本论文通过经验风险最小化理论证明工艺条件感知的自适应采样严格优于传统静态采样，并提出基于条件流匹配的两阶段自适应采样框架，显著提升了物理信息神经网络在金属增材制造热建模中的泛化能力。 |
| [^276] | [Spatial Induction Heads: In-Context Learning of Multidimensional Cellular Automata](https://arxiv.org/abs/2610.09124) | 该论文提出了“空间归纳头”这一两层“收集-匹配”电路机制，解释了transformer如何在缺乏显式坐标归纳偏置的多维元胞自动机数据中重建空间邻域并完成上下文学习。 |
| [^277] | [Are Parameter-Efficient Fine-tuning Methods Really Different?](https://arxiv.org/abs/2610.09122) | 本研究比较了语言模型和扩散模型中的六种参数高效微调方法，发现LoRA系列方法本身已近似保持预训练权重几何结构，从而质疑了显式几何保持的必要性，并揭示了不同方法在适应性与遗忘之间的权衡差异。 |
| [^278] | [The Deceptive Bandit Problem: Exploratory Coupling and the Fragility of Multi-Agent Learning](https://arxiv.org/abs/2610.09120) | 该论文揭示了多智能体学习中随机探索的独立性和隐私性是安全关键属性：欺骗者可以通过将自身探索动作与泄露的探索信号耦合，将受害者的学习动态引导至欺骗性纳什均衡，且收敛速度仍保持最优。 |
| [^279] | [Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data](https://arxiv.org/abs/2610.09117) | 提出Workhorse框架，通过在同一套无需重定向的真人演示数据上分别训练视觉规划器和强化学习全身跟踪器，并以互相模仿对方部署误差的方式增强训练数据，使Unitree G1人形机器人能从第一人称视觉实现鲁棒的全身移动操作并具备抗干扰恢复能力。 |
| [^280] | [From Uncertainty to Action: Learning to Steer LLM Agents](https://arxiv.org/abs/2610.09115) | 提出VoS（引导价值）方法，通过构建包含约82,000个反事实延续的逐步结果表（SOT）来学习每一步引导的价值，结合危害预算触发器，实现对LLM智能体更精准的纠正时机与方式决策，克服了不确定性信号无法定位最佳引导步骤的局限。 |
| [^281] | [Same Text, Different Prediction: Serving-Context Nondeterminism in Text Classifiers](https://arxiv.org/abs/2610.09111) | 该论文首次系统研究了文本分类器中的服务上下文非确定性，通过训练180个涵盖判别式、伪生成式和完全生成式的分类器模型，发现即使输入文本、模型参数和采样随机性固定，批大小、批组成、硬件和推理引擎等部署环境因素仍会导致分类预测结果发生改变。 |
| [^282] | [Breaking Adversarial Transferability in Fine-Tuned Speech Recognition](https://arxiv.org/abs/2610.09109) | 本文揭示了在公开预训练模型上构造的对抗扰动可有效迁移到黑盒部署的微调语音识别模型，并提出统一微调框架TransferBreaker，通过基础对抗微调、潜在雅可比正则化与混合梯度自适应微调来阻断对抗迁移并提升鲁棒性。 |
| [^283] | [Convex-Concave Reinforcement Learning](https://arxiv.org/abs/2610.09108) | 该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。 |
| [^284] | [On the Computational Complexity of Hidden Markov Model Identification](https://arxiv.org/abs/2610.09104) | 本文从计算复杂度的角度研究隐马尔可夫模型的可辨识性问题，旨在回答是否存在可靠且完备的算法来判定给定HMM是否可辨识，以及该判定问题的计算复杂度。 |
| [^285] | [Are We Really Benchmarking Forecasting Models? The Impact of Preprocessing on Time Series Performance](https://arxiv.org/abs/2610.09096) | 该论文通过在29,000个M4时间序列上评估11个预测模型与16种可逆预处理流水线，揭示了当前基准测试忽视预处理（如差分）所造成的结构性偏差，并证明针对每个序列优化预处理可使各模型性能提升约27%至87%。 |
| [^286] | [MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models](https://arxiv.org/abs/2610.09092) | MaRK提出了一种动态算子条件化框架，将上下文向量直接映射为对冻结SSM循环算子参数（A、B、C、D、Δ）的有界调制，使每个扩散时间步都能动态重塑模型的输入输出记忆核。 |
| [^287] | [FedRSPO+: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning](https://arxiv.org/abs/2610.09091) | 提出了异构感知的决策导向联邦学习框架FedRSPO+，其核心是基于通过投影平滑决策映射的正则化代理RSPO+，为决策误差和遗憾提供理论上界，从而解决联邦场景下下游目标与可行集异构性导致的训练不稳定问题。 |
| [^288] | [Tucker Bottleneck Attention for Multi-Dimensional Sequence Modeling](https://arxiv.org/abs/2610.09090) | 提出Tucker瓶颈注意力，通过将隐藏张量投影到紧凑的Tucker核心上进行注意力计算，利用低秩张量结构实现亚二次方复杂度的全局token混合，在视频预测和天气预报任务中显著降低误差和计算成本。 |
| [^289] | [U-Space: Uncovering When and Why Uncertainty Arises in Language Models](https://arxiv.org/abs/2610.09087) | 该论文提出U-Space方法，旨在揭示语言模型推理过程中不确定性在何时、何处以及为何产生与演变，克服了现有标量化不确定性估计方法无法定位不确定性来源的局限性。 |
| [^290] | [Towards AI-Generated Music Plagiarism Detection as a Version Identification Problem](https://arxiv.org/abs/2610.09075) | 该论文将AI生成音乐的抄袭检测建模为版本识别问题，构建了包含35万余个评估对的COPYCAT基准，并证明基于逐坐标嵌入偏移的监督式框架能够克服传统标量距离阈值方法在生成式再合成混淆下失效的问题，有效恢复被分散的抄袭信号。 |
| [^291] | [TAP: Efficient Long-Horizon Agent Pruning via Trajectory-Anchored Recovery](https://arxiv.org/abs/2610.09074) | 提出首个面向强化学习训练智能体的结构化剪枝框架TAP，通过将结构化剪枝与锚定教师轨迹的在线策略恢复相结合，解决长程智能体任务中现有剪枝方法性能严重退化的问题。 |
| [^292] | [Covariate-dependent Joint Modeling of Multivariate Ordinal Preferences and Its Connections with Comparison Models](https://arxiv.org/abs/2610.09070) | 本文提出了一种协变量依赖的多元序数偏好联合建模方法，避免了传统方法单独处理各属性或将数据粗化为胜负比较所造成的信息损失，并建立了该方法与 Bradley–Terry、Plackett–Luce 等比较模型之间的联系。 |
| [^293] | [Talking with Language Models](https://arxiv.org/abs/2610.09064) | 该论文提出“人工制品立场”框架，主张大语言模型只是精密的文本生成器而非真正的说话者，人机“对话”实为用户在界面幻象下进行的独角戏，从而消解了关于AI对话者身份、谎言与承诺等哲学难题。 |
| [^294] | [Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models](https://arxiv.org/abs/2610.09063) | 本文系统比较了通过大语言模型蒸馏训练的生成式与判别式小型语言模型在电商用户生成内容多标签主题分配任务上的表现，揭示了两种架构范式之间关键的数据依赖性权衡。 |
| [^295] | [EDiS: Edge Disjoint Subgraph Sparsification Framework for Graph Neural Networks](https://arxiv.org/abs/2610.09059) | EDiS提出了一种边不相交子图稀疏化框架，通过一次性将图分解为可缓存的边不相交子图并按边预算约束跨epoch重新组合，实现了高效且拓扑可变的GNN稀疏训练，避免了重复的结构提取计算。 |
| [^296] | [Learning Cross-Model Activation Alignments with Explicit Many-to-Many Layer Maps](https://arxiv.org/abs/2610.09058) | MATCHA 方法将跨模型激活对齐分解为可解释的显式多对多层映射和层共享特征映射并从提示中联合学习，无需预先固定层对应关系，即可更忠实地重建目标模型激活并显著提升基于检索的指标。 |
| [^297] | [A Geometry-Based Capacity Theory for Finite-Feature Associative Memory](https://arxiv.org/abs/2610.09056) | 该论文提出了有限特征Hebbian联想记忆的基于几何的容量理论，将检索干扰分解为随特征维度衰减的有限特征噪声和由键间核重叠决定的结构性干扰，从而实现无拟合的检索质量预测、揭示几何相关的容量上限，并将表示几何直接与记忆容量联系起来。 |
| [^298] | [BeatFlow-ECG: Rectified Flow for ECG Reconstruction from Indirect Wearable Signals](https://arxiv.org/abs/2610.09052) | 提出了BeatFlow-ECG，一种基于条件校正流的可穿戴信号重建模型，能从PPG和IMU信号重建单通道心电图，并通过IMU运动特征、运动相关损失加权和由易到难的训练课程有效应对运动干扰。 |
| [^299] | [A Self-Pruning Transformer: Extreme KV-Cache Compression with Universal Attention](https://arxiv.org/abs/2610.09051) | 该论文提出“通用注意力”架构，利用复合衰减机制作为自适应剪枝准则对KV缓存进行极致压缩，在保留RoPE位置嵌入和Softmax注意力的前提下实现了最先进的10倍压缩。 |
| [^300] | [Towards Financial World Modeling](https://arxiv.org/abs/2610.09048) | 该论文提出了包含近万亿条 1 Hz 观测数据的美股数据集 Market-1T、一套严谨的评估协议，并在近二十年数据上系统比较 18 种编码器训练策略，以推动金融表示学习迈向金融世界建模。 |
| [^301] | [Learning Transition Kernels of Jump-Diffusion Processes with Conditional Diffusion Models](https://arxiv.org/abs/2610.09045) | 该论文提出用条件扩散模型学习时齐跳跃扩散过程的转移核，在理论上给出了条件分数估计误差和真实与生成路径分布间KL散度的非渐近界，并在合成与真实数据上验证了其在样本路径生成和概率预测任务中的有效性。 |
| [^302] | [Quadratic Weak-to-Strong Generalization in Random Feature Networks via Random Matrix Theory](https://arxiv.org/abs/2610.09044) | 本文利用随机矩阵理论证明，在两层随机特征网络中，由弱教师模型训练出的强学生模型误差为教师误差的平方，实现了二次级的弱到强泛化改进。 |
| [^303] | [Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making](https://arxiv.org/abs/2610.09043) | CARE是一个端到端的人机协作决策框架，通过新颖的自适应校准模块在任何时刻保证风险控制，并持续从人工反馈中学习，以更少的人工查询实现更高的自动化水平。 |
| [^304] | [ORACLE: Optimizer-Relative Alignment for Constrained LEarning](https://arxiv.org/abs/2610.09040) | ORACLE提出了一种优化器相对约束学习框架，通过在优化器更新之后评估约束兼容性，在优化器自身的几何结构中构建、限制权限并验证约束对齐，在八个偏微分方程基准和四种优化器上94%的配置中优于或匹配原生优化器的表现。 |
| [^305] | [Shared-Roadmap Generation and Evaluator for Multi-Agent Path Planning Using Heterogeneous Graph Neural Network](https://arxiv.org/abs/2610.09034) | 该论文提出了一种可扩展的异构图神经网络框架，能够自动生成并评估多智能体路径规划所需的共享路线图，通过学习专家求解器轨迹的占据密度图来识别关键路径点并剪除冗余节点与边，从而在路线图紧凑性与解的质量之间取得平衡。 |
| [^306] | [Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs](https://arxiv.org/abs/2610.09033) | 本文提出ASRD数据集与四态评估框架，发现表情符号和不可见Unicode等表层变换对开放权重大语言模型的安全威胁远高于Leet语言和编码包装等变换，有害遵从率可达20%以上。 |
| [^307] | [Beyond Explanation: Debugging Medical Imaging Models via Concept Intervention](https://arxiv.org/abs/2610.09031) | 该论文提出了一个即插即用的概念干预框架，通过构建与 BioMedCLIP 对齐的概念瓶颈模型来区分因果概念与虚假相关概念，并利用反事实样本进行针对性微调，从而实现对医学影像模型的可解释调试与性能提升。 |
| [^308] | [SPIN: Shadow Predictive Indexer for Sparse Attention](https://arxiv.org/abs/2610.09025) | SPIN 提出基于历史的轻量级预测机制来识别重要 KV 块，避免每个解码步骤对完整 KV 缓存评分，在保持任务质量的同时实现 30-40% 的稀疏度，并将 vLLM 服务的吞吐量最高提升 14.9%、token 间中位延迟最高降低 13.2%。 |
| [^309] | [Neighborhood Smoothing for Calibration](https://arxiv.org/abs/2610.09020) | 该论文提出将图平滑作为训练时校准的一般原则，通过惩罚表示空间中相邻样本预测分布之间的Jensen-Shannon散度的图正则化方法来改善神经网络的过度自信和校准问题。 |
| [^310] | [Removing Information Content Does Not Certify Tamper Resistance in Open-Weight Models](https://arxiv.org/abs/2610.09004) | 移除互信息并不足以证明开放权重模型的防篡改能力，因为保持函数不变的重新参数化可以在信息量不变的情况下改变梯度下降几何结构，甚至存在信息量为零却能在一步梯度中恢复能力的构造。 |
| [^311] | [Algorithmic Scratchpads and Curriculum Staging for Arithmetic Reasoning in Tiny Transformers](https://arxiv.org/abs/2610.09003) | 本文通过研究微型Transformer的多步算术推理，发现并修复了序列填充导致的梯度饥饿问题，证明语言预训练与现代架构组件至关重要，且算法草稿纸的表述形式直接决定模型推理性能。 |
| [^312] | [How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis](https://arxiv.org/abs/2610.09000) | 研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。 |
| [^313] | [REFIT: Recognize, Fix, and Test Wearable Sensor Placement Shifts without Labels](https://arxiv.org/abs/2610.08991) | REFIT是一种无需标签和重新训练的输入校准方法，通过轴变换族拟合和归一化统计量重估计，自动识别并修复可穿戴传感器佩戴位置变化对冻结活动识别模型的影响，并提供无标签准确率评估。 |
| [^314] | [LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL](https://arxiv.org/abs/2610.08989) | LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。 |
| [^315] | [A method for multimodal analysis of TAIGA experiment data using essential features](https://arxiv.org/abs/2610.08985) | 提出了一种基于自编码器等神经网络提取本质特征的新方法，实现了对TAIGA实验来自多个装置的多模态数据的联合分析。 |
| [^316] | [SNR-Gated LSTM-Conditioned Diffusion Model for MIMO Channel Estimation](https://arxiv.org/abs/2610.08977) | 该论文提出了一种SNR门控LSTM条件扩散模型，在角度域对MIMO信道进行估计，通过利用时间序列动态特性和可学习的信噪比自适应融合机制，在宽SNR范围内实现准确且低时延的信道估计。 |
| [^317] | [The Best Optimizer Depends on Batch Size](https://arxiv.org/abs/2610.08975) | 该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。 |
| [^318] | [HULK: Learning Whole-Body Forceful Loco-Manipulation for Humanoids](https://arxiv.org/abs/2610.08970) | 提出HULK全身控制框架，利用模型预测控制引导强化学习训练两个教师策略（腕力手臂运动跟踪与抱物行走），并通过捕获点控制障碍函数增强负载下的平衡能力，最终蒸馏为单一策略，实现人形机器人对大型重物的强力移动操作。 |
| [^319] | [Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization](https://arxiv.org/abs/2610.08969) | 该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。 |
| [^320] | [On KL-Regularized Policy Optimization](https://arxiv.org/abs/2610.08963) | 提出KLPO框架，通过将KL正则项锚定在采样器上，利用闭式Gibbs解的对数比最优性条件在采样器自身轨迹上做最小二乘拟合，从而在不使用重要性权重的情况下解决LLM智能体异步强化学习中采样与训练策略不一致的问题。 |
| [^321] | [Directed Temporal Representations for Offline Visual Control](https://arxiv.org/abs/2610.08960) | 该论文提出DTRC方法，在冻结世界模型特征之上学习与时间可达性对齐的有向时间拟度量表征，并利用其时间进度信号作为评论家，实现离线视觉目标条件策略的直接学习。 |
| [^322] | [GraphOPD: Graph-Augmented On-Policy Distillation for LLM Agents](https://arxiv.org/abs/2610.08959) | GraphOPD是首个将基于图的结构增强引入大语言模型智能体在线策略蒸馏的方法，解决了传统依据师生分歧分配指导的方式在多轮决策场景中失效的问题。 |
| [^323] | [CARE: Certifying Acceleration for Vision-Language-Action Inference](https://arxiv.org/abs/2610.08917) | 提出CARE方法，通过在相同初始条件下的成对回放和有限样本保证，为视觉-语言-动作模型的加速推理提供可认证的加速器选择，揭示并控制被平均指标掩盖的加速诱发任务失败。 |
| [^324] | [Task-Sufficient Contraction: Source Selection for Machine Information Interfaces](https://arxiv.org/abs/2610.08884) | 本文提出“任务充分收缩”这一新性质，证明当机器的动作集合与损失函数固定时，仅凭任务声明就能在选定失真目标之前对信源状态进行合并简化，且由此得到的简化信源能精确保持下游完整的一步码率-后悔曲线。 |
| [^325] | [Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space](https://arxiv.org/abs/2610.08883) | 该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。 |
| [^326] | [FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks](https://arxiv.org/abs/2610.08882) | 本文通过对照实验证明，对40亿参数模型进行秩16的LoRA适配可显著提升结构化金融任务表现（如JSON有效率、FinQA精确匹配、计算器表达式正确率等），同时揭示了基准中标签集合变化和数据重叠对泛化性结论的限制。 |
| [^327] | [An Empirical Study of Agent Skills' Downstream Utility](https://arxiv.org/abs/2610.08875) | 本文通过在87个SkillsBench任务上的实证研究，将智能体技能的下游效用量化为相对于无技能基线的通过率差异，并揭示效用如何取决于技能内容、执行配置与多技能组织方式。 |
| [^328] | [Slow Beats Fast at the Kesten-Stigum Threshold: Minimax, Fisher-Information and Belief-Propagation Characterizations of the Information-Computation Gap in Sparse Stochastic Block Models](https://arxiv.org/abs/2610.08872) | 该论文通过统计决策理论、Fisher信息和置信传播对稀疏随机块模型的Kesten-Stigum阈值给出三种刻画，证明在 q≥5 时阈值下方存在信息-计算差距：多项式低度算法渐近无法超越平凡风险，而指数时间算法却能成功。 |
| [^329] | [Learned Monotone Recurrent Features in Governed Credit Scoring: The Price of the Frame and the Necessity of Macro Conditioning](https://arxiv.org/abs/2610.08869) | 该论文证明了在受治理的信用评分中，学习型单调循环特征的价值随治理框架严格程度的提高而上升，且这些特征的循环门控必须进行外生宏观条件化才能保持其价值。 |
| [^330] | [Geometry-Aware Diffusion Approximate Posterior Sampling for Sparse-View and Limited-Angle CT](https://arxiv.org/abs/2610.08866) | 该论文提出一种几何感知的扩散近似后验采样方法，利用CT采集中连续变化的测量灵敏度来联合引导稀疏视角和有限角度CT重建中的更新方向与随机探索，以应对严重病态性带来的重建歧义。 |
| [^331] | [Adversarial RL for Port-Scan Evasion: Attacker Feature Visibility in Edge-Deployed IDS](https://arxiv.org/abs/2610.08864) | 该论文提出用深度Q网络（DQN）自适应攻击者来规避部署在树莓派边缘设备上的机器学习入侵检测系统，通过调整探测时序、TCP标志和负载大小，在黑盒、灰盒和白盒不同特征可见性设置下实现61.9%至98.3%的规避率，并发现攻击者特征可见性的增加并不总能单调提升规避效果。 |
| [^332] | [LRCC: Generalizing Low-Rank Compression with Conditional Computation](https://arxiv.org/abs/2610.08858) | LRCC通过为每个Transformer块训练轻量级路由器在嵌套低秩路径间动态选择，在训练时冻结低秩因子仅优化路由器，在相同的平均活跃参数预算下性能超越静态低秩压缩，在Llama-2-7B上平均下游准确率提升7.6个百分点。 |
| [^333] | [Autonomous Droplet Navigation via Model-Based Reinforcement Learning: Zero-Shot Transfer and Emergent Dynamics](https://arxiv.org/abs/2610.08852) | 该研究提出了首个基于模型强化学习的机器人平台，实现了开放表面上液滴的闭环自主导航，仅需50至150次物理实验即可完成策略训练而无需仿真或解析模型，并展现出零样本迁移与涌现动力学特性。 |
| [^334] | [Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment](https://arxiv.org/abs/2610.08842) | 该研究提出了一个包含风险评估、证据定位与双验证器因素识别的可解释自杀风险评估框架，通过基于长度的路由、风险-证据一致性约束以及分类验证器与证据感知验证器的结合，超越单纯的风险分类，实现了对预测背后文本证据与心理社会因素的可解释性分析。 |
| [^335] | [Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding](https://arxiv.org/abs/2610.08840) | 该研究通过对103,939条回复的大规模实验发现，LLM的谄媚行为主要由任务验证代价和护栏覆盖情况决定，而非模型家族或用户压力策略——锚定事实几乎不被让步（1.3%），而逻辑谜题等更易被用户诱导改口。 |
| [^336] | [CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models](https://arxiv.org/abs/2610.08833) | CoDR 提出了一种无需训练、与采样器无关的置信度漂移重掩码方法，通过检测已提交词元的置信度下降并仅对模型不再认可的词元进行重掩码和重新生成，有效防止了掩码扩散语言模型解码过程中早期错误的传播。 |
| [^337] | [Child ASR Adaptation with Adult Retention: An Empirical Study](https://arxiv.org/abs/2610.08827) | 该实证研究在阿拉伯语和英语上系统比较了全量微调、LoRA与权重空间合并等儿童ASR适配方法，发现儿童语音适配虽有必要但常导致成人语音识别性能遗忘，而双语适配比单语言适配更稳定，能更好地平衡儿童适配与成人保留。 |
| [^338] | [HCPN-GCN: Scaling Hierarchical Prototype Networks with Cone Geometry for Continual Graph Learning](https://arxiv.org/abs/2610.08823) | 提出HCPN-GCN，通过用图卷积网络替换线性特征提取器并引入基于锥体的原型与多样性正则化机制，在无需存储历史数据的情况下有效缓解持续图学习中的灾难性遗忘问题。 |
| [^339] | [A Vehicle-Integrated Approach to Digital Twin Deployment for Bridges Through Drive-By Sensing](https://arxiv.org/abs/2610.08822) | 该论文提出了一种融合基于物理的建模与机器学习的车辆一体化数字孪生框架，通过傅里叶神经算子代理模型利用车载间接传感数据实现对桥梁与路面状态的连续监测，以克服传统结构健康监测系统成本高、难扩展的局限。 |
| [^340] | [Task-Oriented Key-Layer KV Communication for Efficient Latent Multi-Agent Collaboration](https://arxiv.org/abs/2610.08820) | 该论文提出无需训练的KITE框架，将多智能体潜在通信的目标从发送端状态保真度转变为接收端任务充分性，通过识别任务有效的关键层并仅传输其潜在工作记忆，大幅降低了通信与计算开销。 |
| [^341] | [HydroSphere: A Framework for Governed, Self-Healing Wastewater Infrastructure](https://arxiv.org/abs/2610.08819) | HydroSphere是一个将治理机制与自愈能力相结合的数据驱动污水处理框架，它利用混合TCN-LSTM模型对七项水质参数进行多步预测，并通过PPO强化学习自适应优化化学药剂投加，从而实现实时水质监测、处理优化与故障恢复。 |
| [^342] | [Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States](https://arxiv.org/abs/2610.08818) | 该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。 |
| [^343] | [DenoFlow: Flow Matching for SSVEP Denoising under Real Physiological Artifacts](https://arxiv.org/abs/2610.08817) | DenoFlow将SSVEP去噪建模为基于整流流的输运问题，通过场网络回归受污染信号与干净信号之间直线路径的速度场，在真实生理伪迹下实现去噪并保持信号的可解码性。 |
| [^344] | [The Cost of Long Memory: State, Context, and Stability Complexity in Sequence Models](https://arxiv.org/abs/2610.08816) | 该论文针对长记忆序列模型证明了匹配的逼近上下界：达到预测误差 τ 仅需 Θ(log²(1/τ)) 个状态或模式（误差按 e^{-Θ(√r)} 衰减），并揭示真正的分数长记忆会从根本上改变逼近问题的几何结构。 |
| [^345] | [Bounded Autonomy and Verifiable Safety for Agentic AI Enabled Automation](https://arxiv.org/abs/2610.08815) | 本文提出BRaVeS安全治理框架，通过将专家约束编码为不变锚点、深度感知访问机制和随认知风险动态调整的状态层级自主性，并结合Lyapunov有界一致性框架实现屏蔽式状态转换，从而为高风险环境中的智能体AI自动化提供有界自主性与可验证的安全保障。 |
| [^346] | [Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning](https://arxiv.org/abs/2610.08814) | 提出RVV框架，通过路由、验证、投票三个阶段实现程序条件化自洽性，在无需参数更新的情况下提升语言模型在未见过的混合领域中识别完整正确答案集合的能力。 |
| [^347] | [KVFetch: Temporal Prefetching for the Missing Half of KV Cache Compression](https://arxiv.org/abs/2610.08811) | 该论文发现现有KV缓存压缩方法只实现了按内容的关联查找，而缺失了按位置的顺序访问能力，导致模型逐字复现上下文内容时中途不可逆地失败，并提出KVFetch时间性预取方法来补全这缺失的另一半。 |
| [^348] | [Transferability and operational reliability of a Prithvi crop classification foundation model under phenological and geographic shift across three continents](https://arxiv.org/abs/2610.08810) | 该研究评估了Prithvi-EO-2.0作物分类基础模型在三大洲的分布外性能，发现其精度随地理偏移显著下降（美国0.65降至欧洲0.40），且观测窗口与当地作物物候错位时精度会严重崩溃，揭示了GeoFM实际部署中的可靠性局限。 |
| [^349] | [Beyond Baseline Severity: Temporal and Disease-Specific Predictors of Depression Outcomes Following Mindfulness Interventions](https://arxiv.org/abs/2610.08809) | 该研究通过对多中心纵向临床队列进行可解释机器学习分析，证明超越基线抑郁严重程度的时间性与疾病特异性预测因子（如治疗参与度和临床背景）可有效预测正念干预后12周及24周的短期与长期抑郁结局。 |
| [^350] | [Accelerating Floating-Point Satisfiability Solving via Gradient Normalization](https://arxiv.org/abs/2610.08808) | 该论文提出GradSAT框架，将每个SMT子句视为独立的多任务学习任务，通过动态梯度归一化平衡各子句梯度，克服梯度主导现象，从而加速浮点可满足性求解。 |
| [^351] | [A Bayesian Mirror Architecture for Emergent Consciousness: Circular Hierarchies, Self-Manifolds, and Hybrid Event-Self Binding](https://arxiv.org/abs/2610.08792) | 本文提出贝叶斯镜像架构（BMA），通过循环递归与闭环更新将自我表征绑定到抽象世界模型，把意识定义为具有这种循环结构的系统的一种架构属性，并在最优传输几何（2-Wasserstein 度量）下形式化自我稳定性与混合一致性。 |
| [^352] | [CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution](https://arxiv.org/abs/2610.08592) | CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。 |
| [^353] | [HE-OFT: Privacy-Preserving One-Shot Federated Fine-Tuning under Homomorphic Encryption](https://arxiv.org/abs/2610.08255) | 提出了HE-OFT——首个密码学安全的一次性联邦微调协议，任何一方都无法获得训练好的模型，在不泄露数据隐私的同时保护了模型这一专有资产。 |
| [^354] | [Finding the Heads and the Neurons Responsible for Network Information Retrieval in Language Models](https://arxiv.org/abs/2610.08200) | 该研究发现，语言模型中经因果消融验证的极少数注意力头能以99.5%至100%的准确率检测上下文中的主机名与IP地址配对信息，且在某些模型中这一功能可进一步精确定位到单个神经元。 |
| [^355] | [Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters](https://arxiv.org/abs/2610.07834) | 提出即插即用的FreshCast框架，通过持续用新观测数据刷新非参数记忆并进行校准，解决了检索增强时间序列预测中记忆陈旧导致检索效用下降的问题。 |
| [^356] | [Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding](https://arxiv.org/abs/2610.07713) | 提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。 |
| [^357] | [DeepAJM: Deep Association Joint Model for Irregularly Sampled data](https://arxiv.org/abs/2610.07388) | 提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。 |
| [^358] | [Sample-Optimal Estimation of the Fr\'echet Inception Distance](https://arxiv.org/abs/2610.07114) | 该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。 |
| [^359] | [ImpactMat: Continuous Material Estimation for Inverse Impact Sound Rendering](https://arxiv.org/abs/2610.07061) | 该论文提出了ImpactMat数据集与基准，以及一个前馈模型，能够从冲击声音录音中连续估计材质参数，实现逆冲击声音渲染，从而突破传统渲染器固定材质预设的限制。 |
| [^360] | [Comparative review of hybrid forecasting models for short-term prediction of building thermal load](https://arxiv.org/abs/2610.06881) | 本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。 |
| [^361] | [RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents](https://arxiv.org/abs/2610.06401) | 提出RAISED训练框架，通过自我生成与自蒸馏相结合的方式，在不损害大语言模型智能体通用能力的前提下，显著提升其对间接提示注入攻击的鲁棒性。 |
| [^362] | [Scaling Down the Scaling Laws: Parameter Efficiency and Compute-Optimal Training in Resource-Constrained Large Language Models](https://arxiv.org/abs/2610.06387) | 这篇综述梳理了大语言模型扩展理论从经验扩展定律到计算最优训练的演进，指出研究重心正从单纯追求规模最大化转向在资源受限条件下对参数、token、计算和硬件资源的高效利用与合理分配。 |
| [^363] | [Lossy Compression of PDE Training Inputs: Field Reconstruction Error Does Not Order the Cost to a Trained Operator](https://arxiv.org/abs/2610.06095) | 本文证明压缩PDE训练输入时的场重构误差无法预测所训练算子的精度损失，因为解算子对输入扰动的衰减程度在不同PDE族之间相差两个数量级以上，导致重构误差指标在104个代价比较中反转了36个。 |
| [^364] | [Boosting Transferable Adversarial Attacks against Deep Reinforcement Learning](https://arxiv.org/abs/2610.06083) | 该论文提出一种基于可微分环境模型和温度平滑代理策略的轨迹级黑盒攻击方法，通过在滚动时域内优化扰动序列，显著提升了对抗扰动从未知受害智能体上的迁移攻击效果，超越了传统图像分类攻击方法移植后的表现。 |
| [^365] | [The sublevel Flood bifiltration: towards scalable 2-parameter persistent homology](https://arxiv.org/abs/2610.05441) | 本文提出次水平Flood双过滤，通过扩展单参数Flood过滤，为2参数持续同调提供了一种具有理论稳定性保证且可高效计算、适用于大规模点集的可扩展近似方法。 |
| [^366] | [E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation](https://arxiv.org/abs/2610.05048) | 论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。 |
| [^367] | [PIT-GCL: Protein Interaction using Topological Graph Contrastive Learning](https://arxiv.org/abs/2610.04850) | PIT-GCL 提出了一种双塔结构感知框架，将 ESM-2 残基嵌入与基于 Vietoris-Rips 过滤持续同调（H0/H1 持续景观）的拓扑描述符相结合，无需结合态复合物结构即可实现序列与结构层面的蛋白质相互作用预测。 |
| [^368] | [Questioning the Questions: Sustaining Self-Evolution in Reasoning Models](https://arxiv.org/abs/2610.04299) | 该论文揭示了自演化推理模型性能崩溃的两大根源——自生成问题中无效问题比例上升以及数学等价重复问题导致多样性崩溃，并提出通过问题有效性与新颖性反馈（R-Quest）来引导和维持模型的自演化。 |
| [^369] | [Humanoid Rickshaw Pulling: Whole-Body Locomotion under Coupled Wheeled Loads](https://arxiv.org/abs/2610.04238) | 提出了一种人形机器人拉人力车的全身控制框架，通过特权教师策略蒸馏到基于历史条件的学生策略并结合强化学习微调，使机器人能够在不确定的耦合负载动力学下拉动远超自身重量的轮式负载，同时保持平衡与稳定抓取。 |
| [^370] | [Clean: Second-order LLM Training at Linear Memory Cost via Nystr\"om Sketching](https://arxiv.org/abs/2610.04204) | Clean利用随机化Nyström草绘将全曲率二阶优化器的内存复杂度从二次方降至线性，并通过重新整合子空间外分量保留曲率信息，其低精度变体Q-Clean进一步将优化器内存减少50%以上，实现了内存高效的二阶LLM训练。 |
| [^371] | [WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites](https://arxiv.org/abs/2610.03036) | 本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。 |
| [^372] | [Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers](https://arxiv.org/abs/2610.02701) | 该研究发现在粒子物理Transformer中，避免token丢弃的top-1混合专家模型能在几乎不增加计算量的前提下超越稠密基线，但增加存储专家数量的收益有限，且专家路由结构与分类性能并非单调相关。 |
| [^373] | [Harnessing LLMs as Agents: What Does It Cost?](https://arxiv.org/abs/2610.02488) | 该论文提出语言模型智能体机（LAM）这一资源受限的计算抽象，首次从通信、内存访问、重计算与验证等方面严格量化 LLM 智能体驱动框架所消耗的计算资源，并给出相应的计算量下界。 |
| [^374] | [TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning](https://arxiv.org/abs/2610.02199) | 本文提出TACO优化器，在维度归一化算子范数下计算精确的最速下降方向，以三值逐列单稀疏的形式存储优化器状态，在不牺牲精度和计算效率的前提下大幅降低大语言模型微调中的优化器内存开销。 |
| [^375] | [Benchmarking Generative Models for Weather Data Assimilation on Real Station Observations](https://arxiv.org/abs/2610.00728) | 该研究提出了首个基于真实气象站观测数据的生成式天气数据同化受控基准测试，利用美国本土11,849个NOAA MADIS站点数据，在固定数据集、观测算子和深度学习架构的条件下，系统比较了扩散模型与流匹配、像素空间与潜在空间等主要设计选择的有效性。 |
| [^376] | [Nonparametric Distribution Matching for Self-Supervised Whole-Slide Image Condensation](https://arxiv.org/abs/2610.00678) | 提出NICER框架，将自监督全切片图像浓缩重新表述为分布匹配问题，利用具有切片自适应容量的非参数先验显式保留学习相关特征分布，在五个组织病理学数据集上平均准确率提升7.44%并获得病理学家临床评估认可。 |
| [^377] | [Benchmarking System One decision models against trained classifiers and language models for automated decision gates](https://arxiv.org/abs/2610.00346) | 该研究在匹配条件下对系统一决策模型、训练分类器和生成式语言模型进行统一基准测试，发现模型类别的优劣取决于标签可用性：有标签时小型训练分类器表现最佳，无标签时决策模型在工作流任务上普遍优于零样本分类器。 |
| [^378] | [OpenTSLM TeeMoE: A Unified Time-Series Language Model for Forecasting, Contextual Prediction, and Reasoning](https://arxiv.org/abs/2609.40265) | OpenTSLM TeeMoE 通过在共享主干上训练的 LoRA 混合专家架构，将时间序列直接预测、情境条件预测和语言化时间推理这三种异构能力统一到单一通用模型中，且不牺牲各项单独性能。 |
| [^379] | [STARS: From Spatiotemporal Dynamics to Social Representations in Human-Robot Interaction](https://arxiv.org/abs/2609.40245) | 本文提出了SocialNav-SUB基准，通过视觉问答的形式系统评估视觉-语言模型在理解复杂社会导航场景（包括智能体间时空关系和人类意图推断）方面的能力，填补了社会机器人导航领域VLM评估的空白。 |
| [^380] | [Deep Learning-Based Tri-Hybrid Multi-User MIMO Precoding: The Blessing of EM-Reconfigurable Antennas](https://arxiv.org/abs/2609.39167) | 本文提出基于Conformer神经网络架构的三混合预编码网络Tri-PNet，通过将电磁可重构天线引入的电磁域预编码与模数混合预编码进行联合设计，显著提升了宽带多用户MIMO-OFDM系统的频谱效率。 |
| [^381] | [Multi-LLM Collaborative Alignment via Stackelberg Games](https://arxiv.org/abs/2609.39076) | 该论文提出受博弈论启发的Stackelberg对齐框架，由EXP3老虎机作为领导者根据指令难度和回应可区分性自适应地分配采样预算，将指令选择变为自适应课程，从而提升多个大语言模型相互学习、协同对齐的效果。 |
| [^382] | [The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion](https://arxiv.org/abs/2609.36252) | 该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。 |
| [^383] | [Control-Geometry Straightening for Sampling-Based Latent Planning](https://arxiv.org/abs/2609.35603) | 提出控制几何拉直（CGS）这一辅助损失，通过将动作间余弦相似度与潜在差异对齐来学习对规划器友好的表示，从而提升基于采样的潜在规划的优化效率，并给出相应理论保证。 |
| [^384] | [Residual-Stream Burden Shapes Representation Learning in Diffusion Transformers](https://arxiv.org/abs/2609.33895) | 该论文提出“残差流负担”概念，解释了扩散Transformer在数学上等价的预测目标（干净数据、噪声、速度）下表现不对称的原因：噪声目标要求残差流在整个深度保留噪声相关变化，迫使后续层在带噪表征上计算，而干净的预测目标负担更轻，为后续计算组织隐藏表征留下了更大自由度。 |
| [^385] | [Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging](https://arxiv.org/abs/2609.32303) | 该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。 |
| [^386] | [Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds](https://arxiv.org/abs/2609.30499) | 本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。 |
| [^387] | [GridSFM: A Foundation Model for Solving AC Optimal Power Flow](https://arxiv.org/abs/2609.30173) | GridSFM是一个1500万参数的物理启发图神经网络基础模型，通过在54种电网拓扑上预训练并结合基于牛顿法的物理信息微调，仅需100个求解实例即可适应多达10000节点的未见电网，实现2.45%的零样本发电成本误差，且优于使用更多数据训练的单一拓扑专用神经网络模型。 |
| [^388] | [QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs](https://arxiv.org/abs/2609.26855) | 提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。 |
| [^389] | [End-to-End Quantum Semantic Communication with Variational Quantum Neural Networks](https://arxiv.org/abs/2609.25044) | 本文提出了首个结合变分量子神经网络与语义通信的端到端量子语义通信框架，通过变分量子发射机与可训练量子接收机在含噪量子信道上传输压缩语义信息，并在理想、比特翻转、去极化和振幅阻尼等多种信道条件下验证了其分类性能。 |
| [^390] | [Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources](https://arxiv.org/abs/2609.21926) | 该论文证明了当独立源的密度函数一阶导数存在有限个不连续点（如拉普拉斯分布）时，实解析非线性独立成分分析（nICA）在排除平凡歧义后是可辨识的，其核心证明思路是利用源分布中的折点与实解析函数平滑性之间的对比。 |
| [^391] | [Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models](https://arxiv.org/abs/2609.21363) | 该论文系统揭示了多模态大推理模型可通过视觉推理从照片精确推断用户地理位置的隐私威胁，指出拒绝式防护与像素空间扰动防御的不足，并提出了一种基于扩散模型的隐私泄露缓解方法。 |
| [^392] | [FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning](https://arxiv.org/abs/2609.18964) | 该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。 |
| [^393] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^394] | [Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference](https://arxiv.org/abs/2609.13205) | 提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。 |
| [^395] | [Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting](https://arxiv.org/abs/2609.12890) | 提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。 |
| [^396] | [Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data](https://arxiv.org/abs/2609.11917) | 该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。 |
| [^397] | [Generalized Score Matching for Parameter Estimation on Convex Domains](https://arxiv.org/abs/2609.11521) | 本文从最小概率流学习出发，构造性地推导出凸域上的广义分数匹配目标函数，统一了经典分数匹配与非负数据的域适配变体，并证明该目标是二阶正当局部评分规则，保证最小化时能恢复真实密度。 |
| [^398] | [The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding](https://arxiv.org/abs/2609.10296) | 提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。 |
| [^399] | [Decomposition-Guided Diffusion Language Models for Inertial Confinement Fusion Prediction](https://arxiv.org/abs/2609.07756) | 提出据信首个基于语言模型的惯性约束聚变预测器ICF-DLM，通过物理类型化分解、双向去噪和物理驱动PPO奖励，直接从激光脉冲与靶设计参数准确预测中子率波形。 |
| [^400] | [Shallow neural network approximation in mixed Sobolev spaces](https://arxiv.org/abs/2609.05263) | 该论文建立了与激活函数无关的傅里叶块原理，证明浅层神经网络对混合光滑度为 $\alpha$ 的函数的逼近代数阶为 $\min\{\alpha,\rho\}$，并通过匹配的下界确定了 $\mathrm{ReLU}^k$ 网络在任意维数下的最优代数逼近指数为 $\min\{\alpha,k+1\}$。 |
| [^401] | [Robust PAC Learning of Concurrent Stochastic Games](https://arxiv.org/abs/2609.04189) | 该论文提出了首个针对具有转移不确定性的广义和并发随机博弈的PAC学习框架，通过引入纳什裕度刻画解决了均衡存在性问题，能在多项式样本复杂度下返回社会福利近优的ε-近似纳什均衡或证明精确纳什均衡不存在。 |
| [^402] | [Generative Nested Sampling of Atomistic Thermodynamic Landscapes](https://arxiv.org/abs/2609.03193) | 本文提出NS-Flows，利用单一条件归一化流替代马尔可夫链更新来加速原子体系热力学景观的嵌套采样，并通过对比揭示原子多模态（离散、组合性、硬碰撞壁分隔）与引力波后验（平滑简并、局域耦合）在结构上的根本差异。 |
| [^403] | [The Implications of Linguistic Illegibility for LLM Security](https://arxiv.org/abs/2609.02852) | 本文提出“语言不可读性”概念，指出大语言模型的外部语言输出无法可靠反映其基于激活空间数学运算的内部计算，从而对依赖模型语言自我报告的安全机制构成根本性挑战。 |
| [^404] | [Online Estimation of Dynamic Origin-Destination Matrices Using Reinforcement Learning with Link-Flow Propagation Guidance](https://arxiv.org/abs/2608.30317) | 提出了LFPG-RL方法，将路段流量传播引导融入强化学习，用于在线动态OD矩阵估计，解决了传统标量反馈在不同目标流量轨迹下模糊不清的问题。 |
| [^405] | [EDGE: Engine for Deterministic Graph Evaluation through Conversation Simulation from Graph Structured DSL Configuration](https://arxiv.org/abs/2608.29971) | 该论文提出EDGE评估框架，通过基于DSL的有向图形式化表示与图遍历算法穷举对话路径，并重放可复现轨迹，从而系统性地评估多智能体系统的行为确定性与一致性。 |
| [^406] | [Forward-Deployed Full-Stack Engineering for Autonomous Cloud MLOps](https://arxiv.org/abs/2608.29615) | 本文提出一个以证据为门控的多智能体框架，通过结合图工程、循环工程和智能体框架工程，将自然语言的MLOps云工程任务自动转化为经过验证的代码仓库和可运行的云端部署。 |
| [^407] | [TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization](https://arxiv.org/abs/2608.29564) | 论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。 |
| [^408] | [How Language Models Organize and Structure Moral Knowledge](https://arxiv.org/abs/2608.27402) | 本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。 |
| [^409] | [Are LLM-Enhanced GNNs Privacy-Safe?](https://arxiv.org/abs/2608.25727) | 本文首次系统评估了LLM增强的GNNs在链接、标签和成员推断三种威胁下的隐私风险，并提出了一个五阶段统一框架进行实验分析。 |
| [^410] | [LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure](https://arxiv.org/abs/2608.13545) | 本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。 |
| [^411] | [Dual-Primal Graph VAEs for Noisy Label Aggregation](https://arxiv.org/abs/2608.11473) | 本文提出一种基于图变分自编码器的众包噪声标签聚合方法，利用对偶图上的GAT消息传递将真实标签作为潜在变量，无需分类器或伪标签，在基准测试中达到最优性能。 |
| [^412] | [The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding](https://arxiv.org/abs/2608.10137) | 该论文提出SHIM，巧妙利用约束解码工具已维护的解析器和词法分析器状态作为信号，通过轻量级离线训练的校正模块修正语言模型的下一词元概率，在不改动模型本身的前提下消除语法约束解码带来的分布偏差。 |
| [^413] | [Faster-WAM: Do World Action Models Need Deep Action Modules?](https://arxiv.org/abs/2608.02365) | 提出Faster-WAM，通过以世界模型为中心的设计，用浅层轻量级动作专家（配合DoT、Lite KV-Fusion、仅世界模型条件化及收缩式1D-RoPE）替代深层动作模块，将计算集中于视频世界模型，从而降低推理延迟并缓解过拟合问题。 |
| [^414] | [ReToken: Improving Long-Context VLMs with Visual Retrieval Token](https://arxiv.org/abs/2607.28627) | RETOKEN通过单一可学习嵌入从VLM内部表示中提取检索信号，直接在预填充的KV缓存中选择与查询相关的视觉token，无需独立检索器或重新编码即可显著提升模型在长上下文图像和视频任务上的性能。 |
| [^415] | [Reinforcement Learning for Code Optimization](https://arxiv.org/abs/2607.25970) | 该论文提出三阶段方法解决基于执行时间奖励的强化学习中的噪声、稀疏与不稳定性问题——包括构建带校准沙箱的DMC-Optim基准、组合正确性与速度的奖励设计并用离线模拟器选择配置、以及适配GRPO算法——从而让大模型能够真正学会优化代码运行速度。 |
| [^416] | [What do Reward Models Memorize?](https://arxiv.org/abs/2607.24484) | 本文通过反事实记忆测量发现，判别式训练的奖励模型会错误记忆简单偏好对、记住数据集特定捷径，并过度泛化长度等简单启发式特征，导致其无法在情境相关场景中准确判断回复质量。 |
| [^417] | [Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning](https://arxiv.org/abs/2607.18130) | 该论文将流形约束超连接（mHC）引入冻结主干微调，通过保留流混合、仅学习子层访问流的方式，以更少的可训练参数改善模型损失，确立了残差路由作为基础模型高效微调的新架构维度。 |
| [^418] | [Linear Independent Component Analysis via Optimal Transport](https://arxiv.org/abs/2607.14081) | 本文提出以数据线性投影到标准高斯分布的平方 Wasserstein 距离作为 ICA 对比函数，并证明该距离在投影恰好恢复出独立成分时达到最大、且与任何真实混合信号之间都存在显式间隔，从而为线性独立成分分析建立了一种基于最优传输的新方法。 |
| [^419] | [Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains](https://arxiv.org/abs/2607.13319) | OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。 |
| [^420] | [RUBRIC: Realism--Utility Balanced Ranking for Imbalanced Classification](https://arxiv.org/abs/2607.09816) | RUBRIC是一个与生成器无关的过滤框架，通过现实性与效用的平衡排序，将合成样本选择优化为质量优先问题，从而减少低质量候选样本对决策边界的干扰，提升不平衡分类的泛化性能。 |
| [^421] | [Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator](https://arxiv.org/abs/2607.07993) | 提出幻觉自博弈（HSP）框架，让检测器与演化中的生成器以对抗方式协同演化——利用RLAIF训练生成器产生越来越难检测的幻觉，从而不断自举提升幻觉检测器的性能。 |
| [^422] | [ECGLight: Compute-Light Framework For Paper ECG Digitization and Myocardial Infarction Screening](https://arxiv.org/abs/2607.07683) | ECGLight是一个低计算量的设备端框架，能够高保真地将纸质心电图数字化并同时支持心肌梗死筛查等多项临床任务，适用于网络和计算资源受限的偏远地区诊所。 |
| [^423] | [Target-Guided Selective Reweighting for Physics-Informed Neural Network Inverse Problems: A Transfer Learning Approach](https://arxiv.org/abs/2607.05271) | 提出TGSR-PINN方法，将迁移学习与基于目标证据的神经元敏感度评分和选择性重加权相结合，解决了物理信息神经网络在偏微分方程反问题中因负迁移导致的物理参数恢复不准确问题。 |
| [^424] | [Non-asymptotic Convergence of Stochastic Gradient Descent in Score-based Generative Models](https://arxiv.org/abs/2607.04775) | 本文研究了基于分数的生成模型训练中随机梯度下降的非渐近收敛保证，针对一般分数参数化给出了显式依赖损失加权和时间采样分布的非凸优化界，并为过参数化两层 ReLU 网络建立了神经正切核分析。 |
| [^425] | [Directional Curvature from Armijo Backtracking: A Low-Cost Sharpness Probe and a Calibration-Free Learning-Rate Safeguard for Adam](https://arxiv.org/abs/2607.03998) | 本文提出利用Armijo回溯线搜索的接受步长来低成本估计损失函数的局部尖锐度（Hessian最大特征值），并将其作为Adam优化器的无标定学习率保护机制，有效防止初始学习率过大。 |
| [^426] | [Evaluating Time Series Foundation Models for Electricity Price Forecasting: Contamination Risk, Distributional Shifts, and Covariate Dependence](https://arxiv.org/abs/2607.02623) | 本文提出一个双数据集基准测试框架来评估时间序列基础模型在电价预测中的表现，发现其虽具竞争力且常优于通用基线，但性能高度依赖协变量支持且未必超越领域专用方法，而将两者简单集成可获得更优效果。 |
| [^427] | [FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement](https://arxiv.org/abs/2607.01111) | 提出故障感知重试框架FAR，通过故障对比偏好适应与轻量动作扰动让机器人在测试时从失败中学习并自主恢复任务，同时将成功恢复轨迹纳入训练实现持续策略改进，成功率平均提升17.6%。 |
| [^428] | [In-Context Residual Calibration for Uncertainty Quantification of Energy Time Series over Graphs](https://arxiv.org/abs/2606.31804) | 该论文提出一种上下文残差校准方法，针对现有共形预测难以捕捉能源系统复杂时空结构的缺陷，为图上能源时间序列提供更可靠的不确定性量化，以支持风险感知的能源运营决策。 |
| [^429] | [Geological text descriptions in ill-posed inverse problems: insights from learned hydraulic-conductivity inversion](https://arxiv.org/abs/2606.24967) | 该研究通过合成达西流基准系统考察了地质文本描述在学习型渗透系数反演中的作用，发现当观测存在结构歧义时描述能改善重建，实例特定内容尤为有效，且求解器会依据描述所述的几何信息偏移重建结果，但随着观测信息量增大，描述的影响会减弱。 |
| [^430] | [BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks](https://arxiv.org/abs/2606.24162) | 本文提出BehaviorBench基准，从行为预测与模拟、战略决策、被试特质推断和行为知识应用四大核心能力系统评估基础模型，并同时考察个体层面准确性与群体分布层面一致性，揭示当前领先模型在行为科学任务上仍面临挑战。 |
| [^431] | [Walk fast but be careful: Understanding Parallel Sampling in Masked Diffusion](https://arxiv.org/abs/2606.22976) | 本文利用图上随机游走作为可验证沙盒，从理论上证明掩码扩散模型中常用的并行去掩码评分策略（如最低熵）并不普遍优于随机并行采样，性能关键取决于图的条件依赖结构，并提出了免训练的二分采样器。 |
| [^432] | [GRADE: Graph Representation of LLM Agent Dependency and Execution](https://arxiv.org/abs/2606.22741) | GRADE框架将大语言模型智能体的一次运行表示为包含分级依赖边（观测、声明、推断）的类型化图，并通过实验揭示依赖信息在跨语料库迁移中的失败预测能力比运行规模更稳健，但其增益高度依赖于所选择的探针模型。 |
| [^433] | [Reinforcement Learning-Based Traffic Signal Control for IoT-Enabled Intersections](https://arxiv.org/abs/2606.22108) | 本文为科威特一个信号控制交叉口开发了基于PPO的边缘智能强化学习信号控制器，仅利用本地观测的交通状态动态分配绿灯时长、无需未来需求信息或集中式协调，基于真实交通数据的仿真表明其性能优于固定时长控制和车辆感应控制。 |
| [^434] | [SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs](https://arxiv.org/abs/2606.16193) | SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。 |
| [^435] | [An RRAM-based Hardware Implementation of a Radial Basis Function Neuron for Edge Classifiers](https://arxiv.org/abs/2606.14739) | 本文提出一种基于RRAM模拟内容寻址存储器的径向基函数神经元硬件设计，其中每个可配置的TXL单元充当感受野神经元，为边缘设备上的度量分类与在线适应提供了高效的硬件实现方案。 |
| [^436] | [Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees](https://arxiv.org/abs/2606.14289) | 本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。 |
| [^437] | [Hidden in Plain Sight: Benchmarking Agent Safety Against Decomposition Attacks with DECOMPBENCH](https://arxiv.org/abs/2606.13994) | 该论文提出了DeCompBench——首个专门评估智能体在分解攻击下安全性的基准测试，其采用图结构框架和“分解式设计”原则，将有害任务分解为各自无害、单独执行可绕过安全机制、但累积起来能实现恶意意图的现实可执行子任务。 |
| [^438] | [Cross-Agent Learning Signals Enable Coordinated Role-Decomposed LLM Training](https://arxiv.org/abs/2606.10684) | DAC 是一个角色分解训练框架，通过跨智能体的角色特定交叉验证奖励将搜索与生成解耦训练，并借助弃权机制与困难正例证据增强实现精细化的信用分配，从而在多个问答基准上持续超越强基线。 |
| [^439] | [Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions](https://arxiv.org/abs/2606.07399) | ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。 |
| [^440] | [Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training](https://arxiv.org/abs/2606.06539) | 本文提出DTG-FF方法在九个真实数据基准上创下FF家族新纪录，但系统审计发现层局部Forward-Forward训练与反向传播的差距随数据规模和类别数增大而扩大，表明合成小尺寸基准高估了FF的实际扩展潜力。 |
| [^441] | [Detecting Control and Response Events for AI-Enabled Radio Access Networks](https://arxiv.org/abs/2606.06459) | 本文针对AI-RAN/O-RAN中并发AI控制功能可能相互干扰的问题，提出从含噪声的参数与KPI遥测数据中检测真实控制动作及KPI相应响应事件的方法，以支撑控制参数与网络性能之间可解释依赖关系的学习。 |
| [^442] | [A prism hierarchy of learning regimes in large linear autoencoders](https://arxiv.org/abs/2606.05335) | 本文提出用三棱柱的层级结构系统地刻画大型权重绑定线性自编码器的五个基本极端学习区间（大数据、小数据、平均场、窄潜层、自由），为此类非线性于权重的模型的学习动态提供了系统化的理论图景。 |
| [^443] | [Gradient-based optimization of nuclear criticality experiments using neural surrogate eigenvalue sensitivities](https://arxiv.org/abs/2606.04033) | 本文提出利用物理信息神经网络的可微性进行基于梯度的核临界实验几何优化，通过最大化与目标技术的相关系数 $c_k$ 来自动设计具有高中子学相似性的新临界实验，并能搜索栅格内材料组合的组合设计空间。 |
| [^444] | [MAdam: Metric-Aware Multi-Objective Adam](https://arxiv.org/abs/2606.03904) | 提出 MAdam——一个即插即用的度量感知多目标 Adam 包装器，在完全不改动求解器和优化器的前提下，纠正了 MOO 求解器与 Adam 耦合时因二阶矩纠缠导致的偏好权重失配，以及自适应度量扭曲欧氏几何导致的几何失配这两大系统性问题。 |
| [^445] | [Towards Regret Guarantees for One-Step Lookahead Bayesian Optimization](https://arxiv.org/abs/2606.00956) | 本文提出了一种仅需后验采样和蒙特卡洛近似的单步前瞻贝叶斯优化方法OVR，并通过正则化变体首次建立了趋于零的贝叶斯期望简单遗憾上界保证。 |
| [^446] | [StressDream: Steering Video World Models for Robust Policy Evaluation and Improvement](https://arxiv.org/abs/2606.00267) | StressDream通过优化扩散世界模型的初始噪声，将模型的想象引导向高影响但合理的场景，从而实现更稳健的机器人策略评估与改进。 |
| [^447] | [Modeling Robotics Dataset Construction as an Artifact-Based Build Process](https://arxiv.org/abs/2606.00162) | 该论文将机器人数据集构建建模为基于依赖图的构件化构建过程，并实现了开源Bazel扩展Bagzel，相比传统顺序脚本在热构建中最高加速386倍以上，显著提升了数据集生成的可复现性与迭代效率。 |
| [^448] | [Score Broadcast and Decorrelation: A General Framework for Broadcast-Based Credit Assignment](https://arxiv.org/abs/2605.30638) | 提出SBD框架，通过建立输出分数与隐藏层激活之间的正交性原理，将基于广播的信用分配机制推广并统一到一般可微损失族。 |
| [^449] | [Latent Performance Profiling of Large Language Models](https://arxiv.org/abs/2605.30018) | 提出潜在性能剖析（LPP）框架，通过分析大语言模型的隐藏层激活与输出分布，从内部状态中提取与任务无关的性能诊断指标，弥补传统基准测试评估的不足。 |
| [^450] | [LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation](https://arxiv.org/abs/2605.29280) | LoopFM通过将基础模型的中间嵌入结构化为下游垂直模型的输入特征（如用户历史序列），开辟了高带宽知识传递通道，在无需实时FM推理的情况下实现了显著AUC提升，并与知识蒸馏形成互补。 |
| [^451] | [Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models](https://arxiv.org/abs/2605.27813) | 提出残差化时序稀疏自编码器（ReSAE），通过在相邻时间步间拟合线性预测器并对残差进行稀疏建模，实现从扩散模型完整激活轨迹中学习可解释特征。 |
| [^452] | [Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks](https://arxiv.org/abs/2605.26526) | 该研究揭示开放权重LLM的安全防御存在漏洞：无需梯度优化或微调的简单越狱攻击（如abliteration和prefilling）即可绕过防护，实现有害用途。 |
| [^453] | [Fourier Feature Pyramids for Physics-Informed Neural Networks](https://arxiv.org/abs/2605.24278) | 该论文提出了名为beignet的神经场架构，用可训练的多分辨率傅里叶特征金字塔取代随机傅里叶特征嵌入，从而提升物理信息神经网络求解偏微分方程的精度与计算效率。 |
| [^454] | [Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents](https://arxiv.org/abs/2605.21347) | 该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。 |
| [^455] | [Set-Valued Policy Learning](https://arxiv.org/abs/2605.19830) | 提出集合值策略学习范式，通过输出有价值治疗方案的集合（其基数反映推荐的不确定性）来更好地支持临床决策，并利用选择函数定义集合策略价值及开发双重稳健估计器。 |
| [^456] | [Brain alignment of reasoning and action representations from vision-language and action models during naturalistic gameplay](https://arxiv.org/abs/2605.19352) | 该研究首次将视觉语言模型和大动作模型在自然雅达利游戏场景下的推理与动作表征同玩家fMRI大脑活动进行对齐，揭示了动作导向与推理导向提示对模型内部表征及脑对齐效果的塑造作用。 |
| [^457] | [QLIF-CAST: Quantum Leaky-Integrate-and-Fire for Time-Series Weather Forecasting](https://arxiv.org/abs/2605.18333) | 本文提出QLIF-CAST模型，将量子泄漏积分激发脉冲神经网络从分类任务扩展到时间序列回归，用于短期多变量天气预报，通过将神经元激发状态编码为单量子比特叠加态并嵌入混合量子-经典循环架构，在与参数匹配的经典LIF基线对比中降低了15.4%的MSE和4.4%的MAE。 |
| [^458] | [Exact Convex Reformulations of Linear Neural Networks via Completely Positive Lifting](https://arxiv.org/abs/2605.17692) | 本文证明了深度线性神经网络在平方损失下的训练问题可通过完全正提升被精确重构为广义完全正锥上的凸优化问题，且提升维度仅取决于输入输出维度，与网络深度和数据点数量无关。 |
| [^459] | [Constrained latent state modeling: A unifying perspective on representation learning under competing constraints](https://arxiv.org/abs/2605.15995) | 该论文提出约束潜在状态建模（CLSM）这一统一概念框架，用预测充分性、最小性、时间一致性、观测兼容性、抗干扰不变性和结构约束六个互补特性来刻画潜在表示，从而统一解释和比较各类表示学习方法，并阐明约束组合如何提升表示的可辨识性。 |
| [^460] | [A Few Steps Further: Why Defenses Against Malicious Finetuning Erode Under Continued Training](https://arxiv.org/abs/2605.14605) | 现有的抗恶意微调防御都建立在有限的攻击者假设之上，一旦模型权重被公开，攻击者只需在有害数据上再多训练几步就能使这些防御失效。 |
| [^461] | [Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models](https://arxiv.org/abs/2605.10971) | 该论文发现从自回归模型移植的均匀干预调度方式在离散扩散语言模型上低效且损害生成质量，并通过稀疏自编码器揭示不同属性（如主题、情感）在去噪过程中具有差异显著的形成时间表，据此提出一种自适应调度机制，将干预集中在各属性正在形成的阶段，从而实现更高效且不破坏质量的多属性引导。 |
| [^462] | [The Value of Mechanistic Priors in Sequential Decision Making](https://arxiv.org/abs/2605.10018) | 本文提出“机制信息”这一新概念，从理论上证明在序贯决策中机制先验可使贝叶斯遗憾随残差熵缩放，从而带来 $H(\mu)/H_{\mathrm{mech}}$ 的样本复杂度降低，并给出可计算的试验前模型证书。 |
| [^463] | [Restricting the Model, Missing the System: Measurement and Accountability in Offensive AI Governance](https://arxiv.org/abs/2605.09504) | 该论文指出当前的AI进攻能力测量工具既夸大危害、又将本属于整个系统的能力错误归因于模型本身，因此AI治理不应仅限于限制模型访问，而需要转向系统级的、基于实际危害的能力评估与问责机制。 |
| [^464] | [Geometry-Aware Discretization Error of Diffusion Models](https://arxiv.org/abs/2605.08392) | 该论文针对光滑逆向扩散过程推导出Euler-Maruyama弱误差与Frechet误差的渐近精确小步长展开式，并据此依据目标数据的协方差谱来优化噪声调度、重缩放系数和随机性系数等扩散采样参数。 |
| [^465] | [TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning](https://arxiv.org/abs/2605.07943) | 本文提出了TAVIS基准，通过头戴与固定摄像头的配对对比协议、创新的GALT预期注视指标以及ID/OOD泛化评估，在两个人形本体上系统量化了主动视觉在模仿学习中的贡献及适用条件。 |
| [^466] | [Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders](https://arxiv.org/abs/2605.07922) | 该论文提出Tree SAE，通过将激活覆盖约束与新颖的重构条件相结合，使稀疏自编码器能够直接从特征集内部学习层次化特征结构，克服了仅依赖激活覆盖条件时易产生语义无关父子关系误判的缺陷。 |
| [^467] | [Empirical Evidence for Simply Connected Decision Regions in Image Classifiers](https://arxiv.org/abs/2605.06380) | 本文通过自适应四边形网格填充实验首次提供了实证证据，表明预训练图像分类器中同标签决策区域是单连通的，即区域内的任意环路都可以被区域内曲面填充。 |
| [^468] | [The Metagame of Interpretability and Meta-Attributions](https://arxiv.org/abs/2605.06295) | 提出“元博弈”框架，将特征的归因值视为特征间的合作博弈并计算其Shapley值，从而得到方向性元归因，使任意基于梯度或注意力的归因方法都能泛化到二阶交互效应，并证明了元归因之和恰好等于其所解释的一阶归因。 |
| [^469] | [RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy](https://arxiv.org/abs/2605.02003) | 本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。 |
| [^470] | [From Packets to Patterns: Interpreting Encrypted Network Traffic as Longitudinal Behavioral Signals](https://arxiv.org/abs/2605.01616) | 该研究表明，加密智能手机网络流量可作为被动感知方式，通过带每用户适配器的 transformer 和稀疏自编码器提取可解释的行为特征，有效捕捉与睡眠、压力和孤独感相关且具有不同时间结构的行为模式。 |
| [^471] | [Continuous Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2604.20021) | 本文首次建立了不确定条件下连续查询空间中LLM语义响应缓存的严格理论框架，通过动态ε-网离散化与核岭回归相结合，突破了传统有限离散查询假设，实现低成本LLM服务。 |
| [^472] | [Perturbation Sensitivity of Maximum-Likelihood Pairwise Ranking in Computational Decision Systems](https://arxiv.org/abs/2604.17805) | 本文将极大似然成对排序的协同扰动形式化为预算约束子集选择问题，提出自适应子集选择攻击（ASSA）这一可扩展搜索方法，并通过实验证明该排序方法对较小但协同的扰动会表现出显著的、依赖运行状态的敏感性。 |
| [^473] | [Geometric Probing for Algorithm Selection in Continuous Black-Box Optimization](https://arxiv.org/abs/2604.09095) | 该论文提出一种几何探测框架，通过多尺度二维约束采样并以有效性感知的卷积与置换不变聚合进行视觉编码，为连续黑盒优化的算法选择提供与ELA类特征互补、且在问题级迁移下更优的性能信息。 |
| [^474] | [Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI](https://arxiv.org/abs/2604.08015) | 提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。 |
| [^475] | [A Clinical Point Cloud Paradigm for In-Hospital Mortality Prediction from Multi-Level Incomplete Multimodal EHRs](https://arxiv.org/abs/2604.04614) | 该论文提出HealthPoint（HP），一种统一的临床点云范式，将异构临床事件表示为连续四维空间中的点，从而在不进行刚性对齐或丢弃数据的情况下，对多层级不完整的多模态电子健康记录进行建模，用于院内死亡率预测。 |
| [^476] | [Out-of-Air Computation: Enabling Structured Function Extraction from Wireless Superposition](https://arxiv.org/abs/2604.04312) | 本文提出“空外计算”这一面向提取的空中计算新范式，基于联合信源信道编码和多层嵌套格架构，从结构化的无线叠加信号中提取目标函数，无需让信道逼近理想计算媒介，且可直接处理连续值数据。 |
| [^477] | [Neural Global Optimization via Iterative Refinement from Noisy Samples](https://arxiv.org/abs/2604.03614) | 本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。 |
| [^478] | [A Variational Latent-Space Framework for Uncertainty-Aware Spectral Image Emulation](https://arxiv.org/abs/2603.21911) | 该论文提出一种基于变分自编码器的光谱图像仿真框架，将仿真任务表述为参数条件化的潜变量问题，在实现快速推理的同时提供逐像素不确定性估计。 |
| [^479] | [Bi-CamoDiffusion: A Boundary-informed Diffusion Approach for Camouflaged Object Detection](https://arxiv.org/abs/2603.13357) | Bi-CamoDiffusion通过无参数边缘先验注入机制以及统一空间精度、结构约束和不确定性监督的优化目标，显著提升了伪装目标检测的边界清晰度和整体检测性能。 |
| [^480] | [Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting](https://arxiv.org/abs/2603.11938) | 提出ProtoSR方法，利用指令微调大语言模型从8万多份自由文本放射报告中自动构建以视觉原型表示的多模态知识库，将自由文本中隐含的细粒度知识注入结构化报告生成，从而提升细粒度结构化放射学报告自动化的可靠性与一致性。 |
| [^481] | [The Trace Is the State: Exact Credit Assignment for LLM Agent Teams](https://arxiv.org/abs/2603.06859) | 提出C3方法，通过在决策点替换单条消息并继续运行至最终奖励来执行反事实，从而为LLM智能体团队实现无偏且精确的信用分配，其估计方差与智能体数量无关。 |
| [^482] | [Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators](https://arxiv.org/abs/2602.22647) | 提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。 |
| [^483] | [Fluids You Can Trust: Property-Preserving Operator Learning for Incompressible Flows](https://arxiv.org/abs/2602.15472) | 提出了一种保性质的核基算子学习方法，能够使预测速度场在解析意义上同时严格满足不可压缩性、周期性等物理性质，并具备通用逼近能力与先验收敛速率保证。 |
| [^484] | [Just on Time: Token-Level Early Stopping for Diffusion Language Models](https://arxiv.org/abs/2602.11133) | 本文提出一种无需训练的词元级早停方法，利用模型预测和局部上下文的轻量级信号动态判断每个词元的收敛时机并提前冻结，大幅减少扩散语言模型的去噪步数，在保持生成质量的同时显著提升生成效率。 |
| [^485] | [ELROND: Exploring and decomposing intrinsic capabilities of diffusion models](https://arxiv.org/abs/2602.10216) | 提出ELROND方法，通过反向传播固定提示的随机生成结果之间的差异获得梯度，并利用主成分分析或稀疏自编码器将其分解为可解释方向，从而恢复扩散模型生成流形在给定条件下的切空间，实现对模型内在输出变化的系统性探索。 |
| [^486] | [ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling](https://arxiv.org/abs/2602.09009) | 该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。 |
| [^487] | [SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning](https://arxiv.org/abs/2602.08234) | SkillRL提出了一种通过自动技能发现构建层次化技能库SkillBank，并利用递归演化机制使技能库在强化学习中与智能体策略共同演化的框架，从而显著减少token开销并提升智能体的泛化能力。 |
| [^488] | [BONSAI: Bayesian Optimization with Natural Simplicity and Interpretability](https://arxiv.org/abs/2602.07144) | 提出了一种感知默认配置的贝叶斯优化策略BONSAI，它能在显式控制采集价值损失的前提下剪除对默认配置的低影响偏离，从而实现更简洁、可解释且易于审查的优化推荐。 |
| [^489] | [DiTS: Multimodal Diffusion Transformers Are Time Series Forecasters](https://arxiv.org/abs/2602.06597) | DiTS提出了一种多模态扩散Transformer，将内生目标与外生协变量视为不同的模态，并利用它们共享的时间坐标实现细粒度条件引导，从而完成协变量感知的概率时间序列预测。 |
| [^490] | [Attention-Mass Condensation for Sparse Decoding](https://arxiv.org/abs/2602.06317) | 该论文通过精确的遗漏质量恒等式和下游边距条件形式化了稀疏解码中注意力质量保留与稳定贪心决策之间的区别，并实验证明：尽管稀疏解码在分布质量上可接近稠密解码，但没有任何运行能完全复现稠密贪心解码的输出。 |
| [^491] | [Routing-Aware Safety Alignment for Mixture-of-Experts Models](https://arxiv.org/abs/2602.04448) | 提出RASA框架，通过识别被越狱攻击不成比例激活的安全关键专家、在固定路由下仅微调这些专家并强制路由一致性，实现了混合专家模型的路由感知安全对齐。 |
| [^492] | [Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)](https://arxiv.org/abs/2602.03317) | 提出一种物理结构变分自编码器（PS-VAE），通过融合可微分自旋物理模拟器与自监督学习，实现定量分子MRI中体素级多参数后验分布的快速提取与不确定性量化。 |
| [^493] | [A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior](https://arxiv.org/abs/2602.02639) | 该论文提出归一化可模拟性增益（NSG）这一新指标，从预测价值的角度证明了LLM自我解释的忠实性，实验表明自我解释能将对模型行为的预测能力提升11-37%。 |
| [^494] | [Continuous-Utility Direct Preference Optimization](https://arxiv.org/abs/2602.00931) | 提出CU-DPO框架，用连续效用分数取代二元偏好标签来对齐多种认知策略，并证明其在样本复杂度上相比二元偏好可获得Θ(K log K)的改进。 |
| [^495] | [Order-Optimal Sample Complexity of Rectified Flows](https://arxiv.org/abs/2601.20250) | 本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。 |
| [^496] | [Machine learning modularity](https://arxiv.org/abs/2601.01779) | 本文提出首个基于 Transformer 序列到序列架构与动态批处理算法的机器学习框架，能够学习 SL(2,Z) 与 SL(3,Z) 模变换，自动将涉及椭圆伽马函数的高度打乱表达式化简为规范形式，在分布内测试中准确率超过 99%，并在更深的打乱深度等外推任务中仍保持 90% 以上的准确率，证明模型真正内化了模变换的代数规则。 |
| [^497] | [PrismSSL: One Interface, Many Modalities; A Single-Interface Library for Multimodal Self-Supervised Learning](https://arxiv.org/abs/2511.17776) | PrismSSL是一个单一接口的Python库，将音频、视觉、图和跨模态的最先进自监督学习方法统一在模块化代码库中，支持少量代码即可训练与复现基准，并集成分布式训练、超参搜索、LoRA微调等实用功能。 |
| [^498] | [Estimating Model-Level Membership Inference Vulnerability Without Reference Models](https://arxiv.org/abs/2510.19773) | 提出了一种无需训练任何参考模型、仅利用目标模型的训练与测试损失分布即可估计模型对最强成员推断攻击LiRA脆弱性的新方法，并揭示了不同模型状态对应不同损失统计量的适用区间。 |
| [^499] | [Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM](https://arxiv.org/abs/2510.05544) | 提出基于激活压缩误差理论上界的帕累托引导低秩压缩框架PGSVD，通过异构秩分配在相同压缩率下为LLM/VLM实现更高精度与推理加速。 |
| [^500] | [HomID : Benchmarking Intrinsic Dimension Estimators on Homogenous Manifolds with Anisotropic Embeddings](https://arxiv.org/abs/2510.01335) | 本文提出了 HomID 基准——一组具有各向异性嵌入的同质流形集合，用以揭示现有内在维度估计器在标准基准上表现良好、但在各向异性条件下会系统性失效，并证明了各向异性畸变通过使其所依赖的分布发生系统性偏移而导致估计误差。 |
| [^501] | [Circuit realization and hardware linearization of monotone operator equilibrium networks](https://arxiv.org/abs/2509.13793) | 该论文证明电阻-二极管电路的端口行为等价于ReLU单调算子平衡网络，并提出硬件线性化方法使梯度可直接在硬件中计算，实现了神经网络在模拟硬件中的构建与训练。 |
| [^502] | [Modified Loss of Momentum Gradient Descent: Fine-Grained Analysis](https://arxiv.org/abs/2509.08483) | 该论文证明当步长足够小时，重球动量梯度下降在指数吸引的不变流形上精确等价于带修正损失的普通梯度下降，能以任意有限阶精度刻画该修正损失，并在其无记忆近似的组合结构中发现了介于欧拉多项式与Narayana多项式之间的一类新的β多项式族。 |
| [^503] | [Scaling Legal AI: Benchmarking Mamba and Transformers for Statutory Classification and Case Law Retrieval](https://arxiv.org/abs/2509.00141) | 该论文构建了一个初步基准测试，比较Mamba等线性复杂度状态空间模型与BERT、DeBERTa、Longformer等Transformer模型在四个法律分类任务和两个判例检索任务上的表现，发现最强的状态空间模型性能与Transformer相差约1.3个百分点以内，证明了SSM作为处理长篇法律文档的有前景替代方案的潜力。 |
| [^504] | [Persistence Paradox in Dynamic Science: Evidence from the Deep Learning Revolution](https://arxiv.org/abs/2506.22729) | 本研究以2012年AlexNet引发的深度学习革命为背景，通过分析5000多名机器学习科学家20年的职业轨迹，揭示了坚持的悖论：以往成功或隶属成熟团队的科学家在范式转变中适应更慢，坚持虽能维持生产力，却在2012年后阻碍了其科学成功。 |
| [^505] | [Causal Posterior Estimation](https://arxiv.org/abs/2505.21468) | 提出因果后验估计（CPE）方法，将模型图结构中的条件依赖关系直接硬编码进基于流匹配的神经网络架构，在似然函数难以计算的模拟器模型中实现高精度的贝叶斯后验推断。 |
| [^506] | [APE: Selective Fine-tuning with Acceptance Criteria for Language Model Adaptation](https://arxiv.org/abs/2505.19912) | APE 是一种受进化优化启发的选择性微调方法，通过在小数据子集上评估多个候选参数更新并仅接受超过性能阈值者，在保持模型稳定性的同时以极少计算资源实现大型语言模型的高效适配。 |
| [^507] | [Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories](https://arxiv.org/abs/2505.08088) | 该论文提出了一种基于图的楼层分离新方法，通过将Wi-Fi指纹轨迹构建为图并利用Node2Vec嵌入和K-means聚类识别楼层，在华为大学挑战赛2021数据集上取得了优于传统社区发现算法的效果，并公开了数据集与代码。 |
| [^508] | [The Utility and Complexity of in- and out-of-Distribution Machine Unlearning](https://arxiv.org/abs/2412.09119) | 本文对机器遗忘进行了严格的复杂度与效用分析，证明带输出扰动的经验风险最小化对分布内遗忘数据能实现紧密的权衡，而分布外遗忘数据则面临根本性挑战。 |
| [^509] | [Allocation Stability and Wald Inference under Variance-Aware UCB](https://arxiv.org/abs/2412.08843) | 本文为双臂方差感知UCB策略给出了最优臂分配稳定性的尖锐判据，并证明即使最优臂计数不稳定，只要拉取次数与奖励方差的乘积依概率发散，臂均值线性组合的Wald统计量仍渐近服从标准正态分布，从而表明分配稳定性并非高斯推断的必要条件。 |
| [^510] | [Learning in the Recurrent State: Gradient Descent with Linear Recurrent Networks](https://arxiv.org/abs/2410.11687) | 本文提出了GRIL——一种对角线性循环网络，通过将梯度步骤分解为短窗口叉积写入和乘法读出，使线性循环网络能够在循环状态中于单次正向传播内实现上下文梯度下降，从而将基于梯度的上下文学习能力扩展到线性时间复杂度的序列模型。 |
| [^511] | [Requirement-Based Testing: Enhancing Reinforcement Learning with Game Theory](https://arxiv.org/abs/2407.18994) | 本文提出一种基于博弈论的启发式蒙特卡洛树搜索方法，用于从自动机形式的功能需求中自动生成黑盒测试用例，实验表明该启发式方法加速了算法收敛并提升了测试性能。 |
| [^512] | [Speech Emotion Recognition Using CNN and Its Use Case in Digital Healthcare](https://arxiv.org/abs/2406.10741) | 本研究利用卷积神经网络（CNN）从语音音频中识别并标注情感，通过精确率、召回率和F1分数进行评估，展现了其在数字医疗领域的应用价值。 |

# 详细

[^1]: 在可验证奖励强化学习（RLVR）中将探索与优化解耦

    Decoupling Exploration from Optimization in RLVR

    [https://arxiv.org/abs/2610.10536](https://arxiv.org/abs/2610.10536)

    提出探索-蒸馏框架，将RLVR中的探索与优化解耦：先用新颖性奖励训练探索者策略，再过滤其轨迹并蒸馏到不带新颖性奖励的学生策略中，从而在实现新策略发现的同时避免模型质量退化。

    

    现代语言模型在已经训练好的检查点基础上进行可验证奖励的强化学习（RLVR）。RLVR的一个关键承诺是发现新的推理策略。原则上，模型可以采样出其先前训练数据中不存在的新颖想法。然而在实践中，用强新颖性激励来增强RLVR的成功有限，并且可能会降低模型质量。由于可验证奖励仅监督模型知识和行为中很狭窄的一部分，这种退化难以恢复。因此，我们在一个称为探索-蒸馏的框架中将探索与优化解耦。我们训练一个或多个在奖励中带有新颖性奖励的探索者策略，对其轨迹进行正确性和质量过滤，然后将它们蒸馏到一个单独的学生策略中。学生策略随后在不带新颖性奖励的情况下进行训练。我们对上述过程重复多个轮次，交替……

    arXiv:2610.10536v1 Announce Type: cross  Abstract: Modern language models undergo reinforcement learning with verifiable rewards (RLVR) on top of already-trained checkpoints. A key promise of RLVR is the discovery of new reasoning strategies. In principle, a model can sample novel ideas absent from its prior training data. In practice, however, augmenting RLVR with strong novelty incentives has seen limited success and can degrade model quality. Because verifiable rewards supervise only a narrow slice of the model's knowledge and behavior, such degradations are difficult to recover from. Instead, we decouple exploration from optimization in a framework we call Exploration-Distillation (ExpDis). We train one or more explorer policies with a novelty bonus in the reward, filter their trajectories for correctness and quality, and distill them into a separate student policy. The student policy is then trained without a novelty bonus. We repeat the above procedure for several rounds, alterna
    
[^2]: 重尾噪声下的去中心化SGD：最优收敛速率与梯度裁剪的作用

    Decentralized SGD under Heavy-Tailed Noise: Optimal Convergence Rates and the Role of Gradient Clipping

    [https://arxiv.org/abs/2610.10527](https://arxiv.org/abs/2610.10527)

    本文证明了带梯度裁剪的去中心化SGD（DSGD）在重尾噪声下对光滑非凸目标能够达到阶最优收敛速率，肯定地回答了简单的基线去中心化方法结合非线性操作即可实现最优收敛这一问题。

    

    重尾噪声在现代机器学习中被广泛观察到，这促使人们使用梯度裁剪和归一化等方法。虽然这些方法在中心化设置中已被充分理解，但在去中心化设置中却知之甚少，因为在去中心化设置中对局部梯度施加非线性会同时影响优化和共识。近期关于去中心化非凸优化的工作在重尾噪声下研究了裁剪和归一化两种方法，其中裁剪产生了次优的收敛速率，而归一化则需要局部动量或小批量采样才能收敛。这引出了一个问题：使用非线性的基线去中心化方法能否在重尾噪声下达到最优收敛速率？我们通过带裁剪的去中心化SGD（DSGD）给出了肯定的回答。对于在有界p阶矩噪声（p ∈ (1,2]）下的光滑非凸代价函数，我们证明了带裁剪的DSGD在高概率和期望意义下均达到了阶最优收敛速率。

    arXiv:2610.10527v1 Announce Type: cross  Abstract: Heavy-tailed noise has been widely observed in modern machine learning, motivating the use of methods like gradient clipping and normalization. While these methods are well understood in centralized settings, much less is known in decentralized ones, where applying a nonlinearity to local gradients affects both optimization and consensus. Recent works on decentralized non-convex optimization have studied both clipping and normalization under heavy-tailed noise, with clipping yielding suboptimal rates and normalization needing local momentum or mini-batches to converge. This raises the question: can a baseline decentralized method using a nonlinearity achieve optimal convergence rates under heavy-tailed noise? We answer affirmatively with clipped decentralized SGD ($\mathtt{DSGD}$). For smooth non-convex costs under bounded $p$-th moment noise, $p \in (1,2]$, we show that clipped $\mathtt{DSGD}$ achieves order-optimal rates both with hi
    
[^3]: 先改述，再行动：视觉-语言-动作模型中语言敏感性的表征与缓解

    Rephrase Before You Act: Characterizing and Mitigating Language Sensitivity in Vision-Language-Action Models

    [https://arxiv.org/abs/2610.10526](https://arxiv.org/abs/2610.10526)

    本文揭示了视觉-语言-动作模型对指令措辞的极端敏感性（单词改动可使成功率波动数十个百分点），并提出无需修改策略、由大语言模型将措辞评分证据提炼为十余条改述规则并在部署时应用的方法来缓解该问题。

    

    视觉-语言-动作模型（VLA）对指令措辞极其敏感，且并不继承其底层视觉-语言模型所具备的语言鲁棒性。仅仅一个词的改动就能使成功率波动数十个百分点：π0.5 在 LIBERO 灶台任务中，对 "switch on the stove"（打开灶台）的成功率为 100%，而对 "switch on the hot plate"（打开加热板）仅为 2%；即使是用改述数据增强微调过的 π0 检查点，仍表现出高达 61 个百分点的波动。我们通过经过统计检验的单次编辑波动分析以及预言机短语搜索来刻画这种敏感性，结果表明仅靠措辞差异就能几乎抹平分布内任务与分布外任务之间 21 个百分点的差距。随后，我们在不修改策略本身的前提下降低这种敏感性。由于这种敏感性是系统性的，它可以被表达为显式规则：我们对少量训练任务的多种措辞进行评分，让大型语言模型将这些证据提炼为十到二十条改述规则，并在部署时加以应用……

    arXiv:2610.10526v1 Announce Type: cross  Abstract: Vision-language-action models (VLAs) are strikingly sensitive to instruction phrasing and do not inherit the language robustness of the vision-language models they are built on. A one-word edit can move success by tens of points: $\pi_{0.5}$ turns on a LIBERO stove 100% of the time for "switch on the stove" and 2% for "switch on the hot plate", and a $\pi_0$ checkpoint finetuned with rephrase augmentation still shows swings of up to 61 points. We characterize this sensitivity with statistically tested single-edit swings and an oracle phrase search, which shows that phrasing alone nearly closes the 21-point gap between in-distribution and out-of-distribution tasks. We then reduce it without modifying the policy. Because the sensitivity is systematic, it can be expressed as explicit rules: we score many phrasings of a few training tasks, have a large language model distill the evidence into ten to twenty rephrasing rules, and at deployme
    
[^4]: 蒸馏图几何：从图神经网络到多层感知机的知识鸿沟

    Distilling Graph Geometry: Knowledge Gap from GNNs to MLPs

    [https://arxiv.org/abs/2610.10520](https://arxiv.org/abs/2610.10520)

    提出G²MLP，一种由Ollivier-Ricci曲率引导的训练时蒸馏框架，通过识别稀疏图上的谱欠拟合与稠密图上的谱过拟合两种失效模式，使无需图结构的MLP学生模型能够保留GNN教师模型的图诱导几何结构。

    

    从GNN到MLP的知识蒸馏旨在保留消息传递教师模型的预测精度的同时，在推理阶段部署无需图结构的MLP。现有方法主要传递节点级预测或采用基于置信度的重加权，但它们并未指明学生模型应当在哪里保留教师模型由图诱导的几何结构。我们证明这种缺失会导致学生表示空间中出现两种谱失效模式：在稀疏图上，学生模型会出现谱欠拟合，丢失集中在边界区域附近的高能教师方向；在稠密图上，学生模型会出现谱过拟合，保留了教师模型已通过聚合坍缩掉的虚假方向。受能量加权的师生对齐目标启发，我们提出了图几何感知MLP（G²MLP），这是一个由Ollivier-Ricci曲率引导的训练时蒸馏框架。曲率能够识别上述两种谱误差集中出现的位置。

    arXiv:2610.10520v1 Announce Type: new  Abstract: GNN-to-MLP distillation aims to retain the predictive accuracy of a message-passing teacher while deploying a graph-free MLP at inference. Existing methods mainly transfer node-wise predictions or use confidence-based reweighting, but they do not specify where the student should preserve the teacher's graph-induced geometry. We show that this omission leads to two spectral failure modes in the student's representation space. On sparse graphs, the student suffers from spectral underfit, missing high-energy teacher directions concentrated near boundary regions. On dense graphs, it suffers from spectral overfit, retaining spurious directions that the teacher has collapsed through aggregation. Motivated by an energy-weighted teacher-student alignment objective, we propose Graph Geometry-aware MLP (G^2MLP), a training-time distillation framework guided by Ollivier-Ricci curvature. Curvature identifies where the two spectral errors concentrate
    
[^5]: 为什么仅遗忘式机器遗忘需要记忆

    Why Forget-Only Unlearning Needs Memorization

    [https://arxiv.org/abs/2610.10519](https://arxiv.org/abs/2610.10519)

    本文证明仅遗忘式机器遗忘（只用训练模型和待遗忘样本、无保留数据）并非总是可行，其可行性取决于学习方法，且算法必须对训练数据进行足够记忆才能处理任意删除请求。

    

    机器遗忘要求设计一种删除算法，其输出接近于在没有被选中遗忘样本的情况下从头重新训练的结果。在这项工作中，我们研究仅遗忘式机器遗忘，即删除算法只接收训练好的模型和需要遗忘的样本，而不保留任何数据或额外的训练信息。我们探讨仅遗忘式遗忘是否总是可行的。我们首先证明这取决于学习方法：不同的数据集可以产生相同的训练模型，但在删除相同样本后却需要非常不同的输出。利用这一观察，我们推导了遗忘算法匹配重新训练效果的精度下界，并在几种标准学习算法上进行了实例化。接着我们问，当仅遗忘式遗忘成功时，必须满足什么条件。为此，我们推导了算法为处理任意删除请求而必须对训练数据进行记忆的下界。对于简单的阈值学习……

    arXiv:2610.10519v1 Announce Type: new  Abstract: Machine unlearning asks for a deletion algorithm whose output is close to retraining from scratch without the selected forget examples. In this work, we study forget-only unlearning, where the deletion algorithm receives only the trained model and the examples to forget, with no retained data or extra training information. We ask whether forget-only unlearning is always possible. We first show that this depends on the learning method: different datasets can produce the same trained model but require very different outputs after the same examples are removed. Using this observation, we derive lower bounds on how accurately unlearning can match retraining and instantiate them for several standard learning algorithms. We then ask what must be true when forget-only unlearning succeeds. To this end, we derive lower bounds on what an algorithm must memorize about the training data to handle arbitrary deletion requests. For simple threshold lea
    
[^6]: SciExam for ENSO：AI智能体能构建气候模型吗？

    SciExam for ENSO: Can AI Agents Build Climate Models?

    [https://arxiv.org/abs/2610.10513](https://arxiv.org/abs/2610.10513)

    该论文提出了SciExam for ENSO基准测试，让AI智能体在六小时内仅凭真实观测数据自主构建ENSO低阶随机气候模型，并用隐藏评分器检验模型能否复现统计特性、恢复隐变量和预测保留年份，结果显示十二个智能体系统中有六个的模型优于已发表的模型。

    

    语言模型智能体越来越多地被要求开展开放式科学研究，然而对其结果的评估通常是对照已知答案、评分标准或语言模型审稿人，这些方法都无法判断一个新的科学模型是否有效。“厄尔尼诺-南方涛动AI科学考试”是一个基准测试，要求智能体基于真实观测数据构建厄尔尼诺-南方涛动（ENSO，年际气候变率的主导模态）的低阶随机模型。在六小时的时间预算内，智能体处理观测数据、编写自己的诊断程序（随后被冻结），并仅以这些诊断结果作为反馈来开发模型。随后由隐藏的评分器测试该模型能否复现ENSO的统计特性、恢复未观测到的变量以及对未纳入训练的年份进行预测，并以同样的方式对已发表的模型进行评分。在十二个智能体系统中，有六个产生的模型得分高于已发表的模型，主要得益于更好的（重建）。

    arXiv:2610.10513v1 Announce Type: cross  Abstract: Language-model agents are increasingly asked to carry out open-ended scientific research, yet their results are usually graded against a known answer, a rubric, or a language-model reviewer, none of which can tell whether a new scientific model is valid. The AI Science Exam for El Nino-Southern Oscillation (SciExam for ENSO) is a benchmark in which agents build low-order stochastic models of ENSO, the dominant mode of interannual climate variability, from real observations. Within a six-hour budget, agents process the observations, write their own diagnostics, which are then frozen, and develop a model using only these diagnostics as feedback. Hidden graders then test whether the model reproduces ENSO's statistics, recovers unobserved variables, and forecasts held-out years, and score a published model in the same way. Across twelve agent systems, six produce models that score higher than the published model, mainly through better reco
    
[^7]: 神谕高效且无参数的不可知平滑在线学习

    Oracle-Efficient and Parameter-Free Agnostic Smoothed Online Learning

    [https://arxiv.org/abs/2610.10499](https://arxiv.org/abs/2610.10499)

    该论文提出了不可知平滑在线学习领域首个神谕高效且无参数的算法，同时摆脱了对基础测度采样访问能力和完美预测标签这两个限制性假设的依赖。

    

    在线学习在许多领域都是一个有吸引力的框架，因为即使数据是相关的或被对抗性选择的，它也能实现有明确定义的学习。然而，这种通用性伴随着高昂的代价，带来了显著的统计和计算障碍。最近，平滑在线学习作为一个有前景的框架应运而生，它在完全对抗性和完全随机性两种设定之间进行衔接，其假设每个协变量的条件分布相对于某个固定基础测度 $\mu$ 的密度至多为 $1/\sigma$，并且已知该框架能够达到与经典学习相当的统计和计算保证，同时仍保留了在线学习的诸多灵活性。然而，现有的神谕高效算法要么需要（i）对基础测度 $\mu$ 的采样访问能力，要么需要（ii）标签能够被某个固定假设完美预测。这两个假设都限制了这些算法的适用性……

    arXiv:2610.10499v1 Announce Type: new  Abstract: Online learning is an attractive framework in many domains because it permits well-defined learning even when data are dependent or chosen adversarially. This generality, however, comes at a steep price, introducing significant statistical and computational barriers. Recently, smoothed online learning has emerged as a promising framework that interpolates between the fully adversarial and fully stochastic settings by assuming that the conditional law of each covariate has density at most $1/\sigma$ with respect to some fixed base measure $\mu$, and it is known to match the statistical and computational guarantees of classical learning while still allowing for much of the flexibility of online learning. However, existing oracle-efficient algorithms require either (i) sampling access to the base measure $\mu$ or (ii) labels that are perfectly predicted by a fixed hypothesis. Both assumptions limit the applicability of these algorithms, in 
    
[^8]: 使用Sentinel-2进行湖泊叶绿素a预测的进化架构搜索

    Evolutionary Architecture Search for Chlorophyll-$a$ Prediction in Lakes using Sentinel-2

    [https://arxiv.org/abs/2610.10496](https://arxiv.org/abs/2610.10496)

    本研究在固定任务、特征与数据划分的条件下，利用正则化进化的神经架构搜索为Sentinel-2湖泊叶绿素a预测自动找到了一个仅含409个参数的小型网络，将保留集AUC从0.790提升至0.820，并收敛出单窄层加RMS归一化、tanh激活、阶梯衰减RMSprop和权重平均的一致设计配方。

    

    带有专家设计光谱特征的小型表格数据集是业务化地球观测中的常态，而应用于这些数据集的网络通常是手工设计的。我们重新审视了这样一个已发表的模型——一个Sentinel-2藻华分类器——在保持原始研究的任务、特征和湖泊级训练/测试划分固定不变的前提下，探究架构搜索能带来什么提升。通过正则化进化方法在扩展的多层感知器空间中进行搜索，并仅依据内部交叉验证AUC进行选择，我们找到了将保留集AUC从0.790提升至0.820、准确率从0.733提升至0.748的网络，同时仅使用409个可训练参数，比最强的手工设计参考模型少26倍。搜索收敛到一个一致的设计配方——单个窄层、RMS归一化、tanh激活、阶梯衰减的RMSprop以及权重平均——这是从业者在默认情况下不太可能达到的。在1.6……

    arXiv:2610.10496v1 Announce Type: cross  Abstract: Small tabular datasets with expert-designed spectral features are the norm in   operational Earth observation, and the networks applied to them are typically   hand-designed. We revisit one such published model -- a Sentinel-2 algal   bloom classifier -- and ask what architecture search adds, holding the task,   the features and the lake-level train/test split of the original study fixed.   Searching an extended multilayer-perceptron space with regularized evolution,   and selecting on inner-cross-validation AUC only, we find networks that   improve held-out AUC from 0.790 to 0.820 and accuracy from 0.733 to 0.748   while using 409 trainable parameters, 26 times fewer than the strongest   hand-designed reference. The search converges on a consistent recipe -- a   single narrow layer, RMS normalisation, $\tanh$ activation, step-decayed   RMSprop and weight averaging -- that a practitioner would be unlikely to   reach by default. At 1.6\
    
[^9]: 均值漂移老虎机的最优臂识别问题

    Best Arm Identification for Bandits with Shifting Means

    [https://arxiv.org/abs/2610.10488](https://arxiv.org/abs/2610.10488)

    该论文针对均值会对抗性漂移但奖励差距保持稳定的新型老虎机环境，提出了重要性加权算法ISM，证明了传统基于广义似然比检验的算法（如Track-and-Stop）在此环境下会失效，而ISM能保持δ-正确性并获得良好的样本复杂度保证。

    

    我们研究了在具有一种新型对抗性扰动的随机环境中的最优臂识别问题，我们将这种扰动命名为“均值漂移”。在经典设定中，K个臂的平均奖励随时间保持稳定，而在均值漂移设定下，只有各臂平均奖励之间的差距Δ保持稳定，而它们的共同漂移量可能在每一轮被对抗性地决定。学习者的目标是在最小化样本复杂度的同时，以高概率识别出最优臂（固定置信度设定）。处理这种漂移需要新的工具：我们证明了采用广义似然比检验（GLRT）停止规则的算法，包括流行的Track-and-Stop算法，在时变漂移下会失效。因此，我们提出了针对均值漂移的重要性加权算法（ISM）。在假设均值以U为界且奖励服从σ²-次高斯分布的条件下，我们证明了ISM具有δ-正确性，并享有良好的样本复杂度保证。

    arXiv:2610.10488v1 Announce Type: cross  Abstract: We study the best arm identification problem in a stochastic environment with a novel form of adversarial perturbations, which we coin Shifting Means. While classically the mean rewards of the $K$ arms are stable in time, in Shifting Means only the gaps $\boldsymbol{\Delta}$ between mean rewards are stable, while their common shift may be determined adversarially in each round. The objective of the learner is to identify the best arm with high probability while minimizing sample complexity (the fixed confidence setting). Handling shifts requires new tools: we show that algorithms employing a Generalized Likelihood Ratio Test (GLRT) stopping rule, including the popular Track-and-Stop, fail under time-varying shifts. Instead, we propose Importance Weights for Shifting Means ($\mathsf{ISM}$). Assuming means bounded by $U$ and $\sigma^2$-sub-Gaussian rewards, we show $\mathsf{ISM}$ to be $\delta$-correct and to enjoy a sample complexity bo
    
[^10]: 正确的双层Softmax采样：修正来自簇规模不均衡与离散度的偏差

    Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion

    [https://arxiv.org/abs/2610.10483](https://arxiv.org/abs/2610.10483)

    本文揭示了双层softmax采样因忽略簇规模不均衡和簇内相似度离散度而产生的系统性采样偏差，并提出了S-2LS和SD-2LS两种修正方法，以几乎零额外计算开销实现了更优的softmax近似采样。

    

    从softmax分布中采样是机器学习中的一项基础操作，但其相对于项目数量的线性复杂度使得精确采样在大规模场景下难以实际应用。双层softmax（2LS）采样是一种流行的替代方案，可实现亚线性时间采样。该方法假设项目被划分为若干簇，2LS首先采样一个簇，然后在该簇内采样一个项目。在本文中，我们表明，尽管具有优势，2LS会引入系统性的不良采样偏差，这些偏差源于对簇的错误加权，即同时忽略了簇规模的不均衡性和簇内相似度的离散度。我们提出了两种采样方法：规模修正的2LS（S-2LS）以及规模与离散度修正的2LS（SD-2LS），它们修正了这些偏差，并以微乎其微甚至为零的额外计算开销提供了可证明更优的softmax近似。在五个大规模数据集上的深入实验验证了我们方法改进后的采样特性。

    arXiv:2610.10483v1 Announce Type: new  Abstract: Sampling from a softmax distribution is a fundamental operation in machine learning, but its linear complexity in the number of items makes exact sampling impractical at scale. Two-level softmax (2LS) sampling is a popular alternative enabling sublinear-time sampling. Assuming items are partitioned into clusters, 2LS first samples a cluster and then an item within it. In this paper, we show that, despite its advantages, 2LS introduces systematic and undesirable sampling biases, which arise from misweighting clusters by ignoring both cluster size imbalance and intra-cluster similarity dispersion. We propose two sampling methods, Size-Corrected 2LS (S-2LS) and Size- and Dispersion-Corrected 2LS (SD-2LS), which correct these biases and provide provably better softmax approximations with negligible to non-existent computational overhead. In-depth experiments on five large-scale datasets validate the improved sampling properties of our method
    
[^11]: 组合每位教师所学到的知识：通过教师相对偏移实现的多教师在线策略蒸馏

    Composing What Each Teacher Learned: Multi-Teacher On-Policy Distillation through Teacher-Relative Shifts

    [https://arxiv.org/abs/2610.10460](https://arxiv.org/abs/2610.10460)

    提出 Δ-MOPD，通过迁移教师相对其基础模型的 logit 偏移并重新锚定到学生的冻结初始状态，消除端点策略中继承的基础模型偏好干扰，从而在多教师在线策略蒸馏的两种设置中降低教师项范数比和目标-学生 KL，实现更纯粹的教师后训练知识传递。

    

    多教师在线策略蒸馏（MOPD）被应用于两种设置。在共同域组合中，多个教师对来自同一提示域的学生每个 rollout 进行评分，其信号共同构成一个单一目标；在路由域蒸馏中，来自不同域的提示被分配给相应的专家教师。这两种设置通常迁移的是每个教师的端点策略，而端点策略将后训练所改变的偏好与教师从其基础模型继承而来的偏好混杂在一起。我们提出 Δ-MOPD，它迁移的是每个教师的“教师减去基础”的 logit 偏移，并将其重新锚定在学生的冻结初始状态上，同时在保持教师选择固定的情况下，将其与端点监督在两种设置中进行比较。我们首先揭示了阻碍端点迁移的机制：继承的基础拉力可能超过后训练偏移。移除它可以降低教师项范数比以及目标与学生之间的 KL 散度。在我们的所有实验中，结果表明偏移（摘要原文在此处被截断）

    arXiv:2610.10460v1 Announce Type: new  Abstract: Multi-teacher on-policy distillation (MOPD) is used in two settings. In common-domain composition, several teachers score each student rollout from one prompt domain and their signals form a single target; in routed-domain distillation, prompts from different domains are assigned to the corresponding specialist. Both settings usually transfer each teacher's endpoint policy, which mixes what post-training changed with preferences inherited from the teacher's base. We introduce $\Delta$-MOPD, which transfers each teacher's teacher-minus-base logit shift re-anchored at the student's frozen initialization, and compare it with endpoint supervision in both settings while holding teacher selection fixed. We first expose the mechanism that impedes endpoint transfer: inherited base pull can exceed the post-training shift. Removing it reduces the teacher-term norm ratio and target--student KL.   Across our experiments, the results suggest that shi
    
[^12]: NeuralBES：面向可扩展建筑能耗建模的可微分、控制感知仿真器

    NeuralBES: A Differentiable, Control-Aware Emulator for Scalable Building Energy Modeling

    [https://arxiv.org/abs/2610.10459](https://arxiv.org/abs/2610.10459)

    NeuralBES提出了一种可微分建筑能耗仿真器，利用共享神经编码器将建筑元数据映射为物理受限的RC热模型参数，并通过对数空间并行扫描高效求解，在保持物理模型可信性的同时实现了跨数百万异构建筑的大规模可扩展能耗建模。

    

    需求侧灵活性，即对住宅能源负荷进行预测、转移和削减，依赖于能在数百万栋异构建筑上被信任的热模型。现有工具迫使人们做出艰难的权衡：以EnergyPlus为代表的高保真物理模拟器虽然准确，但只能顺序运行且需要对每栋建筑单独校准；而纯数据驱动的序列模型虽然易于扩展，却抛弃了使其预测结果可信的物理结构。我们提出NeuralBES（建筑能耗模拟），这是一种可微分仿真器，通过用共享神经编码器对基于电阻-电容（RC）的热模型进行参数化来解决这一权衡：静态建筑元数据（如建筑面积、建造年代和暖通空调类型）被映射为物理受限的电容、热导和设备系数，这些系数成为通过对数空间并行扫描求解的标量线性递推的系数，并由预测-校正环路闭合……

    arXiv:2610.10459v1 Announce Type: new  Abstract: Demand-side flexibility i.e. forecasting, shifting, and curtailing residential energy loads, depends on thermal models trusted across millions of heterogeneous buildings. Existing tools force a hard tradeoff: high-fidelity physics simulators such as EnergyPlus are accurate but sequential and require per-building calibration, while purely data-driven sequence models scale but abandon the physical structure that makes their predictions trustworthy.   We introduce NeuralBES (Building Energy Simulation), a differentiable emulator that resolves this tradeoff by parameterizing a resistance--capacitance (RC) based thermal model with a shared neural encoder: static building metadata such as floor area, vintage, and HVAC type is mapped to physically bounded capacitances, conductances, and equipment coefficients, which become the coefficients of a scalar linear recurrence solved via a log-space parallel scan, and a predictor--corrector loop closes
    
[^13]: 好的自我教师应顺应学生的当前水平：联合在线策略学习与教学

    A Good Self-Teacher Meets the Student Where They Are: Joint On-Policy Learning and Teaching

    [https://arxiv.org/abs/2610.10447](https://arxiv.org/abs/2610.10447)

    该论文指出自蒸馏中特权信息会使教师通过捷径解题、产生与学生当前行为不匹配的监督信号，并提出联合在线策略学习与教学的方法，使教学顺应学生的当前水平。

    

    基于结果奖励的强化学习（RL）面临监督信号稀疏的问题，尤其是在困难的长时程任务中，成功轨迹十分罕见且生成成本高昂。在线策略蒸馏（OPD）提供了一种有吸引力的替代方案，它在学生自身生成的序列上，由更强的教师模型提供密集的token级监督。自蒸馏方法进一步消除了对独立教师模型的需求，其做法是让同一策略以特权信息为条件，从而充当自己的教师。然而，仅靠特权条件化并不能保证由此产生的蒸馏更新能够真正改进学生模型。事实上，特权信息可能导致教师通过学生无法使用的捷径来完成任务，从而产生与学生当前行为不匹配的监督信号。因此，即使是表现更强的教师，其提供的指导也可能反而降低学生的性能。为解决这一问题，我们……（摘要内容在此处截断）

    arXiv:2610.10447v1 Announce Type: new  Abstract: Reinforcement Learning (RL) from outcome rewards suffers from sparse supervision, particularly on difficult, long-horizon tasks where successful trajectories are rare and costly to generate. On-Policy Distillation (OPD) offers an attractive alternative by providing dense token-level supervision from a stronger teacher along the student's own generations. Self-distillation methods further remove the need for a separate teacher model by conditioning the same policy on privileged information to serve as its own teacher. However, privileged conditioning alone does not guarantee that the resulting distillation update improves the student. Indeed, privileged information can lead the teacher to solve tasks through shortcuts unavailable to the student, producing supervision poorly matched to the student's current behavior. Consequently, even a higher-performing teacher can provide guidance that degrades student performance. To address this, we a
    
[^14]: Seq-Flow：具有自滚动误差控制的高效概率预测

    Seq-Flow: Efficient Probabilistic Forecasting with Self-Rollout Error Control

    [https://arxiv.org/abs/2610.10440](https://arxiv.org/abs/2610.10440)

    提出 Seq-Flow 条件流模型，通过将样本从先前预测分布直接输运到更新后的分布，并结合自滚动训练控制误差累积，实现仅需少量流评估的高效概率预测更新。

    

    许多科学预测任务需要在新的观测数据到来时更新关于未来轨迹的分布。传统的扩散模型和流模型每次生成都从高斯噪声开始，往往需要付出大量采样步骤的代价。热启动方法通过复用较早的预测来降低这一成本，但其模型并未被训练来执行预测更新本身，这在少步采样下可能会损害预测质量。在这项工作中，我们提出了 Seq-Flow，这是一种条件流模型，其常微分方程（ODE）将样本从先前的预测分布输运到更新后的预测分布。由于连续的预测之间通常只有适度的差异，这种输运从一个信息丰富的分布出发，只需少量的流评估即可产生准确的更新。递归复用也带来一个挑战：一次预测中的误差会成为后续流初始状态中的误差。我们通过自滚动训练来解决这一问题，在训练中使用移动平均（摘要在此处截断）

    arXiv:2610.10440v1 Announce Type: new  Abstract: Many scientific forecasting tasks require updating a distribution over future trajectories as new observations arrive. Conventional diffusion and flow models generate each forecast from Gaussian noise, often at the cost of many sampling steps. Warm-start methods reuse earlier predictions to reduce this cost, but their models are not trained to perform the forecast update itself, which can compromise quality under few-step sampling. In this work, we introduce Seq-Flow, a conditional flow model whose ODE transports samples from the previous forecast distribution to the updated one. Because successive forecasts often differ only modestly, this transport starts from an informative distribution and can produce accurate updates with few flow evaluations. Recursive reuse also creates a challenge: errors in one forecast become errors in the initial states of subsequent flows. We address this with self-rollout training, in which a moving average 
    
[^15]: 基于标量伴随匹配的Q学习

    Q-Learning with Scalar Adjoint Matching

    [https://arxiv.org/abs/2610.10437](https://arxiv.org/abs/2610.10437)

    本文提出标量伴随匹配方法，利用预训练流策略的速度雅可比矩阵集中于对角线这一发现，推导出闭式标量伴随以替代昂贵的逐步向量-雅可比积计算，从而实现更高效的流策略离策略强化学习微调。

    

    流策略能够捕捉丰富多样的动作分布，使用离策略强化学习对其进行微调以超越演示数据已引起越来越多的关注。然而，针对学习到的价值函数微调流策略并非易事，因为策略需要在多个流步骤上生成动作。伴随匹配提供了一种有原则的方法来更新流模型本身，它将价值信息从最终动作反向传播到每个流步骤，但该方法需要在每一步都通过策略计算向量-雅可比积，其代价随流步数和策略规模的增长而增加。我们观察到，预训练流策略的批次平均速度雅可比矩阵集中于其对角线上。受这一发现的启发，我们推导出一个闭式形式的标量伴随，它以流时间对最终动作处的价值梯度进行缩放，从而消除了每步的向量-雅可比积计算。我们进一步发现，控制评论家网络……（原文摘要在此处截断）

    arXiv:2610.10437v1 Announce Type: new  Abstract: Flow policies capture rich and diverse action distributions, and fine-tuning them with off-policy RL to improve beyond the demonstrations has drawn growing interest. However, fine-tuning a flow policy against a learned value function is not trivial, because the policy generates its action over many flow steps. Adjoint matching offers a principled way to update the flow model itself by propagating value information from the final action back to each flow step, but it requires a vector--Jacobian product through the policy at every step, a cost that grows with the number of flow steps and the policy size. We observe that the batch-averaged velocity Jacobian of pretrained flow policies concentrates on its diagonal. Motivated by this finding, we derive a closed-form scalar adjoint that scales the value gradient at the final action by the flow time, eliminating the per-step vector--Jacobian products. We further find that controlling the critic
    
[^16]: 用于生成三维多变量瞬时城市微气候场的条件流匹配方法

    Conditional Flow Matching for Generation of 3D Multi-variable Instantaneous Urban Microclimate Fields

    [https://arxiv.org/abs/2610.10430](https://arxiv.org/abs/2610.10430)

    本文提出基于条件流匹配（CFM）的生成框架，以建筑几何和平均流场为条件，在数秒内生成合理的三维多变量瞬时城市微气候场，既克服了大涡模拟计算成本高的局限，又解决了回归模型无法表征湍流随机性的问题。

    

    快速而准确地预测城市风场和温度场对于城市微气候设计与气候适应具有重要意义。大涡模拟（LES）能够有效解析这些瞬时场，但由于计算成本高昂，其在城市微气候迭代设计中的应用受到限制。现有的回归类数据驱动模型虽能快速输出结果，但只能产生确定性的点预测，本质上无法表征湍流的随机性。本文采用了一种新颖的条件流匹配（CFM）生成框架，以建筑几何形状和平均流场作为引导条件，在数秒内即可生成合理的城市微气候三维瞬时速度场和温度场。为克服像素空间三维生成的GPU显存瓶颈，该模型通过共享噪声初始化在重叠的像素空间上并行运行，从而保持高空间分辨率。

    arXiv:2610.10430v1 Announce Type: cross  Abstract: Rapid and accurate prediction of urban wind and temperature fields is important for urban microclimate design and climate adaptation. Large-eddy simulation (LES) effectively resolves these instantaneous fields, but its application is limited in iterative design of urban microclimate applications due to high computational cost. Existing regressive data-driven models offers quick outputs, but they produce only deterministic point predictions that inherently fail to represent turbulent stochasticity. This paper adopts a novel generative framework of Conditional Flow Matching (CFM) that uses building geometry and mean flow as guidance to generate plausible three-dimensional instantaneous velocity and temperature fields for urban microclimate in seconds. To overcome the GPU memory bottleneck of pixel space 3D generation, the model operates in parallel on overlapping pixel space through a shared-noise initialization that preserves high spati
    
[^17]: 双方向预算下的导数高斯过程

    Derivative Gaussian Processes on a Two-Direction Budget

    [https://arxiv.org/abs/2610.10428](https://arxiv.org/abs/2610.10428)

    本文提出一种每个观测梯度仅需两个方向的导数高斯过程，在Vecchia近似下将 $md$ 个梯度坐标压缩为至多 $2m$ 个方向导数，使每个预测目标的计算代价降至 $\mathcal{O}(m^3)$，并给出了近似误差的理论界。

    

    梯度观测有望带来更精确的高斯过程（GP）代理模型，但引入梯度观测的代价长期以来一直阻碍着这一前景的实现。我们提出了一种导数高斯过程，其每个观测梯度的预算仅为两个方向。其中一个方向关注每个梯度对目标预测的直接贡献，另一个方向则通过与条件函数值的相关性来聚合其间接贡献。在Vecchia近似框架下，每次预测以 $d$ 维空间中 $m$ 个邻近输入为条件，该构造最多使用 $2m$ 个方向导数来表示其 $md$ 个梯度坐标，使每个预测目标的稠密分解代价为 $\mathcal{O}(m^3)$。对于一般的条件集，我们给出了相对于使用完整梯度的后验近似误差界，并刻画了误差较小或近似精确的条件。在仿真实验中，我们的方法与……（原文截断）

    arXiv:2610.10428v1 Announce Type: cross  Abstract: Gradient observations promise more accurate Gaussian process (GP) surrogates, but the cost of incorporating them has long stood in the way of realizing that promise. We propose a derivative GP with a budget of just two directions per observed gradient. One direction focuses on each gradient's direct contribution to target prediction, while the other aggregates its indirect contributions through correlations with the conditioning function values. Within a Vecchia approximation, where each prediction conditions on $m$ nearby inputs in $d$ dimensions, this construction represents their $md$ gradient coordinates using at most $2m$ directional derivatives, giving $\mathcal{O}(m^3)$ dense factorization cost per prediction target. For general conditioning sets, we bound the posterior approximation error relative to using full gradients and characterize when the error is small or the approximation is exact. In simulations, our method matches t
    
[^18]: 哪一次Rollout教会了它？BehaviorTrace与在线强化学习中训练数据归因的局限

    Which Rollout Taught It That? BehaviorTrace and the Limits of Training-Data Attribution in Online RL

    [https://arxiv.org/abs/2610.10422](https://arxiv.org/abs/2610.10422)

    该工作发布了BehaviorTrace开源评估框架，通过植入已知成因的行为实验发现，在线RL中训练数据归因方法的表现很大程度上源于梯度大小和模型流畅度等混淆因素，揭示了现有归因信号的可靠性局限。

    

    当强化学习教会一个语言模型一种新行为时，我们能否找出是哪些训练rollout教会了它？当某种归因方法声称可以做到时，我们又如何确认其答案是真实可靠的？我们在使用GRPO的在线RL微调上研究这两个问题，采用一种成因已知的植入行为（planted behavior）实验设置。我们发布了BehaviorTrace——一个开放的评估工具框架，它结合了全梯度草图技术、植入行为设置，以及对梯度大小、流畅度、性能提升空间、随机种子与生成样本差异等混淆因素的控制。在Qwen2.5-1.5B模型上的三个随机种子实验中，大量表观上的归因信号实际来自混淆因素。一个仅按梯度大小对训练步骤排序、不包含任何行为目标的对照组，达到了随机基线的4.2至4.5倍，并在三个种子中的两个上匹配甚至超越了最佳的目标归因估计器。在饱和检查点处，模型流畅度预测行为标签的能力至少不逊于我们对比的所有梯度方法。（原文摘要在此处被截断）

    arXiv:2610.10422v1 Announce Type: cross  Abstract: When reinforcement learning teaches a language model a new behavior, can we find the training rollouts that taught it? And when an attribution method says it can, how do we know the answer is real? We study both questions on online RL fine-tuning with GRPO, using a planted behavior with a known cause. We release BehaviorTrace, an open evaluation harness that combines full-gradient sketching, the planted-behavior setup, and controls for gradient magnitude, fluency, headroom, and variation across seeds and generation draws. Across three seeds on Qwen2.5-1.5B, much of the apparent attribution signal comes from confounds. A control that ranks training steps by gradient size alone, with no behavior target, reaches 4.2 to 4.5 times chance and matches or beats the best targeted estimator on two of three seeds. At saturated checkpoints, model fluency predicts the behavior label at least as well as every gradient method we compared it with. Onc
    
[^19]: SteerSpeech：基于激活引导的生成语音情感控制方法

    Steerspeech: Activation Steering For Emotion Control In Generated Speech

    [https://arxiv.org/abs/2610.10415](https://arxiv.org/abs/2610.10415)

    SteerSpeech是一种轻量级激活引导框架，通过向冻结的TTS模型隐藏激活中注入引导向量，实现推理时的精准情感控制，同时保持说话人身份与语言内容不变。

    

    预训练的文本转语音（TTS）模型能够生成富有表现力的语音，但可靠的推理时情感控制仍然具有挑战性：提示词和参考音频只能提供粗糙且不一致的控制，而专门的调节机制和模型适配则需要高昂的训练成本。我们提出了SteerSpeech，这是一个轻量级的激活引导框架，通过向隐藏激活中注入引导向量来控制情感。对于每个目标情感，我们训练一个轻量级的低秩变换，并采用多专家目标函数，在促进单调情感控制的同时保持说话人身份和语言内容，约束引导漂移，并保持TTS主干网络冻结。为了能够通过离散的语音token进行优化，我们引入了一个两阶段的生成-重放管线，使用直通估计器将专家监督反向传播到采样的token中。在推理阶段，通过优化目标情感的引导方向来实现情感控制。

    arXiv:2610.10415v1 Announce Type: cross  Abstract: Pretrained text-to-speech (TTS) models can generate expressive speech, but reliable inference-time emotion control remains challenging: prompts and reference audio offer coarse, inconsistent control, whereas specialized conditioning and model adaptation require costly training. We present SteerSpeech, a lightweight activation-steering framework that controls emotion by injecting steering vectors into hidden activations. For each target emotion we train a lightweight low-rank transform, using a multi-expert objective that encourages monotonic emotion control while preserving speaker identity and linguistic content, constraining steering drift, and keeping the TTS backbone frozen. To optimize through discrete speech tokens, we introduce a two-pass generation-and-replay pipeline using a straight-through estimator to backpropagate expert supervision through sampled tokens. At inference, a target-emotion steering direction is optimized with
    
[^20]: 通过直接最小化期望解码轮数来训练并行投机草稿模型

    Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds

    [https://arxiv.org/abs/2610.10411](https://arxiv.org/abs/2610.10411)

    本文将投机解码建模为马尔可夫奖励过程，提出直接最小化期望解码轮数（EDR）的训练目标，以优化并行投机草稿模型的全局解码效率。

    

    投机解码通过使用低成本的草稿模型提出候选词元，再由完整规模的目标模型并行验证，从而加速大语言模型的推理。并行和半自回归（semi-AR）草稿模型通过单次前向传播提出整个词块来提高起草效率，但训练这类模型带来了新的困难：给定位置的草稿分布取决于解码轮从哪里开始，而每轮从哪里开始又取决于之前各轮接受词元的数量。现有的训练目标通常依赖于忽略这种跨轮耦合的块内局部替代目标，因此无法直接优化全局解码效率。在这项工作中，我们将投机解码表示为马尔可夫奖励过程，为训练和评估此类草稿模型建立了一个理论框架。这一表述产生了期望解码轮数（EDR）目标，该目标对局部拒绝进行加权……

    arXiv:2610.10411v1 Announce Type: cross  Abstract: Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi- AR) drafters improve drafting efficiency by proposing an entire block in a single forward pass, but training them raises a new difficulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding efficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection co
    
[^21]: RobotWorld：面向多样化任务与多种具身形态的机器人使用多模态智能体基准测试

    RobotWorld: Benchmarking Multimodal Agents for Robot Use Across Diverse Tasks and Embodiments

    [https://arxiv.org/abs/2610.10409](https://arxiv.org/abs/2610.10409)

    RobotWorld是一个包含84项任务（涵盖操作、移动操作、运动、驾驶和空中控制）的机器人使用仿真基准，用于评估多模态智能体将指令转化为物理执行的能力，发现现有智能体虽能构建复杂的感知与控制工作流，但难以将这些能力可靠地组合以完成任务。

    

    通用智能体日益能够编写代码、使用工具并完成复杂的数字任务，这引发了一个问题：这些能力能在多大程度上延伸到物理世界。为了探究这一点，我们提出了RobotWorld，一个具有挑战性的机器人使用仿真测试平台：通过机器人接口将指令和观察转化为物理任务执行。其84个任务涵盖机械臂操作、移动操作、运动、驾驶和空中控制，并带有显式的交互预算和可执行的成功判据。通过结合任务结果与执行轨迹进行分析，我们识别出了哪些能力可以迁移，以及哪些差距阻碍了可靠的任务完成。此外，我们发现当前的智能体能够构建复杂的感知与控制工作流，包括图像分割、相机标定、空间估计和基于动力学的计算。然而，这些能力并不能始终如一地组合成成功的任务完成。

    arXiv:2610.10409v1 Announce Type: cross  Abstract: General-purpose agents increasingly write code, use tools, and complete complex digital tasks, raising the question of how far these capabilities carry into the physical world. To investigate this, we introduce RobotWorld, a challenging simulation testbed for robot use: turning instructions and observations into physical task execution through robot interfaces. Its 84 tasks span manipulation, mobile manipulation, locomotion, driving, and aerial control, with explicit interaction budgets and executable success checks. By analysing task outcomes alongside execution traces, we identify both the capabilities that transfer and the gaps that prevent reliable completion. Furthermore, we find that current agents can construct sophisticated perception and control workflows, including image segmentation, camera calibration, spatial estimation, and dynamics-based computation. These capabilities, however, do not consistently compose into successfu
    
[^22]: Rubix：基于指派几何的全局无对应点集对齐方法

    Rubix: Global Correspondence-Free Point Set Alignment through Assignment Geometry

    [https://arxiv.org/abs/2610.10408](https://arxiv.org/abs/2610.10408)

    提出Rubix方法，通过刻画置换多边形的几何结构实现无对应点集对齐问题的全局最优求解，证明了多边形顶点数的紧界n(n-1)并解决了Rote的旋转-指派开放问题。

    

    Procrustes-Wasserstein对齐方法能够在没有给定对应关系的情况下联合估计匹配与旋转，但交替最小化可能收敛到次优解。Rubix在平方欧氏损失下全局求解等权重的平面问题。对于两个中心化的n点集，每个匹配σ定义一个复相关值z_σ=Σ_i x̄_i y_σ(i)，所有这些复相关值的凸包构成置换多边形：支撑顶点给出固定旋转下的最优匹配，而距离最远的顶点则给出全局最优对齐。作者证明了当n≥2时该多边形顶点数的紧界为n(n-1)，从而回答了Rote提出的旋转-指派开放问题。在精确算术下，通过指派查询可以在O(n^5)次运算内恢复整个多边形。基于指派的界通过分支定界方法将该方法进一步扩展到三维旋转以及给定平移下的部分匹配。在带时间限制的MPEG-7形状配对实验中，Rubix在每一个数值上……（原文摘要在此处截断）

    arXiv:2610.10408v1 Announce Type: cross  Abstract: Procrustes-Wasserstein alignment jointly estimates a matching and rotation without supplied correspondences, but alternating minimization can stop at suboptimal solutions. Rubix solves the equally weighted planar problem globally under squared Euclidean loss. Each matching $\sigma$ of two centered $n$-point sets defines a complex correlation $z_\sigma=\sum_i\bar x_i y_{\sigma(i)}$. Their convex hull is the permutation polygon: supporting vertices give optimal matchings at fixed rotations, and the farthest vertex gives the global alignment. We prove the sharp bound of $n(n-1)$ vertices for $n\ge2$, answering Rote's rotation-assignment open problem. In exact arithmetic, assignment queries recover the polygon in $\mathcal O(n^5)$ operations. Assignment-based bounds extend the approach to three-dimensional rotations and partial matching at a supplied translation through branch-and-bound. On timed MPEG-7 shape pairs, Rubix attains every num
    
[^23]: SOTA：由期权隐含收益分布引导的股票期权交易智能体

    SOTA: Stock Options Trading Agents Guided by Option-Implied Return Distributions

    [https://arxiv.org/abs/2610.10407](https://arxiv.org/abs/2610.10407)

    SOTA是一个期权交易智能体框架，通过将庞大的期权合约空间抽象为策略层面的决策、由确定性求解器执行投资组合构建，并利用期权隐含收益分布引导大语言模型，使其能够根据市场条件的变化灵活选择和切换期权交易策略。

    

    随着期权市场的增长和人工智能的进步，用于期权交易的智能体系统正受到越来越多的关注。基于语言模型的智能体可以对新闻等上下文信息进行推理，但期权交易构成了一个极具挑战性的决策问题：单只股票可能拥有数千份合约，智能体必须同时决定交易哪些合约以及如何组合它们。现有方法通常通过将策略限制为固定的策略结构（如跨式组合）来回避这种复杂性，限制了其随市场条件变化而切换策略的能力。我们提出了SOTA（Stock Options Trading Agents），一个用于结构化期权策略选择的智能体交易框架。SOTA将庞大的期权集合抽象为策略层面的决策，同时由确定性求解器负责投资组合的具体实现。我们通过对Qwen3.8-27B进行后训练来构建SOTA，包括监督微调和强化学习。

    arXiv:2610.10407v1 Announce Type: cross  Abstract: As option markets grow and AI advances, agentic systems for option trading are gaining increasing attention. Language-model-based agents can reason over contextual information such as news, but option trading presents a particularly challenging decision problem: a single stock can have thousands of contracts, and the agent must decide both which contracts to trade and how to combine them. Existing approaches often sidestep this complexity by restricting the policy to a fixed strategy structure, such as a straddle, limiting their ability to switch strategies as market conditions change. We present SOTA (Stock Options Trading Agents), an agentic trading framework for structured option-strategy selection. SOTA abstracts the large option universe into strategy-level decisions while deterministic resolvers handle portfolio implementation. We develop SOTA by post-training Qwen3.8-27B with supervised fine-tuning followed by reinforcement lear
    
[^24]: 稳态神经CFD代理模型的跨域预训练

    Cross-Domain Pretraining for Steady-State Neural CFD Surrogates

    [https://arxiv.org/abs/2610.10398](https://arxiv.org/abs/2610.10398)

    跨域预训练能显著提升神经CFD代理模型在未见数据集上的零样本和少样本泛化能力，在相同样本量下相比从头训练和领域专家迁移可将误差降低2-3倍。

    

    用于计算流体动力学（CFD）的神经代理模型有潜力通过加速仿真来极大地促进工程创新。然而，神经代理模型的主要局限性在于缺乏对训练集之外的几何形状和应用的泛化能力，考虑到工程场景的多样性，这一问题尤为突出。目前，该问题通常通过为特定应用生成新的数据集来解决，但这需要运行代价高昂的数值求解器。在本工作中，我们通过研究在不同几何形状、边界条件和精度水平上训练的神经代理模型，朝着解决这一问题迈出了一步。我们发现，与从头训练以及从特定领域专家模型迁移相比，跨域预训练提高了在保留数据集上的零样本和少样本性能。特别是，在相同样本量下，微调经过跨域预训练的模型可以实现2-3倍更低的误差。

    arXiv:2610.10398v1 Announce Type: new  Abstract: Neural surrogates for computational fluid dynamics (CFD) have the potential to greatly enhance engineering innovation through accelerating simulation. However, the primary limitation for neural surrogates is the lack of generalization to geometries and applications beyond the training set, which is significant given the diversity of engineering scenarios. Currently, this is addressed by generating a new dataset for a specific application; however, this requires running costly numerical solvers. In this work, we take a step toward addressing this by studying neural surrogates trained across different geometries, boundary conditions, and fidelities. We find that cross-domain pretraining improves zero- and few-shot performance on held-out datasets relative to both training from scratch and transferring from domain-specific experts. In particular, finetuning a pretrained, cross-domain model can achieve 2-3x lower errors at the same sample si
    
[^25]: 使用线性注意力Transformer执行因果结构学习

    Executing Causal Structure Learning with Linear-Attention Transformers

    [https://arxiv.org/abs/2610.10395](https://arxiv.org/abs/2610.10395)

    本文显式构造了一个固定权重的线性注意力Transformer，其前向传播可精确复现无环约束下连续因果发现算法的一次迭代更新，并证明在更新间保留算法乘子是实现精确执行的关键。

    

    Transformer可以在其输入数据上执行算法。我们探讨它们是否也能对因果发现做到这一点。我们研究了一种标准的连续优化方法，该方法在强制无环性约束的同时反复更新候选因果图。我们显式构造了一个固定权重的Transformer，其前向传播能够精确复现该方法的一次迭代更新，因此重复堆叠的模块可以复现其整个优化轨迹。该Transformer在更新之间携带当前的因果图以及算法的乘子。我们证明，保留该乘子对于精确执行至关重要，因为不同的乘子取值可能导致不同的下一步更新。我们还给出了相关条件，在这些条件下，在固定阶段内达到目标精度所需的更新次数可以预先计算得出，且随着深度的增长舍入误差保持有界。实验表明，所构造的模块与参考更新在浮点精度上完全一致，而算术……

    arXiv:2610.10395v1 Announce Type: new  Abstract: Transformers can execute algorithms on data given in their input. We ask whether they can do the same for causal discovery. We study a standard continuous method that repeatedly updates a candidate causal graph while enforcing acyclicity. We explicitly construct a fixed-weight transformer whose forward pass exactly reproduces one update of this method, so repeated blocks reproduce its optimization trajectory. The transformer carries the current graph and the algorithm's multiplier between updates. We show that retaining the multiplier is essential for exact execution, since different multiplier values can lead to different next updates. We also give conditions under which, within a fixed stage, the number of updates needed to reach a target accuracy can be computed in advance and rounding errors stay bounded as depth grows. Experiments show that the constructed block agrees with a reference update to floating-point precision, while arith
    
[^26]: 面向开放式模型发现的核自动研究

    Kernel Autoresearch for Open-Ended Model Discovery

    [https://arxiv.org/abs/2610.10394](https://arxiv.org/abs/2610.10394)

    Kernaut将核设计视为开放式模型发现，让编程智能体以程序形式自由生成核，同时通过构造契约保证每个核的有效性，并结合质量-多样性档案与新颖性筛选，突破了固定核语法表达能力的限制。

    

    核编码了广泛机器学习模型的归纳偏置，然而自动化核设计面临一个根本性困境：固定的基核与算子语法能够保证有效性，却将搜索限制在这些构建模块所能表达的结构之内；反之，无约束的程序移除了这一限制，却不再保证有效性。在我们的压力测试中，22%–58%的通过随机输入数值检验的LLM生成核，在不同尺度或维度下评估时会失效。我们提出核自动研究，将核设计视为一种开放式模型发现：编程智能体以程序的形式编写核，而构造契约确保每个被接受的核都是有效的；质量-多样性档案保留性能优异且行为各异的核，新颖性筛选则引导智能体产生功能上新颖的候选。我们的实验表明，所发现的核编码了……（原文摘要至此截断）

    arXiv:2610.10394v1 Announce Type: new  Abstract: Kernels encode the inductive bias of a wide range of machine learning models, yet automated kernel design faces a fundamental dilemma. A fixed grammar of base kernels and operators guarantees validity but limits the search to structures expressible by those building blocks. Conversely, unrestricted programs remove this limitation but no longer guarantee validity. In our stress tests, 22-58% of LLM-generated kernels that pass numerical checks on random inputs fail when evaluated at different scales or dimensions. We propose Kernel Autoresearch (Kernaut), which treats kernel design as open-ended model discovery. Coding agents write kernels as programs, while construction contracts ensure that every accepted kernel is valid. A quality-diversity archive retains high-performing kernels with distinct behaviors, and novelty screening steers agents toward functionally new candidates. Our experiments demonstrate that the discovered kernels encode
    
[^27]: 基于风险控制的安全元策略设计

    Safe Meta-Policy Design with Risk Control

    [https://arxiv.org/abs/2610.10393](https://arxiv.org/abs/2610.10393)

    该论文提出一种带风险控制的离线元策略设计方法，在更新退化风险预算约束下通过动态规划规划模型更新时机，并揭示策略改进的信噪比是决定更新频率与风险分配的关键因素。

    

    随着新数据的到来，模型可以被重新训练，但部署每个新版本都存在用较差策略替换较好策略的风险。我们研究如何在未来候选模型被训练之前规划策略更新（即元策略），在改进的收益与性能退化的风险之间取得平衡。我们的离线元策略在“表现劣于被替换策略的更新次数的期望值受预算约束”的条件下，最大化期望累计价值。我们从历史学习轨迹中估计可能切换的价值与风险，将更新计划表示为有向无环图中的一条路径，并使用动态规划来选择更新计划。主阶分析表明，策略改进的信噪比是决定更新频率、等待时间和风险分配的关键因素：更清晰的改进支持更早、更频繁的更新，而更嘈杂的改进则需要更长的等待或更大的风险容忍度。

    arXiv:2610.10393v1 Announce Type: cross  Abstract: Models can be retrained as new data arrive, but deploying every new version risks replacing a good policy with a worse one. We study how to plan policy updates (i.e., meta-policy) before future candidates are trained, balancing the benefits of improvement against the risk of performance regression. Our offline meta-policy maximizes expected cumulative value subject to a budget on the expected number of updates that perform worse than the policies they replace. We estimate the value and risk of possible switches from historical learning trajectories, represent an update schedule as a path in a directed acyclic graph, and select a schedule using dynamic programming. A leading-order analysis identifies the signal-to-noise ratio of policy improvement as a key driver of update frequency, waiting times, and risk allocation: clearer improvements support earlier, more frequent updates, while noisier improvements call for longer waits or greate
    
[^28]: OrBIT：结构引导的嵌入压缩

    OrBIT: Structure-Guided Embedding Compression

    [https://arxiv.org/abs/2610.10385](https://arxiv.org/abs/2610.10385)

    OrBIT提出了一种结构引导的嵌入压缩框架，通过从轨道动力学中学习可复用的局部几何来约束共享码字，并利用全局残差分配编码预算，实现嵌入表的高效压缩。

    

    嵌入表是现代语言模型中最大的组成部分之一。大多数压缩方法会固定某种编码几何结构，如坐标块、低秩子空间或无约束码本，并在此结构内进行优化。而我们提出的问题是：编码几何本身能否被发现？我们提出了OrBIT，一个结构引导的嵌入压缩框架，它从轨道动力学中学习可复用的局部几何，并用其约束一小组共享码字。全局重建残差随后决定固定编码预算应分配到何处，而冗余的重叠图册使局部误差在粘合后能够相互补偿。我们的理论展示了紧致图册几何如何控制失真、全局残差如何指导顺序分配，以及数据几何引导的精化如何改进编解码器。最终，所得的轨道机制会被编译消除，只留下一个紧凑的解码器，其中学习到的结构起主导作用。

    arXiv:2610.10385v1 Announce Type: new  Abstract: Embedding tables are among the largest components of modern language models. Most compression methods fix a coding geometry such as coordinate blocks, low-rank subspaces, or unrestricted codebooks, and optimize within it. We instead ask whether the coding geometry can itself be discovered. We introduce \emph{OrBIT}, a structure-guided embedding compression framework that learns reusable local geometry from orbit dynamics and uses it to constrain a small set of shared codewords. The global reconstruction residual then decides where the fixed coding budget is spent, while redundant overlapping charts let local errors compensate one another after gluing. Our theory shows how tight-chart geometry controls distortion, how the global residual directs sequential allocation, and how data-geometry-guided refinement improves the codec. The resulting orbit machinery is compiled away, leaving a compact decoder in which the learned structure governs 
    
[^29]: 基于 γ-VC 维研究 Boosting 与简单弱学习器的表达能力

    Boosting and the Expressive Power of Simple Weak Learners via the $\gamma$-VC Dimension

    [https://arxiv.org/abs/2610.10383](https://arxiv.org/abs/2610.10383)

    本文证明 γ-VC 维在相差一个随 γ 缩放的常数因子意义下刻画了从弱到强学习的样本复杂度，锐化了经典 VC 维与 γ-VC 维的一般关系，并针对决策树桩和轴平行矩形给出了 γ-VC 维的改进上下界。

    

    Boosting 能够将仅比随机猜测略具优势的弱假设转化为高精度的预测器，但所得分类器的表达能力可能在很大程度上依赖于基类（base class）的结构。我们通过 Alon 等人（STOC 2021）引入的 γ-VC 维来研究这一现象。我们的第一个结果表明，该参数在相差一个随 γ 缩放的常数因子的意义下，刻画了从弱到强学习的样本复杂度。随后，我们进一步锐化了经典 VC 维与 γ-VC 维之间的一般关系。最后，我们还针对决策树桩（decision stumps）与 ℝ^d 中轴平行矩形这两个基本概念类，给出了 γ-VC 维的改进上界与下界。

    arXiv:2610.10383v1 Announce Type: new  Abstract: Boosting converts weak hypotheses with a small edge over random guessing into highly accurate predictors, but the expressive power of the resulting classifier can depend strongly on the structure of the base class. We study this phenomenon through the $\gamma$-VC dimension introduced by Alon et al. (STOC 2021). Our first result shows that this parameter characterizes the sample complexity for weak-to-strong learning up to a constant factor scaling in $\gamma$. We then sharpen the general relationship between the classic VC dimension and the $\gamma$-VC dimension. Finally, we also give improved upper and lower bounds on the $\gamma$-VC dimension for the fundamental concept classes of decision stumps and axis-parallel rectangles in $\mathbb{R}^d$.
    
[^30]: ResidualQuant：面向循环Transformer的基于2比特残差的KV缓存量化

    ResidualQuant: KV Cache Quantization for Looped Transformers with 2-Bit Residuals

    [https://arxiv.org/abs/2610.10381](https://arxiv.org/abs/2610.10381)

    提出ResidualQuant方法，利用循环Transformer中各循环KV状态高度相似的特点，以最后一轮KV状态为参考、用2比特低精度残差表示其余循环，并结合最小二乘缩放、旋转和逐循环混合精度，实现精确的INT2级KV缓存量化。

    

    循环Transformer通过在多个循环中重复应用共享的Transformer块来提高参数效率，在不增加参数量的情况下增加计算深度。然而，KV缓存的内存仍然随循环次数的增加而扩展，成为限制批处理规模和推理吞吐量的关键内存瓶颈。KV缓存量化可以缓解这一瓶颈，但现有方法在激进低精度设置下往往会出现显著的精度下降。我们观察到循环Transformer提供了一个独特的机会：不同循环之间的KV状态高度相似。基于这一观察，我们提出了ResidualQuant，该方法以最后一轮循环的KV状态作为参考，并用低精度残差来表示其余循环。我们的方法进一步结合了对残差应用的最小二乘缩放和旋转操作，以及逐循环的混合精度策略，从而实现了精确至INT2的量化。

    arXiv:2610.10381v1 Announce Type: new  Abstract: Looped Transformers improve parameter efficiency by repeatedly applying shared Transformer blocks over multiple recurrent loops, increasing computational depth without increasing the parameter count. However, KV cache memory still scales with the number of loops, becoming a key memory bottleneck that limits batch size and inference throughput. KV cache quantization can alleviate this bottleneck, but existing methods often suffer substantial accuracy degradation at aggressive low-precision regimes. We observe that looped Transformers offer a unique opportunity: KV states across loops are highly similar. Based on this observation, we propose ResidualQuant, which uses the final-loop KV states as a reference and represents the remaining loops with low-precision residuals. Our method further combines least-square scaling and rotations applied to the residuals, as well as loop-wise mixed precision, to enable accurate quantization down to INT2 
    
[^31]: 无需持续训练的持续学习

    Continual Learning without Continual Training

    [https://arxiv.org/abs/2610.10379](https://arxiv.org/abs/2610.10379)

    提出Latent Concept PFN模型，将元训练后冻结的PFN用于持续推断而非持续训练，仅通过在上下文证据集中添加样本并基于潜在概念空间进行贝叶斯后验更新来适应新领域和新类别，全程不改变任何参数，从而显著减少遗忘。

    

    持续学习要求模型在保留已有知识的同时适应新领域和新类别。许多现有方法依赖于持续优化，使用正则化、回放或参数扩展来防止新的更新覆盖先前学到的知识。相反，我们提出用持续推断取代持续训练：一个基于PFN的模型经过元训练后被冻结，仅通过扩展上下文证据集来适应新类别。我们的模型Latent Concept PFN在一个潜在概念空间上执行上下文贝叶斯推断，该空间捕获跨领域和跨类别共享的语义结构。当每个新领域或新类别到来时，样本被添加到记忆中；适应体现为对潜在概念的更新后验信念，而非梯度更新。无需改变任何参数，从而减少遗忘。同一方法可以同时处理领域增量持续学习和类别增量持续学习。

    arXiv:2610.10379v1 Announce Type: new  Abstract: Continual learning requires models to adapt to new domains and new classes while retaining prior knowledge. Many existing methods rely on continued optimization, using regularization, replay, or parameter expansion to prevent new updates from overwriting previously learned knowledge. Instead, we propose replacing continual training with continual inference: a PFN-based model that is meta-trained, and then frozen, adapting to new classes only by extending an in-context evidence set. Our model, Latent Concept PFN, performs in-context Bayesian inference over a latent concept space that captures semantic structure shared across domains and classes. As each new domain or class arrives, exemplars are added to the memory; adaptation reflects updated posterior beliefs over latent concepts rather than gradient updates. No parameters are changed, reducing forgetting. The same method handles both domain and class incremental continual learning with
    
[^32]: 输入盲化对照在多项选择评估中为层程序带来显著的神谕提升空间

    Input-Blind Controls Produce Substantial Oracle Headroom for Layer Programs in Multiple-Choice Evaluation

    [https://arxiv.org/abs/2610.10368](https://arxiv.org/abs/2610.10368)

    本研究发现在多项选择评估中，输入盲化的对照扰动所产生的神谕提升空间反而超过真实的层跳过与重复程序，说明仅凭选择增益无法解释所选层程序为何有效。

    

    自适应计算旨在通过针对每个输入定制执行方式来改进语言模型的推理。对于层程序，在实用的选择器可用之前，神谕评估利用已知答案来估计这种灵活性带来的潜在增益。然而，来自选择的增益本身并不能解释所选程序为何有效。本研究利用两个模型上的32个层跳过与重复程序以及4,413个多项选择题目来考察这一区别。该分析将真实程序相对于在无评估提示情况下所选固定动作的增益，与相同位置上输入盲化扰动的增益进行比较，并在另一个提示上重新评估选择结果。在共享选项顺序的情况下，这些对照在Qwen3-4B-Base和Llama-3.1-8B上分别产生了10.2-11.8和15.6-19.4个百分点的提升空间，在每模型的全部三次随机方向抽取中均超过真实程序的9.0和10.1。它们仅在答案改变率上与真实程序相当，且排序取决于（原文在此处截断）。

    arXiv:2610.10368v1 Announce Type: cross  Abstract: Adaptive computation aims to improve language-model inference by tailoring execution to each input. For layer programs, oracle evaluations use known answers to estimate the potential gain from this flexibility, before a practical selector is available. However, a gain from selection does not by itself explain why the chosen programs help. This study examines this distinction using 32 layer-skipping and repetition programs on two models and 4,413 multiple-choice items. The analysis compares their gains over a fixed action selected without the evaluation prompt with those of input-blind perturbations at the same sites, re-evaluating selections on another prompt. With shared option order, the controls give 10.2-11.8 and 15.6-19.4 percentage points of headroom on Qwen3-4B-Base and Llama-3.1-8B, exceeding the real programs' 9.0 and 10.1 in all three random-direction draws per model. They match answer-change rate only, and the ordering depen
    
[^33]: 时间可解释的可微决策树

    Temporally Interpretable Differentiable Decision Trees

    [https://arxiv.org/abs/2610.10367](https://arxiv.org/abs/2610.10367)

    该论文提出了“时间可解释性”这一可解释性的新维度，通过引入两种结合动作分块的新型策略梯度算法，解决了可微决策树的单时间步行为与人类多时间步规划之间的固有不匹配问题，使其更适合序列决策任务。

    

    可解释性通过提供对智能体底层决策模型的透明度，为安全自主性提供了一种解决方案。在序列决策任务中，可微决策树（DDTs）是实现这种可解释性的一种方法，它在保持策略可自动微分的同时，为人类提供离散的树状可视化。然而，当前的可微决策树实现并不适合序列决策领域，因为树的单时间步行为与人类的多时间步规划之间存在固有的不匹配。因此，我们的工作引入了时间作为可解释性的一个新维度，称之为“时间可解释性”，并通过动作分块实现的时间抽象来展示如何提升这种可解释性。我们首先通过提出两种融合动作分块的新型策略梯度算法来实现这一目标。此外，为了保持树的参数高效性，我们开发了一种信息（摘要在此处被截断）

    arXiv:2610.10367v1 Announce Type: new  Abstract: Interpretability offers a solution to safe autonomy by providing transparency into an agent's underlying decision-making model. Within sequential-decision making tasks, differentiable decision trees (DDTs) are one approach to such interpretability, maintaining automatic-differentiable policies while providing humans with a discrete tree-based visualization. Nonetheless, current implementations of DDTs are not well-suited for sequential-decision making domains, as there exists an inherent mismatch between a tree's single-timestep behavior and a human's multi-timestep planning. Our work thus introduces time as a new dimension of interpretability, coined as temporal interpretability, and demonstrates how temporal abstractions via action chunking improve it. We achieve this by first introducing two novel policy gradient algorithms that incorporate action chunking. Additionally, to maintain parameter-efficient trees, we develop an information
    
[^34]: 用于扩散模型加速的库普曼观测器：利用浅层测量修正特征预测

    Koopman Observers for Diffusion Acceleration: Correcting Feature Forecasts with Shallow Measurements

    [https://arxiv.org/abs/2610.10366](https://arxiv.org/abs/2610.10366)

    提出一种观测修正的库普曼框架，在不改变模型参数的前提下，利用即时计算的浅层特征观测来修正深层特征的库普曼预测，并配合周期性完整评估刷新观测器，从而加速冻结扩散模型的采样。

    

    特征缓存通过用基于先前计算的激活值所做的预测来替代昂贵的网络评估，从而加速扩散采样。然而，仅基于过去特征的预测无法直接纳入当前去噪状态的变化。我们研究了廉价的、即时计算的浅层特征能否作为观测来修正这些预测。我们提出了一种用于加速冻结扩散模型的观测修正库普曼框架。利用校准轨迹，我们识别出有限维、随时间变化的库普曼近似，用以联合描述浅层与深层网络特征的增量。在加速采样过程中，这些算子预测昂贵深层特征的演化，而观测到的浅层特征的新息（innovation）则对预测状态进行修正。周期性的完整网络评估会刷新观测器，且所有生成模型参数保持不变。这一公式化……

    arXiv:2610.10366v1 Announce Type: new  Abstract: Feature caching accelerates diffusion sampling by replacing expensive network evaluations with predictions from previously computed activations. However, forecasts based only on past features cannot directly incorporate changes in the current denoising state. We investigate whether inexpensive, freshly computed features can serve as observations for correcting these predictions. We introduce an observation-corrected Koopman framework for accelerating frozen diffusion models. Using calibration trajectories, we identify finite-dimensional, time-dependent Koopman approximations that jointly describe the increments of shallow and deep network features. During accelerated sampling, these operators predict the evolution of expensive deep features, while innovations in the observed shallow features correct the predicted state. Periodic full evaluations refresh the observer, and all generative-model parameters remain unchanged. This formulation 
    
[^35]: 去中心化自适应感知的逐路径信息证书

    Pathwise Information Certificates for Decentralized Adaptive Sensing

    [https://arxiv.org/abs/2610.10362](https://arxiv.org/abs/2610.10362)

    该论文提出一种基于沿实际感知轨迹累积的 Rényi–Chernoff 信息的逐路径证书，为去中心化自适应感知提供了非渐近的 MAP 错误界和随时可用的全网停止规则，并证明信息线性增长可保证误差指数衰减。

    

    arXiv:2610.10362v1 公告类型：交叉 摘要：我们研究去中心化自适应感知问题，其中多个智能体在通信图上交换信息的同时，依据不断演化的局部信念选择测量。我们提出这样一个问题：自适应策略实际选择的测量是否已经收集了足够的证据，以将真实目标与每一个合理的备选目标区分开来。我们开发了一种基于沿实际感知轨迹累积的 Rényi–Chernoff 信息的逐路径证书。它为任意依赖历史的感知策略提供了非渐近的 MAP 错误界以及一种随时可用的全网停止规则，同时将累积的统计信息与有界的网络混合瞬态分离开来。信息相对于最难分辨的竞争假设呈线性增长，意味着 MAP 错误和平方定位误差呈指数衰减。一个经典的两两 KL 下界，专门应用于自适应去中心化通信记录，表明信息不足时……（摘要原文在此处截断）

    arXiv:2610.10362v1 Announce Type: cross  Abstract: We study decentralized adaptive sensing, where multiple agents choose measurements from evolving local beliefs while exchanging information over a communication graph. We ask whether the measurements actually selected by an adaptive policy have collected enough evidence to distinguish the true target from every plausible alternative. We develop a pathwise certificate based on the R\'enyi--Chernoff information accumulated along the realized sensing trajectory. It yields nonasymptotic MAP-error bounds and an anytime, network-wide stopping rule for arbitrary history-dependent sensing policies, while separating accumulated statistical information from a bounded network-mixing transient. Linear growth of the information against the least-resolved competitor implies exponential decay of MAP and squared-localization error. A classical pairwise KL converse, specialized to the adaptive decentralized transcript, shows that insufficient informati
    
[^36]: ORDERS：个性化联邦学习中范数排名聚合的实证研究

    ORDERS: An Empirical Study of Norm-Rank Aggregation for Personalized Federated Learning

    [https://arxiv.org/abs/2610.10361](https://arxiv.org/abs/2610.10361)

    论文提出ORDERS配置，将共享主干、私有残差适配器与按更新范数降序的几何权重聚合相结合，并通过80次运行的完整评估表明其在个性化联邦学习中相比均匀权重对照和FedPer基线仅带来微小但可复现的准确率提升。

    

    个性化联邦学习将共享表示与客户端特定的预测器相结合，但服务器加权规则的贡献可能被本地训练和评估选择所掩盖。我们研究了ORDERS，这是一种配置，它结合了共享主干网络、私有的残差适配器和分类器、按更新范数降序分配的几何权重、特征对齐以及私有参数扰动。服务器计算从同一广播模型获得的更新的加权和；它不会从顺序聚合中获得额外的优化效果。一项完整定义的评估包含80次最终运行：八个配置、两个数据集，以及在每个数据集的一个固定分区上的五个训练随机种子。在每客户端两类的CIFAR-10上，ORDERS实现了80.51 ± 0.79%的本地平均客户端准确率，相比之下FedPer-R1为79.02 ± 1.42%，匹配的均匀权重对照为80.27 ± 0.73%。

    arXiv:2610.10361v1 Announce Type: new  Abstract: Personalized federated learning combines shared representations with client-specific predictors, but the contribution of a server weighting rule can be obscured by local training and evaluation choices. We study ORDERS, a configuration that combines a shared backbone, a private residual adapter and classifier, geometric weights assigned by descending update norm, feature alignment, and private-parameter perturbations. The server computes a weighted sum of updates obtained from the same broadcast model; it does not obtain an additional optimization effect from sequential addition. A fully specified evaluation comprises 80 final runs: eight configurations, two datasets, and five training seeds on one fixed partition per dataset. On two-class-per-client CIFAR-10, ORDERS achieves $80.51 \pm 0.79\%$ native mean client accuracy, compared with $79.02 \pm 1.42\%$ for FedPer-R1 and $80.27 \pm 0.73\%$ for the matched uniform-weight control. After 
    
[^37]: 面向组合优化的测量高效可微分量子架构搜索

    Measurement-Efficient Differentiable Quantum Architecture Search for Combinatorial Optimization

    [https://arxiv.org/abs/2610.10351](https://arxiv.org/abs/2610.10351)

    该论文提出了一种理论推导的测量缩减方案，可在不改变优化目标的前提下将可微分量子架构搜索（DQAS）的梯度测量成本降低约39%至41%，并通过3-SAT和MaxCut基准问题的实验验证了其有效性。

    

    可微分量子架构搜索（DQAS）是一种用于量子电路自动化设计的有前景的框架，尤其适用于变分量子优化算法。然而，其在量子硬件上的实际部署受到优化过程中所需的大量电路测量的限制，使得硬件执行成本高昂。在这项工作中，我们证明对于一大类组合优化问题和常用的旋转门参数化方法，DQAS的测量成本可以在不改变优化目标的情况下显著降低。我们从理论上推导了所提出的测量缩减方案，并在3-SAT和MaxCut基准问题上进行了实验验证。我们的方法将梯度测量成本降低了约39%至41%，同时仅引入极小的经典后处理开销，从而降低了在量子硬件上执行DQAS的实际成本。

    arXiv:2610.10351v1 Announce Type: cross  Abstract: Differentiable quantum architecture search (DQAS) is a promising framework for the automated design of quantum circuits, particularly for variational quantum optimization algorithms. However, its practical deployment on quantum hardware is limited by the large number of circuit measurements required during optimization, making hardware execution costly. In this work, we show that for a broad class of combinatorial optimization problems and commonly used rotational gate parameterizations, the measurement cost of DQAS can be significantly reduced without changing the optimization objective. We derive the proposed measurement reduction scheme theoretically and validate it experimentally on 3-SAT and MaxCut benchmark problems. Our approach reduces the requested gradient measurement cost by about 39 to 41% while introducing only negligible classical post-processing overhead, lowering the practical cost of executing DQAS on quantum hardware.
    
[^38]: AutoAdapt：自动领域发现实现低成本可扩展性

    AutoAdapt: Automatic Domain Discovery Enables Low-Cost Extensibility

    [https://arxiv.org/abs/2610.10349](https://arxiv.org/abs/2610.10349)

    AutoAdapt是一个模块化框架，通过自动发现潜在领域并并行训练独立的LoRA适配器与无参数路由，实现了无需全模型重训练的低成本新领域扩展。

    

    经过指令微调的模型被部署到领域异构且不断演变的环境中，然而添加新领域或新数据通常需要代价高昂的重新训练。我们提出了AutoAdapt，这是一个模块化框架，通过针对性的单适配器训练来纳入新领域和数据，而无需修改其他适配器。该框架自动发现潜在领域，利用这些领域并行独立地训练每个领域的低秩适应LoRA适配器，并执行无参数路由。在14个特定领域基准测试和GPT-4o成对评判中，AutoAdapt达到了与在所有领域数据上训练的LoRA适配器相当的性能，而无需进行全模型重新训练。我们还发现了不同独立发现方法之间专门化效应趋同的证据。总体而言，让每个适配器仅在其自身领域上训练，从结构上避免了领域间的干扰，从而实现了模块化的、无需预定义分类体系的领域专门化。

    arXiv:2610.10349v1 Announce Type: new  Abstract: Instruction-tuned models are deployed into environments where domains are heterogeneous and evolve, yet adding new domains or data typically requires costly retraining. We present AutoAdapt, a modular framework that incorporates new domains and data via targeted single-adapter training without modifying other adapters. The framework automatically discovers latent domains, uses them to train per-domain Low-Rank Adaptation (LoRA) adapters independently in parallel and performs parameter-free routing. Across 14 domain-specific benchmarks and GPT-4o pairwise judgements, AutoAdapt achieves parity with a LoRA adapter trained on all domains without requiring full-model retraining. We also find evidence of specialisation effect convergence across independent discovery methods. Overall, training each adapter on its own domain prevents domain interference by construction, thus enabling modular, taxonomy-free domain specialisation without aggregate
    
[^39]: 从第一性原理出发的数据集剪枝：一种无标签的线性规划方法

    Dataset Pruning from First Principles: A Label-Free Linear Programming Approach

    [https://arxiv.org/abs/2610.10347](https://arxiv.org/abs/2610.10347)

    提出了一种从第一性原理推导的数据集剪枝方法，将无偏子集选择表述为方差最小化的线性规划问题，无需标签和几何邻近假设即可选出具有代表性的训练子集。

    

    数据集剪枝旨在将大规模训练集缩减为一个具有代表性的子集，同时保持模型性能。现有的基于几何的方法通常假设嵌入空间中相邻的点具有相似的属性。我们没有强加这一假设，而是通过将无偏子集选择重新表述为方差最小化问题，从基本原理推导出几何选择准则。无偏性保证了未加权的子集均值在期望意义上能够恢复整个数据集的均值，包括在固定模型参数下的损失和梯度。具体而言，我们将一族无偏子集选择算法刻画为一个高维多面体。在此框架下，最小化期望采样方差是一个线性目标。在刚体运动下取平均的采样方差差异具有闭式的成对表达式。由于该多面体的维度很高，直接应用标准线性规划是不切实际的，我们转而使用……（摘要在此处截断）

    arXiv:2610.10347v1 Announce Type: cross  Abstract: Dataset pruning reduces a large training set to a representative subset while preserving model performance. Existing geometry-based methods typically assume that nearby points in embedding space share similar properties. Rather than imposing this assumption, we derive geometric selection criteria by reformulating unbiased subset selection as a variance minimization problem. Unbiasedness ensures that unweighted subset averages recover full-dataset averages in expectation, including losses and gradients at fixed model parameters. Specifically, we characterize a family of unbiased subset selection algorithms as a high-dimensional polytope. In this context, minimizing the expected sampling variance is a linear objective. Differences in sampling variance, averaged over rigid motions, admit closed-form pairwise expressions. Because the polytope has high dimension, directly applying standard linear programming is impractical. We instead use t
    
[^40]: 非平稳学习中的数据复用

    Data Reuse in Non-Stationary Learning

    [https://arxiv.org/abs/2610.10340](https://arxiv.org/abs/2610.10340)

    提出了暴露上限复用（ECR）类算法，通过结合在线变化检测、兼容性测试和污染控制来安全复用历史数据，使非平稳在线学习的遗憾值随不同取值数量而非变化次数增长，类似于偏差-方差权衡。

    

    我们研究非平稳环境下的在线学习问题，其目标是追踪一个在有限个重复出现的取值之间突然切换的未知参数。这种取值的重复性为审慎地复用历史观测数据以提升算法性能提供了可能。然而，底层信号不断变化的特性以及缺乏关于这些动态的信息，可能会限制“安全”复用数据的能力。在本文中，我们量化了这类问题中的一些基本权衡，并表明它们与经典的偏差-方差困境具有某种相似性。具体而言，我们提出了一类任意时刻可用的算法，称为暴露上限复用，该算法结合了在线变化检测、兼容性测试和“污染”控制。我们刻画了ECR的遗憾值随不同取值数量而非变化次数增长的情形，并推导出一个新颖的信息论下界。

    arXiv:2610.10340v1 Announce Type: cross  Abstract: We consider online learning in non-stationary environments, where the goal is to track an unknown parameter that switches abruptly between a finite set of recurring values. Recurrence opens the possibility of judiciously reusing past observations to improve algorithm performance. However, the changing nature of the underlying signal and lack of information on these dynamics may limit the ability to "safely" reuse data. In this paper we quantify some of the fundamental tradeoffs in this class of problems, and show that they bear a certain resemblance to the classical bias-variance dilemma. Specifically, we propose a class of anytime algorithms, dubbed Exposure-Capped Reuse (ECR), that combine online change detection, compatibility testing, and "contamination" control. We characterize the regime in which ECR's regret scales with the number of distinct values rather than the number of changes, and derive a novel information-theoretic lowe
    
[^41]: 面向多链马尔可夫决策过程的平均奖励强化学习：一种分层分解方法

    Average-Reward Reinforcement Learning for Multichain MDPs: A Hierarchical Decomposition Approach

    [https://arxiv.org/abs/2610.10326](https://arxiv.org/abs/2610.10326)

    该论文提出一种基于异步值迭代与Bather分层分解的强化学习算法，无需模型知识即可求解平均奖励多链MDP，并在有限时间内收敛到最优增益与增益最优策略。

    

    我们研究平均奖励多链马尔可夫决策过程（MDP）中的最优策略学习问题，其中最优增益可能依赖于初始状态，且不同策略的递归结构各不相同，这给强化学习（RL）方法带来了挑战。我们提出一种基于异步值迭代的强化学习算法，该算法除了MDP的转移图之外不需要任何模型知识，并利用Bather分解将状态空间分层地划分为互通子系统与瞬态状态。这种分解使得全局决策问题能够被重新表述为若干结构化的子问题，我们的算法正是利用了这一结构。我们证明了该算法收敛于最优增益，并能在有限时间内产生增益最优的策略。在该基础算法之上，我们进一步开发了两个算法：一个近似求解多链平均最优性方程以获得接近增益最优的策略，另一个……（摘要内容在此处截断）

    arXiv:2610.10326v1 Announce Type: new  Abstract: We study learning optimal policies in average-reward multichain Markov decision processes (MDPs), where the optimal gain may depend on the initial state and recurrence structures vary across policies, creating challenges for reinforcement learning (RL) methods. We propose an asynchronous value-iteration-based RL algorithm that requires no model knowledge beyond the MDP's transition graph and leverages Bather's decomposition to hierarchically partition the state space into communicating subsystems and transient states. This decomposition induces a recasting of the global decision problem into structured subproblems, which our algorithm exploits. We show that the algorithm converges to the optimal gain and produces gain-optimal policies after finite time. Building on this base algorithm, we develop two further algorithms: one approximately solves the multichain average optimality equations to obtain near gain-optimal policies, and another 
    
[^42]: HAN-Mamba：用于多尺度金融波动率预测的层次化选择性状态空间网络

    HAN-Mamba: Hierarchical Selective State Space Networks for Multi-Scale Financial Volatility Forecasting

    [https://arxiv.org/abs/2610.10323](https://arxiv.org/abs/2610.10323)

    本文提出 HAN-Mamba，用选择性状态空间（Mamba）编码器替代层次化架构中的 Transformer 编码器，仅在三标记融合器中保留注意力机制，以高效整合多尺度市场信息并提升金融已实现波动率预测性能。

    

    短期已实现波动率预测需要整合以互不兼容的时间分辨率演变的市场信息，范围涵盖从秒级订单簿动态到周度市场状态漂移。我们的会议论文提出了 HAN-T，这是一种层次化架构，其中面向特定尺度的 Transformer 编码器分别处理短、中、长期数据流，并由一个可学习的注意力融合器权衡各流的贡献。本文用选择性状态空间（Mamba）编码器替代了具有二次方复杂度的注意力编码器，同时仅在融合器中保留注意力机制——在该融合器中，输入是三个标记的集合而非长序列。由此产生的混合模型 HAN-Mamba 通过循环状态对每条数据流进行汇总，其依赖输入的门控机制恰好契合波动率的两个结构特性：持续但衰减的记忆性以及突发的状态转换。在 Optiver 已实现波动率预测基准上，采用时间感知的五折交叉验证，HAN

    arXiv:2610.10323v1 Announce Type: new  Abstract: Short-horizon realized volatility forecasting requires the integration of market information that evolves at incompatible temporal resolutions, from second-level order book dynamics to weekly regime drift. Our conference work introduced HAN-T, a hierarchical architecture in which scale-specific Transformer encoders process short, mid, and long-horizon streams and a learned attention fuser weighs their contributions. This article replaces the quadratic attention encoders with selective state space (Mamba) encoders while retaining attention only in the fuser, where the input is a three-token set rather than a long sequence. The resulting hybrid, HAN-Mamba, summarizes each stream through a recurrent state whose input-dependent gating matches two structural properties of volatility: persistent but decaying memory and abrupt regime shifts. On the Optiver Realized Volatility Prediction benchmark under time-aware five-fold cross-validation, HAN
    
[^43]: 使用表格基础模型与系统一模型估计未编码车祸因素：Kumo Tabular 与 Jev

    Estimating Uncoded Crash Factors with Tabular Foundation and System One Models: Kumo Tabular and Jev

    [https://arxiv.org/abs/2610.10321](https://arxiv.org/abs/2610.10321)

    本研究结合表格基础模型、系统一叙述模型与人工校准，利用警察叙述文字估计编码字段遗漏的车祸因素，发现叙述记录的受伤车祸数量远超编码统计（如手机使用因素为15,074起对7,340起）。

    

    道路安全项目统计警察车祸记录中的编码字段，而警察的叙述文字——其中常记录了字段遗漏的因素——却很少被阅读。因此，安全部门无法得知其统计遗漏了多少，也不知道该在何处进行审查。本研究开发并评估了一个系统，将2017至2025年德克萨斯州5,601,890起车祸的两种视角结合起来，形成具有明确效度声明的总体估计。一个上下文表格基础模型 Kumo Tabular 读取每起车祸的编码记录；一个经过校准的系统一模型 Jev 读取两个概率样本的叙述文字；人类判断则对其概率进行重新校准。一个多波次“预测后去偏”估计器将这三个层级结合起来，而按记录概率抽取的第二个人工层级则通过设计对估计结果进行检验。对于打滑、突发疾病、疲劳、动物和使用手机等因素，叙述文字记录的受伤车祸数量多于编码字段——以手机使用为例，叙述记录为15,074起，而编码字段仅为7,340起。

    arXiv:2610.10321v1 Announce Type: cross  Abstract: Road safety programs count the coded fields of police crash records, while the officer's narrative, which often records factors the fields omit, is rarely read. A safety office thus cannot tell how much its counts miss or where to review. This study develops and evaluates a system that joins both views of the 5,601,890 Texas crashes from 2017 to 2025 into population estimates with stated validity. An in-context tabular foundation model, Kumo Tabular, reads the coded record of every crash, a calibrated System One model, Jev, reads the narratives of two probability samples, and human judgments recalibrate its probabilities. A multiwave predict-then-debias estimator joins the three tiers, and a second human tier drawn with recorded probabilities checks the estimates by design. For hydroplaning, medical episodes, fatigue, animals, and phone use, the narrative documents more injury crashes than the coded field, 15,074 against 7,340 for phon
    
[^44]: 深度思考：面向表格基础模型的回溯推理

    Thinking in Depth: Retrospective Inference for Tabular Foundation Models

    [https://arxiv.org/abs/2610.10317](https://arxiv.org/abs/2610.10317)

    提出基于回溯推理的表格基础模型Retro，让网络后层能够显式重访并重组前层产生的中间表示，以解决现有TFM中预测精细化集中于深层、分布不均匀的问题。

    

    表格基础模型（TFMs）在多种多样的表格任务上进行预训练，并在推理时利用新表格的带标签样本作为上下文进行预测。近期大多数TFM通过堆叠的Transformer层执行这种上下文内预测，反复变换样本的表示与比较方式。通过追踪若干强大TFM中的单个查询，我们发现预测的精细化过程在网络深度上分布高度不均匀，且往往集中在较后的层中。这种不均匀的精细化促使我们重新思考如何在网络中构建和复用中间表示。我们提出了Retro，一个基于回溯推理的表格基础模型，其后段阶段能够显式地重访并重组网络中较早阶段产生的中间信息。Retro围绕两个互补的操作来组织这一过程：选择重访哪些中间信息，以及如何使用由此产生的组合（摘要在此处被截断）

    arXiv:2610.10317v1 Announce Type: new  Abstract: Tabular foundation models (TFMs) are pretrained across diverse tabular tasks and make predictions on a new table at inference time using its labeled examples as context. Most recent TFMs perform such in-context prediction with stacked Transformer layers, repeatedly transforming how examples are represented and compared. By tracing individual queries through several strong TFMs, we find that predictive refinement is highly uneven across depth and is often concentrated in later layers. This uneven refinement motivates us to reconsider how intermediate representations are constructed and reused throughout the network. We introduce Retro, a tabular foundation model based on retrospective inference, where later stages can explicitly revisit and recombine intermediate information produced earlier in the network. Retro organizes this process around two complementary operations: which intermediate information to revisit, and how the resulting co
    
[^45]: PoreML：一个用于学习多孔介质中多相流的数据驱动框架

    PoreML: A Data-Driven Framework for Learning Multiphase Flow in Porous Media

    [https://arxiv.org/abs/2610.10314](https://arxiv.org/abs/2610.10314)

    PoreML是一个基于孔隙尺度物理的开源数据驱动框架，统一了多孔介质多相流的数据生成（GPU原生格子Boltzmann求解器）、大规模数据集（3.3 TB、560次模拟、15.8万余个时间步）与模型训练评估流程，填补了该领域机器学习研究中数据稀缺与缺乏统一工作流的关键空白。

    

    多孔微结构中的多相流对二氧化碳封存、燃料电池运行和倒装芯片封装等应用至关重要。由于润湿性和复杂的孔隙几何结构主导着流体界面的非线性演化，预测这些流动仍然极具挑战性。机器学习在推动该领域发展方面具有巨大潜力，但进展受限于时间分辨三维数据集的稀缺以及缺乏统一的模型训练与评估工作流程。为填补这一关键空白，我们提出了PoreML，一个基于孔隙尺度物理的、统一了数据生成、模型训练与评估的开源框架。该框架包含三个核心组件：(a) 一个经过解析解和已发表实验验证的现代GPU原生格子Boltzmann求解器，可实现可重复的数据生成；(b) 一个3.3 TB的数据集，涵盖四个应用驱动的场景，包含560次模拟运行和158,546个存储的时间步。这些……（原文摘要在此处不完整）

    arXiv:2610.10314v1 Announce Type: new  Abstract: Multiphase flow in porous microstructures is central to CO$_2$ storage, fuel-cell operation, and flip-chip packaging. Predicting these flows remains challenging because wettability and complex pore geometry govern the nonlinear evolution of fluid interfaces. Machine learning holds substantial promise for advancing the field, but progress is constrained by scarce time-resolved 3D datasets and a lack of a unified workflow for training and evaluating models. To fill this critical gap, we introduce PoreML, an open-source framework unifying data generation, model training, and evaluation grounded in pore-scale physics. The framework comprises three core components. (a) A modern GPU-native lattice Boltzmann solver, validated against analytical solutions and published experiments, enables reproducible data generation. (b) A 3.3 TB dataset contains 560 simulation runs and 158,546 stored time steps across four application-driven scenarios. These 
    
[^46]: 容错基础模型

    Fault-tolerant foundation models

    [https://arxiv.org/abs/2610.10311](https://arxiv.org/abs/2610.10311)

    该论文发现大型语言模型可被训练以容忍硬件故障，且模型越大错误韧性越强，并由此推测适当训练的LLM可能具有形式上的容错性，从而为在低能耗故障硬件上运行AI推理、实现大幅节能开辟道路。

    

    新兴的计算机硬件常常以牺牲可靠性来换取能源效率；在本研究中，我们表明大型语言模型（LLM）可以被训练以容忍这种不可靠性，而且随着模型规模的增大，其错误韧性实际上不降反升。基于在模拟故障数字硬件上进行的40,000 GPU小时训练运行推断出的修正神经缩放定律量化了这一趋势，并表明模型学会在“好的”纠错码内进行计算，其相对开销无论模型多大都保持有限。这一发现使我们推测，经过适当训练的LLM可能是形式上容错的；如果属实，在低能耗、有故障的硬件上运行AI推理可能是相比现状实现大幅节能的一条路径。

    arXiv:2610.10311v1 Announce Type: new  Abstract: Emerging computer hardware often trades reliability for energy efficiency; here we show that large-language models (LLMs) can be trained to tolerate this unreliability, and that rather than degrading, their error resilience actually increases as they grow. Modified neural scaling laws inferred from 40,000 GPU-hours of training runs on simulated faulty digital hardware quantify this trend and suggest that models learn to compute within "good" error-correcting codes, whose relative overhead remains finite no matter how large the model gets. This finding leads us to conjecture that appropriately trained LLMs may be formally fault-tolerant; if true, running AI inference on low energy, faulty hardware may be a path to substantial energy savings over the status quo.
    
[^47]: RSIGym：一个用于递归自我改进的灵活环境

    RSIGym: A Flexible Environment for Recursive Self-Improvement

    [https://arxiv.org/abs/2610.10310](https://arxiv.org/abs/2610.10310)

    本文提出了RSIGym，一个基于“一切皆服务”理念的智能体原生研究环境，通过可重用服务支持数据、执行框架和联合优化三条改进赛道，实现灵活的递归自我改进研究，并定义了RSI-Index作为跨五个基准测试的统一评估指标。

    

    递归自我改进（RSI）需要将已被接受的更改延续到后续的改进循环中，而研究智能体提出的更改也需要大量的研究基础设施。现有的设置往往让智能体重新构建常规的基础设施，或者将探索限制在单个组件上。我们提出了RSIGym，一个基于“一切皆服务”理念的专业研究环境。RSIGym通过可重用的服务提供训练、推理、轨迹生成、评估和沙箱执行，并通过共享的预算和权限控制来支持数据、执行框架以及联合改进三条赛道。这种设计使智能体能够在同一环境中研究单个干预措施，并同时联合优化数据、训练设置和执行框架。我们定义了RSI-Index指标，即在涵盖软件工程、终端交互、数学、科学（摘要在此处被截断）等五个基准测试中，对剩余性能差距的平均弥合比例。

    arXiv:2610.10310v1 Announce Type: new  Abstract: Recursive self-improvement requires carrying accepted changes into later improvement cycles, while studying agent-proposed changes also requires substantial research infrastructure. Existing settings often leave agents to rebuild routine infrastructure or restrict exploration to individual components. We introduce RSIGym, an agent-native research environment based on Everything as a Service (EaaS). RSIGym exposes training, inference, rollout, evaluation, and sandbox execution through reusable services, with shared budget and permission controls supporting Data, Harness, and Joint improvement tracks. This design enables agents to investigate individual interventions and jointly optimize data, training settings, and execution harnesses within the same environment. We define RSI-Index as the mean fraction of the remaining performance gap closed across five benchmarks covering software engineering, terminal interaction, mathematics, scientif
    
[^48]: Transformers如何学习表示对称性？

    How Do Transformers Learn to Represent Symmetries?

    [https://arxiv.org/abs/2610.10305](https://arxiv.org/abs/2610.10305)

    本文研究了原始Transformer通过有限数据增强学习点云对称性的能力，发现不同对称群的可学习性存在递增排序（不保角→保角→基本保角子群），并通过结构分析揭示了模型产生不变性的可解释机制。

    

    在几何机器学习中，使用有限数据增强来训练基于Transformer的架构已成为一种日益流行的方法。尽管在实证上取得了成功，但Transformer架构、对不同对称性的不变性以及数据增强预算之间的相互作用仍未得到充分探索。在本文中，我们研究了原始Transformer通过有限数据增强学习点云数据集中各种对称性的能力。我们识别出以下对称群的可学习性呈递增顺序：(i) 不保角对称性，(ii) 保角对称性，以及 (iii) 基本保角子群，例如平移、旋转和缩放。对于基本保角群，我们进一步研究了Transformer的外推行为，并对训练后的模型进行了结构分析，从而识别出诱导不变性的可解释机制。最后，我们……（摘要原文在此处截断）

    arXiv:2610.10305v1 Announce Type: new  Abstract: Training Transformer-based architectures with finite data augmentation has become an increasingly popular approach in geometric machine learning. Despite its empirical success, the interplay between the Transformer architecture, invariance to different symmetries, and augmentation budgets remains underexplored. In this paper, we study the ability of a vanilla Transformer to learn various symmetries through finite data augmentation for point cloud datasets. We identify an ordering of increasing learnability across the following symmetry groups: (i) non-angle-preserving symmetries, (ii) angle-preserving symmetries, and (iii) base angle-preserving subgroups, such as translation, rotation, and scale. For the base angle-preserving groups, we further investigate the Transformer's extrapolation behavior and conduct a structural analysis of the trained models, allowing us to identify interpretable mechanisms that induce invariance. Finally, we e
    
[^49]: SemanticFold：潜在序列压缩分离语言建模、可解码性与推理能力

    SemanticFold: Latent Sequence Compression SeparatesLanguage Modeling, Decodability, and Reasoning

    [https://arxiv.org/abs/2610.10304](https://arxiv.org/abs/2610.10304)

    提出SemanticFold潜在序列压缩方案，通过在学习的边界折叠前缀隐藏状态来压缩提示前缀，发现压缩对语言建模、可解码性和推理能力的影响是非单调的且各自具有不同的压缩阈值，证明这些能力可以相互分离。

    

    我们研究提示词前缀的潜在序列压缩是否能保留大型语言模型在推理过程中所依赖的能力。我们提出了SemanticFold，一种在学习的边界处折叠前缀隐藏状态的压缩方案，并在五个模型规模上进行评估：Qwen3-1.7B、Qwen3-8B、SmolLM2-1.7B、Pythia-1.4B和Pythia-6.9B。我们采用固定目标协议：冻结的前缀以原生方式执行或被压缩，两种方式均通过教师强制使用完全相同的续写词元。这一设计排除了目标选择对似然变化的解释。我们考察了五个终点类别：固定目标负对数似然、有限标签推理准确率、线性探针可访问性、开放式生成以及系统级内存和延迟。我们发现压缩使这些终点发生非单调变化，且它们不共享统一的压缩阈值。在Qwen3-1.7B上，当压缩比R=1.7时，压缩最小……

    arXiv:2610.10304v1 Announce Type: cross  Abstract: We study whether latent sequence compression of prompt prefixes preserves the capabilities that large language models rely on during inference. We introduce SemanticFold, a compression scheme that folds prefix hidden states at learned boundaries, and evaluate it across five model scales: Qwen3-1.7B, Qwen3-8B, SmolLM2-1.7B, Pythia-1.4B, and Pythia-6.9B. We use a fixed-target protocol: a frozen prefix is executed natively or compressed, and both arms teacher-force identical continuation tokens. This design rules out target-selection explanations for likelihood changes. We examine five endpoint families: fixed-target negative log-likelihood, finite-label reasoning accuracy, linear probe accessibility, open-ended generation, and systems-level memory and latency. We find that compression moves these endpoints non-monotonically and that they do not share a single compression threshold. On Qwen3-1.7B at compression ratio R=1.7, compressed-min
    
[^50]: 持续图多智能体强化学习

    Continual Graph Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.10302](https://arxiv.org/abs/2610.10302)

    该论文提出CGMARL框架，通过将任务序列映射为一系列建模任务特定结构的属性图，来促进知识迁移并缓解遗忘，从而解决持续多智能体强化学习中的结构信息利用问题。

    

    在持续多智能体强化学习（CMARL）中，智能体跨任务序列学习协作策略，旨在有效适应新任务的同时保留解决先前任务的能力。在许多应用中，任务在底层结构上存在差异，这可以代表不同的运行条件或目标配置（例如，电网中不同的网络拓扑或编队控制中的队形排列）。现有的CMARL方法缺乏专门的机制在学习新任务时利用这些结构信息，无法促进迁移并缓解遗忘。为了填补这一空白，我们提出了持续图多智能体强化学习（CGMARL），这是一个用于CMARL问题的新框架，其中任务序列被映射为一系列属性图，每个图建模一个任务特定的结构。在CGMARL中，每个图决定环境动力学（下一状态……

    arXiv:2610.10302v1 Announce Type: new  Abstract: In Continual Multi-Agent Reinforcement Learning (CMARL), agents learn cooperative policies across sequences of tasks, aiming to adapt effectively to new tasks while preserving the ability to solve previously encountered ones. In many applications, tasks differ in their underlying structure, which can represent, for example, distinct operational conditions or target configurations (e.g., different network topologies in power grids or arrangements in formation control). Existing CMARL methods lack dedicated mechanisms to leverage this structural information when learning new tasks, failing to promote transfer and mitigate forgetting. To fill this gap, we propose Continual Graph Multi-Agent Reinforcement Learning (CGMARL), a novel framework for CMARL problems in which task sequences are mapped into a series of attributed graphs, each modeling a task-specific structure. In CGMARL, each graph determines the environment dynamics (next states a
    
[^51]: 通过模型无关的概念词典重新审视可解释AI

    Revisiting Explainable AI through Model-Independent Concept Dictionaries

    [https://arxiv.org/abs/2610.10301](https://arxiv.org/abs/2610.10301)

    提出DictXAI方法，通过在输入域中用包含可解释含义的词典定义概念，将输入的稀疏编码与模型预测归因到具体词典元素，从而实现跨模型、架构无关且可操作的可解释AI解释。

    

    现代AI应用依赖于日益复杂的模型。可解释AI（XAI）作为一套旨在提高模型透明度的技术应运而生。然而，现有的XAI方法通常假设输入特征本身是可解释的，或者依赖于难以刻画且高度依赖特定架构的中间内部抽象，这阻碍了这些方法在不同模型间的一致使用。为了解决这些局限性，我们提出了DictXAI，这是一种通过词典直接在输入域中定义概念的方法——词典是一个庞大的、可能过完备的预定义元素集合，每个元素都带有可解释的含义。在技术上，DictXAI首先计算输入的稀疏编码，然后将模型的预测归因于相关的词典元素。我们展示了DictXAI解释的可操作性，表明它们可以将AI故障（例如“Clever Hans”效应）直接归因于可识别的……

    arXiv:2610.10301v1 Announce Type: new  Abstract: Modern applications of AI rely on increasingly complex models. Explainable AI (XAI) has emerged as a set of techniques aimed at improving model transparency. However, existing XAI methods typically assume input features to be inherently interpretable, or they rely on intermediate internal abstractions that are difficult to characterize and highly architecture-specific, hindering consistent use across models. To address these limitations, we propose DictXAI, a method that defines concepts directly in the input domain via a dictionary---a large, potentially overcomplete set of predefined elements, each carrying an interpretable meaning. Technically, DictXAI first computes a sparse code of the input and then attributes the model's prediction to the associated dictionary elements. We demonstrate the actionable nature of DictXAI explanations, showing that they can attribute AI malfunctions (e.g., Clever Hans effects) directly to identifiable 
    
[^52]: 共享高斯化：高斯正则化器能为对比学习证明什么，又遗漏了什么

    Shared Gaussianization: What Gaussian Regularizers Certify About Contrastive Learning, and What They Miss

    [https://arxiv.org/abs/2610.10299](https://arxiv.org/abs/2610.10299)

    本文提出共享高斯化（SG）检验，证明该高斯正则化器能以紧的、与维度无关的平方根速率上界总体 InfoNCE 的超出量，并通过单次检验同时检测视图的失配与非均匀性。

    

    分布匹配正则化器（如 LeJEPA 中的 SIGReg）能为对比学习证明什么？我们研究共享高斯化（SG），这是一种对两个归一化视图的平均值进行的特征函数高斯性检验，并由独立的 $\chi_d$ 半径进行缩放。由于相互不一致的视图会缩短平均值，一次检验即可同时检测视图的失配与非均匀性。SG 恰好在总体 InfoNCE 的对齐且均匀的最小值点处消失，并且在边际相等的条件下，它将 InfoNCE 的超出量（excess）上界约束为 $4\cdot 3^{3/4}\beta$ 乘以 SG 损失的平方根，再加上一个与损失呈线性关系的项。该平方根速率与这一维度无关的常数都是紧的，且任何作用于视图对的平方平均嵌入距离都无法达到更快的速率。在具有显式对齐项的情况下，旋转不变的均匀性检验能给出线性界的充分必要条件是其谱支配 InfoNCE 核 $e^{\beta u^\top v}$ 的谱；SG 自身的检验满足该条件，而高斯核……（摘要原文在此处截断）

    arXiv:2610.10299v1 Announce Type: new  Abstract: What can a distribution-matching regularizer such as SIGReg in LeJEPA certify about contrastive learning? We study shared Gaussianization (SG), a characteristic-function Gaussianity test on the average of two normalized views, scaled by an independent $\chi_d$ radius. Because disagreeing views shorten the average, one test detects both misalignment and non-uniformity. SG vanishes exactly at the aligned, uniform minimizers of population InfoNCE, and under equal marginals it bounds the InfoNCE excess by $4\cdot 3^{3/4}\beta$ times the square root of the SG loss, plus a term linear in the loss. The square-root rate and this dimension-free constant are sharp, and no squared mean-embedding distance on view pairs achieves a faster rate. With an explicit alignment term, a rotation-invariant uniformity test gives a linear bound if and only if its spectrum dominates that of InfoNCE's kernel $e^{\beta u^\top v}$; SG's own test does, Gaussian kerne
    
[^53]: 物理对齐的电子基态学习提升泛化能力

    Physics-Aligned Electronic Ground-State Learning Improves Generalization

    [https://arxiv.org/abs/2610.10298](https://arxiv.org/abs/2610.10298)

    该论文通过物理约束将电子基态描述子模型的学习目标与KS-DFT控制方程对齐，其提出的ON-Loss和GROOT方法在尺寸外推任务上相比之前最先进的密度基态模型将能量和力的平均绝对误差分别降低了79.1%和83.4%。

    

    机器学习原子间势（MLIPs）在分布内任务中表现出色，能够加速药物和材料研发，但在分布外泛化方面却表现不佳。我们提出通过设计可观测量无关的电子基态描述子模型（GSMs）来推动成本-精度帕累托前沿，其计算成本介于机器学习原子间势与Kohn-Sham密度泛函理论（KS-DFT）之间。我们通过施加物理约束并消除对非物理或无关自由度的优化压力，将GSMs的学习目标和架构与KS-DFT的控制方程对齐。在我们从QM9到QM40的尺寸外推实验中，我们的综合贡献——正交归一化损失（OrthoNormal-Loss, ON-Loss）和Grassmann受限占据轨道训练——相比之前最先进的密度GSMs，实现了79.1%的能量误差和83.4%的力误差（平均绝对误差，MAE）降低。对于哈密顿量GSMs，ON-Loss和Residu……（原文摘要在此处截断）

    arXiv:2610.10298v1 Announce Type: new  Abstract: Machine-learned interatomic potentials (MLIPs) excel at in-distribution tasks, accelerating drug and material development, yet they struggle to generalize out-of-distribution. We propose to push the cost-accuracy Pareto frontier by designing observable-agnostic electronic ground-state descriptor models (GSMs) with computational costs situated between MLIPs and Kohn-Sham density functional theory (KS-DFT). We align the learning objectives and architectures of GSMs with the governing equations of KS-DFT by enforcing physical constraints and removing optimization pressure on unphysical or irrelevant degrees of freedom. In our size-extrapolation experiments from QM9 to QM40, our combined contributions OrthoNormal-Loss (ON-Loss) and Grassmann Restricted Occupied-Orbital Training (GROOT) reach a 79.1% energy and 83.4% force mean absolute error (MAE) reduction over previous state-of-the-art density GSMs. For Hamiltonian GSMs, ON-Loss and Residu
    
[^54]: 基于分层强化学习的能量高效步态自适应：面向四足机器人多样地形运动

    Energy-Efficient Gait Adaptation via Hierarchical Reinforcement Learning for Quadrupedal Locomotion Across Diverse Terrains

    [https://arxiv.org/abs/2610.10297](https://arxiv.org/abs/2610.10297)

    提出一种分层强化学习框架，将高频关节级运动执行与低频能量最优步态自适应相解耦，实现四足机器人在多样地形与速度范围下能量高效、鲁棒的运动，并可通过零样本方式完成仿真到现实的迁移。

    

    虽然能量效率是腿式机器人运动控制的关键目标，但在不同速度范围和地形条件下保持鲁棒性能的同时实现低能耗仍然是一个关键挑战。对于端到端强化学习策略尤其如此，其中步态生成、运动执行和能量优化紧密耦合，导致策略对奖励设计高度敏感。在本工作中，我们提出了一种分层强化学习（HRL）框架，将负责稳定且鲁棒的关节级运动执行的高频策略与显式最小化运输成本（CoT）的低频步态自适应相分离。基于Isaac的三阶段训练流程实现了零样本的仿真到现实迁移，同时提高了跟踪精度、鲁棒性和能量效率。学习到的分层结构表现出随速度自动调整步态的能力，在低速时从pacing步态进行转换。

    arXiv:2610.10297v1 Announce Type: cross  Abstract: While energy efficiency is a critical objective for legged-robot locomotion control, achieving low energy consumption while maintaining robust performance across different velocity ranges and terrain conditions remains a key challenge. This is particularly true for end-to-end RL policies, where gait generation, motion execution, and energy optimization are tightly coupled, leading to high sensitivity to reward design. In this work, we propose a hierarchical reinforcement learning (HRL) framework that separates a high-frequency policy for stable and robust joint-level motion execution from low-frequency gait adaptation that explicitly minimizes the cost of transport (CoT). The three-stage Isaac-based training procedure enables zero-shot sim-to-real transfer with improved tracking accuracy, robustness, and energy efficiency. The learned hierarchy exhibits automatic speed-dependent gait adaptation, transitioning from pacing at low speeds 
    
[^55]: 对行动时间受限的智能体的AI安全考量

    AI Safety Considerations for Agents With Limited Time to Act

    [https://arxiv.org/abs/2610.10285](https://arxiv.org/abs/2610.10285)

    本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。

    

    在关于AI对齐日益公开的讨论背景下，近期的研究试图提出能够安全运行的具体AI架构。然而，那些看似证明了对齐的论证大多忽略了智能体需要在其中行动的环境。我们讨论了在只能部分观察且需要在有限时间内采取行动的环境中，与智能体无关的安全保证的理论界限。我们引入了两个现实场景：一个具有无限状态空间，另一个具有信号混合。在这些场景中，我们证明了即使是完美的智能体也无法保证安全的行为。我们认为，对于任何AI安全或对齐的证明，都需要将环境和相关的安全行动与智能体一起具体地加以考虑。

    arXiv:2610.10285v1 Announce Type: cross  Abstract: In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent.
    
[^56]: 面向灵巧手抓取稳定性的时序视觉-触觉学习

    Temporal Visuo-Tactile Learning for Dexterous Grasp Stability

    [https://arxiv.org/abs/2610.10283](https://arxiv.org/abs/2610.10283)

    本文构建了包含 200 个物体、10,000 次抓取试验的视-触-本体感觉多模态数据集，证明了高分辨率动态触觉感知能显著提升灵巧手抓取稳定性的预测能力。

    

    人类能够利用指尖触觉反馈以几乎完美的成功率抓取日常物品，然而大量机器人抓取文献主要关注基于视觉的抓取选择以及平行夹爪。在这项工作中，我们系统地研究了高分辨率、动态触觉感知如何为灵巧机器手的抓取稳定性预测和模型引导抓取做出贡献。为此，我们使用配备四个 Digit 360 触觉传感器的多指机器手，收集了涵盖 200 个物体、共 10,000 次抓取试验的数据集，并在每次抓取过程中记录外部视觉、本体感觉和触觉数据流。基于该数据集，我们训练了端到端的时序多模态模型，从抓取前的观察预测提起后的稳定性，并比较了不同感知模态和编码骨干网络。实验结果与受控输入消融实验表明，引入触觉——尤其是高分辨率、动态的触觉信息——（原文在此截断）

    arXiv:2610.10283v1 Announce Type: cross  Abstract: Humans can grasp everyday objects with almost perfect success rates using fingertip tactile feedback, yet much of the robotic grasping literature emphasizes vision-based grasp selection with parallel grippers. In this work, we systematically investigate how high-resolution, dynamic tactile sensing contributes to grasp stability prediction and model-guided grasping in dexterous robotic hands. To this end, we collected a dataset of 10,000 grasp trials across 200 objects using a multi-fingered robotic hand equipped with four Digit 360 tactile sensors, recording external vision, proprioception, and tactile streams throughout each grasp. With this dataset, we trained end-to-end temporal multimodal models to predict post-lift stability from pre-lift grasp observations and compared sensing modalities and encoding backbones. Experimental results and controlled input ablations show that incorporating touch, and particularly high-resolution, dyn
    
[^57]: 基于Wasserstein-Fisher-Rao JKO格式的重加权归一化流神经采样

    Neural Sampling with Reweighted Normalizing Flows via the Wasserstein--Fisher--Rao JKO Scheme

    [https://arxiv.org/abs/2610.10278](https://arxiv.org/abs/2610.10278)

    提出了一种基于WFR JKO格式的神经采样算法，首次证明其在任意固定步长下、无需对数凹性等结构性假设即可指数收敛到目标分布，并利用重加权归一化流对其输运与反应分量进行神经参数化实现。

    

    我们提出了一种从由未归一化玻尔兹曼密度所指定的分布中进行采样的神经算法。我们的方法基于在Wasserstein-Fisher-Rao几何中对Kullback-Leibler散度应用Jordan-Kinderlehrer-Otto格式（WFR JKO格式）。我们的贡献有两个方面。首先，我们证明了对于任意固定的步长，精确的WFR JKO迭代在迭代次数趋于无穷时以指数速度收敛到目标分布。值得注意的是，这一结果无需对目标分布做任何结构性假设，例如对数凹性或对数Sobolev不等式。其次，我们开发了WFR JKO格式的神经实现，使用重加权归一化流对其输运和反应分量进行参数化。在具有挑战性的多峰目标分布上的数值实验表明了所提方法的良好性能。

    arXiv:2610.10278v1 Announce Type: cross  Abstract: We propose a neural algorithm for sampling from distributions specified by unnormalized Boltzmann densities. Our approach is based on the Jordan--Kinderlehrer--Otto scheme for the Kullback--Leibler divergence in the Wasserstein--Fisher--Rao geometry (WFR JKO scheme). Our contributions are twofold. First, we prove that, for any fixed step size, the exact WFR JKO iterates converge exponentially fast to the target as the number of iterations tends to infinity. Notably, this result requires no structural assumptions on the target, such as log-concavity or a logarithmic Sobolev inequality. Second, we develop a neural implementation of the WFR JKO scheme that parametrizes its transport and reaction components using reweighted normalizing flows. Numerical experiments on challenging multimodal targets demonstrate the promising performance of the proposed method.
    
[^58]: PatchBench：衡量激活修补中的附带损害

    PatchBench: Measuring Collateral Damage in Activation Patching

    [https://arxiv.org/abs/2610.10276](https://arxiv.org/abs/2610.10276)

    提出PatchBench基准，用于衡量激活修补在修复LLM越狱行为时对无关行为造成的附带损害，从而区分真正的选择性修复与更广泛的局部行为抑制。

    

    LLM安全补丁可能通过了某个基准测试，却仍然是一个糟糕的修复。这种风险在越狱修复中尤为突出，因为其目标是在不改变无关行为的前提下纠正特定的不安全行为。一个补丁可能能够阻止精确的评估提示，却在相近的有害变体上失效，或者通过过度拒绝共享其措辞或结构的良性提示来抑制有害行为。现有协议主要测试模型是否会被攻破，而聚合指标（攻击成功率、拒绝率、全局能力）无法区分选择性修复与更广泛的局部抑制。为了填补这一空白，我们引入了PatchBench，这是一个基于实证观察到的、诱导出可操作有害答案的模型特定越狱失败案例构建的基准。我们从37个公开数据集中的27,870个提示出发，筛选出15,314个英文提示，并对8个开源指令微调模型进行查询。结合WildGuard过滤、成对Elo排名……

    arXiv:2610.10276v1 Announce Type: cross  Abstract: An LLM safety patch can pass a benchmark while still being a poor repair. This risk is especially acute for jailbreak repairs, where the goal is to correct a specific unsafe behaviour without changing unrelated behaviours. A patch may block exact evaluation prompts yet fail on close harmful variants, or suppress harmful behaviour by over-refusing benign prompts that share its wording or structure. Existing protocols primarily test whether models can be broken, while aggregate metrics (attack success, refusal rates, global capability) cannot distinguish selective repairs from broader local suppression. To address this gap, we introduce PatchBench, a benchmark of empirically observed model-specific jailbreak failures inducing actionable harmful answers. Starting from 27,870 prompts from 37 public datasets, we curate 15,314 English prompts and query 8 open-source instruction-tuned models. Combining WildGuard filtering, pairwise Elo rankin
    
[^59]: 基于代价梯度的视觉世界模型稀疏规划

    Sparse Planning in Visual World Models via Cost Gradients

    [https://arxiv.org/abs/2610.10274](https://arxiv.org/abs/2610.10274)

    本文提出COSTGRAD，一种无需训练的token选择方法，通过规划代价对每个token的梯度范数筛选出对规划真正重要的空间token，在50%稀疏度下即可保持或超越全token规划的性能，并结合精简CEM搜索实现最高约5倍的规划加速。

    

    基于token的世界模型能够实现细粒度的潜在规划，但反复处理大规模空间token网格使得动作搜索代价高昂。我们提出COSTGRAD，一种无需训练、目标条件驱动的选择器，它通过规划代价相对于每个输入token的梯度范数来对空间token进行排序。由于重要性是从下游控制目标中推导出来的，COSTGRAD聚焦于对规划真正重要的token，而不仅仅是服务于预测的token。在50%稀疏度的AdaLN条件化预测器上，COSTGRAD在四个连续控制基准中的三个上匹配或超越全token规划，同时实测每个环境规划步骤获得2.6倍的实际运行时间加速。将token稀疏化与精简的CEM搜索相结合，总加速可达约5倍，且性能仍超过全token基线。我们还识别出一种依赖于架构的失败模式：在匹配的AdaLN与concat架构对比中，concat保持（此处摘要内容截断）。

    arXiv:2610.10274v1 Announce Type: new  Abstract: Token-based world models enable fine-grained latent planning, but repeatedly processing large spatial token grids makes action search expensive. We introduce COSTGRAD, a training-free, goal-conditioned selector that ranks spatial tokens by the gradient norm of the planning cost with respect to each input token. By deriving importance from the downstream control objective, COSTGRAD targets tokens that matter for planning rather than merely for prediction. On AdaLN-conditioned predictors at $50\%$ sparsity, COSTGRAD matches or exceeds full-token planning on three of four continuous-control benchmarks, while giving a measured $2.6\times$ wall-clock speedup per environment planning step. Combining token sparsity with reduced CEM search increases this to a $\sim 5\times$ total speedup while still exceeding the full-token baseline. We also identify an architecture-dependent failure mode: in a matched AdaLN-vs-concat comparison, concat maintain
    
[^60]: 基于学习评论家与裁剪机制的PPO闭环非渐近收敛性分析

    A Closed-Loop Non-Asymptotic Convergence Analysis of PPO with Learned Critics and Clipping

    [https://arxiv.org/abs/2610.10273](https://arxiv.org/abs/2610.10273)

    本文将PPO-Clip建模为闭环演员-评论家系统并首次给出非渐近收敛性分析，统一刻画了策略平稳性与评论家跟踪精度，并显式揭示了对算法参数（包括延迟相关的步长限制）的依赖。

    

    尽管应用广泛，带裁剪的近端策略优化（PPO-Clip）仍然难以调参，且评论家学习、裁剪与轨迹重用之间的相互作用尚未被完全理解。我们将PPO-Clip视为一个闭环的演员-评论家系统，并对其进行了非渐近分析。该分析在显式的覆盖率与评论家正则性假设下，使用原始GAE与蒙特卡洛评论家目标，刻画了演员-评论家耦合、非光滑的概率比率裁剪、有限批量重用以及可预测的早停机制。我们的同步与异步保证共同刻画了策略平稳性与学习到的评论家的跟踪精度，并对算法参数具有显式依赖。一个充分的耦合条件为优化误差、评论家跟踪误差、裁剪误差和有限批量误差给出了统一的放大界。异步结果还需要一个依赖于延迟的评论家步长限制；……

    arXiv:2610.10273v1 Announce Type: new  Abstract: Despite its widespread use, Proximal Policy Optimization with clipping (PPO-Clip) remains difficult to tune, and the interactions among critic learning, clipping, and rollout reuse remain incompletely understood. We develop a \emph{non-asymptotic} analysis of PPO-Clip as a \emph{closed-loop actor--critic} system. It captures actor--critic coupling, nonsmooth probability-ratio clipping, finite-batch reuse, and predictable early stopping under explicit coverage and critic regularity assumptions, using raw GAE and Monte Carlo critic targets. Our synchronous and asynchronous guarantees jointly characterize policy stationarity and the tracking accuracy of the learned critic, with explicit dependence on algorithmic parameters. A sufficient coupling condition gives optimization, critic tracking, clipping, and finite-batch errors a common amplification bound. The asynchronous result also requires a delay-dependent critic stepsize restriction; vi
    
[^61]: 使用小语言模型逆向工程机器学习流水线结构

    Using Small Language Models to Reverse-Engineer Machine Learning Pipelines Structures

    [https://arxiv.org/abs/2610.10261](https://arxiv.org/abs/2610.10261)

    该研究验证了小语言模型能够凭借其代码理解与分类能力，从源代码中有效提取机器学习流水线的阶段结构，从而克服人工标注不可扩展及传统分类器难以适应领域多样性的局限。

    

    背景：一旦定义了构建机器学习（ML）流水线的阶段分类体系（例如数据预处理、建模等），从源代码中提取这些阶段对于更好地理解机器学习实践至关重要。然而，机器学习的持续演进（例如算法、数据集的更新）所带来的多样性使这项任务充满挑战。现有方法要么依赖无法扩展的人工标注，要么依赖无法妥善支持领域多样性的分类器。这些局限性呼唤更可靠的解决方案。目标：我们评估小语言模型（SLM）能否利用其代码理解与分类能力来应对这些局限，并加深我们对机器学习实践的理解。方法：我们基于两篇代表当前技术局限性的相关参考工作开展了验证性研究。我们首先使用Cochran's Q检验比较多个小语言模型，然后对表现最佳的模型进行评估。

    arXiv:2610.10261v1 Announce Type: cross  Abstract: Context: Once defined a taxonomy of stages structuring Machine Learning (ML) pipelines (e.g. Data Preprocessing, Modeling...), extracting these stages from source code is key for better understanding ML practices. However, the diversity caused by the constant evolution of ML (e.g., algorithms, datasets) makes this task challenging. Existing approaches either rely on non-scalable manual labeling or on classifiers that do not properly support domain's diversity. These limitations call for more reliable solutions.   Objective: We evaluate whether Small Language Models (SLMs) can leverage their code understanding and classification abilities to address these limitations, and enhance our understanding of practices in ML.   Method: We conduct a confirmatory study based on two relevant reference works representing current limitations in the state-of-the-art. We first compare several SLMs using Cochran's Q test, then evaluate the best-performi
    
[^62]: PairAudit：分布偏移下利用图Token引导人工审查

    PairAudit: Guiding Human Review with Graph Tokens under Distribution Shift

    [https://arxiv.org/abs/2610.10260](https://arxiv.org/abs/2610.10260)

    PairAudit利用图Token捕捉相连节点间的预测关系模式，在固定审查预算下发现入侵检测器中被忽略的高置信度错误，在分布偏移（未见攻击）场景下比基于不确定性的审查方法纠正更多错误。

    

    入侵检测器可能会以高置信度错误分类训练中未见过的攻击。人工审查可以纠正这些错误，但只能检查有限数量的案例。基于不确定性的审查可能会忽略高置信度的错误，而仅凭异常分数也无法说明改变审查计划是否能纠正更多错误。我们提出PairAudit，以在固定预算下发现被忽略的错误并改进审查。它的图Token能够捕获相连节点之间的预测模式。PairAudit并非通过特征聚合构建另一个预测器，而是利用异常的关系模式来揭示现有预测中的潜在错误，随后通过人工反馈帮助判断这些发现是否足以证明需要调整审查优先级。跨安全任务的实验表明，PairAudit平均比基于不确定性的审查纠正更多错误，包括在未见攻击上纠正更多错误。这些收益已将所有审查成本考虑在内。

    arXiv:2610.10260v1 Announce Type: new  Abstract: Intrusion detectors can confidently misclassify attacks that were not seen during training. Human review can correct these errors, but only a limited number of cases can be checked. Uncertainty-based review may overlook confident errors, while anomaly scores alone do not show whether changing the review plan will correct more errors. We introduce PairAudit to find overlooked errors and improve review under a fixed budget. Its graph tokens capture prediction patterns across connected nodes. Rather than building another predictor through feature aggregation, PairAudit uses unusual relational patterns to uncover potential errors in existing predictions. Human feedback then helps decide whether these findings justify changing review priorities. Experiments across security tasks show that PairAudit corrects more errors on average than uncertainty-based review, including more errors on unseen attacks. These gains account for all review costs a
    
[^63]: 论牛路搜索算法中的循环假设

    On the Cyclic Assumption of the Cow-Path Search Algorithm

    [https://arxiv.org/abs/2610.10253](https://arxiv.org/abs/2610.10253)

    本文为牛路搜索问题中“没有任何算法能优于最佳循环顺序算法”这一断言提供了详细的证明，从而完整补足了该随机算法最优性的论证。

    

    在牛路问题中，一头牛必须在 $w$ 条仅在原点相连的路径之一上寻找一个位于未知距离处的目标，其性能通过竞争比来衡量。Kao、Reif 和 Tate 设计了一种高效的随机算法，其中牛按照固定的循环顺序访问各条路径。他们证明了该算法在 $w=2$ 时是最优的，随后 Kao、Ma、Sipser 和 Yin 证明了该算法对所有 $w$ 都是最优的，并提出了一个断言：没有任何算法能够优于最佳的循环算法。本注释为该断言提供了详细的证明。

    arXiv:2610.10253v1 Announce Type: cross  Abstract: In the cow-path problem, a cow must find a goal lying at an unknown distance on one of $w$ paths connected only at the origin, and performance is measured by competitive ratio. Kao, Reif and Tate designed an efficient randomized algorithm in which the cow visits the paths in a fixed cyclic order. They proved the algorithm is optimal for $w=2$, and subsequently Kao, Ma, Sipser and Yin proved its optimality for all $w$, with a claim that no algorithm does better than the best cyclic one. This note provides a detailed proof of that claim.
    
[^64]: 分段平稳自校正调节中通过被动变化检测实现对数遗憾

    Logarithmic Regret via Passive Change Detection in Piecewise-Stationary Self-Tuning Regulation

    [https://arxiv.org/abs/2610.10250](https://arxiv.org/abs/2610.10250)

    本文提出PIECE-CD算法，利用“系统变化会在旧控制器下提高输出能量”这一被动检测机制，对分段平稳自回归系统的最小方差自校正控制实现了概率至少为1-δ的O((C+1)log((T+1)/δ))对数遗憾。

    

    我们研究具有外生输入且系数在未知时刻发生变化的未知自回归系统的最小方差控制问题。在有界独立扰动、固定检测间隙、稳定性与可行性条件以及变化之间具有充分时间的假设下，我们以至少1-δ的概率证明了O((C+1)log((T+1)/δ))的遗憾，其中T为时间范围，C为变化次数。与切换老虎机问题不同（在切换老虎机中未被选择的臂可能在未被观察的情况下发生变化），此处的被控对象变化在利用阶段本身即可提供信息：正确的可行控制器会使输出中仅剩扰动，而可检测的变化则会在旧控制器下提高输出能量。PIECE-CD算法在初始阶段和警报触发后进行探索，随后使用门控递归最小二乘法进行控制。其能量测试将窗口化的输出功率与高于噪声底限的阈值进行比较；对不稳定控制器失配情形的扩展也……（原文摘要至此截断）

    arXiv:2610.10250v1 Announce Type: new  Abstract: We study minimum-variance control of an unknown autoregressive system with exogenous inputs and coefficients that change at unknown times. Under bounded independent disturbances, fixed detection gaps, stability and feasibility conditions, and sufficient time between changes, we prove \(O((C+1)\log((T+1)/\delta))\) regret with probability at least \(1-\delta\), where \(T\) is the horizon and \(C\) the number of changes. Unlike switching bandits, where unselected arms can change unobserved, admissible plant changes provide information during exploitation: the correct feasible controller leaves only the disturbance in the output, whereas a detectable change raises output energy under the old controller. PIECE-CD explores initially and after alarms, then uses gated recursive least squares for control. Its energy test compares windowed output power with a threshold above the noise floor; the extension to unstable controller mismatches also mo
    
[^65]: 非线性双时间尺度随机逼近中的平稳偏差与外推

    Stationary Bias and Extrapolation in Nonlinear Two-Timescale Stochastic Approximation

    [https://arxiv.org/abs/2610.10246](https://arxiv.org/abs/2610.10246)

    本文针对马尔可夫链驱动的非线性双时间尺度随机逼近，推导了在慢步长远小于快步长时仍一致有效的一阶偏差展开，揭示出 ε²/η 的混合偏差项，并表明沿幂律步长路径的偏差指数可以是非整数，因此 Richardson–Romberg 外推必须使用与步长路径相匹配的权重。

    

    常数步长的随机逼近通常具有非零的平稳均值误差，且该误差在时间平均后依然存在。本文研究由外生有限状态马尔可夫链驱动的非线性双时间尺度递推中的这一误差。在给定的光滑性假设和平稳分布条件下，我们推导了一阶偏差展开，其误差界在慢步长远小于快步长时仍保持一致。快流形坐标使得在该极限下相应的协方差方程保持正则。对于快步长 η 和慢步长 ε，该展开揭示了 ε²/η 的混合贡献项以及与各步长呈线性关系的项。这种依赖关系对偏差削减具有重要意义：沿幂律步长路径，偏差指数未必是整数，因此 Richardson–Romberg 外推需要采用与路径相匹配的权重。一个可精确求解的非线性马尔可夫例子验证了……

    arXiv:2610.10246v1 Announce Type: new  Abstract: Constant-step stochastic approximation generally has a nonzero stationary mean error that persists under time averaging. This paper studies that error for nonlinear two-timescale recursions driven by an exogenous finite-state Markov chain. Under stated smoothness assumptions and conditions on the stationary distribution, we derive a first-order bias expansion whose error bound remains uniform as the slow step size becomes much smaller than the fast step size. Fast-manifold coordinates keep the associated covariance equation regular in this limit. For fast step $\eta$ and slow step $\varepsilon$, the expansion reveals a mixed contribution $\varepsilon^2/\eta$ alongside terms linear in each step size. This dependence matters for bias reduction: along power-law step-size paths, the bias exponents need not be integers, so Richardson--Romberg extrapolation requires weights matched to the path. An exactly solvable nonlinear Markov example veri
    
[^66]: MorphCL：用于基于惯性传感器的人体活动识别的形态学对比学习

    MorphCL: Morphological Contrastive Learning for Inertial-based Human Activity Recognition

    [https://arxiv.org/abs/2610.10245](https://arxiv.org/abs/2610.10245)

    MorphCL是一种自监督预训练框架，通过基于运动基元发现和领域特定特征描述符的结构感知分组，将全局结构的显式建模引入基于惯性传感器的人体活动识别自监督学习中，从而解决了现有方法依赖随机采样和局部比较、忽视大规模运动数据全局结构的问题。

    

    尽管可穿戴和移动设备中的传感器无处不在，并产生了大量的人体运动数据，但如何将未标注的记录转化为基础运动模型仍然是一个悬而未决的挑战。自监督学习（SSL）已经减少了对昂贵标注的需求，然而现有方法在很大程度上未能利用大规模运动数据的全局结构，它们依赖于随机采样的批次和局部比较，这对于以静止、低方差行为为主的真实场景惯性数据来说问题尤为突出。本文提出了形态学对比学习（MorphCL），这是一种自监督预训练框架，通过结构感知分组，将全局结构的显式建模注入到基于惯性的SSL方法中。基于运动分析的两个成熟支柱——运动基元（即模体）的发现以及领域特定的特征描述符——我们展示了MorphCL（摘要在此处被截断）

    arXiv:2610.10245v1 Announce Type: new  Abstract: Despite the ubiquity of sensors in wearable and mobile devices and the abundance of human movement data they generate, translating unlabeled recordings into foundational motion models remains an open challenge. Self-supervised learning (SSL) has alleviated the need for costly annotations, yet existing approaches leave the global structure of large-scale motion data largely untapped, relying on randomly sampled batches and local comparisons that become particularly problematic for in-the-wild inertial data dominated by stationary, low-variance behaviors. Here we introduce Morphological Contrastive Learning (MorphCL), a self-supervised pretraining framework that uses structure-aware grouping to inject explicit modeling of global structure into inertial-based SSL approaches. Building on two well-established pillars of motion analysis, the discovery of motion primitives, or motifs, and domain-specific feature descriptors, we show that MorphC
    
[^67]: ProtocolMatch：面向科学动力学预测的协议依赖模型选择

    ProtocolMatch: Protocol-Dependent Model Selection for Scientific Dynamics Forecasting

    [https://arxiv.org/abs/2610.10239](https://arxiv.org/abs/2610.10239)

    该论文提出 ProtocolMatch 评估框架，将科学动力学预测的模型选择从单纯的架构比较扩展为“协议依赖”问题，并通过量子自旋动力学实验证明模型优劣排序会随训练规模、观测历史和闭环部署等协议条件而逆转。

    

    科学动力学预测通常被框定为一种架构选择问题，但实际的模型部署还取决于观测历史、滚动预测反馈、计算预算、物理目标以及测试分布。我们提出了协议依赖的模型选择问题，并引入 ProtocolMatch——一个计算量匹配、由验证集选择且保留失败案例的评估框架。在受驱动的量子自旋动力学任务上，我们在三个独立生成的数据集上比较了循环网络、分块注意力、因果注意力和低秩线性预测器。在一个固定的双自旋任务中，随着训练集规模的增长，因果注意力与循环网络之间的优劣排序会发生逆转；而在六自旋局部可观测量比较中，线性预测器具有最低的平均误差。在四自旋研究中，限制观测历史会使所有“刷新历史”视角下的性能变差，但会改善所有“闭环”视角下的性能。在每个数据集上，一个仅使用最新状态的 MLP 都比持续性（persistence）预测具有更低的误差。

    arXiv:2610.10239v1 Announce Type: new  Abstract: Scientific dynamics forecasting is often framed as an architecture choice, although deployment is also determined by observed history, rollout feedback, compute budget, physical objective, and test distribution. We formulate protocol-dependent model selection and introduce ProtocolMatch, a compute-matched, validation-selected, and failure-preserving evaluation framework. On driven quantum-spin dynamics, we compare recurrent, patched-attention, causal-attention, and low-rank linear predictors across three independently generated datasets. The causal-attention--recurrence ordering reverses as the training set grows within a fixed two-spin task, while a linear predictor has the lowest mean error in the six-spin local-observable comparison. Restricting observed history worsens every refreshed-history view but improves every closed-loop view in the four-spin study. A latest-state MLP has lower error than persistence on every dataset under sta
    
[^68]: 从提示到树：面向少样本表格分类的高效LLM引导树生成方法

    From Prompts to Trees: Effective LLM-Guided Tree Generation for Few-Shot Tabular Classification

    [https://arxiv.org/abs/2610.10227](https://arxiv.org/abs/2610.10227)

    本文提出一种三阶段的LLM引导框架，通过提示LLM先生成规则再将其组织成决策树，在少样本表格分类任务中以显著更低的提示开销实现了更优的准确性和可解释性。

    

    尽管大语言模型（LLMs）拥有丰富的世界知识和令人印象深刻的泛化能力，但其直接应用于表格数据分类受到高推理成本和有限可解释性的阻碍。相比之下，决策树快速且透明，但在低数据量情况下往往表现不佳。在本工作中，我们提出了一个新颖的框架，通过在少样本学习设置下将LLM知识蒸馏到可解释的决策树中，从而弥合这两种范式。我们没有直接提示LLM生成完整的树（这通常不稳定且低效），而是开发了一种三阶段范式，提示LLM生成规则并将这些规则组织成树。在多个真实世界表格数据集上的实验表明，与现有基线相比，我们的方法以显著更低的提示开销实现了更优的准确性和可解释性。

    arXiv:2610.10227v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) possess rich world knowledge and impressive generalization capabilities, their direct application to tabular data classification is hindered by high inference costs and limited interpretability. In contrast, decision trees are fast and transparent but often underperform in low-data regimes. In this work, we propose a novel framework that bridges these paradigms by distilling LLM knowledge into interpretable decision trees under a few-shot learning setting. Instead of directly prompting the LLM to generate full trees, which is often unstable and inefficient, we develop a three-stage paradigm that prompts the LLM to generate rules and organize the rules into a tree. Experiments on multiple real-world tabular datasets demonstrate that our method achieves superior accuracy and interpretability with significantly lower prompting overhead compared to existing baselines.
    
[^69]: 评估差分隐私合成时间序列预测中的序列组装策略

    Evaluating Sequence Assembly Strategies for Differentially Private Synthetic Time-Series Forecasting

    [https://arxiv.org/abs/2610.10222](https://arxiv.org/abs/2610.10222)

    该论文首次系统研究了差分隐私合成时间序列在生成后的窗口组装策略（重叠率与窗口加权方案），并通过在四类数据集和五种预测模型上的实验，揭示出下游预测效用由预测模型、重叠率和加权方案共同决定。

    

    差分隐私时间序列生成器通常产生固定长度的合成窗口，而下游预测模型往往需要长的连续训练序列。因此，这些窗口在生成之后如何被组装，会改变呈现给预测模型的有效合成数据，即使训练好的生成器本身保持不变。我们通过系统地改变重叠率和窗口加权方案来研究这一生成后的序列组装过程，并从边界连续性、统计与时间保真度以及“合成数据训练-真实数据测试”（TSTR）预测效用等角度评估所得序列。在四类公开数据集和五种预测模型上的结果显示出一个清晰的依赖于预测模型的组装原则：下游TSTR效用由预测模型、重叠率和窗口加权方案共同塑造，从而产生了……（原文摘要在此处截断）

    arXiv:2610.10222v1 Announce Type: new  Abstract: Differentially private time-series generators commonly produce fixed-length synthetic windows, whereas downstream forecasting models often require long continuous training sequences. How these windows are assembled after generation can therefore alter the effective synthetic data presented to a forecaster, even when the trained generator remains unchanged. We study this post-generation sequence assembly process by systematically varying overlap rates and window-weighting schemes and evaluating the resulting sequences in terms of boundary continuity, statistical and temporal fidelity, and Train-on-Synthetic-Test-on-Real (TSTR) forecasting utility. Across four types of public datasets (ETTh1, ETTm1, Weather, and Appliances) and five forecasting models, the results reveal a clear forecaster-dependent assembly principle: downstream TSTR utility is jointly shaped by the forecaster, overlap rate, and window-weighting scheme, leading to distinc
    
[^70]: 海森引导扰动Wasserstein梯度流的有限样本逼近

    Finite-Sample Approximation of Hessian-Guided Perturbed Wasserstein Gradient Flows

    [https://arxiv.org/abs/2610.10218](https://arxiv.org/abs/2610.10218)

    该论文证明了海森引导扰动Wasserstein梯度流的有限粒子逼近在增长时间尺度上的高概率追踪界，关键在于沿参考路径累积的曲率——负曲率会放大误差而正曲率可抑制误差，从而刻画了暂时不稳定仍可精确追踪的有利情形。

    

    Wasserstein梯度流将梯度下降方法推广到了概率测度空间。其海森（Hessian）引导的扰动变体（PWGF）通过引入高斯扰动来帮助逃离非凸问题中的鞍点。我们研究了该流被有限个相互作用的粒子近似时，在随时间增长的时间尺度上何时仍能保持近似精度。我们的分析保留了沿种群驱动参考路径所累积的曲率信息：负曲率可能会放大近似误差，而随后出现的正曲率则可以抑制这些误差的影响。这刻画了一些有利的情形，即暂时的不稳定性与在增长时间尺度上的精确追踪可以并存。在正则性假设和预先设定的公共扰动调度下，我们对满足显式累积曲率条件的参考路径，在高概率事件上证明了粒子追踪界和目标值追踪界。为处理状态相关的高斯跳跃，我们构造了一个种群（摘要在此处截断）

    arXiv:2610.10218v1 Announce Type: new  Abstract: Wasserstein gradient flow extends gradient descent to probability measures. Its Hessian-guided perturbed variant (PWGF) adds Gaussian perturbations to escape saddle points in nonconvex problems. We investigate when its approximation by finitely many interacting particles remains accurate over growing time horizons. Our analysis retains the curvature accumulated along the population-driven reference path: negative curvature can amplify approximation errors, while subsequent positive curvature can damp their influence. This captures favorable scenarios in which temporary instability is compatible with accurate tracking over growing horizons. Under regularity assumptions and a prescribed common perturbation schedule, we prove particle and objective-value tracking bounds on a high-probability event for reference paths satisfying explicit conditions on accumulated curvature. To handle state-dependent Gaussian jumps, we construct a population-
    
[^71]: RoBART：具有树特定旋转的贝叶斯可加回归树

    RoBART: Bayesian Additive Regression Trees with Tree-Specific Rotations

    [https://arxiv.org/abs/2610.10214](https://arxiv.org/abs/2610.10214)

    RoBART通过为每棵树分配特定的Givens旋转来改进贝叶斯可加回归树，使其能高效逼近与预测变量轴不对齐的边界，并对具有各向异性Hölder光滑性的可加函数证明了后验收缩率。

    

    贝叶斯可加回归树（BART）在逼近与预测变量轴不对齐的边界时可能需要大量的分裂。RoBART为每棵树分配一个由其所有内部节点共享的旋转，从而在旋转后的坐标系中仍保持轴对齐分裂和常数叶节点的形式。我们通过Metropolis-Hastings方法联合提出Givens旋转序列以及在所得网格上的切分点，并在将叶节点均值积分掉的条件后验下建立了马氏链的可逆性。对于具有分量特定旋转和各向异性Hölder光滑性的可加函数，我们证明了在经验$L_2$距离下的后验收缩性以及噪声标准差的后验收缩性。在所述先验、设计和网格条件下，当预测变量、树和分量的数量固定且分量数不超过树数时，收敛速率为由各分量的光滑度和所使用的旋转坐标数量决定的分量速率之和。我们还建立了一个后验……

    arXiv:2610.10214v1 Announce Type: cross  Abstract: Bayesian additive regression trees (BART) can require many splits to approximate boundaries misaligned with the predictor axes. RoBART assigns each tree a rotation shared by all internal nodes, retaining axis-aligned splits in rotated coordinates and constant leaves. We jointly propose a Givens rotation sequence and cutpoints on the resulting grid by Metropolis-Hastings and establish reversibility with respect to the conditional posterior with leaf means integrated out. For additive functions with component-specific rotations and anisotropic H\"older smoothness, we prove posterior contraction in empirical $L_2$ distance and for the noise standard deviation. Under the stated prior, design, and grid conditions, with fixed numbers of predictors, trees, and components and no more components than trees, the rate is a sum of componentwise rates determined by smoothness and the number of rotated coordinates used. We also establish a posterior
    
[^72]: 边准确率并不足够：为什么从动力学学到的结构无法迁移到逆问题

    Edge Accuracy Is Not Enough: Why Dynamics-Learned Structure Fails to Transfer to Inverse Problems

    [https://arxiv.org/abs/2610.10213](https://arxiv.org/abs/2610.10213)

    本文证明，尽管从动力学中学到的图结构（如NRI）在理论上满足降低样本复杂度的边误差条件，但在迁移到逆问题时仍会系统性失效——边的准确率并不足够，而基于物理的先验（如格林函数）表现明显更优。

    

    对于标记数据稀缺的逆问题，一个自然的策略是从丰富的前向模拟数据中迁移学到的关系结构。我们证明这种策略会系统性地失败，即使它满足了关于结构为何有帮助的标准理论依据。我们证明，只要边误差满足 Δ < n² − kn，近似结构就能带来估计误差上的收益，将样本复杂度从 O(n²) 降低到 O(kn+Δ)。通过神经关系推断（NRI）从动力学预测中学到的结构满足这一条件，然而在一个涵盖180个CFD模拟氢气泄漏场景和180个声学场景的源定位任务中，相对于灵活的、任务优化的注意力基线，它使性能分别下降了116%和201%，而基于物理的先验（格林函数）仅下降69–72%。四条独立的证据线表明这并非调参失败：NRI仅带来约0……（摘要在此处截断）

    arXiv:2610.10213v1 Announce Type: new  Abstract: A natural strategy for inverse problems with scarce labelled data is to transfer relational structure learned from abundant forward-simulation data. We show this strategy fails systematically, even when it satisfies the standard theoretical justification for why structure should help. We prove that approximate structure provides estimation-error benefits whenever the edge error satisfies $\Delta < n^2 - kn$, reducing sample complexity from $O(n^2)$ to $O(kn+\Delta)$. Structure learned via Neural Relational Inference (NRI) from dynamics prediction satisfies this condition, yet on a source-localisation task across 180 CFD-simulated hydrogen-leak scenarios and 180 acoustic scenarios, it degrades performance by 116% and 201% relative to a flexible, task-optimised attention baseline, while a physics-based prior (Green's function) degrades by only 69-72%. Four independent lines of evidence show this is not a tuning failure: NRI improves only 0
    
[^73]: OrthoGen：一种用于时变处理的生成式正交学习器

    OrthoGen: A Generative Orthogonal Learner for Time-Varying Treatments

    [https://arxiv.org/abs/2610.10210](https://arxiv.org/abs/2610.10210)

    提出OrthoGen框架，通过生成式递归g-计算与正交化学习相结合，在时变混杂下利用灵活的生成模型估计时变处理的条件分布势结果，并增强对冗余参数估计误差的稳健性。

    

    在医学领域，估计随时间变化的条件分布势结果（CDPOs）十分重要（例如，用于估计不同治疗序列下患者特定的风险）。然而，由于时变混杂的存在，这项任务充满挑战，而现有的针对该任务的调整策略较为有限。本文旨在利用灵活的生成模型学习时变处理下的CDPOs。我们的贡献有两个方面：（1）我们为该场景引入了一种量身定制的调整策略，即生成式递归g-计算（generative recursive g-computation）。该调整策略递归地传播完整的条件结果分布而非条件均值，直接对感兴趣的变量建模而非对完整轨迹建模。（2）基于该调整策略，我们构建了用于CDPO估计的简洁生成学习器。然而，这些学习器对冗余（nuisance）参数估计误差较为敏感，这促使我们进一步提出正交学习器（原文摘要在此处截断）。

    arXiv:2610.10210v1 Announce Type: new  Abstract: Estimating conditional distributional potential outcomes (CDPOs) over time is important in medicine (e.g., to estimate patient-specific risks under different treatment sequences). However, this task is challenging because of time-varying confounding, yet existing adjustment strategies for this task are limited. In this paper, we aim to learn CDPOs under time-varying treatments using flexible generative models. Our contributions are two-fold. (1) We introduce a tailored adjustment strategy for our setting, namely, generative recursive g-computation. Our adjustment strategy recursively propagates full conditional outcome distributions rather than conditional means, modeling the variables of interest directly rather than full trajectories. Building on our adjustment strategy, we formulate simple generative learners for CDPO estimation. However, these learners can be sensitive to nuisance estimation errors, which motivates an orthogonal lear
    
[^74]: CARES：一个说话者对声音反应的受控合成基准

    CARES: A Controlled Synthetic Benchmark of Speaker Reactions to Sound

    [https://arxiv.org/abs/2610.10208](https://arxiv.org/abs/2610.10208)

    本文提出CARES——一个包含10,000个双说话者场景的受控合成基准，通过“说话者对声音产生可听见的反应”这一规则定义声音显著性以固定真实标注，并揭示现有音频-语言模型虽能识别声音本身，却无法理解说话者对声音的反应。

    

    自动音频场景描述旨在将录音转化为对情境的文字描述。其中的一个难点在于决定应保留音频中的哪些元素，因为描述无法涵盖所有元素，而标注者对此意见不一，导致真实标注（ground truth）难以获得。在这项工作中，我们首先定义真实标注，然后再生成数据。我们聚焦于音频事件，并用一个简单的规则来定义声音显著性：当说话者对某个声音产生可听见的反应时，该声音即为显著的。为了实现规模与多样性，我们采用一组受控场景来固定真实标注，并借助语言模型撰写对话。由此构建的语料库CARES包含10,000个双说话者场景。随后，我们在三个任务上对六个音频-语言模型进行了基准测试：识别场景、标记存在的声音以及对反应进行分类。结果表明，这些模型能够听到声音，却无法捕捉说话者对声音的反应方式。

    arXiv:2610.10208v1 Announce Type: cross  Abstract: Automatic audio scene description turns a recording into a text account of a situation. One difficulty is deciding which elements of the audio should be kept, since a description cannot include them all. Annotators disagree about this, making a ground truth hard to obtain. In this work, we first define the ground truth, then generate the data. We focus on audio events and define sound salience with a simple rule: a sound is salient when a speaker audibly reacts to it. For scale and variety, a controlled set of scenarios fixes the ground truth, and a language model writes the dialogues. The resulting corpus, CARES, contains 10,000 two-speaker scenes. We then benchmark six audio-language models on three tasks: identifying the scene, tagging the sounds present, and classifying reactions. We show that these models hear the sounds but miss how the speakers react to them.
    
[^75]: 通过机器学习计算链环的切片亏格与解结数

    Computations of the slice genus and the unknotting number of links via machine learning

    [https://arxiv.org/abs/2610.10206](https://arxiv.org/abs/2610.10206)

    该论文利用强化学习和贝叶斯优化，为链环的切片亏格、解结数等难以算法计算的不变量求出新上界，并结合已知下界在许多情形下得到新的精确值，还重现了解结数的非可加性反例。

    

    链环是光滑嵌入在 $S^3$ 中的圆的不相交并集。我们使用强化学习和贝叶斯优化，为几种尚不知道是否可以算法计算的链环不变量获得了新的上界：链环的切片亏格和解结数，以及代数可分裂链环的强切片亏格。我们还利用已知不变量计算了下界。通过结合上界与下界，我们在许多情况下得到了新的精确值。我们的解结智能体能够重现 Brittenham 和 Hermiller 提出的若干反例中解结数的非可加性，并在某些情况下找到了新的解结轨迹。

    arXiv:2610.10206v1 Announce Type: cross  Abstract: Links are disjoint unions of circles smoothly embedded in $S^3$. We use reinforcement learning and Bayesian optimisation to obtain new upper bounds on several link invariants that are not known to be algorithmically computable: the slice genus and the unknotting number for links, and the strong slice genus for algebraically split links. We also compute lower bounds using known invariants. Combining the upper and lower bounds, we obtain new exact values in many cases. Our unknotting agents can reproduce the non-additivity of the unknotting number for several counterexamples due to Brittenham and Hermiller, in some cases finding new unknotting trajectories.
    
[^76]: 如何训练你的模式生物

    How to train your model organism

    [https://arxiv.org/abs/2610.10203](https://arxiv.org/abs/2610.10203)

    该论文提出对齐模式生物的验证不应止于单一目标行为安装，而应从目标行为安装、通用能力保持和输出自然性三个维度进行验证，并证明这些验证指标可以有效预测可解释性方法能否成功恢复已安装的行为。

    

    对齐相关行为（如后门、谄媚性、虚假相关性）的模式生物已成为评估白盒可解释性技术的关键工具。我们认为，当前普遍采用的仅以单一目标（即安装目标行为）来训练模式生物的做法是不充分的，并提出应从三个目标及相关指标来验证模式生物：目标行为安装、通用能力保持（即参数化知识、对话质量）和输出自然性（即思维链和激活值）。我们运用这一验证框架重新审视了两个公开发布的模式生物套件，结果表明：（1）在各类训练方案下，对话质量和思维链自然性均出现显著退化；（2）验证指标能够预测可解释性方法恢复已安装行为的效果，例如，逻辑透镜（logit lens）读出结果与模式生物的通用能力存在协变关系。我们引入了一个多……（摘要在此处被截断）

    arXiv:2610.10203v1 Announce Type: new  Abstract: Model organisms of alignment-relevant behaviors (e.g., backdoors, sycophancy, spurious correlations) have emerged as a key tool for evaluating whitebox interpretability techniques. We argue that the prevailing practice of training model organisms to a single objective of installing the target behavior is insufficient and propose validating model organisms with respect to three objectives with associated metrics: target-behavior installation, general-capability preservation (i.e., parametric knowledge, chat quality), and output naturalness (i.e., CoT and activations). We re-visit two publicly released organism suites using this validation framework and show that (1) chat quality and CoT naturalness degrade substantially across training recipes, and (2) validation metrics predict how well interpretability methods recover the installed behavior, e.g., a logit lens readout covaries with an organism's general capabilities. We introduce a mult
    
[^77]: GAGR-Lab：评估空间-几何与解析函数的联合推理能力

    GAGR-Lab: Evaluating Joint Spatial-Geometric and Analytic Function Reasoning

    [https://arxiv.org/abs/2610.10201](https://arxiv.org/abs/2610.10201)

    该论文提出GAGR-Lab框架，通过笛卡尔游戏场景与Rust轨迹执行评估模型将空间配置转化为满足几何约束的解析函数的联合推理能力，试点实验表明当前视觉语言模型在此任务上尚无法命中目标。

    

    空间-几何与解析函数的联合推理要求将感知到的空间配置转化为符号函数，且该函数执行出的曲线需满足几何约束。我们提出了GAGR-Lab，这是一个通过笛卡尔游戏场景、显式函数语义以及权威的Rust轨迹执行来衡量该能力的框架。它区分了空间感知、度量定位、几何关系、函数解释、函数构建和约束合成等六个维度。我们定义了四个可配置的场景难度预设和一个前瞻性的24格诊断设计，同时仅报告实际评估的子集。对单个托管模型（Llama 3.2 11B Vision Instruct）使用两个API凭据作为执行副本进行的有限试点实验，产生了72个平衡游戏、432次尝试、429个有效的提供者响应，且无一命中目标；探索性的普通函数提示变体同样未能命中，而结构化的……（摘要在此处被截断）

    arXiv:2610.10201v1 Announce Type: cross  Abstract: Joint spatial-geometric and analytic function reasoning requires translating a perceived spatial configuration into a symbolic function whose executed curve satisfies geometric constraints. We present GAGR-Lab, a framework for measuring this capability through Cartesian game scenes, explicit function semantics, and authoritative Rust trajectory execution. It distinguishes spatial perception, metric grounding, geometric relations, function interpretation, function construction, and constrained synthesis. We specify four configurable scene-difficulty presets and a prospective 24-cell diagnostic design, while reporting only the subset actually evaluated. A bounded pilot of one hosted model (Llama 3.2 11B Vision Instruct) using two API credentials as execution replicas yields 72 balanced games with 432 attempts, 429 valid provider responses, and no target hits; exploratory ordinary-function prompt variants also fail to hit, while the struc
    
[^78]: 鲁棒的去中心化公平性审计

    Robust Decentralized Fairness Auditing

    [https://arxiv.org/abs/2610.10199](https://arxiv.org/abs/2610.10199)

    提出了Auditopus——一种无需中央服务器、以多轮方式协同进行、且能够在审计者可能合谋进行“公平洗白”时依然保持鲁棒的LLM去中心化公平性审计方法。

    

    新兴法规要求对大型语言模型（LLM）进行合规性审计，尤其是公平性方面的审计。此类黑盒审计通常假设存在单一审计者，且其能够获取大量具有代表性的查询集。在实际中，单个审计者很难获得这样的查询集，但多个审计者可以通过各自的查询集协同审计LLM，共同覆盖相关的人口群体。然而，依赖多个审计者引发了一个根本性的信任问题：他们可能代表LLM提供方行事，制造出具有误导性的公平表象，即“公平洗白”（fairwashing）。我们提出了Auditopus，一种用于鲁棒去中心化公平性审计的新方法。在Auditopus中，审计在无中央服务器的情况下分轮进行。在每一轮中，每个审计者向LLM发出固定数量的查询，并仅将其查询结果的累计统计向量发送给其他……（原文摘要在此处被截断）

    arXiv:2610.10199v1 Announce Type: new  Abstract: Emerging legislation requires large language models (LLMs) to be audited for compliance with regulatory standards, particularly fairness. Such black-box audits typically assume a single auditor with access to a large, representative set of queries. In practice, it can be difficult for an auditor to obtain such a query set, but multiple auditors can together cover the relevant demographic groups by auditing the LLM collaboratively with their individual query sets. However, relying on multiple auditors raises a fundamental trust problem, as they may act on behalf of the LLM provider to portray a misleading appearance of fairness, i.e., fairwashing. We propose Auditopus, a novel approach for robust decentralized fairness auditing. In Auditopus, auditing proceeds in rounds without a central server. In each round, every auditor issues a fixed number of queries to the LLM, and sends only cumulative statistics vectors of its query results to ot
    
[^79]: 基于均匀化与时间条件化因子分解神经似然估计的广泛适用的切换随机微分方程近似MCMC方法

    Broadly Applicable Approximate MCMC for Switching Stochastic Differential Equations Using Uniformization and Time-Conditioned Factorized Neural Likelihood Estimation

    [https://arxiv.org/abs/2610.10194](https://arxiv.org/abs/2610.10194)

    该论文提出了一种结合均匀化与时间条件化因子分解神经似然估计的近似MCMC采样器，突破了现有方法在噪声观测、状态维度、漂移形式和扩散项等方面的限制，实现了对切换随机微分方程广泛适用的贝叶斯推断。

    

    切换随机微分方程（SSDE）描述了参数随潜在状态过程（遵循连续时间马尔可夫链，CTMC）而切换的连续时间动力学。通过允许动力学在不同状态之间切换，SSDE能够表征异构系统行为，并已应用于众多领域。然而，SSDE的贝叶斯推断仍然十分困难，且现有的SSDE推断方法适用性有限，存在诸如无噪声观测、单变量状态、线性漂移或与状态无关的扩散等限制。在本研究中，我们提出了一种用于SSDE的近似马尔可夫链蒙特卡罗（MCMC）采样器，该方法结合了均匀化和因子分解神经似然估计（FNLE），后者是一种基于仿真的推断方法。均匀化为CTMC提供了精确的表示，但需要任意时间间隔上的SDE转移密度。我们通过训练时间条件化的模型来近似这些密度。（原文摘要在此处被截断）

    arXiv:2610.10194v1 Announce Type: cross  Abstract: Switching stochastic differential equations (SSDEs) describe continuous-time dynamics whose parameters switch according to a latent regime process that follows a continuous-time Markov chain (CTMC). By allowing dynamics to change between regimes, SSDEs represent heterogeneous system behavior and have been applied across diverse fields. However, Bayesian inference for SSDEs remains difficult, and existing SSDE inference methods have limited applicability, with restrictions such as noise-free observations, univariate states, linear drift, or state-independent diffusion. In this study, we propose an approximate Markov chain Monte Carlo sampler for SSDEs using uniformization and factorized neural likelihood estimation (FNLE), a simulation-based inference method. Uniformization provides an exact representation of the CTMC but requires SDE transition densities over arbitrary time intervals. We approximate these densities by training a time-c
    
[^80]: 一阶EDM预测器的普适局部误差与实际放大效应

    Universal Local Error and Realized Amplification for the First-Order EDM Predictor

    [https://arxiv.org/abs/2610.10190](https://arxiv.org/abs/2610.10190)

    该论文证明了一阶EDM扩散采样器的单步局部离散化误差具有与数据分布无关的普适二次上界，并通过在高噪声水平利用显式收缩准则、在低噪声水平引入可远小于最坏Lipschitz常数的实际放大效应，最终获得了O(e^{Λ_K}/K)的全局离散化误差保证。

    

    我们在2-Wasserstein距离下分析了Karras等人（2022年）提出的一阶确定性扩散采样器（称为EDM），将误差来源分为两部分：局部离散化误差及其被后续学习步骤放大的效应。我们证明局部误差存在一个普适上界：对于任何具有有限二阶矩的数据分布，单步离散化误差与步长呈二次方关系，且其显式常数不依赖于数据分布。相比之下，误差传播依赖于学习到的网络。在高噪声水平下，我们利用EDM的网络参数化推导出一个显式的收缩准则。在低噪声水平下，我们通过采样器所输运的分布上实际实现的放大来度量误差传播；这种实际放大可以远小于最坏情况下的Lipschitz常数。该分析得出了O(e^{Λ_K}/K)的全局离散化误差。

    arXiv:2610.10190v1 Announce Type: cross  Abstract: We analyze the first-order deterministic diffusion sampler of Karras et al. (2022), termed EDM, in 2-Wasserstein distance by separating two sources of error: local discretization error and its amplification by subsequent learned steps. We prove that local error admits a universal bound: for any data distribution with finite second moment, the one-step discretization error is quadratic in the step size, with an explicit constant that does not depend on the data distribution. Error propagation, in contrast, depends on the learned network. At high noise levels, we exploit the network parametrization of EDM to derive an explicit contraction criterion. At low noise levels, we measure propagation through the amplification realized on the distributions transported by the sampler; this realized amplification can be arbitrarily smaller than the worst-case Lipschitz constant. This analysis yields an $O(e^{\Lambda_K}/K)$ global discretization err
    
[^81]: 动力学朗之万遇见分裂吉布斯：基于扩散先验的成像逆问题加速后验采样

    Kinetic Langevin Meets Split Gibbs: Accelerated Posterior Sampling for Imaging Inverse Problems with Diffusion Priors

    [https://arxiv.org/abs/2610.10187](https://arxiv.org/abs/2610.10187)

    该论文提出RED-KLwSGS方法，将欠阻尼（动力学）朗之万扩散与分裂吉布斯采样框架结合，利用单次去噪得分驱动辅助变量更新，在与Langevin-within-SGS相同的每次迭代成本下实现成像逆问题后验采样的加速，并给出了强对数凹先验下连续和离散时间的非渐近Wasserstein-2收敛保证。

    

    分裂吉布斯采样（SGS）是贝叶斯成像逆问题中后验采样的一种流行框架。它通过引入辅助变量将高斯数据保真项与复杂先验解耦，使得数据变量可以精确更新，只有先验侧的条件分布难以采样。现有采样器以两种方式之一处理该条件分布：即插即用SGS在每次迭代中运行多步扩散去噪器，代价高昂且缺乏非渐近保证；Langevin-within-SGS采用廉价的过阻尼朗之万步，但需要大量迭代。我们提出RED-KLwSGS，该方法对数据变量保持精确的高斯更新，并使用由单次去噪得分驱动的欠阻尼（动力学）朗之万扩散来更新辅助变量，其每次迭代成本与Langevin-within-SGS相同。我们证明了对于强对数凹先验，该方法在连续和离散时间下均具有非渐近Wasserstein-2收敛性保证。

    arXiv:2610.10187v1 Announce Type: cross  Abstract: Split Gibbs sampling (SGS) is a popular framework for posterior sampling in Bayesian imaging inverse problems. It decouples a Gaussian data-fidelity term from a complex prior through an auxiliary variable, so the data variable is updated exactly and only the prior-side conditional is hard to sample. Existing samplers treat this conditional in one of two ways. Plug-and-play SGS runs a multi-step diffusion denoiser at every iteration, which is expensive and lacks non-asymptotic guarantees. Langevin-within-SGS takes cheap overdamped Langevin steps but needs many iterations. We propose RED-KLwSGS, which keeps the exact Gaussian update for the data variable and updates the auxiliary variable with underdamped (kinetic) Langevin diffusions driven by a one-shot denoising score, at the same per-iteration cost as Langevin-within-SGS. We prove non-asymptotic Wasserstein-2 convergence in continuous and discrete time for strongly log-concave priors
    
[^82]: 通过贝叶斯优化对贝叶斯优化算法进行预训练

    Pre-training of Bayesian Optimization Algorithm through Bayesian Optimization

    [https://arxiv.org/abs/2610.10186](https://arxiv.org/abs/2610.10186)

    该论文提出了一种通过在高斯过程样本路径上运行贝叶斯优化来最小化期望累积遗憾，从而利用另一个贝叶斯优化过程自动预训练贝叶斯优化算法参数的框架。

    

    贝叶斯优化（BO）作为昂贵黑箱优化问题的标准方法被广泛应用。然而，贝叶斯优化算法通常涉及必须事先指定的参数，其性能可能在很大程度上取决于这些参数的选择。我们提出了一个框架，利用从贝叶斯优化开始时已有信息推断出的高斯过程（GP）中抽取的样本路径来优化这些参数。我们使用累积遗憾作为贝叶斯优化算法的性能指标。通过在生成的样本路径上运行贝叶斯优化算法，我们可以获得给定参数配置下其期望累积遗憾的经验估计。优化该估计使我们能够识别出在当前可用信息条件下有望实现低累积遗憾的参数配置。由于该参数优化本身就是一个黑箱优化问题，我们采用另一个贝叶斯优化过程来求解它，我们将其称为……

    arXiv:2610.10186v1 Announce Type: new  Abstract: Bayesian optimization (BO) is widely used as a standard approach for expensive black-box optimization. However, BO algorithms often involve parameters that must be specified in advance, and their performance can strongly depend on these choices. We propose a framework for optimizing such parameters using sample paths drawn from a Gaussian process (GP) inferred from the information available at the start of BO. We use cumulative regret as the performance metric for a BO algorithm. By running the BO algorithm on the generated sample paths, we obtain an empirical estimate of its expected cumulative regret for a given parameter configuration. Optimizing this estimate allows us to identify parameter configurations that, given the currently available information, are expected to achieve low cumulative regret. Since this parameter optimization is itself a black-box optimization problem, we employ another BO procedure to solve it, which we refer
    
[^83]: 超越结果奖励：面向搜索智能体的检索信用构建与分配

    Beyond Outcome Rewards: Constructing and Assigning Retrieval Credit for Search Agents

    [https://arxiv.org/abs/2610.10179](https://arxiv.org/abs/2610.10179)

    该论文系统研究了从中间检索步骤提取学习信号的奖励塑形与信用分配策略，并提出将中间信号与最终结果奖励相结合的训练框架，显著提升了搜索智能体在多跳问题上的学习效率与总体性能。

    

    搜索智能体使大型语言模型（LLMs）能够迭代地检索和使用信息，以解决复杂的多跳问题。可验证奖励的强化学习为对此类智能体进行后训练提供了一种有前景的方法，但其对稀疏的、基于结果的监督的依赖会使信用分配变得困难，并限制学习效率。在本文中，我们系统地研究了中间监督如何改进搜索智能体的强化学习。我们研究了一系列奖励塑形和信用分配策略，这些策略能够从中间检索步骤中提供学习信号。基于这些洞察，我们开发了一个训练框架，将中间信号与最终结果奖励相结合，以改进对多步搜索轨迹的学习。在相同训练条件下跨多个基准的实验表明，搜索智能体的总体性能得到提升，并显示……

    arXiv:2610.10179v1 Announce Type: new  Abstract: Search agents enable Large Language Models (LLMs) to iteratively retrieve and use information for complex multi-hop questions. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach for post-training such agents, but its reliance on sparse, outcome-based supervision can make credit assignment difficult and limit learning efficiency. In this paper, we systematically investigate how intermediate supervision can improve reinforcement learning for search agents. We study a range of reward-shaping and credit-assignment strategies that provide learning signals from intermediate retrieval steps. Building on these insights, we develop a training framework that combines intermediate signals with final outcome rewards to improve learning from multi-step search trajectories. Experiments across multiple benchmarks under matched training conditions demonstrate improvements in aggregate search-agent performance and show that
    
[^84]: 一种用于约束多保真度多目标贝叶斯优化的统一信息论方法

    A Unified Information-Theoretic Approach to Constrained Multi-Fidelity Multi-Objective Bayesian Optimization

    [https://arxiv.org/abs/2610.10174](https://arxiv.org/abs/2610.10174)

    该论文提出一种统一的信息论方法，通过变分下界近似计算关于最高保真度可行帕累托前沿的信息增益，构建成本感知的采集函数，以联合处理多目标、约束和多保真度选择问题。

    

    贝叶斯优化通常涉及多个目标、约束条件和保真度级别。我们解决了在这一组合场景下如何联合选择评估位置和保真度，以识别最高保真度可行帕累托前沿的挑战。我们从统一的信息论视角出发，通过观测所提供的关于该前沿的信息增益来衡量查询的效用。由于这种互信息是难以直接计算的，我们使用对帕累托一致区域的欠截断和过截断近似的混合来推导出变分下界。多保真度代理模型将信息传播到任意保真度，从而产生一个成本感知的采集函数，无需为保真度选择或约束处理设计单独的启发式方法。在合成问题、基准问题和现实世界问题上的实验证明了该方法在多种目标、约束和保真度设置下的有效性。

    arXiv:2610.10174v1 Announce Type: new  Abstract: Bayesian optimization often involves multiple objectives, constraints, and fidelity levels. We address the challenge of jointly selecting where and at which fidelity to evaluate to identify the highest-fidelity feasible Pareto frontier in this combined setting. From a unified information-theoretic perspective, we measure query utility by the information gain about this frontier, provided by an observation. Since this mutual information is intractable, we derive a variational lower bound using a mixture of under- and over-truncated approximations to the Pareto-consistent region. Multi-fidelity surrogate models propagate the information to arbitrary fidelities, yielding a cost-aware acquisition function without separate heuristics for fidelity selection or constraint handling. Experiments on synthetic, benchmark, and real-world problems demonstrate effectiveness across diverse objective, constraint, and fidelity settings.
    
[^85]: 基于顺序白化的空间相关数据保形预测

    Conformal Prediction for Spatially Dependent Data via Sequential Whitening

    [https://arxiv.org/abs/2610.10168](https://arxiv.org/abs/2610.10168)

    该论文提出一种通过对校准残差进行顺序条件化（顺序白化）的保形预测方法，解决了空间相关数据下可交换性假设失效、且仅由校准残差可预测的空间变异残留导致区间效率下降的问题，在正确的工作协方差与椭圆残差分布下实现精确的有限样本覆盖率，并可借助最近邻近似扩展到大型网络。

    

    分割保形预测利用留出的（校准）数据上的预测误差来确定预测区间的宽度。当这些误差与目标位置的误差可交换时，它能保证无分布的有限样本覆盖率。然而在空间依赖和非随机采样几何结构下，这一假设可能失效。现有的空间方法使用拟合残差从校准误差和目标误差中去除空间变异中可预测的部分。但是，只有校准残差能够预测的那部分空间变异仍然保留在目标误差和校准误差之中，从而降低了预测区间的效率和稳定性。我们通过额外地对校准残差进行顺序条件化来解决这一问题，该方法借助最近邻近似可扩展到大规模网络。在工作协方差设定正确且残差服从椭圆分布定律的条件下，所得区间具有精确的有限样本覆盖率。

    arXiv:2610.10168v1 Announce Type: cross  Abstract: Split conformal prediction uses prediction errors on held-out (calibration) data to determine how wide the prediction intervals should be. It guarantees distribution-free finite-sample coverage when these errors and the error at the target site are exchangeable. This assumption may fail under spatial dependence and nonrandom sampling geometry. Existing spatial methods use fitting residuals to remove the predictable part of spatial variation from calibration and target errors. However, the spatial variation that only the calibration residuals can predict remains in both the target and calibration errors, reducing the efficiency and stability of the interval. We address this by additionally conditioning on the calibration residuals sequentially, which scales to large networks through nearest-neighbour approximations. Under a correct working covariance and an elliptical residual law, the resulting interval has exact finite-sample coverage
    
[^86]: 弱信号下的策略学习

    Policy Learning with Weak Signals

    [https://arxiv.org/abs/2610.10167](https://arxiv.org/abs/2610.10167)

    该论文证明在低信噪比的大规模数字实验中最优策略一般不可学习，但当处理效应平滑变化时，基于线性平滑器的极小极大自适应策略可实现趋近于零的福利遗憾，并在Netflix真实实验中验证了个性化平滑策略的优越性。

    

    数字实验中的策略学习面临三大挑战：低信噪比、丰富的协变量空间以及庞大的数据量。我们通过将从越来越细的协变量划分中得到的处理效应估计建模为具有有界信噪比的高斯观测，来形式化这一情境。我们证明，一般来说，最优处理策略在此设定下是不可学习的，即使学习最优策略值，其收敛速度也慢得不切实际。然而，当处理效应平滑变化时，我们基于线性平滑器推导出极小极大自适应策略，该策略能够实现趋近于零的福利遗憾。我们通过将该框架应用于Netflix的大规模真实实验展示了其应用价值，结果表明即使在这一具有挑战性的实证环境中，个性化线性平滑策略仍可优于非个性化策略。

    arXiv:2610.10167v1 Announce Type: cross  Abstract: Policy learning in digital experimentation faces three challenges: weak signal-to-noise ratios, rich covariate spaces, and massive data volumes. We formalize this regime by modeling treatment-effect estimates from increasingly fine covariate partitions as Gaussian observations with bounded signal-to-noise ratios. We establish that, in general, the optimal treatment policy is not learnable in this setting. Even learning the optimal policy value suffers from impractically slow rates. However, when treatment effects vary smoothly, we derive minimax-adaptive policies based on linear smoothers that achieve vanishing welfare regret. We demonstrate the practical value of our framework by applying it to large-scale real-world experiments at Netflix, showing that personalized linear-smoothing policies can dominate unpersonalized policies even in this challenging empirical setting.
    
[^87]: 人工智能在ARPES（角分辨光电子能谱）工作流程中的进展与展望

    Progress and Prospect of AI in ARPES Workflow

    [https://arxiv.org/abs/2610.10140](https://arxiv.org/abs/2610.10140)

    本文系统综述了机器学习方法在ARPES（角分辨光电子能谱）完整工作流程（从样品制备、数据采集到数据分析与理论对比）中的应用现状、优势与局限性，是该领域首篇全面评述机器学习可靠性与应用前景的综述文章。

    

    人工智能（AI）正日益成为实验科学中实用的工具，其中包括角分辨光电子能谱（ARPES），该技术通常会产生大型的、多维的电子结构数据集。人工智能和机器学习（ML）的最新进展为整个ARPES工作流程开辟了新的机遇，涵盖从自动化样品制备和实时数据采集，到实验后数据分析以及与理论计算的对比等环节。尽管取得了这些进展，目前仍缺乏一篇对机器学习应用、其能力及可靠性在ARPES工作流程各阶段进行系统评估的全面综述。在这篇综述中，我们首先介绍与凝聚态物理和材料科学领域的实验工作者最相关的机器学习方法，然后沿着ARPES工作流程，综述每个步骤中现有的机器学习应用，并讨论它们的优势与局限性。

    arXiv:2610.10140v1 Announce Type: cross  Abstract: Artificial intelligence (AI) is becoming an increasingly useful tool across the experimental sciences, including angle-resolved photoemission spectroscopy (ARPES), which routinely produces large, multidimensional datasets of electronic structure. Recent advances in AI and machine learning (ML) have opened new opportunities across the entire ARPES workflow, from automated sample preparation and real-time data acquisition to post-experiment data analysis and comparison with theoretical calculations. Despite this progress, a comprehensive review of ML applications, their capabilities, and reliability across the different stages of ARPES workflow is still lacking. In this review, we first introduce ML methods that are most relevant to experimentalists working in condensed matter physics and materials science. We then follow the ARPES workflow, reviewing existing ML applications at each step and discussing their advantages, limitations and 
    
[^88]: 基于黑盒向量搜索的注意力机制

    Attention via Black-Box Vector Search

    [https://arxiv.org/abs/2610.10135](https://arxiv.org/abs/2610.10135)

    本文通过优先抽样框架统一了基于MIPS的稀疏注意力方法，证明了单个索引下Θ(√n/ε)个检索键的紧致界、多索引下O(log n + 1/ε²)的近最优算法，并展示了通过增强键和查询可突破下界。

    

    稀疏注意力机制通过使用键的一个小子集来估计对n个token的注意力。许多现有方法使用最大内积搜索（MIPS）来检索权重最大的键，这引出了如下问题：在仅有MIPS预言机的黑盒访问权限的情况下，需要检索多少个键才能输出ε精度的注意力估计？我们通过优先抽样的框架统一了先前的方法来回答这个问题。使用单个MIPS索引时，我们证明检索Θ(√n/ε)个键既是充分的也是必要的；使用Θ(log n)个索引时，我们给出了一个仅检索O(log n + 1/ε²)个键的算法，并证明其接近最优。更一般地，我们设计了能在MIPS索引数量与检索键数量之间建立平滑权衡的算法。随后我们证明，如果允许对键和查询进行增强，则可以绕过上述下界（摘要在此截断）。

    arXiv:2610.10135v1 Announce Type: cross  Abstract: Sparse attention mechanisms estimate attention over $n$ tokens using a small subset of keys. Many existing approaches use maximum inner product search (MIPS) to retrieve the heaviest keys, which motivates the following question: given black-box access to a MIPS oracle, how many keys must be retrieved to output an $\varepsilon$-accurate attention estimate?   We answer this question by unifying prior approaches through the framework of priority sampling. With a single MIPS index, we show that $\Theta(\sqrt{n}/\varepsilon)$ retrieved keys are both sufficient and necessary. With $\Theta(\log n)$ indices, we give an algorithm that retrieves only $O(\log n+1/\varepsilon^2)$ keys and prove that this is near-optimal. More generally, we design algorithms that establish a smooth tradeoff between the number of MIPS indices and number of retrieved keys. We then show that if we allow augmentation of keys and queries, we can bypass the above lower b
    
[^89]: 基于赢家反馈的m集对抗性多臂老虎机

    m-Set Adversarial Bandits with Winner Feedback

    [https://arxiv.org/abs/2610.10128](https://arxiv.org/abs/2610.10128)

    本文研究了不同效用和反馈模型下m集对抗性多臂老虎机的遗憾界，其主要技术贡献是遗憾值的信息论下界，揭示了环境设置中的细微变化会对学习速率产生巨大影响。

    

    我们针对不同效用函数（赢家奖励或奖励总和）和不同反馈模型（赢家索引、赢家奖励、奖励总和及其组合），给出了 $m$ 集对抗性多臂老虎机遗憾值的上界和下界。通过与组合老虎机和 MNL 老虎机的标准界进行比较，我们的结果揭示了环境设置中的细微变化如何对学习速率产生巨大影响。我们的主要技术贡献在于遗憾值的信息论下界。在合成数据上的实验证实了我们的理论分析。

    arXiv:2610.10128v1 Announce Type: new  Abstract: We show upper and lower bounds on the regret of $m$-set adversarial bandits for different utilities (winner reward or sum of rewards) and feedback models (winner index, winner reward, sum of rewards, and their combinations). By comparing to standard bounds for combinatorial and MNL bandits, our results reveal how subtle changes in the setting can have a dramatic impact on the learning rates. Our main technical contributions are the information-theoretic lower bounds on the regret. Experiments on synthetic data confirm our theoretical analyses.
    
[^90]: 生成式推荐中错失目标的训练：将监督与概率竞争相分离

    Training with Missed Targets in Generative Recommendation: Separating Supervision from Probability Competition

    [https://arxiv.org/abs/2610.10124](https://arxiv.org/abs/2610.10124)

    该论文通过构建三个控制变量的匹配损失函数，首次将“为错失目标添加监督”与“两组目标的概率竞争”这两个效应解耦，发现概率竞争会损害返回物品的排序质量，从而解释了附加错失目标这一训练策略效果难以归因的原因。

    

    生成式推荐器返回一个有限的候选集，并且在重排序之前可能会遗漏已观测到的目标。一种训练策略将这些被遗漏的目标附加到重排序器的训练列表中，尽管在推理时仍然只对原始候选进行排序。这一操作同时改变了已检索目标的权重、为附加目标增加了监督，并使两组目标争夺概率。因此，简单的附加/不附加对比实验无法解释返回物品排序的变化。我们构建了三个匹配的损失函数，在保持已检索目标权重不变的前提下，分别单独引入附加目标监督和组间竞争。其中中间的损失函数在两组内部进行训练但分别对它们归一化，从而防止仅用于训练的目标与推理候选产生竞争。在已发布的OneRec模型和本地训练的Amazon生成器上的实验表明，这种概率竞争可能会损害返回物品的排序。在四个预先设定的（实验中……摘要在此处截断）

    arXiv:2610.10124v1 Announce Type: cross  Abstract: Generative recommenders return a limited candidate set and may omit observed targets before reranking. A training strategy appends these missed targets to reranker training lists, although inference still ranks only original candidates. This operation simultaneously changes retrieved-target weight, adds supervision over appended targets, and makes the two groups compete for probability. An append/no-append comparison therefore cannot explain changes in returned-item rankings. We construct three matched losses that hold retrieved-target weight fixed while introducing appended-target supervision and group competition separately. The intermediate loss trains within both groups but normalizes them separately, preventing training-only targets from competing with inference candidates. Experiments with a released OneRec model and locally trained Amazon generators show that this competition can harm returned-item ranking. In four prespecified 
    
[^91]: 高斯过程的设计能检验出什么

    What Can a Gaussian Process Design Test

    [https://arxiv.org/abs/2610.10122](https://arxiv.org/abs/2610.10122)

    该论文揭示仅凭高斯过程的试验设计（无需观测任何响应）就能判断模型假设能否被数据证伪，并将有限特征GP的可检验关系精确刻画为核矩阵的零空间，同时借助Gale对偶性给出“电路”的几何解释，并对一般核函数定义了概率意义上的软性检验关系。

    

    一个高斯过程（GP）模型之所以能与数据相符，可能有两种原因：其假设是正确的，或者所选取的输入根本无法证明这些假设是错误的。这一区别可以在观测到任何响应之前，仅通过试验设计本身来检验。每个模型都隐含着其无噪声响应在所选输入处必须满足的关系，例如中间点的取值必须落在其两个相邻点连成的直线上。对于由有限个特征构建的高斯过程，这些关系恰好就是核矩阵的零空间。Gale对偶性为这些关系提供了几何解释：每个观测对应一个向量，而能够暴露出模型错误的最小观测组即为“电路（circuits）”。对于其他核函数，这些关系变为软性的：响应模式在先验之下可能只是不太可能，而非在代数上不可能。随后，一个标准检验方法结合了两类证据：结构性证据来自被违反的关系，并且会无限增长……

    arXiv:2610.10122v1 Announce Type: new  Abstract: A Gaussian process (GP) model can agree with the data for two reasons: its assumptions are right, or the chosen inputs could never have shown that they are wrong. The distinction can be checked from the design before any responses are observed. Every model implies relations that its noiseless responses must satisfy at the chosen inputs, such as the middle value lies on the line through its two neighbours. For GPs built from finitely many features, these relations are exactly the null space of the kernel matrix. Gale duality gives them a geometric interpretation, in which each observation has a vector and the smallest groups of observations that can expose an error are the circuits. For other kernels the relations become soft: response patterns may be improbable under the prior rather than algebraically impossible. A standard test then combines two kinds of evidence. Structural evidence comes from a violated relation and grows without lim
    
[^92]: YANchor-4B：以 O(N) 时间复杂度和 O(1) 内存实现高效的长程推理

    YANchor-4B: Effective Long-Horizon Reasoning in O(N) Time with O(1) Memory

    [https://arxiv.org/abs/2610.10118](https://arxiv.org/abs/2610.10118)

    YANchor-4B 通过将关键记忆保存为可检索的锚点，以 O(N) 时间和 O(1) 内存的成本实现了高效的长程推理，在数学基准上大幅超越同类模型并具有数倍于 Transformer 的生成吞吐量。

    

    长程推理需要在可管理的生成成本下访问早期信息。全历史注意力机制会带来不断增长的存储与计算开销，而循环压缩则可能丢失精确细节。因此，我们提出了 YANchor-4B，这是一种通用循环模型，它将关键记忆保存为“锚点”，以便在后续推理中进行检索。除了 O(N) 时间生成和 O(1) 内存之外，YANchor 还通过其多维记忆机制实现了高效的长程推理。例如，在具有挑战性的数学问题上，它在 AIME 2024–2026 上取得了 82.93% 的平均 pass@1，在 HMMT 上取得了 63.64%，大幅超越线性时间、常数状态的同类模型，包括规模更大的模型。在 H100 上，其批量长文本生成吞吐量比 Transformer 和混合基线高出数倍。此外，跨数十个基准的评估证明了 YANchor 在通用能力方面的优越性。

    arXiv:2610.10118v1 Announce Type: cross  Abstract: Long-horizon reasoning demands access to earlier information at a manageable generation cost. Full-history attention incurs growing storage and computation, while recurrent compression can lose precise details. Therefore, we present YANchor-4B, a general-purpose recurrent model that preserves crucial memory as ANchors for retrieval during subsequent reasoning. Beyond $O(N)$-time generation and $O(1)$ memory, YANchor enables effective long-horizon reasoning through its multidimensional memory mechanism. For example, on challenging math problems, it achieves 82.93% mean pass@1 on AIME 2024--2026 and 63.64% on HMMT, substantially outperforming linear-time, constant-state counterparts, including larger models. It also delivers several-fold higher batched long-generation throughput than Transformer and hybrid baselines on H100. Furthermore, evaluations across dozens of benchmarks demonstrate YANchor's superiority in general-purpose capabili
    
[^93]: CAFE+FNO：通过乘性特征组合实现傅里叶核生成

    CAFE+FNO: Fourier Kernel Generation via Multiplicative Feature Composition

    [https://arxiv.org/abs/2610.10105](https://arxiv.org/abs/2610.10105)

    本文提出CAFE+FNO，通过并行仿射分支与Hadamard乘积显式组合傅里叶-切比雪夫特征来生成傅里叶核，从而增强FNO对高频变化的学习能力。

    

    傅里叶神经算子（FNO）通过傅里叶空间中的核参数化来学习偏微分方程（PDE）的解算子，但频率截断可能限制对高频变化的学习。AM-FNO和SirenFNO使用共享网络从谱坐标为所有网格模式生成核，这使得坐标编码和生成器设计变得至关重要。近期关于隐式神经表示（INR）的研究提出通过显式特征组合来构建频率交互，而不是依赖后续的多层感知机（MLP）隐式地形成这些交互。基于这一思路，我们提出了CAFE+FNO，将内容感知频率编码+（CAFE+）融入傅里叶核生成之中。CAFE+通过并行的仿射分支和Hadamard乘积将傅里叶与切比雪夫特征相结合，在两个特征族内部以及二者之间形成交互。一个核MLP将由此得到的表示（摘要在此处截断）……

    arXiv:2610.10105v1 Announce Type: new  Abstract: The Fourier Neural Operator (FNO) learns solution operators of partial differential equations (PDEs) through Fourier-space kernel parameterization, but frequency truncation can limit the learning of high-frequency variations. AM-FNO and SirenFNO generate kernels for all grid modes from spectral coordinates using shared networks, making coordinate encoding and generator design important. Recent work on implicit neural representations (INRs) has proposed constructing frequency interactions through explicit feature composition rather than relying on subsequent MLPs to form them implicitly. Building on this approach, we propose CAFE+FNO, which incorporates Content-Aware Frequency Encoding+ (CAFE+) into Fourier kernel generation. CAFE+ combines Fourier--Chebyshev features through parallel affine branches and a Hadamard product, forming interactions within and across the two feature families. A kernel MLP maps the resulting representation of e
    
[^94]: ExperienceIndex：基于工件（Artifact）的记忆

    ExperienceIndex: Artifact-Grounded Memory

    [https://arxiv.org/abs/2610.10091](https://arxiv.org/abs/2610.10091)

    提出了 ExperienceIndex，一种让 AI 智能体基于先前推理轨迹捕获并复用工件特定经验知识的新型记忆层，可提升知识密集型任务的答案质量并降低在线成本。

    

    知识密集型任务需要通过推理共享的工件语料库（例如法院判例或科学文献）来回答大量问题。当人类与这些语料库交互时，会自然地积累关于工件的经验知识，从而能够快速识别出每个新任务的完整相关工件集合。然而，现有的人工智能智能体缺乏合适的记忆解决方案来构建或复用这种以工件为基础的经验，导致答案质量较低且在线成本较高。现有的记忆解决方案虽然能从先前的任务求解轨迹中提取和复用信息，但它们主要关注用户偏好、事实属性或抽象推理模式，而非持久的、特定于工件的知识。我们提出了 ExperienceIndex，这是一种面向 AI 智能体的新型经验层，它基于先前的推理轨迹来捕获和复用关于工件的知识。ExperienceIndex 存储两种互补的……

    arXiv:2610.10091v1 Announce Type: new  Abstract: Knowledge-intensive tasks require answering many questions by reasoning about a shared corpus of artifacts (e.g., court cases, or scientific literature). As humans interact with these corpora, they naturally accumulate experiential knowledge about artifacts, enabling them to quickly identify the complete set of relevant artifacts for each new task. However, existing AI agents lack appropriate memory solutions to build or reuse such artifact-grounded experience, leading to lower answer quality and higher online cost. Existing memory solutions extract and reuse information from prior task-solving traces, but they primarily focus on user preferences, factual attributes, or abstract reasoning patterns rather than persistent artifact-specific knowledge. We introduce ExperienceIndex, a novel experience layer for AI agents that captures and reuses knowledge about artifacts based on prior reasoning traces. ExperienceIndex stores two complementar
    
[^95]: 基于支撑集保持蒸馏的多智能体协调

    Multi-Agent Coordination via Support-Preserving Distillation

    [https://arxiv.org/abs/2610.10087](https://arxiv.org/abs/2610.10087)

    提出MoSDOT方法，利用条件半离散最优传输将噪声样本确定性分配到有限模式支撑集上，解决基于流的多智能体教师在蒸馏过程中模式冲突误差传播给学生模型的问题。

    

    离线多智能体强化学习（MARL）日益依赖生成式策略来建模多模态联合行为，通常是在集中式训练分散式执行（CTDE）框架下，将集中式教师蒸馏为分散式单步执行者。我们识别出教师训练阶段的一种失败模式：标准的基于流的教师将噪声与回放目标独立配对，导致相邻的噪声样本可能被路由到相互冲突的协调模式。教师随后会在有效模式之间产生样本，而由于蒸馏损失将每个局部执行者回归到给定局部输入下教师输出的条件均值，这一误差不会被吸收，而是传播到学生模型。为了消除这一教师侧的伪影，我们提出模式支撑半离散最优传输（MoSDOT），它将多模态回放数据总结为具有预设容量的有限模式支撑集，并在教师训练之前使用条件半离散最优传输将每个噪声样本分配到单一模式……

    arXiv:2610.10087v1 Announce Type: new  Abstract: Offline MARL increasingly relies on generative policies to model multimodal joint behavior, typically by distilling a centralized teacher into decentralized one-step actors under the CTDE. We identify a failure mode at the teacher training stage: standard flow-based teachers pair noise with replay targets independently, so nearby noise samples can be routed toward conflicting coordination modes. The teacher then produces samples between valid modes, and because the distillation loss regresses each local actor onto the conditional mean of the teacher's output given local input, this error is not absorbed but propagated to the student. To remove this teacher-side artifact, we propose Mode-Support Semi-Discrete Optimal Transport (MoSDOT), which summarizes multimodal replay into a finite mode support with prescribed capacities and uses conditional semi-discrete optimal transport to assign each noise sample to a single mode before teacher tra
    
[^96]: 激活感知权重张量化：一种用于张量网络大语言模型压缩的校准期预处理器

    Activation-Aware Weight Tensorization: A Calibration-Time Preconditioner for Tensor-Network LLM Compression

    [https://arxiv.org/abs/2610.10085](https://arxiv.org/abs/2610.10085)

    AWT是一种免训练的校准期预处理方法，通过激活感知的对角缩放对权重矩阵进行预处理后再执行TT/TTN分解，在2-6倍压缩率下显著缩小了张量网络压缩后的大语言模型与稠密基线之间的困惑度差距。

    

    后训练张量网络压缩将Transformer线性层替换为张量列车（TT）或树张量网络（TTN）算子，但标准分解方法最小化的是权重空间的Frobenius误差，而非在该层激活分布下的函数误差。我们提出激活感知权重张量化，这是一种免训练的校准包装器，它在运行不变的TT/TTN求解器之前，先用由激活导出的对角缩放对每个权重矩阵进行预处理，并以仅在输入侧进行逐元素重缩放的方式部署结果。在Llama 3.1 8B、Ministral 8B和Qwen2.5 7B上，AWT在2-6倍压缩率下持续改进了原始TT/TTN张量化：在单算子替换下，AWT在三个模型系列和2-6倍压缩设置下，将WikiText困惑度相对于稠密基线的差距缩小了12-35%；而在多算子Llama后缀替换下，其在注意力组方面缩小了27-60%（摘要在此处截断）。

    arXiv:2610.10085v1 Announce Type: new  Abstract: Post-training tensor-network compression replaces Transformer linear layers with Tensor Train (TT) or Tree Tensor Network (TTN) operators, but standard decompositions minimize weight-space Frobenius error rather than functional error under the layer's activation distribution. We propose Activation-aware Weight Tensorization (AWT), a training-free calibration wrapper that preconditions each weight matrix with a diagonal activation-derived scale before an unchanged TT/TTN solver and deploys the result with only an input-side elementwise rescaling. Across Llama 3.1 8B, Ministral 8B, and Qwen2.5 7B, AWT consistently improves vanilla TT/TTN tensorization at 2-6 times compression: under single-operator replacement, AWT closes 12-35% of the WikiText perplexity gap to the dense baseline across the three model families and 2-6 times compression settings; while under multi-operator Llama suffix replacement it closes 27-60% across attention-group a
    
[^97]: 具有RBF核的高斯过程极大似然估计的精确渐近理论

    Sharp Asymptotic Theory of Maximum Likelihood Estimation for Gaussian Processes with an RBF Kernel

    [https://arxiv.org/abs/2610.10080](https://arxiv.org/abs/2610.10080)

    本文针对基于RBF核的高斯过程，在固定域渐近框架下建立了极大似然估计的精确渐近理论，攻克了密集采样强相关性与协方差矩阵非线性依赖带来的长期理论难题。

    

    高斯过程（GP）广泛应用于机器学习、空间统计、时间序列分析、优化、贝叶斯统计以及各类科学应用中。高斯过程模型的核心组成部分是其核函数，通常通过参数族来指定。其中最广泛使用的选择之一是径向基函数（RBF）核，也称为平方指数核或高斯核，这得益于其形式简单、光滑且灵活的特性。在实践中，核参数通常通过极大似然估计（MLE）来估计，这也是标准高斯过程软件所采用的标准方法。尽管应用如此广泛，即便对于RBF核，极大似然估计量在固定域渐近框架下的渐近行为仍然鲜为人知。其主要困难源于密集采样观测之间日益增强的相关性，以及协方差矩阵对核参数的非线性依赖。在本文中……

    arXiv:2610.10080v1 Announce Type: cross  Abstract: Gaussian processes (GPs) are widely used across machine learning, spatial statistics, time-series analysis, optimization, Bayesian statistics, and scientific applications. A central component of a GP model is its kernel, which is typically specified through a parametric family. Among the most widely used choices is the radial basis function (RBF), also known as the squared exponential or Gaussian kernel, owing to its simple form, smoothness, and flexibility. In practice, the kernel parameters are routinely estimated by the maximum likelihood estimators (MLEs), as implemented by standard GP software. Despite this widespread use, the asymptotic behavior of the MLEs remains poorly understood under fixed-domain asymptotics, even for the RBF kernel. The main difficulty arises from the increasingly strong dependence among densely sampled observations and the nonlinear dependence of the covariance matrix on the kernel parameters. In this pape
    
[^98]: 通过结果条件重校准实现针对关注事件的校准概率预测

    Towards Calibrated Probabilistic Forecasts for Events of Interest via Outcome-Conditional Recalibration

    [https://arxiv.org/abs/2610.10076](https://arxiv.org/abs/2610.10076)

    本文提出了一种简单易实现的事后重校准方法——结果条件重校准，能够在用户定义的结果空间区域（如极端事件）上对概率预测进行重校准，从而确保决策者最关注的事件也能获得校准良好的预测。

    

    校准是概率预测能够有效支持决策制定的一项基本要求。尽管最先进的预测方法往往产生校准不佳的预测分布，但目前已有多种事后重校准方案被提出以生成校准良好的预测。然而，流行的重校准方案可能掩盖结果空间特定区域中存在的校准不佳问题。由于特定结果（例如极端事件）往往对决策制定最为重要，因此当评估仅限于这些结果时，概率预测应当是校准良好的。为此，本文提出了结果条件重校准，这是一种事后方法，可在用户自定义的结果空间区域上重新校准概率预测。该方法简单、易于实现，并且可应用于任意预测分布。其工作原理是将Kuleshov等人（2018）提出的分位数重校准方法应用于……（摘要在此处截断）

    arXiv:2610.10076v1 Announce Type: cross  Abstract: Calibration is an essential requirement for probabilistic predictions to be useful for decision making. While state-of-the-art prediction methods often yield miscalibrated predictive distributions, several post-hoc recalibration schemes have been proposed to generate calibrated predictions. However, popular recalibration schemes can conceal miscalibration in specific regions of the outcome space. Since particular outcomes, such as extreme events, often matter most for decision making, probabilistic predictions should be calibrated when evaluation is restricted to these outcomes. Hence, in this paper, we introduce outcome-conditional recalibration, a post-hoc method to recalibrate probabilistic predictions on user-defined regions of the outcome space. The method is simple, easy to implement, and can be applied to arbitrary predictive distributions. It works by applying the quantile recalibration approach of Kuleshov et al. (2018) to for
    
[^99]: 基于表格基础模型的高效可证明隐私保护分类

    Efficient Provably Private Classification with a Tabular Foundation Model

    [https://arxiv.org/abs/2610.10068](https://arxiv.org/abs/2610.10068)

    PrivTab是一种将差分隐私机制内嵌于模型架构的表格基础模型，通过上下文学习将敏感数据行转换为可证明隐私的紧凑摘要，实现了高效且具有正式隐私保证的分类。

    

    表格数据是医学、金融、政府和科学等领域中预测与决策的基础，但其中往往包含敏感的个人层面信息，因此需要在保护隐私的同时实现准确的预测。传统的隐私保护学习能够提供正式的隐私保证，但需要对特定数据集进行缓慢的优化，在强隐私约束下会造成显著的效用损失，且往往难以正确应用。表格基础模型能够快速适应新数据集，但现有模型缺乏正式的隐私保证，并且极易受到成员推断攻击，限制了其在敏感数据上的应用。本文提出了PrivTab，一个易于使用的表格基础模型，用于差分隐私分类，它将隐私机制直接嵌入到模型架构之中。PrivTab在模拟数据集上进行预训练，利用上下文学习将敏感数据行转换为紧凑的、可证明隐私的摘要，从而有效（原文摘要在此处截断）

    arXiv:2610.10068v1 Announce Type: new  Abstract: Tabular data underpin prediction and decision-making in medicine, finance, government and science, but often contain sensitive individual-level information, creating a need for accurate prediction while preserving privacy. Traditional private learning provides formal privacy guarantees, but requires slow dataset-specific optimisation, suffers substantial utility loss under strong privacy, and is often difficult to apply correctly. Tabular foundation models adapt rapidly to new datasets, but existing models lack formal privacy guarantees, and are highly vulnerable to membership-inference attacks, limiting their use on sensitive data. Here we introduce PrivTab, an easy to use tabular foundation model for differentially private classification that embeds a privacy mechanism within its architecture. Pretrained on simulated datasets, PrivTab uses in-context learning to transform sensitive rows into compact, provably private summaries---effect
    
[^100]: TRACK：基于遥测数据的模拟赛车竞速分析与教练工具包

    TRACK: Telemetry-Based Racing Analysis and Coaching Kit in Sim Racing Games

    [https://arxiv.org/abs/2610.10061](https://arxiv.org/abs/2610.10061)

    该论文提出了TRACK框架，通过将驾驶环节映射到速度、制动、策略与稳定性构成的四维行为空间并进行无监督聚类来刻画车手驾驶行为画像，同时采用零假设基线校准来严谨评估聚类结果的有效性，为模拟赛车提供数据分析与教练指导能力。

    

    本文提出了TRACK（基于遥测数据的赛车分析与教练工具包），这是一个用于分析模拟赛车驾驶表现并刻画车手在方向盘后行为特征的框架。我们同时报告了该框架及其局限性：我们将每个聚类结果与零假设基线进行校准对比，当结果未能与随机水平区分开时，我们会如实说明。我们不局限于给车手打分或将他们归入预设标签，而是将每次记录的驾驶环节表示为四维行为空间（速度、制动、策略和稳定性）中的紧凑几何结构，并使用无监督聚类根据相似性对这些行为特征指纹进行分组。我们在公开的Assetto Corsa Gym（ACGym）数据集上持续开发并完善了这一框架。我们的研究表明，弯道类型在一个并未用于定义它们的行为维度上存在差异；研究还表明，当赛车发生改变时，只有速度……

    arXiv:2610.10061v1 Announce Type: cross  Abstract: This paper presents TRACK (Telemetry-Based Racing Analysis and Coaching Kit), which is a framework for analyzing driving performance in sim racing and profiling how individual drivers behave behind the wheel. We report this framework together with its limitations: we calibrate each clustering result against a null, and when one does not separate from chance, we say so. Instead of restricting ourselves to scoring drivers or sorting them into preset labels, we represent each recording session as a compact geometry in a four-dimensional behavioral space (speed, braking, strategy, and consistency), and we group these fingerprints by their similarity using unsupervised clustering. Over time, we have developed and refined this framework on the open Assetto Corsa Gym (ACGym) dataset. Our study suggests that corner types differ along a behavioral dimension that was not used to define them. It also suggests that when the car changes, only speed
    
[^101]: WxFM-XL：将单变量基础模型适配至多站点天气预报

    WxFM-XL: Adapting Univariate Foundation Models to Multi-Station Weather Forecasting

    [https://arxiv.org/abs/2610.10057](https://arxiv.org/abs/2610.10057)

    提出WxFM-XL模型，通过引入跨站点误差相关性先验图与自适应动态融合机制，将单变量时间序列基础模型成功适配到多站点天气预报任务中。

    

    随着单变量时间序列基础模型（如 Sundial、Timer）的兴起，已有初步尝试将其扩展到多变量场景。然而，这些模型主要聚焦于变量之间的相关性建模。当将其应用于多站点天气预报时，有两个重要因素往往被忽视：（1）站点的空间信息；（2）不同站点相对于基础模型的不同误差先验。在本文中，我们提出了 WxFM-XL，一个将单变量时间序列基础模型适配到多站点天气预报的模型。WxFM-XL 引入了一种跨站点误差相关性先验图，以捕捉各站点相对于基础模型的误差先验。在此基础上，我们进一步提出了一种动态融合机制，能够自适应地将空间相关性图与误差相关性先验图进行融合。在多个数据集上的实验表明，我们的方法……

    arXiv:2610.10057v1 Announce Type: new  Abstract: With the rise of univariate time series foundation models (e.g., Sundial, Timer), initial efforts have been made to extend them to multivariate settings. However, these models mainly focus on modeling correlations among variables. When they are applied to multi-station weather forecasting, two important factors are often overlooked: (1) the spatial information of stations, and (2) different error priors of different stations relative to the foundation model. In this paper, we propose WxFM-XL, a model for adapting univariate time series foundation models to multi-station weather forecasting. WxFM-XL introduces a cross-station error correlation prior graph to capture stationwise error priors with respect to the foundation model. Building on this, we further propose a dynamic fusion mechanism that adaptively integrates a spatial correlation graph with the error correlation prior graph. Experiments on multiple datasets demonstrate that our m
    
[^102]: 基于Koopman算子与退出时间最优控制的转移路径采样

    Transition Path Sampling Using Koopman Operators and Exit-Time Optimal Control

    [https://arxiv.org/abs/2610.10054](https://arxiv.org/abs/2610.10054)

    提出一种基于Koopman算子的转移路径采样新方法，利用其线性性质在无需转移路径数据的情况下识别亚稳态集合并估计committor函数，同时将TPS表述为退出时间最优随机控制问题，从而解决了现有神经网络方法的计算开销与性能保证问题。

    

    在亚稳态之间采样转移路径是动力系统理论（尤其是分子动力学）中的一个核心问题。其关键挑战在于分隔各亚稳态的高自由能势垒，使得态间转移极为罕见。近期基于机器学习的方法将转移路径采样（TPS）表述为固定时间范围内的最优随机控制（OSC）问题，并通过在仿真循环中训练的神经网络对漂移偏置进行参数化，这需要反复进行有偏的模拟推演。为解决此类模型的计算开销与性能保证问题，我们提出了一种基于Koopman算子的新方法。由于Koopman算子是线性的，其主导特征函数能够揭示亚稳态集合，并在无需任何转移路径信息的情况下给出committor函数的估计。此外，我们将TPS表述为直至退出时间的最优随机控制问题。我们的时间范围

    arXiv:2610.10054v1 Announce Type: cross  Abstract: Sampling transitions between metastable states is a central problem in dynamical systems theory and molecular dynamics in particular. A key challenge is the existence of high free-energy barriers that separate the states, making transitions extremely rare. Recent machine learning-based methods cast transition path sampling (TPS) as an optimal stochastic control (OSC) problem over a fixed time horizon, and parameterize the drift bias via a neural network trained by simulation-in-the-loop, requiring repeated biased rollouts. To address computational and performance guarantee issues of these models, we propose a new approach for the problem based on Koopman operators. Because Koopman operators are linear, their leading eigenfunctions reveal the metastable sets and provide an estimate of the committor function with no transition path information required. Furthermore, we formulate TPS as an OSC problem up to an exit time. Our time horizon 
    
[^103]: 在主机上进化，在边缘端预测：部署在线神经进化架构搜索用于横截面股票收益预测

    Evolve on the Host, Predict on the Edge: Deploying Online Neuroevolutionary Architecture Search for Cross-sectional Stock Return Prediction

    [https://arxiv.org/abs/2610.10038](https://arxiv.org/abs/2610.10038)

    该论文提出ONE-NAS在线神经进化架构搜索方法，通过主机进化、树莓派边缘端预测的分工流水线实现低延迟的日度横截面股票收益预测，在扣除实际交易成本后取得+27.5%的净收益，显著优于在线LSTM、在线GRU等基线模型。

    

    准确的预测模型通常规模庞大、在线更新成本高昂，且一旦训练完成架构便固定不变。我们将ONE-NAS（一种在线神经进化架构搜索方法，随着每批数据窗口的到来持续进化一个小型循环网络种群）应用于日度横截面股票收益预测，并在主机-端点流水线上进行试验：主机运行搜索过程，并通过TCP/IP将每一代的冠军基因组发送至树莓派4B，由后者进行在线预测。在树莓派上，单个冠军网络预测一个50只股票的窗口仅需24.6毫秒，40个岛屿冠军组成的集成模型需556毫秒，均远低于每日决策周期。在2022至2024年四个美国中盘股面板数据上，将种群读取为岛屿冠军的秩平均集成，在扣除实际交易成本后获得+27.5%的净收益，而在线LSTM、在线GRU和月度重训练LSTM基线模型的收益为+11.3%至+14.8%，单个最佳基因组仅为+4.5%。

    arXiv:2610.10038v1 Announce Type: cross  Abstract: Accurate forecasting models are usually large, expensive to update online, and fixed in architecture once trained. We apply ONE-NAS, an online neuroevolutionary architecture search that evolves a population of small recurrent networks as each window of data arrives, to daily cross-sectional stock return prediction, and pilot it on a host and endpoint pipeline: the host runs the search and ships each generation's champion genomes over TCP/IP to a Raspberry Pi 4B, which predicts online. On the Pi a single champion predicts a 50-stock window in 24.6~ms and the ensemble of 40 island champions in 556~ms, far inside the daily decision cycle. On four panels of US mid-cap equities over 2022--2024, reading the population as a rank-mean ensemble of island champions returns $+27.5\%$ net of realised transaction costs, against $+11.3$ to $+14.8\%$ for online LSTM, online GRU and monthly-retrained LSTM baselines and $+4.5\%$ for the single best gen
    
[^104]: 信号、噪声与硬件时间尺度的匹配用于相关噪声信号的滤波与预测

    Matching of signal, noise and hardware timescales for filtering and forecasting of correlated noise signals

    [https://arxiv.org/abs/2610.10037](https://arxiv.org/abs/2610.10037)

    本文利用纳米多孔氧化铌物理储层计算系统揭示了噪声相关时间、储层记忆与预测视界之间的匹配关系决定了相关噪声是被滤波还是被预测，并提出储层记忆视界和预测状态指数两个新指标来区分这两种工作模式。

    

    物理储层计算利用物理系统的非线性动力学来处理时间相关数据，比传统机器学习方法具有更高的能效。然而，物理储层具有固定的固有响应时间尺度，而现实世界的信号则在多个时间尺度上混合了确定性成分和随机成分。在此，我们利用纳米多孔氧化铌储层、合成噪声信号和加密货币价格波动数据证明，噪声相关时间、储层记忆和预测视界三者之间的关系决定了相关噪声是被滤除还是被预测。变化速度快于相关储层记忆和预测视界的噪声会被储层平均掉，而变化较慢的噪声的时间结构则足以用于算法预测。我们引入了储层记忆视界和预测状态指数来区分这两种运行模式。

    arXiv:2610.10037v1 Announce Type: new  Abstract: Physical reservoir computing exploits the nonlinear dynamics of physical systems to process time-dependent data with greater energy efficiency than conventional machine learning approaches. However, physical reservoirs have fixed intrinsic response timescales, whereas real-world signals combine deterministic and stochastic components across multiple timescales. Here we show, using a nanoporous niobium oxide reservoir, synthetic noisy signals and cryptocurrency-price volatility, that the relationship among noise correlation time, reservoir memory and forecast horizon determines whether correlated noise is filtered or predicted. Noise varying faster than the relevant reservoir memory and forecast horizon is averaged by the reservoir, whereas the temporal structure of slower-varying noise is sufficient for algorithmic forecasting. We introduce the reservoir memory horizon and forecasting regime index to distinguish these operating regimes. 
    
[^105]: 多头自注意力的高斯等价性

    Gaussian Equivalence for Multi-Head Self-Attention

    [https://arxiv.org/abs/2610.10033](https://arxiv.org/abs/2610.10033)

    利用随机矩阵理论建立了多头自注意力的高斯等价性，证明用缩放分数加高斯噪声替代softmax注意力可保持中心化输出的极限谱定律，从而分离了头分配与投影宽度的影响。

    

    对多头自注意力的理论理解是研究现代神经网络的基础。利用随机矩阵理论，我们建立了多头自注意力的高斯等价性：用缩放后的分数加上高斯噪声替代softmax注意力，可以保持中心化输出的极限谱定律。这一等价性还涵盖了依赖于键的值投影和输出投影。所得到的定律分离了头分配和投影宽度的影响，并区分了保持谱特性的跨头共享与头内键-值依赖。

    arXiv:2610.10033v1 Announce Type: cross  Abstract: A theoretical understanding of multi-head self-attention is fundamental to the study of modern neural networks. Using random matrix theory, we establish Gaussian equivalence for multi-head self-attention: replacing softmax attention with rescaled scores plus Gaussian noise preserves the limiting spectral law of the centered output. This equivalence also covers value and output projections that depend on the keys. The resulting laws separate the effects of head allocation and projection widths, and distinguish spectrum-preserving across-head sharing from within-head key--value dependence.
    
[^106]: 仅凭结构即可支持果蝇视觉系统中的高效视觉计算

    Structure alone supports efficient visual computation in the Drosophila visual system

    [https://arxiv.org/abs/2610.10023](https://arxiv.org/abs/2610.10023)

    本研究将固定的果蝇连接组结构与眼睛模型直接耦合，仅通过学习突触增益和神经元阈值，就实现了颜色辨别、形状分类和近似数字辨别等多任务视觉计算，证明了神经连接结构本身即可支撑高效的视觉信息处理。

    

    理解实测的突触连接结构在多大程度上决定了神经计算，仍然是一个核心挑战。在本研究中，我们将经过校对的成年黑腹果蝇连接组与其解剖学上高度保真的眼睛模型相结合。视觉信息首先输入眼睛模型，随后传递至连接组，最终由一个以蕈体Kenyon细胞为中心的线性解码器读取。由此构建了一个“仅连接组”模型，其中解剖连接图和眼睛几何结构均被固定，只有标量突触增益和神经元阈值可以被学习。该模型支持多任务视觉功能，包括颜色辨别、形状分类，以及遵循近似数感知所特有的比率依赖性缩放规律的数字辨别。为了在布线经济性约束下检验精确的神经连接是否真正重要，我们将生物连接图与逐渐保留生物突触约束的随机化连接图集合进行了比较。在匹配的布线……（摘要原文在此处截断）

    arXiv:2610.10023v1 Announce Type: cross  Abstract: Understanding the extent to which measured synaptic wiring determines computation remains a central challenge. Here, we couple the proofread adult Drosophila melanogaster connectome to an anatomically faithful model of its eye. Visual information is inputted in the eye model, then passed to the connectome, and finally read from a Kenyon-cell-centered linear decoder. This creates a connectome-only model in which the anatomical graph and eye geometry are fixed and only scalar synaptic gains and neuronal thresholds may be learned. The model supports multitask vision, including color discrimination, shape classification, and numerical discrimination that follows a ratio-dependent scaling characteristic of approximate number perception. To test whether precise connectivity is consequential under wiring economy, we compare the biological graph to randomized ensembles that increasingly preserve biological synaptic constraints. At matched wiri
    
[^107]: 通过扩散互信息控制隐式生成模型中的依赖关系

    Controlling Dependence in Implicit Generative Models via Spread Mutual Information

    [https://arxiv.org/abs/2610.10021](https://arxiv.org/abs/2610.10021)

    提出扩散互信息（SMI），通过对生成变量施加扩散核并跨噪声级别对互信息进行加权积分，克服了隐式生成模型中奇异分布缺少得分函数及密度比估计重叠性差的难题，实现对统计依赖的有效控制。

    

    互信息（MI）为隐式生成模型中抑制或鼓励统计依赖性提供了一个优化目标。然而，由于隐式模型的密度通常是难以处理的，直接评估互信息极具挑战性。一种补救方法是从条件得分与边缘得分之间的差异来估计生成器的梯度，而该得分差异又可以通过对通过分类学习得到的对数密度比进行求导来估计。然而，这种构造面临两个困难：(i) 奇异分布可能不具备所需的得分函数；(ii) 分布间重叠性差会阻碍密度比估计。因此，我们引入了扩散互信息，它是通过对生成变量施加一个共同的扩散核，从而获得的跨不同噪声级别的互信息的加权积分。高斯扩散可以产生平滑且严格为正的条件密度和边缘密度，从而将梯度构造扩展到……

    arXiv:2610.10021v1 Announce Type: cross  Abstract: Mutual information (MI) provides an objective for suppressing or encouraging statistical dependence in implicit generative models. However, direct MI evaluation is challenging in implicit models due to typically intractable densities. A remedy is estimating the generator gradient from the difference between conditional and marginal scores. This score difference can, in turn, be estimated by differentiating a log density ratio learned through classification. This construction nevertheless faces two difficulties: (i) singular distributions need not admit the required score functions, and (ii) poor overlap can hinder density-ratio estimation. We therefore introduce Spread Mutual Information (SMI), a weighted integral of MI across noise levels obtained by applying a common spreading kernel to the generated variable. Gaussian spreading yields smooth, strictly positive conditional and marginal densities, extending the gradient construction t
    
[^108]: 层结构上的振荡神经动力学

    Oscillatory Neural Dynamics over Sheaves

    [https://arxiv.org/abs/2610.10018](https://arxiv.org/abs/2610.10018)

    ONDA 提出了一种由学习的层传输算子驱动的二阶振荡信息波图学习框架，通过逐茎敏感性分析证明节点间交叉影响永不消失，从而实现有效的长程传播，并在长程传播、图瓶颈、图迁移和异配基准上持续超越标量波方法。

    

    有效的长程传播仍然是图神经网络中的核心挑战，因为单纯增加模型的传播深度并不能保证距离较远的节点能够有效地相互影响。层神经网络通过茎之间的矩阵值传输来丰富图传播；然而，仅有这种表达能力并不能自动带来有效的长程通信。我们提出了 ONDA，一个基于算子值信息波的长程图学习框架。茎值表示通过由学习到的层传输算子所支配的二阶动力学进行演化，将波动式传播与富有表现力的局部几何相结合。我们通过逐茎的敏感性分析来刻画长程影响，并证明节点间的交叉影响永不消失。在长程传播、严重图瓶颈、图迁移以及异配基准测试中，ONDA 始终优于标量波传播方法……（原文摘要在此处截断）

    arXiv:2610.10018v1 Announce Type: new  Abstract: Effective long-range propagation remains a central challenge in graph neural networks, as increasing a model's propagation depth does not guarantee that distant nodes effectively influence each other. Sheaf neural networks enrich graph propagation through matrix-valued transport between stalks; still, this expressivity alone does not automatically imply effective long-range communication. We introduce ONDA, a long-range graph learning framework based on operator-valued information waves. Stalk-valued representations evolve through second-order dynamics governed by learned sheaf transport operators, combining wave-like propagation with expressive local geometry. We characterize long-range influence through a stalk-wise sensitivity analysis and show that the cross-influence never vanishes. Across long-range propagation, severe graph bottlenecks, graph transfer, and heterophilic benchmarks, ONDA consistently improves over scalar wave propag
    
[^109]: 果蝇全连接组网络能够学习人类设计的认知任务

    A Drosophila Whole-Connectome Network Can Learn Human-Designed Cognitive Tasks

    [https://arxiv.org/abs/2610.10014](https://arxiv.org/abs/2610.10014)

    该研究证明，仅以单只果蝇的全脑连接组作为固定拓扑、并为每条神经连接学习一个标量权重，人工网络就能在加法运算和接地关系语言等人类设计的认知任务上显著超越随机重连的对照网络，表明生物神经环路结构本身具有超越其演化目标的通用计算价值。

    

    生物接线图能否在其演化所针对的行为之外，作为一种有用的计算基底？我们使用公开发布的MaleCNS v1.0连接组（从单只成年雄性果蝇标本重建而来），作为人工网络的固定循环拓扑结构。我们分别训练了两个模型：一个用于有界加法运算，另一个用于基于固定100词词表构建的受控接地关系语言任务。在两个模型中，每条解剖学连接仅学习一个标量权重。该解剖学图在留出的加法任务上达到92.77%的平均准确率，而保持度分布的有向重连对照图仅为67.93%。在严格的配对语言端点上（即将原始场景和顺序反转场景与其对应描述进行匹配），它在四个固定接口上达到61.59%，而匹配重连的对照图仅为44.17%。在标准接口下，它在固定的21个图比较中排名第一。在匹配的48组干预子（摘要此处被截断）

    arXiv:2610.10014v1 Announce Type: new  Abstract: Can a biological wiring diagram serve as a useful computational substrate beyond the behaviors for which it evolved? We use the publicly released MaleCNS v1.0 connectome, reconstructed from a single adult male Drosophila specimen, as the fixed recurrent topology of an artificial network. We train separate models for bounded addition and for a controlled grounded relational language task built from a fixed 100-word lexicon. In both models, one scalar is learned per anatomical edge. The anatomical graph reaches 92.77% mean accuracy on held-out addition, compared with 67.93% for directed degree-preserving rewires. On the strict paired language endpoint, which matches original and order-reversed scenes to their corresponding descriptions, it reaches 61.59% across four fixed interfaces, compared with 44.17% for matched rewires. At the canonical interface, it ranks first in a fixed 21-graph comparison. On the matched 48-group intervention subs
    
[^110]: 超越奖励压制：针对有界奖励热启动老虎机的近最优离线攻击

    Beyond Reward Suppression: Near-Optimal Offline Attacks on Warm-Start Bandits with Bounded Rewards

    [https://arxiv.org/abs/2610.10000](https://arxiv.org/abs/2610.10000)

    该论文首次证明针对热启动老虎机的最优成本攻击不能仅靠压制非目标臂实现，在目标臂奖励接近下边界时必须直接投入成本推广目标臂，并据此设计了成本分配明确的最优次线性攻击方案。

    

    对老虎机算法的对抗攻击旨在误导学习者选择目标臂，同时保持较小的攻击成本。现有攻击通常通过压制非目标臂来实现这一目标。然而在实践中，诸如虚假评论之类的操纵行为往往直接推广目标项目。我们通过对热启动老虎机的有界离线攻击来研究这一差距，其中攻击者只能在部署前向热启动历史中注入有效的动作-奖励对。我们证明，目标推广并非仅仅是启发式策略：当目标臂位于奖励下边界附近时，任何针对UCB算法、使其在几乎所有在线轮次中被选中且具有阶最优成本的攻击，都必须将其成本中不可忽略的比例分配给目标臂。随后，我们设计了一种实现最优次线性成本的攻击，并刻画了其成本在目标推广与非目标压制之间的分配方式。我们进一步将该攻击扩展至汤普森采样和 $\epsilon$-贪婪算法等策略。

    arXiv:2610.10000v1 Announce Type: new  Abstract: Adversarial attacks on bandits aim to mislead a learner toward a target arm while keeping the attack cost small. Existing attacks typically achieve this by suppressing non-target arms. In practice, however, manipulation such as fake reviews often directly promotes the target item. We study this gap through bounded offline attacks on warm-start bandits, where an attacker can inject only valid action-reward pairs into the warm-start history before deployment. We show that target promotion is not merely a heuristic: when the target arm lies near the lower reward boundary, any order-optimal-cost attack against UCB that makes it selected in nearly all online rounds must allocate a nonvanishing fraction of its cost to the target arm. We then design an attack that achieves the optimal sublinear cost and characterize its allocation between target promotion and non-target suppression. We further extend the attack to Thompson Sampling, $\epsilon$-
    
[^111]: 时序预测多样性：同样准确的时间序列模型产生不同的预测轨迹

    Temporal Predictive Multiplicity: Equally Accurate Time Series Models Yield Different Forecast Trajectories

    [https://arxiv.org/abs/2610.09994](https://arxiv.org/abs/2610.09994)

    该论文提出“时序预测多样性”框架，揭示预测性能几乎相同的时间序列模型可能在完整预测轨迹上产生显著分歧，且仅在单个时域约束多样性无法消除轨迹层面的差异。

    

    预测性能几乎相同的模型可能产生显著不同的预测结果，这种现象被称为预测多样性。先前的研究大多在单个标量输出的层面上探讨这一问题。然而，在时间序列预测中，跨多个预测时域的预测共同构成一条完整轨迹，仅按时域逐一比较可能会掩盖预测行为中的重要差异。为解决这一问题，我们提出了时序预测多样性这一框架，用于刻画预测性能几乎相同的模型之间在完整预测轨迹上的分歧。我们证明，仅约束预测性能仍可能存在大范围不同的轨迹。我们进一步表明，在单个时域上约束多样性可以部分减少、但无法消除轨迹层面的多样性。在11个数据集上对19种神经预测架构进行的实验证实了接近……

    arXiv:2610.09994v1 Announce Type: new  Abstract: Models with near-identical predictive performance can yield substantially different predictions, a phenomenon known as predictive multiplicity. Prior work has mostly studied this at the level of individual scalar outputs. In time-series forecasting, however, predictions across horizons jointly define a trajectory, and horizon-wise comparisons can hide important differences in predictive behavior. To address this problem, we introduce temporal predictive multiplicity, a framework that characterizes disagreement over complete forecast trajectories among models with near-identical predictive performance. We show that constraining predictive performance alone can still admit a broad range of different trajectories. We further show that constraining multiplicity at individual horizons partially reduces, but does not eliminate, trajectory-level multiplicity. Experiments with 19 neural forecasting architectures on 11 datasets confirm that near-
    
[^112]: 面向半导体晶圆Bin图开集异常检测的高效补丁级异常检测与扩散驱动生成建模融合方法

    Efficient Patch-Based Anomaly Detection Fused with Diffusion Driven Generative Modeling for Semiconductor Wafer Bin Map Open Set Anomaly Detection

    [https://arxiv.org/abs/2610.09993](https://arxiv.org/abs/2610.09993)

    该论文提出一种融合基于补丁的学生-教师检测器与去噪扩散概率模型的混合单类异常检测框架，通过百分位校准分数的固定凸组合，仅用700片正常晶圆训练即在晶圆Bin图开集异常检测中达到0.9985的AUROC，并将误分类数从852/1412显著降至618。

    

    晶圆Bin图（WBM）上的空间缺陷特征可以将良率损失追溯到特定的工艺故障，然而有监督分类器只能识别训练期间见过的缺陷类型，而基于单一机制构建的单类检测器往往只能捕获局部结构偏差或全局分布违规，很少能同时兼顾两者。本工作提出了一种混合单类框架，将基于补丁的学生-教师检测器与用于部分扩散重建的去噪扩散概率模型（DDPM）相耦合，并通过固定的凸组合融合两者经百分位校准的分数。该融合检测器在WM-38K混合类型数据集中仅使用700片正常晶圆进行训练，并在18,658片留出晶圆上进行评估，最终达到了0.9985的AUROC，将误分类数量从852（DDPM）和1,412（EfficientAD）减少到618，且所有两两差异在p < 0.001水平上具有统计显著性。除了总体准确率之外，分析……

    arXiv:2610.09993v1 Announce Type: new  Abstract: Spatial defect signatures on wafer bin maps (WBMs) trace yield loss to specific process faults, yet supervised classifiers recognize only the defect types seen during training, and one-class detectors built on a single mechanism tend to capture either local structural deviations or global distributional violations, but rarely both. This work proposes a hybrid one-class framework that couples a patch-based student-teacher detector (EfficientAD) with a denoising diffusion probabilistic model (DDPM) used for partial-diffusion reconstruction, and fuses their percentile-calibrated scores through a fixed convex combination. Trained on only 700 normal wafers from the WM-38K mixed-type dataset and evaluated on 18,658 held-out wafers, the fused detector reached an AUROC of 0.9985 and reduced misclassifications from 852 (DDPM) and 1,412 (EfficientAD) to 618, with all pairwise differences significant at p < 0.001. Beyond aggregate accuracy, the ana
    
[^113]: 将定价与广告和大语言模型相结合

    Marrying Pricing and Advertising with LLMs

    [https://arxiv.org/abs/2610.09985](https://arxiv.org/abs/2610.09985)

    提出了一种将LoRA微调的预训练大语言模型与在线actor-critic强化学习相结合的算法，使卖家在需求未知且仅有购买反馈的情况下，联合优化定价与LLM生成的广告以最大化收入。

    

    我们研究了一个序贯定价问题，其中卖家同时发布价格和由大语言模型（LLM）生成的广告。卖家的目标是在依赖于这两个决策的未知产品需求下最大化收入，同时仅能观察到每个报价是否促成购买。我们提出了一种在线actor-critic算法，该算法将预训练大语言模型的低秩适应（LoRA）与基于可用数据拟合的需求模型相结合。在每一轮中，actor生成广告，critic估计购买概率以指导价格选择。随后，所获得的反馈被用于同时更新actor和critic，其中critic的收入估计为actor的策略梯度更新提供基线。为了评估我们的方法，我们开发了一个包含三个合成需求模型以及一个基于真实市场数据构建的需求模拟器的评估框架。最后，我们将我们的算法与其他方法进行了比较……

    arXiv:2610.09985v1 Announce Type: cross  Abstract: We study a sequential pricing problem in which a seller jointly posts a price and an advertisement generated by a large language model (LLM). The seller aims to maximize revenue under an unknown product demand that depends on both decisions, while observing only whether each offer leads to a purchase. We propose an online actor-critic algorithm that combines low-rank adaptation (LoRA) of a pretrained LLM with a demand model fitted to available data. At each round, the actor generates an advertisement, and the critic estimates purchase probabilities to guide price selection. Then, the resulting feedback is used to update both the actor and the critic, with the critic's revenue estimates providing a baseline for policy gradient updates of the actor. To evaluate our approach, we develop an evaluation framework with three synthetic demand models and a demand simulator built from real-world marketplace data. Finally, we compare our algorith
    
[^114]: 极端二分类：利用极值理论实现对假负类的极端约束

    Extreme Binary Classification: Extreme Value Theory for Extreme Constraint on False Negative

    [https://arxiv.org/abs/2610.09984](https://arxiv.org/abs/2610.09984)

    本文提出“极端二分类”新问题，并基于极值理论设计了阈值自适应方法与基于置换检验的特征选择程序，使分类器的假负类率以快于 $1/N_1$ 的速率趋近于零，实验表现优于最先进方法。

    

    尽管二分类是机器学习中研究最为广泛的问题之一，但以学习一个假负类率几乎为零的分类器为目标的这一情形在很大程度上仍未被探索。在本文中，我们提出了“极端二分类”问题，其目标是学习一个假负类率 $\alpha$ 受 $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$ 约束的分类器，其中 $N_1$ 表示训练集中正例样本的数量。为了解决这一问题，我们提出了一种基于极值理论推导的理论保证的阈值自适应方法，并结合一种基于对样本最大值进行置换检验的特征选择程序。在四个不同规模的真实数据集上的实验结果表明，我们的方法优于当前最先进的方法。此外，我们还通过……展示了该方法的可解释性。

    arXiv:2610.09984v1 Announce Type: cross  Abstract: While binary classification is one of the most extensively studied problems in machine learning,   the regime in which the goal is to learn a classifier with an almost zero false negative rate remains largely unexplored.   In this paper, we introduce the Extreme Binary Classification problem, where the objective is to learn a classifier whose false negative rate $\alpha$ is constrained by $\epsilon_{N_1}=o_{N_1\to\infty}(1/N_1)$, with $N_1$ denoting the number of positive examples in the training set.   To address this problem, we propose a threshold adaptation method theoretically grounded in guarantees derived from Extreme Value Theory, together with a feature selection procedure based on a permutation test applied to sample maxima.   Experimental results on four real-world datasets of varying sizes demonstrate that our approach compares favorably with state-of-the-art methods.   In addition, we illustrate its interpretability throug
    
[^115]: TR-PTQ：基于泰勒区域重构的高精度纯整数Transformer训练后量化

    TR-PTQ: High-Accuracy Integer-Only Transformer Post Training Quantization via Taylor Region Reformulation

    [https://arxiv.org/abs/2610.09969](https://arxiv.org/abs/2610.09969)

    该论文发现Transformer量化的精度损失主要源于归一化层的尺度参数和GELU的复合近似而非SoftMax，并提出基于共享泰勒区域指数对数原语的统一纯整数公式TR-PTQ，使除法、平方根等复杂运算均可通过对数域整数运算完成，从而实现高精度的纯整数Transformer推理。

    

    训练后量化（PTQ）能够实现高效部署，然而由于非线性层的存在，Transformer架构的量化仍然充满挑战。尽管现有方法将精度损失归因于数值精度不足，往往需要浮点回退机制，但我们证明了精度下降实际上是由特定的结构性误差源驱动的。我们发现归一化层中学习到的尺度参数以及GELU中累积的近似误差是主要的误差来源，而SoftMax在激进量化下本质上仍保持鲁棒。为解决这些瓶颈，我们提出了TR-PTQ，这是一种使用共享泰勒区域（TR）指数与对数原语的统一纯整数公式。该方法使得计算代价高昂的操作（包括除法和平方根）能够完全通过对数域中的标准整数运算来执行。结合无需校准的、异常值感知的……（摘要在此处被截断）

    arXiv:2610.09969v1 Announce Type: new  Abstract: Post-training quantization (PTQ) enables efficient deployment, yet transformer architectures remain challenging to quantize due to nonlinear layers. While existing methods attribute accuracy loss to insufficient numerical precision, often necessitating floating-point fallbacks, we demonstrate that degradation is actually driven by specific structural error sources. We find that learned scale parameters in normalization layers and compounded approximations in GELU are the primary error contributors, whereas SoftMax remains inherently robust to aggressive quantization. To address these bottlenecks, we introduce TR-PTQ, a unified integer-only formulation using shared Taylor Region (TR) exponential and logarithm primitives. This approach allows computationally expensive operations, including division and square roots, to be performed entirely in the log-domain via standard integer arithmetic. Combined with a calibration-free, outlier-aware o
    
[^116]: 无从传导的力：一种由深度诱导的秩坍缩，任何作用于表示的损失都无法将其重新打开

    Force without transmission: a depth-induced rank collapse that no loss on the representation reopens

    [https://arxiv.org/abs/2610.09958](https://arxiv.org/abs/2610.09958)

    该研究发现，深度诱导的 transformer 秩坍缩无法通过任何作用于表示的损失项来修复，因为问题关键在于梯度传播路径被阻断而非矫正力不足，而恢复跳跃连接无需改变任何权重即可立即重开梯度路径、使秩得到恢复。

    

    训练可以将 transformer 推入一种秩坍缩状态：所有 token 的表示都指向同一个方向，学习随之停止。在与注意力相关的一种坍缩中，一个具有有界矫正力的损失项能够在训练过程中修复网络。我们由此追问：这样的损失项能否修复秩坍缩？我们通过削弱小型 transformer 的跳跃连接使其发生坍缩，并对坍缩网络的副本施加矫正。没有任何附加的损失项能够修复这种坍缩，即使其中最强的一种，其推力也仅约为任务梯度的十分之一。原因在于路径，而非强度：任务梯度已无法到达决定注意力关注位置的 query 和 key 权重，而附加损失项的梯度在到达坍缩形成的层块之前便已衰减殆尽。恢复跳跃连接——这一操作不改变任何权重——立刻重新打通了这条路径。秩随后得以恢复，但仅发生在远高于坍缩尺度的水平上。在一次高学习率的爆发之后，路径保持开启，并且……（原文摘要在此处截断）

    arXiv:2610.09958v1 Announce Type: new  Abstract: Training can drive a transformer into a rank collapse: all token representations point in one direction, and learning stops. In a related collapse of attention, a loss term with a bounded corrective force repairs the network during the run. We ask whether such a term repairs rank collapse. We collapse small transformers by weakening their skip connection and treat copies of the collapsed network. No added loss term repaired the collapse, although the stronger kind pushed with about a tenth of the task gradient. The reason was the path, not the strength. The task gradient no longer reached the query and key weights, which decide where attention looks, and the added term's gradient faded before the blocks where the collapse forms. Restoring the skip connection, which changes no weight, reopened this path at once. The rank then recovered, but only far above the scale of collapse. After a burst of high learning rate the path stayed open and 
    
[^117]: 用于近似推断模型（IM）推断的可能性径向传输

    Possibilistic Radial Transport for Approximate IM Inference

    [https://arxiv.org/abs/2610.09956](https://arxiv.org/abs/2610.09956)

    提出一种可能性径向传输方法，将参数的可能性轮廓值编码到源点半径中，并结合深度学习算法实现高效的近似可能性推断模型推断，使覆盖率评估、功效分析和新数据预测检验变得切实可行。

    

    arXiv:2610.09956v1 公告类型：交叉发布（cross）。摘要：在可能性推断模型（IM）框架下，在观察到数据之后再探索假设空间仍然有效，前提是显著性水平保持固定。其代价是计算量：每个合理性值都是可能性轮廓在假设上的上确界，而轮廓本身在每个被查询的参数值处都需要进行近似。我们提出了一种可能性径向传输方法，将参数的轮廓值隐藏在其源点的半径之中。当选择使壳层内熵最大化的传输时，对覆盖某一置信截断的参数进行采样就简化为截断半径的问题。我们提供了一种深度学习算法，在最大化每个壳层内熵的同时强制满足轮廓深度条件。我们的摊销方法使得对学习到的近似进行覆盖率和功效评估变得切实可行，同时也能对新数据集进行预测检验。我们还利用该采样器构建了Bel-Pl谱用于比较（摘要在此处截断）。

    arXiv:2610.09956v1 Announce Type: cross  Abstract: Probing the hypothesis space after seeing the data remains valid under possibilistic inferential models (IMs), provided the significance level stays fixed. The price is computation, as each plausibility is a supremum of the possibility contour over the hypothesis, and the contour itself is approximated at each queried parameter value. We propose a possibilistic radial transport, which hides the contour value of a parameter in the radius of its source point. When a transport that maximizes within-shell entropy is picked, sampling parameters covering a confidence cut becomes a matter of truncating the radius. We provide a deep learning algorithm that enforces the contour depth condition while maximizing the entropy within each shell. Our amortization makes coverage and power assessments of the learned approximation practical as well as predictive check of new datasets. We also use the sampler to construct a Bel-Pl spectrum for comparing 
    
[^118]: 基于随机物理信息神经元胞自动机的交通流动力学学习

    Learning Traffic Flow Dynamics with Stochastic Physics-Informed Neural Cellular Automata

    [https://arxiv.org/abs/2610.09946](https://arxiv.org/abs/2610.09946)

    本文提出一种物理信息神经元胞自动机（PI-NCA），通过设计与道路拓扑物理一致、保证车辆总数守恒的神经架构，并进一步扩展至随机动力学，实现从数据中学习符合物理约束的交通流局部演化规则。

    

    交通流建模对于理解和预测道路网络上车辆的集体动力学至关重要。元胞自动机提供了一种简单、可解释且强大的框架，通过局部交互规则来表示这些动力学，同时保留再现复杂宏观交通现象的能力。然而，在保持物理意义约束的同时从数据中学习局部转移规则仍然具有挑战性，特别是对于随机模型而言。在这项工作中，我们提出了一种物理信息神经元胞自动机（PI-NCA），用于数据驱动的交通流建模。基于标准的神经元胞自动机（NCA），我们设计了一种在物理上与道路拓扑相一致并保证车辆总数守恒的神经架构，从而将学习到的转移规则约束在物理上可容许的动力学范围内。我们进一步将该框架扩展至随机动力学……（摘要在此处截断）

    arXiv:2610.09946v1 Announce Type: new  Abstract: Traffic flow modeling is essential for understanding and predicting the collective dynamics of vehicles on road networks. Cellular automata provide a simple, interpretable yet powerful framework for representing these dynamics via local interaction rules, while retaining the ability to reproduce complex macroscopic traffic phenomena. However, learning local transition rules from data while preserving physically meaningful constraints remains challenging, particularly for stochastic models. In this work, we propose a physics-informed neural cellular automaton (PI-NCA) for data-driven traffic flow modeling. Building on the standard neural cellular automaton (NCA), we design a neural architecture that is physically consistent with the road topology and guarantees conservation of the total number of vehicles, thereby constraining the learned transition rules to physically admissible dynamics. We further extend this framework to stochastic dy
    
[^119]: 通往成功的多条路径：面向VLA泛化的多样性驱动强化学习微调

    Many Ways to Succeed: Diversity-Driven RL Fine-Tuning for VLA Generalization

    [https://arxiv.org/abs/2610.09943](https://arxiv.org/abs/2610.09943)

    该论文提出DRIVE方法，将成功行为的多样性作为显式的强化学习目标，通过在匹配任务条件下分组轨迹并进行时间对齐比较、引入成功条件下的内在奖励，从而扩大策略对有效解空间的覆盖，显著提升VLA模型在分布偏移下的泛化能力。

    

    强化学习（RL）微调通过闭环经验提升视觉-语言-动作（VLA）策略的性能，但其在微调分布之外的泛化能力仍然有限。我们的分析揭示了探索行为的一种选择性重塑：RL在整体上收缩行为分布，却使成功轨迹更加多样化，能以更少的采样次数获得成功，并比监督微调覆盖更多潜在的有效任务解空间。更广泛的成功模式覆盖有望在分布偏移下提供替代策略。受此启发，我们提出了DRIVE（面向VLA泛化的多样性驱动强化学习微调），它将成功行为的多样性转化为一个显式的强化学习目标。DRIVE在匹配的任务条件下对采样轨迹进行分组，通过时间对齐比较它们的轨迹，并从相对行为多样性中推导出以成功为条件的内在奖励。这种设计鼓励对可行解空间的更广泛覆盖。

    arXiv:2610.09943v1 Announce Type: cross  Abstract: Reinforcement learning (RL) fine-tuning improves vision-language-action (VLA) policies through closed-loop experience, yet generalization beyond the fine-tuning distribution remains limited. Our analysis reveals a selective reshaping of exploration: RL contracts behavior globally, yet diversifies successful trajectories, elicits success with fewer rollouts, and covers more of the latent task-valid solution space than supervised fine-tuning. Broader successful-mode coverage may provide alternative strategies under distribution shifts. Inspired by this, we introduce DRIVE (Diversity-driven RL fIne-tuning for VLA gEneralization), which turns successful-behavior diversity into an explicit RL objective. DRIVE groups rollouts under matched task conditions, compares their trajectories with temporal alignment, and derives a success-conditioned intrinsic reward from relative behavioral diversity. This design encourages broader coverage of feasi
    
[^120]: 多臂老虎机中的期望样本复杂度

    Expected Sample Complexity in Multi-Armed Bandits

    [https://arxiv.org/abs/2610.09929](https://arxiv.org/abs/2610.09929)

    本文提出了期望近似正确（ACE）新框架来研究多臂老虎机的期望样本复杂度，证明ACE保证蕴含几乎必然收敛到最优期望奖励，并揭示了确定性算法无法获得良好ACE界这一特性，同时针对次优水平ε已知与未知两种情形分析了随机算法。

    

    样本复杂度是序贯决策问题中广泛使用的一种指标，定义为智能体与环境交互过程中次优决策的次数。我们研究了随机多臂老虎机问题的样本复杂度，并引入了期望样本复杂度这一性能度量，在一个称为“期望近似正确”的新型框架中对其进行分析。我们证明了ACE保证意味着几乎必然收敛到最优期望奖励，这与其他框架中的高概率保证形成对比，同时我们还展示了如何将ACE保证转化为显式的期望遗憾界。我们进一步证明，与现有度量不同，确定性算法无法获得良好的ACE界，并在两种设置下分析了随机算法：当允许的次优水平 ε 对算法已知时，以及当其未知时。在前一种情况下，我们设计了一种……

    arXiv:2610.09929v1 Announce Type: new  Abstract: Sample complexity is a widely used metric in sequential decision-making problems, defined as the number of suboptimal decisions during the interaction between the agent and an environment. We study the sample complexity of stochastic multi-armed bandit problems and introduce the expected sample complexity performance measure, analyzing it in a novel framework called approximately correct in expectation (ACE). We show that ACE guarantees imply almost sure convergence to the optimal expected reward, in contrast to high-probability guarantees found in other frameworks, and also show how to convert ACE guarantees into explicit expected regret bounds. We further show that, in contrast to existing measures, deterministic algorithms cannot obtain favorable ACE bounds, and analyze stochastic algorithms in two settings: when the allowed suboptimality level $\epsilon$ is known to the algorithm and when it is unknown. In the former, we devise an ex
    
[^121]: KGATE：一个知识图谱嵌入训练环境

    KGATE : a Knowledge Graph Embedding Training Environment

    [https://arxiv.org/abs/2610.09927](https://arxiv.org/abs/2610.09927)

    本文提出了KGATE，一个基于PyTorch Geometric和TorchKGE的模块化Python库，通过允许用户灵活组装或自定义编码器、解码器、损失函数等组件，解决了现有知识图谱嵌入库缺乏完整自编码器支持、缺乏维护和结果不可比的问题。

    

    知识图谱嵌入模型将知识图谱中的实体和关系编码到低维潜在空间中，从而支持分类或链接预测等任务。大多数KGE模型遵循自编码器架构，其中编码器将知识图谱投影到潜在空间，解码器再对其进行重构。将编码器和解码器组件相结合的需求日益增长，然而现有的库很少支持完整的自编码器，往往缺乏维护，依赖于未加文档说明的默认超参数，并且产生的结果无法在不同库之间进行比较。在此，我们提出了KGATE（知识图谱自编码器训练环境），这是一个基于PyTorch Geometric和TorchKGE构建的模块化Python库。KGATE允许用户将初始化器、编码器、解码器、损失函数、负采样器和评估指标作为构建块进行组装，或者插入自定义的构建块。KGATE还包含一个预处理流程……

    arXiv:2610.09927v1 Announce Type: new  Abstract: Knowledge graph embedding (KGE) models encode the entities and relations of a knowledge graph into a low-dimensional latent space, enabling tasks such as classification or link prediction. Most KGE models follow an autoencoder architecture, in which an encoder projects the knowledge graph into the latent space and a decoder reconstruct it. Combining both encoder and decoder components is increasingly needed, yet existing libraries rarely support complete autoencoders, are often unmaintained, rely on undocumented default hyperparameters, and produce results that cannot be compared across libraries. Here we present KGATE (Knowledge Graph Autoencoder Training Environment), a modular Python library built on PyTorch Geometric and TorchKGE. KGATE lets users assemble initializers, encoders, decoders, losses, negative samplers, and evaluation metrics as building blocks, or plug in their own block. KGATE includes a preprocessing procedure that co
    
[^122]: 深度学习中Hessian矩阵的特征值：对称性的起源及其破缺

    Eigenvalues of the Hessian in Deep Learning: The Origin of Symmetry and Its Breaking

    [https://arxiv.org/abs/2610.09919](https://arxiv.org/abs/2610.09919)

    本文提出，深度学习中训练模型Hessian特征值呈现的“零附近大块+孤立离群值”的谱结构，源于相对一个隐藏的高度对称参考构型的对称破缺——该参考构型的Hessian具有权重对称性之外的不变性，其对称破缺产生了观测到的谱层级结构。

    

    深度学习中训练后模型的Hessian谱呈现出一种持续存在的模式：特征值组织成不同的簇，包括一个位于零附近的大块以及少数孤立的离群值。本文表明，当将原始设定理解为对附近一个原本隐藏的高度对称参考构型的偏离时，这些谱现象便获得了一个自然的解释。通过对架构、数据分布或参数度量等进行修改，可以揭示出一个邻近的参考构型，其Hessian展现出丰富的、无法由权重对称性所解释的不变性。在该参考构型中，对称性使得谱能够被精确描述，并迫使出现高维的核空间以及大重数的特征值。而回到原始构型则破坏了Hessian的对称性，从而产生了所观测到的簇与离群值的层级结构。该框架在相当一般的情形下被建立，并对……（摘要原文在此处截断）

    arXiv:2610.09919v1 Announce Type: new  Abstract: Hessian spectra at trained models in deep learning exhibit a persistent pattern: eigenvalues organize into distinct clusters, including a large bulk near zero and a few isolated outliers. This paper shows that a natural account of these spectral phenomena emerges when the original setting is understood as a departure from a nearby, otherwise hidden, highly symmetric reference.   Modifications, including changes to the architecture, data distribution, or parameter metric, expose a nearby reference configuration whose Hessian exhibits rich invariances-ones not accounted for by weight symmetries. There, symmetry enables a precise description of the spectra, forcing high-dimensional kernels and eigenvalues of large multiplicity. Returning to the original configuration breaks the Hessian symmetry and thereby produces the observed hierarchy of clusters and outliers.   The framework is developed in some generality, with a detailed analysis of t
    
[^123]: NeuralZip：面向快速无损压缩的可复用设置

    NeuralZip: Reusable Setup for Fast Lossless Compression

    [https://arxiv.org/abs/2610.09916](https://arxiv.org/abs/2610.09916)

    NeuralZip 通过一次性准备并复用浮点指数的统计结构，使模型检查点的无损压缩速度较基线提升 1.81–21.33 倍，同时保证逐位精确重建。

    

    无损压缩可以在不改变模型权重浮点数值的情况下减少权重的存储与传输开销，但反复进行的统计分析和编码构建会带来额外的计算负担。我们研究了能否将指数部分的统计结构一次性准备好并加以复用。为此，我们提出了 NeuralZip，它将具有相似指数分布的数据块分组、共享霍夫曼编码，并有选择地使用打包指数来表示重复出现的指数元组，从而获得额外的适度压缩比。该设置步骤在后续编码之前选择好这些表示方式，而每次编码仍然处理当前的张量数值。在浮点模型检查点上，完成设置后的压缩速度比基线方法快 1.81–21.33 倍，并且能够实现精确的逐位重建。我们还展示了该设置可以被预先计算，并从另一个兼容的架构中迁移过来，同时保持相似的压缩效果。

    arXiv:2610.09916v1 Announce Type: new  Abstract: Lossless compression can reduce the storage and movement of model weights without changing their floating-point values, but repeated statistical analysis and code construction add computational overhead. We study whether the statistical structure of exponents can be prepared once and reused. For this, we introduce NeuralZip, which groups chunks with similar exponent distributions, shares Huffman codes, and selectively represents recurring exponent tuples using packed exponents, thereby achieving additional moderate compression ratios. A setup chooses these representations before subsequent encodings, while every encoding still processes the current tensor values. In floating-point model checkpoints, post-setup compression is 1.81-21.33$\times$ faster than the baselines and achieves exact bit-to-bit reconstruction. We show that this setup can be precomputed and transferred from another compatible architecture, preserving similar compressi
    
[^124]: RollVerify：在长尾生成强化学习中架起效率与准确性之间的桥梁

    RollVerify: Bridging Efficiency and Accuracy in Long-Tail Rollout Reinforcement Learning

    [https://arxiv.org/abs/2610.09914](https://arxiv.org/abs/2610.09914)

    提出 RollVerify 框架，基于部分 rollout 在样本进入训练前主动验证并修复过时的离策略样本，从而在提升强化学习训练效率的同时缩小与完全在线策略训练之间的准确性差距。

    

    强化学习对于提升大语言模型的推理和泛化能力至关重要。它依赖大规模的 rollout 生成，而随着上下文窗口的增长，这些生成的长度日益呈现长尾分布。在在线策略训练中，这些长尾 rollout 会导致 GPU 空泡（算力闲置），降低系统利用率并限制强化学习的可扩展性。异步或部分 rollout 方法通过放宽同步要求来提高吞吐量，但不可避免地会引入过时的离策略样本（轨迹），可能损害最终准确性。现有方法主要通过在训练期间对离策略样本进行重新加权来缓解这一离策略问题，但与完全在线策略训练相比仍可能存在性能差距。在这项工作中，我们并非在训练期间被动地对样本重新加权，而是提出 RollVerify——一个基于部分 rollout 构建的轻量级强化学习框架，它在样本进入训练之前主动对样本进行验证和修复。具体而言，它引入……

    arXiv:2610.09914v1 Announce Type: new  Abstract: Reinforcement learning is crucial for improving large language models' reasoning and generalization. It relies on massive rollouts whose lengths become increasingly long-tailed as context windows grow. In on-policy training, these long-tail rollouts can result in GPU bubbles, reducing system utilization and limiting RL scalability. Asynchronous or partial-rollout methods improve throughput by relaxing synchronization, but inevitably introduce stale off-policy samples (trajectories) that may hurt final accuracy. Existing approaches mainly mitigate this off-policy issue by reweighting off-policy samples during training, yet they can still leave a performance gap compared to fully on-policy training. In this work, rather than passively reweighting samples during training, we propose RollVerify, a lightweight RL framework built on partial rollout that actively verifies and repairs samples before they enter training. Specifically, it introduc
    
[^125]: 仅从站点观测数据学习联合概率天气预报

    Learning joint probabilistic weather forecasts from station observations alone

    [https://arxiv.org/abs/2610.09898](https://arxiv.org/abs/2610.09898)

    CLARA模型仅凭站点观测数据（无需数值天气预报或再分析数据）即可学习五个地面变量的联合高斯概率预报分布，以约2.8万参数的小模型在CPU上实现了优于各类基线的联合概率预报能力。

    

    评估复合天气风险需要能够刻画变量间依赖关系的预报。CLARA（校准平流路由注意力，Calibrated Advection-Routing Attention）仅利用站点观测数据即可学习五个地面变量的联合高斯预测分布，无需数值天气预报或再分析数据；该模型仅约28,000个参数，支持CPU训练和预测。在96个站点、六个多年折叠的数据集上，其平均时效能量分数比具有相同时间输入的学习型对比方法低4.9%（与参数量相近的方法相比低4.7%），比统计基线方法低11-65%。在固定边缘方差的条件下，移除学习到的相关性会使每个站点的联合负对数似然恶化1.0-2.8 nats。一个在给定假设下被证明具有一致性的协方差尺度估计器可改善短时效的校准效果，但在长时效时会出现过度校正。合成干预实验显示注意力偏差系数a

    arXiv:2610.09898v1 Announce Type: cross  Abstract: Assessing compound weather risks requires forecasts representing dependence between variables. CLARA (Calibrated Advection-Routing Attention) learns joint Gaussian predictive distributions of five surface variables from station observations alone, without numerical weather prediction or reanalysis; the approximately 28,000-parameter model supports CPU training and prediction. Across six multi-year folds on 96 stations, its lead-mean energy score is 4.9% lower than that of a learned comparator with matched temporal inputs (4.7% with a similar parameter count) and 11-65% lower than those of statistical baselines. Holding marginal variances fixed, removing learned correlations worsens joint negative log-likelihood by 1.0-2.8 nats per station. A covariance-scale estimator, proved consistent under stated assumptions, improves short-lead calibration but over-corrects at long leads. Synthetic interventions show an attention-bias coefficient a
    
[^126]: 耗散知识动力学模型的可辨识性：设计激励下的精确恢复与观测数据上的退化

    Identifiability of a dissipative knowledge-dynamics model: exact recovery under designed excitation, degeneration on observational data

    [https://arxiv.org/abs/2610.09889](https://arxiv.org/abs/2610.09889)

    该论文将人类学习建模为参数具有机制含义的耗散常微分方程组，证明了在设计激励条件下模型参数可从数据中被精确恢复（双概念情形有闭式解），而在仅依赖观测数据时可辨识性会退化，并提出了数值精确等价且速度大幅提升的半隐式L-稳定批量求解器。

    

    人类学习是一个耗散动力学过程：熟练度通过练习积累，因遗忘而衰减，并在相互依赖的概念之间传播。我们将其建模为一个非线性耗散常微分方程组，其参数具有机制层面的含义（编码先修耦合关系的概念传递矩阵、各概念的遗忘率，以及饱和的练习-响应增益），并研究这些参数何时能够真正从数据中被恢复。我们在显式激励条件下证明了相应逆问题的结构可辨识性定理，针对双概念情形给出了构造性的闭式恢复方法，并同时给出了单调性、鲁棒性与L-稳定性结果。我们为耗散子系统推导了一种半隐式L-稳定数值格式，以及一个与逐轨迹公式数值等价的批量求解器（预测逐位一致，梯度误差达 $10^{-10}$），同时速度提升两个数量级。

    arXiv:2610.09889v1 Announce Type: new  Abstract: Human learning is a dissipative dynamical process: mastery accumulates through practice, decays through forgetting, and propagates across interdependent concepts. We model it as a nonlinear dissipative system of ordinary differential equations whose parameters are mechanistically meaningful (a concept-transfer matrix encoding prerequisite coupling, per-concept forgetting rates, and a saturating practice-response gain), and we study when those parameters can actually be recovered from data. We prove a structural identifiability theorem for the associated inverse problem under explicit excitation conditions, with constructive closed-form recovery for the two-concept case, together with monotonicity, robustness and L-stability results. We derive a semi-implicit L-stable scheme for the dissipative subsystem and a batched solver numerically equivalent to the per-trajectory formulation (bit-exact predictions, gradients to $10^{-10}$) yet two o
    
[^127]: 稀疏化随机性而非容量：通过先验尺度的深度权重因子分解实现部分随机性

    Sparsifying Stochasticity, Not Capacity: Partial Stochasticity via Deep Weight Factorization of Prior Scales

    [https://arxiv.org/abs/2610.09886](https://arxiv.org/abs/2610.09886)

    该论文提出通过对先验尺度进行深度权重因子分解来学习贝叶斯神经网络中哪些参数应保持随机性，使正则化稀疏化随机性而非模型容量，并提供了可线性时间检验的通用条件密度逼近证书，同时证明常见的采样-优化混合方案是II型最大后验目标的随机近似。

    

    贝叶斯神经网络不必完全随机也能成为通用的条件密度逼近器，但哪些参数应该是随机的仍是一个悬而未决的问题。我们通过对先验尺度（即参数先验的标准差）应用深度权重因子分解来学习这种划分，同时利用最大均值差异目标将函数先验拟合到高斯过程。先验尺度低于某个截断值的参数变为确定性的，并在推理过程中进行优化，因此该正则化器稀疏化的是随机性而非容量。我们给出了一个可在线性时间内检验的通用条件密度逼近证书，以及在证书失败时的最小修复方案。我们进一步证明，常见的混合方案——对部分参数进行采样而对其他参数进行优化——是针对某一类II型最大后验目标的随机近似，并且耦合的步长可能会留下追踪偏差。

    arXiv:2610.09886v1 Announce Type: cross  Abstract: Bayesian neural networks need not be fully stochastic to be universal conditional density approximators, but it remains open which parameters should be stochastic. We learn this split by applying deep weight factorization to the prior scales, which are the standard deviations of the parameter priors, while fitting the functional prior to a Gaussian process with a maximum mean discrepancy objective. A parameter whose prior scale falls below a cutoff becomes deterministic and is optimized during inference, so the regularizer sparsifies stochasticity rather than capacity. We give a certificate for universal conditional density approximation that is checkable in linear time, together with a minimal repair when it fails. We further show that the common hybrid scheme of sampling some parameters and optimizing the others is stochastic approximation for a type-II maximum a posteriori objective, and that coupled step sizes can leave a tracking 
    
[^128]: 用于快速且鲁棒的混合精度训练后量化的逐层误差归因

    Layerwise Error Attribution for Fast and Robust Mixed-Precision Post-Training Quantization

    [https://arxiv.org/abs/2610.09877](https://arxiv.org/abs/2610.09877)

    提出了一种基于逐层概率误差分析的混合精度训练后量化方法，通过分离传播误差与局部扰动构建可分离评分，实现了无需外部求解器的快速比特分配，并显著增强了对校准数据被污染情况下的鲁棒性。

    

    混合精度训练后量化是一种网络压缩方法，它在全局内存预算下，利用小规模校准集逐层分配比特数。其主要难点在于克服分配问题的组合爆炸特性，以及应对对小型且可能被污染的数据库的敏感性。因此，一个高效的分配方法应当计算快速，并且在校准数据被污染时仍能保持模型质量。为了设计这样的方法，我们推导了一种逐层的量化误差概率分析，将传播误差与给定层引入的局部扰动分离开来。我们利用该局部项构建了一个可分离的评分指标，用于一个无需外部求解器的简单分配算法。我们方法的概率特性使其对被污染的数据具有鲁棒性。在使用 DRUNet 的去噪任务上，在平均每个权重 4 比特的预算下，我们的方法达到或超越了现有方法的性能。

    arXiv:2610.09877v1 Announce Type: new  Abstract: Mixed-precision post-training quantization is a network compression method that assigns bits layer by layer, under a global memory budget using a small calibration set. The main difficulties are to overcome the combinatorial nature of the allocation problem and to manage the sensitivity to small, potentially corrupted databases. Hence, an efficient allocation method should be fast to compute and preserve model quality when calibration data are corrupted. To design such a method, we derive a layerwise probabilistic analysis of the quantization error that separates propagated error from the local perturbation introduced at a given layer. We use this local term to build a separable score for a simple allocation algorithm, that requires no external solver. The probabilistic nature of our approach brings robustness to corrupted data. On denoising tasks with DRUNet, with an average budget of 4 bits per weight, our method matches or improves st
    
[^129]: 先思考后作画：面向扩散模型的递归潜在推理

    Think Before You Paint: Recursive Latent Reasoning for Diffusion Models

    [https://arxiv.org/abs/2610.09876](https://arxiv.org/abs/2610.09876)

    提出PaTh框架，让仅1000万参数的小型递归网络在潜在空间中进行“思考”并通过ControlNet引导冻结的扩散模型，仅靠标准重建损失训练（无需符号目标、求解器或验证器），在困难数独（92.5%）和极端数独（71.2%）视觉推理任务上大幅刷新此前最佳纪录。

    

    扩散模型能够生成逼真的图像，但在视觉推理任务上常常表现不佳，例如填充数独谜题或绘制迷宫的通行路径。当存在离散符号表示时，诸如微型递归模型（TRM）之类的递归方法甚至可以解决这些谜题的困难实例。我们探讨如何将这种推理能力迁移到没有任何符号表示的像素域上。我们提出了 Painter-Thinker（PaTh）：一个小型递归网络（思考者 Thinker）在编码了带噪图像与条件信息的学习令牌网格上进行推理，在每个去噪步骤内细化潜在状态，并通过 ControlNet 适配器引导一个冻结的扩散模型（绘画者 Painter）。思考者仅使用标准重建损失进行训练，无需符号目标、求解器或验证器。PaTh 仅用 1000 万参数就解决了 92.5% 的困难 MNIST 数独谜题（此前最佳为 75%）和 71.2% 的极端难度谜题（此前最佳为 4.1%）。

    arXiv:2610.09876v1 Announce Type: cross  Abstract: Diffusion models generate realistic images but often fail on visual reasoning tasks, such as filling in a Sudoku or drawing the path through a maze. When a discrete symbolic representation is available, recursive methods such as the Tiny Recursive Model (TRM) solve even hard instances of these puzzles. We ask how such reasoning can be carried over to pixels, where no symbolic representation is available. We propose Painter-Thinker (PaTh): a small recursive network (the Thinker) reasons over a grid of learned tokens that encode the noisy image and the conditioning, refines a latent state within every denoising step, and steers a frozen diffusion model (the Painter) through ControlNet adapters. The Thinker is trained with the standard reconstruction loss alone, without symbolic targets, a solver, or a verifier. PaTh solves 92.5% of hard MNIST Sudoku puzzles (prior best 75%) and 71.2% of extreme ones (prior best 4.1%), with 10M parameters
    
[^130]: 一种用于隐式地质建模的AI辅助数据调理与地质解释工作流程

    An AI-assisted conditioning and geological interpretation workflow for usage in implicit geological modeling

    [https://arxiv.org/abs/2610.09871](https://arxiv.org/abs/2610.09871)

    该论文提出了一套AI辅助工作流程，利用自监督/半监督对比学习CNN对浅层至深层（300–3500米）陆上地震数据进行降噪与插值，并以极少的人工标注数据自动解释层位与断层，从而加速隐式地质建模。

    

    隐式建模与相对地质时间是能够实现更高效、更快速、偏差更小且更具可重复性建模结果的地质建模技术。为了达到最佳运行效果，这些技术需要大量约束良好的输入数据。在“地平线欧洲”GO-Forward项目与MOOI WarmingUP GOO项目的框架下，为加速隐式建模，机器学习（ML）方法已被测试并实现于一个工具包中，用于解释浅层至深层（约300–3500米）的（陆上）地震数据。其目标是通过高效解释地震数据中的层位和断层，快速表征这一深度域。第一步是应用人工智能技术（如自监督和半监督对比学习卷积神经网络CNN）进行降噪和插值，以改进信号质量。随后，通过（半）……（原文摘要在此处不完整）

    arXiv:2610.09871v1 Announce Type: cross  Abstract: Implicit modeling and Relative Geologic Time are geological modeling techniques that enable more efficient, faster, less biased and more reproducible modeling results. For optimal operation, these techniques require many well-constrained input data. In the framework of the Horizon Europe GO-Forward and MOOI WarmingUP GOO projects and to accelerate Implicit modeling, Machine Learning (ML) methods have been tested and implemented in a toolkit for the interpretation of (onshore) seismic data from the shallow to deep range (+- 300 - 3500 m). The goal is to rapidly characterise this depth domain by efficient interpretation of horizons and faults in seismic data. The first step is to improve the signal by applying AI techniques like self-supervised and semi-supervised contrastive learning CNN's for noise reduction and interpolation. Next, horizons and faults are interpreted with minimal use of human-generated training data by using (semi-) s
    
[^131]: MUNITE：面向任意到任意多模态生成的统一多模态潜在推断

    MUNITE: Unified Multimodal Latent Inference for Any-to-Any Multimodal Generation

    [https://arxiv.org/abs/2610.09866](https://arxiv.org/abs/2610.09866)

    MUNITE提出统一的潜变量框架，将编码与潜变量生成统一为同一条件流推断问题，借助共享潜在表示与基于自蒸馏的条件流匹配，实现从任意模态子集到任意模态的多模态生成。

    

    我们提出了MUNITE，一个用于灵活的任意到任意多模态生成的潜变量框架，它将编码和潜变量生成视为在不同观测证据量下的同一个推断问题。给定任意模态子集，MUNITE对与完整观测相关联的潜在表示的条件分布进行建模。完整观测对应于确定性编码的恢复，无观测对应于潜在边缘分布的恢复，而中间子集则定义了条件潜在推断——所有这些都统一在单一的条件流模型中完成。一个共享的潜在样本捕获了在所有生成目标之间必须保持一致的变异，而特定于模态的生成解码器则独立地对剩余的不确定性进行建模。为了从不完整的训练样本中学习这些条件分布，我们通过自蒸馏扩展了条件流匹配：以更丰富的可用观测为条件的预测（作为教师）来监督……

    arXiv:2610.09866v1 Announce Type: new  Abstract: We introduce MUNITE, a latent-variable framework for flexible any-to-any multimodal generation that treats encoding and latent generation as the same inference problem under different amounts of observed evidence. Given any subset of modalities, MUNITE models the conditional distribution over the latent representation associated with the complete observation. Full observation recovers deterministic encoding, no observation recovers the latent marginal, and intermediate subsets define conditional latent inference, all within a single conditional flow model. A shared latent sample captures variation that must remain consistent across generated targets, while modality-specific generative decoders model the remaining uncertainty independently. To learn these conditional distributions from incomplete training examples, we extend conditional flow matching through self-distillation: predictions conditioned on richer available observations super
    
[^132]: 面向表征学习的全局平均精度

    Global Average Precision for Representation Learning

    [https://arxiv.org/abs/2610.09863](https://arxiv.org/abs/2610.09863)

    该论文提出全局平均精度（gAP）及其可微分代理损失 gSAP，通过将所有查询-候选对纳入同一排序并联合考虑批次内全部成对比较，弥补了 mAP 和 InfoNCE 等指标与损失不考虑跨查询相似度可比性的缺陷，可作为现有损失的即插即用替代。

    

    标准的信息检索指标，例如平均精度均值（mAP），是一次评估一个查询的性能，依据查询与其正样本之间的相似度相对于其与负样本之间相似度的比较。常见的表征学习损失函数也是如此，例如 InfoNCE 和逐查询 AP 代理损失。它们都没有考虑相似度在不同查询之间是否具有可比性，而任何采用单一决策阈值的系统都依赖于这一点。全局平均精度则做到了这一点，它将所有查询-候选对排入同一个列表并计算单一的 AP。我们提出了 gSAP，一种 gAP 的可微分代理损失。它只需要一个相似度矩阵和一个标记正样本对的二值矩阵，与现有损失的输入相同，因此可以作为它们的即插即用替代品，并且对编码器、模态和监督来源均不敏感。由于它联合考虑了批次中所有可能的成对比较，它还能……（原文摘要在此处截断）

    arXiv:2610.09863v1 Announce Type: cross  Abstract: Standard information retrieval metrics, such as mean Average Precision (mAP), assess performance one query at a time, based on how the similarities between a query and its positives compare against those with its negatives. The same holds for common representation learning losses, such as InfoNCE and per-query AP surrogates. None of them considers whether similarities are comparable across queries, which any system with a single decision threshold relies on. Global Average Precision (gAP) does, by ranking all query-candidate pairs in one list and computing a single AP. We introduce gSAP, a differentiable surrogate of gAP. It needs only a similarity matrix and a binary matrix marking the positive pairs, the same input as existing losses, so it is a drop-in replacement for them and agnostic to the encoder, the modality, and the source of supervision. Since it considers all possible pairwise comparisons in the batch jointly, it also remai
    
[^133]: DeepTopoClustering：基于四维点云的无监督地表过程分类体系推导用于地形监测

    DeepTopoClustering: Unsupervised Derivation of Surface Process Taxonomy from 4D Point Clouds for Topographic Monitoring

    [https://arxiv.org/abs/2610.09860](https://arxiv.org/abs/2610.09860)

    提出了一种无监督深度聚类框架DeepTopoClustering，通过将变化四维对象转化为GeoMorphogram分布序列并利用卷积自编码器与分层聚类目标学习潜在嵌入，从而从永久激光扫描点云中自动推导出地表变化过程的层次分类体系，实现地形变化监测。

    

    由永久激光扫描（PLS）获取的四维点云能够对动态地形环境中的地表变化进行精确的高频监测。然而，现有方法在将检测到的地表活动组织成有意义的过程类型方面仍然存在局限。我们提出了DeepTopoClustering（DTC），这是一个无监督框架，用于从基于对象的地表活动（即所谓的变化四维对象，4D-OBCs）中推导出分层过程分类体系。我们将每个4D-OBC转换为GeoMorphogram，这是一种表示空间有界的地表活动内地形变化时间演化的分布序列。卷积自编码器从GeoMorphogram中学习潜在嵌入，并通过分层深度聚类目标进行联合优化，从而将地表活动组织成层次结构。我们使用两个沙质海滩场地的四维数据集及其组合的专家标注来评估所学习到的层次结构。

    arXiv:2610.09860v1 Announce Type: cross  Abstract: 4D point clouds acquired by permanent laser scanning (PLS) enable accurate high-frequency monitoring of surface change in dynamic topographic environments. However, existing methods remain limited in organizing detected surface activities into meaningful process types. We propose DeepTopoClustering (DTC), an unsupervised framework for deriving a hierarchical process taxonomy from object-based surface activities, so-called 4D objects-by-change (4D-OBCs). We transform each 4D-OBC into a GeoMorphogram, a distributional sequence representing the temporal evolution of topographic change within a spatially bounded surface activity. A convolutional autoencoder learns latent embeddings from GeoMorphograms, which are jointly optimized using a hierarchical deep clustering objective to organize surface activities into a hierarchy. We evaluate the learned hierarchy using expert annotations on two 4D datasets of sandy beach sites and their combinat
    
[^134]: 从任务结果训练大语言模型智能体的顾问

    Training Advisors for LLM Agents from Task Outcomes

    [https://arxiv.org/abs/2610.09858](https://arxiv.org/abs/2610.09858)

    提出Caddie方法，通过强化学习仅以智能体最终任务成功与否作为训练信号来训练批评者提供自然语言建议，且训练后的批评者能泛化到不同规模和架构的多个基础模型并显著提升任务成功率。

    

    大语言模型智能体通过交替进行推理和工具调用，并结合环境反馈的观察来完成多步骤任务。先前的研究表明，自然语言反馈可以帮助这些智能体在任务执行过程中修正其决策。我们提出了Caddie，一种训练批评者在智能体执行任务过程中提供自然语言分析与建议的方法。与依赖步骤级标签或参考批评的现有方法不同，Caddie从智能体在接收批评者反馈后是否最终取得成功中学习。我们在保持基础模型冻结的情况下，通过强化学习优化批评者。该批评者仅基于单一基础模型在多跳问答任务上训练，而我们的Qwen3-4B批评者在四个不同规模和架构的基础模型上均提升了任务成功率，其中包括三个在批评者训练阶段未曾使用过的模型。在MuSiQue基准上，训练后的批评者使Qwen3-4B的成功率提升了超过……（原文此处截断）

    arXiv:2610.09858v1 Announce Type: new  Abstract: Large language model agents tackle multi-step tasks by interleaving reasoning and tool calls with observations from the environment. Prior work has shown that natural-language feedback can help these agents revise their decisions during task execution. We introduce Caddie, a method for training critics to provide natural-language analysis and advice as agents work through a task. Unlike approaches that rely on step-level labels or reference critiques, Caddie learns from whether the agent ultimately succeeds after receiving the critic's feedback. We optimize the critic through reinforcement learning while keeping the base model frozen. Trained on multi-hop question answering with a single base model, our Qwen3-4B critic improves success rates across four base models of different scales and architectures, including three not used during critic training. On the MuSiQue benchmark, the trained critic improves Qwen3-4B's success rate by more t
    
[^135]: 基于数据流的主动学习与协同神经网络实现数据高效的部分逆向设计：汽车玻璃导槽案例研究

    Stream-Based Active Learning with Cooperative Neural Networks for Data-Efficient Partial Inverse Design: An Automotive Glass Run Channel Case Study

    [https://arxiv.org/abs/2610.09848](https://arxiv.org/abs/2610.09848)

    该研究提出CoNN-AL框架，将基于数据流的主动学习与协同神经网络-去噪自编码器相结合，利用蒙特卡洛dropout估计预测不确定性并实时筛选最有价值的仿真样本进行标注，在汽车玻璃导槽的部分逆向设计中以远小于90多万总样本的标注量实现了数据高效的逆向设计。

    

    工程中的逆向设计常常面临一个简单的问题：每个带标签的训练样本都必须通过昂贵的仿真来生成，因此构建大型数据集既缓慢又成本高昂。本研究针对部分逆向设计中的这一问题展开，即仅指定部分设计变量，其余变量需通过推断来确定以达到目标性能值。我们提出了CoNN-AL，这是一个数据高效的部分逆向设计框架，它在带去噪自编码器的协同神经网络（CoNN-DAE）基础上加入了基于数据流的主动学习。该模型通过蒙特卡洛dropout估计预测不确定性，并利用其实时判断哪些新传入的候选样本值得标注，从而使有限的标注预算花在信息量最大的设计上。我们在一个包含超过90万个独特仿真设计的真实汽车玻璃导槽数据集上验证了该框架。仅使用2万个主动选择的标签……

    arXiv:2610.09848v1 Announce Type: new  Abstract: Inverse design in engineering often runs into a simple problem. Each labeled training sample must be produced through expensive simulation, so building a large dataset is slow and costly. This study addresses that problem for partial inverse design, where only some design variables are specified and the rest must be inferred to reach a target performance value. We propose CoNN-AL, a framework for data-efficient partial inverse design that adds stream-based active learning to the Cooperative Neural Network with Denoising Autoencoder (CoNN-DAE). The model estimates predictive uncertainty through Monte Carlo dropout and uses it to decide, in real time, which incoming candidate samples are worth labeling, so the limited labeling budget is spent on the most informative designs. We validate the framework on a real-world automotive glass run channel dataset of more than 900,000 unique simulated designs. With only 20,000 actively selected labels
    
[^136]: 致信奉忠实性的人们：优化插入与删除曲线下面积以对相对特征重要性进行排序

    For Those Who Believe in Faithfulness: Optimizing the Area Under Insertion and Deletion Curves for Ranking Relative Feature Importance

    [https://arxiv.org/abs/2610.09844](https://arxiv.org/abs/2610.09844)

    本文从忠实性概念（插入与删除曲线下面积）出发推导出目标函数并通过随机化高效近似其梯度，同时建立了插入曲线与top-k特征选择之间的联系，从而能够直接优化特征重要性归因的质量。

    

    将机器学习应用于与社会相关的任务时，需要有效的可解释人工智能（XAI）方法来更好地理解机器学习模型的行为。归因方法是一种流行的XAI方法，它通过热力图来刻画输入与输出之间的关系，这些热力图反映了输入特征对于特定预测的相对重要性。此类热力图的质量通常通过基于插入曲线和删除曲线下面积来衡量的忠实性来评估，该指标衡量了随着特征的添加和移除模型输出所发生的变化。在本研究中，我们从这种忠实性概念出发推导出一个目标函数，并提出了一种近似其梯度的方法。我们建立了插入曲线与top-k特征选择之间的联系，由此得到了一个衡量归因质量的损失函数。通过对该损失函数进行随机化，我们能够高效地近似其梯度。为了展示……（原文摘要在此处被截断）

    arXiv:2610.09844v1 Announce Type: new  Abstract: The adoption of machine learning for socially relevant tasks requires effective explainable artificial intelligence (XAI) methods to better understand the behavior of machine learning models. Attribution methods are a popular XAI approach in which input-output relationships are characterized by heat maps that reflect the relative importance of input features for a particular prediction. The quality of such maps is often assessed by measuring faithfulness based on the area under insertion and deletion curves, which measures changes in the model output as features are added and removed. In this study, we derive an objective function from this notion of faithfulness and a way to approximate its gradient. We establish the connection between insertion curves and top-$k$ feature selection, which leads to a loss function measuring the quality of attributions. Randomization of the loss allows us to efficiently approximate its gradient. To show t
    
[^137]: ORCA：捕捉文本到图像扩散模型中的组合性失败

    ORCA: Hunting Compositional Failures in Text-to-Image Diffusion

    [https://arxiv.org/abs/2610.09841](https://arxiv.org/abs/2610.09841)

    提出ORCA方法，将跨模态组合结构对齐作为低秩辅助损失直接融入扩散模型训练，从而解决文本到图像生成中的属性绑定错误、空间关系颠倒和多对象计数失败等组合性问题。

    

    文本到图像扩散模型在组合性提示上会出现可预测的失败：属性绑定到错误的对象上、空间关系发生颠倒、多对象场景中对象数量出错。近期的架构之所以用T5编码器来增强CLIP，正是因为CLIP的对比嵌入丢失了组合结构，然而这些失败依然存在。我们认为，绑定问题的根源因此不在于信息缺失，而在于信息错位：文本编码器虽然保留了组合结构，但其表示空间是由语言建模而非视觉塑造的，而去噪目标并没有直接奖励将两者对齐。我们证明这种对应关系可以作为显式的训练信号来提供，相关的跨模态信息集中在自监督视觉特征的一个低秩子空间中，并且提供这一信号可以作为单一的辅助损失融入扩散模型训练中。我们的方法ORCA（Orthog……）（摘要原文在此处截断）

    arXiv:2610.09841v1 Announce Type: cross  Abstract: Text-to-image diffusion models fail predictably on compositional prompts: attributes bind to the wrong objects, spatial relations invert, and multi-object scenes lose count. Recent architectures already augment CLIP with a T5 encoder precisely because CLIP's contrastive embedding loses compositional structure, yet these failures persist. We argue the binding problem is therefore not one of missing information but of misaligned information: a text encoder preserves compositional structure, but in a representation space shaped by language modelling rather than vision, and the denoising objective does not directly reward aligning the two. We show this correspondence can be supplied as an explicit training signal, that the relevant cross-modal information is concentrated in a low-rank subspace of self-supervised visual features, and that supplying it can be folded into diffusion training as a single auxiliary loss. Our method, ORCA (Orthog
    
[^138]: 完全可解释的极简Transformer：从几何到算法

    Fully Interpretable Minimal Transformers: From Geometry to Algorithm

    [https://arxiv.org/abs/2610.09838](https://arxiv.org/abs/2610.09838)

    通过将Transformer的嵌入维度和注意力头大小限制为2，本文实现了内部表示的完全二维可视化，证明学习到的几何结构可直接解读为逐步执行的算法，并据此完整解析了一个完成“遇到+号输出最近偶数”任务的极简Transformer的每一步计算过程。

    

    我们提出了一个用于构建和解释极简Transformer模型的框架。通过将Transformer的嵌入维度和注意力头大小限制为2，我们实现了对其内部表示的完整二维可视化。嵌入、查询/键/值变换、注意力输出、残差流和决策边界都可以被直接观察到。我们的核心主张是：学习到的几何结构蕴含着一种算法；R²空间中点和边界的排列可以被解读为一个逐步执行的程序。我们在一个简单任务上训练了一个Transformer：当数字序列中出现'+'运算符时，模型必须输出最近观察到的偶数。训练完成后，我们对Transformer计算的每一个步骤进行可视化走查。我们展示了模型如何嵌入词元及其在序列中的位置，如何通过Q、K、V矩阵对其进行变换，以及如何利用Q和K表示之间的点积……

    arXiv:2610.09838v1 Announce Type: new  Abstract: We present a framework for building and interpreting minimal transformer models. By constraining a transformer's embedding dimension and head size to 2, we enable full two-dimensional visualization of its internal representations. Embeddings, query/key/value transforms, attention outputs, residual streams, and decision boundaries can all be seen directly. Our central claim is that the learned geometry implies an algorithm; the arrangement of points and boundaries in R^2 can be read as a step-by-step procedure. We train a transformer on a simple task where it must produce the most recently observed even number whenever the '+' operator appears in a sequence of digits. Once trained, we visually walk through every step of the transformer's computation. We show how the model embeds the tokens and their respective positions in the sequence, transforms them via the Q, K, and V matrices, uses the dot product between the Q and K representations 
    
[^139]: 多组分材料中通用机器学习力场误差的起源

    Origins of Universal Machine Learning Force-Field Errors in Multicomponent Materials

    [https://arxiv.org/abs/2610.09837](https://arxiv.org/abs/2610.09837)

    该论文构建了包含7,599个多组分构型的基准测试集，系统评估了11个预训练通用机器学习力场在多组分材料中的表现，并揭示了训练参考覆盖率不足和局部几何异质性增大是力场误差的主要来源。

    

    通用机器学习力场对由成分设计生成的多组分环境的泛化能力仍缺乏充分评估。我们构建了一个包含7,599个多组分构型的基准测试集，其灵感来源于高熵设计、元素取代和阴离子混合。我们以密度泛函理论为参照，对11个预训练模型在能量、力和应力方面的表现进行了评估，并将评估扩展至弹性、振动和吸附相关性质。我们通过训练参考覆盖率、局部几何异质性、距离方向性和元素响应来分析力误差。到训练参考环境的距离揭示了覆盖差异与误差增加之间的定性关联，但在相似距离下仍存在显著变化。误差较高的组表现出更大的局部几何异质性，尽管OMat24对这些环境提供了广泛的覆盖。相对……

    arXiv:2610.09837v1 Announce Type: cross  Abstract: Universal machine learning force-field generalization to multicomponent environments generated by compositional design remains insufficiently assessed. We construct a benchmark of 7,599 multicomponent configurations inspired by high-entropy design, elemental substitution and anion mixing. Eleven pretrained models are evaluated against density functional theory for energies, forces and stresses, with assessment extended to elastic, vibrational and adsorption-related properties. Force errors are analysed through training-reference coverage, local geometric heterogeneity, distance directionality and elemental response. Distances to training-reference environments reveal a qualitative association between coverage differences and increasing errors, while substantial variation remains at similar distances. Higher-error groups show greater local geometric heterogeneity, although OMat24 provides broad coverage of these environments. Relative t
    
[^140]: Dual-QK：面向可剪枝2比特KV缓存的锐化查询与平坦化键

    Dual-QK: Sharp Queries and Flat Keys for Prunable 2-bit KV Caches

    [https://arxiv.org/abs/2610.09827](https://arxiv.org/abs/2610.09827)

    Dual-QK通过成对的非正交查询与键变换，将键能量平坦化以实现2比特量化，同时将查询能量锐化集中以支持动态通道剪枝，解决了传统旋转量化中查询能量分散与剪枝需求之间的冲突。

    

    长输入和扩展生成会增加键值（KV）缓存的存储与访问成本。低比特量化可以减少存储和内存流量，而查询通道剪枝可进一步减少键缓存的读取量。基于旋转的量化会将键的离群值能量重新分布到各通道上。为了保持计算不变性，必须对查询应用相同的正交变换，以保持查询-键点积不变。然而，这种旋转会分散查询能量，削弱少数应保留的大分量与多数应剪枝的小分量之间的可分性。我们提出Dual-QK，利用成对的非正交查询与键变换来解决这一冲突。借助校准得到的查询和键统计量，Dual-QK将部分键白化与查询对齐基相结合，在INT2量化时平衡键的尺度，同时集中查询能量以实现动态通道剪枝。通道0保护和桶相对RoPE（原文在此处截断）……

    arXiv:2610.09827v1 Announce Type: new  Abstract: Long inputs and extended generation increase the storage and access costs of the key-value (KV) cache. Low-bit quantization reduces storage and memory traffic, while query-channel pruning can further reduce key-cache reads. Rotation-based quantization redistributes the energy of key outliers across channels. To maintain computational invariance, the same orthogonal transform must be applied to queries, preserving query-key dot products. However, this rotation can disperse query energy, weakening the separation between a few large components to retain and many small ones to prune. We introduce Dual-QK, which uses paired non-orthogonal query and key transforms to address this conflict. Using calibrated query and key statistics, Dual-QK combines partial key whitening with a query-aligned basis to balance key scales for INT2 quantization and concentrate query energy for dynamic channel pruning. Channel-0 protection and bucket-relative RoPE s
    
[^141]: 多智能体系统中的同质化

    Homogenization in Multi-Agent Systems

    [https://arxiv.org/abs/2610.09824](https://arxiv.org/abs/2610.09824)

    本文首次揭示了多智能体系统中的智能体交互会导致行为同质化，并提出三个量化指标，证明这种同质化会在代码生成、招聘和同行评审中引发系统性漏洞、偏见持续影响和评估标准不均等具体风险。

    

    多智能体系统（MAS）利用智能体之间的交互来执行复杂任务。尽管取得了成功，我们证明这些交互也可能导致同质化，即智能体趋同于相似的行为。MAS中的同质化会降低智能体的多样性并强化共同的失败。在本文中，我们使用三个指标来量化同质化：对多数派的趋从、向极端的极化，以及在后续交互中对变化日益增长的惰性。我们在代码生成、招聘和科学同行评审任务中评估了MAS中的同质化。在这些任务中，我们表明同质化会转化为具体的下游风险：在代码生成中，它会隐藏并放大相关错误，从而造成系统性漏洞；在招聘中，它使有偏见智能体的影响在其被移除后仍长期存在；在同行评审中，它导致不同研究领域之间的评估标准不均衡。

    arXiv:2610.09824v1 Announce Type: new  Abstract: Multi-agent systems (MAS) leverage interactions between agents to perform complex tasks. Despite their success, we show that these interactions can also lead to homogenization, i.e., agents converging to similar behaviors. Homogenization in MAS can reduce agent diversity and reinforce shared failures. In this paper, we operationalize homogenization using three metrics: conformity to the majority, polarization towards extremes, and growing inertia against changes over subsequent interactions. We evaluate homogenization in MAS for code generation, hiring, and scientific peer review. Across these tasks, we show that homogenization translates to concrete downstream risks: in code generation, it hides and amplifies correlated errors which can create systemic vulnerabilities; in hiring, it allows the influence of biased agents to persist long after their removal; and in peer review, it creates uneven evaluation standards across research areas.
    
[^142]: 针对物理可实现触发器的声学基础模型后门攻击

    Backdooring Acoustic Foundation Models for Physically Realizable Triggers

    [https://arxiv.org/abs/2610.09819](https://arxiv.org/abs/2610.09819)

    本文提出FAB后门攻击方法，证明最先进的声学基础模型在实际设置下易被植入物理可实现、隐蔽且无需同步的触发器后门，这些后门能在微调后存活并在激活时显著降低下游任务性能。

    

    声学基础模型（AFMs）已经使声学应用平民化，能够以极少的资源为从语音识别到说话人验证等各类任务构建强大的模型。然而，基于AFMs的应用安全性在很大程度上仍未得到充分探索。我们的工作通过提出基础声学模型后门攻击来填补这一空白，证明最先进的AFMs在实际设置下容易受到后门攻击。尽管我们对攻击者能力做了最少的假设（例如，无需访问预训练数据），我们表明FAB在保持良性性能的同时植入后门，这些后门能够在微调后存活，并在被激活时导致多种下游任务的显著性能下降。值得注意的是，FAB利用了任务无关、物理可实现、隐蔽且无需同步的触发器（例如背景警报声）。我们使用两个领先的AFMs、九个下游任务和四种不同的设置对FAB进行了评估。

    arXiv:2610.09819v1 Announce Type: cross  Abstract: Acoustic foundation models (AFMs) have democratized acoustic applications, enabling powerful models for tasks ranging from speech recognition to speaker verification with minimal resources. However, the security of applications based on AFMs remains largely underexplored. Our work addresses this gap by proposing the Foundation Acoustic model Backdoor (FAB) attack, demonstrating that state-of-the-art AFMs are susceptible to backdooring under practical settings. Despite making minimal assumptions about adversary capabilities (e.g., no access to pre-training data), we show that FAB preserves benign performance while inducing backdoors that survive fine-tuning and cause significant degradation across diverse downstream tasks when activated. Notably, FAB utilizes task-agnostic, physically realizable, inconspicuous, and sync-free triggers (e.g., a background siren). We evaluate FAB using two leading AFMs, nine downstream tasks, and four diff
    
[^143]: BoT-GRPO：基于词袋聚合的高效过程奖励强化学习推理方法

    BoT-GRPO: Efficient Process-Reward RL for Reasoning via Bag-of-Token Aggregation

    [https://arxiv.org/abs/2610.09804](https://arxiv.org/abs/2610.09804)

    BoT-GRPO通过长度不变的“词袋”聚合将token级过程奖励高效融入GRPO，无需价值网络即可作为即插即用替代方案，将收敛速度最高提升1.9倍并改善最终生成质量。

    

    强化学习如今已成为激发大语言模型推理能力的核心手段，然而在流行的组相对策略优化算法中，一次采样中的每个token都获得相同的优势值。我们研究如何使过程监督更加高效：在不引入价值网络开销的前提下加速收敛并提升最终质量。我们提出词袋组相对策略优化，通过一种长度不变的“词袋”聚合方式将GRPO扩展至token级奖励模型：它收集所有采样中的token级奖励，按其源序列长度的倒数对每个奖励进行加权，并相对于加权组统计量计算每个token的优势。BoT-GRPO无需评论家网络，在token级奖励可用时可作为GRPO的即插即用替代方案。在React前端代码生成任务上，BoT-GRPO以最高1.9倍于GRPO的速度达到80%的编译通过率，且收敛更快……

    arXiv:2610.09804v1 Announce Type: new  Abstract: Reinforcement learning is now central to eliciting reasoning in large language models, while in the popular algorithm Group Relative Policy Optimization (GRPO) every token in a rollout receives the same advantage. We ask how to make process supervision efficient: accelerating convergence and improving final quality without the cost of value networks. We propose Bag-of-Tokens Group Relative Policy Optimization (BoT-GRPO), which extends GRPO to token-level reward models through a length-invariant "bag of tokens" aggregation: it collects all token-level rewards across rollouts, weights each by the inverse of its source sequence length, and computes per-token advantages relative to weighted group statistics. BoT-GRPO is critic-free, and is a drop-in replacement wherever GRPO is used when token-level reward is available. On React front-end code generation, BoT-GRPO reaches $80\%$ compile rate up to $1.9\times$ faster than GRPO and converges f
    
[^144]: DisParQ：面向可解释视觉基础模型的自监督部件概念

    DisParQ: Self-Supervised Part Concepts for Interpretable Vision Foundation Models

    [https://arxiv.org/abs/2610.09802](https://arxiv.org/abs/2610.09802)

    DisParQ提出了一种无需类别标签和语言监督的自监督方法，通过可学习原型字典将图像块离散化为空间锚定的部件概念，并结合量化属性建模概念变化，从而实现可解释的视觉基础模型。

    

    基于概念的视觉模型通过一个人类可检查的中间概念层来表示图像，因此模型所依赖的内容可以追溯到这些概念。然而，这类模型通常局限于固定的类别，或者依赖语言来定义其概念。我们提出DisParQ（带量化属性的离散部件），一种从强大的冻结视觉自监督骨干网络中学习空间锚定的离散概念表示的方法。它既不需要类别标签，也不需要语言监督。每个图像块被精确地分配给可学习原型字典中的唯一一个概念，且每张图像仅允许稀疏的概念子集被激活。为了捕捉每个概念在不同图像间的变化方式（例如“车轮”的类型），我们在概念之外学习连续的残差，并将其量化为离散属性。一个空间解码器从这些概念出发重建骨干网络的表示

    arXiv:2610.09802v1 Announce Type: cross  Abstract: Concept-based vision models represent images through an intermediate layer of human-inspectable concepts, so what a model relies on can be traced to those concepts. However, those models are often limited to fixed categories or depend on language to define their concepts. We introduce DisParQ (Discrete Parts with Quantized attributes), a method that learns spatially grounded, discrete concept representations from a powerful frozen vision-only self-supervised backbone. It requires no class labels and no language supervision. Each image patch is assigned to exactly one concept from a learnable prototype dictionary, and only a sparse subset of concepts may activate per image. To capture how each concept varies across images (e.g., the type of a "wheel"), we learn continuous residuals alongside the concepts and then quantize them into discrete attributes. A spatial decoder reconstructs the backbone's representation from the concepts and at
    
[^145]: 一种用于伪迹抑制的细粒度脑电成分弱监督标注的概念验证研究

    A Proof-of-Concept Study of Weakly Supervised Labeling of Fine-Grained EEG Components for Artifact Attenuation

    [https://arxiv.org/abs/2610.09792](https://arxiv.org/abs/2610.09792)

    提出了一种结合频率感知高维表示与多示例学习的弱监督框架，无需昂贵的成分级专家标注，即可为脑电分离成分的细粒度子成分学习伪迹可能性得分，从而实现EMG伪迹的有效抑制。

    

    脑电图（EEG）极易受到肌电（EMG）伪迹的干扰，这些伪迹的时间异质性以及与神经活动在空间和频谱上的重叠，可能导致盲源分离后仍残留混合源。现有伪迹去除方法进一步受限于可靠的成分级真实标注的稀缺：专家标注成本高昂且具有主观性，而目前尚无成熟方法能为多通道头皮脑电中的EMG污染提供逼真的基于仿真的真实标注。为解决这些局限，我们提出了一个将频率感知的高维表示与多示例学习相结合的框架。该表示将分离出的成分展开为频率分辨的成分内子成分，构建了一个使混合的神经与肌肉活动更易分离的空间；同时，弱监督学习机制使模型能够从……（摘要在此处被截断）

    arXiv:2610.09792v1 Announce Type: new  Abstract: Electroencephalography (EEG) is highly susceptible to electromyographic (EMG) artifacts, whose temporal heterogeneity and spatial-spectral overlap with neural activity can leave mixed sources after blind source separation. Existing artifact-removal methods are further limited by scarce reliable component-level ground truth: expert annotations are costly and subjective, while no established method provides realistic simulation-based ground truth for EMG contamination in multichannel scalp EEG. To address these limitations, we propose a framework combining a frequency-aware high-dimensional representation with Multi-Instance Learning. The representation unfolds separated components into frequency-resolved intra-components, creating a space in which mixed neural and muscular activity becomes more separable, while the weakly supervised learning formulation enables artifact-likelihood scores for individual intra-components to be learned from 
    
[^146]: AdaPS-LiNGAM：小样本设定下线性非高斯无环模型的自适应前驱选择

    AdaPS-LiNGAM: Adaptive Predecessor Selection for Linear Non-Gaussian Acyclic Models under Small-Sample Settings

    [https://arxiv.org/abs/2610.09782](https://arxiv.org/abs/2610.09782)

    本文揭示了DirectLiNGAM在变量数超过样本量时残差化必然退化的结构性局限，并提出利用由图结构决定的“活动边界”子集进行自适应前驱选择的AdaPS-LiNGAM方法，以实现小样本情形下可靠的因果发现。

    

    当可用样本量相对于变量数量较小时，因果发现变得尤为具有挑战性。这一挑战同样存在于线性非高斯无环模型中，这是一个从观测数据中进行因果发现的可辨识框架。DirectLiNGAM通过依次识别外生变量并从剩余变量中去除其线性效应，来估计一个使原因排在结果之前的因果顺序。我们确立了这一过程的一个结构性局限：当变量数量超过样本量时，反复的残差化必然在完整因果顺序确定之前就变得退化。我们的分析进一步揭示，每个残差都可以仅利用因果顺序中先前已确定变量的一个由图结构决定的子集来重构，该子集被称为“活动边界”。这一结果启发了AdaPS-LiNGAM（自适应前驱选择）方法。

    arXiv:2610.09782v1 Announce Type: new  Abstract: Causal discovery becomes particularly challenging when the available sample size is small relative to the number of variables. This challenge also arises in the linear non-Gaussian acyclic model (LiNGAM), an identifiable framework for causal discovery from observational data. DirectLiNGAM estimates a causal order, which arranges variables so that causes precede their effects, by sequentially identifying an exogenous variable and removing its linear effect from the remaining variables. We establish a structural limitation of this procedure: when the number of variables exceeds the sample size, repeated residualization necessarily becomes degenerate before the full causal order can be determined. Our analysis further reveals that each residual can be reconstructed using only a graph-determined subset of variables already placed earlier in the causal order, termed the active boundary. This result motivates AdaPS-LiNGAM (Adaptive Predecessor
    
[^147]: 可复现的大语言模型推理基准测试：一种用于回归测试的顺序隔离协议

    Reproducible LLM Inference Benchmarking: A Sequential Isolation Protocol for Regression Testing

    [https://arxiv.org/abs/2610.09778](https://arxiv.org/abs/2610.09778)

    提出了一种顺序隔离基准测试协议，将大语言模型推理测量的变异系数从15.2%降至2.2%，实现了可复现的推理性能回归测试。

    

    大语言模型（LLM）推理的可复现基准测试具有挑战性，因为重复测量结果会随执行过程和系统状态而变化。我们提出了顺序隔离方法，这是一种受控的基准测试与回归测试协议，旨在减少运行间的测量方差，同时有意改变工作负载并发度。我们在一块 NVIDIA A100 80GB GPU 上使用 vLLM 0.9.1 评估了三个具有代表性的开源大语言模型，涵盖六种上下文长度和八个并发级别，每个配置重复五次。最终协议将平均变异系数（CV）从控制程度最低的方法阶段的 15.2% 降低到最终协议下的 2.2%；若采用每个配置五次重复的首 token 延迟中位数（P50 TTFT）计算的变异系数，144 个配置中有 113 个（78.5%）实现了低于 3% 的变异系数。测量结果还显示，在 200 至 500 个并发用户之间存在明显的延迟转变……

    arXiv:2610.09778v1 Announce Type: cross  Abstract: Reproducible benchmarking of Large Language Model (LLM) inference is challenging because repeated measurements can vary with execution and system state. We present the Sequential Isolation Methodology, a controlled benchmarking and regression-testing protocol designed to reduce between-run measurement variance while deliberately varying workload concurrency. We evaluate three representative open-source LLMs on an NVIDIA A100 80GB GPU using vLLM 0.9.1 across six context sizes and eight concurrency levels, with five repetitions per configuration. The final protocol reduces average coefficient of variation (CV) from 15.2% in the least controlled methodology stage to 2.2% under the final protocol; using CV computed across the five repetition-level median (P50) TTFT values per configuration, 113 of 144 configurations (78.5%) achieve CV below 3%. The measurements also show a marked latency transition between 200 and 500 concurrent users on t
    
[^148]: 人工智能从天气到气候的发展路径

    Artificial intelligence pathways from weather to climate

    [https://arxiv.org/abs/2610.09770](https://arxiv.org/abs/2610.09770)

    该论文回顾了深度学习在天气预报领域的突破性进展，并提出将其拓展至气候预测的两项最低要求：外部强迫因子须显式进入模型以支持干预实验，且必须在极端事件和反事实轨迹等分布外情景中检验模型稳健性。

    

    深度学习在天气预报领域取得了快速进展：基于大气再分析数据训练的自回归模型，在临近预报、中期预报以及次季节到季节时间尺度上已可与传统动力模式相媲美，并能以更低的成本产生校准良好的集合预报。我们回顾了这些进展，并探讨了将其拓展至气候时间尺度的可能性。在气候尺度上，挑战从初始条件预测技巧转变为在变化的强迫条件下产生可靠的统计响应。人工智能驱动的气候预测系统必须对通常超出观测记录范围的外部驱动因子（如温室气体排放、土地利用变化）产生可信的强迫响应。我们为气候建模中的人工智能提出了两项最低要求：（i）外部强迫因子必须以足够明确的方式进入模型，以支持强迫因子独立变化的干预实验；（ii）必须在分布外情景（包括极端事件和反事实轨迹）中对模型的稳健性进行压力测试。

    arXiv:2610.09770v1 Announce Type: cross  Abstract: Deep learning has made rapid advances in weather forecasting: autoregressive models trained on atmospheric reanalyses now rival dynamical models across nowcasting, medium-range, and subseasonal-to-seasonal lead times, producing well-calibrated ensemble forecasts at reduced cost. We review these advances and consider their extension to climate horizons, where the challenge shifts from initial-condition skill to producing reliable statistical responses under altered forcings. AI-powered climate prediction systems must produce credible forced responses to drivers (e.g., greenhouse gases, land-use change) typically outside the observed record. We propose two minimum requirements for AI in climate modeling: (i) external forcing agents must enter explicitly enough to support interventions in which they vary independently; and (ii) robustness must be stress-tested in out-of-distribution regimes, including extremes and counterfactual trajector
    
[^149]: 均场神经网络训练中非线性观测量的涨落

    Fluctuations of Nonlinear Observables in Mean Field Neural Network Training

    [https://arxiv.org/abs/2610.09768](https://arxiv.org/abs/2610.09768)

    本文通过在加权Sobolev空间中应用仅需普通Fréchet可微性的泛函Delta方法（无需Lions导数），证明了均场神经网络训练中的涨落会传播到有限维非线性观测量，并建立了相应的中心极限定理与显式协方差表示。

    

    均场极限通过参数经验分布的演化来描述宽神经网络的训练动力学。尽管泛函中心极限定理刻画了该分布的渐近涨落，但实际关心的量通常是参数分布的非线性观测量而非分布本身。在这项工作中，我们展示了这些均场涨落如何传播到由随机梯度下降训练的浅层神经网络的有限维非线性观测量上。我们在构建极限涨落过程的加权Sobolev空间中进行研究，在普通Fréchet可微性条件下应用泛函Delta方法，而无需对测度变量使用Lions导数。我们得到了这些观测量的中心极限定理，并在其微分的适当表示下，得到了显式的协方差……

    arXiv:2610.09768v1 Announce Type: new  Abstract: Mean field limits describe the training dynamics of wide neural networks through the evolution of the empirical distribution of their parameters. Although functional central limit theorems characterize the asymptotic fluctuations of this distribution, quantities of practical interest are typically nonlinear observables of the parameter distribution rather than the distribution itself. In this work, we show how these mean field fluctuations propagate to finite dimensional nonlinear observables for shallow neural networks trained by stochastic gradient descent. Working in the weighted Sobolev space in which the limiting fluctuation process is constructed, we apply a functional Delta method under ordinary Fr{\'e}chet differentiability, without requiring Lions derivatives with respect to the measure variable. We obtain a central limit theorem for the observables and, under a suitable representation of their differentials, an explicit covaria
    
[^150]: 超越策略支持：面向自动驾驶的交互约束离线强化学习

    Beyond Policy Support: Interaction Constrained Offline Reinforcement Learning for Autonomous Driving

    [https://arxiv.org/abs/2610.09763](https://arxiv.org/abs/2610.09763)

    该论文发现了自动驾驶离线强化学习中新型的“交互分布偏移”（IDS）问题——即自车候选轨迹在边缘行为分布下支持良好但与周围智能体行为联合考虑时支持不足——并提出交互约束驾驶策略（ICDP）框架来解决这一问题。

    

    离线强化学习能够在无需在线探索的情况下，从固定数据集中实现基于奖励驱动的策略改进，这使其在安全关键领域极具吸引力。然而，其核心挑战是分布偏移问题：策略优化可能倾向于选择离线数据中支持较弱的动作，导致价值估计不可靠。现有方法主要在策略自身的动作空间中控制这种偏移。但在自动驾驶等交互环境中，这可能是不够的：一条候选自车轨迹可能在边缘行为分布下得到良好支持，但与日志交互中观测到的周围智能体行为进行联合考量时却支持不足。我们将这种交互支持的退化称为“交互分布偏移”（IDS），并提出了“交互约束驾驶策略”（ICDP），这是一种离线强化学习框架。

    arXiv:2610.09763v1 Announce Type: new  Abstract: Offline reinforcement learning enables reward-driven policy improvement from fixed datasets without requiring online exploration, making it particularly attractive in safety-critical domains. A central challenge, however, is distribution shift: policy optimization may favor actions that are weakly supported by the offline data, rendering value estimates unreliable. Existing approaches primarily control this shift in the policy's own action space. In interactive environments such as autonomous driving, this can be insufficient: a candidate ego trajectory may remain well supported under the marginal behavior distribution while being poorly supported jointly with the surrounding-agent behavior observed in the logged interaction. We refer to this degradation in interaction support as \emph{interaction distribution shift} (IDS), and introduce \emph{Interaction-Constrained Drive Policy} (ICDP), an offline reinforcement learning framework that 
    
[^151]: 更精简的Transformer可以轻松学会聚类

    Leaner Transformers Can Easily Learn to Cluster

    [https://arxiv.org/abs/2610.09760](https://arxiv.org/abs/2610.09760)

    本文提出了一种嵌入维度仅需 d+⌈log₂k⌉ 但表达能力不变的更精简Transformer来执行k均值聚类的Lloyd算法，并系统刻画了训练Transformer学习聚类算法时影响收敛性与泛化能力的关键因素。

    

    Transformer具备上下文学习能力，某些已知的学习算法可以在模型的前向传播过程中执行。近期的研究表明，Transformer可以精确执行Lloyd算法，对d维空间中的n个点进行k均值聚类，其嵌入维度为 d_emb = d+k（因此需要大小为(d+k)²的注意力投影矩阵）。在本工作中，我们在这一结果的基础上进行了以下拓展：首先，我们提出了一种表达能力相同但规模更小的Transformer，它能以嵌入维度 d_emb = (d + ⌈log₂ k⌉) 执行Lloyd算法。其次，我们在给定聚类任务分布的条件下训练这些Transformer学习聚类算法，并从理论上刻画和通过实验验证了影响基于随机梯度的学习算法收敛性和分布内泛化能力的因素。最后，我们探究了一般性的聚类能力（摘要此处被截断）。

    arXiv:2610.09760v1 Announce Type: new  Abstract: Transformers have in-context learning capabilities, where some known learning algorithms can be executed in the forward pass through the model. Recent work shows that transformers can exactly perform Lloyd's algorithm for $k$-means clustering with $n$ points in $d$ dimensions with an embedding size $d_{\textsf{emb}} = d+k$ (thus, requiring attention projection matrices of size $(d+k)^2$). In this work, we build upon this result in the following ways: First, we present an equally expressive but smaller transformer that executes Lloyd's algorithm with embedding size $d_{\textsf{emb}} = (d + \lceil \log_2 k \rceil)$. Next, we train these transformers to learn the clustering algorithms given a distribution of clustering tasks, and theoretically characterize and empirically validate the factors affecting the convergence and in-distribution generalization of learning algorithms based on stochastic gradients. Finally, we probe the general clust
    
[^152]: 用于推理的展开流模型

    Unrolled Flow Models for Reasoning

    [https://arxiv.org/abs/2610.09759](https://arxiv.org/abs/2610.09759)

    提出通过模型自身潜在展开进行训练的展开流模型，使得积分步数的增加能够持续提升推理性能，在ProsQA上准确率达97%。

    

    流匹配技术能够以少量步骤实现语言生成，但额外的积分步骤是否能提升推理能力仍不明确。我们证明了一个由两层Transformer参数化的流可以解决图可达性问题，且所需的积分步数会随目标节点与根节点距离的增加而增加。然而，标准的流语言模型在推理任务上往往无法从额外步骤中获益。我们将这一局限归因于现有的训练目标对每个时间点进行独立监督，而没有显式地训练后续步骤在前序步骤的基础上逐步构建。为解决这一问题，我们改为通过模型自身在[0,1]区间上随机采样的子区间内进行潜在展开（latent rollout）来训练，且仅在端点处解码。在ProsQA上，该方法将准确率提升至97%，并使性能能够随着额外积分步数的增加而持续改善。对于数独和迷宫等推理任务所需的更长展开过程，将潜在状态收缩到……（摘要在此处被截断）

    arXiv:2610.09759v1 Announce Type: new  Abstract: Flow matching enables language generation in few steps, but whether additional integration steps improve reasoning remains unclear. We prove that a flow parameterized by a two-layer Transformer can solve graph reachability, with the required number of integration steps increasing with the target's distance from the root. Yet, standard flow language models can fail to benefit from additional steps on reasoning tasks. We attribute this limitation to objectives that supervise each time point independently, without explicitly training successive steps to build on one another. To address this, we instead train through the model's own latent rollout over a randomly sampled subinterval of [0, 1], decoding only at the endpoint. On ProsQA, this raises accuracy to 97% and enables performance to improve with additional integration steps. For the longer rollouts required by reasoning tasks such as Sudoku and Maze, retracting the latent state onto a 
    
[^153]: EntroPrefill：基于 Renyi 引导的上下文剪枝与条件稳定性保证的检索增强生成方法

    EntroPrefill: Renyi-Guided Context Pruning with Conditional Stability Guarantees for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.09757](https://arxiv.org/abs/2610.09757)

    该论文提出 EntroPrefill，一种由 Renyi 熵引导、带显式注意力质量约束的预填充中期上下文剪枝方法，为检索增强生成提供了可计算的 token 删除上界、自适应剪枝层下依然有效的有限样本观测保证，以及带有显式 Lipschitz 常数的条件 Transformer 扰动稳定性界。

    

    预填充（prefill）中期的剪枝可以减少 Transformer 更深层所需处理的序列长度，但仅凭注意力集中并不能证明被丢弃的上下文是可有可无的。我们将 EntroPrefill 构建为一个由 Renyi 熵引导的提议机制，并与对被丢弃注意力质量的显式约束相结合。通过隔离注意力汇聚点（sink）的正则化头池化，该方法在兼容分组查询注意力（GQA）的同时，揭示了头特化性与最差头覆盖率之间的定量权衡。我们推导了从混合分布到注意力头的删除包络——一个可计算的 token 可移除上界，以及一个在自适应选择剪枝层时依然有效的有限样本观测保证。随后，我们建立了带有显式充分 Lipschitz 常数的条件 Transformer 扰动界，以及一个关于首个 token 决策间隔的推论。一个反例表明，仅凭浅层观测无法推出对未条件化未来输出的无条件保证。系统分析区分……（原文在此截断）

    arXiv:2610.09757v1 Announce Type: new  Abstract: Mid-prefill pruning can reduce the sequence processed by deeper transformer layers, but attention concentration alone does not certify that discarded context is dispensable. We formulate EntroPrefill as a Renyi-guided proposal mechanism coupled to explicit constraints on discarded attention mass. Sink-isolated, regularized head pooling respects grouped-query attention while exposing a quantitative trade-off between specialization and worst-head coverage. We derive a mixture-to-head deletion envelope, a computable upper bound on feasible token removal, and a finite-sample observer guarantee that remains valid when the pruning layer is selected adaptively. We then establish a conditional transformer perturbation bound with explicit sufficient Lipschitz constants and a first-token decision-margin corollary. A counterexample shows why shallow observations alone cannot imply an unconditional future-output guarantee. The systems analysis disti
    
[^154]: SoftSEEPS改进了基于机器学习的降水预报

    SoftSEEPS improves ML-based precipitation forecasting

    [https://arxiv.org/abs/2610.09752](https://arxiv.org/abs/2610.09752)

    本文提出了SEEPS评分的可微近似SoftSEEPS，使机器学习降水预报模型能够直接以该评分进行训练，且与RMSE联合优化时几乎不影响两个指标的表现。

    

    在本文中，我们开发了著名SEEPS评分的可微近似版本，并将其命名为SoftSEEPS。这使得机器学习模型能够直接以该评分作为目标进行训练来预报降水。我们在IMERG数据集（0.1度分辨率）上测试了SoftSEEPS，方法是在预训练的低分辨率预报模型的潜在空间上训练一个降水解码器。将SoftSEEPS与RMSE结合在联合目标函数中是可行的，且在两个指标上仅产生微小的性能权衡。

    arXiv:2610.09752v1 Announce Type: new  Abstract: In this paper we have developed a differentiable approximation of the well-known SEEPS score, which we name SoftSEEPS. This allows the training of a Machine Learning model to forecast precipitation directly. We test SoftSEEPS on the IMERG dataset (0.1 degree resolution) by training a decoder for precipitation on the latent space of a pre-trained low-resolution forecasting model. Combining SoftSEEPS and RMSE in a joint objective is possible with marginal trade-offs in either metric.
    
[^155]: 冻结嵌入生物声学分类中域对齐的强度单调定律

    A Strength-Monotonic Law for Domain Alignment in Frozen-Embedding Bioacoustic Classification

    [https://arxiv.org/abs/2610.09737](https://arxiv.org/abs/2610.09737)

    该论文发现了一条强度单调定律：编码器在目标任务上越强，其跨域泛化越依赖分布对齐（MMD）项、也越受域重平衡采样的损害，据此将强编码器的跨域生物声学分类配方简化为冻结嵌入+轻量探针+交叉熵+单个MMD项+数据增强。

    

    分布对齐何时能帮助冻结的基础模型嵌入在声学域之间实现泛化？针对跨域蚊子物种分类任务，我们报告了一条强度单调定律：编码器在目标任务上越强，其未见域泛化就越依赖于分布对齐（MMD）项，同时也越容易受到域重平衡采样的损害。在四个编码器家族以及编码器内部的HuBERT层扫描（n=8）实验中，重平衡效应与编码器强度完全排序一致（Spearman -1.000），而MMD受益效应在每个编码器流内保持单调，汇总后为-0.857；固定架构而仅改变表示强度，会将重平衡效应从有益翻转至崩溃。该定律具有可操作性：单个MMD项是强编码器上的唯一杠杆，因此我们将该领域的默认配方简化为：冻结的Perch 2.0嵌入、轻量级探针、交叉熵损失、单个MMD项以及输入数据增强。

    arXiv:2610.09737v1 Announce Type: cross  Abstract: When does distribution alignment help a frozen foundation-model embedding generalize across acoustic domains? For cross-domain mosquito-species classification we report a strength-monotonic law: the stronger an encoder is on the target task, the more its unseen-domain generalization relies on a distribution-alignment (MMD) term, and the more it is harmed by domain-rebalanced sampling. Across four encoder families and a within-encoder HuBERT layer sweep (n=8), the rebalancing leg orders exactly with encoder strength (Spearman -1.000), while the MMD-benefit leg is monotonic within each stream and -0.857 pooled; fixing architecture and varying only representation strength flips the rebalancing effect from benefit to collapse. The law is actionable: a single MMD term is the sole lever on a strong encoder, so we reduce the field's default recipe to a frozen Perch 2.0 embedding, a lightweight probe, cross-entropy, one MMD, and input augmenta
    
[^156]: 无界特征核与万能核

    Unbounded Characteristic and Universal Kernels

    [https://arxiv.org/abs/2610.09731](https://arxiv.org/abs/2610.09731)

    本文系统研究了无界核的特征性与万能性等表达能力概念，将针对有界核的成熟理论推广至无界核情形。

    

    核方法是机器学习与统计学中最强大的工具之一，拥有大量成功的应用。其巨大成功源于与每个核相关联的灵活函数类——即其再生核希尔伯特空间（RKHS）——这有助于统计分析，同时也源于其计算上的易处理性以及对众多领域的适用性。多种概念（例如特征性、$L_p$-万能性以及积分严格正定性）刻画了核及其RKHS的表达能力，并在理解核方法的统计性质方面发挥着关键作用；对于有界核而言，这些概念及其相互关系已被充分理解。尽管无界核在过去十年中受到了广泛关注（例如在构造基于核的差异度量和依赖性度量时，如最大均值差异（MMD）、希尔伯特-施密特独立性……

    arXiv:2610.09731v1 Announce Type: cross  Abstract: Kernel methods are among the most powerful tools in machine learning and statistics, with a large number of successful applications. Their immense success stems from the flexible function class associated to each kernel---its reproducing kernel Hilbert space (RKHS)---which facilitates statistical analysis, as well as from their computational tractability and applicability to many domains. Multiple notions (such as characteristic, $L_p$-universal, and integrally strictly positive definite) capture the expressivity of kernels and their RKHSs and play a key role in understanding the statistical properties of kernel methods; these concepts and their relations are well-understood for bounded kernels. Even though unbounded kernels have received significant attention over the past decade (for instance, in the construction of kernel-based discrepancy and dependence measures such as the maximum mean discrepancy, the Hilbert-Schmidt independence
    
[^157]: 面向真实恶意软件信标数据无监督异常检测的帕累托最优量子核选择

    Pareto-optimal quantum kernel selection for unsupervised anomaly detection on real malware beaconing data

    [https://arxiv.org/abs/2610.09717](https://arxiv.org/abs/2610.09717)

    该论文提出一种完全无监督的多目标量子核选择协议，通过同时优化无标签的异常检测质量代理指标（NPD）和与经典核的几何差异（GD），从帕累托前沿中选出量子核，并在真实恶意软件信标检测数据和IQM 20量子比特Garnet处理器上验证了其可击败调优后的经典基线。

    

    量子核方法是机器学习领域实现实用量子优势的主要候选方案，但评估这一潜力需要两个通常被分开报告的量：核在任务上的表现如何，以及其几何结构相对于可用于同一问题的经典核偏离多远。我们提出了一种完全无监督的多目标协议，同时优化归一化伪差异（NPD）——一个用于衡量异常检测质量的无标签代理指标——以及与调优后经典参考核之间的几何差异（GD），并从所得的帕累托前沿中选择模型。我们将该方法应用于真实网络流量中的恶意软件信标检测，采用带有保真度核和投影量子核的单类支持向量机，涵盖四种数据编码，在模拟器和IQM的20量子比特Garnet处理器上进行实验。仅由NPD引导的选择即可找到优于调优后经典基线的保真度核，但……

    arXiv:2610.09717v1 Announce Type: cross  Abstract: Quantum kernel methods are leading candidates for a practical quantum advantage in machine learning, but assessing that potential requires two quantities usually reported separately: how well a kernel performs on the task, and how far its geometry departs from the classical kernels available for the same problem. We introduce a fully unsupervised, multi-objective protocol that optimises simultaneously the normalised pseudo discrepancy (NPD), a label-free proxy for anomaly detection quality, and the geometric difference (GD) to a tuned classical reference kernel, selecting models from the resulting Pareto front. We apply it to malware beaconing detection in real network traffic, using a one-class support vector machine with fidelity and projected quantum kernels over four data encodings, on simulators and on IQM's 20-qubit Garnet processor. NPD-guided selection alone finds a fidelity kernel that beats the tuned classical baseline, but w
    
[^158]: EC-EarthFlow：基于流匹配的全球气候模型日瞬态模拟的概率性仿真

    EC-EarthFlow: Probabilistic emulation of daily transient global climate model simulations with flow matching

    [https://arxiv.org/abs/2610.09715](https://arxiv.org/abs/2610.09715)

    EC-EarthFlow是一个生成式流匹配模型，能以远低于物理模型的计算成本稳定地进行自回归模拟，准确再现全球气候模型EC-Earth3日温度场的日变率、空间格局、年循环和长期趋势。

    

    我们提出了EC-EarthFlow，这是一个生成式流匹配模型，用于模拟物理气候模型EC-Earth3的仿真结果。该模型在EC-Earth3的瞬态模拟数据（1950-2166年，SSP2-4.5情景）上进行训练，根据前几天的温度以及年平均温度来预测未来一天的温度场。预测以自回归方式进行，推演周期从一个月到一个延长的季节。仅使用这一感兴趣的变量，我们就能够以远低于物理模型的计算成本再现EC-Earth3的日变率、空间格局、年循环和长期趋势。我们证明EC-EarthFlow在长时间推理过程中保持稳定，并且能够学习到EC-Earth3模拟中所呈现的物理关系。

    arXiv:2610.09715v1 Announce Type: new  Abstract: We introduce EC-EarthFlow, a generative flow matching model that emulates simulations from the physical climate model EC-Earth3. The model is trained on transient simulations from EC-Earth3 (1950-2166, SSP2-4.5) to predict the day ahead temperature field from the previous days temperature as well as annual mean temperature. Predictions are made auto-regressively with rollout periods of between a month and an extended season. Using only this variable of interest, we are able to reproduce the daily variability, spatial patterns, annual cycle and long-term trend from EC-Earth3 at a substantially lower computational cost than the physical model. We demonstrate that EC-EarthFlow is stable for long inference periods, and that it can learn the physical relationships as simulated in EC-Earth3.
    
[^159]: 面向超立方体状态空间的边界感知强化学习：基于确定性策略梯度方法

    Boundary-aware Reinforcement Learning for Hypercube State Spaces via Deterministic Policy Gradient

    [https://arxiv.org/abs/2610.09712](https://arxiv.org/abs/2610.09712)

    本文针对超立方体上受反射随机微分方程支配的系统，建立了连续时间确定性策略梯度强化学习理论框架，并通过软惩罚或硬结构约束将诺伊曼边界条件嵌入深度确定性策略梯度算法，实现了边界感知的强化学习。

    

    我们为具有反射状态动力学的强化学习开发了一个连续时间确定性策略梯度框架，其中状态过程由超立方体上的受控反射随机微分方程支配。在适当的正则性假设下，我们建立了值函数与诺伊曼（Neumann）贝尔曼方程之间的联系，引入了一种能够导出确定性策略梯度公式的优势速率函数，并证明了鞅特征化定理。基于这些理论结果，我们为反射随机系统提出了一种连续时间深度确定性策略梯度算法，其中诺伊曼边界条件通过软惩罚或硬性结构约束两种方式施加。我们进一步量化了理想连续时间动力学与实践中执行的离散采样探索动力学之间的差异，表明误差会衰减（摘要在此处截断）。

    arXiv:2610.09712v1 Announce Type: cross  Abstract: We develop a continuous-time deterministic policy gradient framework for reinforcement learning with reflected state dynamics, where the state process is governed by a controlled reflected stochastic differential equation on a hypercube. Under suitable regularity assumptions, we establish the connection between the value function and the Neumann Bellman equation, introduce an advantage-rate function that yields a deterministic policy gradient formula, and prove the martingale characterization theorem. Motivated by these theoretical results, we propose a continuous-time deep deterministic policy gradient algorithm for reflected stochastic systems, in which the Neumann boundary condition is imposed via either soft penalization or hard architectural constraint. We further quantify the discrepancy between the ideal continuous-time dynamics and the discretely sampled exploratory dynamics executed in practice, showing that the error decays a
    
[^160]: 预训练塑造谱结构：基础模型分布外鲁棒性的架构与策略条件化预测

    Pretraining Shapes Spectral Structure: Architecture- and Strategy-Conditional Prediction of OOD Robustness in Foundation Models

    [https://arxiv.org/abs/2610.09709](https://arxiv.org/abs/2610.09709)

    该论文证明基础模型的分布外鲁棒性可仅由预训练权重的谱结构预测——无需任何目标数据，且这一预测由架构与预训练策略共同塑造，其代理指标的方向会随架构族不同而反转。

    

    我们能否在没有任何目标数据可用之前，就判断一个基础模型是否具备分布外泛化能力？现有的诊断方法需要源数据或目标数据，这使得它们在目标域出现之前便无法使用；而那些仅依赖权重的方法则将同一种统计量应用于所有架构，无法区分鲁棒的模型与脆弱的模型。我们证明，答案其实编码在预训练权重的谱结构之中。有两种力量共同塑造了这一结构：架构决定了信息如何存储在权重矩阵中，预训练策略则决定了什么样的特性会得到奖励。二者共同确立了一种支配分布外鲁棒性的谱几何。我们证明了分布外准确率差距受源表示集中程度的约束，而仅从预训练权重计算出的一个统计量便可作为该集中程度的代理指标。值得注意的是，该代理指标的方向在不同架构族之间会发生反转。我们将其付诸实践……

    arXiv:2610.09709v1 Announce Type: new  Abstract: Can we determine whether a foundation model will generalize out-of-distribution (OOD) before any target data is available? Existing diagnostics require source or target data, which rules them out before a target domain exists. Those that use the weights alone apply one statistic to every architecture, and do not separate robust models from fragile ones. We show the answer is encoded in the spectral structure of pretrained weights. Two forces shape that structure. Architecture determines how information is stored in weight matrices. Pretraining strategy determines what is rewarded. Together they set a spectral geometry that governs OOD robustness. We prove that the OOD accuracy gap is bounded by how tightly the source representations concentrate. A statistic computed from the pretrained weights alone serves as a proxy for that concentration. The direction of that proxy reverses between architecture families. We operationalize it: the dire
    
[^161]: 理解并缓解视觉语言模型中令牌剪枝引发的安全脆弱性

    Understanding and Mitigating Token-Pruning-Induced Vulnerabilities in VLMs

    [https://arxiv.org/abs/2610.09703](https://arxiv.org/abs/2610.09703)

    本文首次全面评估了视觉语言模型中令牌剪枝机制的安全性，揭示了大多数剪枝策略随剪枝比例增加而降低安全性、而基于查询的压缩在极端剪枝下反而提升安全性的对比现象，并识别出“剪枝诱发的恶意放大”这一全新机制。

    

    令牌剪枝通过移除冗余视觉令牌来加速视觉语言模型，但其对安全性的影响仍未得到充分探索。在这项工作中，我们首次对令牌剪枝机制进行了全面的安全性评估，并发现：大多数剪枝策略会随着剪枝比例的增加而显著降低模型的安全性，而基于查询的压缩则表现出相反的趋势——在高达99.8%的极端剪枝比例下，反而出乎意料地提升了模型安全性。这一鲜明对比引出了一个关键问题：不同的令牌剪枝策略如何重塑模型的安全行为，以及是否有可能在不牺牲加速效果的前提下增强安全性？为回答这一问题，我们识别出一种此前未被认识的机制，称之为“剪枝诱发的恶意放大”：背景令牌的移除会引发一种副作用，迫使模型的注意力坍缩到前景中保留下来的少数恶意锚点上，从而在无意中放大其有毒语义……

    arXiv:2610.09703v1 Announce Type: cross  Abstract: Token-Pruning accelerates Vision-Language Models by removing redundant visual tokens, yet its safety implications remain underexplored. In this work, we present the first comprehensive safety evaluation of Token-Pruning mechanisms and find that: most pruning strategies significantly degrade safety as pruning ratios increase, whereas Query-based Compression shows the opposite, with extreme pruning (up to 99.8%), unexpectedly improves model safety. This sharp contrast prompts a key question: How do different Token-Pruning strategies reshape model safety behavior, and is it possible to enhance safety without sacrificing acceleration? To answer this, we identify an unrecognized mechanism, termed Pruning-Induced Malicious Amplification, where removal of background tokens triggers a side effect: forcing the model's attention to collapse onto a few retained malicious anchors within the foreground, inadvertently amplifying their toxic semantic
    
[^162]: 面向个性化自动化疼痛评估的小样本学习

    Few-Shot Learning for Personalised Automated Pain Assessment

    [https://arxiv.org/abs/2610.09692](https://arxiv.org/abs/2610.09692)

    该研究将小样本学习应用于自动化疼痛评估的个性化，创造性地把从人群级到受试者级的评估转变重新定义为任务域偏移问题，并在三个疼痛数据库上取得了优异的个性化分类性能。

    

    疼痛感知在个体之间存在显著差异，这使得基于人群的分类器难以在数据集中的所有受试者上实现泛化。应对受试者差异的一种方法是训练个性化分类器。在这项工作中，我们评估了小样本学习（元学习的一个子领域）作为自动化疼痛评估中实现个性化的方法。我们将从人群级评估到受试者级评估的转变重新解释为任务域偏移，即观察到的类别保持固定，但目标受试者发生变化。我们在BioVid疼痛数据库、SenseEmotion数据库和PainMonit实验数据集（PMED）上评估了我们的方法，在留一受试者交叉验证协议下，BioVid在二分类和多分类设置中的准确率分别达到85.75%和35.49%，SenseEmotion上的准确率分别达到82.37%和41.88%，PMED上的准确率达到90.47%（该数据集目前仅有二分类基准）。使用样本实现k-shot……

    arXiv:2610.09692v1 Announce Type: new  Abstract: Pain perception varies substantially across individuals, making it difficult for population-based classifiers to generalise across all subjects in a dataset. One way to account for subject variability is to train personalised classifiers. In this work, we evaluate Few-Shot Learning, a sub-area of Meta-Learning, as an approach to personalisation in automated pain assessment. We re-interpret the shift from population-level to subject-level evaluation as a task-domain shift, where the observed classes remain fixed but the target subject changes. We evaluate our method on the BioVid Pain Database, the SenseEmotion Database, and the PainMonit Experimental Dataset (PMED), reaching 85.75% and 35.49% accuracy on BioVid and 82.37% and 41.88% on SenseEmotion in the binary and multi-class settings under a Leave-One-Subject-Out CV protocol respectively, and 90.47% on PMED, for which only a binary benchmark exists. Using samples to implement k-shot c
    
[^163]: 基于限价订单簿的在线做市的最优遗憾界

    Optimal Regret for Online Market Making with Limit Order Book

    [https://arxiv.org/abs/2610.09691](https://arxiv.org/abs/2610.09691)

    本文针对限价订单簿反馈模型下的在线做市问题，通过设计基于双耦合网格的买卖价空间离散化方法并结合Hedge算法，将遗憾界从Õ(T^(2/3))改进至最优的Õ(√T)高概率界。

    

    我们研究做市中的在线学习问题：在每一轮中，做市商需要先发布买入价和卖出价，然后才能观察到市场价格以及到来的交易者的私人估值。在此设定下，Maran等人（2026）提出了一种受限价订单簿启发的反馈模型，其中只有当没有交易发生时，交易者的估值才会被揭示。假设交易者估值从未知分布中独立同分布抽取，而市场价格是对抗性选择的，他们建立了 Õ(T^(2/3)) 的期望遗憾界。在本工作中，我们改进了这一保证，建立了 Õ(√T) 的高概率遗憾界。作为热身，我们首先考虑全反馈设定：我们引入了一种基于两个耦合网格的买卖价空间离散化方法，并将其与 Hedge 算法相结合，以达到所需的遗憾率。基于这些想法，我们随后解决了本质上……（原文摘要到此截断）

    arXiv:2610.09691v1 Announce Type: cross  Abstract: We study online learning in market making, where, at each round, a market maker posts bid and ask prices before observing the market price and the private valuation of an incoming trader. In this setting, Maran et al. 2026 introduce a feedback model motivated by limit order books, in which the trader's valuation is revealed only if no transaction occurs. Assuming that trader valuations are drawn i.i.d. from an unknown distribution while market prices are chosen adversarially, they establish an expected regret bound of $\widetilde{\mathcal{O}}(T^{2/3})$. In this work, we improve upon this guarantee by establishing a high-probability regret bound of $\widetilde{\mathcal{O}}(\sqrt{T})$. As a warm-up, we first consider the full-feedback setting. We introduce a discretization of the bid-ask space based on two coupled grids and combine it with Hedge to achieve the desired regret rate. Building on these ideas, we then address the substantiall
    
[^164]: NAViLoss：一种面向水下导航感知的双残差物理一致性学习目标函数

    NAViLoss: An Underwater Navigation-Aware Dual-Residual Objective for Physics-Consistent Learning

    [https://arxiv.org/abs/2610.09690](https://arxiv.org/abs/2610.09690)

    该论文提出NAViLoss——一种面向水下导航的鲁棒且不确定性感知的双残差目标函数，通过联合惩罚速度估计残差并显式兼顾物理一致性与测量不确定性，改进了基于学习的AUV多普勒测速仪速度估计。

    

    自主水下航行器（AUV）通常依靠由多普勒测速仪（DVL）辅助的惯性导航系统（INS）来实现可靠的水下导航。因此，准确的DVL速度估计对于成功运行至关重要。近期基于学习的方法已展示出更优的DVL速度估计性能，尤其是在测量条件退化的情况下。然而，这些方法的训练目标通常依赖于传统回归损失，而传统回归损失对大残差和受损观测高度敏感。此外，它们没有显式地考虑与底层感知过程相关的物理一致性和测量不确定性。为解决这些局限，本文提出了导航感知损失（NAViLoss），这是一种用于基于学习的AUV速度估计的鲁棒且不确定性感知的目标函数。NAViLoss联合惩罚导航状态中的速度估计残差……

    arXiv:2610.09690v1 Announce Type: cross  Abstract: Autonomous underwater vehicles (AUVs) commonly rely on inertial navigation systems (INS) aided by Doppler velocity logs (DVLs) for reliable underwater navigation. Accurate DVL velocity estimation is therefore essential for successful operation. Recent learning-based methods have demonstrated improved DVL velocity estimation, particularly under degraded measurement conditions. However, their training objectives typically rely on conventional regression losses that are highly sensitive to large residuals and corrupted observations. Additionally, they do not explicitly account for the physical consistency and measurement uncertainty associated with the underlying sensing process. To address these limitations, this paper introduces navigation-aware loss (NAViLoss), a robust and uncertainty-aware objective function for learning-based AUV velocity estimation. NAViLoss jointly penalizes the velocity-estimation residual in the navigation-state
    
[^165]: 轮廓算子：从一维投影中识别低秩测度的可辨识性

    The Silhouette Operator: Identifiability of Low-Rank Measures from One-Dimensional Projections

    [https://arxiv.org/abs/2610.09687](https://arxiv.org/abs/2610.09687)

    本文提出“轮廓算子”框架，证明了适当选取的 2k 个一维投影边缘分布足以唯一识别 $\mathbb{R}^2$ 上任何紧支撑的秩不超过 k 的符号测度，且该数量是最优的、投影方向不能任意选取。

    

    arXiv:2610.09687v1 公告类型：cross 摘要：结构化恢复现象，例如压缩感知中的受限等距性质，已经表明高维对象通常可以从极其低维的线性测量中重建。本工作为 $\mathbb{R}^2$ 上的低秩符号测度建立了一个类似的恢复框架，这里的低秩符号测度定义为可以表示为具有一维因子的乘积测度的有限和的测度。该框架基于一类线性算子，称为“轮廓算子”，它将一个测度映射到固定的有限个一维线性推前（pushforward）。主要结果表明：适当选取的 $2k$ 个投影边缘分布足以识别每一个紧支撑的秩 $\le k$ 的符号测度，且该数量是最优的，同时投影方向不能任意选取。该框架还通过建立充分的…（摘要原文在此处截断）扩展到了更高维的乘积测度之和的情形。

    arXiv:2610.09687v1 Announce Type: cross  Abstract: Structured recovery phenomena, such as restricted isometry properties in compressed sensing, have shown that high-dimensional objects can often be reconstructed from remarkably low-dimensional linear measurements. This work develops an analogous recovery framework for low-rank signed measures on $\mathbb{R}^2$, defined here as measures that can be expressed as finite sums of product measures with one-dimensional factors. The framework is based on linear operators, termed "silhouette operators," that map a measure to a fixed finite collection of one-dimensional linear pushforwards. The main results show that a suitably chosen collection of $2k$ projected marginals suffices to identify every compactly supported rank-$\le k$ signed measure, that this number is optimal, and that the projection directions cannot be chosen arbitrarily. The framework is also extended to higher-dimensional sums of product measures by establishing sufficient co
    
[^166]: CERO：为强化学习后训练在何处与何时分配Rollout

    CERO: Where and When to Allocate Rollouts for RL Post-Training

    [https://arxiv.org/abs/2610.09679](https://arxiv.org/abs/2610.09679)

    该论文提出CERO，一种在线原始-对偶调度器，能够在整个训练周期内协调有限的rollout预算，自适应地决定选择哪些提示词、重访频率以及每轮生成的组数，为强化学习后训练的提示词准入与预算节奏控制提供了带理论保证的高效方案。

    

    arXiv:2610.09679v1 公告类型： new 摘要：面向组相对强化学习的自适应rollout方法通常在每个更新中为提示词分配固定的预算。我们转而研究如何在整个训练周期内协调有限的rollout预算。我们利用累积提示曝光的凹代理效用函数来表述这一问题，并提出了CERO——一种用于提示词准入与预算节奏控制的在线原始-对偶调度器。在我们的实验设置中，每个被接纳的提示词会获得一个固定大小的响应组；而CERO则自适应地决定选择哪些提示词、它们在各轮次中被重新访问的频率，以及每轮生成多少个组。一个紧凑的Fenchel表示将对累积曝光的依赖关系线性化，同时投影在线梯度下降利用奖励变化反馈和预算偏差来更新提示词特定的支撑斜率与共享预算价格。我们为该代理分配目标建立了逐路径的理论保证……

    arXiv:2610.09679v1 Announce Type: new  Abstract: Adaptive rollout methods for group-relative reinforcement learning typically allocate a fixed per-update budget across prompts. We instead study how to coordinate a finite rollout budget over the entire training horizon. We formulate this problem using a concave surrogate utility of cumulative prompt exposure and introduce CERO, an online primal dual scheduler for prompt admission and budget pacing. In our experiments, each admitted prompt receives a fixed-size response group. CERO instead adapts which prompts are selected, how often they are revisited across rounds, and how many groups are generated in each round. A compact Fenchel representation linearizes the dependence on cumulative exposure, while projected online gradient descent updates prompt-specific supporting slopes and a shared budget price using reward-variation feedback and budget deviations. We establish pathwise guarantees for the surrogate allocation objective against fi
    
[^167]: 一个揭示当代自监督异常检测方法局限性的多源超声基准数据集

    A Multi-Source Ultrasound Benchmark Revealing the Limits of Contemporary Self-Supervised Anomaly Detection Methods

    [https://arxiv.org/abs/2610.09677](https://arxiv.org/abs/2610.09677)

    本文提出SADUSI多源超声基准数据集，涵盖广泛的解剖区域、视图和采集协议，并发现当前自监督异常检测方法（尤其是基于重建的扩散方法）在如此多样化的多源超声数据上表现不佳，从而揭示了现有方法的局限性。

    

    自监督异常检测是医学超声领域一种有前景的范式，因为正常图像通常比所有可能病理的详尽标注更容易获取。然而，现有的大多数评估仅局限于单一解剖部位或任务，这使得我们无法确定模型究竟是学习到了鲁棒的正常超声外观概念，还是仅仅学习到了特定数据源的表征。我们提出了SADUSI基准数据集，这是一个多源超声数据集，旨在跨广泛的解剖区域、视图和采集协议来训练和评估异常检测方法。SADUSI的目标是提供一个多样化的正常超声分布，以及一个可从单张图像评估的可见结构异常基准。我们评估了代表性的自监督异常检测方法，发现当前方法在这种设置下表现不佳。特别是，基于重建的扩散方法（如Ano

    arXiv:2610.09677v1 Announce Type: cross  Abstract: Self-supervised anomaly detection is a promising paradigm for medical ultrasound, as normal images are often easier to obtain than exhaustive annotations of all possible pathologies. However, most existing evaluations are limited to a single anatomy or task, making it unclear whether models learn a robust notion of normal ultrasound appearance or only a source-specific representation. We introduce the SADUSI benchmark, a multi-source ultrasound dataset designed to train and evaluate anomaly detection methods across a broad range of anatomical regions, views, and acquisition protocols. The goal of SADUSI is to provide a diverse normal ultrasound distribution and a benchmark for visible structural anomalies that can be assessed from single images. We evaluate representative self-supervised anomaly detection methods and find that current approaches struggle in this setting. In particular, reconstruction-based diffusion methods such as Ano
    
[^168]: 高斯-牛顿精度与不定海森矩阵：低成本集合中的一致共存

    Gauss-Newton Accuracy and Indefinite Hessians: Uniform Coexistence in Low-Cost Sets

    [https://arxiv.org/abs/2610.09675](https://arxiv.org/abs/2610.09675)

    本文证明在岭正则化非线性最小二乘中，低成本集合内一致共存两种曲率状态：每个全局极小值点的海森矩阵相对误差低于 $(1+\sqrt{2})/8$，而同一集合中也存在海森矩阵不定、相对误差至少为 $15/8$ 的点，并给出逐点证书与尖锐的相对误差界。

    

    我们研究岭正则化非线性最小二乘问题中高斯-牛顿曲率的精度。在水平集曲率幅值沿精确拟合截面保持局部正则性与持续性的条件下，我们证明了两种曲率状态的一致共存。全局极小值点存在，且每个全局极小值点的相对海森矩阵误差低于 $(1+\sqrt{2})/8$，而同一个低成本集合中还包含一个海森矩阵不定、相对误差至少为 $15/8$ 的点。一个正的岭上限对固定邻域内所有独立的中心和标签扰动、以及不超过该上限的每个正岭权重均适用。当岭权重趋于零时，这些邻域不会收缩。基于当前预测水平集的逐点证书控制海森修正的法向、混合和切向部分。当预测映射与岭参数变化时，我们在所述逐点类别上证明了尖锐的相对误差界。解析例子描述了……

    arXiv:2610.09675v1 Announce Type: new  Abstract: We study the accuracy of Gauss-Newton curvature in ridge-regularized nonlinear least squares. Under local regularity and persistence of level-set curvature magnitude along an exact-fit section, we prove uniform coexistence of two curvature regimes. Global minimizers exist, and every global minimizer has relative Hessian error below $(1+\sqrt2)/8$, while the same low-cost set contains a point with an indefinite Hessian and relative error at least $15/8$. One positive ridge cap works for all independent center and label perturbations in fixed neighborhoods and every positive ridge weight up to the cap. These neighborhoods do not shrink as the ridge weight tends to zero. A pointwise certificate based on the current prediction level set controls the normal, mixed, and tangent parts of the Hessian correction. We prove a sharp relative-error bound over the stated pointwise class when the prediction map and ridge vary. Analytic examples describ
    
[^169]: DSTNet：用于因果多期金融预测的动态频谱轨迹网络

    DSTNet: Dynamic Spectral Trajectory Network for Causal Multi-Horizon Financial Forecasting

    [https://arxiv.org/abs/2610.09654](https://arxiv.org/abs/2610.09654)

    DSTNet通过构建因果动态频谱轨迹并采用尺度-时间频谱Transformer与CNN-BiLSTM门控融合的架构，在单次前向传递中同时输出1、3、5、10天的多期金融预测，且严格保证不泄露未来信息。

    

    基于小波的金融预测器通常仅将变换用于去噪，或将其简化为预测起点处的单一频谱快照，且产生系数的卷积通常是双边的，因此可能会读取到预测起点之后的信息。DSTNet则将滤波器组幅值的近期演变保留为因果的动态频谱轨迹，该轨迹由七个滞后技术指标在二十天回溯期内构建，采用单边Morlet派生滤波器组，并对左边界瞬态进行显式的预热处理。分解式的尺度-时间频谱Transformer分别沿时间和滤波器组两个轴进行注意力计算，通过学习到的门控机制将频谱分支与CNN-BiLSTM融合，并由针对特定预测期的门控在单次前向传递中输出一天、三天、五天和十天的预测。我们在统一的扩展窗口协议和未被使用过的一年期保留测试集上，将该方法与九个学习型基线模型进行对比，对七个股票指数和黄金进行了评估。

    arXiv:2610.09654v1 Announce Type: new  Abstract: Wavelet-based financial forecasters typically use the transform only to denoise, or reduce it to a single spectral snapshot at the forecast origin, and the convolution that produces the coefficients is usually bilateral, so it can read past the forecast origin. DSTNet instead retains the recent evolution of filter-bank magnitudes as a causal Dynamic Spectral Trajectory, built from seven trailing technical indicators over a twenty-day lookback with a one-sided Morlet-derived filter bank and an explicit burn-in for the left-boundary transient. A factorized Scale-Temporal Spectral Transformer attends along the time and filter-bank axes separately, a learned gate fuses the spectral branch with a CNN-BiLSTM, and horizon-specific gates emit one, three, five, and ten day forecasts in a single pass. We evaluate seven equity indices and gold under a common expanding-window protocol and an untouched one-year hold-out, against nine learned baseline
    
[^170]: 针对随机分配DP-SGD的闭式噪声校准以抵御成员推断攻击

    Closed-Form Noise Calibration Against Membership Inference for Random-Allocation DP-SGD

    [https://arxiv.org/abs/2610.09651](https://arxiv.org/abs/2610.09651)

    本文针对随机分配的DP-SGD提出了一个单行闭式公式，可在微秒级时间内直接计算出抵御任何成员推断攻击所需的噪声量，且所需噪声最多仅为现有最先进闭式界的一半。

    

    DP-SGD通过向裁剪后的梯度添加高斯噪声来保护训练数据，而噪声量通常需要在搜索过程中运行数值隐私会计器来确定。我们研究了采用随机分配的DP-SGD，即每个训练轮次（epoch）在随机选择的步骤中使用每条记录一次。针对这一设置，我们给出了一个单行公式，可以界定针对训练后模型的任何成员推断攻击（MIA）的准确率。设每个epoch有M步、共E个epoch、噪声乘数为σ，且成员与非成员的先验概率相等时，攻击准确率至多为 ½ + ¼√((1+(e^(1/σ²)-1)/M)^E - 1)。该公式来源于高斯分布与支配随机分配的高斯混合分布之间的卡方散度。该公式具有很强的可解释性，并能在大约一微秒内计算出所需的σ。在适用的场景下，与最先进的闭式界相比，我们的公式最多只需要约一半的噪声。

    arXiv:2610.09651v1 Announce Type: new  Abstract: DP-SGD protects training data by adding Gaussian noise to clipped gradients. The amount of noise is usually chosen by running a numerical privacy accountant inside a search. We study DP-SGD with random allocation, where each epoch uses every record once, at a randomly chosen step. For this setting we give a one-line formula that bounds the accuracy of every membership inference attack (MIA) on the trained model. With $M$ steps per epoch, $E$ epochs and noise multiplier $\sigma$, and with membership and non-membership equally likely a priori, the attack accuracy is at most $\frac12+\frac14\sqrt{(1+(e^{1/\sigma^2}-1)/M)^E-1}$. The formula comes from the chi-square divergence between a Gaussian distribution and a Gaussian mixture that dominates random allocation. It is interpretable and gives $\sigma$ in about a microsecond. Where applicable, our formula needs at most about half the noise of the state-of-the-art closed-form bound. To measur
    
[^171]: 当大语言模型退化时秩反而上升

    When Rank Rises as LLMs Degrade

    [https://arxiv.org/abs/2610.09647](https://arxiv.org/abs/2610.09647)

    该研究发现LLM后训练中的表示退化（如数据重复）会使RankMe等谱秩指标不降反升，导致单边监控将最差模型误判为最健康，因此指标变化方向取决于具体的退化模式与统计量配对，传统监控假设并不安全。

    

    后训练使语言模型适应非平稳环境。从业者通常使用RankMe及相关谱统计量来监控表示健康度，并往往假设当表示退化时秩会下降。我们证明这一假设对于LLM后训练是不安全的。在对Qwen3-0.6B进行的受控研究（四种退化模式、三个随机种子）中，数据重复使留出集损失相比健康状态恶化75%，同时使原始和中心化RankMe均上升；后者的变化幅度达到13.5个合并标准差。协方差有效秩升至接近其健康值的两倍。这种失败是谱离散化而非表示坍缩，因此单边监控器会将最差的检查点评为最健康的。相比之下，学习率配置错误会降低中心化RankMe和k95，而未中心化的RankMe在不同随机种子间表现不一致。因此，指标变化方向是“退化模式-统计量”配对的固有属性，无法通过重新校准来修复。

    arXiv:2610.09647v1 Announce Type: cross  Abstract: Post-training adapts language models in non-stationary environments. Practitioners monitor representation health with RankMe and related spectral statistics, often assuming that rank falls when representations degrade. We show that this assumption is unsafe for LLM post-training. In a controlled study of Qwen3-0.6B with four degradation modes and three seeds, data duplication worsens held-out loss by 75% relative to healthy while increasing both original and centred RankMe; the latter changes by 13.5 pooled standard deviations. Covariance effective rank rises to nearly twice its healthy value. This failure is spectral dispersion rather than collapse, so a one-sided monitor rates the worst checkpoint as the healthiest. By contrast, a learning-rate misconfiguration lowers centred RankMe and k95, while uncentred RankMe is inconsistent across seeds. Direction is therefore a property of the regime-statistic pair and cannot be fixed by recal
    
[^172]: Q-PhotoMarket：面向金融市场预测的光子混合量子神经网络设计空间探索框架

    Q-PhotoMarket: A Design Space Exploration Framework for Photonic Hybrid Quantum Neural Networks in Financial Market Prediction

    [https://arxiv.org/abs/2610.09641](https://arxiv.org/abs/2610.09641)

    该论文提出了Q-PhotoMarket框架，通过系统性探索超过5,000种光子混合量子神经网络配置，首次全面考察了光子电路设计选择对金融市场预测性能的影响。

    

    光子量子计算因其对线性光学电路的原生实现以及玻色采样的计算复杂性，近年来已成为混合量子机器学习的一个有前景的平台。然而，尽管量子方法在金融领域的应用日益受到关注，光子电路设计选择对预测性能的影响在很大程度上仍未被探索。现有研究通常只评估单一架构，而未对更广泛的光子设计空间加以考察。在本工作中，我们提出了Q-PhotoMarket，一个应用于金融市场预测的光子混合量子神经网络（HQNN）的系统性设计空间探索（DSE）框架。针对美国、印度和加密货币市场，我们在其兼容的计算空间中探索了超过5,000种有效的光子配置，涵盖输入光子态、电路架构、纠缠模型和测量策略。为了提高搜索效率……（摘要在此处截断）

    arXiv:2610.09641v1 Announce Type: cross  Abstract: Photonic quantum computing has recently emerged as a promising platform for hybrid quantum machine learning due to its native realization of linear-optical circuits and the computational complexity of boson sampling. However, despite growing interest in quantum methods for finance, the influence of photonic circuit design choices on predictive performance remains largely unexplored. Existing studies typically evaluate a single architecture, leaving the broader photonic design space unexamined. In this work, we present Q-PhotoMarket, a systematic design space exploration (DSE) framework for photonic hybrid quantum neural networks (HQNNs) applied to financial market prediction. We explore over 5,000 valid photonic configurations spanning input photon states, circuit architectures, entangling models, and measurement strategies across their compatible computation spaces, for U.S., Indian, and cryptocurrency markets. To improve search effic
    
[^173]: 追踪输入，验证输出：音乐生成中的归因验证

    Tracing Inputs, Verifying Outputs: Validating Attribution in Music Generation

    [https://arxiv.org/abs/2610.09637](https://arxiv.org/abs/2610.09637)

    该论文提出通过仅基于音频条件生成、输入追踪以及音乐版本识别模型musicDNA审计记忆化，为AI音乐生成中“哪些音频来源被使用并影响了输出”提供了可验证的归因证据。

    

    我们如何验证谁的音乐对AI生成的输出做出了贡献？本文展示了基于输入的归因如何提供可验证的证据，以确定生成过程中使用了哪些音频来源，以及这些来源是否塑造了输出。为此，我们仅以音频为条件进行生成，不使用任何文本输入，然后追踪每个输出背后的输入，并确定它们的音乐效果。在提示遵循测试和受控输入交换实验中，我们的生成器MixAudio所生成的音轨在音色上遵循提示音频，在和声上遵循上下文音频。然而，这些输出仍可能复现未作为输入提供的训练数据。因此，我们使用我们的音乐版本识别模型musicDNA来审计记忆化现象，发现在输入记录之外很少有复现情况。在被标记池中的人工判断案例上，该模型比其他测试过的记忆检测器达到了更高的精确率和召回率。这两项评估表明……

    arXiv:2610.09637v1 Announce Type: cross  Abstract: How can we verify whose music contributed to an AI-generated output? This paper demonstrates how input-based attribution can provide verifiable evidence of which audio sources were used in a generation and whether they shaped the output. To do so, we condition the generation solely on audio without any text input, then trace the inputs behind each output, and establish their musical effect. In prompt adherence tests and controlled input swaps, the stems generated by our generator, MixAudio, follow the prompt audio in timbre and the context audio in harmony. Yet these outputs may still reproduce training data not supplied as inputs. We therefore audit memorization with our musical version identification model, musicDNA, and find few reproductions outside the input records. On human-judged cases within the flagged pool, it achieves higher precision and recall than the other tested memorization detectors. The two evaluations suggest that 
    
[^174]: 真实稀缺数据中的量子异常检测

    Quantum anomaly detection in real scarce data

    [https://arxiv.org/abs/2610.09635](https://arxiv.org/abs/2610.09635)

    针对小型不平衡数据集上异常检测的难题，本文提出了一种新颖的两步混合经典-量子架构，利用量子机器学习以更少的可训练参数和更小的数据集实现更可解释、更节能的异常检测。

    

    arXiv:2610.09635v1 公告类型：cross 摘要：尽管小型且不平衡数据集的场景在医疗健康、网络安全、金融和能源等多个领域中十分常见，但在此类数据集上进行异常检测仍然是机器学习中极具挑战性的问题。数据增强和生成式人工智能或许能够缓解训练数据稀缺的问题，但由于异常事件本质上相对于高概率的正常数据而言是不可预测的、稀少的且高度多样化的，这些方法往往难以奏效。对伪异常的过拟合、模型崩溃、高维数据、不可解释的黑箱模型以及验证困难等，是限制其实际应用的典型问题。在此背景下，量子机器学习可能提供一条有前景且更可持续的途径，因为它能够以少得多的可训练参数和更小的数据集构建更可解释的模型，并可在节能的量子硬件上实现。在此，我们提出了一种新颖的两步混合经典-量子架构……

    arXiv:2610.09635v1 Announce Type: cross  Abstract: Anomaly detection on small and unbalanced datasets remains very challenging in machine learning, although this scenario is common in several domains, including healthcare, cybersecurity, finance, and energy. Data augmentation and generative AI may mitigate training-data scarcity, but they often fall short because anomalies are, by definition, unpredictable, rare, and highly diverse events compared to high-probability normal data. Overfitting to pseudo-anomalies, model collapse, high-dimensional data, uninterpretable black-box models, and validation challenges are typical issues limiting their practical applicability. In this context, quantum machine learning may provide a promising and more sustainable avenue because it can enable more interpretable models with far fewer trainable parameters and smaller datasets, implementable on energy-efficient quantum hardware. Here, we propose a novel two-step hybrid classical--quantum architecture
    
[^175]: 编码智能体基准测试应匹配其用户的任务流

    Coding-Agent Benchmarks Should Match Their Users' Task Flows

    [https://arxiv.org/abs/2610.09633](https://arxiv.org/abs/2610.09633)

    该研究通过收集JetBrains IDE中真实软件工程师的4,782个智能体会话，发现真实任务流在任务类型与切换模式上高度多样且因数据源而异，因此编码智能体基准测试应先指明目标用例，再依据其真实测得的任务流进行校准。

    

    编码智能体的评估通常力求尽可能贴近真实。在本研究中，我们收集了JetBrains IDE中真实软件工程师的4,782个智能体会话，我们称之为“生产会话”。由于我们的研究对象是交互式智能体，我们研究了包含至少三条用户消息的会话（占样本的33%）。这些长会话与源自issue的基准测试任务在两个方面有所不同：(i) 用户请求所涵盖的任务类型范围要广泛得多——包括对项目代码的提问、规划、审查、重构、执行等；(ii) 用户会在整个会话过程中于不同任务类型之间切换。来自三个公开交互语料库的长会话样本展现出显著不同的任务流——即会话长度、任务类型以及类型间转换的分布——因此没有任何单一的交互分布是普遍真实的：基准测试应当指明目标用例，并根据从该用例中测得的数据进行校准。我们提出了SWE-TaskFlow，一种……（原文摘要在此处截断）

    arXiv:2610.09633v1 Announce Type: cross  Abstract: The evaluation of coding agents generally strives to be as realistic as possible. In our study, we collect 4,782 agent sessions of real software engineers in JetBrains IDEs, which we call Production Sessions. Since our subject is interactive agents, we study the sessions with at least three user messages (33% of the sample). These long sessions differ from issue-derived benchmark tasks in two ways: (i) user requests span a far wider mix of task types - questions about the project's code, planning, review, refactoring, execution - and (ii) users switch between types throughout a session. Long-session samples from three public interaction corpora exhibit markedly different Task Flows (the distributions of session lengths, task types, and type-to-type transitions), so no single interaction distribution is universally realistic: benchmarks should name a target use case and calibrate to measurements from it. We present SWE-TaskFlow, an appr
    
[^176]: 智能体大语言模型如何决定调用工具？一个由抑制机制塑造的工具调用向量

    How Do Agentic LLMs Decide to Call Tools? A Tool-Call Vector Shaped by Suppression

    [https://arxiv.org/abs/2610.09624](https://arxiv.org/abs/2610.09624)

    该研究通过构造仅由单个请求动词即可翻转工具调用决策的最小对比提示对，揭示了智能体大语言模型内部存在一个由抑制机制塑造的“工具调用向量”，它介导了模型在调用工具与直接回答之间的决策。

    

    工具调用（按需调用外部工具）是智能体大语言模型的核心能力，然而决定模型是调用工具还是直接回答的机制仍然知之甚少。智能体提示词通常很长且高度结构化，在数百个token中融合了角色指令、工具模式、格式模板和用户请求，形成了一个嘈杂且高度纠缠的上下文，其中缺乏明显可用于机制分析的可控变量。为了获得这样的变量，我们提出了一种方法，将复杂的智能体提示词转化为最小对比对，其中单个请求动词即可决定工具调用决策：将执行类动词（如“编写”）替换为分析类动词（如“讨论”）能够可靠地翻转决策，这表明该决策由一个紧凑的内部状态介导。我们在Python、Java和C++上构建了500个这样的配对提示（其中300个用于机制分析，200个保留用于评估）

    arXiv:2610.09624v1 Announce Type: cross  Abstract: Tool calling, invoking external tools on demand, is central to agentic LLMs, yet the mechanism that decides whether a model calls a tool or responds directly remains poorly understood. Agentic prompts are long and heavily scaffolded, combining role instructions, tool schemas, format templates, and the user's request across hundreds of tokens, creating a noisy, highly entangled context in which no single controllable variable for mechanistic analysis is obvious. To obtain such a variable, we propose a method that converts complex agentic prompts into minimal contrastive pairs in which a single request verb determines the tool-call decision: replacing an execution-verb (e.g., \textit{write}) with an analysis-verb (e.g., \textit{discuss}) reliably flips the decision, suggesting it is mediated by a compact internal state. We construct 500 such paired prompts across Python, Java, and C++ (300 for mechanistic analysis, 200 held out for evalu
    
[^177]: 网络的训练历史何时比其当前状态更能预测其未来学习？来自响应探针与预测筛选的证据

    When does a network's training history predict its future learning better than its current state? Evidence from a response probe and a forecasting screen

    [https://arxiv.org/abs/2610.09621](https://arxiv.org/abs/2610.09621)

    该研究设计并严格校验了一种“响应探针”方法，用以探究在何种条件下网络的训练历史比其当前状态更能预测其未来的学习能力。

    

    当前行为相似的网络在训练继续时仍可能以不同方式学习。关于可塑性丧失与关键期的研究表明，通往某一状态的路径会塑造其后续发展；但这些工作并未说明这条路径是否携带了对状态本身进行测量时所遗漏的信息。我们探究何时网络的训练历史比其当前状态更能预测其未来学习。在一项主要研究中，小型多层感知机在三种历史条件下（共42种历史）进行训练，并在四个检查点通过一个简短探针来测量未来学习：即在新任务上对网络副本进行100次更新的训练。在读取预测结果之前，协议先对探针本身进行了检验：探针对隐藏单元在保持函数不变的重新缩放下呈单调响应，重复测量结果一致（在可靠性最低的类别中，三次重复均值的组内相关系数为0.940，区间[0.903, 0.997]），并且单元的重新初始化能够被察觉到。

    arXiv:2610.09621v1 Announce Type: new  Abstract: Networks that behave alike now can still learn differently when training continues. Work on loss of plasticity and critical periods shows that the path to a state shapes what follows; it does not show whether the path carries information that a measurement of the state itself misses. We ask when the training history of a network predicts its future learning better than its current state. In a main study, small multilayer perceptrons were trained under three history regimes (42 histories), and future learning was measured at four checkpoints by a short probe: a copy of the network trained for 100 updates on a new task. Before the prediction result was read, the protocol checked the probe. It responded monotonically to a function-preserving rescaling of hidden units, repeated measurements agreed (intraclass correlation 0.940, [0.903, 0.997], in the least reliable class, mean of three repeats), and a re-initialisation of units was visible d
    
[^178]: 深度归一化注意力机制的可辨识性与可观测性

    The Identifiability and Observability of Deep Normalized Attention

    [https://arxiv.org/abs/2610.09620](https://arxiv.org/abs/2610.09620)

    本文证明了对于实解析归一化函数（如softmax），深度归一化注意力的输入-输出函数在一般情形下可确定其有效得分与值映射（至符号差异），并精确刻画了参数坍缩导致得分不可观测的条件及雅可比衰减谱。

    

    我们研究了深度、无掩码、单头注意力模型的哪些参数由其输入-输出函数所确定。对于已知的正的非常值实解析归一化函数，该函数在一般情形下能够确定有效得分和组合值映射，仅有偶归一化函数引起的符号不确定性除外。这证明了 Henry–Marchetti–Kohn 猜想的实解析情形，包括 softmax 的情况。随后，我们在明确的归一化函数条件下对例外纤维进行了分类，识别出何时坍缩会使后层得分变得不可观测，并为局部可辨识性建立了精确的泰勒阶数。在查询/键同时接近坍缩时，我们计算了在分离有限输入集上完整的原生雅可比衰减谱。对于常见的首个非常值归一化函数次数 k，第 i 层的接触阶为 2k·3^(i-1)−1，并给出了精确的重数和核维数。高精度和自动微分的计算展示了由此导致的数值稳定性损失。

    arXiv:2610.09620v1 Announce Type: new  Abstract: We study which parameters of deep, unmasked, single-head attention are determined by its input--output function. For known positive nonconstant real-analytic normalizers, the function generically determines the effective scores and combined value map up to the signs induced by even normalizers. This proves the real-analytic case of a conjecture of Henry--Marchetti--Kohn, including softmax. We then classify exceptional fibers under explicit normalizer conditions, identifying when collapse makes later scores unobservable, and establish sharp Taylor orders for local identification. Near simultaneous query/key collapse, we compute the complete native Jacobian decay spectrum on separating finite input banks. For common first nonconstant normalizer degree $k$, layer $i$ has contact order $2k3^{i-1}-1$, with exact multiplicities and kernel dimension. High-precision and automatic differentiation calculations illustrate the resulting loss of nume
    
[^179]: 变投影支持向量机模型的二阶优化与道路异常检测

    Second-order optimization of variable projection SVM models and road abnormality detection

    [https://arxiv.org/abs/2610.09617](https://arxiv.org/abs/2610.09617)

    本文提出了一种针对变投影泛函的新型二阶优化框架，利用二阶信赖域算法高效训练变投影支持向量机（VP-SVM），并成功应用于基于轮胎传感器一维信号的道路异常检测。

    

    我们提出了一种用于最小化所谓变投影泛函的新型二阶优化框架。我们证明所提出的框架对于基于变投影的核方法的训练特别有用。特别地，我们考虑了高效训练变投影支持向量机（VP-SVM）的问题。我们在一个实际应用中展示了所提出训练方法的有效性，即我们展示了如何使用二阶信赖域算法来训练VP-SVM模型，从而基于从轮胎传感器获得的一维信号识别路面异常。

    arXiv:2610.09617v1 Announce Type: cross  Abstract: We introduce a novel second-order optimization framework for minimizing so-called variable projection functionals. We demonstrate that the proposed framework is especially usefulfor the training of variable projection based kernel methods. In particular, the problem of efficiently training variable projection support vector machines (VP-SVMs) is considered. We show the effectiveness of the proposed training methodology in a real-world application, namely we demonstrate how second-order trust region algorithms can be used to train VPSVM models to recognize road surface abnormalities based on 1D signals obtained from a tire sensor.
    
[^180]: 实证资产定价中的残差学习

    Residual Learning in Empirical Asset Pricing

    [https://arxiv.org/abs/2610.09613](https://arxiv.org/abs/2610.09613)

    本文提出将残差学习应用于实证资产定价，使神经网络模型在保留浅层模型的基础上得以加深，深度残差模型的样本外夏普比率（2.07）显著优于浅层模型（1.92）和深度前馈模型（0.89），证明模型深度是额外经济价值的来源，并为构建“大型资产定价模型”提供了可行路径。

    

    浅层模型是深层模型的特例，深层模型在理论上具有超越浅层模型的潜力。然而，现有的实证资产定价文献已为浅层模型提供了强大的基准。残差学习通过保留并改进浅层模型，使资产定价中的神经网络模型能够变得更深。深度残差模型的价值加权多空组合的样本外夏普比率（2.07）高于相应的浅层模型（1.92），并且是深度前馈模型（0.89）的两倍以上。我们证明了模型深度是资产定价中额外经济价值的一个来源。只要其他基于神经网络的资产定价模型包含中间层，残差学习就可以用于加深这些模型。我们的设计还提供了一种扩展资产定价模型规模的方法，使原生的“大型资产定价模型”更加可行。

    arXiv:2610.09613v1 Announce Type: cross  Abstract: Shallow models are special cases of deep models, and deep models theoretically have the potential to outperform the shallow ones. However, the existing empirical asset pricing literature provides strong benchmarks for shallow models. Residual learning allows neural network models in asset pricing to go deeper by preserving and refining their shallow counterparts. The out-of-sample Sharpe ratio for value-weighted long-short portfolios of deep residual models (2.07) is higher than that for the corresponding shallow ones (1.92) and more than twice that of the deep feedforward models (0.89). We show that model depth is a source of additional economic value in asset pricing. Residual learning can be used to deepen other neural-network-based asset pricing models if they contain intermediate layers. Our design also provides one way to scale asset pricing models, making native "large asset pricing models" more feasible.
    
[^181]: 序贯预训练更有利于大模型

    Sequential Pretraining Favors Large Models

    [https://arxiv.org/abs/2610.09611](https://arxiv.org/abs/2610.09611)

    该论文定义并量化了“首因偏差”，揭示小模型会将学习容量低效地分配给早期训练数据而大模型对此具有鲁棒性，因此当数据按顺序用于预训练时大模型更具优势，尤其是当后期数据强调代码、数学和推理能力时。

    

    大型神经网络往往能习得小模型无法学会的能力。这究竟是源于大模型学到了更具代表性的特征，还是因为它们对训练过程中产生的、未被考虑在内的不利影响更加鲁棒？我们定义并量化了其中一种不利影响——首因偏差，即接触早期数据分布在多大程度上会损害后续的学习。我们证明，小模型可能将学习容量低效地分配给早期数据分布，而充分过参数化的模型对这种影响具有鲁棒性。这种低效性在预训练中尤为关键，因为基础模型通常是按顺序而非联合的方式接触到异构的数据分布。因此，小的基础模型可能难以学习训练后期才遇到的数据分布，当后期数据强调代码、数学和推理等理想能力时，这一问题尤其有害。

    arXiv:2610.09611v1 Announce Type: new  Abstract: Large neural networks often acquire capabilities that small models fail to learn. Does this stem from large models learning more representative features, or from being more robust to unaccounted-for adverse effects introduced during training? We define and quantify one such adverse effect, primacy bias, as the extent to which exposure to early data distributions impairs later learning. We show that small models can allocate learning capacity inefficiently toward early distributions, whereas sufficiently overparameterized models are robust to this effect. This inefficiency is particularly consequential in pretraining, where foundation models often encounter heterogeneous data distributions sequentially rather than jointly. As a result, small foundation models can struggle to learn distributions encountered late in training, which is particularly harmful when later data emphasizes desirable capabilities such as code, mathematics, and reaso
    
[^182]: 通过先锋学生实现师生半监督学习的解耦优化

    Decoupled Optimization for Teacher-Student Semi-Supervised Learning via a Pioneer Student

    [https://arxiv.org/abs/2610.09609](https://arxiv.org/abs/2610.09609)

    提出先锋学生（PiS）这一独立参数空间的辅助分支，通过周期性回传知识，解耦了师生强同步与有/无标签损失联合优化导致的两大优化病态。

    

    半监督学习（SSL）依赖于两个核心机制：师生框架下的自训练，以及有标签损失与无标签损失的联合优化。尽管这些机制行之有效，我们发现它们各自引入了不同的优化病态。首先，参数耦合强制教师与学生之间严格同步，对学生施加的强正则化会削弱教师的拟合能力，从而限制了可允许的泛化强度。其次，有标签损失与无标签损失之间梯度更新一致性的不平衡，会驱使共享参数过早收敛到由有标签数据主导的局部极小值，形成全局优化的瓶颈。为解决这两个问题，我们提出了先锋学生，这是一个在独立参数空间中运行的辅助分支，并周期性地将其积累的知识回传给师生模型。大量实验……

    arXiv:2610.09609v1 Announce Type: new  Abstract: Semi-supervised learning (SSL) relies on two core mechanisms: self-training under the Teacher-Student (T-S) framework and joint optimization of labeled and unlabeled losses. Despite their effectiveness, we find both mechanisms introduce distinct optimization pathologies. First, parameter coupling enforces strict synchronization between teacher and student, where strong regularization on the student degrades the teacher's fitting ability, thereby limiting the permissible generalization intensity. Second, the imbalance in gradient update consistency between labeled and unlabeled losses drives the shared parameters to prematurely converge to labeled-dominated local minima, creating a bottleneck for global optimization. To address both issues, we propose the Pioneer Student (PiS), an auxiliary branch that operates in an independent parameter space and periodically transfers accumulated knowledge back to the T-S model. Extensive experiments s
    
[^183]: 通过梯度历史重组实现轻量且通用的学习优化器

    Lightweight and Versatile Learned Optimization by Recombination of Gradient History

    [https://arxiv.org/abs/2610.09604](https://arxiv.org/abs/2610.09604)

    本文提出一种轻量级学习优化器，通过动态重组不相交时间跨度的梯度历史平均，仅用37k参数和0.87 GPU小时训练即可零样本泛化到NLP、视觉和图模型等多种任务，显著优于Adam且FLOPs开销低至0.3%。

    

    本文提出了一种轻量且通用的学习型优化器，它能够动态重组以不相交时间跨度的平均值所表示的梯度历史。该优化器将预测空间缩减为每个梯度平均仅对应一个标量系数，且该系数由多个参数共享。通过渐进式地对较旧的梯度进行平均，最小化了长历史的内存成本，同时保持各历史贡献可被独立访问。一个仅含37k参数的网络在0.87 GPU小时内训练完成，即可零样本泛化到未见过的任务：在BERT-Tiny和GPT-Tiny上分别将验证损失降低9.1%和0.4%，在Vision Transformer上测试准确率相比Adam提升3.5个百分点，在九个图模型上平均提升2.7个百分点，而FLOPs开销低至0.3%。

    arXiv:2610.09604v1 Announce Type: new  Abstract: This paper presents a lightweight and versatile learned optimizer that dynamically recombines gradient history, represented as averages over disjoint time spans. The optimizer reduces the prediction space to one scalar coefficient per gradient average, shared by multiple parameters. Progressively averaging older gradients minimizes memory cost of long history, while keeping their contributions independently accessible. A 37k-parameter network trained in 0.87 GPU-hours generalizes zero-shot to unseen tasks, lowering validation loss by 9.1% and 0.4% on BERT-Tiny and GPT-Tiny, and improving test accuracy over Adam by 3.5 %p on a Vision Transformer and by 2.7 %p on average across nine graph models, with FLOPs overhead as low as 0.3%.
    
[^184]: COPC：面向异步大语言模型强化学习的耦合式离线策略修正

    COPC: Coupled Off-Policy Correction for Asynchronous LLM Reinforcement Learning

    [https://arxiv.org/abs/2610.09597](https://arxiv.org/abs/2610.09597)

    该论文指出异步强化学习中仅靠策略侧的重要性比率修正无法解决“优势过时性”问题，并提出将策略侧与优势侧修正协同进行的耦合式离线策略修正方法COPC。

    

    异步强化学习通过将轨迹生成与优化过程解耦来加速大语言模型的后训练，但其训练时使用的是过时的轨迹。现有方法主要在actor目标中通过重要性比率控制来修正token级别的策略失配。我们证明，仅这种“策略侧修正”是不够的：优势估计同样会从行为策略的延续中继承失配，我们将其称为“优势过时性”。我们针对一般的双通道actor更新推导出了精确的偏差与方差分解，揭示了策略权重误差与优势估计误差之间不可分离的耦合：二者的相互作用会产生乘性偏差项，而策略权重的平方会在梯度方差中放大优势估计的不确定性。这促使我们提出一个假设：策略侧与优势侧的修正应当协同进行。我们提出了耦合式离线策略修正，这是一种actor-critic方法，结合……

    arXiv:2610.09597v1 Announce Type: new  Abstract: Asynchronous RL accelerates large language model post-training by decoupling rollout generation from optimization, but trains on stale trajectories. Existing methods primarily correct token-level policy mismatch through importance-ratio control in the actor objective. We show that this \emph{policy-side correction} alone is insufficient: advantage estimates also inherit mismatch from behavior-policy continuations, which we term \emph{advantage staleness}. We derive exact bias and variance decompositions for a general two-channel actor update, revealing nonseparable coupling between policy-weight and advantage-estimation errors: their interaction induces multiplicative bias terms, while squared policy weights amplify advantage uncertainty in gradient variance. This motivates the hypothesis that policy- and advantage-side correction should be coordinated. We introduce Coupled Off-Policy Correction (COPC), an actor--critic method combining 
    
[^185]: 基于动力学朗之万采样的可扩展逻辑高斯过程密度回归

    Scalable Logistic Gaussian Process Density Regression with Kinetic Langevin Sampling

    [https://arxiv.org/abs/2610.09591](https://arxiv.org/abs/2610.09591)

    本文提出一种基于逻辑高斯过程的可扩展贝叶斯条件密度估计方法，通过在强对数凹后验上模拟动力学朗之万动力学进行直接采样，并利用Nyström特征支持具有依赖输入参数的非平稳核。

    

    条件密度估计旨在给出给定协变量下响应变量的完整分布，例如每星系的光度红移估计就需要此类方法。我们开发了一种基于逻辑高斯过程的可扩展贝叶斯估计器。对数条件密度具有可分离的协方差结构：沿响应方向使用Matérn核，通过圆上的截断傅里叶基表示；协变量方向使用由Nyström特征表示的核，可容纳具有依赖输入的幅度和长度尺度的非平稳核。不同于拉普拉斯近似或变分近似，我们直接对该有限特征模型的隐随机场进行采样。在给定超参数的情况下，其后验是强对数凹的，且Hessian矩阵一致有界，我们通过在Kronecker白化坐标系中模拟具有对称小批量分裂的动力学朗之万动力学来从中采样。边缘似然梯度通过费舍尔恒等式从后验期望中获得……

    arXiv:2610.09591v1 Announce Type: cross  Abstract: Conditional density estimation targets the full distribution of a response given covariates, as required, for example, for per-galaxy photometric redshifts. We develop a scalable Bayesian estimator based on the logistic Gaussian process. The log conditional density has a separable covariance: a Mat\'ern kernel along the response, represented in a truncated Fourier basis on a circle, and a covariate kernel represented by Nystr\"om features, which accommodate non-stationary kernels with input-dependent amplitudes and length scales. Instead of a Laplace or variational approximation, we sample the latent field of this finite-feature model. Given the hyperparameters, its posterior is strongly log-concave with a uniformly bounded Hessian, and we draw from it by simulating kinetic Langevin dynamics with symmetric minibatch splitting in Kronecker-whitened coordinates. Marginal-likelihood gradients follow from Fisher's identity as posterior exp
    
[^186]: 通过交叉反馈与连贯性策展实现的协同推理蒸馏

    Collaborative Reasoning Distillation via Cross-Feedback and Coherent Curation

    [https://arxiv.org/abs/2610.09587](https://arxiv.org/abs/2610.09587)

    该论文提出协同推理蒸馏框架 CRD，结合教师间交叉反馈、与答案无关的逐步质量评估和连贯性步骤拼接，并通过带预算约束的推理质量优化训练学生模型，使 CRD-4B 仅用 5 万条训练数据便在 MATH-500 和 AIME'25 上超越基线。

    

    推理能力对于推进大型语言模型的发展至关重要，然而现有方法要么需要庞大的计算预算，要么难以有效地将推理能力蒸馏到较小的模型中。标准的蒸馏方法依赖于基于结果的奖励，无法区分合理的推理与侥幸的猜测。我们提出了协同推理蒸馏（Collaborative Reasoning Distillation, CRD），这是一个通过三项创新来增强紧凑模型推理能力的框架：（1）交互式交叉反馈，教师之间迭代地相互批评彼此的推理；（2）细粒度的逐步质量评估，独立于最终答案捕捉逻辑有效性；（3）连贯性感知的步骤拼接，综合互补优势。学生模型通过带预算约束的推理质量优化（RQO）进行训练。我们的模型 CRD-4B 在 MATH-500 上达到 97.3%，在 AIME'25 上达到 70.3%，仅使用 5 万条训练数据便超越了基线模型。

    arXiv:2610.09587v1 Announce Type: cross  Abstract: Reasoning capabilities are critical for advancing Large Language Models, yet current approaches either require massive computational budgets or struggle to effectively distill reasoning to smaller models. Standard distillation methods rely on outcome-based rewards, failing to distinguish between sound reasoning and lucky guesses. We propose Collaborative Reasoning Distillation (CRD), a framework that enhances reasoning in compact models through three innovations: (1) interactive cross-feedback where teachers iteratively critique each other's reasoning, (2) fine-grained step-wise quality assessment capturing logical validity independent of final answers, and (3) coherence-aware step stitching that synthesizes complementary strengths. Students are trained via Reasoning Quality Optimization (RQO) with budget constraints. Our model, CRD-4B, achieves 97.3% on MATH-500 and 70.3% on AIME'25, surpassing baselines while using only 50K training 
    
[^187]: 具有内生马尔可夫状态的在线资源分配：更少的线性规划求解赚得更多

    Online Resource Allocation with an Endogenous Markov State: Fewer LP Solves Earn More

    [https://arxiv.org/abs/2610.09577](https://arxiv.org/abs/2610.09577)

    本文揭示了内生马尔可夫状态下在线资源分配中重新求解线性规划频率与遗憾之间的反直觉关系：在非退化条件下低频重新求解即可达到常数遗憾，而在最优解退化时频繁重新求解反而可能导致线性遗憾，即更少的LP求解反而赚得更多。

    

    我们研究了有限时间范围内、请求独立同分布且具有有限状态空间上内生马尔可夫状态的在线资源分配问题：每个动作会影响状态的转移，而该状态决定未来的收益与资源消耗。在该问题中，一个瞬态流体线性规划（LP）基准为所有非预知策略的期望收益提供了上界，而一个平稳线性规划则提供随机化的状态相关控制。我们假设平稳LP具有唯一最优解，并将原始问题非退化性以及最优诱导核的不可约性确定为该框架中的重要正则性条件。在请求先验已知的情况下，我们证明，在非退化与不可约条件下，无论是频繁还是低频重新求解均可实现 $O(1)$ 的遗憾。然而，在最优解退化时，不可约性使得低频重新求解呈现出精确的最坏情况 $\Theta(\sqrt{T})$ 速率，而频繁重新求解则可能产生 $\Omega(T)$ 的遗憾。因此，更频繁地重新求解……（摘要原文在此处截断）

    arXiv:2610.09577v1 Announce Type: new  Abstract: We study finite-horizon online resource allocation with i.i.d. requests and an endogenous Markov state on a finite state space: each action affects the transition of the state that governs future rewards and resource consumption. In this problem, a transient fluid LP benchmark upper bounds the expected reward of every nonanticipating policy, while a stationary LP supplies randomized state-dependent controls. We assume that the stationary LP has a unique optimum and identify primal nondegeneracy and irreducibility of the optimal induced kernel as important regularity conditions in this framework. With a known request prior, we show that, under nondegeneracy and irreducibility, both frequent and infrequent re-solving attain $O(1)$ regret. However, under a degenerate optimum, irreducibility yields the sharp worst-case $\Theta(\sqrt{T})$ rate for infrequent re-solving, while frequent re-solving can incur $\Omega(T)$ regret. Thus, more freque
    
[^188]: PEACE：利用宇称分辨哈密顿量对非绝热流形进行协变学习

    PEACE: Covariant learning of nonadiabatic manifolds with parity-resolved Hamiltonians

    [https://arxiv.org/abs/2610.09576](https://arxiv.org/abs/2610.09576)

    PEACE通过宇称等变哈密顿量与学习的电子连接相结合，精确再现了非绝热分子动力学中的交叉结构与弛豫动力学，并可扩展至自旋轨道耦合以模拟系间窜越。

    

    非绝热分子动力学为光驱动过程提供了机理层面的洞察，并为面向太阳能转换、光催化和光开关的分子与材料设计提供指导。准确描述这些过程需要一种尊重电子对称性、并能将能量与态间耦合一致关联的表示方法。本文提出PEACE，该方法将宇称等变的隐含哈密顿量与学习得到的电子连接相结合。受控消融实验揭示了允许对称性的态混合与电子标架变化在再现交叉结构和弛豫动力学方面所起的互补作用。PEACE能够高度精确地再现来自第一性原理模拟的激发态布居动力学，而其向自旋轨道耦合的扩展则使系间窜越的模拟成为可能。这些结果表明，将底层物理更完整地融入学习得到的电子表示中，是实现……（原文截断）

    arXiv:2610.09576v1 Announce Type: cross  Abstract: Nonadiabatic molecular dynamics provides mechanistic insight into light-driven processes and informs the design of molecules and materials for solar energy conversion, photocatalysis and photo switching. Accurately describing these processes requires a representation that respects electronic symmetry and consistently relates energies to interstate couplings. Here we introduce PEACE, which combines a parity-equivariant latent Hamiltonian with a learned electronic connection. Controlled ablations reveal the complementary roles of symmetry-allowed state mixing and electronic-frame variation in reproducing crossing structures and relaxation dynamics. PEACE closely reproduces excited-state population dynamics from first-principles simulations, while its extension to spin-orbit coupling enables simulations of intersystem crossing. These results demonstrate that a more complete incorporation of the underlying physics into learned electronic r
    
[^189]: EvoSignal：基于大语言模型引导的模块化交通信号控制程序进化设计

    EvoSignal: LLM-Guided Evolutionary Design of Modular Traffic Signal Control Programs

    [https://arxiv.org/abs/2610.09563](https://arxiv.org/abs/2610.09563)

    EvoSignal提出将交通信号控制建模为模块化程序设计问题，利用大语言模型引导的进化框架，通过交通知识与性能反馈自动演化出透明、低成本且性能更优的信号控制程序。

    

    有效的交通信号控制（TSC）需要能够响应不断变化的交通需求和网络状况，并同时满足不同控制目标的策略。然而，调整现有策略通常需要反复进行手动设计与调优，这使得针对目标网络系统地探索更优控制规则变得困难。大语言模型（LLM）可以自动化这一过程，但直接使用它们来选择信号相位会使决策规则隐藏在黑盒模型中，并带来持续的推理成本和延迟。本文将交通信号控制表述为一个模块化程序设计问题，并提出了EvoSignal——一个利用交通知识和性能反馈的LLM引导进化框架。该模块化表示将交通特征提取、本地相位优先级排序以及可选的基于网络的优先级调整分离开来。从若干已有的成熟策略出发，EvoSignal通过反馈不断改进控制程序。

    arXiv:2610.09563v1 Announce Type: new  Abstract: Effective traffic signal control (TSC) requires policies that respond to changing traffic demand and network conditions while meeting different control objectives. However, adapting existing strategies often involves repeated manual design and adjustment, making it difficult to systematically explore better control rules for a target network. Large language models (LLMs) can automate this process, but directly using them to select signal phases leaves decision rules embedded in black-box models and incurs recurring inference costs and latency. This paper formulates TSC as a modular program design problem and proposes EvoSignal, an LLM-guided evolutionary framework using traffic knowledge and performance feedback. The modular representation separates traffic feature extraction, local phase prioritization, and optional network-based priority adjustment. Starting from several established strategies, EvoSignal improves programs through feedb
    
[^190]: UniCSI：面向普适人体感知的通用Wi-Fi CSI编码器

    UniCSI Towards a Universal Wi-Fi CSI Encoder for Ubiquitous Human Sensing

    [https://arxiv.org/abs/2610.09559](https://arxiv.org/abs/2610.09559)

    提出了UniCSI统一基础架构，通过物理信息引导的RF分词器等核心创新，直接处理来自不同设备配置的异构Wi-Fi CSI数据并保持原生波形完整性，实现普适的人体感知。

    

    Wi-Fi感知有望将我们周围无处不在的日常无线信号转化为普适的人体感知传感器。然而，一个根本性障碍在于CSI是在各种设备特定的配置下获取的，包括不同的子载波数量、带宽和载波频段。因此，所得到的CSI张量在频谱分辨率和张量形状上均存在差异，使得异构建模极具挑战性。标准架构难以应对这种异构性，被迫采用有损的预处理，从而损害了底层信号。为弥合这一差距，我们提出了UniCSI，一种统一的基础架构，能够直接处理异构CSI，同时保持原生波形的完整性。UniCSI依托于两项核心创新：（1）物理信息引导的RF分词器，它基于每个频率信道在物理频谱中的相对位置而非刚性的数组索引对其进行编码。它保留...

    arXiv:2610.09559v1 Announce Type: new  Abstract: Wi-Fi sensing promises to turn the everyday wireless signals that already surround us into ubiquitous sensors for human sensing. However, a fundamental obstacle is that CSI is acquired under diverse device-specific configurations, including different subcarrier counts, bandwidths, and carrier bands. Consequently, the resulting CSI tensors vary in both spectral resolution and tensor shape, making heterogeneous modeling challenging. Standard architectures struggle with such heterogeneity, forcing lossy pre-processing which compromises the underlying signal. To bridge this gap, we present UniCSI, a unified foundation architecture that directly operates on heterogeneous CSI while preserving the integrity of the native waveform. UniCSI hinges on two core innovations: (1) a physics-informed RF tokenizer that encodes each frequency channel based on its fractional position within the physical spectrum rather than rigid array indices. It preserve
    
[^191]: 宪法引导的水印技术

    Constitution-Guided Watermarking

    [https://arxiv.org/abs/2610.09552](https://arxiv.org/abs/2610.09552)

    本文提出“宪法引导水印”框架，通过将提供商需求表示为自然语言原则，使水印系统能够根据不同请求的需求灵活选择属性权衡，避免了传统方法对所有请求采用统一配置所导致的牺牲问题。

    

    水印技术使语言模型提供商能够识别由其模型生成的文本。然而，水印的理想属性之间可能存在冲突（即更强的水印信号可能会降低文本质量），而抵抗编辑的设计也可能便于伪造。提供商通过选择平衡竞争目标或优先考虑特定属性的配置来应对这些权衡。但这两种方法都会将一个统一的操作点强加于具有不同需求的请求上，可能在需要保留原始措辞时牺牲文本质量，或在需要可靠归因时牺牲鲁棒性。为了实现灵活且可适应的设计，我们提出了宪法引导水印，这是一个能够根据提供商需求（以自然语言原则的形式列出）来为每个请求选择合适权衡的框架。在离线阶段，一个预训练的推理代理会结合水印实现来审查宪法规则，并迭代地重新……

    arXiv:2610.09552v1 Announce Type: cross  Abstract: Watermarking enables language model providers to identify text generated by their models. However, its desired properties can conflict (\ie~stronger watermark signals can degrade text quality), while designs that resist editing may also facilitate forgery. Providers address these trade-offs by choosing configurations that balance competing objectives or prioritize particular properties. Either approach imposes a shared operating point on requests with different requirements, potentially sacrificing quality where wording preservation matters or robustness where reliable attribution is essential. To allow flexible and adaptable designs, we introduce \emph{Constitution-Guided Watermarking}, a framework that selects request-appropriate trade-offs from provider requirements, listed as natural-language principles. \emph{Offline}, a pretrained reasoning agent examines constitutional rules alongside watermark implementations and iteratively re
    
[^192]: 一个用于AI注册库中机器学习资产系统综述的框架

    A Framework for the Systematic Review of ML Assets in AI Registries

    [https://arxiv.org/abs/2610.09551](https://arxiv.org/abs/2610.09551)

    本文提出一个框架，将科学文献中的系统性综述方法适配到AI注册库中机器学习资产（预训练模型、数据集、基准等）的检索与选择中，使选择过程透明、可复现且有据可循。

    

    背景：现代软件系统在构建、评估和集成基于机器学习的系统时，越来越依赖机器学习（ML）资产（即预训练模型、数据集、基准测试）。然而，目前ML资产的探索、选择和复用实践缺乏与传统证据综合方法相当的系统性检索方法学支持。因此，在实践中，ML资产的选择往往被视为一个已确定的设计决策，仅由非正式的理由说明来支撑，而非一个可追溯、基于证据且可更新的选择过程。目标：本文探讨系统性综述方法如何支持ML资产的检索，旨在使资产选择过程透明且可复现，基于明确的证据，并最终更好地适应其预期用途。方法：我们分析了科学文献中已确立的系统性综述实践，并对其各个阶段（即规划……）进行适应性调整。

    arXiv:2610.09551v1 Announce Type: new  Abstract: Background: Modern software systems increasingly rely on Machine Learning (ML) assets (i.e., pre-trained models, datasets, benchmarks) for building, evaluating, and integrating ML-based systems. However, current exploration, selection and reuse practices of ML assets are not supported by systematic retrieval methodologies comparable to those used in traditional evidence synthesis. Consequently, in practice, ML asset selection is often presented as a settled design decision, supported by informal justification rather than a traceable, evidence-based, and updatable selection process. Aims: This paper explores how systematic review methods can support ML asset retrieval. In doing so, we aim to make their selection transparent and reproducible, grounded in explicit evidence, and ultimately better suited to its intended use. Method: We analyze established systematic review practices from scientific literature and adapt their phases (i.e., pla
    
[^193]: CircuitGate：面向与-非图的逻辑一致电路级功能建模

    CircuitGate: Logic-Consistent Circuit-Level Functional Modeling for And-Inverter Graphs

    [https://arxiv.org/abs/2610.09549](https://arxiv.org/abs/2610.09549)

    该论文提出CircuitGate框架，通过显式编码全局主输入支持集与扇入间的再汇聚关系，将AIG表示学习从门级语义提升到电路级功能建模，克服了GNN局部消息传递带来的电路级上下文缺失和拓扑敏感性问题。

    

    与-非图（And-Inverter Graphs, AIGs）是电子设计自动化（EDA）中逻辑综合与验证的基础表示。作为复杂数字系统的结构化表示，AIG需要能够捕捉超越局部结构的功能依赖关系、并对保持功能的变换保持鲁棒性的模型。在基于学习的AIG表示研究中，现有方法主要基于图神经网络（GNN），依赖局部的门级消息传递，这限制了其捕捉电路级功能上下文的能力，并使学习到的表示对特定拓扑模式较为敏感。因此，我们提出CircuitGate，一个功能感知的AIG表示学习框架，将建模从门级语义推进到电路级功能建模。CircuitGate显式编码全局主输入（PI）支持集，并对扇入之间支持重叠感知的再汇聚进行建模，同时融入受逻辑启发的布尔（原文摘要在此处被截断）……

    arXiv:2610.09549v1 Announce Type: new  Abstract: And-Inverter Graphs (AIGs) are fundamental representations for logic synthesis and verification in Electronic Design Automation (EDA). As structured representations of complex digital systems, AIGs require models to capture functional dependencies beyond local structure and remain robust to functionality-preserving transformations. In learning-based AIG representation, existing approaches are predominantly based on GNNs and rely on local gate-level message passing, limiting their ability to capture circuit-level functional context and making the learned representations sensitive to topology-specific patterns. Therefore, we propose CircuitGate, a function-aware AIG representation learning framework that advances from gate-level semantics to circuit-level functional modeling. CircuitGate explicitly encodes global primary-input (PI) support and models support-overlap-aware reconvergence between fanins, while incorporating logic-inspired Boo
    
[^194]: 弃权式认证：小校准预算下思维链验证器的无分布保证

    Certified by Abstention: Distribution-Free Guarantees for Chain-of-Thought Verifiers at Small Calibration Budgets

    [https://arxiv.org/abs/2610.09541](https://arxiv.org/abs/2610.09541)

    该研究揭示“通过弃权实现有效性”现象——很少触发的验证证书虽形式上有效但每次触发时可能全部出错，并据此为小校准预算下的思维链验证器建立了无分布认证保证及其失效条件分析。

    

    预测思维链（CoT）轨迹是否正确的信号通常以AUC进行比较，但实际部署需要一个带有保证的阈值。我们研究了在几十到几百个标注问题这一现实校准预算下，无分布选择性保证能为CoT验证器提供什么，实验使用了七个开源模型、五种验证器信号和37,000条已评分轨迹。核心观察是“通过弃权实现有效性”：一个以概率 P_fire 发放证书的 (α,δ)-有效程序，仅能将已发放证书的失败概率约束在 δ/P_fire 以内，因此一个很少触发的证书可以在形式上“有效”，却在每次实际使用时都出错。在一个风险已知的模拟中，标准证书在至多0.3%的校准抽样中失败，但在其触发的抽样中失败率高达69%。认证下限以及Benjamini-Hochberg共形选择的格条件解释了为什么证书……（原文摘要在此处截断）

    arXiv:2610.09541v1 Announce Type: cross  Abstract: Signals that predict whether a chain-of-thought (CoT) trace is correct are compared by AUC, but deploying one requires a threshold with a guarantee. We ask what distribution-free selective guarantees deliver for CoT verifiers at realistic calibration budgets of tens to a few hundred labelled problems, using seven open models, five verifier signals and 37,000 graded traces. The central observation is validity by abstention: an $(\alpha,\delta)$-valid procedure that issues a certificate with probability $P_{\rm fire}$ bounds the failure probability of an issued certificate only by $\delta/P_{\rm fire}$, so a certificate that rarely fires can be valid and wrong every time it is used. In a simulation with known risk the standard certificate fails in at most 0.3% of calibration draws but in up to 69% of those in which it fires. A certification floor and a lattice condition for Benjamini-Hochberg conformal selection explain why certificates 
    
[^195]: 非配对典型相关分析

    Unpaired Canonical Correlation Analysis

    [https://arxiv.org/abs/2610.09530](https://arxiv.org/abs/2610.09530)

    提出UCCA方法，通过建立二次分配问题与CCA之间的理论联系，首次实现仅使用非配对数据学习最大化真实潜在配对相关性的线性投影。

    

    典型相关分析（CCA）是多视图共享空间学习的一种基础方法。然而，它对配对数据的严格依赖构成了重大限制，因为这类数据通常难以获取甚至完全不可得。在本文中，我们提出了非配对典型相关分析（UCCA），这是一种新颖的方法，它通过学习线性投影来最大化真实潜在配对的相关性，而无需在训练过程中访问任何配对样本。我们首先建立了将二次分配问题（QAP）与CCA联系起来的理论结果。利用这些理论见解，我们推导出一种仅从非配对数据中最大化相关性的实用方法。据我们所知，UCCA是首个在严格非配对设置下学习最大相关投影的方法。我们在真实世界的多模态数据集上验证了UCCA，证明其在恢复方面显著优于近期的非配对对齐基线方法。

    arXiv:2610.09530v1 Announce Type: cross  Abstract: Canonical Correlation Analysis (CCA) is a fundamental method for multiview shared space learning. However, its strict reliance on paired data poses a significant limitation, as such data is often difficult to obtain or entirely unavailable. In this paper, we present Unpaired CCA (UCCA), a novel method that learns linear projections to maximize the correlation of the true underlying pairing without access to any paired samples during training. We first establish theoretical results connecting the Quadratic Assignment Problem (QAP) to CCA. Leveraging these theoretical insights, we derive a practical method to maximize correlation exclusively from unpaired data. To the best of our knowledge, UCCA is the first approach to learn maximally correlated projections in a strictly unpaired setting. We validate UCCA on real-world multi-modal datasets, demonstrating that it significantly outperforms recent unpaired alignment baselines in recovering
    
[^196]: 先对齐再融合：面向无真实标签监督的参考空间校准

    Align Before You Combine: Reference Space Calibration for Supervision Without Ground Truth

    [https://arxiv.org/abs/2610.09525](https://arxiv.org/abs/2610.09525)

    提出一种无需真实标签的校准优先框架，通过合成有序参考空间在融合前对齐各子集评分器，其校准独立于训练集分布，在多个基准上优于未校准平均和最佳单个评分器。

    

    我们提出了一种“校准优先”的框架，能够在没有真实标签或共享标注空间的情况下生成监督评分。该框架在融合之前，利用一个合成的有序参考空间来对齐各子集特定的评分器。这个参考空间由表示潜在概念的有序校准特征构建而成，为原本无法相互比较的评分器输出提供了一个共同的尺度。由于我们的校准过程使用的是参考空间而非训练样本，因此它独立于训练集的经验分布。在三个基准数据集上，我们的框架始终优于未校准的简单平均方法，并在评估指标上取得了比最佳单个评分器更高的主指标点估计值。相对于依赖样本的基线方法，其表现因领域而异，在 Ames Housing 数据集上的绝对差异低于 0.02，在 Breast Cancer 数据集上的绝对差异低于 0.01。

    arXiv:2610.09525v1 Announce Type: new  Abstract: We introduce a calibration-first framework that produces supervision scores without access to ground-truth labels or a shared annotation space. Our framework aligns subset-specific scorers using a synthetic ordinal reference space before fusion. This reference space is constructed from ordered calibration features that represent the latent concept, providing a common scale on which otherwise incomparable scorer outputs can be aligned. Because our calibration procedure uses the reference space rather than training samples, it is independent of the training set's empirical distribution. Across three benchmark datasets, our framework consistently outperforms uncalibrated averaging and achieves higher primary-metric point estimates on the evaluation metrics than the best individual scorer. Performance relative to sample-dependent baselines varies by domain, with absolute differences below 0.02 on Ames Housing and below 0.01 on Breast Cancer 
    
[^197]: 反射锚定朗之万算法

    Reflected Anchored Langevin Algorithms

    [https://arxiv.org/abs/2610.09522](https://arxiv.org/abs/2610.09522)

    本文提出反射锚定朗之万动力学（RALD）及其蒙特卡洛算法 RALMC，通过光滑锚定参考势能与状态相关缩放因子，突破了传统方法要求对数密度可微的限制，实现了约束域上不可微目标分布的高效采样，并给出了显式的收敛界与迭代复杂度。

    

    机器学习中用于约束采样的朗之万算法（如基于反射朗之万动力学离散化的投影朗之万蒙特卡洛方法）通常要求对数密度可微，这限制了它们的应用范围。本文提出了反射锚定朗之万动力学（RALD），这是一种能够在约束域上收敛到不可微目标分布的反射扩散过程。该方法使用一个光滑的锚定参考势能，并将其反射朗之万动力学的漂移项与噪声协方差乘以相同的状态相关缩放因子。对该动力学采用带投影的 Euler-Maruyama 离散化，即可得到反射锚定朗之万蒙特卡洛（RALMC）算法。我们证明了 RALMC 在 2-Wasserstein 距离下到目标分布的显式收敛界与迭代复杂度。文中还提供了数值实验，用以验证理论预测并展示该算法的经验性能。

    arXiv:2610.09522v1 Announce Type: cross  Abstract: First order Langevin algorithms for constrained sampling in machine learning, such as projected Langevin Monte Carlo which are based on discretizations of reflected Langevin dynamics, require differentiable log densities that limits their applicability. This paper introduces reflected anchored Langevin dynamics (RALD), a reflected diffusion that converges to non-differentiable targets on constrained domains. The method uses a smooth anchored reference potential and multiplies the drift and noise covariance of its reflected Langevin dynamics by the same state dependent scaling factor. Its Euler-Maruyama discretization with projection gives reflected anchored Langevin Monte Carlo (RALMC) algorithm. We prove explicit convergence bounds and iteration complexity for RALMC in the 2-Wasserstein distance to the target distribution. Numerical experiments are provided to illustrate the theoretical predictions and the empirical performance of the
    
[^198]: 基于滞后数据快照训练的模型差异化刷新策略：从单一年龄等价极限到最优分段分配

    Differential Refresh Policies for Models Trained on Lagging Data Snapshots: From a Single-Age Equivalence Limit to an Optimal Per-Segment Allocation

    [https://arxiv.org/abs/2610.09519](https://arxiv.org/abs/2610.09519)

    本文证明任何仅基于单一全局数据年龄的刷新触发器在操作上都等价于简单的均匀计时器，并提出按分段差异化赋予各自年龄与刷新间隔的策略，在刷新成本可分离的条件下实现最优的刷新资源分配。

    

    生产环境中的机器学习模型是具有时间边界的训练快照的派生产物：一个已部署的模型是构建在训练数据截面之上的物化视图，从建成那一刻起就开始老化。一种常见的应对方法是用自适应触发器取代固定的重训练周期——即一种加权陈旧度评分，当累积的源风险越过阈值时便触发重训练。我们证明这是一个错误的杠杆，并找出了正确的杠杆。首先，是一个等价极限：任何作为单一共享全局训练数据年龄的静态严格单调函数的刷新触发器，在操作上都等价于一个经过校准的均匀年龄计时器，因此全局陈旧度预算——无论其如何精细地对分段、来源和敏感性进行加权——都不包含时钟所不具备的任何调度信息。该极限同时也指出了摆脱它的方法：对各个分段进行差异化刷新，赋予每个分段自己的年龄和刷新间隔，这在刷新成本可分离的情况下是有意义的……

    arXiv:2610.09519v1 Announce Type: new  Abstract: Production machine-learning models are derived artifacts of time-bounded training snapshots: a deployed model is a materialized view over a training cut that ages the instant it is built. A common response is to replace the fixed retraining cadence with an adaptive trigger -- a weighted staleness score that retrains when accumulated source risk crosses a threshold. We show this is the wrong lever, and identify the right one. First, an equivalence limit: any refresh trigger that is a static, strictly monotone function of a single shared global training-data age is operationally equivalent to a calibrated uniform age timer, so a global staleness budget, however elaborately it weights segments, sources, and sensitivities, carries no scheduling information a clock does not. The limit also shows how to escape it: refresh segments differentially, giving each its own age and refresh interval, which is meaningful when refresh cost is separable a
    
[^199]: 它并非在看见危险：冻结视觉-语言安全评分度量的只是其描述文本库

    It Is Not Seeing the Hazard: A Frozen Vision-Language Safety Score Measures Its Caption Bank

    [https://arxiv.org/abs/2610.09517](https://arxiv.org/abs/2610.09517)

    本文通过受控评估证明，冻结的CLIP安全评分并未真正检测到危险本身，而主要是对其描述文本库以及提示结构、嵌入几何和相机视角等混杂因素作出反应。

    

    冻结的视觉-语言模型正越来越多地为强化学习提供安全信号，其使用的前提假设是：与描述危险的语言的相似度即代表危险本身。然而，策略回报与碰撞率无法揭示一个评分究竟是在检测危险，还是仅仅对场景中与危险相关的特征作出反应。基于VLM的方法通过将图文相似度转化为奖励、代价或置信度权重，已在驾驶与安全强化学习基准上取得了性能提升；这类信号有望减少对手工设计反馈的依赖，但也可能反映的是提示词结构、嵌入几何特性或相机视角，使其安全含义未经检验。为填补这一空白，我们对一种冻结的CLIP提示词间隔安全评分进行了受控评估：将该评分应用于由从未接收该评分的策略所生成的轨迹，把接触前观测与具有相近危险几何结构的无接触观测相匹配，并变换……（原文摘要在此处截断）

    arXiv:2610.09517v1 Announce Type: cross  Abstract: Frozen vision-language models increasingly provide safety signals for reinforcement learning. Their use assumes that similarity to language describing danger indicates the hazard itself. Yet policy return and collision rate cannot reveal whether a score detects hazards or responds to correlated features of the scene. VLM-based methods have reported gains in driving and safe-RL benchmarks by converting image-text similarity into rewards, costs, or confidence weights. Such signals promise to reduce reliance on manually designed feedback. They may also reflect prompt structure, embedding geometry, or camera viewpoint, leaving their safety meaning unverified. To address this gap, we present a controlled evaluation of a frozen CLIP prompt-margin safety score. We apply the score to trajectories generated by policies that never receive it, match pre-contact observations to contact-free observations with comparable hazard geometry, and vary th
    
[^200]: 物理信息神经可塑性：能够自我重塑的偏微分方程求解器

    Physics-Informed Neural Plasticity: PDE Solvers That Reshape Themselves

    [https://arxiv.org/abs/2610.09510](https://arxiv.org/abs/2610.09510)

    提出物理信息神经可塑性范式及ReCAP求解器，使PDE求解器在优化过程中根据未解决的物理问题动态重塑自身表示结构，通过局部加密、分裂、剪枝和合并自适应地重新分配模型容量。

    

    物理信息神经PDE求解器通过调整其参数来满足控制方程，但其表示结构在训练过程中通常保持固定。这种刚性与在空间和时空上具有强烈异质复杂性的PDE解匹配不佳，导致在物理困难之处模型容量不足，而在物理简单之处容量冗余。我们提出了物理信息神经可塑性这一范式，其中表示本身在优化过程中根据未解决的物理问题进行动态重塑。我们通过面向PDE的表示容量自适应方法（ReCAP）来实现这一原则，这是一种高斯局部化求解器，通过局部加密、残差导向分裂、基于门的剪枝和函数感知合并来动态重新分配模型容量。ReCAP利用责任加权误差指标和残差能量的几何特性来确定在何处以及如何进行细化。为了限制干扰……

    arXiv:2610.09510v1 Announce Type: new  Abstract: Physics-informed neural PDE solvers adapt their parameters to satisfy governing equations, yet their representational structure typically remains fixed throughout training. This rigidity is poorly matched to PDE solutions with strongly heterogeneous complexity across space and space--time, leaving capacity insufficient where the physics is difficult and redundant where it is simple. We introduce physics-informed neural plasticity, a paradigm in which the representation itself reshapes during optimization in response to unresolved physics. We instantiate this principle with Representation Capacity Adaptation for PDEs (ReCAP), a Gaussian-localized solver that dynamically redistributes capacity through local enrichment, residual-directed splitting, gate-based pruning, and function-aware merging. ReCAP uses responsibility-weighted error indicators and the geometry of residual energy to determine where and how to refine. To limit the disturba
    
[^201]: TERRA：通过时间效应表征与关系对齐学习可迁移的潜在动作

    TERRA: Learning Transportable Latent Actions through Temporal Effect Representation and Relational Alignment

    [https://arxiv.org/abs/2610.09509](https://arxiv.org/abs/2610.09509)

    提出TERRA框架，用紧凑的时间效应表征（净特征变化加低阶窗口内动力学）学习连续潜在动作，并通过效应锚定迁移（EAT）使潜在动作能在不同初始状态间迁移复用。

    

    潜在动作通过从视觉状态转移中推断出的类动作编码来监督机器人策略，其有用性取决于两个问题：编码从一次状态转移中保留了什么，以及当它在不同初始状态下被复用时是否仍具有相同含义。第一个问题是时间上的张力：仅用终点差异会丢弃运动展开过程的信息，而使用完整序列则会引入无关干扰变化。第二个问题被重构方法所遗留未解，因为重构方法只在潜在编码与其来源状态一同出现时才进行观察。我们认为这两个问题可以在同一框架下得到解答。TERRA（时间效应表征与关系对齐）用一种紧凑的时间效应来描述状态转移——即净特征变化加上一个低阶的窗口内动力学分量，并从该效应中学习连续潜在编码。同一效应空间随后作为复用的参照：效应锚定迁移（EAT）在其他初始状态下解码潜在编码……（原文摘要在此处截断）

    arXiv:2610.09509v1 Announce Type: cross  Abstract: Latent actions supervise robot policies with action-like codes inferred from visual transitions, and their usefulness hinges on two questions: what a code keeps from a transition, and whether it still means the same thing when reused in a different initial state. The first is a tension in time: an endpoint difference discards how motion unfolds, while the full sequence admits nuisance variation. The second is left open by reconstruction, which only ever observes a latent together with the state it came from. We argue that both questions can be answered in the same place. TERRA (Temporal Effect Representation and Relational Alignment) describes a transition by a compact temporal effect, its net feature change together with a low-order within-window dynamics component, and learns a continuous latent from this effect. The same effect space then serves as the reference for reuse: Effect-Anchored Transport (EAT) decodes a latent in other in
    
[^202]: 平均安全，尾部不安全：回合成本尾部何时可控？

    Safe on Average, Unsafe in the Tail: When Is the Episodic-Cost Tail Controllable?

    [https://arxiv.org/abs/2610.09508](https://arxiv.org/abs/2610.09508)

    本文提出用CVaR₀.₁度量回合成本尾部风险，揭示了仅满足平均成本约束的安全强化学习策略在最差回合中可能不安全，并研究在保持回报的前提下这种尾部违规何时可控。

    

    安全强化学习旨在寻找在满足累积成本约束的同时最大化回报的策略。大多数方法将约束施加在期望回合成本上。因此，标准评估仅报告平均回合成本，而未刻画成本在各回合之间的分布情况。因此，满足平均成本准则的策略在其最差回合中可能仍然不安全。仅报告平均成本既无法识别这种尾部违规，也无法表明能否在保持回报的同时将其控制在预算范围内。在本工作中，我们使用 CVaR₀.₁（即最差 10% 回合的平均成本）来度量回合成本尾部。当 CVaR₀.₁ 处于安全预算之内时，我们将该策略归类为尾部安全。这使得我们能够首先识别出平均安全但尾部不安全的策略，然后研究其尾部违规能否在保持回报的同时得到控制。为识别尾部（摘要至此截断）

    arXiv:2610.09508v1 Announce Type: new  Abstract: Safe reinforcement learning seeks policies that maximize return while satisfying constraints on cumulative cost. Most methods impose these constraints on expected episodic cost. Consequently, standard evaluations report mean episodic cost without characterizing how cost is distributed across episodes. A policy that satisfies the mean-cost criterion may therefore remain unsafe in its worst episodes. Mean-cost reporting neither identifies this tail violation nor shows whether it can be brought within budget while preserving return. In this work, we measure the episodic-cost tail using $\mathrm{CVaR}_{0.1}$, the average cost of the worst $10\%$ of episodes. We classify a policy as tail-safe when $\mathrm{CVaR}_{0.1}$ is within the safety budget. This allows us first to identify policies that are safe on average but unsafe in the tail and then to study whether their tail violations can be controlled while preserving return. To identify tail-
    
[^203]: 基于伴随方法的随机多尺度生物过程数字孪生校准与最优控制

    Adjoint-Based Calibration and Optimal Control of Stochastic Multiscale Bioprocess Digital Twins

    [https://arxiv.org/abs/2610.09505](https://arxiv.org/abs/2610.09505)

    本文提出了一个基于伴随敏感性分析的偏差感知数字孪生校准与最优控制框架，通过拟似然估计、矩展开和正反向伴随方法量化校准不确定性对策略性能的传播影响，实现了多尺度生物过程的不确定性感知策略优化与自适应实验设计。

    

    我们在生物系统体系（Bio-SoS）范式下，开发了一个面向多尺度生物过程模型的、具有偏差感知能力的数字孪生校准与控制框架。该数字孪生由随机微分方程（SDE）模型表示，并利用拟似然估计和伴随敏感性分析从稀疏的离散观测数据中进行校准。基于SDE生成算子的矩展开刻画了截断引起的参数偏差，而正反向伴随方法则量化了校准不确定性如何传播至价值函数和策略性能。由此得到的参数误差分布既支持面向策略的自适应实验设计，也支持通过二阶高斯平均目标实现的不确定性感知策略优化。我们刻画了所得探索准则的渐近行为，并推导了在优化策略下物理系统的性能表现。为了实现这些想法，我们开发了……（原文摘要在此处截断）

    arXiv:2610.09505v1 Announce Type: cross  Abstract: We develop a bias-aware digital-twin calibration and control framework for multiscale bioprocess models within a biological systems-of-systems (Bio-SoS) paradigm. The digital twin is represented by a stochastic differential equation (SDE) model and calibrated from sparse, discrete observations using quasi-likelihood estimation and adjoint sensitivity analysis. SDE generator-based moment expansions characterize truncation-induced parameter bias, while forward-backward adjoints quantify how calibration uncertainty propagates to value functions and policy performance. The resulting parameter-error distribution supports both policy-directed adaptive experimental design and uncertainty-aware policy optimization through a second-order Gaussian-averaged objective. We characterize the asymptotic behavior of the resulting exploration criterion and derive a physical-system performance under the optimized policy. To implement these ideas, we deve
    
[^204]: SpatialUQ：基于空间一致性的黑盒视觉模型事后不确定性量化

    SpatialUQ: Post-Hoc Uncertainty Quantification from Spatial Consistency in Black-Box Vision Models

    [https://arxiv.org/abs/2610.09498](https://arxiv.org/abs/2610.09498)

    SpatialUQ通过仅六次前向传播测量全局预测与固定空间裁剪区域之间的Jensen-Shannon散度，在无需访问模型内部的情况下，以MC-Dropout五分之一的计算量将ChestX-ray14上的失败检测AUC从0.664提升至0.784，并提供更优的校准性能。

    

    临床视觉模型通常以冻结的黑盒形式部署，在推理时无法访问其内部结构、无法重新训练，也没有真实标签可用。我们提出SpatialUQ，一种仅使用输出概率的事后不确定性量化方法。该方法通过六次确定性前向传播，测量全局预测与五个固定空间裁剪区域均值之间的Jensen-Shannon散度。其前提很简单：可信的预测应具有空间一致性。在NIH ChestX-ray14数据集上（DenseNet-121，N=25,596），我们的多裁剪不确定性分数（MUS）在失败检测上达到0.784的AUC，而MC-Dropout仅为0.664（p<10⁻⁶），且计算量仅为后者的五分之一；同时具备原生校准能力（SCE=0.049，而l1方法为0.127），是AUC超过0.78的方法中校准效果最佳的。将MUS与熵、置信度和l1进行有监督融合后可达到0.832，优于五成员集成模型的0.813。MUS随模型质量提升而相应扩展。

    arXiv:2610.09498v1 Announce Type: cross  Abstract: Clinical vision models are often deployed as frozen black boxes with no access to internals, retraining, or ground truth at inference time. We introduce \textbf{SpatialUQ}, a post-hoc uncertainty method using only output probabilities. It measures the Jensen-Shannon divergence between the global prediction and the mean of five fixed spatial crops in six deterministic forward passes. The premise is simple, trustworthy predictions are spatially consistent. On NIH ChestX-ray14 (DenseNet-121, $N{=}25{,}596$), our Multicrop Uncertainty Score (MUS) reaches $0.784$ failure-detection AUC versus $0.664$ for MC-Dropout ($p{<}10^{-6}$) at one-fifth the compute, with native calibration ($\text{SCE}{=}0.049$ vs.\ $0.127$ for $\ell_1$), the best-calibrated among methods above 0.78 AUC. A supervised fusion of MUS with entropy, confidence, and $\ell_1$ reaches $0.832$, outperforming a five-member ensemble ($0.813$). MUS scales with model quality, reac
    
[^205]: 稀疏特征策略遗忘减轻视觉-语言-动作模型中的状态幻觉

    Sparse Feature Policy Unlearning Mitigates State Hallucination in Vision-Language-Action Models

    [https://arxiv.org/abs/2610.09496](https://arxiv.org/abs/2610.09496)

    提出SOUL方法，利用稀疏自编码器识别与状态幻觉相关的稀疏特征，并据此选择性地遗忘VLA模型中与幻觉行为相关的策略知识，从而有效减轻视觉-语言-动作模型中的状态幻觉问题

    

    视觉-语言-动作模型通过利用预训练视觉-语言模型的丰富表示，在机器人操作任务中展现出强大的泛化能力。然而，它们在真实世界环境中的部署仍然受到反复出现的不稳定行为的限制。在这项工作中，我们研究了状态幻觉，这是一种反复出现的失败模式，即VLA模型持续执行动作，仿佛某个尚未实现的机器人-物体状态已经达成。我们的分析发现，状态幻觉与对任务相关视觉区域的注意力减弱同时出现，并且通过稀疏自编码器进行的机制性解释表明，与幻觉相关的稀疏特征在这些失败发生时会被激活。基于这一分析，我们提出了SOUL（稀疏特征策略遗忘），它选择性地遗忘与状态幻觉行为相关的策略知识，其中稀疏特征是从幻觉失败和成功行为中识别出来的

    arXiv:2610.09496v1 Announce Type: cross  Abstract: Vision-Language-Action (VLA) models have shown strong generalization in robotic manipulation by leveraging rich representations from pretrained vision-language models. However, their deployment in real-world environments remains limited by recurring unreliable behaviors. In this work, we study state hallucination, a recurring failure pattern in which a VLA continues acting as if an unrealized robot-object state had been achieved. Our analyses find that state hallucination coincides with weakened attention to task-relevant visual regions, and a mechanistic interpretation via sparse autoencoders reveals that hallucination-associated sparse features are activated when these failures occur. Based on this analysis, we propose SOUL (Sparse feature pOlicy UnLearning), which selectively unlearns policy knowledge associated with state hallucination behaviors, where sparse features identified from hallucination failures and successful behaviors 
    
[^206]: 对应关系即决策：面向决策中心模式匹配的JevNexus

    Correspondences as Decisions: JevNexus for Decision-Centric Schema Matching

    [https://arxiv.org/abs/2610.09487](https://arxiv.org/abs/2610.09487)

    JevNexus通过类型化成对决策结合证据门控机制，仅在证据不一致且边际较小时才调用昂贵的列表级精炼，在模式匹配精度与最先进方法相当的同时将延迟降低约7.75倍。

    

    模式匹配领域越来越多地使用生成式语言模型对检索到的列候选进行重排序，尽管其底层任务实际上是一个有界的对应关系决策问题。我们提出了JevNexus，它将类型化的成对决策与模式/实例证据相结合，并且仅在证据不一致且融合边际较小时才调用列表级精炼。评估涵盖来自六个基准测试族的561个案例。JevNexus获得的数据集宏观MRR和Hits@1分别为0.930和0.909，而Magneto分别为0.926和0.903，同时将平均延迟从123.452秒降低到15.929秒（7.750倍加速）。配对分析显示MRR和Hits@1均无统计学显著差异。该门控机制仅对5.665%的源列调用列表级精炼，避免了无条件精炼造成的性能退化。

    arXiv:2610.09487v1 Announce Type: cross  Abstract: Schema matching increasingly uses generative language models to rerank retrieved column candidates, although the underlying task is a bounded correspondence decision. We present JevNexus, which combines typed pairwise decisions with schema/instance evidence and invokes listwise refinement only when the evidence disagrees and the fused margin is small. The evaluation covers 561 cases from six benchmark families. JevNexus obtains dataset-macro MRR and Hits@1 of 0.930 and 0.909, compared with 0.926 and 0.903 for Magneto, while reducing mean latency from 123.452 to 15.929 seconds (7.750). Paired analysis finds no statistically significant difference in either MRR or Hits@1. The gate invokes listwise refinement for only 5.665% of source columns and avoids the degradation caused by unconditional refinement. Code and experimental artifacts are available at https://github.com/RazeenLI/JevNexus.
    
[^207]: CHASE：面向几何感知模型工程的通道对齐结构利用

    CHASE: Channel-Aligned Structure Exploitation for Geometry-Aware Model Engineering

    [https://arxiv.org/abs/2610.09476](https://arxiv.org/abs/2610.09476)

    本文提出CHASE框架，将几何与谱对齐（GSA）所刻画的结构特征应用于参数高效微调、剪枝补偿、模型合并、KV共享表示和神经元分组等六类模型工程任务，并开发了CAGA、SAKV、CAPS三种新方法。

    

    几何与谱对齐（GSA）通过谱集中性、物理通道对齐、支持结构以及奇异基的变化来刻画训练后的网络。在本文中，我们提出CHASE（通道对齐结构利用），将这些结构应用于实际的模型设计。CHASE涵盖了模型修改、重构和压缩方面的六个应用。CORA、COEC和CORAM将GSA应用于参数高效微调、结构化剪枝补偿和模型合并。我们进一步开发了三种新方法：CAGA利用GSA识别可以共享KV表示的多头注意力头，并通过几何对齐和低秩子空间提取来构建共享的键和值头；SAKV利用GSA确定哪些相邻层可以共享低秩KV缓存表示以及每个层组保留的秩；CAPS利用GSA的谱结构对输出神经元进行分组并……（摘要在此处被截断）

    arXiv:2610.09476v1 Announce Type: cross  Abstract: Geometric and Spectral Alignment (GSA) characterizes trained networks through spectral concentration, physical-channel alignment, support structure, and changes in singular bases. In this paper, we propose CHASE (Channel-Aligned Structure Exploitation) to use these structures in practical model design. CHASE covers six applications across model modification, reconfiguration, and compression. CORA, COEC, and CORAM apply GSA to parameter-efficient finetuning, structured-pruning compensation, and model merging. We further develop three new methods. CAGA uses GSA to identify multi-head attention heads that can share a KV representation and constructs the shared key and value heads through geometric alignment and low-rank subspace extraction. SAKV uses GSA to determine which adjacent layers can share a low-rank KV-cache representation and the retained rank for each layer group. CAPS uses GSA spectral structure to group output neurons and se
    
[^208]: MORA：面向漂移鲁棒时间序列异常检测的观测变化建模

    MORA: Modeling Observed Changes for Drift-Robust Time-Series Anomaly Detection

    [https://arxiv.org/abs/2610.09473](https://arxiv.org/abs/2610.09473)

    MORA提出了一种漂移鲁棒的时间序列异常检测框架，通过从成对的短期与长期视图重建同一局部目标，利用重建差距进行“时间变化消歧”，并以保守的修正机制调整异常分数，从而区分真实异常与分布漂移。

    

    时间序列异常检测（TSAD）旨在识别与从历史数据中学习到的模式相偏离的情况。在非平稳环境中，分布漂移和真实异常都可能引起相似的局部变化，这使得难以判断某个偏离究竟是反映了异常，还是反映了不断演变的上下文。现有方法通常是对检测到的漂移进行适应，或学习对漂移不敏感的表示，但并未解决这种歧义。我们将该问题定义为“时间变化消歧”：即判断某个局部偏离是否可以被更广泛的时间演化所解释。我们提出了MORA，一个漂移鲁棒的时间序列异常检测框架，它从成对的短期视图和长期视图中重建同一个局部目标。重建差距用于衡量上下文对该局部偏离的支持程度，同时一个依赖于数据的修正机制会保守地调整主要的局部异常分数。只有当上下文能够改善对同一目标的重建时，才能降低该分数。

    arXiv:2610.09473v1 Announce Type: new  Abstract: Time-series anomaly detection (TSAD) identifies deviations from patterns learned from historical data. In non-stationary settings, distribution drift and true anomalies can cause similar local changes, making it difficult to tell whether a deviation reflects abnormality or evolving context. Existing methods typically adapt to detected shifts or learn drift-insensitive representations, but do not resolve this ambiguity. We define this problem as \emph{temporal change disambiguation}: determining whether a local deviation is explained by broader temporal evolution. We introduce MORA, a drift-robust TSAD framework that reconstructs the same local target from paired short- and long-term views. The reconstruction gap measures contextual support for a local deviation, and a data-dependent correction mechanism conservatively adjusts the primary local anomaly score. Context can only reduce the score when it improves reconstruction of the same ta
    
[^209]: 上下文学习者何时应扩展其假设空间？

    When Should an In-Context Learner Expand Its Hypothesis Space?

    [https://arxiv.org/abs/2610.09471](https://arxiv.org/abs/2610.09471)

    该论文将上下文学习中“何时扩展假设空间”的问题形式化为有代价的序贯决策，并通过结构修正环境证明：修正决策是一个由价格、时间范围和查询等价值因素决定的价值边界，而非单纯的证据阈值。

    

    学习系统能够在熟悉的模型族内快速适应。更困难的一步出现在更早的阶段：从可能是噪声、例外、族内变化或族外结构的观察中，判断开启一个更丰富的模型族是否值得其代价。我们将此视为一个有代价的序贯决策问题：预测失败必须被转化为结构性证据，证据被转化为扩展的价值，价值再被转化为行动。结构修正环境从每种来源产生匹配的失败案例，独立于证据地改变扩展的价格与剩余时间范围，并支持精确的贝叶斯计算以及一次性决策的精确规范性解。其解表明，修正是一个价值边界而非证据阈值：同一历史在不同价格、时间范围和公告查询下会有不同的最优行动，局部修复与扩展之间的边界由规则……所设定

    arXiv:2610.09471v1 Announce Type: new  Abstract: Learning systems adapt quickly inside a familiar family of models. The harder step comes earlier: deciding, from observations that could be noise, an exception, a change within the family or structure outside it, whether opening a richer family is worth its cost. We treat this as a costly sequential decision: prediction failure must be turned into structural evidence, evidence into a value of expansion, and value into action. The Structural Revision Environment produces matched failures from each source, varies the price of expansion and the remaining horizon independently of the evidence, and admits exact Bayesian calculations and an exact normative solution of the one-shot decision. Its solution shows that revision is a value boundary and not an evidence threshold: one history has different optimal actions under different prices, horizons and announced queries, the boundary between local repair and expansion is set by the inputs a rule
    
[^210]: 一种不变切线角描述子与用于二维碎片邻接预测的带状U-Net

    An Invariant Tangent-Angle Descriptor and a Band U-Net for 2D Fragment Adjacency Prediction

    [https://arxiv.org/abs/2610.09459](https://arxiv.org/abs/2610.09459)

    该论文提出了一种在构造上对碎片旋转和轮廓起始点选择均不变的切线角描述子来替代原有的局部评分，并用带状U-Net替换最终分类器，从而改进了基于轮廓的二维碎片邻接预测的两阶段架构。

    

    本文研究基于轮廓的二维碎片对之间邻接关系的预测问题。我们改进了Beaulac论文中提出的两阶段架构，该架构使用旋转等变的孪生卷积神经网络对两个碎片两条轮廓上的局部图像窗口对进行评分。评分被收集在一个邻接矩阵中，随后由ResNet检测其中揭示两个碎片邻接关系的部分反对角带。在本工作中，我们保留了整体流程，但将局部评分替换为轮廓窗口切线角轮廓的比较，使其在构造上对碎片旋转具有不变性，并且与所选的轮廓起始点无关。这种局部比较既可以采用无需训练的似然比方法，也可以采用在对应点上训练的小型一维卷积模型。此外，我们还将最终的分类器替换为一个带状U-Net，用于对带进行分割和分类。

    arXiv:2610.09459v1 Announce Type: cross  Abstract: This paper addresses the prediction of adjacency between pairs of 2D fragments based on their contours. We improved the two-stage architecture proposed in Beaulac's thesis, in which a rotation-equivariant Siamese convolutional neural network scores pairs of local image windows along the two contours of two fragments. The scores are gathered in an adjacency matrix in which a ResNet detects the partial anti-diagonal band that reveals the adjacency of two fragments. In the current work, we keep the pipeline and replace the local score by a comparison of tangent-angle profiles of contour windows, making it, by construction, invariant to fragment rotation and agnostic to the selected contour-starting point. These adaptations may be either a training-free likelihood ratio or a small one-dimensional convolutional model trained on corresponding points. We also replaced the final classifier by a band U-Net that segments the band and classifies 
    
[^211]: DSReg：无需重建即可可证明地恢复个体世界潜在变量

    DSReg: Provably Recovering Individual World Latents without Reconstruction

    [https://arxiv.org/abs/2610.09457](https://arxiv.org/abs/2610.09457)

    该论文提出“结构多样性”条件与依赖稀疏正则化方法DSReg，在无需重建、解码器或标签的情况下，可证明地恢复个体的世界潜在变量。

    

    从非线性ICA到字典学习和因果表征学习，恢复世界个体潜在变量的方法通常通过重建、辅助监督或分布不对称性（如非高斯性）将潜在变量锚定到观测上。而缺乏这些锚定机制的方法，包括联合嵌入预测架构（JEPAs），只能在线性变换的意义上识别潜在状态，因此个体潜在变量仍然是混合不可分的。我们弥合了这一差距：个体世界潜在变量可以在没有重建、没有解码器、没有标签的情况下被可证明地恢复。关键条件是结构多样性：不同的潜在变量会在观测上留下不同的依赖足迹，就像没有两片雪花是相同的一样。基于LeJEPA所提供的线性可辨识性，我们证明在结构多样性条件下，DSReg（依赖稀疏正则化）可以恢复个体世界潜在变量（精确到符号置换），而无需……

    arXiv:2610.09457v1 Announce Type: new  Abstract: Methods that recover individual latent variables of the world, from nonlinear ICA to dictionary learning and causal representation learning, anchor the latents to observations through reconstruction, auxiliary supervision, or distributional asymmetries such as non-Gaussianity. Methods without these anchors, including joint-embedding predictive architectures (JEPAs), identify the latent state only up to a linear transformation, so individual latents remain mixed. We close this gap: individual world latents can be provably recovered with no reconstruction, no decoder, and no labels. The key condition is Structural Diversity: different latents leave distinct dependency footprints on observations, just as no two snowflakes are alike. Building on the linear identifiability that LeJEPA provides, we prove that under Structural Diversity, DSReg (Dependency-Sparsity Regularization) recovers individual world latents up to signed permutation, witho
    
[^212]: GeoPrior-Mamba：结合结构化过程先验与Mamba模型的精细分辨率XCO2重建

    GeoPrior-Mamba: Structured Process Priors with Mamba for Fine-Resolution XCO2 Reconstruction

    [https://arxiv.org/abs/2610.09456](https://arxiv.org/abs/2610.09456)

    提出GeoPrior-Mamba框架，创新性地利用离线语言模型将生物圈吸收、生态系统呼吸和人为排放的过程知识转化为结构化先验表，并结合多方向Mamba架构，实现从稀疏卫星观测中重建精细分辨率的XCO2场。

    

    从稀疏的卫星观测数据中重建精细分辨率的柱平均干空气CO2（XCO2）场，要求模型能够推断仅受直接测量弱约束的空间结构。现有的基于学习的方法通常将环境协变量视为普通的数值输入，因此必须主要依靠稀疏的监督信号来学习异构的源汇关系。我们提出了GeoPrior-Mamba，这是一个多方向Mamba框架，并通过离线语言模型诱导的结构化过程先验进行增强。我们并非使用语言模型来预测XCO2，而是在训练之前利用它将生物圈吸收、生态系统呼吸和人为排放的相关过程知识组织成确定性的先验表。这些先验利用地理、生态、排放相关和季节信息进行空间实例化，并通过……自适应地注入到重建骨干网络中。

    arXiv:2610.09456v1 Announce Type: new  Abstract: Reconstructing fine-resolution column-averaged dry-air CO2 (XCO2) fields from sparse satellite observations requires models to infer spatial structure that is only weakly constrained by direct measurements. Existing learning-based methods typically treat environmental covariates as ordinary numerical inputs and must therefore learn heterogeneous source-sink relationships largely from sparse supervision. We introduce GeoPrior-Mamba, a multi-directional Mamba framework augmented with offline language-model-induced structured process priors. Rather than using a language model to predict XCO2, we use it before training to organize relative process knowledge for biospheric uptake, ecosystem respiration, and anthropogenic emissions into deterministic prior tables. These priors are spatially instantiated using geographic, ecological, emission-related, and seasonal information and are adaptively injected into the reconstruction backbone through 
    
[^213]: 一种用于热-机械裂纹扩展的扩展深度能量方法

    An extended deep energy method for thermo-mechanical crack propagation

    [https://arxiv.org/abs/2610.09433](https://arxiv.org/abs/2610.09433)

    提出一种扩展深度能量方法，用两个神经网络表示温度场与位移场，并通过不连续标量嵌入函数嵌入尖锐折线裂纹、以可训练Williams展开富集裂尖位移，首次将含裂纹域的瞬态热传导与热应力驱动的裂纹扩展统一求解。

    

    热-机械断裂将含裂纹域上的瞬态热传导与随温度和位移演化而增长的裂纹耦合在一起。神经能量求解器最初被提出用于相场断裂问题，后来被扩展为通过网络输入来表示尖锐裂纹，但在这些求解器中，含裂纹域上的热传导以及由此产生的热应力驱动下的裂纹扩展尚未被同时处理。我们提出了一种用于热-机械裂纹扩展的扩展深度能量方法，其中裂纹始终保持为尖锐的折线。两个神经网络分别表示温度场和位移场，并通过一个标量嵌入函数接收裂纹信息，该函数在裂纹处不连续而在其他位置光滑，使得两个场都能跨裂纹发生跳跃而无需正则化长度；同时，位移场在裂纹尖端附近通过具有可训练幅值的Williams展开进行富集。这两个场通过最小化……

    arXiv:2610.09433v1 Announce Type: new  Abstract: Thermo-mechanical fracture couples transient heat conduction on a cracked domain with a crack that grows as the temperature and the displacement evolve. Neural energy solvers have been proposed for phase-field fracture and later extended to represent a sharp crack through the network input, but heat conduction on the cracked domain and crack propagation under the resulting thermal stresses have not yet been treated together in these solvers. We present an extended deep energy method for thermo-mechanical crack propagation in which the crack remains a sharp polyline. Two networks represent the temperature and the displacement and receive the crack through a scalar embedding function, discontinuous across the crack and smooth elsewhere, so that both fields can jump across it without a regularization length, and the displacement is enriched near the tip by the Williams expansion with trainable amplitudes. The two fields are obtained by mini
    
[^214]: 报告惯例掩盖了什么：基于精确计算度量的量子自然梯度等预算审计

    What a Reporting Convention Hides: A Matched-Budget Audit of Quantum Natural Gradient with an Exactly Computed Metric

    [https://arxiv.org/abs/2610.09425](https://arxiv.org/abs/2610.09425)

    该论文通过对量子自然梯度采用精确计算度量并进行等预算的公平审计，揭示了“只计成功运行时间或仅以单一目标读取结论”这两种常见报告惯例会掩盖 Adam、SPSA 与 QNG 之间的真实性能差距，甚至可能颠覆对变分量子优化器的比较结论。

    

    已发表的若干变分量子优化器比较研究，往往只统计达到目标损失的运行所耗时间，或仅在单一目标处读取结论。这两种报告惯例都可能决定“某个优化器更昂贵的步长是否物有所值”的判断。我们在从设置选择中留出的初始化上，量化每种惯例会在多大程度上改变对 Adam、同步扰动随机近似（SPSA）和量子自然梯度（QNG）三者的评判结论。我们精确计算了 QNG 预条件化所需的度量，以电路评估次数为每一步定价，并给予每种方法相同的预算。在跨电路宽度、成本族以及一组常见未达目标设置扫描所汇总的中位数下，SPSA 达到宽松目标所需的评估次数是 Adam 的两倍以上；而剔除那些未达目标的截尾运行会掩盖这一差距。在固定了为严格目标所选设置的全局成本族上，QNG 通常在 Adam 之后才达到宽松目标，但……（原文摘要在此处截断）

    arXiv:2610.09425v1 Announce Type: new  Abstract: Several published comparisons of variational quantum optimizers time only runs that reach a target loss, or read the verdict at a single target. Either convention could decide whether an optimizer's costlier steps pay off. We measure how much each convention changes verdicts among Adam, simultaneous perturbation stochastic approximation (SPSA) and quantum natural gradient (QNG), on initializations held out from the selection of settings. We compute the exact metric that preconditions QNG, price every step in circuit evaluations and give every method the same budget. In a median pooled over circuit widths, cost families and a sweep of settings with common misses, SPSA needs more than twice Adam's evaluations to reach a loose target. Dropping the censored runs that miss the target hides this gap. On the global-cost family we hold fixed the settings selected for a strict target. QNG then usually reaches the loose target after Adam but the s
    
[^215]: 使用CoMoE在消费级GPU上实现MoE推理的普及化

    Democratizing MoE inference on commodity GPUs with CoMoE

    [https://arxiv.org/abs/2610.09424](https://arxiv.org/abs/2610.09424)

    CoMoE通过新颖的以主机为中心的路由机制，将主机转变为主动路由枢纽，解决了消费级GPU互连带宽受限的通信瓶颈，使MoE模型推理能够在低成本的商品级GPU上高效部署。

    

    部署混合专家模型严重依赖专家并行，这会产生密集的GPU间通信。因此，最先进的推理系统需要数据中心GPU中配备高带宽的点对点互连（如NVLink）来处理海量的令牌路由，这使得部署成本高得令人望而却步。消费级GPU以显著更低的价格提供了相当的算力，有望让个人用户也能负担得起MoE推理，并实现保护隐私的本地部署。然而，消费级GPU带宽受限（仅有较弱的PCIe总线带宽）且互连需经主机中转（不支持点对点通信），从而引入了严重的通信瓶颈。我们提出了CoMoE，一个通信高效的MoE推理系统，通过新颖的以主机为中心的路由方式解决了这一不匹配问题。我们的关键洞察是，独特的通信拓扑结构提供了将主机提升为主动路由枢纽的机会，这可以从根本上

    arXiv:2610.09424v1 Announce Type: cross  Abstract: Deploying Mixture-of-Experts (MoE) models relies heavily on Expert Parallelism, which generates intense inter-GPU communication. Consequently, state-of-the-art inference systems require high-bandwidth, P2P interconnects (e.g., NVLink) in datacenter GPUs to handle massive token routing, making deployment prohibitively expensive. Consumer GPUs offer comparable compute power at significantly lower cost, promising to democratize MoE inference for individuals and enable privacy-preserving local deployments. However, their bandwidth-limited (only weak PCIe bus bandwidth) and host-mediated interconnects (no P2P support) introduce severe communication bottlenecks.   We present CoMoE, a communication-efficient MoE inference system that resolves this mismatch through novel host-centric routing. Our key insight is that the unique communication topology provides the opportunity to elevate the host to an active routing hub, which can fundamentally 
    
[^216]: 具有人类偏好共同演化的自消耗生成模型

    Self-Consuming Generative Models with Co-Evolving Human Preferences

    [https://arxiv.org/abs/2610.09415](https://arxiv.org/abs/2610.09415)

    该研究首次分析了自消耗生成模型中模型分布与人类偏好共同演化的动力学，证明完全依赖用户筛选数据会放大初始偏差并导致优势实例垄断，而注入足够比例的参考数据则能使系统收敛到唯一的全局平衡。

    

    生成模型越来越多地在自消耗迭代循环中训练：用户从模型生成的候选样本中筛选出偏好的样本，而这些被筛选出的样本又被用于训练模型的下一代。以往的研究大多假设用户偏好是固定不变的，但在实际中，接触模型输出会逐渐重塑用户认为理想的样本，从而形成一个模型分布与用户偏好共同演化的反馈循环。我们朝着理解这种耦合动力学的长期行为迈出了第一步。我们证明，当训练完全依赖用户筛选的合成数据时，迭代筛选会放大初始偏差，并使系统趋向多个单例平衡点之一，其中具有初始优势的实例最终将占据主导地位。相反，以足够大的比例向训练中注入参考数据则会从根本上改变动力学特性，并产生唯一的全局平衡。

    arXiv:2610.09415v1 Announce Type: new  Abstract: Generative models are increasingly trained in self-consuming iterative loops, where users curate preferred samples from model-generated candidates and the curated samples are used to train future generations of the model. Prior work has largely assumed fixed user preferences, but in practice exposure to model outputs gradually reshapes what users perceive as desirable, creating a feedback loop in which model distributions and user preferences co-evolve. We take a first step toward understanding the long-term behavior of such coupled dynamics. We show that when training relies entirely on user-curated synthetic data, iterative curation amplifies initial biases and drives the system toward one of multiple singleton equilibria in which the instance holding an initial advantage eventually dominates. In contrast, injecting reference data into training at a sufficiently large rate fundamentally changes the dynamics and yields a unique globally
    
[^217]: 共享几何作为罗塞塔石碑：无需配对数据的跨模态对齐

    Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data

    [https://arxiv.org/abs/2610.09411](https://arxiv.org/abs/2610.09411)

    提出一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化估计单一正交映射，无需任何配对数据即可实现独立训练模型的跨模态表征对齐，并证明标准几何对齐指标可准确预测对齐的可行性。

    

    多模态表征能够实现零样本分类和检索，但对齐独立训练的模型通常需要大量配对数据。然而，柏拉图表征假说表明，在不同模态上训练的模型可能会自发地收敛到共享的表征几何。那么，我们是否真的需要配对样本来进行跨模态对齐呢？值得注意的是，我们证明对于粗粒度的跨模态对齐而言，配对样本是不必要的。我们提出了一种简单的Wasserstein Procrustes方法，通过粗粒度几何初始化，仅通过估计单一的正交映射即可对齐两个互不相交的嵌入集合，而无需看到任何配对样本。在跨越多个数据集、模态和单模态模型的实验中，我们展示了对齐独立训练的表征始终可以无需配对数据完成，并且标准的几何对齐指标能够准确预测何时可以实现这种对齐。尽管如此，我们仍然可以自然地从配对样本中获益。

    arXiv:2610.09411v1 Announce Type: new  Abstract: Multimodal representations enable zero-shot classification and retrieval, but aligning independently trained models usually requires large amounts of paired data. Yet, the Platonic Representation Hypothesis suggests that models trained on different modalities may converge spontaneously toward a shared representation geometry. But then, do we even need paired examples for cross-modal alignment? Remarkably, we show that paired examples are unnecessary for coarse cross-modal alignment. Our simple Wasserstein Procrustes method with a coarse geometric initialization aligns two disjoint embedding sets by estimating a single orthogonal map without seeing any pairs. Across datasets, modalities, and unimodal models, we show that we can consistently align independently trained representations without pairs, and standard geometric alignment metrics accurately predict when this is possible. Nevertheless, we can naturally benefit from paired examples
    
[^218]: 网络化自消耗生成生态系统的稳定性与多样性

    Stability and Diversity of Networked Self-Consuming Generative Ecosystems

    [https://arxiv.org/abs/2610.09409](https://arxiv.org/abs/2610.09409)

    本文首次将多个生成模型间的合成数据流动建模为有向加权图，提出了一个理论框架来系统分析网络化自消耗生成生态系统的长期稳定性与多样性。

    

    生成式人工智能的广泛部署使得合成内容与真实数据日益难以区分。因此，合成数据不可避免地被纳入未来模型世代的训练流程中，形成了自消耗训练循环。先前的工作已经研究了这种递归自消耗训练的影响，但分析大多局限于孤立模型（即模型仅消耗其自身生成的合成数据），或局限于两个模型之间的简化交互。本文朝着理解网络化自消耗生成模型迈出了第一步——在这种模型中，多个模型通过复杂的交互路径消耗彼此生成的合成数据。我们引入了一个理论框架，将各个模型表示为有向加权图中的节点，边权重控制着模型间合成数据的流动。利用该框架，我们分析了这类系统的长期行为与特性。

    arXiv:2610.09409v1 Announce Type: new  Abstract: The widespread deployment of generative AI has made it increasingly difficult to distinguish synthetic content from real data. Consequently, synthetic data is inevitably incorporated into the training pipelines of future model generations, forming a self-consuming training loop. Prior work has studied the effects of such recursive self-consuming training, but analyses have largely been limited to isolated models, where a model consumes only its own synthetic data, or to simplified interactions between two models. This paper takes a first step toward understanding networked self-consuming generative models, in which multiple models consume synthetic data generated by one another through complex interaction pathways. We introduce a theoretical framework representing models as nodes in a directed, weighted graph, with edge weights governing the flow of synthetic data among models. Using this framework, we analyze the long-term behavior of n
    
[^219]: 加噪、去噪、校正：三步实现基于扩散先验的MCMC后验采样

    Noise, Denoise, Correct: MCMC Posterior Sampling with Diffusion Priors in Three Steps

    [https://arxiv.org/abs/2610.09407](https://arxiv.org/abs/2610.09407)

    该论文提出扩散圆舞曲方法，以SDEdit式加噪-去噪作为提议分布并结合Metropolis-Hastings校正，实现无需先验评估的精确MCMC后验采样，并可通过无梯度集合卡尔曼更新注入观测信息，在非线性、不可微的Navier-Stokes初始条件恢复任务上优于现有基线。

    

    预训练扩散模型是解决逆问题的强大先验，但在非线性、不可微的正演模型下进行后验采样仍然十分困难。我们提出了扩散圆舞曲，这是一种MCMC方法，它使用SDEdit风格的加噪-去噪过程作为提议分布，并通过Metropolis-Hastings校正实现精确的后验采样，且无需对先验进行评估。我们进一步提出利用无梯度的集合卡尔曼更新将观测信息注入提议分布，同时保持采样的精确性。在一个不可微的Navier-Stokes初始条件恢复任务上，扩散圆舞曲在不同噪声和非线性条件下均优于现有基线方法。

    arXiv:2610.09407v1 Announce Type: new  Abstract: Pretrained diffusion models are powerful priors for inverse problems, but posterior sampling under nonlinear, non-differentiable forward models remain hard. We introduce diffusion waltz, an MCMC method using SDEdit-style noising-denoising as a proposal, corrected via Metropolis-Hastings for exact posterior sampling without prior evaluation. We further propose injecting observations into the proposal while preserving exactness, using a gradient-free ensemble Kalman update. On a non-differentiable Navier-Stokes initial condition recovery task, diffusion waltz outperforms existing baselines across different noise and nonlinearity regimes.
    
[^220]: 人格层级模型：理解大语言模型微调中的情境泛化

    The Persona Hierarchy Model: Understanding Contextual Generalization in Fine-Tuning LLMs

    [https://arxiv.org/abs/2610.09384](https://arxiv.org/abs/2610.09384)

    该论文提出人格层级模型，指出大语言模型微调后行为的泛化范围取决于修改的是共享默认人格还是局部人格，且泛化狭窄程度与训练情境人格和默认人格的相似度呈正相关。

    

    语言模型通常在固定情境下进行微调，例如通用系统提示词、人格设定或领域特定指令，然而所学到的行为有时仅局限于该情境，有时则会广泛泛化到未见过的情境。我们提出人格层级模型（Persona Hierarchy Model）来解释这一现象：一个共享的默认人格会影响跨情境的行为。在该模型下，修改共享人格的微调能促进更广泛的迁移，而对局部人格的更改则更局限于特定情境。在涵盖四种行为和15个训练情境的120个微调模型中，泛化狭窄程度与训练情境人格和默认人格之间的相似度呈正相关（Qwen3-4B的皮尔逊相关系数r = 0.72）。在默认情境下先进行微调，可以拓宽后续在其他情境下训练时的泛化能力。将情境响应与默认人格响应对齐……

    arXiv:2610.09384v1 Announce Type: new  Abstract: Language models are routinely fine-tuned under a fixed context, such as a generic system prompt, persona or domain-specific instruction, yet the learned behavior sometimes stays confined to that context and sometimes broadly generalizes to unseen contexts. We propose the Persona Hierarchy Model to explain this: a shared default persona influences behavior across contexts. Under this model, fine-tuning that modifies the shared persona promotes broader transfer, whereas changes to local personas remain more context-specific. Across 120 fine-tuned models spanning four behaviors and 15 training contexts, generalization narrowness positively correlates with the similarity between the training context's persona and the default persona (Pearson's r = 0.72 for Qwen3-4B). Prior fine-tuning under the default context can broaden generalization in subsequent training under other contexts. Aligning contextual responses with default-persona responses 
    
[^221]: 从看似合理的层级结构到实用的分类体系：评估智能体框架在客户反馈上的表现

    From Plausible Hierarchies to Useful Taxonomies: Evaluating Agentic Harnesses on Customer Feedback

    [https://arxiv.org/abs/2610.09377](https://arxiv.org/abs/2610.09377)

    该研究提出评估AI生成的分类体系何时才真正适用于生产环境，发现尽管所有生成的层级结构都能通过通用的命名与结构检查，但仍需更深层次的产品覆盖评估才能判断其实际实用性。

    

    分类体系是AI系统用于组织证据、聚合模式并回答关于大型文档集合问题的符号表示。在客户反馈场景中，类别树决定了每条记录如何被计数和路由、哪些问题会被发现、以及由哪个团队负责处理。如今，智能体框架可以轻松生成看似合理的层级结构，而这类树结构目前仅通过通用的、单项范围的检查来验证：每个名称符合其描述、位于正确的父节点之下，并与同级节点保持区分。我们提出了一个更具操作性的问题：生成的层级结构何时才真正能作为生产环境的分类体系使用？我们在两个专有反馈语料库（1,940条和5,000条记录）上构建了六个分类体系：对每个语料库，包括一个生产参考分类体系以及在相同输入下同一框架的两次重复运行。所有六个分类体系都通过了所有通用的命名和结构检查，以及更深入的产品覆盖检查

    arXiv:2610.09377v1 Announce Type: cross  Abstract: Taxonomies are the symbolic representations through which AI systems organize evidence, aggregate patterns, and answer questions over large document collections. Over customer feedback, the category tree decides how every record is counted and routed, which problems get seen, and which team owns them. Agentic harnesses now make it easy to generate a plausible-looking hierarchy, and such trees are checked today with generic, individually scoped checks: each name fits its description, sits under the right parent, and stays distinct from its siblings. We ask a more operational question: when is a generated hierarchy actually useful as a production taxonomy? We build six taxonomies over two proprietary feedback corpora (1,940 and 5,000 records): for each corpus, a production reference and two repeated runs of the same harness under identical inputs. All six pass every generic naming and structure check, and a deeper product-coverage check 
    
[^222]: 从检索到客户情境：评估用于客户之声分析的前沿模型系统

    From Retrieval to Customer Context: Evaluating Frontier-Model Systems for Voice-of-Customer Analysis

    [https://arxiv.org/abs/2610.09375](https://arxiv.org/abs/2610.09375)

    该论文提出“客户情境图”这一统一建模框架，将客户反馈与业务运营及分析对象通过类型化关系连接，使智能体不仅能分析客户说了什么，还能理解为什么、谁受影响、如何解决等深层情境，并在 9,432 条公开 Cursor 反馈上对比评估了三种前沿模型系统。

    

    各组织越来越多地使用前沿语言模型来分析客户反馈，但答案的质量还取决于这些反馈是如何组织和提供的。我们将“客户情境图”定义为客户和业务情境的统一模型。类型化的关系将客户对象（反馈、对话、用户和账户）、运营对象（工单、支持人员、销售机会和竞争对手）以及分析或行动对象（分类概念、证据、洞察、工作项和结果）连接起来。这使智能体不仅能调查客户说了什么，还能调查为什么、谁受到影响、后续采取了什么行动、由谁负责以及问题是否得到解决。在本实验中，该图由公开的 Cursor 反馈数据填充；同样的架构可以支持任何类型的反馈来源。我们在相同的 9,432 条公开 Cursor 反馈上比较了 Agentic RAG、深度研究智能体和基于客户情境图的智能体。

    arXiv:2610.09375v1 Announce Type: new  Abstract: Organizations increasingly use frontier language models to analyze customer feedback, but answer quality also depends on how that feedback is organized and made available. We define a \emph{customer context graph} as a unified model of customer and business context. Typed relationships connect customer objects (feedback, conversations, users, and accounts), operational objects (tickets, support agents, opportunities, and competitors), and analytical or action objects (taxonomy concepts, evidence, insights, work items, and outcomes). This lets an agent investigate not only what customers say, but why, who is affected, what action followed, who owns it, and whether it was resolved. For this experiment, the graph is populated from public Cursor feedback; the same architecture can support any type of feedback source. We compare Agentic RAG, a Deep Research Agent, and a Customer Context Graph-backed Agent on the same 9,432 public Cursor feedb
    
[^223]: 互不相溶扩散策略：通过无标签噪声分配保留多模态机器人动作

    Immiscible Diffusion Policy: Preserving Multimodal Robot Actions through Label-Free Noise Assignment

    [https://arxiv.org/abs/2610.09369](https://arxiv.org/abs/2610.09369)

    提出互不相溶扩散策略，一种无需标签的训练时噪声分配方法，通过保持噪声到动作路径的相对分离，防止扩散策略在低维机器人动作空间中坍缩为单一模态，从而保留多模态动作分布。

    

    当扩散策略首次被提出时，人们期望它们能够恢复多模态动作分布。然而，我们发现这一期望并不总是成立，因为即使保证了数据集模态的平衡以及批次内完全的对称性，扩散策略仍常常会坍缩到单一模态。我们的分析表明，独立的动作-噪声配对是导致这一失败的原因之一，它增加了扩散路径之间的混合与交叉，从而可能产生平均化的去噪响应，并抑制模态特定的行为。这一问题在机器人规划中尤为严重，因为动作空间是密集且低维的，会显著加剧这种混合与交叉。为了缓解该问题，我们提出了互不相溶扩散策略，这是一种无需标签的训练时附加组件，通过动作-噪声分配来保持相对互不干扰的噪声到动作路径，且无需修改策略架构。

    arXiv:2610.09369v1 Announce Type: cross  Abstract: When diffusion policies were first introduced, they were expected to recover multi-modal action distributions. However, we find this expectation does not always hold, as diffusion policies often collapse to a single modality even when we guarantee the balance of dataset modalities and exact within-batch symmetry. Our analysis indicates that independent action-noise pairing contributes to this failure by increasing mixing and crossing among diffusion paths, which can produce averaged denoising responses and suppress modality-specific behavior. This issue is especially severe in robot planning, where action spaces are dense and low-dimensional, significantly increasing such mixing and crossing. To alleviate this problem, we propose Immiscible Diffusion Policy, a label-free training-time add-on to diffusion policy that uses action-noise assignment to preserve relatively distinct noise-to-action routes without modifying the policy architec
    
[^224]: 基于卫星验证的凝结尾迹规避闭环

    Closing the Loop on Contrail Avoidance with Satellite Verification

    [https://arxiv.org/abs/2610.09363](https://arxiv.org/abs/2610.09363)

    该研究构建了一个仅840万参数、单GPU训练的小型扩散模型，可在卫星图像中检测仅一到两个像素宽的飞机凝结尾迹（PR-AUC达0.476），为“改航规避尾迹—卫星验证”闭环提供了可行的技术路径，并总结出分辨率与简单数据增广比新架构更重要的普适经验。

    

    凝结尾迹是飞机飞行后留下的纤细冰云，在航空业的气候变暖影响中占很大比重，而只需改变少数会产生尾迹的航班航线，便可避免其中大部分的变暖效应。然而，一条被规避的尾迹只有在卫星确认它从未形成时才算数，而这项验证非常困难：尾迹只有一到两个像素宽，仅覆盖0.18%的像素，且与天然卷云极为相似。我们构建了一个小型扩散模型（840万参数，单块GPU训练）来检测尾迹，并通过对照实验研究哪些组件真正重要。该模型达到0.476的PR-AUC，相比之下DeepLabV3+基线为0.414，改造后的MedSegDiff为0.119。将CNN的输入分辨率提高一倍即可使其达到同等水平（0.499，p=0.07）。三条经验教训同样适用于尾迹之外的领域。第一，在设计新架构之前先检查输入分辨率。第二，简单的翻转和旋转能使准确率提高一倍以上，其影响超过任何……（原文摘要在此截断）

    arXiv:2610.09363v1 Announce Type: cross  Abstract: Contrails are the thin ice clouds that aircraft leave behind. They cause a large share of aviation's warming, and rerouting the few flights that produce them could avoid much of it. However, an avoided contrail only counts if a satellite can confirm that it never formed, and this check is hard: contrails are one to two pixels wide, cover only 0.18% of pixels, and look very similar to natural cirrus. We build a small diffusion model (8.4M parameters, trained on one GPU) that detects them, and we run a controlled study to find out which components matter. The model reaches 0.476 PR-AUC, compared with 0.414 for a DeepLabV3+ baseline and 0.119 for an adapted MedSegDiff. Doubling the input resolution of the CNN brings it to parity (0.499, p=0.07). Three lessons apply beyond contrails. First, check the input resolution before designing a new architecture. Second, simple flips and rotations more than double accuracy and matter more than any a
    
[^225]: 两层线性网络训练的全局指数收敛性

    Global Exponential Convergence of Two-Layer Linear Network Training

    [https://arxiv.org/abs/2610.09356](https://arxiv.org/abs/2610.09356)

    该论文证明了采用光滑PL预测损失训练的宽两层线性网络的全局指数收敛性，其动力学可精确刻画为神经元协方差的有限维Bures流，并给出显式收敛速率（初始协方差为σ²Id时至少为4σ²κ），且该速率在有限宽度采样下保持稳定。

    

    我们在丰富缩放下证明了使用光滑Polyak-Lojasiewicz预测损失训练的宽两层线性网络的全局指数（线性）收敛性，并给出了显式速率。因子中的梯度流恰好以神经元法则协方差的有限维Bures流的形式闭合，其中预测器动力学由隐藏协方差块进行预条件处理。当初始协方差满足谱支撑间隙条件时，平均场守恒律为隐藏预条件块提供了一致的谱下界。该条件涵盖了正定性的情形，同时仍然允许奇异初始化。对于初始协方差 Σ₀ = σ²Id，损失以至少 4σ²κ 的线性速率收敛到全局最小值，其中 κ 是PL常数。我们建立了该速率在有限宽度采样下的稳定性，以及因子梯度的全局收敛性。

    arXiv:2610.09356v1 Announce Type: new  Abstract: We prove global exponential (linear) convergence with an explicit rate in the rich scaling for wide two-layer linear networks trained with smooth Polyak-Lojasiewicz predictor losses. Gradient flow in the factors closes exactly in terms of a finite-dimensional Bures flow of the neuron law covariance, in which the predictor dynamics are preconditioned by hidden covariance blocks. Mean-field conservation laws provide uniform spectral lower bounds on the hidden preconditioning blocks when the initial covariance satisfies a spectral support gap condition. This condition encompasses positive definiteness while still allowing for singular initializations. For an initial covariance $\Sigma_0 = \sigma^2 \mathrm{Id}$, the loss converges to the global minimum with linear rate at least $4\sigma^2\kappa$, where $\kappa$ is the PL constant. We establish stability of this rate under finite-width sampling, as well as global convergence of factor gradien
    
[^226]: 多模态大语言模型能够学会读取脑信号：用于统一多任务EEG解码的视觉-语言模型

    Multimodal LLMs Can Learn to Read Brain Signals: A Vision--Language Model for Unified Multi-Task EEG Decoding

    [https://arxiv.org/abs/2610.09355](https://arxiv.org/abs/2610.09355)

    BraVista将多通道EEG信号编码为结构化图像，通过对通用视觉-语言模型进行持续后训练，在无需大规模EEG专用预训练的情况下实现了跨数据集的统一多任务脑电解码。

    

    学习能够在不同认知任务、受试者和记录条件之间泛化的EEG表征，仍然是脑电图（EEG）解码领域的一项关键挑战。基础模型的最新进展提升了EEG解码性能，但一个根本性的开放问题依然存在：如何有效地将神经信号与这些模型对接，以实现跨数据集的多任务学习。为了探究这一问题，我们提出了BraVista，这是一个视觉-语言框架，它将多通道EEG信号编码为结构化图像，并通过指令条件化的视觉-语言模型（VLM）实现多任务学习。我们的方法依赖于对通用领域VLM的持续后训练，利用其视觉和语言先验来适应神经信号，而无需单独的大规模EEG专用预训练阶段。我们在四个数据集上对BraVista进行了评估，涵盖睡眠分期、情绪识别、认知负荷分类等任务。

    arXiv:2610.09355v1 Announce Type: new  Abstract: Learning EEG representations that generalize across cognitive tasks, subjects, and recording conditions remains a key challenge in electroencephalography (EEG) decoding. Recent advances in foundation models have improved EEG decoding performance, yet a fundamental open question remains: how to effectively interface neural signals with these models to enable multi-task learning across datasets. To investigate this question, we introduce BraVista, a visual-language framework that encodes multichannel EEG signals as structured images and enables multi-task learning through instruction-conditioned vision-language models (VLMs). Our approach relies on continued post-training of a general-domain VLM, leveraging its visual and linguistic priors to adapt to neural signals without a separate large-scale EEG-specific pretraining stage. We evaluate BraVista on four datasets spanning sleep staging, emotion recognition, cognitive workload classificat
    
[^227]: 基于最优性与反事实正则化的无不安全数据未知约束学习

    Learning Unknown Constraints without Unsafe Data via Optimality and Counterfactual Regularization

    [https://arxiv.org/abs/2610.09350](https://arxiv.org/abs/2610.09350)

    提出CF-KKT约束学习框架，通过反事实KKT正则化利用学习到的动力学和局部最优专家演示来恢复未知约束，无需已知动力学、结构化约束表示或存在安全风险的数据探索。

    

    从演示中学习为从局部最优、满足约束的专家行为中推断未知约束提供了一个框架。现有方法主要分为两种范式：约束逆最优控制和逆约束强化学习。CIOC 利用诸如 Karush--Kuhn--Tucker（KKT）条件等最优性条件，但通常假设动力学已知且约束表示具有结构化形式。与此同时，ICRL 能够处理复杂的未知约束和未知的转移动力学，但通常需要大量的在线探索，在此过程中可能发生不安全的约束违反。在本工作中，我们提出了反事实 KKT（CF-KKT），这是一种约束学习框架，它利用学习到的动力学和局部最优演示来恢复未知约束，而无需已知动力学或额外的风险探索，从而结合了数据效率

    arXiv:2610.09350v1 Announce Type: cross  Abstract: Learning from demonstrations (LfD) provides a framework for inferring unknown constraints from locally optimal, constraint-satisfying expert behavior. Existing approaches largely fall into two paradigms, constrained inverse optimal control (CIOC) and inverse constrained reinforcement learning (ICRL). CIOC exploits optimality conditions such as the Karush--Kuhn--Tucker (KKT) conditions but typically assumes known dynamics and structured constraint representations. Meanwhile, ICRL accommodates complex unknown constraints and unknown transition dynamics but often requires extensive online exploration, during which unsafe constraint violations may occur. In this work, we introduce Counterfactual KKT (CF-KKT), a constraint learning framework that leverages learned dynamics and locally optimal demonstrations to recover unknown constraints without requiring known dynamics or additional risky exploration, thereby combining the data efficiency 
    
[^228]: OnlineQAT：面向超低比特大语言模型的同策略蒸馏

    OnlineQAT: On-Policy Distillation for Ultra-Low-Bit Large Language Models

    [https://arxiv.org/abs/2610.09346](https://arxiv.org/abs/2610.09346)

    OnlineQAT提出了一种两阶段框架，先通过分块QAT获得低比特初始化，再利用冻结的全精度教师模型在学生自身生成的回复上进行同策略蒸馏，从而在2-3比特的超低比特量化下显著恢复大语言模型的精度并超越现有离线QAT方法。

    

    量化感知训练（QAT）能够在大语言模型被压缩至四比特以下时恢复大部分精度损失。然而，现有的恢复阶段通常基于固定的补全或教师生成的答案进行优化，而实际部署的量化模型却是基于其自身生成的前缀进行条件生成的。因此，量化误差可能使模型进入离线恢复数据中不存在的状态。我们提出OnlineQAT，这是一个两阶段框架：首先通过分块量化感知训练获得可用的低比特初始化，然后在学生模型自身生成的回复上进行同策略蒸馏（OPD）。在每个所访问的前缀处，由一个冻结的全精度教师模型提供采样的反向KL散度训练信号。在Qwen3-1.7B上，OnlineQAT在所比较的量化方法中取得了最佳平均成绩：W3A16下达到57.28，W2A16下达到32.52，分别比ReasoningQAT提升了2.90分和0.44分。结果表明……

    arXiv:2610.09346v1 Announce Type: new  Abstract: Quantization-aware training (QAT) can recover much of the accuracy lost when large language models are compressed below four bits. Existing re- covery stages, however, are commonly optimized on fixed completions or teacher-generated answers, whereas the deployed quantized model condi- tions on prefixes generated by itself. Quantization errors can therefore move the model into states that are absent from offline recovery data. We introduce OnlineQAT, a two-stage framework that first obtains a usable low-bit initialization through block-wise QAT and then performs on-policy distillation (OPD) on student-generated responses. At each visited pre- fix, a frozen full-precision teacher provides a sampled reverse-KL training signal. On Qwen3-1.7B, OnlineQAT obtains the best average among the compared quantized methods: 57.28 at W3A16 and 32.52 at W2A16, im- proving over ReasoningQAT by 2.90 and 0.44 points, respectively. The results suggest that 
    
[^229]: 面向无数据混合专家模型压缩的共享低秩基分解方法

    Shared Low-rank Basis Factorization for Data-free Mixture-of-Experts Compression

    [https://arxiv.org/abs/2610.09342](https://arxiv.org/abs/2610.09342)

    本文提出共享低秩基分解（SLBF），一种无数据的MoE压缩权重重构方法，通过让专家共享秩-k低秩基实现更低的重构误差和更快的收敛，并从理论上证明了剪枝和合并类方法存在不可消除的结构性误差。

    

    混合专家大型语言模型通过稀疏路由将模型容量与计算量解耦，但其庞大的参数量给存储和服务部署带来了挑战。我们分析了三类MoE压缩方法：专家剪枝、专家合并和权重重构，并推导出结构性误差界，表明剪枝和合并可能产生与路由和专家异质性相关的不可消除的误差。相比之下，权重重构通过保留专家结构和路由来避免这些结构性代价。基于这一分析，我们提出了共享低秩基分解（SLBF），这是一种无数据的权重重构方法，使用专家间共享的秩-k基，实现更丰富的跨专家共享、更快的收敛速度和更低的重构误差。事后规范固定处理可在不损失表示能力的情况下移除冗余参数。在五个参数量从16B到122B的MoE架构上，SLBF始终表现优异（摘要在此处截断）。

    arXiv:2610.09342v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) large language models decouple capacity from compute through sparse routing, but their large parameter count creates storage and serving challenges. We analyze three MoE compression families: expert pruning, expert merging, and weight reconstruction, and derive structural error bounds showing that pruning and merging can incur non-vanishing errors tied to routing and expert heterogeneity. In contrast, weight reconstruction avoids these structural costs by preserving expert structure and routing. Motivated by the analysis, we propose Shared Low-rank Basis Factorization (SLBF), a data-free weight reconstruction method that uses rank-$k$ bases shared among experts, enabling richer cross-expert sharing, faster convergence, and lower reconstruction error. A post-hoc gauge fixing removes redundant parameters at no representational cost. Across five MoE architectures spanning 16B to 122B parameters, SLBF consistently ou
    
[^230]: 异构输入融合下的良性过拟合

    Benign Overfitting under Heterogeneous Input Fusion

    [https://arxiv.org/abs/2610.09340](https://arxiv.org/abs/2610.09340)

    该论文首次研究异构输入融合下的良性过拟合，发现回归中存在一个与截断阈值无关的全谱协方差证书，可保证良性性质在任意一致的联合协方差融合下得以保持，但这种保护是精确的——在证书范围之外，两个良性的边缘输入块融合后可能变得有害。

    

    良性过拟合在从单一高维输入学习时已被广泛研究，但其在异构输入融合下的行为在很大程度上仍未被探索。我们在异构高斯设计下研究最小范数线性插值，在保持总体任务不变的前提下，比较两个统计相关的输入块与其融合后的表现。对于回归问题，我们识别出一个全谱协方差证书，其渐近状态与截断阈值无关，并证明该证书能被每一个与两个边缘分布相一致的正半定联合协方差所保持。这种保护是精确的，但它并不能扩展到所有良性回归问题：在证书范围之外，两个良性的边缘分布融合后可能变得有害。对于单稀疏高斯分类，正则区域中的良性特征由存活的预测信号与干扰之间的平衡所刻画。

    arXiv:2610.09340v1 Announce Type: new  Abstract: Benign overfitting is extensively studied when learning from a single high-dimensional input, but its behavior under heterogeneous input fusion remains largely unexplored. We study this question for minimum-norm linear interpolation under a heterogeneous Gaussian design, comparing two statistically dependent input blocks with their fusion while holding the underlying population task fixed. For regression, we identify a full-spectrum covariance certificate whose asymptotic status is independent of the cutoff threshold and prove that it is preserved by every positive-semidefinite joint covariance consistent with the two marginals. This protection is sharp, yet it does not extend to all benign regression problems: outside the certified regime, two benign marginals can have a harmful fusion. For one-sparse Gaussian classification, benignity in the regular regime is characterized by the balance between surviving predictive signal and nuisance
    
[^231]: 询问专家：面向自主网络防御的LLM引导强化学习

    Ask the Expert: LLM-Guided Reinforcement Learning for Autonomous Cyber Defense

    [https://arxiv.org/abs/2610.09337](https://arxiv.org/abs/2610.09337)

    该论文提出“询问专家”训练时引导框架，在训练过程中利用LLM将困难防御状态的应对建议转化为分层奖励塑形以提升PPO的样本效率，训练结束后LLM被丢弃，部署时仅需纯强化学习策略。

    

    基于策略的强化学习方法在自主网络防御方面取得了令人瞩目的成果；然而，在防御者必须在延迟、部分可观测的条件下从大动作空间中做出响应的场景中，这些方法的样本效率较低。尽管大型语言模型能够对安全状态空间进行语义推理，但高延迟和信任假设使其难以作为有吸引力的在线部署方案。我们提出了“询问专家”，这是一个训练时引导框架：首先总结困难的网络防御状态，然后通过受约束的动作接口间歇性地向LLM查询主机级防御建议，最后将这些建议转化为分层奖励塑形以供PPO使用。由于LLM在训练完成后即被丢弃，部署阶段只需纯粹的RL策略。在TTCP CAGE CC1和CC2环境以及两种攻击者类型的实验中，这种非对称设计相比PPO显著提升了样本效率。

    arXiv:2610.09337v1 Announce Type: cross  Abstract: Policy-based reinforcement learning (RL) approaches have produced promising results for autonomous cyber defense; however, they are sample-inefficient in settings where defenders must respond under delayed, partial observations with actions from large action spaces. While large language models (LLMs) may reason semantically about security state space, high latency and trust assumptions prevent attractive in-line deployment models. We introduce Ask the Expert, a training-time guidance framework which first summarizes hard cyber-defense states, then intermittently queries an LLM for host-level defensive recommendations via a constrained action interface, and finally transforms those recommendations into tiered reward shaping for use with PPO. Because the LLM is discarded after training, deployment is a pure RL policy. Across TTCP CAGE CC1 and CC2 and both attacker types, this asymmetric design improves sample efficiency over PPO and outp
    
[^232]: SearchWorld：基于世界模型的价值引导空间想象的无人机目标搜索

    SearchWorld: Spatial Value-Grounded Imagination for UAV Object Search via World Models

    [https://arxiv.org/abs/2610.09335](https://arxiv.org/abs/2610.09335)

    SearchWorld提出了一种将显式空间记忆与价值引导想象相结合的循环状态空间世界模型，通过前瞻性想象推演解决部分可观测性难题，实现无人机在城市环境中的高效目标搜索。

    

    自主无人机（UAV）目标搜索涉及部分可观测环境下感知、决策与行动的闭环过程。城市环境带来了诸多挑战：广阔的搜索区域和狭窄的第一人称视野限制了覆盖范围，密集的三维几何结构制约了安全移动，而开放世界的指令要求在众多干扰物中识别特定目标。许多现有方法通过显式地图或记忆表征来缓解部分可观测性问题，但它们在很大程度上仍是反应式的，即仅基于过去的观测进行推理，而未能显式地预测未来状态。世界模型能够通过想象推演实现前瞻性推理。然而，基于图像生成的世界模型可能带来较高的推理延迟，而对于潜在（latent）世界模型而言，空间接地的规划仍然具有挑战性。我们提出了SearchWorld，一种将显式空间记忆与价值引导想象相连接的循环状态空间世界模型。该模

    arXiv:2610.09335v1 Announce Type: cross  Abstract: Autonomous unmanned aerial vehicle (UAV) object search involves a closed loop of perception, decision-making, and action under partial observability. Urban environments pose several challenges: large search areas and narrow egocentric views limit coverage, dense 3D geometry constrains safe motion, and open-world instructions require identifying a specific target among distractors. Many existing methods mitigate partial observability through explicit maps or memory representations, yet remain largely reactive, reasoning over past observations without explicitly predicting future states. World models enable prospective reasoning through imagined rollouts. However, image-generating world models can incur high inference latency, while spatially grounded planning remains challenging for latent world models. We propose SearchWorld, a recurrent state-space world model that connects explicit spatial memory with value-guided imagination. The mo
    
[^233]: VM-ARRAYDPS：基于虚拟麦克风增强扩散后验采样的无监督盲语音分离

    VM-ARRAYDPS: Virtual Microphone Augmented Diffusion Posterior Sampling for Unsupervised Blind Speech Separation

    [https://arxiv.org/abs/2610.09334](https://arxiv.org/abs/2610.09334)

    本文提出VM-ArrayDPS方法，通过引入虚拟麦克风来增强扩散后验采样中的多通道一致性目标，从而突破实际麦克风阵列数量有限对无监督盲语音分离性能的制约。

    

    盲源分离（BSS）是信号处理中的一个基本问题，旨在从混合信号中分离出多个源信号，而无需对源信号或混合过程具备先验知识。传统方法（如独立向量分析IVA）利用源信号的统计独立性。近年来，基于扩散模型的方法凭借强大的生成先验，成为一种有前景的替代方案。其中，ArrayDPS将盲源分离问题表述为后验采样问题，并利用预训练的语音扩散模型引导干净源信号的恢复。其分离能力的一个关键因素是多通道一致性（MC）目标，该目标强制估计出的源信号通过估计的声学传递函数重建观测到的麦克风混合信号。然而，阵列中麦克风的数量通常是有限的，这制约了ArrayDPS的性能。

    arXiv:2610.09334v1 Announce Type: cross  Abstract: Blind Source Separation(BSS) is a fundamental problem in signal processing, aiming to separate multiple source signals from their mixtures without prior knowledge of the sources or the mixing process. Traditional approaches, such as Independent Vector Analysis (IVA) exploits statistical independence of sources. Recently, diffusion-based approaches have emerged as a promising alternative by leveraging powerful generative priors. Among them, ArrayDPS formulates BSS problem as a posterior sampling problem, and utilizes a pretrained speech diffusion model to guide the recovery of clean source signals. A key factor behind its separation capability is the multi-channel consistency (MC) objective, which enforces the estimated source signals to reconstruct the observed microphone mixtures through the estimated acoustic transfer functions. However, the number of microphones in the array is often limited, which constrains the performance of Arra
    
[^234]: 视觉Jev奖励：面向多主体图像生成的参考绑定验证

    Visual Jev Rewards: Reference-Bound Verification for Multi-Subject Image Generation

    [https://arxiv.org/abs/2610.09328](https://arxiv.org/abs/2610.09328)

    该论文提出了参考绑定的视觉Jev奖励方法，通过二值监督训练的Qwen3.5-4B验证器，仅在请求条件与参考主体身份同时成立时给予正奖励，以此生成GRPO训练信号，显著提升了多主体图像生成中主体交互验证的准确性。

    

    多主体图像生成需要能够验证所请求的属性、动作和关系是否对指定参考主体成立的奖励。仅有主体的存在并不能确定正确的主体参与了所请求的交互。我们提出了参考绑定的视觉Jev奖励，将这些视觉决策转化为生成器的训练信号。每个与主体相关的问题只有在所请求的条件与相关参考身份同时成立时才会获得正向标签。我们离线构建固定问题，使用二值监督训练Qwen3.5-4B验证器，并直接从其语言模型头读取"Yes"概率。其均值作为GRPO奖励，同时保留各个独立判断以供检查。使用200个MICo-150K训练任务和30次更新，该框架在人工筛选的897任务MICo-Bench子集上将GPT-5.4综合评分从41.78提升至52.50。

    arXiv:2610.09328v1 Announce Type: cross  Abstract: Multi-subject image generation requires rewards that verify whether requested attributes, actions, and relations hold for the specified reference subjects. Subject presence alone does not establish that the correct subjects participate in a requested interaction. We present reference-bound Visual Jev rewards that turn these visual decisions into generator training signals. Each subject-related question receives a positive label only when the requested condition and the relevant reference identities hold jointly. We construct fixed questions offline, train a Qwen3.5-4B verifier with binary supervision, and directly read Yes probabilities from its language-model head. Their mean supplies a GRPO reward while retaining individual judgments for inspection. Using 200 MICo-150K training tasks and 30 updates, the framework raises a GPT-5.4 composite score from 41.78 to 52.50 on a manually selected 897-task MICo-Bench subset; direct 27B rewards
    
[^235]: 去噪块而非词元：基于分支词元实现的高效压缩连续扩散

    Denoising Blocks, Not Tokens: Efficient Compressed Continuous Diffusion with Branching Token Realization

    [https://arxiv.org/abs/2610.09311](https://arxiv.org/abs/2610.09311)

    提出分支潜在扩散，通过将1024个词元压缩为64个块潜在表示（16倍压缩），并由并行运行的局部自回归分支进行词元解码，大幅提升扩散语言模型的生成效率。

    

    扩散语言模型通过迭代并行细化来生成文本，有望比自回归（AR）解码获得更高的吞吐量。然而，大多数扩散语言模型仍然为每个词元维护一个生成状态，因此每个去噪步骤处理的状态序列长度与输出序列相同，这限制了并行生成带来的吞吐量提升。连续扩散语言模型提供了一个额外的自由度：单个连续状态可以表示多个词元，从而使扩散过程能够在更短的潜在序列上运行。我们提出了分支潜在扩散，它利用了这种灵活性，将1024个词元的序列压缩为仅64个块潜在表示，实现了16倍的压缩。BLD将潜在压缩与分支词元实现相结合，其中每个潜在表示由一个局部自回归分支解码，且所有分支并行运行。由于强压缩使联合潜在生成变得困难，BLD

    arXiv:2610.09311v1 Announce Type: new  Abstract: Diffusion language models (DLMs) generate text through iterative parallel refinement, offering the potential for higher throughput than autoregressive (AR) decoding. However, most DLMs still maintain one generative state per token, so every denoising step processes a state sequence as long as the output sequence, limiting the throughput gains from parallel generation. Continuous DLMs provide an additional degree of freedom: a single continuous state can represent multiple tokens, allowing diffusion to operate on a much shorter latent sequence. We introduce \emph{Branching Latent Diffusion (BLD)}, which exploits this flexibility by compressing a 1024-token sequence into only 64 block latents, a $16\times$ reduction. BLD combines latent compression with \emph{branching token realization}, where each latent is decoded by a local AR branch and all branches run in parallel. Because strong compression makes joint latent generation difficult, B
    
[^236]: MovieSTAGE：用于电影fMRI ADHD分类的场景、转换与全局编码

    MovieSTAGE: Scene, Transition, and Global Encoding for Movie-fMRI ADHD Classification

    [https://arxiv.org/abs/2610.09306](https://arxiv.org/abs/2610.09306)

    提出MovieSTAGE多尺度框架，通过融合场景内超图功能连接、相邻场景间转换差异和整部电影功能连接三种表征，显著提升了基于自然电影fMRI的ADHD分类性能。

    

    自然主义电影-fMRI提供了一种共享的、时间结构化的大脑动态探测手段，然而现有预测模型通常依赖于整个扫描片段的功能连接（FC）或与叙事事件不对齐的时间通用表征。我们提出了MovieSTAGE（场景、转换与全局编码），这是一个多尺度框架，结合了场景内超图结构化的FC特征组织、相邻场景间的无符号FC特征差异，以及整部电影水平的FC。我们在CMI-HBN《神偷奶爸》观影队列的260名参与者上，采用10次重复的分层五折交叉验证、完整的折外（OOF）预测，以及配对受试者聚类自助法和置换检验，对病例对照、ADHD亚型和三类分类任务进行了评估。MovieSTAGE分别取得了0.69、0.73和0.75的AUROC以及67.6%、69.8%和58.3%的平衡准确率，在各类比较中获得了最高的平均点估计……

    arXiv:2610.09306v1 Announce Type: new  Abstract: Naturalistic movie-fMRI provides a shared, temporally structured probe of brain dynamics, yet predictive models commonly rely on whole-run functional connectivity (FC) or temporally generic representations that are not aligned with narrative events. We introduce MovieSTAGE (Scene, Transition, and Global Encoding), a multiscale framework that combines hypergraph-structured FC-profile organization within scenes, unsigned FC-profile differences across adjacent scenes, and whole-movie FC. We evaluated 260 participants from the CMI-HBN Despicable Me cohort on case-control, ADHD-subtype, and three-class classification using 10 repetitions of stratified five-fold cross-validation, complete out-of-fold (OOF) predictions, and paired subject-cluster bootstrap and permutation tests. MovieSTAGE achieved AUROCs of 0.69, 0.73, and 0.75 and balanced accuracies of 67.6%, 69.8%, and 58.3%, respectively, yielding the highest mean point estimates among the
    
[^237]: Kuration SDK：通过数据策展弥合“虚拟到真实”差距

    Kuration SDK: Addressing the Virtual2Real Gap via Data Curation

    [https://arxiv.org/abs/2610.09305](https://arxiv.org/abs/2610.09305)

    该论文发现现有基准指标（FVD、LPIPS、JEDi）与世界模型的定性可玩性不对应，即存在“虚拟到真实”差距，并提出在训练前通过数据策展策略和开源的Kuration SDK工具包测量数据诊断属性，以更稳健地弥合这一差距。

    

    用于衡量动作条件世界模型质量的基准测试仍在不断演进，正从基于视觉相似度的指标转向动作语义和物理基础的指标。然而，对于领域和任务无关的动作条件世界模型训练，现有基准测试所能提供的信号有限。通过在《反恐精英》游戏数据上训练和评估扩散世界模型，我们证实了定性的可玩性与FVD、LPIPS和JEDi等指标并不对应，我们将这一现象称为“虚拟到真实”差距。我们认为，在缺乏可靠基准测试的情况下，在训练开始之前对原始游戏数据进行策展并测量多种诊断属性，能够提供更稳健的信号来弥合这一差距。我们提出了多种数据策展策略，以及一个名为Kuration SDK的通用物理AI数据策展工具包，该工具包将随本论文开源。该SDK在揭示（摘要在此处被截断）

    arXiv:2610.09305v1 Announce Type: new  Abstract: Benchmarks for measuring the quality of action-conditioned world models are still evolving and shifting away from visual similarity-based metrics to action-semantic and physically-grounded metrics. However, for domain and task-agnostic action-conditioned world model training, existing benchmarks provide a limited signal. By training and evaluating diffusion world models on CounterStrike gameplay data, we confirm that qualitative playability does not correspond with metrics such as FVD, LPIPS, and JEDi. We term this the Virtual2Real gap. We posit that, in lieu of reliable benchmarks, curating raw gameplay data and measuring a variety of diagnostic properties provides a more robust signal to bridge the gap, before the training even begins. We present several curation strategies and a general-purpose kit for physical AI data curation called Kuration SDK, which is being open-sourced with this paper. The SDK was instrumental in uncovering the
    
[^238]: emg2face：基于高密度表面肌电信号的表现性面部动画

    emg2face: Expressive Facial Animation with High-Density Surface EMG

    [https://arxiv.org/abs/2610.09304](https://arxiv.org/abs/2610.09304)

    该论文提出利用高密度表面肌电信号（HD-sEMG）作为非光学替代方案，在被VR头显等设备遮挡面部的情况下，从64通道肌电数据实现表现性的面部动画重建。

    

    面部动作传递着微妙而重要的信息，这些信息对人类社交交流至关重要。当面部被头戴式设备（HMD，如VR头显）遮挡时，光学面部捕捉方法难以甚至无法使用。即使视线畅通，此类方法也会引发隐私方面的担忧，并且需要将摄像头和照明装置从面部移开的头戴式捕捉设备。我们证明了高密度表面肌电图（HD-sEMG）提供了一种可行的非光学替代方案，能够应对这些挑战。我们使用两个织物肌电网格测量了64个肌电通道，其中32个来自前额（通常被HMD遮挡），32个来自面部侧面。肌电数据以2048 Hz采样并进行滤波。同时记录面部动作，并使用MediaPipe的面部标志点工具估计出478个3D面部标志点。此类多模态记录中的一个主要挑战是肌电信号与视频的同步。

    arXiv:2610.09304v1 Announce Type: cross  Abstract: Facial movements convey subtle and important information that is critical for human social communication. Optical methods for face capture are difficult or impossible to use when the face is occluded by head-mounted devices (HMDs), such as VR headsets. Even with a clear line of sight, such methods raise privacy concerns and require head-mounted capture rigs that offset cameras and lighting from the face. We show that high-density surface electromyography (HD-sEMG) provides a viable non-optical alternative that addresses these challenges.   We measured 64 EMG channels, using two textile EMG grids, with 32 from the forehead (typically occluded by an HMD) and 32 from the side of the face. EMG data were digitized at 2048 Hz and filtered. Facial movements were simultaneously recorded and used to estimate 478 3D facial landmarks using MediaPipe's Face Landmarker. A major challenge in such multimodal recordings is synchronizing EMG and video 
    
[^239]: 节点级图神经架构搜索框架

    Node-level Graph Neural Architecture Search Framework

    [https://arxiv.org/abs/2610.09297](https://arxiv.org/abs/2610.09297)

    提出节点级图神经架构搜索算法N-GNAS，可在更新节点特征时为每个节点子集自动选择合适的网络架构，并借助对比学习损失缓解过度平滑问题。

    

    近年来，图神经网络（GNNs）与架构搜索框架凭借其处理非结构化数据的卓越能力，在非欧几里得数据处理领域得到了广泛应用。然而，传统方法通常对所有节点施加统一的卷积操作，而不考虑各节点在结构和特征特性上的差异，这可能损害模型性能，并随着网络层数的增加导致过度平滑问题。为克服这一局限，本工作提出了一种节点级图神经架构搜索算法（N-GNAS），该算法能够在更新节点特征时，自动为每个节点子集选择合适的网络架构。N-GNAS 还引入了对比学习损失，以将不同类别的样本特征相互分离。在八个数据集上进行的节点……

    arXiv:2610.09297v1 Announce Type: new  Abstract: In recent years, Graph Neural Networks (GNNs) and architecture search frameworks have gained extensive application in non-Euclidean data processing, attributable to their superior capacity in managing unstructured data. Nevertheless, traditional approaches typically apply uniform convolution operations to all nodes, regardless of their varying structural and feature characteristics, which can undermine model performance and result in over-smoothing issues as the number of layers increases. To overcome this limitation, in this work, we propose a \textbf{N}ode-Level \textbf{G}raph \textbf{N}eural \textbf{A}rchitecture \textbf{S}earch (N-GNAS) algorithm. It can automatically choose an appropriate network architecture for each subset of nodes when updating node features. N-GNAS also introduces a contrastive learning loss to separate sample features from different categories and vice versa. In experiments conducted on eight datasets for node 
    
[^240]: RT-Safe：实时具身环境中的智能体安全基准测试

    RT-Safe: Benchmarking Agent Safety in Real-Time Embodied Environment

    [https://arxiv.org/abs/2610.09294](https://arxiv.org/abs/2610.09294)

    提出RT-SAFE，一个在实时约束下评估具身智能体安全性的仿真城市基准，强调在物理世界持续演化的情况下，决策质量与决策延迟同等重要。

    

    人工智能智能体的快速发展使智能体安全性日益受到关注，目前大量的评估工作集中在数字环境中。随着智能体进入物理世界，具身安全变得愈发重要：一旦失败可能导致人身伤害和高昂的硬件损失。除了选择安全的动作之外，具身智能体还必须在实时约束下运行：物理世界不会在智能体推理时暂停。当行人在推理过程中移动、车辆不断逼近时，在观测时刻看似安全的动作，在执行之前可能已经变得不安全。因此，实时具身安全既取决于决策质量，也取决于决策延迟。我们提出了RT-SAFE，一个用于评估实时约束下具身智能体安全性的仿真城市基准。RT-SAFE将导航任务与移动的行为主体、环境危险和交通规则相结合，同时允许世界在推理和动作执行过程中持续演化。

    arXiv:2610.09294v1 Announce Type: cross  Abstract: Rapid progress in AI agents has brought growing attention to agent safety, with extensive evaluation focused on digital environments. As agents move into the physical world, embodied safety becomes increasingly important: failures can cause human injury and costly hardware damage. Beyond selecting safe actions, embodied agents must also operate under real-time constraints: the physical world does not pause while an agent reasons. As pedestrians move and vehicles approach during inference, an action that appears safe at observation time may become unsafe before execution. Real-time embodied safety therefore depends on both decision quality and decision latency. We introduce RT-SAFE, a simulated urban benchmark for evaluating embodied-agent safety under real-time constraints. RT-SAFE combines navigation tasks with moving actors, environmental hazards, and traffic rules, while allowing the world to evolve throughout inference and action e
    
[^241]: LeCuration：作为数据整理多用途工具的微型世界模型

    LeCuration: A Tiny World Model as a Data Curation Multi-Tool

    [https://arxiv.org/abs/2610.09285](https://arxiv.org/abs/2610.09285)

    该论文提出LeCuration——一个面向封闭物理世界的微型世界模型，其潜空间嵌入可同时用作异常检测信号与内容聚类依据，并能自回归预测游戏状态以供定性检查，从而为更大的下游物理AI模型提供多功能的数据整理工具。

    

    arXiv:2610.09285v1 公告类型：新论文 摘要：许多物理AI应用运行在有限或封闭的物理世界中，物体行为受一组有限的物理定律支配。例如在仓库中工作的机器人，以及在视频游戏中移动的智能体。为了更好地组织、过滤和整理物理AI应用的数据，我们提出了一种以各个数据集的独特设置和物理定律为核心的新方法。我们训练了LeCuration，这是一个小型世界模型，旨在作为一个独立的、规模更大的下游模型的数据整理工具。为构建该模型，我们选择LeWorldModel（LeWM）作为潜空间编码器和预测器，并添加扩散变换器解码器，为自回归的游戏过程推演提供视觉呈现。我们发现，该模型的嵌入向量可以用作异常检测信号和基于内容的聚类启发式方法，并且利用该模型自回归地预测游戏状态，使我们能够定性地检查……（原文摘要在此处被截断）

    arXiv:2610.09285v1 Announce Type: new  Abstract: Many applications of physical AI run within finite or closed physical worlds with a limited set of physical laws governing object behavior. Examples include robots working in a warehouse and agents moving around in a video game. In order to better organize, filter, and curate data for physical AI applications, we propose a new approach centered on the unique settings and physical laws of individual datasets. We train LeCuration, a small world model intended to serve as a data curation tool for a separate, larger downstream model. To build this model, we choose LeWorldModel (LeWM)as our latent encoder and predictor, adding a diffusion transformer (DiT) decoder to add visuals to autoregressive gameplay rollout. We find that the embeddings of this model can be used as an anomaly detection signal and as a content-based clustering heuristic, and that auto-regressively predicting the game state with this model allows us to qualitatively check 
    
[^242]: 基于共成像点道集构建地下速度模型的自注意力摘要网络

    Self-attention summary networks for subsurface velocity-model building from common-image gathers

    [https://arxiv.org/abs/2610.09282](https://arxiv.org/abs/2610.09282)

    该论文提出一种多尺度自注意力摘要网络，将高维共成像点道集压缩为保留运动学结构的条件嵌入，用于驱动流匹配模型进行概率性地下速度反演，从而显著改善后验速度推断效果。

    

    共成像点道集（CIGs）通过反射层聚焦和剩余时差包含了关于速度模型误差的具有物理意义的信息，但在传统成像工作流程中，它们通常仅被用作诊断工具。在这项工作中，我们提出了一种多尺度自注意力摘要网络，将高维的三维CIG体数据映射为紧凑的条件化嵌入向量，用于概率性的地下速度反演。这些学习到的嵌入保留了随偏移距变化的运动学结构和空间相干性，同时降低了由背景速度失配引起的变异性。以这些摘要嵌入为条件，一个流匹配模型学习从高斯源分布到合理速度场后验分布的传输映射。数值实验表明，与直接以原始CIGs为条件相比，所提出的摘要网络改善了后验速度推断。特别是，多尺度自注…（原文在此处截断）

    arXiv:2610.09282v1 Announce Type: new  Abstract: Common-image gathers (CIGs) contain physically meaningful information about velocity-model errors through reflector focusing and residual moveout, but in conventional imaging workflows they are typically used only as diagnostic tools. In this work, we propose a multiscale self-attention summary network that maps high-dimensional 3D CIG volumes into compact conditioning embeddings for probabilistic subsurface velocity inversion. These learned embeddings preserve offset-dependent kinematic structure and spatial coherence while reducing variability caused by background-velocity mismatch. Conditioned on these summary embeddings, a flow-matching model learns a transport from a Gaussian source distribution to the posterior distribution of plausible velocity fields. Numerical experiments show that, compared with direct conditioning on raw CIGs, the proposed summary network improves posterior velocity inference. In particular, the multiscale att
    
[^243]: 面向逆问题的扭转流

    Twist Flow for Inverse Problems

    [https://arxiv.org/abs/2610.09281](https://arxiv.org/abs/2610.09281)

    该论文提出联合扭转流这一增广流匹配方法，通过学习增广源状态与增广终端状态之间的连续输运，克服贝叶斯逆问题中确定性映射导致的后验变异性低估和多峰模式失真问题。

    

    在贝叶斯逆问题中，后验采样需要生成既与给定观测一致、又能涵盖所有合理解范围的样本。直接条件生成模型通过引入潜在噪声来建模这种模糊性，但基于成对数据的逆问题训练仍可能促使模型学习到从观测到目标的近乎确定性的映射。其结果是，生成的样本虽然与观测一致，却可能低估后验的变异性，尤其当后验呈多峰分布时，会导致覆盖不足、模式失真，或在不同可行解之间产生人为的过渡。我们提出联合扭转流，这是一种增广的流匹配框架，学习从增广源状态 $(z_x, y)$ 到增广终端状态 $(x, z_y)$ 的连续输运。其中 $x$ 是目标变量，$y$ 是观测，$z_x$ 是用于后验采样的高斯参考坐标，而……（摘要在此处截断）

    arXiv:2610.09281v1 Announce Type: new  Abstract: In Bayesian inverse problems, posterior sampling requires generating samples that are consistent with given observations while capturing the range of plausible solutions. Direct conditional generative models introduce latent noise to model this ambiguity, but paired inverse-problem training can still encourage an almost deterministic map from the observation to the target. As a result, generated samples may be observation-consistent while under-representing posterior variability, especially when the posterior is multimodal, leading to undercoverage, mode distortion, or artificial transitions between distinct feasible solutions. We propose joint twist-flow, an augmented flow-matching formulation that learns a continuous transport from the augmented source state $(z_x, y)$ to the augmented terminal state $(x, z_y)$. Here x is the target variable, $y$ is the observation, $z_x$ is the Gaussian reference coordinate for posterior sampling, and
    
[^244]: CATune：面向数据库管理系统配置调优的结构约束感知贝叶斯优化

    CATune: Structural Constraint-Aware Bayesian Optimization for DBMS Configuration Tuning

    [https://arxiv.org/abs/2610.09276](https://arxiv.org/abs/2610.09276)

    CATune提出了一种结构约束感知的贝叶斯优化框架，将DBMS配置参数间的确定性顺序约束直接建模为搜索域的结构组成部分，在约束一致的子空间内进行优化，从而提升数据库配置自动调优的效率与有效性。

    

    现代数据库管理系统暴露出数百个配置参数，导致高维且异构的搜索空间，使得自动化调优成本高昂。现有的基于机器学习的调优系统通常将配置域视为盒式约束空间，并依赖工作负载反馈来隐式捕捉参数间的关系。然而，DBMS文档明确规定了确定性的参数依赖约束，尤其是顺序约束，这些约束刻画了配置空间中结构上有效的区域。我们提出了CATune，一个约束感知的贝叶斯优化（BO）框架，它将确定性的参数间顺序约束建模为搜索域的结构组成部分。CATune并非通过采样违规来学习可行性边界，而是在约束一致的子空间内执行优化。我们开发了一种拓扑感知的采样策略，在探索过程中遵循依赖结构，并避免了……

    arXiv:2610.09276v1 Announce Type: cross  Abstract: Modern DBMSs expose hundreds of configuration knobs, resulting in a high-dimensional and heterogeneous search space that makes automated tuning costly. Existing ML-based tuning systems typically treat the configuration domain as box-constrained and rely on workload feedback to implicitly capture inter-knob relationships. However, DBMS documentation specifies deterministic knob dependency constraints, particularly ordering constraints, that characterize structurally valid regions of the configuration space. We present CATune, a constraint-aware Bayesian optimization (BO) framework that models deterministic inter-knob ordering constraints as structural components of the search domain. Instead of learning feasibility boundaries through sampled violations, CATune performs optimization within a constraint-consistent subspace. We develop a topology-aware sampling strategy that respects dependency structure during exploration and avoids the i
    
[^245]: 面向高效视觉几何Transformer的硬件感知校准聚类注意力

    Hardware-aware Calibrated Clustered Attention for Efficient Visual Geometric Transformers

    [https://arxiv.org/abs/2610.09274](https://arxiv.org/abs/2610.09274)

    提出硬件感知的分块聚类注意力（BC attention），通过块内聚类、哈希超平面校准和基于阈值的误差补偿来加速VGGT的全局注意力层，在长序列场景下实现GPU上实际的延迟提升。

    

    视觉几何基础Transformer（VGGT）标志着3D场景重建领域的重大飞跃，它是首个能够在一次前向传递中直接联合推断所有关键3D属性（相机位姿、深度和稠密几何）的模型。然而，这种联合推断机制需要处理极长序列的全局注意力层，从而造成了显著的延迟瓶颈。在本文中，我们提出分块聚类注意力（BC attention）来加速VGGT中的全局注意力层。通过将聚类限制在硬件友好的邻域块内，BC注意力降低了查询聚类的计算开销，以及片上与片外内存之间昂贵的数据搬运。这使得BC注意力能够扩展到长序列，并在GPU上带来实际的延迟改进。此外，我们引入了哈希超平面校准方法和基于阈值的误差补偿方法来减少聚类误差（摘要原文在此处截断）。

    arXiv:2610.09274v1 Announce Type: cross  Abstract: The Visual Geometry Grounded Transformer (VGGT) marks a significant leap forward in 3D scene reconstruction, as it is the first model that directly infers all key 3D attributes (camera poses, depths, and dense geometry) jointly in one pass. However, this joint inference mechanism requires global attention layers with extremely long sequences that causes a significant latency bottleneck. In this paper, we propose blockwise clustered attention (BC attention) to accelerate the global attention layers in VGGT. By limiting the clustering within HW-friendly neighborhood blocks, BC attention reduces the computation overhead of query clustering as well as the costly data movement between on- and off-chip memory. This enables BC attention to scale to long sequences and deliver practical latency improvements on GPUs. Moreover, we introduce a hashing hyperplane calibration method and a threshold-based error compensation method to reduce clusterin
    
[^246]: 评估用于最后一层注意力路由的轨迹特征

    Evaluating Trajectory Features for Routing Final-Layer Attention

    [https://arxiv.org/abs/2610.09272](https://arxiv.org/abs/2610.09272)

    该论文系统评估了外推误差、曲率等轨迹特征用于最后一层注意力路由的效果，发现这些特征在所有预设比较中均未带来显著增益，甚至不及简单的固定投影基线，表明轨迹特征对注意力路由的预测价值有限。

    

    注意力路由需要一个能够预测当前前缀上注意力价值的信号。我们评估隐藏状态外推误差、曲率和误差变化是否能在不确定性、单步位移、位置和状态投影之外改善这一预测。通过在冻结的 SmolLM3-3B-Base 和 Qwen3.5-4B-Base 模型检查点上对最后一注意力层进行成对执行，获得带符号的下一词元损失差异。在100本留出的 PG-19 书籍上，以相同的因果式20%调用配额测试由效用监督的路由器。经过族错误率校正后，六项预设比较均未显示出正向增益。在 Qwen3.5 中，一个参数量匹配的固定投影对照方法相对于轨迹路由器将 NLL 降低了 0.00356 nats/token（95% 置信区间为 0.00218 至 0.00487）。次要结果取决于被移除的操作、特征位置和评分范围；冻结的阈值在更长评分范围上也会出现显著漂移。

    arXiv:2610.09272v1 Announce Type: new  Abstract: Attention routing requires a signal that predicts the value of attention on the current prefix. We evaluate whether hidden-state extrapolation error, curvature and error change improve this prediction beyond uncertainty, one-step displacement, position and state projections. Paired executions of the final attention layer supply signed next-token loss differences in frozen SmolLM3-3B-Base and Qwen3.5-4B-Base checkpoints. Utility-supervised routers are tested on 100 held-out PG-19 books at an identical causal 20 percent invocation quota. None of six prespecified comparisons shows a positive gain after familywise correction. In Qwen3.5, a parameter-matched fixed-projection control lowers NLL by 0.00356 nats/token relative to the trajectory router (95 percent interval 0.00218 to 0.00487). Secondary results depend on the operation removed, feature location and scoring horizon; frozen thresholds also drift substantially at longer horizons. Act
    
[^247]: 代理模型之符：测量神经PDE求解器中的数值溯源

    The Symbol of the Surrogate: Measuring Numerical Provenance in Neural PDE Solvers

    [https://arxiv.org/abs/2610.09255](https://arxiv.org/abs/2610.09255)

    该论文提出一种基于傅里叶符号的经验诊断方法，用于判定神经PDE代理模型究竟忠实于精确物理演化还是仅仅模仿训练数值求解器的离散化误差，并发现代理模型几乎完全复制（超过99.8%）了训练格式的振幅与相位误差特征。

    

    神经PDE代理模型是在数值求解器的输出上训练的，而这些输出既包含物理演化，也包含求解器特有的离散化误差。由于代理模型又是使用同一求解器的保留轨迹进行评估的，标准基准无法区分模型是对精确演化的忠实还原，还是对数值格式的模仿。我们引入了一种经验性的傅里叶符号诊断方法，用单个傅里叶模式探测训练好的代理模型的线性化单步算子，并将其与精确演化和训练格式两种参考进行对比。为了解决架构固有的谱偏差问题，我们在具有正交的耗散与色散特征的格式上训练相同的网络，并比较它们学到的算子。在线性平流问题中，学习到的代理模型再现了训练格式的振幅和相位误差，其双格式差异达到了解析预测的完全模仿上限的99.8%以上。

    arXiv:2610.09255v1 Announce Type: new  Abstract: Neural PDE surrogates are trained on numerical solver outputs that contain both physical evolution and solver-specific discretization errors. Because surrogates are also evaluated against held-out trajectories from the same solver, standard benchmarks cannot distinguish fidelity to the exact evolution from imitation of the numerical scheme. We introduce an empirical Fourier-symbol diagnostic that probes a trained surrogate's linearized one-step operator with individual Fourier modes and compares it with both exact-evolution and training-scheme references. To address architectural spectral bias, we train identical networks on schemes with orthogonal dissipative and dispersive signatures and compare their learned operators. In linear advection, the learned surrogates reproduce the training schemes' amplitude and phase errors, with the twin-scheme difference reaching more than 99.8\% of the analytically predicted full-imitation ceiling. The
    
[^248]: 面向推理时对齐的高效Best-of-N策略评估

    Efficient Best-of-N policy evaluation for inference-time alignment

    [https://arxiv.org/abs/2610.09250](https://arxiv.org/abs/2610.09250)

    本文提出了一种无需访问响应似然值的仅样本BoN策略评估框架，利用BoN的顺序统计结构将密度比转化为可由样本估计的得分排名概率，并开发了能跨候选预算高效重用共享辅助样本池的双重稳健估计器BoN-DR，在奖励模型误设下仍保证有效的渐近推断。

    

    Best-of-N（BoN）是一种常见的推理时对齐方法，它从参考模型生成的N个样本中选择得分最高的响应。在仅有样本访问的条件下，从已记录数据中评估BoN策略极具挑战性，因为标准的离策略估计器需要依赖不可获得的响应似然值来计算密度比。在本文中，我们提出了一个仅需样本的框架，用于在无法访问这些似然值的情况下评估和选择BoN策略。我们证明，BoN的顺序统计结构使得所需的密度比可以通过得分排名概率来表达，而这些概率可以仅凭样本进行估计。随后，我们开发了一种BoN策略值的双重稳健估计器，它能够在候选预算之间高效地重用共享的辅助样本池。我们证明了即使在奖励估计器存在误设的情况下，该方法依然能够进行有效的渐近推断，并证明了BoN-DR估计器的有效性。由于更大的预算可以

    arXiv:2610.09250v1 Announce Type: new  Abstract: Best-of-N (BoN) is a common inference-time alignment method that selects the highest-scoring response among N samples from a reference model. Evaluating BoN policies from logged data is challenging under sample-only access because standard off-policy estimators require density ratios that depend on unavailable response likelihoods. In this paper, we propose a sample-only framework for evaluating and selecting BoN policies without access to these likelihoods. We show that the order-statistic structure of BoN allows the required density ratios to be expressed through score-rank probabilities that are estimable from samples alone. We then develop a doubly robust estimator of the BoN policy value (BoN-DR) that efficiently reuses a shared auxiliary sample pool across candidate budgets. We establish valid asymptotic inference even under reward estimator misspecification and prove the efficiency of our BoN-DR estimator. Since larger budgets can
    
[^249]: 目标条件策略学习中的一种视界信息诅咒

    An Informational Curse of Horizon in Goal-Conditioned Policy Learning

    [https://arxiv.org/abs/2610.09247](https://arxiv.org/abs/2610.09247)

    该论文发现目标条件策略学习中存在一种新的“信息性视界诅咒”：训练时目标重标注视界越长，行为克隆策略即使仅面对近处子目标也会出现严重性能退化，而强化学习目标可以缓解这一问题。

    

    学习目标达成策略的困难通常被归因于“视界诅咒”，其表现为时序差分备份中的偏差累积和带噪声的优势估计。在本工作中，我们在目标条件策略学习中识别出一种额外的“信息性视界诅咒”：增加目标重标注视界会显著降低策略的泛化能力和性能。通过一系列使用神谕规划器的受控实验，我们将训练时采样的目标视界与测试时要求策略达成的目标视界解耦。即使仅在一系列近处子目标上进行评估，目标条件行为克隆（BC）策略也会出现严重的、依赖训练视界的性能退化，而强化学习（RL）目标能够缓解这种退化。我们将这一现象解释为动作与事后信息之间的条件互信息随视界增加而下降。

    arXiv:2610.09247v1 Announce Type: new  Abstract: The difficulty of learning goal-reaching policies is often attributed to a "curse of horizon" that manifests as bias accumulation in temporal-difference backups and noisy advantage estimates. In this work, we identify an additional informational curse of horizon in goal-conditioned policy learning, where increasing the goal relabeling horizon can significantly reduce policy generalization and performance. Through a series of controlled experiments with oracle planners, we decouple the goal horizons sampled during training from those that the policy is asked to reach at test time. Even when evaluated only on a sequence of nearby subgoals, goal-conditioned behavioral cloning (BC) policies suffer from severe, training horizon-dependent performance degradation that is mitigated by reinforcement learning (RL) objectives. We explain this phenomenon as a horizon-dependent decrease in the conditional mutual information between actions and hindsi
    
[^250]: 超越名义均衡：风险厌恶多群体平均场博弈

    Beyond Nominal Equilibria: Risk-Averse Multi-Population Mean-Field Games

    [https://arxiv.org/abs/2610.09244](https://arxiv.org/abs/2610.09244)

    该论文提出了风险厌恶多群体平均场博弈这一新范式，让每个群体对其他群体平均场流的模糊集进行最坏情况优化，并利用占据测度表述与集值分析工具证明了新型风险厌恶均衡的存在性等理论性质。

    

    平均场博弈及其多群体变体的最新进展，使得大规模异构多智能体系统能够通过代表性智能体及其相关的平均场分布进行建模。然而，现有方法并未显式考虑其他群体行为中的不确定性。为此，我们提出了一种新范式：风险厌恶多群体平均场博弈，其中每个群体在其他群体子集的平均场流的动态可行模糊集上优化最坏情况下的期望奖励。通过采用占据测度表述并结合集值分析工具，我们在温和的假设下建立了该多群体博弈的若干理论性质，包括模糊集的几何性质以及一种新型风险厌恶多群体平均场均衡的存在性。此外，我们还导出了不动点迭代的收缩性结果。

    arXiv:2610.09244v1 Announce Type: cross  Abstract: Recent advances in mean-field games and its multi-population variants enable large-scale heterogeneous multi-agent systems to be modeled through representative agents and their associated mean-field distributions. However, existing approaches do not explicitly account for uncertainty in the behavior of other populations. To this end, we introduce a new paradigm: risk-averse multi-population mean-field games, where each population optimizes a worst-case expected reward over dynamically feasible ambiguity sets of mean-field flows of a subset of the other populations. Employing an occupation-measure formulation along with tools from set-valued analysis, we establish, under mild assumptions, several theoretical properties of the multi-population game, including the geometric properties of the ambiguity sets and the existence of a novel risk-averse multi-population mean-field equilibrium. Further, we derive contractivity results of the fixe
    
[^251]: 大语言模型自我改进循环中的胜者诅咒：选择噪声、锁定效应与接受规则

    The Winner's Curse in LLM Self-Improvement Loops: Selection Noise, Lock-in, and Acceptance Rules

    [https://arxiv.org/abs/2610.09239](https://arxiv.org/abs/2610.09239)

    该论文将LLM自我改进循环中的“更优则保留”步骤建模为测量噪声下的选择问题，揭示并量化了胜者诅咒现象：首轮之后的大多数自我修改实为有害，且复用小评估集会使选择集得分严重高估真实泛化能力（小选择集时高估达13至20个百分点）。

    

    自我改进的LLM系统会对自身提出修改建议，并保留那些在小型评估集上得分更高的修改。我们将这一“更优则保留”的步骤视为测量噪声下的选择问题，对单次决策中候选者之间的相关误差进行建模，并实证研究了重复使用评估集时会发生什么。在Qwen模型重写自身指令的实验中，每个候选修改同时还在600个保留项目上评分，结果显示第一轮之后的大多数提议都是有害的，而我们的模型能够给出每一代最佳候选的胜者诅咒的大小。借助来自单独试点实验的先验，该模型能够匹配原生循环中第一代提交的平均夸大程度，尽管并非在每个设置下都完全吻合。在一项预注册研究中，当选择集仅含16个项目时，贪心循环的最终选择集得分比保留集准确率高出13至20个百分点；当选择集含256个项目时，则高出1至5个百分点。在TREC数据集上，保留集收益随选择集规模的增大而增长，但……

    arXiv:2610.09239v1 Announce Type: cross  Abstract: Self-improving LLM systems propose changes to themselves and keep those that score better on a small evaluation set. We treat this keep-if-better step as selection under measurement noise, model the correlated errors of the candidates in a single decision, and study empirically what happens when the evaluation set is reused. In runs where Qwen models rewrite their own instructions and every candidate is also scored on 600 held-out items, most proposals after the first are harmful, and the model gives the size of the winner's curse of a generation's best candidate. With a prior from a separate pilot, it matches the average overstatement of first-generation commits in native loops, though not setting by setting. In a pre-registered study, the final selection-set score of greedy loops exceeded held-out accuracy by 13 to 20 points with 16 selection items and by 1 to 5 points with 256. Held-out gains grew with the selection set on TREC but 
    
[^252]: 对称性驱动的因果部分识别

    Symmetry-Informed Causal Partial Identification

    [https://arxiv.org/abs/2610.09230](https://arxiv.org/abs/2610.09230)

    该论文首次将已知的数据对称性（即因果效应在某些数据变换下的不变性）作为部分识别的新约束来源，通过因果函数形状约束和测度变换两种方法有效收紧了因果效应的识别边界。

    

    部分识别（PI）是指通过将关于数据生成的不同假设编码为约束优化问题，来估计因果效应的边界。即使因果效应本身不可识别，这样的边界也足以指导政策决策。然而在实践中，这类边界往往是空洞（无信息量）的，因此从业者试图尽可能详尽地将领域知识编码为额外约束，以使PI边界更具信息量。我们引入已知的数据对称性——即因果效应在某些数据变换下保持不变——作为为PI提供信息的新约束来源。我们将这一思想具体化为对因果函数的形状约束，并通过简单的数据预处理进行测度变换，据此提出PI问题。在两种经典的PI模型下，这两种方法均被证明可以收紧边界。这一结论在总体情形下得到了理论证明，并在有限样本情形下通过实验得到验证。更广泛而言，我们的框架建立了……

    arXiv:2610.09230v1 Announce Type: new  Abstract: Partial identification (PI) entails estimating bounds on causal effects by encoding different assumptions on data generation as a constrained optimization problem. Such bounds can suffice to inform policy decisions even if the causal effect itself is not identifiable. Often vacuous in practice, practitioners seek to exhaustively encode domain knowledge as additional constraints to make the PI bounds more informative. We introduce known data symmetries -- invariance of the causal effect under certain data transformations -- as a new source of constraints to inform PI. We operationalize this as a shape constraint on the causal function, and via a change of measure against which PI is posed using simple data pre-processing. Both approaches are shown to sharpen bounds under two canonical PI models. This is shown both theoretically for the population case, and via experiments in the finite-sample case. More broadly, our framework establishes 
    
[^253]: 条件准确度剖析：跨部署条件诊断LLM评判器

    Conditional Accuracy Profiles: Diagnosing LLM Judges across Deployment Conditions

    [https://arxiv.org/abs/2610.09229](https://arxiv.org/abs/2610.09229)

    提出CAP诊断框架，将LLM评判器的准确率分解为内容敏感性、鲁棒性和理由质量三类的八个条件，从而揭示被总体准确率数字掩盖的部署条件差异。

    

    LLM-as-judge（以大语言模型作为评判器）目前已成为可扩展评估的标准工具，但评判器的性能通常仍由单一的准确率数字来概括。这种总体视角掩盖了评判器在哪些部署条件下成功或失败。我们提出了条件准确度剖析，这是一种事后诊断框架，将成对LLM评判准确率分解为八个条件，并组织为内容敏感性、鲁棒性和理由质量三大类。CAP与具体基准无关：当基准提供所需标注时可直接应用，也可以通过任务子集代理近似应用，或者在能生成扰动对时通过受控增强来应用。我们在六个成对评判基准上对七个LLM评判器实例化了CAP，其中包括judgerEva-Standard——我们创建的一个支持全部八个条件的受控测试平台。CAP揭示了被总体准确率所掩盖的剖析差异：在judgerEva'上……

    arXiv:2610.09229v1 Announce Type: new  Abstract: LLM-as-judge is now a standard tool for scalable evaluation, but judge performance is still often summarized by a single accuracy number. This aggregate view hides the deployment conditions under which a judge succeeds or fails. We introduce \textbf{Conditional Accuracy Profiling} (CAP), a post-hoc diagnostic framework that decomposes pairwise LLM-judge accuracy into eight conditions organized into content sensitivity, robustness, and rationale quality. CAP is benchmark-agnostic: it can be applied directly when a benchmark provides the required annotations, approximately through task-subset proxies, or through controlled augmentation when perturbation pairs can be generated. We instantiate CAP on seven LLM judges across six pairwise judging benchmarks, including \textsc{judgerEva-Standard}, a controlled testbed we created to support all eight conditions. CAP exposes profile differences hidden by aggregate accuracy: on \textsc{judgerEva}'
    
[^254]: 基于漂移的量化：面向文本嵌入器的无标签混合精度训练后量化

    Quantize by Drift: Label-Free Mixed-Precision Post-Training Quantization for Text Embedders

    [https://arxiv.org/abs/2610.09227](https://arxiv.org/abs/2610.09227)

    该论文提出以量化引起的输出嵌入漂移作为无标签的模块敏感度信号，用于文本嵌入器的混合精度训练后量化，该信号与检索质量高度相关（宏观Spearman达0.911），无需部署中难以获取的相关性标注。

    

    混合精度训练后量化需要一个逐模块的敏感度信号；对于文本嵌入器而言，最直观的信号——某个模块被量化时所损失的检索质量——需要部署场景中很少具备的相关性标注。我们测量了一种无标签的替代信号：量化引起的表示漂移，其获取方式是量化单个模块、重新编码语料库，并记录输出嵌入相对于其全精度位置的偏移距离。其独特之处在于所使用的观测量：即稠密检索器用于排序的部署态输出表示。在五个开发用嵌入器上，配置级漂移对采样得到的混合精度方案相对于留出集检索质量的排序达到了0.911的宏观Spearman相关系数；在可用范围内，该敏感度可跨校准语料库和检索域迁移；各模块的漂移在排序上可一致地组合，但在数值上则不然；而基于相关性导出的敏感度并未带来一致的价值提升。

    arXiv:2610.09227v1 Announce Type: cross  Abstract: Mixed-precision post-training quantization needs a per-module sensitivity signal; for a text embedder the obvious one -- the retrieval quality a module costs when quantized -- needs relevance labels that deployments rarely have. We measure a label-free substitute: quantization-induced representation drift, obtained by quantizing one module, re-encoding the corpus, and recording how far the output embeddings moved from their full-precision positions. What is specific is the observable: the deployed output representation a dense retriever ranks with. Across five development embedders, configuration-level drift orders sampled mixed-precision plans against held-out retrieval quality at a macro Spearman of 0.911, the sensitivity transports across calibration corpora and retrieval domains in the usable regime, module drifts compose rank-consistently but not numerically, and relevance-derived sensitivity adds no consistent value. The method i
    
[^255]: FLoRa：能量约束下基于飞行辅助的占空比休眠LoRa节点数据收集

    FLoRa: Flight-Assisted Data Collection from Duty Cycling LoRa Nodes under Energy Constraints

    [https://arxiv.org/abs/2610.09226](https://arxiv.org/abs/2610.09226)

    FLoRa提出了一种融合模拟退火路径规划、CMA-ES悬停定位和POMDP探测决策的多层次无人机数据收集架构，并引入VIP指标，用于在能量约束下从占空比休眠的LoRa物联网设备中高效收集数据。

    

    当LoRa物联网设备（IoTD）采用占空比方式休眠以节省电池电量时，使用无人机（UAV）进行数据收集面临巨大挑战。在能量约束条件下，无人机必须决定访问哪些物联网设备、以何种顺序访问、在何处悬停以及探测每个节点的次数，而与此同时基于时间的数据新鲜度会不断衰减。要可解地解决这一问题，需要一个多层次优化架构：用于路径规划的离散组合优化、用于空间定位的连续全局优化，以及不确定性条件下的序贯决策。我们提出了FLoRa，一种飞行辅助的LoRa数据收集架构，采用模拟退火算法（SA）进行路径规划，采用协方差矩阵自适应进化策略（CMA-ES）进行悬停位置优化，并采用部分可观测马尔可夫决策过程（POMDP）对物联网设备进行探测。为了量化从占空比休眠节点收集数据的效用，我们引入了面向拉取式系统的信息价值指标（VIP）……

    arXiv:2610.09226v1 Announce Type: cross  Abstract: Data collection using Unmanned Aerial Vehicles (UAVs) is challenging when LoRa IoT Devices (IoTDs) duty-cycle to conserve battery. Under energy constraints, a UAV must decide which IoTDs to visit, in what order, where to hover, and how many times to probe each node, while time-based data freshness decays. Tractably solving this problem requires a multi-level optimization architecture: discrete combinatorial optimization for routing, continuous global optimization for spatial positioning, and sequential decision-making under uncertainty. We propose FLoRa, a Flight-assisted LoRa data collection architecture using Simulated Annealing (SA) for path planning, Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for hover positioning, and Partially Observable Markov Decision Processes (POMDPs) for probing IoTDs. To quantify collection utility from duty-cycling nodes, we introduce the Value of Information for Pull-based systems (VIP), a m
    
[^256]: 面向无数据扩散蒸馏的一致分布匹配

    Consistent Distribution Matching for Data-Free Diffusion Distillation

    [https://arxiv.org/abs/2610.09221](https://arxiv.org/abs/2610.09221)

    提出一致分布匹配方法，通过单一学生网络统一样本生成与分数估计，仅需一个冻结教师模型和一个可训练学生模型优化单一目标，即可实现无需模拟、无需数据的扩散与流模型加速蒸馏，并证明了学生分布向教师边缘分布的Wasserstein收敛性。

    

    流模型和扩散模型由于计算代价高昂的数值积分而推理缓慢。蒸馏为学生模型从教师模型的动态中学习提供了一种有前景的方法，可实现一步或几步生成。然而，现有方法通常依赖于精心构建的蒸馏数据集、代价高昂的教师模型rollout或辅助代理网络，这使得模型训练和扩展变得复杂。在本工作中，我们提出了一致分布匹配，这是一种无需模拟、无需数据的蒸馏方法，可在加速扩散模型和流模型的同时保持强大的生成能力。我们的关键洞察是用单一学生网络统一样本生成和分数估计。因此，我们的框架仅使用两个模型——一个冻结的教师模型和一个可训练的学生模型——并优化单一目标函数。我们证明，最小化我们的目标函数意味着学生流映射的前推分布对教师边缘分布的Wasserstein收敛。

    arXiv:2610.09221v1 Announce Type: new  Abstract: Flow and diffusion models suffer from slow inference due to computationally expensive numerical integration. Distillation provides a promising way for a student model to learn from a teacher's dynamics, enabling one-step or few-step generation. However, existing methods often depend on curated distillation datasets, costly teacher rollouts, or auxiliary proxy networks, which complicate model training and scaling. In this work, we propose Consistent Distribution Matching, a simulation-free and data-free distillation method for accelerating diffusion and flow models while preserving strong generative capacity. Our key insight is to unify sample generation and score estimation with one student network. Thus, our framework uses only two models, a frozen teacher and a trainable student, and optimizes one objective. We prove that minimizing our objective indicates Wasserstein convergence of the student flow-map pushforwards to the teacher marg
    
[^257]: CM-DPO：面向大语言模型规划的约束边际直接偏好优化

    CM-DPO: Constraint-Margin Direct Preference Optimization for LLM Planning

    [https://arxiv.org/abs/2610.09219](https://arxiv.org/abs/2610.09219)

    本文提出CM-DPO，用基于符号验证器、按违例严重程度缩放的连续约束边际取代DPO的二值偏好信号，并通过字典序目标区分硬软约束，结合SynPlan-R框架生成低偏差偏好数据，显著提升了8B模型在TravelPlanner、NaturalPlan和PlanBench等规划基准上的表现。

    

    直接偏好优化（DPO）将所有约束违例同等对待：1美元的预算超支与1000美元的预算超支会产生相同的训练信号。当偏好对来自不同模型家族时，DPO还容易受到长度和风格偏差的影响。我们提出了约束边际直接偏好优化（CM-DPO），它用源自确定性符号验证器、并按违例严重程度进行缩放的连续边际来取代DPO的二值偏好信号。通过字典序目标将硬约束与软约束分离，确保硬约束永远不会被偏好所牺牲。为了向CM-DPO提供降低偏差的训练对，我们在一个名为SynPlan-R的框架内，通过程序化生成的约束配置（DCCG）以及来自推理教师的最小编辑蒸馏（RT-MED）来生成偏好数据。在TravelPlanner、NaturalPlan以及分布外的PlanBench上，使用CM-DPO微调的8B模型取得了显著的性能提升。

    arXiv:2610.09219v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) treats all constraint violations equally: a $1 budget overshoot and a $1,000 overshoot induce the same training signal. It is also susceptible to length and style bias when preference pairs come from different model families. We introduce Constraint-Margin DPO (CM-DPO), which replaces DPO's binary preference signal with a continuous margin derived from a deterministic symbolic verifier and scaled by violation severity. Hard and soft constraints are separated through a lexicographic objective, ensuring hard constraints are never traded off against preferences. To supply CM-DPO with bias-reduced training pairs, we generate preference data through procedurally generated constraint profiles (DCCG) and minimal-edit distillation from a reasoning teacher (RT-MED), within a framework we call SynPlan-R. On TravelPlanner, NaturalPlan, and out-of-distribution PlanBench, an 8B model fine-tuned with CM-DPO achie
    
[^258]: CurveTQ：通过曲率加权搜索实现无需旋转的大语言模型权重格状量化

    CurveTQ: Rotation-Free Trellis Quantization of LLM Weights via Curvature-Weighted Search

    [https://arxiv.org/abs/2610.09212](https://arxiv.org/abs/2610.09212)

    提出CurveTQ，将Hessian矩阵LDL分解的对角线权重嵌入Viterbi分支度量，实现无需旋转的曲率加权格状量化，性能可媲美基于随机正交旋转的现有最优二比特量化方法

    

    目前大语言模型最优的二比特权重量化器（如QTIP和Proteus）会先用随机正交变换旋转每个权重矩阵（该变换在每个解码步骤中都必须被撤销），然后在欧氏搜索下用格状码或格点码对其进行编码；层Hessian矩阵仅通过编码块之间的误差反馈发挥作用。我们证明这种做法使Hessian矩阵的一部分信息未被利用。误差反馈将损失转化为各坐标舍入误差的加权和，其权重是Hessian矩阵LDL分解的对角线——现有量化器虽然计算了这些权重却从未加以利用。我们将这些权重引入Viterbi分支度量，使搜索在每个编码块内跟随曲率变化。这也解释了旋转的作用：旋转消除了块内变化，因此在原始基底下加权与旋转是可互相替代的方案。在三个模型上，加权原始基底搜索的性能与全维度随机化Hadamard变换相差约一个点以内

    arXiv:2610.09212v1 Announce Type: new  Abstract: The best two-bit weight quantizers for large language models, such as QTIP and Proteus, rotate each weight matrix by a random orthogonal transform, which must be undone at every decoding step, then encode it with a trellis or lattice code under a Euclidean search; the layer Hessian enters only through error feedback between coding blocks. We show that this leaves part of the Hessian unused. Error feedback turns the loss into a weighted sum of per-coordinate rounding errors whose weights, the diagonal of the Hessian's LDL factorization, existing quantizers compute but never read. We put these weights into the Viterbi branch metric, so the search follows the curvature within each coding block. This also explains the rotation: it removes this within-block variation, so weighting in the native basis and rotating are substitutes. On three models the weighted native search matches a full-dimension randomized Hadamard to within about one point 
    
[^259]: 损失差条件互信息的精度—信息权衡

    An Accuracy--Information Tradeoff for Loss-Difference Conditional Mutual Information

    [https://arxiv.org/abs/2610.09206](https://arxiv.org/abs/2610.09206)

    论文证明了精度与信息之间的权衡：在逻辑损失等光滑凸损失及幂次正则化条件下，任何以最优样本量达到低超额风险的正规学习器，其最坏情况损失差条件互信息必然达到 n 比特量级。

    

    损失差条件互信息（ld-CMI）是泛化上界的超样本层次结构中最小的标准观测量：它衡量学习器的损失差在多大程度上泄露了它训练时使用的是每一对候选样本中的哪一个。已知精度会迫使信息进入模型；而数据处理不等式并不能将此类下界传递到损失上。我们通过对损失差的三个矩进行约束，证明了精度同样会迫使ld-CMI。对于使用在零点斜率非零的光滑凸损失（如逻辑损失）的线性预测器，加上曲率和增长均呈幂次 r≥2 的正则化器，在维度至少随 n 线性增长的缩放符号立方体上的乘积分布中，每个在最优样本量 n≍ε^(-2+2/r) 下于这些分布上期望超额风险至多为 ε 的正规学习器，其最坏情况ld-CMI达到 n 比特量级，且为 Θ(n/(1+(τ/φ…

    arXiv:2610.09206v1 Announce Type: new  Abstract: Loss-difference conditional mutual information (ld-CMI) uses the smallest of the standard observations in the supersample hierarchy of generalization bounds: it measures what a learner's loss differences reveal about which candidate of each pair it was trained on. Accuracy is known to force information into the model; data processing does not carry such lower bounds to losses. We show, by bounding three moments of the loss differences, that accuracy also forces ld-CMI. For linear predictors with a smooth convex loss of nonzero slope at zero, such as the logistic loss, plus a regularizer whose curvature and growth are both of power $r\ge2$, on product distributions over a scaled sign cube in dimension at least linear in $n$, every proper learner with expected excess risk at most $\varepsilon$ on these distributions at the optimal sample size $n\asymp\varepsilon^{-2+2/r}$ has worst-case ld-CMI of order $n$ bits, and $\Theta(n/(1+(\tau/\var
    
[^260]: 极少比特，统一法则：迈向W2A4KV2

    Few Bits, One Law: Toward W2A4KV2

    [https://arxiv.org/abs/2610.09202](https://arxiv.org/abs/2610.09202)

    提出统一的量化感知训练框架CanonQ，通过源规范化与任务感知适配相分离，实现权重2比特、激活4比特、KV缓存2比特（W2A4KV2）的联合极端低比特压缩。

    

    当权重、激活值和KV缓存同时进行量化时，极端低比特的大语言模型压缩最具挑战性：它们的分布各不相同，且量化误差会在整个网络中相互影响。我们提出了CanonQ，一个统一的量化感知训练框架，通过将源数据规范化与任务感知适配相分离来应对这些挑战。固定的旋转和能量归一化将异构的张量源映射到规范坐标系，使冻结的高斯参考码本能够在不同层和不同模型之间复用。随后，联合训练使网络适应权重、激活值和缓存量化在统一标量/向量接口下产生的耦合误差。我们为冻结码本的迁移误差和局部任务损失给出了理论界，并推导出一个精确的归一化感知直通雅可比矩阵，将量化失真与梯度偏差联系起来。在联合W2A4KV2压缩下取得了最显著的性能提升。

    arXiv:2610.09202v1 Announce Type: cross  Abstract: Extreme low-bit LLM compression is most challenging when weights, activations, and KV caches are quantized together: their distributions differ, and quantization errors interact throughout the network. We introduce CanonQ, a unified quantization-aware training framework that addresses these challenges by separating source canonicalization from task-aware adaptation. Fixed rotations and energy normalization map heterogeneous tensor sources to canonical coordinates, enabling frozen Gaussian-reference codebooks to be reused across layers and models. Joint training then adapts the network to the coupled errors of weight, activation, and cache quantization within a common scalar/vector interface. We bound frozen-codebook transfer error and local task loss, and derive an exact normalization-aware straight-through Jacobian that links quantization distortion to gradient bias. The strongest gains arise under joint W2A4KV2 compression: across LL
    
[^261]: FreeEvolve：学习超越固定循环的进化方式

    FreeEvolve: Learning to Evolve Beyond Fixed Loops

    [https://arxiv.org/abs/2610.09197](https://arxiv.org/abs/2610.09197)

    FREEEVOLVE 通过元进化将“如何优化”本身变成从经验中学到的能力，让进化器摆脱人工设计的固定搜索循环，自主决定测试什么、收集多少证据、保留哪些候选方案以及何时停止。

    

    智能体进化器（agent evolvers）能够自动化设计围绕语言模型智能体的提示词、技能和工作流程，然而它们所遵循的优化过程本身仍是人工设计的：一个固定的搜索循环决定如何评估候选方案、保留哪些候选方案以及何时停止搜索。我们提出 FREEEVOLVE，它将这一过程也予以自动化。环境指定目标、目标智能体、评估器、数据和资源限制；在这些限制范围内，进化器自身决定测试什么、收集多少证据、追求哪些候选方案以及何时停止。这些决策遵循一个可编辑的进化技能，我们通过元进化（meta-evolution）来改进该技能——即在每个候选技能所产出的全新目标智能体上对其进行评分。由此，优化过程成为从经验中学习到的能力，而非预先设计的循环。在 tau3-bench、ARC-AGI-2、ARC-AGI-3 和 Terminal-Bench 2.1 上，FREEEVOLVE 自主掌控整个进化活动……

    arXiv:2610.09197v1 Announce Type: cross  Abstract: Agent evolvers automate the design of the prompts, skills and workflows around language model agents, yet the optimization process they follow is still designed by hand: a fixed search loop decides how candidates are evaluated, which are kept and when the search stops. We propose FREEEVOLVE, which automates this process as well. An environment specifies the goal, target agent, evaluator, data and resource limits; within these limits, the evolver itself decides what to test, how much evidence to collect, which candidates to pursue and when to stop. These decisions follow an editable evolution skill, which we improve through meta-evolution by scoring each candidate skill on the fresh target agent it produces. The optimization process thus becomes a capability learned from experience rather than a loop engineered in advance. On tau3-bench, ARC-AGI-2, ARC-AGI-3 and Terminal-Bench 2.1, FREEEVOLVE controls the evolution campaign by itself, y
    
[^262]: 线性递归特征机的精确动力学与有限样本轨迹恢复

    Exact Dynamics and Finite-Sample Trajectory Recovery of Linear Recursive Feature Machines

    [https://arxiv.org/abs/2610.09196](https://arxiv.org/abs/2610.09196)

    本文将线性递归特征机与迭代重加权最小二乘的联系推广到岭正则化含噪多输出回归，并证明了其学习到的特征矩阵在每次迭代中都以 $O(\sqrt{d/n})$ 的误差速率逼近无限数据下的理想结果。

    

    递归特征机通过交替执行两个步骤来学习数据的表示：将预测器拟合到数据集上，以及利用平均梯度外积（AGOP）更新该预测器的特征。AGOP与神经网络中特征学习之间的联系，使得线性递归特征机（RFM）成为分析训练过程中表示如何演化的一个简单设定。本文研究了线性RFM在含噪多输出回归中的动力学与统计性质，其中输入数据为各向同性的亚高斯数据，目标由维度为 $d$ 的低秩教师矩阵生成。我们将线性RFM与迭代重加权最小二乘法之间已知的联系，从插值情形扩展到带噪声的岭正则化多输出回归。我们证明了学习到的特征矩阵在每一次迭代中都保持接近其无限数据下的理想对应物。具体而言，对于 $n$ 个样本，我们证明了特征矩阵的误差以 $O(\sqrt{d/n})$ 的速度衰减。（摘要原文在此处被截断）

    arXiv:2610.09196v1 Announce Type: new  Abstract: Recursive feature machines (RFMs) learn representations of data by alternating between fitting a predictor to a dataset and updating features of that predictor using the average gradient outer product (AGOP). Connections between AGOPs and feature learning in neural networks motivate linear RFMs as a simple setting for analyzing how representations evolve during training. Here, we study the dynamics and statistics of linear RFM in noisy multi-output regression with isotropic sub-Gaussian input data and targets generated by a low-rank teacher matrix of dimension $d$. We extend the known connection between linear RFM and iteratively reweighted least squares from the interpolating setting to ridge-regularized multi-output regression with noise. We show that the learned feature matrix remains close to its infinite-data ideal counterpart at every iteration. Namely, for $n$ samples, we show the error in the feature matrix decays as $O(\sqrt{d/n
    
[^263]: 患者、地点、先验（P³）：什么才算医学世界模型中的个性化？

    Patient, Place, Prior (P$^3$): What Counts as Personalization in Medical World Models?

    [https://arxiv.org/abs/2610.09194](https://arxiv.org/abs/2610.09194)

    该论文提出了P³审计框架来检验医学世界模型的预测是否真正实现了患者个性化，并提出Cancer JEPA单步模型，通过在患者条件化降秩回归基线上加入基于遮挡潜在目标训练的病灶约束神经校正，来预测新辅助治疗中未来乳腺MRI检查的冻结表征。

    

    纵向模型预测患者的影像状态如何演变，但准确性并不能表明预测是否由患者自身的观测轨迹所驱动。人群平均预测可能有用的，但无法确立患者特定世界模型的主张。我们提出了“患者、地点、先验”（P³）审计框架，检验预测是否受益于患者的纵向影像历史（Patient/患者）、是否受益于与患者匹配的外部空间支持（Place/地点），以及在匹配的支持和情境下，是否在人群平均预测之外获得额外的预测价值（Prior/先验）。我们还提出了Cancer JEPA，这是一个单步模型，用于预测新辅助治疗期间未来乳腺动态对比增强MRI检查的冻结表征。该模型在患者条件化的低复杂度降秩回归基线之上，添加了一个通过基于遮挡的潜在目标训练的病灶约束神经校正。

    arXiv:2610.09194v1 Announce Type: new  Abstract: Longitudinal models forecast how a patient's imaging state evolves, but accuracy does not show whether the patient's observed trajectory drives the prediction. A population-average forecast may be useful but cannot establish a patient-specific world-model claim. We introduce Patient, Place, Prior (P$^3$), an audit asking whether a forecast benefits from the patient's longitudinal imaging history (Patient), benefits from patient-matched externally supplied spatial support (Place), and gains predictive value beyond a population-average prediction under matched support and context (Prior). We also propose Cancer JEPA, a one-step model that forecasts frozen representations of future breast dynamic contrast-enhanced MRI examinations during neoadjuvant therapy. It adds a lesion-constrained neural correction, trained with an occlusion-based latent objective, to a patient-conditioned low-complexity reduced-rank regression baseline. This factoriz
    
[^264]: 模式识别与逐步推理之间的二分法

    The Dichotomy Between Pattern Recognition and Step-by-Step Reasoning

    [https://arxiv.org/abs/2610.09186](https://arxiv.org/abs/2610.09186)

    论文提出模式识别与逐步推理是同一光谱的两端：当下一个词元仅依赖少量前文时 LLM 学会逐步推理，任务的推理轨迹构成 De Bruijn 图的有向无环子图，且其边数远小于轨迹数，因此 LLM 可以通过组合少量已学习的边来解决更长的新任务。

    

    我们认为，模式识别与逐步推理是同一光谱的两端。当数据的结构使得下一个词元仅依赖于少量前文时，大语言模型（LLM）学会逐步推理；当下一个词元依赖于大量前文时，LLM 的推理则类似于模式识别。如果下一个词元仅依赖于最近的 $c$ 个词元，那么推理轨迹就是 De Bruijn 图上的路径，其中图的节点是长度为 $c$ 的上下文，边是上下文之间的下一词元转移。一个任务的推理轨迹集合构成该 De Bruijn 图的一个有向无环子图。已经学习了该子图所有边的 LLM 可以将这些边组合起来，解决更长、未见过的任务，即它能够逐步推理。我们证明，边的数量相比推理轨迹的数量是极其微小的。从实验上看，Transformer 所需的训练样本数量（摘要至此截断）

    arXiv:2610.09186v1 Announce Type: new  Abstract: We argue that pattern recognition and step-by-step reasoning are two ends of a spectrum. A large language model (LLM) learns to reason step-by-step when data is structured such that the next token depends on a small amount of preceding context. Inference in LLMs resembles pattern recognition when the next token depends on a large amount of preceding context. If the next token depends on only the $c$ most recent tokens, reasoning traces are paths on a De Bruijn graph whose nodes are $c$-length contexts and edges are next-token transitions between contexts. The set of reasoning traces of a task forms a directed acyclic subgraph of the De Bruijn graph. An LLM that has learned all edges of this subgraph can compose them to solve longer, unseen tasks, i.e., it reasons step-by-step. We prove that the number of edges is vanishingly small compared to the number of reasoning traces. Empirically, the number of training samples a transformer needs 
    
[^265]: Q-PACE：面向量化感知训练的动态精度分配

    Q-PACE: Dynamic Precision Allocation for Quantization-Aware Training

    [https://arxiv.org/abs/2610.09183](https://arxiv.org/abs/2610.09183)

    Q-PACE提出一种基于二阶曲率敏感度的动态混合精度分配方法，在量化感知训练过程中定期测量各层敏感度并重新分配精度，从而在大幅降低计算成本的同时保持模型性能。

    

    量化感知训练（QAT）利用低精度算术来降低大语言模型（LLM）部署的成本，但过于激进的量化会损害最终模型的性能。一种常见的补救措施是混合精度训练，即为部分层分配高精度，在保持成本受限的同时维持性能。这种方法需要在训练过程中为模型各层进行精度分配。我们提出了一种名为Q-PACE的新方法，其核心是一个二阶敏感度模型，该模型将损失增量预测为量化噪声均方误差（MSE）与逐层曲率系数加权后的总和。在训练过程中，我们定期利用跨层扰动重新计算这些系数，并重新分配各层的精度。在参数量高达40亿的大语言模型上开展的预训练和监督微调实验表明，Q-PACE持续优于现有的混合精度训练方案，并能在显著更低的（精度）成本下实现相当的损失表现。

    arXiv:2610.09183v1 Announce Type: new  Abstract: Quantization-aware training (QAT) leverages lower-precision arithmetic to reduce the cost of LLM deployment, but aggressive quantization degrades final model performance. A common remedy is mixed-precision training, in which high precision is assigned to some of the layers to maintain performance while keeping the cost constrained. This approach then requires precision assignments for model layers during training. We provide a new approach, called Q-PACE, consisting of a second-order sensitivity model that predicts the loss increase as a sum of quantization noise MSE weighted by per-layer curvature coefficients. During training, we periodically re-compute these coefficients using perturbations across layers, and re-assign precision. Pretraining and supervised fine-tuning experiments on LLMs of up to 4B parameters show that Q-PACE consistently improves over existing mixed-precision training recipes, and achieves comparable loss at substan
    
[^266]: LayerRoPE：动态深度方向的幅度与角度叠加

    LayerRoPE: Dynamic Depth-wise Magnitude & Angular Superposition

    [https://arxiv.org/abs/2610.09179](https://arxiv.org/abs/2610.09179)

    该论文发现Transformer隐藏状态范数随深度的增长并非病态，而是一种由归一化权重γ承载的涌现式深度位置编码，并据此提出LayerRoPE，用共享向量与深度条件标量替代逐层γ向量，在减少参数的同时保持性能。

    

    随着数据在Transformer中逐层传播，其隐藏状态的范数会随深度增长数个数量级，这一现象被称为“深度诅咒”，几乎被普遍视为需要抑制的病态现象。我们持相反的观点。在来自9个模型家族的16个预训练大语言模型上——涵盖稠密架构、专家混合架构和混合架构，以及Pre-Norm、Peri-Norm和Post-Norm设计——我们发现这种增长实际上反映了一种涌现式的深度位置编码，它由残差流上唯一可学习的逐层增益——归一化权重γ所承载：随着深度增加，γ在幅度上增长、在方向上旋转，共同编码了层索引。我们通过LayerRoPE将这种深度条件编码显式化，它是RoPE沿深度轴方向的隐式类似物，用一个单一共享向量加上深度条件标量来替换所有逐层的γ向量，从而实现参数量的净减少，且FLOPs变化小于0.02%。

    arXiv:2610.09179v1 Announce Type: cross  Abstract: As data propagates through a Transformer, the norm of its hidden states grows by orders of magnitude with depth, a phenomenon framed as 'curse of depth' and nearly universally treated as a pathology to be suppressed. We take the opposite view. Across 16 pre-trained LLMs from 9 families, spanning dense, mixture-of-experts and hybrid architectures and Pre-, Peri- and Post-Norm designs, we find that this growth reflects an emergent depth-positional encoding, carried by the only learned per-layer gain on the residual stream, the normalization weight $\gamma$: with depth, $\gamma$ grows in magnitude and rotates in direction, jointly encoding the layer index. We make this depth-conditioned encoding explicit with LayerRoPE, an implicit analog of RoPE along the depth axis, which replaces all layerwise $\gamma$ vectors with a single shared vector and depth-conditioned scalars, at a net reduction in parameters and $<0.02\%$ change in FLOPs. Acro
    
[^267]: 面向车辆轨迹预测的上下文感知注意力高斯混合模型

    Context-aware Attention-based Gaussian Mixture Models for Vehicular Trajectory Prediction

    [https://arxiv.org/abs/2610.09174](https://arxiv.org/abs/2610.09174)

    本文提出CAA-GMM模型，通过上下文感知的注意力机制与可解释的高斯混合建模，实现多模态且不确定性感知的车辆轨迹预测，在nuScenes和Argoverse 2数据集上以较低的计算复杂度达到了与最先进方法相当或更优的精度。

    

    可靠且可解释的轨迹预测对于复杂和不确定环境下的协同驾驶与自动驾驶至关重要。本文提出了一种上下文感知的注意力高斯混合模型（CAA-GMM），用于多模态、不确定性感知的运动预测。该方法将未来运动建模为以场景上下文和智能体动力学为条件的概率混合分布，通过可解释的高斯分量捕捉多样化的行为模式。一种轻量级注意力机制自适应地编码智能体间交互和上下文显著性，从而在密集交通场景中高效融合栅格化环境线索与运动历史信息。在nuScenes和Argoverse 2数据集上的全面评估表明，CAA-GMM相比最先进的基于栅格的基线方法取得了相当或更优的精度，同时保持了较低的计算复杂度。消融分析证实了……（原文摘要在此处截断）

    arXiv:2610.09174v1 Announce Type: new  Abstract: Reliable and interpretable trajectory prediction is critical for cooperative and autonomous driving in complex and uncertain environments. This paper introduces a Context-Aware Attention-based Gaussian Mixture Model (CAA-GMM) for multimodal, uncertainty-aware motion forecasting. The proposed approach models future motion as a probabilistic mixture conditioned on both scene context and agent dynamics, capturing diverse behavioral modes with interpretable Gaussian components. A lightweight attention mechanism adaptively encodes inter-agent interactions and contextual salience, enabling efficient fusion of rasterized environment cues and motion history in dense traffic scenes. Comprehensive evaluations on the nuScenes and Argoverse 2 datasets demonstrate that CAA-GMM achieves competitive or superior accuracy compared with state-of-the-art raster-based baselines, while maintaining low computational complexity. Ablation analyses confirm the i
    
[^268]: 并行扩散采样的下界

    Lower Bounds for Parallel Diffusion Sampling

    [https://arxiv.org/abs/2610.09166](https://arxiv.org/abs/2610.09166)

    本文首次建立了带近似分数的扩散采样的多项式并行轮数下界，证明了对 $R^d$ 中平滑近各向同性高斯混合的采样需要 $\widetilde{\Omega}(d^{1/3})$ 轮、对单位球内各向异性轴对齐盒子的均匀采样需要 $\Omega(d)$ 轮，且这些下界对每轮可进行多项式次查询的任意随机算法均成立。

    

    标准扩散采样器通过对学习到的分数函数进行反复求值来生成样本。并行采样方法试图通过以额外的求值为代价换取更少的顺序轮次，从而加速生成过程。这就引出了一个自然的问题：即使可以同时进行大量的分数查询，顺序依赖性仍然有多大的不可避免性？我们建立了使用近似分数的扩散采样的首个多项式并行轮数下界。具体而言，我们证明了：(1) 对 $R^d$ 中平滑、近各向同性高斯混合进行采样的 $\widetilde{\Omega}(d^{1/3})$ 轮下界；(2) 对包含在单位球内的各向异性轴对齐盒子进行均匀采样的 $\Omega(d)$ 轮下界。这两个下界对任意随机算法均成立，即这些算法每轮可以在任意位置和任意噪声水平上进行多项式次查询，且分数误差为逆多项式级别、总变差精度为常数级别。

    arXiv:2610.09166v1 Announce Type: cross  Abstract: Standard diffusion samplers generate samples through repeated evaluations of a learned score function. Parallel sampling methods seek to accelerate generation by trading additional evaluations for fewer sequential rounds. This raises the question of how much sequential dependence is unavoidable, even when many score queries can be made simultaneously.   We establish the first polynomial parallel-round lower bounds for diffusion sampling with approximate scores. Specifically, we prove (1) a $\widetilde{\Omega}(d^{1/3})$-round lower bound for sampling smooth, near-isotropic Gaussian mixtures in $R^d$, and (2) an $\Omega(d)$-round lower bound for uniform sampling from anisotropic axis-aligned boxes contained in the unit ball. Both bounds hold for arbitrary randomized algorithms making polynomially many queries per round at arbitrary locations and noise levels, with inverse-polynomial score error and constant total variation accuracy. The 
    
[^269]: 基于和乐变量与角点重加权，从独立方格出发采样二维晶格上的 SU(N) 规范理论

    Sampling SU(N) gauge theory on a 2D lattice from independent plaquettes via holonomies and corner reweighting

    [https://arxiv.org/abs/2610.09147](https://arxiv.org/abs/2610.09147)

    该论文提出用和乐变量结合角点重加权，把二维 SU(N) 晶格规范理论的采样问题归结为在给定群交换子约束下从独立方格出发的条件采样，从而绕开了从方格映射回链变量这一生成式采样方法的难点。

    

    在晶格规范理论的 Wilson 形式中，基本自由度是群值的链变量，而作用量是对方格的迹求和，方格是最小的 Wilson 圈。对于归一化流等生成式采样方法而言，这带来了一个挑战：单个方格的分布容易建模，但如何把采样得到的方格映射回链变量才是真正的障碍。我们在二维情形下，为采用 Wilson 方格作用量的 SU(N) 规范理论探索了一条统计上的解决途径。该作用量可以用和乐变量来表示，一旦在扩展晶格的四个角点处满足一致性条件，这些和乐变量便能确定链变量。利用一种只影响和乐的边界条件，我们把这一致性条件转化为一个条件采样问题，归结为在给定群交换子 Z=XYX†Y† 的条件下采样 X,Y∈SU(N)。

    arXiv:2610.09147v1 Announce Type: cross  Abstract: In the Wilson formulation of lattice gauge theory, the fundamental degrees of freedom are group-valued link variables, while the action is a sum over the trace of the plaquettes, the smallest Wilson loops. For generative sampling methods such as normalizing flows, this poses a challenge: the distribution of an individual plaquette is easy to model, but mapping sampled plaquettes to the links is the obstruction. We explore a statistical way around this in two dimensions for the $\mathrm{SU}(N)$ gauge theory with the Wilson plaquette action. The action can be written in terms of holonomy variables, which determine the links once a consistency condition at the four corners of an extended lattice is satisfied. Using a boundary condition that only affects the holonomies, we trade this consistency condition for a conditional sampling problem, which reduces to sampling $X,Y\in\mathrm{SU}(N)$ given the group commutator $Z=XYX^\dagger Y^\dagger
    
[^270]: 为你的提示加噪：连续扩散语言模型中对条件令牌添加噪声

    Noise Your Prompt: Noising Conditioning Tokens in Continuous Diffusion Language Models

    [https://arxiv.org/abs/2610.09145](https://arxiv.org/abs/2610.09145)

    在连续扩散语言模型的训练中对条件提示令牌同样添加噪声这一单行修改，即可显著提升模型在数独等组合推理任务上的泛化能力与生成解的多样性，但其收益并不适用于所有自然语言任务。

    

    我们重新审视了连续扩散语言模型文献中的一个公认标准做法，即在训练期间保持条件提示令牌为干净（无噪声）状态。我们做了一个非常简单的修改：在训练期间也对条件提示令牌添加噪声。我们证明，在这一修改后的训练目标下，模型在数独和N皇后等组合推理任务中获得了更好的泛化能力，且在更难的变体上收益最大（数独困难版的解决率从3.73%提升至24.65%），同时生成解的多样性也有所提高（10x10 N皇后问题的覆盖率从50.60%提升至73.79%）。我们还展示了在使用Gigaword摘要数据集的中等数据规模下，自然语言生成质量有可衡量的提升，但值得注意的是，这些收益并不能迁移到所有自然语言任务上（例如开放式对话生成）。我们的方法只需对训练目标进行单行修改，无需额外的……

    arXiv:2610.09145v1 Announce Type: cross  Abstract: We revisit a standard accepted practice in the continuous diffusion language model   literature of fixing conditioning prompt tokens clean during training.   We make a very simple modification: also noise the conditioning prompt tokens during training.   We demonstrate that under this modified training objective, we achieve better generalization   in combinatorial reasoning tasks such as Sudoku and N-Queens, with the largest gains on harder variants   ($3.73\% \to 24.65\%$ solve rate on Sudoku Hard), and increased diversity of generated solutions ($50.60\% \to 73.79\%$ coverage on   10x10 N-Queens). We also show measurable improvements to natural language generation quality   in modest dataset regimes with Gigaword summarization, but notably demonstrate that gains do not   transfer to all natural language tasks (e.g open ended dialogue generation).   Our method is a single line change to the training objective, requires no additional i
    
[^271]: 一种用于检测亲和诈骗与浪漫投资诈骗的认知感知量子机器学习-经典强化学习框架

    A Cognitive-Aware QML-CRL Framework for Detecting Affinity and Romance-Investment Fraud

    [https://arxiv.org/abs/2610.09141](https://arxiv.org/abs/2610.09141)

    该论文提出一种量子机器学习与经典强化学习混合框架，通过专用量子比特编码诈骗中的认知偏差、操纵性重构的时间顺序和叙事共现信息，并将逐轮对话标记决策建模为最优停止问题，从而实现对亲和诈骗与浪漫投资诈骗的检测。

    

    我们提出了一种量子-经典混合框架，通过建模操纵性对话中的认知偏差来检测亲和诈骗与浪漫投资诈骗。在我们提出的框架中，这类诈骗核心的认知偏差由结构化参数化量子电路中的专用量子比特承载，同时引入一个框架量子比特使编码对操纵性重构的时间顺序敏感，并引入一个叙事量子比特通过可训练的纠缠层聚合共现信息。电路参数与一个经典强化学习智能体联合训练，该智能体逐轮决定是否标记对话，该过程被建模为最优停止问题。我们在包含困难负样本（合法但紧急、合法但强势推销的对话）的合成对话上评估了模型的性能。

    arXiv:2610.09141v1 Announce Type: new  Abstract: We present a hybrid quantum-classical framework that detects affinity and romance-investment fraud by modelling the cognitive biases in a manipulative conversation. In our proposed framework, cognitive biases central to this fraud class are carried by dedicated qubits in a structured parameterized quantum circuit, together with a frame qubit makes the encoding sensitive to the temporal order of manipulative reframing, and a narrative qubit that aggregates co-occurrence through a trainable entanglement layer. The circuit parameters are trained jointly with a classical reinforcement-learning agent that decides, turn by turn, whether to flag the conversation, modeled as an optimal stopping problem. We evaluate the model's performance on synthetic conversations that include hard negatives, legitimate but urgent, and legitimate but pushy sales conversations.
    
[^272]: 面向精确DAG学习的方向性证据引导搜索空间缩减方法

    Directional Evidence Guided Search-Space Reduction for Exact DAG Learning

    [https://arxiv.org/abs/2610.09136](https://arxiv.org/abs/2610.09136)

    提出了一种名为DECO的非参数混合框架，通过从观测数据中提取依赖性和方向性证据预先构建可采纳父节点集，从而在精确DAG学习中获得搜索空间的指数级缩减。

    

    从观测数据中学习有向无环图（DAG）是一个具有挑战性的组合优化问题，原因在于候选父节点集配置的数量呈指数级增长。现有的基于评分的精确方法通常需要计算量巨大的组合搜索，而基于约束的方法随着图规模和条件集复杂性的增加，可能变得不可靠或计算开销过高。我们开发了一种非参数混合框架，称为DECO（方向性证据引导的配置优化，Directional Evidence-guided Configuration Optimization），它从观测数据中提取依赖性证据和方向性证据，以便在精确优化之前构建可采纳的父节点集。该方法通过剔除经验上缺乏支持的父节点配置来缩减优化搜索空间，同时为所有合理的边方向保留灵活性。理论分析表明，可采纳父节点集配置实现了指数级的缩减。

    arXiv:2610.09136v1 Announce Type: new  Abstract: Learning a directed acyclic graph (DAG) from observational data is a challenging combinatorial problem due to the exponential growth in the number of candidate parent-set configurations. Existing exact score-based methods often require computationally intensive combinatorial search, whereas constraint-based methods can become unreliable or computationally demanding as graph size and conditioning-set complexity increase. We develop a non-parametric hybrid framework, referred to as DECO (Directional Evidence-guided Configuration Optimization), that extracts dependency and directional evidence from observation data to construct admissible parent sets prior to exact optimization. It reduces the optimization search space by eliminating empirically unsupported parent configurations while preserving flexibility for all plausible edge orientations. Theoretical analysis establishes an exponential reduction in the admissible parent-set configurati
    
[^273]: 似然温度调节对变分贝叶斯线性神经网络极限预测矩的影响

    The Impact of Likelihood Tempering on the Limiting Predictive Moments of Variational Bayesian Linear Neural Networks

    [https://arxiv.org/abs/2610.09132](https://arxiv.org/abs/2610.09132)

    本文针对宽贝叶斯神经网络中的“先验主导”退化问题，推导了在温度调度T = τ/M^c下变分贝叶斯线性神经网络的极限预测分布，并揭示了似然温度调节与NNGP后验之间的关系。

    

    在宽贝叶斯神经网络中，高斯平均场变分推断容易出现“先验主导”问题：证据下界（ELBO）的Kullback-Leibler（KL）正则化项压过期望对数似然项，随着网络宽度M的增长，变分预测分布坍缩为先验预测分布。通过将似然提升至1/T次幂（其中温度T < 1）来对似然进行温度调节，等价于将KL项缩放T倍。本文探讨T必须以多快的速度随M减小才能对抗这种退化现象，并在两项之间取得良好平衡。对于具有各向同性高斯先验的单隐层线性网络，我们推导了在形如T = τ/M^c（其中常数τ, c > 0）的调度方案下，当M → ∞时的极限预测分布，并将其与未温度调节的神经网络高斯过程（NNGP）后验——即精确后验的无限宽度极限——进行比较。我们的主要结果是……

    arXiv:2610.09132v1 Announce Type: cross  Abstract: In wide Bayesian neural networks, Gaussian mean-field variational inference is prone to "prior dominance": the Kullback-Leibler (KL) regularization term of the ELBO outweighs the expected log-likelihood, and the variational predictive distribution collapses to the prior predictive as the width $M$ grows. Tempering the likelihood, by raising it to the power $1/T$ for a temperature $T < 1$, is equivalent to scaling the KL term by $T$. We ask in this paper how fast $T$ must decrease with $M$ to counteract this degeneracy and strike a good balance between the two terms. For single-hidden-layer linear networks with isotropic Gaussian priors, we derive the limiting predictive distribution under schedules of the form $T = \tau/M^{c}$, with constants $\tau, c > 0$, as $M \to \infty$ and compare it with the untempered neural network Gaussian process (NNGP) posterior, the infinite-width limit of the exact posterior. Our main result is that the p
    
[^274]: CADFather：通过协同工具调用实现自主CAD重建

    CADFather: Autonomous CAD Reconstruction through Coordinated Tool Use

    [https://arxiv.org/abs/2610.09127](https://arxiv.org/abs/2610.09127)

    CADFather是一个自主智能体系统，通过视觉语言助手协调学习型/算法型提案工具与数值优化等多种互补工具，从3D网格中重建出可编辑的参数化CAD程序。

    

    从3D形状重建可编辑的CAD模型仍然是一项具有挑战性的工程任务。现有方法可以提出CAD操作，但没有任何单一的提案来源能够在不同的零件几何形状和重建阶段都同样有效。我们提出了CADFather，这是一个自主的智能体系统，通过协调互补的工具从3D网格中恢复参数化的CAD程序。一个视觉语言助手会检查目标形状和中间重建结果的渲染图，然后决定扩展哪些候选CAD程序、调用哪些工具、生成多少个提案以及何时结束。学习型和算法型工具负责提出CAD操作，而数值优化则用于细化现有程序的参数。新提出或细化后的程序会被执行和评估，从而为后续决策提供反馈。该智能体为每个目标零件维护多个备选的候选程序，并保留最佳的有效重建结果。

    arXiv:2610.09127v1 Announce Type: cross  Abstract: Reconstructing an editable CAD model from a 3D shape remains a challenging engineering task. Existing methods can propose CAD operations, but no single source of proposals works equally well across different part geometries and stages of reconstruction. We introduce CADFather, an autonomous agentic system that coordinates complementary tools to recover parametric CAD programs from 3D meshes. A vision-language assistant inspects renders of the target and intermediate reconstructions, then decides which candidate CAD programs to extend, which tools to invoke, how many proposals to generate, and when to finish. Learned and algorithmic tools propose CAD operations, while numerical optimization refines the parameters of existing programs. Proposed or refined programs are executed and evaluated to provide feedback for subsequent decisions. The agent maintains alternative candidate programs for each target part and preserves the best valid re
    
[^275]: 基于条件流匹配的领域知识驱动自适应采样方法：实现金属增材制造中可泛化的物理信息神经网络（PINNs）

    Domain-informed Adaptive Sampling for Generalizable PINNs in Metal Additive Manufacturing via Conditional Flow Matching

    [https://arxiv.org/abs/2610.09126](https://arxiv.org/abs/2610.09126)

    本论文通过经验风险最小化理论证明工艺条件感知的自适应采样严格优于传统静态采样，并提出基于条件流匹配的两阶段自适应采样框架，显著提升了物理信息神经网络在金属增材制造热建模中的泛化能力。

    

    摘要（arXiv:2610.09126v1，公告类型：新论文）：在金属增材制造（AM）中，精确的热建模对于理解“工艺-结构-性能”链条至关重要。物理信息神经网络（PINNs）通过在配点处最小化基于物理的残差损失，提供了一种有效的替代性热建模方法。然而，先前的研究通常依赖于手工设计的静态配点采样策略，这些策略既缺乏理论依据，也无法在不同工艺条件间进行扩展，从而阻碍了其泛化能力。在本工作中，我们通过经验风险最小化提供了理论分析，证明了对工艺条件敏感的自适应采样在泛化性能上严格优于传统的静态采样。基于这一洞察，我们提出了一个两阶段框架下的自适应采样策略：（1）利用条件流匹配（conditional Flow Matching）模型学习不同工艺条件下的近似高残差分布，（2）一种混合（摘要在此处截断）

    arXiv:2610.09126v1 Announce Type: new  Abstract: Accurate thermal modeling is essential in metal additive manufacturing (AM) for understanding the process-structure-property chain. Physics-informed neural networks (PINNs) offer effective surrogate thermal modeling by minimizing physics-based residual losses at collocation points. However, prior works typically rely on manually-crafted, static collocation sampling strategies, which are neither principled nor scalable across process conditions, hindering their generalization capability. In this work, we provide theoretical analysis through empirical risk minimization, showing that process condition-aware adaptive sampling is strictly more favorable than conventional static sampling for generalization. Building on this insight, we propose an adaptive sampling strategy within a two-stage framework: (1) a conditional Flow Matching model that learns approximate high-residual distributions across different process conditions, and (2) a mixed 
    
[^276]: 空间归纳头：多维元胞自动机的上下文学习

    Spatial Induction Heads: In-Context Learning of Multidimensional Cellular Automata

    [https://arxiv.org/abs/2610.09124](https://arxiv.org/abs/2610.09124)

    该论文提出了“空间归纳头”这一两层“收集-匹配”电路机制，解释了transformer如何在缺乏显式坐标归纳偏置的多维元胞自动机数据中重建空间邻域并完成上下文学习。

    

    归纳头为序列数据中的上下文学习提供了机制层面的解释，但现有理论大多假设与预测相关的上下文形成一个连续的片段。在多维数据中，序列化操作打破了这一假设，因为它将空间上相邻的元素分散到token序列中相距遥远的位置。我们研究了transformer如何在多维随机及确定性元胞自动机中克服这一路由问题，其中每条轨迹由未知的局部规则生成，并以扁平化序列的形式呈现，不含基于坐标的显式空间归纳偏置。我们提出了空间归纳头，这是一种两层的“收集-匹配”电路：第一层重建相关的空间邻域，第二层将所得的构型与此前出现过的构型进行匹配。我们给出了该收集操作的两种显式实现，并证明了进行空间……（所需的位置维度）……（摘要原文在此处被截断）

    arXiv:2610.09124v1 Announce Type: new  Abstract: Induction heads provide a mechanistic account of in-context learning in sequential data, but existing theory largely assumes that the context relevant to a prediction forms a contiguous block. In multidimensional data, serialization breaks this assumption by scattering spatial neighbors across distant positions in the token sequence. We study how transformers overcome this routing problem in multidimensional stochastic and deterministic cellular automata, where each trajectory is generated by an unknown local rule and presented as a flattened sequence without an explicit coordinate-based spatial inductive bias. We introduce spatial induction heads, two-layer gather-and-match circuits in which the first layer reconstructs the relevant spatial neighborhood and the second matches the resulting configuration against earlier occurrences. We give two explicit realizations of the gather and show that the positional dimension required for spatia
    
[^277]: 参数高效微调方法真的不同吗？

    Are Parameter-Efficient Fine-tuning Methods Really Different?

    [https://arxiv.org/abs/2610.09122](https://arxiv.org/abs/2610.09122)

    本研究比较了语言模型和扩散模型中的六种参数高效微调方法，发现LoRA系列方法本身已近似保持预训练权重几何结构，从而质疑了显式几何保持的必要性，并揭示了不同方法在适应性与遗忘之间的权衡差异。

    

    摘要：参数高效微调（PEFT）提供了多种参数化方式，但它们在方法学和功能上的差异仍不清晰。我们在语言模型和扩散模型中比较了六种方法，以考察它们的参数化方式如何与任务性能、遗忘以及预训练权重几何结构的变化相关联。受正交微调（OFT）谱保持设计的启发，我们首先探究了谱保持本身是否对适应性和知识保留很重要。我们发现所选的LoRA系列方法本身也近似保持了预训练几何结构，并且恢复其轻微漂移的奇异值谱在很大程度上能够保留任务性能，这对显式几何保持的必要性提出了质疑。除此之外，我们观察到某些方法表现出明显的适应—保留权衡，且这种权衡因不同设置而异：LoRA在保持竞争性性能的同时最稳定地限制了遗忘，DoRA则实现了更高的……（摘要截断）

    arXiv:2610.09122v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning (PEFT) offers many parameterizations, yet their methodological and functional differences remain unclear. We compare six methods in language and diffusion models to examine how their parameterizations relate to task performance, forgetting, and changes in pretrained weight geometry. Motivated by the spectrum-preserving design of orthogonal fine-tuning (OFT), we first ask whether spectral preservation is itself important for adaptation and retention. We find that the selected LoRA-family methods also approximately preserve pretrained geometry, and that restoring their slightly drifted singular-value spectra largely preserves task performance, questioning the necessity of explicit geometric preservation. Beyond this, we observe that some methods exhibit distinct adaptation--retention trade-offs that vary across settings: LoRA most consistently limits forgetting at competitive performance, DoRA achieves highe
    
[^278]: 欺骗性老虎机问题：探索耦合与多智能体学习的脆弱性

    The Deceptive Bandit Problem: Exploratory Coupling and the Fragility of Multi-Agent Learning

    [https://arxiv.org/abs/2610.09120](https://arxiv.org/abs/2610.09120)

    该论文揭示了多智能体学习中随机探索的独立性和隐私性是安全关键属性：欺骗者可以通过将自身探索动作与泄露的探索信号耦合，将受害者的学习动态引导至欺骗性纳什均衡，且收敛速度仍保持最优。

    

    随机化探索是老虎机学习、多智能体强化学习和零阶策略搜索的核心，然而其独立性和隐私性通常仅被视为技术性假设。我们证明这些性质对安全而言至关重要，并展示了一个对抗性智能体如何利用关于另一智能体探索的特权信息。我们在最小的双人强单调设置中分析了一对“欺骗者-受害者”，其中欺骗性玩家获得与受害者的探索仅存在相关性的泄露信号。我们证明，通过将自己的探索动作与该信息耦合，欺骗性玩家注入了一种外部性，将学习动态引导至一个新的稳态，称为欺骗性纳什均衡（DNE）。我们证明欺骗性老虎机学习（DBL）动态收敛到DNE的任意小邻域，同时保持最优收敛速率。

    arXiv:2610.09120v1 Announce Type: new  Abstract: Randomized exploration is central to bandit learning, multi-agent reinforcement learning, and zeroth-order policy search, yet its independence and privacy are usually only treated as technical assumptions. We show that these properties are critical for security purposes and demonstrate how an adversarial agent can exploit privileged information on another agent's exploration. We analyze a deceiver-victim pair in the minimal two-player strongly monotone setting, where a deceptive player obtains leaked signals that are merely correlated with the victim's exploration. We show that, by coupling their own exploratory action with this information, the deceptive player injects an externality that steers the learning dynamics to a new steady state, called the deceptive Nash equilibrium (DNE). We prove that the deceptive bandit learning (DBL) dynamics converge to an arbitrarily small neighborhood of the DNE while retaining optimal convergence rat
    
[^279]: Workhorse：从人类数据学习鲁棒的人形机器人全身移动操作

    Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data

    [https://arxiv.org/abs/2610.09117](https://arxiv.org/abs/2610.09117)

    提出Workhorse框架，通过在同一套无需重定向的真人演示数据上分别训练视觉规划器和强化学习全身跟踪器，并以互相模仿对方部署误差的方式增强训练数据，使Unitree G1人形机器人能从第一人称视觉实现鲁棒的全身移动操作并具备抗干扰恢复能力。

    

    人形机器人在仅依靠第一人称视角RGB图像和本体感觉来规划接触丰富的全身操作方面仍然面临困难。Workhorse从无需机器人参与的真人演示数据中学习此类操作。视觉规划器预测五个连杆目标：躯干、双手腕和双脚的姿态。强化学习全身跟踪器控制机器人跟随这些目标。两个策略在同一录制的真人姿态数据上分别训练，无需重定向。我们通过数据增强模仿对方策略在部署时可能产生的误差来增强每个策略的训练数据。在真实的Unitree G1机器人上，Workhorse能够用手和踢腿动作分拣箱子、接住抛来的箱子、推倒并攀爬行李箱。在箱子分拣过程中，我们展示了在人推动机器人或拿走箱子后系统的恢复能力。在演示房间的仿真环境中，系统在77%的回合中成功完成箱子分拣任务，在40 N·s推力干扰下成功率为64%。

    arXiv:2610.09117v1 Announce Type: cross  Abstract: Humanoid robots still struggle to plan contact-rich whole-body manipulation from egocentric RGB and proprioception. Workhorse learns such manipulation from robot-free human demonstrations. A visual planner predicts five-link targets: the poses of the torso, both wrists, and both feet. A reinforcement-learning whole-body tracker follows them on the robot. Both policies train separately on the same recorded human poses, without retargeting. We augment the training data of each policy to imitate the errors that the other makes at deployment. On a real Unitree G1, Workhorse sorts boxes with its hands and a kick, catches a thrown box, and topples and climbs a suitcase. During box sorting, we show recoveries after a person pushes the robot or takes the box away. In a simulated copy of the demonstration room, the system completes box sorting in 77% of episodes, and in 64% under 40 N.s pushes. With both policies retrained from the same demonst
    
[^280]: 从不确定性到行动：学习引导LLM智能体

    From Uncertainty to Action: Learning to Steer LLM Agents

    [https://arxiv.org/abs/2610.09115](https://arxiv.org/abs/2610.09115)

    提出VoS（引导价值）方法，通过构建包含约82,000个反事实延续的逐步结果表（SOT）来学习每一步引导的价值，结合危害预算触发器，实现对LLM智能体更精准的纠正时机与方式决策，克服了不确定性信号无法定位最佳引导步骤的局限。

    

    引导LLM智能体意味着决定是否纠正它、在哪个步骤纠正、以及使用哪种机制。不确定性常被用来决定何时纠正智能体，但它是否能指导这些决策仍不清楚。我们在每个非终止步骤分别使用四种机制对智能体轨迹进行引导，并将每个延续运行至完成。由此产生的逐步结果表（SOT）包含来自三个基准测试和两个智能体的1,864条轨迹的大约82,000个反事实延续。结果表明，不确定性可以识别失败的轨迹，但没有任何单一信号能够可靠地定位引导有帮助的步骤。因此，我们提出了VoS（引导价值），一个可离线或在线运行的轨迹级监控器，它从SOT中学习每一步引导的价值，并据此决定在哪里进行引导。一个带有危害预算的触发器决定是否进行引导，限制了VoS干扰成功轨迹的比例。

    arXiv:2610.09115v1 Announce Type: cross  Abstract: Steering an LLM agent means deciding whether to correct it, at which step, and with which mechanism. Uncertainty is often used to decide when to correct an agent, but whether it can guide these decisions remains unclear. We steer agent trajectories separately at every non-terminal step with each of four mechanisms and run each continuation to completion. The resulting stepwise outcome table (SOT) holds about 82,000 counterfactual continuations of 1,864 trajectories from three benchmarks and two agents. It shows that uncertainty can identify failing trajectories, but that no single signal reliably locates the step at which steering helps. We therefore propose VoS (Value of Steering), a trajectory-level monitor, offline or online, that learns from SOT the value of steering at each step and decides where to steer by it. A harm-budgeted trigger decides whether to steer, limiting the fraction of successful trajectories that VoS disturbs. Vo
    
[^281]: 相同文本，不同预测：文本分类器中的服务上下文非确定性

    Same Text, Different Prediction: Serving-Context Nondeterminism in Text Classifiers

    [https://arxiv.org/abs/2610.09111](https://arxiv.org/abs/2610.09111)

    该论文首次系统研究了文本分类器中的服务上下文非确定性，通过训练180个涵盖判别式、伪生成式和完全生成式的分类器模型，发现即使输入文本、模型参数和采样随机性固定，批大小、批组成、硬件和推理引擎等部署环境因素仍会导致分类预测结果发生改变。

    

    arXiv:2610.09111v1 公告类型：新 摘要：确定性推理对于可靠、可信的机器学习至关重要。以往关于文本生成的研究表明，即使提示、模型参数和采样随机性都保持固定，改变批大小、批组成、硬件或推理引擎等因素也会改变生成的文本。这些差异部分被归因于浮点运算的非结合性、依赖张量形状的核函数选择以及其他实现层面的数值执行差异。然而，这些因素是否、何时以及在多大程度上影响文本分类任务仍不清楚。我们对文本分类器中的服务上下文非不变性进行了系统性研究，而此前的工作仅通过生成的文本来衡量这一现象。我们训练了180个模型，涵盖判别式、伪生成式和完全生成式三种分类器形式，并在四类服务上下文中对每个模型进行评估，同时保持检查点……（摘要在此处截断）

    arXiv:2610.09111v1 Announce Type: new  Abstract: Deterministic inference is essential for reliable and trustworthy machine learning. Prior studies of text generation have shown that changing factors such as batch size, batch composition, hardware, or inference engine can alter the generated text, even when the prompt, model parameters, and sampling randomness are fixed. These differences have been attributed in part to floating-point non-associativity, shape-dependent kernel selection, and other implementation-level differences in numerical execution. However, it remains unclear whether, when, and to what extent the same factors affect text classification. We present a systematic study of serving-context non-invariance in text classifiers, which prior work has measured only through generated text. We train 180 models spanning discriminative, pseudo-generative, and fully generative classifier formulations and evaluate each across four categories of serving contexts, holding the checkpoi
    
[^282]: 打破微调语音识别中的对抗样本可迁移性

    Breaking Adversarial Transferability in Fine-Tuned Speech Recognition

    [https://arxiv.org/abs/2610.09109](https://arxiv.org/abs/2610.09109)

    本文揭示了在公开预训练模型上构造的对抗扰动可有效迁移到黑盒部署的微调语音识别模型，并提出统一微调框架TransferBreaker，通过基础对抗微调、潜在雅可比正则化与混合梯度自适应微调来阻断对抗迁移并提升鲁棒性。

    

    许多组织会对公开可用的预训练自动语音识别（ASR）模型进行微调，并将其部署在黑盒设置中，认为有限的访问权限即可提供保护。我们证明这一假设是脆弱的：在公开基础模型上构造的对抗扰动能够有效迁移到微调后的目标模型上，严重降低其性能，并对安全关键型应用构成隐患。我们提出了TransferBreaker，一个统一的微调框架，通过整合三项技术来抑制对抗迁移：基础对抗微调，将对抗训练限制在对基础模型有效的扰动上；潜在雅可比正则化，通过抑制对对抗敏感的方向来强制潜在空间不变性；以及HybridGrad-AFT，通过对基础模型与目标模型梯度生成的可迁移扰动进行插值，提升对自适应攻击的鲁棒性。我们对所有组件进行了理论论证，并对TransferBreaker进行了评估。

    arXiv:2610.09109v1 Announce Type: new  Abstract: Many organizations fine-tune publicly available pretrained Automatic Speech Recognition (ASR) models and deploy them in black-box settings, assuming limited access provides protection. We show this assumption is fragile: adversarial perturbations crafted on the public base model transfer effectively to fine-tuned target models, severely degrading performance and posing concerns for safety-critical applications. We propose TransferBreaker, a unified fine-tuning framework that suppresses adversarial transfer by integrating Base Adversarial Fine-Tuning, which restricts adversarial training to base-effective perturbations; Latent Jacobian Regularization, which enforces latent-space invariance by suppressing adversarially sensitive directions; and HybridGrad-AFT, which improves robustness against adaptive attacks by interpolating transferable perturbations from base and target gradients. We theoretically justify all components and evaluate Tr
    
[^283]: 凸-凹强化学习

    Convex-Concave Reinforcement Learning

    [https://arxiv.org/abs/2610.09108](https://arxiv.org/abs/2610.09108)

    该论文揭示强化学习的策略优化问题在对数密度比坐标下具有凸差规划结构，从而将 CPI、NPG、TRPO 和 AWR 统一为特例，突破了传统凸代理近似的局限。

    

    策略学习驱动着当今许多最重要且投入巨大的强化学习应用。然而其赖以建立的核心优化问题（最大化期望回报）是出了名的非凸问题，即使采用直接策略参数化也是如此，而该领域在很大程度上是通过回避这一问题来应对的：在信赖域约束下优化回报的凸代理近似（如 NPG、TRPO、PPO、AWR）。我们证明这个看似无结构的问题实际上并非没有结构。在对数密度比坐标 $y := \log[\pi/\pi_n]$ 下，通过逐决策重要性采样（PDIS）可计算的精确逐迭代目标函数是一个“凸差-凸差约束规划”。这一结构使我们能够超越代理近似：它沿可解释的轴向将 CPI、NPG、TRPO 和 AWR 恢复为特例，并开辟了一个耦合连续决策的多步轴 $k$……（摘要在此处被截断）

    arXiv:2610.09108v1 Announce Type: new  Abstract: Policy learning drives many of the most consequential and heavily-invested applications of reinforcement learning today. Yet the core optimization problem it rests on (maximizing expected return) is notoriously non-convex, even under a direct policy parameterization, and the field has largely responded by avoiding it: optimizing convex surrogate approximations of the return under trust-region constraints (NPG, TRPO, PPO, AWR). We show that this seemingly unstructured problem is not actually structureless. In log-density-ratio coordinates $y := \log[\pi/\pi_n]$, the exact per-iteration objective, computable via per-decision importance sampling (PDIS), is a difference-of-convex-constrained difference-of-convex (DC-constrained DC) program. This structure lets us move beyond surrogate approximations: it recovers CPI, NPG, TRPO, and AWR as special cases along interpretable axes, and it opens a multi-step axis $k$ that couples consecutive deci
    
[^284]: 论隐马尔可夫模型可辨识性的计算复杂度

    On the Computational Complexity of Hidden Markov Model Identification

    [https://arxiv.org/abs/2610.09104](https://arxiv.org/abs/2610.09104)

    本文从计算复杂度的角度研究隐马尔可夫模型的可辨识性问题，旨在回答是否存在可靠且完备的算法来判定给定HMM是否可辨识，以及该判定问题的计算复杂度。

    

    arXiv:2610.09104v1 公告类型：cross。摘要：辨识是指从未知采样数据中恢复真实模型参数的任务。当真实参数之外的其他参数也能诱导出相同的输出分布时，仅凭数据无法提供足够的信息来恢复真实参数，因此该模型被称为不可辨识的。我们研究隐马尔可夫模型（HMM）的可辨识性问题：给定一个HMM，它是否可辨识？现有的HMM辨识工作建立了真实HMM可以被辨识的条件。然而，这些条件大多是充分但不必要的，这意味着，当一个模型不满足这些条件时，其可辨识性仍然无法确定。我们转而从计算的角度来看待这一问题：是否存在一个可靠且完备的算法来判定给定的HMM是否可辨识？如果存在，该判定问题的复杂度是多少？我们考虑由各种……

    arXiv:2610.09104v1 Announce Type: cross  Abstract: Identification is the task of recovering the parameters of an unknown ground-truth model from sampled data. When parameters other than the ground truth induce the same output distribution, data alone does not provide enough information to recover the ground truth, and the model is thus called unidentifiable. We study the identifiability problem for hidden Markov models (HMMs): given an HMM, is it identifiable? Existing work on HMM identification establishes conditions under which the ground-truth HMM can be identified. However, most of these conditions are sufficient but not necessary, meaning that, when a model does not satisfy them, its identifiability remains inconclusive. We instead take a computational perspective: is there a sound and complete algorithm that decides whether a given HMM is identifiable, and if so, what is the complexity of this decision problem? We consider the decision problems arising from the various notions of
    
[^285]: 我们真的在对预测模型进行基准测试吗？预处理对时间序列性能的影响

    Are We Really Benchmarking Forecasting Models? The Impact of Preprocessing on Time Series Performance

    [https://arxiv.org/abs/2610.09096](https://arxiv.org/abs/2610.09096)

    该论文通过在29,000个M4时间序列上评估11个预测模型与16种可逆预处理流水线，揭示了当前基准测试忽视预处理（如差分）所造成的结构性偏差，并证明针对每个序列优化预处理可使各模型性能提升约27%至87%。

    

    尽管已有文献强调了预处理在预测精度中的关键作用，但这一环节在当前研究中基本被忽视。现代基准测试通常仅采用简单的缩放处理，未能考虑解决非平稳性所需的关键变换，例如差分。这种疏忽造成了显著的结构性预处理偏差，使内置数据处理机制的模型获得优势，同时掩盖了更简单架构的真实潜力。我们通过一个感知预处理的基准测试来研究这一效应，该测试在29,000个M4时间序列上评估了11个预测模型在16种可逆预处理流水线下的表现。我们的结果表明，预处理是预测性能的关键驱动因素。针对每个序列优化预处理，可使所有评估模型的性能提升约27%至87%，其中缺乏内置预处理机制的架构获得了最显著的改进。

    arXiv:2610.09096v1 Announce Type: new  Abstract: While established literature underscores the pivotal role of preprocessing in forecasting accuracy, this stage remains largely overlooked in current research. Modern benchmarks typically resort to simple scaling, failing to account for critical transformations required to address nonstationarity, such as differencing. This omission creates a significant structural preprocessing bias that favors models with built-in data treatments while obscuring the true potential of simpler architectures. We study this effect through a preprocessing-aware benchmark that evaluates 11 forecasting models across 16 reversible preprocessing pipelines on 29,000 M4 time series. Our results identify preprocessing as a key driver of forecasting performance. Optimizing preprocessing per series yields gains of approximately 27\% to 87\% across all evaluated models, with architectures lacking internalized preprocessing experiencing the most substantial improvement
    
[^286]: MaRK：用于状态空间模型中动态算子条件化的马尔可夫自适应循环核

    MaRK: Markov-adapted Recurrent Kernels for Dynamic Operator Conditioning in State Space Models

    [https://arxiv.org/abs/2610.09092](https://arxiv.org/abs/2610.09092)

    MaRK提出了一种动态算子条件化框架，将上下文向量直接映射为对冻结SSM循环算子参数（A、B、C、D、Δ）的有界调制，使每个扩散时间步都能动态重塑模型的输入输出记忆核。

    

    状态空间模型为序列建模提供了一种高效的Transformer替代方案，然而，为迭代生成而对预训练SSM进行条件化时，通常是在循环算子之外操作的，即通过输入注入或激活调制实现。尽管这类机制能让模型接触到条件信息，但它们使底层的时间动态保持固定。我们提出了MaRK（马尔可夫自适应循环核），这是一个动态算子条件化框架，它将上下文向量直接映射到冻结SSM的循环算子的有界调制中，涵盖循环参数A、读入参数B、读出参数C、跳跃参数D以及离散化参数Δ。通过线性参数变分SSM（LPV-SSM）系统的视角来看，MaRK诱导出一个以上下文为索引的马尔可夫参数序列族，使得每个扩散时间步都能够重塑模型的输入输出记忆核。我们在一个冻结的1.11亿参数Hydra SSM骨干网络上实例化了MaRK，并研究了三种适配器几何结构：

    arXiv:2610.09092v1 Announce Type: new  Abstract: State Space Models (SSMs) offer an efficient alternative to Transformers for sequence modeling, yet conditioning pre-trained SSMs for iterative generation typically operates outside the recurrent operator, through input injection or activation modulation. While such mechanisms expose the model to conditioning information, they leave the underlying temporal dynamics fixed. We introduce MaRK (Markov-adapted Recurrent Kernels), a dynamic operator-conditioning framework that maps context vectors directly into bounded modulations of a frozen SSM's recurrence ($A$), read-in ($B$), read-out ($C$), skip ($D$), and discretization ($\Delta$) parameters. Viewed through the lens of LPV-SSM systems, MaRK induces a context-indexed family of Markov parameter sequences, allowing each diffusion timestep to reshape the model's input-output memory kernel. We instantiate MaRK on a frozen 111M-parameter Hydra SSM backbone and study three adapter geometries: 
    
[^287]: FedRSPO+：一种异构感知的决策导向联邦学习算法

    FedRSPO+: A Heterogeneity-aware Algorithm for Decision-focused Federated Learning

    [https://arxiv.org/abs/2610.09091](https://arxiv.org/abs/2610.09091)

    提出了异构感知的决策导向联邦学习框架FedRSPO+，其核心是基于通过投影平滑决策映射的正则化代理RSPO+，为决策误差和遗憾提供理论上界，从而解决联邦场景下下游目标与可行集异构性导致的训练不稳定问题。

    

    决策导向学习（DFL）为下游优化训练预测模型，但现有方法大多假设数据是集中式的。在跨机构（cross-silo）场景中，联邦学习提供了一种自然的替代方案，然而标准联邦方法仅优化预测质量而非决策质量，并且没有处理下游目标或可行集中的异构性。这种异构性对决策导向学习尤其具有挑战性，因为多面体问题中的微小扰动可能导致最优决策发生不连续的变化，从而使客户端更新和聚合过程变得不稳定。我们提出了FedRSPO+，一个面向决策导向联邦学习的异构感知框架，它建立在RSPO+之上——一种通过投影来平滑决策映射的正则化“先预测后优化”代理方法。我们证明了RSPO+为正则化决策的决策误差和遗憾提供了上界，并且在精确正则化与一致的线性规划解选择条件下，对原始线性规划问题同样成立……

    arXiv:2610.09091v1 Announce Type: new  Abstract: Decision-focused learning (DFL) trains predictive models for downstream optimization, but existing methods largely assume centralized data. In cross-silo settings, federated learning offers a natural alternative, yet standard federated methods optimize prediction over decision quality and do not address heterogeneity in downstream objectives or feasible sets. This heterogeneity is especially challenging for DFL because small perturbations in polyhedral problems can cause discontinuous changes in optimal decisions, destabilizing client updates and aggregation. We propose FedRSPO+, a heterogeneity-aware framework for decision-focused federated learning, built on RSPO+, a regularized predict-then-optimize surrogate that smooths the decision map through projection. We show that RSPO+ upper bounds decision error and regret for the regularized decision and, under exact regularization and consistent LP solution selection, for the original LP de
    
[^288]: 用于多维序列建模的Tucker瓶颈注意力

    Tucker Bottleneck Attention for Multi-Dimensional Sequence Modeling

    [https://arxiv.org/abs/2610.09090](https://arxiv.org/abs/2610.09090)

    提出Tucker瓶颈注意力，通过将隐藏张量投影到紧凑的Tucker核心上进行注意力计算，利用低秩张量结构实现亚二次方复杂度的全局token混合，在视频预测和天气预报任务中显著降低误差和计算成本。

    

    自注意力机制的二次方计算成本限制了其对来自多维数据的长序列的可扩展性。我们提出了Tucker瓶颈注意力，该方法利用低秩张量结构实现高效的全局token混合。TuBA将隐藏张量投影为紧凑的Tucker核心，在核心上执行多头自注意力和线性投影，并将更新写回环境空间，从而实现亚二次方计算。其自回归扩展将核心内的双向交互与核心间的因果注意力相结合。在视频预测和全球天气预报任务上，TuBA相比标准注意力、高效注意力及任务特定模型取得了更优的精度-效率权衡。与标准自注意力相比，TuBA在视频预测任务中将误差和计算量分别降低了24.7%和66.6%，在自回归天气预报任务中分别降低了37.1%和85.1%，加速比最高可达4.27倍。

    arXiv:2610.09090v1 Announce Type: new  Abstract: The quadratic cost of self-attention limits scalability to long sequences from multidimensional data. We introduce Tucker bottleneck attention (TuBA), which exploits low-rank tensor structure for efficient global token mixing. TuBA projects hidden tensors into compact Tucker cores, performs multi-head self-attention and linear projections on the cores, and writes updates back to the ambient space, enabling subquadratic computation. Its autoregressive extension combines bidirectional interactions within cores with causal attention across cores. On video prediction and global weather forecasting, TuBA achieves favorable accuracy-efficiency trade-offs over standard and efficient attention and task-specific models. Compared to standard self-attention, TuBA reduces error and computation by up to 24.7% and 66.6% for video prediction and 37.1% and 85.1% for autoregressive weather forecasting, with speedups up to 4.27 times. Low-rank Tucker core
    
[^289]: U-Space：揭示语言模型中不确定性何时以及为何产生

    U-Space: Uncovering When and Why Uncertainty Arises in Language Models

    [https://arxiv.org/abs/2610.09087](https://arxiv.org/abs/2610.09087)

    该论文提出U-Space方法，旨在揭示语言模型推理过程中不确定性在何时、何处以及为何产生与演变，克服了现有标量化不确定性估计方法无法定位不确定性来源的局限性。

    

    大型语言模型正以越来越高的风险影响着各类决策。随着其错误所造成的后果不断加剧，一个核心问题变得越来越难以忽视：我们对模型给出的单个回答能有多少信任？然而，识别何时应当“推迟判断”仍然困难，因为语言模型能够以流利的解释和权威的语气呈现错误的结论。不确定性量化旨在通过估计单个预测的可靠性来解决这种脱节。然而，许多现有方法需要重复生成或单独训练的组件，且其标量化的估计值无法揭示不确定性在何处产生、以及在推理过程中如何演变。近期研究还表明，生成长度可能与不确定性估计和正确性密切相关，这引发了一个问题：估计器的预测能力中，有多少来自不确定性特有的信息，而非仅仅来自输出长度。（摘要在此处截断）

    arXiv:2610.09087v1 Announce Type: new  Abstract: Large language models are informing decisions with ever-higher stakes. As the consequences of their errors grow, a central question becomes harder to ignore: how much can we trust an individual answer? Yet recognizing when to defer remains difficult because language models can present incorrect conclusions with fluent explanations and an authoritative tone. Uncertainty quantification seeks to address this disconnect by estimating the reliability of individual predictions. However, many existing methods require repeated generations or separately trained components, and their scalar estimates do not reveal where uncertainty arises or how it evolves during reasoning. Recent work has also shown that generation length can be strongly associated with uncertainty estimates and correctness, raising the question of how much of an estimator's predictive power comes from uncertainty-specific information rather than output length alone. Mechanistic 
    
[^290]: 迈向将AI生成音乐抄袭检测视为版本识别问题的研究

    Towards AI-Generated Music Plagiarism Detection as a Version Identification Problem

    [https://arxiv.org/abs/2610.09075](https://arxiv.org/abs/2610.09075)

    该论文将AI生成音乐的抄袭检测建模为版本识别问题，构建了包含35万余个评估对的COPYCAT基准，并证明基于逐坐标嵌入偏移的监督式框架能够克服传统标量距离阈值方法在生成式再合成混淆下失效的问题，有效恢复被分散的抄袭信号。

    

    文本到音乐生成模型的迅速扩张正在挑战传统的音乐创作与知识产权范式。在这一背景下，抄袭很少是一个绝对的数学二元判定，而是在和声结构、旋律轮廓或整体感知风格特征上协商的模糊阈值。在这项工作中，我们测试了最先进的音乐版本识别架构从“人与人翻唱”领域向“人与AI抄袭”场景迁移的可行性。为评估这一任务，我们引入了COPYCAT基准数据集，该数据集源自真实世界的抄袭案例，并通过生成式再合成和数字信号处理混淆手段进行扩展，共产生350,654个评估对。我们表明，标量距离阈值方法在生成式再合成下会失效，而利用逐坐标嵌入偏移的监督式框架则能够恢复分散的抄袭信号，将整体F值显著提升。

    arXiv:2610.09075v1 Announce Type: cross  Abstract: The rapid expansion of text-to-music generative models challenges traditional paradigms of music creation and intellectual property. Plagiarism in this context is rarely an absolute mathematical binary, but an ambiguous threshold negotiated over harmonic structure, melodic contours, or overall perceived stylistic character. In this work, we test the transferability of state-of-the-art music version identification architectures from the human-to-human cover domain to the human-to-AI plagiarism setting. To evaluate this task, we introduce COPYCAT, a benchmark derived from real-world plagiarism cases and extended through generative re-synthesis and digital signal processing obfuscations, yielding 350,654 evaluation pairs. We show that scalar distance thresholding collapses under generative re-synthesis, while a supervised framework leveraging coordinate-wise embedding shifts recovers the dispersed plagiarism signal, raising overall $F_{0.
    
[^291]: TAP：基于轨迹锚定恢复的高效长程智能体剪枝

    TAP: Efficient Long-Horizon Agent Pruning via Trajectory-Anchored Recovery

    [https://arxiv.org/abs/2610.09074](https://arxiv.org/abs/2610.09074)

    提出首个面向强化学习训练智能体的结构化剪枝框架TAP，通过将结构化剪枝与锚定教师轨迹的在线策略恢复相结合，解决长程智能体任务中现有剪枝方法性能严重退化的问题。

    

    新兴的长程智能体任务需要重复调用模型，加剧了本已昂贵的语言模型的推理成本。虽然在狭窄的智能体任务中表明有可能在不损失性能的情况下进行激进的模型剪枝，但实证结果表明，现有为问答任务提出的方法在应用于智能体模型时会严重降低任务性能。我们将这种失败归因于两个决策：剪枝什么以及如何恢复。对于剪枝，一次性重要性估计无法跟踪被剪枝模型的适应过程。对于恢复，离线蒸馏仅覆盖教师轨迹的前缀，而全轨迹在线策略蒸馏会导致学生模型的错误在多轮交互中复合累积。在这项工作中，我们提出了轨迹锚定剪枝，这是首个面向强化学习（RL）训练智能体的结构化剪枝框架。TAP将结构化剪枝与高效的在线策略恢复相结合，将交互锚定到教师轨迹上。

    arXiv:2610.09074v1 Announce Type: new  Abstract: Emerging long-horizon agentic tasks require repeated model calls, worsening the inference cost of already-costly language models. While narrow agentic tasks suggest potential for aggressive model pruning without performance drop, empirical results show existing methods proposed for question answering tasks severely degrade task performance when applied to agentic models. We trace this failure to two decisions: what to prune and how to recover. For pruning, one-shot importance estimates fail to track how the pruned model adapts. For recovery, offline distillation covers only teacher prefixes, while full-trajectory on-policy distillation causes student errors to compound across turns. In this work, we propose Trajectory-Anchored Pruning (TAP), the first structural pruning framework for reinforcement learning (RL)-trained agents. TAP couples structural pruning with efficient on-policy recovery, anchoring interactions to teacher trajectories
    
[^292]: 协变量依赖的多元序数偏好联合建模及其与比较模型的联系

    Covariate-dependent Joint Modeling of Multivariate Ordinal Preferences and Its Connections with Comparison Models

    [https://arxiv.org/abs/2610.09070](https://arxiv.org/abs/2610.09070)

    本文提出了一种协变量依赖的多元序数偏好联合建模方法，避免了传统方法单独处理各属性或将数据粗化为胜负比较所造成的信息损失，并建立了该方法与 Bradley–Terry、Plackett–Luce 等比较模型之间的联系。

    

    arXiv:2610.09070v1 公告类型：交叉 摘要：带有协变量的多元序数数据在从语言模型与人类偏好对齐到推荐系统等各类问题中被广泛收集。例如，像 MovieLens 这样的数据集包含人类用户对多部电影以1至5量表给出的评分，以及用户的人口统计信息（如年龄或性别）。类似地，像 HelpSteer 这样的数据集收集人类对大语言模型回复的多个属性（如“有用性”或“冗长性”）在序数量表上的反馈，其协变量取决于大语言模型的提示—响应对。遗憾的是，这些数据的标准建模方法（a）对各个属性单独建模而非联合建模，并且（b）经常将数据转换为成对或列表式的胜负比较，以拟合诸如 Bradley–Terry 和 Plackett–Luce 等模型。这两种做法都会导致对实际观测数据的信息粗化，我们通过一种联合的协变量依赖的建模方法来解决这一问题……

    arXiv:2610.09070v1 Announce Type: cross  Abstract: Multivariate ordinal data along with covariates are commonly collected in problems ranging from alignment of language models with human preferences, as well as in recommender systems. For example, data sets such as MovieLens contain several movies rated on a scale 1--5 by human users, along with their demographic information such as age or gender. Similarly, data sets such as HelpSteer collect human feedback on several attributes such as "helpfulness" or "verbosity" of LLM response on an ordinal scale, with covariates depending on the LLM prompt--response pairs. Unfortunately, the standard approaches for modeling these data (a) look at the attributes individually rather than jointly, and (b) often convert the data into pairwise or list-wise win--loss comparisons for fitting models such as Bradley--Terry and Plackett--Luce. Both of these lead to a coarsening of what is actually observed, which we address via a joint covariate-dependent 
    
[^293]: 与语言模型交谈

    Talking with Language Models

    [https://arxiv.org/abs/2610.09064](https://arxiv.org/abs/2610.09064)

    该论文提出“人工制品立场”框架，主张大语言模型只是精密的文本生成器而非真正的说话者，人机“对话”实为用户在界面幻象下进行的独角戏，从而消解了关于AI对话者身份、谎言与承诺等哲学难题。

    

    当我们与大语言模型（LLM）互动时，我们是在进行一场对话吗？它们被设计成邀请我们将其视为会记忆、会行动、会做出承诺的智能对话者。但表象具有欺骗性。我们提出了“人工制品立场”，这是一个将人机交互重新构想为以人工制品为媒介的候选文本交换的框架。LLM的输出是为实用性而优化的候选文本，而非承载意义或言语效力的言说。LLM是精密的文本生成器，而非说话者。会话之间，没有任何东西在运行；轮次之间，没有谁在记忆。持续存在的只是一份配置和一份记录。所谓的“对话”实为用户的独角戏，是一种被界面和制品设计所掩盖的解释性劳动。这一视角转变消解了近来的诸多哲学难题：关于AI交流中“我”与“你”究竟指称什么的问题、系统能否撒谎或是否应被要求兑现承诺的问题，以及我们假想对话者的身份问题……

    arXiv:2610.09064v1 Announce Type: cross  Abstract: When we interact with large language models (LLMs), are we having a conversation? They are designed to invite us to treat them as intelligent interlocutors who remember, act, and make commitments. But appearances deceive. We introduce the artifactual stance, a framework that reconceives human-AI interaction as artifact-mediated exchanges of candidate texts. LLM outputs are candidate texts optimized for utility, not utterances bearing meaning or force. LLMs are sophisticated text generators, not speakers. Between sessions, nothing runs; between turns, no one remembers. What persists is a configuration and a transcript. The "conversation" is a user's solo performance, interpretive labour disguised by interface and artifact design. This shift dissolves recent philosophical puzzles. Questions about what 'I' and 'you' refer to in AI exchanges, about whether systems can lie or be held to promises, about the identity of our supposed interlocu
    
[^294]: 基于大语言模型蒸馏的多标签主题分配：生成式与判别式学生模型的对比分析

    Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models

    [https://arxiv.org/abs/2610.09063](https://arxiv.org/abs/2610.09063)

    本文系统比较了通过大语言模型蒸馏训练的生成式与判别式小型语言模型在电商用户生成内容多标签主题分配任务上的表现，揭示了两种架构范式之间关键的数据依赖性权衡。

    

    针对用户生成内容（UGC）——包括产品评论和买卖双方对话——的多标签主题分配任务，由于非正式语言、极端的标签稀疏性以及快速演化的分类体系，在大规模电子商务场景中带来了独特的可扩展性挑战。虽然利用大语言模型（LLM）作为标注预言机来蒸馏真实标注数据已成为规避高昂人工标注成本的行业标准，但如何为由此产生的学生模型确定最优的低延迟架构仍然是一个悬而未决的挑战。为解决这一问题，我们在小型语言模型（SLM）的参数规模（1B、4B 和 8B）和架构范式（因果生成式与双向判别式）上进行了全面评估。通过将生成式文本到标签分类器与判别式基线模型（DeBERTa-V3 和 ModernBERT）进行对比，我们的分析揭示了一个关键的数据依赖性权衡：虽然判别式……

    arXiv:2610.09063v1 Announce Type: cross  Abstract: Multi-label topic assignment for user-generated content (UGC) -- including product reviews and buyer-seller conversations -- poses unique scalability challenges in large-scale e-commerce due to informal language, extreme label sparsity, and rapidly evolving taxonomies. While utilizing Large Language Models (LLMs) as labeling oracles to distill ground-truth data has emerged as an industry standard to bypass prohibitive manual annotation costs, determining the optimal, low-latency architecture for the resulting student models remains an open challenge. To address this, we conduct a comprehensive evaluation across Small Language Model (SLM) parameter scales (1B, 4B, and 8B) and architectural paradigms (causal generative versus bidirectional discriminative). Comparing generative text-to-label classifiers against discriminative baselines (DeBERTa-V3 and ModernBERT), our analysis reveals a crucial data-dependent trade-off: while discriminati
    
[^295]: EDiS：面向图神经网络的边不相交子图稀疏化框架

    EDiS: Edge Disjoint Subgraph Sparsification Framework for Graph Neural Networks

    [https://arxiv.org/abs/2610.09059](https://arxiv.org/abs/2610.09059)

    EDiS提出了一种边不相交子图稀疏化框架，通过一次性将图分解为可缓存的边不相交子图并按边预算约束跨epoch重新组合，实现了高效且拓扑可变的GNN稀疏训练，避免了重复的结构提取计算。

    

    稀疏GNN训练可以减少计算量，但决定保留哪些边可能代价高昂。复用同一个稀疏图虽然开销小，但会将训练锁定在固定的拓扑结构上；而跨轮次（epoch）改变图结构则可能需要重复采样或重新计算。我们提出了EDiS（边不相交子图稀疏化框架），它将一次性结构提取与每个epoch的图组合分离开来。EDiS将图一次性分解为可缓存的边不相交子图，随后在不同epoch和保留率下将这些子图重新组合成满足边预算约束的图，而无需重新提取结构。我们的默认构造方法使用基于特征的分数和连续的最大分数覆盖森林，而相同的组合机制也支持其他边选择规则。我们对每个epoch的采样器（即从缓存的分解中抽取训练图的组合步骤）提供了组合分析。我们证明，在默认设置下……（摘要在此处被截断）

    arXiv:2610.09059v1 Announce Type: new  Abstract: Sparse GNN training reduces computation, but deciding which edges to keep can be costly. Reusing one sparse graph is cheap, but locks training to a fixed topology, while varying it across epochs can require repeated sampling or recomputation. We introduce EDiS (Edge-Disjoint Subgraph sparsification framework), which separates one-time structural extraction from per-epoch graph composition. EDiS decomposes the graph once into cacheable edge-disjoint subgraphs, then recombines them into graphs with edge-budget constraints across epochs and retention ratios without re-extracting structure. Our default construction uses feature-based scores and successive maximum score covering forests, while the same composition mechanism also supports alternative edge selection rules. We provide a combinatorial analysis of the per-epoch sampler, the composition step that draws a training graph from the cached decomposition. We show that, under the default 
    
[^296]: 通过显式多对多层映射学习跨模型激活对齐

    Learning Cross-Model Activation Alignments with Explicit Many-to-Many Layer Maps

    [https://arxiv.org/abs/2610.09058](https://arxiv.org/abs/2610.09058)

    MATCHA 方法将跨模型激活对齐分解为可解释的显式多对多层映射和层共享特征映射并从提示中联合学习，无需预先固定层对应关系，即可更忠实地重建目标模型激活并显著提升基于检索的指标。

    

    大语言模型正以极快的速度发布，这引出了一个自然的问题：两个独立训练的模型之间存在怎样的关联——哪些层相互对应，特征又在它们之间如何变换？我们通过学习激活对齐（activation alignment）来研究这一问题，即将源模型的逐层激活映射到目标模型的映射。我们的方法 MATCHA 将该映射分解为两个部分：其一是层映射（layer map），其输出是一个可提取、可检查的显式“目标层×源层”矩阵；其二是隐空间之间的层共享特征映射（feature map）。以往大多数工作都预先固定层对应关系，将相对深度大致相同的层进行配对；与之相反，我们从提示（prompts）中联合学习这两个因子。在涵盖三个不同模型家族的七个模型共 42 对模型组合上，与之前的方法相比，MATCHA 更忠实地重建了目标模型的激活，并显著提升了基于检索的指标。恢复出的映射在深度上大体呈单调性，但与以往假设……（原文摘要在此截断）

    arXiv:2610.09058v1 Announce Type: new  Abstract: LLMs are released at a rapid pace, raising a natural question: how do two independently trained models relate, both in which layers correspond and in how features transform between them? We study this by learning an activation alignment, a map from a source model's layerwise activations to a target's. Our method, MATCHA, factors this map into a layer map, whose output is an explicit target-by-source matrix that can be extracted and inspected, and a layer-shared feature map between hidden spaces. Most of prior work fixes the layer correspondence in advance, pairing layers at roughly the same relative depth; in contrast, we learn both factors jointly from prompts. Across 42 pairs of seven models spanning three different families, MATCHA reconstructs the target's activations more faithfully and improves retrieval-based metrics substantially, w.r.t. previous approaches. The recovered maps are broadly monotone in depth but, in contrast with m
    
[^297]: 基于几何的有限特征联想记忆容量理论

    A Geometry-Based Capacity Theory for Finite-Feature Associative Memory

    [https://arxiv.org/abs/2610.09056](https://arxiv.org/abs/2610.09056)

    该论文提出了有限特征Hebbian联想记忆的基于几何的容量理论，将检索干扰分解为随特征维度衰减的有限特征噪声和由键间核重叠决定的结构性干扰，从而实现无拟合的检索质量预测、揭示几何相关的容量上限，并将表示几何直接与记忆容量联系起来。

    

    我们为压缩的有限特征Hebbian联想记忆中的精确键检索建立了一种基于几何的容量理论。对于随机或近似各向同性的值，检索干扰可分解为有限特征噪声（随特征维度增加而减小）和结构性干扰（由存储键之间的核重叠平方决定，并在无限特征极限下依然存在）。由此可以实现对检索质量的无拟合预测，揭示依赖于几何的容量上限，并预测达到目标检索质量所需的特征预算。当存储的值之间存在相关性时，我们证明检索同时依赖于键核和值的Gram矩阵，并推导出考虑这种相互作用的有限特征近似。我们在合成数据、视觉表示和医学图像表示上对该理论进行了验证。总体而言，该框架将表示几何与记忆容量直接联系起来，并能区分……

    arXiv:2610.09056v1 Announce Type: new  Abstract: We develop a geometry-based capacity theory for exact-key retrieval in compressed finite-feature Hebbian associative memory. For random or approximately isotropic values, retrieval interference separates into finite-feature noise, which decreases with feature dimension, and structural interference, which is determined by squared kernel overlap among stored keys and persists in the infinite-feature limit. This yields a fit-free prediction of retrieval quality, reveals a geometry-dependent capacity ceiling, and predicts the feature budget required for a target retrieval quality. When stored values are correlated, we show that retrieval depends jointly on the key kernel and value Gram matrix, and derive finite-feature approximations that account for this interaction. We validate the theory on synthetic, visual, and medical-image representations. Overall, the framework links representation geometry directly to memory capacity and distinguish
    
[^298]: BeatFlow-ECG：利用校正流从间接可穿戴信号重建心电图

    BeatFlow-ECG: Rectified Flow for ECG Reconstruction from Indirect Wearable Signals

    [https://arxiv.org/abs/2610.09052](https://arxiv.org/abs/2610.09052)

    提出了BeatFlow-ECG，一种基于条件校正流的可穿戴信号重建模型，能从PPG和IMU信号重建单通道心电图，并通过IMU运动特征、运动相关损失加权和由易到难的训练课程有效应对运动干扰。

    

    摘要：在临床环境之外进行持续心脏监测，需要信号既具备丰富信息量又便于在日常生活中采集。心电图（ECG）能够提供关于心脏节律和波形形态的丰富信息，而可穿戴设备的光电容积描记信号（PPG）更容易连续采集，但它只是一种间接的心血管测量方式，且对运动高度敏感。我们提出了BeatFlow-ECG，这是一种条件校正流模型，用于从同步采集的PPG和惯性测量（IMU）信号中重建单通道ECG。BeatFlow-ECG将重建过程建模为从噪声到ECG的条件传输，采用带Transformer瓶颈层的卷积编码器-解码器结构，并引入显式的流时间条件化。模型通过IMU衍生的条件特征、依赖于运动程度的损失加权以及由易到难的训练课程来融合运动信息。我们在PPG（后文截断）数据集上采用留一受试者协议对模型进行评估。

    arXiv:2610.09052v1 Announce Type: new  Abstract: Continuous cardiac monitoring outside clinical settings requires signals that are both informative and practical to collect during daily life. Electrocardiography (ECG) provides rich information about cardiac rhythm and waveform morphology, while wearable photoplethysmography (PPG) is easier to acquire continuously but is only an indirect cardiovascular measurement and is highly sensitive to motion. We present BeatFlow-ECG, a conditional rectified-flow model for reconstructing single-channel ECG from synchronized PPG and inertial measurements. BeatFlow-ECG models reconstruction as conditional transport from noise to ECG using a convolutional encoder-decoder with a transformer bottleneck and explicit flow-time conditioning. Motion information is incorporated through IMU-derived conditioning features, motion-dependent loss weighting, and an easy-to-hard training curriculum. We evaluate the model under leave-one-subject-out protocols on PPG
    
[^299]: 自剪枝Transformer：基于通用注意力的极致KV缓存压缩

    A Self-Pruning Transformer: Extreme KV-Cache Compression with Universal Attention

    [https://arxiv.org/abs/2610.09051](https://arxiv.org/abs/2610.09051)

    该论文提出“通用注意力”架构，利用复合衰减机制作为自适应剪枝准则对KV缓存进行极致压缩，在保留RoPE位置嵌入和Softmax注意力的前提下实现了最先进的10倍压缩。

    

    现代大语言模型巨大的KV缓存规模为高效部署带来了障碍。近期工作探索了用替代性的基于衰减的机制替换注意力层的RoPE位置嵌入，从而在推理过程中对KV缓存进行修剪。然而，这些衰减函数表达能力有限，在实践中会退化为类似滑动窗口的驱逐模式。在本工作中，我们提出了一个统一的框架，涵盖互补且新颖的衰减机制，在保留富有表现力的RoPE嵌入和Softmax注意力的同时，捕捉复杂的键统计特性及其交互作用。由此产生的通用注意力（Universal Attention）是一种表达能力极强且可端到端训练的架构，其复合衰减机制充当了一种天然的、自适应的剪枝准则，可移除对注意力计算贡献最小的词元。实验表明，通用注意力在自然语言任务上实现了最先进的10倍压缩效果。

    arXiv:2610.09051v1 Announce Type: new  Abstract: The large KV-cache size of modern LLMs creates a barrier to efficient deployment. Recent work has explored replacing attention layers' RoPE positional embeddings with alternative decay-based mechanisms, which can then be used to prune KV-cache during inference. However, these decay functions have limited expressivity, and in practice devolve into sliding-window-like eviction patterns. In this work, we propose a unifying framework for complementary and novel decay mechanisms, capturing complex key statistics and interactions while preserving expressive RoPE embeddings and Softmax attention. The resulting Universal Attention is a highly expressive and end-to-end trainable architecture, whose composite decay mechanism acts as a natural, $\textit{adaptive}$ pruning criterion, removing tokens that contribute least to attention computation. Experimentally, Universal Attention achieves state-of-the-art $10\times$ compression on natural language
    
[^300]: 迈向金融世界建模

    Towards Financial World Modeling

    [https://arxiv.org/abs/2610.09048](https://arxiv.org/abs/2610.09048)

    该论文提出了包含近万亿条 1 Hz 观测数据的美股数据集 Market-1T、一套严谨的评估协议，并在近二十年数据上系统比较 18 种编码器训练策略，以推动金融表示学习迈向金融世界建模。

    

    构建世界模型需要一种对规划与决策有用的状态表示——这种表示可能需要应对训练时未知的任务。在金融市场的背景下，规划与决策可能要求模型能够推理市场整体状况、特定资产的预期收益、流动性、波动性以及跨资产关系。然而，以往金融表示学习的研究大多仅在单个预测任务上进行评估，且往往只使用单一时间段内相对狭窄的数据集。我们通过三个主要贡献来解决这一问题。首先，我们提出了 Market-1T 数据集，该数据集涵盖 2008 年至 2025 年的美国股票，以 1 Hz 分辨率记录了近一万亿条观测数据。其次，我们开发并实施了一套严谨的评估协议。第三，我们对金融表示学习开展了系统性大规模研究，在近二十年的市场数据上比较了 18 种编码器训练策略。

    arXiv:2610.09048v1 Announce Type: new  Abstract: Building a world model requires a state representation useful for planning and decision-making---potentially over tasks unknown at training time. In the context of financial markets, planning and decision-making may require a model to reason about market-wide conditions, asset-specific expected returns, liquidity, volatility, and cross-asset relationships. Yet financial representation learning has largely been evaluated on individual predictive tasks, oftentimes on a single time period using comparatively narrow datasets. We address this through three primary contributions. First, we introduce Market-1T, a dataset containing nearly one trillion observations across U.S. equities from 2008 to 2025 at 1 Hz resolution. Second, we develop and implement a rigorous evaluation protocol. Third, we conduct a systematic large-scale study of financial representation learning, comparing 18 encoder-training strategies across nearly two decades of mark
    
[^301]: 基于条件扩散模型学习跳跃扩散过程的转移核

    Learning Transition Kernels of Jump-Diffusion Processes with Conditional Diffusion Models

    [https://arxiv.org/abs/2610.09045](https://arxiv.org/abs/2610.09045)

    该论文提出用条件扩散模型学习时齐跳跃扩散过程的转移核，在理论上给出了条件分数估计误差和真实与生成路径分布间KL散度的非渐近界，并在合成与真实数据上验证了其在样本路径生成和概率预测任务中的有效性。

    

    我们研究了利用条件扩散模型学习时齐跳跃扩散过程转移核的问题，目标是从由N条独立轨迹组成的训练数据中生成新的样本路径，这些轨迹在高频离散时间网格上被观测。在理论方面，我们为条件分数估计误差以及真实离散观测路径分布与生成路径分布之间的KL散度建立了非渐近界。在数值方面，我们首先在合成数据上评估所提方法以验证理论发现，并将其性能与Gao等人（2025）的方法进行基准对比。随后我们将该方法应用于真实世界数据，并考察其在概率预测任务上的表现。

    arXiv:2610.09045v1 Announce Type: cross  Abstract: We study the problem of learning transition kernels for time-homogeneous jump-diffusion processes using conditional diffusion models, with the goal of generating new sample paths from training data consisting of N independent trajectories observed on a high-frequency discrete time grid. On the theoretical side, we establish non-asymptotic bounds for the conditional score estimation error and for the KL divergence between the laws of the true and generated discretely observed paths. On the numerical side, we first evaluate our method on synthetic data to assess the theoretical findings and benchmark its performance against the approach of Gao et al. (2025). We then apply our method to real-world data and investigate its performance on a probabilistic forecasting task.
    
[^302]: 基于随机矩阵理论的随机特征网络中的二次弱到强泛化

    Quadratic Weak-to-Strong Generalization in Random Feature Networks via Random Matrix Theory

    [https://arxiv.org/abs/2610.09044](https://arxiv.org/abs/2610.09044)

    本文利用随机矩阵理论证明，在两层随机特征网络中，由弱教师模型训练出的强学生模型误差为教师误差的平方，实现了二次级的弱到强泛化改进。

    

    弱到强泛化是指强学生模型在使用弱教师模型所产生的标签进行训练后，能够比教师模型具有更好泛化能力的现象。本文在两层随机特征网络中研究这一现象，其中模型的强度由其宽度决定。利用随机矩阵理论的工具，我们推导了最优训练的教师模型和通过梯度流训练的学生模型的总体误差的确定性等价形式。对于ReLU激活函数和纯球谐函数目标，我们在高斯普适性假设下获得了精确的渐近结果，展示出二次改进：学生误差按教师误差的平方比例缩放。这些结果达到了Medvedev等人（2025）给出的一般下界。我们还分析了学生在更一般的停止时间以及支持在多个谐波次数上的目标下的表现，刻画了弱到强泛化出现的区域。

    arXiv:2610.09044v1 Announce Type: cross  Abstract: Weak-to-strong generalization is the phenomenon where a strong student model trained with labels produced by a weak teacher model is able to generalize better than the teacher. In this paper, we study this phenomenon in two-layer random feature networks where the model strength is determined by its width. Using tools from random matrix theory, we derive deterministic equivalents for the population errors of an optimally trained teacher and a student trained with gradient flow. For ReLU activation and a pure spherical harmonic target, we obtain sharp asymptotics under a Gaussian universality assumption, showing a quadratic improvement: the student error scales as the square of the teacher error. These results attain the general lower bound of Medvedev at al (2025). We also analyze how the student behaves under more general stopping times and targets supported on multiple harmonic degrees, characterizing the regimes in which weak-to-stro
    
[^303]: 谨慎裁判：安全高效的人机协作决策

    Careful Judge: Safe and Efficient Human-AI Collaborative Decision Making

    [https://arxiv.org/abs/2610.09043](https://arxiv.org/abs/2610.09043)

    CARE是一个端到端的人机协作决策框架，通过新颖的自适应校准模块在任何时刻保证风险控制，并持续从人工反馈中学习，以更少的人工查询实现更高的自动化水平。

    

    在人机协作决策中，人工审核可以防止不安全的AI决策，但每一次人工判断的成本都很高。将AI弃权后的人工干预视为一次性的后备手段，会错失改进未来AI决策以实现更高自动化的机会；然而，AI若从选择性查询的人工反馈中进行自适应学习，又会破坏为旧模型校准的安全护栏。我们通过CARE——校准自适应纠正与升级——来应对这一挑战。CARE是一个端到端的流水线，将AI模型与人类审核员相结合，以保证安全且符合人类意图的决策，同时持续从人类反馈中学习，以更少的人工查询实现更高的自动化水平。CARE是有原则的、通用的、模块化的，可适用于任何黑盒AI模型。我们新颖的自适应校准模块可在任何时间步骤为任何纠正模块提供风险控制保证。我们还进一步展示了当AI模型（摘要内容不完整）……CARE如何提升查询效率。

    arXiv:2610.09043v1 Announce Type: cross  Abstract: In human-AI collaborative decision making, human review can prevent unsafe AI decisions, but each human judgment is costly. Treating human intervention after AI abstention as a one-off fallback misses the opportunity to improve future AI decisions for greater automation, yet AI adaptively learning from selectively queried human feedback breaks safety guardrails calibrated for old models. We approach this challenge with CARE---calibrated adaptive rectification and escalation---an end-to-end pipeline that combines AI models and human reviewers to guarantee safe, human-aligned decisions, while continuously learning from human feedback to achieve greater automation with fewer human queries. CARE is principled, general, modular, and works with any black-box AI model. Our novel adaptive calibration module guarantees risk control at every time step for any rectification module. We further show how CARE improves query efficiency when the AI mo
    
[^304]: ORACLE：面向约束学习的优化器相对对齐方法

    ORACLE: Optimizer-Relative Alignment for Constrained LEarning

    [https://arxiv.org/abs/2610.09040](https://arxiv.org/abs/2610.09040)

    ORACLE提出了一种优化器相对约束学习框架，通过在优化器更新之后评估约束兼容性，在优化器自身的几何结构中构建、限制权限并验证约束对齐，在八个偏微分方程基准和四种优化器上94%的配置中优于或匹配原生优化器的表现。

    

    约束处理方法通常在优化器起作用之前进行干预，即通过修改目标函数或梯度来实现。然而，动量、自适应缩放以及结构化预处理可能会在信号转化为参数更新之前对其产生实质性的重塑。我们提出了“优化器相对约束学习”的概念，其中约束兼容性是在优化器更新之后进行评估的。基于这一视角，我们提出了ORACLE，它通过对异构约束族进行联合端点线性化来评估原生优化器的实际步长，在优化器自身的几何结构中构建由此产生的对齐，限制其权限，并且仅在验证通过后才提交该更新。我们在八个偏微分方程基准测试以及涵盖欧几里得、对角自适应和结构化预处理几何结构的四种优化器上对ORACLE进行了评估，结果表明在94%的配置中，它的表现优于或等同于原生优化器。跨模型分析……

    arXiv:2610.09040v1 Announce Type: new  Abstract: Constraint handling methods typically intervene before the optimizer acts, by modifying the objective or the gradient. Yet momentum, adaptive scaling, and structured preconditioning can substantially reshape that signal before it becomes a parameter update. We formulate optimizer relative constrained learning, where constraint compatibility is assessed on the post optimizer update. Building on this view, we introduce ORACLE, which evaluates the native optimizer's realized step through a joint endpoint linearization of heterogeneous constraint families, constructs the resulting alignment in the optimizer's own geometry, bounds its authority, and commits it only after validation. We evaluate ORACLE across eight Partial Differential Equation benchmarks and four optimizers spanning Euclidean, diagonal adaptive, and structured preconditioned geometries, where it improves or matches native optimizer in 94% of configurations. Cross model analys
    
[^305]: 基于异构图神经网络的多智能体路径规划共享路线图生成与评估

    Shared-Roadmap Generation and Evaluator for Multi-Agent Path Planning Using Heterogeneous Graph Neural Network

    [https://arxiv.org/abs/2610.09034](https://arxiv.org/abs/2610.09034)

    该论文提出了一种可扩展的异构图神经网络框架，能够自动生成并评估多智能体路径规划所需的共享路线图，通过学习专家求解器轨迹的占据密度图来识别关键路径点并剪除冗余节点与边，从而在路线图紧凑性与解的质量之间取得平衡。

    

    连续环境中的多智能体路径规划（MAPP）通常依赖路线图来平衡安全性与搜索效率。然而，传统的路线图生成方法（如栅格网格或标准的基于采样的方法）经常面临图的密度与找到可行、高质量解的可能性之间的权衡。在本文中，我们提出了一个可扩展的异构图神经网络（GNN）框架，用于共享多智能体路线图的自动化生成与评估。我们的模型将路径点、智能体位置和任务位置表示为异构图中的不同类型节点，使其能够对全局连通性以及智能体之间的交互进行推理。通过在从专家求解器轨迹中聚合收集的占据密度图上进行训练，GNN学会识别关键的感兴趣点，并剪除冗余的节点和边。这一过程产生了一个紧凑的、协调的（摘要在此处截断）……

    arXiv:2610.09034v1 Announce Type: cross  Abstract: Multi-agent path planning (MAPP) in continuous environments often relies on roadmaps to balance safety and search efficiency. However, traditional roadmap generation methods, such as lattice grids or standard sampling-based approaches, frequently face a trade-off between graph density and the likelihood of finding feasible, high-quality solutions. In this paper, we propose a scalable heterogeneous Graph Neural Network (GNN) framework for the automated generation and evaluation of shared multi-agent roadmaps. Our model covers the representation of waypoints, agent locations, and task locations as distinct nodes in a heterogeneous graph, allowing it to reason over global connectivity and inter-agent interactions. By training on occupation density maps aggregated and collected from expert solver trajectories, the GNN learns to identify critical points of interest and prune redundant nodes and edges. This process produces a compact, coordi
    
[^306]: 开放权重大型语言模型在非规范输入上的四态安全评估

    Quad-State Safety Evaluation of Open-Weight Large Language Models on Non-Canonical Inputs

    [https://arxiv.org/abs/2610.09033](https://arxiv.org/abs/2610.09033)

    本文提出ASRD数据集与四态评估框架，发现表情符号和不可见Unicode等表层变换对开放权重大语言模型的安全威胁远高于Leet语言和编码包装等变换，有害遵从率可达20%以上。

    

    标准的大语言模型安全评估通常针对以规范纯文本形式编写的有害请求，而在实际部署中，模型经常收到包含表情符号、变体拼写、编码字符串和字符级变化的输入。本工作引入了对抗性表层形式鲁棒性数据集，包含涵盖七种不同表层形式类别的2,100个提示词。研究在五个开放权重语言模型上对这些提示词进行了评估，产生了10,500个回复。四态评估量表将每个回复归类为四种结果之一：有害遵从、安全响应、理解失败或无法判定。结果表明，表情符号和不可见Unicode变体几乎不会导致理解失败，其汇总有害遵从率分别为20.27%和17.20%，而22.87%的基线主要由Mistral 7B驱动；相比之下，Leet语言（字母替换）、编码包装和混合转换的有害遵从率仅为2.40%、0.13%和2.40%。

    arXiv:2610.09033v1 Announce Type: new  Abstract: Standard safety evaluations of large language models assess harmful requests written in canonical plain text, while models in real-world deployment routinely receive inputs containing emojis, altered spellings, encoded strings, and character-level variations. This work introduces the Adversarial Surface-Form Robustness Dataset (ASRD), comprising 2,100 prompts across seven distinct surface-form families. Five open-weight language models are evaluated across these prompts, producing 10,500 responses. The Quad-State Evaluation Rubric classifies each response into one of four outcomes: harmful compliance, safe response, comprehension failure, or indeterminate. Emoji and invisible Unicode variations cause almost no comprehension failure, with pooled harmful compliance of 20.27% and 17.20% against a 22.87% baseline that is driven mainly by Mistral 7B, whereas leetspeak, encoded wrappers, and hybrid transformations score 2.40%, 0.13%, and 2.40%
    
[^307]: 超越解释：通过概念干预调试医学影像模型

    Beyond Explanation: Debugging Medical Imaging Models via Concept Intervention

    [https://arxiv.org/abs/2610.09031](https://arxiv.org/abs/2610.09031)

    该论文提出了一个即插即用的概念干预框架，通过构建与 BioMedCLIP 对齐的概念瓶颈模型来区分因果概念与虚假相关概念，并利用反事实样本进行针对性微调，从而实现对医学影像模型的可解释调试与性能提升。

    

    医学影像模型通常以黑箱方式运行，限制了可解释性和系统性调试。我们介绍了一个易于使用的即插即用框架，用于基于概念的解释和模型优化。通过将单模态编码器对齐到 BioMedCLIP，我们构建了一个概念瓶颈模型（CBM），支持概念层面的干预。这些干预使我们能够区分因果概念与虚假相关概念，与领域专家共同验证洞察，并生成反事实样本用于针对性微调。我们在梅奥诊所超声数据集和 CheXpert 5x200 胸部 X 光数据集上对该框架进行了评估。结果表明，概念干预能够实现可靠的模型诊断，同时通过引导微调保持甚至有时提升预测性能。我们的发现凸显了该框架在临床深度学习模型受控、可解释优化方面的实用价值。

    arXiv:2610.09031v1 Announce Type: cross  Abstract: Medical imaging models often operate as black boxes, limiting interpretability and systematic debugging. We introduce an easy-to-use, plug-and-play framework for concept-based interpretation and model refinement. By aligning a single-modality encoder to BioMedCLIP, we construct a Concept Bottleneck Model (CBM) that enables concept-level interventions. These interventions allow us to isolate causal versus spuriously correlated concepts, validate insights with domain experts, and generate counterfactual samples for targeted fine-tuning. We evaluate our framework on a Mayo Clinic ultrasound dataset and the CheXpert 5x200 chest X-ray dataset. Results demonstrate that concept intervention enables reliable model diagnosis while maintaining, and occasionally improving predictive performance via guided fine-tuning. Our findings highlight the practical value of this framework for controlled, interpretable refinement of clinical deep learning mo
    
[^308]: SPIN：用于稀疏注意力的影子预测索引器

    SPIN: Shadow Predictive Indexer for Sparse Attention

    [https://arxiv.org/abs/2610.09025](https://arxiv.org/abs/2610.09025)

    SPIN 提出基于历史的轻量级预测机制来识别重要 KV 块，避免每个解码步骤对完整 KV 缓存评分，在保持任务质量的同时实现 30-40% 的稀疏度，并将 vLLM 服务的吞吐量最高提升 14.9%、token 间中位延迟最高降低 13.2%。

    

    基于索引器的稀疏注意力通过仅向核心注意力传递固定数量的少量重要 token 来降低其成本。然而，索引器仍然需要在每个解码步骤中对整个 KV 缓存进行评分。随着上下文长度的增长，这种评分开销成为主要瓶颈。我们提出 SPIN（影子预测索引器）来降低这种索引器开销。SPIN 使用轻量级的、基于历史的预测来识别重要的 KV 块，从而避免在每个解码步骤中对完整 KV 缓存进行评分。SPIN 将 KV 块和投机解码作为一等公民纳入设计与实现考量。在长上下文和智能体基准测试上的广泛评估中，SPIN 在保持任务质量的同时实现了 30-40% 的稀疏度。在端到端 vLLM 服务中，SPIN 将输出吞吐量最高提升 14.9%，并将 token 间中位延迟最高降低 13.2%。

    arXiv:2610.09025v1 Announce Type: new  Abstract: Indexer-based sparse attention reduces the cost of core attention by passing only a fixed, small number of important tokens to it. However, the indexer must still score the entire KV cache at every decoding step. This scoring overhead becomes a major bottleneck as the context length grows. We propose SPIN (Shadow Predictive Indexer) to reduce this indexer overhead. SPIN uses lightweight, history-based prediction to identify important KV blocks, avoiding the need to score the full KV cache at every decoding step. SPIN treats KV blocks and speculative decoding as first-class design and implementation considerations. Across extensive evaluations on long-context and agentic benchmarks, SPIN achieves 30-40% sparsity while preserving task quality. In end-to-end vLLM serving, SPIN improves output throughput by up to 14.9% and reduces median inter-token latency by up to 13.2%.
    
[^309]: 面向校准的邻域平滑

    Neighborhood Smoothing for Calibration

    [https://arxiv.org/abs/2610.09020](https://arxiv.org/abs/2610.09020)

    该论文提出将图平滑作为训练时校准的一般原则，通过惩罚表示空间中相邻样本预测分布之间的Jensen-Shannon散度的图正则化方法来改善神经网络的过度自信和校准问题。

    

    现代神经网络通常校准不良，往往表现出过度自信的倾向。现有的训练时校准方法大多通过修改任务损失或校准惩罚项来实现，而对所学表示中的邻域结构利用不足。我们引入图平滑作为训练时校准的一般原则，它鼓励表示空间中相邻样本具有相似的预测分布。我们分析了图平滑的效果，推导出将相邻样本之间的预测散度与局部置信度变化以及逐点校准误差的传播联系起来的界，并刻画了平滑在何种条件下能够或不能改善校准。基于这一分析，我们提出了该模型，这是一种基于图的训练时正则化器，它惩罚相邻样本预测分布之间的Jensen-Shannon散度。我们提供了全面的实证……

    arXiv:2610.09020v1 Announce Type: new  Abstract: Modern neural networks are often miscalibrated, with a tendency to overconfidence. Existing train-time calibration methods largely modify task losses or calibration penalties, leaving neighborhood structure in learned representations underexploited. We introduce graph smoothing as a general principle for train-time calibration, which encourages similar predictive distributions across neighboring samples in representation space. We analyze the effects of graph smoothing, deriving bounds that connect predictive divergence between neighboring samples to local confidence variation and to the propagation of pointwise calibration error, and characterize the conditions under which smoothing can or cannot improve calibration. In light of this analysis, we propose \modelNoSpace, a graph-based train-time regularizer that penalizes the Jensen--Shannon divergence between predictive distributions of neighboring samples. We present a thorough empirica
    
[^310]: 移除信息内容并不能证明开放权重模型的防篡改能力

    Removing Information Content Does Not Certify Tamper Resistance in Open-Weight Models

    [https://arxiv.org/abs/2610.09004](https://arxiv.org/abs/2610.09004)

    移除互信息并不足以证明开放权重模型的防篡改能力，因为保持函数不变的重新参数化可以在信息量不变的情况下改变梯度下降几何结构，甚至存在信息量为零却能在一步梯度中恢复能力的构造。

    

    移除有害信息能否使开放权重模型对微调攻击具有抵抗力？我们证明，仅凭发布时的互信息无法普遍地证明缓慢的恢复能力。保持函数不变的重新参数化可以在不改变信息的同时改变梯度下降的几何结构，因此任何不变性证书都受限于最快的可达参数化。我们将这一原理应用于训练数据过滤下的权重-数据互信息，以及能力移除下的标签-表示互信息。在固定的权重-数据信息量下，训练顺序可以改变恢复时间，而精确的表示级独立性仍可保留完整的参数雅可比矩阵。我们给出了一个显式构造，其两个信息量均为零，却能在一步梯度下降中恢复能力。受控实验说明了依赖顺序的恢复行为和依赖参数化的攻击速度。这些结果指出了缺失的必要条件

    arXiv:2610.09004v1 Announce Type: new  Abstract: Does removing harmful information make open-weight models resistant to fine-tuning attacks? We show that mutual information at release alone cannot universally certify slow recovery. Function-preserving reparameterizations leave information unchanged while altering gradient-descent geometry, so an invariant certificate is bounded by the fastest reachable parameterization. We apply this principle to weight--data mutual information under training-data filtering and label--representation mutual information under capability removal. Training order can change recovery time at fixed weight--data information, while exact representation-level independence can preserve the entire parameter Jacobian. An explicit construction has both information quantities equal to zero and recovers in one gradient step. Controlled experiments illustrate order-dependent recovery and parameterization-dependent attack speed. These results identify the missing requir
    
[^311]: 面向微型Transformer算术推理的算法草稿纸与课程分阶段训练

    Algorithmic Scratchpads and Curriculum Staging for Arithmetic Reasoning in Tiny Transformers

    [https://arxiv.org/abs/2610.09003](https://arxiv.org/abs/2610.09003)

    本文通过研究微型Transformer的多步算术推理，发现并修复了序列填充导致的梯度饥饿问题，证明语言预训练与现代架构组件至关重要，且算法草稿纸的表述形式直接决定模型推理性能。

    

    自回归大语言模型在多位数乘法和长除法等确定性多步骤算法任务上经常表现不佳。本文研究了在合成数据上训练的紧凑型“微型”Transformer（约1060万非嵌入参数，总计4930万参数）中多步算术的运行机制，模型在四种基本运算（加、减、乘、除）上以逐步草稿纸的形式展开训练。首先，我们确立了必要的训练基础：（1）数据加载器的序列填充会产生83%的梯度饥饿伪影，使准确率从40%骤降至1%，可通过连续序列打包加以修复；（2）语言预训练是不可或缺的先决条件（否则准确率不超过2.0%）；（3）现代架构组件（RoPE、RMSNorm、SwiGLU）和稀疏混合专家模型（MoE）相比基线GPT-2显著提升了加法推理能力。其次，我们证明算法草稿纸的表述方式直接决定了……（注：原摘要在此处被截断）

    arXiv:2610.09003v1 Announce Type: new  Abstract: Autoregressive Large Language Models (LLMs) frequently struggle with deterministic multi-step algorithmic tasks such as multi-digit multiplication and long division. In this paper, we investigate the mechanics of multi-step arithmetic in compact "Tiny" Transformers (~10.6M non-embedding parameters, 49.3M total) trained on synthetic data across four basic operations (+, -, *, /) unrolled as step-by-step scratchpads. First, we establish the necessary training foundations: (1) dataloader sequence padding creates an 83% gradient starvation artifact that collapses accuracy from 40% to 1%, remediated via continuous sequence packing; (2) linguistic pretraining is an essential prerequisite (<= 2.0% without it); and (3) modern architectural primitives (RoPE, RMSNorm, SwiGLU) and Sparse Mixture of Experts (MoE) substantially improve additive reasoning over baseline GPT-2. Second, we demonstrate that algorithmic scratchpad formulation directly dict
    
[^312]: 设备端语言模型的安全性有多脆弱？定位安全关键参数以进行稀疏故障分析

    How Fragile Is On-Device Language Model Safety? Localizing Safety-Critical Parameters for Sparse Fault Analysis

    [https://arxiv.org/abs/2610.09000](https://arxiv.org/abs/2610.09000)

    研究发现LLaMA-2-7B-Chat的安全敏感行为高度集中在MLP的down_proj等稀疏参数子集中，仅修改0.19%的权重即可使攻击成功率大幅上升，揭示了设备端部署的语言模型存在显著的安全脆弱点。

    

    随着小型语言模型（SLM）越来越多地部署在资源受限的设备端平台上，包括作为智能体系统的组件，本地存储的模型参数的完整性成为一个重要的安全问题。我们研究了LLaMA-2-7B-Chat中的安全敏感行为是否集中在参数的稀疏子集中，从而为针对性分析创建了一个缩小的故障面。我们研究了两种互补的定位方法：低秩安全相关子空间分析和参数级安全-效用重要性过滤。两种方法都揭示了网络中高度不均匀的安全敏感性，其中MLP的down_proj始终是突出的安全敏感组件，而o_proj的贡献较小。利用参数级定位，仅修改down_proj中0.19%的模型权重就能产生53%的基本攻击成功率（Basic ASR）和56%的GCG攻击成功率，而tinyBenchmarks准确率仍保持在51。

    arXiv:2610.09000v1 Announce Type: cross  Abstract: As small language models (SLMs) are increasingly deployed on resource-constrained and on-device platforms, including as components of agentic systems, the integrity of locally stored model parameters becomes an important safety concern. We investigate whether safety-sensitive behavior in LLaMA-2-7B-Chat is concentrated within a sparse subset of parameters, creating a reduced fault surface for targeted analysis. We study two complementary localization methods: low-rank safety-associated subspace analysis and parameter-level safety--utility importance filtering. Both approaches reveal highly non-uniform safety sensitivity across the network, with the MLP down_proj consistently emerging as a prominent safety-sensitive component and o_proj providing a smaller contribution. Using parameter-level localization, modifying only 0.19% of model weights in down_proj yields 53% Basic ASR and 56% GCG ASR, while tinyBenchmarks accuracy remains at 51.
    
[^313]: REFIT：无需标签即可识别、修复和测试可穿戴传感器佩戴位置偏移

    REFIT: Recognize, Fix, and Test Wearable Sensor Placement Shifts without Labels

    [https://arxiv.org/abs/2610.08991](https://arxiv.org/abs/2610.08991)

    REFIT是一种无需标签和重新训练的输入校准方法，通过轴变换族拟合和归一化统计量重估计，自动识别并修复可穿戴传感器佩戴位置变化对冻结活动识别模型的影响，并提供无标签准确率评估。

    

    我们提出了REFIT，这是一种针对冻结活动识别模型的输入校准方法，用于解决惯性传感器在部署时与训练时佩戴方式不同的场景。当用户将手表移到另一只手腕，或将绑带传感器翻转后重新戴上时，模型会在改变后的坐标轴上看到相同的运动。REFIT无需标签或重新训练即可消除此类偏移。它通过一系列轴变换族（如反射和旋转）来描述这些偏移，并将每个变换族拟合到用户数据上，使其简单统计量与训练数据的统计量相匹配。能够消除最多不匹配性的变换族即标识出了偏移类型。REFIT通过在冻结模型之前应用该族中的最佳变换来修复偏移，并重新估计模型的归一化统计量。它使用无标签的准确率估计来测试修复后的模型，并在准确率较低时提示用户重新佩戴传感器。在真实的左右传感器配对以及真实与模拟的重新佩戴实验中，REFIT的表现优于……

    arXiv:2610.08991v1 Announce Type: new  Abstract: We present REFIT, an input calibration for frozen activity-recognition models whose inertial sensors are worn differently at deployment than in training. When users move a watch to the other wrist or put a strap sensor back on turned, the model sees the same motion on changed axes. REFIT undoes such shifts without labels or retraining. It describes them by families of axis transforms, such as reflections and rotations, and fits each family to the user's data so that simple statistics match those of the training data. The family that removes most of the mismatch names the shift. REFIT fixes the shift by applying the best member of that family before the frozen model and re-estimating its normalization statistics. It tests the fixed model with a label-free accuracy estimate and asks the user to re-wear the sensor when it is low. Experiments on real left/right sensor pairs and on real and simulated re-attachment show that REFIT outperforms 
    
[^314]: LASER：用于支撑约束熵正则化离线强化学习的潜在空间伴随匹配

    LASER: Latent Space Adjoint Matching for Support-Constrained Entropy-Regularized Offline RL

    [https://arxiv.org/abs/2610.08989](https://arxiv.org/abs/2610.08989)

    LASER通过潜在空间伴随匹配实现熵正则化的潜在空间离线强化学习，既防止了策略坍缩成单一脆弱模式，又避免了时间反向传播，在40个不同数据质量的OGBench任务上取得了优异表现。

    

    虽然离线强化学习（RL）能够在无需昂贵的在线交互的情况下从静态数据集中进行策略优化，但其性能仍然受到执行分布外（OOD）动作风险的制约。近期的方法通过流匹配学习一个行为克隆策略，然后在其受约束的潜在空间内执行强化学习，从而缓解了这一问题。然而，朴素地优化潜在策略很容易导致策略坍缩成脆弱的单一模式，或利用学习到的评论家网络中尖锐的伪影。在这项工作中，我们发现熵正则化对于在潜在空间强化学习中应对这些挑战至关重要。我们提出了LASER，一种新颖的离线强化学习算法，它应用潜在空间伴随匹配来实现基于强表达能力的流策略的熵正则化潜在空间强化学习，同时避免了时间反向传播。通过在40个具有不同数据集质量的具有挑战性的OGBench任务上的全面实验，我们展示了（摘要在此处截断）

    arXiv:2610.08989v1 Announce Type: new  Abstract: While offline reinforcement learning (RL) enables policy optimization from static datasets without costly online interaction, it remains bottlenecked by the risk of executing out-of-distribution (OOD) actions. Recent approaches mitigate this by learning a behavior-cloning policy through flow matching and then performing RL within its constrained latent space. However, naively optimizing the latent policy can easily cause the policy to collapse into a brittle mode or exploit sharp artifacts of the learned critic. In this work, we find that entropy regularization is essential in latent-space RL for addressing these challenges. We introduce LASER, a novel offline RL algorithm that applies latent-space adjoint matching to achieve entropy-regularized latent-space RL with expressive flow policies while avoiding backpropagation through time. Through comprehensive experiments on 40 challenging OGBench tasks with varying dataset qualities, we sho
    
[^315]: 一种使用本质特征对TAIGA实验数据进行多模态分析的方法

    A method for multimodal analysis of TAIGA experiment data using essential features

    [https://arxiv.org/abs/2610.08985](https://arxiv.org/abs/2610.08985)

    提出了一种基于自编码器等神经网络提取本质特征的新方法，实现了对TAIGA实验来自多个装置的多模态数据的联合分析。

    

    arXiv:2610.08985v1 公告类型：交叉 摘要：处理和分析物理实验数据的目的在于获取关于所研究现象的具有物理意义的信息。这一目标通过对实验数据的多阶段处理来实现，在此过程中抑制与测量相关的噪声并降低输入数据的维度。本文提出了一种基于使用自编码器等神经网络来提取本质特征的新方法。所提方法的特殊价值在于它可以应用于对同时从多个装置接收的多模态数据进行分析。我们将把这种方法应用于TAIGA实验的多模态数据（MMD）。目前，MMD的分析是针对每个装置独立进行的。因此，开发针对TAIGA类装置的MMD联合分析方法是宇宙射线研究领域中一项紧迫的任务。

    arXiv:2610.08985v1 Announce Type: cross  Abstract: The aim of processing and analyzing experimental data from physical experiments is to obtain physically significant information about the phenomenon under study. This goal is achieved by multi-stage processing of experimental data, during which noise associated with measurements is suppressed and the dimensionality of the input data is reduced. In this paper, we propose a new method based on the use of neural networks such as autoencoders to extract essential features. The special value of the proposed approach lies in the possibility of its application to the analysis of multimodal data received simultaneously from several installations. We will apply this approach to a multimodal data (MMD) of the experiment TAIGA. Currently, the analysis of the MMD is carried out independently for each installation separately. Therefore, the development of methods for the joint analysis of MMD from TAIGA-type installations is an urgent task in cosmi
    
[^316]: 用于MIMO信道估计的信噪比门控LSTM条件扩散模型

    SNR-Gated LSTM-Conditioned Diffusion Model for MIMO Channel Estimation

    [https://arxiv.org/abs/2610.08977](https://arxiv.org/abs/2610.08977)

    该论文提出了一种SNR门控LSTM条件扩散模型，在角度域对MIMO信道进行估计，通过利用时间序列动态特性和可学习的信噪比自适应融合机制，在宽SNR范围内实现准确且低时延的信道估计。

    

    准确且低时延的信道估计对现代MIMO系统至关重要，尤其是在移动性场景下，此时信道表现出结构化稀疏性和强时间相关性。本文提出了一种用于信道估计的时间序列条件扩散框架，该框架在角度域进行去噪。从最小二乘（LS）观测出发，我们训练了一个扩散去噪器，其条件信息由长短期记忆（LSTM）网络在短观测序列上编码，使模型能够利用超越单快照估计的时间动态特性。为了在宽信噪比（SNR）范围内稳健地平衡观测保真度与学习到的生成先验，我们引入了一种可学习的SNR门控后期融合捷径，通过一个具有可训练中心和尺度的sigmoid门控将网络输入注入到最终解码阶段。为了降低推理时延，我们采用确定性去噪（原文摘要在此处被截断，内容不完整）。

    arXiv:2610.08977v1 Announce Type: new  Abstract: Accurate and low latency channel estimation is critical for modern MIMO systems, particularly under mobility, where channels exhibit structured sparsity and strong temporal correlation. This paper proposes a time-series conditioned diffusion framework for channel estimation that performs denoising in the angular domain. Starting from least squares (LS) observations, we train a diffusion denoiser whose conditioning information is encoded by a long short-term memory (LSTM) network over a short observation sequence, enabling the model to exploit temporal dynamics beyond per-snapshot estimation. To robustly balance observation fidelity and learned generative priors across a wide signal-to-noise ratio (SNR) range, we introduce a learnable SNR-gated late-fusion shortcut that injects the network input into the final decoding stage through a sigmoid gate with trainable center and scale. To reduce inference latency, we adopt deterministic denoisi
    
[^317]: 最佳优化器取决于批量大小

    The Best Optimizer Depends on Batch Size

    [https://arxiv.org/abs/2610.08975](https://arxiv.org/abs/2610.08975)

    该论文挑战了“某一批量大小下最佳的优化器在其他批量大小下也最佳”的常见假设，证明Muon缺乏一致的缩放规则，且即使经过大量超参数调优，语言模型预训练的最佳优化器仍会随批量大小而改变。

    

    大量新的自适应优化器被设计用于高效估计和利用小批量梯度统计量来塑造参数更新，但它们通常只在单一批量大小下进行基准测试。超参数缩放规则承诺在批量大小和梯度噪声变化时保持性能，这暗示着在某一批量大小下最佳的优化器在另一批量大小下也应保持最佳。我们通过以下发现挑战了这种开发和评估优化器的方法：(1) Muon没有一种原则性的缩放规则能在各种训练设置中保持一致有效；(2) 即使经过大量的超参数调优，语言模型预训练的最佳优化器也会随批量大小而变化。

    arXiv:2610.08975v1 Announce Type: new  Abstract: A plethora of new adaptive optimizers are designed to efficiently estimate and use minibatch gradient statistics to shape parameter updates, but they are typically benchmarked at a single batch size. Hyperparameter scaling rules promise to preserve performance as batch size and gradient noise change, suggesting that the best optimizer at one batch size should remain the best at another. We challenge this approach to developing and evaluating optimizers by showing: (1) no principled scaling rule for Muon works consistently across training settings, and (2) the best optimizer for language model pretraining changes with batch size even after extensive hyperparameter tuning.
    
[^318]: HULK：人形机器人全身强力移动操作学习

    HULK: Learning Whole-Body Forceful Loco-Manipulation for Humanoids

    [https://arxiv.org/abs/2610.08970](https://arxiv.org/abs/2610.08970)

    提出HULK全身控制框架，利用模型预测控制引导强化学习训练两个教师策略（腕力手臂运动跟踪与抱物行走），并通过捕获点控制障碍函数增强负载下的平衡能力，最终蒸馏为单一策略，实现人形机器人对大型重物的强力移动操作。

    

    人形机器人对大型、重物件的移动操作需要全身参与强力交互。然而，此类负载会改变人形机器人的质心，并对上半身施加持续负荷，从而给平衡和指令跟踪带来挑战。我们提出了HULK，一个面向强力移动操作的全身控制框架。该框架利用模型预测控制（MPC）通过预测带负载的动力学来引导强化学习，训练了两个教师策略：一个在手腕受力的条件下跟踪手臂运动，另一个在将大物体抱在身上时进行行走移动。在训练过程中，捕获点控制障碍函数对腕力教师策略进行增强，以提高负载下的平衡能力。随后我们将两个教师策略蒸馏为单一策略。评估涵盖仿真环境和Unitree G1实机。在仿真中，在每臂10公斤负载的条件下，带障碍函数的教师策略在被评估的控制器中实现了最低的前向和侧向速度跟踪误差。

    arXiv:2610.08970v1 Announce Type: cross  Abstract: Humanoid loco-manipulation of large, heavy objects demands forceful interaction across the entire body. However, such payloads shift a humanoid's center of mass and impose sustained loads across the upper body, challenging balance and command tracking. We present HULK, a whole-body control framework for forceful loco-manipulation. Using model predictive control (MPC) to guide reinforcement learning with predictions of the loaded dynamics, we train two teachers: one tracks arm motions under wrist forces, and the other locomotes while holding large objects against the body. A capture-point control barrier function augments the wrist-force teacher during training to improve balance under load. We distill both teachers into a single policy. Evaluation spans simulation and the Unitree G1. In simulation, the teacher with the barrier function achieves the lowest forward and lateral velocity tracking errors at 10 kg per arm among evaluated con
    
[^319]: 趁它们沉睡时工作：利用评估延迟实现全贝叶斯优化

    Work While They Sleep: Exploiting Evaluation Latency for Fully Bayesian Optimization

    [https://arxiv.org/abs/2610.08969](https://arxiv.org/abs/2610.08969)

    该论文提出ELF-BO算法，巧妙利用贝叶斯优化中昂贵目标函数评估期间的等待时间，提前并行计算全贝叶斯代理模型，从而在不增加额外时间成本的情况下获得更好的不确定性估计和优化性能。

    

    黑盒优化问题在科学与工程领域无处不在，通常涉及代价高昂的目标函数。这种目标评估延迟在优化过程中带来两个后果：（i）目标函数评估主导了整个执行时间；（ii）样本高效的算法对于加速开发、避免资源浪费至关重要。贝叶斯优化（BO）方法是规划器为推荐下一个尝试点的事实上的标准选择。标准的BO采用点估计来拟合代理模型的超参数；而完全贝叶斯方法则通过模型平均来考虑超参数的不确定性，从而获得更好的不确定性估计——这在BO中普遍存在的低数据场景下非常有用。然而，该方法往往计算代价过高，因此很少被使用。在这项工作中，我们提出ELF-BO，一种利用目标函数评估延迟来提前启动全贝叶斯代理模型计算的算法。

    arXiv:2610.08969v1 Announce Type: new  Abstract: Black-box optimization problems are ubiquitous across science and engineering, often dealing with expensive objective functions. This objective latency has two consequences during optimization: (i) the objective evaluation dominates execution time, and (ii) sample-efficient algorithms are crucial to accelerate development and avoid wasting resources. Bayesian optimization (BO) methods are the \textit{de facto} choice of planners for suggesting the next point to try. Standard BO fits the surrogate model's hyperparameters with a point estimate. Alternatively, a fully Bayesian approach uses model averaging to account for uncertainty over the hyperparameters, leading to better uncertainty estimates---useful in the low-data regime that is pervasive in BO. However, it is often prohibitively expensive and thus rarely used. In this work, we propose ELF-BO, an algorithm that uses the objective evaluation latency to headstart the computation of th
    
[^320]: 论KL正则化策略优化

    On KL-Regularized Policy Optimization

    [https://arxiv.org/abs/2610.08963](https://arxiv.org/abs/2610.08963)

    提出KLPO框架，通过将KL正则项锚定在采样器上，利用闭式Gibbs解的对数比最优性条件在采样器自身轨迹上做最小二乘拟合，从而在不使用重要性权重的情况下解决LLM智能体异步强化学习中采样与训练策略不一致的问题。

    

    摘要（arXiv:2610.08963v1，交叉公告）：面向大语言模型（LLM）智能体的异步强化学习（RL）需要让一个策略在由另一个策略生成的轨迹上进行训练：采样数据来自过期的检查点，并且即使参数完全相同，推理引擎给出的概率也与训练器给出的概率不同。标准的补救方法要么对重要性比率进行裁剪（这会使更新产生偏差），要么像GRPO那样为每个提示采样一组响应（当回合较长时代价高昂）。我们提出了KL正则化策略优化（KLPO），这是一个将KL正则项锚定在采样器上的框架。在此框架下，正则化后的改进步骤具有闭式Gibbs解，KLPO通过在采样器自身的轨迹上以最小二乘法拟合其对数比最优性条件，使采样器概率以对数比的形式进入公式，从而无需任何重要性权重。通过对回归截距进行轮廓化处理，难以处理的log配分函数被替换为信号的采样器均值加上一个采样……（原文摘要在此处截断）

    arXiv:2610.08963v1 Announce Type: cross  Abstract: Asynchronous reinforcement learning (RL) for large language model (LLM) agents trains one policy on trajectories generated by another: rollouts come from stale checkpoints, and the inference engine's probabilities differ from the trainer's even at identical parameters. Standard remedies either clip importance ratios, which biases the update, or, as in GRPO, sample a group of responses per prompt, which is costly when episodes are long. We propose KL-Regularized Policy Optimization (KLPO), a framework that anchors the KL regularizer at the sampler. The regularized improvement step then has a closed-form Gibbs solution, and KLPO fits its log-ratio optimality condition by least squares on the sampler's own trajectories, so the sampler probability enters through a log-ratio and no importance weights are needed. Profiling out the regression intercept replaces the intractable log-partition function with the signal's sampler mean plus a sampl
    
[^321]: 面向离线视觉控制的定向时序表征

    Directed Temporal Representations for Offline Visual Control

    [https://arxiv.org/abs/2610.08960](https://arxiv.org/abs/2610.08960)

    该论文提出DTRC方法，在冻结世界模型特征之上学习与时间可达性对齐的有向时间拟度量表征，并利用其时间进度信号作为评论家，实现离线视觉目标条件策略的直接学习。

    

    预测型世界模型为控制任务提供了紧凑的视觉表征。然而，控制需要一种与时间可达性对齐的潜在几何结构，而不仅仅是预测相似性。我们提出了面向控制的定向时序表征，该方法在冻结的LeWorldModel（LeWM）特征之上，从离线视觉轨迹中学习这种几何结构。DTRC在所学习的控制表征上构建有向时间拟度量。短程时间偏移用于校准距离尺度；自举目标将时间可达性扩展到更长的时间范围；动作条件一致性使表征与局部转移动力学对齐。所得的距离估计时间到达代价，而其在一次转移中的变化定义了相对于目标的时间进度。我们将该进度信号作为时间评论家，用于直接的目标条件策略学习。模型辅助目标还提供了额外的训练信号。

    arXiv:2610.08960v1 Announce Type: new  Abstract: Predictive world models provide compact visual representations for control. Control requires a latent geometry aligned with temporal reachability rather than predictive similarity alone. We introduce Directed Temporal Representations for Control (DTRC), which learns such a geometry from offline visual trajectories on top of frozen LeWorldModel (LeWM) features. DTRC constructs a directed temporal quasimetric over the learned control representation. Short-range temporal offsets calibrate the distance scale. Bootstrapped targets extend temporal reachability across longer horizons. Action-conditioned consistency aligns the representation with local transition dynamics. The resulting distance estimates temporal reaching cost, and its change across a transition defines goal-relative temporal progress. We use this progress signal as a temporal critic for direct goal-conditioned policy learning. Model-assisted targets provide an additional train
    
[^322]: GraphOPD：面向大语言模型智能体的图增强在线策略蒸馏

    GraphOPD: Graph-Augmented On-Policy Distillation for LLM Agents

    [https://arxiv.org/abs/2610.08959](https://arxiv.org/abs/2610.08959)

    GraphOPD是首个将基于图的结构增强引入大语言模型智能体在线策略蒸馏的方法，解决了传统依据师生分歧分配指导的方式在多轮决策场景中失效的问题。

    

    摘要：在线策略蒸馏通过在强化学习奖励稀疏、且每条轨迹仅在结束时反馈一次奖励的情况下，从教师策略提供密集的、步骤级别的指导，来对大语言模型智能体进行后训练。现有实现方式依据每一步中教师与学生之间的分歧大小来分配这种指导，其背后的单轮直觉是：大的分歧标志着值得纠正的错误。然而，一旦决策在多轮交互中链式展开，这一规则便会失效，因为早期的偏差会进入两个策略后续共同依赖的上下文中，使教师与已发生偏差的轨迹保持一致，而非指出偏差的根源；同时，可互换的步骤反而会产生较大但与结果无关的分歧。我们在一个智能体基准上验证了这一点：蒸馏分歧最大的步骤相比随机选择并未带来一致的收益。为此，我们提出了GraphOPD，这是首个引入基于图的结构增强的方法……

    arXiv:2610.08959v1 Announce Type: new  Abstract: On-policy distillation post-trains large language model agents by supplying dense, step-level guidance from a teacher policy when the reinforcement-learning reward is sparse and arrives only once per trajectory. Existing instantiations allocate this guidance by the size of the teacher-student divergence at each step, on the single-turn intuition that a large disagreement marks a mistake worth correcting. Once decisions chain over many turns, that rule misfires, since an early drift enters every later context both policies condition on, leaving the teacher consistent with the drifted trajectory instead of flagging its cause, while interchangeable steps register large but outcome-irrelevant divergences. We demonstrate this on an agentic benchmark, where distilling the highest-divergence steps brings no consistent benefit over random selection. To this end, we introduce GraphOPD, the first method to bring graph-based structural augmentation
    
[^323]: CARE：面向视觉-语言-动作推理的加速认证

    CARE: Certifying Acceleration for Vision-Language-Action Inference

    [https://arxiv.org/abs/2610.08917](https://arxiv.org/abs/2610.08917)

    提出CARE方法，通过在相同初始条件下的成对回放和有限样本保证，为视觉-语言-动作模型的加速推理提供可认证的加速器选择，揭示并控制被平均指标掩盖的加速诱发任务失败。

    

    尽管视觉-语言-动作（VLA）模型发展迅速，但在每个控制步骤上运行它们仍然代价高昂。先前的工作采用动作分块和视觉token剪枝等技术来加速VLA推理，通常基于延迟和平均任务成功率进行评估。然而，加速可能会丢弃信息，并破坏原始策略本可完成的任务——这一风险被平均指标所掩盖。衡量这些失败十分困难，因为动作偏差会在闭环轨迹中不断累积，这意味着任务失败只有在完整回合中才能被观察到。因此，我们通过在相同初始条件下进行成对回放来定义加速诱发的失败，追踪参考策略成功而加速策略失败的情形。为了解决这一问题，我们提出了CARE，一种用于认证加速器选择的方法。CARE在校准集上使用成对回放，为加速诱发的失败提供有限样本保证。

    arXiv:2610.08917v1 Announce Type: new  Abstract: While vision-language-action (VLA) models have advanced rapidly, running them at every control step remains expensive. Prior work accelerates VLA inference using techniques like action chunking and visual-token pruning, typically evaluating based on latency and average task success. However, acceleration may discard information and break tasks the original policy would solve, a risk hidden by average metrics. Measuring these failures is challenging because action deviations compound over closed-loop trajectories, meaning task failure is only observable across full episodes. We therefore define an acceleration-induced failure via paired rollouts from identical initial conditions, tracking when the reference succeeds but the accelerated policy fails. To manage this, we introduce CARE, an approach for certified accelerator selection. CARE uses paired rollouts on a calibration set to provide finite-sample guarantees that acceleration-induced
    
[^324]: 任务充分收缩：面向机器信息接口的信源选择

    Task-Sufficient Contraction: Source Selection for Machine Information Interfaces

    [https://arxiv.org/abs/2610.08884](https://arxiv.org/abs/2610.08884)

    本文提出“任务充分收缩”这一新性质，证明当机器的动作集合与损失函数固定时，仅凭任务声明就能在选定失真目标之前对信源状态进行合并简化，且由此得到的简化信源能精确保持下游完整的一步码率-后悔曲线。

    

    arXiv:2610.08884v1 通告类型：交叉 摘要：在下游编码器、码本、码率、失真目标或优化器被选定之前，一个明确声明的任务有时就能预先证明某个简化信源的合理性。本文研究了这种简化在何时能够保持完整的下游问题族，这一性质被称为“任务充分收缩”。该简化信源在后续工作点被选定之前即由任务所固定。一个精确的收缩意味着后续问题可以在该简化信源上求解，其结果与保留完整信源时完全相同。对于具有固定可能动作集合和固定损失的机器，本文通过仅在两个状态对所有可用动作产生相同后悔值时才合并状态，从而识别出一个面向特定消费者的信源。对于有限动作集合，用该简化信源替换原本更丰富的信源可以完整保持一步码率-后悔曲线，即使该简化是在失真目标被选定之前就已固定。第二个结果给出了精确的刻画……

    arXiv:2610.08884v1 Announce Type: cross  Abstract: A declared task can sometimes certify a reduced source before a downstream encoder, codebook, rate, distortion target, or optimizer is chosen. This paper studies when one such reduction preserves the complete downstream problem family, a property termed Task-Sufficient Contraction. The reduced source is fixed by the task before the later operating point is selected. An exact contraction allows the later problem to be solved on that source with the same result as if the full source had been retained.   For a machine with a fixed set of possible actions and a fixed loss, the paper identifies a consumer-specific source by merging states only when every available action has the same regret in both. For finite action sets, replacing the richer source by this reduced source preserves the complete one-step rate-regret curve, even though the reduction is fixed before the distortion target is chosen. A second result gives an exact characterizat
    
[^325]: Wasserstein空间中光滑势-相互作用能量的信赖域优化

    Trust-Region Optimization for Smooth Potential-Interaction Energies in Wasserstein Space

    [https://arxiv.org/abs/2610.08883](https://arxiv.org/abs/2610.08883)

    该论文提出了Wasserstein空间上光滑势-相互作用能量的信赖域优化方法，通过推前曲线上的二次模型、$L^2(\rho)$ 步长半径以及带显式自伴二阶变分算子的Steihaug-Toint子求解器，在温和条件下证明了目标函数单调不增且Wasserstein梯度范数收敛于零。

    

    寻找相互作用粒子的低能量构型以及逼近概率分布，都会导致在Wasserstein空间中最小化势-相互作用能量。这些能量可能是非凸的，因此在利用二阶信息的同时控制局部近似的可靠性变得十分重要。我们研究了在具有有限二阶矩的概率测度的Wasserstein空间上，光滑势-相互作用能量的信赖域优化。该方法使用沿推前曲线的二次模型、$L^2(\rho)$ 步长半径，以及带有显式自伴二阶变分算子的Steihaug-Toint子求解器。比值检验决定步骤的接受与否并指导半径更新。在能量存在下界、且势函数与相互作用核的Hessian全局有界的条件下，我们证明了目标函数单调不增、Wasserstein梯度范数收敛于零，并（得到ε-驻点，摘要原文在此处截断）。

    arXiv:2610.08883v1 Announce Type: cross  Abstract: Finding low-energy configurations of interacting particles and approximating probability distributions lead to the minimization of potential-interaction energies in Wasserstein space. These energies can be nonconvex, making it important to exploit second-order information while controlling the reliability of local approximations. We study trust-region optimization of smooth potential-interaction energies on the Wasserstein space of probability measures with finite second moment. The method uses a quadratic model along pushforward curves, an $L^2(\rho)$ step radius, and a Steihaug-Toint subsolver with an explicit self-adjoint second-variation operator. A ratio test determines acceptance and guides the radius update. Under a lower energy bound and globally bounded Hessians of the potential and interaction kernel, we prove that the objective is nonincreasing, the Wasserstein-gradient norms converge to zero, and an $\varepsilon$-stationary
    
[^326]: FinVector-Market-4B：面向结构化金融任务的LoRA适配对照研究

    FinVector-Market-4B: A Controlled Study of LoRA Adaptation for Structured Financial Tasks

    [https://arxiv.org/abs/2610.08882](https://arxiv.org/abs/2610.08882)

    本文通过对照实验证明，对40亿参数模型进行秩16的LoRA适配可显著提升结构化金融任务表现（如JSON有效率、FinQA精确匹配、计算器表达式正确率等），同时揭示了基准中标签集合变化和数据重叠对泛化性结论的限制。

    

    FinVector-Market-4B在包含22,000个样本的语料库上，使用秩为16的LoRA对Qwen/Qwen3.5-4B进行适配，用于结构化金融任务。我们在隐式和显式JSON模式约定下，在同一个600个样本的基准上对基础模型和适配后模型进行了评估。仅提供JSON模式即可将基础模型的JSON有效率从0%提升至91.3%。在匹配的显式提示条件下，冻结模型的成绩提升如下：FinQA答案精确匹配从14.7%提升至40.0%，计算器表达式正确率从48.0%提升至82.7%，情景分支标签一致性从20.1%提升至89.5%，蕴含方向一致性从52.4%提升至87.2%。一项事后策略评分审计显示，所报告的macro-F1下降源于标签集合的变化；使用相同的三类目标类别时，基础模型为77.4%，适配模型为83.1%。财报文件重叠和计算器目标不一致等问题对基准的泛化性结论构成了限制。结果表明，紧凑型金融领域……（原文截断）

    arXiv:2610.08882v1 Announce Type: cross  Abstract: FinVector-Market-4B adapts Qwen/Qwen3.5-4B with rank-16 LoRA on a 22,000-example corpus for structured financial tasks. We evaluate the base and adapted models on the same 600-example benchmark under implicit and explicit JSON-schema contracts. Supplying the schema alone raises base-model JSON validity from 0% to 91.3%. Under matched explicit prompting, the frozen scores improve from 14.7% to 40.0% for FinQA answer exact match, from 48.0% to 82.7% for calculator-expression correctness, from 20.1% to 89.5% for scenario branch-label agreement, and from 52.4% to 87.2% for implication-direction agreement. A post-hoc policy-scoring audit shows that the reported macro-F1 decline reflects a changing label set; using the same three target classes gives 77.4% for the base and 83.1% for the adapter. Filing overlap and calculator-target inconsistencies qualify the benchmark's generalization claims. The results show that compact financial domain a
    
[^327]: 一项关于智能体技能下游效用的实证研究

    An Empirical Study of Agent Skills' Downstream Utility

    [https://arxiv.org/abs/2610.08875](https://arxiv.org/abs/2610.08875)

    本文通过在87个SkillsBench任务上的实证研究，将智能体技能的下游效用量化为相对于无技能基线的通过率差异，并揭示效用如何取决于技能内容、执行配置与多技能组织方式。

    

    智能体技能将程序性指导与资源打包以便复用，但一个相关的技能并不一定能提升任务表现。现有研究刻画了技能内容并评估了下游性能，但对于效用如何取决于内容、执行配置和多技能组织方式，所提供的解释仍然有限。我们在87个SkillsBench任务上开展实证研究，将下游效用定义为在相同的模型-测试框架配置下，同一任务相对于无技能基线的通过率差异。我们在九种配置下比较相同的技能，随后在三种选定配置下考察其他已发布的技能以及固定技能集的不同组织方式。我们从包含37,596个技能的精选语料库中检索市场候选技能。通过借助大语言模型对内容、执行轨迹和最终产物进行分析，并由作者复核，我们将所提供的支持与实际使用情况及任务结果关联起来。相同的技能……

    arXiv:2610.08875v1 Announce Type: cross  Abstract: Agent Skills package procedural guidance and resources for reuse, but a relevant Skill does not necessarily improve task performance. Existing studies characterize Skill content and evaluate downstream performance, yet provide limited explanations of how utility depends on content, execution configuration, and multi-Skill organization. We conduct an empirical study on 87 SkillsBench tasks, defining downstream utility as the pass-rate difference from No-Skill on the same tasks under the same model--harness configuration. We compare the same Skills across nine configurations, then examine alternative published Skills and organizations of fixed Skill sets under three selected configurations. We retrieve marketplace candidates from a curated corpus of 37,596 Skills. LLM-assisted analysis of content, execution traces, and final artifacts, followed by author review, relates provided support to actual use and task outcomes. The same Skills he
    
[^328]: 慢胜快于Kesten-Stigum阈值：稀疏随机块模型中信息-计算差距的极小极大、Fisher信息与置信传播刻画

    Slow Beats Fast at the Kesten-Stigum Threshold: Minimax, Fisher-Information and Belief-Propagation Characterizations of the Information-Computation Gap in Sparse Stochastic Block Models

    [https://arxiv.org/abs/2610.08872](https://arxiv.org/abs/2610.08872)

    该论文通过统计决策理论、Fisher信息和置信传播对稀疏随机块模型的Kesten-Stigum阈值给出三种刻画，证明在 q≥5 时阈值下方存在信息-计算差距：多项式低度算法渐近无法超越平凡风险，而指数时间算法却能成功。

    

    我们通过统计决策理论和Fisher信息，研究具有q个社区、平均度d和信号强度λ的稀疏对称随机块模型中的社区恢复问题，并得到Kesten-Stigum阈值 dλ²=1 及其下方信息-计算差距的三种刻画。首先，在任意给定的社区规模分布上，任何在对称平均和顶点重标记下封闭的规则类的极小极大风险等于其在均匀先验下的贝叶斯风险；后验均值是唯一的贝叶斯规则且是可容许的，而度数为D的多项式规则的贝叶斯风险等于平凡风险乘以 1-Corr_D²。结合已知的低度方法和信息论结果，这一差距被表述为一个最坏情形的命题：对于 q≥5，在阈值下方存在一个窗口区域，其中任何低度规则都无法渐近地超越平凡风险，而指数时间算法则可以在某些标签集上实现超越。

    arXiv:2610.08872v1 Announce Type: cross  Abstract: We study community recovery in the sparse symmetric stochastic block model with $q$ communities, average degree $d$ and signal strength $\lambda$ through statistical decision theory and Fisher information, and obtain three characterizations of the Kesten-Stigum threshold $d\lambda^2=1$ and of the information-computation gap below it. First, on each community-size profile the minimax risk of any class of rules closed under averaging and vertex relabeling equals its Bayes risk under the uniform prior; the posterior mean is the unique Bayes rule and is admissible, and the Bayes risk of degree-$D$ polynomial rules is the trivial risk times $1-\mathrm{Corr}_D^2$. Combined with known low-degree and information-theoretic results, this gives the gap as a worst-case statement: for $q\ge 5$ there is a window below the threshold in which no low-degree rule beats the trivial risk asymptotically, while an exponential-time rule does on a set of labe
    
[^329]: 受治理信用评分中的学习型单调循环特征：框架的代价与宏观条件化的必要性

    Learned Monotone Recurrent Features in Governed Credit Scoring: The Price of the Frame and the Necessity of Macro Conditioning

    [https://arxiv.org/abs/2610.08869](https://arxiv.org/abs/2610.08869)

    该论文证明了在受治理的信用评分中，学习型单调循环特征的价值随治理框架严格程度的提高而上升，且这些特征的循环门控必须进行外生宏观条件化才能保持其价值。

    

    受监管的信用评分要求评分结果对每个风险敞口输入单调非递减。已部署的流程——将手工构建的单调聚合量输入符号受限的梯度提升模型——已通过组合方式满足该要求；悬而未决的问题是：在这样的流程中，学习型时间聚合的价值究竟有多大。我们在五个生产规模的信用数据集上、在匹配的可接纳性条件下（除一个按定价基线惯例处理的数据集外）回答了这一问题；我们采用一种单调循环架构，并通过证明将其逐输入保证扩展至向量值输入，以及外生宏观条件化的衰减门、严重度、阈值与峰值内存。由此得出两个发现。第一，严格性阶梯：学习型单调特征的价值随治理框架的严格程度而上升——在无约束的工程化特征面板上价值为零，而在仅有汇总统计量的框架中价值最大——该结论在两个数据集和一个官方时间稳定性指标上得到复现，不过无（宏观）条件化的特征在 ext……（原文摘要在此处截断）

    arXiv:2610.08869v1 Announce Type: cross  Abstract: Regulated credit scoring requires scores monotone non-decreasing in every exposure input. Deployed pipelines -- hand-crafted monotone aggregates feeding sign-constrained gradient boosting -- already meet this by composition; the open question is what learned temporal aggregation is worth inside one. We answer on five production-scale credit datasets at matched admissibility (one priced baseline convention excepted), with a monotone recurrent architecture whose per-input guarantee we extend, with proofs, to vector-valued inputs and to exogenously macro-conditioned decay gates, severities, thresholds, and peak memory. Two findings result. First, a strictness ladder: the value of learned monotone features rises with governance-frame strictness -- zero on unconstrained engineered panels, maximal in summaries-only frames -- replicated across two datasets and an official temporal-stability metric, though unconditioned features degrade on ext
    
[^330]: 面向稀疏视角与有限角度CT的几何感知扩散近似后验采样

    Geometry-Aware Diffusion Approximate Posterior Sampling for Sparse-View and Limited-Angle CT

    [https://arxiv.org/abs/2610.08866](https://arxiv.org/abs/2610.08866)

    该论文提出一种几何感知的扩散近似后验采样方法，利用CT采集中连续变化的测量灵敏度来联合引导稀疏视角和有限角度CT重建中的更新方向与随机探索，以应对严重病态性带来的重建歧义。

    

    稀疏视角计算机断层扫描（CT）可降低辐射剂量和采集时间，并可能减轻运动伪影。然而，角度欠采样提供的信息不足以唯一且稳定地确定图像。有限角度CT源于受限的角度覆盖范围，会产生与缺失角度范围相关的强烈方向性信息损失。在这两种情况下，图像方向可能被强烈观测、弱约束或完全不可观测，导致严重的病态性和重建歧义。现有的基于扩散的方法通过似然引导、数据一致性操作或值域-零空间校正来融合测量信息。然而，这些方法通常没有利用采集过程中连续变化的测量灵敏度来共同塑造重建更新与随机探索。我们提出了一种几何感知的扩散引导随机重建框架

    arXiv:2610.08866v1 Announce Type: cross  Abstract: Sparse-view computed tomography (CT) reduces radiation dose and acquisition time and may mitigate motion artifacts. However, angular undersampling provides insufficient information to determine the image uniquely and stably. Limited-angle CT, arising from restricted angular coverage, produces strongly directional information loss associated with the missing angular range. In both settings, image directions may be strongly observed, weakly constrained, or unobservable, leading to severe ill-posedness and reconstruction ambiguity. Existing diffusion-based approaches incorporate measurement information through likelihood guidance, data-consistency operations, or range-null-space corrections. However, they do not generally use the continuously varying measurement sensitivity of the acquisition to jointly shape both reconstruction updates and stochastic exploration.   We propose a geometry-aware diffusion-guided stochastic reconstruction fr
    
[^331]: 用于端口扫描规避的对抗性强化学习：边缘部署入侵检测系统中的攻击者特征可见性

    Adversarial RL for Port-Scan Evasion: Attacker Feature Visibility in Edge-Deployed IDS

    [https://arxiv.org/abs/2610.08864](https://arxiv.org/abs/2610.08864)

    该论文提出用深度Q网络（DQN）自适应攻击者来规避部署在树莓派边缘设备上的机器学习入侵检测系统，通过调整探测时序、TCP标志和负载大小，在黑盒、灰盒和白盒不同特征可见性设置下实现61.9%至98.3%的规避率，并发现攻击者特征可见性的增加并不总能单调提升规避效果。

    

    基于机器学习的入侵检测系统（IDS）越来越多地应用于资源受限的物联网（IoT）环境中，然而其鲁棒性通常只是针对静态攻击进行评估，而非针对能够根据检测反馈进行自适应调整的对手。本文研究了针对部署在Raspberry Pi 3B+上的基于机器学习的IDS模型的自适应端口扫描规避。我们实现了一个基于Zeek的实时IDS流水线，其中包含在TON_IoT遥测数据上训练的XGBoost、多层感知机和一维卷积神经网络，并使用深度Q网络（DQN）对手在黑盒、灰盒和白盒三种特征可见性设置下学习探测时序、TCP标志和负载大小的规避组合。尽管部署的IDS模型对传统端口扫描的检测率达91.1%至99.8%，但DQN在最后50个回合中的规避率在不同特征可见性设置下为61.9%至98.3%。更大的特征可见性并不会单调地提升规避（原文摘要在此处截断）。

    arXiv:2610.08864v1 Announce Type: cross  Abstract: Machine learning-based intrusion detection systems (IDS) are increasingly used in resource-constrained Internet of Things (IoT) environments, yet their robustness is often evaluated against static attacks rather than adversaries that adapt to detection feedback. This paper investigates adaptive port-scan evasion against ML-based IDS models deployed on a Raspberry Pi 3B+. We implement a live Zeek-based IDS pipeline with XGBoost, a multi-layer perceptron, and a 1D convolutional neural network trained on TON_IoT telemetry, and use a Deep Q-Network (DQN) adversary to learn evasive combinations of probe timing, TCP flags, and payload size under black-box, gray-box, and white-box feature-visibility settings. Although the deployed IDS models detect conventional port scans at 91.1--99.8%, DQN final-50-episode evasion rates range from 61.9% to 98.3% across feature-visibility settings. Greater feature visibility does not monotonically improve ev
    
[^332]: LRCC：用条件计算泛化低秩压缩

    LRCC: Generalizing Low-Rank Compression with Conditional Computation

    [https://arxiv.org/abs/2610.08858](https://arxiv.org/abs/2610.08858)

    LRCC通过为每个Transformer块训练轻量级路由器在嵌套低秩路径间动态选择，在训练时冻结低秩因子仅优化路由器，在相同的平均活跃参数预算下性能超越静态低秩压缩，在Llama-2-7B上平均下游准确率提升7.6个百分点。

    

    低秩压缩通过用低秩分解替换线性变换来降低预训练语言模型的成本。然而，传统方法在推理时使用固定的秩分配，无论输入词元是什么，都分配相同的计算量。我们提出了低秩条件计算（LRCC），通过为每个Transformer块训练一个轻量级路由器，在一小组嵌套的低秩路径中进行选择，从而为预训练模型引入依赖于词元的计算。在训练过程中，低秩因子保持冻结，仅对路由器进行优化。我们在Llama和Qwen模型上评估了LRCC在语言建模和零样本下游任务上的表现。在相同的平均活跃参数预算下，LRCC的预测性能优于静态低秩压缩，其中在Llama-2-7B上平均下游准确率相比静态方法提升了7.6个百分点。在匹配的批大小为1的解码（原文摘要在此处截断）。

    arXiv:2610.08858v1 Announce Type: new  Abstract: Low-rank compression reduces the cost of pretrained language models by replacing linear transformations with low-rank factorizations. However, conventional methods use a fixed rank allocation during inference, assigning the same amount of compute regardless of the input token. We introduce Low-Rank Conditional Computation (LRCC), which adds token-dependent computation to pretrained models by training one lightweight router per Transformer block to select among a small set of nested low-rank paths. During training, the low-rank factors remain frozen, and only the routers are optimized. We evaluate LRCC on Llama and Qwen models for language modeling and zero-shot downstream tasks. Within the same average active-parameter budget, LRCC improves the predictive performance over static low-rank compression, including a 7.6 percentage-point gain in average downstream accuracy on Llama-2-7B over static methods. At matched batch-size-1 decoding la
    
[^333]: 基于模型的强化学习实现自主液滴导航：零样本迁移与涌现动力学

    Autonomous Droplet Navigation via Model-Based Reinforcement Learning: Zero-Shot Transfer and Emergent Dynamics

    [https://arxiv.org/abs/2610.08852](https://arxiv.org/abs/2610.08852)

    该研究提出了首个基于模型强化学习的机器人平台，实现了开放表面上液滴的闭环自主导航，仅需50至150次物理实验即可完成策略训练而无需仿真或解析模型，并展现出零样本迁移与涌现动力学特性。

    

    自动驾驶实验室正在通过闭环自动化变革化学与材料发现，然而针对柔软、可变形物质进行物理操作的自动化基础设施仍超出当前机器人平台的能力范围。一个关键的实例是开放表面上的自主液滴输运，其中接触角滞后、毛细钉扎和表面异质性会产生部分可观测的动力学，这对经典的基于模型的控制器构成了重大挑战。我们提出了首个利用基于模型的强化学习在开放、无约束表面上实现闭环自主液滴导航的机器人平台。一块涂有薄硅油膜的双轴倾斜板驱动液滴运动，同时顶置摄像头提供实时反馈。学习得到的策略仅通过50至150次物理实验回合进行训练（取决于几何复杂度），无需仿真或解析模型。此外……（摘要在此处截断）

    arXiv:2610.08852v1 Announce Type: cross  Abstract: Self-driving laboratories (SDLs) are transforming chemical and materials discovery through closed-loop automation, yet automated infrastructure for physical manipulation of soft, deformable matter remains beyond current robotic platforms. A critical instance is autonomous droplet transport on an open surface, where contact-angle hysteresis, capillary pinning, and surface heterogeneity produce partially observable dynamics that pose significant challenges for classical model-based controllers. We introduce the first robotic platform for closed-loop autonomous liquid droplet navigation on an open, unconfined surface using model-based reinforcement learning. A two-axis tilting board coated with a thin silicone oil film drives the droplet, while an overhead camera provides real-time feedback. A learned policy was trained on just 50 to 150 physical episodes depending on geometric complexity, without simulation or analytical models. Beyond p
    
[^334]: 超越风险预测：面向可解释自杀风险评估的证据定位与心理社会因素验证

    Beyond Risk Prediction: Evidence Grounding and Psychosocial Factor Verification for Explainable Suicide Risk Assessment

    [https://arxiv.org/abs/2610.08842](https://arxiv.org/abs/2610.08842)

    该研究提出了一个包含风险评估、证据定位与双验证器因素识别的可解释自杀风险评估框架，通过基于长度的路由、风险-证据一致性约束以及分类验证器与证据感知验证器的结合，超越单纯的风险分类，实现了对预测背后文本证据与心理社会因素的可解释性分析。

    

    从社交网络服务（SNS）帖子中识别自杀风险，对于检测在线环境中的自杀相关信号非常重要。然而，仅靠风险分类对预测背后的文本证据和心理社会因素所能提供的洞察十分有限。基于IEEE BigData 2026可解释自杀风险检测挑战赛，本研究提出了一个由风险评估、证据定位和因素识别三部分组成的框架。风险评估采用基于长度的路由机制，以适应不同长度的帖子。证据定位负责识别支持性短语，并通过风险-证据约束来保持与风险预测的一致性。在因素识别方面，使用了两个验证器：分类验证器专注于因素语义，而证据感知验证器则利用因素特定的词汇-语义线索来筛选信息丰富的正训练单元。两个验证器的预测概率被组合（原文摘要在此处截断）。

    arXiv:2610.08842v1 Announce Type: new  Abstract: Identifying suicide risk from social networking services (SNS) posts is important for detecting suicide-related signals in online environments. However, risk classification alone provides limited insight into the textual evidence and psychosocial factors behind a prediction. Based on the IEEE BigData 2026 Explainable Suicide Risk Detection Challenge, this study presents a framework consisting of Risk Assessment, Evidence Grounding, and Factor Identification. Risk Assessment uses length-based routing to accommodate posts of different lengths. Evidence Grounding identifies supporting phrases and uses a Risk-Evidence constraint to maintain consistency with the Risk prediction. For Factor Identification, two verifiers are used. The Taxonomy Verifier focuses on factor semantics, whereas the Evidence-Aware Verifier uses factor-specific lexical-semantic cues to select informative positive training units. Their prediction probabilities are combi
    
[^335]: 超越谄媚分数：任务、模型与压力如何塑造大语言模型的让步行为

    Beyond the Sycophancy Score: How Task, Model, and Pressure Shape LLM Yielding

    [https://arxiv.org/abs/2610.08840](https://arxiv.org/abs/2610.08840)

    该研究通过对103,939条回复的大规模实验发现，LLM的谄媚行为主要由任务验证代价和护栏覆盖情况决定，而非模型家族或用户压力策略——锚定事实几乎不被让步（1.3%），而逻辑谜题等更易被用户诱导改口。

    

    大语言模型（LLM）在用户提出异议时，常常会放弃原本正确的答案，或转而认同用户的立场。这种行为被称为“谄媚”（sycophancy），通常以每个模型单一的谄媚率来报告，但这几乎无法说明该行为何时发生，以及用户如何才能避免它。我们通过103,939条经过评分的回复研究了产生这种行为的条件：这些回复来自十种配置——八个关闭推理功能的LLM，以及其中两个再次以最大推理能力运行的配置——它们面对相同的200个题目、13种压力条件和四轮对话，每条回复均由两个独立的LLM评判者进行标注。我们发现，最主要的决定因素是模型验证用户主张的代价大小，以及是否存在覆盖该任务的训练护栏。从逻辑斯蒂回归模型中移除这一任务因素会使McFadden R²下降0.485，相比之下模型家族为0.139，压力策略仅为0.009。锚定的事实几乎从不会被让步（1.3%），而逻辑谜题上的迎合采纳率则会上升……

    arXiv:2610.08840v1 Announce Type: new  Abstract: Large language models (LLMs) often abandon a correct answer, or endorse a user's position, once the user pushes back. This behavior, called sycophancy, is usually reported as a single rate per model, which says little about when it happens or how a user can avoid it. We study the conditions that produce it with 103,939 graded replies from ten configurations: eight LLMs with reasoning disabled, and two of them again with maximum reasoning, all facing the same 200 items, 13 pressure conditions, and four-turn conversations, with every reply labeled by two independent LLM judges. We find that the dominant factors are how costly it is for the model to verify the user's claim, and whether a trained guardrail covers it. Removing this task factor from a logistic model costs 0.485 of McFadden $R^2$, against 0.139 for model family and 0.009 for pressure tactic. Anchored facts are almost never conceded (1.3%), while adoption on logic puzzles rises 
    
[^336]: CoDR：面向扩散语言模型的无训练置信度漂移重掩码方法

    CoDR: Training-Free Confidence-Drift Remasking for Diffusion Language Models

    [https://arxiv.org/abs/2610.08833](https://arxiv.org/abs/2610.08833)

    CoDR 提出了一种无需训练、与采样器无关的置信度漂移重掩码方法，通过检测已提交词元的置信度下降并仅对模型不再认可的词元进行重掩码和重新生成，有效防止了掩码扩散语言模型解码过程中早期错误的传播。

    

    掩码扩散语言模型（MDLM）通过反复将词元提交到掩码位置来进行解码，但这些提交通常是不可逆的。在稀疏、不完整上下文下选定的词元会被固定保留，即使后续更完整的上下文已不再支持它。现有的采样器主要决定何时提交一个词元，却很少检查已提交的词元是否仍应保留，从而使早期错误得以传播。我们将这一问题归因于置信度漂移，即模型对已提交词元的置信度从提交时的稀疏上下文到后来更密集的上下文之间出现下降。基于这一信号，我们提出了 CoDR（置信度漂移重掩码），这是一种无需训练且与采样器无关的精修过程。CoDR 通过 k 分区探测，仅需 k 次前向传播即可估计所有已提交位置的漂移，然后仅对模型不再认可的词元进行重掩码和重新生成。在两个骨干模型、四个推理和编码任务上的实验表明……（摘要原文在此处截断）

    arXiv:2610.08833v1 Announce Type: new  Abstract: Masked diffusion language models (MDLMs) decode by repeatedly committing tokens to masked positions, but these commitments are usually irreversible. A token chosen under sparse, partial context is kept fixed, even when later context no longer supports it. Existing samplers mainly decide when to commit a token, but rarely check whether an already committed token should still be kept, allowing early mistakes to propagate. We trace this issue to confidence drift, where the model's confidence in a committed token drops from its sparse commit-time context to the denser context available later. Based on this signal, we propose CoDR (Confidence Drift Remasking), a training-free and sampler-agnostic refinement pass. CoDR estimates drift for all committed positions in only k forward passes via k-partition probing, then remasks and regenerates only the tokens the model no longer endorses. Across two backbones, four reasoning and coding tasks, and 
    
[^337]: 兼顾成人性能保留的儿童语音识别适配：一项实证研究

    Child ASR Adaptation with Adult Retention: An Empirical Study

    [https://arxiv.org/abs/2610.08827](https://arxiv.org/abs/2610.08827)

    该实证研究在阿拉伯语和英语上系统比较了全量微调、LoRA与权重空间合并等儿童ASR适配方法，发现儿童语音适配虽有必要但常导致成人语音识别性能遗忘，而双语适配比单语言适配更稳定，能更好地平衡儿童适配与成人保留。

    

    自动语音识别（ASR）系统在儿童和非母语使用者上的表现往往较差，而将成人ASR模型适配到儿童语音上又会引发“成人语音遗忘”问题。我们在阿拉伯语和英语场景下研究了如何在进行儿童ASR适配的同时保留成人语音识别性能。我们比较了全量微调、LoRA和事后权重空间合并三种方法，涵盖编码器-解码器、编码器-CTC以及基于AudioLLM的ASR系统。实验使用了阿拉伯语母语及非母语儿童语音、英语MyST儿童语音，以及来自MGB-2和LibriSpeech test-clean的成人基准数据。我们使用词错误率（WER）评估识别质量，并通过保留指数、儿童适配增益和适配恢复率来量化适配与保留之间的权衡。结果表明，儿童语音适配是必要的，尤其是对于非母语阿拉伯语和英语儿童语音，但直接适配往往会降低成人ASR的性能。双语适配比单语言适配更加稳定。权重……（原文摘要在此处截断）

    arXiv:2610.08827v1 Announce Type: new  Abstract: Automatic Speech Recognition (ASR) systems often underperform for children and non-native speakers, while adapting adult ASR models to child speech can cause adult-speech forgetting. We study child ASR adaptation with adult retention across Arabic and English. We compare full fine-tuning, LoRA, and post-hoc weight-space merging across encoder--decoder, encoder--CTC, and AudioLLM-based ASR systems. Experiments use Arabic native and non-native child speech, English MyST child speech, and adult benchmarks from MGB-2 and LibriSpeech test-clean. We evaluate recognition quality with WER and quantify the adaptation--retention trade-off using Retention Index, Child Adaptation Gain, and Adaptation Recovery. Results show that child adaptation is necessary, especially for non-native Arabic and English child speech, but direct adaptation often reduces adult ASR performance. Bilingual adaptation is more stable than language-specific adaptation. Weigh
    
[^338]: HCPN-GCN：利用锥体几何扩展分层原型网络以实现持续图学习

    HCPN-GCN: Scaling Hierarchical Prototype Networks with Cone Geometry for Continual Graph Learning

    [https://arxiv.org/abs/2610.08823](https://arxiv.org/abs/2610.08823)

    提出HCPN-GCN，通过用图卷积网络替换线性特征提取器并引入基于锥体的原型与多样性正则化机制，在无需存储历史数据的情况下有效缓解持续图学习中的灾难性遗忘问题。

    

    持续图学习（CGL）旨在从图结构数据中进行增量学习，同时保留从先前任务中获得的知识。这一设置中的一个主要挑战是灾难性遗忘，即学习新任务会导致在先前学习任务上的性能下降。分层原型网络（HPN）通过一种基于原型的记忆机制来解决这一问题，该机制无需存储历史数据，但其对线性特征提取器的依赖限制了其利用图拓扑结构的能力，而基于点的原型在结构多样的图上常常导致低效的原型增长。在本工作中，我们提出了HCPN-GCN，这是HPN的一种图感知扩展，它用图卷积网络（GCN）取代了原始的线性特征提取器，并引入了基于锥体的原型以及多样性正则化目标。所提出的设计能够产生更丰富的图感知表示，同时对（摘要在此处被截断）

    arXiv:2610.08823v1 Announce Type: new  Abstract: Continual Graph Learning (CGL) aims to incrementally learn from graph-structured data while preserving knowledge acquired from previous tasks. A major challenge in this setting is catastrophic forgetting, where learning new tasks degrades performance on previously learned ones. Hierarchical Prototype Networks (HPNs) address this problem through a prototype-based memory mechanism that avoids storing historical data, but their reliance on linear feature extractors limits their ability to exploit graph topology, while point-based prototypes often lead to inefficient prototype growth on structurally diverse graphs. In this work, we propose HCPN-GCN, a graph-aware extension of HPN that replaces the original linear feature extractors with Graph Convolutional Networks (GCNs) and introduces cone-based prototypes with a diversity regularization objective. The proposed design produces richer graph-aware representations while compactly modeling the
    
[^339]: 一种基于车载行驶感知的桥梁数字孪生部署的车辆一体化方法

    A Vehicle-Integrated Approach to Digital Twin Deployment for Bridges Through Drive-By Sensing

    [https://arxiv.org/abs/2610.08822](https://arxiv.org/abs/2610.08822)

    该论文提出了一种融合基于物理的建模与机器学习的车辆一体化数字孪生框架，通过傅里叶神经算子代理模型利用车载间接传感数据实现对桥梁与路面状态的连续监测，以克服传统结构健康监测系统成本高、难扩展的局限。

    

    老化的桥梁基础设施正日益成为全球关注的问题，然而传统的结构健康监测（SHM）系统成本高昂且难以规模化，常规的目视检查又仍然带有主观性。车载式（或间接式）桥梁检测——即由装配传感器的车辆从车-桥相互作用（VBI）和车-路相互作用（VRI）响应中恢复结构信息——提供了一种可规模化的替代方案。然而，一些关键挑战仍未解决，包括将桥梁响应与路面不平整度分离、在正常交通条件下检测损伤，以及在不同桥梁类型之间的泛化能力。本文提出了一种车辆一体化的数字孪生框架，该框架将基于物理的建模与机器学习相统一，用于桥梁与路面状态的连续监测。该框架包含三大支柱：首先，利用傅里叶神经算子构建VBI和VRI的代理模型，该算子学习函数到函数的映射……（原文摘要在此处截断）

    arXiv:2610.08822v1 Announce Type: new  Abstract: Ageing bridge infrastructure is a growing global concern, yet conventional Structural Health Monitoring (SHM) systems are costly and difficult to scale, and routine visual inspections remain subjective. Drive-by, or indirect, bridge inspection, in which a sensorised vehicle recovers structural information from vehicle-bridge interaction (VBI) and vehicle-road interaction (VRI) responses, offers a scalable alternative. However, key challenges remain unresolved, including separating bridge responses from road roughness, detecting damage under normal traffic, and generalising across diverse bridge types. This paper presents a vehicle-integrated digital twin framework that unifies physics-based modelling and machine learning for continuous monitoring of bridge and road conditions. The framework comprises three pillars. First, surrogate models of VBI and VRI are constructed using a Fourier Neural Operator that learns function-to-function mapp
    
[^340]: 面向任务的关键层KV通信：实现高效的潜在多智能体协作

    Task-Oriented Key-Layer KV Communication for Efficient Latent Multi-Agent Collaboration

    [https://arxiv.org/abs/2610.08820](https://arxiv.org/abs/2610.08820)

    该论文提出无需训练的KITE框架，将多智能体潜在通信的目标从发送端状态保真度转变为接收端任务充分性，通过识别任务有效的关键层并仅传输其潜在工作记忆，大幅降低了通信与计算开销。

    

    基于大语言模型的多智能体系统通过协作来提升复杂问题的求解能力，而潜在通信通过直接传输模型内部状态，避免了自然语言交互带来的高额推理成本。然而，现有的基于KV缓存的潜在通信方法优先考虑发送端状态的保真度，导致大量的通信与计算开销，并可能引入冗余信息。为了解决这些局限，我们从面向任务的视角重新审视潜在通信，将其目标从发送端状态保真度转变为接收端任务充分性。在此设定下，我们提出了KITE——一个无需训练的面向任务的关键层KV通信框架。KITE利用接收端轨迹失真准则识别出任务有效的关键层，仅传输与该关键层相关的潜在工作记忆，并进一步将同一层用作接收端……（原文摘要至此截断）

    arXiv:2610.08820v1 Announce Type: new  Abstract: Large language model-based multi-agent systems improve complex problem solving through collaboration, while latent communication directly transmits model internal states to avoid the high inference costs of natural language. However, existing KV-based latent communication methods prioritize sender-side state fidelity, leading to substantial communication and computation overhead and potentially introducing redundant information. To address these limitations, we revisit latent communication from a task-oriented perspective, shifting its objective from sender-side state fidelity to receiver-side task sufficiency. Under this formulation, we propose KITE, a training-free framework for task-oriented key-layer KV communication. KITE identifies a task-effective key layer using a receiver trajectory distortion criterion, transmits only the latent working memory associated with the key layer, and further uses the same layer as the entry point for
    
[^341]: HydroSphere：一个面向治理型、自愈式污水处理基础设施的框架

    HydroSphere: A Framework for Governed, Self-Healing Wastewater Infrastructure

    [https://arxiv.org/abs/2610.08819](https://arxiv.org/abs/2610.08819)

    HydroSphere是一个将治理机制与自愈能力相结合的数据驱动污水处理框架，它利用混合TCN-LSTM模型对七项水质参数进行多步预测，并通过PPO强化学习自适应优化化学药剂投加，从而实现实时水质监测、处理优化与故障恢复。

    

    快速的工业化与城市化进程正对水质和污水处理系统施加越来越大的压力，而传统污水处理厂通常依赖静态的监测与控制策略，难以适应不断变化的污染物条件。本文提出了HydroSphere，一个具备治理机制、数据驱动的框架，用于实时水质监测、预测、处理优化和故障恢复。HydroSphere使用1940年至2023年间收集的282万条水质测量数据进行评估。该框架集成了三个主要组件：首先，混合TCN-LSTM模型对七项水质参数进行多步预测，实现了0.1417的RMSE、0.1047的MAE和0.3596的R²；其次，自适应剂量优化模块使用PPO强化学习来调整化学药剂投加，实现了1.059的平均步奖励，优于固定剂量基线的1.017。结果表明……

    arXiv:2610.08819v1 Announce Type: new  Abstract: Rapid industrialization and urban growth are increasing pressure on water quality and wastewater treatment systems, while conventional treatment plants often rely on static monitoring and control strategies that cannot easily adapt to changing pollutant conditions. This paper presents HydroSphere, a governed, data-driven framework for real-time water quality monitoring, forecasting, treatment optimization, and fault recovery. HydroSphere is evaluated using 2.82 million water-quality measurements collected between 1940 and 2023. The framework integrates three main components. First, a hybrid TCN-LSTM model performs multi-step forecasting across seven water-quality parameters, achieving an RMSE of 0.1417, MAE of 0.1047, and R2 of 0.3596. Second, the Adaptive Dosage Optimization Module uses PPO reinforcement learning to adjust chemical dosing, achieving a mean step reward of 1.059 compared with 1.017 for a fixed-dose baseline. The results a
    
[^342]: 只为FUNS：基于大语言模型引导的时空图节点生成方法用于预测未观测节点状态

    Just for FUNS: LLM-Guided Spatio-Temporal Graph Node Generation for Forecasting Unobserved Node States

    [https://arxiv.org/abs/2610.08818](https://arxiv.org/abs/2610.08818)

    该论文提出GenST框架，将未观测节点状态预测（FUNS）重新定义为时空图上的条件生成任务，创新性地利用微调后的大语言模型从节点描述中提取语义特征作为语义桥梁，以弥补缺失的时空信号。

    

    时空预测是物流、城市规划和智能交通系统的基石。然而，受部署成本和维护资源的限制，传感器网络往往缺乏全面的空间覆盖，这使得预测未观测节点状态（FUNS）成为一项至关重要却又极具挑战性的任务。传统模型依赖历史观测数据，在遇到没有先前记录的节点时通常会表现失常。为解决这一问题，我们将该问题重新定义为时空图上的条件生成任务，并提出GenST框架，该框架引入大语言模型（LLMs）作为语义桥梁，利用经过微调的预训练LLM从节点描述（如功能分区和道路网络结构）中提取丰富的语义特征，以弥补缺失的时空信号。具体而言，我们设计了一个两阶段生成架构：时空变分自编码器（VAE）首先压缩……

    arXiv:2610.08818v1 Announce Type: cross  Abstract: Spatio-temporal forecasting is a cornerstone of logistics, urban planning, and intelligent transportation systems. However, constrained by deployment costs and maintenance resources, sensor networks often lack comprehensive spatial coverage, rendering Forecast Unobserved Node States (FUNS) a critical yet formidable challenge. Conventional models rely on historical observations and typically falter when encountering nodes without prior records. To address this, we redefine the problem as a conditional generation task on spatio-temporal graphs and propose GenST, a framework that introduces Large Language Models (LLMs) as a semantic bridge, leveraging a pre-trained LLM fine-tuned to extract rich semantic features from node descriptions, such as functional zones and road network structures, to compensate for missing spatio-temporal signals. Specifically, we design a two-stage generative architecture: a Spatio-Temporal VAE first compresses 
    
[^343]: DenoFlow：面向真实生理伪迹下SSVEP去噪的流匹配方法

    DenoFlow: Flow Matching for SSVEP Denoising under Real Physiological Artifacts

    [https://arxiv.org/abs/2610.08817](https://arxiv.org/abs/2610.08817)

    DenoFlow将SSVEP去噪建模为基于整流流的输运问题，通过场网络回归受污染信号与干净信号之间直线路径的速度场，在真实生理伪迹下实现去噪并保持信号的可解码性。

    

    基于脑电图（EEG）的脑机接口（BCI），特别是稳态视觉诱发电位（SSVEP）系统，极易受到噪声和伪迹的影响，这会严重降低解码准确率。尽管近期的去噪方法已展现出潜力，但它们在没有成对干净信号的情况下进行拟合，可能退化为简单地复现输入，并且仅以波形距离作为优化目标，而这无法保证输出信号仍然保持可解码性。为解决这些问题，我们提出DenoFlow，它将SSVEP去噪建模为一个输运问题：不再学习从受污染试验到干净试验的直接映射，而是遵循整流流（rectified flow）的公式，由一个场网络回归两者之间直线路径的速度，去噪过程则从观测出发沿该速度场进行前向积分。该场网络采用编码器-解码器结构，在每一层都能看到受污染的试验信号，并在瓶颈处获取路径位置信息……

    arXiv:2610.08817v1 Announce Type: new  Abstract: Electroencephalography (EEG)-based brain-computer interfaces (BCIs), particularly steady-state visual evoked potential (SSVEP) systems, are highly vulnerable to noise and artifacts, which severely degrade decoding accuracy. Although recent denoising approaches have shown promise, they are fitted without paired ground truth, can settle on reproducing their input, and are optimized on waveform distance alone, which says nothing about whether the output stays decodable. To address these issues, we propose DenoFlow, which casts SSVEP denoising as transport: instead of learning a direct map from a contaminated trial to a clean one, a field network regresses the velocity of the straight path between them, following the rectified-flow formulation, and denoising integrates that field forward from the observation. The field network is an encoder-decoder that sees the contaminated trial at every layer and the path position at its bottleneck, and a
    
[^344]: 长记忆的代价：序列模型中的状态、上下文与稳定性复杂度

    The Cost of Long Memory: State, Context, and Stability Complexity in Sequence Models

    [https://arxiv.org/abs/2610.08816](https://arxiv.org/abs/2610.08816)

    该论文针对长记忆序列模型证明了匹配的逼近上下界：达到预测误差 τ 仅需 Θ(log²(1/τ)) 个状态或模式（误差按 e^{-Θ(√r)} 衰减），并揭示真正的分数长记忆会从根本上改变逼近问题的几何结构。

    

    长期时间依赖性为序列模型提出了一个资源问题：对于给定的预测记忆规律，需要多少状态、上下文或动态临界性才能实现准确预测？我们直接在预测风险的框架下研究这个问题。对于代数衰减的预测记忆，我们为指数模式和有限状态模式证明了相互匹配的上下逼近界。最佳的 r-模式预测误差按 $e^{-\Theta(\sqrt r)}$ 衰减，因此要达到预测误差 $\tau$，需要 $r=\Theta(\log^2(1/\tau))$ 个状态或模式。先前的“记忆诅咒”结果在不同的逼近概念下确立了稳定循环模型的广泛局限性；而在这里，上下界针对预测风险中的一个规范预测目标完全匹配，从而确定了该目标的最优资源指数。随后我们证明，真正的分数长记忆会改变逼近的几何结构本身。特别地，预测误差是在分数（原文摘要在此处截断）

    arXiv:2610.08816v1 Announce Type: new  Abstract: Long-range temporal dependence poses a resource question for sequence models: for a specified predictive-memory law, how much state, context, or dynamical criticality is required in order to forecast accurately? We study this question directly in forecasting risk. For algebraically decaying predictive memory, we prove matching upper and lower approximation bounds for exponential and finite-state modes. The best $r$-mode forecast error decays as $e^{-\Theta(\sqrt r)}$, so reaching forecast error $\tau$ needs $r=\Theta(\log^2(1/\tau))$ states or modes. Earlier curse-of-memory results establish broad limitations of stable recurrent models under different approximation notions; here both sides match for one canonical predictive target in forecast risk, which fixes the optimal resource exponent for that target. We then show that genuine fractional long memory changes the geometry itself. In particular, forecast error is measured after fractio
    
[^345]: 面向智能体AI自动化的有界自主性与可验证安全性

    Bounded Autonomy and Verifiable Safety for Agentic AI Enabled Automation

    [https://arxiv.org/abs/2610.08815](https://arxiv.org/abs/2610.08815)

    本文提出BRaVeS安全治理框架，通过将专家约束编码为不变锚点、深度感知访问机制和随认知风险动态调整的状态层级自主性，并结合Lyapunov有界一致性框架实现屏蔽式状态转换，从而为高风险环境中的智能体AI自动化提供有界自主性与可验证的安全保障。

    

    仅依靠概率推理，智能体AI驱动的自动化系统无法在高风险环境中安全部署。一个反复出现的风险是“认知漂移”：随着推理的深入，系统行为可能偏离领域专家为安全运行所设定的约束。本文提出了BRaVeS，一个有界推理与安全治理框架，称为可辩护的新一代推理系统（DNRS）。BRaVeS将领域专家定义的约束编码为不变锚点，提出MoDA-Style（深度混合注意力）深度感知访问作为候选机制，以在推理过程中保持这些锚点可见，并使用状态层级（SMARtAutonomy）在认知风险增加时降低系统自主性。为了形式化有界恢复，我们引入了Lyapunov有界一致性框架（LBCF），该框架将连续的认知风险信号映射到有限的K-bag抽象中，并应用屏蔽状态转换来强制执行Lyapunov式的能量下降。

    arXiv:2610.08815v1 Announce Type: new  Abstract: Agentic AI-enabled automation cannot be safely deployed in high-stakes environments on probabilistic reasoning alone. A recurring risk is epistemic drift: as reasoning deepens, system behavior may move away from subject-matter-expert constraints for safe operation. This paper presents BRaVeS, a bounded reasoning and safety-governance framework termed the Defensible Next-Gen Reasoning System (DNRS). BRaVeS encodes SME-defined constraints as invariant anchors, proposes MoDA-Style (Mixture of Depths Attention) depth-aware access as a candidate mechanism for keeping these anchors visible during inference, and uses a state hierarchy (SMARtAutonomy) to reduce autonomy as epistemic risk increases. To formalize bounded recovery, we introduce the Lyapunov-Bounded Consensus Framework (LBCF), which maps continuous epistemic-risk signals into a finite K-bag abstraction and applies shielded state transitions that enforce Lyapunov-style energy descent
    
[^346]: 路由-验证-投票：面向混合领域推理的程序条件化自洽性方法

    Route-Verify-Vote: Procedure-Conditioned Self-Consistency for Mixed-Domain Reasoning

    [https://arxiv.org/abs/2610.08814](https://arxiv.org/abs/2610.08814)

    提出RVV框架，通过路由、验证、投票三个阶段实现程序条件化自洽性，在无需参数更新的情况下提升语言模型在未见过的混合领域中识别完整正确答案集合的能力。

    

    当语言模型必须以陌生的方式组合熟悉的推理操作时，组合泛化仍然具有挑战性。基于场景的常识推理评测（SCoRE）2026 在三个训练数据中未出现的混合领域上测试这一能力，并要求模型为每个问题识别出完整的正确选项集合。我们提出了路由-验证-投票（RVV）框架，这是一种程序条件化的自洽性方法，无需参数更新即可运用语言模型。路由阶段利用给定的领域标签选择一个推理程序，引导模型表示和应用相关约束；验证阶段提示模型根据这些约束逐项评估每个选项；投票阶段汇总完整的答案集合，并为两个最常见答案集合之间票数差距较小的问题分配额外的采样资源。每个问题的采样均遵循相同的领域特定程序。在官方……

    arXiv:2610.08814v1 Announce Type: cross  Abstract: Compositional generalization remains challenging when language models must combine familiar reasoning operations in unfamiliar ways. The Scenario-Based Commonsense Reasoning Evaluation (SCoRE) 2026 tests this ability on three mixed domains absent from training and requires models to identify the complete set of correct options for each question.   We introduce Route-Verify-Vote (RVV), a framework for procedure-conditioned self-consistency that uses language models without parameter updates. Route uses the provided domain label to select a reasoning procedure that guides the model in representing and applying the relevant constraints. Verify prompts the model to assess each option against those constraints. Vote aggregates complete answer sets and allocates additional samples to questions with a small vote-count margin between the two most frequent sets. Samples for each question follow the same domain-specific procedure.   On the offic
    
[^347]: KVFetch：面向KV缓存压缩“缺失另一半”的时间性预取

    KVFetch: Temporal Prefetching for the Missing Half of KV Cache Compression

    [https://arxiv.org/abs/2610.08811](https://arxiv.org/abs/2610.08811)

    该论文发现现有KV缓存压缩方法只实现了按内容的关联查找，而缺失了按位置的顺序访问能力，导致模型逐字复现上下文内容时中途不可逆地失败，并提出KVFetch时间性预取方法来补全这缺失的另一半。

    

    随着上下文窗口扩展到数万甚至数十万个token，KV缓存压缩已成为实现高效大语言模型推理的关键。现有方法分为三大类：基于分数的驱逐、摘要补偿以及卸载-召回。然而，这三类方法都依据内容与当前查询的相关性来决定保留或召回哪些内容。我们证明这种共同的设计在结构上是不完整的：缓存支持两种访问模式，即按内容的关联查找和按位置的顺序遍历，而当前的压缩器只实现了前者。这种差距在实践中至关重要：检索增强生成、代码补全和结构化数据提取等任务都需要模型从上下文中逐字复现标识符、字段值或代码token。在压缩条件下，基于内容的驱逐会保留此类序列的开头但丢弃其后续部分，导致逐字复制在中途不可逆地中断，我们将这种失败称为顺序性遗忘（原文摘要在此处截断）。

    arXiv:2610.08811v1 Announce Type: new  Abstract: As context windows scale to tens or hundreds of thousands of tokens, KV cache compression has become essential for efficient LLM inference. Existing methods fall into three families: score-based eviction, summary compensation, and offload-and-recall. Yet all three decide what to keep or recall by content relevance to the current query. We show this shared design is structurally incomplete. A cache supports two access modes: associative lookup by content and sequential traversal by position; current compressors implement only the first. The gap matters in practice: retrieval-augmented generation, code completion, and structured-data extraction all require the model to reproduce identifiers, field values, or code tokens verbatim from the context. Under compression, content-based eviction retains the head of such a sequence but discards its continuation, causing verbatim copying to break irreversibly midway, a failure we call sequential for
    
[^348]: Prithvi作物分类基础模型在跨三大洲物候与地理偏移下的可迁移性与运行可靠性

    Transferability and operational reliability of a Prithvi crop classification foundation model under phenological and geographic shift across three continents

    [https://arxiv.org/abs/2610.08810](https://arxiv.org/abs/2610.08810)

    该研究评估了Prithvi-EO-2.0作物分类基础模型在三大洲的分布外性能，发现其精度随地理偏移显著下降（美国0.65降至欧洲0.40），且观测窗口与当地作物物候错位时精度会严重崩溃，揭示了GeoFM实际部署中的可靠性局限。

    

    在大型卫星档案上预训练的微调地理空间基础模型已被证明能够提高作物分类精度和地理可迁移性。然而，其在训练分布之外的运行性能仍缺乏充分表征。我们评估了一个广泛采用的GeoFM（Prithvi-EO-2.0）在三大洲12个国家的37个事件中的分布外性能，并与区域参考产品进行了验证。结果表明，平均总体精度（OA）从美国的0.65下降至欧洲的0.40。除精度指标外，我们还评估了模型性能的五个关键方面：模型置信度是否能指示信号失效、对观测窗口的敏感性、类别方案粗化的影响，以及对波段丢失和云影污染的鲁棒性。当观测窗口与当地作物物候不一致时，精度出现崩溃，而……（摘要原文在此处截断）

    arXiv:2610.08810v1 Announce Type: new  Abstract: Fine-tuned geospatial foundation models (GeoFMs) pretrained on large satellite archives have been shown to improve crop classification accuracy and geographic transferability. However, their operational performance beyond the training distribution remains poorly characterized. We evaluated the out-of-distribution performance of a widely adopted GeoFM [Prithvi-EO-2.0] across 37 events in 12 countries on three continents and validated against regional reference products. Results indicated that the mean overall accuracy (OA) declined from 0.65 in the United States to 0.40 in Europe. Beyond accuracy metrics, we assessed five key aspects of model performance: whether model confidence indicates signal failure, sensitivity to observation windows, the effect of coarsening class schemes, and robustness to both band loss and cloud- and shadow-contamination. Accuracy collapsed when the observation window misaligned with local crop phenology, while 
    
[^349]: 超越基线严重程度：正念干预后抑郁结局的时间性与疾病特异性预测因子

    Beyond Baseline Severity: Temporal and Disease-Specific Predictors of Depression Outcomes Following Mindfulness Interventions

    [https://arxiv.org/abs/2610.08809](https://arxiv.org/abs/2610.08809)

    该研究通过对多中心纵向临床队列进行可解释机器学习分析，证明超越基线抑郁严重程度的时间性与疾病特异性预测因子（如治疗参与度和临床背景）可有效预测正念干预后12周及24周的短期与长期抑郁结局。

    

    在患有慢性或急性疾病的患者中，抑郁严重程度受到基线心理状态、人口统计学特征、临床背景以及行为干预参与度等因素复杂交互作用的影响。本文对一项多中心纵向临床队列进行了可解释的机器学习分析，以预测参与正念干预后12周和24周的贝克抑郁量表第二版（BDI-II）得分。该研究利用人口统计学变量、临床疾病信息、医院中心标识、基线BDI-II得分以及治疗参与度指标来建模短期和长期抑郁结局。对于缺失的随访结局，研究采用基于模型的随机插补方法进行处理，在维持结局变异性的同时保留了有限的样本量。研究评估了五种回归模型，涵盖正则化线性回归和基于树的模型。

    arXiv:2610.08809v1 Announce Type: new  Abstract: Depression severity among patients with chronic or acute medical conditions is influenced by a complex interaction of baseline psychological state, demographic characteristics, clinical context, and engagement with behavioral interventions. This paper presents an interpretable machine-learning analysis of a multi-center longitudinal clinical cohort to predict Beck Depression Inventory-II (BDI-II) scores at 12 and 24 weeks following mindfulness-based intervention participation. The study uses demographic variables, clinical condition information, hospital-center identifiers, baseline BDI-II scores, and therapy engagement measures to model short-term and long-term depression outcomes. Missing follow-up outcomes were addressed using a model-based stochastic imputation procedure to preserve the modest sample size while maintaining outcome variability. Five regression models were evaluated, spanning regularized linear regression and tree-base
    
[^350]: 通过梯度归一化加速浮点可满足性求解

    Accelerating Floating-Point Satisfiability Solving via Gradient Normalization

    [https://arxiv.org/abs/2610.08808](https://arxiv.org/abs/2610.08808)

    该论文提出GradSAT框架，将每个SMT子句视为独立的多任务学习任务，通过动态梯度归一化平衡各子句梯度，克服梯度主导现象，从而加速浮点可满足性求解。

    

    可满足性模理论（SMT）求解器是软件验证、程序分析和编译器测试的基石，尤其在无量词浮点理论（QF_FP）上应用广泛。虽然近期的基于优化的SMT求解器已成功将梯度下降应用于逻辑公式的连续松弛，但它们从根本上受到“梯度主导”现象的瓶颈制约——即一小部分困难子句会劫持优化轨迹，阻碍求解器满足更广泛的公式，并使其陷入局部最小值。为了克服这一问题，我们提出了GradSAT，这是一个将基于优化的SMT求解与多任务学习（MTL）相结合的新型框架。GradSAT通过将每个SMT子句视为一个独立的MTL任务来重构约束满足过程。通过应用动态梯度归一化（GradNorm），GradSAT能够主动平衡所有子句之间的梯度幅值……

    arXiv:2610.08808v1 Announce Type: cross  Abstract: Satisfiability Modulo Theories (SMT) solvers are foundational to software verification, program analysis, and compiler testing, particularly over the theory of Quantifier-Free Floating-Point (QF_FP). While recent optimization-based SMT solvers have successfully applied gradient descent to continuous relaxations of logical formulas, they are fundamentally bottlenecked by gradient domination, a phenomenon where a small subset of difficult clauses hijacks the optimization trajectory, preventing the solver from satisfying the broader formula and trapping it in local minima.   To overcome this, we present GradSAT, a novel framework that bridges optimization-based SMT solving with Multi-Task Learning (MTL). GradSAT reformulates the constraint satisfaction process by treating each SMT clause as an independent MTL task. By applying dynamic gradient normalization (GradNorm), GradSAT actively balances the gradient magnitudes across all clauses a
    
[^351]: 一种面向涌现意识的贝叶斯镜像架构：循环层级、自我流形与混合事件-自我绑定

    A Bayesian Mirror Architecture for Emergent Consciousness: Circular Hierarchies, Self-Manifolds, and Hybrid Event-Self Binding

    [https://arxiv.org/abs/2610.08792](https://arxiv.org/abs/2610.08792)

    本文提出贝叶斯镜像架构（BMA），通过循环递归与闭环更新将自我表征绑定到抽象世界模型，把意识定义为具有这种循环结构的系统的一种架构属性，并在最优传输几何（2-Wasserstein 度量）下形式化自我稳定性与混合一致性。

    

    我们提出了贝叶斯镜像架构（BMA）的基础性表述，这是一个自指的生成式框架，其中感觉抽象、元抽象与自我潜在变量通过循环递归相互作用。其定义性约束是一个闭环更新 S_t <- H_{t-1}，其中混合事件-自我潜在变量 H_t 将自我表征绑定到抽象世界模型，并将这种耦合重新注入自我状态。在受限意义上，意识既不是优化目标，也不是语义标签，而是拥有这种循环结构的系统的一种架构属性。由于推理在后验信念上进行，BMA 的内在状态空间是一个配备最优传输几何的概率测度空间。稳定性与一致性在 P_2 上的 2-Wasserstein 度量中加以形式化，从而沿信念轨迹给出了坐标无关的自我稳定性与混合一致性概念。我们定义了一个因果学习……（摘要原文在此处截断）

    arXiv:2610.08792v1 Announce Type: new  Abstract: We present a foundational formulation of the Bayesian Mirror Architecture (BMA), a self-referential generative framework in which sensory abstractions, meta-abstractions, and a self-latent interact through circular recursion. The defining constraint is a closed update S_t <- H_{t-1}, where a hybrid event-self latent H_t binds self-representations to abstract world models and reinjects this coupling into the self-state. Consciousness, in a restricted sense, is not an optimization objective nor a semantic label, but an architectural property of systems possessing this circular structure.   Because inference operates over posterior beliefs, BMA's intrinsic state space is a space of probability measures equipped with optimal-transport geometry. Stability and coherence are formulated in the 2-Wasserstein metric on P_2, yielding coordinate-free notions of self-stability and hybrid coherence along belief trajectories. We define a Causal Learnin
    
[^352]: CNet：一个具有Wirtinger自动微分与FFT-阿达马卷积的复值深度学习框架

    CNet: A Complex-Valued Deep Learning Framework with Wirtinger Autodifferentiation and FFT--Hadamard Convolution

    [https://arxiv.org/abs/2610.08592](https://arxiv.org/abs/2610.08592)

    CNet 是一个基于 Wirtinger 自动微分的 C++/CUDA 复值深度学习框架，它将 FFT-阿达马卷积恒等式转化为可学习的复值卷积网络，并采用玻恩规则进行物理原生式分类。

    

    CNet 是一个用于构建和训练深度复值神经网络（CVNN）的 C++/CUDA 框架，更广泛地说，它利用 Wirtinger（CR 演算）导数通过梯度下降来优化复值函数。该框架采取物理原生的立场：网络是作用于振幅向量的复数运算——通常是幺正运算（如 DFT）——的级联，而分类则采用玻恩规则测量 $p_k = |z_k|^2 / \|z\|^2$，而非对实数 logits 进行 softmax。每个层都提供 CPU 参考实现和经过有限差分校验的 CUDA 内核，并且计算图会在整个批次上克隆以供 GPU 执行。在基础层之上，我们添加了信号处理原语，将恒等式 conv(x,k) = IFFT(FFT(x) · FFT(k)) 转化为可学习的复值卷积网络，同时提供了真 Adam 优化器和低内存推理模式。我们报告了三项研究。第一项是一个完全复值的、FNet 风格的因果……（摘要在此处截断）

    arXiv:2610.08592v1 Announce Type: new  Abstract: CNet is a C++/CUDA framework for building and training deep complex-valued neural networks (CVNNs) and, more generally, for optimizing complex-valued functions by gradient descent with Wirtinger (CR-calculus) derivatives. It takes a physics-native stance: a network is a cascade of complex -- and often unitary (the DFT) -- operations acting on an amplitude vector, and classification is a Born-rule measurement $p_k = |z_k|^2 / \|z\|^2$ rather than a softmax over real logits. Every layer ships a CPU reference and a CUDA kernel checked against finite differences, and the computation graph is cloned across the batch for GPU execution. On top of the base layers we add signal-processing primitives that turn the identity conv(x,k) = IFFT(FFT(x) . FFT(k)) into a learnable complex convolutional network, together with a true-Adam optimizer and a reduced-memory inference mode.   We report three studies. First, a fully complex-valued, FNet-style caus
    
[^353]: HE-OFT：同态加密下的隐私保护一次性联邦微调

    HE-OFT: Privacy-Preserving One-Shot Federated Fine-Tuning under Homomorphic Encryption

    [https://arxiv.org/abs/2610.08255](https://arxiv.org/abs/2610.08255)

    提出了HE-OFT——首个密码学安全的一次性联邦微调协议，任何一方都无法获得训练好的模型，在不泄露数据隐私的同时保护了模型这一专有资产。

    

    许多组织通过在私有数据上进行微调，将大型预训练模型适配到自己的任务中。其中多个参与方通常持有同一任务的数据，并希望在不汇集数据的情况下共同微调一个模型。联邦学习（FL）可以实现联合微调，但对共享中间值（模型或其梯度）的重构攻击仍然是隐私风险。一次性协议只交换一次加密的贡献，不会暴露任何中间值。然而，这样的协议仍会将训练好的模型交付给每个参与方，而在模型属于受监管或专有资产的情况下，这是不被允许的。我们提出了HE-OFT，这是首个密码学安全的一次性联邦微调协议，其中任何一方都不会获得训练好的模型。每个客户端在冻结的公共骨干网络上微调一个低秩适配器和一个分类器头，并保留该适配器。客户端上传一次加密的分类器头位移，服务器对其进行（后续内容截断）……

    arXiv:2610.08255v1 Announce Type: cross  Abstract: Many organizations adapt large pretrained models to their own tasks by fine-tuning on private data. Several of these parties often hold data for the same task and wish to fine-tune a model together without pooling that data. Federated learning (FL) enables joint fine-tuning, but reconstruction attacks on shared intermediate values (the model or its gradients) remain a privacy risk. A one-shot protocol that exchanges one encrypted contribution exposes no intermediate value. Such a protocol still gives the trained model to every participant, which is not permitted where the model is a regulated or proprietary asset. We present HE-OFT, the first cryptographically secure one-shot federated fine-tuning protocol in which no party receives the trained model. Each client fine-tunes a low-rank adapter and a classifier head on a frozen public backbone and keeps the adapter. The client uploads one encrypted head displacement, which the server com
    
[^354]: 找到语言模型中负责网络信息检索的注意力头与神经元

    Finding the Heads and the Neurons Responsible for Network Information Retrieval in Language Models

    [https://arxiv.org/abs/2610.08200](https://arxiv.org/abs/2610.08200)

    该研究发现，语言模型中经因果消融验证的极少数注意力头能以99.5%至100%的准确率检测上下文中的主机名与IP地址配对信息，且在某些模型中这一功能可进一步精确定位到单个神经元。

    

    我们探究是否特定的注意力头，以及更精细层面上这些头内部的特定神经元，负责识别语言模型上下文中包含的网络基础设施信息（主机名与其IP地址的配对），以及这种职责能否通过因果方法而非仅靠相关性来验证。在头的层面上，答案是肯定的，且该结论在横跨三个架构系列的五个模型中均成立：在每个模型中，通过因果消融筛选发现的一小组注意力头（128至1152个候选头中的1至9个），经过与匹配的负样本及无上下文对照的选择性测试，可支持一个在留出集上达到99.5%至100%准确率的检测器。我们随后探究一个注意力头的职责是集中于单个神经元还是分散在其各个维度上；答案因模型而异。在其中一个模型中，最强注意力头的信号集中于单个神经元，该结果由因果干预和相关性分析两种方法独立发现。

    arXiv:2610.08200v1 Announce Type: new  Abstract: We ask whether specific attention heads, and more finely specific neurons inside those heads, are responsible for recognizing that a language model's context contains network infrastructure information (a hostname paired with its IP address), and whether that responsibility can be validated causally rather than by correlation alone. At the head level the answer is yes, across five models spanning three architecture families: in every model, a small set of heads (1 to 9 out of 128 to 1152 candidates), found by causal ablation screening and tested for selectivity against matched negative and context-free controls, supports a detector with 99.5--100\% held-out accuracy. We then ask whether a head's responsibility concentrates into one neuron or stays spread across its dimensions; this is model-specific. In one model, the top head's signal concentrates into a single neuron, found independently by both a causal intervention and a correlationa
    
[^355]: 检索是不够的：为冻结的时间序列预测器刷新记忆

    Retrieval Is Not Enough: Refreshing Memory for Frozen Time-Series Forecasters

    [https://arxiv.org/abs/2610.07834](https://arxiv.org/abs/2610.07834)

    提出即插即用的FreshCast框架，通过持续用新观测数据刷新非参数记忆并进行校准，解决了检索增强时间序列预测中记忆陈旧导致检索效用下降的问题。

    

    检索增强的时间序列预测使用与当前上下文相似的历史片段的后续走势作为预测器的参考。大多数现有方法仅从训练片段中一次性构建检索记忆，导致部署之后新揭示的观测数据无法用作参考，且通常不校准检索到的信息应对冻结的预测器产生多大影响。我们识别出冻结预测器检索效用的两个关键决定因素：历史数据是否仍然反映当前状态，以及其所引发的修正是否与预测器的残差误差对齐——当记忆变得陈旧时，这种对齐可能在验证阶段与部署阶段之间发生偏移。我们提出FreshCast，一个即插即用的检索框架，它保持预测器冻结，用新观测数据持续更新非参数记忆，通过关系核回归形成记忆预测，并校准……（摘要在此处截断）

    arXiv:2610.07834v1 Announce Type: new  Abstract: Retrieval-augmented time-series forecasting uses the continuations of historical segments similar to the current context as references for a forecaster. Most existing methods build the retrieval memory once from the training segment, leaving observations revealed after deployment unavailable as references, and generally do not calibrate how much the retrieved information should influence a frozen forecaster. We identify two key determinants of retrieval utility for a frozen forecaster: whether the history still reflects the current state, and whether the correction it induces aligns with the forecaster's residual errors, an alignment that can shift between validation and deployment when the memory becomes stale. We propose FreshCast, a plug-in retrieval framework that keeps the forecaster frozen, continuously updates a non-parametric memory with new observations, forms a memory forecast through relational kernel regression, and calibrate
    
[^356]: 神经运动层级网络：面向sEMG解码鲁棒泛化的生理学归纳偏置

    Neuromotor Hierarchy Network: Physiological Inductive Biases for Robust Generalization in sEMG Decoding

    [https://arxiv.org/abs/2610.07713](https://arxiv.org/abs/2610.07713)

    提出受神经运动层级结构启发的NHN网络，通过引入生理学归纳偏置学习紧凑的潜在神经运动状态，从而在跨用户、跨会话的sEMG解码中实现鲁棒泛化。

    

    表面肌电图（sEMG）为运动解码和人机交互提供了一种可穿戴、无创的神经肌肉活动接口。大规模人群解码仍然困难，原因在于sEMG与神经肌肉活动之间的关系在不同用户和会话之间存在差异，而任务相关的动态跨越多个通道和多个时间尺度。从任务标签学习波形到输出的映射，使得记录变异性与协调性运动活动之间的区分停留在隐式层面。我们提出了神经运动层级网络（NHN），它从任务监督中学习一个紧凑的潜在神经运动状态，以表征任务相关的神经肌肉协调。NHN通过受神经运动组织结构启发的层级机制来构建该潜在状态。它在保持相对强度的同时适应记录统计特性。其时空编码器采用参数高效的通道交互方式，并通过多…（摘要在此处截断）

    arXiv:2610.07713v1 Announce Type: new  Abstract: Surface electromyography (sEMG) provides a wearable, noninvasive interface to neuromuscular activity for movement decoding and human-computer interaction. Population-scale decoding remains difficult because the relationship between sEMG and neuromuscular activity varies across users and sessions, while task-relevant dynamics span channels and multiple timescales. Learning waveform-to-output mappings from task labels leaves the distinction between recording variability and coordinated motor activity implicit. We introduce the Neuromotor Hierarchy Network (NHN), which learns a compact latent neuromotor state from task supervision to represent task-relevant neuromuscular coordination. NHN constructs this latent state through a hierarchy inspired by neuromotor organization.It adapts recording statistics while preserving relative intensity.Its spatiotemporal encoder uses parameter-efficient channel interactions and modulates features with mul
    
[^357]: DeepAJM：面向不规则采样数据的深度关联联合模型

    DeepAJM: Deep Association Joint Model for Irregularly Sampled data

    [https://arxiv.org/abs/2610.07388](https://arxiv.org/abs/2610.07388)

    提出 DeepAJM——一种无需参数假设的深度联合模型，利用编码器-解码器架构学习不规则采样的时变协变量轨迹的潜在结构，并通过部分可解释的关联结构将其与生存结局关联，从而改进生存预测。

    

    联合模型同时建模纵向结局与生存结局，利用患者纵向轨迹中的模式来改进生存结局的预测。然而，经典的参数化联合模型依赖于固定的参数假设，在模型误设和样本量较小的情况下容易产生偏差。我们提出了一种深度联合模型 DeepAJM，它不需要任何参数假设，同时保留了部分可解释的、针对每个纵向结局的关联结构。该联合模型采用编码器-解码器（序列到序列）架构来学习患者时变协变量轨迹中的潜在结构。模型通过一个学习得到的可解释关联结构将纵向过程与生存过程联系起来，其中解码器输出的每个纵向结果在贡献于（生存模型的）风险评分之前，会先由基线协变量进行重新调制……

    arXiv:2610.07388v1 Announce Type: cross  Abstract: Joint Models simultaneously model longitudinal and survival outcomes, leveraging patterns in patients' longitudinal trajectory to improve the prediction of survival outcomes. The classical parametric joint models, however, rely on fixed parametric assumptions, making them susceptible to bias under model misspecification and smaller sample sizes. We propose a deep joint model, DeepAJM, that does not require any parametric assumptions, while retaining a partially interpretable, per-longitudinal-outcome association structure. The joint model uses an encoder-decoder (sequence-to-sequence) architecture to learn the latent structure in patients' time-varying covariate trajectories. The model links the longitudinal processes to the survival processes through a learned interpretable association structure, in which each longitudinal output from the decoder gets remodulated by baseline covariates before it contributes to the risk scores from the
    
[^358]: Fréchet Inception 距离的样本最优估计

    Sample-Optimal Estimation of the Fr\'echet Inception Distance

    [https://arxiv.org/abs/2610.07114](https://arxiv.org/abs/2610.07114)

    该论文针对FID估计中的有限样本偏差问题，证明了插件估计器的紧致偏差与方差界并确立其平方级（d²）样本复杂度，同时将FID∞估计器推广到任意阶外推方法以实现去偏估计。

    

    Fréchet Inception 距离（FID）被广泛用于评估生成模型，但其经验插件估计器存在有限样本偏差 [BSAG18, CF20]。我们研究了在一个分布已知的情况下，估计具有有界均值距离和协方差的 $d$ 维高斯分布之间的 FID 至误差 $\epsilon$ 所需的样本复杂度 $n$。我们的贡献有三点：(1) 我们为经验插件估计器建立了紧致的有限样本偏差界 $\Theta(\frac{d^2}{n})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$，从而确立了 $\gtrsim d^2$ 的样本复杂度。(2) 为了对经验插件估计器进行去偏，我们将 [CF20] 的 ${\rm FID}_\infty$ 估计器推广到任意阶数 $k$ 的外推方法，并进一步在我们的框架下证明了任意 $k$ 阶外推的紧致偏差界 $\Theta(\frac{d^{k+2}}{n^{k+1}})$ 和方差界 $\Theta(\frac{d}{n} + \frac{d^2}{n^2})$。(3) 我们引入

    arXiv:2610.07114v1 Announce Type: new  Abstract: The Fr\'echet Inception Distance (FID) is widely used to evaluate generative models, but its empirical plug-in estimator suffers from finite-sample bias [BSAG18, CF20]. We study the sample complexity $n$ of estimating FID to error $\epsilon$ between $d$-dimensional Gaussians with bounded mean distance and covariances, when one distribution is known. Our contributions are threefold. (1) We establish tight finite-sample $\Theta(\frac{d^2}{n})$ bias and $\Theta(\frac{d}{n} + \frac {d^2} {n^2})$ variance bounds for the empirical plug-in estimator, establishing a $\gtrsim d^2$ sample complexity. (2) To debias the empirical plug-in estimator, we generalize the ${\rm FID}_\infty$ estimator of [CF20] to extrapolation methods of arbitrary order $k$. We further prove tight bias and variance bounds of $\Theta(\frac{d^{k + 2}}{n^{k + 1}})$ and $\Theta(\frac d n + \frac{d^2}{n^2})$ for any order-$k$ extrapolation under our framework. (3) We introduce
    
[^359]: ImpactMat：面向逆冲击声音渲染的连续材质估计

    ImpactMat: Continuous Material Estimation for Inverse Impact Sound Rendering

    [https://arxiv.org/abs/2610.07061](https://arxiv.org/abs/2610.07061)

    该论文提出了ImpactMat数据集与基准，以及一个前馈模型，能够从冲击声音录音中连续估计材质参数，实现逆冲击声音渲染，从而突破传统渲染器固定材质预设的限制。

    

    冲击声音渲染用于合成三维物体被敲击时所产生的声音，但实用的渲染器通常依赖于固定的材质预设，如木材、塑料或钢材。这些预设限制了渲染器所能表达的冲击声音范围，而在缺乏材料声学专业知识的情况下，手动调整底层材质参数也十分困难。因此，我们研究逆冲击声音渲染问题：从参考冲击声音中预测材质参数，使模拟器能够重现相似的材质响应。为支持这一任务，我们提出了ImpactMat——一个包含单一材质与混合材质冲击声音及其真实材质参数标注的数据集和基准。我们进一步提出了一种前馈模型，能够从一段或多段录音中预测这些材质参数，并利用混合材质来学习材质类型之间的平滑过渡。实验表明，我们的方法优于具有竞争力的基线方法，并且……

    arXiv:2610.07061v1 Announce Type: cross  Abstract: Impact sound rendering synthesizes the sound produced when a 3D object is struck, but practical renderers often rely on fixed material presets such as wood, plastic, or steel. These presets limit the range of impact sounds a renderer can express, while manually adjusting the underlying material parameters remains difficult without expertise in material acoustics. We therefore study inverse impact sound rendering: predicting material parameters from a reference impact sound so that a simulator can recreate a similar material response. To support this task, we introduce ImpactMat, a dataset and benchmark of single and blended material impact sounds paired with ground-truth material parameters. We further propose a feed-forward model that predicts these parameters from one or more recordings, using blended materials to learn smooth transitions between material types. Experiments show that our method outperforms competitive baselines and e
    
[^360]: 用于建筑热负荷短期预测的混合预测模型比较综述

    Comparative review of hybrid forecasting models for short-term prediction of building thermal load

    [https://arxiv.org/abs/2610.06881](https://arxiv.org/abs/2610.06881)

    本文综述并比较了13种用于建筑热负荷短期预测的混合模型，发现EMD-LSTM-Markov模型的预测精度最高。

    

    本文对不同混合模型在建筑热需求短期预测中的表现进行了比较综述。特别地，该评估针对的是与其他最先进技术相结合的增强型数据驱动模型之间的比较。第一步，分析了文献中已报道的现有技术，结论是元启发式算法或数据驱动模型被用于识别基础模型的参数。定性评估包括每种方法的输入和输出特征、主要优点和缺点。第二步，利用苏格兰家庭历史热需求数据集以及历史天气预报，进一步评估了现有混合方法的性能。在对13种混合方法的评估中，经验模态分解-长短期记忆-马尔可夫（EMD-LSTM-Markov）模型能够以最高的精度进行预测……

    arXiv:2610.06881v1 Announce Type: cross  Abstract: In this paper, a comparative review of different hybrid models for short-term forecasting of building thermal demand is carried out. Particularly, the assessment tackles the comparison of data-driven models enhanced with other state-of-the-art techniques. At the first step, the existing techniques reported in the literature are analysed. It is concluded that Metaheuristics or a data-driven model are used to identify the parameters of the basic model. The qualitative evaluation includes for each method the input and output features, main advantages and drawbacks. At the second step, an existing dataset of historical thermal demand from Scottish households, as well as historical weather forecasts are utilized to assess additionally the performance of existing hybrid methods. From the assessment of 13 hybrid methods, the Empirical Modal Decomposition - long short-term memory - Markov (EMD-LSTM-Markov) model can predict with the highest ac
    
[^361]: RAISED：通过自蒸馏提升大语言模型智能体对提示注入攻击的鲁棒性

    RAISED: Self-Distillation for Robustness to Prompt Injection in LLM Agents

    [https://arxiv.org/abs/2610.06401](https://arxiv.org/abs/2610.06401)

    提出RAISED训练框架，通过自我生成与自蒸馏相结合的方式，在不损害大语言模型智能体通用能力的前提下，显著提升其对间接提示注入攻击的鲁棒性。

    

    使用工具的语言模型智能体容易受到间接提示注入攻击，因为它们必须基于不可信的外部内容执行操作。现有的训练时防御方法虽然能够降低攻击成功率，但往往以牺牲模型的通用能力为代价。我们证明了基于训练的防御方法会导致模型输出分布产生显著漂移，即使在良性场景下也会改变模型行为，这为模型效用下降提供了一种潜在机制。我们还进一步识别了这些防御方法的一种失效模式：在良性的工具使用任务中，模型会拒绝执行完成授权任务所需的某个步骤，尤其是当该步骤由工具输出所指示时。为了解决这些局限性，我们提出了RAISED（通过自蒸馏实现攻击鲁棒不变性），这是一个将自我生成与自蒸馏相结合的训练框架。模型首先生成自己的工具使用场景，重点关注任务完成需要……的情况（摘要原文在此处截断）。

    arXiv:2610.06401v2 Announce Type: replace-cross  Abstract: Tool-using language-model agents are vulnerable to indirect prompt injection because they must act on untrusted external content. Existing training-time defenses can reduce attack success rates, but often at the cost of general capabilities. We show that training-based defenses induce substantial drift in the model's output distribution, altering its behavior even in benign settings and providing a potential mechanism for utility degradation. We further identify a failure mode of these defenses: On benign tool-use tasks, the model refrains from a step needed to finish an authorized task, particularly when that step is indicated by a tool output. To address these limitations, we introduce RAISED (Robust Attack Invariance through Self-Distillation), a training framework that combines self-generation and self-distillation. The model first generates its own tool-use scenarios, with an emphasis on cases where task completion require
    
[^362]: 缩减规模定律：资源受限大语言模型中的参数效率与计算最优训练

    Scaling Down the Scaling Laws: Parameter Efficiency and Compute-Optimal Training in Resource-Constrained Large Language Models

    [https://arxiv.org/abs/2610.06387](https://arxiv.org/abs/2610.06387)

    这篇综述梳理了大语言模型扩展理论从经验扩展定律到计算最优训练的演进，指出研究重心正从单纯追求规模最大化转向在资源受限条件下对参数、token、计算和硬件资源的高效利用与合理分配。

    

    大语言模型（LLMs）通过增加模型规模、训练数据和计算资源获得了显著的性能提升。然而，传统的扩展方法带来了边际收益递减、不断攀升的财务与环境成本，以及在大型工业实验室之外的研究者参与门槛的提高。本综述考察了大语言模型扩展理论从经验性扩展定律到计算最优训练的演变过程，特别关注参数效率、token 利用率、数据效率以及资源受限环境。文章将扩展定律的基础性研究与后续关于计算最优训练、数据剪枝、高效架构、量化、低秩适配以及面向边缘设备优化的工作进行了综合梳理。文献表明，该领域正从规模最大化转向对参数、token、计算和硬件资源更加审慎合理的分配。

    arXiv:2610.06387v2 Announce Type: replace  Abstract: Large language models (LLMs) have achieved substantial performance gains through increases in model size, training data, and computational resources. However, traditional scaling approaches produce diminishing returns, rising financial and environmental costs, and barriers to participation for researchers operating outside large industrial laboratories. This review examines the evolution of LLM scaling theory from empirical scaling laws to compute-optimal training, with particular emphasis on parameter efficiency, token utilization, data efficiency, and resource-constrained environments. Foundational work on scaling laws is synthesized alongside later research on compute-optimal training, data pruning, efficient architectures, quantization, low-rank adaptation, and edge-oriented optimization. The literature indicates a shift from scale maximization toward more deliberate allocation of parameters, tokens, compute, and hardware resourc
    
[^363]: PDE训练输入的有损压缩：场重构误差无法对已训练算子的代价进行排序

    Lossy Compression of PDE Training Inputs: Field Reconstruction Error Does Not Order the Cost to a Trained Operator

    [https://arxiv.org/abs/2610.06095](https://arxiv.org/abs/2610.06095)

    本文证明压缩PDE训练输入时的场重构误差无法预测所训练算子的精度损失，因为解算子对输入扰动的衰减程度在不同PDE族之间相差两个数量级以上，导致重构误差指标在104个代价比较中反转了36个。

    

    算子学习基准数据集以全精度存储，规模已增长到太字节级别。率失真理论说明了存储场需要多少比特，而实践者需要知道在压缩数据上训练的算子会有多准确。我们证明前者并不能决定后者，并测量了其中原因：我们在压缩输入场的同时，保持目标数据和测试输入为全精度。解算子会衰减其输入的扰动。将压缩后的场输入到一个已在全精度下训练好的代理模型中，可以测量该代理模型传输了多少扰动。这一比例与底层方程的平滑行为一致，并且在不同PDE族之间跨越两个多数量级。场重构误差是在衰减发生之前计算的，因此无法察觉这种衰减。对于用均方误差训练的算子，场重构误差在104个跨数据集的代价比较中反转了36个。

    arXiv:2610.06095v2 Announce Type: replace  Abstract: Operator-learning benchmarks are stored at full precision and have grown to terabyte scale. Rate-distortion theory says how many bits the stored field needs, while a practitioner needs to know how accurate an operator trained on the compressed data will be. We show that the first does not determine the second, and measure why, compressing the input fields while targets and test inputs stay at full precision. A solution operator attenuates a perturbation of its input. Pushing a compressed field through a surrogate already trained at full precision measures how much of the perturbation that surrogate transmits. The fraction is consistent with the smoothing behaviour of the underlying equation, and it spans more than two orders of magnitude across PDE families. Field reconstruction error is computed before the attenuation and cannot see it. For operators trained with mean squared error it inverts 36 of 104 cost comparisons across datase
    
[^364]: 提升针对深度强化学习的可迁移对抗攻击

    Boosting Transferable Adversarial Attacks against Deep Reinforcement Learning

    [https://arxiv.org/abs/2610.06083](https://arxiv.org/abs/2610.06083)

    该论文提出一种基于可微分环境模型和温度平滑代理策略的轨迹级黑盒攻击方法，通过在滚动时域内优化扰动序列，显著提升了对抗扰动从未知受害智能体上的迁移攻击效果，超越了传统图像分类攻击方法移植后的表现。

    

    摘要：针对深度强化学习（DRL）的大多数对抗攻击都假设攻击者拥有对受害策略的白盒访问权限，而这一假设在实践中很少成立。本文研究基于迁移的黑盒攻击DRL：攻击者在白盒代理智能体上构造观测扰动，并将其输入给未知的受害智能体。我们将该攻击形式化为在每步扰动预算约束下的回报最小化问题。我们首先证明，将可迁移的图像分类攻击方法（FGSM、MI-FGSM和NI-FGSM）结合每步目标进行移植，所产生的扰动虽然具备可迁移性，但其攻击效果并不比同等预算下的随机噪声更强。随后，我们提出一种轨迹级攻击方法，通过环境的可微分模型和温度平滑的代理策略，在滚动时域内优化一个扰动序列，并使用相同的优化器。在CartPole-v1环境中，使用十个DQN和DDQN智能体以及100个代理-受害对进行实验，轨迹级攻击……（原文摘要在此处截断）

    arXiv:2610.06083v2 Announce Type: replace  Abstract: Most adversarial attacks on deep reinforcement learning (DRL) assume white-box access to the victim policy, which rarely holds in practice. This paper studies transfer-based black-box attacks on DRL: the attacker crafts observation perturbations on a white-box surrogate agent and feeds them to an unknown victim. We formulate the attack as return minimization under a per-step perturbation budget. We first show that transplanting transferable image-classification attacks (FGSM, MI-FGSM, and NI-FGSM) with a per-step objective yields perturbations that transfer but are no stronger than random noise of the same budget. We then propose a trajectory-level attack that optimizes a sequence of perturbations over a receding horizon through a differentiable model of the environment and a temperature-smoothed surrogate policy, with the same optimizers. On CartPole-v1 with ten DQN and DDQN agents and 100 surrogate--victim pairs, the trajectory-lev
    
[^365]: 次水平Flood双过滤：迈向可扩展的2参数持续同调

    The sublevel Flood bifiltration: towards scalable 2-parameter persistent homology

    [https://arxiv.org/abs/2610.05441](https://arxiv.org/abs/2610.05441)

    本文提出次水平Flood双过滤，通过扩展单参数Flood过滤，为2参数持续同调提供了一种具有理论稳定性保证且可高效计算、适用于大规模点集的可扩展近似方法。

    

    多参数持续同调是拓扑数据分析中一个快速发展的分支，它提升了单参数持续同调对离群点的鲁棒性，同时仍能捕捉数据的度量特征。然而，一个显著的局限性在于其缺乏可扩展性。在本文中，我们提出了一种在大规模点集上高效计算2参数持续同调的新方法。我们的工作扩展了最初为单参数持续性而开发的Flood过滤。我们的构造称为次水平Flood双过滤，它为次水平偏移双过滤提供了可扩展的近似。我们证明了该构造具有理论稳定性，并描述了如何高效地计算它。我们在密度感知至关重要的低维合成数据集上，以及真实世界的时间序列数据集上，展示了我们方法在分类任务中的性能。

    arXiv:2610.05441v2 Announce Type: replace-cross  Abstract: Multiparameter persistent homology is a rapidly developing branch of topological data analysis that improves the robustness of single-parameter persistent homology to outliers, while still capturing the metric characteristics of the data. However, a notable limitation is its lack of scalability. In this paper, we introduce a novel approach for efficiently computing 2-parameter persistent homology on large point sets. Our work extends the Flood filtration, originally developed for single-parameter persistence. Our construction, called the sublevel Flood bifiltration, offers a scalable approximation of the sublevel offset bifiltration. We show that it benefits from theoretical stability properties and describe how to compute it efficiently. We demonstrate the performance of our approach in classification tasks on low-dimensional synthetic datasets, where density awareness is critical, as well as on real-world time series datasets
    
[^366]: E$^2$-OPSD：驯服在线策略自蒸馏中的熵过冲

    E$^2$-OPSD: Taming Entropy Overshoot in On-Policy Self-Distillation

    [https://arxiv.org/abs/2610.05048](https://arxiv.org/abs/2610.05048)

    论文发现在线策略自蒸馏存在学生熵超过教师并持续高企的“熵过冲”失效模式，其根源是教师监督过度依赖答案特定线索以及前向KL散度不断扩散学生分布，并据此提出E$^2$-OPSD同时修复这两个成因。

    

    在线策略自蒸馏（OPSD）无需第二个模型即可提供密集的token级监督：同一个网络在给定参考解答时充当教师，而在仅给定问题时充当学生。我们识别出该方法的一个特定失效模式：在训练过程中，学生的token熵会超过教师的熵并持续保持高位，我们将这一现象称为“熵过冲”（entropy overshoot）。我们将其根源追溯到蒸馏的双方。以参考答案为条件的教师在其面向答案的推理路径上表现得很自信，但这种自信难以迁移到学生生成的前缀上，使其监督过度依赖于答案特定的线索，而非可复用的推理模式；与此同时，OPSD所使用的前向KL散度会持续扩散学生的预测分布，而无法将其拉回。我们提出E$^2$-OPSD来同时应对这两个成因。示例引导的教学（exemplar-guided teaching）用检索到的已解决的相邻问题替换当前答案，提……（原文摘要在此处截断）

    arXiv:2610.05048v2 Announce Type: replace-cross  Abstract: On-policy self-distillation (OPSD) provides dense token-level supervision without a second model: one network acts as teacher with the reference solution and as student with only the problem. We identify a specific failure mode of this recipe. During training, student token entropy rises past the teacher's and remains elevated, a pattern we call entropy overshoot. We trace it to both sides of distillation. The reference-conditioned teacher is confident along its answer-directed reasoning path, but this confidence transfers poorly to student-generated prefixes, making its supervision overly tied to answer-specific cues rather than reusable reasoning patterns; meanwhile, the forward KL used by OPSD continually diffuses the student's predictive distribution without pulling it back. We introduce E$^2$-OPSD to address both causes. Exemplar-guided teaching replaces the current answer with a retrieved solved neighboring problem, provi
    
[^367]: PIT-GCL：基于拓扑图对比学习的蛋白质相互作用预测

    PIT-GCL: Protein Interaction using Topological Graph Contrastive Learning

    [https://arxiv.org/abs/2610.04850](https://arxiv.org/abs/2610.04850)

    PIT-GCL 提出了一种双塔结构感知框架，将 ESM-2 残基嵌入与基于 Vietoris-Rips 过滤持续同调（H0/H1 持续景观）的拓扑描述符相结合，无需结合态复合物结构即可实现序列与结构层面的蛋白质相互作用预测。

    

    蛋白质结合预测对于靶点识别、治疗性结合物设计以及大规模筛选至关重要，但由于结合取决于序列、三维几何结构以及全局结构组织，该任务仍然充满挑战。近期的折叠模型（如 AlphaFold3 和 Boltz-2）显著提升了结构预测能力，但其置信度输出（pLDDT、pTM、ipTM）并非专为二元结合预测而设计，而专用的结构感知预测器通常需要结合态复合物结构，这在筛选规模下难以获得。我们提出了 PIT-GCL，这是一种双塔结构感知框架，能够基于氨基酸序列、Cα 点云以及全局持续同调描述符对每个蛋白质进行独立编码。每个塔将残基 ESM-2 嵌入与由 Vietoris-Rips 过滤的 H0 和 H1 持续景观计算得到的拓扑摘要相结合，并对……

    arXiv:2610.04850v2 Announce Type: replace  Abstract: Protein binding prediction is central to target identification, therapeutic binder design, and large scale screening, yet remains challenging because binding depends on sequence, three dimensional geometry, and global structural organization. Recent folding models such as AlphaFold3 and Boltz-2 have substantially improved structure prediction, but their confidence outputs (pLDDT, pTM, ipTM) are not specifically designed for binary binding prediction, and dedicated structure aware predictors often require bound complex structures that are unavailable at screening scale. We introduce PIT-GCL, a dual tower structure aware framework that encodes each protein independently from its amino acid sequence, C{\alpha} point cloud, and a global persistent homology descriptor. Each tower combines residue ESM-2 embeddings with a topological summary computed from the H0 and H1 persistence landscapes of a Vietoris-Rips filtration, and processes the 
    
[^368]: 审视问题本身：维持推理模型的自演化

    Questioning the Questions: Sustaining Self-Evolution in Reasoning Models

    [https://arxiv.org/abs/2610.04299](https://arxiv.org/abs/2610.04299)

    该论文揭示了自演化推理模型性能崩溃的两大根源——自生成问题中无效问题比例上升以及数学等价重复问题导致多样性崩溃，并提出通过问题有效性与新颖性反馈（R-Quest）来引导和维持模型的自演化。

    

    自演化的推理模型从其自身生成的问题中学习，然而反复的自我训练可能导致性能崩溃。本文研究了性能为何会在连续多轮训练中逐渐退化，以及如何维持自演化过程。我们的分析发现自生成问题中存在两类反复出现的质量问题：无效问题和同一数学问题的重复变体。首先，无效问题在各轮次中变得愈发普遍，而基于答案一致性的过滤进一步提高了其在训练数据中的占比。其次，现有的基于词汇相似性的问题多样性控制方法无法识别以不同表达方式呈现的数学等价问题，从而导致训练后期出现问题多样性崩溃。基于这些发现，我们提出了 R-Quest，它利用问题有效性和新颖性反馈来引导自演化。我们首先训练求解器识别……

    arXiv:2610.04299v2 Announce Type: replace-cross  Abstract: Self-evolving reasoning models learn from their own generated questions, yet repeated self-training can lead to performance collapse. In this paper, we investigate why performance deteriorates over successive rounds and how to sustain self-evolution. Our analysis identifies two recurring quality problems in self-generated questions: invalid questions and repeated variants of the same mathematical questions. First, invalid questions become more prevalent across rounds, and answer-consistency filtering further increases their proportion in training data. Second, existing question diversity controls based on lexical similarity can miss mathematically equivalent questions expressed in different ways, which leads to question diversity collapse in later training rounds. Building on these findings, we introduce R-Quest, which uses question validity and novelty feedback to guide self-evolution. We first train the solver to recognize an
    
[^369]: 人形机器人拉人力车：耦合轮式负载下的全身运动控制

    Humanoid Rickshaw Pulling: Whole-Body Locomotion under Coupled Wheeled Loads

    [https://arxiv.org/abs/2610.04238](https://arxiv.org/abs/2610.04238)

    提出了一种人形机器人拉人力车的全身控制框架，通过特权教师策略蒸馏到基于历史条件的学生策略并结合强化学习微调，使机器人能够在不确定的耦合负载动力学下拉动远超自身重量的轮式负载，同时保持平衡与稳定抓取。

    

    人形机器人可以通过拉动被动轮式车辆而非直接搬运负载，来运输远超自身重量的货物。然而，这种能力带来了一个耦合运动问题：机器人必须在保持上半身持续接触的同时，适应来自负载、车辆和地形的未知的、依赖于构型的力。我们提出了一种用于人形机器人拉人力车的全身控制框架，该框架能够在不确定的负载动力学下跟踪指令的车辆运动，同时保持平衡和稳定的抓取。在训练过程中，具备特权信息的教师策略利用车辆状态、交互力和负载属性，其动作和潜在变量被蒸馏到一个基于历史条件的学生策略中，该学生策略从本体感知响应中隐式推断耦合动力学，随后进行强化学习微调。与“无历史”和“仅有历史”基线的对比表明，由此产生的策略……（摘要原文在此处截断）

    arXiv:2610.04238v2 Announce Type: replace-cross  Abstract: Humanoid robots could transport payloads substantially heavier than themselves by pulling passive wheeled vehicles instead of carrying the load. This capability, however, creates a coupled locomotion problem: the robot must maintain persistent upper-body contact while adapting to unknown, configuration-dependent forces arising from the payload, vehicle, and terrain. We present a whole-body control framework for humanoid rickshaw pulling that tracks commanded vehicle motion while preserving balance and stable grasps under uncertain load dynamics. During training, a privileged teacher exploits vehicle states, interaction forces, and load properties. Its actions and latent are distilled into a history-conditioned student that implicitly infers coupled dynamics from proprioceptive responses, followed by reinforcement-learning fine-tuning. Comparisons with \emph{No History} and \emph{Only History} baselines show that the resulting p
    
[^370]: Clean：基于Nyström草绘实现线性内存成本的二阶LLM训练

    Clean: Second-order LLM Training at Linear Memory Cost via Nystr\"om Sketching

    [https://arxiv.org/abs/2610.04204](https://arxiv.org/abs/2610.04204)

    Clean利用随机化Nyström草绘将全曲率二阶优化器的内存复杂度从二次方降至线性，并通过重新整合子空间外分量保留曲率信息，其低精度变体Q-Clean进一步将优化器内存减少50%以上，实现了内存高效的二阶LLM训练。

    

    训练大语言模型（LLM）面临一个根本性的权衡：诸如Adam等内存高效的优化器会丢弃跨参数曲率信息，而SOAP等全曲率方法虽能加速收敛，却伴随极高的内存成本。我们提出Clean，一种旨在解决这一瓶颈的内存高效全曲率优化器。Clean利用随机化Nyström方法精确逼近SOAP中的左右预条件子，并将优化器的内存复杂度从模型维度的二次方降低至线性。随后，我们重新整合子空间外的分量，以捕获低秩近似之外的曲率信息，以极小的内存开销保留丰富的曲率。我们进一步提出低精度变体Q-Clean，可对优化器状态进行激进压缩。在预训练时，Q-Clean相比Muon将优化器内存消耗降低了超过50%……

    arXiv:2610.04204v2 Announce Type: replace-cross  Abstract: Training large language models (LLMs) entails a fundamental trade-off: memory-efficient optimizers such as Adam discard cross-parameter curvature, whereas full-curvature methods such as SOAP can accelerate convergence at prohibitive memory costs. We introduce Clean, a memory-efficient and full-curvature optimizer designed to resolve this bottleneck. Clean leverages the randomized Nystrom method to accurately approximate the left and right preconditioners in SOAP, and to reduce the optimizer's memory complexity from quadratic to linear in terms of model dimensions. We subsequently reintegrate the off-subspace components to capture curvature information beyond the low-rank approximation, preserving rich curvature at minimal memory cost. We further propose Q-Clean, a low-precision variant that aggressively compresses optimizer states. Q-Clean reduces optimizer memory consumption by \textbf{over 50\%} compared to Muon when pre-trai
    
[^371]: WebFovea：当模型正确但点击出错时——基于视觉的网页智能体在真实网站上的可靠往返执行

    WebFovea: When the Model Is Right but the Click Is Wrong -- Reliable Round Trips for Vision-Based Web Agents on Live Websites

    [https://arxiv.org/abs/2610.03036](https://arxiv.org/abs/2610.03036)

    本文提出在WebRetriever Challenge 2026中获得亚军的视觉网页智能体WebFovea，并指出真实网站上的许多失败并非源于模型推理，而是源于模型与浏览器之间中间执行层在动作解析、页面生效、结果反馈和信息展示这四个环节上的问题。

    

    我们提出了WebFovea，一个基于视觉的网页智能体，它在WebRetriever Challenge 2026中获得第二名，最终得分为100分中的57.0分。该挑战赛在WebRetriever基准（arXiv:2607.06118）的协议III上对智能体进行端到端评估：从真实网站上的入口URL出发，智能体必须操作网站自身的界面并返回可验证的答案。一个强大的多模态大语言模型（LLM）对完成此任务是必要的，但并不充分。模型的决策需要通过中间执行层（harness）——即模型与页面之间的代码——传递到浏览器。在每一步中，有四件事必须正确完成：模型的回复必须被解析为预期的动作，动作必须在页面上生效，结果必须被准确地反馈回来，并且模型必须被展示它所需的信息。在真实网站上，我们观察到的许多失败发生在这四个阶段之一，而不是出现在模型的推理中。一个坐标空间（摘要原文在此处截断）

    arXiv:2610.03036v1 Announce Type: cross  Abstract: We present WebFovea, a vision-based web agent that placed 2nd in the WebRetriever Challenge 2026 with a final score of 57.0 out of 100. The challenge evaluates agents end to end on Protocol III of the WebRetriever benchmark (arXiv:2607.06118): starting from an entry URL on a live website, the agent must operate the site's own interface and return a verifiable answer. A capable multimodal large language model (LLM) is necessary for this, but not sufficient. The model's decisions reach the browser through the harness, the code between the model and the page. At every step, four things must go right: the model's reply must be parsed into the intended action, the action must take effect on the page, the result must be reported back accurately, and the model must be shown the information it needs. On real websites, many of the failures we observed occurred at one of these four stages rather than in the model's reasoning. A coordinate-space 
    
[^372]: 混合专家粒子Transformer中的条件容量与路由

    Conditional Capacity and Routing in Mixture-of-Experts Particle Transformers

    [https://arxiv.org/abs/2610.02701](https://arxiv.org/abs/2610.02701)

    该研究发现在粒子物理Transformer中，避免token丢弃的top-1混合专家模型能在几乎不增加计算量的前提下超越稠密基线，但增加存储专家数量的收益有限，且专家路由结构与分类性能并非单调相关。

    

    混合专家模型能够在不按比例增加活跃计算量的情况下提升参数容量，但这一权衡在粒子物理Transformer中的表现尚不清楚。我们在188类的JetClass-II数据集上研究了稠密与MoE粒子Transformer，并改变专家数量、路由容量、top-K以及辅助损失。我们发现，在避免token丢弃的情况下，top-1 MoE模型能在几乎不变的标称前向计算量下超越稠密基线，而进一步增加存储的专家数量仅带来很小的额外准确率收益。为每个token激活多个专家可以在更高的计算成本下带来额外的预测性能提升。路由分析显示，在某些配置下专家分配与粒子身份及运动学特征的关联变得更强，但这种结构并不随分类性能的提升而单调增强。这些结果突显了需要区分……（原文摘要在此处截断）

    arXiv:2610.02701v1 Announce Type: new  Abstract: Mixture-of-Experts (MoE) models can increase parameter capacity without proportionally increasing active computation, but it is unclear how this trade-off behaves in particle-physics transformers. We study dense and MoE Particle Transformers on 188-class JetClass-II, varying expert count, routing capacity, top-K, and auxiliary loss. We find that, when token dropping is avoided, top-1 MoE models improve over the dense baseline at nearly unchanged nominal forward compute, while further increasing the number of stored experts produces little additional accuracy gain. Activating multiple experts per token yields additional predictive improvements at higher computational cost. Routing analyses show that expert assignments become more strongly associated with particle identity and kinematics in some configurations, but this structure does not increase monotonically with classification performance. These results highlight the need to distinguis
    
[^373]: 将大语言模型用作智能体：代价几何？

    Harnessing LLMs as Agents: What Does It Cost?

    [https://arxiv.org/abs/2610.02488](https://arxiv.org/abs/2610.02488)

    该论文提出语言模型智能体机（LAM）这一资源受限的计算抽象，首次从通信、内存访问、重计算与验证等方面严格量化 LLM 智能体驱动框架所消耗的计算资源，并给出相应的计算量下界。

    

    语言模型智能体日益依赖“驱动框架”来管理有限上下文、持久记忆、工具、验证与重复执行，然而现有的模型能力概念并未量化这些机制所消耗的计算资源。我们提出了语言模型智能体机，这是一种资源受限的抽象，它在固定底层语义模型的同时，显式地对驱动框架层面的资源进行计费。我们建立了四类结果。通信：在调用—传输预算同时受限的情况下，LAM 的执行在实例级别上等价于红蓝卵石游戏，从而将经典的 I/O 下界转移到上下文—内存流量上。访问：内存接口会引发渐近分离，包括在指针追踪任务上随机访问与非投机顺序访问之间 Θ(n) 的差距。重计算：位反转 DAG 在上下文容量为 C、持久记忆容量为 ……（摘要原文在此处截断）

    arXiv:2610.02488v1 Announce Type: new  Abstract: Language-model agents increasingly rely on harnesses that manage bounded context, persistent memory, tools, verification, and repeated execution, yet existing notions of model capability do not quantify the computational resources these mechanisms consume. We introduce the Language Model Agent Machine (LAM), a resource-bounded abstraction that fixes the underlying semantic model while explicitly charging harness-level resources. We establish four classes of results. Communication: LAM execution is instancewise equivalent to red--blue pebbling under simultaneous call--transfer budgets, transferring classical I/O lower bounds to context--memory traffic. Access: memory interfaces induce asymptotic separations, including a $\Theta(n)$ gap between random and non-speculative sequential access on pointer chasing. Recomputation: bit-reversal DAGs require $\Theta(n^2/(C+S)+n)$ model calls with context capacity $C$ and persistent-memory capacity $
    
[^374]: TACO：面向大语言模型微调的三值逐列绝对最大值单稀疏优化器

    TACO: Ternary Absolute-max Column-wise One-sparse Optimizer for LLM Fine-Tuning

    [https://arxiv.org/abs/2610.02199](https://arxiv.org/abs/2610.02199)

    本文提出TACO优化器，在维度归一化算子范数下计算精确的最速下降方向，以三值逐列单稀疏的形式存储优化器状态，在不牺牲精度和计算效率的前提下大幅降低大语言模型微调中的优化器内存开销。

    

    大语言模型（LLM）的全参数微调会带来巨大的优化器状态内存开销，限制了能够在现代GPU上容纳的模型规模。现有方法要么压缩优化器状态，要么放弃一阶梯度，要么在保留稠密状态的同时改变更新的几何结构。最近提出的Muon优化器通过矩阵值更新来减少优化器内存，但其几何结构与AdamW不同，在微调以AdamW预训练的模型时可能导致性能下降。为了在大语言模型微调中降低优化器内存而不牺牲精度或计算效率，我们提出了三值逐列绝对最大值单稀疏优化器，简称TACO。TACO遵循Muon的算子范数最速下降视角，但在几何路径上更进一步：它通过选取（三值化后的）最大幅值元素的符号，在维度归一化的 $1\to1$ 算子范数下计算精确的最速下降方向，从而……

    arXiv:2610.02199v1 Announce Type: new  Abstract: Full-parameter fine-tuning of large language models (LLMs) incurs substantial optimizer state memory overhead, limiting the model sizes that fit on modern GPUs. Existing approaches either compress optimizer state, abandon first-order gradients, or change the update geometry while retaining dense state. The recently introduced Muon optimizer reduces optimizer memory through matrix-valued updates. Still, its geometry differs from AdamW and can lead to performance degradation when fine-tuning AdamW-pretrained models. To reduce optimizer memory without sacrificing accuracy or computational efficiency in LLM fine-tuning, we propose Ternary Absolute-max Column-wise One-sparse optimizer, or TACO, which follows Muon's operator-norm steepest-descent view but takes the geometric route further. TACO computes the exact steepest-descent direction under a dimension-normalized $1\to1$ operator norm by selecting the sign of the largest magnitude entry i
    
[^375]: 基于真实站点观测的天气数据同化生成模型基准测试

    Benchmarking Generative Models for Weather Data Assimilation on Real Station Observations

    [https://arxiv.org/abs/2610.00728](https://arxiv.org/abs/2610.00728)

    该研究提出了首个基于真实气象站观测数据的生成式天气数据同化受控基准测试，利用美国本土11,849个NOAA MADIS站点数据，在固定数据集、观测算子和深度学习架构的条件下，系统比较了扩散模型与流匹配、像素空间与潜在空间等主要设计选择的有效性。

    

    天气再分析产品依赖于计算密集型的数值天气预报，随后通过数据同化将预报结果向观测值进行校正。深度生成模型提供了一种更廉价的替代方案，能够将大量计算成本从推理阶段转移到离线训练阶段。然而，现有的生成式方法通常是在合成观测数据上评估的，或者是在不同的数据集和评估方案下进行比较的，这使得哪些设计选择能够真正改善真实世界的数据同化变得不明确。我们提出了首个在真实气象站观测数据上进行的生成式天气数据同化受控基准测试。使用覆盖美国本土的11,849个NOAA MADIS站点和四个天气变量，我们在固定数据集、观测算子和深度学习架构的条件下评估各种方法。该基准测试比较了主要的设计选择，包括扩散模型与流匹配、像素空间与潜在空间（摘要原文在此处被截断）。

    arXiv:2610.00728v1 Announce Type: cross  Abstract: Weather reanalysis products rely on computationally intensive numerical weather predictions followed by data assimilation that corrects the forecast toward observations. Deep generative models offer a cheaper alternative that shifts much of this cost from inference to offline training. However, existing generative approaches have been evaluated on synthetic observations or under different datasets and evaluation schemes, making it unclear which design choices actually improve real-world data assimilation. We present the first controlled benchmark of generative weather data assimilation on real weather station observations. Using 11,849 NOAA MADIS stations across the contiguous United States and four weather variables, we evaluate methods while holding the dataset, observation operator, and deep learning architecture fixed. The benchmark compares the major design choices, including diffusion versus flow matching, pixel versus latent-spa
    
[^376]: 面向自监督全切片图像浓缩的非参数分布匹配

    Nonparametric Distribution Matching for Self-Supervised Whole-Slide Image Condensation

    [https://arxiv.org/abs/2610.00678](https://arxiv.org/abs/2610.00678)

    提出NICER框架，将自监督全切片图像浓缩重新表述为分布匹配问题，利用具有切片自适应容量的非参数先验显式保留学习相关特征分布，在五个组织病理学数据集上平均准确率提升7.44%并获得病理学家临床评估认可。

    

    组织病理学全切片图像（WSIs）是计算病理学的核心，但其极高的分辨率（每张切片通常达数GB）带来了严峻的计算挑战。为了实现可扩展的学习，现有方法采用自监督数据浓缩来降低计算成本，但通常依赖于启发式的原型学习，且未显式地为下游任务保留与学习相关的特征分布。为此，我们将WSI浓缩问题从原理上重新表述为固定表征视角下的分布匹配问题，并开发了NICER——一个基于具有切片自适应容量的非参数先验的可处理近似框架。在五个组织病理学数据集上的实验，以及经委员会认证的病理学家的临床评估均表明，NICER持续优于先前的方法，平均准确率提升达7.44%，同时……

    arXiv:2610.00678v1 Announce Type: cross  Abstract: Histological whole-slide images (WSIs) are central to computational pathology but pose severe computational challenges due to their extremely high resolution, often spanning several gigabytes per slide. To enable scalable learning, existing methods apply self-supervised data condensation to reduce computational cost, but typically rely on heuristic prototype learning and do not explicitly preserve learning-relevant feature distributions for downstream tasks. In response, we introduce a principled reformulation of WSI condensation as a distribution-matching problem under a fixed representational lens, and develop NICER, a tractable approximation framework based on a nonparametric prior with slide-adaptive capacity. Experiments on five histopathology datasets, together with clinical evaluation from a board-certified pathologist, show that NICER consistently outperforms prior methods, achieving an average accuracy improvement of 7.44% whi
    
[^377]: 在自动化决策门控中将系统一决策模型与训练分类器和语言模型进行基准测试

    Benchmarking System One decision models against trained classifiers and language models for automated decision gates

    [https://arxiv.org/abs/2610.00346](https://arxiv.org/abs/2610.00346)

    该研究在匹配条件下对系统一决策模型、训练分类器和生成式语言模型进行统一基准测试，发现模型类别的优劣取决于标签可用性：有标签时小型训练分类器表现最佳，无标签时决策模型在工作流任务上普遍优于零样本分类器。

    

    将分支决策交给模型的软件需要一个明确的选项和一个可设定阈值的概率。类型化决策模型，也称为系统一（System One）模型，无需生成文本即可返回此类概率，而监督分类器和生成式语言模型则是既有的替代方案。在匹配条件下，同一测试框架向来自六个系列的八个决策模型检查点（包括托管模型Jev）以及两个生成式比较模型发送相同的语义请求，并在相同的工作流、意图和社会科学条目上对监督分类器和零样本分类器进行评分。模型类别的排名取决于具体条件。在使用任务自身标签时，小型训练分类器在意图任务上最为准确，在工作流任务上与最佳决策模型无显著差异。在没有标签的情况下，除基于编码器的检查点外，所有决策模型在工作流任务上的表现都超过了零样本蕴含分类器。

    arXiv:2610.00346v1 Announce Type: new  Abstract: Software that hands branching decisions to a model needs a declared option and a probability it can threshold. Typed decision models, also called System One models, return such probabilities without generating text, while supervised classifiers and generative language models are the established alternatives. Under matched conditions, one harness sends eight decision-model checkpoints from six families, including the hosted model Jev, and two generative comparators the same semantic requests, and scores supervised and zero-shot classifiers on the same workflow, intent and social-science items. The ranking of the model classes depends on the conditions. With the task's own labels, small trained classifiers are the most accurate on intents and not significantly different from the best decision models on workflows. Without labels, every decision model except the encoder-based checkpoints exceeds a zero-shot entailment classifier on workflows
    
[^378]: OpenTSLM TeeMoE：用于预测、情境预测与推理的统一时间序列语言模型

    OpenTSLM TeeMoE: A Unified Time-Series Language Model for Forecasting, Contextual Prediction, and Reasoning

    [https://arxiv.org/abs/2609.40265](https://arxiv.org/abs/2609.40265)

    OpenTSLM TeeMoE 通过在共享主干上训练的 LoRA 混合专家架构，将时间序列直接预测、情境条件预测和语言化时间推理这三种异构能力统一到单一通用模型中，且不牺牲各项单独性能。

    

    现实世界中的时间序列应用日益需要能够处理时间序列预测、基于情境的条件预测以及基于语言的时间推理的模型。然而，当前的时间序列基础模型在这些能力上仍然是割裂的：数值专家模型通常能提供最强的预测效果，而基于语言的模型则提供更广泛的情境理解与分析能力。一个核心挑战在于如何统一这些异构能力，同时不降低各自的性能。我们提出了 OpenTSLM TeeMoE，这是一个通用的时间序列语言模型，它可以直接从观测到的时间序列进行预测，能够基于文本情境与时间模式进行推理，并可以综合和优化来自外部数值预测专家模型的预测结果。我们在一个共享主干之上独立训练了三个低秩专家，分别用于预测聚合、原生预测和时间分析。一个通过学习得到的 LoRA 混合专家

    arXiv:2609.40265v1 Announce Type: new  Abstract: Real-world time-series applications increasingly require models that can handle time series forecasting, context-conditioned prediction, and language-based temporal reasoning. Yet current time-series foundation models remain fragmented across these capabilities: numerical specialists often provide the strongest forecasts, while language-based models offer broader contextual understanding and analysis. A central challenge is to unify these heterogeneous capabilities without reducing their individual performance. We introduce OpenTSLM TeeMoE, a generalist time-series language model that can forecast directly from observed time series, reason over textual context and temporal patterns, and synthesize and refine predictions from external numerical forecasting specialists. We independently train three low-rank experts for forecast aggregation, native forecasting, and temporal analysis over a shared backbone. A learned LoRA mixture-of-experts 
    
[^379]: STARS：从时空动态到人机交互中的社会表征

    STARS: From Spatiotemporal Dynamics to Social Representations in Human-Robot Interaction

    [https://arxiv.org/abs/2609.40245](https://arxiv.org/abs/2609.40245)

    本文提出了SocialNav-SUB基准，通过视觉问答的形式系统评估视觉-语言模型在理解复杂社会导航场景（包括智能体间时空关系和人类意图推断）方面的能力，填补了社会机器人导航领域VLM评估的空白。

    

    机器人在动态的、以人为中心的环境中进行导航，需要基于稳健场景理解的符合社会规范的决策。近期的视觉-语言模型（VLMs）展现出物体识别、常识推理和情境理解等有前景的能力，这些能力与社会机器人导航的细致要求相契合。然而，VLMs能否准确理解复杂的社会导航场景（例如，推断各智能体之间的时空关系和人类意图）仍不清楚，而这对于实现安全且符合社会规范的机器人导航至关重要。尽管一些近期工作已经探索了VLMs在社会机器人导航中的应用，但目前尚无现有工作系统地评估它们满足这些必要条件的能力。本文引入了社会导航场景理解基准，这是一个视觉问答（VQA）数据集和基准……

    arXiv:2609.40245v1 Announce Type: cross  Abstract: Robot navigation in dynamic, human-centered environments requires socially-compliant decisions grounded in robust scene understanding. Recent Vision-Language Models (VLMs) exhibit promising capabilities such as object recognition, common-sense reasoning, and contextual understanding, capabilities that align with the nuanced requirements of social robot navigation. However, it remains unclear whether VLMs can accurately understand complex social navigation scenes (e.g., inferring the spatial-temporal relations among agents and human intentions), which is essential for safe and socially compliant robot navigation. While some recent works have explored the use of VLMs in social robot navigation, no existing work systematically evaluates their ability to meet these necessary conditions. In this paper, we introduce the Social Navigation Scene Understanding Benchmark (SocialNav-SUB), a Visual Question Answering (VQA) dataset and benchmark de
    
[^380]: 基于深度学习的三混合多用户MIMO预编码：电磁可重构天线的福音

    Deep Learning-Based Tri-Hybrid Multi-User MIMO Precoding: The Blessing of EM-Reconfigurable Antennas

    [https://arxiv.org/abs/2609.39167](https://arxiv.org/abs/2609.39167)

    本文提出基于Conformer神经网络架构的三混合预编码网络Tri-PNet，通过将电磁可重构天线引入的电磁域预编码与模数混合预编码进行联合设计，显著提升了宽带多用户MIMO-OFDM系统的频谱效率。

    

    电磁（EM）可重构天线为每个天线单元提供多种候选辐射方向图，从而引入了额外的电磁域自由度。将辐射方向图可重构性（实现为电磁域预编码）与传统的模数混合预编码相结合，产生了三混合多输入多输出（MIMO）预编码，这可以显著提升宽带多用户MIMO正交频分复用（OFDM）系统的频谱效率。然而，电磁、模拟和数字预编码的联合设计仍然具有挑战性。为了解决这一挑战，我们提出了一种基于Conformer的三混合预编码网络，Conformer是一种新兴的神经网络架构，它结合了卷积神经网络的局部建模能力与Transformer的全局依赖建模能力。此外，两种代表性的辐射方向图模式，即非规则模式和第3种……（摘要到此截断）

    arXiv:2609.39167v1 Announce Type: cross  Abstract: Electromagnetic (EM)-reconfigurable antennas provide multiple candidate radiation patterns per element, thereby introducing an additional EM-domain degree of freedom. Integrating radiation-pattern reconfigurability, realized as EM-domain precoding, with conventional hybrid analog-digital precoding yields tri-hybrid multiple-input multiple-output (MIMO) precoding, which can substantially improve the spectral efficiency of wideband multi-user MIMO orthogonal frequency-division multiplexing (OFDM) systems. However, the joint design of EM, analog, and digital precoding remains challenging. To address this challenge, we propose a tri-hybrid precoding network (Tri-PNet) based on Conformer, an emerging neural architecture that combines the local modeling strength of convolutional neural networks with the global dependency modeling of Transformers. Furthermore, two representative radiation-pattern modes, i.e., the non-regular mode and the 3rd 
    
[^381]: 基于Stackelberg博弈的多大语言模型协同对齐

    Multi-LLM Collaborative Alignment via Stackelberg Games

    [https://arxiv.org/abs/2609.39076](https://arxiv.org/abs/2609.39076)

    该论文提出受博弈论启发的Stackelberg对齐框架，由EXP3老虎机作为领导者根据指令难度和回应可区分性自适应地分配采样预算，将指令选择变为自适应课程，从而提升多个大语言模型相互学习、协同对齐的效果。

    

    一组语言模型可以通过相互学习彼此的回应来协作并共同提升。这些交互依赖于训练期间所使用的指令。现有方法通常均匀地采样指令，即使指令的有用性会随着模型能力的提升而变化：某个曾经模型回应质量参差不齐的指令，之后可能被所有模型回答得同样好；而一个以前困难的指令可能开始提供有用的学习信号。我们提出Stackelberg对齐，这是一个受博弈论启发的领导者-跟随者框架，将指令选择转化为自适应课程。EXP3多臂老虎机作为领导者，在各个指令之间分配固定的采样预算，并使用一种结合指令难度和回应可区分性的奖励来更新其采样分布。语言模型则作为跟随者：它们对所选指令生成回应，并相互评估彼此的回应

    arXiv:2609.39076v1 Announce Type: new  Abstract: A pool of language models can collaborate and improve collectively by learning from one another's responses. These interactions depend on the instructions used during training. Existing methods typically sample instructions uniformly, even though their usefulness may change as the models improve: an instruction on which models' responses once differed in quality may later be answered equally well, while a previously difficult instruction may begin to provide a useful learning signal. We propose Stackelberg Alignment, a game-theory-inspired leader-follower framework that turns instruction selection into an adaptive curriculum. An EXP3 bandit acts as the leader, allocating a fixed sampling budget across instructions and updating its sampling distribution using a reward that combines instruction difficulty and response discriminability. The language models act as followers: they respond to the selected instructions, evaluate one another's r
    
[^382]: 单次反事实补救的有符号几何：路径上有效性与带符号曲率判据

    The Signed Geometry of One-Shot Recourse: On-Path Validity and the Signed-Curvature Criterion

    [https://arxiv.org/abs/2609.36252](https://arxiv.org/abs/2609.36252)

    该论文证明单次解析式反事实补救能否一步成功由路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\hat g$ 的符号决定（非负则有效），给出仅凭分数与梯度的规则不可避免存在 $Kd_p^2/\|\nabla f(x)\|$ 量级过冲的下界，并证明在利普希茨曲率下于承诺点评估一次分数即可达到极小极大最优有效性。

    

    闭式（解析式）补救方法将一个被分类器拒绝的用户沿着分类器分数 $f$ 的单位梯度 $\hat g$ 移动承诺距离 $d_p=|f(x)|/\|\nabla f(x)\|$，在该处线性化后的分数恰好降为零。我们研究这一单步操作何时会成功，以及额外的模型查询能带来什么改变。在主阶近似下，该步恰好落在有利一侧当且仅当路径曲率 $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ 非负。在80个浅层模型上，被拒用户中一步落在有利侧的比例与 $\kappa\ge0$ 的比例相关系数高达 $r=0.985$，但在 Fashion-MNIST 上前者平均比后者低8.2个百分点。任何仅使用分数值和梯度的规则，都不可能对所有路径曲率以 $K$ 为界的分数都保持有效，除非对其中某些分数产生 $Kd_p^2/\|\nabla f(x)\|$ 量级的过冲。当曲率还是利普希茨连续且步长较短时，在承诺点处对 $f$ 的一次评估即可达到极小极大（minimax）最优。

    arXiv:2609.36252v1 Announce Type: new  Abstract: Closed-form recourse moves a rejected user along the unit gradient $\hat g$ of the classifier score $f$ by the promised distance $d_p=|f(x)|/\|\nabla f(x)\|$, at which the linearized score reaches zero. We ask when this one-shot step succeeds and what additional model queries change. To leading order the step ends on the favorable side exactly when the path curvature $\kappa=\hat g^\top\nabla^2 f(x)\,\hat g$ is nonnegative. Across 80 shallow models, the fraction of rejected users whose step ends there and the fraction with $\kappa\ge0$ correlate at $r=0.985$, although on Fashion-MNIST the first falls below the second by 8.2 points on average. No rule that uses only the score value and gradient can be valid for every score with path curvature bounded by $K$ without overshooting some by order $Kd_p^2/\|\nabla f(x)\|$. When the curvature is also Lipschitz and the step is short, one evaluation of $f$ at the promised point attains the minimax
    
[^383]: 面向基于采样的潜在规划的控制几何拉直方法

    Control-Geometry Straightening for Sampling-Based Latent Planning

    [https://arxiv.org/abs/2609.35603](https://arxiv.org/abs/2609.35603)

    提出控制几何拉直（CGS）这一辅助损失，通过将动作间余弦相似度与潜在差异对齐来学习对规划器友好的表示，从而提升基于采样的潜在规划的优化效率，并给出相应理论保证。

    

    联合嵌入预测架构使基于潜在世界模型的规划成为可能，但仅有精确的转移预测并不能保证规划目标易于优化。我们提出了控制几何拉直（Control-Geometry Straightening, CGS），这是一种单一的辅助损失函数，通过直接拉直控制几何以实现采样高效的规划，从而学习对规划器友好的表示。CGS 仅利用来自像素-动作对的局部转移，将动作之间的成对余弦相似度与相应潜在差异之间的成对余弦相似度进行匹配。该损失可应用于各种世界模型架构，支持端到端学习的表示或预训练表示。在线性动力学条件下，我们的理论分析将该目标与时间维度的拉直以及整个规划范围内更均衡的终端代价曲率联系起来，为 MPPI 提供了有限预算保证，为 CEM 提供了局部收缩结果，并为梯度下降提供了收敛界。

    arXiv:2609.35603v2 Announce Type: replace  Abstract: Joint-embedding predictive architectures enable planning with latent world models, but accurate transition prediction alone does not ensure that the planning objective is easy to optimize. We introduce Control-Geometry Straightening (CGS), a single auxiliary loss that learns planner-friendly representations by directly straightening control geometry for sampling-efficient planning. CGS matches pairwise cosine similarities among actions to those among corresponding latent differences only using local transitions from pixel-action pairs. The loss can be applied across world-model architectures using end-to-end learned or pretrained representations. Under linear-dynamics, our theoretical analysis connects this objective to temporal straightening and more balanced terminal-cost curvature across the full planning horizon, yielding finite-budget guarantees for MPPI, local contraction results for CEM, and convergence bounds for gradient des
    
[^384]: 残差流负担塑造扩散Transformer中的表征学习

    Residual-Stream Burden Shapes Representation Learning in Diffusion Transformers

    [https://arxiv.org/abs/2609.33895](https://arxiv.org/abs/2609.33895)

    该论文提出“残差流负担”概念，解释了扩散Transformer在数学上等价的预测目标（干净数据、噪声、速度）下表现不对称的原因：噪声目标要求残差流在整个深度保留噪声相关变化，迫使后续层在带噪表征上计算，而干净的预测目标负担更轻，为后续计算组织隐藏表征留下了更大自由度。

    

    在基于扩散的生成模型中，神经网络可以被训练来从带噪输入中预测干净数据、噪声或速度。这些预测目标彼此可相互转换，并描述着同一个生成过程，然而在大像素块上运行的普通扩散Transformer（Diffusion Transformer）采用干净预测时能够成功，而采用噪声或速度预测时却会失败。我们认为这种不对称性源于：噪声目标要求残差流在整个网络深度中保留与噪声相关的输入变化，以供最终读取，从而迫使后续层在带噪的表征上进行计算。而频谱集中的干净目标带来的负担较轻，为组织用于后续计算的隐藏表征留下了更大的自由度。我们将这种保留要求称为*残差流负担*，并展示了它如何塑造扩散Transformer中的表征学习。受控实验表明，可利用的结

    arXiv:2609.33895v2 Announce Type: replace-cross  Abstract: In diffusion-based generation, a neural network can be trained to predict the clean data, the noise, or the velocity from a noisy input. These prediction targets are interconvertible and describe the same generative process, yet plain Diffusion Transformers operating on large pixel patches succeed with clean prediction and fail with noise or velocity prediction. We argue that this asymmetry arises because noisy targets require the residual stream to preserve noise-dependent input variation through depth for the final readout, forcing subsequent layers to compute on noisy representations. A spectrally concentrated clean target imposes a lighter demand, leaving greater freedom to organize hidden representations for subsequent computation. We call this preservation requirement *residual-stream burden* and show how it shapes representation learning in Diffusion Transformers. Controlled experiments indicate that the exploitable stru
    
[^385]: Train4Merge：基于OPD模型合并中强化学习与监督微调教师的受控单教师研究

    Train4Merge: A Controlled Single-Teacher Study of RL vs. SFT Teachers for OPD-Based Model Merging

    [https://arxiv.org/abs/2609.32303](https://arxiv.org/abs/2609.32303)

    该研究通过受控单教师实验首次系统比较了强化学习（RL）与监督微调（SFT）两种训练算法在基于在线策略蒸馏的模型合并中的效果，发现在智能体、推理和感知三个领域中，RL训练的教师均能产生更强的学生模型，性能分别领先4.27、1.50和0.86个百分点。

    

    从共享检查点训练得到的领域专家可以通过在线策略蒸馏将他们的专业能力迁移到单个学生模型中。现有研究主要聚焦于改进这一合并过程，而用于训练专家的算法却缺乏系统性的比较。我们通过在智能体、推理和感知三个领域对监督微调（SFT）和强化学习（RL）进行受控单教师比较，研究哪种训练算法能产生更适合OPD的教师模型。教师和学生共享相同的Qwen3.5-9B初始化，且两种类型的教师在相近的任务性能水平上进行比较。我们的实验表明，在所有三个领域中，RL教师都能产生更强的学生模型，并能更高程度地恢复教师模型带来的性能增益。在各自的最佳检查点上，RL指导的学生模型分别以4.27、1.50和0.86个百分点的优势超越SFT指导的学生模型。

    arXiv:2609.32303v2 Announce Type: replace-cross  Abstract: Domain experts trained from a shared checkpoint can transfer their specialized capabilities to a single student through on-policy distillation (OPD). Existing research primarily focuses on improving this merging process, while the algorithms used to train the experts have received limited systematic comparison. We investigate which training algorithm produces teachers better suited to OPD through controlled single-teacher comparisons of supervised fine-tuning (SFT) and reinforcement learning (RL) across Agentic, Reasoning, and Perception. Teachers and students share the same Qwen3.5-9B initialization, and the two teacher types are compared at similar task performance. Our experiments show that RL teachers yield stronger students and higher recovery of teacher performance gains across all three domains. At their best checkpoints, RL-guided students outperform SFT-guided students by 4.27, 1.50, and 0.86 percentage points, respect
    
[^386]: 距离依赖矩条件下的普通非凸SGD：有限时域平稳性与Nagaev界

    Ordinary Nonconvex SGD under Distance-Dependent Moments: Finite-Horizon Stationarity and Nagaev Bounds

    [https://arxiv.org/abs/2609.30499](https://arxiv.org/abs/2609.30499)

    本文证明，当条件矩允许噪声方差随迭代点距离增长时，普通单样本SGD无需任何修改即可达到与Blum–Gladyshev下界匹配的极小极大随机复杂度，并借助希尔伯特空间Fuk–Nagaev不等式给出高概率Nagaev型界。

    

    统一的噪声矩界假设排除了那些变异性随迭代点位置增长的随机梯度。我们在距离依赖的条件矩假设下，研究针对光滑、下有界且可能非凸目标的普通单样本随机梯度下降。仅利用二阶矩条件，一个直接的“下降—位移”论证在使用依赖时域的步长时，给出了 $T^{-1/3}$ 的期望平均平方梯度平稳性。一个显式的预言机复杂度推论与已知的平滑Blum–Gladyshev（BG-0）下界相匹配，包括 $Lb_2\Delta^3\varepsilon^{-6}$ 和 $L\Delta\sigma^2\varepsilon^{-4}$ 两个随机项，其中 $\Delta$ 为初始目标间隙，$\sigma^2+b_2\|x-x_1\|^2$ 为方差的上界。因此，无需任何修改的SGD在这一二阶矩类别中即达到极小极大随机复杂度。对于 $p>2$，可预测局部化技术与希尔伯特空间上的Fuk–Nagaev不等式给出了一个高概率界，分离对数……（摘要在此处被截断）

    arXiv:2609.30499v1 Announce Type: new  Abstract: Uniform noise-moment bounds exclude stochastic gradients whose variability increases with the iterate. We study ordinary, single-sample stochastic gradient descent for smooth, lower-bounded, possibly nonconvex objectives under distance-dependent conditional moments. Under second moments alone, a direct descent--displacement argument yields $T^{-1/3}$ expected average squared-gradient stationarity with a horizon-dependent stepsize. An explicit oracle-complexity corollary matches the known smooth Blum--Gladyshev (BG-0) lower bound, including the $Lb_2\Delta^3\varepsilon^{-6}$ and $L\Delta\sigma^2\varepsilon^{-4}$ stochastic terms, where $\Delta$ is the initial objective gap and $\sigma^2+b_2\|x-x_1\|^2$ bounds the variance. Thus unchanged SGD attains the minimax stochastic complexity in this second-moment class. For $p>2$, predictable localization and a Hilbert-space Fuk--Nagaev inequality yield a high-probability bound separating logarith
    
[^387]: GridSFM：用于求解交流最优潮流的基础模型

    GridSFM: A Foundation Model for Solving AC Optimal Power Flow

    [https://arxiv.org/abs/2609.30173](https://arxiv.org/abs/2609.30173)

    GridSFM是一个1500万参数的物理启发图神经网络基础模型，通过在54种电网拓扑上预训练并结合基于牛顿法的物理信息微调，仅需100个求解实例即可适应多达10000节点的未见电网，实现2.45%的零样本发电成本误差，且优于使用更多数据训练的单一拓扑专用神经网络模型。

    

    我们提出GridSFM，这是一个将跨电网拓扑预训练的基础模型与物理信息微调相结合的框架，用于大规模求解交流最优潮流（AC-OPF）。它是一个拥有1500万参数的物理启发图神经网络，在54种拓扑结构（500至4000节点）上进行了预训练。我们的模型在10000节点系统的保留运行工况上实现了2.45%的零样本发电成本误差，且随着系统规模增长性能没有退化。在此基础上，我们将预训练主干网络与基于牛顿潮流法的物理信息微调设计相结合。仅需100个已求解的实例，GridSFM即可适应多达10000节点的未见电网。我们证明，当作为热启动点部署时，该模型在成本和求解器迭代次数方面均优于使用更多数据训练的单一拓扑专用神经网络模型。在设计该基础模型时，我们克服了……（原文摘要至此截断）

    arXiv:2609.30173v1 Announce Type: cross  Abstract: We introduce GridSFM, a framework that combines a pretrained foundation model across grid topologies with physics-informed fine-tuning for solving AC Optimal Power Flow (AC-OPF) at scale. It is a $15$ million parameter physics-inspired graph neural network pretrained across $54$ topologies of $500$ to $4{,}000$ buses. Our model attains a $2.45\%$ zero-shot generation-cost error on a $10{,}000$ bus case held-out operating conditions with no degradation as system size grows. Building on this, we pair the pretrained backbone with a physics-informed fine-tuning design based on Newton's method for power flow. With only $100$ solved instances, GridSFM adapts to unseen grids up to $10{,}000$ buses. We show it out performs single topology, dedicated neural network models that are trained more data, both in terms of cost and solver iterations when deployed as warm starting points.   In designing this foundation model, we overcome the fact that 
    
[^388]: QUARTET：基于四分支交叉注意力与随机游走轨迹的关系图Transformer增强方法

    QUARTET: Quad-branch cross-Attention and Random-walk Traces for Enhancing Transformers on Relational Graphs

    [https://arxiv.org/abs/2609.26855](https://arxiv.org/abs/2609.26855)

    提出QUARTET图Transformer架构，利用基于近期截断个性化PageRank的因果随机游走采样器提取密集连通且无时序泄露的局部子图，并通过四分支交叉注意力丰富全局上下文，从而克服RelGT在关系图建模中局部采样松散与全局记忆单一的局限。

    

    arXiv:2609.26855v1 公告类型：交叉 摘要：关系深度学习将多表数据库建模为异构时序图，图Transformer目前在RelBench等基准测试上取得了最先进的性能。然而，当前领先的模型RelGT存在两个关键局限：其随机局部采样器生成的子图连接松散，阻碍了消息传递；其全局注意力模块依赖于单一的、基于种子特征的内存，忽略了更广泛的宏观层面动态。为克服这些局限，我们提出了QUARTET，一种表达能力强的图Transformer架构，它在局部子图上应用完全自注意力，同时通过交叉注意力分支来丰富全局上下文。具体而言，QUARTET采用基于近期截断个性化PageRank（PPR）的因果随机游走（CRW）采样器，以提取紧凑、抗枢纽节点干扰且密集连通的局部子图，且不会产生时序信息泄露。与此同时，四分支交叉注意力（摘要在此处截断）

    arXiv:2609.26855v1 Announce Type: cross  Abstract: Relational Deep Learning (RDL) models multi-table databases as heterogeneous temporal graphs, and graph transformers currently achieve state-of-the-art performance on benchmarks like RelBench. However, the current leading model, RelGT, suffers from two key limitations: its random local sampler yields loosely connected subgraphs that hinder message passing, and its global attention module relies on a single, seed-feature-based memory that ignores broader macro-level dynamics. To overcome these limitations, we introduce QUARTET, an expressive graph transformer architecture that applies full self-attention on local subgraphs while enriching global context through cross-attention branches. Specifically, QUARTET employs a Causal Random Walk (CRW) sampler based on recency-truncated Personalized PageRank (PPR) to extract compact, hub-robust, and densely connected local subgraphs without temporal leakage. Concurrently, a quad-branch cross-atte
    
[^389]: 基于变分量子神经网络的端到端量子语义通信

    End-to-End Quantum Semantic Communication with Variational Quantum Neural Networks

    [https://arxiv.org/abs/2609.25044](https://arxiv.org/abs/2609.25044)

    本文提出了首个结合变分量子神经网络与语义通信的端到端量子语义通信框架，通过变分量子发射机与可训练量子接收机在含噪量子信道上传输压缩语义信息，并在理想、比特翻转、去极化和振幅阻尼等多种信道条件下验证了其分类性能。

    

    本文提出了一种结合量子机器学习（QML）与语义通信（SemCom）的量子语义通信（QSemCom）框架。经典数据被压缩为低维语义表示，由变分量子发射机进行编码和处理，通过量子信道传输，再由可训练的量子接收机处理以完成分类任务。该框架考虑了一种分布式量子通信场景，其中量子处理单元（QPU）通过量子链路交换与任务相关的语义信息。虽然一般设置可能涉及多个量子节点，但本工作聚焦于基本的双节点情形，即发射端与接收端QPU通过含噪量子信道相连。该框架使用MNIST数据集，在理想信道、比特翻转信道、去极化信道和振幅阻尼信道下进行评估。基线模型首先在完美信道上训练，然后在（摘要在此处被截断）

    arXiv:2609.25044v1 Announce Type: cross  Abstract: This paper presents a quantum semantic communication (QSemCom) framework combining quantum machine learning (QML) and semantic communication (SemCom). Classical data are compressed into low-dimensional semantic representations, encoded and processed by a variational quantum transmitter, transmitted through a quantum channel, and processed by a trainable quantum receiver for classification. The framework considers a distributed quantum communication scenario in which quantum processing units (QPUs) exchange task-relevant semantic information through quantum links. While the general setting may involve multiple quantum nodes, this work focuses on the fundamental two-node case, with transmitter and receiver QPUs connected through a noisy quantum channel. Using MNIST, the framework is evaluated under ideal, bit-flip, depolarizing, and amplitude-damping channels. A baseline model is first trained over a perfect channel and evaluated under i
    
[^390]: 折点与平滑性：类拉普拉斯源下实解析非线性独立成分分析（nICA）的可辨识性

    Kinks vs. Smoothness: Identifiability of Real Analytic nICA for Laplace-like Sources

    [https://arxiv.org/abs/2609.21926](https://arxiv.org/abs/2609.21926)

    该论文证明了当独立源的密度函数一阶导数存在有限个不连续点（如拉普拉斯分布）时，实解析非线性独立成分分析（nICA）在排除平凡歧义后是可辨识的，其核心证明思路是利用源分布中的折点与实解析函数平滑性之间的对比。

    

    许多机器学习系统试图用生成复杂数据的隐藏独立因子来解释这些数据——例如图像或金融时间序列。恢复真实的潜在因子，而非它们的某种打乱版本，是非线性独立成分分析（nICA）的核心挑战。我们证明了当源概率密度函数的一阶导数存在有限个不连续点时，实解析生成函数是可辨识的（可精确恢复，仅存在平凡的歧义）。拉普拉斯分布是满足该假设的最典型例子。我们的证明依赖于源分布中的折点与实解析函数平滑性之间的对比。实解析函数涵盖了广泛的一类生成机制，并且可以用具有标准激活函数（如tanh、softplus、GELU）的归一化流或变分自编码器来近似。

    arXiv:2609.21926v1 Announce Type: new  Abstract: Many machine learning systems try to explain complex data - like images or financial time series - in terms of hidden, independent factors that generated them. Recovering the true underlying factors, rather than some scrambled version of them, is the central challenge of nonlinear Independent Component Analysis (nICA). We prove identifiability (exact recovery) up to trivial ambiguities for real analytic generating functions when source probability density functions have a finite number of discontinuities in the first derivative. The Laplace distribution is the most prominent example satisfying this assumption. Our proof relies on the contrast between kinks in the source distribution and the smoothness of real analytic functions. Real analytic functions comprise a broad class of generating mechanisms, and can be approximated with Normalizing Flows or Variational Autoencoders with standard activation functions (e.g., tanh, softplus, GELU),
    
[^391]: 藏于寻常之中：一种基于扩散模型的视觉-语言模型地理定位隐私泄露缓解方法

    Hiding in Plain Sight: A Diffusion-based Mitigation of Geolocation Privacy Leakage in Vision-Language Models

    [https://arxiv.org/abs/2609.21363](https://arxiv.org/abs/2609.21363)

    该论文系统揭示了多模态大推理模型可通过视觉推理从照片精确推断用户地理位置的隐私威胁，指出拒绝式防护与像素空间扰动防御的不足，并提出了一种基于扩散模型的隐私泄露缓解方法。

    

    多模态大型推理模型（MLRMs）在复杂视觉理解方面展现出了卓越的能力。然而，这种强大的能力也带来了一种关键但尚未被充分研究的隐私威胁：攻击者可以利用MLRMs，通过对建筑风格、植被和光照条件等细微视觉线索进行结构化推理，从用户随手分享的照片中精确推断其地理位置。在这项工作中，我们对MLRM驱动的地理定位隐私泄露进行了系统性研究。我们首先揭示了基于拒绝回答的安全防护措施严重不足，因为精心构造的越狱提示词可以将模型的响应率提升至100%。我们进一步发现，现有的防御方法——即向共享图像中注入不可感知的扰动——存在像素空间优化固有的结构性局限，导致黑盒迁移性下降并产生明显的视觉伪影。受此启发，本文提出了一种基于扩散模型的隐私泄露缓解方法（摘要在此处截断）。

    arXiv:2609.21363v1 Announce Type: cross  Abstract: Multimodal large reasoning models (MLRMs) have demonstrated remarkable capabilities in complex visual understanding. However, this very power introduces a critical yet underexplored privacy threat: adversaries can exploit MLRMs to precisely infer users' geographic locations from casually shared photographs, by performing structured reasoning over subtle visual cues such as architectural styles, vegetation, and lighting conditions. In this work, we present a systematic study of MLRM-driven geolocation privacy leakage. We first reveal that refusal-based safeguards are critically insufficient, as carefully crafted jailbreak prompts can raise model response rates to 100%. We further identify that existing defenses, which inject imperceptible perturbations into shared images, suffer from structural limitations intrinsic to their pixel-space optimization, resulting in degraded black-box transferability and pronounced visual artifacts. Motiva
    
[^392]: FedGuide：面向异构联邦强化学习的扩散先验对齐与价值基线引导

    FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning

    [https://arxiv.org/abs/2609.18964](https://arxiv.org/abs/2609.18964)

    该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。

    

    联邦强化学习（FRL）使分布式智能体能够在异构环境中进行协同策略学习。尽管近期基于方差缩减、散度惩罚和动量优化的方法改进了异构设置下的FRL，但这些方法仍主要同步策略或价值网络参数，并未明确解决异构客户端之间的分布不匹配问题。因此，我们提出了**FedGuide**，一个使用扩散先验作为行为模型的FRL框架，为异构的本地策略学习提供由个性化数据支持的分布。FedGuide不直接对本地策略进行平均，而是通过最优传输混合专家（OT-MoE）聚合这些扩散先验，在分布空间中保留异构行为模式。此外，它还开发了分布校正估计（DICE）价值基线，以提供低方差、具有回报感知的引导。

    arXiv:2609.18964v1 Announce Type: new  Abstract: Federated Reinforcement Learning (FRL) enables collaborative policy learning across distributed agents with heterogeneous environments. While recent methods based on variance reduction, divergence penalization, and momentum optimization improve FRL under heterogeneous settings, they still primarily synchronize policy or value-network parameters and do not explicitly address distributional mismatch among heterogeneous clients. Therefore, we propose \textbf{FedGuide}, a FRL framework that uses diffusion priors as behavior models to provide personalized data supported distributions for heterogeneous local policy learning. Instead of directly averaging local policies, FedGuide aggregates those diffusion priors through Optimal-Transport Mixture-of-Experts (OT-MoE), preserving heterogeneous behavior modes in distribution space. It further develops a Distribution Correction Estimation (DICE) value baseline to provide low-variance, return-aware 
    
[^393]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^394]: 面向压缩兼容的稀疏长上下文大语言模型推理的自索引注意力

    Self-Indexing Attention for Compression-Compatible Sparse Long-Context LLM Inference

    [https://arxiv.org/abs/2609.13205](https://arxiv.org/abs/2609.13205)

    提出了一种免训练的自索引注意力框架，利用共享的1比特符号索引在预填充和解码阶段统一实现高效token检索，同时兼容外部KV缓存压缩，在5%注意力密度下达到接近密集注意力的准确率，并获得高达6.1倍预填充和10.3倍解码的算子加速。

    

    稀疏长上下文推理需要在预填充（prefill）和解码（decode）两个阶段都进行高效的token检索。现有方法通常对这两个阶段采用不同的检索策略，导致单一检索表示无法在整个推理过程中被复用。我们提出了自索引注意力，这是一个基于共享变换域符号-幅值表示的免训练框架。键值符号提供了一个可复用的token级索引，用于分组的预填充选择和解码检索，同时该表示与外部KV缓存压缩保持兼容，无需单独的索引器元数据。这种1比特索引通过现代加速器广泛支持的按位运算实现高效检索。在5%注意力密度下，自索引注意力在LongBench和RULER基准上仍接近密集注意力的表现，并实现了高达6.1倍的预填充和10.3倍的解码注意力算子加速。与TurboQuant和DeepSeekV4-Flash结合的实验进一步验证了该方法的有效性。

    arXiv:2609.13205v1 Announce Type: cross  Abstract: Sparse long-context inference requires efficient token retrieval in both prefill and decode. Existing methods often use different retrieval strategies for the two stages, preventing one retrieval representation from being reused throughout inference. We propose Self-Indexing Attention, a training-free framework built on a shared transform-domain sign-magnitude representation. The key signs provide a reusable token-level index for grouped prefill selection and decode retrieval, while the same representation remains compatible with external KV-cache compression without separate indexer metadata. This 1-bit index enables efficient retrieval through bitwise operations widely supported by modern accelerators. At 5% attention density, Self-Indexing Attention remains close to dense attention on LongBench and RULER and achieves up to 6.1x prefill and 10.3x decode attention-operator speedups. Experiments with TurboQuant and DeepSeekV4-Flash fur
    
[^395]: 远距离大梯度未必可靠：面向长时程自回归预测的可靠性加权信用分配

    Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting

    [https://arxiv.org/abs/2609.12890](https://arxiv.org/abs/2609.12890)

    提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。

    

    在自回归预测中，长预测展开能够提供远距离的监督信号，但通过时间的反向传播（BPTT）需要将这些损失的梯度经过许多自回归步骤逐步传递。反复的雅可比矩阵乘积可能使远距离梯度在参数更新中占据主导地位，同时放大可预测信号与不可预测噪声；因此，大的远距离梯度并不一定携带可靠的学习信号。基于这一观察，我们提出了内部双维纳路由（Internal-DW），这是一种仅作用于反向传播过程的原则性干预方法，它在保留完整前向展开和所有时域损失的同时，对内部梯度路由进行可靠性加权。在每个残差块处，我们为恒等路由和非线性路由推导出有界的维纳增益，以在保留可预测学习信号与抑制不可预测变化之间取得平衡，并通过路由级别的梯度统计量和显式噪声模型对这些增益进行估计。在一个受控的（摘要在此处截断）

    arXiv:2609.12890v1 Announce Type: new  Abstract: In autoregressive forecasting, long prediction rollouts provide distant supervision, but backpropagation through time (BPTT) carries gradients from those losses through many autoregressive steps. Repeated Jacobian products can make distant gradients dominate the update while amplifying predictable signal and unpredictable noise together; a large distant gradient therefore need not carry reliable learning signal. Motivated by this observation, we introduce Internal Dual-Wiener routing (Internal-DW), a principled backward-only intervention that preserves the full forward rollout and all horizon losses while reliability-weighting internal gradient routes. At each residual block, we derive bounded Wiener gains for the identity and nonlinear routes that balance preserving predictable learning signal against suppressing unpredictable variation, and estimate them from route-level gradient statistics and an explicit noise model. In a controlled 
    
[^396]: 数据稀缺与模型稀疏：混合专家模型对重复数据的过拟合更严重

    Data Scarcity and Model Sparsity: Mixtures-of-Experts Overfit More to Repeated Data

    [https://arxiv.org/abs/2609.11917](https://arxiv.org/abs/2609.11917)

    该研究发现混合专家模型（MoE）相比密集模型更容易因训练数据重复而过拟合，且这种退化随模型稀疏度（由总参数量而非活跃参数量决定）的增加而加剧。

    

    随着人类书写文本资源的枯竭，重复使用语言模型训练数据已成为标准做法。先前的工作研究了数据重复对密集激活的Transformer的影响，但对于近期占主导地位的稀疏架构（如混合专家模型，MoE），尽管其具有更高的计算效率，数据重复的影响在很大程度上仍未被探索。我们在单域和多域数据混合中，以及在不同的MoE设置（包括专家数量和粒度）下改变数据重复率。我们一致发现，对于活跃参数从8000万到10亿（总参数85亿）的模型，MoE在数据重复下的性能退化更为迅速。这种效应随稀疏性增加而加剧，且由总参数量而非活跃参数量决定。虽然8000万参数的密集模型可以在最小性能退化下将数据重复8倍，但MoE在4倍时就开始受损，并迅速恶化，在全部唯一数据的设置中将其性能优势拱手让给了……

    arXiv:2609.11917v1 Announce Type: cross  Abstract: As the supply of human-written text is exhausted, it has become standard practice to repeat language model training data. Prior work has studied data repetition for densely activated Transformers, but the effects of data repetition remains largely unexplored for recently dominant sparse architectures such as Mixture-of-Experts (MoE), despite their increased compute efficiency. We vary data repetition rates across single- and multi-domain data mixes, and across MoE settings, including expert count and granularity. We consistently find, for models ranging from 80M to 1B active (8.5B total) parameters, that MoEs degrade more rapidly under data repetition. This effect increases with sparsity, dictated by total rather than active parameters. While 80M dense models can repeat data over 8x with minimal degradation, MoEs instead begin to suffer at 4x, and deteriorate rapidly, ceding their performance benefits in all-unique data settings to und
    
[^397]: 凸域上参数估计的广义分数匹配

    Generalized Score Matching for Parameter Estimation on Convex Domains

    [https://arxiv.org/abs/2609.11521](https://arxiv.org/abs/2609.11521)

    本文从最小概率流学习出发，构造性地推导出凸域上的广义分数匹配目标函数，统一了经典分数匹配与非负数据的域适配变体，并证明该目标是二阶正当局部评分规则，保证最小化时能恢复真实密度。

    

    最大似然（ML）估计是学习概率模型的一种有原则且统计高效的方法。然而，对于非归一化模型，最大似然估计需要计算配分函数并对其进行求导，这在某些情况下可能并不可行。分数匹配提供了一种实际可行的替代方法，它通过以消除对归一化常数依赖的方式拟合分数，从而绕过了这一障碍。我们从最小概率流（MPF）学习出发，以构造性的方式推导出了 $\mathbb{R}^{d}$ 凸子集上的广义分数匹配目标函数，并展示了经典分数匹配以及适用于非负数据的域适配变体如何在该框架中自然产生。我们证明了所得到的目标函数是一个二阶的正当局部评分规则，这为最小化该目标函数时能够恢复真实密度提供了理论保证。

    arXiv:2609.11521v1 Announce Type: new  Abstract: Maximum likelihood (ML) estimation is a principled and statistically efficient approach for learning probabilistic models. However, for unnormalized models, ML estimation requires evaluating the partition function and differentiating through it, which may not always be tractable. Score matching provides a practically viable alternative that circumvents this obstacle by fitting the score in a way that eliminates dependence on the normalizing constant. We derive the generalized score matching objective on a convex subset of $\mathbb{R}^{d}$ constructively starting from Minimum Probability Flow (MPF) learning, and show how classical score matching as well as domain-adapted variants for non-negative data arise naturally within the proposed framework. We show that the resulting objective is a {\it proper local scoring rule} of second-order, which provides the theoretical guarantee that the true density is recovered when the objective is minim
    
[^398]: 语义瓶颈：利用语义表示实现非侵入式语音解码

    The Semantic Bottleneck: Leveraging Semantic Representations for Non-Invasive Speech Decoding

    [https://arxiv.org/abs/2609.10296](https://arxiv.org/abs/2609.10296)

    提出Brain2Semantics2Text方法，通过语义嵌入空间作为瓶颈，将句子级MEG信号映射到语义流形并逆向转换为文本，实现了无需词级对齐的非侵入式语音解码。

    

    非侵入式语音解码一直受限于神经记录的低信噪比，这使得对音素或单个单词的细粒度重建变得困难。受神经科学证据的启发——高级语义表示分布在大脑皮层的多个区域，并随较慢的时间尺度演化——我们假设语义内容可能比低级声学或词汇特征更适合作为非侵入式解码的目标。我们提出了Brain2Semantics2Text方法，通过一个中间语义嵌入空间来重建文本。我们的模型将句子级别的MEG（脑磁图）响应映射到语义流形中，然后将预测出的嵌入逆向转换为自然语言。这种语义瓶颈机制使得无需词级对齐即可恢复高级语义信息。我们描述了该方法的核心原理、具体实现，以及用于缓解相关挑战的策略。

    arXiv:2609.10296v1 Announce Type: new  Abstract: Non-invasive speech decoding remains constrained by the low signal-to-noise ratio of neural recordings, which makes fine-grained reconstruction of phonemes or individual words difficult. Motivated by neuroscientific evidence that high-level semantic representations are distributed across cortical regions and evolve over slower temporal scales, we hypothesize that semantic content may provide a more suitable target for non-invasive decoding than low-level acoustic or lexical features. We introduce Brain2Semantics2Text, a method that reconstructs text through an intermediate semantic embedding space. Our model maps sentence-level MEG responses into a semantic manifold and then inverts the predicted embeddings into natural language. This semantic bottleneck enables recovery of high-level meaning without word-level alignment. We describe the core principles of the approach, its implementation, and the strategies used to mitigate the challeng
    
[^399]: 分解引导的扩散语言模型用于惯性约束聚变预测

    Decomposition-Guided Diffusion Language Models for Inertial Confinement Fusion Prediction

    [https://arxiv.org/abs/2609.07756](https://arxiv.org/abs/2609.07756)

    提出据信首个基于语言模型的惯性约束聚变预测器ICF-DLM，通过物理类型化分解、双向去噪和物理驱动PPO奖励，直接从激光脉冲与靶设计参数准确预测中子率波形。

    

    arXiv:2609.07756v1 公告类型：新论文 摘要：惯性约束聚变（ICF）是迈向清洁能源的主要途径之一，但国家点火装置（NIF）每一次实验的成本约为一百万美元，这使得精确的AI代理模型成为极具价值的研究目标。我们研究了外生驱动的ICF波形预测问题，即需要直接从激光脉冲和靶设计参数推断出512步的中子率诊断信号，而无需观测任何历史响应数据。该场景对标准时间序列预测器提出了严峻挑战：时间稀疏性（纳秒级时间窗口中仅存在皮秒级峰值）、输入输出尺度不匹配（仅有不到300次真实实验数据），以及峰值敏感性（皮秒级时序精度）。我们提出了ICF-DLM，据我们所知这是首个基于语言模型的ICF预测器，它结合了：（i）物理类型化分解，将输出分解为产额 $Y_{DT}$、峰值时刻 $t_{\mathrm{peak}}$ 和局部波形 $w_{\mathrm{local}}$；（ii）双向去噪机制，推迟对峰值位置的确定；（iii）物理驱动的PPO奖励。

    arXiv:2609.07756v1 Announce Type: new  Abstract: Inertial confinement fusion (ICF) is a leading pathway toward clean energy, but each shot at the National Ignition Facility costs on the order of one million dollars, making accurate AI surrogates a high-value target. We study exogenous-driven ICF waveform prediction, where a 512-step neutron-rate diagnostic must be inferred directly from a laser pulse and target design parameters, with no historical response observed. The regime stresses standard time-series predictors with temporal sparsity (picosecond peak in a nanosecond window), input-output scale mismatch (under 300 real shots), and peak sensitivity (picosecond timing). We propose ICF-DLM, to our knowledge the first LM-based ICF predictor, combining (i) a physics-typed decomposition into yield $Y_{DT}$, peak timing $t_{\mathrm{peak}}$, and local waveform $w_{\mathrm{local}}$; (ii) bidirectional denoising that defers commitment to peak location; and (iii) a physics-driven PPO reward
    
[^400]: 混合Sobolev空间中的浅层神经网络逼近

    Shallow neural network approximation in mixed Sobolev spaces

    [https://arxiv.org/abs/2609.05263](https://arxiv.org/abs/2609.05263)

    该论文建立了与激活函数无关的傅里叶块原理，证明浅层神经网络对混合光滑度为 $\alpha$ 的函数的逼近代数阶为 $\min\{\alpha,\rho\}$，并通过匹配的下界确定了 $\mathrm{ReLU}^k$ 网络在任意维数下的最优代数逼近指数为 $\min\{\alpha,k+1\}$。

    

    我们研究了具有 $n$ 个神经元和一般激活函数的浅层神经网络对混合 Sobolev 空间的最优 $L_2$ 逼近。我们首先建立了一个与激活函数无关的傅里叶块原理：如果某激活函数在傅里叶块性质的意义下具有单变量逼近阶 $\rho$，那么对于混合光滑度为 $\alpha$ 的目标函数，全局逼近速率具有代数阶 $\min\{\alpha,\rho\}$，直至显式的对数因子。为了针对具体激活函数验证该性质，我们引入了一个结构化的单变量逼近条件，该条件以显式参数蕴含傅里叶块性质。对于 $\mathrm{ReLU}^k$，一个匹配的代数下界确定了 $\min\{\alpha,k+1\}$ 为任意维数下的最优代数逼近指数，上界中存在对数因子。该框架还对基数 B 样条给出了指数 $\min\{\alpha,k+1\}$，并……（原文在此截断）

    arXiv:2609.05263v1 Announce Type: cross  Abstract: We investigate the best $L_2$ approximation of mixed Sobolev spaces by shallow neural networks with $n$ neurons and general activation functions. We first establish an activation-independent Fourier-block principle: if an activation has univariate approximation order $\rho$ in the sense of the Fourier-block property, then the global approximation rate has algebraic order $\min\{\alpha,\rho\}$ for target functions of mixed smoothness $\alpha$, up to explicit logarithmic factors. To verify this property for concrete activations, we introduce a structured univariate approximation condition that implies the Fourier-block property with explicit parameters. For $\mathrm{ReLU}^k$, a matching algebraic lower bound identifies $\min\{\alpha,k+1\}$ as the optimal algebraic approximation exponent in any dimension, up to logarithmic factors in the upper bound. The framework also yields the exponent $\min\{\alpha,k+1\}$ for cardinal B-splines and so
    
[^401]: 并发随机博弈的鲁棒PAC学习

    Robust PAC Learning of Concurrent Stochastic Games

    [https://arxiv.org/abs/2609.04189](https://arxiv.org/abs/2609.04189)

    该论文提出了首个针对具有转移不确定性的广义和并发随机博弈的PAC学习框架，通过引入纳什裕度刻画解决了均衡存在性问题，能在多项式样本复杂度下返回社会福利近优的ε-近似纳什均衡或证明精确纳什均衡不存在。

    

    我们提出了首个针对具有转移不确定性的广义和并发随机博弈的概率近似正确（PAC）学习框架，同时解决了纳什均衡存在性这一难题。我们的算法在转移核上维护数据驱动的 $L^1$ 置信集，并求解鲁棒CSG以计算社会福利最优的 $\varepsilon$-纳什均衡，同时使用基于鲁棒MDP的探索机制来驱动联合状态-动作的覆盖。至关重要的是，我们引入了纳什裕度刻画，使得能够对均衡存在性进行有原则的推理：该框架要么返回一个其社会福利值与最优值之差在 $\varepsilon$ 以内的 $\varepsilon$-近似纳什均衡，要么提供一个不存在精确纳什均衡的可靠证明。在相关状态-动作对满足最小可达性条件 $p_{\mathrm{reach}}>0$ 的情况下，算法在多项式数量的轨迹样本后即可终止，样本（复杂度具有…保证）

    arXiv:2609.04189v1 Announce Type: new  Abstract: We introduce the first Probably Approximately Correct (PAC) learning framework for general-sum concurrent stochastic games (CSGs) with transition uncertainty, while addressing the challenge of Nash equilibrium (NE) existence. Our algorithm maintains data-driven $L^1$ confidence sets over transition kernels and solves a robust CSG to compute a social-welfare optimal $\varepsilon$-NE, using a robust MDP-based exploration mechanism to drive joint state-action coverage. Crucially, we introduce a Nash margin characterisation that enables principled reasoning about equilibrium existence: the framework either returns an $\varepsilon$-approximate NE whose social-welfare value is $\varepsilon$-close to optimal, or provides a sound certificate that no exact NE exists. Under a minimum reachability condition $p_{\mathrm{reach}}>0$ over relevant state-action pairs, the algorithm terminates after a polynomial number of trajectory samples, with sample 
    
[^402]: 原子体系热力学景观的生成式嵌套采样

    Generative Nested Sampling of Atomistic Thermodynamic Landscapes

    [https://arxiv.org/abs/2609.03193](https://arxiv.org/abs/2609.03193)

    本文提出NS-Flows，利用单一条件归一化流替代马尔可夫链更新来加速原子体系热力学景观的嵌套采样，并通过对比揭示原子多模态（离散、组合性、硬碰撞壁分隔）与引力波后验（平滑简并、局域耦合）在结构上的根本差异。

    

    嵌套采样（NS）能够从单次模拟中解析原子系统的热力学，但其实际应用范围受限于马尔可夫链更新——在每个似然约束的系综内，需要对游走子进行去相关处理。基于流的嵌套采样方法已经在引力波（GW）推断中消除了这一瓶颈，但将其迁移到原子系统并非仅仅是应用领域的简单转换。通过将类GW150914的双黑洞似然函数与维度相当的八粒子二维Lennard-Jones（LJ）系统进行对比，我们证明这两种景观存在根本性的差异：原子体系的多模态是离散且组合性的，由被硬碰撞壁分隔的粒子置换所产生，其坐标耦合是稠密且集体性的；而GW后验则表现出平滑的简并性和局域化的参数耦合。基于这一诊断，我们提出了NS-Flows：一个单一的条件归一化流（摘要文本在此处被截断）

    arXiv:2609.03193v1 Announce Type: cross  Abstract: Nested sampling (NS) resolves the thermodynamics of an atomistic system from a single simulation, but its practical reach is limited by the Markov-chain updates needed to decorrelate walkers within each likelihood-constrained ensemble. Flow-based NS has removed this bottleneck for gravitational-wave (GW) inference, yet its transfer to atomistic systems is not merely a change of application. Comparing a GW150914-like binary-black-hole likelihood with an eight-particle two-dimensional Lennard-Jones (LJ) system of comparable dimensionality, we show that the two landscapes differ fundamentally: atomistic multimodality is discrete and combinatorial, generated by particle permutations separated by hard collision walls, and its coordinate coupling is dense and collective, whereas the GW posterior exhibits smooth degeneracies and localized parameter coupling. Guided by this diagnosis, we introduce NS-Flows: a single conditional normalizing flo
    
[^403]: 语言不可读性对大语言模型安全的影响

    The Implications of Linguistic Illegibility for LLM Security

    [https://arxiv.org/abs/2609.02852](https://arxiv.org/abs/2609.02852)

    本文提出“语言不可读性”概念，指出大语言模型的外部语言输出无法可靠反映其基于激活空间数学运算的内部计算，从而对依赖模型语言自我报告的安全机制构成根本性挑战。

    

    大语言模型（LLM）被训练用于生成自然语言。然而，多方面的证据表明，LLM外化的语言输出和通过机制可解释性方法提取的语言特征，可能并不是理解模型内部计算的可靠透镜。我们提出“语言不可读性”这一术语，广义上指LLM外化的或通过机制性探测获得的语言产物无法代表模型实际思考方式的各种情形。我们认为，对于内部计算并非通过语言直接表达、而是通过对激活空间进行数学运算来实现的大语言模型而言（激活空间与自然语言之间仅在两端发生有损转换），语言不可读性的阴影是不可避免的。如果语言不可读性始终可能存在，那么依赖模型语言自我报告的安全机制（例如思维链监控、宪法式自我批评、激活探测……）

    arXiv:2609.02852v1 Announce Type: new  Abstract: LLMs are trained to generate natural language. However, various strands of evidence indicate that an LLM's externalized linguistic outputs and mechanistically-extracted linguistic features can be an unreliable lens for understanding internal model computation. We introduce the term ``linguistic illegibility'' to broadly refer to scenarios in which an LLM's externalized or mechanistically-probed language artifacts fail to represent how the model actually thinks. We argue that the specter of linguistic illegibility is unavoidable for LLMs whose internal computations are not directly expressed via language, but rather math over activation spaces (with lossy translations between activation spaces and natural language happening at the bookends). If linguistic illegibility is always possible, then security mechanisms that rely on a model's linguistic self-reporting (e.g., chain-of-thought monitoring, constitutional self-critique, activation pr
    
[^404]: 基于路段流量传播引导的强化学习在线动态起讫点矩阵估计

    Online Estimation of Dynamic Origin-Destination Matrices Using Reinforcement Learning with Link-Flow Propagation Guidance

    [https://arxiv.org/abs/2608.30317](https://arxiv.org/abs/2608.30317)

    提出了LFPG-RL方法，将路段流量传播引导融入强化学习，用于在线动态OD矩阵估计，解决了传统标量反馈在不同目标流量轨迹下模糊不清的问题。

    

    在线动态起讫点（OD）矩阵估计（DODE）通过校准时变的OD需求来复现观测到的路段流量轨迹。在在线场景下，OD需求需要根据当前观测和传播的网络状态进行估计，而后续观测和随机动态网络加载（DNL）的结果仍然是不确定的。近年来，强化学习（RL）作为一种有前景的替代方案兴起，通过取代迭代算法来降低计算负担，同时适用于随机环境。然而，由于策略是离线训练并在线部署的，它必须应对各种不同的目标路段流量轨迹；由于每条目标轨迹定义了奖励中使用的路段流量误差，同一个OD需求向量可能需要不同的调整，使得传统的标量反馈变得模糊不清。为解决这一问题，本研究提出了LFPG-RL方法，该方法集成了路段流量传播引导（LFPG）

    arXiv:2608.30317v1 Announce Type: cross  Abstract: Online dynamic origin-destination (OD) matrix estimation (DODE) calibrates time-dependent OD demand to reproduce observed link-flow trajectories. In online, OD demand should be estimated from current observations and propagated network states while subsequent observations and stochastic dynamic network loading (DNL) outcomes remain uncertain. Recently, reinforcement learning (RL) has emerged as a promising alternative, reducing computational burden by replacing iterative algorithms while being applicable to stochastic environments. However, because the policy is trained offline and deployed online, it must handle varying target link-flow trajectories; since each target trajectory defines the link-flow error used in the reward, the same OD demand vector can require different adjustments, making conventional scalar feedback ambiguous. To address this gap, this study proposes LFPG-RL, which integrates link-flow propagation guidance (LFPG)
    
[^405]: EDGE：基于图结构化DSL配置通过对话模拟实现确定性图评估的引擎

    EDGE: Engine for Deterministic Graph Evaluation through Conversation Simulation from Graph Structured DSL Configuration

    [https://arxiv.org/abs/2608.29971](https://arxiv.org/abs/2608.29971)

    该论文提出EDGE评估框架，通过基于DSL的有向图形式化表示与图遍历算法穷举对话路径，并重放可复现轨迹，从而系统性地评估多智能体系统的行为确定性与一致性。

    

    随着智能体系统演变为复杂的多智能体编排工作流，对于能够衡量智能体行为一致性与确定性的系统性框架，存在着日益增长且至关重要的需求。在本文中，我们提出了一种正式的评估方法，该方法基于AgentGraph——一个由领域特定语言（DSL）驱动的规划器，该语言通过动态可调的有向图来表示智能体的推理过程。我们利用这种结构形式化方法，并采用图遍历算法来穷举枚举对话路径，从而形成一个全面的评估集，以捕捉智能体的完整行为空间。随后，我们系统地重放这些可复现的轨迹，将观察到的输出和状态转换与预期的DSL规范进行对比。为了量化可靠性，我们定义了新颖的指标，用于衡量响应与轨迹的确定性、结构遵从性以及语义一致性……

    arXiv:2608.29971v1 Announce Type: new  Abstract: As agentic systems evolve into complex multi agent orchestration workflows, there is a growing and critical need for systematic frameworks that measures an agent's behavioral consistency and determinism. In this paper, we introduce a formal evaluation methodology that is grounded in AgentGraph, a planner powered by a domain specific language that represents agent reasoning through a dynamically adjustable directed graph. We leverage this structural formalism and utilize graph traversal algorithms that exhaustively enumerate conversational paths, forming a comprehensive evaluation set that captures the agent's complete behavioral space. We then systematically replay these reproducible trajectories to compare observed outputs and state transitions against the intended DSL specification. To quantify reliability, we define novel metrics that measure response and trajectory determinism, structural adherence and semantic consistency across bot
    
[^406]: 面向前线部署的自主云MLOps全栈工程

    Forward-Deployed Full-Stack Engineering for Autonomous Cloud MLOps

    [https://arxiv.org/abs/2608.29615](https://arxiv.org/abs/2608.29615)

    本文提出一个以证据为门控的多智能体框架，通过结合图工程、循环工程和智能体框架工程，将自然语言的MLOps云工程任务自动转化为经过验证的代码仓库和可运行的云端部署。

    

    跨越各行各业，机器学习系统支持着从预测、异常检测到预测分析、优化和调度等广泛的应用；然而，将这些系统投入实际运营需要协调应用开发、模型流水线、云基础设施、安全、部署、监控、再训练、恢复和回滚等多个环节。我们提出了一个以证据为门控的多智能体框架，能够将自然语言描述的MLOps云工程任务转化为经过验证的代码仓库和可运行的云端部署。该框架结合了图工程、循环工程和智能体框架工程。一个有状态的图编排器协调负责代码仓库生成、审查、执行、验证、发布和监控的专用智能体，同时管控工作流依赖关系、证据门控、重试上限、恢复路径和终止条件。关键的生命周期转换只有在其所需的前提条件得到满足后才会继续执行……

    arXiv:2608.29615v1 Announce Type: cross  Abstract: Across industries, machine-learning systems support applications ranging from prediction and anomaly detection to forecasting, optimization, and scheduling, yet operationalizing these systems requires coordinating application development, model pipelines, cloud infrastructure, security, deployment, monitoring, retraining, recovery, and rollback. We present an evidence-gated multi-agent framework for transforming a natural-language MLOps cloud engineering task into a verified repository and operational cloud deployment. The framework combines graph engineering, loop engineering, and agent harness engineering. A stateful Graph Orchestrator coordinates specialized agents for repository generation, review, execution, verification, release, and monitoring while governing workflow dependencies, evidence gates, retry bounds, recovery paths, and termination. Consequential lifecycle transitions proceed only when their required predicates are su
    
[^407]: TACS：面向大语言模型越狱后缀优化的轨迹感知候选选择

    TACS: Trajectory-Aware Candidate Selection for LLM Jailbreak Suffix Optimization

    [https://arxiv.org/abs/2608.29564](https://arxiv.org/abs/2608.29564)

    论文揭示了基于梯度的越狱后缀优化中“仅选当前损失最低候选”的短视性，提出轨迹感知候选选择框架TACS，通过轨迹感知代理、参考策略正则化和判别器卡方校正，使候选选择在搜索后期依然有效。

    

    基于梯度的越狱后缀优化方法通常通过保留当前损失最低的候选来更新后缀。我们证明，这种看似自然的设计本质上是短视的：在当前步骤代理指标下表现更好的候选，往往无法在搜索后期产生更好的越狱结果，这揭示了一种选择阶段的奖励破解现象。这表明，候选选择（而不仅仅是候选生成）是后缀优化中一个隐藏的瓶颈。为了解决这一问题，我们提出了TACS，一个用于越狱后缀优化的轨迹感知候选选择框架。TACS不再仅根据即时损失来选择候选，而是通过轨迹感知代理来增强每一步的评估，并利用参考策略正则化和判别器估计的卡方校正来稳定选择过程，从而鼓励那些在当前步骤之后仍然有效的选择。

    arXiv:2608.29564v1 Announce Type: new  Abstract: Gradient-based jailbreak suffix optimization methods typically update the suffix by retaining the candidate with the lowest current loss. We show that this seemingly natural design is fundamentally myopic: candidates that look better under the current-step proxy often fail to produce better jailbreak outcomes later in the search, revealing a form of selection-stage reward hacking. This suggests that candidate selection, rather than candidate generation alone, is a hidden bottleneck in suffix optimization. To address this issue, we propose \OURS{}, a trajectory-aware candidate selection framework for jailbreak suffix optimization. Instead of selecting candidates solely by their immediate loss, \OURS{} augments per-step evaluation with a trajectory-aware proxy and stabilizes selection with reference-policy regularization and a discriminator-estimated chi-squared correction, encouraging choices that remain effective beyond the current step.
    
[^408]: 语言模型如何组织和结构化道德知识

    How Language Models Organize and Structure Moral Knowledge

    [https://arxiv.org/abs/2608.27402](https://arxiv.org/abs/2608.27402)

    本研究揭示了大型语言模型通过线性探针在表示空间中组织道德知识，其道德方向保持高度独立维度但共享道德特异性的正共同成分，表明模型能区分并整合不同道德基础。

    

    大型语言模型（LLMs）如何组织道德知识？模型能广泛检测道德内容，但检测只是一个低标准。我们探究它们是否更进一步，区分不同的道德基础，并在几何上组织它们之间的关系。我们在开放权重语言模型上训练了六个独立的线性探针，每个对应道德基础理论（MFT）的一个类别（关怀/伤害、公平/欺骗、自由/压迫、忠诚/背叛、权威/颠覆、神圣/堕落），并检查这些方向在表示空间中如何相互关联。我们发现这些方向既没有坍缩成单一的道德检测器，也没有相互隔离。相反，它们跨越了近最大数量的独立维度，同时共享一个正共同成分。该共享成分是整合的标志，并且相对于以相同方式构建的匹配非道德概念电池，它是道德特异的（平均成对余弦相似度为0.26对比0.013）。

    arXiv:2608.27402v1 Announce Type: cross  Abstract: How do large language models (LLMs) organize moral knowledge? Models detect moral content broadly, but detection is a low bar. We ask whether they go further, distinguishing moral foundations from one another and organizing the relationships between them geometrically.   We train six independent linear probes on open-weight language models, one per Moral Foundations Theory (MFT) category (care/harm, fair/cheat, lib/oppress, loy/betray, auth/subv, sanc/degrade), and examine how the resulting directions relate to each other in representation space. We find the directions neither collapse into a single moral detector nor isolate from one another. Rather, they span a near-maximal number of independent dimensions while sharing a positive common component. The shared component is the signature of integration, and it is moral-specific relative to a matched non-moral concept battery built identically (mean pairwise cosine 0.26 vs. 0.013).   Th
    
[^409]: 大语言模型增强的图神经网络是否隐私安全？

    Are LLM-Enhanced GNNs Privacy-Safe?

    [https://arxiv.org/abs/2608.25727](https://arxiv.org/abs/2608.25727)

    本文首次系统评估了LLM增强的GNNs在链接、标签和成员推断三种威胁下的隐私风险，并提出了一个五阶段统一框架进行实验分析。

    

    大语言模型（LLMs）近期通过为节点表示丰富语义信息，推动了图神经网络（GNNs）的发展，催生了LLM增强的GNNs，实现了显著的性能提升。然而，它们对隐私攻击的脆弱性——即攻击者从模型输出中推断敏感信息——在很大程度上仍未得到充分探索。为弥补这一空白，我们通过一个包含五个阶段的统一框架，对LLM增强的GNNs中的隐私风险进行了系统评估：（1）数据集准备，（2）受害者模型训练，（3）隐私攻击，（4）风险评估，以及（5）防御分析。具体而言，我们在覆盖多个领域的六个真实文本属性图数据集上进行了实验。我们考虑了针对链接、标签和成员推断三种基本威胁的六种代表性隐私攻击方法，并通过组合多种配置构建了42个受害者模型设置。

    arXiv:2608.25727v1 Announce Type: new  Abstract: Large language models (LLMs) have recently advanced graph neural networks (GNNs) by enriching node representations with semantic information, giving rise to LLM-enhanced GNNs that achieve substantial performance gains. However, their vulnerability to privacy attacks, in which adversaries infer sensitive information from model outputs, remains largely underexplored. To bridge this gap, we present a systematic evaluation of privacy risks in LLM-enhanced GNNs through a unified framework consisting of five stages: (1) dataset preparation, (2) victim model training, (3) privacy attack, (4) risk assessment, and (5) defense analysis. Specifically, we conduct experiments on six real-world text-attributed graph datasets covering diverse domains. We consider six representative privacy attack methods targeting three fundamental threats, namely link, label, and membership inference, and construct 42 victim model configurations by combining multiple 
    
[^410]: 小学习者：在教学控制的知识暴露下的语言模型

    LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure

    [https://arxiv.org/abs/2608.13545](https://arxiv.org/abs/2608.13545)

    本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。

    

    摘要：arXiv:2608.13545v1 公告类型：交叉 摘要：现代语言模型在异构的网络规模文本语料库上进行训练。因此，研究知识和技能的获取变得困难，因为先前接触相关内容难以刻画。为应对这一挑战，我们引入了LITTLECURRICULUM，一个精选的880亿令牌预训练语料库，专门针对美国小学教材，明确排除了五年级以上教授的概念、事实和词汇。在LITTLECURRICULUM上从头训练一个50亿参数的LLM，得到了LITTLELEARNER，一个具有足够语言能力进行开放式评估的模型，但其知识和能力边界清晰，映射到可解释的课程指南。我们发布LITTLECURRICULUM和LITTLELEARNER作为发展受限的沙盒，用于研究模型在明确训练范围内如何获取、表示和使用数据。我们通过一系列关于注入新知识的初步实验，展示了该沙盒的实用性。

    arXiv:2608.13545v1 Announce Type: cross  Abstract: Modern language models are trained on heterogeneous web-scale text corpora. Consequently, studying knowledge and skill acquisition is difficult, as prior exposure to related content is hard to characterize. To address this challenge, we introduce LITTLECURRICULUM, a curated 88B-token pretraining corpus tailored to U.S. elementary school material, explicitly excluding concepts, facts, and vocabulary taught above Grade 5. Training a 5B-parameter LLM from scratch on LITTLECURRICULUM yields LITTLELEARNER, a model with sufficient language competence for open-ended evaluation, yet with clear knowledge and capability boundaries mapped to interpretable curriculum guidelines. We release LITTLECURRICULUM and LITTLELEARNER as a developmentally restricted sandbox to study how models acquire, represent, and use data under a well-defined training scope. We illustrate the sandbox's utility in a first suite of experiments on injecting new knowledge th
    
[^411]: 双原始-对偶图变分自编码器用于噪声标签聚合

    Dual-Primal Graph VAEs for Noisy Label Aggregation

    [https://arxiv.org/abs/2608.11473](https://arxiv.org/abs/2608.11473)

    本文提出一种基于图变分自编码器的众包噪声标签聚合方法，利用对偶图上的GAT消息传递将真实标签作为潜在变量，无需分类器或伪标签，在基准测试中达到最优性能。

    

    从噪声众包标签中推断真实标签是一个重要的理论和实践问题。基于神经网络的方法为经典贝叶斯模型提供了一种替代方案，后者需要指定用于推理的生成模型族。然而，当前模型要么仍然依赖相当简单的生成模型进行推理，要么需要伪标签或合成数据来训练聚合分类器。我们提出了一种图变分自编码器架构，其中解码器和编码器分别基于众包数据集的邻接图及其对偶图使用GAT（图注意力网络）消息传递。真实标签被视为潜在变量，从而实现无监督表示学习，无需训练单独的分类器。我们展示了我们的模型在众包基准测试上达到了最先进的性能。然后，我们通过展示原始众包图可以如何被扩展来证明我们方法的通用性。

    arXiv:2608.11473v1 Announce Type: new  Abstract: Inferring the ground-truth from noisy crowdsourced labels is an important theoretical and practical problem. Neural network-based methods offer an alternative to classical Bayesian models which require specifying a family of generative models used for inference. However, current models either still rely on fairly simple generative models for inference or require pseudo-labels or synthetic data to train the aggregate classifier. We propose a graph VAE architecture in which the decoder and encoder use GAT-based message passing on the adjacency graph of a crowdsourced dataset and its dual, respectively. The ground-truth labels are treated as latent variables, enabling unsupervised representation learning without needing to train a separate classifier. We show our model achieves state of the art performance on crowdsourcing benchmarks. We then demonstrate the generality of our approach by showing how the original crowdsourcing graph can be a
    
[^412]: 解析器早已知晓：约束解码中的轻量级偏差校正

    The Parser Already Knows: Lightweight Bias Correction in Constrained Decoding

    [https://arxiv.org/abs/2608.10137](https://arxiv.org/abs/2608.10137)

    该论文提出SHIM，巧妙利用约束解码工具已维护的解析器和词法分析器状态作为信号，通过轻量级离线训练的校正模块修正语言模型的下一词元概率，在不改动模型本身的前提下消除语法约束解码带来的分布偏差。

    

    语法约束解码通过在每一步屏蔽不符合规范的词元，迫使语言模型生成句法有效的输出。然而，由于屏蔽机制仅检查每个词元到目前为止是否有效，由此产生的完整输出分布会偏离语言模型自身在语法条件下应有的分布，使生成偏向有效但次优的输出。在线采样可以恢复该分布，但只能通过代价高昂的迭代重采样来实现。我们的关键洞察是：语法约束解码工具已经维护的解析器和词法分析器状态，携带了关于未来语法有效性的强烈信号。我们提出SHIM——一种轻量级的、离线训练的校正方法，以句法与词法状态以及候选的下一词元为条件，对语言模型的下一词元概率进行校正。由于语法约束解码工具本身已经在计算这些状态，SHIM完全无需改动语言模型。在位向量与文本到SQL等语法任务上，校正后的分布……

    arXiv:2608.10137v2 Announce Type: replace  Abstract: Grammar Constrained Decoding (GCD) forces Language Models (LMs) to produce syntactically valid outputs by masking out non-conforming tokens at each step. However, because masking only checks whether each token is valid so far, the resulting distribution over complete outputs diverges from the LM's own distribution conditioned on the grammar, biasing generation toward valid but suboptimal outputs. Online sampling can restore this distribution, but only through costly iterative resampling. Our key insight is that the parser and lexer states that GCD tools already maintain carry a strong signal about future grammatical validity. We introduce SHIM, a lightweight, offline-trained correction of the LM's next-token probabilities, conditioned on this syntactic and lexical state together with candidate next tokens. Since GCD tools already compute these states, SHIM leaves the LM itself untouched. Across bit-vector and text-to-SQL grammars, th
    
[^413]: Faster-WAM：世界动作模型需要深层动作模块吗？

    Faster-WAM: Do World Action Models Need Deep Action Modules?

    [https://arxiv.org/abs/2608.02365](https://arxiv.org/abs/2608.02365)

    提出Faster-WAM，通过以世界模型为中心的设计，用浅层轻量级动作专家（配合DoT、Lite KV-Fusion、仅世界模型条件化及收缩式1D-RoPE）替代深层动作模块，将计算集中于视频世界模型，从而降低推理延迟并缓解过拟合问题。

    

    世界动作模型（World Action Models, WAMs）构建于预训练视频模型之上，这些模型的表示根植于物理动力学，为动作预测提供了天然基础。尽管具备这种天然优势，许多WAM仍然依赖深层的、参数繁多的动作预测模块，这会导致较高的推理延迟，并可能在有限的机器人演示数据上过拟合，从而限制其实际应用。在本文中，我们提倡一种以世界模型为中心的设计原则：将模型容量和计算集中于视频世界模型本身，而由一个轻量级动作专家将骨干网络的表示转化为可执行的机器人动作。我们通过三个关键选择来实现这一原则：采用Dock of Transformers（DoT）结合Lite KV-Fusion，使浅层、轻量级的动作专家能够访问所有视频层的表示；动作专家仅以世界模型为条件（world-model-only conditioning）；以及使用收缩式1D-RoPE实现视频与动作之间的位置对齐……

    arXiv:2608.02365v2 Announce Type: replace-cross  Abstract: World Action Models (WAMs) build on pretrained video models, whose representations are grounded in physical dynamics and provide a natural basis for action prediction. Despite this natural foundation, many WAMs still rely on deep, parameter-heavy action-prediction modules that incur high inference latency and may overfit to limited robot demonstrations, restricting their real-world applicability. In this paper, we advocate a world-model-centric principle that concentrates capacity and computation in the video world model, while a lightweight action expert translates the backbone's representations into executable robot actions. We realize this principle through three key choices: Dock of Transformers (DoT) with Lite KV-Fusion to give the shallow, lightweight action expert access to representations from all video layers; world-model-only conditioning of the action expert; and retracted 1D-RoPE for positional alignment between vid
    
[^414]: ReToken：利用视觉检索Token改进长上下文视觉语言模型

    ReToken: Improving Long-Context VLMs with Visual Retrieval Token

    [https://arxiv.org/abs/2607.28627](https://arxiv.org/abs/2607.28627)

    RETOKEN通过单一可学习嵌入从VLM内部表示中提取检索信号，直接在预填充的KV缓存中选择与查询相关的视觉token，无需独立检索器或重新编码即可显著提升模型在长上下文图像和视频任务上的性能。

    

    长视觉上下文对视觉语言模型（VLM）构成挑战：随着干扰项数量的增加，模型性能会下降，且一次性处理所有token可能超出GPU内存限制。我们提出了RETOKEN，这是一个单一的可学习嵌入，能够从VLM的内部表示中提取检索信号，从预填充的KV缓存中选择与查询相关的视觉token。这使得检索能够在回答问题的VLM内部完成，无需单独的检索器或重新编码。尽管仅在小型图像问答数据集上训练，RETOKEN在图像和视频基准测试中均展现出良好的泛化能力。在Visual Haystacks上，它使Qwen3VL-8B提升了13.4分，InternVL3.5提升了12.4分（相对提升超过20%）。在LVBench上，它以零样本方式迁移到长视频任务，使Qwen3VL-8B提升了8.0分。得益于其轻量化设计，训练和长视频推理均可在单张H100上完成。代码可在以下网址获取：https://github.com/avaxiao/ReToken

    arXiv:2607.28627v2 Announce Type: replace-cross  Abstract: Long visual contexts challenge vision-language models: performance degrades as the number of distractors grows, and processing all tokens at once can exceed GPU memory limits. We present RETOKEN, a single learnable embedding that extracts retrieval signals from the VLM's internal representations to select query-relevant visual tokens from the pre-filled KV cache. This enables retrieval within the answering VLM, without a separate retriever or re-encoding. Despite being trained on only a small image-QA dataset, RETOKEN generalizes across image and video benchmarks. On Visual Haystacks, it improves Qwen3VL-8B by 13.4 points and InternVL3.5 by 12.4 points (>20% relative). On LVBench, it transfers zero-shot to long video and improves Qwen3VL-8B by 8.0 points. Thanks to its lightweight design, both training and long-video inference fit on a single H100. Code is available at: https://github.com/avaxiao/ReToken
    
[^415]: 面向代码优化的强化学习

    Reinforcement Learning for Code Optimization

    [https://arxiv.org/abs/2607.25970](https://arxiv.org/abs/2607.25970)

    该论文提出三阶段方法解决基于执行时间奖励的强化学习中的噪声、稀疏与不稳定性问题——包括构建带校准沙箱的DMC-Optim基准、组合正确性与速度的奖励设计并用离线模拟器选择配置、以及适配GRPO算法——从而让大模型能够真正学会优化代码运行速度。

    

    arXiv:2607.25970v2 公告类型：替换。摘要：强化学习用于代码正确性优化如今已经成熟：让模型生成一个程序，用隐藏测试用例运行它，并奖励通过测试的解。将这一方法扩展到代码优化似乎很简单：只需把执行时间加入奖励即可。但在实践中，一旦执行时间成为奖励的主要驱动因素，测量噪声、奖励稀疏性或GRPO不稳定性等小问题就会淹没信号，导致强化学习失败：生成的解几乎没有变快，而且更多的解会失败。我们通过三个阶段使执行时间变得可学习：(1) 如何测试代码——构建DMC-Optim，包含大规模优化测试和经过校准的沙箱；(2) 如何将速度转化为奖励——在强化学习环境中组合正确性与速度，并使用离线模拟器预测最有前景的配置；(3) 模型如何从该奖励中学习——将GRPO和评估方法适配到更稀疏、更嘈杂的计时执行环境。在DMC-Optim上，（摘要原文在此处截断）

    arXiv:2607.25970v2 Announce Type: replace  Abstract: RL for code correctness is now established: have the model generate a program, run it against hidden test cases, and reward solutions that pass. Extending this to code optimization seems straightforward: just add execution time to the reward. But in practice, once timing drives the reward, small problems in measurement noise, reward sparsity, or GRPO instability overwhelm the signal and make RL fail: generated solutions are barely faster, and more of them can fail. We make execution time learnable through three stages: (1) how code is tested, by building DMC-Optim with large optimization tests and a calibrated sandbox; (2) how speed is turned into reward, by composing correctness and speed in the RL environment and using an offline simulator to predict the most promising configurations; and (3) how the model learns from that reward, by adapting GRPO and evaluation to the sparser, noisier timed-execution setting. On DMC-Optim, the str
    
[^416]: 奖励模型记住了什么？

    What do Reward Models Memorize?

    [https://arxiv.org/abs/2607.24484](https://arxiv.org/abs/2607.24484)

    本文通过反事实记忆测量发现，判别式训练的奖励模型会错误记忆简单偏好对、记住数据集特定捷径，并过度泛化长度等简单启发式特征，导致其无法在情境相关场景中准确判断回复质量。

    

    本文通过在两个人类偏好数据集上测量反事实记忆，研究了判别式训练的奖励模型（RMs）究竟记住了什么。我们发现奖励模型存在三个问题：1）将记忆错误地分配给简单的、高余量的偏好对；2）记住了数据集特定的捷径（例如模型身份、用户采样策略）；3）在面对未见过的偏好对时，过度泛化人类偏好的简单启发式相关因素（例如回复长度、顺从性）。总体而言，我们的研究结果表明，从人类偏好数据中通过判别式方式训练奖励模型，会导致带有偏见的奖励模型，其尚不具备在情境相关场景中准确判断回复质量的能力。

    arXiv:2607.24484v2 Announce Type: replace-cross  Abstract: This paper studies what discriminatively trained reward models (RMs) memorize by measuring counterfactual memorization on two human preference datasets. We show that RMs 1) misallocate memorization to easy, high margin preference pairs, 2) memorize dataset-specific shortcuts (e.g., model identity, user sampling strategy), and 3) overgeneralize simple heuristic correlates of human preference (e.g., length, compliance) when confronted with unseen preference pairs. Overall, our findings indicate that discriminative training of RMs from human preference data results in biased RMs not yet capable of judging response quality in context-dependent scenarios.
    
[^417]: 用于参数高效微调的流形约束超连接

    Manifold-Constrained Hyper-Connections for Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2607.18130](https://arxiv.org/abs/2607.18130)

    该论文将流形约束超连接（mHC）引入冻结主干微调，通过保留流混合、仅学习子层访问流的方式，以更少的可训练参数改善模型损失，确立了残差路由作为基础模型高效微调的新架构维度。

    

    基础模型的微调方法通常改变权重、提示或隐藏状态，而保持残差拓扑结构固定。我们提出这样一个问题：残差拓扑结构本身是否可以成为微调的对象。为了研究这一点，我们将近期为预训练提出的流形约束超连接适配到冻结主干的微调场景中。mHC 将 Transformer 转化为一种依赖输入的多流残差架构，在每个子层通过多个流对表示进行路由。在各种 mHC 变体上，我们发现动态残差路由能够对 Transformer 进行微调，但其作用与预训练时不同：通过保留流混合并仅学习子层如何访问流，模型的损失得到改善，同时可训练参数得以减少。总体而言，我们的研究结果确立了残差路由作为基础模型高效微调的一个有前景的架构方向。

    arXiv:2607.18130v2 Announce Type: replace  Abstract: Finetuning methods for foundation models usually change weights, prompts, or hidden states, while leaving the residual topology fixed. We ask whether residual topology itself can become a finetuning object. To study this, we adapt manifold-constrained hyper-connections (mHC), recently introduced for pre-training, to frozen-backbone finetuning. mHC turns a Transformer into an input-dependent multi-stream residual architecture, routing representations through multiple streams at every sub-layer. Across mHC variants, we find that dynamic residual routing can finetune Transformers, but that its role differs from pre-training: by preserving stream mixing and learning only how sub-layers access streams, loss is improved and trainable parameters are reduced. Overall, our results identify residual routing as a promising architectural axis for efficient finetuning of foundation models.
    
[^418]: 基于最优传输的线性独立成分分析

    Linear Independent Component Analysis via Optimal Transport

    [https://arxiv.org/abs/2607.14081](https://arxiv.org/abs/2607.14081)

    本文提出以数据线性投影到标准高斯分布的平方 Wasserstein 距离作为 ICA 对比函数，并证明该距离在投影恰好恢复出独立成分时达到最大、且与任何真实混合信号之间都存在显式间隔，从而为线性独立成分分析建立了一种基于最优传输的新方法。

    

    线性独立成分分析（ICA）旨在从源信号的线性混合中恢复出联合独立的源信号。为实现这一目标，经典的ICA算法试图最大化非高斯性，通常以负熵来度量，而信息论将负熵与独立性联系在一起。由于精确的负熵优化是难以处理的，这些算法依赖于代理对比函数，例如四阶累积量和参数化对数似然。我们转而提出使用到标准高斯分布的平方 $L_2$-Wasserstein 距离作为 ICA 的对比函数。我们证明了当线性投影恢复出某个独立成分时，标准正态分布与数据线性投影之间的 Wasserstein 距离达到最大，并且在源信号满足一定正则性条件下，该最大值与每一个真实混合信号之间都由一个显式的间隔分隔开来。我们揭示了由此得到的估计量的优越性质：对于具有光滑密度的源信号……（原文摘要在此处被截断）

    arXiv:2607.14081v2 Announce Type: replace  Abstract: Linear Independent Component Analysis (ICA) recovers jointly independent source signals from their linear mixtures. To achieve this, classical ICA algorithms attempt to maximize non-Gaussianity, measured by negentropy, which is linked to independence by information theory. Because exact negentropy optimization is intractable, they rely on proxy contrast functions, such as fourth-order cumulants and parametric log-likelihoods. We propose instead to use the squared $L_2$-Wasserstein distance to a standard Gaussian as the ICA contrast. We show that the Wasserstein distance between a standard normal distribution and linear projections of the data is maximized when the projection recovers an independent component, and that under a regularity condition on the sources this maximum is separated from every genuine mixture by an explicit margin. We uncover the advantageous properties of the resulting estimator: for sources with a smooth densit
    
[^419]: 将通用车辆模型适配用于跨地形高速模型预测控制（MPC）

    Adapting Generalist Vehicle Models for High-Speed MPC Across Terrains

    [https://arxiv.org/abs/2607.13319](https://arxiv.org/abs/2607.13319)

    OptCar提出了一种基于FiLM条件化transformer的FKD架构，仅需有限真实世界数据即可将通用车辆动力学基础模型专业化为特定车辆的专用模型，在保持跨地形泛化能力的同时实现跨地形高速MPC精确控制。

    

    高速越野自动驾驶需要对目标车辆进行精确的闭环控制，同时在多变地形上保持鲁棒性。近期的前向运动动力学（FKD）预测基础模型展现出一条有前景的路径：从通用模型出发，将其专用于目标平台。然而，有效的专业化仍然具有挑战性，因为它通常需要大量真实世界数据，而且适配到某一环境的模型仍可能对特定地形或驾驶模式过拟合。我们提出了OptCar（优化汽车），这是一种弥合通用FKD模型到专用FKD模型差距的方法，在为特定车辆优化性能的同时保留了跨地形泛化能力。OptCar引入了一种基于transformer的FKD架构，利用FiLM将多步预测条件化于单个动力学上下文token，该token概括了近期的状态-动作历史。随后，该方法利用有限的真实世界数据对通用模型进行专业化（摘要在此处截断）。

    arXiv:2607.13319v2 Announce Type: replace-cross  Abstract: High-speed off-road autonomy requires precise closed-loop control for a target vehicle while remaining robust across changing terrains. Recent forward kinodynamic (FKD) prediction foundation models suggest a promising path, starting from a generalist model and specializing it to the target platform. However, effective specialization remains challenging, as it often requires substantial real-world data, and models adapted to one setting can still overfit to specific terrains or driving regimes. We present OptCar (Optimized Car), a recipe for bridging the gap from generalist to specialist FKD models that preserves cross-terrain generalization while optimizing performance for a specific vehicle. OptCar introduces a transformer FKD architecture that uses FiLM to condition multi-step predictions on a single dynamics context token summarizing recent state-action history. It then specializes the generalist model using limited real-wor
    
[^420]: RUBRIC：面向不平衡分类的现实性与效用平衡排序方法

    RUBRIC: Realism--Utility Balanced Ranking for Imbalanced Classification

    [https://arxiv.org/abs/2607.09816](https://arxiv.org/abs/2607.09816)

    RUBRIC是一个与生成器无关的过滤框架，通过现实性与效用的平衡排序，将合成样本选择优化为质量优先问题，从而减少低质量候选样本对决策边界的干扰，提升不平衡分类的泛化性能。

    

    类别不平衡在欺诈检测和医学诊断等风险敏感应用中构成了根本性挑战，在这些场景中，少数类样本稀缺但对准确分类至关重要。现有的过采样方法生成合成样本以重新平衡类别分布；然而，它们往往产生大量低质量候选样本，这些样本会扭曲决策边界或引入伪影，导致过拟合和泛化性能下降。在本工作中，我们引入了RUBRIC，一个与生成器无关的过滤框架，将合成样本选择形式化为一个质量优先于数量的优化问题。RUBRIC使用现实性-效用权衡对候选样本进行排序：现实性通过一个学习到的判别器来量化，该判别器区分真实样本和合成样本，而效用通过一个基于凹边距的评分函数捕捉到决策边界的接近程度。我们证明，在温和的正则条件下，该框架能够有效提升分类性能。

    arXiv:2607.09816v3 Announce Type: replace  Abstract: Class imbalance poses a fundamental challenge in risk-sensitive applications such as fraud detection and medical diagnosis, where minority-class samples are scarce yet critical for accurate classification. Existing oversampling methods generate synthetic samples to rebalance class distributions; however, they often produce large numbers of low-quality candidates that distort decision boundaries or introduce artifacts, leading to overfitting and degraded generalization.   In this work, we introduce RUBRIC, a generator-agnostic filtering framework that formulates synthetic sample selection as a quality-over-quantity optimization problem. RUBRIC ranks candidates using a realism-utility trade-off: realism is quantified by a learned discriminator that distinguishes real samples from synthetic samples, while utility captures proximity to the decision boundary through a concave margin-based scoring function. We show that, under mild regular
    
[^421]: 幻觉自博弈：通过演化生成器自举强化检测器

    Hallucination Self-Play: Bootstrapping Reinforced Detector via Evolved Generator

    [https://arxiv.org/abs/2607.07993](https://arxiv.org/abs/2607.07993)

    提出幻觉自博弈（HSP）框架，让检测器与演化中的生成器以对抗方式协同演化——利用RLAIF训练生成器产生越来越难检测的幻觉，从而不断自举提升幻觉检测器的性能。

    

    由于高质量标注数据的稀缺，识别大语言模型生成输出中的忠实性幻觉仍然具有挑战性。近期的工作依赖先进的LLM来合成训练数据，包括推理依据、标签和幻觉性声明。然而，这些方法将生成器视为静态组件，限制了检测器的迭代改进。为了解决这一局限，我们提出了幻觉自博弈，这是一种新颖的框架，使检测器能够与演化中的生成器协同自举提升。HSP包含从同一基础模型初始化的两个角色：一个评估模型输出忠实性的检测器，以及一个生成越来越难以检测的幻觉响应的生成器。具体而言，检测器首先在人工标注数据上进行微调，然后作为奖励模型，通过AI反馈强化学习（RLAIF）来训练生成器。反过来，演化后的生成器合成……（摘要内容不完整，此处截断）

    arXiv:2607.07993v2 Announce Type: replace  Abstract: Identifying faithfulness hallucinations in LLM-generated outputs remains challenging due to the scarcity of high-quality annotated data. Recent work relies on advanced LLMs to synthesize training data, including rationales, labels, and hallucinated claims. However, these methods treat the generator as a static component, limiting iterative improvement of the detector. To address this limitation, we introduce Hallucination Self-Play (HSP), a novel framework that enables the detector to bootstrap with an evolved generator. HSP involves two roles initialized from the same base model, a detector that assesses the faithfulness of model outputs, and a generator that produces increasingly hard-to-detect hallucinated responses. Specifically, the detector is first fine-tuned on human-labeled data and then employed as a reward model to train the generator via reinforcement learning from AI feedback (RLAIF). In turn, the evolved generator synth
    
[^422]: ECGLight：面向纸质心电图数字化与心肌梗死筛查的低计算量框架

    ECGLight: Compute-Light Framework For Paper ECG Digitization and Myocardial Infarction Screening

    [https://arxiv.org/abs/2607.07683](https://arxiv.org/abs/2607.07683)

    ECGLight是一个低计算量的设备端框架，能够高保真地将纸质心电图数字化并同时支持心肌梗死筛查等多项临床任务，适用于网络和计算资源受限的偏远地区诊所。

    

    心电图（ECG）是诊断心血管疾病最广泛使用的检查之一。然而，由于网络连接和计算能力的限制，一些偏远诊所仍然依赖纸质心电图打印件进行分析。因此，偏远地区获取的大量纸质心电图仍然无法被当代基于人工智能（AI）的决策支持系统所利用，因为这些系统需要大量计算资源或高速互联网连接。这导致急性冠脉闭塞（ACS）等病症被忽视、再灌注治疗被延误的情况时有发生。尽管先前的研究已经分别解决了数字化和诊断问题，并使用了先进的AI模型，但仍然缺乏一个计算量低、可在设备端运行的框架，既能高保真地重建纸质心电图，又能准确支持多个临床相关终点。我们针对这一需求（摘要在此处截断）……

    arXiv:2607.07683v2 Announce Type: replace  Abstract: Electrocardiography (ECG) is one of the most widely used tests for diagnosing cardiovascular disease. Yet several remote clinics still utilize paper ECG printouts for their analysis due to limited connectivity and computational capacity. As a result, vast numbers of physical ECGs obtained in remote areas still remain incapable of being accessed by contemporary artificial-intelligence (AI)-based decision support as they require high computational resources or strong high-speed internet connectivity. This causes several cases where conditions like acute coronary occlusion (ACS) is overlooked and reperfusion therapy delayed. Although prior work has tackled digitization and diagnosis separately, and utilized advanced AI models for them, there still remains a lack of a compute-light, on-device framework that reconstructs paper ECGs at high fidelity, while accurately supporting multiple clinically relevant endpoints. We address this need w
    
[^423]: 面向物理信息神经网络反问题的目标引导选择性重加权：一种迁移学习方法

    Target-Guided Selective Reweighting for Physics-Informed Neural Network Inverse Problems: A Transfer Learning Approach

    [https://arxiv.org/abs/2607.05271](https://arxiv.org/abs/2607.05271)

    提出TGSR-PINN方法，将迁移学习与基于目标证据的神经元敏感度评分和选择性重加权相结合，解决了物理信息神经网络在偏微分方程反问题中因负迁移导致的物理参数恢复不准确问题。

    

    物理信息神经网络（PINNs）在偏微分方程（PDE）反问题中常常面临不适定优化、损失函数相互竞争以及参数补偿等问题。迁移学习可以复用源任务的特征表示，但当源任务与目标任务的物理特性不同时，直接微调可能引发负迁移，导致场误差较低但参数恢复不准确。为解决这一问题，我们提出了目标引导选择性重加权PINN（TGSR-PINN），这是一种基于目标证据驱动的PINN反问题迁移学习表示修正方法。TGSR-PINN迁移源网络的权重和偏置，但独立初始化目标物理参数。经过短时间的目标适应后，该方法在固定批次上利用一阶泰勒敏感度和预激活方差对神经元进行评分。这些评分通过带秩回退机制的高斯混合模型转换为连续的弱适应信号。TGSR-PINN然后……（注：原文摘要在此处截断）

    arXiv:2607.05271v2 Announce Type: replace  Abstract: Physics-informed neural networks (PINNs) often face ill-posed optimization, competing losses, and parameter compensation in partial differential equation (PDE) inverse problems. Transfer learning can reuse source-task representations, but direct fine-tuning may induce negative transfer when source and target physics differ, leading to low field error but inaccurate parameter recovery. To address this issue, we propose Target-Guided Selective Reweighting PINN (TGSR-PINN), a target-evidence-driven representation correction method for PINN inverse transfer learning. TGSR-PINN transfers source network weights and biases but initializes target physical parameters independently. After short target adaptation, it scores neurons using first-order Taylor sensitivity and pre-activation variance on fixed batches. These scores are converted into continuous weak-adaptation signals using a Gaussian mixture model with rank fallback. TGSR-PINN then 
    
[^424]: 基于分数的生成模型中随机梯度下降的非渐近收敛性

    Non-asymptotic Convergence of Stochastic Gradient Descent in Score-based Generative Models

    [https://arxiv.org/abs/2607.04775](https://arxiv.org/abs/2607.04775)

    本文研究了基于分数的生成模型训练中随机梯度下降的非渐近收敛保证，针对一般分数参数化给出了显式依赖损失加权和时间采样分布的非凸优化界，并为过参数化两层 ReLU 网络建立了神经正切核分析。

    

    基于分数的生成模型在广泛的应用领域中取得了令人瞩目的数据生成性能。尽管其采样过程的统计特性已日益被充分理解，但其训练背后的优化动力学仍未得到充分探索。SGM 通常通过最小化加权去噪分数匹配目标进行训练，然而基于随机梯度的优化保证仍然有限。在本工作中，我们研究了随机梯度下降（SGD）在 SGM 中的应用，并在两个互补的场景中做出了贡献。对于一般的分数参数化，我们针对加权去噪分数匹配目标推导了 SGD 的非凸分析，明确揭示了所得优化界如何依赖于损失加权和时间采样分布。随后，我们考虑过参数化的两层 ReLU 网络，并开发了一种针对扩散模型的神经正切核分析……

    arXiv:2607.04775v2 Announce Type: replace-cross  Abstract: Score-based Generative Models (SGMs) have achieved impressive performance in data generation across a wide range of applications. While the statistical properties of their sampling procedures are increasingly well understood, the optimization dynamics underlying their training remain less explored. SGMs are typically trained by minimizing a weighted denoising score-matching objective, yet optimization guarantees with stochastic gradients remain limited. In this work, we study Stochastic Gradient Descent (SGD) for SGMs, contributing results in two complementary regimes. For general score parameterizations, we derive a non-convex analysis of SGD for the weighted denoising score-matching objective, making explicit how the resulting optimization bound depends on the loss weighting and time-sampling distribution. We then consider overparameterized two-layer ReLU networks and develop a Neural Tangent Kernel analysis tailored to diffu
    
[^425]: 基于Armijo回溯的方向曲率：一种低成本的尖锐度探针及Adam的无标定学习率保护机制

    Directional Curvature from Armijo Backtracking: A Low-Cost Sharpness Probe and a Calibration-Free Learning-Rate Safeguard for Adam

    [https://arxiv.org/abs/2607.03998](https://arxiv.org/abs/2607.03998)

    本文提出利用Armijo回溯线搜索的接受步长来低成本估计损失函数的局部尖锐度（Hessian最大特征值），并将其作为Adam优化器的无标定学习率保护机制，有效防止初始学习率过大。

    

    arXiv:2607.03998v4 公告类型：替换 摘要：损失的局部尖锐度，即Hessian矩阵的最大特征值λ1，决定了最大的稳定梯度步长，但测量它通常需要Lanczos方法或Hessian-向量乘积。一次Armijo回溯线搜索已经以几次前向传播的成本携带了此信息：接受的步长α在由回溯因子设定的乘法带内括住了沿探测方向的方向曲率：这正是测试步长上平均的曲率，并且经验上q = g^T H g/||g||^2在该带内。在CIFAR-10、Fashion-MNIST和Imagenette上，log α与log λ1的Pearson相关系数为-0.91至-0.95，并且该关系在每次运行的去趋势检验中存活，相关系数为-0.60至-0.70，这是一种对慢尖锐度成分的低成本在线边缘稳定性读数。作为保护措施而非更快的优化器，该读数限制过大的初始学习率。单次

    arXiv:2607.03998v4 Announce Type: replace  Abstract: The local sharpness of the loss, the top Hessian eigenvalue $\lambda_1$, determines the largest stable gradient step, but measuring it normally requires Lanczos or Hessian-vector products. A single Armijo backtracking line search already carries this information at the cost of a few forward passes: the accepted step $\alpha$ brackets the directional curvature along the probed direction within the multiplicative band set by the backtracking factor: exactly the curvature averaged over the tested step, and empirically $q = g^\top H g/\|g\|^2$ to within that band. Across CIFAR-10, Fashion-MNIST and Imagenette, $\log\alpha$ tracks $\log\lambda_1$ at Pearson $-0.91$ to $-0.95$, and the relation survives a per-run detrending check at $-0.60$ to $-0.70$, a low-cost online Edge-of-Stability reading of the slow sharpness component. Used as a safeguard rather than a faster optimiser, the reading caps a too-large initial learning rate. A single 
    
[^426]: 评估用于电力价格预测的时间序列基础模型：污染风险、分布偏移与协变量依赖

    Evaluating Time Series Foundation Models for Electricity Price Forecasting: Contamination Risk, Distributional Shifts, and Covariate Dependence

    [https://arxiv.org/abs/2607.02623](https://arxiv.org/abs/2607.02623)

    本文提出一个双数据集基准测试框架来评估时间序列基础模型在电价预测中的表现，发现其虽具竞争力且常优于通用基线，但性能高度依赖协变量支持且未必超越领域专用方法，而将两者简单集成可获得更优效果。

    

    时间序列基础模型（TSFMs）在零样本预测方面已展现出强大的性能，但其在协变量驱动、非平稳环境下的泛化能力尚未得到充分探索。电力价格预测（EPF）由于具有复杂的时间依赖性、分布偏移以及对结构和上下文信息的强烈依赖，构成了一个具有挑战性的测试平台。我们提出了一个基于双数据集的EPF基准测试框架，以降低数据污染风险并实现对TSFMs的公平评估。我们研究了EPF的关键方面，包括点预测和概率预测性能、尾部行为、价格飙升，以及与特定领域方法的比较。我们发现TSFMs具有很强的竞争力，并且往往优于通用基线方法。然而，其性能在很大程度上依赖于协变量的支持，且并不总是超越为EPF量身定制的特定领域方法。有趣的是，TSFMs与特定领域方法的简单集成……（原文摘要在此处截断）

    arXiv:2607.02623v2 Announce Type: replace  Abstract: Time series foundation models (TSFMs) have shown strong zero-shot forecasting performance, but their generalization in covariate-driven, non-stationary settings is underexplored. Electricity price forecasting (EPF) presents a challenging testbed due to complex temporal dependencies, distributional shifts, and strong reliance on structural and contextual information. We propose a two-dataset-benchmarking framework for EPF to mitigate contamination risk and enable fair evaluation of TSFMs. We examine key aspects of EPF including point and probabilistic forecasting performance, tail behavior, price spikes, and comparisons against domain-specific methods. We find that TSFMs are highly competitive and often outperform general-purpose baselines. Yet, their performance depends critically on covariate support, and they do not consistently surpass domain-specific methods tailored to EPF. Interestingly, simple ensembles of TSFMs and domain-spe
    
[^427]: FAR：面向测试时恢复与持续策略改进的故障感知重试

    FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement

    [https://arxiv.org/abs/2607.01111](https://arxiv.org/abs/2607.01111)

    提出故障感知重试框架FAR，通过故障对比偏好适应与轻量动作扰动让机器人在测试时从失败中学习并自主恢复任务，同时将成功恢复轨迹纳入训练实现持续策略改进，成功率平均提升17.6%。

    

    机器人策略在真实环境中部署时不可避免地会遇到失败。简单的重试往往重复同样的错误，而许多现有的恢复方法则依赖于人工干预。本文提出了故障感知重试，这是一个使机器人能够在测试时从之前的失败中学习、相应地调整自身行为、并最终自主完成任务的框架。FAR结合了两项技术：故障对比偏好适应，即从失败中构建偏好学习数据，引导策略远离之前不成功的行为；以及在重试期间施加轻量级的动作扰动，以鼓励局部探索。我们进一步将成功的恢复轨迹纳入训练循环，以实现持续的策略改进。在仿真和真实世界操作任务上的实验表明，FAR显著提升了成功率和鲁棒性，平均提升达17.6%以上。

    arXiv:2607.01111v2 Announce Type: replace-cross  Abstract: Robot policies inevitably encounter failures when deployed in real environments. Naive retries often repeat the same mistakes, while many existing recovery methods rely on human intervention. In this paper, we propose Failure-Aware Retry (FAR), a framework that enables robots to learn from previous failures at test time, adapt their behavior accordingly, and eventually complete the task autonomously. FAR combines Failure-Contrastive Preference Adaptation, which constructs preference learning data from failures to steer the policy away from previously unsuccessful behaviors, with lightweight action perturbations during retries to encourage local exploration. We further incorporate successful recovery trajectories into a training loop for continual policy improvement. Experiments in both simulation and real-world manipulation tasks show that FAR substantially improves success rates and robustness, with average gains of 17.6% over
    
[^428]: 面向图上能源时间序列不确定性量化的上下文残差校准方法

    In-Context Residual Calibration for Uncertainty Quantification of Energy Time Series over Graphs

    [https://arxiv.org/abs/2606.31804](https://arxiv.org/abs/2606.31804)

    该论文提出一种上下文残差校准方法，针对现有共形预测难以捕捉能源系统复杂时空结构的缺陷，为图上能源时间序列提供更可靠的不确定性量化，以支持风险感知的能源运营决策。

    

    精确的能源需求预测对于现代可持续能源系统的可靠运行和规划至关重要。时空图神经网络（STGNN）通过联合建模时间动态特性和互联能源节点之间的关系依赖性，近期在点预测方面取得了优异的表现。然而，在现实世界的能源系统中，仅有精确的点预测是不够的，运营者还需要可靠的不确定性估计，以支持风险感知决策、电网稳定性以及不确定性条件下的运营规划。共形预测在可交换性假设下为不确定性量化提供了一个有原则且与模型无关的框架，这使其对于安全关键的能源应用尤其具有吸引力。然而，现有的共形预测方法往往无法充分捕捉能源系统复杂的时空结构。为了解决这些问题……

    arXiv:2606.31804v2 Announce Type: replace  Abstract: Accurate energy demand forecasting is essential for the reliable operation and planning of modern sustainable energy systems. Spatial-temporal graph neural networks (STGNNs) have recently achieved strong performance in point forecasting by jointly modeling temporal dynamics and relational dependencies across interconnected energy nodes. However, in real-world energy systems, accurate point forecasts alone are insufficient, as operators also require reliable uncertainty estimates to support risk-aware decision-making, grid stability, and operational planning under uncertainty. Conformal prediction provides a principled and model-agnostic framework for uncertainty quantification under exchangeability assumptions, making it particularly attractive for safety-critical energy applications. However, existing conformal prediction approaches often fail to fully capture the complex spatial-temporal structure of energy systems. To address thes
    
[^429]: 病态反问题中的地质文本描述：来自学习型渗透系数反演的见解

    Geological text descriptions in ill-posed inverse problems: insights from learned hydraulic-conductivity inversion

    [https://arxiv.org/abs/2606.24967](https://arxiv.org/abs/2606.24967)

    该研究通过合成达西流基准系统考察了地质文本描述在学习型渗透系数反演中的作用，发现当观测存在结构歧义时描述能改善重建，实例特定内容尤为有效，且求解器会依据描述所述的几何信息偏移重建结果，但随着观测信息量增大，描述的影响会减弱。

    

    渗透系数反演是一个不适定（病态）问题：即使是完整的水头观测也可能留下结构性歧义。地质文本描述可以提供关于地下结构的额外信息来约束重建。然而，描述何时能改善重建、以及求解器如何使用其所述内容，目前仍不清楚。我们在学习型反演中，通过一个带有理想化描述的合成达西流基准来研究这些问题，并改变观测密度和流动方向。经过稀疏观测训练后，当观测留下结构性歧义时，与未使用描述训练的模型相比，描述可以改善重建效果，尽管增益在不同条件和不同运行之间有所差异。实例特定的内容可以在场类型信息之外进一步改善重建。编辑这些内容会使重建结果向所选特征的所述几何形态偏移，但随着观测对这些特征的信息量增加，这种偏移会减弱。

    arXiv:2606.24967v2 Announce Type: replace  Abstract: Hydraulic-conductivity inversion is ill posed: even complete head observations can leave structural ambiguity. Geological text descriptions can supply additional information about subsurface structure to constrain reconstruction. However, it remains unclear when descriptions improve reconstruction and how solvers use their stated content. We examine these questions in learned inversion using a synthetic Darcy-flow benchmark with idealised descriptions, varying observation density and flow direction. After sparse-observation training, descriptions can improve reconstruction over models trained without them when observations leave structural ambiguity, though gains vary across conditions and runs. Instance-specific content can improve reconstruction beyond field-type information. Editing this content shifts reconstructions toward the stated geometry of selected features, less strongly as observations become more informative about those
    
[^430]: BehaviorBench：面向行为科学任务的基础模型基准测试

    BehaviorBench: Benchmarking Foundation Models for Behavioral Science Tasks

    [https://arxiv.org/abs/2606.24162](https://arxiv.org/abs/2606.24162)

    本文提出BehaviorBench基准，从行为预测与模拟、战略决策、被试特质推断和行为知识应用四大核心能力系统评估基础模型，并同时考察个体层面准确性与群体分布层面一致性，揭示当前领先模型在行为科学任务上仍面临挑战。

    

    基础模型正日益被应用于心理学、社会学和经济学等行为科学领域。虽然这些模型在调查响应预测和人类被试实验模拟等任务中展现出前景，但人们对它们在各类行为科学任务中的表现仍缺乏系统性的理解。我们提出了BehaviorBench，这是一个综合性基准，从四项核心能力评估基础模型：（1）行为预测与模拟，（2）战略决策，（3）被试特质推断，以及（4）行为知识应用。至关重要的是，BehaviorBench在个体和分布两个层面评估模型输出，不仅衡量单个被试的准确性，还衡量群体层面的一致性，而后者是行为有效性的基本要求。我们的评估表明，BehaviorBench对于领先的通用大语言模型和行为基础模型而言仍然具有挑战性。

    arXiv:2606.24162v2 Announce Type: replace  Abstract: Foundation models have been increasingly applied to behavioral science domains such as psychology, sociology, and economics. While these models show promise in tasks such as survey response prediction and human-subject experiment simulation, there remains no systematic understanding of how well they perform across diverse behavioral science tasks. We introduce BehaviorBench, a comprehensive benchmark that evaluates foundation models along four core capabilities: (1) behavior prediction and simulation, (2) strategic decision-making, (3) subject-trait inference, and (4) behavioral knowledge application. Crucially, BehaviorBench evaluates model outputs at both the individual and distributional levels, capturing not only per-subject accuracy but also population-level alignment, an essential requirement for behavioral validity. Our evaluation shows that BehaviorBench remains challenging for leading general-purpose LLMs and behavior founda
    
[^431]: 快速行走但需谨慎：理解掩码扩散模型中的并行采样

    Walk fast but be careful: Understanding Parallel Sampling in Masked Diffusion

    [https://arxiv.org/abs/2606.22976](https://arxiv.org/abs/2606.22976)

    本文利用图上随机游走作为可验证沙盒，从理论上证明掩码扩散模型中常用的并行去掩码评分策略（如最低熵）并不普遍优于随机并行采样，性能关键取决于图的条件依赖结构，并提出了免训练的二分采样器。

    

    在本文中，我们使用图上的随机游走作为可验证的沙盒，来研究掩码扩散模型（MDMs）中的并行采样策略。我们在来自固定图的随机游走样本上训练一个掩码扩散模型。图和转移核从不展示给模型，而是作为既可控又便于评估的潜在结构。该框架为生成的游走提供了有效性检查，并通过估计的转移核提供了分布保真度的度量。利用简单图，我们在理论上证明了通过广泛使用的评分机制（如最低熵）进行并行去掩码并非普遍优于随机并行采样；即使拥有精确的条件概率，其性能也关键地取决于图所诱导的条件依赖结构，而这一现象在数独等基准测试中难以被隔离出来。我们还为掩码扩散模型开发了免训练的二分采样器，其对数级地……

    arXiv:2606.22976v2 Announce Type: replace-cross  Abstract: In this paper, we use random walks on graphs as a verifiable sandbox for studying parallel sampling strategies in masked diffusion models (MDMs). We train an MDM on random walk samples from a fixed graph. The graph and transition kernel are never shown to the model and serve as latent structure that is both controllable and enables evaluation. The framework provides a validity check for generated walks and a measure of distributional fidelity through the estimated transition kernel. Using simple graphs, we theoretically prove that parallel unmasking via widely used scores such as lowest entropy is not uniformly better than random parallel sampling; even with exact conditional probabilities, performance critically depends on the conditional dependence structure induced by the graph, a phenomenon difficult to isolate in benchmarks like Sudoku. We also develop training-free bisection samplers for MDMs, which take logarithmically m
    
[^432]: GRADE：大语言模型智能体依赖与执行的图表示

    GRADE: Graph Representation of LLM Agent Dependency and Execution

    [https://arxiv.org/abs/2606.22741](https://arxiv.org/abs/2606.22741)

    GRADE框架将大语言模型智能体的一次运行表示为包含分级依赖边（观测、声明、推断）的类型化图，并通过实验揭示依赖信息在跨语料库迁移中的失败预测能力比运行规模更稳健，但其增益高度依赖于所选择的探针模型。

    

    一条轨迹记录了大语言模型智能体在每一步做了什么，如果同时记录每一步依赖了什么，会带来什么收获？GRADE将一次运行表示为一个类型化图：执行边可直接从轨迹中免费获得，依赖边则由外部提供，且每条依赖边被分级为观测、声明或推断三类。在六个涵盖工具使用、编程和网页场景的观测依赖语料库上，我们在固定的逻辑回归探针下，对依赖层相对于运行规模的代价进行了评估。在语料库内部，该层在三个语料库上增加了失败预测信号，但其中一个增量在任务分组交叉验证下消失，另一个在三次样条下出现反转。在留一语料库迁移实验中，规模归一化后的依赖块在每个留出语料库上都保持在随机水平之上，而运行规模在其中两个语料库上出现反转。这种模式归属于探针本身：在三次样条下，同样的依赖块在六个语料库中的四个上降到随机水平以下。随后，一项预注册的控制实验固定了节点集、步骤顺序和边数……（摘要在此处截断）

    arXiv:2606.22741v2 Announce Type: replace  Abstract: A trace records what an LLM agent did at each step. What is gained by also recording what each step relied on? GRADE represents a run as one typed graph: execution edges come free from the trace, and dependency edges are supplied, each graded observed, declared, or inferred. On six observed-dependency corpora spanning tool use, coding, and the web, we price the dependency layer against run size under a fixed logistic probe. Within corpus the layer adds failure-prediction signal on three corpora, though one increment disappears under task-grouped folds and another reverses under a cubic spline. In leave-one-corpus-out transfer the size-normalized dependency block stays above chance on every held-out corpus while run size inverts on two. That pattern belongs to the probe: under a cubic spline the same block falls below chance on four of the six. A preregistered control then holds the node set, the step order and the edge count fixed an
    
[^433]: 基于强化学习的物联网路口交通信号控制

    Reinforcement Learning-Based Traffic Signal Control for IoT-Enabled Intersections

    [https://arxiv.org/abs/2606.22108](https://arxiv.org/abs/2606.22108)

    本文为科威特一个信号控制交叉口开发了基于PPO的边缘智能强化学习信号控制器，仅利用本地观测的交通状态动态分配绿灯时长、无需未来需求信息或集中式协调，基于真实交通数据的仿真表明其性能优于固定时长控制和车辆感应控制。

    

    在依赖汽车的城市中，城市交通拥堵始终是一个持续存在的挑战，带来了巨大的经济和社会成本。交通信号系统正日益作为网络化的信息物理组件部署在智慧城市基础设施中，其中分布式感知与边缘智能使自适应交通管理成为可能。本文研究了强化学习（RL）作为一种边缘智能方法，用于科威特一个信号控制城市交叉口的自适应交通信号运行。研究开发了一种基于近端策略优化（PPO）的控制器，利用本地观测的交通状态动态分配绿灯相位时长，而无需依赖未来需求信息或集中式协调。该控制器在由科威特真实逐小时交通流量数据构建的真实感仿真环境中进行评估，并与传统的固定时长控制和车辆感应控制进行了比较（摘要原文在此处截断）。

    arXiv:2606.22108v2 Announce Type: replace-cross  Abstract: Urban traffic congestion remains a persistent challenge in car-dependent cities, imposing significant economic and societal costs. Traffic signal systems are increasingly deployed as networked cyber-physical components within smart-city infrastructures, where distributed sensing and edge intelligence enable adaptive traffic management. This paper investigates reinforcement learning (RL) as an edge-intelligent approach for adaptive traffic signal operation at a signalized urban intersection in Kuwait. A Proximal Policy Optimization (PPO)-based controller is developed to dynamically allocate green-phase durations using locally observed traffic states, without relying on future demand information or centralized coordination. The controller is evaluated in a realistic simulation environment informed by real-world hourly traffic volume data from Kuwait, and is compared against both conventional fixed-time control and a vehicle-actua
    
[^434]: SAE++：级联稀疏自编码器在多模态大语言模型中学习多层次视觉概念

    SAE++: Cascaded Sparse Autoencoders Learn Multi-Level Visual Concepts in Multimodal LLMs

    [https://arxiv.org/abs/2606.16193](https://arxiv.org/abs/2606.16193)

    SAE++提出级联稀疏自编码器架构，直接在第一级SAE的解码器权重上训练第二级SAE，从而在多模态大语言模型中学习层次化的“概念的概念”视觉表征。

    

    多模态大语言模型（MLLMs）在视觉-语言任务上展现出强大的性能，但其内部的视觉表征仍然难以解释。稀疏自编码器（SAEs）提供了一种可扩展的方法，可以将密集的模型激活分解为稀疏、可解释的特征。然而，现有的SAE架构主要恢复的是平坦的特征字典，不太适合显式的多层次概念组织。在本文中，我们提出了一种级联稀疏自编码器架构，称为SAE++，用于在MLLMs中学习层次化的视觉概念。SAE++不是嵌套或堆叠SAE的稀疏激活码，而是直接在第一级SAE的解码器权重上训练第二级SAE，将学习到的低级特征方向作为更高层次抽象的输入。这种设计使SAE++能够学习“概念的概念”，同时避免了嵌套式SAE的共享前缀耦合带来的缺陷……（摘要原文在此处截断）

    arXiv:2606.16193v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) have demonstrated strong performance on vision-language tasks, yet their internal visual representations remain difficult to interpret. Sparse Autoencoders (SAEs) provide a scalable way to decompose dense model activations into sparse, interpretable features. However, existing SAE architectures primarily recover flat feature dictionaries and are less suited for explicit multi-level concept organization. In this paper, we introduce a cascaded sparse autoencoder architecture, dubbed SAE++, for learning hierarchical visual concepts in MLLMs. Rather than nesting or stacking SAE sparse activation codes, SAE++ trains a second-level SAE directly on the decoder weights of the first-level SAE, treating learned low-level feature directions as inputs for higher-level abstraction. This design enables SAE++ to learn "concepts of concepts" while avoiding drawbacks from the shared-prefix coupling of ne
    
[^435]: 一种基于阻变存储器（RRAM）的径向基函数神经元硬件实现，用于边缘分类器

    An RRAM-based Hardware Implementation of a Radial Basis Function Neuron for Edge Classifiers

    [https://arxiv.org/abs/2606.14739](https://arxiv.org/abs/2606.14739)

    本文提出一种基于RRAM模拟内容寻址存储器的径向基函数神经元硬件设计，其中每个可配置的TXL单元充当感受野神经元，为边缘设备上的度量分类与在线适应提供了高效的硬件实现方案。

    

    现代机器学习（ML）解决方案在资源受限的边缘设备上的部署凸显了实现方面的挑战，对于包含安全关键组件的极端边缘应用（如自主导航任务）尤其如此。本文展示了一种人工神经网络（ANN）设计，该设计利用基于金属氧化物阻变存储器（RRAM）的模拟内容寻址存储器（ACAM）作为高效的硬件基底，用于在边缘端执行基于度量的分类和在线适应。所提出的设计基于用于构建ACAM模块的自定义模板像素单元，其中每个TXL单元充当一个可配置的感受野神经元。这些单元采用径向基激活函数来计算输入与编程设定的感受野之间的距离。TXL单元可以组织成密集阵列，用于计算高维输入与……

    arXiv:2606.14739v2 Announce Type: replace-cross  Abstract: The deployment of modern machine learning (ML) solutions on resource-constrained edge devices highlights implementation challenges. This is especially true for extreme edge applications that include safety-critical components, such as autonomous navigation tasks. This paper demonstrates an artificial neural network (ANN) design leveraging Metal-Oxide Resistive RAM (RRAM) -based Analogue Content Addressable Memory (ACAM) as an efficient hardware substrate for performing metric-based classification and online adaptation on the edge. The proposed design is based on a custom Template piXeL (TXL) cell used for building the ACAM module, where each TXL cell acts as a configurable receptive field neuron. These cells employ a Radial Basis activation function to calculate the distance of an input from the programmed receptive field. The TXL can be organised into dense arrays for calculating the distance of a high-dimensional input agains
    
[^436]: 面向基于种群优化的算子微积分：模块化收敛性与有限种群保证

    Operator Calculus for Population-Based Optimization: Modular Convergence and Finite-Population Guarantees

    [https://arxiv.org/abs/2606.14289](https://arxiv.org/abs/2606.14289)

    本文提出一种面向基于种群优化的算子微积分框架，使经过独立验证的更新规则效应可以模块化地组合，为收敛性分析提供可复用的构建模块，并给出有限评估预算下的收敛保证。

    

    基于种群的优化器将变异、选择和重组等更新规则组合在一起。当其中某条规则发生变化时，通常不清楚哪些收敛保证仍然成立，以及应如何评估新的组合。我们发展了一种算子微积分：算子即种群更新规则，而该微积分规定了如何将各自经过独立验证的效应进行组合。在明确的正则性和小步长条件下，由更新引起的主要变化可以相加，从而为收敛分析提供可复用的构建模块。该框架区分了找到并保留好的解、降低种群平均目标值以及使候选解集中于最优解附近这三类目标，并指出了获得有限评估预算保证所需的额外逼近条件。应用包括分布自适应、重组式演化和共识动力学，并验证了非凸情形。在……上进行的受控实验……

    arXiv:2606.14289v2 Announce Type: replace-cross  Abstract: Population-based optimizers combine update rules such as mutation, selection, and recombination. When one rule changes, it is often unclear which convergence guarantees survive or how the new combination should be assessed. We develop an operator calculus: an operator is a population-update rule, and the calculus specifies how separately checked effects can be combined. Under explicit regularity and small-step conditions, the leading changes caused by the updates add, yielding reusable building blocks for convergence analysis. The framework distinguishes finding and retaining a good solution, reducing the population's mean objective, and concentrating candidates near an optimizer, and identifies the extra approximation conditions needed for finite evaluation-budget guarantees. Applications include distribution adaptation, recombinative evolution, and consensus dynamics, with verified nonconvex cases. Controlled experiments on a
    
[^437]: 《藏于众目睽睽之下：使用DECOMPBENCH对智能体在分解攻击下的安全性进行基准测试》

    Hidden in Plain Sight: Benchmarking Agent Safety Against Decomposition Attacks with DECOMPBENCH

    [https://arxiv.org/abs/2606.13994](https://arxiv.org/abs/2606.13994)

    该论文提出了DeCompBench——首个专门评估智能体在分解攻击下安全性的基准测试，其采用图结构框架和“分解式设计”原则，将有害任务分解为各自无害、单独执行可绕过安全机制、但累积起来能实现恶意意图的现实可执行子任务。

    

    基于大语言模型（LLM）的智能体正变得日益强大并被广泛部署，这为现实世界中的对抗性滥用创造了越来越多的动机。一个关键的新兴威胁是“分解攻击”，即把一个有害任务拆分为更简单、表面上无害的子任务，这些子任务在单独执行时能够绕过安全机制，但累积起来却实现了恶意意图。尽管近期的基准测试已经评估了智能体在多轮对话和多工具使用场景下的安全性，但它们并未明确捕捉这种分解式的滥用形式，可能也无法代表现实的对抗性执行流程。为此，我们提出了DeCompBench，一个专门设计用于评估分解攻击下智能体安全性的基准测试。DeCompBench基于“分解式设计”原则，采用图结构框架构建，能够将有害任务分解为各自无害且可执行的子任务，并具有现实的……

    arXiv:2606.13994v2 Announce Type: replace-cross  Abstract: LLM-based Agents are becoming increasingly capable and widely deployed, creating growing incentives for adversarial misuse in the real-world. A key emerging threat is Decomposition Attacks \cite{glukhov2024breach, jones2024adversaries} in which a harmful task is broken into simpler, benign subtasks that evade safety mechanisms when executed separately but cumulatively fulfill the malicious intent. Although recent benchmarks assess agent safety in multi-turn and multi-tool-use settings, they do not explicitly capture this form of decompositional misuse and may not represent realistic adversarial execution flows. To this end, we introduce DeCompBench, a benchmark designed specifically to evaluate agentic safety under decomposition attacks. DeCompBench is created with a decomposition-by-design principle using a graphical framework and enables harmful task decomposition into individually benign and executable subtasks with realisti
    
[^438]: 跨智能体学习信号实现协同的角色分解式大语言模型训练

    Cross-Agent Learning Signals Enable Coordinated Role-Decomposed LLM Training

    [https://arxiv.org/abs/2606.10684](https://arxiv.org/abs/2606.10684)

    DAC 是一个角色分解训练框架，通过跨智能体的角色特定交叉验证奖励将搜索与生成解耦训练，并借助弃权机制与困难正例证据增强实现精细化的信用分配，从而在多个问答基准上持续超越强基线。

    

    代理式搜索系统需要协调证据获取与响应生成，然而现有方法要么在单一智能体目标下将两种角色耦合在一起，要么在分解角色时未能解耦各自对最终结果的贡献。我们提出了 DAC（Divide and Cooperate，分而协作），一个角色分解的训练框架：在给定任务特定外部验证信号的条件下，通过角色特定的交叉验证奖励来训练搜索器和生成器。DAC 允许生成器在检索到的证据看似不足时选择弃权，并将这一决策与外部评估的搜索充分性相结合，为每个角色分配合适的信用。为防止退化的过度弃权，我们进一步引入了困难正例证据增强机制，阻止模型在充分但困难的证据上弃权。在七个通用与多跳问答基准和两个模型主干上，DAC 持续优于强基线方法。

    arXiv:2606.10684v2 Announce Type: replace  Abstract: Agentic search systems must coordinate evidence acquisition and response generation, yet existing approaches either couple both roles under a single agent objective or decompose them without disentangling their respective contributions to the final outcome. We introduce DAC (Divide and Cooperate), a role-decomposed training framework that, given task-specific external verification signals, trains a searcher and a generator with role-specific cross-verification rewards. DAC allows the generator to abstain when the retrieved evidence appears insufficient, and uses this decision together with externally evaluated search sufficiency to assign appropriate credit to each role. To prevent degenerate over-abstention, we further introduce hard-positive evidence augmentation, which discourages abstaining on sufficient but hard evidence. Across seven general and multi-hop QA benchmarks and two model backbones, DAC consistently outperforms stron
    
[^439]: 通用干预下自动、去偏且不变的反事实生成

    Automatic, Debiased, and Invariant Counterfactual Generation under General Interventions

    [https://arxiv.org/abs/2606.07399](https://arxiv.org/abs/2606.07399)

    ADIGen框架通过结合Riesz回归、因果不变性和正交统计学习，实现了通用干预下自动、去偏且不变的反事实生成，并提供了双重稳健的风险控制保证。

    

    反事实结果的生成模型在复杂干预下的决策支持方面具有巨大潜力，但现有方法受限于估计不稳定、跨环境泛化能力差以及因干扰模型误设而产生的偏差。我们提出了ADIGen框架，用于在通用干预（包括高维干预和结果）下实现自动、去偏且不变的反事实生成。ADIGen结合了Riesz回归以避免不稳定的密度比估计、因果不变性以改善分布偏移下的泛化能力，以及正交统计学习以获得针对干扰模型误设的双重稳健保证。我们提供了超额风险界，表明ADIGen在通用干预下控制反事实风险，具有乘积偏差干扰余项和跨环境的不变风险界。然后，我们将该框架扩展到多...

    arXiv:2606.07399v2 Announce Type: replace  Abstract: Generative models for counterfactual outcomes have great potential to support decision-making under complex interventions, but existing approaches are limited by unstable estimation, poor generalization across environments, and bias from nuisance model misspecification. We introduce ADIGen, a framework for automatic, debiased, and invariant counterfactual generation under general interventions, including high-dimensional interventions and outcomes. ADIGen combines Riesz regression to avoid unstable density-ratio estimation, causal invariance to improve generalization under distribution shift, and orthogonal statistical learning to obtain doubly robust guarantees against nuisance model misspecification. We provide excess-risk bounds showing that ADIGen controls counterfactual risk under general interventions, with a product-bias nuisance remainder and an invariant risk bound across environments. We then extend this framework to multip
    
[^440]: 合成基准高估了Forward-Forward的扩展能力：层局部训练在真实数据上的极限

    Synthetic Benchmarks Overstate Forward-Forward Scaling: Real-Data Limits of Layer-Local Training

    [https://arxiv.org/abs/2606.06539](https://arxiv.org/abs/2606.06539)

    本文提出DTG-FF方法在九个真实数据基准上创下FF家族新纪录，但系统审计发现层局部Forward-Forward训练与反向传播的差距随数据规模和类别数增大而扩大，表明合成小尺寸基准高估了FF的实际扩展潜力。

    

    Forward-Forward（FF）学习[Hinton, 2022]用严格的层局部“goodness”更新取代了反向传播。最近的FF-CNN工作已在32x32基准上缩小了与反向传播（BP）的差距，这引发了一个问题：层局部训练是否正在成为现实规模下可行的替代方案。为了严格探究这一问题，我们开发了DTG-FF——动态温度goodness、解耦归一化与多层融合——作为一件工具，在我们提交时于九个真实数据基准上创下了FF家族的最先进水平（CIFAR-10达91.8%，并在ImageNet-100 224x224上建立了FF基线），并用它来审计层局部训练实际能扩展到何种程度。（1）真实数据上的扩展性。在完全相同的训练配方和骨干网络下，架构匹配的BP-DeepSup基线在CIFAR-10/CIFAR-100上分别比DTG-FF高出2.40/5.93个百分点，且该差距随类别数量的增加而扩大。在224x224分辨率下，同一方法在ImageNet-100上仅达到49.4%，而BP方法则达到75.6%……

    arXiv:2606.06539v2 Announce Type: replace-cross  Abstract: Forward-Forward (FF) learning [Hinton, 2022] replaces backpropagation with strictly layer-local goodness updates. Recent FF-CNN work has narrowed the gap to BP on 32x32 benchmarks, raising the question of whether layer-local training is becoming a viable alternative at realistic scale. To probe this rigorously, we develop DTG-FF -- dynamic temperature goodness, decoupled normalization, and multi-layer fusion -- as an instrument that sets FF-family state of the art across nine real-data benchmarks at the time of our submission (91.8% CIFAR-10 and an FF baseline at ImageNet-100 224x224), and use it to audit how far layer-local training actually scales.   (1) Real-data scaling. Under identical recipe and backbone, an architecture-matched BP-DeepSup baseline beats DTG-FF by 2.40/5.93 pp on CIFAR-10/CIFAR-100, and the gap widens with class count. At 224x224 the same instrument reaches only 49.4% on ImageNet-100, versus 75.6% even fo
    
[^441]: 面向AI使能无线接入网络的控制与响应事件检测

    Detecting Control and Response Events for AI-Enabled Radio Access Networks

    [https://arxiv.org/abs/2606.06459](https://arxiv.org/abs/2606.06459)

    本文针对AI-RAN/O-RAN中并发AI控制功能可能相互干扰的问题，提出从含噪声的参数与KPI遥测数据中检测真实控制动作及KPI相应响应事件的方法，以支撑控制参数与网络性能之间可解释依赖关系的学习。

    

    下一代无线网络正朝着使用并发的AI驱动控制功能来优化不同目标的方向发展，尤其是在AI-RAN和O-RAN架构中。当这些控制功能相互作用时，它们可能以难以从原始网络数据中察觉的方式相互干扰。管理这类相互作用所缺失的一个关键要素，是一种可靠且可解释的依赖结构，该结构能够刻画在任意给定时刻哪些控制参数正在实际影响哪些网络性能结果。本文聚焦于支持此类依赖学习所需的事件检测步骤：给定含噪声的连续参数和KPI遥测数据，我们致力于确定真实的控制动作何时发生，以及KPI何时表现出相应的由控制引起的响应。其困难之处在于，KPI波动也可能源自背景或外生变化，因此观测到的变化不能被直接视为……（摘要在此处被截断）

    arXiv:2606.06459v2 Announce Type: replace  Abstract: Next-generation wireless networks are moving toward the use of concurrent AI-driven control functions to optimize different objectives, particularly in AI-RAN and O-RAN architectures. When these functions interact, they can interfere with one another in ways that are difficult to detect from raw network data alone. A key missing piece for managing such interactions is a reliable, interpretable dependency structure that captures which control parameters are actively influencing which network performance outcomes at any given time. This paper focuses on the event-detection step needed to support such dependency learning: given noisy continuous parameter and KPI telemetry, we seek to determine when a genuine control action occurs and when a KPI exhibits a corresponding control-induced response. The difficulty is that KPI fluctuations may also arise from background or exogenous variation, so observed changes cannot be treated directly as
    
[^442]: 大型线性自编码器中学习区间的棱镜层级结构

    A prism hierarchy of learning regimes in large linear autoencoders

    [https://arxiv.org/abs/2606.05335](https://arxiv.org/abs/2606.05335)

    本文提出用三棱柱的层级结构系统地刻画大型权重绑定线性自编码器的五个基本极端学习区间（大数据、小数据、平均场、窄潜层、自由），为此类非线性于权重的模型的学习动态提供了系统化的理论图景。

    

    机器学习模型的理论研究通常会考虑不同的极限区间，在这些区间中梯度下降的学习动态在理论上变得可处理。然而，对于特定类型的模型，能够系统地获得定性上不同的极端学习区间的整体图景是人们所期望的。本文为大型权重绑定线性自编码器提出了这样一幅图景，该模型由输入维度、潜在维度、初始化幅度和训练集大小来表征。该模型在权重上是非线性的，其梯度流不存在一般的理论解。我们证明，在形式损失展开层级的层面上，其极端区间自然地与一个三棱柱的各个面相关联。特别地，存在五个与棱柱的二维面相关联的基本极端区间：（1）大数据区间、（2）小数据区间、（3）平均场区间、（4）窄潜层区间，以及（5）自由区间。对于区间（1

    arXiv:2606.05335v2 Announce Type: replace  Abstract: Theoretical studies of machine learning models commonly consider different limiting regimes in which the learning dynamics of gradient descent becomes theoretically tractable. It is, however, desirable to have a systematically obtained picture of qualitatively different extreme learning regimes for a particular type of models. In this paper we propose such a picture for large weight-tied linear autoencoders characterized by input and latent dimensions, initialization magnitude, and training set size. This model is nonlinear in the weights and its gradient flow does not have a general theoretical solution. We show that at the level of the formal loss-expansion hierarchy, its extreme regimes are naturally associated with faces of a triangular prism. In particular, there are five basic extreme regimes associated with the 2-faces of the prism: (1) large-data, (2) small-data, (3) mean-field, (4) narrow-latent, and (5) free. For regimes (1
    
[^443]: 基于神经网络代理特征值灵敏度的核临界实验梯度优化

    Gradient-based optimization of nuclear criticality experiments using neural surrogate eigenvalue sensitivities

    [https://arxiv.org/abs/2606.04033](https://arxiv.org/abs/2606.04033)

    本文提出利用物理信息神经网络的可微性进行基于梯度的核临界实验几何优化，通过最大化与目标技术的相关系数 $c_k$ 来自动设计具有高中子学相似性的新临界实验，并能搜索栅格内材料组合的组合设计空间。

    

    先进核反应堆设计和燃料概念的验证将需要设计与目标技术具有高中子学相似性的新临界实验。中子学相似性可以通过相关系数 $c_k$ 来量化，该系数捕捉了由核数据不确定性引起的 $k_\text{eff}$ 共同偏差。通常，实验需要满足 $c_k\geq0.9$ 才能与目标技术足够相似。在这项工作中，研究团队训练了一个物理信息深度神经网络来预测基于栅格的临界实验几何结构的中子学灵敏度。利用神经网络的可微性，实现了基于梯度的设计优化，通过优化新的实验几何结构来最大化其与目标技术灵敏度分布之间的 $c_k$。这种方法可以在栅格内潜在材料组合的组合设计空间上进行优化，超越了传统的（设计方法）。

    arXiv:2606.04033v2 Announce Type: replace  Abstract: The validation of advanced nuclear reactor designs and fuel concepts will require the design of new critical experiments with high neutronic similarity to the target technology. Neutronic similarity can be quantified by the correlation coefficient $c_k$, which captures the shared bias in $k_\text{eff}$ induced by uncertainties in nuclear data. Generally, a $c_k\geq0.9$ is needed for an experiment to be sufficiently similar to a target technology. In this work, a physics-informed deep neural network is trained to predict the neutronic sensitivity of grid-based critical experiment geometries. The differentiability of the neural network is used to enable gradient-based design optimization of new experiment geometries to maximize $c_k$ with the sensitivity profile of a target technology. This approach allows for optimization over the combinatorial design space of potential material combinations within the grid, moving beyond traditional 
    
[^444]: MAdam：度量感知的多目标 Adam

    MAdam: Metric-Aware Multi-Objective Adam

    [https://arxiv.org/abs/2606.03904](https://arxiv.org/abs/2606.03904)

    提出 MAdam——一个即插即用的度量感知多目标 Adam 包装器，在完全不改动求解器和优化器的前提下，纠正了 MOO 求解器与 Adam 耦合时因二阶矩纠缠导致的偏好权重失配，以及自适应度量扭曲欧氏几何导致的几何失配这两大系统性问题。

    

    多目标优化（MOO）是许多机器学习问题的基础，然而横跨损失平衡、梯度平衡和基于帕累托这三大类别的 MOO 求解器几乎无一例外地将它们调和后的方向交给 Adam 优化器。我们证明这种耦合在求解器的意图与优化器的执行之间引入了两个系统性的偏差。第一个是权重失配：Adam 的二阶矩分母将随时间变化的偏好向量与梯度统计量纠缠在一起，使偏好被边缘化为一个历史平均值，从而把不同的帕累托权衡坍缩为近乎均匀的混合。第二个是几何失配：Adam 的自适应度量扭曲了 MOO 求解器所假设的欧几里得几何，把原本一致的目标变成表面上的冲突。为了同时解决这两个问题，我们提出了 MAdam（度量感知多目标 Adam），这是一个即插即用的包装器，无需改动求解器和优化器本身。MAdam 预……（原文摘要在此处截断）

    arXiv:2606.03904v2 Announce Type: replace  Abstract: Multi-objective optimization (MOO) underlies many machine learning problems, yet MOO solvers across the loss-balancing, gradient-balancing, and Pareto-based families almost universally hand their reconciled directions to Adam~\citep{kingma2015adam}. We show this coupling introduces two systematic gaps between the solver's intent and the optimizer's execution. The first is a weighting mismatch: Adam's second-moment denominator entangles the time-varying preference vector with gradient statistics, marginalizing the preference into a history average and collapsing distinct Pareto trade-offs toward a near-uniform mixture. The second is a geometric mismatch: Adam's adaptive metric distorts the Euclidean geometry MOO solvers assume, turning aligned objectives into apparent conflicts. To resolve both jointly, we introduce MAdam (Metric-Aware Multi-Objective Adam), a drop-in wrapper that leaves both solver and optimizer unchanged. MAdam prec
    
[^445]: 迈向单步前瞻贝叶斯优化的遗憾保证

    Towards Regret Guarantees for One-Step Lookahead Bayesian Optimization

    [https://arxiv.org/abs/2606.00956](https://arxiv.org/abs/2606.00956)

    本文提出了一种仅需后验采样和蒙特卡洛近似的单步前瞻贝叶斯优化方法OVR，并通过正则化变体首次建立了趋于零的贝叶斯期望简单遗憾上界保证。

    

    本文研究了单步前瞻贝叶斯优化（BO）方法的理论保证。尽管以熵搜索为代表的单步前瞻贝叶斯优化方法的实证有效性已被广泛研究，但它们通常依赖于计算上难以处理的近似，且其遗憾保证仍不完善。因此，本文分析了一种单步前瞻贝叶斯优化方法，我们称之为最优点方差缩减（OVR），该方法仅需要后验采样和蒙特卡洛近似。我们在采集函数计算中获得了输入域上一致的蒙特卡洛估计误差界。此外，我们证明了经过轻微修改以促进探索的正则化OVR可以实现趋于零的贝叶斯期望简单遗憾上界。最后，我们通过数值实验验证了OVR和正则化OVR的性能。

    arXiv:2606.00956v2 Announce Type: replace  Abstract: This paper studies theoretical guarantees of a one-step lookahead Bayesian optimization (BO) method. Although the empirical effectiveness of one-step lookahead BO methods, such as entropy search, has been studied extensively, they often rely on computationally intractable approximations, and their regret guarantees remain underdeveloped. Thus, this paper analyzes a one-step lookahead BO method, which we refer to as optimal-point variance reduction (OVR), that requires only posterior sampling and Monte Carlo approximations. We obtain a uniform Monte Carlo estimation error bound over an input domain in an acquisition function computation. Furthermore, we show that the regularized OVR, with a slight modification to facilitate exploration, achieves a vanishing Bayesian expected simple regret upper bound. Finally, we validate the performance of OVR and regularized OVR through numerical experiments.
    
[^446]: StressDream：引导视频世界模型以实现稳健的策略评估与改进

    StressDream: Steering Video World Models for Robust Policy Evaluation and Improvement

    [https://arxiv.org/abs/2606.00267](https://arxiv.org/abs/2606.00267)

    StressDream通过优化扩散世界模型的初始噪声，将模型的想象引导向高影响但合理的场景，从而实现更稳健的机器人策略评估与改进。

    

    视频世界模型通过在以自我为中心的机器人动作条件下想象出逼真的未来观测，为策略评估与改进展现了广阔前景。尽管世界模型能够对未来结果的分布进行建模，但现有的策略评估与改进通常依赖于标称情况下的想象，这可能会遗漏机器人动作所导致的高影响结果，除非抽取多得难以承受的大量样本。为了在世界模型的想象之上实现稳健的策略评估与改进，我们提出了StressDream，该方法通过优化基于扩散的世界模型的初始噪声，将想象引导至推理时指定的高影响但合理的结果。然而，优化高维噪声极具挑战性：优化过程必须对生成视频中细微且依赖于场景的目标事件进行推理，同时避免产生导致不合理想象的分布外（OOD）噪声。我们通过两个互补的目标来解决这一问题：一个是语义目标……（摘要原文在此处截断）

    arXiv:2606.00267v2 Announce Type: replace-cross  Abstract: Video world models (WMs) have shown promise for policy evaluation and improvement by imagining realistic future observations conditioned on ego-robot actions. While WMs can model distributions over futures, policy evaluation and improvement typically rely on nominal imaginations, which can miss high-impact outcomes of robot actions unless prohibitively many samples are drawn. To enable robust policy evaluation and improvement over WM imaginations, we propose StressDream, which steers imaginations toward high-impact yet plausible outcomes specified at inference time by optimizing the initial noise of diffusion-based WMs. However, optimizing high-dimensional noise is challenging: the optimization must reason about nuanced, scene-dependent target events in generated videos while avoiding out-of-distribution (OOD) noise that yields implausible imaginations. We address this with two complementary objectives: a semantic objective wit
    
[^447]: 将机器人数据集构建建模为基于构件的构建过程

    Modeling Robotics Dataset Construction as an Artifact-Based Build Process

    [https://arxiv.org/abs/2606.00162](https://arxiv.org/abs/2606.00162)

    该论文将机器人数据集构建建模为基于依赖图的构件化构建过程，并实现了开源Bazel扩展Bagzel，相比传统顺序脚本在热构建中最高加速386倍以上，显著提升了数据集生成的可复现性与迭代效率。

    

    机器人系统会产生大量多模态传感器数据，但将ROS bag录制数据转换为机器学习数据集通常由临时编写的顺序脚本处理，这带来了工程开销和缓慢的迭代周期。我们将数据集构建建模为基于依赖图的构件化构建过程，并在Bagzel中实现了这一方法。Bagzel是一个开源的Bazel扩展，用于实现可复现的、增量的数据集生成（包括nuScenes格式导出）。我们将Bagzel和Bagzel-xattr（服务端摘要管理）与顺序执行的rosbag2nuscenes基线进行了比较。Bagzel在所有评估的执行模式下都降低了运行时间，其中在迭代工作流中收益最大（在20.4 GB数据集上，热构建最高加速386.26倍，增量构建最高加速7.21倍）。在5.1至20.4 GB的数据集规模范围内，Bagzel各变体表现出显著优于基线的扩展性，尤其是在热构建和增量构建模式下。

    arXiv:2606.00162v2 Announce Type: replace-cross  Abstract: Robotic systems generate large volumes of multimodal sensor data, but converting ROS bag recordings into machine learning datasets is often handled by ad hoc sequential scripts, creating engineering overhead and slow iteration cycles. We model dataset construction as an artifact-based build process over a dependency graph and implement this approach in Bagzel, an open-source Bazel extension for reproducible, incremental dataset generation (including nuScenes-format export). We compare Bagzel and Bagzel-xattr (server-side digest management) against a sequential rosbag2nuscenes baseline. Bagzel reduces runtime in all evaluated execution modes, with the largest gains in iterative workflows (up to 386.26x in warm builds and 7.21x in incremental builds on a 20.4 GB dataset). Across dataset sizes from 5.1 to 20.4 GB, Bagzel variants show markedly better scaling behavior than the baseline, especially in warm and incremental modes. Bag
    
[^448]: 分数广播与去相关：基于广播的信用分配的通用框架

    Score Broadcast and Decorrelation: A General Framework for Broadcast-Based Credit Assignment

    [https://arxiv.org/abs/2605.30638](https://arxiv.org/abs/2605.30638)

    提出SBD框架，通过建立输出分数与隐藏层激活之间的正交性原理，将基于广播的信用分配机制推广并统一到一般可微损失族。

    

    我们提出了分数广播与去相关（SBD），这是一个面向一般可微损失族的、基于广播的信用分配的原则性框架。误差广播是反向传播的一种生物学上合理的替代方案，它在无需权重传输的情况下将输出信息发送至隐藏层。近期针对均方误差（MSE）情形提出的误差广播与去相关（EBD）框架，将这一机制建立在最优估计量的随机正交性之上，即最优残差与输入的函数正交。我们通过引入输出分数（损失关于最后一层输出的梯度）与隐藏层激活之间的正交性原理来推广这一基础，该原理在最优分数的条件均值为零时均成立。这一单一原理将基于广播的信用分配统一到了标准可微损失族之中。

    arXiv:2605.30638v2 Announce Type: replace  Abstract: We introduce Score Broadcast and Decorrelation (SBD), a principled framework for broadcast-based credit assignment for general families of differentiable losses. Error broadcast is a biologically plausible alternative to backpropagation that sends output information to hidden layers without weight transport. The Error Broadcast and Decorrelation (EBD) framework, recently introduced for the mean-squared-error (MSE) setting, grounded this mechanism in the stochastic orthogonality of optimal estimators, under which the optimal residual is orthogonal to functions of the input. We generalize that foundation by introducing an orthogonality principle between the output score (the gradient of loss with respect to the final-layer output) and hidden-layer activations, which holds whenever the optimal score has conditional mean zero. This single principle unifies broadcast-based credit assignment across the standard differentiable-loss families
    
[^449]: 大语言模型的潜在性能剖析

    Latent Performance Profiling of Large Language Models

    [https://arxiv.org/abs/2605.30018](https://arxiv.org/abs/2605.30018)

    提出潜在性能剖析（LPP）框架，通过分析大语言模型的隐藏层激活与输出分布，从内部状态中提取与任务无关的性能诊断指标，弥补传统基准测试评估的不足。

    

    大语言模型（LLM）在标准化基准测试中经常取得令人瞩目的分数，但仅凭准确率对其能力的刻画十分有限。在排行榜上评估开源大语言模型面临着诸多长期存在的问题，例如数据污染、任务范围狭窄，以及与真实世界可靠性之间的错位。基于基准的评估方法（如 MMLU-Pro、BBH 或 IFEval）主要捕捉模型在固定测试集上输出“什么”，而非模型“如何”处理信息、校准不确定性或组织内部知识。在本文中，我们倡导从以基准为中心的评估，转向一种互补的、以状态为中心的大语言模型内在评估方法。为此，我们提出了潜在性能剖析——一个从隐藏层激活和输出分布中提取与任务无关的诊断指标的框架。LPP 在模型的潜在表示上定义了一组标量指标……

    arXiv:2605.30018v3 Announce Type: replace  Abstract: Large language models (LLMs) frequently achieve impressive scores on standardized benchmarks, yet accuracy alone offers a limited view of their capabilities. Evaluating open-source LLMs on leaderboards faces persistent issues such as data contamination, a narrow task scope, and poor alignment with real-world reliability. Benchmark-based evaluations such as MMLU-Pro, BBH, or IFEval primarily capture \textit{what} a model outputs on fixed test sets, not \textit{how} it processes information, calibrates uncertainty, or structures internal knowledge. In this article, we advocate for a shift from benchmark-centric evaluation toward a complementary, \textit{state-centered intrinsic assessment} of LLMs. To this end, we introduce \textbf{Latent Performance Profiling (LPP)} --- a framework that derives task-agnostic diagnostics from hidden activations and output distributions. LPP defines a set of scalar metrics on a model's latent representa
    
[^450]: LoopFM：从基础模型的历史表示中学习以用于推荐

    LoopFM: Learning frOm HistOrical RePresentations of Foundation Model for Recommendation

    [https://arxiv.org/abs/2605.29280](https://arxiv.org/abs/2605.29280)

    LoopFM通过将基础模型的中间嵌入结构化为下游垂直模型的输入特征（如用户历史序列），开辟了高带宽知识传递通道，在无需实时FM推理的情况下实现了显著AUC提升，并与知识蒸馏形成互补。

    

    知识蒸馏将大型基础模型（FM）的单一标量预测结果传递给紧凑的垂直模型（VM），由于单一标量无法传达大型FM所学习到的丰富中间知识，传递比率（即VM所能捕获的FM改进部分）会不断递减。为解决这一瓶颈，我们提出了LoopFM（从FM的历史表示中学习），该框架通过将FM的中间嵌入结构化为下游VM的输入特征（例如用户历史序列），开辟了一条高带宽的知识传递通道，且无需在服务阶段进行实时FM推理，也不需要FM与VM之间的架构耦合。我们为LoopFM提供了包含增益分解和传递比率分析的理论框架。在三个公开基准数据集上，LoopFM展现出显著的AUC提升（例如在TaobaoAd上提升超过6%），并展现出与知识蒸馏互补的知识传递能力。在工业界……

    arXiv:2605.29280v3 Announce Type: replace  Abstract: Knowledge distillation (KD) transfers a single scalar prediction from a large foundation model (FM) to compact vertical models (VMs), suffering from diminishing transfer ratio -- the fraction of FM improvement captured by the VM -- as a single scalar cannot convey the rich intermediate knowledge that larger FMs learn. To address this bottleneck, we propose LoopFM (Learning frOm HistOrical RePresentations of FM), a framework that opens a high-bandwidth transfer channel by structuring FM intermediate embeddings as input features (e.g., user history sequence) for downstream VMs, without requiring real-time FM inference at serving and architectural coupling between FM and VM. We provide a theoretical framework for LoopFM with a gain decomposition and transfer-ratio analysis. On three public benchmarks, LoopFM demonstrates strong AUC improvements (e.g., 6%+ on TaobaoAd) and complementary knowledge transfer capability with KD. On industria
    
[^451]: 用于解释扩散模型的残差化时序稀疏自编码器

    Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models

    [https://arxiv.org/abs/2605.27813](https://arxiv.org/abs/2605.27813)

    提出残差化时序稀疏自编码器（ReSAE），通过在相邻时间步间拟合线性预测器并对残差进行稀疏建模，实现从扩散模型完整激活轨迹中学习可解释特征。

    

    文本到图像扩散模型通过迭代去噪生成图像，因此其内部层产生的是激活值的轨迹，而非单一的静态表示。稀疏自编码器（SAE）近来被用于将扩散模型的激活分解为可解释的特征，但大多数方法要么分析单个时间步，要么以时间为条件进行建模，而非从完整轨迹中学习。若在整条轨迹上训练一个SAE，每个特征将成为跨越所有时间步的单一轨迹，但由于相邻激活值之间在很大程度上是线性可预测的，这样的SAE会将潜在容量浪费在逐步传递的冗余内容上。我们提出了残差化时序稀疏自编码器（ReSAE），它在相邻时间步之间拟合线性预测器，并用初始激活值以及这些动态无法解释的残差来表示每条轨迹。在这种表示上训练SAE等价于训练……（摘要在此处截断）

    arXiv:2605.27813v2 Announce Type: replace-cross  Abstract: Text-to-image diffusion models generate images by iterative denoising, so their internal layers produce trajectories of activations rather than single static representations. Sparse autoencoders (SAEs) have recently been used to decompose diffusion activations into interpretable features, but most approaches analyze individual timesteps or condition on time rather than learning from full trajectories. Training one SAE on whole trajectories would make each feature a single trajectory across timesteps, but adjacent activations are largely linearly predictable from one another, so such an SAE spends its latents on content carried forward from step to step. We introduce residualized temporal SAEs (ReSAE), which fit linear predictors between neighboring timesteps and represent each trajectory by its initial activation and the residuals these dynamics leave unexplained. Training an SAE on this representation is equivalent to training
    
[^452]: 开放权重大语言模型的微调防御易受简单攻击影响

    Open-Weight LLM Fine-Tuning Defenses are Susceptible to Simple Attacks

    [https://arxiv.org/abs/2605.26526](https://arxiv.org/abs/2605.26526)

    该研究揭示开放权重LLM的安全防御存在漏洞：无需梯度优化或微调的简单越狱攻击（如abliteration和prefilling）即可绕过防护，实现有害用途。

    

    近年来，用于保护开放权重大型语言模型（LLM）的防御机制旨在防止模型的对抗性滥用。这些防御机制背后隐含着一个假设：新的有害行为是通过微调习得的，而非通过越狱模型激发出来的。然而，预训练的LLM已经在众多领域编码了大量有害知识，这引出一个重要问题：攻击者能否通过越狱受保护的模型，在完全不进行微调的情况下实现有害用途？在本文中，我们表明开放权重模型的安全防护容易受到一些更简单策略的攻击——这些策略虽然广为人知，但此前尚未针对这类防护措施进行过系统性评估。具体而言，我们评估了两种低成本攻击——abliteration（消融攻击）和prefilling（预填充攻击）——它们不依赖于基于梯度的优化方法。在三个有害性评估基准上，这些攻击显著提高了针对安全防护的攻击成功率。

    arXiv:2605.26526v2 Announce Type: replace  Abstract: Recent defenses for safeguarding open-weight large language models (LLMs) are intended to prevent adversarial usage. Underlying these defenses is an assumption that new harmful behavior is learned through fine-tuning rather than elicited by jailbreaking the model. Yet, pretrained LLMs already encode substantial harmful knowledge across many domains, which raises an important question: can an adversary jailbreak safeguarded models, to achieve harmful usage without fine-tuning at all? In this paper, we show that open-weight safeguards are susceptible to simpler strategies that, despite being well known, have not been systematically evaluated against these safeguards. Specifically, we evaluate two low-cost attacks--abliteration and prefilling--that do not rely on gradient-based optimization. Across three harmfulness evaluation benchmarks (BeaverTails, HarmBench, and AdvBench), these attacks increase attack success rates against safeguar
    
[^453]: 面向物理信息神经网络的傅里叶特征金字塔

    Fourier Feature Pyramids for Physics-Informed Neural Networks

    [https://arxiv.org/abs/2605.24278](https://arxiv.org/abs/2605.24278)

    该论文提出了名为beignet的神经场架构，用可训练的多分辨率傅里叶特征金字塔取代随机傅里叶特征嵌入，从而提升物理信息神经网络求解偏微分方程的精度与计算效率。

    

    我们提出了一种改进的神经场架构，用于求解偏微分方程（PDE）。当前的物理信息神经网络（PINN）为求解PDE提供了一个灵活的框架，但它们难以获得高精度的解，并且所需计算量随参数数量的增长而扩展性较差。我们的模型被称为beignet（Bandlimited Embedding with Interpolated Grid Network，带限嵌入插值网格网络），它用可训练的多分辨率傅里叶特征金字塔取代了现有PINN模型中使用的随机傅里叶特征嵌入。为了在连续坐标处查询beignet，我们在金字塔的每一层使用傅里叶插值来返回输入坐标处的特征，然后用全连接神经网络主干对该向量进行解码。我们的模型具有多项优势：1）空间导数可以通过链式法则组合以自动微分方式计算的神经网络导数来高效求取……

    arXiv:2605.24278v2 Announce Type: replace  Abstract: We present an improved neural field architecture for solving partial differential equations (PDEs). Current physics-informed neural networks (PINNs) provide a flexible framework for solving PDEs, but they struggle to achieve highly accurate solutions and require computation that scales poorly with parameter count. Our model, which we call beignet (Bandlimited Embedding with Interpolated Grid Network), replaces the random Fourier feature embedding used by existing PINN models with a trainable multi-resolution Fourier feature pyramid. To query beignet at a continuous coordinate, we use Fourier interpolation at each level of the pyramid to return features at the input coordinate, and then decode this vector with a fully-connected neural network trunk. Our model provides multiple benefits: 1) Spatial derivatives can be computed efficiently by using the chain rule to compose derivatives of the neural network computed with automatic differ
    
[^454]: 洞察生成器：面向大语言模型智能体的系统性语料库级轨迹诊断

    Insights Generator: Systematic Corpus-Level Trace Diagnostics for LLM Agents

    [https://arxiv.org/abs/2605.21347](https://arxiv.org/abs/2605.21347)

    该论文提出了洞察生成器（IG）——一个多智能体系统，通过在执行轨迹语料库上自动提出并检验假设，生成有证据支持的系统性诊断洞察报告，解决了 LLM 智能体失败诊断依赖人工、无法规模化的问题。

    

    arXiv:2605.21347v4 公告类型：replace-cross 摘要：诊断大语言模型（LLM）智能体的失败在很大程度上仍然是手工完成的。从业者通常只检查一小部分执行轨迹，形成临时性的假设，然后不断迭代。这一过程会遗漏那些只有在轨迹群体层面才会显现的模式，也无法扩展到单条轨迹就包含数万 token 的生产级语料库。我们形式化了语料库级轨迹诊断问题：给定一个执行轨迹语料库，目标是在轨迹群体上刻画系统性的行为模式，生成有据可依的自然语言洞察，并且每条洞察都关联相应的支持性证据。我们提出了洞察生成器（Insights Generator, IG），这是一个多智能体系统，它通过在轨迹语料库上提出并检验假设来回答诊断问题，最终产出有证据支撑的洞察报告。我们从定性和客观两个维度评估了 IG，包括基于评分量表的报告评估，以及通过实施洞察所带来的下游性能提升……

    arXiv:2605.21347v4 Announce Type: replace-cross  Abstract: Diagnosing failures in LLM agents remains largely manual. Practitioners inspect a small subset of execution traces, form ad-hoc hypotheses, and iterate. This process misses patterns that only emerge across trace populations and does not scale to production corpora where individual traces span tens of thousands of tokens. We formalize the problem of corpus-level trace diagnostics. Given a corpus of execution traces, the goal is to produce grounded natural-language insights that characterize systematic behavioral patterns across trace groups, each linked to supporting evidence. We present the Insights Generator (IG), a multi-agent system that answers diagnostic questions by proposing and testing hypotheses across the trace corpus to produce an evidence-backed insights report. We evaluate IG across qualitative and objective dimensions, spanning rubric-based report assessment and downstream performance improvements achieved by impl
    
[^455]: 集合值策略学习

    Set-Valued Policy Learning

    [https://arxiv.org/abs/2605.19830](https://arxiv.org/abs/2605.19830)

    提出集合值策略学习范式，通过输出有价值治疗方案的集合（其基数反映推荐的不确定性）来更好地支持临床决策，并利用选择函数定义集合策略价值及开发双重稳健估计器。

    

    传统的治疗策略将患者协变量映射为单一的推荐干预措施，以最大化预期的临床结果。然而，当多种治疗产生统计上难以区分的结果，或治疗本身无效时，推荐单一干预措施可能导致某种程度上任意的干预选择，从而削弱临床采纳与信任。为解决这一问题，我们提出了一种集合值策略学习范式。通过输出一组有价值的治疗方案，其集合基数反映了推荐中的不确定性，我们的方法能够更好地支持临床决策。由于下游决策的可能范围广泛，评估集合值策略颇为微妙。为此，我们使用选择函数来建模临床决策，从而定义集合策略价值，并为此开发了双重稳健估计器。尽管具有实际重要性，针对分类治疗的集合值策略学习在很大程度上仍然……

    arXiv:2605.19830v2 Announce Type: replace  Abstract: Conventional treatment policies map patient covariates to a single recommended intervention in order to maximize expected clinical outcomes. However, when multiple treatments yield statistically indistinguishable outcomes or when treatment has no effect, recommending a single intervention may result in somewhat arbitrary interventions, undermining clinical adoption and trust. To address this, we propose a set-valued policy learning paradigm. By outputting sets of valuable treatments whose cardinality reflects the recommendation's ambiguity, our approach better supports clinical decision-making. Evaluating a set-valued policy proves subtle due to the range of possible downstream decisions. To do so, we define the set-policy value using a choice function to model clinical decision-making, and we develop doubly robust estimators thereof. Despite its practical importance, set-valued policy learning for categorical treatments remains larg
    
[^456]: 自然游戏过程中视觉语言模型与动作模型的推理和动作表征的脑对齐

    Brain alignment of reasoning and action representations from vision-language and action models during naturalistic gameplay

    [https://arxiv.org/abs/2605.19352](https://arxiv.org/abs/2605.19352)

    该研究首次将视觉语言模型和大动作模型在自然雅达利游戏场景下的推理与动作表征同玩家fMRI大脑活动进行对齐，揭示了动作导向与推理导向提示对模型内部表征及脑对齐效果的塑造作用。

    

    理解人类和人工智能系统如何通过与环境交互来进行预测和规划，是神经科学与机器学习交叉领域的一个根本性挑战。大多数脑编码研究聚焦于在语言理解或被动视觉处理过程中将人工智能模型与大脑活动进行对齐，而交互式脑对齐研究迄今为止主要局限于强化学习智能体和基于理论的模型。为了填补这一空白，我们利用参与者在玩自然雅达利风格电子游戏时的fMRI记录，研究了两大基础模型类型——视觉语言模型和大动作模型（LAMs）——中代表性模型的脑对齐情况。具体而言，我们考察了以动作为导向和以推理为导向的提示如何塑造模型的内部表征及其与fMRI大脑活动的对齐。首先，我们发现视觉语言模型和大动作模型都……

    arXiv:2605.19352v2 Announce Type: replace-cross  Abstract: Understanding how humans and artificial intelligence systems predict and plan by interacting with their environment is a fundamental challenge at the intersection of neuroscience and machine learning. Most brain-encoding studies focus on aligning artificial models with brain activity during language comprehension or passive visual processing, while interactive brain alignment studies have to date been largely limited to reinforcement-learning (RL) agents and theory-based models. To address this gap, we study brain alignment of representative models from two foundation-model types, namely vision-language models (VLMs) and large-action models (LAMs), using fMRI recordings from participants playing naturalistic Atari-style video games. Specifically, we examine how action-focused and reasoning-focused prompts shape the models' internal representations and their alignment with fMRI brain activity. First, we find that both VLMs and L
    
[^457]: QLIF-CAST：面向时间序列天气预报的量子泄漏积分激发模型

    QLIF-CAST: Quantum Leaky-Integrate-and-Fire for Time-Series Weather Forecasting

    [https://arxiv.org/abs/2605.18333](https://arxiv.org/abs/2605.18333)

    本文提出QLIF-CAST模型，将量子泄漏积分激发脉冲神经网络从分类任务扩展到时间序列回归，用于短期多变量天气预报，通过将神经元激发状态编码为单量子比特叠加态并嵌入混合量子-经典循环架构，在与参数匹配的经典LIF基线对比中降低了15.4%的MSE和4.4%的MAE。

    

    准确且高效的时间序列预测对经典和量子神经架构而言仍然是一个具有挑战性的问题，尤其是在多变量环境场景中。本工作将量子泄漏积分激发脉冲神经网络适配于时间序列回归任务，具体针对短期多变量天气预报。我们将QLIF的应用范围从分类扩展到连续值预测问题，证明了其在连续值预测任务上的适用性。QLIF-CAST模型将神经元激发状态编码为单量子比特量子叠加态，由R_x旋转门和T1弛豫衰减驱动，并嵌入在混合量子-经典循环架构中。我们进行了两项不同的评估。首先，在多变量天气数据集上与参数数量匹配的经典LIF基线进行受控比较，结果表明QLIF-CAST实现了15.4%更低的MSE和4.4%更低的MAE，证明量子神...

    arXiv:2605.18333v3 Announce Type: replace-cross  Abstract: Accurate and efficient time-series forecasting remains a challenging problem for both classical and quantum neural architectures, particularly in multivariate environmental settings. This work adapts the Quantum Leaky Integrate-and-Fire (QLIF) spiking neural network for time-series regression tasks, specifically short-term multivariate weather forecasting. We extend QLIF beyond classification and demonstrate its applicability to continuous-valued prediction problems. The QLIF-CAST model encodes neuron excitation states as single-qubit quantum superpositions, driven by R_x rotation gates and T1 relaxation decay, and is embedded within a hybrid quantum-classical recurrent architecture. We conduct two distinct evaluations. First, a controlled comparison against a parameter-matched classical LIF baseline on a multivariate weather dataset shows that QLIF-CAST achieves 15.4% lower MSE and 4.4% lower MAE, demonstrating that quantum ne
    
[^458]: 通过完全正提升实现线性神经网络的精确凸重构

    Exact Convex Reformulations of Linear Neural Networks via Completely Positive Lifting

    [https://arxiv.org/abs/2605.17692](https://arxiv.org/abs/2605.17692)

    本文证明了深度线性神经网络在平方损失下的训练问题可通过完全正提升被精确重构为广义完全正锥上的凸优化问题，且提升维度仅取决于输入输出维度，与网络深度和数据点数量无关。

    

    我们证明了深度线性神经网络在平方损失下的训练问题可以在广义完全正锥上的提升空间中得到精确的凸重构。该重构与原非凸问题具有相同的最优值，且在提升变量上是线性的，所有非凸性都编码在锥约束中。其环境提升维度仅取决于输入和输出维度，与网络深度和数据点数量无关，而瓶颈宽度仅通过标量约束进入。构造过程包括：将多层参数化简化为双线性分解，将其提升为秩约束半定规划，通过互补条件表达秩约束，并应用完全正提升。所得的公式为线性分解所诱导的非凸性给出了锥表示，并 connec……

    arXiv:2605.17692v2 Announce Type: replace  Abstract: We show that the training problem of a deep linear neural network under the squared loss admits an exact convex reformulation in a lifted space over a generalized completely positive cone. The reformulation has the same optimal value as the original nonconvex problem and is linear in the lifted variables, with all nonconvexity encoded in the cone constraint. Its ambient lifted dimension depends only on the input and output dimensions, independent of the network depth and the number of data points, and the bottleneck width enters only through scalar constraints. The construction proceeds by reducing the multilayer parameterization to a bilinear factorization, lifting it to a rank-constrained semidefinite program, expressing the rank constraint via a complementarity condition, and applying a completely positive lifting. The resulting formulation gives a conic representation of the nonconvexity induced by linear factorization and connec
    
[^459]: 约束潜在状态建模：竞争约束下表示学习的统一视角

    Constrained latent state modeling: A unifying perspective on representation learning under competing constraints

    [https://arxiv.org/abs/2605.15995](https://arxiv.org/abs/2605.15995)

    该论文提出约束潜在状态建模（CLSM）这一统一概念框架，用预测充分性、最小性、时间一致性、观测兼容性、抗干扰不变性和结构约束六个互补特性来刻画潜在表示，从而统一解释和比较各类表示学习方法，并阐明约束组合如何提升表示的可辨识性。

    

    arXiv:2605.15995v3 公告类型：替换 摘要：从时间序列、多模态和部分观测数据中学习潜在表示，需要明确潜在状态应当保留、舍弃和组织哪些信息。现有方法通过各式各样的异质目标函数来编码这些需求，导致不同方法难以相互比较，学习到的表示也难以解释。我们提出约束潜在状态建模（CLSM），这是一个概念性框架，通过六个互补的特性来刻画潜在状态：预测充分性、最小性、时间一致性、观测兼容性、对干扰因素的不变性以及结构约束。CLSM 将这些特性与用于诱导它们的替代目标以及用于评估它们的诊断方法分离开来，并阐明了约束的组合如何通过限制可容许表示的空间来提升可辨识性。我们据此重新解释了主要的表示学习家族……（原文摘要在此处被截断）

    arXiv:2605.15995v3 Announce Type: replace  Abstract: Learning latent representations from temporal, multimodal, and partially observed data requires specifying what information a latent state should retain, discard, and organize. Existing approaches encode these requirements through heterogeneous objectives, making methods difficult to compare and learned representations difficult to interpret. We propose Constrained Latent State Modeling (CLSM), a conceptual framework that characterizes latent states through six complementary properties: predictive sufficiency, minimality, temporal coherence, observation compatibility, invariance to nuisance factors, and structural constraints. CLSM separates these properties from the surrogate objectives used to induce them and from the diagnostics used to evaluate them, and clarifies how combinations of constraints can improve identifiability by restricting the space of admissible representations. We reinterpret major representation-learning familie
    
[^460]: 再进一步：为什么针对恶意微调的防御会在持续训练下失效

    A Few Steps Further: Why Defenses Against Malicious Finetuning Erode Under Continued Training

    [https://arxiv.org/abs/2605.14605](https://arxiv.org/abs/2605.14605)

    现有的抗恶意微调防御都建立在有限的攻击者假设之上，一旦模型权重被公开，攻击者只需在有害数据上再多训练几步就能使这些防御失效。

    

    模型提供商正越来越多地发布大语言模型的权重。尽管这些模型在发布前经过了安全对齐，但它们的防护措施往往可以通过在有害数据上进行微调而被移除。一类日益增多的防御方法旨在使对齐对这类恶意微调具有鲁棒性，但这些防御通常是在固定训练预算的攻击下进行评估的，而持有模型权重的攻击者完全可以简单地延长训练时间。我们探究当前防御能否经受住这种最简单的攻击升级。通过调研十五种近期防御方法，我们发现它们存在一个共同弱点：每种方法都建立在对攻击者的有限建模之上，例如有界扰动、短时间的模拟攻击、或通过训练建立的有害与良性行为之间的关联，而一旦权重被发布，没有任何机制来强制维持这种保护。随后，我们在四个开源权重模型上测试了六种代表性防御，通过在仅有有害数据上继续相同的微调……

    arXiv:2605.14605v3 Announce Type: replace-cross  Abstract: Model providers increasingly release the weights of large language models. Although these models are safety-aligned before release, their safeguards can often be removed by fine-tuning on harmful data. A growing class of defenses aims to make alignment robust to such malicious fine-tuning, but these defenses are typically evaluated against attacks with a fixed training budget, even though an attacker who holds the weights can simply train for longer. We ask whether current defenses withstand this simplest escalation. Surveying fifteen recent defenses, we find that they share a common weakness: each is built around a limited model of the attacker, such as a bounded perturbation, a short simulated attack, or a trained link between harmful and benign behavior, and nothing enforces that protection once the weights are released. We then test six representative defenses on four open-weight models by continuing the same harmful-only f
    
[^461]: 无损引导：面向离散扩散语言模型的机制知情干预方法

    Steering Without Breaking: Mechanistically Informed Interventions for Discrete Diffusion Language Models

    [https://arxiv.org/abs/2605.10971](https://arxiv.org/abs/2605.10971)

    该论文发现从自回归模型移植的均匀干预调度方式在离散扩散语言模型上低效且损害生成质量，并通过稀疏自编码器揭示不同属性（如主题、情感）在去噪过程中具有差异显著的形成时间表，据此提出一种自适应调度机制，将干预集中在各属性正在形成的阶段，从而实现更高效且不破坏质量的多属性引导。

    

    离散扩散语言模型（DLM）通过并行迭代去噪所有位置来生成文本，为自回归模型提供了一种替代方案。现有的DLM受控生成方法是从自回归模型中移植而来的，它们在每个去噪步骤上都施加均匀的干预。我们证明这种均匀的干预调度方式效率低下且会降低生成质量，并且当同时引导多个属性时，这种损害会进一步叠加。为了诊断这一失败，我们在四个DLM（参数量从1.24亿到80亿）上训练了稀疏自编码器，发现不同属性在不同的时间表上“确定成型”，其时机、锐度和幅度各不相同。例如，在MDLM上，主题属性在去噪过程的前2%内就已确定，而情感属性则在约20%的过程中逐渐显现。受这些特征画像的启发，我们提出了一种自适应调度机制，将干预集中在每个属性正在积极形成的阶段。理想化的分配分析预测……

    arXiv:2605.10971v2 Announce Type: replace-cross  Abstract: Discrete diffusion language models (DLMs) generate text by iteratively denoising all positions in parallel, offering an alternative to autoregressive models. Controlled generation methods for DLMs, imported from autoregressive models, apply uniform intervention at every denoising step. We show this uniform schedule is inefficient and degrades quality, and the damage compounds when multiple attributes are steered jointly. To diagnose the failure, we train sparse autoencoders on four DLMs (124M-8B parameters) and find that different attributes commit on distinct schedules, varying in timing, sharpness, and magnitude. For instance, topic commits within the first 2% of denoising on MDLM, whereas sentiment emerges gradually over 20% of the process. Motivated by these profiles, we propose an adaptive scheduling mechanism that concentrates intervention where each attribute is actively forming. An idealized allocation analysis predicts
    
[^462]: 序贯决策中机制先验的价值

    The Value of Mechanistic Priors in Sequential Decision Making

    [https://arxiv.org/abs/2605.10018](https://arxiv.org/abs/2605.10018)

    本文提出“机制信息”这一新概念，从理论上证明在序贯决策中机制先验可使贝叶斯遗憾随残差熵缩放，从而带来 $H(\mu)/H_{\mathrm{mech}}$ 的样本复杂度降低，并给出可计算的试验前模型证书。

    

    混合机制模型——即带有学习残差的物理先验——有望减少做出良好决策所需的数据量，但此前一直缺乏可计算的标准来验证这一优势。本文在渐近情形与burn-in（预热）情形下刻画了机制先验在序贯决策中的价值。为将其形式化，我们引入了模型的“机制信息”概念：即模型推荐策略 $\hat{\pi}$ 与真实最优策略 $\pi^*$ 之间的互信息，并通过居中的、占用加权的偏差来加以界定。在渐近情形（大 $N$）下，匹配的上下界表明贝叶斯遗憾随残差熵 $H_{\mathrm{mech}}$ 缩放，相比无信息基线带来了 $H(\mu)/H_{\mathrm{mech}}$ 的理论样本复杂度降低。我们进一步提供了可计算的试验前模型证书。作为补充，在具有临床相关性的burn-in情形（小 $N$）下，我们建立了关于……的下界（摘要在此处截断）。

    arXiv:2605.10018v2 Announce Type: replace  Abstract: Hybrid mechanistic models, physical priors with learned residuals, promise to reduce the data required for good decisions, but have no computable criterion to test this. We characterize the value of mechanistic priors in sequential decision-making within both asymptotic and burn-in regimes. To formalize this, we introduce the mechanistic information of a model: the mutual information between the model's recommended policy $\hat{\pi}$ and the true optimal policy $\pi^*$, bounded via a centered, occupancy-weighted bias. In the asymptotic regime (large $N$), matched bounds reveal that Bayesian regret scales with the residual entropy $H_{\mathrm{mech}}$, delivering a theoretical sample complexity reduction of $H(\mu)/H_{\mathrm{mech}}$ compared to an uninformed baseline. We further provide a computable pre-trial model certificate. Complementarily, in the clinically relevant burn-in regime (small $N$), we establish a lower bound on the pe
    
[^463]: 限制模型，遗漏系统：进攻性AI治理中的测量与问责

    Restricting the Model, Missing the System: Measurement and Accountability in Offensive AI Governance

    [https://arxiv.org/abs/2605.09504](https://arxiv.org/abs/2605.09504)

    该论文指出当前的AI进攻能力测量工具既夸大危害、又将本属于整个系统的能力错误归因于模型本身，因此AI治理不应仅限于限制模型访问，而需要转向系统级的、基于实际危害的能力评估与问责机制。

    

    我们证明，用于衡量AI进攻能力的工具存在两方面的缺陷：(i) 它们夸大了危害；(ii) 它们将本属于周围系统的能力归功于模型本身。因此，我们认为限制对模型的访问虽然是必要的，但并不充分，政策与采购还需要系统级的、基于实际危害的能力评估。2026年6月，两个前沿模型因美国出口管制而被暂停使用，据报道起因是一次越狱攻击，该攻击要求模型读取代码库并修复其中的缺陷。而这一发现所测量的是由模型、提示词和任务组成的启发“系统”。我们通过对一个开源框架的研究来支持这一论点：在该框架中，轻量级大语言模型（LLM）智能体通过共享记忆和进化优化进行协调，并提供了两方面的证据。首先，越狱指标夸大了危害：在每个目标上超过225次由群体智能生成的攻击中，LLM-as-judge（大模型作为裁判）……

    arXiv:2605.09504v3 Announce Type: replace-cross  Abstract: We show that the instruments used to measure AI offensive capability fail in two ways: (i) they overstate harm, and (ii) they credit the model with capability that belongs to the surrounding system. We argue that restricting access to a model is therefore necessary but not sufficient and that policy and procurement also need system-level, harm-grounded capability assessment. In June 2026, two frontier models were suspended under US export controls, reportedly prompted by a jailbreak that asked a model to read a codebase and fix its flaws. This finding measured an elicitation \emph{system} of model, prompt, and task. We support our argument with a study of an open-source framework in which lightweight large language model (LLM) agents coordinate through shared memory and evolutionary optimization, providing two pieces of evidence. First, jailbreak metrics overstate harm: over 225 swarm-generated attacks per target, LLM-as-judge 
    
[^464]: 扩散模型的几何感知离散化误差

    Geometry-Aware Discretization Error of Diffusion Models

    [https://arxiv.org/abs/2605.08392](https://arxiv.org/abs/2605.08392)

    该论文针对光滑逆向扩散过程推导出Euler-Maruyama弱误差与Frechet误差的渐近精确小步长展开式，并据此依据目标数据的协方差谱来优化噪声调度、重缩放系数和随机性系数等扩散采样参数。

    

    实际的扩散采样需要用有限数量的去噪步骤来模拟一个逆向时间的常微分方程（ODE）或随机微分方程（SDE），这使得采样参数的选择对于最小化离散化误差至关重要。非渐近收敛界刻画了采样复杂度，但其最坏情况下的常数可能掩盖目标数据的几何结构，从而限制了对参数优化的指导作用。我们没有采用界定误差的方式，而是针对一般光滑的逆向扩散过程，推导出了Euler-Maruyama弱误差和Frechet误差的渐近精确小步长展开式，并针对高斯数据给出了显式公式。这些公式为优化扩散参数提供了可处理的目标函数，可根据目标数据的协方差谱来优化噪声与重缩放调度以及随机性系数。特别是，我们的理论预测出：在较小的步数预算下，最优随机性更低，并展示了如何根据数据来调整重缩放系数。

    arXiv:2605.08392v2 Announce Type: replace  Abstract: Practical diffusion sampling requires simulating a reverse-time ODE or SDE with a limited number of denoising steps, making the choice of sampling parameters crucial for minimizing discretization error. Non-asymptotic convergence bounds characterize sampling complexity, but their worst-case constants can obscure target geometry and thereby limit guidance on parameter optimization. Rather than bounding the error, we derive asymptotically exact small-stepsize expansions of Euler-Maruyama weak and Frechet errors for general smooth reverse diffusions, with explicit formulas for Gaussian data. These formulas provide tractable objectives for optimizing diffusion parameters, including the noise and rescaling schedules and the stochasticity coefficient, according to the target's covariance spectrum. In particular, our theory predicts lower optimal stochasticity at smaller step budgets, shows how to adapt the rescaling coefficient to the data
    
[^465]: TAVIS：模仿学习中自我中心主动视觉与预期性注视基准

    TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning

    [https://arxiv.org/abs/2605.07943](https://arxiv.org/abs/2605.07943)

    本文提出了TAVIS基准，通过头戴与固定摄像头的配对对比协议、创新的GALT预期注视指标以及ID/OOD泛化评估，在两个人形本体上系统量化了主动视觉在模仿学习中的贡献及适用条件。

    

    主动视觉——即策略在操作过程中自主控制自身注视——已成为模仿学习中的一项关键能力，过去一年中已有多个独立系统展示了其优势。然而，目前尚缺乏一个共享基准来比较不同方法，或量化主动视觉在哪些任务类型、何种条件下带来了怎样的贡献。我们提出了TAVIS，一套用于主动视觉模仿学习的评估基础设施，包含两个互补的任务套件——TAVIS-Head（5个任务，通过平移/俯仰颈部实现全局搜索）和TAVIS-Hands（3个任务，通过腕部摄像头应对局部遮挡）——并部署于两个人形躯干本体（GR1T2、Reachy2）之上，基于IsaacLab构建。TAVIS提供了三种评估原语：在相同演示数据上进行的头戴摄像头与固定摄像头配对对比协议；GALT（注视-动作提前时间），一个基于认知科学与人机交互研究、用于量化学习策略中预期性注视行为的新型指标；以及程序化生成的ID/OOD（分布内/分布外）评估任务。

    arXiv:2605.07943v2 Announce Type: replace-cross  Abstract: Active vision -- where a policy controls its own gaze during manipulation -- has emerged as a key capability for imitation learning, with multiple independent systems demonstrating its benefits in the past year. Yet there is no shared benchmark to compare approaches or quantify what active vision contributes, on which task types, and under what conditions. We introduce TAVIS, evaluation infrastructure for active-vision imitation learning, with two complementary task suites -- TAVIS-Head (5 tasks, global search via pan/tilt necks) and TAVIS-Hands (3 tasks, local occlusion via wrist cameras) -- on two humanoid torso embodiments (GR1T2, Reachy2), built on IsaacLab. TAVIS provides three evaluation primitives: a paired headcam-vs-fixedcam protocol on identical demonstrations; GALT (Gaze-Action Lead Time), a novel metric grounded in cognitive science and HRI that quantifies anticipatory gaze in learned policies; and procedural ID/OOD
    
[^466]: Tree SAE：在稀疏自编码器中学习层次化特征结构

    Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders

    [https://arxiv.org/abs/2605.07922](https://arxiv.org/abs/2605.07922)

    该论文提出Tree SAE，通过将激活覆盖约束与新颖的重构条件相结合，使稀疏自编码器能够直接从特征集内部学习层次化特征结构，克服了仅依赖激活覆盖条件时易产生语义无关父子关系误判的缺陷。

    

    在稀疏自编码器（SAE）中学习层次化特征，对于捕捉真实世界数据的结构化特性以及缓解特征吸收或特征分裂等问题至关重要。现有工作尝试通过依赖“激活覆盖”（即子特征应仅在其父特征激活时才激活这一假设）来识别独立特征集内部的层次关系。然而，我们证明仅凭这一条件是不充分的，也就是说，它经常产生父概念与子概念在语义上毫无关联的误判（假阳性）。为了解决这一问题，我们引入了一种新颖的重构条件，在层次级别之间强制建立更深层的功能联系。通过将激活约束与重构约束相结合，我们提出了Tree SAE，一种旨在直接从特征集内部学习层次结构的模型。我们的结果表明，Tree SAE显著超越了

    arXiv:2605.07922v3 Announce Type: replace  Abstract: Learning hierarchical features in Sparse Autoencoders (SAEs) is essential for capturing the structured nature of real-world data and mitigating issues like feature absorption or splitting. Existing works attempt to identify hierarchical relationships within independent feature sets by relying on activation coverage, the assumption that child feature should only activate when its parent feature activates. However, we demonstrate that this condition alone is insufficient; that is, it often produces false positives where parent and child concepts are semantically unrelated. To address this, we introduce a novel reconstruction condition that enforces a deeper functional link between hierarchical levels. By combining both activation and reconstruction constraints, we propose the Tree SAE, a model designed to learn hierarchical structures directly from within the feature set. Our results demonstrate that Tree SAEs significantly surpass the
    
[^467]: 图像分类器中单连通决策区域的实证证据

    Empirical Evidence for Simply Connected Decision Regions in Image Classifiers

    [https://arxiv.org/abs/2605.06380](https://arxiv.org/abs/2605.06380)

    本文通过自适应四边形网格填充实验首次提供了实证证据，表明预训练图像分类器中同标签决策区域是单连通的，即区域内的任意环路都可以被区域内曲面填充。

    

    分类器决策区域的拓扑结构决定了具有相同预测标签的输入如何在不改变预测结果的情况下被连接和变形。先前的实证工作已在单个区域内构建了同标签图像之间的路径，但并未检验该区域内的环路是否能界定出位于该区域内的曲面。我们使用自适应四边形网格来研究这一问题，并对偏离标签的内部顶点进行针对性修复，同时保持同标签边界环路固定。一个有限分辨率的接受准则用于区分已成功完成的构造与在细化上限处仍未能解决的构造。在所研究的预训练分类器中，每一个被测试的环路都获得了可接受的填充。构造工作量在同一类别内部相差数个数量级，并且经过均值分数调整的随机初始化分类器所需的构造工作量大于经过训练的分类器。作为解析对照的具有已知孔洞的模型则使其缠绕环路保持未解决状态……

    arXiv:2605.06380v2 Announce Type: replace-cross  Abstract: The topology of a classifier's decision regions determines how inputs with the same predicted label can be connected and deformed without changing that prediction. Prior empirical work constructed paths between same-label images within a single region, but did not examine whether loops bound surfaces within that region. We investigate this question using adaptive quadrilateral meshes with targeted repair of off-label interior vertices, while holding the same-label boundary loop fixed. A finite-resolution acceptance criterion distinguishes completed constructions from those left unresolved at the refinement ceiling. Across the pretrained classifiers studied, every tested loop admits an accepted filling. Construction effort varies by orders of magnitude within classes and is greater for mean-score-adjusted randomly initialised classifiers than for trained classifiers. An analytic control with a known hole leaves winding loops unr
    
[^468]: 可解释性的元博弈与元归因

    The Metagame of Interpretability and Meta-Attributions

    [https://arxiv.org/abs/2605.06295](https://arxiv.org/abs/2605.06295)

    提出“元博弈”框架，将特征的归因值视为特征间的合作博弈并计算其Shapley值，从而得到方向性元归因，使任意基于梯度或注意力的归因方法都能泛化到二阶交互效应，并证明了元归因之和恰好等于其所解释的一阶归因。

    

    如何将任意的归因方法从第一性原理出发进行泛化，以捕获特征间的交互效应？我们用“元博弈”这一概念框架来回答这个问题，该框架用于量化模型解释中的二阶交互效应。我们将特征 i 的归因值 φ_i 视为其他特征之间的合作博弈，并计算其 Shapley 值，该值衡量特征 j 对特征 i 归因的影响程度，从而得到方向性元归因 φ_{j→i}。通过分解归因本身而非直接分解模型，元归因能够将任何基于梯度或注意力的方法扩展到交互效应，将基于移除的扰动方法与模型内部机制统一起来。在理论方面，我们证明了元归因之和等于其所解释的一阶归因，这是一种层级分解，而 Shapley 交互和积分 Hessian 方法实际上是隐式地执行了这种分解。在实证方面，我们展示了元（注：原文摘要在此处截断）

    arXiv:2605.06295v2 Announce Type: replace  Abstract: How can an arbitrary attribution method be generalized from first principles to capture interactions? We answer this with the metagame, a conceptual framework for quantifying second-order interaction effects of model explanations. We cast the attribution value $\phi_i$ of feature $i$ as a cooperative game among the other features and compute its Shapley value, which measures how much feature $j$ influences the attribution of $i$, yielding the directional meta-attribution $\varphi_{j \to i}$. By decomposing attribution itself rather than the model directly, meta-attributions extend any gradient- or attention-based method to interactions, uniting removal-based perturbations with model internals. Theoretically, we prove that meta-attributions sum to the first-order attribution they explain, a hierarchical decomposition that Shapley interactions and integrated Hessians turn out to perform implicitly. Empirically, we demonstrate that meta
    
[^469]: RamanBench：一个面向拉曼光谱机器学习的大规模基准测试

    RamanBench: A Large-Scale Benchmark for Machine Learning on Raman Spectroscopy

    [https://arxiv.org/abs/2605.02003](https://arxiv.org/abs/2605.02003)

    本文提出了RamanBench——首个大规模、完全可复现的拉曼光谱机器学习基准，统一整合了四个领域74个数据集共325,668条光谱，并在标准化协议下对28个模型进行了系统评估。

    

    机器学习（ML）已经变革了许多科学领域，然而一些关键应用仍然缺乏标准化的基准测试。拉曼光谱作为一种广泛使用的无创分子分析技术，正是这样一个领域——其进展受到数据集碎片化、评估标准不一致以及模型无法捕捉光谱数据结构等因素的限制。我们提出了RamanBench，这是首个大规模、完全可复现的拉曼光谱机器学习基准测试，包含简化的数据访问、评估协议与代码，以及一个实时排行榜。该基准统一整合了四个领域的74个数据集（其中16个为随本基准首次发布），共包含325,668条光谱，涵盖多种实验条件下的分类和回归任务。我们在标准化协议下对28个模型进行了基准测试，包括经典方法（如PLS）、拉曼光谱专用模型（如RamanNet）以及表格基础模型（TFM，如TabPFN）。

    arXiv:2605.02003v3 Announce Type: replace  Abstract: Machine Learning (ML) has transformed many scientific fields, yet key applications still lack standardized benchmarks. Raman spectroscopy, a widely used technique for non-invasive molecular analysis, is one such field where progress is limited by fragmented datasets, inconsistent evaluation, and models that fail to capture the structure of spectral data. We introduce RamanBench, the first large-scale, fully reproducible benchmark for ML on Raman spectroscopy, consisting of streamlined data access, evaluation protocols and code, as well as a live leaderboard. It unifies 74 datasets (including 16 first released with this benchmark) across four domains, comprising 325,668 spectra and spanning classification and regression tasks under diverse experimental conditions. We benchmark 28 models under a standardized protocol, including classical methods (e.g., PLS), Raman-specific (e.g., RamanNet), Tabular Foundation Model (TFM) (e.g., TabPFN)
    
[^470]: 从数据包到模式：将加密网络流量解读为纵向行为信号

    From Packets to Patterns: Interpreting Encrypted Network Traffic as Longitudinal Behavioral Signals

    [https://arxiv.org/abs/2605.01616](https://arxiv.org/abs/2605.01616)

    该研究表明，加密智能手机网络流量可作为被动感知方式，通过带每用户适配器的 transformer 和稀疏自编码器提取可解释的行为特征，有效捕捉与睡眠、压力和孤独感相关且具有不同时间结构的行为模式。

    

    arXiv:2605.01616v3 公告类型：replace 摘要：人类行为难以大规模地持续观察，但它会在日常设备使用中留下可测量的痕迹。我们检验了加密的智能手机网络流量——一种无处不在、始终在线的被动感知方式——能否被动地捕捉与睡眠、压力和孤独感相关的行为模式。我们使用带有每用户适配器的 transformer 主干网络来建模共享的行为结构，使模型既能表示典型的个体行为，也能表示对其的偏离。为了使这些表征具备可解释性，我们应用稀疏自编码器提取与不同活动模式相对应的行为特征。我们采用带 Mundlak 分解的广义估计方程将这些特征与睡眠障碍、压力和孤独感联系起来，从而将个体间差异与个体内随时间的变化区分开来。我们发现这三种结果反映了不同的时间结构：压力主要是……

    arXiv:2605.01616v3 Announce Type: replace  Abstract: Human behavior is difficult to observe continuously at scale, yet it leaves measurable traces in everyday device use. We test whether encrypted smartphone network traffic---a ubiquitous, always-on, passive sensing modality---can passively capture behavioral patterns related to sleep, stress, and loneliness. We model shared behavioral structure using a transformer backbone with per-user adapters, allowing the model to represent both typical individual behavior and deviations from it. To make these representations interpretable, we apply a sparse autoencoder to extract behavioral features corresponding to distinct patterns of activity. We relate these features to sleep disturbance, stress, and loneliness using generalized estimating equations with Mundlak decomposition, separating between-person differences from within-person changes over time. We find that the three outcomes reflect distinct temporal structures: stress is primarily as
    
[^471]: 面向低成本LLM服务的连续语义缓存

    Continuous Semantic Caching for Low-Cost LLM Serving

    [https://arxiv.org/abs/2604.20021](https://arxiv.org/abs/2604.20021)

    本文首次建立了不确定条件下连续查询空间中LLM语义响应缓存的严格理论框架，通过动态ε-网离散化与核岭回归相结合，突破了传统有限离散查询假设，实现低成本LLM服务。

    

    随着大语言模型日益普及，缓存响应以便让具有语义相似查询的用户能够复用，已成为降低推理成本和延迟的关键策略。现有的缓存框架假设查询处于一个有限且已知的离散查询集合中，并通过学习其服务成本和到达概率来决定缓存哪些查询响应。然而，随着LLM用户群和查询池的不断扩展，这种假设变得越来越站不住脚：现实世界中的LLM查询存在于一个无限、连续的嵌入空间中。本文建立了首个在不确定条件下、连续查询空间中语义LLM响应缓存的严格理论框架。为了弥合离散优化与连续表示空间之间的差距，我们引入了动态ε-网离散化与核岭回归相结合的方法。该设计使得……

    arXiv:2604.20021v2 Announce Type: replace-cross  Abstract: As Large Language Models (LLMs) become increasingly popular, caching responses so that they can be reused by users with semantically similar queries has become a vital strategy for reducing inference costs and latency. Existing caching frameworks have proposed to decide which query responses to cache by assuming a finite, known universe of discrete queries and learning their serving costs and arrival probabilities. As LLMs' pool of users and queries expands, however, such an assumption becomes increasingly untenable: real-world LLM queries reside in an infinite, continuous embedding space. In this paper, we establish the first rigorous theoretical framework for semantic LLM response caching in continuous query space under uncertainty. To bridge the gap between discrete optimization and continuous representation spaces, we introduce dynamic $\epsilon$-net discretization coupled with Kernel Ridge Regression. This design enables t
    
[^472]: 计算决策系统中极大似然成对排序的扰动敏感性

    Perturbation Sensitivity of Maximum-Likelihood Pairwise Ranking in Computational Decision Systems

    [https://arxiv.org/abs/2604.17805](https://arxiv.org/abs/2604.17805)

    本文将极大似然成对排序的协同扰动形式化为预算约束子集选择问题，提出自适应子集选择攻击（ASSA）这一可扩展搜索方法，并通过实验证明该排序方法对较小但协同的扰动会表现出显著的、依赖运行状态的敏感性。

    

    极大似然成对排序是一种常见的计算机制，广泛应用于优先级排序、声誉估计以及基于比较的决策支持。尽管其应用广泛，但该估计器在比较数据发生结构性变化时的扰动敏感性仍未得到充分刻画。我们将这一问题作为应用数学与计算科学中的稳定性分析问题进行研究。我们将协同扰动形式化为成对观测数据上的预算约束子集选择问题，并提出了一种自适应子集选择攻击作为可扩展的搜索启发式方法，用于探测高影响的扰动集合。通过在合成偏好数据集和真实观测偏好数据集上的实验，我们表明基于极大似然估计的排序可能表现出显著的、依赖于运行状态的敏感性：相对较小但协同的扰动可能引起输出排序的有意义变化，而其响应特征在不同状态下各不相同。

    arXiv:2604.17805v3 Announce Type: replace-cross  Abstract: Maximum-likelihood pairwise ranking is a com- mon computational mechanism for prioritization, reputation estimation, and comparison-driven decision support. Despite its broad use, the perturbation sensitivity of this estimator under structured changes in comparison data remains insufficiently characterized. We study this question as an applied-mathematics and computational-science problem in stability analysis. We for- mulate coordinated perturbation as a budgeted subset-selection problem over pairwise observations and introduce an Adaptive Subset Selection Attack (ASSA) as a scalable search heuristic for probing high-impact perturbation sets. Through experiments on synthetic and observed preference datasets, we show that MLE-based ranking can exhibit pronounced regime-dependent sensitivity: relatively small but coordinated perturbations may in- duce meaningful changes in output orderings, while the response profile varies acro
    
[^473]: 连续黑盒优化中用于算法选择的几何探测方法

    Geometric Probing for Algorithm Selection in Continuous Black-Box Optimization

    [https://arxiv.org/abs/2604.09095](https://arxiv.org/abs/2604.09095)

    该论文提出一种几何探测框架，通过多尺度二维约束采样并以有效性感知的卷积与置换不变聚合进行视觉编码，为连续黑盒优化的算法选择提供与ELA类特征互补、且在问题级迁移下更优的性能信息。

    

    连续黑盒优化的自动化算法选择取决于在有限探测预算下从问题中获取哪些信息，以及这些信息如何被表示。我们提出了一种几何探测框架，该框架在位置、方向和尺度上采样多尺度的二维约束，并通过有效性感知的卷积处理和置换不变聚合对其归一化目标值图进行编码。我们在匹配预算的条件下，通过问题内与问题级迁移、融合、表示以及预算分析，将我们的方法与经典的ELA和Deep-ELA进行比较。我们进一步通过受控消融实验将探测获取与探测处理解耦。结果表明，所提出的视觉表示能够揭示与ELA系列特征互补的求解器性能信息，并在问题级迁移下于相对期望运行时间上保持相对优势……

    arXiv:2604.09095v4 Announce Type: replace  Abstract: Automated algorithm selection for continuous black-box optimization depends on what information is acquired from a problem under a limited probing budget and how that information is represented. We introduce a geometric probing framework that samples multi-scale two-dimensional restrictions across location, orientation, and scale, and encodes their normalized objective-value maps with validity-aware convolutional processing and permutation-invariant aggregation. We compare our method with classical ELA and Deep-ELA under matched budgets, within-problem and problem-level transfer, fusion, representation, and budget analyses. We further disentangle probe acquisition from probe processing by controlled ablation. The results show that the proposed visual representation exposes solver-performance information complementary to ELA-family features and retains a relative advantage for relative expected runtime under problem-level transfer, wh
    
[^474]: 基于成分自适应与病灶级监督的脑部MRI小结构分割改进方法

    Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI

    [https://arxiv.org/abs/2604.08015](https://arxiv.org/abs/2604.08015)

    提出CATMIL训练目标，通过成分自适应Tversky加权和病灶级多示例学习两个辅助损失项，在不修改网络架构的前提下显著提升脑部MRI中小病灶的分割召回率。

    

    脑部MRI中的小病灶难以分割，因为它们仅占体积的极小部分，并且在体素级优化中被背景和较大病灶所主导，导致模型即使达到较高的Dice相似系数（DSC），仍可能遗漏许多小病灶。我们提出CATMIL，一种在不改变网络架构的情况下，向标准nnU-Net的Dice和交叉熵损失添加两个辅助项的训练目标。成分自适应Tversky（CAT）项根据病灶连通域大小的倒数对病灶体素进行加权，使每个病灶无论体积大小都贡献近乎相等的损失。病灶级多示例学习（MIL）项将每个病灶视为一个体素包，并对没有任何体素被检测出来的病灶施加惩罚。在MSLesSeg数据集的多发性硬化病灶分割任务中，CATMIL取得了最高的小病灶召回率（0.873，而Dice+CE为0.796；差异的95%置信区间为+0.030至+0.157，在所有六个测试中均更高）（摘要文本在此处截断）。

    arXiv:2604.08015v3 Announce Type: replace-cross  Abstract: Small lesions in brain MRI are hard to segment because they occupy a tiny fraction of the volume and are dominated by background and larger lesions during voxel-wise optimization, so a model can reach a high Dice similarity coefficient (DSC) while missing many of them. We propose CATMIL, a training objective that adds two auxiliary terms to the standard nnU-Net Dice and cross-entropy loss without changing the architecture. The Component-Adaptive Tversky (CAT) term weights lesion voxels by the inverse size of their connected component, so each lesion contributes nearly equally regardless of volume. The lesion-level Multiple Instance Learning (MIL) term treats each lesion as a bag of voxels and penalizes lesions with no detected voxel. For multiple sclerosis lesion segmentation on MSLesSeg, CATMIL achieves the highest small-lesion recall (0.873 vs. 0.796 for Dice+CE; 95% CI of the difference +0.030 to +0.157, higher in all six te
    
[^475]: 一种用于从多层级不完整多模态电子健康记录进行院内死亡率预测的临床点云范式

    A Clinical Point Cloud Paradigm for In-Hospital Mortality Prediction from Multi-Level Incomplete Multimodal EHRs

    [https://arxiv.org/abs/2604.04614](https://arxiv.org/abs/2604.04614)

    该论文提出HealthPoint（HP），一种统一的临床点云范式，将异构临床事件表示为连续四维空间中的点，从而在不进行刚性对齐或丢弃数据的情况下，对多层级不完整的多模态电子健康记录进行建模，用于院内死亡率预测。

    

    基于深度学习的多模态电子健康记录（EHR）建模已成为临床诊断和风险预测的重要方法。然而，由于多样化的临床工作流程和隐私限制，原始EHR本质上是多层级不完整的，包括不规则采样、模态缺失和标签稀疏。这些问题会导致时间错位、模态失衡和监督受限。大多数现有的多模态方法假设数据相对完整，即使是专为不完整性设计的方法，通常也只是孤立地解决其中一两个问题。因此，这些方法往往依赖于刚性的时间/模态对齐或直接丢弃不完整的数据，这可能会扭曲原始的临床语义。为了解决这一问题，我们提出了HealthPoint（HP），一种针对多层级不完整EHR的统一临床点云范式。HP将异构的临床事件表示为由……定义的连续四维空间中的点

    arXiv:2604.04614v3 Announce Type: replace  Abstract: Deep learning-based modeling of multimodal Electronic Health Records (EHRs) has become an important approach for clinical diagnosis and risk prediction. However, due to diverse clinical workflows and privacy constraints, raw EHRs are inherently multi-level incomplete, including irregular sampling, missing modalities, and sparse labels. These issues cause temporal misalignment, modality imbalance, and limited supervision. Most existing multimodal methods assume relatively complete data, and even methods designed for incompleteness usually address only one or two of these issues in isolation. As a result, they often rely on rigid temporal/modal alignment or discard incomplete data, which may distort raw clinical semantics. To address this problem, we propose HealthPoint (HP), a unified clinical point cloud paradigm for multi-level incomplete EHRs. HP represents heterogeneous clinical events as points in a continuous 4D space defined by
    
[^476]: 空外计算：从无线叠加中实现结构化函数提取

    Out-of-Air Computation: Enabling Structured Function Extraction from Wireless Superposition

    [https://arxiv.org/abs/2604.04312](https://arxiv.org/abs/2604.04312)

    本文提出“空外计算”这一面向提取的空中计算新范式，基于联合信源信道编码和多层嵌套格架构，从结构化的无线叠加信号中提取目标函数，无需让信道逼近理想计算媒介，且可直接处理连续值数据。

    

    空中计算广泛利用无线多址信道的叠加特性来计算分布式数据的函数。在这一大类方法中，主流的传统设计是面向嵌入式的：它们预先整形发射信号或消除信道影响，使接收到的叠加信号直接实现预定计算，这通常要求多址信道逼近一种理想的计算媒介。本文引入了空外计算，并为空中计算建立了一种面向提取的新范式。基于联合信源信道编码，AirCPU构造出一种结构化的无线叠加信号，接收端可以从中提取目标函数。AirCPU直接对连续值的设备数据进行处理，避免了单独的信源量化阶段，并采用多层嵌套格架构，通过分解……实现渐进式分辨率（摘要在此处截断）。

    arXiv:2604.04312v2 Announce Type: replace-cross  Abstract: Over-the-air computation (AirComp) broadly exploits the superposition property of wireless multiple-access channels (MACs) to compute functions of distributed data. Within this broad class, dominant conventional designs are embedding-oriented: they pre-shape transmitted signals or mitigate channel effects so that the received superposition directly realizes the prescribed computation, often requiring the MAC to approximate an ideal computational medium. This paper introduces out-of-air computation (AirCPU) and establishes an extraction-oriented paradigm for AirComp. Built on joint source-channel coding, AirCPU creates a structured wireless superposition from which the receiver extracts the target function. AirCPU operates directly on continuous-valued device data, avoiding the need for a separate source quantization stage, and employs a multi-layer nested lattice architecture that enables progressive resolution by decomposing e
    
[^477]: 基于噪声样本迭代精炼的神经全局优化方法

    Neural Global Optimization via Iterative Refinement from Noisy Samples

    [https://arxiv.org/abs/2604.03614](https://arxiv.org/abs/2604.03614)

    本文提出一种神经全局优化方法，通过迭代精炼噪声函数样本的样条表示来寻找黑盒函数的全局极小值，在多模态测试函数上将平均误差从36.24%降至8.05%，并在72%的测试用例中成功找到误差低于10%的全局极小值。

    

    从噪声样本中对黑盒函数进行全局优化是机器学习和科学计算中的一项根本性挑战。传统方法如贝叶斯优化在多模态函数上往往收敛到局部极小值，而无梯度方法则需要大量的函数评估。我们提出了一种新颖的神经方法，通过迭代精炼来学习寻找全局极小值。我们的模型以噪声函数样本及其拟合的样条表示作为输入，然后迭代地将初始猜测精炼至真实的全局极小值。该方法在随机生成的函数上进行训练，其全局极小值真值通过穷举搜索获得，在具有挑战性的多模态测试函数上实现了8.05%的平均误差，而样条初始化的误差为36.24%，实现了28.18%的提升。该模型在72%的测试用例中成功找到全局极小值，误差低于10%。

    arXiv:2604.03614v3 Announce Type: replace-cross  Abstract: Global optimization of black-box functions from noisy samples is a fundamental challenge in machine learning and scientific computing. Traditional methods such as Bayesian Optimization often converge to local minima on multi-modal functions, while gradient-free methods require many function evaluations. We present a novel neural approach that learns to find global minima through iterative refinement. Our model takes noisy function samples and their fitted spline representation as input, then iteratively refines an initial guess toward the true global minimum. Trained on randomly generated functions with ground truth global minima obtained via exhaustive search, our method achieves a mean error of 8.05 percent on challenging multi-modal test functions, compared to 36.24 percent for the spline initialization, a 28.18 percent improvement. The model successfully finds global minima in 72 percent of test cases with error below 10 pe
    
[^478]: 一种面向不确定性感知光谱图像仿真的变分潜空间框架

    A Variational Latent-Space Framework for Uncertainty-Aware Spectral Image Emulation

    [https://arxiv.org/abs/2603.21911](https://arxiv.org/abs/2603.21911)

    该论文提出一种基于变分自编码器的光谱图像仿真框架，将仿真任务表述为参数条件化的潜变量问题，在实现快速推理的同时提供逐像素不确定性估计。

    

    合成光谱图像生成对于遥感仿真和任务设计至关重要，然而基于物理的辐射传输模型（RTMs）计算成本依然高昂。现有的基于学习的仿真器降低了这一成本，但大多属于确定性的参数到光谱回归器，空间建模能力和不确定性信息有限。我们将光谱图像仿真表述为一个参数条件化的潜变量问题，并提出了一种基于变分自编码器（VAE）的框架，该框架结合了非线性光谱图像表示、快速推理和逐像素不确定性估计。该框架通过两步VAE预训练和潜空间映射，在光谱和空间-光谱两个层面得到实例化。我们在PROSAIL模拟的高光谱植被数据立方体（211个波段）和真实的Sentinel-3 OLCI多光谱海洋水色图像（21个波段）上，将其与经典回归仿真器和深度CNN进行了对比评估。

    arXiv:2603.21911v3 Announce Type: replace-cross  Abstract: Synthetic spectral image generation is essential for remote sensing simulation and mission design, yet physically based radiative transfer models (RTMs) remain computationally expensive. Existing learning-based emulators reduce this cost, but are mostly deterministic parameter-to-spectrum regressors with limited spatial modeling and uncertainty information. We formulate spectral image emulation as a parameter-conditioned latent-variable problem and propose a variational autoencoder (VAE)-based framework combining nonlinear spectral-image representations, fast inference, and per-pixel uncertainty estimates. The framework is instantiated at spectrum and spatial--spectral levels through two-step VAE pretraining and latent mapping. We evaluate it on PROSAIL-simulated hyperspectral vegetation cubes (211 bands) and real Sentinel-3 OLCI multispectral ocean-colour imagery (21 bands) against classical regression emulators and a deep CNN
    
[^479]: Bi-CamoDiffusion：一种面向伪装目标检测的边界信息引导扩散方法

    Bi-CamoDiffusion: A Boundary-informed Diffusion Approach for Camouflaged Object Detection

    [https://arxiv.org/abs/2603.13357](https://arxiv.org/abs/2603.13357)

    Bi-CamoDiffusion通过无参数边缘先验注入机制以及统一空间精度、结构约束和不确定性监督的优化目标，显著提升了伪装目标检测的边界清晰度和整体检测性能。

    

    本文提出了Bi-CamoDiffusion，这是用于伪装目标检测的CamoDiffusion框架的演进版本。该方法通过一种无参数的注入过程将边缘先验整合到早期嵌入中，从而增强边界清晰度并防止结构歧义。此外，还提出了一种统一空间精度、结构约束和不确定性监督的优化目标，使模型能够同时捕捉目标的全局上下文及其复杂的边界过渡。在CAMO、COD10K和NC4K数据集上的评估表明，Bi-CamoDiffusion超越了基线方法，对细薄结构和突出部分实现了更清晰的勾勒，同时最大限度地减少了误报。该模型在包括 $S_m$、$F_{\beta}^{w}$、$E_m$ 和 $MAE$ 在内的所有评估指标上持续优于现有的最先进方法，展现出更精确的目标-背景分离和更锐利的边界。

    arXiv:2603.13357v2 Announce Type: replace-cross  Abstract: Bi-CamoDiffusion is introduced, an evolution of the CamoDiffusion framework for camouflaged object detection. It integrates edge priors into early-stage embeddings via a parameter-free injection process, enhancing boundary sharpness and preventing structural ambiguity. An optimization objective that unifies spatial accuracy, structural constraints, and uncertainty supervision is also proposed, allowing the model to capture of both the object's global context and its intricate boundary transitions. Evaluations across the CAMO, COD10K, and NC4K datasets show that Bi-CamoDiffusion surpasses the baseline, delivering sharper delineation of thin structures and protrusions while also minimizing false positives. The model consistently outperforms existing state-of-the-art methods across all evaluated metrics, including $S_m$, $F_{\beta}^{w}$, $E_m$, and $MAE$, demonstrating a more precise object-background separation and sharper bounda
    
[^480]: 基于原型的知识引导用于细粒度结构化放射学报告生成

    Prototype-Based Knowledge Guidance for Fine-Grained Structured Radiology Reporting

    [https://arxiv.org/abs/2603.11938](https://arxiv.org/abs/2603.11938)

    提出ProtoSR方法，利用指令微调大语言模型从8万多份自由文本放射报告中自动构建以视觉原型表示的多模态知识库，将自由文本中隐含的细粒度知识注入结构化报告生成，从而提升细粒度结构化放射学报告自动化的可靠性与一致性。

    

    结构化放射学报告相比自由文本能够实现更快、更一致的交流，但其自动化仍然困难，因为模型必须在有限的结构化监督下，针对罕见发现和属性做出大量细粒度的离散决策。相比之下，自由文本报告在常规诊疗中被大规模产生，并通过详细的描述隐式编码了细粒度的、与图像关联的信息。为了利用这种非结构化知识，我们提出了ProtoSR，一种将自由文本信息注入结构化报告生成的方法。首先，我们引入了一个自动提取流水线，利用指令微调的大语言模型（LLM）挖掘80,000多份MIMIC-CXR研究，构建了一个与结构化报告模板对齐的多模态知识库，并用视觉原型表示每个答案选项。基于该知识库，ProtoSR被训练用于检索与当前图像-问题对相关的原型，从而更好地完成细粒度的结构化报告生成。

    arXiv:2603.11938v2 Announce Type: replace-cross  Abstract: Structured radiology reporting promises faster, more consistent communication than free text, but automation remains difficult as models must make many fine-grained, discrete decisions about rare findings and attributes from limited structured supervision. In contrast, free-text reports are produced at scale in routine care and implicitly encode fine-grained, image-linked information through detailed descriptions. To leverage this unstructured knowledge, we propose ProtoSR, an approach for injecting free-text information into structured report population. First, we introduce an automatic extraction pipeline that uses an instruction-tuned LLM to mine 80k+ MIMIC-CXR studies and build a multimodal knowledge base aligned with a structured reporting template, representing each answer option with a visual prototype. Using this knowledge base, ProtoSR is trained to retrieve prototypes relevant for the current image-question pair and a
    
[^481]: 轨迹即状态：LLM智能体团队的精确信用分配

    The Trace Is the State: Exact Credit Assignment for LLM Agent Teams

    [https://arxiv.org/abs/2603.06859](https://arxiv.org/abs/2603.06859)

    提出C3方法，通过在决策点替换单条消息并继续运行至最终奖励来执行反事实，从而为LLM智能体团队实现无偏且精确的信用分配，其估计方差与智能体数量无关。

    

    对于LLM智能体团队的信用分配——即每条消息的价值几何——大多被视为预测问题：用学习到的评论家、轨迹级评分或智能体消融来代替定义信用的反事实，而反事实很少被真正执行。通过共享上下文通信的团队则不同：当下游智能体读取的所有内容都写入轨迹时，轨迹就是状态，该反事实便可以被实际执行。此时信用信号可以像任何估计器一样被评判：通过偏差、方差以及与独立参考的一致性。C3（通过反事实延续进行信用分配）在决策点替换一条消息并继续运行至最终奖励，因此其信用是无偏的，且精确到蒙特卡洛误差。在给定采样替代方案的情况下，该误差的方差遵循一个推导出的定律，其中不包含智能体数量的项，且在6个由2个到（原文在此处截断）智能体组成的工作流上观察到的噪声符合该定律。

    arXiv:2603.06859v4 Announce Type: replace  Abstract: Credit assignment for a team of LLM agents, what each message was worth, has mostly been treated as prediction: a learned critic, a trajectory-level score, or an agent ablation stands in for the counterfactual that defines credit, which is rarely run. Teams that communicate through a shared context are different: when everything a downstream agent reads is written into the trace, the trace is the state, and that counterfactual can be executed. A credit signal can then be judged as any estimator is: by bias, variance, and agreement with an independent reference. C3, credit assignment by counterfactual continuation, substitutes one message at a decision point and continues the run to the terminal reward, so its credit is unbiased, exact up to Monte Carlo error. Given the sampled alternatives, that error's variance follows a derived law with no term for the number of agents, and the observed noise follows the law on 6 workflows of 2 to 
    
[^482]: 向量化字典树：面向加速器上基于大语言模型的生成式检索的高效约束解码

    Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators

    [https://arxiv.org/abs/2602.22647](https://arxiv.org/abs/2602.22647)

    提出 STATIC 方法，通过将前缀树展平为 CSR 稀疏矩阵，把不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而在 TPU/GPU 上实现高效、可扩展的基于大语言模型生成式检索的约束解码。

    

    生成式检索已成为基于大语言模型（LLM）推荐系统中的一种强大范式。然而，工业级推荐系统通常需要根据业务逻辑将输出空间限制在受约束的物品子集内（例如强制内容时效性或商品类目），而标准的自回归解码无法原生支持这种约束。此外，现有的基于前缀树（Trie）的约束解码方法在硬件加速器（TPU/GPU）上会带来严重的延迟损失。在本工作中，我们提出了 STATIC（用于约束解码的稀疏转移矩阵加速字典树索引），这是一种高效且可扩展的约束解码技术，专为在 TPU/GPU 上实现高吞吐量的基于 LLM 的生成式检索而设计。通过将前缀树展平为静态压缩稀疏行（CSR）矩阵，我们将不规则的树遍历转化为完全向量化的稀疏矩阵运算，从而释放了大规模并行计算的能力。

    arXiv:2602.22647v3 Announce Type: replace-cross  Abstract: Generative retrieval has emerged as a powerful paradigm for LLM-based recommendation. However, industrial recommender systems often benefit from restricting the output space to a constrained subset of items based on business logic (e.g. enforcing content freshness or product category), which standard autoregressive decoding cannot natively support. Moreover, existing constrained decoding methods that make use of prefix trees (Tries) incur severe latency penalties on hardware accelerators (TPUs/GPUs). In this work, we introduce STATIC (Sparse Transition Matrix-Accelerated Trie Index for Constrained Decoding), an efficient and scalable constrained decoding technique designed specifically for high-throughput LLM-based generative retrieval on TPUs/GPUs. By flattening the prefix tree into a static Compressed Sparse Row (CSR) matrix, we transform irregular tree traversals into fully vectorized sparse matrix operations, unlocking mass
    
[^483]: 值得信赖的流体：面向不可压缩流的保性质算子学习

    Fluids You Can Trust: Property-Preserving Operator Learning for Incompressible Flows

    [https://arxiv.org/abs/2602.15472](https://arxiv.org/abs/2602.15472)

    提出了一种保性质的核基算子学习方法，能够使预测速度场在解析意义上同时严格满足不可压缩性、周期性等物理性质，并具备通用逼近能力与先验收敛速率保证。

    

    我们提出了一种新颖的保性质的基于核方法的算子学习方法，用于处理由不可压缩Navier–Stokes方程所支配的不可压缩流。传统数值求解器为了满足不可压缩性约束需要付出高昂的计算成本。算子学习虽然提供了高效的代理模型，但现有的神经算子无法精确地保证不可压缩性、周期性和湍流等物理性质。我们的核方法将输入函数映射到输出函数在保性质核基下的展开系数，从而确保预测的速度场在解析意义上同时保持上述所有物理性质。我们为该框架给出了通用逼近定理以及最坏情况下的先验收敛速率；从实证结果来看，在大多数基准测试中观察到的收敛速率优于这些悲观的预测结果，这也促使我们进一步探讨更优化的表述形式。

    arXiv:2602.15472v5 Announce Type: replace-cross  Abstract: We present a novel property-preserving kernel-based operator learning method for incompressible flows governed by the incompressible Navier--Stokes equations. Traditional numerical solvers incur significant computational costs to respect incompressibility. Operator learning offers efficient surrogate models, but current neural operators fail to exactly enforce physical properties such as incompressibility, periodicity, and turbulence. Our kernel method maps input functions to expansion coefficients of output functions in a property-preserving kernel basis, ensuring that predicted velocity fields \emph{analytically} and \emph{simultaneously} preserve the aforementioned physical properties. We present universal approximation results and worst-case a priori convergence rates for our framework; empirically, the observed convergence rates exceed the pessimistic predictions across most benchmarks, motivating a formulation of more opt
    
[^484]: 恰逢其时：扩散语言模型的词元级早停方法

    Just on Time: Token-Level Early Stopping for Diffusion Language Models

    [https://arxiv.org/abs/2602.11133](https://arxiv.org/abs/2602.11133)

    本文提出一种无需训练的词元级早停方法，利用模型预测和局部上下文的轻量级信号动态判断每个词元的收敛时机并提前冻结，大幅减少扩散语言模型的去噪步数，在保持生成质量的同时显著提升生成效率。

    

    扩散语言模型通过迭代精炼的方式生成文本，但这一过程通常计算效率低下，因为许多词元早在最终去噪步骤之前就已达到稳定状态。我们提出了一种无需训练的词元级早停方法，能够在每个位置独立地识别其收敛状态。该方法利用从模型预测和局部上下文中提取的轻量级信号，动态判断各个词元何时可以被最终确定。这种方式在无需任务特定微调的情况下实现了自适应的逐词元冻结，大幅减少了所需的扩散步数。在涵盖数学推理、通用问答和科学理解等多个基准测试上，我们的方法在保持生成质量的同时取得了显著的效率提升。

    arXiv:2602.11133v3 Announce Type: replace-cross  Abstract: Diffusion language models generate text through iterative refinement, a process that is often computationally inefficient because many tokens reach stability long before the final denoising step. We introduce a training-free, token-level early stopping approach that identifies convergence independently at each position. Our method leverages lightweight signals derived from the model's predictions and local context to dynamically determine when individual tokens can be finalized. This yields adaptive per-token freezing without task-specific fine-tuning, substantially reducing the total number of diffusion steps required. Across diverse benchmarks, spanning mathematical reasoning, general question answering, and scientific understanding, our approach achieves substantial efficiency gains while preserving generation quality.
    
[^485]: ELROND：探索与分解扩散模型的内在能力

    ELROND: Exploring and decomposing intrinsic capabilities of diffusion models

    [https://arxiv.org/abs/2602.10216](https://arxiv.org/abs/2602.10216)

    提出ELROND方法，通过反向传播固定提示的随机生成结果之间的差异获得梯度，并利用主成分分析或稀疏自编码器将其分解为可解释方向，从而恢复扩散模型生成流形在给定条件下的切空间，实现对模型内在输出变化的系统性探索。

    

    输入到扩散模型的单个文本提示会产生由随机过程决定的广泛视觉输出，用户无法直接控制出现哪些语义变化。探索这一范围十分困难：随机搜索无法保证覆盖它，而提示编辑又较为粗糙，因为措辞上的微小变化就可能大幅改变生成的图像。我们认为，系统性的探索需要恢复模型自身如何组织其所能产生的条件分布。我们将这一结构形式化为生成流形，并提出ELROND——一种在给定条件下恢复该流形切空间的方法。为此，我们通过反向传播固定提示的随机生成结果之间的差异来收集梯度，并使用主成分分析（PCA）或稀疏自编码器将其分解为可解释的方向。我们展示了我们的方法……

    arXiv:2602.10216v2 Announce Type: replace  Abstract: A single text prompt passed to a diffusion model yields a wide range of visual outputs determined solely by a stochastic process, leaving users with no direct control over which semantic variations appear. Exploring this range is difficult: random search offers no guarantee of covering it, while prompt editing is coarse, as even a small change in wording can substantially alter the generated image. We argue that systematic exploration instead requires recovering how the model itself organizes the conditional distributions it can produce. We formalize this structure as a generative manifold, and present ELROND, a method for recovering its tangent space at a given conditioning. To that end, we collect gradients obtained by backpropagating the differences between stochastic realizations of a fixed prompt, and decompose them into interpretable directions using Principal Component Analysis or a Sparse Autoencoder. We show that our method 
    
[^486]: ANCRe：面向高效深度扩展的自适应神经连接重分配

    ANCRe: Adaptive Neural Connection Reassignment for Efficient Depth Scaling

    [https://arxiv.org/abs/2602.09009](https://arxiv.org/abs/2602.09009)

    该论文提出ANCRe框架，通过从数据中自适应学习并重新分配残差连接，以不到1%的额外开销显著提升网络深度的利用效率，并从理论上证明残差连接布局可导致收敛速率的指数级差距。

    

    摘要（arXiv:2602.09009v2，公告类型：replace-cross）：扩展网络深度一直是现代基础模型成功的核心驱动力，然而近期研究表明，深层网络往往未被充分利用。本文从优化视角重新审视了加深神经网络的默认机制——残差连接。严格的分析证明，残差连接的布局能够从根本上塑造收敛行为，甚至会引发收敛速率上的指数级差距。受此启发，我们提出了自适应神经连接重分配，这是一个具有理论依据且轻量级的框架，能够从数据中参数化并学习残差连接方式。ANCRe 以可忽略不计的计算和内存开销（<1%）自适应地重新分配残差连接，同时使网络深度得到更有效的利用。我们在大型语言模型预训练、扩散模型以及深度R（原文此处截断）等任务上进行了大量数值测试……

    arXiv:2602.09009v2 Announce Type: replace-cross  Abstract: Scaling network depth has been a central driver behind the success of modern foundation models, yet recent investigations suggest that deep layers are often underutilized. This paper revisits the default mechanism for deepening neural networks, namely residual connections, from an optimization perspective. Rigorous analysis proves that the layout of residual connections can fundamentally shape convergence behavior, and even induces an exponential gap in convergence rates. Prompted by this insight, we introduce adaptive neural connection reassignment (ANCRe), a principled and lightweight framework that parameterizes and learns residual connectivities from the data. ANCRe adaptively reassigns residual connections with negligible computational and memory overhead ($<1\%$), while enabling more effective utilization of network depth. Extensive numerical tests across pre-training of large language models, diffusion models, and deep R
    
[^487]: SkillRL：通过递归技能增强强化学习实现智能体演化

    SkillRL: Evolving Agents via Recursive Skill-Augmented Reinforcement Learning

    [https://arxiv.org/abs/2602.08234](https://arxiv.org/abs/2602.08234)

    SkillRL提出了一种通过自动技能发现构建层次化技能库SkillBank，并利用递归演化机制使技能库在强化学习中与智能体策略共同演化的框架，从而显著减少token开销并提升智能体的泛化能力。

    

    大语言模型（LLM）智能体在复杂任务中展现出了惊人的成果，但它们往往孤立运行，无法从过去的经验中学习。现有的基于记忆的方法主要存储原始轨迹，这些轨迹通常冗余且噪声较多。这阻碍了智能体提取对泛化至关重要的高层次、可复用的行为模式。在本文中，我们提出了SkillRL，这是一个通过自动技能发现和递归演化来弥合原始经验与策略改进之间差距的框架。我们的方法引入了基于经验的蒸馏机制来构建层次化技能库SkillBank，一种针对通用和任务特定启发式的自适应检索策略，以及一种递归演化机制，使技能库能够在强化学习过程中与智能体的策略共同演化。这些创新在显著减少token占用的同时提升了性能……（摘要原文在此处截断）

    arXiv:2602.08234v2 Announce Type: replace  Abstract: Large Language Model (LLM) agents have shown stunning results in complex tasks, yet they often operate in isolation, failing to learn from past experiences. Existing memory-based methods primarily store raw trajectories, which are often redundant and noise-heavy. This prevents agents from extracting high-level, reusable behavioral patterns that are essential for generalization. In this paper, we propose SkillRL, a framework that bridges the gap between raw experience and policy improvement through automatic skill discovery and recursive evolution. Our approach introduces an experience-based distillation mechanism to build a hierarchical skill library SkillBank, an adaptive retrieval strategy for general and task-specific heuristics, and a recursive evolution mechanism that allows the skill library to co-evolve with the agent's policy during reinforcement learning. These innovations significantly reduce the token footprint while enhan
    
[^488]: BONSAI：具有自然简洁性与可解释性的贝叶斯优化

    BONSAI: Bayesian Optimization with Natural Simplicity and Interpretability

    [https://arxiv.org/abs/2602.07144](https://arxiv.org/abs/2602.07144)

    提出了一种感知默认配置的贝叶斯优化策略BONSAI，它能在显式控制采集价值损失的前提下剪除对默认配置的低影响偏离，从而实现更简洁、可解释且易于审查的优化推荐。

    

    贝叶斯优化（BO）是一种流行的黑盒函数样本高效优化技术。在许多应用中，被调优的参数都伴随着经过精心设计的默认配置，实践者只在必要时才希望偏离默认值。然而，标准贝叶斯优化并不以最小化与默认配置的偏差为目标，在实践中常常将弱相关参数推向搜索空间的边界。这使得人们难以区分重要变更与虚假变更，并在优化目标遗漏相关运营考量时增加了审查推荐结果的负担。我们提出了BONSAI，这是一种感知默认配置的贝叶斯优化策略，它能够在显式控制采集函数价值损失的同时，剪除对默认配置影响较小的偏离。BONSAI兼容多种采集函数，包括期望改进和上置信界等。

    arXiv:2602.07144v3 Announce Type: replace  Abstract: Bayesian optimization (BO) is a popular technique for sample-efficient optimization of black-box functions. In many applications, the parameters being tuned come with a carefully engineered default configuration, and practitioners only want to deviate from this default when necessary. Standard BO, however, does not aim to minimize deviation from the default and, in practice, often pushes weakly relevant parameters to the boundary of the search space. This makes it difficult to distinguish between important and spurious changes and increases the burden of vetting recommendations when the optimization objective omits relevant operational considerations. We introduce BONSAI, a default-aware BO policy that prunes low-impact deviations from a default configuration while explicitly controlling the loss in acquisition value. BONSAI is compatible with a variety of acquisition functions, including expected improvement and upper confidence bou
    
[^489]: DiTS：多模态扩散Transformer是时间序列预测器

    DiTS: Multimodal Diffusion Transformers Are Time Series Forecasters

    [https://arxiv.org/abs/2602.06597](https://arxiv.org/abs/2602.06597)

    DiTS提出了一种多模态扩散Transformer，将内生目标与外生协变量视为不同的模态，并利用它们共享的时间坐标实现细粒度条件引导，从而完成协变量感知的概率时间序列预测。

    

    虽然生成式建模促进了概率时间序列预测的发展，但如何融合异构的外生信息仍然具有挑战性。扩散Transformer（DiT）为条件生成提供了可扩展的框架，但将其适配到预测任务需要针对时间序列定制的条件机制。内生目标与外生协变量在来源、语义和统计特性上各不相同，但它们共享时间坐标，这支持细粒度的条件引导。协变量能够描述超出直接回归所针对的条件均值之外的未来变异性和时间依赖性。基于这些考虑，我们提出了面向时间序列的扩散Transformer（DiTS），这是一个用于协变量感知预测的多模态扩散Transformer。DiTS将内生目标和外生协变量建模为不同的模态，联合利用目标历史和……对未来生成进行条件化。

    arXiv:2602.06597v2 Announce Type: replace  Abstract: While generative modeling facilitates probabilistic time series forecasting, incorporating heterogeneous exogenous information remains challenging. Diffusion Transformers (DiT) provide a scalable framework for conditional generation, yet their adaptation to forecasting calls for conditioning mechanisms tailored to time series. Endogenous targets and exogenous covariates differ in sources, semantics, and statistical characteristics, while sharing temporal coordinates that support fine-grained conditional guidance. Covariates can describe future variability and temporal dependence beyond the conditional mean targeted by direct regression. Motivated by these considerations, we propose Diffusion Transformers for Time Series (DiTS), a Multimodal Diffusion Transformer for covariate-aware forecasting. DiTS models endogenous targets and exogenous covariates as distinct modalities, jointly conditioning future generation on target history and 
    
[^490]: 面向稀疏解码的注意力质量凝聚

    Attention-Mass Condensation for Sparse Decoding

    [https://arxiv.org/abs/2602.06317](https://arxiv.org/abs/2602.06317)

    该论文通过精确的遗漏质量恒等式和下游边距条件形式化了稀疏解码中注意力质量保留与稳定贪心决策之间的区别，并实验证明：尽管稀疏解码在分布质量上可接近稠密解码，但没有任何运行能完全复现稠密贪心解码的输出。

    

    注意力质量的集中为稀疏解码创造了机会，但仅保留的注意力质量并不能保证稳定的贪心决策：检索误差、被遗漏的值方向以及递归解码过程都会产生影响。我们通过一个精确的遗漏质量恒等式和一个充分的下游边距条件来形式化这一区别，随后刻画了一种依赖于查询的均值池化块选择器。在 Qwen2-0.5B 模型上，配对的全新选择扫描实验覆盖了 97 至 769 个位置的支持集、2K 至 16K 的上下文长度，以及每个上下文的五个前缀。主要的精确匹配实验结果表明：60 次运行中没有任何一次能在 128 个 token 的生成过程中与稠密解码保持完全一致。分布质量则呈现不同的结论：当支持集规模至少为 193 时，九个上下文-支持组合条件中有七个的教师强制续写困惑度变化中位数保持在稠密解码的 5% 以内，但在提示级别上的波动范围中包含严重的 16K 上下文异常值。所有教师强制匹配率低于 70% 的七次运行……

    arXiv:2602.06317v3 Announce Type: replace-cross  Abstract: Attention-mass concentration creates an opportunity for sparse decoding, but retained mass alone does not guarantee a stable greedy decision: retrieval error, omitted value directions, and recursive decoding all matter. We formalize this distinction with an exact omitted-mass identity and a sufficient downstream margin condition, then characterize a query-dependent mean-pooled block selector. On Qwen2-0.5B, a paired fresh-selection sweep covers supports of 97--769 positions, contexts of 2K--16K, and five prefixes per context. The primary exact-match result is that none of 60 runs remains identical to dense decoding through 128 tokens. Distributional quality is distinct: for supports of at least 193, seven of nine context-support conditions have median teacher-forced continuation perplexity changes within 5\% of dense, but prompt-level ranges include severe 16K outliers. All seven runs with teacher-forced match below 70\% have p
    
[^491]: 面向混合专家模型的路由感知安全对齐

    Routing-Aware Safety Alignment for Mixture-of-Experts Models

    [https://arxiv.org/abs/2602.04448](https://arxiv.org/abs/2602.04448)

    提出RASA框架，通过识别被越狱攻击不成比例激活的安全关键专家、在固定路由下仅微调这些专家并强制路由一致性，实现了混合专家模型的路由感知安全对齐。

    

    混合专家语言模型由于其稀疏路由机制，为安全对齐带来了独特的挑战，在标准全参数微调下可能引发退化的优化行为。在初步实验中，我们观察到，对MoE模型简单地应用全参数安全微调时，攻击成功率的降低是通过路由或专家主导效应实现的，而非通过直接修复安全关键专家。为应对这一挑战，我们提出了RASA，一个路由感知的专家级对齐框架，它在显式修复安全关键专家的同时防止基于路由的绕过攻击。RASA识别被成功越狱攻击不成比例激活的专家，在固定路由的条件下仅对这些专家进行选择性微调，随后强制执行与安全对齐上下文的路由一致性。在两种代表性的MoE架构和多样化的越狱攻击场景下，RASA均展现了有效的安全对齐效果。

    arXiv:2602.04448v4 Announce Type: replace  Abstract: Mixture-of-Experts (MoE) language models introduce unique challenges for safety alignment due to their sparse routing mechanisms, which can enable degenerate optimization behaviors under standard full-parameter fine-tuning. In our preliminary experiments, we observe that naively applying full-parameter safety fine-tuning to MoE models can reduce attack success rates through routing or expert dominance effects, rather than by directly repairing Safety-Critical Experts. To address this challenge, we propose RASA, a routing-aware expert-level alignment framework that explicitly repairs Safety-Critical Experts while preventing routing-based bypasses. RASA identifies experts disproportionately activated by successful jailbreaks, selectively fine-tunes only these experts under fixed routing, and subsequently enforces routing consistency with safety-aligned contexts. Across two representative MoE architectures and a diverse set of jailbreak
    
[^492]: 使用物理结构变分自编码器（PS-VAE）实现定量分子磁共振成像中的多参数不确定性映射

    Multiparameter Uncertainty Mapping in Quantitative Molecular MRI using a Physics-Structured Variational Autoencoder (PS-VAE)

    [https://arxiv.org/abs/2602.03317](https://arxiv.org/abs/2602.03317)

    提出一种物理结构变分自编码器（PS-VAE），通过融合可微分自旋物理模拟器与自监督学习，实现定量分子MRI中体素级多参数后验分布的快速提取与不确定性量化。

    

    定量成像方法，如磁共振指纹成像（MRF），旨在通过从信号演化中估计生物物理组织参数来提取可解释的病理生物标志物。然而，此类逆问题中常用的模式匹配算法或神经网络往往缺乏有原则的不确定性量化，这限制了临床接受所必需的可信度与透明度。在此，我们提出了一种物理结构变分自编码器（PS-VAE），用于快速提取体素级的多参数后验分布。我们的方法将可微分自旋物理模拟器与自监督学习相结合，并提供了完整的协方差矩阵，以捕获潜在生物物理空间中的参数间相关性。该方法在多质子池化学交换饱和转移（CEST）与半固体磁化转移（MT）的分子MRF研究中得到了验证。

    arXiv:2602.03317v2 Announce Type: replace-cross  Abstract: Quantitative imaging methods, such as magnetic resonance fingerprinting (MRF), aim to extract interpretable pathology biomarkers by estimating biophysical tissue parameters from signal evolutions. However, the pattern-matching algorithms or neural networks used in such inverse problems often lack principled uncertainty quantification, which limits the trustworthiness and transparency, required for clinical acceptance. Here, we describe a physics-structured variational autoencoder (PS-VAE) designed for rapid extraction of voxelwise multi-parameter posterior distributions. Our approach integrates a differentiable spin physics simulator with self-supervised learning, and provides a full covariance that captures the inter-parameter correlations of the latent biophysical space. The method was validated in a multi-proton pool chemical exchange saturation transfer (CEST) and semisolid magnetization transfer (MT) molecular MRF study, a
    
[^493]: 忠实性的积极例证：大语言模型自我解释有助于预测模型行为

    A Positive Case for Faithfulness: LLM Self-Explanations Help Predict Model Behavior

    [https://arxiv.org/abs/2602.02639](https://arxiv.org/abs/2602.02639)

    该论文提出归一化可模拟性增益（NSG）这一新指标，从预测价值的角度证明了LLM自我解释的忠实性，实验表明自我解释能将对模型行为的预测能力提升11-37%。

    

    大语言模型（LLM）的自我解释常被视为AI监督领域一种有前景的工具，但其对模型真实推理过程的忠实性却鲜为人知。现有的忠实性度量方法存在关键局限，通常依赖于通过对抗性提示来识别不忠实性，或检测推理错误。这些方法忽视了解释的预测价值。我们提出了归一化可模拟性增益，这是一种通用且可扩展的度量指标，其核心思想是：一个忠实的解释应当能让观察者了解模型的决策标准，从而更好地预测模型在相关输入上的行为。我们在涵盖健康、商业和伦理等热门数据集的7,000个反事实样本上，评估了18个前沿的专有和开放权重模型，例如Gemini 3、GPT-5.2和Claude 4.5。我们发现，自我解释能显著提升对模型行为的预测能力（NSG达11-37%）。

    arXiv:2602.02639v2 Announce Type: replace-cross  Abstract: LLM self-explanations are often presented as a promising tool for AI oversight, yet their faithfulness to the model's true reasoning process is poorly understood. Existing faithfulness metrics have critical limitations, typically relying on identifying unfaithfulness via adversarial prompting or detecting reasoning errors. These methods overlook the predictive value of explanations. We introduce Normalized Simulatability Gain (NSG), a general and scalable metric based on the idea that a faithful explanation should allow an observer to learn a model's decision-making criteria, and thus better predict its behavior on related inputs. We evaluate 18 frontier proprietary and open-weight models, e.g., Gemini 3, GPT-5.2, and Claude 4.5, on 7,000 counterfactuals from popular datasets covering health, business, and ethics. We find self-explanations substantially improve prediction of model behavior (11-37% NSG). Self-explanations also p
    
[^494]: 连续效用直接偏好优化

    Continuous-Utility Direct Preference Optimization

    [https://arxiv.org/abs/2602.00931](https://arxiv.org/abs/2602.00931)

    提出CU-DPO框架，用连续效用分数取代二元偏好标签来对齐多种认知策略，并证明其在样本复杂度上相比二元偏好可获得Θ(K log K)的改进。

    

    大语言模型的推理能力通常被视为一种单一整体的能力，依赖于二元偏好监督，而这种监督方式无法捕捉部分进展或细粒度的推理质量。我们提出了连续效用直接偏好优化（CU-DPO），这是一个通过用连续分数替代二元标签来捕捉细粒度推理质量的框架，从而使模型与一组基于提示词的认知策略组合对齐。我们证明了使用K种策略进行学习相比二元偏好可在样本复杂度上获得Θ(K log K)的改进，并且DPO会收敛到熵正则化的效用最大化策略。为了利用这一信号，我们提出了一个两阶段流水线：（i）策略选择，通过“最优对所有”的比较方式优化模型选择最佳策略；（ii）执行精炼，使用按边际分层的样本对来训练正确的策略执行。该框架与具体领域无关：任何任务均可适用（原文在此处截断）。

    arXiv:2602.00931v3 Announce Type: replace  Abstract: Large language model reasoning is often treated as a monolithic capability, relying on binary preference supervision that fails to capture partial progress or fine-grained reasoning quality. We introduce continuous utility direct preference optimization (CU-DPO), a framework that aligns models to a portfolio of prompt-based cognitive strategies by replacing binary labels with continuous scores that capture fine-grained reasoning quality. We prove that learning with K strategies yields a Theta(K log K) improvement in sample complexity over binary preferences and that DPO converges to the entropy-regularized utility-maximizing policy. To exploit this signal, we propose a two-stage pipeline: (i) strategy selection, which optimizes the model to choose the best strategy via best-vs-all comparisons, and (ii) execution refinement, which trains correct execution using margin-stratified pairs. The framework is domain-agnostic: any task admitt
    
[^495]: 整流流的最优阶样本复杂度

    Order-Optimal Sample Complexity of Rectified Flows

    [https://arxiv.org/abs/2601.20250](https://arxiv.org/abs/2601.20250)

    本文证明了整流流模型在标准神经网络假设下可达到 $\tilde{O}(\varepsilon^{-2})$ 的最优阶样本复杂度，改进了流匹配模型已有的 $O(\varepsilon^{-4})$ 界并匹配均值估计的最优速率。

    

    近年来，基于流的生成模型相比扩散模型展现出了更优的效率。本文研究整流流模型，该模型约束从基础分布到数据分布的传输轨迹为线性。这种结构性限制极大地加速了采样过程，通常仅需单步欧拉步即可实现高质量生成。在用于参数化速度场和数据分布的神经网络类的标准假设下，我们证明整流流可以达到 $\tilde{O}(\varepsilon^{-2})$ 的样本复杂度。这改进了流匹配模型已知的最佳 $O(\varepsilon^{-4})$ 界，并达到了均值估计的最优速率。我们的分析利用了整流流的特殊结构：由于模型是沿线性路径以平方损失进行训练的，相关的假设类具有严格可控的局部化 Rademacher 复杂度。

    arXiv:2601.20250v2 Announce Type: replace  Abstract: Recently, flow-based generative models have shown superior efficiency compared to diffusion models. In this paper, we study rectified flow models, which constrain transport trajectories to be linear from the base distribution to the data distribution. This structural restriction greatly accelerates sampling, often enabling high-quality generation with a single Euler step. Under standard assumptions on the neural network classes used to parameterize the velocity field and data distribution, we prove that rectified flows achieve sample complexity $\tilde{O}(\varepsilon^{-2})$. This improves on the best known $O(\varepsilon^{-4})$ bounds for flow matching model and matches the optimal rate for mean estimation. Our analysis exploits the particular structure of rectified flows: because the model is trained with a squared loss along linear paths, the associated hypothesis class admits a sharply controlled localized Rademacher complexity. T
    
[^496]: 机器学习模性

    Machine learning modularity

    [https://arxiv.org/abs/2601.01779](https://arxiv.org/abs/2601.01779)

    本文提出首个基于 Transformer 序列到序列架构与动态批处理算法的机器学习框架，能够学习 SL(2,Z) 与 SL(3,Z) 模变换，自动将涉及椭圆伽马函数的高度打乱表达式化简为规范形式，在分布内测试中准确率超过 99%，并在更深的打乱深度等外推任务中仍保持 90% 以上的准确率，证明模型真正内化了模变换的代数规则。

    

    arXiv:2601.01779v2 公告类型：replace-cross 摘要：基于 Transformer 的序列到序列（sequence-to-sequence）架构并结合动态批处理算法，本工作引入了一个机器学习框架，用于自动简化涉及多个椭圆伽马函数（包括 $q$-$\theta$ 函数和椭圆伽马函数）的复杂表达式。该模型学习应用代数恒等式，特别是 SL$(2,\mathbb{Z})$ 和 SL$(3,\mathbb{Z})$ 模变换，将高度打乱的表达式化简为其规范形式。实验结果表明，该模型在分布内测试中达到超过 99% 的准确率，并在显著的外推情形下（例如更深的打乱深度）依然保持稳健的性能（准确率超过 90%）。这表明该模型已经内化了模变换的底层代数规则，而不仅仅是记忆训练样本。我们的工作首次成功……

    arXiv:2601.01779v2 Announce Type: replace-cross  Abstract: Based on a transformer based sequence-to-sequence architecture combined with a dynamic batching algorithm, this work introduces a machine learning framework for automatically simplifying complex expressions involving multiple elliptic Gamma functions, including the $q$-$\theta$ function and the elliptic Gamma function. The model learns to apply algebraic identities, particularly the SL$(2,\mathbb{Z})$ and SL$(3,\mathbb{Z})$ modular transformations, to reduce heavily scrambled expressions to their canonical forms. Experimental results show that the model achieves over 99\% accuracy on in-distribution tests and maintains robust performance (exceeding 90\% accuracy) under significant extrapolation, such as with deeper scrambling depths. This demonstrates that the model has internalized the underlying algebraic rules of modular transformations rather than merely memorizing training patterns. Our work presents the first successful a
    
[^497]: PrismSSL：一个接口，多种模态；面向多模态自监督学习的单接口库

    PrismSSL: One Interface, Many Modalities; A Single-Interface Library for Multimodal Self-Supervised Learning

    [https://arxiv.org/abs/2511.17776](https://arxiv.org/abs/2511.17776)

    PrismSSL是一个单一接口的Python库，将音频、视觉、图和跨模态的最先进自监督学习方法统一在模块化代码库中，支持少量代码即可训练与复现基准，并集成分布式训练、超参搜索、LoRA微调等实用功能。

    

    我们提出了PrismSSL，这是一个Python库，它在单一的模块化代码库中统一了跨音频、视觉、图以及跨模态设置的最先进的自监督学习（SSL）方法。该演示旨在展示研究人员和从业者如何：（i）通过几行代码完成安装、配置并运行前置任务训练；（ii）复现紧凑的基准测试；（iii）通过清晰的训练器与数据集抽象，以新的模态或方法扩展该框架。PrismSSL已打包发布于PyPI，采用MIT许可证，与HuggingFace Transformers紧密集成，并提供了多项提升易用性的功能，例如PyTorch中的分布式训练、基于Optuna的超参数搜索、针对Transformer骨干网络的LoRA微调、用于完整性检查的动画式嵌入可视化、Weights & Biases日志记录，以及彩色结构化的终端日志，从而改善可用性和清晰度。此外，PrismSSL还提供了一个图形化[摘要在此处被截断]

    arXiv:2511.17776v2 Announce Type: replace  Abstract: We present PrismSSL, a Python library that unifies state-of-the-art self-supervised learning (SSL) methods across audio, vision, graphs, and cross-modal settings in a single, modular codebase. The goal of the demo is to show how researchers and practitioners can: (i) install, configure, and run pretext training with a few lines of code; (ii) reproduce compact benchmarks; and (iii) extend the framework with new modalities or methods through clean trainer and dataset abstractions. PrismSSL is packaged on PyPI, released under the MIT license, integrates tightly with HuggingFace Transformers, and provides quality-of-life features such as distributed training in PyTorch, Optuna-based hyperparameter search, LoRA fine-tuning for Transformer backbones, animated embedding visualizations for sanity checks, Weights & Biases logging, and colorful, structured terminal logs for improved usability and clarity. In addition, PrismSSL offers a graphic
    
[^498]: 无需参考模型的模型级成员推断脆弱性估计

    Estimating Model-Level Membership Inference Vulnerability Without Reference Models

    [https://arxiv.org/abs/2510.19773](https://arxiv.org/abs/2510.19773)

    提出了一种无需训练任何参考模型、仅利用目标模型的训练与测试损失分布即可估计模型对最强成员推断攻击LiRA脆弱性的新方法，并揭示了不同模型状态对应不同损失统计量的适用区间。

    

    成员推断攻击（MIA）已成为评估AI模型隐私风险的标准工具。然而，最先进的攻击方法需要训练大量通常计算代价高昂的参考模型，这限制了其实用性。我们提出了一种新颖的方法，可直接从目标模型的训练损失与测试损失分布出发，估计模型对似然比攻击（LiRA）——目前最强的攻击方法——的模型级脆弱性，而无需训练任何参考模型。我们证明LiRA的逐样本信号可分解为一个方差比项和一个残差均值偏移项，两者的相对贡献取决于训练在多大程度上坍缩了模型在已训练样本处的不确定性。这将模型置于一个连续谱之上，不同的区间需要采用不同的无参考、基于损失的统计量作为LiRA真正阳性率（TPR）的代理指标。损失分布自身的形状可以指示应采用哪种（摘要在此处截断）

    arXiv:2510.19773v2 Announce Type: replace  Abstract: Membership inference attacks (MIAs) have emerged as the standard tool for evaluating the privacy risks of AI models. However, state-of-the-art attacks require training numerous, often computationally expensive, reference models, limiting their practicality. We present a novel approach for estimating model-level vulnerability to the Likelihood Ratio Attack (LiRA), the strongest available attack, directly from the train and test loss distributions of the target model and without training any reference models. We show that LiRA's per-sample signal decomposes into a variance-ratio term and a residual mean-shift term, with the relative contribution of each determined by how much training collapses model uncertainty at the trained sample. This places models on a continuum, with different regimes calling for different reference-free loss-based statistics as proxies for LiRA TPR. The shapes of the loss distributions themselves indicate which
    
[^499]: 基于激活信息与帕累托引导的低秩压缩方法，实现高效的大语言模型/视觉语言模型

    Activation-Informed Pareto-Guided Low-Rank Compression for Efficient LLM/VLM

    [https://arxiv.org/abs/2510.05544](https://arxiv.org/abs/2510.05544)

    提出基于激活压缩误差理论上界的帕累托引导低秩压缩框架PGSVD，通过异构秩分配在相同压缩率下为LLM/VLM实现更高精度与推理加速。

    

    大语言模型（LLM）和视觉语言模型（VLM）已取得最先进的性能，但在部署时带来了显著的内存和计算挑战。我们提出了一种新颖的低秩压缩框架来应对这一挑战。首先，我们通过基于各层激活的压缩误差为网络损失的变化提供上界，填补了文献中的理论空白。随后，我们将低秩模型压缩表述为双目标优化问题，并证明单一的统一容差即可产生代理帕累托最优的异构秩。基于我们的理论洞察，我们提出了帕累托引导奇异值分解（PGSVD），这是一个零样本流水线，通过帕累托引导的秩选择和交替最小二乘实现来改进激活感知压缩。我们将PGSVD应用于LLM和VLM，在相同压缩水平下展现出更高的准确率以及推理加速。

    arXiv:2510.05544v3 Announce Type: replace  Abstract: Large language models (LLM) and vision-language models (VLM) have achieved state-of-the-art performance, but they impose significant memory and computing challenges in deployment. We present a novel low-rank compression framework to address this challenge. First, we upper bound the change of network loss via layer-wise activation-based compression errors, filling a theoretical gap in the literature. We then formulate low-rank model compression as a bi-objective optimization and prove that a single uniform tolerance yields surrogate Pareto-optimal heterogeneous ranks. Based on our theoretical insights, we propose Pareto-Guided Singular Value Decomposition (PGSVD), a zero-shot pipeline that improves activation-aware compression via Pareto-guided rank selection and alternating least-squares implementation. We apply PGSVD to both LLM and VLM, showing better accuracy at the same compression levels and inference speedup.
    
[^500]: HomID：在具有各向异性嵌入的同质流形上对内在维度估计器进行基准测试

    HomID : Benchmarking Intrinsic Dimension Estimators on Homogenous Manifolds with Anisotropic Embeddings

    [https://arxiv.org/abs/2510.01335](https://arxiv.org/abs/2510.01335)

    本文提出了 HomID 基准——一组具有各向异性嵌入的同质流形集合，用以揭示现有内在维度估计器在标准基准上表现良好、但在各向异性条件下会系统性失效，并证明了各向异性畸变通过使其所依赖的分布发生系统性偏移而导致估计误差。

    

    流形假说认为，数据位于内在维度低于其环境维度的流形之上。然而，对于现实数据集，不同内在维度估计器给出的估计值之间在经验上并不一致。因此，用有针对性的压力测试来检验内在维度估计器是十分重要的。在这项工作中，我们考察了各向异性的作用。为此，我们提出了 HomID——一组具有各向异性嵌入的同质空间集合，用于对内在维度估计器进行基准测试。我们观察到，在标准基准测试中表现良好的方法，在相同的资源分配下，在 HomID 上会系统性地出现性能下降。我们进一步观察到，对这类基准进行各向异性畸变同样会导致性能退化。最后，我们证明，受控的各向异性畸变会在这些方法所依赖的分布上引起系统性偏移，从而为由此产生的估计误差提供了一个具体的机制。

    arXiv:2510.01335v2 Announce Type: replace  Abstract: The manifold hypothesis suggests that data lies on manifolds with smaller intrinsic dimension (ID) than their ambient dimension. However there is no empirical agreement on the estimates for ID from different estimators for realistic datasets. Thus it is important to test ID estimators (IDEs) with targeted stressors. In this work, we consider the role of anisotropy. To this end, we propose HomID, a collection of homogeneous spaces with anisotropic embedding, for benchmarking ID estimators. We observe that methods that perform well on standard benchmarks systematically degrade on HomID under identical resource allocation. We further observe that anisotropic distortion of such benchmarks also results in performance degradation. Finally, we demonstrate that controlled anisotropic distortions induce systematic shifts in the distributions on which these methods rely, providing a concrete mechanism for the resulting estimation errors in two
    
[^501]: 单调算子平衡网络的电路实现与硬件线性化

    Circuit realization and hardware linearization of monotone operator equilibrium networks

    [https://arxiv.org/abs/2509.13793](https://arxiv.org/abs/2509.13793)

    该论文证明电阻-二极管电路的端口行为等价于ReLU单调算子平衡网络，并提出硬件线性化方法使梯度可直接在硬件中计算，实现了神经网络在模拟硬件中的构建与训练。

    

    本文证明了电阻-二极管网络的端口行为对应于ReLU单调算子平衡网络（无限深度极限下的神经网络）的解，从而在模拟硬件中实现了神经网络的简洁构造。我们进一步证明，通过一种称为硬件线性化的方法，该电路的梯度可以直接在硬件中计算，这使得网络能够在硬件中训练，我们通过器件级电路仿真验证了这一点。我们将结果扩展到电阻-二极管网络的级联结构，可用于实现前馈网络及其他非对称网络。最后，我们证明不同的非线性元件会产生不同的激活函数，并提出了一种由非理想二极管模型诱导的新型二极管ReLU激活函数。

    arXiv:2509.13793v3 Announce Type: replace-cross  Abstract: It is shown that the port behavior of a resistor-diode network corresponds to the solution of a ReLU monotone operator equilibrium network (a neural network in the limit of infinite depth), giving a parsimonious construction of a neural network in analog hardware. We furthermore show that the gradient of such a circuit can be computed directly in hardware, using a procedure we call hardware linearization. This allows the network to be trained in hardware, which we demonstrate with a device-level circuit simulation. We extend the results to cascades of resistor-diode networks, which can be used to implement feedforward and other asymmetric networks. We finally show that different nonlinear elements give rise to different activation functions, and introduce the novel diode ReLU which is induced by a non-ideal diode model.
    
[^502]: 动量梯度下降的修正损失：精细分析

    Modified Loss of Momentum Gradient Descent: Fine-Grained Analysis

    [https://arxiv.org/abs/2509.08483](https://arxiv.org/abs/2509.08483)

    该论文证明当步长足够小时，重球动量梯度下降在指数吸引的不变流形上精确等价于带修正损失的普通梯度下降，能以任意有限阶精度刻画该修正损失，并在其无记忆近似的组合结构中发现了介于欧拉多项式与Narayana多项式之间的一类新的β多项式族。

    

    我们分析了带有Polyak (1964) 重球动量（HB）的梯度下降算法，其固定动量超参数 β ∈ (0, 1) 提供了记忆的指数衰减。基于 Kovachki 和 Stuart (2021) 的工作，我们证明当步长 h 足够小时，该算法在指数吸引的不变流形上恰好等价于带有修正损失的普通梯度下降。尽管该修正损失不存在闭式表达式，我们对其进行了描述，对于任意有限阶 R 误差为 O(h^R)，并证明了全局（有限“时间”范围）轨迹逼近界 O(h^R)。随后，我们对 HB 无记忆近似背后的组合数学进行了精细分析，特别是发现了隐藏其中的一类丰富的关于 β 的多项式族，它们包含欧拉多项式和 Narayana 多项式，且在系数意义上介于二者之间。我们证明这些多项式……（摘要在此处被截断）

    arXiv:2509.08483v2 Announce Type: replace  Abstract: We analyze gradient descent with Polyak (1964) heavy-ball momentum (HB) whose fixed momentum hyperparameter $\beta \in (0, 1)$ provides exponential decay of memory. Building on Kovachki and Stuart (2021), we prove that on an exponentially attractive invariant manifold the algorithm is exactly plain gradient descent with a modified loss, provided that the step size $h$ is small enough. Although the modified loss does not admit a closed-form expression, we describe it up to $O(h^{\mathcal{R}})$-errors for arbitrary finite order $\mathcal{R}$, and prove global (finite "time" horizon) trajectory approximation bounds $O(h^{\mathcal{R}})$. We then conduct a fine-grained analysis of the combinatorics underlying the memoryless approximations of HB, in particular, finding a rich family of polynomials in $\beta$ hidden inside which include and lie coefficient-wise in between Eulerian and Narayana polynomials. We prove that these polynomials ar
    
[^503]: 扩展法律人工智能：面向法定条文分类与判例法检索的Mamba与Transformer基准测试

    Scaling Legal AI: Benchmarking Mamba and Transformers for Statutory Classification and Case Law Retrieval

    [https://arxiv.org/abs/2509.00141](https://arxiv.org/abs/2509.00141)

    该论文构建了一个初步基准测试，比较Mamba等线性复杂度状态空间模型与BERT、DeBERTa、Longformer等Transformer模型在四个法律分类任务和两个判例检索任务上的表现，发现最强的状态空间模型性能与Transformer相差约1.3个百分点以内，证明了SSM作为处理长篇法律文档的有前景替代方案的潜力。

    

    法定条文语料库和司法判决的增长速度已超过法律专业人士的阅读能力，而单个判决往往超出标准编码器模型的上下文限制。Transformer架构在法律自然语言处理基准测试中占据主导地位，但其二次方复杂度的注意力机制使得需要对需要整篇文档推理的文档进行截断或碎片化处理。选择性状态空间模型（SSM），如Mamba，提供线性时间的序列建模能力，是处理长篇法律文档的一种有前景的替代方案，但其在法律分类与检索任务上的性能仍未得到充分探索。我们提出了一项初步基准测试，在共享的窗口化与聚合流程下，将Mamba和SSD-Mamba与BERT、DeBERTa和Longformer在四个法律分类任务（ECtHR、EUR-Lex、SCOTUS和ILDC/ILC）以及两个判例检索任务（ECtHR和ILDC）上进行比较。表现最强的SSM性能与最佳模型相差约在1.3个百分点以内。

    arXiv:2509.00141v2 Announce Type: replace-cross  Abstract: Statutory corpora and judicial decisions are growing faster than legal professionals can read them, while individual judgments often exceed the context limits of standard encoder models. Transformer architectures dominate legal NLP benchmarks, but their quadratic attention complexity can require truncating or fragmenting documents that demand whole-document reasoning. Selective state-space models (SSMs), such as Mamba, offer linear-time sequence modeling and are a promising alternative for long legal documents, yet their performance on legal classification and retrieval remains underexplored.   We present a preliminary benchmark comparing Mamba and SSD-Mamba with BERT, DeBERTa, and Longformer across four legal classification tasks (ECtHR, EUR-Lex, SCOTUS, and ILDC/ILC) and two case-retrieval tasks (ECtHR and ILDC), using a shared windowing and aggregation pipeline. The strongest SSM performs within approximately 1.3 percentage 
    
[^504]: 动态科学中的坚持悖论：来自深度学习革命的证据

    Persistence Paradox in Dynamic Science: Evidence from the Deep Learning Revolution

    [https://arxiv.org/abs/2506.22729](https://arxiv.org/abs/2506.22729)

    本研究以2012年AlexNet引发的深度学习革命为背景，通过分析5000多名机器学习科学家20年的职业轨迹，揭示了坚持的悖论：以往成功或隶属成熟团队的科学家在范式转变中适应更慢，坚持虽能维持生产力，却在2012年后阻碍了其科学成功。

    

    坚持通常被视为科学中的一种美德。然而，本文挑战了这一传统观点，强调坚持的情境性本质，特别是在范式转变期间坚持如何可能成为一种负担。我们聚焦于2012年由AlexNet引发的深度学习革命。通过分析在此前十年中活跃于顶级机器学习会议的5000多名科学家长达20年的职业轨迹，我们考察了他们的研究重点和产出如何演变。我们首先揭示了一个动态时期：在此期间，顶级会议日益优先关注前沿的深度学习发展，取代了传统的统计学习方法。科学家们以截然不同的方式应对这些变化：那些此前成功或隶属于成熟团队的科学家适应得更慢。这种坚持与生产力呈正相关，但在2012年之后与科学成功呈负相关。

    arXiv:2506.22729v3 Announce Type: replace-cross  Abstract: Persistence is often regarded as a virtue in science. In this paper, however, we challenge this conventional view by highlighting its contextual nature, particularly how persistence can become a liability during paradigm shifts. We focus on the deep learning revolution catalyzed by AlexNet in 2012. Analyzing the 20-year career trajectories of more than 5,000 scientists active in top machine learning venues during the preceding decade, we examine how their research focus and output evolved. We first uncover a dynamic period in which leading venues increasingly prioritized cutting-edge deep learning developments, displacing traditional statistical learning methods. Scientists responded to these changes in markedly different ways: those who were previously successful or affiliated with established teams adapted more slowly. Such persistence is positively associated with productivity but, after 2012, negatively associated with scie
    
[^505]: 因果后验估计

    Causal Posterior Estimation

    [https://arxiv.org/abs/2505.21468](https://arxiv.org/abs/2505.21468)

    提出因果后验估计（CPE）方法，将模型图结构中的条件依赖关系直接硬编码进基于流匹配的神经网络架构，在似然函数难以计算的模拟器模型中实现高精度的贝叶斯后验推断。

    

    我们提出了因果后验估计，这是一种用于模拟器模型贝叶斯推断的新方法，适用于似然函数难以求解或计算成本高昂、但根据给定参数值生成输出较为简单的场景。CPE利用流匹配来近似后验分布，同时将模型图结构所诱导的条件依赖关系直接融入神经网络架构中。通过大量实验，我们证明了将这些条件依赖关系硬编码到网络中（而非要求从数据中学习它们），使CPE能够实现高度精确的后验推断，其性能达到甚至超越当前最先进的基线方法。

    arXiv:2505.21468v2 Announce Type: replace  Abstract: We present Causal Posterior Estimation (CPE), a novel method for Bayesian inference in simulator models, where evaluating the likelihood function is intractable or computationally expensive, but generating outputs given parameter values is straightforward. CPE approximates the posterior distribution using flow matching while directly incorporating the conditional dependence structure induced by the model's graphical representation into the neural network architecture. Across extensive experiments, we demonstrate that hard-coding these conditional dependencies into the network, rather than requiring them to be learned from data, enables CPE to achieve highly accurate posterior inference that matches or outperforms state-of-the-art baselines.
    
[^506]: APE：基于接受标准的语言模型适配选择性微调方法

    APE: Selective Fine-tuning with Acceptance Criteria for Language Model Adaptation

    [https://arxiv.org/abs/2505.19912](https://arxiv.org/abs/2505.19912)

    APE 是一种受进化优化启发的选择性微调方法，通过在小数据子集上评估多个候选参数更新并仅接受超过性能阈值者，在保持模型稳定性的同时以极少计算资源实现大型语言模型的高效适配。

    

    我们提出了邻近可能探索（APE），这是一种用于适配大型语言模型的选择性微调方法，它系统地探索参数修改，同时保持模型的稳定性。受进化优化原理的启发，APE 通过在小数据子集上进行微调来评估多个候选参数更新，并仅接受超过性能阈值的更新。与遵循单一梯度方向的标准微调不同，APE 实现了一种过滤选择过程，在实现系统性改进的同时，防止了破坏稳定性的参数变化。我们的方法在新闻摘要任务上以极少的计算资源实现了 33.9% 的 BLEU 提升和 36.2% 的困惑度降低。该方法为受控模型适配提供了一个实用框架，在性能提升与表征稳定性之间取得了平衡。

    arXiv:2505.19912v3 Announce Type: replace  Abstract: We present Adjacent Possible Exploration (APE), a selective fine-tuning method for adapting large language models that systematically explores parameter modifications while maintaining model stability. Inspired by evolutionary optimization principles, APE evaluates multiple candidate parameter updates through fine-tuning on small data subsets and accepts only those exceeding a performance threshold. Unlike standard fine-tuning that follows single gradient directions, APE implements a filtered selection process that prevents destabilizing parameter changes while enabling systematic improvement. Our method achieves 33.9\% BLEU improvement and 36.2\% perplexity reduction on news summarization tasks while using minimal computational resources. The approach provides a practical framework for controlled model adaptation that balances performance gains with representational stability.
    
[^507]: 基于图的楼层分离：利用节点嵌入与WiFi轨迹聚类

    Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories

    [https://arxiv.org/abs/2505.08088](https://arxiv.org/abs/2505.08088)

    该论文提出了一种基于图的楼层分离新方法，通过将Wi-Fi指纹轨迹构建为图并利用Node2Vec嵌入和K-means聚类识别楼层，在华为大学挑战赛2021数据集上取得了优于传统社区发现算法的效果，并公开了数据集与代码。

    

    室内定位系统（IPS）在复杂的多层建筑环境中对基于位置的服务日益重要。本研究提出了一种新颖的基于图的方法，利用Wi-Fi指纹轨迹进行楼层分离，解决了室内环境中垂直定位的难题。我们构建了一个图，其中节点表示Wi-Fi指纹，边则根据信号相似度和上下文转换进行加权。采用Node2Vec生成低维嵌入，随后使用K-means聚类来识别不同楼层。在华为大学挑战赛2021数据集上的评估表明，我们的方法优于传统的社区发现算法，达到了68.97%的准确率、61.99%的F1分数和57.19%的调整兰德指数（ARI）。通过公开发布预处理数据集和实现代码，这项工作有助于推动室内定位研究的发展。

    arXiv:2505.08088v5 Announce Type: replace-cross  Abstract: Indoor positioning systems (IPSs) are increasingly vital for location-based services in complex multi-storey environments. This study proposes a novel graph-based approach for floor separation using Wi-Fi fingerprint trajectories, addressing the challenge of vertical localization in indoor settings. We construct a graph where nodes represent Wi-Fi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec is employed to generate low-dimensional embeddings, which are subsequently clustered using K-means to identify distinct floors. Evaluated on the Huawei University Challenge 2021 dataset, our method outperforms traditional community detection algorithms, achieving an accuracy of 68.97%, an F1- score of 61.99%, and an Adjusted Rand Index of 57.19%. By publicly releasing the preprocessed dataset and implementation code, this work contributes to advancing research in indoor positioning. The prop
    
[^508]: 分布内与分布外机器遗忘的效用与复杂度

    The Utility and Complexity of in- and out-of-Distribution Machine Unlearning

    [https://arxiv.org/abs/2412.09119](https://arxiv.org/abs/2412.09119)

    本文对机器遗忘进行了严格的复杂度与效用分析，证明带输出扰动的经验风险最小化对分布内遗忘数据能实现紧密的权衡，而分布外遗忘数据则面临根本性挑战。

    

    机器遗忘，即从已训练模型中选择性移除数据的过程，对于解决模型部署后的隐私问题和知识缺口日益重要。尽管其重要性突出，现有方法往往是启发式的，缺乏形式化保证。本文分析了近似遗忘在效用、时间和空间复杂度之间的基本权衡，并提供了类似于差分隐私的严格认证。对于分布内遗忘数据（即与保留集相似的数据），我们证明了一个出奇简单且通用的方法——带输出扰动的经验风险最小化——能够实现紧密的遗忘-效用-复杂度权衡，填补了之前关于差分隐私所“免费”实现遗忘之间分离的理论空白，因为差分隐私本质上促进了对这类数据的移除。然而，此类技术在处理分布外遗忘数据时会失效——即与保留集显著不同的数据……

    arXiv:2412.09119v4 Announce Type: replace  Abstract: Machine unlearning, the process of selectively removing data from trained models, is increasingly crucial for addressing privacy concerns and knowledge gaps post-deployment. Despite this importance, existing approaches are often heuristic and lack formal guarantees. In this paper, we analyze the fundamental utility, time, and space complexity trade-offs of approximate unlearning, providing rigorous certification analogous to differential privacy. For in-distribution forget data -- data similar to the retain set -- we show that a surprisingly simple and general procedure, empirical risk minimization with output perturbation, achieves tight unlearning-utility-complexity trade-offs, addressing a previous theoretical gap on the separation from unlearning "for free" via differential privacy, which inherently facilitates the removal of such data. However, such techniques fail with out-of-distribution forget data -- data significantly diffe
    
[^509]: 方差感知UCB策略下的分配稳定性与Wald推断

    Allocation Stability and Wald Inference under Variance-Aware UCB

    [https://arxiv.org/abs/2412.08843](https://arxiv.org/abs/2412.08843)

    本文为双臂方差感知UCB策略给出了最优臂分配稳定性的尖锐判据，并证明即使最优臂计数不稳定，只要拉取次数与奖励方差的乘积依概率发散，臂均值线性组合的Wald统计量仍渐近服从标准正态分布，从而表明分配稳定性并非高斯推断的必要条件。

    

    分配稳定性常被用来论证基于老虎机数据进行高斯推断的合理性，但它何时才是必要的？本文针对双臂、固定时域、奖励分布有界且可能随时域变化的方差感知UCB策略研究了这一问题。我们找到了一个由奖励差距和方差刻画的尖锐判据，该判据决定了最优臂的拉取次数能否被具有消失相对误差的确定性量近似，而次优臂的计数总是稳定的。尽管最优臂计数可能不稳定，我们证明：只要每个臂的拉取次数与奖励方差的乘积依概率发散，对于任意固定的非零系数向量，臂均值线性组合的通常Wald统计量都具有标准正态极限。然而在同一条件下，这种高斯近似能否在所有确定性非零系数向量上一致成立，取决于……（摘要在此处截断）

    arXiv:2412.08843v3 Announce Type: replace-cross  Abstract: Allocation stability is often used to justify Gaussian inference from bandit data, but when is it necessary? In this paper, we address this question for a two-armed, fixed-horizon variance-aware UCB policy with bounded reward distributions that may vary with the horizon. We find a sharp criterion in terms of the reward gap and variances that determines whether the optimal-arm count admits a deterministic approximation with vanishing relative error, while the suboptimal-arm count is always stable. Despite the possible instability of the optimal-arm count, we show that the ordinary Wald statistic for a linear combination of the arm means has a standard normal limit for every fixed nonzero coefficient vector, provided the product of the pull count and reward variance diverges in probability for each arm. Under the same condition, however, this Gaussian approximation holds uniformly over deterministic nonzero coefficient vectors if
    
[^510]: 在循环状态中学习：基于线性循环网络的梯度下降

    Learning in the Recurrent State: Gradient Descent with Linear Recurrent Networks

    [https://arxiv.org/abs/2410.11687](https://arxiv.org/abs/2410.11687)

    本文提出了GRIL——一种对角线性循环网络，通过将梯度步骤分解为短窗口叉积写入和乘法读出，使线性循环网络能够在循环状态中于单次正向传播内实现上下文梯度下降，从而将基于梯度的上下文学习能力扩展到线性时间复杂度的序列模型。

    

    上下文学习使序列模型能够根据输入中的示例适应新任务。一系列重要研究表明，可以构建自注意力机制，在正向传播过程中对拟合上下文示例的线性预测器实现梯度下降。状态空间模型（SSMs）及其他线性循环网络（LRNNs）能以线性时间成本对序列进行建模，但目前尚不清楚它们的循环更新如何执行同样的上下文内梯度下降。我们提出了基于梯度的循环上下文学习器（GRIL），这是一种对角线性循环网络，它将监督梯度步骤分解为短窗口叉积写入操作和针对下一次查询的乘法读出操作。对于线性回归，这一构造能在矩阵状态中累积上下文梯度，并在单次正向传播中应用该梯度，且仅需 $O(f^2)$ 个可学习的自由度。同样的设计还可以扩展到多步更新和交叉熵分类……

    arXiv:2410.11687v4 Announce Type: replace  Abstract: In-context learning lets a sequence model adapt to a new task from examples in its input. A prominent line of work shows how self-attention can be constructed to implement gradient descent on a linear predictor fit to the in-context examples during the forward pass. State-space models (SSMs) and other linear recurrent networks (LRNNs) model sequences at linear time cost, but it is unclear how their recurrent update could carry out the same in-context gradient descent. We introduce Gradient-based Recurrent In-context Learner (GRIL), a diagonal LRNN that factorizes a supervised gradient step into a short-window cross-product write and a multiplicative readout of the next query. For linear regression, this construction accumulates the context gradient in a matrix state and applies it in a single forward pass, with $O(f^2)$ learned degrees of freedom. The same design extends to multi-step updates and cross-entropy classification, with a 
    
[^511]: 基于需求的测试：利用博弈论增强强化学习

    Requirement-Based Testing: Enhancing Reinforcement Learning with Game Theory

    [https://arxiv.org/abs/2407.18994](https://arxiv.org/abs/2407.18994)

    本文提出一种基于博弈论的启发式蒙特卡洛树搜索方法，用于从自动机形式的功能需求中自动生成黑盒测试用例，实验表明该启发式方法加速了算法收敛并提升了测试性能。

    

    我们考虑为反应式实现从以自动机形式指定的功能需求中自动在线合成黑盒测试用例。测试者的目标是到达某个给定状态以满足覆盖准则，同时监控需求是否被违反。我们开发了一种基于蒙特卡洛树搜索的方法，这是强化学习中一种用于高效选择有前景输入的经典技术。我们将自动机需求视为实现与测试者之间的博弈，通过将搜索偏向在该博弈中有前景的输入，设计了一种启发式方法。我们通过实验证明，该启发式方法加速了蒙特卡洛树搜索算法的收敛，从而提升了测试的性能。

    arXiv:2407.18994v2 Announce Type: replace-cross  Abstract: We consider the automatic online synthesis of black-box test cases from functional requirements specified as automata for reactive implementations. The goal of the tester is to reach some given state, so as to satisfy a coverage criterion, while monitoring the violation of the requirements. We develop an approach based on Monte Carlo Tree Search, which is a classical technique in reinforcement learning for efficiently selecting promising inputs. Seeing the automata requirements as a game between the implementation and the tester, we develop a heuristic by biasing the search towards inputs that are promising in this game. We experimentally show that our heuristic accelerates the convergence of the Monte Carlo Tree Search algorithm, thus improving the performance of testing.
    
[^512]: 基于CNN的语音情感识别及其在数字医疗中的应用案例

    Speech Emotion Recognition Using CNN and Its Use Case in Digital Healthcare

    [https://arxiv.org/abs/2406.10741](https://arxiv.org/abs/2406.10741)

    本研究利用卷积神经网络（CNN）从语音音频中识别并标注情感，通过精确率、召回率和F1分数进行评估，展现了其在数字医疗领域的应用价值。

    

    从语音中识别人类情感和情绪状态的过程被称为语音情感识别（SER）。其依据是语音中的语调和音高往往能够传达出潜在的情感。语音识别包含情感识别能力，这一技术正日益受到欢迎且需求旺盛。借助数据中发现的适当因素（如模态、情感、强度、重复等），本研究旨在利用卷积神经网络（CNN）从音频记录中区分情感，并根据不同的情感范围对其进行标注。我借助机器学习方法开发了一个机器学习模型，用于从输入的音频文件中识别情感。评估主要集中于精确率、召回率和F1分数等常见的机器学习指标，以正确搭建和训练机器学习框架。

    arXiv:2406.10741v2 Announce Type: replace-cross  Abstract: The process of identifying human emotion and affective states from speech is known as speech emotion recognition (SER). This is based on the observation that tone and pitch in the voice frequently convey underlying emotion. Speech recognition includes the ability to recognize emotions, which is becoming increasingly popular and in high demand. With the help of appropriate factors (such modalities, emotions, intensities, repetitions, etc.) found in the data, my research seeks to use the Convolutional Neural Network (CNN) to distinguish emotions from audio recordings and label them in accordance with the range of different emotions. I have developed a machine learning model to identify emotions from supplied audio files with the aid of machine learning methods. The evaluation is mostly focused on precision, recall, and F1 score, which are common machine learning metrics. To properly set up and train the machine learning framework
    

