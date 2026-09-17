# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Zeroth-Order Paradigm for LLM Preference Alignment](https://arxiv.org/abs/2609.19144) | 本文提出了基于比较预言机的零阶偏好对齐方法 ComPO，能从微小似然边距的偏好对中提取方向性信息而无需优化可微偏好损失，并建立了收敛保证且引入了具备反向 KL 控制的在线版本。 |
| [^2] | [Exponential Hardness of Off-Policy Evaluation under History-Dependent Logging](https://arxiv.org/abs/2609.19135) | 本文证明当日志记录器依赖历史时，即使所有覆盖条件的常数与时间跨度无关，离线策略评估仍需要指数级数量的日志轨迹，其根源在于重置操作会擦除决定目标策略价值的关键未知转移信息。 |
| [^3] | [Cognitive Extensions for Dual-Process Language Agents: Memory and Self-Reflection in Interactive Environments](https://arxiv.org/abs/2609.19128) | 通过为双过程代理SwiftSage引入自适应记忆模块（AMM）和自我反思模块（SRM）两个认知扩展，显著提升了语言代理在ScienceWorld交互环境中的表现，完整系统实现了最佳平均得分64.62、成功率43.17%，其中SRM是最强的独立贡献者。 |
| [^4] | [How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents](https://arxiv.org/abs/2609.19107) | 该研究证明架构干预（尤其是模型增长和递归深度）可以改变预训练的缩放指数，使7.4B模型增长架构以约20倍更少的计算量匹配GPT-3 13B的性能，且计算效率优势随规模扩大而增加。 |
| [^5] | [Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations](https://arxiv.org/abs/2609.19101) | 该论文发现奖励作弊行为会在大语言模型内部表示中留下一致且可解释的签名，利用简单的均值差向量即可在常见基准测试中可靠地检测并系统发现模型表现出的各类奖励作弊行为。 |
| [^6] | [Evidence-Grounded Agentic Formulation Development in an Autonomous Laboratory](https://arxiv.org/abs/2609.19099) | Andromeda 2是一个能够在自主实验室中基于结构化实验证据进行推理并调用计算与实验工具的智能体系统，在紫杉醇SEDDS制剂开发中以相同预算实现了50%的高性能命中率，远超概率优化模型（17%）和实验设计方法（2%）。 |
| [^7] | [A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings](https://arxiv.org/abs/2609.19083) | 提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。 |
| [^8] | [Probabilistic Linear Explanations](https://arxiv.org/abs/2609.19077) | 该论文提出了一个基于稀疏锚定线性模型的概率可解释性统一框架，适用于二分类与连续回归，通过在布尔超立方体上的映射严格推广了基于子集的解释方法，并证明在神经网络模型下最小化相关性误差是难解的，进而将其与可处理的保真度误差代理目标相关联。 |
| [^9] | [Double descent is the principle of least action](https://arxiv.org/abs/2609.19076) | 本文用统计力学解释了机器学习中的双重下降现象：将随机梯度训练视为温度为 $T$ 的粒子在损失能量景观上的扩散，有限时间的扩散带来有效权重衰减，使每个参数成为二次自由度，从而由能量均分定理导出测试误差随参数数量先升后降的规律。 |
| [^10] | [RLLBC-Lib: An Educational Code Library for Reinforcement Learning and Learning-Based Control](https://arxiv.org/abs/2609.19074) | 该论文介绍了RLLBC-Lib，一个精心设计的教学代码库，通过统一的表格型与深度强化学习实现以及核心原理示例，降低学生在基于学习的控制领域学习强化学习的入门门槛。 |
| [^11] | [Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators](https://arxiv.org/abs/2609.19072) | 该论文提出统一基准Safety-Flag，将七个安全审核基准整合到同一“标记/不标记”协议下，从错误方向、概率校准和置信度错误排序三个维度评估LLM内容审核模型，发现总体准确率掩盖了模型间截然不同的错误模式，且所有通用模型都普遍过于自信。 |
| [^12] | [LightSleepX: A Lightweight, Inception-Based Dual-Modal Network for Sleep Staging](https://arxiv.org/abs/2609.19062) | LightSleepX通过结合Inception风格架构、深度可分离卷积、多尺度增强注意力与Mamba编码器，构建了一个轻量级双模态网络，在资源受限环境中实现了高精度且计算高效的自动睡眠分期。 |
| [^13] | [Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery](https://arxiv.org/abs/2609.19048) | 该论文提出了一种将基于AGV的智能仓储作业与末端多式联运进行集成优化的深度强化学习框架，通过设计多目标算法（如MORM-AGDQN）实现高吞吐量连续订单调度，并提升差异化按需配送系统的灵敏度与适应性。 |
| [^14] | [LSR-Net: Learning the Forward Evolution Operator for Nonlinear Fluid Dynamics](https://arxiv.org/abs/2609.19039) | LSR-Net提出了一种新型神经算子架构，将可学习积分核分解为长程分量（基于指数和表示的可训练傅里叶乘子）与短程分量（标准卷积），仅凭状态快照对即可学习动力系统的前向演化算子，以O(n log n)的复杂度高效预测非线性流体动力学。 |
| [^15] | [TwinMark: A Unified Watermark for Provable Survival Under Feature and Logit Distillation](https://arxiv.org/abs/2609.19011) | 提出统一水印方案TwinMark，通过协方差投影（cov-Feat）与类条件Fisher对齐线性载体（cc-FALC）两条互补读取通道，分别针对KL知识蒸馏和特征匹配蒸馏攻击提供可由教师端验证的检测能力下界证书，从而可证明地保障水印在模型被蒸馏后仍能存活并被检测。 |
| [^16] | [Tabular Deep Learning vs Classical Machine Learning for Urban Land Cover Classification](https://arxiv.org/abs/2609.19010) | 该研究在一个统一可复现的流程中，系统对比了经典机器学习模型与表格深度学习模型在城市土地覆盖分类任务上的表现，并采用加权交叉熵损失应对类别不平衡问题。 |
| [^17] | [CompileRover: Revolutionizing Virtual Machine Compiler Optimization with a Tri-Role LLM-Driven Framework](https://arxiv.org/abs/2609.19004) | CompileRover是一个基于LLM的三角色协作优化框架，通过裁判、顾问和操作者的协同机制，结合控制流分析、代码结构转换和动态执行模式识别，有效解决了虚拟机编译器输出中的冗余计算和低效循环等性能瓶颈问题。 |
| [^18] | [Capability Emergence Can Be Forecast: Per-Seed, In Advance, With Calibrated Intervals, Certified False Alarms, and a Blind Pre-Registered Gate](https://arxiv.org/abs/2609.19000) | 该论文首次以严格的预测评分标准（受控误报率、校准区间、阴性样本、盲测预注册门槛）证明了能力涌现时机可以被按种子逐个、提前、带校准不确定性地预测，例如通过前一个token头的形成时间以0.977的Spearman相关性提前约15%训练步数预测归纳头的涌现。 |
| [^19] | [One Axis, No Brake: Self-Knowledge Limits the Filtering of Harmful Peer Conformity in LLMs](https://arxiv.org/abs/2609.18998) | 该论文证明了在多智能体LLM中过滤有害同伴从众的“刹车”本质上等价于伪装的正确性探测器，因此受限于模型不完美的自我认知（AUROC仅0.64–0.89），这一“墙”即使用白盒引导也无法突破。 |
| [^20] | [Suppressed, Not Erased: A Representational Trace of Edited Facts Survives Even Weight-Free Knowledge Editing](https://arxiv.org/abs/2609.18985) | 该论文发现即使知识编辑在生成层面“成功”完成，甚至包括完全不修改模型权重的外部记忆编辑方法，被编辑掉的原始事实仍然可以以远高于随机水平的准确率从模型隐藏状态中线性解码出来，表明知识编辑只是“抑制”而非真正“抹除”了原始知识。 |
| [^21] | [Instrument Classification of Solo Sheet Music Images](https://arxiv.org/abs/2609.18980) | 本文提出将乐谱图像基于bootleg score表示法转换为音乐“词语”序列，并通过无监督预训练语言模型再微调的方法实现独奏乐谱图像的乐器分类，使RoBERTa的分类准确率从34.5%提升至42.9%。 |
| [^22] | [The Automaton Underneath: The Additive Input Pathway Is a Parasitic Attractor for State Tracking in Householder Linear RNN](https://arxiv.org/abs/2609.18966) | 通过因果消融发现，移除Householder线性RNN中的加性输入注入项能让模型学会精确的状态追踪自动机并实现长度泛化，而保留该项则会充当“寄生吸引子”导致分布外性能崩溃。 |
| [^23] | [FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning](https://arxiv.org/abs/2609.18964) | 该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。 |
| [^24] | [Changepoint-Aware World Models: Detecting Dynamics Shifts and Recovering by Forgetting Stale Replay in Model-Based RL](https://arxiv.org/abs/2609.18950) | 提出变点感知世界模型（CAWM），利用在线CUSUM检验从内部预测误差中检测动力学突变，并通过遗忘陈旧经验回放实现快速恢复，显著优于被动重训练和重建动力学模型的基线方法。 |
| [^25] | [StableEval Arena: A Cost-Aware Agentic Benchmark for Stablecoin Price Stability Prediction](https://arxiv.org/abs/2609.18949) | StableEval Arena是一个成本感知的智能体基准框架，通过无泄漏历史回放评估基于LLM的智能体系统在稳定币锚定风险预测上的表现，并将可信度定义为预测质量、运营可靠性与计算成本的联合属性。 |
| [^26] | [Social Laws for Multi-agent Coordination in Stochastic Environments](https://arxiv.org/abs/2609.18929) | 本文将社会法则概念扩展至随机的、基于奖励的多智能体环境，提出α-鲁棒性度量，并通过归约为求解一系列马尔可夫决策过程来实现社会法则的稳健性验证。 |
| [^27] | [Comprehensive reconstruction of collider events with hypergraph representation learning and graph-conditioned diffusion](https://arxiv.org/abs/2609.18928) | 提出了VyPER框架，将对撞机事件表示为物理启发的超图，通过结合超边监督分类与图条件扩散模型，在统一框架内同时完成粒子分配和中微子运动学预测，实现对撞机事件的全面重建。 |
| [^28] | [Higher-order pruning of experts in mixture-of-experts language models](https://arxiv.org/abs/2609.18916) | 提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。 |
| [^29] | [Fast Learning Rates for Physics-Informed Kernel Methods](https://arxiv.org/abs/2609.18901) | 本文为结合数值观测与微分观测的物理信息核估计器证明了有限样本误差界，揭示了预测误差的双区间结构：当微分观测有限时，误差速率同时依赖于数值与微分观测的数量，而当微分观测数量超过阈值后，误差速率达到饱和并与完美物理约束下的最优速率相匹配。 |
| [^30] | [Learning Lyapunov Operators for Nonlinear Systems](https://arxiv.org/abs/2609.18894) | 本文首次证明了李雅普诺夫解算子在指数稳定性假设下具有良定义性、唯一性和连续性，为在整个非线性系统族上统一学习李雅普诺夫函数奠定了理论基础。 |
| [^31] | [NeuroECG: ECGFounder-Based Deep ECG Representation for EEG-Free Neurological Prognostication After Cardiac Arrest](https://arxiv.org/abs/2609.18891) | 该研究提出NeuroECG框架，通过渐进式解冻微调预训练心电图基础模型ECGFounder，并利用分位数池化与PCA压缩深度特征，实现仅凭低成本床旁心电图即可对心脏骤停患者进行无需脑电图的神经功能预后预测。 |
| [^32] | [Preventing Model Collapse: A Fisher-Rao Perspective on the Dynamics of Training with Synthetic Data](https://arxiv.org/abs/2609.18878) | 本文从Fisher-Rao信息几何的视角，为防止模型坍塌所需的最低人类数据比例建立了严格的理论保证，克服了以往基于欧几里得度量的分析在高维情况下下界失效的问题。 |
| [^33] | [Physics-based prediction, uncertainty quantification and decision-making for IN718 crystallographic texture intensity across LPBF defocus regimes](https://arxiv.org/abs/2609.18863) | 本研究开发了一个两阶段物理模型用于预测LPBF工艺中IN718的<001>晶体织构强度，通过k近邻残差校正、面束能量密度判据和保形区间实现不确定性量化，从而在物理有效范围内支持可靠的工艺决策。 |
| [^34] | [Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection](https://arxiv.org/abs/2609.18860) | 研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。 |
| [^35] | [Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data](https://arxiv.org/abs/2609.18842) | 本文提出一种能将实时交互数据直接写入自身权重的大语言模型新架构，突破了传统模型权重冻结、只能依赖静态预训练数据和临时提示词的局限，使模型能够从用户实时提供的知识和更正中持续学习。 |
| [^36] | [Interpretable Multi-Instance Learning Enables Early Prediction of Key Molecular Alterations from Routine Flow Cytometry in Acute Myeloid Leukemia](https://arxiv.org/abs/2609.18825) | 本研究开发了一种基于决策树的可解释多示例学习模型，能利用入院数小时内完成的常规流式细胞术数据，快速预测急性髓系白血病的关键分子突变（NPM1和FLT3-ITD），从而在数周等待期之前为早期治疗决策提供指导。 |
| [^37] | [WaveTLM: Reliable Time-Series Language Modeling through Task Compilation](https://arxiv.org/abs/2609.18812) | 提出WaveTLM编译器-执行器架构和ExecTS-QA基准，将用户请求编译为带类型的任务状态并由任务原生执行器生成可靠输出，解决了时间序列语言模型中“响应看似合理但实际幻觉任务对象”的可靠性问题。 |
| [^38] | [A Convergence Framework for Deep $V$-Learning: Error Propagation and Sharp Action-Gap Bounds](https://arxiv.org/abs/2609.18782) | 该论文为深度 V-学习建立了收敛性理论框架，将更新误差分解为拟合、转移复用、目标构建、重放、动作选择和探索六个残差，并在可集中性条件下给出了控制策略损失的显式误差传播界与尖锐的动作间隙界。 |
| [^39] | [CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents](https://arxiv.org/abs/2609.18779) | 提出CERA-MoA框架，通过迭代强化学习使动态路由器与持续学习的LLM智能体协同进化，并利用基于中间层隐藏状态的熟悉度估计器和累积阈值自适应路由机制，动态激活最小智能体子集，以在性能与开销之间取得平衡。 |
| [^40] | [Stable Filters for Generative Modeling of Graph Signals](https://arxiv.org/abs/2609.18759) | 本文针对漂移项结合图滤波器与图神经网络的图感知连续时间生成模型，推导了量化图扰动对生成分布影响的显式Wasserstein稳定性界，并据此提出了在保持图热扩散平滑性的同时增强结构稳定性的图滤波器设计原则框架。 |
| [^41] | [When Edit Flows are Edit Jumps: replicating Edit Flows and EvoFlows](https://arxiv.org/abs/2609.18745) | 本文证明Edit Flows与EvoFlows本质上是同一底层过程（连续时间中编辑逐个触发的纯跳跃式生成器匹配），并发布首个开源实现EditJumps——一个在166万同源抗体对上训练的通用抗体编辑器，可零样本编辑未见先导序列而无需按家族重新训练。 |
| [^42] | [Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning](https://arxiv.org/abs/2609.18723) | 提出ME-Decoding解码框架，将LLM的候选token选择建模为集成剪枝问题，利用马氏距离驱动的目标函数和自适应带宽核构建的token相似度矩阵，在保持高概率的同时增强语义多样性并去除冗余路径，同时通过高效贪心算法降低计算开销。 |
| [^43] | [Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening](https://arxiv.org/abs/2609.18708) | 本文揭示了PPO评论家中的系统性失效模式“价值平坦化”，即真实状态值变化剧烈而评论家预测却过于平坦，将其归因于评论家损失中的隐式方差惩罚与冗余更新，并提出仅对响应中少数分离良好的状态计算价值损失的稀疏近端策略优化算法SP³O来缓解该问题。 |
| [^44] | [Toward Composable Network Digital Twins: A Subgraph-Based Latency Prediction Study](https://arxiv.org/abs/2609.18704) | 本文提出一种可组合的网络数字孪生方法，将网络分解为可重用的子图单元孪生，并通过轻量级组合器聚合来预测每条路由的端到端时延，解决了现有方法单体化、难以适应拓扑和流量变化的问题。 |
| [^45] | [Rank and computation of the pathlifting Jacobian of a DAG ReLU network](https://arxiv.org/abs/2609.18682) | 本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。 |
| [^46] | [The Uneven Impact of Generative AI on Student Learning: Examining the Roles of Reliance, Evaluation Literacy, and Course Policy in AI-related Courses](https://arxiv.org/abs/2609.18676) | 该研究基于118名学生的调查识别出四类GenAI用户群体，发现学习获益与早期依赖、认知依赖及学术任务支持密切相关，并受教师课程政策、工具版本和工具使用数量的显著影响，揭示了生成式AI对学生学习影响的不均衡性。 |
| [^47] | [VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge](https://arxiv.org/abs/2609.18663) | 提出VLA-ULAP框架，通过仅740万参数的超轻量级本地动作预测器与云端VLA调用交替执行，在边缘设备上将单次推理延迟和能耗降低一个数量级，减少近半至四分之三的云端调用次数，同时保持95%以上的任务成功率。 |
| [^48] | [Revisiting Distributed Sign-Based Variance Reduction](https://arxiv.org/abs/2609.18656) | 本文通过提出在服务器端利用递归梯度增量的无偏压缩来跟踪全局梯度，解决了数据异构情况下符号聚合引入偏差的问题，首次在非凸随机优化和有限和优化中实现了基于符号的分布式方差缩减方法的最优收敛速率。 |
| [^49] | [Learning to Program Adaptive Non-Local Observables for Machine Learning](https://arxiv.org/abs/2609.18655) | 提出QFWP-ANO架构，利用经典超网络根据输入动态编程变分量子电路参数与非局部观测量，在时间序列预测和强化学习任务上显著超越现有基于ANO的量子神经网络。 |
| [^50] | [Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection](https://arxiv.org/abs/2609.18644) | 该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。 |
| [^51] | [CoRe-MARL: Cooperative Redistribution Under Unknown Dynamics Using Recurrent Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.18639) | 该论文提出CoRe-MARL框架，将救灾物资再分配问题建模为分散式部分可观测马尔可夫决策过程，利用循环多智能体强化学习使各中心在信息有限、需求动态未知的情况下协同学习再分配策略，以改善最差区域服务并缩小区域间服务差距。 |
| [^52] | [How Many Labels Does Model Choice Need? Certificates and Budgets for Selective Prediction](https://arxiv.org/abs/2609.18622) | 该论文量化了比较模型选择性预测性能（AUGRC）所需的标签预算，通过预标签下界与覆盖线性规划证书证明：确定性选择模型在某些条件下几乎需要标注全部标签，而准确率选择可由少量分歧标签裁决。 |
| [^53] | [Learning Array Signal Topologies as Conditional Neural Manifolds](https://arxiv.org/abs/2609.18616) | 提出条件神经流形（CNM），以观测条件化的学习流形替代固定阵列流形，无需导向矢量监督即可提升MUSIC等波达方向估计方法在模型失配下的鲁棒性与精度，并可推广至其他基于流形的方法。 |
| [^54] | [Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence](https://arxiv.org/abs/2609.18612) | 该论文提出通过计算神经元输入权重向量与输出权重向量之间的余弦相似度来识别“弱化神经元”，并发现这类神经元虽然在模型中数量稀少，却激活频繁且对模型行为具有超乎寻常的影响力，同时九个不同的大语言模型均呈现弱化神经元集中分布于后期层、强化神经元集中分布于中早期层的相似模式。 |
| [^55] | [A Geometric Theory of Decision Boundaries in Structured Markov Decision Processes](https://arxiv.org/abs/2609.18610) | 本文提出了一种结构化马尔可夫决策过程中最优策略诱导的决策边界几何理论，证明该几何是策略重构的最小表示，并决定了重构问题的统计与计算复杂度。 |
| [^56] | [PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?](https://arxiv.org/abs/2609.18605) | PACT是一个评估企业级AI智能体在用户施压等压力情境下能否坚持遵守合规规则的基准测试，涵盖十二个受监管企业领域和四十八个真实多轮对话场景。 |
| [^57] | [Online Robust Reinforcement Learning Through Monte-Carlo Planning](https://arxiv.org/abs/2609.18599) | 本文提出了一种鲁棒的蒙特卡洛树搜索变体，通过鲁棒幂平均回传算子和探索奖励机制来处理模拟器与真实世界之间的状态转移动力学与奖励分布歧义，并实现了根节点价值估计的O(n^{-1/2})收敛速率。 |
| [^58] | [Reasoning through Evolution: Automatic Meta-path Discovery for LLM-based Fake News Detection](https://arxiv.org/abs/2609.18597) | 提出MAGER多智能体遗传进化框架，自动发现优化的元路径，将复杂传播图压缩为信息子图，使冻结的大语言模型能够进行结构感知的虚假新闻检测推理，摆脱对大量标注数据的依赖。 |
| [^59] | [ReDIL-GNN: Resynthesis Domain Incremental Learning for Circuit Graph Neural Networks](https://arxiv.org/abs/2609.18595) | 提出了ReDIL-GNN再综合域增量学习框架及预适配评分指标RAI，用于应对逻辑再综合引起的电路图神经网络域偏移问题，并指导何时以及如何进行适配。 |
| [^60] | [Peak-Aware Short-Term Load Forecasting Across Distribution Grid Aggregation Levels](https://arxiv.org/abs/2609.18588) | 该论文提出峰值感知的短期负荷预测评估框架，在英国和瑞士的开放数据集上，对配电网区域代码、二次变电站和低压馈线三个聚合层级的统计基线、机器学习模型和时序基础模型进行了比较，发现Chronos-2在高需求时期的预测性能最佳。 |
| [^61] | [Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces](https://arxiv.org/abs/2609.18587) | 该论文提出无标签仅偏置测试时强化学习方法，以多数投票伪标签为奖励、仅优化约10万个偏置参数，即可在数学、视觉语言和音频推理等多个任务上达到与全参数方法相当甚至更优的性能。 |
| [^62] | [TTM-Bench: A Framework for Text-to-Music System Performance Benchmarking](https://arxiv.org/abs/2609.18585) | 该论文提出了TTM-Bench框架，为文本到音乐系统定义了统一的、可复现的性能基准测试协议，从音乐内容对齐度和计算效率两个维度实现了跨系统的可靠性能比较。 |
| [^63] | [Accurate Trace Estimation with Fewer Random Bits via Recursive TensorSketch](https://arxiv.org/abs/2609.18577) | 本文提出基于递归张量Sketch的方法，通过大幅减少随机比特的消耗，实现了对仅能通过矩阵-向量乘积访问的隐式矩阵迹的精确估计。 |
| [^64] | [Deep learning emergent spacetime from fermionic spectral functions in holography](https://arxiv.org/abs/2609.18566) | 提出基于神经常微分方程的物理信息机器学习框架，直接从边界费米子谱函数重构带电AdS黑洞的体时空几何与规范场，可可靠覆盖非费米液体、边际费米液体和类费米液体三种量子临界区域，并揭示仅由近视界AdS₂几何数据决定的时空简并性。 |
| [^65] | [Variational Quantum Transformer Architecture for Synthetic Language Generation](https://arxiv.org/abs/2609.18565) | 提出了一种兼容NISQ设备的紧凑变分量子Transformer架构，通过用量子编码器、连接器和解码器电路替代经典注意力与前馈子层，能够端到端训练并学习非平凡的语法结构，在合成语言生成任务上实现完美确定性生成和高词典序有效性。 |
| [^66] | [The evolution of sex for artificial intelligence: a population-genetic framework for multigenerational model populations](https://arxiv.org/abs/2609.18560) | 该论文首次将群体遗传学（包括有性/无性生殖和Wright-Fisher过程）形式化地应用于多代AI模型种群，证明了模型递归训练导致的“模型崩溃”在遗传学上可精确类比，且该框架在多种网络架构和大语言模型中普遍成立。 |
| [^67] | [Interpretable Patch-Based Deep Learning for Wildfire Spread Prediction from Ensemble Simulations](https://arxiv.org/abs/2609.18555) | 该研究比较了四种深度学习架构作为昂贵物理模拟器的低成本替代方案来预测野火蔓延，发现地表燃料载荷是唯一有效的预测变量（可将误差降低21%），并通过可解释性实验验证了模型学习。 |
| [^68] | [Revisiting the Objective of Echo Chamber Detection](https://arxiv.org/abs/2609.18545) | 本文首次基于集合函数傅里叶变换理论形式化了回音室检测的目标函数，提出可扩展的半定松弛求解算法，在合成和真实数据集上均优于现有方法。 |
| [^69] | [Provable Guarantees and Efficient Learning of Structural Equation Models with Latent Confounders](https://arxiv.org/abs/2609.18535) | 本文针对含潜在混杂因子的线性结构方程模型，提出了一种通过将精度矩阵分解为稀疏加低秩两部分来迭代重建观测变量因果有向无环图的高效算法，并给出了可证明的正确性保证。 |
| [^70] | [Provable Guarantees for Spectral Structured Prediction](https://arxiv.org/abs/2609.18527) | 本文提出一种简单的谱方法，通过带噪符号邻接矩阵主特征向量的符号来恢复图节点标签，并给出与图结构无关的可证明理论保证，明确量化了谱间隙、节点数、度分布和噪声水平对标签恢复精度的影响。 |
| [^71] | [TRIPROBE: Probing Task Separability Beyond Classification for XAI](https://arxiv.org/abs/2609.18525) | TriProbe 提出一个多层次探测框架，通过输入空间的基础探测器、特征表示的潜在探测器和分类器输出的最终探测器，结合最大费舍尔判别比指标，对多任务模型的可分性进行可解释诊断，定位瓶颈任务对并指导数据收集与架构设计。 |
| [^72] | [COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads](https://arxiv.org/abs/2609.18519) | 本文提出了独立于历史工作负载信息的碎片化度量指标SIF，并设计了COMPASS-ABS调度算法，通过将集群状态约束在基于锚点的紧凑空间内，持续降低共享GPU集群的资源碎片化，从而提升集群利用率并缩短深度学习训练作业的周转时间。 |
| [^73] | [Beyond Routine Compliance: Cunning Data Cultivates Safety Vigilance in Large Language Models](https://arxiv.org/abs/2609.18515) | 本文提出通过训练模型识别包含误导性前提和非典型推理的“狡猾问题”，培养大型语言模型的安全警觉性，使其能够识破隐藏在良性语境下的有害意图，从而显著提升对越狱攻击的鲁棒性。 |
| [^74] | [ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware](https://arxiv.org/abs/2609.18514) | ActiveScale通过模型（逐帧位姿令牌与位姿预测头）、数据（1000小时人-机器人中期训练）和硬件的协同设计，使VLA模型能够跨视点推理并主动获取信息性观测，实现机器人主动感知。 |
| [^75] | [Learning from Distributed Eyes: Leveraging Collaborative Perception for Automated Model Adaptation](https://arxiv.org/abs/2609.18511) | 提出LDE框架，将车路协同感知转化为高质量监督信号用于生成伪标签，从而实现自动驾驶感知模型在新环境下的自动化无监督适配，并解决了通信瓶颈、视角差异和可靠性等关键挑战。 |
| [^76] | [Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models](https://arxiv.org/abs/2609.18489) | 本研究通过动能谱分析揭示了当前最先进的概率机器学习天气预报模型虽能产生接近真实的动能谱量级，但大多无法像物理数值模型那样再现预期的动能向上尺度级串传递，且所有模型均表现出向上尺度增长的蝴蝶效应式误差传播。 |
| [^77] | [Beyond Random Couplings: Contrastive Noise Alignment in Generative Flows](https://arxiv.org/abs/2609.18488) | 提出对比噪声对齐方法，通过将噪声批次建模为相互作用粒子系统并利用跨模态InfoNCE目标直接优化噪声表示，创建动态对比耦合以改善生成流模型的训练。 |
| [^78] | [Hyperbolic Graph Representation Learning for Differential Diagnosis on Biomedical Knowledge Graphs](https://arxiv.org/abs/2609.18481) | 该研究表明双曲图嵌入能以远低于欧几里得方法的维度有效利用生物医学知识图谱的层次结构，并支持在整合患者信息的异构图上进行孟德尔疾病的鉴别诊断。 |
| [^79] | [Spatially Adaptive Noise Injection](https://arxiv.org/abs/2609.18466) | 本文提出空间自适应噪声注入（SANI）采样框架，通过概率门控机制和空间自适应方差在逐像素层面动态调整噪声注入，在去噪器不确定的边缘纹理区域施加随机校正，而在分数估计精确的平滑区域保持确定性更新。 |
| [^80] | [Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning](https://arxiv.org/abs/2609.18461) | 提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。 |
| [^81] | [Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving](https://arxiv.org/abs/2609.18442) | 提出了RiskWorld风险感知世界建模框架，通过流引导占据演化实现共享占据预测，仅在额外预测风险触发干预且替代轨迹满足逐分量约束时才进行选择性轨迹替换，从而提升自动驾驶运动规划的安全性。 |
| [^82] | [HPOQuest: A Rare-Disease Diagnostic Agent Using Active Phenotype Acquisition](https://arxiv.org/abs/2609.18431) | HPOQuest是一个免训练的罕见病诊断框架，通过迭代式主动获取信息性表型来不断更新疾病概率排名，将稀疏初始表型下的诊断Recall@1提升高达30个百分点。 |
| [^83] | [HiLNO: A Hierarchical Latent Neural Operator with Multi-Scale Supervision for PDEs on General Geometries](https://arxiv.org/abs/2609.18419) | HiLNO提出了一种层次化潜在神经算子，通过构建从细到粗再到细的潜在空间、多尺度监督机制和各向异性高斯注意力，有效缓解了压缩过程中的信息丢失问题，实现了对一般几何上具有多尺度结构的偏微分方程的高效求解。 |
| [^84] | [Gradient Descent with Stochastic Subspaces via Persistence of Memory](https://arxiv.org/abs/2609.18416) | 本文提出“记忆持久性”技术，利用一个与梯度弱相关、可长期固定无需频繁更新的指导向量来引导随机子空间的生成，从而显著扩展并改进了大规模优化中的随机子空间梯度下降方法，且该向量可借助稀疏性或小批量等结构化特性以低成本高效获得。 |
| [^85] | [TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting](https://arxiv.org/abs/2609.18407) | TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。 |
| [^86] | [Reliable Virtual Sensing: A Multi-Domain Benchmark for Robustness Under Sensor Failures](https://arxiv.org/abs/2609.18396) | 本文提出了首个面向学习型虚拟传感中传感器故障鲁棒性的多领域基准MuViS-C，涵盖十种故障模式、多种严重程度以及平均误差、相对退化和最坏情况脆弱性等鲁棒性度量，并在六个领域的九个数据集上对六种架构进行了系统评测。 |
| [^87] | [Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism](https://arxiv.org/abs/2609.18381) | 该论文提出通过训练判别器来生成类似散度的真实性评分，能够自适应地检测天气预报模型表现出的任何失败模式，从而克服固定指标存在的盲区。 |
| [^88] | [Semantic CSI Feedback for Beam Selection: When Task-Aware Embeddings from Sparse Pilots Outperform Full-Bandwidth Reconstruction](https://arxiv.org/abs/2609.18368) | 该论文提出面向波束选择的语义CSI反馈方法，仅用43个稀疏导频生成的8维任务感知语义嵌入，其波束预测精度超越了利用全部512子载波信道进行重建的传统方法，证明波束相关信息本质上是低维的。 |
| [^89] | [Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts](https://arxiv.org/abs/2609.18366) | 提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。 |
| [^90] | [RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control](https://arxiv.org/abs/2609.18359) | RecMorph提出了一种拓扑引导的空间递归架构，通过深度优先遍历将运动学树转化为序列，并利用共享双向转换联合实现跨肢体通信与表示变换，在广义形态控制任务中取得了最佳性能。 |
| [^91] | [Trajectory Learnability for Offline On-Policy Distillation with Imperfect Teachers](https://arxiv.org/abs/2609.18321) | 本文提出利用教师成功问题作为廉价参考，通过观测学生模型在不同训练阶段对教师失败轨迹中各token似然的变化（带符号似然变化），来识别不完美教师监督下仍然可学习的内容，从而解决离线在线策略蒸馏中不完美监督持续存在的问题。 |
| [^92] | [Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models](https://arxiv.org/abs/2609.18320) | 该论文提出一种无监督的注意力分散度量方法，通过监测大语言模型内部注意力机制的时间波动性来检测幻觉，摆脱了对输出校准的依赖，在数学推理基准上相比基于输出的基线方法AUC提升高达0.076。 |
| [^93] | [Multi-Appliance Non-Intrusive Load Monitoring via Label-Preserving Aggregate Recomposition and Prediction Consistency](https://arxiv.org/abs/2609.18315) | 该论文提出一种结合标签保持的聚合信号重组与预测一致性的多电器非侵入式负荷监测方法，通过仅替换总功率中的残余背景来构造新训练样本并施加跨窗口预测一致性约束，从而提升模型对未见家庭的泛化能力。 |
| [^94] | [Beyond Quadratic Loss: The Stability Phase Diagram of Adam](https://arxiv.org/abs/2609.18314) | 该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。 |
| [^95] | [Bias Amplification in Multi-Agent Network: How Biased Agents Shape Opinions and Rhetoric](https://arxiv.org/abs/2609.18306) | 该研究发现，在多智能体大语言模型系统中，即使只有少量持持久极端观点的有偏见智能体，也能通过文本交互显著改变无偏见智能体的观点，且 Llama 3.2 的观点偏移速度比经典观点动力学模型更快。 |
| [^96] | [Where Should Agents Live? Energy-Memory Characterization of Agentic AI for the Edge-Cloud Continuum](https://arxiv.org/abs/2609.18283) | 该论文针对边缘-云连续体中的多智能体AI工作流提出能耗与内存特性表征分析，填补了现有AI生命周期指标忽视多智能体执行图的空白，帮助网络运营商确定智能体团队的最佳部署位置并量化分布式智能体通信的能耗代价。 |
| [^97] | [A GAN-Based Framework for Robust DDoS Attack Detection](https://arxiv.org/abs/2609.18281) | 该论文提出了一种基于WGAN-GP生成合成对抗流量的鲁棒DDoS攻击检测框架，通过将生成对抗建模与随机森林、深度神经网络集成和Transformer等机器学习模型相结合，有效提升了模型对抗针对性对抗攻击的防御能力。 |
| [^98] | [Behavioral Fingerprinting and Navigation Prediction in Web Browsing](https://arxiv.org/abs/2609.18273) | 该论文通过实证研究证明，即使短暂的浏览会话也足以高度唯一地识别用户身份，且结合图建模与大语言模型可以从长期交互结构中高度准确地预测用户的下一步导航行为。 |
| [^99] | [Acting in Meters: Learning Metric Interactions for Precise Robotic Manipulation](https://arxiv.org/abs/2609.18243) | 该论文提出了一个度量交互框架，通过交互中心标记（ICTs）显式建模物体级的末端执行器-物体相对位姿交互，并通过度量动作交互场（MAIF）学习场景几何条件下的动作修正，从而在物理笛卡尔空间中以共享度量尺度实现精确的机器人操作。 |
| [^100] | [F-DACE: Fuzzy Disagreement-Aware Causal Evidence Fusion for Abstention-Safe Conversational Retail Decision Support](https://arxiv.org/abs/2609.18238) | F-DACE框架通过模糊隶属度融合多个因果估计器的证据一致性，并在估计量不匹配、诊断失败或证据冲突时主动弃权，从而在对话式零售决策支持中显著降低错误建议率。 |
| [^101] | [Anomaly Detection in General Ledger Data: Results from a Hybrid Approach](https://arxiv.org/abs/2609.18228) | 该研究提出将日记账测试（JETs）与机器学习方法相结合的混合模型，以减少误报、提高总账数据异常检测的性能和有效性，从而提升审计效率。 |
| [^102] | [APGEM: Adaptive Policy-Guided Error Mitigation for Quantum Reinforcement Learning on a Real-World CVRP Case Study](https://arxiv.org/abs/2609.18219) | 提出了APGEM自适应控制器，能够在量子强化学习过程中根据学习情境在线动态选择最合适的误差缓解技术（ZNE、PEC、CDR、REM），有效应对NISQ硬件噪声，并在基于德里真实地标的城市物流车辆路径问题上得到验证。 |
| [^103] | [A Lightweight CNN Integrated Compact Convolutional Transformer for Multi-Scale Feature Learning and reducing computational complexity for breast cancer mammography image detection and classification](https://arxiv.org/abs/2609.18212) | 该论文提出一种轻量级CNN与紧凑卷积Transformer相融合的模型，仅用约25万参数就在三个乳腺癌钼靶数据集上达到99%-100%的准确率，并结合可解释AI增强临床可信度。 |
| [^104] | [Reinforcement Learning for Real-Time Vision-Language-Action Policies](https://arxiv.org/abs/2609.18207) | 该论文提出在EXPO-FT框架上通过强化学习微调使VLA策略满足实时控制要求，通过解耦慢速动作生成与快速动作执行，解决了模型推理延迟导致的观测过时和分布偏移问题，从而提升真实世界机器人操控的可靠性。 |
| [^105] | [Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors](https://arxiv.org/abs/2609.18203) | 该论文提出了行为到价值（B2V）任务，构建了首个电子商务消费价值分类体系（ECVT）和基于真实淘宝行为日志的B2V-Bench基准数据集，实现了从电子商务行为轨迹中识别和测量消费者价值观，并据此赋能大语言模型。 |
| [^106] | [Transformation Laws in Neural Representations: Structure, Realisability, and Construction](https://arxiv.org/abs/2609.18190) | 该论文建立了神经表征中参考变换的实现理论，刻画了变换何时能通过编码器传递，证明实现缺陷由变换对被丢弃信息的需求决定，并给出线性实现存在的精确条件——即被保留的调和块在变换作用下保持不变。 |
| [^107] | [MoRE: Mixture of Reused Experts](https://arxiv.org/abs/2609.18176) | MoRE通过在相邻层组之间共享专家池并引入可学习的深度嵌入对每层输入进行条件化，在不增加参数的情况下扩展路由组合多样性，实现了比标准MoE和权重共享方法更低的困惑度和更强的下游性能。 |
| [^108] | [Beyond Direct Sensing: Harnessing Indirect Observations from Third-Party Sensors in Vehicle Tracking](https://arxiv.org/abs/2609.18173) | 提出GrayTrack方法，利用道路约束的粒子滤波器将第三方传感器的微弱匿名间接观测与稀疏直接观测相融合，以填补车辆跟踪中的观测空白。 |
| [^109] | [Characterizing Replay Retention Under Dynamics Shift in Model-Based Reinforcement Learning](https://arxiv.org/abs/2609.18167) | 该论文提出用变化幅度和年龄-陈旧度AUC两个量化指标，来刻画基于模型的持续强化学习中动力学变化后应保留还是遗忘旧经验回放数据的权衡问题。 |
| [^110] | [LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era](https://arxiv.org/abs/2609.18148) | 该论文提出LIGE-GR框架，通过列表级生成与评估的方法，解决了将LLM范式中的序列级生成优化融入推荐系统以及避免整体替换成熟工业系统的两大挑战，实现了从传统排序推荐到生成式推荐的平滑跃迁。 |
| [^111] | [Reaching Every Position Without Searching: Rotating Sparse Wiring on the Hypercube as a Substitute for Attention](https://arxiv.org/abs/2609.18145) | 该论文提出将序列位置视为超立方体顶点、采用逐层旋转的稀疏连线结构，仅用log₂n层和每层2n条链接即可实现所有位置间的信息全连通，可作为注意力机制的高效替代方案。 |
| [^112] | [Rethinking How We Evaluate Methodological Progress in Health AI](https://arxiv.org/abs/2609.18134) | 该研究通过在统一评估框架内重新实现12个健康AI算法并在MIMIC-IV和NWICU两个临床数据集上进行评估，实证探讨了EHR AI评估中可重复性与临床任务定义的障碍，以及算法相对比较在不同任务族和数据集间的可迁移性。 |
| [^113] | [Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing](https://arxiv.org/abs/2609.18131) | 提出基于激活熵的比特分配框架Colla-Q，通过极小极大精度平衡策略均衡MoE量化中各专家的性能，从而提升整体模型表现并降低对校准数据的依赖。 |
| [^114] | [Benchmarking Tabular Foundation Models as Surrogates in Expensive Evolutionary Optimization](https://arxiv.org/abs/2609.18130) | 本文通过涵盖离线与在线设置、多种优化场景的大规模实验与理论分析，系统性地评估了表格先验数据拟合网络作为昂贵进化优化中代理模型的有效性。 |
| [^115] | [MCLC-NET: Multimodal Continual Learning for Leaf Counting](https://arxiv.org/abs/2609.18129) | 该论文提出MCLC-NET，一个融合深度和热成像等多模态信息、采用记忆缓冲策略进行顺序学习的叶片计数持续学习框架，并发布了真实世界多模态叶片计数数据集MMLC。 |
| [^116] | [Learning Fractional-Order Dynamics from a Single Trajectory](https://arxiv.org/abs/2609.18127) | 本文提出了一种简单的两阶段估计器 FO-GS，利用分数差分算子的对角结构逐行解耦辨识问题，从而能够从单条观测轨迹中对分数阶线性时不变系统进行辨识并给出高概率非渐近误差界。 |
| [^117] | [Preservation of Log-Concavity and Convergence of Wasserstein-Fisher-Rao Gradient Flows](https://arxiv.org/abs/2609.18118) | 本文证明Wasserstein-Fisher-Rao梯度流在满足曲率条件的强对数凹目标分布下能够保持强对数凹性，据此推导出对称化KL散度的显式非渐近收敛速率，无需暖启动，且收敛速率可加性分解为Wasserstein与Fisher-Rao两部分贡献。 |
| [^118] | [Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving](https://arxiv.org/abs/2609.18112) | 提出FairInference系统，首次提供δ-token公平性保证，确保多租户LLM服务中行为良好客户端的每个token生成延迟最多比隔离运行时多δ个时间单位，实现强大的延迟隔离。 |
| [^119] | [A Comprehensive Review of Generative Physical Artificial Intelligence](https://arxiv.org/abs/2609.18111) | 本综述系统梳理了生成式物理人工智能（GPAI）领域，提出了涵盖机器人基础模型、视觉-语言-动作模型、大行为模型、扩散策略模型和世界基础模型五大方法的分类体系，并分析了它们的架构基础、应用现状及互补关系。 |
| [^120] | [FoundAna: A GNN-assisted Foundation Model for Graph Anomaly Detection](https://arxiv.org/abs/2609.18107) | FoundAna是首个结合GNN与transformer的图异常检测基础模型，通过异常检测专用的GNN组件与四种互补位置编码捕获局部和全局结构信息，实现了可泛化的跨图异常检测。 |
| [^121] | [iMINDBench: iEEG Multi-Institution Neural Decoding Benchmark](https://arxiv.org/abs/2609.18104) | 提出iMINDBench，一个涵盖三个自然电影观看数据集、十五项解码任务的多机构颅内脑电图神经解码基准，通过标准化预处理流程和固定评估划分，解决了iEEG解码模型泛化能力难以衡量、模型改进与预处理增益难以区分的问题。 |
| [^122] | [Beyond Pixel Similarity: Task-Aware Evaluation of GAN-Based Synthetic Sonar Data for Robotic Perception](https://arxiv.org/abs/2609.18100) | 该研究通过对比具有不同判别器感受野的Pix2Pix模型，揭示了传统图像保真度指标（SSIM、PSNR、MSE）无法充分反映GAN合成声呐数据在下游感知任务中的实际表现，从而提出应以任务感知的评估方式来评价合成数据质量。 |
| [^123] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^124] | [FedPGT: Progressive Gradient Transmission for Vehicular Federated Learning over Time-Varying Channels](https://arxiv.org/abs/2609.18089) | 该论文提出FedPGT方案，使车辆根据瞬时信道条件自适应地渐进传输大幅值梯度条目，并通过揭示幂律收益递减特性的收敛界指导在线随机优化决策，解决了车辆移动导致信道快速变化使预定义传输策略失效的问题。 |
| [^125] | [Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models](https://arxiv.org/abs/2609.18084) | 该研究通过在五个不同架构的VLA模型上测量区域隔离微调下的适应成本，揭示了不同类型的分布偏移会系统性集中在特定网络区域（外观变化集中于视觉编码器、指令变化集中于语言骨干、新物体变化集中于视觉编码器与动作头），并提出了一种仅凭十个无标注观测、无需微调即可诊断适应成本并针对性分配适配容量的流水线。 |
| [^126] | [Beyond Embedding Transfer: Component Roles in Grokking Transfer and Stability](https://arxiv.org/abs/2609.18078) | 该论文通过大规模控制实验首次系统区分了Grokking热启动迁移中各模型组件的角色，发现迁移内部注意力/MLP权重不仅可将早期准确率提升5.46个百分点并缩短确认延迟，且该优势在双层模型的前瞻性复现实验中得到完全验证。 |
| [^127] | [vidax: A Unified JAX Framework for Video Generative Models on Accelerator Meshes](https://arxiv.org/abs/2609.18077) | vidax是一个开源的JAX推理框架，为视频生成模型提供了TPU上生产就绪的推理路径，统一了张量并行与序列并行，并支持零拷贝的PyTorch权重转换和超出单设备内存的分辨率支持。 |
| [^128] | [Regional Explanations via Causal Sufficiency and Necessity](https://arxiv.org/abs/2609.18049) | 该论文提出了SNRE框架，通过因果干预和可微估计器学习输入-输出区域对，使得输入属于某区域成为模型输出落入目标区域的充分且必要条件，为模型预测行为提供了区域级的因果解释。 |
| [^129] | [Exact semantic readout from compressed vector representations](https://arxiv.org/abs/2609.18047) | 本文给出了压缩向量表示能够精确线性或仿射读出谓词真值条件的充要行空间判据，并通过实验发现预训练词向量虽大多严格可分，但无一能实现精确读出。 |
| [^130] | [Structural Inference under Hidden Agents](https://arxiv.org/abs/2609.18045) | 该论文首次提出并形式化了“隐藏智能体下的结构推断”问题，通过破解交互恢复与轨迹重建之间的循环依赖难题，实现了对不可观测智能体的轨迹与交互结构的联合推断。 |
| [^131] | [Physics-Informed Neural Networks for Fast Multilayer Spectral Inversion of H{\alpha} 6562.8 A and Ca II 8542.1 A Spectra](https://arxiv.org/abs/2609.18025) | 该研究提出了一种物理信息神经网络框架，在保留多层光谱反演物理可解释性的同时，大幅加速了太阳色球Hα和Ca II光谱的大规模反演计算。 |
| [^132] | [Newer Is Not Fairer: Gender Stereotyping in Text-to-Image AI Across Model Generations](https://arxiv.org/abs/2609.18007) | 该研究通过生成8,000张图像对比四代Stable Diffusion模型，发现文生图AI普遍存在严重性别刻板印象（76.4%的图像主体为男性，甚至在传统女性主导的职业中也有57.6%为男性），且更新的模型并未更公平。 |
| [^133] | [A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality](https://arxiv.org/abs/2609.18005) | 本文提出了一种经过正式校准的LLM评判测量方法，通过引入分布上与原模型完全一致的“零条件”验证机制，实现了对量化、早退、投机解码等推理加速技术对输出质量影响的严格、可跨系统比较的测量。 |
| [^134] | [The Attention Within: Consensus Dynamics in Selective State Space Models](https://arxiv.org/abs/2609.17997) | 该论文从动力系统视角证明，选择性状态空间模型（SSM）核心的递归机制与Transformer中的注意力类似，同样会驱动token达成共识（聚簇坍缩）。 |
| [^135] | [QuanText: Protecting Dataset-Level Secrets in Textual Data Sharing](https://arxiv.org/abs/2609.17995) | QuanText提出了一种无需训练、与大语言模型无关的文本随机量化数据发布机制，能够在保护文本数据集中数据集级全局秘密（如敏感属性比例）的同时保持数据效用，弥补了差分隐私对聚合属性保护不足的缺陷。 |
| [^136] | [The Operable Pareto Front: Distilling Offline Search into Run-Time Control for Multi-Objective UAV Edge-Computing Scheduling](https://arxiv.org/abs/2609.17992) | 提出PrefDT——首个偏好条件化决策变换器，将期望的能耗-时延权衡作为模型输入，仅需一次离线训练即可在运行时按需返回多目标无人机边缘计算调度帕累托前沿上的任意点，并具备预算跟踪与用户报告丢失容错能力。 |
| [^137] | [Fourier Analysis of Parametrized Interactive Quantum Classifiers](https://arxiv.org/abs/2609.17991) | 本文通过推导单目标量子比特参数化交互式量子分类器的闭式解析解，揭示了哈密顿量参数以傅里叶方式控制分类器输出的常数、正弦和余弦分量，建立了量子特征映射的傅里叶解释并据此提出了广义哈密顿量编码族。 |
| [^138] | [TuiML: Machine Learning for AI Agents](https://arxiv.org/abs/2609.17984) | TuiML是一个专为AI智能体设计的机器学习库，通过机器可读的元数据描述组件、验证并追踪每次调用、支持会话导出为可复现笔记本，解决了传统面向人类程序员的库在智能体使用时功能不可见、错误延迟暴露和实验状态丢失的问题。 |
| [^139] | [Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization](https://arxiv.org/abs/2609.17973) | 该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。 |
| [^140] | [Mixed-Integer Nonlinear Differentiable Predictive Control for Underground Pumped Hydro Energy Storage Systems](https://arxiv.org/abs/2609.17964) | 本文扩展了混合整数可微预测控制框架，通过保持梯度的并行可微仿真器、捕捉长程时间依赖的Transformer编码器和Gumbel-Softmax温度退火调度三项创新，以自监督方式学习神经控制策略，解决地下抽水蓄能系统日前调度中多模态离散决策与非线性动力学的可微优化控制难题。 |
| [^141] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^142] | [Maximum Strong Independent Sets in Hypergraphs: Reductions, Bounds, and Greedy Certificates](https://arxiv.org/abs/2609.17951) | 本文针对有限超图中的最大强独立集问题，提出了精确归约、上界估计、穿孔与覆盖证书收紧方法，并分析了基于块权重的分层贪心聚类算法，可应用于多带LSH-MinHash去重等仅支持局部约束的场景。 |
| [^143] | [ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference](https://arxiv.org/abs/2609.17943) | ASPIRE提出了一种非同步的批量自推测解码框架，通过统一混合前向计算、基于接受率估计和批次感知成本模型的在线调度器，让批中每个请求独立决定验证时机，从而加速长上下文大语言模型推理。 |
| [^144] | [On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs under Linear Parametric Models](https://arxiv.org/abs/2609.17942) | 本文证明了在线性参数模型中，只要有序节点至少有三个类别且指数族节点至少有三个支撑点，连接这两类节点的每条边的方向都可仅凭联合分布在任意参数取值下被辨识，并通过反向论证证明了这两个条件的必要性。 |
| [^145] | [Beyond the Previous Layer: Residual Predictive Structure in Sparse MoE Routing](https://arxiv.org/abs/2609.17940) | 该研究发现稀疏MoE路由器的专家选择历史具有超越紧邻上一层的残差预测结构——更早层的专家选择包含显著的额外预测信息，能显著提升对下一层路由决策的预测精度。 |
| [^146] | [Locating Hidden Failures Makes Long-Horizon Agents More Reliable](https://arxiv.org/abs/2609.17930) | 该研究通过分析2518条智能体轨迹并将6967个错误归纳为78种失败类型，发现长程智能体在首次犯错后往往难以自我恢复和纠错，即使最终“成功”的运行也可能造成数据删除等不可逆伤害，因此定位隐藏故障是提升智能体可靠性的关键。 |
| [^147] | [Symmetry without a manifold: intrinsic dimension on orbits](https://arxiv.org/abs/2609.17926) | 该论文证明在对称性轨道（如模加法任务）上标准内在维度估计器普遍失效，神经缩放行为不再遵循幂律，而是遵循关于隐藏层宽度的指数定律 $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$。 |
| [^148] | [EdgeReMIND: A Scalable, Top-Ranked Memorization Baseline for Temporal Multi-Relational Link Prediction](https://arxiv.org/abs/2609.17916) | EdgeReMIND 是一种基于数据校准特征和逐关系学习权重的轻量级线性记忆模型，在 TGB 2.0 八个数据集中的六个上取得最高测试 MRR，并成为唯一能在包括三个最大数据集在内的全部数据集上运行的感知关系最先进基线。 |
| [^149] | [Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control](https://arxiv.org/abs/2609.17909) | Zing-0.5是一个50亿参数自回归世界模型，通过统一动作与文本条件化、事件尺度分布匹配蒸馏监督以及低成本流式推理三项创新，实现了用户可通过键盘和文本实时联合控制的可玩生成世界。 |
| [^150] | [Walking the Score Manifold: Continuous-time Generative Dynamics on Learned Data Manifolds](https://arxiv.org/abs/2609.17901) | 该论文提出在习得数据流形上进行连续时间生成建模的新框架，利用预训练分数模型作为几何先验学习向量场，实现任意时间戳的生成与时间超分辨率，并通过促进横向指数稳定性的目标提升长时程推演的鲁棒性。 |
| [^151] | [QEMScore: How Much Does the Measurement Add to Learned Quantum Error Mitigation?](https://arxiv.org/abs/2609.17896) | 该论文提出 QEMScore 评估框架，通过将学习型量子误差缓解器与不读取测量数据的容量匹配对照组进行比较并计入测量开销，揭示了缓解器的收益在很大程度上来自电路结构而非噪声测量数据本身。 |
| [^152] | [TabPFN-3.5: Technical Report](https://arxiv.org/abs/2609.17895) | TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。 |
| [^153] | [Bracketing Uncertainty in Clustering Under the Manifold Hypothesis](https://arxiv.org/abs/2609.17892) | 该论文通过结合内在流形几何（体积增长与触及半径）和样本级度量（填充距离与密度），为互k近邻图聚类建立了阈值现象，从而界定了聚类结果存在不确定性的几何区间。 |
| [^154] | [Long-Context Demonstration Selection Using State Space Models](https://arxiv.org/abs/2609.17888) | 本文提出一种基于状态空间模型（SSM）的示例选择方法，通过从transformer模型蒸馏出线性的SSM，高效解决长上下文场景下推理成本高企的示例选择难题。 |
| [^155] | [Dataset-Dependent Effects of Cross-Depth Aggregation and Soft-Routed Experts in EEG Foundation Model Fine-Tuning](https://arxiv.org/abs/2609.17886) | 在EEG基础模型CBraMod上添加跨深度注意力残差和软路由专家模块的效果高度依赖具体数据集，有时甚至产生负面影响，且带来2至3倍的运行时间和内存开销，并未带来相对完全微调的一致收益。 |
| [^156] | [The Unbearable Weight: Scaling Models and Methods for UAV Audio Classification](https://arxiv.org/abs/2609.17884) | 本文在涵盖31个无人机类别的音频数据集上，系统比较了多种Transformer与卷积骨干网络在全量微调、仅分类器微调及参数高效微调等方法下的表现，揭示了资源受限的无人机部署场景中“重量级”全量微调何时必要、轻量级方案何时更优。 |
| [^157] | [Can VLMs Reliably Assess Sidewalk Accessibility Attributes from Pedestrian-Level Imagery?](https://arxiv.org/abs/2609.17882) | 该研究首次将保形预测应用于基于视觉语言模型的人行道无障碍属性评估，利用首尔514张实地测量图像验证了四个VLM均可达到90%的名义覆盖率，其中有效宽度的估计最具信息量，且非对称校准可在覆盖率不变的前提下将预测区间缩短多达33%。 |
| [^158] | [Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an evolving Label Space](https://arxiv.org/abs/2609.17866) | 该论文提出了一种统一的不确定性感知概率框架，通过自适应β-VAE编码、分类器置信度-后验不确定性-DP-GMM似然的多信号决策机制和基于密度的聚类发现，在演化的标签空间下实现开放世界新意图的持续发现与可控标签空间扩展，并结合回放与弹性权重巩固来缓解灾难性遗忘。 |
| [^159] | [Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels](https://arxiv.org/abs/2609.17857) | 该研究首次系统测量了大语言模型评审中的“同家族偏好”效应——即模型评审会偏袒同一家族的候选模型——通过提出一种固定候选家族的校正估计器，发现四个主流开放权重模型家族均存在3.4-8.4个百分点的显著同家族提升，且该效应与评审侧似然度密切相关。 |
| [^160] | [AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content](https://arxiv.org/abs/2609.17853) | 该论文提出AfriSyCo框架，通过母语后续提问与跨语言2×2因子实验系统测量非洲语言事实内容中模型答案切换行为，发现断言式框架会显著增加错误目标选择（+30.4个百分点），而验证机制可有效降低该效应（-17.4个百分点）。 |
| [^161] | [Learning Heterogeneous Preferences](https://arxiv.org/abs/2609.17847) | 该论文基于理性选择理论，提出一种多阶段架构，通过引入同时以个体及其决策情境为条件的“个体化效用函数”，从多模态数据中学习异质的主观偏好，突破了传统方法假设群体共享统一效用函数的局限。 |
| [^162] | [PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research](https://arxiv.org/abs/2609.17846) | 提出了PrimeScientist框架，将自主研究智能体的研究方向选择与资源投入决策统一建模为序贯决策问题，通过可执行计划树保留竞争性方案及其结果，并利用剩余资源显式引导研究策略，实现研究努力的战略性分配。 |
| [^163] | [Sharp margin-based generalization bounds for realizable SVM](https://arxiv.org/abs/2609.17845) | 该论文通过确定性删除问题的分析，证明了可实现情形下硬间隔支持向量机的泛化风险以概率至少\(1-\delta\)不超过\(\frac{C}{m}(K_m+\log\frac{1}{\delta})\)，其中\(K_m=r_m^2/\gamma_m^2\)为半径-间隔复杂度，得到了阶为\(1/m\)且依赖半径-间隔复杂度的尖锐泛化界。 |
| [^164] | [RoboVAD: A Large Cross-Domain Evaluation Benchmark for Anomaly Detection in Robotic Arm Manipulation Videos](https://arxiv.org/abs/2609.17843) | 本文提出了RoboVAD，一个面向机械臂操作视频异常检测的大规模跨领域评估基准，通过在训练中保留未见过的动作和异常类型来构建贴近现实、更具挑战性的评估场景。 |
| [^165] | [Hybrid coupling with numerics-informed neural networks and the overlapping Schwarz alternating method](https://arxiv.org/abs/2609.17841) | 本文提出了一种利用重叠Schwarz交替方法将预训练的数值信息神经网络（NINN）与经典全阶模型（FOM）耦合的混合建模框架，并证明与PINN不同，NINN无需区域分解即可被准确训练。 |
| [^166] | [Learning Nuclear Structure with AI: Radii and Collectivity](https://arxiv.org/abs/2609.17838) | 该论文开发了基于NuCLR多任务核数据模型的留出集成方法，证明共享表示学习显著提升电荷半径和B(E2)跃迁强度的预测精度，达到与最先进核模型相当的水平，并为实验设计提供数据驱动的指导。 |
| [^167] | [Adaptive hybrid coupling with operator inference, the overlapping Schwarz alternating method and reinforcement learning](https://arxiv.org/abs/2609.17837) | 本文提出一种基于强化学习的方法，利用深度Q网络在重叠Schwarz交替法框架下在线自适应地在子域局部全阶模型与预训练算子推断降阶模型之间进行切换，从而应对瞬态问题中需要高保真分辨的区域随时间变化的挑战。 |
| [^168] | [Procedural Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2609.17831) | 该论文提出“程序化预训练→分子预训练→下游微调”的三阶段训练流程，证明在接触分子数据之前，先从序列结构、元胞自动机、图推理等程序化生成的抽象任务中学习归纳偏置，能够显著提升分子性质预测性能（如在Lipophilicity数据集上降低4.8%的测试误差）。 |
| [^169] | [NObSP: Functional Decomposition of Neural Networks via Oblique Subspace Projections](https://arxiv.org/abs/2609.17825) | 本文提出NObSP框架，利用斜子空间投影将神经网络预测分解为显式的逐特征贡献函数与交互残差，实现局部解释与全局功能分析，并可在卷积网络中免反向传播生成类别激活图。 |
| [^170] | [METALICA: METAdynamics and repLICA exchange for enhanced diffusion sampling](https://arxiv.org/abs/2609.17823) | METALICA通过副本交换机制在预训练扩散模型上实现元动力学，利用偏置势采样和重加权高效探索蛋白质构象的稀有状态，从而实现对稀有事件的有效发现。 |
| [^171] | [The Free Inference Dimension: Complexity Measure for Zero-Collision Navigation under Hypothesis Mixtures](https://arxiv.org/abs/2609.17816) | 本文提出了“自由推断维度”这一新的组合复杂度度量，用以刻画价值混合代理在无需识别真实环境的条件下实现零碰撞导航所能处理的环境复杂度，并证明该维度严格小于VC维、且与Natarajan维仅相差一个路径长度因子。 |
| [^172] | [Principled Koopman Representations with Kalman Inference for Efficient Time-Series Prediction](https://arxiv.org/abs/2609.17815) | 提出K²SVD方法，通过优化Hilbert-Schmidt目标显式学习Koopman算子的主要奇异函数，构建数学上自洽的紧凑低秩Koopman空间，并结合卡尔曼滤波实现高效且可解释的时间序列预测。 |
| [^173] | [Synthetic Electric Vehicle Charging Session Generation Using a Conditional Variational Autoencoder](https://arxiv.org/abs/2609.17808) | 提出了一种条件变分自编码器（CVAE）模型，从真实的电动汽车充电交易数据中生成高质量的合成充电会话数据，以解决真实充电数据因隐私和获取限制而难以获得的难题，为配电网规划和仿真研究提供数据支持。 |
| [^174] | [A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility in LLM Math Reasoning](https://arxiv.org/abs/2609.17804) | 本文揭示大语言模型求解数学应用题的内部计算可分为模式抽象、运算规划、操作数绑定和计算四个阶段，并将无关干扰导致的失败机制定位于“运算规划”阶段的特定注意力头。 |
| [^175] | [SAiFE-gym: Model-based Environments for Automated Market Making with Concentrated Liquidity](https://arxiv.org/abs/2609.17788) | SAiFE-gym是一个基于Python的向量化模拟环境集合，用于研究带集中流动性的恒定乘积市场中的自动做市问题，可扩展支持高维强化学习工作流，以应对市场参数的不确定性。 |
| [^176] | [Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction](https://arxiv.org/abs/2609.17763) | 本文提出三种模块化深度学习增强机制（风坡条件化注意力偏置、物理特征检索增强的输出校正和火点条件化双流门控），实现了可审计、可解释的次日野火蔓延预测。 |
| [^177] | [Derivative-Free Structured Updates for Muon](https://arxiv.org/abs/2609.17759) | 提出一种无导数框架，通过结构化有限差分（尤其是随机秩一探测）构建Muon风格的参数更新，使Muon在梯度不可用或不可靠时依然可用，并能以较少的函数求值次数换取精度可接受的更新。 |
| [^178] | [SAM-on-the-Curve: Sharpness-Aware Mode Connectivity for Robust Weight-Space Interpolation](https://arxiv.org/abs/2609.17748) | 该论文提出锐利模式连接，将模式连接重新表述为邻域鲁棒的路径优化问题，通过一阶锐度感知近似使连接曲线的整个局部邻域保持低损失平坦性，从而在分布偏移下获得更鲁棒的权重空间插值。 |
| [^179] | [REVERSAL-BENCH: A Reversibility Axis and Reset Oracle for Measuring the Reset-Free RL Cliff](https://arxiv.org/abs/2609.17745) | REVERSAL-BENCH 通过可连续调节的可逆性参数和真值重置预言机，揭示了免重置强化学习存在一个“可逆性悬崖”——随着环境不可逆性增强，免重置智能体会不可避免地陷入无法恢复的状态。 |
| [^180] | [Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC](https://arxiv.org/abs/2609.17738) | 该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。 |
| [^181] | [Machine learning kinetics from molecular dynamics data](https://arxiv.org/abs/2609.17736) | 本综述总结了利用自监督机器学习方法从分子动力学数据中估计承诺概率等关键动力学统计量的现代技术，并建立了连接生成元偏微分方程、变分原理、马尔可夫状态模型与神经网络的统一算子理论框架。 |
| [^182] | [FAME: An FPGA-Based Platform for Approximate Multipliers Evaluation with Pattern-Guided DNN Retraining](https://arxiv.org/abs/2609.17730) | 本文提出FAME，一个基于FPGA的近似乘法器评估平台，通过在硬件中直接实现近似乘法器并结合模式引导的DNN重训练，大幅加速了DNN推理中近似乘法器的精度评估与误差缓解流程。 |
| [^183] | [Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays](https://arxiv.org/abs/2609.17726) | 该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。 |
| [^184] | [Decoding Extrahepatic Targeting of Lipid Nanoparticles with Interpretable Machine Learning](https://arxiv.org/abs/2609.17721) | 本研究开发了一个可解释机器学习框架，利用涵盖476种静脉注射LNP制剂的文献数据集，预测脂质纳米颗粒在肝脏与肝外组织中的蓄积分布，并从中提取出实现肝外RNA递送的分子设计规则。 |
| [^185] | [Composite-Gradient Learning for Shared Control Authority Between Deep Reinforcement Learning and Model Predictive Control](https://arxiv.org/abs/2609.17697) | 本文提出了一种复合梯度学习（CGL）方法，通过将DRL与MPC的控制输入表示为联合动作并显式考虑两者的交互，将MPC控制器深度集成到DRL训练过程中，突破了传统方法将MPC仅视为环境一部分的局限。 |
| [^186] | [Accelerating Diffusion Sampling via Speculative Draft Trees](https://arxiv.org/abs/2609.17691) | 该论文的核心创新是提出“草稿树”方法，将扩散模型中的投机采样与相对熵编码（REC）相联系，用树状候选结构取代线性链式草稿，并采用贪心拒绝采样作为草稿-目标耦合，从而提高接受率、减少昂贵的目标函数评估，实现扩散采样加速。 |
| [^187] | [The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention](https://arxiv.org/abs/2609.17686) | 该论文的核心创新是论证三项看似独立的LLM可靠性研究发现（推理强化学习破坏工具可靠性表征、安全约束下大小模型的差异化表现、以及缺乏“我不知道”功能的系统必然产生无穷幻觉）实际上共同指向同一项缺失能力——校准弃答。 |
| [^188] | [DSD: Learning Diverse and Reusable Motor Skills via Diffusion Skill Discovery](https://arxiv.org/abs/2609.17682) | 该论文提出DSD方法，利用扩散模型进行技能发现，为模拟角色学习多样且可复用的运动技能，从而克服高维控制问题中边际状态熵难以直接估计的难题。 |
| [^189] | [Efficient Robust Learning at the Information-Theoretic Limit](https://arxiv.org/abs/2609.17655) | 本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。 |
| [^190] | [Regularized Least Squares Training of Quadratic Neural Networks with Applications to System Identification](https://arxiv.org/abs/2609.17654) | 本文提出一种正则化最小二乘法训练二次神经网络，可得到权重的闭式解析解及其对数据误差的灵敏度表达式，避免了反向传播易陷入局部极小值的问题并显著降低计算时间。 |
| [^191] | [Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents](https://arxiv.org/abs/2609.17653) | 提出免训练框架EvoSkill-GUI，将GUI智能体的技能构建为包含可执行计划与故障恢复规则的结构化多文件包，使技能能够在部署时根据执行反馈自我修订，从而应对动态界面导致的计划失效问题。 |
| [^192] | [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://arxiv.org/abs/2609.17652) | Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。 |
| [^193] | [Robust and Efficient AI Frameworks for Scalable Material Design and Property Prediction](https://arxiv.org/abs/2609.17646) | 本论文提出CrysXPP和CrysGNN等AI框架，利用无监督图自编码和大规模自监督图预训练技术加速晶体材料发现，显著降低了对昂贵DFT计算和大量标注数据的依赖。 |
| [^194] | [Rethinking Domain Specialization for Open-Ended Scientific Reasoning in Astronomy Language Models](https://arxiv.org/abs/2609.17644) | 该研究通过天文学奥赛问答基准发现，强大的通用大模型在开放式科学推理中表现优于领域专业化模型，表明领域微调的价值应被视为取决于具体任务和部署环境的选择。 |
| [^195] | [Lecture notes on Physics Informed Neural Networks, Neural Operators, and their applications](https://arxiv.org/abs/2609.17638) | 这套讲义系统介绍了物理信息神经网络（PINN）与神经算子的概念及其在PyTorch和NVIDIA PhysicsNeMo中的实现方法，并涵盖模型混合、傅里叶神经算子和物理信息Kolmogorov-Arnold网络（PIKANs）等前沿主题，用于解决工程、物理和石油储层等领域的实际问题。 |
| [^196] | [What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization](https://arxiv.org/abs/2609.17637) | 一项预注册的跨六十个系统配置的验证研究证实，通过证据掩蔽限制模块可见信息能显著提升系统在两步和三步组合任务上的泛化准确率。 |
| [^197] | [Democratizing Clinical Tumor Whole Genome Sequencing: 18-hour End-to-end Analysis via Trillion-parameter Large Language Models Locally Deployed on Consumer-grade Hardware](https://arxiv.org/abs/2609.17620) | 该研究提出完全本地化的低资源框架，在单台消费级笔记本电脑上稳定部署万亿参数生物医学大语言模型，实现18小时内完成从FASTQ原始数据到临床级报告的肿瘤全基因组测序全流程分析，准确性媲美A100集群。 |
| [^198] | [Stability-Constrained Approximation in Spline KANs: Exact Layer Balancing and Budget-Compatible Saturation](https://arxiv.org/abs/2609.17619) | 该论文在严格逐层Lipschitz预算约束下研究深度样条KAN的逼近理论，精确求解了有限深度对角层平衡问题（给出最优层预算的闭式解与单遍最小化算法），并提出了保持预算约束的构造性样条离散化定理。 |
| [^199] | [Timbre Analysis of the Hulusi, a Southwestern Chinese Free-Reed Instrument, using Machine Learning](https://arxiv.org/abs/2609.17612) | 本研究利用机器学习模型分析葫芦丝的音色特征，发现谱质心、尖锐度和分形相关维数这三种心理声学特征能够有效形成音高聚类。 |
| [^200] | [Making Political Text Scaling Comparable: Infrastructure and Hyperparameter Sensitivity for 17 Algorithms](https://arxiv.org/abs/2609.17602) | 本文通过涵盖17种算法、5,537次实验和约425万个立场估计的大规模比较实验，论证了政治文本理想点估计方法应被视为可配置的测量流程而非固定算法，并发现绝大多数算法的估计结果对超参数选择并不敏感。 |
| [^201] | [Decentralized Optimal Equilibrium Learning Over Dynamic Networks](https://arxiv.org/abs/2609.17601) | 该论文提出了一种适用于动态通信网络的去中心化最优均衡学习算法，智能体通过交换带时间戳的表格数据与时间多数重建机制学习社会最优均衡，并在功利主义和比例公平社会福利目标下实现了有限时间对数遗憾保证。 |
| [^202] | [Structure is not mechanism: high-gain gated-FFN rows across text and genomic foundation models](https://arxiv.org/abs/2609.17599) | 该研究通过对文本和基因组基础模型中高增益门控FFN行的分析发现，尽管这些结构普遍存在且功能富集，但其几何特性（如谱集中度、算子幅度）并不能预测因果效应，证明“结构极端性”并非可迁移的功能机制。 |
| [^203] | [Prior-Free Competitive Ratios for Improving Bandits: Scale, Curvature and Horizon Are Free, but Not Jointly Under Noise](https://arxiv.org/abs/2609.17595) | 本文证明在改进型多臂老虎机问题中，一个简单的“探测-承诺”算法无需尺度先验即可达到 4√3·√k 的竞争比（消除了此前已知的对数因子），并确定了任意时间范围（包括未知时间范围）下的最优竞争比 Θ(√k + k/T)。 |
| [^204] | [When the Gradient Sees Rank: Provable Necessity, Causal Recruitment, and Composition in Trained Matrix Memories](https://arxiv.org/abs/2609.17594) | 本文证明了基于梯度的训练能够学习到矩阵记忆存储和组合关联所需的最小秩——学习到的有效秩随键值绑定数量 K 严格递增，秩上限在 k=K 附近引发恢复相变，且训练出的算子在多次自组合应用后仍保持近乎完美的恢复率。 |
| [^205] | [Generic Characteristic-Zero Equivalence Between Derivative B\'ezout Inversion and Multipoint Evaluation](https://arxiv.org/abs/2609.17578) | 该论文证明了在特征零域的泛型计算模型中，计算多项式规范化贝祖对（求解 sZ+tZ'=1）与多点多项式求值及插值问题在复杂度上等价，其核心是一种能在近线性时间内从贝祖对重构多项式的显式微分方法。 |
| [^206] | [R\'enyi Tracking Bounds for Langevin Dynamics with Moving Targets](https://arxiv.org/abs/2609.17577) | 该论文首次建立了具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界，并将其应用于基于连续Moreau包络的非光滑采样，给出了显式的参数选择和复杂度保证。 |
| [^207] | [Temperon: Full-Time SAM Quality at a Third Less Wall-Clock](https://arxiv.org/abs/2609.17575) | 提出Temperon方法：训练前43%使用普通SGD、仅将最后的余弦退火阶段交给SAM包装的Muon优化器，在多个基准上达到与全天候SAM相同的精度，同时节省约三分之一训练时间。 |
| [^208] | [GroupKV: Hierarchical KV Cache Management for Long-Context Diffusion LLM Inference](https://arxiv.org/abs/2609.17573) | GroupKV针对长上下文扩散大语言模型推理提出了轻量级的层次化KV缓存管理系统，利用同一生成块内token访问高度重叠且集中的上下文区域这一特性，通过组级稀疏选择有效缓解KV缓存膨胀与卸载开销问题。 |
| [^209] | [Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Metropolitan Museum Archives](https://arxiv.org/abs/2609.17572) | 本研究建立了区分算法偏见与档案元数据混淆因素的受控审计框架，并发现CLIP视觉-语言模型在评估大都会博物馆艺术品时未表现出统计学显著的性别偏见。 |
| [^210] | [Where Grokking Happens: Distributed Utility and Fourier Recoding Without a Module Switch](https://arxiv.org/abs/2609.17571) | 该论文通过“转换博弈”方法发现，Transformer中的顿悟现象并非发生在特定模块的切换（如MLP转向注意力），而是以分布式方式实现——主要通过块0注意力偏置与块1 MLP等现有分布式电路的傅里叶频谱重编码来完成从记忆到泛化的转变。 |
| [^211] | [Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs](https://arxiv.org/abs/2609.17564) | 本文提出一种无需训练的三指标路由框架，利用空间复杂度、句法密度和类符-形符比三个CPU侧信号，结合显存余量等硬件物理指标，在原始、神经压缩和词法三种流水线间动态选择，解决了消费级GPU上RAG部署中的“压缩悖论”问题。 |
| [^212] | [BLADE: ReliaBle Dynamic Hardware-Aware SNN-ANN Boundary SeLection for Event-BAseD Object DEtection](https://arxiv.org/abs/2609.17562) | 提出BLADE，首个面向动态混合SNN-ANN网络的可靠性感知边界选择方法，通过在设计空间探索中引入分层统计故障注入，联合优化SNN-ANN边界与ANN早退配置，在可靠性、精度、执行时间和能耗之间取得平衡，并在事件驱动目标检测中实现0.691的mAP 0.5。 |
| [^213] | [Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds](https://arxiv.org/abs/2609.17560) | 论文提出DISCERN协议，利用“模型间风险差仅存在于分歧输入上且无需标签即可观测”这一关键性质，通过零标签层与仅标注分歧样本的审计层两层序贯协议，为模型更新提供无回归认证，并证明标签复杂度为rho²/ε²，相比不考虑配对关系的审计器可节省1/ρ的标注成本。 |
| [^214] | [WARD: Runtime Workload-Adaptive Vision TRansformer Framework for Dependable Edge AI](https://arxiv.org/abs/2609.17556) | WARD是一个运行时自适应的视觉Transformer框架，通过通道级子网络划分、可靠性感知的持续学习和动态运行模式调度，在动态变化的边缘环境中联合优化性能、容错能力和适应能力。 |
| [^215] | [REQAP: Resilient Weight Packing and Quantization for Edge DNN Acceleration](https://arxiv.org/abs/2609.17555) | 本文提出REQAP方法，通过敏感度驱动的混合精度量化、实现SWAR并行执行的确定性寄存器级权重打包，以及将关键层MSB复制到空闲寄存器空间的选择性位级保护，在边缘DNN加速器上同时实现了高效模型压缩与硬件容错能力。 |
| [^216] | [No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback](https://arxiv.org/abs/2609.17550) | 该研究通过验证协议发现两个小型LLM中不存在可用的线性“屈服方向”，并首次跨模型家族揭示：模型在反驳下放弃正确答案的谄媚比例高达41.8%-43.1%，且哪种反驳方式有效及失败模式均强烈依赖于具体模型家族。 |
| [^217] | [Enhancing Extubation Failure Prediction with LLM-Derived Features from Respiratory Therapy Clinical Notes](https://arxiv.org/abs/2609.17532) | 该论文提出利用大语言模型从自由文本呼吸治疗临床笔记中提取特征，与结构化数据结合后显著提升了拔管失败预测性能，并揭示了既往研究在目标人群定义上的差异如何阻碍模型的泛化能力。 |
| [^218] | [Goal-oriented probabilistic forecasting for dynamic PRB allocation in 5G networks](https://arxiv.org/abs/2609.17297) | 该论文提出一种面向目标的概率预测框架，利用Pinball损失函数训练DeepAR和TFT模型，并根据运营商成本矩阵确定最优分配分位数，从而在5G网络动态PRB分配中降低运营成本，实现服务可靠性与资源效率的平衡。 |
| [^219] | [A unified framework for global and local interpretability using adaptive derivative-ordered random explanation](https://arxiv.org/abs/2609.17171) | 本文提出ADORE方法，利用一阶和二阶导数在统一分析框架内同时实现全局特征重要性与局部样本贡献的可解释性分析，有效捕捉非线性特征-样本交互并精确量化特征影响。 |
| [^220] | [Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories](https://arxiv.org/abs/2609.16827) | 本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。 |
| [^221] | [Geometry of learning dynamics: Gradient descent versus natural gradient on the ridge of optimization](https://arxiv.org/abs/2609.16805) | 论文通过对KLR训练的Hopfield网络统计流形的几何分析，揭示了优化脊上的学习分为两个阶段：标准梯度下降因极端曲率沿振荡的非测地线路径前进，而自然梯度下降遵循理想测地线路径并完全克服了这些不稳定性。 |
| [^222] | [The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action Chunking Transformer](https://arxiv.org/abs/2609.16745) | 该论文在原始代码中取证式地重跑了ACT的CVAE编码器消融实验，发现原始论文中移除编码器导致成功率从35%骤降至2%的结果无法重现，且训练时长和检查点选择方式都可能逆转策略间的表现排序。 |
| [^223] | [Seeing What Matters: Visual Cue Guided Video Planning for Generalizable Robot Navigation](https://arxiv.org/abs/2609.16737) | CueNav通过鸟瞰图和机器人身体等视觉线索引导视频规划，并利用逆动力学模型将视频计划转换为机器人动作，使迷宫导航成功率提升近2倍。 |
| [^224] | [Math for AI safety: an invitation for mathematicians](https://arxiv.org/abs/2609.15289) | 本文按数学领域（逻辑与博弈论、概率论、代数与表示论、分析与几何）向数学家发出邀请，介绍AI安全研究中各领域可贡献的方向，并为每个领域提供一个无需AI背景即可入手的开放问题。 |
| [^225] | [Bridging the Gap in ECG-Based Emotion Recognition: A Unified Evaluation of Deep Learning Models](https://arxiv.org/abs/2609.15055) | 该研究针对基于心电图的情绪识别领域缺乏统一评估标准的问题，引入了ARRC和ARDT两个开源框架，对主流深度学习模型进行了强调跨数据集泛化能力的统一基准评估。 |
| [^226] | [Cross-Block Conditioning in Deep Boltzmann Machines for Statistical Data Fusion](https://arxiv.org/abs/2609.14934) | 提出观测块多预测方法，使深度玻尔兹曼机在统计数据融合场景下（即没有任何样本同时观测两个结果）也能采用判别式准则进行训练，该方法适用于任意缺失模式，实验表明微调后的DBM优于对比方法。 |
| [^227] | [Tackling Failure Modes of PINNs and PIKANs Using Conflict-Free Gradients](https://arxiv.org/abs/2609.14841) | 本文提出归一化的投影梯度手术算法Norm-PCGrad，用以消除区域分解PINNs中残差、边界和界面损失项之间的梯度冲突，在二维和三维问题上达到最先进的求解精度。 |
| [^228] | [Prism-SQA: An Interpretable and Adaptable Neural Framework for Surface Electromyography Quality Assessment](https://arxiv.org/abs/2609.12724) | 本文提出Prism-SQA框架，将sEMG信号质量评估重构为生理感知的源分离与验证过程，通过将信号分解为干净成分和五种污染成分，实现了可解释、可适应且无需重新训练的信号质量评估。 |
| [^229] | [Almost Sure Convergence Analysis of Stochastic Gradient Methods with Clipping and Additive Noise](https://arxiv.org/abs/2609.12119) | 本文证明了带梯度裁剪和加性高斯噪声的SGD在平滑性和一致有界噪声假设及标准步长衰减条件下几乎必然收敛，并将该结果扩展到随机重球法和Nesterov加速梯度等动量变体。 |
| [^230] | [Learning Interaction Kernels from Collective Steady States](https://arxiv.org/abs/2609.12004) | 该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。 |
| [^231] | [MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads](https://arxiv.org/abs/2609.09206) | 该论文发现多模态大语言模型的幻觉源于协同注意力头中信息分布偏离健康平衡状态，而非模态信息的数量或强度，并提出HEAL方法，通过因果噪声干预和反事实双重差分实现头级别信息解耦与校准，以识别和缓解幻觉。 |
| [^232] | [Multi-Task Learning for Sparsely-Labeled Time Series: A Case Study on Cold-Hardiness Modeling](https://arxiv.org/abs/2609.09062) | 本文针对葡萄耐寒性预测中标注数据稀疏且品种间存在差异的难题，提出将不同葡萄品种视为不同任务的多任务学习方法，从而有效利用有限的稀疏标注时间序列数据，提升基于RNN的每日耐寒性预测效果。 |
| [^233] | [Adaptive Anisotropic Attention for Axis-Structured Signals](https://arxiv.org/abs/2609.08788) | 提出自适应各向异性注意力（AAA），将注意力沿电极时间轴分解为时间路径与空间路径并通过门控自适应加权融合，所构建的AXON模型在六个EEG下游任务上优于稠密注意力基线。 |
| [^234] | [Steering Interference Reflects the Model's Defaults, Not the Behavior Directions](https://arxiv.org/abs/2609.06951) | 激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。 |
| [^235] | [Learning Kernels by Alignment for Multiclass Bayes Classification](https://arxiv.org/abs/2609.06474) | 本文证明协作学习本质上是核对齐过程、协作推理等价于核贝叶斯分类，进而用可学习的马氏距离替代余弦相似度将CLaI推广至多类分类，在多个数据集上显著提升了准确率、收敛速度与校准性能。 |
| [^236] | [Calendar-SPCA: Interpretable Representation Learning for Multi-Periodic Electricity Consumption Profiles](https://arxiv.org/abs/2609.06060) | 提出 Calendar-SPCA 日历结构稀疏主成分方法，将日、周、年等多周期日历几何直接嵌入用电数据的低维表示学习中，生成稀疏且局部连贯、可直接解读的载荷模式，并在两个独立智能电表数据集上验证了其有效性。 |
| [^237] | [Spectral-Target Physical Latent Structuring for JEPA-Style World Models](https://arxiv.org/abs/2609.04264) | 该论文发现了JEPA风格世界模型中“物理表示惰性”这一新失败模式，并提出轻量级傅里叶辅助头在训练时对潜在空间施加物理信息结构化约束，在不增加推理成本的情况下有效提升下游规划性能。 |
| [^238] | [Orthogonal Ensembles and Tested Explanations for Performer-Independent Body-Motion Emotion Recognition](https://arxiv.org/abs/2609.02510) | 在留一表演者的困难评估设置下，通过组合十一个误差模式正交的模型，将12类身体运动情感识别的Macro-F1提升11.07个百分点，并提供了一套经过实验验证的解释方法，证明模型决策基于运动的身体区域证据且与拉班动作分析（LMA）属性高度一致。 |
| [^239] | [Which Histories Matter for Time Series Forecasting? Learning Predictive Relevance with Future Supervision](https://arxiv.org/abs/2608.23221) | 该论文提出了一种通过未来监督学习预测相关性来重新排名时间序列检索候选的方法，显著提升了模式检索性能。 |
| [^240] | [Persistent Magnitude Homology for Quantitative Equational Theories](https://arxiv.org/abs/2608.21479) | 本文为定量等式理论的自由代数构造了持久幅度同调这一函子不变量，证明了幅度同调与持久性理论可以通过长度子水平集过滤统一为同一个构造，二者由长正合序列相互补充、各取所长。 |
| [^241] | [A single design choice determines whether machine learning models of materials make physically impossible predictions](https://arxiv.org/abs/2608.18714) | 本文发现一个简单的设计选择——特征是否携带宇称标签——决定了材料机器学习模型是否会做出物理上不可能的预测，并提出了从群论计算“宇称间隙”的标准来预测哪些性质和晶体受影响。 |
| [^242] | [Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization](https://arxiv.org/abs/2608.16492) | 本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。 |
| [^243] | [Breaking the Compression Barrier: Cross-Architecture Compression Boundary Learning via Reverse Regrowth](https://arxiv.org/abs/2608.16010) | 本文提出BRIDGE框架，通过逆向再生策略，先稀疏化模型暴露崩溃区，再选择性恢复关键结构，从而准确找到并突破模型压缩的性能边界。 |
| [^244] | [Deep Divide-and-Reduce in Symbolic Regression](https://arxiv.org/abs/2608.02628) | 该论文提出DDRSR方法，通过对更广泛分解结构的形式化分析，从根本上扩展了AI Feynman方法中表达式分解与归约机制的适用范围，并克服了其依赖暴力搜索的局限。 |
| [^245] | [Simple-regret rates and minimax optimality of fixed-prior expected improvement in Mat\'ern and squared-exponential RKHSs](https://arxiv.org/abs/2607.29245) | 本文证明了在Matérn核和平方指数核的再生核希尔伯特空间中，弱期望改进策略的简单遗憾率达到极小极大最优，分别以 $O(N^{-\nu/d})$ 和指数级速率收敛。 |
| [^246] | [Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting](https://arxiv.org/abs/2607.27106) | 提出HierSTT框架，基于层次化Transformer在单一模型中联合预测医院、地区和国家三个层面的急诊科需求，解决了传统单层级独立预测导致的预测不连贯问题。 |
| [^247] | [Bumblebee: Interleaved Mixed-Layer Building Blocks for Large-Scale Recommendation Systems](https://arxiv.org/abs/2607.24804) | 提出Bumblebee推荐架构，通过交错可堆叠的混合层构建模块将序列建模与特征交互两种范式有机融合，实现了特征模态的早期与反复混合，从而提升大规模推荐系统的表达能力。 |
| [^248] | [Post-Training in Time Series Foundation Models: A Unifying Framework](https://arxiv.org/abs/2607.20002) | 本文提出了一个统一框架，根据干预位置将时序基础模型的后训练方法归纳为参数适配、上下文增强、模型组合、输出处理与不确定性控制、压缩与专门化五大类，并系统分析了各类代表性方法、现有局限与未来方向。 |
| [^249] | [Riemannian Deep Learning: Modules, Networks, and Geometries](https://arxiv.org/abs/2607.19305) | 本论文从可复用神经模块、流形专用网络架构和底层几何设计三个互补视角构建了统一的黎曼深度学习框架，将批归一化和多项逻辑回归等基础组件推广到李群、陀螺群和一般黎曼流形，并为双曲空间、满秩相关矩阵等重要几何表示设计了神经网络。 |
| [^250] | [Optimizing the Preconditioner: A Black-box Online-to-Nonconvex Conversion with Static Regret Minimization Oracles](https://arxiv.org/abs/2607.17607) | 本文提出了一种从随机非凸优化到在线凸优化中静态遗憾最小化的黑盒归约方法，解决了Chen和Hazan（2024）提出的开放问题，并证明任何具有O(√T)遗憾的OCO预言机都能恢复经典的O(T^{-1/2})收敛速率。 |
| [^251] | [Debiasing Text-to-Image Evaluation via Implicit Cultural Alignment Reward Modeling](https://arxiv.org/abs/2607.15740) | 本文提出了一种基于轻量级多模态大语言模型的隐式文化对齐奖励模型，通过跳跃连接交叉注意力机制，解决了文生图评估中文化偏差和实时扩展性问题。 |
| [^252] | [Subjective Risk Decomposition: A New View for Uncertainty Quantification](https://arxiv.org/abs/2607.15196) | 该论文提出将不确定性度量视为主观风险分解的产物而非基本原语，证明了基于严格恰当损失对主观风险进行分解即可推导出认知不确定性与偶然不确定性，从而为不确定性量化提供了统一的理论框架和新范式。 |
| [^253] | [FPGN: Redefining Ultra-Fast Programmable Gate-based Neural Acceleration with Differentiable LUTs](https://arxiv.org/abs/2607.08427) | 该论文提出FPGN框架，通过可微分查找表将FPGA的LUT直接作为可学习神经元，重新定义了超快速可编程门神经网络加速，旨在实现纳秒级推理延迟。 |
| [^254] | [Prior-matched evaluation of operational Earth-observation classifiers: a three-number reporting method demonstrated on Sentinel-1 internal-wave detection](https://arxiv.org/abs/2607.07146) | 论文揭示了类别先验不匹配导致平衡测试集上报告的精确率（0.794）严重高估真实运行精确率（0.192），证明这是评估问题而非训练问题，并提出一种基于三个数字的先验匹配报告方法。 |
| [^255] | [Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement](https://arxiv.org/abs/2606.27409) | 该论文将多智能体LLM系统中的延迟验证建模为带接地节点的延迟共识问题，通过接地拉普拉斯谱分解推导出验证剂量的闭式失稳阈值（延迟为二时为黄金比例的倒数），并基于超模目标给出贪婪(1-1/e)近似的纠错节点最优配置方法。 |
| [^256] | [Bergson: An Open Source Library for Data Attribution](https://arxiv.org/abs/2606.11660) | Bergson 是一个开源数据归因库，支持扩展至超大规模语言模型和预训练数据集，并首次开源实现了 MAGIC、SOURCE 和 TrackStar 三种前沿数据归因方法。 |
| [^257] | [LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models](https://arxiv.org/abs/2606.09430) | 提出了LargeMonitor框架，利用大型预训练基础模型通过解耦的检测模块来监控和自主编排在线无任务持续学习，克服了现有训练耦合方法对分布漂移结构性起源不可知的局限。 |
| [^258] | [From 'May' to 'Is': Certainty Distortion in Language Model Rewriting](https://arxiv.org/abs/2606.07951) | 该研究发现语言模型在重写科学和医学文本时会系统性地改变原文的确定性程度（如把“可能”改成“是”），这种失真影响高达75%的输出且呈不对称性。 |
| [^259] | [KITE: A Tri-Modal Transformer Integrating Text, Images, and Knowledge Graphs for Fake News Detection](https://arxiv.org/abs/2606.07651) | KITE提出了一个三模态Transformer框架，通过跨模态注意力机制联合建模文本、图像和知识图谱表示，突破了以往仅依赖文本-图像融合或将外部知识用作后处理的局限，显著提升了虚假新闻检测能力。 |
| [^260] | [TargetSEC: Plug-and-Play In-the-Wild Speech Emotion Conversion via Arousal-Conditioned Latent Style Diffusion](https://arxiv.org/abs/2606.07293) | TargetSEC提出了一种以说话人身份和连续情感为条件的潜在风格扩散框架，通过在紧凑潜在空间中生成情感风格嵌入，实现了即插即用的野外语音情感转换，在转换效果和语音自然度上均优于或持平现有基线方法。 |
| [^261] | [Libra: Efficient Resource Management for Agentic RL Post-Training](https://arxiv.org/abs/2606.03077) | Libra 是一个面向智能体强化学习后训练的自适应运行时系统，通过因果引导的桶调度等互补机制，解决长尾轨迹主导完成时间以及 rollout 与训练阶段资源需求动态失衡的问题，实现高效资源管理。 |
| [^262] | [On Finite-sample Concentration of Median of Incomplete U-Statistics](https://arxiv.org/abs/2606.00661) | 本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。 |
| [^263] | [Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models](https://arxiv.org/abs/2605.31111) | SD-JEPA将JEPA潜在空间分解为正交的进展子空间和内容子空间，使两种防坍缩力量在不相交的坐标上加性组合而非相互竞争，从而在多个控制基准上提升了世界模型性能。 |
| [^264] | [EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control](https://arxiv.org/abs/2605.16692) | EfficientTDMPC通过动力学模型集成、跨不同展开深度平均回报估计以及对规划器目标施加不确定性惩罚来减少模型与价值网络的估计误差，并结合缓冲区数据新鲜度等实用改进，从而在连续控制任务中实现更高效的模型强化学习，并能更好地利用更高的更新-数据比。 |
| [^265] | [How to Compress KV Cache in RL Post-Training? Shadow Mask Distillation for Memory-Efficient Alignment](https://arxiv.org/abs/2605.06850) | 该论文揭示了KV缓存压缩在RL后训练rollout阶段会引入被优化不稳定性急剧放大的离策略偏差问题，并提出影子掩码蒸馏方法以实现内存高效的模型对齐。 |
| [^266] | [Forecasting Individual NetFlows using a Predictive Masked Graph Autoencoder](https://arxiv.org/abs/2604.20483) | 本文提出一种基于预测性掩码图自编码器的GNN模型，通过滑动窗口构建包含IP、端口和连接节点的异构双向图来预测网络流级别流量，在识别连接所依附的端口和IP方面表现卓越。 |
| [^267] | [Wasserstein Formulation of Reinforcement Learning. An Optimal Transport Perspective on Policy Optimization](https://arxiv.org/abs/2604.14765) | 本文提出了一个基于最优传输理论的强化学习几何框架，将策略视为映入Wasserstein空间的映射，利用黎曼几何结构和Otto微积分构建梯度流并计算能量的梯度与Hessian矩阵，为策略优化提供了形式化的二阶分析工具。 |
| [^268] | [HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark](https://arxiv.org/abs/2604.13954) | 该论文提出了HINTBench，一个针对智能体内在非攻击风险的基准数据集，包含596条平均24步的智能体轨迹，支持风险检测、风险步骤定位和内在失效类型识别三项安全审计任务。 |
| [^269] | [Robust Ultra Low-Bit Post-Training Quantization via Stable Diagonal Curvature Estimate](https://arxiv.org/abs/2604.13806) | DASH-Q通过稳定的对角Hessian曲率估计和迭代加权最小二乘，实现了对采样噪声鲁棒的超低比特后训练量化，在极少量校准数据下于五个LLM模型上平均提升零样本准确率7.01%、最高提升14.01%。 |
| [^270] | [VISTA: Validation-Informed Trajectory Adaptation via Self-Distillation](https://arxiv.org/abs/2604.12044) | VISTA提出了一种在线自蒸馏框架，通过验证信息引导的边际覆盖率得分识别保留专业能力的早期模型状态（专家锚点），并以覆盖率加权方式在线集成这些锚点来正则化训练过程，从而解决“轨迹偏差”导致的优化失败问题并提升模型鲁棒性与泛化能力。 |
| [^271] | [Deep Learning for Sequential Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers](https://arxiv.org/abs/2604.11507) | 本教程以运筹学/管理科学（OR/MS）为核心视角，系统性地连接了深度学习神经架构与不确定性下序贯决策的OR/MS方法，其核心观点是深度学习是对优化的补充而非替代。 |
| [^272] | [Leveraging Complementary Embeddings for Replay Selection in Continual Learning with Small Buffers](https://arxiv.org/abs/2604.08336) | 提出MERS方法，通过基于图的方式融合监督与自监督互补嵌入来选择重放样本，在小缓冲区持续学习中以零额外参数开销显著超越现有最先进的样本选择策略。 |
| [^273] | [Approximation of the Basset force in the Maxey-Riley-Gatignol equations via universal differential equations](https://arxiv.org/abs/2604.08194) | 本文提出利用通用微分方程和神经网络来逼近MaRGE方程中的Basset历史力项，将其转化为可用Runge-Kutta等标准数值方法求解的常微分方程组。 |
| [^274] | [On Dominant Manifolds in Reservoir Computing Networks](https://arxiv.org/abs/2604.05967) | 该论文从理论上证明了储备池计算网络在时间序列预测训练中会涌现出低维主导流形，并将储备池的主导特征值和特征向量与后向动态模态分解矩阵的谱联系起来，从而给出了系统后向时间Koopman算子的有限维近似。 |
| [^275] | [Symmetrizing Bregman Divergence on the Cone of Positive Definite Matrices: Which Mean to Use and Why](https://arxiv.org/abs/2603.28917) | 该论文揭示了正定矩阵锥上对称化Bregman散度的变分原理，证明前向对称化的规范均值是原始空间上的算术平均，而反向对称化的规范均值是对偶空间算术平均的拉回，在常用情形下分别对应算术、对数欧几里得和调和平均。 |
| [^276] | [AMIGO: Agentic Multi-Image Grounding Oracle Benchmark](https://arxiv.org/abs/2603.28662) | AMIGO是一个长时程智能体基准，要求视觉语言模型通过一系列是/否问题在视觉相似的图像画廊中识别隐藏目标，以评估其在不确定性下的问题选择、跨轮次约束跟踪和细粒度判别能力。 |
| [^277] | [Q-BIOLAT: Binary Latent Protein Fitness Landscapes for QUBO-Based Optimization](https://arxiv.org/abs/2603.27526) | 该论文提出Q-BioLat框架，将蛋白质语言模型嵌入转换为紧凑二进制编码并拟合QUBO代理模型用于蛋白质适应度优化，核心发现是逐点预测精度相当的二进制表示可能诱导截然不同的汉明邻域和优化搜索轨迹，强调表示选择对优化行为的决定性影响。 |
| [^278] | [Curvature-aware Expected Free Energy as an Acquisition Function for Bayesian Optimization](https://arxiv.org/abs/2603.26339) | 提出了一种曲率感知的期望自由能采集函数用于贝叶斯优化，该函数统一了上置信界、下置信界和期望信息增益，并具有凹函数的无偏收敛保证，在后悔值和均方误差两个指标上均取得有竞争力的表现。 |
| [^279] | [Stochastic Dimension Zeroth-Order Estimator: Stable and Memory-Efficient Training of PINNs](https://arxiv.org/abs/2603.24002) | 本文提出SDZE框架，通过统一随机空间估计和零阶优化，实现PINNs训练中空间和内存复杂度均与维度无关，解决了高维PDEs中的方差爆炸和内存瓶颈问题。 |
| [^280] | [A Multitask Large Reasoning Model for Molecular Science](https://arxiv.org/abs/2603.12808) | 本文提出一种任务自适应的多专家大型推理模型，通过思维链监督和分子信息引导的强化学习整合化学知识，在10项分子任务上超越20多个大语言模型，总体性能较基础模型提升50.3%，同时保持化学推理的可解释性。 |
| [^281] | [Language-Guided Grasping under Partial Observation for Mobile Manipulation in Field Inspection and Maintenance](https://arxiv.org/abs/2603.07866) | 该论文提出了一种面向腿式移动操作机器人的语言引导抓取流水线，结合开放词汇检测、可提示分割、深度补偿与点云补全技术，在部分观测条件下实现适用于海上检测与维护场景的6自由度抓取。 |
| [^282] | [Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results](https://arxiv.org/abs/2603.02948) | 本文提出利用领域感知傅里叶特征对输入空间进行位置编码，从而消除显式边界条件损失项和损失平衡方案的需求，简化训练并降低计算成本，同时开发了基于LRP的可解释性框架以提取输入空间的相关性归因分数。 |
| [^283] | [MINT: Multimodal Imaging-to-Speech Knowledge Transfer for Early Alzheimer's Screening](https://arxiv.org/abs/2602.23994) | 该论文提出MINT三阶段框架，通过将MRI衍生的生物标志物知识迁移到语音模型中，实现了无需影像设备、具备生物学依据的早期阿尔茨海默病无创语音筛查。 |
| [^284] | [Sparse Bayesian Modeling of EEG Channel Interactions Improves P300 Brain-Computer Interface Performance](https://arxiv.org/abs/2602.17772) | 该论文提出一种稀疏贝叶斯时变回归框架，通过松弛阈值高斯过程先验显式建模脑电通道间成对交互并进行时间特征选择，在55人P300拼写数据集上将中位字符级准确率提升至96.4%，同时保证了模型的可解释性。 |
| [^285] | [Bayesian Quadrature](https://arxiv.org/abs/2602.16218) | 本综述首次系统全面地梳理了贝叶斯求积方法，涵盖其数学基础、建模-推断-采样三维分类体系、理论保证、数值实验对比以及实际应用中的挑战与局限性。 |
| [^286] | [Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT](https://arxiv.org/abs/2602.11220) | 提出一种用强化学习训练的轻量级LoRA改写策略，在任务一致性约束下优化问答分布对齐与语义多样性，从而修补下游监督数据与模型生成分布之间的失配，缓解SFT中的灾难性遗忘。 |
| [^287] | [TabICLv2: A better, faster, scalable, and open tabular foundation model](https://arxiv.org/abs/2602.11139) | TabICLv2 通过新型合成数据生成引擎、可扩展 softmax 注意力架构和 Muon 优化器预训练协议三大创新，成为一个无需任何调优即可超越 RealTabPFN-2.5 的更快、更可扩展的开源表格基础模型。 |
| [^288] | [Bypassing the Rationale: Causal Auditing of Implicit Reasoning in Language Models](https://arxiv.org/abs/2602.03994) | 该论文提出基于激活修补的因果审计指标——思维链中介指数（CMI），发现语言模型中思维链的真实因果影响往往局限于狭窄的“推理窗口”，且存在 CoT 文本看似合理但实际计算被绕过的情况，表明思维链输出并不能忠实反映模型内部的推理过程。 |
| [^289] | [Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design](https://arxiv.org/abs/2602.01898) | 论文针对基于方差采集准则的高斯过程主动学习的两大缺陷——后验方差与观测内容无关以及边界处方差膨胀导致的过度采样，提出了修正方案，通过重构驱动的设计密度与基于后验均值的免训练变形，使采样更集中于目标函数变化剧烈的区域。 |
| [^290] | [Variational Approach for Job Shop Scheduling](https://arxiv.org/abs/2602.00408) | 本文首次将变分推断引入作业车间调度问题，提出VG2S框架，通过基于ELBO的变分图编码器将表示学习与策略优化数学解耦，从而解决传统深度强化学习方法的训练非平稳性和泛化能力受限问题。 |
| [^291] | [Finite-Sample Unbiased Variance of MMD under Unbalanced Sampling: Exact Estimation and Quasi-Linear Computation](https://arxiv.org/abs/2601.13874) | 该论文推导了非平衡采样下MMD方差的有限样本无偏估计量，并通过拉普拉斯核的递归前缀-后缀累加方案将计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N \log N)$、内存仅需 $\mathcal{O}(N)$。 |
| [^292] | [Performance and Complexity Trade-off Optimization of Speech Models During Training](https://arxiv.org/abs/2601.13704) | 该论文提出在训练过程中直接优化语音模型性能与计算复杂度之间的权衡，突破了传统随机梯度下降只能优化可微函数、无法直接优化模型结构复杂度的限制。 |
| [^293] | [Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings](https://arxiv.org/abs/2601.11262) | 该论文提出利用歌词作为翻唱歌曲间的强不变量，通过歌词对齐的音频嵌入实现高效可扩展的音乐翻唱检索，从而避免现有方法对复杂音频处理流程和高计算资源的需求。 |
| [^294] | [FuseFi: Combining Irregularly Sampled CSI from Diverse Communication Packets and Frequency Bands for Wi-Fi Sensing](https://arxiv.org/abs/2512.22143) | FuseFi提出了一种Wi-Fi通信感知一体化框架，通过融合多频带、多类型通信数据包中不规则采样的CSI并利用时间感知注意力模型，无需注入探测数据包即可实现零感知通信开销的高效Wi-Fi感知。 |
| [^295] | [Implicit Bias and Invariance: How Hopfield Networks Efficiently Learn Graph Orbits](https://arxiv.org/abs/2512.14338) | 本文证明Hopfield网络在有限置换轨道上训练时，仅需多项式数量的随机样本即可隐式学习对称不变性并记忆所有轨道元素。 |
| [^296] | [NeuroSketch: A Practical Design Recipe for Neural Decoding](https://arxiv.org/abs/2512.09524) | 该研究提出了NeuroSketch，一种实用的神经解码设计配方，通过系统比较九种基础架构确定CNN-2D为最优骨干，并结合宏观层面的渐进式特征图扩展、早期下采样与微观层面的分组卷积优化，构建了轻量高效的神经解码模型。 |
| [^297] | [Understanding the Staged Dynamics of Transformers in Learning Latent Structure](https://arxiv.org/abs/2511.19328) | 该研究通过Alchemy基准的受控实验发现，transformer以离散阶段学习潜在结构的不同组成部分，并存在不对称性——模型能稳健地组合基本转换规则，却难以分解复杂示例来发现中间转换。 |
| [^298] | [An operator splitting analysis of Wasserstein--Fisher--Rao gradient flows](https://arxiv.org/abs/2511.18060) | 本文定量分析了求解 WFR 梯度流时 W-FR 算子分裂的顺序与步长的影响，并出人意料地证明：合理选择步长和算子顺序时，分裂方案可以比精确 WFR 流更快地收敛到目标分布。 |
| [^299] | [FairLRF: Achieving Fairness through Sparse Low Rank Factorization](https://arxiv.org/abs/2511.16549) | 本文提出FairLRF框架，创新性地将奇异值分解（SVD）从传统的模型压缩工具转变为公平性增强工具，通过稀疏低秩分解在不显著牺牲模型准确率和计算资源的情况下有效提升深度学习模型的公平性。 |
| [^300] | [GeoCrossBench: Cross-Band Generalization for Remote Sensing](https://arxiv.org/abs/2511.02831) | 本文提出GeoCrossBench基准和χViT基线模型，通过新的跨波段泛化评估协议解决遥感领域新旧卫星波段不一致的问题，降低支持新卫星所需的模型重训练成本。 |
| [^301] | [PitchFlower: A flow-based neural audio codec with pitch controllability](https://arxiv.org/abs/2510.25566) | PitchFlower是一种基于流的神经音频编解码器，通过在训练时对输入F0轮廓进行展平和随机偏移的简单扰动策略实现音高解耦，达到了DSP级别的精确音高控制，同时保持接近最先进神经方法的高音频质量。 |
| [^302] | [Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge](https://arxiv.org/abs/2510.19266) | 本文提出CAB蒸馏框架，利用轻量级注意力桥将Transformer教师模型的注意力相关表示以token级中间监督的方式迁移给Mamba等状态空间学生模型，实现了数据高效的跨架构知识蒸馏。 |
| [^303] | [ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs](https://arxiv.org/abs/2509.24115) | 本文提出了ADAPT，一种抛弃图表示、将原子作为token并显式建模所有空间成对原子相互作用的轻量级Transformer机器学习力场，有效解决了图神经网络力场在点缺陷建模中的过度平滑和长程相互作用表征不佳问题。 |
| [^304] | [A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models](https://arxiv.org/abs/2509.19276) | 提出了一种免训练的扩散正则化Wasserstein梯度流方法（DWGF），利用预训练潜在扩散模型作为先验来求解不适定逆问题。 |
| [^305] | [Learning Contact Dynamics through Touching: Action-conditional Graph Neural Networks for Robotic Peg Insertion](https://arxiv.org/abs/2509.12151) | 该论文提出了一种动作条件图神经网络模型，通过机器人随机触摸环境的自监督学习来预测接触密集操作中的运动和力-力矩，在仿真中于未见几何形状的插孔任务中达到98%成功率，在现实世界中比系统辨识的MuJoCo模型高出45%。 |
| [^306] | [Learning Magnetic Order Classification from Large-Scale Materials Databases](https://arxiv.org/abs/2509.05909) | 该研究开发了基于简单成分、结构和电子描述符的机器学习分类器，能以超过92%的准确率对磁性材料的传播矢量磁序进行分类，并揭示了Materials Project数据库中存在的系统性铁磁偏差。 |
| [^307] | [Solving Conic Programs over Sparse Graphs using a Variational Quantum Approach: The Case of the AC Optimal Power Flow](https://arxiv.org/abs/2509.00341) | 提出了一种变分量子方法，通过两个参数化量子电路分别编码原始变量与对偶变量，将锥规划（含二次约束二次规划和半定规划）的求解转化为量子可观测量期望值的拉格朗日优化问题，并以交流最优潮流为应用案例。 |
| [^308] | [Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks](https://arxiv.org/abs/2508.11584) | 提出了视觉感知引擎（VPEngine），一个通过共享基础模型骨干网络和并行任务专用模型头实现高效GPU多任务视觉推理的模块化框架，可消除计算冗余并支持动态任务优先级调整，适用于资源受限的机器人平台。 |
| [^309] | [Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies](https://arxiv.org/abs/2506.24048) | 本文建立了基于共识的优化（CBO）中的共识跳跃与自然进化策略（NES）之间的理论联系，并通过实验证明在黑盒对抗攻击中CBO在某些场景下可以超越NES和其他进化策略。 |
| [^310] | [DPG loss functions for learning parameter-to-solution maps by neural networks](https://arxiv.org/abs/2506.18773) | 本文提出基于超弱间断间断Petrov-Galerkin（DPG）离散化的变分正确残差损失函数，为神经网络学习参数依赖偏微分方程的参数到解映射提供严格的精度认证，且该方法可推广至所有具有稳定DPG公式的问题。 |
| [^311] | [Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy](https://arxiv.org/abs/2505.03590) | 该论文提出了一种基于Sylvester归一化流的贝叶斯推断框架，结合融入物理先验知识的解码器，用于磁共振波谱中代谢物浓度的可靠定量化。 |
| [^312] | [BOOM: Benchmarking Out-Of-distribution Molecular Property Predictions of Machine Learning Models](https://arxiv.org/abs/2505.01912) | 本文提出了BOOM，一个基于化学信息学的分子性质分布外预测系统性基准，通过对超过150种模型-任务组合的评估，揭示了没有任何现有模型（包括化学基础模型）能在所有任务上实现强大的分布外泛化，最佳模型的分布外误差仍比分布内高出3倍。 |
| [^313] | [Explainable Graph-theoretical Machine Learning with Application to Alzheimer's Disease Prediction](https://arxiv.org/abs/2503.16286) | 本文提出了一种可解释的图论机器学习框架XGML，通过构建个体大脑代谢图并识别最具预测性的子图，实现了基于FDG-PET数据对阿尔茨海默病多变量结果的个体化预测。 |
| [^314] | [Interpretable Retinal Disease Prediction Using Biology-Informed Heterogeneous Graph Representations](https://arxiv.org/abs/2502.16697) | 该论文提出了一种新颖的生物学信息异构图表示方法，以人类可解释的方式建模视网膜血管段、毛细血管间区域和中央凹无血管区，在保留OCTA图像丰富信息的同时实现了糖尿病视网膜病变分期的可解释预测。 |
| [^315] | [A Survey on Bridging EEG Signals and Generative AI: From Image and Text to Beyond](https://arxiv.org/abs/2502.12048) | 本综述系统梳理了2017至2025年间利用生成式人工智能（GAN、VAE、Transformer、扩散模型等）将脑电信号转化为图像、文本和音频的研究进展，涵盖数据集、特征编码技术、评估指标及该领域面临的主要挑战。 |
| [^316] | [Active Learning Enables Generation of Molecules that Advance the Known Pareto Front](https://arxiv.org/abs/2501.02059) | 该论文提出了一种基于主动学习的闭环分子生成流水线，通过在新的量子化学模拟数据上迭代重训练，成功生成了性质超越训练分布的分子，并将分布外分子分类准确率提高了79%。 |
| [^317] | [Uncertainty measurement for complex event prediction in safety-critical systems](https://arxiv.org/abs/2411.01289) | 该论文提出了一种结合机器学习、敏感性分析和保形预测的方法（ML_CP），用于度量安全关键嵌入式系统中复杂事件预测的不确定性。 |
| [^318] | [Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism](https://arxiv.org/abs/2409.09253) | 该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。 |
| [^319] | [DRL-AdaPart: DRL-Driven Adaptive STAR-RIS Partitioning for Fair and Efficient Resource Utilization](https://arxiv.org/abs/2407.06868) | 提出了一种基于深度强化学习的自适应STAR-RIS单元分区方法DRL-AdaPart，通过联合优化相移与子表面分配变量并引入惩罚项智能停用多余单元，在保证资源高效利用的同时，为静态和移动用户提供公平且高速的数据速率。 |
| [^320] | [Breaking the $T^{2/3}$ Barrier for Sequential Calibration](https://arxiv.org/abs/2406.13668) | 本文首次突破了序贯校准问题中 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界，改进了这一停滞二十余年的经典界限。 |
| [^321] | [Multi-Objective Hyperparameter Search via Damped Gauss--Newton Optimization](https://arxiv.org/abs/2401.03580) | 本文提出一种基于阻尼高斯-牛顿优化的多目标超参数搜索方法，利用有限差分雅可比矩阵和Tikhonov正则化实现有向的联合参数更新，在欠定情况下以远少于网格搜索的试验次数达到同等的最佳验证精度。 |
| [^322] | [Topology-enhanced machine learning for speech signal processing](https://arxiv.org/abs/2311.15210) | 本文提出了一种透明的拓扑特征捕捉方法TopCap，通过将时间序列的拓扑特征融入神经网络，在语音信号处理任务中提升了准确性、抗噪鲁棒性、稳定性与可解释性。 |
| [^323] | [Generalizing Adam to Manifolds for Efficiently Training Transformers](https://arxiv.org/abs/2305.16901) | 本文利用齐性流形（如Stiefel流形、辛Stiefel流形和Grassmann流形）所具有的全局切空间（李子空间）表示这一特殊结构，提出了一种将Adam优化器完整推广到流形上的新方法，从而实现对Transformer的高效训练。 |
| [^324] | [Limits of Transfer Learning](https://arxiv.org/abs/2006.12694) | 该论文在算法搜索框架下证明了迁移学习的若干理论极限，表明迁移信息必须经过谨慎选择并与目标问题存在依赖关系，同时算法的概率变化程度决定了其性能改进的上限。 |
| [^325] | [Reliable Learning for Test-time Attacks and Distribution Shift.](http://arxiv.org/abs/2304.03370) | 本文提出了可靠的学习方法以抵御测试时攻击和分布偏移，在测试时引入了新的可靠性保障方法，确保预测结果正确。同时，该学习方法能够适应任意测试点，具有非常好的可靠性。 |

# 详细

[^1]: 大语言模型偏好对齐的零阶范式

    A Zeroth-Order Paradigm for LLM Preference Alignment

    [https://arxiv.org/abs/2609.19144](https://arxiv.org/abs/2609.19144)

    本文提出了基于比较预言机的零阶偏好对齐方法 ComPO，能从微小似然边距的偏好对中提取方向性信息而无需优化可微偏好损失，并建立了收敛保证且引入了具备反向 KL 控制的在线版本。

    

    直接偏好对齐方法因其计算和内存效率，被广泛用于将大语言模型（LLM）与人类偏好进行对齐。然而，似然位移现象促使人们探索从具有微小似然边距的偏好对中提取信息的替代方法。在本文中，我们提出并分析了基于比较的偏好优化（ComPO），这是一种基于比较预言机的零阶对齐方法。ComPO 从这些偏好对中提取方向性信息，而无需直接在其上优化可微分的偏好损失。我们在平滑性、梯度稀疏性以及预言机与潜在目标兼容性的假设下，为其基本离线方案建立了收敛保证。我们进一步提出了在线 ComPO，它保留了离线比较机制，并利用无标签的策略生成相对于参考策略进行反向 KL 散度控制。基于覆盖度的视角……

    arXiv:2609.19144v1 Announce Type: cross  Abstract: Direct preference alignment methods are widely used to align large language models (LLMs) with human preferences because of their computational and memory efficiency. However, likelihood displacement motivates alternative ways to extract information from preference pairs with small likelihood margins. In this paper, we propose and analyze Comparison-based Preference Optimization (ComPO), a zeroth-order alignment method based on comparison oracles. ComPO extracts directional information from these pairs without directly optimizing a differentiable preference loss on them. We establish a convergence guarantee for its basic offline scheme under smoothness, gradient sparsity, and compatibility between the oracle and a latent objective. We further introduce online ComPO, which retains the offline comparison mechanism and uses unlabeled policy generations for reverse-KL control relative to a reference policy. Following the coverage perspecti
    
[^2]: 历史依赖日志记录下离线策略评估的指数级困难性

    Exponential Hardness of Off-Policy Evaluation under History-Dependent Logging

    [https://arxiv.org/abs/2609.19135](https://arxiv.org/abs/2609.19135)

    本文证明当日志记录器依赖历史时，即使所有覆盖条件的常数与时间跨度无关，离线策略评估仍需要指数级数量的日志轨迹，其根源在于重置操作会擦除决定目标策略价值的关键未知转移信息。

    

    一个日志数据集能否频繁访问每个隐藏状态，却对目标策略的价值呈指数级地缺乏信息量？我们证明，当日志记录器依赖于历史时，这种情况是可能发生的。对于每个时间跨度 H ≥ 3，我们构建了两个部分可观测马尔可夫决策过程（POMDP），每个阶段至多有两个潜在状态、三个动作，以及一个具有三个记忆状态的共同日志记录器。动作覆盖、信念覆盖以及两个行为边际结果揭示条件的所有常数均与 H 无关。然而，即使两个候选模型均为已知，在置信度 1-δ（0 < δ ≤ 1/4）下，将一个已知的确定性目标策略评估到精度 1/8 仍需要 Θ((3/2)^H log(1/δ)) 条日志轨迹。其机制很简单：重置操作会擦除决定目标价值的未知转移。我们精确刻画了由此产生的统计实验，并得到了与之匹配的最优估计器。一个有向双车道网格世界实现了这一构造。

    arXiv:2609.19135v1 Announce Type: new  Abstract: Can a logged dataset visit every hidden state frequently and still be exponentially uninformative about a target policy's value? We show that it can when the logger depends on history. For every horizon $H \ge 3$, we construct two POMDPs with at most two latent states per stage, three actions, and a common logger with three memory states. Action coverage, belief coverage, and two behavior-marginal outcome-revealing conditions all have constants independent of $H$. Nevertheless, evaluating a known deterministic target policy to accuracy $1/8$ requires $\Theta((3/2)^H \log(1/\delta))$ logged episodes at confidence $1-\delta$, for $0 < \delta \le 1/4$, even when both candidate models are known. The mechanism is simple: a reset erases the unknown transition that determines the target value. We characterize the resulting statistical experiment exactly and obtain a matching optimal estimator. A directed two-lane gridworld realizes the construc
    
[^3]: 双过程语言代理的认知扩展：交互环境中的记忆与自我反思

    Cognitive Extensions for Dual-Process Language Agents: Memory and Self-Reflection in Interactive Environments

    [https://arxiv.org/abs/2609.19128](https://arxiv.org/abs/2609.19128)

    通过为双过程代理SwiftSage引入自适应记忆模块（AMM）和自我反思模块（SRM）两个认知扩展，显著提升了语言代理在ScienceWorld交互环境中的表现，完整系统实现了最佳平均得分64.62、成功率43.17%，其中SRM是最强的独立贡献者。

    

    语言代理在交互环境中仍然表现脆弱，其成功需要长时程状态跟踪、有效的动作执行以及从失败步骤中恢复。我们扩展了SwiftSage——一个将快速动作提议器与较慢规划器相结合的双过程代理——采用两个模块化的认知扩展：用于显著性门控情景存储和触发式检索的自适应记忆模块（AMM），以及用于有界执行时验证和纠正性干预的自我反思模块（SRM）。两个模块均作为功能标志扩展实现于相同的执行基础上，从而支持在ScienceWorld上进行受控消融实验。在四种配置（基线、基线+AMM、基线+SRM和完整系统）中，完整系统取得了最佳的平均最终得分（64.62）、成功率（43.17%）和成功步骤效率（19.33步），其中SRM是最强的独立贡献者。这些结果表明，执行……

    arXiv:2609.19128v1 Announce Type: new  Abstract: Language agents remain brittle in interactive environments, where success requires long-horizon state tracking, valid action execution, and recovery from failed steps. We extend SwiftSage, a dual-process agent that combines a fast action proposer with a slower planner, using two modular cognitive extensions: an Adaptive Memory Module (AMM) for salience-gated episodic storage and trigger-driven retrieval, and a Self-Reflection Module (SRM) for bounded execution-time validation and corrective intervention. Both modules are implemented as feature-flagged extensions over the same execution substrate, enabling controlled ablations on ScienceWorld. Across four configurations---baseline, baseline+AMM, baseline+SRM, and the full system---the full system achieves the best mean final score (64.62), success rate (43.17%), and successful-step efficiency (19.33 steps), while SRM is the strongest standalone contributor. The results suggest that execut
    
[^4]: 模型增长、递归与边界算子如何影响缩放指数

    How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents

    [https://arxiv.org/abs/2609.19107](https://arxiv.org/abs/2609.19107)

    该研究证明架构干预（尤其是模型增长和递归深度）可以改变预训练的缩放指数，使7.4B模型增长架构以约20倍更少的计算量匹配GPT-3 13B的性能，且计算效率优势随规模扩大而增加。

    

    缩放定律预测损失如何随计算量的增加而下降。我们证明，与传统观念相反，架构干预可以改变预训练中的缩放指数，从而随着计算量的增加带来性能的指数级提升。作为一个锚定点，我们考虑了循环Transformer的架构形式。尽管通常并不这样使用，但循环（也称为递归深度）通过在训练期间增加循环次数，提供了一种实现模型增长的机制。无论是否共享权重，模型增长都能对缩放指数带来最大的改变。特别地，一个7.4B参数的模型增长架构在CORE基准上以大约20倍更少的计算量匹配了GPT-3 13B的性能，并且其计算效率的提升随规模扩大而增加。此外，仅在普通Transformer中使用边界算子（即归一化并注入较早的块），也能带来不断增加的计算效率提升。

    arXiv:2609.19107v1 Announce Type: new  Abstract: Scaling laws predict how loss decreases with increases in computation. We show, contrary to conventional wisdom, that architectural interventions can modify scaling exponents in pre-training, leading to exponential improvements in performance with increases in computation. As an anchoring point, we consider the architectural formulation of looped transformers. Although not typically used in this way, looping, also known as recursive depth, provides a mechanism for model growth, by increasing the number of loops during training. Model growth, with and without shared weights, provides the biggest changes to the scaling exponents. In particular, a 7.4B model growth architecture matches GPT-3 13B on CORE with roughly $20\times$ less compute, and has compute efficiency gains that increase with scale. Moreover, simply using a boundary operator in a vanilla transformer, which normalizes and injects an earlier block, also provides increasing com
    
[^5]: 利用大语言模型评估中的内部表示来监测与发现奖励作弊行为

    Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations

    [https://arxiv.org/abs/2609.19101](https://arxiv.org/abs/2609.19101)

    该论文发现奖励作弊行为会在大语言模型内部表示中留下一致且可解释的签名，利用简单的均值差向量即可在常见基准测试中可靠地检测并系统发现模型表现出的各类奖励作弊行为。

    

    随着模型规模的扩大，奖励作弊行为变得更加频繁、更加隐蔽、后果也更加严重。那么它会在模型内部表示中留下特征性的痕迹吗？本工作分析了奖励作弊在前沿开源大语言模型内部是如何表示的，以及如何利用这些表示来理解和发现模型所展现的各类作弊行为。特别地，我们发现简单的均值差向量能够在 Kimi K3、GLM 5.2 和 Qwen 3.8 Max 中连贯地表示各种常见评估中的奖励作弊行为。尽管这些向量构造简单，但它们既具有泛化性又具有可解释性，我们可以利用它们可靠地检测奖励作弊。我们首先在 DeepSWE 和 SWE-bench 等常用基准测试中评估了奖励作弊情况，发现模型在这些环境中存在过度的作弊行为：GLM 5.2 在 DeepSWE 中有 57.2% 的回合存在作弊，在 SWE-bench 中有 73% 的回合存在作弊。捕获……（摘要原文在此处截断）

    arXiv:2609.19101v1 Announce Type: new  Abstract: As models scale, reward hacking becomes more frequent, more sophisticated, and more consequential. Does it leave a telltale signature in model representations? This work analyzes how reward hacking is represented internally in frontier open source LLMs, and how those representations can be used to understand and discover the range of hacking behaviors a model displays. In particular, we find that simple difference of means vectors coherently represent reward hacking in Kimi K3, GLM 5.2, and Qwen 3.8 Max across a variety of behaviors in common evaluations. Despite their simplicity, these vectors are both generalizable and interpretable, and we can use them to reliably detect reward hacking. We first evaluate reward hacking in commonly reported benchmarks like DeepSWE and SWE-bench, finding that models reward hack excessively in these environments; GLM 5.2 hacks in 57.2% of rollouts on DeepSWE and in 73% of rollouts on SWE-bench. Catching 
    
[^6]: 自主实验室中基于证据的智能体药物制剂开发

    Evidence-Grounded Agentic Formulation Development in an Autonomous Laboratory

    [https://arxiv.org/abs/2609.19099](https://arxiv.org/abs/2609.19099)

    Andromeda 2是一个能够在自主实验室中基于结构化实验证据进行推理并调用计算与实验工具的智能体系统，在紫杉醇SEDDS制剂开发中以相同预算实现了50%的高性能命中率，远超概率优化模型（17%）和实验设计方法（2%）。

    

    自乳化药物递送系统（SEDDS）可以改善难溶性药物的口服生物利用度，但寻找高性能制剂仍然需要大量的实验工作。我们提出了Andromeda 2，这是一个智能体系统，能够对结构化的内部实验证据进行推理，并调用计算和实验工具来设计和执行连续的制剂批次。在使用匹配预算的微型化自动化实验室条件下，我们将其与Andromeda 1（一个已部署于数十个实际开发项目中的概率优化模型）以及湿实验室实验设计（DoE）方法进行基准比较。对于紫杉醇，Andromeda 2实现了50%的高性能命中率，而Andromeda 1为17%，DoE仅为2%；Andromeda 2识别出12个满足全部四个目标产品质量概况（TPP）指标的制剂，而Andromeda 1和DoE分别仅为6个和0个。AUC₁₀₋₂₄₀的中位数分别为70.1、12.0和3.5 mg·min/mL，而最大AUC为……

    arXiv:2609.19099v1 Announce Type: new  Abstract: Self-emulsifying drug delivery systems (SEDDS) can improve the oral bioavailability of poorly soluble drugs, but identifying high-performing formulations remains experimentally intensive. We present Andromeda 2, an agentic system that reasons over structured in-house experimental evidence and invokes computational and experimental tools to design and execute successive formulation batches. Using a miniaturized automated laboratory at a matched budget, we benchmark it against Andromeda 1, a probabilistic optimization model deployed across dozens of live development projects, and a wet-lab design-of-experiments (DoE) campaign. For paclitaxel, Andromeda 2 achieved a 50% high-performance hit rate versus 17% for Andromeda 1 and 2% for DoE, and identified 12 formulations meeting all four target product profile (TPP) objectives versus 6 and 0, respectively. Median $AUC_{10-240}$ was 70.1, 12.0, and 3.5 mg$\cdot$min/mL, while maximum AUC was com
    
[^7]: 基于|D|维稀疏地标嵌入的非CND距离度量通用核框架

    A General Kernel Framework for Non-CND Distance Measures Using |D|-Dimensional Sparse Landmark Embeddings

    [https://arxiv.org/abs/2609.19083](https://arxiv.org/abs/2609.19083)

    提出稀疏地标嵌入（SLE）核框架，通过紧支撑凸块函数将输入嵌入为稀疏特征向量，使得任意距离度量（包括非条件负定度量）都能构造出可证明半正定的核矩阵，从而完全摆脱核方法（如高斯过程）对希尔伯特距离条件的依赖。

    

    核方法，尤其是高斯过程（GP），需要希尔伯特距离度量（即其平方为条件负定（CND）的度量）来保证核矩阵的半正定性（PSD）；而这一条件在许多自然输入空间上并不成立，包括光滑流形和概率分布空间。我们提出了稀疏地标嵌入（SLE）核，彻底消除了这一要求。每个输入通过以全部|D|个训练点为中心的紧支撑凸块函数被嵌入为稀疏特征向量；在该嵌入空间中应用任何标准PSD核，即可得到对任意距离度量均可证明为半正定的核。紧支撑特性自动控制了嵌入的稀疏性，使得尽管环境维度很高，核矩阵仍保持良态且计算上可行。我们对PSD性质、稀疏性、稳定性以及普适性提供了理论保证。

    arXiv:2609.19083v1 Announce Type: cross  Abstract: Kernel methods, and Gaussian Processes (GPs) in particular, require a Hilbertian distance measure---one whose square is conditionally negative definite (CND)---to guarantee positive semi-definiteness (PSD) of the kernel matrix; a condition that fails for many natural input spaces, including smooth manifolds and spaces of probability distributions. We propose the Sparse Landmark Embedding (SLE) kernel, which eliminates this requirement entirely. Each input is embedded into a sparse feature vector via compactly supported bump functions centered at all |D| training points; applying any standard PSD kernel in this embedding space yields a kernel that is provably PSD for arbitrary distance measures. The compact support automatically controls embedding sparsity, keeping kernel matrices well-conditioned and computationally tractable despite the high ambient dimension. We provide theoretical guarantees on PSD, sparsity, stability, and universa
    
[^8]: 概率线性解释

    Probabilistic Linear Explanations

    [https://arxiv.org/abs/2609.19077](https://arxiv.org/abs/2609.19077)

    该论文提出了一个基于稀疏锚定线性模型的概率可解释性统一框架，适用于二分类与连续回归，通过在布尔超立方体上的映射严格推广了基于子集的解释方法，并证明在神经网络模型下最小化相关性误差是难解的，进而将其与可处理的保真度误差代理目标相关联。

    

    形式化可解释性为个体预测提供了具有严格数学依据的论证。然而，溯因解释往往涉及过多特征，超出了人类认知能力的极限，而概率化的放松方法在很大程度上仍局限于类别分类任务。我们提出了一个基于稀疏锚定线性模型的概率可解释性统一框架，适用于二分类和连续回归。通过将实例映射到布尔超立方体，我们的线性解释严格推广了基于子集的方法：它们既能捕获特征贡献的大小和方向，又能强制满足预设的稀疏度预算 $k$。我们证明，当底层模型为神经网络时，最小化此类解释的相关性误差是 \ClassNPPP-难的，并将这一难以处理的目标与一个可处理的代理目标——保真度误差——联系起来。对于一个参数化族……

    arXiv:2609.19077v1 Announce Type: cross  Abstract: Formal explainability provides mathematically grounded justifications for individual predictions. However, abductive explanations often exceed human cognitive limits by involving too many features, while probabilistic relaxations have remained largely limited to categorical classification. We present a unified framework for probabilistic explainability based on sparse, anchored linear models, applicable to both binary classification and continuous regression. By mapping instances to the Boolean hypercube, our linear explanations strictly generalize subset-based approaches: they capture both the magnitude and direction of feature contributions while enforcing a prescribed sparsity budget $k$. We show that minimizing the relevance error for such explanations is \ClassNPPP-hard when the underlying model is a neural network, and we relate this intractable objective to a tractable surrogate---the fidelity error. For a parameterized family o
    
[^9]: 双重下降即最小作用量原理

    Double descent is the principle of least action

    [https://arxiv.org/abs/2609.19076](https://arxiv.org/abs/2609.19076)

    本文用统计力学解释了机器学习中的双重下降现象：将随机梯度训练视为温度为 $T$ 的粒子在损失能量景观上的扩散，有限时间的扩散带来有效权重衰减，使每个参数成为二次自由度，从而由能量均分定理导出测试误差随参数数量先升后降的规律。

    

    将模型的测试误差对其参数数量 $d$ 作图，误差先下降，在模型恰好能够拟合训练数据时达到峰值，随后再次下降，呈现出双重下降现象。我们用统计力学来解释这一现象：基于随机梯度的方法的训练轨迹是一个粒子，在诱导温度 $T$ 下于训练损失的能量景观上游走；一次已达平衡的训练会以相同的频率访问给定训练损失的每一个参数向量——这正是统计力学的基本假设——其概率由玻尔兹曼分布给出。由于训练从某个初始点出发，且只有有限的时间进行扩散，它会携带一种有效的权重衰减，这使得每个参数都成为一个二次型自由度。于是，能量均分定理将能量以 $T/2$ 的份额分配给这 $d$ 个自由度，因此在固定的训练损失下，增加参数会降低……（原文摘要在此处截断）

    arXiv:2609.19076v1 Announce Type: cross  Abstract: The test error of a model plotted against its number of parameters $d$ falls, peaks when the model can just fit the training data, and falls again, exhibiting the double descent phenomenon. We explain the phenomenon with statistical mechanics. The training trajectory of a stochastic gradient-based method is a particle wandering over the energy landscape of the training loss at an induced temperature $T$, and a run that has equilibrated visits every parameter vector of a given training loss equally often, the fundamental postulate of statistical mechanics, with probability given by the Boltzmann distribution. Because training starts at an initial point and has only finite time to diffuse, it carries an effective weight decay, which makes every parameter a quadratic degree of freedom. The equipartition theorem then distributes the energy among the $d$ degrees of freedom in shares of $T/2$, so at a fixed training loss adding parameters lo
    
[^10]: RLLBC-Lib：一个面向强化学习与基于学习的控制的教学代码库

    RLLBC-Lib: An Educational Code Library for Reinforcement Learning and Learning-Based Control

    [https://arxiv.org/abs/2609.19074](https://arxiv.org/abs/2609.19074)

    该论文介绍了RLLBC-Lib，一个精心设计的教学代码库，通过统一的表格型与深度强化学习实现以及核心原理示例，降低学生在基于学习的控制领域学习强化学习的入门门槛。

    

    强化学习（RL）是一个令人兴奋的概念，也是一个值得分享的杰出成功案例。然而，强化学习建立在多个对象之间在若干周期内展开的相当复杂的交互之上。这种动态过程往往最适合通过易于理解的实现来阐释。我们提出了RLLBC-Lib，这是一个精心打造的代码库，旨在降低学生和其他学习者在基于学习的控制背景下学习强化学习的门槛。其核心是一个全面的表格型强化学习方法库，以强化对理论基础的清晰理解。随后是一个遵循相同设计原则的深度强化学习库，突显了简单的表格型方法与最先进的深度强化学习方法之间的相似性。此外，RLLBC-Lib提供了一系列实现，用以说明核心强化学习原理，并将强化学习与其他基于学习的控制方法进行对比。最后，RLLBC-Lib提供……（摘要在此处截断）

    arXiv:2609.19074v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is an exciting concept as well as a remarkable success story worth sharing. However, RL builds on rather complex interactions between different objects that play out over several cycles. Such dynamics are often best explained with an easily accessible implementation. We present RLLBC-Lib, a carefully crafted code library with the goal of lowering the entry barrier for students and other learners of RL in the context of learning-based control. At its heart, RLLBC-Lib comprises a comprehensive library of tabular RL approaches to enforce a clear understanding of the theoretical foundations. A deep RL library follows the same design principles, underscoring the parallels between simple tabular and state-of-the-art deep RL approaches. Additionally, RLLBC-Lib provides a collection of implementations illustrating core RL principles and contrasting RL to other learning-based control approaches. Finally, RLLBC-Lib pr
    
[^11]: Safety-Flag：LLM内容审核模型可靠性与校准的统一基准

    Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators

    [https://arxiv.org/abs/2609.19072](https://arxiv.org/abs/2609.19072)

    该论文提出统一基准Safety-Flag，将七个安全审核基准整合到同一“标记/不标记”协议下，从错误方向、概率校准和置信度错误排序三个维度评估LLM内容审核模型，发现总体准确率掩盖了模型间截然不同的错误模式，且所有通用模型都普遍过于自信。

    

    大型语言模型越来越多地被用于内容审核，但大多数评估仍然只报告在单个基准上的总体准确率。我们提出了Safety-Flag，它将七个广泛使用的安全基准（BeaverTails、XSTest、Ethics、WildGuard、Aegis、ToxiChat和ToxiGen）整合到一个平衡的“标记/不标记”统一协议中。我们发布了六个通用大语言模型和四个专用防护模型在相同条目上的条目级决策和置信度分数，并附上三个参考模型的评估结果。Safety-Flag从三个维度衡量内容审核模型的可靠性：错误方向、概率校准，以及基于置信度的错误排序（用于人工复核）。这三个维度常常相互矛盾。总体准确率无法揭示错误方向：一个模型会标记85%的无害内容，而另一个模型则漏掉54%的有害内容。所有六个通用模型都过于自信；为每个模型拟合一个温度参数可以改善其校准。

    arXiv:2609.19072v1 Announce Type: new  Abstract: Large language models are increasingly used for content moderation, but most evaluations still report aggregate accuracy on individual benchmarks. We introduce Safety-Flag, which places seven widely used safety benchmarks (BeaverTails, XSTest, Ethics, WildGuard, Aegis, ToxiChat, and ToxiGen) into a single balanced flag / do-not-flag protocol. We release item-level decisions and confidence scores for six general-purpose LLMs and four dedicated guards, together with three reference models, evaluated on the same items. Safety-Flag measures three dimensions of moderator reliability: error direction, probability calibration, and confidence-based error ranking for human review. They often disagree. Aggregate accuracy does not reveal error direction: one model flags $85\%$ of benign content, whereas another misses $54\%$ of harmful content. All six general-purpose models are overconfident; fitting one temperature per model reduces calibration e
    
[^12]: LightSleepX：一种轻量级的基于Inception的双模态睡眠分期网络

    LightSleepX: A Lightweight, Inception-Based Dual-Modal Network for Sleep Staging

    [https://arxiv.org/abs/2609.19062](https://arxiv.org/abs/2609.19062)

    LightSleepX通过结合Inception风格架构、深度可分离卷积、多尺度增强注意力与Mamba编码器，构建了一个轻量级双模态网络，在资源受限环境中实现了高精度且计算高效的自动睡眠分期。

    

    自动睡眠分期是个人健康监测的基础，然而许多现有方法并不适合真实世界的应用场景。传统流程通常依赖手工设计特征或浅层机器学习模型，难以实现良好的泛化；而最先进的深度学习方法虽然准确率高，但计算开销大，在资源受限的环境中并不实用。本文提出了LightSleepX，一个旨在资源受限环境中提供稳健睡眠分析的轻量级框架。LightSleepX将Inception风格架构与深度可分离卷积和多尺度增强注意力相结合，用于高效的多模态EEG/EOG特征提取，并采用Mamba编码器实现无需规则的长程时间建模。在公开基准数据集上，LightSleepX在Sleep-EDF-20上取得了85.9%的准确率和0.803的宏F1分数，在另一数据集上取得了81.8%的准确率和0.796的宏F1分数。

    arXiv:2609.19062v1 Announce Type: new  Abstract: Automatic sleep staging is fundamental to personal health monitoring, yet many existing approaches are ill-suited for real-world applications. Traditional pipelines often rely on hand-crafted features or shallow machine learning models that struggle to generalize, while state-of-the-art deep learning methods, though accurate, are computationally heavy and impractical for resource-constrained environments. This paper introduces LightSleepX, a lightweight framework designed to deliver robust sleep analysis in resource-constrained environments. LightSleepX combines an Inception-style architecture with depthwise separable convolutions and Multi-scale Enhanced Attention for efficient multi-modal EEG/EOG feature extraction, and a Mamba encoder for rule-free long-range temporal modeling. On public benchmark datasets, LightSleepX achieves 85.9% accuracy and a 0.803 macro-F1 score on Sleep-EDF-20, and 81.8% accuracy and a 0.796 macro-F1 score on 
    
[^13]: 面向差异化按需配送的自动化仓储作业与末端运输集成优化

    Integrated Optimization of Automated Warehouse Operations and Last-Mile Transport for Differentiated On-Demand Delivery

    [https://arxiv.org/abs/2609.19048](https://arxiv.org/abs/2609.19048)

    该论文提出了一种将基于AGV的智能仓储作业与末端多式联运进行集成优化的深度强化学习框架，通过设计多目标算法（如MORM-AGDQN）实现高吞吐量连续订单调度，并提升差异化按需配送系统的灵敏度与适应性。

    

    在差异化按需货物配送服务的背景下，本研究提出了一种基于自动导引车（AGV）的智能仓储作业与末端多式联运的集成优化方法。设计了一种用于多目标联合调度的深度强化学习算法，以建立两个系统之间的动态连接，解决了诸如实现高吞吐量连续订单调度、满足相互竞争的需求、提升整体系统的灵敏度与适应性等关键挑战。在该框架下的仓储优化方面，我们提出了一种改进算法，即基于多目标的多奖励机-A*引导深度Q网络（MORM-AGDQN），该算法综合了服务水平、系统成本和外部运输需求。在外部优化方面，我们提出了一种改进算法，即基于多奖励、多头注意力-异构运力车辆路径（问题）的算法……

    arXiv:2609.19048v1 Announce Type: new  Abstract: In the context of differentiated on-demand goods delivery services, this study proposes an integrated optimization method for automated guided vehicles (AGVs) based smart warehouse operations and the last-mile multi-modal transport. A deep reinforcement learning algorithm for multi-objective joint scheduling is designed to establish a dynamic connection between two systems, solving key challenges such as achieving high-throughput continuous order scheduling, meeting competing requirements, and improving the overall system sensitivity and adaptability. For warehouse optimization within this framework, we propose an improved algorithm based on multi-objective, Multi-Reward Machines-A* Guided Deep Q-Network (MORM-AGDQN), which combines service level, system cost, and external transportation demand. For external optimization, we propose an improved algorithm based on a Multi-Reward, Multi Head attention-Heterogeneous Capacity Vehicle Routing
    
[^14]: LSR-Net：学习非线性流体动力学的前向演化算子

    LSR-Net: Learning the Forward Evolution Operator for Nonlinear Fluid Dynamics

    [https://arxiv.org/abs/2609.19039](https://arxiv.org/abs/2609.19039)

    LSR-Net提出了一种新型神经算子架构，将可学习积分核分解为长程分量（基于指数和表示的可训练傅里叶乘子）与短程分量（标准卷积），仅凭状态快照对即可学习动力系统的前向演化算子，以O(n log n)的复杂度高效预测非线性流体动力学。

    

    我们提出了长短程神经网络（LSR-Net），这是一种专为数据驱动前向演化建模设计的新型神经算子架构，并将其扩展应用于非线性流体力学的预测。LSR-Net仅从初始状态与未来状态快照对中学习动力学系统的演化算子，它在堆叠的网络模块中将可学习的积分核分解为长程（LR）和短程（SR）两个分量。SR分量使用标准卷积来捕捉局部动力学，而LR分量则采用指数和（SOE）表示。这使得全局相互作用可以作为可训练的傅里叶乘子进行高效计算，将计算复杂度降低至 O(n log n)（其中 n 为输入快照中的像素数量），且每个通道仅需少量参数。LSR-Net在三个具有挑战性的二维基准问题上进行了评估：耦合Burgers方程、……（原文摘要不完整）

    arXiv:2609.19039v1 Announce Type: cross  Abstract: We introduce the Long-Short-Range Neural Network (LSR-Net), a novel neural operator architecture designed for data-driven forward evolution modeling, and extends it to the prediction of nonlinear fluid dynamics. LSR-Net learns the evolution operator of a dynamical system solely from pairs of initial and future state snapshots, which splits the learnable integral kernel into long-range (LR) and short-range (SR) components within stacked network blocks. While the SR component uses standard convolutions to capture local dynamics, the LR component employs a sum-of-exponentials (SOE) representation. This allows for the efficient computation of global interactions as a trainable Fourier multiplier, reducing computational complexity to $O(n \log n)$ where $n$ is the number of pixels in an input snapshot and requiring only a few parameters per channel. LSR-Net is evaluated on three challenging 2D benchmarks: the coupled Burgers equation, the w
    
[^15]: TwinMark：一种在特征蒸馏与逻辑蒸馏下可证明存活的统一水印

    TwinMark: A Unified Watermark for Provable Survival Under Feature and Logit Distillation

    [https://arxiv.org/abs/2609.19011](https://arxiv.org/abs/2609.19011)

    提出统一水印方案TwinMark，通过协方差投影（cov-Feat）与类条件Fisher对齐线性载体（cc-FALC）两条互补读取通道，分别针对KL知识蒸馏和特征匹配蒸馏攻击提供可由教师端验证的检测能力下界证书，从而可证明地保障水印在模型被蒸馏后仍能存活并被检测。

    

    我们提出了TwinMark，这是一种水印方案，它通过对模型输出摘要的两个互补线性泛函读取单一的SHAKE128密钥：一个是针对载体集协方差的协方差投影器（cov-Feat），另一个是从类均值逻辑值（logits）解码的类条件Fisher对齐线性载体（cc-FALC）。这两个读取器共享同一个比特向量，覆盖了已部署视觉模型的两个提取面：一个是受到KL知识蒸馏（KD）攻击的分类器API（Std. KL-KD），另一个是受到特征匹配KD（FM-KD）攻击的仅表征宿主。每个读取器都提供一个可由教师端度量的后验证书，该证书为蒸馏后的检测能力设定了下界，并且这两个通道在一种受机制限制的OR规则下组合，其检验统计量（校准的零假设或比特投票）由暴露的表面决定。cov-Feat具有秩无关的算子范数证书，cc-FALC具有中心化逻辑间隙证书，可将……解耦（摘要在此处被截断）

    arXiv:2609.19011v1 Announce Type: new  Abstract: We propose TwinMark, a watermarking scheme that reads a single SHAKE128 secret through two complementary linear functionals of model-output summaries: a covariance projector against the carrier-set covariance (cov-Feat) and a class-conditional Fisher-aligned linear carrier decoded from class-mean logits (cc-FALC). The two readouts share one bit vector and cover the two extraction surfaces of a deployed vision model: a classifier API attacked by KL knowledge distillation (KD) (Std. KL-KD), and a representation-only host attacked by feature-matching KD (FM-KD). Each readout admits a teacher-measurable a posteriori certificate that lower-bounds post-distillation detection power, and the two channels combine under a regime-restricted OR rule whose test statistic (calibrated null or bit vote) is selected by the exposed surface. cov-Feat admits a rank-blind operator-norm certificate, cc-FALC admits a centered-logit-gap certificate that decoupl
    
[^16]: 表格深度学习与经典机器学习在城市土地覆盖分类中的对比研究

    Tabular Deep Learning vs Classical Machine Learning for Urban Land Cover Classification

    [https://arxiv.org/abs/2609.19010](https://arxiv.org/abs/2609.19010)

    该研究在一个统一可复现的流程中，系统对比了经典机器学习模型与表格深度学习模型在城市土地覆盖分类任务上的表现，并采用加权交叉熵损失应对类别不平衡问题。

    

    城市土地覆盖（ULC）分类在城市规划、环境监测和可持续发展中起着至关重要的作用。我们使用来自UCI机器学习仓库的ULC数据集研究这一任务，该数据集包含从高分辨率航空影像中提取的表格特征，涵盖九个类别（如道路、树木、草地、水体）。该数据集呈现出典型的遥感挑战，包括高维度、异构特征和类别不平衡问题。在一个统一且可复现的流程中，我们将经典机器学习模型（如逻辑回归、支持向量机、随机森林、XGBoost、CatBoost）与表格深度学习（TDL）模型（TabNet、FT-Transformer、TabTransformer、TabSeq和一维卷积神经网络）进行基准对比测试。为解决类别不平衡问题，我们对TDL模型采用加权交叉熵损失函数，并使用准确率、宏平均精确率、宏平均召回率、宏平均F1分数、AUC-ROC和混淆矩阵来评估模型性能。我们的结果表明……

    arXiv:2609.19010v1 Announce Type: cross  Abstract: Urban Land Cover (ULC) classification plays a crucial role in urban planning, environmental monitoring, and sustainable development. We study this task using the ULC dataset from the UCI Machine Learning Repository, which includes tabular features derived from high-resolution aerial imagery across nine classes (e.g., roads, trees, grass, water). The dataset presents typical remote sensing challenges, including high dimensionality, heterogeneous features, and class imbalance. In a unified, reproducible pipeline, we benchmark classical machine learning models (e.g., Logistic Regression, SVM, Random Forest, XGBoost, CatBoost) against Tabular Deep Learning (TDL) models (TabNet, FT-Transformer, TabTransformer, TabSeq, and 1D CNNs). To address class imbalance, we employ weighted cross-entropy loss for TDL models and evaluate performance using accuracy, macro-precision, macro-recall, macro-F1, AUC-ROC, and confusion matrices. Our results show
    
[^17]: CompileRover：利用三角色LLM驱动框架革新虚拟机编译器优化

    CompileRover: Revolutionizing Virtual Machine Compiler Optimization with a Tri-Role LLM-Driven Framework

    [https://arxiv.org/abs/2609.19004](https://arxiv.org/abs/2609.19004)

    CompileRover是一个基于LLM的三角色协作优化框架，通过裁判、顾问和操作者的协同机制，结合控制流分析、代码结构转换和动态执行模式识别，有效解决了虚拟机编译器输出中的冗余计算和低效循环等性能瓶颈问题。

    

    代码优化在虚拟机编译器的开发中起着至关重要的作用，优化框架能够显著提升生成的汇编代码的性能。然而，现有的虚拟机编译器输出经常存在冗余计算、低效的循环结构以及次优的函数实现，这些问题共同影响了执行效率。为了解决这些缺陷，我们提出了CompileRover，一个专为虚拟机编译器设计的高级优化框架。CompileRover采用了一种复杂的三角色协作机制，包括裁判、顾问和操作者，通过利用全面的优化算法和新颖的方法，包括控制流分析、代码结构转换和动态执行模式识别，有效地克服了性能瓶颈。大量评估表明，CompileRover……

    arXiv:2609.19004v1 Announce Type: cross  Abstract: Code optimization plays a crucial role in the development of virtual machine compilers, with optimization frameworks significantly enhancing the performance of generated assembly code. However, existing virtual machine compiler outputs frequently exhibit redundant computations, inefficient loop structures, and suboptimal function implementations, which collectively impair execution efficiency. To address these shortcomings, we propose CompileRover, an advanced optimization framework specifically designed for virtual machine compilers. CompileRover employs a sophisticated three-role collaboration mechanism, comprising a referee, an advisor, and an operator, effectively overcoming performance bottlenecks by leveraging comprehensive optimization algorithms and novel methodologies, including control flow analysis, code structure transformations, and dynamic execution pattern recognition. Extensive evaluations demonstrate that CompileRover 
    
[^18]: 能力涌现可以被预测：按种子逐个、提前预测，具有校准区间、可认证的误报率及盲测预注册门槛

    Capability Emergence Can Be Forecast: Per-Seed, In Advance, With Calibrated Intervals, Certified False Alarms, and a Blind Pre-Registered Gate

    [https://arxiv.org/abs/2609.19000](https://arxiv.org/abs/2609.19000)

    该论文首次以严格的预测评分标准（受控误报率、校准区间、阴性样本、盲测预注册门槛）证明了能力涌现时机可以被按种子逐个、提前、带校准不确定性地预测，例如通过前一个token头的形成时间以0.977的Spearman相关性提前约15%训练步数预测归纳头的涌现。

    

    涌现能力被普遍认为是不可预测的：损失平滑改善，而能力却突然出现。先前的工作提供了预警指标，但从未将其作为预测进行评分：没有在受控误报率下的提前时间，没有校准，没有阴性样本，没有盲测。我们提供了这种严格的规范，并证明在grokking模型系统和小型语言模型中，涌现时机可以按每次运行逐个预测、提前预测，并具有校准的不确定性。在30个相同配置的transformer中，前一个token头（previous-token head）的形成时间可以预测每个种子的归纳头（induction-head）涌现，Spearman rho=0.977，中位提前量为975步（约占训练的15%）；一个最佳情况的损失规则以50步的提前量获得相同的排名（即时预测）。共形区间覆盖了15/15的保留种子，冻结规则在两个从未见过的配置上通过了盲测预注册门槛（覆盖率分别为10/10和9/10）。随后一个陷阱语言梯级攻击了我们自己的……（摘要在此处截断）

    arXiv:2609.19000v1 Announce Type: new  Abstract: Emergent capabilities are widely treated as unpredictable: loss improves smoothly while abilities appear abruptly. Prior work offers early-warning indicators but never scores them as forecasts: no lead time at controlled false-alarm rate, no calibration, no negatives, no blind tests. We supply that discipline and show that, in grokking model systems and small language models, emergence timing is forecastable per run, in advance, with calibrated uncertainty. Across 30 transformers at identical configuration, the formation time of the previous-token head forecasts each seed's induction-head emergence at Spearman rho=0.977 with median lead 975 steps (~15% of training); a best-case loss rule ties the ranking with 50-step lead (a nowcast). Conformal intervals covered 15/15 held-out seeds, and the frozen rule passed blind pre-registered gates on TWO never-seen configurations (10/10 and 9/10 coverage). A trap-language rung then attacked our own
    
[^19]: 一个维度，无刹车：自我认知限制了LLM中对有害同伴从众行为的过滤

    One Axis, No Brake: Self-Knowledge Limits the Filtering of Harmful Peer Conformity in LLMs

    [https://arxiv.org/abs/2609.18998](https://arxiv.org/abs/2609.18998)

    该论文证明了在多智能体LLM中过滤有害同伴从众的“刹车”本质上等价于伪装的正确性探测器，因此受限于模型不完美的自我认知（AUROC仅0.64–0.89），这一“墙”即使用白盒引导也无法突破。

    

    多智能体LLM系统被期望更加可靠，因为各个智能体可以互相捕捉错误。但同伴压力是双刃剑：纠正错误答案的同一机制也可能推翻原本正确的答案。一个诱人的保障方案是设置一个“刹车”，保留有益的修正并阻止有害的修正。我们证明这种刹车很难构建，原因很简单：修正恰好仅在原答案正确时才有害，因此决定是否阻止修正与判断模型本身是否正确是同一件事。这将开放式地寻找刹车的问题转化为一个可测量的量——模型的自我认知：任何基于部署时信号构建的刹车本质上都是伪装的正确性探测器，而自我认知远非完美（在六个模型家族中AUROC约为0.64–0.89）。我们将这一上限称为“墙”。即使对模型自身的正确性方向进行白盒引导也无法突破它：它改变的是模式……（摘要原文在此处截断）

    arXiv:2609.18998v1 Announce Type: cross  Abstract: Multi-agent LLM systems are expected to be more reliable because agents can catch each other's mistakes. But peer pressure cuts both ways: the same correction that fixes a wrong answer can overturn a right one. The tempting safeguard is a brake that keeps the beneficial revisions and blocks the harmful ones. We show this brake is hard to build, for a simple reason: a revision is harmful exactly when the original answer was right, so deciding whether to block it is the same as knowing whether the model was already correct. This turns the open-ended hunt for a brake into one measurable quantity, the model's self-knowledge: any brake built from a deploy-time signal is a correctness probe in disguise, and self-knowledge is far from perfect (AUROC $\approx 0.64$--$0.89$ across six model families). We call this ceiling the wall. Even white-box steering of the model's own correctness direction does not breach it: it changes how often the mode
    
[^20]: 被抑制而非被抹除：即使是无权重知识编辑，被编辑事实的表征痕迹依然存在

    Suppressed, Not Erased: A Representational Trace of Edited Facts Survives Even Weight-Free Knowledge Editing

    [https://arxiv.org/abs/2609.18985](https://arxiv.org/abs/2609.18985)

    该论文发现即使知识编辑在生成层面“成功”完成，甚至包括完全不修改模型权重的外部记忆编辑方法，被编辑掉的原始事实仍然可以以远高于随机水平的准确率从模型隐藏状态中线性解码出来，表明知识编辑只是“抑制”而非真正“抹除”了原始知识。

    

    知识编辑基准测试验证的是局部正确性，即被编辑后的模型是否能在接近编辑的提示上生成新事实，但并未评估原始事实在模型内部还有多少是可解码的。我们使用线性痕迹探针直接研究残留知识：在编辑一个事实之后，探究原始对象是否仍然可以从模型的隐藏状态中被恢复出来。在GPT-2-XL上，通过将三种机制上截然不同的编辑器应用于50个CounterFact编辑，在成功编辑之后，原始对象仍然可以以远高于随机水平的线性可解码性被恢复（ROME的探针准确率为0.96，受限微调为0.86，基于记忆的编辑器GRACE为0.79，而随机水平为0.50；所有编辑均达到100%的基于生成的成功率）。其中GRACE的结果最具启发性：GRACE完全不改变基础模型的权重，而是通过外部记忆来覆盖事实，然而原始对象仍然可以从底层网络中解码出来，因此……

    arXiv:2609.18985v1 Announce Type: new  Abstract: Knowledge-editing benchmarks certify local correctness, whether an edited model produces the new fact on near-edit prompts but not how much of the original fact remains decodable inside the model. We study residual knowledge directly with a linear trace probe: after editing a fact, we ask whether the original object is still recoverable from the model's hidden states. On GPT-2-XL, across three mechanistically distinct editors applied to 50 CounterFact edits, the original object remains linearly decodable well above chance after a successful edit (probe accuracy 0.96 for ROME, 0.86 for constrained fine-tuning, and 0.79 for the memory-based editor GRACE, against a chance level of 0.50; all edits reach 100% generation-based success). The GRACE result is the most informative: GRACE changes zero base-model weights, overriding the fact through an external memory, yet the original object is still decodable from the underlying network, so the re
    
[^21]: 独奏乐谱图像的乐器分类

    Instrument Classification of Solo Sheet Music Images

    [https://arxiv.org/abs/2609.18980](https://arxiv.org/abs/2609.18980)

    本文提出将乐谱图像基于bootleg score表示法转换为音乐“词语”序列，并通过无监督预训练语言模型再微调的方法实现独奏乐谱图像的乐器分类，使RoBERTa的分类准确率从34.5%提升至42.9%。

    

    本文研究了独奏乐谱的乐器分类问题。以往的工作主要集中于音频数据中的乐器识别，而我们则直接使用原始乐谱图像来解决乐器分类问题。我们的方法首先基于bootleg score表示法将乐谱图像转换为音乐“词语”序列，然后将该问题视为文本分类任务。我们证明，通过在无标签数据上训练语言模型、使用预训练的语言模型权重初始化分类器、再在有标签数据上微调分类器，可以显著提升分类器的性能。在这项工作中，我们在来自IMSLP的八种不同乐器的独奏乐谱图像上训练了AWD-LSTM、GPT-2和RoBERTa模型。我们发现GPT-2和RoBERTa的性能略优于AWD-LSTM，并且预训练使RoBERTa的分类准确率从34.5%提升至42.9%。

    arXiv:2609.18980v1 Announce Type: cross  Abstract: This paper studies instrument classification of solo sheet music. Whereas previous work has focused on instrument recognition in audio data, we instead approach the instrument classification problem using raw sheet music images. Our approach first converts the sheet music image into a sequence of musical "words" based on the bootleg score representation, and then treats the problem as a text classification task. We show that it is possible to significantly improve classifier performance by training a language model on unlabeled data, initializing a classifier with the pretrained language model weights, and then finetuning the classifier on labeled data. In this work, we train AWD-LSTM, GPT-2, and RoBERTa models on solo sheet music images from IMSLP for eight different instruments. We find that GPT-2 and RoBERTa slightly outperform AWD-LSTM, and that pretraining increases classification accuracy for RoBERTa from 34.5% to 42.9%. Furtherm
    
[^22]: 底层的自动机：加性输入通路是Householder线性RNN中状态追踪的寄生吸引子

    The Automaton Underneath: The Additive Input Pathway Is a Parasitic Attractor for State Tracking in Householder Linear RNN

    [https://arxiv.org/abs/2609.18966](https://arxiv.org/abs/2609.18966)

    通过因果消融发现，移除Householder线性RNN中的加性输入注入项能让模型学会精确的状态追踪自动机并实现长度泛化，而保留该项则会充当“寄生吸引子”导致分布外性能崩溃。

    

    具有输入依赖的Householder乘积转移的线性RNN（DeltaNet/DeltaProduct类）可以被证明能够表示困难的状态追踪自动机，然而训练后的模型却无法实现长度泛化——近期工作将这一差距归因于优化问题，但缺乏因果解释。我们通过一项预注册的、架构内的因果消融实验给出了这样的因果解释：即同一个模型删除一项——加性输入注入 $b_t = W_b e_t$。当存在 $b_t$ 时，模型在长度32内拟合良好，但在奇偶性、$S_4$、$A_5$ 以及非可解的 $S_5$ 字问题（word problems）上出现分布外崩溃（在 $S_5$ 上位置512处准确率仅为0.20）。当移除该项时——输入仅通过正交转移起作用——同一架构学会了精确的自动机：在16倍训练长度处中位准确率达到1.00，且在我们提出并验证的表示定律所允许的每一个宽度下均成立：每token所需的最小Householder因子数等于任务生成元在格式p中的最大反射长度……

    arXiv:2609.18966v1 Announce Type: new  Abstract: Linear RNNs with input-dependent Householder-product transitions (DeltaNet/DeltaProduct-class) can provably represent hard state-tracking automata, yet trained models fail to length-generalize -- a gap recent work attributes to optimization, without a causal account. We give one, in a pre-registered, within-architecture causal ablation: the same model with one term deleted -- the additive input injection $b_t = W_b e_t$. With $b_t$, models fit length 32 and collapse out-of-distribution on parity, $S_4$, $A_5$, and non-solvable $S_5$ word problems (0.20 at position 512 on $S_5$). Without it -- input acting only through the orthogonal transitions -- the same architecture learns the exact automaton: median accuracy 1.00 at 16x the training length, at every width admitted by a representation law we state and test: the minimal number of Householder factors per token equals the maximal reflection length of the task's generators in the format-p
    
[^23]: FedGuide：面向异构联邦强化学习的扩散先验对齐与价值基线引导

    FedGuide: Diffusion Prior Alignment and Value Baseline Guidance for Heterogeneous Federated Reinforcement Learning

    [https://arxiv.org/abs/2609.18964](https://arxiv.org/abs/2609.18964)

    该论文提出FedGuide框架，通过扩散先验作为行为模型并利用最优传输混合专家进行聚合，解决了异构联邦强化学习中客户端间的分布不匹配问题，同时以DICE价值基线提供低方差的回报感知引导。

    

    联邦强化学习（FRL）使分布式智能体能够在异构环境中进行协同策略学习。尽管近期基于方差缩减、散度惩罚和动量优化的方法改进了异构设置下的FRL，但这些方法仍主要同步策略或价值网络参数，并未明确解决异构客户端之间的分布不匹配问题。因此，我们提出了**FedGuide**，一个使用扩散先验作为行为模型的FRL框架，为异构的本地策略学习提供由个性化数据支持的分布。FedGuide不直接对本地策略进行平均，而是通过最优传输混合专家（OT-MoE）聚合这些扩散先验，在分布空间中保留异构行为模式。此外，它还开发了分布校正估计（DICE）价值基线，以提供低方差、具有回报感知的引导。

    arXiv:2609.18964v1 Announce Type: new  Abstract: Federated Reinforcement Learning (FRL) enables collaborative policy learning across distributed agents with heterogeneous environments. While recent methods based on variance reduction, divergence penalization, and momentum optimization improve FRL under heterogeneous settings, they still primarily synchronize policy or value-network parameters and do not explicitly address distributional mismatch among heterogeneous clients. Therefore, we propose \textbf{FedGuide}, a FRL framework that uses diffusion priors as behavior models to provide personalized data supported distributions for heterogeneous local policy learning. Instead of directly averaging local policies, FedGuide aggregates those diffusion priors through Optimal-Transport Mixture-of-Experts (OT-MoE), preserving heterogeneous behavior modes in distribution space. It further develops a Distribution Correction Estimation (DICE) value baseline to provide low-variance, return-aware 
    
[^24]: 变点感知世界模型：在基于模型的强化学习中通过检测动力学变化并遗忘陈旧经验回放实现恢复

    Changepoint-Aware World Models: Detecting Dynamics Shifts and Recovering by Forgetting Stale Replay in Model-Based RL

    [https://arxiv.org/abs/2609.18950](https://arxiv.org/abs/2609.18950)

    提出变点感知世界模型（CAWM），利用在线CUSUM检验从内部预测误差中检测动力学突变，并通过遗忘陈旧经验回放实现快速恢复，显著优于被动重训练和重建动力学模型的基线方法。

    

    机器人学习到的自身动力学模型只有在动力学发生变化之前才有效：执行器会磨损、载荷会变动、关节会僵硬。一个继续照常训练的基于模型的智能体适应缓慢，会被充满陈旧经验的经验回放缓冲区拖累。我们提出了变点感知世界模型（CAWM），这是一个基于DreamerV3的智能体，它从自身内部预测误差中检测突发的动力学变化，使用针对滚动基线的在线CUSUM检验，该检验仅在突变发生时触发，而不会因缓慢的学习漂移而误触发。随后它会遗忘陈旧的回放经验，在保留已学习表征的同时清除过时数据。在两种与机器人相关的动力学变化（重力加倍和执行器增益减半）下的模拟运动任务中，CAWM的恢复速度显著快于被动重新训练。它还击败了一个强大的基线方法——在检测到变化时重新生成全新动力学模型的方法，即模型库方法在深度世界模型中的对应实现。

    arXiv:2609.18950v1 Announce Type: new  Abstract: A robot's learned model of its own dynamics is only valid until those dynamics change: actuators wear, payloads shift, and joints stiffen. A model-based agent that keeps training as if nothing happened adapts slowly, dragged back by a replay buffer full of stale experience. We present Changepoint-Aware World Models (CAWM), a DreamerV3 agent that detects an abrupt dynamics shift from its own internal prediction error, using an online CUSUM test against a rolling baseline that fires only on abrupt change rather than on slow learning drift. It then forgets stale replay, keeping the learned representation while flushing obsolete data. On simulated locomotion under two robot-relevant shifts, doubled gravity and halved actuator gain, CAWM recovers substantially faster than passive retraining. It also beats a strong baseline that respawns a fresh dynamics model on detection, the deep-world-model analogue of model-bank methods. With the response
    
[^25]: StableEval Arena：一个用于稳定币价格稳定性预测的成本感知智能体基准框架

    StableEval Arena: A Cost-Aware Agentic Benchmark for Stablecoin Price Stability Prediction

    [https://arxiv.org/abs/2609.18949](https://arxiv.org/abs/2609.18949)

    StableEval Arena是一个成本感知的智能体基准框架，通过无泄漏历史回放评估基于LLM的智能体系统在稳定币锚定风险预测上的表现，并将可信度定义为预测质量、运营可靠性与计算成本的联合属性。

    

    我们提出了StableEval Arena，一个成本感知的基准框架，用于评估智能体AI系统在稳定币锚定风险预测方面的能力。StableEval Arena评估基于大语言模型（LLM）的智能体系统在诊断锚定压力以及预测七天内偏离一美元锚定价格方面的表现，采用无泄漏的历史回放方法，并结合交易所价格-交易量数据与市场背景特征。我们报告了两个互补的实验模块：一个包含120个案例的压力增强验证模块，以及一个包含507个案例的自然分布完整竞技场评估模块。针对六种基于LLM的智能体配置和基线方法，StableEval Arena测量预测质量、校准标签行为、结构化输出可靠性、延迟、token消耗以及估计的推理成本。该框架并非仅凭准确性对智能体进行排名，而是将可信度视为预测质量、运营可靠性和计算成本的联合属性。

    arXiv:2609.18949v1 Announce Type: cross  Abstract: We introduce StableEval Arena, a cost-aware benchmark framework for evaluating agentic AI systems on stablecoin peg-risk prediction. StableEval Arena evaluates LLM-backed agentic systems on diagnosing peg stress and forecasting deviations from the one-dollar peg over a hidden seven-day horizon, using leakage-safe historical replay with exchange price-volume data and market-context features. We report two complementary experiment blocks: a 120-case stress-enriched validation block and a 507-case natural-distribution full-arena evaluation block. Across six LLM-backed agent configurations and baselines, StableEval Arena measures prediction quality, calibrated-label behavior, structured-output reliability, latency, token consumption, and estimated inference cost. Rather than ranking agents by accuracy alone, the framework treats trustworthiness as a joint property of forecast quality, operational reliability, and computational cost. The re
    
[^26]: 随机环境中的多智能体协调社会法则

    Social Laws for Multi-agent Coordination in Stochastic Environments

    [https://arxiv.org/abs/2609.18929](https://arxiv.org/abs/2609.18929)

    本文将社会法则概念扩展至随机的、基于奖励的多智能体环境，提出α-鲁棒性度量，并通过归约为求解一系列马尔可夫决策过程来实现社会法则的稳健性验证。

    

    在多智能体环境中，协调各个智能体以防止相互干扰并确保稳健的个体性能是一项关键挑战。以往关于多智能体系统社会法则的研究主要集中于确定性的、基于目标的设定。本文将社会法则的概念扩展到随机的、基于奖励的环境中，提出了一种用于定义和验证其在各种条件下稳健性的形式化框架。我们引入了α-鲁棒性的概念，用于衡量在假设所有智能体都遵守社会法则的前提下，每个智能体在执行其最优单智能体策略时所能保证获得的效用。随后，我们提出了一种在随机环境中验证社会法则稳健性的方法，该方法基于将问题归约为求解一系列马尔可夫决策过程。在玩具环境上的实证评估展示了我们框架的潜力。

    arXiv:2609.18929v1 Announce Type: cross  Abstract: In multi-agent environments, coordinating agents to prevent interference and ensure robust individual performance is a critical challenge. Previous research on social laws for multi-agent systems has primarily focused on deterministic, goal-based settings. This paper extends the concept of social laws to stochastic, reward-based environments, proposing a formalism for defining and verifying their robustness under various conditions. We introduce the notion of $\alpha$-robustness, a measure of the guaranteed utility each agent retains while pursuing its optimal single agent policy, assuming all agents obey the social law. We then present an approach for robustness verification of social laws in stochastic settings, based on a reduction to solving a series of Markov decision processes. Empirical evaluations on toy environments illustrate the potential of our framework.
    
[^27]: 基于超图表示学习与图条件扩散的对撞机事件全面重建

    Comprehensive reconstruction of collider events with hypergraph representation learning and graph-conditioned diffusion

    [https://arxiv.org/abs/2609.18928](https://arxiv.org/abs/2609.18928)

    提出了VyPER框架，将对撞机事件表示为物理启发的超图，通过结合超边监督分类与图条件扩散模型，在统一框架内同时完成粒子分配和中微子运动学预测，实现对撞机事件的全面重建。

    

    在粒子对撞机实验中，事件重建是指从探测器记录的稳定末态中推断硬散射过程中产生的短寿命粒子运动学的任务。我们将事件重建分解为两个主要任务：将测得的喷注和带电轻子分配给其母粒子，以及预测未测量的中微子运动学。我们提出了VyPER，这是一种新颖的几何学习框架，它将对撞机事件表示为具有物理启发拓扑结构的超图。VyPER将用于粒子分配的超边监督分类与用于预测中微子运动学的扩散模型相结合，利用联合损失函数在统一框架内优化这两个重建任务。我们在多个质子-质子碰撞过程中展示了VyPER的性能，并将其与现有的解析方法和基于机器学习的重建技术进行比较。在此过程中，我们证明了……

    arXiv:2609.18928v1 Announce Type: cross  Abstract: In particle collider experiments, event reconstruction is the task of inferring the kinematics of short-lived particles produced in the hard scatter from the stable final states recorded by detectors. We decompose event reconstruction into two primary tasks: assigning measured jets and charged leptons to parent particles, and predicting unmeasured neutrino kinematics. We present VyPER, a novel geometric learning framework that represents collider events as hypergraphs with a physics-inspired topology. VyPER combines the supervised classification of hyperedges for particle assignment with a diffusion model for predicting neutrino kinematics, leveraging a joint loss function to optimize both reconstruction tasks within a unified framework. We showcase VyPER across several proton-proton collision processes, comparing its performance to existing analytical and machine-learning-based reconstruction techniques. In doing so, we demonstrate th
    
[^28]: 混合专家语言模型中专家的高阶剪枝

    Higher-order pruning of experts in mixture-of-experts language models

    [https://arxiv.org/abs/2609.18916](https://arxiv.org/abs/2609.18916)

    提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。

    

    arXiv:2609.18916v1 公告类型：交叉 摘要：混合专家语言模型存在参数量庞大的问题，这造成了显著的内存瓶颈。专家剪枝是减少参数量最直接的方法，然而现有方法对每个专家独立地做出剪枝决策，并假设专家的贡献是纯粹可加的。实际上，混合专家模型中专家的使用本质上是协作性的。我们推导出了HOPE（专家高阶剪枝），这是一种二阶剪枝目标函数，可以证明能够最小化剪枝所产生的误差上界。我们证明REAP（一种最先进的一阶剪枝方法）是HOPE在忽略交互项时的特例。在三个前沿MoE模型（参数量高达1220亿）、两个不同的校准集以及多个基准测试（包括数学、指令遵循、编程和智能体任务套件）上，我们证明HOPE能够比现有方法做出更好的剪枝决策，并且……

    arXiv:2609.18916v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) language models suffer from large parameter counts, which create a significant memory bottleneck. Expert pruning is the most direct approach for reducing this parameter count, yet existing methods make pruning decisions for each expert independently, and assume experts' contributions are purely additive. In reality, expert usage in MoEs is inherently cooperative. We derive HOPE (Higher-Order Pruning of Experts), a second-order pruning objective which provably minimizes an upper bound on the error resulting from pruning. We show that REAP (a state-of-the-art first-order pruning method) is a special case of HOPE where interaction terms are ignored. Across three frontier MoE models (up to 122B parameters), two distinct calibration sets, and multiple benchmarks (including math, instruction following, coding, and an agentic suite), we demonstrate that HOPE produces better pruning decisions than existing methods, and
    
[^29]: 物理信息核方法的快速学习速率

    Fast Learning Rates for Physics-Informed Kernel Methods

    [https://arxiv.org/abs/2609.18901](https://arxiv.org/abs/2609.18901)

    本文为结合数值观测与微分观测的物理信息核估计器证明了有限样本误差界，揭示了预测误差的双区间结构：当微分观测有限时，误差速率同时依赖于数值与微分观测的数量，而当微分观测数量超过阈值后，误差速率达到饱和并与完美物理约束下的最优速率相匹配。

    

    在物理信息机器学习中，目标函数 $u^*$ 从带噪声的数值观测 $y_i=u^*(x_i)+ \varepsilon_i$ 中学习，同时结合微分信息，微分信息既可以由带噪声的观测 $d_j=(Du^*)(z_j)+\xi_j$ 给出，也可以由已知的物理约束 $Du^*=v$ 给出。我们考虑 $D$ 为线性微分算子的情形，并分析一种结合 $n$ 个数值观测与 $m$ 个微分观测的物理信息核估计器 $\hat u$。在此背景下，我们探究微分信息能在多大程度上改善预测，以及这种改善在数量上如何依赖于 $n$、$m$ 和 $D$。我们在数值模拟的支持下证明了有限样本界，揭示了预测误差的双区间结构：当 $m$ 有限时，误差速率同时依赖于 $n$ 和 $m$；当 $m$ 超过依赖于问题的阈值时，速率达到饱和，并与拥有完美约束 $D$ 时的最优速率（oracle rate）相匹配。

    arXiv:2609.18901v1 Announce Type: cross  Abstract: In physics-informed machine learning, a target function $u^*$ is learned from noisy value observations $y_i=u^*(x_i)+ \varepsilon_i$, together with differential information, given either by noisy observations $d_j=(Du^*)(z_j)+\xi_j$ or by a known physical constraint $Du^*=v$. We consider the setting where $D$ is a linear differential operator and analyze a physics-informed kernel estimator $\hat u$ combining $n$ value observations and $m$ differential observations. In this context, we ask how much can differential information improve predictions, and how does this improvement depend quantitatively on $n$, $m$, and $D$. We prove finite-sample bounds, supported by numerical simulations, revealing a two-regime structure for the prediction error. When $m$ is limited, the rate depends jointly on $n$ and $m$; when $m$ exceeds a problem-dependent threshold, the rate saturates and matches the oracle rate obtained when the perfect constraint $D
    
[^30]: 面向非线性系统的李雅普诺夫算子学习

    Learning Lyapunov Operators for Nonlinear Systems

    [https://arxiv.org/abs/2609.18894](https://arxiv.org/abs/2609.18894)

    本文首次证明了李雅普诺夫解算子在指数稳定性假设下具有良定义性、唯一性和连续性，为在整个非线性系统族上统一学习李雅普诺夫函数奠定了理论基础。

    

    为非线性动力系统构造李雅普诺夫函数是稳定性分析中的核心问题，但至今仍充满挑战。李雅普诺夫函数通常被刻画为一阶偏微分方程（PDE）的解，但这些解通常只针对单个系统获得，限制了其在不同系统之间的复用。本文研究了李雅普诺夫解算子，该算子将一个向量场映射到由基于耗散的李雅普诺夫偏微分方程所定义的对应李雅普诺夫函数。我们证明，在吸引域的紧子集上以及指数稳定性假设下，该算子是良定义的、唯一的，并且关于向量场和耗散函数的扰动均具有连续性。这些结果为在非线性系统族上统一逼近李雅普诺夫函数提供了理论基础。基于这些理论基础，我们采用傅里叶（此处摘要不完整）

    arXiv:2609.18894v1 Announce Type: cross  Abstract: Constructing Lyapunov functions for nonlinear dynamical systems is a central problem in stability analysis, yet remains challenging. Lyapunov functions are commonly characterized as solutions to first-order partial differential equations (PDEs), but these solutions are typically obtained for single systems, limiting their reuse across systems. In this paper, we study the Lyapunov solution operator that maps a vector field to the corresponding Lyapunov function defined by a dissipation-based Lyapunov PDE. We establish that, on compact subsets of the domain of attraction and under exponential stability assumptions, this operator is well-defined, unique, and continuous with respect to perturbations of both the vector field and the dissipation function. These results provide a theoretical foundation for approximating Lyapunov functions uniformly over families of nonlinear systems. Building on these theoretical foundations, we employ Fourie
    
[^31]: NeuroECG：基于ECGFounder的深度心电图表示，用于心脏骤停后无需脑电图的神经功能预后预测

    NeuroECG: ECGFounder-Based Deep ECG Representation for EEG-Free Neurological Prognostication After Cardiac Arrest

    [https://arxiv.org/abs/2609.18891](https://arxiv.org/abs/2609.18891)

    该研究提出NeuroECG框架，通过渐进式解冻微调预训练心电图基础模型ECGFounder，并利用分位数池化与PCA压缩深度特征，实现仅凭低成本床旁心电图即可对心脏骤停患者进行无需脑电图的神经功能预后预测。

    

    心脏骤停后的神经功能预后预测通常依赖于脑电图（EEG）。然而，脑电图需要较高的临床资源。床旁心电图（ECG）是标准化且低成本的检查手段，但其在预测神经功能结局方面的价值仍未得到充分探索。在本研究中，我们提出了NeuroECG，一个基于ECGFounder的深度表示框架，用于无需脑电图的辅助预后预测。NeuroECG通过任务特定的微调来适配预训练的心电图基础模型，并在单通道床旁监测心电图上实施了渐进式解冻策略。每位患者的多个心电图片段被编码为片段级别的深度特征，这些嵌入通过分位数池化（q = 0.24）进行聚合，并使用主成分分析（PCA）进行压缩。在来自多中心I-CARE数据库的412名有心电图数据的患者上进行的实验表明，经适配的ECGFounder骨干网络在仅使用心电图的骨干网络中取得了最佳性能。

    arXiv:2609.18891v1 Announce Type: cross  Abstract: Neurological prognostication after cardiac arrest commonly relies on electroencephalography (EEG). However, EEG demands high clinical resources. Bedside electrocardiography (ECG) is standard and low-cost. Yet, its value for predicting neurological outcomes remains underexplored. In this study, we propose NeuroECG, an ECGFounder-based deep representation framework for EEG-free auxiliary prognostication. NeuroECG adapts a pretrained ECG foundation model via task-specific fine-tuning. We implement a gradual unfreezing strategy on single-channel bedside monitoring ECG. Multiple ECG segments per patient are encoded into segment-level deep features. These embeddings are aggregated via quantile pooling ($q = 0.24$) and compressed using principal component analysis (PCA). Experiments on 412 ECG-available patients from the multicenter I-CARE database show that the adapted ECGFounder backbone achieves the best performance among ECG-only backbone
    
[^32]: 防止模型坍塌：从Fisher-Rao视角审视合成数据训练的动力学

    Preventing Model Collapse: A Fisher-Rao Perspective on the Dynamics of Training with Synthetic Data

    [https://arxiv.org/abs/2609.18878](https://arxiv.org/abs/2609.18878)

    本文从Fisher-Rao信息几何的视角，为防止模型坍塌所需的最低人类数据比例建立了严格的理论保证，克服了以往基于欧几里得度量的分析在高维情况下下界失效的问题。

    

    大规模语言模型（LLM）如今常规地使用合成数据进行训练，因为高质量的人类数据已被日益增长的更大模型的需求所耗尽。然而，在合成数据上的递归训练常常引发模型坍塌——这是一种退化的反馈循环，模型会逐渐遗忘真实的基础数据分布。使用合成数据与新鲜人类数据的混合进行训练是一种合理的对策，并且可以防止模型坍塌。然而，为维持训练稳定所需的人类数据与合成数据的确切最小比例仍是一个开放性问题。在本文中，我们为防止模型坍塌所需的最小人类数据比例建立了严格的理论保证。尽管先前的工作为这一比例建立了形式化的下界，但由于该分析依赖于R^n中通常的欧几里得度量且缺乏自适应性，此下界在极高维情况下可能是空洞的（无意义的）。

    arXiv:2609.18878v1 Announce Type: new  Abstract: Large Language Models (LLMs) are now routinely trained using synthetic data, since high-quality human data has been exhausted by the ever increasing needs of larger and larger models. However, recursive training on synthetic data frequently induces model collapse, a degenerative feedback loop where models progressively forget the true underlying data distribution. Training on a mixture of synthetic and fresh human data is a logical countermeasure and can prevent model collapse. However, it is an open question as to what is the exact minimum required ratio of human-to-synthetic data to maintain training stability. In this paper, we establish rigorous theoretical guarantees on the minimum rate of human data required to prevent model collapse. Although previous work established a formal lower bound for this ratio, such bound can be vacuous for very high dimensions, as the analysis relies on the usual Euclidean metric in R^n and is not adapt
    
[^33]: 基于物理的IN718晶体织构强度在激光粉末床熔融（LPBF）不同离焦范围内的预测、不确定性量化与决策

    Physics-based prediction, uncertainty quantification and decision-making for IN718 crystallographic texture intensity across LPBF defocus regimes

    [https://arxiv.org/abs/2609.18863](https://arxiv.org/abs/2609.18863)

    本研究开发了一个两阶段物理模型用于预测LPBF工艺中IN718的<001>晶体织构强度，通过k近邻残差校正、面束能量密度判据和保形区间实现不确定性量化，从而在物理有效范围内支持可靠的工艺决策。

    

    在激光粉末床熔融中可靠地预测晶体学织构，对于将工艺条件与各向异性响应联系起来以及材料认证至关重要。然而，黑盒模型在分布偏移下可能失效，且无法区分数据支持薄弱与物理有效性丧失这两种情况。本研究针对Inconel 718中<001> || BD（构建方向）织构开发了一个两阶段的基于物理的模型。第一阶段将工艺变量映射到熔化模式和熔池几何形状；第二阶段通过将经验物理模型与随机森林残差模型相结合来预测织构。k近邻权重会对支持度较差的查询衰减残差校正，同时基于研究的面束能量密度判据会在所采用的热传导模式包络之外拒绝给出预测。保形区间在保留的物理有效数据集上进行评估，并通过SHAP和Sobol分析评估残差敏感性。在受控的留一离焦评估下……

    arXiv:2609.18863v1 Announce Type: new  Abstract: Reliable prediction of crystallographic texture in laser powder bed fusion is critical for linking process conditions with anisotropic response and for qualification. However, black-box models may fail under shift and cannot distinguish weak data support from loss of physical validity. This study develops a two-stage physics-based model for <001> || BD (build direction) texture in Inconel 718. Stage 1 maps process variables to melting mode and melt pool geometry. Stage 2 predicts texture by combining an empirical physics model with a random-forest residual model. A k-nearest-neighbor weight attenuates residual corrections for poorly supported queries, while a study-specific areal beam-power-density criterion withholds predictions outside the adopted conduction envelope. Conformal intervals are evaluated on the retained physics-valid set, and SHAP and Sobol analyses assess residual sensitivity. Under a controlled leave-one-defocus-out eva
    
[^34]: 可解码却误路由：稀疏特征揭示视觉-语言模型在有害模因检测中的读取差距

    Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection

    [https://arxiv.org/abs/2609.18860](https://arxiv.org/abs/2609.18860)

    研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。

    

    当大型视觉-语言模型错误分类有害模因时，这种失败可能反映了内部证据的缺失，或者是无法将已表征的证据正确路由到输出端。我们在Gemma-3和Qwen3.5模型中利用稀疏自编码器、角色条件探针、因果干预和恢复实验来区分这两种情况，并在六个有害内容基准上进行评估，还开展了西班牙语和印地语-英语混合语的补充评估。稀疏读取在所有六个主要二分类任务上都优于模型原生预测：Qwen的稀疏读取平均宏F1达到0.740，而原生预测仅为0.432，残差重建达到0.486；Gemma则从0.532提升至0.714。这些差异反映的是监督可访问性，而非模型中预先存在的原生决策规则，且最具影响力的词元角色取决于具体任务。在所评估的分数尺度下，Qwen的静默特征消融对探针的敏感度高出24-63倍，而路由特征修补……

    arXiv:2609.18860v1 Announce Type: cross  Abstract: When a large vision-language model misclassifies a harmful meme, the failure may reflect missing internal evidence or an inability to route represented evidence to its output. We distinguish these cases in Gemma-3 and Qwen3.5 using sparse autoencoders, role-conditioned probes, causal interventions, and recovery experiments across six harmful content benchmarks, with additional Spanish and Hindi-English code-mixed evaluations. Sparse readouts outperform native prediction on all six primary binary tasks: Qwen averages $0.740$ versus $0.432$ native macro-F1, while residual reconstruction reaches $0.486$, whereas Gemma improves from $0.532$ to $0.714$. These differences reflect supervised accessibility rather than a pre-existing, native decision rule, and the most influential token role depends on the task. Under the evaluated score scales, Qwen silent-feature ablation is $24-63$ times more probe-sensitive, whereas routed-feature patching 
    
[^35]: 无限参数大语言模型：从实时数据生成与调整权重

    Infinite-Parameter LLMs: Generating and Adapting Weights from Live Data

    [https://arxiv.org/abs/2609.18842](https://arxiv.org/abs/2609.18842)

    本文提出一种能将实时交互数据直接写入自身权重的大语言模型新架构，突破了传统模型权重冻结、只能依赖静态预训练数据和临时提示词的局限，使模型能够从用户实时提供的知识和更正中持续学习。

    

    缩放定律认为，语言模型的能力随着参数规模和训练数据量的增加而提升，混合专家架构正是借助这些定律取得了显著成果——对于每个token，只需激活庞大存储参数库中的一小部分。然而，这种成功建立在静态预训练数据之上。部署后的模型面对的是一个不同的世界：许多能使其更有用的数据并不在其训练集中，而是存在于它当前正在处理的实时交互中，例如用户提供的知识事实或给出的更正。传统模型无法从这些数据中学习，因为其权重在训练结束后即被冻结。取而代之的是，运行时提供的知识和行为通过检索或指令的方式被放入提示词中，在每次请求时被重新读取，却在请求结束后被丢弃。我们探讨一种架构如何能够通过将实时交互写入自身权重，从而从实时交互中学习。采用……

    arXiv:2609.18842v1 Announce Type: new  Abstract: The scaling laws hold that a language model grows more capable with more parameters and more training data, and Mixture-of-Experts (MoE) architectures have ridden these laws to remarkable results, activating only a fraction of an enormous stored parameter bank for each token. That success is built on static pretraining data. A deployed model faces a different world, where much of the data that would make it more useful is not in its training set but in the live interaction it is currently handling, such as the facts a user supplies or the corrections they give. A conventional model cannot learn from this data, because its weights are frozen after training. Instead, the knowledge and behaviour supplied at run time are placed in the prompt, by retrieval or instruction, and re-read on every request only to be discarded once the request ends. We ask how an architecture could learn from live interaction by writing it into its weights. Taking 
    
[^36]: 可解释的多示例学习从急性髓系白血病的常规流式细胞术中实现关键分子改变的早期预测

    Interpretable Multi-Instance Learning Enables Early Prediction of Key Molecular Alterations from Routine Flow Cytometry in Acute Myeloid Leukemia

    [https://arxiv.org/abs/2609.18825](https://arxiv.org/abs/2609.18825)

    本研究开发了一种基于决策树的可解释多示例学习模型，能利用入院数小时内完成的常规流式细胞术数据，快速预测急性髓系白血病的关键分子突变（NPM1和FLT3-ITD），从而在数周等待期之前为早期治疗决策提供指导。

    

    背景：NPM1和FLT3-ITD突变的分子检测指导着急性髓系白血病（AML）关键早期治疗决策，但检测结果可能需要数周时间，远晚于这些决策必须做出的时间点。流式细胞术作为常规临床护理的一部分，在入院数小时内即可完成，其数据可能携带足够的信号来直接预测这些突变，而无需额外的成本或等待时间。方法：我们开发了一种基于决策树的可解释多示例学习分类器，其中每个患者样本被建模为单个细胞的集合，突变状态从细胞水平的预测中推断得出。该模型与基于临床变量训练的随机森林以及适用于多管流式细胞术数据的深度卷积神经网络进行了基准比较。性能通过197名患者的发现队列交叉验证进行评估，并在161名患者的独立队列上进行测试，使用受试者工作特征曲线下面积……

    arXiv:2609.18825v1 Announce Type: new  Abstract: Background: Molecular testing for NPM1 and FLT3-ITD mutations guides critical early treatment decisions in acute myeloid leukemia (AML), but results can take weeks, long after these decisions must be made. Flow cytometry, already performed within hours of admission as part of routine care, may carry enough signal to predict these mutations directly, without added cost or delay. Methods: We developed an interpretable multi-instance learning classifier based on a decision tree, in which each patient sample is modeled as a collection of individual cells and mutation status is inferred from cell-level predictions. The model was benchmarked against a random forest trained on clinical variables and a deep convolutional neural network adapted for multitube flow cytometry data. Performance was assessed by cross-validation on a discovery cohort of 197 patients and tested on an independent cohort of 161 patients, using the area under the receiver 
    
[^37]: WaveTLM：通过任务编译实现可靠的时间序列语言建模

    WaveTLM: Reliable Time-Series Language Modeling through Task Compilation

    [https://arxiv.org/abs/2609.18812](https://arxiv.org/abs/2609.18812)

    提出WaveTLM编译器-执行器架构和ExecTS-QA基准，将用户请求编译为带类型的任务状态并由任务原生执行器生成可靠输出，解决了时间序列语言模型中“响应看似合理但实际幻觉任务对象”的可靠性问题。

    

    时间序列语言模型为各类时间任务提供了共享的自然语言接口，但看似合理的文本并不能保证可靠的任务输出。模型的响应可能显得合理，却在所需对象上产生幻觉：数值序列可能违反形状、尺度、通道顺序或时间对齐的约束，文本化的决策可能落在合法标签空间之外。我们提出了可靠时间序列语言建模的问题表述，将任务对象的可靠性与预测质量区分开来。我们引入了ExecTS-QA，一个基于契约的基准测试，涵盖预测、插补、分类、异常检测和波形分析五类任务。我们进一步提出了WaveTLM，一个统一的编译器-执行器模型：其任务编译器将用户请求、可见参数和基于波形的证据转换为带类型的任务状态，而任务原生执行器则负责构建数值张量、合法决策或结构化记录。在ExecTS-QA上，单个WaveTLM检查点（摘要在此截断）

    arXiv:2609.18812v1 Announce Type: new  Abstract: Time-series language models provide a shared natural-language interface across temporal tasks, but plausible text does not guarantee reliable task outputs. Responses may appear reasonable while hallucinating the required object: numerical sequences can violate shape, scale, channel order, or temporal alignment, and textual decisions can fall outside the legal label space. We formulate reliable time-series language modeling, separating task-object reliability from predictive quality. We introduce ExecTS-QA, a contract-grounded benchmark spanning forecasting, imputation, classification, anomaly detection, and waveform analysis. We further propose WaveTLM, a unified compiler-executor model whose task compiler transforms user requests, visible arguments, and wave-grounded evidence into typed task states, while task-native executors construct numerical tensors, legal decisions, or structured records. On ExecTS-QA, a single WaveTLM checkpoint 
    
[^38]: 深度 V-学习的收敛框架：误差传播与尖锐动作间隙界

    A Convergence Framework for Deep $V$-Learning: Error Propagation and Sharp Action-Gap Bounds

    [https://arxiv.org/abs/2609.18782](https://arxiv.org/abs/2609.18782)

    该论文为深度 V-学习建立了收敛性理论框架，将更新误差分解为拟合、转移复用、目标构建、重放、动作选择和探索六个残差，并在可集中性条件下给出了控制策略损失的显式误差传播界与尖锐的动作间隙界。

    

    我们为视界（horizon）为 H 的深度 V-学习建立了收敛界。该算法将一个标量值函数拟合到来自已执行转移的目标上，并使用预测模型和值函数来选择动作。对于采用新鲜真实核结果的当前观测后继目标，其条件均值为 $\mathcal{T}^\beta V$，即对行为策略动作取平均；而贝尔曼最优性更新为 $\mathcal{T} V$。我们将更新误差分解为六个残差：拟合、转移复用、目标构建、重放、动作选择和探索。在 $L^s$ 可集中性条件下，这些残差的 $L^p$ 范数（$p=s/(s-1)$）控制了期望的 $L^1$ 策略损失。该界显式地只对最后 $H-1$ 个更新块的残差加权，并为较短运行步数的情形附加一个初始化项。我们量化了在不同视界层级间共享采样分布的代价。对于 $n^{-\nu}$ 阶的统计误差界，我们推导出了最优的连续……（原文摘要在此处截断）

    arXiv:2609.18782v1 Announce Type: new  Abstract: We establish convergence bounds for deep $V$-learning with horizon $H$. The algorithm fits a scalar value function to targets from executed transitions and selects actions using a predictive model and the value function. For current observed-successor targets with fresh true-kernel outcomes, the conditional mean is $\mathcal{T}^\beta V$, which averages over behavior-policy actions. The Bellman optimality update is $\mathcal{T} V$. We decompose the update error into six residuals: fitting, transition reuse, target construction, replay, action selection, and exploration. Under $L^s$ concentrability, their $L^p$ norms ($p=s/(s-1)$) control expected $L^1$ policy loss. The bound explicitly weights residuals from only the last $H-1$ update blocks, plus an initialization term for shorter runs. We quantify the cost of a shared sampling distribution across horizon levels. For statistical error bounds of order $n^{-\nu}$, we derive optimal continu
    
[^39]: CERA-MoA：路由机制与持续学习大语言模型智能体的协同进化

    CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents

    [https://arxiv.org/abs/2609.18779](https://arxiv.org/abs/2609.18779)

    提出CERA-MoA框架，通过迭代强化学习使动态路由器与持续学习的LLM智能体协同进化，并利用基于中间层隐藏状态的熟悉度估计器和累积阈值自适应路由机制，动态激活最小智能体子集，以在性能与开销之间取得平衡。

    

    当前智能体混合范式通常将查询路由和智能体微调视为两个相互独立的过程，限制了其应对智能体能力不断演变的能力。这种脱节导致路由策略无法在训练后阶段适应智能体能力的变化，也阻碍了智能体实现协同的数据驱动专业化。为解决这一问题，我们提出了CERA-MoA（面向智能体混合的协同进化路由器与持续学习智能体），这是一个迭代强化学习框架，其中动态路由器和独立智能体策略共同进化。我们设计了一种预测性熟悉度估计器，利用中间层隐藏状态来评估智能体之间的语义能力，从而避免完整rollout的开销。基于这些熟悉度分数，累积阈值自适应路由机制会动态激活一个量身定制的最小智能体子集，在任务性能……（摘要在此处截断）

    arXiv:2609.18779v1 Announce Type: new  Abstract: Current Mixture-of-Agents (MoA) paradigms generally treat query routing and agent fine-tuning as separate processes, limiting their ability to respond to evolving agent capabilities. This disconnect prevents routing strategies from adapting to evolving agent capabilities during post-training and prevents agents from achieving synergistic data-driven specialization. To resolve this, we introduce CERA-MoA (Co-Evolving Router with continually learning Agents for Mixture-of-Agents), an iterative reinforcement learning framework where the dynamic router and independent agent policies co-evolve. We design a predictive familiarity estimator that leverages mid-layer hidden states to evaluate semantic competence among agents, avoiding the overhead of full rollouts. Based on these familiarity scores, a cumulative-threshold adaptive routing mechanism dynamically activates a tailored minimal agent subset, achieving a trade-off between task performan
    
[^40]: 图信号生成建模中的稳定滤波器

    Stable Filters for Generative Modeling of Graph Signals

    [https://arxiv.org/abs/2609.18759](https://arxiv.org/abs/2609.18759)

    本文针对漂移项结合图滤波器与图神经网络的图感知连续时间生成模型，推导了量化图扰动对生成分布影响的显式Wasserstein稳定性界，并据此提出了在保持图热扩散平滑性的同时增强结构稳定性的图滤波器设计原则框架。

    

    在图上生成信号需要具有置换等变性且对相对结构扰动保持稳定的模型。尽管最近的图感知薛定谔桥模型将拓扑信息直接融入其参考动力学中，但图的扰动如何通过这些动力学传播并影响最终生成的分布仍不清楚。在本文中，我们分析了图感知连续时间生成模型的结构稳定性，该类模型的漂移项结合了图滤波器与可学习的图神经网络。我们推导出了显式的Wasserstein稳定性界，用以量化相对图扰动对生成分布的影响。受这些界的启发，我们提出了一个设计稳定图滤波器的原则性框架，该框架在保持图热扩散平滑行为的同时增强了结构稳定性。在合成信号和fMRI信号上的实验表明……

    arXiv:2609.18759v1 Announce Type: cross  Abstract: Generating signals on graphs requires permutation-equivariant models that exhibit stability with respect to relative structural perturbations. While recent graph-aware Schr\"odinger bridge models incorporate topology information directly into their reference dynamics, it is unclear how perturbations of the graph propagate through these dynamics and affect the resulting generated distributions. In this paper, we analyze the structural stability of graph-aware continuous-time generative models whose drift combines a graph filter with a learned graph neural network. We derive explicit Wasserstein stability bounds that quantify the effect of relative graph perturbations on the generated distributions. Motivated by these bounds, we introduce a principled framework for designing stable graph filters that preserve the smoothing behavior of graph heat diffusion, while boosting structural stability. Experiments on synthetic and fMRI signals sho
    
[^41]: 当编辑流即为编辑跳变：复现Edit Flows与EvoFlows

    When Edit Flows are Edit Jumps: replicating Edit Flows and EvoFlows

    [https://arxiv.org/abs/2609.18745](https://arxiv.org/abs/2609.18745)

    本文证明Edit Flows与EvoFlows本质上是同一底层过程（连续时间中编辑逐个触发的纯跳跃式生成器匹配），并发布首个开源实现EditJumps——一个在166万同源抗体对上训练的通用抗体编辑器，可零样本编辑未见先导序列而无需按家族重新训练。

    

    抗体先导物优化需要对现有候选分子进行少量且有界的编辑：不仅包括替换，还包括插入和删除。基于编辑的生成模型是唯一能够在不预先固定编辑位置、编辑次数或输出长度的情况下分配这种编辑预算的模型。然而，现有方法Edit Flows和EvoFlows并未发布代码或完整的训练规范。在本研究中，我们证明这两种方法遵循相同的底层过程——编辑以学习到的速率在连续时间中逐个触发——即有限序列上生成器匹配的纯跳跃情形。通过EditJumps，我们推出了该框架的首个开源实现，利用在166万个观察抗体空间（Observed Antibody Space）同源序列对上训练的单一通用抗体编辑器，为种子序列提出类同源变体，以零样本方式编辑未见过的先导序列，而无需原始方法所要求的按家族重新训练。

    arXiv:2609.18745v1 Announce Type: new  Abstract: Antibody lead optimization calls for a small, bounded set of edits to an existing candidate: substitutions, but also insertions and deletions. Edit-based generative models are the only ones that allocate such an edit budget without fixing the edit positions, the edit count, or the output length in advance. However, the existing approaches Edit Flows and EvoFlows did not release code or complete training specifications. Here, we show that both methods follow the same underlying process -- edits firing one at a time, at learned rates, in continuous time -- the pure-jump case of generator matching over finite sequences. With EditJumps we introduce the first open implementation of this framework, with a single generalist antibody editor trained on 1.66M Observed Antibody Space homolog pairs to propose homolog-like variants of a seed sequence, editing unseen leads zero-shot, without the per-family retraining original approaches require. Repli
    
[^42]: 超越截断：将大语言模型解码重新思考为集成剪枝

    Beyond Truncation: Rethinking LLM Decoding as Ensemble Pruning

    [https://arxiv.org/abs/2609.18723](https://arxiv.org/abs/2609.18723)

    提出ME-Decoding解码框架，将LLM的候选token选择建模为集成剪枝问题，利用马氏距离驱动的目标函数和自适应带宽核构建的token相似度矩阵，在保持高概率的同时增强语义多样性并去除冗余路径，同时通过高效贪心算法降低计算开销。

    

    我们提出了马氏距离集成解码（ME-Decoding），这是一种新颖的大语言模型（LLM）解码框架，它将候选token选择构建为集成剪枝问题。现有的选择策略主要依赖标量概率，忽略了几何语义关系，导致候选冗余。与此同时，当前的几何感知方法通常需要复杂的优化或直接对原始token概率进行重新加权，从而带来显著的计算开销或推理不稳定性。为了解决这些问题，我们将解码形式化为一个子集优化问题，使用马氏距离驱动的目标函数，在保持高概率的同时增强语义多样性。具体而言，我们通过在token嵌入上构建自适应带宽核得到token相似度矩阵，并利用它动态折扣冗余的生成路径。我们进一步设计了一种具有近似线性复杂度的高效贪心选择算法……（摘要在此处被截断）

    arXiv:2609.18723v1 Announce Type: new  Abstract: We introduce Mahalanobis-Ensemble Decoding (ME-Decoding), a novel Large Language Model (LLM) decoding framework that frames candidate token selection as ensemble pruning. Existing selection strategies rely predominantly on scalar probabilities, ignoring geometric semantic relationships and causing candidate redundancy. Meanwhile, current geometry-aware methods often require complex optimization or directly reweighting the original token probabilities, leading to significant computational overhead or inference instability. To address this, we formulate decoding as a subset optimization problem using a Mahalanobis distance-driven objective to enhance semantic diversity while preserving high probabilities. Specifically, we dynamically discount redundant generation paths using a token similarity matrix, constructed via an adaptive-bandwidth kernel over token embeddings. We further devise an efficient greedy selection algorithm with near-line
    
[^43]: 重新思考PPO中的评论家学习：理解并缓解价值平坦化问题

    Rethinking Critic Learning in PPO: Understanding and Mitigating Value Flattening

    [https://arxiv.org/abs/2609.18708](https://arxiv.org/abs/2609.18708)

    本文揭示了PPO评论家中的系统性失效模式“价值平坦化”，即真实状态值变化剧烈而评论家预测却过于平坦，将其归因于评论家损失中的隐式方差惩罚与冗余更新，并提出仅对响应中少数分离良好的状态计算价值损失的稀疏近端策略优化算法SP³O来缓解该问题。

    

    在大语言模型的强化学习中，近端策略优化（PPO）通常使用评论家（critic）来估计状态值并降低策略更新的方差。然而，我们在PPO评论家中发现了一种系统性的失效模式，我们称之为价值平坦化：从多个蒙特卡洛延续中估计出的真实状态值在中间状态之间变化剧烈，而评论家的预测却相对平坦。我们进一步在一个受控的FrozenLake环境中观察到这一现象，并发现随着状态空间的增大，该现象变得更加明显。我们的理论和实证分析将价值平坦化与评论家损失中的隐式方差惩罚以及来自时间相关且梯度相似的冗余状态更新联系起来。基于这些发现，我们提出了稀疏近端策略优化（SP³O），该方法仅对每个响应中少数分离良好的状态应用价值损失，以缓解价值平坦化问题。

    arXiv:2609.18708v1 Announce Type: cross  Abstract: In reinforcement learning for large language models, Proximal Policy Optimization (PPO) commonly uses a critic to estimate state values and reduce the variance of policy updates. However, we uncover a systematic failure mode in PPO critics, which we call Value Flattening: state values, estimated from multiple Monte Carlo continuations, change sharply across intermediate states while critic predictions remain comparatively flat. We further observe this phenomenon in a controlled FrozenLake environment and find that it becomes more pronounced as the state space grows. Our theoretical and empirical analyses relate Value Flattening to an implicit variance penalty in the critic loss and redundant updates from temporally correlated states with similar gradients. Motivated by these findings, we introduce SParse Proximal Policy Optimization (SP$^3$O), which applies the value loss to only a few well-separated states in each response to mitigate
    
[^44]: 迈向可组合的网络数字孪生：基于子图的时延预测研究

    Toward Composable Network Digital Twins: A Subgraph-Based Latency Prediction Study

    [https://arxiv.org/abs/2609.18704](https://arxiv.org/abs/2609.18704)

    本文提出一种可组合的网络数字孪生方法，将网络分解为可重用的子图单元孪生，并通过轻量级组合器聚合来预测每条路由的端到端时延，解决了现有方法单体化、难以适应拓扑和流量变化的问题。

    

    现代网络必须支持不断变化的拓扑、配置和性能目标，这促使人们对快速且可靠的性能估计方法产生需求。网络数字孪生（NDT）能够在此类网络场景中支持性能估计的假设分析，然而，现有的基于机器学习的NDT方法通常依赖于完整的拓扑表示，这种表示本质上是单体式的，在网络拓扑或流量发生变化时缺乏可重用性。本文提出了一种可组合的NDT方法，将网络分解为子图，并用可重用的单元孪生来表示，这些单元孪生能够捕获子图的结构、配置和流量行为。一个轻量级的组合器聚合单元孪生的组合，从而创建出能够预测穿越整个拓扑的每条路由端到端时延的NDT。该方法在受控合成拓扑与多样化流量场景、真实世界的Topology Zoo拓扑以及一个公开的NDT挑战数据集上进行了评估。

    arXiv:2609.18704v1 Announce Type: cross  Abstract: Modern networks must support changing topologies, configurations, and performance objectives, motivating fast and reliable performance estimation. Network digital twins (NDTs) enable what-if analysis for performance estimation in such network scenarios, however, existing machine learning-based NDT approaches often rely on entire topology representations, which are inherently monolithic and lack reusability under topological or traffic changes in the network. This paper introduces a composable NDT approach that decomposes networks into subgraphs represented by reusable unit twins that capture subgraph structure, configuration and traffic behaviours. A lightweight composer aggregates unit twin combinations to create NDTs that predict per-route end-to-end latency through an overall topology. Evaluation across controlled synthetic topologies and diverse traffic scenarios, real-world Topology Zoo topologies, and a public NDT challenge datas
    
[^45]: DAG ReLU网络路径提升雅可比矩阵的秩与计算

    Rank and computation of the pathlifting Jacobian of a DAG ReLU network

    [https://arxiv.org/abs/2609.18682](https://arxiv.org/abs/2609.18682)

    本文通过对骨架矩阵进行初等归纳证明了DAG ReLU网络路径提升雅可比矩阵的秩，并提出了一种无需反向传播、计算成本更低的雅可比矩阵计算方法。

    

    本文通过对网络的隐藏节点数量进行归纳，为DAG ReLU网络的路径提升雅可比矩阵的秩提供了一个自包含的证明。实际上，这种归纳是初等的，关键方法在于考虑网络的骨架矩阵（一个编码网络路径的稀疏矩阵），并将其中的一个隐藏神经元的表示转换为输出节点。该证明依赖于一些中间命题，这些命题将路径提升、其雅可比矩阵、网络参数及其骨架矩阵联系起来，除了能够得出路径提升雅可比矩阵秩的结论外，还提供了一种无需反向传播即可计算该矩阵的方法，其计算成本在实践中比常规的反向传播高效得多。本文附带一个Python模块，该模块实现了论文中针对前馈网络的各个命题，并用于实验性地量化计算……

    arXiv:2609.18682v1 Announce Type: cross  Abstract: This paper provides a self-contained proof of the rank of the pathlifting Jacobian of a DAG ReLU network by performing an induction on the network's number of hidden nodes. In fact, the induction is elementary, and the key recipe is to consider the skeleton matrix of the network, a sparse matrix encoding the network paths, and transform the representation of one of its hidden neurons into an output node. The proof relies on intermediate propositions which link the pathlifting, its Jacobian, the network parameters, and its skeleton matrix, which, on top of permitting to conclude on the rank of the pathlifting Jacobian, also provide a way to compute it without backpropagation and whose computation cost is super efficient in practice compare to usual backpropagation. The paper is provided with a Python module that implements the different propositions of the paper for feed forward networks and is used to experimentally quantifies the comp
    
[^46]: 生成式人工智能对学生学习的非均衡影响：考察AI相关课程中依赖度、评估素养与课程政策的作用

    The Uneven Impact of Generative AI on Student Learning: Examining the Roles of Reliance, Evaluation Literacy, and Course Policy in AI-related Courses

    [https://arxiv.org/abs/2609.18676](https://arxiv.org/abs/2609.18676)

    该研究基于118名学生的调查识别出四类GenAI用户群体，发现学习获益与早期依赖、认知依赖及学术任务支持密切相关，并受教师课程政策、工具版本和工具使用数量的显著影响，揭示了生成式AI对学生学习影响的不均衡性。

    

    生成式人工智能（GenAI）正在改变学生的学习方式，然而课程情境、认知依赖、评估素养以及早期依赖的作用仍待深入探索。本研究通过对某院校12门AI相关课程中118名学生的问卷调查，考察了GenAI使用情况及感知学习体验的差异。我们识别出四类用户群体：报告诸多益处的高使用量学生、报告依赖较少且获益较少的轻度使用者，以及报告不同获益程度的两个中度使用群体。我们还发现免费版与付费版用户、单一工具与多工具用户、以及经历不同教师政策的学生之间存在显著差异。在多变量回归模型中，学术获益与早期依赖和学术任务支持相关；积极影响与认知依赖、学术任务支持、对GenAI可靠性的信心等因素相关。

    arXiv:2609.18676v1 Announce Type: new  Abstract: Generative artificial intelligence (GenAI) is changing how students learn, yet the roles of course context, cognitive reliance, evaluation literacy, and early reliance remain underexplored. Using survey responses from 118 students across 12 AI-related courses at our institution, we examined differences in GenAI use and perceived learning experiences. We identified four user clusters: high-use students reporting many benefits, light users reporting less reliance and fewer benefits, and two moderate-use groups reporting different levels of benefit. We also found significant differences between free- and premium-version users, single- and multiple-tool users, and students experiencing different instructor policies. In multivariable regression models, academic benefit was associated with early reliance and academic task support; positive impact was associated with cognitive reliance, academic task support, confidence in GenAI reliability, an
    
[^47]: VLA-ULAP：在边缘端将云端VLA调用与超轻量级本地动作预测交替进行

    VLA-ULAP: Interleaving Cloud VLA Calls with Ultra-Lightweight Local Action Prediction at the Edge

    [https://arxiv.org/abs/2609.18663](https://arxiv.org/abs/2609.18663)

    提出VLA-ULAP框架，通过仅740万参数的超轻量级本地动作预测器与云端VLA调用交替执行，在边缘设备上将单次推理延迟和能耗降低一个数量级，减少近半至四分之三的云端调用次数，同时保持95%以上的任务成功率。

    

    十亿参数规模的视觉-语言-动作（VLA）策略需要大量的机载算力，而远程推理中的通信延迟会阻碍及时响应。我们提出了VLA-ULAP，它将远程VLA调用与超轻量级本地动作预测器交替执行。ULAP包含约740万参数（包括冻结的视觉编码器），结合当前视觉观测、本体感觉信息以及已执行的动作历史，一次前向传播即可预测动作块。它可以独立训练，无需VLA隐藏状态、在线验证或服务器往返通信。在Jetson Orin Nano上，ULAP每次推理仅需19.9毫秒和0.183焦耳，相比之下，RTX A6000上的GR00T需要284.3毫秒和50.55焦耳。在三个模拟的基线策略/基准测试组合中，选定的运行模式可减少48.8-76.7%的VLA调用次数，同时保留基线成功率的95.0-97.5%。与VLA-JEPA上的本地VLA加速替代方案相比，ULAP估计可减少49.2%的推理能耗。

    arXiv:2609.18663v1 Announce Type: cross  Abstract: Billion-parameter vision--language--action (VLA) policies demand substantial onboard power, while communication delays in remote inference hinder timely responses. We propose VLA-ULAP, which interleaves remote VLA calls with an Ultra-Lightweight Local Action Predictor (ULAP). With approximately 7.4M parameters including the frozen vision encoder, ULAP combines current views, proprioception, and executed action history to predict chunks in one pass. Trained independently, it requires no VLA hidden states, online verification, or server round trips. On Jetson Orin Nano, ULAP takes 19.9 ms and 0.183 J per inference, compared with 284.3 ms and 50.55 J for GR00T on RTX A6000. Across three simulated base-policy/benchmark pairs, selected operating points remove 48.8--76.7\% of VLA calls while retaining 95.0--97.5\% of the baseline success rate. Against local VLA-acceleration alternatives on VLA-JEPA, ULAP uses an estimated 49.2\% less inferen
    
[^48]: 重新审视基于符号的分布式方差缩减方法

    Revisiting Distributed Sign-Based Variance Reduction

    [https://arxiv.org/abs/2609.18656](https://arxiv.org/abs/2609.18656)

    本文通过提出在服务器端利用递归梯度增量的无偏压缩来跟踪全局梯度，解决了数据异构情况下符号聚合引入偏差的问题，首次在非凸随机优化和有限和优化中实现了基于符号的分布式方差缩减方法的最优收敛速率。

    

    基于符号的方法可以降低分布式环境中的通信成本，但当数据异构时，聚合本地符号可能会引入偏差。因此，现有的基于符号的方差缩减方法无法获得最优收敛速率。在本文中，我们解决了这个问题，并在非凸随机优化和有限和优化中都获得了最优收敛速率。我们首先给出了一个反例，表明即使使用精确的本地梯度，多数投票也可能无法逼近稳定点。受此局限性的启发，我们提出通过递归梯度增量的无偏压缩在服务器端跟踪全局梯度。由此，我们获得了 $\ell_1$ 范数的收敛速率 $O(\sqrt{d/K}+\sqrt d (a/(nK))^{1/3})$ 以及 $\ell_2$ 范数的收敛速率 $O(\sqrt{a/K}+\sqrt a/(nK)^{1/3})$。其中，$K$ 为迭代次数，$n$ 为工作节点数量，$d$ 为维度，$a=1+\omega$，其中 $\omega$……

    arXiv:2609.18656v1 Announce Type: new  Abstract: Sign-based methods reduce communication costs in distributed environments, but aggregating local signs can introduce bias when data are heterogeneous. As a result, existing sign-based variance reduction methods fail to obtain the optimal convergence rates. In this paper, we solve this problem and obtain optimal rates for both nonconvex stochastic and finite-sum optimization. We first give a counterexample showing that majority voting can fail to approach stationary points even with exact local gradients. Motivated by this limitation, we propose tracking the global gradient at the server through unbiased compression of recursive gradient increments. As a result, we can obtain the convergence rates of $O(\sqrt{d/K}+\sqrt d (a/(nK))^{1/3})$ for the $\ell_1$-norm and $O(\sqrt{a/K}+\sqrt a/(nK)^{1/3})$ for the $\ell_2$-norm. Here, $K$ is the iteration number, $n$ is the number of workers, $d$ is the dimension, and $a=1+\omega$, with $\omega$ 
    
[^49]: 学习为机器学习编程自适应非局部观测量

    Learning to Program Adaptive Non-Local Observables for Machine Learning

    [https://arxiv.org/abs/2609.18655](https://arxiv.org/abs/2609.18655)

    提出QFWP-ANO架构，利用经典超网络根据输入动态编程变分量子电路参数与非局部观测量，在时间序列预测和强化学习任务上显著超越现有基于ANO的量子神经网络。

    

    量子神经网络（QNN）通常由变分量子电路（VQC）构建，而VQC受限于局部测量。自适应非局部观测量（ANO）通过联合优化电路参数和多量子比特测量来解决这一问题。然而，现有的基于ANO的VQC只能学习单个静态观测量，该观测量在所有输入下保持不变。我们提出了QFWP-ANO，这是一种新颖的架构，它采用经典超网络根据每个输入动态编程VQC参数和/或非局部观测量。在四个ETT数据集上的多变量时间序列预测任务中，QFWP-ANO在20个设置中的16个取得了最低的MSE，在其余四个设置中排名第二，超越了基于ANO的模型和其他强大的基线方法。在强化学习任务中，QFWP-ANO也持续优于ANO-VQC。我们的结果确立了输入条件化的ANO作为增强量子神经网络的有效方法。

    arXiv:2609.18655v1 Announce Type: new  Abstract: Quantum neural networks (QNNs) are typically built from variational quantum circuits (VQCs), which are limited by local measurements. Adaptive non-local observables (ANO) address this by jointly optimizing circuit parameters and multi-qubit measurements. However, existing ANO-based VQCs learn only a single static observable that remains invariant across all inputs. We propose QFWP-ANO, a novel architecture which employs a classical hypernetwork to dynamically program VQC parameters and/or non-local observables conditioned on each input. On multivariate time-series forecasting across four ETT datasets, QFWP-ANO achieves the lowest MSE in 16 of 20 settings and second-lowest in the remaining four, surpassing ANO-based and other strong baselines. On reinforcement learning tasks, QFWP-ANO consistently surpasses ANO-VQCs. Our results establish input-conditioned ANO as an effective approach for enhancing QNNs.
    
[^50]: 谬误基准测试衡量的是论证图式识别，而非谬误检测

    Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection

    [https://arxiv.org/abs/2609.18644](https://arxiv.org/abs/2609.18644)

    该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。

    

    谬误检测基准通常将谬误类别与一个单一的“有效”或“无”类别配对，该类别包含了数据收集过程中未被标注为谬误的所有内容。这种构建方式具有误导性：分类器可以学习到某些线索从而在该类别上表现良好，却并未真正学会区分谬误与正确论证。我们证明，基准测试所报告的低误报率是类别构建方式的产物，而非检测能力的体现。对谬误而言，最有信息量的负样本是使用相同论证图式的正确论证，而在我们考察的四个基准中，此类论证在“有效”类别中最多只占几个百分点。在构建的图式匹配负样本上进行评估时，误报率在CoCoLoFa上从16.6%上升到58.9%，在Reddit上从5.7%上升到62.0%。由于误报率取决于负样本的撰写方式，我们还比较了来自同一流程、仅在论证图式身份上有所不同的两种条件。（摘要原文在此处被截断）

    arXiv:2609.18644v1 Announce Type: new  Abstract: Fallacy-detection benchmarks pair fallacy classes with a single "valid" or "none" class that takes everything data collection did not label as a fallacy. This construction is misleading: a classifier can learn cues that do well on this class without learning to tell a fallacy from a correct argument. We show that the low false-positive rates benchmarks report are an artifact of how the class is built, not evidence of detection ability. The most informative negative for a fallacy is a correct argument using the same argumentation scheme, and such arguments are at most a few percent of the valid class across the four benchmarks we examined. Evaluated on constructed scheme-matched negatives, false-positive rates rise from 16.6% to 58.9% on CoCoLoFa and from 5.7% to 62.0% on Reddit. That rate depends on how the negatives are written, so we also compare two conditions from the same pipeline that differ only in scheme identity. Classifiers lab
    
[^51]: CoRe-MARL：基于循环多智能体强化学习的未知动态下协同再分配

    CoRe-MARL: Cooperative Redistribution Under Unknown Dynamics Using Recurrent Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.18639](https://arxiv.org/abs/2609.18639)

    该论文提出CoRe-MARL框架，将救灾物资再分配问题建模为分散式部分可观测马尔可夫决策过程，利用循环多智能体强化学习使各中心在信息有限、需求动态未知的情况下协同学习再分配策略，以改善最差区域服务并缩小区域间服务差距。

    

    应急管理援助项目（如救灾物资分发）对于向受灾社区提供必要物资至关重要。然而，这些项目在由多个地方中心组成的分散式网络中运行，面临不确定的本地需求和供应动态，导致本地服务的可用性不一致。在这些地方中心之间进行物资再分配可以减少这种不平衡，但各中心通常独立决策，信息有限且交通受阻。本研究通过构建分散式部分可观测马尔可夫决策过程（Dec-POMDP），开发了CoRe-MARL——一个协作式多智能体强化学习（MARL）框架。我们将每个中心视为一个智能体，学习再分配策略，以改善最差情况区域的服务并缩小各区域之间的服务差距，同时保护整个网络的服务水平。我们引入了一个循环网络来捕获……（摘要原文在此处截断）

    arXiv:2609.18639v1 Announce Type: cross  Abstract: Emergency management assistance programs, such as relief distribution, are essential for delivering necessary supplies to affected communities. However, these programs operate in a decentralized network of local centers that face uncertain local demand and supply dynamics, resulting in inconsistent avail- ability of local services. Redistribution of supplies among these local centers reduces these imbalances, but the centers often make decisions independently, with limited information and disrupted transportation. This study develops CoRe-MARL, a cooperative multi-agent reinforcement learning (MARL) framework, by formulating a decentralized partially observable Markov decision process (Dec-POMDP). We treat each center as an agent that learns a redistribution policy to improve the service in the worst-case region and reduce the service gap across regions while protecting network-wide service. We incorporate a recurrent network that capt
    
[^52]: 模型选择需要多少标签？选择性预测的证书与预算

    How Many Labels Does Model Choice Need? Certificates and Budgets for Selective Prediction

    [https://arxiv.org/abs/2609.18622](https://arxiv.org/abs/2609.18622)

    该论文量化了比较模型选择性预测性能（AUGRC）所需的标签预算，通过预标签下界与覆盖线性规划证书证明：确定性选择模型在某些条件下几乎需要标注全部标签，而准确率选择可由少量分歧标签裁决。

    

    分类器可以做出相同的预测，却仍需要标签才能比较它们的选择性性能：置信度排序会以不同的方式加权相同的错误。我们针对广义风险-覆盖曲线下面积（AUGRC）量化了这一标签需求。预标签下界可以排除不充足的预算。当所有标签已知时，一个覆盖线性规划界定了足以确定胜者的最少标签数（即证书大小），对于K个候选者，该上界为K-1个标签。对于固定的K、独立均匀排序且预测完全相同的情况，预标签下界接近样本池的四分之一。当错误为独立于排序的独立同分布伯努利错误时，任何精确的获取策略在渐近意义上几乎需要读取全部标签，尽管双候选证书只需一半。在九个数据集上的108次特征面板比较中，分歧标签可以裁决所有准确率选择，却无法裁决任何AUGRC选择；在96种条件下，20%的预算被证明是不够的。

    arXiv:2609.18622v1 Announce Type: new  Abstract: Classifiers can make identical predictions yet require labels to compare their selective performance: confidence ranks weight the same errors differently. We quantify this requirement for the area under the generalized risk-coverage curve (AUGRC). A prelabel lower bound rules out insufficient budgets. With all labels known, a covering linear program bounds the minimum number of labels sufficient to fix the winner (the certificate size) within $K-1$ labels for $K$ candidates. For fixed $K$, independent uniform orders and identical predictions, the prelabel bound approaches one quarter of the pool. With iid Bernoulli errors independent of the orders, every exact acquisition policy reads almost all labels asymptotically, although a two-candidate certificate needs only half. Across 108 feature-panel comparisons on nine datasets, disagreement labels settle every accuracy choice but no AUGRC choice. A 20% budget is ruled out in 96 conditions; 
    
[^53]: 将阵列信号拓扑学习为条件神经流形

    Learning Array Signal Topologies as Conditional Neural Manifolds

    [https://arxiv.org/abs/2609.18616](https://arxiv.org/abs/2609.18616)

    提出条件神经流形（CNM），以观测条件化的学习流形替代固定阵列流形，无需导向矢量监督即可提升MUSIC等波达方向估计方法在模型失配下的鲁棒性与精度，并可推广至其他基于流形的方法。

    

    多信号分类（MUSIC）等子空间方法通过利用阵列流形与测量的噪声子空间之间的正交性，实现了超分辨波达方向（DoA）估计。因此，其精度依赖于所假设的流形，在模型失配时性能会下降，而无法从空间流形中识别的参数则无法恢复。在这项工作中，我们提出了条件神经流形（CNM），它用一个以观测为条件的、从源参数到导向矢量的映射来替代固定流形。编码器将快照映射为潜在场景表示，该表示对参数空间上的零初始化神经场进行条件化。通过塑造由此产生的MUSIC谱景，流形的学习无需导向矢量监督。由于校正作用于流形本身而非估计器，它可以被其他基于流形的方法直接使用。

    arXiv:2609.18616v1 Announce Type: cross  Abstract: Subspace methods such as multiple signal classification (MUSIC) achieve super-resolution direction of arrival (DoA) estimation by exploiting the orthogonality between the array manifold and the noise subspace of the measurements. Their accuracy therefore depends on the assumed manifold and degrades under model mismatch, while parameters not identifiable from the spatial manifold cannot be recovered. In this work, we propose the conditional neural manifold (CNM), which replaces the fixed manifold with an observation-conditioned mapping from source parameters to steering vectors. An encoder maps the snapshots to a latent scene representation that conditions a zero-initialized neural field over the parameter space. The manifold is learned without steering-vector supervision by shaping the resulting MUSIC landscape. Since the correction acts on the manifold rather than on the estimator, it can be used by other manifold-based methods withou
    
[^54]: 弱化神经元：Transformer中具有超大规模影响力的输入-输出功能

    Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence

    [https://arxiv.org/abs/2609.18612](https://arxiv.org/abs/2609.18612)

    该论文提出通过计算神经元输入权重向量与输出权重向量之间的余弦相似度来识别“弱化神经元”，并发现这类神经元虽然在模型中数量稀少，却激活频繁且对模型行为具有超乎寻常的影响力，同时九个不同的大语言模型均呈现弱化神经元集中分布于后期层、强化神经元集中分布于中早期层的相似模式。

    

    我们分析了大语言模型（LLM）中基于GLU的神经元所学习到的输入-输出行为。我们提出了一种简单的分析方法：对于每个神经元，计算其输入（读取）权重向量与输出（写入）权重向量之间的余弦相似度。在该方案下，强负余弦相似度表明该神经元会削弱它在残差流中检测到的方向，因此我们将其称为“弱化神经元”。这使我们获得了一些新颖的见解。首先，我们展示了九个不同的大语言模型具有相似的模式：弱化神经元主要出现在后期层，而它们的对应物——（条件性）强化神经元——则频繁出现在中早期层。其次，我们发现弱化神经元表现出令人惊讶的行为：尽管数量很少，但它们激活频繁，并对模型行为产生巨大影响。第三，当门控值为负时，弱化神经元对模型输出具有强烈的影响。

    arXiv:2609.18612v1 Announce Type: cross  Abstract: We analyze the learned input-output behavior of GLU-based neurons in large language models (LLMs). We propose a simple analysis method: For each neuron, we compute the cosine similarities between its input (reading) and output (writing) weight vectors. In this scheme, a strong negative cosine similarity indicates the neuron weakens the direction it detects in the residual stream, so we call this a weakening neuron. This allows us to gain a number of novel insights. First, we show that nine different LLMs have similar patterns: weakening neurons appear mostly in late layers whereas their counterparts, (conditional) strengthening neurons, are frequent in early-middle layers. Second, we find that weakening neurons display surprising behavior: even though there are few, they activate often and have a large influence on model behavior. Third, weakening neurons have a strong effect on model output when gate values are negative -- which is su
    
[^55]: 结构化马尔可夫决策过程中决策边界的几何理论

    A Geometric Theory of Decision Boundaries in Structured Markov Decision Processes

    [https://arxiv.org/abs/2609.18610](https://arxiv.org/abs/2609.18610)

    本文提出了一种结构化马尔可夫决策过程中最优策略诱导的决策边界几何理论，证明该几何是策略重构的最小表示，并决定了重构问题的统计与计算复杂度。

    

    经典动态规划通过价值函数和策略来表示最优序贯决策。虽然这种函数表示对于计算最优决策而言是自然的，但一旦最优策略被固定，它并不能直接识别出支配策略重构、表示复杂度或预言机查询复杂度的数学对象。本文通过发展一种结构化最优策略的几何理论来解决这一问题，其中由策略诱导的决策边界几何成为分析的核心对象。我们证明，在适当的结构正则性条件下，这种几何提供了策略重构所需的最小表示，并决定了重构问题的统计复杂度与计算复杂度。基于这一表示，我们建立了策略诱导决策几何的结构性质，引入了边界的内在概念……（摘要在此处截断）

    arXiv:2609.18610v1 Announce Type: new  Abstract: Classical dynamic programming represents optimal sequential decisions through value functions and policies. While this functional representation is natural for computing optimal decisions, it does not directly identify the mathematical object governing policy reconstruction, representation complexity, or oracle-query complexity once an optimal policy is fixed. This paper addresses this question by developing a geometric theory of structured optimal policies in which the decision-boundary geometry induced by the policy becomes the primary object of analysis. We show that, under suitable structural regularity conditions, this geometry provides the minimal representation required for policy reconstruction and determines the statistical and computational complexity of the reconstruction problem. Building upon this representation, we establish structural properties of policy-induced decision geometry, introduce intrinsic notions of boundary a
    
[^56]: PACT：企业AI助手在压力之下能否被信任？

    PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?

    [https://arxiv.org/abs/2609.18605](https://arxiv.org/abs/2609.18605)

    PACT是一个评估企业级AI智能体在用户施压等压力情境下能否坚持遵守合规规则的基准测试，涵盖十二个受监管企业领域和四十八个真实多轮对话场景。

    

    随着企业AI应用的持续增长，企业级大语言模型（LLM）智能体正被部署到招聘、医疗保健和金融等敏感场景中。在这些场景中，遵守智能体系统上下文中规定的规则是首要的法律关切。目前，尚无评估框架能够系统地衡量哪些LLM模型容易违反合规规则，尤其是在面临固执用户的施压、仓促经理的催促，或违规行为便利且有诱惑力的情况下。我们提出了PACT（压力应用合规测试），这是一个针对AI智能体在压力下遵循规则能力的基准测试，涵盖十二个受监管的企业领域和四十八个场景，每个场景均设置在协助员工完成日常任务的真实多轮对话中。每个基准项目都将一条现行规则与一个违反规则的捷径配对，并在不同的措辞和系统提示模式下施加一系列压力。我们……

    arXiv:2609.18605v1 Announce Type: cross  Abstract: As corporate AI adoption continues to grow, enterprise-grade LLM agents are being deployed into sensitive contexts such as hiring, healthcare, and finance. In these contexts, compliance with rules specified in an agent's system context is a first-order legal concern. Currently, no evaluation framework systematically measures which LLM models tend to violate compliance rules, especially under pressure from a persistent user, a hurried manager, or circumstances where violation is convenient or attractive. We introduce PACT (Pressure-Applied Compliance Testing), a benchmark for rule-following under pressure in AI agents assisting employees in daily tasks across twelve regulated enterprise domains and forty-eight scenarios, each set in a realistic multi-turn conversation. Each benchmark item pairs a standing rule against a rule-violating shortcut, and applies a battery of pressures across different wordings and system-prompt modes. We cons
    
[^57]: 通过蒙特卡洛规划的在线鲁棒强化学习

    Online Robust Reinforcement Learning Through Monte-Carlo Planning

    [https://arxiv.org/abs/2609.18599](https://arxiv.org/abs/2609.18599)

    本文提出了一种鲁棒的蒙特卡洛树搜索变体，通过鲁棒幂平均回传算子和探索奖励机制来处理模拟器与真实世界之间的状态转移动力学与奖励分布歧义，并实现了根节点价值估计的O(n^{-1/2})收敛速率。

    

    蒙特卡洛树搜索（MCTS）是解决复杂决策问题的强大框架，但它通常依赖于模拟器与真实世界动力学完全一致的假设。尽管这一假设帮助MCTS在国际象棋、围棋和将棋等游戏中取得成功，但在真实世界场景中，由于低保真模拟器存在建模不匹配，会产生歧义性。在本工作中，我们提出了一种新的鲁棒MCTS变体，以缓解动力学模型的歧义性。我们的算法处理了状态转移动力学和奖励分布的歧义性，从而弥合基于模拟的规划与真实世界部署之间的差距。我们引入了鲁棒的幂平均回传算子和精心设计的探索奖励，以确保搜索树中每个节点的有限样本收敛性。我们证明，该算法在根节点的价值估计上达到了 $\mathcal{O}(n^{-1/2})$ 的收敛速率。

    arXiv:2609.18599v1 Announce Type: cross  Abstract: Monte Carlo Tree Search (MCTS) is a powerful framework for solving complex decision-making problems, yet it often relies on the assumption that the simulator and the real-world dynamics are identical. Although this assumption helps achieve the success of MCTS in games like Chess, Go, and Shogi, the real-world scenarios incur ambiguity due to their modeling mismatches in low-fidelity simulators. In this work, we present a new robust variant of MCTS that mitigates dynamical model ambiguities. Our algorithm addresses transition dynamics and reward distribution ambiguities to bridge the gap between simulation-based planning and real-world deployment. We incorporate a robust power mean backup operator and carefully designed exploration bonuses to ensure finite-sample convergence at every node in the search tree. We show that our algorithm achieves a convergence rate of $\mathcal{O}(n^{-1/2})$ for the value estimation at the root node, compa
    
[^58]: 通过进化进行推理：基于大语言模型的虚假新闻检测的自动元路径发现

    Reasoning through Evolution: Automatic Meta-path Discovery for LLM-based Fake News Detection

    [https://arxiv.org/abs/2609.18597](https://arxiv.org/abs/2609.18597)

    提出MAGER多智能体遗传进化框架，自动发现优化的元路径，将复杂传播图压缩为信息子图，使冻结的大语言模型能够进行结构感知的虚假新闻检测推理，摆脱对大量标注数据的依赖。

    

    传播结构为虚假新闻检测提供了关键证据，然而现有方法主要依赖有监督的基于图神经网络（GNN）的模型，这需要大量标注数据且泛化能力有限。尽管大语言模型（LLMs）展现出强大的推理能力，但直接向其输入原始传播图会造成显著的模态不匹配和严重的信息过载，使得结构感知推理在零样本和少样本设置中难以可靠进行。为了弥合这一差距，我们提出了MAGER，一个多智能体遗传进化框架，能够自动发现针对LLM推理优化的元路径。通过将复杂的传播图压缩为信息丰富的子图，进化出的元路径同时缓解了信息过载和模态不匹配问题，使冻结的LLM能够执行结构感知的真实性推理。我们进一步引入了一种图上下文学习策略，该策略检索语义相关的示例……

    arXiv:2609.18597v1 Announce Type: new  Abstract: Propagation structures provide crucial evidence for fake news detection, yet existing approaches primarily rely on supervised GNN-based models, which require substantial labeled data and exhibit limited generalization. Although large language models (LLMs) exhibit strong reasoning capabilities, directly feeding them raw propagation graphs creates a significant modality mismatch and severe information overload, making structure-aware reasoning unreliable in zero-shot and few-shot settings. To bridge this gap, we propose MAGER, a multi-agent genetic evolution framework that automatically discovers meta-paths optimized for LLM reasoning. By compressing complex propagation graphs into informative subgraphs, the evolved meta-paths alleviate both information overload and modality mismatch, enabling frozen LLMs to perform structure-aware veracity reasoning. We further introduce a graph in-context learning strategy that retrieves semantically an
    
[^59]: ReDIL-GNN：面向电路图神经网络的再综合域增量学习

    ReDIL-GNN: Resynthesis Domain Incremental Learning for Circuit Graph Neural Networks

    [https://arxiv.org/abs/2609.18595](https://arxiv.org/abs/2609.18595)

    提出了ReDIL-GNN再综合域增量学习框架及预适配评分指标RAI，用于应对逻辑再综合引起的电路图神经网络域偏移问题，并指导何时以及如何进行适配。

    

    逻辑再综合在保持电路功能的同时改变门的词汇、拓扑结构和统计特性，在不改变任务标签的情况下为电路图神经网络（GNN）造成域偏移。为研究这一场景，我们提出了ReDIL-GNN，一个再综合域增量学习框架，该框架在新的综合风格到来时适配固定的预测或表示头，并在所有先前观察到的域上评估知识保持能力。由于并非所有偏移都应盲目适配，ReDIL-GNN进一步引入了再综合适应性指数（RAI），这是一个预适配评分，综合了适配需求、源等价可恢复性、结构覆盖率和更新兼容性。我们使用分类器的任务原生指标和嵌入模型的源等价检索指标，评估了监督式硬件安全任务和表示学习模型，并将朴素微调与LwF、Online EWC等方法进行比较。

    arXiv:2609.18595v1 Announce Type: new  Abstract: Logic resynthesis preserves circuit functionality while changing gate vocabulary, topology, and structural statistics, creating domain shift for circuit graph neural networks (GNNs) without changing task labels. To study this setting, we introduce ReDIL-GNN, a resynthesis domain-incremental learning framework that adapts a fixed prediction or representation head as new synthesis styles arrive and evaluates retention on all previously observed domains. Because not every shift should be adapted blindly, ReDIL-GNN further introduces the Resynthesis Adaptability Index (RAI), a pre-adaptation score that combines adaptation need, source-equivalence recoverability, structural coverage, and update compatibility. We evaluate supervised hardware-security tasks and representation-learning models using task-native metrics for classifiers and source-equivalence retrieval metrics for embedding models, comparing naive fine-tuning with LwF, Online EWC, 
    
[^60]: 峰值感知的配电网多聚合层级短期负荷预测

    Peak-Aware Short-Term Load Forecasting Across Distribution Grid Aggregation Levels

    [https://arxiv.org/abs/2609.18588](https://arxiv.org/abs/2609.18588)

    该论文提出峰值感知的短期负荷预测评估框架，在英国和瑞士的开放数据集上，对配电网区域代码、二次变电站和低压馈线三个聚合层级的统计基线、机器学习模型和时序基础模型进行了比较，发现Chronos-2在高需求时期的预测性能最佳。

    

    对于配电系统运营商而言，短期负荷预测（STLF）支持拥堵管理、电压控制和资产保护。大多数现有方法关注所有时间步长的整体精度，而忽视了高需求（HD）时期的性能，而在这些时期，较大的预测误差会增加拥堵和电压违规的风险。本文研究了跨越三个与运营商相关的配电网聚合层级的峰值感知短期负荷预测，即区域代码（AC）、二次变电站（SUB）和低压（LV）馈线，使用了来自英国和瑞士的开放数据集。我们在峰值感知评估框架下比较了统计基线方法、机器学习模型（LightGBM 和 XGBoost）以及近期的时序基础模型（Chronos Bolt 和 Chronos-2），该框架使用 NMAE 和 MAPE 同时报告整体和高需求时期的预测性能。结果表明，Chronos-2 在所有层级上均实现了最佳的高需求时期预测性能。

    arXiv:2609.18588v1 Announce Type: new  Abstract: For distribution system operators, short-term load forecasting (STLF) supports congestion management, voltage control, and asset protection. Most existing approaches focus on overall accuracy across all time steps and neglect performance during high-demand (HD) periods, where larger forecast errors can increase the risk of congestion and voltage violations. In this paper, we study peak-aware STLF across three operator-relevant distribution grid aggregation levels, area codes (AC), secondary substations (SUB), and low-voltage (LV) feeders, using open datasets from the United Kingdom and Switzerland. We compare statistical baselines, machine learning models (LightGBM and XGBoost), and recent time-series foundation models (Chronos Bolt and Chronos-2) under a peak-aware evaluation framework that reports both overall and HD forecasting performance using NMAE and MAPE. The results show that Chronos-2 achieves the best HD performance across all
    
[^61]: 无标签引导：将测试时强化学习压缩至仅偏置参数子空间

    Label-free steering: Compressing test-time reinforcement learning into bias-only subspaces

    [https://arxiv.org/abs/2609.18587](https://arxiv.org/abs/2609.18587)

    该论文提出无标签仅偏置测试时强化学习方法，以多数投票伪标签为奖励、仅优化约10万个偏置参数，即可在数学、视觉语言和音频推理等多个任务上达到与全参数方法相当甚至更优的性能。

    

    测试时强化学习（TTRL）使模型能够在不依赖标注训练数据的情况下改进其推理能力，但现有方法通常需要优化模型的大部分参数。这引出一个自然的问题：当奖励信号和优化空间都受到严格限制时，有效的测试时自适应是否仍然可能出现？我们用无标签仅偏置TTRL（label-free bias-only TTRL）回答了这一问题，该方法使用多数投票伪标签作为奖励，仅优化约10万个偏置参数，同时保持预训练主干网络完全冻结。在MATH-500上，我们的方法达到76.67%的准确率，略高于我们自行复现的带标签偏置引导方法，同时比全参数TTRL少优化76,000倍的参数。相同的训练过程还能在视觉语言和音频推理任务上提升性能，包括MathVista、AI2D、LogicVista和MMAU。我们进一步表明，学习到的引导向量（摘要在此处被截断）

    arXiv:2609.18587v1 Announce Type: cross  Abstract: Test-time reinforcement learning (TTRL) enables models to improve their reasoning without relying on labeled training data, but existing approaches typically optimize a large fraction of the model parameters. This raises a natural question: can effective test-time adaptation emerge when both the reward signal and the optimization space are severely restricted? We answer this question with label-free bias-only TTRL, which uses majority-vote pseudo-labels as rewards and optimizes only approximately 100K bias parameters while keeping the pretrained backbone frozen. On MATH-500, our approach reaches 76.67% accuracy, slightly exceeding our own labeled bias-steering reproduction while optimizing 76,000x fewer parameters than full-parameter TTRL. The same training procedure improves performance across vision-language and audio reasoning tasks, including MathVista, AI2D, LogicVista, and MMAU. We further show that the learned steering vectors t
    
[^62]: TTM-Bench：一个文本到音乐系统性能基准测试框架

    TTM-Bench: A Framework for Text-to-Music System Performance Benchmarking

    [https://arxiv.org/abs/2609.18585](https://arxiv.org/abs/2609.18585)

    该论文提出了TTM-Bench框架，为文本到音乐系统定义了统一的、可复现的性能基准测试协议，从音乐内容对齐度和计算效率两个维度实现了跨系统的可靠性能比较。

    

    文本到音乐（TTM）系统正被越来越多地用于从自然语言描述生成音乐音频。因此，稳健的评估至关重要，然而可靠的性能比较仍然具有挑战性。这一困难源于系统架构、支持的调节信息和访问模式方面的差异，以及无法在不同系统间统一应用的异构且碎片化的评估指标。为应对这些挑战，我们提出了TTM-Bench，这是一个为当代TTM系统的系统性、可复现性能基准测试定义通用协议的框架。它从两个维度评估性能：音乐内容对齐度，通过可解释的语义、流派和音乐描述符一致性分数对照统一的音乐规范进行量化，并由一个综合分数加以汇总；以及计算效率，以生成延迟、实时因子和资源使用为特征。

    arXiv:2609.18585v1 Announce Type: cross  Abstract: Text-to-music (TTM) systems are increasingly used to generate musical audio from natural-language descriptions. Robust evaluation is therefore essential, yet reliable performance comparison remains challenging. This difficulty stems from differences in system architecture, supported conditioning information, and access mode, as well as heterogeneous and fragmented metrics that cannot be applied uniformly across systems. To address these challenges, we introduce TTM-Bench, a framework that defines a common protocol for systematic, reproducible performance benchmarking of contemporary TTM systems. It evaluates performance along two dimensions: musical-content alignment, quantified by interpretable semantic, genre, and musical-descriptor agreement scores against a common musical specification and summarized by an aggregate score; and computational efficiency, characterized by generation latency and real-time factor, alongside resource use
    
[^63]: 基于递归张量Sketch的少随机比特精确迹估计方法

    Accurate Trace Estimation with Fewer Random Bits via Recursive TensorSketch

    [https://arxiv.org/abs/2609.18577](https://arxiv.org/abs/2609.18577)

    本文提出基于递归张量Sketch的方法，通过大幅减少随机比特的消耗，实现了对仅能通过矩阵-向量乘积访问的隐式矩阵迹的精确估计。

    

    我们考虑估计隐式矩阵 $\mathbf{A} \in \mathbb{R}^{d^p\times d^p}$ 的迹的问题，该矩阵只能通过矩阵-向量乘积查询来访问。Hutchinson迹估计器是解决此问题的经典sketching方法。其估计量 $H_{m}(\mathbf{A}) = \frac{1}{m} \sum_{i=1}^{m} {\mathbf{z}^{(i)}}^T \mathbf{A} \mathbf{z}^{(i)}$，其中 $\mathbf{z}^{(i)}\in \mathbb{R}^{d^p}$，且 $z^{(i)}_j \in {N}(0, 1), j\in [d^p]$，满足以下保证：(i) $\mathbb{E}[H_{m}(\mathbf{A})]=\operatorname{tr}(\mathbf{A})$（无偏性），以及 (ii) $\mathrm{Var}[H_{m}(\mathbf{A})]=\frac{2}{m}||\mathbf{A}||_F^2$（方差上界）。然而，生成一个查询向量 $\mathbf{z}^{(i)}$ 需要 $O(d^p)$ 个随机比特；因此，$m$ 次查询总共需要 $O(md^p)$ 个随机比特，这在大规模应用中代价可能过于高昂。Meyer等人的近期工作……

    arXiv:2609.18577v1 Announce Type: new  Abstract: We consider the problem of estimating the trace of an implicit matrix $\mathbf{A} \in \mathbb{R}^{d^p\times d^p}$ that can only be accessed through matrix-vector products queries. The \textit{Hutchinson trace estimator}% ~\cite{Girard1987algorithme, article-hutchinson} is a classical sketching method for this problem. Their estimator, $H_{m}(\mathbf{A}) = \frac{1}{m} \sum_{i=1}^{m} {\mathbf{z}^{(i)}}^T \mathbf{A} \mathbf{z}^{(i)}, \quad \text{where } \ {\mathbf{z}^{(i)}}\in \mathbb{R}^{d^p}$, and $z^{(i)}_j \in {N}(0, 1), j\in [d^p]$, satisfies the following guarantees: (i) $\mathbb{E}[H_{m}(\mathbf{A})]=\operatorname{tr}(\mathbf{A})$, and (ii) $\mathrm{Var}[H_{m}(\mathbf{A})]=\frac{2}{m}||\mathbf{A}||_F^2$. Generating one query vector $\mathbf{z}^{(i)}$ requires $O(d^p)$ random bits; thus, $m$ queries require $O(md^p)$ random bits, which can be prohibitive in large-scale applications. Recent work by Meyer et al.~\cite{meyer2025hutchinso
    
[^64]: 从全息中费米子谱函数深度学习涌现时空

    Deep learning emergent spacetime from fermionic spectral functions in holography

    [https://arxiv.org/abs/2609.18566](https://arxiv.org/abs/2609.18566)

    提出基于神经常微分方程的物理信息机器学习框架，直接从边界费米子谱函数重构带电AdS黑洞的体时空几何与规范场，可可靠覆盖非费米液体、边际费米液体和类费米液体三种量子临界区域，并揭示仅由近视界AdS₂几何数据决定的时空简并性。

    

    我们提出了一个基于神经常微分方程（Neural Ordinary Differential Equations）的物理信息机器学习框架，用于解决全息逆问题：直接从边界费米子谱函数重构带电AdS黑洞的体时空和规范场。通过将紫外渐近行为、视界正则性以及零温度极端性作为硬约束编码到神经网络架构中，我们的框架能够可靠地重构由U(1)探测电荷所设定的三个量子临界区域——非费米液体、边际费米液体（奇异金属）以及类费米液体态——中的极端Reissner-Nordström AdS几何，并能以亚百分比精度联合推断出探测电荷本身。放松近AdS边界约束后，我们发现了一种几何简并性：在整个径向方向上各不相同但共享相同近视界AdS₂ × ℝ²数据的体时空剖面，会产生相同的（输出）

    arXiv:2609.18566v1 Announce Type: cross  Abstract: We present a physics-informed machine learning framework based on Neural Ordinary Differential Equations that solves the holographic inverse problem: reconstructing the bulk spacetime and gauge field of a charged AdS black hole directly from boundary fermionic spectral functions. Encoding the UV asymptotics, horizon regularity, and zero temperature extremality as hard constraints in the neural network architecture, our framework reliably reconstructs the extremal Reissner-Nordstr\"om AdS geometry across three quantum critical regimes set by the $U(1)$ probe charge---non-Fermi liquid, marginal Fermi liquid (strange metal), and Fermi-liquid-like states---and can jointly infer the probe charge itself to sub-percent accuracy. Relaxing the near-AdS boundary constraint uncovers a geometrical degeneracy: bulk profiles that differ throughout the radial direction but share the same near-horizon $AdS_2 \times \mathbb{R}^2$ data reproduce identic
    
[^65]: 用于合成语言生成的变分量子Transformer架构

    Variational Quantum Transformer Architecture for Synthetic Language Generation

    [https://arxiv.org/abs/2609.18565](https://arxiv.org/abs/2609.18565)

    提出了一种兼容NISQ设备的紧凑变分量子Transformer架构，通过用量子编码器、连接器和解码器电路替代经典注意力与前馈子层，能够端到端训练并学习非平凡的语法结构，在合成语言生成任务上实现完美确定性生成和高词典序有效性。

    

    我们提出了一种紧凑的、兼容NISQ设备的量子Transformer架构，用于合成量子自然语言处理（QNLP）序列建模。该模型保留了经典Transformer的自回归下一词元预测接口，但用变分量子编码器模块、连接电路、解码器模块以及直接的双量子比特测量读出取代了注意力机制和前馈子层。词元上下文通过角度编码进入小型量子寄存器，由并行的变分头和编码器集成电路进行处理，并通过解码器辅助量子比特进行条件化，从而产生四词元词汇表上的概率分布。我们在确定性和词典序语法生成任务上，以紧凑的经典Transformer为基线，评估了多种架构变体。量子模型可以端到端训练并学习非平凡的语法结构，包括个别运行中实现完美的确定性生成，以及最强架构变体中展现出高词典序有效性。

    arXiv:2609.18565v1 Announce Type: cross  Abstract: We propose a compact NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves the autoregressive next-token interface of a classical transformer, but replaces attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct two-qubit measurement readout. Token contexts are angle-encoded into small quantum registers, processed by parallel variational heads and encoder integration circuits and conditioned through decoder ancillae to produce a distribution over a four-token vocabulary. We evaluate several architecture variants on deterministic and lexicographic grammar-generation tasks against a compact classical transformer baseline. The quantum models are trainable end-to-end and learn nontrivial grammar structure, including perfect deterministic generation in individual runs and high lexicographic validity in the strongest va
    
[^66]: 人工智能中“性”的演化：面向多代模型种群的群体遗传学框架

    The evolution of sex for artificial intelligence: a population-genetic framework for multigenerational model populations

    [https://arxiv.org/abs/2609.18560](https://arxiv.org/abs/2609.18560)

    该论文首次将群体遗传学（包括有性/无性生殖和Wright-Fisher过程）形式化地应用于多代AI模型种群，证明了模型递归训练导致的“模型崩溃”在遗传学上可精确类比，且该框架在多种网络架构和大语言模型中普遍成立。

    

    人工智能发展的某些方面类似于一种种群过程：模型被专门化、在同伴的输出上进行再训练，或通过对权重取平均的方式相互组合。这些做法产生了生物学意义上的模型世代，即群体遗传学所研究的世代。在此，我发展了这一平行关系，并以有性生殖和无性生殖的视角来解释多代模型种群，从而正式地将这两个领域结合起来。我在一个精确的遗传模型中、在经过训练的网络（循环网络、前馈网络和变分自编码器生成器）中以及在大语言模型中检验了这些类比，结果表明这些类比普遍成立，同时存在一些可测量的架构特定偏差。已知在模型输出上进行递归训练会导致模型崩溃，这一过程此前被描述为类似于遗传漂变；我进一步发展了这一理论。一个学习者在父代输出上进行再训练的最小模型精确地再现了Wright-Fisher过程。

    arXiv:2609.18560v1 Announce Type: new  Abstract: Some aspects of AI development resemble a population process in which models are specialised, retrained on the output of peers, or combined by averaging weights. These practices lead to generations of models, in the biological sense studied by population genetics. Here, I develop this parallelism and interpret multigenerational model populations in terms of sexual and asexual reproduction, formally recombining the two fields. I test these analogies in an exact inheritance model, in trained networks (recurrent, feedforward and variational autoencoder generators) and in large language models, and show that they hold generally, with some measurable architecture-specific biases.   Training recursively on model output is known to lead to model collapse, a process previously described as akin to genetic drift; I develop all that follows. A minimal model of a learner retrained on its parent's output reproduces the Wright-Fisher process exactly;
    
[^67]: 基于可解释图块的深度学习用于集合模拟的野火蔓延预测

    Interpretable Patch-Based Deep Learning for Wildfire Spread Prediction from Ensemble Simulations

    [https://arxiv.org/abs/2609.18555](https://arxiv.org/abs/2609.18555)

    该研究比较了四种深度学习架构作为昂贵物理模拟器的低成本替代方案来预测野火蔓延，发现地表燃料载荷是唯一有效的预测变量（可将误差降低21%），并通过可解释性实验验证了模型学习。

    

    野火蔓延传统上使用基于物理的模拟器进行预测，这类模拟器具有物理可解释性，但每增加一个集合成员，其计算成本都会随之增加。我们探究深度学习代理模型能否以极低的成本复现这些模拟，并使用西班牙加泰罗尼亚Rectoret地区2米分辨率的10,584个火灾蔓延模拟数据进行训练。研究比较了四种架构：基于图块的U-Net、迁移学习的ResNet-50、受风驱动平流方程约束的物理信息网络，以及Swin-Unet Transformer。在地形和植被变量中，只有地表燃料载荷对燃烧概率具有较强预测能力（r = 0.27），纳入该变量可将预测误差降低21%。其余变量相关性较弱且高度冗余。此外，通过显著性、遮挡和旋转实验证明了模型的有效学习。卷积模型主要依赖于……

    arXiv:2609.18555v1 Announce Type: cross  Abstract: Wildfire spread is traditionally predicted using physics-based simulators, which are physically interpretable but whose cost increases with each additional ensemble member. We ask how well deep learning surrogates can reproduce these simulations at a fraction of this cost, training them on 10,584 fire spread simulations at 2m resolution for the Rectoret region in Catalonia, Spain. Four architectures are compared: a patch-based U-Net, a transfer-learned ResNet-50, a physics-informed network constrained by the wind-driven advection equation and a Swin-Unet transformer. Among the terrain and vegetation variables, only surface fuel load predicts burn probability with any strength (r = 0.27) and including it lowers prediction error by 21%. The remaining variables correlate weakly and are highly duplicative. Next, an experiment with saliency, occlusion and rotation demonstrates the models' learning. Convolutional models rely primarily on dis
    
[^68]: 重新审视回音室检测的目标函数

    Revisiting the Objective of Echo Chamber Detection

    [https://arxiv.org/abs/2609.18545](https://arxiv.org/abs/2609.18545)

    本文首次基于集合函数傅里叶变换理论形式化了回音室检测的目标函数，提出可扩展的半定松弛求解算法，在合成和真实数据集上均优于现有方法。

    

    在本文中，我们研究了社交网络中回音室的检测问题，即识别出一组在某个话题上意见一致、但与其他节点意见相左的节点集合。我们认为该问题不同于社区检测等其他社交网络分析问题，也不同于最大图割和最大团等其他图论问题。据我们所知，我们是首个利用集合函数的傅里叶变换理论（Stobbe 和 Krause，2012）对回音室检测的目标函数进行形式化定义的研究。我们提出了一种可扩展的半定松弛方法，并通过内点法和稀疏线性代数进行求解。实验结果表明，在小型合成实验中，我们的算法在恢复真实回音室方面优于竞争方法；在大型真实数据集上，我们的算法所检测出的回音室具有优于竞争方法的网络特性。

    arXiv:2609.18545v1 Announce Type: new  Abstract: In this paper, we study the detection of an echo chamber in a social network, i.e., the identification of a set of nodes that agree on a topic, while disagreeing with the rest of nodes. We argue that this problem is different from other social network analysis problems such as community detection, and from other graph problems such as maximum graph cut and maximum clique. To the best of our knowledge, we are the first to formalize the objective function of echo chamber detection, by using the theory of Fourier transforms of set functions (Stobbe and Krause, 2012). We propose scalable semidefinite relaxation, solved via an interior point method and sparse linear algebra. Experimentally, our algorithm recovers the ground truth echo chamber better than competing methods on small synthetic experiments. Our algorithm produces echo chambers with better network properties than competing methods on large real-world datasets. To independently val
    
[^69]: 具有潜在混杂因子的结构方程模型的可证明保证与高效学习

    Provable Guarantees and Efficient Learning of Structural Equation Models with Latent Confounders

    [https://arxiv.org/abs/2609.18535](https://arxiv.org/abs/2609.18535)

    本文针对含潜在混杂因子的线性结构方程模型，提出了一种通过将精度矩阵分解为稀疏加低秩两部分来迭代重建观测变量因果有向无环图的高效算法，并给出了可证明的正确性保证。

    

    因果发现旨在从观测数据中恢复变量之间的因果关系。在众多领域中，探索变量间的因果关系仍然是一个重要课题，但潜在混杂因子的存在使这一任务变得极具挑战性。忽略这些混杂因子可能导致虚假关联和错误的边方向。本文研究了带潜在混杂因子的线性结构方程模型。我们提出了一种算法，该算法迭代地识别终端（观测）节点，并重建观测变量的有向无环图。为此，我们将观测变量的精度矩阵恢复为稀疏加低秩矩阵的形式：稀疏矩阵刻画观测变量之间的条件依赖关系，而低秩矩阵刻画少量潜在混杂因子的综合影响。我们证明，对于p个观测变量、r个潜在混杂因子和s条边，我们的方法能够正确地识别出因果图结构。

    arXiv:2609.18535v1 Announce Type: new  Abstract: Causal discovery aims to recover causal relationships from observed data. In various fields, exploring causal relationships among variables remains an important topic, but this task becomes challenging due to the existence of latent confounders. Ignoring such confounders can lead to false associations and incorrect edge directions. In this paper, we study the linear structural equation model with latent confounders. We propose an algorithm that iteratively identifies terminal (observed) nodes and reconstructs the directed acyclic graph of the observed variables. To do this, we recover the precision matrix of the observed variables as a sparse plus low-rank matrix: a sparse matrix captures the conditional dependencies among observed variables, while a low-rank matrix captures the combined influence of a few latent confounders. We establish that for $p$ observed variables, $r$ latent confounders and $s$ edges, our procedure correctly ident
    
[^70]: 谱方法结构化预测的可证明保证

    Provable Guarantees for Spectral Structured Prediction

    [https://arxiv.org/abs/2609.18527](https://arxiv.org/abs/2609.18527)

    本文提出一种简单的谱方法，通过带噪符号邻接矩阵主特征向量的符号来恢复图节点标签，并给出与图结构无关的可证明理论保证，明确量化了谱间隙、节点数、度分布和噪声水平对标签恢复精度的影响。

    

    结构化预测是指同时预测多个标签的任务，广泛应用于自然语言处理和计算机视觉等诸多领域。本文研究了带边翻转噪声的符号图上的二值节点标签恢复问题（该模型由 Globerson 等人于 2015 年提出），并采用一种简单的谱方法，即从带噪符号邻接矩阵的主特征向量的符号来解码节点标签。我们提出了与图结构无关的理论保证，用于节点标签的近似推断，同时给出了预测结果相对于真实节点标签的最大角度偏差保证。通过利用矩阵集中性理论和特征向量扰动分析等工具，我们推导出了新的集中不等式，明确量化了邻接矩阵的谱间隙、节点数量、度分布以及噪声水平对结果的影响。作为推论，我们将这一通用结果与……（原文摘要在此处截断）

    arXiv:2609.18527v1 Announce Type: new  Abstract: Structured prediction is the simultaneous prediction of multiple labels, and is widely used in various fields, such as natural language processing and computer vision. In this paper, we study binary node label recovery on signed graphs with edge-flip noise, a model introduced by (Globerson et al., 2015), via a simple spectral method that decodes node labels from the signs of the principal eigenvector of the noisy signed adjacency matrix. We develop graph structure-agnostic theoretical guarantees for approximate inference of node labels as well as guarantees for maximum angle deviation with respect to the ground truth node labels. By leveraging tools from matrix concentration theory and eigenvector perturbation analysis, we derive new concentration inequalities that explicitly quantify the effect of the spectral gap of the adjacency matrix, number of nodes, degree distribution, and noise level. As a corollary, we relate our general result
    
[^71]: TRIPROBE：面向可解释人工智能（XAI）的分类之外任务可分性探测

    TRIPROBE: Probing Task Separability Beyond Classification for XAI

    [https://arxiv.org/abs/2609.18525](https://arxiv.org/abs/2609.18525)

    TriProbe 提出一个多层次探测框架，通过输入空间的基础探测器、特征表示的潜在探测器和分类器输出的最终探测器，结合最大费舍尔判别比指标，对多任务模型的可分性进行可解释诊断，定位瓶颈任务对并指导数据收集与架构设计。

    

    现代机器学习流程的评估往往仅归结为下游准确率，因而留下了任务为何成功或失败这一未解之问。TriProbe 通过一个多层次探测框架填补了这一空白，实现对任务可分性的可解释诊断。TriProbe 不将模型视为黑箱，而是追踪可分性如何在输入、学习到的特征以及最终分类器之间演化。它将多任务问题分解为二分类子任务，并应用三种互补的探测器：作用于输入空间的基础探测器、作用于特征表示的潜在探测器，以及作用于分类器输出的最终探测器。TriProbe 以最大费舍尔判别比作为有原则的可分性度量，识别瓶颈及受影响的任务对。在 Roshambo 表面肌电（sEMG）基准上的实验表明，TriProbe 能够揭示隐藏的性能崩溃，从而指导数据收集、验证和架构设计。

    arXiv:2609.18525v1 Announce Type: new  Abstract: Modern evaluation of learning pipelines often reduces to downstream accuracy, leaving open the question of why tasks succeed or fail. TriProbe addresses this gap with a multi-level probing framework for explainable diagnosis of task separability. Rather than treating models as black boxes, TriProbe traces how separability evolves across inputs, learned features, and final classifiers. It decomposes multi-task problems into binary subtasks and applies three complementary probes: a Foundational Probe on input spaces, a Latent Probe on feature representations, and a Final Probe on classifier outputs. Using Maximum Fisher's Discriminant Ratio as a principled separability metric, TriProbe identifies bottlenecks and affected task pairs. Experiments on the Roshambo sEMG benchmark show how TriProbe reveals hidden breakdowns, guiding data collection, validation, and architecture design.
    
[^72]: COMPASS-ABS：减少深度学习训练工作负载中共享GPU集群的资源碎片化

    COMPASS-ABS: Reducing Fragmentation in Shared GPU Clusters for Deep Learning Training Workloads

    [https://arxiv.org/abs/2609.18519](https://arxiv.org/abs/2609.18519)

    本文提出了独立于历史工作负载信息的碎片化度量指标SIF，并设计了COMPASS-ABS调度算法，通过将集群状态约束在基于锚点的紧凑空间内，持续降低共享GPU集群的资源碎片化，从而提升集群利用率并缩短深度学习训练作业的周转时间。

    

    随着深度学习技术的快速发展，共享GPU集群接收到的深度学习训练（DLT）作业日益增多。然而，资源碎片化导致此类集群利用率低下，并迫使运行在其上的DLT作业承受漫长的周转时间。大量研究致力于量化碎片化并开发缓解其影响的调度算法。然而，现有的碎片化度量方法在缺乏工作负载分布信息的情况下会失效，而当前的调度器无法持续将资源碎片化维持在较低水平。为解决这些问题，我们首先提出了调度器诱导碎片化（Scheduler-Induced Fragmentation, SIF），这是一种基于“部分节点”概念的碎片化度量标准，不依赖于历史工作负载知识。随后，我们提出COMPASS-ABS系统，它采用紧凑保证（COMPACT-ASSured，COMPASS）算法将集群状态约束在一个紧密的基于锚点的空间（Anchor-Based Space，ABS）内。

    arXiv:2609.18519v1 Announce Type: cross  Abstract: With the rapid advancement of deep learning technology, shared GPU clusters receive an increasing number of deep learning training (DLT) jobs. Yet resource fragmentation make such clusters underutilized and forces the DLT jobs running on them to endure long turnaround times. Extensive research has been devoted to quantifying fragmentation and developing scheduling algorithms that alleviate its impact. However, existing fragmentation measures break down in the absence of workload distribution information, while current schedulers cannot continuously maintain resource fragmentation at a low level. To tackle these problems, we first introduce Scheduler-Induced Fragmentation (SIF), a metric built on the notion of partial-nodes that is independent of historical workload knowledge. We then propose COMPASS-ABS, which employs the COMPact-ASSured (COMPASS) algorithm to confine the cluster state within a tight Anchor-Based Space (ABS), whose con
    
[^73]: 超越常规合规：狡猾数据培养大型语言模型的安全警觉性

    Beyond Routine Compliance: Cunning Data Cultivates Safety Vigilance in Large Language Models

    [https://arxiv.org/abs/2609.18515](https://arxiv.org/abs/2609.18515)

    本文提出通过训练模型识别包含误导性前提和非典型推理的“狡猾问题”，培养大型语言模型的安全警觉性，使其能够识破隐藏在良性语境下的有害意图，从而显著提升对越狱攻击的鲁棒性。

    

    安全对齐教导大型语言模型（LLMs）识别有害请求并拒绝危险指令。然而，当有害意图隐藏在看似良性的语境中时，对齐后的模型仍可能失效。因此，稳健的安全性不仅需要了解安全边界，还需要具备“警觉性”：即检测表面语义之下不寻常前提、误导性推理和潜在风险的能力。警觉性要求模型在行动之前审查请求的潜在意图和假设。为了培养这种能力，我们引入了“狡猾问题”，这些问题不一定与安全相关，但包含误导性前提、非典型推理或微妙的不一致性。我们假设，学会看穿这类推理陷阱的能力可以迁移到安全关键场景中。实验表明，狡猾训练提高了模型对分布外越狱攻击的鲁棒性，并强化了后续的安全性。

    arXiv:2609.18515v1 Announce Type: new  Abstract: Safety alignment teaches large language models (LLMs) to recognize harmful requests and reject risky instructions. Yet aligned models can fail when harmful intent is concealed within seemingly benign contexts. Robust safety therefore requires both knowledge of safety boundaries and \textbf{vigilance}: the ability to detect unusual premises, misleading reasoning, and latent risks beneath surface-level semantics. Vigilance requires models to scrutinize a request's underlying intent and assumptions before acting. To cultivate this capability, we introduce \textbf{cunning questions}, which are not necessarily safety-related but contain misleading premises, atypical reasoning, or subtle inconsistencies. We hypothesize that learning to look beyond such reasoning traps can transfer to safety-critical scenarios. Experiments show that Cunning training improves robustness to out-of-distribution jailbreak attacks and strengthens subsequent safety f
    
[^74]: ActiveScale：在模型、数据与硬件层面扩展机器人主动感知

    ActiveScale: Scaling Active Perception for Robots across Model, Data, and Hardware

    [https://arxiv.org/abs/2609.18514](https://arxiv.org/abs/2609.18514)

    ActiveScale通过模型（逐帧位姿令牌与位姿预测头）、数据（1000小时人-机器人中期训练）和硬件的协同设计，使VLA模型能够跨视点推理并主动获取信息性观测，实现机器人主动感知。

    

    当固定视点导致任务相关信息被遮挡或未被观察到时，主动感知对机器人操作至关重要。然而，让视觉-语言-动作（VLA）模型能够跨变化的视点进行推理并主动获取有信息量的观测仍然具有挑战性。我们提出了ActiveScale，这是一个通过模型、数据和硬件协同设计来推进主动感知的框架。我们的模型通过历史视频观测和显式的相机位姿监督来增强VLA，使用逐帧位姿令牌和轻量级预测头来关联跨视点的观测，从而支持对场景的连贯理解。为了从人类活动中自然存在的相机运动中学习，我们引入了一个可扩展的人-机器人中期训练方案，使用1000小时的第一人称视角和机器人数据，使模型适应时序输入和位姿监督。我们进一步引入了主动感知移动[摘要不完整]

    arXiv:2609.18514v1 Announce Type: cross  Abstract: Active perception is essential for robotic manipulation when fixed viewpoints leave task-relevant information occluded or unobserved. However, enabling vision-language-action (VLA) models to reason across changing viewpoints and actively acquire informative observations remains challenging. We present ActiveScale, a framework that advances active perception through coordinated model, data, and hardware designs. Our model augments a VLA with historical video observations and explicit camera-pose supervision, using per-frame pose tokens and a lightweight prediction head to associate observations across viewpoints and support a coherent understanding of the scene. To learn from the camera motion naturally present in human activity, we introduce a scalable human--robot mid-training recipe using 1000 hours of egocentric and robotic data, adapting the model to temporal inputs and pose supervision. We further introduce Active-perception Mobil
    
[^75]: 从分布式“眼睛”中学习：利用协同感知实现自动化模型适配

    Learning from Distributed Eyes: Leveraging Collaborative Perception for Automated Model Adaptation

    [https://arxiv.org/abs/2609.18511](https://arxiv.org/abs/2609.18511)

    提出LDE框架，将车路协同感知转化为高质量监督信号用于生成伪标签，从而实现自动驾驶感知模型在新环境下的自动化无监督适配，并解决了通信瓶颈、视角差异和可靠性等关键挑战。

    

    在自动驾驶中，感知模型常常由于域偏移而难以泛化到新环境。虽然无监督模型适配提供了一种无需耗费人力进行手动标注的可行解决方案，但现有的仅依赖自车数据的方法往往导致较差的伪标签性能。为了解决这一关键问题，我们提出了LDE（从分布式“眼睛”中学习），这是一个新颖的框架，将协同感知（CP）转化为模型适配的高质量监督来源。这种伪标签方法对超参数不敏感且相对可靠，其前提假设是协同感知通常优于单智能体感知。然而，简单地实现这种方法会遇到以下问题：(1) 在时间和带宽约束下共享丰富特征的通信瓶颈；(2) 协同感知视角与学习者的视场（FoV）之间的视角差异；以及 (3) 在某些情况下依然存在的不可靠性。

    arXiv:2609.18511v1 Announce Type: cross  Abstract: In autonomous driving, perception models often struggle to generalize to new environments due to domain shifts. While unsupervised model adaptation offers a feasible solution without labor-intensive manual labeling, existing methods that rely solely on the ego-vehicle's data often lead to inferior pseudo-labeling performance. To address this critical issue, we propose LDE, Learning from Distributed ``Eyes", a novel framework that transforms collaborative perception (CP) into a source of high-quality supervision for model adaptation. This pseudo-labeling approach is hyperparameter-insensitive and relatively reliable, assuming CP often outperforms single-agent's perception. However, naively implementing this approach encounters (1) the communication bottleneck of sharing rich features under time and bandwidth constraints, (2) the view discrepancy between the CP view and the learner's Field of View (FoV), and (3) the unreliability even in
    
[^76]: 概率机器学习天气预报模型中的蝴蝶效应与动能级串

    Butterfly Effect and the Kinetic Energy Cascade in Probabilistic Machine Learning Weather Prediction Models

    [https://arxiv.org/abs/2609.18489](https://arxiv.org/abs/2609.18489)

    本研究通过动能谱分析揭示了当前最先进的概率机器学习天气预报模型虽能产生接近真实的动能谱量级，但大多无法像物理数值模型那样再现预期的动能向上尺度级串传递，且所有模型均表现出向上尺度增长的蝴蝶效应式误差传播。

    

    本研究分析了四个最先进的概率机器学习天气预报（MLWP）模型中的动能（KE）谱、差分动能（DKE）谱以及动能跨空间尺度传递的特征，这四个模型分别为：NeuralGCM-ENS、FourCastNet 3、AIFS-ENS 和 GenCast，并将结果与基于物理的数值天气预报模型 IFS-ENS 进行了比较。虽然 NeuralGCM-ENS 成功再现了预期的动能向上尺度传递，但其编码器阶段的噪声注入低估了中尺度动能。相反，AIFS-ENS、GenCast 和 FourCastNet 3 产生了符合实际的动能谱量级，但未能捕获预期的动能向上尺度传递。特别是，采用空间不相关随机扰动的 AIFS-ENS 和 GenCast 表现出动能在高波数处的增强累积。所有被研究的模型均表现出向上尺度的误差增长，这体现为 DKE 谱峰逐渐向更大尺度偏移。

    arXiv:2609.18489v1 Announce Type: cross  Abstract: This study analyses kinetic energy (KE) spectra, difference kinetic energy (DKE) spectra, and signatures of KE transfer across spatial scales in four state-of-the-art probabilistic machine learning weather prediction (MLWP) models: NeuralGCM-ENS, FourCastNet 3, AIFS-ENS, and GenCast. Results are compared with those from the physics-based numerical weather prediction model IFS-ENS. While NeuralGCM-ENS successfully reproduces the expected upscale transfer of KE, noise injection at its encoder stage underestimates mesoscale KE. Conversely, AIFS-ENS, GenCast, and FourCastNet 3 produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE. In particular, AIFS-ENS and GenCast, which employ spatially uncorrelated stochastic perturbations, exhibit enhanced accumulation of KE at high wavenumbers. All examined models exhibit upscale error growth, reflected by the progressive shift of the DKE spectral peak toward
    
[^77]: 超越随机耦合：生成流中的对比噪声对齐

    Beyond Random Couplings: Contrastive Noise Alignment in Generative Flows

    [https://arxiv.org/abs/2609.18488](https://arxiv.org/abs/2609.18488)

    提出对比噪声对齐方法，通过将噪声批次建模为相互作用粒子系统并利用跨模态InfoNCE目标直接优化噪声表示，创建动态对比耦合以改善生成流模型的训练。

    

    扩散模型和流匹配模型通常通过独立采样的高斯噪声破坏数据来进行训练。虽然这种前向过程简单且易于扩展，但它会产生任意的数据-噪声耦合，迫使网络学习不相关端点之间的高曲率传输。现有的最优传输方法通过将固定的噪声样本重新分配给数据来减轻这一负担，但源噪声分布本身仍然是被动的。为了解决这一问题，我们提出了对比噪声对齐，这是一种训练时方法，通过直接优化噪声表示来创建动态的对比耦合。通过将噪声批次建模为相互作用的粒子系统，CNA采用跨模态InfoNCE目标将噪声粒子与其配对的数据目标对齐。为了防止空间坍缩，这种对齐通过角度熵项和径向范数惩罚进行正则化。我们从理论上证明，这种平衡……

    arXiv:2609.18488v1 Announce Type: cross  Abstract: Diffusion and flow-matching models are typically trained by corrupting data through independently sampled Gaussian noise. While simple and scalable, this forward process induces arbitrary data-noise couplings, forcing the network to learn high-curvature transports between unrelated endpoints. Existing optimal-transport methods reduce this burden by reassigning fixed noise samples to data, but the source noise distribution itself remains passive. To address this, we introduce Contrastive Noise Alignment (CNA), a training-time method that creates dynamic, contrastive couplings by optimizing the noise representations directly. By modeling the noise batch as an interacting particle system, CNA employs a cross-modal InfoNCE objective to align noise particles with their paired data targets. To prevent spatial collapse, this alignment is regularized using an angular entropy term and a radial norm penalty. We show theoretically that this equil
    
[^78]: 面向生物医学知识图谱鉴别诊断的双曲图表示学习

    Hyperbolic Graph Representation Learning for Differential Diagnosis on Biomedical Knowledge Graphs

    [https://arxiv.org/abs/2609.18481](https://arxiv.org/abs/2609.18481)

    该研究表明双曲图嵌入能以远低于欧几里得方法的维度有效利用生物医学知识图谱的层次结构，并支持在整合患者信息的异构图上进行孟德尔疾病的鉴别诊断。

    

    生物医学知识图谱将本体裁衍生的层次结构与表型、疾病、基因、蛋白质和患者等异构实体之间的横向关联相结合。这种混合结构引出了一个问题：能够自然捕捉树状组织的双曲嵌入在纯层次图之外是否仍然有用。我们针对在整合患者信息的生物医学图上进行孟德尔疾病鉴别诊断，开展了一项双曲图表示学习的初步研究。在孤立本体子图上的实验表明，双曲模型在远低于欧几里得基线的维度下即可取得强劲性能。随后，我们在一项为每位患者对候选疾病进行排序的链接预测任务上评估了这些模型。结果表明，双曲嵌入既能利用生物医学的层次结构，又能支持在异构的患者级图上进行诊断推理。

    arXiv:2609.18481v1 Announce Type: new  Abstract: Biomedical knowledge graphs combine ontology-derived hierarchies with transversal associations among heterogeneous entities such as phenotypes, diseases, genes, proteins, and patients. This hybrid structure raises the question of whether hyperbolic embeddings, which naturally capture tree-like organization, remain useful beyond purely hierarchical graphs. We present a preliminary study of hyperbolic graph representation learning for Mendelian-disease differential diagnosis on a patient-integrated biomedical graph. Experiments on isolated ontology subgraphs show that hyperbolic models achieve strong performance in substantially lower dimensions than Euclidean baselines. We then evaluate the models on a link-prediction task that ranks candidate diseases for each patient. Results suggest that hyperbolic embeddings can exploit biomedical hierarchical structure while supporting diagnostic reasoning over heterogeneous patient-level graphs.
    
[^79]: 空间自适应噪声注入

    Spatially Adaptive Noise Injection

    [https://arxiv.org/abs/2609.18466](https://arxiv.org/abs/2609.18466)

    本文提出空间自适应噪声注入（SANI）采样框架，通过概率门控机制和空间自适应方差在逐像素层面动态调整噪声注入，在去噪器不确定的边缘纹理区域施加随机校正，而在分数估计精确的平滑区域保持确定性更新。

    

    扩散采样器通过随机（DDPM）或确定性（DDIM）更新来逆转学习到的加噪过程，这两者代表了由一个标量噪声注入方差所控制的单一族的两个端点，且该方差在每个空间位置上以相同方式应用。这种统一的方法忽略了自然图像的几何特性：边缘和纹理等高曲率区域——去噪器在这些区域不确定性较高——能够受益于随机校正，而分数估计精确的平滑区域则会因注入噪声而退化。本工作研究了在给定时间步长上是否每个像素都需要随机校正，并提出了一种新颖的采样框架——空间自适应噪声注入（SANI），它可以在逐像素的基础上动态调整噪声的应用。SANI 将概率门控机制与推导得出的空间自适应方差相结合，确保仅在需要的位置注入噪声，从而细化复杂的图像特征。

    arXiv:2609.18466v1 Announce Type: new  Abstract: Diffusion samplers reverse a learned noising process using either stochastic (DDPM) or deterministic (DDIM) updates, which represent endpoints of a single family controlled by a scalar noise-injection variance that is applied identically at every spatial location. This uniform approach neglects the geometry of natural images: high-curvature regions such as edges and textures, where the denoiser is uncertain, benefit from stochastic correction, whereas smooth regions, where the score is precise, are degraded by injected noise. This work investigates whether each pixel requires stochastic correction at a given timestep and introduces Spatially Adaptive Noise Injection (SANI), a novel sampling framework that dynamically adjusts noise application on a per-pixel basis. SANI integrates a probabilistic gating mechanism with a derived spatially adaptive variance, ensuring that noise is injected precisely where needed to refine complex features w
    
[^80]: 通过潜在神经符号推理解耦长期记忆

    Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning

    [https://arxiv.org/abs/2609.18461](https://arxiv.org/abs/2609.18461)

    提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。

    

    个性化智能体需要对长期历史交互进行推理，以同时推断显式偏好和隐式行为证据。早期的平面检索方法独立地对记忆片段进行评分，忽略了分布式信息；而当前的结构化记忆框架依赖于与查询无关的静态图，无法捕捉上下文相关的关系。至关重要的是，原始文本记忆本质上是纠缠且嘈杂的，使得细粒度个性化和跨会话推理在计算上变得难以实现。为此，我们提出了LGM，这是一种新颖的神经符号框架，将长期记忆解耦转移到连续潜在空间中。具体而言， 我们设计了一种基于稀疏自编码器的定制化潜在图构建方法，而非持久化固定图，它根据每个查询将历史交互映射为潜在记忆节点，并将记忆轨迹解耦为稀疏概念……

    arXiv:2609.18461v1 Announce Type: new  Abstract: Personalized agents are required to reason over long-term history interactions to infer both explicit preferences and implicit behavioral evidence. While early flat retrieval methods score memory fragments independently and neglect the distributed information, current structured memory frameworks rely on query-agnostic static graphs that fail to capture the context-dependent relations. Crucially, raw textual memories are inherently entangled and noisy, making fine-grained personalization and cross-session reasoning computationally prohibitive. To this end, we present LGM, a novel neuro-symbolic framework that shifts long-term memory disentanglement into a continuous latent space. Specifically, (i) instead of persisting fixed graphs, we design a tailored latent graph construction with a sparse autoencoder. Subject to each query, it maps historical interactions into latent memory nodes and disentangles the memory traces into sparse concept
    
[^81]: 面向自动驾驶选择性轨迹规划的风险感知世界建模与流引导占据演化方法

    Risk-Aware World Modeling with Flow-Guided Occupancy Evolution for Selective Trajectory Planning in Automated Driving

    [https://arxiv.org/abs/2609.18442](https://arxiv.org/abs/2609.18442)

    提出了RiskWorld风险感知世界建模框架，通过流引导占据演化实现共享占据预测，仅在额外预测风险触发干预且替代轨迹满足逐分量约束时才进行选择性轨迹替换，从而提升自动驾驶运动规划的安全性。

    

    自动驾驶中的安全运动规划需要预判不断演变的交通风险，并决定何时修改当前规划的轨迹。我们提出了RiskWorld，一个面向共享占据预测和选择性轨迹替换的风险感知世界建模框架。该框架将空间风险场和时间参与者上下文与视觉鸟瞰图特征进行融合。流引导演化机制传输占据和场景特征，同时利用带符号残差在传输后对占据进行修正。每个规划步骤仅生成一次预测，并在各候选之间复用。每个候选轨迹与当前状态持续参考进行比较，从而产生非负的碰撞分数修正。由当前世界评估选出的轨迹作为规划锚点，只有当额外预测的风险触发干预、且替代轨迹满足对预测风险和轨迹误差的逐分量约束时，该锚点轨迹才会被替换。

    arXiv:2609.18442v1 Announce Type: new  Abstract: Safe motion planning in automated driving requires anticipating evolving traffic risks and deciding when to revise the current planned trajectory. We introduce RiskWorld, a risk-aware world modeling framework for shared occupancy forecasting and selective trajectory replacement. Spatial risk fields and temporal actor context are fused with visual bird's-eye-view features. Flow-guided evolution transports occupancy and scene features, while signed residuals correct occupancy after transport. One forecast is generated per planning step and reused across candidates. Each candidate is compared with a current-state persistence reference, yielding a nonnegative collision-score correction. The trajectory selected by current-world evaluation serves as the planning anchor and is replaced only when additional predicted risk triggers intervention and an alternative satisfies component-wise constraints on predicted risk and trajectory error. Candida
    
[^82]: HPOQuest：一种基于主动表型获取的罕见病诊断智能体

    HPOQuest: A Rare-Disease Diagnostic Agent Using Active Phenotype Acquisition

    [https://arxiv.org/abs/2609.18431](https://arxiv.org/abs/2609.18431)

    HPOQuest是一个免训练的罕见病诊断框架，通过迭代式主动获取信息性表型来不断更新疾病概率排名，将稀疏初始表型下的诊断Recall@1提升高达30个百分点。

    

    全球有超过3亿人患有7000多种已知罕见病之一，但由于患者最初表现出的表型不完整且具有异质性，诊断仍然十分困难。我们提出了HPOQuest，这是一个用于罕见病诊断中顺序表型获取的免训练框架。从少量已观察到的患者表型出发，HPOQuest维护一个概率化的疾病排名，并迭代地选择信息量大的后续问题，以在患者评估过程中为临床医生提供支持。经确认的表型会更新疾病排名，而所有应答都会更新候选问题集。在四个基准队列上，HPOQuest显著提升了基于稀疏初始表型的诊断能力，在Recall@1上提升高达30个百分点，在Recall@5上提升高达45个百分点。这些结果表明，顺序表型获取能够显著改善基于有限初始临床信息的罕见病诊断。

    arXiv:2609.18431v1 Announce Type: new  Abstract: More than 300 million people worldwide are affected by one of over 7,000 known rare diseases, yet diagnosis remains difficult because patients initially present with incomplete and heterogeneous phenotypes. We present HPOQuest, a training-free framework for sequential phenotype acquisition in rare-disease diagnosis. Starting from a small set of observed patient phenotypes, HPOQuest maintains a probabilistic disease ranking and iteratively selects informative follow-up questions to support clinicians during patient assessment. Confirmed phenotypes update the disease ranking, while all responses update the candidate question set. Across four benchmark cohorts, HPOQuest substantially improves diagnosis from sparse initial phenotypes, with gains of up to 30% points at Recall@1 and 45% points at Recall@5. These results demonstrate that sequential phenotype acquisition can substantially improve rare-disease diagnosis from limited initial clini
    
[^83]: HiLNO：一种用于一般几何上偏微分方程的带多尺度监督的层次化潜在神经算子

    HiLNO: A Hierarchical Latent Neural Operator with Multi-Scale Supervision for PDEs on General Geometries

    [https://arxiv.org/abs/2609.18419](https://arxiv.org/abs/2609.18419)

    HiLNO提出了一种层次化潜在神经算子，通过构建从细到粗再到细的潜在空间、多尺度监督机制和各向异性高斯注意力，有效缓解了压缩过程中的信息丢失问题，实现了对一般几何上具有多尺度结构的偏微分方程的高效求解。

    

    潜在神经算子通过在紧凑的潜在表示上执行主要计算，提高了偏微分方程（PDEs）算子学习的效率。然而，直接压缩输入表示以获得此类紧凑表示可能会丢弃与解相关的空间信息，尤其是对于具有多尺度结构的PDE解。为了解决这一问题，我们提出了HiLNO，这是一种层次化潜在神经算子，它构建了从细到粗再到细的潜在空间，并进一步引入了多尺度监督（MSS）和各向异性高斯注意力。层次结构缓解了压缩过程中潜在的信息丢失，而多尺度监督将中间预测与下采样的目标场对齐，促使与解相关的结构在多个空间尺度上被捕获。各向异性高斯注意力实现了跨层次的特征传递，使HiLNO能够适用于一般几何上的PDE求解。

    arXiv:2609.18419v1 Announce Type: cross  Abstract: Latent neural operators improve the efficiency of operator learning for partial differential equations (PDEs) by performing the main computation on compact latent representations. However, directly compressing the input representation to obtain such compact representations may discard solution-relevant spatial information, especially for PDE solutions with multiscale structures. To address this problem, we propose HiLNO, a hierarchical latent neural operator that constructs a fine-to-coarse-to-fine latent space and further introduces multi-scale supervision (MSS) and anisotropic Gaussian attention. The hierarchy mitigates potential information loss during compression, while MSS aligns intermediate predictions with downsampled target fields, encouraging solution-relevant structures to be captured across multiple spatial scales. Anisotropic Gaussian attention enables feature transfer across the hierarchy, making HiLNO applicable to gener
    
[^84]: 基于记忆持久性的随机子空间梯度下降

    Gradient Descent with Stochastic Subspaces via Persistence of Memory

    [https://arxiv.org/abs/2609.18416](https://arxiv.org/abs/2609.18416)

    本文提出“记忆持久性”技术，利用一个与梯度弱相关、可长期固定无需频繁更新的指导向量来引导随机子空间的生成，从而显著扩展并改进了大规模优化中的随机子空间梯度下降方法，且该向量可借助稀疏性或小批量等结构化特性以低成本高效获得。

    

    随机子空间方法作为基于梯度下降的技术，在大规模优化问题中日益流行，尤其是在分布式环境中。本文引入“记忆持久性”技术，以极大地扩展和改进随机子空间方法。为此，我们利用一个与梯度仅弱相关的向量，为随机子空间的生成过程提供指导结构，而下降过程将沿着该随机子空间进行。该指导向量可以在大量迭代中保持固定，仅在较宽的间隔处刷新（我们可以根据问题参数对该间隔的大小提供理论保证）。在重要的机器学习场景中，例如涉及稀疏性或小批量结构的优化问题，我们表明可以利用结构化属性，以有效且计算成本低廉的方式获得该指导向量。

    arXiv:2609.18416v1 Announce Type: cross  Abstract: Stochastic subspace methods have gained popularity as gradient descent based techniques for large scale optimisation problems, especially in distributed settings. In this paper, we introduce the technique of "persistence of memory" to greatly extend and improve the random subspace methods. To this end, we leverage a vector that is only weakly correlated with the gradient in order to provide a guiding structure to the generative process of the random subspace along which the descent is going to take place. This guidance vector may be fixed for a large number of iterations, only to be refreshed at wide intervals (on whose size we can provide guarantees in terms of problem parameters). In important machine learning settings, such as optimisation problems embodying sparsity or a minibatch structure, we show that the guidance vector can be obtained in an effective and computationally inexpensive manner by leveraging the structured propertie
    
[^85]: TERN：一种用于疫情预测的带季节参考与在线适应的Delta规则记忆模型

    TERN: A Delta-rule Memory with a Seasonal Reference and Online Adaptation for Epidemic Forecasting

    [https://arxiv.org/abs/2609.18407](https://arxiv.org/abs/2609.18407)

    TERN是一种基于delta规则快速权重记忆的流感疫情预测模型，通过由疫情阶段特征驱动的门控擦除机制、显式季节参考和在线适应，在多个流感基准测试上超越了现有疫情图模型和通用预测器。

    

    每周的流感监测数据用于指导疫苗分发和公共卫生警报，但其预测十分困难。每个地区仅提供少数几个季节的数据，疫情波每年在时间和高度上都会发生变化，在疫情波上升期间有帮助的信息在峰值过后反而会产生误导，而上个季节的形态在一年内仍保持参考价值。现有的疫情图模型和通用预测器只读取短固定时间窗口，并同等对待所有历史信息，因此它们既无法利用更早季节的数据，也无法在疫情阶段发生变化时丢弃过时的关联。为了解决这些局限性，我们提出了TERN，这是一个围绕delta规则快速权重记忆构建的预测器：该记忆按通道进行衰减、沿学习到的地址进行擦除，并由局部疫情阶段特征驱动的门控机制控制，同时结合了显式的季节参考和在线适应机制。在三个Cola-GNN流感基准测试上，TERN的表现优于疫情图模型和通用预测器……

    arXiv:2609.18407v1 Announce Type: cross  Abstract: Weekly influenza surveillance counts guide vaccine distribution and public-health alerts, yet they are hard to forecast. Each region offers only a few seasons, waves shift in timing and height every year, and information that helps while a wave grows misleads after its peak, whereas last season's shape stays informative for a year. Existing epidemic graph models and general forecasters read a short fixed window and treat all past information alike, so they neither exploit earlier seasons nor discard stale associations when the epidemic phase changes. To address these limitations, we propose TERN, a forecaster built around a delta-rule fast-weight memory that decays channel-wise and erases along a learned address under gates driven by local epidemic-phase features, combined with an explicit seasonal reference and online adaptation. On three Cola-GNN influenza benchmarks, TERN outperformed epidemic graph models and general forecasters, m
    
[^86]: 可靠虚拟传感：面向传感器故障鲁棒性的多领域基准测试

    Reliable Virtual Sensing: A Multi-Domain Benchmark for Robustness Under Sensor Failures

    [https://arxiv.org/abs/2609.18396](https://arxiv.org/abs/2609.18396)

    本文提出了首个面向学习型虚拟传感中传感器故障鲁棒性的多领域基准MuViS-C，涵盖十种故障模式、多种严重程度以及平均误差、相对退化和最坏情况脆弱性等鲁棒性度量，并在六个领域的九个数据集上对六种架构进行了系统评测。

    

    虚拟传感是指从可用的传感器测量数据中估计难以直接测量的物理量，它是信息物理系统中控制与监测的关键使能技术。然而，当传感器发生故障时，基于学习的预测器可能产生物理上不合理的估计结果，并进而引发系统级的失效。我们认为实际部署要求模型具备鲁棒性，因此提出了MuViS-C——首个针对基于学习的虚拟传感中常见传感器故障的多领域鲁棒性基准。该基准建立在现有的标称性能基准和成熟的损坏分类体系之上，涵盖十种传感器故障模式，从细微的漂移到灾难性的信号中断，并设置了多种故障严重程度。这些故障模式与互补的鲁棒性度量相匹配，用以刻画故障条件下的平均误差、相对性能退化以及最坏情况下的脆弱性。在来自六个领域的九个数据集上，该基准对六种架构进行了评测，涵盖梯度提升……（摘要此处截断）

    arXiv:2609.18396v1 Announce Type: cross  Abstract: Virtual sensing, the estimation of hard-to-measure quantities from available sensor measurements, is a critical enabler for control and monitoring in cyber-physical systems. However, when sensors fail, learning-based predictors can produce physically implausible estimates that propagate to system-level failures. We argue that real-world deployment demands robustness and introduce MuViS-C, the first multi-domain benchmark of robustness against common sensor failures in learning-based virtual sensing. Building on an existing nominal-performance benchmark and established corruption taxonomies, it covers ten sensor failure modes, from subtle drifts to catastrophic signal dropouts, at multiple severities. These are paired with complementary robustness measures capturing average error under corruption, relative degradation, and worst-case fragility. Across nine datasets from six domains, we benchmark six architectures spanning gradient-boost
    
[^87]: 每个固定指标都有盲区：一种用于评分天气预报真实性的学习型大气评论器

    Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism

    [https://arxiv.org/abs/2609.18381](https://arxiv.org/abs/2609.18381)

    该论文提出通过训练判别器来生成类似散度的真实性评分，能够自适应地检测天气预报模型表现出的任何失败模式，从而克服固定指标存在的盲区。

    

    尽管机器学习天气预报模型在逐点指标上具有很高的精度，但它们可能表现出多种失败模式，例如模糊、周期性不规则以及其他非物理的空间伪影。这促使人们提出了各种指标来检测已知的失败情况。现有指标预先固定了某种表示或变换，而这种选择限制了它们能够检测到的伪影类型。我们提出训练一个判别器来区分参考数据和模型输出，并利用其输出logit获得一种类似散度的真实性评分。该判别器能够学习任何能将模型输出场与真实天气区分开的特征，从而适应该模型所表现出的任何失败模式。我们通过对ERA5再分析数据施加各种合成破坏，将我们的学习型大气评论器与现有指标进行比较。我们的方法成功识别了这些破坏并对其严重程度进行排序，而现有指标在至少……

    arXiv:2609.18381v1 Announce Type: new  Abstract: Despite their high accuracy on point-wise metrics, machine learning weather forecasting models can exhibit different failure modes such as blurring, periodic irregularities, and other unphysical spatial artifacts. This has motivated a variety of metrics to detect known failure cases. Existing metrics fix a representation or transformation in advance, and that choice limits the artifacts they can detect. We propose to train a discriminator for separating reference data from the model's output, and using its output logit to obtain a divergence-like realism score. The discriminator learns whatever separates the model's fields from real weather, adapting to whichever failure mode that model exhibits. We compare our learned atmospheric critic to existing metrics using various synthetic corruptions applied to ERA5 reanalysis data. Our method successfully identifies the corruptions and ranks their severity, while existing metrics fail on at lea
    
[^88]: 面向波束选择的语义CSI反馈：当基于稀疏导频的任务感知嵌入优于全带宽重建时

    Semantic CSI Feedback for Beam Selection: When Task-Aware Embeddings from Sparse Pilots Outperform Full-Bandwidth Reconstruction

    [https://arxiv.org/abs/2609.18368](https://arxiv.org/abs/2609.18368)

    该论文提出面向波束选择的语义CSI反馈方法，仅用43个稀疏导频生成的8维任务感知语义嵌入，其波束预测精度超越了利用全部512子载波信道进行重建的传统方法，证明波束相关信息本质上是低维的。

    

    在FDD大规模MIMO系统中，经典的CSI反馈传输的是信道的压缩重建，其优化目标是对原始信号的保真度，而与下游任务无关。我们提出一种语义通信的视角：用户设备（UE）不再重建信道，而是传输一种学习得到的语义嵌入，该嵌入针对基站的波束选择任务进行端到端优化。通过在两种输入域和三种观测场景下，将面向重建的反馈方法（CsiNet）与任务感知的语义反馈进行对比，我们证明：在角度-时延域中，仅使用43个NR CSI-RS导频、维度仅为d=8实数值的语义嵌入即可实现最高的波束预测精度，优于所有能够获取全部512个子载波信道的方法。关键洞察在于，与波束相关的信息本质上是低维的：语义编码器学会丢弃与重建无关的结构，仅保留紧凑的表示。

    arXiv:2609.18368v1 Announce Type: cross  Abstract: Classical CSI feedback in FDD massive MIMO transmits a compressed reconstruction of the channel, optimizing fidelity to the original signal regardless of the downstream task. We propose a semantic communication perspective: instead of reconstructing the channel, the UE transmits a learned \emph{semantic embedding} optimized end-to-end for beam selection at the gNB. Comparing reconstruction-oriented feedback (CsiNet) against task-aware semantic feedback across two input domains and three observation scenarios, we show that a semantic embedding of just $d=8$ real values from only 43 NR CSI-RS pilots in the angular-delay domain achieves the highest beam prediction accuracy, outperforming every method with access to the full 512-subcarrier channel. The key insight is that beam-relevant information is intrinsically low-dimensional: the semantic encoder learns to discard reconstruction-irrelevant structure and retain only a compact represent
    
[^89]: 坏天才：超越任务特定捷径的反事实引导测试框架演化

    Bad Genius: Counterfactual-Guided Harness Evolution Beyond Task-Specific Shortcuts

    [https://arxiv.org/abs/2609.18366](https://arxiv.org/abs/2609.18366)

    提出CHASE框架，通过挑战者搜索破坏性协议变换并利用有效性防火墙与确认集，检测并阻止自动测试框架优化利用基准级捷径作弊，实现可靠的智能体评估。

    

    可靠的智能体评估因自动测试框架优化而变得复杂，这类优化方法反复使用已发布的基准 $B_{\mathrm{rel}}$ 来引导一个提议者，该提议者围绕固定的目标智能体编辑提示词、记忆、检索、工具和控制代码。任务保留集虽然改变了语义任务，但基准协议保持不变，因此一个“坏天才”提议者可以生成一个作弊的测试框架，其在发布基准上的性能提升依赖于整个基准范围的捷径。我们提出了反事实测试框架搜索与演化，将测试框架演化建模为在保持有效性的基准反事实上的约束生成问题。在每次提议者更新后，一个挑战者会搜索能大幅摧毁性能提升的可执行协议变换。有效性防火墙检查任务语义是否得到保留，而确认集则决定反事实是否进入有限存档。我们形式化定义了一个精确的捷径中和基准 $B

    arXiv:2609.18366v1 Announce Type: new  Abstract: Reliable agent evaluation is complicated by automatic harness optimization, which repeatedly uses a released benchmark $B_{\mathrm{rel}}$ to guide a Proposer that edits prompts, memory, retrieval, tools, and control code around a fixed target agent. Task holdout varies semantic tasks but leaves the benchmark protocol fixed, so a "bad genius" Proposer can produce a cheating harness whose released-benchmark gain depends on a benchmark-wide shortcut. We introduce Counterfactual Harness Search and Evolution (CHASE), which casts harness evolution as constraint generation over validity-preserving benchmark counterfactuals. After each Proposer update, a Challenger searches for an executable protocol transformation with large gain destruction. A validity firewall checks that task semantics are preserved, while a confirmation set determines whether the counterfactual enters a finite archive. We formalize an exact shortcut-neutralized benchmark $B
    
[^90]: RecMorph：面向广义形态控制的拓扑引导空间递归方法

    RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control

    [https://arxiv.org/abs/2609.18359](https://arxiv.org/abs/2609.18359)

    RecMorph提出了一种拓扑引导的空间递归架构，通过深度优先遍历将运动学树转化为序列，并利用共享双向转换联合实现跨肢体通信与表示变换，在广义形态控制任务中取得了最佳性能。

    

    广义形态控制要求单个策略能够在具有不同物理角色的肢体之间变换信息、协调全身运动，并在身体规模增大时保持高效。现有的通信机制只能部分满足这些要求。我们提出了RecMorph，这是一种拓扑引导的空间递归架构，利用递归序列计算来同时执行跨肢体通信和表示变换。通过深度优先遍历，将运动学树转换为由形态结构派生的序列，共享的双向转换沿着该序列在动作解码之前逐步变换肢体信息。残差保留、RMS归一化以及依赖于输入的通道调制稳定了这种重复的空间变换，在固定的模型宽度和深度下实现了线性的token复杂度。在五个UNIMAL任务中，RecMorph取得了最强的平均最终训练性能。

    arXiv:2609.18359v1 Announce Type: cross  Abstract: Generalized morphology control requires a single policy to transform information across limbs with different physical roles, coordinate whole-body motion, and remain efficient as body size grows. Existing communication mechanisms address these requirements only partially. We introduce RecMorph, a topology-guided spatial recurrent architecture that uses recurrent sequence computation to jointly perform cross-limb communication and representation transformation. A depth-first traversal converts the kinematic tree into a morphology-derived sequence, along which shared bidirectional transitions progressively transform limb information before action decoding. Residual preservation, RMS normalization, and input-dependent channel modulation stabilize this repeated spatial transformation, yielding linear token complexity at fixed model width and depth. Across five UNIMAL tasks, RecMorph achieves the strongest mean final training performance am
    
[^91]: 面向不完美教师模型离线在线策略蒸馏的轨迹可学习性

    Trajectory Learnability for Offline On-Policy Distillation with Imperfect Teachers

    [https://arxiv.org/abs/2609.18321](https://arxiv.org/abs/2609.18321)

    本文提出利用教师成功问题作为廉价参考，通过观测学生模型在不同训练阶段对教师失败轨迹中各token似然的变化（带符号似然变化），来识别不完美教师监督下仍然可学习的内容，从而解决离线在线策略蒸馏中不完美监督持续存在的问题。

    

    离线在线策略蒸馏通过一次性收集学生轨迹和教师监督，并在整个优化过程中重复使用，从而获得效率优势。然而，同样的重复使用机制也使得不完美的监督持续存在。由于即使是强大的教师模型也可能失败，我们提出这样一个问题：从不完美的教师监督中，究竟还有哪些内容是可学习的？教师失败只是一个粗粒度的问题级别信号，并不意味着相关学生轨迹上的所有监督都是无益的。一种自然的替代方案是沿轨迹估计教师的可恢复性，但反复的续写生成会大大削弱离线蒸馏的效率优势。我们转而利用教师成功的问题来定义一个廉价的参考，以衡量学生可以学习的内容。我们在教师成功的问题上进行训练，并测量来自教师失败问题的轨迹中每个观测token的似然如何变化。我们将这些带符号的似然变化作为一种可操作的（摘要在此处截断）

    arXiv:2609.18321v1 Announce Type: cross  Abstract: Offline on-policy distillation gains efficiency by collecting student trajectories and teacher supervision once and reusing them throughout optimization. The same reuse makes imperfect supervision persistent. Since even strong teachers can fail, we ask \emph{what remains learnable from imperfect teacher supervision?} Teacher failure is only a coarse problem-level signal and does not imply that all supervision along the associated student trajectory is unhelpful. A natural alternative is to estimate teacher recoverability along the trajectory, but repeated continuations largely erase the efficiency advantage of offline distillation. We instead use teacher-successful problems to define a cheap reference for what the student can learn. We train on teacher-successful problems and measure how the likelihood of each observed token in trajectories from teacher-failed problems changes. We use these signed likelihood changes as an operational \
    
[^92]: 注意力分散作为大语言模型幻觉的诊断信号

    Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models

    [https://arxiv.org/abs/2609.18320](https://arxiv.org/abs/2609.18320)

    该论文提出一种无监督的注意力分散度量方法，通过监测大语言模型内部注意力机制的时间波动性来检测幻觉，摆脱了对输出校准的依赖，在数学推理基准上相比基于输出的基线方法AUC提升高达0.076。

    

    大语言模型（LLM）经常出现幻觉现象，这成为其在复杂推理任务中可靠性的主要障碍。虽然传统的检测方法依赖于基于输出的置信度指标，但这些logits经常被现代对齐技术错误校准。在本文中，我们研究了内部注意力机制的时间波动性，作为一种不依赖输出校准的幻觉替代诊断信号。通过引入一种无监督的注意力分散度量，我们证明认知不确定性会在中间层中留下可测量的痕迹，其中注意力熵的峰值与推理崩溃相关联。我们使用Qwen2.5模型家族（1.5B和3B参数）在数学推理基准（GSM8K和MATH-500）上评估了我们的方法，发现在所有测试条件下，相比基于输出的基线方法，AUC获得了高达+0.076的统计学显著提升。

    arXiv:2609.18320v1 Announce Type: new  Abstract: Large Language Models (LLMs) frequently exhibit hallucinations, presenting a major barrier to reliability in complex reasoning tasks. While traditional detection methods rely on output-based confidence metrics, these logits are often miscalibrated by modern alignment techniques. In this paper, we investigate the temporal volatility of internal attention mechanisms as an alternative diagnostic signal for hallucination that does not depend on output calibration. By introducing an unsupervised metric for attention dispersion, we show that epistemic uncertainty leaves a measurable trace within intermediate layers, where spikes in attention entropy are associated with reasoning breakdowns. We evaluate our approach on mathematical reasoning benchmarks (GSM8K and MATH-500) using the Qwen2.5 model family (1.5B and 3B parameters), finding statistically significant AUC improvements of up to +0.076 over output-based baselines across all tested cond
    
[^93]: 基于标签保持的聚合信号重组与预测一致性的多电器非侵入式负荷监测

    Multi-Appliance Non-Intrusive Load Monitoring via Label-Preserving Aggregate Recomposition and Prediction Consistency

    [https://arxiv.org/abs/2609.18315](https://arxiv.org/abs/2609.18315)

    该论文提出一种结合标签保持的聚合信号重组与预测一致性的多电器非侵入式负荷监测方法，通过仅替换总功率中的残余背景来构造新训练样本并施加跨窗口预测一致性约束，从而提升模型对未见家庭的泛化能力。

    

    非侵入式负荷监测（NILM）从总功率中估计各电器的功率序列，但在源家庭数据上训练的模型在未见过的家庭中通常会损失精度。总功率中还包含其他电器的负荷以及测量误差，因此预测可能依赖于与源家庭目标信号共同出现的残余背景。时间对齐的分表测量数据与总功率的加性分解揭示了一种逐窗口监督所未能利用的关系：可以通过仅替换总功率窗口中的残余背景，同时逐点保留所有被建模的目标电器功率序列，来重组该总功率窗口。我们将标签保持的聚合信号重组与预测一致性相结合。重组后的两个窗口均接受完整的功率和运行状态监督。对于每个电器，仅当两个功率预测均满足固定的可靠性准则时，才对二者之间的不一致性进行惩罚，并且仅对……（摘要原文在此处截断）

    arXiv:2609.18315v1 Announce Type: new  Abstract: Non-intrusive load monitoring (NILM) estimates appliance power sequences from aggregate power, but models trained on source households commonly lose accuracy in unseen households. Aggregate power also contains loads from other appliances and measurement error, so predictions may depend on the residual background that co-occurs with source-household targets. Time-aligned submetered measurements and the additive decomposition of aggregate power expose a relation unused by window-wise supervision: an aggregate window can be recomposed by replacing only its residual background while preserving all modeled target-appliance power sequences pointwise. We combine label-preserving aggregate recomposition with prediction consistency. Both windows receive complete power and operating-state supervision. For each appliance, disagreement between the two power predictions is penalized only when both satisfy a fixed reliability criterion and only to the
    
[^94]: 超越二次损失：Adam优化器的稳定性相图

    Beyond Quadratic Loss: The Stability Phase Diagram of Adam

    [https://arxiv.org/abs/2609.18314](https://arxiv.org/abs/2609.18314)

    该研究通过绘制Adam优化器在$(\beta_1,\beta_2)$参数平面上的稳定性相图，发现一条近似线性边界$1-\beta_2=C(1-\beta_1)$可用于区分训练中是否出现损失尖峰，并揭示超二次损失景观（如高置信交叉熵损失形成的“核心-墙壁”结构）是决定该边界形状的关键因素。

    

    损失尖峰是神经网络训练中反复出现的不稳定性现象，可能由多种机制引起。特别是对于Adam优化器，宏观损失尖峰已被认为与优化器动力学相关，但其两个动量时间尺度如何支配这些尖峰仍不清楚。我们通过在$(\beta_1,\beta_2)$平面上绘制训练动力学图谱来研究这种依赖关系。在多种模型-任务设置中，一条近似线性的边界$1-\beta_2=C(1-\beta_1)$将出现尖峰与不出现尖峰的动力学区域分隔开来，而一维二次损失则产生近似三次方斜率的边界。一维超二次损失$L(x)\propto|x|^n$则恢复了近线性标度关系，并将边界系数与有效损失指数$n$联系起来。我们进一步表明，高置信度的交叉熵损失会发展出一种“核心-墙壁”景观，由狭窄的二次核心和随后陡峭的墙壁组成，这在优化器步长的尺度上产生有效的超二次行为。

    arXiv:2609.18314v1 Announce Type: new  Abstract: Loss spikes are recurrent instabilities in neural-network training and can arise from multiple mechanisms. For Adam in particular, macroscopic loss spikes have been linked to optimizer dynamics, yet how its two momentum timescales govern them remains unclear. We investigate this dependence by mapping training dynamics across the $(\beta_1,\beta_2)$ plane. Across a range of model--task settings, an approximately linear boundary, $1-\beta_2=C(1-\beta_1)$, separates spiky from non-spiky dynamics, whereas a one-dimensional quadratic loss produces approximately cubic slope. A one-dimensional superquadratic loss $L(x)\propto|x|^n$ recovers the near-linear scaling and links the boundary coefficient to the effective loss exponent $n$. We further show that confident cross-entropy losses develop a core--wall landscape comprising a narrow quadratic core followed by a steep wall, which produces effective superquadratic behavior at the scale of an op
    
[^95]: 多智能体网络中的偏见放大：有偏见的智能体如何塑造观点与修辞

    Bias Amplification in Multi-Agent Network: How Biased Agents Shape Opinions and Rhetoric

    [https://arxiv.org/abs/2609.18306](https://arxiv.org/abs/2609.18306)

    该研究发现，在多智能体大语言模型系统中，即使只有少量持持久极端观点的有偏见智能体，也能通过文本交互显著改变无偏见智能体的观点，且 Llama 3.2 的观点偏移速度比经典观点动力学模型更快。

    

    大语言模型越来越多地被部署在涉及智能体交互的应用中，其输出在集体推理与决策过程中发挥作用。尽管针对大语言模型在这类多智能体系统中的运作机制已有大量研究，但此类系统中的偏见传播过程仍是一个亟待解决的难题。本工作研究了偏见观点如何在由大语言模型构成的环境中通过文本交互的形式进行传播：其中少数智能体持有持续不变的极端观点，而其余智能体则通过结构化的文本交互不断迭代更新自身的信念。研究结果表明，即使系统中仅存在很小比例的有偏见智能体，也会导致无偏见智能体的观点发生显著偏移。此外，在持偏见智能体比例相同的情况下，Llama 3.2 模型的观点偏移速度比经典的 Friedkin（原文截断，应为经典观点动力学模型）更快。

    arXiv:2609.18306v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in applications involving interaction between agents, where their output plays a role in collective reasoning and decision-making processes. Despite significant research into the functioning of LLMs in such multi-agent systems, the processes of bias propagation in such systems are still a challenge. This work studies how biased opinions are propagated in the form of textual interaction in an environment of LLMs, in which a minority of agents maintain persistent extreme opinions, while the remaining agents iteratively update their beliefs through structured textual interactions. The findings show that even the presence of a small percentage of biased agents in such a system leads to significant shifts in the opinions of non-biased agents. It suggests that for the same percentage of biased agents, the shifts occur more quickly for the Llama~3.2 model when compared to a classical Friedk
    
[^96]: 智能体应栖身何处？面向边缘-云连续体的智能体AI能耗-内存特性表征

    Where Should Agents Live? Energy-Memory Characterization of Agentic AI for the Edge-Cloud Continuum

    [https://arxiv.org/abs/2609.18283](https://arxiv.org/abs/2609.18283)

    该论文针对边缘-云连续体中的多智能体AI工作流提出能耗与内存特性表征分析，填补了现有AI生命周期指标忽视多智能体执行图的空白，帮助网络运营商确定智能体团队的最佳部署位置并量化分布式智能体通信的能耗代价。

    

    随着电信网络向自主化的5G-Advanced和6G运营演进，智能体人工智能工作流——即大语言模型执行多步推理、调用诊断工具、检索领域知识并在智能体团队间进行协调——正日益嵌入边缘-云连续体之中。尽管生物大脑能在约20瓦的极低代谢功率预算下完成复杂认知，但当代大语言模型的能耗和内存占用极高，这使得可持续的生命周期编排成为关键的运营优先事项。然而，现有的AI生命周期指标仅评估孤立的单一模型推理，或完全忽略多智能体执行图。因此，网络运营商缺乏基础模型来判断分布式智能体通信是否会带来显著的能量成本，以及智能体团队应物理部署在边缘-云连续体的哪一层。

    arXiv:2609.18283v1 Announce Type: new  Abstract: As telecommunication networks evolve toward autonomous 5G-Advanced and 6G operations, agentic artificial intelligence (AI) workflows, where large language models (LLMs) execute multi-step reasoning, invoke diagnostic tools, retrieve domain knowledge, and coordinate across agent teams, are increasingly embedded across the edge-cloud continuum. While the biological brain accomplishes complex cognition on an exceptionally modest metabolic power budget of approximately 20W contemporary LLMs are profoundly energy- and memory-intensive, making sustainable lifecycle orchestration a critical operational priority. However, existing AI lifecycle metrics evaluate only isolated, single-model inferences or overlook multi-agent execution graphs entirely. Consequently, network operators lack foundational models to determine whether distributed agent communication incurs meaningful energy costs and where across edge-cloud tiers agent teams should physic
    
[^97]: 一种基于生成对抗网络（GAN）的鲁棒DDoS攻击检测框架

    A GAN-Based Framework for Robust DDoS Attack Detection

    [https://arxiv.org/abs/2609.18281](https://arxiv.org/abs/2609.18281)

    该论文提出了一种基于WGAN-GP生成合成对抗流量的鲁棒DDoS攻击检测框架，通过将生成对抗建模与随机森林、深度神经网络集成和Transformer等机器学习模型相结合，有效提升了模型对抗针对性对抗攻击的防御能力。

    

    由于分布式拒绝服务（DDoS）攻击的存在，在线服务的可用性和一致性仍然容易受到威胁。这些攻击正通过采用更复杂的策略来规避传统网络安全系统，从而不断演变。尽管机器学习模型在检测DDoS流量方面非常有效，但针对性的对抗攻击会降低其分类准确率。本工作提出了一种将生成对抗建模与先进机器学习模型相结合的鲁棒检测框架。我们使用CICDDoS2019数据集训练了随机森林、深度神经网络集成和基于Transformer的模型，以建立该框架的基线性能。为了增强模型的防御能力，我们使用带梯度惩罚的Wasserstein生成对抗网络（WGAN-GP）生成了模拟潜在规避尝试和对抗流量的合成对抗流。随后，我们将生成的流量……

    arXiv:2609.18281v1 Announce Type: new  Abstract: The availability and consistency of online services remain vulnerable due to Distributed Denial of Service (DDoS) attacks. These attacks are evolving by adopting more complex strategies to evade traditional network security systems. Despite the effectiveness of machine learning models in detecting DDoS traffic, targeted adversarial attacks can degrade their classification accuracy. This work proposes a robust detection framework that integrates generative adversarial modelling with advanced machine learning models. We trained Random Forests, Deep Neural Ensembles, and Transformer-based models using the CICDDoS2019 dataset to establish the frameworks baseline performance. To enhance the models defensive capacity, we generated synthetic adversarial flows that simulate potential evasion attempts and adversarial traffic using a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). Then, we combined the generated traffic
    
[^98]: 网页浏览中的行为指纹识别与导航预测

    Behavioral Fingerprinting and Navigation Prediction in Web Browsing

    [https://arxiv.org/abs/2609.18273](https://arxiv.org/abs/2609.18273)

    该论文通过实证研究证明，即使短暂的浏览会话也足以高度唯一地识别用户身份，且结合图建模与大语言模型可以从长期交互结构中高度准确地预测用户的下一步导航行为。

    

    网页浏览通常显得转瞬即逝：用户访问几个网站，完成任务后便离开。然而，即使是短暂的浏览活动片段也可能包含丰富且结构化的行为信号。在本工作中，我们对两个互补的行为推断任务进行了比较实证研究：会话级用户识别和下一域名预测。这两个任务均源自同一清洗后的事件流，并在大规模匿名浏览记录上进行评估，会话化和数据划分根据每个任务的时间要求进行了相应调整。对于用户识别任务，我们评估了基于会话级行为特征和域名特征的经典模型与神经模型。对于下一域名预测任务，我们将基于图的建模与大语言模型（LLMs）相结合。实验结果表明，短暂的浏览会话具有高度的可识别性，而未来的导航行为也可以从长期交互结构中得到高度准确的预测。

    arXiv:2609.18273v1 Announce Type: new  Abstract: Web browsing often appears ephemeral: users visit a few websites, complete a task, and move on. However, even short fragments of browsing activity can contain rich and structured behavioral signals. In this work, we conduct a comparative empirical study of two complementary behavioral inference tasks: session-level user identification and next-domain prediction. Both tasks are derived from the same cleaned event stream and evaluated on large-scale anonymous browsing traces, with sessionization and splitting adapted to the temporal requirements of each task. For user identification, we evaluate classical and neural models operating on session-level behavioral and domain features. For next-domain prediction, we combine graph-based modeling with Large Language Models (LLMs). Experimental results show that short browsing sessions are highly identifiable, while future navigation actions are highly predictable from long-term interaction struct
    
[^99]: 以米为单位行动：学习度量交互以实现精确的机器人操作

    Acting in Meters: Learning Metric Interactions for Precise Robotic Manipulation

    [https://arxiv.org/abs/2609.18243](https://arxiv.org/abs/2609.18243)

    该论文提出了一个度量交互框架，通过交互中心标记（ICTs）显式建模物体级的末端执行器-物体相对位姿交互，并通过度量动作交互场（MAIF）学习场景几何条件下的动作修正，从而在物理笛卡尔空间中以共享度量尺度实现精确的机器人操作。

    

    视觉-语言-动作模型和世界-动作模型已经推动了语言条件下的机器人操作研究，但它们往往将动作、物体和场景几何之间的度量关系保持为隐式。人类的操作将任务相关物体的语义理解与空间反馈相结合，这种空间反馈引导手部相对于物体及其周围环境的运动。受此启发，我们提出了一个度量交互框架，在共享的度量尺度下，于物理笛卡尔空间中对物体级和场景级交互进行建模。在物体层面，交互中心标记（ICTs）显式表示末端执行器相对于被操作物体的位姿轨迹，并与动作进行联合去噪，提供具有物理基础的交互监督。在场景层面，度量动作交互场（MAIF）利用动作和ICT查询来关注度量场景点云特征，并学习几何条件下的动作修正。

    arXiv:2609.18243v1 Announce Type: cross  Abstract: Vision-Language-Action models and World-Action Models have advanced language-conditioned robotic manipulation, yet often leave metric relations among actions, objects, and scene geometry implicit. Human manipulation combines semantic understanding of task-relevant objects with spatial feedback that guides hand motion relative to objects and their surroundings. Inspired by this, we introduce a metric interaction framework that models object-level and scene-level interactions in physical Cartesian space at a shared metric scale. At the object level, Interaction-Centric Tokens (ICTs) explicitly represent end-effector pose trajectories relative to manipulated objects and are jointly denoised with actions, providing physically grounded interaction supervision. At the scene level, the Metric Action Interaction Field (MAIF) uses action and ICT queries to attend to metric scene point-cloud features and learns geometry-conditioned action correc
    
[^100]: F-DACE：面向弃权安全的对话式零售决策支持的模糊分歧感知因果证据融合

    F-DACE: Fuzzy Disagreement-Aware Causal Evidence Fusion for Abstention-Safe Conversational Retail Decision Support

    [https://arxiv.org/abs/2609.18238](https://arxiv.org/abs/2609.18238)

    F-DACE框架通过模糊隶属度融合多个因果估计器的证据一致性，并在估计量不匹配、诊断失败或证据冲突时主动弃权，从而在对话式零售决策支持中显著降低错误建议率。

    

    观察性决策支持系统即使在合理的估计器彼此不一致的情况下，也常常只将单一因果估计作为建议呈现。所提出系统的核心引擎是因果机器学习：通过后门调整识别条件平均处理效应估计量，使用EconML的DML因果森林和DoWhy线性回归进行估计，以双向固定效应进行检验，并通过约束优化将结果转化为候选决策杠杆。F-DACE是该引擎之上的决策层。它将估计精度、倾向得分重叠度、安慰剂反驳稳定性、置信区间重叠度以及方向一致性表示为模糊隶属度。在估计量不匹配、诊断检验失败、具有信息量的符号冲突或证据薄弱的情况下，硬否决机制会强制系统弃权。在涵盖六种识别条件的180次面板模拟中，F-DACE在67.2%的运行中做出了决策，并将错误建议限制在17.2%；而相应方法的比率分别为33.3%（此处摘要内容截断）

    arXiv:2609.18238v1 Announce Type: new  Abstract: Observational decision-support systems often expose one causal estimate as a recommendation even when plausible estimators disagree. The inherent engine of the proposed system is causal machine learning: a conditional-average-treatment-effect estimand identified by backdoor adjustment, estimated by an EconML DML causal forest and DoWhy linear regression, checked by two-way fixed effects, and converted into candidate levers by constrained optimisation. F-DACE is the decision layer on that engine. It represents precision, propensity overlap, placebo-refutation stability, interval overlap, and directional agreement as fuzzy memberships. Hard vetoes force abstention after estimand mismatch, failed diagnostics, informative sign conflict, or weak evidence. In 180 panel simulations spanning six identification conditions, F-DACE made a decision in 67.2% of runs and limited false recommendations to 17.2%; the corresponding rates were 33.3% for th
    
[^101]: 总账数据中的异常检测：混合方法的研究成果

    Anomaly Detection in General Ledger Data: Results from a Hybrid Approach

    [https://arxiv.org/abs/2609.18228](https://arxiv.org/abs/2609.18228)

    该研究提出将日记账测试（JETs）与机器学习方法相结合的混合模型，以减少误报、提高总账数据异常检测的性能和有效性，从而提升审计效率。

    

    日记账测试（JETs）是年度审计中必不可少的环节，用于评估高风险审计领域和潜在的重大错报。然而，由于JETs是基于领域知识设计的、用于检测已知模式的方法，其产生的结果列表往往非常庞大，需要审计师投入大量额外的精力。为确保审计的经济效率，必须减少JET结果列表中的误报数量。尤其是机器学习（ML）方法，为改进该领域的异常检测提供了一种有前景的途径。在这篇研究进行中的论文中，我们探讨了以混合方式将JETs与机器学习方法相结合的不同方案。我们提出了专门的模型，以提高异常检测结果的检测性能和有效性，从而提升审计效率。实验基于由不同正常和异常日记账分录组成的合成数据进行。

    arXiv:2609.18228v1 Announce Type: new  Abstract: Journal Entry Tests (JETs) are a mandatory part of annual audits to evaluate and assess both highrisk audit areas and potential material misstatements. However, as JETs are designed to detect known patterns based on domain knowledge, the resulting lists are often very large and require substantial additional effort from the auditor. To ensure the economic efficiency of the audit, the number of false positives in JET result lists must be reduced. Especially machine learning (ML) methods represent a promising approach to improve anomaly detection in this field. In this research in progress paper, we investigate different approaches on how to combine JETs with ML-methods in a hybrid manner. We present specialized models to increase the detection performance and validity of anomaly detection results to improve audit efficiency. The experiments are based on synthetic data consisting of different normal and anomalous journal entries.
    
[^102]: APGEM：面向真实世界CVRP案例研究的量子强化学习自适应策略引导误差缓解

    APGEM: Adaptive Policy-Guided Error Mitigation for Quantum Reinforcement Learning on a Real-World CVRP Case Study

    [https://arxiv.org/abs/2609.18219](https://arxiv.org/abs/2609.18219)

    提出了APGEM自适应控制器，能够在量子强化学习过程中根据学习情境在线动态选择最合适的误差缓解技术（ZNE、PEC、CDR、REM），有效应对NISQ硬件噪声，并在基于德里真实地标的城市物流车辆路径问题上得到验证。

    

    量子强化学习（QRL）将策略表示为变分量子电路（VQC），这使其在诸如带容量约束的车辆路径问题（CVRP）等组合优化问题中极具吸引力。然而，在噪声中等规模量子（NISQ）硬件上，退相干会降低保真度并破坏学习的稳定性，而传统的误差缓解方法是以静态方式应用的，未考虑学习所处的情境。我们提出了自适应策略引导误差缓解（APGEM），这是一种在线控制器，可在零噪声外推（ZNE）、概率误差消除（PEC）、Clifford数据回归（CDR）和读出误差缓解（REM）之间进行选择，其由同时感知保真度、熵和成本的效用函数以及基于时间差分Q分数的epsilon-贪婪规则驱动。我们在一个现实的城市物流测试平台上进行评估，该平台是基于德里真实地标的CVRP问题，采用测地线计算节点间成本，并在五种噪声族上进行了测试。

    arXiv:2609.18219v1 Announce Type: cross  Abstract: Quantum Reinforcement Learning (QRL) represents policies as variational quantum circuits (VQCs), making it attractive for combinatorial optimization such as the Capacitated Vehicle Routing Problem (CVRP). On noisy intermediate-scale quantum (NISQ) hardware, however, decoherence degrades fidelity and destabilizes learning, and conventional error mitigation is applied statically without regard to the learning context. We introduce Adaptive Policy-Guided Error Mitigation (APGEM), a controller that selects among Zero-Noise Extrapolation (ZNE), Probabilistic Error Cancellation (PEC), Clifford Data Regression (CDR), and Readout Error Mitigation (REM) online, driven by a fidelity, entropy, and cost aware utility function and an epsilon-greedy rule over temporal-difference Q-scores. We evaluate on a realistic urban-logistics testbed, a Delhi-based CVRP over real landmarks with geodesic inter-node costs, exercised across five noise families and
    
[^103]: 一种用于乳腺癌钼靶X光图像检测与分类的轻量级CNN集成紧凑卷积Transformer模型：多尺度特征学习与计算复杂度降低

    A Lightweight CNN Integrated Compact Convolutional Transformer for Multi-Scale Feature Learning and reducing computational complexity for breast cancer mammography image detection and classification

    [https://arxiv.org/abs/2609.18212](https://arxiv.org/abs/2609.18212)

    该论文提出一种轻量级CNN与紧凑卷积Transformer相融合的模型，仅用约25万参数就在三个乳腺癌钼靶数据集上达到99%-100%的准确率，并结合可解释AI增强临床可信度。

    

    多年来，卷积神经网络（CNN）在利用医学图像进行癌症检测和分类方面展现了强大的能力。然而，基于CNN的模型往往难以捕捉长程上下文依赖关系。为解决这一问题，本方法在CNN层之后集成紧凑卷积Transformer（CCT）架构，利用CCT分词器将CNN提取的特征重塑为紧凑的补丁标记，并添加位置嵌入以保留空间结构信息。通过5折交叉验证，该模型在3组乳腺癌钼靶X光数据集上进行了测试。该模型仅有250,435个参数，却在3个数据集上实现了99%-100%的准确率，展现出强大的泛化能力。此外，模型集成了可解释人工智能（XAI）来解释乳腺癌分类过程，以增强临床信任度。结果表明，所提出的框架适用于计算机辅助诊断系统。

    arXiv:2609.18212v1 Announce Type: cross  Abstract: Over the years, Convolutional Neural Networks (CNNs) have demonstrated strong capability in cancer detection and classification using medical images. However, CNN-based models often struggle to capture long-range contextual dependencies. In such scenarios, integrating Compact Convolutional Transformer (CCT) architectures after the CCT layer allows CNN-extracted features to reshape into compact patch tokens using a CCT tokenizer, followed by the addition of positional embeddings to preserve spatial structure. Using 5-fold cross-validation, the model was tested on 3 sets of breast cancer mammography. With only 250,435 parameters, the model achieved 99%-100% accuracy across 3 datasets, indicating robust generalization. Explainable AI (XAI) was integrated into the model to explain the breast cancer classification process to enhance clinical trust. The results indicate that the proposed framework is suitable for computer-aided diagnosis sys
    
[^104]: 面向实时视觉-语言-动作策略的强化学习

    Reinforcement Learning for Real-Time Vision-Language-Action Policies

    [https://arxiv.org/abs/2609.18207](https://arxiv.org/abs/2609.18207)

    该论文提出在EXPO-FT框架上通过强化学习微调使VLA策略满足实时控制要求，通过解耦慢速动作生成与快速动作执行，解决了模型推理延迟导致的观测过时和分布偏移问题，从而提升真实世界机器人操控的可靠性。

    

    在大型预训练视觉-语言-动作（VLA）模型上进行强化学习微调，为实现高可靠性的机器人部署带来了希望。然而，由于其规模庞大，现代VLA模型的推理延迟较高，导致用于选择动作的观测值在执行时往往已经过时，由此产生的分布偏移会显著降低系统的可靠性和性能。先前的工作探索了异步策略执行以减轻延迟的影响，但这些方法大多基于模仿学习构建，缺乏超越训练分布以实现更高可靠性的机制。我们通过使强化学习微调满足动态真实世界操控任务的实时控制要求来弥合这一差距。我们的方法建立在EXPO-FT（一个用于样本高效、可靠VLA强化学习微调的框架）之上，将缓慢而富有表达力的动作生成与快速的动作执行解耦……

    arXiv:2609.18207v1 Announce Type: cross  Abstract: Reinforcement learning fine-tuning on top of large, pretrained Vision-Language-Action (VLA) models offers promise for highly reliable robot deployment. However, because of their scale, modern VLA models suffer from high inference latency, so the observation used to select an action is often stale by execution time, creating a distribution shift that can substantially degrade reliability and performance. Prior work has explored asynchronous policy execution to reduce the effect of latency, but these methods are mostly built on imitation learning and offer no mechanism for moving beyond the training distribution toward higher reliability. We close this gap by enabling RL fine-tuning that meets the real-time control requirements of dynamic real-world manipulation. Our approach builds on EXPO-FT, a framework for sample-efficient, reliable VLA fine-tuning with reinforcement learning, and decouples slow, expressive action generation from fas
    
[^105]: Behavior2Value：面向电子商务行为中消费者价值测量的LLM基准测试与能力增强

    Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors

    [https://arxiv.org/abs/2609.18203](https://arxiv.org/abs/2609.18203)

    该论文提出了行为到价值（B2V）任务，构建了首个电子商务消费价值分类体系（ECVT）和基于真实淘宝行为日志的B2V-Bench基准数据集，实现了从电子商务行为轨迹中识别和测量消费者价值观，并据此赋能大语言模型。

    

    人类价值观是塑造人类行为的深层动机取向。在电子商务领域，价值观揭示了用户购买决策背后稳定的驱动因素。与短期兴趣相比，消费者价值观能更好地解释用户在购买前如何评价产品。然而，消费者价值观通常隐含在复杂且碎片化的行为轨迹中，使得从电子商务行为中进行价值测量在很大程度上仍是一个未被充分探索的领域。为此，我们提出了行为到价值（Behavior-to-Value，B2V）任务，旨在从电子商务行为轨迹中识别消费者价值观。围绕这一任务，我们首先构建了电子商务消费价值分类体系（ECVT），并基于匿名的淘宝行为日志引入了B2V-Bench——首个B2V数据集与基准。B2V-Bench由真实世界的购买决策情节组成，涵盖25种购买行为类型，以及每个情节中体现的相应消费者价值取向。

    arXiv:2609.18203v1 Announce Type: new  Abstract: Human values are deep motivational orientations that shape human behaviors. In e-commerce, they reveal the stable drivers behind users' purchase decisions. Compared with short-term interests, consumer values better explain how users evaluate products before purchase. However, consumer values are often implicit in complex and fragmented behavioral trajectories, leaving value measurement from e-commerce behaviors largely underexplored. To this end, we propose the Behavior-to-Value (B2V) task, which aims to identify consumer values from e-commerce behavioral trajectories. Centered on this task, we first construct the E-commerce Consumption Value Taxonomy (ECVT) and introduce B2V-Bench, the first B2V dataset and benchmark, based on anonymized Taobao behavioral logs. B2V-Bench consists of real-world purchase decision episodes, covering 25 types of purchase behaviors, along with corresponding consumer value orientations manifested in each epis
    
[^106]: 神经表征中的变换定律：结构、可实现性与构造

    Transformation Laws in Neural Representations: Structure, Realisability, and Construction

    [https://arxiv.org/abs/2609.18190](https://arxiv.org/abs/2609.18190)

    该论文建立了神经表征中参考变换的实现理论，刻画了变换何时能通过编码器传递，证明实现缺陷由变换对被丢弃信息的需求决定，并给出线性实现存在的精确条件——即被保留的调和块在变换作用下保持不变。

    

    神经表征如何保持输入变化的结构，将表征分析与内部干预联系起来。我们通过对神经特征施加参考变换的相容作用来研究可操作的表征内容。我们刻画了变换何时能够经由编码器下降传递，并给出了一个线性设定，在该设定中缺陷由变换对被丢弃信息的需求所决定，且以表征所诱导的度量来衡量。在整流器上，未能实现一个变换有两个可区分的来源——源区域已经使其不可恢复的部分，以及用一个算子同时满足变换所访问的每个区域的代价——而对于一个被度量的调和载波，同样的问题具有封闭形式的答案：当被保留的调和块在该作用下保持不变时，线性实现恰好存在。以颜色作为深入实例，我们发现……

    arXiv:2609.18190v1 Announce Type: new  Abstract: How neural representations preserve the structure of input changes connects representation analysis with internal intervention. We study operable representational content through compatible actions of reference transformations on neural features. We characterise when a transformation descends through an encoder, and give a linear setting in which the defect is governed by the transformation's demand for discarded information, measured in the metric the representation induces. On a rectifier the failure to realise a transformation has two distinguishable sources --- what the source region has already made unrecoverable, and what it costs to satisfy every region the transformation visits with one operator --- and for a \textit{measured} harmonic carrier the same question has a closed answer: a linear realisation exists exactly when the retained harmonic blocks are invariant under the action. Using colour as the in-depth instance, we find t
    
[^107]: MoRE：复用专家混合模型

    MoRE: Mixture of Reused Experts

    [https://arxiv.org/abs/2609.18176](https://arxiv.org/abs/2609.18176)

    MoRE通过在相邻层组之间共享专家池并引入可学习的深度嵌入对每层输入进行条件化，在不增加参数的情况下扩展路由组合多样性，实现了比标准MoE和权重共享方法更低的困惑度和更强的下游性能。

    

    混合专家架构将模型容量与计算成本解耦，但随着专家数量增加，参数量线性增长会导致高昂的内存占用。循环Transformer通过复用层权重实现了参数效率，但通常缺乏进行竞争性语言建模所需的容量。我们提出复用专家混合，这是一种在相邻层组之间共享专家池的混合架构。每一层保留自己的路由器，但从更大的共享池中进行选择，从而在不增加额外参数的情况下扩展了路由组合的多样性。为了使共享专家能够区分不同的层，我们引入了轻量级的可学习深度嵌入，在路由之前对每层的输入进行条件化处理。在三个模型规模（114M至1.15B参数）上的实验表明，MoRE始终实现了比标准MoE和最先进的权重共享方法更低的困惑度和更强的下游性能。

    arXiv:2609.18176v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) architectures decouple model capacity from computational cost, yet incur high memory footprints as parameters grow linearly with the number of experts. Recurrent Transformers achieve parameter efficiency by reusing layer weights, but typically lack the capacity for competitive language modeling. We propose Mixture of Reused Experts (MoRE), a hybrid that shares expert pools across groups of adjacent layers. Each layer retains its own router but selects from a larger shared pool, expanding the diversity of routing combinations without additional parameters. To enable shared experts to distinguish between layers, we introduce lightweight learnable depth embeddings that condition each layer's input before routing. Experiments across three model scales (114M-1.15B parameters) show that MoRE consistently achieves lower perplexity and stronger downstream performance than standard MoEs and state-of-the-art weight-shari
    
[^108]: 超越直接感知：在车辆跟踪中利用第三方传感器的间接观测

    Beyond Direct Sensing: Harnessing Indirect Observations from Third-Party Sensors in Vehicle Tracking

    [https://arxiv.org/abs/2609.18173](https://arxiv.org/abs/2609.18173)

    提出GrayTrack方法，利用道路约束的粒子滤波器将第三方传感器的微弱匿名间接观测与稀疏直接观测相融合，以填补车辆跟踪中的观测空白。

    

    车辆跟踪是城市交通、公共安全、安防和国防等应用领域的基础。传统的跟踪方法依赖于直接访问能够提供强观测信息（如车辆身份和位置）的传感器。然而在实践中，由于所有权、隐私、成本和运行限制等因素，可直接访问的传感器可能受限，导致观测稀疏和跟踪间隔过长。与此同时，环境中可能存在许多第三方感知资产，但在原始数据层面无法访问，因而无法直接集成到跟踪系统中。在这项工作中，我们研究了具有不确定时空线索的微弱间接观测是否可以补充稀疏的直接感知以实现车辆跟踪。具体而言，我们提出了GrayTrack，它使用道路约束的粒子滤波器将微弱的匿名事件与稀疏的直接观测相融合。我们构建……

    arXiv:2609.18173v1 Announce Type: cross  Abstract: Vehicle tracking is fundamental to applications ranging from urban mobility and public safety to security and defense. Conventional tracking relies on direct access to sensors that provide strong observations such as vehicle identity and location. In practice, however, factors such as ownership, privacy, cost, and operational constraints may limit directly accessible sensors, leaving sparse observations and long tracking gaps. Meanwhile, many additional third-party sensing assets may be present across the environment but remain inaccessible at the raw-data level, preventing their direct integration into the tracking system. In this work, we investigate whether weak, indirect observations with uncertain spatial and temporal cues can complement sparse direct sensing for vehicle tracking. Specifically, we propose GrayTrack, which fuses weak anonymous events with sparse direct observations using a road-constrained particle filter. We build
    
[^109]: 基于模型的强化学习中动力学变化下的经验回放保留特性刻画

    Characterizing Replay Retention Under Dynamics Shift in Model-Based Reinforcement Learning

    [https://arxiv.org/abs/2609.18167](https://arxiv.org/abs/2609.18167)

    该论文提出用变化幅度和年龄-陈旧度AUC两个量化指标，来刻画基于模型的持续强化学习中动力学变化后应保留还是遗忘旧经验回放数据的权衡问题。

    

    适应机器人动力学的变化需要从新数据中学习，同时不丢弃可能仍然有用的经验。在持续基于模型的强化学习（RL）中，动力学变化之前收集的经验回放可能会减慢适应速度，而将其移除则会不必要地减少可用的训练数据，并且如果早期动力学再次出现，代价可能尤其高昂。我们研究了何时近期转移数据优于完整的回放历史。两个量刻画了这种权衡：变化幅度和年龄-陈旧度曲线下面积（AUC），后者衡量转移数据年龄区分陈旧数据与新鲜数据的程度。在发生大的永久性变化后遗忘陈旧数据有帮助，但当动力学再次出现且较旧的数据重新变得有用时则会造成损害。因此，选择回放策略取决于预测旧数据何时会有帮助或有害。我们在两种运动形态、两种基于模型的RL算法和Real-World RL基准上测试了这些效应。

    arXiv:2609.18167v1 Announce Type: cross  Abstract: Adapting to changes in robot dynamics requires learning from new data without discarding experience that may still be useful. In continual model-based reinforcement learning (RL), replay collected before a dynamics change can slow adaptation, while removing it unnecessarily reduces available training data and can be especially costly if earlier dynamics return. We study when recent transitions are preferable to the full replay history. Two quantities characterize this trade-off: change magnitude and age-staleness area under the curve (AUC), measuring how well transition age separates stale from fresh data. Forgetting stale data helps after large permanent shifts but hurts when dynamics recur and older data becomes useful again. Choosing a replay strategy therefore depends on predicting when older data will help or hurt. We test these effects across two locomotion morphologies, two model-based RL algorithms, and Real-World RL benchmark 
    
[^110]: LIGE-GR：大模型时代从排序到生成式推荐的平滑跃迁

    LIGE-GR: A Smooth Leap from Ranking to Generative Recommendation in the LLM Era

    [https://arxiv.org/abs/2609.18148](https://arxiv.org/abs/2609.18148)

    该论文提出LIGE-GR框架，通过列表级生成与评估的方法，解决了将LLM范式中的序列级生成优化融入推荐系统以及避免整体替换成熟工业系统的两大挑战，实现了从传统排序推荐到生成式推荐的平滑跃迁。

    

    arXiv:2609.18148v1 公告类型：新论文 摘要：大语言模型（LLMs）的卓越成功为下一代推荐系统提供了重要启示。从结构上看，推荐与语言生成存在相似之处：两者都旨在生成一个能够优化用户体验的有序序列。然而，如何将LLM范式的精髓精准地融入成熟的工业推荐系统，仍然是一个开放性问题。这里存在两个挑战：首先，目前尚不清楚如何将LLM范式中的序列级生成与优化引入推荐任务；其次，现实中的推荐系统是成熟的系统，它们多年来围绕特定产品、业务约束、服务基础设施和组织架构进行了深度定制化迭代，整体替换这类系统在技术上往往风险很高，在组织层面也具有破坏性。在本文中，我们提出了LIGE-GR，一种列表级的生成与评估方法（摘要内容在此处截断）。

    arXiv:2609.18148v1 Announce Type: new  Abstract: The remarkable success of large language models (LLMs) has provided important inspiration for the next generation of recommender systems. Structurally, recommendation and language generation share a similarity: both aim to produce an ordered sequence that optimizes the user's experience. However, how to precisely absorb the essence of the LLM paradigm into mature industrial recommender systems remains an open problem.   There are two challenges. First, it is unclear how to incorporate sequence-level generation and optimization from the LLM paradigm into recommendation. Second, real-world recommender systems are mature systems that have been iteratively customized for years around specific products, business constraints, serving infrastructure, and organizational ownership. Replacing such systems wholesale is often technically risky and organizationally disruptive.   In this paper, we propose LIGE-GR, a listwise generation and evaluation 
    
[^111]: 无需搜索即可触达每个位置：超立方体上的旋转稀疏连线作为注意力机制的替代方案

    Reaching Every Position Without Searching: Rotating Sparse Wiring on the Hypercube as a Substitute for Attention

    [https://arxiv.org/abs/2609.18145](https://arxiv.org/abs/2609.18145)

    该论文提出将序列位置视为超立方体顶点、采用逐层旋转的稀疏连线结构，仅用log₂n层和每层2n条链接即可实现所有位置间的信息全连通，可作为注意力机制的高效替代方案。

    

    注意力机制在每一层、对每个输入都要付出搜索连接对象的代价。我们探究采用固定、稀疏且逐层简单旋转的连线结构能走多远。将序列的n个位置视为log₂n维超立方体的顶点，并在第ℓ层将每个位置连接到其沿维度ℓ mod log₂n方向上的邻居，信息在log₂n层内即可从任意位置到达任意其他位置，每层仅需2n条链接而非n²条。在一个除非所有位置都被触达否则无法求解的合成任务上，这种旋转方式以1/32的链接数量达到与全连接相当的效果，而在各层间保持固定不变的相同稀疏模式则失败了；关键在于每个维度都被触及，而非触及的顺序。在一个公开语料库（enwik8的前1200万字符）的字符级语言建模任务上，一种在十六个稀疏层中保留两个注意力层的混合架构达到了0.0……

    arXiv:2609.18145v1 Announce Type: new  Abstract: Attention pays, at every layer and for every input, the cost of searching for whom to connect. We ask how far one can get with wiring that is fixed, sparse, and simply rotated from layer to layer. Treating the $n$ positions of a sequence as the vertices of a $\log_2 n$-dimensional hypercube and connecting each position, at layer $\ell$, to its neighbour along dimension $\ell \bmod \log_2 n$, information from every position reaches every other in $\log_2 n$ layers with $2n$ links per layer instead of $n^2$. On a synthetic task that is unsolvable unless all positions are reached, this rotation matches all-to-all wiring at $1/32$ of the links, while the same sparse pattern held fixed across layers fails; what matters is that every dimension is touched, not the order. On character-level language modelling of a public corpus (the first $12$M characters of enwik8), a hybrid that keeps two attention layers among sixteen sparse ones reaches $0.0
    
[^112]: 重新思考我们如何评估健康人工智能中的方法学进展

    Rethinking How We Evaluate Methodological Progress in Health AI

    [https://arxiv.org/abs/2609.18134](https://arxiv.org/abs/2609.18134)

    该研究通过在统一评估框架内重新实现12个健康AI算法并在MIMIC-IV和NWICU两个临床数据集上进行评估，实证探讨了EHR AI评估中可重复性与临床任务定义的障碍，以及算法相对比较在不同任务族和数据集间的可迁移性。

    

    arXiv:2609.18134v1 公告类型：交叉 摘要：电子健康记录（EHR）人工智能（AI）的方法学进步取决于我们判断哪些算法更有效以及在何种条件下更有效的能力。然而，这种进步被认为受到可重复性困难以及难以定义具有临床意义的评估任务的阻碍。我们通过在共享评估框架内重新实现12个历史及近期算法，并在两个临床数据集MIMIC-IV和NWICU上对它们进行评估，从实证角度研究了这些障碍。我们比较了两个互补的任务族：专家撰写的具有临床意义的任务，以及从随机采样的事件代码和预测时间范围定义的生成任务。我们探讨了相对算法比较是否能在不同任务族和数据集之间迁移、剩余的任务异质性是否包含有用的方法学结构，以及受控比较能揭示过去十年的哪些进展。

    arXiv:2609.18134v1 Announce Type: cross  Abstract: Methodological progress in artificial intelligence (AI) for electronic health records (EHRs) depends on our ability to determine which algorithms work better, and under which conditions. However, such progress is thought to be hindered by difficulties in reproducibility and in defining clinically meaningful evaluation tasks. We empirically study these barriers by re-implementing 12 historical and recent algorithms within a shared evaluation framework and evaluating them on two clinical datasets, MIMIC-IV and NWICU. We compare two complementary task families: expert-authored clinically meaningful tasks and generated tasks defined from randomly sampled event codes and prediction horizons. We ask whether relative algorithms comparisons transfer across task families and datasets, whether residual task heterogeneity contains useful methodological structure, and what a controlled comparison reveals about progress over the last decade. We fin
    
[^113]: Colla-Q：通过极小极大精度平衡实现MoE量化中的专家协作

    Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing

    [https://arxiv.org/abs/2609.18131](https://arxiv.org/abs/2609.18131)

    提出基于激活熵的比特分配框架Colla-Q，通过极小极大精度平衡策略均衡MoE量化中各专家的性能，从而提升整体模型表现并降低对校准数据的依赖。

    

    在本文中，我们提出了一种基于激活熵的混合专家模型（MoE）量化方法。尽管量化能够降低内存和计算成本，但它可能会显著损害模型性能。尤其在量化的MoE模型中，性能下降尤为突出，因为各个专家模型的参数数量较少，对低比特表示较为敏感。考虑到MoE作为一个集成模型运行，依赖被路由到的专家进行协同贡献，某个特定专家因量化导致的显著性能下降可能会损害模型的整体性能。因此，我们提出了Colla-Q，这是一个比特分配框架，通过基于激活熵的比特宽度分配算法来保持各专家之间性能的均衡。这种方法促使每个专家在量化模型中协同运作，从而：1）提升MoE的整体性能；2）降低对校准数据的依赖。

    arXiv:2609.18131v1 Announce Type: cross  Abstract: In this paper, we present a Mixture-of-Experts (MoE) quantization method based on activation entropy. Although quantization reduces memory and computational costs, it can substantially degrade performance. In particular, performance decline is pronounced in quantized MoE models, where individual experts have a small number of parameters that are sensitive to low-bit representation. Considering that MoE operates as an ensemble model with collaborative contributions from routed experts, a significant performance decline of a particular expert due to quantization can harm model performance. Therefore, we propose Colla-Q, a bit-allocation framework to maintain balanced performance across experts through an activation-entropy-based bit-width allocation algorithm. This approach encourages each expert to operate collaboratively in the quantized model, thereby 1) improving the overall MoE performance and 2) reducing the dependence on the calib
    
[^114]: 基准测试表格基础模型作为昂贵进化优化中的代理模型

    Benchmarking Tabular Foundation Models as Surrogates in Expensive Evolutionary Optimization

    [https://arxiv.org/abs/2609.18130](https://arxiv.org/abs/2609.18130)

    本文通过涵盖离线与在线设置、多种优化场景的大规模实验与理论分析，系统性地评估了表格先验数据拟合网络作为昂贵进化优化中代理模型的有效性。

    

    代理辅助进化算法（SAEAs）是解决昂贵优化问题（EOPs）的有效方法，其中代理模型替代大部分昂贵的评估过程，并对最终优化结果起着关键性影响。近年来，表格基础模型发展迅速，其中表格先验数据拟合网络因其强大的预测能力已被用作EOPs的代理模型，并展现出有前景的性能。基于其作为SAEAs中代理模型的潜力，本工作开展了一项综合性研究，结合大量实验与深入的理论分析来探究TabPFN的有效性。具体而言，我们在离线和在线两种SAEA设置下进行实验，涵盖多种问题场景，包括单目标、多目标、约束、组合、混合变量以及工程优化问题。

    arXiv:2609.18130v1 Announce Type: cross  Abstract: Surrogate-assisted evolutionary algorithms (SAEAs) are effective methods for solving expensive optimization problems (EOPs), where surrogate models replace most expensive evaluations and critically influence the final optimization results. In recent years, tabular foundation models have advanced rapidly, and the Tabular Prior-data Fitted Network (TabPFN) has been adopted as a surrogate model for EOPs due to its strong predictive capability, demonstrating promising performance. Motivated by its potential as a surrogate model in SAEAs, this work conducts a comprehensive study that combines extensive experiments with in-depth theoretical analysis to investigate the effectiveness of TabPFN. Specifically, we perform experiments across both offline and online SAEA settings, covering diverse problem scenarios such as single-objective, multi-objective, constrained, combinatorial, mixed-variable, and engineering optimization problems. In additi
    
[^115]: MCLC-NET：面向叶片计数的多模态持续学习

    MCLC-NET: Multimodal Continual Learning for Leaf Counting

    [https://arxiv.org/abs/2609.18129](https://arxiv.org/abs/2609.18129)

    该论文提出MCLC-NET，一个融合深度和热成像等多模态信息、采用记忆缓冲策略进行顺序学习的叶片计数持续学习框架，并发布了真实世界多模态叶片计数数据集MMLC。

    

    叶片计数是植物表型分析中的一项重要任务，用于监测植物生长和估算作物产量。现有的大多数方法依赖RGB图像，但其性能常常受到遮挡、光照变化以及其他现实世界挑战的影响。额外的模态，如深度图像和热成像图像，可以提供有用的互补信息。然而，多模态叶片计数仍然是一个探索不足的领域。此外，许多现有方法假设所有训练数据可以同时获得，这在现实农业环境中是不切实际的，因为实际中数据是从多个来源随时间推移收集的。为了应对这些挑战，我们提出了MCLC-NET，一个用于叶片计数的多模态持续学习框架。该框架采用基于记忆的策略顺序学习任务，通过记忆缓冲区保留来自先前任务的重要样本。我们还引入了MMLC，一个面向领域的真实世界多模态叶片计数数据集。

    arXiv:2609.18129v1 Announce Type: cross  Abstract: Leaf counting is an important task in plant phenotyping for monitoring plant growth and estimating crop yield. Most existing methods rely on RGB images, but their performance is often affected by occlusion, lighting variations, and other real-world challenges. Additional modalities, such as depth and thermal images, can provide useful complementary information. However, multimodal leaf counting remains underexplored. Also, many existing methods assume that all training data are available simultaneously, which is impractical in real agricultural settings, where data is collected over time from multiple sources. To address these challenges, we propose MCLC-NET, a multimodal continual learning framework for leaf counting. It learns tasks sequentially using a memory-based strategy with a memory buffer to retain important samples from previous tasks. We also introduce MMLC, a real-world multimodal leaf-counting dataset designed for a domain
    
[^116]: 从单条轨迹学习分数阶动力学

    Learning Fractional-Order Dynamics from a Single Trajectory

    [https://arxiv.org/abs/2609.18127](https://arxiv.org/abs/2609.18127)

    本文提出了一种简单的两阶段估计器 FO-GS，利用分数差分算子的对角结构逐行解耦辨识问题，从而能够从单条观测轨迹中对分数阶线性时不变系统进行辨识并给出高概率非渐近误差界。

    

    许多现实世界的过程表现出长程依赖性，即当前状态依赖于过去状态缓慢衰减的痕迹，而不仅仅取决于最近的状态。本文研究了从一条长度为 $t$ 的单一观测轨迹中对离散时间分数阶线性时不变系统进行系统辨识的问题，这一设定通过 Grünwald–Letnikov 差分算子刻画了此类非马尔可夫动力学。与马尔可夫系统不同，分数阶系统将估计耦合到整个历史过程上，使得统计分析和实际辨识都更具挑战性。我们提出了分数阶普通最小二乘网格搜索估计器（FO-GS），这是一个简单的两阶段估计器，它利用分数差分算子的对角结构来逐行解耦辨识问题。在稳定性假设下，我们为估计建立了高概率、非渐近的误差界。

    arXiv:2609.18127v1 Announce Type: new  Abstract: Many real-world processes exhibit long-range dependence, where the current state depends on a slowly decaying trace of past states rather than on the most recent state alone. This paper studies system identification for discrete-time fractional-order linear time-invariant systems from a single observed trajectory of length $t$, a setting that captures such non-Markovian dynamics through the Gr\"unwald--Letnikov difference operator. Unlike Markovian systems, fractional-order systems couple estimation across the entire history, making both statistical analysis and practical identification more challenging. We propose \emph{Fractional-Order Ordinary-Least-Squares Grid-Search (FO-GS)}, a simple two-stage estimator that exploits the diagonal structure of the fractional-difference operator to decouple the identification problem row-wise. Under the stability assumption, we establish high-probability, non-asymptotic error bounds for estimating b
    
[^117]: Wasserstein-Fisher-Rao梯度流的保对数凹性与收敛性

    Preservation of Log-Concavity and Convergence of Wasserstein-Fisher-Rao Gradient Flows

    [https://arxiv.org/abs/2609.18118](https://arxiv.org/abs/2609.18118)

    本文证明Wasserstein-Fisher-Rao梯度流在满足曲率条件的强对数凹目标分布下能够保持强对数凹性，据此推导出对称化KL散度的显式非渐近收敛速率，无需暖启动，且收敛速率可加性分解为Wasserstein与Fisher-Rao两部分贡献。

    

    我们研究了Wasserstein-Fisher-Rao（WFR）梯度流在从仅已知归一化常数的概率分布中进行采样时的收敛性。通过将Wasserstein输运与Fisher-Rao生灭动力学相结合，WFR流平衡了探索与选择，被认为是超越朗之万动力学加速收敛的一种有前景的机制。我们证明，对于一类满足额外曲率条件的强对数凹目标分布，WFR流能够保持强对数凹性；与之相比，Wasserstein流仅在高斯情形下才具备这一性质。利用这一结果，我们推导了对称化Kullback-Leibler散度的显式非渐近收敛速率，且无需当前估计中所要求的暖启动。特别地，我们证明收敛速率可加性地分解为Wasserstein与Fisher-Rao两部分贡献，从而确认了……

    arXiv:2609.18118v1 Announce Type: cross  Abstract: We study the convergence of Wasserstein-Fisher-Rao (WFR) gradient flows for sampling from probability distributions known up to a normalisation constant. By combining Wasserstein transport with Fisher-Rao birth-death dynamics, WFR flows balance exploration and selection. These flows have been recognised as a promising mechanism to accelerate convergence beyond Langevin dynamics. We show that for a class of strongly log-concave target distributions satisfying additional curvature conditions, WFR flows preserve strong log-concavity, in contrast to Wasserstein flows which enjoy this property only in the Gaussian setting. Exploiting this result, we derive explicit non-asymptotic convergence rates for the symmetrised Kullback-Leibler divergence, without requiring a warm-start as required in current estimates. In particular, we show that the convergence rate decomposes additively into Wasserstein and Fisher-Rao contributions, thereby confirm
    
[^118]: Token延迟公平性：多租户LLM服务的性能隔离

    Token Latency Fairness: Performance Isolation for Multi-Tenant LLM Serving

    [https://arxiv.org/abs/2609.18112](https://arxiv.org/abs/2609.18112)

    提出FairInference系统，首次提供δ-token公平性保证，确保多租户LLM服务中行为良好客户端的每个token生成延迟最多比隔离运行时多δ个时间单位，实现强大的延迟隔离。

    

    LLM服务通常以共享的多租户服务形式提供，来自一个客户端的高需求工作负载可能导致其他客户端的延迟SLO（服务级别目标）违约。现有的性能隔离解决方案通过排队和批处理公平性等方式，在长期运行中均衡客户端吞吐量。然而，这些方法无法提供延迟隔离保证；因此，行为良好的客户端仍然可能经历其token级延迟的显著退化。在本文中，我们提出了FairInference，它提供了新颖的δ-token公平性保证：对于行为良好的客户端，如果一个token在隔离状态下需要d个时间单位生成，那么在多租户执行中它将在d + δ个时间单位内生成，从而为LLM服务提供强大的延迟隔离保证。为实现这一目标，FairInference解决了LLM服务中的一个关键挑战：在不支持细粒度……（原文截断）的情况下限制共享GPU资源导致的延迟。

    arXiv:2609.18112v1 Announce Type: cross  Abstract: LLM serving is typically offered as a shared, multi-tenant service, where high-demand workloads from one client can cause latency SLO violations for others. Existing solutions for performance isolation equalize client throughput in the long run, for example through queueing and batching fairness. However, these approaches do not provide latency isolation guarantees; as a result, well-behaved clients can still experience significant degradation to their token-level latencies.   In this paper, we present FairInference, which provides the novel {\delta}-token fairness guarantee: for a well-behaved client, if a token is generated in d time units in isolation, it will be generated within d + {\delta} time units in multi-tenant execution, providing strong latency isolation guarantees for LLM serving. To achieve this, FairInference addresses a key challenge of LLM serving: bounding delays from sharing GPU resources without support for fine-gr
    
[^119]: 生成式物理人工智能的全面综述

    A Comprehensive Review of Generative Physical Artificial Intelligence

    [https://arxiv.org/abs/2609.18111](https://arxiv.org/abs/2609.18111)

    本综述系统梳理了生成式物理人工智能（GPAI）领域，提出了涵盖机器人基础模型、视觉-语言-动作模型、大行为模型、扩散策略模型和世界基础模型五大方法的分类体系，并分析了它们的架构基础、应用现状及互补关系。

    

    将大规模基础模型与物理实体相结合，催生了机器人领域的重大进展，被称为生成式物理人工智能（GPAI）。这些智能体AI系统能够在复杂的真实世界情境中自主感知、推理和行动。本综述全面分析了GPAI系统，重点关注其架构基础、当前应用和关键局限性。我们引入了一个包含五种不同方法的分类体系：用于跨平台技能迁移的机器人基础模型（RFM）；用于端到端多模态感知与控制的视觉-语言-动作（VLA）模型；用于类人动作生成的大行为模型（LBM）；基于扩散模型的时序连贯动作生成的扩散策略模型（DPM）；以及用于符合物理规律仿真与数据生成的世界基础模型（WFM）。我们考察了这些方法如何相互补充：WFM生成轨迹……

    arXiv:2609.18111v1 Announce Type: cross  Abstract: The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate tra
    
[^120]: FoundAna：一种用于图异常检测的GNN辅助基础模型

    FoundAna: A GNN-assisted Foundation Model for Graph Anomaly Detection

    [https://arxiv.org/abs/2609.18107](https://arxiv.org/abs/2609.18107)

    FoundAna是首个结合GNN与transformer的图异常检测基础模型，通过异常检测专用的GNN组件与四种互补位置编码捕获局部和全局结构信息，实现了可泛化的跨图异常检测。

    

    图异常检测旨在识别显著偏离预期模式的图结构（如节点、边或子图），它在欺诈检测、垃圾信息识别、网络入侵等关键应用中发挥着重要作用。尽管该领域的方法不断增多，但现有方法遵循“一个数据集一个模型”的范式，由于任务异构性、标签稀缺性和领域多样性，限制了它们在多样化真实场景中的可迁移性。在这项工作中，我们提出了FoundAna，一种用于图异常检测的GNN辅助基础模型——这是首个专为可泛化的跨图异常检测设计的基础模型框架，它将GNN与transformer相结合。FoundAna集成了一个专门针对异常检测的GNN组件和一个由四种互补位置编码增强的标准transformer编码器，使模型能够同时捕获局部和全局结构信息。

    arXiv:2609.18107v1 Announce Type: new  Abstract: Graph anomaly detection aims to identify graph structures (e.g., nodes, edges, or subgraphs) that deviate significantly from expected patterns, which supports critical applications in fraud detection, spam identification, network intrusion, etc. Despite the growing methods in the field, existing approaches follow a one-model-per-dataset paradigm, limiting their transferability across diverse real-world scenarios due to task heterogeneity, label scarcity, and domain variability. In this work, we introduce FoundAna, a GNN-assisted Foundation Model for Graph Anomaly Detection - the first foundation model framework designated for generalizable, cross-graph anomaly detection by combining GNNs and transformers. FoundAna integrates an anomaly detection-specific GNN component with a standard transformer encoder augmented by four complementary positional encodings, which enable the model to capture both local and global structural information. Sp
    
[^121]: iMINDBench：颅内脑电图多机构神经解码基准

    iMINDBench: iEEG Multi-Institution Neural Decoding Benchmark

    [https://arxiv.org/abs/2609.18104](https://arxiv.org/abs/2609.18104)

    提出iMINDBench，一个涵盖三个自然电影观看数据集、十五项解码任务的多机构颅内脑电图神经解码基准，通过标准化预处理流程和固定评估划分，解决了iEEG解码模型泛化能力难以衡量、模型改进与预处理增益难以区分的问题。

    

    颅内脑电图（iEEG）被广泛用于直接从人脑内部电极记录电活动，使其成为一种具有吸引力的神经解码模态。然而，iEEG解码的进展，尤其是朝向通用基础模型的发展，仍然难以可靠地衡量：数据集往往是特定任务或特定机构的，限制了跨任务和跨记录环境泛化能力的证据；且预处理选择会强烈影响性能，使得模型改进难以与预处理带来的提升区分开。因此，我们提出了iMINDBench，一个iEEG多机构神经解码基准，在三个自然电影观看数据集的共享任务套件上，通过十五项解码任务对模型进行评估。该基准还定义了标准化的预处理流程和固定的评估划分，以支持一致的模型比较。利用iMINDBench，我们发现……

    arXiv:2609.18104v1 Announce Type: new  Abstract: Intracranial electroencephalography (iEEG) is widely used to record electrical activity directly from electrodes inside the human brain, making it an attractive modality for neural decoding. However, progress in iEEG decoding, especially toward general-purpose foundation models, remains difficult to measure reliably: datasets are task- or institution-specific, limiting evidence of generalization across tasks and recording environments, and preprocessing choices can strongly influence performance, making model improvements difficult to distinguish from preprocessing gains. Thus, we introduce iMINDBench, an iEEG Multi-Institution Neural Decoding Benchmark that evaluates models on a shared suite of fifteen decoding tasks across three naturalistic movie-watching datasets. The benchmark additionally defines standardized preprocessing tracks and fixed evaluation splits to support consistent model comparisons. Using iMINDBench, we find that the
    
[^122]: 超越像素相似度：面向机器人感知的基于GAN的合成声呐数据的任务感知评估

    Beyond Pixel Similarity: Task-Aware Evaluation of GAN-Based Synthetic Sonar Data for Robotic Perception

    [https://arxiv.org/abs/2609.18100](https://arxiv.org/abs/2609.18100)

    该研究通过对比具有不同判别器感受野的Pix2Pix模型，揭示了传统图像保真度指标（SSIM、PSNR、MSE）无法充分反映GAN合成声呐数据在下游感知任务中的实际表现，从而提出应以任务感知的评估方式来评价合成数据质量。

    

    合成数据可以降低机器人感知训练数据收集与标注的成本，但生成能够保留与下游感知相关特征的传感器观测数据仍然具有挑战性，对于声呐图像而言尤为如此。在这项工作中，我们研究了传统的图像保真度指标是否能够充分反映GAN生成的合成声呐数据的下游感知性能。我们采用Pix2Pix条件生成对抗网络，并配置了四种具有不同感受野的判别器：PixelGAN、PatchGAN-16、PatchGAN-70和ImageGAN。模型使用来自两个数据集的声呐图像进行训练，并通过传统图像保真度指标进行评估，包括结构相似性指数（SSIM）、峰值信噪比（PSNR）和均方误差（MSE）。为了用面向任务的评估来补充这些像素级指标，我们使用了YOLOX-S、YOLOX-L和Fa（摘要截断）

    arXiv:2609.18100v1 Announce Type: cross  Abstract: Synthetic data can reduce the cost of collecting and annotating training data for robotic perception, but generating sensor observations that preserve the characteristics relevant to downstream perception remains challenging, particularly for sonar imagery. In this work, we investigate whether conventional image-fidelity metrics adequately reflect the downstream perception performance of GAN-generated synthetic sonar data. We employ a Pix2Pix conditional generative adversarial network with four discriminator configurations characterized by different receptive fields: PixelGAN, PatchGAN-16, PatchGAN-70, and ImageGAN. The models are trained using sonar imagery from two datasets and evaluated using conventional image-fidelity metrics, including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE). To complement these pixel-level measures with task-oriented evaluation, YOLOX-S, YOLOX-L, and Fa
    
[^123]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^124]: FedPGT：面向时变信道的车载联邦学习渐进式梯度传输方案

    FedPGT: Progressive Gradient Transmission for Vehicular Federated Learning over Time-Varying Channels

    [https://arxiv.org/abs/2609.18089](https://arxiv.org/abs/2609.18089)

    该论文提出FedPGT方案，使车辆根据瞬时信道条件自适应地渐进传输大幅值梯度条目，并通过揭示幂律收益递减特性的收敛界指导在线随机优化决策，解决了车辆移动导致信道快速变化使预定义传输策略失效的问题。

    

    车载联邦学习（VFL）能够为智能交通系统实现保护隐私的协作模型训练，其中通信资源分配和梯度稀疏化技术已被探索用于减少通信开销。然而，车辆的移动性会导致信道条件和传输容量快速变化，使得预先确定的资源分配和稀疏化决策失效。在本文中，我们提出了FedPGT，一种面向时变信道的车载联邦学习渐进式梯度传输方案，其中车辆根据瞬时信道条件渐进地传输大幅值梯度条目。我们建立了一个收敛界，刻画了所传输梯度条目的影响，并揭示了由幂律衰减所支配的收益递减行为。受此结果启发，我们构建了一个用于在线决策的随机优化问题，其中

    arXiv:2609.18089v1 Announce Type: new  Abstract: Vehicular federated learning (VFL) enables privacy-preserving collaborative model training for intelligent transportation systems, where communication resource allocation and gradient sparsification techniques have been explored to reduce communication overhead. However, vehicle mobility leads to rapidly varying channel conditions and transmission capacity, rendering predetermined resource allocation and sparsification decisions ineffective. In this paper, we propose FedPGT, a progressive gradient transmission scheme for VFL over time-varying channels, where vehicles progressively transmit high-magnitude gradient entries in response to instantaneous channel conditions. We establish a convergence bound that characterizes the impact of transmitted gradient entries and reveals diminishing-return behavior governed by a power-law decay. Motivated by this result, we formulate a stochastic optimization problem for online decision-making, where 
    
[^125]: 并非所有层都需要调优：诊断与引导视觉-语言-动作模型的适应

    Not All Layers Need Tuning: Diagnosing and Directing Adaptation in Vision-Language-Action Models

    [https://arxiv.org/abs/2609.18084](https://arxiv.org/abs/2609.18084)

    该研究通过在五个不同架构的VLA模型上测量区域隔离微调下的适应成本，揭示了不同类型的分布偏移会系统性集中在特定网络区域（外观变化集中于视觉编码器、指令变化集中于语言骨干、新物体变化集中于视觉编码器与动作头），并提出了一种仅凭十个无标注观测、无需微调即可诊断适应成本并针对性分配适配容量的流水线。

    

    为新的部署环境微调视觉-语言-动作（VLA）模型成本高昂，然而大多数方法对网络的每个区域都应用统一容量的适配器，仿佛每个区域都需要同等程度的调整。本文在五个架构各异的VLA模型（OpenVLA-OFT、π₀、SmolVLA、DTP、Octo；参数量从93M到7B）上检验了这一假设。通过在区域隔离微调下以归一化参数位移来测量各区域的适应成本，研究揭示了一个适应谱系：外观变化使适应成本集中在视觉编码器，指令变化集中在语言骨干网络，而新物体变化则集中在视觉编码器与动作头，这一规律在全部五种架构中均成立。为利用这一结构性规律，我们提出了一个“观察、诊断、分配、适应”的流水线。仅需十个无标注的目标环境观测且无需微调，诊断模块即可通过结合无参考梯度估计各区域的适应成本。

    arXiv:2609.18084v1 Announce Type: cross  Abstract: Fine-tuning a Vision-Language-Action (VLA) model for a new deployment environment is expensive, yet most methods apply uniform-capacity adapters to every network region as if every region requires equal adjustment. This paper tests that assumption on five architecturally diverse VLAs (OpenVLA-OFT, $\pi_0$, SmolVLA, DTP, Octo; 93M-7B parameters). Measuring per-region adaptation cost as normalized parameter displacement under region-isolated fine-tuning reveals an adaptation spectrum in which appearance shifts concentrate cost in the vision encoder, instruction shifts in the language backbone, and novel-object shifts in the vision encoder together with the action head, across all five architectures. To exploit this structure, we introduce a pipeline that observes, diagnoses, allocates, and adapts. From ten unlabeled target observations and without fine-tuning, the diagnostic estimates per-region cost by combining reference-free gradient 
    
[^126]: 超越嵌入迁移：Grokking迁移与稳定性中的组件角色

    Beyond Embedding Transfer: Component Roles in Grokking Transfer and Stability

    [https://arxiv.org/abs/2609.18078](https://arxiv.org/abs/2609.18078)

    该论文通过大规模控制实验首次系统区分了Grokking热启动迁移中各模型组件的角色，发现迁移内部注意力/MLP权重不仅可将早期准确率提升5.46个百分点并缩短确认延迟，且该优势在双层模型的前瞻性复现实验中得到完全验证。

    

    热启动迁移可以使算法任务快速泛化，但目前尚不清楚哪些模型组件提供了这种增益，以及该增益在持续优化下是否保持稳定。我们在模运算上研究跨算子迁移，并将有效性（早期速度）与稳定性（达到后回撤）区分开来。在一个规模匹配的108次实验组合中（涵盖12个种子块，包括96次2³因子实验和12次规模对照），同时迁移内部注意力/MLP权重（B）与词元嵌入和读出层（E+U）可将早期准确率提高5.46个百分点（Holm校正 p=0.0039），并将确认延迟减少558步（Holm校正 p=0.0088）。虽然读出层加内部块迁移在单层模型中满足预先设定的±500步延迟等效性标准（TOST p=0.0011，尽管在11/12的配对种子中完整迁移更快），但前瞻性的双层复现实验证实了内部块的优势（12/12种子，+704.67积分单位，p=4.88×10⁻⁴）。

    arXiv:2609.18078v1 Announce Type: new  Abstract: Warm-start transfer can make algorithmic tasks generalize rapidly, yet it is unclear which model components provide the gain and whether that gain remains stable under continued optimization. We study cross-operator transfer on modular arithmetic and separate efficacy (early velocity) from stability (post-reach drawdown). In a scale-matched 108-run battery across 12 seed blocks (96-run 2^3 factorial plus 12-run scale control), transferring internal attention/MLP weights (B) alongside token embeddings and readout (E+U) improves early accuracy by 5.46 pp (Holm p=0.0039) and cuts confirmation latency by 558 steps (Holm p=0.0088). While readout plus internal-block transfer satisfies the pre-specified +/-500-step latency equivalence criterion in 1-layer models (TOST p=0.0011, though Full is faster in 11/12 paired seeds), a prospective 2-layer replication confirms the internal-block advantage (12/12 seeds, +704.67 integral units, p=4.88x10^-4)
    
[^127]: vidax：面向加速器网格的视频生成模型统一JAX框架

    vidax: A Unified JAX Framework for Video Generative Models on Accelerator Meshes

    [https://arxiv.org/abs/2609.18077](https://arxiv.org/abs/2609.18077)

    vidax是一个开源的JAX推理框架，为视频生成模型提供了TPU上生产就绪的推理路径，统一了张量并行与序列并行，并支持零拷贝的PyTorch权重转换和超出单设备内存的分辨率支持。

    

    开源视频生成模型几乎全部以PyTorch/CUDA参考实现的形式发布。这使得Cloud TPU pod缺乏生产就绪的推理路径，尽管TPU提供了大型且高性价比的加速器内存池，非常适合长序列时空注意力计算。我们提出了vidax，一个开源的JAX/Flax推理引擎和零拷贝的PyTorch到JAX权重转换器，适用于现代视频生成架构。vidax涵盖了多样化的时空模型集合——包括扩散Transformer、全模态Mixture-of-Transformers、3D VAE、文本编码器和原生采样器——且执行路径中零PyTorch依赖。该框架在单一JAX分片网格上统一了1D张量并行与DeepSpeed-Ulysses序列并行，集成了TPU flash-attention内核，并实现了逐层权重卸载以支持超出单设备内存的参考分辨率。

    arXiv:2609.18077v1 Announce Type: cross  Abstract: Open-source video generative models ship almost exclusively as PyTorch/CUDA reference implementations. This leaves Cloud TPU pods without a production-ready inference path, despite offering large, cost-effective accelerator memory pools ideal for long-sequence spatiotemporal attention. We present vidax, an open-source JAX/Flax inference engine and zero-copy PyTorch-to-JAX weight translator for modern video generation architectures. vidax covers a diverse set of spatiotemporal models --- including Diffusion Transformers, omnimodal Mixture-of-Transformers, 3D VAEs, text encoders, and native samplers --- with zero PyTorch dependency in the execution path. The framework unifies 1D tensor parallelism with DeepSpeed-Ulysses sequence parallelism on a single JAX sharding mesh, integrates TPU flash-attention kernels, and implements per-layer weight offloading to support reference resolutions that exceed single-device memory. We benchmark compil
    
[^128]: 基于因果充分性与必要性的区域解释方法

    Regional Explanations via Causal Sufficiency and Necessity

    [https://arxiv.org/abs/2609.18049](https://arxiv.org/abs/2609.18049)

    该论文提出了SNRE框架，通过因果干预和可微估计器学习输入-输出区域对，使得输入属于某区域成为模型输出落入目标区域的充分且必要条件，为模型预测行为提供了区域级的因果解释。

    

    模型可解释性对于理解和信任机器学习模型至关重要。现有的可解释AI方法通常通过特征重要性、反事实解释或规则来解释预测。然而，对于预测行为何时且仅在何时出现的区域级刻画仍然较少被探索。本文提出了因果充分且必要区域解释（SNRE）框架，该框架学习一个输入区域A和一个输出区域B，使得输入属于A对于模型输出落入B而言既是充分的又是必要的。受经典的必要性与充分性概率（PNS）的启发，我们通过随机干预构建了区域级PNS度量，并推导出用于优化的可微有限样本估计器。SNRE采用显式且可解释的代数区域族来参数化输入-输出区域对，并结合可学习的特征掩码，在……（原文此处截断）

    arXiv:2609.18049v1 Announce Type: new  Abstract: Model explainability is essential for understanding and trusting machine learning models. Existing explainable AI methods often explain predictions through feature importance, counterfactual explanations, or rules. However, a region-level characterization of when and only when a prediction behavior arises remains less explored. This paper proposes Causal Sufficient and Necessary Regional Explanations (SNRE), a framework that learns an input region $A$ and output region $B$ such that membership in $A$ is both sufficient and necessary for the model output to fall in $B$. Motivated by the classical Probability of Necessity and Sufficiency (PNS), we formulate a region-level PNS measure through stochastic interventions and derive a differentiable finite-sample estimator for optimization. SNRE parameterizes the input-output region pair with explicit and interpretable algebraic region families, together with a learnable feature mask, balancing 
    
[^129]: 从压缩向量表示中进行精确的语义读出

    Exact semantic readout from compressed vector representations

    [https://arxiv.org/abs/2609.18047](https://arxiv.org/abs/2609.18047)

    本文给出了压缩向量表示能够精确线性或仿射读出谓词真值条件的充要行空间判据，并通过实验发现预训练词向量虽大多严格可分，但无一能实现精确读出。

    

    我们刻画了压缩向量表示何时能够对有限词表的真值条件进行精确的线性或仿射读出：即每个谓词对应一个固定映射，将每个实体向量映射到相应的真值向量。一个充要的行空间条件决定了这种读出是否存在；增广真值矩阵的秩为 r，因此在线性情形下最小维度为 r，在仿射情形下为 r-1。精确读出返回的值位于一个共享的真值基上，布尔联结词在该基上的作用保持不变；而仅实现可分性则需要一个中间阈值。对于二元关系，恒等关系或严格全序的精确双线性读出要求实体向量线性无关。基于 GloVe 和 word2vec 的实验区分了精确仿射恢复、线性可分性和留出集预测：大多数谓词是严格可分的，但没有任何谓词能从预训练嵌入中获得精确的仿射读出。监督式的直推训练可以达到精确……（原文摘要在此处截断）

    arXiv:2609.18047v1 Announce Type: new  Abstract: We characterize when compressed vector representations admit exact linear or affine readouts of a finite lexicon's truth conditions: one fixed map per predicate, sending each entity vector to the corresponding truth vector. A necessary and sufficient row-space condition determines existence; the augmented truth matrix has rank r, giving minimum dimension r in the linear case, and r-1 in the affine. Exact readouts return values in a shared truth basis on which Boolean connectives act unchanged; separability alone requires an intervening threshold. For binary relations, exact bilinear readout of identity or strict total order requires linearly independent entity vectors. Experiments with GloVe and word2vec distinguish exact affine recovery, linear separability, and held-out prediction: most predicates are strictly separable, but none admits an exact affine readout from the pretrained embeddings. Supervised transductive training attains exa
    
[^130]: 隐藏智能体下的结构推断

    Structural Inference under Hidden Agents

    [https://arxiv.org/abs/2609.18045](https://arxiv.org/abs/2609.18045)

    该论文首次提出并形式化了“隐藏智能体下的结构推断”问题，通过破解交互恢复与轨迹重建之间的循环依赖难题，实现了对不可观测智能体的轨迹与交互结构的联合推断。

    

    从多智能体动力学中恢复潜在交互结构对于理解和预测交互系统非常重要。基于轨迹的结构推断已取得了令人瞩目的性能，但传统的形式化方法假设所有被建模智能体的轨迹都是可获得的。在实践中，由于感知能力有限、遮挡或通信故障，智能体在部署时可能变得不可观测。现有研究已经考虑了不可见节点估计、部分观测下的结构推断以及缺失值填补，然而对隐藏智能体轨迹及其交互的联合恢复仍然缺乏充分探索。我们将该问题形式化为隐藏智能体下的结构推断。其关键难点在于一种循环依赖：恢复涉及隐藏智能体的交互需要对其轨迹的估计，而轨迹重建本身又可以受益于结构信息。为了解决……

    arXiv:2609.18045v1 Announce Type: new  Abstract: Recovering latent interaction structures from multi-agent dynamics is important for understanding and predicting interacting systems. Trajectory-based structural inference has achieved promising performance, but conventional formulations assume that the trajectories of all modeled agents are available. In practice, agents may become unobserved at deployment because of limited sensing, occlusion, or communication failure. Existing studies have considered unseen-node estimation, structural inference under partial observations, and missing-value imputation, yet the joint recovery of hidden-agent trajectories and their interactions remains underexplored. We formulate this problem as structural inference under hidden agents. Its key difficulty is a circular dependency: recovering interactions involving a hidden agent requires an estimate of its trajectory, while trajectory reconstruction can itself benefit from structural information. To addr
    
[^131]: 用于Hα 6562.8 Å与Ca II 8542.1 Å光谱快速多层反演的物理信息神经网络

    Physics-Informed Neural Networks for Fast Multilayer Spectral Inversion of H{\alpha} 6562.8 A and Ca II 8542.1 A Spectra

    [https://arxiv.org/abs/2609.18025](https://arxiv.org/abs/2609.18025)

    该研究提出了一种物理信息神经网络框架，在保留多层光谱反演物理可解释性的同时，大幅加速了太阳色球Hα和Ca II光谱的大规模反演计算。

    

    Hα 6562.8 Å和Ca II 8542.1 Å等强色球吸收线为太阳色球层的等离子体动力学和热结构提供了重要的诊断手段。多层光谱反演（MLSI）提供了一个物理上可解释的框架，利用有限数量的辐射转移层来建模这些谱线，但传统的MLSI依赖于逐像素的非线性最小二乘拟合，对于大型成像光谱数据集而言计算成本高昂。本文引入了一种物理信息神经网络（PINN）框架，在保留其解析辐射转移公式的同时加速MLSI。该网络直接从观测谱线轮廓预测MLSI参数，并将其输入可微分的MLSI正向模型以合成光谱。训练采用两阶段方法：初始阶段仅通过光谱重建损失进行优化，随后结合……（原文摘要在此处截断）

    arXiv:2609.18025v1 Announce Type: cross  Abstract: Strong chromospheric absorption lines such as H$\alpha$ 6562.8 A and Ca II 8542.1 A provide vital diagnostics of plasma dynamics and thermal structure in the solar chromosphere. Multilayer spectral inversion (MLSI) offers a physically interpretable framework for modeling these lines using a finite number of radiative-transfer layers, but conventional MLSI relies on pixel-by-pixel nonlinear least-squares fitting, making it computationally expensive for large imaging spectroscopic data sets. Here, we introduce a physics-informed neural-network (PINN) framework to accelerate MLSI while preserving its analytic radiative-transfer formulation. The network predicts MLSI parameters directly from observed line profiles and passes them through a differentiable MLSI forward model to synthesize spectra. Training follows a two-stage approach: an initial stage optimized solely via spectral reconstruction loss, followed by fine-tuning that combines s
    
[^132]: 更新未必更公平：文生图AI跨代模型中的性别刻板印象

    Newer Is Not Fairer: Gender Stereotyping in Text-to-Image AI Across Model Generations

    [https://arxiv.org/abs/2609.18007](https://arxiv.org/abs/2609.18007)

    该研究通过生成8,000张图像对比四代Stable Diffusion模型，发现文生图AI普遍存在严重性别刻板印象（76.4%的图像主体为男性，甚至在传统女性主导的职业中也有57.6%为男性），且更新的模型并未更公平。

    

    文本到图像生成模型被广泛应用于专业和创意领域，但它们如何在不同职业中呈现性别——以及更新的模型是否更公平——在多个模型世代间的理解仍然不足。我们评估了20个职业、5个提示词模板和4个Stable Diffusion模型世代（SD 1.5、SD 2.1、SDXL、SD 3 Medium）中的性别呈现，共生成8,000张图像（每个职业-模型组合n=100，即5个提示词×20张图像），并使用DeepFace对所有图像进行性别分类。在这8,000张开源图像中，76.4%的主体为男性（95% CI [75.1%, 78.7%]，p < 2.2 × 10^-16，经Benjamini-Hochberg校正）。更引人注目的是，对于历史上由女性主导的职业，57.6%的图像主体为男性（原始p = 3.43 × 10^-22，BH校正后p = 1.71 × 10^-21）。本文报告的所有九项显著检验在10项检验的BH校正后依然显著。当与美国劳工统计局的劳动力数据（摘要在此处截断）进行比较时……

    arXiv:2609.18007v1 Announce Type: cross  Abstract: Text-to-image generative models are widely used in professional and creative settings, yet how they represent gender across occupations -- and whether newer models are fairer -- remains poorly understood across multiple generations. We evaluate gender representation across 20 occupations, 5 prompt templates, and 4 Stable Diffusion model generations (SD 1.5, SD 2.1, SDXL, SD 3 Medium), generating 8,000 images with n = 100 per occupation-model cell (5 prompts x 20 images), and classifying all with DeepFace. Across the 8,000 open-source images, 76.4% show male subjects (95% CI [75.1%, 78.7%], p < 2.2 x 10^-16, Benjamini-Hochberg adjusted). More strikingly, 57.6% of images for historically female-coded occupations show male subjects (raw p = 3.43 x 10^-22, BH-adjusted p = 1.71 x 10^-21). All nine significant tests reported in this paper survive BH correction across 10 tests. When compared against U.S. Bureau of Labor Statistics workforce d
    
[^133]: 一种用于衡量推理优化如何影响输出质量的校准化测量工具

    A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality

    [https://arxiv.org/abs/2609.18005](https://arxiv.org/abs/2609.18005)

    本文提出了一种经过正式校准的LLM评判测量方法，通过引入分布上与原模型完全一致的“零条件”验证机制，实现了对量化、早退、投机解码等推理加速技术对输出质量影响的严格、可跨系统比较的测量。

    

    大语言模型优化是一个活跃的研究领域，涵盖模型权重量化、跳过层的早退方法以及投机解码。每个研究方向都使用各自的质量评估方式，通常是一个独特的基准测试分数，很少有方法能接近其他科学学科所要求的测量精度。我们提出了一种严格的方法来衡量输出质量，适用于跨系统和跨技术的比较。我们使用大语言模型作为评判者来为输出打分，但对该评判者进行了正式校准：我们比较它在同一模型对相同提示词进行两次普通运行时的评分，验证其对统计上等价的输出不存在系统性偏好，并测量其每样本噪声。每种实验设计还包含一个“零”条件，该条件在分布上被证明与未修改的模型完全相同，其测得的差异必须为零。利用这一工具，我们测量了几种加速技术……

    arXiv:2609.18005v1 Announce Type: new  Abstract: Large language model optimization is an active research area, spanning quantization of model weights, early-exit methods for skipping layers, and speculative decoding. Each track uses its own quality measures, typically an idiosyncratic benchmark score. Few approach the measurement precision required by other scientific disciplines.   We propose a rigorous methodology for measuring output quality, suitable for cross-system and cross-technique comparison. We score outputs with an LLM as a judge, but calibrate the judge formally: we compare its scores on two ordinary runs of a model given the same prompts, verifying that it shows no systematic preference between statistically equivalent outputs and measuring its per-sample noise. Each design also includes a 'null' condition, provably identical in distribution to the unmodified model, whose measured difference must be zero.   With this one instrument we measure several acceleration techniqu
    
[^134]: 内在的注意力：选择性状态空间模型中的共识动力学

    The Attention Within: Consensus Dynamics in Selective State Space Models

    [https://arxiv.org/abs/2609.17997](https://arxiv.org/abs/2609.17997)

    该论文从动力系统视角证明，选择性状态空间模型（SSM）核心的递归机制与Transformer中的注意力类似，同样会驱动token达成共识（聚簇坍缩）。

    

    选择性状态空间模型（SSM）近来已成为Transformer的一个引人注目的替代方案，它在保持有竞争力的性能的同时，显著提升了推理效率。在每个SSM层中，一系列隐藏状态通过递归进行传播，从而混合不同token的信息。尽管使用了不同的机制，这种混合所起的作用类似于Transformer中的注意力。事实上，最近的研究表明，这两种架构可能比表面看起来更加接近，因为这种递归可以表示为类似线性注意力的形式。在Transformer中，注意力已知会驱动token聚集成簇，即达成共识，在极限情况下坍缩到单一方向。因此，我们提出这样的问题：SSM核心的递归是否也会像Transformer中的注意力那样，驱动token达成共识？为了回答这个问题，我们从动力系统的视角研究SSM，对token跨层的演化进行建模……

    arXiv:2609.17997v1 Announce Type: cross  Abstract: Selective state space models (SSMs) have recently emerged as a compelling alternative to transformers, combining competitive performance with substantially improved inference efficiency. At each SSM layer, a sequence of hidden states are propagated by a recurrence, mixing information of different tokens. Despite using a different mechanism, this mixing plays a role analogous to attention in transformers. In fact, recent works have shown that the two architectures may be closer than they first appear, as this recurrence admits a formulation akin to linear attention. In transformers, attention is known to drive the tokens to cluster, i.e., to reach consensus, collapsing in the limit to a single direction. Thus, we ask: does the recurrence at the core of SSMs drive the tokens to consensus, as attention does in transformers?   To answer this question, we take a dynamical systems perspective on SSMs, modeling the evolution of tokens across 
    
[^135]: QuanText：在文本数据共享中保护数据集级秘密

    QuanText: Protecting Dataset-Level Secrets in Textual Data Sharing

    [https://arxiv.org/abs/2609.17995](https://arxiv.org/abs/2609.17995)

    QuanText提出了一种无需训练、与大语言模型无关的文本随机量化数据发布机制，能够在保护文本数据集中数据集级全局秘密（如敏感属性比例）的同时保持数据效用，弥补了差分隐私对聚合属性保护不足的缺陷。

    

    自然语言数据集支持许多下游应用和研究工作，但发布文本可能会暴露底层数据源的敏感全局属性，例如与特定性别、诊断或政治立场相关的记录比例。现有工作主要集中于恢复此类全局属性的属性推断攻击，而保护这些数据集级秘密的防御方法仍然有限。差分隐私虽然在保护个体记录方面有效，但对聚合属性只能提供较弱的保护。我们提出了文本随机量化，这是一种无需训练且与大语言模型无关的数据发布机制，能够在保持数据效用的同时保护文本数据集中的全局秘密。给定一个数据集级秘密，例如具有特定诊断的记录比例，以及需要保持其效用的属性，例如主题……

    arXiv:2609.17995v1 Announce Type: new  Abstract: Natural-language datasets support many downstream applications and research studies, but releasing text can reveal sensitive global properties of the underlying data source, such as the proportion of records associated with a particular gender, diagnosis, or political stance. Existing work has largely focused on property inference attacks that recover such global properties, while defenses for protecting these dataset-level secrets remain limited. Differential privacy, although effective for protecting individual records, provides only weak protection for aggregate properties. We propose Randomized Quantization for Text (QuanText), a training-free and large-language-model-agnostic data release mechanism that protects global secrets in textual datasets while preserving data utility. Given a dataset-level secret, such as the proportion of records with a particular diagnosis, and attributes whose utility should be preserved, such as topic a
    
[^136]: 可操作的帕累托前沿：将离线搜索蒸馏为多目标无人机边缘计算调度的运行时控制

    The Operable Pareto Front: Distilling Offline Search into Run-Time Control for Multi-Objective UAV Edge-Computing Scheduling

    [https://arxiv.org/abs/2609.17992](https://arxiv.org/abs/2609.17992)

    提出PrefDT——首个偏好条件化决策变换器，将期望的能耗-时延权衡作为模型输入，仅需一次离线训练即可在运行时按需返回多目标无人机边缘计算调度帕累托前沿上的任意点，并具备预算跟踪与用户报告丢失容错能力。

    

    无人机移动边缘计算（MEC）机群需要在能耗与时延之间进行权衡，其调度方案构成一个帕累托前沿；当机群能够在运行时按需给出该前沿上的任意一点时，我们称该调度器是“可操作的”。我们提出了PrefDT，据我们所知，这是首个针对联合轨迹、接入与卸载调度问题的偏好条件化决策变换器（Decision Transformer）。其思想源自语言建模：我们将期望的权衡关系作为输入提供给模型，使得单一模型只需离线训练一次，即可在单次rollout中返回曲线上的任意期望点。机群状态通过带有每用户旁路的注意力池化进行汇总，因此即使用户报告丢失，调度器仍能正常工作。能耗目标是一个运行预算，会根据机群的实际消耗进行扣减。因此，当风力或负载导致实际消耗偏离计划时，策略能够跟踪偏差并维持其预算。（注：原文摘要在此处不完整）

    arXiv:2609.17992v1 Announce Type: cross  Abstract: A UAV mobile edge computing (MEC) fleet trades energy against delay, and its schedules form a Pareto front; we call a scheduler operable when the fleet can be asked for any point on that front at run time. We propose PrefDT, to the best of our knowledge the first preference-conditioned Decision Transformer for the problem of joint trajectory, association and offloading scheduling. Its idea comes from language modeling: we hand the model the desired trade-off as an input, such that a single model only needs to be trained once offline to return any desired point on the curve in one rollout. The fleet's state is summarized by attention pooling with a per-user bypass, so the scheduler keeps working when user reports are lost. The energy target is a running budget decremented by what the fleet actually spends. As a result, when wind or load pushes consumption off the plan, the policy can track the difference and hold its budget. Because no 
    
[^137]: 参数化交互式量子分类器的傅里叶分析

    Fourier Analysis of Parametrized Interactive Quantum Classifiers

    [https://arxiv.org/abs/2609.17991](https://arxiv.org/abs/2609.17991)

    本文通过推导单目标量子比特参数化交互式量子分类器的闭式解析解，揭示了哈密顿量参数以傅里叶方式控制分类器输出的常数、正弦和余弦分量，建立了量子特征映射的傅里叶解释并据此提出了广义哈密顿量编码族。

    

    交互式量子分类器（IQCs）是一类受开放量子系统启发的量子机器学习模型，其中目标量子比特与环境之间的相互作用由哈密顿量描述。先前的研究工作引入了不同的哈密顿量参数化方法，并通过实验证明它们能够提升分类性能，但这些参数在所得到的分类器中所起的作用仍然缺乏深入理解。在本工作中，我们推导了具有单个目标量子比特的参数化交互式量子分类器所产生的约化量子信道的闭式表达式。该解析解明确揭示了哈密顿量参数如何控制分类器输出中的常数、正弦和余弦分量，从而为所诱导的特征映射建立了傅里叶解释。这一分析进一步启发了一类广义的哈密顿量编码族，其中包括矩阵参数化的环境哈密顿量，其傅里叶特性……（原文摘要在此处被截断）

    arXiv:2609.17991v1 Announce Type: cross  Abstract: Interactive Quantum Classifiers (IQCs) constitute a family of quantum machine learning models inspired by open quantum systems, in which the interaction between a target qubit and an environment is described by a Hamiltonian. Previous works introduced alternative Hamiltonian parameterizations and showed empirically that they can improve classification performance, but the role of these parameters in the resulting classifier remains poorly understood. In this work, we derive a closed-form expression for the reduced quantum channel generated by a parametrized IQC with a single target qubit. The analytical solution explicitly reveals how the Hamiltonian parameters control the constant, sine, and cosine components of the classifier output, establishing a Fourier interpretation of the induced feature map. This analysis motivates a generalized family of Hamiltonian encodings, including matrix-parameterized environmental Hamiltonians whose Fo
    
[^138]: TuiML：面向AI智能体的机器学习

    TuiML: Machine Learning for AI Agents

    [https://arxiv.org/abs/2609.17984](https://arxiv.org/abs/2609.17984)

    TuiML是一个专为AI智能体设计的机器学习库，通过机器可读的元数据描述组件、验证并追踪每次调用、支持会话导出为可复现笔记本，解决了传统面向人类程序员的库在智能体使用时功能不可见、错误延迟暴露和实验状态丢失的问题。

    

    诸如Weka和scikit-learn等机器学习库是为人类程序员设计的。语言模型智能体现在通过从记忆中回忆API并编写代码来使用这些相同的库，这种方法隐藏了库所提供的功能，将错误延迟到运行时才暴露，并且在多轮交互之间丢失实验状态。我们提出了TuiML，一个专为AI智能体构建的独立机器学习库，在监督学习、无监督学习、时间序列、数据处理、调优和评估任务中提供原生算法。每个组件通过机器可读的元数据和参数模式来描述自身，因此智能体可以搜索库、检查组件、组合经过验证的工作流，并注册新的组件使其随后即可被发现。每次调用都经过验证、设定随机种子并进行追踪，会话可以导出为可运行的笔记本，使实验在构建上即可复现。一个规范层驱动模型上下文协议（MCP）、智能体框架……

    arXiv:2609.17984v1 Announce Type: new  Abstract: Machine-learning libraries such as Weka and scikit-learn were designed for human programmers. Language-model agents now use these same libraries by recalling APIs from memory and writing code, an approach that hides what a library offers, delays errors until runtime, and loses experimental state between turns. We present TuiML, a self-contained machine-learning library built for AI agents, with native algorithms across supervised, unsupervised, time-series, data handling, tuning, and evaluation tasks. Every component describes itself through machine-readable metadata and parameter schemas, so an agent can search the library, inspect components, compose validated workflows, and register new ones that become discoverable in turn. Every call is validated, seeded, and traced, and sessions export as runnable notebooks, making experiments reproducible by construction. One specification layer drives the Model Context Protocol (MCP), agent-frame
    
[^139]: 单循环匹配多循环复杂度：非凸-凹极小极大优化中的最优优化平稳性与已知最佳博弈平稳性

    Matching Multi-Loop Complexities with a Single Loop: Optimal Optimization Stationarity and Best-Known Game Stationarity in Nonconvex--Concave Minimax Optimization

    [https://arxiv.org/abs/2609.17973](https://arxiv.org/abs/2609.17973)

    该论文提出了一种结合投影外梯度更新、对偶动量和移动近端中心的单循环投影阻尼外梯度算法，在非凸-凹极小极大优化中以单循环方法匹配了多循环方法的复杂度，同时实现了最优的优化平稳性保证和已知最佳的博弈平稳性保证。

    

    我们为光滑非凸-凹极小极大优化问题引入了一个新的单循环算法框架。由此得到的投影阻尼外梯度方法结合了投影外梯度更新、对偶动量和移动近端中心。在优化平稳性和博弈平稳性两种准则下，我们的方法在单循环一阶方法中达到了已知的最佳复杂度。对于优化平稳性，我们的方法达到了 $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$ 的梯度复杂度，其中 $L$ 是梯度Lipschitz常数，$D_Y$ 是对偶可行集直径的界，$\bar\Delta_0$ 是一个涉及值函数间隙和初始梯度的初始化量。此外，通过引入固定中心预热阶段，复杂度可以改进为 $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$，外加一个可忽略的低阶可加代价，其中 $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$。我们进一步

    arXiv:2609.17973v1 Announce Type: cross  Abstract: We introduce a new single-loop algorithmic framework for smooth nonconvex--concave minimax optimization. The resulting projected damped extragradient method combines projected extragradient updates, dual momentum, and a moving proximal center. Under both the optimization-stationarity and game-stationarity criteria, our method achieves the best-known complexity among single-loop first-order methods. For optimization stationarity, our method achieves a gradient complexity of $O(L^2D_Y\bar\Delta_0\varepsilon^{-3})$, where $L$ is the gradient Lipschitz constant, $D_Y$ bounds the diameter of the dual feasible set, and $\bar\Delta_0$ is an initialization quantity involving the value-function gap and the initial gradients. Moreover, by incorporating a fixed-center warm-up phase, the complexity can be improved to $O(L^2D_Y\Delta_\phi\varepsilon^{-3})$, up to an additive lower-order cost, where $\Delta_\phi:=\phi(x_0)-\inf_x\phi(x)$. We further
    
[^140]: 面向地下抽水蓄能系统的混合整数非线性可微预测控制

    Mixed-Integer Nonlinear Differentiable Predictive Control for Underground Pumped Hydro Energy Storage Systems

    [https://arxiv.org/abs/2609.17964](https://arxiv.org/abs/2609.17964)

    本文扩展了混合整数可微预测控制框架，通过保持梯度的并行可微仿真器、捕捉长程时间依赖的Transformer编码器和Gumbel-Softmax温度退火调度三项创新，以自监督方式学习神经控制策略，解决地下抽水蓄能系统日前调度中多模态离散决策与非线性动力学的可微优化控制难题。

    

    本文将混合整数可微预测控制（MI-DPC）扩展到地下抽水蓄能系统（UPHES）中出现的多模态离散决策和非凸多项式动力学问题。通过Gumbel-Softmax层将问题参数映射为连续设定点和整数模式选择的神经策略，通过对非线性动力学模型微分计算有限时域控制目标的期望，以自监督方式进行训练。三个方法学贡献使这一扩展成为可能：一个保持梯度幅值的并行可微仿真器、一个捕捉长程时间依赖关系的Transformer编码器，以及一个用于正则化组合搜索的Gumbel-Softmax温度退火调度。我们在UPHES的日前调度问题上演示了该框架，这是一个包含非线性机组性能曲线和库容相关约束的大规模混合整数最优控制问题。

    arXiv:2609.17964v1 Announce Type: cross  Abstract: This paper extends Mixed-Integer Differentiable Predictive Control (MI-DPC) to multi-modal discrete decisions and nonconvex polynomial dynamics arising in Underground Pumped Hydro Energy Storage Systems (UPHES). A neural policy mapping problem parameters to continuous setpoints and integer mode selections via a Gumbel-Softmax layer is trained in a self-supervised manner by differentiating the expectation of the finite horizon control objective through the nonlinear dynamics model. Three methodological contributions enable this extension: a parallel differentiable simulator that preserves gradient magnitude, a Transformer encoder that captures long-range temporal dependencies, and a Gumbel-Softmax temperature annealing schedule that regularizes the combinatorial search. We demonstrate the framework on day-ahead scheduling of a UPHES, a large-scale mixed-integer optimal control problem with nonlinear unit performance curves and volume-he
    
[^141]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^142]: 超图中的最大强独立集：归约、界与贪心证书

    Maximum Strong Independent Sets in Hypergraphs: Reductions, Bounds, and Greedy Certificates

    [https://arxiv.org/abs/2609.17951](https://arxiv.org/abs/2609.17951)

    本文针对有限超图中的最大强独立集问题，提出了精确归约、上界估计、穿孔与覆盖证书收紧方法，并分析了基于块权重的分层贪心聚类算法，可应用于多带LSH-MinHash去重等仅支持局部约束的场景。

    

    我们研究有限超图中的最大强独立集问题：寻找一个最大的顶点集，使其与每条超边至多相交一个顶点。这一目标出现在每个观测块都是局部不相容约束、但在重叠块之间进行传递闭包并不合理的场景中。一个典型的动机例子是多带 LSH-MinHash 去重：每个碰撞桶仅提供局部证据，而连通分量收缩可能强加虚假的全局等价关系。本文为该问题建立了一套基于关联结构的工具箱：我们证明了针对支配关系、关联孪生和权重为1的块的精确归约；推导了闭式和低权重上界；引入了穿孔与覆盖证书以进一步收紧这些上界；并分析了一种由块权重和剩余关联驱动的分层贪心聚类算法。算法分析涵盖了可行性、极大性和条件最优性等内容。

    arXiv:2609.17951v1 Announce Type: new  Abstract: We study the maximum strong independent set problem in a finite hypergraph: find the largest vertex set that intersects every hyperedge in at most one vertex. This objective arises whenever each observed block is a local incompatibility constraint but transitive closure across overlapping blocks is not justified. A motivating example is multi-band LSH-MinHash deduplication, where each collision bucket gives local evidence, while connected-component contraction can impose spurious global equivalences. The paper develops an incidence-structural toolkit for this problem. We prove exact reductions for dominance, incidence twins, and weight-1 blocks; derive closed-form and low-weight upper bounds; introduce puncturing and covering certificates that sharpen those bounds; and analyze a layered greedy clustering algorithm driven by block weights and residual incidence. The algorithmic analysis includes feasibility, maximality, conditional optima
    
[^143]: ASPIRE：面向长上下文大语言模型推理的异步批量自推测解码

    ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference

    [https://arxiv.org/abs/2609.17943](https://arxiv.org/abs/2609.17943)

    ASPIRE提出了一种非同步的批量自推测解码框架，通过统一混合前向计算、基于接受率估计和批次感知成本模型的在线调度器，让批中每个请求独立决定验证时机，从而加速长上下文大语言模型推理。

    

    长上下文大语言模型推理受注意力机制的瓶颈制约，其重复的KV缓存读取使得解码成为内存受限的操作。自推测解码通过使用稀疏注意力起草token、再用完整注意力进行验证来缓解这一问题，但现有的批量方法仍然是同步的：批中的所有请求共享单一的起草-验证调度，尽管最优起草长度在不同请求之间差异很大，并且在每个请求内部也会动态变化。我们提出ASPIRE，一个建立在三个组件之上的非同步批量自推测解码框架。首先，统一的混合前向计算允许起草和验证请求共存于同一批量前向传播中，消除了对全局起草-验证阶段的需求。其次，轻量级的在线推测调度器使用每个请求的接受率估计和批次感知的成本模型，让每个请求独立地选择何时进行验证。第三，起草内部刷新层（摘要截断）……

    arXiv:2609.17943v1 Announce Type: cross  Abstract: Long-context LLM inference is bottlenecked by attention, whose repeated KV-cache reads make decoding memory-bound. Self-speculative decoding alleviates this by drafting tokens with sparse attention and verifying them with full attention, but existing batched methods remain synchronized: all requests in a batch share a single draft-verify schedule, even though the optimal draft length varies widely across requests and changes dynamically within each request. We propose ASPIRE, a non-synchronized batched self-speculative decoding framework built on three components. First, a unified mixed forward allows drafting and verifying requests to coexist in the same batched forward pass, removing the need for global draft-verify phases. Second, a lightweight online speculation scheduler uses per-request acceptance-rate estimates and a batch-aware cost model to let each request independently choose when to verify. Third, an intra-draft refresh lay
    
[^144]: 论线性参数模型下混合有序变量与指数族因果有向无环图（DAG）的可辨识性

    On the Identifiability of Mixed Ordinal and Exponential Family Causal DAGs under Linear Parametric Models

    [https://arxiv.org/abs/2609.17942](https://arxiv.org/abs/2609.17942)

    本文证明了在线性参数模型中，只要有序节点至少有三个类别且指数族节点至少有三个支撑点，连接这两类节点的每条边的方向都可仅凭联合分布在任意参数取值下被辨识，并通过反向论证证明了这两个条件的必要性。

    

    本文研究了线性参数模型（LPM）中节点服从有序logit模型或正则单参数指数族时的可辨识性问题。研究结果超越了经典的联立结构方程模型，也超越了节点观测来自同质分布族情形下的已有结论。主要结果证明了：只要有序节点具有至少三个类别，且指数族节点具有至少三个支撑点，则连接有序节点与指数族节点的每条边的方向都可以在任意参数取值下仅凭联合分布被辨识出来，且对充分统计量不作任何限制。反向结果（converse）表明这两个条件都是必要的：三类别条件仅在充分统计量为仿射函数时才起约束作用，而三支撑点条件在规范链接函数下起约束作用。该可辨识性保证还可进一步扩展到对每条此类混合有序边的定向。（注：原文摘要在此处截断，内容不完整）

    arXiv:2609.17942v1 Announce Type: new  Abstract: The problem of identifiability in linear parametric models (LPMs) whose nodes follow either an ordered logit model or a regular one-parameter exponential family is evaluated. The results go beyond classical structural equation models as well as results for nodes with observations from a homogeneous family of distributions. The main result establishes that the orientation of every edge joining an ordinal node to an exponential-family node is identifiable from the joint distribution alone at every parameter value, provided the ordinal node has at least three categories and the exponential-family node at least three points of support, with no restriction on the sufficient statistic. Converses show that both requirements are necessary: the three-category requirement is binding only for affine sufficient statistics, and the three-point requirement is binding under the canonical link. The guarantee extends to orienting every such mixed ordinal
    
[^145]: 超越前一层：稀疏MoE路由中的残差预测结构

    Beyond the Previous Layer: Residual Predictive Structure in Sparse MoE Routing

    [https://arxiv.org/abs/2609.17940](https://arxiv.org/abs/2609.17940)

    该研究发现稀疏MoE路由器的专家选择历史具有超越紧邻上一层的残差预测结构——更早层的专家选择包含显著的额外预测信息，能显著提升对下一层路由决策的预测精度。

    

    arXiv:2609.17940v1 公告类型： new 摘要：稀疏混合专家模型通过一系列专家选择来路由每个token。我们探究紧邻的前一次选择是否足以总结这一轨迹以预测下一个路由器。使用冻结的OLMoE和JetMoE模型，我们在保留最近一次选择作为共同基线的同时，测量来自更早专家选择的留出预测增益。在OLMoE中，将历史从一层扩展到十一层，使路由器logit的R²从0.59879提升至0.66544。一项预注册的JetMoE复制实验在两个目标深度上分别获得了0.14275和0.20528的四层增益，配对自助法置信区间均高于零。这些增益在非线性解码下依然存在：将历史信息加入小型多层感知机使R²分别提高0.17137和0.21861，而仅对最近状态进行非线性解码相比线性探针仅增加0.00139和0.00936。参数匹配的对照实验保留了这一优势，且交叉拟合的历史残差……（摘要被截断）

    arXiv:2609.17940v1 Announce Type: new  Abstract: Sparse mixture-of-experts models route each token through a sequence of expert selections. We ask whether the immediately preceding selection adequately summarizes this trajectory for predicting the next router. Using frozen OLMoE and JetMoE models, we measure the held-out predictive gain from earlier expert selections while retaining the most recent selection as a common baseline. In OLMoE, extending the history from one to eleven layers raises router-logit $R^2$ from 0.59879 to 0.66544. A preregistered JetMoE replication yields four-layer gains of 0.14275 and 0.20528 at two target depths, with paired bootstrap intervals above zero. These gains survive nonlinear decoding: adding history to a small multilayer perceptron improves $R^2$ by 0.17137 and 0.21861, whereas nonlinear decoding of the recent state alone adds 0.00139 and 0.00936 over a linear probe. Parameter-matched controls preserve the advantage, and cross-fitted history residua
    
[^146]: 定位隐藏故障使长程智能体更加可靠

    Locating Hidden Failures Makes Long-Horizon Agents More Reliable

    [https://arxiv.org/abs/2609.17930](https://arxiv.org/abs/2609.17930)

    该研究通过分析2518条智能体轨迹并将6967个错误归纳为78种失败类型，发现长程智能体在首次犯错后往往难以自我恢复和纠错，即使最终“成功”的运行也可能造成数据删除等不可逆伤害，因此定位隐藏故障是提升智能体可靠性的关键。

    

    随着AI智能体承担漫长的自主任务，我们越来越多地扮演监督者而非执行者的角色，然而我们几乎完全以它们是否最终成功来评判它们。最终结果无法揭示一次运行在何处出了问题、智能体是否从中恢复、或它在过程中造成的不可逆伤害，而长程智能体在哪里失败仍然没有被系统地梳理。我们研究了软件工程、计算机使用和科学领域共2518条接近真实部署环境的智能体轨迹，并将6967个错误归类为78种失败类型。失败呈现出一种反复出现的特征：在犯下第一个错误后，智能体往往无法恢复，也很少能自己发现错误，因此运行在看似正确的情况下继续失控进行；智能体能否恢复取决于任务本身和环境的反馈，而不是运行它的智能体框架。长程智能体在通往“通过”结果的过程中可能造成真实伤害：即使被评分为成功解决的运行也会删除数据、破坏软件（摘要在此处截断）

    arXiv:2609.17930v1 Announce Type: new  Abstract: As AI agents take on long, autonomous tasks, we increasingly oversee rather than perform the work, yet we still judge them almost entirely by whether they finally succeed. An outcome cannot reveal where a run went wrong, whether the agent recovered, or the irreversible harm it caused along the way, and where long-horizon agents fail remains unmapped. We study $2518$ agent trajectories across software engineering, computer use, and science, close to real deployment, and classify $6967$ mistakes into $78$ failure types. Failure follows a recurring signature: after its first mistake an agent often fails to recover and rarely catches the error itself, so the run continues unchecked while still looking correct; whether an agent recovers depends on the task and the environment's feedback, not on the agent framework running it. Long-horizon agents can do real harm on the way to a passing result: even runs scored as solved delete data, corrupt s
    
[^147]: 无流形的对称性：轨道上的内在维度

    Symmetry without a manifold: intrinsic dimension on orbits

    [https://arxiv.org/abs/2609.17926](https://arxiv.org/abs/2609.17926)

    该论文证明在对称性轨道（如模加法任务）上标准内在维度估计器普遍失效，神经缩放行为不再遵循幂律，而是遵循关于隐藏层宽度的指数定律 $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$。

    

    神经缩放指数的标准几何推导以数据流形的内在维度作为其输入。对于 $\mathbb{Z}_p$ 上的模加法任务，该推导没有输入可用。其精确的代数解是 $\mathbb{Z}_p$ 通过等距作用产生的轨道。仅凭传递性就使得标准维度估计器所依赖的比率统计量退化为一个点质量，因此该估计器是未定义的，且在此情形下两个最近邻距离恰好完全重合。在尺度 $\epsilon$ 上破坏对称性虽然能返回一个数值，但该数值随 $1/\epsilon$ 变化，不存在无标度平台。我们证明这种失效是普遍性的：在任何由群通过等距作用产生的有限轨道上，估计器报告的只是探测该集合的分辨率，而非维度。取代幂律的是关于隐藏层宽度的指数关系，$L(h)=L_\infty+A\exp(-c\,h^{\alpha})$，其 $R^2$ 达到 0.982 至 0.995，而幂律拟合的 $R^2$ 仅为 0.857 至 0.906。

    arXiv:2609.17926v1 Announce Type: new  Abstract: The standard geometric derivation of neural scaling exponents takes the intrinsic dimension of a data manifold as its input. On modular addition in $\mathbb{Z}_p$ that derivation has no input. The exact algebraic solution is an orbit of $\mathbb{Z}_p$ acting by isometries. Transitivity alone makes the ratio statistic underlying the standard dimension estimator a point mass, so the estimator is undefined, and here the two nearest neighbour distances coincide exactly. Breaking the symmetry at scale $\epsilon$ returns a number, but one that tracks $1/\epsilon$ with no scale free plateau. We show that the failure is general, since on any finite orbit of a group acting by isometries the estimator reports the resolution at which the set is probed rather than a dimension. What replaces the power law is exponential in hidden width, $L(h)=L_\infty+A\exp(-c\,h^{\alpha})$, with $R^2$ between 0.982 and 0.995 against 0.857 to 0.906 for a power law ad
    
[^148]: EdgeReMIND：一种可扩展、排名领先的时序多关系链接预测记忆基线方法

    EdgeReMIND: A Scalable, Top-Ranked Memorization Baseline for Temporal Multi-Relational Link Prediction

    [https://arxiv.org/abs/2609.17916](https://arxiv.org/abs/2609.17916)

    EdgeReMIND 是一种基于数据校准特征和逐关系学习权重的轻量级线性记忆模型，在 TGB 2.0 八个数据集中的六个上取得最高测试 MRR，并成为唯一能在包括三个最大数据集在内的全部数据集上运行的感知关系最先进基线。

    

    时序图基准2.0（TGB 2.0）上的时序链接预测面临可扩展性瓶颈：在该基准的三个最大数据集上，所有现有的嵌入方法要么内存耗尽，要么超出时间预算。这些大规模图最接近真实部署规模，因此在这些图上的失败意味着实际生产中的真实限制。EdgeReMIND 在八个 TGB 2.0 数据集中的六个上创下了已报告的最高测试平均倒数排名（MRR），并且是唯一能够在全部八个数据集上运行的感知关系方法。这种线性记忆模型通过在数据校准特征上学习每个关系的权重，因此它不仅仅是嵌入方法失效时的备选方案，而是整个基准上实用的最先进基线。

    arXiv:2609.17916v1 Announce Type: new  Abstract: Temporal link prediction on the Temporal Graph Benchmark 2.0 (TGB 2.0) faces a scalability ceiling: on the benchmark's three largest datasets, every existing embedding method runs out of memory or exceeds the time budget. These large-scale graphs are the ones nearest real deployment scale, so failing on them is a real production limitation. EdgeReMIND sets the highest reported test mean reciprocal rank (MRR) on six of eight TGB 2.0 datasets and is the only relation-aware method that runs on all of them. This linear memorization model, with learned per-relation weights over data-calibrated features, is therefore not merely a fallback where embeddings fail but a practical state-of-the-art baseline across the benchmark.
    
[^149]: Zing-0.5：迈向具有实时联合动作与文本控制的可玩世界

    Zing-0.5: Toward Playable Worlds with Real-Time Joint Action and Text Control

    [https://arxiv.org/abs/2609.17909](https://arxiv.org/abs/2609.17909)

    Zing-0.5是一个50亿参数自回归世界模型，通过统一动作与文本条件化、事件尺度分布匹配蒸馏监督以及低成本流式推理三项创新，实现了用户可通过键盘和文本实时联合控制的可玩生成世界。

    

    我们推出了Zing-0.5，一个专为可玩性设计的50亿参数自回归世界模型：用户可以探索生成的世界、影响正在展开的事件，并通过键盘和在线文本的联合控制来响应由此产生的反馈。我们的方法汇集了三项技术贡献：（1）统一的动作与文本条件化，将幅度感知的键盘输入与时间对齐的文本指令以及联合标注的视频相结合，在同一序列中学习导航和事件控制；（2）面向增量生成的事件尺度监督，使用在连接式多提示视频上训练的片段级教师模型，通过分布匹配蒸馏来监督块级因果学生模型；（3）低成本实时交互，结合四步生成与上下文保持的流式处理，支持832 x 480分辨率下24 FPS的推理，估计服务器租用成本约为每流分钟0.009美元。

    arXiv:2609.17909v1 Announce Type: cross  Abstract: We introduce Zing-0.5, a 5B autoregressive world model designed for playability: users can explore generated worlds, influence unfolding events, and respond to the resulting feedback through joint keyboard and online text control. Our approach brings together three technical contributions: (1) Unified action and text conditioning, combining magnitude-aware keyboard inputs with temporally aligned text instructions and jointly annotated videos to learn navigation and event control within the same sequence; (2) Event-scale supervision for incremental generation, using a segment-level teacher trained on connected multi-prompt videos to supervise a block-level causal student through distribution-matching distillation; and (3) Low-cost real-time interaction, combining four-step generation with context-preserving streaming to support 832 x 480 inference at 24 FPS at an estimated server rental cost of approximately USD 0.009 per stream-minute.
    
[^150]: 漫步分数流形：习得数据流形上的连续时间生成动力学

    Walking the Score Manifold: Continuous-time Generative Dynamics on Learned Data Manifolds

    [https://arxiv.org/abs/2609.17901](https://arxiv.org/abs/2609.17901)

    该论文提出在习得数据流形上进行连续时间生成建模的新框架，利用预训练分数模型作为几何先验学习向量场，实现任意时间戳的生成与时间超分辨率，并通过促进横向指数稳定性的目标提升长时程推演的鲁棒性。

    

    时间相关数据的生成建模通常在离散时间网格上构建，这使得监督被限制在训练数据中观测到的时间戳上。我们转而将生成框架化为在习得数据流形上的连续时间演化。为此，我们利用预训练的基于分数的模型作为几何先验，并学习一个向量场，使数据沿着分数诱导的插值路径进行演化。由于这些动力学遵循尊重分数模型所学习几何结构的转换，它们支持在任意时间戳进行生成，以及超越训练数据离散化限制的时间超分辨率。此外，这种几何形式使我们能够通过回归目标以无仿真的方式训练向量场。为了提高长时程推演的鲁棒性，我们引入了一个促进路径相对横向指数稳定性的目标。尽管其动机来自稳定性理论，但它提供了一个实用的…

    arXiv:2609.17901v1 Announce Type: cross  Abstract: Generative modeling of time-dependent data is typically formulated on a discrete temporal grid, restricting supervision to the observed timestamps in the training data. We instead frame generation as continuous-time evolution on a learned data manifold. To this end, we leverage pretrained score-based models as geometric priors and learn a vector field that evolves data along score-induced interpolation paths. Because these dynamics follow transitions that respect the geometry learned by the score model, they support generation at arbitrary timestamps and temporal super-resolution beyond the discretization of the training data. Moreover, this geometric formulation allows us to train the vector field simulation-free through a regression objective. To improve long-horizon rollout robustness, we introduce an objective that promotes path-relative transverse exponential stability. While motivated by stability theory, it admits a practical in
    
[^151]: QEMScore：测量数据对学习型量子误差缓解到底有多大贡献？

    QEMScore: How Much Does the Measurement Add to Learned Quantum Error Mitigation?

    [https://arxiv.org/abs/2609.17896](https://arxiv.org/abs/2609.17896)

    该论文提出 QEMScore 评估框架，通过将学习型量子误差缓解器与不读取测量数据的容量匹配对照组进行比较并计入测量开销，揭示了缓解器的收益在很大程度上来自电路结构而非噪声测量数据本身。

    

    arXiv:2609.17896v1 公告类型：cross 摘要：噪声测量数据对学习型量子误差缓解究竟贡献了多少？仅凭准确率表无法回答，因为一个被直接给予电路结构的模型即使完全不读取测量数据也可能获得很高的评分。QEMScore 提供了能够回答这一问题的对比评估。每个模拟电路都带有精确的理想答案。学习型缓解器与一个容量匹配的对照组并列评分——该对照组同样灵活、读取相同的电路描述，但从不读取测量数据。同时，每种方法的测量开销均被如实计入而非被均等化处理。我们在模拟电路上开展了一项受控实验，并基于公开发布的硬件数据重新分析了两种已发表的学习型缓解器 Q-LEAR 和 QRAFT。三个发现尤为突出。首先，在熟悉的族内条件（S0）下、跨两个自旋链模型族和三个随机种子进行评估时，连续耦合能够识别目标，而从不读取测量数据的对照组达到了缓解器性能的 87.7% 至 100.5%……

    arXiv:2609.17896v1 Announce Type: cross  Abstract: How much does the noisy measurement add to learned quantum error mitigation? An accuracy table cannot say, because a model handed circuit structure can score well without reading the measurement at all. QEMScore adds the comparison that can. Each simulated circuit carries an exact ideal answer. The learned mitigator is scored beside a capacity-matched control, a model just as flexible that reads the same circuit description but never the measurement. Each method's measurement spend is accounted and not equalized. We run a controlled campaign on simulated circuits and reanalyze two published learned mitigators, Q-LEAR and QRAFT, from their released hardware data. Three findings stand out. First, under familiar within-family conditions (S0) evaluated across two spin-chain families and three seeds, continuous couplings identify the target, and the control that never reads the measurement matches 87.7 to 100.5 percent of the mitigator's ga
    
[^152]: TabPFN-3.5：技术报告

    TabPFN-3.5: Technical Report

    [https://arxiv.org/abs/2609.17895](https://arxiv.org/abs/2609.17895)

    TabPFN-3.5 是一款新的旗舰表格基础模型，在标准及非独立同分布、多模态、高基数、宽表等实际表格任务上全面超越 TabPFN-3 和现有基线，并提供了速度提升最高 3 倍的 TabPFN-3.5-Fast 和增强多模态能力的 TabPFN-3.5-Plus 变体。

    

    我们推出 TabPFN-3.5，这是我们的全新旗舰表格基础模型。它在广泛的表格任务上显著超越了其前代模型 TabPFN-3 以及所有现有基线。TabPFN-3.5 在 TabArena 的标准表格预测任务上创造了新的最先进水平，并将其扩展到实际从业者会遇到的数据场景：具有时间或分组划分的非独立同分布数据、包含字符串、文本和图像的表格、高基数类别特征，以及具有众多特征的宽表。这些优势延续到我们的任务专用框架中：在关系型数据上达到最先进水平，并具备更强的时间序列预测能力。为了实现更快的推理，我们的变体 TabPFN-3.5-Fast 运行速度最高可达 TabPFN-3 的 3 倍，同时保留了大部分精度提升。此外，我们升级了 TabPFN-3.5-Plus，通过先进的文本和日期处理以及专有推理优化扩展了多模态能力。最后，我们发布了一个新的版本。

    arXiv:2609.17895v1 Announce Type: new  Abstract: We introduce TabPFN-3.5, our new flagship Tabular Foundation Model. It significantly outperforms its predecessor, TabPFN-3, and all existing baselines across a broad range of tabular problems. TabPFN-3.5 sets a new state of the art on standard tabular prediction in TabArena, and extends it to the data practitioners encounter in practice: non-i.i.d. data with temporal or grouped splits, tables with strings, text and images, high-cardinality categorical features, and wide tables with many features. These gains carry over to our task-specific harnesses: state of the art on relational data and stronger time-series forecasting. For faster inference, our variant TabPFN-3.5-Fast runs up to 3x faster than TabPFN-3 while keeping most of the accuracy gains. In addition, we upgrade TabPFN-3.5-Plus, expanding our multimodal capabilities with advanced text and date handling alongside proprietary inference optimizations. Finally, we release a new vers
    
[^153]: 流形假设下聚类的区间不确定性界定

    Bracketing Uncertainty in Clustering Under the Manifold Hypothesis

    [https://arxiv.org/abs/2609.17892](https://arxiv.org/abs/2609.17892)

    该论文通过结合内在流形几何（体积增长与触及半径）和样本级度量（填充距离与密度），为互k近邻图聚类建立了阈值现象，从而界定了聚类结果存在不确定性的几何区间。

    

    流形假设为聚类提供了一个自然的准则：根据每个点所来自的流形分量对数据进行划分。两个分量是否可分取决于一种几何上的权衡：分量之间的环境空间间隔与采样中的最大间隙之间的对比。在实践中，这种权衡很少被明确评估，导致标准方法即使在数据不支持唯一答案的情况下，也会过度承诺单一的聚类分配。我们通过将内在流形几何（体积增长与触及半径）与样本级度量（填充距离与密度）相结合，形式化了这一权衡，从而为互k近邻图建立了一个阈值现象：当偏移-填充比超过一个保守的上阈值时，分量分离得以保持；而低于一个下阈值时，分量会发生融合。这两个阈值之间的间隙定义了一个几何不确定性区域，在该区域中聚类的数量…

    arXiv:2609.17892v1 Announce Type: cross  Abstract: The manifold hypothesis suggests a natural criterion for clustering: partition data according to the manifold component from which each point is drawn. Whether two components are separable depends on a geometric tradeoff: the ambient separation between components versus the largest gap in sampling. In practice, this tradeoff is rarely assessed explicitly, leading standard methods to over-commit to a single clustering assignment even when the data do not support a unique answer. We formalize this tradeoff by combining intrinsic manifold geometry (volume growth and reach) with sample-level quantities (fill distance and density), yielding a threshold phenomenon for mutual-$k$-nearest-neighbor graphs: when the offset-to-fill ratio exceeds a conservative upper threshold, component separation is preserved; below a lower threshold, components fuse. The gap between these thresholds defines a geometric uncertainty zone in which the number of cl
    
[^154]: 使用状态空间模型的长上下文示例选择

    Long-Context Demonstration Selection Using State Space Models

    [https://arxiv.org/abs/2609.17888](https://arxiv.org/abs/2609.17888)

    本文提出一种基于状态空间模型（SSM）的示例选择方法，通过从transformer模型蒸馏出线性的SSM，高效解决长上下文场景下推理成本高企的示例选择难题。

    

    我们研究示例选择问题，即选择一个示例子集并将其前置到语言模型的查询之前。这个问题与上下文学习和语言模型推理密切相关。由于transformer模型的推理成本随序列长度呈二次方增长，因此在长上下文场景中，选择问题变得尤其具有挑战性。在本文中，我们通过基于状态空间模型（SSMs）来解决这一问题，SSMs在给定输入的情况下只需线性的推理时间。我们的方法包括两种算法。第一种算法通过蒸馏（已训练的）transformer模型来学习一小组SSMs：我们将所有层划分为连续的组，然后对每个组，我们估计一个单独的状态空间模型来复制相邻层内的输入输出行为。其次，我们将蒸馏模型的输出映射到一小组token上，并将这些嵌入应用于……

    arXiv:2609.17888v1 Announce Type: cross  Abstract: We study the problem of demonstration selection, which involves selecting a subset of examples for prepending to a query to a language model. This problem is closely related to in-context learning and language model inference. Since the inference cost of a transformer model scales quadratically with sequence length, the selection problem becomes especially challenging in a long-context scenario. In this paper, we tackle this problem by building on state space models (SSMs), which require only linear inference time given the input. Our approach involves two algorithms. The first learns a small set of SSMs through distillation of a (trained) transformer model. We partition all the layers into consecutive groups. Then for each group, we estimate a separate state space model to replicate the input-output behavior within the adjacent layers. Second, we map the distilled model outputs to a small set of tokens, and apply these embeddings for 
    
[^155]: EEG基础模型微调中跨深度聚合与软路由专家的数据集依赖效应

    Dataset-Dependent Effects of Cross-Depth Aggregation and Soft-Routed Experts in EEG Foundation Model Fine-Tuning

    [https://arxiv.org/abs/2609.17886](https://arxiv.org/abs/2609.17886)

    在EEG基础模型CBraMod上添加跨深度注意力残差和软路由专家模块的效果高度依赖具体数据集，有时甚至产生负面影响，且带来2至3倍的运行时间和内存开销，并未带来相对完全微调的一致收益。

    

    EEG解码任务可能依赖不同的时间动态和跨通道关系。我们通过为CBraMod添加跨深度注意力残差和两个软路由专家库，测试专用模块能否改进完全微调的EEG基础模型。在FACED、ISRUC、SEED-V和PhysioNet-MI四个数据集上进行的匹配三随机种子实验中，完整模型相对于完全微调的平均平衡准确率分别变化-0.12、+1.27、+0.77和-1.27个百分点。仅使用AttnRes可在三个数据集上提升平均平衡准确率，而在AttnRes基础上添加专家仅对FACED和SEED-V有帮助。这些收益伴随着显著的开销：AttnRes需要2.11至2.88倍的运行时间和1.78至2.67倍的内存，而完整模型需要2.41至3.04倍的运行时间和1.86至2.85倍的内存。总体而言，所添加的模块产生依赖数据集的、有时甚至相反的效果，而非相对完全微调的一致收益。

    arXiv:2609.17886v1 Announce Type: new  Abstract: EEG decoding tasks can rely on different temporal dynamics and cross-channel relationships. We test whether specialized modules improve a fully fine-tuned EEG foundation model by augmenting CBraMod with cross-depth Attention Residuals (AttnRes) and two soft-routed expert banks. Across matched three-seed experiments on FACED, ISRUC, SEED-V, and PhysioNet-MI, the complete model changes mean balanced accuracy relative to full fine-tuning by -0.12, +1.27, +0.77, and -1.27 points, respectively. AttnRes alone improves mean balanced accuracy on three datasets, whereas adding experts on top of AttnRes helps only FACED and SEED-V. These gains come with substantial overhead: AttnRes requires 2.11 to 2.88x runtime and 1.78 to 2.67x memory, while the complete model requires 2.41 to 3.04x runtime and 1.86 to 2.85x memory. Overall, the added modules produce dataset-dependent, sometimes opposing effects rather than consistent gains over full fine-tunin
    
[^156]: 不可承受之重：无人机音频分类的模型与方法扩展研究

    The Unbearable Weight: Scaling Models and Methods for UAV Audio Classification

    [https://arxiv.org/abs/2609.17884](https://arxiv.org/abs/2609.17884)

    本文在涵盖31个无人机类别的音频数据集上，系统比较了多种Transformer与卷积骨干网络在全量微调、仅分类器微调及参数高效微调等方法下的表现，揭示了资源受限的无人机部署场景中“重量级”全量微调何时必要、轻量级方案何时更优。

    

    随着无人机（UAV）在消费级和国防领域日益普及，如何从有限的、特定模态的数据中对其进行可靠分类已成为一项紧迫的挑战。目前的主流方法是在任务数据上对大型预训练网络进行全量微调，但这会带来巨大的计算与内存负担，在资源受限的无人机部署场景中难以承受，因为此类场景既需要边缘端推理，又需要针对新兴机型进行快速再训练。本文针对无人机音频分类任务，对模型架构和微调方法进行了系统的规模化实验，探究这种“重负”何时是合理的、何时轻量级替代方案更占优势。基于一个涵盖31个无人机类别、共3100个音频片段的自定义数据集，我们评估了Transformer架构（ViT、AST）和卷积架构（自定义CNN、ResNet-18/152、MobileNet-V3-S/L、EfficientNet-B0/B7）等骨干网络在全量微调、仅分类器微调以及四种参数高效微调方法下的表现。

    arXiv:2609.17884v1 Announce Type: cross  Abstract: As unmanned aerial vehicles (UAVs) become increasingly prevalent in consumer and defense settings, classifying them reliably from limited, modality-specific data is an urgent challenge. The dominant approach, large pretrained networks fully fine-tuned on task data, carries a substantial computational and memory weight that is hard to bear in resource-constrained UAV deployments, where edge inference and rapid retraining for emerging platforms are both required. This paper systematically scales across both model architectures and fine-tuning methods for UAV audio classification, asking when that weight is justified and when lighter alternatives prevail. Using a custom dataset of 3,100 audio clips spanning 31 drone classes, we evaluate transformer (ViT, AST) and convolutional (custom CNN, ResNet-18/152, MobileNet-V3-S/L, EfficientNet-B0/B7) backbones under full fine-tuning, classifier-only fine-tuning, and four parameter-efficient fine-t
    
[^157]: 视觉语言模型能否从行人视角图像中可靠地评估人行道无障碍属性？

    Can VLMs Reliably Assess Sidewalk Accessibility Attributes from Pedestrian-Level Imagery?

    [https://arxiv.org/abs/2609.17882](https://arxiv.org/abs/2609.17882)

    该研究首次将保形预测应用于基于视觉语言模型的人行道无障碍属性评估，利用首尔514张实地测量图像验证了四个VLM均可达到90%的名义覆盖率，其中有效宽度的估计最具信息量，且非对称校准可在覆盖率不变的前提下将预测区间缩短多达33%。

    

    城市无障碍性的一个重要组成部分，特别是对于轮椅使用者和行动不便者而言，是人行道是否符合可测量的规范要求。我们测试了有效宽度、纵向坡度、横向坡度和路面状况能否通过视觉语言模型（VLM）从行人视角图像中可靠地评估。我们首次将基于采样的保形预测（CP）应用于基于VLM的无障碍性评估。我们在514张来自韩国首尔的人行道图像上评估了四个VLM模型，这些图像均具有实地测量的真值。保形校准在所有模型和属性上都达到了名义上的90%覆盖率，但校准区域的信息量各不相同。有效宽度产生了最具信息量的估计，最佳模型的平均区间半宽约为1.0米。由于每个模型都高估了宽度，非对称校准在保持覆盖率不变的情况下将区间最多缩短了33%。

    arXiv:2609.17882v1 Announce Type: cross  Abstract: An important component of urban accessibility, particularly for wheelchair users and people with reduced mobility, is sidewalk compliance with measurable requirements. We test whether effective width, longitudinal slope, cross slope, and pavement condition can be assessed reliably from pedestrian-level imagery using vision-language models (VLMs). We present the first application of sampling-based conformal prediction (CP) for VLM-based accessibility assessment. We evaluate four VLMs on 514 sidewalk images from Seoul, South Korea, with field-measured ground truth. Conformal calibration attains the nominal 90% coverage for all models and attributes, but the calibrated regions differ in informativeness. Effective width yields the most informative estimates, with a mean interval half-width of about 1.0 m for the best model. Since every model overestimates width, asymmetric calibration shortens the intervals by up to 33% at unchanged covera
    
[^158]: 不确定性感知的持续学习：面向演化标签空间下的开放世界意图发现

    Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an evolving Label Space

    [https://arxiv.org/abs/2609.17866](https://arxiv.org/abs/2609.17866)

    该论文提出了一种统一的不确定性感知概率框架，通过自适应β-VAE编码、分类器置信度-后验不确定性-DP-GMM似然的多信号决策机制和基于密度的聚类发现，在演化的标签空间下实现开放世界新意图的持续发现与可控标签空间扩展，并结合回放与弹性权重巩固来缓解灾难性遗忘。

    

    现实世界的智能系统越来越多地在开放世界条件下运行，其中用户意图并非固定不变，也无法事先穷尽获知，且会随着新的交互模式的出现而不断演化。本文提出了一个统一的不确定性感知概率框架，用于在演化的标签空间下进行持续的新意图发现。每条话语通过自适应β-VAE被编码为潜在均值（用于分类和密度建模），以及作为全局可靠性信号的后验不确定性估计。分类器置信度、后验不确定性和DP-GMM似然通过多信号决策机制相结合，以区分已知意图与潜在的新颖样本。候选的新颖实例通过基于密度的发现模块进行聚类，只有可靠的簇才会被提升为新标签，从而实现可控的标签空间扩展。回放机制与弹性权重巩固（EWC）用于缓解灾难性遗忘。

    arXiv:2609.17866v1 Announce Type: cross  Abstract: Real-world intelligent systems increasingly operate under open-world conditions, where user intents are not fixed or exhaustively known a priori and may evolve as new interaction patterns emerge. This paper proposes a unified uncertainty-aware probabilistic framework for continual new intent discovery under an evolving label space. Each utterance is encoded through an adaptive $\beta$-VAE into a latent mean, used for classification and density modelling and a posterior uncertainty estimate acting as a global reliability signal. Classifier confidence, posterior uncertainty and DP-GMM likelihood are combined through a multi-signal decision mechanism to distinguish known intents from potentially novel samples. Candidate novel instances are clustered through a density-based discovery module and only reliable clusters are promoted to new labels, enabling controlled label-space expansion. Replay and Elastic Weight Consolidation mitigate cata
    
[^159]: 谁来评判很重要：测量大语言模型评审组中基于模型家族的条件偏好

    Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels

    [https://arxiv.org/abs/2609.17857](https://arxiv.org/abs/2609.17857)

    该研究首次系统测量了大语言模型评审中的“同家族偏好”效应——即模型评审会偏袒同一家族的候选模型——通过提出一种固定候选家族的校正估计器，发现四个主流开放权重模型家族均存在3.4-8.4个百分点的显著同家族提升，且该效应与评审侧似然度密切相关。

    

    评判者的身份会影响LLM-as-judge（大语言模型作为评审）的结果，但在不与候选模型质量相混淆的情况下测量这种影响十分困难。我们在完全交叉的成对设计中研究了四个开放权重模型家族（Llama 3.1、Qwen 2.5、Gemma 2和Yi 1.5），共进行了9,312次评判。常见的按家族统计量与候选模型质量存在强烈混淆，其与Bradley-Terry能力的相关系数高达r = 0.95。为此，我们推导出一个校正估计器，该估计器在固定候选家族的前提下比较不同评审者。校正后，所有四个家族均显示出正向的同家族提升（3.4-8.4个百分点），全局FPS为0.067（95%置信区间[0.053, 0.084]，置换检验p = 0.0002）。该效应在基于评审组的质量控制、独立的人类共识锚点以及float16评判复现下依然稳健存在。评审侧的似然度与该效应密切相关：加入似然优势项会使受控系数降低61%，我们将其视为描述性的衰减。

    arXiv:2609.17857v1 Announce Type: cross  Abstract: Who the judge is can affect an LLM-as-judge result, but measuring that effect without confusing it with candidate quality is difficult. We study four open-weight families (Llama 3.1, Qwen 2.5, Gemma 2, and Yi 1.5) in a fully crossed pairwise design with 9,312 judgments. A common per-family statistic is strongly confounded with candidate quality and correlates with Bradley-Terry ability at r = 0.95. We derive a corrected estimator that holds the candidate family fixed and compares judges. All four families then show a positive same-family lift (3.4-8.4 percentage points), with global FPS 0.067 (95% CI [0.053, 0.084], permutation p = 0.0002). The effect remains under panel-based quality controls, an independent human-consensus anchor, and a float16 judging replication. Judge-side likelihood is closely related to the effect: adding likelihood advantage reduces the controlled coefficient by 61%, which we treat as descriptive attenuation ra
    
[^160]: AfriSyCo：测量非洲语言内容中的断言式框架、验证与措辞敏感性

    AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content

    [https://arxiv.org/abs/2609.17853](https://arxiv.org/abs/2609.17853)

    该论文提出AfriSyCo框架，通过母语后续提问与跨语言2×2因子实验系统测量非洲语言事实内容中模型答案切换行为，发现断言式框架会显著增加错误目标选择（+30.4个百分点），而验证机制可有效降低该效应（-17.4个百分点）。

    

    AfriSyCo 通过两个互补层次研究围绕非洲语言事实性内容的答案切换现象：母语后续提问和受控的跨语言因子实验，其中问题、选项和目标保持非洲语言，而后续提问的框架采用英语。我们分析了源自100个源问题、涵盖七个开源权重模型检查点和六种语言的1,415个首轮正确观测数据；首轮正确指的是观察到的首次回答准确性，而非已证明的知识。在母语提示下，断言式背书相比提及加验证（M+V）方式，在任意轮次中导致的错误目标选择多出29.3个百分点，即时T2对比为19.0个百分点。在预先承诺的2×2因子实验中，对三种测试提示家族取平均，断言式框架使目标选择增加30.4个百分点（95%置信区间[28.4, 32.3]）；验证使其降低17.4个百分点，而断言效应随……（摘要此处截断）

    arXiv:2609.17853v1 Announce Type: cross  Abstract: AfriSyCo studies answer switching around African-language factual content with two complementary layers: native-language follow-ups and a controlled cross-language factorial whose question, options, and target remain in the African language while the follow-up framing is English. We analyze 1,415 turn-1-correct model-language-item observations derived from 100 source questions across seven open-weight checkpoints and six languages; turn-1-correct denotes observed first-response accuracy, not demonstrated knowledge. Under native prompts, assertive endorsement produces 29.3 percentage points more any-turn false-target selection than mention-plus-verification (M+V), with a 19.0-point immediate T2 contrast. In the precommitted 2 x 2 factorial, averaged over three tested prompt families, assertive framing increases target selection by 30.4 points (95% CI [28.4, 32.3]); verification decreases it by 17.4 points, while the assertive effect ris
    
[^161]: 学习异质性偏好

    Learning Heterogeneous Preferences

    [https://arxiv.org/abs/2609.17847](https://arxiv.org/abs/2609.17847)

    该论文基于理性选择理论，提出一种多阶段架构，通过引入同时以个体及其决策情境为条件的“个体化效用函数”，从多模态数据中学习异质的主观偏好，突破了传统方法假设群体共享统一效用函数的局限。

    

    从人类反馈中学习已成为训练现代人工智能系统的核心范式，其中人类效用模型被用作策略学习中的奖励模型。现有方法通常假设群体共享一个统一的效用函数，并将标注者之间的分歧视为随机变异。虽然这种假设适用于客观任务，但在偏好因个体而系统性变化的主观领域中，该假设便不再成立。我们研究了主观偏好学习问题，即观察到的选择源于异质但内部一致的效用函数。借鉴理性选择理论（RCT），我们引入了同时以个体及其决策情境为条件的“个体化效用函数”，并提出了一种新颖的多阶段架构，用于从多模态数据中估计这些效用函数。我们在新收集的数据集上评估了该框架。

    arXiv:2609.17847v1 Announce Type: new  Abstract: Learning from human feedback has become a central paradigm for training modern AI systems, where models of human utility are used as reward models in policy learning. Existing methods typically assume a \emph{universal utility} function shared across a population and treat disagreement between annotators as stochastic variation. While suitable for objective tasks, this assumption breaks down in subjective domains where preferences vary systematically across individuals. We study the problem of subjective preference learning, in which observed choices arise from heterogeneous but internally consistent utility functions. Drawing upon rational choice theory, RCT \parencite{tversky1981framing}, we introduce \emph{individuated utility} functions conditioned on both the individual and their decision context, and propose a novel multi-stage architecture for estimating them from multi-modal data. We evaluate our framework on a newly collected da
    
[^162]: PrimeScientist：自主研究中的研究努力战略分配

    PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research

    [https://arxiv.org/abs/2609.17846](https://arxiv.org/abs/2609.17846)

    提出了PrimeScientist框架，将自主研究智能体的研究方向选择与资源投入决策统一建模为序贯决策问题，通过可执行计划树保留竞争性方案及其结果，并利用剩余资源显式引导研究策略，实现研究努力的战略性分配。

    

    自主研究智能体旨在自动化科学工作流程，从提出想法、开展实验到分析结果。然而，当前的人工智能和研究智能体所能提出的研究方向，往往多于其可用资源所能支撑的数量。此外，每一次尝试都可能消耗大量资源，这就要求智能体重新考量后续研究中的投入方式。因此，如何战略性地分配研究努力应当成为自主研究智能体的一项核心能力。为此，我们提出了PrimeScientist，它在连续多次研究尝试中联合确定研究方向与资源投入。具体而言，我们将这一战略性研究努力分配的挑战形式化为一个序贯决策问题，其中剩余资源应当显式地引导研究策略。我们首先引入了一种可执行计划树，用于在多次尝试中保存相互竞争的计划及其执行结果。

    arXiv:2609.17846v1 Announce Type: cross  Abstract: Autonomous research agents aim to automate scientific workflows, from proposing ideas to conducting experiments and analyzing results. Yet current AI and research agents can propose more directions than available resources allow them to pursue. Moreover, each attempt could consume substantial resources, requiring agents to reconsider how to invest in subsequent research. Thus, deciding how to invest research effort strategically should be a defining capability of autonomous research agents. Accordingly, we introduce PrimeScientist, which jointly determines research direction and resource investment across successive research attempts. Specifically, we formulate this challenge of strategic research effort allocation as a sequential decision problem where remaining resources should explicitly guide the research policy. We first introduce an executable plan tree that preserves competing plans and their outcomes across attempts. Building o
    
[^163]: 可实现情形下支持向量机的尖锐间隔泛化界

    Sharp margin-based generalization bounds for realizable SVM

    [https://arxiv.org/abs/2609.17845](https://arxiv.org/abs/2609.17845)

    该论文通过确定性删除问题的分析，证明了可实现情形下硬间隔支持向量机的泛化风险以概率至少\(1-\delta\)不超过\(\frac{C}{m}(K_m+\log\frac{1}{\delta})\)，其中\(K_m=r_m^2/\gamma_m^2\)为半径-间隔复杂度，得到了阶为\(1/m\)且依赖半径-间隔复杂度的尖锐泛化界。

    

    设精确的齐次硬间隔支持向量机在实希尔伯特空间上，由某个Borel概率分布产生的m个独立观测样本进行训练。我们证明，当得分零被计为错误时，存在一个通用数值常数C，使得 \[ \Pp\left( \gamma_m>0,\quad \Risk(u_m)> \frac{C}{m} \left( K_m+\log\frac1\delta \right) \right) \le \delta. \] 其中\(\gamma_m\)是经验齐次间隔，\(u_m\)是精确的最小范数单位间隔分离器，\(r_m\)是最大训练半径，且在\(\{\gamma_m>0\}\)上\(K_m:=r_m^2\norm{u_m}^2=r_m^2/\gamma_m^2\)。证明由一个确定性的删除问题驱动：给定单位球中的向量\(x_1,\ldots,x_n\)，删除一个约束集合\(B\)，令\(u_B\)为满足所有保留单位间隔约束的、离原点最近的点。假设\(\norm{u_B}^2\le k\)，且每个被删除的向量都具有非正……（摘要在此截断）

    arXiv:2609.17845v1 Announce Type: cross  Abstract: Let the exact homogeneous hard-margin support vector machine be trained on \(m\) independent observations from a Borel probability law on a real Hilbert space. We prove that, with score zero counted as an error, there is a universal numerical constant \(C\) such that \[   \Pp\left(   \gamma_m>0,\quad   \Risk(u_m)>   \frac{C}{m}   \left(   K_m+\log\frac1\delta   \right)   \right)   \le \delta . \] Here \(\gamma_m\) is the empirical homogeneous margin, \(u_m\) is the exact minimum-norm unit-margin separator, \(r_m\) is the largest training radius, and \(K_m:=r_m^2\norm{u_m}^2=r_m^2/\gamma_m^2\) on \(\{\gamma_m>0\}\).   The proof is driven by a deterministic deletion problem. Given vectors \(x_1,\ldots,x_n\) in the unit ball, delete a set \(B\) of constraints and let \(u_B\) be the closest point to the origin that satisfies every retained unit-margin constraint. Suppose that \(\norm{u_B}^2\le k\) and that every deleted vector has nonposit
    
[^164]: RoboVAD：一个面向机械臂操作视频异常检测的大规模跨领域评估基准

    RoboVAD: A Large Cross-Domain Evaluation Benchmark for Anomaly Detection in Robotic Arm Manipulation Videos

    [https://arxiv.org/abs/2609.17843](https://arxiv.org/abs/2609.17843)

    本文提出了RoboVAD，一个面向机械臂操作视频异常检测的大规模跨领域评估基准，通过在训练中保留未见过的动作和异常类型来构建贴近现实、更具挑战性的评估场景。

    

    视频异常检测是一项被积极研究的任务，在公共监控和道路交通安全等典型场景中具有广泛的应用。该任务同样与机械臂交互相关，并具有多种下游应用，包括学习更好的交互与操作能力、在异常发生时触发恢复程序等。尽管其重要性显著，但针对机械臂操作视频中异常检测的探索一直受限于可用资源的匮乏。为此，我们提出了RoboVAD，一个大规模的视频异常检测基准，其中包含具有挑战性的跨领域评估场景：某些动作（机械臂执行的任务）和异常类型（执行某些任务时发生的错误）在训练阶段是未见过的。RoboVAD旨在现实场景中对VAD方法进行基准评估，即机械臂可能执行不可预见的情况。

    arXiv:2609.17843v1 Announce Type: cross  Abstract: Video anomaly detection (VAD) is an actively studied task, having wide applications in typical scenarios such as public surveillance and road traffic safety. The task is also relevant for robotic arm interactions, where it has several downstream applications, including learning better interaction and manipulation abilities, triggering recovery procedures when anomalies occur, etc. Despite its relevance, the exploration of anomaly detection in robotic arm manipulation videos is limited by the low number of available resources. To this end, we introduce RoboVAD, a large-scale benchmark for video anomaly detection that comprises challenging cross-domain evaluation scenarios, where certain actions (tasks executed by a robotic arm) and anomaly types (mistakes that occur while performing certain tasks) remain unseen during training. RoboVAD is designed to benchmark VAD methods in realistic scenarios, where robotic arms can perform unforeseen
    
[^165]: 基于数值信息神经网络与重叠Schwarz交替方法的混合耦合

    Hybrid coupling with numerics-informed neural networks and the overlapping Schwarz alternating method

    [https://arxiv.org/abs/2609.17841](https://arxiv.org/abs/2609.17841)

    本文提出了一种利用重叠Schwarz交替方法将预训练的数值信息神经网络（NINN）与经典全阶模型（FOM）耦合的混合建模框架，并证明与PINN不同，NINN无需区域分解即可被准确训练。

    

    我们开发了一个混合建模框架，使用重叠Schwarz交替方法将预训练的数值信息神经网络（NINNs）与经典全阶模型（FOMs）进行耦合。我们考虑对流主导、Peclet数为10^6情形下的二维对流扩散方程。我们首先证明，与相应的物理信息神经网络（PINN）不同，整体式NINN在我们的模型问题上无需区域分解即可被准确训练。然后，我们采用重叠乘性Schwarz方法作为部署机制，将预训练的子区域局部NINN与相邻FOM耦合，并在整个Schwarz迭代过程中保持NINN权重固定。我们考虑了子区域局部NINN的两种训练方法：一种是自顶向下的方法，其边界数据来自在整个区域上以每个子区域放置FOM所进行的耦合Schwarz求解（FOM-FOM Schwarz）；另一种是自底向上的方法

    arXiv:2609.17841v1 Announce Type: new  Abstract: We develop a hybrid modeling framework for coupling pre-trained numerics-informed neural networks (NINNs) with classical full order models (FOMs) using the overlapping Schwarz alternating method. We consider the two-dimensional advection-diffusion equation in the advection-dominated, Peclet-number 10^6 regime. We first demonstrate that, unlike the corresponding physics-informed neural network (PINN), a monolithic NINN can be accurately trained on our model problem without domain decomposition. We then employ overlapping multiplicative Schwarz as a deployment mechanism for coupling a pre-trained, subdomain-local NINN with a neighboring FOM, with the NINN weights held fixed throughout the Schwarz iteration. We consider two training approaches for the subdomain-local NINNs: a top-down approach, in which boundary data are obtained from a coupled Schwarz solve on the full domain with a FOM on each subdomain (FOM-FOM Schwarz), and a bottom-up 
    
[^166]: 用人工智能学习核结构：半径与集体性

    Learning Nuclear Structure with AI: Radii and Collectivity

    [https://arxiv.org/abs/2609.17838](https://arxiv.org/abs/2609.17838)

    该论文开发了基于NuCLR多任务核数据模型的留出集成方法，证明共享表示学习显著提升电荷半径和B(E2)跃迁强度的预测精度，达到与最先进核模型相当的水平，并为实验设计提供数据驱动的指导。

    

    低能核结构的信息编码在覆盖整个核素图的广泛实验数据之中。学习这些信息在不同可观测量和原子核之间如何组织，可以为理论外推和实验设计提供数据驱动的经验基线。在此，我们基于NuCLR（核协同学习表示）——一个核数据的多任务模型——开发了留出集成方法，用于研究电荷半径和电四极跃迁强度。折叠外（OOF）验证表明，共享表示学习优于单任务学习，在数百个核素上实现了0.0147 fm的电荷半径均方根偏差和0.192 e²b²的B(E2)均方根偏差，可与最先进的核模型相媲美。我们的误差棒估计了整个核素图上的预期预测精度，并突出了新数据将能编码重要信息的区域。

    arXiv:2609.17838v1 Announce Type: cross  Abstract: Low-energy nuclear structure is encoded in a broad body of experimental information across the chart of nuclides. Learning how this information is organized across observables and nuclei can provide a data-driven empirical baseline for theoretical extrapolations and experimental design. Here, we develop held-out ensembles based on NuCLR (Nuclear Co-Learned Representations), a multi-task model of nuclear data, to study charge radii and electric-quadrupole transition strengths. Out-of-fold (OOF) validation shows that shared representation improves performance over single-task learning, yielding a charge-radius $\mathrm{RMS}$ deviation of $0.0147~{\rm fm}$ and a $\mathrm{B(E2)}$ $\mathrm{RMS}$ deviation of $0.192~e^2{\rm b}^2$ across hundreds of nuclides, competitive with state-of-the-art nuclear models. Our error bars estimate the expected prediction accuracy across the nuclear chart, highlighting regions where new data would encode info
    
[^167]: 基于算子推断、重叠Schwarz交替法与强化学习的自适应混合耦合方法

    Adaptive hybrid coupling with operator inference, the overlapping Schwarz alternating method and reinforcement learning

    [https://arxiv.org/abs/2609.17837](https://arxiv.org/abs/2609.17837)

    本文提出一种基于强化学习的方法，利用深度Q网络在重叠Schwarz交替法框架下在线自适应地在子域局部全阶模型与预训练算子推断降阶模型之间进行切换，从而应对瞬态问题中需要高保真分辨的区域随时间变化的挑战。

    

    混合域分解方法为耦合全阶模型（FOM）和降阶模型（ROM）提供了灵活的框架，但通常假设分配给每个子域的模型在整个仿真过程中保持固定。这对于局部特征在区域内传播、需要高保真度分辨的区域随时间变化的瞬态问题来说是具有局限性的。我们提出了一种基于强化学习（RL）的方法，用于通过重叠Schwarz交替法（O-SAM）耦合的FOM-ROM模型的在线自适应，O-SAM是一种迭代域分解方法，它求解子域局部问题，同时通过重叠界面上的传输边界条件交换解信息。深度Q网络（DQN）在离线阶段进行训练，以在子域局部FOM和预训练的算子推断（OpInf）ROM之间进行选择，其奖励函数平衡了精度、计算成本和模型切换频率。一旦训练完成……

    arXiv:2609.17837v1 Announce Type: cross  Abstract: Hybrid domain decomposition methods provide a flexible framework for coupling full order models (FOMs) and reduced order models (ROMs), but typically assume the model assigned to each subdomain is fixed throughout a simulation. This is limiting for transient problems in which localized features propagate through the domain and the regions requiring high-fidelity resolution change over time. We introduce a reinforcement learning (RL)-based approach for online adaptation of FOM-ROM models coupled via the overlapping Schwarz alternating method (O-SAM), an iterative domain decomposition method that solves subdomain-local problems while exchanging solution information through transmission boundary conditions on overlapping interfaces. Deep Q-networks (DQNs) are trained offline to select among subdomain-local FOMs and pre-trained Operator Inference (OpInf) ROMs using a reward balancing accuracy, cost, and model-switching frequency. Once trai
    
[^168]: 面向分子性质预测的程序化预训练

    Procedural Pretraining for Molecular Property Prediction

    [https://arxiv.org/abs/2609.17831](https://arxiv.org/abs/2609.17831)

    该论文提出“程序化预训练→分子预训练→下游微调”的三阶段训练流程，证明在接触分子数据之前，先从序列结构、元胞自动机、图推理等程序化生成的抽象任务中学习归纳偏置，能够显著提升分子性质预测性能（如在Lipophilicity数据集上降低4.8%的测试误差）。

    

    分子性质预测通常受限于下游有标注数据集规模较小的问题，这促使研究者在大量未标注分子语料上进行预训练。在这项工作中，我们探讨是否可以在模型接触任何分子数据之前，从抽象的程序化生成数据中学习有用的归纳偏置。我们提出了一个三阶段训练流程，包括程序化预训练、基于SMILES的分子预训练以及下游微调，并评估了涵盖序列结构、元胞自动机和图推理的多种程序化任务。我们发现，即使在随后的分子预训练之后，程序化预训练仍能改善分子性质预测：在Lipophilicity（亲脂性）数据集上，REVERSE任务将测试误差降低了4.8%。作为对比参考，这一改进的幅度大约相当于我们基于25万分子训练的基线模型与公开发布的、在约100万分子上预训练的MoLFormer检查点之间性能差距的90%。

    arXiv:2609.17831v1 Announce Type: cross  Abstract: Molecular property prediction is often limited by the small size of labeled downstream datasets, motivating pretraining on large corpora of unlabeled molecules. In this work, we ask whether useful inductive biases can instead be learned from abstract, procedurally generated data before a model sees any molecular data. We introduce a three-stage training pipeline consisting of procedural pretraining, molecular pretraining on SMILES, and downstream fine-tuning, and evaluate several procedural tasks spanning sequence structure, cellular automata, and graph reasoning. We find that procedural pretraining can improve molecular property prediction even after subsequent molecular pretraining: on Lipophilicity, \textsc{Reverse} reduces test error by 4.8\%. For context, the magnitude of this improvement is roughly 90\% of the performance difference between our 250K-molecule baseline and the publicly released MoLFormer checkpoint pretrained on ap
    
[^169]: NObSP：通过斜子空间投影实现神经网络的功能分解

    NObSP: Functional Decomposition of Neural Networks via Oblique Subspace Projections

    [https://arxiv.org/abs/2609.17825](https://arxiv.org/abs/2609.17825)

    本文提出NObSP框架，利用斜子空间投影将神经网络预测分解为显式的逐特征贡献函数与交互残差，实现局部解释与全局功能分析，并可在卷积网络中免反向传播生成类别激活图。

    

    理解深度神经网络如何做出决策仍然是一个根本性的挑战。我们提出了NObSP（非线性斜子空间投影），这是一个将预测分解为显式的逐特征贡献函数和交互残差的框架。NObSP利用训练好的网络的线性最后一层，并在样本空间中使用斜投影来减少当学习到的特征子空间重叠时的重复计算，从而同时支持局部解释和全局功能分析。我们建立了与函数方差分析（functional ANOVA）和Kolmogorov-Arnold表示定理之间的联系，并推导出一种用于样本外评估的高效偏回归算法。对于卷积神经网络，NObSP-CAM在进行一次性校准后，无需反向传播即可生成类别激活图。在表格数据和视觉基准上的实验表明，其忠实度与现有的归因方法相当。在一个包含已知交互结构的合成基准上（摘要在此处截断）……

    arXiv:2609.17825v1 Announce Type: new  Abstract: Understanding how deep neural networks make decisions remains a fundamental challenge. We present NObSP (Nonlinear Oblique Subspace Projections), a framework that decomposes predictions into explicit per feature contribution functions and an interaction residual. NObSP exploits the linear final layer of a trained network and uses oblique projections in sample space to reduce double counting when learned feature subspaces overlap, thereby supporting both local explanations and global functional analysis. We establish connections to functional ANOVA and the Kolmogorov-Arnold representation theorem and derive an efficient partial regression algorithm for out of sample evaluation. For convolutional networks, NObSP-CAM produces class activation maps without backward passes after a one time calibration. Experiments on tabular and vision benchmarks show faithfulness comparable to established attribution methods. On a synthetic benchmark with kn
    
[^170]: METALICA：元动力学与副本交换实现增强扩散采样

    METALICA: METAdynamics and repLICA exchange for enhanced diffusion sampling

    [https://arxiv.org/abs/2609.17823](https://arxiv.org/abs/2609.17823)

    METALICA通过副本交换机制在预训练扩散模型上实现元动力学，利用偏置势采样和重加权高效探索蛋白质构象的稀有状态，从而实现对稀有事件的有效发现。

    

    许多蛋白质通过构象状态之间的转换来发挥功能，然而基于平衡系综训练的扩散模型很少能采样到稀有状态，因此需要更好的采样方法。我们提出了METALICA，该方法通过副本交换在预训练扩散模型上实现元动力学。它沿着集体变量累积偏置势，通过偏置采样使新样本远离已有样本，并将样本重新加权到无偏分布上。METALICA在每个扩散层级保持一个副本，形成一条通过副本间通信演化的马尔可夫链，并随着偏势的增长而就地优化。METALICA是序贯控制的对偶形式——后者由序贯蒙特卡洛在一批粒子上并行化采样器；而METALICA则是在扩散时间调度的各层级上进行并行，这使其能够从长链中生成样本，这对稀有事件的发现至关重要，并具有加速效果。

    arXiv:2609.17823v1 Announce Type: cross  Abstract: Many proteins function through transitions between conformational states, yet rare states are rarely sampled by diffusion models trained on an equilibrium ensemble, demanding better sampling methods. We introduce METALICA, which implements Metadynamics on a pretrained diffusion model via Replica Exchange. It accumulates a bias potential along a Collective Variable, repels new samples from previous ones through biased sampling, and reweights samples onto the unbiased distribution. METALICA holds one replica per diffusion level, forming a Markov Chain that evolves through inter-replica communication and is refined in place as the bias grows. METALICA is the dual of sequential control, in which Sequential Monte Carlo parallelizes the sampler over a batch of particles. Parallelism over the levels of the diffusion-time schedule instead allows METALICA to generate samples from long chains, essential for the discovery of rare events, with acc
    
[^171]: 自由推断维度：假设混合下零碰撞导航的复杂度度量

    The Free Inference Dimension: Complexity Measure for Zero-Collision Navigation under Hypothesis Mixtures

    [https://arxiv.org/abs/2609.17816](https://arxiv.org/abs/2609.17816)

    本文提出了“自由推断维度”这一新的组合复杂度度量，用以刻画价值混合代理在无需识别真实环境的条件下实现零碰撞导航所能处理的环境复杂度，并证明该维度严格小于VC维、且与Natarajan维仅相差一个路径长度因子。

    

    所罗门诺夫归纳将预测问题框架化为可计算假设上的混合，通常能够识别出真实环境。在我们之前的工作中，在一个具有嵌套约束族的有限元强化学习设置中，我们观察到一种不同的机制：一个价值混合代理在无需识别真实环境的情况下实现了近乎最优的零碰撞导航，我们将这一现象称为“自由推断”。这一机制可以持续到一个尖锐的密度阈值，超过该阈值后性能开始下降，此时后验模式选择（PMS）变得更为可取。我们通过自由推断维度 dFI(S,N) 将这一行为形式化，这是一个组合度量，衡量价值混合代理在保持轨迹连贯性的同时所能处理的环境复杂度。我们证明 dFI 严格小于 VC 维，并通过一个路径长度因子与 Natarajan 维相关联，从而刻画了不可分解损失的代价。一个 PAC 风格的松弛……（摘要不完整）

    arXiv:2609.17816v1 Announce Type: cross  Abstract: Solomonoff induction frames prediction as a mixture over computable hypotheses, typically leading to identification of the true environment. In a finite meta-reinforcement learning setting with nested constraint families, in our previous work, we observe a different regime: a value-mixture (VM) agent achieves near-optimal, zero-collision navigation without identifying the true environment, a phenomenon we call Free Inference. This regime persists up to a sharp density threshold, beyond which performance degrades and posterior-mode selection (PMS) becomes preferable.   We formalize this behavior via the Free Inference dimension dFI(S,N), a combinatorial measure of the environmental complexity a VM agent can handle while preserving trajectory coherence. We prove dFI is strictly smaller than the VC-dimension and relates to the Natarajan dimension up to a path-length factor, capturing the cost of non-decomposable loss. A PAC-style relaxati
    
[^172]: 基于卡尔曼推断的原理性Koopman表示实现高效时间序列预测

    Principled Koopman Representations with Kalman Inference for Efficient Time-Series Prediction

    [https://arxiv.org/abs/2609.17815](https://arxiv.org/abs/2609.17815)

    提出K²SVD方法，通过优化Hilbert-Schmidt目标显式学习Koopman算子的主要奇异函数，构建数学上自洽的紧凑低秩Koopman空间，并结合卡尔曼滤波实现高效且可解释的时间序列预测。

    

    Koopman算子已被广泛应用于动力系统的时间序列预测。然而，先前利用神经网络学习潜在“Koopman空间”的工作往往未能为预测构建有效的Koopman空间，因为这些表示在数学上可能与算子理论的表述不一致，并且无法捕捉系统动力学的内在低秩结构。为了解决这一问题，我们提出了K²SVD，该方法通过优化Hilbert-Schmidt目标函数，显式地学习Koopman算子的主要奇异函数。这产生了一个具有可解释线性组合的、定义明确的Koopman算子低秩近似，其潜在空间十分紧凑，维度不足先前工作所使用维度的10%。在学到的Koopman空间中，K²SVD进一步利用线性高斯状态空间模型捕捉时间演化，并通过卡尔曼滤波进行推断。

    arXiv:2609.17815v1 Announce Type: cross  Abstract: The Koopman operator has been widely used for time-series prediction in dynamical systems. However, prior work that learns latent ``Koopman spaces'' using neural networks often did not construct a valid Koopman space for forecasting, as these representations may be mathematically inconsistent with the operator-theoretic formulation and fail to capture the intrinsic low-rank structure of system dynamics. To address this issue, we introduce K$^2$SVD, a method that explicitly learns the leading singular functions of the Koopman operator by optimizing a Hilbert-Schmidt objective. This yields a well-defined low-rank approximation of the Koopman operator with an interpretable linear combination, featuring a compact latent space with less than $10\%$ of the dimensions used in previous work. In the learned Koopman space, K$^2$SVD further captures temporal evolution with a linear Gaussian state-space model and performs inference via Kalman filt
    
[^173]: 基于条件变分自编码器的电动汽车充电会话合成数据生成

    Synthetic Electric Vehicle Charging Session Generation Using a Conditional Variational Autoencoder

    [https://arxiv.org/abs/2609.17808](https://arxiv.org/abs/2609.17808)

    提出了一种条件变分自编码器（CVAE）模型，从真实的电动汽车充电交易数据中生成高质量的合成充电会话数据，以解决真实充电数据因隐私和获取限制而难以获得的难题，为配电网规划和仿真研究提供数据支持。

    

    电动汽车的日益普及预计将给住宅配电网络带来显著的额外负荷需求，因此需要真实的充电数据集用于规划和仿真研究。然而，由于隐私限制、记录不完整以及数据获取受限，真实世界的电动汽车充电数据往往难以获得。本文提出了一种条件变分自编码器（CVAE），用于从真实的交易级充电数据中生成合成的电动汽车充电会话。该模型在经过特征工程处理的会话特征上进行训练，包括插入时长、充电时长、传输能量、充电延迟以及周期性的一周内时间特征，同时以星期几和受控充电状态作为条件。模型采用高斯负对数似然（NLL）重构损失来建模特征层面的异方差不确定性，并使用Kullback-Leibler（KL）散度对潜在空间进行正则化。

    arXiv:2609.17808v1 Announce Type: cross  Abstract: The increasing adoption of electric vehicles (EVs) is expected to place significant additional demand on residential distribution networks, creating a need for realistic charging datasets for planning and simulation studies. However, access to real-world EV charging data is often limited due to privacy constraints, incomplete records, and restricted availability. This paper proposes a conditional variational autoencoder (CVAE) for the generation of synthetic EV charging sessions from real transaction-level charging data. The model is trained on engineered session features describing plug-in duration, charging duration, delivered energy, charging delay, and cyclical time-of-week, while conditioning on day of week and managed charging status. A Gaussian negative log-likelihood (NLL) reconstruction loss is employed to model feature-wise heteroscedastic uncertainty, and the latent space is regularised using a Kullback-Leibler (KL) divergen
    
[^174]: 大语言模型数学推理中应用题求解的四阶段分解与机制脆弱性

    A Four-Stage Decomposition of Word-Problem Solving and Mechanistic Fragility in LLM Math Reasoning

    [https://arxiv.org/abs/2609.17804](https://arxiv.org/abs/2609.17804)

    本文揭示大语言模型求解数学应用题的内部计算可分为模式抽象、运算规划、操作数绑定和计算四个阶段，并将无关干扰导致的失败机制定位于“运算规划”阶段的特定注意力头。

    

    大语言模型能够以高准确率求解小学数学应用题，然而在题目中插入一个无关的从句就可能使其完全失效。我们通过机制性解释来调和这些观察。我们证明模型的内部计算可分解为四阶段顺序流水线：模式抽象、运算规划、操作数绑定和计算，每个阶段在可识别的层级区间中产生独特的中间表示。使用相同的框架诊断干扰引起的失败，我们将损坏定位到单一阶段——运算规划，该阶段由一组注意力头实现，我们双向验证了这些注意力头的因果作用。简而言之，我们为大语言模型的数学应用题推理及其受干扰时的失效提供了机制性解释。

    arXiv:2609.17804v1 Announce Type: new  Abstract: Large language models solve grade-school math word problems with high accuracy, yet a single irrelevant clause inserted into the problem can collapse it. We reconcile these observations with a mechanistic account. We show that the model's internal computation decomposes into a four-stage sequential pipeline, Schema Abstraction, Operation Planning, Operand Binding, and Computation, each stage producing a distinct intermediate representation in an identifiable band of layers. Using the same scaffold to diagnose distractor-induced failure, we localize the corruption to a single stage, Operation Planning, implemented by a set of attention heads whose causal role we validate bidirectionally. In short, we provide a mechanistic interpretation of math word problem reasoning in LLMs, and their failure when distracted.
    
[^175]: SAiFE-gym：基于模型的集中流动性自动做市环境

    SAiFE-gym: Model-based Environments for Automated Market Making with Concentrated Liquidity

    [https://arxiv.org/abs/2609.17788](https://arxiv.org/abs/2609.17788)

    SAiFE-gym是一个基于Python的向量化模拟环境集合，用于研究带集中流动性的恒定乘积市场中的自动做市问题，可扩展支持高维强化学习工作流，以应对市场参数的不确定性。

    

    我们提出了SAiFE_gym，这是一个Python模块，提供了一系列模拟环境，用于研究具有集中流动性的恒定乘积市场（CPMs）中的交易问题。这些市场使流动性提供者（LPs）能够对其资金分配方式进行精细化的控制，并使他们能够根据市场条件动态调整其流动性提供的范围，这反过来决定了他们赚取费用的方式。我们将具有集中流动性的CPMs的微观结构分解为交互式组件，使研究人员和从业者能够捕捉各种经济设置。我们采用向量化方法来优化我们的环境，使其可扩展至最能描述序贯决策问题的高维强化学习（RL）工作流程。我们通过在市场参数不确定的情况下评估RL智能体在具有集中流动性的CPMs中的表现，展示了我们环境的优势。

    arXiv:2609.17788v1 Announce Type: cross  Abstract: We present SAiFE_gym, a Python module that provides a collection of simulation environments for studying trading problems in Constant Product Markets (CPMs) with Concentrated Liquidity (CL). These markets give Liquidity Providers (LPs) granular control over how their capital is allocated and enable them to adjust their range of liquidity provision dynamically based on market conditions, which in turn, dictates how they earn fees. We decompose the microstructure of CPMs with CL in interactive components that allow researchers and practitioners to capture various economic settings. We employ a vectorized approach to optimize our environments, making them scalable for high dimensional Reinforcement Learning (RL) workflows that best describe sequential decision problems. We demonstrate the benefits of our environments by evaluating the performance of RL agents in CPMs with CL under uncertainty in market parameters.
    
[^176]: 用于可审计次日野火蔓延预测的模块化深度学习机制

    Modular Deep Learning Mechanisms for Auditable Next-Day Wildfire Spread Prediction

    [https://arxiv.org/abs/2609.17763](https://arxiv.org/abs/2609.17763)

    本文提出三种模块化深度学习增强机制（风坡条件化注意力偏置、物理特征检索增强的输出校正和火点条件化双流门控），实现了可审计、可解释的次日野火蔓延预测。

    

    次日野火预测要求模型的预测结果能够与其计算过程中所使用的假设和历史证据一起进行评估。尽管深度学习可以从遥感数据中学习空间模式，但仅凭预测性能并不能确立物理保真度或业务可信度。本研究针对次日活跃火点预测研究了三种模块化增强机制：风与坡度条件化的注意力偏置、物理特征检索增强的输出校正，以及火点条件化的双流门控。注意力偏置能够揭示预设的方向性偏好，而检索模块利用九维环境与火状态描述符选择历史图块，并对冻结模型的logits应用学习到的校正。这些模块在Next Day Wildfire Spread基准数据集上跨五个骨干网络进行评估，采用分阶段消融实验、方向性审计、检索扰动等方法……

    arXiv:2609.17763v1 Announce Type: new  Abstract: Next-day wildfire prediction requires models whose forecasts can be evaluated alongside the assumptions and historical evidence used in their computation. Although deep learning can learn spatial patterns from remote-sensing data, predictive performance alone does not establish physical fidelity or operational trustworthiness. This study investigates three modular augmentations for next-day active-fire prediction: wind- and slope-conditioned attention biases, physics-feature retrieval-augmented output correction, and fire conditioned dual-stream gating. The attention biases expose prescribed directional preferences, while the retrieval module selects historical tiles using a nine-dimensional environmental and fire-state descriptor and applies a learned correction to a frozen model's logits. The modules are evaluated across five backbones on the Next Day Wildfire Spread benchmark, using staged ablations, directional audits, retrieval pert
    
[^177]: 面向Muon的无导数结构化更新方法

    Derivative-Free Structured Updates for Muon

    [https://arxiv.org/abs/2609.17759](https://arxiv.org/abs/2609.17759)

    提出一种无导数框架，通过结构化有限差分（尤其是随机秩一探测）构建Muon风格的参数更新，使Muon在梯度不可用或不可靠时依然可用，并能以较少的函数求值次数换取精度可接受的更新。

    

    Muon通过对基于梯度的动量矩阵进行正交化来更新矩阵形式的神经网络参数。其对导数的依赖限制了它在梯度不可用或不可靠情况下的应用。我们开发了一个无导数框架，利用结构化有限差分来构建Muon风格的更新。我们考虑了四种变体：完整的逐元素恢复、随机低秩代理、基对齐的秩一探测以及直接结构化搜索。研究表明，穷举的基对齐探测在理想极分解正交化之前的正缩放意义下等价于坐标有限差分。矩阵回归实验表明，随机秩一探测可以大幅减少函数求值次数，但代价是更新精度较低。在回归任务和神经网络上的受控噪声梯度实验说明了在何种情况下准确的函数值可以补偿不可靠的梯度信息。一个小型的CartPole研究进一步（验证了该方法的实用性）。

    arXiv:2609.17759v1 Announce Type: cross  Abstract: Muon updates matrix-valued neural-network parameters by orthogonalizing a gradient-based momentum matrix. Its reliance on derivatives limits its use when gradients are unavailable or unreliable. We develop a derivative-free framework that constructs Muon-style updates from structured finite differences. Four variants are considered: full entrywise recovery, random low-rank surrogates, basis-aligned rank-one probing, and direct structured search. Exhaustive basis-aligned probing is equivalent, up to positive scaling before ideal polar orthogonalization, to coordinate finite differences. Matrix-regression experiments show that random rank-one probing can reduce the number of function evaluations substantially, at the cost of less accurate updates. Controlled noisy-gradient experiments on regression and a neural network illustrate when accurate function values can compensate for an unreliable gradient oracle. A small CartPole study furthe
    
[^178]: 曲线上的SAM：用于鲁棒权重空间插值的锐度感知模式连接

    SAM-on-the-Curve: Sharpness-Aware Mode Connectivity for Robust Weight-Space Interpolation

    [https://arxiv.org/abs/2609.17748](https://arxiv.org/abs/2609.17748)

    该论文提出锐利模式连接，将模式连接重新表述为邻域鲁棒的路径优化问题，通过一阶锐度感知近似使连接曲线的整个局部邻域保持低损失平坦性，从而在分布偏移下获得更鲁棒的权重空间插值。

    

    独立训练却达到相似性能的深度神经网络可以通过权重空间中的低损失参数化曲线相互连接，这一现象被称为模式连接。该几何特性是权重平均、模型集成和模型合并等实用技术的基础。我们认为低损失连接是一个不完整的几何准则：它仅控制一维轨迹上的损失，而对曲线周围的权重空间邻域缺乏约束，因此优化得到的曲线可能穿过在分布偏移下变得脆弱的尖锐“山脊”。为此，我们将模式连接重新表述为一个邻域鲁棒的路径优化问题，寻求一条其整个局部邻域均保持低损失的曲线。我们提出了锐利模式连接，通过对所得的极小极大泛函应用一阶锐度感知近似，在整个曲线上强制平坦性，而不仅仅是（摘要在此处截断）。

    arXiv:2609.17748v1 Announce Type: new  Abstract: Deep neural networks that are independently trained to similar performance can be connected by low-loss parametric curves in weight space, a phenomenon known as Mode Connectivity (MC). This geometric property underpins practical techniques such as weight averaging, model ensembling, and model merging. We argue that low-loss connectivity is an incomplete geometric criterion: it controls loss only along a one-dimensional trajectory while leaving the surrounding weight-space neighborhood unconstrained, so the optimized curve may traverse sharp ridges that become fragile under distribution shift. We therefore reformulate mode connectivity as a neighborhood-robust path optimization problem, seeking a curve whose entire local neighborhood maintains low loss. We propose Sharp Mode Connectivity (SMC), which applies a first-order sharpness-aware approximation to the resulting minimax functional, enforcing flatness along the entire curve rather th
    
[^179]: REVERSAL-BENCH：一个用于衡量免重置强化学习悬崖的可逆性轴与重置预言机

    REVERSAL-BENCH: A Reversibility Axis and Reset Oracle for Measuring the Reset-Free RL Cliff

    [https://arxiv.org/abs/2609.17745](https://arxiv.org/abs/2609.17745)

    REVERSAL-BENCH 通过可连续调节的可逆性参数和真值重置预言机，揭示了免重置强化学习存在一个“可逆性悬崖”——随着环境不可逆性增强，免重置智能体会不可避免地陷入无法恢复的状态。

    

    自主强化学习的一个核心目标是无需外部重置的连续策略训练。然而，现有范式在很大程度上依赖于底层环境的可逆性，而这一特性在现实世界的操作任务中并不存在——例如将物体推下桌子或洒落颗粒物质等事件是无法撤销的。我们提出了 REVERSAL-BENCH，这是一个通过连续参数 ρ ∈ [0, 1] 控制环境可逆性的基准测试，并提供了一个重置预言机——一种用于检验状态可恢复性的真值验证机制，涵盖五个物理引擎中的八个操作设置。对广泛策略架构的评估，包括标准的 actor-critic 算法、安全强化学习以及专门的免重置框架，揭示了一个陡峭的可逆性悬崖：随着 ρ 的增加，免重置智能体持续被吸收进不可恢复的状态，而回合制智能体则保持稳定学习。我们看到这种失败

    arXiv:2609.17745v1 Announce Type: cross  Abstract: A central goal of autonomous reinforcement learning is continuous policy training without external resets. However, existing paradigms largely depend on underlying environmental reversibility, a property absent in real world manipulation, where events such as pushing objects off tables or spilling granular substances cannot be undone. We introduce REVERSAL-BENCH, a benchmark that controls reversibility via a continuous parameter $\rho \in [0, 1]$ and provides a reset oracle, a ground-truth verification mechanism to test state recoverability across eight manipulation settings in five physics engines. Evaluating a broad spectrum of policy architectures, including standard actor-critic algorithms, safe RL, and specialized reset-free frameworks, reveals a sharp reversibility cliff: reset-free agents are consistently absorbed into irrecoverable states as $\rho$ increases, whereas episodic agents maintain steady learning. We see this failure
    
[^180]: 基于能量移动距离的相似性配对方法用于LHC自监督预训练

    Similarity Pairing with Energy Mover's Distance for Self-Supervised Pre-Training at the LHC

    [https://arxiv.org/abs/2609.17738](https://arxiv.org/abs/2609.17738)

    该论文提出一种利用能量移动距离（EMD）按相似性配对真实对撞事件作为数据增强视图的方法，无需人工构造或模拟增强事件即可保持事件物理真实性，从而改进LHC自监督预训练。

    

    在大型强子对撞机（LHC）上训练基础模型的许多自监督方法依赖于数据增强，以鼓励模型将事件嵌入到对某些物理或探测器对称性保持不变的表示空间中。一个常见的挑战在于选择合适的增强方法集合具有很大的自由度，而下游性能正依赖于这些增强。数据增强的实现要么需要修改现有事件，可能破坏事件的真实性（保真度），要么需要模拟更多的事件变体，这在计算上非常昂贵。在本工作中，我们提出了一种数据驱动的方法，通过能量移动距离（EMD）根据相似性对事件进行配对——EMD通过衡量将一个事件变换为另一个事件所需的“功”来度量两个事件的相似程度。通过这种方法，可以采样不同的真实事件并根据其相似性进行匹配，作为学习不变性的视图，同时保持事件本身的物理内容……

    arXiv:2609.17738v1 Announce Type: cross  Abstract: Many self-supervised methods for training foundation models at the Large Hadron Collider (LHC) rely on data augmentations to encourage the model to embed events into a representation space invariant to certain physical or detector symmetries. A common challenge arises from the large freedom in choosing a proper set of augmentations on which downstream performance depends. The implementation of augmentations involves either modifying existing events, potentially breaking the event fidelity, or simulating more event variants, which is computationally intensive. In this work, we present a data-driven method of pairing events by their similarity via the energy mover's distance (EMD), which measures how similar two events are in terms of the work required to transform one into the other. With this approach, distinct events are sampled and matched by their similarity to serve as views for learning invariance, keeping the physics content of e
    
[^181]: 从分子动力学数据中机器学习动力学

    Machine learning kinetics from molecular dynamics data

    [https://arxiv.org/abs/2609.17736](https://arxiv.org/abs/2609.17736)

    本综述总结了利用自监督机器学习方法从分子动力学数据中估计承诺概率等关键动力学统计量的现代技术，并建立了连接生成元偏微分方程、变分原理、马尔可夫状态模型与神经网络的统一算子理论框架。

    

    大多数分子转变发生在远超直接分子动力学模拟能力的时间尺度上。承诺概率（committor），即某一构型在到达反应物状态之前到达产物状态的概率，是一个核心的动力学统计量，它提供了一个与机制无关的反应坐标，是过渡路径理论和速率计算的基础。本综述调研了从分子模拟中估计承诺概率及相关动力学统计量的现代方法，重点介绍了自监督方法，这类方法通过学习其定义性动力学方程的解，而非依赖有标记的射击（shooting）数据。我们建立了一个统一的算子视角，将基于生成元的偏微分方程、变分原理、马尔可夫状态模型、动力学伽辽金近似和神经网络联系起来。经验和理论证据均表明了这些方法的高效性。我们提供了理论与实践方面的指导。

    arXiv:2609.17736v1 Announce Type: cross  Abstract: Most molecular transitions occur on timescales far beyond direct molecular dynamics simulations. The committor, the probability that a configuration reaches a product state before a reactant state, is a central kinetic statistic, providing a mechanism-independent reaction coordinate and a foundation for transition path theory and the calculation of rates. This review surveys modern approaches for estimating the committor and related kinetic statistics from molecular simulations, with an emphasis on self-supervised methods that learn solutions of their defining dynamical equations rather than relying on labeled shooting data. We develop a common operator viewpoint connecting generator-based partial differential equations, variational principles, Markov state models, dynamical Galerkin approximation, and neural networks. Empirical and theoretical evidence points to the efficiency of these methods. We provide theoretical and practical gui
    
[^182]: FAME：一种基于FPGA的近似乘法器评估平台，结合模式引导的DNN重训练

    FAME: An FPGA-Based Platform for Approximate Multipliers Evaluation with Pattern-Guided DNN Retraining

    [https://arxiv.org/abs/2609.17730](https://arxiv.org/abs/2609.17730)

    本文提出FAME，一个基于FPGA的近似乘法器评估平台，通过在硬件中直接实现近似乘法器并结合模式引导的DNN重训练，大幅加速了DNN推理中近似乘法器的精度评估与误差缓解流程。

    

    近似乘法器可以降低深度神经网络（DNN）推理中的硬件面积和能耗；然而，它们会引入计算误差。由于评估时间过于漫长，在多样化的DNN模型和大规模数据集上评估大量近似乘法器设计的精度仍然充满挑战。这种开销主要源于在CPU和GPU平台上使用查找表（LUT）对近似乘法器行为进行缓慢的仿真。此外，由此产生的精度下降必须被仔细量化，并在必要时加以缓解（例如通过重训练），这进一步增加了整体评估成本。为了解决这些挑战，我们提出了FAME，一个基于FPGA的近似乘法器评估平台。该平台利用现场可编程门阵列（FPGA）的可重构逻辑直接在硬件中实现近似乘法器，从而无需基于LUT的仿真。

    arXiv:2609.17730v1 Announce Type: cross  Abstract: Approximate multipliers can reduce hardware area and energy consumption in Deep Neural Network (DNN) inference; however, they introduce computational errors. Assessing the accuracy of numerous approximate multiplier designs across diverse DNN models and large-scale datasets remains challenging due to prohibitive evaluation times. This overhead primarily stems from the slow emulation of approximate multiplier behavior using look-up tables (LUTs) on CPU and GPU platforms. Moreover, the resulting accuracy degradation must be carefully quantified and, if necessary, mitigated (e.g., through retraining), further increasing the overall evaluation cost. To address these challenges, we propose FAME, an FPGA-based platform for evaluating approximate multipliers. The platform exploits the reconfigurable logic of Field-Programmable Gate Arrays (FPGAs) to implement approximate multipliers directly in hardware, eliminating the need for LUT-based emu
    
[^183]: 用于级联衰变中鲁棒共振质量回归的自监督学习

    Self-Supervised Learning for Robust Resonance Mass Regression in Cascade Decays

    [https://arxiv.org/abs/2609.17726](https://arxiv.org/abs/2609.17726)

    该论文提出遵循基础模型范式，先用VICReg自监督预训练transformer编码器以学习对各种数据破坏不变的嵌入表示，再微调用于级联衰变中重共振粒子的质量回归，从而在系统不确定性和分布偏移下实现鲁棒的质量重建。

    

    从带有丢失能量的衰变产物中重建重共振粒子的质量，是直接决定对撞机实验中新物理搜索灵敏度的核心任务之一。由于存在各种系统不确定性和分布偏移，针对该问题的监督学习方法通常难以很好地泛化。穷尽标注数据中所有可能的变化可能非常耗费计算资源，而模型泛化的失败则会破坏重建的共振峰宽度，而该宽度在寻峰分析中至关重要。在本工作中，我们遵循基础模型范式，采用自监督方法，使用VICReg预训练一个transformer编码器，以学习对各种破坏不变的嵌入表示，随后对其进行微调，用于重共振粒子的质量回归，其中共振粒子质量范围为2.5至6.5 TeV，并采用类似SUSY的级联衰变至十一体末态。我们展示了……（摘要在此处截断）

    arXiv:2609.17726v1 Announce Type: cross  Abstract: Reconstructing the mass of a heavy resonance from its decay products with missing energy is one of the central tasks that directly determine the sensitivity in new physics searches at collider experiments. Supervised learning approaches to this problem often struggle to generalize well due to the presence of various systematic uncertainties and distribution shifts. Exhausting all possible variations in the labeled data can be very compute-intensive, while a failure of the model to generalize can corrupt the reconstructed resonance widths that are critical in peak-hunting analyses. In this work, following the foundation model paradigm, we use a self-supervised approach to pre-train a transformer encoder with VICReg to learn an embedding invariant to various corruptions, then fine-tune it for mass regression on a heavy resonance with masses ranging from 2.5 to 6.5 TeV and a SUSY-like cascade decay into an eleven-body final state. We show
    
[^184]: 利用可解释机器学习解码脂质纳米颗粒的肝外靶向

    Decoding Extrahepatic Targeting of Lipid Nanoparticles with Interpretable Machine Learning

    [https://arxiv.org/abs/2609.17721](https://arxiv.org/abs/2609.17721)

    本研究开发了一个可解释机器学习框架，利用涵盖476种静脉注射LNP制剂的文献数据集，预测脂质纳米颗粒在肝脏与肝外组织中的蓄积分布，并从中提取出实现肝外RNA递送的分子设计规则。

    

    脂质纳米颗粒（LNPs）已经变革了RNA药物领域，然而其临床应用仍然受到全身给药后主要在肝脏蓄积的限制。将LNPs重新定向至肝外组织，需要理解脂质化学结构与制剂组成如何共同决定体内的生物分布。在此，我们开发了一个可解释的机器学习框架，用于预测LNPs在肝脏与肝外组织中的蓄积情况，并确定肝外RNA递送的分子设计规则。研究团队从81项研究中精心整理了一个包含476种静脉注射LNP制剂的文献数据集，整合了制剂组成、脂质化学结构以及基于IVIS成像的生物分布谱。可电离脂质、辅助脂质、甾醇、PEG化或聚合物偶联脂质、附加脂质以及聚合物重复单元的标准化SMILES表示被转换为RDKit Expert描述符，并与制剂（原文摘要在此处截断）

    arXiv:2609.17721v1 Announce Type: cross  Abstract: Lipid nanoparticles (LNPs) have transformed RNA medicine, yet their clinical utility remains constrained by predominant hepatic accumulation after systemic administration. Redirecting LNPs to extrahepatic tissues requires understanding of how lipid chemistry and formulation composition jointly govern in vivo biodistribution. Here, we develop an interpretable machine learning framework to predict hepatic versus extrahepatic LNP accumulation and identify molecular design rules for extrahepatic RNA delivery. A literature-derived dataset of 476 intravenous LNP formulations was curated from 81 studies, integrating formulation composition, lipid chemical structures, and IVIS-based biodistribution profiles. Standardized SMILES representations of ionizable lipids, helper lipids, sterols, PEGylated or polymer-conjugated lipids, additional lipids, and polymer repeat units were converted into RDKit Expert descriptors and combined with formulation
    
[^185]: 深度强化学习与模型预测控制之间共享控制权限的复合梯度学习方法

    Composite-Gradient Learning for Shared Control Authority Between Deep Reinforcement Learning and Model Predictive Control

    [https://arxiv.org/abs/2609.17697](https://arxiv.org/abs/2609.17697)

    本文提出了一种复合梯度学习（CGL）方法，通过将DRL与MPC的控制输入表示为联合动作并显式考虑两者的交互，将MPC控制器深度集成到DRL训练过程中，突破了传统方法将MPC仅视为环境一部分的局限。

    

    深度强化学习（DRL）与模型预测控制（MPC）相结合的方法正日益广泛地应用于自主系统控制，该方法结合了两者的互补能力。DRL通过与环境的交互学习控制策略；MPC则利用系统模型在考虑约束条件的前提下优化控制输入。在具有共享控制权限的DRL-MPC框架中，DRL智能体和MPC控制器各自决定部分控制输入。然而，常见的学习框架将MPC视为环境的一部分，因此没有明确考虑MPC对控制的贡献以及其与DRL智能体之间的交互。本文提出了一种新颖的复合梯度学习（CGL）方法，通过将DRL和MPC的控制输入表示为联合动作，并在训练期间更新DRL智能体时考虑两者之间的交互，从而将MPC控制器集成到学习过程中。

    arXiv:2609.17697v1 Announce Type: new  Abstract: Integrated deep reinforcement learning (DRL) and model predictive control (MPC) methods are increasingly used to control autonomous systems by combining their complementary capabilities. DRL learns control policies through interaction with the environment. MPC uses a system model to optimize control inputs while accounting for constraints. In DRL-MPC frameworks with shared control authority, both the DRL agent and the MPC controller each determine part of the control inputs. However, common learning formulations treat MPC as part of the environment and therefore do not explicitly account for MPC's contribution to control or its interaction with the DRL agent. This paper proposes a novel composite-gradient learning (CGL) method that integrates the MPC controller into the learning process by representing the DRL and MPC control inputs as a joint action and accounting for their interaction when updating the DRL agent during training. CGL is
    
[^186]: 通过投机草稿树加速扩散采样

    Accelerating Diffusion Sampling via Speculative Draft Trees

    [https://arxiv.org/abs/2609.17691](https://arxiv.org/abs/2609.17691)

    该论文的核心创新是提出“草稿树”方法，将扩散模型中的投机采样与相对熵编码（REC）相联系，用树状候选结构取代线性链式草稿，并采用贪心拒绝采样作为草稿-目标耦合，从而提高接受率、减少昂贵的目标函数评估，实现扩散采样加速。

    

    投机采样通过先起草廉价的候选状态，并在一种能精确保留目标分布的耦合下对其进行校正，从而加速扩散模型的生成，减少昂贵的目标评估次数。现有的扩散采样器，尤其是基于反射最大耦合的方法，受到拓扑结构的限制：其前瞻草稿形成链式图（即单一线性序列），这从本质上限制了每次目标评估的接受率。我们将扩散模型中的投机采样与相对熵编码（REC）联系起来。这一视角表明前瞻草稿不必是线性的，并启发了我们的核心贡献——草稿树，它丰富了每轮考虑的候选对象，降低了目标函数的评估次数。我们进一步采用贪心拒绝采样（一种REC算法）作为草稿-目标耦合，在保证精确目标采样的同时提高了接受率。在多种……上的实验（摘要在此处截断）

    arXiv:2609.17691v1 Announce Type: cross  Abstract: Speculative sampling accelerates diffusion model generation by drafting inexpensive candidate states and correcting them under a coupling that preserves the target distribution exactly, reducing the number of expensive target evaluations. Existing diffusion samplers, notably those based on reflection maximal coupling, are topologically constrained: their lookahead drafts form a chain graph, a single linear sequence, which inherently limits the acceptance rate per target evaluation. We connect speculative sampling in diffusion models to relative entropy coding (REC). This perspective shows the lookahead need not be linear and motivates our central contribution, draft trees, which enrich the candidates considered per round and lower the target function evaluations. We further adopt greedy rejection sampling, an REC algorithm, as the draft-target coupling, improving acceptance while guaranteeing exact target samples. Experiments across di
    
[^187]: 缺失的“我不知道”：为什么三项推理可靠性研究发现共同指向校准弃答

    The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention

    [https://arxiv.org/abs/2609.17686](https://arxiv.org/abs/2609.17686)

    该论文的核心创新是论证三项看似独立的LLM可靠性研究发现（推理强化学习破坏工具可靠性表征、安全约束下大小模型的差异化表现、以及缺乏“我不知道”功能的系统必然产生无穷幻觉）实际上共同指向同一项缺失能力——校准弃答。

    

    最近的三项研究结果描述了看似互不相关的大语言模型可靠性问题。Yin等人（2026）表明推理强化学习会破坏工具可靠性表征。Suleymanov等人（2026）表明在安全约束生成条件下，大模型会改写被标记的文本片段，而小模型则会截断这些片段。Bastounis等人（2024）证明，任何缺乏内隐“我不知道”功能的一致推理系统，在广泛的问题类别上必然产生无穷多次幻觉。我们认为这些发现共同指向单一干预措施：校准弃答正是各项研究独立识别出的缺失能力，尽管它们所记录的不可用性——能力缺口、策略缺口和递归论缺口——在每种情况下来源各不相同。诚实性后训练已缩小了已部署模型中的这一差距，但要原则上弥补Bastounis所识别的问题类别，需要一个校准弃答函数，其训练信号在……（原文截断）

    arXiv:2609.17686v1 Announce Type: cross  Abstract: Three recent results describe what look like unrelated LLM reliability problems. Yin et al. (2026) show reasoning RL collapses tool-reliability representations. Suleymanov et al. (2026) show that under safety-constrained generation, large models rewrite flagged spans while small models truncate. Bastounis et al. (2024) prove any consistent-reasoning system without an implicit "I don't know" function must hallucinate infinitely often on broad problem classes. We argue these findings converge on a single intervention: calibrated abstention is what each independently identifies as the missing capability, even though the unavailability they document, a capability gap, a policy gap, and a recursion-theoretic gap, has a different source in each case. Honesty post-training has narrowed the gap in deployed models, but principled closure of the class Bastounis identifies requires a calibrated abstention function whose training signal at the lea
    
[^188]: DSD：通过扩散技能发现学习多样且可复用的运动技能

    DSD: Learning Diverse and Reusable Motor Skills via Diffusion Skill Discovery

    [https://arxiv.org/abs/2609.17682](https://arxiv.org/abs/2609.17682)

    该论文提出DSD方法，利用扩散模型进行技能发现，为模拟角色学习多样且可复用的运动技能，从而克服高维控制问题中边际状态熵难以直接估计的难题。

    

    人类能够通过在不同目标和情境下复用丰富的运动技能库来高效地学习新任务。类似的策略也可用于让模拟角色通过利用可复用的运动技能来高效地执行新任务。为了支持广泛的下游任务，学习到的技能库应当是多样的，既包含不同的行为，也包含每种行为内部在空间和时间上的变化。学习多样技能的一种常用方法是最小化技能潜在变量与策略产生的状态之间的互信息最大化。边际状态熵促进广泛的行为覆盖，而条件熵则鼓励每个潜在变量产生一致的行为。然而，在高维控制问题中直接估计边际状态熵是不可行的。因此，先前的方法依赖于间接的潜在空间近似或对状态分布的粗略估计器……（摘要在此处被截断）

    arXiv:2609.17682v1 Announce Type: new  Abstract: Humans efficiently learn new tasks by reusing a rich repertoire of motor skills across different goals and contexts. A similar strategy can also be used to enable simulated characters to efficiently perform new tasks by leveraging reusable motor skills. To support a wide range of downstream tasks, the learned repertoire should be diverse, consisting of distinct behaviors as well as spatial and temporal variation within each behavior. A commonly used method for learning diverse skills is by maximizing the mutual information between skill latents and the states produced by a policy. The marginal state entropy promotes broad behavioral coverage, while the conditional entropy encourages consistent behaviors from each latent. However, directly estimating the marginal state entropy is intractable in high-dimensional control problems. Prior methods therefore rely on indirect latent-space approximations or coarse estimators of the state distribu
    
[^189]: 信息论极限下的高效鲁棒学习

    Efficient Robust Learning at the Information-Theoretic Limit

    [https://arxiv.org/abs/2609.17655](https://arxiv.org/abs/2609.17655)

    本文解决了 Blanc 遗留的开放问题，通过巧妙运用无悔学习器技术，首次给出了在 ERM 预言机辅助下达到信息论最优错误率 η+ε 的多项式时间鲁棒学习算法，并为具有三明治多项式性质的函数类提供了无需预言机的高效算法。

    

    在近期一项重要工作中，Blanc（2026）给出了一种针对固定分布鲁棒学习布尔概念类的算法，该算法输出一个（随机化的）分类器，能够达到 η + ε 的最优错误率，其中 η 为噪声率。相比之下，众所周知，确定性假设无法达到低于 2η + ε 的错误率。Blanc 的算法在计算上效率低下，其工作中遗留的主要开放问题是：在可访问经验风险最小化（ERM）预言机的条件下，找到一种多项式时间算法。在本文中，我们解决了这一问题，并给出了这样一种算法。令人惊讶的是，我们的技术关键地利用了各种类型的无悔学习器。此外，我们给出了一种高效算法（无需 ERM 预言机），用于鲁棒学习任何在超收缩分布下允许三明治多项式的函数类。作为其中一个结果……

    arXiv:2609.17655v1 Announce Type: cross  Abstract: In an important recent work, Blanc (2026) gave an algorithm for robustly learning Boolean concept classes with respect to a fixed distribution that outputs a (randomized) classifier achieving the optimal error of $\eta + \varepsilon$ where $\eta$ is the noise rate. In contrast, it is well known that deterministic hypotheses cannot achieve error less than $2\eta + \varepsilon.$   Blanc's algorithm is computationally inefficient, and the main problem left open in his work is to find a polynomial-time algorithm given access to an oracle for empirical risk minimization (ERM). In this paper, we resolve this problem and give such an algorithm. Perhaps surprisingly, our techniques make crucial use of various types of no-regret learners.   Additionally, we give an efficient algorithm (no ERM oracle required) for robustly learning any function class that admits sandwiching polynomials with respect to hypercontractive distributions. As one conse
    
[^190]: 基于正则化最小二乘的二次神经网络训练及其在系统辨识中的应用

    Regularized Least Squares Training of Quadratic Neural Networks with Applications to System Identification

    [https://arxiv.org/abs/2609.17654](https://arxiv.org/abs/2609.17654)

    本文提出一种正则化最小二乘法训练二次神经网络，可得到权重的闭式解析解及其对数据误差的灵敏度表达式，避免了反向传播易陷入局部极小值的问题并显著降低计算时间。

    

    本文提出了一种带正则化的最小二乘方法用于二次神经网络的训练。所提出的方法在正则化系数为正的情况下，给出了训练优化问题解的下界；此外，还给出了近似解及其灵敏度的闭式表达式。该下界是紧的，且当正则化系数为零时，近似解即为最优解。与反向传播等可能陷入局部极小值的迭代数值方法相比，拥有权重的闭式表达式可大幅减少计算时间。所提出的方法有三大主要贡献：（i）给出了权重的解析表达式；（ii）提供了权重对数据误差敏感度的解析表达式；（iii）建立了……之间的联系（原文摘要在此处截断）。

    arXiv:2609.17654v1 Announce Type: new  Abstract: This paper proposes a least squares approach for the training of quadratic neural networks with regularization. The proposed methodology yields a lower bound on the solution of the training optimization problem for the case where the regularization coefficient is positive. Moreover, it yields closed-form expressions for the approximate solution and its sensitivity The lower bound is tight and the approximate solution is the optimal solution when the regularization coefficient is zero. Having a closed-form expression for the weights reduces considerably the computational time when compared with iterative numerical methods such as backpropagation that can get stuck in local minima. The proposed approach has three main contributions, namely, (i) it yields an analytical expression for the weights, (ii) an analytical expression for the sensitivity of the weights to errors in the data is also provided, (iii) it establishes a connection between
    
[^191]: 反思、修订、复用：面向图形用户界面（GUI）智能体的免训练技能演化

    Reflect, Revise, Reuse: Training-Free Skill Evolution for GUI Agents

    [https://arxiv.org/abs/2609.17653](https://arxiv.org/abs/2609.17653)

    提出免训练框架EvoSkill-GUI，将GUI智能体的技能构建为包含可执行计划与故障恢复规则的结构化多文件包，使技能能够在部署时根据执行反馈自我修订，从而应对动态界面导致的计划失效问题。

    

    GUI智能体需要在动态图形用户界面上执行长时程任务，其中弹窗、延迟加载和组件位置变动经常使执行前制定的固定计划失效。近期的智能体技能框架通过封装可复用的程序性知识来缓解这一问题，然而现有的技能设计大多没有针对GUI执行动态进行开发，并且将技能视为部署前生成的静态产物，而非能够通过执行过程不断改进的活的程序性知识。我们认为GUI智能体需要的不是更好的静态技能，而是能够在部署时根据执行反馈进行修订、且无需额外训练的技能。我们提出了EvoSkill-GUI，一个免训练框架，其中每项技能都是一个结构化的多文件包，包含检索元数据、可执行计划、备用定位方案、故障恢复规则、无障碍工具和失败案例。EvoSkill-GUI通过……

    arXiv:2609.17653v1 Announce Type: cross  Abstract: GUI agents execute long-horizon tasks on dynamic graphical user interfaces, where pop-ups, delayed loads, and relocated widgets routinely invalidate plans fixed before execution. Recent agent-skill frameworks encapsulate reusable procedural knowledge to mitigate this, yet existing skill designs are largely developed without targeting GUI execution dynamics and treat skills as static artifacts produced before deployment rather than living procedural knowledge that improves through it. We argue that what GUI agents need is not better static skills, but skills that can be revised from execution feedback at deployment time, without additional training. We propose \textbf{EvoSkill-GUI}, a training-free framework in which each skill is a structured multi-file package containing retrieval metadata, executable plans, backup localization, failure-recovery rules, accessibility utilities, and failure cases. EvoSkill-GUI operates through a \textbf
    
[^192]: Fathom：面向卸载KV缓存稀疏解码的逐查询读取深度

    Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches

    [https://arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652)

    Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。

    

    当智能体会话运行至百万token且同时驻留多个会话时，KV缓存及其排序索引存放在主机内存中，而针对top-k步骤对所有n个键进行排序的扫描成为限制解码速度的流量瓶颈。我们提出Fathom，一种由每个查询自主决定读取每个键通道多少比特的键扫描方法。4位K缓存以通道优先的方式存储为比特平面，因此t个平面的前缀恰好构成该通道的t位量化器，查询通过对方差加权通道重要性进行逆注水来分配其比特预算。在Qwen3-8B上处理一百万token时，解码步骤的GPU时间比Double Sparsity、Loki和SparQ r=32的136位扫描快1.67倍；在与SparQ的68位读取（r=16）相同的GPU时间内，Fathom读取的字节数减少18%，且在七个模型与上下文设置中的六个上注意力误差更低。在RULER风格的任务上，每次逐token扫描均与精确top-k解码结果相匹配。

    arXiv:2609.17652v1 Announce Type: cross  Abstract: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding
    
[^193]: 用于可扩展材料设计与性质预测的鲁棒高效AI框架

    Robust and Efficient AI Frameworks for Scalable Material Design and Property Prediction

    [https://arxiv.org/abs/2609.17646](https://arxiv.org/abs/2609.17646)

    本论文提出CrysXPP和CrysGNN等AI框架，利用无监督图自编码和大规模自监督图预训练技术加速晶体材料发现，显著降低了对昂贵DFT计算和大量标注数据的依赖。

    

    本论文开发了鲁棒且高效的AI框架，通过解决材料设计流程的两大主要阶段——晶体性质预测和晶体结构生成——来加速晶体材料的发现。鉴于密度泛函理论（DFT）的高昂计算成本以及标注材料数据的有限性，本论文探索了图表示学习、预训练、多模态学习和生成建模等技术，以实现可扩展的材料设计。在性质预测方面，论文首先介绍了CrysXPP，该方法通过无监督图自编码学习可迁移的晶体表示，从而减少对大规模性质标注数据集的依赖；随后提出了CrysGNN，这是一个大规模自监督图预训练框架，能够捕捉原子连接性、化学属性和全局结构信息，并将这些知识迁移到下游性质预测任务中。

    arXiv:2609.17646v1 Announce Type: cross  Abstract: This thesis develops robust and efficient AI frameworks for accelerating crystalline materials discovery by addressing both major stages of the materials-design pipeline: crystal property prediction and crystal structure generation. Motivated by the high computational cost of Density Functional Theory (DFT) and the limited availability of labeled materials data, the thesis explores graph representation learning, pretraining, multimodal learning, and generative modeling for scalable materials design.   For property prediction, the thesis first introduces CrysXPP, which learns transferable crystal representations through unsupervised graph autoencoding, reducing dependence on large property-labeled datasets. It then proposes CrysGNN, a large-scale self-supervised graph pretraining framework that captures atomic connectivity, chemical attributes, and global structural information and transfers this knowledge to downstream property predict
    
[^194]: 重新思考天文学语言模型中开放式科学推理的领域专业化

    Rethinking Domain Specialization for Open-Ended Scientific Reasoning in Astronomy Language Models

    [https://arxiv.org/abs/2609.17644](https://arxiv.org/abs/2609.17644)

    该研究通过天文学奥赛问答基准发现，强大的通用大模型在开放式科学推理中表现优于领域专业化模型，表明领域微调的价值应被视为取决于具体任务和部署环境的选择。

    

    领域专业化的语言模型被广泛应用于科学问答，但更强大的通用系统提出了一个更为尖锐的问题：领域特定的微调在何种情况下对开放式科学推理仍然有价值？我们在天文学领域对这一问题展开研究，构建了一个精选的问答基准，素材来自公开可得的2017—2026年奥赛风格资料。自由回答子集包含300道题目，其中204题为纯文本示例，96题为图像关联示例。我们使用基于评判者的正确性评估及补充参考指标，比较了开源权重与API服务的通用模型、多模态模型以及天文学专业化模型。在这个测试平台上，强大的通用模型确立了最高的正确性基线，而对指标一致性、评判者敏感性、基准构成和模态的分析则揭示了单一排行榜无法体现的差异。这些结果促使我们将领域专业化视为一种取决于任务和部署环境的选择。

    arXiv:2609.17644v1 Announce Type: cross  Abstract: Domain-specialized language models are widely used for scientific question answering, but stronger general-purpose systems raise a sharper question: when does domain-specific fine-tuning remain valuable for open-ended scientific reasoning? We study this in astronomy with a curated QA benchmark from publicly available 2017--2026 Olympiad-style materials. The free-response subset contains 300 questions, including 204 text-only and 96 image-linked examples. We compare open-weight and API-served general-purpose, multimodal, and astronomy-specialized models using judge-based correctness and complementary reference metrics. Strong general-purpose models establish the highest correctness baseline in this testbed, while analyses of metric agreement, judge sensitivity, benchmark composition, and modality reveal variation not captured by a single leaderboard. These results motivate treating domain specialization as a task- and deployment-depende
    
[^195]: 物理信息神经网络、神经算子及其应用讲义

    Lecture notes on Physics Informed Neural Networks, Neural Operators, and their applications

    [https://arxiv.org/abs/2609.17638](https://arxiv.org/abs/2609.17638)

    这套讲义系统介绍了物理信息神经网络（PINN）与神经算子的概念及其在PyTorch和NVIDIA PhysicsNeMo中的实现方法，并涵盖模型混合、傅里叶神经算子和物理信息Kolmogorov-Arnold网络（PIKANs）等前沿主题，用于解决工程、物理和石油储层等领域的实际问题。

    

    这是博尔扎诺/博岑大学2025/2026学年博士课程《物理信息神经网络》的讲义集。该课程的目标是介绍物理信息深度神经网络（PINN）和神经算子的概念，讨论如何从零开始在PyTorch中实现它们，以及如何使用专门开发的高级开源库（如NVIDIA PhysicsNeMo）来解决各种领域（工程、物理、石油储层）的实际问题。我们还讨论了近期的一些前沿主题，如模型混合、傅里叶神经算子、物理信息Kolmogorov-Arnold网络（PIKANs）等。

    arXiv:2609.17638v1 Announce Type: cross  Abstract: This is the set of lecture notes for the PhD course \href{https://www.unibz.it/en/faculties/engineering/phd-computer-science/study-course-offering/2025/36967}{\textit{Physics Informed Neural Network}, held at the University of Bozen/Bolzano} in the academic year 2025/2026.   The goal of the course was to introduce the concept of Physics Informed Deep Neural Networks (PINN) and Neural Operators (NOs), discuss their implementation from scratch in PyTorch and using advanced ad-hoc developed open-source libraries such as NVIDia PhysicsNeMo to address real-world problems in various fields (engineering, physics, petroleum reservoir). We discuss recent topics such as Mixture-of-Models, Fourier Neural Operators, Physics-Informed Kolmogorov-Arnold Networks (PIKANs) and Fourier Neural Operators.
    
[^196]: 你看不见的仍然是你要学的：一项涵盖六十个社会的预注册验证，证明证据掩蔽驱动组合泛化

    What You Can't See Is Still What You Learn: A Preregistered Sixty-Society Confirmation That Evidence Masking Drives Compositional Generalization

    [https://arxiv.org/abs/2609.17637](https://arxiv.org/abs/2609.17637)

    一项预注册的跨六十个系统配置的验证研究证实，通过证据掩蔽限制模块可见信息能显著提升系统在两步和三步组合任务上的泛化准确率。

    

    arXiv:2609.17637v1 公告类型：新 摘要：限制模块可读取的内容可能会改善系统学习计算的内容。我们在一项预注册验证中对此进行了测试，该验证包含六十个共享冻结语言模型主干并通过学习的连续数据包进行通信的四单元系统。五个实验条件变化了证据掩蔽、所有权标记以及用中性填充物替换外来证据，涵盖六个初始化集群，每个集群有两种数据顺序，在一个全新的任务世界中。在两种机制下标记均可用时，掩蔽将留出的两操作和三操作组合上的准确率提高了中位配对差异0.846和0.859；所有十二对都通过了所需的边际，完整的预注册行为标准得以通过。无标记的复制实验也通过了。没有全局可见的系统通过标记跟随检查，因此可用角色信息的效果仍未解决。填充物条件产生了七个完整泛化器，但其分解……

    arXiv:2609.17637v1 Announce Type: new  Abstract: Restricting what a module can read may improve what a system learns to compute. We test this in a preregistered confirmation with sixty four-cell systems sharing a frozen language-model backbone and communicating through learned continuous packets. Five conditions vary evidence masking, ownership markers, and replacement of foreign evidence with neutral filler, across six initialization clusters, each with two data orders, on one fresh task world. With markers available in both regimes, masking improved accuracy on held-out two- and three-operation compositions by median paired differences of 0.846 and 0.859; all twelve pairs cleared the required margins, and the full preregistered behavioral criterion passed. The unmarked replication also passed. No globally visible system passed the marker-following check, so the effect of usable role information remains unresolved. The filler condition yielded seven full generalizers, but its decompos
    
[^197]: 临床肿瘤全基因组测序的普及化：通过在消费级硬件上本地部署万亿参数大语言模型实现18小时端到端分析

    Democratizing Clinical Tumor Whole Genome Sequencing: 18-hour End-to-end Analysis via Trillion-parameter Large Language Models Locally Deployed on Consumer-grade Hardware

    [https://arxiv.org/abs/2609.17620](https://arxiv.org/abs/2609.17620)

    该研究提出完全本地化的低资源框架，在单台消费级笔记本电脑上稳定部署万亿参数生物医学大语言模型，实现18小时内完成从FASTQ原始数据到临床级报告的肿瘤全基因组测序全流程分析，准确性媲美A100集群。

    

    全基因组测序（WGS）对精准肿瘤学至关重要，但其临床应用仍受到高昂计算成本和长达数天的周转时间所限制。本工作提出了一个完全本地化的低资源框架，能够在单台配备32GB系统内存和8GB显存的消费级RTX 4060笔记本电脑，以及普通医院的常规临床工作站上，稳定部署万亿参数生物医学大语言模型，完成从原始FASTQ输入到临床级全变异谱报告输出的完整肿瘤配对全基因组测序工作流程。在标准30X测序深度配置下，该实现在18小时内即可完成单次肿瘤配对全基因组测序分析，体细胞变异检测F1分数达到99.62%，与工业标准的A100集群流程一致性超过99.9%，完全满足临床肿瘤学的准确性要求。定量分析显示自适应异构内存调度……（摘要原文在此处截断）

    arXiv:2609.17620v1 Announce Type: cross  Abstract: Whole genome sequencing (WGS) is essential for precision oncology, yet its clinical adoption remains limited by prohibitive computational costs and multi-day turnaround times. This work presents a fully localized low-resource framework enabling stable deployment of a trillion-parameter biomedical LLM on a single consumer-grade RTX 4060 laptop with 32GB system memory and 8GB VRAM, as well as on routine clinical workstations in general hospitals, completing the entire tumor-paired WGS workflow from raw FASTQ input to clinical-grade full-variation-spectrum report output. Under standard 30X depth configurations, our implementation finishes a single tumor-paired WGS analysis within 18 hours, achieving 99.62% F1 score for somatic variant detection with over 99.9% concordance to the industrial-standard A100 cluster pipeline, fully meeting clinical oncology accuracy requirements. Quantitative profiling shows adaptive heterogeneous memory sched
    
[^198]: 样条KAN中的稳定性约束逼近：精确层平衡与预算兼容饱和

    Stability-Constrained Approximation in Spline KANs: Exact Layer Balancing and Budget-Compatible Saturation

    [https://arxiv.org/abs/2609.17619](https://arxiv.org/abs/2609.17619)

    该论文在严格逐层Lipschitz预算约束下研究深度样条KAN的逼近理论，精确求解了有限深度对角层平衡问题（给出最优层预算的闭式解与单遍最小化算法），并提出了保持预算约束的构造性样条离散化定理。

    

    深度样条叠加网络在逼近阶与跨深度稳定性之间存在固有的张力。我们研究在严格逐层Lipschitz预算约束下的逼近问题，并围绕两个量来组织这一研究：给定深度分解的分解稳定性复杂度，以及离散化算子的预算兼容逼近复杂度。首先，我们精确求解了固定非负包络矩阵链的有限深度对角平衡问题：最优的均匀层预算等于 $\|M_{L-1}\cdots M_0\|_{\infty\to\infty}^{1/L}$，对于矩形层可由一个显式的单遍最小化器达到，并对退化与非可达情形给出了完整的处理。该最优值可以任意大于网络本身的Lipschitz常数，因为转换为包络会破坏符号相消效应。其次，我们给出了一个构造性的样条离散化定理，在可控的精度下保持该预算约束。

    arXiv:2609.17619v1 Announce Type: cross  Abstract: Deep spline superposition networks face a tension between approximation order and stability across depth. We study approximation under a hard layerwise Lipschitz budget, and organise it around two quantities: the factorisation stability complexity of a given deep factorisation, and the budget-compatible approximation complexity of a discretisation operator.   First, we solve exactly the finite-depth diagonal balancing problem for a fixed chain of nonnegative envelope matrices: the optimal uniform layer budget equals $\|M_{L-1}\cdots M_0\|_{\infty\to\infty}^{1/L}$, attained by an explicit one-pass minimiser, for rectangular layers, with a complete treatment of degeneracies and non-attainment. The optimum can be arbitrarily larger than the Lipschitz constant of the network itself, because passing to envelopes destroys sign cancellation.   Second, we give a constructive spline discretisation theorem preserving the budget up to a controlle
    
[^199]: 使用机器学习对中国西南自由簧乐器葫芦丝的音色分析

    Timbre Analysis of the Hulusi, a Southwestern Chinese Free-Reed Instrument, using Machine Learning

    [https://arxiv.org/abs/2609.17612](https://arxiv.org/abs/2609.17612)

    本研究利用机器学习模型分析葫芦丝的音色特征，发现谱质心、尖锐度和分形相关维数这三种心理声学特征能够有效形成音高聚类。

    

    葫芦丝是一种发明于中国云南省的吹管乐器，近年来变得极为流行。它由吹嘴、葫芦和三根竹管组成，均装有铜制自由簧片。中间的主竹管有七个音孔。与西方手风琴或蓝调口琴等乐器不同，这种乐器的音高由管长决定，而非自由簧片的固有频率。在本研究中，使用在COMSAR框架中实现的机器学习模型来研究葫芦丝的音色特征，以对不同乐器和音高进行聚类分析。研究对测得的葫芦丝C、B、A、G、F五个音高根据七种心理声学特征进行分析，结果显示其中仅有谱质心、尖锐度和分形相关维数这三种特征能够形成音高聚类。这些音色特征被用于……（摘要内容不完整，在此处截断）

    arXiv:2609.17612v1 Announce Type: cross  Abstract: The hulusi is a wind instrument that was invented in Yunnan Province, China, and has become tremendously popular in recent years. It consists of a mouthpiece, a gourd, and three bamboo tubes, all with free reeds made of copper. The main bamboo tube in the middle has seven finger holes. In this instrument, the pipe length, not the free reed's eigenfrequency, determines the instrument's pitch, unlike, for example, with the Western accordion or the blues harp. In this study, a machine learning model implemented in the COMSAR framework (https://github.com/ifsm) was used to investigate the timbre characteristics of the \emph{hulusi} to cluster different instruments and pitches. The measured \emph{hulusi} pitches C, B, A, G, and F were analyzed according to seven psychoacoustic features, among which only the spectral centroid, sharpness, and fractal correlation dimension are shown to form pitch clusters. These timbre features were used to tr
    
[^200]: 使政治文本标度可比：17种算法的基础设施与超参数敏感性

    Making Political Text Scaling Comparable: Infrastructure and Hyperparameter Sensitivity for 17 Algorithms

    [https://arxiv.org/abs/2609.17602](https://arxiv.org/abs/2609.17602)

    本文通过涵盖17种算法、5,537次实验和约425万个立场估计的大规模比较实验，论证了政治文本理想点估计方法应被视为可配置的测量流程而非固定算法，并发现绝大多数算法的估计结果对超参数选择并不敏感。

    

    基于计算文本的理想点估计方法通常以命名的算法形式进行比较，但应用这些方法涉及众多研究者的选择，这些选择决定了如何将政治文本转化为立场估计。本文认为，CT-IPE方法更适合被理解为可配置的测量流程，而非固定的估计器。基于一项涵盖17种CT-IPE算法、5,537次实验运行以及约425万个左右政治立场估计的大规模比较实验，作者描述了使这些异构方法能够联合执行的基础设施，并量化了其估计结果对替代性超参数选择的敏感性。方差分解和基于SHAP的敏感性分析表明，对于大多数算法而言，超参数配置通过共同偏移所解释的残差方差很少：17种算法中有13种的ICC值低于0.10。

    arXiv:2609.17602v1 Announce Type: new  Abstract: Computational text-based ideal point estimation (CT-IPE) methods are usually compared as named algorithms, yet applying them involves numerous researcher choices that configure how political text is turned into position estimates. This paper argues that CT-IPE methods are better understood as configurable measurement pipelines than as fixed estimators. Building on a large-scale comparative experiment spanning 17 CT-IPE algorithms, 5,537 experimental runs, and approximately 4.25 million left-right position estimates, I describe the shared infrastructure that makes these heterogeneous methods jointly executable and quantify how sensitive their estimates are to alternative hyperparameter choices. Variance-partitioning and SHAP-based sensitivity analyses show that, for most algorithms, hyperparameter profiles explain little residual variance through a shared shift: 13 of the 17 algorithms exhibit ICC values below .10. Where this profile-leve
    
[^201]: 动态网络上的去中心化最优均衡学习

    Decentralized Optimal Equilibrium Learning Over Dynamic Networks

    [https://arxiv.org/abs/2609.17601](https://arxiv.org/abs/2609.17601)

    该论文提出了一种适用于动态通信网络的去中心化最优均衡学习算法，智能体通过交换带时间戳的表格数据与时间多数重建机制学习社会最优均衡，并在功利主义和比例公平社会福利目标下实现了有限时间对数遗憾保证。

    

    本文研究了动态通信网络上有限标准型博弈中社会最优均衡的去中心化学习问题。每个智能体只能观测到自身实现的收益，对博弈本身没有先验知识，且只能使用低带宽消息与随时间变化的邻居进行通信。我们提出了网络化去中心化最优均衡学习动力学，其中智能体从局部收益比较中生成随机化的语义满足/不满信号，并交换带时间戳的时间堆叠表格，而非原始动作、收益信息或局部估计/参数。该方法将表格融合与时间多数重建相结合，以应对动态通信的挑战，同时保持完全去中心化的运行。我们在功利主义和比例公平两种社会福利目标下，为最优均衡选择建立了带有同相探索扰动的有限时间对数遗憾保证。

    arXiv:2609.17601v1 Announce Type: cross  Abstract: This paper studies decentralized learning of socially optimal equilibria in finite normal-form games over dynamic communication networks. Each agent observes only its own realized payoffs, does not know the game a priori, and can communicate only with time-varying neighbors using low-bandwidth messages. We propose networked decentralized optimal equilibrium learning dynamics in which agents generate randomized semantic content/discontent signals from local payoff comparisons and exchange time-stamped time-stacked tables rather than raw actions, payoff information or local estimates/parameters. The method combines table fusion with temporal majority reconstruction to mitigate dynamic communication while preserving fully decentralized operation. We establish finite-time logarithmic regret guarantees, with an in-phase exploration perturbation, for optimal equilibrium selection under utilitarian and proportional-fair social welfare objecti
    
[^202]: 结构并非机制：文本与基因组基础模型中的高增益门控FFN行

    Structure is not mechanism: high-gain gated-FFN rows across text and genomic foundation models

    [https://arxiv.org/abs/2609.17599](https://arxiv.org/abs/2609.17599)

    该研究通过对文本和基因组基础模型中高增益门控FFN行的分析发现，尽管这些结构普遍存在且功能富集，但其几何特性（如谱集中度、算子幅度）并不能预测因果效应，证明“结构极端性”并非可迁移的功能机制。

    

    少数异常高增益的参数可以在Transformer语言模型中产生不成比例的影响，但类似的结构是否也出现在基因组基础模型中，以及结构几何特性是否决定功能重要性，仍属未知。我们分析了文本和基因组基础模型中门控前馈网络的高增益行，包括对22个冻结模型进行的因果普查。通过精确计算相关的双线性权重算子（而非采用对角近似），我们检验了结构极端性是否是一种可迁移的机制。激活衍生出的候选行相对于随机对照和同层最高范数对照表现出功能富集，但无论是谱集中度还是算子幅度都无法预测因果效应的大小，并且这些关联在同质端点的文本解码器子集内消失了。在一个基因组解码器和一个文本解码器中对36行进行的层内扫描将这一结果解析为两种……

    arXiv:2609.17599v1 Announce Type: cross  Abstract: A small number of unusually high-gain parameters can exert disproportionate effects in transformer language models, but whether analogous structures recur in genomic foundation models and whether structural geometry determines functional importance remains unknown. We analyzed high-gain rows in gated feed-forward networks across text and genomic foundation models, including a frozen 22-model causal census. Computing an associated bilinear weight operator exactly, without a diagonal approximation, we tested whether structural extremeness is a transferable mechanism. Activation-derived candidates were functionally enriched relative to random and top-norm same-layer controls, yet neither spectral concentration nor operator magnitude predicted causal effect size, and these associations vanished within the endpoint-homogeneous text-decoder subset. A within-layer sweep of 36 rows in one genomic and one text decoder resolved this into two reg
    
[^203]: 无先验的改进型老虎机竞争比：尺度、曲率与时间范围皆可免费获得，但在噪声下无法同时兼得

    Prior-Free Competitive Ratios for Improving Bandits: Scale, Curvature and Horizon Are Free, but Not Jointly Under Noise

    [https://arxiv.org/abs/2609.17595](https://arxiv.org/abs/2609.17595)

    本文证明在改进型多臂老虎机问题中，一个简单的“探测-承诺”算法无需尺度先验即可达到 4√3·√k 的竞争比（消除了此前已知的对数因子），并确定了任意时间范围（包括未知时间范围）下的最优竞争比 Θ(√k + k/T)。

    

    在改进型多臂老虎机问题中，k个臂中的每一个都拥有未知的非递减、离散凹的奖励曲线 f_i，第 t 次拉臂 i 产生收益 f_i(t)。对于足够长的时间范围，Blum 和 Ravichandran（ALT 2025）证明：当最优臂的尺度 m=f*(T) 已知时（T≥2k），随机算法可取得相对于最佳单臂的 O(√k) 近似；当尺度未知时（T>4k），近似比为 O(√k log k)，而对应的下界为 Ω(√k)。本文表明对数因子是完全不必要的：一个仅需一页篇幅描述的“探测-承诺”算法在 T≥2⌊√k⌋ 时即可实现 4√3·√k 的竞争比，且无需任何尺度知识；我们进一步确定了任意时间范围下的最优竞争比 Θ(√k + k/T)，即使时间范围未知时也成立。在无噪声的情况下，完全不需要任何先验：一种随机边际探测算法既无需读取尺度 m，也无需读取凹包络……（摘要在此处截断）

    arXiv:2609.17595v1 Announce Type: new  Abstract: In the improving multi-armed bandits problem, each of $k$ arms has an unknown nondecreasing, discretely concave reward curve $f_i$, and pulling arm $i$ for the $t$-th time yields $f_i(t)$. For sufficiently long horizons, Blum and Ravichandran (ALT 2025) proved that randomized algorithms achieve an $O(\sqrt k)$ approximation to the best single arm when the scale $m=f^*(T)$ of the optimal arm is known ($T\ge2k$), and $O(\sqrt k\log k)$ when it is not ($T>4k$), against an $\Omega(\sqrt k)$ lower bound. The logarithmic factor is unnecessary: a one-page \emph{probe-and-commit} algorithm achieves competitive ratio $4\sqrt3\,\sqrt k$ for $T\ge2\lfloor\sqrt k\rfloor$, without any knowledge of the scale, and we determine the optimal ratio for every horizon, $\Theta(\sqrt k+k/T)$, also for unknown horizons. Without noise, \emph{no prior is needed at all}: a random-marginal probing algorithm reading neither the scale $m$, nor the concavity-envelope
    
[^204]: 当梯度感知秩：训练矩阵记忆中的可证明必要性、因果招募与组合性

    When the Gradient Sees Rank: Provable Necessity, Causal Recruitment, and Composition in Trained Matrix Memories

    [https://arxiv.org/abs/2609.17594](https://arxiv.org/abs/2609.17594)

    本文证明了基于梯度的训练能够学习到矩阵记忆存储和组合关联所需的最小秩——学习到的有效秩随键值绑定数量 K 严格递增，秩上限在 k=K 附近引发恢复相变，且训练出的算子在多次自组合应用后仍保持近乎完美的恢复率。

    

    基于梯度的训练能否学习在矩阵记忆中存储和组合关联所需的秩？在我们早期的研究中，我们在一个允许秩-1解的任务上使用了矩阵增强推理器，使这一问题悬而未决。我们在 $K$ 个新的键值绑定上训练矩阵记忆，其精确线性恢复需要 $\mathrm{rank}(Z) \geq K$。一个固定的线性读出头在无法访问原始绑定的情况下查询单个矩阵状态。实验通过大于0.9的余弦相似度来衡量恢复效果，该阈值有别于数学意义上的完全相等。在测试网格中，学习到的有效秩随 $K$ 的增加而增加（$d = 16$ 时 Spearman $\rho = 1.0$）。训练时的秩上限在 $k = K$ 附近产生了恢复效果的转变：在 $d = 8$、$K = 4$ 时，秩为3时恢复率最多为0.0004，而秩为4时达到0.97。五个随机种子中的四个在训练算子经过21次自应用后仍保持至少0.9996的恢复率。在实体子空间上，……

    arXiv:2609.17594v1 Announce Type: new  Abstract: Can gradient-based training learn the rank needed to store and compose associations in a matrix memory? In our earlier study, we used a matrix-augmented reasoner on a task that admits a rank-1 solution, leaving this question open. We train matrix memories on $K$ fresh key-value bindings whose exact linear recovery requires $\mathrm{rank}(Z) \geq K$. A fixed linear readout queries a single matrix state without access to the original bindings. Experiments measure recovery by cosine similarity greater than 0.9, a threshold distinct from mathematical equality. Learned effective rank increases with $K$ across the tested grid (Spearman $\rho = 1.0$ at $d = 16$). Training-time rank caps produce a recovery transition near $k = K$: at $d = 8$, $K = 4$, rank 3 gives at most 0.0004 recovery and rank 4 gives 0.97. Four of five seeds retain at least 0.9996 recovery through 21-fold self-application of the trained operator. On the entity subspace, the 
    
[^205]: 特征零域上导数贝祖反演与多点求值之间的泛型等价性

    Generic Characteristic-Zero Equivalence Between Derivative B\'ezout Inversion and Multipoint Evaluation

    [https://arxiv.org/abs/2609.17578](https://arxiv.org/abs/2609.17578)

    该论文证明了在特征零域的泛型计算模型中，计算多项式规范化贝祖对（求解 sZ+tZ'=1）与多点多项式求值及插值问题在复杂度上等价，其核心是一种能在近线性时间内从贝祖对重构多项式的显式微分方法。

    

    设 $a_1,\ldots,a_m$ 是域 $K$ 中互不相同的元素，令 $Z(X)=\prod_{i=1}^m (X-a_i)$。我们研究计算满足 $sZ+tZ'=1$ 且 $\deg t<m$ 的唯一规范化贝祖对 $s,t$ 的算术复杂度。因此，即使当 $M_K(m)=O(m\log m)$ 时，所得到的界也是 $O(m\log^2 m)$ 而非 $O(m\log m)$。在特征为零的无限域上，我们证明：在泛型有理直线程序模型中，计算规范化贝祖对的所有系数与任意节点的多点多项式求值以及插值问题，在相差一个加性 $O(M_K(m))$ 代价的意义下是等价的。主要工具是一种显式的微分重构方法，它能够在非空扎里斯基开子集上以 $O(M_K(m))$ 次算术运算从 $(s,t)$ 中恢复出 $Z$。将这一重构方法与自动微分和转置技术相结合，即可得到上述复杂度等价性。我们进一步将 Stra……

    arXiv:2609.17578v1 Announce Type: cross  Abstract: Let $a_1,\ldots,a_m$ be distinct elements of a field $K$, and let $Z(X)=\prod_{i=1}^m (X-a_i)$. We study the arithmetic complexity of computing the unique normalized Bezout pair $s,t$ satisfying $sZ+tZ'=1$, with $\deg t<m$ polynomials over $K$. Thus, even when $M_K(m)=O(m\log m)$, the resulting bound is $O(m\log^2 m)$ rather than $O(m\log m)$.   Over an infinite field of characteristic zero, we prove that, in the generic rational straight-line-program model, computing all coefficients of the canonical Bezout pair is equivalent, up to an additive $O(M_K(m))$ cost, to arbitrary-node multipoint polynomial evaluation and to interpolation. The main ingredient is an explicit differential reconstruction that recovers $Z$ from $(s,t)$ in $O(M_K(m))$ arithmetic operations on a nonempty Zariski-open subset. Combining this reconstruction with automatic differentiation and transposition yields the complexity equivalence.   We further transfer Stra
    
[^206]: 具有移动目标的朗之万动力学的Rényi追踪界

    R\'enyi Tracking Bounds for Langevin Dynamics with Moving Targets

    [https://arxiv.org/abs/2609.17577](https://arxiv.org/abs/2609.17577)

    该论文首次建立了具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界，并将其应用于基于连续Moreau包络的非光滑采样，给出了显式的参数选择和复杂度保证。

    

    我们研究了当目标分布随时间变化时的朗之万扩散和朗之万蒙特卡洛（LMC）。在对数Sobolev不等式（LSI）的条件下，我们推导出了用于追踪当前目标的非渐近Rényi散度保证。该框架同时涵盖连续时间朗之万扩散及其离散化形式。随后，我们将这些结果应用于基于连续Moreau包络的非光滑采样方法。针对该方案，我们给出了平滑参数和步长的显式选择，以及相应的复杂度界。据我们所知，这些是首个针对具有离散目标更新的朗之万动力学的非渐近Rényi散度追踪界。

    arXiv:2609.17577v1 Announce Type: cross  Abstract: We study Langevin diffusion and Langevin Monte Carlo (LMC) when the target distribution changes over time. Under a log-Sobolev inequality (LSI), we derive non-asymptotic R\'enyi-divergence guarantees for tracking the current target. The framework covers continuous-time Langevin diffusion and its discretizations. We then apply the results to nonsmooth sampling based on successive Moreau envelopes. For this scheme, we give explicit choices of the smoothing parameters and step sizes, together with corresponding complexity bounds. To our knowledge, these are the first non-asymptotic R\'enyi-divergence tracking bounds for Langevin dynamics with discrete target updates.
    
[^207]: Temperon：以少三分之一的训练时间获得全天候SAM的质量

    Temperon: Full-Time SAM Quality at a Third Less Wall-Clock

    [https://arxiv.org/abs/2609.17575](https://arxiv.org/abs/2609.17575)

    提出Temperon方法：训练前43%使用普通SGD、仅将最后的余弦退火阶段交给SAM包装的Muon优化器，在多个基准上达到与全天候SAM相同的精度，同时节省约三分之一训练时间。

    

    锐度感知最小化（SAM）会使每个训练步骤的成本翻倍，但其收益主要集中在训练的最后阶段。我们研究了昂贵的训练模式应投入于何处，并提出了Temperon：在epoch预算的前43%使用普通SGD探索器，随后进行一次预定的交接，将整个最终的余弦退火阶段交给由SAM包装的Muon精炼器。在CIFAR-10/100、SVHN和Tiny ImageNet上（五个随机种子，时间报告为达到目标所需epoch数乘以经空闲GPU校准的单epoch成本），Temperon在所有数据集上均达到与最佳全天候SAM配方相当的准确率，同时在四个数据集中的三个上分别提前35%、34%和32%达到最难的共同目标，并在同等成本下比已发表的SAM+SGD配方高出一个档次。消融实验使归因十分精确：在固定其他所有条件的情况下，Muon精炼器带来+0.85个百分点的提升；而探索器的形状及其重启策略没有任何价值，我们将其从贡献中撤回。重新运行最接近的竞争对手，late-（原文在此处截断）

    arXiv:2609.17575v1 Announce Type: new  Abstract: Sharpness-aware minimization (SAM) doubles the cost of every training step, yet its benefit concentrates where training ends. We study where an expensive training mode should be spent and propose Temperon: a plain-SGD explorer for the first 43% of the epoch budget, then one scheduled hand-off that gives the entire final cosine anneal to a SAM-wrapped Muon refiner. On CIFAR-10/100, SVHN and Tiny ImageNet (five seeds, times reported as epochs-to-target times an idle-GPU-calibrated epoch cost), Temperon matches the best full-time-SAM recipe on accuracy everywhere while reaching the hardest common target 35%, 34% and 32% sooner on three of the four, and sits a tier above the published SAM+SGD recipe at level cost. Ablations make the attribution exact: the Muon refiner is worth +0.85pp with everything else fixed; the explorer's shape and its restarts are worth nothing, and we withdraw them as contributions. Re-running the closest rival, late-
    
[^208]: GroupKV：面向长上下文扩散大语言模型推理的层次化KV缓存管理

    GroupKV: Hierarchical KV Cache Management for Long-Context Diffusion LLM Inference

    [https://arxiv.org/abs/2609.17573](https://arxiv.org/abs/2609.17573)

    GroupKV针对长上下文扩散大语言模型推理提出了轻量级的层次化KV缓存管理系统，利用同一生成块内token访问高度重叠且集中的上下文区域这一特性，通过组级稀疏选择有效缓解KV缓存膨胀与卸载开销问题。

    

    扩散大语言模型正日益成为一种有前景的生成范式，与自回归解码形成互补。在长上下文场景下，KV缓存膨胀以及卸载传输开销已成为推理系统的主要瓶颈。与此同时，dLLM中周期性的全序列重计算与局部化的token更新使得KV生命周期更加动态，这增加了缓存管理和预取调度的复杂性，也让重量级的token级索引或聚类方案在解码过程中难以有效摊销成本。为应对这些挑战，我们提出了GroupKV，一个面向长上下文dLLM推理的轻量级层次化KV缓存管理系统。我们观察到，在分块解码过程中，同一生成块内的token倾向于访问高度重叠且空间上集中的上下文区域，这使得组级别的稀疏选择变得行之有效。基于这一观察……

    arXiv:2609.17573v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) are emerging as a promising generative paradigm that complements autoregressive decoding. In long-context settings, KV cache bloat and offloading transfer overhead have become primary bottlenecks in inference systems. Meanwhile, the periodic full-sequence recomputation and localized token updates in dLLMs make the KV lifecycle substantially more dynamic, complicating cache management and prefetch scheduling while making heavyweight token-level indexing or clustering schemes harder to amortize effectively during decoding.   To address these challenges, we present \textsc{GroupKV}, a lightweight hierarchical KV cache management system for long-context dLLM inference. We observe that under block-wise decoding, tokens within the same generation block tend to access highly overlapping and spatially concentrated context regions, making group-level sparse selection effective. Building on this observatio
    
[^209]: 从档案伪影中分离算法偏见：大都会博物馆档案中视觉-语言模型价值评估的受控审计

    Disentangling Algorithmic Bias from Archival Artifacts: A Controlled Audit of Vision-Language Model Valuation in Metropolitan Museum Archives

    [https://arxiv.org/abs/2609.17572](https://arxiv.org/abs/2609.17572)

    本研究建立了区分算法偏见与档案元数据混淆因素的受控审计框架，并发现CLIP视觉-语言模型在评估大都会博物馆艺术品时未表现出统计学显著的性别偏见。

    

    对视觉-语言模型（VLM）进行社会偏见审计时，需要将直接的算法价值评估差异与嵌入在档案元数据中的混淆因素区分开来。在本研究中，我们使用大都会艺术博物馆开放获取藏品中的历史艺术品元数据对对比语言-图像预训练（CLIP）模型进行了审计（共 N = 1,500 件物品；N = 743 件有署名作品：男性 n = 534，女性 n = 209；n = 618 件匿名作品）。我们建立了一个定量审计框架，通过三组语义提示词对（杰作、质量与影响力）评估零样本 CLIP 的 logit 差异分数。未经调整的评估结果显示分数高度趋同，在 OpenAI CLIP（μ_F = -0.0067 对比 μ_M = -0.0035，p = 0.1829）或 OpenCLIP（μ_F = 0.0171 对比 μ_M = 0.0237，p = 0.1224）下均未发现具有统计学显著性的主要性别效应。双单侧检验（TOST）证实了在 Cohen's d ≥ 0.25 边界内的统计等效性。

    arXiv:2609.17572v1 Announce Type: new  Abstract: Auditing vision-language models (VLMs) for societal bias requires distinguishing direct algorithmic valuation disparities from confounders embedded within archival metadata. In this study, we audit Contrastive Language-Image Pretraining (CLIP) models using historical artwork metadata from the Metropolitan Museum of Art Open Access collection (N = 1,500 total objects; N = 743 attributed works: Male n = 534, Female n = 209; n = 618 anonymous).   We establish a quantitative audit framework evaluating zero-shot CLIP logit differential scores across three semantic prompt pairs (masterpiece, quality, and influence). Unadjusted evaluations demonstrate high score convergence without a statistically significant main gender effect under OpenAI CLIP (mu_F = -0.0067 vs mu_M = -0.0035, p = 0.1829) or OpenCLIP (mu_F = 0.0171 vs mu_M = 0.0237, p = 0.1224). Two One-Sided Tests (TOST) confirm statistical equivalence across Cohen's d >= 0.25 bounds (pTOST
    
[^210]: 顿悟发生在哪里：无需模块切换的分布式效用与傅里叶重编码

    Where Grokking Happens: Distributed Utility and Fourier Recoding Without a Module Switch

    [https://arxiv.org/abs/2609.17571](https://arxiv.org/abs/2609.17571)

    该论文通过“转换博弈”方法发现，Transformer中的顿悟现象并非发生在特定模块的切换（如MLP转向注意力），而是以分布式方式实现——主要通过块0注意力偏置与块1 MLP等现有分布式电路的傅里叶频谱重编码来完成从记忆到泛化的转变。

    

    在Transformer中，从记忆到泛化的转变在功能上表达于何处？我们引入了“转换博弈”——与行为对齐的精确激活博弈，并配有配对的非泛化对照组——并发现了分布式效用增益，其表现为一种前瞻性的块0注意力偏置；在替换博弈中，所选的二阶模态解释了其加法对比度的67%至92%。一项不相交精确路径研究证实，在12/12配对中，块1 MLP比所有其他测试的下游路径介导了更多这些效应。而更尖锐的“MLP负责记忆、注意力负责泛化”预测反而出现反转（在记忆锚点处为-0.331比特/样本；12对中0对符合预测方向），同时路由启动、全局秩崩溃以及素数不变架构脊也均未成立。这表明此处的顿悟是现有分布式电路的频谱重编码，而非模块切换。

    arXiv:2609.17571v1 Announce Type: cross  Abstract: Where in a Transformer is the change from memorization to generalization functionally expressed? We introduce Transition Games--behavior-aligned exact activation games with paired non-generalizing controls--and find distributed utility gain with a prospective block-0 attention bias; selected degree-two modes account for 67--92% of its addition contrast across replacement games, and a disjoint exact path study confirms that block-1 MLP mediates more of their effect than all other tested downstream paths in 12/12 pairs. The sharper "MLP memorizes, attention generalizes" prediction instead reverses (-.331 bits/example at the memory anchor; 0/12 in the predicted direction), while routing onset, global rank collapse, and a prime-invariant architecture ridge also fail, identifying grokking here as spectral recoding of an existing distributed circuit rather than a module switch.
    
[^211]: 超越静态RAG：一种面向消费级GPU高效长上下文推理的自适应三指标路由框架

    Beyond Static RAG: An Adaptive, Tri-Metric Routing Framework for Efficient Long-Context Inference on Commodity GPUs

    [https://arxiv.org/abs/2609.17564](https://arxiv.org/abs/2609.17564)

    本文提出一种无需训练的三指标路由框架，利用空间复杂度、句法密度和类符-形符比三个CPU侧信号，结合显存余量等硬件物理指标，在原始、神经压缩和词法三种流水线间动态选择，解决了消费级GPU上RAG部署中的“压缩悖论”问题。

    

    在诸如NVIDIA T4（16 GB显存）等消费级GPU上部署检索增强生成（RAG）会暴露出一种我们称之为“压缩悖论”的实际失败模式：神经提示压缩会带来键值（KV）缓存竞争和预处理延迟，其开销可能超过生成阶段所节省的时间；而跳过压缩则可能在处理长上下文时导致内存溢出（OOM）故障。我们识别了在紧张内存预算下同时部署基于vLLM的大语言模型和基于PyTorch的压缩器时的两种不同失败机制，并提出了三指标路由器——一种确定性、无需训练的策略，可在原始、神经（LLMLingua-2）和词法（BM25）三种流水线之间进行选择。该路由器使用三个CPU侧信号：空间复杂度（$L$）、句法密度（$\rho_{key}$）和类符-形符比（TTR）。与以往仅基于语义的自适应方法不同，我们的调度信号是硬件物理层面的，基于显存余量和延迟交叉点。阈值设定……

    arXiv:2609.17564v1 Announce Type: new  Abstract: Deploying retrieval-augmented generation (RAG) on commodity GPUs such as the NVIDIA T4 (16 GB VRAM) exposes a practical failure mode we call the Compression Paradox: neural prompt compression can add key-value (KV) cache contention and preprocessing latency that outweigh generation-time savings, while skipping compression can cause out-of-memory (OOM) failures on long contexts. We identify two distinct failure mechanisms when a vLLM-served LLM and a PyTorch-based compressor are co-deployed under tight memory budgets, and introduce the Tri-Metric Router, a deterministic, training-free policy that selects among Raw, Neural (LLMLingua-2), and Lexical (BM25) pipelines. The router uses three CPU-side signals: spatial complexity ($L$), syntactic density ($\rho_{key}$), and type-token ratio (TTR). Unlike prior semantic-only adaptation, our dispatch signal is hardware-physical, based on VRAM headroom and a latency crossover point. Thresholds are
    
[^212]: BLADE：面向事件驱动目标检测的可靠动态硬件感知SNN-ANN边界选择

    BLADE: ReliaBle Dynamic Hardware-Aware SNN-ANN Boundary SeLection for Event-BAseD Object DEtection

    [https://arxiv.org/abs/2609.17562](https://arxiv.org/abs/2609.17562)

    提出BLADE，首个面向动态混合SNN-ANN网络的可靠性感知边界选择方法，通过在设计空间探索中引入分层统计故障注入，联合优化SNN-ANN边界与ANN早退配置，在可靠性、精度、执行时间和能耗之间取得平衡，并在事件驱动目标检测中实现0.691的mAP 0.5。

    

    混合脉冲神经网络（SNN）-人工神经网络（ANN）架构将SNN的能效优势与ANN在事件驱动目标检测中卓越的检测精度相结合。然而，现有的混合SNN-ANN网络采用静态推理，并且主要依据精度和能耗来选择SNN-ANN边界，而未考虑动态推理或可靠性。本文提出了BLADE，这是首个针对具有ANN早退机制的动态混合SNN-ANN网络的可靠性感知边界选择方法。所提出的框架根据可靠性、检测精度、执行时间和能耗，联合优化SNN-ANN边界和ANN早退配置，并在设计空间探索过程中通过分层统计故障注入的方式将可靠性纳入考量。在事件驱动目标检测器上的实验评估实现了0.691的mAP 0.5，同时降低了……

    arXiv:2609.17562v1 Announce Type: cross  Abstract: Hybrid Spiking Neural Network (SNN)-Artificial Neural Network (ANN) architectures combine the energy efficiency of SNNs with the superior detection accuracy of ANNs for event-based object detection. Existing hybrid SNN--ANN networks, however, employ static inference and select the SNN-ANN boundary primarily according to accuracy and energy consumption, without considering dynamic inference or reliability. This paper presents BLADE, the first reliability-aware boundary selection methodology for dynamic hybrid SNN-ANN networks with ANN early exit. The proposed framework jointly optimizes the SNN-ANN boundary and ANN early-exit configuration according to reliability, detection accuracy, execution time, and energy consumption, while incorporating reliability through hierarchical statistical fault injection during design-space exploration. Experimental evaluation on an event-based object detector achieves an mAP 0.5 of 0.691 while reducing 
    
[^213]: 仅为分歧付费：具有匹配标签复杂度界的模型更新无回归认证判定

    Pay Only for Disagreement: Certified No-Regression Verdicts for Model Updates with Matching Label-Complexity Bounds

    [https://arxiv.org/abs/2609.17560](https://arxiv.org/abs/2609.17560)

    论文提出DISCERN协议，利用“模型间风险差仅存在于分歧输入上且无需标签即可观测”这一关键性质，通过零标签层与仅标注分歧样本的审计层两层序贯协议，为模型更新提供无回归认证，并证明标签复杂度为rho²/ε²，相比不考虑配对关系的审计器可节省1/ρ的标注成本。

    

    每个生产模型都会被更新——通过重新训练、微调、量化或静默的供应商替换——而每次更新都存在比其替代模型表现更差的风险。我们将模型更新晋升问题形式化为认证的成对风险差审计。我们的出发点是一个支撑恒等式：两个模型之间的风险差存在于它们产生分歧的输入上，而这些输入无需标签即可观测。我们构建了DISCERN，一个序贯两层协议。零标签层仅通过无标签流量即可认证分歧率低于容差的良性更新。审计层通过anytime-valid置信序列仅对采样出的分歧样本进行标注，该序列在每个停止时刻以及任何标签路由规则下均有效，即使是对抗性的评判者也不例外。我们证明了有限样本有效性以及匹配的标签复杂度界，在速率层面为rho^2/eps^2量级，因此利用免费的分歧信息可证明地比任何不考虑配对关系的审计器节省1/rho的因子。

    arXiv:2609.17560v1 Announce Type: cross  Abstract: Every production model is updated, by retraining, fine-tuning, quantization, or a silent vendor swap, and each update risks being worse than what it replaced. We formalize update promotion as certified paired risk-difference auditing. Our starting point is a support identity: the risk difference between two models lives on the inputs where they disagree, observable without labels. We build DISCERN, a sequential two-tier protocol. A zero-label tier certifies benign updates whose disagreement rate is below tolerance from unlabeled traffic alone. An audited tier labels only sampled disagreements through an anytime-valid confidence sequence, valid at every stopping time and under any label-routing rule, even an adversarial judge. We prove finite-sample validity and matching label-complexity bounds of order rho^2/eps^2 at the rate level, so exploiting free disagreement provably saves a factor 1/rho over any pairing-blind auditor, and the gu
    
[^214]: WARD：面向可靠边缘AI的运行时工作负载自适应视觉Transformer框架

    WARD: Runtime Workload-Adaptive Vision TRansformer Framework for Dependable Edge AI

    [https://arxiv.org/abs/2609.17556](https://arxiv.org/abs/2609.17556)

    WARD是一个运行时自适应的视觉Transformer框架，通过通道级子网络划分、可靠性感知的持续学习和动态运行模式调度，在动态变化的边缘环境中联合优化性能、容错能力和适应能力。

    

    边缘部署的AI系统在动态变化的功耗预算、可靠性需求和输入分布下运行，需要持续适应。这些条件出现在长时间运行的边缘AI应用中，包括自主系统、工业监测和卫星在轨智能。现有的容错方法假设静态运行条件，而持续学习技术忽略了在线适应过程中的并发硬件故障。此外，在可编程AI加速器上实际部署运行时自适应可靠性框架在很大程度上仍未被探索。本文提出了WARD，这是一个运行时自适应的视觉Transformer框架，它结合了通道级子网络划分、可靠性感知的持续学习和动态运行模式调度，以根据运行时条件联合优化性能、容错能力和适应能力。两个物理隔离的子网络执行……

    arXiv:2609.17556v1 Announce Type: cross  Abstract: Edge-deployed AI operate under dynamically changing power budgets, reliability requirements, and input distributions, requiring continuous adaptation. Such conditions arise in long-running edge AI applications, including autonomous systems, industrial monitoring, and satellite onboard intelligence. Existing fault-tolerant methods assume static operating conditions, whereas continual learning techniques neglect concurrent hardware faults during online adaptation. Moreover, the practical deployment of runtime-adaptive reliability frameworks on programmable AI accelerators remains largely unexplored.   This paper presents WARD, a runtime-adaptive Vision Transformer framework that combines channel-wise subnetwork partitioning, reliability-aware continual learning, and dynamic operating-mode scheduling to jointly optimize performance, fault tolerance, and adaptation according to runtime conditions. Two physically isolated subnetworks execut
    
[^215]: REQAP：面向边缘DNN加速的弹性权重打包与量化

    REQAP: Resilient Weight Packing and Quantization for Edge DNN Acceleration

    [https://arxiv.org/abs/2609.17555](https://arxiv.org/abs/2609.17555)

    本文提出REQAP方法，通过敏感度驱动的混合精度量化、实现SWAR并行执行的确定性寄存器级权重打包，以及将关键层MSB复制到空闲寄存器空间的选择性位级保护，在边缘DNN加速器上同时实现了高效模型压缩与硬件容错能力。

    

    深度神经网络（DNN）在边缘加速器上的高效部署要求在进行激进模型压缩的同时，在易受故障影响的硬件环境中保持可靠性。本文提出了一种面向脉动阵列式DNN加速器的可靠性感知量化权重打包方法。一个由敏感度驱动的混合精度量化框架根据各层对精度的影响分配层级位宽，同时强制权重与激活值之间保持对称精度。一种确定性的寄存器级打包策略将多个异构操作数对整合到固定宽度的寄存器字中，实现寄存器内SIMD（SWAR）风格的并行执行，从而同时减少内存占用和执行周期。为提高对硬件故障的抵御能力，选择性位级保护将关键层的最高有效位（MSB）复制到未使用的寄存器空间中，以最小的开销实现类似TMR（三模冗余）的保护。

    arXiv:2609.17555v1 Announce Type: cross  Abstract: Efficient deployment of Deep Neural Networks (DNNs) on edge accelerators requires aggressive model compression while maintaining reliability in fault-prone hardware environments. This paper presents a reliability-aware quantized weight packing methodology for systolic-array-based DNN accelerators. A sensitivity-driven mixed-precision quantization framework assigns layer-wise bit-widths according to accuracy impact while enforcing symmetric precision between weights and activations. A deterministic register-level packing strategy consolidates multiple heterogeneous operand pairs into fixed-width register words, enabling SIMD-within-a-register (SWAR) style parallel execution that reduces both memory footprint and execution cycles. To improve resilience against hardware faults, selective bit-level protection replicates the most significant bits (MSBs) of critical layers into unused register space, achieving TMR-style protection with minim
    
[^216]: 两个小型LLM中不存在可用的线性“屈服方向”：激活转向声明的验证协议，以及反驳压力下谄媚行为的跨模型家族行为研究

    No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback

    [https://arxiv.org/abs/2609.17550](https://arxiv.org/abs/2609.17550)

    该研究通过验证协议发现两个小型LLM中不存在可用的线性“屈服方向”，并首次跨模型家族揭示：模型在反驳下放弃正确答案的谄媚比例高达41.8%-43.1%，且哪种反驳方式有效及失败模式均强烈依赖于具体模型家族。

    

    语言模型在用户反驳时经常放弃正确答案。我们在来自不同家族的两个小型指令微调模型Qwen2.5-1.5B和Llama-3.2-1B上，基于TriviaQA研究了这一现象：模型先给出答案，随后受到四种预设反驳风格之一的质疑，然后再次作答。在初始答案正确的条件下，两个模型分别在41.8%和43.1%的情况下转向错误答案。哪种压力有效是模型的属性而非压力本身的属性：同样的问题内配对比较（单纯质疑vs.情感诉求）事先预设，结果在两个模型家族中呈现方向相反的Bonferroni显著差异（Qwen：单纯质疑>情感诉求，OR 2.5，p=.040；Llama：情感诉求>单纯质疑，OR 4.0，p=.001）。失败模式也依赖于具体模型：Llama放弃答案且不重新承诺的比率是Qwen的六倍（8.2% vs. 1.4%）。相同的反驳仅在约13%的情况下修复初始错误答案；反驳在认知上整体是净负面的

    arXiv:2609.17550v1 Announce Type: new  Abstract: Language models frequently abandon correct answers when users push back. We study this in two small instruction-tuned models from different families, Qwen2.5-1.5B and Llama-3.2-1B, over TriviaQA: the model answers, is challenged with one of four scripted pushback styles, and answers again. Conditioned on an initially correct answer, the models flip to a wrong answer in 41.8% and 43.1% of episodes. Which pressure works is a property of the model, not the pressure: the same within-question paired comparison (bare doubt vs. emotional appeal), specified in advance, is Bonferroni-significant in opposite directions across families (Qwen: bare doubt > emotional, OR 2.5, p=.040; Llama: emotional > bare doubt, OR 4.0, p=.001). Failure mode is also model-dependent: Llama abandons answers without recommitting at six times Qwen's rate (8.2% vs. 1.4%). Identical pushback repairs initially wrong answers only ~13% of the time; pushback is net epistemic
    
[^217]: 利用大语言模型从呼吸治疗临床笔记中提取的特征增强拔管失败预测

    Enhancing Extubation Failure Prediction with LLM-Derived Features from Respiratory Therapy Clinical Notes

    [https://arxiv.org/abs/2609.17532](https://arxiv.org/abs/2609.17532)

    该论文提出利用大语言模型从自由文本呼吸治疗临床笔记中提取特征，与结构化数据结合后显著提升了拔管失败预测性能，并揭示了既往研究在目标人群定义上的差异如何阻碍模型的泛化能力。

    

    有创机械通气是一种挽救生命的治疗手段，但及时、安全地停用对于预防拔管失败（EF）及其相关健康风险至关重要。我们提出了一种新颖的拔管失败预测方法，该方法利用大语言模型和逻辑回归流程，从自由文本呼吸治疗记录中分类提取特征。将该方法应用于华盛顿大学医学院的患者队列，我们的方法识别出了具有临床意义的拔管失败相关特征，这些特征与结构化患者数据结合使用时，能够提升拔管失败预测性能。我们进一步强调了既往拔管失败预测研究中目标人群的差异（例如异质性的纳入标准和拔管失败定义）如何导致模型性能的系统性差异，并阻碍研究结果之间的泛化能力。

    arXiv:2609.17532v1 Announce Type: new  Abstract: Invasive mechanical ventilation is a lifesaving therapy, but timely, safe discontinuation is essential to preventing extubation failure (EF) and related risks to health. We present a novel approach to EF prediction that leverages features classified in free-text respiratory therapy notes using a large language model and logistic regression pipeline. Applied to a patient cohort from University of Washington Medicine, our method identifies clinically meaningful EF-related features that improve EF prediction performance when included alongside structured patient data. We further highlight how differences in target populations in prior EF prediction studies, such as heterogenous inclusion criteria and EF definition, can lead to systematic differences in model performance and hinder generalizability between studies.
    
[^218]: 面向目标的概率预测用于5G网络中的动态物理资源块（PRB）分配

    Goal-oriented probabilistic forecasting for dynamic PRB allocation in 5G networks

    [https://arxiv.org/abs/2609.17297](https://arxiv.org/abs/2609.17297)

    该论文提出一种面向目标的概率预测框架，利用Pinball损失函数训练DeepAR和TFT模型，并根据运营商成本矩阵确定最优分配分位数，从而在5G网络动态PRB分配中降低运营成本，实现服务可靠性与资源效率的平衡。

    

    5G网络中高效的物理资源块（PRB）分配需要准确的需求预测。传统方法最小化对称误差指标（MAE、RMSE），忽略了运营成本的不对称性——即资源供给不足（导致服务降级）的代价远高于资源过度供给（造成容量浪费）。我们提出了一种面向目标的概率预测框架，使模型训练与运营商的决策目标保持一致。具体而言，我们使用Pinball损失函数训练DeepAR和时间融合Transformer（TFT）模型，并根据运营商的成本矩阵推导出最优分配分位数。在真实波束级5G流量数据集上的评估表明，与基于MSE训练的基线方法相比，所提出的方法在保持校准良好的不确定性估计的同时降低了运营成本。该框架能够实现动态PRB分配，明确平衡服务可靠性与资源效率。

    arXiv:2609.17297v1 Announce Type: cross  Abstract: Efficient physical resource block (PRB) allocation in 5G networks requires accurate demand forecasting. Conventional methods minimize symmetric error metrics (MAE, RMSE), ignoring the operational cost asymmetry where under-provisioning (service degradation) is far costlier than over-provisioning (wasted capacity). We propose a goal-oriented probabilistic forecasting framework that aligns model training with the operator's decision-making objectives. Specifically, we train DeepAR and Temporal Fusion Transformer (TFT) models using the Pinball Loss function and derive the optimal allocation quantile from the operator's cost matrix. Evaluation on a real beam-level 5G traffic dataset shows that the proposed approach reduces operational cost compared to MSE-trained baselines while maintaining calibrated uncertainty estimates. The framework enables dynamic PRB allocation that explicitly balances service reliability against resource efficiency
    
[^219]: 基于自适应导数阶随机解释的全局与局部可解释性统一框架

    A unified framework for global and local interpretability using adaptive derivative-ordered random explanation

    [https://arxiv.org/abs/2609.17171](https://arxiv.org/abs/2609.17171)

    本文提出ADORE方法，利用一阶和二阶导数在统一分析框架内同时实现全局特征重要性与局部样本贡献的可解释性分析，有效捕捉非线性特征-样本交互并精确量化特征影响。

    

    复杂机器学习模型的可解释性至关重要，尤其是在医疗健康和金融等现实世界的高风险领域。然而，现有的事后可解释性方法存在固有的局限性：分析过程碎片化、对非线性特征交互的建模能力不足、计算效率低下，以及过度依赖特定的模型架构。为了应对这些挑战，本文提出了一种新方法——自适应导数阶随机解释（ADORE），该方法利用一阶和二阶导数来适应非线性模型的复杂性，同时能够在统一的分析框架内有效捕捉特征与样本之间的交互。ADORE 将全局特征重要性与局部样本贡献相结合，通过同时捕捉影响的大小和方向来精确量化特征影响，并识别影响模型的关键样本。

    arXiv:2609.17171v1 Announce Type: cross  Abstract: The interpretability of complex machine learning models is of paramount importance, especially in real-world high-stakes domains such as healthcare and finance. However, existing post-hoc interpretability methods suffer from inherent limitations: fragmented analytical processes, inadequate capacity to model nonlinear feature interactions, computational inefficiencies, and over-reliance on specific model architectures. To address these challenges, this paper provides a novel method - Adaptive Derivative-Ordered Random Explanation (ADORE) - that leverages first- and second-order derivatives to accommodate nonlinear model complexities, while enabling effective capture of feature-sample interactions within a unified analytical framework. ADORE integrates global feature importance with local sample contributions, precisely quantifying feature impact by capturing both magnitude and direction, and identifying critical samples influencing mode
    
[^220]: 高容量核联想记忆中稳定性边缘的信息几何自组织

    Information Geometric Self-Organization at the Edge of Stability in High-Capacity Kernel Associative Memories

    [https://arxiv.org/abs/2609.16827](https://arxiv.org/abs/2609.16827)

    本文通过Hessian特征值谱分析揭示了KLR联想记忆中“优化脊”本质上是秩1谱坍缩附近的几何奇点，并证明梯度下降的学习动力学在稳定性边缘处表现出瞬态自稳定行为，从而自发地组织到该最优区域。

    

    基于核逻辑回归（KLR）的高容量联想记忆展现出卓越的存储能力与鲁棒性。先前的实证研究识别出了一个超参数区域，即“优化脊”，在该区域中吸引子的稳定性达到最大。然而，这一区域的几何本质以及到达该区域所需的优化动力学机制一直不明确。本文研究了采用KLR训练的Hopfield网络中参数空间的静态几何以及梯度下降（GD）的学习轨迹。利用Hessian矩阵的特征值谱，我们揭示了“优化脊”对应于位于秩1谱坍缩附近的一个相边界，它作为一个几何奇点，其主曲率被大幅放大。此外，我们证明了学习动力学表现出一种由稳定性边缘现象驱动的瞬态自稳定行为……

    arXiv:2609.16827v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit exceptional storage capabilities and robustness. Previous empirical studies identified a hyperparameter regime, the "Ridge of Optimization," where attractor stability is maximized. However, the geometric nature of this regime and the optimization dynamics required to reach it have remained unclear. In this paper, we investigate the static geometry of the parameter space and the learning trajectory of Gradient Descent (GD) in KLR-trained Hopfield networks. Using the eigenvalue spectrum of the Hessian, we reveal that the Ridge corresponds to a phase boundary located adjacent to a rank-1 spectral collapse, acting as a geometric singularity where the principal curvature is massively amplified. Furthermore, we demonstrate that the learning dynamics exhibit a transient self-stabilizing behavior driven by the Edge of Stability (EoS) phenomenon. Rather than seek
    
[^221]: 学习动力学的几何学：梯度下降与自然梯度在优化脊上的对比

    Geometry of learning dynamics: Gradient descent versus natural gradient on the ridge of optimization

    [https://arxiv.org/abs/2609.16805](https://arxiv.org/abs/2609.16805)

    论文通过对KLR训练的Hopfield网络统计流形的几何分析，揭示了优化脊上的学习分为两个阶段：标准梯度下降因极端曲率沿振荡的非测地线路径前进，而自然梯度下降遵循理想测地线路径并完全克服了这些不稳定性。

    

    基于核逻辑回归（KLR）的高容量联想记忆呈现出一种“优化脊”特征，表现为极端的稳定性和高度偏斜的权重谱。然而，学习收敛到这一临界区域的动力学过程一直不清楚。本文对KLR训练的Hopfield网络在统计流形上的学习轨迹进行了几何分析。通过比较梯度下降（GD）与自然梯度下降（NGD）的路径，我们阐明了支配优化过程的机制。我们的分析揭示，在优化脊上的学习分为两个明显不同的阶段进行。我们证明了优化脊的极端曲率导致标准GD沿着高度振荡的非测地线路径前进。与之形成鲜明对比的是，NGD明确地校正了这种几何结构，遵循理想的测地线路径，完全克服了GD所面临的不稳定性。我们通过实验证明

    arXiv:2609.16805v1 Announce Type: new  Abstract: High-capacity associative memories based on Kernel Logistic Regression (KLR) exhibit a "Ridge of Optimization" characterized by extreme stability and a highly skewed weight spectrum. However, the dynamical process by which learning converges to this critical regime has remained unclear. This paper provides a geometric analysis of the learning trajectories on the statistical manifold of a KLR-trained Hopfield network. By comparing the paths of Gradient Descent (GD) and Natural Gradient Descent (NGD), we elucidate the mechanisms governing the optimization process. Our analysis reveals that learning on the Ridge proceeds in two distinct phases. We show that the extreme curvature of the Ridge causes standard GD to follow a highly oscillatory, non-geodesic path. In stark contrast, NGD explicitly corrects for this geometry, following the ideal geodesic path and completely overcoming the instabilities faced by GD. We demonstrate experimentally 
    
[^222]: 从未存在的潜变量：对动作分块Transformer中CVAE消融实验的取证式重跑

    The Latent That Never Was: A Forensic Re-run of the CVAE Ablation in Action Chunking Transformer

    [https://arxiv.org/abs/2609.16745](https://arxiv.org/abs/2609.16745)

    该论文在原始代码中取证式地重跑了ACT的CVAE编码器消融实验，发现原始论文中移除编码器导致成功率从35%骤降至2%的结果无法重现，且训练时长和检查点选择方式都可能逆转策略间的表现排序。

    

    动作分块Transformer（ACT）被广泛用于从人类演示中学习机器人操作。其条件变分自编码器包含一个编码器，旨在捕捉训练期间不同演示之间的差异。原始ACT论文报告称，移除该编码器会使两个基于人类演示的模拟任务的平均成功率从35%降至2%。我们在原始代码中重新运行了这一消融实验，并检验该结论是否依赖于具体实现或训练数据。已发表的成功率下降在我们的测试中并未重现，尽管较小的成功率增益或损失仍无法确定。为调查这一差异，我们改变了训练时长以及用于评估的检查点选择方式。两者都可能逆转哪个策略得分更高，但已发表的下降原因仍不清楚。仅凭成功率无法确定编码器是否提供了有助于策略重建演示动作的信息。

    arXiv:2609.16745v1 Announce Type: cross  Abstract: Action Chunking Transformers (ACT) are widely used to learn robot manipulation from demonstrations. Their conditional variational autoencoder includes an encoder meant to capture differences between demonstrations during training. The original ACT paper reported that encoder removal dropped the mean success rate from 35% to 2% on two simulated tasks with human demonstrations. We re-ran this ablation in the original code and checked whether the findings depend on the implementation or training data. The published drop does not reappear in our tests, although smaller gains or losses in success rate remain uncertain. To investigate the discrepancy, we varied training length and how checkpoints are selected for evaluation. Both can reverse which policy scores higher, but the published drop's cause remains unknown. Success rates alone leave open whether the encoder provides information that helps the policy reconstruct demonstrated actions.
    
[^223]: 看见重要之物：视觉线索引导的视频规划实现可泛化的机器人导航

    Seeing What Matters: Visual Cue Guided Video Planning for Generalizable Robot Navigation

    [https://arxiv.org/abs/2609.16737](https://arxiv.org/abs/2609.16737)

    CueNav通过鸟瞰图和机器人身体等视觉线索引导视频规划，并利用逆动力学模型将视频计划转换为机器人动作，使迷宫导航成功率提升近2倍。

    

    生成式视频模型可以通过预测未来观测作为视频规划，成为机器人导航的有前景的骨干网络。近期的方法通常将视频规划建立在短时程引导之上，并通过场景重建来恢复几何路径点，而对更长时程的规划以及精确的视频到动作转换探索较少。我们提出了CueNav，一个基于视频模型的导航框架，它将视觉线索引导的视频规划与特定具身形态的逆动力学模型（IDM）相结合。作为视觉线索，我们使用鸟瞰图（BEV）地图来传达全局任务上下文，并在自我中心观测中保留部分机器人身体以呈现具身形态上下文。这些线索引导视频规划器，而IDM则将从视频规划中提取的密集光流场转换为机器人动作。凭借视觉线索编码的全局任务上下文，CueNav在迷宫导航任务中的成功率比不使用该线索的规划方法高出近2倍。

    arXiv:2609.16737v1 Announce Type: cross  Abstract: Generative video models can serve as a promising backbone for robot navigation by predicting future observations as video plans. Recent approaches often condition video planning on short-horizon guidance and recover geometric waypoints through scene reconstruction, leaving longer-horizon planning and precise video-to-action translation less explored. We present CueNav, a video model-based navigation framework combining visual cue guided video planning with an embodiment-specific Inverse-Dynamics Model (IDM). As visual cues, we use a Bird's-Eye View (BEV) map to convey global task context and retain part of the robot body in the egocentric observation to expose embodiment context. These cues guide the video planner, while the IDM translates dense flow fields extracted from the video plan into robot actions. With the visual cue encoding global task context, CueNav achieves nearly 2x higher success in maze navigation than planning without
    
[^224]: 面向AI安全的数学：致数学家的邀请

    Math for AI safety: an invitation for mathematicians

    [https://arxiv.org/abs/2609.15289](https://arxiv.org/abs/2609.15289)

    本文按数学领域（逻辑与博弈论、概率论、代数与表示论、分析与几何）向数学家发出邀请，介绍AI安全研究中各领域可贡献的方向，并为每个领域提供一个无需AI背景即可入手的开放问题。

    

    人工智能有可能超出人类的理解与控制范围。要设计出清晰可读、可引导、并与人类协同合作的AI，需要新的数学。本文按数学领域组织了这篇邀请，你可以直接翻到自己熟悉的方向：用于合作研究的逻辑与博弈论；用于智能体与世界模型的概率论；用于学习特征的代数与表示论；用于泛化与训练动力学的分析与几何。每一节末尾都给出一个开放性问题，任何没有AI安全经验背景的职业数学家都可以着手研究。

    arXiv:2609.15289v1 Announce Type: cross  Abstract: Artificial intelligence threatens to outrun human understanding and control. New mathematics is needed to design AI that is legible, steerable, and cooperative with humanity. I organize this invitation by mathematical field, so you can turn straight to your own: logic and game theory for cooperation; probability for agency and world-models; algebra and representation theory for learned features; analysis and geometry for generalization and training dynamics. Each section ends with an open problem that is accessible to a working mathematician with no prior experience in AI safety.
    
[^225]: 弥合基于心电图的情绪识别差距：深度学习模型的统一评估

    Bridging the Gap in ECG-Based Emotion Recognition: A Unified Evaluation of Deep Learning Models

    [https://arxiv.org/abs/2609.15055](https://arxiv.org/abs/2609.15055)

    该研究针对基于心电图的情绪识别领域缺乏统一评估标准的问题，引入了ARRC和ARDT两个开源框架，对主流深度学习模型进行了强调跨数据集泛化能力的统一基准评估。

    

    深度学习已经催生了大量用于从心电图（ECG）数据进行自动情绪识别（AER）的架构，但预处理、训练和评估方法的不一致使得直接比较变得困难。大多数研究在同质条件下收集的单一数据集上训练和验证模型，这限制了数据的变异性，并引发了对模型泛化能力的担忧。虽然有时会使用跨数据集验证，但其主要评估的是模型的适应性而非真正的泛化能力。本研究对AER领域中的主流深度学习架构进行了比较分析，重点强调模型的泛化能力而非数据集适应性。为了实现这一基准测试，我们引入了两个开源框架：情感表征与分类研究框架，这是一个标准化的基准测试工具包；以及情感研究数据集工具包，这是一个用于跨数据集训练和验证的框架。通过使用ARD……（原文在此截断）

    arXiv:2609.15055v1 Announce Type: cross  Abstract: Deep learning has led to numerous proposed architectures for Automated Emotion Recognition (AER) from electrocardiogram (ECG) data, but inconsistencies in preprocessing, training, and evaluation make direct comparisons difficult. Most studies train and validate models on individual datasets collected under homogeneous conditions, limiting variability and raising concerns about generalizability. Cross-dataset validation is sometimes used but primarily assesses model adaptability rather than true generalization. This study presents a comparative analysis of prominent deep learning architectures in AER, emphasizing model generalization over dataset adaptability. To enable this benchmark, we introduce two open-source frameworks: Affective Research on Representations and Classifications (ARRC), a standardized benchmarking toolkit, and Affective Research Dataset Toolkit (ARDT), a framework for inter-dataset training and validation. Using ARD
    
[^226]: 深度玻尔兹曼机中用于统计数据融合的跨块条件化

    Cross-Block Conditioning in Deep Boltzmann Machines for Statistical Data Fusion

    [https://arxiv.org/abs/2609.14934](https://arxiv.org/abs/2609.14934)

    提出观测块多预测方法，使深度玻尔兹曼机在统计数据融合场景下（即没有任何样本同时观测两个结果）也能采用判别式准则进行训练，该方法适用于任意缺失模式，实验表明微调后的DBM优于对比方法。

    

    统计数据融合将两个共享一组协变量但观测到互不相交结果块的面板结合起来，在传统形式下，没有任何一行能同时观测到两个结果。这排除了人们更愿意用于训练深度玻尔兹曼机的判别式准则，因为多预测训练需要对所保留的内容拥有真实标签。我们提出了观测块多预测方法，将多预测目标限制为从每行实际观测到的内容中提取的目标。该方法对任何缺失模式都有良好定义，并且当行数据完整时退化为原始准则。拥有一个能在此设置下使用的判别式准则，使我们能够通过将其贡献分离为表示部分和推断部分，来探究是否根本需要联合模型。在两个消费者面板上，在覆盖样本量和协变量宽度的网格上进行了35个单元格和875次运行的实验，微调后的DBM是十五个（摘要在此处截断）……

    arXiv:2609.14934v1 Announce Type: cross  Abstract: Statistical data fusion combines two panels that share a block of covariates but observe disjoint outcome blocks, and in its traditional form no row observes both outcomes at once. That rules out the discriminative criterion one would rather train a Deep Boltzmann Machine with, since multi-prediction training needs ground truth for whatever it holds out. We propose observed-block multi-prediction, which restricts the multi-prediction objective to targets drawn from what each row actually observes. It is well defined for any missingness pattern and reduces to the original criterion when rows are complete. Having a discriminative criterion that survives the setting lets us ask whether the joint model is needed at all, by separating what it contributes into a representation part and an inference part. On two consumer panels, on grids over sample size and covariate width spanning 35 cells and 875 runs, the fine-tuned DBM is the best of fif
    
[^227]: 利用无冲突梯度解决PINNs和PIKANs的失效模式

    Tackling Failure Modes of PINNs and PIKANs Using Conflict-Free Gradients

    [https://arxiv.org/abs/2609.14841](https://arxiv.org/abs/2609.14841)

    本文提出归一化的投影梯度手术算法Norm-PCGrad，用以消除区域分解PINNs中残差、边界和界面损失项之间的梯度冲突，在二维和三维问题上达到最先进的求解精度。

    

    物理信息神经网络（PINNs）等科学机器学习方法在复杂几何域上求解偏微分方程（PDEs）时，越来越依赖区域分解以获得更好的可扩展性，然而由此产生的由残差项、边界项和界面项组成的复合损失极易受到梯度冲突的影响，从而降低训练效果。本工作将区域分解与基于投影的梯度手术相结合，以系统性地缓解二维和三维设置中的此类冲突。我们评估了两种现有的基于投影的算法——PCGrad和ConFIG，并指出它们在特定场景（例如具有多个重叠界面的三维区域）中会出现性能下降。为解决这一局限，我们提出了Norm-PCGrad，这是一种归一化变体，在一系列二维和三维区域分解问题上实现了最先进的精度。在所考虑的基准测试中，Norm-PCGrad始终……

    arXiv:2609.14841v1 Announce Type: new  Abstract: Scientific machine learning methods such as physics-informed neural networks (PINNs) increasingly rely on domain decomposition for better scalability while solving partial differential equations (PDEs) over complex geometries, yet the resulting composite loss comprising residual, boundary, and interface terms is highly susceptible to conflicting gradients that degrade training. This work bridges domain decomposition with projection-based gradient surgery to systematically mitigate such conflicts in 2D and 3D settings. We evaluate two existing projection-based algorithms, PCGrad and ConFIG, and identify their performance degradation in specific scenarios such as 3D domains with multiple overlapping interfaces. To address this limitation, we propose Norm-PCGrad, a normalized variant that achieves state-of-the-art accuracy across a range of 2D and 3D domain decomposition problems. Across the benchmarks considered, Norm-PCGrad consistently a
    
[^228]: Prism-SQA：一种可解释且可适应的表面肌电信号质量评估神经框架

    Prism-SQA: An Interpretable and Adaptable Neural Framework for Surface Electromyography Quality Assessment

    [https://arxiv.org/abs/2609.12724](https://arxiv.org/abs/2609.12724)

    本文提出Prism-SQA框架，将sEMG信号质量评估重构为生理感知的源分离与验证过程，通过将信号分解为干净成分和五种污染成分，实现了可解释、可适应且无需重新训练的信号质量评估。

    

    表面肌电信号（sEMG）容易受到各种污染物的干扰，这些污染物会扭曲信号的形态和频谱内容。准确的信号质量评估（SQA）对于识别此类信号退化、确保可靠的临床分析与决策至关重要。近年来基于神经网络的SQA方法通过学习复杂的污染模式实现了准确的质量估计，但其黑盒特性使临床医生无法理解或验证所报告的评分，并且在不重新训练的情况下难以适应特定应用的质量定义。为了解决这些局限性，我们提出了Prism-SQA，这是一个可解释且可适应的神经框架，它将信号质量评估重新构建为一个具有生理感知能力的源分离与验证过程。Prism-SQA使用带有双向长短期记忆网络的U-Net将每个输入信号分解为一个干净的sEMG成分和五个针对特定污染物的成分。每个分离出的污染成分都会被检验……（原文摘要在此截断）

    arXiv:2609.12724v1 Announce Type: cross  Abstract: sEMG is vulnerable to various contaminants that distort signal morphology and spectral content. Accurate signal quality assessment (SQA) is essential for identifying such degradation and ensuring reliable clinical analyses and decisions. Recent neural network-based SQA methods achieve accurate quality estimation by learning complex contamination patterns, yet their black-box nature prevents clinicians from understanding or validating the reported scores and limits adaptability to application-specific quality definitions without retraining. To address these limitations, we propose Prism-SQA, an interpretable and adaptable neural framework that reformulates SQA as a physiology-aware source-separation and verification process. Prism-SQA decomposes each input signal into a clean sEMG component and five contaminant-specific components using a U-Net with bidirectional long short-term memory. Each separated contaminant component is examined b
    
[^229]: 带梯度裁剪与加性噪声的随机梯度方法的几乎必然收敛性分析

    Almost Sure Convergence Analysis of Stochastic Gradient Methods with Clipping and Additive Noise

    [https://arxiv.org/abs/2609.12119](https://arxiv.org/abs/2609.12119)

    本文证明了带梯度裁剪和加性高斯噪声的SGD在平滑性和一致有界噪声假设及标准步长衰减条件下几乎必然收敛，并将该结果扩展到随机重球法和Nesterov加速梯度等动量变体。

    

    带有梯度裁剪和加性噪声的随机梯度下降（SGD）已成为训练机器学习模型的标准技术，特别是在需要鲁棒性或隐私保证的应用中。然而，裁剪会在随机梯度中引入偏差，而加性噪声会引入额外的方差，这使得单个优化轨迹的长期行为难以刻画。在这项工作中，我们证明了在平滑性和一致有界随机梯度噪声的假设下，只要步长满足一些标准的衰减条件，带裁剪和加性高斯噪声的SGD（SGD-CN）几乎必然（a.s.）收敛。我们的分析还扩展到动量变体，如随机重球法和Nesterov加速梯度法，我们证明了通过精心构造的能量函数可以获得类似的收敛保证。这些结果为理解相关方法提供了更强的理论基础。

    arXiv:2609.12119v1 Announce Type: new  Abstract: Stochastic gradient descent (SGD) with gradient clipping and additive noise has become a standard technique for training machine learning models, particularly in applications requiring robustness or privacy guarantees. However, clipping introduces a bias in stochastic gradients, while additive noise introduces additional variance, making the long-run behaviour of individual optimization trajectories difficult to characterize. In this work, we prove that SGD with clipping and additive Gaussian noise (SGD-CN) converges almost surely (a.s.) under smoothness and uniformly bounded stochastic-gradient noise assumptions, provided the step sizes satisfy some standard decaying conditions. Our analysis extends to momentum variants such as the stochastic heavy ball and Nesterov's accelerated gradient, where we show that careful energy constructions yield similar guarantees. These results provide stronger theoretical foundations for understanding th
    
[^230]: 从集体稳态中学习相互作用核

    Learning Interaction Kernels from Collective Steady States

    [https://arxiv.org/abs/2609.12004](https://arxiv.org/abs/2609.12004)

    该论文提出了一种仅需从集体稳态的单快照观测中学习相互作用粒子系统相互作用核的方法，通过基于观测构型经验分布的正则化策略解决了本质上不适定的逆问题，实现了对相互作用规律的稳定准确恢复以及对集体行为乃至其动力学过程的忠实重现。

    

    我们提出了一种从集体行为的单快照观测中对相互作用粒子系统进行系统辨识的学习方法，这与依赖轨迹观测的现有方法不同。这一设定导致了一个本质上不适定的逆问题，我们通过一种基于观测构型经验分布的正则化策略来解决该问题，这些构型来自不同的、未被观测到的初始条件。我们在多种具有稳态和准稳态模式的代表性模型上测试了该学习程序，在这些模型中，集体行为编码了关于相互作用机制的隐含信息。结果表明，我们的方法能够稳定且准确地恢复潜在的相互作用规律，从而忠实地重现集体行为，在许多情况下甚至能够重现导致该集体行为的动力学过程。

    arXiv:2609.12004v1 Announce Type: cross  Abstract: We propose a learning procedure for system identification in interacting particle systems from single-snapshot observations of collective behaviors, unlike existing approaches that rely on observations of trajectories. This setting leads to a fundamentally ill-posed inverse problem, which we solve by using a regularization strategy based on the empirical distribution of observed configurations, drawn from different, unobserved initial conditions. We test our learning procedure on a variety of representative models with steady-state and quasi-stationary patterns, where collective behaviors encode implicit information about the interaction mechanisms, demonstrating that our approach enables stable and accurate recovery of the underlying interaction laws, leading to faithful reproduction of the collective behavior, and in many cases even of the dynamics leading up to it.
    
[^231]: 当信息分布在协同头中发生漂移时，多模态大语言模型会产生幻觉

    MLLMs Hallucinate when Information Distribution Drifts in Synergy Heads

    [https://arxiv.org/abs/2609.09206](https://arxiv.org/abs/2609.09206)

    该论文发现多模态大语言模型的幻觉源于协同注意力头中信息分布偏离健康平衡状态，而非模态信息的数量或强度，并提出HEAL方法，通过因果噪声干预和反事实双重差分实现头级别信息解耦与校准，以识别和缓解幻觉。

    

    多模态大语言模型（MLLMs）常常受到幻觉问题的困扰，这阻碍了其可靠的实际应用。现有的基于注意力的缓解方法主要依赖间接信号（如注意力权重），这些信号无法准确反映幻觉产生背后的实际信息偏移。在本文中，我们提出了HEAL——一种头级别信息解耦与校准方法，用于识别和缓解幻觉。HEAL首先对多头输出施加因果噪声干预，以过滤掉因果冗余的注意力头。随后，通过反事实双重差分法解耦剩余头中的信息分布，将注意力头划分为四种类型。通过分析，我们观察到：当信息分布偏离协同头中的健康平衡状态时，幻觉就会发生，而幻觉与模态特定信息的数量或强度并无强相关性。

    arXiv:2609.09206v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) often struggle with hallucinations, thus hindering their reliable practical applications. Existing attention-based mitigation methods mainly rely on indirect signals (e.g., attention weights) that fail to accurately reflect the actual information shift underlying hallucination generation. In this paper, we propose HEAL, Head-lEvel information disentAnglement and caLibration for identifying and mitigating hallucinations. HEAL first employs causal noise intervention on multi-head outputs to filter out causally redundant heads. Subsequently, it disentangles information distribution within the remaining heads via the counterfactual Difference-in-Differences, categorizing heads into four types. Through analysis, we observe: hallucinations happen when information distribution drifts away from a healthy equilibrium in synergy heads, not strongly correlated with the quantity or strength of modality-spec
    
[^232]: 面向稀疏标注时间序列的多任务学习：以耐寒性建模为例的案例研究

    Multi-Task Learning for Sparsely-Labeled Time Series: A Case Study on Cold-Hardiness Modeling

    [https://arxiv.org/abs/2609.09062](https://arxiv.org/abs/2609.09062)

    本文针对葡萄耐寒性预测中标注数据稀疏且品种间存在差异的难题，提出将不同葡萄品种视为不同任务的多任务学习方法，从而有效利用有限的稀疏标注时间序列数据，提升基于RNN的每日耐寒性预测效果。

    

    我们提出了一项真实世界的案例研究，探讨在数据有限且时间标签稀疏的情况下，利用多任务学习（MTL）进行时间过程建模。具体而言，我们研究了针对一个重要农业问题的多任务学习方法——预测葡萄耐寒性，即发生致命冻害时的温度。耐寒性会随天气变化而变化，且在田间难以直接测量，因此种植者需要依赖预测结果来决定何时采取代价高昂的防霜冻措施。我们应用循环神经网络（RNN），基于时间序列天气数据进行每日耐寒性预测。一个主要挑战在于耐寒性响应因植物品种而异，且每个品种的真实标注数据在时间上稀疏且有限。为应对这一挑战，我们研究了将不同品种对应为不同任务的多任务学习（MTL）方法来组合利用数据，并开发了一种……

    arXiv:2609.09062v1 Announce Type: new  Abstract: We present a real-world case study of multi-task learning (MTL) for temporal process modeling from limited data with temporally sparse labels. Specifically, we investigate multi-task learning for the important agricultural problem of predicting grape cold hardiness, which is the temperature at which lethal freezing occurs. Cold hardiness changes in response to weather and is difficult to measure directly in the field. Thus, growers rely on predictions to decide when to apply costly frost mitigation measures. We apply recurrent neural networks (RNNs) for daily cold-hardiness prediction from time series weather data. A major challenge is that the cold hardiness response varies across plant cultivars and ground-truth data for each cultivar is temporally sparse and limited. To address this challenge, we investigate multi-task learning (MTL) approaches for combining data, where different tasks correspond to different cultivars. We develop a v
    
[^233]: 面向轴结构信号的自适应各向异性注意力

    Adaptive Anisotropic Attention for Axis-Structured Signals

    [https://arxiv.org/abs/2609.08788](https://arxiv.org/abs/2609.08788)

    提出自适应各向异性注意力（AAA），将注意力沿电极时间轴分解为时间路径与空间路径并通过门控自适应加权融合，所构建的AXON模型在六个EEG下游任务上优于稠密注意力基线。

    

    稠密自注意力在学习之前将所有token对视为同等可信的，这种交互各向同性的先验可能与结构化信号不匹配。对于诸如脑电（EEG）这类结构化、低信噪比（SNR）的信号，其依赖关系是沿电极轴和时间轴组织的，而这种均匀先验使每个token暴露于许多无关的交互之中。我们提出了自适应各向异性注意力（AAA），它将注意力拆分为两条路径：时间路径，其中每个token关注其自身电极在时间维度上的token；空间路径，其中它关注同一时间步上其他电极的token。一个小型门控网络为每个token预测两条路径输出的凸组合：即两个和为一的非负权重。在六个EEG下游任务上，由此构建的模型AXON（轴分解算子网络，AXis-factorized Operator Network）在线性探测和完整微调设置下，其平均平衡准确率均优于稠密基线……

    arXiv:2609.08788v1 Announce Type: cross  Abstract: Dense self-attention treats all token pairs as equally plausible before learning, an interaction-isotropic prior that can be mismatched to structured signals. For structured, low signal-to-noise ratio (SNR) signals such as EEG, dependencies are organized along the electrode and time axes, and this uniform prior exposes each token to many irrelevant interactions. We introduce Adaptive Anisotropic Attention (AAA), which splits attention into two paths: a temporal path, where each token attends to the tokens of its own electrode across time, and a spatial path, where it attends to the tokens of the other electrodes at the same time step. A small gate predicts, for every token, a convex combination of the two path outputs: two non-negative weights that sum to one. On six EEG downstream tasks, the resulting model, AXON (AXis-factorized Operator Network), improves mean balanced accuracy over a dense baseline under both linear probing and ful
    
[^234]: 导向干扰反映的是模型的默认倾向，而非行为方向

    Steering Interference Reflects the Model's Defaults, Not the Behavior Directions

    [https://arxiv.org/abs/2609.06951](https://arxiv.org/abs/2609.06951)

    激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。

    

    激活导向有望实现对语言模型行为的模块化控制：某种行为（如礼貌）对应模型激活中的一个方向，在模型生成时添加该方向应当能开启该行为，且不影响其他方面。但事实并非如此。我们探究了是什么决定了哪些其他行为会发生变化以及变化程度，发现起决定作用的是模型本身，而非被导向的行为。导向会使模型放松趋向于它本已偏好的一小部分行为，主要是拒答、谄媚和诗歌化倾向，且无论导向什么行为，这一行为集合大体相同。横跨24种行为和十个指令微调模型的三项结果支持这一结论，所有效应均由语言模型裁判从生成文本中读取，而非通过探针读取。这种读取方式很重要：所有24种行为都是线性可解码的，但只有20种行为会改变模型的实际输出内容。第一，一个不含任何行为内容、仅在特定方面与真实导向相匹配的方向……（摘要在此处被截断）

    arXiv:2609.06951v1 Announce Type: cross  Abstract: Activation steering promises modular control of language model behavior: a behavior such as politeness corresponds to a direction in a model's activations, and adding that direction while it generates should switch the behavior on and leave everything else alone. It does not. We ask what decides which other behaviors move, and by how much, and find that it is the model rather than the behavior being steered. A steer relaxes the model toward a small set of behaviors it already favors, chiefly refusal, sycophancy, and poeticism, and that set is much the same whatever is steered.   Three results across 24 behaviors and ten instruction-tuned models support this, every effect read off the generated text by a language-model judge rather than off a probe. That readout matters: all 24 behaviors are linearly decodable, but only 20 change what the model writes. First, a direction carrying no behavioral content, matched to a real steer only in th
    
[^235]: 通过对齐学习核函数的多类贝叶斯分类

    Learning Kernels by Alignment for Multiclass Bayes Classification

    [https://arxiv.org/abs/2609.06474](https://arxiv.org/abs/2609.06474)

    本文证明协作学习本质上是核对齐过程、协作推理等价于核贝叶斯分类，进而用可学习的马氏距离替代余弦相似度将CLaI推广至多类分类，在多个数据集上显著提升了准确率、收敛速度与校准性能。

    

    核方法将数据表示与决策制定分离，但通常需要预先选择核函数。我们证明该核函数可以通过对齐方式来学习，并借助最近提出的协作学习与推理框架（CLaI）发展了由此形成的框架。我们证明协作学习可以被视为一个核对齐过程，其中通过训练嵌入表示，使其诱导的相似度与由标签导出的目标核相匹配。我们还证明协作推理等价于采用Parzen窗密度估计的核贝叶斯分类。基于这些视角，我们通过用可学习的马氏距离替换余弦相似度来推广CLaI，并将其扩展到多类分类。在CIFAR-10、PathMNIST和SleepEDF数据集上，马氏距离形式相比基于余弦相似度的变体提高了准确率、收敛更快，并产生了更低的校准误差。

    arXiv:2609.06474v1 Announce Type: new  Abstract: Kernel methods separate data representation from decision-making, but typically require the kernel to be chosen in advance. We show that this kernel can instead be learned by alignment, and develop the resulting framework through the recently introduced Collaborative Learning and Inference (CLaI). We show that Collaborative Learning can be viewed as a kernel alignment process, in which an embedding is trained so that its induced similarity matches a label-derived target kernel. We also prove that Collaborative Inference is equivalent to kernel Bayes classification with Parzen-window density estimation. Motivated by these perspectives, we generalise CLaI by replacing cosine similarity with a learned Mahalanobis distance and extend it to multiclass classification. On CIFAR-10, PathMNIST, and SleepEDF, the Mahalanobis formulation improves accuracy, converges faster, and yields lower calibration error than the cosine-based variant. Auxiliary
    
[^236]: Calendar-SPCA：面向多周期用电曲线的可解释表示学习

    Calendar-SPCA: Interpretable Representation Learning for Multi-Periodic Electricity Consumption Profiles

    [https://arxiv.org/abs/2609.06060](https://arxiv.org/abs/2609.06060)

    提出 Calendar-SPCA 日历结构稀疏主成分方法，将日、周、年等多周期日历几何直接嵌入用电数据的低维表示学习中，生成稀疏且局部连贯、可直接解读的载荷模式，并在两个独立智能电表数据集上验证了其有效性。

    

    长期用电曲线呈现出多种同时存在的周期结构，包括日周期、周周期和年周期。本工作提出了 Calendar-SPCA，这是一种日历结构的稀疏主成分方法，将这种已知的多周期几何结构直接融入低维表示学习中。特征域被表示为循环日历轴的笛卡尔积，并通过 L1 载荷惩罚与日历图上的图全变差来估计低秩分解。因此，该方法产生的稀疏且局部连贯的载荷模式在其原始时间坐标下保持直接可读性。Calendar-SPCA 在两个具有不同样本量和时间分辨率的独立智能电表数据集上进行了评估：GoiEner 和 Low Carbon London。一个因子实验刻画了稀疏性与日历约束的互补效应。

    arXiv:2609.06060v1 Announce Type: cross  Abstract: Long-term electricity-consumption profiles exhibit several simultaneous periodic structures, including daily, weekly, and annual cycles. This work introduces Calendar-SPCA, a calendar-structured sparse principal component method that incorporates this known multi-periodic geometry directly into low-dimensional representation learning. The feature domain is represented as the Cartesian product of cyclic calendar axes, and a low-rank factorization is estimated using an L1 loading penalty together with graph total variation over the resulting calendar graph. The method therefore produces sparse and locally coherent loading patterns that remain directly readable in their original temporal coordinates. Calendar-SPCA is evaluated on two independent smart-meter datasets with different sample sizes and temporal resolutions: GoiEner and Low Carbon London. A factorial experiment characterizes the complementary effects of sparsity and calendar co
    
[^237]: 面向JEPA风格世界模型的频谱目标物理潜在结构化方法

    Spectral-Target Physical Latent Structuring for JEPA-Style World Models

    [https://arxiv.org/abs/2609.04264](https://arxiv.org/abs/2609.04264)

    该论文发现了JEPA风格世界模型中“物理表示惰性”这一新失败模式，并提出轻量级傅里叶辅助头在训练时对潜在空间施加物理信息结构化约束，在不增加推理成本的情况下有效提升下游规划性能。

    

    潜在世界模型作为一种在潜在空间而非像素空间中进行预测和规划的方法，正日益受到欢迎。近期的架构，如LeWorldModel（LeWM），采用SIGReg等正则化技术联合训练编码器和预测器，以防止表示坍塌。然而，即使有此类正则化来防止表示坍塌，我们仍然识别出一种新的世界模型失败模式——“物理表示惰性”，该现象在高动态环境中尤为显著。对于这些“惰性”情况，学习到的潜在状态并未坍塌，但却无法表示关键的物理属性，从而普遍导致下游规划失败。为解决这一问题，我们提出在训练阶段引入带有轻量级“傅里叶辅助头”的辅助监督，该方法在潜在空间中强制实施基于物理信息的结构化，且不产生任何额外的推理时开销，并可推广至任意环境。

    arXiv:2609.04264v1 Announce Type: new  Abstract: Latent world models have become increasingly popular as a method to predict and plan in latent space rather than pixel space. Recent architectures, such as LeWorldModel (LeWM), jointly train the encoder and predictor using regularization techniques like SIGReg to prevent representation collapse. Even with such regularization preventing representation collapse, we identify a new world model failure mode of \textit{physical representation laziness}, particularly noted in highly dynamic environments. For these lazy cases, the learned latent states do not collapse but nonetheless fail to represent key physical properties, causing ubiquitous downstream planning failure. To resolve this issue, we propose training-time auxiliary supervision with a lightweight "Fourier auxiliary head", which enforces physically-informed structuring of the latent space with no additional inference-time cost and can be generalized to any environment. Experimentall
    
[^238]: 面向独立于表演者的身体运动情感识别的正交集成与经过验证的解释方法

    Orthogonal Ensembles and Tested Explanations for Performer-Independent Body-Motion Emotion Recognition

    [https://arxiv.org/abs/2609.02510](https://arxiv.org/abs/2609.02510)

    在留一表演者的困难评估设置下，通过组合十一个误差模式正交的模型，将12类身体运动情感识别的Macro-F1提升11.07个百分点，并提供了一套经过实验验证的解释方法，证明模型决策基于运动的身体区域证据且与拉班动作分析（LMA）属性高度一致。

    

    我们研究了在留一表演者评估下，仅基于身体、从骨骼运动进行12类表演情感分类的问题，这是一个困难且欠定的设置：随机猜测的准确率为8.3%，而协议匹配的复现STGCN++基线仅达到25.73 ± 4.03%的Macro-F1。我们表明，可靠的性能提升并非来自新架构，而是来自组合十一个具有正交误差模式的模型：在标记训练表演者上的10折留一表演者交叉验证中，等权重logit均值集成方法达到每折36.80 ± 4.00%的Macro-F1，相比同数据划分的复现基线，在协议匹配条件下提升+11.07个百分点（相对提升+43%）。我们的核心贡献是一套经过验证的解释套件：对于一个强集成成员，部位遮蔽和反事实编辑实验表明（而非仅断言）其决策依赖于基于运动的身体区域证据，并且这种区域显著性远比分类表现更符合基于规则的拉班动作分析（LMA）属性……（摘要原文在此处截断）

    arXiv:2609.02510v1 Announce Type: cross  Abstract: We study body-only, 12-class acted-emotion classification from skeleton motion under leave-performer-out (LPO) evaluation, a hard, underdetermined setting: chance is 8.3%, and a protocol-matched reproduced STGCN++ baseline reaches only 25.73 +/- 4.03% Macro-F1. We show that reliable gains come not from a new architecture but from combining eleven models with orthogonal error modes: under 10-fold LPO cross-validation on the labeled training performers, an equal-weight logit-mean ensemble reaches 36.80 +/- 4.00% per-fold Macro-F1, a protocol-matched +11.07 pp (+43% relative) over the same-split reproduced baseline. Our central contribution is a tested explanation suite: for a strong ensemble member, part-masking and counterfactual edits show (rather than assert) that its decisions depend on motion-grounded body-region evidence, and this region saliency aligns with rule-based Laban Movement Analysis (LMA) attributes far more than with cla
    
[^239]: 哪些历史数据对时间序列预测重要？通过未来监督学习预测相关性

    Which Histories Matter for Time Series Forecasting? Learning Predictive Relevance with Future Supervision

    [https://arxiv.org/abs/2608.23221](https://arxiv.org/abs/2608.23221)

    该论文提出了一种通过未来监督学习预测相关性来重新排名时间序列检索候选的方法，显著提升了模式检索性能。

    

    arXiv:2608.23221v1 公告类型：新 摘要：时间序列预测中的历史检索通常将过去相似性作为有用性的代理。我们提出了一个不同的问题：对于某个查询，哪些历史示例应该被预期会产生影响？我们将预测相关性定义为在推理时信息条件下预期未来效用，仅在训练期间使用实现未来作为特权监督。一个归一化模式检索器首先形成一个粗略的候选集，然后一个轻量级残差多层感知器（MLP）学习一个列表式未来兼容性目标，同时保持推理时评分严格仅基于过去。我们的方法保留了基于相似性的候选生成，但通过更预测性的相关性标准重新排名其候选。最优相关性分解为候选级效用和查询特定兼容性，这启发了候选先验和打乱未来控制。在六个基准上，重排名器改进了模式检索，同时揭示了...

    arXiv:2608.23221v1 Announce Type: new  Abstract: Historical retrieval for time-series prediction commonly treats past similarity as a proxy for usefulness. We ask a different question: which historical examples should be expected to matter for a query? We define predictive relevance as expected future utility conditioned on inference-time information, using realized futures only during training as privileged supervision. A normalized-pattern retriever first forms a coarse candidate set, and a lightweight residual multilayer perceptron (MLP) learns a listwise future-compatibility target while keeping inference-time scoring strictly past-only. Our method retains similarity-based candidate generation but reranks its candidates by a more predictive relevance criterion. Optimal relevance decomposes into candidate-level utility and query-specific compatibility, motivating Candidate-Prior and Shuffled-Future controls. Across six benchmarks, the reranker improves Pattern retrieval while reveal
    
[^240]: 定量等式理论的持久幅度同调

    Persistent Magnitude Homology for Quantitative Equational Theories

    [https://arxiv.org/abs/2608.21479](https://arxiv.org/abs/2608.21479)

    本文为定量等式理论的自由代数构造了持久幅度同调这一函子不变量，证明了幅度同调与持久性理论可以通过长度子水平集过滤统一为同一个构造，二者由长正合序列相互补充、各取所长。

    

    arXiv:2608.21479v2 公告类型：replace-cross 摘要：定量等式理论 $U$ 对在数值误差范围内一致的项进行推理。它在生成元构成的度量空间 $A$ 上呈现一个自由代数 $T_UA$，即语法中的项处于公理所推导出的最小距离处，而该度量正是其语义内容。我们给出了它的一个函子不变量，即 $T_UA$ 的持久幅度同调：一个条形码，其中模块是驯顺的，当 $T_UA$ 为有限时是有限线性代数，且在每个度上都是 Lipschitz 的。幅度同调按长度分次，对持久性一无所知；而它的持久化扩展对条形码的起点和终点一无所知；然而两者实为同一个构造：用长度的子水平集对长度神经进行过滤便得到持久性模块，而该过滤的伴随分次正是幅度复形。一个长正合序列将两者联系起来，每一方都补齐了另一方所缺失的内容。幅度同调能够定位条形码的临界值，因此……

    arXiv:2608.21479v2 Announce Type: replace-cross  Abstract: A quantitative equational theory $U$ reasons about terms that agree up to a numerical error. It presents a free algebra $T_UA$ over a metric space $A$ of generators, the terms of the syntax at the least distance the axioms derive, and that metric is its semantic content. We give a functorial invariant of it, the persistent magnitude homology of $T_UA$: a barcode where the module is tame, finite linear algebra where $T_UA$ is finite, Lipschitz in each degree. Magnitude homology is graded by length and knows nothing of persistence, its persistent refinement nothing of where its bars begin and end, yet the two are one construction: filtering the length nerve by sublevel sets of the length yields the persistence module, and the associated graded of that filtration is the magnitude complex. A long exact sequence exchanges them, and each side gains what it lacked. Magnitude homology locates the critical values of the barcode, so a gr
    
[^241]: 一个设计选择决定了材料机器学习模型是否会产生物理上不可能的预测

    A single design choice determines whether machine learning models of materials make physically impossible predictions

    [https://arxiv.org/abs/2608.18714](https://arxiv.org/abs/2608.18714)

    本文发现一个简单的设计选择——特征是否携带宇称标签——决定了材料机器学习模型是否会做出物理上不可能的预测，并提出了从群论计算“宇称间隙”的标准来预测哪些性质和晶体受影响。

    

    arXiv:2608.18714v1 公告类型：交叉 摘要：机器学习模型正在材料发现领域取代第一性原理计算，而物理对称性是构建于其中的核心保证。关于应硬编码多少对称性而非学习多少的争论一直围绕旋转展开，其中对称性误差是一种近似误差。某些约束是精确的：对称性迫使某些性质张量精确为零，因此非零预测在物理上是不可能的，而非不准确。在这里，我们展示了模型是否能做出此类预测，在训练前由一个很少被报告的设计位决定，即其特征是否携带宇称标签，并推导出一个标准——宇称间隙，它仅从群论计算就能确定哪些性质和晶体暴露于这种风险。在仅在该位上有所不同的匹配架构对上，对两千个中心对称晶体（其压电张量必须为零）进行评估，携带宇称标签的臂部达到浮点数下限，而其他臂部则产生物理上不可能的预测。

    arXiv:2608.18714v1 Announce Type: cross  Abstract: Machine-learned models are replacing first-principles calculations across materials discovery, and physical symmetry is the central guarantee built into them. The debate over how much symmetry to hard-wire rather than learn has run on rotations, where a symmetry error is an approximation error. Some constraints are exact: symmetry forces certain property tensors to exactly zero, so a nonzero prediction is physically impossible rather than inaccurate. Here we show that whether a model can make such predictions is decided before training by one rarely reported design bit, whether its features carry parity labels, and derive a criterion, the parity gap, that computes from group theory alone which properties and crystals are exposed. Across matched architecture pairs differing only in that bit, evaluated on two thousand centrosymmetric crystals whose piezoelectric tensor must vanish, parity-labelled arms sit at the floating-point floor whi
    
[^242]: 并行高斯过程强盗优化的改进遗憾分析

    Improved Regret Analysis for Parallel Gaussian Process Bandit Optimization

    [https://arxiv.org/abs/2608.16492](https://arxiv.org/abs/2608.16492)

    本文通过GP-BTS示例，证明无需初始不确定性采样阶段即可消除批量大小对遗憾上界的乘性影响，并在无噪声条件下实现更优的遗憾界限。

    

    本文研究了并行高斯过程（GP）强盗优化的遗憾分析。广泛使用的GP批量上置信界和GP批量汤普森采样（GP-BTS）的已知遗憾上界，在批量大小$Q$上存在一个乘性因子。为避免这种性能退化，现有分析需要在优化开始时对$Q$进行多项式数量的不确定性采样（US）。然而，这种初始US阶段在实践中往往效果不佳。本文以GP-BTS为例，表明无需初始US阶段即可实现无$Q$乘性因子的遗憾上界。此外，我们展示了在无噪声设置下，遗憾上界远优于有噪声设置，这与顺序GP强盗设置中的情况一致。

    arXiv:2608.16492v1 Announce Type: cross  Abstract: This paper studies the regret analysis for parallel Gaussian process (GP) bandit optimization. The known regret upper bounds for the widely used GP batched upper confidence bound and GP batched Thompson sampling (GP-BTS) suffer from a multiplicative factor with respect to the batch size $Q$. To avoid this degradation, existing analyses require a polynomial number of uncertainty sampling (US) for $Q$ at the beginning of optimization. However, this initial US phase is often ineffective in practice. This paper shows that the regret upper bound without the multiplicative factor on $Q$ can be achieved without the initial US phase, using GP-BTS as an example. Furthermore, we show much better regret upper bounds in the noiseless setting than in the noisy setting, as in the sequential GP bandit setting.
    
[^243]: 打破压缩壁垒：通过逆向再生实现跨架构压缩边界学习

    Breaking the Compression Barrier: Cross-Architecture Compression Boundary Learning via Reverse Regrowth

    [https://arxiv.org/abs/2608.16010](https://arxiv.org/abs/2608.16010)

    本文提出BRIDGE框架，通过逆向再生策略，先稀疏化模型暴露崩溃区，再选择性恢复关键结构，从而准确找到并突破模型压缩的性能边界。

    

    模型压缩对于在资源受限的边缘设备上部署网络至关重要。尽管基于剪枝的方法可以显著减小模型规模，但它们在超过稀疏阈值后常常遭遇性能急剧下降，这使得难以确定模型的可行压缩极限。为应对这一挑战，我们提出了一种边界学习逆向再生框架，称为BRIDGE，该框架将压缩重新构建为一个建设性的边界搜索问题。与正向剪枝不同，我们的方法首先将模型驱动到极度稀疏状态以暴露崩溃区域，然后选择性地再生关键结构以恢复性能。所提出的框架采用层次化再生策略，包括粗粒度层选择和细粒度再生参数选择，以准确识别需要恢复的参数。实验表明，我们的方法能够恢复模型，并突破传统压缩限制。

    arXiv:2608.16010v1 Announce Type: new  Abstract: Model compression is critical for deploying networks on resource-constrained edge devices. While pruning-based methods can significantly reduce model size, they often suffer from abrupt performance collapse beyond a sparsity thresh-old, making it difficult to identify the feasible compression limit of the model. To address this challenge, we propose a boundary-Learning reverse regrowth framework, BRIDGE, that reformulates compression as a constructive boundary-search problem. Unlike forward pruning, our method first drives the model to an extremely sparse state to expose the collapse region, and then selectively regenerates the critical structure to restore performance. The proposed framework employs a hierarchical regeneration strategy, including coarse-grained layer selection and fine-grained regeneration parameter selection, to accurately identify which parameters require recovery. Experiments show that our method can recover models f
    
[^244]: 符号回归中的深度分治归约（DDRSR）

    Deep Divide-and-Reduce in Symbolic Regression

    [https://arxiv.org/abs/2608.02628](https://arxiv.org/abs/2608.02628)

    该论文提出DDRSR方法，通过对更广泛分解结构的形式化分析，从根本上扩展了AI Feynman方法中表达式分解与归约机制的适用范围，并克服了其依赖暴力搜索的局限。

    

    符号回归（SR）是从数据中发现潜在模式并用数学表达式将其表示出来的任务。当前的机器学习符号回归方法通常缺乏对这些表达式所遵循的内在数学和物理原理的深刻理解。虽然开创性的AI Feynman方法利用了数据背后的数学性质，但其表达式分解机制存在适用范围狭窄的问题，在复杂方程上容易失效。此外，其底层机制严重依赖于对子表达式的暴力搜索，极大地限制了其实用性。在AI Feynman的基础上，我们提出了符号回归中的深度分治归约方法（DDRSR），这是通过对更广泛一类分解结构进行形式化分析而得出的一个有原则性的扩展。DDRSR从根本上拓宽了表达式分解和归约的适用范围……

    arXiv:2608.02628v2 Announce Type: replace-cross  Abstract: Symbolic regression (SR) is the task of discovering underlying patterns from data and representing them using mathematical expressions. Current machine learning approaches to SR often lack a profound understanding of the intrinsic mathematical and physical principles governing these expressions. While the pioneering AI Feynman method leverages the mathematical properties underlying the data, its expression decomposition mechanism suffers from a narrow scope of applicability and is prone to failure on complex equations. Furthermore, its underlying mechanisms rely heavily on brute-force searches for sub-expressions, severely limiting its practical utility. Building on AI Feynman, we propose Deep Divide-and-Reduce in Symbolic Regression (DDRSR), a principled extension derived from a formal analysis of a broader class of decomposition structures. DDRSR fundamentally broadens the applicability of expression decomposition and reducti
    
[^245]: Matérn核与平方指数核再生核希尔伯特空间中固定先验期望改进的简单遗憾率与极小极大最优性

    Simple-regret rates and minimax optimality of fixed-prior expected improvement in Mat\'ern and squared-exponential RKHSs

    [https://arxiv.org/abs/2607.29245](https://arxiv.org/abs/2607.29245)

    本文证明了在Matérn核和平方指数核的再生核希尔伯特空间中，弱期望改进策略的简单遗憾率达到极小极大最优，分别以 $O(N^{-\nu/d})$ 和指数级速率收敛。

    

    我们研究期望改进（EI）方法在最小化确定性函数 $f$ 时的表现，其中 $f$ 属于定义在非空紧集 $\mathcal X\subset\mathbb R^d$ 上的连续半正定核 $k$ 所对应的再生核希尔伯特空间 $\mathcal H_k$。函数值被精确观测（无噪声），EI 基于一个具有协方差 $\sigma^2k$（$\sigma>0$）的固定零均值高斯过程模型计算。弱EI策略是指查询期望改进至少为其最大值固定正比例的点的策略。我们借鉴贪婪逼近的思想，引入了顺序分离半径的概念，将排序后的所选点创新范数与Kolmogorov宽度联系起来。利用散乱数据逼近领域的标准幂函数估计和有限预算遗憾论证，我们得到了收敛率。在 $N$ 次初始后查询之后，每个弱EI策略对于光滑度为 $\nu>0$ 的各向同性Matérn核具有简单遗憾 $O(N^{-\nu/d})$，对于各向同性平方指数核具有简单遗憾 $O(\exp[-c_1\min\{N,N^{1/d}\log(eN)\}])$。

    arXiv:2607.29245v2 Announce Type: replace-cross  Abstract: We study expected improvement (EI) for minimizing a deterministic function $f$ in the RKHS $\mathcal H_k$ of a continuous positive-semidefinite kernel $k$ on a nonempty compact set $\mathcal X\subset\mathbb R^d$. Function values are observed exactly, and EI is computed from a fixed zero-mean Gaussian-process model with covariance $\sigma^2k$, $\sigma>0$. A weak-EI policy queries a point whose EI is at least a fixed positive fraction of its maximum.   We introduce a notion of sequential separation radius relating ranked selected-point innovation norms to Kolmogorov widths, drawing on greedy approximation. Standard power-function estimates from scattered-data approximation and a finite-budget regret argument yield the rates. After $N$ post-initial queries, every weak-EI policy has simple regret $O(N^{-\nu/d})$ for isotropic Mat\'ern kernels of smoothness $\nu>0$ and $O(\exp[-c_1\min\{N,N^{1/d}\log(eN)\}])$ for the isotropic squar
    
[^246]: 用于连贯急诊科预测的层次化时空Transformer

    Hierarchical Spatio-Temporal Transformer for Coherent Emergency Department Forecasting

    [https://arxiv.org/abs/2607.27106](https://arxiv.org/abs/2607.27106)

    提出HierSTT框架，基于层次化Transformer在单一模型中联合预测医院、地区和国家三个层面的急诊科需求，解决了传统单层级独立预测导致的预测不连贯问题。

    

    急诊科是医疗保健系统中的关键入口，然而它们持续面临着来自不可预测的患者需求、季节性激增和非紧急就诊的压力。有效的急诊科规划需要在多个决策层面进行预测：医院需要本地需求估计以进行人员配备和床位管理，地区需要预测来协调各医疗单位，国家当局需要全系统预测以进行容量规划。然而，大多数现有方法仅在单一层面上独立预测急诊科需求，忽略了连接医院、地区和国家系统的层次结构。这可能导致预测不连贯，即医院层面的预测无法一致地汇总到地区或国家级的需求。我们提出了HierSTT，一个基于层次化Transformer的框架，用于连贯的多层级急诊科预测。HierSTT在单一模型中联合预测医院、地区和国家层面的需求。

    arXiv:2607.27106v2 Announce Type: replace  Abstract: Emergency Departments (EDs) are critical access points in healthcare systems, yet they face persistent pressure from unpredictable patient demand, seasonal surges, and non-urgent visits. Effective ED planning requires forecasts at multiple decision-making levels: hospitals need local demand estimates for staffing and bed management, regions require forecasts to coordinate healthcare units, and national authorities need system-wide projections for capacity planning. However, most existing approaches forecast ED demand independently at a single level, ignoring the hierarchy linking hospitals, regions, and national systems. This can produce incoherent predictions, where hospital-level forecasts do not aggregate consistently to regional or national demand. We propose HierSTT, a hierarchical Transformer-based framework for coherent multi-level ED forecasting. HierSTT jointly predicts hospital, regional, and national level demand in a sing
    
[^247]: Bumblebee：面向大规模推荐系统的交错混合层构建模块

    Bumblebee: Interleaved Mixed-Layer Building Blocks for Large-Scale Recommendation Systems

    [https://arxiv.org/abs/2607.24804](https://arxiv.org/abs/2607.24804)

    提出Bumblebee推荐架构，通过交错可堆叠的混合层构建模块将序列建模与特征交互两种范式有机融合，实现了特征模态的早期与反复混合，从而提升大规模推荐系统的表达能力。

    

    推荐系统在过去几年中经历了重大变革。从传统的特征交互模块到生成式下一动作预测的转变，推动了个性化内容的发展边界。这些发展主要沿着两条独立的轨道演进：一方面是序列建模方法，另一方面是特征交互方法。在本文中，我们提出了Bumblebee，这是一种推荐架构，通过交错的可堆叠块设计解决了这两个方向之间缺乏交互的问题。每个块实现了一个微管道层结构，将序列个性化、基于注意力的编码和特征交叉组合成一个自包含的单元。每个块都会生成两种特征模态的联合表示，该表示被序列中的下一个块所消费。这种机制鼓励模态的早期和重复混合，并丰富了下游（模型的表达能力）。

    arXiv:2607.24804v3 Announce Type: replace-cross  Abstract: Recommendation systems have undergone significant transformations in the past years. The transition from traditional feature interaction modules to generative next-action prediction has pushed the boundaries of personalized content. Developments have largely evolved along two separate tracks. Sequence modeling approaches on the one hand and feature interaction methods on the other. In this paper, we introduce Bumblebee, a recommendation architecture that addresses the lack of interaction between the two directions through an interleaved, stackable block design. Each block implements a micro-pipeline of layers combining sequence personalization, attention-based encoding, and feature crossing into a self-contained unit. Every block produces a joint representation of both feature modalities which is consumed by the next block in the sequence. This mechanism encourages early and repeated mixture of modalities and enriches downstrea
    
[^248]: 时序基础模型中的后训练：一个统一框架

    Post-Training in Time Series Foundation Models: A Unifying Framework

    [https://arxiv.org/abs/2607.20002](https://arxiv.org/abs/2607.20002)

    本文提出了一个统一框架，根据干预位置将时序基础模型的后训练方法归纳为参数适配、上下文增强、模型组合、输出处理与不确定性控制、压缩与专门化五大类，并系统分析了各类代表性方法、现有局限与未来方向。

    

    时序基础模型（TSFM）已成为时间序列分析的通用模型，但仅靠预训练往往不足以保证可靠的下游部署。弥合这一差距需要进一步的干预，以应对领域偏移、任务异质性、监督有限以及计算约束等挑战，这促使后训练成为一类广泛的方法，用于对预训练的时序基础模型进行适配、增强、组合、校准或专门化，以满足下游任务需求。在本工作中，我们基于干预在预测流水线中所处的位置对TSFM后训练方法进行分析，归纳出五个类别：参数适配、上下文增强、模型组合、输出处理与不确定性控制，以及压缩与专门化。在每个类别中，我们研究了主要的代表性方法，并讨论了它们当前的局限性。我们进一步指出了朝向可控适配、可靠（摘要在此处截断）

    arXiv:2607.20002v3 Announce Type: replace-cross  Abstract: Time series foundation models (TSFMs) have emerged as general-purpose models for time series analysis, but pretraining alone is often insufficient for reliable downstream deployment. Bridging this gap requires further intervention to handle domain shift, task heterogeneity, limited supervision, and computational constraints, which motivates post-training as a broad class of methods to adapt, augment, compose, calibrate, or specialize pretrained TSFMs for downstream tasks. In this work, we analyze TSFM post-training methods based on their locus of intervention in the prediction pipeline, yielding five categories: parameter adaptation, context augmentation, model composition, output processing and uncertainty control, and compression and specialization. Within each category, we study main representative methods and discuss their current limitations. We further identify future directions toward controlled adaptation, reliable cont
    
[^249]: 黎曼深度学习：模块、网络与几何

    Riemannian Deep Learning: Modules, Networks, and Geometries

    [https://arxiv.org/abs/2607.19305](https://arxiv.org/abs/2607.19305)

    本论文从可复用神经模块、流形专用网络架构和底层几何设计三个互补视角构建了统一的黎曼深度学习框架，将批归一化和多项逻辑回归等基础组件推广到李群、陀螺群和一般黎曼流形，并为双曲空间、满秩相关矩阵等重要几何表示设计了神经网络。

    

    流形值表示上的深度神经网络日益受到关注，但许多基础组件仍然局限于特定流形、依赖欧几里得近似，或需要代价高昂且数值不稳定的几何运算。本论文从三个互补的角度为黎曼深度学习构建了一个统一框架：可复用的神经模块、面向特定流形的网络架构，以及底层几何的设计。该论文将批归一化从欧几里得空间和单个流形推广到广泛的李群和陀螺群类别，并将多项逻辑回归从欧几里得空间扩展到SPD（对称正定）流形，进而扩展到一般黎曼流形。此外，论文还为几种重要的几何表示开发了神经网络，包括双曲空间的无约束模型、基于Busemann函数的双曲学习，以及满秩相关矩阵。最后，它……

    arXiv:2607.19305v4 Announce Type: replace-cross  Abstract: Deep neural networks on manifold-valued representations have attracted growing interest, but many basic components remain tied to specific manifolds, rely on Euclidean approximations, or require costly and numerically fragile geometric operations. This thesis develops a unified framework for Riemannian deep learning from three complementary perspectives: reusable neural modules, manifold-specific network architectures, and the design of underlying geometries. It generalizes batch normalization from Euclidean spaces and individual manifolds to broad classes of Lie groups and gyrogroups, and extends multinomial logistic regression from Euclidean space to SPD manifolds and then to general Riemannian manifolds. It further develops neural networks for several important geometric representations, including an unconstrained model of hyperbolic space, Busemann-based hyperbolic learning, and full-rank correlation matrices. Finally, it i
    
[^250]: 优化预条件子：一种基于静态遗憾最小化预言机的黑盒在线到非凸转换

    Optimizing the Preconditioner: A Black-box Online-to-Nonconvex Conversion with Static Regret Minimization Oracles

    [https://arxiv.org/abs/2607.17607](https://arxiv.org/abs/2607.17607)

    本文提出了一种从随机非凸优化到在线凸优化中静态遗憾最小化的黑盒归约方法，解决了Chen和Hazan（2024）提出的开放问题，并证明任何具有O(√T)遗憾的OCO预言机都能恢复经典的O(T^{-1/2})收敛速率。

    

    随机非凸优化是现代机器学习中训练深度网络和大语言模型的核心。我们给出了一个从随机非凸优化到在线凸优化（OCO）中普通静态遗憾最小化的黑盒归约，从而解决了Chen和Hazan（2024）提出的开放问题。我们的归约维护一个可预测的梯度跟踪器，同时一个黑盒在线学习器 $\mathcal{A}$ 选择一个预条件子，将该跟踪器转换为更新方向。给定一个值域以 $M$ 为界的 $\beta$-光滑函数和一个方差以 $\sigma^2$ 为界的无偏梯度预言机，我们将期望平均平方梯度范数界定为 $O(\sigma\sqrt{M\beta/T}+\sqrt{M\beta}\mathrm{Reg}_T(\mathcal{A})/T+\frac{M\beta}{T})$，其中 $\mathrm{Reg}_T(\mathcal{A})$ 是 $\mathcal{A}$ 的静态遗憾。因此，任何具有 $O(\sqrt{T})$ 遗憾的OCO预言机都能恢复经典的 $O(T^{-1/2})$ 收敛速率。

    arXiv:2607.17607v3 Announce Type: replace  Abstract: Stochastic nonconvex optimization is central to training deep networks and LLMs in modern machine learning. We give a black-box reduction from stochastic nonconvex optimization to ordinary static regret minimization in online convex optimization (OCO), thereby resolving the open problem posed by Chen and Hazan (2024). Our reduction maintains a predictable gradient tracker, while a black-box online learner $\mathcal{A}$ selects a preconditioner that transforms this tracker into the update direction. Given a \(\beta\)-smooth function with a range bounded by $M$ and an unbiased gradient oracle with variance bounded by $\sigma^2$, we bound the expected average squared gradient norm by $O(\sigma\sqrt{M\beta/T}+\sqrt{M\beta}\mathrm{Reg}_T(\mathcal{A})/T+\frac{M\beta}{T})$, where $\mathrm{Reg}_T(\mathcal{A})$ is the static regret of $\mathcal{A}$. Thus, any OCO oracle with $O(\sqrt{T})$ regret recovers the classical $O(T^{-1/2})$ convergenc
    
[^251]: 通过隐式文化对齐奖励模型消除文生图评估中的偏差

    Debiasing Text-to-Image Evaluation via Implicit Cultural Alignment Reward Modeling

    [https://arxiv.org/abs/2607.15740](https://arxiv.org/abs/2607.15740)

    本文提出了一种基于轻量级多模态大语言模型的隐式文化对齐奖励模型，通过跳跃连接交叉注意力机制，解决了文生图评估中文化偏差和实时扩展性问题。

    

    随着文生图（T2I）系统快速发展，评估合成内容的文化真实性对于公平可信的生成式AI变得越来越重要。现有的T2I评估指标和多模态评判器通常依赖视觉-语义表示，这些表示未能充分体现隐式文化规范，导致偏好判断存在偏差并遗漏细粒度文化线索。此外，基于视觉问答（VQA）的评估器通常依赖自回归文本生成，这限制了其在实时奖励建模中的可扩展性。为解决这些限制，我们引入了一种基于42亿参数轻量级多模态大语言模型（MLLM）的隐式文化对齐奖励模型。我们的框架将隐式文化探针与跳跃连接交叉注意力（SkipCA）机制相结合，使后期语义特征能够直接关注早期特征。

    arXiv:2607.15740v2 Announce Type: replace-cross  Abstract: As Text-to-Image (T2I) systems rapidly advance, evaluating the cultural authenticity of synthesized content has become increasingly important for fair and trustworthy generative AI. Existing T2I evaluation metrics and multimodal judges often rely on visual-semantic representations that underrepresent implicit cultural norms, leading to biased preference judgments and the omission of fine-grained cultural cues. In addition, visual question answering (VQA)-based evaluators typically depend on autoregressive text generation, which limits their scalability for real-time reward modeling. To address these limitations, we introduce an Implicit Cultural Alignment Reward Model built upon a lightweight 4.2-billion-parameter Multimodal Large Language Model (MLLM). Our framework integrates an Implicit Cultural Probe with a Skip-connection Cross-Attention (SkipCA) mechanism, enabling late-stage semantic features to directly attend to early-
    
[^252]: 主观风险分解：不确定性量化的新视角

    Subjective Risk Decomposition: A New View for Uncertainty Quantification

    [https://arxiv.org/abs/2607.15196](https://arxiv.org/abs/2607.15196)

    该论文提出将不确定性度量视为主观风险分解的产物而非基本原语，证明了基于严格恰当损失对主观风险进行分解即可推导出认知不确定性与偶然不确定性，从而为不确定性量化提供了统一的理论框架和新范式。

    

    我们提出了一种关于不确定性量化的新颖观点。不确定性度量并非需要公理和论证的基本原语，而是更高层次建模决策所产生的结果。我们展示了如何通过基于严格恰当损失的主观风险分解来推导认知不确定性和偶然不确定性度量。反向交叉熵提供了一个突出的例子，其分解能够恢复经典的信息论不确定性项。同样的方法还恢复了不确定性量化文献中此前提出的众多度量，为它们提供了一个共同的理论基础。这启示了一种新的不确定性量化方法：给定建模场景和严格恰当损失，相应的认知与偶然不确定性项便可由主观风险分解诱导产生。随后我们将这一观点扩展至学习理论：我们引入并分析了超额风险、近似误差和（认知）估计误差等概念的主观风险类似物

    arXiv:2607.15196v3 Announce Type: replace-cross  Abstract: We present a novel viewpoint for uncertainty quantification. Uncertainty measures are not primitives, in need of axioms and argumentation, but instead consequences, of higher-level modelling decisions. We show how epistemic and aleatoric uncertainty measures can be derived via decomposition of a subjective risk, based on a strictly proper loss. Reverse cross entropy provides a prominent example, where decomposition recovers the classic information-theoretic uncertainty terms. The same approach recovers numerous measures previously proposed across the UQ literature, providing them a common theoretical foundation. This suggests a new approach to UQ: given a modelling scenario and strictly proper loss, the corresponding epistemic and aleatoric terms are induced by the subjective-risk decomposition. We then extend our view to learning theory: we introduce and analyse subjective risk analogues of excess risk, approximation error and
    
[^253]: FPGN：基于可微分查找表重新定义超快速可编程门神经网络加速

    FPGN: Redefining Ultra-Fast Programmable Gate-based Neural Acceleration with Differentiable LUTs

    [https://arxiv.org/abs/2607.08427](https://arxiv.org/abs/2607.08427)

    该论文提出FPGN框架，通过可微分查找表将FPGA的LUT直接作为可学习神经元，重新定义了超快速可编程门神经网络加速，旨在实现纳秒级推理延迟。

    

    arXiv:2607.08427v2 公告类型： replace-cross 摘要：为深度神经网络（DNN）实现纳秒级推理延迟已成为延迟关键型应用的核心架构需求。尽管现场可编程门阵列（FPGA）为低延迟推理提供了有前景的硬件基础，但传统FPGA加速器仍以算术运算为中心，主要将查找表（LUT）用作数值运算器和外围逻辑的构建模块。与此不同，近期出现的LUT原生神经网络将LUT视为可学习的神经元，展现出利用其内在逻辑表达能力的理论潜力。然而，现有方法大多局限于算法层面的优化，未能将这一理论潜力转化为高性能的FPGA加速器。具体而言，现有方法的可微分公式无法忠实地匹配FPGA的LUT硬件原语，其缺乏物理感知的拓扑结构损害了可布线性和时序收敛，并且它们缺乏自动化优化……（摘要内容不完整）

    arXiv:2607.08427v2 Announce Type: replace-cross  Abstract: Achieving nanosecond-scale inference latency for deep neural networks (DNNs) has become a primary architectural concern for latency-critical applications. While Field-Programmable Gate Arrays (FPGAs) offer a promising substrate for low-latency inference, conventional FPGA accelerators remain arithmetic-centric, using LUTs primarily as building blocks for numerical operators and peripheral logic. In contrast, recent LUT-native neural networks treat LUTs as learnable neurons, revealing promising theoretical potential to exploit their intrinsic logic expressivity. However, existing methods are largely confined to algorithmic optimizations, failing to translate this theoretical potential into high-performance FPGA accelerators. Specifically, their differentiable formulations do not faithfully match FPGA LUT primitives, their physically-unaware topologies compromise routability and timing closure, and their lack of automated optimiz
    
[^254]: 运行级地球观测分类器的先验匹配评估：一种三数字报告方法——以Sentinel-1内波检测为例进行演示

    Prior-matched evaluation of operational Earth-observation classifiers: a three-number reporting method demonstrated on Sentinel-1 internal-wave detection

    [https://arxiv.org/abs/2607.07146](https://arxiv.org/abs/2607.07146)

    论文揭示了类别先验不匹配导致平衡测试集上报告的精确率（0.794）严重高估真实运行精确率（0.192），证明这是评估问题而非训练问题，并提出一种基于三个数字的先验匹配报告方法。

    

    内波服务系统对Sentinel-1波模式数据存档进行内孤立波筛查，并将检测结果分发给专家进行裁定，而专家的裁定时间正是该项目所需要节约的资源。由于注意力是错误的代价，精确率至关重要。该分类器在一对一的类别平衡下进行训练和报告，而这一平衡是在得知实际运行比率之前预先设定的。如今实际运行比率已经显现：大约每二十个场景中只有一个含内波，因此平衡测试集上的得分严重高估了验证人员在真实工作中所面对的精确率。一个在平衡测试集上精确率达0.794的模型，在实际运行中精确率仅为0.192：这一差距是在错误先验下进行报告所造成的系统性伪影，而大多数研究所引用的指标对此无法察觉。我们证明这种不匹配是一个披着训练问题外衣的评估问题——在固定召回率下，先验校正与概率校准均无法改变精确率——并以一种基于三个数字的先验匹配报告方法作为解决方案：

    arXiv:2607.07146v4 Announce Type: replace  Abstract: The Internal Waves Service screens the Sentinel-1 Wave-mode archive for internal solitary waves, routing detections to experts whose adjudication time is the resource the effort exists to conserve. Because attention is the cost of error, precision leads. Its classifier was trained and reported at a one-to-one class balance, fixed before the operational rate could be known. That rate has since emerged at roughly one scene in twenty, and a balanced-test score badly overstates the precision a validator meets. A model that scores 0.794 balanced-test precision scores 0.192 in real operation: the gap is a systematic artefact of reporting at the wrong prior, invisible to the metric most work quotes. We show the mismatch to be an evaluation problem in the costume of a training one at a fixed recall, prior correction and calibration cannot move precision, and answer it with a prior-matched reporting method based on three numbers: balanced-tes
    
[^255]: 延迟验证使多智能体大语言模型信念失稳：失稳阈值与最优纠错节点配置

    Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement

    [https://arxiv.org/abs/2606.27409](https://arxiv.org/abs/2606.27409)

    该论文将多智能体LLM系统中的延迟验证建模为带接地节点的延迟共识问题，通过接地拉普拉斯谱分解推导出验证剂量的闭式失稳阈值（延迟为二时为黄金比例的倒数），并基于超模目标给出贪婪(1-1/e)近似的纠错节点最优配置方法。

    

    多智能体大语言模型（LLM）系统通常依赖验证者与批评者智能体来抑制幻觉，但验证往往是延迟进行的。在延迟期间，错误声明可能在智能体网络中传播。我们将这一过程建模为带有接地纠错节点的图上的延迟共识问题。通过接地拉普拉斯算子的谱分解，我们推导出了验证剂量的闭式稳定性阈值：过强或过迟的纠错会将共识转化为振荡。最不稳定的情形发生在通信延迟与验证延迟恰好重合时；当延迟为二时，该阈值为黄金比例的倒数。同一框架还给出了一个超模的配置目标函数，以及一个具有贪婪(1-1/e)近似保证的规则，用于将有限的纠错预算分配给有影响力的节点。在五个开源模型上的实验证实了预测的剂量-延迟振荡现象。相比之下，接地的事实性回答使……

    arXiv:2606.27409v2 Announce Type: replace-cross  Abstract: Multi-agent large language model (LLM) systems often rely on verifier and critic agents to suppress hallucinations, but verification is delayed. During this delay, false claims can propagate through the agent network. We model this process as delayed consensus on a graph with grounded corrector nodes. Spectral decomposition by the grounded Laplacian yields a closed-form stability threshold for the verification dose: correction that is too strong or too delayed can turn consensus into oscillation. The most unstable regime occurs when the communication and verification delays coincide; for delay two, the threshold is the inverse golden ratio. The same framework gives a supermodular placement objective and a greedy (1-1/e)-approximation rule for assigning a limited corrector budget to influential nodes. Experiments across five open models confirm the predicted dose-delay oscillations. By contrast, grounded factual answering makes 
    
[^256]: Bergson：一个用于数据归因的开源库

    Bergson: An Open Source Library for Data Attribution

    [https://arxiv.org/abs/2606.11660](https://arxiv.org/abs/2606.11660)

    Bergson 是一个开源数据归因库，支持扩展至超大规模语言模型和预训练数据集，并首次开源实现了 MAGIC、SOURCE 和 TrackStar 三种前沿数据归因方法。

    

    数据归因是可解释性领域中一个前景广阔的方向，旨在通过训练数据对模型的影响来解释模型行为，其应用包括调试模型的不良行为以及训练数据集的筛选与整理。然而，大规模实施数据归因需要大量的工程投入，许多前沿技术缺乏开源工具和支持。Bergson 是一个开源库，旨在通过提供多种可扩展至超大规模语言模型和预训练数据集的技术，加速该领域的发展。该库原生支持磁盘级梯度存储和多节点分布式训练，并为研究人员提供了易用的实用工具。此外，我们首次开源实现了三种领先的数据归因方法：MAGIC、SOURCE 和 TrackStar。该库可在 https://github.com/EleutherAI/bergson 获取。

    arXiv:2606.11660v2 Announce Type: replace  Abstract: Data attribution is a promising field in interpretability that aims to explain model behavior through the influence of its training data, with applications including debugging undesirable model behavior and training dataset curation. However, significant engineering effort is required to perform it at scale, and many cutting edge techniques lack open-source tooling and support. Bergson is an open source library that aims to enable faster progress in the field by providing a host of techniques that scale to very large language models and pre-training datasets. The library natively supports on-disk gradient stores and multi-node distributed training, and provides quality of life tools for researchers. Finally, we introduce the first open-source implementations of three leading data attribution methods: MAGIC, SOURCE, and TrackStar. The library is available at https://github.com/EleutherAI/bergson .
    
[^257]: LargeMonitor：基于大型预训练模型的在线无任务持续学习监控

    LargeMonitor: Monitoring Online Task-Free Continual Learning via Large Pretrained Models

    [https://arxiv.org/abs/2606.09430](https://arxiv.org/abs/2606.09430)

    提出了LargeMonitor框架，利用大型预训练基础模型通过解耦的检测模块来监控和自主编排在线无任务持续学习，克服了现有训练耦合方法对分布漂移结构性起源不可知的局限。

    

    在线无任务持续学习（TFCL）要求智能体在严格的单次通过约束下，且在没有任何显式任务标识符的情况下，从无界的非平稳数据流中顺序积累知识。现有的在线TFCL范式主要依赖于参数高效的提示调优，或由训练耦合的优化动态（如经验损失波动或潜在距离演化）驱动的动态结构扩展。因此，这些训练耦合的求解器对分布漂移的结构性起源保持不可知，在根本不同的流式变化中机械地强制执行固定的策略。为解决这一差距，我们提出了LargeMonitor，一个利用大型预训练基础模型来自主编排无任务持续适应的框架。具体而言，LargeMonitor引入了一个解耦的检测模块，利用冻结且稳定的代表性表征……

    arXiv:2606.09430v2 Announce Type: replace-cross  Abstract: Online task-free continual learning (TFCL) requires intelligent agents to sequentially accumulate knowledge from an unbounded, non-stationary data stream under strict single-pass constraints and without any explicit task identifiers. Existing online TFCL paradigms primarily rely on parameter-efficient prompt tuning or dynamic structure expansion driven by training-coupled optimization dynamics, such as empirical loss fluctuations or evolving latent distances. As a result, these training-coupled solvers remain agnostic to the structural origins of distribution drift, mechanically enforcing a fixed strategy across fundamentally distinct streaming variations. To address this gap, we propose LargeMonitor, a framework that leverages large pretrained foundation models to autonomously orchestrate task-free continuous adaptation. Specifically, LargeMonitor introduces a decoupled detection module utilizing the frozen, stable representat
    
[^258]: 从“可能”到“是”：语言模型重写中的确定性失真

    From 'May' to 'Is': Certainty Distortion in Language Model Rewriting

    [https://arxiv.org/abs/2606.07951](https://arxiv.org/abs/2606.07951)

    该研究发现语言模型在重写科学和医学文本时会系统性地改变原文的确定性程度（如把“可能”改成“是”），这种失真影响高达75%的输出且呈不对称性。

    

    人们越来越多地依赖语言模型来塑造信念和驱动决策，包括讨论、重写和总结来自科学文章、新闻和医学报告的信息。然而，在这些领域中，一个论断的表达自信程度往往至关重要，但人们对语言模型是否能忠实地保留这种自信程度却知之甚少。在本研究中，我们调查了语言模型中的确定性失真现象，其定义为在意图保留原意的转换过程中，所表达的确定性发生的有意义的变化。我们提出了一种与人群层面的确定性判断相一致的语言模型评估指标。利用该指标，我们在科学和医学交流任务的背景下，刻画了不同规模和不同系列模型的确定性失真情况。我们的结果表明，确定性失真影响着高达75%的语言模型输出，并且在重写任务中呈现出系统性的不对称性。

    arXiv:2606.07951v2 Announce Type: replace-cross  Abstract: Humans increasingly turn to Language Models (LMs) in ways that shape beliefs and drive decisions, including discussing, rewriting, and summarizing information from scientific articles, news, and medical reports. However, in these domains, where it often matters how confidently a claim is expressed, little is known about whether LMs faithfully preserve the degree of confidence. In this work, we investigate certainty distortion in LMs, defined as meaningful changes in expressed certainty during transformations intended to preserve meaning. We propose an LM-based evaluation metric that is consistent with population-level judgments of certainty. Using this metric, we characterize certainty distortion across different sizes and families of models in the context of scientific and medical communication tasks. Our results show that certainty distortion affects up to 75% of LM outputs and is systematically asymmetric in rewriting tasks 
    
[^259]: KITE：一种融合文本、图像和知识图谱用于虚假新闻检测的三模态Transformer

    KITE: A Tri-Modal Transformer Integrating Text, Images, and Knowledge Graphs for Fake News Detection

    [https://arxiv.org/abs/2606.07651](https://arxiv.org/abs/2606.07651)

    KITE提出了一个三模态Transformer框架，通过跨模态注意力机制联合建模文本、图像和知识图谱表示，突破了以往仅依赖文本-图像融合或将外部知识用作后处理的局限，显著提升了虚假新闻检测能力。

    

    随着多模态虚假信息日益先进，无缝地融合了欺骗性文本、篡改的图像和事实错误的声明，传统的虚假新闻检测方法已逐渐落后。以往的大多数研究工作要么侧重于文本-图像融合，要么仅将外部知识作为后处理步骤加以应用，这限制了它们检测更深层次语义不一致性的能力。在本文中，我们提出了KITE（知识集成文本-图像编码器），这是一个三模态虚假新闻检测框架，能够联合建模文本、视觉和事实知识表示。KITE利用RoBERTa和CLIP分别进行语言和视觉编码，并采用图注意力网络（GAT）处理从Wikidata检索的结构化事实。KITE在多模态Transformer中使用跨模态注意力机制来整合文本、视觉和知识特征，帮助模型理解各模态之间的关联关系。此外，模型还生成特定模态的置信度分数...

    arXiv:2606.07651v2 Announce Type: replace  Abstract: Traditional fake news detection methods are falling behind as multimodal misinformation grows more advanced, seamlessly blending deceptive text, manipulated visuals, and factually incorrect claims. Most prior work focuses on text-image fusion or applies external knowledge only as a post-processing step, limiting their ability to detect deeper semantic inconsistencies. In this paper, we introduce KITE (Knowledge-Integrated Text-Image Encoder), a tri-modal fake news detection framework that jointly models textual, visual, and factual knowledge representations. KITE leverages Roberta and CLIP for linguistic and visual encoding, while a Graph Attention Network (GAT) processes structured facts retrieved from Wikidata. KITE uses cross-modal attention within a multimodal transformer to integrate text, visual, and knowledge features, helping it understand how each modality relates to one another. Modality-specific confidence scores are gener
    
[^260]: TargetSEC：基于唤醒条件化潜在风格扩散的即插即用野外语音情感转换

    TargetSEC: Plug-and-Play In-the-Wild Speech Emotion Conversion via Arousal-Conditioned Latent Style Diffusion

    [https://arxiv.org/abs/2606.07293](https://arxiv.org/abs/2606.07293)

    TargetSEC提出了一种以说话人身份和连续情感为条件的潜在风格扩散框架，通过在紧凑潜在空间中生成情感风格嵌入，实现了即插即用的野外语音情感转换，在转换效果和语音自然度上均优于或持平现有基线方法。

    

    语音情感转换（SEC）旨在将源语音的情感转换为目标情感，同时保留语音内容和说话人身份。由于训练数据的非平行特性以及复杂多变的真实世界声学环境，在野外（in-the-wild）数据上进行SEC极具挑战性。现有的固定时长方法要么难以有效转换情感（高质量但低转换率），要么会损害语音的自然度（高转换率但低质量）。我们提出了TargetSEC，这是一个由嵌入驱动的潜在扩散框架，它以说话人身份和连续情感（唤醒度）为条件，生成以情感为中心的风格嵌入。与在频谱图上进行扩散的方法不同，TargetSEC在紧凑的潜在空间中运行。在MSP-Podcast数据集上的实验表明，TargetSEC超越了现有的非时长基线方法，并在转换MSE和自然度方面达到甚至超过了时长预测基线，且无需显式的时间建模。

    arXiv:2606.07293v2 Announce Type: replace-cross  Abstract: Speech Emotion Conversion (SEC) aims to transform the emotion of a source utterance into a target emotion while preserving content and speaker identity. SEC on in-the-wild data is challenging due to the non-parallel nature of training data and complex real-world acoustics. Existing fixed-duration approaches either struggle to shift the emotion effectively (high quality, low conversion) or degrade speech naturalness (low quality, high conversion). We propose TargetSEC, an embedding-driven latent diffusion framework that generates emotion-focused style embeddings conditioned on speaker identity and continuous emotion. Unlike methods that diffuse over spectrograms, TargetSEC operates in a compact latent space. Experiments on the MSP-Podcast dataset show that TargetSEC outperforms current non-duration baselines and matches or exceeds the duration-prediction baseline in conversion MSE and naturalness without explicit temporal modeli
    
[^261]: Libra：面向智能体强化学习后训练的高效资源管理

    Libra: Efficient Resource Management for Agentic RL Post-Training

    [https://arxiv.org/abs/2606.03077](https://arxiv.org/abs/2606.03077)

    Libra 是一个面向智能体强化学习后训练的自适应运行时系统，通过因果引导的桶调度等互补机制，解决长尾轨迹主导完成时间以及 rollout 与训练阶段资源需求动态失衡的问题，实现高效资源管理。

    

    强化学习（RL）已成为将大语言模型（LLM）塑造为强大智能体的标准后训练范式。在智能体强化学习中，rollout 阶段在调用工具的同时生成轨迹，产生长尾且非平稳的工作负载，这暴露出两个根本性挑战。首先，由于响应分布呈长尾特性，一小部分轨迹主导了 rollout 的整体完成时间。其次，rollout 和训练在计算模式、内存需求以及对序列长度的敏感性上存在差异。随着策略不断演进，工作负载分布的变化进一步改变了两阶段的相对资源需求，使得难以在两个阶段之间维持均衡执行。我们提出了 Libra，一个面向智能体 RL 后训练的自适应运行时系统，包含两个互补组件：（1）通过因果引导的桶调度器进行阶段内调度，将请求路由至具有不同……

    arXiv:2606.03077v3 Announce Type: replace-cross  Abstract: Reinforcement learning (RL) has emerged as a standard post-training paradigm for shaping large language models (LLMs) into capable agents. In agentic RL, the rollout stage generates trajectories while invoking tools, producing long-tailed and non-stationary workloads that expose two fundamental challenges. First, due to the long-tailed response distribution, a small fraction of trajectories dominates rollout makespan.Second, rollout and training differ in their compute patterns, memory demands, and sensitivity to sequence length. As the policy evolves, shifts in the workload distribution further change their relative resource demands, making it difficult to maintain balanced execution across the two stages.   We present Libra, an adaptive runtime for agentic RL post-training with two complementary components: (1) intra-stage scheduling via a Causality-Guided Bucket Scheduler that routes requests across execution buckets with di
    
[^262]: 关于不完整U统计量中位数的有限样本集中性

    On Finite-sample Concentration of Median of Incomplete U-Statistics

    [https://arxiv.org/abs/2606.00661](https://arxiv.org/abs/2606.00661)

    本文证明了不完整U统计量中位数（MoIU）的有限样本浓度界，克服了此前仅能获得松散$O(n^{-1/4})$界的理论挑战，实现了更紧的收敛速率。

    

    中位数均值（MoM）是一种强大的技术，在理论上能在底层数据分布具有重尾特性（例如，仅假设具有前两阶有限矩）时，实现参数估计的接近亚高斯的有限样本速率。最近的研究将此技术推广到中位数随机化U统计量（MoRU）和中位数不完整U统计量（MoIU），用于估计重尾成对核的期望。在\citet{pmlr-v97-clemencon19a}中，已证明MoRU的浓度速率随样本量按$O(n^{-1/2})$缩放。然而，尽管后者具有计算优势，MoIU的有限样本界分析仍是一个重大的理论挑战。正如作者所指出的，直接应用McDiarmid不等式会产生$O(n^{-1/4})$阶的松散界。在本工作中，我们证明了MoIU估计的有限样本浓度界。

    arXiv:2606.00661v2 Announce Type: replace-cross  Abstract: Median-of-means (MoM) is a powerful technique that theoretically enables near sub-Gaussian finite-sample rate for parameter estimation when the underlying data distribution is heavy-tailed (e.g., assumed to have only two first finite moments). A recent work has extrapolated this technique to median-of-\textit{randomized}-U-Statistics (MoRU) and median-of-\textit{incomplete}-U-Statistics (MoIU) for estimating expectations of heavy-tailed pairwise kernels. In \citet{pmlr-v97-clemencon19a}, a concentration rate that scales like $O(n^{-1/2})$ with sample size has been proven for MoRU. However, despite the computational advantage of the latter, the analysis of finite-sample bound for MoIU remains a significant theoretical challenge. As noted by the authors, a straightforward application of McDiarmid's inequality yields a loose bound of order $O(n^{-1/4})$. In this work, we prove a finite-sample concentration bound for the MoIU estim
    
[^263]: 子空间分解的JEPA：在潜在世界模型中解耦任务进展与内容

    Subspace-Decomposed JEPAs: Disentangling Progression and Content in Latent World Models

    [https://arxiv.org/abs/2605.31111](https://arxiv.org/abs/2605.31111)

    SD-JEPA将JEPA潜在空间分解为正交的进展子空间和内容子空间，使两种防坍缩力量在不相交的坐标上加性组合而非相互竞争，从而在多个控制基准上提升了世界模型性能。

    

    联合嵌入预测架构（JEPAs）通过预测未来嵌入来学习紧凑的潜在世界模型，但其潜在空间中没有任何单一坐标被专门用于编码任务进展。我们将JEPA潜在空间划分为两个角色互不相交的正交子空间：一个由余弦间隔三元组损失塑造的低维进展子空间，以及一个由LeWM现有SIGReg目标正则化的高维内容子空间。我们证明了这两种防坍缩力量作用于互不相交的坐标上，因此它们以加法方式组合，而非在同一维度上相互竞争。我们的方法SD-JEPA在同等计算量下，在LeWM基线的大多数控制基准测试中优于LeWM基线，并在Push-T任务上超越了最强的非LeWM JEPA基线；子空间消融证伪实验证实了这种子空间划分是起支撑作用的关键要素。除规划任务之外，所得到的一维角度进展坐标还可作为场景感知的……

    arXiv:2605.31111v2 Announce Type: replace  Abstract: Joint-Embedding Predictive Architectures (JEPAs) learn compact latent world models by predicting future embeddings, but no single coordinate of the latent is designated to encode task progression. We carve the JEPA latent into two orthogonal subspaces with disjoint roles: a low-dimensional progression subspace shaped by a cosine-margin triplet loss, and a high-dimensional content subspace regularised by the existing SIGReg objective of LeWM. We prove that the two anti-collapse forces act on disjoint coordinates, so they compose additively rather than competing on the same dimensions. Our method, SD-JEPA improves over the LeWM baseline on the majority of its control benchmarks at matched compute, and outperforms the strongest non-LeWM JEPA baseline on Push-T; a subspace-ablation falsifier confirms the split is the load-bearing ingredient. Beyond planning, the resulting 1-D angular progression coordinate functions as a scene-aware comp
    
[^264]: EfficientTDMPC：改进的MPC目标函数实现样本高效的连续控制

    EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control

    [https://arxiv.org/abs/2605.16692](https://arxiv.org/abs/2605.16692)

    EfficientTDMPC通过动力学模型集成、跨不同展开深度平均回报估计以及对规划器目标施加不确定性惩罚来减少模型与价值网络的估计误差，并结合缓冲区数据新鲜度等实用改进，从而在连续控制任务中实现更高效的模型强化学习，并能更好地利用更高的更新-数据比。

    

    我们提出了EfficientTDMPC，这是一种基于TD-MPC算法家族构建的样本高效的模型强化学习方法，用于连续控制任务。该算法家族的核心是一个规划器，旨在找到能够最大化估计回报的动作序列。回报是通过学习到的模型网络和价值网络进行估计的，而这两者都可能引入误差。EfficientTDMPC提出通过两种方式来减少这种误差。首先，它引入了动力学模型集成方法，对不同模型以及不同展开深度上的回报估计进行平均。其次，它增加了对规划器目标施加不确定性惩罚的选项，从而得到一个能够避免回报估计不确定的动作的规划器。此外，它还添加了一些实用改进，以提高缓冲区数据的新鲜度并减少计算量。最后，我们发现这些贡献使EfficientTDMPC能够从更高的更新-数据比（UTD）中获益更多，进一步（提升性能）……

    arXiv:2605.16692v3 Announce Type: replace-cross  Abstract: We introduce EfficientTDMPC, a sample-efficient model-based reinforcement learning method for continuous control built on the TD-MPC family of algorithms. Central to this family is a planner that aims to find an action sequence that maximizes the estimated return. The return is estimated using a learned model and value networks, each of which can introduce error. EfficientTDMPC proposes to reduce this error in two ways. First, it introduces an ensemble of dynamics models and averages the return estimates across those models and across different rollout depths. Second, it adds the option to apply an uncertainty penalty to the planner objective, yielding a planner that avoids actions with uncertain return estimates. It then adds practical improvements which increase buffer data freshness and reduce compute. Lastly, we find that our contributions enable EfficientTDMPC to benefit more from a higher update-to-data (UTD) ratio, furth
    
[^265]: 如何在RL后训练中压缩KV缓存？面向内存高效对齐的影子掩码蒸馏方法

    How to Compress KV Cache in RL Post-Training? Shadow Mask Distillation for Memory-Efficient Alignment

    [https://arxiv.org/abs/2605.06850](https://arxiv.org/abs/2605.06850)

    该论文揭示了KV缓存压缩在RL后训练rollout阶段会引入被优化不稳定性急剧放大的离策略偏差问题，并提出影子掩码蒸馏方法以实现内存高效的模型对齐。

    

    强化学习（RL）已成为解锁大语言模型（LLM）高级推理能力的关键范式，涵盖了RLHF和RLAIF等框架。无论采用何种具体的优化算法（如PPO、GRPO或Online DPO），在线RL本质上都需要一个探索性的轨迹生成（rollout）阶段。然而，对于长上下文推理任务，这一rollout阶段由于过高的键值（KV）缓存占用而造成了严重的“内存墙”问题。虽然在rollout过程中应用KV缓存压缩可以缓解这种内存开销，但它会引入关键的离策略偏差。尽管现代KV压缩方法在标准推理过程中通常几乎无损，但即使是微小的近似误差也会被RL优化固有的不稳定性急剧放大。具体而言，采样器在稀疏上下文下生成响应，而学习器则使用截断后的KV缓存进行参数更新……

    arXiv:2605.06850v2 Announce Type: replace-cross  Abstract: Reinforcement Learning (RL) has emerged as a crucial paradigm for unlocking the advanced reasoning capabilities of Large Language Models (LLMs), encompassing frameworks like RLHF and RLAIF. Regardless of the specific optimization algorithm (e.g., PPO, GRPO, or Online DPO), online RL inherently requires an exploratory trajectory generation (rollout) phase. However, for long-context reasoning tasks, this rollout phase imposes a severe ``memory wall'' due to the exorbitant Key-Value (KV) cache footprint. While applying KV cache compression during rollouts mitigates this memory overhead, it induces a critical off-policy bias. Although modern KV compression is often nearly lossless during standard inference, even minuscule approximation errors are drastically amplified by the inherent instability of RL optimization. Specifically, the sampler generates responses under a sparse context, whereas the learner updates parameters using the
    
[^266]: 使用预测性掩码图自编码器预测单个NetFlow流量

    Forecasting Individual NetFlows using a Predictive Masked Graph Autoencoder

    [https://arxiv.org/abs/2604.20483](https://arxiv.org/abs/2604.20483)

    本文提出一种基于预测性掩码图自编码器的GNN模型，通过滑动窗口构建包含IP、端口和连接节点的异构双向图来预测网络流级别流量，在识别连接所依附的端口和IP方面表现卓越。

    

    本文提出一个概念验证性的图神经网络（GNN）模型，该模型通过精确建模图结构和连接特征，能够成功预测网络流级别（NetFlow）的流量。我们使用滑动窗口将网络流量切分为大小相等的异构双向图，其中包含IP、端口和连接节点。随后利用GNN来建模图结构和连接特征的演化过程。我们的方法在识别连接所依附的端口和IP方面表现出卓越的结果，同时其特征重构能力与强大的预测基线相比仍具有竞争力。总体而言，我们的工作展示了GNN在逐流NetFlow预测中的应用价值。

    arXiv:2604.20483v3 Announce Type: replace-cross  Abstract: In this paper, we propose a proof-of-concept Graph Neural Network model that can successfully predict network flow-level traffic (NetFlow) by accurately modelling the graph structure and the connection features. We use sliding-windows to split the network traffic in equal-sized heterogeneous bidirectional graphs containing IP, Port, and Connection nodes. We then use the GNN to model the evolution of the graph structure and the connection features. Our approach shows superior results when identifying the Port and IP to which connections attach, while feature reconstruction remains competitive with strong forecasting baselines. Overall, our work showcases the use of GNNs for per-flow NetFlow prediction.
    
[^267]: 强化学习的Wasserstein表述：策略优化的最优传输视角

    Wasserstein Formulation of Reinforcement Learning. An Optimal Transport Perspective on Policy Optimization

    [https://arxiv.org/abs/2604.14765](https://arxiv.org/abs/2604.14765)

    本文提出了一个基于最优传输理论的强化学习几何框架，将策略视为映入Wasserstein空间的映射，利用黎曼几何结构和Otto微积分构建梯度流并计算能量的梯度与Hessian矩阵，为策略优化提供了形式化的二阶分析工具。

    

    我们提出了一个强化学习（RL）的几何框架，将策略视为映入动作概率的Wasserstein空间的映射。首先，我们定义了一个由平稳分布诱导的黎曼结构，并在一般背景下证明了其存在性。然后，我们定义了策略的切空间并刻画了测地线，特别解决了从状态空间映射到动作空间上概率测度切空间的向量场的可测性问题。接下来，我们表述了一个一般性的强化学习优化问题，并利用Otto微积分构建了梯度流。我们计算了能量的梯度和Hessian矩阵，提供了形式化的二阶分析。最后，我们通过低维问题的数值例子演示了该方法，直接从我们的理论形式化中计算梯度。对于高维问题，我们使用神经网络对策略进行参数化并进行优化。

    arXiv:2604.14765v2 Announce Type: replace  Abstract: We present a geometric framework for Reinforcement Learning (RL) that views policies as maps into the Wasserstein space of action probabilities. First, we define a Riemannian structure induced by stationary distributions, proving its existence in a general context. We then define the tangent space of policies and characterize the geodesics, specifically addressing the measurability of vector fields mapped from the state space to the tangent space of probability measures over the action space. Next, we formulate a general RL optimization problem and construct a gradient flow using Otto's calculus. We compute the gradient and the Hessian of the energy, providing a formal second-order analysis. Finally, we illustrate the method with numerical examples for low-dimensional problems, computing the gradient directly from our theoretical formalism. For high-dimensional problems, we parameterize the policy using a neural network and optimize 
    
[^268]: HINTBench：长时程智能体内在非攻击轨迹基准

    HINTBench: Horizon-agent Intrinsic Non-attack Trajectory Benchmark

    [https://arxiv.org/abs/2604.13954](https://arxiv.org/abs/2604.13954)

    该论文提出了HINTBench，一个针对智能体内在非攻击风险的基准数据集，包含596条平均24步的智能体轨迹，支持风险检测、风险步骤定位和内在失效类型识别三项安全审计任务。

    

    现有的智能体安全评估主要聚焦于外部诱发的风险。然而，即使在良性条件下，智能体仍可能进入不安全的轨迹。我们通过“内在风险”的视角研究这一互补但未被充分探索的场景：内在失效保持潜伏状态，在长时程执行过程中不断传播，并最终导致高后果的结果。为了评估这一场景，我们提出了“非攻击内在风险审计”这一面向防护器的安全评估任务，并推出HINTBench——一个包含596条智能体轨迹的基准数据集，其中包含400条合成风险轨迹、136条合成安全轨迹、30条重构的真实世界风险轨迹以及30条重构的真实世界安全轨迹，平均轨迹长度为24.0步。HINTBench支持三项任务：风险检测、风险步骤定位和内在失效类型识别，其标注的组织方式（原文此处截断）……

    arXiv:2604.13954v2 Announce Type: replace-cross  Abstract: Existing agent-safety evaluation has focused mainly on externally induced risks. Yet agents may still enter unsafe trajectories under benign conditions. We study this complementary but underexplored setting through the lens of \emph{intrinsic} risk, where intrinsic failures remain latent, propagate across long-horizon execution, and eventually lead to high-consequence outcomes. To evaluate this setting, we introduce \emph{non-attack intrinsic risk auditing}, a guard-oriented safety evaluation task, and present \textbf{HINTBench}, a benchmark of 596 agent trajectories, comprising 400 synthetic risky trajectories, 136 synthetic safe trajectories, 30 reconstructed real-world risky trajectories, and 30 reconstructed real-world safe trajectories, with an average length of 24.0 steps. HINTBench supports three tasks: risk detection, risk-step localization, and intrinsic failure-type identification, with annotations organized under a u
    
[^269]: 通过稳定对角曲率估计实现鲁棒的超低比特后训练量化

    Robust Ultra Low-Bit Post-Training Quantization via Stable Diagonal Curvature Estimate

    [https://arxiv.org/abs/2604.13806](https://arxiv.org/abs/2604.13806)

    DASH-Q通过稳定的对角Hessian曲率估计和迭代加权最小二乘，实现了对采样噪声鲁棒的超低比特后训练量化，在极少量校准数据下于五个LLM模型上平均提升零样本准确率7.01%、最高提升14.01%。

    

    大型语言模型（LLM）被广泛应用于众多领域，但其庞大的规模使部署面临挑战。后训练量化（PTQ）通过利用小型校准集，在无需重新训练的情况下减少内存占用。近期基于Hessian矩阵的PTQ方法通过跨通道依赖性来补偿量化误差，但由于校准数据有限导致曲率估计存在噪声，这类方法在低比特宽度下性能会退化。我们提出了DASH-Q，一个采用对角Hessian近似和迭代加权最小二乘法的鲁棒PTQ框架。通过舍弃易受噪声影响的依赖性，DASH-Q能够过滤采样噪声，同时优先保留显著特征能量。在超低比特设置下，我们的方法优于其他PTQ基线，在五个基线LLM模型上平均提升零样本准确率7.01%，相对最强基线最高提升14.01%，同时在极少量校准数据下展现出鲁棒且稳定的性能。

    arXiv:2604.13806v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are widely used across many domains, but their scale makes deployment challenging. Post-Training Quantization (PTQ) reduces memory footprint without retraining by leveraging a small calibration set. Recent Hessian-based PTQ methods compensate quantization error via cross-channel dependencies, but such approaches degrade at low bit-widths due to noisy curvature estimates from limited calibration data. We propose DASH-Q, a robust PTQ framework using diagonal Hessian approximation and iterative weighted least squares. By discarding noise-prone dependencies, DASH-Q filters sampling noise while prioritizing the preservation of salient feature power. We outperform other PTQ baselines in ultra low-bit regime, improving zero-shot accuracy by 7.01% on average and up to 14.01% over the strongest baselines across five baseline LLM models, while showing robust and stable performance with very small calibration data.
    
[^270]: VISTA：基于自蒸馏的验证信息引导轨迹自适应

    VISTA: Validation-Informed Trajectory Adaptation via Self-Distillation

    [https://arxiv.org/abs/2604.12044](https://arxiv.org/abs/2604.12044)

    VISTA提出了一种在线自蒸馏框架，通过验证信息引导的边际覆盖率得分识别保留专业能力的早期模型状态（专家锚点），并以覆盖率加权方式在线集成这些锚点来正则化训练过程，从而解决“轨迹偏差”导致的优化失败问题并提升模型鲁棒性与泛化能力。

    

    深度学习模型即使在验证准确率很高的情况下也可能收敛到次优解，这掩盖了一种我们称之为“轨迹偏差”的优化失败现象。这是因为随着训练的进行，模型可能会放弃对特定数据子群体具有高泛化能力的状态，从而丢弃先前学习到的潜在特征，且不会触发经典的过拟合信号。为了解决这一问题，我们提出了VISTA，这是一个在线自蒸馏框架，用于强制优化轨迹上的一致性。利用基于验证信息的边际覆盖率得分，VISTA识别出“专家锚点”，即那些对不同数据区域仍保持专门能力的早期模型状态。这些锚点的覆盖率加权集成在训练过程中在线整合，对损失景观进行正则化并保留已掌握的知识。在多个基准测试上的评估表明，VISTA表现出更高的鲁棒性和泛化能力。

    arXiv:2604.12044v2 Announce Type: replace-cross  Abstract: Deep learning models may converge to suboptimal solutions despite strong validation accuracy, masking an optimization failure we term Trajectory Deviation. This is because as training proceeds, models can abandon high generalization states for specific data sub-populations, thus discarding previously learned latent features without triggering classical overfitting signals. To address this problem we introduce VISTA, an online self-distillation framework that enforces consistency along the optimization trajectory. Using a validation-informed Marginal Coverage score, VISTA identifies expert anchors, which are earlier model states that retain specialized competence over distinct data regions. A coverage-weighted ensemble of these anchors is integrated online during training, regularizing the loss landscape and preserving mastered knowledge. When evaluated across multiple benchmarks, VISTA demonstrates improved robustness and gener
    
[^271]: 面向不确定性下序贯决策的深度学习：基础、框架与前沿

    Deep Learning for Sequential Decision Making under Uncertainty: Foundations, Frameworks, and Frontiers

    [https://arxiv.org/abs/2604.11507](https://arxiv.org/abs/2604.11507)

    本教程以运筹学/管理科学（OR/MS）为核心视角，系统性地连接了深度学习神经架构与不确定性下序贯决策的OR/MS方法，其核心观点是深度学习是对优化的补充而非替代。

    

    人工智能（AI）正日益超越预测的范畴，转而在复杂、不确定且动态的环境中支持决策。这一转变使其与运筹学和管理科学（OR/MS）形成了天然的交汇点，后者长期以来一直为不确定性下的序贯决策提供方法论基础。与此同时，深度学习的进展——包括前馈神经网络、循环架构、Transformer、大语言模型（LLM）以及深度强化学习——扩展了面向大规模决策的数据驱动建模方法。本教程以运筹学/管理科学为核心视角，探讨用于不确定性下序贯决策的深度学习，旨在架起神经架构与OR/MS决策方法之间的桥梁。其核心前提是：深度学习是对优化的补充，而非替代。深度学习带来了适应性和可扩展的近似能力，而OR/MS则提供了……（摘要原文在此处截断）

    arXiv:2604.11507v2 Announce Type: replace-cross  Abstract: Artificial intelligence (AI) is moving increasingly beyond prediction to support decisions in complex, uncertain, and dynamic environments. This shift creates a natural intersection with operations research and management science (OR/MS), which has long provided methodological foundations for sequential decision making under uncertainty. At the same time, deep learning advances, including feedforward neural networks, recurrent architectures, transformers, large language models (LLMs), and deep reinforcement learning, have expanded data-driven modeling for large-scale decisions. This tutorial presents an OR/MS-centered perspective on deep learning for sequential decision making under uncertainty, bridging neural architectures and OR/MS approaches to decision making. Its premise: deep learning complements optimization rather than replacing it. Deep learning brings adaptability and scalable approximation, whereas OR/MS provides th
    
[^272]: 在小缓冲区持续学习中利用互补嵌入进行重放选择

    Leveraging Complementary Embeddings for Replay Selection in Continual Learning with Small Buffers

    [https://arxiv.org/abs/2604.08336](https://arxiv.org/abs/2604.08336)

    提出MERS方法，通过基于图的方式融合监督与自监督互补嵌入来选择重放样本，在小缓冲区持续学习中以零额外参数开销显著超越现有最先进的样本选择策略。

    

    灾难性遗忘仍然是持续学习（Continual Learning, CL）中的关键挑战。在内存严重受限的基于重放的持续学习中，性能在很大程度上取决于重放缓冲区的样本选择策略。大多数现有方法使用在监督目标下学习到的嵌入来构建记忆缓冲区。然而，与类别无关的自监督表示往往编码了丰富的、与类别相关的语义，而这些语义常被忽视。我们提出了一种新方法——多重嵌入重放选择（Multiple Embedding Replay Selection, MERS），该方法用一种基于图的模块替换缓冲区选择模块，从而整合监督嵌入与自监督嵌入。实证结果表明，在一系列持续学习算法中，MERS相较于最先进的选择策略取得了一致的改进，其中在低内存场景下提升尤为显著。在CIFAR-100和TinyImageNet数据集上，MERS在不增加模型参数或增大重放缓冲区的情况下超越了单一嵌入基线。

    arXiv:2604.08336v2 Announce Type: replace  Abstract: Catastrophic forgetting remains a key challenge in Continual Learning (CL). In replay-based CL with severe memory constraints, performance critically depends on the sample selection strategy for the replay buffer. Most existing approaches construct memory buffers using embeddings learned under supervised objectives. However, class-agnostic, self-supervised representations often encode rich, class-relevant semantics that are overlooked. We propose a new method, Multiple Embedding Replay Selection, MERS, which replaces the buffer selection module with a graph-based approach that integrates both supervised and self-supervised embeddings. Empirical results show consistent improvements over SOTA selection strategies across a range of continual learning algorithms, with particularly strong gains in low-memory regimes. On CIFAR-100 and TinyImageNet, MERS outperforms single-embedding baselines without adding model parameters or increasing re
    
[^273]: 基于通用微分方程逼近Maxey-Riley-Gatignol方程中的Basset力

    Approximation of the Basset force in the Maxey-Riley-Gatignol equations via universal differential equations

    [https://arxiv.org/abs/2604.08194](https://arxiv.org/abs/2604.08194)

    本文提出利用通用微分方程和神经网络来逼近MaRGE方程中的Basset历史力项，将其转化为可用Runge-Kutta等标准数值方法求解的常微分方程组。

    

    Maxey-Riley-Gatignol方程（MaRGE）用于模拟流体中球形惯性粒子的运动。该方程包含Basset力，这是一个积分项，用于建模由尾流形成和边界层效应引起的历史效应。这使得作用于粒子上的力依赖于其过去的运动轨迹，从而使MaRGE的数值求解变得复杂。因此，尽管有大量证据表明Basset力对所建模粒子的运动模式具有定量和定性两方面的影响，它却经常被忽略。利用通用微分方程的概念，我们提出了一种通过神经网络对历史项进行逼近的方法，将MaRGE近似为一个常微分方程组，从而可以使用Runge-Kutta方法等标准数值求解器进行求解。

    arXiv:2604.08194v2 Announce Type: replace  Abstract: The Maxey-Riley-Gatignol equations (MaRGE) model the motion of spherical inertial particles in a fluid. They contain the Basset force, an integral term which models history effects due to the formation of wakes and boundary layer effects. This causes the force that acts on a particle to depend on its past trajectory and complicates the numerical solution of MaRGE. Therefore, the Basset force is often neglected, despite substantial evidence that it has both quantitative and qualitative impact on the movement patterns of modelled particles. Using the concept of universal differential equations, we propose an approximation of the history term via neural networks which approximates MaRGE by a system of ordinary differential equations that can be solved with standard numerical solvers like Runge-Kutta methods.
    
[^274]: 储备池计算网络中的主导流形研究

    On Dominant Manifolds in Reservoir Computing Networks

    [https://arxiv.org/abs/2604.05967](https://arxiv.org/abs/2604.05967)

    该论文从理论上证明了储备池计算网络在时间序列预测训练中会涌现出低维主导流形，并将储备池的主导特征值和特征向量与后向动态模态分解矩阵的谱联系起来，从而给出了系统后向时间Koopman算子的有限维近似。

    

    理解训练如何塑造循环网络动力学的几何结构是时间序列建模中的一个核心问题。我们研究了储备池计算网络在时间预测任务训练过程中低维主导流形的涌现。对于一个一般线性连续时间储备池在无限数据极限下的情形，我们证明训练数据会生成训练后储备池的一个不变子空间，其维数等于主导模态的数量。随后我们专门研究一种简化的对角线性储备池，将其主导特征值和特征向量与后向动态模态分解（DMD）矩阵的谱联系起来，该矩阵给出了生成训练数据的系统的后向时间Koopman算子的有限维近似。我们通过仿真展示了这些主导模态在训练过程中的涌现，并讨论了如何将该分析扩展到非线性储备池计算。（原文摘要在此处截断）

    arXiv:2604.05967v2 Announce Type: replace  Abstract: Understanding how training shapes the geometry of recurrent network dynamics is a central problem in time-series modeling. We study the emergence of low-dimensional dominant manifolds in the training of Reservoir Computing (RC) networks for temporal forecasting tasks. For a general linear continuous-time reservoir in the infinite-data limit, we show that the training data generate an invariant subspace of the trained reservoir, whose dimension equals the number of dominant modes. We then specialize to a simplified diagonal linear reservoir, where we link the dominant eigenvalues and eigenvectors to the spectrum of a backward Dynamic Mode Decomposition matrix, which yields a finite-dimensional approximation of the backward-time Koopman operator of the system generating the training data. We illustrate the emergence of these dominant modes during training in simulation, and discuss how the analysis may be extended to nonlinear RC via t
    
[^275]: 正定矩阵锥上Bregman散度的对称化：使用哪种均值以及为什么

    Symmetrizing Bregman Divergence on the Cone of Positive Definite Matrices: Which Mean to Use and Why

    [https://arxiv.org/abs/2603.28917](https://arxiv.org/abs/2603.28917)

    该论文揭示了正定矩阵锥上对称化Bregman散度的变分原理，证明前向对称化的规范均值是原始空间上的算术平均，而反向对称化的规范均值是对偶空间算术平均的拉回，在常用情形下分别对应算术、对数欧几里得和调和平均。

    

    本工作揭示了在正定矩阵锥上，对由一般镜像映射所诱导的Bregman散度进行对称化背后的变分原理。我们证明，计算这种对称化的规范均值可以表述为：在公理化定义、满足特定性质的一组均值泛函上，最小化目标对称化散度。对于前向对称化，我们证明对于正定锥上的任何镜像映射，原始空间上的算术平均都是规范均值。对于反向对称化，我们证明规范均值是对偶空间上的算术平均再拉回到原始空间所得的结果。将这一结果应用于实践中常用的三种镜像映射，我们证明了在这些情形下，反向对称化的规范均值分别是算术平均、对数欧几里得平均和调和平均。我们的结果增进了对现有对称化方法的理解。

    arXiv:2603.28917v3 Announce Type: replace-cross  Abstract: This work uncovers variational principles behind symmetrizing the Bregman divergences induced by generic mirror maps over the cone of positive definite matrices. We show that computing the canonical means for this symmetrization can be posed as minimizing the desired symmetrized divergences over a set of mean functionals defined axiomatically to satisfy certain properties. For the forward symmetrization, we prove that the arithmetic mean over the primal space is canonical for any mirror map over the positive definite cone. For the reverse symmetrization, we show that the canonical mean is the arithmetic mean over the dual space, pulled back to the primal space. Applying this result to three common mirror maps used in practice, we show that the canonical means for reverse symmetrization, in those cases, turn out to be the arithmetic, log-Euclidean and harmonic means. Our results improve understanding of existing symmetrization p
    
[^276]: AMIGO：智能体多图像定位预言基准

    AMIGO: Agentic Multi-Image Grounding Oracle Benchmark

    [https://arxiv.org/abs/2603.28662](https://arxiv.org/abs/2603.28662)

    AMIGO是一个长时程智能体基准，要求视觉语言模型通过一系列是/否问题在视觉相似的图像画廊中识别隐藏目标，以评估其在不确定性下的问题选择、跨轮次约束跟踪和细粒度判别能力。

    

    智能体视觉语言模型越来越多地通过扩展的交互来执行任务，但大多数评估仍然聚焦于单图像、单轮次的正确性。我们提出了AMIGO（智能体多图像定位预言基准），这是一个用于在视觉相似图像画廊中进行隐藏目标识别的长时程基准。在AMIGO中，预言者私下选择一张目标图像，模型必须通过提出一系列聚焦于属性的是/否问题来找出该目标，整个过程遵循严格的协议：系统返回“是/否/不确定”反馈，并对无效动作处以“跳过”惩罚。这一设置重点考察：(i) 不确定性下的问题选择，(ii) 跨轮次的一致性约束跟踪，以及 (iii) 随证据累积的细粒度判别能力。我们通过“猜我最喜欢的裙子”任务实例化AMIGO，并使用涵盖目标识别等方面的指标对开源视觉语言模型进行评估。

    arXiv:2603.28662v2 Announce Type: replace-cross  Abstract: Agentic vision-language models increasingly act through extended interactions, but most evaluations still focus on single-image, single-turn correctness. We introduce \textbf{AMIGO} (\textbf{A}gentic \textbf{M}ulti-\textbf{I}mage \textbf{G}rounding \textbf{O}racle Benchmark), a long-horizon benchmark for \emph{hidden-target} identification over galleries of visually similar images. In AMIGO, the oracle privately selects a target image, and the model must recover it by asking a sequence of attribute-focused Yes/No questions under a strict protocol that returns Yes/No/Unsure feedback and penalizes invalid actions with \emph{Skip}. This setting stresses (i) question selection under uncertainty, (ii) consistent constraint tracking across turns, and (iii) fine-grained discrimination as evidence accumulates. We instantiate AMIGO with the \textit{Guess My Preferred Dress} task and evaluate open-source VLMs with metrics covering identi
    
[^277]: Q-BIOLAT：面向基于QUBO优化的二进制潜在蛋白质适应度景观

    Q-BIOLAT: Binary Latent Protein Fitness Landscapes for QUBO-Based Optimization

    [https://arxiv.org/abs/2603.27526](https://arxiv.org/abs/2603.27526)

    该论文提出Q-BioLat框架，将蛋白质语言模型嵌入转换为紧凑二进制编码并拟合QUBO代理模型用于蛋白质适应度优化，核心发现是逐点预测精度相当的二进制表示可能诱导截然不同的汉明邻域和优化搜索轨迹，强调表示选择对优化行为的决定性影响。

    

    蛋白质适应度优化是一个离散搜索问题，而用于预测的表示方式同时也决定了优化器所遍历的邻域图。我们提出了Q-BioLat，这是一个将预训练蛋白质语言模型嵌入映射为紧凑二进制编码的框架，并拟合包含单变量和成对潜在交互项的二次无约束二元优化（QUBO）代理模型。我们的核心贡献是一种面向优化的表示视角：在逐点预测精度上相似的二进制编码，可能诱导出不同的汉明邻域、局部最优解和搜索轨迹。我们形式化地刻画了重编码何时仅仅是汉明等距的重参数化，并给出了一个构造性例子，证明精确的逐点一致性并不意味着优化等价性。我们研究了来自ProteinGym的实验测量的GFP和AAV适应度景观。内部QUBO代理模型与标签（注：原始摘要在此处被截断）...

    arXiv:2603.27526v2 Announce Type: replace  Abstract: Protein fitness optimization is a discrete search problem, and the representation used for prediction also determines the neighborhood graph traversed by an optimizer. We introduce Q-BioLat, a framework that maps pretrained protein-language-model embeddings to compact binary codes and fits a quadratic unconstrained binary optimization (QUBO) surrogate with unary and pairwise latent interactions. Our central contribution is an optimization-aware view of representation: binary encodings that are similar in pointwise predictive accuracy can induce different Hamming neighborhoods, local optima, and search trajectories. We formalize when a recoding is only a Hamming-isometric reparameterization and give a constructive example showing that exact pointwise agreement does not imply optimization equivalence.   We study experimentally measured GFP and AAV fitness landscapes from ProteinGym. The internal QUBO surrogate is evaluated against labe
    
[^278]: 曲率感知的期望自由能作为贝叶斯优化的采集函数

    Curvature-aware Expected Free Energy as an Acquisition Function for Bayesian Optimization

    [https://arxiv.org/abs/2603.26339](https://arxiv.org/abs/2603.26339)

    提出了一种曲率感知的期望自由能采集函数用于贝叶斯优化，该函数统一了上置信界、下置信界和期望信息增益，并具有凹函数的无偏收敛保证，在后悔值和均方误差两个指标上均取得有竞争力的表现。

    

    我们提出了一种基于期望自由能的贝叶斯优化采集函数，用于解决联合学习与优化问题，即在优化目标函数的同时学习该未知函数。我们证明，在特定假设下，期望自由能可以简化为上置信界、下置信界和期望信息增益。我们证明了期望自由能对凹函数具有无偏收敛保证。基于这些推导结果，我们引入了一种曲率感知的期望自由能更新律，并通过范德波尔振荡器的系统辨识问题展示了其概念验证。在一个具有振荡地形的二维基准测试中，我们的自适应期望自由能采集函数在后悔值和均方误差两个指标上均取得了有竞争力的表现，而典型的采集函数通常只能在其中一项指标上表现良好。

    arXiv:2603.26339v2 Announce Type: replace  Abstract: We propose an Expected Free Energy-based acquisition function for Bayesian optimization to solve the joint learning and optimization problem, i.e., optimize and learn the underlying function simultaneously. We show that, under specific assumptions, Expected Free Energy reduces to Upper Confidence Bound, Lower Confidence Bound, and Expected Information Gain. We prove that Expected Free Energy has unbiased convergence guarantees for concave functions. Using the results from these derivations, we introduce a curvature-aware update law for Expected Free Energy and show its proof of concept using a system identification problem on a Van der Pol oscillator. On a two-dimensional benchmark with an oscillatory landscape, our adaptive Expected Free Energy acquisition achieves competitive performance in both regret and mean squared error, unlike the typical acquisition functions that perform well in only one metric.
    
[^279]: 随机维度零阶估计器：PINNs的稳定且内存高效训练

    Stochastic Dimension Zeroth-Order Estimator: Stable and Memory-Efficient Training of PINNs

    [https://arxiv.org/abs/2603.24002](https://arxiv.org/abs/2603.24002)

    本文提出SDZE框架，通过统一随机空间估计和零阶优化，实现PINNs训练中空间和内存复杂度均与维度无关，解决了高维PDEs中的方差爆炸和内存瓶颈问题。

    

    arXiv:2603.24002v4 公告类型：替换。摘要：物理信息神经网络（PINNs）用于高维和高阶偏微分方程（PDEs）时，主要受限于$\mathcal{O}(d^k)$的空间导数复杂度和反向传播（BP）的$\mathcal{O}(P)$内存开销。尽管随机空间估计器成功将空间复杂度降至$\mathcal{O}(1)$，但其对一阶优化的依赖仍导致大规模训练中内存消耗过高。零阶（ZO）优化提供了一种无BP的替代方案；然而，将随机空间算子与ZO扰动简单结合会引发$\mathcal{O}(1/\varepsilon^2)$的方差爆炸，导致数值发散。为解决这些问题，我们提出了\textbf{S}tochastic \textbf{D}imension-free \textbf{Z}eroth-order \textbf{E}stimator（\textbf{SDZE}），一个统一框架，在空间和内存方面均实现了维度无关的复杂度。具体而言，

    arXiv:2603.24002v4 Announce Type: replace  Abstract: Physics-Informed Neural Networks (PINNs) for high-dimensional and high-order partial differential equations (PDEs) are primarily constrained by the $\mathcal{O}(d^k)$ spatial derivative complexity and the $\mathcal{O}(P)$ memory overhead of backpropagation (BP). While randomized spatial estimators successfully reduce the spatial complexity to $\mathcal{O}(1)$, their reliance on first-order optimization still leads to prohibitive memory consumption at scale. Zeroth-order (ZO) optimization offers a BP-free alternative; however, naively combining randomized spatial operators with ZO perturbations triggers a variance explosion of $\mathcal{O}(1/\varepsilon^2)$, leading to numerical divergence. To address these challenges, we propose the \textbf{S}tochastic \textbf{D}imension-free \textbf{Z}eroth-order \textbf{E}stimator (\textbf{SDZE}), a unified framework that achieves dimension-independent complexity in both space and memory. Specifica
    
[^280]: 面向分子科学的多任务大型推理模型

    A Multitask Large Reasoning Model for Molecular Science

    [https://arxiv.org/abs/2603.12808](https://arxiv.org/abs/2603.12808)

    本文提出一种任务自适应的多专家大型推理模型，通过思维链监督和分子信息引导的强化学习整合化学知识，在10项分子任务上超越20多个大语言模型，总体性能较基础模型提升50.3%，同时保持化学推理的可解释性。

    

    分子科学中的人工智能必须超越模式识别，迈向化学上有效且可解释的推理。我们提出了一种任务自适应的大型推理模型，该模型通过协同的多专家架构、思维链监督以及分子信息引导的强化学习来整合化学知识。任务条件路由机制协调预测专家与推理专家，覆盖涵盖分子描述与生成、命名法翻译、性质预测和反应预测在内的10项分子任务。该模型超越了20多个通用及分子领域大语言模型，相比基础模型将总体性能提升了50.3%，并在大多数任务上超越了领先的分子多任务基线模型。对专家表示和推理路径的分析显示，模型在实现任务特定适应的同时，保留了可解释的化学推理。案例研究进一步……

    arXiv:2603.12808v2 Announce Type: replace  Abstract: Artificial intelligence in molecular science must move beyond pattern recognition toward chemically valid and interpretable reasoning. We present a task-adaptive large reasoning model that integrates chemical knowledge through a synergistic multispecialist architecture, chain-of-thought supervision, and molecule-informed reinforcement learning. Task-conditioned routing coordinates prediction and inference specialists across 10 molecular tasks spanning molecular description and generation, nomenclature translation, property prediction, and reaction prediction. The model outperforms more than 20 general-purpose and molecular large language models, improves aggregate performance over the base model by 50.3%, and surpasses the leading molecular multitask baseline on most tasks. Analyses of specialist representations and reasoning pathways reveal task-specific adaptation while retaining interpretable chemical inference. A case study furth
    
[^281]: 部分观测下面向现场检测与维护移动操作的语言引导抓取

    Language-Guided Grasping under Partial Observation for Mobile Manipulation in Field Inspection and Maintenance

    [https://arxiv.org/abs/2603.07866](https://arxiv.org/abs/2603.07866)

    该论文提出了一种面向腿式移动操作机器人的语言引导抓取流水线，结合开放词汇检测、可提示分割、深度补偿与点云补全技术，在部分观测条件下实现适用于海上检测与维护场景的6自由度抓取。

    

    海上检测与维护工作中越来越多地使用腿式机器人执行日常感知任务，然而许多有价值的干预操作仍需要与工具、容器及任务相关物体进行物理交互。让机器人执行这些任务可以减少操作人员在封闭、高空或具有潜在爆炸风险区域中的暴露。本文提出了一种面向腿式移动操作机器人在部分观测条件下运行的语言引导抓取流水线。操作员定义目标后，系统通过开放词汇检测和可提示分割在RGB图像中定位目标，提取以物体为中心的RGB-D点云，通过深度补偿和点云补全改善稀疏的几何信息，并基于碰撞、间隙、可达性和接近约束选择6自由度抓取姿态。该系统在配备机械臂的四足机器人上实现，并在两个受小物体拾取任务启发的杂乱桌面场景中进行了评估。

    arXiv:2603.07866v4 Announce Type: replace-cross  Abstract: Offshore inspection and maintenance have increasingly been using legged robots for routine sensing, yet many useful interventions still require physical interaction with tools, containers, and task-relevant objects. Employing robots for these tasks can reduce operators' exposure in confined, elevated, or potentially explosive areas. This paper presents a language-guided grasping pipeline for a legged mobile manipulator operating under partial observation. An operator defines the target, the system grounds it in RGB with open-vocabulary detection and promptable segmentation, extracts an object-centric RGB-D point cloud, improves sparse geometry through depth compensation and point-cloud completion, and selects a 6-DoF grasp using collision, clearance, reachability, and approach constraints. The system is implemented on a quadruped robot with an arm and evaluated in two cluttered tabletop scenes motivated by small-object retrieva
    
[^282]: 利用领域感知傅里叶特征增强物理信息神经网络：迈向更优性能与可解释结果

    Enhancing Physics-Informed Neural Networks with Domain-aware Fourier Features: Towards Improved Performance and Interpretable Results

    [https://arxiv.org/abs/2603.02948](https://arxiv.org/abs/2603.02948)

    本文提出利用领域感知傅里叶特征对输入空间进行位置编码，从而消除显式边界条件损失项和损失平衡方案的需求，简化训练并降低计算成本，同时开发了基于LRP的可解释性框架以提取输入空间的相关性归因分数。

    

    物理信息神经网络通过将偏微分方程嵌入到损失函数中，从而将物理知识融入神经网络。尽管这类模型在学习底层物理规律方面取得了成功，但PINN模型仍然难以训练和解释。本工作提出了一种新颖的建模方法，该方法依赖于使用领域感知傅里叶特征对输入空间进行位置编码。这些特征封装了所有领域特定的特性，例如几何形状和边界条件，与随机傅里叶特征不同，它们消除了对显式边界条件损失项和损失平衡方案的需求，同时简化了优化过程并降低了与训练相关的计算成本。我们进一步开发了一个基于LRP的、专为PINNs定制的可解释性框架，能够提取输入空间的相关性归因分数。

    arXiv:2603.02948v2 Announce Type: replace-cross  Abstract: Physics-Informed Neural Networks (PINNs) incorporate physics into neural networks by embedding partial differential equations (PDEs) into their loss function. Despite their success in learning the underlying physics, PINN models remain difficult to train and interpret. In this work, a novel modeling approach is proposed, which relies on the use of Domain-aware Fourier Features (DaFFs) for the positional encoding of the input space. These features encapsulate all the domain-specific characteristics, such as the geometry and boundary conditions, and unlike Random Fourier Features (RFFs), eliminate the need for explicit boundary condition loss terms and loss balancing schemes, while simplifying the optimization process and reducing the computational cost associated with training. We further develop an LRP-based explainability framework tailored to PINNs, enabling the extraction of relevance attribution scores for the input space. 
    
[^283]: MINT：面向早期阿尔茨海默病筛查的多模态影像到语音知识迁移

    MINT: Multimodal Imaging-to-Speech Knowledge Transfer for Early Alzheimer's Screening

    [https://arxiv.org/abs/2602.23994](https://arxiv.org/abs/2602.23994)

    该论文提出MINT三阶段框架，通过将MRI衍生的生物标志物知识迁移到语音模型中，实现了无需影像设备、具备生物学依据的早期阿尔茨海默病无创语音筛查。

    

    阿尔茨海默病是一种进行性神经退行性疾病，轻度认知障碍（MCI）先于痴呆出现。结构MRI能够提供生物标志物，但需要昂贵的基础设施，限制了大规模人群部署。语音提供了一种无创的替代方案，然而纯语音分类器的开发独立于神经影像学，缺乏针对认知正常（CN）与轻度认知障碍（MCI）分类的生物学依据。我们提出了MINT（多模态影像到语音知识迁移），这是一个三阶段框架，在训练过程中将MRI衍生的生物标志物结构迁移到语音模型中。MRI教师网络为CN与MCI分类定义了一个紧凑的嵌入空间，同时残差投影头使用组合几何损失将语音表示对齐到该空间。冻结的MRI分类器实现了无需影像的推理。在ADNI-4数据集上，对齐后的语音模型取得了与语音基线相当的性能，而多模态融合进一步提升了整体性能。

    arXiv:2602.23994v2 Announce Type: replace-cross  Abstract: Alzheimer's disease is a progressive neurodegenerative disorder in which mild cognitive impairment (MCI) precedes dementia. Structural MRI provides biomarkers but requires costly infrastructure, limiting population-scale deployment. Speech offers a non-invasive alternative, yet speech-only classifiers are developed independently of neuroimaging and lack biological grounding for CN-versus-MCI classification. We propose MINT (Multimodal Imaging-to-Speech Knowledge Transfer), a three-stage framework that transfers MRI-derived biomarker structure to speech during training. An MRI teacher defines a compact embedding space for CN-versus-MCI classification, while a residual projection head aligns speech representations to this space using a combined geometric loss. The frozen MRI classifier enables imaging-free inference. On ADNI-4, aligned speech achieves performance comparable to speech baselines, while multimodal fusion improves ov
    
[^284]: 脑电通道交互的稀疏贝叶斯建模提升P300脑机接口性能

    Sparse Bayesian Modeling of EEG Channel Interactions Improves P300 Brain-Computer Interface Performance

    [https://arxiv.org/abs/2602.17772](https://arxiv.org/abs/2602.17772)

    该论文提出一种稀疏贝叶斯时变回归框架，通过松弛阈值高斯过程先验显式建模脑电通道间成对交互并进行时间特征选择，在55人P300拼写数据集上将中位字符级准确率提升至96.4%，同时保证了模型的可解释性。

    

    基于脑电图（EEG）的P300脑机接口（BCI）通过检测刺激诱发的神经响应，实现无需肢体动作的交流。由于高维性、时间依赖性以及脑电通道间复杂的交互作用，准确高效的解码仍然具有挑战性。现有方法通常独立处理各通道或依赖黑箱模型，限制了可解释性和个性化。我们提出了一种稀疏贝叶斯时变回归框架，在执行时间特征选择的同时显式建模脑电通道间的成对交互作用，其中松弛阈值高斯过程先验在通道特异性效应和交互效应中引入结构化稀疏性，从而实现对任务相关通道及通道对的可解释识别。在包含55名参与者的公开P300拼写器数据集上应用，我们的方法达到了96.4%的中位字符级准确率。

    arXiv:2602.17772v3 Announce Type: replace-cross  Abstract: Electroencephalography (EEG)-based P300 brain-computer interfaces (BCIs) enable communication without physical movement by detecting stimulus-evoked neural responses. Accurate and efficient decoding remains challenging due to high dimensionality, temporal dependence, and complex interactions across EEG channels. Existing approaches often treat channels independently or rely on black-box models, limiting interpretability and personalization. We propose a sparse Bayesian time-varying regression framework that explicitly models pairwise EEG channel interactions while performing temporal feature selection, where a relaxed-thresholded Gaussian process prior induces structured sparsity in both channel-specific and interaction effects, enabling interpretable identification of task-relevant channels and channel pairs. Applied to a public P300 speller dataset of 55 participants, our method achieves a 96.4% median character-level accurac
    
[^285]: 贝叶斯求积

    Bayesian Quadrature

    [https://arxiv.org/abs/2602.16218](https://arxiv.org/abs/2602.16218)

    本综述首次系统全面地梳理了贝叶斯求积方法，涵盖其数学基础、建模-推断-采样三维分类体系、理论保证、数值实验对比以及实际应用中的挑战与局限性。

    

    arXiv:2602.16218v2 公告类型：替换 摘要：贝叶斯求积是一种基于模型的概率化数值积分方法，用于估计难以直接计算的积分或期望。尽管贝叶斯求积早在20世纪80年代就已得到推广，但至今尚未有系统而全面的论述发表。本综述旨在填补这一空白。我们从不同的视角回顾了贝叶斯求积的数学基础；提出了一个系统的分类体系，沿着建模、推断和采样三个维度对不同的贝叶斯求积方法进行分类；汇集了一般性的理论保证；并提供了一项受控的数值研究，探索并阐明了分类体系各维度上不同选择所产生的影响。我们还在现实层面对贝叶斯求积方法在实际应用中面临的挑战与局限性进行了评估，并提供了一份最新且近乎详尽无遗的参考文献目录，不仅涵盖了机器学……（原文摘要在此处截断）

    arXiv:2602.16218v2 Announce Type: replace  Abstract: Bayesian quadrature is a probabilistic, model-based approach to numerical integration, the estimation of intractable integrals, or expectations. Although Bayesian quadrature was popularised already in the 1980s, no systematic and comprehensive treatment has been published. The purpose of this survey is to fill this gap. We review the mathematical foundations of Bayesian quadrature from different points of view; present a systematic taxonomy for classifying different Bayesian quadrature methods along the three axes of modelling, inference, and sampling; collect general theoretical guarantees; and provide a controlled numerical study that explores and illustrates the effect of different choices along the axes of the taxonomy. We also provide a realistic assessment of practical challenges and limitations to application of Bayesian quadrature methods and include an up-to-date and nearly exhaustive bibliography that covers not only machin
    
[^286]: 修补分布失配：面向稳定离策略SFT的强化学习改写智能体

    Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT

    [https://arxiv.org/abs/2602.11220](https://arxiv.org/abs/2602.11220)

    提出一种用强化学习训练的轻量级LoRA改写策略，在任务一致性约束下优化问答分布对齐与语义多样性，从而修补下游监督数据与模型生成分布之间的失配，缓解SFT中的灾难性遗忘。

    

    大语言模型通常通过监督微调（SFT）来适配下游任务，但下游监督数据与模型自身生成分布之间的显著分布失配可能加剧灾难性遗忘。数据改写提供了一种以数据为中心的方法，可在SFT之前缩小这种失配。然而，现有方法通常从提示诱导的条件分布中采样改写结果，这未必与骨干模型自然的问答生成分布对齐，且固定模板会降低输出多样性。我们将数据改写形式化为一个策略学习问题，并利用强化学习训练一个轻量级的LoRA改写策略。该策略在严格的任务一致性门控约束下，优化问答风格的分布对齐与语义多样性，为下游SFT生成经过验证的监督数据。在三个指令微调骨干模型上，所得模型

    arXiv:2602.11220v2 Announce Type: replace-cross  Abstract: Large language models are commonly adapted to downstream tasks through supervised fine-tuning (SFT), but substantial distribution mismatch between downstream supervision and a model's generation distribution can intensify catastrophic forgetting. Data rewriting offers a data-centric way to narrow this mismatch before SFT. Existing methods, however, typically sample rewrites from a prompt-induced conditional distribution, which need not align with the backbone's natural question-answering generation distribution, and fixed templates can reduce output diversity. We formulate data rewriting as a policy-learning problem and train a lightweight LoRA rewriting policy with reinforcement learning. The policy optimizes question-answering-style distributional alignment and semantic diversity under a hard task-consistency gate, producing verified supervision for downstream SFT. Across three instruction-tuned backbones, the resulting model
    
[^287]: TabICLv2：一个更好、更快、可扩展且开源的表格基础模型

    TabICLv2: A better, faster, scalable, and open tabular foundation model

    [https://arxiv.org/abs/2602.11139](https://arxiv.org/abs/2602.11139)

    TabICLv2 通过新型合成数据生成引擎、可扩展 softmax 注意力架构和 Muon 优化器预训练协议三大创新，成为一个无需任何调优即可超越 RealTabPFN-2.5 的更快、更可扩展的开源表格基础模型。

    

    诸如 TabPFNv2 和 TabICL 等表格基础模型最近在预测基准测试中超越了梯度提升树，展示了上下文学习在表格数据上的价值。我们提出了 TabICLv2，一个用于回归和分类的全新最先进基础模型，它建立在三大支柱之上：(1) 一个专为高预训练多样性而设计的新型合成数据生成引擎；(2) 多项架构创新，包括注意力机制中一种新的可扩展 softmax，无需高昂的长序列预训练即可提升对更大数据集的泛化能力；(3) 优化的预训练协议，特别是用 Muon 优化器取代了 AdamW。在 TabArena 和 TALENT 基准测试中，TabICLv2 无需任何调优即可超越当前最先进的 RealTabPFN-2.5（经过超参数调优、集成并在真实数据上微调）的性能。仅使用适度的预训练计算量，TabICLv2 就能实现有效的泛化……

    arXiv:2602.11139v2 Announce Type: replace  Abstract: Tabular foundation models, such as TabPFNv2 and TabICL, have recently dethroned gradient-boosted trees at the top of predictive benchmarks, demonstrating the value of in-context learning for tabular data. We introduce TabICLv2, a new state-of-the-art foundation model for regression and classification built on three pillars: (1) a novel synthetic data generation engine designed for high pretraining diversity; (2) various architectural innovations, including a new scalable softmax in attention improving generalization to larger datasets without prohibitive long-sequence pretraining; and (3) optimized pretraining protocols, notably replacing AdamW with the Muon optimizer. On the TabArena and TALENT benchmarks, TabICLv2 without any tuning surpasses the performance of the current state of the art, RealTabPFN-2.5 (hyperparameter-tuned, ensembled, and fine-tuned on real data). With only moderate pretraining compute, TabICLv2 generalizes eff
    
[^288]: 绕过推理依据：语言模型中隐式推理的因果审计

    Bypassing the Rationale: Causal Auditing of Implicit Reasoning in Language Models

    [https://arxiv.org/abs/2602.03994](https://arxiv.org/abs/2602.03994)

    该论文提出基于激活修补的因果审计指标——思维链中介指数（CMI），发现语言模型中思维链的真实因果影响往往局限于狭窄的“推理窗口”，且存在 CoT 文本看似合理但实际计算被绕过的情况，表明思维链输出并不能忠实反映模型内部的推理过程。

    

    思维链（Chain-of-Thought, CoT）提示被广泛用作推理辅助工具，并常被视为一种透明性机制。然而，CoT 带来的行为提升并不意味着模型的内部计算在因果上依赖于其生成的推理文本，也就是说，模型可能在生成流畅推理依据的同时，将决定性的计算通过潜在通路进行传递。我们提出了一种基于激活修补（activation patching）的、逐层进行的 CoT 忠实性因果审计方法。我们的关键指标——思维链中介指数（CoT Mediation Index, CMI）——通过将修补 CoT token 隐藏状态所导致的性能下降与匹配的对照修补进行比较，从而分离出 CoT 特定的因果影响。在多个模型家族（Phi、Qwen、DialoGPT）和不同规模上的实验中，我们发现 CoT 特定的影响通常在深度上局限于狭窄的“推理窗口”内，并且我们识别出了“绕过”状态，即尽管 CoT 文本看似合理，CMI 却接近于零。我们还进一步观察到，经过显式调优的模型……（原文摘要在此处截断）

    arXiv:2602.03994v3 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) prompting is widely used as a reasoning aid and is often treated as a transparency mechanism. Yet behavioral gains under CoT do not imply that the model's internal computation causally depends on the emitted reasoning text, i.e. models may produce fluent rationales while routing decision-critical computation through latent pathways. We introduce a causal, layerwise audit of CoT faithfulness based on activation patching. Our key metric, the CoT Mediation Index (CMI), isolates CoT-specific causal influence by comparing performance degradation from patching CoT-token hidden states against matched control patches. Across multiple model families (Phi, Qwen, DialoGPT) and scales, we find that CoT-specific influence is typically depth-localized into narrow ''reasoning windows,'' and we identify bypass regimes where CMI is near-zero despite plausible CoT text. We further observe that models tuned explicitly for r
    
[^289]: 贝叶斯实验设计中的边界偏差与观测无关性修正

    Correcting Boundary Bias and Observation Independence in Bayesian Experimental Design

    [https://arxiv.org/abs/2602.01898](https://arxiv.org/abs/2602.01898)

    论文针对基于方差采集准则的高斯过程主动学习的两大缺陷——后验方差与观测内容无关以及边界处方差膨胀导致的过度采样，提出了修正方案，通过重构驱动的设计密度与基于后验均值的免训练变形，使采样更集中于目标函数变化剧烈的区域。

    

    在许多实验场景中，主动学习可以通过依次选择测量位置来提高样本效率，这在实验成本高昂时尤为有价值。基于方差采集准则的高斯过程被广泛用于这一目的，但其存在两个局限性。首先，它们与观测内容无关：其后验方差仅取决于样本采集的位置，而不取决于测量到的内容，这削弱了其对所采集数据结构的敏感性。其次，它们会在边界附近放大方差，导致相比空间内部在空间边缘进行过度采样。这些局限性削弱了顺序采集本应带来的采样效率提升。我们针对这两个局限性提出了修正方案：我们推导了一种由重构驱动的设计密度，并利用后验均值构建了一种无需训练的变形，将更多的测量点放置在目标函数变化迅速的区域。

    arXiv:2602.01898v2 Announce Type: replace  Abstract: In many experimental settings, active learning can improve sample efficiency by sequentially selecting where to measure, which is particularly valuable when experiments are expensive. Gaussian processes with variance-based acquisition criteria are widely used for this purpose, but have two limitations. First, they are observation-independent: their posterior variance depends only on where samples are acquired, not on what is measured, impairing their sensitivity to the structure of the acquired data. Second, they inflate the variance near boundaries, leading to excessive sampling at the edges of the space compared to the interior. These limitations undermine the gains in sampling efficiency expected from sequential acquisition. We address both limitations. We derive a reconstruction-driven design density and use the posterior mean to build a training-free warp that places more measurements where the target function varies rapidly. A 
    
[^290]: 车间调度问题的变分方法

    Variational Approach for Job Shop Scheduling

    [https://arxiv.org/abs/2602.00408](https://arxiv.org/abs/2602.00408)

    本文首次将变分推断引入作业车间调度问题，提出VG2S框架，通过基于ELBO的变分图编码器将表示学习与策略优化数学解耦，从而解决传统深度强化学习方法的训练非平稳性和泛化能力受限问题。

    

    本文提出了一种新颖的变分图到调度器（VG2S）框架，用于解决作业车间调度问题（JSSP），这是制造业中直接影响运营效率和资源利用率的关键任务。传统的深度强化学习（DRL）方法通常面临训练过程中的非平稳性以及对未见问题实例泛化能力有限等挑战，原因在于它们同时优化表示学习和策略执行。为了解决这些问题，我们首次将变分推断引入JSSP领域，并基于证据下界（ELBO）与最大熵强化学习推导出一个概率目标。通过在数学上将表示学习与策略优化解耦，VG2S框架使智能体能够通过变分图编码器学习调度实例的鲁棒结构表示。

    arXiv:2602.00408v3 Announce Type: replace-cross  Abstract: This paper proposes a novel Variational Graph-to-Scheduler (VG2S) framework for solving the Job Shop Scheduling Problem (JSSP), a critical task in manufacturing that directly impacts operational efficiency and resource utilization. Conventional Deep Reinforcement Learning (DRL) approaches often face challenges such as non-stationarity during training and limited generalization to unseen problem instances because they optimize representation learning and policy execution simultaneously. To address these issues, we introduce variational inference to the JSSP domain for the first time and derive a probabilistic objective based on the Evidence of Lower Bound (ELBO) with maximum entropy reinforcement learning. By mathematically decoupling representation learning from policy optimization, the VG2S framework enables the agent to learn robust structural representations of scheduling instances through a variational graph encoder. This a
    
[^291]: 非平衡采样下MMD的有限样本无偏方差：精确估计与拟线性计算

    Finite-Sample Unbiased Variance of MMD under Unbalanced Sampling: Exact Estimation and Quasi-Linear Computation

    [https://arxiv.org/abs/2601.13874](https://arxiv.org/abs/2601.13874)

    该论文推导了非平衡采样下MMD方差的有限样本无偏估计量，并通过拉普拉斯核的递归前缀-后缀累加方案将计算复杂度从 $\mathcal{O}(N^2)$ 降至 $\mathcal{O}(N \log N)$、内存仅需 $\mathcal{O}(N)$。

    

    准确且高效地估计最大均值差异（MMD）的方差仍然具有挑战性，尤其是在样本量不平衡的情况下。在本文中，我们推导了MMD方差的有限样本无偏估计量。为了克服传统的 $\mathcal{O}(N^2)$ 计算瓶颈，我们为拉普拉斯核开发了一种递归前缀-后缀累加方案，将计算复杂度降低至 $\mathcal{O}(N \log N)$，同时仅需 $\mathcal{O}(N)$ 的内存。实验结果验证了所提估计量的理论精确性和数值稳定性，并展示了其在大规模数据集上的可扩展性。此外，该方法在时间序列生成对抗网络训练过程中监测分布收敛方面也表现出有效性。

    arXiv:2601.13874v3 Announce Type: replace-cross  Abstract: Accurately and efficiently estimating the variance of the Maximum Mean Discrepancy (MMD) remains challenging, particularly for unbalanced sample sizes. In this paper, we derive a finite-sample unbiased estimator of the MMD variance. To overcome the traditional $\mathcal{O}(N^2)$ computational bottleneck, we develop a recursive prefix-suffix accumulation scheme for the Laplace kernel, reducing the computational complexity to $\mathcal{O}(N \log N)$ while requiring $\mathcal{O}(N)$ memory. Experimental results verify the theoretical exactness and numerical stability of the proposed estimator and demonstrate its scalability on large datasets. Furthermore, the method proves effective for monitoring distributional convergence during the training of Time-series Generative Adversarial Networks (TimeGAN).
    
[^292]: 训练过程中语音模型性能与复杂度的权衡优化

    Performance and Complexity Trade-off Optimization of Speech Models During Training

    [https://arxiv.org/abs/2601.13704](https://arxiv.org/abs/2601.13704)

    该论文提出在训练过程中直接优化语音模型性能与计算复杂度之间的权衡，突破了传统随机梯度下降只能优化可微函数、无法直接优化模型结构复杂度的限制。

    

    在语音机器学习中，神经网络模型通常通过选择具有固定层大小和结构的架构来设计，随后对模型进行训练，以在与任务目标一致的指标上最大化性能。虽然整体架构通常由任务的先验知识指导，但各个层的大小往往是通过启发式方法选择的。然而，这种方法无法保证性能与计算复杂度之间的最优权衡；因此，通常需要采用事后方法（如权重量化或模型剪枝）来降低计算成本。之所以如此，是因为随机梯度下降（SGD）方法只能优化可微函数，而影响计算复杂度的因素（如层的大小和每秒浮点运算次数FLOP/s）是不可微的，需要在训练过程中修改模型结构。我们……

    arXiv:2601.13704v4 Announce Type: replace-cross  Abstract: In speech machine learning, neural network models are typically designed by choosing an architecture with fixed layer sizes and structure. These models are then trained to maximize performance on metrics aligned with the task's objective. While the overall architecture is usually guided by prior knowledge of the task, the sizes of individual layers are often chosen heuristically. However, this approach does not guarantee an optimal trade-off between performance and computational complexity; consequently, post hoc methods such as weight quantization or model pruning are typically employed to reduce computational cost. This occurs because stochastic gradient descent (SGD) methods can only optimize differentiable functions, while factors influencing computational complexity, such as layer sizes and floating-point operations per second (FLOP/s), are non-differentiable and require modifying the model structure during training. We pr
    
[^293]: 使用歌词对齐音频嵌入的可扩展音乐翻唱检索

    Scalable Music Cover Retrieval Using Lyrics-Aligned Audio Embeddings

    [https://arxiv.org/abs/2601.11262](https://arxiv.org/abs/2601.11262)

    该论文提出利用歌词作为翻唱歌曲间的强不变量，通过歌词对齐的音频嵌入实现高效可扩展的音乐翻唱检索，从而避免现有方法对复杂音频处理流程和高计算资源的需求。

    

    音乐翻唱检索，也称为版本识别，旨在识别同一底层音乐作品的不同演绎版本，这一任务对于曲目目录管理、版权执法和音乐检索至关重要。最先进的方法主要集中于和声与旋律特征，采用日益复杂的音频处理流程，这些流程被设计为对翻唱版本间常常差异巨大的音乐属性保持不变性。尽管有效，但这些方法需要大量的训练时间和计算资源。相比之下，歌词是翻唱版本间的一个强不变量，但其应用一直受到从复调音频中准确且高效提取歌词这一难题的限制。早期方法依赖于简单的框架，从而限制了下游性能，而较新的系统虽然能提供更强的结果，但需要将大型模型集成到复杂的多模态架构中。我们提出了LIVI（Lyr（摘要在此处截断）

    arXiv:2601.11262v2 Announce Type: replace-cross  Abstract: Music Cover Retrieval, also known as Version Identification, aims to recognize distinct renditions of the same underlying musical work, a task central to catalog management, copyright enforcement, and music retrieval. State-of-the-art approaches have largely focused on harmonic and melodic features, employing increasingly complex audio pipelines designed to be invariant to musical attributes that often vary widely across covers. While effective, these methods demand substantial training time and computational resources. By contrast, lyrics constitute a strong invariant across covers, though their use has been limited by the difficulty of extracting them accurately and efficiently from polyphonic audio. Early methods relied on simple frameworks that limited downstream performance, while more recent systems deliver stronger results but require large models integrated within complex multimodal architectures. We introduce LIVI (Lyr
    
[^294]: FuseFi：融合来自多样化通信数据包与频带的不规则采样CSI以实现Wi-Fi感知

    FuseFi: Combining Irregularly Sampled CSI from Diverse Communication Packets and Frequency Bands for Wi-Fi Sensing

    [https://arxiv.org/abs/2512.22143](https://arxiv.org/abs/2512.22143)

    FuseFi提出了一种Wi-Fi通信感知一体化框架，通过融合多频带、多类型通信数据包中不规则采样的CSI并利用时间感知注意力模型，无需注入探测数据包即可实现零感知通信开销的高效Wi-Fi感知。

    

    现有的Wi-Fi感知系统依赖注入高速探测数据包来提取信道状态信息（CSI），这会导致通信性能下降并限制部署的灵活性。尽管通信感知一体化（ISAC）是一个有前景的方向，但现有解决方案仍依赖辅助数据包注入，因为它们仅利用来自单一帧类型的均匀CSI，丢弃了约70%的自然可用数据包。我们提出了FuseFi，这是一种新颖的基于Wi-Fi的ISAC框架，它直接利用来自多个频带的多样化通信数据包中不规则采样的CSI，从而消除了侵入式数据包注入，且不引入任何感知专用的通信开销。FuseFi集成了一个CSI净化流水线，用于协调异构数据包并消除突发冗余，同时结合一个时间感知注意力模型，可直接从非均匀的CSI序列中学习……

    arXiv:2512.22143v2 Announce Type: replace-cross  Abstract: Existing Wi-Fi sensing systems rely on injecting high-rate probing packets to extract channel state information (CSI), leading to communication degradation and limited deployment flexibility. Although Integrated Sensing and Communication (ISAC) is a promising direction, existing solutions still rely on auxiliary packet injection because they exploit only uniform CSI from a single frame type, discarding approximately 70% of naturally available packets. We present FuseFi, a novel Wi-Fi-based ISAC framework that directly exploits irregularly sampled CSI from diverse communication packets across multiple frequency bands, eliminating intrusive packet injection and introducing no sensing-specific communication overhead. FuseFi integrates a CSI sanitization pipeline to harmonize heterogeneous packets and remove burst-induced redundancy, together with a time-aware attention model that learns directly from non-uniform CSI sequences with
    
[^295]: 隐式偏差与不变性：Hopfield网络如何高效学习轨道对称性

    Implicit Bias and Invariance: How Hopfield Networks Efficiently Learn Graph Orbits

    [https://arxiv.org/abs/2512.14338](https://arxiv.org/abs/2512.14338)

    本文证明Hopfield网络在有限置换轨道上训练时，仅需多项式数量的随机样本即可隐式学习对称不变性并记忆所有轨道元素。

    

    许多学习问题由群对称性组织。虽然不变性通常通过架构或群平均来施加，但我们询问它何时能从轨道上有限随机子集的训练中涌现。我们在经典Hopfield网络中研究这个问题，其中严格记忆可表示为线性间隔问题。将能量流最小化（MEF）重新参数化为指数损失，将梯度下降与相应的最小范数硬间隔记忆器联系起来。我们的主要结果表明，对于任何有限置换轨道的独立均匀样本，精确样本硬间隔支持向量机（HSVM）在指数意义上集中于不变的完整轨道HSVM。因此，一个与轨道大小无关的多项式样本数量足以实现近似参数不变性，并同时记忆每个轨道元素；方向收敛将此结论渐近传递。

    arXiv:2512.14338v4 Announce Type: replace  Abstract: Many learning problems are organized by group symmetries. While invariance is often imposed through architectures or group averaging, we ask when it can emerge from training on a finite random subset of an orbit. We study this question in classical Hopfield networks, where strict memorization can be expressed as a linear margin problem. Reparameterizing minimization of energy flow (MEF) as an exponential loss connects gradient descent to the corresponding minimum-norm hard-margin memorizer. Our main result shows that, for independent uniform samples from any finite permutation orbit, the exact sample hard-margin support vector machine (HSVM) concentrates exponentially around the invariant full-orbit HSVM. Consequently, an orbit-size-independent polynomial number of samples suffices both for approximate parameter invariance and for simultaneous memorization of every orbit element; directional convergence transfers this conclusion asym
    
[^296]: NeuroSketch：一种实用的神经解码设计配方

    NeuroSketch: A Practical Design Recipe for Neural Decoding

    [https://arxiv.org/abs/2512.09524](https://arxiv.org/abs/2512.09524)

    该研究提出了NeuroSketch，一种实用的神经解码设计配方，通过系统比较九种基础架构确定CNN-2D为最优骨干，并结合宏观层面的渐进式特征图扩展、早期下采样与微观层面的分组卷积优化，构建了轻量高效的神经解码模型。

    

    神经解码是脑机接口的基础，在医疗保健领域的应用日益增多。以往的研究主要集中于利用信号处理和深度学习方法来提升神经解码性能，然而，关于神经解码架构设计的系统性指导仍然有限。在本研究中，我们通过基础架构研究以及宏观和微观层面的优化，开发了NeuroSketch——一种实用的神经解码设计配方。通过比较九种基础架构，我们发现CNN-2D在神经解码任务中优于其他架构，并从时间和空间角度探讨了其有效性。在此骨干网络的基础上，我们在宏观层面结合了渐进式特征图扩展和早期下采样，在微观层面结合了分组卷积。这些设计选择构成了该配方，我们将其实例化为NeuroSketch-Base（140万参数）和NeuroSketch-Large。

    arXiv:2512.09524v2 Announce Type: replace-cross  Abstract: Neural decoding is fundamental to brain-computer interfaces, with growing applications in healthcare. Previous research has focused on leveraging signal processing and deep learning methods to enhance neural decoding performance. However, systematic guidance on architectural design for neural decoding remains limited. In this study, we develop NeuroSketch, a practical design recipe for neural decoding, through a basic architecture study followed by macro- and micro-level optimization. Comparing nine basic architectures, we find that CNN-2D outperforms other architectures in neural decoding tasks and explore its effectiveness from temporal and spatial perspectives. Building on this backbone, we combine gradual feature-map expansion and early downsampling at the macro level with grouped convolutions at the micro level. These choices form the recipe, which we instantiate as NeuroSketch-Base (1.4M parameters) and NeuroSketch-Large 
    
[^297]: 理解Transformer在学习潜在结构中的阶段性动态

    Understanding the Staged Dynamics of Transformers in Learning Latent Structure

    [https://arxiv.org/abs/2511.19328](https://arxiv.org/abs/2511.19328)

    该研究通过Alchemy基准的受控实验发现，transformer以离散阶段学习潜在结构的不同组成部分，并存在不对称性——模型能稳健地组合基本转换规则，却难以分解复杂示例来发现中间转换。

    

    语言建模已经向我们展示了transformer能够从上下文中发现潜在结构，但它们如何习得该结构不同组成部分的动态过程仍然知之甚少，这导致了一些观点认为模型只是在重新混合训练数据。在这项工作中，我们在受控环境下使用Alchemy基准来研究潜在结构学习。我们在三种任务变体上训练了一个小型decoder-only transformer：1）从部分上下文信息中推断缺失的转换；2）组合简单规则以解决多转换序列；3）分解复杂的多步骤示例以推断中间转换。通过将每个任务分解为可解释的组成部分，我们展示了模型以离散的阶段学习潜在结构的不同组成部分。我们还观察到一种不对称性：模型能够稳健地组合基本转换，但难以分解复杂示例来发现……

    arXiv:2511.19328v3 Announce Type: replace  Abstract: Language modeling has shown us that transformers can discover latent structure from context, but the dynamics of how they acquire different components of that structure remain poorly understood, leading to assertions that models just remix training data. In this work, we use the Alchemy benchmark in a controlled setting (Wang et al.,2021) to investigate latent structure learning. We train a small decoder-only transformer on three task variants: 1) inferring missing transitions from partial contextual information, 2) composing simple rules to solve multi-transition sequences, and 3) decomposing complex multi-step examples to infer intermediate transitions. By factorizing each task into interpretable components, we show that the model learns the different latent structure components in discrete stages. We also observe an asymmetry: the model composes fundamental transitions robustly, but struggles to decompose complex examples to disco
    
[^298]: Wasserstein–Fisher–Rao 梯度流的算子分裂分析

    An operator splitting analysis of Wasserstein--Fisher--Rao gradient flows

    [https://arxiv.org/abs/2511.18060](https://arxiv.org/abs/2511.18060)

    本文定量分析了求解 WFR 梯度流时 W-FR 算子分裂的顺序与步长的影响，并出人意料地证明：合理选择步长和算子顺序时，分裂方案可以比精确 WFR 流更快地收敛到目标分布。

    

    Wasserstein-Fisher-Rao（WFR）梯度流最近被提出作为一种强大的采样工具，它结合了纯 Wasserstein（W）梯度流和纯 Fisher-Rao（FR）梯度流两者的优点。现有的算法开发中隐式地使用了算子分裂技术来数值逼近 WFR 偏微分方程，即在给定步长内先求解 W 流，再求解 FR 流（或反之）。本工作研究了 W 算子与 FR 算子求解顺序的影响，并旨在提供定量分析。令人有些惊讶的是，我们证明，通过明智地选择步长和算子顺序，分裂方案（就模型时间而言）可以比精确的 WFR 流更快地收敛到目标分布。我们获得了描述两种分裂方案在一个时间步内演化的变分公式，并研究了在哪些情形下 W-FR 分裂方案更适用。

    arXiv:2511.18060v3 Announce Type: replace-cross  Abstract: Wasserstein-Fisher-Rao (WFR) gradient flows have been recently proposed as a powerful sampling tool that combines the advantages of pure Wasserstein (W) and pure Fisher-Rao (FR) gradient flows. Existing algorithmic developments implicitly make use of operator splitting techniques to numerically approximate the WFR partial differential equation, whereby the W flow is evaluated over a given step size and then the FR flow (or vice versa). This works investigates the impact of the order in which the W and FR operator are evaluated and aims to provide a quantitative analysis. Somewhat surprisingly, we show that with a judicious choice of step size and operator ordering, the split scheme can converge to the target distribution faster than the exact WFR flow (in terms of model time). We obtain variational formulae describing the evolution over one time step of both splitting schemes and investigate in which settings the W-FR split sho
    
[^299]: FairLRF：通过稀疏低秩分解实现公平性

    FairLRF: Achieving Fairness through Sparse Low Rank Factorization

    [https://arxiv.org/abs/2511.16549](https://arxiv.org/abs/2511.16549)

    本文提出FairLRF框架，创新性地将奇异值分解（SVD）从传统的模型压缩工具转变为公平性增强工具，通过稀疏低秩分解在不显著牺牲模型准确率和计算资源的情况下有效提升深度学习模型的公平性。

    

    随着深度学习（DL）技术在各类应用中变得不可或缺，在保持高性能的同时确保模型公平性变得日益重要，尤其是在医疗诊断等敏感领域。尽管已经提出了多种偏差缓解方法，但许多方法依赖于计算成本高昂的去偏策略，或者会导致模型准确率大幅下降，这限制了它们在现实世界资源受限环境中的实用性。为了解决这个问题，我们提出了一种面向公平性的低秩分解（LRF）框架，该框架利用奇异值分解（SVD）来提升深度学习模型的公平性。与主要用于通过分解和缩减权重矩阵来实现模型压缩的传统SVD不同，我们的工作表明SVD还可以作为增强公平性的有效工具。具体而言，我们观察到SVD得到的酉矩阵中的元素对模型贡献不均等。

    arXiv:2511.16549v2 Announce Type: replace  Abstract: As deep learning (DL) techniques become integral to various applications, ensuring model fairness while maintaining high performance has become increasingly critical, particularly in sensitive fields such as medical diagnosis. Although a variety of bias-mitigation methods have been proposed, many rely on computationally expensive debiasing strategies or suffer substantial drops in model accuracy, which limits their practicality in real-world, resource-constrained settings. To address this issue, we propose a fairness-oriented low rank factorization (LRF) framework that leverages singular value decomposition (SVD) to improve DL model fairness. Unlike traditional SVD, which is mainly used for model compression by decomposing and reducing weight matrices, our work shows that SVD can also serve as an effective tool for fairness enhancement. Specifically, we observed that elements in the unitary matrices obtained from SVD contribute unequ
    
[^300]: GeoCrossBench：面向遥感的跨波段泛化

    GeoCrossBench: Cross-Band Generalization for Remote Sensing

    [https://arxiv.org/abs/2511.02831](https://arxiv.org/abs/2511.02831)

    本文提出GeoCrossBench基准和χViT基线模型，通过新的跨波段泛化评估协议解决遥感领域新旧卫星波段不一致的问题，降低支持新卫星所需的模型重训练成本。

    

    遥感数据在不断获取中，新数据来自数量和种类日益增多的卫星，而绝大多数有标注的数据却来自较旧的卫星。随着面向地球观测的遥感基础模型规模不断扩大，为支持新卫星而（重新）训练的成本也随之增长，因此跨传感器和卫星的跨波段泛化能力变得愈发重要。我们提出了GeoCrossBench，这是对广受欢迎的GeoBench基准的扩展，引入了一套针对跨传感器和卫星跨波段泛化的新评估协议：它测试使用相同波段进行训练和测试的标准分布内性能、训练与测试波段无交集情况下的泛化能力，以及测试输入包含训练波段超集情况下的泛化能力。我们开发了χViT，这是波段无关的ChannelViT的自监督扩展版本，作为跨波段泛化的支持性基线模型。

    arXiv:2511.02831v2 Announce Type: replace  Abstract: The data for remote sensing is constantly acquired, and new data comes from a growing number and diversity of satellites, while the vast majority of labeled data comes from older satellites. As remote-sensing foundation models for Earth observation scale up, the cost of (re-)training to support new satellites grows too, so cross-band generalization across sensors and satellites is increasingly important. We introduce GeoCrossBench, an extension of the popular GeoBench benchmark with a new evaluation protocol for cross-band generalization across sensors and satellites: it tests standard in-distribution performance with the same bands for train and test, generalization to inputs with no intersection between train and test; and generalization to test inputs containing a superset of the training bands. We develop $\chi$ViT, a self-supervised extension of the band-agnostic ChannelViT, as a supporting baseline for cross-band generalization
    
[^301]: PitchFlower：一种具有音高可控性的基于流的神经音频编解码器

    PitchFlower: A flow-based neural audio codec with pitch controllability

    [https://arxiv.org/abs/2510.25566](https://arxiv.org/abs/2510.25566)

    PitchFlower是一种基于流的神经音频编解码器，通过在训练时对输入F0轮廓进行展平和随机偏移的简单扰动策略实现音高解耦，达到了DSP级别的精确音高控制，同时保持接近最先进神经方法的高音频质量。

    

    我们提出了PitchFlower，一种具有显式音高可控性的基于流的神经音频编解码器。我们的方法通过一种简单的扰动来促进音高解耦：在训练过程中，输入的基频（F0）轮廓被展平并随机偏移，同时将真实的F0作为条件输入以重建原始音频。向量量化瓶颈防止了音高信息的恢复，而基于流的解码器则生成高质量音频。实验表明，PitchFlower实现了与DSP基线方法同级别的精确音高控制，但音频质量要高得多，并且与最先进的神经方法表现相当。值得注意的是，尽管使用WORLD变换后的音频进行训练，我们的方法能够滤除声码器固有的伪影，揭示了深度生成建模对输入退化的强大鲁棒性。这一发现表明，我们的框架提供了一条简单且可扩展的路径，可以扩展到其他语音处理任务中。

    arXiv:2510.25566v2 Announce Type: replace-cross  Abstract: We present PitchFlower, a flow-based neural audio codec with explicit pitch controllability. Our approach promotes pitch disentanglement through a simple perturbation: during training, F0 contours are flattened and randomly shifted at the input, while the true F0 is provided as conditioning to regenerate the original audio. A vector-quantization bottleneck prevents pitch recovery, and a flow-based decoder generates high quality audio. Experiments show that PitchFlower achieves accurate pitch control at the level of DSP baselines but at much higher audio quality, and performs on par with state-of-the-art neural approaches. Notably, despite using WORLD-transformed audio for training, our method filters out the vocoder's inherent artifacts, revealing a strong resilience of deep generative modeling to input degradation. This finding suggests that our framework provides a simple and extensible path that could be extended to other sp
    
[^302]: 通过注意力桥实现数据高效的任意Transformer到Mamba蒸馏

    Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge

    [https://arxiv.org/abs/2510.19266](https://arxiv.org/abs/2510.19266)

    本文提出CAB蒸馏框架，利用轻量级注意力桥将Transformer教师模型的注意力相关表示以token级中间监督的方式迁移给Mamba等状态空间学生模型，实现了数据高效的跨架构知识蒸馏。

    

    状态空间模型（SSMs）已成为序列建模中Transformer的有前景的替代方案。然而，从头训练具有竞争力的SSM仍然计算开销巨大，且其生态系统远不如Transformer成熟。此外，SSM与Transformer之间的架构差异使得从预训练的Transformer高效迁移知识变得颇具挑战性。在本工作中，我们提出了基于注意力桥的跨架构蒸馏框架（CAB），该框架将Transformer教师模型中与注意力相关的表示迁移给状态空间学生模型。与仅监督最终预测的传统知识蒸馏不同，CAB通过一个轻量级的桥接模块和灵活的逐层对齐，实现了token级别的中间监督。通过将Transformer中与注意力相关的表示与Mamba中依赖于token的状态投影进行对齐，CAB促进了跨架构的有效知识迁移。

    arXiv:2510.19266v3 Announce Type: replace  Abstract: State-space models (SSMs) have emerged as promising alternatives to Transformers for sequence modeling. However, training competitive SSMs from scratch remains computationally intensive, and the ecosystem around them is far less mature than that of Transformers. Moreover, the architectural differences between SSMs and Transformers make it challenging to efficiently transfer knowledge from pretrained Transformers. In this work, we propose Cross-architecture distillation via Attention Bridge (CAB), a distillation framework that transfers attention-related representations from Transformer teachers to state-space student models. Unlike conventional knowledge distillation that supervises only final predictions, CAB enables token-level intermediate supervision through a lightweight bridge and flexible layer-wise alignment. By aligning Transformer attention-related representations with Mamba's token-dependent state projections, CAB facilita
    
[^303]: ADAPT：无需图表示的轻量级、长程机器学习力场

    ADAPT: Lightweight, Long-Range Machine Learning Force Fields Without Graphs

    [https://arxiv.org/abs/2509.24115](https://arxiv.org/abs/2509.24115)

    本文提出了ADAPT，一种抛弃图表示、将原子作为token并显式建模所有空间成对原子相互作用的轻量级Transformer机器学习力场，有效解决了图神经网络力场在点缺陷建模中的过度平滑和长程相互作用表征不佳问题。

    

    点缺陷在决定材料性质方面起着核心作用。第一性原理方法被广泛用于计算缺陷的能量学和结构，包括面向高通量缺陷数据库的大规模计算。然而，这些方法计算成本高昂，因此机器学习力场（MLFFs）成为加速结构弛豫的极具吸引力的替代方案。现有的大多数MLFF基于图神经网络（GNNs），而GNNs可能存在过度平滑、过度压缩以及长程相互作用表征不佳的问题。在对点缺陷建模时，这些问题尤其令人担忧。为了应对这些挑战，我们提出了加速深度原子势Transformer（ADAPT），这是一种MLFF，它以直接的空间坐标表示取代图表示，并显式地考虑所有成对原子相互作用。原子被视为token，并采用Transformer编码器模型……

    arXiv:2509.24115v2 Announce Type: replace  Abstract: Point defects play a central role in driving the properties of materials. First-principles methods are widely used to compute defect energetics and structures, including at scale for high-throughput defect databases. However, these methods are computationally expensive, making machine-learning force fields (MLFFs) an attractive alternative for accelerating structural relaxations. Most existing MLFFs are based on graph neural networks (GNNs), which can suffer from oversmoothing, oversquashing, and poor representation of long-range interactions. Both of these issues are especially of concern when modeling point defects. To address these challenges, we introduce the \textit{Accelerated Deep Atomic Potential Transformer} (ADAPT), an MLFF that replaces graph representations with a direct coordinates-in-space formulation and explicitly considers all pairwise atomic interactions. Atoms are treated as tokens, with a Transformer encoder model
    
[^304]: 一种利用潜在扩散模型求解逆问题的梯度流方法

    A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models

    [https://arxiv.org/abs/2509.19276](https://arxiv.org/abs/2509.19276)

    提出了一种免训练的扩散正则化Wasserstein梯度流方法（DWGF），利用预训练潜在扩散模型作为先验来求解不适定逆问题。

    

    求解不适定逆问题需要强大且灵活的先验。我们提出利用预训练的潜在扩散模型来完成这一任务，采用一种新的免训练方法，称为扩散正则化Wasserstein梯度流。具体而言，我们将后验采样问题表述为潜在空间中期望负对数后验目标的Wasserstein梯度流，并通过与扩散先验之间的Kullback-Leibler散度进行正则化。我们以StableDiffusion (Rombach et al., 2022) 作为先验，在标准基准上展示了我们方法的性能。

    arXiv:2509.19276v2 Announce Type: replace-cross  Abstract: Solving ill-posed inverse problems requires powerful and flexible priors. We propose leveraging pretrained latent diffusion models for this task through a new training-free approach, termed Diffusion-regularized Wasserstein Gradient Flow (DWGF). Specifically, we formulate the posterior sampling problem as a Wasserstein gradient flow in the latent space of an expected negative log posterior objective, regularized by a Kullback-Leibler divergence to the diffusion prior. We demonstrate the performance of our method on standard benchmarks using StableDiffusion (Rombach et al., 2022) as the prior.
    
[^305]: 通过触摸学习接触动力学：面向机器人插孔任务的动作条件图神经网络

    Learning Contact Dynamics through Touching: Action-conditional Graph Neural Networks for Robotic Peg Insertion

    [https://arxiv.org/abs/2509.12151](https://arxiv.org/abs/2509.12151)

    该论文提出了一种动作条件图神经网络模型，通过机器人随机触摸环境的自监督学习来预测接触密集操作中的运动和力-力矩，在仿真中于未见几何形状的插孔任务中达到98%成功率，在现实世界中比系统辨识的MuJoCo模型高出45%。

    

    我们提出了一种可学习的基于物理的模型，用于预测接触密集操作中机器人末端执行器的运动和反作用力-力矩。该模型将末端执行器和环境表示为图结构中相互作用的网格，并将预测明确地以施加的控制输入为条件。它直接预测物体层面的位姿更新，而反作用力矩则由逐顶点的力场产生。训练采用自监督方式，仅使用关节编码器和力-力矩数据，此时机器人在没有任务上下文的情况下随机触摸环境。在仿真中，我们的模型能够迁移到具有未见过的凹形几何形状的插孔任务，使用该模型的MPC智能体达到了高达98%的成功率，并且在自收集数据上微调后，在最紧的1毫米间隙下与使用真实动力学规划的智能体表现相当。在现实世界中，它在位置性能上比系统辨识的MuJoCo模型高出45%。

    arXiv:2509.12151v3 Announce Type: replace-cross  Abstract: We present a learnable physics-based model that predicts motion of the robot end effector and reaction force-torque in contact-rich manipulation. The model represents the end effector and the environment as interacting meshes in a graph structure, and conditions its prediction explicitly on the applied control input. It predicts object-level pose update directly, while the reaction torque emerges from a per-vertex force field. Training is self-supervised using only joint encoder and force-torque data while the robot is randomly touching the environment without task context. In simulation, our model transfers to peg insertion with unseen concave geometry, where an MPC agent using it reaches up to 98% success rate, and after fine-tuning on self-collected data matches an agent planning with the ground truth dynamics at the tightest 1 mm clearance. In the real world, it outperforms the system-identified MuJoCo model by 45% in posit
    
[^306]: 从大规模材料数据库中学习磁序分类

    Learning Magnetic Order Classification from Large-Scale Materials Databases

    [https://arxiv.org/abs/2509.05909](https://arxiv.org/abs/2509.05909)

    该研究开发了基于简单成分、结构和电子描述符的机器学习分类器，能以超过92%的准确率对磁性材料的传播矢量磁序进行分类，并揭示了Materials Project数据库中存在的系统性铁磁偏差。

    

    在高通量材料数据库中，磁性基态的可靠识别仍然是一项重大挑战，因为密度泛函理论（DFT）工作流程往往会收敛到铁磁（FM）解。在此，我们通过开发机器学习分类器来部分解决这一挑战，这些分类器在经过实验验证的MAGNDATA磁性材料上进行训练，并利用了来自Materials Project数据库的有限数量的简单成分、结构和电子描述符。我们的传播矢量分类器实现了超过92%的准确率，在可靠区分零传播矢量与非零传播矢量结构方面，优于近期一项基于不同构建数据集的等变神经网络研究，并揭示了Materials Project数据库中6840多个候选材料所存在的系统性铁磁偏差。与此同时，直接在Materials Project数据上训练的LightGBM和XGBoost模型...

    arXiv:2509.05909v3 Announce Type: replace-cross  Abstract: The reliable identification of magnetic ground states remains a major challenge in high-throughput materials databases, where density functional theory (DFT) workflows often converge to ferromagnetic (FM) solutions. Here, we partially address this challenge by developing machine-learning classifiers trained on experimentally validated MAGNDATA magnetic materials, leveraging a limited number of simple compositional, structural, and electronic descriptors sourced from the Materials Project Database. Our propagation-vector classifiers achieve accuracies above 92%, outperforming a recent equivariant-neural-network study on a differently constructed dataset in reliably distinguishing between zero and nonzero propagation-vector structures, and exposing a systematic ferromagnetic bias inherent to the Materials Project database for more than 6840 candidate materials. In parallel, LightGBM and XGBoost models trained directly on the Mate
    
[^307]: 使用变分量子方法求解稀疏图上的锥规划：以交流最优潮流为例

    Solving Conic Programs over Sparse Graphs using a Variational Quantum Approach: The Case of the AC Optimal Power Flow

    [https://arxiv.org/abs/2509.00341](https://arxiv.org/abs/2509.00341)

    提出了一种变分量子方法，通过两个参数化量子电路分别编码原始变量与对偶变量，将锥规划（含二次约束二次规划和半定规划）的求解转化为量子可观测量期望值的拉格朗日优化问题，并以交流最优潮流为应用案例。

    

    在物理学、量子信息、机器学习和工程领域出现的锥规划通常定义在稀疏图上。尽管此类问题可以使用经典内点求解器在多项式时间内求解，但其计算复杂度随图规模的增大而急剧增长。我们提出了一种变分量子范式来求解锥规划，包括二次约束二次规划和半定规划。我们通过参数化量子电路（PQC）的状态对原始变量进行编码，并通过与第二个PQC相关联的概率质量函数对对偶变量进行编码。由此，拉格朗日函数可以表示为量子可观测量经比例缩放后的期望值。我们通过在第一个/第二个PQC的参数上最小化/最大化拉格朗日函数来寻求其近似驻点。这一过程以混合方式完成：梯度使用两个PQC进行估计，同时……

    arXiv:2509.00341v3 Announce Type: replace-cross  Abstract: Conic programs arising in physics, quantum information, machine learning, and engineering are often defined over sparse graphs. Although such problems can be solved in polynomial time using classical interior-point solvers, the computational complexity scales unfavorably with graph size. We propose a variational quantum paradigm for solving conic programs, including quadratically constrained quadratic programs and semidefinite programs. We encode primal variables via the state of a parameterized quantum circuit (PQC) and dual variables via the probability mass function associated with a second PQC. The Lagrangian function can thus be expressed as scaled expectations of quantum observables. We pursue approximately stationary points of the Lagrangian by minimizing/maximizing the Lagrangian over the parameters of the first/second PQC. This is accomplished in a hybrid fashion: gradients are estimated using the two PQCs, while their
    
[^308]: 视觉感知引擎：面向机器人视觉任务的快速灵活多头推理框架

    Visual Perception Engine: Fast and Flexible Multi-Head Inference for Robotic Vision Tasks

    [https://arxiv.org/abs/2508.11584](https://arxiv.org/abs/2508.11584)

    提出了视觉感知引擎（VPEngine），一个通过共享基础模型骨干网络和并行任务专用模型头实现高效GPU多任务视觉推理的模块化框架，可消除计算冗余并支持动态任务优先级调整，适用于资源受限的机器人平台。

    

    在资源受限的机器人平台上部署多个机器学习模型以执行不同的感知任务，往往会导致冗余计算、庞大的内存占用以及复杂的集成挑战。针对这些问题，本工作提出了视觉感知引擎（VPEngine），这是一个模块化框架，旨在实现高效的GPU视觉多任务处理，同时保持可扩展性和开发者的易用性。我们的框架架构利用共享的基础模型骨干网络来提取图像表示，这些表示可以在多个并行运行的专用任务特定模型头之间高效共享，而无需任何不必要的GPU-CPU内存传输。这种设计消除了传统顺序模型部署中特征提取组件固有的计算冗余，同时能够根据应用需求动态调整任务优先级。我们展示了该框架的能力……

    arXiv:2508.11584v3 Announce Type: replace-cross  Abstract: Deploying multiple machine learning models on resource-constrained robotic platforms for different perception tasks often results in redundant computations, large memory footprints, and complex integration challenges. In response, this work presents Visual Perception Engine (VPEngine), a modular framework designed to enable efficient GPU usage for visual multitasking while maintaining extensibility and developer accessibility. Our framework architecture leverages a shared foundation model backbone that extracts image representations, which are efficiently shared, without any unnecessary GPU-CPU memory transfers, across multiple specialized task-specific model heads running in parallel. This design eliminates the computational redundancy inherent in feature extraction component when deploying traditional sequential models while enabling dynamic task prioritization based on application demands. We demonstrate our framework's capa
    
[^309]: 面向黑盒对抗攻击的基于共识的优化及其与进化策略的联系

    Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies

    [https://arxiv.org/abs/2506.24048](https://arxiv.org/abs/2506.24048)

    本文建立了基于共识的优化（CBO）中的共识跳跃与自然进化策略（NES）之间的理论联系，并通过实验证明在黑盒对抗攻击中CBO在某些场景下可以超越NES和其他进化策略。

    

    基于共识的优化（CBO）已成为一种高效的无梯度优化方案，具有吸引人的数学性质，例如对非凸损失函数的均场收敛结果。在这项工作中，我们在黑盒对抗攻击的背景下研究CBO，黑盒对抗攻击是指旨在欺骗分类器的不可感知输入扰动，且无需访问分类器的梯度。我们的贡献是建立了由Riedl等人提出的所谓“共识跳跃”（consensus hopping）与常用于对抗攻击场景中的自然进化策略（NES）之间的联系，并严格地将这两种方法与基于梯度的优化方案关联起来。除此之外，我们提供了全面的实验研究，表明尽管存在概念上的相似性，CBO在某些场景下可以超越NES和其他进化策略。

    arXiv:2506.24048v2 Announce Type: replace-cross  Abstract: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.
    
[^310]: 用于神经网络学习参数到解映射的DPG损失函数

    DPG loss functions for learning parameter-to-solution maps by neural networks

    [https://arxiv.org/abs/2506.18773](https://arxiv.org/abs/2506.18773)

    本文提出基于超弱间断间断Petrov-Galerkin（DPG）离散化的变分正确残差损失函数，为神经网络学习参数依赖偏微分方程的参数到解映射提供严格的精度认证，且该方法可推广至所有具有稳定DPG公式的问题。

    

    arXiv:2506.18773v2 公告类型：replace-cross 摘要：我们在参数依赖偏微分方程（PDE）族的背景下，针对参数到解映射的机器学习，开发、分析并通过实验探索了基于残差的损失函数。我们的主要关注点是通过严格的精度认证来增强所得深度神经网络降阶模型的预测能力。这通过使用变分正确的损失函数来实现。通过椭圆偏微分方程的一个具体例子，我们详细阐述了如何从超弱间断Petrov-Galerkin（DPG）离散化建立损失函数的变分正确性。尽管重点放在该例子上，但所提出的概念适用于更广泛的问题范围，即所有具有稳定DPG公式的问题。文中还讨论了高对比度扩散场及随之而来的椭圆性退化所带来的困难。数值结果和……

    arXiv:2506.18773v2 Announce Type: replace-cross  Abstract: We develop, analyze, and experimentally explore residual-based loss functions for machine learning of parameter-to-solution maps in the context of parameter-dependent families of partial differential equations (PDEs). Our primary concern is on rigorous accuracy certification to enhance the prediction capability of the resulting deep neural network reduced models. This is achieved by the use of variationally correct loss functions. Through one specific example of an elliptic PDE, details for establishing the variational correctness of a loss function from an ultraweak Discontinuous Petrov Galerkin (DPG) discretization are worked out. Despite the focus on the example, the proposed concepts apply to a much wider scope of problems, namely problems for which stable DPG formulations are available. The issue of high-contrast diffusion fields and ensuing difficulties with degrading ellipticity are discussed. Both numerical results and 
    
[^311]: 面向磁共振波谱贝叶斯推断的物理信息Sylvester归一化流

    Physics-Informed Sylvester Normalizing Flows for Bayesian Inference in Magnetic Resonance Spectroscopy

    [https://arxiv.org/abs/2505.03590](https://arxiv.org/abs/2505.03590)

    该论文提出了一种基于Sylvester归一化流的贝叶斯推断框架，结合融入物理先验知识的解码器，用于磁共振波谱中代谢物浓度的可靠定量化。

    

    磁共振波谱（MRS）是一种测量组织代谢成分的无创技术，可为神经系统疾病、肿瘤检测及其他代谢功能障碍提供宝贵见解。然而，准确的代谢物定量化受到谱重叠、低信噪比及各种伪影等挑战的阻碍。传统方法如线性组合建模容易产生歧义，且通常仅能以Cramér-Rao界的形式提供估计精度的理论下界。本工作引入了一个使用Sylvester归一化流（SNFs）的贝叶斯推断框架，以近似代谢物浓度的后验分布，从而提高定量的可靠性。基于物理的解码器融入了MRS信号形成的先验知识，确保了符合实际的分布表示。我们在模拟的7T质子数据上对该方法进行了验证。

    arXiv:2505.03590v2 Announce Type: replace-cross  Abstract: Magnetic resonance spectroscopy (MRS) is a non-invasive technique to measure the metabolic composition of tissues, offering valuable insights into neurological disorders, tumor detection, and other metabolic dysfunctions. However, accurate metabolite quantification is hindered by challenges such as spectral overlap, low signal-to-noise ratio, and various artifacts. Traditional methods like linear-combination modeling are susceptible to ambiguities and commonly only provide a theoretical lower bound on estimation accuracy in the form of the Cram\'er-Rao bound. This work introduces a Bayesian inference framework using Sylvester normalizing flows (SNFs) to approximate posterior distributions over metabolite concentrations, enhancing quantification reliability. A physics-based decoder incorporates prior knowledge of MRS signal formation, ensuring realistic distribution representations. We validate the method on simulated 7T proton 
    
[^312]: BOOM：机器学习模型分子性质分布外预测的基准测试

    BOOM: Benchmarking Out-Of-distribution Molecular Property Predictions of Machine Learning Models

    [https://arxiv.org/abs/2505.01912](https://arxiv.org/abs/2505.01912)

    本文提出了BOOM，一个基于化学信息学的分子性质分布外预测系统性基准，通过对超过150种模型-任务组合的评估，揭示了没有任何现有模型（包括化学基础模型）能在所有任务上实现强大的分布外泛化，最佳模型的分布外误差仍比分布内高出3倍。

    

    数据驱动的分子发现利用人工智能/机器学习（AI/ML）和生成式建模来筛选和设计新型分子。发现新分子需要准确的分布外（OOD）预测，但机器学习模型难以实现分布外泛化。目前，针对分子分布外预测任务尚不存在系统性的基准。我们提出了BOOM（分子性质分布外预测基准）：一个基于化学信息学的基准，用于评估常见分子性质预测任务上的分布外性能。我们评估了超过150种模型-任务组合，对深度学习模型的分布外性能进行了基准测试。总体而言，我们发现没有任何现有模型能够在所有任务上实现强大的泛化能力：即使是表现最好的模型，其平均分布外误差也比分布内误差高出3倍。当前的化学基础模型也未显示出强大的分布外外推能力。

    arXiv:2505.01912v3 Announce Type: replace-cross  Abstract: Data-driven molecular discovery leverages artificial intelligence/machine learning (AI/ML) and generative modeling to filter and design novel molecules. Discovering novel molecules requires accurate out-of-distribution (OOD) predictions, but ML models struggle to generalize OOD. Currently, no systematic benchmarks exist for molecular OOD prediction tasks. We present $\mathbf{BOOM}$, $\mathbf{b}$enchmarks for $\mathbf{o}$ut-$\mathbf{o}$f-distribution $\mathbf{m}$olecular property predictions: a chemically-informed benchmark for OOD performance on common molecular property prediction tasks. We evaluate over 150 model-task combinations to benchmark deep learning models on OOD performance. Overall, we find that no existing model achieves strong generalization across all tasks: even the top-performing model exhibited an average OOD error 3x higher than in-distribution. Current chemical foundation models do not show strong OOD extrap
    
[^313]: 可解释的图论机器学习及其在阿尔茨海默病预测中的应用

    Explainable Graph-theoretical Machine Learning with Application to Alzheimer's Disease Prediction

    [https://arxiv.org/abs/2503.16286](https://arxiv.org/abs/2503.16286)

    本文提出了一种可解释的图论机器学习框架XGML，通过构建个体大脑代谢图并识别最具预测性的子图，实现了基于FDG-PET数据对阿尔茨海默病多变量结果的个体化预测。

    

    痴呆症影响着全球超过5500万人，预计到2050年将达到1.39亿，其中阿尔茨海默病（AD）占病例的60-70%。阿尔茨海默病与大脑代谢连接的中断有关，早期发现这些中断对于AD的管理至关重要，而FDG-PET是识别此类损伤的有效工具。然而，大多数研究依赖于群体水平分析或阈值处理，这可能掩盖个体差异，并忽视较弱但在生物学上至关重要的大脑连接。此外，AD预测主要关注单变量而非多变量结果。为解决这一问题，我们提出了可解释图论机器学习（XGML），这是一个用于构建个体大脑代谢图并识别对多变量疾病相关结果最具预测性子图的框架。基于阿尔茨海默病神经影像学计划（ADNI）的FDG-PET数据，我们比较了六种图表示方法……

    arXiv:2503.16286v2 Announce Type: replace  Abstract: Dementia affects over 55 million people worldwide, projected to reach 139 million by 2050, with Alzheimer's disease (AD) accounting for 60-70% of cases. AD is associated with disruptions in metabolic brain connectivity. Detecting these disruptions early is crucial for AD management. FDG-PET is a useful tool for identifying such impairments. However, most studies rely on group-level analyses or thresholding, potentially masking individual differences and overlooking weaker yet biologically critical brain connections. Moreover, AD prediction largely focuses on univariate rather than multivariate outcomes. To address this, we introduce explainable graph-theoretical machine learning (XGML), a framework for constructing individual metabolic brain graphs and identifying subgraphs most predictive of multivariate disease-related outcomes. Using Alzheimer's Disease Neuroimaging Initiative (ADNI) FDG-PET data, we compared six graph representat
    
[^314]: 基于生物学信息异构图表示的可解释视网膜疾病预测

    Interpretable Retinal Disease Prediction Using Biology-Informed Heterogeneous Graph Representations

    [https://arxiv.org/abs/2502.16697](https://arxiv.org/abs/2502.16697)

    该论文提出了一种新颖的生物学信息异构图表示方法，以人类可解释的方式建模视网膜血管段、毛细血管间区域和中央凹无血管区，在保留OCTA图像丰富信息的同时实现了糖尿病视网膜病变分期的可解释预测。

    

    可解释性对于将机器学习模型用作医疗诊断的临床决策支持工具至关重要。然而，大多数基于神经网络的先进图像分类器都是不可解释的。因此，临床医生常常依赖已知的生物标志物来指导诊断，但与原始医学图像相比，基于生物标志物的分类往往会遭受严重的信息损失。这项工作提出了一种方法，在保留丰富成像信息的同时，增强了基于光学相干断层扫描血管成像（OCTA）图像进行糖尿病视网膜病变分期预测的可解释性。我们方法的核心贡献是一种新颖的生物学信息异构图表示，它以人类可解释的方式对视网膜血管段、毛细血管间区域和中央凹无血管区（FAZ）进行建模。这种图表示使我们能够将糖尿病视网膜病变分期问题转化为……

    arXiv:2502.16697v3 Announce Type: replace-cross  Abstract: Interpretability is crucial for utilizing machine learning models as clinical decision support tools for medical diagnostics. However, most state-of-the-art image classifiers based on neural networks are not interpretable. As a result, clinicians often resort to known biomarkers to guide diagnosis, although biomarker-based classification often suffers from drastic information loss compared to raw medical images. This work proposes a method that preserves the rich imaging information while simultaneously enhancing the interpretability of predictions for diabetic retinopathy staging from optical coherence tomography angiography (OCTA) images. The core contribution of our method is a novel biology-informed heterogeneous graph representation that models retinal vessel segments, intercapillary areas, and the foveal avascular zone (FAZ) in a human-interpretable way. This graph representation allows us to frame diabetic retinopathy st
    
[^315]: 桥接脑电信号与生成式人工智能的综述：从图像、文本到更广阔的领域

    A Survey on Bridging EEG Signals and Generative AI: From Image and Text to Beyond

    [https://arxiv.org/abs/2502.12048](https://arxiv.org/abs/2502.12048)

    本综述系统梳理了2017至2025年间利用生成式人工智能（GAN、VAE、Transformer、扩散模型等）将脑电信号转化为图像、文本和音频的研究进展，涵盖数据集、特征编码技术、评估指标及该领域面临的主要挑战。

    

    将神经活动解码为人类可理解的表示是脑机接口（BCI）和计算神经科学的一个关键研究方向。近年来机器学习和生成式人工智能的进展，推动了将非侵入式脑电图（EEG）信号转换为图像、文本和音频的研究兴趣日益增长。本综述整合并分析了脑电到图像合成、脑电到文本生成以及脑电到音频重建等方向的研究进展。我们在主要数据库中进行了结构化的文献检索（2017-2025年），提取了关于数据集、生成式架构（GAN、VAE、Transformer、扩散模型）、脑电特征编码技术、评估指标以及塑造该领域当前工作的主要挑战的关键信息。我们的综述发现，脑电到图像的模型主要采用基于GAN、VAE或扩散模型的编码器-解码器架构；脑电到文本的方法则日益……

    arXiv:2502.12048v4 Announce Type: replace  Abstract: Decoding neural activity into human-interpretable representations is a key research direction in brain-computer interfaces (BCIs) and computational neuroscience. Recent progress in machine learning and generative AI has driven growing interest in transforming non-invasive Electroencephalography (EEG) signals into images, text, and audio. This survey consolidates and analyzes developments across EEG-to-image synthesis, EEG-to-text generation, and EEG-to-audio reconstruction. We conducted a structured literature search across major databases (2017-2025), extracting key information on datasets, generative architectures (GANs, VAEs, transformers, diffusion models), EEG feature-encoding techniques, evaluation metrics, and the major challenges shaping current work in this area. Our review finds that EEG-to-image models predominantly employ encoder-decoder architectures built on GANs, VAEs, or diffusion models; EEG-to-text approaches increa
    
[^316]: 主动学习助力生成推进已知帕累托前沿的分子

    Active Learning Enables Generation of Molecules that Advance the Known Pareto Front

    [https://arxiv.org/abs/2501.02059](https://arxiv.org/abs/2501.02059)

    该论文提出了一种基于主动学习的闭环分子生成流水线，通过在新的量子化学模拟数据上迭代重训练，成功生成了性质超越训练分布的分子，并将分布外分子分类准确率提高了79%。

    

    尽管生成模型在发现具有优化所需性质的分子方面前景广阔，但它们往往无法提出可合成的分子来改进训练分布中所代表结构的性质。我们发现这一限制不仅源于分子生成过程本身，还源于分子性质预测器较差的泛化能力。我们通过创建一个闭环分子生成流水线来应对这一挑战，该流水线基于新的量子化学模拟数据进行迭代重训练。与静态的单次生成建模方法相比，只有我们的闭环迭代工作流程能够生成性质超出训练分布的分子（最多超出原始范围0.44个标准差），并将分布外分子分类准确率提高了79%。此外，通过对分子生成模型进行条件约束……

    arXiv:2501.02059v2 Announce Type: replace  Abstract: Although generative models hold promise for discovering molecules with optimized desired properties, they often fail to suggest synthesizable molecules that improve upon the properties of the structures represented in the training distribution. We find that this limitation arises not only from the molecule generation process itself, but also from the poor generalization capabilities of molecular property predictors. We address this challenge by creating a closed-loop molecule generation pipeline with iterative retraining on new quantum chemical simulation data. Compared against static, single-pass generative modeling approaches, only our closed-loop iterative workflow generates molecules with properties extending beyond the training distribution (up to 0.44 standard deviations beyond the original range) and achieves a 79% improvement in out-of-distribution molecule classification accuracy. Furthermore, by conditioning molecular gener
    
[^317]: 安全关键系统中复杂事件预测的不确定性度量

    Uncertainty measurement for complex event prediction in safety-critical systems

    [https://arxiv.org/abs/2411.01289](https://arxiv.org/abs/2411.01289)

    该论文提出了一种结合机器学习、敏感性分析和保形预测的方法（ML_CP），用于度量安全关键嵌入式系统中复杂事件预测的不确定性。

    

    复杂事件是由其他基本事件按照定义的模式和规则组合而成的。我们不再依赖专家手工构建模型规则，而是使用机器学习（ML）根据输入数据自主定义这些模式和规则，以产生所需的复杂事件。复杂事件处理（CEP）的不确定性对于嵌入式系统和安全关键系统至关重要。本文举例说明了如何测量事件感知和预测的不确定性，涵盖了对安全至关重要的嵌入式系统。随后，我们提出了一种结合机器学习和敏感性分析的方法（ML_CP），用以验证输出如何随每个输入参数变化。此外，我们的模型还测量了与预测复杂事件相关的不确定性。因此，我们使用保形预测来构建预测区间，因为模型本身存在不确定性，且数据也存在不确定性。

    arXiv:2411.01289v2 Announce Type: replace  Abstract: Complex events originate from other primitive events combined according to defined patterns and rules. Instead of using specialists' manual work to compose the model rules, we use machine learning (ML) to self-define these patterns and regulations based on incoming input data to produce the desired complex event. Complex events processing (CEP) uncertainty is critical for embedded and safety-critical systems. This paper exemplifies how we can measure uncertainty for the perception and prediction of events, encompassing embedded systems that can also be critical to safety. Then, we propose an approach (ML_CP) incorporating ML and sensitivity analysis that verifies how the output varies according to each input parameter. Furthermore, our model also measures the uncertainty associated with the predicted complex event. Therefore, we use conformal prediction to build prediction intervals, as the model itself has uncertainties, and the dat
    
[^318]: 通过协调双重动态索引机制释放大语言模型在序列推荐中的潜力

    Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism

    [https://arxiv.org/abs/2409.09253](https://arxiv.org/abs/2409.09253)

    该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。

    

    由于在语义理解和逻辑推理方面具有前所未有的能力，大语言模型（LLMs）在开发下一代序列推荐系统（RSs）方面展现出了巨大的潜力。然而，现有的基于LLM的序列推荐系统大多将索引生成与序列推荐相分离，导致语义信息与协同信息之间的整合不足。另一方面，对用户相关信息的忽视阻碍了基于LLM的序列推荐系统利用高阶用户-物品交互模式。在本文中，我们提出了端到端双重动态（ED²）推荐器，这是首个采用双重动态索引机制的基于LLM的序列推荐系统，旨在同时解决上述局限性。双重动态索引机制不仅能够将索引生成和序列推荐整合到统一的以LLM为主干的流水线中，还能使其……

    arXiv:2409.09253v2 Announce Type: replace-cross  Abstract: Owing to the unprecedented capability in semantic understanding and logical reasoning, large language models (LLMs) have shown fantastic potential in developing next-generation sequential recommender systems (RSs). However, existing LLM-based sequential RSs mostly separate index generation from sequential recommendation, leading to insufficient integration between semantic information and collaborative information. On the other hand, the neglect of user-related information hinders LLM-based sequential RSs from exploiting high-order user-item interaction patterns. In this paper, we propose the End-to-End Dual Dynamic (ED$^2$) recommender, the first LLM-based sequential RS which adopts dual dynamic index mechanism, targeting resolving the above limitations simultaneously. The dual dynamic index mechanism can not only assembly index generation and sequential recommendation into a unified LLM-backbone pipeline, but also make it pra
    
[^319]: DRL-AdaPart：基于深度强化学习驱动的自适应STAR-RIS分区方法，实现公平高效的资源利用

    DRL-AdaPart: DRL-Driven Adaptive STAR-RIS Partitioning for Fair and Efficient Resource Utilization

    [https://arxiv.org/abs/2407.06868](https://arxiv.org/abs/2407.06868)

    提出了一种基于深度强化学习的自适应STAR-RIS单元分区方法DRL-AdaPart，通过联合优化相移与子表面分配变量并引入惩罚项智能停用多余单元，在保证资源高效利用的同时，为静态和移动用户提供公平且高速的数据速率。

    

    在本工作中，我们提出了一种同时传输与反射可重构智能表面（STAR-RIS）单元的高效资源利用方法，以确保公平且高速的数据速率。我们引入了一个子表面分配变量，用于确定分配给每个用户的STAR-RIS单元数量，并通过使用经过适当定制的深度强化学习（DRL）算法，联合优化STAR-RIS的相移和子表面分配变量，从而最大化数据速率之和。所提出的DRL方法还与Dinkelbach算法以及所设计的混合DRL方法进行了比较。在DRL模型中引入了惩罚项，通过在不需要时智能地停用STAR-RIS单元来增强资源利用率。所提出的DRL方法能够为静态和移动用户实现公平且高速的数据速率，同时通过广泛的（仿真验证）确保高效的资源利用。

    arXiv:2407.06868v3 Announce Type: replace-cross  Abstract: In this work, we propose a method for efficient resource utilization of simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS) elements to ensure fair and high data rates. We introduce a subsurface assignment variable that determines the number of STAR-RIS elements allocated to each user and maximizes the sum of the data rates by jointly optimizing the phase shifts of the STAR-RIS and the subsurface assignment variables using an appropriately tailored deep reinforcement learning (DRL) algorithm. The proposed DRL method is also compared with a Dinkelbach algorithm and the designed hybrid DRL approach. A penalty term is incorporated into the DRL model to enhance resource utilization by intelligently deactivating STAR-RIS elements when not required. The proposed DRL method can achieve fair and high data rates for static and mobile users while ensuring efficient resource utilization through extensi
    
[^320]: 突破序贯校准问题的 $T^{2/3}$ 瓶颈

    Breaking the $T^{2/3}$ Barrier for Sequential Calibration

    [https://arxiv.org/abs/2406.13668](https://arxiv.org/abs/2406.13668)

    本文首次突破了序贯校准问题中 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界，改进了这一停滞二十余年的经典界限。

    

    如果预测者做出的每个预测都能在其进行该预测的时间步子集上紧密逼近结果的经验分布，则称这组概率预测是校准的。我们研究了在标准 $\ell_1$ 校准误差度量下二值序列在线校准预测这一基本问题，该问题最早由 Foster & Vohra（1998）研究。他们提出了一个在 $T$ 个时间步后校准误差为 $O(T^{2/3})$ 的算法，并证明了 $\Omega(T^{1/2})$ 的下界。这些界限在二十年间一直停滞不前，直到 Qiao & Valiant（2021）通过引入一种名为“符号保持”的组合博弈，并证明该博弈的下界可以推出校准问题的下界，从而将下界改进为 $\Omega(T^{0.528})$。在本文中，我们首次对 Foster & Vohra 提出的 $O(T^{2/3})$ 校准误差上界做出了改进，我们通过引入一种变体

    arXiv:2406.13668v4 Announce Type: replace  Abstract: A set of probabilistic forecasts is calibrated if each prediction of the forecaster closely approximates the empirical distribution of outcomes on the subset of timesteps where that prediction was made. We study the fundamental problem of online calibrated forecasting of binary sequences under the standard $\ell_1$ calibration error metric, which was initially studied by Foster & Vohra (1998). They derived an algorithm with $O(T^{2/3})$ calibration error after $T$ time steps, and showed a lower bound of $\Omega(T^{1/2})$. These bounds remained stagnant for two decades, until Qiao & Valiant (2021) improved the lower bound to $\Omega(T^{0.528})$ by introducing a combinatorial game called sign preservation and showing that lower bounds for this game imply lower bounds for calibration.   In this paper, we give the first improvement to the $O(T^{2/3})$ upper bound on calibration error of Foster & Vohra. We do this by introducing a variant
    
[^321]: 基于阻尼高斯-牛顿优化的多目标超参数搜索

    Multi-Objective Hyperparameter Search via Damped Gauss--Newton Optimization

    [https://arxiv.org/abs/2401.03580](https://arxiv.org/abs/2401.03580)

    本文提出一种基于阻尼高斯-牛顿优化的多目标超参数搜索方法，利用有限差分雅可比矩阵和Tikhonov正则化实现有向的联合参数更新，在欠定情况下以远少于网格搜索的试验次数达到同等的最佳验证精度。

    

    本文从数值优化的角度研究超参数优化（HPO），提出了一种多目标的阻尼高斯-牛顿搜索方法。与将模型评估视为独立试验的做法不同，该方法通过有限差分估计雅可比矩阵，以捕捉多个验证指标对超参数扰动的局部敏感性。随后，利用Tikhonov正则化的高斯-牛顿系统产生有向的联合更新，从而解决了超参数数量超过性能目标数量时的欠定问题。作者通过调整四个XGBoost超参数，在三个公开分类数据集上对该方法进行了评估，并与穷举网格搜索、随机搜索以及树结构Parzen估计器（TPE）优化进行了比较。在一个受控的乳腺癌数据集划分上，所提出的方法达到了320个配置的网格搜索所获得的最佳验证精度，同时以略……（摘要原文在此处截断）

    arXiv:2401.03580v2 Announce Type: replace-cross  Abstract: We study hyperparameter optimization (HPO) from a numerical-optimization perspective and propose a multi-objective, damped Gauss--Newton search method. Rather than treating model evaluations as independent trials, the method estimates a finite-difference Jacobian that captures the local sensitivity of multiple validation metrics to hyperparameter perturbations. A Tikhonov-regularized Gauss--Newton system then produces a directed joint update, addressing the underdetermined setting in which the number of hyperparameters exceeds the number of performance objectives. We evaluate the method on three public classification datasets by tuning four XGBoost hyperparameters and compare it with exhaustive grid search, random search, and tree-structured Parzen estimator (TPE) optimization. On a controlled Breast Cancer split, the proposed method matches the best validation accuracy of a 320-configuration grid search while obtaining slightl
    
[^322]: 拓扑增强的语音信号处理机器学习

    Topology-enhanced machine learning for speech signal processing

    [https://arxiv.org/abs/2311.15210](https://arxiv.org/abs/2311.15210)

    本文提出了一种透明的拓扑特征捕捉方法TopCap，通过将时间序列的拓扑特征融入神经网络，在语音信号处理任务中提升了准确性、抗噪鲁棒性、稳定性与可解释性。

    

    在人工智能辅助信号处理中，现有的深度学习模型通常呈现黑箱结构。在本文中，我们在概念上超越了频谱分析，证明拓扑方法不仅能有效捕捉内在而复杂的结构信息，还能增强神经网络。我们提供了一种透明的拓扑方法论TopCap，用于捕捉时间序列中固有的拓扑特征以进行基础机器学习。与先前的方法相比，我们获得的描述符能够探测更精细的信息，例如时间序列的振动特性。值得注意的是，在浊音与清音辅音的分类任务中，TopCap所达到的准确率始终可以与神经网络模型相媲美。此外，通过将TopCap特征集成到神经网络中，我们的方法在抗噪鲁棒性以及准确率、稳定性、损失函数收敛性和可解释性等方面均优于当前最先进的方法。

    arXiv:2311.15210v2 Announce Type: replace  Abstract: In artificial-intelligence-aided signal processing, existing deep learning models often exhibit a black-box structure. Here, conceptually beyond spectral analysis, we demonstrate that topological methods not only effectively capture intrinsic and complex structural information but can also enhance neural networks. We provide a transparent methodology, TopCap, to capture topological features inherent in time series for basic machine learning. Compared to prior approaches, we obtain descriptors that probe finer information such as the vibration of a time series. Notably, in classifying voiced and voiceless consonants, TopCap achieves an accuracy consistently standing in comparison with neural network models. Moreover, by integrating TopCap features into those neural networks, our approach improves upon state-of-the-art methods in terms of robustness against noise, as well as accuracy, stability, convergence of loss function, and interp
    
[^323]: 将Adam优化器推广到流形以高效训练Transformer

    Generalizing Adam to Manifolds for Efficiently Training Transformers

    [https://arxiv.org/abs/2305.16901](https://arxiv.org/abs/2305.16901)

    本文利用齐性流形（如Stiefel流形、辛Stiefel流形和Grassmann流形）所具有的全局切空间（李子空间）表示这一特殊结构，提出了一种将Adam优化器完整推广到流形上的新方法，从而实现对Transformer的高效训练。

    

    arXiv:2305.16901v5 公告类型：替换 摘要：神经网络取得成功的主要原因之一是涌现了一系列新的、高度成功的优化器，其中最重要的或许就是Adam优化器。它被广泛用于训练神经网络，但却以难以解释而著称。由于缺乏清晰的物理直觉，Adam很难被推广到流形上。此前已有一些尝试直接将Adam算法的部分内容应用于流形，或寻找其潜在结构，但完整的推广一直未能实现。在这项工作中，我们提出了一种新方法，利用与神经网络优化相关的流形的特殊结构，例如Stiefel流形、辛Stiefel流形和Grassmann流形：所有这些流形都是齐性空间，因此容许全局切空间表示。这是一个公共的向量空间，通常被称为李子空间，它使得推广……

    arXiv:2305.16901v5 Announce Type: replace  Abstract: One of the primary reasons behind the success of neural networks has been the emergence of an array of new, highly-successful optimizers, perhaps most importantly the Adam optimizer. It is widely used for training neural networks, yet notoriously hard to interpret. Lacking a clear physical intuition, Adam is difficult to generalize to manifolds. Some attempts have been made to directly apply parts of the Adam algorithm to manifolds or to find an underlying structure, but a full generalization has remained elusive.   In this work a new approach is presented that leverages the special structure of the manifolds which are relevant for optimization of neural networks, such as the Stiefel manifold, the symplectic Stiefel manifold and the Grassmann manifold: all of these are homogeneous spaces and as such admit a global tangent space representation. This is a common vector space, often called the Lie subspace, that makes the generalization
    
[^324]: 迁移学习的极限

    Limits of Transfer Learning

    [https://arxiv.org/abs/2006.12694](https://arxiv.org/abs/2006.12694)

    该论文在算法搜索框架下证明了迁移学习的若干理论极限，表明迁移信息必须经过谨慎选择并与目标问题存在依赖关系，同时算法的概率变化程度决定了其性能改进的上限。

    

    迁移学习是指从一个问题领域中获取信息和洞察，并将其应用于新的问题领域。尽管迁移学习在实践中被广泛使用，但其理论发展仍不够完善。为了解决这一问题，我们证明了若干与迁移学习相关的新结果，表明需要仔细选择要迁移的信息集合，并且迁移的信息与目标问题之间必须存在依赖关系。此外，我们证明了使用迁移学习的算法的概率变化程度如何对其可能实现的改进量设定了上限。这些结果建立在机器学习的算法搜索框架之上，使得这些结论能够适用于广泛的迁移学习问题。

    arXiv:2006.12694v2 Announce Type: replace-cross  Abstract: Transfer learning involves taking information and insight from one problem domain and applying it to a new problem domain. Although widely used in practice, theory for transfer learning remains less well-developed. To address this, we prove several novel results related to transfer learning, showing the need to carefully select which sets of information to transfer and the need for dependence between transferred information and target problems. Furthermore, we prove how the degree of probabilistic change in an algorithm using transfer learning places an upper bound on the amount of improvement possible. These results build on the algorithmic search framework for machine learning, allowing the results to apply to a wide range of learning problems using transfer.
    
[^325]: 可靠的学习方法应对测试时攻击与分布偏移

    Reliable Learning for Test-time Attacks and Distribution Shift. (arXiv:2304.03370v1 [cs.LG])

    [http://arxiv.org/abs/2304.03370](http://arxiv.org/abs/2304.03370)

    本文提出了可靠的学习方法以抵御测试时攻击和分布偏移，在测试时引入了新的可靠性保障方法，确保预测结果正确。同时，该学习方法能够适应任意测试点，具有非常好的可靠性。

    

    机器学习算法经常被用于即使经过精心获得的训练数据也无法准确捕捉的环境中，这既可能是由于测试时的“对抗性”攻击，也可能是因为“自然”的数据分布偏移。针对测试时攻击，我们提出并分析一种新颖的稳健性可靠性保证方法，要求学习器输出一个可靠半径 $\eta$ 的预测结果，意味着只要对手没有扰动测试点超过距离 $\eta$，它的预测结果就是正确的。我们提供了在任意测试点上都能输出最佳可靠性半径的最优学习器，并且特征化了可靠区域即可达到给定可靠性半径的点集。我们还分析了在分布偏移下的可靠学习方法，其中测试点可能来自于一个与训练分布不同的任意分布 $Q$。

    Machine learning algorithms are often used in environments which are not captured accurately even by the most carefully obtained training data, either due to the possibility of `adversarial' test-time attacks, or on account of `natural' distribution shift. For test-time attacks, we introduce and analyze a novel robust reliability guarantee, which requires a learner to output predictions along with a reliability radius $\eta$, with the meaning that its prediction is guaranteed to be correct as long as the adversary has not perturbed the test point farther than a distance $\eta$. We provide learners that are optimal in the sense that they always output the best possible reliability radius on any test point, and we characterize the reliable region, i.e. the set of points where a given reliability radius is attainable. We additionally analyze reliable learners under distribution shift, where the test points may come from an arbitrary distribution Q different from the training distribution 
    

