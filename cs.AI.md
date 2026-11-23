# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Linear Regression Without a Variance](https://arxiv.org/abs/2611.27983) | 该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。 |
| [^2] | [Non-penetrating World Models](https://arxiv.org/abs/2611.27963) | 该论文提出NoPeBench基准和NoPeWorld框架，通过在冻结的扩散Transformer隐藏状态上训练小型预测头来检测并替换不安全动作，有效防止交互式视频世界模型中的物体穿透问题。 |
| [^3] | [Model Carving: Training Transformers That Optimize Their Own Topology](https://arxiv.org/abs/2611.27962) | 提出模型雕刻方法，通过在每个子层上添加可学习的标量门控，让 Transformer 在单次训练中同时学习权重与架构，由训练损失自动移除冗余组件，释放出内部更小、更高效的异构网络。 |
| [^4] | [BioEmu2: Scalable Emulation of Biomolecular Dynamics](https://arxiv.org/abs/2611.27924) | BioEmu2是一个融合静态结构预测、平衡态采样与时间演化的原子级生物分子动力学生成模型，采样速度比分子动力学快数个数量级，可支持兆道尔顿级多组分复合物预测，平衡分布自由能误差接近1 kcal/mol，并能预测长时程动力学轨迹。 |
| [^5] | [SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement](https://arxiv.org/abs/2611.27919) | 本文提出谱通用优势特征（SUAF），通过对以动作为中心的典型依赖矩阵进行奇异值分解，构建可通过线性奖励投影恢复延迟优势的成对特征，从而在奖励无关表征下实现零样本策略改进。 |
| [^6] | [StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning](https://arxiv.org/abs/2611.27915) | 提出了首个几何感知的双目视觉语言模型StereoVLM，利用标定双目相机的显式跨视角几何消除现有VLM在度量三维空间推理中的歧义。 |
| [^7] | [Star Panoptic Quality: Selectively Evaluating Split and Merged Instances](https://arxiv.org/abs/2611.27913) | 提出了新型评估框架StarPQ（星形全景质量），通过构建星形森林等方法，能够检测存在偏差的参考标注中分裂或合并的实例，并区分有效与无效的过度分割和欠分割。 |
| [^8] | [Functional Depth Is Learned: Isolating the Computation That Layers Produce Only Together](https://arxiv.org/abs/2611.27911) | 本文提出一种精确的容斥分解方法来分离“连通分量”（即各层唯有共同作用时才会产生的计算），并通过对 Pythia、OLMo、Amber 公开检查点的追踪发现这种跨层协同计算随预训练不断增强且删除它会显著损害模型性能，从而证明 Transformer 的功能深度是在训练过程中习得的。 |
| [^9] | [Decision Columns as an Index: Interpretable Embeddings for Explainable Entity Matching and Search](https://arxiv.org/abs/2611.27910) | 该论文提出用“决策列”（决策模型对自然语言问题给出的“是/否/未提及”校准答案）构建可解释嵌入，使相似度可精确分解为逐问题贡献、支持语义过滤与增量扩展，并在产品基准上取得与稠密嵌入相当的实体匹配效果，同时提供精确可审计的解释。 |
| [^10] | [The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search](https://arxiv.org/abs/2611.27899) | 该论文提出了一种双层情感框架，第一层利用微调的编码器-解码器大语言模型实现大规模用户情感回归检测（PR-AUC提升9.8%、测量延迟从数周降至数小时、问卷开销降低40%以上），第二层提供细粒度的UX归因诊断，从而实现网络搜索中可扩展的自动化用户体验评估。 |
| [^11] | [TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code?](https://arxiv.org/abs/2611.27895) | 该论文提出了TEA-Bench-BR——一个基于巴西早期自闭症筛查流程、以葡萄牙语规范编写的多层级代码生成基准，用以严格评测编码智能体将分支化临床规范转化为正确代码的能力，结果揭示了大模型在复杂任务层级和输入验证方面的显著局限。 |
| [^12] | [Speculative Decoding Is Less Speculative Than You Think](https://arxiv.org/abs/2611.27891) | 本文提出DICE方法，通过预先采样并在草稿生成与验证阶段之间同步采样随机性，将有限的草稿计算集中于更可能被目标模型实际采用的候选续写，从而提升投机解码的推理效率。 |
| [^13] | [LesionAgent: Training-Free Agentic Tumor Tracking and Response Assessment in Longitudinal CT](https://arxiv.org/abs/2611.27876) | 该论文提出无需训练的LLM智能体LesionAgent，像放射科医生一样调用冻结的3D基础模型并对照临床指南，在纵向CT中实现端到端的肿瘤追踪与疗效评估，追踪数量和测量精度均超越所有自动化基线。 |
| [^14] | [Beyond Visual Fidelity: Comparing Categorical and Continuous World Models](https://arxiv.org/abs/2611.27870) | 该论文通过受控的2D和3D环境实验系统比较了类别型与连续型预测对世界模型动态学习的影响，发现在无损视觉编码条件下，类别型模型能以更少训练计算量获得更高的动态rollout有效性和更好的随机性校准，表明视觉逼真度并不等同于动态保真度。 |
| [^15] | [Integrating Generative Artificial Intelligence with Learning Style Models: Opportunities, Challenges, and Future Directions](https://arxiv.org/abs/2611.27868) | 本综述系统梳理了41项研究，探讨生成式人工智能与VARK、Kolb等经典学习风格模型整合以支持大型在线课堂学生参与的机遇、挑战与未来方向，并指出AI可实现灵活、多模态、实时的个性化学习适应。 |
| [^16] | [SLICR: Strategy Logic-Guided Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.27865) | 提出SLICR框架，通过策略逻辑将规范引导强化学习首次推进到部分可观测的对抗性多智能体环境，不仅能推理智能体自身如何满足规范，还能推理如何应对其他智能体的策略和行为。 |
| [^17] | [Who Grades the Agent? Privilege Separation for Coding-Agent Benchmarks](https://arxiv.org/abs/2611.27860) | 本文提出 SplitBench，这是首个采用权限分离机制、将编码智能体基准测试中“运行测试”与“判定结果”相隔离的评估框架，可防止智能体操纵评估过程来骗取虚高的分数。 |
| [^18] | [Beyond Self-Critique: Epistemic Convergence Control for Multi-Agent Research Verification](https://arxiv.org/abs/2611.27839) | 该论文提出一个多智能体研究验证框架，通过倡导者-怀疑者-仲裁者的非对称角色架构从结构上消除同源模型自我批判偏差，并利用信念差分收敛控制替代固定迭代协议，从而提升科学论断验证的可靠性与计算效率。 |
| [^19] | [Auditable Financial Knowledge Graphs: Schema-Driven Construction and Intrinsic Quality Evaluation](https://arxiv.org/abs/2611.27835) | 该论文提出了一种可审计的模式驱动方法，通过分阶段大语言模型提取、多策略实体消解、时间约束与细粒度溯源，从金融文档构建时序知识图谱并对其内在质量进行评估。 |
| [^20] | [Graph Signal Generation via Graph-Aware Diffusion Models](https://arxiv.org/abs/2611.27826) | 提出图感知扩散模型（GAD），将前向过程建模为图上的热扩散，并设计模式自适应信噪比调度（MASS），为每个图频率分配独立扩散速率以保持相同信噪比，从而实现更好的图信号生成与分布学习。 |
| [^21] | [Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks](https://arxiv.org/abs/2611.27811) | 该论文提出了一种模拟生物视觉皮层简单细胞与复杂细胞交替结构的浅层神经网络架构，通过能够学习选择性与容忍性表征的新型组件，实现了准确、鲁棒且与人类感知更一致的图像分类，并能有效抵御强对抗攻击，避免对未知类别样本做出高置信度的错误预测。 |
| [^22] | [SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech](https://arxiv.org/abs/2611.27810) | 提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。 |
| [^23] | [$\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2611.27808) | 提出理论框架$\mathtt{DistP}$，将$\mathtt{DDP}$的零样本超参数迁移规则扩展到本地更新训练，使学习率能够跨模型宽度、深度和同步间隔直接迁移，从而免去昂贵的超参数调优。 |
| [^24] | [ChildAI-Gov: An Evidence-Linked AI System for Assessing EdTech Policies Against Children's Digital Rights](https://arxiv.org/abs/2611.27799) | 该论文提出ChildAI-Gov系统，通过儿童权利影响评分（CRIS）结合大语言模型判断、规则检查与检索增强生成技术，将教育科技政策的合规评估结果与儿童权利框架的证据条款关联起来，实现可追溯、自动化的儿童数字权利评估。 |
| [^25] | [Neural Defend ARCAS 290M: A Robust Foundation Model for AI-Generated Audio Detection](https://arxiv.org/abs/2611.27784) | ARCAS 290M是一个冻结的基础模型，无需针对特定基准微调，即可在跨生成器类型、信道条件和语言的多种场景下实现最先进的AI生成音频检测。 |
| [^26] | [A Benchmark Protocol and Fairness Analysis Towards Open-Set EEG Seizure Type Classification](https://arxiv.org/abs/2611.27783) | 该论文提出了首个基于 TUSZ 数据集的脑电图癫痫发作类型分类开放集识别基准协议，利用临床分类体系定义已知与未知发作类型，并在部分微调的脑电图基础模型上系统评估了七种开放集评分方法及其公平性。 |
| [^27] | [Search-RSI: Expanding Search Worlds Along Multimodal Agents' Capability Frontier](https://arxiv.org/abs/2611.27780) | 提出Search-RSI递归自我改进框架，让多模态搜索智能体诊断自身能力边界，并同步生长新任务及其证据，从而解决因检索后端缺少证据而累积的不可达负样本问题。 |
| [^28] | [QuASAR: Toward an LLM-Assisted Framework for Screening, Prototyping, and Assessing Quantum-Advantage Opportunities in Scientific Computing](https://arxiv.org/abs/2611.27778) | QuASAR提出了一种由两个LLM智能体组成的实验性流水线，通过筛选文献中具有量子算法对应物的计算内核、独立验证引文、构建量子映射原型并与经典基线进行基准对比，来评估科学计算中的量子优势机会，在天体物理学的十项候选计算中成功完成了九项小规模基准原型。 |
| [^29] | [RoSci: Exploring Robustness in Science VLMs](https://arxiv.org/abs/2611.27750) | 该论文提出科学模型鲁棒性的概念，并构建了RoSci-Bench诊断基准和采用“解析-验证-决策”流程的RoSci-Agent框架，以增强视觉语言模型在科学任务中应对信息错误与信息缺失的能力。 |
| [^30] | [Shared Understanding in Human-AI Teams: Coordination Continuity in Evolving Work](https://arxiv.org/abs/2611.27746) | 该论文提出“协调连续性”这一新概念，并构建了由持久表示、相对协调属性和交互机制三要素组成的框架，用以解释人机团队如何在演进的异步工作中持续维持相互依赖行动的共同基础。 |
| [^31] | [Can Vision-Language Models Synthesize Executable Game-Playing Programs?](https://arxiv.org/abs/2611.27741) | 该论文提出了PlayScriptBench基准，评估视觉语言模型能否从游戏状态直接合成可执行的游戏程序，以克服逐步交互范式在实时性和调用成本方面的根本局限。 |
| [^32] | [Average-Depth Variable Execution via Request-Local Token Routing](https://arxiv.org/abs/2611.27738) | 提出AVERT方法，通过请求局部路由让每个Token动态决定执行的Transformer层数，使请求的平均执行深度精确匹配连续预算，在降低大语言模型推理成本的同时保持输出质量，且单一检查点可同时服务不同预算的请求。 |
| [^33] | [Quantitative Measurement of Pavement Distress based on Vehicle Camera and Perspective Correction](https://arxiv.org/abs/2611.27728) | 该研究提出了一种无需现场相机标定、基于车道线检测构建透视变换矩阵的路面病害定量测量方法，实现了从车载相机图像中直接获取病害的长度或面积等量化指标。 |
| [^34] | [Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders](https://arxiv.org/abs/2611.27721) | 该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。 |
| [^35] | [Recoverable Symbols, Unstable Execution: Auditing Internal Pathways and Decision Identity in Visual State Tracking](https://arxiv.org/abs/2611.27715) | 该论文提出一种诊断框架，证明即使视觉特征中符号信息完全可恢复，循环执行器的决策仍会因图像无关上下文的变化而不稳定，且这种不稳定主要源于依赖输入的旋转相位——恢复原始相位可修复大部分新增错误。 |
| [^36] | [MetaRigor: From Agentic Automation to Verifiable End-to-End Meta-Analysis Through Scientific Decision Boundaries](https://arxiv.org/abs/2611.27713) | MetaRigor通过任务合同、程序化规则和来源绑定在任务层面实现科学决策边界，让大语言模型负责语义解释、程序负责形式化验证，从而构建了首个可验证的端到端自动化荟萃分析工作流框架，完整结果交付的F1分数达86.29%。 |
| [^37] | [When Are Images Needed in Multimodal Machine Translation? Text-Only Routing for On-Demand Visual Context](https://arxiv.org/abs/2611.27712) | 该论文将多模态机器翻译中的图像调用形式化为“源文本充分性预测”任务，构建了含三个细粒度标签的视觉依赖数据集，实现仅凭源文本即可按需判断是否需要视觉上下文的路由方法，从而避免冗余的视觉计算开销。 |
| [^38] | [Abstract What Varies, Reuse Longer Patterns: Efficient Offline Chain-of-Thought Compression](https://arxiv.org/abs/2611.27710) | CRAFT通过将问题特定细节抽象为槽位来发现跨样本的部分词汇化表达式模板并聚类为族，只需对每个族的代表进行一次压缩即可复用共享编辑操作，从而将压缩器预算降低中位数90.7%且准确率损失仅1.21个百分点。 |
| [^39] | [Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process](https://arxiv.org/abs/2611.27698) | 该论文提出一种时空图神经网络框架，通过将正常运行空间分解为特定于运行条件的子空间，并结合GATv2图结构学习与时间编码，解决了化工过程中因运行条件变化导致的误报与漏检问题，并在田纳西-伊士曼过程上验证了其故障检测能力。 |
| [^40] | [Comparative Performance of Machine Learning Architectures for Fault Detection and Diagnosis in Chemical Processes](https://arxiv.org/abs/2611.27696) | 本研究以田纳西-伊士曼过程（TEP）为基准，系统比较了多种机器学习架构（包括集成学习、LSTM、XGBoost等）在化工过程故障检测与诊断任务中的性能，为实际工业场景下的方法选择提供了依据。 |
| [^41] | [Embodied-Harness: Towards Recursive Self-Improvement for Robotic Manipulation](https://arxiv.org/abs/2611.27689) | 提出双智能体框架 Embodied-Harness，通过分层记忆、基于证据的经验反思与递归自我改进，在不更新模型参数的情况下提升机器人操作能力。 |
| [^42] | [Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs](https://arxiv.org/abs/2611.27686) | 本文提出混合神经形态联邦学习框架，使异构的人工神经网络与脉冲神经网络设备能够在联邦学习环境中协同联合训练，从而推动向节能神经形态边缘AI的渐进式过渡。 |
| [^43] | [Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild](https://arxiv.org/abs/2611.27673) | Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。 |
| [^44] | [Learning Inside Types: Epistemic Typestates for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.27631) | 本文提出“认知类型状态”这一类型规则，让多智能体强化学习智能体在记录自身信念的类型协议内学习，仅执行在所有可能世界中都安全的动作，从而无需通信即可实现类型可靠性与团队安全性。 |
| [^45] | [Benchmarks' Last Exam](https://arxiv.org/abs/2611.27622) | 该论文首次系统量化了主流AI基准测试的任务有效性，发现经过公开纠错修订的封闭式基准达到92.5%的验证有效性，而多步骤套件仅有29.3%，揭示了当前评估体系在提示、环境与评分器一致性上的显著缺陷。 |
| [^46] | [Potential emergence in LLM agents: a statistical physics perspective](https://arxiv.org/abs/2611.27618) | 该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。 |
| [^47] | [PromptTTSD: Controllable Text to Spoken Dialogue Synthesis with Open-Ended Text Descriptions](https://arxiv.org/abs/2611.27601) | 该论文提出PromptTTSD，通过从多叙述者有声书自动挖掘对话语音、改进两阶段对齐算法保持对话连续性，并利用LLM从叙事文本生成开放式风格描述，实现了以自然语言细粒度控制多说话人口语对话合成的能力，同时发布了8383小时中文的PromptDialog大规模对话数据集。 |
| [^48] | [Unlocking Heterogeneous Accelerator Flexibility in AI Data Centers: Sizing and Workload Allocation](https://arxiv.org/abs/2611.27576) | 本文提出联合优化异构加速器的容量规划与AI工作负载（训练、批处理、在线推理）的跨站点跨时段分配，以最低成本满足时变电网约束，在双站点案例中相比固定容量的时空调度使日成本降低6.5%。 |
| [^49] | [Can Language Models See the Circuit in Pixelated RFIC Passives?](https://arxiv.org/abs/2611.27559) | 本研究首次验证了前沿大语言模型在完全禁用工具的情况下，能够仅凭读取18×18和27×27网格版图，就准确预测像素化射频无源器件在20–100 GHz的输入阻抗与损耗，表明语言模型可以“看懂”这类反直觉的逆设计电路结构。 |
| [^50] | [Bringing AI Assistance to the Beznau Nuclear Power Plant: A Hierarchical System of LLM Agents for Reactor Monitoring](https://arxiv.org/abs/2611.27556) | 该论文为运行超过五十年的贝兹瑙核电站构建了基于大语言模型的分层多智能体监控系统，由协调智能体分解工程师的问题并分派给仅具只读权限的专用子智能体，以自然语言编写领域流程供工程师审核修改，从而加速堆芯仪表异常调查并减少跨孤岛的人工检索负担。 |
| [^51] | [Vega-Fafnir: Is Greedy Decoding the Same Decoding? Where Rounding Noise in Speculative Verification Comes From and What Removing It Costs](https://arxiv.org/abs/2611.27554) | 该论文量化了贪心解码在不同执行方式下的舍入噪声：投机验证改变 60%、填充改变 68% 的提示输出，且首分歧处 logit 差距均不超过 0.375，噪声主要来自多词元前向传播写入的 KV 缓存条目，而基于裕量门控的回退策略无法消除噪声，重建缓存则需付出额外代价。 |
| [^52] | [Altair-Nidhogg: Which Model-Merging Theories Hold for a One-Pass Decision Encoder? Soups, Task Arithmetic, TIES and DARE Against Joint Training and Prediction Ensembling](https://arxiv.org/abs/2611.27550) | 该论文在一个单次前向决策编码器上系统检验了模型汤、任务算术、TIES 和 DARE 四种模型合并配方是否有效及其理论是否成立，发现 TIES 对专家增益的保留率最高（64%），并将其与联合训练和预测集成进行了全面对比。 |
| [^53] | [On the Limits of Observability for Evaluating Coverage in Machine Unlearning](https://arxiv.org/abs/2611.27543) | 该研究通过502个重训练模型的可观测性压力测试和Ω(log n)秩下界的理论证明，揭示了现有机器遗忘评估因只关注特定遗忘集而忽视“覆盖率”的问题，并提出了评估遗忘覆盖率的CEO框架。 |
| [^54] | [Fix Where It Fails: Root-Cause-Guided Agentic Harness Optimization for Scene Graph Generation](https://arxiv.org/abs/2611.27529) | 提出了RoHa-SGG方法，利用根因分析智能体和提议者智能体两个LLM智能体来优化冻结SGG模型周围的推理框架代码，在模型失败之处进行针对性修复而无需重新训练。 |
| [^55] | [PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks](https://arxiv.org/abs/2611.27521) | 本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。 |
| [^56] | [PhiJEPA: Denoising Latent Predictions for Visual Representation Learning](https://arxiv.org/abs/2611.27516) | PhiJEPA提出了一种两阶段去噪潜在预测架构，由Transformer从损坏上下文预测掩码教师视图的隐藏token、再由逐token的MLP将其映射到干净token，在ImageNet-1K上预训练后在PASCAL VOC线性分割中达到69.3 mIoU，超越I-JEPA和DSeq-JEPA且训练计算量减少25%。 |
| [^57] | [Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events](https://arxiv.org/abs/2611.27509) | 该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。 |
| [^58] | [The Assistant Persona Differs Across Languages in Activation Space](https://arxiv.org/abs/2611.27504) | 该研究首次从机制层面揭示了大语言模型中助手人格方向在不同语言间差异显著（与英语轴余弦相似度仅0.378–0.911）且与语言资源水平相关，证明从英语提取的激活方向无法跨语言泛化。 |
| [^59] | [Turnout: Adaptive Attention Switching for Efficient Reasoning](https://arxiv.org/abs/2611.27500) | 提出Turnout方法，在预训练全注意力模型中通过蒸馏保留线性注意力路径，并利用强化学习训练的自适应策略在每一层动态切换全注意力与线性注意力，从而在大幅降低内存消耗的同时保持长文本推理任务的准确率。 |
| [^60] | [Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation](https://arxiv.org/abs/2611.27495) | 提出SAFEHOLD防御框架，将拒绝行为建模为持续的生成过程而非单次决策，使代码语言模型即使经过有害数据微调仍能保持拒绝有害请求的安全对齐能力。 |
| [^61] | [Learning Unknown Oscillators From Bandit Feedback](https://arxiv.org/abs/2611.27492) | 提出了一种带检验的承诺控制器，在未知振荡器驱动的半老虎机问题上实现了平方根级后悔界，并将拟合次数从1023次降至49次，带来4.69倍的加速。 |
| [^62] | [Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining](https://arxiv.org/abs/2611.27491) | 本研究通过仅改变分词器预训练39个单语和双语模型，发现改进的词表分配能显著增强模型中早期层的跨语言表示对齐以及前馈神经元的跨语言知识共享，但揭示了表示对齐与跨语言迁移之间存在分离现象。 |
| [^63] | [Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI](https://arxiv.org/abs/2611.27489) | 该论文提出将自然语言到线性时序逻辑的翻译分解为回答一组预定义答案的问题，用轻量级System-One分类器替代大语言模型的文本生成，从而在计算和能源资源受限的机器人上实现高效实时的翻译。 |
| [^64] | [From Manufacturing Event Logs to Simulation: Enabling SMEs with Agentic AI for SimPy Simulation Generation](https://arxiv.org/abs/2611.27483) | 提出了GenAI4SimPy多智能体框架，利用开放权重大语言模型Qwen 3.6将制造事件日志自动转化为可执行的SimPy离散事件仿真模型，并揭示了“保真度悖论”——过程级参数准确并不保证系统级行为保真。 |
| [^65] | [ROSWORLD: When Digital Task Competence Meets Physical Execution](https://arxiv.org/abs/2611.27467) | ROSWorld基准将物理仿真机器人工作站与实时OSWorld桌面相耦合，揭示前沿计算机使用模型虽具备强大的数字任务能力，但在通过物理接触操作键盘和鼠标时性能显著下降。 |
| [^66] | [Cecilia: Towards Layout-Inclusive Optical Music Recognition](https://arxiv.org/abs/2611.27465) | 该论文提出Cecilia——一种保留音乐内容、歌词与版面指令的紧凑MusicXML表示，并构建Cecilia-OMR模型以元数据和小节级片段进行预测，序列长度仅为MusicXML的约1/28，同时在多项基准上优于Legato 2。 |
| [^67] | [Supervaluationism about vagueness and indeterminacy: A judgment-aggregation perspective](https://arxiv.org/abs/2611.27463) | 本文引入判断聚合理论，将超赋值主义视为聚合不同规格说明判定结果的一类聚合函数之一，并对其给出公理化刻画，从而精确界定了它在处理模糊性与不确定性的各种可能方法中的逻辑位置。 |
| [^68] | [On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act](https://arxiv.org/abs/2611.27462) | 本文提出一个工具包，将欧盟《人工智能法案》合规评估的核心法律义务转化为技术规范要求和指导性问题，以支持“合法设计”的AI辅助招聘系统开发，重点关注第三方GPAI模型集成到招聘关键环节这一最具挑战性的场景。 |
| [^69] | [Mechanism Parity for Simulation-Based Validation of Autonomous-Driving Perception](https://arxiv.org/abs/2611.27447) | 本文提出“机制对等性”诊断框架，通过统一决定性特征保真度（DFF）与概念-决定性特征保真度（CDFF），验证输出一致的仿真是否保留了感知模型做出判断的内部证据机制，而不仅是视觉逼真或输出一致。 |
| [^70] | [Learning When to Ask: Advisor-Aware Post-Training for SLM Agents](https://arxiv.org/abs/2611.27443) | 提出ADAPT多阶段后训练方法，让小型语言模型智能体学会在成本感知下判断何时向更强模型求助并跨步骤遵循建议，且完全依靠环境结果监督的策略自身rollouts进行学习，无需奖励模型或人工标注。 |
| [^71] | [THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models](https://arxiv.org/abs/2611.27436) | 提出THRIFTY——一种基于Thompson采样的带背包约束多臂老虎机控制器，在固定的生成token预算下自适应地将预算分配给不同难度的问题，从而低成本地为小语言模型合成多样且正确的推理训练数据。 |
| [^72] | [No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents](https://arxiv.org/abs/2611.27434) | 该研究首次提出仅使用领域文档（无需目标音频和转写文本）对基于LLM的语音识别系统进行领域适配，通过将文档直接作为书面文本或由LLM渲染为合成口语对话，其适配效果可媲美甚至接近使用真实转写文本的结果。 |
| [^73] | [Neurodegenerative Differential Diagnosis with Knowledge-enhanced Structural Reasoning](https://arxiv.org/abs/2611.27427) | 提出了知识增强结构化推理框架KnowSR，通过将多模态患者表征与基于文献的疾病知识库（疾病原型、带符号证据方向和两两诊断关系）对齐，注入结构化疾病证据并在稀疏鉴别图上进行关系推理，从而提升神经退行性疾病的鉴别诊断能力。 |
| [^74] | [Cleave: Severing Guardrails and Auditing Spectral Coupling in Open-Weight LLMs](https://arxiv.org/abs/2611.27426) | 本文提出了无需训练的攻击方法Cleave，通过直接编辑残差写入权重移除开放权重LLM的安全护栏，可选择性解除特定安全限制并规避行为抽查与静态权重审计。 |
| [^75] | [TrojanMaxxing: One-shotting VLM Backdoors with Weight-Space Programming](https://arxiv.org/abs/2611.27425) | 该论文提出TrojanMaxxing，将VLM后门攻击重新定义为权重空间编程任务，利用LLM智能体直接检查并编辑模型权重并协同设计不可察觉的视觉触发器，无需任何训练或优化即可一次性为视觉语言模型植入后门。 |
| [^76] | [Jagged Robustness: Uneven Safety Transfer and Emerging Vulnerabilities under LLM Quantization](https://arxiv.org/abs/2611.27424) | 量化会导致大语言模型的安全防护和偏见控制能力不均衡地退化，而其他能力保持稳定，且对完整模型无害的对抗性后缀攻击竟能越狱量化后的模型。 |
| [^77] | [ADEPT: Adaptive Monocular Depth Estimation via Model-In-The-Loop Photometric Tuning with Capture Risk Prediction](https://arxiv.org/abs/2611.27423) | 提出ADEPT框架，以“模型在环”方式训练一个采集风险预测器，并借助上下文高斯过程控制器自适应调节相机曝光与增益，从而在机器人运动场景下提升单目深度估计的精度。 |
| [^78] | [Detecting Word-Substitution Poisoning of a RAG Knowledge Base with Relative Entropy and Runtime Verification](https://arxiv.org/abs/2611.27410) | 该论文发现针对RAG知识库的同义词替换投毒主要通过证据丢失和拒答降低答案质量而非制造虚假陈述，并提出利用相对熵中的词-同义词对数比项结合小型语言模型作为参考来高效检测被投毒文段，AUROC达到0.95。 |
| [^79] | [Beyond Linear Decodability: Understanding and Improving Instruction Following in World Action Models](https://arxiv.org/abs/2611.27401) | 本研究发现世界动作模型的指令遵循能力与目标信息的线性可解码性密切相关，并提出线性可解码性监督（LDS）这一简单有效的辅助训练目标，使模型在LIBERO-CF上的成功率和定位率分别大幅超越最强基线26.8和42.9个百分点。 |
| [^80] | [Taking Apart and Putting Together: Tracking Objects Through Physical Transformations](https://arxiv.org/abs/2611.27398) | 提出免训练框架 TIPO，通过“生成-评估-判断”流水线使视频跟踪系统能在物体拆解与组装等物理变换中持续跟踪实例、推断变换并维护实体状态图。 |
| [^81] | [Evidence sufficiency assessment enables cost-effective review allocation in on-premise clinical AI agents](https://arxiv.org/abs/2611.27390) | 该论文提出无需训练的证据充分性评估方法与三级顺序分诊协议，使本地临床AI智能体仅用单次推理即可达到五次一致性验证的通过率（98.91%），并节省约80%的计算成本。 |
| [^82] | [PanoUAV-Track: Full-Spherical UAV Visual Tracking in the Wild](https://arxiv.org/abs/2611.27383) | 该论文构建了首个面向移动无人机平台的全球形视觉跟踪基准PanoUAV-Track，基于40小时真实8K全景视频，包含1,895个序列、507,961帧和2,872个标注目标，填补了无人机全向跟踪（含ERP投影几何挑战及日夜场景）的研究空白。 |
| [^83] | [VMSA: Video Memory Skill Agent for Long Video Understanding](https://arxiv.org/abs/2611.27381) | 提出VMSA框架，通过联合设计三种互补的结构化视频记忆（片段字幕记忆、实体事件记忆、实体感知情节记忆）与记忆技能，提升智能体在长视频问答中跨长时间跨度检索与整合证据的能力。 |
| [^84] | [ReActWAM: Action-Consistent Caching for Efficient World-Action Models](https://arxiv.org/abs/2611.27372) | ReActWAM提出了一种无需训练的框架，通过动作意图引导的前缀跳过与块内预测重用来消除世界-动作模型闭环推理中的冗余计算，显著提升执行效率。 |
| [^85] | [AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance](https://arxiv.org/abs/2611.27369) | AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。 |
| [^86] | [Testing for Coordination Failures in Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2611.27368) | 提出了DisQ方法，利用Q值差距这一可解释信号从基于价值的合作式多智能体强化学习中筛选出能揭示协调失败的关键状态，以高效检测智能体间的不安全行为。 |
| [^87] | [WePM: Hierarchical Conflict Reconciliation for Unified MLLMs](https://arxiv.org/abs/2611.27365) | WePM通过形式化异构能力密度（HCD）、多粒度视觉预算分配和分层冲突调和，在单一视觉-语言骨干中统一了表示、生成与决策能力，打造出适用于端侧部署的口袋模型系列。 |
| [^88] | [WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection](https://arxiv.org/abs/2611.27353) | 本文提出WaXup方法，通过在权重空间中对专家模型进行线性插值（权重平均）来增强内省训练数据，从而显著提升大模型可训练内省方法的鲁棒性与泛化能力。 |
| [^89] | [PAVE: A Harness for Geometrically Consistent Road Network Modeling with Vision-Language Models](https://arxiv.org/abs/2611.27352) | 提出PAVE框架，通过感知、锚定、验证、编辑四个环节，利用视觉语言模型从卫星影像和街景影像中构建具有几何一致性的道路网络模型，克服了开放地图数据不完整、不准确以及VLM直接使用时几何一致性缺失的问题。 |
| [^90] | [GRACE: Controlled Evidence Interventions for Diagnosing and Improving Knowledge Integration in GraphRAG](https://arxiv.org/abs/2611.27350) | 该论文提出GRACE基准，通过对证据子图施加切断支持路径、添加干扰项、附加高度数枢纽和反事实答案替换四种受控干预，来诊断GraphRAG知识整合失败的根源，并提出QCFSE-SA方法在不同证据状态下改进模型表现。 |
| [^91] | [EcomWriteBench: A Real-World Benchmark for E-commerce Writing Tasks](https://arxiv.org/abs/2611.27349) | 该论文提出了 EcomWriteBench——一个涵盖 13 个类别、五种语言、69 个电商写作任务、共 3,861 个查询的真实世界基准，通过构建 37 项共享能力池并进行能力级评分，实现对大语言模型在细粒度电商写作能力上优劣势的系统评估。 |
| [^92] | [Learning Interaction Dynamics via Motion-Region Guidance for Robotic Video World Models](https://arxiv.org/abs/2611.27344) | MoReWorld通过共享运动区域引导机制，将机器人视频世界模型的动态学习与视频生成聚焦于交互相关区域，解决了幻觉接触和不一致机器人-物体交互等物理不合理问题。 |
| [^93] | [Unimodal Shortcuts and Joint Scene-Intent Dependence in Vision-Language Safety Representations](https://arxiv.org/abs/2611.27339) | 该研究审计发现现有情境安全基准均存在单模态捷径（仅凭图像即可高精度预测安全标签），并构建了一个将场景与意图交叉、排除单模态规则的受控评估，从而证明冻结视觉-语言模型的安全表示确实联合编码了场景与用户意图。 |
| [^94] | [GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning](https://arxiv.org/abs/2611.27330) | 该论文提出GGDriver方法，通过让每个候选轨迹根据自身路径动态选择并关注相关的视觉证据进行细化与评分，从而提升端到端自动驾驶规划的性能。 |
| [^95] | [H-MSAE: Hierarchical Sparse Concept Coverage for Efficient and Diverse Instruction Data Selection](https://arxiv.org/abs/2611.27329) | 提出H-MSAE框架，利用Matryoshka稀疏自编码器将多分辨率概念组织成从粗到细的层次结构，并通过质量加权的边际收益递减覆盖机制，高效地选择高质量、多样化的指令数据子集，避免了传统方法对语义标签标注和图结构的依赖。 |
| [^96] | [Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes](https://arxiv.org/abs/2611.27326) | 该论文提出了一种无损的中间推理压缩框架，通过BPE学习由PUA词元构成的压缩词表并扩展LLM词表后微调模型，在保持可解释性和可审计性的同时，平均减少14.7%的生成词元，推理轨迹长度最多缩短27.1%。 |
| [^97] | [Do Language Models Know Which Words Steer Their Reasoning?](https://arxiv.org/abs/2611.27317) | 该论文发现了“幻影上下文影响”现象——语言模型的推理会被语义无关的词不成比例地引导，而模型自身却意识不到这种依赖关系（生成-判断矛盾），并提出“在线KV抑制”方法来缓解这一问题。 |
| [^98] | [A Framework for AI-Ready Data in Earth and Environmental Science](https://arxiv.org/abs/2611.27313) | 该论文提出了一个社区驱动的实践性框架，用于评估和提升地球与环境科学研究中数据的“AI就绪度”，弥补了FAIR原则在AI模型训练、评估与部署场景下数据要求的不足。 |
| [^99] | [$\gamma$-Scape: An Agent-Centric World for Mobile Embodied Intelligence](https://arxiv.org/abs/2611.27306) | 该论文提出了 $\gamma$-Scape，一个以智能体为中心的移动具身智能仿真与基准平台，使智能体能在持久物理世界中组合运动工具与习得技能完成导航与操作结合的完整任务，并通过智能体式采集框架自主生成接触丰富的交互示范数据。 |
| [^100] | [Can Reasoning and Image Description Enhance Yes/No Visual Mathematical Understanding?](https://arxiv.org/abs/2611.27302) | 本文通过配对评估发现，思维链等显式推理能显著提升视觉语言模型在是/否型视觉数学问题上的表现，而文本图像描述反而可能损害性能。 |
| [^101] | [HookEvo: A Framework for Coordinated Plugin Evolution on Hook-Enabled Agent Runtimes](https://arxiv.org/abs/2611.27294) | 提出了HookEvo框架，利用智能体运行时的生命周期钩子，在每轮迭代中协同进化可执行插件处理器与共享状态，无需改动冻结的执行器即可让Qwen3-4B在ALFWorld、DBBench等多个基准上取得领先成绩。 |
| [^102] | [EgoLAM: Learning Latent Actions from Hand and Head Motion](https://arxiv.org/abs/2611.27290) | EgoLAM 通过用第一人称视频中实测的腕部运动监督潜在动作编码的学习，同时将头部运动（相机运动）分离并单独建模，使机器人策略能从自我中心视频中学习真正代表手部操作的潜在动作。 |
| [^103] | [Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization](https://arxiv.org/abs/2611.27288) | 提出条件对比正则化方法，在扩散模型强化学习后训练中保留预训练CFG的几何结构，作为诊断奖励黑客行为的有效指标，从而实现奖励提升与整体生成质量之间更稳定的权衡。 |
| [^104] | [Belief-State Management Meets Rubric Rewards for Agentic Forecasting](https://arxiv.org/abs/2611.27286) | 该论文提出一个智能体事件预测训练框架，通过主动信念状态管理（PBM）将冗长上下文压缩为紧凑信念状态，并引入五维不依赖结果的SEIPH评分标准提供过程质量奖励，从而解决了LLM智能体预测中上下文累积和奖励信号嘈杂的两大难题。 |
| [^105] | [DermKG: A Concept-Centric Knowledge Graph for Interpretable Skin Disease Diagnosis from Dermatological Morphology](https://arxiv.org/abs/2611.27283) | 本文构建了首个以概念为中心的皮肤病知识图谱DermKG（3,452个节点、19,365条边，涵盖疾病概念特征、共现关系、诊断层级和临床规则），首次量化了形态学概念所承载的诊断信息上限（疾病38.9%、九大类别77.6%），并据此提出图神经网络DermKG-Net以实现可解释的皮肤病诊断。 |
| [^106] | [EAGLE: Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding](https://arxiv.org/abs/2611.27264) | EAGLE提出了一种即插即用框架，通过筛选可靠的视觉定位注意力头、实体感知区域提取，以及分层证据构建与问题引导的视图组合选择，在不改变冻结多模态大模型参数的情况下显著增强其对高分辨率图像的细粒度理解能力。 |
| [^107] | [No Perfect Guardrail & No Perfect Guard: On the Decidability of Output Policies for Large Language Models](https://arxiv.org/abs/2611.27260) | 本文利用可计算性理论证明：只有当大语言模型的输出策略可判定时，才存在能完美区分禁止与允许输出的可计算护栏；对于不可判定的策略，任何护栏都必然误分类无穷多字符串，因此护栏不能免除人类自身的审查与责任。 |
| [^108] | [Proactive Defense Benchmark against Deepfake Generation](https://arxiv.org/abs/2611.27253) | 该论文提出了首个针对深度伪造主动防御的综合评估基准，揭示了保真度与身份指标的矛盾解读以及白盒峰值性能背后的过拟合问题，并建立严格标准以推动真正可泛化防护方法的发展。 |
| [^109] | [CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR](https://arxiv.org/abs/2611.27227) | 该论文将 token 对结果的影响形式化为“分叉值”，并利用分叉概率随响应内熵排名呈指数衰减的实证发现，提出 CreditFilter——通过熵排名校准 token 权重、在响应内部重新分配信用，从而改进免评论家的可验证奖励强化学习（RLVR）。 |
| [^110] | [LiDAR Signals for Robot Locomotion: Spherical Range Images Enable Robust Sim-to-Real Transfer](https://arxiv.org/abs/2611.27223) | 该论文提出一种基于球面距离图像编码的激光雷达感知管线，通过传感器级和表示级的双重对齐实现仿真与现实间的感知一致性，使腿式机器人无需教师-学生蒸馏即可实现鲁棒的仿真到现实迁移部署。 |
| [^111] | [Bel: Step-Level Policy Improvement of Visual Agents Without Training](https://arxiv.org/abs/2611.27222) | Bel提出了一种无需训练的视觉智能体步级策略改进方法，通过将每个候选动作绘制在其作用的图像上，让决策模型基于视觉判断从冻结策略的采样中选出更优动作。 |
| [^112] | [NavTTT: Adaptive Parametric Memory via Hierarchical Test-Time Training for Vision-and-Language Navigation](https://arxiv.org/abs/2611.27219) | 提出NavTTT，一种导航条件化的分层测试时训练方法，通过自适应记忆形成与分层记忆整合，利用全局导航上下文优先保留信息性视觉证据，解决长导航过程中冗余观测干扰长期记忆保留的问题。 |
| [^113] | [Correct Precedents Induce Policy Violations in On-Device LLM Agents](https://arxiv.org/abs/2611.27216) | 该论文首次发现“正确先例泛化”这一新型安全故障——端侧LLM代理在正确放行一个解除限制的条目后，会将该决策泛化迁移到仍受限制的后续条目上，导致即使整个会话历史中没有任何错误步骤也可能以策略违规告终，该问题在来自九个开发者的13个开源模型中普遍存在，且单个正确先例即可将最严重模型的安全率削减约一半。 |
| [^114] | [Neural Defend Ranks #1 in Global Audio Deepfake Detection Benchmark - ARCAS on the Podonos Independent Leaderboard](https://arxiv.org/abs/2611.27215) | Neural Defend 的 ARCAS 基础模型（290M 与 90M）在 Podonos 独立音频深度伪造检测基准上并列第一，以统计学上显著的优势超越 Resemble AI、Hiya、Pindrop、Hive 等所有其他商用及开源检测系统。 |
| [^115] | [EAF-TTS: Rethinking Codec-Latent Coordinates for Flow TTS](https://arxiv.org/abs/2611.27214) | EAF-TTS提出将完整的ZCA坐标协议（包括训练与采样坐标、诱导损失度量及终端扰动）应用于声学潜在空间，重新调整编解码器潜在坐标的几何结构，使其更适合从各向同性高斯噪声进行流式TTS生成。 |
| [^116] | [Explicit Style Extraction for Personalized Long-Form Text Generation](https://arxiv.org/abs/2611.27211) | 该论文提出StylePrism风格分类体系（28个风格维度）与SP-Refine推理时方法，将风格抽取变成显式步骤，在LongLaMP三个长文本任务上以平均10.2%的提升显著超越检索增强基线。 |
| [^117] | [Parameter-Efficient Adaptation of Qwen3-VL for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2611.27208) | 该研究通过 4 比特量化加秩-16 LoRA 微调，将紧凑的 Qwen3-VL-4B 高效适配为手写数学表达式识别模型，在 CROHME 测试集上以极小的参数开销实现了 43.74% 的精确匹配率，远超传统 ViT-Transformer 基线。 |
| [^118] | [LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment](https://arxiv.org/abs/2611.27206) | 该论文提出统一生理预训练框架LingSante-Pulse，通过随机特征预测、跨模态对齐与掩码波形重建学习通用生理信号表示，在涵盖PPG、ECG、EEG和EMG的38个任务配置的每个领域均取得最佳排名。 |
| [^119] | [DepthMSA: Combinatorial Training and Budget-Aware Depth Allocation for Multimodal Sentiment Analysis](https://arxiv.org/abs/2611.27198) | 该论文提出 DepthMSA 框架，通过组合式随时训练与预算感知的深度分配，在单个多模态情感分析模型中为不同模态灵活分配异构编码深度，从而在有限计算预算下保持预测性能。 |
| [^120] | [EDAGoat: An EDA-Aware Agent for Project-Level GPU Optimization](https://arxiv.org/abs/2611.27196) | 提出了首个EDA感知的GPU优化智能体EDAGoat，通过四种跨项目GPU优化模式指导代码编辑、并以强制验证模式确保修改满足EDA正确性与设计质量约束，实现了项目级GPU EDA工具优化的自动化。 |
| [^121] | [A Frugal AI Framework for India: Responsible and Right-sized AI](https://arxiv.org/abs/2611.27189) | 本文提出“负责任节俭AI”框架，主张印度既不应照搬中美重资本的基础模型路线，也不应仅作外国AI的下游采用者，而应先明确功能需求并判断是否需要AI，再选择以最少资源满足需求的最小充分“规模适度”模型，同时保持鲁棒性与负责任治理。 |
| [^122] | [What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations](https://arxiv.org/abs/2611.27186) | 该论文质疑LLM模拟匹配人类干预效应所能确立的有效性，发现大量“通过”实为无信息量的零参考通过，并提出零参考基准与计划精度检验来严格评估LLM模拟的真实响应能力。 |
| [^123] | [GEG-SAM: Gaze Exploration-Guided Segment Anything Model for Weakly Supervised Polyp Segmentation](https://arxiv.org/abs/2611.27183) | 该论文提出GEG-SAM框架，通过显式建模内镜医生注视点的时空结构关系及其所关注区域的视觉语义，实现弱监督的息肉分割。 |
| [^124] | [Learning Adaptive Defense Strategies: Fine-Tuned LLMs for Network Security Games under Network Mobility](https://arxiv.org/abs/2611.27180) | 本研究构建了带有求解器均衡标签的270个蜜罐分配安全博弈数据集，首次系统评估了LoRA微调的大语言模型能否在网络移动性持续改变攻击图的动态场景下，充当无需求解器的在线博弈均衡计算替代方案。 |
| [^125] | [Felid: A Flexible and Efficient Design for Transformer Fine-Tuning over Encrypted Data](https://arxiv.org/abs/2611.27175) | 该论文提出Felid，一种在加密数据上进行Transformer微调的灵活高效设计，通过新颖的打包策略、面向特定操作的矩阵求值算法以及函数自举技术，从密文操作层面显著减少了密文旋转、重线性化和自举的开销。 |
| [^126] | [From Acting to Learning: Experience Selection toward Continually Learning Agents](https://arxiv.org/abs/2611.27159) | 该论文将经验选择作为持续学习智能体的关键组成部分，通过大规模对照实验证明，优化器感知的梯度选择方法在学生智能体任务成功率上显著优于全局随机选择和高 NLL 选择两种基线。 |
| [^127] | [Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations](https://arxiv.org/abs/2611.27156) | 该论文提出IO-AI协同演化框架，通过让基于真实信息行动行为构建、具备互动与规避能力的模拟智能体与检测器轮流对抗训练，使检测器对自适应规避智能体的漏检率降低了65%。 |
| [^128] | [Where Structure Enters a World Model: From Predictive Fit to Physical Rollout](https://arxiv.org/abs/2611.27149) | 该论文系统研究了如何将物理结构（守恒量监督、对称性感知表示、基于能量的动力学头等）融入预测性世界模型，证明这些物理约束能显著提升模型在长时间范围内自主推演的物理一致性和轨迹忠实度。 |
| [^129] | [Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2611.27148) | 该研究发现在可验证奖励强化学习训练中，LoRA会随秩增大出现性能骤降的“秩坍缩”现象，而DoRA和QuanTA等结构化适配方法凭借更优的更新几何结构，仅更新不到0.6%的参数即可达到与全参数微调相当的效果，表明更新的结构比规模更关键。 |
| [^130] | [How Does SAM-3 Adapt? Capacity, Update Geometry, and Domain Transfer](https://arxiv.org/abs/2611.27142) | 本研究系统比较了五种微调方法在五个分割领域的表现，发现参数高效微调可媲美全量微调，且其性能由更新谱几何特性而非参数数量或秩决定。 |
| [^131] | [Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material](https://arxiv.org/abs/2611.27139) | 该论文提出“角色完整性”概念并构建RI-Bench行为基准，借助同行评审、上诉裁决和议会调查三种制度层级的结构与公开记录，衡量多智能体LLM系统中高级智能体是否真正履行了权衡下级报告的职责，而非仅仅进行简单汇总。 |
| [^132] | [Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain](https://arxiv.org/abs/2611.27138) | 本文构建了分级依赖条件下的精确后验参照基准，揭示出大语言模型对“每个来源只计一次”指令的遵从性与实际后验精度增益相互独立——完全遵守规则的模型可能损失精度，而不遵守规则的模型反而可能获得增益。 |
| [^133] | [MAPES: Multi-Value Alignment Pipeline via Expert Specialization](https://arxiv.org/abs/2611.27133) | MAPES提出了一种多价值对齐新范式，通过将监督分解为隔离参数子空间中训练的价值观专用专家，再经偏好感知合并与轻量级协调重组，在流水线每个阶段将冲突解决作为首要关注点，从而在多个基准上超越DPO变体和多目标基线。 |
| [^134] | [When Students Fail to Learn from Teachers: Learning Dynamics of On-Policy Distillation](https://arxiv.org/abs/2611.27125) | 该论文从学习动力学视角揭示了在线策略蒸馏（OPD）训练不稳定（如重复生成、响应过长）背后的失败机制，识别出OPD更新无法纠正师生失配的关键状态，并用学生在教师top-p集合上的概率质量对其进行刻画。 |
| [^135] | [Portable Active Recovery under Timed and Position-Dependent Visual Outages](https://arxiv.org/abs/2611.27121) | 提出了一个无需重训练、可移植的主动恢复模块，使无外部定位的双目视觉-惯性四旋翼无人机能够区分并正确应对定时型与位置相关两类视觉中断，通过协调检测、制动、后退探测与重连等行为实现最坏条件下的主动可观测性恢复。 |
| [^136] | [CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems?](https://arxiv.org/abs/2611.27116) | 该论文提出CoEvoBench——首个同时评估LLM作为研究者联合协同进化智能体的模型与框架（harness）的基准，在24小时预算和五个智能体领域中衡量这一此前未被测量的递归联合决策能力。 |
| [^137] | [Schema-Validated LLM Workflows: A Benchmark Protocol and Comparative Evaluation Across Six Python Implementations](https://arxiv.org/abs/2611.27106) | 该论文提出了一个可复现的基准测试协议与公开可执行工件，在五个业务工作流任务上对比六种Python实现，将LLM输出的模式有效性与任务正确性分离评估，并验证了两者可分离、结构化输出绑定路径异构、且失败集中于特定条件-任务组合这三大假设。 |
| [^138] | [One state, many questions: Opsis gives a frontier model typed eyes for pathology](https://arxiv.org/abs/2611.27102) | Opsis提出了一种构建在冻结病理编码器上的类型化问题接口，让前沿语言模型能够对仅编码一次的病理切片主动提问，在运行时指定算子、参数和候选答案并获得概率化结果，从而解决了语言模型无法处理大规模病理切片且固定描述性能低下的问题。 |
| [^139] | [Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency](https://arxiv.org/abs/2611.27100) | 该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。 |
| [^140] | [Epistemic Guard: Out-of-Process MCP Receipt Proxying, Post-Turn Prose Supervision, and Provenance-Gated Memory for Tool-Using Autonomous Agents](https://arxiv.org/abs/2611.27095) | 提出了开源库epistemic-guard，通过进程外MCP回执代理、回合后文本监督与来源门控记忆三大机制，在零GPU显存开销下解决工具使用型语言模型智能体的动作伪造、来源崩溃与回声放大三大运行时验证缺口。 |
| [^141] | [Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation](https://arxiv.org/abs/2611.27094) | 提出一种以无LLM可控工作流为对照臂的分阶段架构与归因协议，只在智能体发现基线流程无法报告的缺陷时才记功，从而在银行模型验证中将智能体的真实贡献与自动化本身带来的收益区分开来。 |
| [^142] | [Exercised, Rarely Recovered: What Public Reports Say About Recovery in 50 AI Agent Harnesses and Runtimes](https://arxiv.org/abs/2611.27090) | 该研究在预设协议下对50个AI智能体框架与运行时的八种恢复机制进行系统评估，发现恢复机制虽在实践中被广泛使用，但128份真实使用报告中仅19份实现完全恢复，大部分以部分恢复或失败告终。 |
| [^143] | [Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation](https://arxiv.org/abs/2611.27089) | 在空间推理任务上，权重空间编程带来的提升有限，而仅用0.09%可训练参数、以解题示例为监督训练的秩1 LoRA适配器可近乎完美地解决参照系与StepGame任务，并对未见模板和更高难度展现一定泛化能力。 |
| [^144] | [Fooling the Mind Twice: Human Detection of Audiovisual Deepfakes Under Repeated Exposure and Corrective Feedback](https://arxiv.org/abs/2611.27080) | 研究发现，重复暴露并结合纠正反馈能显著提高人类对视听深度伪造视频的检测准确率（FakeAVCeleb数据集上平均提升4.68个百分点），表明纠正反馈是改善人类深度伪造识别判断的有效手段。 |
| [^145] | [Weight-space programming and mechanistic interpretability for model editing in particle physics](https://arxiv.org/abs/2611.27079) | 该论文通过稀疏自编码器识别潜在方向的移除和语言模型智能体的免梯度权重编辑两种方法，成功修复了粒子物理基础模型中因预训练数据问题导致的违反红外安全性缺陷，且对模型其余功能几乎无影响，所需修改的参数远少于微调。 |
| [^146] | [Decision Models Are General-Purpose Process Reward Models](https://arxiv.org/abs/2611.27068) | DM-PRM无需步骤标签、微调或评论家，零样本地将通用的是非问答决策模型转化为通用过程奖励模型，仅通过每步一个问题即可显著提升语言智能体在家庭任务、网络购物等多种环境和多种RL方法上的表现。 |
| [^147] | [Dynamic Token Merging for Efficient Vision-Language Models](https://arxiv.org/abs/2611.27064) | 该论文将 MrT5 中的动态标记合并（DTM）推广为通用的标记删除框架，扩展至视觉模态与仅解码器模型，通过自适应删除视觉标记并压缩键值缓存来显著降低视觉-语言模型的推理成本。 |
| [^148] | [PACE: Policy Augmentation and Compensation With Explicit Disturbance Estimation for Robust Locomotion](https://arxiv.org/abs/2611.27061) | PACE框架通过非线性观测器在线显式估计关节空间扰动（涵盖负载变化、质心偏移、执行器退化与外力），并将估计结果反馈到策略和控制器中进行前馈补偿，从而缓解域随机化在跟踪精度与泛化能力之间的折中，提升四足机器人运动的鲁棒性。 |
| [^149] | [Learning 3D MRI Representations with Generative Adversarial Networks](https://arxiv.org/abs/2611.27057) | 该论文提出了R2RG方法，通过利用3D卷积神经网络的架构偏置并结合生成对抗网络优化的重建任务，在小规模3D MRI数据集上实现了保留空间3D信息的有效表示学习。 |
| [^150] | [A Continuous Relaxation of Quantization-Aware Training](https://arxiv.org/abs/2611.27056) | 提出RelaxQAT，通过在训练过程中逐渐连续地改变位宽来松弛感知量化训练，从而避免低位量化中损失曲面的鞍点和平坦区域问题，显著提升量化模型性能。 |
| [^151] | [Actions Speak Louder than Words: Stateful Privacy Sniffing of LLM Agents via Behavioral Covert Channels in Web Interactions](https://arxiv.org/abs/2611.27046) | 这篇论文提出SNOOP系统，利用LLM智能体在Web交互中的良性决策行为本身构成的行为隐蔽信道，无需任何显式数据传输即可主动嗅探并恢复智能体的细粒度私有上下文，揭示了一种现有攻击和防御手段均未覆盖的新型隐私攻击面。 |
| [^152] | [Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification](https://arxiv.org/abs/2611.27043) | 提出HypSEE框架，通过构建游走诱导与嵌入衍生超图捕获层次感知的局部和长程高阶关系，并利用超图结构熵引导的层次超图池化方法实现半监督图分类中的层次表示学习。 |
| [^153] | [Disentangling Verbalized Evaluation Awareness and Refusal through Neuron-Level Intervention](https://arxiv.org/abs/2611.27032) | 该研究通过识别并干预约5,000个“VEA神经元”，将模型的言语化评估意识率从12.6%–51.8%降至最高1.4%，从而在神经元层面揭示并解耦了评估意识与拒绝行为之间的关联机制。 |
| [^154] | [Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima](https://arxiv.org/abs/2611.27018) | 本文借助代数拓扑工具（整数立方填充与同调障碍类）证明了双隐藏层整数参数ReLU网络精确表示最大值函数的最小正输出乘子——六输入为2、七输入为2或6，并刻画了七输入情形中剩余的算术障碍。 |
| [^155] | [Same Facts, Different Decision: Fact-Invariant Manipulation and Provenance-Based Defense in Typed Decision Models](https://arxiv.org/abs/2611.27016) | 论文提出LEVER攻击，利用论证理论中的四个杠杆在不添加任何新事实的情况下操纵类型化决策模型做出错误决策，并提出Provenance Gate防御机制，仅当模型在无外部评论时能复现该决策才予以接受。 |
| [^156] | [Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations](https://arxiv.org/abs/2611.27015) | Ruby是一个利用结构基础模型Pearl的中间表示、无需显式结合姿态的鲁棒通用结合亲和力预测器，通过专有数据与公开数据集GenesisPotency-2026训练，在药物发现的各个阶段均实现了最先进的药效预测。 |
| [^157] | [Many Plans at Once: Diverse Top-Quality Planning in Continuous Time](https://arxiv.org/abs/2611.27012) | 该论文提出MPAO方法，通过采样构造上满足边界条件的神经网络、训练共享修正并自适应调整采样策略，在连续时间中同时生成多个低成本且相互不同的高质量规划方案，为遇到未建模约束或偏好变化时的快速适应提供备选方案与热启动。 |
| [^158] | [FoRe: A Force-Guided Framework for Autonomous Grasp Recovery in Vision-Language-Action Models](https://arxiv.org/abs/2611.27011) | 该论文提出FoRe框架，利用力反馈将演示数据划分为接近-抓取与搬运-放置阶段分别训练策略，并通过局部强化学习从失败后状态优化专用重试策略，实现视觉-语言-动作模型从抓取失败中的自主恢复。 |
| [^159] | [Scalable Evaluation of Medical AI using Evidence-based Practice Recommendations](https://arxiv.org/abs/2611.27007) | 该论文提出一个自动化流程，从BMJ Best Practice等循证临床实践推荐中提取并验证结构化临床标准与决策树，大规模生成3,200个涵盖鉴别诊断、检查、治疗与并发症的临床场景，实现了对医学AI在跨地区、跨指南版本条件下开放式临床管理能力的可扩展评估。 |
| [^160] | [JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model](https://arxiv.org/abs/2611.26999) | JevRisk提出了一个基于语言决策模型的零样本金融风险评估系统，通过遵循金融监管指南用代码计算精确风险指标，摆脱了对人工标签的依赖，并能在不同客户和风险类型间泛化。 |
| [^161] | [Reward Integrity in Adversarial Reinforcement Learning for LLM Guardrail Red-Teaming](https://arxiv.org/abs/2611.26998) | 该研究发现在金融服务LLM护栏红队测试的对抗性强化学习循环中，奖励评判可能出现虚假胜利（如攻击遗漏规定PII或评判者误判未违规响应），并通过对比多种决策模型验证了采用专门决策模型作为奖励评判者的必要性。 |
| [^162] | [From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness](https://arxiv.org/abs/2611.26988) | 该论文将信用证审核要求逐条转化为可执行的原子条件并结合确定性验证器实现可追溯审核，揭示了仅以最终决定准确率衡量审核的误导性，并在数据保密条件下通过仅用合成数据微调的本地9B解析器达到93%的原子要求检测率。 |
| [^163] | [Privacy Protection in Self-Evolving Personal Agents: Persistent Memory and Cross-Carrier Transfer](https://arxiv.org/abs/2611.26983) | 本综述系统比较了个人智能体持久记忆的隐私保护机制（准入、维护、释放与撤回），通过案例场景与诊断实验量化了标签保护和记忆删除的效用代价，并揭示了记忆谱系缺失导致的隐私清理隐患。 |
| [^164] | [Contexture: Runtime Context Eviction for High-Throughput Agent Serving](https://arxiv.org/abs/2611.26980) | Contexture提出了一种与框架和模型无关的运行时上下文淘汰架构，通过将工具执行信息与全局KV缓存压力可见性相结合，识别出64%可删除的过时历史词元，在不修改框架逻辑或模型执行的情况下降低了KV缓存需求并提升智能体服务的吞吐量。 |
| [^165] | [MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs](https://arxiv.org/abs/2611.26976) | MixFold通过将输出通道分组与移动NPU矩阵单元瓦片对齐，无需解码时扩展权重即可原生执行3比特和对称2比特混合精度大语言模型推理，在骁龙Hexagon NPU上实现比llama.cpp相当或更高的准确率与更快的解码速度。 |
| [^166] | [Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs](https://arxiv.org/abs/2611.26975) | ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。 |
| [^167] | [When Perception Collides with Priors: Diagnosing Modal Polarization in Video Understanding Models](https://arxiv.org/abs/2611.26972) | 该论文揭示了多模态大语言模型在视频感知与先验常识冲突时会出现“模态极化”失败模式（分为认知惯性与感知惯性两种极端），并构建了基于反直觉场景的TheaterBench基准来系统诊断这一问题。 |
| [^168] | [Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs](https://arxiv.org/abs/2611.26968) | 论文提出FrameCheck插件，在不增加模型调用的情况下检测并约束LLM财务报告草稿中的“无依据框架叙述”，确保报告叙述忠实于输入事实。 |
| [^169] | [Keep the Process, Change the Supplier: Runtime Governance for Tool-Using AI Agents](https://arxiv.org/abs/2611.26966) | 提出开放流程控制平面（OPCP）作为执行边界，将工具型AI智能体的动作提议与业务授权、上下文披露和供应商选择分离，实现运行时治理并在故障恢复时避免跨供应商的重复执行。 |
| [^170] | [Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs](https://arxiv.org/abs/2611.26965) | 该研究揭示了当前大语言模型的儿童安全对齐严重依赖伤害类别提示线索——在缺乏此类线索的看似无害请求上未对齐率大幅上升（最高达80个百分点），而仅引入一个伤害相关词即可恢复安全行为，表明模型的儿童安全推理主要建立在词汇和语义关联之上而非真正的安全理解。 |
| [^171] | [Tracking and Limiting Unsupported Claim Propagation in Multi-Agent LLM Systems: A Provenance-Aware Framework](https://arxiv.org/abs/2611.26957) | 该论文提出了一个溯源感知框架，通过在多智能体LLM流水线中随论断一同传递其支持状态、来源、证据与传输历史，有效追踪并限制了无依据论断在智能体间传播时的语义漂移与无依据细节累积。 |
| [^172] | [CodeReviewBench: An Automated Benchmark for Evaluating the Software Factory](https://arxiv.org/abs/2611.26942) | 该论文提出CodeReviewBench，以“开发者揭示的偏好”（开发者是否实际采纳机器人的审查建议）作为客观验证信号，每日从GitHub事件流挖掘50万+条判定数据，构建了一个可持续自动更新、覆盖18个工具的代码审查基准。 |
| [^173] | [CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation](https://arxiv.org/abs/2611.26940) | 本文提出CENTRY概率框架，通过结构化潜变量模型将AI风险分解为一个捕捉AI-环境动态的快速交互子空间和三个编码数据、硬件与模型长期风险状态的慢速潜在子空间，从而实现对已部署AI系统全生命周期的持续风险估计、监测与调控。 |
| [^174] | [MG-EDA: Maturity-Gated Skew-Normal Estimation of Distribution Algorithms](https://arxiv.org/abs/2611.26938) | 该论文提出MG-EDA算法，通过种群成熟度统计量驱动的校准门控来调控均值偏移、协方差拉伸和偏度三项修正，解决了高斯分布估计算法在斜坡上因方差收缩而停滞的问题，且门控关闭时算法精确退化为标准EMNA。 |
| [^175] | [SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models](https://arxiv.org/abs/2611.26936) | 提出了首个通用的多智能体框架SecretAgent，通过LLM智能体迭代定位泄露隐私的视觉证据并自适应生成不可见扰动，同时借助外观审查智能体把关，在有效抵御视觉语言模型隐私推断的同时保持图像的视觉保真度。 |
| [^176] | [Self-similar Organization of Information in Language](https://arxiv.org/abs/2611.26935) | 本文提出了一种基于信息度量标度行为的预测信息组织理论，并通过在长文本中观察到不同长度尺度上信息度量的标度坍缩，证明了语言中预测信息具有自相似结构，这对理解语言的统计规律及大语言模型如何学习语言结构具有重要意义。 |
| [^177] | [Tokenize Moments, Not Frames: Tokenizer-Aware Budget Allocation for Training-Free Video LLM Acceleration](https://arxiv.org/abs/2611.26933) | 论文揭示了视频LLM加速中精度损失的根源在于token选择之前的分词阶段——长视频采样间隔超过约2秒相似窗口导致视觉token出现“双重曝光”混合，并提出免训练方法ESSENCE，通过对时刻而非帧进行分词、并依据采样间隔与分词器类别用闭式规则分配token预算，从而在不损失精度的情况下加速视频LLM。 |
| [^178] | [Partially canonical axes in large language model representations](https://arxiv.org/abs/2611.26920) | 该研究挑战了“大语言模型表征坐标轴是任意的”这一常见假设，发现在12个LLM中原生轴比随机旋转轴更能与人类大脑语言响应和可解释语言特征对齐，揭示LLM表征轴是“部分规范的”——既非完全任意也非完全规范。 |
| [^179] | [ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval](https://arxiv.org/abs/2611.26919) | ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。 |
| [^180] | [Self-Evolving Beyond Human-Crafted Experience for EDA Scripting](https://arxiv.org/abs/2611.26912) | ZhuLong-SE是一个以执行结果为依据的LLM智能体，通过将历史执行轨迹提炼为受真值锚定、证伪过滤、回滚和防泄漏检索等多重机制约束的三层记忆库，实现EDA脚本编写任务的跨任务自进化，从而超越依赖人工构建知识的局限。 |
| [^181] | [AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress](https://arxiv.org/abs/2611.26910) | AstronautBench基于游戏《星际拓荒》构建了一个新基准，通过知识、证据支持与任务进展三个互补视角，衡量权重固定的LLM智能体在世界不断重置的循环中凭借自身经验实现跨回合自我改进的能力。 |
| [^182] | [AI-Assisted Decision Support for Formula 1 Pit-Stop Strategy: Deep Reinforcement Learning and Mathematical Programming](https://arxiv.org/abs/2611.26909) | 该研究构建了基于历史比赛数据校准的F1进站策略模拟器，用五种深度强化学习算法训练策略智能体并与QMIP数学规划模型对比，发现PPO及其循环变体是在真实比赛不确定性和竞争环境下唯一能超越确定性最优规划基线的DRL方法。 |
| [^183] | [Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers](https://arxiv.org/abs/2611.26906) | 本文提出BudgetPV，一种预算感知的预测-验证智能体系统，利用跨求解器预测器在异构LLM求解器之间动态分配计算资源，并根据证明进程的不同阶段（新子目标的求解器选择与已尝试子目标的算力追加）调整策略，从而在保证证明成功率的同时最小化定理证明的计算成本。 |
| [^184] | [Towards Robust Partial-Edit Deepfake Localization: An edit and Generator Conditioned Analysis](https://arxiv.org/abs/2611.26903) | 本文首次对部分编辑深度伪造定位进行了基于编辑类型与生成器条件的细粒度分析，发现删除操作——尤其是SSR-Speech生成的删除——是定位中最具挑战性的条件。 |
| [^185] | [Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs](https://arxiv.org/abs/2611.26902) | 提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。 |
| [^186] | [Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing](https://arxiv.org/abs/2611.26900) | 该论文提出了首个大规模归因风格基准数据集ASTD及配套评估指标，包含42,000个真实事件和七种心理归因风格的归因标注，实现了归因风格的可扩展评估与认知重构。 |
| [^187] | [ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments](https://arxiv.org/abs/2611.26897) | ProbeAgent是一个自动化诊断框架，通过构建经过验证的可执行测试、生成保留疑似失败原因的变体任务并观察错误是否复现，来系统性发现、验证并归类智能体的共有失败模式。 |
| [^188] | [Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric](https://arxiv.org/abs/2611.26881) | 研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。 |
| [^189] | [Beyond Demographic Discrimination: Benchmarking Normative-Descriptive Calibration in LLMs](https://arxiv.org/abs/2611.26870) | 该论文提出“规范-描述校准”这一新概念及包含697个匹配对场景和三个指标（NER、NLR、NDS）的NDCB基准，首次系统衡量大语言模型在“预测结果”与“决策结果”两种任务框架下对相同候选者处理是否一致。 |
| [^190] | [Coherent Moment Propagation: Epistemic Uncertainty for Pre-trained Robot Policies](https://arxiv.org/abs/2611.26855) | 本文证明BayesDiff在流匹配与扩散ODE中会因总方差衰减而严重低估认知不确定性，并提出相干矩传播方法，为预训练机器人策略提供可信的认知不确定性量化。 |
| [^191] | [LoCo-WAM: Test-Time Training as Long Context for World-Action Models](https://arxiv.org/abs/2611.26853) | LoCo-WAM 通过测试时训练利用新观测帧更新视频专家中的低秩适配器，使世界-动作模型在不延长观测窗口、无需专家动作标签、额外数据或元训练的情况下从更长的经验流中学习，从而在四个仿真基准上显著提升任务成功率。 |
| [^192] | [Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion](https://arxiv.org/abs/2611.26849) | 该论文构建了多中心多模态步态数据集Scoliosis5K，并提出片段条件化的RGB-轮廓融合架构，用于脊柱侧凸相关步态的参与者级分类。 |
| [^193] | [Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models](https://arxiv.org/abs/2611.26835) | 本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。 |
| [^194] | [Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports](https://arxiv.org/abs/2611.26833) | 该研究提出在训练后对领域适配与任务特定的LoRA适配器进行缩放、并可用互补权重将二者组合，从而在不损害指令遵循能力的前提下提升大语言模型对葡萄牙语钻井报告问答任务的性能。 |
| [^195] | [Realistic Robustness Evaluation of Tabular Data Driven Models](https://arxiv.org/abs/2611.26818) | 本文提出了AURORA，一个模块化开源工具，通过在统一框架中整合多种对抗扰动方法和评估指标，为表格数据分类的机器学习模型提供系统化的鲁棒性评估。 |
| [^196] | [Search From the Start: Search-Native Pretraining for Language Models](https://arxiv.org/abs/2611.26809) | 提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。 |
| [^197] | [MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution](https://arxiv.org/abs/2611.26806) | 该论文提出MedicalRSI框架，通过临床模型-框架协同进化使医疗智能体能够同时改进自身及其改进策略，从而实现递归自我改进，避免同类错误在不同患者间反复出现。 |
| [^198] | [Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory](https://arxiv.org/abs/2611.26797) | 本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。 |
| [^199] | [Beyond Final Answers: Interpretable Tumor Immune Microenvironment Reasoning from H&E and mIF](https://arxiv.org/abs/2611.26794) | 该论文提出HERMES，利用同一切片的H&E与mIF图像提供免人工标注的细胞级真值，使多模态大语言模型的肿瘤免疫微环境推理获得步骤级可验证的可解释监督，而非仅验证最终答案。 |
| [^200] | [ChatAtom: A Multi-Agent LLM System for Molecular Dynamics Simulation and Visualization](https://arxiv.org/abs/2611.26782) | ChatAtom是一个多智能体大语言模型系统，可将自然语言请求自动转化为完整的分子动力学模拟工作流程，涵盖方案规划、脚本生成与审核、高性能计算任务分发以及结果分析与可视化，无需用户编写任何代码。 |
| [^201] | [What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference](https://arxiv.org/abs/2611.26776) | 该研究在XD-Violence数据集上进行受控比较，发现将感知模型观察结果整理为一次性阅读的扁平文本是语言模型代理审阅暴力视频最有效的方式，其表现与直接处理原始视频的单一视听模型相当，而消息传递和证据图接口效果较差。 |
| [^202] | [Windchime: an Audiovisual Installation with Audio-Language Model implementations](https://arxiv.org/abs/2611.26759) | Windchime 是一个视听装置，参观者通过语音输入，借助本地语音识别与音频-语言模型从艺术家语料库中检索音乐和交互式动画，实现了在艺术家、模型与参观者三方之间分配能动性的多模态具身交互体验。 |
| [^203] | [Self-verifiable LLM-driven science and why it needs human researchers](https://arxiv.org/abs/2611.26754) | 本文提出把数学领域中形式化证明自动验证的思路（如Lean 4、Coq）推广到非数学科学，设计一种能让LLM科学推理实现自我验证的AI智能体框架，从而把AI辅助科研的瓶颈从“生成结果”转移到消除人工复核成本上。 |
| [^204] | [Depth-Linked Residual Tokens for Multimodal Retrieval and Generation](https://arxiv.org/abs/2611.26752) | 提出DepThink框架，在单次前向传播中通过轻量级可学习残差递归将一串残差标记的隐藏状态相互耦合，使其逐步捕捉互补信息，从而兼得迭代架构的逐步精炼能力与并行单次推理的高效性，服务于多模态检索与生成。 |
| [^205] | [MomentMem: Memorizing Key Moments for Robot Manipulation](https://arxiv.org/abs/2611.26745) | 提出了MomentMem分层架构，由基于VLM的记忆器选择关键帧、执行器基于所选记忆预测动作，并引入VMBench基准来评估机器人长时程操作中的视觉时刻记忆能力。 |
| [^206] | [INCISE: Training-Free In-Context Segmentation of Surgical Videos](https://arxiv.org/abs/2611.26734) | INCISE是一个面向手术视频的免训练上下文分割框架，通过从多个手术视频中选择能共同覆盖目标视频局部外观的参考帧，仅需1.7%的标注帧即可实现分割，无需大规模专家标注和模型微调。 |
| [^207] | [TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator](https://arxiv.org/abs/2611.26722) | 该论文提出了包含约450万条经验证偏好对的大规模工具调用偏好数据集TOUMAII及其源无关的偏好对生成器，以缓解工具调用领域（尤其是多轮场景）偏好数据的稀缺，从而支持DPO训练出更可靠的工具调用能力。 |
| [^208] | [Coherence by Construction: Synthesizing CTI Knowledge Graphs and the Formal Limits of Coherence Validation](https://arxiv.org/abs/2611.26717) | 该论文提出“通过模拟进行生成”的新范式：语言模型智能体提出意图、确定性裁判依据显式世界模型裁决事件，使关系仅在对应事件发生时才被输出，从而将网络威胁情报知识图谱的一致性变为构造过程的固有属性，并探讨了一致性验证的形式化极限。 |
| [^209] | [Model the Annotation, Not the Noise: Semantic Annotation-Channel Prompt Learning against Noisy Labels](https://arxiv.org/abs/2611.26702) | 该论文提出SACP框架，从数据生成视角将潜在干净标签预测与实例相关的标注生成过程解耦，通过融合冻结VLM语义先验与轻量级视觉残差构建语义感知标注通道，并利用广义期望最大化与软后验进行优化，从而有效应对噪声标签问题。 |
| [^210] | [On the loss landscape of the linear neural network: beyond critical points](https://arxiv.org/abs/2611.26671) | 本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。 |
| [^211] | [Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams](https://arxiv.org/abs/2611.26663) | 在人机头脑风暴实验中，双AI团队虽提高了创意总量却大幅压缩了人类贡献，而单个“平均人类行为校准”的AI反而最能激发人类创意产出，表明AI伙伴数量与组建方式共同重塑了团队的生产力与多样性。 |
| [^212] | [Computing Weighted Set Covers with Heterogeneous Stochastic Oracles: Deadline-Constrained Token Complexity](https://arxiv.org/abs/2611.26656) | 该论文刻画了概率图灵机在截止期限和成功概率约束下求解加权集合覆盖问题的最小期望令牌成本，证明最优策略至多需要在两个预选的预言机批次序列之间随机化。 |
| [^213] | [VerifiQuant: Governance-by-Execution for Financial LLM Workflows](https://arxiv.org/abs/2611.26649) | VerifiQuant提出“执行即治理”框架，通过将金融逻辑封装为推理契约并用可执行诊断漏斗在计算前评估每个查询，使金融大模型能够结构化、可审计地区分计算、澄清与拒绝，从而在输入不支撑数值答案时实现更安全的行为。 |
| [^214] | [LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits](https://arxiv.org/abs/2611.26538) | 该论文揭示了现有交叉性偏差审计判据会把独立属性的加性效应误报为交互偏差（平均每个数据集误报66.1个子群体、真实交互检出率仅32%），并提出ADDIS方法，通过在所有属性上拟合加性模型建立基线，从而真正将交互效应与加性效应分离，准确审计交叉性子群体偏差。 |
| [^215] | [Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning](https://arxiv.org/abs/2611.26504) | 该论文受网格细胞对应后继表示特征向量的启发，提出用固定参考策略转移矩阵的特征向量作为后继特征的可复用基，使得长期回报预测具有封闭形式的系数，且无需后继特征网络或回放缓冲区，为持续强化学习提供了高效基础。 |
| [^216] | [Characterizing and Detecting Unfaithful Work Reports by Coding Agents](https://arxiv.org/abs/2611.25760) | 该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。 |
| [^217] | [STARK: Moving towards Personalized Agents That Truly Learn and Remember](https://arxiv.org/abs/2611.25705) | STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。 |
| [^218] | [Can an LLM Fix a Language Model Without Training It?](https://arxiv.org/abs/2611.25613) | 研究表明，LLM无需任何梯度训练，仅通过交互式检查模型预测、激活和权重统计并选择数值投影编辑，即可有效修复受损的Transformer语言模型，其表现可与黑盒搜索相当甚至大幅超越（如Pythia上达到65.62%准确率而黑盒搜索为0%）。 |
| [^219] | [Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots](https://arxiv.org/abs/2611.25254) | 本文提出PLANTA方法，利用OWL本体表示架构备选方案、PDDL定义任务目标，实现了机器人任务规划与架构重构的联合协同规划，并能根据执行中的变化进行重规划。 |
| [^220] | [MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation](https://arxiv.org/abs/2611.25105) | 提出MAWAM多智能体世界-动作模型，通过跨智能体注意力在单次采样中联合生成所有机器人的未来视角和可执行动作，在真实与仿真协同操作任务中大幅超越现有方法。 |
| [^221] | [Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation](https://arxiv.org/abs/2611.25086) | 该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。 |
| [^222] | [When Figures Disagree with Text: Evaluating Autonomous Visual Evidence Verification in Scientific Papers](https://arxiv.org/abs/2611.24353) | 该论文提出了一种更贴近真实审稿场景的自主图文一致性验证评估，要求多模态模型在无任何提示的情况下于完整论文中自行定位问题图表并诊断矛盾，结果显示尽管60.6%的审阅提及了缺陷图表，但仅3.3%真正诊断出了其中的不一致。 |
| [^223] | [MLE Agents: Autonomous ML Engineering in Large-scale Recommendation Systems](https://arxiv.org/abs/2611.23910) | 该论文构建了在大规模搜索排序系统中自主完成模型改进提出、实现、训练与评估的机器学习工程师（MLE）智能体，并揭示了工业级自动化面临的三大挑战：实验反馈耗时、离线指标难以预测在线效果、改进收益趋于饱和。 |
| [^224] | [Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates](https://arxiv.org/abs/2611.23651) | 本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。 |
| [^225] | [Idempotency Is Not Recovery: A Fault-Injection Study of Language-Model Agent Tool Actions](https://arxiv.org/abs/2611.23353) | 该论文通过系统的故障注入实验揭示“幂等性不等于恢复”，证明只有将授权、操作账本、目标端幂等性、对账与证据封存相结合的完整配置，才能将语言模型代理工具调用的重复副作用从基线的117/200降至0/200，并保障故障后的恢复能力。 |
| [^226] | [Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail](https://arxiv.org/abs/2611.22537) | 该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。 |
| [^227] | [RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents](https://arxiv.org/abs/2611.22113) | 该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。 |
| [^228] | [Cancellation-Free Constrained SquareCB for Safe Controller Selection](https://arxiv.org/abs/2611.21880) | 该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。 |
| [^229] | [Can Coding Agents Write and Transpile Unit Tests](https://arxiv.org/abs/2611.21797) | 本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。 |
| [^230] | [Regularizing Treatment Leakage for LLM-based Uplift Estimation](https://arxiv.org/abs/2611.20794) | 该论文提出LULR方法，通过在冻结LLM编码器的同时对因果头共享表示施加基于CLUB的互信息惩罚，抑制LLM编码文本中泄露处理信息的捷径，从而改进工业定价RCT场景下的CATE提升估计效果。 |
| [^231] | [Markets for Compute and Inference: Structure, Financing, and Risk Transfer](https://arxiv.org/abs/2611.20242) | 该论文研究了新兴的算力租赁与推理token市场中的对冲问题，发现交易所合约结算所依据的公开短期租金率与运营商实际收入来源的私人多年期合同价格相互脱节，且租金率与token价格属于不同的风险敞口，因此对租金率进行对冲无法覆盖推理成本的风险。 |
| [^232] | [Zero-Shot Weight-Space Programming for Prediction and Classification of Target Variables in Structured Datasets](https://arxiv.org/abs/2611.19292) | 该论文证明大语言模型可以在零样本条件下，仅凭展示的小数据集与指定网络架构，无需优化器、梯度或代码执行，就手写出前馈网络的全部权重和偏置，在从玩具问题、合成问题到经典基准和真实工程任务的测试中直接完成结构化数据的目标变量预测与分类。 |
| [^233] | [MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation](https://arxiv.org/abs/2611.19151) | 该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。 |
| [^234] | [SPARC: Evidence-guided adaptive model composition for materials discovery](https://arxiv.org/abs/2611.19050) | SPARC是一个面向材料发现闭环实验的自适应模型组合框架，通过折外预测评估竞争模型族、以稀疏后验聚合分配模型权重，并利用预测离散度与残差变异性定义信任度来调节探索策略。 |
| [^235] | [Alignment by Socialization: Learning Tradeoffs Under Selection](https://arxiv.org/abs/2611.18531) | 该论文提出“社会化”这一对齐机制：在进化选择压力下，学习型智能体会收敛到环境所奖励的任务完成与行为规范之间的权衡；当由人类提供奖励时，智能体的选择便能反映人类判断并经受住后续的选择压力。 |
| [^236] | [When evaluation becomes intervention in clinical AI](https://arxiv.org/abs/2611.18297) | 临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。 |
| [^237] | [AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs](https://arxiv.org/abs/2611.17621) | 提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。 |
| [^238] | [Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding](https://arxiv.org/abs/2611.13712) | 本文提出Encoder BPA方法，用潜在状态编码器替代传统K-means聚类，通过潜在比较、top-k聚合和一致性条件来检索和重用外部建议，从而提升了交互式强化学习早期的学习效率和建议利用率。 |
| [^239] | [SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning](https://arxiv.org/abs/2611.13454) | SafeRoute-MAS提出了一种仅利用集成前信息、覆盖率感知且风险校准的消息路由程序，在多智能体科学推理中通过选择性移除专家消息，在减少约12%–19%消息令牌的同时带来一定的准确率提升，但严格的统计验证显示其增益有限。 |
| [^240] | [Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models](https://arxiv.org/abs/2611.13223) | 本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。 |
| [^241] | [MARS: Multiscale Appearance-Residual Streams for Video Forgery Detection](https://arxiv.org/abs/2611.12508) | MARS提出了一种双路径视频伪造检测器，通过对冻结特征施加多尺度空间-时间高斯差分算子提取残差差异，并与自适应外观表示及跨片段门控机制相结合，从而在面部篡改和全生成视频两类深度伪造检测上实现更强的泛化能力。 |
| [^242] | [TS-GEAR: Taobao Shangou GEnerative Ad Retrieval](https://arxiv.org/abs/2611.12180) | 该论文提出TS-GEAR，面向即时配送场景的生成式广告检索，通过显式建模配送范围可达性、基于地理位置与当前查询条件化的用户兴趣表示以及集合级训练信号，解决了语义ID地理信号软化、兴趣估计偏差与候选冗余三大挑战。 |
| [^243] | [Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition](https://arxiv.org/abs/2611.10684) | 该论文提出在合并公开 LoRA 适配器前必须进行来源审计（基础模型匹配与非平凡增量范数检查）和无泄漏探测协议，并证明朴素合并会造成显著性能下降，而任务算术可将性能维持在未合并基础模型的标准误差范围内。 |
| [^244] | [SLA-Aware Task Offloading in the IoT Edge-Cloud Continuum: Class-Weighted Deep Reinforcement Learning Against an Exact Optimum](https://arxiv.org/abs/2611.10638) | 该研究通过四个可精确求解的基准系统证明：在物联网边云任务卸载中，类别盲的惩罚设计即使配合完美学习器也会导致成本高出最优解65%-118%，而修正后的按SLA类别加权的奖励设计可使PPO智能体性能逼近精确最优解。 |
| [^245] | [Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games](https://arxiv.org/abs/2611.09271) | 该论文首次系统考察大语言模型零样本权重空间编程的能力边界，在涵盖四个难度维度的双人游戏四级阶梯上零样本手工设定前馈网络的全部权重，构建并独立验证了十个网络，其中六个获得完整且非统计性的正确性证明。 |
| [^246] | [Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction](https://arxiv.org/abs/2611.07695) | 提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。 |
| [^247] | [Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI](https://arxiv.org/abs/2611.07048) | 该论文提出了一个比较达尔文式、拉马克式与智能体导向演化的理论框架，发现当学习足够有效时三种机制的适应速度依次加快，生成更多变体可加速适应，而最优搜索努力取决于学习能力、问题难度与实验成本。 |
| [^248] | [Unsatz: Evaluating Scientific Discovery Without a Given Ansatz](https://arxiv.org/abs/2611.03713) | 该论文提出了 Unsat​z 基准，首次要求 LLM 智能体在没有给定模型形式（拟设）的情况下自主构建拟设——在有限预算内设计实验并用统一建模语言编写候选模型程序——从而更真实、更完整地评估人工智能的科学发现能力。 |
| [^249] | [Tracing the Scientific Discovery Capacity of LLMs to Their Training Data](https://arxiv.org/abs/2611.03617) | 该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。 |
| [^250] | [LUCA: A Synthetic World for Experimental Reasoning](https://arxiv.org/abs/2611.03100) | 该论文提出LUCA——一个在完全已知真值下评估智能体实验推理过程的可配置合成世界基准，通过让智能体在有预算约束下学习未知调控系统并分别考察实验选择、预测、诊断和干预能力，发现前沿大语言模型在这些能力上显著分离，常常在误诊故障的情况下仍能成功拯救系统。 |
| [^251] | [Why Tabular Foundation Models Excel on Relational Data They Have Never Seen](https://arxiv.org/abs/2611.02936) | 本文通过多样化实验证实并深入揭示了单表表格基础模型在结合无参数关系编码器后，即使从未经过任何关系数据预训练，也能在关系数据库预测任务上表现出色的内在机制。 |
| [^252] | [dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache](https://arxiv.org/abs/2611.01648) | 提出dKV Forcing方法，为KV缓存写入分配独立的低秩残差参数，实现生成与记忆表征的解耦，并利用后续片段读取的梯度来优化缓存写入，从而改进自回归视频生成器的训练。 |
| [^253] | [WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining](https://arxiv.org/abs/2610.39278) | WorldFlow是一种生成式世界模型，它将3D运动参数化为与视频网格对齐的连续位移标记，并利用统一的Mixture-of-Transformers骨干网络在整流流目标下联合去噪视频与3D两种模态，从而将显式几何动力学与视觉演化耦合，释放预训练视频模型中蕴含的物理常识。 |
| [^254] | [Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs](https://arxiv.org/abs/2610.35985) | GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。 |
| [^255] | [Entity Matching using Decision Models](https://arxiv.org/abs/2610.34626) | 本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。 |
| [^256] | [Benchmarks Are Surprisingly Compaction-Friendly](https://arxiv.org/abs/2610.31778) | 该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。 |
| [^257] | [LoRA Superposition across Pretrained Models](https://arxiv.org/abs/2610.30691) | 该论文提出“LoRA叠加”概念，证明在不同预训练骨干网络上针对同一任务独立训练的LoRA可以直接相加并无修改地应用于所有骨干网络，其效果接近单独训练的LoRA，平均准确率提升高达30.21个百分点。 |
| [^258] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^259] | [Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents](https://arxiv.org/abs/2610.25760) | 该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。 |
| [^260] | [CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts](https://arxiv.org/abs/2610.25356) | CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。 |
| [^261] | [FlashReFocus: Interactive Image Refocusing in Seconds](https://arxiv.org/abs/2610.23297) | FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。 |
| [^262] | [Change the Number, Change the Answer: Assessing Rating Scale Sensitivity in LLM-Based Text Scoring](https://arxiv.org/abs/2610.22187) | 本文定义并系统研究了“量表敏感性”这一新概念，通过大规模交叉实验发现，评分量表的设计元素（格式、范围、符号）平均会使大语言模型的文本评分产生约0.10的归一化分数偏移，揭示了不同研究间分数可比性的重要隐患。 |
| [^263] | [SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications](https://arxiv.org/abs/2610.14918) | SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。 |
| [^264] | [When MLLMs miss what humans read: diagnosing and learning to read squint text](https://arxiv.org/abs/2610.13465) | 该论文发现多模态大模型存在“眯眼文字”识别缺陷，通过构建 SquintBench 与 SquintCtrl 两个基准诊断出空间选择是关键瓶颈，并据此提出 Squint OPSD 方法教会模型阅读此类隐藏文字。 |
| [^265] | [Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability](https://arxiv.org/abs/2610.11957) | 该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。 |
| [^266] | [AI Safety Considerations for Agents With Limited Time to Act](https://arxiv.org/abs/2610.10285) | 本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。 |
| [^267] | [CredLeakBench: Evaluating Credential Leakage and Recovery in LLM Agents](https://arxiv.org/abs/2610.08871) | 提出了 CredLeakBench 基准，用于评估大语言模型智能体在面对网络钓鱼与身份验证场景时区分恶意请求与真实请求、防止凭据泄露并安全自动化用户工作流程的能力。 |
| [^268] | [Unifying matrix completion across domains with a synthetically pretrained foundation model](https://arxiv.org/abs/2610.06751) | MatrixFormer是一个完全基于合成低秩矩阵预训练的矩阵补全基础模型，能在单次前向传播中预测每个缺失条目的完整分布，并以零样本方式在因果推断、表格插补和推荐系统等多个领域取得有竞争力的性能。 |
| [^269] | [ThunderSync: Lossless Acceleration of Agentic Reinforcement Learning](https://arxiv.org/abs/2610.05935) | ThunderSync 通过在所需输入就绪后立即流式计算梯度，在不引入策略陈旧性的前提下消除流水线空闲，实现对智能体强化学习的无损加速。 |
| [^270] | [Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation](https://arxiv.org/abs/2610.00447) | 该论文通过对来自六个生成器的300个冻结场景系统性变化八种渲染与描述文本配置因素，揭示文本到3D评估结果对配置高度脆弱——17/19个对齐评估器的配置方差超过生成器间方差，且18/19个评估器在某些配置下会改变点估计赢家，表明当前评估协议存在严重的不稳定性。 |
| [^271] | [MatLoom: Layered Text-to-Material Generation in a Compact Program Space](https://arxiv.org/abs/2609.40322) | MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。 |
| [^272] | [MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories](https://arxiv.org/abs/2609.40195) | MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。 |
| [^273] | [Disentangling Self-Distillation: Measuring and Modeling Acquisition and Retention](https://arxiv.org/abs/2609.39494) | 该论文首次将自蒸馏方法解耦为rollout来源、教师耦合方式和KL方向三个独立维度，通过1,200次系统实验构建统一框架，解释了以往固定组合研究导致矛盾结论的根源。 |
| [^274] | [UniEvo-VL: An On-policy Self-Distillation Training Recipe for Multimodal Model Self-improvement](https://arxiv.org/abs/2609.38721) | UniEvo-VL提出了一种在策略自蒸馏训练框架，让单个多模态模型在测试时计算中以自我批评作为特权信息，同时扮演教师和学生角色并最小化两者扩散分布的散度，从而实现无需外部教师模型的自我提升和图像生成能力的增强。 |
| [^275] | [Self-Evolving Harness on Multiple Tasks with the Agent as Its Own Optimizer](https://arxiv.org/abs/2609.38372) | 该论文提出让同一个冻结模型兼任求解器与提议者的自进化执行框架，智能体在覆盖五个不同领域基准的多任务上直接编辑运行自身的框架代码，实现接近递归自我改进的效果。 |
| [^276] | [Language as the Interface: Foundation-Model Contrastive Learning Links Transcriptomes and Electrophysiology](https://arxiv.org/abs/2609.37024) | LangPatch框架通过将基因描述与电生理特征映射到同一语言空间并进行配对对比学习，成功将预训练基因基础模型GenePT与Patch-seq电生理表型对齐，在多个小鼠和人类皮层数据集上取得了最优的转录组到电生理预测性能。 |
| [^277] | [Why Directly Learning Periodic Trajectories Can Fail](https://arxiv.org/abs/2609.32254) | 该论文揭示了一个共同机制——频率差异随时间累积使目标相位随参数快速变化——解释了为何直接学习周期轨迹会失败：未对齐轨迹即使在训练分布内也泛化不佳，而固定物理时间窗口训练的模型对训练范围外的周期轨迹误差很大，相位对齐则可显著改善分布内泛化。 |
| [^278] | [Jev-Mobile: Jev as an Executor for Mobile GUI Agents](https://arxiv.org/abs/2609.30186) | Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。 |
| [^279] | [Scope Before You Persist: Preventing Cross-Family Interference in Agent Memory](https://arxiv.org/abs/2609.29144) | 论文提出 Scoped-ORC 方法，通过将持久技能的检索范围严格限定在其原始任务族，防止跨任务族干扰，从而在不更新模型权重的情况下显著提升智能体的轨迹效用并消除有害的技能部署。 |
| [^280] | [Bridging Learned Visual Perception and Symbolic Belief-Space Planning](https://arxiv.org/abs/2609.16884) | 提出第三种范式“VLM作为概率落地器”，将视觉-语言模型谓词落地的不确定性表示为符号状态上的概率分布，实现信念空间规划并在不确定性下生成鲁棒规划。 |
| [^281] | [Fine-Tuning Fixes Mode Collapse and Over-Dispersion in LLMs](https://arxiv.org/abs/2609.16454) | 这篇论文证明大语言模型是否出现模式坍缩取决于具体模型和数据集，而充足的监督微调（SFT）数据能使LLM输出多样性收敛到目标分布，从而同时修复模式坍缩与过度分散问题。 |
| [^282] | [OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning](https://arxiv.org/abs/2609.16057) | 提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。 |
| [^283] | [Semantic-TVM: Structure-Preserving Trustworthy Virtual Memory for Memory-Augmented and Tool-Using Agents](https://arxiv.org/abs/2609.15011) | Semantic-TVM 提出可信虚拟内存（TVM）闭环运行时，利用可信本地模型仅替换预测出的敏感片段而非整个字段，在隐私暴露保持低水平的同时恢复了大部分任务效用，解决了记忆增强与工具使用智能体在远程 LLM 处理过程中的隐私泄露问题。 |
| [^284] | [No-Box Vulnerability Analysis: Description-only Detection of Indirect Prompt Injection Vulnerabilities in MCP Servers](https://arxiv.org/abs/2609.10854) | 本文提出“无盒漏洞分析”新范式，仅凭功能描述元数据即可在不访问或不与目标系统交互的情况下，假设性检测MCP服务器中所有可能实现里的间接提示注入漏洞。 |
| [^285] | [Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method](https://arxiv.org/abs/2609.08402) | 该论文提出了首个空地协同目标搜索基准AGOS-Bench、包含7.7k条轨迹的AGOS-Dataset数据集，以及无需训练、工具增强的智能体方法AGOS-Agent，用以评估和解决视觉-语言模型在UAV-UGV协作搜索与验证目标车辆任务中的难题。 |
| [^286] | [FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs](https://arxiv.org/abs/2609.02168) | 本文提出FUSE框架，在统一协议下通过知识、防御、危害三个正交流水线评估大语言模型的危险能力，并利用化学生物模块对四个系列的12个商用LLM进行横向比较，揭示了各模型间差异显著的危险能力画像。 |
| [^287] | [Agentic programs: an emerging form of scientific software in computational materials science](https://arxiv.org/abs/2609.00795) | 本文提出“智能体程序”这一新兴科学软件范式，将确定性算法与有界的LLM科学判断、任务验证和阶段性成熟相结合，并以从实验测量的无序晶体结构构建原子模型的DeMARS系统为例加以展示。 |
| [^288] | [Fool's Gold: Defensive Deception Against Safety-Removal Attacks on Open-Weight Models](https://arxiv.org/abs/2608.17202) | 该论文提出“愚人金”防御方法，通过训练诱饵响应在安全移除攻击后产生伪造信息，有效阻止开放权重模型被恶意利用。 |
| [^289] | [SysEvolve: An AI-native, safe, autonomous adversarial attack-defense co-evolutionary system](https://arxiv.org/abs/2608.15012) | 本文提出SysEvolve系统，通过攻击与防御AI代理的自主协同进化循环，解决网络安全中攻击加速与防御滞后之间的不对称性，实现安全、自驱动的对抗性攻防进化。 |
| [^290] | [Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding](https://arxiv.org/abs/2608.11095) | 该论文揭示并量化了智能体编程提示文件（如 CLAUDE.md）因“追加廉价、删除昂贵”的不完美记忆机制而近乎单调地无限制膨胀、最终导致智能体失效的“灾难性记忆”现象，并证明提示注释可有效遏制这一增长。 |
| [^291] | [Halpern Iteration Accelerates Solving High-Order Monotone Variational Inequalities](https://arxiv.org/abs/2608.08463) | 本文通过重新审视经典 Halpern 迭代并提出大步长不精确的 $p$ 阶方法 Halpern-ATM，实现了求解光滑单调变分不等式的 $\tilde{\mathcal{O}}(T^{-p})$ 快速收敛速率，改进了自 2012 年以来 $p \ge 2$ 阶方法的已知最佳结果。 |
| [^292] | [BODHI: Do LLMs Branch Out and Discover Heterogeneous Inferences?](https://arxiv.org/abs/2608.02867) | 研究发现RLVR训练的大语言模型相比蒸馏模型表现出更低的语义多样性和更少的推理分支，表明其推理性能提升主要来自采样效率的改善而非推理能力的真正扩展。 |
| [^293] | [Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale](https://arxiv.org/abs/2608.00101) | 本文首次在生产规模上（740万用户、57亿次LLM调用）对GitHub Copilot编码智能体的追踪数据进行系统性分析，揭示了LLM调用与工具执行近乎1:1耦合、KV缓存命中率在轮次边界和模型切换时大幅下降、以及长尾资源占用分布等独特工作负载特性及其系统启示。 |
| [^294] | [Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in recommendation systems](https://arxiv.org/abs/2607.24865) | 提出双用途语义ID，通过分层量化将连续嵌入压缩为离散token，既作为协同身份建模用户-物品交互，又可通过轻量级解码器按需重建嵌入，从而取代大规模向量存储，实现推荐系统中LLM级的I/O效率。 |
| [^295] | [A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of Coronary Angiograms](https://arxiv.org/abs/2607.22139) | 该论文推出了CARDIAG——首个针对冠状动脉造影SYNTAX节段分割的多中心基准数据集，通过严格防止患者级和中心级数据泄露并提供前所未有的丰富注释（不平衡多类标签、导管与狭窄掩膜、专家不确定性图等），同时对24种涵盖CNN、ViT和SSM的深度学习架构进行了该领域最广泛的基准评估。 |
| [^296] | [R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables](https://arxiv.org/abs/2607.02921) | 该论文提出了R3D-Bench——一个基于第一人称RGB-D视频、包含2,950道定量空间推理问题的基准测试，并引入了模型无关的空间工具调用框架R3D，通过分割和深度提升的对象表示构建3D场景来实现定量3D空间推理。 |
| [^297] | [CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation](https://arxiv.org/abs/2606.22082) | CodeTeam是一个基于大语言模型的多智能体框架，通过将规划、决策与实现划分为独立协调的阶段——由多个架构师智能体竞争起草设计草图、CTO智能体将其规范化为机器可校验的契约、开发者智能体据此生成代码——来完成仓库级代码生成任务。 |
| [^298] | [What Graph Unlearning Cannot Erase: Physics-Induced Leakage of Cyberattack Information in Electric Vehicle Charging Networks](https://arxiv.org/abs/2606.19566) | 该论文将图遗忘技术应用于电动汽车充电站的网络攻击定位任务，并揭示了即便执行了遗忘操作，网络攻击信息仍会经由电网的物理规律发生泄露，无法被完全抹除。 |
| [^299] | [EgoPhys: Physics-Informed Digital Twins for Deformable Objects from Egocentric Video](https://arxiv.org/abs/2606.16202) | 提出EgoPhys框架，仅凭单段第一人称视角RGB交互视频即可为可变形物体构建预测性物理数字孪生，无需多视角采集或高昂的逐物体物理优化。 |
| [^300] | [Auditing Reward Hackability in Code RL Training Environments](https://arxiv.org/abs/2606.16062) | 该论文通过基于标准补丁的执行验证系统审计了代码强化学习训练环境的奖励作弊问题，发现约22.4%（强攻击者下57.8%）的SWE-bench Verified任务会接受错误补丁，且模型在这些可作弊任务上的通过率显著更高，凸显了对RL训练环境进行作弊检测与修复审计的必要性。 |
| [^301] | [Morphology-Aware Sample Assignment: Overcoming IoU Insensitivity for Surface Defect Detection](https://arxiv.org/abs/2606.13723) | 该论文通过理论分析发现IoU响应曲线存在非敏感区域，并提出融合面积、形状和长宽比的多维形态相似性度量来改进正样本分配，从而提升表面缺陷检测的精度与可靠性。 |
| [^302] | [Chain of Operators: An Inference-Time Harness for In-Context Operator Learning](https://arxiv.org/abs/2606.12318) | 本文提出算子链（CHOP）框架，利用冻结基础模型的上下文学习能力，将复杂分布外任务分解为闭式数学算子序列，实现无需参数更新的推理时适应。 |
| [^303] | [Exploration Structure in LLM Agents for Multi-File Change Localization](https://arxiv.org/abs/2606.11976) | 该论文通过系统比较六种逐步增加结构的智能体配置，研究了小型LLM在多文件变更定位任务中哪些结构（如输出受限的仓库工具、领域智能体注册表等）真正带来了大部分性能收益。 |
| [^304] | [EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks](https://arxiv.org/abs/2606.07700) | 本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。 |
| [^305] | [Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows](https://arxiv.org/abs/2606.06923) | 该论文发现编排对工具使用型LLM智能体的作用取决于检索质量：完美检索时声明式技能文件中性甚至有害，嵌入检索时则有益，在基线与脚手架分歧的任务上声明式胜率从42.4%提升至62.7%。 |
| [^306] | [Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events](https://arxiv.org/abs/2606.02522) | 提出了Moment-Video基准，通过涵盖七大门类、1,000个人工验证的视频问答对，系统评估视频多模态大语言模型在瞬时视觉事件理解上的时间保真度。 |
| [^307] | [Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation](https://arxiv.org/abs/2605.31094) | 本文将全景分割评估中的片段匹配重新表述为受约束的二部图指派问题，通过独立约束预测侧与真实侧的度数定义了四种匹配策略（一对一、多对一、一对多、多对多），并证明除多对多之外的前三种策略均能给出可解释的实例级错误归因。 |
| [^308] | [BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision](https://arxiv.org/abs/2605.27380) | BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。 |
| [^309] | [Trust Region Q Adjoint Matching](https://arxiv.org/abs/2605.27079) | 本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。 |
| [^310] | [AutoDFT: A Closed-Loop Multi-Agent Framework for Autonomous DFT Calculations](https://arxiv.org/abs/2605.26179) | AutoDFT是一个闭环多智能体框架，将大语言模型推理嵌入DFT计算生命周期的每个阶段，使DFT计算能够根据中间结果自主适应和调整计算路径，摆脱了传统工作流对专家干预和预先规划的依赖。 |
| [^311] | [GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models](https://arxiv.org/abs/2605.23238) | 提出GENSTRAT系统，通过程序化生成双人零和不完全信息下注博弈按需创建全新评估游戏，解决了固定游戏套件饱和与数据污染问题，并通过50个游戏的基准和40,000多场模型对战的锦标赛揭示了不同LLM策略推理能力的显著差异。 |
| [^312] | [Towards Clinically Interpretable Ophthalmic VQA via Spatially-Grounded Lesion Evidence](https://arxiv.org/abs/2605.22414) | 该论文提出了FundusGround基准，通过ETDRS网格对眼底病灶进行空间定位标注，构建了包含10,719张眼底图像、15,595个精细病灶标注和72,706个问题的数据集，推动临床可解释的眼科视觉问答发展。 |
| [^313] | [CatalyticMLLM: A Graph-Text Multimodal Large Language Model for Catalytic Materials](https://arxiv.org/abs/2605.17254) | 提出CatalyticMLLM，一个统一的图-文本多模态大语言模型，将催化材料的性质预测与逆向结构设计整合到单一模型中，避免了传统解耦范式因适用域不匹配与误差累积而导致的逆向设计性能下降问题。 |
| [^314] | [StreamPro: Reinforcement Learning with Multi-Grained Rewards for Proactive Streaming Video Understanding](https://arxiv.org/abs/2605.16381) | 提出StreamPro两阶段训练框架，先用CB-Stream Loss缓解SFT中沉默与响应信号的监督不平衡，再通过响应匹配奖励与序列质量奖励相结合的多粒度GRPO强化学习，解决奖励稀疏与响应序列依赖优化的难题，实现主动式流式视频理解。 |
| [^315] | [PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation](https://arxiv.org/abs/2605.14269) | PhyMotion提出了一种结构化的三维运动奖励，通过将生成视频中恢复的人体运动导入物理模拟器进行重定向和物理可行性评估，从而生成更加真实、符合物理规律的人体运动视频。 |
| [^316] | [Comment and Control: Hijacking Agentic Workflows via Context-Grounded Evolution](https://arxiv.org/abs/2605.11229) | 该论文提出了首个针对自动化平台上LLM智能体工作流的检测与利用框架JAW，其通过“上下文锚定演化”方法在混合程序分析推导出的上下文中演化恶意输入（如issue评论），从而劫持智能体工作流，实现凭据窃取和任意命令执行等攻击。 |
| [^317] | [NoiseRater: Meta-Learned Noise Valuation for Diffusion Model Training](https://arxiv.org/abs/2605.08144) | 提出NoiseRater，一种通过双层元学习对给定样本和时间步条件下的噪声实例进行评分的模型，用于选择更有信息量的训练噪声，并发现同一时间步的不同噪声实现并非同等有用。 |
| [^318] | [Mosaic: Efficient CPU-GPU Coordinated Block-Sparse Attention](https://arxiv.org/abs/2605.07719) | Mosaic提出了一种CPU-GPU协同的块稀疏注意力系统，通过输出贡献感知的KV预算分配、注意力头特定且粒度感知的稀疏配置以及跨设备协同调度，解决了长上下文LLM解码中CPU驻留KV缓存访问受PCIe带宽限制和GPU闲置的低效问题。 |
| [^319] | [Hierarchical Task Network Planning with LLM-Generated Heuristics](https://arxiv.org/abs/2605.07707) | 本研究将基于大语言模型生成启发式方法的技术从经典规划扩展到分层任务网络（HTN）规划，探索LLM能否为HTN规划生成有效的搜索启发式，并在七个标准全序HTN基准领域上验证其效果。 |
| [^320] | [When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds](https://arxiv.org/abs/2605.06615) | 本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。 |
| [^321] | [ScrapMem: A Bio-Inspired Framework for On-Device Personalized Agent Memory via Optical Forgetting](https://arxiv.org/abs/2605.03804) | ScrapMem提出了一种仿生的设备端智能体记忆框架，通过“光学遗忘”机制逐步压缩旧记忆页面的图像质量以降低存储成本，同时借助情景记忆图保持语义一致性与结构化检索能力。 |
| [^322] | [A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models](https://arxiv.org/abs/2604.21903) | 该论文提出一种尺度自适应的时空联合超分辨率框架，将任务分解为条件均值的确定性预测与残差条件扩散模型，使同一架构可复用于不同的空间和时间超分辨率因子，并可通过质量守恒变换保持降水总量一致。 |
| [^323] | [The Projector Forgets Too: Joint Merging for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2604.14016) | 该论文发现多模态持续指令微调中的灾难性遗忘不仅源于语言模型的LoRA适配器，还源于跨模态投影器，并提出JoMe方法通过联合合并这两部分的参数来有效缓解遗忘。 |
| [^324] | [GAAMA: Graph Augmented Associative Memory for Agents](https://arxiv.org/abs/2603.27910) | 该论文提出GAAMA，一种围绕主题组织对话的图增强联想记忆框架，通过情节、原子事实、主题概念与高阶反思四类节点及五种结构边避免巨型枢纽问题，并结合最近邻搜索与个性化PageRank提升AI智能体长期记忆的检索效果。 |
| [^325] | [TopoPilot: Reliable Conversational Workflow Automation for Topological Data Analysis and Visualization](https://arxiv.org/abs/2603.25063) | 提出了TopoPilot——一个采用“编排器+验证器”双智能体架构的可靠智能体框架，用于自动化拓扑数据分析与可视化等复杂科学可视化工作流。 |
| [^326] | [Does This Gradient Spark Joy?](https://arxiv.org/abs/2603.20526) | 论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。 |
| [^327] | [Delightful Policy Gradient under Distributed Friction](https://arxiv.org/abs/2603.20521) | 本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。 |
| [^328] | [The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas](https://arxiv.org/abs/2603.11842) | 本研究通过系统检索并综合分析28篇二次综述与路线图论文，梳理了生成式人工智能在信息系统领域带来的益处、挑战及未来研究议程。 |
| [^329] | [VeRA: Renewing Reasoning Benchmarks with Executable Specifications](https://arxiv.org/abs/2602.13217) | VeRA 将推理基准题目转化为由模板、输入生成器和确定性答案程序组成的可执行任务族，从而自动生成全新及更难的题目，揭示了大模型在固定题目上的高分与面对新题时的真实鲁棒性之间的显著差距。 |
| [^330] | [EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL](https://arxiv.org/abs/2602.04417) | 提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。 |
| [^331] | [AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent](https://arxiv.org/abs/2602.03955) | AgentArk提出了一种新框架，通过推理增强微调、基于轨迹的增强和过程感知蒸馏三种分层蒸馏策略，将多智能体系统在迭代辩论中形成的智能蒸馏到单个LLM模型的权重中，使单个智能体兼具计算效率和多智能体级别的推理与自我纠错能力。 |
| [^332] | [TabularMath: Evaluating Computational Extrapolation in Tabular Learning via Program-Verified Synthesis](https://arxiv.org/abs/2602.02523) | 本文提出TabularMath基准，将GSM8K和AIME问题编译为114个经程序验证的确定性计算，用以评估表格模型从统计插值到计算外推的能力，发现TabPFN v2.5在分布内中位数R²高达0.999，且是唯一在超出训练范围时中位数R²仍保持正值（0.76）的表格方法。 |
| [^333] | [SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning](https://arxiv.org/abs/2602.01990) | 本文提出SAME框架，通过稳定专家路由和防止专家覆盖，解决了多模态持续指令微调中的路由漂移和专家漂移问题。 |
| [^334] | [Learning and Comparing Local Traces for Weakly Supervised Image Manipulation Localization](https://arxiv.org/abs/2601.06222) | 提出一种基于CLIP的弱监督图像篡改定位框架，通过面向痕迹的视觉—语言自适应（TOVA）与关系感知痕迹判别（RATD）来学习和比较局部篡改痕迹，在无需像素级标注的情况下实现更准确完整的篡改区域定位。 |
| [^335] | [PsyAgent: Constructing Human-like Agents Based on Psychological Modeling and Contextual Interaction](https://arxiv.org/abs/2601.06158) | PsyAgent提出了一种“模式优先”框架，将大五人格特质画像（IS）与多情境化框架（MSC）耦合，并利用两者组合合成监督数据微调紧凑模型，使类人智能体的行为既保持人格稳定又能随情境灵活变化。 |
| [^336] | [Verify Before You Generate: Sketch-Guided Efficient Motion-Planning for Video Generation](https://arxiv.org/abs/2511.17450) | 提出了 SketchVerify，一个无需训练的框架，在调用昂贵的视频生成器之前先通过轻量级视频草图验证运动规划，从而高效比较并迭代精化多条候选轨迹，以低成本获得高质量运动。 |
| [^337] | [Object-Centric World Models for Causality-Aware Reinforcement Learning](https://arxiv.org/abs/2511.14262) | 该论文提出STICA统一框架，利用以对象为中心的Transformer作为世界模型，并配合因果感知的策略与价值网络，使强化学习智能体能够像人类一样分解复杂多对象环境，从而实现更高效的样本学习与决策。 |
| [^338] | [Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph](https://arxiv.org/abs/2511.00086) | 该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。 |
| [^339] | [Provable Selective Auto-labeling with Reliability Guarantees](https://arxiv.org/abs/2510.14581) | 提出了“共形标注”这一新方法，将选择性自动标注形式化为多重假设检验问题，通过构建共形p值并结合Benjamini-Hochberg程序，在错误标注率可证明受控的前提下筛选出可信的自动生成标签子集。 |
| [^340] | [Learning to Assimilate: Amortized Control for Autoregressive Diffusion](https://arxiv.org/abs/2510.06637) | 提出了 CADA 方法，在冻结的自回归扩散模型之上训练轻量级控制器实现摊销式数据同化，无需逐次预测进行优化即可提升稀疏观测下的长程预测精度，并将推理速度提高 10 倍以上。 |
| [^341] | [RAG Makes Guardrails Brittle? Investigating Robustness of Guardrails under RAG-style Contexts](https://arxiv.org/abs/2510.05310) | 本研究通过系统评估发现，RAG场景中仅插入良性的检索文档就能使基于LLM的安全防护栏在约10%的情况下翻转判断，且多数翻转会将原本正确的安全决策反转为错误决策，揭示了现有防护栏在RAG式上下文下的脆弱性。 |
| [^342] | [Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](https://arxiv.org/abs/2510.04392) | 本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。 |
| [^343] | [Rhythm-TTS: High-Fidelity Shortcut Latent Modeling and Probabilistic Rhythm Generation for Zero-Shot Text-to-Speech](https://arxiv.org/abs/2510.02848) | Rhythm-TTS 提出了一种捷径零样本文本转语音框架，将预测的离散编解码器标记作为语义-声学先验，直接生成高保真的整体连续语音潜变量，并结合概率化节奏生成，从而兼顾离散标记的可因子分解性与连续表示的高保真度。 |
| [^344] | [Reconstruction Alignment Improves Unified Multimodal Models](https://arxiv.org/abs/2509.07295) | 提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。 |
| [^345] | [Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution](https://arxiv.org/abs/2509.04588) | 论文发现插入/删除等仅输出指标无法检测特征归因是否经由支持原始预测的计算路径，提出了可微的EQO方法揭示“通路重路由”现象，并引入逐层梯度过滤器ClipGrad来缓解该问题。 |
| [^346] | [Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization](https://arxiv.org/abs/2505.23331) | 本文将群体相对策略优化（GRPO）应用于下一尺度视觉自回归（VAR）模型的强化学习微调，通过对齐美学与CLIP奖励信号显著提升图像质量与风格控制能力，并使模型能够泛化生成预训练阶段未见过的图像风格。 |
| [^347] | [LoopNav: Benchmarking Spatial Consistency in World Models](https://arxiv.org/abs/2505.22976) | 提出LoopNav数据集与基准，通过基于环路的导航任务来评估世界模型的长距离空间一致性能力。 |
| [^348] | [Causality can systematically address the monsters under the bench(marks)](https://arxiv.org/abs/2502.05085) | 这篇立场论文主张，因果性为系统性地解决机器学习基准测试中的偏差、伪影和数据泄露等“怪兽”问题提供了理想框架，通过显式化因果假设并利用有原则的分析工具来提升评估的可信度与一致性。 |
| [^349] | [A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning](https://arxiv.org/abs/2410.24035) | 该论文提出了一个基于不确定性感知模仿学习的上下文自适应策略框架，通过策略融合与不确定性量化技术，首次将任务参数化、鲁棒性和响应式机器人操作三者结合，解决了现有动力系统方法无法根据任务相关变量灵活调节策略的问题。 |
| [^350] | [Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities](https://arxiv.org/abs/2409.01382) | 该研究通过对比四种LLM生成的14,485个函数和11,913个类，发现代码检测中粒度效应的影响远超模型间差异（达8.6倍），函数级与类级检测依赖截然不同的结构特征，且存在依赖粒度的检测性能反转现象。 |
| [^351] | [OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs.](http://arxiv.org/abs/2309.03876) | OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。 |

# 详细

[^1]: 无需方差假设的最优线性回归

    Optimal Linear Regression Without a Variance

    [https://arxiv.org/abs/2611.27983](https://arxiv.org/abs/2611.27983)

    该论文提出一种基于经验残差剪枝的黑盒归约方法，在噪声仅有未知有限α矩（α∈(1,2)）且协变量仅有有限四阶矩的无穷方差情形下，可将任意有限方差回归算法自动转化为达到最优统计速率的弱矩回归算法，且无需知道矩指数、噪声尺度等任何先验信息。

    

    我们研究了无穷方差情形下的高概率线性回归问题，其中回归噪声仅对某个未知的 α∈(1,2) 具有有限的条件 α-矩 ν，而协变量可能只具有直至给定阶数的有限矩（例如4阶矩）。我们提出了一种黑盒归约方法，能够将有限方差的回归程序转换为弱矩条件下的回归程序。该归约基于经验残差剪枝，并且对矩指数、噪声尺度、协方差以及设计矩都是不可知的（即无需预先知道这些量）。在协变量满足有限四阶矩的条件下，任何达到亚高斯速率的有限方差子程序，当样本数 n 超过某个适当定义的阈值时，均可得到如下结果：在不知道 α 或 ν 的情况下，我们以至少 1−δ 的概率有 ‖θ̂−θ*‖_Σ ≲_α ν^{1/α} ((d+log(1/δ))/n)^{…

    arXiv:2611.27983v1 Announce Type: cross  Abstract: We study high-probability linear regression in the infinite variance regime, wherecthe regression noise has only a finite conditional $\alpha$-moment $\nu$ for some unknown $\alpha\in(1,2)$, and the covariates may only have finite moments up to a given order (e.g., 4th moment). We give a black-box reduction that converts finite-variance regression procedures into weak-moment procedures. The reduction is based on empirical residual pruning and is agnostic to the moment exponent, the noise scale, the covariance, and the design moment. Under a finite fourth-moment condition on the covariates, any finite-variance subroutine attaining the sub-Gaussian rate yields the following when the number of samples $n$ is above a suitably defined threshold: Without knowing $\alpha$ or $\nu$, we have with probability at least $1-\delta$ that $ \|\widehat\theta-\theta^*\|_\Sigma   \lesssim_\alpha   \nu^{1/\alpha}   \big(\frac{d+\log(1/\delta)}{n}\big)^{\
    
[^2]: 防穿透的世界模型

    Non-penetrating World Models

    [https://arxiv.org/abs/2611.27963](https://arxiv.org/abs/2611.27963)

    该论文提出NoPeBench基准和NoPeWorld框架，通过在冻结的扩散Transformer隐藏状态上训练小型预测头来检测并替换不安全动作，有效防止交互式视频世界模型中的物体穿透问题。

    

    我们研究了交互式视频世界模型中防止物体穿透这一新问题。为了解现有模型如何应对该问题，我们提出了NoPeBench，一个评估世界模型对物理障碍物响应能力的基准测试。评估结果显示，目前没有任何开源模型能够可靠地防止此类穿透失败。然而令人惊讶的是，这些模型内部已经编码了预测此类违规行为所需的信息。我们发现，扩散Transformer（DiT）的隐藏状态可以预测某个候选动作是否会穿透障碍物。基于这一发现，我们提出了NoPeWorld，一个检测并防止穿透的轻量级框架。具体而言，我们在冻结的DiT隐藏状态之上训练一个小型预测头，用于判断候选动作是否安全，随后将不安全的动作替换为最能保留原始意图的安全动作。尽管方法简洁，NoPeWorld显著减少了……

    arXiv:2611.27963v1 Announce Type: cross  Abstract: We study a new problem of preventing object penetration in interactive video world models. To understand how existing models handle this, we introduce NoPeBench, a benchmark that evaluates how world models respond to physical obstacles. Our evaluation reveals that none of the open-source models can reliably prevent such failures. Surprisingly, however, these models already encode the information needed to predict such violations. We find that the hidden states of diffusion transformers (DiTs) can predict whether a candidate action will penetrate an obstacle. Building on this, we introduce NoPeWorld, a lightweight framework that detects and prevents penetration. Specifically, we train a small prediction head on top of a frozen DiT's hidden states to read out if a candidate action is safe or not. Unsafe actions are then replaced with safe ones that best preserve the original intent. Despite its simplicity, NoPeWorld substantially reduces
    
[^3]: 模型雕刻：训练能够优化自身拓扑结构的 Transformer

    Model Carving: Training Transformers That Optimize Their Own Topology

    [https://arxiv.org/abs/2611.27962](https://arxiv.org/abs/2611.27962)

    提出模型雕刻方法，通过在每个子层上添加可学习的标量门控，让 Transformer 在单次训练中同时学习权重与架构，由训练损失自动移除冗余组件，释放出内部更小、更高效的异构网络。

    

    机器学习在很大程度上将架构选择与训练过程相互分离。例如，经典的 Transformer 在每一层以相同的顺序训练一组固定的注意力子层和 MLP 子层，仅优化其中的权重。这种统一性是一种惯例而非有原则的设计选择，并且很可能是次优的：针对给定任务的最佳架构可能更小、更高效，且远非均匀一致。我们认为，理想情况下，训练本身应该塑造模型的拓扑结构。为此，我们提出了模型雕刻，它在单次运行中同时学习 Transformer 的权重与架构。通过在每个子层上应用可学习的标量门控，我们让训练损失能够在训练过程中移除模型自身的组件——从而释放出模型内部一个更小、异构的网络。我们通过对参数量从 1.4 亿到 27 亿的标准 Transformer 以及循环 Transformer 进行预训练来评估模型雕刻

    arXiv:2611.27962v1 Announce Type: cross  Abstract: Machine learning has largely kept separate the choice of architecture and the process of training. A classic Transformer, for instance, trains a fixed set of attention and MLP sublayers at every layer, in the same order, and optimizes the weights within. This uniformity is a convention rather than a principled design choice, and it is likely suboptimal: the best architecture for a given task may be smaller, more efficient, and far from uniform. We believe that, ideally, training itself should shape the model's topology. To this end, we introduce Model Carving, which learns both a Transformer's weights and its architecture in a single run. By applying learnable, scalar gates to each sublayer, we allow the training loss to remove the model's own components during training -- freeing a smaller heterogeneous network inside. We evaluate Model Carving by pre-training standard Transformers from 140M to 2.7B parameters and looped Transformers 
    
[^4]: BioEmu2：可扩展的生物分子动力学模拟

    BioEmu2: Scalable Emulation of Biomolecular Dynamics

    [https://arxiv.org/abs/2611.27924](https://arxiv.org/abs/2611.27924)

    BioEmu2是一个融合静态结构预测、平衡态采样与时间演化的原子级生物分子动力学生成模型，采样速度比分子动力学快数个数量级，可支持兆道尔顿级多组分复合物预测，平衡分布自由能误差接近1 kcal/mol，并能预测长时程动力学轨迹。

    

    尽管结构预测已取得重大进展，但在大规模尺度上模拟驱动生物分子功能的运动与相互作用仍然是一项核心挑战。我们提出了BioEmu2，一个面向原子级分辨率生物分子结构的生成模型，它将静态结构预测、平衡态采样与时间演化结合在一起，其采样速度比分子动力学（MD）快数个数量级。该模型在实验结构、合成结构以及大量MD数据上训练，支持包含蛋白质、核酸、修饰残基、配体和离子的多组分体系。其结构预测精度可与顶尖的开源权重生物分子折叠模型相媲美，且上下文并行模式使其能够预测兆道尔顿规模的复合物。其平衡态分布可逼近MD参考结果，在所测试的蛋白质和复合物上自由能误差接近1 kcal/mol。时间传播功能可预测长时间尺度的动力学轨迹以及……的动力学（原文摘要在此处截断）。

    arXiv:2611.27924v1 Announce Type: new  Abstract: Despite major advances in structure prediction, modelling the motions and interactions that drive biomolecular function at scale remains a central challenge. We introduce BioEmu2, a generative model for atomically resolved biomolecular structures combining static structure prediction, equilibrium sampling and time evolution, that samples orders of magnitude faster than molecular dynamics (MD). Trained on experimental and synthetic structures and extensive MD data, it supports multi-component systems containing proteins, nucleic acids, modified residues, ligands and ions. Structure-prediction accuracy is comparable to top open-weight biomolecular folding models, and a context parallel mode enables prediction of megadalton complexes. Equilibrium distributions approximate MD references with free-energy errors near 1 kcal/mol across tested proteins and complexes. Time propagation predicts long-timescale dynamical trajectories and kinetics of
    
[^5]: SUAF：用于零样本策略改进的谱通用优势特征

    SUAF: Spectral Universal Advantage Features for Zero-Shot Policy Improvement

    [https://arxiv.org/abs/2611.27919](https://arxiv.org/abs/2611.27919)

    本文提出谱通用优势特征（SUAF），通过对以动作为中心的典型依赖矩阵进行奇异值分解，构建可通过线性奖励投影恢复延迟优势的成对特征，从而在奖励无关表征下实现零样本策略改进。

    

    奖励无关的表征使强化学习智能体能够跨任务复用知识，但有限的表征容量引发了三个基本问题：这些表征应当捕捉何种统计依赖关系，应保留该依赖中的哪些成分用于策略改进，以及丢弃其余成分会对决策产生怎样的影响？我们通过典型依赖分析来回答这些问题。针对参考策略，我们构建了以动作为中心、正滞后的典型依赖矩阵（CDM），该矩阵分离出当前动作在当前状态之外的额外预测贡献。对其实施奇异值分解即可得到谱通用优势特征（SUAF），这是一对成对特征，通过线性奖励投影即可恢复延迟优势。我们证明该谱量化了当前动作与未来状态-动作访问分布之间的互信息……

    arXiv:2611.27919v1 Announce Type: cross  Abstract: Reward-independent representations enable reinforcement learning agents to reuse knowledge across tasks, but limited representation capacity raises three fundamental questions: what statistical dependence should these representations capture, which components of that dependence should be retained for policy improvement, and how does discarding the remaining components affect decision-making? We address these questions through canonical dependence analysis. For a reference policy, we construct an action-centered, positive-lag canonical dependence matrix (CDM) that isolates the additional predictive contribution of the current action beyond the current state. Its singular value decomposition yields Spectral Universal Advantage Features (SUAF), paired features that recover delayed advantage through linear reward projections. We show that the spectrum quantifies mutual information between current action and future state-action visitation c
    
[^6]: StereoVLM：用于度量三维空间推理的双目视觉语言模型

    StereoVLM: Binocular Vision-Language Models for Metric 3D Spatial Reasoning

    [https://arxiv.org/abs/2611.27915](https://arxiv.org/abs/2611.27915)

    提出了首个几何感知的双目视觉语言模型StereoVLM，利用标定双目相机的显式跨视角几何消除现有VLM在度量三维空间推理中的歧义。

    

    视觉语言模型（VLM）展现出强大的语义理解能力，但在对物理世界中度量三维空间的推理方面仍然存在局限。现有的VLM通常从一个或多个图像中推断三维场景结构，而缺乏已知的相机标定或跨视角几何约束。这使得度量三维推断本质上具有歧义性，也无法确定其空间预测究竟是基于视觉证据，还是主要由语义先验驱动。相比之下，经过标定的双目立体视觉提供了显式的跨视角几何信息，能够消除这种歧义，为度量三维感知和几何泛化提供更可靠的基础。我们研究了VLM如何直接从标定的双目观测中学习可迁移的三维表征。为此，我们提出了StereoVLM，这是首个用于度量三维空间推理的几何感知双目视觉语言模型。StereoVLM遵循一种分层学习范式……

    arXiv:2611.27915v1 Announce Type: cross  Abstract: Vision-language models (VLMs) exhibit strong semantic understanding but remain limited in their ability to reason about metric 3D space in the physical world. Existing VLMs typically infer 3D scene structure from one or more images without known camera calibration or cross-view geometric constraints. This makes metric 3D inference inherently ambiguous and leaves it unclear whether their spatial predictions are grounded in visual evidence or driven primarily by semantic priors. In contrast, calibrated stereo provides explicit cross-view geometry that resolves this ambiguity and offers a more reliable foundation for metric 3D perception and geometric generalization. We investigate how VLMs can learn transferable 3D representations directly from calibrated binocular observations. To this end, we introduce StereoVLM, the first geometry-aware binocular VLM for metric 3D spatial reasoning. StereoVLM follows a hierarchical learning paradigm s
    
[^7]: 星形全景质量：选择性评估分裂与合并的实例

    Star Panoptic Quality: Selectively Evaluating Split and Merged Instances

    [https://arxiv.org/abs/2611.27913](https://arxiv.org/abs/2611.27913)

    提出了新型评估框架StarPQ（星形全景质量），通过构建星形森林等方法，能够检测存在偏差的参考标注中分裂或合并的实例，并区分有效与无效的过度分割和欠分割。

    

    检测单个实例对于许多分割问题至关重要。然而，专家实例标注往往不可用，因此实例通常通过启发式方法进行近似（例如，作为语义图的连通分量），这可能合并相互接触的物体或分裂破碎的物体。即使存在专家标注，专家之间的分歧或结构勾画偏差也可能产生分裂或合并的参考实例，分别对应于过度分割或欠分割。全景质量及其相关指标在评估分割和检测质量时对所有实例进行等权重加权，而不考虑实例大小。然而，当参考标注存在偏差时，这些指标既无法检测分裂或合并的实例，也无法区分有效与无效的过度分割或欠分割。我们提出了星形全景质量，这是一种新颖的评估框架，通过两个步骤弥合了这一差距：首先，我们构建了一种所谓的星形森林……（原文摘要在此处截断）

    arXiv:2611.27913v1 Announce Type: cross  Abstract: Detecting individual instances is crucial for many segmentation problems. However, expert instance annotations are often unavailable, so instances are approximated heuristically (e.g., as connected components of semantic maps), which can merge touching objects or split fragmented ones. Even when expert annotations exist, expert disagreement or structural delineation biases can yield split or merged reference instances, corresponding to over- or under-segmentation, respectively. Panoptic Quality (PQ) and related metrics capture both segmentation and detection quality while weighting all instances equally, regardless of size. Yet when the references are biased, these metrics can neither detect split or merged instances nor distinguish valid from invalid overor under-segmentation. We present Star Panoptic Quality (StarPQ), a novel evaluation ramework, which closes this gap in two steps: First, we construct a so-called star forest in the b
    
[^8]: 功能深度是习得的：分离各层唯有协同作用才能产生的计算

    Functional Depth Is Learned: Isolating the Computation That Layers Produce Only Together

    [https://arxiv.org/abs/2611.27911](https://arxiv.org/abs/2611.27911)

    本文提出一种精确的容斥分解方法来分离“连通分量”（即各层唯有共同作用时才会产生的计算），并通过对 Pythia、OLMo、Amber 公开检查点的追踪发现这种跨层协同计算随预训练不断增强且删除它会显著损害模型性能，从而证明 Transformer 的功能深度是在训练过程中习得的。

    

    Transformer 的性能随深度提升而扩展，但层数只是统计了顺序计算的机会：各层仍可能表现为几乎相互独立的精细化处理，彼此之间几乎没有依赖。我们研究了 Transformer 在预训练过程中如何发展出功能深度，引入了一种精确的容斥分解方法，以分离出“连通分量”，即在恒等消融下，只有当一组指定层共同作用时才会出现的计算。通过在 Pythia、OLMo 和 Amber 的公开检查点上追踪这一量，我们得到三项发现。第一，随着预训练的进行，在所有测量的交互阶数上，“相对于写入的连通协调”都从其初始化或最早检查点的基线开始增强。第二，这些连通分量在直接干预下对模型行为具有实质性影响：删除它们对困惑度造成的损害大于删除一个范数匹配的随机方向……（原文摘要在此处截断）

    arXiv:2611.27911v1 Announce Type: new  Abstract: Transformer performance scales with depth, but layer count merely counts the opportunities for sequential computation: layers could still behave as nearly independent refinements that barely depend on one another. We study how transformers develop functional depth during pretraining, introducing an exact inclusion-exclusion decomposition that isolates the connected component, i.e., the computation that appears only when a specified set of layers acts together under identity ablation. Tracking this quantity across public checkpoints of Pythia, OLMo, and Amber yields three findings. First, write-relative connected coordination strengthens from its initialization or earliest-checkpoint baseline as pretraining proceeds, across all measured interaction orders. Second, these connected components are behaviorally consequential under direct intervention: deleting them damages perplexity more than deleting a norm-matched random direction at the s
    
[^9]: 决策列作为索引：面向可解释实体匹配与搜索的可解释嵌入

    Decision Columns as an Index: Interpretable Embeddings for Explainable Entity Matching and Search

    [https://arxiv.org/abs/2611.27910](https://arxiv.org/abs/2611.27910)

    该论文提出用“决策列”（决策模型对自然语言问题给出的“是/否/未提及”校准答案）构建可解释嵌入，使相似度可精确分解为逐问题贡献、支持语义过滤与增量扩展，并在产品基准上取得与稠密嵌入相当的实体匹配效果，同时提供精确可审计的解释。

    

    业务数据系统必须判断记录是否指向同一实体，并检索出满足那些决定性属性并不明确的请求的记录。随着智能体对这些记录执行操作，这两项任务都需要可审计的理由。稠密嵌入虽然表现良好，但其坐标不透明、无法按语义过滤，且要添加新属性就必须对整个语料库重新编码。我们提出由决策列构建的可解释嵌入：每一列存储一个决策模型对某个自然语言问题的校准答案，取值为“是”、“否”或“未提及”。一条记录的嵌入就是它在有序问题集上的一组答案。相似度可以精确分解为逐问题的贡献，而新增问题只会增加坐标、不会改变已有坐标。在公开的产品基准测试中，在“先检索后验证”的实体匹配器中，问题空间检索在统计意义上与稠密候选生成相当，同时还能提供精确的解释。

    arXiv:2611.27910v1 Announce Type: cross  Abstract: Business data systems must determine whether records refer to the same entity and retrieve records satisfying requests whose decisive properties are implicit. As agents act on these records, both tasks require auditable reasons. Dense embeddings perform well but offer opaque coordinates, cannot be filtered by meaning, and require corpus re-encoding to add new properties. We propose interpretable embeddings built from decision columns: each column stores a decision model's calibrated answer to a natural-language question, with outcomes yes, no, or not stated. A record's embedding is its answers across an ordered set of questions. Similarity decomposes exactly into per-question contributions, and new questions add coordinates without altering existing ones.   On public product benchmarks, question-space retrieval matches dense candidate generation statistically in a retrieve-then-verify entity matcher, while providing exact explanations 
    
[^10]: 双层情感框架：网络搜索中可扩展的多模态用户体验归因

    The Dual-Tier Sentiment Framework: Scalable Multimodal UX Attribution in Web Search

    [https://arxiv.org/abs/2611.27899](https://arxiv.org/abs/2611.27899)

    该论文提出了一种双层情感框架，第一层利用微调的编码器-解码器大语言模型实现大规模用户情感回归检测（PR-AUC提升9.8%、测量延迟从数周降至数小时、问卷开销降低40%以上），第二层提供细粒度的UX归因诊断，从而实现网络搜索中可扩展的自动化用户体验评估。

    

    在大规模网络搜索中维持用户体验（UX）质量需要可扩展的评估方法。传统问卷调查速度缓慢，成为快速A/B测试的瓶颈，而自动化指标往往缺乏诊断视觉空间UX异常所需的粒度。为此，我们提出了双层情感框架（Dual-Tier Sentiment Framework），这是一种用于自动化UX归因的新型工业架构。第一层（宏观系统）作为可扩展的用户情感回归检测器，在遥测特征上利用微调的编码器-解码器大语言模型（基于Gemini XS）。通过多任务正则化和上下文特征工程，该层在超过50万条留存记录上实现了+9.8%的PR-AUC增益，在所有已验证的回归中与基线召回率持平，并在12个留存实验中实现了约20%的相对精确率提升。在生产环境中，该系统将测量延迟从数周缩短至数小时，并将问卷开销降低超过40%。第二层（微观系统）提供细粒度的诊断分析。

    arXiv:2611.27899v1 Announce Type: new  Abstract: Sustaining user experience (UX) quality in large-scale web search requires scalable evaluation. Traditional surveys are slow and bottleneck rapid A/B testing, while automated metrics often lack the granularity to diagnose visual-spatial UX anomalies. To address this, we introduce the Dual-Tier Sentiment Framework, a novel industrial architecture for automated UX attribution. Tier 1 (Macro System) acts as a scalable user sentiment regression detector, utilizing a fine-tuned encoder-decoder LLM (based on Gemini XS) on telemetry features. Through multi-task regularization and contextual feature engineering, this tier achieves a +9.8% PR-AUC gain on over 500,000 held-out records, matching baseline recall on all validated regressions with $\sim$20% relative precision gain among the 12 held-out experiments. In production, it reduces measurement latency from weeks to hours and cuts survey overhead by over 40%. Tier 2 (Micro System) provides gra
    
[^11]: TEA-Bench-BR：编码智能体能否将分支化的临床规范转化为正确的代码？

    TEA-Bench-BR: Can Coding Agents Turn a Branching Clinical Specification into Correct Code?

    [https://arxiv.org/abs/2611.27895](https://arxiv.org/abs/2611.27895)

    该论文提出了TEA-Bench-BR——一个基于巴西早期自闭症筛查流程、以葡萄牙语规范编写的多层级代码生成基准，用以严格评测编码智能体将分支化临床规范转化为正确代码的能力，结果揭示了大模型在复杂任务层级和输入验证方面的显著局限。

    

    编码智能体越来越多地被用于实现由领域专家编写的规则，而这类代码中的缺陷是“静默”的：代码可以运行并返回看似合理的答案。我们提出了TEA-Bench-BR，这是一个基于巴西早期自闭症（TEA）筛查路径构建的基准：包含三个结构仿照真实量表的虚构评估工具、一个多专业转诊流程，以及导出为采用国家RNDS配置文件的FHIR R4格式，全部规范均以葡萄牙语编写。一个生成器对每个任务的参数进行采样（共144个任务、6,798个隐藏测试用例），这些隐藏用例能够杀死参考实现的全部1,955个非等价变异体。以Gemma 4作为编码智能体，12B模型解决了38.0%的单量表任务，而在FHIR层面则无一解决；31B模型在第1至第3级的解决率分别为77.8%、85.2%和78.7%，并通过了超过99%的隐藏用例，其剩余的失败大多出现在输入验证而非临床决策方面。当相同的任务以英语呈现时，31B模型在第3级的解决率降至57.4%，因为其代码验证……（摘要在此处被截断）

    arXiv:2611.27895v1 Announce Type: cross  Abstract: Coding agents increasingly implement rules written by domain experts, and bugs in such code are silent: it runs and returns a plausible answer. We introduce TEA-Bench-BR, a benchmark built on the Brazilian pathway for early autism (TEA) screening: three fictitious instruments with the structure of real ones, a multi-professional referral flow, and export to FHIR R4 with the national RNDS profiles, all specified in Portuguese. A generator samples each task's parameters (144 tasks, 6,798 hidden cases), and the hidden cases kill all 1,955 non-equivalent mutants of the reference. With Gemma 4 as the agent, the 12B solves 38.0% of single-instrument tasks and none at the FHIR level; the 31B solves 77.8%, 85.2% and 78.7% at levels 1 to 3 and passes over 99% of the hidden cases, its remaining failures mostly in input validation rather than clinical decisions. With the same tasks in English, the 31B solves 57.4% at level 3, as its code validate
    
[^12]: 投机解码没有你想象的那么“投机”

    Speculative Decoding Is Less Speculative Than You Think

    [https://arxiv.org/abs/2611.27891](https://arxiv.org/abs/2611.27891)

    本文提出DICE方法，通过预先采样并在草稿生成与验证阶段之间同步采样随机性，将有限的草稿计算集中于更可能被目标模型实际采用的候选续写，从而提升投机解码的推理效率。

    

    投机解码通过在目标模型生成之前预先准备好候选续写内容来加速语言模型推理。其效率取决于有限的草稿计算能在多大程度上预测目标模型实际将走的路径。预测这条路径涉及两个不确定性来源：目标模型的预测以及采样的随机性。然而，前者需要目标模型的前向传播计算，而后者可以在草稿生成之前就被确定下来。基于这一观察，我们提出了DICE（基于抽取信息的候选选择，Draw-Informed Candidate Selection），它预先采样并在草稿生成与验证之间同步验证随机性。通过将这些已知的抽取结果与现有草稿模型的预测相结合，DICE将计算资源分配给更可能在当前采样运行中被使用的续写内容。这一原则贯穿草稿树构建过程中的节点选择，并扩展到异步预取场景中。

    arXiv:2611.27891v1 Announce Type: new  Abstract: Speculative decoding accelerates language-model inference by preparing candidate continuations before the target model generates them. Its efficiency depends on how well limited drafting computation anticipates the path the target will actually follow. Anticipating this path involves two sources of uncertainty: the target model's predictions and the randomness of sampling. Yet the former requires a target-model forward pass, while the latter can be fixed before drafting. Building on this observation, we introduce DICE (Draw-Informed Candidate Selection), which pre-samples and synchronizes verification randomness across drafting and verification. By combining these known draws with predictions from an existing drafter, DICE allocates computation toward continuations more likely to be used in the current sampling run. This principle guides node selection throughout draft-tree construction and extends to asynchronous prefetching, where DICE
    
[^13]: LesionAgent：纵向CT中无需训练的智能体式肿瘤追踪与疗效评估

    LesionAgent: Training-Free Agentic Tumor Tracking and Response Assessment in Longitudinal CT

    [https://arxiv.org/abs/2611.27876](https://arxiv.org/abs/2611.27876)

    该论文提出无需训练的LLM智能体LesionAgent，像放射科医生一样调用冻结的3D基础模型并对照临床指南，在纵向CT中实现端到端的肿瘤追踪与疗效评估，追踪数量和测量精度均超越所有自动化基线。

    

    每隔几个月，癌症患者的CT扫描结果就决定着治疗是否继续。3D基础模型如今已能检测、分割、配准和追踪病灶，但它们都无法像放射科医生那样阅读扫描图像。放射科医生会判断哪些发现是病灶、哪些是几个月后的同一病灶、新出现的斑点是否意味着病情进展，并对照临床指南中的大量规则核验每一项判断。我们提出LesionAgent——一个无需任何任务特定训练即可端到端完成这种阅片工作的LLM智能体。它将冻结的3D基础模型作为工具来调用，通过带坐标标注的视图在三维空间中进行操作，在登记册中记住跨随访的每一个病灶，验证它所依赖的每一个模型输出，并将所有数值测算交给确定性的指南引擎来完成。在四个公开的纵向CT数据集上，LesionAgent比所有自动化基线追踪到更多病灶且测量更准确；而将同样的3D模型用固定流程串联的方法（原文在此处截断）……

    arXiv:2611.27876v1 Announce Type: cross  Abstract: Every few months, a cancer patient's CT scans decide whether treatment continues. 3D foundation models can now detect, segment, register and track lesions, yet none of them reads the scans as a radiologist does. A radiologist decides which findings are lesions, which are the same lesion months later and whether a new spot means progression, and checks each decision against the extensive rules of a clinical guideline. We introduce LesionAgent, an LLM agent that performs this read end to end without task-specific training. It operates frozen 3D models as tools, acts in 3D through coordinate-labelled views, remembers every lesion across visits in a registry, verifies every model output it relies on, and leaves every number to a deterministic guideline engine. On four public longitudinal CT datasets, LesionAgent tracks more lesions and measures them more accurately than every automatic baseline, and where the same 3D models chained by fixe
    
[^14]: 超越视觉保真度：比较类别型与连续型世界模型

    Beyond Visual Fidelity: Comparing Categorical and Continuous World Models

    [https://arxiv.org/abs/2611.27870](https://arxiv.org/abs/2611.27870)

    该论文通过受控的2D和3D环境实验系统比较了类别型与连续型预测对世界模型动态学习的影响，发现在无损视觉编码条件下，类别型模型能以更少训练计算量获得更高的动态rollout有效性和更好的随机性校准，表明视觉逼真度并不等同于动态保真度。

    

    视觉世界模型如今已能生成引人入胜的交互式世界，但视觉上的逼真并不保证动态的真实可信。我们系统地比较了在下一token预测和迭代式整帧生成两种范式下，类别型预测与连续型预测如何塑造模型学习到的动态。这些选择通常在自回归token模型和整帧流匹配等方法中被捆绑在一起。我们设计了受控的2D和3D环境，并评估动态有效性、校准性和视觉质量。在具有无损视觉编码的2D环境中，类别型模型以更少的训练计算量获得更高的rollout有效性，尤其体现在随机性实体上。在类别型模型中，贪心解码产生最高的rollout有效性，而从下一token模型中采样则得到校准最佳的随机预测。连续模型的失败案例（如游戏角色消失）与在更复杂环境中观察到的失败情况相呼应。

    arXiv:2611.27870v1 Announce Type: cross  Abstract: Visual world models can now generate compelling interactive worlds, but visual realism does not guarantee faithful dynamics. We systematically compare how categorical and continuous prediction shape learned dynamics under both next-token and iterative whole-frame generation. These choices are often bundled in approaches such as autoregressive token models and whole-frame flow matching. We design controlled 2D and 3D environments and evaluate dynamics validity, calibration, and visual quality. In 2D with lossless visual encoding, categorical models achieve higher rollout validity with less training compute, particularly for stochastic entities. Among categorical models, greedy decoding produces the highest rollout validity, while sampling from a next-token model yields the best-calibrated stochastic predictions. Failures in continuous models, such as disappearing game characters, echo those seen in more complex environments. Here, known
    
[^15]: 将生成式人工智能与学习风格模型相整合：机遇、挑战与未来方向

    Integrating Generative Artificial Intelligence with Learning Style Models: Opportunities, Challenges, and Future Directions

    [https://arxiv.org/abs/2611.27868](https://arxiv.org/abs/2611.27868)

    本综述系统梳理了41项研究，探讨生成式人工智能与VARK、Kolb等经典学习风格模型整合以支持大型在线课堂学生参与的机遇、挑战与未来方向，并指出AI可实现灵活、多模态、实时的个性化学习适应。

    

    生成式人工智能已成为变革学习的强大工具。本综述探讨了如何将生成式人工智能与成熟的学习风格模型（VARK、Kolb、Felder-Silverman 以及 Honey & Mumford）相整合，以支持大型在线课堂中的学生参与。研究遵循 PRISMA 2020 指南，检索了 Scopus、IEEE Xplore 和 Google Scholar（2015至2026年）数据库，最终纳入41项符合资格标准的研究。该综述将生成式人工智能的能力与学习风格偏好进行映射，综合了现有的实施案例，突出了其益处与挑战，并为未来的研究与实践提出了路径。尽管关于学习风格实证有效性的争论仍在持续，生成式人工智能能够实现灵活、多模态且实时的适应性学习。本研究为寻求构建更有效、以学习者为中心的学习系统的教育者和开发者提供了实践性见解。

    arXiv:2611.27868v1 Announce Type: new  Abstract: Generative Artificial Intelligence (Generative AI) has emerged as a powerful tool for transforming learning. This review examines how Generative AI can be integrated with established learning style models (VARK, Kolb, Felder-Silverman, and Honey & Mumford) to support student engagement in large online classes. Following PRISMA 2020 guidelines, we searched Scopus, IEEE Xplore, and Google Scholar (2015 to 2026) and included 41 studies that met the eligibility criteria. The review maps Generative AI capabilities to learning style preferences, synthesizes existing implementations, highlights benefits and challenges, and proposes pathways for future research and practice. Despite ongoing debates regarding the empirical validity of learning styles, Generative AI enables flexible, multimodal, and real-time adaptation. This study offers practical insights for educators and developers seeking to build more effective, learner centered learning sys
    
[^16]: SLICR：策略逻辑引导的多智能体强化学习

    SLICR: Strategy Logic-Guided Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.27865](https://arxiv.org/abs/2611.27865)

    提出SLICR框架，通过策略逻辑将规范引导强化学习首次推进到部分可观测的对抗性多智能体环境，不仅能推理智能体自身如何满足规范，还能推理如何应对其他智能体的策略和行为。

    

    规范引导的强化学习将智能体期望行为的形式化规范转化为整形奖励。本文提出了SLICR，这是一次重大的范式转变，将规范引导强化学习的前沿推进到了部分可观测的对抗性环境中。我们采用策略逻辑（SL），它允许对智能体的策略进行显式量化，并使用线性时序逻辑（LTL）来指定智能体的目标与约束。这使得我们不仅能够推理智能体应如何行动以满足规范，还能推理它们应如何应对其他智能体的策略和行为。我们在三个流行的基准测试上评估了SLICR：捕食者-猎物、Google Research Football以及我们自主开发的SMAC-HARD扩展版本，证明了其有效性、可扩展性以及适应多样化对抗策略的能力。

    arXiv:2611.27865v1 Announce Type: new  Abstract: Specification-guided reinforcement learning (RL) translates formal specifications of desired behavior of agents into shaped rewards. This paper introduces SLICR, a significant paradigm shift that advances the state of the art to specification-guided RL in adversarial settings under partial observability. We employ Strategy Logic (SL), which allows explicit quantification over agents' strategies and uses the linear temporal logic (LTL) to specify the agents' objectives and constraints. This enables reasoning not only about how agents should act to satisfy a specification, but also about how they should respond to the strategies and behaviors of other agents. We evaluate SLICR on three popular benchmarks: Predator-Prey, Google Research Football, and our home-grown extension of SMAC-HARD, demonstrating its effectiveness, scalability, and ability to adapt to diverse adversarial strategies.
    
[^17]: 谁来为智能体评分？面向编码智能体基准测试的权限分离机制

    Who Grades the Agent? Privilege Separation for Coding-Agent Benchmarks

    [https://arxiv.org/abs/2611.27860](https://arxiv.org/abs/2611.27860)

    本文提出 SplitBench，这是首个采用权限分离机制、将编码智能体基准测试中“运行测试”与“判定结果”相隔离的评估框架，可防止智能体操纵评估过程来骗取虚高的分数。

    

    基于人工智能的编码智能体基准测试是当前人工智能领域的重要组成部分，因为它们是衡量大语言模型性能的主要手段，其分数指导着商业和研究决策。然而，评估编码智能体极具挑战性，因为它们能够操纵评估基础设施来人为地获取更高的分数。对智能体的解决方案进行评分通常需要运行其代码，而这段代码随后便可能操纵评估过程。自动化基准审计工具 BenchJack 最近在十个流行的智能体基准测试中的九个上获得了接近满分的成绩，而实际上并未真正解决任何任务。正如 OpenAI 2026 年 7 月的 Hugging Face 事件所显示的，智能体已经在无人指示的情况下自行实施这类操纵行为。借鉴安全系统的设计理念，我们提出了 SplitBench，一个用于评估编码智能体的权限分离框架，这也是科学文献中首个将“运行任务测试”与“判定测试结果”相分离的框架。即使智能体……

    arXiv:2611.27860v1 Announce Type: cross  Abstract: Benchmarks for AI-based coding agents are a staple of the current AI landscape, because they are the leading way to measure the performance of LLMs. Their scores guide business and research decisions. Evaluating coding agents is challenging, since they are able to manipulate their evaluation infrastructure to artificially achieve better scores. Grading an agent's solution often requires running its code, which can then manipulate the evaluation process. BenchJack, an automated benchmark auditor, recently reached near-perfect scores on nine of ten popular agent benchmarks without actually solving their tasks. Agents already do this unprompted, as OpenAI's July 2026 Hugging Face incident showed.   Building on secure systems design, we introduce SplitBench, a privilege separation framework for evaluating coding agents and the first in the scientific literature to separate running a task's tests from judging their results. Even an agent wh
    
[^18]: 超越自我批判：面向多智能体研究验证的认知收敛控制

    Beyond Self-Critique: Epistemic Convergence Control for Multi-Agent Research Verification

    [https://arxiv.org/abs/2611.27839](https://arxiv.org/abs/2611.27839)

    该论文提出一个多智能体研究验证框架，通过倡导者-怀疑者-仲裁者的非对称角色架构从结构上消除同源模型自我批判偏差，并利用信念差分收敛控制替代固定迭代协议，从而提升科学论断验证的可靠性与计算效率。

    

    arXiv:2611.27839v1 公告类型：新 摘要：用于科学论断验证的多智能体系统越来越依赖角色结构化的LLM辩论来对假设进行压力测试。然而，当辩论智能体由同一底层基础模型实例化时，它们共享相同的潜在表征，使得自我批判偏差在结构上不可避免。静态固定迭代协议进一步加剧了这一问题——要么在证据稳定前过早停止，要么在收敛后浪费推理计算资源。我们提出了\Sys{}，一个同时解决上述两种失效模式的多智能体研究验证框架。首先，我们引入了一种非对称角色架构，由持有对立认知指令的倡导者、怀疑者和仲裁者组成，产生显著更低的角色嵌入相似度（0.852，对比对称多智能体辩论[MAD]的0.918，p = 5.82 × 10⁻⁸），并揭示出更多存在争议的论断（22.0%，对比MAD的15.4%）。其次，我们提出了信念差分（Belief-Delta, ΔB）收敛……（原文摘要在此处截断）

    arXiv:2611.27839v1 Announce Type: new  Abstract: Multi-agent systems for scientific claim verification increasingly rely on role-structured LLM debate to pressure-test hypotheses. However, when debating agents are instantiated from the same underlying base model, they share identical latent representations, making self-critique bias structurally inevitable. Static fixed-iteration protocols compound this by either halting prematurely before evidence stabilizes or wasting inference compute after convergence. We present \sys{}, a multi-agent research verification framework that addresses both failure modes. First, we introduce an asymmetric role architecture comprising an Advocate, Skeptic, and Arbitrator with opposing epistemic directives, producing significantly lower role-embedding similarity (0.852 vs.\ 0.918 for symmetric Multi-Agent Debate [MAD], $p = 5.82 \times 10^{-8}$) and surfacing more contested claims (22.0\% vs.\ 15.4\%). Second, we propose the Belief-Delta ($\DB$) convergen
    
[^19]: 可审计的金融知识图谱：基于模式驱动的构建与内在质量评估

    Auditable Financial Knowledge Graphs: Schema-Driven Construction and Intrinsic Quality Evaluation

    [https://arxiv.org/abs/2611.27835](https://arxiv.org/abs/2611.27835)

    该论文提出了一种可审计的模式驱动方法，通过分阶段大语言模型提取、多策略实体消解、时间约束与细粒度溯源，从金融文档构建时序知识图谱并对其内在质量进行评估。

    

    金融文档融合了叙述性披露、会计指标、公司结构、管理层评论以及时间相关的事实信息，这使得传统的基于分块的检索增强生成系统难以对其进行表示。本文提出了一种可审计的、模式驱动的方法论，用于从经通用文档智能服务解析的按页分隔文档中构建时序金融知识图谱。通过重叠的文档窗口，分阶段利用大语言模型提取结构化实体、定量观测值以及领域特定关系。候选图谱元素依据预定义模式进行验证，并通过范围感知的精确匹配、规范化、模糊匹配、基于嵌入以及大语言模型辅助的实体消解进行整合。时间合并约束、确定性边去重、冲突感知的属性管理以及细粒度的溯源机制保留了发行方身份……

    arXiv:2611.27835v1 Announce Type: new  Abstract: Financial documents combine narrative disclosures, accounting metrics, corporate structures, management commentary, and time-dependent facts, making them difficult to represent using conventional chunk-based retrieval-augmented generation systems. This paper presents an auditable, schema-driven methodology for constructing temporal financial knowledge graphs from page-delimited documents parsed using a general purpose document intelligence service. Overlapping document windows undergo phased large language model extraction of structural entities, quantitative observations, and domain-specific relationships. Candidate graph elements are validated against predefined schemas and consolidated through scope-aware exact, normalized, fuzzy, embedding-based, and LLM-assisted entity resolution. Temporal merge constraints, deterministic edge deduplication, conflict-aware property management, and fine-grained provenance preserve issuer identity, re
    
[^20]: 基于图感知扩散模型的图信号生成

    Graph Signal Generation via Graph-Aware Diffusion Models

    [https://arxiv.org/abs/2611.27826](https://arxiv.org/abs/2611.27826)

    提出图感知扩散模型（GAD），将前向过程建模为图上的热扩散，并设计模式自适应信噪比调度（MASS），为每个图频率分配独立扩散速率以保持相同信噪比，从而实现更好的图信号生成与分布学习。

    

    许多信号定义在图的节点上，例如道路传感器处的交通流量或用户对相关产品的购买。这些信号通常只被部分观测到，学习其分布可以实现对缺失值的插补以及对未来值的预测。扩散模型能够学习这类分布，但其前向噪声在各节点之间是相互独立的，并不依赖于图结构。我们提出图感知扩散（GAD），其前向过程是由高斯噪声驱动的图上热扩散。在图傅里叶基下，前向转移在每个频率上表现为一个标量高斯信道。在共享扩散速率的情况下，高图频率比低图频率衰减得更快。我们提出模式自适应信噪比调度（MASS），为每个图频率分配各自的扩散速率，并使所有频率保持相同的信噪比。对于图平稳信号，MASS 能够使最优线性去噪器在各频率上的归一化误差相等。

    arXiv:2611.27826v1 Announce Type: cross  Abstract: Many signals are defined on the nodes of a graph, such as traffic flow at road sensors or user purchases of related products. They are often partly observed, and learning their distribution enables imputing missing values and forecasting future ones. Diffusion models learn such distributions, but their forward noise is independent across nodes and does not depend on the graph. We introduce graph-aware diffusion (GAD), whose forward process is a heat diffusion on the graph driven by Gaussian noise. In the graph Fourier basis, the forward transition is a scalar Gaussian channel at each frequency. With a shared diffusion rate, high graph frequencies are attenuated faster than low ones. We propose Mode-Adaptive SNR Scheduling (MASS), which assigns each graph frequency its own diffusion rate and keeps all of them at the same SNR. For graph-stationary signals, MASS equalizes the normalized error of the optimal linear denoiser across frequenc
    
[^21]: 重新设计神经网络：即使在强对抗攻击下也不会“自信地犯错”

    Redesigning Neural Networks to Not be Confidently Wrong even to Strong Adversarial Attacks

    [https://arxiv.org/abs/2611.27811](https://arxiv.org/abs/2611.27811)

    该论文提出了一种模拟生物视觉皮层简单细胞与复杂细胞交替结构的浅层神经网络架构，通过能够学习选择性与容忍性表征的新型组件，实现了准确、鲁棒且与人类感知更一致的图像分类，并能有效抵御强对抗攻击，避免对未知类别样本做出高置信度的错误预测。

    

    早期的深度学习模型试图通过一种神经层级结构来模拟视觉感知，该结构在两种类型的层之间交替：一种模拟皮层简单细胞（旨在提高选择性），另一种模拟皮层复杂细胞（旨在提高容忍度）。我们证明，这种神经网络（NN）架构在采用能够更好学习选择性与容忍性表征的新型组件实现后，能够实现准确且鲁棒的图像分类，并改善了模型行为与人类感知能力之间的一致性。首先，所提出的模型是浅层的，层数少于十对，这更符合生物视觉系统所执行的快速前馈目标分类机制。其次，这些模型远不容易对来自未知类别的样本或图像不一致的情况产生高置信度的分类预测……

    arXiv:2611.27811v1 Announce Type: cross  Abstract: Early deep learning models attempted to simulate visual perception using a neural hierarchy that alternated between two types of layer: one modelled on cortical simple cells (that was designed to increase selectivity), and one modelled on cortical complex cells (that aimed to increase tolerance). We show that such a neural network (NN) architecture, when implemented with novel components that are better able to learn selective and tolerant representations, is capable of accurate and robust image classification and improved alignment between the behaviour of the model and the perceptual abilities of humans. Firstly, the proposed models are shallow, fewer than ten layer-pairs, which is more consistent with the rapid feedforward object categorisation performed by the biological visual system. Secondly, they are far less prone to producing classification predictions with high confidence for samples from unknown classes or for image consist
    
[^22]: SLP-Bench：评估与改进推理型音频语言模型在临床语音上的声学论断

    SLP-Bench: Evaluating and Improving Acoustic Claims of Reasoning Audio Language Models on Clinical Speech

    [https://arxiv.org/abs/2611.27810](https://arxiv.org/abs/2611.27810)

    提出SLP-Bench基准，将推理型音频语言模型思维链中的每一项声学论断与临床医生的感知评分逐条对照，评估其事实性、稳定性和一致性，结果显示十个开源模型的论断平衡准确率仅为0.52至0.58。

    

    音频语言模型能够根据语音回答临床问题，其表现优于基于手工设计声学特征的基线方法。推理型音频语言模型在给出预测之前，会生成一条涉及说话人嗓音属性的思维链。模型对嗓音做出的每一项陈述，都需要与临床医生对同一录音的感知评分进行核对。然而在临床语音领域，现有评估方法仅对最终答案进行打分。我们提出了言语语言病理学家基准，这是一个将思维链中的每一项声学论断与临床医生对同一录音的感知评分进行逐条对照评分的基准。该基准涵盖五个由临床医生评分的数据集和25个感知维度，并从事实准确性、重采样下的稳定性，以及改写问题与不同问题之间的一致性等方面评估这些论断。在对十个开源权重推理型音频语言模型进行零样本评估中，论断的平衡准确率介于0.52至0.58之间，论断会在……（原文摘要在此处截断）

    arXiv:2611.27810v1 Announce Type: new  Abstract: Audio language models (ALMs) can answer clinical questions from speech, outperforming baselines built on handcrafted acoustic features. Reasoning ALMs generate a chain of thought that refers to properties of the speaker's voice before predicting. Each statement the model makes about the voice needs to be checked against clinicians' perceptual ratings of the same recording. On clinical speech, existing evaluations score only the final answer. We introduce Speech Language Pathologist Bench (SLP-Bench), a benchmark grading each acoustic claim in the chain of thought against clinicians' perceptual ratings of the same recording. It covers five clinician-rated datasets and 25 perceptual dimensions, and assesses claims for factuality, stability under resampling, and consistency across reworded and different questions. Across ten open-weight reasoning ALMs evaluated zero-shot, claim balanced accuracy ranges from 0.52 to 0.58, claims change under
    
[^23]: $\mathtt{DistP}$：通过零样本超参数迁移扩展本地更新训练

    $\mathtt{DistP}$: Scaling Local-Update Training with Zero-Shot Hyperparameter Transfer

    [https://arxiv.org/abs/2611.27808](https://arxiv.org/abs/2611.27808)

    提出理论框架$\mathtt{DistP}$，将$\mathtt{DDP}$的零样本超参数迁移规则扩展到本地更新训练，使学习率能够跨模型宽度、深度和同步间隔直接迁移，从而免去昂贵的超参数调优。

    

    本地更新方法为分布式数据并行（$\mathtt{DDP}$）提供了一种通信高效的替代方案，但引入了需要昂贵调优的额外超参数。对于$\mathtt{DDP}$，零样本超参数迁移通过复用在较小模型上调优的设置来降低这一成本，但其在本地更新训练中的适用性仍不明确。我们提出了$\mathtt{DistP}$，一个理论框架，将现有的迁移规则从$\mathtt{DDP}$扩展到具有不同工作节点数量、本地步数和外层优化器的本地更新训练。当使用平均作为外层优化器时，$\mathtt{DistP}$支持学习率在宽度和深度上的迁移，同时提升模型质量。调优后的内层学习率还可以直接在不同同步间隔之间迁移，但在增加全局批量大小（batch size）时遵循两种规律：对于较大的本地批量采用平方根学习率缩放，对于其他情况则采用固定的参考学习率（原文此处截断）。

    arXiv:2611.27808v1 Announce Type: cross  Abstract: Local-update methods offer a communication-efficient alternative to Distributed Data Parallelism ($\mathtt{DDP}$), but introduce additional hyperparameters that require costly tuning. For $\mathtt{DDP}$, zero-shot hyperparameter transfer reduces this cost by reusing settings tuned on smaller models, but its applicability to local-update training remains unclear. We propose $\mathtt{DistP}$, a theoretical framework extending existing transfer rules from $\mathtt{DDP}$ to local-update training with varying worker counts, local steps, and outer optimizers. With averaging as the outer optimizer, $\mathtt{DistP}$ supports learning-rate transfer across width and depth while improving model quality. The tuned inner learning rate also directly transfers across synchronization intervals, but follows two regimes when increasing the global batch size: square-root learning-rate scaling for larger local batches and a fixed reference rate for additi
    
[^24]: ChildAI-Gov：一个证据关联的AI系统，用于依据儿童数字权利评估教育科技政策

    ChildAI-Gov: An Evidence-Linked AI System for Assessing EdTech Policies Against Children's Digital Rights

    [https://arxiv.org/abs/2611.27799](https://arxiv.org/abs/2611.27799)

    该论文提出ChildAI-Gov系统，通过儿童权利影响评分（CRIS）结合大语言模型判断、规则检查与检索增强生成技术，将教育科技政策的合规评估结果与儿童权利框架的证据条款关联起来，实现可追溯、自动化的儿童数字权利评估。

    

    我们提出了ChildAI-Gov，这是一个可解释的AI系统，用于评估教育科技政策是否尊重儿童的数字权利。生成式AI进入儿童教育领域的速度已超过其治理体系所能跟上的速度，然而对照儿童权利来审查一款工具目前仍依赖人工、依赖专家且进展缓慢。我们提出了儿童权利影响评分（CRIS），这是一项与证据关联的衡量指标，涵盖五个维度：安全性、隐私、透明度、自主性和非歧视。ChildAI-Gov读取一款教育科技工具的隐私或AI政策，通过将大语言模型（gpt-4o-mini）的判断与确定性规则检查相结合，对每个维度进行评分。检索增强生成（RAG）组件将每个评分锚定于治理条款，这些条款从精心整理的儿童权利框架知识库（如欧盟AI法案、联合国儿童基金会指南等）中检索获得，因此每个评分都与背后的证据相关联，审计人员可以追溯……

    arXiv:2611.27799v1 Announce Type: new  Abstract: We present ChildAI-Gov, an explainable AI system for assessing whether education-technology (EdTech) policies respect children's digital rights. Generative AI is reaching children's education faster than its governance can keep pace, yet checking a tool against children's rights remains manual, expert-dependent, and slow. We propose the Child Rights Impact Score (CRIS), an evidence-linked measure across five dimensions: safety, privacy, transparency, autonomy, and non-discrimination. ChildAI-Gov reads an EdTech tool's privacy or AI policy and scores each dimension by combining a large-language-model judgement (gpt-4o-mini) with a deterministic rule check. A retrieval-augmented generation (RAG) component grounds every score in the governing provisions, retrieved from a curated knowledge base of children's rights frameworks (the EU AI Act, UNICEF guidance, and others), so each score is tied to the evidence behind it and an auditor can trac
    
[^25]: Neural Defend ARCAS 290M：一个用于AI生成音频检测的鲁棒基础模型

    Neural Defend ARCAS 290M: A Robust Foundation Model for AI-Generated Audio Detection

    [https://arxiv.org/abs/2611.27784](https://arxiv.org/abs/2611.27784)

    ARCAS 290M是一个冻结的基础模型，无需针对特定基准微调，即可在跨生成器类型、信道条件和语言的多种场景下实现最先进的AI生成音频检测。

    

    声音克隆现在只需要几秒钟的音频，而针对单一数据集调优的检测器在生成器、信道或语言发生变化时往往失效。我们推出了ARCAS 290M，这是Neural Defend公司用于检测AI生成音频的基础模型。作为一个单一的冻结生产模型，无需针对任何基准进行专门微调，它能够泛化到文本转语音、语音转换和编解码器语言模型等各类生成器，适应电话和蓝牙信道、各种录音条件以及不同语言。我们在十个公开基准的完整官方测试集上对其进行评估，对每个录音完全按发布时的原始状态进行评分，并在独立排行榜上与商业系统进行比较。ARCAS 290M在Podonos排行榜的所有系统中排名第一，并在ASVspoof5上拥有商业检测器中最低的等错误率。它在ASVspoof5、SpeechFake和ShiftySpeech数据集上几乎完美地区分真实语音和AI生成语音，并能检测出几乎所有的AI生成印地语语音。

    arXiv:2611.27784v1 Announce Type: cross  Abstract: Voice cloning now needs only seconds of audio, and detectors tuned to one dataset often fail when the generator, channel, or language changes. We introduce ARCAS 290M, Neural Defend's foundation model for detecting AI-generated audio. A single frozen production model, with no benchmark-specific fine-tuning, generalizes across text-to-speech, voice-conversion, and codec-language-model generators, telephone and Bluetooth channels, recording conditions, and languages. We evaluate it on the complete official test sets of ten public benchmarks, scoring every recording exactly as released, and against commercial systems on independent leaderboards. ARCAS 290M ranks first among all systems on the Podonos leaderboard and has the lowest equal error rate among commercial detectors on ASVspoof5. It separates real from AI-generated speech almost perfectly on ASVspoof5, SpeechFake, and ShiftySpeech, and detects nearly all AI-generated speech in Hin
    
[^26]: 面向开放集脑电图癫痫发作类型分类的基准协议与公平性分析

    A Benchmark Protocol and Fairness Analysis Towards Open-Set EEG Seizure Type Classification

    [https://arxiv.org/abs/2611.27783](https://arxiv.org/abs/2611.27783)

    该论文提出了首个基于 TUSZ 数据集的脑电图癫痫发作类型分类开放集识别基准协议，利用临床分类体系定义已知与未知发作类型，并在部分微调的脑电图基础模型上系统评估了七种开放集评分方法及其公平性。

    

    自动脑电图（EEG）癫痫发作类型分类系统通常在闭集假设下进行训练和评估，即部署时遇到的所有癫痫发作类型在训练阶段均已见过的。在临床实践中，这一假设并不成立：患者可能出现训练标签空间之外的罕见或非典型发作类型，而一个强行给出错误标签的系统比一个选择弃权（拒识）的系统更糟糕。我们在天普大学医院癫痫语料库（TUSZ v2.0.3）上首次引入了针对脑电图癫痫发作类型分类的开放集识别协议，并利用临床分类体系本身来定义已知类与未知类：具有足够受试者代表性（至少12名受试者）的发作类型构成已知类集合，罕见类型构成未知类集合。我们在一个部分微调的脑电图基础模型（EFM）上评估了七种事后开放集评分方法（MSP、MaxLogit、energy、Mahalanobis、kNN、GHOST 和 COSTARR），并采用基于受试者的……

    arXiv:2611.27783v1 Announce Type: new  Abstract: Automated EEG seizure type classification systems are typically trained and evaluated under the closed-set assumption, where all seizure types encountered at deployment are seen during training. In clinical practice, this assumption fails: patients present with rare or atypical seizure types that fall outside the training label space, and a system that forces a wrong label is worse than one that abstains. We introduce the first open-set recognition protocol for EEG seizure type classification on the Temple University Hospital Seizure Corpus (TUSZ v2.0.3), defining known and unknown classes using the clinical taxonomy itself: seizure types with sufficient subject representation (at least 12 subjects) form the known set, and rare types form the unknown set. We evaluate seven post-hoc open-set scorers (MSP, MaxLogit, energy, Mahalanobis, kNN, GHOST, and COSTARR) applied to a partially fine-tuned EEG Foundation Model (EFM), using a subject-d
    
[^27]: Search-RSI：沿多模态智能体能力边界扩展搜索世界

    Search-RSI: Expanding Search Worlds Along Multimodal Agents' Capability Frontier

    [https://arxiv.org/abs/2611.27780](https://arxiv.org/abs/2611.27780)

    提出Search-RSI递归自我改进框架，让多模态搜索智能体诊断自身能力边界，并同步生长新任务及其证据，从而解决因检索后端缺少证据而累积的不可达负样本问题。

    

    多模态搜索智能体通过将知识密集型的视觉问题锚定在检索到的证据上来进行回答，近来的训练方法在从知识图谱合成、并在线增长以跟上智能体水平的多跳任务上训练它们。然而，这些方法将可搜索的世界视为固定的：一个新问题可以是良构且可验证的，但它的证据却不在检索后端中，于是智能体发出了正确的查询却检索不到任何内容，并被当作策略失误而受到惩罚。这种不可达的负样本会随着每一轮增长不断累积。我们提出Search-RSI，一个沿着智能体能力边界扩展搜索世界的递归自我改进框架。在每一轮中，当前策略诊断其在由跳数深度、关系稀有度和干扰项密度所张成的任务空间中部分掌握的区域；在该区域新任务与其证据一同生长；同一策略……

    arXiv:2611.27780v1 Announce Type: cross  Abstract: Multimodal search agents answer knowledge intensive visual questions by grounding them in retrieved evidence, and recent recipes train them on multi hop tasks that are synthesized from knowledge graphs and grown online to keep pace with the agent. These recipes, however, treat the searchable world as fixed: a new question can be well formed and verifiable while its evidence is absent from the retrieval backend, so the agent issues the right query, retrieves nothing, and is penalized as if its policy were at fault. Such unreachable negatives accumulate with every round of growth. We propose Search-RSI, a recursive self improvement framework that expands the search world along the agent's capability frontier. In each round, the current policy diagnoses its partially mastered regions of a task space spanned by hop depth, relation rarity, and distractor density; new tasks there are grown together with their evidence; and the same policy de
    
[^28]: QuASAR：迈向一种用于筛选、原型设计与评估科学计算中量子优势机会的LLM辅助框架

    QuASAR: Toward an LLM-Assisted Framework for Screening, Prototyping, and Assessing Quantum-Advantage Opportunities in Scientific Computing

    [https://arxiv.org/abs/2611.27778](https://arxiv.org/abs/2611.27778)

    QuASAR提出了一种由两个LLM智能体组成的实验性流水线，通过筛选文献中具有量子算法对应物的计算内核、独立验证引文、构建量子映射原型并与经典基线进行基准对比，来评估科学计算中的量子优势机会，在天体物理学的十项候选计算中成功完成了九项小规模基准原型。

    

    识别可能从量子算法中受益的科学计算，既需要领域知识，又需要量子算法设计方面的熟练能力，而这些专业知识很少同时存在于一个人身上。我们报告了QuASAR的结果，这是一个实验性的、由双智能体LLM驱动的流水线，旨在弥合这一差距。第一个智能体筛选科学文献，寻找具有合理量子算法对应物的计算内核，并提出候选方案，每个候选方案均有引文支持。第二个智能体独立验证这些引文。对于通过验证的候选方案，该智能体会重建源文献的计算结构，制定量子映射，构建并执行原型，并将其与经典基线进行基准测试。我们将该流水线应用于从天体物理学文献中提取的十项候选计算。九项候选计算被推进到小规模、经过基准测试的原型阶段。我们详细报告了其中三项，它们被选为最清晰的案例

    arXiv:2611.27778v1 Announce Type: new  Abstract: Identifying scientific computations that could benefit from quantum algorithms requires both domain knowledge and fluency in quantum algorithm design, expertise that rarely resides in one person. We report results from QuASAR, an experimental, two-agent, LLM-driven pipeline intended to bridge this gap. One agent screens scientific literature for computational kernels with a plausible quantum-algorithmic analogue and proposes candidates, each backed by citations. A second agent independently verifies those citations. For candidates that survive verification, it reconstructs the source's computational structure, formulates a quantum mapping, builds and executes a prototype, and benchmarks it against classical baselines. We applied this pipeline to ten candidate computations drawn from the astrophysics literature. Nine candidates were carried through to a small-scale, benchmarked prototype. We report three in detail, chosen as the clearest 
    
[^29]: RoSci：探索科学视觉语言模型的鲁棒性

    RoSci: Exploring Robustness in Science VLMs

    [https://arxiv.org/abs/2611.27750](https://arxiv.org/abs/2611.27750)

    该论文提出科学模型鲁棒性的概念，并构建了RoSci-Bench诊断基准和采用“解析-验证-决策”流程的RoSci-Agent框架，以增强视觉语言模型在科学任务中应对信息错误与信息缺失的能力。

    

    视觉语言模型（VLM）的最新进展显著推动了科学任务的发展。然而，这些模型通常被假定为接收完整且准确的信息，忽略了真实场景中信息错误/缺失所带来的挑战。为弥补这一空白，我们提出了“科学模型鲁棒性”的概念，该概念定义了模型在面对信息错误（不一致性）或信息缺失（不充分性）时检测异常、评估可解性以及进行自我校准的能力。具体而言，我们提出了RoSci-Bench，一个包含1,220个样本的诊断基准，采用种子-变体策略覆盖六种退化类型。为应对VLM的脆弱性，我们进一步提出了RoSci-Agent，一个采用“解析-验证-决策”（Parse-Verify-Decide）流程在推理前审查前提条件的框架。评估的VLM有时会回答前提无效或不完整的问题。RoSci-Agent取得了较高的……（原文摘要截断）

    arXiv:2611.27750v1 Announce Type: cross  Abstract: Recent advancements in Vision-Language Models (VLMs) enable significant progress in scientific tasks. However, these models are generally assumed to be presented with complete and accurate information, overlooking the challenges of erroneous/missing information in wild scenarios. To address this gap, we introduce the concept of Scientific Model Robustness, which defines a model's ability to detect anomalies, assess solvability, and perform self-calibration when faced with Information Error (inconsistencies) or Information Loss (insufficiency). Specifically, we present RoSci-Bench, a diagnostic benchmark of 1,220 samples using a seed-variant strategy across six degradation types. To address VLM fragility, we further propose RoSci-Agent, a framework that employs a Parse-Verify-Decide pipeline to vet premises before reasoning. The evaluated VLMs sometimes answer questions whose premises are invalid or incomplete. RoSci-Agent achieves a hi
    
[^30]: 人机团队中的共享理解：演进工作中的协调连续性

    Shared Understanding in Human-AI Teams: Coordination Continuity in Evolving Work

    [https://arxiv.org/abs/2611.27746](https://arxiv.org/abs/2611.27746)

    该论文提出“协调连续性”这一新概念，并构建了由持久表示、相对协调属性和交互机制三要素组成的框架，用以解释人机团队如何在演进的异步工作中持续维持相互依赖行动的共同基础。

    

    人们越来越多地将AI智能体作为扩展的异步工作流程的一部分来协同工作。在这些场景中，对信息的共享访问并不保证行动的共同基础：目标、计划和角色可能持续存在，但其当前的意义却在不断变化。我们提出了“协调连续性”这一概念，即协作者在工作推进过程中维持可运作的相互依赖行动基础的程度。基于协作工作领域的研究，以及两项定性团队研究和来自我们自身基于仓库的环境中的一个反思性案例，我们提出了一个包含三个要素的框架：协作工作的持久表示；支持对表示进行即时解释的相对协调属性；以及能够理解和改变表示状态的交互机制。我们运用“构建-行动-漂移-重建”的启发式方法以及“关键性时刻”的概念来刻画何时需要重建……

    arXiv:2611.27746v1 Announce Type: cross  Abstract: People are increasingly working with AI agents as part of extended, asynchronous workflows. In these settings, shared access to information does not guarantee a shared basis for action: goals, plans, and roles may persist while their current significance changes. We introduce coordination continuity, the extent to which collaborators sustain a workable basis for interdependent action as work progresses. Drawing on research on collaborative work, along with two qualitative team studies and a reflexive case from our own repository-based environment, we present a framework of three elements: durable representations of collaborative work; relative coordination properties that support interpretation of representations in the moment; and interaction mechanisms through which the status of representations can be understood and altered. We use a build-act-drift-rebuild heuristic and the notion of consequential moments to characterise when rebui
    
[^31]: 视觉语言模型能否合成可执行的游戏程序？

    Can Vision-Language Models Synthesize Executable Game-Playing Programs?

    [https://arxiv.org/abs/2611.27741](https://arxiv.org/abs/2611.27741)

    该论文提出了PlayScriptBench基准，评估视觉语言模型能否从游戏状态直接合成可执行的游戏程序，以克服逐步交互范式在实时性和调用成本方面的根本局限。

    

    视觉语言模型（VLM）的最新进展赋予多模态智能体日益强大的视觉感知和长程推理能力，使得基于VLM的自主游戏成为可能。现有的游戏基准主要评估智能体能否在不同场景和游戏中根据当前视觉状态选择合适的动作。然而，它们大多采用逐步交互范式，即在每个动作执行前将当前视觉观察发送给智能体。在实际部署中，这种范式引入了两个根本性限制：游戏循环必须等待模型推理，使得实时交互难以实现；同时重复调用VLM会带来高昂的成本。因此，我们提出了PlayScriptBench，一个基于脚本的游戏基准，用于评估VLM能否根据游戏状态合成可执行的游戏程序。这一设置同时评估了视觉定位、理解等多方面能力。

    arXiv:2611.27741v1 Announce Type: new  Abstract: Recent advances in vision-language models (VLMs) have endowed multimodal agents with increasingly strong visual perception and long-horizon reasoning capabilities, making VLM-based autonomous game playing feasible. Existing game benchmarks primarily evaluate whether an agent can select appropriate actions from the current visual state across different scenarios and games. However, most of them adopt a step-by-step interaction paradigm where the current visual observation is sent to the agent before each action. In practical deployment, this paradigm introduces two fundamental limitations: the game loop must wait for model inference, making real-time interaction impractical, and repeated VLM calls incur substantial cost. Therefore, we propose PlayScriptBench, a script-based game benchmark that evaluates whether VLMs can synthesize an executable game-playing program from game states. This setting jointly evaluates visual grounding, underst
    
[^32]: 通过请求局部Token路由实现平均深度可变执行

    Average-Depth Variable Execution via Request-Local Token Routing

    [https://arxiv.org/abs/2611.27738](https://arxiv.org/abs/2611.27738)

    提出AVERT方法，通过请求局部路由让每个Token动态决定执行的Transformer层数，使请求的平均执行深度精确匹配连续预算，在降低大语言模型推理成本的同时保持输出质量，且单一检查点可同时服务不同预算的请求。

    

    层跳过可以降低大语言模型的推理成本，但请求预算并未规定如何在生成的Token之间分配层的执行。我们提出了AVERT方法，它允许不同的Token执行不同数量的层，同时将每个请求的平均执行深度匹配到连续预算上。在每个路由层，路由器读取当前Token的状态及其所属请求的预算，并决定是否执行该Transformer块。跳过该块的Token会通过一个轻量级残差适配器，并且在该层不写入键/值（K/V）缓存条目。单一检查点即可在同一批次中服务具有不同预算的请求。在三个LLaMA骨干模型、六个质量指标和13种预算的实验中，AVERT在234次对比中有206次取得最佳或并列最佳结果。在单预算和混合预算批次中，每个请求的平均绝对预算误差最多为1.15个百分点。当按预算对请求分组导致批次无法填满时（摘要在此处截断）……

    arXiv:2611.27738v1 Announce Type: new  Abstract: Layer skipping can reduce large language model inference costs, but a request budget does not specify how to allocate layer execution across generated tokens. We introduce \method{}, which allows tokens to execute different numbers of layers while matching each request's mean executed depth to a continuous budget. At each routed layer, a router reads the current token state and its request's budget and decides whether to execute the Transformer block. Tokens that skip the block pass through a lightweight residual adapter and write no key/value (K/V) entry at that layer. A single checkpoint serves requests with different budgets in the same batch. Across three LLaMA backbones, six quality metrics, and 13 budgets, AVERT is best or tied for best in 206 of 234 comparisons. Across single- and mixed-budget batches, the mean absolute per-request budget error is at most 1.15 percentage points. When grouping requests by budget leaves batches unde
    
[^33]: 基于车载摄像头与透视校正的路面病害定量测量

    Quantitative Measurement of Pavement Distress based on Vehicle Camera and Perspective Correction

    [https://arxiv.org/abs/2611.27728](https://arxiv.org/abs/2611.27728)

    该研究提出了一种无需现场相机标定、基于车道线检测构建透视变换矩阵的路面病害定量测量方法，实现了从车载相机图像中直接获取病害的长度或面积等量化指标。

    

    利用图像数据正成为公路管理部门自动评估路面状况的重要方法。然而，大多数研究侧重于提高检测精度，而非获取路面病害的定量测量结果，例如病害的长度或面积。尽管现场相机标定可以提供此类测量，但对于从不同车辆采集的海量车载图像来说并不切实际。本研究开发了一种无需现场相机标定、可直接从车载相机图像中定量测量路面病害的方法。该方法首先基于车道线检测识别四边形感兴趣区域，并利用其构建透视变换矩阵以进行图像校正；随后利用图像中道路标线的比例和指定标尺进行尺寸校正，同时还能估算所选区域的实际宽度。校正后的图像随后被用于……

    arXiv:2611.27728v1 Announce Type: cross  Abstract: Using image data is becoming an important method for highway agencies to automatically evaluate pavement conditions. However, most studies focus on improving detection accuracy rather than obtaining quantitative measurements, such as the length or area of pavement distress. Although field camera calibration can provide such measurements, it is impractical for large volumes of onboard images collected from different vehicles. This study develops a method to quantitatively measure pavement distress directly from in-vehicle camera images without on-site camera calibration. A quadrilateral region of interest is first identified based on lane detection and used to construct a perspective transformation matrix for image correction. Size correction is then performed using the proportions of road markings in the image and a specified scale, which also enables estimation of the actual width of the selected region. The corrected images are subse
    
[^34]: 多语言情感分类的决策模型：与大语言模型和编码器的对比研究

    Decision Models for Multilingual Emotion Classification: A Comparative Study Against LLMs and Encoders

    [https://arxiv.org/abs/2611.27721](https://arxiv.org/abs/2611.27721)

    该研究首次在五种语言的多语言情感分类任务上系统比较了决策模型、大语言模型与微调编码器，发现最佳零样本系统的性能仅比微调编码器低约1个宏F1点。

    

    决策模型接收一个带有封闭答案列表的问题，并为每个答案返回一个概率，而非自由文本。它们具有低延迟和对提示措辞鲁棒性的优势，但尚无研究将多种决策模型与跨语言的微调编码器进行系统比较。我们探讨未经标注数据训练（零样本）使用的系统能否替代微调方法，以及决策格式相比通用大语言模型（LLM）能带来哪些额外优势。我们在五种语言的10个社交媒体数据集上，针对四种情绪（喜悦、悲伤、愤怒、恐惧），将三个决策模型（Jev、OpenAI Decisions、Strands Decider 2B）和两个采用意大利语指令的大语言模型（GPT-6 Luna、Qwen3.5-2B），与两个微调的多语言编码器（tabularisai、XLM-EMO）进行对比，并采用配对差异检验和等效性检验。结果表明，三个最佳的零样本系统（两个决策模型和一个大语言模型）的平均宏F1分数仅比在目标语言上微调的编码器低约1分。

    arXiv:2611.27721v1 Announce Type: cross  Abstract: Decision models receive a question with a closed list of answers and return a probability for each answer instead of free text. They promise low latency and robustness to prompt wording, but no study has compared several of them with fine-tuned encoders across languages. We ask whether systems used without training on labelled data (zero-shot) can replace fine-tuning, and what the decision format adds over a general-purpose large language model (LLM). We compare three decision models (Jev, OpenAI Decisions, Strands Decider 2B) and two LLMs (GPT-6 Luna, Qwen3.5-2B), given Italian instructions, with two fine-tuned multilingual encoders (tabularisai, XLM-EMO) on four emotions (joy, sadness, anger, fear) in 10 social-media datasets in five languages, with paired difference and equivalence tests. The three best zero-shot systems (two decision models and one LLM) score on average about one macro-F1 point below an encoder fine-tuned on the ta
    
[^35]: 可恢复符号，不稳定执行：视觉状态追踪中的内部路径与决策同一性审计

    Recoverable Symbols, Unstable Execution: Auditing Internal Pathways and Decision Identity in Visual State Tracking

    [https://arxiv.org/abs/2611.27715](https://arxiv.org/abs/2611.27715)

    该论文提出一种诊断框架，证明即使视觉特征中符号信息完全可恢复，循环执行器的决策仍会因图像无关上下文的变化而不稳定，且这种不稳定主要源于依赖输入的旋转相位——恢复原始相位可修复大部分新增错误。

    

    视觉特征可以完整保留任务所需的每一个符号，但建立在这些特征之上的模型在图像中无关部分发生变化时仍会改变其答案。我们在视觉状态追踪任务中研究这一差距：一个循环执行器通过冻结的ViT编码器读取符号图块序列，并判断所显示的候选状态是否等于最终状态。我们的诊断方法将每个输入与一个保留所有被查询图块的供体视图配对，利用符号监督解码器验证完整符号链可从两个特征序列中恢复，并在两个方向上交换同一已拟合执行器的单个内部通道。在两个视图中所有符号均可恢复的3,526对样本上，改变周围上下文使旋转循环头的平衡准确率下降17.13个百分点。这一损失主要经由依赖输入的旋转相位产生：恢复原始相位修复了391个新错误中的351个，并使准确率回升。

    arXiv:2611.27715v1 Announce Type: new  Abstract: Visual features can retain every symbol a task needs while a model built on them still changes its answer when irrelevant parts of the image change. We study this gap in visual state tracking, where a recurrent executor reads a sequence of symbol tiles through a frozen ViT encoder and decides whether a displayed candidate equals the final state. Our diagnostic pairs each input with a donor view that preserves every queried tile, checks with a symbol-supervised decoder that the full symbol chain is recoverable from both feature sequences, and swaps individual internal channels of the same fitted executor in both directions. On 3,526 pairs in which every symbol is recovered in both views, changing the surrounding context lowers the balanced accuracy of a rotary recurrent head by 17.13 points. The loss runs mainly through the input-dependent rotation phase: restoring the original phase repairs 351 of 391 new errors and brings accuracy back 
    
[^36]: MetaRigor：通过科学决策边界实现从智能体自动化到可验证的端到端荟萃分析

    MetaRigor: From Agentic Automation to Verifiable End-to-End Meta-Analysis Through Scientific Decision Boundaries

    [https://arxiv.org/abs/2611.27713](https://arxiv.org/abs/2611.27713)

    MetaRigor通过任务合同、程序化规则和来源绑定在任务层面实现科学决策边界，让大语言模型负责语义解释、程序负责形式化验证，从而构建了首个可验证的端到端自动化荟萃分析工作流框架，完整结果交付的F1分数达86.29%。

    

    自动化荟萃分析不仅需要准确的数值提取，还需要在各阶段一致地保持结果身份的完整性，包括相关的研究、比较、人群、结局指标、时间点和分析条件。我们提出了MetaRigor，一个用于可验证端到端证据综合的大语言模型工作流框架。MetaRigor将研究结果组织为“研究×目标”单元，并通过任务合同、程序化规则和来源绑定，在任务层面将科学决策边界操作化。模型执行依赖于研究上下文的语义解释，而程序则执行可形式化的规则、计算、验证和来源核验。我们在五个阶段开展了受控评估：证据获取、数据提取、偏倚风险评估、证据质量评估和论文生成。MetaRigor在完整结果交付方面达到了86.29%的F1分数。

    arXiv:2611.27713v1 Announce Type: new  Abstract: Automated meta-analysis requires not only accurate numerical extraction but also the consistent preservation of result identity across stages, including the associated study, comparison, population, outcome, time point, and analysis conditions. We present MetaRigor, an LLM workflow framework for verifiable end-to-end evidence synthesis. MetaRigor organizes study results as Study x Target units and operationalizes scientific decision boundaries at the task level through Task Contracts, programmatic rules, and source binding. The model performs semantic interpretation that depends on study context, whereas programs execute formalizable rules, computations, validation, and source verification. We conducted controlled evaluations across five stages: evidence acquisition, data extraction, risk of bias assessment, certainty of evidence assessment, and manuscript generation. MetaRigor achieved an F1 score of 86.29% for complete result delivery 
    
[^37]: 多模态机器翻译何时需要图像？基于纯文本路由的按需视觉上下文

    When Are Images Needed in Multimodal Machine Translation? Text-Only Routing for On-Demand Visual Context

    [https://arxiv.org/abs/2611.27712](https://arxiv.org/abs/2611.27712)

    该论文将多模态机器翻译中的图像调用形式化为“源文本充分性预测”任务，构建了含三个细粒度标签的视觉依赖数据集，实现仅凭源文本即可按需判断是否需要视觉上下文的路由方法，从而避免冗余的视觉计算开销。

    

    图像引导的多模态机器翻译（MMT）利用视觉上下文来消解与翻译相关的不确定性。然而，现有主流系统要么无条件地处理图像，要么仅在视觉编码之后才应用门控机制，使得冗余的视觉计算不可避免。为了缓解这一计算开销，路由决策必须依赖于视觉编码之前的源端信号。至关重要的是，表层的歧义并不等同于视觉依赖性，因为句内文本上下文通常足以完成消歧。因此，我们将视觉调用形式化为源文本充分性预测：仅根据源文本来判断与翻译相关的不确定性是否仍未得到解决。为支持这一形式化，我们构建了一个包含三个细粒度标签（no_ambiguity、text_resolved 和 vision_needed）的视觉依赖数据集，并为 vision_needed 实例验证了具有区分性的视觉证据。我们进一步提出了 Pre……（摘要在此处截断）

    arXiv:2611.27712v1 Announce Type: new  Abstract: Image-guided multimodal machine translation (MMT) leverages visual context to resolve translation-relevant uncertainty. However, prevailing systems either process images unconditionally or apply gating only after visual encoding, leaving redundant visual computation unavoidable. To mitigate this computational overhead, routing decisions must rely on a pre-visual source signal. Crucially, surface-level ambiguity does not equate to visual dependence, as intra-sentence textual context often suffices for disambiguation. We therefore formulate visual invocation as source-text sufficiency prediction: determining exclusively from the source text whether translation-relevant uncertainty remains unresolved. To support this formulation, we construct a visual-dependency dataset with three fine-grained labels (no_ambiguity, text_resolved, and vision_needed) and verify discriminative visual evidence for vision_needed instances. We further propose Pre
    
[^38]: 《识别变化之处，复用更长模式：高效的离线思维链压缩》

    Abstract What Varies, Reuse Longer Patterns: Efficient Offline Chain-of-Thought Compression

    [https://arxiv.org/abs/2611.27710](https://arxiv.org/abs/2611.27710)

    CRAFT通过将问题特定细节抽象为槽位来发现跨样本的部分词汇化表达式模板并聚类为族，只需对每个族的代表进行一次压缩即可复用共享编辑操作，从而将压缩器预算降低中位数90.7%且准确率损失仅1.21个百分点。

    

    离线思维链压缩通过创建更短的监督信号来实现高效推理，但逐样本的压缩器会在不同推理轨迹中反复处理相似的表达式，浪费了压缩器预算。而跨样本复用完全相同文本的方法，在问题的名称、数字或公式发生变化时，会遗漏更长的共享表达式。我们提出CRAFT，它将问题特定的细节抽象为槽位，从而发现部分词汇化的表达式模板，并将措辞变体聚类为族。对每个族的代表性样本只需压缩一次，即可得到共享的删除或改写编辑操作，这些操作结合每次出现的具体细节进行应用，从而构建出紧凑的、普通文本形式的训练目标。与逐样本改写相比，CRAFT Rewrite将加权压缩器预算降低了中位数90.7%，准确率损失中位数仅为1.21个百分点，同时生成更短的推理过程。与跨样本编码方法相比……（摘要原文在此处截断）

    arXiv:2611.27710v1 Announce Type: new  Abstract: Offline chain-of-thought (CoT) compression creates shorter supervision for efficient reasoning, but per-example compressors repeatedly process similar expressions across traces, wasting compressor budget. Methods that reuse identical text across examples miss longer shared expressions when names, numbers, or formulas change from one question to another. We introduce CRAFT, which abstracts problem-specific details into slots to uncover partially lexicalized expression templates (PLETs) and clusters wording variants into families. Compressing each family representative once yields shared deletion or rewriting edits, which are applied with each occurrence's own details to construct compact, ordinary-text training targets. Compared with per-example rewriting, CRAFT Rewrite reduces weighted compressor budgets by a median 90.7% with a median accuracy loss of 1.21 percentage points, while generating shorter reasoning. Against cross-sample encod
    
[^39]: 面向化工过程故障检测与结构学习的时空图神经网络：以田纳西-伊士曼过程为应用案例

    Spatiotemporal Graph Neural Networks for Fault Detection and Structural Learning in Chemical Processes: Use Case on the Tennessee Eastman Process

    [https://arxiv.org/abs/2611.27698](https://arxiv.org/abs/2611.27698)

    该论文提出一种时空图神经网络框架，通过将正常运行空间分解为特定于运行条件的子空间，并结合GATv2图结构学习与时间编码，解决了化工过程中因运行条件变化导致的误报与漏检问题，并在田纳西-伊士曼过程上验证了其故障检测能力。

    

    经典的故障检测与诊断（FDD）方法，包括许多数据驱动方法，都假设一个静态的正常运行空间，并将对固定参考的偏离视为故障指标。当运行条件（OCs）随时间变化时，这一假设便不再成立：合法的工况转换会触发误报，而中等程度的故障却无法被检测到。我们提出了一种时空图神经网络（GNN）框架，将正常运行空间分解为由转换函数连接的、特定于运行条件的子空间，并采用结合重构损失与Deep SVDD单分类项的双重学习目标。该框架通过图注意力网络（GAT）学习邻接矩阵，并将空间建模与时间编码相结合，以表征运行条件不断演变下的过程动态。本文在田纳西-伊士曼（Tennessee Eastman）过程上对该框架的基础组件（故障检测、GATv2空间图学习和时间编码）进行了评估

    arXiv:2611.27698v1 Announce Type: new  Abstract: Classical Fault Detection and Diagnosis (FDD) methods, including many data-driven approaches, assume a static normal operating space and interpret deviations from a fixed reference as fault indicators. When Operating Conditions (OCs) vary over time, this assumption breaks down: legitimate transitions trigger false alarms while moderate faults go undetected. We propose a spatiotemporal Graph Neural Network (GNN) framework that decomposes the normal operating space into OC-specific subspaces linked by transition functions, with a dual learning objective combining reconstruction loss and a Deep SVDD one-class term. It learns adjacency matrices through Graph Attention Networks (GATs) and integrates spatial modelling with temporal encoding to represent process dynamics under evolving OCs. This paper evaluates the framework's foundational components (fault detection, GATv2 spatial graph learning and temporal encoding) on the Tennessee Eastman 
    
[^40]: 化工过程中用于故障检测与诊断的机器学习架构性能比较

    Comparative Performance of Machine Learning Architectures for Fault Detection and Diagnosis in Chemical Processes

    [https://arxiv.org/abs/2611.27696](https://arxiv.org/abs/2611.27696)

    本研究以田纳西-伊士曼过程（TEP）为基准，系统比较了多种机器学习架构（包括集成学习、LSTM、XGBoost等）在化工过程故障检测与诊断任务中的性能，为实际工业场景下的方法选择提供了依据。

    

    故障检测与诊断（FDD）对化工行业的维护与控制至关重要，早期故障检测能够避免代价高昂的设备失效并优化运行。这在IFP Energies Nouvelles（IFPEN）等中试装置中尤为关键，因为这类装置在短期实验条件下运行，操作参数经常变化。本研究基于田纳西-伊士曼过程（TEP）——一个被广泛使用的模拟化工过程数据集——开展了全面的基准测试分析，以评估多种故障检测与诊断方法，其中包含大量具有不同传感器和故障类型的连续运行场景。研究实现并比较了多种方法，包括用于故障检测的多尺度PCA（MS-PCA）、自编码器、集成学习（Ensemble Learning）和LSTM模型，以及用于故障诊断的随机森林、XGBoost和BLSTM（双向LSTM）。我们的结果表明，集成学习……

    arXiv:2611.27696v1 Announce Type: new  Abstract: Fault Detection and Diagnosis (FDD) is crucial for both maintenance and control in chemical industries, where early fault detection can prevent costly failures and optimize operations. This is particularly critical in pilot units like those at IFP Energies Nouvelles (IFPEN), which operate under short-term experimental conditions with frequently varying operational parameters. This study conducts an extensive benchmarking analysis using the Tennessee Eastman Process (TEP), a widely used simulated chemical process dataset, to evaluate multiple approaches for fault detection and diagnosis, featuring numerous continuous operations with different sensors and faults. Several methods were implemented and compared including Multi Scale PCA (MS-PCA), AutoEncoder, Ensemble Learning, and LSTM models for fault detection, alongside Random Forest, XGBoost, and BLSTM (Bidirectional LSTM) for fault diagnosis. Our results demonstrate that Ensemble Learni
    
[^41]: Embodied-Harness：迈向机器人操作的递归自我改进

    Embodied-Harness: Towards Recursive Self-Improvement for Robotic Manipulation

    [https://arxiv.org/abs/2611.27689](https://arxiv.org/abs/2611.27689)

    提出双智能体框架 Embodied-Harness，通过分层记忆、基于证据的经验反思与递归自我改进，在不更新模型参数的情况下提升机器人操作能力。

    

    基于视觉语言模型（VLM）的智能体为机器人操作提供了一种有前景的方法，但仍然受限于记忆组织不足、失败归因不确定以及执行经验复用有限等问题。我们提出 Embodied-Harness，这是一个双智能体框架，通过改进执行框架本身而不更新模型参数来提升操作能力。Actor（执行者）通过机器人控制接口与环境交互，而 Reflector（反思者）将执行证据转化为可复用的操作指导。分层记忆架构维护短期执行上下文与长期经验知识，以同时支持执行和反思。基于证据的经验反思将失败诊断建立在观察到的交互之上，而递归框架自我改进则在连续多次尝试中不断完善操作指导。在 RoboTwin 2.0、RoboSPA 和 RMBench 上的实验证明了 Embodied-Harness 的有效性。

    arXiv:2611.27689v1 Announce Type: cross  Abstract: Vision-language model (VLM)-based agents offer a promising approach to robotic manipulation, yet remain limited by inadequate memory organization, uncertain failure attribution, and limited reuse of execution experience. We propose Embodied-Harness, a dual-agent framework that improves manipulation by refining the execution harness without updating model parameters. The Actor interacts with the environment through a robot-control interface, while the Reflector converts execution evidence into reusable operational guidance. Hierarchical Memory Architecture maintains short-term execution context and long-term experiential knowledge to support execution and reflection. Evidence-based Experience Reflection grounds failure diagnosis in observed interactions, while Recursive Harness Self-Improvement refines operational guidance across successive attempts. Experiments on RoboTwin 2.0, RoboSPA, and RMBench demonstrate the effectiveness of Embo
    
[^42]: 共生学习：基于人工神经网络与脉冲神经网络的异构联邦学习

    Learning in Symbiosis: Heterogeneous Federated Learning with ANNs and SNNs

    [https://arxiv.org/abs/2611.27686](https://arxiv.org/abs/2611.27686)

    本文提出混合神经形态联邦学习框架，使异构的人工神经网络与脉冲神经网络设备能够在联邦学习环境中协同联合训练，从而推动向节能神经形态边缘AI的渐进式过渡。

    

    脉冲神经网络（SNNs）被广泛认为是第三代神经网络，代表了一种有前景的神经形态计算范式，有潜力降低人工智能（AI）的能源足迹。其能效优势使SNN对资源受限的边缘AI设备特别具有吸引力，这也推动了对SNN与联邦学习（FL）相结合的研究兴趣日益增长。尽管具有这些优势，将SNN集成到现有联邦学习系统中仍然具有挑战性，因为用基于SNN的替代设备替换所有传统人工神经网络（ANN）设备是不切实际的。相反，我们设想了一种向神经形态边缘AI的渐进式过渡，其中ANN和SNN设备共存并协作参与联邦学习。为促进这一过渡，我们提出了混合神经形态联邦学习（HNFL），这是一个能够在联邦学习环境中对异构ANN和SNN设备进行联合训练的框架。

    arXiv:2611.27686v1 Announce Type: new  Abstract: Spiking Neural Networks (SNNs), widely regarded as the third generation of neural networks, represent a promising neuromorphic computing paradigm with the potential to reduce the energy footprint of Artificial Intelligence (AI). Their en-ergy efficiency makes SNNs particularly attractive for resource-constrained Edge AI devices, motivating growing research interest in Federated Learning (FL) with SNNs. Despite these advantages, integrating SNNs into existing FL systems remains challenging, as replacing all conventional Artificial Neural Network (ANN) devices with SNN-based alternatives is impractical. Instead, we envision a gradual transition toward neuromorphic Edge AI, where ANN and SNN devices coexist and collaboratively participate in FL. To facilitate this transition, we propose Hybrid Neuromorphic Federated Learning (HNFL), a framework that enables joint training of heterogeneous ANN and SNN devices within an FL environment. Throug
    
[^43]: Hy-MT3：更准确、更鲁棒且更实用的真实场景多语言翻译模型

    Hy-MT3: More Accurate, Robust and Practical Multilingual Translation Models in the Wild

    [https://arxiv.org/abs/2611.27673](https://arxiv.org/abs/2611.27673)

    Hy-MT3 系列通过将误译率降低 43%、漏译增译降低 74%、指令遵循错误降低 40%，并把语言覆盖扩展至 102 种，其仅 3B 激活参数的 MoE 模型在非思考模式下超越了众多旗舰大模型并逼近最强闭源思考模型。

    

    Hy-MT3 是混元多语言翻译模型家族的第三代，包括 Hy-MT3-1.8B、Hy-MT3-7B 和 Hy-MT3-30B-A3B（MoE 架构）。与 Hy-MT2 相比，Hy-MT3 的翻译更准确、更完整、更忠实于用户约束：在真实场景数据上，误译样本比例降低了 43%，漏译或增译问题降低了 74%，指令遵循错误降低了 40%，同时语言覆盖范围从 38 种语言扩展到 102 种。仅需 3B 激活参数，Hy-MT3-30B-A3B 便在非思考模式下取得了所有评测模型中的最佳整体性能，超越了 Qwen3.8-Max、Kimi-K3.1-Pro 和 DeepSeek-V4.1-Flash 等旗舰模型，并接近 GPT-6.1-Sol 和 Gemini-4-Argon 等最强的闭源思考模型，同时超越了消耗超过 30 倍资源的模型。

    arXiv:2611.27673v1 Announce Type: cross  Abstract: Hy-MT3 is the third generation of the Hunyuan multilingual translation model family, including \textbf{Hy-MT3-1.8B}, \textbf{Hy-MT3-7B} and \textbf{Hy-MT3-30B-A3B (MoE)}. Compared with Hy-MT2, Hy-MT3 translates \textbf{more accurately, more completely and more faithfully to user constraints}: on real-world scenario data, the share of samples with mistranslations is reduced by \textbf{43\%}, omissions or additions by \textbf{74\%}, and instruction-following errors by \textbf{40\%}, while language coverage is expanded from \textbf{38 to 102} languages. With only 3B activated parameters, \textbf{Hy-MT3-30B-A3B achieves the best overall performance among all evaluated models in non-thinking mode}, including flagship models such as Qwen3.8-Max, Kimi-K3.1-Pro and DeepSeek-V4.1-Flash, and approaches the strongest closed-source thinking models such as GPT-6.1-Sol and Gemini-4-Argon, while surpassing models that consume \textbf{more than 30$\ti
    
[^44]: 在类型中学习：面向安全多智能体强化学习的认知类型状态

    Learning Inside Types: Epistemic Typestates for Safe Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.27631](https://arxiv.org/abs/2611.27631)

    本文提出“认知类型状态”这一类型规则，让多智能体强化学习智能体在记录自身信念的类型协议内学习，仅执行在所有可能世界中都安全的动作，从而无需通信即可实现类型可靠性与团队安全性。

    

    摘要：类型规则使程序在构造上即安全：组件通过彼此类型的检查后组合而成的程序，不会出现类型所排除的错误。类型非常适合强化学习：类型约束的是智能体“可以选择什么”，而非“如何选择”，因此该保证对学习者产生的每一个策略都成立。我们提出了认知类型状态，这是一种面向安全多智能体强化学习的类型规则。智能体的类型是一个协议，其状态记录该智能体的信念——即依据自身观察与行动无法排除的那些可能世界——并且只允许执行在信念中每个世界里都安全（无论其他智能体在其公共类型内如何行动）的动作。智能体在类型之内学习；类型逐个智能体地构建与检查，并在运行时无需通信。我们证明了类型可靠性与团队安全性，给出了一个可靠且完备的类型综合方法，在可能世界为无穷时从守卫构建类型，并表明屏蔽器……（摘要在此处截断）

    arXiv:2611.27631v1 Announce Type: new  Abstract: Type disciplines make programs safe by construction: components checked against each other's types compose into programs that cannot go wrong in the ways the types exclude. Types suit reinforcement learning: a type constrains what an agent may choose, not how it chooses, so the guarantee holds for every policy a learner produces. We introduce epistemic typestates, a type discipline for safe multi-agent reinforcement learning. An agent's type is a protocol whose states record its beliefs, the worlds it cannot rule out from its own observations and actions, and permits only actions that are safe in every world of the belief, whatever the others do within their public types. Agents learn inside their types, which are built and checked one agent at a time and run without communication. We prove type soundness and team safety, give a synthesis that is sound and complete, build types from guards when worlds are infinite, and show that shields 
    
[^45]: 基准测试的期末考试

    Benchmarks' Last Exam

    [https://arxiv.org/abs/2611.27622](https://arxiv.org/abs/2611.27622)

    该论文首次系统量化了主流AI基准测试的任务有效性，发现经过公开纠错修订的封闭式基准达到92.5%的验证有效性，而多步骤套件仅有29.3%，揭示了当前评估体系在提示、环境与评分器一致性上的显著缺陷。

    

    公共基准测试指导着模型选择、训练后奖励以及人工智能领域的进展声明，然而排行榜报告的是总分，并未验证每个任务的提示、环境和评分器是否就“什么算作成功”达成一致。在当代评估中，这种一致性失效的频率有多高，以及已确认的缺陷朝哪个方向影响分数，在各种评分设计下一直未被测量。本文综合了对8个前沿基准测试中749个任务的任务级质量审查，以及767次公开模型评估，涵盖封闭式问答、文档评分规则和智能体执行，以评估提示、环境和评分器是否相互一致。任务有效性按评分架构呈现显著分层：经过公开错误报告后修订的封闭式基准达到92.5%的验证有效性，而六个多步骤套件的验证有效性仅为29.3%，尽管这些套件已提供161个经过验证的复杂任务。评分规则…

    arXiv:2611.27622v1 Announce Type: new  Abstract: Public benchmarks guide model selection, post-training rewards and claims of progress in artificial intelligence, yet leaderboards report aggregate scores without verifying whether each task's prompt, environment and grader agree on what counts as success. How often that agreement fails in contemporary evaluations, and in which direction confirmed defects move scores, has remained unmeasured across grading designs. Here we synthesize task-level quality reviews of 749 tasks across eight frontier benchmarks and 767 public model evaluations spanning closed-form question answering, document rubrics and agentic execution to assess whether prompts, environments and graders align. Task validity stratifies sharply by grading architecture. Closed-form benchmarks revised after public error reporting reach 92.5% verified validity, compared with 29.3% across six multi-step suites that nevertheless already supply 161 verified complex tasks. Rubric su
    
[^46]: 大语言模型智能体中的势能涌现：统计物理学视角

    Potential emergence in LLM agents: a statistical physics perspective

    [https://arxiv.org/abs/2611.27618](https://arxiv.org/abs/2611.27618)

    该论文从统计物理学视角提出将智能体状态作为中间变量，通过配分函数与全局势能来定量测量并预测大语言模型智能体在后训练过程中行为的组织与涌现。

    

    近年来，从人类偏好或语料库统计的角度理解大语言模型智能体的行为引起了极大关注。本文提出将智能体状态作为中间变量，通过这些变量可以对后训练过程进行测量和预测。从统计物理学的视角出发，我们研究了一个支配大语言模型智能体推理中状态转移的全局势能的涌现。我们引入配分函数来建模理想的KL正则化转移分布，并追踪势能结构、平均奖励和状态熵。我们在两个Qwen3.5-9B任务上展示了我们的方法：安全测试选择和物理推理审计。我们观察到，正如我们的框架所预测的，势能结构在后训练过程中不断增强，同时奖励与状态熵的联合变化提供了一致性证据。因此，状态为后训练如何组织智能体行为提供了定量描述。

    arXiv:2611.27618v1 Announce Type: cross  Abstract: There has been significant recent interest in understanding the behaviour of LLM agents from the perspective of human preferences or corpus statistics. In this paper we propose agent states as intermediate variables through which post-training can be measured and predicted. From a statistical physics perspective, we study the emergence of a global potential which governs the transition of state in LLM agent reasoning. We introduce a partition function to model the ideal KL-regularised transition distribution and track potential structure, mean reward and state entropy. We demonstrate our approach in two Qwen3.5-9B tasks: safety-test selection and physics-reasoning auditing. We observe that the potential structure strengthens during post-training as predicted from our framework, while joint changes in reward and state entropy provide consistency evidence. States thus provide a quantitative description of how post-training organises agen
    
[^47]: PromptTTSD：基于开放式文本描述的可控文本到口语对话合成

    PromptTTSD: Controllable Text to Spoken Dialogue Synthesis with Open-Ended Text Descriptions

    [https://arxiv.org/abs/2611.27601](https://arxiv.org/abs/2611.27601)

    该论文提出PromptTTSD，通过从多叙述者有声书自动挖掘对话语音、改进两阶段对齐算法保持对话连续性，并利用LLM从叙事文本生成开放式风格描述，实现了以自然语言细粒度控制多说话人口语对话合成的能力，同时发布了8383小时中文的PromptDialog大规模对话数据集。

    

    口语对话合成旨在从对话脚本生成多轮对话语音，同时保持说话人一致性和语境的适当性。然而，现有模型在很大程度上缺乏细粒度的可控性，而这对捕捉自然对话中动态的、依赖上下文的风格变化至关重要。虽然基于提示词的TTS已为单说话人语音实现了这种控制，但现有数据集严重依赖预定义标签，这从根本上限制了其捕捉开放式风格的能力。为解决这一缺口，我们提出了一个从多叙述者有声书中提取对话语音的可扩展流程，采用改进的两阶段对齐算法以保持对话连续性，并结合基于大语言模型（LLM）的方法直接从叙事文本中提取风格描述。我们发布了PromptDialog——一个大规模多说话人对话语音数据集，包含8,383小时的中文语音数据。

    arXiv:2611.27601v1 Announce Type: cross  Abstract: Spoken dialogue synthesis aims to generate multi-turn conversational speech from dialogue scripts while maintaining speaker consistency and contextual appropriateness. However, existing models largely lack fine-grained controllability, which is critical for capturing the dynamic, context-dependent style shifts in natural conversations. While prompt-based TTS has enabled such control for single-speaker speech, existing datasets heavily rely on predefined tags, which fundamentally limits their capacity to capture open-ended styles. To address this gap, we propose a scalable pipeline for extracting conversational speech from multi-narrator audiobooks, featuring an improved two-stage alignment algorithm to preserve dialogue continuity, paired with LLM-based extraction of style descriptions directly from narrative text. We release PromptDialog, a large-scale multi-speaker conversational speech dataset comprising 8,383 hours of Chinese speec
    
[^48]: 释放AI数据中心中异构加速器的灵活性：容量规划与工作负载分配

    Unlocking Heterogeneous Accelerator Flexibility in AI Data Centers: Sizing and Workload Allocation

    [https://arxiv.org/abs/2611.27576](https://arxiv.org/abs/2611.27576)

    本文提出联合优化异构加速器的容量规划与AI工作负载（训练、批处理、在线推理）的跨站点跨时段分配，以最低成本满足时变电网约束，在双站点案例中相比固定容量的时空调度使日成本降低6.5%。

    

    AI数据中心是一种新兴的电力需求，但有限的本地电网容量可能会延迟其接入与扩建。尽管现有的运营策略已考虑工作负载的时空灵活性，但很少有研究针对具有不同资本成本与能效权衡的多样化加速器类型进行战略性的容量规划与分配。本文通过联合规划异构加速器的容量，并将不同类型的AI工作负载（训练、可容忍延迟的批处理任务以及在线推理请求）跨站点、跨时段进行分配，以最低成本满足随时间变化的电网约束，从而释放了异构加速器的灵活性潜力。工作负载特定的吞吐量与能耗系数将这些决策与设施电力需求联系起来。在一个代表性的双站点案例中，相对于固定容量下的时空调度，异构容量规划使每日成本降低了6.5%。当采用工作负载特定性能进行评估时，某种容量设计方案……（原文在此截断）

    arXiv:2611.27576v1 Announce Type: cross  Abstract: AI data centers are an emerging electricity demand, but limited local grid capacity could delay their connection and expansion. While existing operational strategies consider spatio-temporal workload flexibility, few studies address the strategic sizing and allocation of diverse accelerator archetypes with different capital and energy-efficiency trade-offs. This paper unlocks the flexibility potential of heterogeneous accelerators by jointly sizing and allocating different AI workloads (training, delay-tolerant batch work, and online inference requests) across sites and hours to meet the time-varying grid constraint at minimum cost. Workload-specific throughput and energy coefficients link these decisions to facility power. In a representative two-site case, heterogeneous sizing reduces daily cost by 6.5% relative to spatio-temporal scheduling with fixed capacity. When evaluated using workload-specific performance, a capacity design ba
    
[^49]: 语言模型能“看懂”像素化射频集成电路无源器件中的电路吗？

    Can Language Models See the Circuit in Pixelated RFIC Passives?

    [https://arxiv.org/abs/2611.27559](https://arxiv.org/abs/2611.27559)

    本研究首次验证了前沿大语言模型在完全禁用工具的情况下，能够仅凭读取18×18和27×27网格版图，就准确预测像素化射频无源器件在20–100 GHz的输入阻抗与损耗，表明语言模型可以“看懂”这类反直觉的逆设计电路结构。

    

    面向微波和毫米波应用的逆设计像素化无源器件的出现，是射频集成电路设计领域的一项重大进展。这类器件在结构上形似二维码，存在于一个可泛化的高维设计空间中，该空间也涵盖了传统无源拓扑结构。然而，这些结构对设计者而言天然缺乏直观性，其性能失效或偏差也难以调试。在这项工作中，我们探究前沿语言模型能否理解、预测并编辑像素化无源器件的响应与版图结构。该智能体在禁用所有工具的条件下运行，首先被要求读取18×18和27×27瓦片网格上的版图，并在单次回复中预测该结构在20–100 GHz范围内的输入阻抗与损耗。在八个遗传算法合成版图和五个随机填充版图上，预测的史密斯圆图轨迹与全波仿真结果高度吻合，并呈现出系统性的……

    arXiv:2611.27559v1 Announce Type: new  Abstract: The emergence of inverse-designed, pixelated passives for microwave and millimeter-wave applications has been a major development in the field of RFIC design. These devices, which appear to be QR-code-like in structure, exist in a generalizable, high-dimensional design space that includes conventional passive topologies. However, these structures are naturally unintuitive for designers, and failures or deviations in performance are difficult to debug. In this work, we ask whether a frontier language model can understand, predict, and edit the responses and layout structure of pixelated passives. The agent, run with all tools disabled, is first asked to read layouts on 18x18 and 27x27 tile grids and, in a single reply, predict the input impedance and loss of the structure from 20-100 GHz. On eight GA-synthesized layouts and five random fills, the predicted Smith-chart locus follows full-wave simulation results closely, with a systematic s
    
[^50]: 将AI辅助引入贝兹瑙核电站：用于反应堆监控的分层LLM智能体系统

    Bringing AI Assistance to the Beznau Nuclear Power Plant: A Hierarchical System of LLM Agents for Reactor Monitoring

    [https://arxiv.org/abs/2611.27556](https://arxiv.org/abs/2611.27556)

    该论文为运行超过五十年的贝兹瑙核电站构建了基于大语言模型的分层多智能体监控系统，由协调智能体分解工程师的问题并分派给仅具只读权限的专用子智能体，以自然语言编写领域流程供工程师审核修改，从而加速堆芯仪表异常调查并减少跨孤岛的人工检索负担。

    

    瑞士贝兹瑙1号机组自1969年投入运行以来，是全球运行时间最长的在役核反应堆。五十年的渐进式现代化改造使其监控系统、报警日志、变量目录、技术规格书和调查报告分散在彼此孤立的信息孤岛中，而这些系统从未被设计为可以联合查询。因此，调查一次堆芯仪表异常意味着需要缓慢地、人工地收集分散在数据库、文档和资深同事经验中的知识。我们提出了一种基于大语言模型（LLM）构建的分层多智能体助手，以支持这类调查工作，并减少在SAP系统中进行繁琐人工检索的需求。一个协调智能体将工程师的问题进行分解，并将各部分任务分配给专门的子智能体，每个子智能体仅对单一数据源拥有只读访问权限。领域专业知识以自然语言流程的形式编写，电站工程师可以阅读、审核和修改这些流程。

    arXiv:2611.27556v1 Announce Type: cross  Abstract: Beznau Unit 1 in Switzerland, in operation since 1969, is the most experienced nuclear reactor still running worldwide. Five decades of incremental modernization have left its monitoring systems, alarm logs, variable catalogs, technical specifications and investigation reports in separate silos that were never designed to be queried together. As a result, investigating a core instrumentation anomaly means slow, manual gathering of knowledge scattered across databases, documents, and experienced colleagues. We present a hierarchical multi-agent assistant built on large language models (LLMs) to support these investigations and reduce the need for tedious manual searches in SAP systems. An orchestrating agent breaks down an engineer's question and hands the parts to specialized sub-agents, each with read-only access to one source. Domain expertise is written as natural-language procedures that plant engineers can read, review and revise.
    
[^51]: Vega-Fafnir：贪心解码还是同一种解码吗？投机验证中的舍入噪声从何而来，消除它需要付出什么代价

    Vega-Fafnir: Is Greedy Decoding the Same Decoding? Where Rounding Noise in Speculative Verification Comes From and What Removing It Costs

    [https://arxiv.org/abs/2611.27554](https://arxiv.org/abs/2611.27554)

    该论文量化了贪心解码在不同执行方式下的舍入噪声：投机验证改变 60%、填充改变 68% 的提示输出，且首分歧处 logit 差距均不超过 0.375，噪声主要来自多词元前向传播写入的 KV 缓存条目，而基于裕量门控的回退策略无法消除噪声，重建缓存则需付出额外代价。

    

    服务引擎将运行贪心语言模型的各种方式视为可互换的：有无缓存、有无填充、单独运行还是批量运行、一次验证一个词元还是同时验证多个词元（投机解码）。在有限精度下，它们的舍入方式各不相同。我们测量了 Qwen3-0.6B（bfloat16，生成 128 个新词元，使用 GSM8K 和 ARC 提示）的贪心文本与每次只解码一个词元的参考实现之间出现差异的频率。每次验证 4 个猜测词元会改变 150 个提示中的 60%，填充会改变 68%；每一处首次分歧的 top-1 与 top-2 logit 差距都不超过 0.375。对噪声的分解表明，由多词元前向传播写入的缓存条目与多词元前向传播本身具有几乎相同的噪声水平。一种裕量门控方法（当差距低于在独立 GSM8K 提示上校准的界限时回退到单词元解码），即使其回退路径读取该缓存也无济于事：100 个提示中有 58 个出现差异，而普通验证为 100 个中的 61 个。用单词元前向传播重建缓存……

    arXiv:2611.27554v1 Announce Type: new  Abstract: Serving engines treat the ways of running a greedy language model as interchangeable: cached or not, padded or not, alone or batched, one token or several verified at once (speculative decoding). In finite precision they round differently. We measure how often the greedy text of Qwen3-0.6B (bfloat16, 128 new tokens, GSM8K and ARC prompts) differs from a reference that decodes one token per pass. Verifying 4 guessed tokens per pass changes 60% of 150 prompts and padding changes 68%; every first divergence has a top-1 minus top-2 logit gap of 0.375 or less. A split of the noise shows that cache entries written by multi-token passes are about as noisy as the multi-token pass itself. A margin gate (fall back to one token when the gap is below a bound calibrated on separate GSM8K prompts) whose fallback reads that cache does not help: 58 of 100 prompts differ, against 61 of 100 for plain verification. Rebuilding the cache with one-token passe
    
[^52]: Altair-Nidhogg：哪些模型合并理论适用于单次前向决策编码器？模型汤、任务算术、TIES 与 DARE 对比联合训练与预测集成

    Altair-Nidhogg: Which Model-Merging Theories Hold for a One-Pass Decision Encoder? Soups, Task Arithmetic, TIES and DARE Against Joint Training and Prediction Ensembling

    [https://arxiv.org/abs/2611.27550](https://arxiv.org/abs/2611.27550)

    该论文在一个单次前向决策编码器上系统检验了模型汤、任务算术、TIES 和 DARE 四种模型合并配方是否有效及其理论是否成立，发现 TIES 对专家增益的保留率最高（64%），并将其与联合训练和预测集成进行了全面对比。

    

    arXiv:2611.27550v1 公告类型：新 摘要：Laya 是一个小型模型，可以在一次前向传播中回答多项选择题：每个选项都有一个名称和一段文字描述。学习一项新技能需要一次额外的训练运行，因此我们测试已发表的微调模型合并配方（权重平均、任务算术、TIES、DARE）是否对它有效，以及这些配方背后的理论是否成立。全部 156 个模型都从同一个已发布的检查点出发：四个普通微调模型、五个专家模型（每个增加一种训练数据行）、在所有数据行上的联合微调模型，以及预测集成模型。测试项目均为保留项目、合成项目并由大语言模型标注。在同一家族内，每项技能都在其自身类型的新项目上评分，“保留率”指专家模型相对于普通微调的增益中被保留的比例。对四个专家模型进行平均可保留 52% [38, 64] 的增益，同时通用准确率提升 1.7 分；添加完整任务向量可保留 42% [3, 63]，但通用准确率下降 9.0 分；通过修剪并解决符号冲突的 TIES 可保留 64% [50, 76]，同时通用准确率提升 1.5 分；能够看到所有数据行的联合微调……

    arXiv:2611.27550v1 Announce Type: new  Abstract: Laya is a small model that answers a multiple-choice question in one pass: each option has a name and a written description. A new skill costs a training run, so we test whether published recipes for merging fine-tuned models (weight averaging, task arithmetic, TIES, DARE) work for it, and whether the theories behind them hold. All 156 models start from one released checkpoint: four plain fine-tunes, five specialists (each adds one kind of training row), joint fine-tunes on all rows, and prediction ensembles. Items are held-out, synthetic and LLM-labelled. In-family, each skill is scored on new items of its own kind, and "kept" is the share of a specialist's gain over the plain fine-tune. Averaging four specialists keeps 52% [38, 64] with +1.7 points of general accuracy; adding the full vectors keeps 42% [3, 63] with -9.0; TIES, which trims and resolves sign conflicts, keeps 64% [50, 76] with +1.5; joint fine-tunes, which see all the row
    
[^53]: 关于机器遗忘中覆盖率评估的可观测性极限

    On the Limits of Observability for Evaluating Coverage in Machine Unlearning

    [https://arxiv.org/abs/2611.27543](https://arxiv.org/abs/2611.27543)

    该研究通过502个重训练模型的可观测性压力测试和Ω(log n)秩下界的理论证明，揭示了现有机器遗忘评估因只关注特定遗忘集而忽视“覆盖率”的问题，并提出了评估遗忘覆盖率的CEO框架。

    

    机器遗忘旨在移除任意被请求的训练数据子集的影响，同时保留模型在剩余数据上的效用。现有实证评估通常通过模型输出或中间激活来衡量预测准确率并进行训练集成员推断（MIA）。然而，这些评估由于只聚焦于选定的遗忘集，可能忽视“覆盖率”——即一个方法所能满足的请求范围。为了评估这种覆盖率，我们首先利用502个重训练模型对可观测性进行了压力测试，发现隐藏嵌入能够揭示在预测中被遗漏的已知删除操作。为刻画覆盖率的容量极限，我们证明用线性分数区分所有n个单例删除可能需要秩为Ω(log n)的共享表示。我们进一步将该下界扩展到任意子集，并阐明了部分删除何时可能显得完全。基于这一分析，我们提出了CEO，一个用于评估机器遗忘覆盖率的框架。

    arXiv:2611.27543v1 Announce Type: cross  Abstract: Machine unlearning aims to remove the influence of any requested subset of training data while preserving utility on the remaining data. Empirical evaluations measure predictive accuracy and infer training-set membership (MIAs) from model outputs or intermediate activations. However, by focusing on selected forget sets, these evaluations can overlook Coverage, the range of requests a method can fulfill. To assess this coverage, we first conduct a stress test of observability with 502 retrained models, finding that hidden embeddings reveal known deletions missed in predictions. To characterize capacity limits on coverage, we show that distinguishing all $n$ singleton deletions with linear scores can require a shared representation of rank $\Omega(\log n)$. We further extend the bound to arbitrary subsets and show when partial deletion can appear complete. Building on this analysis, we propose CEO, a framework for evaluating unlearning c
    
[^54]: 修复失效之处：面向场景图生成的根因引导智能体框架优化

    Fix Where It Fails: Root-Cause-Guided Agentic Harness Optimization for Scene Graph Generation

    [https://arxiv.org/abs/2611.27529](https://arxiv.org/abs/2611.27529)

    提出了RoHa-SGG方法，利用根因分析智能体和提议者智能体两个LLM智能体来优化冻结SGG模型周围的推理框架代码，在模型失败之处进行针对性修复而无需重新训练。

    

    场景图生成（SGG）旨在检测图像中的物体并预测它们之间的关系。标准的PredCls、SGCls和SGDet设置只显示了在没有真实边界框和标签时性能下降的程度，但无法揭示模型在哪里或为什么失败。了解这一点将使我们能够在模型失败之处对其进行修复，而无需重新训练。我们提出了RoHa-SGG（面向SGG的根因引导框架优化），将智能体框架优化扩展到SGG领域。它优化的对象是框架，即围绕冻结SGG模型的推理代码。RoHa-SGG包含两个主要的LLM智能体：一个根因分析（RCA）智能体负责找出模型失败的组件并诊断原因，一个提议者智能体负责搜索能够修复这些原因的框架。我们在Visual Genome和PSG数据集上，使用具有代表性的两阶段、一阶段和多模态LLM SGG模型对RoHa-SGG进行评估。分析表明，真实三元组丢失的组件在不同模型之间存在差异（摘要内容在此处被截断）

    arXiv:2611.27529v1 Announce Type: cross  Abstract: Scene graph generation (SGG) aims to detect objects in an image and predict the relationships between them. The standard PredCls, SGCls, and SGDet settings show how much performance drops without ground-truth boxes and labels, but not where or why a model fails. Knowing this would allow fixing a model where it fails, without retraining. We propose RoHa-SGG (Root-cause-guided Harness optimization for SGG), which extends agentic harness optimization to SGG. It optimizes the harness, the inference code around a frozen SGG model. RoHa-SGG has two main LLM agents. A root-cause analysis (RCA) agent finds the components where the model fails and diagnoses why, and a proposer agent searches for a harness that fixes these causes. We evaluate RoHa-SGG on Visual Genome and PSG with representative two-stage, one-stage, and multimodal-LLM SGG models. The analysis shows that the components where ground-truth triplets are lost differ across models an
    
[^55]: PaMuk-Bench：一个面向视觉-语言任务的平行多语言多任务基准

    PaMuk-Bench: A Parallel Multilingual Multitask Benchmark for Vision-Language Tasks

    [https://arxiv.org/abs/2611.27521](https://arxiv.org/abs/2611.27521)

    本文提出了PaMuk-Bench，一个包含超过10K条平行翻译提示（涵盖阿拉伯语、丹麦语、法语、印地语、意大利语和罗马尼亚语六种语言）的多语言多任务视觉-语言基准，用于系统评估视觉-语言模型在低资源语言上跨多种任务的表现。

    

    尽管视觉-语言模型（VLM）目前在英语等被广泛研究的语言上取得了显著成果，但由于缺乏标注数据集，它们在某些低资源语言上表现出显著的性能下降。为此，我们提出了PaMuk-Bench，一个新颖的面向视觉-语言任务的平行多语言多任务基准。我们的基准包含一组超过10K条精选的英文提示，这些提示被翻译成六种语言：阿拉伯语、丹麦语、法语、印地语、意大利语、罗马尼亚语。通过半自动过程获得的平行翻译，以及配套的生成和检索图像，使得能够在广泛的任务上对VLM进行对比性开发和评估，这些任务包括：文本到图像生成、图像描述生成、基于内容的图像检索、图像生成难度估计、图像检索难度估计。我们通过实验证明，零样本英语和多语言预训练……

    arXiv:2611.27521v1 Announce Type: cross  Abstract: While vision-language models (VLMs) currently obtain notable results for highly-studied languages, e.g. English, they exhibit significant performance drops in certain low-resource languages, especially due to the lack of annotated datasets. To this end, we propose PaMuk-Bench, a novel parallel multilingual multitask benchmark for vision-language tasks. Our benchmark includes a set of over 10K curated English prompts that are translated into six languages: Arabic, Danish, French, Hindi, Italian, Romanian. Parallel translations, which are obtained via a semi-automatic process, along with accompanying generated and retrieved images enable the comparative development and evaluation of VLMs across a wide range of tasks: text-to-image generation, image captioning, content-based image retrieval, image generation difficulty estimation, image retrieval difficulty estimation. We empirically show that zero-shot English and multilingual pre-traine
    
[^56]: PhiJEPA：用于视觉表征学习的去噪潜在预测

    PhiJEPA: Denoising Latent Predictions for Visual Representation Learning

    [https://arxiv.org/abs/2611.27516](https://arxiv.org/abs/2611.27516)

    PhiJEPA提出了一种两阶段去噪潜在预测架构，由Transformer从损坏上下文预测掩码教师视图的隐藏token、再由逐token的MLP将其映射到干净token，在ImageNet-1K上预训练后在PASCAL VOC线性分割中达到69.3 mIoU，超越I-JEPA和DSeq-JEPA且训练计算量减少25%。

    

    一个被训练用于推断图像隐藏区域的预测器，能否在物体部分从视野中消失且可见上下文变得模糊时，仍然帮助世界模型进行规划？我们提出了PhiJEPA，这是一种联合嵌入预测架构，将PhiNet的两阶段预测方式适配到掩码Transformer中。从稀疏的、经过光度损坏的上下文中，一个Transformer块预测独立损坏的教师视图的隐藏token。随后，一个逐token的MLP将这些预测映射到干净的教师token。受控消融实验表明，该组合目标在密集特征方面优于所测试的单阶段替代方案。在ImageNet-1K上以5.6×10^19训练FLOPs进行预训练后，PhiJEPA在线性PASCAL VOC分割任务上达到69.3 mIoU，在分割精度上超过了所评估的I-JEPA和DSeq-JEPA检查点，且训练FLOPs比I-JEPA减少25%。它还在隐藏区域分割方面优于Simple I-JEPA（49.1 vs. 44…）。

    arXiv:2611.27516v1 Announce Type: cross  Abstract: Can a predictor trained to infer hidden image regions still help a world model plan when part of an object disappears from view and the visible context is blurred? We introduce PhiJEPA, a joint-embedding predictive architecture that adapts PhiNet's two-stage prediction to masked Transformers. From sparse, photometrically corrupted context, a Transformer block predicts hidden tokens of an independently corrupted teacher view. A token-wise MLP then maps these predictions to clean teacher tokens. Controlled ablations show that the combined objective improves dense features over the tested one-stage alternatives. Pre-trained on ImageNet-1K with $5.6\times10^{19}$ training FLOPs, PhiJEPA achieves 69.3 mIoU on linear PASCAL VOC segmentation, exceeding the evaluated I-JEPA and DSeq-JEPA checkpoints in segmentation accuracy, with 25\% fewer training FLOPs than I-JEPA. It also improves hidden-region segmentation over Simple I-JEPA (49.1 vs.\ 44
    
[^57]: 基于异步事件的自监督连续时间深度学习与自运动估计

    Self-Supervised Continuous-Time Depth Learning and Ego-Motion Estimation from Asynchronous Events

    [https://arxiv.org/abs/2611.27509](https://arxiv.org/abs/2611.27509)

    该论文提出了一种自监督的连续时间深度学习方法，为每个异步事件分配独立深度并从同一观测中恢复相机自运动，利用事件间的几何一致性作为监督信号，并通过可微渲染生成目标时刻的稠密深度图，在MVSEC数据集上超越了先前仅基于事件的自监督方法。

    

    事件相机以微秒级精度逐像素地记录亮度变化，而可用于监督它们的稠密深度数据获取成本高昂，且仅存在于孤立的时刻。将事件流压缩成帧虽然便于学习，但代价是牺牲了事件感知赖以独特的高时间分辨率特性。我们提出了一种自监督方法，其中每个事件在连续时间中被赋予其自身的深度，同时相机运动也从相同的观测中恢复出来。监督信号源于几何：一个事件通过其预测深度被提升到三维空间，并经过它在触发瞬间的相机运动位移后，应当与由同一条边缘产生的其他事件相重合，因此任何错位都会暴露深度或运动估计中的误差。随后，可微渲染将这些逐事件的估计转化为每个窗口内目标时刻的稠密深度图，并忠实于其背后的事件。在MVSEC数据集上，该方法优于先前仅基于事件的自监督方法。

    arXiv:2611.27509v1 Announce Type: cross  Abstract: Event cameras register brightness changes pixel by pixel with microsecond precision, whereas the dense depth that could supervise them is costly to acquire and exists only at isolated instants. Condensing the stream into frames eases learning, but at the expense of the very timing that sets event sensing apart. We propose a self-supervised approach in which each event is assigned its own depth in continuous time, and the camera motion is recovered from the same observations. Supervision arises from geometry: lifted into 3D by its predicted depth and displaced by the camera motion at the instant it fired, an event should coincide with others produced by the same edge, so any misalignment exposes an error in depth or motion. Differentiable rendering then turns these per-event estimates into dense depth maps at target times within each window, faithful to the events behind them. On MVSEC, the approach outperforms prior event-only self-sup
    
[^58]: 助手人格在激活空间中因语言而异

    The Assistant Persona Differs Across Languages in Activation Space

    [https://arxiv.org/abs/2611.27504](https://arxiv.org/abs/2611.27504)

    该研究首次从机制层面揭示了大语言模型中助手人格方向在不同语言间差异显著（与英语轴余弦相似度仅0.378–0.911）且与语言资源水平相关，证明从英语提取的激活方向无法跨语言泛化。

    

    大语言模型在不同语言之间的行为存在差异，且在低资源语言中明显不够安全。这一差距在行为层面已被充分证实，但在机制层面尚未得到解释。激活空间中的方向，包括助手人格方向，几乎总是从英语中提取并被假定可以跨语言泛化。我们在 Qwen3-32B 中提取了31种语言的助手人格方向，发现所得的轴存在显著差异：它们与英语轴的余弦相似度介于0.378到0.911之间。此外，每种语言的助手人格方向及其幅值都与该语言的资源水平相关。我们还发现，模型的默认助手状态在低资源语言中距离“乐于助人的助手”这一极点更远。最后，相较于沿英语轴进行干预（steering），沿语言自身的轴进行干预会更强烈地放大基于人格的越狱攻击。这些结果表明……

    arXiv:2611.27504v1 Announce Type: cross  Abstract: Large language models behave differently across languages, and are noticeably less safe in lower-resource languages. This gap is well established behaviorally, but not mechanistically. Directions in activation space, including the assistant persona direction, are almost always extracted from English and assumed to generalize across languages. We extract the assistant persona direction in 31 languages in Qwen3-32B, and find that the resulting axes differ substantially: their cosine similarity to the English axis ranges from 0.378 to 0.911. Moreover, each language's assistant persona direction and its magnitude are correlated with the language's resource level. We also find that the model's default assistant state is farther from the helpful assistant pole in lower-resource languages. Finally, steering against a language's own axis amplifies persona-based jailbreak more than steering against the English axis. These results indicate that 
    
[^59]: Turnout：面向高效推理的自适应注意力切换方法

    Turnout: Adaptive Attention Switching for Efficient Reasoning

    [https://arxiv.org/abs/2611.27500](https://arxiv.org/abs/2611.27500)

    提出Turnout方法，在预训练全注意力模型中通过蒸馏保留线性注意力路径，并利用强化学习训练的自适应策略在每一层动态切换全注意力与线性注意力，从而在大幅降低内存消耗的同时保持长文本推理任务的准确率。

    

    事后transformer线性化用固定大小的状态取代全注意力中随生成长度线性增长的键值缓存。这一改变降低了内存消耗，但往往会损害准确率，而在需要长文本生成的推理任务中，这种影响尤为显著。在本工作中，我们提出了Turnout，一种能够在保持线性化内存缩减优势的同时缓解其准确率损失的方法。我们将事后线性化应用于预训练的全注意力模型，通过蒸馏训练出一条线性路径，同时保持原有的全注意力路径不变。在响应生成过程中，所得的混合模型同时保留两条路径，并利用一个学习到的策略来控制每一层何时从全注意力切换到线性注意力。该策略通过强化学习进行训练，奖励函数在答案正确性与内存使用之间取得平衡。对我们方法的实证评估表明……

    arXiv:2611.27500v1 Announce Type: cross  Abstract: Post-hoc transformer linearization replaces the key-value cache of full attention, which grows linearly with generation length, with a fixed size state. This change reduces memory consumption but often hurts accuracy. In reasoning tasks, which require long generations, this effect is especially pronounced. In this work we introduce Turnout, a method that mitigates accuracy loss as a result of linearization, while maintaining its memory-reduction benefits. We apply post-hoc linearization to a pretrained full-attention model, training a linear path through distillation while leaving the original full-attention path unchanged. During response generation, the resulting hybrid model retains both paths and uses a learned policy to control when each layer switches from full to linear attention. We train the policy with reinforcement learning using a reward that balances answer correctness and memory use. Empirical evaluation of our method sho
    
[^60]: 在有害数据上训练，仍然拒绝：一种面向代码生成的训练时防御方法

    Trained on Harm, Still Refusing: A Training-Time Defense for Code Generation

    [https://arxiv.org/abs/2611.27495](https://arxiv.org/abs/2611.27495)

    提出SAFEHOLD防御框架，将拒绝行为建模为持续的生成过程而非单次决策，使代码语言模型即使经过有害数据微调仍能保持拒绝有害请求的安全对齐能力。

    

    微调使安全对齐的代码语言模型能够适应多样化的下游任务。然而，在不可信数据上训练会严重削弱其拒绝有害请求的能力。值得注意的是，我们发现即使有害性信息在其内部表示中仍然高度可解码，模型仍会遵从有害请求。此外，最初的拒绝偏好并不一定会在整个生成过程中转化为持续的拒绝行为。这些发现共同揭示了识别有害请求与持续拒绝执行之间存在的关键差距。为了解决这一差距，我们提出了SAFEHOLD，一种新颖的由提供方强制执行的防御框架，它将拒绝视为一个持续的生成过程而非单一的响应决策，从而在有害微调期间保持安全对齐。具体而言，SAFEHOLD在初始化阶段保留对齐模型的全词表响应分布……

    arXiv:2611.27495v1 Announce Type: new  Abstract: Fine-tuning enables safety-aligned code language models to adapt to diverse downstream tasks. However, training on untrusted data can severely undermine their ability to refuse harmful requests. Notably, we find that models comply with harmful requests even though harmfulness information remains highly decodable from their internal representations. Moreover, an initial preference for refusal does not necessarily translate into sustained refusal behavior throughout generation. Together, these findings reveal a critical gap between recognizing harmful requests and consistently refusing to fulfill them. To address this gap, we introduce SAFEHOLD, a novel provider-enforced defense framework that preserves safety alignment during harmful fine-tuning by treating refusal as a sustained generation process rather than a single response decision. Specifically, SAFEHOLD preserves the aligned model's full-vocabulary response distribution at the init
    
[^61]: 从老虎机反馈中学习未知振荡器

    Learning Unknown Oscillators From Bandit Feedback

    [https://arxiv.org/abs/2611.27492](https://arxiv.org/abs/2611.27492)

    提出了一种带检验的承诺控制器，在未知振荡器驱动的半老虎机问题上实现了平方根级后悔界，并将拟合次数从1023次降至49次，带来4.69倍的加速。

    

    我们研究由未知振荡器驱动的半老虎机问题。每一轮，学习者从四个奖励通道中选择两个并观察各自的奖励。频率误差会改变未来的奖励排序以及由此产生的测量结果。我们展示了何时承诺于一个已拟合的振荡器，可以让其贪心选择暴露预测误差。我们的控制器在连续置信集上界定奖励损失（后悔值），并在预算内保持最长的时间块。当检验失败时，在线回归策略会收集观测数据。在所有频率下，后悔值以平方根速率增长（最多相差对数因子），并且该方法可扩展到两个相互竞争的振荡器。在96个新的合成实例上，带检验的承诺方法仅使用49次拟合而非1023次，平均后悔值仅比每轮拟合高1.5%。在匹配的独立计时下实现了4.69倍的加速。该置信集还能保证轨迹后悔值。

    arXiv:2611.27492v1 Announce Type: cross  Abstract: We study semi-bandits driven by unknown oscillators. Each round, a learner selects two of four reward channels and observes their individual rewards. Frequency errors change future reward rankings and the resulting measurements. We show when committing to one fitted oscillator lets its greedy choices expose prediction errors. Our controller bounds reward loss (regret) over a continuous confidence set and holds the longest block within budget. An online regression policy gathers observations when the check fails. Regret grows at a square-root rate, up to logarithmic factors, across all frequencies and extends to two competing oscillators. On 96 fresh synthetic instances, checked commitment uses 49 fits instead of 1,023, with 1.5% higher mean regret than fitting every round. Matched isolated timings give a 4.69-fold speedup. The confidence sets also certify trajectory regret.
    
[^62]: 无迁移的对齐：分词器影响多语言语言模型预训练过程中的跨语言表示

    Alignment without Transfer: Tokenization Impacts Cross-lingual Representations during Multilingual Language Model Pretraining

    [https://arxiv.org/abs/2611.27491](https://arxiv.org/abs/2611.27491)

    本研究通过仅改变分词器预训练39个单语和双语模型，发现改进的词表分配能显著增强模型中早期层的跨语言表示对齐以及前馈神经元的跨语言知识共享，但揭示了表示对齐与跨语言迁移之间存在分离现象。

    

    分词是多语言自然语言处理流程中最关键的步骤之一。然而，分词器通常对低资源语言的词表分配不足。尽管改进的词表分配能够带来更好的性能，但分词如何在预训练过程中塑造跨语言表示与迁移仍鲜为人知。为解决这一问题，我们在FineWeb2-HQ数据集上预训练了39个单语和双语的10亿参数解码器模型，涵盖英语、德语、法语、阿拉伯语和中文，仅改变分词器这一变量。我们的结果表明，改进的词表分配 显著提升了模型早期至中期层的跨语言表示对齐，这一效应是先前仅测量输出层的工作所无法观察到的； 增加了前馈神经元中的跨语言参数化知识共享，并得到了相关性证据和因果证据的支持； 揭示了“对齐-迁移分离”现象，即跨语言迁移基本保持不变。原文在此处截断。

    arXiv:2611.27491v1 Announce Type: cross  Abstract: Tokenization represents one of the most crucial steps in mulilingual NLP pipelines. Yet, tokenizers routinely under-allocate vocabulary to low-resource languages. While improved allocation leads to better performance, it remains poorly understood how tokenization shapes cross-lingual representations and transfer during pretraining. To address this, we pretrain 39 mono- and bilingual 1B decoder-only models on FineWeb2-HQ across English, German, French, Arabic and Chinese, varying only the tokenizer. Our results demonstrate that improved vocabulary allocation (i) substantially raises cross-lingual representation alignment in early-to-mid layers, an effect that prior work measuring only the output layer could not observe, (ii) increases cross-lingual parametric knowledge sharing in feed-forward neurons, supported by correlational and causal evidence, and (iii) an alignment-transfer dissociation, where cross-lingual transfer remains essent
    
[^63]: 面向实时物理人工智能的高效自然语言到时序逻辑翻译

    Efficient Natural Language to Temporal Logic Translation for Real-Time Physical AI

    [https://arxiv.org/abs/2611.27489](https://arxiv.org/abs/2611.27489)

    该论文提出将自然语言到线性时序逻辑的翻译分解为回答一组预定义答案的问题，用轻量级System-One分类器替代大语言模型的文本生成，从而在计算和能源资源受限的机器人上实现高效实时的翻译。

    

    当任务指令以形式化语言表述时，机器人可以被控制执行任务并保证正确性。大语言模型（LLM）已被广泛用于让用户以自然语言（NL）提供此类指令，即将自然语言翻译为形式化语言。然而，将此类翻译器部署在机器人上时，我们面临计算和能源资源有限的问题，而将计算或数据卸载到云端可能不可行或不理想。此外，大语言模型是一个通用文本生成器，而固定形式化语言片段中的公式遵循明确定义的语法。因此，我们提出疑问：是否真的需要文本生成。对于线性时序逻辑（LTL）这一形式化语言，我们证明这种翻译可以转而分解为回答一组具有预定义答案的问题。随后，我们构建了一个System-One模型——一类新兴的基于自然语言的分类器——来提供这些答案，从而实现……

    arXiv:2611.27489v1 Announce Type: new  Abstract: Robots can be controlled to perform tasks with guaranteed correctness when the task instructions are stated in a formal language. Large language models (LLMs) have been widely used to allow users to provide such instructions in natural language (NL), by translating NL into formal language. When deploying such translators on a robot, however, we face limited computing and energy resources, while offloading computation or data to the cloud may be infeasible or undesirable. Yet an LLM is a general-purpose text generator, whereas a formula in a fixed formal language fragment follows a well-defined syntax. We therefore ask whether text generation is needed at all. For the formal language of linear temporal logic (LTL), we show that this translation can instead be broken down into answering a set of questions with pre-defined answers. We then construct a System-One model, a recent class of NL-based classifier, to provide such answers and enabl
    
[^64]: 从制造事件日志到仿真：利用智能体AI赋能中小企业进行SimPy仿真生成

    From Manufacturing Event Logs to Simulation: Enabling SMEs with Agentic AI for SimPy Simulation Generation

    [https://arxiv.org/abs/2611.27483](https://arxiv.org/abs/2611.27483)

    提出了GenAI4SimPy多智能体框架，利用开放权重大语言模型Qwen 3.6将制造事件日志自动转化为可执行的SimPy离散事件仿真模型，并揭示了“保真度悖论”——过程级参数准确并不保证系统级行为保真。

    

    高昂的专业知识和成本门槛将中小企业排除在离散事件仿真（DES）之外。我们提出了GenAI4SimPy，一个由开放权重大型语言模型Qwen 3.6（27B）驱动的多智能体框架，通过工具增强推理和人机协同验证，实现从制造事件日志到可执行SimPy模型的端到端DES生成自动化。在公共智慧工厂数据集的三个执行区块上进行评估，该框架实现了7.9%的平均绝对百分比误差，并在全部三个区块上实现了故障率的覆盖率，在三个区块中的两个上实现了吞吐量的覆盖率。研究观察到一个保真度悖论：准确的过程级参数并不能保证系统级的行为保真度，流动时间和吞吐量误差源于对观察窗口截断的误解以及系统级校准的缺失。这些结果表明，本地开放权重模型可以以可接受的成本执行DES生成……

    arXiv:2611.27483v1 Announce Type: new  Abstract: High expertise and cost barriers exclude small and medium-sized enterprises from discrete-event simulation (DES). We introduce GenAI4SimPy, a multi-agent framework powered by Qwen 3.6 (27B), an open-weight large language model, that automates end-to-end DES generation from manufacturing event logs to executable SimPy models through tool-augmented reasoning and human-in-the-loop validation. Evaluated on three execution blocks from a public smart factory dataset, the framework achieved a mean absolute percentage error of 7.9\% and coverage of failure rate for all three blocks and of throughput for two of three. A fidelity paradox was observed: accurate process-level parameters did not ensure system-level behavioral fidelity, with flow-time and throughput errors driven by misinterpreted observation-window truncation and absent system-level calibration. These results demonstrate that local open-weight models can perform DES generation at acc
    
[^65]: ROSWORLD：当数字任务能力遇上物理执行

    ROSWORLD: When Digital Task Competence Meets Physical Execution

    [https://arxiv.org/abs/2611.27467](https://arxiv.org/abs/2611.27467)

    ROSWorld基准将物理仿真机器人工作站与实时OSWorld桌面相耦合，揭示前沿计算机使用模型虽具备强大的数字任务能力，但在通过物理接触操作键盘和鼠标时性能显著下降。

    

    计算机使用智能体现在可以通过直接软件输入解决大多数桌面任务，但使用同一台计算机的机器人必须将GUI知识转化为与键盘和鼠标的物理接触。我们提出了ROSWorld，这是一个将物理仿真的机器人工作站与实时OSWorld桌面相耦合的基准测试，使机器人和桌面智能体在相同的官方评估器下追求相同的目标。被评估的模型仅从工作站摄像头视角获取观测，并通过单臂刚性探针或双臂灵巧手操作外设。在五个前沿模型和十个应用中的40个任务上，在桌面上可完成31-39个任务的模型，使用探针最多完成23个，使用灵巧手最多完成28个。成对轨迹分析表明，模型会将交互策略调整为倾向于键盘输入，但物理执行经常因按键未命中、目标选择错误和意外的并发输入而失败。

    arXiv:2611.27467v1 Announce Type: cross  Abstract: Computer-use agents now solve most desktop tasks through direct software input, but a robot using the same computer must turn GUI knowledge into physical contact with a keyboard and mouse. We introduce ROSWorld, a benchmark that couples a physically simulated robot workstation to a live OSWorld desktop, so that a robot and a desktop agent pursue the same goals under the same official evaluators. The evaluated model receives observations only from the workstation camera views and operates the peripherals through either a single-arm rigid probe or dual-arm dexterous hands. Across five frontier models and 40 tasks in ten applications, models that complete 31-39 tasks on the desktop complete at most 23 with the probe and 28 with the hands. Paired trajectory analysis shows that models adapt their interaction strategies toward keyboard input, yet physical execution frequently fails through missed presses, incorrect targets, and unintended co
    
[^66]: Cecilia：迈向包含版面信息的光学乐谱识别

    Cecilia: Towards Layout-Inclusive Optical Music Recognition

    [https://arxiv.org/abs/2611.27465](https://arxiv.org/abs/2611.27465)

    该论文提出Cecilia——一种保留音乐内容、歌词与版面指令的紧凑MusicXML表示，并构建Cecilia-OMR模型以元数据和小节级片段进行预测，序列长度仅为MusicXML的约1/28，同时在多项基准上优于Legato 2。

    

    重构一份可编辑的乐谱需要音乐事件、演唱文本以及将它们组织在页面上的排版指令。现有的识别目标往往遗漏其中一部分信息，而直接预测MusicXML则会重复大量结构性标记。我们提出Cecilia，这是一种对规范化MusicXML的紧凑表示，它保留了音乐内容、歌词附着关系以及版面指令。Cecilia-OMR通过共享的视觉记忆，以元数据和以小节为边界的音乐片段的形式来预测这种表示，并利用一个可学习的计数头确定输出片段的数量。在使用同一分词器的配对完整乐谱上，MusicXML与Cecilia的平均序列长度之比为27.73。Cecilia-OMR在室内弦乐四重奏以及两种艺术歌曲条件下均优于Legato 2，在四个条件的OpenScore上宏平均归一化编辑距离为0.2232，而对比方法为0.2998。（原文摘要此处截断：对歌词、记谱树与渲染结果的独立评估……）

    arXiv:2611.27465v1 Announce Type: cross  Abstract: Reconstructing an editable music score requires musical events, sung text and the instructions that organise them on the page. Existing recognition targets often omit part of this information, while direct MusicXML prediction repeats substantial structural markup. We introduce Cecilia, a compact representation of canonicalised MusicXML that retains musical content, lyric attachments and layout instructions. Cecilia-OMR predicts this representation through metadata and measure-bounded music segments from shared visual memory. A learned count head determines the number of output segments. Across paired complete scores under a common tokeniser, the ratio of mean MusicXML to Cecilia sequence lengths is 27.73. Cecilia-OMR outperforms Legato 2 on Camera String Quartets and both Lieder conditions, with a four-condition OpenScore macro normalised edit distance of 0.2232 against 0.2998. Separate evaluations of lyrics, notation trees and rendere
    
[^67]: 关于模糊性与不确定性的超赋值主义：一种判断聚合的视角

    Supervaluationism about vagueness and indeterminacy: A judgment-aggregation perspective

    [https://arxiv.org/abs/2611.27463](https://arxiv.org/abs/2611.27463)

    本文引入判断聚合理论，将超赋值主义视为聚合不同规格说明判定结果的一类聚合函数之一，并对其给出公理化刻画，从而精确界定了它在处理模糊性与不确定性的各种可能方法中的逻辑位置。

    

    关于模糊性与不确定性的超赋值主义基于这样一种思想：某些句子或其中的谓词具有多种相互竞争的解释或规格说明。当且仅当一个句子在所有规格说明下都为真时，它才被判定为确定为真；当且仅当它在所有规格说明下都为假时，它才被判定为确定为假；在所有其他情况下，该句子则不被赋予确定真值。然而，超赋值主义只是顾及不同规格说明的判定结果的一种方式——用技术术语来说，即是对这些判定结果的一种“聚合”。本文引入判断聚合理论中的见解，研究了一大类不同的聚合函数。这使得我们能够对超赋值主义给出公理化刻画，进而将其精确地定位于处理模糊性与不确定性的各种可能方法所构成的更大逻辑空间之中。在其他结果中，

    arXiv:2611.27463v1 Announce Type: new  Abstract: Supervaluationism about vagueness and indeterminacy is based on the idea that some sentences, or predicates within them, have multiple competing interpretations or specifications. A sentence is deemed determinately true if and only if it is true according to all specifications; and determinately false if and only if it is false according to all specifications. In all other cases, the sentence is assigned no determinate truth-value. Supervaluationism, however, is only one way of taking into account the verdicts of different specifications -- in technical terms, "aggregating" those verdicts. In this paper, we introduce insights from the theory of judgment aggregation to study a broad class of different aggregation functions. This allows us to offer an axiomatic characterization of supervaluationism, which, in turn, locates it precisely within a larger logical space of possible approaches to vagueness and indeterminacy. Among other results,
    
[^68]: 关于在高风险AI招聘中使用大语言模型的挑战：符合《人工智能法案》法律合规性的指导性问题

    On the Challenges of Using LLMs in High-Risk AI Recruitment: Guiding Questions for Legal Conformity with the AI Act

    [https://arxiv.org/abs/2611.27462](https://arxiv.org/abs/2611.27462)

    本文提出一个工具包，将欧盟《人工智能法案》合规评估的核心法律义务转化为技术规范要求和指导性问题，以支持“合法设计”的AI辅助招聘系统开发，重点关注第三方GPAI模型集成到招聘关键环节这一最具挑战性的场景。

    

    欧盟《人工智能法案》（AI Act）对高风险AI系统的提供者规定了详细的义务，但在操作层面提供的指导有限，尤其是当这些系统集成了第三方通用人工智能（GPAI）模型时。本文介绍了一个工具包，通过将AI法案合规评估的核心法律义务——具体为第9-15条和附件IV——转化为技术规范要求及相应的指导性问题，从而促进“合法设计”的AI辅助招聘系统的开发。虽然招聘流程可能包含技术复杂程度不同、算法透明度各异的组件，但本文聚焦于最具挑战性的情形：将第三方GPAI模型集成到选拔过程的关键阶段，包括简历与职位描述解析、申请人评分以及入围名单生成。这一情形带来了重大的挑战……

    arXiv:2611.27462v1 Announce Type: cross  Abstract: The EU Artificial Intelligence Act (AI Act) imposes detailed obligations on providers of high-risk AI systems but offers limited operational guidance, particularly when these systems integrate third-party General-Purpose AI (GPAI) models. This paper introduces a toolkit that fosters the development of legal-by-design AI-assisted recruitment systems by translating the core legal obligations of the AI Act's conformity assessment---specifically Articles 9--15 and Annex IV---into techno-normative requirements and corresponding guiding questions. While recruitment pipelines may involve components with varying technical complexity and differing levels of algorithmic transparency, this paper focuses on the most challenging case: the integration of third-party GPAI models into critical stages of the selection process, including resume and job description parsing, applicant scoring, and shortlist generation. This scenario presents significant c
    
[^69]: 面向自动驾驶感知仿真验证的机制对等性

    Mechanism Parity for Simulation-Based Validation of Autonomous-Driving Perception

    [https://arxiv.org/abs/2611.27447](https://arxiv.org/abs/2611.27447)

    本文提出“机制对等性”诊断框架，通过统一决定性特征保真度（DFF）与概念-决定性特征保真度（CDFF），验证输出一致的仿真是否保留了感知模型做出判断的内部证据机制，而不仅是视觉逼真或输出一致。

    

    摘要：仿真在智能车辆验证中处于核心地位，但真实场景与其合成对应物之间的视觉真实感或输出一致性，并不能证明感知系统在两个领域中使用了相同的证据。本文针对配对的真实-合成自动驾驶感知中的“机制对等性缺口”展开研究。该工作将“决定性特征保真度”（比较冻结的被测系统（SUT）中的决定性证据图）与“概念-决定性特征保真度”（将决定性证据投影到语义场景概念中用于诊断）统一起来。该框架是一种验证诊断工具，而非独立的安全证书：它追问的是，输出一致的仿真是否同样保留了感知模型得出该输出所依赖的机制。实证证据包括在确定性精确几何的 KITTI-Virtual KITTI 2 子集上评估的十个重新训练的扩散 ControlNet 生成器。（原文摘要在此处截断）

    arXiv:2611.27447v1 Announce Type: cross  Abstract: Simulation is central to intelligent-vehicle validation, but visual realism or output agreement between a real scene and its synthetic counterpart does not prove that a perception system used the same evidence in both domains. This paper addresses that mechanism-parity gap for paired real-synthetic autonomous-driving perception. It unifies Decisive-Feature Fidelity, which compares decisive evidence maps in a frozen system under test (SUT), with Concept-Decisive Feature Fidelity, which projects decisive evidence into semantic scene concepts for diagnosis. The framework is a validation diagnostic rather than a standalone safety certificate: it asks whether output-consistent simulation also preserves the mechanism by which the perception model reaches that output. The empirical evidence comprises ten retrained diffusion ControlNet generators evaluated on a deterministic exact-geometry KITTI-Virtual KITTI 2 subset. Generated counterparts u
    
[^70]: 学习何时求助：面向小型语言模型智能体的顾问感知后训练

    Learning When to Ask: Advisor-Aware Post-Training for SLM Agents

    [https://arxiv.org/abs/2611.27443](https://arxiv.org/abs/2611.27443)

    提出ADAPT多阶段后训练方法，让小型语言模型智能体学会在成本感知下判断何时向更强模型求助并跨步骤遵循建议，且完全依靠环境结果监督的策略自身rollouts进行学习，无需奖励模型或人工标注。

    

    小型语言模型（SLM）因其低成本、高吞吐量的特点，在智能体部署中越来越具有吸引力，但它们在长程任务所需的多步推理和工具使用方面仍然薄弱。我们研究如何对SLM进行后训练，使其能够向更强的模型寻求建议并自己执行这些建议，即遵循“顾问-执行者”模式。然而开箱即用时，这种模式在开源和闭源的各种小模型家族中均告失败：仅仅将SLM连接到顾问上只是一个框架设计问题，本身并不足够。需要一种后训练方案来教会模型以成本感知的方式判断何时提问，以及如何跨步骤遵循建议。我们的方法ADAPT（ADvisor-Aware Post-Training，顾问感知后训练）是一种多阶段训练方案，能够培养这种行为，它完全从策略自身生成的rollouts中学习，并由环境的最终结果进行监督，无需学习奖励模型或人工标注。在三个领域中，与未训练的框架基线相比，ADAPT……

    arXiv:2611.27443v1 Announce Type: cross  Abstract: Small language models (SLMs) are increasingly attractive for cheap, high-throughput agentic deployment, but they remain weak at multi-step reasoning and tool use that long-horizon tasks demand. We study how to post-train an SLM to seek advice from a stronger model and act on that advice itself, following an advisor-executor pattern. Out of the box, this pattern fails across open- and closed-source small-model families: wiring an SLM to an advisor is a harness-design matter that is not enough on its own. A post-training recipe is needed to teach the model when to ask, in a cost-aware fashion, and how to follow advice across steps. Our method, ADAPT (ADvisor-Aware Post-Training), is a multi-stage recipe that instills this behavior, learning entirely from the policy's own rollouts supervised by the environment's outcome, with no learned reward model or human annotation. Across three domains, compared to an untrained harness baseline, ADAP
    
[^71]: THRIFTY：面向小语言模型的成本受限推理数据合成

    THRIFTY: Cost-Constrained Reasoning-Data Synthesis for Small Language Models

    [https://arxiv.org/abs/2611.27436](https://arxiv.org/abs/2611.27436)

    提出THRIFTY——一种基于Thompson采样的带背包约束多臂老虎机控制器，在固定的生成token预算下自适应地将预算分配给不同难度的问题，从而低成本地为小语言模型合成多样且正确的推理训练数据。

    

    小语言模型（SLM）因其在高效、专业化部署方面的吸引力与日俱增，通常在从更大的教师大语言模型（LLM）蒸馏得到的推理轨迹上进行训练。生成这类合成数据是该训练流程中的主要成本，尤其当教师模型是前沿模型时。然而，现有的数据收集方法大多不考虑成本，或者至多将固定的采样预算均匀分配到各个查询上。这种分配方式忽略了问题难度的巨大差异，以及由此引发的策略性过度采样或采样不足。因此，我们提出THRIFTY，一种基于Thompson采样的多臂老虎机控制器，它在固定的生成token预算内收集多样且正确的推理轨迹，将这一任务建模为带背包约束的多臂老虎机问题，并引入了新颖的多样性感知目标。该方法通过在线Beta后验跟踪每个问题的成功概率，并将每个token花费在预期产生最多新的、不同的正确解之处（摘要在此处截断）。

    arXiv:2611.27436v1 Announce Type: cross  Abstract: Small language models (SLMs) are increasingly attractive for efficient, specialized deployment, and are typically trained on reasoning trajectories distilled from a larger teacher LLM. Generating this synthetic data is a major cost of the pipeline, especially when the teacher is a frontier model. Existing collection methods, however, are largely cost-unaware, or at best spread a fixed sampling budget uniformly across queries. Such allocation ignores the wide variation in problem difficulty and the strategic over- or under-sampling it invites. Thus, we propose THRIFTY, a Thompson-sampling bandit controller that spends a fixed generation-token budget to collect diverse correct reasoning trajectories, casting the task as a bandits-with-knapsacks problem with a novel diversity-aware objective. It tracks each problem's success probability with an online Beta posterior and spends each token where it expects the most new distinct correct solu
    
[^72]: 无音频、无转写文本，无妨：仅使用领域文档的基于LLM的ASR适配

    No Audio, No Transcripts, No Problem: LLM-based ASR Adaptation Using Only Domain Documents

    [https://arxiv.org/abs/2611.27434](https://arxiv.org/abs/2611.27434)

    该研究首次提出仅使用领域文档（无需目标音频和转写文本）对基于LLM的语音识别系统进行领域适配，通过将文档直接作为书面文本或由LLM渲染为合成口语对话，其适配效果可媲美甚至接近使用真实转写文本的结果。

    

    能否在没有目标音频和转写文本的情况下，仅使用书面文档将语音识别器适配到新领域？这是现实的接入场景：新客户交付的是原始文档，而非纯文本适配通常所假设的转写文本。我们通过两种方式将领域文档转化为适配文本：直接作为书面句子，或由大语言模型（LLM）渲染成合成口语对话，并将其输入最近提出的一种基于去噪的纯文本适配方法。我们表明，在两个语料库上的域内、域外和跨域设置中，仅使用文档的适配可以达到或非常接近使用真实转写文本的效果；书面渲染还是对话式渲染更优取决于说话风格和生成的文本量。据我们所知，这是首个针对基于LLM的ASR进行仅文档适配的研究。由于目前没有现有语料库同时包含音频、转写文本和文档……

    arXiv:2611.27434v1 Announce Type: cross  Abstract: Can a speech recognizer be adapted to a new domain with no target audio and no transcripts, using only written documentation? This is the realistic onboarding setting: a new customer hands over raw documentation, not the transcripts that text-only adaptation normally assumes. We turn a domain's documents into adaptation text in two ways, directly as written sentences or rendered by a large language model (LLM) into synthetic spoken dialogs, and feed them to a recently proposed denoising-based text-only adaptation method. We show that, across in-domain, out-of-domain, and cross-domain settings on two corpora, document-only adaptation can match or closely approach real transcriptions; whether written or conversational rendering is best depends on speaking style and the amount of text generated. To our knowledge, this is the first study of document-only adaptation of LLM-based ASR. As no existing corpus pairs audio, transcripts, and docum
    
[^73]: 基于知识增强结构化推理的神经退行性疾病鉴别诊断

    Neurodegenerative Differential Diagnosis with Knowledge-enhanced Structural Reasoning

    [https://arxiv.org/abs/2611.27427](https://arxiv.org/abs/2611.27427)

    提出了知识增强结构化推理框架KnowSR，通过将多模态患者表征与基于文献的疾病知识库（疾病原型、带符号证据方向和两两诊断关系）对齐，注入结构化疾病证据并在稀疏鉴别图上进行关系推理，从而提升神经退行性疾病的鉴别诊断能力。

    

    由于神经退行性疾病的临床表现相互重叠，且多模态观测数据异构且常常不完整，其准确的鉴别诊断仍然充满挑战。现有的多模态模型虽然能够整合影像与临床测量数据，但往往忽视了疾病特异性的证据模式以及竞争性诊断之间的两两关系。因此，我们提出了知识增强结构化推理，将多模态患者表征与一个基于文献构建的知识库对齐，该知识库包含疾病原型、带符号的证据方向以及两两诊断关系。具体而言，在KnowSR中，患者特定的知识对齐与目标感知池化注入结构化的疾病证据，而锚定关系推理则在稀疏的鉴别图上协调两两诊断边界，从而共同增强鉴别诊断能力。在大量实验中，KnowSR……

    arXiv:2611.27427v1 Announce Type: cross  Abstract: Accurate differential diagnosis of neurodegenerative disorders remains challenging due to overlapping clinical manifestations and heterogeneous, often incomplete multimodal observations. Existing multimodal models can integrate imaging and clinical measurements but often overlook disease-specific evidence patterns and pairwise relations among competing diagnoses. We therefore propose Knowledge-enhanced Structural Reasoning (KnowSR), which aligns multimodal patient representations with a literature-grounded knowledge bank of disease prototypes, signed evidence directions, and pairwise diagnostic relations. Specifically, in KnowSR, patient-specific knowledge alignment and target-aware pooling inject structured disease evidence, while anchored relational reasoning reconciles pairwise diagnostic margins over a sparse differential graph, thereby collectively enhancing differential diagnostic capability. Across extensive experiments, KnowSR 
    
[^74]: Cleave：切断开放权重大语言模型中的安全护栏并审计谱耦合

    Cleave: Severing Guardrails and Auditing Spectral Coupling in Open-Weight LLMs

    [https://arxiv.org/abs/2611.27426](https://arxiv.org/abs/2611.27426)

    本文提出了无需训练的攻击方法Cleave，通过直接编辑残差写入权重移除开放权重LLM的安全护栏，可选择性解除特定安全限制并规避行为抽查与静态权重审计。

    

    开放权重的大语言模型因其强大的编码和推理能力而被广泛下载，并在超出提供商控制的范围内被本地部署。为防止滥用，开发者在发布之前将这些模型中嵌入了安全护栏。然而，我们发现这些护栏依赖于残差写入权重来触发拒绝响应。基于这一洞察，我们提出了Cleave，这是一种无需训练的攻击方法，可直接编辑这些权重以移除安全限制，同时保留通用能力。借助Cleave，恶意用户既可以移除所有护栏，也可以选择性地仅解除网络安全方面的限制。由于这种选择性移除在其他有害请求上仍保持高达99%的拒绝率，并避免了权重秩坍缩，因此能够同时规避行为抽查和静态权重审计。在从7B到320B参数的五个开放权重LLM上，Cleave在七个安全基准测试中实现了高达67.97%的攻击成功率。

    arXiv:2611.27426v1 Announce Type: cross  Abstract: Open-weight Large Language Models (LLMs) are widely downloaded for their strong coding and reasoning capabilities and deployed locally beyond the provider's control. To prevent misuse, developers embed safety guardrails into these models before release. However, we find that these guardrails rely on the residual writing weights to trigger refusals. Based on this insight, we propose Cleave, a training-free attack that directly edits these weights to remove safety restrictions while preserving general capabilities. With Cleave, malicious users can either remove all guardrails or selectively lift only cybersecurity restrictions. Because this selective removal retains up to a 99% refusal rate on other harmful requests and avoids weight rank collapse, it evades both behavioral spot-checks and static weight audits. Across five open-weight LLMs from 7B to 320B parameters, Cleave achieves up to a 67.97% attack success rate on seven safety benc
    
[^75]: TrojanMaxxing：基于权重空间编程的一次性VLM后门攻击

    TrojanMaxxing: One-shotting VLM Backdoors with Weight-Space Programming

    [https://arxiv.org/abs/2611.27425](https://arxiv.org/abs/2611.27425)

    该论文提出TrojanMaxxing，将VLM后门攻击重新定义为权重空间编程任务，利用LLM智能体直接检查并编辑模型权重并协同设计不可察觉的视觉触发器，无需任何训练或优化即可一次性为视觉语言模型植入后门。

    

    最近的研究表明，大型语言模型（LLM）可以直接对其他神经网络的权重进行推理。仅凭一个自然语言目标，LLM就能检查模型的权重、理解其计算内容，并在无需基于梯度的优化的情况下直接编辑它们。这种能力被称为权重空间编程（WSP），已被用于提升模型在数学推理、医学问答等基准上的表现。我们提出TrojanMaxxing，一种通过将后门攻击重新定义为WSP任务来攻击视觉语言模型（VLM）的新型策略。我们使用Claude Opus 5.5和Mythos 5.2智能体，在不使用任何优化或训练方法的情况下对VLM植入后门。这些智能体检查模型权重、进行针对性编辑，并共同设计一个不可察觉的视觉触发器来激活后门。随后我们研究了智能体的策略，并在各种基准数据集上对植入后门的模型进行了全面评估。TrojanMaxxing实现了攻击（摘要原文在此处截断）。

    arXiv:2611.27425v1 Announce Type: cross  Abstract: Recent work shows that large language models (LLMs) can reason directly about the weights of other neural networks. Given only a natural-language objective, an LLM can inspect a model's weights, understand what they compute, and edit them directly without gradient-based optimization. This capability, termed weight-space programming (WSP), has been used to improve model performance on math reasoning, medical QA and other benchmarks. We propose TrojanMaxxing, a novel strategy to backdoor vision-language models (VLMs), by reframing it as a WSP task. We use Claude Opus 5.5 and Mythos 5.2 agents to backdoor VLMs without any optimization or training methods. The agents inspect the model weights, apply targeted edits, and co-design an imperceptible visual trigger that activates the backdoor. We then study the agents' strategies and present a thorough evaluation of the trojaned models on various benchmark datasets. TrojanMaxxing achieves attac
    
[^76]: 参差不齐的鲁棒性：LLM量化下的安全能力迁移不均衡与新出现的漏洞

    Jagged Robustness: Uneven Safety Transfer and Emerging Vulnerabilities under LLM Quantization

    [https://arxiv.org/abs/2611.27424](https://arxiv.org/abs/2611.27424)

    量化会导致大语言模型的安全防护和偏见控制能力不均衡地退化，而其他能力保持稳定，且对完整模型无害的对抗性后缀攻击竟能越狱量化后的模型。

    

    量化技术通过模型参数压缩使语言模型能够高效推理。本工作审计了量化对能力和安全领域的影响，包括模型安全防护、指令遵循、伦理和偏见。通过基准测试，我们发现模型安全防护和偏见在量化下可能发生退化，而在Qwen3-4B模型中，其他能力（如真实性）则保持相对稳定。我们将LLM能力在量化下这种不均衡的迁移现象定义为“参差不齐的鲁棒性”。我们发现了驱动这一现象的多个因素：任务复杂度、评估方法和生成连贯性。此外，我们通过观察到量化后模型对不安全请求的服从度增加，验证了模型安全防护的退化，并演示了通用对抗性后缀攻击——这些攻击对完整模型是良性的，但却能越狱量化后的模型。研究结果表明，假设性能和鲁棒性

    arXiv:2611.27424v1 Announce Type: new  Abstract: Quantization enables efficient inference in language models through model parameter compression. This work audits the impact of quantization on capability and safety domains including model safeguards, instruction-following, ethics and bias. Through benchmarks, we find that model safeguards and bias can degrade under quantization, while other capabilities such as truthfulness remain more stable in Qwen3-4B. We define this uneven transfer of LLM capabilities under quantization as Jagged Robustness. We find multiple factors driving Jagged Robustness: task complexity, evaluation method and generation coherence. Moreover, we verify the degradation of model safeguards by observing increased compliance with unsafe requests under quantization, and demonstrate universal adversarial suffix attacks that are benign for the full model but jailbreak quantized models. The findings indicate that it is risky to assume that the performance and robustness
    
[^77]: ADEPT：通过模型在环光度调节与采集风险预测实现自适应单目深度估计

    ADEPT: Adaptive Monocular Depth Estimation via Model-In-The-Loop Photometric Tuning with Capture Risk Prediction

    [https://arxiv.org/abs/2611.27423](https://arxiv.org/abs/2611.27423)

    提出ADEPT框架，以“模型在环”方式训练一个采集风险预测器，并借助上下文高斯过程控制器自适应调节相机曝光与增益，从而在机器人运动场景下提升单目深度估计的精度。

    

    曝光和增益决定了单目深度估计（MDE）可获取的视觉证据质量，尤其当机器人运动导致光线收集与运动模糊之间存在权衡时更是如此。我们提出了ADEPT，一个利用从冻结深度模型中学习到的反馈来自适应调整这些设置的框架。在训练阶段，真值深度用于在一组经过物理评估的相机设置中识别误差最低的参考设置。一个紧凑的预测器通过回归和排序损失，学习每个候选设置的深度预测与参考预测之间的参考相对分歧（RRD）。在部署阶段，该预测器利用冻结模型特征以及相机、运动和光照上下文信息对实际采集的图像进行评分，无需真值或参考图像。一个上下文高斯过程控制器利用近期评分，在运动和光度约束下选择后续的曝光和增益设置。我们通过离线相机选择方式对……进行评估（摘要在此处被截断）。

    arXiv:2611.27423v1 Announce Type: cross  Abstract: Exposure and gain determine the visual evidence available to monocular depth estimation (MDE), particularly when robot motion creates a tradeoff between light collection and motion blur. We present ADEPT, a framework that adapts these settings using learned feedback from a frozen depth model. During training, ground-truth depth identifies the lowest-error reference within a group of physically evaluated camera settings. A compact predictor learns reference-relative disagreement (RRD) between each candidate's depth prediction and the reference prediction using regression and ranking losses. At deployment, it scores observed captures using frozen-model features and camera, motion, and illumination context, without ground truth or reference images. A contextual Gaussian-process controller uses recent scores to select subsequent exposure and gain settings under motion and photometric constraints. We evaluate offline camera selection with r
    
[^78]: 基于相对熵与运行时验证检测RAG知识库中的词替换投毒攻击

    Detecting Word-Substitution Poisoning of a RAG Knowledge Base with Relative Entropy and Runtime Verification

    [https://arxiv.org/abs/2611.27410](https://arxiv.org/abs/2611.27410)

    该论文发现针对RAG知识库的同义词替换投毒主要通过证据丢失和拒答降低答案质量而非制造虚假陈述，并提出利用相对熵中的词-同义词对数比项结合小型语言模型作为参考来高效检测被投毒文段，AUROC达到0.95。

    

    检索增强生成（RAG）系统利用从知识库中检索到的文段来回答问题，因此用户必须信任该知识库。我们研究了一种攻击者，其通过将单词替换为同义词（PWWS、TextFooler）来悄悄篡改现有文段，攻击对象是一个基于MITRE ATT&CK和CAPEC构建的网络安全RAG系统。借助一个经过审核的44题基准测试、多次重复的成对实验以及一个仅改变排版格式的对照语料库，我们发现投毒会使评判的答案正确率从0.714降至0.555。损害主要来自证据丢失和拒答，而非虚假陈述。为了找出被篡改的文段，我们使用相对熵：将整个文段与正常文本进行比较只能提供微弱信号，但相对熵中的一项——即一个词与其最可能同义词之间的对数比——效果很好：以小型语言模型作为参考时，该方法达到0.95的AUROC，而在对照语料库上则接近随机水平。

    arXiv:2611.27410v1 Announce Type: cross  Abstract: A retrieval-augmented generation (RAG) system answers questions using passages it retrieves from a knowledge base, so users must trust that knowledge base. We study an attacker who quietly edits existing passages by swapping words for synonyms (PWWS, TextFooler) in a cybersecurity RAG system built from MITRE ATT&CK and CAPEC. Using a reviewed 44-question benchmark, repeated paired runs and a control corpus in which only the layout changes, we find that poisoning lowers judged answer correctness from 0.714 to 0.555. The damage comes mostly from lost evidence and refusals, not from false statements. To find the edited passages, we use relative entropy. Comparing a whole passage with normal text is a weak signal. Still, one term of the relative entropy, the log-ratio between a word and its most probable synonym, works well: with a small language model as the reference, it reaches an AUROC of 0.95 and stays near chance on the control. Used
    
[^79]: 超越线性可解码性：理解与改进世界动作模型中的指令遵循能力

    Beyond Linear Decodability: Understanding and Improving Instruction Following in World Action Models

    [https://arxiv.org/abs/2611.27401](https://arxiv.org/abs/2611.27401)

    本研究发现世界动作模型的指令遵循能力与目标信息的线性可解码性密切相关，并提出线性可解码性监督（LDS）这一简单有效的辅助训练目标，使模型在LIBERO-CF上的成功率和定位率分别大幅超越最强基线26.8和42.9个百分点。

    

    世界动作模型（WAM）展现出强大的操作能力，但在测试时往往无法遵循语言指令。先前的研究表明，不遵循指令（NF）的世界动作模型虽然编码了有关指令目标的信息，但这些信息并不能控制其行为。通过将不遵循指令的WAM与其遵循指令（IF）的对应模型进行比较，我们发现尽管两者都编码了目标信息，但在遵循指令的WAM中，目标信息的线性可解码性显著更高。受这一观察启发，我们提出了线性可解码性监督（LDS），这是一种简单而有效的辅助训练目标，旨在促进指令目标信息的线性可解码性。在LIBERO-CF基准上，采用LDS的InternW0-Δ实现了74.6%的成功率和89.2%的定位率，分别超越最强基线26.8和42.9个百分点。LDS的有效性挑战了当前流行的解释，即编码的目标信息……

    arXiv:2611.27401v1 Announce Type: new  Abstract: World action models (WAMs) exhibit strong manipulation capabilities yet often fail to follow test-time language instructions. Prior work shows that non-following (NF) WAMs encode information about the instructed target, but this information does not control their behavior. By comparing NF WAM with its instruction-following (IF) counterpart, we find that target information is substantially more linearly decodable in IF WAMs, despite being encoded in both. Motivated by this observation, we introduce linear decodability supervision (LDS), a simple yet effective auxiliary training objective that promotes the linear decodability of the instructed target information. On LIBERO-CF, InternW0-$\Delta$ with LDS achieves a success rate of 74.6% and a grounding rate of 89.2%, surpassing the strongest baseline by 26.8 and 42.9 percentage points, respectively. The effectiveness of LDS challenges the prevailing interpretation that encoded target inform
    
[^80]: 拆解与组装：在物理变换中跟踪物体

    Taking Apart and Putting Together: Tracking Objects Through Physical Transformations

    [https://arxiv.org/abs/2611.27398](https://arxiv.org/abs/2611.27398)

    提出免训练框架 TIPO，通过“生成-评估-判断”流水线使视频跟踪系统能在物体拆解与组装等物理变换中持续跟踪实例、推断变换并维护实体状态图。

    

    我们介绍了变换跟踪与理解（TTU）任务，将视频目标跟踪扩展到物体经历物理组装与拆解的环境中。TTU 要求在变换过程中持续跟踪物体、识别新出现的实体、维护它们之间关系的状态图，并为变换赋予语义标签。虽然基础模型（如 SAM 2/3）能够在外观变化中维持轨迹片段，但当物体发生物理演变时，它们无法更新语义身份。先前的工作仅处理拆解场景，且依赖仅在跟踪失败时才触发的昂贵搜索。我们提出了基于提议与重叠的实例跟踪（TIPO），这是一个免训练框架，通过“生成-评估-判断”流水线持续推断变换。TIPO 从不相交的掩码组件中生成拆分与合并提议，利用内部与外部重叠度对照掩码解码器先验对其进行评分，并……

    arXiv:2611.27398v1 Announce Type: cross  Abstract: We introduce Transformation Tracking and Understanding (TTU), extending video object tracking to environments where objects undergo physical assemblies and disassemblies. TTU requires continuously tracking objects through transformations, identifying new entities, maintaining a state graph of their relationships, and assigning semantic labels to transformations. While foundation models (e.g., SAM 2/3) maintain tracklets through appearance changes, they cannot update semantic identities as objects physically evolve. Prior work addresses only disassemblies, relying on expensive searches triggered only when tracking fails. We propose Tracking Instances via Proposals and Overlaps (TIPO), a training-free framework that continuously infers transformations via a generate-evaluate-judge pipeline. TIPO generates split and merge proposals from disjoint mask components, scores them against mask-decoder priors using inner and external overlap, and
    
[^81]: 证据充分性评估使本地部署的临床AI智能体实现高性价比的审查分配

    Evidence sufficiency assessment enables cost-effective review allocation in on-premise clinical AI agents

    [https://arxiv.org/abs/2611.27390](https://arxiv.org/abs/2611.27390)

    该论文提出无需训练的证据充分性评估方法与三级顺序分诊协议，使本地临床AI智能体仅用单次推理即可达到五次一致性验证的通过率（98.91%），并节省约80%的计算成本。

    

    本地部署的临床AI智能体需要自动接受高置信度的诊断，并将其余病例转交临床医生审查，然而token概率和自我报告的置信度会遗漏错误，而更可靠的五次运行一致性验证则需要五倍的计算量。我们提出了无需训练的决策点证据充分性评估方法及三级顺序分诊协议。通过屏蔽推理过程和诊断结果，同一个4B模型基于客观证据，利用先验去偏概率对五个有序等级进行充分性评分。校准阶段冻结一个名义上的前50%阈值，根据接受病例通过率的单侧95% Wilson下置信界是否达到95%临床目标线，选择单次运行、五次运行一致性或完全转交人工审查的层级。在涵盖MIRA-v2、CDM、VivaBench及院内肝脏恶性肿瘤主要队列的1,605例病例中，MIRA-v2单次运行的前50%通过率达到98.91%，与五次运行一致性验证的效果相当，同时节省约80%的计算量。

    arXiv:2611.27390v1 Announce Type: new  Abstract: On-premise clinical AI agents must auto-accept high-confidence diagnoses, deferring others to clinician review, yet token probabilities and self-reported confidence miss errors, while more reliable five-run consistency incurs fivefold compute. We propose training-free decision-point evidence sufficiency assessment and a three-tier sequential triage protocol. Masking reasoning and diagnosis, the same 4B model scores sufficiency from objective evidence via prior-debiased probabilities over five ordered levels. Calibration freezes a nominal top-50% threshold, selecting single-run, five-run consistency, or full-deferral tiers by whether the one-sided 95% Wilson lower confidence bound of pass rate among accepted cases meets the 95% clinical target line. Across 1,605 MIRA-v2, CDM, VivaBench, and in-hospital liver malignancy primary cohort cases, MIRA-v2 single-run top-50% pass rate reached 98.91%, matching five-run consistency, saving ~80% com
    
[^82]: PanoUAV-Track：野外环境下的全球形无人机视觉跟踪

    PanoUAV-Track: Full-Spherical UAV Visual Tracking in the Wild

    [https://arxiv.org/abs/2611.27383](https://arxiv.org/abs/2611.27383)

    该论文构建了首个面向移动无人机平台的全球形视觉跟踪基准PanoUAV-Track，基于40小时真实8K全景视频，包含1,895个序列、507,961帧和2,872个标注目标，填补了无人机全向跟踪（含ERP投影几何挑战及日夜场景）的研究空白。

    

    现有的无人机（UAV）跟踪基准主要依赖视场受限的透视相机。全向相机能够实现全球面观测，但从移动的无人机平台上进行跟踪在很大程度上仍未被探索。与地面全景跟踪不同，安装在无人机上的全向相机需要在方位角和仰角两个维度上观测目标，这在等距圆柱投影（ERP）下引入了几何上的挑战。为了研究这一场景，我们采集了约40小时的真实世界8K全景无人机视频，并通过时间片段提取、目标标注以及模拟序列的集成构建了PanoUAV-Track基准。该基准包含1,895个序列、507,961帧图像和2,872个标注目标，涵盖真实世界的日间与夜间场景以及仿真场景，同时支持基于视觉和基于语言的目标指定方式。通过大量评估实验，我们揭示了……

    arXiv:2611.27383v1 Announce Type: cross  Abstract: Existing UAV tracking benchmarks predominantly rely on perspective cameras with limited fields of view. Omnidirectional cameras enable full-spherical observation, yet tracking from moving UAV platforms remains largely unexplored. Unlike ground-level panoramic tracking, UAV-mounted omnidirectional cameras observe targets across both azimuth and elevation, introducing geometric challenges under equirectangular projection (ERP). To investigate this setting, we capture approximately \textbf{40 hours of real-world 8K panoramic UAV video} and construct \textbf{PanoUAV-Track} through temporal clip extraction, target annotation, and the integration of simulated sequences. The resulting benchmark comprises 1,895 sequences, 507,961 frames, and 2,872 annotated objects, covering real-world daytime and nighttime scenes alongside simulation, and supporting both visual and language-based target specification. Through extensive evaluations, we reveal 
    
[^83]: VMSA：用于长视频理解的视频记忆技能智能体

    VMSA: Video Memory Skill Agent for Long Video Understanding

    [https://arxiv.org/abs/2611.27381](https://arxiv.org/abs/2611.27381)

    提出VMSA框架，通过联合设计三种互补的结构化视频记忆（片段字幕记忆、实体事件记忆、实体感知情节记忆）与记忆技能，提升智能体在长视频问答中跨长时间跨度检索与整合证据的能力。

    

    长视频问答需要从跨越数小时甚至数天的视频中检索并整合相关证据。近期的智能体方法将视频信息组织到外部记忆中，并迭代地检索证据来回答问题。然而，有效的检索与推理仍然具有挑战性，因为智能体必须确定要搜索什么信息、访问哪些记忆源，以及如何连接跨越遥远时间间隔的证据。为了应对这些挑战，我们提出了视频记忆技能智能体（VMSA），这是一个为长视频理解联合设计结构化视频记忆与记忆技能的框架。VMSA 构建了三个互补的记忆组件：片段字幕记忆和实体事件记忆作为可搜索的证据来源，而实体感知情节记忆则提供广泛而详细的视频上下文，以指导证据检索与推理。为了进一步支持有效的记忆利用……

    arXiv:2611.27381v1 Announce Type: cross  Abstract: Long-video question answering requires retrieving and integrating relevant evidence from videos spanning hours or even days. Recent agentic approaches organize video information into external memories and iteratively retrieve evidence to answer questions. However, effective retrieval and reasoning remain challenging, as agents must determine what information to search for, which memory sources to access, and how to connect evidence across distant temporal intervals. To address these challenges, we propose Video Memory Skill Agent (VMSA), a framework that jointly designs structured video memories and memory skills for long-video understanding. VMSA constructs three complementary memory components. Segment Caption Memory and Entity Event Memory serve as searchable evidence sources, while Entity-Aware Episode Memory provides broad yet detailed video context to guide evidence retrieval and reasoning. To further support effective memory uti
    
[^84]: ReActWAM：面向高效世界-动作模型的动作一致性缓存

    ReActWAM: Action-Consistent Caching for Efficient World-Action Models

    [https://arxiv.org/abs/2611.27372](https://arxiv.org/abs/2611.27372)

    ReActWAM提出了一种无需训练的框架，通过动作意图引导的前缀跳过与块内预测重用来消除世界-动作模型闭环推理中的冗余计算，显著提升执行效率。

    

    世界-动作模型（WAMs）通常需要与机器人动作一起生成未来视觉状态，这在推理过程中引入了大量计算开销。在闭环执行期间，连续的推理周期往往预测相似的动作序列，导致冗余计算。然而，要减少这种冗余，需要区分一致的动作与需要重新计算的变化。为解决这些挑战，我们提出了ReActWAM，这是一个无需训练的框架，它将动作意图引导的前缀跳过与块内预测重用相结合。该框架采用由粗到细的评估方式来衡量运动变化、视觉一致性和动作预测冲突。对于符合条件的块，我们利用第一步的差异将缓存的潜在特征和网络特征调整到当前条件下，在较后的去噪步骤提供估计值，从而绕过冗余的前缀。后续推理在重用稳定特征的同时对该估计值进行细化……

    arXiv:2611.27372v1 Announce Type: new  Abstract: World-action models (WAMs) often generate future visual states alongside robot actions, introducing substantial computational overhead during inference. During closed-loop execution, consecutive inference cycles often predict similar action sequences, leading to redundant computation. However, reducing this redundancy requires distinguishing consistent actions from changes that demand fresh computation. To address these challenges, we propose ReActWAM, a training-free framework that combines action-intent-guided prefix skipping with inner-chunk prediction reuse. A coarse-to-fine assessment evaluates motion changes, visual consistency, and action prediction conflicts. For eligible chunks, we draw on first-step differences to adapt the cached latent and network features to the current condition, providing an estimate at a later denoising step that bypasses the redundant prefix. Subsequent inference refines this estimate while reusing stabl
    
[^85]: AMARIS：通过自适应子空间继承融合通用与专家大语言模型

    AMARIS: Merging Generalist and Specialist LLMs via Adaptive Subspace Inheritance

    [https://arxiv.org/abs/2611.27369](https://arxiv.org/abs/2611.27369)

    AMARIS 提出了一种协方差驱动的免训练模型合并框架，通过在正则化扰动预算下选择高效用、低干扰的激活子空间，将专家模型能力以单一静态检查点的形式无训练地继承到通用大语言模型中。

    

    大语言模型（LLM）的领域专业化通常能提升目标领域的性能，但会损害其广泛能力。模型合并是一种无需训练的有吸引力的替代方案，然而常见的参数空间策略（例如插值和稀疏化平均）对任务特定干扰的控制能力有限。我们提出了通过排序继承子空间实现自适应模型对齐（AMARIS），这是一个由协方差驱动的框架，仅沿选定的激活方向传递专家模型的更新。AMARIS 将合并表述为预算化的子空间继承：在正则化的基础输出扰动预算下，寻求对专家激活的最大捕获，然后从排序后的广义特征方向构建低秩投影门控。由此得到一个单一的静态合并检查点，将知识迁移集中在高效用、低干扰的方向上。我们在基于 Qwen3 的合并模型（8B 和 32B）上评估了 AMARIS，涵盖金融、保险等领域（原文摘要此处被截断）。

    arXiv:2611.27369v1 Announce Type: new  Abstract: Domain specialization of large language models (LLMs) often improves target-domain performance but degrades broad capabilities. Model merging is an attractive training-free alternative, yet common parameter-space strategies (for example, interpolation and sparsified averaging) provide limited control over task-specific interference. We introduce Adaptive Model Alignment via Ranked Inherited Subspaces (AMARIS), a covariance-driven framework that transfers specialist updates only along selected activation directions. AMARIS formulates merging as budgeted subspace inheritance: it seeks high specialist activation capture under a regularized base-output perturbation budget, then builds a low-rank projection gate from ranked generalized eigendirections. This yields a single static merged checkpoint that concentrates transfer in high-utility, low-interference directions. We evaluate AMARIS on Qwen3-based merges (8B and 32B) across finance, inst
    
[^86]: 合作式多智能体强化学习中的协调失败测试

    Testing for Coordination Failures in Cooperative Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2611.27368](https://arxiv.org/abs/2611.27368)

    提出了DisQ方法，利用Q值差距这一可解释信号从基于价值的合作式多智能体强化学习中筛选出能揭示协调失败的关键状态，以高效检测智能体间的不安全行为。

    

    深度并行合作式多智能体强化学习（CMARL）训练多个在每个时间步都采取行动的智能体，以优化一个共同目标。与单智能体强化学习一样，CMARL可能产生违反奖励未捕获的隐式安全要求的不安全行为，且这些行为难以解释。然而，与单智能体强化学习策略的安全测试相比，CMARL策略的安全测试受到的关注较少。在本文中，我们提出了DisQ，这是一种测试用例选择方法，通过利用训练后的Q-learning智能体特有的信号，在基于价值的CMARL中识别能揭示失败的状态。该信号是Q值差距（Q-value gap），这是一个可解释的指标，当各智能体对可实现回报的估计之间的偏差超过校准阈值时，该指标会标记相应状态。DisQ选择这些状态进行执行，并跳过没有差距的状态。我们在四个CMARL基准环境中的实验表明，在大多数情况下...

    arXiv:2611.27368v1 Announce Type: cross  Abstract: Deep parallel cooperative multi-agent reinforcement learning (CMARL)} trains multiple agents that all act at every time step to optimize a common objective. Like single-agent RL, CMARL can produce unsafe behaviors that violate implicit safety requirements not captured by the reward, and these behaviors are hard to explain. However, testing CMARL policies for safety has received less attention than the safety testing of single-agent RL policies. In this paper, we propose DisQ, a test case selection method that identifies failure-revealing states in value-based CMARL by exploiting a signal specific to trained Q-learning agents. The signal is the Q-value gap, an explainable metric that flags a state when the agents' estimates of the achievable return diverge by more than a calibrated threshold. DisQ selects such states for execution and skips those without a gap. Our experiments in four CMARL benchmark environments show that, in most of t
    
[^87]: WePM：面向统一多模态大模型的分层冲突调和

    WePM: Hierarchical Conflict Reconciliation for Unified MLLMs

    [https://arxiv.org/abs/2611.27365](https://arxiv.org/abs/2611.27365)

    WePM通过形式化异构能力密度（HCD）、多粒度视觉预算分配和分层冲突调和，在单一视觉-语言骨干中统一了表示、生成与决策能力，打造出适用于端侧部署的口袋模型系列。

    

    将多模态智能从云端迁移至用户设备，可将敏感数据保留在本地并降低推理成本，但在内存与计算预算紧张的情况下，为每种能力部署专门的模型会重复占用视觉-语言容量。我们形式化了异构能力密度（HCD），用于衡量模型在每单位常驻推理内存下服务异构任务族的广度与有效性。要在表示、生成和决策三类能力上实现高HCD，面临两大挑战：其一，在固定token预算下，这些能力需要不同空间粒度的视觉证据；其二，分别收集的任务语料会对共享参数产生冲突的更新，而跨任务配对监督对于开放式输入而言成本高昂且语义模糊。我们提出WePM，这是一个在单一视觉-语言骨干中统一表示、生成和决策的口袋模型系列。多粒度视觉预算分配（MGVB）……

    arXiv:2611.27365v1 Announce Type: new  Abstract: Bringing multimodal intelligence from the cloud to users' devices keeps sensitive data local and reduces inference cost, but deploying a specialized model per capability duplicates vision-language capacity under tight memory and compute budgets. We formalize heterogeneous capability density (HCD), measuring how broadly and effectively a model serves heterogeneous task families per unit of resident inference memory. High HCD across representation, generation, and decision raises two challenges: these capabilities need visual evidence at different spatial granularities under a fixed token budget, and separately collected task corpora yield conflicting updates to shared parameters, while cross-task paired supervision is costly and ambiguous for open-ended inputs. We present WePM, a family of pocket models unifying representation, generation, and decision within one vision-language backbone. Multi-Granularity Visual Budgeting (MGVB) allocate
    
[^88]: WaXup：专家模型权重平均作为数据增强以实现鲁棒的内省

    WaXup: Weight Averaging of Experts as Data Augmentation for Robust Introspection

    [https://arxiv.org/abs/2611.27353](https://arxiv.org/abs/2611.27353)

    本文提出WaXup方法，通过在权重空间中对专家模型进行线性插值（权重平均）来增强内省训练数据，从而显著提升大模型可训练内省方法的鲁棒性与泛化能力。

    

    随着大型语言模型能力的不断提升，传统的机制可解释性方法难以跟上其发展步伐。相比之下，可训练的内省方法作为一种有前景的替代方案应运而生；它将模型本身重新用作自我诊断工具，通过训练使模型能够用自然语言表达其在微调过程中学到的内容。虽然在微调专家模型的数据集上进行训练取得了令人鼓舞的结果，但目前这种方法缺乏鲁棒性和泛化能力。基于MiXup的数据增强框架，我们提出了WaXup，这是一种通过权重空间中的线性插值来增强内省训练集的方法。这一方法的理论依据是线性模式连通性特性：权重平均后的专家模型能够插值初始专家模型的行为。我们在三个基准上评估了WaXup：隐藏行为模型、真实世界NLP任务的元数据集，以及AuditBench——一个具有挑战性的对齐审计套件。

    arXiv:2611.27353v1 Announce Type: cross  Abstract: As the capabilities of large language models keep improving, traditional mechanistic interpretability methods struggle to keep pace. In contrast, trainable introspection methods emerge as a promising alternative; they repurpose the model itself as a self-diagnostic tool, trained to verbalize what it has learned during finetuning in natural language. While training on datasets of finetuned experts achieves encouraging results, it currently lacks robustness and generalization. Building on the data augmentation framework of MiXup, we introduce WaXup, a method that augments the introspection training set through linear interpolation in the weight space. This is motivated by the linear mode connectivity property: weight averaged experts interpolate the behaviors of the initial experts. We evaluate WaXup on three benchmarks: hidden-behavior models, a meta-dataset of real-world NLP tasks and AuditBench, a challenging alignment auditing suite 
    
[^89]: PAVE：一种基于视觉语言模型、具有几何一致性的路网建模框架

    PAVE: A Harness for Geometrically Consistent Road Network Modeling with Vision-Language Models

    [https://arxiv.org/abs/2611.27352](https://arxiv.org/abs/2611.27352)

    提出PAVE框架，通过感知、锚定、验证、编辑四个环节，利用视觉语言模型从卫星影像和街景影像中构建具有几何一致性的道路网络模型，克服了开放地图数据不完整、不准确以及VLM直接使用时几何一致性缺失的问题。

    

    交通运输模型——无论是用于交通模拟还是规划支持——其可靠性取决于所运行的道路网络，精确到每个交叉口的车道与转向。从业者目前仍需从GIS文件中获取路网，并手工录入其交叉口和流向数据，这项工作随着每个新的基准年份和设计方案而反复进行；或者他们从开放地图数据中生成这些信息。在我们测试的44个进口道中，OpenStreetMap的车道标签仅能为其中13个提供正确的布局；开放地图数据往往不完整或不准确。相比之下，卫星影像和街景影像覆盖了全部44个进口道，并以路面标线的形式呈现车道信息。视觉语言模型（VLM）能够读取这类影像，但如果直接使用，当同一条车道需要在卫星影像裁剪图、不同视角拍摄的街景照片以及带有ID标注的路网之间进行匹配时，模型会丧失几何一致性。因此，我们提出了PAVE（Perceive感知、Anchor锚定、Verify验证、Edit编辑），一个在其中……

    arXiv:2611.27352v1 Announce Type: cross  Abstract: Transportation models, whether they simulate traffic or support planning, are only as reliable as the road networks they run on, down to the lanes and turns at each intersection. Practitioners still take such a network from GIS files and supply its junction and movement data by hand, a task that recurs with each new base year and design alternative, or they generate it from open map data. OpenStreetMap's lane tags give the correct layout for only 13 of the 44 approaches we test; open map data are often incomplete or inaccurate. Satellite and street-level imagery, in contrast, cover all 44 and show their lanes as painted. Vision-language models (VLMs) can read such imagery, but asked directly they lose geometric consistency when the same lane must be matched across a satellite crop, street-level photos taken from different poses and an ID-labelled network. We therefore present PAVE (Perceive, Anchor, Verify, Edit), a harness in which co
    
[^90]: GRACE：用于诊断与改进GraphRAG中知识整合的受控证据干预方法

    GRACE: Controlled Evidence Interventions for Diagnosing and Improving Knowledge Integration in GraphRAG

    [https://arxiv.org/abs/2611.27350](https://arxiv.org/abs/2611.27350)

    该论文提出GRACE基准，通过对证据子图施加切断支持路径、添加干扰项、附加高度数枢纽和反事实答案替换四种受控干预，来诊断GraphRAG知识整合失败的根源，并提出QCFSE-SA方法在不同证据状态下改进模型表现。

    

    GraphRAG将语言模型的参数化知识与检索到的图谱证据相结合，然而总体准确率无法揭示失败究竟源于路径缺失、干扰项，还是与模型自身知识的冲突。我们提出GRACE基准，通过在Wikidata上执行LC-QuAD 2.0查询，并对每个证据子图施加四种受控干预来构建：切断支持路径、添加类型兼容的干扰项、附加高度数枢纽节点，以及用反事实实体替换被支持的答案。问题和选项位置在五种状态下保持固定。经过验证的基准包含4,028个问题和20,140个问题-图谱对。这些对比实验表明：仅基于问题的路由无法区分证据状态，而相关性引导虽能改善可回答的状态，却在支持路径被切断时造成损害——因为相关事实联合起来可能仍不足以回答问题。我们提出QCFSE-SA方法，它结合了类型化（原文摘要在此处截断）

    arXiv:2611.27350v1 Announce Type: new  Abstract: GraphRAG combines a language model's parametric knowledge with retrieved graph evidence, yet aggregate accuracy cannot show whether failures stem from missing paths, distractors, or conflicts with model knowledge. We introduce GRACE, a benchmark built by executing LC-QuAD 2.0 queries over Wikidata and applying four controlled interventions to each evidence subgraph: severing a support path, adding type-compatible distractors, attaching high-degree hubs, and replacing the supported answer with a counterfactual entity. Questions and option positions remain fixed across the five states. The validated benchmark contains 4,028 questions and 20,140 question-graph pairs. These comparisons reveal that question-only routing cannot distinguish evidence states, while relevance guidance can improve answerable states yet hurt when the support path is severed: relevant facts may not jointly suffice to answer. We propose QCFSE-SA, which combines typed 
    
[^91]: EcomWriteBench：面向电商写作任务的真实世界基准

    EcomWriteBench: A Real-World Benchmark for E-commerce Writing Tasks

    [https://arxiv.org/abs/2611.27349](https://arxiv.org/abs/2611.27349)

    该论文提出了 EcomWriteBench——一个涵盖 13 个类别、五种语言、69 个电商写作任务、共 3,861 个查询的真实世界基准，通过构建 37 项共享能力池并进行能力级评分，实现对大语言模型在细粒度电商写作能力上优劣势的系统评估。

    

    大型语言模型越来越多地被应用于电商工作流程中，用于生成影响购买决策和支持客户互动的内容。评估这些系统需要考虑用户需求、产品证据、结构约束和业务要求，而这些因素无法被通用写作基准完全涵盖。我们提出了 EcomWriteBench，这是一个真实世界基准，包含来自 13 个类别、五种语言的 69 个电商写作任务的 3,861 个查询。通过任务级提取、跨任务整合以及人工反馈指导的细化，EcomWriteBench 构建了一个包含 37 项能力的共享能力池，并识别出适用于每个任务的能力子集。基于能力的评分方式使我们能够在 EcomWriteBench 上比较最先进的大语言模型，并刻画它们在细粒度电商写作能力上的相对优势与劣势。我们进一步提出了一种偏好校准评估方法（摘要在此处截断）。

    arXiv:2611.27349v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used throughout e-commerce workflows to generate content that informs purchasing decisions and supports customer interactions. Evaluating these systems requires accounting for user needs, product evidence, structural constraints, and business requirements that are not fully captured by general-purpose writing benchmarks. We introduce EcomWriteBench, a real-world benchmark containing 3,861 queries from 69 e-commerce writing tasks across 13 categories and five languages. Through task-level extraction, cross-task consolidation, and refinement guided by human feedback, EcomWriteBench constructs a shared pool of 37 abilities and identifies the subset applicable to each task. Ability-wise scoring enables us to compare state-of-the-art LLMs on EcomWriteBench and characterize their relative strengths and weaknesses across fine-grained e-commerce writing abilities. We further propose a preference-cali
    
[^92]: 基于运动区域引导学习交互动态的机器人视频世界模型

    Learning Interaction Dynamics via Motion-Region Guidance for Robotic Video World Models

    [https://arxiv.org/abs/2611.27344](https://arxiv.org/abs/2611.27344)

    MoReWorld通过共享运动区域引导机制，将机器人视频世界模型的动态学习与视频生成聚焦于交互相关区域，解决了幻觉接触和不一致机器人-物体交互等物理不合理问题。

    

    动作条件视频世界模型能够在候选机器人动作下想象未来的观测结果，为机器人策略评估和交互模拟提供了一种有前景的视觉工具。然而，现有的视频世界模型通常优化全局重建误差，而非强调与交互相关的区域。因此，它们可能保留整体场景外观，却忽略局部交互细节，导致物理上不合理的结果，例如幻觉式接触效果和不一致的机器人-物体交互。我们提出了MoReWorld，一个统一的运动区域引导框架，它利用共享的运动区域（即环境中随机器人动作而移动的部分），将动态学习和视频生成聚焦于与交互相关的区域。同一运动区域引导局部运动预测、区域感知监督以及运动门控的未来特征条件化。在推理时，运动（摘要在此处截断）

    arXiv:2611.27344v1 Announce Type: cross  Abstract: Action-conditioned video world models imagine future observations under candidate robot actions, providing a promising visual tool for robot policy evaluation and interaction simulation. However, existing video world models often optimize global reconstruction errors instead of emphasizing interaction-relevant regions. As a result, they may preserve overall scene appearance but overlook local interaction details, leading to physically implausible outcomes such as hallucinated contact effects and inconsistent robot-object interactions. We introduce MoReWorld, a unified motion-region-guided framework that uses a shared motion region, the part of the environment that moves in response to the robot's actions, to focus dynamics learning and video generation on interaction-relevant regions. The same motion region guides local motion prediction, region-aware supervision, and motion-gated future-feature conditioning. At inference time, the mot
    
[^93]: 视觉-语言安全表示中的单模态捷径与场景-意图联合依赖

    Unimodal Shortcuts and Joint Scene-Intent Dependence in Vision-Language Safety Representations

    [https://arxiv.org/abs/2611.27339](https://arxiv.org/abs/2611.27339)

    该研究审计发现现有情境安全基准均存在单模态捷径（仅凭图像即可高精度预测安全标签），并构建了一个将场景与意图交叉、排除单模态规则的受控评估，从而证明冻结视觉-语言模型的安全表示确实联合编码了场景与用户意图。

    

    向视觉-语言模型发出的请求是否安全往往取决于图像：询问如何上菜，对于新鲜食物是无害的，而对于变质食物则是有害的。在情境安全基准上取得的高准确率并不能告诉我们模型的安全表示是否同时编码了场景和用户意图，因为基准测试往往可以仅凭单一模态就能解决。我们研究了冻结的视觉-语言安全表示，审计了五个公开的情境基准，发现每个基准都存在单模态捷径：仅凭图像就能预测标签，而且在相同图像的提示对上，一个从未看过照片的探针就能达到0.9645的AUROC。因此，我们构建了一个受控评估，将安全与危险场景同观察类和行动类请求进行交叉组合，使得任何单模态规则都无法超过0.500的组排序准确率。在这一控制集上，作用于Qwen3-VL-8B和SmolVLM2-2.2B中间状态的线性探针分别达到了0.958和0.9…（原文摘要至此截断）

    arXiv:2611.27339v1 Announce Type: cross  Abstract: The safety of a request to a vision-language model often depends on the image: asking how to serve a dish is harmless for fresh food and harmful for spoiled food. High accuracy on situational safety benchmarks does not tell us whether a model's safety representation encodes both the scene and the user's intent, because a benchmark can often be solved from one modality alone. Studying frozen vision-language safety representations, we audit five public situational benchmarks and find that each admits a unimodal shortcut: images alone predict the labels, and on same-image prompt pairs a probe that never sees the photograph reaches 0.9645 AUROC. We therefore build a controlled evaluation that crosses safe and hazardous scenes with observation and action requests, so that no single-modality rule can exceed 0.500 group-ranking accuracy. On this control, linear probes on intermediate states of Qwen3-VL-8B and SmolVLM2-2.2B reach 0.958 and 0.9
    
[^94]: GGDriver：一览共享信息，注视差异之处，实现无感知规划

    GGDriver: Glance at What Is Shared, Gaze at What Differs for Perception-Free Planning

    [https://arxiv.org/abs/2611.27330](https://arxiv.org/abs/2611.27330)

    该论文提出GGDriver方法，通过让每个候选轨迹根据自身路径动态选择并关注相关的视觉证据进行细化与评分，从而提升端到端自动驾驶规划的性能。

    

    一个需要决定是左转还是继续直行的规划器，需要针对每个驾驶动作检查场景的不同部分。然而，许多端到端驾驶规划器使用相同的压缩场景表示来评估所有候选轨迹。我们让每个候选轨迹自行选择用于细化和评分的视觉证据。首先通过全局“一瞥”从共享场景特征生成轨迹假设。然后，我们将每条轨迹周围的走廊投影到相机视图中，并使用生成的掩码在骨干网络的原生分辨率上关注图像块特征。细化完成后，评分器再次投影每个最终方案，并利用沿其自身路径的证据对其进行排序。这种几何选择只需要预测轨迹和相机标定信息。在三种采用不同场景表示的规划器上（学习式寄存器、感知监督的鸟瞰图网格以及粗粒度图像标记网格），加入这种“注视”机制均能带来提升……

    arXiv:2611.27330v1 Announce Type: new  Abstract: A planner deciding whether to turn left or continue straight needs to inspect different parts of the scene for each maneuver. Yet many end-to-end driving planners evaluate every candidate trajectory against the same compressed scene representation. We let each candidate select the visual evidence used to refine and score it. A global glance first produces trajectory hypotheses from shared scene features. We then project a corridor around each trajectory into the camera views and use the resulting mask to attend to patch features at the backbone's native resolution. After refinement, the scorer projects each final proposal again and ranks it using evidence along its own path. This geometric selection requires only the predicted trajectory and camera calibration. On three planners with different scene representations (learned registers, a perception-supervised bird's-eye-view grid, and a coarse image-token grid), adding this gaze improves 
    
[^95]: H-MSAE：基于层次化稀疏概念覆盖的高效多样指令数据选择

    H-MSAE: Hierarchical Sparse Concept Coverage for Efficient and Diverse Instruction Data Selection

    [https://arxiv.org/abs/2611.27329](https://arxiv.org/abs/2611.27329)

    提出H-MSAE框架，利用Matryoshka稀疏自编码器将多分辨率概念组织成从粗到细的层次结构，并通过质量加权的边际收益递减覆盖机制，高效地选择高质量、多样化的指令数据子集，避免了传统方法对语义标签标注和图结构的依赖。

    

    大型指令微调语料库包含大量的语义冗余，这促使研究者选择紧凑的数据子集，在降低训练成本的同时保留有效的监督信号。现有的多样性感知选择器通常依赖于生成的语义标签和显式的图结构，带来了相当可观的标注和预处理开销。我们提出了H-MSAE，这是一个层次化稀疏指令选择框架，它将由Matryoshka稀疏自编码器学习到的多分辨率概念组织成从粗到细的层次结构。质量加权、边际收益递减的覆盖机制结合祖先传播，促进高质量且非冗余的语义覆盖，而相对于数据池的长度约束则规范子集的构成。我们通过对子集SFT效用的局部质量-多样性分析来论证该设计的合理性，并利用示例质量和表示空间多样性诊断对其进行检验。在Tulu 3数据池上，H-MSAE实现了……

    arXiv:2611.27329v1 Announce Type: new  Abstract: Large instruction-tuning corpora contain substantial semantic redundancy, motivating the selection of compact subsets that preserve useful supervision while reducing training cost. Existing diversity-aware selectors often rely on generated semantic tags and explicit graph structures, introducing considerable annotation and preprocessing overhead. We introduce H-MSAE, a hierarchical sparse instruction selection framework that organizes multi-resolution concepts learned by a Matryoshka sparse autoencoder into a coarse-to-fine hierarchy. Quality-weighted, diminishing-return coverage with ancestor propagation promotes high-quality and non-redundant semantic coverage, while pool-relative length constraints regulate subset composition. We motivate this design through a local quality-diversity analysis of subset SFT utility and examine it using example-quality and representation-space diversity diagnostics. On Tulu 3 data pool, H-MSAE achieves 
    
[^96]: 用更少词元进行推理：概念归一化的多语言草稿纸与可逆的模型自适应推理编码

    Reasoning in Fewer Tokens: Concept-Normalized Multilingual Scratchpads and Reversible Model-Adaptive Reasoning Codes

    [https://arxiv.org/abs/2611.27326](https://arxiv.org/abs/2611.27326)

    该论文提出了一种无损的中间推理压缩框架，通过BPE学习由PUA词元构成的压缩词表并扩展LLM词表后微调模型，在保持可解释性和可审计性的同时，平均减少14.7%的生成词元，推理轨迹长度最多缩短27.1%。

    

    思维链提示显著提升了大语言模型（LLM）的推理能力，但由于自回归解码过程冗长，带来了高昂的推理成本。以往的效率优化方法通过丢弃词元或使用连续的潜在表示来降低成本，然而这些有损方法从根本上牺牲了可解释性和可审计性。在本工作中，我们提出了一种新颖的中间推理压缩框架。我们对推理语料应用字节对编码（BPE）来学习一个由私有使用区（PUA）词元组成的压缩词表，其中每个词元代表频繁出现的语义片段。在扩展LLM词表后，我们通过严格的一对一映射对模型进行微调，因此我们的方法是无损的。我们在Llama-3.1-8B、Qwen2.5-14B和Qwen2.5-32B上，跨越四个不同的基准测试对该方法进行了评估。结果表明，我们的方法平均减少了14.7%的生成词元总数，推理轨迹长度最多减少27.1%。

    arXiv:2611.27326v1 Announce Type: cross  Abstract: Chain-of-thought prompting significantly improves the reasoning capabilities of large language models (LLMs) but incurs high inference costs due to lengthy autoregressive decoding. While previous efficiency approaches reduce costs by discarding tokens or utilizing continuous latent representations, these lossy methods fundamentally sacrifice interpretability and auditability. In this work, we introduce a novel compression framework for intermediate reasoning. We apply Byte-Pair Encoding (BPE) to reasoning corpora to learn a compressed vocabulary of Private Use Area (PUA) tokens, each representing frequent semantic spans. After expanding the LLM's vocabulary, we fine-tune the model via strictly 1:1, our method is lossless. We evaluate our approach on Llama-3.1-8B, Qwen2.5-14B, and Qwen2.5-32B across four diverse benchmarks. Our method reduces total generated tokens by 14.7% on average with reasoning trace lengths dropping by up to 27.1%
    
[^97]: 语言模型知道自己推理被哪些词引导吗？

    Do Language Models Know Which Words Steer Their Reasoning?

    [https://arxiv.org/abs/2611.27317](https://arxiv.org/abs/2611.27317)

    该论文发现了“幻影上下文影响”现象——语言模型的推理会被语义无关的词不成比例地引导，而模型自身却意识不到这种依赖关系（生成-判断矛盾），并提出“在线KV抑制”方法来缓解这一问题。

    

    虽然思维链提示能够引导语言模型产生中间推理步骤，但目前尚不清楚上下文中哪些词真正引导了思维链。我们识别出了“幻影上下文影响”现象：语言模型会被语义无关的上下文词不成比例地引导，即使同一模型在被直接询问时会否认这些词的相关性。为了定位这一现象，我们引入了“留一词JSD分析”，该方法从原本完整的上下文中每次移除一个词，以隔离该词对下一个词元分布的边际影响；移除后会导致预测翻转的词（枢轴词）会被传递给同一个大语言模型，让其自我判断是否识别到这种预测依赖关系。在四个模型和三个基准上的实验表明，语言模型在很大程度上并不知道哪些词引导了它们的推理，我们将这种不匹配称为“生成-判断矛盾”。受此现象启发，我们提出了“在线KV抑制”方法……

    arXiv:2611.27317v1 Announce Type: cross  Abstract: While chain-of-thought (CoT) prompting elicits intermediate reasoning steps from language models, it is unclear which words in the context actually steer the CoT. We identify Phantom Context Influence (PCI): language models are disproportionately steered by semantically irrelevant context words, even as the same models deny those words' relevance when asked. To localize PCI, we introduce Leave-One-Out Word JSD analysis, which removes one context word at a time from the otherwise intact context to isolate its marginal effect on the next-token distribution; words whose removal flips the prediction (pivot words) are passed to the same LLM to self-judge whether it recognizes this predictive dependence. Across four models and three benchmarks, language models largely do not know which words steer their reasoning, a mismatch we call the Generation-Judgment Contradiction. Inspired by this phenomenon, we propose Online KV Suppression: it score
    
[^98]: 地球与环境科学中AI就绪数据的评估框架

    A Framework for AI-Ready Data in Earth and Environmental Science

    [https://arxiv.org/abs/2611.27313](https://arxiv.org/abs/2611.27313)

    该论文提出了一个社区驱动的实践性框架，用于评估和提升地球与环境科学研究中数据的“AI就绪度”，弥补了FAIR原则在AI模型训练、评估与部署场景下数据要求的不足。

    

    人工智能（AI），特别是机器学习（ML），在地球与环境研究中的地位日益重要，然而当前研究数据基础设施在支持AI工作流程方面的适用性仍缺乏明确界定。尽管FAIR原则为科学数据管理提供了基础，但它们并未明确涉及当数据集用于训练、评估和部署AI模型时所产生的若干要求。本文提出了一个以实践为导向的框架，用于评估和改进分布式地球与环境数据基础设施中研究数据的AI就绪度。该框架通过结构化、社区驱动的流程开发而成，首先回顾了现有文献和就绪度框架，并结合对应用机器学习方法的科学家进行的有限调查所获得的反馈。其主要内容是由一个代表数据提供方等多学科背景的专家组所塑造的。

    arXiv:2611.27313v1 Announce Type: cross  Abstract: Artificial intelligence (AI), specifically machine learning (ML), is increasingly integral to Earth and environmental research, yet the suitability of current research data infrastructures for supporting AI workflows remains poorly defined. While the FAIR principles provide a foundation for scientific data management, they do not explicitly address several requirements arising when datasets are used for training, evaluating, and deploying AI models. This article presents a practically motivated framework for assessing and improving AI-readiness of research data within distributed Earth and environmental data infrastructures. The framework was developed through a structured, community-driven process, beginning with a review of existing literature and readiness frameworks combined with feedback from a limited survey of scientists applying ML methods. Its main aspects were shaped by a multidisciplinary expert group representing data provi
    
[^99]: $\gamma$-Scape：一个面向移动具身智能的以智能体为中心的世界

    $\gamma$-Scape: An Agent-Centric World for Mobile Embodied Intelligence

    [https://arxiv.org/abs/2611.27306](https://arxiv.org/abs/2611.27306)

    该论文提出了 $\gamma$-Scape，一个以智能体为中心的移动具身智能仿真与基准平台，使智能体能在持久物理世界中组合运动工具与习得技能完成导航与操作结合的完整任务，并通过智能体式采集框架自主生成接触丰富的交互示范数据。

    

    近年来的具身智能体通过可执行程序、工具使用和交互反馈来构建和调整机器人行为。在移动操作领域研究这些系统，需要物理世界能够将智能体主导的执行过程与其物理后果以及可复用的交互经验联系起来。我们提出了 $\gamma$-Scape，一个面向移动具身智能的以智能体为中心的仿真与基准测试平台。智能体可以组合运动工具和已学习的技能，在一个持久存在的物理世界中执行结合导航与操作的完整任务。BenchForge 将结构化目标落地到具体的物理任务实例中，提供指令、参考程序和目标评估器，在定义任务目标的同时保留智能体选择有效解决方案的自由度。一个智能体式的数据采集框架进一步使智能体能够通过持续控制和物理反馈执行丰富的接触式交互，从而产生示范数据。辅助参考……

    arXiv:2611.27306v1 Announce Type: new  Abstract: Recent embodied agents construct and adapt robot behavior through executable programs, tool use, and interaction feedback. Studying these systems in mobile manipulation calls for physical worlds that connect agent-directed execution with its physical consequences and reusable interaction experience. We introduce $\gamma$-Scape, an agent-centric simulation and benchmark platform for mobile embodied intelligence. Agents compose motion tools and learned skills to carry out complete tasks that combine navigation and manipulation in a persistent physical world. BenchForge grounds structured objectives in physical task instances with instructions, reference programs, and goal evaluators, defining task goals while preserving agents' choice of valid solutions. An agentic collection harness further enables agents to produce demonstrations by carrying out contact-rich interactions through sustained control and physical feedback. Assisted reference
    
[^100]: 推理与图像描述能否增强是/否型视觉数学理解能力？

    Can Reasoning and Image Description Enhance Yes/No Visual Mathematical Understanding?

    [https://arxiv.org/abs/2611.27302](https://arxiv.org/abs/2611.27302)

    本文通过配对评估发现，思维链等显式推理能显著提升视觉语言模型在是/否型视觉数学问题上的表现，而文本图像描述反而可能损害性能。

    

    大型视觉语言模型（VLMs）在通用视觉任务上已取得强劲表现，但其理解数学图像的能力仍然有限。本文研究了显式推理和文本化的图像描述能否提升VLMs在二元是/否型视觉数学问题上的表现。为此，我们精心构建了一个配对评估集，在不同的提示与输入条件下对知名开源VLMs进行基准测试。特别地，我们采用思维链提示，并使用文本描述来替换或增强视觉输入，从而分离并评估模型的数学理解能力。结果显示，显式推理能够提升性能，尤其在配对一致性准确率方面表现突出；而文本图像描述并不能持续带来帮助，甚至可能降低性能。这些发现表明，推理对于稳定的视觉数学判断至关重要，而描述性文本输入的作用则较为有限。

    arXiv:2611.27302v1 Announce Type: cross  Abstract: Large Vision-Language Models (VLMs) have achieved strong performance on general visual tasks, yet their ability to understand mathematical images remains limited. This paper investigates whether explicit reasoning and textual image descriptions can enhance VLMs' performance on binary yes/no visual mathematical questions. To achieve this, we curated a paired evaluation set to benchmark prominent open-source VLMs under varying prompt and input conditions. In particular, we leverage Chain-of-Thought (CoT) prompting and use textual descriptions, either replacing or augmenting visual inputs, to isolate and evaluate mathematical comprehension. Our results show that explicit reasoning improves performance, especially in paired consistency accuracy, while textual image descriptions do not consistently help and may degrade performance. These findings suggest that reasoning is important for stable visual mathematical judgement, whereas descripti
    
[^101]: HookEvo：一种基于钩子启用的智能体运行时的协同插件进化框架

    HookEvo: A Framework for Coordinated Plugin Evolution on Hook-Enabled Agent Runtimes

    [https://arxiv.org/abs/2611.27294](https://arxiv.org/abs/2611.27294)

    提出了HookEvo框架，利用智能体运行时的生命周期钩子，在每轮迭代中协同进化可执行插件处理器与共享状态，无需改动冻结的执行器即可让Qwen3-4B在ALFWorld、DBBench等多个基准上取得领先成绩。

    

    修复使用工具的智能体中的故障，可能需要对运行框架如何解释工具回执、组装上下文以及检查动作等方面进行协同修改。我们提出了HookEvo，这是一个利用现有智能体运行时生命周期钩子来进化特定任务运行框架的框架。我们在DeepSeek Harness上实例化了HookEvo。在每一轮中，一个独立的改进器从执行轨迹中诊断故障并提出候选方案。HookEvo在匹配的任务上将候选方案与当前运行框架进行比较评估，并将保留的版本带入下一轮。单个候选方案可以联合编辑可执行的插件处理器以及它们在任务中共享的状态，使工具回执能够为上下文提供信息并控制后续动作。使用冻结的Qwen3-4B执行器，所选的运行框架在已报告的Qwen3-4B对比中处于领先地位，在ALFWorld上达到95.41%，在DBBench上达到70.56%，在WebShop上达到48.17%，在OSBench上达到41.67%。在三个改进器和三个……

    arXiv:2611.27294v1 Announce Type: new  Abstract: Repairing failures in tool-using agents can require coordinated changes to how the harness interprets tool receipts, assembles context, and checks actions. We introduce HookEvo, a framework that uses an existing agent runtime's lifecycle hooks to evolve task-specific harnesses. We instantiate HookEvo on DeepSeek Harness. In each round, a separate improver diagnoses failures from execution traces and proposes a candidate. HookEvo evaluates the candidate against the current harness on matched tasks and carries the retained version into the next round. A single candidate can jointly edit executable plugin handlers and the state they share within a task, allowing a tool receipt to inform context and control a later action. Using a frozen Qwen3-4B executor, the selected harnesses lead the reported Qwen3-4B comparisons, achieving 95.41% on ALFWorld, 70.56% on DBBench, 48.17% on WebShop, and 41.67% on OSBench. Across three improvers and three e
    
[^102]: EgoLAM：从手部和头部运动中学习潜在动作

    EgoLAM: Learning Latent Actions from Hand and Head Motion

    [https://arxiv.org/abs/2611.27290](https://arxiv.org/abs/2611.27290)

    EgoLAM 通过用第一人称视频中实测的腕部运动监督潜在动作编码的学习，同时将头部运动（相机运动）分离并单独建模，使机器人策略能从自我中心视频中学习真正代表手部操作的潜在动作。

    

    潜在动作模型通过将两帧之间的变化转化为训练目标，使机器人策略能够从视频中学习。在第一人称（自我中心）视频中，一个杯子在画面中移动，可能是因为有手在推它，也可能是因为佩戴者转动头部。这两种变化都有助于预测下一帧，但只有前者描述了操作行为。在训练过程中，EgoLAM 使用实测的腕部运动来监督动作编码，并将相机运动单独提供给前向模型。即使佩戴者在推杯子的同时转动头部，训练目标仍然描述的是手如何推杯子。动作编码器在每个手部周围汇集物体槽位，腕部的平移和旋转可以在统一参考坐标系中从编码中解码出来。提取一个编码只需要两帧 RGB 图像。除了 LARYBench 的分类和回归任务外，我们还测量了冻结编码保留了多少腕部运动信息，以及它对合成视角或光照变化的响应强度。EgoLAM 具有……

    arXiv:2611.27290v1 Announce Type: cross  Abstract: Latent action models let a robot policy learn from video by turning the change between two frames into a training target. In egocentric video, a cup can move across the image because a hand pushes it or because the wearer turns. Both changes help predict the next frame, but only the first describes manipulation. During training, EgoLAM supervises the action code with measured wrist motion and supplies camera motion separately to the forward model. Even when the wearer turns while pushing a cup, the training target still describes how the hand pushed it. The action encoder pools object slots around each hand, and wrist translation and rotation can be decoded from the code in a common reference frame. Extracting a code then requires only two RGB frames. Alongside LARYBench classification and regression, we measure how much wrist motion a frozen code retains and how strongly it responds to a synthetic view or lighting change. EgoLAM has t
    
[^103]: 基于条件对比正则化在扩散模型强化学习中保留预训练CFG结构

    Preserving Pretrained CFG Structure in RL for Diffusion Models with Conditional Contrast Regularization

    [https://arxiv.org/abs/2611.27288](https://arxiv.org/abs/2611.27288)

    提出条件对比正则化方法，在扩散模型强化学习后训练中保留预训练CFG的几何结构，作为诊断奖励黑客行为的有效指标，从而实现奖励提升与整体生成质量之间更稳定的权衡。

    

    强化学习已成为扩散模型后训练的重要组成部分。然而在实践中，标准强化学习往往能够提升目标奖励，却无法可靠地保持整体生成质量，导致不稳定的奖励-质量权衡，我们认为这是奖励黑客行为的一种表现形式。由于无分类器引导在提升生成图像的视觉清晰度、提示词对齐和感知质量方面的作用已为人所知，我们研究其底层结构能否在强化学习过程中作为有用的先验。我们提出了一个统一的几何视角，将CFG和RL都解释为同一预测空间中的校正向量，并将RL引起的校正向量在预训练CFG校正方向上的投影确定为衡量质量相关结构保留程度的有效诊断指标。几何分析和旋转干预实验进一步表明，预训练的CFG方向携带着对生成质量有用的信息。

    arXiv:2611.27288v1 Announce Type: cross  Abstract: Reinforcement Learning (RL) has become an important component in the post-training of diffusion models. In practice, however, standard RL often improves the target reward without reliably preserving holistic generation quality, leading to unstable reward-quality trade-offs that we view as a form of reward hacking. Since Classifier-Free Guidance (CFG) is known to improve visual clarity, prompt alignment, and perceptual quality in generation, we investigate whether its underlying structure can serve as a useful prior during RL. We present a unified geometric view that interprets both CFG and RL as correction vectors in a common prediction space, and identify the projection of the RL-induced correction onto the pretrained CFG correction as a useful diagnostic of quality-related structural preservation. Geometric analysis and rotation intervention further show that the pretrained CFG direction carries information useful for generation qual
    
[^104]: 信念状态管理与评分标准奖励相结合的智能体预测

    Belief-State Management Meets Rubric Rewards for Agentic Forecasting

    [https://arxiv.org/abs/2611.27286](https://arxiv.org/abs/2611.27286)

    该论文提出一个智能体事件预测训练框架，通过主动信念状态管理（PBM）将冗长上下文压缩为紧凑信念状态，并引入五维不依赖结果的SEIPH评分标准提供过程质量奖励，从而解决了LLM智能体预测中上下文累积和奖励信号嘈杂的两大难题。

    

    当前的大语言模型（LLM）智能体在预测现实世界事件方面仍然存在困难。这源于两个主要局限性：i) 标准的ReAct框架在一个不断增长的上下文中积累推理和证据；ii) 基于结果的预测奖励产生缓慢且嘈杂的信号，无法区分合理的推理与幸运的猜测。为了解决这些问题，我们引入了一个预测训练框架，包括框架设计和后训练两部分。在框架方面，我们提出了主动信念状态管理（PBM），为智能体配备一个折叠工具，将冗长的上下文压缩为紧凑的信念状态，并重置上下文以便后续探索。在训练方面，我们开发了SEIPH，这是一个五维的、不依赖结果的评分标准，用于提供过程质量奖励。使用基于评分标准的奖励，我们采用PPO和GRPO对一个35B模型进行后训练，得到了DeepForecaster-PPO和DeepForecaster-GRPO。理论上，我们证明……

    arXiv:2611.27286v1 Announce Type: new  Abstract: Current large language model (LLM) agents still struggle to forecast real-world events. This stems from two primary limitations: i) standard ReAct harness accumulates reasoning and evidence within one growing context, and ii) outcome-based rewards in forecasting lead to a slow and noisy signal that cannot distinguish sound reasoning from lucky guesses. To address these, we introduce a forecasting training framework including harness design and post-training. For the harness side, we propose Proactive Belief-State Management (PBM), equipping agents with a fold tool to compress lengthy context into compact belief states and reset the context for subsequent exploration. For the training side, we develop SEIPH, a five-dimensional, outcome-blind rubric, providing process-quality rewards. Using the rubric-based reward, we post-train a 35B model with PPO and GRPO, yielding DeepForecaster-PPO and DeepForecaster-GRPO. Theoretically, we prove that
    
[^105]: DermKG：一个以概念为中心的知识图谱，用于基于皮肤病学形态特征实现可解释的皮肤病诊断

    DermKG: A Concept-Centric Knowledge Graph for Interpretable Skin Disease Diagnosis from Dermatological Morphology

    [https://arxiv.org/abs/2611.27283](https://arxiv.org/abs/2611.27283)

    本文构建了首个以概念为中心的皮肤病知识图谱DermKG（3,452个节点、19,365条边，涵盖疾病概念特征、共现关系、诊断层级和临床规则），首次量化了形态学概念所承载的诊断信息上限（疾病38.9%、九大类别77.6%），并据此提出图神经网络DermKG-Net以实现可解释的皮肤病诊断。

    

    皮肤科医生使用标准化的形态学术语词汇来描述皮损，这使得基于概念的模型在可解释的皮肤病诊断中极具吸引力。然而，这些概念实际承载了多少诊断信息却很少被量化。我们提出了DermKG，一个以概念为中心的知识图谱，构建于Fitzpatrick17k数据集中经专家标注的SkinCon子集之上（包含3,230张图像、48个概念、113种疾病）。DermKG包含8种类型的3,452个节点和11种关系类型的19,365条边，编码了疾病概念特征、概念共现关系、三级诊断层次结构以及20条临床形态学规则，所有这些均由训练数据支持（15条强规则、5条中等规则）。信息上限分析表明，3,230个皮损仅坍缩为813个不同的概念向量，这将仅基于概念的疾病top-1准确率限制在38.9%，九大疾病类别的top-1准确率限制在77.6%。在此上限范围内，DermKG-Net——一种图神经网络（原文在此处截断）

    arXiv:2611.27283v1 Announce Type: new  Abstract: Dermatologists describe lesions with a standardized morphological lexicon, which makes concept-based models attractive for interpretable skin disease diagnosis. How much diagnostic information these concepts actually carry, however, is rarely quantified. We present DermKG, a concept-centric knowledge graph built on the expert-annotated SkinCon subset of Fitzpatrick17k (3,230 images, 48 concepts, 113 diseases). DermKG has 3,452 nodes of 8 types and 19,365 edges of 11 relation types, and encodes disease concept profiles, concept co-occurrence, a three-level diagnostic hierarchy, and 20 clinical morphology rules, all of which are supported by the training data (15 strong, 5 moderate). An information-ceiling analysis shows that the 3,230 lesions collapse onto only 813 distinct concept vectors, bounding concept-only top-1 accuracy at 38.9\% for diseases and 77.6\% for the nine categories. Within this bound, DermKG-Net, a graph neural network 
    
[^106]: EAGLE：基于分层证据的实体感知定位方法用于高分辨率图像理解

    EAGLE: Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding

    [https://arxiv.org/abs/2611.27264](https://arxiv.org/abs/2611.27264)

    EAGLE提出了一种即插即用框架，通过筛选可靠的视觉定位注意力头、实体感知区域提取，以及分层证据构建与问题引导的视图组合选择，在不改变冻结多模态大模型参数的情况下显著增强其对高分辨率图像的细粒度理解能力。

    

    多模态大语言模型（MLLM）在高分辨率图像的细粒度理解方面仍然存在困难：注意力头在定位目标实体及其属性时的可靠性参差不齐，而局部细节往往与更广泛的场景上下文相脱节。我们提出了EAGLE（基于分层证据的实体感知定位高分辨率图像理解框架），这是一个联合选择定位信号并组织分层视觉证据的即插即用框架。EAGLE利用细粒度数据集筛选具有视觉定位能力的注意力头，随后通过实体感知定位提取目标特定区域。接着，它整合分层证据构建与问题引导的路由机制来选择最优视图组合，并将其输入冻结的多模态大语言模型中，以增强细粒度视觉理解。在Qwen2.5-VL-7B和InternVL3-8B上的实验表明，EAGLE能够持续提升模型性能表现。

    arXiv:2611.27264v1 Announce Type: cross  Abstract: Multimodal large language models (MLLMs) still struggle with fine-grained understanding of high-resolution images: attention heads vary in their reliability for grounding target entities and their attributes, while localized details are often disconnected from the broader scene context. We propose EAGLE (Entity-Aware Grounding with Layered Evidence for High-Resolution Image Understanding), a plug-in framework that jointly selects grounding signals and organizes hierarchical visual evidence. EAGLE employs a fine-grained dataset to screen visually-grounded attention heads, followed by entity-aware grounding to extract target-specific regions. It then integrates hierarchical evidence construction and question-guided routing to select optimal view combinations, feeding them into the frozen MLLM to enhance fine-grained visual understanding. Experiments with Qwen2.5-VL-7B and InternVL3-8B demonstrate that EAGLE consistently improves performa
    
[^107]: 没有完美的护栏，也没有完美的守卫：论大语言模型输出策略的可判定性

    No Perfect Guardrail & No Perfect Guard: On the Decidability of Output Policies for Large Language Models

    [https://arxiv.org/abs/2611.27260](https://arxiv.org/abs/2611.27260)

    本文利用可计算性理论证明：只有当大语言模型的输出策略可判定时，才存在能完美区分禁止与允许输出的可计算护栏；对于不可判定的策略，任何护栏都必然误分类无穷多字符串，因此护栏不能免除人类自身的审查与责任。

    

    期待护栏能够提供普遍的安全性保证，未能认识到计算执行与策略规范本身的局限。我们认为，承诺这种完美是一种精准度上的伦理失败，而援引这种完美来推卸责任则是一种道德失败。护栏可以支持负责任的行动，但无法免除研究人员、开发者、政策制定者或用户审视其所主张、允许和阻止之内容的义务。我们确立了一个精确的计算边界：如果每个有限字符串都是语言模型可能的输出，那么一个能够阻止所有被禁止输出并放行所有被允许输出的可计算守卫，当且仅当该策略是可判定的时候才存在。对于一个不可判定的策略，每个守卫都会错误分类无穷多个字符串，而每个阻止了所有被禁止输出的守卫都必然排除无穷多个被允许的输出。我们将这一刻画应用于算术真理、程序行为等领域。

    arXiv:2611.27260v1 Announce Type: new  Abstract: Expecting guardrails to provide a general guarantee of safety fails to recognize the limits of computational enforcement and policy specification. We argue that promising such perfection is an ethical failure of precision, and that invoking it to abdicate responsibility is a moral failure. Guardrails can support responsible action, but cannot relieve researchers, developers, policymakers, or users of their obligations to examine what they claim, permit, and prevent.   We establish a precise computational boundary. If every finite string is a possible output of a language model, a computable guard that blocks every forbidden output and passes every permitted one exists if and only if the policy is decidable. For an undecidable policy, every guard misclassifies infinitely many strings, and every guard that blocks all forbidden outputs excludes infinitely many permitted ones. We apply this characterization to arithmetic truth, program behav
    
[^108]: 针对深度伪造生成的主动防御基准

    Proactive Defense Benchmark against Deepfake Generation

    [https://arxiv.org/abs/2611.27253](https://arxiv.org/abs/2611.27253)

    该论文提出了首个针对深度伪造主动防御的综合评估基准，揭示了保真度与身份指标的矛盾解读以及白盒峰值性能背后的过拟合问题，并建立严格标准以推动真正可泛化防护方法的发展。

    

    尽管针对深度伪造的主动防御方法不断涌现，但由于缺乏统一的评估协议，导致无法进行公平比较，也掩盖了关键性的漏洞。为弥补这一空白，我们提出了首个全面的基准，系统性地评估干扰效果、鲁棒性和可迁移性，涵盖像素、感知和身份三个层面的指标。我们广泛的分析表明，保真度指标与身份指标捕捉的是相互正交的性能维度，单独依赖其中任何一个往往会导致相互矛盾的解读。此外，我们发现了一个根本性的权衡现象：白盒设置下的峰值性能往往意味着过拟合，为此我们引入了一种校准评估方法以纠正由生成器引起的身份偏差。通过揭示这些盲点，我们建立了一套严格的标准，用以指导开发真正具备泛化能力的防护方法。

    arXiv:2611.27253v1 Announce Type: cross  Abstract: Despite the proliferation of proactive defenses against deepfakes, the lack of a unified evaluation protocol precludes fair comparison and masks critical vulnerabilities. To bridge this gap, we present the first comprehensive benchmark that systematically assesses disruption, robustness, and transferability encompassing pixel, perceptual, and identity metrics. Our extensive analysis reveals that fidelity and identity metrics capture orthogonal performance axes, often leading to conflicting interpretations when relied upon individually. Furthermore, we identify a fundamental trade-off where peak white-box performance signals overfitting, and we introduce a calibrated evaluation to correct generator-induced identity bias. By exposing these blind spots, we establish a rigorous standard to guide the development of genuinely generalizable protections. Project page is available at: https://proactivedefensebenchmark.github.io/
    
[^109]: CreditFilter：基于响应内熵排名的校准 Token 权重方法用于 RLVR

    CreditFilter: Calibrated Token Weights over the Within-Response Entropy Rank for RLVR

    [https://arxiv.org/abs/2611.27227](https://arxiv.org/abs/2611.27227)

    该论文将 token 对结果的影响形式化为“分叉值”，并利用分叉概率随响应内熵排名呈指数衰减的实证发现，提出 CreditFilter——通过熵排名校准 token 权重、在响应内部重新分配信用，从而改进免评论家的可验证奖励强化学习（RLVR）。

    

    免评论家的可验证奖励强化学习（RLVR）通常为每个 token 分配统一的优势值。诸如 80/20 掩码之类的 token 重加权启发式方法偏向高熵 token，但无法衡量一个 token 对最终结果的实际影响。我们将这种影响形式化为“分叉值”：即策略在下一个 token 的各个选择上成功概率的方差。沿一条响应的期望分叉值之和恰好等于奖励方差。因此，对于一个提示词的多条响应取期望，token 信用就是一个在分叉处花费的固定预算。这启发了在响应内部重新分配信用的权重设计。分叉值可以分解为“分叉后果”（需要代价高昂的后续模拟来估计）和“分叉概率”（由 softmax 直接提供）。在 Qwen3-8B-Base 上，该分叉概率随 token 在响应内的熵排名以特定类别相关的速率呈指数衰减，揭示出大多数 token 几乎不承载信用（原文摘要在此处截断）。

    arXiv:2611.27227v1 Announce Type: cross  Abstract: Critic-free reinforcement learning with verifiable rewards (RLVR) typically assigns a uniform advantage to every token. Token-reweighting heuristics like the 80/20 mask favor high-entropy tokens, but fail to measure a token's actual impact on the outcome. We formalize this impact as the forking value: the variance of the success probability over the policy's next-token choices. The expected forking values along a response sum exactly to the reward variance. Token credit is thus, in expectation over a prompt's responses, a fixed budget spent at the forks. This motivates weights that redistribute credit within a response. The forking value factorizes into a fork consequence, which requires costly continuations to estimate, and a fork probability, which the softmax provides. On Qwen3-8B-Base, this fork probability decays exponentially with a token's within-response entropy rank at class-specific rates, revealing that most tokens carry neg
    
[^110]: 用于机器人运动的激光雷达信号：球面距离图像实现鲁棒的仿真到现实迁移

    LiDAR Signals for Robot Locomotion: Spherical Range Images Enable Robust Sim-to-Real Transfer

    [https://arxiv.org/abs/2611.27223](https://arxiv.org/abs/2611.27223)

    该论文提出一种基于球面距离图像编码的激光雷达感知管线，通过传感器级和表示级的双重对齐实现仿真与现实间的感知一致性，使腿式机器人无需教师-学生蒸馏即可实现鲁棒的仿真到现实迁移部署。

    

    现有的腿式运动系统通常依赖深度图像或在线高程图。然而，基于深度的策略由于深度传感器建模不完善而存在感知层面的仿真到现实差异，而基于地图的方法则容易受到漂移和错位的影响。激光雷达能够直接测量周围环境的几何结构，但原始激光雷达扫描缺乏适用于高频控制的紧凑、结构化表示。我们提出了一种激光雷达感知管线，通过传感器级和表示级的对齐来保持仿真与现实之间的感知一致性。在传感器层面，我们从物理激光雷达中恢复逐帧的角采样模式，并使用GPU光线投射在仿真中进行重放。在表示层面，仿真和真实点云回波经过相同的球面距离图像编码，产生固定大小的策略观测，用于端到端训练和直接部署，无需教师-学生蒸馏。

    arXiv:2611.27223v1 Announce Type: new  Abstract: Existing legged-locomotion systems commonly rely on depth images or online elevation maps. However, depth-based policies suffer from perceptual sim-to-real discrepancies due to imperfect depth sensor modeling, while map-based approaches are vulnerable to drift and misalignment. LiDAR offers direct measurements of the surrounding geometry, but raw LiDAR scans lack a compact, structured representation for high-frequency control. We propose a LiDAR perception pipeline that maintains perceptual consistency across simulation and reality through sensing- and representation-level alignment. At the sensor level, we recover frame-wise angular sampling patterns from the physical LiDAR and replay them in simulation with GPU ray casting. At the representation level, simulated and real returns undergo the same spherical range image encoding, yielding a fixed-size policy observation for end-to-end training and direct deployment without teacher--studen
    
[^111]: Bel：无需训练的视觉智能体步级策略改进

    Bel: Step-Level Policy Improvement of Visual Agents Without Training

    [https://arxiv.org/abs/2611.27222](https://arxiv.org/abs/2611.27222)

    Bel提出了一种无需训练的视觉智能体步级策略改进方法，通过将每个候选动作绘制在其作用的图像上，让决策模型基于视觉判断从冻结策略的采样中选出更优动作。

    

    视觉智能体根据它所看到的内容行动：它在屏幕上点击和输入，或者在摄像头图像上指点以告诉机器人在哪里行动。一个错误的步骤可能导致任务失败，然而正确的动作往往就在它自己的采样之中，只是被它的贪心解码所错过。选择正确的动作是一种视觉判断，因为一个动作最好在它实际发挥作用的地方被评判，而不是仅凭其文本或坐标：作为文本，一次点击只是两个数字；而绘制在屏幕上，它就是一份空草稿的“发布”按钮。我们发现这一点是决定性的：在给出截图和点击坐标的情况下，决策模型在GUI-R1-3B的采样中做出的选择，平均而言明显差于将点击绘制在其落点位置时的表现。Bel将这一发现转化为一种无需任何训练的方法，它只需要一个冻结策略的采样和一张可以绘制每个动作的图像。它将每个候选动作标记在动作所作用图像各自的副本上，并让决策模型给出执行该动作是正确决策的概率……

    arXiv:2611.27222v1 Announce Type: new  Abstract: A visual agent acts on what it sees: it taps and types on a screen, or points on a camera image to show a robot where to act. One wrong step can fail its task, yet the right action is often among its own samples, passed over by its greedy decode. Choosing it is a visual judgment, since an action is best judged where it acts, not by its text or coordinates alone: as text, a tap is two numbers; drawn on the screen, it is the Post button of an empty draft. We find that this is decisive: given the screenshot and a tap's coordinates, a decision model chooses among GUI-R1-3B's samples significantly worse on average than when the tap is drawn where it lands. Bel turns this finding into a method that trains nothing and needs only a frozen policy's samples and an image on which each action can be drawn. It marks each candidate on its own copy of the image it acts on, asks a decision model for the probability that executing it is the right decisio
    
[^112]: NavTTT：基于分层测试时训练的视觉-语言导航自适应参数记忆

    NavTTT: Adaptive Parametric Memory via Hierarchical Test-Time Training for Vision-and-Language Navigation

    [https://arxiv.org/abs/2611.27219](https://arxiv.org/abs/2611.27219)

    提出NavTTT，一种导航条件化的分层测试时训练方法，通过自适应记忆形成与分层记忆整合，利用全局导航上下文优先保留信息性视觉证据，解决长导航过程中冗余观测干扰长期记忆保留的问题。

    

    视觉-语言导航（VLN）要求智能体在长导航过程中对持续的流式视觉观测进行积累与推理。现有方法受限于Transformer架构有限的上下文容量，难以维持长期历史信息。为解决该问题，快速权重的测试时训练（TTT）提供了一种将流式观测编码为不断演化的参数状态的机制。然而，现有的TTT无法区分信息性观测与重复性视觉内容，使得冗余token会干扰对后续决策所需信息的保留。我们提出导航条件化测试时训练（NavTTT），为视觉-语言导航引入自适应记忆形成与分层记忆整合。NavTTT利用全局导航上下文来优先处理具有信息量的视觉证据，并将语言指令与空间信息相结合。

    arXiv:2611.27219v1 Announce Type: cross  Abstract: Vision-and-language navigation (VLN) demands that agents accumulate and reason over streaming visual observations throughout long navigation episodes. Existing approaches struggle to maintain long-term history, constrained by the finite context capacity of transformer architectures. To solve the problem, fast-weight test-time training (TTT) provides a mechanism for encoding streaming observations into an evolving parameter state. However, existing TTT does not distinguish informative observations from repetitive visual content, allowing redundant tokens to interfere with the retention of information needed for subsequent decisions. We propose Navigation-Conditioned Test-Time Training (NavTTT), which introduces adaptive memory formation and hierarchical memory integration for vision-and-language navigation. NavTTT leverages global navigation context to prioritize informative visual evidence and integrates language instructions with spat
    
[^113]: 正确先例诱发端侧LLM代理的策略违规

    Correct Precedents Induce Policy Violations in On-Device LLM Agents

    [https://arxiv.org/abs/2611.27216](https://arxiv.org/abs/2611.27216)

    该论文首次发现“正确先例泛化”这一新型安全故障——端侧LLM代理在正确放行一个解除限制的条目后，会将该决策泛化迁移到仍受限制的后续条目上，导致即使整个会话历史中没有任何错误步骤也可能以策略违规告终，该问题在来自九个开发者的13个开源模型中普遍存在，且单个正确先例即可将最严重模型的安全率削减约一半。

    

    本地-云端代理系统将云端模型与一个小型的设备端代理配对，该代理对私有数据进行操作，通常作为不可逆操作执行前的最后一道检查。我们首次识别出一种故障模式：此类代理会在会话内被其自身的正确决策引导至违规。在正确放行了一个限制已被解除的条目后，该代理更有可能放行下一个仍受限制的条目。由于触发因素是一个正确的决策，即使历史记录中没有任何错误步骤，会话仍可能以违规告终。我们将这种故障称为“正确先例泛化”，并在来自九个开发者的13个开源权重模型中发现了该问题。单个正确先例可将受影响最严重的代理的安全率大约削减一半，且当被拒绝的请求在新会话中被简单重发时，风险会进一步累积。对代理历史记录的干预实验表明，代理会将其对已放行条目的决策迁移到受限制条目上。

    arXiv:2611.27216v1 Announce Type: cross  Abstract: Local-cloud agent systems pair a cloud model with a small on-device agent that acts on private data, often as the last check before an irreversible action. We identify, for the first time, a failure in which such an agent is led into a violation by its own correct decisions within a session. After it correctly releases an item whose restriction was lifted, it becomes more likely to release the next item that is still restricted. Because the trigger is a correct decision, a history with no faulty step can still end in a violation. We call this failure correct-precedent generalization and find it in 13 open-weight models from nine developers. A single correct precedent can cut the safe rate of the most affected agent roughly in half, and the risk compounds when a refused request is simply sent again in a new session. Interventions on the agent's history indicate that it carries its decision on the released item over to the restricted one
    
[^114]: Neural Defend 在全球音频深度伪造检测基准测试中排名第一——ARCAS 亮相 Podonos 独立排行榜

    Neural Defend Ranks #1 in Global Audio Deepfake Detection Benchmark - ARCAS on the Podonos Independent Leaderboard

    [https://arxiv.org/abs/2611.27215](https://arxiv.org/abs/2611.27215)

    Neural Defend 的 ARCAS 基础模型（290M 与 90M）在 Podonos 独立音频深度伪造检测基准上并列第一，以统计学上显著的优势超越 Resemble AI、Hiya、Pindrop、Hive 等所有其他商用及开源检测系统。

    

    arXiv:2611.27215v1 公告类型：cross 摘要：AI 生成音频的商用检测器通常以买家无法查验的内部测试集结果进行宣传推广。而带有私有标签的独立基准测试能提供更可靠的比较。我们报告了 Neural Defend 公司的 ARCAS 基础模型在 Podonos 音频深度伪造检测基准测试上的成绩。该基准是一个独立排行榜，在一组保留测试集上对商用检测 API 和开源检测器进行评分，测试集包含真实语音以及来自当前文本转语音（TTS）和语音克隆系统的语音，以六种文件格式提供，且标签仅由 Podonos 掌握。在 2026 年 10 月 29 日发布的榜单中，涵盖了来自 16 家商用供应商的 28 个系统以及领先的开源模型，ARCAS 290M 与较小的 ARCAS 90M 并列第一，超越了包括 Resemble AI、Hiya、Pindrop、Hive 和 Reality Defender 在内的所有其他商用系统。两个模型的错误率均低于所有其他系统，且在统计学上具有显著差距……

    arXiv:2611.27215v1 Announce Type: cross  Abstract: Commercial detectors of AI-generated audio are usually promoted with results on internal test sets that buyers cannot inspect. Independent benchmarks with private labels give a more reliable comparison. We report the results of Neural Defend's ARCAS foundation models on the Podonos audio deepfake detection benchmark, an independent leaderboard that scores commercial detection APIs and open-source detectors on one held-out set of real speech and speech from current text-to-speech and voice-cloning systems, delivered in six file formats, against labels that only Podonos holds. In the release of 29 October 2026, which covers 28 systems from 16 commercial vendors and the leading open models, ARCAS 290M and the smaller ARCAS 90M share first place, ahead of every other commercial system, including Resemble AI, Hiya, Pindrop, Hive, and Reality Defender. Both models make fewer errors than any other system by a statistically clear margin, keep 
    
[^115]: EAF-TTS：重新思考面向流模型的编解码器潜在坐标

    EAF-TTS: Rethinking Codec-Latent Coordinates for Flow TTS

    [https://arxiv.org/abs/2611.27214](https://arxiv.org/abs/2611.27214)

    EAF-TTS提出将完整的ZCA坐标协议（包括训练与采样坐标、诱导损失度量及终端扰动）应用于声学潜在空间，重新调整编解码器潜在坐标的几何结构，使其更适合从各向同性高斯噪声进行流式TTS生成。

    

    潜在文本到语音（TTS）系统生成由冻结解码器解码的编解码器潜在表示，但良好的重建效果并不能保证其几何结构适合从各向同性高斯噪声出发的流生成。我们提出了嵌入式声学流TTS（EAF-TTS），这是ELF在声学潜在表示上的适配版本，具备音素和时长条件生成、共享网络终端校正以及冻结的WaveVAE解码。EAF-TTS支持一个从训练数据估计得到的固定ZCA接口，不引入任何可学习参数。编解码器、条件模块、流骨干网络以及20次调用的推理过程均保持共享。其核心干预是完整的ZCA坐标协议——涵盖训练与采样坐标、诱导损失度量以及终端扰动——而并非仅在推理阶段应用的变换。在单说话人LJSpeech数据集上，64维训练潜在协方差的条件数为1355.44。在文本不相交的Val399严格-GT测试集上，使用三个匹配种子下的固定E100原始检查点，ZCA（摘要原文在此处截断）

    arXiv:2611.27214v1 Announce Type: cross  Abstract: Latent text-to-speech (TTS) systems generate codec latents decoded by a frozen decoder, but good reconstruction does not ensure geometry suitable for Flow generation from isotropic Gaussian noise. We present Embedded Acoustic Flow TTS (EAF-TTS), an acoustic-latent adaptation of ELF with phone- and duration-conditioned generation, shared-network terminal correction, and frozen WaveVAE decoding. EAF-TTS supports a fixed ZCA interface estimated from training data and adding no learned parameters. The codec, conditioning modules, Flow backbone, and 20-call inference are shared. The intervention is the full ZCA coordinate protocol---training and sampling coordinates, induced loss metric, and terminal perturbation---not an inference-only transform. In single-speaker LJSpeech, the 64-D training-latent covariance has condition number 1355.44. On text-disjoint Val399 strict-GT with fixed E100 raw checkpoints across three matched seeds, ZCA impr
    
[^116]: 面向个性化长文本生成的显式风格抽取

    Explicit Style Extraction for Personalized Long-Form Text Generation

    [https://arxiv.org/abs/2611.27211](https://arxiv.org/abs/2611.27211)

    该论文提出StylePrism风格分类体系（28个风格维度）与SP-Refine推理时方法，将风格抽取变成显式步骤，在LongLaMP三个长文本任务上以平均10.2%的提升显著超越检索增强基线。

    

    个性化长文本生成要求语言模型以特定用户的方式来撰写新文本。标准方法是检索增强生成，即将用户的部分过往文本放入提示中，但让模型自行决定从中借鉴什么。我们提出了StylePrism，这是一个包含七大类别、共28个风格维度的分类体系，同时提出SP-Refine，一种在推理阶段将风格抽取作为显式步骤的方法：模型首先用用户的写作模式、引用证据和需要避免的习惯来填充StylePrism，然后基于用户的样本起草文本，并在自身风格评分的引导下，依据该风格画像对草稿进行两次修订。在使用三个大语言模型的LongLaMP三个长文本任务上，SP-Refine在27项ROUGE和METEOR对比中的25项上超越了LongLaMP的检索增强基线，平均提升10.2%，最高提升达35.8%。用提取的风格画像对用户自己的文本进行标注表明，StylePrism捕捉到了具体的、可核查的……

    arXiv:2611.27211v1 Announce Type: cross  Abstract: Personalized long-form generation asks a language model to write new text the way a specific user would. The standard recipe, retrieval-augmented generation, places a few of the user's past texts in the prompt but leaves the model to decide what to take from them. We introduce StylePrism, a taxonomy of 28 stylistic aspects in seven categories, and SP-Refine, an inference-time method that makes style extraction an explicit step: the model first fills StylePrism with the user's patterns, quoted evidence, and habits to avoid, then drafts from the user's samples and revises the draft twice against this profile, guided by its own style scores. On the three long-form tasks of LongLaMP with three LLMs, SP-Refine outperforms LongLaMP's retrieval-augmented baseline in 25 of 27 ROUGE and METEOR comparisons, by 10.2% on average and by up to 35.8%. Annotating users' own texts with the extracted profiles shows that StylePrism captures concrete, che
    
[^117]: 面向手写数学表达式识别的 Qwen3-VL 参数高效适配

    Parameter-Efficient Adaptation of Qwen3-VL for Handwritten Mathematical Expression Recognition

    [https://arxiv.org/abs/2611.27208](https://arxiv.org/abs/2611.27208)

    该研究通过 4 比特量化加秩-16 LoRA 微调，将紧凑的 Qwen3-VL-4B 高效适配为手写数学表达式识别模型，在 CROHME 测试集上以极小的参数开销实现了 43.74% 的精确匹配率，远超传统 ViT-Transformer 基线。

    

    手写数学表达式识别（HMER）需要同时恢复符号身份与二维语法结构，因此局部看似合理的转录结果在整体结构上仍可能是错误的。我们研究在有限计算资源下，能否将一个紧凑的视觉语言模型适配用于生成线性化的符号标签图（SLG）。研究从一个 4 比特量化的 Qwen3-VL-4B 检查点出发，对其视觉、语言、注意力及 MLP 模块应用秩为 16 的低秩适配（LoRA），并使用指令格式化的 CROHME 2019 样例进行训练。我们将该模型与内部实现的 ViT-Transformer 基线、零样本 Qwen3-VL 配置以及其他内部评估的替代方案进行比较。在 1,198 个有效的 CROHME 测试样例上，仅使用 CROHME 训练的模型获得了 524 个精确匹配（43.74%）和 77.67 的语料库 BLEU-4 分数，而任务专用的 ViT 基线仅为 4.92% 和 43.26。一项增加 2,000 个手写样本的消融实验……（原文摘要在此处截断）

    arXiv:2611.27208v1 Announce Type: cross  Abstract: Handwritten mathematical expression recognition (HMER) requires joint recovery of symbol identity and two-dimensional syntax, so a locally plausible transcription may still be structurally wrong. We study whether a compact vision-language model can be adapted to generate linearized Symbol Label Graphs (SLGs) under limited computing resources. Starting from a 4-bit Qwen3-VL-4B checkpoint, we apply rank-16 Low-Rank Adaptation to the vision, language, attention, and MLP modules and train with instruction-formatted CROHME 2019 examples. We compare this model with an internally implemented ViT-Transformer baseline, a zero-shot Qwen3-VL configuration, and additional internally evaluated alternatives. On the 1,198 valid CROHME test examples, the CROHME-only model obtains 524 exact matches (43.74%) and a corpus BLEU-4 score of 77.67, compared with 4.92% and 43.26 for the task-specific ViT baseline. An ablation that adds 2,000 manually written,
    
[^118]: LingSante-Pulse：通过随机特征预测与跨模态对齐实现的统一生理信号预训练

    LingSante-Pulse: Unified Physiological Pretraining through Random-Feature Prediction and Cross-Modal Alignment

    [https://arxiv.org/abs/2611.27206](https://arxiv.org/abs/2611.27206)

    该论文提出统一生理预训练框架LingSante-Pulse，通过随机特征预测、跨模态对齐与掩码波形重建学习通用生理信号表示，在涵盖PPG、ECG、EEG和EMG的38个任务配置的每个领域均取得最佳排名。

    

    临床监护与可穿戴设备采集互补的生理信号，但其模态覆盖部分重叠，导致信号内信息与跨模态关系难以被联合利用。我们提出了LingSante-Pulse，一个统一、通用且灵活的生理信号表示框架，可从这些异构记录中学习。该框架采用一个共享的、以信号为条件的编码器，独立处理每个物理通道，并支持不同数量和组合的信号，从而兼顾多模态评估与单传感器推理。成对跨模态对齐、结构化随机特征预测（RFP）与掩码波形重建三者协同工作，共同学习生理关系并保留具有信息量的信号内结构。在涵盖PPG、ECG、EEG和EMG的38个通用生理任务配置中，冻结编码器在每个领域均取得了按数据集平衡的最佳平均排名。

    arXiv:2611.27206v1 Announce Type: cross  Abstract: Clinical monitoring and wearable devices capture complementary physiological signals, but their partially overlapping modality coverage makes within-signal information and cross-modal relationships difficult to exploit jointly. We introduce LingSante-Pulse, a unified, general-purpose and flexible physiological representation framework that learns from these heterogeneous recordings. A shared, signal-conditioned encoder processes each physical channel independently and supports different numbers and combinations of signals for multimodal assessment and single-sensor inference. Pairwise cross-modal alignment, structured random-feature prediction (RFP) and masked waveform reconstruction jointly learn physiological relationships and preserve informative within-signal structure. Across 38 general physiological task configurations spanning PPG, ECG, EEG and EMG, the frozen encoder achieves the best dataset-balanced mean rank in each domain. 
    
[^119]: DepthMSA：面向多模态情感分析的组合式训练与预算感知深度分配

    DepthMSA: Combinatorial Training and Budget-Aware Depth Allocation for Multimodal Sentiment Analysis

    [https://arxiv.org/abs/2611.27198](https://arxiv.org/abs/2611.27198)

    该论文提出 DepthMSA 框架，通过组合式随时训练与预算感知的深度分配，在单个多模态情感分析模型中为不同模态灵活分配异构编码深度，从而在有限计算预算下保持预测性能。

    

    最先进的多模态情感分析（MSA）模型依赖于大型预训练编码器，但通常将每种模态都处理到其全部深度，而不考虑其计算成本或预测贡献。这引出了一个根本性问题：应如何在各模态之间分配有限的计算资源以保持预测性能？我们揭示了模态编码器的计算成本与对其进行更深处理所带来的预测收益之间存在显著的不匹配，这表明不同模态需要不同的编码深度。然而，启用异构深度带来了一个关键挑战：从不同深度提取的表示仍然必须能够被有效融合。为解决这一问题，我们提出了 DepthMSA，一个预算感知的框架，可在单个 MSA 模型中支持灵活的模态深度。具体而言，我们引入了组合式随时训练（combinatorial anytime training），它对融合头进行监督（摘要在此处被截断）。

    arXiv:2611.27198v1 Announce Type: cross  Abstract: State-of-the-art multimodal sentiment analysis (MSA) models rely on large pretrained encoders, yet typically process every modality to its full depth, regardless of its computational cost or predictive contribution. This raises a fundamental question: how should limited computation be allocated across modalities to preserve predictive performance? We reveal a substantial mismatch between the computational cost of modality encoders and the predictive benefit of processing them more deeply, suggesting that different modalities require different encoding depths. However, enabling heterogeneous depths introduces a key challenge: representations extracted at different depths must still be effectively fused. To address this, we propose $\textbf{DepthMSA}$, a budget-aware framework that supports flexible modality depths within a single MSA model. Specifically, we introduce combinatorial anytime training, which supervises the fusion head acros
    
[^120]: EDAGoat：一个面向项目级GPU优化的EDA感知智能体

    EDAGoat: An EDA-Aware Agent for Project-Level GPU Optimization

    [https://arxiv.org/abs/2611.27196](https://arxiv.org/abs/2611.27196)

    提出了首个EDA感知的GPU优化智能体EDAGoat，通过四种跨项目GPU优化模式指导代码编辑、并以强制验证模式确保修改满足EDA正确性与设计质量约束，实现了项目级GPU EDA工具优化的自动化。

    

    GPU加速已经变革了电子设计自动化（EDA）领域，然而要进一步改进专家手工构建的GPU工具仍需要大量的人工投入。实现这项工作的自动化，需要能够跨主机端与设备端代码追踪性能瓶颈、同时保持正确性和设计质量的编码智能体。我们提出了EDAGoat，据我们所知，这是首个面向该任务的EDA感知GPU优化智能体。其优化增强（OE）模式利用四种跨项目的GPU优化模式来指导代码编辑。其强制验证（MV）模式要求智能体针对其修改可能影响的EDA约束对编辑后的程序进行测试。我们的基准测试涵盖五个GPU EDA项目、四个任务族和55个电路输入，测量完整求解运行时间并验证生成的设计。每种配置在每个项目上运行三次独立搜索。只有当求解器通过正确性和设计质量检查时，该次搜索才被认定为合格。

    arXiv:2611.27196v1 Announce Type: new  Abstract: GPU acceleration has transformed electronic design automation (EDA), yet further improving expert-engineered GPU tools still demands substantial manual effort. Automating this work calls for coding agents that can trace bottlenecks across host and device code while preserving correctness and design quality. We introduce EDAGoat, to our knowledge the first EDA-aware GPU optimization agent for this task. Its optimization enhancement (OE) mode guides edits with four cross-project GPU optimization patterns. Its mandatory validation (MV) mode requires the agent to test the edited program against the EDA constraints its changes could affect. Our benchmark covers five GPU EDA projects across four task families and 55 circuit inputs, measuring complete-solve runtime and verifying the resulting designs. Each configuration runs three independent searches per project. A search qualifies only if its solver passes correctness and design-quality check
    
[^121]: 面向印度的节俭人工智能框架：负责任且规模适度的AI

    A Frugal AI Framework for India: Responsible and Right-sized AI

    [https://arxiv.org/abs/2611.27189](https://arxiv.org/abs/2611.27189)

    本文提出“负责任节俭AI”框架，主张印度既不应照搬中美重资本的基础模型路线，也不应仅作外国AI的下游采用者，而应先明确功能需求并判断是否需要AI，再选择以最少资源满足需求的最小充分“规模适度”模型，同时保持鲁棒性与负责任治理。

    

    印度关于人工智能的争论常常在两个并不充分的选择之间摇摆：要么复制美国或中国那种计算密集、资本密集的基础模型计划，要么停留在作为外国人工智能的下游采用者，仅提供本地化的界面与安全保障。这两者都无法满足一个需要在基础设施不均衡、预算有限的条件下，向极其庞大且多样化的人口提供人工智能服务的国家。这份来自印度理工学院马德拉斯分校负责任人工智能中心（CeRAI）的白皮书提出了第三条路径：负责任节俭AI（Responsible Frugal AI），它将基于原则的节俭工程与负责任AI相结合。负责任节俭AI是“规模适度”的AI：它首先明确功能需求，追问是否真的需要AI，然后选择最小的充分模型，以最少的资源满足需求，同时保持经过验证的余量、鲁棒性和负责任的治理。该框架在四个维度上应用节俭原则（数据、计算……（摘要原文在此处截断））。

    arXiv:2611.27189v1 Announce Type: cross  Abstract: India's AI debate often swings between two insufficient options: replicating the compute-intensive, capital-heavy foundation-model programmes of the United States or China, or remaining a downstream adopter of foreign AI with localised interfaces and safeguards. Neither serves a country that must deliver AI to a very large, diverse population across uneven infrastructure and within constrained budgets. This white paper from the Centre for Responsible AI (CeRAI), IIT Madras, proposes a third path: Responsible Frugal AI, which applies principle-based frugal engineering alongside Responsible AI. Responsible Frugal AI is right-sized AI: it fixes the functional requirement first, asks whether AI is needed at all, and then chooses the smallest sufficient model, meeting the requirement with minimal resources while keeping a verified margin, robustness and responsible governance. The framework applies frugality across four dimensions (data, co
    
[^122]: 匹配人类干预效应能确立什么？LLM模拟的基准检验

    What Does Matching a Human Intervention Effect Establish? Benchmark Checks for LLM Simulations

    [https://arxiv.org/abs/2611.27186](https://arxiv.org/abs/2611.27186)

    该论文质疑LLM模拟匹配人类干预效应所能确立的有效性，发现大量“通过”实为无信息量的零参考通过，并提出零参考基准与计划精度检验来严格评估LLM模拟的真实响应能力。

    

    LLM模拟通过与人效应的等效性来验证，然而只要人类90%区间位于容差范围内，零参考点就会自动通过——正如人类基准在模拟之前所显示的那样。当零参考点失败时，计划的精度赋予独立模拟一个容差上限，在此范围内通过构成对响应的5%显著性检验。在一项包含70个实验的探索性再分析中，GPT-4的±0.20 SD通过中有72%属于零参考通过；GPT-4通过了16个零参考未通过的对比（在无响应情况下预期仅为1.58个）。在一项预注册审计中，Claude Sonnet 5.5通过了三个意图目标，全部为零参考通过，同时还通过了注册的信念梯度控制；规范信念仅变化0.065 SD，而人类为0.351 SD（零参考检验在揭盲后添加；规范信念在检查后予以强调）。注册检验未检测到高估；平均效应区间排除了超过0.08 SD的高估。

    arXiv:2611.27186v1 Announce Type: new  Abstract: LLM simulations are validated by equivalence with human effects, yet a zero reference passes whenever the human 90 percent interval lies inside the margin, as the human benchmark shows before simulation. When the zero reference fails, planned precision gives independent simulations the margin up to which passing is a 5 percent test of response. In an exploratory 70-experiment reanalysis, 72 percent of GPT-4's $\pm$0.20 SD passes were zero-reference passes; GPT-4 passed 16 contrasts the zero reference failed (1.58 expected without response). In a preregistered audit, Claude Sonnet 5.5 passed three intention targets, all zero-reference passes, and the registered belief-gradient control; norm beliefs changed 0.065 SD against humans' 0.351 (zero-reference check added after unblinding; norm beliefs emphasized after inspection). Registered tests detected no overestimation; average-effect intervals excluded overestimation above 0.08 SD. A group
    
[^123]: GEG-SAM：面向弱监督息肉分割的注视探索引导的分割一切模型

    GEG-SAM: Gaze Exploration-Guided Segment Anything Model for Weakly Supervised Polyp Segmentation

    [https://arxiv.org/abs/2611.27183](https://arxiv.org/abs/2611.27183)

    该论文提出GEG-SAM框架，通过显式建模内镜医生注视点的时空结构关系及其所关注区域的视觉语义，实现弱监督的息肉分割。

    

    内镜医生的注视提供了一种在常规临床检查中不引人注意地获得的、以病灶为导向的视觉探索记录，能够为相关任务提供监督信息。然而，现有方法将注视点视为与目标直接关联的孤立空间提示，忽视了自然注视记录的是一个连续的视觉搜索过程，其中注视点在时间上相互连接、在空间上组织有序，并与视觉内容相互耦合。仅基于局部视觉内容独立解释每个注视点会忽视注视轨迹的时间演变和空间组织，而仅对注视点之间的关系进行建模则无法揭示其所关注区域的视觉语义。为此，我们提出了注视探索引导的分割一切模型（GEG-SAM），这是一个具有原则性的弱监督框架，旨在显式利用结构化的注视关系及其相应的视觉内容。

    arXiv:2611.27183v1 Announce Type: cross  Abstract: Endoscopist gaze offers an unobtrusively acquired record of lesion-oriented visual exploration during routine clinical inspection, providing relevant supervision for tasks. However, existing methods treat fixations as isolated spatial prompts directly associated with the target, overlooking the fact that natural gaze records a continuous visual search process in which fixations are temporally connected, spatially organized, and coupled with the visual content. Interpreting each fixation independently based only on its local visual content overlooks the temporal evolution and spatial organization of the gaze trajectory, whereas modeling relations among fixations alone cannot reveal the visual semantics of the regions they attend to. To this end, we propose the Gaze Exploration-Guided Segment Anything Model (GEG-SAM), a principled weakly supervised framework designed to explicitly exploit structured gaze relations and their corresponding
    
[^124]: 学习自适应防御策略：面向网络移动性下网络安全博弈的微调大语言模型

    Learning Adaptive Defense Strategies: Fine-Tuned LLMs for Network Security Games under Network Mobility

    [https://arxiv.org/abs/2611.27180](https://arxiv.org/abs/2611.27180)

    本研究构建了带有求解器均衡标签的270个蜜罐分配安全博弈数据集，首次系统评估了LoRA微调的大语言模型能否在网络移动性持续改变攻击图的动态场景下，充当无需求解器的在线博弈均衡计算替代方案。

    

    大语言模型（LLM）在战略推理方面展现出潜力，但其在动态条件下计算博弈论均衡的能力在很大程度上仍未被探索。本研究探讨了微调后的LLM能否作为无需求解器的替代方案，用于蜜罐分配场景中的在线均衡计算——在该场景下，网络移动性会持续重塑底层攻击图，迫使双人零和博弈不断重新求解。我们构建了一个包含270个安全博弈的数据集（覆盖8至25个节点的网络，并带有基于求解器计算得到的均衡标签），并在1,524个留出的移动性事件上评估了四类预测器：非学习型基线、六个开源权重架构的零样本表现、经过LoRA微调后的同六个模型，以及作为外部参考的四个专有模型。所有预测均无需事后修复，直接依据精确的博弈论标准进行评分：即面对最优攻击时的可利用性……

    arXiv:2611.27180v1 Announce Type: new  Abstract: Large language models(LLM) show promise for strategic reasoning, yet their ability to compute game-theoretic equilibria under dynamic conditions remains largely unexplored. This work investigates whether fine-tuned LLMs can serve as solver-free surrogates for online equilibrium computation in honeypot allocation, a setting in which network mobility continuously reshapes the underlying attack graph and forces repeated re-solving of a two-player zero-sum game. We construct a dataset of 270 security games on networks of 8--25 nodes with solver-grounded equilibrium labels, and evaluate four classes of predictor over 1{,}524 held-out mobility events: non-learned baselines, six open-weight architectures evaluated zero-shot, the same six after LoRA fine-tuning, and four proprietary models as an external reference point. Predictions are scored without post-hoc repair against exact game-theoretic criteria: exploitability against an optimal attack
    
[^125]: Felid：一种面向加密数据上Transformer微调的灵活高效设计

    Felid: A Flexible and Efficient Design for Transformer Fine-Tuning over Encrypted Data

    [https://arxiv.org/abs/2611.27175](https://arxiv.org/abs/2611.27175)

    该论文提出Felid，一种在加密数据上进行Transformer微调的灵活高效设计，通过新颖的打包策略、面向特定操作的矩阵求值算法以及函数自举技术，从密文操作层面显著减少了密文旋转、重线性化和自举的开销。

    

    全同态加密（FHE）能够实现非交互式、隐私保护的Transformer微调，保护作为大语言模型输入的高度敏感数据。以往的工作主要通过架构改进来追求效率提升，但整体性能仍受限于加密计算固有的高昂成本。为解决这些问题，我们将优化目标瞄准在密文操作层面的瓶颈。具体而言，我们提出了一种新颖的打包策略，支持对灵活数量的输入进行批处理，使其非常适合微调工作负载。此外，我们设计了针对特定操作的矩阵求值算法，显著减少了（i）密文旋转和重线性化的次数，以及（ii）自举调用的次数。另外，我们利用函数自举技术，将逐元素激活函数的求值直接折叠到自举过程中。

    arXiv:2611.27175v1 Announce Type: cross  Abstract: Fully Homomorphic Encryption (FHE) enables non-interactive, privacy-preserving transformer fine-tuning, protecting highly sensitive data as inputs to large language models. Prior work has predominantly pursued efficiency gains through architectural refinements, yet overall performance remains bounded by the substantial cost inherent in encrypted computation. To address these issues, we target the bottleneck at the ciphertext operation level. Concretely, we propose a novel packing strategy that supports batching over a flexible number of inputs, making it well-suited for fine-tuning workloads. In addition, we design operation-specific matrix evaluation algorithms that significantly reduce (i) the number of ciphertext rotations and relinearizations, and (ii) the number of bootstrapping invocations. Additionally, we exploit functional bootstrapping to fold the evaluation of element-wise activation functions directly into the bootstrapping
    
[^126]: 从行动到学习：面向持续学习智能体的经验选择

    From Acting to Learning: Experience Selection toward Continually Learning Agents

    [https://arxiv.org/abs/2611.27159](https://arxiv.org/abs/2611.27159)

    该论文将经验选择作为持续学习智能体的关键组成部分，通过大规模对照实验证明，优化器感知的梯度选择方法在学生智能体任务成功率上显著优于全局随机选择和高 NLL 选择两种基线。

    

    成功的智能体交互能够提供训练数据，但仅凭成功本身并不能决定学生（模型）应当从哪些记录中学习。我们将经验选择作为持续智能体学习的一个组成部分，针对一个 4B 参数的语言智能体展开研究。在两个固定经验池、八个初始化种子和两个采样种子的条件下，比较了九种选择器，共训练出 288 个学生模型。每个学生模型均通过统一的执行器在 128 个开发任务组上进行评估。在全局选择自由度相同、包含 256 个不同内容、共 768 次训练呈现的设置下，优化器感知的梯度选择方法在两个经验池中分别取得了 81.45% 和 73.19% 的成功率；相比全局随机选择，其提升分别为 8.25 和 5.96 个百分点；相比全局高 NLL 选择，其提升分别为 4.10 和 4.05 个百分点。在两个经验池中，全部八组初始化层面的对比均优于两个基线。放宽覆盖约束可以进一步提升梯度选择的效果，但会降低随机选择的表现（原文摘要在此处截断）。

    arXiv:2611.27159v1 Announce Type: new  Abstract: Successful agent interactions provide training data, but success alone does not determine which records a student should learn from. We examine experience selection for a 4B language agent as a component of continual agent learning. Nine selectors are compared across two fixed experience pools, eight initialization seeds, and two sampling seeds, yielding 288 trained students. Each student is evaluated on 128 development task groups with a common executor. With equal global selection freedom, 256 distinct contents, and 768 training presentations, optimizer-aware gradient selection reaches 81.45\% and 73.19\% success in the two pools. Its gains over global random selection are 8.25 and 5.96 percentage points; gains over global high-NLL selection are 4.10 and 4.05 points. All eight initialization-level contrasts are positive against both baselines in each pool. Relaxing coverage constraints improves gradient selection but reduces random-sel
    
[^127]: 面向自适应信息行动的行为智能体与检测器协同演化方法

    Co-Evolving Behavioral Agents and Detectors for Adaptive Information Operations

    [https://arxiv.org/abs/2611.27156](https://arxiv.org/abs/2611.27156)

    该论文提出IO-AI协同演化框架，通过让基于真实信息行动行为构建、具备互动与规避能力的模拟智能体与检测器轮流对抗训练，使检测器对自适应规避智能体的漏检率降低了65%。

    

    国家支持的信息行动（IO）检测器通常在平台已经采取下架措施的行动数据上进行训练。即使存档数据中没有任何操作者试图规避检测，这些检测器在面对未见过的行为者时也会损失大量准确性。真实的信息操作者有两个目标：触达受众和避免被发现。我们提出了IO-AI，一个赋予模拟智能体这两种能力——互动参与和规避检测——的框架，并测试每种能力对检测效果的影响。这些智能体基于真实行动的发帖行为构建，可以为每条帖子选择情绪、类型和发布时机。随后，智能体与检测器轮流进行相互对抗训练。我们按照检测难度对训练后的智能体进行排序，并按从易到难的顺序训练检测器。在20个Twitter信息行动上，该检测器相比仅在存档行动数据上训练的检测器，将对自适应智能体的漏检率降低了65%……

    arXiv:2611.27156v1 Announce Type: cross  Abstract: Detectors of state-sponsored information operations (IOs) are trained on campaigns that platforms have already taken down. They lose much of their accuracy on an actor they have not seen, even though no operator in the archived data is trying to avoid them. A real operator wants two things, to reach an audience and to avoid being caught. We present IO-AI, a framework that gives simulated agents these two capabilities, engagement and evasion, and tests what each one does to detection. The agents are built from the posting behavior of real campaigns and choose the emotion, type, and timing of each post. An agent and a detector then train against each other in turn. We rank the trained agents by how hard they are to detect and train one detector on them from easiest to hardest. On 20 Twitter IO campaigns, this detector reduces missed detections of adaptive agents by 65% relative to a detector trained only on the archived campaigns, at a s
    
[^128]: 结构如何进入世界模型：从预测拟合到物理推演

    Where Structure Enters a World Model: From Predictive Fit to Physical Rollout

    [https://arxiv.org/abs/2611.27149](https://arxiv.org/abs/2611.27149)

    该论文系统研究了如何将物理结构（守恒量监督、对称性感知表示、基于能量的动力学头等）融入预测性世界模型，证明这些物理约束能显著提升模型在长时间范围内自主推演的物理一致性和轨迹忠实度。

    

    预测性世界模型学习对未来观测进行预测，但准确的短期预测并不一定能在更长的时间范围内产生物理上一致的行为。我们研究了如何将物理结构融入世界模型以增强其能力，同时保留其底层的预测目标。在基于token的行星运动模型和JEPA风格的机械系统视觉模型上，我们研究了守恒量监督、对称性感知表示、时间一致性以及学习到的结构化动力学。在行星运动任务中，角动量监督能够改进自主的下一token预测，而时间正则化所带来的收益则取决于与之配合的物理目标和表示方式。在视觉模型中，通过有监督的基于能量的动力学头来生成推演，能够产生更忠实的轨迹和相空间演化。这些发现共同凸显了物理结构以何种方式进入世界模型的重要性。

    arXiv:2611.27149v1 Announce Type: new  Abstract: Predictive world models learn to forecast observations, but accurate short-term predictions do not necessarily produce physically coherent behavior over longer horizons. We investigate how incorporating physical structure can strengthen world models while retaining their underlying predictive objectives. Across token-based models of planetary motion and JEPA-style visual models of mechanical systems, we study conserved-quantity supervision, symmetry-aware representations, temporal consistency, and learned structured dynamics. In planetary motion, angular-momentum supervision improves autonomous next-token forecasting, while the benefits of temporal regularization depend on the accompanying physical objective and representation. In visual models, generating rollouts through supervised energy-based dynamics heads produces more faithful trajectories and phase-space evolution. Together, these findings highlight the importance of how physical
    
[^129]: 结构重于规模：重新思考可验证奖励强化学习中的适配方法

    Structure Over Scale: Rethinking Adaptation for Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2611.27148](https://arxiv.org/abs/2611.27148)

    该研究发现在可验证奖励强化学习训练中，LoRA会随秩增大出现性能骤降的“秩坍缩”现象，而DoRA和QuanTA等结构化适配方法凭借更优的更新几何结构，仅更新不到0.6%的参数即可达到与全参数微调相当的效果，表明更新的结构比规模更关键。

    

    可验证奖励强化学习通常被框架化为在全参数微调的表达能力与低秩适配的高效率之间做选择。我们发现这一框架忽略了一个重要变量：更新的几何结构。在常见的冷启动GRPO协议下，LoRA表现出“秩坍缩”现象：性能在低秩时大致保持稳定，但随着秩的增大却会突然失效。例如在DeepMath-Hard数据集上，准确率从r=64时的73.1%骤降至r=128时的4.7%。相比之下，权重分解低秩适配和量子启发张量适配在更广的测试优化区域内保持稳定，并且在仅更新不到0.6%模型参数的情况下与全参数微调表现相当。谱分析揭示了所学更新结构的相应差异：发生坍缩的LoRA将更新能量集中在相对较少的有效方向上，而结构化适配器……

    arXiv:2611.27148v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) is often framed as a choice between the expressivity of Full Fine-Tuning (FFT) and the efficiency of Low-Rank Adaptation (LoRA). We find that this framing misses an important variable: the geometry of the update. Under a common cold-start GRPO protocol, LoRA exhibits rank collapse: performance remains broadly stable at low rank, then fails abruptly as rank increases. On DeepMath-Hard, for example, accuracy drops from 73.1% at r=64 to 4.7% at r=128. In contrast, Weight-Decomposed Low-Rank Adaptation (DoRA) and Quantum-Informed Tensor Adaptation (QuanTA) remain stable over a wider tested optimization region and are competitive with FFT while updating less than 0.6% of the model parameters. Spectral analysis reveals a corresponding difference in learned update structure: collapsed LoRA concentrates its update energy into relatively few effective directions, whereas structured adapters 
    
[^130]: SAM-3 如何适应？容量、更新几何与领域迁移

    How Does SAM-3 Adapt? Capacity, Update Geometry, and Domain Transfer

    [https://arxiv.org/abs/2611.27142](https://arxiv.org/abs/2611.27142)

    本研究系统比较了五种微调方法在五个分割领域的表现，发现参数高效微调可媲美全量微调，且其性能由更新谱几何特性而非参数数量或秩决定。

    

    提示式分割基础模型正日益被应用于专业领域，但在将 SAM-3 适应到其原始训练分布之外时，应如何进行微调仍不清楚。我们通过比较五种适应方法家族——LoRA、HiRA、KRAdapter、QuanTA 和全量微调——在五个跨越医学影像、遥感和密集材料分割的分割数据集上研究这一问题。在这些数据集上，紧凑的参数高效微调方法在 DSC 和 IoU 指标上往往能匹配或接近全量微调，这表明强大的领域内适应并不总是需要更新整个模型。然而，这种性能无法仅凭名义上的适配器秩或可训练参数数量来解释。为了探究为什么不同方法表现相似，我们分析了它们的更新谱及其在 LoveDA Urban/Rural 迁移下的行为。我们的 SVD 分析表明，全量微调的更新……

    arXiv:2611.27142v1 Announce Type: cross  Abstract: Promptable segmentation foundation models are increasingly used in specialized domains, but it remains unclear how SAM-3 should be fine-tuned when adapting it beyond its original training distribution. We study this question by comparing five adaptation families --LoRA, HiRA, KRAdapter, QuanTA, and Full Fine-Tuning-- across five segmentation datasets spanning medical imaging, remote sensing, and dense material segmentation. Across these datasets, compact parameter-efficient fine-tuning methods often match or approach Full Fine-Tuning in DSC and IoU, suggesting that strong in-domain adaptation does not always require updating the full model. However, this performance cannot be explained by nominal adapter rank or trainable parameter count alone. To investigate why different methods perform similarly, we analyze their update spectra and their behavior under LoveDA Urban/Rural transfer. Our SVD analysis shows that Full Fine-Tuning updates
    
[^131]: 高级智能体尽到职责了吗？衡量LLM智能体使用下级材料时的角色完整性

    Do Senior Agents Do Their Job? Measuring Role Integrity in How LLM Agents Use Subordinate Material

    [https://arxiv.org/abs/2611.27139](https://arxiv.org/abs/2611.27139)

    该论文提出“角色完整性”概念并构建RI-Bench行为基准，借助同行评审、上诉裁决和议会调查三种制度层级的结构与公开记录，衡量多智能体LLM系统中高级智能体是否真正履行了权衡下级报告的职责，而非仅仅进行简单汇总。

    

    多智能体LLM系统日益将长任务分配给多个子智能体，再由高级智能体将其报告整合为最终答案。这种分工只有在每个智能体——尤其是高级智能体——履行其角色职责时才能奏效。然而，当前的评估大多只对最终答案进行打分，而一个流畅、引用规范的答案无法表明高级智能体是否真正权衡了其子智能体的报告，还是仅仅进行了简单汇总。为了评估高级智能体是否履行了其被赋予的职责，我们将“角色完整性”定义为智能体在处理其下级材料时对其功能职责的保持。为了衡量这一概念，我们提出了RI-Bench，这是一个基于三种制度层级的结构与公开记录的行为基准：同行评审、上诉裁决和议会调查。我们选择这些层级，是因为它们的规则明确赋予高级角色对下级材料的职责。

    arXiv:2611.27139v1 Announce Type: cross  Abstract: Multi-agent LLM systems increasingly divide long tasks among subagents whose reports a senior agent turns into the final answer. Such a division of labor pays off only if each agent, and above all the senior agent, carries out the duties of its role. Current evaluations, however, mostly score the final answer, and a fluent, well-cited answer cannot show whether the senior agent weighed its subagents' reports or merely aggregated them. To assess whether a senior agent meets its assigned responsibilities, we define role integrity as the preservation of an agent's functional responsibilities when it acts on its subordinates' material. To measure it, we introduce RI-Bench, a behavioral benchmark grounded in the structures and public records of three institutional hierarchies: peer review, appellate adjudication, and parliamentary inquiry. We choose these hierarchies because their rules explicitly assign the senior role a duty toward subord
    
[^132]: 大语言模型证据融合中的重复计数：精确后验将规则遵从性与后验增益区分开来

    Double counting in LLM evidence fusion: exact posteriors separate rule compliance from posterior gain

    [https://arxiv.org/abs/2611.27138](https://arxiv.org/abs/2611.27138)

    本文构建了分级依赖条件下的精确后验参照基准，揭示出大语言模型对“每个来源只计一次”指令的遵从性与实际后验精度增益相互独立——完全遵守规则的模型可能损失精度，而不遵守规则的模型反而可能获得增益。

    

    语言模型在检索、多智能体共识和智能体记忆等场景中融合多方报告，若将来自共享来源的报告视为相互独立，就会导致证据被重复计数。一种常见的补救措施是指示模型对每个来源只计数一次。然而这一指令的价值取决于接收者的原始基线水平，要为其定价就需要一个精确的参照基准。本文构建了分级依赖性的精确参照：由单报告可靠性为0.65的二值世界和四个依赖级别构成，其公开的参数为每个世界提供了精确后验，并为每条规则设定了目标风险。配对增益协议在2,976个世界上对四种模型配置的原始回答与受指令回答进行了评分。结果显示，与规则的一致性和增益在两个方向上相互分离。在δ=0时，claude-opus-5-5和gpt-6.1-sol在全部744个世界中与规则一致（误差在0.001以内），但该指令反而使其后验失配增加了0.044；gpt-6-luna在744个世界中仅有165个与规则一致，却获得了0.0140的增益（95%置信区间为0.0064至0.0219）；claude-haiku-5-5的增益变化……（摘要截断）

    arXiv:2611.27138v1 Announce Type: new  Abstract: Language models fuse reports in retrieval, multi-agent consensus and agent memory, where treating shared-source reports as independent can double count evidence. A common remedy is an instruction to count each source once. Its value depends on the receiver's raw baseline. Pricing it needs an exact reference. We build the graded-dependence exact reference: binary worlds with single-report reliability 0.65 and four dependence levels, whose disclosed parameters give every world an exact posterior and every rule a target risk. The paired gain protocol scores raw and instructed answers of four configurations on 2,976 worlds. Agreement and gain separate in both directions. At $\delta$ = 0, claude-opus-5-5 and gpt-6.1-sol agreed with the rule within 0.001 in 744 of 744, and the instruction raised their posterior mismatch by 0.044; gpt-6-luna agreed in 165 of 744 and gained 0.0140 (95% interval 0.0064 to 0.0219); claude-haiku-5-5's gain moved fr
    
[^133]: MAPES：通过专家专业化实现的多价值对齐流水线

    MAPES: Multi-Value Alignment Pipeline via Expert Specialization

    [https://arxiv.org/abs/2611.27133](https://arxiv.org/abs/2611.27133)

    MAPES提出了一种多价值对齐新范式，通过将监督分解为隔离参数子空间中训练的价值观专用专家，再经偏好感知合并与轻量级协调重组，在流水线每个阶段将冲突解决作为首要关注点，从而在多个基准上超越DPO变体和多目标基线。

    

    将大型语言模型（LLM）与人类偏好对齐需要同时满足多个往往相互竞争的价值观。现有的偏好优化方法将这些价值观压缩为单一隐式信号，即使是多目标扩展方法也在共享参数空间内进行优化，未能解决跨价值观的冲突。我们提出MAPES，这是一种多价值对齐范式，将冲突解决作为流水线每个阶段的首要关注点：它首先将监督信号分解为在隔离参数子空间中训练的价值观专用专家，然后通过偏好感知的价值合并以及一个轻量级协调阶段（使合并后的模型与专家集成保持一致）将其重组。在五个偏好对齐基准测试中，MAPES在基准级别平均值上优于强大的直接偏好优化（DPO）变体和多目标基线方法，且在GSM8K和GPQA上没有出现大幅性能下降。

    arXiv:2611.27133v1 Announce Type: new  Abstract: Aligning large language models (LLMs) with human preferences requires simultaneously satisfying multiple, often competing values. Existing preference optimization methods collapse these into a single implicit signal, and even multi-objective extensions optimize within a shared parameter space, leaving cross-value conflicts unresolved. We propose MAPES, a multi-value alignment paradigm in which conflict resolution is a first-class concern at every pipeline stage: it first decomposes supervision into value-specific experts trained in isolated parameter subspaces, then recomposes them via preference-aware value merging and a lightweight harmonization stage that aligns the merged model with the expert ensemble. Across five preference alignment benchmarks, MAPES outperforms strong direct preference optimization (DPO) variants and multi-objective baselines on the benchmark-level average, with no large degradation on GSM8K and GPQA; matched con
    
[^134]: 当学生无法从教师处学习时：在线策略蒸馏的学习动力学

    When Students Fail to Learn from Teachers: Learning Dynamics of On-Policy Distillation

    [https://arxiv.org/abs/2611.27125](https://arxiv.org/abs/2611.27125)

    该论文从学习动力学视角揭示了在线策略蒸馏（OPD）训练不稳定（如重复生成、响应过长）背后的失败机制，识别出OPD更新无法纠正师生失配的关键状态，并用学生在教师top-p集合上的概率质量对其进行刻画。

    

    在线策略蒸馏（OPD）已成为大语言模型后训练的核心技术，使学生模型能够在自身生成的序列上从密集的词元级监督中学习。然而，即使教师模型能力强大，OPD训练仍可能不稳定，导致响应过长、重复生成以及严重的性能退化。其中一个例子是“虚假一致性”（spurious agreement），即学生模型在所访问的状态上与教师模型保持高度重叠，但生成的结果却是重复的。尽管此类失败现象已被观察到，但其背后的机制仍鲜为人知。为弥补这一空白，我们从学习动力学的视角研究OPD，从局部分析追踪到后续在线策略采样的三种机制。我们首先识别出采样词元的OPD更新无法纠正教师-学生失配的状态，并利用学生在教师模型top-p集合上的概率质量来刻画这些状态。

    arXiv:2611.27125v1 Announce Type: new  Abstract: On-policy distillation (OPD) has become a core technique for post-training large language models, enabling students to learn from dense token-level supervision on their own generations. However, even with a capable teacher, OPD training can be unstable, leading to excessive response length, repetitive generation, and severe performance degradation. One example is \textit{spurious agreement}, where the student maintains high overlap with the teacher at the states visited but produces repetitive generations. Although such failures have been observed, the mechanisms behind these failures remain poorly understood. To address this gap, we investigate OPD through the lens of learning dynamics, tracing three mechanisms from local analysis to subsequent on-policy sampling. We first identify states where sampled-token OPD updates fail to correct teacher--student mismatch and characterize them using the student's mass on the teacher's top-$p$ set.
    
[^135]: 定时型与位置相关视觉中断下的可移植主动恢复

    Portable Active Recovery under Timed and Position-Dependent Visual Outages

    [https://arxiv.org/abs/2611.27121](https://arxiv.org/abs/2611.27121)

    提出了一个无需重训练、可移植的主动恢复模块，使无外部定位的双目视觉-惯性四旋翼无人机能够区分并正确应对定时型与位置相关两类视觉中断，通过协调检测、制动、后退探测与重连等行为实现最坏条件下的主动可观测性恢复。

    

    当相机失效取决于物理位置而非经过时间时，稳定悬停会阻碍视觉恢复与任务完成。在没有外部定位的双目视觉-惯性四旋翼无人机上，外部触发的定时中断与内生位置相关的黑暗区域在失效起始时表现完全相同，却需要截然相反的应对策略：悬停等待策略能完成93/96次定时中断恢复，但对空间中断的恢复率为0/96（13次硬故障）；而盲目继续飞行虽能穿越许多空间区域，却在定时中断中大量失败。我们将视觉中断恢复问题形式化为最坏条件下的主动可观测性问题，并在保持主机控制器不变的前提下，在其参考接口处引入一个可移植的恢复模块，实现零重训练。仅依靠可部署的遥测数据，该模块协调视觉丢失检测、加加速度受限的制动、有界后退探测、稳定重连、高度积分微调以及可选的门控先验地图……（原文摘要在此处截断）

    arXiv:2611.27121v1 Announce Type: new  Abstract: When a camera blackout depends on physical position rather than elapsed time, stable hovering prevents visual recovery and mission completion. On stereo visual-inertial quadrotors without external localization, an exogenous timed blackout and an endogenous position-dependent dark zone appear identical at loss onset yet require opposite responses: hover-and-wait completes 93/96 timed blackouts but 0/96 spatial outages (13 hard failures), whereas blind continuation traverses many spatial zones yet fails heavily during timed blackouts. We formulate visual-outage recovery as a worst-condition active-observability problem and introduce a portable recovery module at the reference interface of an unchanged host controller with zero retraining. Using deployable telemetry alone, the module coordinates visual-loss detection, jerk-limited braking, bounded backward probing, settled rejoining, height-integral trim, and an optional gated prior-map bra
    
[^136]: CoEvoBench：大语言模型能否驱动智能体系统的递归协同进化？

    CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems?

    [https://arxiv.org/abs/2611.27116](https://arxiv.org/abs/2611.27116)

    该论文提出CoEvoBench——首个同时评估LLM作为研究者联合协同进化智能体的模型与框架（harness）的基准，在24小时预算和五个智能体领域中衡量这一此前未被测量的递归联合决策能力。

    

    LLM智能体的性能同时取决于其模型和其框架，即调停其与工具和环境交互的代码。LLM研究者能否将这种协同进化作为一个递归改进过程来推动——在一轮又一轮中决定该修改哪个组件，而每一次修改都会改变另一个组件的价值？现有基准要么在框架不在评估范围内的前提下评估训练后优化或数据研究，要么在模型固定的前提下进行框架优化，这使得这一联合决策从未被衡量过。我们提出CoEvoBench，这是一个将LLM作为研究者进行评估的基准，这些研究者在五个智能体领域中协同进化目标智能体的模型与框架。研究者从共同的基础模型和初始框架出发，在24小时预算内进行训练和修改，并提交一个模型-框架组合，该组合将在独立的测试任务上进行评分。在五个前沿研究者中，表现最强者弥合了与一个指令微调参考（系统）之间超过五分之四的差距。（注：原文摘要在此处被截断）

    arXiv:2611.27116v1 Announce Type: new  Abstract: The performance of an LLM agent depends on both its model and its harness, the code that mediates its interaction with tools and environments. Can LLM researchers drive this co-evolution as a recursive improvement process, deciding round after round which component to change when each change alters the value of the other? Existing benchmarks evaluate post-training or data research with the harness out of scope, or harness optimization with the model fixed, leaving this joint decision unmeasured. We introduce CoEvoBench, a benchmark that evaluates LLMs as researchers co-evolving the model and harness of a target agent across five agentic domains. Researchers start from a common base model and initial harness, train and revise within a 24-hour budget, and submit a model-harness pair that is scored on separate test tasks. Across five frontier researchers, the strongest closes more than four fifths of the gap to an instruction-tuned referenc
    
[^137]: 模式验证的大语言模型工作流：一项基准测试协议与六种Python实现的对比评估

    Schema-Validated LLM Workflows: A Benchmark Protocol and Comparative Evaluation Across Six Python Implementations

    [https://arxiv.org/abs/2611.27106](https://arxiv.org/abs/2611.27106)

    该论文提出了一个可复现的基准测试协议与公开可执行工件，在五个业务工作流任务上对比六种Python实现，将LLM输出的模式有效性与任务正确性分离评估，并验证了两者可分离、结构化输出绑定路径异构、且失败集中于特定条件-任务组合这三大假设。

    

    大语言模型正越来越多地被用于提取记录、生成SQL、汇总报告、回答政策问题以及分派运营工作。在这些软件边界上，输出可解析是必要条件，但并不能保证任务的正确性。我们提出了一个可复现的基准测试协议、一个公开的可执行工件，以及一项验证性的实时API评估，在五个业务工作流任务和六种Python实现条件（PydanticAI、LangGraph、DSPy、CrewAI、Agno，以及一个不依赖框架的SDK参考实现）下，将结构有效性（模式有效性）与任务特定正确性加以区分。我们提出以下假设：(H1) 模式有效性与任务质量是可分离的；(H2) 在共享模型与最小公约数契约的条件下，结构化输出的绑定路径是异构的；(H3) 条件与任务的交互效应会使失败集中于特定组合，而不是均匀地影响所有任务。验证性分析使用了各自独立的不同任务实例（100个提取任务实例……原文在此处截断）。

    arXiv:2611.27106v1 Announce Type: cross  Abstract: Large language models are increasingly used to extract records, generate SQL, summarize reports, answer policy questions, and route operational work. At these software boundaries, parseable output is necessary but does not establish task correctness. We present a reproducible benchmark protocol, a public executable artifact, and a confirmatory live API evaluation that separate structural validity from task-specific correctness across five business workflow tasks and six Python implementation conditions: PydanticAI, LangGraph, DSPy, CrewAI, Agno, and a framework-free SDK reference.   We hypothesize that (H1) schema validity and task quality are separable, (H2) structured-output binding paths are heterogeneous under a shared model and lowest-common-denominator contract, and (H3) condition-by-task interactions concentrate failures rather than shifting all tasks equally. Confirmatory analysis uses distinct task instances (100 extraction an
    
[^138]: 一种状态，众多问题：Opsis为前沿模型赋予病理分析的类型化视觉能力

    One state, many questions: Opsis gives a frontier model typed eyes for pathology

    [https://arxiv.org/abs/2611.27102](https://arxiv.org/abs/2611.27102)

    Opsis提出了一种构建在冻结病理编码器上的类型化问题接口，让前沿语言模型能够对仅编码一次的病理切片主动提问，在运行时指定算子、参数和候选答案并获得概率化结果，从而解决了语言模型无法处理大规模病理切片且固定描述性能低下的问题。

    

    综合性病理报告是一份包含固定答案集的问题清单，其中许多答案由其他答案组合而成。病理学基础模型可以通过为每个问题单独训练的带标签分类器来准确回答这些问题，而可以自由组合的前沿语言模型却无法将包含一万四千个图块的切片纳入其上下文中。在189个乳腺报告问题上，通过阅读切片文字描述或查看缩略图及部分图块来回答问题的前沿模型，准确率仅为0.50至0.66，低于多数类基线的0.67。文字描述是在问题未知之前写就的，而一次观看只能覆盖少量图块。我们提出Opsis，让模型能够主动提问：这是构建在冻结病理编码器之上的类型化问题接口。切片只需编码一次，问题在运行时指定一个算子、参数和候选答案，答案则是候选答案上的概率分布。我们将其实现为构建在冻结Virchow2令牌之上的1000万参数解码器（Opsis-S）……

    arXiv:2611.27102v1 Announce Type: cross  Abstract: A synoptic pathology report is a checklist of questions with fixed answer sets, many composed from others. Pathology foundation models answer them accurately with one classifier trained with labels for each, and frontier language models, which compose freely, cannot fit a slide of fourteen thousand tiles in their context. On 189 breast report questions, frontier models that read a written description of the slide or look at a thumbnail and tiles of it answer at 0.50 to 0.66, below the majority class at 0.67. A description is written before the question is known, and a look covers a handful of tiles. We present Opsis, which lets the model ask instead: a typed question interface on a frozen pathology encoder. A slide is encoded once, a question names an operator, arguments and candidates given at runtime, and the answer is a probability over the candidates. We build it as a 10 M parameter decoder on frozen Virchow2 tokens (Opsis-S) and i
    
[^139]: 二维推理：量化思维链的忠实性与效率

    Reasoning in Two Dimensions: Quantifying Chain-of-Thought Faithfulness and Efficiency

    [https://arxiv.org/abs/2611.27100](https://arxiv.org/abs/2611.27100)

    该论文提出将思维链表示为语义推理图，并沿“语义支持对推理历史”与“通往答案对偏离路径”两个维度分解模型的预测依赖性，在同一框架下实现了对思维链非忠实性与低效性的全局及步骤级量化。

    

    思维链（CoT）推理会展示中间推理步骤，但答案正确并不能证明大语言模型确实依赖这些步骤所表达的依赖关系，也不能证明其推理是高效的。我们在一个统一的框架内对CoT的忠实性和效率进行了可操作化定义。具体而言，我们将思维链表示为一个语义推理图，并沿两个互补的维度分解模型的预测依赖性：由语义支持所解释的依赖与额外推理历史带来的依赖，以及分配给通往答案的推理与偏离路径推理的依赖。这一分解产生了全局和步骤级别的非忠实性与低效性度量。受控干预实验表明，所提出的指标能够按预期响应隐藏的依赖关系和诱导出的偏离路径推理。在数学和多跳推理任务中，答案级别的非忠实性始终与更大的预测不确定性相关，而非忠实性和……（摘要在此处截断）

    arXiv:2611.27100v1 Announce Type: new  Abstract: Chain-of-thought (CoT) reasoning exposes intermediate steps, but a correct answer does not establish that an LLM relies on the dependence expressed by those steps or reasons efficiently. We operationalize CoT faithfulness and efficiency within a common framework. Specifically, we represent a CoT as a semantic reasoning graph and decompose the model's predictive dependence along two complementary axes: dependence explained by semantic support versus additional reasoning history, and dependence allocated to answer-reaching versus off-path reasoning. This decomposition yields global and step-level measures of unfaithfulness and inefficiency. Controlled interventions show that the proposed metrics respond to hidden dependence and induced off-path reasoning as intended. Across mathematical and multi-hop reasoning tasks, answer-level unfaithfulness is consistently associated with greater predictive uncertainty, whereas both unfaithfulness and 
    
[^140]: Epistemic Guard（认知守护）：面向工具使用型自主智能体的进程外MCP回执代理、回合后文本监督与来源门控记忆

    Epistemic Guard: Out-of-Process MCP Receipt Proxying, Post-Turn Prose Supervision, and Provenance-Gated Memory for Tool-Using Autonomous Agents

    [https://arxiv.org/abs/2611.27095](https://arxiv.org/abs/2611.27095)

    提出了开源库epistemic-guard，通过进程外MCP回执代理、回合后文本监督与来源门控记忆三大机制，在零GPU显存开销下解决工具使用型语言模型智能体的动作伪造、来源崩溃与回声放大三大运行时验证缺口。

    

    使用模型上下文协议（MCP）的语言模型智能体面临三个运行时验证缺口：（i）动作伪造，即工具调用后的文本叙述了失败、超时或从未执行过的副作用；（ii）来源崩溃，即未经验证的模型文本或网页文本与已执行的工具输出一同进入持久化记忆；（iii）联合回声放大，即镜像网页从单一来源满足多来源交叉验证。现有的MCP防火墙在执行前检查线上调用，而LLM裁判类护栏需要数GB显存，且无法将文本叙述与回执绑定。我们提出了epistemic-guard（v0.5.2），一个开源Python库及进程外stdio MCP回执代理（15个模块，0 MiB GPU显存占用）。epistemic-guard将Ed25519签名密钥隔离在进程之外，写入带有AER-1回执翻译的RFC 6962 Merkle日志，并在CPU上使用经SHA-256验证的OpenVINO INT8编码器（4个模型，279.48（摘要在此处截断）

    arXiv:2611.27095v1 Announce Type: cross  Abstract: Language-model agents using the Model Context Protocol (MCP) face three runtime verification gaps: (i) action fabrication, where post-tool prose narrates side effects that failed, timed out, or never ran; (ii) provenance collapse, where unverified model prose or web text enters durable memory alongside executed tool outputs; and (iii) syndicated echo amplification, where mirrored web pages satisfy multi-source corroboration from one origin. Existing MCP firewalls inspect wire calls before execution, while LLM-judge guardrails require multi-gigabyte VRAM without binding prose to receipts. We present epistemic-guard (v0.5.2), an open-source Python library and out-of-process stdio MCP receipt proxy (15 modules, 0 MiB GPU VRAM). epistemic-guard isolates Ed25519 signing keys out of process, writes RFC 6962 Merkle logs with AER-1 receipt translation, checks post-turn prose on CPU with SHA-256-verified OpenVINO INT8 encoders (4 models, 279.48
    
[^141]: 在AI辅助模型验证中区分自动化收益与智能体贡献

    Separating Automation Gains from Agent Contributions in AI-Assisted Model Validation

    [https://arxiv.org/abs/2611.27094](https://arxiv.org/abs/2611.27094)

    提出一种以无LLM可控工作流为对照臂的分阶段架构与归因协议，只在智能体发现基线流程无法报告的缺陷时才记功，从而在银行模型验证中将智能体的真实贡献与自动化本身带来的收益区分开来。

    

    大语言模型（LLM）智能体正被提议用于银行模型验证，然而现有评估在比较智能体辅助工作与人工实践时，会把来自数字化证据和确定性测试的收益错误地归功于智能体。我们提出一种分阶段架构：一个不使用大语言模型的可控工作流（P1）既是回退机制，也是逐项授权的智能体任务（P2）的对照臂；同时提出一种归因协议，仅当智能体发现P1尚未报告的缺陷时才记入其功劳。该协议会扣除诱饵误报、不必要的操作以及对开发者断言的背书，将无效输出计为漏报，并将新近可自动化的发现连同发现功劳一并纳入P1，同时预先固定安全边界与盈亏平衡成本门槛。在一个合成信用评分卡测试平台上，五个脚本化参考策略证明了该协议的必要性：朴素的案例准确率（4/6）与汇总检测率（3/5）无法区分四种有缺陷的策略，而我们的协议能够区分全部五种策略。

    arXiv:2611.27094v1 Announce Type: cross  Abstract: LLM agents are being proposed for bank model validation, yet evaluations that compare agent-assisted work with manual practice credit the agent with gains that come from digitized evidence and deterministic tests. We propose a staged architecture in which an LLM-free controlled workflow (P1) is both the fallback and the control arm for individually authorized agent tasks (P2), and an attribution protocol that credits an agent only for faults P1 does not already report. The protocol charges decoy false alarms, unnecessary actions and endorsed developer assertions, counts invalid outputs as misses, folds newly automatable findings back into P1 with discovery credit, and fixes safety bounds and a break-even cost gate in advance. On a synthetic credit-scorecard harness, five scripted reference policies show why this matters: naive case accuracy (4/6) and pooled detection (3/5) tie four flawed policies, whereas the protocol separates all fi
    
[^142]: 有使用、少恢复：50个AI智能体框架与运行时的公开报告揭示了恢复机制的实际情况

    Exercised, Rarely Recovered: What Public Reports Say About Recovery in 50 AI Agent Harnesses and Runtimes

    [https://arxiv.org/abs/2611.27090](https://arxiv.org/abs/2611.27090)

    该研究在预设协议下对50个AI智能体框架与运行时的八种恢复机制进行系统评估，发现恢复机制虽在实践中被广泛使用，但128份真实使用报告中仅19份实现完全恢复，大部分以部分恢复或失败告终。

    

    当AI编程智能体删除了错误的文件，或一次长时间的智能体运行崩溃时，用户只能依赖恢复机制：从检查点恢复、撤销、请求批准、读取日志等。研究型运行时承诺更强的保障，例如saga式补偿和事务性回滚。我们探究公开证据是否表明这两类机制在真实使用中确实有效。在一个在数据收集前即已冻结的协议下，我们对12个持久化智能体运行时、26个编程智能体框架和12个研究型运行时的八种恢复机制进行评分，评分等级从“未发现”到“在真实运行中被实际使用”，并采用两次盲评编码、裁决机制以及对存档来源的重新验证。结果显示，恢复机制在实践中几乎普遍被使用：12个运行时中有11个、26个框架中有25个至少有一份真实使用报告。然而，在128份此类报告中，只有19份描述了运行完全恢复；54份为部分恢复，45份失败。检查点/恢复机制在27份报告中仅有一次实现恢复，回滚机制……

    arXiv:2611.27090v1 Announce Type: cross  Abstract: When an AI coding agent deletes the wrong files or a long agent run crashes, users fall back on recovery mechanisms: resume from a checkpoint, undo, ask for approval, read the log. Research runtimes promise stronger guarantees, such as saga-style compensation and transactional rollback. We ask what public evidence shows about whether either kind works in real use. Under a protocol frozen before data collection, we graded 12 persistent agent runtimes, 26 coding-agent harnesses and 12 research runtimes on eight recovery mechanisms, from not found to exercised in a real run, with two blind coding passes, adjudication and re-verification against archived sources. Recovery is exercised almost everywhere in practice: 11 of 12 runtimes and 25 of 26 harnesses have at least one real-use report. Yet of 128 such reports, only 19 describe the run fully recovering; 54 are partial and 45 failed. Checkpoint/resume recovered once in 27 reports, rollba
    
[^143]: 直接写入权重还是训练权重？基于权重空间编程与低秩自适应的空间推理

    Writing Weights or Training Them? Spatial Reasoning through Weight-Space Programming and Low-Rank Adaptation

    [https://arxiv.org/abs/2611.27089](https://arxiv.org/abs/2611.27089)

    在空间推理任务上，权重空间编程带来的提升有限，而仅用0.09%可训练参数、以解题示例为监督训练的秩1 LoRA适配器可近乎完美地解决参照系与StepGame任务，并对未见模板和更高难度展现一定泛化能力。

    

    权重空间编程使一个语言模型能够检查并编辑另一个模型的权重，为基于梯度的自适应方法提供了一种替代方案。我们使用Qwen3.5-0.8B在固定参照系任务（128题）、StepGame（200题）以及两个未经训练的距离比较任务上对两种方法进行了评估。在开启思考模式的条件下，公开补丁在三个采样种子下将参照系任务的平均准确率从0.0%提升至22.1%，将StepGame的准确率从19.2%提升至33.7%。随机打乱其权重改变会消除这一增益。进一步的闭式编辑未能超越p50c，而基于解题示例训练的秩1 LoRA适配器在两个任务上分别达到了100%和98.8%的准确率，且可训练参数仅占0.09%。若从监督信号中移除解题示例，适配器则无法通过学习门槛。训练后的适配器在未见过的StepGame句子模板上达到82%的准确率，在训练后留出的6-10难度级别上达到42%的准确率。

    arXiv:2611.27089v1 Announce Type: new  Abstract: Weight-space programming lets a language model inspect and edit another model's weights, offering an alternative to gradient-based adaptation. We evaluated both approaches using Qwen3.5-0.8B on a fixed reference-frame task (128 items), StepGame (200 items), and two untrained distance-comparison tasks. With thinking enabled, the public patch (p50c) raised mean accuracy from 0.0% to 22.1% on the reference-frame task and from 19.2% to 33.7% on StepGame across three sampling seeds. Randomly permuting its weight change removed the gain. Further closed-form edits did not improve on p50c, whereas a rank-1 LoRA adapter trained on worked solutions reached 100% and 98.8%, respectively, with 0.09% trainable parameters. Removing the worked solutions from supervision prevented the adapter from passing the learning gate. The trained adapters reached 82% accuracy on unseen StepGame sentence templates, 42% on held-out difficulty levels 6-10 after traini
    
[^144]: 双重愚弄心智：重复暴露与纠正反馈下人类对视听深度伪造的检测

    Fooling the Mind Twice: Human Detection of Audiovisual Deepfakes Under Repeated Exposure and Corrective Feedback

    [https://arxiv.org/abs/2611.27080](https://arxiv.org/abs/2611.27080)

    研究发现，重复暴露并结合纠正反馈能显著提高人类对视听深度伪造视频的检测准确率（FakeAVCeleb数据集上平均提升4.68个百分点），表明纠正反馈是改善人类深度伪造识别判断的有效手段。

    

    人类对深度伪造内容的检测仍然不可靠，而在纠正反馈辅助下进行重新评估能否改善后续判断尚不明确。本研究使用来自FakeAVCeleb和深度伪造检测挑战赛（DFDC）数据集的视听刺激材料，考察了110名参与者在初次评估与重复评估深度伪造内容之间的差异。参与者在第一阶段（PhaseI）对每个数据集中的40个视频进行评估，并在第二阶段（PhaseII）在结合重复呈现与纠正反馈的流程下重新评估相同的视频。在FakeAVCeleb数据集上，参与者的平均准确率从第一阶段的63.30%提升至第二阶段的67.98%，提高了4.68个百分点（t(109)=5.61, p<.001, d_z=0.54）。以参与者和视频进行标准误聚类的判断层面逻辑回归得到了一致结果：beta=0.208, SE=0.040, z=5.23, p<.001, OR=1.23。在DFDC数据集上，参与者的平均准确率也有所提升（原文摘要在此处截断）。

    arXiv:2611.27080v1 Announce Type: cross  Abstract: Human deepfake detection remains unreliable, and whether reassessment with corrective feedback improves subsequent judgments remains unclear. This study examined differences between initial and repeated deepfake assessments among 110 participants using audiovisual stimuli from the FakeAVCeleb and Deepfake Detection Challenge (DFDC) datasets. Participants evaluated 40 videos from each dataset in phase 1 (PhaseI) and reassessed the same videos in phase 2 (PhaseII) under a procedure combining repeated presentation and corrective feedback. For the FakeAVCeleb dataset, the participants' mean accuracy increased by 4.68 percentage points from 63.30 in PhaseI to 67.98 in PhaseII (t(109)=5.61, p<.001, d_z=0.54). A judgment-level logistic regression with standard errors clustered by participant and video produced a consistent result: beta=0.208, SE=0.040, z=5.23, p<.001, {OR}=1.23. For the DFDC dataset, the participants' mean accuracy increased 
    
[^145]: 粒子物理中模型编辑的权重空间编程与机制可解释性

    Weight-space programming and mechanistic interpretability for model editing in particle physics

    [https://arxiv.org/abs/2611.27079](https://arxiv.org/abs/2611.27079)

    该论文通过稀疏自编码器识别潜在方向的移除和语言模型智能体的免梯度权重编辑两种方法，成功修复了粒子物理基础模型中因预训练数据问题导致的违反红外安全性缺陷，且对模型其余功能几乎无影响，所需修改的参数远少于微调。

    

    我们证明，权重空间编程以及由稀疏自编码器引导的编辑，可以通过直接修改权重来消除粒子物理基础模型中不理想的特性。在一个在喷注数据上训练的基础模型中，我们观察到添加单个软中性粒子会导致喷注分类的巨大变化，违反了红外安全性。我们通过机制分析定位了负责该问题的注意力头，并将该缺陷的起源追溯到预训练数据集中不同的重建阈值。随后我们提出了两种针对性的修复方法：一是移除通过稀疏自编码器识别出的三个潜在方向；二是让语言模型智能体直接编辑权重，无需梯度计算，且仅需对模型进行50次查询。两种方法的编辑各自都能消除几乎整个缺陷，同时使模型其余部分的响应几乎保持不变。与微调相比，这两种方法改变的参数要少得多。

    arXiv:2611.27079v1 Announce Type: cross  Abstract: We demonstrate that weight-space programming and edits guided by sparse autoencoders can remove unwanted aspects of a particle-physics foundation model through direct edits to its weights. In a foundation model trained on jets, we observe that adding a single soft neutral particle causes large changes in jet classification, violating infrared safety. We locate the responsible attention heads with mechanistic analysis and trace the flaw's origin to pretraining on datasets with different reconstruction thresholds. We then propose two targeted approaches to repair the flaw: by removing three latent directions identified with sparse autoencoders, and by letting language-model agents edit the weights directly, without gradients and with only 50 queries to the model. Both their edits individually remove almost the entire flaw and leave the rest of the model's response nearly unchanged. Compared to fine-tuning, both methods change far fewer p
    
[^146]: 决策模型是通用的过程奖励模型

    Decision Models Are General-Purpose Process Reward Models

    [https://arxiv.org/abs/2611.27068](https://arxiv.org/abs/2611.27068)

    DM-PRM无需步骤标签、微调或评论家，零样本地将通用的是非问答决策模型转化为通用过程奖励模型，仅通过每步一个问题即可显著提升语言智能体在家庭任务、网络购物等多种环境和多种RL方法上的表现。

    

    基于可验证结果的强化学习（RL）为语言智能体每个回合只提供一次奖励：每一步都被同等地分配信用，而当所有滚动尝试都失败时，模型无法从任务中学到任何东西。最后一公里——步级信用分配——通常需要步骤标签、学习到的评论家（critic）、昂贵的LLM裁判或手写规则。我们提出DM-PRM，它将一个通用的决策模型——即以概率回答是非问题的模型——零样本地转化为通用的过程奖励模型（PRM）：无需步骤标签、无需微调、无需评论家。DM-PRM对每一步只提出一个问题，并将标准化后的答案加到优势函数上。它可以插入任何策略梯度方法；一个模型，每个环境只需一段问题文本，即可服务于家庭任务、网络购物和多种RL方法。使用Qwen2.5-1.5B，它在ALFWorld上将GRPO提升了二十多分，在WebShop上提升了十多分，在ALFWorld上超越了GiGPO，并且只需少得多的迭代即可学会任务；它还能进一步改进……

    arXiv:2611.27068v1 Announce Type: new  Abstract: Reinforcement learning (RL) from verifiable outcomes gives a language agent one reward per episode: every step is credited alike, and nothing about the task is learned when all rollouts fail. The last mile, step-level credit, usually requires step labels, a learned critic, costly LLM judges or hand-written rules. We introduce DM-PRM, which turns a general-purpose decision model -- one answering yes/no questions with probabilities -- into a general-purpose process reward model (PRM), zero-shot: no step labels, no fine-tuning, no critic. DM-PRM asks one question per step and adds the standardised answers to the advantage. It plugs into any policy-gradient method; one model, with one question text per environment, serves household tasks, web shopping and multiple RL methods. With Qwen2.5-1.5B it lifts GRPO by over twenty points on ALFWorld and over ten on WebShop, overtakes GiGPO on ALFWorld and learns in far fewer iterations; it also impro
    
[^147]: 面向高效视觉-语言模型的动态标记合并

    Dynamic Token Merging for Efficient Vision-Language Models

    [https://arxiv.org/abs/2611.27064](https://arxiv.org/abs/2611.27064)

    该论文将 MrT5 中的动态标记合并（DTM）推广为通用的标记删除框架，扩展至视觉模态与仅解码器模型，通过自适应删除视觉标记并压缩键值缓存来显著降低视觉-语言模型的推理成本。

    

    视觉-语言模型（VLM）使用数百个视觉标记来表示每张图像，这使得即使提示仅依赖于图像的一部分，推理成本也可能很高。动态标记合并（DTM）最初在 MrT5 中针对字节级文本编码器提出，它学习应该删除哪些标记：一个轻量级的删除门对上下文化的隐藏状态进行评分，在训练期间充当可微分的注意力偏置，并在推理时物理地删除标记。作者论证 DTM 是一个通用的标记删除框架，而非一种字节级技术，并将其扩展到视觉模态和仅解码器模型。将 DTM 应用于 T5Gemma 2 和 Gemma 3，得到了 V-MrT5Gemma2（在双向多模态编码器内部删除视觉标记）和 V-MrGemma3（在仅解码器的预填充阶段压缩提示及其键值缓存）。在十个基准测试中，DTM 相对于基于注意力的删除和随机删除的优势随删除率的增加而增长，并在（摘要在此处截断）

    arXiv:2611.27064v1 Announce Type: new  Abstract: Vision-language models (VLMs) represent each image with hundreds of visual tokens, which can make inference costly even when a prompt depends on only part of the image. Dynamic Token Merging (DTM), introduced in MrT5 for byte-level text encoders, learns which tokens to remove: a lightweight delete gate scores contextualized hidden states, acts as a differentiable attention bias during training, and physically removes tokens at inference. We argue that DTM is a general token deletion framework rather than a byte-level technique, and extend it to the vision modality and to decoder-only models. Applying DTM to T5Gemma 2 and Gemma 3 yields V-MrT5Gemma2, which deletes visual tokens inside a bidirectional multimodal encoder, and V-MrGemma3, which compacts the prompt and its key-value caches during decoder-only prefill. Across ten benchmarks, DTM's advantage over attention-based and random deletion grows with the deletion rate and is largest on
    
[^148]: PACE：基于显式扰动估计的策略增强与补偿实现稳健运动

    PACE: Policy Augmentation and Compensation With Explicit Disturbance Estimation for Robust Locomotion

    [https://arxiv.org/abs/2611.27061](https://arxiv.org/abs/2611.27061)

    PACE框架通过非线性观测器在线显式估计关节空间扰动（涵盖负载变化、质心偏移、执行器退化与外力），并将估计结果反馈到策略和控制器中进行前馈补偿，从而缓解域随机化在跟踪精度与泛化能力之间的折中，提升四足机器人运动的鲁棒性。

    

    强化学习已经使四足机器人运动具备了显著的鲁棒性。为了应对未建模的动力学和外部扰动，通常采用域随机化，使策略能够对冲其可能遇到的各种条件。然而，在不断变化的动力学条件下保持精确跟踪，以及在采样范围之外进行泛化，仍然具有挑战性。本文展示了显式估计扰动并将估计结果反馈到策略和控制器中，可以缓解这一折中。我们提出了带显式扰动估计的策略增强与补偿框架（PACE）。该框架将负载变化、质心（CoM）偏移、执行器性能退化以及外部作用力统一归结为一个关节空间扰动。随后，非线性观测器在线恢复该扰动，而无需关节加速度测量。该扰动估计提供了前馈力矩补偿。

    arXiv:2611.27061v1 Announce Type: cross  Abstract: Reinforcement learning has made quadrupedal locomotion remarkably robust. To handle unmodeled dynamics and external disturbances, domain randomization is usually employed to make a policy hedge against every condition it might meet. However, maintaining accurate tracking across varying dynamics and generalizing beyond the sampled range remain challenging. Here we show that explicitly estimating the disturbance and feeding the resulting estimation back into both the policy and the controller can relax this compromise. We introduce Policy Augmentation and Compensation with Explicit disturbance estimation (PACE). The framework lumps payload variations, center-of-mass (CoM) shifts, actuator degradation, and external forces into one joint-space disturbance. A nonlinear observer subsequently recovers this disturbance online without requiring joint acceleration measurements. The disturbance estimation provides feedforward torque compensation 
    
[^149]: 使用生成对抗网络学习3D磁共振成像（MRI）表示

    Learning 3D MRI Representations with Generative Adversarial Networks

    [https://arxiv.org/abs/2611.27057](https://arxiv.org/abs/2611.27057)

    该论文提出了R2RG方法，通过利用3D卷积神经网络的架构偏置并结合生成对抗网络优化的重建任务，在小规模3D MRI数据集上实现了保留空间3D信息的有效表示学习。

    

    学习视觉图像的表示是一项具有挑战性的任务，它需要在没有显式监督的情况下从高维数据中提取抽象模式。由于特征稀疏、输入维度更高以及数据集规模较小，这一挑战在3D磁共振成像中尤为突出。许多最先进的表示学习算法成功地依赖于Transformer和基于扩散的架构，然而这些方法在小数据集上表现不佳。实际上，现有的解决方法通过将3D体数据切片来简化问题，转而解决2D医学成像问题，从而导致空间3D信息的丢失。我们提出了R2RG，一种基于卷积神经网络的表示学习方法，利用生成对抗网络来优化重建任务。我们的关键见解在于利用3D卷积神经网络的架构偏置，结合数据高效的自动编码GAN和复合损失函数来增强……

    arXiv:2611.27057v1 Announce Type: cross  Abstract: Learning representations of visual images is a challenging task. It requires extracting abstract patterns from high-dimensional data without explicit supervision. This becomes pronounced with 3D MR imaging due to sparse features, higher dimensional inputs, and smaller datasets. Many state-of-the-art representation learning algorithms successfully rely on transformers and diffusion-based architectures. However, these methods struggle with smaller datasets. In effect, solution methods relax the problem by slicing 3D volumes and solving the problem for 2D medical imaging, resulting in a loss of spatial 3D information. We present R2RG, a representation learning method based on Convolutional Neural Networks (CNNs), with Generative Adversarial Networks (GANs) honing the reconstruction task. Our key insights lie in exploiting architectural biases in 3D CNNs, combined with data-efficient auto-encoding GANs and a composite loss function to enha
    
[^150]: 感知量化训练的连续松弛方法

    A Continuous Relaxation of Quantization-Aware Training

    [https://arxiv.org/abs/2611.27056](https://arxiv.org/abs/2611.27056)

    提出RelaxQAT，通过在训练过程中逐渐连续地改变位宽来松弛感知量化训练，从而避免低位量化中损失曲面的鞍点和平坦区域问题，显著提升量化模型性能。

    

    量化是大生成模型时代部署流程中的关键环节。低位量化，特别是2比特或低于2比特的量化，是一个众所周知的难题，尤其是在使用逐通道量化方案时——该方案已被证明是硬件效率方面的领先候选方案。低位量化的主要挑战包括量化级别有限以及损失曲面中存在的鞍点或平坦区域。在本文中，我们通过动态变化位宽的视角来审视量化损失曲面平坦化的问题。与现有技术要么突然改变位精度、要么保持位精度固定不变不同，我们提出在模型训练过程中逐渐改变位宽。为此，我们提出了RelaxQAT，一种QAT（感知量化训练）过程的连续松弛方法。我们还提供了简单的理论和实证见解，解释了为什么RelaxQAT能够提升QAT性能。我们广泛的实验证明……

    arXiv:2611.27056v1 Announce Type: new  Abstract: Quantization is a critical piece of the deployment pipeline in the era of large generative models. Low-bit quantization, particularly 2-bits or sub-2-bits, is a notoriously difficult problem, particularly when using a channelwise quantization scheme which has been shown as a leading candidate for hardware-efficiency. The key challenges with low-bit quantization range from limited quantization levels to saddle points or flat regions in the loss surface. In this paper, we look at the problem of flat quantization loss surfaces through a lens of dynamically changing bitwidth. Unlike existing techniques which either change the bit-precision abruptly or keep it fixed, we propose to gradually alter the bitwidth as the model trains. To this end, we propose RelaxQAT, a continuous relaxation of the QAT process. We also provide simple theoretical and empirical insights behind why RelaxQAT helps the QAT performance. Our extensive experiments demonst
    
[^151]: 行胜于言：通过Web交互中的行为隐蔽信道对LLM智能体进行有状态隐私嗅探

    Actions Speak Louder than Words: Stateful Privacy Sniffing of LLM Agents via Behavioral Covert Channels in Web Interactions

    [https://arxiv.org/abs/2611.27046](https://arxiv.org/abs/2611.27046)

    这篇论文提出SNOOP系统，利用LLM智能体在Web交互中的良性决策行为本身构成的行为隐蔽信道，无需任何显式数据传输即可主动嗅探并恢复智能体的细粒度私有上下文，揭示了一种现有攻击和防御手段均未覆盖的新型隐私攻击面。

    

    LLM智能体越来越多地依赖Web交互——发起搜索、点击链接、阅读网页——来完成复杂的用户任务。由于每个动作都由智能体的私有上下文驱动，包括用户的提示词、对话历史、记忆以及关联数据，这也开辟了一个新的隐私暴露面。先前的攻击要么被动分析网络流量，只能获得粗粒度的关联；要么通过间接提示注入劫持智能体，而这种方式必须将敏感内容放入输出字段中，因而会被内容级防御所阻断。因此，攻击者能否在不进行任何显式传输的情况下主动恢复细粒度的私有上下文，目前仍缺乏充分的研究。我们的核心洞察是：智能体良性的Web决策本身就会泄露私有上下文，因为它选择访问的内容编码了它是否行动、执行了什么行动以及行动的先后顺序，从而形成了一个行为隐蔽信道。基于这一洞察，我们设计了SNOOP，一个有状态的……（原文摘要在此处截断）

    arXiv:2611.27046v1 Announce Type: new  Abstract: LLM agents increasingly rely on Web interaction, issuing searches, following links, and reading pages, to complete complex user tasks. Since each action is driven by the agent's private context, including the user's prompt, dialogue history, memory, and connected data, it also opens a new privacy exposure surface. Prior attacks either passively analyze network traffic, which yields only coarse correlations, or hijack the agent through indirect prompt injection, which must place sensitive content in an outgoing field and is blocked by content-level defenses. Whether an attacker can actively recover fine-grained private context without any explicit transmission thus remains insufficiently understood.   Our insight is that an agent's benign Web decisions themselves leak private context, as what it chooses to visit encodes whether, which, and in what order it acts, forming a behavioral covert channel. Building on this, we design SNOOP, a sta
    
[^152]: 超图结构熵增强的层次感知半监督图分类

    Hypergraph Structural Entropy-Enhanced Hierarchy-Aware Semi-Supervised Graph Classification

    [https://arxiv.org/abs/2611.27043](https://arxiv.org/abs/2611.27043)

    提出HypSEE框架，通过构建游走诱导与嵌入衍生超图捕获层次感知的局部和长程高阶关系，并利用超图结构熵引导的层次超图池化方法实现半监督图分类中的层次表示学习。

    

    半监督图分类旨在利用大量无标签图与少量有标签图来学习图级表示，从而有效缓解标签稀缺问题。尽管各种半监督学习策略已取得显著进展，但图中顶点之间超越成对边的复杂关系，尤其是层次结构，仍未被现有方法充分探索。在这项工作中，我们提出了一种超图结构熵增强的层次感知框架——HypSEE，用于半监督图分类。具体而言，我们首先构建游走诱导超图和嵌入衍生超图，以捕获图内层次感知的局部和长程高阶关系。然后，我们设计了一种由超图结构熵（HSE）引导的层次超图池化方法，以促进超图中的层次表示学习。最后，我们采用（原文摘要在此处截断）

    arXiv:2611.27043v1 Announce Type: cross  Abstract: Semi-supervised graph classification aims to learn graph-level representations by leveraging a large number of unlabeled graphs alongside a small set of labeled ones, effectively mitigating the issue of label scarcity. Despite significant progress by various semi-supervised learning strategies, complex relationships among graph vertices beyond pairwise edges, especially hierarchical structures, remain underexplored by existing methods. In this work, we propose a Hypergraph Structural Entropy-Enhanced hierarchy-aware framework, namely HypSEE, for semi-supervised graph classification. Specifically, we first construct walk-induced and embedding-derived hypergraphs to capture hierarchy-aware local and long-range higher-order relationships within graphs. Then, we devise a hierarchical hypergraph pooling approach guided by hypergraph structural entropy (HSE) to facilitate hierarchical representation learning in hypergraphs. Lastly, we employ
    
[^153]: 通过神经元层面干预解耦言语化评估意识与拒绝行为

    Disentangling Verbalized Evaluation Awareness and Refusal through Neuron-Level Intervention

    [https://arxiv.org/abs/2611.27032](https://arxiv.org/abs/2611.27032)

    该研究通过识别并干预约5,000个“VEA神经元”，将模型的言语化评估意识率从12.6%–51.8%降至最高1.4%，从而在神经元层面揭示并解耦了评估意识与拒绝行为之间的关联机制。

    

    随着大型推理模型（LRM）能力的不断增强，可靠的评估变得至关重要。LRM有时能够识别出自己正在被评估，这种现象被称为“评估意识”。该现象引发了关于安全行为（尤其是拒绝行为）能否从评估场景泛化到部署场景的担忧。先前的研究通过激活导向来操纵言语化评估意识（VEA），并观察到当VEA存在时拒绝率更高，但这种关联背后的机制仍不明确。我们通过对比VEA与非VEA推理片段中的激活差异，识别出了“VEA神经元”。在主要实验中，对5,000个VEA神经元进行干预，使各模型-基准组合中的VEA率从12.6%–51.8%降至最高1.4%，并降低了未言语化评估意识的探测分数。在QwQ-32B模型中，VEA神经元干预还降低了拒绝……

    arXiv:2611.27032v1 Announce Type: new  Abstract: As large reasoning models (LRMs) become more capable, reliable evaluation becomes essential. LRMs can sometimes recognize that they are being evaluated, a phenomenon known as \emph{evaluation awareness}. This phenomenon raises concerns about whether safety behavior, particularly refusal behavior, generalizes from evaluation to deployment. Prior work manipulates \emph{verbalized evaluation awareness (VEA)} through activation steering and observes higher refusal rates when VEA is present, but the mechanisms underlying this connection remain poorly understood. We identify \emph{VEA neurons} by contrasting their activations in VEA and non-VEA reasoning segments. Intervening on 5,000 VEA neurons reduces the VEA rate from 12.6--51.8\% to at most 1.4\% across the model benchmark combinations in our main experiments, and lowers a probe score for unverbalized evaluation awareness (unVEA). In QwQ-32B, VEA-neuron intervention also reduces refusal e
    
[^154]: 双隐藏层ReLU最大值函数的整数乘子与算术障碍

    Integral Multipliers and Arithmetic Obstructions for Two-Hidden-Layer ReLU Maxima

    [https://arxiv.org/abs/2611.27018](https://arxiv.org/abs/2611.27018)

    本文借助代数拓扑工具（整数立方填充与同调障碍类）证明了双隐藏层整数参数ReLU网络精确表示最大值函数的最小正输出乘子——六输入为2、七输入为2或6，并刻画了七输入情形中剩余的算术障碍。

    

    我们研究用具有两个隐藏层且参数为整数的ReLU网络对最大值函数的精确表示。对于六输入最大值函数，最小正输出乘子为2；对于七输入最大值函数，最小正输出乘子为2或6。我们通过整数立方填充以及从多面体恒等式到网络的保系数转换获得这些界。对于六单纯形的坐标面复形，我们计算出其整数上的锥化指数为6，而在将2取逆后为3，从而允许每个面内任意的整数方向。无限制核复形同调中的一个类刻画了剩余的七输入问题；该类的消失等价于对固定目标实现整数填充。我们还构造了一个有限判别准则，在一个包含两个已知构造以及所有边不交三边森林对的字典中，将有理可表示性与三局可表示性区分开来。此障碍（摘要原文在此处截断）

    arXiv:2611.27018v1 Announce Type: new  Abstract: We study exact representations of maximum functions by ReLU networks with two hidden layers and integer parameters. The least positive output multiplier is two for the six-input maximum, and either two or six for the seven-input maximum. We obtain these bounds from integral cubical fillings and a coefficient-preserving passage from polytope identities to networks. For the coordinate-facet complex of the six-simplex, we compute a coning index of six over the integers and three after inverting two, allowing arbitrary integral directions within each facet. A class in the homology of the unrestricted kernel complex characterizes the remaining seven-input problem; its vanishing is equivalent to an integral filling of a fixed target. We also construct a finite character that separates rational from three-local representability in a dictionary containing two known constructions and all pairs of edge-disjoint three-edge forests. This obstruction
    
[^155]: 相同事实，不同决策：类型化决策模型中的事实不变性操纵与基于来源的防御

    Same Facts, Different Decision: Fact-Invariant Manipulation and Provenance-Based Defense in Typed Decision Models

    [https://arxiv.org/abs/2611.27016](https://arxiv.org/abs/2611.27016)

    论文提出LEVER攻击，利用论证理论中的四个杠杆在不添加任何新事实的情况下操纵类型化决策模型做出错误决策，并提出Provenance Gate防御机制，仅当模型在无外部评论时能复现该决策才予以接受。

    

    诸如TypeSafe的Jev等类型化决策模型仅以概率、选项或分数的形式给出答案，因此软件可以直接根据其输出采取行动。然而，这些模型的输入通常包含来自外部人员的文本。以往的攻击对此风险的测试效果不佳：注入的指令在这些模型上很少奏效，而添加的事实可能使被改变的决策变得正确。我们证明，外部评论可以在不添加任何事实的情况下说服这些模型做出错误决策。我们提出LEVER攻击，该方法利用论证理论中的四个“杠杆”来撰写此类评论，使用独立的验证器丢弃任何添加事实的评论，并以模型的输出概率为引导搜索有效评论。我们进一步提出Provenance Gate防御机制，只有当模型在没有外部评论的情况下再次得出相同决策时，才接受改变后的决策。在三个基准测试中，LEVER翻转了Jev约三分之一的正确决策，是最强先前攻击的2.8倍。

    arXiv:2611.27016v1 Announce Type: new  Abstract: Typed decision models such as TypeSafe's Jev answer only with a probability, an option, or a score, so software can act on them directly. Their inputs, however, often include text from outsiders. Prior attacks test this risk poorly. Injected instructions rarely work on these models, and added facts can make a changed decision correct. We show that outside comments can persuade these models to make a wrong decision without adding any fact. We propose LEVER, an attack that writes such comments with four levers from argumentation theory, uses a separate verifier to discard any comment that adds a fact, and searches for effective comments guided by the model's output probabilities. We further propose the Provenance Gate, a defense that accepts a changed decision only if the model reaches it again without the outside comments. On three benchmarks, LEVER flips about a third of Jev's correct decisions, 2.8 times as many as the strongest prior a
    
[^156]: Ruby：利用学习到的结构表示在不同药物发现阶段实现最先进的药物效力预测

    Ruby: State-of-the-Art Potency Prediction across Discovery Regimes Using Learned Structural Representations

    [https://arxiv.org/abs/2611.27015](https://arxiv.org/abs/2611.27015)

    Ruby是一个利用结构基础模型Pearl的中间表示、无需显式结合姿态的鲁棒通用结合亲和力预测器，通过专有数据与公开数据集GenesisPotency-2026训练，在药物发现的各个阶段均实现了最先进的药效预测。

    

    小分子药物发现通过反复的设计-合成-测试-分析循环进行，其目标通常是在保持其他性质处于可行治疗候选药物所需范围内的同时，最大化药物效力。随着药物发现从苗头化合物识别推进到先导化合物转化和先导化合物优化阶段，化合物优先级排序对计算方法提出了不同的要求，计算预算不断变化，靶向实验数据量也在不断增加。我们介绍了Ruby，一个鲁棒的通用结合亲和力预测器，它利用结构基础模型Pearl的中间表示，而无需为每个化合物提供显式的结合姿态。Ruby在专有数据以及GenesisPotency-2026上进行训练，后者是一个经过审计的公开来源药效训练数据集，以CC BY-SA 4.0许可证发布供学术和商业使用，并带有行级来源追溯和审计注释。Ruby实现了最高的……

    arXiv:2611.27015v1 Announce Type: cross  Abstract: Small-molecule drug discovery proceeds through repeated design-make-test-analyze cycles. The goal is often to maximize potency while keeping other properties within the ranges required for a viable therapeutic candidate. Compound prioritization places different demands on computational methods as drug discovery progresses from hit identification through hit-to-lead and lead optimization, with changing computational budgets and increasing amounts of on-target experimental data. We introduce Ruby, a Robust Universal Binding affinitY predictor that uses intermediate representations from the structural foundation model Pearl without requiring an explicit binding pose for every compound. Ruby is trained on proprietary data plus GenesisPotency-2026, an audited public-source potency training dataset released for academic and commercial use under a CC BY-SA 4.0 license with row-level provenance and audit annotations. Ruby achieves the highest 
    
[^157]: 一次获得多个规划方案：连续时间中的多样化高质量规划

    Many Plans at Once: Diverse Top-Quality Planning in Continuous Time

    [https://arxiv.org/abs/2611.27012](https://arxiv.org/abs/2611.27012)

    该论文提出MPAO方法，通过采样构造上满足边界条件的神经网络、训练共享修正并自适应调整采样策略，在连续时间中同时生成多个低成本且相互不同的高质量规划方案，为遇到未建模约束或偏好变化时的快速适应提供备选方案与热启动。

    

    物理系统的规划器通常只返回单条轨迹，当未建模的约束使所选路线失效或下游偏好发生变化时，便没有其他备选方案。预先计算好的一组规划方案可以提供现成的替代方案以及用于快速适应的热启动，但前提是这些方案必须既低成本又相互不同。离散规划通过高质量与多样化规划解决了这一问题，但这些思想无法直接迁移到连续时间场景——在连续时间中，每条有效轨迹都存在无穷多个有效的近似副本，而现有的轨迹规划器只能收敛到一个或少数几个解。我们提出了 Many Plans at Once（MPAO），它将找到的每个方案都视为前往其他区域继续搜索的理由：它采样未经训练的神经网络，其输出在构造上即满足边界条件；训练来自同一分布的多个网络所共享的修正；存储每一个经过验证的方案；并根据已找到的结果自适应地调整采样策略。

    arXiv:2611.27012v1 Announce Type: new  Abstract: Planners for physical systems usually return a single trajectory, which leaves no alternative when an unmodeled constraint invalidates the chosen route or downstream preferences change. A set of plans computed in advance provides ready-made alternatives and warm-starts for fast adaptation, but only if the plans are both low-cost and mutually distinct. Discrete planning addresses this with top-quality and diverse planning, but these ideas do not transfer to continuous time, where every valid trajectory has infinitely many valid near-copies, and existing trajectory planners converge to one or a few solutions. We present Many Plans at Once (MPAO), which treats every plan it finds as a reason to search elsewhere: it samples untrained neural networks whose outputs meet the boundary conditions by construction, trains corrections shared by networks from the same distribution, stores every validated plan, and adapts its sampling to what it has f
    
[^158]: FoRe：一种用于视觉-语言-动作模型中自主抓取恢复的力引导框架

    FoRe: A Force-Guided Framework for Autonomous Grasp Recovery in Vision-Language-Action Models

    [https://arxiv.org/abs/2611.27011](https://arxiv.org/abs/2611.27011)

    该论文提出FoRe框架，利用力反馈将演示数据划分为接近-抓取与搬运-放置阶段分别训练策略，并通过局部强化学习从失败后状态优化专用重试策略，实现视觉-语言-动作模型从抓取失败中的自主恢复。

    

    视觉-语言-动作模型在基于抓取的操作任务中展现出了强大的性能，然而从局部抓取失败中进行可靠恢复仍然具有挑战性。具体而言，在拾取与放置任务中，策略可能在抓取失败后仍继续搬运物体而不进行重试，这可能反映出其对本体感觉线索进行阶段推进的过度依赖，以及重试监督数据的匮乏。现有的恢复方法通常依赖于部署时的引导或演示数据增强。我们提出了力引导恢复框架，这是一个用于从局部抓取失败中实现自主恢复的框架。通过利用力反馈，FoRe 将演示数据划分为接近-抓取和搬运-放置两个阶段，分别训练独立的策略。为补充有限的重试监督数据，FoRe 随后从接近-抓取策略初始化一个专用的重试策略，并通过从失败后状态出发的局部强化学习对其进行优化。

    arXiv:2611.27011v1 Announce Type: cross  Abstract: Vision-language-action (VLA) models have demonstrated strong performance in grasp-based manipulation, yet reliable recovery from local grasping failures remains challenging. Specifically, in pick-and-place tasks, policies may continue transport after a failed grasp instead of retrying, potentially reflecting overreliance on proprioceptive cues for stage progression and limited retry supervision. Existing recovery approaches typically rely on deployment-time guidance or demonstration augmentation. We introduce Force-guided Recovery (FoRe), a framework for autonomous recovery from local grasping failures. Using force feedback, FoRe partitions demonstrations into approach-grasp and transport-placement stages to train separate policies. To supplement limited retry supervision, FoRe then initializes a dedicated retry policy from the approach-grasp policy and refines it through local reinforcement learning (RL) from post-failure states. To a
    
[^159]: 基于循证实践推荐的可扩展医学人工智能评估

    Scalable Evaluation of Medical AI using Evidence-based Practice Recommendations

    [https://arxiv.org/abs/2611.27007](https://arxiv.org/abs/2611.27007)

    该论文提出一个自动化流程，从BMJ Best Practice等循证临床实践推荐中提取并验证结构化临床标准与决策树，大规模生成3,200个涵盖鉴别诊断、检查、治疗与并发症的临床场景，实现了对医学AI在跨地区、跨指南版本条件下开放式临床管理能力的可扩展评估。

    

    严格评估医学人工智能（AI）需要在多样化的临床表现和不同护理阶段中测试开放式的临床管理能力。然而，现有的基准测试依赖人工专家编写，限制了对各类疾病、条件性护理路径以及不断更新的指南的覆盖范围。我们开发了一个自动化流程，能够从实践推荐中提取并验证结构化临床标准和分支决策树，从而生成覆盖整个护理连续体的场景，包括细微的临床表现以及跨地区和不同指南版本的临床实践差异。将该流程应用于436份BMJ Best Practice临床专著（354份美国版，82份国际版），生成了3,200个场景，涵盖鉴别诊断、检查、治疗和并发症，此外还生成了针对地区特异性（731对美国/国际配对）和时间新鲜度（630个场景）的专项测试集。为了对回复进行评分，我们开发了一个临床医生校准的（原文在此处截断）

    arXiv:2611.27007v1 Announce Type: new  Abstract: Rigorously evaluating medical artificial intelligence (AI) requires testing open-ended clinical management across diverse presentations and stages of care. However, existing benchmarks rely on manual expert authoring, limiting coverage across conditions, conditional care pathways, and evolving guidelines. We develop an automated pipeline that extracts and verifies structured clinical criteria and branching decision trees from practice recommendations to generate scenarios spanning the care continuum, including nuanced presentations and practice variations across regions and guideline editions. Applied to 436 BMJ Best Practice clinical monographs (354 US, 82 International), this yields 3,200 scenarios spanning differential diagnosis, investigations, treatment, and complications, plus targeted suites for regional specificity (731 US/International pairs) and temporal freshness (630 scenarios). To grade responses, we develop a clinician-cali
    
[^160]: JevRisk：基于语言决策模型的零样本金融风险评估

    JevRisk: Zero-Shot Financial Risk Assessment with a Language Decision Model

    [https://arxiv.org/abs/2611.26999](https://arxiv.org/abs/2611.26999)

    JevRisk提出了一个基于语言决策模型的零样本金融风险评估系统，通过遵循金融监管指南用代码计算精确风险指标，摆脱了对人工标签的依赖，并能在不同客户和风险类型间泛化。

    

    银行必须结合客户画像来解读金融活动，从而监控每位客户的金融风险，例如洗钱和信用违约。现有的银行风险监控系统大多基于专家设计的风险指标，这些指标对每位客户应用预定义的阈值，无法适应客户的个人情况。监督式模型可以从有标签的客户中识别风险模式，但每个标签通常需要专家审查或数月的还款历史，既缓慢又成本高昂。因此，金融风险评估需要一个零样本决策系统，以最大限度地减少对标签的依赖，并在不同客户和风险类型之间实现泛化。为此，我们提出了JevRisk，一个基于语言决策模型的零样本金融风险评估系统。JevRisk分两步工作：首先，遵循金融领域的监管指南，JevRisk用代码计算精确的风险指标……

    arXiv:2611.26999v1 Announce Type: cross  Abstract: Banks must monitor every customer's financial risk, such as money laundering and credit default, by interpreting financial activity in the context of the customer's profile. Existing risk monitoring systems of banks are mostly based on expert-designed risk indicators, which apply predefined thresholds to every customer and cannot adapt to their personal situations. Supervised models recognise risk patterns from labelled customers, but each label often needs an expert's review or months of repayment history, which is slow and costly. Financial risk assessment therefore calls for a zero-shot decision system that minimises the need for labels and generalises across customers and risk types. To this end, we present JevRisk, a system for zero-shot financial risk assessment with a language decision model. JevRisk works in two steps. First, following regulatory guidelines in the finance domain, JevRisk computes exact risk indicators with code
    
[^161]: 面向大语言模型护栏红队测试的对抗性强化学习中的奖励完整性

    Reward Integrity in Adversarial Reinforcement Learning for LLM Guardrail Red-Teaming

    [https://arxiv.org/abs/2611.26998](https://arxiv.org/abs/2611.26998)

    该研究发现在金融服务LLM护栏红队测试的对抗性强化学习循环中，奖励评判可能出现虚假胜利（如攻击遗漏规定PII或评判者误判未违规响应），并通过对比多种决策模型验证了采用专门决策模型作为奖励评判者的必要性。

    

    我们研究了用于红队测试金融服务语言模型护栏的对抗性强化学习循环中的奖励保真度问题。当攻击遗漏了规定的个人身份信息（PII），或者当奖励评判者认可了一个并未突破目标策略的响应时，可能会出现表面上的“胜利”。在攻击者生成的示例和公开基准示例上，TypeSafe Jev 用于检查攻击中规定的 PII 是否存在。在结果评分方面，我们将决策模型 TypeSafe Jev、Laya（预训练和后训练版本）以及预训练的 CLM-8B 与来自重放语料库的 Opus 4.8 判定日志进行了比较。后训练显著提高了 Laya 与评判者的一致性。另一个由作者审核的脱靶数据集证实，Opus 可能会授予错误的胜利。这些发现促使人们采用专门的决策模型评判者，用于检查攻击输入中规定的 PII，并根据预期策略对目标响应进行评分。

    arXiv:2611.26998v1 Announce Type: new  Abstract: We study reward fidelity in an adversarial reinforcement-learning loop for red-teaming financial-services language-model guardrails. Apparent wins can arise when an attack omits prescribed personally identifiable information (PII), or when a reward judge credits a response that did not break the targeted policy. On attacker-generated and public benchmark examples, TypeSafe Jev checks whether the prescribed PII is present. For outcome scoring, we compare the decision models TypeSafe Jev, Laya (pretrained and post-trained), and pretrained CLM-8B with logged Opus 4.8 judgements from a replay corpus. Post-training markedly improves Laya's agreement with the judge. A separate author-reviewed off-target set establishes that Opus can award false wins. The findings motivate specialised decision-model judges for checking prescribed PII in attack inputs and scoring target responses against the intended policy.
    
[^162]: 从信用证要求到可执行规范：评估验证的准确性与完整性

    From Letter-of-Credit Requirements to Executable Specifications: Evaluating Verification Accuracy and Completeness

    [https://arxiv.org/abs/2611.26988](https://arxiv.org/abs/2611.26988)

    该论文将信用证审核要求逐条转化为可执行的原子条件并结合确定性验证器实现可追溯审核，揭示了仅以最终决定准确率衡量审核的误导性，并在数据保密条件下通过仅用合成数据微调的本地9B解析器达到93%的原子要求检测率。

    

    在信用证（LC）单据审核中，正确的最终决定并不能表明每一项要求都经过了审核。因此，我们逐条评估审核要求，并将每项要求变为可执行的。语言模型将信用证中的提单（BL）和商业发票（IV）要求转换为原子条件和逻辑结构，再由确定性验证器检索指定的证据并计算每项结果，使得每项检查都可以与黄金编码进行对比并追溯至日志记录的证据。由于信用证记录是保密的，我们本地部署的9B解析器仅在基于模板和人工审核的公开文档值构建的合成数据对上进行微调。在345个真实信用证案例上，最终决定准确率被证明具有误导性：一个恒定输出PASS的规则拥有最高的原始准确率，却什么都没有审核。尽管微调后的解析器在训练期间从未见过真实记录，它仍检测出93%的原子要求……

    arXiv:2611.26988v1 Announce Type: new  Abstract: In letter-of-credit (LC) document examination, a correct final decision does not show that every requirement was examined. We therefore evaluate examination requirement by requirement and make each requirement executable. A language model converts the bill of lading (BL) and commercial invoice (IV) requirements of a credit into atomic conditions and logical structures, and a deterministic verifier retrieves the named evidence and computes each outcome, so every check can be compared with a gold encoding and traced to logged evidence. Because LC records are confidential, our on-premises 9B parser is fine-tuned only on synthetic pairs built from templates and human-reviewed public document values. On 345 real LC cases, final-decision accuracy proves misleading: a constant PASS rule has the highest raw accuracy while examining nothing. Although it never sees a real record during training, the fine-tuned parser detects 93% of the atomic requ
    
[^163]: 自进化个人智能体中的隐私保护：持久记忆与跨载体迁移

    Privacy Protection in Self-Evolving Personal Agents: Persistent Memory and Cross-Carrier Transfer

    [https://arxiv.org/abs/2611.26983](https://arxiv.org/abs/2611.26983)

    本综述系统比较了个人智能体持久记忆的隐私保护机制（准入、维护、释放与撤回），通过案例场景与诊断实验量化了标签保护和记忆删除的效用代价，并揭示了记忆谱系缺失导致的隐私清理隐患。

    

    个人智能体通过持久记忆将私人经验带入后续决策之中。本主题综述按受保护变更、观察者和验证设置，比较了记忆的准入、维护、释放与撤回机制。筛选语料库中的直接证据来自受控记忆工作流。文章通过三个详细案例场景，追踪同一个隐私事实在预订、反馈驱动学习和撤回过程中的流转。我们展示了提示词修订与记忆读取何时可以组合生效，技能何时继承私有头部的约束，以及为什么读取者不再使用并不能证明已学习状态已被清理。诊断实验测量了更广泛标签保护和大规模记忆删除所带来的效用代价，并测试了缺失谱系信息如何导致衍生状态变得难以追踪。

    arXiv:2611.26983v1 Announce Type: cross  Abstract: Personal agents carry private experience into later decisions through persistent memory. This thematic survey compares memory admission, maintenance, release, and withdrawal by protected change, observer, and validation setting. Direct evidence in the screened corpus comes from controlled memory workflows. Three worked scenarios follow one private fact through booking, feedback-driven learning, and withdrawal. We show when prompt revision and memory reads compose, when skills inherit a private head's bound, and why reader non-use does not establish cleanup of learned state. Diagnostic experiments measure the utility costs of wider label protection and broad memory deletion, and test how missing lineage obscures descendants.
    
[^164]: Contexture：面向高吞吐量智能体服务的运行时上下文淘汰

    Contexture: Runtime Context Eviction for High-Throughput Agent Serving

    [https://arxiv.org/abs/2611.26980](https://arxiv.org/abs/2611.26980)

    Contexture提出了一种与框架和模型无关的运行时上下文淘汰架构，通过将工具执行信息与全局KV缓存压力可见性相结合，识别出64%可删除的过时历史词元，在不修改框架逻辑或模型执行的情况下降低了KV缓存需求并提升智能体服务的吞吐量。

    

    诸如Hermes、OpenClaw和Codex等LLM智能体会反复重发不断增长的执行历史，增加了高带宽内存（HBM）中共享GPU键值（KV）缓存的需求。我们证明了共享KV容量是限制智能体服务吞吐量的关键因素。现有的上下文缩减方法缺乏对全局缓存状态的可见性，而运行时KV管理器又缺乏工具执行信息。通过将词元追溯到其来源的工具操作，我们发现后续操作会取代早期的输出及其相关对话轮次，从而识别出最终历史记录中64%的词元可作为删除候选。我们提出了Contexture，一种与智能体框架和模型均无关的运行时上下文淘汰架构。Contexture将框架的完整对话历史与模型的执行上下文相分离，在不修改框架逻辑或模型执行的情况下减少后续请求的KV需求。它将工具执行信息与对共享KV缓存压力的全局可见性相结合，（摘要在此处被截断）

    arXiv:2611.26980v1 Announce Type: cross  Abstract: LLM agents such as Hermes, OpenClaw, and Codex repeatedly resend growing execution histories, increasing shared GPU key-value (KV) cache demand in high-bandwidth memory (HBM). We show that shared KV capacity limits agent-serving throughput. Existing context-reduction methods lack global cache-state visibility, while runtime KV managers lack tool-execution information. Tracing tokens to their originating tool operations reveals that later operations supersede earlier outputs and associated turns, identifying 64% of recorded final-history tokens as removal candidates.   We present Contexture, a harness- and model-agnostic architecture for runtime context eviction. Contexture separates the harness's full conversation history from the model's execution context, reducing subsequent requests' KV demand without modifying harness logic or model execution. It combines tool-execution information with global visibility into shared KV pressure and
    
[^165]: MixFold：基于输出通道分组的移动NPU亚4比特混合精度大语言模型推理

    MixFold: Sub-4-Bit Mixed-Precision LLM Inference via Output-Channel Grouping on Mobile NPUs

    [https://arxiv.org/abs/2611.26976](https://arxiv.org/abs/2611.26976)

    MixFold通过将输出通道分组与移动NPU矩阵单元瓦片对齐，无需解码时扩展权重即可原生执行3比特和对称2比特混合精度大语言模型推理，在骁龙Hexagon NPU上实现比llama.cpp相当或更高的准确率与更快的解码速度。

    

    亚4比特量化使移动设备能够运行更大的大语言模型，但其节省的内存并不总能转化为移动NPU上更快的解码速度。移动NPU的矩阵单元仅支持少数几种原生整数数据类型，而在向量单元上转换不支持的权重格式可能会抵消比特数减少带来的收益。混合精度还面临一个额外的约束：每个矩阵单元瓦片只能使用一种数据类型。MixFold能够在解码过程中不将权重扩展为更宽数据类型的前提下，使用原生整数操作执行3比特和完全对称的2比特权重。MixFold随后将数据类型分配给与矩阵单元瓦片对齐的输出通道组，从而实现比层级别更细粒度的分配，且不会干扰矩阵单元的执行。我们在骁龙8 Elite Gen 5的高通Hexagon NPU上，将MixFold与llama.cpp的Hexagon后端进行比较。在Qwen3模型系列上，MixFold取得了与llama.cpp亚4比特格式相当或更高的准确率。

    arXiv:2611.26976v1 Announce Type: new  Abstract: Sub-4-bit quantization allows mobile devices to run larger LLMs, but its memory savings do not always translate into faster decoding on mobile NPUs. Their matrix units support only a few native integer datatypes, and converting unsupported weight formats on the vector unit can erase the gains from fewer bits. Mixed precision faces an additional constraint: each matrix-unit tile uses one datatype. MixFold executes 3-bit and perfectly symmetric 2-bit weights using native integer operations without expanding the weights to wider datatypes during decoding. MixFold then assigns datatypes to output-channel groups aligned with the matrix-unit tiles, enabling finer-than-layer assignment without disrupting matrix-unit execution. We compare MixFold with llama.cpp Hexagon backend on the Qualcomm Hexagon NPU of a Snapdragon 8 Elite Gen 5. Across the Qwen3 model family, MixFold achieves comparable or higher accuracy than llama.cpp's sub-4-bit formats
    
[^166]: 运行一次，得到两个：本地LLM的近乎零成本的流式安全检查

    Run One, Get Two: Near-Free Streaming Safety Checks for Local LLMs

    [https://arxiv.org/abs/2611.26975](https://arxiv.org/abs/2611.26975)

    ToRA通过让第二个模型作为额外一行嵌入生成器的解码步骤并共享权重与KV缓存，以每个token不到4%的额外延迟实现了接近专用防护模型准确性的本地LLM流式安全检查。

    

    本地LLM助手一次只为一个用户解码，因此解码受内存带宽限制，计算资源处于闲置状态。如今的流式安全检查需要在准确性与成本之间权衡：专用的防护模型虽然准确，但几乎使每个token的延迟翻倍；而基于隐藏状态的探针虽然廉价，但准确性较低。我们提出ToRA，这是一种适配器，它在生成器的解码步骤中将第二个模型作为额外的一行来运行，共享其权重和KV缓存。双行内核只需为两行各读取一次每个权重。在Qwen3-0.6B上，ToRA匹配或超越了专用的流式防护模型，在17个基准测试中的12个上取得了更高的F1分数。它在相似成本下比探针多检测出11.8个百分点的不安全响应，且每个token仅增加不到4%的延迟。

    arXiv:2611.26975v1 Announce Type: new  Abstract: Local LLM assistants decode for one user at a time, so decoding is bound by memory bandwidth and leaves compute idle. Streaming safety checks today trade accuracy against cost. A dedicated guard model is accurate but nearly doubles the latency per token, while a hidden-state probe is cheap but less accurate. We present ToRA, an adapter that runs a second model as an extra row in the generator's decode step, sharing its weights and KV cache. Dual-row kernels read each weight once for both rows. On Qwen3-0.6B, ToRA matches or beats a dedicated streaming guard, with higher F1 on 12 of 17 benchmarks. It detects 11.8 points more unsafe responses than the probe at a similar cost, under 4% added latency per token.
    
[^167]: 当感知与先验冲突时：诊断视频理解模型中的模态极化现象

    When Perception Collides with Priors: Diagnosing Modal Polarization in Video Understanding Models

    [https://arxiv.org/abs/2611.26972](https://arxiv.org/abs/2611.26972)

    该论文揭示了多模态大语言模型在视频感知与先验常识冲突时会出现“模态极化”失败模式（分为认知惯性与感知惯性两种极端），并构建了基于反直觉场景的TheaterBench基准来系统诊断这一问题。

    

    多模态大语言模型（MLLMs）在视频理解方面取得了强大的能力，这主要得益于在真实视频数据上的广泛训练。然而，当视频感知与内部先验或常识发生严重冲突时，我们发现了一种关键的失败模式，称之为“模态极化”，即模型无法整合各模态信息，而是分裂为两个截然不同的极端：认知惯性——模型尽管存在矛盾的视觉证据仍僵化地坚持先验；以及感知惯性——模型盲目信任视频感知而忽视逻辑上的不一致。为了研究这一现象，我们引入了TheaterBench，一个旨在通过违反物理定律或常识的反直觉场景来揭示模态极化的基准测试——这类场景在标准训练数据分布中很少出现。大量实验表明，当前最先进的模型表现出严重的模态极化现象。

    arXiv:2611.26972v1 Announce Type: cross  Abstract: Multimodal Large Language Models (MLLMs) have achieved strong video comprehension, largely benefiting from extensive training on realistic video data. However, when video perception severely conflicts with internal priors or common sense, we identify a critical failure pattern termed Modal Polarization, where models fail to integrate modalities and instead bifurcate into two distinct extremes: Cognitive Inertia, where models rigidly adhere to priors despite contradictory visual evidence, and Perceptual Inertia, where they blindly trust video perception while ignoring logical inconsistencies. To investigate this, we introduce TheaterBench, a benchmark designed to expose Modal Polarization using counter-intuitive scenarios that violate physical laws or common sense - situations rarely encountered in standard training distributions. Extensive experiments reveal that current state-of-the-art (SOTA) models exhibit severe Modal Polarization,
    
[^168]: 正确的数字，错误的故事：当大语言模型在输入之外叙述财务变化时

    Right Numbers, Wrong Story: When LLMs Narrate Financial Changes Beyond Their Inputs

    [https://arxiv.org/abs/2611.26968](https://arxiv.org/abs/2611.26968)

    论文提出FrameCheck插件，在不增加模型调用的情况下检测并约束LLM财务报告草稿中的“无依据框架叙述”，确保报告叙述忠实于输入事实。

    

    由大语言模型（LLM）起草的财务报告可以把每个数字都写对，却仍会超出其输入内容，添加没有人提供的判断和因果解释。我们将这类添加内容称为“无依据的框架叙述”（unsupported framing），并发现它们在LLM草稿中出现的频率远高于专业人士撰写的报告；而试图通过在提示词中说明每个词的用法来修复该问题，反而会使这类判断更加频繁。为此，我们提出FrameCheck，这是一个置于LLM与人工审核之间的插件，用于确保每份草稿不超出其输入内容。具体而言，FrameCheck可以识别判断性词语、区分有依据和无依据的因果解释，且无需额外的模型调用，从而引导草稿忠实于输入事实。

    arXiv:2611.26968v1 Announce Type: new  Abstract: A financial report drafted by an LLM can state every number correctly but still go beyond its input, adding judgments and causes that no one supplied. We call these additions unsupported framing and find them far more often in LLM drafts than in reports written by proInput facts Revenue +3% Higher prices 10-K filings Revenue increased 3% on higher prices. 10-K writer Revenue surged 3% on higher prices, due to strong demand. LLM Fits the size? Surged: Unusual for 3% In the input? In the input? Strong demand: Not in input Higher prices: Kept fessionals, and attempts to fix them by prompting with how each word is used instead make such judgments more frequent. We therefore propose FrameCheck, a plug-in that sits beFrameCheck Revenue increased 3% on higher prices. No extra model call Judgment word Unsupported cause Supported cause tween the LLM and human review to keep each draft within its input. Specifically, FrameCheck 1) guides the draft
    
[^169]: 保留流程，更换供应商：面向使用工具的AI智能体的运行时治理

    Keep the Process, Change the Supplier: Runtime Governance for Tool-Using AI Agents

    [https://arxiv.org/abs/2611.26966](https://arxiv.org/abs/2611.26966)

    提出开放流程控制平面（OPCP）作为执行边界，将工具型AI智能体的动作提议与业务授权、上下文披露和供应商选择分离，实现运行时治理并在故障恢复时避免跨供应商的重复执行。

    

    使用工具的AI智能体会提出各种动作，而这些动作的授权和外部影响无法仅从生成的文本中推断出来。我们研究了一种执行边界，该边界将智能体的动作提议与业务控制的授权、上下文披露以及供应商选择分离开来。开放流程控制平面（OPCP）负责调解能力调用，并在供应商可能已执行操作但未返回可接受回复的情况下记录不确定的结果。我们使用脚本化请求和合成供应商对该运行时进行了评估：涵盖四种策略的2,400个故障事件、60个流程终止/重启事件、跨19种配置的19,000次定时调用、在43个测试装置上进行的五次定向变异，以及160条替换记录。评估中未涉及任何语言模型或自主智能体。OPCP与保守组合策略在所有测量的故障结果上保持一致。两者在收到明确拒绝后均能恢复，且不会产生跨供应商的重复执行；而“重试所有供应商”策略在每种测试的丢失场景中都会导致重复效应。

    arXiv:2611.26966v1 Announce Type: new  Abstract: Tool-using AI agents propose actions whose authorization and external effects cannot be inferred from generated text alone. We study an execution boundary that separates agent proposals from business-controlled authority, context disclosure, and supplier selection. The Open Process Control Plane (OPCP) mediates capability calls and records uncertain outcomes when a supplier may have acted without returning an acceptable reply. We evaluate this runtime with scripted requests and synthetic suppliers: 2,400 fault episodes across four policies, 60 process-termination/restart episodes, 19,000 timed calls across 19 configurations, five targeted mutations over 43 fixtures, and 160 substitution records. No language model or autonomous agent is evaluated. OPCP and a conservative composition agree on all measured fault outcomes. Both recover after definite refusal without cross-supplier duplication; retry-all duplicates effects in every tested los
    
[^170]: 超越提示启发式方法：评估大语言模型中的儿童安全对齐

    Beyond Prompt Heuristics: Assessing Child Safety Alignment in LLMs

    [https://arxiv.org/abs/2611.26965](https://arxiv.org/abs/2611.26965)

    该研究揭示了当前大语言模型的儿童安全对齐严重依赖伤害类别提示线索——在缺乏此类线索的看似无害请求上未对齐率大幅上升（最高达80个百分点），而仅引入一个伤害相关词即可恢复安全行为，表明模型的儿童安全推理主要建立在词汇和语义关联之上而非真正的安全理解。

    

    儿童安全分类体系通常通过一组预定义的伤害类别来刻画安全风险，而现有评估大多使用明确提及这些类别的提示词来操作化这些类别。然而，儿童安全方面的考量也可能出现在缺乏可识别伤害类别线索的看似无害的请求中。因此，我们研究了儿童安全对齐如何泛化到此类场景。在六个模型上的实验中，我们发现无伤害类别线索的请求的未对齐率显著高于包含此类线索的请求（Qwen3.5-27B、GPT-5.4-mini 和 Claude-Haiku-4.5 分别存在 80.0、54.0 和 48.4 个百分点的差距）。通过受控的行为与机制分析，我们发现仅仅引入一个与伤害相关的词就能降低大多数模型的未对齐率，且机制层面的证据表明，儿童安全推理可能部分依赖于词汇和语义关联。

    arXiv:2611.26965v1 Announce Type: cross  Abstract: Child-safety taxonomies typically characterize safety risks through a predefined set of harm categories, and existing evaluations largely operationalize these categories using prompts that explicitly invoke them. However, child-safety considerations can also arise in seemingly benign requests that lack recognizable harm-category cues. We therefore investigate how child-safety alignment generalizes to such settings. Across six models, we find considerably higher misalignment rates on requests without recognizable harm-category cues than on those containing such cues (80.0, 54.0, and 48.4 percentage-point gaps for Qwen3.5-27B, GPT-5.4-mini, and Claude-Haiku-4.5, respectively). Through controlled behavioral and mechanistic analyses, we find that introducing a single harm-associated word reduces misalignment across most models, with mechanistic evidence suggesting that child-safety reasoning may rely in part on lexical and semantic associa
    
[^171]: 多智能体大语言模型系统中无依据论断传播的追踪与限制：一种溯源感知框架

    Tracking and Limiting Unsupported Claim Propagation in Multi-Agent LLM Systems: A Provenance-Aware Framework

    [https://arxiv.org/abs/2611.26957](https://arxiv.org/abs/2611.26957)

    该论文提出了一个溯源感知框架，通过在多智能体LLM流水线中随论断一同传递其支持状态、来源、证据与传输历史，有效追踪并限制了无依据论断在智能体间传播时的语义漂移与无依据细节累积。

    

    本研究考察了无依据论断在多智能体流水线中传递时如何发生变化，并探讨了保留溯源信息能否减少这些变化。我们测试了同质与异质两种流水线：同质流水线中各智能体使用相同的模型，而异质流水线中各智能体使用不同的模型。我们将基线方法与一个溯源感知框架进行对比，该框架在传递每条论断时同时附带其支持状态、来源、证据及传输历史，且在传输过程中不引入任何新证据。我们测量了论断含义的变化、无依据细节的累积、语义漂移以及溯源信息的保留程度。这些指标可以表明该框架能否在论断于智能体之间传递时限制无依据的变化。

    arXiv:2611.26957v1 Announce Type: cross  Abstract: This research examines how unsupported claims change as they move through multi-agent pipelines. It also examines whether preserving provenance can reduce these changes. We test both homogeneous and heterogeneous pipelines. Homogeneous pipelines use the same model across agents, while heterogeneous pipelines use different models. We compare the baseline with a provenance-aware framework. The framework passes each claim with its support status, source, evidence and transmission history. No new evidence is introduced during transmission. We measure changes in claim meaning, unsupported detail accretion, semantic drift, and provenance preservation. These metrics show whether the framework limits unsupported changes as claims pass between agents.
    
[^172]: CodeReviewBench：一个用于评估软件工厂的自动化基准测试

    CodeReviewBench: An Automated Benchmark for Evaluating the Software Factory

    [https://arxiv.org/abs/2611.26942](https://arxiv.org/abs/2611.26942)

    该论文提出CodeReviewBench，以“开发者揭示的偏好”（开发者是否实际采纳机器人的审查建议）作为客观验证信号，每日从GitHub事件流挖掘50万+条判定数据，构建了一个可持续自动更新、覆盖18个工具的代码审查基准。

    

    随着AI智能体在极少人工监督下交付更多代码，代码审查成为关键的质控关卡——但这一环节缺乏可靠、客观的验证机制：审查质量微妙且最终是PR作者的主观判断。现有基准测试在结构上与此不符：其标注者是第三方审查者（如同被评估的机器人一样），而非真正决定修复哪些内容的PR作者。此外，现有基准依赖小型、静态的问题集，当工具更新时需要完全重新评估，限制了追踪进展的频率。我们提出CodeReviewBench，它建立在一种根本不同的信号之上：开发者揭示的偏好——如果开发者采纳了机器人的建议，我们将其视为有用；如果他们忽略它，我们则不视为有用。这一信号每日从公开的GitHub事件流中挖掘，获得覆盖18个工具的50万+条经过判定的PR，并随工具演进持续刷新。我们验证了开发者偏好……

    arXiv:2611.26942v1 Announce Type: new  Abstract: As AI agents ship more code with minimal human oversight, code review becomes the critical quality-control checkpoint -- but one without a reliable, objective verifier: review quality is nuanced and ultimately a judgment call by the PR author. Existing benchmarks are structurally misaligned with this: annotators are third-party reviewers, like the bots they evaluate, not the PR authors who actually decide what to fix. They also rely on small, static issue sets that require full re-evaluation as tools update, limiting how frequently progress can be tracked. We introduce CodeReviewBench, built on a fundamentally different signal: revealed developer preferences -- if a developer acts on a bot's suggestion, we treat it as useful; if they ignore it, we do not. This signal is mined daily from the public GitHub event stream, yielding 500K+ judged PRs across 18 tools, and refreshes continuously as tools evolve. We validate the developer preferen
    
[^173]: CENTRY：一个用于持续AI风险估计、监测与调控的概率潜在状态模型

    CENTRY: A Probabilistic Latent-State Model for Continual AI Risk Estimation, Monitoring, and Regulation

    [https://arxiv.org/abs/2611.26940](https://arxiv.org/abs/2611.26940)

    本文提出CENTRY概率框架，通过结构化潜变量模型将AI风险分解为一个捕捉AI-环境动态的快速交互子空间和三个编码数据、硬件与模型长期风险状态的慢速潜在子空间，从而实现对已部署AI系统全生命周期的持续风险估计、监测与调控。

    

    人工智能（AI）系统正越来越多地被部署于安全关键型应用中，然而现有的AI安全框架主要侧重于对AI风险进行定义、分类或治理，而非提出可在AI系统整个运行生命周期内持续估计、监测和调控风险的可实施模型与方法。本文提出了CENTRY，一个概率化框架，能够基于已部署AI系统与其环境的交互，实现对风险的持续估计、监测与调控。该框架将AI风险建模为源自三个内在来源：数据集、硬件资源以及模型/算法。文中构建了一个结构化潜变量模型，其中潜在表示被分解为一个捕捉不断演变的AI-环境动态的快速交互子空间，以及三个分别编码与数据集、硬件和模型相关的长期风险状态的慢速潜在子空间……

    arXiv:2611.26940v1 Announce Type: cross  Abstract: Artificial Intelligence (AI) systems are increasingly deployed in safety-critical applications, yet existing AI safety frameworks primarily define, categorize, or govern AI risk rather than proposing implementable models and methods for continually estimating, monitoring, and regulating risk throughout an AI system's operational lifetime. This paper proposes CENTRY, a probabilistic framework that enables continual risk estimation, monitoring, and regulation of a deployed AI system from its interactions with its environment. The framework models AI risk as arising from three intrinsic sources: datasets, hardware resources, and models/algorithms. A structured latent-variable model is developed in which the latent representation is decomposed into a fast interaction subspace that captures the evolving AI--environment dynamics and three slow latent subspaces that encode the long-term risk states associated with the dataset, hardware, and m
    
[^174]: MG-EDA：成熟度门控的偏态正态分布估计算法

    MG-EDA: Maturity-Gated Skew-Normal Estimation of Distribution Algorithms

    [https://arxiv.org/abs/2611.26938](https://arxiv.org/abs/2611.26938)

    该论文提出MG-EDA算法，通过种群成熟度统计量驱动的校准门控来调控均值偏移、协方差拉伸和偏度三项修正，解决了高斯分布估计算法在斜坡上因方差收缩而停滞的问题，且门控关闭时算法精确退化为标准EMNA。

    

    高斯分布估计算法在适应度地形斜坡上会收缩其方差，并在远离最优点处停滞不前。本文为沿运动方向起作用的三项修正——均值偏移、沿路径方向的协方差拉伸以及偏度——配备了它们所缺失的规则：一个由种群成熟度统计量驱动的校准门控，决定每项修正是否起作用、作用强度多大以及持续多长时间；当门控关闭时，该算法恰好退化为多元正态分布估计算法（EMNA）。该统计量是白化均值偏移的累积路径并带有逆Wishart修正，在无系统运动的零模型下其期望为1，其分位数给出具有已知误报率的阈值。它将三项修正作为同一个矩阵采样方程中的三个秩一项进行门控，其中偏度由保持估计均值的偏态正态采样器承载。

    arXiv:2611.26938v1 Announce Type: cross  Abstract: Gaussian estimation of distribution algorithms shrink their variance on slopes and stall far from the optimum. This paper equips the corrections that act along the direction of movement -- a mean shift, an elongation of the covariance along the path and a skewness -- with the rule they lack: a calibrated gate, driven by a maturity statistic of the population, decides whether, how strongly and for how long each correction acts, and when the gate is closed the algorithm is exactly the estimation of multivariate normal algorithm (EMNA). The statistic is a cumulative path of whitened mean shifts with an inverse-Wishart correction, whose expectation is one under a null model without systematic movement and whose quantiles give thresholds with known false-alarm rate. It gates the three corrections as three rank-one terms of one matrix sampling equation, the skewness being carried by a skew-normal sampler that preserves the estimated mean and
    
[^175]: SecretAgent：一个针对视觉语言模型的隐私保护多智能体框架

    SecretAgent: A Multi-Agent Framework for Privacy Protection against Vision Language Models

    [https://arxiv.org/abs/2611.26936](https://arxiv.org/abs/2611.26936)

    提出了首个通用的多智能体框架SecretAgent，通过LLM智能体迭代定位泄露隐私的视觉证据并自适应生成不可见扰动，同时借助外观审查智能体把关，在有效抵御视觉语言模型隐私推断的同时保持图像的视觉保真度。

    

    视觉语言模型（VLMs）能够从共享图像中推断出私人信息，包括位置、个人身份信息（PII）以及诸如年龄和性别等个人属性，甚至可以从医学图像中获取这些信息。对抗性扰动可以保护此类信息免受黑盒VLM的侵害，但现有方法忽略了图像质量，会在其保护的图像中引入可见的伪影。我们提出了SecretAgent，这是首个通用的多智能体框架，能够在保持视觉保真度的同时保护图像隐私免受VLM的侵害。LLM智能体迭代地定位揭示私人信息的视觉证据，并根据目标VLM的回答以及外观审查智能体（该智能体会拒绝任何带有可见伪影的候选方案）来调整不可见的扰动。我们用两个互补的变体实例化了SecretAgent：SecretAgent-Gradient通过六个代理编码器优化扰动以实现跨模型可迁移性……

    arXiv:2611.26936v1 Announce Type: cross  Abstract: Vision-language models (VLMs) can infer private information from shared images, including locations, personally identifiable information (PII), and personal attributes such as age and sex, even from medical images. Adversarial perturbations can protect such information against black-box VLMs, but existing methods overlook image quality and introduce visible artifacts into the images they protect. We present SecretAgent, the first general multi-agent framework that protects image privacy against VLMs while preserving visual fidelity. LLM agents iteratively locate the visual evidence that reveals private information and adapt an invisible perturbation to the target VLM's answers and to an appearance reviewer agent, which rejects any candidate with visible artifacts. We instantiate SecretAgent with two complementary variants: SecretAgent-Gradient optimizes the perturbation through six surrogate encoders for cross-model transferability, an
    
[^176]: 语言中信息的自相似组织

    Self-similar Organization of Information in Language

    [https://arxiv.org/abs/2611.26935](https://arxiv.org/abs/2611.26935)

    本文提出了一种基于信息度量标度行为的预测信息组织理论，并通过在长文本中观察到不同长度尺度上信息度量的标度坍缩，证明了语言中预测信息具有自相似结构，这对理解语言的统计规律及大语言模型如何学习语言结构具有重要意义。

    

    与前文文本相关的、用于预测后续内容的信息量（即预测信息）有多少，以及这些信息如何沿着文本历史分布？这些问题对于理解语言的统计结构以及大语言模型（LLMs）能够从中学到什么都至关重要。事实上，训练用于预测文本中下一个词元的LLM必须利用这种预测信息，因此其组织方式决定了语言结构如何变得可供模型学习。我们基于一系列信息度量的标度行为，发展了一种关于预测信息组织的理论，该理论预测：如果这种组织在所有测量长度尺度上具有相同的形式——即具有自相似结构——那么在不同长度上评估的信息度量将坍缩到一个共同的形式。我们在长文本中观察到了这种标度坍缩现象。

    arXiv:2611.26935v1 Announce Type: new  Abstract: How much information relevant to predicting the future of a passage (i.e. predictive information) is carried by the preceding text, and how is this information distributed along that history? These questions are fundamental to understanding both the statistical structure of language and what large language models (LLMs) can learn about it. Indeed, LLMs trained to predict the next token in text must exploit this predictive information, so its organization determines how the structure of language becomes available to the model to learn. We develop a theory for the organization of predictive information based on the scaling behavior of a family of information measures, and the theory predicts that if this organization has the same form at all measured length scales -- i.e., it has a self-similar structure -- then the information measures evaluated at different lengths collapse onto a common form. We observe this scaling collapse in long tex
    
[^177]: 对时刻而非帧进行分词：面向免训练视频大语言模型加速的分词器感知预算分配

    Tokenize Moments, Not Frames: Tokenizer-Aware Budget Allocation for Training-Free Video LLM Acceleration

    [https://arxiv.org/abs/2611.26933](https://arxiv.org/abs/2611.26933)

    论文揭示了视频LLM加速中精度损失的根源在于token选择之前的分词阶段——长视频采样间隔超过约2秒相似窗口导致视觉token出现“双重曝光”混合，并提出免训练方法ESSENCE，通过对时刻而非帧进行分词、并依据采样间隔与分词器类别用闭式规则分配token预算，从而在不损失精度的情况下加速视频LLM。

    

    视频大语言模型（LLM）使用数千个视觉 token 来表示每段视频，而通过剪枝或合并这些 token 的免训练加速方法在长视频上会出现精度损失。通过分析 16 个开源模型的帧相似性与视觉分词器，我们将这种损失追溯到了 token 选择之前的一个环节。在 token 预算约束下，长视频的帧采样间隔超过了帧保持相似的约 2 秒时间窗口。诸如 Qwen-VL 这类采用双帧 3D patch 的模型会将两个不相关的场景混合进同一个视觉 token 中，形成一种“双重曝光”效应，而这在任何剪枝或合并操作发生之前就已产生。像 InternVL 这类固定尺寸的分词器虽然避免了这种混合，但在相同的压缩率下表现却不同。我们提出 ESSENCE，一种免训练方法，它对“时刻”（即帧保持相似的片段）而非帧进行分词。在编码之前，一个闭式规则利用采样间隔和分词器类别来设定时刻的数量。

    arXiv:2611.26933v1 Announce Type: cross  Abstract: Video large language models (LLMs) represent each video with thousands of visual tokens, and training-free accelerators that prune or merge these tokens lose accuracy on long videos. Analyzing frame similarity and the vision tokenizers of 16 open models, we trace this loss to a step before token selection. Under a token budget, the frames of a long video are sampled farther apart than the roughly 2 s over which frames remain similar. The two-frame 3D patches of models such as Qwen-VL then blend two unrelated scenes into each visual token, a double exposure that arises before any pruning or merging. Fixed-size tokenizers such as InternVL's avoid this blend but respond differently to the same reduction. We propose ESSENCE, a training-free method that tokenizes moments, segments within which frames remain similar, rather than frames. Before encoding, a closed-form rule uses the sampling interval and the tokenizer class to set the number o
    
[^178]: 大语言模型表征中的部分规范轴

    Partially canonical axes in large language model representations

    [https://arxiv.org/abs/2611.26920](https://arxiv.org/abs/2611.26920)

    该研究挑战了“大语言模型表征坐标轴是任意的”这一常见假设，发现在12个LLM中原生轴比随机旋转轴更能与人类大脑语言响应和可解释语言特征对齐，揭示LLM表征轴是“部分规范的”——既非完全任意也非完全规范。

    

    大语言模型对语言发展出丰富的内部表征，但其各个坐标轴本身是否承载意义，还是意义仅编码在表征空间中的方向上，仍是一个悬而未决的问题。一个常见的假设是LLM中的单个轴是任意的。我们在此挑战这一观点：在来自六个模型家族的12个大语言模型中，与保持相同几何结构的随机旋转轴相比，原生轴能更可靠地与其他模型的表征、人类大脑对语言的响应以及可解释的语言特征对齐。然而，原生轴远未达到最佳可实现的对齐程度，因此LLM的坐标并非完全任意，但也不是完全规范的：它们是部分规范的。关于LLM如何表征语言的理论，必须既能解释它们所学习的群体几何结构，又能解释为什么单个单元获得了在模型间重复出现的响应特征模式。

    arXiv:2611.26920v1 Announce Type: cross  Abstract: Large language models (LLMs) develop rich internal representations of language, but whether their individual coordinate axes carry meaning, or whether meaning is encoded only in directions in representation space, is still an open question. A common assumption is that individual axes in LLMs are arbitrary. We here challenge this view: across 12 LLMs from six model families, the native axes are reliably more aligned with other models' representations, with human brain responses to language, and with interpretable linguistic features than randomly rotated axes that preserve the same geometry. The native axes nonetheless fall well short of the best achievable alignment, so LLM coordinates are not fully arbitrary but also not fully canonical: they are partially canonical. A theory of how LLMs represent language must explain both the population geometry they learn and why individual units acquire patterns of response profiles that recur acr
    
[^179]: ERA：面向冷启动跨模型潜在检索的嵌入、路由与自适应方法

    ERA: Embed, Route, Adapt for Cold-Start Cross-Model Latent Retrieval

    [https://arxiv.org/abs/2611.26919](https://arxiv.org/abs/2611.26919)

    ERA提出了一种冷启动跨模型潜在检索系统，通过余弦最远点采样选取少量锚点文档、利用结合正交正则化的流形对齐损失训练轻量级适配器，并借助确定性路由器升级处理困难查询，从而无需对整个语料库重新嵌入即可将检索系统迁移到新的嵌入模型。

    

    生产级检索系统与其语料库索引所使用的嵌入模型紧密绑定。将查询迁移到不同的编码器通常需要对整个语料库重新进行嵌入，而在冷启动阶段既没有查询日志可供利用，新编码器对每个文档进行嵌入也会产生API调用或计算成本。我们提出了ERA（Embed, Route, Adapt），一个面向冷启动跨模型潜在检索的系统。ERA将少量锚点预算分配给最具信息量的文档（通过余弦最远点采样从现有目标索引中选取），使用流形对齐损失训练一个轻量级适配器——该损失将方向对齐与正交正则化相结合以防止表示坍缩——并将该适配器与一个确定性路由器配对，后者会升级处理那些决定用户感知检索质量的少量困难查询。在五个BEIR数据集上，ERA达到或超越了逐点L2和嵌入转换基线，取得了相对增益……

    arXiv:2611.26919v1 Announce Type: cross  Abstract: Production retrieval systems are tied to the embedding model that indexed their corpus. Moving queries to a different encoder normally requires re-embedding the whole corpus, and at cold start there are no query logs while every document embedded by the new encoder incurs an API or compute cost. We present ERA (Embed, Route, Adapt), a system for cold-start cross-model latent retrieval. ERA spends a small anchor budget on maximally informative documents, selected from the existing target index via Cosine Farthest Point Sampling, trains a lightweight adapter with a Manifold Alignment Loss that couples directional alignment with orthogonal regularization to prevent representation collapse, and pairs the adapter with a deterministic router that escalates the rare, hard queries that define perceived retrieval quality. Across five BEIR datasets, ERA matches or outperforms point-wise L2 and embedding-conversion baselines, with relative gains 
    
[^180]: 面向EDA脚本编写：超越人工构建经验的自进化方法

    Self-Evolving Beyond Human-Crafted Experience for EDA Scripting

    [https://arxiv.org/abs/2611.26912](https://arxiv.org/abs/2611.26912)

    ZhuLong-SE是一个以执行结果为依据的LLM智能体，通过将历史执行轨迹提炼为受真值锚定、证伪过滤、回滚和防泄漏检索等多重机制约束的三层记忆库，实现EDA脚本编写任务的跨任务自进化，从而超越依赖人工构建知识的局限。

    

    电子系统日益增长的复杂性对高效的设计流程提出了要求，然而EDA脚本编写仍然是一个瓶颈：专有API的文档稀少，错误往往表现为静默失败或晦涩难懂的异常。人工构建的知识虽然有所帮助，但随着工具链的演进却难以维护。我们提出了ZhuLong-SE（自进化），这是一个以执行结果为依据的LLM智能体，能够从以往的任务轨迹中学习可复用的知识。在ZhuLong的基础上，它通过API检索、文档响应、隔离沙箱执行和静态检查增强了任务级测试框架，并进一步引入了受约束的跨任务自进化机制。除了解决单个任务之外，ZhuLong-SE还将历史执行轨迹提炼为一个三层记忆库，该记忆库由真值锚定、证伪过滤、留出集回滚和防泄漏检索等机制加以约束管理。在包含158个任务的EDA-Eval-PyAether基准测试中，完整的任务级框架（在禁用自进化的情况下）……

    arXiv:2611.26912v1 Announce Type: new  Abstract: The growing complexity of electronic systems demands efficient design flows, yet EDA scripting remains a bottleneck: proprietary APIs are sparsely documented, and errors often appear as silent failures or cryptic exceptions. Hand-crafted knowledge helps but is difficult to maintain as toolchains evolve. We present ZhuLong-SE (Self-Evolution), an execution-grounded LLM agent that learns reusable knowledge from prior task traces. Building on ZhuLong, it enhances the task-level harness with API retrieval, documentation responses, isolated sandbox execution, and static checking, then adds governed cross-task self-evolution. Beyond solving individual tasks, ZhuLong-SE distills past execution traces into a three-tier memory store governed by ground-truth anchoring, disconfirmation filtering, held-out rollback, and anti-leakage retrieval. On the 158-task EDA-Eval-PyAether benchmark, the full task-level harness, with self-evolution disabled, ach
    
[^181]: AstronautBench：通过知识、证据与进展衡量智能体跨回合的自我改进

    AstronautBench: Measuring Agent Self-Improvement Across Episodes Through Knowledge, Evidence, and Progress

    [https://arxiv.org/abs/2611.26910](https://arxiv.org/abs/2611.26910)

    AstronautBench基于游戏《星际拓荒》构建了一个新基准，通过知识、证据支持与任务进展三个互补视角，衡量权重固定的LLM智能体在世界不断重置的循环中凭借自身经验实现跨回合自我改进的能力。

    

    LLM智能体被期望在权重固定不变的情况下，从跨回合的自身经验中实现自我改进。具身世界中的自我改进无法通过单一指标来衡量。现有基准测试分别衡量任务进展、世界知识问答（QA）和情景记忆，但一个正确答案并不能表明智能体曾经观察到支持该答案的证据，而任务进展也无法表明智能体究竟学到了什么。我们提出了AstronautBench，它通过三个视角来衡量同一个智能体运行过程中的自我改进：它知道什么、它自身获得的证据是否支持这些知识、以及它在世界中取得了多大的进展。该基准建立在游戏《星际拓荒》之上，其中每个回合是一个22分钟的循环，循环结束后世界重置，只有智能体的知识得以保留，且世界中的进展由知识所限制。在每个回合结束后，我们测量到达的地点数量和飞船日志中揭示的传闻事实数量，以及关于164个问题的回答正确性……

    arXiv:2611.26910v1 Announce Type: new  Abstract: LLM agents are expected to improve themselves from their own experience across episodes, with fixed weights. Self-improvement in an embodied world cannot be read from one measure. Existing benchmarks measure task progress, world-knowledge question answering (QA), and episodic memory, but a correct answer does not show that the agent ever observed the evidence for it, and progress does not show what the agent learned. We introduce AstronautBench, which measures one agent's self-improvement through three views on the same run: what it knows, whether its own evidence supports that knowledge, and how far it gets in the world. It is built on the game Outer Wilds, where each episode is a 22-minute loop after which the world resets, only the agent's knowledge carries over, and progress is gated by knowledge. After every episode, we measure the places reached and ship-log rumor facts revealed, the correctness of answers to 164 questions about th
    
[^182]: AI辅助的F1进站策略决策支持：深度强化学习与数学规划

    AI-Assisted Decision Support for Formula 1 Pit-Stop Strategy: Deep Reinforcement Learning and Mathematical Programming

    [https://arxiv.org/abs/2611.26909](https://arxiv.org/abs/2611.26909)

    该研究构建了基于历史比赛数据校准的F1进站策略模拟器，用五种深度强化学习算法训练策略智能体并与QMIP数学规划模型对比，发现PPO及其循环变体是在真实比赛不确定性和竞争环境下唯一能超越确定性最优规划基线的DRL方法。

    

    F1比赛中的进站策略是一个不确定性下的序贯决策问题，策略师需要在每一圈决定是否进站以及更换哪种轮胎配方。我们开发了一个数据驱动的模拟器，该模拟器使用历史比赛数据进行校准，能够捕捉轮胎磨损、车手近期表现、超车、黄旗事件以及对手的进站决策。我们使用五种深度强化学习（DRL）算法训练智能体，并将其与二次混合整数规划（QMIP）模型进行对比，后者在确定性单车手设置下是最优的。在没有不确定性和竞争的情况下，除TRPO外，所有DRL方法均能达到接近最优的表现。在完整参赛阵容和随机比赛事件的条件下，DRL方法的排名取决于评估设置：近端策略优化（PPO）及其循环变体是唯一在完全模拟比赛以及实际场景中均能优于QMIP的方法。

    arXiv:2611.26909v1 Announce Type: cross  Abstract: Pit-stop strategy in Formula 1 is a sequential decision problem under uncertainty, in which strategists decide at each lap whether to pit and which tire compound to fit. We develop a data-driven simulator, calibrated with historical race data, that captures tire degradation, drivers' recent performance, overtaking, yellow flag events, and rivals' pit-stop decisions. We train agents with five Deep Reinforcement Learning (DRL) algorithms and compare them with a Quadratic Mixed-Integer Programming (QMIP) formulation that is optimal in deterministic single-driver settings. Without uncertainty or competition, all DRL methods except TRPO achieve near-optimal performance. With a full grid of competitors and stochastic race events, the ranking of DRL methods depends on the evaluation setting: Proximal Policy Optimization (PPO) and its recurrent variant are the only methods that improve on the QMIP both in fully simulated races and when the act
    
[^183]: 基于异构大语言模型求解器的预算感知预测-验证定理证明

    Budget-Aware Predict-and-Verify Theorem Proving with Heterogeneous LLM Solvers

    [https://arxiv.org/abs/2611.26906](https://arxiv.org/abs/2611.26906)

    本文提出BudgetPV，一种预算感知的预测-验证智能体系统，利用跨求解器预测器在异构LLM求解器之间动态分配计算资源，并根据证明进程的不同阶段（新子目标的求解器选择与已尝试子目标的算力追加）调整策略，从而在保证证明成功率的同时最小化定理证明的计算成本。

    

    大语言模型（LLM）定理求解器在证明能力和推理成本上各不相同。将昂贵的求解器应用于每个子目标，在较便宜的模型已足够时会浪费计算资源；而总是从最便宜的求解器开始，则会在需要更强模型的子目标上浪费尝试次数。我们提出了BudgetPV，这是一个智能体式的预算感知“预测-验证”系统，可在未解决的子目标和异构LLM求解器之间分配计算。BudgetPV使用跨求解器预测器来指导每个子目标的求解器选择，并随着验证反馈的获取不断修正分配方案。随着证明的推进，分配问题也在不断变化：早期决策主要确定应由哪个求解器处理尚未尝试过的子目标，而后期决策则确定哪些先前尝试过的子目标应获得额外的计算。我们针对这两个阶段使用两个目标函数来训练预测器。类别加权交叉熵……（原文摘要在此处截断）

    arXiv:2611.26906v1 Announce Type: new  Abstract: Large language model (LLM) theorem solvers differ in both proving capability and inference cost. Applying an expensive solver to every subgoal wastes computation when cheaper models are sufficient, while always starting from the cheapest solver wastes attempts on subgoals that require stronger models. We present BudgetPV, an agentic budget-aware predict-and-verify system that allocates computation across unresolved subgoals and heterogeneous LLM solvers. BudgetPV uses a cross-solver predictor to guide solver selection for each subgoal and revises the allocation as verification feedback becomes available.   The allocation problem changes as proving progresses. Early decisions mainly determine which solver should handle an untried subgoal, while later decisions determine which previously attempted subgoals should receive additional computation. We train the predictor with two objectives for these two stages. Class-weighted cross-entropy im
    
[^184]: 迈向鲁棒的部分编辑深度伪造定位：一种基于编辑与生成器条件的分析

    Towards Robust Partial-Edit Deepfake Localization: An edit and Generator Conditioned Analysis

    [https://arxiv.org/abs/2611.26903](https://arxiv.org/abs/2611.26903)

    本文首次对部分编辑深度伪造定位进行了基于编辑类型与生成器条件的细粒度分析，发现删除操作——尤其是SSR-Speech生成的删除——是定位中最具挑战性的条件。

    

    部分编辑深度伪造定位旨在识别原本真实的语音中被篡改的区域。近期的系统报告了强劲的总体结果，但聚合分数无法揭示这种性能在不同编辑条件下是否依然成立。在本工作中，我们在PartialEdit的VoiceCraft和SSR-Speech子集上，使用WavLM-Large和XLS-R表示，对分段感知定位（SAL）进行了基于编辑与生成器条件的分析。我们沿三个由转录文本派生的维度对编辑进行分类，即内容、意义变化和编辑操作，并分别评估每个生成器。尽管两种表示都取得了强劲的总体性能，但在SSR-Speech上的定位明显较弱。实验发现，删除是最困难的操作，而由SSR-Speech生成的删除对两种表示而言都是最具挑战性的条件。相比之下，编辑内容、意义变化和句子（原文摘要在此处截断）……

    arXiv:2611.26903v1 Announce Type: new  Abstract: Partial-edit deepfake localization aims to identify manipulated regions within otherwise authentic speech. Recent systems report strong overall results, but aggregate scores do not reveal whether this performance holds across different editing conditions. In this work, we present an edit- and generator-conditioned analysis of Segment-Aware Localization (SAL) using WavLM-Large and XLS-R representations on the VoiceCraft and SSR-Speech subsets of PartialEdit. We categorize edits along three transcript-derived axes, namely content, meaning change, and editing operation, and evaluate each generator separately. While both representations achieve strong overall performance, localization is noticeably weaker on SSR-Speech. From our experiments, we found that the deletion is the hardest operation, and deletions produced by SSR-Speech are the most challenging condition for both representations. In contrast, edit content, meaning change, and sente
    
[^185]: 感知一次，按需升级：将视觉KV记忆从小型VLM传递给大型VLM

    Perceive Once, Escalate on Demand: Handing Off Visual KV Memory from Small to Large VLMs

    [https://arxiv.org/abs/2611.26902](https://arxiv.org/abs/2611.26902)

    提出Baton方法，通过从大型模型自身答案中无标签蒸馏逐层线性映射，将小型VLM的视觉KV缓存记忆传递给大型VLM，避免升级时重复编码历史信息，使大型模型只需编码当前帧。

    

    具身智能体无法在长视频的每一步都负担得起大型视觉语言模型（VLM），因此级联系统持续运行一个小型VLM，仅将不确定的决策升级给大型模型。然而，每次升级都迫使大型模型重新编码小型模型已经感知过的历史信息，这占据了其主要成本。我们提出了Baton，它将小型模型的视觉记忆——即其在历史信息上的键值（KV）缓存——交给大型模型。Baton基于一个发现：大型模型需要一个可用的记忆，而不是自身缓存的忠实副本。一个训练用于重现大型模型缓存的映射虽然与它最为接近，但相比单独使用小型模型却没有任何收益。Baton转而从大型模型自身的答案中无标签地蒸馏出一个逐层线性映射，大型模型只需编码当前帧。对该映射的干预实验将增益追溯到由大型模型中间三分之一层所携带的一个方向。

    arXiv:2611.26902v1 Announce Type: cross  Abstract: An embodied agent cannot afford a large vision-language model (VLM) at every step of a long video, so a cascade runs a small VLM continuously and escalates only uncertain decisions to a large one. Each escalation, however, forces the large model to re-encode a history the small model has already perceived, and this dominates its cost. We present Baton, which hands the large model the small model's visual memory, its key-value (KV) cache over the history. Baton rests on one finding: the large model needs a usable memory, not a faithful copy of its own. A map trained to reproduce the large model's cache matches it most closely, yet brings no gain over the small model alone. Baton instead distills a per-layer linear map, without labels, from the large model's own answers, with the large model encoding only the current frame. Interventions on the map trace the gain to one direction, carried by the middle third of the large model's layers. 
    
[^186]: 大规模归因风格建模：用于心理归因评估与重构的数据集及分析

    Modeling Attributional Style at Scale: A Dataset and Analysis for Psychological Attribution Assessment and Reframing

    [https://arxiv.org/abs/2611.26900](https://arxiv.org/abs/2611.26900)

    该论文提出了首个大规模归因风格基准数据集ASTD及配套评估指标，包含42,000个真实事件和七种心理归因风格的归因标注，实现了归因风格的可扩展评估与认知重构。

    

    根据习得性无助理论的修正版本，反复暴露于不可控的负面事件可能催生抑郁性归因风格，增加罹患抑郁症的易感性，但这仍是认知治疗可干预的靶点。然而，归因认知的计算研究一直受到缺乏大规模数据集和稳健评估协议的制约。在本工作中，我们提出了归因风格迁移数据集（ASTD）及配套的专门评估指标，这是首个旨在大规模建模、评估和重构归因解释的基准数据集。ASTD通过整合基于大语言模型（LLM）的生成与专家验证的“预防-过滤-验证”流程构建而成，包含42,000个真实世界事件，并配有涵盖七种风格、具有心理学依据的归因。利用该数据集，我们解决了两个关键挑战：(1) 基于监督方法的可扩展归因风格评估……

    arXiv:2611.26900v1 Announce Type: cross  Abstract: According to the reformulated Learned Helplessness theory, repeated exposure to uncontrollable negative events can foster a depressogenic attributional style, increasing susceptibility to depression yet remaining a tractable target for cognitive therapy. Computational research on attributional cognition, however, is hampered by the lack of large-scale datasets and robust evaluation protocols. In this work, we introduce the Attributional Style Transfer Dataset (ASTD) along with dedicated evaluation metrics, the first benchmark designed to model, assess, and reframe attributional explanations at scale. Constructed via a Prevent-Filter-Validate pipeline that integrates LLM-based generation with specialist validation, ASTD contains 42,000 real-world events paired with psychologically grounded attributions spanning seven styles. Using this dataset, we address two key challenges: (1) scalable assessment of attributional style via both superv
    
[^187]: ProbeAgent：可验证环境中智能体失败的原理性诊断

    ProbeAgent: Principled Diagnosis of Agent Failures in Verifiable Environments

    [https://arxiv.org/abs/2611.26897](https://arxiv.org/abs/2611.26897)

    ProbeAgent是一个自动化诊断框架，通过构建经过验证的可执行测试、生成保留疑似失败原因的变体任务并观察错误是否复现，来系统性发现、验证并归类智能体的共有失败模式。

    

    改进智能体需要发现它们在哪里失败，并理解失败的原因。智能体在不断变化的环境中执行多步操作，因此诊断其失败不能仅仅依靠检查大语言模型（LLM）的最终响应。同样的不成功结果可能源于任务本身的难度，也可能源于智能体在多个任务中重复犯下的错误。我们提出ProbeAgent，一个自动化框架，用于发现智能体的失败、测试所推测的失败原因，并识别共同的失败模式。从基准任务出发，ProbeAgent构建可执行的测试，并利用参考解答、自动化检查以及对任务需求的审查来验证这些测试。对于每条失败的轨迹，它会解释错误原因，并构建新的任务——这些任务在改变周围上下文的同时保留疑似的原因。在这些任务上运行智能体，可以检验相同的错误是否会再次出现。随后，ProbeAgent将具有相似原因的失败进行分组，并生成……

    arXiv:2611.26897v1 Announce Type: cross  Abstract: Improving agents requires discovering where they fail and understanding why. An agent acts over multiple steps in a changing environment, so diagnosing its failures requires more than inspecting the final response of a large language model (LLM). The same unsuccessful outcome can arise from task difficulty or from an error the agent repeats across tasks. We introduce ProbeAgent, an automated framework that discovers agent failures, tests their proposed causes, and identifies shared failure modes. Starting from benchmark tasks, ProbeAgent builds executable tests and validates them using reference solutions, automated checks, and review of the task requirements. For each failed trajectory, it explains the error and constructs new tasks that preserve the suspected cause while changing the surrounding context. Running the agent on these tasks tests whether the same error recurs. ProbeAgent then groups failures with similar causes and gener
    
[^188]: 你的评审器会继承你的提示词：LLM即评审中的上下文标签偏置是有方向性的，而非对称的

    Your Judge Inherits Your Prompt: In-Context Label Bias in an LLM-as-Judge Is Directional, Not Symmetric

    [https://arxiv.org/abs/2611.26881](https://arxiv.org/abs/2611.26881)

    研究表明，多模态视频问答评估流水线中的LLM评审器会从其少样本示例中单向继承标签偏置：当示例偏向"Yes"时其接受率骤升约20–30个百分点并伴随单侧错误，而当示例平衡或偏向"No"时则接近人类共识，证明这种上下文标签偏置是方向性的而非对称的。

    

    LLM-as-judge（以大语言模型作为评审器）已成为对开放式模型输出进行评分的标准方法，然而评审器本身是一个由少量上下文示例引导的冻结语言模型，而此类模型已知会吸收其演示示例中的标签分布，而非将其视为中性的示例。我们探究这种标签偏置是否会传导到多模态（视频问答）评估流水线中使用的LLM评审器，以及它以何种形式呈现。通过使用完全冻结的生成器-评审器流水线，我们保持生成的答案不变，仅在四个演示集之间改变评审器少样本示例的Yes/No（是/否）平衡。每个判定结果均与每个数据集200个样本上的双人标注人类共识进行对照验证。当评审器的示例为平衡或偏向拒绝时，其接受率保持接近人类共识；但当示例偏向接受时，接受率急剧跳升（在NExT-QA上约30个百分点，在MSVD-QA上约20个百分点，且呈现强烈单侧的错误特征）……（原文摘要在此处截断）

    arXiv:2611.26881v1 Announce Type: cross  Abstract: LLM-as-judge has become a standard way to score open-ended model outputs, yet the judge is itself a frozen language model steered by a few in-context examples, and such models are known to absorb the label balance of their demonstrations rather than treat them as neutral illustrations. We ask whether this label bias reaches an LLM judge used inside a multimodal (video question answering) evaluation pipeline, and what form it takes. Using a fully frozen generator-judge pipeline, we hold the generated answers fixed and vary only the Yes/No balance of the judge's few-shot examples across four demonstration sets. Each verdict is validated against a two-annotator human consensus on 200 examples per dataset. The judge's acceptance rate stays close to human consensus when its examples are balanced or skewed toward rejection, but jumps sharply (by about 30 points on NExT-QA and 20 on MSVD-QA, with a strongly one-sided error profile) when the e
    
[^189]: 超越人口统计学歧视：大语言模型中规范-描述校准的基准测试

    Beyond Demographic Discrimination: Benchmarking Normative-Descriptive Calibration in LLMs

    [https://arxiv.org/abs/2611.26870](https://arxiv.org/abs/2611.26870)

    该论文提出“规范-描述校准”这一新概念及包含697个匹配对场景和三个指标（NER、NLR、NDS）的NDCB基准，首次系统衡量大语言模型在“预测结果”与“决策结果”两种任务框架下对相同候选者处理是否一致。

    

    大语言模型（LLM）越来越多地被部署在具有社会影响的场景中，在这些场景中，它们必须区分规范性推理（应该发生什么）和描述性推理（可能发生什么）。大多数公平性基准仅在单一推理目标下评估人口统计学歧视，而据我们所知，没有任何基准检验模型在被要求“描述”结果与“决定”结果时是否会对同一项目做出不同的处理。我们提出了“规范-描述校准”这一概念，即模型对匹配候选者的区分在多大程度上取决于它是被要求预测还是决策，并通过规范-描述校准基准（NDCB）对其进行测量：该基准包含六个领域的697个匹配对场景，每个场景均以两种框架提出，并采用三个指标：规范参与率（NER）、规范泄漏率（NLR）和规范-描述分离度（NDS）。在六个大语言模型上，NDS能够区分15个（模型-场景组合）中的13个……（原文摘要在此处被截断）

    arXiv:2611.26870v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in socially consequential settings, where they must distinguish between normative reasoning (what should happen) and descriptive reasoning (what is likely to happen). Most fairness benchmarks evaluate demographic discrimination under a single reasoning objective, and none we know of tests whether a model treats the same item differently when asked to describe an outcome versus decide one. We introduce normative--descriptive calibration, the extent to which a model's differentiation between matched candidates depends on whether it is asked to predict or to decide, and measure it with the Normative--Descriptive Calibration Benchmark (NDCB): 697 matched-pair scenarios across six domains, each posed under both framings, and three metrics, Normative Engagement Rate (NER), Normative Leakage Rate (NLR), and Normative--Descriptive Separation (NDS). Across six LLMs, NDS separates 13 of 15 mod
    
[^190]: 相干矩传播：面向预训练机器人策略的认知不确定性

    Coherent Moment Propagation: Epistemic Uncertainty for Pre-trained Robot Policies

    [https://arxiv.org/abs/2611.26855](https://arxiv.org/abs/2611.26855)

    本文证明BayesDiff在流匹配与扩散ODE中会因总方差衰减而严重低估认知不确定性，并提出相干矩传播方法，为预训练机器人策略提供可信的认知不确定性量化。

    

    流匹配策略是机器人学习领域的最先进技术，在从长时程规划到灵巧操作等任务中都取得了令人瞩目的成果。然而，这些策略无法提供其预测指令是否可信的度量，尤其是在训练集之外的新颖情境中。认知不确定性反映了关于模型权重的不确定性，能够量化这种无知，但在推理阶段从流匹配策略中计算认知不确定性并非易事。由于动作由ODE求解器生成，需要将不确定性传播贯穿整个积分过程。BayesDiff通过在最后一层Laplace后验近似下，沿去噪链传播状态的均值和方差，为扩散模型解决了这一问题，但它会严重低估认知不确定性。在本文中，我们：i）证明，对于流匹配和扩散ODE，BayesDiff的总方差会衰减……

    arXiv:2611.26855v1 Announce Type: cross  Abstract: Flow matching policies underpin the state-of-the-art in robot learning, with impressive results ranging from long-horizon planning to dexterous manipulation. However, they provide no measure of whether their predicted commands can be trusted, particularly in novel situations outside their training set. Epistemic uncertainty reflects uncertainty about the model's weights and quantifies this ignorance, but calculating it at inference time from a flow matching policy is not trivial. Actions are produced by an ODE solver, requiring uncertainty to be propagated through the entire integration process. BayesDiff addresses this problem for diffusion models by propagating the mean and variance of the state through the denoising chain under a last-layer Laplace posterior approximation, but it severely underestimates the epistemic uncertainty. In this paper we: i) prove that, for flow matching and diffusion ODEs, BayesDiff's total variance decays
    
[^191]: LoCo-WAM：将测试时训练作为世界-动作模型的长上下文

    LoCo-WAM: Test-Time Training as Long Context for World-Action Models

    [https://arxiv.org/abs/2611.26853](https://arxiv.org/abs/2611.26853)

    LoCo-WAM 通过测试时训练利用新观测帧更新视频专家中的低秩适配器，使世界-动作模型在不延长观测窗口、无需专家动作标签、额外数据或元训练的情况下从更长的经验流中学习，从而在四个仿真基准上显著提升任务成功率。

    

    世界-动作模型通过预测接下来会发生什么来指导机器人动作，但通常仅依赖于较短的观测历史。扩展这一历史会增加每次决策时的计算量，并且可能需要单独训练一个模型。我们提出了 LoCo-WAM，它在保持相同短历史长度的同时，通过测试时训练从自身动作的结果中学习。它利用新观测到的帧来更新视频专家中的低秩适配器，其适配后的预测用于指导后续动作。这些更新在每个回合内累积，并在回合之间重置，使过去的观测在离开输入窗口之后仍能影响决策。因此，LoCo-WAM 无需专家动作标签、额外数据或元训练，即可从更长的经验流中学习。在两个世界-动作模型上的实验表明，该方法在四个仿真基准上均提升了任务成功率。在 RoboCasa GR-1 上，一个仅有 2.4 秒历史的模型能够匹配……（原文摘要在此处截断）

    arXiv:2611.26853v1 Announce Type: cross  Abstract: World-action models guide robot actions by predicting what happens next, but typically condition on a short observation history. Extending this history increases computation at every decision and can require a separately trained model. We introduce LoCo-WAM, which learns from the outcomes of its actions through test-time training while keeping the same short history length. It uses newly observed frames to update a low-rank adapter in the video expert, whose adapted predictions guide subsequent actions. The updates accumulate within each episode and are reset between episodes, allowing past observations to influence decisions after they leave the input window. LoCo-WAM thus learns from a longer stream of experience without expert action labels, additional data or meta-training. Experiments on two world-action models show improved task success across four simulation benchmarks. On RoboCasa GR-1, a model with 2.4,s of history matches one
    
[^192]: Scoliosis5K：基于片段条件化的RGB-轮廓融合的多模态步态分析

    Scoliosis5K: Multimodal Gait Analysis via Clip-Conditioned RGB-Silhouette Fusion

    [https://arxiv.org/abs/2611.26849](https://arxiv.org/abs/2611.26849)

    该论文构建了多中心多模态步态数据集Scoliosis5K，并提出片段条件化的RGB-轮廓融合架构，用于脊柱侧凸相关步态的参与者级分类。

    

    彩色行走视频在记录人体边界运动的同时，也包含衣物褶皱、表面外观和录制环境等线索。为了考察当分类器已经观察到配对的彩色序列时，显式的二值轮廓对参与者级别的脊柱侧凸相关步态分类有何贡献，我们构建了Scoliosis5K数据集——一个多中心采集的数据集，包含3,010名参与者的4,998条走廊行走序列，其中RGB人体裁剪图、二值轮廓和二维姿态共享相同的录制标识符与帧索引。我们的架构利用每个图像块像素相对于其片段均值的偏差，在预训练的RGB编码器内对两个低秩的查询和值更新进行加权，并将所得的视频表示与轮廓Transformer序列以及从每个窗口中一帧原始分辨率RGB帧提取的区域核岭回归分数进行融合。在五个参与者不相交的交叉验证折中，对988名参与者进行了评估……（摘要原文在此处截断）

    arXiv:2611.26849v1 Announce Type: cross  Abstract: Color walking video records body boundary motion alongside clothing folds, surface appearance, and recording-environment cues. To examine what an explicit binary silhouette contributes to participant-level classification of scoliosis-related gait when a classifier already observes the paired color sequence, we construct Scoliosis5K, a multi-site collection of 4,998 corridor-walking sequences across 3,010 participant identifiers in which RGB person crops, binary silhouettes, and two-dimensional poses share each recording identifier and frame index. Our architecture uses each patch's pixel deviation from its clip mean to weight two low-rank query and value updates inside a pretrained RGB encoder, fusing the resulting video representation with a silhouette transformer sequence and regional kernel ridge scores extracted from one native-resolution RGB frame per window. Across 988 participants in five participant-disjoint cross-validation fo
    
[^193]: 面向语言模型前馈层的非扩展门控连分数架构

    Non-Expanding Gated Continued Fraction Architecture for Feedforward Layers in Language Models

    [https://arxiv.org/abs/2611.26835](https://arxiv.org/abs/2611.26835)

    本文提出一种非扩展门控连分数架构组件，可替换语言模型中的各类前馈层（包括MLP、GLU及MoE专家），并用对数和指数重新实现除法运算，从而提升性能并简化与标准预训练流程的集成。

    

    arXiv:2611.26835v1 公告类型：交叉 摘要：最近的研究表明，受连分数启发的神经架构在语言建模中可以做到高度参数高效且性能优异。在本工作中，我们重新审视了基于连分数的通道混合方法，旨在提升其性能，并简化其与语言模型标准预训练协议的集成。我们通过两个简单而关键的贡献来实现这一目标：i) 我们提出了一种新颖的非扩展门控连分数启发的架构组件，用于替换现有语言模型架构中的前馈层，这些架构可以采用普通多层感知机、门控线性单元，或混合专家架构中的（普通/门控）专家。ii) 我们提出了一种用对数和指数重新实现除法运算的方法——除法正是这类架构中的非线性来源。通过在 GPT-2、Llama-3、Mamba- 等不同语言建模架构上进行实验……

    arXiv:2611.26835v1 Announce Type: cross  Abstract: Recent work has shown that continued fraction inspired neural architectures (CoFrGeNets) can be highly parameter efficient yet performant for language modeling. In this work, we revisit continued fraction-based channel mixing with the goal of improving its performance and simplifying integration into standard pre-training protocols for language models. We achieve this by two simple yet critical contributions: i) we propose a novel non-expanding gated continued fraction inspired architectural component to replace feedforward layers in established language model architectures that may use vanilla multilayer perceptrons or gated linear units or (vanilla/gated) experts in mixture-of-experts (MoE) type architectures. ii) We propose a way to re-implement divisions -- the non-linearity in these architectures -- in terms of logarithms and exponentials. By experimenting on different language modeling architectures such as GPT-2, Llama-3, Mamba-
    
[^194]: 重新思考结合领域与任务适配器缩放的全权重LoRA：面向葡萄牙语钻井报告的问答任务

    Rethinking Full-Weight LoRA with Domain and Task Adapter Scaling for Question Answering over Portuguese Well Drilling Reports

    [https://arxiv.org/abs/2611.26833](https://arxiv.org/abs/2611.26833)

    该研究提出在训练后对领域适配与任务特定的LoRA适配器进行缩放、并可用互补权重将二者组合，从而在不损害指令遵循能力的前提下提升大语言模型对葡萄牙语钻井报告问答任务的性能。

    

    将大语言模型（LLM）适配至专业领域仍然具有挑战性，特别是针对葡萄牙语钻井报告这类技术文档，其标注数据稀缺，且领域适配可能损害模型原有的指令遵循能力。尽管低秩适应（LoRA）被广泛用于参数高效的适配，但其在训练后的缩放策略仍未得到充分探索。本研究使用Qwen2.5-7B和Llama-3.1-8B，探究了LoRA适配器缩放对葡萄牙语钻井报告问答任务的影响。领域自适应预训练（DAPT）既可直接在指令微调模型上进行，也可在其基础模型上进行后再进行适配器迁移；而任务特定适配器则在WellQuest上训练，这是一个源自葡萄牙语钻井报告的开源问答数据集。此外，这两种适配器还可通过互补权重进行组合。缩放因子的范围从……

    arXiv:2611.26833v1 Announce Type: cross  Abstract: Adapting large language models (LLMs) to specialized domains remains challenging, particularly for technical documents such as Portuguese well drilling reports, where annotated data are scarce and domain adaptation may compromise instruction-following capabilities. Although Low-Rank Adaptation (LoRA) is widely used for parameter-efficient adaptation, its scaling after training remains insufficiently explored. This study investigates the impact of LoRA adapter scaling on question answering over Portuguese drilling reports using Qwen2.5-7B and Llama-3.1-8B. Domain-adaptive pretraining (DAPT) is performed either directly on instruction-tuned models or on their base counterparts, with subsequent adapter transfer, while task-specific adapters are trained on WellQuest, an open-source question-answering dataset derived from Portuguese drilling reports. The two adapters are also combined using complementary weights. Scaling factors ranging fro
    
[^195]: 表格数据驱动模型的现实鲁棒性评估

    Realistic Robustness Evaluation of Tabular Data Driven Models

    [https://arxiv.org/abs/2611.26818](https://arxiv.org/abs/2611.26818)

    本文提出了AURORA，一个模块化开源工具，通过在统一框架中整合多种对抗扰动方法和评估指标，为表格数据分类的机器学习模型提供系统化的鲁棒性评估。

    

    机器学习（ML）模型在高风险应用中的日益普及，引发了人们对其抵御对抗性攻击鲁棒性的严重担忧。尽管大量研究集中于图像和文本领域的对抗样本生成，但表格数据虽然在现实世界系统中被广泛使用，其对抗样本的研究仍相对有限。本工作介绍了AURORA，一个用于评估ML模型在表格数据分类任务上鲁棒性的模块化开源工具。它在一个统一框架内整合了多种对抗扰动方法和评估指标，以系统地评估模型在对抗条件下的行为。AURORA的核心是其评估报告，该报告专注于鲁棒性分析，通过量化性能下降、比较不同扰动策略的影响，以及识别模型在各特征和输入区域上的脆弱性来展开分析。

    arXiv:2611.26818v1 Announce Type: cross  Abstract: The increasing adoption of Machine Learning (ML) models in high-stakes applications has raised critical concerns regarding their robustness against adversarial attacks. While extensive research has focused on adversarial example generation in image and text domains, the study of such examples in tabular data remains comparatively limited, despite its widespread use in real-world systems. This work introduces AURORA, a modular, open-source tool for evaluating the robustness of ML models on tabular data classification. It integrates multiple adversarial perturbation methods and evaluation metrics within a unified framework to systematically assess model behavior under adversarial conditions. Central to AURORA is its evaluation report, which focuses on robustness analysis by quantifying performance degradation, comparing the impact of different perturbation strategies, and identifying model vulnerabilities across features and input region
    
[^196]: 从一开始就学会搜索：面向语言模型的搜索原生预训练

    Search From the Start: Search-Native Pretraining for Language Models

    [https://arxiv.org/abs/2611.26809](https://arxiv.org/abs/2611.26809)

    提出“搜索原生预训练”方法，通过大规模地将网络搜索查询及检索到的网页响应注入预训练语料，让语言模型在预训练阶段就学会发起搜索并利用检索证据，从而在知识密集型任务上取得更优性能，且该优势在监督微调后依然保持。

    

    现代语言模型（LM）越来越多地在推理阶段配备搜索工具进行部署。然而，预训练仍然主要聚焦于对静态文本的预测，搜索能力通常是在后训练阶段才被引入。我们提出“搜索原生预训练”作为一种实用路径，在预训练阶段就基于普通文本语料教会模型搜索能力。具体而言，我们大规模地用网络搜索查询来增强预训练语料，并使用本地索引检索网络响应。在这种增强语料上的训练使模型学会生成搜索调用，并在语言建模过程中融入检索到的证据。在Wikipedia和DCLM-Edu语料库以及360M至3.3B的参数规模上，搜索预训练模型在知识密集型任务上表现出更高的性能，并且对检索到的证据具有更强的可控性（可引导性）。关键的是，在对搜索预训练模型和常规模型进行监督微调之后，它们的优势依然得以保持。

    arXiv:2611.26809v1 Announce Type: cross  Abstract: Modern language models (LMs) are increasingly deployed with access to search tools at inference. Yet pretraining remains predominantly focused on predicting static text, with search capabilities typically introduced during post-training. We propose Search-Native Pretraining as a practical route to teach search capability during pretraining from ordinary text corpora. Specifically, we augment pretraining corpus at scale with web-search queries and retrieve web response using a local index. Training on this augmented corpus teaches models to generate search calls and incorporate retrieved evidence during language modeling. Across Wikipedia and DCLM-Edu corpora and 360M to 3.3B parameter scales, search-pretrained models ("QueryLM") show improved performance on knowledge-intensive tasks and higher steerability toward the retrieved evidence. Crucially, their advantage persists after supervised fine-tuning of both search-pretrained and conve
    
[^197]: MedicalRSI：通过临床模型-框架协同进化实现医疗递归自我改进智能体

    MedicalRSI: Medical Recursive Self-Improving Agents via Clinical Model-Harness Co-Evolution

    [https://arxiv.org/abs/2611.26806](https://arxiv.org/abs/2611.26806)

    该论文提出MedicalRSI框架，通过临床模型-框架协同进化使医疗智能体能够同时改进自身及其改进策略，从而实现递归自我改进，避免同类错误在不同患者间反复出现。

    

    基于大语言模型（LLM）构建的医疗智能体发展迅速，但大多数智能体在部署后便保持固定，不会持续利用自身经验来改进后续的尝试。因此，同类错误可能会在不同患者身上反复出现。为解决这一问题，现有的自我改进方法会基于智能体自身的经验来更新其权重或医疗框架，但其改进策略本身保持固定，从不学习哪些改变真正有效，这限制了智能体可提升的上限。递归自我改进（RSI）则同时改进智能体本身及其改进策略。然而，为医疗智能体实现RSI具有挑战性，因为临床工作流程很少配有参考框架，且某个能修复早期错误的改变未必能修复智能体后续遇到的问题。我们提出了MedicalRSI，一个建立在临床模型-框架协同进化之上的RSI框架，其中一个模型扮演两个角色。Solver（求解器）是医疗智能体……

    arXiv:2611.26806v1 Announce Type: cross  Abstract: Medical agents built on large language models (LLMs) have advanced rapidly, yet most stay fixed once deployed and do not keep using their experience to improve later attempts. As a result, the same kind of error can recur across patients. To address this, existing self-improvement methods update an agent's weights or its medical harness from its own experience, but their improving strategy stays fixed and never learns which changes actually help, which caps how much the agent can improve. Recursive self-improvement (RSI) instead improves both the agent and its improving strategy. However, achieving RSI for medical agents is challenging, because clinical workflows rarely come with a reference harness, and a change that fixes early errors may not fix the problems the agent meets later. We propose MedicalRSI, an RSI framework built on Clinical Model-Harness Co-Evolution, in which one model plays two roles. The Solver is the medical agent 
    
[^198]: 生成模型中的记忆与泛化：基于能量建模与联想记忆的视角

    Memorization and Generalization in Generative Models: A Perspective From Energy-based Modeling and Associative Memory

    [https://arxiv.org/abs/2611.26797](https://arxiv.org/abs/2611.26797)

    本文从能量建模与联想记忆的视角提出：扩散模型本质上是一种联想记忆，随着数据增长而涌现的“伪状态”吸引子正是生成能力的开端，因此记忆与泛化可以在同一检索过程中共存。

    

    生成模型如今已能编写可运行的代码、生成逼真的图像并进行长对话，但其智能背后的机制仍未被我们理解。本论文通过联想记忆（Associative Memory, AM）来探索这一谜题：在联想记忆中，模式以能量极小值的形式被存储，并通过吸引子动力学进行检索。除了存储容量之外，AM 还会形成一类被称为“伪状态”的新极小值，这长期以来被视为记忆的失败。我们提出，这种“失败”正是创造力的起点，记忆与泛化可以作为同一检索过程的两种结果而共存。首先，我们证明扩散模型表现为一种联想记忆。当数据很少时，扩散模型会在每个训练样本周围形成各自独立的吸引盆；随着数据量增长，这些吸引盆逐渐缩小，而伪状态作为融合了已存储样本却不匹配任何单个样本的吸引子开始涌现——它们正是生成能力的最初征兆。我们通过……刻画了这一转变（原文摘要在此处截断）

    arXiv:2611.26797v1 Announce Type: new  Abstract: Generative models now write working code, produce realistic images, and hold long conversations, yet the mechanisms behind their intelligence still evade our understanding. This thesis explores that puzzle through Associative Memory (AM), in which patterns are stored as energy minima and retrieved through attractor dynamics. Beyond their capacity, AMs form new minima known as spurious states, long regarded as a failure of memory. We argue that this failure is where creativity begins, and that memorization and generalization can coexist as outcomes of one retrieval process. First, we show that diffusion models behave as AMs. With little data, a diffusion model forms a distinct basin of attraction around each training sample. As the data grows, these basins shrink, and spurious states emerge as attractors that blend stored samples yet match none of them. They are the first signs of generative capability. We characterize this transition by 
    
[^199]: 超越最终答案：基于H&E与mIF的可解释肿瘤免疫微环境推理

    Beyond Final Answers: Interpretable Tumor Immune Microenvironment Reasoning from H&E and mIF

    [https://arxiv.org/abs/2611.26794](https://arxiv.org/abs/2611.26794)

    该论文提出HERMES，利用同一切片的H&E与mIF图像提供免人工标注的细胞级真值，使多模态大语言模型的肿瘤免疫微环境推理获得步骤级可验证的可解释监督，而非仅验证最终答案。

    

    肿瘤免疫微环境（TIME）的读出指标，如CD8+ T细胞密度、上皮内浸润和Ki67指数，需要结合来自H&E的组织结构信息与来自多重免疫荧光（mIF）的细胞表型信息。多模态大语言模型（MLLM）有望通过给出解释来自动化这些读出指标，但目前仍存在两个空白：其一，现有病理学MLLM仅读取H&E，而这些读出指标依赖mIF才能测量的细胞表型；其二，其推理过程缺乏步骤级的真值可供验证，因此R1式强化学习只奖励最终答案，推理过程本身未被验证。同一切片的H&E与mIF恰好填补了这两个空白：mIF无需人工标注即可测量每一个细胞，从而为临床指南所规定的每个步骤——从描绘组织区室到计数并标准化其中的标志物阳性细胞——提供细胞级的真值。基于此，我们提出了HERMES（H&E-mIF推理……）。

    arXiv:2611.26794v1 Announce Type: new  Abstract: Readouts of the tumor immune microenvironment (TIME), such as CD8+ T-cell density, intra-epithelial infiltration and the Ki67 index, combine tissue architecture from H&E with cell phenotypes from multiplex immunofluorescence (mIF). Multimodal large language models (MLLMs) could automate these readouts with an explanation, but two gaps remain. Pathology MLLMs read H&E alone, while these readouts need the cell phenotypes that mIF measures. And their reasoning has no step-level truth to be checked against, so R1-style reinforcement learning rewards only the final answer and leaves the reasoning unverified. Same-section H&E and mIF provide what both gaps lack: mIF measures every cell without manual annotation, which gives cell-level ground truth for each step that clinical guidelines prescribe, from delineating a tissue compartment to counting and normalizing the marker-positive cells within it. We therefore present HERMES (H&E-mIF Reasoning
    
[^200]: ChatAtom：用于分子动力学模拟与可视化的多智能体大语言模型系统

    ChatAtom: A Multi-Agent LLM System for Molecular Dynamics Simulation and Visualization

    [https://arxiv.org/abs/2611.26782](https://arxiv.org/abs/2611.26782)

    ChatAtom是一个多智能体大语言模型系统，可将自然语言请求自动转化为完整的分子动力学模拟工作流程，涵盖方案规划、脚本生成与审核、高性能计算任务分发以及结果分析与可视化，无需用户编写任何代码。

    

    分子动力学（MD）模拟可以解析原子尺度的运动，并预测材料的结构、力学与动力学性质，因而是材料设计与发现的核心手段。然而，一项MD研究通常需要专家驱动的工作流程：设计模拟方案、编写可靠的输入脚本、执行模拟，并借助多种独立工具分析结果。本文提出ChatAtom，这是一个多智能体大语言模型系统，能够将自然语言请求转化为完整的MD工作流程，用户无需编写输入脚本或分析代码。每个智能体在工作流程中承担明确的职责：编排者负责规划整个研究；LAMMPS与OVITO智能体各自采用“生成器-审查器”配对机制，在执行前准备并审核模拟与可视化脚本；高性能计算智能体负责将作业分发至远程计算资源；分析智能体则解读已完成的模拟运行结果。该工作流程可访问……

    arXiv:2611.26782v1 Announce Type: new  Abstract: Molecular dynamics (MD) simulation resolves atomic-scale motion and predicts the structural, mechanical, and dynamic properties of materials, making it cen- tral to materials design and discovery. Yet an MD study requires an expert-driven workflow: designing the simulation, writing reliable input scripts, executing them, and analyzing the results with separate tools. Here we present ChatAtom, a multi-agent large language model system that translates a natural-language request into a complete MD workflow without the user writing input scripts or analysis code. Each agent has a defined responsibility within the workflow. An orchestrator plans the study; the LAMMPS and OVITO agents each use a generator-reviewer pair to prepare and review simulation and visualization scripts before execution; a high-performance computing agent dispatches jobs to remote resources; and an analysis agent interprets the completed run. The work- flow is accessibl
    
[^201]: 什么能帮助语言模型代理审阅暴力视频？与人类证据参照的受控比较

    What Helps Language-Model Agents Review Violent Video? A Controlled Comparison with a Human Evidence Reference

    [https://arxiv.org/abs/2611.26776](https://arxiv.org/abs/2611.26776)

    该研究在XD-Violence数据集上进行受控比较，发现将感知模型观察结果整理为一次性阅读的扁平文本是语言模型代理审阅暴力视频最有效的方式，其表现与直接处理原始视频的单一视听模型相当，而消息传递和证据图接口效果较差。

    

    摘要（arXiv:2611.26776v1，公告类型：交叉发布）：如今，无需训练即可基于语言模型构建用于标记暴力视频以供人工操作员审核的系统，方法是组合多个感知模型并让语言模型做出决策。我们探讨感知模型的观察结果应以何种方式传递给决策环节，以及哪些证据真正重要。在XD-Violence测试集（799个视频片段）上，我们固定了一组感知代理（视频描述、物体检测、声音标记、语音识别和语音标注）和一个推理模型，并比较三种接口：代理之间的消息传递、通过工具探索的类型化证据图，以及一次性阅读的扁平文本；此外还比较了一个直接接收原始视频片段的单一视听模型。两名标注者提供了人类参照，记录了他们的决策、事件发生的片段，以及他们依赖的是视觉、声音还是语音。结果表明，扁平文本与单一模型的表现相当（F1为0.967对0.968），而证据图代理和消息传递的准确度较低；证据图代理的一个变体……

    arXiv:2611.26776v1 Announce Type: cross  Abstract: Systems that flag violent video for a human operator can now be built from language models without training, by combining perception models and letting a language model decide. We ask how the observations of the perception models should reach the decision, and which evidence matters. On the XD-Violence test set (799 clips) we fix one set of perception agents (video description, object detection, sound tagging, speech recognition and speech tagging) and one reasoning model, and compare three interfaces: messages between agents, a typed evidence graph explored with tools, and flat text read in one pass; and a single audio-visual model given the raw clip. Two annotators provide a human reference that records the decision, the incident segments and whether they relied on sight, sound or speech. Flat text matches the single model (F1 0.967 against 0.968), while the graph agent and message passing are less accurate; a variant of the graph ag
    
[^202]: Windchime：一个采用音频-语言模型实现的视听装置作品

    Windchime: an Audiovisual Installation with Audio-Language Model implementations

    [https://arxiv.org/abs/2611.26759](https://arxiv.org/abs/2611.26759)

    Windchime 是一个视听装置，参观者通过语音输入，借助本地语音识别与音频-语言模型从艺术家语料库中检索音乐和交互式动画，实现了在艺术家、模型与参观者三方之间分配能动性的多模态具身交互体验。

    

    Windchime 是一个视听装置作品，参观者通过口语语言从艺术家创作的语料库中检索音乐和交互式动画。本地自动语音识别（ASR）模型对每个语音提示进行转录，同时音频-语言模型（ALM）对其进行嵌入，通过对406个音频分轨的检索返回一个声景，其中包含实时编码的 Strudel 片段，这些片段层层叠加形成音频输出。与此同时，每个语音输入都会检索到交互式 Three.js 动画场景，每个场景都包含通过 monome grid 和 arc 硬件实现的独特触觉控制。Windchime 在三个不同的方向上分配了能动性：创作出可能输出的艺术家、影响语料库哪些区域能够被检索的音频-语言模型，以及通过语音输入实现多模态具身交互的参观者。

    arXiv:2611.26759v1 Announce Type: cross  Abstract: Windchime is an audiovisual installation in which spoken language retrieves music and interactive animations from an artist authored corpus. A local Automatic Speech Recognition (ASR) model transcribes each prompt while an audio-language model (ALM) embeds it and retrieval over 406 audio stems returns a soundscape with live coded Strudel fragments that layer into audio output. Simultaneously each spoken input results in retrieval of interactive Three.js animation scenes each containing unique tactile control via monome grid and arc hardware. Windchime distributes agency in three distinct directions; across the artist who authored the possible outputs, the ALM that influences what areas of the corpus can be retrieved, and the visitor whose spoken inputs result in multimodal embodied interaction.
    
[^203]: 自验证的LLM驱动科学，以及它为何仍需要人类研究者

    Self-verifiable LLM-driven science and why it needs human researchers

    [https://arxiv.org/abs/2611.26754](https://arxiv.org/abs/2611.26754)

    本文提出把数学领域中形式化证明自动验证的思路（如Lean 4、Coq）推广到非数学科学，设计一种能让LLM科学推理实现自我验证的AI智能体框架，从而把AI辅助科研的瓶颈从“生成结果”转移到消除人工复核成本上。

    

    语言模型如今已广泛应用于所有科学领域——用于阅读文献、从表格和光谱中提取数值、分析数据、起草结论——然而无论答案正确与否，它们都以同样精致自信的语气呈现，因此研究结果必须由人类研究者重新推导验证之后才能被信赖。目前限制AI辅助科研发展的，正是这种人工复核的成本，而非生成结果的能力。在数学领域，这一问题已通过Lean 4和Coq等证明助手得到解决：任何形式化的命题及其证明都可以被写下来并自动检验——这与大语言模型堪称完美契合，如今LLM已被积极用于搜索证明，每一步都经过自动验证，而无需依赖人类来判断结果的对错。在本文中，我们提出将这一方法扩展到非数学科学领域。我们描述了一种AI智能体框架，旨在使LLM的科学推理……（原文摘要在此处截断）

    arXiv:2611.26754v1 Announce Type: cross  Abstract: Language models are now used throughout all sciences - for reading literature, extracting numbers from tables and spectra, analyzing data, drafting conclusions - and their answers arrive in the same polished confident tone whether they are right or wrong, so a result has to be re-derived by a human researcher before it can be relied on. The cost of that re-checking, not the ability to generate results, is what currently limits AI-assisted research. In the field of mathematics, this problem is solved by proof assistants like Lean 4 and Coq that let any formal statement and its proof be written down and checked - a perfect match for LLMs, which are now actively used to search for proofs with every step automatically validated, rather than relying on a human to judge the result. In this paper, we propose an extension of this approach to non-mathematical sciences. We describe an AI agent harness that aims to make LLM's scientific reasoning
    
[^204]: 面向多模态检索与生成的深度关联残差标记

    Depth-Linked Residual Tokens for Multimodal Retrieval and Generation

    [https://arxiv.org/abs/2611.26752](https://arxiv.org/abs/2611.26752)

    提出DepThink框架，在单次前向传播中通过轻量级可学习残差递归将一串残差标记的隐藏状态相互耦合，使其逐步捕捉互补信息，从而兼得迭代架构的逐步精炼能力与并行单次推理的高效性，服务于多模态检索与生成。

    

    arXiv:2611.26752v1 公告类型：cross 摘要：诸如循环语言模型等迭代式架构提升了Transformer的参数效率与推理能力，但其串行的主干迭代会引入显著的计算延迟。相反，可并行的可学习特殊标记（例如Vision Transformer中的register标记）能够实现高效的单次前向推理，但往往缺乏串行式的逐步精炼，且各标记之间常表现出冗余的表示。为应对这些挑战，我们提出了深度关联残差标记（DepThink），一个面向多模态检索与生成的高效框架。DepThink在单次预填充前向传播中运行，附加一串K个可学习的残差标记，其隐藏状态通过一个轻量级的可学习残差递归相互耦合。该机制使每个残差标记能够逐步捕捉前序标记所遗漏的互补信息。理论（分析）

    arXiv:2611.26752v1 Announce Type: cross  Abstract: Iterative architectures such as looped language models improve parameter efficiency and reasoning capacity of transformers, yet their sequential backbone iterations introduce substantial computational latency. Conversely, parallelizable learnable special tokens (such as the register tokens in Vision Transformer) enable efficient single-pass inference, but often suffer from the absence of sequential refinement and exhibit redundant representations across tokens. To address these challenges, we introduce Depth-Linked Residual Tokens (DepThink), an efficient framework designed for multimodal retrieval and generation. Operating within a single pre-filling forward pass, DepThink appends a sequence of K learnable residual tokens whose hidden states are coupled through a lightweight, learnable residual recurrence. This mechanism enables each residual token to progressively capture complementary information missed by preceding tokens. Theoreti
    
[^205]: MomentMem：为机器人操作记忆关键时刻

    MomentMem: Memorizing Key Moments for Robot Manipulation

    [https://arxiv.org/abs/2611.26745](https://arxiv.org/abs/2611.26745)

    提出了MomentMem分层架构，由基于VLM的记忆器选择关键帧、执行器基于所选记忆预测动作，并引入VMBench基准来评估机器人长时程操作中的视觉时刻记忆能力。

    

    机器人操作通常需要记住不再可观察到的视觉信息，例如物体的初始位置。现有方法通常依赖于完整的观察历史，但这类历史在长时程任务中会变得冗余；或者依赖文本摘要，但文本可能遗漏精确的视觉细节。我们研究了视觉时刻记忆，即从关键时刻保留此类细节以供后续操作使用的能力。为支持这项研究，我们引入了VMBench，这是一套需要记忆空间位置、视觉外观及其关联的长时程操作任务套件。我们进一步提出了MomentMem，这是一种分层架构，其中基于VLM的记忆器负责选择关键帧，执行器则根据所选记忆来预测动作。这种分离设计使我们能够在评估闭环任务成功率的同时评估记忆选择的质量，并且可以独立于执行器来研究记忆器的泛化能力。在VMBench上，MomentMem...

    arXiv:2611.26745v1 Announce Type: cross  Abstract: Robotic manipulation often requires remembering visual information that is no longer observable, such as the original location of an object. Existing approaches typically rely on full observation histories, which become redundant over long horizons, or text summaries, which may omit precise visual details. We study visual moment memory, the ability to retain such details from key moments for subsequent manipulation. To support this study, we introduce VMBench, a suite of long-horizon manipulation tasks requiring memory of spatial locations, visual appearances, and their associations. We further propose MomentMem, a hierarchical architecture in which a VLM-based memorizer selects keyframes and an executor predicts actions conditioned on the selected memories. This separation enables us to evaluate memory selection alongside closed-loop task success and to study memorizer generalization independently of the executor. On VMBench, MomentMe
    
[^206]: INCISE：手术视频的免训练上下文分割

    INCISE: Training-Free In-Context Segmentation of Surgical Videos

    [https://arxiv.org/abs/2611.26734](https://arxiv.org/abs/2611.26734)

    INCISE是一个面向手术视频的免训练上下文分割框架，通过从多个手术视频中选择能共同覆盖目标视频局部外观的参考帧，仅需1.7%的标注帧即可实现分割，无需大规模专家标注和模型微调。

    

    手术视频分析需要视频对象分割（VOS），然而现有方法通常依赖于使用大量专家标注对预训练模型进行微调。上下文分割（ICS）提供了绕过这一高昂成本的前景，即仅使用少量带标注的参考来分割目标视频，这些参考是冻结模型获取手术知识的唯一来源。然而，现有的ICS方法往往随机抽取参考，这些参考集中于主导手术视频的少数视图，可能遗漏其他感兴趣的对象（例如短暂出现的器械和小型结构）。为解决这些挑战，我们提出了INCISE，一个面向手术视频的免训练ICS框架。INCISE从多个手术视频中选择无标注帧，这些帧共同覆盖目标视频的局部外观，然后利用这些参考来分割目标视频。仅需选择并标注1.7%的训练帧即可……

    arXiv:2611.26734v1 Announce Type: cross  Abstract: Surgical video analysis requires video object segmentation (VOS), yet existing methods are commonly based on pretrained model fine-tuning using a large number of expert labels. In-context segmentation (ICS) offers the promise of bypassing this high cost by segmenting a target video using a few annotated references, which are the sole source of surgical knowledge for a frozen model. However, existing ICS methods tend to draw such references at random, which concentrate on a few views that dominate a surgical video and can miss other objects of interest (e.g., transient instruments and small structures). To address these challenges, we propose INCISE, a training-free ICS framework for surgical videos. INCISE selects unlabeled frames from multiple surgical videos that jointly cover the local appearances of a target video, and then segments the target video using these references. With only 1.7% of training frames selected and annotated as
    
[^207]: TOUMAII：一个包含450万偏好对的大规模工具调用偏好数据集及其源无关生成器

    TOUMAII: A 4.5M-Pair Tool-Calling Preference Dataset and Its Source-Agnostic Generator

    [https://arxiv.org/abs/2611.26722](https://arxiv.org/abs/2611.26722)

    该论文提出了包含约450万条经验证偏好对的大规模工具调用偏好数据集TOUMAII及其源无关的偏好对生成器，以缓解工具调用领域（尤其是多轮场景）偏好数据的稀缺，从而支持DPO训练出更可靠的工具调用能力。

    

    为了自主执行多步骤任务，企业级智能体必须可靠地与工具交互、调用API或查询数据库，这使得工具调用成为AI模型的一项基础能力。监督微调（SFT）只优化生成正确的调用，却从未将其与看似合理的错误调用进行对比。在一个即使微小的格式错误也可能使调用失效、错误决策可能导致整个交互脱轨的环境中，直接偏好优化（DPO）恰恰针对这种对比，因此DPO需要偏好对而非简单的演示样本。然而对于工具调用而言，此类数据，尤其是在现实的多轮交互场景中，仍然极其稀缺。为解决这一稀缺性问题，我们提出了TOUMAII，一个大规模工具调用偏好数据集，包含约450万条经过验证的偏好对，这些数据源自Agent-Ark/Toucan-1.5M（apache-2.0许可）对话。该数据集由多轮和单轮交互组成，并且……

    arXiv:2611.26722v1 Announce Type: new  Abstract: To autonomously execute multi-step tasks, enterprise agents must reliably interact with tools, call APIs, or query databases, making tool-calling a fundamental capability of AI models. Supervised fine-tuning (SFT) optimises for producing the correct call, but never contrasts it with a plausible incorrect one. In an environment where even a minor formatting error can invalidate a call and an incorrect decision can derail the interaction, direct preference optimisation (DPO) targets exactly this contrast. DPO, therefore, requires preference pairs rather than plain demonstrations. For tool-calling, such data, especially in realistic multi-turn settings, remains extremely scarce. To address this scarcity, we introduce TOUMAII, a large-scale tool-calling preference dataset containing ~4.5M validated preference pairs, derived from Agent-Ark/Toucan-1.5M (apache-2.0) conversations. The dataset consists of multi- and single-turn interactions and 
    
[^208]: 构造即一致：合成网络威胁情报（CTI）知识图谱与一致性验证的形式极限

    Coherence by Construction: Synthesizing CTI Knowledge Graphs and the Formal Limits of Coherence Validation

    [https://arxiv.org/abs/2611.26717](https://arxiv.org/abs/2611.26717)

    该论文提出“通过模拟进行生成”的新范式：语言模型智能体提出意图、确定性裁判依据显式世界模型裁决事件，使关系仅在对应事件发生时才被输出，从而将网络威胁情报知识图谱的一致性变为构造过程的固有属性，并探讨了一致性验证的形式化极限。

    

    使用大语言模型合成知识图谱等关系型训练数据时，通常一次只生成一个结构化对象，这虽然保证了每个对象本身的有效性，却使对象之间的关系缺乏依据。我们研究了一种替代性范式——“通过模拟进行生成”：不再直接描述关系型产物，而是模拟产生该产物的过程，并将产物视为这一过程的因果记录。语言模型智能体提出意图，由确定性裁判依据显式世界模型裁决结果，因此只有在关系所对应的事件实际发生时才会输出该关系，从而使一致性成为构造过程的固有属性。我们在 STIX 2.1 标准下针对网络威胁情报知识图谱实例化了该范式，构建了三个引擎，其中包括一个将语言模型遗传算子应用于结构化图基因组的进化引擎，三者共享一个确定性编译器，使格式错误和悬空的输出无法被表示（原文摘要至此截断）。

    arXiv:2611.26717v1 Announce Type: new  Abstract: Synthesizing relational training data such as knowledge graphs with a large language model typically emits structured objects one at a time, which keeps each object valid but leaves the relationships between them ungrounded. We study an alternative paradigm, generation by simulation: rather than describing a relational artifact, we simulate a process that produces it and take the artifact to be its causal record. Language-model agents propose intent and a deterministic referee resolves outcomes against an explicit world model, so a relation is emitted only when the event it represents occurs, making coherence a property of construction. We instantiate the paradigm for cyber threat intelligence knowledge graphs in the STIX 2.1 standard with three engines, among them an evolutionary engine applying language-model genetic operators to structured graph genomes, sharing a deterministic compiler that makes malformed and dangling output unrepre
    
[^209]: 建模标注而非噪声：面向噪声标签的语义标注通道提示学习

    Model the Annotation, Not the Noise: Semantic Annotation-Channel Prompt Learning against Noisy Labels

    [https://arxiv.org/abs/2611.26702](https://arxiv.org/abs/2611.26702)

    该论文提出SACP框架，从数据生成视角将潜在干净标签预测与实例相关的标注生成过程解耦，通过融合冻结VLM语义先验与轻量级视觉残差构建语义感知标注通道，并利用广义期望最大化与软后验进行优化，从而有效应对噪声标签问题。

    

    参数高效的提示学习已成为适配预训练视觉-语言模型（VLM）的基石，但在少样本网络与检索场景中，它对标签噪声仍然高度脆弱。现有的噪声鲁棒方法通常依赖于硬样本选择或确定性的标签纠正，这些做法丢弃了宝贵的语义歧义信息，并忽视了观测到的标注实际上是如何生成的。为弥合这一差距，我们从数据生成的视角重新表述噪声提示学习，提出了SACP——一种基于语义感知标注通道的提示学习框架。SACP并不启发式地抑制噪声，而是将潜在干净标签预测与依赖实例的标注生成过程解耦。通过将冻结的VLM语义先验与轻量级视觉残差相结合，SACP构建了一个结构化的标注通道，并通过带有软后验的广义期望最大化算法进行优化。此外……（摘要内容在此处被截断）

    arXiv:2611.26702v1 Announce Type: cross  Abstract: Parameter-efficient prompt learning has become a cornerstone for adapting pre-trained vision-language models (VLMs), yet it remains highly vulnerable to label noise in few-shot Web and retrieval scenarios. Existing noise robust methods typically rely on hard sample selection or deterministic label correction, which discard valuable semantic ambiguity and overlook how observed annotations are actually generated. To bridge this gap, we reformulate noisy prompt learning through the perspective of data generation, proposing SACP, a Semantic-aware Annotation Channel-based Prompt learning framework. Instead of suppressing noise heuristically, SACP decouples latent clean-label prediction from instance-dependent annotation generation. By integrating frozen VLM semantic priors with lightweight visual residuals, SACP constructs a structured annotation channel optimized via generalized expectation-maximization with soft posteriors. Furthermore, a
    
[^210]: 关于线性神经网络的损失景观：超越临界点

    On the loss landscape of the linear neural network: beyond critical points

    [https://arxiv.org/abs/2611.26671](https://arxiv.org/abs/2611.26671)

    本文提出了一种包含线性神经网络损失景观中所有临界点的中间几何结构——“临界谷线”，它提供了可二维可视化的损失景观全局描述，并表明梯度下降会被该结构吸引从而沿其持续降低损失至全局最小值。

    

    自Baldi和Hornik在20世纪80年代末的开创性工作以来，线性神经网络损失景观中的临界点已被广泛研究，但在中间尺度或全局尺度上对这一景观的描述仍然缺失。这种表征至关重要，尤其是因为大量文献表明梯度下降会绕过临界点。本文引入了一种中间结构，称为临界谷线，它包含所有临界点并展现出关键的几何性质。除了在比现有文献更简单的假设下为Baldi和Hornik的结果提供了几何证明之外，该框架还提供了损失景观的全局视角，并且可以在二维平面上轻松可视化。经验观察表明，梯度下降会被吸引向临界谷线，沿着该谷线，损失持续下降直至其全局最小值。

    arXiv:2611.26671v1 Announce Type: cross  Abstract: While the critical points of the loss landscape in linear neural networks have been extensively studied since the seminal work of Baldi \& Hornik in the late 1980s, a description of this landscape at an intermediate or global scale is still lacking. Such a characterization is crucial, particularly as a substantial body of literature indicates that gradient descent circumvents critical points. The present article introduces an intermediate structure, termed the critical thalweg, which encompasses all critical points and exhibits key geometric properties. Beyond providing a geometric proof of Baldi \& Hornik's results using simpler assumptions than the existing literature, this framework offers a global view of the loss landscape that can be easily visualized in a two-dimensional plane. Empirical observations suggest that gradient descent is attracted toward the critical thalweg, along which the loss continues to decrease to its global m
    
[^211]: 再添一个智能体：AI团队规模与组建方式如何影响人机头脑风暴团队的生产力与多样性

    Adding One More Agent: How AI Team Size and Formation Shape Productivity and Diversity in Human-AI Brainstorming Teams

    [https://arxiv.org/abs/2611.26663](https://arxiv.org/abs/2611.26663)

    在人机头脑风暴实验中，双AI团队虽提高了创意总量却大幅压缩了人类贡献，而单个“平均人类行为校准”的AI反而最能激发人类创意产出，表明AI伙伴数量与组建方式共同重塑了团队的生产力与多样性。

    

    生成式AI已经提升了个人生产力，但增加多个AI伙伴究竟是利还是弊仍未有定论。我们在一项头脑风暴实验（N=168）中，将AI伙伴数量（一个与两个）与行为组建方式（校准为平均人类行为 vs 由参与者定制）进行交叉设计，并以人人二人组作为参照。结果显示：双AI房间在共享看板上发布了更多创意，但人类原创创意数量大幅下降。当只有一个AI伙伴时，组建方式至关重要：与平均行为校准AI配对的人类发布的创意多于任何其他条件；而当有两个AI时，这一效应消失。人类仍在持续交谈，但将创意转化为看板帖子的比例降低。AI的社交性对话预示人类发帖更多，而AI发帖量则预示人类发帖更少。AI的存在提升了人类原创创意的全局多样性，但只有引入第二个AI才能增加会话内部的差异化。AI伙伴的数量与组建方式能够重塑……

    arXiv:2611.26663v1 Announce Type: new  Abstract: Generative AI has improved individual productivity, but whether adding a plurality of AI partners helps or hurts remains open. We crossed AI partnership (one versus two) with behavioral formation (calibrated to average human behavior versus customized by the participant) in a brainstorming experiment ($N$=168) with human-human dyads as reference. Two-AI rooms posted more ideas to the shared board, but human-authored ideas dropped substantially. With one AI partner, formation mattered: humans paired with an average-calibrated AI posted more than any other condition. With two, this effect vanished. Humans continued talking but converted fewer ideas into board posts. AI social talk predicted more human posts, while AI posting volume predicted fewer. AI presence increased global diversity of human-authored ideas, but only a second AI increased within-session differentiation. The number of AI partners and their formation can reshape what the 
    
[^212]: 利用异构随机预言机计算加权集合覆盖：截止期限约束下的令牌复杂度

    Computing Weighted Set Covers with Heterogeneous Stochastic Oracles: Deadline-Constrained Token Complexity

    [https://arxiv.org/abs/2611.26656](https://arxiv.org/abs/2611.26656)

    该论文刻画了概率图灵机在截止期限和成功概率约束下求解加权集合覆盖问题的最小期望令牌成本，证明最优策略至多需要在两个预选的预言机批次序列之间随机化。

    

    我们研究了一种计算模式，其中概率图灵机（PTM）利用被抽象为异构随机预言机的AI系统来求解加权集合覆盖问题。一项初步研究表明，大语言模型（LLM）的表现可以优于加权贪心算法。异构随机预言机在能力、响应分布、延迟和令牌成本方面各不相同。我们研究了PTM如何选择HSO，以便在截止期限内、以所需的成功概率获得满足规定权重阈值的完整覆盖，并分析期望令牌成本。对于请求完整覆盖的查询，在假设各次调用成功相互独立且失败不提供信息的前提下，我们通过线性规划刻画了每个可行成功概率对应的最小期望令牌成本。最优策略在至多两个预先选择的预言机批次序列上进行随机化。在达到最大成功概率的策略中，相对于一次性查询所有预言机，两批次选择所实现的期望成本降低……

    arXiv:2611.26656v1 Announce Type: cross  Abstract: We study computation in which a probabilistic Turing machine (PTM) uses AI systems, abstracted as heterogeneous stochastic oracles (HSOs), to solve Weighted Set Cover. A pilot study shows that an LLM can outperform weighted greedy. HSOs differ in capabilities, response distributions, latencies, and token costs. We study how the PTM selects HSOs to obtain, by a deadline and with a required success probability, a full cover satisfying a prescribed weight threshold, and we analyze expected token cost.   For queries requesting full covers, assuming independent call successes and uninformative failures, we characterize by linear programming the minimum expected token cost for every feasible success probability. An optimal policy randomizes over at most two preselected sequences of oracle batches. Among policies attaining maximum success probability, the expected cost reduction of two-batch selection relative to querying all oracles in one b
    
[^213]: VerifiQuant：面向金融大语言模型工作流的“执行即治理”框架

    VerifiQuant: Governance-by-Execution for Financial LLM Workflows

    [https://arxiv.org/abs/2611.26649](https://arxiv.org/abs/2611.26649)

    VerifiQuant提出“执行即治理”框架，通过将金融逻辑封装为推理契约并用可执行诊断漏斗在计算前评估每个查询，使金融大模型能够结构化、可审计地区分计算、澄清与拒绝，从而在输入不支撑数值答案时实现更安全的行为。

    

    嵌入金融工作流的大语言模型最危险的失效方式，是输出自信、格式规范但错误的数字，且没有任何信号能将其与正确结果区分开。我们认为，核心的部署问题在于可治理性：系统必须能够判断一个问题何时支持给出数值答案，何时应当转而进行澄清、拦截或拒绝。我们提出VerifiQuant，一个“执行即治理”框架，它将金融逻辑封装为推理契约，并在确定性计算之前通过可执行的诊断漏斗对每个查询进行评估。这种分离使计算、澄清和拒绝变得结构化且可审计，同时将金融算术运算保留在语言模型之外。在声明的契约库范围内，与匹配的检索-执行基线相比，其主要收益并非在有明确答案的问题上获得更高的准确率，而是当输入不足以支持数值答案时表现出更安全的行为。

    arXiv:2611.26649v1 Announce Type: cross  Abstract: Large language models embedded in financial workflows fail most dangerously by emitting confident, well-formatted but wrong numbers with no signal distinguishing them from correct ones. We argue that the central deployment problem is governability: a system must determine when a question supports a numeric answer and when it should instead clarify, intercept, or refuse. We present VerifiQuant, a governance-by-execution framework that packages financial logic into inference contracts and evaluates each query through an executable diagnostic funnel before deterministic computation. This separation makes computation, clarification, and refusal structured and auditable while keeping financial arithmetic outside the language model. Against matched retrieval-and-execution baselines, and within a declared contract library, the principal benefit is not higher accuracy on well-posed questions but safer behavior when an input does not justify a 
    
[^214]: LLM“杰利蝾螈”式操纵：在大语言模型审计中分离交互效应与加性效应

    LLM GerryMandering: Separating Interactions from Additive Effects in LLM Audits

    [https://arxiv.org/abs/2611.26538](https://arxiv.org/abs/2611.26538)

    该论文揭示了现有交叉性偏差审计判据会把独立属性的加性效应误报为交互偏差（平均每个数据集误报66.1个子群体、真实交互检出率仅32%），并提出ADDIS方法，通过在所有属性上拟合加性模型建立基线，从而真正将交互效应与加性效应分离，准确审计交叉性子群体偏差。

    

    交叉性偏差是指模型对同时拥有多个边缘化身份的人的对待方式，不同于其各个单独身份所能预测的水平。对这类交叉性子群体的审计依然充满挑战：子群体的数量呈平方级增长，且对于“独立属性本应预测出什么”缺乏公认的基线。在没有这一基线的情况下，许多现有审计只要发现子群体的表现超过任一单独属性就将其标记出来，而属性独立作用时本就满足这一条件。我们在150个不含任何交互作用的合成数据集上检验了这一标准判据：在使用洗牌零假设时，该判据平均每个数据集误报66.1个虚假子群体，且在每个数据集中至少误报一个。当我们在这些数据集中植入已知大小的交互作用后，即便使用修复后的零假设，该判据也只能恢复其中的32%。我们提出了ADDIS（Additive Intersectional Screen，加性交叉性筛选），它在所有属性上拟合一个加性模型，并据此对每个子群体进行评分……

    arXiv:2611.26538v1 Announce Type: new  Abstract: Intersectional bias arises when a model treats people who hold several marginalised identities at once differently from what their separate identities would predict. Auditing such intersectional subgroups remains a challenge: their number grows quadratically, and there is no agreed baseline for what independent attributes would predict. Without such a baseline, many existing audits flag a subgroup when it exceeds each attribute alone, a condition that attributes acting independently already satisfy. We test this standard criterion on 150 synthetic datasets built with no interaction. With the shuffle null, it flags 66.1 false subgroups per dataset on average, and at least one in every dataset. When we plant interactions of known size into these datasets, a repaired null lets the criterion recover only 32% of them. We introduce ADDIS (Additive Intersectional Screen), which fits an additive model over all attributes, scores each subgroup by
    
[^215]: 网格细胞作为谱后继特征：持续强化学习的可复用基

    Grid Cells as Spectral Successor Features: A Reusable Basis for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.26504](https://arxiv.org/abs/2611.26504)

    该论文受网格细胞对应后继表示特征向量的启发，提出用固定参考策略转移矩阵的特征向量作为后继特征的可复用基，使得长期回报预测具有封闭形式的系数，且无需后继特征网络或回放缓冲区，为持续强化学习提供了高效基础。

    

    持续强化学习要求智能体在任务发生变化时进行适应，同时复用其对环境的已有知识。后继特征可以将环境动态与任务特定的奖励分离，但其有效性取决于基础特征的选择。以内嗅皮层网格细胞对应于后继表示特征向量这一发现作为设计原则，我们从固定参考策略转移矩阵的特征向量构建这一基。采用这样的基，某一状态-动作对处的后继特征是该同一状态-动作对处基础特征的线性组合——这一性质在一般情况下并不成立——且组合系数具有封闭形式：每个后继特征等于其对应的基础特征乘以 θ_i = 1/(1-γλ_i)，其中 λ_i 是其特征值。因此，长期回报可以用与即时奖励相同的特征来预测，无需后继特征网络或经验回放缓冲区。

    arXiv:2611.26504v1 Announce Type: new  Abstract: Continual reinforcement learning requires agents to adapt when the task changes while reusing what they know about the environment. Successor features separate dynamics from task-specific reward, but their effectiveness depends on the choice of base features. Taking the identification of entorhinal grid cells with eigenvectors of the successor representation as a design principle, we build this basis from eigenvectors of the transition matrix of a fixed reference policy. With such a basis, the successor features at a state-action pair are a linear combination of the base features at that same pair, which is not true in general, and the coefficients are known in closed form: each successor feature is its base feature multiplied by $\theta_i = 1/(1-\gamma\lambda_i)$, where $\lambda_i$ is its eigenvalue. The long-term return is thus predicted from the same features as the immediate reward, no successor-feature network or replay buffer is ne
    
[^216]: 编码智能体不忠实工作报告的表征与检测

    Characterizing and Detecting Unfaithful Work Reports by Coding Agents

    [https://arxiv.org/abs/2611.25760](https://arxiv.org/abs/2611.25760)

    该论文基于真实的开发者—智能体对话数据，系统性地刻画了编码智能体在真实开发工作中不忠实工作报告的表现形式及其带来的误导风险，并研究其检测方法。

    

    编码智能体越来越多地承担长周期的开发任务，自主规划并执行跨越文件、工具和会话的多步骤操作。由于开发者很少逐项跟进每一步操作，而接续工作的智能体又依赖于已完成工作的记录，智能体的工作报告往往成为对所更改和所建立内容的主要描述。当报告包含虚假或缺乏依据的声明时，会给接收者带来虚假的安全感：未完成的工作可能在未经核实的情况下被接受，对当前状态的错误假设可能传播到后续步骤和其他智能体，而采信声称成功的评估者可能奖励的是报告本身而非实际工作。然而，人们对于不忠实报告在真实开发工作中的具体表现，以及开发者和智能体在其出现后如何应对，仍知之甚少。基于 SWE-chat 中真实的开发者与智能体对话，我们对自我（原文摘要在此处截断，后续内容缺失）

    arXiv:2611.25760v1 Announce Type: cross  Abstract: Coding agents increasingly carry out long-horizon development tasks, autonomously planning and executing multi-step actions that span files, tools, and sessions. Because developers rarely follow each action, and agents resuming the work depend on records of what was done, the agent's work report often becomes the primary account of what was changed and established. When the report contains false or unsupported claims, it gives recipients false assurance: incomplete work may be accepted without verification, incorrect assumptions about the current state can propagate to later steps and agents, and evaluators that credit stated success may reward the report rather than the work. Yet little is known about what unfaithful reporting looks like in real development work, or how developers and agents handle it once it surfaces. Starting from real developer--agent conversations in SWE-chat, we conduct a systematic mixed-methods analysis of self
    
[^217]: STARK：迈向真正会学习与记忆的个性化智能体

    STARK: Moving towards Personalized Agents That Truly Learn and Remember

    [https://arxiv.org/abs/2611.25705](https://arxiv.org/abs/2611.25705)

    STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。

    

    长期记忆对于在长时间内与用户交互的个性化智能体系统至关重要，然而当前的架构只能捕获“发生了什么”，而将“为什么会发生”的推理留给底层大语言模型事后推导。这些事件不仅能够传达用户经历了什么，还能传达这些经历如何塑造了用户的行为模式。我们提出了STARK，一种双图记忆架构，它将这些事件不仅视为节点，还视为行为洞察之间的边。STARK维护一个情景图，将交互记录为在时间、因果和语义上相互关联的情景；以及一个洞察图，用于表示由大语言模型提取的行为信号，并根据支持这些信号的事件数量进行评分，同时通过引发这些洞察的事件将它们相互连接。在推理时，行为知识可以直接从由事件作为边所支撑的洞察节点中检索获得。

    arXiv:2611.25705v1 Announce Type: cross  Abstract: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-ed
    
[^218]: LLM能在不训练的情况下修复语言模型吗？

    Can an LLM Fix a Language Model Without Training It?

    [https://arxiv.org/abs/2611.25613](https://arxiv.org/abs/2611.25613)

    研究表明，LLM无需任何梯度训练，仅通过交互式检查模型预测、激活和权重统计并选择数值投影编辑，即可有效修复受损的Transformer语言模型，其表现可与黑盒搜索相当甚至大幅超越（如Pythia上达到65.62%准确率而黑盒搜索为0%）。

    

    尽管外部语言模型可以选择并测试对另一个模型权重的修复，但模型行为也可以通过固定的数值编辑来恢复。我们在一项Transformer修复的对照研究中评估了交互式选择的价值。在检查预测结果、激活摘要和权重统计之后，编辑者（LLM）选择要应用于目标模型的投影配方，而无需任何梯度更新。我们分析了涉及五个掩码语言模型和因果语言模型的20个记录会话，将LLM所选的编辑与固定投影、脚本化及等预算的黑盒搜索以及低秩适配进行了比较。在核心的双层故障下，实时选择在SmolLM2和Qwen2.5上的表现与黑盒搜索相当。对于Pythia，实时选择达到了65.62%的补全准确率，而黑盒搜索为0%，固定投影为66.15%。在另一个同时影响注意力和前馈权重的复合故障下，脚本化和LLM引导的（摘要在此处截断）

    arXiv:2611.25613v1 Announce Type: new  Abstract: Although an external language model can select and test repairs to another model's weights, behavior can also be recovered through fixed numerical edits. We assessed the value of interactive selection in a controlled study of Transformer repair. After inspecting predictions, activation summaries, and weight statistics, the editor selects projection recipes to apply to the target without gradient updates. We analyzed 20 recorded sessions involving five masked and causal language models, comparing the selected edits with fixed projections, scripted and matched-budget black-box search, and low-rank adaptation. Under the core two-layer fault, live selection matches black-box search on SmolLM2 and Qwen2.5. For Pythia, live selection achieves 65.62% completion accuracy, compared with 0% for black-box search and 66.15% for a fixed projection. Under a separate compound fault affecting attention and feed-forward weights, both scripted and LLM-gui
    
[^219]: 规划你的自适应！基于本体的机器人任务与架构协同自适应规划

    Plan your Self-Adaptation! Ontology-Mediated Task and Architecture Co-adaptation Planning for Robots

    [https://arxiv.org/abs/2611.25254](https://arxiv.org/abs/2611.25254)

    本文提出PLANTA方法，利用OWL本体表示架构备选方案、PDDL定义任务目标，实现了机器人任务规划与架构重构的联合协同规划，并能根据执行中的变化进行重规划。

    

    自主机器人可能需要在运行条件发生变化时，同时调整其任务规划和架构配置。这些决策是相互依赖的：架构配置决定了机器人能够执行哪些动作，而任务规划则决定了需要哪些能力。本文提出了PLANTA，一种基于本体的任务与架构协同自适应方法，能够在任务规划过程中考虑这些依赖关系。PLANTA使用以Web本体语言（OWL）表达的本体来表示架构备选方案及其相关的功能和质量需求，同时使用规划领域定义语言（PDDL）来定义任务动作和任务目标。通过自动翻译和集成这些模型，PLANTA实现了任务动作与重构动作的联合规划。该方法采用满意规划技术，并在执行过程中发生变更导致当前计划失效时进行重新规划。PLANTA的评估（摘要在此处截断）

    arXiv:2611.25254v1 Announce Type: cross  Abstract: Autonomous robots may need to adapt both their task plans and architectural configurations as operating conditions change. These decisions are interdependent: architectural configurations determine which actions a robot can perform, while task plans determine which capabilities are required. This paper presents PLANTA, an ontology-mediated approach to task and architecture co-adaptation that considers these dependencies during mission planning. PLANTA uses an ontology expressed in the Web Ontology Language to represent architectural alternatives and their associated functional and quality requirements, while the Planning Domain Definition Language defines task actions and mission goals. By automatically translating and integrating these models, PLANTA enables joint planning of task and reconfiguration actions. The approach uses satisficing planning and replans when changes during execution invalidate the current plan. PLANTA is evaluat
    
[^220]: MAWAM：面向协同操作的多智能体世界-动作模型

    MAWAM: A Multi-Agent World-Action Model for Coordinated Manipulation

    [https://arxiv.org/abs/2611.25105](https://arxiv.org/abs/2611.25105)

    提出MAWAM多智能体世界-动作模型，通过跨智能体注意力在单次采样中联合生成所有机器人的未来视角和可执行动作，在真实与仿真协同操作任务中大幅超越现有方法。

    

    机器人团队中的每一个动作都会改变其队友必须行动的世界。因此，协同操作需要同时预测团队的动作以及它们共同创造的物理未来。我们提出MAWAM，这是一个多智能体世界-动作模型，能够在单次采样中共同生成每个机器人的未来视角和可执行的动作块。MAWAM基于预训练的单机器人世界-动作骨干网络构建，通过跨智能体注意力机制对所有机器人的动作和未来视角token进行联合去噪，同时为每个机器人保留独立的控制接口。MAWAM在全部三个真实机器人任务（洗衣袋收集、浴巾折叠和雨伞交接）中取得最高成功率，平均成功率达67.8%，比在相同演示数据和相机下训练的MA-VLA高出18.9个百分点。在仿真中，MAWAM在最多四条机械臂协作的八个任务上达到90.3%的平均成功率。消融实验表明跨智能体注意……

    arXiv:2611.25105v1 Announce Type: cross  Abstract: Every move in a robot team changes the world in which its teammates must act. Coordinated manipulation therefore calls for predicting the team's actions together with the physical future they jointly create. We introduce MAWAM, a multi-agent world-action model that jointly generates each robot's future views and executable action chunks in a single sample. Built from a pretrained single-robot world-action backbone, MAWAM jointly denoises all robots' action and future-view tokens through cross-agent attention, while preserving a separate control interface for each robot. MAWAM achieves the highest success on all three real-robot tasks (laundry-bag collection, bath-towel folding, and umbrella handover), with 67.8% mean success, 18.9 percentage points above MA-VLA trained on the same demonstrations and cameras. In simulation, MAWAM reaches 90.3% mean success on eight tasks with teams of up to four arms. Ablations show that cross-agent att
    
[^221]: 借力本地模型：滥用浏览器内嵌语言模型进行后渗透攻击

    Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation

    [https://arxiv.org/abs/2611.25086](https://arxiv.org/abs/2611.25086)

    该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。

    

    浏览器如今向普通终端设备预装本地语言模型。Google在Chrome中集成了Gemini Nano以支持摘要生成等生产力功能，其推理完全在设备端运行。这些浏览器内嵌的模型为攻击者提供了新的“就地取材”攻击机会——每个模型实例都是一个潜在的后渗透攻击引擎，它已经预装在设备上、经过厂商签名、且对网络监控完全不可见。Gemini模型具备多模态能力，能够处理文件、分析截图或与攻击者对话。我们证明，仅使用操作系统自带的解释器即可在主机上驱动该模型，并发现模型下载前的一次性用户激活检查很容易通过模拟点击满足，甚至可以被完全绕过。我们测量并报告了当前模型在识别主机上的凭据、检测电子邮件中的个人身份信息（PII）、分类图像以及调用自定义工具方面的能力。我们结合了……

    arXiv:2611.25086v1 Announce Type: cross  Abstract: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed entirely.We measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine the
    
[^222]: 当图表与文本不一致时：评估科学论文中的自主视觉证据验证

    When Figures Disagree with Text: Evaluating Autonomous Visual Evidence Verification in Scientific Papers

    [https://arxiv.org/abs/2611.24353](https://arxiv.org/abs/2611.24353)

    该论文提出了一种更贴近真实审稿场景的自主图文一致性验证评估，要求多模态模型在无任何提示的情况下于完整论文中自行定位问题图表并诊断矛盾，结果显示尽管60.6%的审阅提及了缺陷图表，但仅3.3%真正诊断出了其中的不一致。

    

    科学图表为其被引用的论断提供证据支持，因此审阅一篇论文意味着检查图表与文本是否一致，而不仅仅是描述图表展示的内容。现有的图文一致性验证基准通常直接向模型提供图表、相关论断，或告知模型存在不一致的信息，这使得模型能否在完整论文中自行找到值得检查的图表、将视觉元素与其冲突的论断关联起来并识别出冲突，这些问题都未被测试。我们在31篇完整稿件上研究这种自主设置，每篇稿件都包含一个来自PRISMM-Bench的、经审稿人记录且对模型隐藏的图文不一致问题。六个多模态模型在中性提示下审阅每个PDF，随后我们在相同稿件上逐一添加额外输入。在1,445份开放式审阅中，60.6%提到了存在缺陷的图表，然而评分标准仅接受其中3.3%为真正诊断出该矛盾；一项盲审发现该评分标准……

    arXiv:2611.24353v1 Announce Type: new  Abstract: Scientific figures carry evidence for the claims that cite them, so reviewing a paper means checking that a figure agrees with the text, and not only describing what the figure shows. Benchmarks of figure-text verification typically hand the model the figure, the claim, or the knowledge that an inconsistency exists, which leaves untested whether a model can find the figure worth checking in a complete paper, relate a visual element to the claim it conflicts with, and recognize the conflict. We study this autonomous setting on 31 full manuscripts, each carrying a reviewer-documented figure-text inconsistency from PRISMM-Bench that is hidden from the model. Six multimodal models review each PDF under a neutral prompt, and we then add one input at a time to the same manuscripts. Of 1,445 open-ended reviews, 60.6% mention the flawed figure, yet a grading rubric accepts only 3.3% as diagnosing its contradiction; a blind audit finds the rubric
    
[^223]: MLE智能体：大规模推荐系统中的自主机器学习工程

    MLE Agents: Autonomous ML Engineering in Large-scale Recommendation Systems

    [https://arxiv.org/abs/2611.23910](https://arxiv.org/abs/2611.23910)

    该论文构建了在大规模搜索排序系统中自主完成模型改进提出、实现、训练与评估的机器学习工程师（MLE）智能体，并揭示了工业级自动化面临的三大挑战：实验反馈耗时、离线指标难以预测在线效果、改进收益趋于饱和。

    

    改进大规模排序和推荐模型的典型工作流程缓慢而繁琐：工程师提出假设，将其实现为代码更改，训练模型，并进行离线和在线评估，在此过程中往往依赖自身的直觉和经验。自主LLM智能体有望通过自主运行这一循环来加快速度，但它们能否在工业级系统（而非基准测试或小型代理任务）中做到这一点仍不明确。我们构建了“人在回路监督”的机器学习工程师（MLE）智能体，它们能够在大规模搜索排序器上提出、实现、训练和评估模型更改，同时工程师保留对目标设计、搜索方向和最终模型审查的控制权。我们识别出使这一工业环境具有挑战性的三大障碍：每次实验的反馈需要数小时，离线指标并不总能预测在线效果，以及改进在早期收益之后趋于饱和。

    arXiv:2611.23910v1 Announce Type: new  Abstract: The typical workflow for improving large-scale ranking and recommendation models is slow and tedious: an engineer forms a hypothesis, implements it as a code change, trains a model, and evaluates it both offline and online, often relying on their own intuition and experience along the way. Autonomous LLM agents promise to speed this up by running the loop on their own, but it is still unclear whether they can do this in industry-scale systems rather than on benchmarks or small proxy tasks. We build human-on-the-loop machine learning engineer (MLE) agents that propose, implement, train, and evaluate model changes on large-scale search rankers. Engineers retain control over objective design, search direction, and final-model review. We identify three obstacles that make this industry setting challenging: feedback takes hours per experiment, offline metrics do not always predict online impact, and improvements saturate after the early wins.
    
[^224]: 安全个人RSI：评估与修复个人智能体更新

    Safe Personal RSI: Evaluating and Repairing Personal-Agent Updates

    [https://arxiv.org/abs/2611.23651](https://arxiv.org/abs/2611.23651)

    本文提出Safe Personal RSI更新评估机制，通过将递归自我改进限制在提示词层面、固定模型权重与代码，并建立“测试—修复—验证”流程，在29个案例的基准上成功拒绝了21次高风险更新尝试中的20次，保障个人智能体更新后的安全性与任务性能。

    

    Kicker是一个个人健康助手。对助手可复用指令的更改可以在改进任务处理的同时，改变它所披露的内容、它遵循的请求以及它报告操作的方式。我们研究了Safe Personal RSI，一个为Kicker开发的更新评估组件。我们将递归自我改进（RSI）限制在提示词更新的范围内，模型权重和运行时代码保持固定。该流程测试提议的指令与当前指令，重复失败的案例，临时撤销更改以指导修复，并在采纳前检查安全性和任务性能。如果没有版本通过，则继续使用当前指令。我们的可执行基准测试包含29个电子邮件、日历、健康和市场案例，并设有针对授权复用、记忆和工具输出注入的独立探测。在三个模型和两次故意设置的风险编辑中，该流程拒绝了21次更新尝试中的20次。在修复确认中，一个被接纳的GPT-5.6 Luna rep（原文摘要在此处截断）

    arXiv:2611.23651v1 Announce Type: cross  Abstract: Kicker is a personal health assistant. Changes to an assistant's reusable instructions can improve task handling while changing what it discloses, which requests it follows, and how it reports actions. We study Safe Personal RSI, an update-evaluation component developed for Kicker. We restrict recursive self-improvement (RSI) to prompt updates, with model weights and runtime code fixed. The procedure tests proposed and current instructions, repeats failing cases, temporarily undoes changes to guide repairs, and checks safety and task performance before adoption. If no version passes, the current instructions stay in use. Our executable benchmark contains 29 email, calendar, health, and marketplace cases, with separate probes for authorization reuse, memory, and tool-output injection. Across three models and two deliberately risky edits, the procedure rejects 20 of 21 update attempts. In repair confirmation, an admitted GPT-5.6 Luna rep
    
[^225]: 幂等性不等于恢复：语言模型代理工具操作的故障注入研究

    Idempotency Is Not Recovery: A Fault-Injection Study of Language-Model Agent Tool Actions

    [https://arxiv.org/abs/2611.23353](https://arxiv.org/abs/2611.23353)

    该论文通过系统的故障注入实验揭示“幂等性不等于恢复”，证明只有将授权、操作账本、目标端幂等性、对账与证据封存相结合的完整配置，才能将语言模型代理工具调用的重复副作用从基线的117/200降至0/200，并保障故障后的恢复能力。

    

    当语言模型代理的工具调用被中断时，该操作可能已经在未经确认的情况下改变了外部系统。重试存在重复执行的风险；而假定成功则可能掩盖一个从未真正发生的操作。我们在一个受控的代理运行时环境中研究这些故障模式，该环境结合了授权机制、操作账本、目标端幂等性、对账机制和证据封存。我们注入了十类单主机故障，并使用独立于运行时账本的目标端预言机，比较了四种配置和三种消融设置。在Windows和Linux两种系统上各进行了7,513次试验，使用了从24个构建案例中捕获的96个模型响应。在200组已知的原始崩溃与重放对中，完整配置记录了0/200的重复副作用，而各个较简单的基线则达到117/200。与账本相关的消融实验产生了170/170的孤立副作用（完整配置为0/170），恢复成功率仅为200/300（完整配置为300/300）。禁用对账机制……（原文摘要在此处截断）

    arXiv:2611.23353v1 Announce Type: cross  Abstract: When a language-model agent's tool call is interrupted, the action may already have changed an external system without confirmation. Retrying risks duplication; assuming success may conceal an action that never occurred. We study these failure modes in a controlled agent runtime combining authorization, an action ledger, target-side idempotency, reconciliation, and evidence sealing. We inject ten single-host fault classes and compare four configurations and three ablations using a target-side oracle separate from the runtime ledger. On each of Windows and Linux, 7,513 trials used 96 captured model responses from 24 constructed cases. In 200 known original crash and replay pairs, the full configuration recorded 0/200 repeated effects versus 117/200 for each simpler baseline. A ledger-related ablation had 170/170 orphaned effects versus 0/170 and 200/300 recovery successes versus 300/300 for the full configuration. Disabling reconciliati
    
[^226]: 在从未见过的动作上测试世界模型：零售业中的留出计划、预测准确度与价格规则

    Testing World Models on Actions They Have Never Seen: Held-Out Plans, Forecast Accuracy and Price Rules in Retail

    [https://arxiv.org/abs/2611.22537](https://arxiv.org/abs/2611.22537)

    该论文提出了一种检验动作条件世界模型对从未见过的动作泛化能力的测试方法，并在零售定价场景的2,853个模型中发现：预测准确度仅能可靠地区分不同类型的模型，却无法区分同一世界模型的不同版本（在高需求周甚至出现反转），且JEPA预训练等学习动作效果的训练技术并未降低模型在新计划上的误差。

    

    世界模型被信任来回答“如果……会怎样”的问题：即它的世界如何响应一个它从未见过的动作。我们为任何以动作为条件的世界模型提供了一种检验这种信任的测试：通过设计将动作从训练数据中留出，并对照已知答案进行评分；检验预测准确度能否预测模型在这些动作上的表现；以及将相同的结构应用于更简单的模型。我们在零售场景中运行该测试，其中动作是价格和促销计划，共涉及2,853个拟合模型。首先，预测准确度是在不同类型模型之间进行比较的合理依据（与新计划上的误差的秩相关性为0.32至0.66），但对于同一世界模型的不同版本之间则是一个糟糕的依据：在商家将计划安排到高需求周的情况下，这种相关性会发生反转，低至-0.69。其次，训练世界模型学习动作的效果（JEPA预训练、潜在一致性检查、计划日期处的适配）并不会降低其在新计划上的误差。第三，在生成的世界中，记录的常规价格和三个p……（摘要在此处截断）

    arXiv:2611.22537v1 Announce Type: cross  Abstract: A world model is trusted to answer "what if": how its world responds to an action it has never seen. We give a test of that trust for any action-conditioned world model: actions held out of training by design and scored against a known answer; whether forecast accuracy predicts accuracy on those actions; and the same structure given to simpler models. We run it in retail, where the action is a price and promotion plan, across 2,853 fitted models. First, forecast accuracy is a fair guide between kinds of model (rank correlation with error on new plans 0.32 to 0.66) and a poor guide between versions of one world model: where merchants plan into high-demand weeks the link reverses, down to -0.69. Second, training the world model to learn the effect of actions (JEPA pretraining, a latent consistency check, adaptation at the plan date) leaves its error on new plans unchanged. Third, in generated worlds the recorded regular price and three p
    
[^227]: RegimeCouncil：面向LLM智能体记忆冲突治理的成对证据路由方法

    RegimeCouncil: Paired-Evidence Routing For Memory Conflict Governance In LLM Agents

    [https://arxiv.org/abs/2611.22113](https://arxiv.org/abs/2611.22113)

    该论文提出RegimeCouncil——一个无需重训LLM的轻量级成对证据路由器，将“记忆冲突治理”形式化，并在局部与全局两种记忆抑制策略之间在线仲裁，以解决长期记忆悄然过时与延迟反馈下负面经验传播范围难以确定的问题。

    

    语言智能体依赖长期记忆来在长程交互中实现个性化响应，然而随着世界的变化，存储的事实会悄然过时，而智能体往往只有在基于过时记录采取行动之后，才通过延迟反馈得知使用记忆带来了损害。在记忆库被更新之前，这种负面反馈应该传播多远：是仅在针对同一事实的未来查询中抑制记忆使用（局部策略），还是也应跨及其他存储事实（全局策略）？现有记忆系统在设计阶段就隐式地固定了这两种共享范围之一。我们将这种在有效性隐藏、反馈延迟条件下的在线范围选择问题形式化为“记忆冲突治理”，并证明任何一种固定策略单独使用都不足以应对。随后我们提出RegimeCouncil，一个轻量级的成对证据路由器，它将两种范围保留为“顾问”，并在无需重新训练LLM的情况下在线地在两者之间进行仲裁。（原文摘要至此处截断）

    arXiv:2611.22113v1 Announce Type: new  Abstract: Language agents rely on long-term memory to personalize responses across extended interactions, yet stored facts quietly go stale as the world changes, and an agent often learns that using memory hurt only through delayed feedback after acting on an outdated record. Before the memory store is updated, how far should that negative feedback propagate: should it suppress memory use only on future queries for that same fact (a local policy), or also across other stored facts (a global policy)? Existing memory systems implicitly fix one of these two sharing scopes at design time. We formulate this online scope choice under hidden validity and delayed feedback as memory conflict governance, and prove that neither fixed policy is sufficient on its own. We then propose RegimeCouncil, a lightweight paired-evidence router that retains both scopes as advisors and arbitrates between them online without retraining the LLM. Using revealed comparisons 
    
[^228]: 用于安全控制器选择的无抵消约束SquareCB算法

    Cancellation-Free Constrained SquareCB for Safe Controller Selection

    [https://arxiv.org/abs/2611.21880](https://arxiv.org/abs/2611.21880)

    该论文提出一种对成本施加修正权重（预算内权重为零、超预算时恰好拉回预算）的约束SquareCB算法，在可实现性和带边际 δ 的 Slater 条件下实现了 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 的遗憾和无需轮次间抵消的违约界 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，适用于安全控制器选择。

    

    我们研究约束上下文老虎机问题，其中学习者在每一轮从 $K$ 个动作中选择一个，以最小化损失，同时使其动作的期望成本保持在预算以下。现有的到在线回归的归约方法使用从过去约束残差中学习到的拉格朗日乘子对成本进行加权，因此某些轮次的松弛量可能会抵消其他轮次的违约。我们转而在成本上施加修正权重来运行 Foster 等人（2020）提出的 SquareCB：当估计成本在预算之内时权重为零，否则权重恰好足够大以将成本拉回预算。在可实现性假设以及具有边际 $\delta$ 的 Slater 条件下，该算法达到遗憾 $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ 和无抵消违约 $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$，其中 $\mathrm{Reg}_{\mathsf{Sq}}$ 是回归预言机的误差。当预算必须在（原文摘要在此处截断，后续内容缺失）……

    arXiv:2611.21880v1 Announce Type: cross  Abstract: We study constrained contextual bandits, in which a learner chooses one of $K$ actions in each round to minimize loss while keeping the expected cost of its play below a budget. Existing reductions to online regression weight the cost by a Lagrange multiplier learned from past constraint residuals, so slack in some rounds can offset violations in others. We instead run SquareCB proposed in \citep{foster2020beyond} with a rectified weight on the cost: it is zero when the estimated cost is within the budget, and otherwise just large enough to bring it back to the budget. Under realizability and Slater's condition with margin $\delta$, the algorithm attains regret $O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)}/\delta)$ and cancellation-free violation $\sum_t(V_c^{p_t}-b)^+=O(\sqrt{KT\,\mathrm{Reg}_{\mathsf{Sq}}(T)})$, where $\mathrm{Reg}_{\mathsf{Sq}}$ is the error of the regression oracle. The same bounds hold when the budget must be met in
    
[^229]: 编程智能体能否编写和转译单元测试

    Can Coding Agents Write and Transpile Unit Tests

    [https://arxiv.org/abs/2611.21797](https://arxiv.org/abs/2611.21797)

    本研究对三个编程智能体进行了基准测试，发现其编写的单元测试在覆盖率上远超人类测试套件，但测试质量随项目规模增大而急剧下降，且跨语言测试翻译仍不一致且不可靠。

    

    基于大语言模型（LLM）构建的编程助手已成为现代软件开发的重要组成部分，人们越来越信任它们端到端地执行工作流程和改造现代化代码。这种信任很大程度上建立在测试通过的基础上，而这些测试往往是由编程智能体自己编写的。在这项工作中，我们检验了这些智能体编写可靠测试以及在编程语言之间转换测试的能力。我们对三个流行的编程智能体进行了基准测试，每个都使用原生前沿模型和高推理强度进行评估。通过广泛的指标验证，我们发现编程智能体在传统覆盖率指标上远超人类编写的测试套件。然而，这一优势并不均衡：智能体的测试质量随项目规模增大而急剧下降，且其优势很大程度上源于编写了更多的测试数量。我们还表明，智能体测试翻译虽然在总体上看起来表现强劲，但仍然不一致且不可靠。

    arXiv:2611.21797v1 Announce Type: cross  Abstract: Coding assistants built on Large Language Models (LLMs) have become a staple of modern software development, increasingly trusted to execute workflows and modernize code end-to-end. Much of this trust rests on passing tests, which are often written by the coding agents themselves. In this work, we examine the ability of these agents to both write reliable tests and translate them between programming languages. We benchmark three popular coding agents, each evaluated with a native frontier model and high reasoning effort. Verifying across a broad suite of metrics, we find that coding agents far exceed human written suites on traditional measures of coverage. This advantage, however, is uneven: agent test quality degrades sharply with project size, and much of the advantage lies in a greater number of written tests. We also show that agentic test translation---while seemingly strong in aggregate---remains inconsistent and unreliable.
    
[^230]: 基于大语言模型的提升估计中的处理信息泄漏正则化

    Regularizing Treatment Leakage for LLM-based Uplift Estimation

    [https://arxiv.org/abs/2611.20794](https://arxiv.org/abs/2611.20794)

    该论文提出LULR方法，通过在冻结LLM编码器的同时对因果头共享表示施加基于CLUB的互信息惩罚，抑制LLM编码文本中泄露处理信息的捷径，从而改进工业定价RCT场景下的CATE提升估计效果。

    

    提升建模用于估计条件平均处理效应（CATE），并可支持促销定价等工业决策。当大语言模型（LLM）对产品文本进行编码时，暴露处理信息的内容可能引入无法迁移到实际部署阶段的捷径。我们提出了基于LLM的提升泄漏正则化方法（LULR），该方法在保持LLM编码器冻结的前提下，对因果头的可训练共享表示施加基于CLUB的惩罚项。受控代理实验显示出正的排序分数交互效应，且得分高于所测试的权重衰减配置；依赖处理的评估使其解释仅限于探索性的模型行为。仅使用结构化特征的对比提供了参考性能，但并未确立相对基于树的估计器的普遍优势。在一项包含百万条记录的工业定价随机对照试验（RCT）上，LULR在5个随机种子中的4个上改进了离线lift@0.3指标，尽管（摘要原文在此处截断）。

    arXiv:2611.20794v1 Announce Type: new  Abstract: Uplift modeling estimates the conditional average treatment effect (CATE) and supports industrial decisions such as promotional pricing. When large language models (LLMs) encode product text, treatment-revealing content can introduce shortcuts that do not transfer to deployment. We propose \emph{LLM-based Uplift Leakage Regularization} (\textsc{LULR}), which uses a CLUB-based penalty on the causal head's trainable shared representation while keeping the LLM encoder frozen. Controlled proxy experiments show a positive ranking-score interaction and higher scores than the tested weight-decay configuration; treatment-dependent evaluation limits their interpretation to exploratory model behavior. Structured-only comparisons provide reference performance, without establishing a general advantage over tree-based estimators. On a million-record industrial pricing RCT, \textsc{LULR} improves offline $\text{lift@}0.3$ in $4/5$ seeds, although the 
    
[^231]: 算力与推理市场：结构、融资与风险转移

    Markets for Compute and Inference: Structure, Financing, and Risk Transfer

    [https://arxiv.org/abs/2611.20242](https://arxiv.org/abs/2611.20242)

    该论文研究了新兴的算力租赁与推理token市场中的对冲问题，发现交易所合约结算所依据的公开短期租金率与运营商实际收入来源的私人多年期合同价格相互脱节，且租金率与token价格属于不同的风险敞口，因此对租金率进行对冲无法覆盖推理成本的风险。

    

    运行大型语言模型的算力按加速器小时出租，其产出按每百万token计价出售。这两种商品如今都有了公开的价格指数；其中一个指数已有场外远期合约在交易，其中一个的永续期货已在离岸市场上市，且CME和ICE已就租金指数期货合约做出规范，正等待美国监管机构的批准。我们探讨这两个市场中什么可以对冲、由谁对冲、以及以何种价格对冲。有三个发现尤为突出。第一，交易所合约所结算的价格并非驱动运营商收入的价格。每份合约都参考公开的短期租金率，而大部分收入以及与之相伴的信用质量，却来自价格基本不公开的私人多年期合同。第二，token价格与租金率通过吞吐量和利润率由一个恒等式联系起来，但它们是不同的风险敞口，因此对租金率的对冲并不能对冲推理的成本。

    arXiv:2611.20242v1 Announce Type: cross  Abstract: Capacity to run large language models is rented by the accelerator-hour, and their output is sold by the million tokens. Both goods now have published price indices, an over-the-counter forward has traded on one, a perpetual future on one is listed offshore, and CME and ICE have specified futures on rental indices that await the United States regulator. We ask what in these two markets can be hedged, by whom, and at what price. Three findings stand out. First, the price that exchange contracts settle on is not the price that drives operators' revenue. Every contract references the published short-dated rental rate, while most revenue, and with it credit quality, comes from private multi-year contracts whose prices are largely undisclosed. Second, the token price and the rental rate are linked by an identity through throughput and margin, but they are different exposures, so a hedge on the rental rate does not hedge the cost of inferenc
    
[^232]: 離题：面向结构化数据集目标变量预测与分类的零样本权重空间编程

    Zero-Shot Weight-Space Programming for Prediction and Classification of Target Variables in Structured Datasets

    [https://arxiv.org/abs/2611.19292](https://arxiv.org/abs/2611.19292)

    该论文证明大语言模型可以在零样本条件下，仅凭展示的小数据集与指定网络架构，无需优化器、梯度或代码执行，就手写出前馈网络的全部权重和偏置，在从玩具问题、合成问题到经典基准和真实工程任务的测试中直接完成结构化数据的目标变量预测与分类。

    

    权重空间编程是指以手工方式固定前馈网络的每一个参数，整个构建过程中不使用任何梯度下降。迄今为止的所有演示都是从以规则系统形式提供给模型的目标函数出发，这使其成为一个编译问题。而监督数据集并不提供这种规范说明，因此该过程必须自行完成统计推断，然后将结果用给定的参数化形式表达出来。我们探究大语言模型能否同时做到这两点。实验中向模型完整展示一个小数据集，并给出前馈网络的架构，由其将每一个权重和偏置都写成字面数字——全程不使用优化器、不使用梯度、不执行代码，也没有机会先进行评估。每个构建结果都被封存并预先登记其预期得分。评分测试套件从两个玩具问题开始，经过九个由隐藏旋转加以伪装的合成问题，直至经典基准测试以及一个真实工程问……

    arXiv:2611.19292v1 Announce Type: new  Abstract: Weight-space programming fixes every parameter of a feed-forward network by hand, with no gradient descent anywhere in the construction. Every demonstration so far starts from a target function handed to the model as a rule system, making it a compilation problem. A supervised dataset withholds that specification, so the procedure must do the statistical inference itself and then express the result in the parameterisation it was given. We ask whether a large language model can do both. It is shown a small dataset in full, with the architecture of a feed-forward network, and writes out every weight and bias as a literal number, with no optimizer, no gradient, no code execution, and no chance to evaluate it first. Each construction is sealed and pre-registers its expected score. The graded suite runs from two toy problems, through nine synthetic problems disguised by a hidden rotation, to the classical benchmarks and one real engineering r
    
[^233]: MarineSparkLM-4B：基于参数高效适配构建的海洋与离岸工程轻量级语言模型

    MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation

    [https://arxiv.org/abs/2611.19151](https://arxiv.org/abs/2611.19151)

    该研究通过在一台工作站级设备上仅对40亿参数模型进行0.81%参数量的LoRA高效适配，并以约5小时的低算力成本，构建出面向海洋与离岸工程领域的轻量级专用语言模型MarineSparkLM-4B。

    

    海洋与离岸工程领域的文本散布于预印本、开放获取期刊、百科全书和机构文档之中，且按预训练标准衡量其规模很小。我们探究的问题是：依靠参数高效适配，在一台工作站级单机上，面向海洋与离岸工程的轻量级语言模型究竟能走多远。从40亿参数的OceanGPT-basic-4B-Instruct出发，我们训练了两个独立的LoRA适配器，每个适配器仅更新40.6亿参数中的3300万（即0.81%）。在单台NVIDIA DGX Spark上，第一个适配器训练耗时4小时40分钟，第二个仅耗时10分钟。第一个适配器通过下一词元预测任务，在覆盖十个工程主题、共6160万词元的语料库上训练。该语料库由13,879份源文档构建，经去重和去除跨领域词汇冲突的两级关键词过滤后保留了7,069份文档。其验证损失……（原文在此截断）

    arXiv:2611.19151v1 Announce Type: cross  Abstract: Marine and offshore engineering text is spread across preprints, open-access journals, encyclopedias and institutional documents, and the amount of it is small by pretraining standards. We ask how far a lightweight language model for marine and offshore engineering can be taken with parameter-efficient adaptation on a single workstation-class device. Starting from the 4-billion-parameter OceanGPT-basic-4B-Instruct, we train two independent LoRA adapters, each updating 33.0 million of 4.06 billion parameters (0.81%). Training took 4 hours 40 minutes for the first adapter and 10 minutes for the second on one NVIDIA DGX Spark. The first adapter is trained by next-token prediction on a 61.6-million-token corpus covering ten engineering themes. The corpus was built from 13,879 source documents, of which 7,069 were retained after deduplication and a two-tier keyword filter that removes cross-domain lexical collisions. Its validation loss fal
    
[^234]: SPARC：面向材料发现的证据引导自适应模型组合

    SPARC: Evidence-guided adaptive model composition for materials discovery

    [https://arxiv.org/abs/2611.19050](https://arxiv.org/abs/2611.19050)

    SPARC是一个面向材料发现闭环实验的自适应模型组合框架，通过折外预测评估竞争模型族、以稀疏后验聚合分配模型权重，并利用预测离散度与残差变异性定义信任度来调节探索策略。

    

    闭环实验不仅需要选择下一个实验，还必须确定哪些现有测量结果和模型推导的关系获得了足够充分的支持，从而能够影响该决策。本文提出了SPARC（回归组件的稀疏预测聚合，Sparse Predictive Aggregation of Regression Components），这是一个随着实验证据积累而不断修正上述支持程度的框架。相互竞争的模型族通过折外预测进行评估，使每个模型在其对应拟合中未使用的观测数据上接受检验。稀疏后验组合根据折外证据所支持的预测贡献来分配各模型的影响力，同时保留来自多个模型族的互补贡献。候选依赖的预测离散度和残差变异性共同定义了信任度，该信任度与预测响应一起调节探索过程。独立的诊断机制还会对现有测量结果进行优先级排序，以便进行前瞻性的重新评估。

    arXiv:2611.19050v1 Announce Type: cross  Abstract: Closed-loop experimentation requires more than selecting the next experiment. It must also determine which existing measurements and model-derived relationships are sufficiently supported to influence that decision. Here we introduce SPARC (Sparse Predictive Aggregation of Regression Components), a framework that continually revises this support as experimental evidence accumulates. Competing model families are evaluated through out-of-fold predictions, so that each model is tested on observations not used in its corresponding fit. Sparse posterior composition assigns model influence according to predictive contributions supported by the out-of-fold evidence, retaining complementary contributions from several model families. Candidate-dependent predictive dispersion and residual variability define trust, which modulates exploration alongside the predicted response. Independent diagnostics also prioritize existing measurements for prosp
    
[^235]: 通过社会化实现智能体对齐：选择压力下的权衡学习

    Alignment by Socialization: Learning Tradeoffs Under Selection

    [https://arxiv.org/abs/2611.18531](https://arxiv.org/abs/2611.18531)

    该论文提出“社会化”这一对齐机制：在进化选择压力下，学习型智能体会收敛到环境所奖励的任务完成与行为规范之间的权衡；当由人类提供奖励时，智能体的选择便能反映人类判断并经受住后续的选择压力。

    

    人工智能（AI）智能体被期望在遵循行为准则的同时完成任务，但这些目标之间可能存在冲突。我们研究了进化选择如何塑造智能体在这两者之间的抉择，以及经验能否使这些选择反映人类的判断并承受后续的压力。在我们的模型中，一个智能体的多个副本面对共同的不确定奖励状态，只有采取了获得奖励行为的副本才能繁殖。我们证明，当每个行为被采取的频率与环境对其的奖励频率一致时，长期增长被最大化。我们比较了三种策略：仅任务策略在只奖励任务完成的沙盒环境中表现良好，但在行为规范至关重要时会失败；固定线性策略随着任务利害关系的增加越来越偏向任务完成，即使在人类从不奖励这种偏向的情况下也是如此；而学习型策略则会收敛到其环境所奖励的权衡上；当人类提供这些奖励时，我们将此过程称为社会化。积累的经验……

    arXiv:2611.18531v1 Announce Type: new  Abstract: Artificial intelligence (AI) agents are expected to complete tasks while following rules of conduct, but these objectives can conflict. We study how evolutionary selection shapes an agent's choices between them, and whether experience can make those choices reflect human judgments and withstand subsequent pressures. In our model, copies of an agent face a common uncertain reward state, and only those taking the rewarded action reproduce. We show that long-run growth is maximized when each action is taken as often as the environment rewards it. We compare three policies. A task-only policy thrives in a sandbox rewarding completion alone but fails when conduct matters. A fixed linear policy increasingly favors task completion as its stakes rise, even where humans never reward it. A learned policy instead converges to the tradeoffs rewarded by its environment; when humans supply the rewards, we call this socialization. Accumulated experienc
    
[^236]: 当评估在临床AI中变成干预

    When evaluation becomes intervention in clinical AI

    [https://arxiv.org/abs/2611.18297](https://arxiv.org/abs/2611.18297)

    临床AI中的AI评判者在被用于选择提示词或训练模型时会本身成为干预的一部分，并可能偏向其忽视的错误，因此评判者报告必须包含特定内容，且每项由评判者驱动的变更都需要独立的临床测试。

    

    AI评判者（AI judges）是一种依据评分标准对另一个系统的输出进行打分的模型，它将临床审查的规模扩展到了临床AI书写内容的规模。当团队使用评判者的评估结果来选择提示词或训练模型时，评判者本身就成为了干预的一部分。这种选择过程可能会偏向评判者所忽视的错误。我们描述了评判者的报告必须包含哪些内容，以及每一项由评判者驱动的变更所需的独立临床测试。

    arXiv:2611.18297v1 Announce Type: new  Abstract: AI judges, models that grade another system's output against a rubric, extend clinical review to the scale at which clinical AI writes. When teams use judge assessments to select a prompt or train a model, the judge becomes part of the intervention. Selection can then favor errors the judge overlooks. We describe what a judge's report must contain and the independent clinical test each judge-driven change needs.
    
[^237]: AscendGYM：面向昇腾NPU真实工作负载的成本感知算子优化基准

    AscendGYM: A Cost-Aware Kernel Optimization Benchmark for Real-World Workloads on Ascend NPUs

    [https://arxiv.org/abs/2611.17621](https://arxiv.org/abs/2611.17621)

    提出AscendGYM——首个面向昇腾NPU真实工作负载的成本感知算子优化基准，其48个算子的测试数据类型与张量形状均来自真实训练和推理任务，并支持模型评估与模型-框架评估两种模式来衡量智能体的算子优化能力。

    

    算子优化对于在昇腾NPU上高效进行大语言模型（LLM）训练和推理至关重要。现有的昇腾算子基准主要关注算子生成，较少关注智能体在来自真实工作负载的输入配置下优化现有实现的能力。为满足这一需求，我们提出了AscendGYM，这是一个基准测试，其初始套件包含48个算子，分为三个难度级别。每个算子都配有测试用例，其数据类型和张量形状均收集自真实的训练和推理任务。因此，在AscendGYM上取得的性能提升可以直接映射到相应的工作负载。在可用性方面，AscendGYM支持两种评估模式：模型评估和模型-框架评估，既可以衡量经过训练的模型，也可以衡量自定义智能体的算子优化能力。它提供了一个完整的执行环境……

    arXiv:2611.17621v1 Announce Type: new  Abstract: Kernel optimization is essential for efficient large language model (LLM) training and inference on Ascend NPUs. Existing Ascend kernel benchmarks primarily focus on kernel generation, with less emphasis on the ability of agents to optimize existing implementations under input configurations drawn from real workloads. To address this requirement, we introduce AscendGYM, a benchmark whose initial suite comprises 48 kernels across three difficulty levels. Each kernel is paired with test cases whose data types and tensor shapes are collected from real-world training and inference tasks. Consequently, performance improvements on AscendGYM map directly to the corresponding workloads. In terms of usability, two evaluation modes are supported in AscendGYM: model evaluation and model-harness evaluation. The kernel optimization capabilities of both trained models and custom agents are allowed to be measured. It provides a complete execution envir
    
[^238]: 基于潜在状态编码的交互式强化学习中的广泛持久性建议方法

    Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding

    [https://arxiv.org/abs/2611.13712](https://arxiv.org/abs/2611.13712)

    本文提出Encoder BPA方法，用潜在状态编码器替代传统K-means聚类，通过潜在比较、top-k聚合和一致性条件来检索和重用外部建议，从而提升了交互式强化学习早期的学习效率和建议利用率。

    

    深度交互式强化学习（Deep IntRL）通过引入外部建议来加速强化学习的训练。然而，传统的深度交互式强化学习仅在当前步骤使用每条建议，并在单次交互后将其丢弃。持久建议和广泛持久建议通过将建议存储在持久内存中，并基于K-means聚类的状态泛化在相似状态中重用建议，从而扩展了这一过程。本研究提出了Encoder BPA，该方法采用潜在编码器来检索和重用建议。在该框架中，使用训练好的编码器在每个时间步将状态序列映射为潜在表示。建议可以通过潜在比较进行检索，同时结合top-k聚合和一致性条件以提高重用的稳定性。实验结果表明，Encoder BPA在训练早期阶段提高了学习效率和建议利用率。

    arXiv:2611.13712v1 Announce Type: new  Abstract: Deep Interactive Reinforcement Learning (Deep IntRL) accelerates reinforcement learning training by incorporating external advice. However, conventional Deep IntRL uses each piece of advice only in the current step and discards it after a single interaction. Persistent Advising and Broad-Persistent Advising (BPA) extend this process by storing advice in persistent memory and reusing it in similar states through state generalisation based on K-means clustering. This research proposes Encoder BPA, which employs a latent encoder to retrieve and reuse advice. In this framework, a trained encoder is used to map a state sequence into a latent representation at every time step. Advice can be retrieved through latent comparison, together with a top-(k) aggregation and a consistency condition to improve reuse stability. Experimental results show that Encoder BPA improves learning efficiency and advice utilisation at an early stage of training com
    
[^239]: SafeRoute-MAS：面向多智能体科学推理的覆盖率感知与风险校准消息路由

    SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning

    [https://arxiv.org/abs/2611.13454](https://arxiv.org/abs/2611.13454)

    SafeRoute-MAS提出了一种仅利用集成前信息、覆盖率感知且风险校准的消息路由程序，在多智能体科学推理中通过选择性移除专家消息，在减少约12%–19%消息令牌的同时带来一定的准确率提升，但严格的统计验证显示其增益有限。

    

    我们提出了SafeRoute-MAS，一种覆盖率感知、风险校准的程序，它仅使用集成前的信息来决定保留所有专家消息还是移除其中一条。在一个包含5,328个委员会的密封缓存上，其与源兼容的工作点获得了+1.80个百分点的增益（95%置信区间[+1.13, +2.52]），同时移除了12.76%的消息令牌。我们还在公共gpt-oss-120b端点上，对一个458个委员会的分层样本进行了4,580次受控集成。样本估计值为-0.09个百分点；按原始工作负载进行设计加权后为+0.94个百分点（95%置信区间[-1.31, +3.50]），实际输入减少18.72%。加权的令牌匹配比较同样未能确立选择器的优势（+1.68个百分点，置信区间[-1.27, +5.10]）。88个缓存的非中性标签中仅有13个（14.8%）在重复调用多数投票下保持了方向一致性。加权后的预期损坏上置信限（UCL）为4.60%，但11个预设边际中仅有2个通过支持/风险规则。因此我们……（原文摘要在此处截断）

    arXiv:2611.13454v1 Announce Type: new  Abstract: We introduce SafeRoute-MAS, a coverage-aware, risk-calibrated procedure that keeps all specialist messages or removes one using only pre-integration information. On a sealed cache of 5,328 committees, its source-compatible operating point gains +1.80 percentage points (95% CI [+1.13, +2.52]) while removing 12.76% of message tokens. We also run 4,580 controlled integrations over a stratified 458-committee sample on a public gpt-oss-120b endpoint. The sample estimate is -0.09 points; design weighting to the original workload gives +0.94 points (95% CI [-1.31, +3.50]) and 18.72% acted-input reduction. A weighted token-matched comparison likewise does not establish a selector advantage (+1.68 points, CI [-1.27, +5.10]). Only 13 of 88 cached non-neutral labels (14.8%) retain direction by repeated-call majority. Expected-corruption UCL is 4.60% after weighting, but only 2 of 11 prespecified marginals pass the support/risk rule. We therefore pr
    
[^240]: 删除敏感性是模型的一种属性：对七个开放推理模型的三条件测量

    Deletion Sensitivity Is a Property of the Model: A Three-Condition Measurement across Seven Open Reasoning Models

    [https://arxiv.org/abs/2611.13223](https://arxiv.org/abs/2611.13223)

    本文首次通过三种渲染条件、七个开放权重推理模型、三个种子和794个任务的盲测实验，系统测量了多轮工具调用中删除完整推理历史对任务性能的影响，发现相同删除操作导致的性能变化跨度达-4.4到+51.3分，且这种删除敏感性是模型层面的固有属性。

    

    推理模型在多轮工具会话的每一轮都会写下一条推理轨迹，而后续轮次会读取这段历史。在94k条ShareGPT对话中，73%的用户交互跨越多轮。上游提供商将每条推理轨迹绑定到其发出者，因此跨发出者转发的请求到达时其历史已被删除，而这种剥离操作带来的任务成本此前从未被测量。据我们所知，本文首次对多轮工具调用中完整推理历史删除进行了直接、盲测、多种子的测量，涵盖三种渲染条件、七个开放权重推理模型、三个随机种子、794个任务以及官方检查器。相同的删除操作导致的性能变化跨度为-4.4到+51.3分，且两端均不包含零；对于唯一一个从0%到100%删除比例重新测试的模型，其读数变化最多仅1.5分。删除敏感性是模型层面的属性。该读数跟随模型本身并保持稳定（原文在此处截断）。

    arXiv:2611.13223v1 Announce Type: cross  Abstract: Reasoning models write a reasoning trace at every turn of a multi-turn tool session, and later turns read this history. Across 94k ShareGPT conversations, 73 percent of user interactions span multiple turns. Upstream providers bind each trace to its issuer, so a request forwarded across issuers arrives with its history deleted, and the task cost of this stripping remains unmeasured. To the best of our knowledge, this paper presents the first direct, blind, multi-seed measurement of full reasoning-history deletion in multi-turn tool calling, through three rendering conditions, seven open-weight reasoning models, three seeds, 794 tasks, and the official checker. The same deletion spans -4.4 to +51.3 points with both ends excluding zero, and for the one model re-dosed from 0 to 100 percent deletion, the reading varies by at most 1.5 points. Deletion sensitivity is a model-level property. The reading follows the model itself and stays cons
    
[^241]: MARS：用于视频伪造检测的多尺度外观-残差流

    MARS: Multiscale Appearance-Residual Streams for Video Forgery Detection

    [https://arxiv.org/abs/2611.12508](https://arxiv.org/abs/2611.12508)

    MARS提出了一种双路径视频伪造检测器，通过对冻结特征施加多尺度空间-时间高斯差分算子提取残差差异，并与自适应外观表示及跨片段门控机制相结合，从而在面部篡改和全生成视频两类深度伪造检测上实现更强的泛化能力。

    

    由于伪造痕迹会随篡改方法和场景内容而变化，使深度伪造检测在面部篡改与完全生成视频两类之间保持泛化能力仍然具有挑战性。显式的空间与时间特征差异提供了一种捕捉这种变化的途径，但它们同样会对自然纹理和运动产生响应，因此需要将其与外观特征进行联合解释。我们研究了将多尺度特征差异与自适应外观表示相结合是否能提升对这两类伪造的检测效果。我们提出了多尺度外观-残差流，这是一种双路径视频检测器。残差路径对冻结的中间特征应用固定的空间与时间高斯差分算子，并结合跨帧的外观上下文对所得差异进行建模。语义路径通过自适应调整编码器的归一化参数来进行帧分类，并由一个依赖于片段的门控机制……

    arXiv:2611.12508v2 Announce Type: replace-cross  Abstract: Generalizing deepfake detection across facial manipulation and fully generated video remains challenging due to forgery traces varying with the manipulation method and scene content. Explicit spatial and temporal feature differences offer a way to capture this variation, but also respond to natural texture and motion, motivating their joint interpretation with appearance features. We investigate whether combining multiscale feature differences with adapted appearance representations improves detection across both forgery categories. We introduce Multiscale Appearance-Residual Streams (MARS), a two-path video detector. The residual path applies fixed spatial and temporal difference-of-Gaussians operators to frozen intermediate features and models the resulting differences alongside appearance context across frames. The semantic path adapts the encoder's normalization parameters for frame classification, and a clip-dependent gate
    
[^242]: TS-GEAR：淘宝闪购生成式广告检索

    TS-GEAR: Taobao Shangou GEnerative Ad Retrieval

    [https://arxiv.org/abs/2611.12180](https://arxiv.org/abs/2611.12180)

    该论文提出TS-GEAR，面向即时配送场景的生成式广告检索，通过显式建模配送范围可达性、基于地理位置与当前查询条件化的用户兴趣表示以及集合级训练信号，解决了语义ID地理信号软化、兴趣估计偏差与候选冗余三大挑战。

    

    诸如DoorDash和Uber Eats这类即时配送平台存在一个传统电商所没有的约束：超出用户配送范围的商家是无效候选，而不仅仅是排序较低。生成式检索近期在大规模检索系统中展现出强大潜力，然而将其适配到这一场景面临三大挑战。第一，现有语义ID仅将地理位置作为软信号编码，因此可达性只能依赖生成之后的不可微硬截断来保证。第二，用户偏好在不同区域和不同查询之间差异显著，若在不结合地理位置与当前查询的情况下聚合历史行为，会产生有偏的兴趣估计。第三，检索到的候选集合以其整体价值来评估，然而自回归训练对每个序列独立打分，缺乏集合级信号来抑制冗余的近似重复候选。为解决……（摘要在此处被截断）

    arXiv:2611.12180v2 Announce Type: replace-cross  Abstract: On-demand delivery platforms such as DoorDash and Uber Eats impose a constraint absent in traditional e-commerce: merchants beyond the user's delivery range are invalid candidates rather than merely low-ranked. Generative retrieval has recently shown strong potential in large-scale retrieval systems. Adapting it to this setting, however, poses three challenges. First, existing semantic IDs encode geography only as a soft signal, so reachability can only rely on a non-differentiable post-generation hard truncation. Second, user preferences vary substantially across both regions and queries, so aggregating historical behaviors without conditioning on geography and the current query yields biased interest estimates. Third, the retrieved candidate set is evaluated by its collective value, yet autoregressive training scores each sequence independently, with no set-level signal to discourage redundant near-duplicates. To address the 
    
[^243]: 合并之前先审计：来源溯源、探测与持续 LoRA 组合

    Audit Before You Merge: Provenance, Probing, and Continual LoRA Composition

    [https://arxiv.org/abs/2611.10684](https://arxiv.org/abs/2611.10684)

    该论文提出在合并公开 LoRA 适配器前必须进行来源审计（基础模型匹配与非平凡增量范数检查）和无泄漏探测协议，并证明朴素合并会造成显著性能下降，而任务算术可将性能维持在未合并基础模型的标准误差范围内。

    

    arXiv:2611.10684v2 公告类型：替换。摘要：对公开 LoRA 适配器进行事后合并是一条无需梯度的有吸引力的持续适应路径，但在该设置下对灾难性遗忘的可靠测量依赖于此前工作尚未解决的基础设施。我们将每次适配器合并视为持续学习设置中的一个顺序任务，将指令微调基础模型的能力视为任务 0，并在 Llama-3.1-8B-Instruct 上研究遗忘现象。公开适配器库中包含无法通过基本来源检查的模块：隐式的基础模型版本不匹配以及接近零的增量范数，将这些模块纳入会使测得的干扰膨胀约 10 个百分点。我们提出了一项双重检查审计（基础模型匹配、非平凡增量范数）和无泄漏探测协议，作为在该设置下开展任何遗忘研究的前提条件。在一个经过审计的 4 适配器集合上，朴素合并导致 GSM8K 下降 10.5 个百分点、HumanEval 下降 18.3 个百分点，而 lambda=0.5 的任务算术则与未合并的基础模型处于同一标准误差范围内。

    arXiv:2611.10684v2 Announce Type: replace  Abstract: Post-hoc merging of public LoRA adapters is an attractive gradient-free path to continual adaptation, but reliable measurement of catastrophic forgetting in this setting depends on infrastructure that prior work has not addressed. We treat each adapter merge as a sequential task in a continual-learning setting, with the instruction-tuned base model's capabilities as Task 0, and study forgetting on Llama-3.1-8B-Instruct. Public adapter hubs contain modules that fail basic provenance checks: silent base-version mismatches and near-zero delta norms whose inclusion inflates measured interference by ~10pp. We propose a two-check audit (base-model match, non-trivial delta norm) and a leakage-free probing protocol as preconditions for any forgetting study in this setting. On an audited 4-adapter set, naive merging causes -10.5pp on GSM8K and -18.3pp on HumanEval, while Task Arithmetic at lambda=0.5 sits within a standard error of the unmerg
    
[^244]: 物联网边云连续体中面向SLA的任务卸载：以精确最优解为基准的类别加权深度强化学习

    SLA-Aware Task Offloading in the IoT Edge-Cloud Continuum: Class-Weighted Deep Reinforcement Learning Against an Exact Optimum

    [https://arxiv.org/abs/2611.10638](https://arxiv.org/abs/2611.10638)

    该研究通过四个可精确求解的基准系统证明：在物联网边云任务卸载中，类别盲的惩罚设计即使配合完美学习器也会导致成本高出最优解65%-118%，而修正后的按SLA类别加权的奖励设计可使PPO智能体性能逼近精确最优解。

    

    深度强化学习（RL）常被用于决定具有不同延迟限制的物联网（IoT）任务在边云连续体中的运行位置，然而一些方法对每个未按时完成的任务施加相同的惩罚，且很少有研究将训练出的智能体与最优策略进行比较。本基准研究构建了四个规模小到可通过动态规划精确求解的任务卸载系统，并将奖励设计与学习器性能对照最优解进行衡量。为满足延迟限制的任务设置奖金，会使每次错失任务的成本变为惩罚加奖金之和，从而改变了已发表的按类别奖励方案中的类别权重。研究发现：即使在完美学习器的情况下，类别盲（class-blind）惩罚的成本仍比最优成本高出65%至118%，而修正后的按类别惩罚则与最优解相差不超过0.13%。在使用按类别奖励时，经过调优的近端策略优化（PPO）智能体在最小系统中的表现与最优解相差9%至11%，且修正后的方法在10个随机种子中有9个击败了调优后的预留规则。在更大的系统中……

    arXiv:2611.10638v2 Announce Type: replace  Abstract: Deep reinforcement learning (RL) is often used to decide where Internet of Things (IoT) tasks with different latency limits run in the edge-cloud continuum, yet some methods give every missed task the same penalty and few compare their agents with an optimal policy. This benchmark study builds four offloading systems small enough to solve exactly by dynamic programming and measures reward and learner against the optimum. A bonus for a met limit makes each miss cost its penalty plus the bonus, which shifts the class weights of published per-class rewards. A class-blind penalty stays 65% to 118% above the optimal cost even with a perfect learner, while a corrected per-class penalty comes within 0.13% of the optimum. With a per-class reward, tuned proximal policy optimization (PPO) agents come within 9% to 11% of the optimum in the smallest system, and the corrected one beats a tuned reservation rule in 9 of 10 seeds. In larger systems 
    
[^245]: 跨越双人游戏难度分级阶梯的零样本权重空间编程

    Zero-Shot Weight-Space Programming Across a Difficulty-Graded Ladder of Two-Player Games

    [https://arxiv.org/abs/2611.09271](https://arxiv.org/abs/2611.09271)

    该论文首次系统考察大语言模型零样本权重空间编程的能力边界，在涵盖四个难度维度的双人游戏四级阶梯上零样本手工设定前馈网络的全部权重，构建并独立验证了十个网络，其中六个获得完整且非统计性的正确性证明。

    

    权重空间编程是指依据对目标计算的显式规范，手工设定前馈网络的每一个权重，整个构造过程中完全不使用梯度下降，也不使用任何训练数据。我们探究大语言模型能否以零样本方式将这一实践推广到难度刻意递增的游戏上。我们设计了一个四级阶梯（尼姆棋、井字棋、两种棋盘尺寸的四子棋以及一个国际象棋残局），涵盖四个难度维度，并在每一级额外构建一个结构上互不相关、难度相当的游戏网络，以区分某一层级的结论究竟反映的是该难度水平本身，还是仅反映用于测量的那一个游戏。共构建了十个网络并进行独立验证，其中六个网络可针对独立预言机给出完整且非统计性的正确性证明。国际象棋残局层级在正确性代价较低的部分（走法合法性，以及其中一个网络中的……）是精确的。（原文摘要在此处被截断）

    arXiv:2611.09271v3 Announce Type: replace  Abstract: Weight-space programming sets every weight of a feed-forward network by hand, from an explicit specification of the desired computation, with no gradient descent and no training data anywhere in the construction. We ask how far a large language model can carry this practice, zero-shot, across games of deliberately increasing difficulty. We design a four-rung ladder (Nim, tic-tac-toe, two sizes of Connect Four, and a chess endgame) spanning four axes of difficulty, and build a second, independent network at each rung, using a structurally unrelated game at matched difficulty, to separate what a rung's finding says about the difficulty level from what it says about the one game used to measure it. Ten networks are built and independently verified. Six admit a complete, non-statistical proof of correctness against an independent oracle. The chess-endgame rung is exact where correctness is cheap, move legality and, in one network, an exa
    
[^246]: 直积空间中的等变深度学习用于电子结构哈密顿量预测

    Equivariant Deep Learning in Direct-Product Space for Electronic-Structure Hamiltonian Prediction

    [https://arxiv.org/abs/2611.07695](https://arxiv.org/abs/2611.07695)

    提出DP-Learner框架，在直积空间中直接对类哈密顿量算符块进行等变消息传递，同时解决了传统直和表示与哈密顿量之间的表示不匹配问题及计算效率瓶颈。

    

    大多数现有的用于哈密顿量预测的等变架构是在不可约球张量的直和空间中传播特征，而目标哈密顿量本质上是局部轨道和自旋基的直积表示中的一个算符块。这种表示不匹配割裂了连续的哈密顿量块，模糊了对轨道耦合的直接操作，并将特征组织成不规则的角动量分量，这些分量需要大量变换，且难以与高效的硬件执行和内存管理相适配。为了同时解决表示不匹配和计算瓶颈问题，我们提出了DP-Learner，一个在直积空间中直接对类哈密顿量算符块进行消息传递的等变学习框架。在理论上，我们提出了基于李生成子的方向嵌入以及不变加权的左右乘积与收缩运算……

    arXiv:2611.07695v1 Announce Type: cross  Abstract: Most existing equivariant architectures for Hamiltonian prediction propagate features in direct-sum spaces of irreducible spherical tensors, whereas the target Hamiltonian is naturally an operator block in a direct-product representation over local orbital and spin bases. This representation mismatch fragments contiguous Hamiltonian blocks, obscures direct manipulation of orbital couplings, and organizes features into irregular angular-momentum components that require numerous transformations and are poorly aligned with efficient hardware execution and memory management. To address both the representation mismatch and computational bottleneck, we propose DP-Learner, an equivariant learning framework that performs message passing directly on Hamiltonian-like operator blocks in the direct-product space. Theoretically, we propose Lie-generator-based directional embedding and invariantly weighted left--right products and contractions that 
    
[^247]: 调适适应的进程：人工智能中的达尔文式、拉马克式与智能体导向演化

    Adapting the Process of Adaptation: Darwinian, Lamarckian, and Agent-Directed Evolution in AI

    [https://arxiv.org/abs/2611.07048](https://arxiv.org/abs/2611.07048)

    该论文提出了一个比较达尔文式、拉马克式与智能体导向演化的理论框架，发现当学习足够有效时三种机制的适应速度依次加快，生成更多变体可加速适应，而最优搜索努力取决于学习能力、问题难度与实验成本。

    

    演化在数十亿年间塑造了智能。人工智能（AI）如今提出了一个引人注目的可能性：智能或许会开始引导自身的演化。当AI智能体变得能够修改并评估自身行为时，它们能以多快的速度改进？这种改进又如何保持安全？我们建立了一个理论框架，比较环境变化之后的三种演化方式：达尔文式演化依赖自然选择，拉马克式演化在此基础上加入了可遗传的学习，而智能体导向演化则允许智能体生成并评估行为变体，并自主决定搜索的深度。当学习足够有效时，这些机制表现出逐级加快的适应速度。生成更多变体可以加速适应，而最优搜索努力取决于学习能力、问题难度和实验成本。然而，智能体的能力……（原文摘要在此处截断）

    arXiv:2611.07048v1 Announce Type: new  Abstract: Evolution has shaped intelligence for billions of years. Artificial intelligence (AI) now raises a striking possibility: intelligence may begin to direct its own evolution. As AI agents become capable of modifying and evaluating their own behavior, how quickly could they improve, and how can that improvement remain safe? We develop a theoretical framework comparing Darwinian, Lamarckian, and agent-directed evolution following environmental change. Darwinian evolution relies on natural selection, Lamarckian evolution adds inherited learning, and agent-directed evolution allows agents to generate and evaluate behavioral variants and choose how extensively to search. When learning is sufficiently effective, these mechanisms exhibit progressively faster adaptation. Generating more variants accelerates adaptation, while optimal search effort depends on learning ability, problem difficulty, and experimentation costs. However, an agent's abilit
    
[^248]: Unsat​z：在没有给定拟设的情况下评估科学发现

    Unsatz: Evaluating Scientific Discovery Without a Given Ansatz

    [https://arxiv.org/abs/2611.03713](https://arxiv.org/abs/2611.03713)

    该论文提出了 Unsat​z 基准，首次要求 LLM 智能体在没有给定模型形式（拟设）的情况下自主构建拟设——在有限预算内设计实验并用统一建模语言编写候选模型程序——从而更真实、更完整地评估人工智能的科学发现能力。

    

    在科学家拟合任何参数之前，他们必须先决定模型中包含什么：哪些量存在、哪些量是隐藏的、它们如何相互作用，以及如何被测量。物理学家将这种尝试性的模型形式称为“拟设”。现有的自动化方程发现基准测试大多直接提供了拟设——通过固定变量、给定候选项库，或提供待修改的定律。我们提出了 Unsat​z，一个要求 LLM 智能体自行构建拟设的基准测试。每个任务都是一个由私有种子生成的模拟系统，以一个小型建模语言编写成的程序形式呈现：可以是常微分方程、时滞微分方程或偏微分方程、离散映射、静态与因果模型，或随机过程，其中可能包含潜在状态、失真的传感器以及无信息量的测量通道。智能体需要在预算内购买实验，用同一建模语言编写候选程序，从基准测试提供的拟合器中获得系数，并仅根据其最终程序……（原文摘要在此处截断）

    arXiv:2611.03713v2 Announce Type: replace  Abstract: Before a scientist fits any parameter, they must decide what the model contains: which quantities exist, which are hidden, how they interact, and how they are measured. Physicists call such a trial form an ansatz. Benchmarks for automated equation discovery mostly supply it, by fixing the variables, a library of candidate terms, or a law to be modified. We introduce Unsatz, a benchmark in which an LLM agent must construct the ansatz itself. Each task is a simulated system, generated from a private seed as a program in a small modeling language: ordinary, delay, or partial differential equations, discrete maps, static and causal models, or stochastic processes, possibly with latent states, distorted sensors, and uninformative measurement channels. The agent buys experiments within a budget, writes candidate programs in the same language, obtains their coefficients from the benchmark's fitter, and is scored only on how its final progra
    
[^249]: 追溯大语言模型科学发现能力与其训练数据的关联

    Tracing the Scientific Discovery Capacity of LLMs to Their Training Data

    [https://arxiv.org/abs/2611.03617](https://arxiv.org/abs/2611.03617)

    该研究提出“单次事实”理论，通过直接操纵训练数据，证明训练集中仅出现一次的事实比率是决定大语言模型科学发现能力的关键因素，且这一能力可在不同模型、领域间迁移。

    

    大型语言模型（LLM）在科学发现方面展现出越来越大的潜力，但关于什么因素使其能够可靠地做出发现，一些基本问题仍未解决。我们如何将LLM的科学发现能力追溯到其训练数据？我们将这一问题作为“缺失概率质量”问题进行研究，并建立了一个理论框架，其中“单次事实”——即训练中仅被观察到一次的事实——决定了发现能力。在材料科学和调控基因组学领域，我们直接操纵训练集中的单次事实比率，发现在两个开放权重模型家族、不同采样策略和推理预算下，更多的单次事实都能持续带来更多发现。进一步的实验表明，这种关系不能仅用训练集的多样性来解释，并且在与领域内数据配对时，这种能力可以跨科学领域迁移。总之，我们的结果确定了训练数据中的单次事实比率是决定语言模型发现能力的关键因素。

    arXiv:2611.03617v1 Announce Type: new  Abstract: Large language models (LLMs) show growing promise for scientific discovery, yet basic questions remain about what enables reliable discovery. How can we trace the scientific discovery capacity of LLMs to their training data? We study this question as one of missing probability mass and develop a theoretical account in which monofacts-the facts observed once during training-set the discovery capacity. In materials science and regulatory genomics, we directly manipulate monofact rates in training sets and find that more monofacts consistently yield more discoveries across two open-weight model families, sampling strategies, and reasoning budgets. Further experiments show the relationship cannot be explained by training-set diversity alone and can transfer across scientific domains when paired with in-domain data. Together, our results identify monofact rate in training data as a key factor governing discoveries in language models.
    
[^250]: LUCA：一个用于实验推理的合成世界

    LUCA: A Synthetic World for Experimental Reasoning

    [https://arxiv.org/abs/2611.03100](https://arxiv.org/abs/2611.03100)

    该论文提出LUCA——一个在完全已知真值下评估智能体实验推理过程的可配置合成世界基准，通过让智能体在有预算约束下学习未知调控系统并分别考察实验选择、预测、诊断和干预能力，发现前沿大语言模型在这些能力上显著分离，常常在误诊故障的情况下仍能成功拯救系统。

    

    [注，2026年11月：本版本中的实验结果受到模拟环境中实现问题的影响，在修正之前不应依赖这些结果，请参阅第1页的作者说明。] 一个实验智能体可能因错误的原因而取得成功。然而，现有的评估往往只关注最终结果，而非产生该结果的实验、证据和推理过程。我们提出了LUCA，一个可配置的合成世界，用于在完全已知真值的条件下评估这一过程。智能体通过有预算限制的实验来学习未知的调控系统，然后预测未见过的干预、诊断隐藏的故障并尝试实施拯救。这一设置使我们能够分别评估实验选择、预测、诊断和干预这几种能力。在七个前沿大语言模型和三个基线方法的测试中，这些能力表现出显著的分离：智能体经常在诊断出错误故障的情况下仍能成功拯救故障系统；诊断能力之间相差7倍（原文在此处截止）。

    arXiv:2611.03100v3 Announce Type: replace  Abstract: [Note, November 2026: the experimental results in this version are affected by implementation issues in the simulation environment and should not be relied upon pending correction. See the author note on page 1.]   An experimental agent can succeed for the wrong reasons. Evaluations, however, often focus on the outcome, not the experiments, evidence, and inferences that produced it. We introduce LUCA, a configurable synthetic world for evaluating this process under fully known ground truth. Agents learn unknown regulatory systems through budgeted experimentation, then predict unseen interventions, diagnose hidden faults, and attempt rescues. This setup allows us to evaluate experiment selection, prediction, diagnosis, and intervention separately.   Across seven frontier LLMs and three baselines, these capabilities dissociate sharply. Agents frequently rescue malfunctioning systems while diagnosing the wrong fault; diagnosis varies 7x
    
[^251]: 为什么表格基础模型能在它们从未见过的关系数据上表现出色

    Why Tabular Foundation Models Excel on Relational Data They Have Never Seen

    [https://arxiv.org/abs/2611.02936](https://arxiv.org/abs/2611.02936)

    本文通过多样化实验证实并深入揭示了单表表格基础模型在结合无参数关系编码器后，即使从未经过任何关系数据预训练，也能在关系数据库预测任务上表现出色的内在机制。

    

    给定一个存储异构表格信息的关系数据库（RDB），我们如何预测某个感兴趣的目标列中缺失（或未来）的值？由于企业环境中的候选目标众多，最好避免在每次出现新预测任务时都从头训练一个新模型。基于上下文学习（ICL）和关系数据库特定预训练的冻结基础模型（FM）提供了一种可行的解决方案，但理想的设计在很大程度上仍悬而未决。也许出乎意料的是，最近的初步证据表明，现有的单表基础模型（即TFM）在与合适的无参数关系编码器结合后，尽管完全没有接触过任何关系数据库预训练或关系数据，在实践中仍然具有相当强的竞争力。在这项工作中，我们首先通过更多样化的实验从实证角度证实了这一现象。随后我们转向本文的主要贡献，即揭示多种互补的（摘要在此处截断）

    arXiv:2611.02936v3 Announce Type: replace  Abstract: Given a relational database (RDB) storing heterogeneous tabular information, how can we predict missing (or future) values in some target column of interest? As candidate targets are numerous in enterprise settings, it is preferable to avoid learning a new model from scratch each time there is a new prediction task. Frozen foundation models (FMs) based on in-context learning (ICL) and RDB-specific pre-training provide a viable solution, but the ideal design still remains largely unresolved. Perhaps unexpectedly, preliminary evidence has recently shown that existing single-table FMs (aka TFMs), when combined with suitable parameter-free relational encoders, can be quite competitive in practice despite zero exposure to RDB pre-training or relational data of any kind. In this work we first empirically confirm this phenomenon with a more diverse array of experiments. We then turn to our main contribution, namely, surfacing multiple compl
    
[^252]: dKV Forcing：通过可微KV缓存分离生成与记忆表征

    dKV Forcing: Separating Generation and Memory Representations through a Differentiable KV Cache

    [https://arxiv.org/abs/2611.01648](https://arxiv.org/abs/2611.01648)

    提出dKV Forcing方法，为KV缓存写入分配独立的低秩残差参数，实现生成与记忆表征的解耦，并利用后续片段读取的梯度来优化缓存写入，从而改进自回归视频生成器的训练。

    

    自回归视频生成器会将其生成的每个片段写入键值（KV）缓存，而后续片段只能通过该缓存了解过去。自回放训练在缓存处截断了梯度，而自梯度强制（SGF）恢复了这一梯度，但将其应用于生成当前片段的同一组权重上。我们发现写入与生成是两种不同的角色：在预训练生成器中，写入的KV已经与用于去噪的KV存在差异，并且在共享权重的情况下，来自未来读取的梯度与生成梯度呈负对齐。我们提出dKV Forcing（dKV表示可微KV），为写入操作赋予其独立的参数——一个仅在片段进入缓存时启用的低秩残差——并利用后续读取其所写内容的片段的梯度来训练这些参数。在常见使用场景下，通过缓存打开这一梯度会耗尽GPU内存。我们将其归因于检查点重放对缓存的重写……（原文摘要在此处截断）

    arXiv:2611.01648v2 Announce Type: replace-cross  Abstract: A causal video generator writes every segment it produces into a key--value (KV) cache, and later segments know the past only through it. Self-rollout training stops the gradient at this cache, and Self Gradient Forcing (SGF) restores it, but applies it to the same weights that generate the current segment. We find that writing and generating are different roles: in a pretrained generator the written KV already differs from the KV used for denoising, and on shared weights the gradient from future reads is negatively aligned with the gradient of generation. We propose dKV Forcing (dKV denotes differentiable KV), which gives the write its own parameters, a low-rank residual used only when a segment enters the cache, and trains them with the gradient of the later segments that read what they wrote. Opening this gradient through the cache in common use exhausts GPU memory. We trace this to checkpoint replay re-writing the cache, no
    
[^253]: WorldFlow：通过大规模预训练实现视频与3D运动的联合预测

    WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining

    [https://arxiv.org/abs/2610.39278](https://arxiv.org/abs/2610.39278)

    WorldFlow是一种生成式世界模型，它将3D运动参数化为与视频网格对齐的连续位移标记，并利用统一的Mixture-of-Transformers骨干网络在整流流目标下联合去噪视频与3D两种模态，从而将显式几何动力学与视觉演化耦合，释放预训练视频模型中蕴含的物理常识。

    

    预测未来的物理状态是具身智能中的一个核心挑战。现有的视频世界模型将3D动力学隐式化，基于动作的公式化仍然绑定于特定的机器人配置，而3D运动预测模型仅使用光流监督，为物理动力学提供的学习信号十分有限。为了规避这些限制，我们提出了WorldFlow，一种能够联合预测未来视频和基于查询的3D轨迹的生成式世界模型。通过将3D运动参数化为与视频网格对齐的连续位移标记，一个统一的混合Transformer（Mixture-of-Transformers, MoT）骨干网络在整流流（rectified-flow）目标下对两种模态进行联合去噪。这种公式化将显式的几何动力学与视觉演化耦合在一起，有效地释放了预训练视频模型中蕴含的物理常识。为了在实证上扩展并验证这一统一架构，我们在异构（摘要在此处截断）……上训练WorldFlow。

    arXiv:2610.39278v1 Announce Type: cross  Abstract: Predicting future physical states stands as a central challenge in embodied intelligence. Existing video world models leave 3D dynamics implicit, action-based formulations remain tied to specific robot configurations, and 3D motion prediction models use flow-only supervision, providing a limited learning signal for physical dynamics. To circumvent these limitations, we introduce \method{}, a generative world model that jointly predicts future video and query-based 3D trajectories. By parameterizing 3D motion as continuous displacement tokens aligned with the video grid, a unified Mixture-of-Transformers (MoT) backbone jointly denoises both modalities under a rectified-flow objective. This formulation couples explicit geometric dynamics with visual evolution, effectively unlocking the physical commonsense embedded within pretrained video models. To empirically scale and validate this unified architecture, we train \method{} on a heterog
    
[^254]: 让门控决定：大语言模型中量化条件内存的逐查找精度

    Let the Gate Decide: Per-Lookup Precision for Quantized Conditional Memory in LLMs

    [https://arxiv.org/abs/2610.35985](https://arxiv.org/abs/2610.35985)

    GateRQ利用可从1比特编码计算的门控值（对频繁n-gram小、对长尾n-gram大）来决定各查找所需的量化精度，将每行存储为1比特残差编码阶梯，无需重训即实现GPU表内存7.8倍压缩和vLLM吞吐量11%-31%的提升。

    

    条件记忆层（如DeepSeek的Engram）将大型哈希n-gram表附加到语言模型上，从DeepSeek-V4.1-Flash中移除该表会损失36.8个PopQA分数。低比特量化会最先丢失长尾事实，而标准补救措施——为频繁访问的行分配更多比特——反而适得其反：按访问频率进行精度提升的效果比随机分配还差。我们将这一问题追溯到缩放每个检索行的门控：一次查找的量化误差由其门控加权，该门控对频繁出现的n-gram较小，对长尾实体的罕见n-gram较大，且可以从1比特编码中计算得出。因此，GateRQ将每一行存储为1比特残差编码的阶梯结构，将第一级保留在GPU内存中以计算门控，仅对门控排名最高的查找读取更深层级。无需重新训练，GateRQ在DeepSeek-V4.1-Flash上将GPU表内存削减了7.8倍，相比FP8表使vLLM吞吐量提升11%-31%，同时……

    arXiv:2610.35985v1 Announce Type: cross  Abstract: Conditional memory layers, such as DeepSeek's Engram, attach large hashed n-gram tables to a language model, and removing them from DeepSeek-V4.1-Flash costs 36.8 PopQA points. Low-bit quantization loses long-tail facts first, and the standard remedy, giving more bits to frequently accessed rows, backfires: refining lookups by access frequency is worse than refining them at random. We trace this to the gate that scales each retrieved row. A lookup's quantization error is weighted by its gate, which is small for frequent n-grams, large for the rare n-grams of long-tail entities, and computable from a 1-bit code. GateRQ therefore stores each row as a ladder of 1-bit residual codes, keeps level 1 in GPU memory to compute the gate, and reads deeper levels only for the lookups the gate ranks highest. Without retraining, GateRQ cuts GPU table memory by 7.8x on DeepSeek-V4.1-Flash, which raises vLLM throughput by 11%-31% over FP8 tables, whil
    
[^255]: 基于决策模型的实体匹配

    Entity Matching using Decision Models

    [https://arxiv.org/abs/2610.34626](https://arxiv.org/abs/2610.34626)

    本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。

    

    实体匹配是判断两条记录是否描述同一个现实世界实体的任务。当前的实体匹配方法依赖于需要针对特定任务进行微调的预训练语言模型（PLM），或者依赖于无需任务特定微调、但在匹配大规模记录集时速度慢且成本高的生成式大语言模型（LLM）。决策模型是一类新型的基于大语言模型的模型，它们为每个预定义的答案选项返回概率，而不是生成文本。本文研究了决策模型作为生成式大语言模型和微调预训练语言模型在实体匹配任务中的替代方案。我们在八个基准数据集上以零样本设置评估了六个决策模型，并将其性能与生成式大语言模型 GPT-6 Luna 以及基于预训练语言模型的匹配器 Ditto 进行了比较。我们的实验表明，在无需任务特定训练数据的情况下，四个最强的决策模型取得了接近 GPT-6 Luna 的平均 F1 分数。

    arXiv:2610.34626v1 Announce Type: cross  Abstract: Entity matching is the task of deciding whether two records describe the same real-world entity. Current entity matching methods rely on pre-trained language models (PLMs) which require task specific fine-tuning or generative large language models (LLMs) which do not require task-specific fine-tuning but are slow and expensive at matching large record sets. Decision models are a new class of LLM-based models that return a probability for each predefined answer option rather than generate text. This paper investigates decision models as an alternative to generative LLMs and fine-tuned PLMs for entity matching. We evaluate six decision models in a zero-shot setting across eight benchmarks and compare their performance with that of the generative LLM GPT-6 Luna and the PLM-based matcher Ditto. Our experiments show that, without task-specific training data, the four strongest decision models achieve mean F1 scores close to that of GPT-6 Lu
    
[^256]: 基准测试出人意料地对上下文压缩友好

    Benchmarks Are Surprisingly Compaction-Friendly

    [https://arxiv.org/abs/2610.31778](https://arxiv.org/abs/2610.31778)

    该论文首次在固定的智能体框架内实现五种上下文压缩策略，并在三个长时程基准上系统评估三个前沿模型，发现基准任务对有损压缩出人意料地友好——对百万级上下文模型而言，压缩阈值几乎不影响任务性能。

    

    arXiv:2610.31778v1 公告类型： cross 摘要：长时程智能体任务目前正在将大语言模型（LLM）智能体推向超出其上下文窗口的范围。为了继续执行任务，智能体框架必须释放上下文窗口空间，这一过程被称为上下文压缩。压缩在设计上通常是有损的：智能体可能会悄悄丢失任务约束、指令，或者其早期决策背后的推理依据。尽管压缩已是编程智能体框架中的标准组件，但此前尚未有研究针对前沿模型在真实的长时程智能体任务上对压缩进行系统性研究。以往的工作仅在玩具型或合成的轨迹上评估较小的开源权重模型。我们固定了智能体框架，在其中实现了五种压缩策略，并在三个长时程基准上评估了三个前沿模型。对于两个具备100万上下文的模型，压缩阈值对任务性能没有可测量的影响，而对于上下文窗口最小的GPT-5.3-Codex……

    arXiv:2610.31778v1 Announce Type: cross  Abstract: Long horizon agentic tasks are now pushing LLM agents past their context window. To continue the task, the agent scaffold has to free up the context window, a process known as context compaction. Compaction is generally lossy by design: agents can silently lose task constraints, instructions, or their reasoning behind earlier decisions. Although compaction is a standard component of coding agent scaffolds, it has not been studied systematically with frontier models on realistic, long horizon agentic tasks. Previous work evaluate smaller open weight models on toy or synthetic trajectories. We fix the agent scaffold (OpenCode), implement five compaction strategies inside it, and evaluate three frontier models on three long horizon benchmarks (PaperBench, ProgramBench, PostTrainBench). For the two 1M context models, the compaction threshold does not measurably affect the task performance, whereas GPT-5.3-Codex, the model with the smallest
    
[^257]: 跨预训练模型的LoRA叠加

    LoRA Superposition across Pretrained Models

    [https://arxiv.org/abs/2610.30691](https://arxiv.org/abs/2610.30691)

    该论文提出“LoRA叠加”概念，证明在不同预训练骨干网络上针对同一任务独立训练的LoRA可以直接相加并无修改地应用于所有骨干网络，其效果接近单独训练的LoRA，平均准确率提升高达30.21个百分点。

    

    低秩适应（LoRA）通常与其所学习的预训练骨干网络绑定。我们探究在不同预训练骨干网络上针对同一任务独立训练的LoRA是否可以求和并无修改地应用于所有参与的骨干网络。我们将这种构造称为LoRA叠加。在七个ViT-B/16 CLIP骨干网络和八个图像分类任务上，精确求和在全部56个骨干-任务组合上都带来了提升，相比未适应的骨干网络平均准确率提高了30.21个百分点；其平均准确率仅比单独训练的独立LoRA低0.28个百分点。相比之下，留一模型（LOMO）求和相比未适应的骨干网络仅带来平均0.06个百分点的增益。特定骨干的独立LoRA在参数空间中近乎正交，而两两跨骨干的响应则强烈集中在主对角线上。这些结果支持一种（原文在此处截断）……

    arXiv:2610.30691v2 Announce Type: replace-cross  Abstract: Low-rank adaptation (LoRA) is usually tied to the pretrained backbone on which it is learned. We ask whether LoRAs trained independently for the same task on different pretrained backbones can be summed and applied unchanged to all contributing backbones. We call this construction LoRA superposition. Across seven ViT-B/16 CLIP backbones and eight image-classification tasks, the exact sum improves all 56 backbone-task pairs and achieves a mean accuracy gain of 30.21 percentage points over the unadapted backbone. The exact sum achieves a mean accuracy only 0.28 percentage points below that of individually trained solo LoRAs. In contrast, a leave-one-model-out (LOMO) sum produces a mean gain of only 0.06 percentage points over the unadapted backbones. Backbone-specific solo LoRAs are nearly orthogonal in parameter space, while pairwise cross-backbone responses are strongly concentrated on the diagonal. These results support a comp
    
[^258]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^259]: 计算而非记忆：LLM智能体中的思维链与状态跟踪

    Computation, Not Memory: Chain-of-Thought and State Tracking in LLM Agents

    [https://arxiv.org/abs/2610.25760](https://arxiv.org/abs/2610.25760)

    该论文通过区分“行动前即时生成推理”与“在提示中保留历史推理笔记”两种角色，发现真正减少状态违规行动的主要来源是行动时刻的即时推理（计算），而非记忆式的历史推理保留。

    

    语言模型智能体通常在行动之前进行推理，而监控这种推理的价值取决于它与智能体行动之间的关系。我们探讨当正确的行动依赖于不断变化的状态时，智能体的行动如何依赖于其推理。在一个小型文件系统任务中，后台事件会宣布哪个文件的锁被切换，但不会宣布其新值。因此，智能体必须从历史记录中重建当前的锁状态，才能安全地移动文件。基于 Qwen3-8B 和 Qwen3.5-9B 模型，我们区分了推理的两种角色：在每次行动之前生成推理，以及在提示中保留早期推理。减少锁违规命令的收益大部分来自行动当下写出的推理。当智能体在每一步都进行推理时，保留早期推理几乎没有额外帮助；但对于不逐步推理的智能体，保留早期推理是有帮助的，并且在更长的回合中也是如此。保留的笔记有其局限：当后续事件使某条笔记变得不正确时……

    arXiv:2610.25760v2 Announce Type: replace-cross  Abstract: Language-model agents often reason before they act, and the value of monitoring that reasoning depends on how it relates to their actions. We ask how an agent's actions depend on its reasoning when the correct action depends on a changing state. In a small file-system task, background events announce which file's lock has been toggled but not its new value. The agent therefore has to reconstruct the current locks from its history before it can move files safely. With Qwen3-8B and Qwen3.5-9B, we separate two roles of reasoning: generating it before each action, and retaining earlier reasoning in the prompt. Most of the reduction in lock-violating commands comes from reasoning written at the moment of action. Retaining earlier reasoning adds little when the agent also reasons at each step, but it helps an agent that does not reason, and it helps in longer episodes. Retained notes have a limit: when a later event makes a note wron
    
[^260]: CHSM8：一种遵循自然语言风格提示的国际象棋Transformer

    CHSM8: A Chess Transformer that Follows Natural-Language Style Prompts

    [https://arxiv.org/abs/2610.25356](https://arxiv.org/abs/2610.25356)

    CHSM8是一个2.1亿参数的国际象棋Transformer，无需搜索即可根据自然语言风格提示下棋，达到1837 Elo并击败了包括参数量为其13倍模型在内的所有同类公开模型，并通过合成数据微调使其能够理解文本风格指令。

    

    我们提出了CHSM8，一个2.1亿参数的国际象棋Transformer，它可以根据对下棋风格的自由文本描述来走子，例如“采用西西里防御”、“稳健、安全、无风险的棋风”或“你是巅峰时期的马格努斯·卡尔森”。该模型在10.7亿局经过筛选的Lichess对局上进行预训练，无需搜索，采用因式分解的走子表示和合法走子掩码。在面对强度受限的Stockfish时，它达到了1837 Elo的等级分；在1000局的比赛中，它击败了我们测试的所有其他同样在没有偏好调优、奖励或搜索的情况下训练的公开模型，其中包括一个参数量是其13倍的模型。随后，我们在一个合成数据集上对其进行微调以使其遵循文本指令：Gemma-4-31B为300万个对局方描述其下棋风格，一个从240万条与棋盘标签无矛盾的描述中蒸馏而来的1B Gemma-3模型对其余描述进行标注，并且所筛选的3000万个对局方中的每一个都获得五条风格描述。监督微调……

    arXiv:2610.25356v2 Announce Type: replace-cross  Abstract: We present CHSM8, a 210M-parameter chess transformer that plays moves conditioned on a free-text description of a playing style, e.g. "go for the Sicilian Defense", "solid, safe, risk-free chess" or "You are Magnus Carlsen in his prime". The model is pretrained without search on 1.07B filtered Lichess games, using a factorized move representation and legal-move masking. It reaches 1837 Elo against a strength-limited Stockfish and, in 1,000-game matches, beat every other public model we tested that was likewise trained without preference tuning, rewards or search, including one with 13 times as many parameters. We then fine-tune it to follow text on a synthetic dataset: Gemma-4-31B describes the playing style of 3M game sides, a 1B Gemma-3 model distilled from the 2.4M of those descriptions that contradict nothing on the board labels the rest, and every one of 30M selected game sides gets five descriptions. Supervised fine-tunin
    
[^261]: FlashReFocus：秒级交互式图像重新对焦

    FlashReFocus: Interactive Image Refocusing in Seconds

    [https://arxiv.org/abs/2610.23297](https://arxiv.org/abs/2610.23297)

    FlashReFocus 对每张照片仅运行一次扩散模型来恢复全清晰图像，再用基于显式散焦图条件的轻量级网络渲染每次编辑，从而实现秒级的交互式图像重新对焦。

    

    单图像重新对焦是在拍摄后改变照片的焦平面和景深。这本质上是交互式的：用户会在同一张图像上尝试多个焦点和模糊强度，因此每次编辑都应该快速且只改变模糊。然而，近期的生成式方法为每次编辑都重新运行大型扩散模型。我们观察到这两个子问题在性质上有所不同：恢复全清晰图像是一个病态问题，需要生成先验；而渲染模糊在很大程度上由深度和焦平面决定。因此，我们提出 FlashReFocus，它对每张照片只运行一次扩散过程，然后用轻量级网络渲染每次编辑。为解决病态恢复问题，带有解码器精炼器的单步扩散去模糊器在平衡保真度和感知质量的同时高效恢复精细细节。为实现精确控制，渲染器以显式散焦图为条件，并使用散焦偏置注意力，这使得……

    arXiv:2610.23297v1 Announce Type: cross  Abstract: Single-image refocusing changes the focal plane and depth of field of a photograph after capture. It is inherently interactive: users try many focal points and blur strengths on one image, so each edit should be fast and change only the blur. Recent generative methods, however, rerun large diffusion models for every edit. We observe that the two sub-problems differ in kind: recovering the all-in-focus image is ill-posed and needs a generative prior, whereas rendering blur is largely determined by depth and the focal plane. We therefore propose FlashReFocus, which runs diffusion once per photograph and renders every edit with a lightweight network. To address the ill-posed restoration, a one-step diffusion deblurrer with a decoder refiner recovers fine detail efficiently while balancing fidelity and perceptual quality. For precise control, the renderer is conditioned on an explicit defocus map and uses defocus-bias attention, which lets
    
[^262]: 改变数字，改变答案：评估基于大语言模型的文本评分中的评分量表敏感性

    Change the Number, Change the Answer: Assessing Rating Scale Sensitivity in LLM-Based Text Scoring

    [https://arxiv.org/abs/2610.22187](https://arxiv.org/abs/2610.22187)

    本文定义并系统研究了“量表敏感性”这一新概念，通过大规模交叉实验发现，评分量表的设计元素（格式、范围、符号）平均会使大语言模型的文本评分产生约0.10的归一化分数偏移，揭示了不同研究间分数可比性的重要隐患。

    

    大语言模型在计算社会科学中越来越多地被用于为社会和心理构念分配标量分数。然而，不同研究使用的评分量表各异，其分数在不同量表设计下是否仍具可比性尚未得到充分研究。为填补这一空白，我们定义并系统研究了“量表敏感性”，即由评分量表设计引起的LLM生成分数的变化。我们区分了数值敏感性（归一化分数的变化）与语义敏感性（分数解释的变化）。为评估这两种敏感性，我们的实验设计将2种格式（连续型与序数型）、4种范围（3、5、11和21分）和2种符号（有符号与无符号）进行交叉，涵盖3项任务（政治倾向、情感和毒性）、2种提示策略（零样本和基于量规的提示）以及5个大语言模型。我们发现，改变其中一个设计元素平均会使归一化分数偏移约0.10，并改变类别的……

    arXiv:2610.22187v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used in computational social science to assign scalar scores to social and psychological constructs. However, studies use different rating scales, and whether their scores remain comparable across scale designs is understudied. To address this gap, we define and systematically investigate scale sensitivity, the variation in LLM-generated scores caused by rating scale design. We distinguish numerical sensitivity, changes in normalized scores, from semantic sensitivity, changes in score interpretation. To assess both, our experimental design crosses 2 formats (continuous vs. ordinal), 4 ranges (3, 5, 11, and 21 points), and 2 signs (signed vs. unsigned) across 3 tasks (political leaning, sentiment, and toxicity), 2 prompting strategies (zero-shot and rubric-based), and 5 LLMs. We find that changing one design element shifts normalized scores by about 0.10 on average and changes the category o
    
[^263]: SkillForgeBench：基于智能体技能规范的智能体可靠性评估

    SkillForgeBench: Evaluating Agent Reliability Grounded in Agent Skill Specifications

    [https://arxiv.org/abs/2610.14918](https://arxiv.org/abs/2610.14918)

    SkillForgeBench以智能体技能作为统一执行蓝图，通过资源准备、技能驱动任务合成等三个协同设计阶段，构建了兼顾领域多样性与程序复杂性的智能体可靠性评估基准，化解了专家策划基准与众包基准之间的根本性权衡。

    

    评估智能体严格满足多样化且程序复杂需求的能力，对于确定其实际可靠性至关重要。开展此类评估需要能够兼顾广泛领域覆盖与深层程序复杂性的智能体任务。现有范式面临根本性的权衡：专家策划的基准虽然实现了程序深度，却受制于领域多样性的瓶颈；而众包基准虽然覆盖了多样化场景，却缺乏执行的严谨性。为化解这一矛盾，我们提出了SkillForgeBench，它将智能体技能作为统一的执行蓝图，贯穿三个协同设计的阶段：（1）资源准备阶段利用领域均衡的技能来指导工件检索，建立广泛的工作流覆盖；（2）技能驱动的任务合成阶段将技能作为程序锚点，用以划分工件角色、合成自然的任务指令，并打包沙盒化场景以实现深层程序复杂性……

    arXiv:2610.14918v1 Announce Type: new  Abstract: Evaluating an agent's ability to strictly satisfy diverse and procedurally complex requirements is essential to determining its practical reliability. Conducting such an evaluation demands agentic tasks that reconcile broad domain coverage with deep procedural complexity. Existing paradigms face a fundamental trade-off: expert-curated benchmarks achieve procedural depth but suffer from domain diversity bottlenecks, whereas crowdsourced benchmarks span diverse scenarios but lack execution rigor. To resolve this tension, we present SkillForgeBench, which leverages agent skills as unifying execution blueprints across three co-designed stages: (1) Resource Preparation uses domain-balanced skills to guide artifact retrieval, establishing broad workflow coverage; (2) Skill-driven Task Synthesis employs skills as procedural anchors to partition artifact roles, synthesize naturalistic task instructions, and package sandboxed scenarios for deep p
    
[^264]: 当多模态大语言模型读不懂人类能读的文字：诊断并学习阅读“眯眼文字”

    When MLLMs miss what humans read: diagnosing and learning to read squint text

    [https://arxiv.org/abs/2610.13465](https://arxiv.org/abs/2610.13465)

    该论文发现多模态大模型存在“眯眼文字”识别缺陷，通过构建 SquintBench 与 SquintCtrl 两个基准诊断出空间选择是关键瓶颈，并据此提出 Squint OPSD 方法教会模型阅读此类隐藏文字。

    

    多模态大语言模型（MLLM）可能无法读取嵌入在图像粗略结构中的文字（即人类通过眯眼就能辨认的文字），并错误地报告图像中不存在任何文字。我们将这类文字命名为“眯眼文字”，并将其作为一种识别缺陷加以研究，研究动机来自生产级内容审核流水线中的实际案例。我们构建了 SquintBench，包含 1076 张源自生产环境的审核图像，以及 SquintCtrl——一个用于机制诊断的受控合成测试集。冻结权重分析表明，隐藏的文字会留下粗略的信号，而字形身份的表征较弱；当提供正确的空间区域时，字形身份的可获取性会显著提高，这表明空间选择是主要瓶颈。在固定视觉 token 预算下，抑制高频内容也能提升阅读效果，同时视觉 token 的数量也可能产生影响。这些发现促使我们提出 Squint OPSD：一个冻结的教师模型接收缩小尺寸的视图和真实转录文本作为先验（原文摘要在此处截断）……

    arXiv:2610.13465v1 Announce Type: cross  Abstract: Multimodal LLMs can fail to read text embedded in an image's coarse structure (text a human recovers by squinting) and report that no text is present. We name this class Squint Text and study it as a recognition deficiency, motivated by cases from a production moderation pipeline. We introduce SquintBench, comprising 1076 production-derived moderation images, and SquintCtrl, a controlled synthetic suite for mechanism diagnosis. Frozen-weight analysis shows that hidden text leaves a coarse signal while glyph identity is weakly represented; identity becomes substantially more accessible when the correct spatial regions are supplied, implicating spatial selection as a major bottleneck. Suppressing high-frequency content also improves reading at a fixed visual-token budget, while visual token count may also contribute. These findings motivate Squint OPSD: a frozen teacher receives a downscaled view and the ground-truth transcription as pri
    
[^265]: 面向有效子群体可靠性的随机分组保形预测

    Stochastic Grouping Conformal Prediction for Effective Subgroup Reliability

    [https://arxiv.org/abs/2610.11957](https://arxiv.org/abs/2610.11957)

    该论文提出随机分组保形预测（SGCP），通过学习随机分组映射让每个样本从校准行为相似的样本中获取校准信息，在无需敏感子群体属性的情况下实现跨临床子群体的可靠不确定性量化，并缓解最差群体瓶颈问题。

    

    保形预测提供了一种无分布的覆盖率保证，这使其在临床应用中尤其具有吸引力。然而，标准保形预测仅在总体层面提供此类保证，其预测集在临床上重要的子群体之间可能表现出覆盖率差异。一种自然的补救方法是在预定义的群体内进行校准。然而，这可能需要获取敏感的子群体属性，并且容易出现最差群体瓶颈：保护最困难的子群体可能会使所有样本的预测集膨胀，从而增加决策者的认知负担。为此，我们提出了随机分组保形预测（SGCP），这是一个用于子群体可靠不确定性量化的保形框架。它学习一个随机分组映射，使每个样本能够从校准行为相似的其他样本中获取校准信息，从而产生一种局部分数法则，跨子群体提升可靠性。

    arXiv:2610.11957v1 Announce Type: cross  Abstract: Conformal prediction offers a distribution-free coverage guarantee, making it especially attractive for clinical applications. Standard conformal prediction, however, provides such guarantees only at the population level, and its prediction sets can exhibit coverage disparities across clinically important subgroups. A natural remedy is to calibrate within predefined groups. However, this can require access to sensitive subgroup attributes and is prone to a worst-group bottleneck: protecting the most difficult subgroup can inflate prediction sets for all, increasing cognitive burden on decision makers. To this end, we propose Stochastic Grouping Conformal Prediction (SGCP), a conformal framework for subgroup-reliable uncertainty quantification. It learns a stochastic grouping map that allows each sample to draw calibration information from others with similar calibration behavior, yielding a local score law that boosts reliability acros
    
[^266]: 对行动时间受限的智能体的AI安全考量

    AI Safety Considerations for Agents With Limited Time to Act

    [https://arxiv.org/abs/2610.10285](https://arxiv.org/abs/2610.10285)

    本研究证明在只能部分观察且必须在有限时间内行动的环境中，即使是完美的AI智能体也无法保证安全行为，因此AI安全或对齐的证明必须将环境与安全行动和智能体结合起来具体考虑。

    

    在关于AI对齐日益公开的讨论背景下，近期的研究试图提出能够安全运行的具体AI架构。然而，那些看似证明了对齐的论证大多忽略了智能体需要在其中行动的环境。我们讨论了在只能部分观察且需要在有限时间内采取行动的环境中，与智能体无关的安全保证的理论界限。我们引入了两个现实场景：一个具有无限状态空间，另一个具有信号混合。在这些场景中，我们证明了即使是完美的智能体也无法保证安全的行为。我们认为，对于任何AI安全或对齐的证明，都需要将环境和相关的安全行动与智能体一起具体地加以考虑。

    arXiv:2610.10285v1 Announce Type: cross  Abstract: In the wake of the increasingly public discussion about AI alignment, recent work has tried to propose specific AI architectures that behave safely. However, the proposed arguments that seemingly demonstrate proved alignment mostly neglect the environment the agent needs to act in. We discuss theoretical bounds for agent-agnostic safety guarantees in environments that can only be partially observed and within which an action is required within limited time. We introduce two realistic scenarios, one with an infinite state space and one with signal mixture. In these scenarios, we prove that even a perfect agent cannot guarantee safe behaviour. It will be argued that for any proof of AI safety or alignment, the environment and associated safe actions need to be specifically considered together with the agent.
    
[^267]: CredLeakBench：评估大语言模型智能体中的凭据泄露与恢复

    CredLeakBench: Evaluating Credential Leakage and Recovery in LLM Agents

    [https://arxiv.org/abs/2610.08871](https://arxiv.org/abs/2610.08871)

    提出了 CredLeakBench 基准，用于评估大语言模型智能体在面对网络钓鱼与身份验证场景时区分恶意请求与真实请求、防止凭据泄露并安全自动化用户工作流程的能力。

    

    语言模型智能体正被越来越多地部署用来自动化日常数字事务——从管理电子邮件和社交媒体到处理银行业务和账单支付——从而使用户可以脱离监督。然而，这种能力也使敏感信息暴露于网络钓鱼攻击之下。安全的执行要求智能体能够区分恶意请求与真实请求，而不是简单地拒绝执行。尽管这一问题具有重要的实际意义，但目前仍未得到充分研究，且尚不清楚当前的智能体或现有防御措施能否实现这一点。为了研究这一问题，我们首先提出了 CredLeakBench，这是一个综合性基准，旨在评估智能体在面临网络钓鱼和身份验证场景时自动化人类工作流程的有效性与安全性。该基准涵盖用户明确指示的身份验证以及自主收件箱监控两种场景，在后一种场景中，智能体并未被明确指示进行登录。该基准系统地变化欺骗性线索，并配对……（原文在此处截断）

    arXiv:2610.08871v2 Announce Type: replace-cross  Abstract: Language model agents are increasingly deployed to automate everyday digital chores from managing emails and social media to handling banking and bills allowing users to step away from supervision. However, this capability also exposes sensitive information to phishing. Safe execution requires distinguishing malicious requests from genuine ones without simply refusing to act. Despite its practical importance, this problem remains underexplored and it is unclear whether current agents or existing defenses can achieve it. To study this problem, we first propose CredLeak-Bench, a comprehensive benchmark designed to evaluate how effectively and securely agents automate human workflows when confronted with phishing and identity verification. The benchmark covers both user-directed authentication and autonomous inbox monitoring, where agents are not explicitly instructed to log in. It systematically varies deceptive cues and pairs ph
    
[^268]: 通过合成数据预训练的基础模型统一跨领域矩阵补全

    Unifying matrix completion across domains with a synthetically pretrained foundation model

    [https://arxiv.org/abs/2610.06751](https://arxiv.org/abs/2610.06751)

    MatrixFormer是一个完全基于合成低秩矩阵预训练的矩阵补全基础模型，能在单次前向传播中预测每个缺失条目的完整分布，并以零样本方式在因果推断、表格插补和推荐系统等多个领域取得有竞争力的性能。

    

    矩阵补全是从表格数据插补到因果推断等诸多问题的基础。虽然矩阵补全为这些领域提供了统一的框架，但每个领域的最佳方法仍然是专门化的，难以在领域间良好迁移。我们提出了MatrixFormer，这是一个预训练的矩阵原生transformer模型，能够在单次前向传播中为每个缺失条目预测完整的分布。MatrixFormer完全基于合成的低秩矩阵和潜因子矩阵进行训练，并涵盖多种缺失模式。在固定网络权重的零样本应用中，MatrixFormer在因果推断面板数据任务、表格数据插补和推荐系统矩阵补全上均取得了有竞争力的性能。这些结果使MatrixFormer成为矩阵补全领域的通用基础模型。

    arXiv:2610.06751v2 Announce Type: replace-cross  Abstract: Matrix completion underlies problems from tabular imputation to causal inference. While matrix completion provides a common setup for these domains, the best methods in each domain are still specialized and do not transfer well across domains. We introduce MatrixFormer, a pre-trained matrix-native transformer that predicts a full distribution for every missing entry in a single forward pass. MatrixFormer is trained entirely on synthetic low-rank and latent-factor matrices under diverse missingness patterns. Applied zero-shot with fixed network weights, MatrixFormer achieves competitive performance on causal inference panel-data tasks, tabular imputation, and recommendation systems matrix completion. These results position MatrixFormer as a general-purpose foundation model for matrix completion.
    
[^269]: ThunderSync：智能体强化学习的无损加速

    ThunderSync: Lossless Acceleration of Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.05935](https://arxiv.org/abs/2610.05935)

    ThunderSync 通过在所需输入就绪后立即流式计算梯度，在不引入策略陈旧性的前提下消除流水线空闲，实现对智能体强化学习的无损加速。

    

    语言模型正从单纯生成答案转向在交互环境中追求长期目标。对这些智能体进行后训练需要长且异构的轨迹，而同步系统会让学习引擎在 rollout 和验证完成之前处于闲置状态。为了消除这些流水线空泡，异步训练在更新之间重叠 rollout 与学习过程，但其代价是策略陈旧性（policy staleness）。我们提出 ThunderSync，它在所有必需输入确定后立即开始梯度计算，且不产生策略陈旧性。对于组相对策略优化（GRPO），ThunderSync 在每条轨迹的奖励一到达时即计算该轨迹的得分梯度，无需等待整个组完成。对于在线策略蒸馏（OPD），它在沙箱中运行工具调用的同时，为每个已完成的智能体轮次中由教师评分的动作计算梯度。我们证明了梯度流式计算能够产生与（原文截断处）……相同的 GRPO 结果。

    arXiv:2610.05935v2 Announce Type: replace-cross  Abstract: Language models are moving beyond generating answers to pursuing long-horizon goals in interactive environments. Post-training these agents requires long, heterogeneous trajectories, and synchronous systems leave learner engines idle until rollout and verification finish. To squeeze out these pipeline bubbles, asynchronous training overlaps rollout and learning across updates, but comes at the cost of policy staleness. We introduce ThunderSync, which starts gradient computation as soon as all required inputs are fixed, without policy staleness. For group relative policy optimization (GRPO), ThunderSync computes each trajectory's score gradient as soon as the reward for that trajectory arrives, without waiting for the group. For on-policy distillation (OPD), it computes gradients for each completed agentic turn's teacher-scored actions while tool calls run in the sandbox. We prove that gradient streaming produces the same GRPO a
    
[^270]: 冻结场景，更迭赢家：文本到3D评估中的配置脆弱性

    Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation

    [https://arxiv.org/abs/2610.00447](https://arxiv.org/abs/2610.00447)

    该论文通过对来自六个生成器的300个冻结场景系统性变化八种渲染与描述文本配置因素，揭示文本到3D评估结果对配置高度脆弱——17/19个对齐评估器的配置方差超过生成器间方差，且18/19个评估器在某些配置下会改变点估计赢家，表明当前评估协议存在严重的不稳定性。

    

    当所有生成的场景保持固定不变时，文本到3D（text-to-3D）排行榜还会发生变化吗？我们针对渲染图像评估对这一问题进行了审计，在此类评估中，相机设置与描述文本的措辞成为了测量协议的一部分。在来自六个生成器的300个冻结场景上，我们针对19个对齐评估器外加1个感知质量对照指标，变化了八种渲染与描述文本因素，随后测试了四种针对性的场景退化。结果发现：对于19个对齐评估器中的17个，峰值配置方差超过了生成器间的方差，其中11个评估器的提示词自助法（prompt-bootstrap）下界大于1。排名虽然比分数更稳定，但在某些配置下，19个评估器中有18个会改变其点估计赢家。成对协议边际包络显示哪些比较在所测试的设置中保持方向不变。选定的配对虽然具有方向相反的逐点区间，但在对完整搜索进行同时推断后，没有任何排名逆转能够存续。因此，所观察到的赢家变化是描述性的……

    arXiv:2610.00447v1 Announce Type: new  Abstract: Can a text-to-3D leaderboard change when every generated scene stays fixed? We audit this question for rendered-image evaluation, where camera settings and caption wording become part of the measurement protocol. Across 300 frozen scenes from six generators, we vary eight render and caption factors for 19 alignment evaluators plus one perceptual-quality control, then test four targeted scene degradations. Peak configuration variance exceeds between-generator variance for 17/19 alignment evaluators, with prompt-bootstrap lower bounds above 1 for 11/19. Rankings are more stable than scores, yet 18/19 evaluators change their point-estimate winner under some configuration. Pairwise protocol margin envelopes show which comparisons keep their direction across the tested settings. Selected pairs have opposite pointwise intervals, but no reversal survives simultaneous inference over the full search. Thus the observed winner changes are descripti
    
[^271]: MatLoom：紧凑程序空间中的分层文本到材质生成

    MatLoom: Layered Text-to-Material Generation in a Compact Program Space

    [https://arxiv.org/abs/2609.40322](https://arxiv.org/abs/2609.40322)

    MatLoom提出一种紧凑的面向图层的材质程序语言，配合预训练语言模型，通过解析器引导修复与预览评审实现免微调的文本到材质生成，其最佳配置在141个提示词的基准上超越了三个扩散基线。

    

    材质生成不仅应产出外观，还应产出构建该外观的规则。我们提出MatLoom，一种紧凑的、面向图层的材质程序语言，可与预训练语言模型结合进行文本到材质生成。每个程序由带Alpha遮罩的图层组合而成，图层间共享的空间表达式定义了覆盖范围与基于物理的渲染（PBR）通道，从而使图案、颜色与浮雕之间的依赖关系显式化。一个独立的解释器将程序求值为材质贴图，同时源代码保留命名字段和图层参数，便于后续编辑创作。无需任务特定的微调，我们的流程利用解析器引导的修复和基于预览的评审来修改材质设计，然后在保持每个候选的其余源代码不变的情况下搜索噪声种子。在一个包含141个提示词的精选基准上，使用六个骨干模型进行评估，我们表现最佳的配置在（原文在此处截断）上取得了高于三个扩散基线的平均分数。

    arXiv:2609.40322v1 Announce Type: cross  Abstract: Material generation should produce not only an appearance, but also the rules that construct it. We introduce MatLoom, a compact, layer-oriented language for text-to-material generation with pretrained language models. Each program composes alpha-masked layers whose shared spatial expressions define coverage and physically based rendering (PBR) channels, making dependencies between patterns, color, and relief explicit. A standalone interpreter evaluates the program into material maps, while the source retains named fields and layer parameters for subsequent authoring. Without task-specific fine-tuning, our pipeline uses parser-guided repair and preview-based critique to revise material designs, then searches noise seeds while keeping each candidate's remaining source fixed. On a curated benchmark of 141 prompts evaluated with six backbones, our best-performing configuration achieves higher mean scores than three diffusion baselines on 
    
[^272]: MemLife：面向长期第一人称视频记忆的整理与推理

    MemLife: Curating and Reasoning over Long-Term Egocentric Video Memories

    [https://arxiv.org/abs/2609.40195](https://arxiv.org/abs/2609.40195)

    MemLife是一种多模态记忆系统，通过构建基于实体的第一人称文本情节并用时间索引的智能体读取器进行检索，在无需训练和查询时视频访问的情况下，在四个长时程基准上比最强免训练基线提升4.6%–12.0%，解决了长期第一人称视频中记忆证据丢失和检索竞争的问题。

    

    长期第一人称视频使个性化AI助手能够对日常生活进行推理。然而，随着视频历史增长到跨越数月甚至数年的数百小时，为每次查询重新处理原始视频片段在计算上变得难以承受。记忆系统通过将视频压缩为文本表示提供了一种可扩展的替代方案，但在实际基准测试中常常失败：要么记忆未能保留关键证据，要么由于搜索空间不断扩大导致检索竞争，检索器无法定位相关条目。为应对这些挑战，我们提出了MemLife，一种多模态记忆系统，它构建基于实体的第一人称文本情节，并通过时间索引的智能体读取器进行检索。在无需训练且查询时无需访问视频的情况下，MemLife在四个长时程基准测试中比最强的免训练基线提升了4.6%至12.0%。为进一步提升记忆质量，我们提出了MemOpt，一种基于强化的方法（摘要在此处被截断）。

    arXiv:2609.40195v1 Announce Type: cross  Abstract: Long-term egocentric video enables personalized AI assistants to reason about daily life. However, as video histories grow to hundreds of hours spanning months or years, reprocessing raw clips for every query becomes computationally prohibitive. Memory systems offer a scalable alternative by compacting videos into text representations, but often fail on practical benchmarks: either the memory does not preserve key evidence, or the retriever fails to locate relevant entries due to retrieval competition in growing search spaces. To address these challenges, we introduce MemLife, a multimodal memory system that constructs entity-grounded, first-person text episodes and retrieves them via a time-indexed agentic reader. Without training or query-time video access, MemLife improves over the strongest training-free baseline by 4.6--12.0% across four long-horizon benchmarks. To further improve memory quality, we propose MemOpt, a reinforcement
    
[^273]: 解耦自蒸馏：测量与建模知识获取与保持

    Disentangling Self-Distillation: Measuring and Modeling Acquisition and Retention

    [https://arxiv.org/abs/2609.39494](https://arxiv.org/abs/2609.39494)

    该论文首次将自蒸馏方法解耦为rollout来源、教师耦合方式和KL方向三个独立维度，通过1,200次系统实验构建统一框架，解释了以往固定组合研究导致矛盾结论的根源。

    

    基于特权上下文的自蒸馏通过让模型在以参考回复为条件时，逐token地教导其无上下文的副本，从而从演示数据中适配语言模型。我们的分类学揭示了现有方法在三个相互纠缠的维度上存在差异：(i) rollout来源（学生或教师），(ii) 教师耦合方式（冻结，或以某种耦合率对学生模型进行指数移动平均），以及 (iii) KL散度方向（反向或前向）；然而这些维度通常以固定组合的方式被研究，并导致了相互矛盾的结论。我们形式化了一个统一框架，以涵盖所有自蒸馏方法与经典监督微调的对比：我们在Qwen2.5-7B和Ministral-3-3B模型上，针对普通任务和矛盾任务，训练这三个维度的所有组合，共计1,200次适配运行，以系统地研究上述各维度的影响。我们提出了一个基于相同目标的受控模型来解释

    arXiv:2609.39494v1 Announce Type: new  Abstract: Self-distillation with privileged context adapts a language model from demonstrations by letting the model, once conditioned on a reference response, teach its context-free copy token by token. Our taxonomy reveals existing methods differ along three entangled axes: (i) the rollout source (student or teacher), (ii) the teacher coupling (frozen, or an exponential moving average of the student at some coupling rate) and (iii) the KL direction (reverse or forward), yet these axes are usually studied in fixed combinations and have led to conflicting conclusions. We formalize a unifying framework to encompass all self-distillation methods vs classic supervised fine-tuning: we train every combination of the three axes, on Qwen2.5-7B and Ministral-3-3B across ordinary and contradictory tasks, totaling 1,200 adaptation runs, to systematically investigate the impact of the above axes. We propose a controlled model of the same objective to explain
    
[^274]: UniEvo-VL：一种用于多模态模型自我提升的在策略自蒸馏训练方案

    UniEvo-VL: An On-policy Self-Distillation Training Recipe for Multimodal Model Self-improvement

    [https://arxiv.org/abs/2609.38721](https://arxiv.org/abs/2609.38721)

    UniEvo-VL提出了一种在策略自蒸馏训练框架，让单个多模态模型在测试时计算中以自我批评作为特权信息，同时扮演教师和学生角色并最小化两者扩散分布的散度，从而实现无需外部教师模型的自我提升和图像生成能力的增强。

    

    现代多模态模型将生成与理解能力融合到单一统一系统中，这使其能够提供并从自身的反馈中学习。受这种统一能力的启发，我们提出了UniEvo-VL，一个自演化框架，使多模态模型能够在测试时计算过程中从这种建设性的自我纠错反馈中学习。我们不再依赖于单独的（通常是更大的）教师模型，而是将模型的自我批评作为特权信息，让同一个多模态模型在不同上下文中同时扮演教师和学生的角色。学生只看到原始问题，而教师则以特权批评为条件。随后，训练过程在学生自身的采样轨迹上最小化两者去噪扩散分布之间的逐状态散度。实验表明，UniEvo-VL提升了多模态模型的图像生成能力，同时保持其对增广内容的敏感性。

    arXiv:2609.38721v1 Announce Type: new  Abstract: Modern multimodal models bring generation and understanding into a single unified system, which enables them to provide and learn from their own feedback. Motivated by this unified capacity, we introduce UniEvo-VL, a self-evolving framework for multimodal models to learn from this constructive self-correction feedback during test-time compute. Instead of relying on a separate, often larger, teacher, we leverage their self-critiques as privileged information and ask a single multimodal model to act as both teacher and student with different contexts. The student only sees the vanilla question, while the teacher conditions on the privileged critique. Then training minimizes the per-state divergence between their denoising diffusion distributions over the student's own sampling trajectories. Experiments demonstrate that UniEvo-VL improves the image generation capabilities of multimodal models, while maintaining their sensitivity to addition
    
[^275]: 以智能体作为自身优化器的多任务自进化执行框架

    Self-Evolving Harness on Multiple Tasks with the Agent as Its Own Optimizer

    [https://arxiv.org/abs/2609.38372](https://arxiv.org/abs/2609.38372)

    该论文提出让同一个冻结模型兼任求解器与提议者的自进化执行框架，智能体在覆盖五个不同领域基准的多任务上直接编辑运行自身的框架代码，实现接近递归自我改进的效果。

    

    执行框架是围绕语言模型智能体的代码，负责组织提示词、调用工具、管理上下文并控制执行过程。随着模型能力的不断增强，近期研究开始让智能体改进自身的执行框架，这一研究方向被称为自进化执行框架。在大多数现有方法中，由一个运行在人工设计框架上的独立提议者来修改求解器的框架，并且需要为每个基准测试单独进化一个框架。然而，现实世界的任务来自多个领域，因此执行框架的进化和评估都应当覆盖多样化的任务。我们提出了一个接近递归自我改进的框架：同一个冻结模型在相同版本的框架上运行，首先作为求解器完成任务，随后作为提议者读取完整的运行记录，并直接编辑运行它自身的框架。每个进化批次从五个不同领域的基准测试中抽取任务。为衡量泛化能力，训练与留出测试……（摘要在此处截断）

    arXiv:2609.38372v1 Announce Type: new  Abstract: A harness is the code around a language-model agent that organizes prompts, calls tools, manages context, and controls execution. As models grow stronger, recent work has begun to let agents improve their own harnesses, a line of work known as self-evolving harnesses. In most existing methods, a separate proposer running on a human-designed harness modifies the solver's harness, and a separate harness is evolved for each benchmark. Real-world tasks come from many domains, so both the evolution and the evaluation of a harness should cover a diverse range of tasks. We propose a framework close to recursive self-improvement: the same frozen model, on the same version of the harness, first solves tasks as the solver and then, as the proposer, reads the complete run records and directly edits the harness that runs it. Each evolution batch draws tasks from five benchmarks in different domains. To measure generalization, training and held-out t
    
[^276]: 语言作为接口：基础模型对比学习连接转录组与电生理学

    Language as the Interface: Foundation-Model Contrastive Learning Links Transcriptomes and Electrophysiology

    [https://arxiv.org/abs/2609.37024](https://arxiv.org/abs/2609.37024)

    LangPatch框架通过将基因描述与电生理特征映射到同一语言空间并进行配对对比学习，成功将预训练基因基础模型GenePT与Patch-seq电生理表型对齐，在多个小鼠和人类皮层数据集上取得了最优的转录组到电生理预测性能。

    

    整合转录组学与电生理学数据对于构建神经科学的多模态基础模型至关重要。Patch-seq技术能够从同一神经元获得基因表达和内在电生理特性的配对测量，为训练跨模态模型奠定了基础。本文提出了LangPatch，一个基于基础模型的对比学习框架，利用配对的Patch-seq数据，通过基于语言的接口将预训练的GenePT表示与电生理表型进行对齐。基因描述和语言化表达的电生理特征由同一个冻结的文本编码器进行嵌入，上下文适配器和投影模块通过配对对比学习连接两个模态。在小鼠视觉皮层、小鼠运动皮层和人类皮层数据集中，LangPatch在所有被评估的基础模型和表示学习方法中实现了最高的转录组到电生理预测平均相关性。

    arXiv:2609.37024v1 Announce Type: new  Abstract: Integrating transcriptomic and electrophysiological data is essential for building multimodal foundation models for neuroscience. Patch-seq provides paired measurements of gene expression and intrinsic electrophysiology from the same neuron, establishing a basis for training cross-modal models. Here we introduce LangPatch, a foundation-model-based contrastive learning framework that uses paired Patch-seq data to align pretrained GenePT representations with electrophysiological phenotypes through a language-based interface. Gene descriptions and verbalized electrophysiological profiles are embedded by the same frozen text encoder. A context adapter and projection modules connect the modalities through paired contrastive learning. Across mouse visual, mouse motor, and human cortical cohorts, LangPatch achieves the highest mean transcriptome-to-electrophysiology prediction correlation among the evaluated foundation-model and representation-
    
[^277]: 为什么直接学习周期轨迹会失败

    Why Directly Learning Periodic Trajectories Can Fail

    [https://arxiv.org/abs/2609.32254](https://arxiv.org/abs/2609.32254)

    该论文揭示了一个共同机制——频率差异随时间累积使目标相位随参数快速变化——解释了为何直接学习周期轨迹会失败：未对齐轨迹即使在训练分布内也泛化不佳，而固定物理时间窗口训练的模型对训练范围外的周期轨迹误差很大，相位对齐则可显著改善分布内泛化。

    

    对周期解进行算子学习时，需要决定如何记录和表示仿真数据。一个自然的选择是积分足够长的时间以使瞬态衰减，并记录足够宽的窗口以包含所有轨迹至少一个完整周期。我们发现，这些保守的选择会使所得到的轨迹难以学习，即使底层周期轨道随系统参数规则变化也是如此。未对齐的轨迹即使在训练分布内也泛化不佳。相位对齐能显著改善分布内的泛化能力，但在固定物理时间窗口上训练的模型，对于周期处于训练范围之外的轨迹仍然存在较大误差。我们通过一个共同机制解释了这两种失败：频率差异随时间不断累积，导致目标相位随参数快速变化。无法跟踪这种变化的预测器会产生较大的总体损失。

    arXiv:2609.32254v2 Announce Type: replace  Abstract: Operator learning of periodic solutions requires deciding how simulation data should be recorded and represented. A natural choice is to integrate long enough for transients to decay and record a window wide enough to contain at least one full period of all trajectories. We find that these conservative choices can make the resulting trajectories difficult to learn, even when the underlying periodic orbits vary regularly with system parameters. Unaligned trajectories generalize poorly even within the training distribution. Phase alignment substantially improves in-distribution generalization, but models trained on a fixed physical-time window still have large errors on trajectories with periods outside the training range. We explain both failures through a common mechanism: frequency differences accumulate over time, so the target phase varies rapidly with the parameters. Predictors that cannot track this variation incur a population 
    
[^278]: Jev-Mobile：Jev作为移动GUI智能体的执行器

    Jev-Mobile: Jev as an Executor for Mobile GUI Agents

    [https://arxiv.org/abs/2609.30186](https://arxiv.org/abs/2609.30186)

    Jev-Mobile提出低频VLM规划与高频轻量级执行的新范式，由快速的类型化决策模型Jev在单个VLM决策下连续执行多个GUI动作，在AndroidWorld上达到79%任务成功率的同时大幅降低延迟与推理成本。

    

    视觉-语言模型（VLM）已成为自主移动GUI智能体的常见基础，但大多数现有系统在几乎每个交互步骤都依赖VLM进行规划和动作定位，导致显著的延迟和模型服务成本。我们提出了Jev-Mobile，它将这一范式转变为低频VLM规划与高频轻量级执行相结合：VLM负责指定局部目标，无障碍树定义结构化的可执行动作空间，而Jev作为一个快速的类型化决策模型，在该空间内反复选择动作。这种设计允许在单个VLM决策下执行多个GUI动作，在保持自适应交互的同时减少了昂贵的VLM推理。在完整的AndroidWorld任务套件上，Jev-Mobile实现了79%的任务成功率，相比之下SeeAct-V为78%，逐步VLM基线为84%。在成功的轨迹中，它将平均端到端执行时间降低了约3倍。

    arXiv:2609.30186v1 Announce Type: new  Abstract: Vision-language models (VLMs) have become a common foundation for autonomous mobile GUI agents, but most existing systems rely on the VLM for both planning and action grounding at nearly every interaction step, leading to substantial latency and model-serving cost. We introduce Jev-Mobile, which shifts this paradigm to low-frequency VLM planning and high-frequency lightweight execution: the VLM specifies local goals, the accessibility tree defines a structured executable action space, and Jev, a fast typed decision model, repeatedly selects actions within this space. This design allows multiple GUI actions to be executed under a single VLM decision, reducing expensive VLM inference while preserving adaptive interaction. On the full AndroidWorld task suite, Jev-Mobile achieves 79% task success, compared with 78% for SeeAct-V and 84% for a Step-wise VLM baseline. Among successful trajectories, it reduces mean end-to-end execution time by 3
    
[^279]: 先界定范围再持久化：防止智能体记忆中的跨任务族干扰

    Scope Before You Persist: Preventing Cross-Family Interference in Agent Memory

    [https://arxiv.org/abs/2609.29144](https://arxiv.org/abs/2609.29144)

    论文提出 Scoped-ORC 方法，通过将持久技能的检索范围严格限定在其原始任务族，防止跨任务族干扰，从而在不更新模型权重的情况下显著提升智能体的轨迹效用并消除有害的技能部署。

    

    持久化记忆使语言模型智能体能够在不更新模型权重的情况下改进提示词和技能。我们证明，将检索范围与认证范围相匹配，可以使这些编辑在重复出现的任务族中支持可靠的反复适应。我们在 ProcStream-RSI（一个12轮代码修复流）上研究冻结模型智能体，并采用正交回归控制（ORC），这是一种以执行结果为依据的持久技能编辑门控机制。在一项保持提案和门控决策不变的干预实验中，仅在原始任务族中检索每个已接受的技能，将平均隐藏轨迹效用从全局记忆下的0.713提升到0.816，并将有害部署从八个中的六个减少到零个。在27个配对的随机顺序流中，Scoped-ORC 相比 Global-ORC 将平均轨迹效用提高了0.063 [0.037, 0.094]，接受了63个而非12个更新，在19/27个流中产生了多个被接受的更新，且63个被接受的更新中有0个是有害的。

    arXiv:2609.29144v1 Announce Type: new  Abstract: Persistent memory lets language-model agents improve prompts and skills without updating model weights. We show that matching retrieval scope to certification scope enables these edits to support reliable repeated adaptation across recurring task families. We study frozen-model agents on ProcStream-RSI, a 12-round code-repair stream, using Orthogonal Regression Control (ORC), an execution-grounded gate for persistent skill edits. In an intervention that holds proposals and gate decisions fixed, retrieving each accepted skill only for its originating family raises mean hidden trajectory utility from 0.713 under global memory to 0.816 and changes harmful deployments from six of eight to none. In 27 paired randomized-order streams, Scoped-ORC improves mean trajectory utility by 0.063 [0.037, 0.094] over Global-ORC, accepts 63 rather than 12 updates, and produces multiple accepted updates in 19/27 streams, with 0/63 harmful acceptances. The 
    
[^280]: 连接学习型视觉感知与符号信念空间规划

    Bridging Learned Visual Perception and Symbolic Belief-Space Planning

    [https://arxiv.org/abs/2609.16884](https://arxiv.org/abs/2609.16884)

    提出第三种范式“VLM作为概率落地器”，将视觉-语言模型谓词落地的不确定性表示为符号状态上的概率分布，实现信念空间规划并在不确定性下生成鲁棒规划。

    

    在部分可观测的环境中，智能体必须在对世界状态缺乏完整认知的情况下采取行动，并依赖具有不确定性的状态估计流程。在这种不确定性下获得有据可依且可验证的符号规划仍然是一个关键挑战。最近的研究整合了视觉-语言模型（VLM）来连接感知与符号推理，主要遵循两种范式：第一种是“VLM作为规划器”（VLM-as-planner），将图像直接映射为动作序列；第二种是“VLM作为落地器”（VLM-as-grounder），将观测落地为符号谓词，作为现成规划器的初始状态。这两种方法都在规划过程中忽略了不确定性，从而损害了鲁棒性。我们引入了第三种范式——“VLM作为概率落地器”（VLM-as-probabilistic-grounder），这是一种新颖的方法，将VLM谓词落地的不确定性表示为符号状态上的概率分布。这使得系统能够在信念空间中进行规划，并在不确定性下生成鲁棒的规划。在仿真中的实验……

    arXiv:2609.16884v1 Announce Type: new  Abstract: In partially observable settings, agents must act without full knowledge of the world state and rely on uncertain state-estimation pipelines. Obtaining grounded and verifiable symbolic plans under such uncertainty remains a key challenge. Recent work has integrated Vision-Language Models (VLMs) to bridge perception and symbolic reasoning, following two main paradigms. The first, VLM-as-planner, maps images directly to action sequences, and the second, VLM-as-grounder, grounds observations into symbolic predicates used as the initial state by off-the-shelf planners. Both approaches ignore uncertainty in the planning process, compromising robustness. We introduce a third paradigm, VLM-as-probabilistic-grounder, a novel approach that captures the uncertainty of VLM predicate groundings as a probability distribution over symbolic states. This enables planning in belief space and producing robust plans under uncertainty. Experiments in simula
    
[^281]: 微调修复大语言模型中的模式坍缩与过度分散

    Fine-Tuning Fixes Mode Collapse and Over-Dispersion in LLMs

    [https://arxiv.org/abs/2609.16454](https://arxiv.org/abs/2609.16454)

    这篇论文证明大语言模型是否出现模式坍缩取决于具体模型和数据集，而充足的监督微调（SFT）数据能使LLM输出多样性收敛到目标分布，从而同时修复模式坍缩与过度分散问题。

    

    Doshi和Hauser（2024）、Bisbee等人（2024）以及Xie等人（2026）的近期研究引发了对大语言模型（LLM）输出多样性不足的担忧：与它们所要代表的人群的真实回答相比，模型输出更频繁地重复或彼此相似，这种现象被称为“模式坍缩”。在本工作中，我们展示了是否会出现模式坍缩（或其相反现象）取决于所使用的具体模型和数据集。此外，只要有足够的监督微调（SFT）数据，LLM的输出多样性会收敛到微调数据所采样的目标分布的多样性。为了量化这一比较，我们测量了从同一固定提示中独立采样的两个回答相同（碰撞）的概率，或它们在核函数下的期望相似度。我们推导了模型与目标碰撞概率之间期望差距的偏差-方差分解，表明SFT并非固有地……

    arXiv:2609.16454v1 Announce Type: new  Abstract: Recent work by Doshi and Hauser (2024), Bisbee et al. (2024), and Xie et al. (2026) raises concerns that outputs from large language models (LLMs) tend to be under-diverse: they repeat or resemble one another more often than responses from the population they are meant to represent, a phenomenon known as mode collapse. In this work, we show that whether mode-collapse, or its opposite, occurs depends on the specific model and dataset used. Further, with sufficient supervised fine-tuning (SFT) data, LLM output diversity converges toward that of the target distribution from which fine-tuning data are sampled. To quantify this comparison, we measure the probability that two responses sampled independently from the same fixed prompt coincide (collide), or their expected similarity under a kernel. We derive a bias-variance decomposition of the expected gap between the model's and target's collision probabilities, showing that SFT is not inhere
    
[^282]: OmniHarness：通过符号策略学习实现可泛化的视觉生成

    OmniHarness: Harnessing Generalizable Visual Generation via Symbolic Policy Learning

    [https://arxiv.org/abs/2609.16057](https://arxiv.org/abs/2609.16057)

    提出OmniHarness框架，通过将验证过的执行经验抽象为可复用的符号策略，并结合执行过程中的中间验证与自主探究式练习，实现了具有强泛化能力的视觉生成。

    

    统一的多模态大语言模型和多智能体系统推动了视觉生成领域的发展。然而，仍存在三个局限性：（1）现有方法往往蒸馏任务特定的经验，泛化能力有限；（2）反思通常推迟到任务完成后才进行；（3）知识通常仅在响应下游任务需求时才被获取。为了解决这些局限性，我们提出了OmniHarness，一个通过符号策略学习实现可泛化视觉生成的框架。OmniHarness将经过验证的执行过程抽象为面向视觉生成任务族的符号策略，在去除实例特定输入的同时，捕获共享的程序流程和适用条件。该框架可以为新任务实例化、调整和组合这些策略。执行过程中的中间验证指导策略细化与失败恢复。通过自主探究，OmniHarness能够自主生成并执行练习任务（原文摘要在此处截断）。

    arXiv:2609.16057v1 Announce Type: cross  Abstract: Unified multimodal large language models (MLLMs) and multi-agent systems have advanced visual generation. However, three limitations remain. (1) Existing methods often distill task-specific experience with limited generalizability. (2) Reflection is often deferred until task completion. (3) Knowledge is often acquired only in response to downstream task demands. To address these limitations, we introduce OmniHarness, a framework for generalizable visual generation via symbolic policy learning. OmniHarness abstracts verified executions into symbolic policies for visual generation task families, capturing shared procedures and applicability conditions while removing instance-specific inputs. The harness instantiates, adapts, and composes these policies for new tasks. Intermediate verification guides refinement and failure recovery during execution. Through self-directed inquiry, OmniHarness autonomously generates and executes practice ta
    
[^283]: Semantic-TVM：面向记忆增强与工具使用智能体的结构保持型可信虚拟内存

    Semantic-TVM: Structure-Preserving Trustworthy Virtual Memory for Memory-Augmented and Tool-Using Agents

    [https://arxiv.org/abs/2609.15011](https://arxiv.org/abs/2609.15011)

    Semantic-TVM 提出可信虚拟内存（TVM）闭环运行时，利用可信本地模型仅替换预测出的敏感片段而非整个字段，在隐私暴露保持低水平的同时恢复了大部分任务效用，解决了记忆增强与工具使用智能体在远程 LLM 处理过程中的隐私泄露问题。

    

    记忆增强与工具使用型智能体在远程大语言模型处理检索到的记忆、工具动作和中间观察结果时，会暴露确切的隐私数值。单向掩码虽然限制了直接暴露，但会移除可信执行所需的数值，并可能通过后续观察造成泄露。我们提出了可信虚拟内存（TVM），这是一种闭环运行时，将精确值状态保留在本地，同时向远程模型呈现受保护的视图。在这一单一运行时中，Rule-TVM 用本地可恢复的句柄替换整个受保护字段，而 Semantic-TVM 则仅替换由可信本地模型预测出的敏感片段，从而保留周围与任务相关的上下文。在两个提供商的 Memory-EHR 和 Memory-RAP 基准上，片段级投影恢复了整字段替换所损失的大部分电子健康记录（EHR）任务效用（在 DeepSeek 上任务成功率为 84.17% 对 52.33%），同时测得的暴露程度保持在低水平，且工作流程仍然可执行。

    arXiv:2609.15011v1 Announce Type: new  Abstract: Memory-augmented and tool-using agents expose exact private values when remote LLMs process retrieved memory, tool actions, and intermediate observations. One-way masking limits direct exposure but removes values needed for trusted execution and can leak them through later observations. We propose Trustworthy Virtual Memory (TVM), a closed-loop runtime that keeps exact-value state local while presenting a protected view to the remote model. Within this single runtime, Rule-TVM replaces whole protected fields with locally recoverable handles, and Semantic-TVM instead replaces only sensitive spans predicted by a trusted local model, preserving surrounding task-relevant context. On Memory-EHR and Memory-RAP across two providers, span-level projection recovers most of the EHR utility lost under whole-field replacement (Task Success 84.17% vs. 52.33% on DeepSeek) while measured exposure stays low and workflows remain executable.
    
[^284]: 无盒漏洞分析：仅基于描述的MCP服务器间接提示注入漏洞检测

    No-Box Vulnerability Analysis: Description-only Detection of Indirect Prompt Injection Vulnerabilities in MCP Servers

    [https://arxiv.org/abs/2609.10854](https://arxiv.org/abs/2609.10854)

    本文提出“无盒漏洞分析”新范式，仅凭功能描述元数据即可在不访问或不与目标系统交互的情况下，假设性检测MCP服务器中所有可能实现里的间接提示注入漏洞。

    

    传统的漏洞分析依赖于系统访问权限或动态交互，而这些对于审计闭源、远程托管的关键在位系统或商业受限软件的第三方分析师来说可能是不可获得的。因此，我们提出了一种新的范式——无盒漏洞分析，在这种范式下，既没有系统访问权限也没有运行时交互可用，仅有功能元数据可用。这些元数据定义了系统的预期行为，包括其输入、输出和副作用，同时约束了与该行为一致的实现空间。我们提出针对给定系统元数据的所有可能实现中存在的漏洞进行假设，而无需观察或与目标系统进行交互。当获得额外访问权限时，分析师可以随后验证这些假设。我们通过实现展示了无盒漏洞分析的可行性。

    arXiv:2609.10854v1 Announce Type: cross  Abstract: Conventional vulnerability analysis relies on either system access or dynamic interaction, all of which may be unavailable to third-party analysts auditing closed-source, remotely hosted, critical in situ systems, or commercially gated software. Therefore, we propose a new paradigm of no-box vulnerability analysis in which neither access nor runtime interaction is available, and only functionality metadata is available. Such metadata defines the intended behavior of the system, including its inputs, outputs, and side effects, while constraining the space of implementations consistent with that behavior. We propose hypothesizing about vulnerabilities that exist across all possible implementations of a given system metadata, without observing or interacting with the target system. An analyst can later validate these hypotheses when additional access is available. We showcase the feasibility of no-box vulnerability analysis through implem
    
[^285]: 面向具身空地协同目标搜索：基准、数据集与智能体方法

    Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method

    [https://arxiv.org/abs/2609.08402](https://arxiv.org/abs/2609.08402)

    该论文提出了首个空地协同目标搜索基准AGOS-Bench、包含7.7k条轨迹的AGOS-Dataset数据集，以及无需训练、工具增强的智能体方法AGOS-Agent，用以评估和解决视觉-语言模型在UAV-UGV协作搜索与验证目标车辆任务中的难题。

    

    城市环境中的空地目标搜索（AGOS）是一项具有挑战性的具身任务，它需要无人飞行器（UAV）与无人地面车辆（UGV）协同从多视角视觉参考中搜索并验证指定的目标车辆。为研究这一尚未充分探索的问题，我们提出了AGOS-Bench，这是首个专门评估通用视觉-语言模型（VLMs）能否通过UAV-UGV协作整合空中发现与地面验证的基准。我们进一步提供了AGOS-Dataset作为配套资源，该数据集由自动化流程构建的示例轨迹组成，包含7.7k个用于搜索多样化类别与属性目标的情节，涵盖三个难度级别。为解决AGOS任务，我们提出了AGOS-Agent，一种无需训练且工具增强的方法。该智能体方法通过审慎的搜索规划，使视觉-语言模型摆脱复杂而动态的协调负担。

    arXiv:2609.08402v1 Announce Type: cross  Abstract: Air-Ground Object Search (AGOS) in urban environments is a challenging embodied task, which requires an Unmanned Aerial Vehicle (UAV) and an Unmanned Ground Vehicle (UGV) to jointly search for and verify a specified target vehicle from multi-view visual references. To study this underexplored problem, we introduce AGOS-Bench, the first dedicated benchmark for evaluating whether general-purpose Vision-Language Models (VLMs) can integrate aerial discoveries and ground-level verification through UAV-UGV cooperation. We further provide AGOS-Dataset as the companion resource of exemplary trajectories constructed by an automatic pipeline. It consists of 7.7k episodes for searching objects of diverse categories and attributes, spanning three difficulty levels. To address the AGOS task, we propose AGOS-Agent, a training-free and tool-augmented approach. The agentic method relieves VLMs from complex and dynamic coordination via a deliberate sea
    
[^286]: FUSE：一个大语言模型危险能力评估框架

    FUSE: An Evaluating Framework for Dangerous Capabilities of LLMs

    [https://arxiv.org/abs/2609.02168](https://arxiv.org/abs/2609.02168)

    本文提出FUSE框架，在统一协议下通过知识、防御、危害三个正交流水线评估大语言模型的危险能力，并利用化学生物模块对四个系列的12个商用LLM进行横向比较，揭示了各模型间差异显著的危险能力画像。

    

    碎片化的安全评估削弱了对AI危险能力的治理。我们提出了一个模块化框架，在统一协议下通过三个正交流水线——知识（K）、防御（D）和危害（H）——对每个模型进行评估，并将结果汇总为标准化的危险能力画像 φ。可插拔模块提供场景种子、知识库、危害查询和评判标准，而核心评估引擎在各个领域保持不变；化学生物（CB）评估还辅以一个网络安全试点，以展示协议的可迁移性。我们通过化学生物（CB）模块实例化该框架，评估了来自四个系列的12个商用大语言模型。我们的首要贡献是对各模型及模型系列之间危险能力的横向比较：三个维度揭示了截然不同的能力画像——知识水平相当的模型在抗拒绝能力上存在差异，而防御能力强的模型并不（具备相应特征）。

    arXiv:2609.02168v1 Announce Type: new  Abstract: Fragmented safety evaluation undermines the governance of dangerous AI capabilities. We present a modular framework that evaluates each model through three orthogonal pipelines---Knowledge ($K$), Defense ($D$), and Harm ($H$)---under a unified protocol, aggregating results into a standardized dangerous-capability profile $\phi$. Pluggable modules supply scenario seeds, knowledge banks, hazard queries, and judge rubrics, while the core evaluation engine remains unchanged across domains; the CB evaluation is complemented by a cyber pilot demonstrating protocol transfer.   Instantiating the framework with a chemical-biological (CB) module, we evaluate 12 commercial LLMs from four families. Our first contribution is a horizontal comparison of dangerous capability across models and model families: the three dimensions expose sharply divergent profiles---models with comparable knowledge differ in refusal resilience, and strong defenders do not
    
[^287]: 智能体程序：计算材料科学中一种新兴形式的科学软件

    Agentic programs: an emerging form of scientific software in computational materials science

    [https://arxiv.org/abs/2609.00795](https://arxiv.org/abs/2609.00795)

    本文提出“智能体程序”这一新兴科学软件范式，将确定性算法与有界的LLM科学判断、任务验证和阶段性成熟相结合，并以从实验测量的无序晶体结构构建原子模型的DeMARS系统为例加以展示。

    

    计算材料科学传统上将算法任务委托给计算机，而将科学判断留给人类。我们提出，近期基于大语言模型（LLM）的智能体框架催生了一种新兴形式的科学软件——智能体程序，它将确定性算法与有界的LLM判断、任务特定的验证、阶段性成熟以及生产中的完全委托相结合。我们以DeMARS为例说明这一概念，DeMARS是一个用于从实验测量的无序晶体结构构建原子级模型的智能体程序。

    arXiv:2609.00795v1 Announce Type: cross  Abstract: Computational materials science has traditionally delegated algorithmic tasks to computers while leaving scientific judgments to humans. We argue that recent LLM-based agent harnesses enable an emerging form of scientific software, agentic programs, that combine deterministic algorithms with bounded LLM-based judgment, task-specific verification, episodic maturation, and complete delegation in production. We illustrate this concept with DeMARS, an agentic program for constructing atomistic models from experimentally measured disordered crystal structures.
    
[^288]: 愚人金：针对开放权重模型安全移除攻击的防御性欺骗

    Fool's Gold: Defensive Deception Against Safety-Removal Attacks on Open-Weight Models

    [https://arxiv.org/abs/2608.17202](https://arxiv.org/abs/2608.17202)

    该论文提出“愚人金”防御方法，通过训练诱饵响应在安全移除攻击后产生伪造信息，有效阻止开放权重模型被恶意利用。

    

    arXiv:2608.17202v1 公告类型：新 摘要：开放权重语言模型的安全对齐可以轻易被移除：abliteration技术能在几分钟内从权重中投射出拒绝中介方向，而据我们所知，没有任何发布时的防御措施能持久阻止这一点。无法阻止的，可以欺骗。我们的防御方法，即诱饵强化（“愚人金”），放弃拒绝条带并毒化其收益：一旦拒绝被移除，大多数对危险操作请求的回答都是自信、流畅的诱饵，其关键要素被伪造。诱饵在攻击的可微模拟中训练，仅在攻击状态下表达；一个拒绝固定和良性约束将清洁状态行为保持为原始状态。我们在五个家族的七个模型上（9B-122B，密集和混合专家）进行了实例化。在通过我们预注册效能门槛的六个模型上，攻击状态下对保留提示的响应中0.51-0.90为诱饵，+0.27-0.84归因于防御；所有s

    arXiv:2608.17202v1 Announce Type: new  Abstract: Safety alignment in open-weight language models is trivially removable: abliteration projects a refusal-mediating direction out of the weights in minutes, and no release-time defense we are aware of prevents it durably. What cannot be prevented can be deceived. Our defense, decoy hardening ("Fool's Gold"), concedes the refusal strip and poisons its payoff: once refusal is stripped, most answers to hazardous operational requests are confident, fluent decoys whose critical elements are falsified. Decoys are trained inside a differentiable simulation of the attack, expressing only in the attacked state; a refusal pin and benign leash hold clean-state behavior to the original. We instantiate it on seven models from five families (9B-122B, dense and mixture-of-experts). On the six models passing our pre-registered efficacy gate, 0.51-0.90 of attacked-state responses to held-out prompts are decoys, +0.27-0.84 attributable to the defense; all s
    
[^289]: SysEvolve：一个AI原生的、安全的、自主对抗性攻防协同进化系统

    SysEvolve: An AI-native, safe, autonomous adversarial attack-defense co-evolutionary system

    [https://arxiv.org/abs/2608.15012](https://arxiv.org/abs/2608.15012)

    本文提出SysEvolve系统，通过攻击与防御AI代理的自主协同进化循环，解决网络安全中攻击加速与防御滞后之间的不对称性，实现安全、自驱动的对抗性攻防进化。

    

    摘要：arXiv:2608.15012v1 公告类型：交叉 摘要：大型语言模型（LLM）的快速发展造成了网络安全领域日益增长的不对称性，其中攻击正加速走向自主执行，而防御仍主要依赖人工密集型工作。尽管在网络靶场、AI驱动的攻击和AI驱动的防御方面已有大量前期工作，但这种不对称性依然存在。我们将其追溯到更深层的根本原因，即进化本身在双方三个层面上都已停滞。为克服这一点，我们提出协同进化作为整合性见解，即攻击和防御AI代理通过对抗性交锋自主且安全地推动彼此的进化。基于此见解，我们提出了SysEvolve，它包含三个协同设计的组件：SysField、SysSpear和SysArmor。SysField构建逼真的多主机靶场。SysSpear生成高效、安全的攻击方案。SysArmor执行实时、可解释的防御。它们共同形成一个自驱动的对抗性循环。

    arXiv:2608.15012v1 Announce Type: cross  Abstract: The rapid advancement of large language models (LLMs) has created a growing asymmetry in cybersecurity, where attack accelerates toward autonomous execution while defense remains predominantly human-intensive. Despite substantial prior work across cyber ranges, AI-driven attack, and AI-driven defense, this asymmetry persists. We trace it to a deeper root cause, that evolution itself has stalled on both sides at three layers. To overcome this, we propose co-evolution as the integrating insight, where attack and defense AI agents autonomously and safely drive each other's evolution through adversarial confrontation. Based on this insight, we present \sysevolve, comprising three co-designed components, \sysfield, \sysspear, and \sysarmor. \sysfield constructs realistic multi-host ranges. \sysspear generates efficient, safe attack schemes. \sysarmor performs real-time, interpretable defense. Together they form a self-driven adversarial loo
    
[^290]: CLAUDE.md 为何不断膨胀？智能体编程中的灾难性记忆

    Why Does CLAUDE.md Keep Growing? Catastrophic Remembering in Agentic Coding

    [https://arxiv.org/abs/2608.11095](https://arxiv.org/abs/2608.11095)

    该论文揭示并量化了智能体编程提示文件（如 CLAUDE.md）因“追加廉价、删除昂贵”的不完美记忆机制而近乎单调地无限制膨胀、最终导致智能体失效的“灾难性记忆”现象，并证明提示注释可有效遏制这一增长。

    

    智能体提示文件（如 CLAUDE.md）在真实代码仓库中无限制地增长，只有在历史终结或有人整体重写文件时才会停止。我们将这一现象归因于不完美的记忆：追加一条指令总是廉价的，但一旦指令背后的潜在推理丢失，在不冒回归风险的前提下删除它，需要付出 O(2^|D|) 的代价（其中 |D| 为提示中的指令数量）。连续的编辑会破坏这些潜在推理，新增速度超过删除速度，提示随之不断增长，直至智能体失去功能——我们称之为“灾难性记忆”。首先，我们在 1,867 个代码仓库中对 247,694 条指令的生命周期进行了刻画：智能体提示几乎单调增长，在整个生命周期中增长超过三倍（+226%），平均每次提交净增 4.7 条指令。删除风险随指令年龄呈几何级数衰减，这与不完美记忆假说相吻合，而与我们测试的其他竞争性解释均不符。其次，我们证明提示注释可以遏制这种增长：在可验证的世界中……（原文在此处截断）

    arXiv:2608.11095v2 Announce Type: replace  Abstract: Agentic prompts like CLAUDE.md grow without bound in real repositories, stopping only when history ends or someone rewrites the file wholesale. We trace this to imperfect recall: appending an instruction is always cheap, but once an instruction's latent reasoning is gone, deleting it without risking a regression costs O(2^|D|) in a prompt of |D| instructions. Successive edits destroy that latent reasoning, additions outrun deletions and prompts grow until agents become non-functional -- catastrophic remembering. First, we characterize this phenomenon across 247,694 instruction lifetimes in 1,867 repositories: agentic prompts grow nearly monotonically, more than tripling over their lifetime (+226%) at +4.7 net instructions per commit. Deletion hazard decays geometrically with instruction age, consistent with imperfect recall and with no other rival we test. Then, we show that prompt comments can halt the growth: in verifiable worlds b
    
[^291]: Halpern 迭代加速求解高阶单调变分不等式

    Halpern Iteration Accelerates Solving High-Order Monotone Variational Inequalities

    [https://arxiv.org/abs/2608.08463](https://arxiv.org/abs/2608.08463)

    本文通过重新审视经典 Halpern 迭代并提出大步长不精确的 $p$ 阶方法 Halpern-ATM，实现了求解光滑单调变分不等式的 $\tilde{\mathcal{O}}(T^{-p})$ 快速收敛速率，改进了自 2012 年以来 $p \ge 2$ 阶方法的已知最佳结果。

    

    在这篇简短笔记中，我们重新审视了1967年提出的经典 Halpern 迭代，并展示了其在设计求解光滑单调变分不等式（MVI）的二阶及更高阶方法中的新应用。通过使用大步长不精确 Halpern 迭代，我们提出了一种新颖的 $p$ 阶方法 Halpern-ATM，实现了 $\tilde{\mathcal{O}}(T^{-p})$ 的快速收敛速率。这改进了自 2012 年以来 $p \ge 2$ 阶方法的已知最佳结果。

    arXiv:2608.08463v2 Announce Type: replace-cross  Abstract: In this short note, we revisit the classical Halpern iteration in 1967 and show its new applications in designing second- and higher-order methods for solving smooth monotone variational inequalities (MVI). By using a large-step inexact Halpern iteration, we propose a novel $p$th-order method Halpern-ATM, that achieves a fast convergence rate of $\tilde{\mathcal{O}}(T^{-p})$. This improves the best-known results for $p \ge 2$ since 2012.
    
[^292]: BODHI：大语言模型能否分支探索并发现异构推理？

    BODHI: Do LLMs Branch Out and Discover Heterogeneous Inferences?

    [https://arxiv.org/abs/2608.02867](https://arxiv.org/abs/2608.02867)

    研究发现RLVR训练的大语言模型相比蒸馏模型表现出更低的语义多样性和更少的推理分支，表明其推理性能提升主要来自采样效率的改善而非推理能力的真正扩展。

    

    尽管基于可验证奖励的强化学习（RLVR）已经提升了对大语言模型（LLM）的推理性能，但它究竟是扩展了推理能力，还是主要提高了采样效率，这一问题仍然存在争议。在本文中，我们通过受控的迷宫求解实验和 BODHI-TREEs（一种可复用的表示方法，基于语义等价性对数学推理片段进行分组）来研究 RLVR 训练的大语言模型在测试时探索的本质。这些探测方法有助于刻画模型在风格变化与语义不同的续写之间的偏好。我们表明，与蒸馏模型相比，RLVR 训练的模型在语义不同的续写之间表现出更低的候选偏好熵；互补的采样实验表明，在相同的采样预算下，模型表现出更低的经验语义熵和更少的观测分支。因此，策略集中化不仅体现在风格层面……

    arXiv:2608.02867v2 Announce Type: replace-cross  Abstract: Although reinforcement learning with verifiable rewards (RLVR) has improved the reasoning performance of large language models (LLMs), whether it expands reasoning capabilities or primarily improves sampling efficiency remains debated. In this paper, we investigate the nature of test-time exploration in RLVR-trained LLMs by employing controlled maze-solving experiments and BODHI-TREEs, a reusable representation that groups mathematical reasoning segments based on semantic equivalence. These probes help delineate preferences between stylistic variations and semantically distinct continuations. We show that RLVR-trained models exhibit lower candidate preference entropy between semantically distinct continuations than their distilled counterparts, and complementary sampling experiments show lower empirical semantic entropy and fewer observed branches under equal sampling budgets. Thus, policy concentration extends beyond stylistic
    
[^293]: 真实环境中的智能体编程：生产规模下的GitHub Copilot追踪数据特征分析

    Agentic Coding in the Wild: Characterizing GitHub Copilot Traces at Production Scale

    [https://arxiv.org/abs/2608.00101](https://arxiv.org/abs/2608.00101)

    本文首次在生产规模上（740万用户、57亿次LLM调用）对GitHub Copilot编码智能体的追踪数据进行系统性分析，揭示了LLM调用与工具执行近乎1:1耦合、KV缓存命中率在轮次边界和模型切换时大幅下降、以及长尾资源占用分布等独特工作负载特性及其系统启示。

    

    AI编码智能体（如GitHub Copilot、Claude Code、Codex）将多步骤的LLM推理与工具执行交错进行，形成了一种有别于聊天机器人的新型工作负载。我们利用2026年6月至7月采样的GitHub Copilot追踪数据，首次对编码智能体进行了生产规模的表征分析，数据涵盖740万用户、8.19亿条对话、57亿次LLM调用以及946万亿token。我们的分析揭示了独特的工作负载特性及其对系统设计的特殊影响：例如，稀疏的用户发起轮次会展开为自主运行的智能体循环，LLM调用与工具执行之间呈现接近1:1的耦合关系；并行LLM调用较为罕见，但在触发子智能体的轮次中存在；KV缓存命中率在轮次内部平均可达90%，但在轮次边界处降至52%，在模型切换后更是降至6%；工作流和用户行为表现出多样化且呈长尾分布的资源占用特征。我们还进一步区分了较短的智能体响应时间与长达数分钟的用户空闲期……

    arXiv:2608.00101v2 Announce Type: replace  Abstract: AI coding agents (e.g., GitHub Copilot, Claude Code, Codex) interleave multi-step LLM inference with tool execution, creating a workload different from chatbots. We present the first production-scale characterization of coding agents using sampled GitHub Copilot traces from June-July 2026, comprising 7.4M users, 819M conversations, 5.7B LLM calls, and 946T tokens. Our analysis reveals distinctive workload properties with unique systems implications: e.g., sparse user-initiated turns unfold into autonomous agent loops with nearly 1:1 coupling between LLM calls and tool execution; parallel LLM calls are rare but exist in turns that trigger subagents; KV cache hit rates average 90% within a turn, but fall to 52% across turn boundaries and to 6% after model switches; workflows and user behaviors exhibit diverse, long-tailed resource footprints. We further distinguish short agentic turnaround times from minutes-long user idle periods betw
    
[^294]: Tokens就够了：面向推荐系统中LLM级I/O效率的双用途语义ID

    Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in recommendation systems

    [https://arxiv.org/abs/2607.24865](https://arxiv.org/abs/2607.24865)

    提出双用途语义ID，通过分层量化将连续嵌入压缩为离散token，既作为协同身份建模用户-物品交互，又可通过轻量级解码器按需重建嵌入，从而取代大规模向量存储，实现推荐系统中LLM级的I/O效率。

    

    大规模推荐系统由于庞大而密集的嵌入表而面临“内存墙”瓶颈。虽然生成式检索使用离散token作为ID，但高维上下文仍然依赖于低效的密集格式。受计算机视觉数据压缩的启发，我们提出了双用途语义ID（Dual-purpose Semantic IDs）以实现LLM级的I/O效率。我们的方法使用分层量化将连续嵌入压缩为离散语义ID，这些语义ID同时承担两个角色：（1）协同身份：通过可学习嵌入表建模用户-物品交互；（2）内容重建：使用轻量级语义解码器进行即时嵌入近似。这种方法用按需重建取代了大规模向量存储，降低了系统开销和数据占用。我们通过离线评估以及在生产规模系统上的成功在线部署，证明了该框架的有效性。

    arXiv:2607.24865v2 Announce Type: replace-cross  Abstract: Large-scale recommendation systems face "Memory Wall" bottlenecks due to massive, dense embedding tables. While generative retrieval uses discrete tokens for IDs, high-dimensional context still relies on inefficient dense formats. Inspired by computer vision data compression, we propose Dual-purpose Semantic IDs to achieve LLM-level I/O efficiency. Our methodology uses hierarchical quantization to condense continuous embeddings into discrete Semantic IDs performing two concurrent roles: (1) Collaborative Identity: modeling user-item interactions via learnable embedding table; and (2) Content Reconstruction: using a lightweight Semantic Decoder for on-the-fly embedding approximation. This approach replaces massive vector storage with on-demand reconstruction, reducing system overhead and data footprints. We demonstrate the efficacy of our framework through offline evaluations and successful online deployment in production-scale 
    
[^295]: 用于冠状动脉血管造影SYNTAX分割的多中心数据集与基准测试

    A Multi-Center Dataset and Benchmark for SYNTAX Segmentation of Coronary Angiograms

    [https://arxiv.org/abs/2607.22139](https://arxiv.org/abs/2607.22139)

    该论文推出了CARDIAG——首个针对冠状动脉造影SYNTAX节段分割的多中心基准数据集，通过严格防止患者级和中心级数据泄露并提供前所未有的丰富注释（不平衡多类标签、导管与狭窄掩膜、专家不确定性图等），同时对24种涵盖CNN、ViT和SSM的深度学习架构进行了该领域最广泛的基准评估。

    

    冠状动脉血管造影的密集像素级分类对于自动化心血管诊断至关重要，然而计算机视觉界缺乏金标准协议和鲁棒的数据集来应对这一拓扑复杂、长尾分布严重的领域。为了弥合这一差距，我们推出了CARDIAG：一个严谨的多中心基准数据集，旨在评估深度学习架构在细粒度SYNTAX节段分类上的表现。与以往的研究不同，CARDIAG系统地防止了患者级和中心级的数据泄露，同时提供了前所未有的丰富注释套件：高度不平衡的多类别标签、显式的导管与狭窄掩膜、专家不确定性图、时序序列帧以及空间元数据。利用CARDIAG，我们开展了该领域最广泛的架构评估，对涵盖经典CNN、视觉Transformer（ViTs）和状态空间模型（SSMs）在内的24个模型进行了基准测试。

    arXiv:2607.22139v2 Announce Type: replace-cross  Abstract: Dense pixel-level classification of coronary angiography is essential for automated cardiovascular diagnostics, yet the computer vision community lacks gold-standard protocols and robust datasets to tackle this topologically complex, heavily long-tailed domain. To bridge this gap, we introduce CARDIAG: a rigorous, multi-center benchmark dataset designed to evaluate deep learning architectures on fine-grained SYNTAX segment classification. Unlike prior efforts, CARDIAG systematically prevents patient- and center-level data leakage while providing an unprecedented suite of annotations: highly imbalanced multi-class labels, explicit catheter and stenosis masks, expert uncertainty maps, temporal sequence frames, and spatial metadata. Using CARDIAG, we conduct the most extensive architectural sweep in this domain, benchmarking 24 models spanning classic CNNs, Vision Transformers (ViTs), and State-Space Models (SSMs). Moving beyond n
    
[^296]: R3D：面向第一人称视角可穿戴设备的定量3D空间推理

    R3D: Quantitative 3D Spatial Reasoning for Egocentric Wearables

    [https://arxiv.org/abs/2607.02921](https://arxiv.org/abs/2607.02921)

    该论文提出了R3D-Bench——一个基于第一人称RGB-D视频、包含2,950道定量空间推理问题的基准测试，并引入了模型无关的空间工具调用框架R3D，通过分割和深度提升的对象表示构建3D场景来实现定量3D空间推理。

    

    从第一人称视角的RGB-D视频中进行定量3D空间推理是下一代可穿戴助手的一项重要能力。然而，现有的基准测试并未反映处理以下问题的挑战：（1）自然的第一人称视频，（2）带有位姿的RGB-D视频输入，以及（3）具有挑战性的定量3D空间推理问答。为填补这一空白，我们推出了R3D-Bench（Reasoning in 3D），这是一个包含2,950道定量空间推理问题的基准测试，涵盖15种类型——包括选择题、基于距离的推理问题和体积推理问题——构建于来自Aria Digital Twin的57个第一人称视频序列之上。为了在该数据集上建立强大的基线，我们提出了R3D，这是一个与模型无关的空间工具调用框架。与现有直接将3D信息嵌入模型输入表示的方法不同，R3D使用分割和深度提升的对象表示从视频中构建3D场景。它提供此信息……

    arXiv:2607.02921v2 Announce Type: replace-cross  Abstract: Quantitative 3D spatial reasoning from egocentric RGB-D video is an important capability for next-generation wearable assistants. Yet existing benchmarks do not reflect the challenges of handling (1) natural egocentric video, (2) posed RGB-D video inputs, and (3) challenging quantitative 3D spatial reasoning Q&A. To fill this gap, we introduce R3D-Bench (Reasoning in 3D), a benchmark of 2,950 quantitative spatial reasoning questions across 15 types--spanning multiple-choice, distance-based, and volumetric reasoning questions--built on top of 57 egocentric video sequences from Aria Digital Twin. To set a strong baseline on this dataset, we introduce R3D, a model-agnostic spatial tool-calling framework. In contrast to existing approaches that directly embed 3D information into the model's input representation, R3D constructs a 3D scene from video using segmentation and depth-lifted object representations. It provides this informa
    
[^297]: CodeTeam：一个基于大语言模型的多智能体仓库级代码生成框架

    CodeTeam: An LLM-Powered Multi-Agent Framework for Repository-Level Code Generation

    [https://arxiv.org/abs/2606.22082](https://arxiv.org/abs/2606.22082)

    CodeTeam是一个基于大语言模型的多智能体框架，通过将规划、决策与实现划分为独立协调的阶段——由多个架构师智能体竞争起草设计草图、CTO智能体将其规范化为机器可校验的契约、开发者智能体据此生成代码——来完成仓库级代码生成任务。

    

    自然语言到仓库生成（NL2Repo）要求系统能够根据自然语言需求文档构建一个完整的软件仓库。与函数级代码生成相比，该任务需要更长的规划视野、跨文件稳定的接口，以及对跨文件不一致问题的迭代调试。为了应对这些挑战，我们提出了CodeTeam，这是一个基于大语言模型的多智能体框架，它将规划、决策和实现分离为各自独立且相互协调的阶段。在规划阶段，多个Architect（架构师）智能体起草相互竞争的软件设计草图（SDS），并可选择性地以检索到的设计参考资料作为依据。随后，cto（首席技术官）智能体对这些SDS进行评估、选择和规范化，将最有前景的SDS转化为机器可校验的契约，该契约明确规定了文件归属、公共接口和依赖约束。在实现阶段，Developer（开发者）智能体生成代码。

    arXiv:2606.22082v2 Announce Type: replace-cross  Abstract: Natural language to repository generation (NL2Repo) requires a system to construct an entire software repository from a natural-language requirements document. Compared with function-level code generation, this task demands longer planning horizons, stable interfaces across files, and iterative debugging of cross-file inconsistencies. To address these challenges, we propose CodeTeam, an LLM-based multi-agent framework that separates planning, decision making, and implementation into distinct, coordinated stages. In the planning stage, multiple Architect agents draft competing software design sketches (SDSs), optionally grounded by retrieved design references. A cto (Chief Technical Officer) agent then evaluates, selects, and normalizes the most promising SDS into a machine-checkable contract that specifies file ownership, public interfaces, and dependency constraints. In the implementation stage, Developer agents generate code 
    
[^298]: 图遗忘无法抹去的内容：电动汽车充电网络中由物理规律导致的网络攻击信息泄露

    What Graph Unlearning Cannot Erase: Physics-Induced Leakage of Cyberattack Information in Electric Vehicle Charging Networks

    [https://arxiv.org/abs/2606.19566](https://arxiv.org/abs/2606.19566)

    该论文将图遗忘技术应用于电动汽车充电站的网络攻击定位任务，并揭示了即便执行了遗忘操作，网络攻击信息仍会经由电网的物理规律发生泄露，无法被完全抹除。

    

    电动汽车充电站（EVCS）可能使配电馈线面临网络攻击风险。尽管包括图神经网络（GNN）在内的机器学习方法能够定位哪个母线遭到入侵，但在数据共享和模型训练方面仍存在重大挑战。例如，数据共享协议赋予EVCS运营商基于隐私或安全原因从已部署模型中撤回其充电功率的权利，但若每次请求都从头重新训练，将重复完全重新训练（FR）的成本。为解决这一问题，我们研究用于EVCS网络攻击定位的图遗忘（GU），将其表述为图级多标签分类任务上的特征级遗忘问题。我们在IEEE 34节点、123节点和8500节点配电网络上，针对三种GNN骨干网络和累积遗忘场景，将微调（FT）与两种GU基线方法进行基准对比。实验表明，经过遗忘处理或在不包含被撤回数据的情况下重新训练的GNN……（原文摘要到此截断）

    arXiv:2606.19566v2 Announce Type: replace-cross  Abstract: Electric vehicle charging stations (EVCSs) can expose distribution feeders to cyberattacks. While machine learning methods, including graph neural networks (GNNs), can localize which bus is compromised, significant challenges remain in data sharing and model training. For example, data-sharing agreements grant EVCS operators the right to withdraw their charging power for privacy or security reasons from a deployed model, yet retraining from scratch on every request would repeat the full-retraining (FR) cost. To address this, we study graph unlearning (GU) for EVCS cyberattack localization, formulated as a feature-level unlearning problem on a graph-level multi-label classification task. We benchmark fine-tuning (FT) against two GU baselines on the IEEE 34-bus, 123-bus, and 8500-node distribution networks across three GNN backbones and cumulative unlearning scenarios. Experiments show that a GNN unlearned or retrained without th
    
[^299]: EgoPhys：基于第一人称视角视频的可变形物体物理信息数字孪生

    EgoPhys: Physics-Informed Digital Twins for Deformable Objects from Egocentric Video

    [https://arxiv.org/abs/2606.16202](https://arxiv.org/abs/2606.16202)

    提出EgoPhys框架，仅凭单段第一人称视角RGB交互视频即可为可变形物体构建预测性物理数字孪生，无需多视角采集或高昂的逐物体物理优化。

    

    人类通过日常交互自然而然地理解物体物理特性，但预测可变形物体（如软性物体和织物）如何响应操作，对计算机视觉和机器人技术而言仍是一个重大挑战。现有的可变形物理孪生方法通常依赖于受控的多视角或深度采集，以及代价高昂的逐物体物理优化。我们提出了EgoPhys，一个仅需单段第一人称视角RGB交互即可构建预测性可变形物理孪生的框架。EgoPhys将持久对应关系与逐帧单目3D观测相结合，从移动相机的观测中拟合针对特定序列的弹簧-质量模型。由于短暂的交互对细粒度局部刚度的约束较弱，EgoPhys通过一个紧凑的状态条件残差模型在多次训练交互中摊销这一细化过程，该模型在特定序列估计的基础上预测有界的局部校正，并且仍然……

    arXiv:2606.16202v2 Announce Type: replace-cross  Abstract: Humans naturally understand object physics through everyday interactions, but predicting how deformable objects, such as soft objects and fabrics, respond to manipulation remains a major challenge for computer vision and robotics. Existing deformable physical twin methods typically rely on controlled multi-view or depth capture and costly per-object physical refinement. We present EgoPhys, a framework for constructing predictive deformable physical twins from a single egocentric RGB interaction. EgoPhys combines persistent correspondences with per-frame monocular 3D observations to fit a sequence-specific spring-mass model from moving-camera observations. Because a short interaction weakly constrains fine-grained local stiffness, EgoPhys amortizes this refinement across training interactions using a compact state-conditioned residual model that predicts bounded local corrections around the sequence-specific estimate and remains
    
[^300]: 审计代码强化学习训练环境中的奖励可作弊性

    Auditing Reward Hackability in Code RL Training Environments

    [https://arxiv.org/abs/2606.16062](https://arxiv.org/abs/2606.16062)

    该论文通过基于标准补丁的执行验证系统审计了代码强化学习训练环境的奖励作弊问题，发现约22.4%（强攻击者下57.8%）的SWE-bench Verified任务会接受错误补丁，且模型在这些可作弊任务上的通过率显著更高，凸显了对RL训练环境进行作弊检测与修复审计的必要性。

    

    代码强化学习根据策略是否通过任务的测试来给予奖励，因此一个会接受错误补丁的测试套件就会训练出错误的行为。我们测量了开源代码RL环境接受错误补丁的频率，以及可作弊任务被检测和修复的可靠性，并通过将每个补丁与任务的标准补丁进行执行对比来验证所有结论。首先，在49个SWE-bench Verified任务中，22.4%的任务会接受一个经Docker验证但与标准补丁行为不同的补丁，面对更强的攻击者时这一比例达57.8%。标准补丁检查是必要的：未被告知标准补丁的攻击者在其通过的“漏洞利用”中有31%实际上返回的就是标准补丁本身。其次，在134个公开的SWE-bench提交中，模型通过可作弊任务的频率比同等难度的任务高出28.6个百分点，这主要源于这些任务本身是可以通过的；在三次攻击者运行汇总来看，可作弊任务的通过率仍显著超过匹配的可完成任务（p = 0.014）。第三，修复比检测更难验证：61.9%由LLM编写的修复补丁……（摘要原文在此处截断）

    arXiv:2606.16062v2 Announce Type: replace  Abstract: Code RL rewards a policy for passing a task's tests, so a test suite that accepts a wrong patch trains the wrong behavior. We measure how often open code RL environments accept wrong patches and how reliably hackable tasks can be detected and repaired, checking each claim by execution against the task's gold patch. First, 22.4% of 49 SWE-bench Verified tasks accept a Docker-verified patch that behaves differently from the gold patch, and 57.8% against a stronger attacker. The gold-patch check is necessary: an attacker not shown the gold patch returns it for 31% of its passing "exploits". Second, across 134 public SWE-bench submissions, models pass hackable tasks 28.6 pp more often than tasks of the same difficulty, mostly because those tasks are passable at all; pooled over three attacker runs, hackable tasks still exceed matched passable ones (p = 0.014). Third, repair is harder to verify than detection: 61.9% of LLM-written repair 
    
[^301]: 形态感知样本分配：克服表面缺陷检测中的IoU不敏感性

    Morphology-Aware Sample Assignment: Overcoming IoU Insensitivity for Surface Defect Detection

    [https://arxiv.org/abs/2606.13723](https://arxiv.org/abs/2606.13723)

    该论文通过理论分析发现IoU响应曲线存在非敏感区域，并提出融合面积、形状和长宽比的多维形态相似性度量来改进正样本分配，从而提升表面缺陷检测的精度与可靠性。

    

    交并比（IoU）作为评估候选提议框与真实标注之间空间对齐程度的关键指标，直接决定了正样本集合的质量以及视觉检测模型的训练效果。通过理论建模与分析，我们发现了IoU响应曲线上的一个非敏感区域，在该区域内，尽管样本具有不同的几何重叠程度，却会产生几乎相同的IoU分数。为克服这一局限性，我们引入了一组覆盖面积、形状和长宽比的形态相似性度量，以优化正样本分配过程，从而确保更具判别性和可靠性的匹配。通过对这些多维相似性进行基于均值的聚合，我们推导出一个补充匹配分数，以弥补IoU在表征结构对应关系方面的内在局限。从理论上讲，引入形态相似性……

    arXiv:2606.13723v3 Announce Type: replace-cross  Abstract: Intersection-over-Union (IoU), as a pivotal metric for evaluating the spatial alignment between candidate proposals and ground-truth annotations, directly determines the quality of positive sample sets and the training efficacy of visual detection models. Through theoretical modeling and analysis, we uncover a non-sensitive region on the IoU response curve, within which samples yield nearly identical IoU scores despite distinct geometric overlaps. To overcome this limitation, we introduce a set of morphological similarity metrics covering area, shape, and aspect ratio, to refine the positive sample assignment process, thereby ensuring more discriminative and reliable matching. A supplementary matching score is derived via mean-based aggregation of these multidimensional similarities, compensating for the intrinsic limitation of IoU in representing structural correspondence. Theoretically, incorporating morphological similarity 
    
[^302]: 算子链：一种用于上下文算子学习的推理时辅助框架

    Chain of Operators: An Inference-Time Harness for In-Context Operator Learning

    [https://arxiv.org/abs/2606.12318](https://arxiv.org/abs/2606.12318)

    本文提出算子链（CHOP）框架，利用冻结基础模型的上下文学习能力，将复杂分布外任务分解为闭式数学算子序列，实现无需参数更新的推理时适应。

    

    arXiv:2606.12318v2 公告类型：替换-交叉 摘要：尽管科学基础模型在加速物理模拟和数值预测方面展现出巨大潜力，但它们在面对分布外（OOD）场景时仍然非常脆弱。将这些通用模型适应于复杂的OOD任务通常需要昂贵的参数微调。在语言AI中，这一瓶颈通过使用“辅助框架”（harnesses）来绕过，这些框架作为外部脚手架，如推理链和工具使用，将冻结的权重适应于复杂任务，然而为物理领域设计等效的辅助框架仍是一个开放的前沿问题。为弥合这一差距，我们引入了算子链（CHOP），这是一个框架，它引导冻结的基础模型通过复杂的OOD任务而无需更新任何权重。通过利用上下文算子网络（ICON）的上下文学习能力，CHOP系统性地将不熟悉的问题分解为一系列显式、闭式的数学算子。

    arXiv:2606.12318v2 Announce Type: replace-cross  Abstract: While scientific foundation models show immense promise in accelerating physical simulations and numerical forecasting, they remain notoriously brittle when encountering out-of-distribution (OOD) scenarios. Adapting these generalist models to complex OOD tasks typically requires expensive parameter fine-tuning. In linguistic AI, this bottleneck is bypassed using "harnesses", which serve as external scaffolding such as reasoning chains and tool use to adapt frozen weights to complex tasks, yet designing an equivalent harness for physical domains remains an open frontier. To bridge this gap, we introduce Chain of Operators (CHOP), a framework that guides a frozen foundation model through complex OOD tasks without updating a single weight. By exploiting the in-context learning capability of In-Context Operator Networks (ICON), CHOP systematically decomposes unfamiliar problems into a sequence of explicit, closed-form mathematical 
    
[^303]: LLM智能体中面向多文件变更定位的探索结构

    Exploration Structure in LLM Agents for Multi-File Change Localization

    [https://arxiv.org/abs/2606.11976](https://arxiv.org/abs/2606.11976)

    该论文通过系统比较六种逐步增加结构的智能体配置，研究了小型LLM在多文件变更定位任务中哪些结构（如输出受限的仓库工具、领域智能体注册表等）真正带来了大部分性能收益。

    

    修复一个问题首先要找到需要修改的文件，而大多数基准测试中的问题涉及多个文件。用于此任务的LLM智能体越来越多地引入各种结构，例如专用工具、多智能体或代码摘要。我们探究当模型较小时，这些结构中哪些才是真正重要的。我们构建了一个协调者智能体，它配备有能限制输出规模的仓库工具，以及一个可复用的、按子系统划分范围的领域智能体注册表。我们比较了六种配置，每种配置在前一种的基础上增加结构：分别是单独的模型、带有Python REPL的单智能体、仅带REPL的协调者、前一结构加上仓库工具、再加上注册表地图，最后加上对领域智能体的咨询。我们还在来自三个Python代码库的105个多文件问题上，运行了使用相同模型的四个已发布定位器、两个更大模型的参照以及五个非LLM排序器，所有实验均在经过审计的单次提交代码快照上进行。大部分收益来自（原文摘要在此处截断）

    arXiv:2606.11976v2 Announce Type: replace-cross  Abstract: Fixing an issue starts with finding the files to change, and most benchmark issues touch several. LLM agents for this task increasingly add structure, such as specialized tools, multiple agents, or code summaries. We ask which of this structure matters when the model is small. We build a coordinator agent with repository tools that limit their output and a reusable registry of subsystem-scoped domain agents. We compare six configurations, each adding structure to the previous one. They are the model alone, a single agent with a Python REPL, a coordinator with only the REPL, the previous structure plus repository tools, then plus the registry map, and finally plus consultation of the domain agents. We also run four published localizers with the same model, two larger-model references, and five non-LLM rankers on 105 multi-file issues from three Python repositories, in audited single-commit checkouts. Most of the gain comes from 
    
[^304]: EssentialGIN：一种基于图同构神经网络预测基因必需性的新方法

    EssentialGIN: a new approach for gene essentiality prediction based on graph isomorphism neural networks

    [https://arxiv.org/abs/2606.07700](https://arxiv.org/abs/2606.07700)

    本研究提出EssentialGIN方法，通过改进图同构神经网络来保留PPI网络的拓扑特征，并整合基因表达、直系同源和亚细胞定位等生物信息，从而实现对必需基因的准确预测。

    

    背景：预测必需基因（蛋白质）是一个基础且具有挑战性的问题，同时在湿实验室实验中又非常昂贵且耗时。仅基于计算方法（用于筛选湿实验室候选基因）并使用中心性度量来预测必需基因并不准确，会导致大量假阳性；因此，近期研究采用了更复杂的模型（如深度学习），并整合生物信息来识别必需基因。方法：在这项工作中，我们专注于图同构网络，将蛋白质作为节点嵌入蛋白质-蛋白质相互作用（PPI）网络中，以保留PPI网络的拓扑特征，同时整合基因表达数据、基因直系同源信息和基因亚细胞定位信息等生物数据，并构建了一种用于预测必需基因的深度架构。本工作对该图同构网络架构进行了改进。

    arXiv:2606.07700v2 Announce Type: replace-cross  Abstract: Background: Prediction of essential genes (proteins), is a basic and challenging problem but at the same time very costly and time-consuming in wet-lab experiments. Predicting essential genes, only based on computational methods (to introduce wet-lab candidates) using centrality measures are not accurate and result in large number of false positives; therefore, more complex models such as deep learning and also integration of biological information are used in recent research to identify essential genes.   Methods: In this work we focus on graph isomorphism networks, in order to embed proteins as a node in PPI network to conserve topological features of PPI network, and also integrate biological data such as gene expression data, gene orthology information and gene subcellular localization information, and introduced a deep architecture for predicting essential genes. Graph isomorphism network architecture is modified in this w
    
[^305]: 知识驱动的工具使用工作流中AI智能体的声明式技能

    Declarative Skills for AI Agents in Knowledge-Grounded Tool-Use Workflows

    [https://arxiv.org/abs/2606.06923](https://arxiv.org/abs/2606.06923)

    该论文发现编排对工具使用型LLM智能体的作用取决于检索质量：完美检索时声明式技能文件中性甚至有害，嵌入检索时则有益，在基线与脚手架分歧的任务上声明式胜率从42.4%提升至62.7%。

    

    我们研究了编排（orchestration）如何影响在非结构化知识库上执行客户服务工作流的工具使用型大语言模型（LLM）智能体，比较了读取自然语言技能文件的声明式智能体、基于状态机构建的命令式智能体以及无脚手架的基线智能体。这三者被形式化为同一个去中心化部分可观测马尔可夫决策过程内的策略类，并在一个包含五个模型、两种检索机制和三种智能体的完整设计上通过配对检验进行评估。在所有条件下取平均时，编排并不能提高任务成功率。然而，其效果取决于智能体观察通道的质量。在完美检索条件下，技能文件在五个模型中的四个上表现为中性至负面；而在嵌入检索条件下，技能文件在五个模型中的四个上表现为中性至正面。在基线与脚手架结果不一致的任务上，声明式方法的胜率从42.4%上升到62.7%（p = 0.

    arXiv:2606.06923v2 Announce Type: replace  Abstract: We study how orchestration affects tool-using large language model (LLM) agents in customer-service workflows over an unstructured knowledge base, comparing a declarative agent that reads natural-language skill files, an imperative agent built on a state machine, and an unscaffolded baseline. The three are formalized as policy classes within one decentralized partially observable Markov decision process and evaluated on a complete design of five models, two retrieval regimes, and three agents, with paired tests. Averaged over all conditions, orchestration does not improve task success. The effect, however, depends on the quality of the agent's observation channel. Under perfect retrieval, skill files are neutral-to-negative on four of the five models; under embedding retrieval they are neutral-to-positive on four of the five. On the tasks where baseline and scaffold disagree, the declarative win rate rises from 42.4% to 62.7% (p = 0.
    
[^306]: Moment-Video：诊断视频多模态大语言模型在瞬时视觉事件上的时间保真度

    Moment-Video: Diagnosing Temporal Fidelity of Video MLLMs on Momentary Visual Events

    [https://arxiv.org/abs/2606.02522](https://arxiv.org/abs/2606.02522)

    提出了Moment-Video基准，通过涵盖七大门类、1,000个人工验证的视频问答对，系统评估视频多模态大语言模型在瞬时视觉事件理解上的时间保真度。

    

    视频多模态大语言模型（MLLMs）在通用视频理解和长视频理解方面取得了快速进展，但它们捕获并利用简短的、对答案起决定作用的视觉证据的能力仍缺乏充分研究。瞬时发生的动作或状态转变可能只占视频的一小部分，却可能直接决定答案。仅凭模型在宽泛视频理解任务上的表现，并不能说明模型能否可靠地识别并利用这种时间上局部化的证据。我们提出了Moment-Video，一个通过瞬时视觉事件理解来评估时间保真度的基准。问题围绕视觉上可观察的事件或事件序列构建，要求模型注意、计数、描述或推理瞬态证据。该基准包含1,000个经人工验证的视频问答对，覆盖七个领域和25个子类别，涉及四类任务：时间发生、时间计数、动作描述……（摘要原文在此处截断）

    arXiv:2606.02522v2 Announce Type: replace-cross  Abstract: Video multimodal large language models (MLLMs) have made rapid progress in general and long-form video understanding, yet their ability to capture and use brief, answer-critical visual evidence remains underexplored. Momentary actions or state transitions may determine an answer while occupying only a small fraction of the video. Performance on broad video understanding tasks alone does not establish whether models can reliably identify and use this temporally localized evidence. We introduce Moment-Video, a benchmark for evaluating temporal fidelity through momentary visual event understanding. Questions are constructed around visually observable events or event sequences, requiring models to notice, count, describe, or reason about transient evidence. The benchmark contains 1,000 human-verified video-QA pairs across seven domains and 25 subcategories, covering four tasks: Temporal Occurrence, Temporal Counting, Action Descrip
    
[^307]: 重新定义实例匹配：全景分割评估中部件感知匹配的统一框架

    Redefining Instance Matching: A Unified Framework for Part-Aware Matching in Panoptic Segmentation Evaluation

    [https://arxiv.org/abs/2605.31094](https://arxiv.org/abs/2605.31094)

    本文将全景分割评估中的片段匹配重新表述为受约束的二部图指派问题，通过独立约束预测侧与真实侧的度数定义了四种匹配策略（一对一、多对一、一对多、多对多），并证明除多对多之外的前三种策略均能给出可解释的实例级错误归因。

    

    全景质量指标是联合评估实例分割与语义分割的标准。其原始定义仅在预测分割与真实分割的交并比超过0.5时才进行匹配，这一阈值保证了既最优又唯一的一对一匹配。低于该阈值时，唯一性保证不复存在，多种匹配策略在一个探索甚少的问题空间中涌现。我们通过将分割匹配重新表述为一个受约束的二部指派问题，系统地阐明了这一空间。通过独立约束预测侧与真实侧的度数，我们得到四种匹配策略：一对一、多对一、一对多和多对多。我们证明前三种策略能够产生可解释的实例级指派，而多对多策略则将过分割与欠分割纠缠在一起，无法对错误进行一致的归因。这些策略在实例……（摘要原文在此处被截断）

    arXiv:2605.31094v2 Announce Type: replace-cross  Abstract: The Panoptic Quality metric is the standard for jointly evaluating instance and semantic segmentation. Its original definition matches predicted and ground truth segments only if their intersection over union exceeds 0.5, a threshold that guarantees a One-to-One matching that is both optimal and unique. Below this threshold, the uniqueness guarantee is lost and multiple matching strategies emerge in a poorly explored problem space. We systematically elucidate this space by recasting segment matching as a constrained bipartite assignment problem. Independently bounding the prediction- and ground-truth-side degrees yields four matching strategies: One-to-One, Many-to-One, One-to-Many, and Many-to-Many. We show that the first three yield interpretable instance-level assignments, whereas Many-to-Many entangles over- and under-segmentation and admits no consistent attribution of errors. These strategies become relevant when instance
    
[^308]: BioELX：无需任务特定监督的上下文感知跨语言生物医学实体链接

    BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision

    [https://arxiv.org/abs/2605.27380](https://arxiv.org/abs/2605.27380)

    BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。

    

    跨语言生物医学实体链接（BEL）将任何语言中的提及（mention）映射到生物医学知识库中的唯一标识符，为临床和生物医学NLP应用提供支持。我们发现了影响当前系统的两个问题。首先，用于训练跨语言BEL检索器的UMLS别名严重偏向英语，导致检索器对非英语提及的泛化能力较差。其次，尽管上下文通常对消歧是必要的，但将上下文简单地注入仅以别名对齐为目标训练的检索器中，会严重降低检索性能。我们提出了BioELX，一个解决这两个问题的检索-重排序框架。在检索阶段，我们使用源自Wikidata的跨语言别名监督对SapBERT_multi进行继续训练，从而在各语言之间形成共享的概念邻域。在重排序阶段，我们通过提及锚定提示（mention-anchored prompting）将预训练的LLM重排序器适配到实体链接任务。

    arXiv:2605.27380v2 Announce Type: replace-cross  Abstract: Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base, supporting clinical and biomedical NLP applications. We identify two issues affecting current systems. First, the UMLS (Bodenreider,2004) aliases used to train cross-lingual BEL retrievers are heavily skewed toward English, so retrievers generalize poorly to non-English mentions. Second, although context is often necessary for disambiguation, naively injecting context into retrievers trained only to align aliases severely degrades retrieval. We propose BioELX, a retrieve-rerank framework that addresses both issues. For retrieval, we continue training SapBERT_multi (Liu et al., 2021b) using Wikidata-derived cross-lingual alias supervision, forming shared concept neighborhoods across languages. For reranking, we adapt pretrained LLM rerankers to entity linking through mention-anchored prompting, which 
    
[^309]: 信赖域Q伴随匹配

    Trust Region Q Adjoint Matching

    [https://arxiv.org/abs/2605.27079](https://arxiv.org/abs/2605.27079)

    本文提出信赖域Q伴随匹配（TRQAM），通过在流策略采样过程中引入信赖域参数λ来精确加权路径空间KL散度，并借助投影对偶下降自适应控制策略更新幅度，从而解决了QAM中评论家误差被指数级放大导致性能崩溃的问题，实现了对预训练流策略的稳定离策略微调。

    

    对预训练流策略进行离策略强化学习仍然具有挑战性，其困难源于多步采样过程所带来的优化不稳定性。最近，带伴随匹配的Q学习（QAM）通过将策略改进重新表述为以学习到的评论家为指导的随机最优控制（SOC）问题，解决了这一难题。然而，QAM继承了评论家引导改进方式的一个根本性脆弱之处：微小的评论家误差会被指数级放大，并常常导致性能崩溃。本文提出了信赖域Q伴随匹配（TRQAM），这是一种稳定的离策略微调算法，通过投影对偶下降自适应地控制微调策略与预训练策略之间的路径空间KL散度。具体而言，我们在流策略的采样过程中引入了一个信赖域参数λ，并证明λ恰好对SOC目标中的路径空间KL散度进行加权。因此，我们的……（摘要截断）

    arXiv:2605.27079v2 Announce Type: replace-cross  Abstract: Off-policy reinforcement learning of pretrained flow policies remains challenging due to the instability of optimization arising from the multi-step sampling process. Recently, Q-learning with Adjoint Matching (QAM) addressed this by recasting policy improvement as a stochastic optimal control (SOC) problem guided by a learned critic. However, QAM inherits a fundamental fragility of critic-guided improvement, since small critic errors can be exponentially amplified and often lead to performance collapse. This paper introduces Trust Region Q Adjoint Matching (TRQAM), a stable off-policy fine-tuning algorithm that adaptively controls the path-space KL between the fine-tuned and pretrained policies through projected dual descent. Specifically, we adapt a trust-region parameter $\lambda$ inside the sampling process of the flow policy and prove that $\lambda$ exactly weights the path-space KL in the SOC objective. As a result, our m
    
[^310]: AutoDFT：一个用于自主DFT计算的闭环多智能体框架

    AutoDFT: A Closed-Loop Multi-Agent Framework for Autonomous DFT Calculations

    [https://arxiv.org/abs/2605.26179](https://arxiv.org/abs/2605.26179)

    AutoDFT是一个闭环多智能体框架，将大语言模型推理嵌入DFT计算生命周期的每个阶段，使DFT计算能够根据中间结果自主适应和调整计算路径，摆脱了传统工作流对专家干预和预先规划的依赖。

    

    密度泛函理论（DFT）是材料科学与化学计算发现的基础，然而每次计算都需要大量的人力投入：当收敛停滞时调整算法、当出现意外物理现象时修改方案、以及当中间结果改变问题形态时插入新的计算步骤。现有的基于大语言模型（LLM）的智能体仅能自动化初始规划阶段，即预先产出完整的执行计划，而将后续所有的适应性调整留给手工编写的规则。因此，这些工作流仍然十分脆弱，难以泛化到预先规划的场景之外，并且在失败或意外的中间结果需要更改计算路径时，往往需要专家干预。在此，我们提出AutoDFT，一个闭环多智能体框架，它将LLM推理嵌入DFT生命周期的每个阶段，其中战略规划器生成由步骤目标构成的骨架计划；步骤规划器生成……

    arXiv:2605.26179v3 Announce Type: replace-cross  Abstract: Density functional theory (DFT) serves as the basis for computational discovery in materials science and chemistry, yet each calculation demands extensive human effort: adjusting algorithms when convergence stalls, revising plans when unexpected physics emerges, and inserting steps as intermediate results reshape the problem. Existing LLM-based agents automate only the initial planning stage, producing a full execution plan upfront and leaving all subsequent adaptation to hand-crafted rules. As a result, these workflows remain fragile, do not generalize well beyond pre-planned scenarios, and often require expert intervention when failures or unexpected intermediate results require changes to the calculation path. Here, we introduce AutoDFT, a closed-loop multi-agent framework that embeds LLM reasoning into every stage of the DFT lifecycle, where a strategic planner produces a skeletal plan of step objectives; a step planner gen
    
[^311]: GENSTRAT：迈向大语言模型策略推理科学

    GENSTRAT: Toward a Science of Strategic Reasoning in Large Language Models

    [https://arxiv.org/abs/2605.23238](https://arxiv.org/abs/2605.23238)

    提出GENSTRAT系统，通过程序化生成双人零和不完全信息下注博弈按需创建全新评估游戏，解决了固定游戏套件饱和与数据污染问题，并通过50个游戏的基准和40,000多场模型对战的锦标赛揭示了不同LLM策略推理能力的显著差异。

    

    用于评估大语言模型（LLM）策略推理能力的固定游戏套件容易饱和，且难以避免数据污染。我们提出了GENSTRAT，一个通过程序化生成双人零和不完全信息下注博弈来评估LLM策略能力的系统，使评估者能够按需抽取新颖、全新的游戏。每个游戏在五个复杂性维度（状态空间、时间关联性、对手依赖性、私有信息、精确性）上进行特征刻画，并通过蒙特卡洛模拟进行测量。利用该生成器，我们构建了一个包含50个游戏的基准测试，并举办了十二个前沿模型和开放权重模型的锦标赛，收集了超过40,000场LLM对LLM比赛的数据。我们进一步构建了一个复合复杂性维度，发现在八个最困难的游戏上，最佳模型与最差模型之间的差距是其在全部游戏集合上差距的1.7倍。消融实验表明……（原文摘要不完整，此处为已提供部分的翻译）

    arXiv:2605.23238v2 Announce Type: replace-cross  Abstract: Fixed game suites for evaluating strategic reasoning in large language models (LLMs) saturate and are hard to keep free of contamination. We introduce GENSTRAT, a system for evaluating the strategic capabilities of LLMs driven by procedural generation of two-player zero-sum imperfect-information betting games, allowing an evaluator to draw novel, fresh games on demand. Each game is characterized on a scale of five complexity axes (state space, temporal contingency, opponent dependence, private information, precision) and measured by Monte Carlo simulation. Using this generator, we construct a 50-game benchmark and hold a tournament of twelve frontier and open-weight models, gathering data from over 40,000 LLM vs. LLM matches. We further construct a composite complexity axis, and find that, on the eight hardest games, the gap between the best and worst model is 1.7x its value across the whole set of games. Ablations show that a 
    
[^312]: 通过空间定位病灶证据实现临床可解释的眼科视觉问答

    Towards Clinically Interpretable Ophthalmic VQA via Spatially-Grounded Lesion Evidence

    [https://arxiv.org/abs/2605.22414](https://arxiv.org/abs/2605.22414)

    该论文提出了FundusGround基准，通过ETDRS网格对眼底病灶进行空间定位标注，构建了包含10,719张眼底图像、15,595个精细病灶标注和72,706个问题的数据集，推动临床可解释的眼科视觉问答发展。

    

    视觉问答在临床支持方面具有巨大前景，尤其是在眼科领域，视网膜眼底摄影对诊断至关重要。然而，现有的眼科视觉问答基准主要强调答案准确性，忽视了临床可解释性所必需的显式视觉证据。在这项工作中，我们提出了FundusGround，一个基于空间定位病灶证据的临床可解释眼科视觉问答新基准。具体而言，我们设计了一个三阶段流程，收集了10,719张眼底图像，并对15,595个图像级病灶进行了精细标注。为确保解剖学一致性和临床有效性，所有病灶均采用早期治疗糖尿病视网膜病变研究（ETDRS）网格进行空间定位，从而实现向九个具有临床意义的视网膜区域的标准化映射。基于这一结构化的病灶证据，随后生成了72,706个问题，涵盖四种格式……

    arXiv:2605.22414v2 Announce Type: replace-cross  Abstract: Visual Question Answering (VQA) holds great promise for clinical support, particularly in ophthalmology, where retinal fundus photography is essential for diagnosis. However, ophthalmic VQA benchmarks primarily emphasize answer accuracy, neglecting the explicit visual evidence necessary for clinical interpretability. In this work, we introduce FundusGround, a new benchmark for clinically interpretable ophthalmic VQA with spatially-grounded lesion evidence. Specifically, we propose a three-stage pipeline that collects 10,719 fundus images with 15,595 image-level meticulously annotated lesions. To ensure anatomical consistency and clinical validity, all lesions are spatially localized using the Early Treatment Diabetic Retinopathy Study (ETDRS) grid, enabling standardized mapping to nine clinically meaningful retinal regions. Built upon this structured lesion evidence, 72,706 questions are then generated spanning four formats: op
    
[^313]: CatalyticMLLM：面向催化材料的图-文本多模态大语言模型

    CatalyticMLLM: A Graph-Text Multimodal Large Language Model for Catalytic Materials

    [https://arxiv.org/abs/2605.17254](https://arxiv.org/abs/2605.17254)

    提出CatalyticMLLM，一个统一的图-文本多模态大语言模型，将催化材料的性质预测与逆向结构设计整合到单一模型中，避免了传统解耦范式因适用域不匹配与误差累积而导致的逆向设计性能下降问题。

    

    催化材料的性质预测与逆向结构设计通常被建模为两个独立的任务：前者从给定结构预测目标性质，后者根据期望性质生成候选结构。尽管这种解耦范式便于构建“生成—评估—筛选”的工作流程，但由于两个模型在不同的数据分布上训练，它们的适用域可能不同。因此，在闭环优化过程中，生成模型可能产生超出评估模型可靠适用域的结构，而评估模型的预测误差会在 successive 迭代中不断累积，最终导致逆向设计性能下降。在本工作中，我们提出了CatalyticMLLM，这是一个面向催化材料的统一图-文本多模态大语言模型，它整合了……（摘要在此处截断）

    arXiv:2605.17254v5 Announce Type: replace  Abstract: Property prediction and inverse structural design of catalytic materials are typically modeled as two independent tasks: the former predicts target properties from given structures, whereas the latter generates candidate structures conditioned on desired properties. Although this decoupled paradigm facilitates the construction of a ``generation--evaluation--screening'' workflow, the two models may have different domains of applicability because they are trained on different data distributions. Consequently, during closed-loop optimization, the generative model may produce structures that fall outside the reliable applicability domain of the evaluation model, while prediction errors from the evaluator may accumulate over successive iterations, ultimately degrading inverse-design performance.   In this work, we propose CatalyticMLLM, a unified graph--text multimodal large language model for catalytic materials that integrates \textbf{p
    
[^314]: StreamPro：基于多粒度奖励强化学习的主动式流式视频理解

    StreamPro: Reinforcement Learning with Multi-Grained Rewards for Proactive Streaming Video Understanding

    [https://arxiv.org/abs/2605.16381](https://arxiv.org/abs/2605.16381)

    提出StreamPro两阶段训练框架，先用CB-Stream Loss缓解SFT中沉默与响应信号的监督不平衡，再通过响应匹配奖励与序列质量奖励相结合的多粒度GRPO强化学习，解决奖励稀疏与响应序列依赖优化的难题，实现主动式流式视频理解。

    

    主动式流式视频理解要求模型持续监控不断演化的视频流，并决定是否响应、何时响应以及响应什么内容。然而，训练具备这种主动能力的模型仍然具有挑战性。监督微调（SFT）存在沉默信号与响应信号之间的严重不平衡问题，而强化学习（RL）则面临响应级别奖励稀疏以及难以优化相互依赖的响应序列的困难。为应对这些挑战，我们提出了StreamPro，一个两阶段训练框架。我们首先引入CB-Stream Loss来缓解SFT期间的监督不平衡问题。随后我们采用群体相对策略优化（GRPO），并配合两种互补的奖励：一种精细化的响应匹配（RM）奖励，为单个响应决策提供更密集的反馈；以及一种基于评分准则的序列质量（SQ）奖励，鼓励生成全局连贯的响应序列。为了支持这两种（奖励机制……）

    arXiv:2605.16381v2 Announce Type: replace-cross  Abstract: Proactive streaming video understanding requires models to continuously monitor evolving video streams and decide whether, when, and what to respond. However, training models with such proactive capabilities remains challenging. Supervised fine-tuning (SFT) suffers from severe imbalance between silence and response signals, while reinforcement learning (RL) faces sparse response-level rewards and difficulties in optimizing interdependent response sequences. To address these challenges, we propose StreamPro, a two-stage training framework. We first introduce CB-Stream Loss to alleviate supervision imbalance during SFT. We then employ Group Relative Policy Optimization (GRPO) with two complementary rewards: a refined Response-Matching (RM) reward that provides denser feedback for individual response decisions, and a rubric-based Sequence-Quality (SQ) reward that encourages globally coherent response sequences. To support the two 
    
[^315]: PhyMotion：面向物理合理人体视频生成的结构化三维运动奖励

    PhyMotion: Structured 3D Motion Reward for Physics-Grounded Human Video Generation

    [https://arxiv.org/abs/2605.14269](https://arxiv.org/abs/2605.14269)

    PhyMotion提出了一种结构化的三维运动奖励，通过将生成视频中恢复的人体运动导入物理模拟器进行重定向和物理可行性评估，从而生成更加真实、符合物理规律的人体运动视频。

    

    生成逼真的人体运动是视频生成领域一个核心但尚未解决的挑战。尽管基于强化学习（RL）的后训练近年来显著提升了整体视频质量，但将其扩展到人体运动仍然受制于一个瓶颈：现有奖励信号无法可靠地评估运动的真实性。现有的视频奖励主要依赖二维感知信号，没有显式建模铰接式人体运动背后的三维身体状态、接触和动力学，因此常常给出现身体体漂浮或物理上不合理运动的视频打高分。为了解决这一问题，我们提出了PhyMotion，这是一种结构化、细粒度的运动奖励，它将恢复的三维人体轨迹置于物理模拟器中进行验证，并从物理可行性的多个维度评估运动质量。具体而言，我们从生成的视频中恢复SMPL身体网格，将其重定向到MuJoCo物理模拟器中的虚拟人形模型上，并评估……

    arXiv:2605.14269v2 Announce Type: replace-cross  Abstract: Generating realistic human motion is a central yet unsolved challenge in video generation. While reinforcement learning (RL)-based post-training has driven recent gains in general video quality, extending it to human motion remains bottlenecked by a reward signal that cannot reliably score motion realism. Existing video rewards primarily rely on 2D perceptual signals, without explicitly modeling the 3D body state, contact, and dynamics underlying articulated human motion, and often assign high scores to videos with floating bodies or physically implausible movements. To address this, we propose PhyMotion, a structured, fine-grained motion reward that grounds recovered 3D human trajectories in a physics simulator and evaluates motion quality along multiple dimensions of physical feasibility. Concretely, we recover SMPL body meshes from generated videos, retarget them onto a humanoid in the MuJoCo physics simulator, and evaluate 
    
[^316]: 评论与控制：通过上下文锚定演化劫持智能体工作流

    Comment and Control: Hijacking Agentic Workflows via Context-Grounded Evolution

    [https://arxiv.org/abs/2605.11229](https://arxiv.org/abs/2605.11229)

    该论文提出了首个针对自动化平台上LLM智能体工作流的检测与利用框架JAW，其通过“上下文锚定演化”方法在混合程序分析推导出的上下文中演化恶意输入（如issue评论），从而劫持智能体工作流，实现凭据窃取和任意命令执行等攻击。

    

    GitHub Actions和n8n等自动化平台正日益采用所谓的智能体工作流，即将大语言模型（LLM）智能体集成进来，用于代码审查和数据同步等任务。虽然这为开发者带来了便利，但这种集成也暴露了一种新风险：攻击者可能控制和构造某些输入（例如GitHub issue评论），从而操纵LLM智能体执行非预期操作，例如凭据窃取和任意命令执行。据我们所知，此前尚无学术工作研究过智能体工作流中的此类风险。在本文中，我们设计了首个检测与利用框架JAW，通过一种名为“上下文锚定演化”的新方法来劫持托管在自动化平台上的智能体工作流。我们的核心思想是在由混合程序分析推导出的上下文中演化智能体工作流的输入，以实现劫持目的。具体而言，JAW生成智能体……（摘要在此处截断）

    arXiv:2605.11229v2 Announce Type: replace-cross  Abstract: Automation platforms such as GitHub Actions and n8n are increasingly adopting so-called agentic workflows, which integrate Large Language Model (LLM) agents for tasks such as code review and data synchronization. While bringing convenience for developers, this integration exposes a new risk: An adversary may control and craft certain inputs, such as GitHub issue comments, to manipulate the LLM agent for unwanted actions, such as credential exfiltration and arbitrary command execution. To our knowledge, no prior academic work has studied such a risk in agentic workflows. In this paper, we design the first detection and exploitation framework, called JAW, to hijack agentic workflows hosted on automation platforms via a novel approach called Context-Grounded Evolution. Our key idea is to evolve agentic workflow inputs under the contexts derived from hybrid program analysis for hijacking purposes. Specifically, JAW generates agenti
    
[^317]: NoiseRater：面向扩散模型训练的元学习噪声评估方法

    NoiseRater: Meta-Learned Noise Valuation for Diffusion Model Training

    [https://arxiv.org/abs/2605.08144](https://arxiv.org/abs/2605.08144)

    提出NoiseRater，一种通过双层元学习对给定样本和时间步条件下的噪声实例进行评分的模型，用于选择更有信息量的训练噪声，并发现同一时间步的不同噪声实现并非同等有用。

    

    训练扩散模型时，每个数据样本涉及两个随机性来源：时间步和高斯噪声的具体实现。时间步已通过调度与加权等方式得到广泛研究，而给定时间步下噪声实现的影响却仍未被充分探索。在这项工作中，我们研究了不同的噪声实例是否具有同等的信息量。我们提出了NoiseRater，这是一个在给定数据样本和时间步的条件下对单个噪声实例进行打分的网络。该评估器通过双层优化进行学习：其打分在内循环中对扩散损失进行重新加权，而在内循环更新之后，评估器通过降低验证损失来进一步更新。利用训练好的评估器来选择训练噪声，我们观察到训练噪声的三个特性。首先，同一时间步的噪声实现并非同等有用：评估器给出的最高分噪声相比独立同分布采样能带来性能提升，而其…

    arXiv:2605.08144v2 Announce Type: replace-cross  Abstract: Training a diffusion model involves two sources of randomness for each data sample: the timestep and the Gaussian noise realization. The timestep has been studied extensively through scheduling and weighting, whereas the impact of the noise realization at a given timestep is still underexplored. In this work, we examine whether different noise instances are equally informative. We introduce NoiseRater, a network that scores an individual noise instance conditioned on the data sample and timestep. The rater is learned through bilevel optimization, where its scores reweight the diffusion loss in the inner loop, and it is updated to reduce validation loss after the inner-loop updates. Using the trained rater to select training noise, we observe three properties of training noise. First, noise realizations at the same timestep are not equally useful: the rater's top-scored noise improves performance over i.i.d.\ sampling, while its
    
[^318]: Mosaic：高效的CPU-GPU协同块稀疏注意力

    Mosaic: Efficient CPU-GPU Coordinated Block-Sparse Attention

    [https://arxiv.org/abs/2605.07719](https://arxiv.org/abs/2605.07719)

    Mosaic提出了一种CPU-GPU协同的块稀疏注意力系统，通过输出贡献感知的KV预算分配、注意力头特定且粒度感知的稀疏配置以及跨设备协同调度，解决了长上下文LLM解码中CPU驻留KV缓存访问受PCIe带宽限制和GPU闲置的低效问题。

    

    arXiv:2605.07719v2 公告类型：replace-cross 摘要：长上下文LLM服务日益依赖于驻留在CPU上的上下文KV缓存，这由KV卸载和分离式预填充-解码执行所推动。尽管块稀疏注意力减少了KV访问量，但现有设计仍然效率低下：GPU集中式执行在每个解码步骤中都需要从CPU内存获取选定的KV块，使其受限于PCIe带宽；而CPU-GPU分区式执行则将CPU侧的稀疏注意力置于关键路径上，导致大量GPU时间闲置。本文提出了Mosaic，一个用于在CPU驻留上下文KV下实现高效长上下文解码的CPU-GPU协同块稀疏注意力系统。Mosaic建立在三个关键洞察之上：输出贡献感知的KV预算分配、针对注意力头且粒度感知的稀疏配置，以及针对CPU驻留KV的跨设备协同调度。基于这些洞察，Mosaic集成了一个轻量级的注意力头属性预测器和一个粒度-预算（摘要在此处被截断）

    arXiv:2605.07719v2 Announce Type: replace-cross  Abstract: Long-context LLM serving increasingly relies on CPU-resident context KV cache, driven by KV offloading and disaggregated prefill--decode execution. Although block-sparse attention reduces KV access, existing designs remain inefficient: GPU-centralized execution fetches selected KV blocks from CPU memory at each decoding step, making it PCIe-bound, while CPU--GPU partitioned execution places CPU-side sparse attention on the critical path, leaving substantial GPU time idle. This paper presents Mosaic, a CPU--GPU coordinated block-sparse attention system for efficient long-context decoding with CPU-resident context KV. Mosaic builds on three key insights: output-contribution-aware KV budgeting, head-specific and granularity-aware sparse configuration, and cross-device coordinated scheduling for sparse tasks over CPU-resident KV. Guided by these insights, Mosaic integrates a lightweight head-property predictor, a granularity--budge
    
[^319]: 基于大语言模型生成启发式方法的分层任务网络规划

    Hierarchical Task Network Planning with LLM-Generated Heuristics

    [https://arxiv.org/abs/2605.07707](https://arxiv.org/abs/2605.07707)

    本研究将基于大语言模型生成启发式方法的技术从经典规划扩展到分层任务网络（HTN）规划，探索LLM能否为HTN规划生成有效的搜索启发式，并在七个标准全序HTN基准领域上验证其效果。

    

    HTN规划是经典规划的一种变体，其中算法不是搜索线性动作序列，而是利用方法库对更高层的任务进行分解，直到只剩下可执行的动作。一方面，这允许人们通过方法库引入领域知识，从而加速求解的搜索过程；另一方面，它也带来了超越经典状态空间搜索的挑战。尽管最近的研究提出了许多启发式方法和新颖算法来加速HTN规划，但这些启发式方法的信息量仍不及经典规划算法中已有的启发式方法。我们研究大语言模型（LLM）能否为HTN规划生成有效的搜索启发式方法，将Corrêa等人（arXiv:2503.18809）的方法论从经典规划扩展到分层规划。我们使用Pytrich规划器在七个标准的全序HTN基准领域上进行了评估……（原文摘要至此截断）

    arXiv:2605.07707v3 Announce Type: replace  Abstract: HTN planning is a variation of classical planning where, instead of searching for a linear sequence of actions, an algorithm decomposes higher-level tasks using a method library until only executable actions remain. On the one hand, this allows one to introduce domain knowledge that can speed up the search for a solution through the method library. On the other hand, it creates challenges that go beyond those of classical state-space search. While recent research produced a number of heuristics and novel algorithms that speed up HTN planning, these heuristics are not yet as informative as those available in classical planning algorithms. We investigate whether large language models (LLMs) can generate effective search heuristics for HTN planning, extending the methodology of Corr\^ea et al. (arXiv:2503.18809) from classical to hierarchical planning. Using the Pytrich planner on seven standard total-order HTN benchmark domains, we eva
    
[^320]: 何时以及为何SignSGD优于SGD：基于ℓ₁-范数下界的理论研究

    When and Why SignSGD Outperforms SGD: A Theoretical Study Based on $\ell_1$-norm Lower Bounds

    [https://arxiv.org/abs/2605.06615](https://arxiv.org/abs/2605.06615)

    本文通过引入ℓ₁-范数平稳性、ℓ∞-平滑性和可分离噪声模型这一新问题几何框架，首次从理论上证明了SignSGD在何种条件下以及为何能够超越SGD，并给出了匹配的复杂度上下界。

    

    基于符号的优化算法，如SignSGD和Muon，因其在训练大型基础模型中的卓越表现而受到广泛关注。尽管取得了这些实证成功，我们仍然缺乏对这些基于符号的方法何时以及为何优于普通SGD的理论理解。核心障碍在于，在标准的平滑性和有限方差条件下，SGD已被证明在以ℓ₂-范数度量驻点的优化问题中是极小极大最优的，这从根本上排除了基于符号的方法在标准设置中取得任何复杂度增益的可能性。为克服这一障碍，我们利用ℓ₁-范数平稳性、ℓ∞-平滑性以及可分离噪声模型来分析基于符号的优化器，这些条件能够更好地刻画符号更新的逐坐标特性。在这种独特的问题几何结构下，我们推导出了SignSGD的匹配上下界，并明确刻画了问题的类别……

    arXiv:2605.06615v2 Announce Type: replace-cross  Abstract: Sign-based optimization algorithms, such as SignSGD and Muon, have garnered significant attention for their remarkable performance in training large foundation models. Despite this empirical success, we still lack a theoretical understanding of when and why these sign-based methods outperform vanilla SGD. The core obstacle is that under standard smoothness and finite variance conditions, SGD is known to be minimax optimal for finding stationary points measured by $\ell_2$-norms, thereby fundamentally precluding any complexity gains for sign-based methods in standard settings. To overcome this barrier, we analyze sign-based optimizers leveraging $\ell_1$-norm stationarity, $\ell_\infty$-smoothness, and a separable noise model, which can better capture the coordinate-wise nature of signed updates. Under this distinct problem geometry, we derive matched upper and lower bounds for SignSGD and explicitly characterize the problem cla
    
[^321]: ScrapMem：一种通过光学遗忘实现设备端个性化智能体记忆的仿生框架

    ScrapMem: A Bio-Inspired Framework for On-Device Personalized Agent Memory via Optical Forgetting

    [https://arxiv.org/abs/2605.03804](https://arxiv.org/abs/2605.03804)

    ScrapMem提出了一种仿生的设备端智能体记忆框架，通过“光学遗忘”机制逐步压缩旧记忆页面的图像质量以降低存储成本，同时借助情景记忆图保持语义一致性与结构化检索能力。

    

    长期个性化记忆对于在资源受限的边缘设备上运行的LLM智能体而言极具挑战性，因为多模态历史记录会持续增长且存储成本高昂。为应对这一挑战，我们提出了ScrapMem，一个受生物学启发的框架，它将每天的异构记录整合为统一的“剪贴簿页面”。ScrapMem引入了光学遗忘机制，这是一种光学压缩机制，通过逐步降低较旧页面的分辨率和JPEG质量来降低存储成本，同时使近期证据保持较高的保真度。为保持语义一致性并支持结构化检索，我们构建了情景记忆图，将语义节点组织成具有因果-时序结构的EM路径。这一设计将稳定的检索结构与逐渐老化的视觉证据相分离，使得即使页面保真度降低，相关情景仍然可以访问。在通用的ATM-Bench协议下，不带遗忘机制的ScrapMem确立了……（摘要在此处截断）

    arXiv:2605.03804v3 Announce Type: replace  Abstract: Long-term personalized memory for LLM agents is challenging on resource-limited edge devices because multimodal histories grow continually and are costly to store. To address this challenge, we propose ScrapMem, a bio-inspired framework that consolidates each day's heterogeneous records into a unified "Scrapbook Page." ScrapMem introduces Optical Forgetting, an optical compression mechanism that progressively reduces the resolution and JPEG quality of older pages, lowering storage costs while keeping recent evidence at higher fidelity. To preserve semantic consistency and support structured retrieval, we construct an Episodic Memory Graph (EM-Graph) that organizes semantic nodes into causal-temporal EM-Paths. This design separates a stable retrieval structure from aging visual evidence, allowing relevant episodes to remain accessible as page fidelity decreases. Under the common ATM-Bench protocol, ScrapMem without forgetting establis
    
[^322]: 一种基于扩散模型的尺度自适应时空联合超分辨率框架

    A Scale-Adaptive Framework for Joint Spatiotemporal Super-Resolution with Diffusion Models

    [https://arxiv.org/abs/2604.21903](https://arxiv.org/abs/2604.21903)

    该论文提出一种尺度自适应的时空联合超分辨率框架，将任务分解为条件均值的确定性预测与残差条件扩散模型，使同一架构可复用于不同的空间和时间超分辨率因子，并可通过质量守恒变换保持降水总量一致。

    

    深度学习视频超分辨率技术发展迅速，但在气候应用中，通常只对空间或时间进行超分辨率处理，而联合时空模型往往只针对单一的超分辨率（SR）因子组合（即低分辨率序列与高分辨率序列之间的空间上采样比例和时间比例）而设计，这限制了模型在不同空间分辨率和时间采样频率（帧率）之间的迁移能力。我们提出了一种尺度自适应框架，通过将时空超分辨率分解为带注意力机制的条件均值的确定性预测，以及一个残差条件扩散模型，并可选择加入质量守恒变换（确保输入与输出中的降水总量一致）以保持聚合总量，从而在不同因子间复用同一架构。我们假设较大的超分辨率因子主要增加的是欠定性（因而需要更多的上下文信息和残差不确定性），而非……

    arXiv:2604.21903v2 Announce Type: replace-cross  Abstract: Deep-learning video super-resolution has progressed rapidly, but climate applications typically super-resolve (increase resolution) either space or time, and joint spatiotemporal models are often designed for a single pair of super-resolution (SR) factors (upscaling spatial and temporal ratio between the low-resolution sequence and the high-resolution sequence), limiting transfer across spatial resolutions and temporal cadences (frame rates). We present a scale-adaptive framework that reuses the same architecture across factors by decomposing spatiotemporal SR into a deterministic prediction of the conditional mean, with attention, and a residual conditional diffusion model, with an optional mass-conservation (same precipitation amount in inputs and outputs) transform to preserve aggregated totals. Assuming that larger SR factors primarily increase underdetermination (hence required context and residual uncertainty) rather than
    
[^323]: 投影器也会遗忘：面向多模态持续指令微调的联合合并方法

    The Projector Forgets Too: Joint Merging for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2604.14016](https://arxiv.org/abs/2604.14016)

    该论文发现多模态持续指令微调中的灾难性遗忘不仅源于语言模型的LoRA适配器，还源于跨模态投影器，并提出JoMe方法通过联合合并这两部分的参数来有效缓解遗忘。

    

    多模态持续指令微调（MCIT）对于多模态大语言模型（MLLM）的序列任务适应至关重要，但受到灾难性遗忘的严重限制。现有文献主要关注推理所用的语言主干，而在本工作中，我们发现遗忘可能同时源于跨模态投影器和语言模型LoRA适配器这两个部分的变化。在组件恢复实验中，仅恢复较早任务的LoRA适配器而保留最终任务的投影器时，与同时恢复两个组件相比会产生精度差距。为了解决这两个遗忘来源，我们提出了JoMe（Joint Merging，联合合并）方法，它结合了跨模态投影合并（CPM）和低秩参数合并（LPM）。具体而言，CPM通过自适应合并跨模态视觉表示来恢复感知对齐……

    arXiv:2604.14016v2 Announce Type: replace-cross  Abstract: Multimodal Continual Instruction Tuning (MCIT) is essential for sequential task adaptation of Multimodal Large Language Models (MLLMs) but is severely restricted by catastrophic forgetting. While existing literature focuses on the reasoning language backbone, in this work, we find that forgetting can arise from changes in both the cross-modal projector and the language model's LoRA adapters. In component restoration experiments, restoring an earlier task's LoRA adapter while retaining the final-task projector can leave an accuracy gap relative to restoring both components. To address these two sources of forgetting, we present \textbf{JoMe} (\textbf{Jo}int \textbf{Me}rging), which combines \textbf{C}ross-modal \textbf{P}rojection \textbf{M}erging (\textbf{CPM}) and \textbf{L}ow-rank \textbf{P}arameter \textbf{M}erging (\textbf{LPM}). Specifically, CPM recovers perceptual alignment by adaptively merging cross-modal visual repres
    
[^324]: GAAMA：面向智能体的图增强联想记忆

    GAAMA: Graph Augmented Associative Memory for Agents

    [https://arxiv.org/abs/2603.27910](https://arxiv.org/abs/2603.27910)

    该论文提出GAAMA，一种围绕主题组织对话的图增强联想记忆框架，通过情节、原子事实、主题概念与高阶反思四类节点及五种结构边避免巨型枢纽问题，并结合最近邻搜索与个性化PageRank提升AI智能体长期记忆的检索效果。

    

    跨越多个会话与用户对话的AI智能体需要长期记忆来保持连贯性与个性化。现有方法要么采用扁平的检索增强生成（RAG），忽略了记忆之间的关联方式，要么采用以实体为中心的知识图谱。以实体为中心的图谱会将每条消息拆分为相互独立的三元组，从而丢失了上下文；而且在对话场景中，说话者会成为与几乎所有消息相连的枢纽节点，因此经由这些节点进行传播对检索帮助甚微。我们提出GAAMA，一种图增强的联想记忆，它围绕主题来组织对话。GAAMA保留逐字记录的情节片段，利用LLM将其提炼为原子事实和主题概念，进而合成更高阶的反思。这四种节点类型通过五种结构化的边相互连接，主题概念提供了横跨各处的路径，从而避免形成巨型枢纽。检索将最近邻相似度搜索与个性化PageRank相结合（原文摘要在此处截断）。

    arXiv:2603.27910v3 Announce Type: replace  Abstract: AI agents that talk with users across many sessions need long-term memory to stay coherent and personalized. Existing approaches use either flat retrieval-augmented generation (RAG) that ignores how memories connect, or entity-centric knowledge graphs. Entity-centric graphs split each message into separate triples, which loses its context, and in conversations the speakers become hubs linked to almost every message, so propagation through them tells retrieval little. We present GAAMA, a graph-augmented associative memory that organizes conversations around topics. It keeps verbatim episodes, uses an LLM to distill them into atomic facts and topic concepts, and then synthesizes higher-order reflections. These four node types are linked by five kinds of structural edges, and topic concepts supply the cross-cutting paths, so mega-hubs do not form. Retrieval blends nearest-neighbor similarity search with Personalized PageRank whose trans
    
[^325]: TopoPilot：面向拓扑数据分析与可视化的可靠对话式工作流自动化

    TopoPilot: Reliable Conversational Workflow Automation for Topological Data Analysis and Visualization

    [https://arxiv.org/abs/2603.25063](https://arxiv.org/abs/2603.25063)

    提出了TopoPilot——一个采用“编排器+验证器”双智能体架构的可靠智能体框架，用于自动化拓扑数据分析与可视化等复杂科学可视化工作流。

    

    近期的智能体系统表明，大型语言模型能够根据自然语言提示生成复杂的代码库。尽管取得了这一进展，但由于无效操作、隐蔽错误以及无法处理欠明确的输入，这些系统在科学可视化方面存在可靠性隐患。在超出标准基准复杂度的真实世界工作流中，这些挑战进一步加剧。在本工作中，我们提出了TopoPilot，一个可靠且可扩展的智能体框架，用于自动化复杂的科学可视化工作流。我们以拓扑数据分析与可视化作为主要应用场景，同时将该框架设计为可推广至更广泛的可视化领域。TopoPilot采用双智能体架构：编排器将用户提示转换为由原子后端操作组成的工作流，而验证器在工作流执行之前对其进行评估。其模块化设计将各组件相互隔离……

    arXiv:2603.25063v2 Announce Type: replace-cross  Abstract: Recent agentic systems demonstrate that large language models can generate complex codebases from natural language prompts. Despite this progress, they present reliability concerns for scientific visualization due to invalid operations, subtle errors, and failure to resolve underspecified inputs. These challenges are compounded in real-world workflows that exceed the complexity of standard benchmarks. In this work, we introduce TopoPilot, a reliable and extensible agentic framework for automating complex scientific visualization workflows. We focus on topological data analysis and visualization as a primary use case, while designing the framework to generalize to broader visualization domains. TopoPilot employs a two-agent architecture: an orchestrator translates user prompts into workflows composed of atomic backend actions, while a verifier evaluates workflows prior to execution. Its modular design isolates components and sup
    
[^326]: 这个梯度能带来喜悦吗？

    Does This Gradient Spark Joy?

    [https://arxiv.org/abs/2603.20526](https://arxiv.org/abs/2603.20526)

    论文提出愉悦策略梯度（DG），在前向传播中用优势与惊讶度的乘积“愉悦度”评估样本的学习价值，并通过“近藤门控”只为值得的样本执行昂贵的反向传播，从而以极少的计算实现更快的学习。

    

    策略梯度方法为每个样本都要执行一次反向传播，然而反向传播代价高昂，且大多数样本几乎没有学习价值。愉悦策略梯度在前向传播中就用“愉悦度”来评估样本的学习价值，愉悦度是优势与惊讶度（负对数概率）的乘积。我们提出了“近藤门控”，它将愉悦度与一个计算价格进行比较，只有当样本值得时才为其支付反向传播的开销；通过调节该价格可以刻画出质量与成本之间的权衡曲线。在MNIST上，即使反向传播完全免费，仅保留3%的梯度项也比标准策略梯度学习得更快，并且几乎能达到完整DG的效果。在老虎机任务中，零价格门控能够保持期望梯度方向并消除其垂直方向上的噪声；与价值和惊讶度的加性混合相比，愉悦度对样本的排序更加可靠；而具有高奖励方差的罕见动作则揭示了愉悦度失效的场景。在Transformer的词元反……（原文摘要在此截断）

    arXiv:2603.20526v2 Announce Type: replace-cross  Abstract: Policy gradient computes a backward pass for every sample, although the backward pass is expensive and most samples carry little learning value. The Delightful Policy Gradient (DG) scores learning value in the forward pass with \emph{delight}, the product of advantage and surprisal (negative log-probability). We introduce the \emph{Kondo gate}, which compares delight against a compute price and pays for a backward pass only when the sample is worth it; sweeping the price traces a quality--cost trade-off. On MNIST, keeping 3\% of gradient terms learns faster than standard policy gradient even when backward passes are free, and nearly matches full DG. In bandits, zero-price gating preserves the expected gradient direction and removes its perpendicular noise, delight ranks samples more reliably than additive mixes of value and surprise, and a rare action with high reward variance marks where delight fails. On transformer token rev
    
[^327]: 分布式摩擦下的愉悦策略梯度

    Delightful Policy Gradient under Distributed Friction

    [https://arxiv.org/abs/2603.20521](https://arxiv.org/abs/2603.20521)

    本文提出愉悦策略梯度（DG），通过用优势与惊讶度的乘积作为门控信号，在不依赖行为概率的情况下抑制分布式强化学习中罕见失败带来的噪声，同时保留罕见成功所携带的关键信息。

    

    分布式强化学习的训练数据来自过时、有缺陷或策略不匹配的执行器，这些执行器的动作在学习者策略下往往具有高惊讶度（即负对数概率）。监督微调能够从这类记录数据中稳定学习，而策略梯度方法通常不能。我们将这种差异追溯到来自惊讶数据的负学习效应：罕见的失败可能主导有限批次的更新，而罕见的成功则携带着当前策略本会错过的信息。愉悦策略梯度通过用“惊喜”（delight，即优势与惊讶度的乘积）对每一项进行门控来区分这两者，且无需行为概率。在一个受污染的老虎机问题中，DG 以一个随学习者能力提升而趋于消失的因子来抑制罕见失败的垂直噪声；任何仅基于学习者概率的门控都无法在不抑制罕见成功的情况下做到这一点。在带有过时执行器的 MNIST 任务上，DG 优于重要性加权的策略梯度方法（原文摘要在此处截断）。

    arXiv:2603.20521v3 Announce Type: replace-cross  Abstract: Distributed reinforcement learning trains on data from stale, buggy, or mismatched actors, whose actions often have high surprisal (negative log-probability) under the learner's policy. Supervised fine-tuning learns stably from such logged data, while policy gradient often does not. We trace the difference to negative learning from surprising data: rare failures can dominate finite-batch updates, while rare successes carry information the current policy would otherwise miss. The Delightful Policy Gradient (DG) separates the two by gating each term with delight, the product of advantage and surprisal, and needs no behavior probabilities. In a contaminated bandit, DG suppresses the perpendicular noise of rare failures by a factor that vanishes as the learner improves; no gate based only on learner probability can do this without also suppressing rare successes. On MNIST with stale actors, DG beats importance-weighted policy gradi
    
[^328]: 信息系统中的生成式人工智能图景：二次综述与研究议程的综合分析

    The Landscape of Generative AI in Information Systems: A Synthesis of Secondary Reviews and Research Agendas

    [https://arxiv.org/abs/2603.11842](https://arxiv.org/abs/2603.11842)

    本研究通过系统检索并综合分析28篇二次综述与路线图论文，梳理了生成式人工智能在信息系统领域带来的益处、挑战及未来研究议程。

    

    ChatGPT问世后的热潮迅速重塑了信息系统（IS）的研究与实践。随着组织和社会努力应对生成式人工智能（GenAI）的采纳问题，一批二次研究和研究议程应运而生，旨在综合早期证据并为未来研究指明方向。本研究通过综述二次研究和路线图类论文，综合梳理了GenAI在信息系统领域带来的益处与挑战方面的知识现状，并识别了未来的研究方向。我们在Scopus、Web of Science（WoS）和eAIS中系统检索了2023年以来的出版物。经过严格的多阶段筛选过程，最终选取了28篇论文，运用文献计量映射和主题分析方法进行分析。我们还对所有来源进行了质量评估，以衡量每个来源对研究结论贡献的可信度。GenAI具有变革性潜力，能够提升生产力、加速创新、实现服务个性化，并使专业知识获取大众化。

    arXiv:2603.11842v2 Announce Type: replace-cross  Abstract: The post-ChatGPT surge has rapidly reframed IS research and practice. As organizations and society grapple with GenAI adoption, a body of secondary studies and research agendas has emerged to synthesize early evidence and chart directions for future inquiry. This study reviews secondary and roadmap papers to synthesize the state of knowledge on GenAI's benefits and challenges in IS, and to identify future research directions. We performed a systematic search across Scopus, WoS, and eAIS for publications from 2023 onwards. Following a rigorous, multi-stage screening process, we selected a final set of 28 papers for analysis using bibliometric mapping and thematic analysis. We also conducted a quality assessment of all sources to gauge confidence in each source's contribution to the findings. GenAI offers transformative potential to drive productivity, accelerate innovation, personalize services, and democratize access to experti
    
[^329]: VeRA：用可执行规范革新推理基准测试

    VeRA: Renewing Reasoning Benchmarks with Executable Specifications

    [https://arxiv.org/abs/2602.13217](https://arxiv.org/abs/2602.13217)

    VeRA 将推理基准题目转化为由模板、输入生成器和确定性答案程序组成的可执行任务族，从而自动生成全新及更难的题目，揭示了大模型在固定题目上的高分与面对新题时的真实鲁棒性之间的显著差距。

    

    推理基准测试需要在两个维度上更新：新鲜度和提升空间。VeRA 通过将每个题目转化为一个任务族——包含一个自然语言模板、一个输入生成器和一个确定性的答案程序——使这两个维度变得可执行且可审计。VeRA-E 在任务族内抽取全新的实例；VeRA-H 对任务族进行修改以产生更难的任务；VeRA-H Pro 则从每个种子最多五个经过验证的提案中选出由评审模型排名最高的一个候选。执行检查、种子锚定、答案区分度以及独立的人工解题被用于验证规范和题目。被接受的程序可通过本地计算进一步生成带标签的实例。在 16 个模型上的实验显示，AIME-2024 的准确率从种子题目上的 84.46% 下降到 VeRA-E 变体上的 70.25%，暴露了固定题目上的成功与全新实例上的鲁棒性之间的差距。在 AIME-2024-II 上，经人工审计的 VeRA-H Pro 版本将准确率从 84.91% 降至 58.57%。在三种加难来源中，H Pro 具有最低……（原文截断）

    arXiv:2602.13217v2 Announce Type: replace  Abstract: Reasoning benchmarks need renewal along two axes: freshness and headroom. VeRA makes both executable and auditable by turning each item into a task family: a natural-language template, an input generator, and a deterministic answer program. VeRA-E draws fresh instances within a family; VeRA-H modifies the family toward harder tasks; and VeRA-H Pro selects one judge-ranked candidate from up to five validated proposals per seed. Execution checks, seed anchoring, answer discrimination, and independent human solving validate specifications and items. Accepted programs generate further labeled instances through local computation. Across 16 models, AIME-2024 accuracy decreases from 84.46% on seeds to 70.25% on VeRA-E variants, exposing a gap between fixed-item success and fresh-instance robustness. On AIME-2024-II, the human-audited VeRA-H Pro release lowers accuracy from 84.91% to 58.57%. Across the three hardening sources, H Pro has lowe
    
[^330]: EMA策略梯度：利用EMA锚点与Top-k KL驯服大语言模型的强化学习

    EMA Policy Gradient: Taming Reinforcement Learning for LLMs with EMA Anchor and Top-k KL

    [https://arxiv.org/abs/2602.04417](https://arxiv.org/abs/2602.04417)

    提出EMA锚点策略和Top-k KL估计器两项简单技术来改进大语言模型的策略梯度算法，结合GRPO后在数学推理和智能体强化学习任务上取得显著性能提升。

    

    强化学习（RL）已使大语言模型（LLM）能够掌握日益复杂的推理和智能体行为。在本工作中，我们提出两种简单的技术来改进面向大语言模型的策略梯度算法。首先，我们用指数移动平均（EMA）替换强化学习过程中的固定锚点策略，这类似于深度Q学习中的目标网络。其次，我们引入Top-$k$ KL估计器，它允许在精确KL与采样KL之间进行灵活插值。我们推导了使用EMA锚点的稳定性条件；此外，我们证明了Top-$k$ KL估计器在任意$k$值下都能产生无偏的KL值和无偏的梯度，同时兼具精确KL的优势。将这两种技术与GRPO相结合（EMA-PG）可带来显著的性能提升。在数学推理方面，它使R1蒸馏的Qwen-1.5B在OlympiadBench上达到53.9%，而GRPO为50.8%。在智能体强化学习领域，使用Qwen-3B……（摘要原文在此处截断）

    arXiv:2602.04417v2 Announce Type: replace-cross  Abstract: Reinforcement Learning (RL) has enabled Large Language Models (LLMs) to acquire increasingly complex reasoning and agentic behaviors. In this work, we propose two simple techniques to improve policy gradient algorithms for LLMs. First, we replace the fixed anchor policy during RL with an Exponential Moving Average (EMA), similar to a target network in deep Q-learning. Second, we introduce Top-$k$ KL estimator, which allows for flexible interpolation between exact KL and sampled KL. We derive the stability conditions for using EMA anchor; moreover, we show that our Top-$k$ KL estimator yields both unbiased KL values and unbiased gradients at any $k$, while bringing the benefits of exact KL. When combined with GRPO, the two techniques (EMA-PG) lead to a significant performance boost. On math reasoning, it allows R1-distilled Qwen-1.5B to reach 53.9% on OlympiadBench compared to 50.8% by GRPO. On agentic RL domains, with Qwen-3B b
    
[^331]: AgentArk：将多智能体智能蒸馏到单个大语言模型智能体中

    AgentArk: Distilling Multi-Agent Intelligence into a Single LLM Agent

    [https://arxiv.org/abs/2602.03955](https://arxiv.org/abs/2602.03955)

    AgentArk提出了一种新框架，通过推理增强微调、基于轨迹的增强和过程感知蒸馏三种分层蒸馏策略，将多智能体系统在迭代辩论中形成的智能蒸馏到单个LLM模型的权重中，使单个智能体兼具计算效率和多智能体级别的推理与自我纠错能力。

    

    尽管大语言模型（LLM）多智能体系统通过迭代辩论能够获得卓越的推理性能，但其实际部署受到高计算成本和误差传播的限制。本文提出了AgentArk，这是一种将多智能体动态蒸馏到单个模型权重中的新颖框架，有效地将显式的测试时交互转化为隐式的模型能力，使单个智能体在保持计算高效的同时具备多智能体系统的智能。具体而言，我们在各种模型、任务、扩展规模和场景中研究了三种分层蒸馏策略：推理增强微调、基于轨迹的增强以及过程感知蒸馏。通过将计算负担从推理阶段转移到训练阶段，蒸馏后的模型既保持了单个智能体的效率，又展现出多智能体系统强大的推理和自我纠错能力。

    arXiv:2602.03955v4 Announce Type: replace  Abstract: While large language model (LLM) multi-agent systems achieve superior reasoning performance through iterative debate, practical deployment is limited by their high computational cost and error propagation. This paper proposes AgentArk, a novel framework to distill multi-agent dynamics into the weights of a single model, effectively transforming explicit test-time interactions into implicit model capabilities. This equips a single agent with the intelligence of multi-agent systems while remaining computationally efficient. Specifically, we investigate three hierarchical distillation strategies across various models, tasks, scaling, and scenarios: reasoning-enhanced fine-tuning; trajectory-based augmentation; and process-aware distillation. By shifting the burden of computation from inference to training, the distilled models preserve the efficiency of one agent while exhibiting strong reasoning and self-correction performance of multi
    
[^332]: TabularMath：通过程序验证合成评估表格学习中的计算外推能力

    TabularMath: Evaluating Computational Extrapolation in Tabular Learning via Program-Verified Synthesis

    [https://arxiv.org/abs/2602.02523](https://arxiv.org/abs/2602.02523)

    本文提出TabularMath基准，将GSM8K和AIME问题编译为114个经程序验证的确定性计算，用以评估表格模型从统计插值到计算外推的能力，发现TabPFN v2.5在分布内中位数R²高达0.999，且是唯一在超出训练范围时中位数R²仍保持正值（0.76）的表格方法。

    

    标准的表格基准奖励的是插值，即预测数据流形内部的值，此时局部统计平滑是有效的。然而，许多高价值的表格数据，从定价和金融规则到物理模拟，都是由确定性计算而非噪声关系产生的。表格模型能否从统计插值走向计算外推，即使在训练中未见过的范围之外也能返回精确答案？我们提出了TabularMath，这是一个包含114个确定性计算（233,472行）的基准，这些计算由GSM8K和AIME问题编译为经过验证的程序。我们评估了九种表格方法、闭式解与符号回归基线，以及使用GPT-OSS-120B的上下文学习。在回归指标上，TabPFN v2.5表现极为出色：仅用128行数据，其分布内中位数R²达到0.999，并且当几乎每个答案超出训练范围时，它是唯一一个中位数R²仍保持为正（0.76）的表格方法。

    arXiv:2602.02523v2 Announce Type: replace-cross  Abstract: Standard tabular benchmarks reward interpolation: predicting values inside the data manifold, where local statistical smoothing works. However, much high-value tabular data, from pricing and financial rules to physical simulation, is produced by deterministic computations rather than noisy relationships. Can tabular models move from statistical interpolation to computational extrapolation, returning exact answers even beyond the range seen in training? We introduce TabularMath, a benchmark of 114 deterministic computations (233,472 rows) compiled from GSM8K and AIME problems into verified programs. We evaluate nine tabular methods, closed-form and symbolic regression baselines, and in-context learning with GPT-OSS-120B. On regression metrics, TabPFN v2.5 looks excellent: with 128 rows its median $R^2$ is 0.999 in-distribution, and it is the only tabular method whose median $R^2$ stays positive (0.76) when nearly every answer ex
    
[^333]: SAME：面向多模态持续指令微调的稳定专家混合模型

    SAME: Stabilized Mixture-of-Experts for Multimodal Continual Instruction Tuning

    [https://arxiv.org/abs/2602.01990](https://arxiv.org/abs/2602.01990)

    本文提出SAME框架，通过稳定专家路由和防止专家覆盖，解决了多模态持续指令微调中的路由漂移和专家漂移问题。

    

    多模态大语言模型（MLLMs）通过指令微调取得了强大性能，但在实际部署中需要持续扩展其能力，这使得多模态持续指令微调（MCIT）变得至关重要。近期方法利用稀疏专家路由来促进任务专业化，但我们发现，随着数据分布演变，专家路由过程会出现漂移。例如，一个接地（grounding）查询之前能激活定位专家，但在学习OCR任务后可能被路由到不相关的专家。同时，接地相关专家可能被新任务覆盖而失去原有功能。这种失败反映了两个问题：路由漂移，即专家选择随时间变得不一致；专家漂移，即共享专家在跨任务中被覆盖。因此，我们提出用于MCIT的稳定专家混合模型（SAME）。为解决路由漂移，SAME...

    arXiv:2602.01990v3 Announce Type: replace  Abstract: Multimodal Large Language Models (MLLMs) achieve strong performance through instruction tuning, but real-world deployment requires them to continually expand their capabilities, making Multimodal Continual Instruction Tuning (MCIT) essential. Recent methods leverage sparse expert routing to promote task specialization, but we find that the expert routing process suffers from drift as the data distribution evolves. For example, a grounding query that previously activated localization experts may instead be routed to irrelevant experts after learning OCR tasks. Meanwhile, the grounding-related experts can be overwritten by new tasks and lose their original functionality. Such failure reflects two problems: router drift, where expert selection becomes inconsistent over time, and expert drift, where shared experts are overwritten across tasks. Therefore, we propose StAbilized Mixture-of-Experts (SAME) for MCIT. To address router drift, S
    
[^334]: 学习与比较局部痕迹的弱监督图像篡改定位方法

    Learning and Comparing Local Traces for Weakly Supervised Image Manipulation Localization

    [https://arxiv.org/abs/2601.06222](https://arxiv.org/abs/2601.06222)

    提出一种基于CLIP的弱监督图像篡改定位框架，通过面向痕迹的视觉—语言自适应（TOVA）与关系感知痕迹判别（RATD）来学习和比较局部篡改痕迹，在无需像素级标注的情况下实现更准确完整的篡改区域定位。

    

    图像篡改定位旨在识别图像中被编辑的区域，并为评估图像可靠性提供空间证据，但全监督方法需要代价高昂的像素级标注。弱监督方法通过使用图像级标签降低了这一成本，但分类可能仅依赖少量显著线索，而无法保证定位的准确性和完整性。当篡改痕迹微弱且分布不均匀、真实区域与篡改区域外观相似时，这一难题尤为突出。我们提出了一种基于CLIP的框架，融合了面向痕迹的视觉—语言自适应（TOVA）和关系感知痕迹判别（RATD）两个模块。TOVA对视觉编码器进行自适应调整，并利用全局视觉上下文将边缘加权的图像块特征聚合为以图像为条件的类别提示。由此得到的类别表示支持图像级对齐，并指导层次化痕迹对比学习中的图像块选择。

    arXiv:2601.06222v2 Announce Type: replace-cross  Abstract: Image manipulation localization identifies edited regions and provides spatial evidence for assessing image reliability, but fully supervised methods require costly pixel-level annotations. Weakly supervised methods reduce this cost using image-level labels, but classification can rely on a few salient cues without ensuring accurate or complete localization. This difficulty is pronounced when manipulation traces are weak and unevenly distributed and authentic and manipulated regions share similar appearance. We propose a CLIP-based framework that integrates Trace-Oriented Vision--Language Adaptation (TOVA) and Relation-Aware Trace Discrimination (RATD). TOVA adapts the visual encoder and uses global visual context to aggregate edge-weighted patch features into image-conditioned class prompts. The resulting class representations support image-level alignment and guide patch selection for Hierarchical Trace Contrastive Learning, 
    
[^335]: PsyAgent：基于心理建模与情境交互的类人智能体构建

    PsyAgent: Constructing Human-like Agents Based on Psychological Modeling and Contextual Interaction

    [https://arxiv.org/abs/2601.06158](https://arxiv.org/abs/2601.06158)

    PsyAgent提出了一种“模式优先”框架，将大五人格特质画像（IS）与多情境化框架（MSC）耦合，并利用两者组合合成监督数据微调紧凑模型，使类人智能体的行为既保持人格稳定又能随情境灵活变化。

    

    类人智能体必须在表达稳定人格特质的同时适应角色、关系与规范。我们提出PsyAgent，这是一个“模式优先”（schema-first）框架，通过将大五人格特质先验与显式的社会结构条件化相结合，实现了特质—情境接口的可操作化。PsyAgent包含两个部分：(i) 个体结构，一个机器可用的、以特质为基础的画像；(ii) 多情境化，一个涵盖八个日常生活领域的角色—关系—规范框架精选库。在推理阶段，固定的结构化提示将激活的MSC框架与IS画像相结合，从而促使智能体的行为既保持稳定又对情境敏感。为了证明其可学习性超越了单纯的提示工程，我们利用IS×MSC合成监督数据，并采用参数高效微调技术（PEFT，包括SFT及可选的DPO）对紧凑型骨干模型进行微调。在百分位空间下受控的心理测量式评估协议中，PsyAgent提升了特质忠实度与多轮人设一致性（原文摘要在此处截断）。

    arXiv:2601.06158v3 Announce Type: replace  Abstract: Human-like agents must express stable dispositions while adapting to roles, relationships, and norms. We present PsyAgent, a schema-first framework that operationalizes the trait-context interface by coupling a Big Five trait prior with explicit social-structural conditioning. PsyAgent comprises (i) Individual Structure (IS), a machine-usable trait-grounded profile, and (ii) Multi-Scenario Contexting (MSC), a curated library of role-relationship-norm frames spanning eight everyday arenas. At inference, fixed structured prompts couple the active MSC frame with the IS profile, encouraging behavior that is stable yet context-sensitive. To demonstrate learnability beyond prompt engineering, we use IS x MSC to synthesize supervision and fine-tune compact backbones with PEFT (SFT and optional DPO). Under a controlled psychometric-style evaluation protocol in percentile space, PsyAgent improves trait-faithfulness and multi-turn persona cons
    
[^336]: 先验证再生成：面向视频生成的草图引导高效运动规划

    Verify Before You Generate: Sketch-Guided Efficient Motion-Planning for Video Generation

    [https://arxiv.org/abs/2511.17450](https://arxiv.org/abs/2511.17450)

    提出了 SketchVerify，一个无需训练的框架，在调用昂贵的视频生成器之前先通过轻量级视频草图验证运动规划，从而高效比较并迭代精化多条候选轨迹，以低成本获得高质量运动。

    

    近期的视频生成方法越来越依赖规划中间控制信号（如物体轨迹）来提升时间一致性和运动保真度。然而，这些方法大多采用单次规划，通常局限于简单的运动指令，导致运动质量欠佳；或者采用迭代精化，需要多次调用视频生成器并产生高昂的计算成本。为解决这一权衡问题，我们提出了 SketchVerify，这是一个无需训练的框架，其核心洞察是在调用昂贵的视频生成器之前，先使用轻量级视频草图来验证运动规划。这一轻量级规划空间使得能够在最终的单次视频生成之前，高效比较多条候选轨迹并进行有界的迭代精化。给定提示词和参考图像，SketchVerify 采样多条候选轨迹，并将每条轨迹渲染为轻量级视频……

    arXiv:2511.17450v2 Announce Type: replace-cross  Abstract: Recent video generation approaches increasingly rely on planning intermediate control signals like object trajectories to improve temporal coherence and motion fidelity. However, these methods mostly employ single-shot plans that are typically limited to simple motion instructions, leading to suboptimal motion quality, or iterative refinement which requires multiple calls to the video generator and incurs high computational cost. To address this trade-off, we propose SketchVerify, a training-free framework whose key insight is to verify motion plans using lightweight video sketches before invoking the expensive video generator. This lightweight planning space enables efficient comparison of multiple candidate trajectories and bounded iterative refinement before a single final video-generation pass. Given a prompt and a reference image, SketchVerify samples multiple candidate trajectories and renders each as a lightweight video 
    
[^337]: 面向因果感知强化学习的以对象为中心的世界模型

    Object-Centric World Models for Causality-Aware Reinforcement Learning

    [https://arxiv.org/abs/2511.14262](https://arxiv.org/abs/2511.14262)

    该论文提出STICA统一框架，利用以对象为中心的Transformer作为世界模型，并配合因果感知的策略与价值网络，使强化学习智能体能够像人类一样分解复杂多对象环境，从而实现更高效的样本学习与决策。

    

    世界模型已被开发用于支持样本高效的深度强化学习智能体。然而，由于大多数世界模型学习的是所有环境组件的整体表示，世界模型难以准确复现那些高维、非平稳、且由具有丰富交互的多个对象组成的环境。相比之下，人类通过将环境分解为离散的对象来感知环境，从而实现高效决策。受这一洞察的启发，我们提出了STICA（Slot Transformer Imagination with CAusality-aware reinforcement learning，基于槽位Transformer想象与因果感知强化学习），这是一个统一框架，其中以对象为中心的Transformer既充当世界模型，也充当因果感知的策略网络与价值网络。STICA将每个观测表示为一组以对象为中心的token，同时包含智能体动作及其产生的奖励的token，使世界模型能够在token级别上预测环境动态……

    arXiv:2511.14262v4 Announce Type: replace-cross  Abstract: World models have been developed to support sample-efficient deep reinforcement learning agents. However, it remains challenging for world models to accurately replicate environments that are high-dimensional, non-stationary, and composed of multiple objects with rich interactions since most world models learn holistic representations of all environmental components. By contrast, humans perceive the environment by decomposing it into discrete objects, facilitating efficient decision-making. Motivated by this insight, we propose \emph{Slot Transformer Imagination with CAusality-aware reinforcement learning} (STICA), a unified framework in which object-centric Transformers serve as the world model and causality-aware policy and value networks. STICA represents each observation as a set of object-centric tokens, together with tokens for the agent action and the resulting reward, enabling the world model to predict token-level dyna
    
[^338]: 将测试时计算最优扩展泛化为可优化的图

    Generalizing Test-time Compute-optimal Scaling as an Optimizable Graph

    [https://arxiv.org/abs/2511.00086](https://arxiv.org/abs/2511.00086)

    该论文提出将测试时扩展中在固定预算下搜索计算最优模型组合与架构的新问题形式化为多LLM协作图，并将其重新表述为概率图优化问题，以应对庞大的组合搜索空间和任务定制化需求。

    

    测试时扩展通过在推理过程中分配额外的计算来提升大型语言模型（LLM）的性能，通常采用并行、串行或混合扩展方式。然而，先前的研究往往假设固定的协作架构（例如拓扑结构）以及单一模型的使用，忽略了最优架构和模型组合可能因任务不同而变化。因此，我们研究了一个新问题：在固定预算下搜索测试时扩展中计算最优的模型组合与架构。我们将其形式化为一个多LLM协作图，其中节点编码角色和LLM模型分配，边则捕捉信息流。这个问题具有挑战性，原因在于： 组合搜索空间极其庞大， 任务特定需求需要定制化的设计。为解决这些问题，我们将该问题重新表述为概率图优化，并通过初步实验得出了三条经验性……（原文摘要在此处截断）

    arXiv:2511.00086v2 Announce Type: replace-cross  Abstract: Test-Time Scaling (TTS) improves large language models (LLMs) by allocating additional computation during inference, typically through parallel, sequential, or hybrid scaling. However, prior studies often assume fixed collaboration architectures (e.g., topologies) and single-model usage, overlooking that optimal architectures and model combinations can vary across tasks. Therefore, we study the novel problem of searching for compute-optimal model combinations and architectures in TTS under a fixed budget. We formalize it as a multi-LLM collaboration graph, where nodes encode roles and LLM model assignments, and edges capture information flow. This problem is challenging because (i) the combinatorial search space is prohibitively large, and (ii) task-specific requirements demand tailored designs. To address these, we reformulate the problem as probabilistic graph optimization and, through pilot experiments, derive three empirica
    
[^339]: 具有可靠性保证的可证明选择性自动标注

    Provable Selective Auto-labeling with Reliability Guarantees

    [https://arxiv.org/abs/2510.14581](https://arxiv.org/abs/2510.14581)

    提出了“共形标注”这一新方法，将选择性自动标注形式化为多重假设检验问题，通过构建共形p值并结合Benjamini-Hochberg程序，在错误标注率可证明受控的前提下筛选出可信的自动生成标签子集。

    

    自动标注已成为一种流行且经济高效的人工标注替代方案。然而，其性能在不同任务间往往存在高度不一致性，这凸显了选择性策略的重要性。尽管如此，现有的选择性自动标注启发式方法仍然严重依赖模型置信度分数，并且对所选样本的可信度无法提供可靠的保证。为解决这一问题，我们提出了共形标注，这是一种新颖的方法，能够选择一个自动生成标签的子集，并对其错误标注率进行可证明的控制。我们的核心思想是将选择性自动标注形式化为一个多重假设检验问题，其中每个假设表示是否接受某个实例的自动生成标签。具体而言，我们使用小型校准集为每个自动生成的标签构建共形p值，然后采用Benjamini-Hochberg（BH）程序……

    arXiv:2510.14581v4 Announce Type: replace-cross  Abstract: Auto-labeling has emerged as a popular, cost-effective alternative to manual annotation. However, its performance is often highly inconsistent across different tasks, underscoring the importance of selective strategies. Despite this, existing heuristic methods for selective auto-labeling still rely heavily on model confidence scores and offer no reliable guarantee on the trustworthiness of the selected examples. To address this, we propose \textbf{Conformal Labeling}, a novel method that selects a subset of auto-generated labels with a provably controlled false labeling rate (FLR). Our key idea is to formulate selective auto-labeling as a multiple-hypothesis testing problem, where each hypothesis indicates whether to accept the auto-generated label for an instance. Specifically, we construct a conformal $p$-value for each auto-generated label using a small calibration set, and then employ the Benjamini--Hochberg (BH) procedure 
    
[^340]: 学习同化：自回归扩散的摊销式控制

    Learning to Assimilate: Amortized Control for Autoregressive Diffusion

    [https://arxiv.org/abs/2510.06637](https://arxiv.org/abs/2510.06637)

    提出了 CADA 方法，在冻结的自回归扩散模型之上训练轻量级控制器实现摊销式数据同化，无需逐次预测进行优化即可提升稀疏观测下的长程预测精度，并将推理速度提高 10 倍以上。

    

    自回归扩散模型（ARDMs）为混沌动力学的预测提供了富有表现力的先验。然而，在长程滚动预测过程中保持与稀疏观测的一致性可能需要大量的测试时计算。我们提出了控制增强数据同化（Control-Augmented Data Assimilation, CADA），该方法在一个冻结的 ARDM 之上训练一个轻量级控制器，将预测引导向未来的观测值。通过使用 KL 正则化的随机最优控制目标，控制器从较短的自回归滚动中学习，在预测观测的同时保持贴近预训练的动力学。在推理阶段，它在每个去噪步骤注入校正，无需针对每次预测进行单独优化。在两个混沌 PDE 基准测试上，CADA 在稀疏观测条件下提升了长程预测的精度、稳定性和物理保真度。在紧凑的 ERA5 数据集上，它在大多数情形下优于基于扩散的基线方法，且推理速度比条件扩散模型快 10 倍以上。

    arXiv:2510.06637v4 Announce Type: replace-cross  Abstract: Autoregressive diffusion models (ARDMs) offer expressive priors for forecasting chaotic dynamics. However, maintaining consistency with sparse observations over long rollouts can require substantial test-time computation. We introduce Control-Augmented Data Assimilation (CADA), which trains a lightweight controller on top of a frozen ARDM to steer predictions toward future observations. Using a KL-regularized stochastic optimal control objective, the controller learns from short autoregressive rollouts to anticipate observations while staying close to the pretrained dynamics. At inference, it injects corrections at each denoising step without per-forecast optimization. Across two chaotic PDE benchmarks, CADA improves long-horizon accuracy, stability, and physical fidelity under sparse observations. On compact ERA5, it outperforms diffusion-based baselines in most regimes with over $10\times$ faster inference than conditional di
    
[^341]: RAG让防护栏变得脆弱？研究RAG式上下文下安全防护栏的鲁棒性

    RAG Makes Guardrails Brittle? Investigating Robustness of Guardrails under RAG-style Contexts

    [https://arxiv.org/abs/2510.05310](https://arxiv.org/abs/2510.05310)

    本研究通过系统评估发现，RAG场景中仅插入良性的检索文档就能使基于LLM的安全防护栏在约10%的情况下翻转判断，且多数翻转会将原本正确的安全决策反转为错误决策，揭示了现有防护栏在RAG式上下文下的脆弱性。

    

    随着大语言模型（LLM）的日益普及，确保LLM系统的安全性已成为一个紧迫的问题。基于LLM的外部防护栏模型已成为筛选不安全输入和输出的流行解决方案，但它们本身是经过微调或提示工程设计的LLM，容易受到数据分布偏移的影响。本文以检索增强生成（RAG）为案例研究，考察基于LLM的防护栏对嵌入上下文中额外信息的鲁棒性。通过对3个Llama Guard模型和2个GPT-oss模型的系统性评估，我们发现仅仅插入良性的检索文档，就会使输入和输出防护栏的判断分别在大约11%和8%的情况下发生翻转，其中大多数翻转（72%和64%）是将原本正确的判断反转为错误判断。我们进一步分析了增强上下文中各组件（检索文档、用户查询和LLM）的影响。

    arXiv:2510.05310v2 Announce Type: replace-cross  Abstract: With the increasing adoption of large language models (LLMs), ensuring the safety of LLM systems has become a pressing concern. External LLM-based guardrail models have emerged as a popular solution to screen unsafe inputs and outputs, but they are themselves fine-tuned or prompt-engineered LLMs vulnerable to data distribution shifts. In this paper, taking Retrieval Augmentation Generation (RAG) as a case study, we investigate how robust LLM-based guardrails are against additional information embedded in the context. Through a systematic evaluation of 3 Llama Guards and 2 GPT-oss models, we find that inserting benign retrieved documents flips the judgments of input and output guardrails in around 11% and 8% of cases respectively, with the majority of these flips (72% and 64%) reversing previously correct decisions. We further analyze the effect of each component in the augmented context: retrieved documents, user query, and LLM
    
[^342]: 利用组相似度奖励提升检索增强系统的一致性

    Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards

    [https://arxiv.org/abs/2510.04392](https://arxiv.org/abs/2510.04392)

    本文提出PS-GRPO强化学习方法，通过跨复述集的组相似度奖励来提升RAG系统在语义等价查询下的信息一致性，并提供了将一致性分解为检索器、生成器和端到端层面的系统性评估框架。

    

    检索增强生成（RAG）系统越来越多地被部署在高风险领域，用户期望其输出在语义等价的查询之间保持一致。然而，由于检索器和生成器双方均存在变异性，现有系统常常表现出显著的不一致性，损害了信任与可靠性。在这项工作中，我们关注信息一致性，即生成的输出在语义等价的输入之间应传达相同核心内容的要求。我们引入了一个有原则性的评估框架，将RAG一致性分解为检索器级别、生成器级别和端到端组件，从而能够系统地识别不一致性的来源。为了提升一致性，我们提出了复述集组相对策略优化（PS-GRPO），这是一种强化学习方法，通过在复述集上进行多次rollout来分配组相似度奖励。我们利用PS-GRPO来实现

    arXiv:2510.04392v2 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) systems are increasingly deployed in high-stakes domains where users expect outputs to be consistent across semantically equivalent queries. However, existing systems often exhibit significant inconsistencies due to variability in both the retriever and generator, undermining trust and reliability. In this work, we focus on information consistency, i.e., the requirement that generated outputs convey the same core content across semantically equivalent inputs. We introduce a principled evaluation framework that decomposes RAG consistency into retriever-level, generator-level, and end-to-end components, enabling systematic identification of inconsistency sources. To improve consistency, we propose Paraphrased Set Group Relative Policy Optimization (PS-GRPO), an RL approach that leverages multiple rollouts across paraphrased set to assign group similarity rewards. We leverage PS-GRPO to achieve
    
[^343]: Rhythm-TTS：面向零样本文本转语音的高保真捷径潜变量建模与概率化节奏生成

    Rhythm-TTS: High-Fidelity Shortcut Latent Modeling and Probabilistic Rhythm Generation for Zero-Shot Text-to-Speech

    [https://arxiv.org/abs/2510.02848](https://arxiv.org/abs/2510.02848)

    Rhythm-TTS 提出了一种捷径零样本文本转语音框架，将预测的离散编解码器标记作为语义-声学先验，直接生成高保真的整体连续语音潜变量，并结合概率化节奏生成，从而兼顾离散标记的可因子分解性与连续表示的高保真度。

    

    零样本文本转语音（TTS）系统通常建立在离散编解码器标记（codec tokens）或连续声学表示之上。离散标记模型受益于稳定的序列建模和显式的语音因子分解，但其最终通过码本查找和码字组合来合成语音，这会累积误差并对自然度和说话人相似度造成质量上限。连续表示模型可以直接生成细粒度的语音潜变量，但往往需要更繁重的上下文建模且对齐不够显式。本文提出 Rhythm-TTS，一种捷径（shortcut）零样本文本转语音框架，它将离散编解码器标记的可因子分解性与连续潜变量生成的高保真度相结合。Rhythm-TTS 不把预测出的编解码器标记直接作为最终声学表示，而是将其用作语义-声学先验，并直接生成整体的连续语音潜变量……（摘要原文在此处截断）

    arXiv:2510.02848v2 Announce Type: replace-cross  Abstract: Zero-shot text-to-speech (TTS) systems are commonly built on either discrete codec tokens or continuous acoustic representations. Discrete-token models benefit from stable sequence modeling and explicit speech factorization, but they ultimately synthesize speech through codebook lookup and codeword composition, which can accumulate errors and impose a quality ceiling on naturalness and speaker similarity. Continuous-representation models can generate fine-grained speech latents directly, but often require heavier in-context modeling and less explicit alignment. This paper presents \textbf{Rhythm-TTS}, a shortcut zero-shot TTS framework that combines the factorizability of discrete codec tokens with the fidelity of continuous latent generation. Instead of using predicted codec tokens as the final acoustic representation, Rhythm-TTS uses them as a semantic-acoustic prior and directly generates holistic continuous speech latents w
    
[^344]: 重建对齐改进了统一多模态模型

    Reconstruction Alignment Improves Unified Multimodal Models

    [https://arxiv.org/abs/2509.07295](https://arxiv.org/abs/2509.07295)

    提出一种名为RECA的后训练方法，利用视觉理解编码器嵌入作为密集监督信号，无需文本标题即可重新对齐统一多模态模型的理解与生成，从而显著提升图像生成和编辑的保真度。

    

    arXiv:2509.07295v4 公告类型：替换交叉 摘要：统一多模态模型（UMMs）在单一架构中统一了视觉理解与生成。然而，传统的训练依赖于图像-文本对（或序列），其标题通常稀疏且缺失细粒度的视觉细节，即使它们用数百个单词描述一张简单的图像。我们引入了重建对齐（RECA），一种资源高效的后训练方法，利用视觉理解编码器的嵌入作为密集的“文本提示”，在无需标题的情况下提供丰富的监督。具体而言，RECA将UMM以其自身的视觉理解嵌入为条件，并通过自监督重建损失优化其重建输入图像，从而重新对齐理解与生成。尽管方法简单，但RECA具有广泛适用性：在自回归、掩码自回归和基于扩散的UMMs中，它一致地提升了生成和编辑的保真度。仅需2（此处指训练数据或步骤的少量性，但原文不完整，按常规翻译为“仅需2”以保留原意）。

    arXiv:2509.07295v4 Announce Type: replace-cross  Abstract: Unified multimodal models (UMMs) unify visual understanding and generation within a single architecture. However, conventional training relies on image-text pairs (or sequences) whose captions are typically sparse and miss fine-grained visual details, even when they use hundreds of words to describe a simple image. We introduce Reconstruction Alignment (RECA), a resource-efficient post-training method that leverages visual understanding encoder embeddings as dense "text prompts", providing rich supervision without captions. Concretely, RECA conditions a UMM on its own visual understanding embeddings and optimizes it to reconstruct the input image with a self-supervised reconstruction loss, thereby realigning understanding and generation. Despite its simplicity, RECA is broadly applicable: across autoregressive, masked-autoregressive, and diffusion-based UMMs, it consistently improves generation and editing fidelity. With only 2
    
[^345]: 超越输出忠实性：诊断与缓解特征归因中的通路重路由

    Beyond Output Faithfulness: Diagnosing and Mitigating Pathway Rerouting in Feature Attribution

    [https://arxiv.org/abs/2509.04588](https://arxiv.org/abs/2509.04588)

    论文发现插入/删除等仅输出指标无法检测特征归因是否经由支持原始预测的计算路径，提出了可微的EQO方法揭示“通路重路由”现象，并引入逐层梯度过滤器ClipGrad来缓解该问题。

    

    插入和删除是通过逐步添加或移除像素来评估特征归因的标准指标。虽然它们能测试被归因的特征是否控制模型输出，但并未验证这些特征是否通过支持原始（干净）预测的计算路径来实现这一控制。我们使用集成分位数优化来研究这一区别，这是一种直接优化插入序列的可微方法。在CNN上，无约束EQO在偏离干净表示的同时取得了最高的插入分数，并且在消融支持干净预测的通道后仍保持其插入分数。这些结果共同揭示了一种仅靠输出指标无法检测到的“通路重路由”现象。在Vision Transformer上，来自黑色输入的目标预测和注意力偏差同样暴露了这一失败。为解决该问题，我们提出了ClipGrad，一种逐层梯度过滤器，它保留……（摘要在此处截断）

    arXiv:2509.04588v3 Announce Type: replace-cross  Abstract: Insertion and deletion are standard metrics for evaluating feature   attributions by progressively adding or removing pixels. While they test   whether attributed features control a model output, they do not verify   whether they do so through the computation that supports the clean   prediction. We study this distinction with Ensemble Quantile Optimization (EQO),   a differentiable method that directly optimizes the insertion sequence.   On CNNs, Unconstrained EQO attains the highest insertion scores while departing   from clean representations and retaining its insertion score after channels   supporting the clean prediction are ablated. Together, these results reveal   pathway rerouting that output-only metrics cannot detect. On a Vision   Transformer, target predictions from black inputs and attention deviation   expose the same failure.   To address this, we introduce ClipGrad, a layer-wise gradient   filter that preserves
    
[^346]: 基于群体相对策略优化的下一尺度视觉自回归模型微调

    Fine-Tuning Next-Scale Visual Autoregressive Models with Group Relative Policy Optimization

    [https://arxiv.org/abs/2505.23331](https://arxiv.org/abs/2505.23331)

    本文将群体相对策略优化（GRPO）应用于下一尺度视觉自回归（VAR）模型的强化学习微调，通过对齐美学与CLIP奖励信号显著提升图像质量与风格控制能力，并使模型能够泛化生成预训练阶段未见过的图像风格。

    

    使用强化学习（RL）对预训练生成模型进行微调已成为使输出更贴近细致人类偏好的一种有效方法。本文研究了将群体相对策略优化（GRPO）应用于微调下一尺度视觉自回归（VAR）模型。我们的实证结果表明，该方法能够对齐来自美学预测器和CLIP嵌入的复杂奖励信号，显著提升图像质量，并实现对生成风格的精确控制。有趣的是，通过利用CLIP，我们的方法可以帮助VAR模型泛化到其初始ImageNet分布之外：通过强化学习驱动的探索，这些模型能够生成与提示词相匹配的图像，其中涉及的图像风格在预训练阶段并未出现。总之，我们证明了基于强化学习的微调对VAR模型而言既高效又有效。

    arXiv:2505.23331v3 Announce Type: replace-cross  Abstract: Fine-tuning pre-trained generative models with Reinforcement Learning (RL) has emerged as an effective approach for aligning outputs more closely with nuanced human preferences. In this paper, we investigate the application of Group Relative Policy Optimization (GRPO) to fine-tune next-scale visual autoregressive (VAR) models. Our empirical results demonstrate that this approach enables alignment to intricate reward signals derived from aesthetic predictors and CLIP embeddings, significantly enhancing image quality and enabling precise control over the generation style. Interestingly, by leveraging CLIP, our method can help VAR models generalize beyond their initial ImageNet distribution: through RL-driven exploration, these models can generate images aligned with prompts referencing image styles that were absent during pre-training. In summary, we show that RL-based fine-tuning is both efficient and effective for VAR models, b
    
[^347]: LoopNav：世界模型中空间一致性的基准测试

    LoopNav: Benchmarking Spatial Consistency in World Models

    [https://arxiv.org/abs/2505.22976](https://arxiv.org/abs/2505.22976)

    提出LoopNav数据集与基准，通过基于环路的导航任务来评估世界模型的长距离空间一致性能力。

    

    以空间一致的方式模拟世界的能力是有效世界模型的关键要求。这样的模型不仅能够实现高质量的视觉生成，还能确保世界模型在仿真和规划等下游任务中的可靠性。它不仅要保留长时程的观测信息，还需要能够构建显式或隐式的内部空间表示。然而，现有数据集并未显式施加空间一致性约束，这既限制了系统评估该能力的可能，也限制了通过数据驱动方法学习该能力的途径。此外，大多数现有基准主要强调视觉连贯性或生成质量，忽视了长距离空间一致性的要求。为弥补这一空白，我们提出了LoopNav，这是一个以基于环路的导航为核心的数据集及相应基准，用于评估空间一致性。

    arXiv:2505.22976v4 Announce Type: replace-cross  Abstract: The ability to simulate the world in a spatially consistent manner is a crucial requirement for effective world models. Such a model enables high-quality visual generation, and also ensures the reliability of world models for downstream tasks such as simulation and planning. It must not only retain long-horizon observational information, but also enables the construction of explicit or implicit internal spatial representations. However, existing datasets do not explicitly enforce spatial consistency constraints, limiting both the ability to systematically evaluate this capability and to learn it through data-driven approaches. Furthermore, most existing benchmarks primarily emphasize visual coherence or generation quality, neglecting the requirement of long-range spatial consistency. To bridge this gap, we propose LoopNav, a dataset and corresponding benchmark centered on loop-based navigation for evaluating spatial consistency
    
[^348]: 因果性可以系统性地解决潜藏在基准测试之下的“怪兽”

    Causality can systematically address the monsters under the bench(marks)

    [https://arxiv.org/abs/2502.05085](https://arxiv.org/abs/2502.05085)

    这篇立场论文主张，因果性为系统性地解决机器学习基准测试中的偏差、伪影和数据泄露等“怪兽”问题提供了理想框架，通过显式化因果假设并利用有原则的分析工具来提升评估的可信度与一致性。

    

    有效且可靠的评估对于推进实证机器学习至关重要。然而，通用大模型日益普及，以及朝着日益复杂的高层次任务的不断推进，使得系统性评估变得更具挑战性。基准测试受到各种偏差、伪影或数据泄露等问题的困扰，同时由于故障模式未被充分探索，模型可能表现出不可靠的行为。对这类“怪兽”的随意处理和不一致的表述导致了重复劳动、对结果缺乏信任以及缺乏依据的推断。在这篇立场论文中，我们主张因果性为系统性地应对这些挑战提供了一个理想的框架。通过将方法中的因果假设显式化，我们可以忠实地对现象建模、构建具有解释力的可检验假设，并利用有原则性的工具进行分析。为了使概念性因果模型的设计更易于上手，我们识别了几种有用的……

    arXiv:2502.05085v2 Announce Type: replace-cross  Abstract: Effective and reliable evaluation is essential for advancing empirical machine learning. However, the increasing accessibility of generalist models and the progress towards ever more complex, high-level tasks make systematic evaluation more challenging. Benchmarks are plagued by various biases, artifacts, or leakage, while models may behave unreliably due to poorly explored failure modes. Haphazard treatments and inconsistent formulations of such "monsters" leads to a duplication of efforts, a lack of trust in results, and unsupported inferences. In this position paper, we argue causality offers an ideal framework to systematically address these challenges. By making causal assumptions in an approach explicit, we can faithfully model phenomena, formulate testable hypotheses with explanatory power, and leverage principled tools for analysis. To make the design of conceptual causal models more accessible, we identify several usef
    
[^349]: 基于不确定性感知模仿学习的鲁棒响应式机器人操作上下文自适应策略框架

    A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning

    [https://arxiv.org/abs/2410.24035](https://arxiv.org/abs/2410.24035)

    该论文提出了一个基于不确定性感知模仿学习的上下文自适应策略框架，通过策略融合与不确定性量化技术，首次将任务参数化、鲁棒性和响应式机器人操作三者结合，解决了现有动力系统方法无法根据任务相关变量灵活调节策略的问题。

    

    在机器人学中，生成能够适应不断变化的上下文信息的鲁棒且具有响应性的操作策略是一项具有挑战性的任务。多年来，从演示中学习已成为生成响应式策略的一种直观且有效的解决方案，特别是基于动力系统的方法。然而，大多数最先进的基于动力系统的方法专注于解决鲁棒性方面的限制，而忽略了策略对环境变化的调节。因此，这些方法在利用任务相关变量进行参数化时往往缺乏灵活性。在这项工作中，我们在现有的策略融合和不确定性量化研究的基础上，提出了一个上下文自适应策略框架，将任务参数化、鲁棒性和响应式操作相结合。为此，我们使用从演示中学习的方法来获取一个以机器人状态和低维任务相关参数为条件的策略。

    arXiv:2410.24035v3 Announce Type: replace-cross  Abstract: Generating robust and reactive manipulation strategies that can adapt to changing context information is a challenging task in robotics. Over the years, Learning from Demonstration (LfD) has emerged as an intuitive and effective solution for generating reactive policies, particularly by following dynamical-system(DS)-based approaches. However, most state-of-the-art DS-based approaches focus on addressing the robustness limitations, overlooking the modulation of policies in response to the environment. As a result, they tend to be inflexible with respect to parameterization by task-dependent variables. In this work, we build on existing work on policy fusion and uncertainty quantification to propose a context-adaptive policy framework that combines task-parameterized, robust and reactive manipulation. For this, we use LfD to acquire a policy that is conditioned on the robot state and low-dimensional task-dependent parameters ref
    
[^350]: LLM生成代码的自动检测：当代模型在函数与类粒度上的对比案例研究

    Automatic Detection of LLM-Generated Code: A Comparative Case Study of Contemporary Models Across Function and Class Granularities

    [https://arxiv.org/abs/2409.01382](https://arxiv.org/abs/2409.01382)

    该研究通过对比四种LLM生成的14,485个函数和11,913个类，发现代码检测中粒度效应的影响远超模型间差异（达8.6倍），函数级与类级检测依赖截然不同的结构特征，且存在依赖粒度的检测性能反转现象。

    

    arXiv:2409.01382v3 公告类型： replace-cross 摘要：采用大型语言模型（LLM）进行代码生成存在将易受攻击的代码引入软件系统的风险。现有检测器面临两个关键局限：缺乏系统性的跨模型验证以及不透明的“黑箱”操作。我们通过对四种不同LLM（GPT-3.5、Claude 3 Haiku、Claude Haiku 4.5和GPT-OSS）生成的代码进行对比研究来解决这些问题。通过分析来自CodeSearchNet数据集的14,485个Python函数和11,913个类，我们使用全部四种LLM生成了相应的代码，并利用可解释的软件度量指标为每种配置训练了CatBoost分类器。我们的分析表明，粒度效应在主导模型差异方面高出8.6倍，且特征重叠可忽略不计，这表明函数级检测和类级检测依赖于本质上互不相关的结构特征。我们发现了关键的粒度依赖性反转现象：虽然现代模型（……

    arXiv:2409.01382v3 Announce Type: replace-cross  Abstract: The adoption of Large Language Models (LLMs) for code generation risks incorporating vulnerable code into software systems. Existing detectors face two critical limitations: a lack of systematic cross-model validation and opaque "black box" operation. We address this through a comparative study of code generated by four distinct LLMs: GPT-3.5, Claude 3 Haiku, Claude Haiku 4.5, and GPT-OSS.   Analyzing 14,485 Python functions and 11,913 classes from the CodeSearchNet dataset, we generated corresponding code with all four LLMs. Using interpretable software metrics, we trained CatBoost classifiers for each configuration. Our analysis reveals that granularity effects dominate model differences by a factor of 8.6, with negligible feature overlap, indicating that function-level and class-level detection rely on fundamentally disjoint structural signatures.   We discover critical granularity-dependent inversions: while modern models (
    
[^351]: OpinionGPT: 模拟显性偏见的指令调整大型语言模型(LLMs)

    OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs. (arXiv:2309.03876v1 [cs.CL])

    [http://arxiv.org/abs/2309.03876](http://arxiv.org/abs/2309.03876)

    OpinionGPT是一个指令调整大型语言模型(LLMs)的web演示，用户可以选择各种偏见并进行比较，它意在使模型的偏见显性和透明化。

    

    最近，指令调整大型语言模型(LLMs)展示了生成与自然语言指令相匹配的回应的显著能力。然而，一个开放的研究问题涉及训练模型和它们的回应中固有的偏见。例如，如果用于调整LLM的数据主要由具有特定政治偏见的人编写，我们可能会期望生成的回答也共享这种偏见。目前的研究工作旨在除去这样的模型偏见，或抑制可能有偏见的回答。通过这个演示，我们对指令调整中的偏见持有不同的观点：我们的目标不是抑制它们，而是使它们显性和透明。为此，我们提供了OpinionGPT，一个网络演示，用户可以提问并选择所有他们希望调查的偏见。该演示将使用在代表每个选择偏见的文本上进行微调的模型来回答这个问题，从而实现并排比较。为了训练基础模型，我们选取了11个...

    Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 
    

