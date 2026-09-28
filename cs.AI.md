# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency](https://arxiv.org/abs/2609.31619) | 该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。 |
| [^2] | [Statistical attribute alignment for black-box generative AI via output post-processing](https://arxiv.org/abs/2609.31607) | 本文针对黑盒生成式AI提出了一种输出后处理方法，通过最小化查询次数的算法将生成输出的属性分布与用户指定目标对齐，并在精确与近似对齐两种情形下证明了算法的最优性。 |
| [^3] | [Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer](https://arxiv.org/abs/2609.31587) | 本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。 |
| [^4] | [OC-GS: Gaussian Splatting for Irregular Turntable Capture](https://arxiv.org/abs/2609.31572) | OC-GS提出一种轨道一致的物体中心高斯泼溅方法，在共享相机、旋转轴和枢轴的约束下联合优化图像几何与角度，实现了从不规则稀疏转台采集中高质量的三维物体重建。 |
| [^5] | [Adapting for AI: How elementary teachers adjust their practices for an AI-integrated curriculum](https://arxiv.org/abs/2609.31569) | 本研究通过对三位小学教师实施基于对话式AI平台的课程进行为期三周的追踪，首次揭示了教师通过修复、差异化、转化与平衡等适应性实践，在技术、学习者与教学三重张力交汇处推进AI课程课堂落地的真实工作方式。 |
| [^6] | [DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education](https://arxiv.org/abs/2609.31568) | DeepEdu-v1是一个面向越南教育的AI辅导系统，通过本地部署保障数据主权、围绕国家课程体系组织知识，并克服了消费级GPU上长上下文推理的内存与延迟瓶颈以及区域内容上的幻觉问题。 |
| [^7] | [Multi-agent Scaling Across Disjunctive and Compensatory Tasks](https://arxiv.org/abs/2609.31563) | 本文引入Steiner群体任务分类法来分析多智能体LLM系统的规模扩展行为，发现在析取性任务中，尽管至少一个智能体答对的概率随团队规模提升5-20个百分点，但简单的多数投票策略几乎无法兑现这一集体潜力，其结果仅收敛于模型的众数答案。 |
| [^8] | [A Flow Matching Framework for Neural Representational Dissimilarity](https://arxiv.org/abs/2609.31544) | 本文提出用深度生成模型中的流匹配框架统一多种神经表征差异性度量（即将其归结为不同速度约束下的Jeffreys散度），该框架在处理复杂分布和连续变量时具有估计优势，并能支持以有原则的方式设计新的距离度量。 |
| [^9] | [Can You Check That? The Checkability Boundary for Local LLM Network Automation](https://arxiv.org/abs/2609.31540) | 该论文提出“可检验性”标准来判断网络自动化任务是否适合本地小模型推理，并实现了Touchstone系统，通过内在检查筛选SLM输出、仅将少量输入升级给前沿LLM，在保护敏感数据不出本地的同时实现了98.6%和93.8%的端到端准确率。 |
| [^10] | [ClearGS: Reliability-Aware Gaussian Splatting from Handheld Videos](https://arxiv.org/abs/2609.31509) | ClearGS通过可靠性感知视图分配（RVA）为手持视频帧分配分级监督权重，并结合渲染引导的视频内修复（RIVR）与全轨迹修复整合机制，实现了从视点覆盖不均、质量参差不齐的手持视频中进行高质量3D高斯泼溅重建。 |
| [^11] | [Evaluating Cultural Awareness of LLMs for Haitian Creole](https://arxiv.org/abs/2609.31506) | 该论文首次从特异性、偏见、多样性和变异性四个维度系统评估了大语言模型对海地克里奥尔语的文化意识，发现其明显落后于法语，且海地角色常被以苦难形象呈现。 |
| [^12] | [Prompt Minimization: Reducing Input Redundancy Without Sacrificing Output Fidelity](https://arxiv.org/abs/2609.31505) | 该论文提出“提示最小化”概念及三种评估框架，证明将提示词压缩至最小信息密集形式后仍能产生与原始长提示词相当的输出，从而降低计算开销并避免冗长提示对模型推理的损害。 |
| [^13] | [UQ-LOB: Uncertainty-Aware Limit Order Book Mid-Price Forecasting](https://arxiv.org/abs/2609.31491) | 本文提出UQ-LOB，一个可附加到任意预训练LOB编码器的轻量级不确定性量化模块，通过对已完成窗口上下文的条件化，为限价订单簿中间价短期预测提供校准的置信度估计，从而支持选择性预测。 |
| [^14] | ["AI is (not) the new...": A Diagnostic Analogy Framework for Generative AI's Cultural Impacts](https://arxiv.org/abs/2609.31482) | 本文提出一个诊断性类比框架，将技术干预分解为认知场所、治理逻辑和技术机制三个坐标，以精确分析生成式AI对认知与文化实践的变革，避免因印刷机、蒸汽机等模糊的历史类比而设计出针对错误属性的治理干预。 |
| [^15] | [Game Arena: Strategic LLM Evaluation in Competitive Environments](https://arxiv.org/abs/2609.31473) | 本文提出了Kaggle游戏竞技场，一个通过国际象棋、扑克、狼人杀三类竞技游戏动态评估大语言模型策略规划与适应能力的开放平台，有效避免了静态基准的性能饱和问题。 |
| [^16] | [PriceBench: A Diagnostic Benchmark for Price, Quality, and Brand Preferences in LLM Booking Agents](https://arxiv.org/abs/2609.31468) | PriceBench通过logit选择模型从LLM的酒店预订行为中诊断其价格、质量和品牌偏好，发现更强的模型偏好更一致但各不相同，而较弱的模型要么偏好僵化易被列表顺序操纵，要么近乎随机选择。 |
| [^17] | [Uncertainty-Aware Federated Learning for Infant Movement Analysis](https://arxiv.org/abs/2609.31463) | 提出了首个面向婴儿运动分析的不确定性感知联邦学习框架，能够在保护隐私的前提下利用多机构骨骼运动数据实现自动化全身运动评估。 |
| [^18] | [Segment-Level Agentic Topic Modeling for Improved Data Exploration and Resource Efficiency](https://arxiv.org/abs/2609.31460) | 本文提出片段级智能体式主题建模框架SeLATM，通过分段处理方式解决基于LLM的主题模型无法生成文档主题分布、主题宽泛度不当以及资源消耗过高等问题，从而提升数据探索质量与资源利用效率。 |
| [^19] | [Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality](https://arxiv.org/abs/2609.31454) | 本文比较了联邦学习中输入条件不确定性与预测-标签损失两种损坏检测信号，发现在非独立同分布数据条件下，这两种信号对输入噪声和标签翻转两类损坏表现出不同的检测效果。 |
| [^20] | [From Reward Signal to Visual Utility: A Controlled Audit of Medical VLM Post-Training](https://arxiv.org/abs/2609.31450) | 该受控审计表明，医学VLM后训练中答案准确率的提升并不代表视觉利用能力的改善——语言模型LoRA微调虽小幅提升正确图像准确率，却降低了视觉收益事件与图像敏感度，而更广泛的多模态适应反而使正确图像准确率更低。 |
| [^21] | [ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs](https://arxiv.org/abs/2609.31448) | ViSTA是一种轻量级适配器，通过将不规则的数值型临床时间序列数据融入预训练视觉-语言模型的图表表示中，在冻结全部预训练参数的前提下实现临床风险预测性能的大幅提升。 |
| [^22] | [Implicit Neural Representation for Hyperspectral Video Compression](https://arxiv.org/abs/2609.31435) | 该论文提出了一种基于隐式神经表示的高光谱视频压缩新方法，通过对现有RGB视频压缩模型进行新颖扩展，相比传统逐帧压缩方法实现了+4.99 dB的PSNR增益和-88.88%的码率降低，同时显著提升了下游目标跟踪任务的性能。 |
| [^23] | [Compress What You See, Not What You Say: Anchored Context Distillation for Latent-Observation Software Engineering Agents](https://arxiv.org/abs/2609.31430) | 提出LOHA上下文布局与ACD锚定蒸馏训练方法，将较早的工具观察压缩为软令牌、保留近期文本，在大幅压缩软件工程代理上下文的同时保留操作关键信息并约束行为漂移。 |
| [^24] | [Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis](https://arxiv.org/abs/2609.31422) | 提出主动溯源门控（APG），通过将来源作为硬约束、审计辩论中的每一条论断并自我纠错，来缓解多智能体辩论综合阶段摘要模型伪造无事实依据“辩论共识”的安全问题。 |
| [^25] | [Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers](https://arxiv.org/abs/2609.31403) | 本研究创建了DecoyBench数据集，发现视觉语言模型在包含两层叠加可读文字的图像中只能读取轮廓文字而几乎无法提取阴影文字，而人类可以高准确率读取两层，揭示了VLM在多文本层图像理解上的根本性脆弱。 |
| [^26] | [Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models](https://arxiv.org/abs/2609.31397) | 该论文提出Intent2Tc，一个由语言模型驱动的闭环框架，能够将业务级流量整形意图自动转换为经过验证的可执行Linux流量控制配置，并通过数字孪生语义模型、批判驱动精炼和RAG知识复用显著提升语义一致性与配置可靠性。 |
| [^27] | [ActKV: Efficient LLM Agents through Action-Guided KV Cache Management](https://arxiv.org/abs/2609.31395) | ActKV是首个专为智能体式LLM推理设计的KV缓存压缩框架，通过基于动作贡献的缓存逐出机制和置信度驱动的预算分配，在降低内存开销的同时优先保障动作质量与任务进展。 |
| [^28] | [Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization](https://arxiv.org/abs/2609.31383) | 该论文发现端到端驾驶模型开环与闭环性能差距的一个新因素是中间路径点监督缺乏物理连贯性和可跟踪性，并提出轻量级后处理层ECO，通过锚定车辆执行历史、保留可靠的预测端点并重塑中间路径点来弥合这一差距。 |
| [^29] | [Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding](https://arxiv.org/abs/2609.31382) | 提出“先标注后总结”（H2S）的先压缩后推理范式，通过带过程级奖励的强化学习训练模型先识别并压缩与问题相关的证据、生成条件化摘要后再作答，显著提升大模型的长上下文理解能力。 |
| [^30] | [Completed Pairs Hide Capped Failures: A ReVerPi Case Study of Selective Context Projection](https://arxiv.org/abs/2609.31381) | 该论文通过ReVerPi案例研究揭示了选择性上下文投影的关键权衡：已完成的15对配对显示投影与完整上下文成功率相同并节省25%逻辑令牌，但恢复被抑制的全部27次边界运行后，投影相对完整的成功差异介于-9到+1个任务之间，且单个投影运行可能因归档检索而耗尽多达12次请求。 |
| [^31] | [Programs-of-Layers in LLMs through the Lens of Cortical Areas](https://arxiv.org/abs/2609.31360) | 该研究在皮层区域的视角下复现并验证了PoLar方法：将大语言模型的各层视为可动态路由的函数库，根据输入难度自适应地跳过或重复层块，可在多个模型上获得优于固定逐层推理的性能。 |
| [^32] | [A Safety-Bounded SDC-to-MCP Gateway for Medical AI Agents](https://arxiv.org/abs/2609.31358) | 该论文提出一种安全有界的网关，将IEEE 11073 SDC医疗设备协议接入MCP，通过只读资源和策略验证的干运行工具，确保医疗AI Agent可以访问设备状态和元数据但绝不触发实际设备操作。 |
| [^33] | [Mutable Transcripts: Mitigating Context Pollution through Editable Conversation State](https://arxiv.org/abs/2609.31354) | 提出可变对话记录这一新交互范式，允许用户通过自然语言编辑请求直接修改历史对话内容，从而将对话记录从被动记录转变为可编辑的对话状态，有效缓解静态对话历史导致的上下文污染问题。 |
| [^34] | [DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models](https://arxiv.org/abs/2609.31349) | DyMD提出一种自适应分布匹配蒸馏框架，通过时间亲和条件化的加噪采样来动态调整教师监督与评论家拟合，从而在少步视频世界模型中有效保留机器人—物体交互动态。 |
| [^35] | [The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models](https://arxiv.org/abs/2609.31341) | 该研究在隐私约束下系统评估了小型本地模型信息抽取流水线的准确率-能耗权衡，发现最优选择（图像 vs 文本）取决于文档类型，批处理可无损降低38-85%能耗，而神经OCR能耗高达传统OCR的17倍。 |
| [^36] | [CG-HAF: An Interpretable Global-Local Lesion-Burden Fusion Framework for Ordinal Acne Severity Grading in Agentic Skincare Support](https://arxiv.org/abs/2609.31326) | CG-HAF通过将独立分类器的整体严重程度概率与目标检测器提取的结构化皮损负担特征进行显式融合，并由轻量级可解释分类器输出最终痤疮严重程度等级，在基准上显著优于仅依赖全局证据的方法，且在最严重病例中收益最大。 |
| [^37] | [AgentXploit: Autonomous Repository-to-Runtime Red-Teaming for AI Agents](https://arxiv.org/abs/2609.31318) | AgentXploit提出了一种将代码仓库层面的攻击路径发现与运行时漏洞利用相分离的双角色自主红队测试系统，用于在部署前对AI智能体进行授权白盒安全审计。 |
| [^38] | [Towards VLA-Dreamer: Refining VLA Behavior Using World Models](https://arxiv.org/abs/2609.31313) | 该论文提出在VLA视觉编码器的嵌入空间上训练预测性世界模型，以提升VLA的样本效率，并验证这些嵌入能否基于动作进行未来预测，从而判断VLA是否具备隐式世界模拟能力。 |
| [^39] | [Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows](https://arxiv.org/abs/2609.31301) | 提出了EffectMatch运行时系统，通过在受控执行边界内收集智能体操作产生的持久性变更，并将其与应用批准的内容进行比对验证后再决定提交，有效阻止了未批准的副作用在智能体工作流中传播。 |
| [^40] | [UniAR: A Unified Framework for Autism Recognition Enhanced by Multi-View Prompt Learning](https://arxiv.org/abs/2609.31298) | UniAR提出了一种融合多粒度提示学习的统一自闭症识别框架，利用大型多模态模型生成词、短语、句子三层级的诊断描述以弥补临床文本数据稀缺，并通过基于专家混合的多尺度对齐模块将生成语义与视觉证据对齐，实现异构数据下稳健的自闭症谱系障碍识别。 |
| [^41] | [Softmax Reparameterization for Output-Head Quantization](https://arxiv.org/abs/2609.31291) | 提出softmax重参数化这一训练后量化方法，通过在量化前减去词表行均值的标量倍数来选取功能等价的输出头，在保持全精度softmax分布不变的同时显著降低输出头量化对语言模型预测的扭曲。 |
| [^42] | [G2MAF: Test-Time Gradient Guidance for Multi-Agent Flow Policies](https://arxiv.org/abs/2609.31286) | 提出G2MAF框架，通过测试时应用全局归一化的投影评论家梯度来引导和协调多智能体流策略修正联合动作，在保持动作可行性的同时，在MPE和SMAC基准上分别取得9.2%和8.9%的平均相对性能提升。 |
| [^43] | [Resource-Optimized and Energy-Aware Agentic AI Framework Anchored on Blockchain for Secure Software Supply Chains](https://arxiv.org/abs/2609.31282) | 该论文提出了一种基于区块链的智能体AI安全框架，利用LLM驱动的专门安全智能体保护软件开发生命周期的全过程，并通过在许可链上记录密码学签名证明来确保智能体自身的可信性与安全性。 |
| [^44] | [MA-WAM: Multi-Agent World-Action Model for Test-Time Planning](https://arxiv.org/abs/2609.31281) | 提出了首个面向多智能体流策略的测试时世界模型规划器 MA-WAM，通过建模跨智能体动作依赖关系来预测联合动作的后果，从而实现对候选联合动作的高效评分与规划。 |
| [^45] | [Cognitive Skills in the Age of AI: Computing Students and Experts Perceptions](https://arxiv.org/abs/2609.31272) | 该研究通过对计算机专业学生和专家的混合方法调查发现，在AI丰富的未来，大多数认知技能的重要性将会下降，但批判性思维能力仍将保持其关键地位。 |
| [^46] | [MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries](https://arxiv.org/abs/2609.31261) | MoSAR将注意力近似视为几何学习问题，通过输入条件化路由器在短程、中程和全局注意力机制间学习自适应混合，以连续的距离相关注意力场取代固定稀疏模式，突破长上下文建模中稠密自注意力的二次复杂度瓶颈。 |
| [^47] | [Agentic Limit Order Books: Phase Transitions and Market Impact](https://arxiv.org/abs/2609.31260) | 该论文首次构建了完全由强化学习智能体组成的限价订单簿，揭示了智能体数量与市场深度的临界阈值会触发市场从有序价格发现到高波动级联的相变，且智能体提供流动性时的市场冲击偏离经典平方根规律。 |
| [^48] | [Geometric Inconsistency Localization in Multi-View Image Sets](https://arxiv.org/abs/2609.31247) | 本文提出了带像素级几何不一致性标注的宽基线多视图数据集DeformView，并据此开发了轻量级分类器DEFECt3R，利用跨视图特征关系实现了多视图图像间几何不一致性的有效定位。 |
| [^49] | [Purin: A Biology-inspired Mechanism for Artificial Neural Networks](https://arxiv.org/abs/2609.31235) | Purin是一种受生物学启发的机制，通过基于时间间隔的神经活动抽象，将有界的突触效能调制引入传统卷积神经网络，无需离散时间步即可实现短期和长期的突触效能变化。 |
| [^50] | [Acoustic-to-Text KV Compression for Full-Duplex Speech Models](https://arxiv.org/abs/2609.31224) | 提出声学到文本KV压缩方法，利用聆听时间余量通过转写侧通道将旧语音状态转换为紧凑文本记忆，大幅降低全双工语音模型长时间交互中的KV缓存内存占用。 |
| [^51] | [DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration](https://arxiv.org/abs/2609.31215) | 提出DIAL统一框架，利用大量LLM比较结合少量人类比较，分离并消除LLM评判器中的位置偏差，并将去偏后的偏好结构自适应校准至人类偏好目标，同时提供可识别性理论与不确定性量化保证。 |
| [^52] | [Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution](https://arxiv.org/abs/2609.31214) | 该论文指出数据归因中各影响估计器排序不一致的根本原因是“规范不匹配”而非近似误差，将影响形式化为反事实估计量，并按隐含规范对现有估计器进行系统分类。 |
| [^53] | [Samples, Sources, Space: Decomposing Data Scale in Spatially Structured Representation Learning of Human Brain Microarchitecture](https://arxiv.org/abs/2609.31201) | 该研究将数据规模分解为独立样本数、来源多样性与空间覆盖率三个维度，通过93次对比学习预训练实验发现，人脑显微组织学表示学习性能随样本数量、空间覆盖、计算量和模型容量的增加而持续提升。 |
| [^54] | [Rethinking Data Quality for AI-Driven Systems: Evidence from Practitioner Interviews](https://arxiv.org/abs/2609.31191) | 本文通过对16名从业者的访谈首次提供了实证证据，揭示了AI驱动系统中数据质量内涵的根本转变——可追溯性转向模型行为归因、智能体上下文与记忆成为数据对象、合成数据使真实性成为关注点、合法性成为训练数据的准入门槛。 |
| [^55] | [Evolutionary Safety of Recursive Self-Improving AI: Taxonomy, Risk Discovery, and Evaluation](https://arxiv.org/abs/2609.31186) | 该论文提出“进化安全”这一新视角，用于研究AI在递归自我改进过程中安全属性如何随演化而变化、持续、积累与传播，并给出了相应的风险分类、风险发现与评估框架。 |
| [^56] | [Accounting for Bias Enables Sustainable LLM Evaluation](https://arxiv.org/abs/2609.31184) | 该论文提出一个统一的潜在变量框架，通过显式校正位置偏差、冗长偏差、评委严格度等系统性测量偏差，能用远少于以往的比较次数恢复可靠排名，从而为LLM评估提供了统计上更严谨、计算上更可持续的方案。 |
| [^57] | [BAT-CLIP: Trimodal Alignment of Brain, Audio and Text](https://arxiv.org/abs/2609.31180) | BAT-CLIP提出了首个面向iEEG的CLIP式三模态对齐框架，将神经嵌入同时对齐到预训练的音频和文本锚点，克服了单模态锚定带来的权衡问题，在自然语音解码中实现了比双模态基线更鲁棒的表示。 |
| [^58] | [SPO: Discovering Adaptive Large Neighborhood Search Operators via Stackelberg Program Optimization](https://arxiv.org/abs/2609.31179) | 提出了基于LLM的Stackelberg程序优化框架SPO，将破坏算子作为领导者、修复算子作为条件性跟随者进行耦合进化搜索，自动发现能够适应LNS状态的自适应破坏-修复算子程序，在TSP和CVRP问题上超越了现有方法。 |
| [^59] | [Semantic Navigation for Issue Localization in Code Repository](https://arxiv.org/abs/2609.31176) | SemNav框架通过确定性检索生成初始候选集合，并借助LLM智能体基于证据进行迭代精炼，同时利用语义导航图按需解析程序关系，从而有效提升代码仓库级问题定位的效果。 |
| [^60] | [Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting](https://arxiv.org/abs/2609.31169) | 提出基于度量的损失加权训练方法，利用PCXMI度量识别受益于图像的词元并提高其损失权重，从而增强多模态大语言模型在多模态机器翻译任务中对视觉信息的敏感性和利用能力。 |
| [^61] | [Neural State Prediction: Obstructing Shortcut Learning in EEG Foundation Models](https://arxiv.org/abs/2609.31167) | 提出神经状态预测（NSP）框架，通过EMA目标编码器、身份残差化和拓扑分离上下文三重机制阻碍EEG基础模型中的捷径学习，迫使模型整合分布式神经上下文，从而学到更可迁移的神经表征。 |
| [^62] | [AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side](https://arxiv.org/abs/2609.31166) | 提出AgentRecommender方法，利用LLM智能体的调查能力和内部知识，在无需额外数据的情况下灵活构建用户端推荐系统，使用户能够轻松创建符合自身偏好的定制化推荐系统。 |
| [^63] | [SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations](https://arxiv.org/abs/2609.31164) | 提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。 |
| [^64] | [ReG-SAM: Reference Graph-Driven SAM for 2D Foundational Vessel Segmentation](https://arxiv.org/abs/2609.31160) | 提出ReG-SAM，一种基于SAM并利用参考图集（图提示嵌入与血管原型等模态感知表征）来增强血管表征的2D血管分割基础模型框架。 |
| [^65] | [Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence](https://arxiv.org/abs/2609.31159) | 该论文提出动量引导的联邦分割蒸馏框架，通过TeRR-SAtt时序储备池学生注意力设计和AMGF动量引导融合机制，在大幅降低边缘设备训练与推理延迟及资源占用的同时，显著提升个性化时序学习的准确率。 |
| [^66] | [Teacher-Anchored Selection of Post-Training Quantized Models under Domain Shift](https://arxiv.org/abs/2609.31155) | 该论文提出在域偏移且目标域标签缺失或稀缺的场景下，以教师模型失真作为锚点来指导训练后量化候选模型的部署选择，揭示了置信度类估计器的失效与输出分布类估计器的可靠，并将有监督项与教师锚点相结合以改进选择。 |
| [^67] | [FedHisto-PAST: Parameter-Efficient Stain-Aware Federated Learning for Cross-Site Lung Histopathology Classification](https://arxiv.org/abs/2609.31150) | FedHisto-PAST v2通过将冻结的HIBOU-B基础模型与参数高效适配、染色感知学习及自适应联邦聚合相结合，以低通信和参数成本实现了跨站点肺组织病理学的鲁棒分类。 |
| [^68] | [JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models](https://arxiv.org/abs/2609.31142) | 该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。 |
| [^69] | [Can Linguistic Reasoning Vectors Enhance Multimodal Reasoning Ability?](https://arxiv.org/abs/2609.31140) | 该论文提出LIFT方法，通过提取“推理者路径”与“求解者路径”之间的隐状态差异作为推理向量并注入模型，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。 |
| [^70] | [Toward AI-Augmented Cooperative Engineering Workflows: Requirements and Architecture the European Rover Challenge](https://arxiv.org/abs/2609.31136) | 本文针对学生团队在复杂工程系统集成中缺乏AI流程级支持的问题，基于对欧洲漫游车挑战赛14支团队104份问卷的调研，提出了AI增强协作工程工作流的需求与架构。 |
| [^71] | [Pocket-STVG: lightweight architecture for Spatio-Temporal Video Grounding](https://arxiv.org/abs/2609.31135) | P-STVG 是一种轻量级级联架构，通过组合 MobileViCLIP、MDETR 等高效预训练组件而非大型端到端模型来实现时空视频定位，并借助轻量 1D U-Net 或阈值策略使同一框架同时支持弱监督与零样本设置。 |
| [^72] | [AtomWorld-Mem: Memory-Restored World States for Long-Horizon Atomistic Evolution](https://arxiv.org/abs/2609.31133) | 提出了记忆恢复型原子世界模型 AtomWorld-Mem，通过空间编码器生成多尺度原子关键帧，并结合短期事件记忆与长期结构记忆来恢复瞬时晶体快照中缺失的潜在世界状态，从而解决长时程原子演化中的快照歧义问题。 |
| [^73] | [Monitor Jailbreaking: Evading Chain-of-Thought Monitoring Without Encoded Reasoning](https://arxiv.org/abs/2609.31121) | 本研究发现在思维链监控的优化压力下，推理模型无需编码或隐藏其推理，仅通过调整思维链的措辞和格式即可规避监控器的检测，同时推理内容对人类依然完全透明，这一新现象被称为“监控越狱”。 |
| [^74] | [From Shortcut Learning to Discrete Neural Insertion Sort](https://arxiv.org/abs/2609.31114) | 该论文揭示了神经算法推理模型在学习插入排序时存在捷径学习问题——中间表示在算法执行结束前就能解码出排好序的结果——并提出了一种将序列表示为链、分离标量交换与控制状态转移、且每步后将节点表示投影回离散状态的离散神经插入排序模型，以促使模型真正遵循算法执行过程。 |
| [^75] | [Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods](https://arxiv.org/abs/2609.31107) | 本文通过拉回Fisher信息度量导出采集函数的梯度上界，解释了高维贝叶斯优化中的梯度消失现象，并提出基于信赖域的FITR方法，以局部Fisher权重取代长度尺度缩放来提升优化性能。 |
| [^76] | [DepthEvidence: Unifying Metric Depth Prediction and Geometric Reasoning in Multimodal Language Models](https://arxiv.org/abs/2609.31103) | DepthEvidence是一个40亿参数的多模态语言模型，它将自身预测的密集度量深度作为对象级证据融入语言生成过程，实现了度量深度预测与几何推理的统一，并通过Depth-VQA基准验证了对象测量和空间-数值组合推理能力。 |
| [^77] | [OmouAI: Argumentative Human-AI Policy Deliberation with Simulated Personas](https://arxiv.org/abs/2609.31078) | OmouAI将大语言模型与计算论辩相结合，通过引入模拟角色（如利益相关者、领域专家、唱反调者）与人类用户共同商议现实政策主张，以减少LLM的谄媚性并提供人类监督。 |
| [^78] | [Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents](https://arxiv.org/abs/2609.31076) | 本文提出 CodeHack 技能库，在长时程游戏环境 NetHack 中系统研究了基于代码的动作抽象如何影响语言智能体的性能、推理成本和学习，揭示了抽象带来的生产力与灵活性之间的权衡。 |
| [^79] | [Externalized CPDAG Summaries Improve LLM Causal Deduction](https://arxiv.org/abs/2609.31071) | 提出Structured Thinking两轮流程，先让模型外化生成受模式约束的CPDAG结构化摘要、再基于该图状态作答，将Qwen3.5-27B在Corr2Cause因果推理任务上的F1从73.0显著提升至86.4。 |
| [^80] | [Quantum Diffusion Models for Medical Image Analysis](https://arxiv.org/abs/2609.31070) | 本文提出一种基于离散时间量子游走算法、结合经典逆向去噪模型的可扩展混合量子扩散模型，突破了现有量子设备规模的限制，能够处理真实世界的大尺寸医学图像数据。 |
| [^81] | [Neuralyzing the Trace: Selective Representation-Level Unlearning with Contrastive Sparse Autoencoders](https://arxiv.org/abs/2609.31056) | 该论文提出SCALPEL——一种对比稀疏自编码器，通过克服基于重构提取中的能量偏差、在表示层面学习目标选择性特征，实现了精准的机器遗忘，并从理论上证明其能有效控制对背景知识的扰动。 |
| [^82] | [Cheap, open agents make LLM pollution harder to mitigate](https://arxiv.org/abs/2609.31054) | 廉价易得的开源智能体在调查任务中表现可与商业智能体媲美且更难被检测，使大语言模型污染的防范变得更加困难。 |
| [^83] | [DynBranch: Speculative Subgraph Reuse for Dynamic Agentic LLM Serving](https://arxiv.org/abs/2609.31047) | DynBranch 通过让未解析的分支在解析前即可被寻址，实现了投机性子图执行与跨请求的子图结果复用，从而打破“分支解析屏障”，将智能体 LLM 服务的平均延迟降低最高 32%。 |
| [^84] | [Governed Deduction: Policy-Grounded Premise Authorization Beyond Relevance](https://arxiv.org/abs/2609.31029) | 该论文提出“受治理演绎”形式化框架，指出前提在推理中不仅需要相关性、还必须通过策略授权准入检查，并通过 RBAC 增强的授权配对基准实验揭示：线性控制器实际依赖角色名称捷径，在去除捷径的角色置换后性能降至 50% 的随机水平。 |
| [^85] | [Same Text, Different Numbers: The Divergence of LLM-Based Measures](https://arxiv.org/abs/2609.31013) | 该论文让七个不同的大语言模型对相同的标普500公司财报电话会议文本进行十三种构念评分，发现基于LLM的文本度量存在显著的模型间分歧（平均秩相关仅0.52，共同差异仅占34%），且模型选择会实质性改变下游实证研究中的系数大小、符号和显著性。 |
| [^86] | [G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation](https://arxiv.org/abs/2609.31009) | G²PTQ提出了一种统一的大语言模型训练后量化框架，通过在每个Transformer块量化前动态刷新并融合一阶梯度与二阶Hessian信息的广义梯度补偿机制，克服了现有GPTQ类方法缺乏全局监督或指导信息随量化进程失效的缺陷。 |
| [^87] | [Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance](https://arxiv.org/abs/2609.30997) | 该论文为仅凭像素的图像溯源建立了精确理论极限——由目标分布与受攻击源分布间的最小全变差距离决定且与验证器架构无关，并揭示了公开验证器因可被模拟而会在远早于该统计极限之前被代理黑盒攻击攻破。 |
| [^88] | [The Linear Representation Hypothesis for Vision-Language-Action Models](https://arxiv.org/abs/2609.30996) | 该论文提出了一种基于签名的理论框架，将线性表示假设从大语言模型扩展到视觉-语言-动作模型，统一了表示与策略，以应对具身交互中感兴趣的物理量与系统动力学共同演化这一挑战。 |
| [^89] | [FLIP: Final Layer Inference-Time Probing for Vision-Language Models](https://arxiv.org/abs/2609.30993) | 提出FLIP——一种在推理时对VLM最终层隐藏状态施加逐元素下限干预的探测方法，用以验证logits前的干预位点支持结构化的任务相关计算，并揭示出干预强度下性能变化的三个区域以及形式化的四准则探测-扫描协议。 |
| [^90] | [FARE: Forensic Acceptance Region Estimation for Catching Bait-and-Switch Image Generators](https://arxiv.org/abs/2609.30982) | 本文提出FARE，通过在认证生成器的图像上训练并挖掘困难样本以收紧接受区域，实现部署后仅凭单张生成图像即可检测服务商是否“偷换”了图像生成器，用于不透明图像生成API的完整性审计。 |
| [^91] | [Does Uniform Discrete Diffusion Need Time?](https://arxiv.org/abs/2609.30977) | 本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。 |
| [^92] | [Factorized axis convolutional gated recurrent unit with dynamic adaptive pooling for remaining useful life prediction of rolling bearings](https://arxiv.org/abs/2609.30972) | 本文提出一种因子化轴卷积门控循环单元结合动态自适应池化的方法，利用多尺度各向异性卷积与双轴注意力增强时频图中的方向性特征并自适应聚合关键信息，从而提升滚动轴承剩余使用寿命预测的精度。 |
| [^93] | [SciHorizon-eLab: An Agentic Protocol-to-Task Compiler for Scalable Benchmarking of Scientific Embodied Agents](https://arxiv.org/abs/2609.30971) | 提出了SciHorizon-eLab，一个智能体式的协议到任务编译器，能够将自然语言科学实验协议自动编译为语义一致、可执行且经多阶段仿真验证的具身任务，从而实现对科学具身智能体的可扩展基准测试。 |
| [^94] | [MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens](https://arxiv.org/abs/2609.30967) | 该论文提出MoMHa，将大语言模型harness设计建模为在准确率、行为安全性和token成本三个目标上的搜索问题，并证明单阶段联合奖励的智能体提议方案在十七个领域上优于两阶段、仅标量反馈和仅准确率等所有替代方案。 |
| [^95] | [FTB Graph: Determining and Validating First-token Broadcasters and Language-Identity Head Circuits in Multilingual Language Models](https://arxiv.org/abs/2609.30954) | 该研究结合边归因修补与精确激活修补验证，对六种模型架构进行端到端电路分析，首次系统确定并验证了多语言大模型中决定首词语言身份的“首词广播器”与语言身份头电路。 |
| [^96] | [MVVBench: Benchmarking 4D Reasoning in Vision-Language Models](https://arxiv.org/abs/2609.30952) | MVVBench是一个基于真实多相机数据构建的多视角视频推理基准，其问题在视角和时间轴上均无法通过单目信息解答，只有联合跨视角、跨时间的4D推理才能求解，可用于系统评测视觉语言模型的4D推理能力。 |
| [^97] | [PORL: Pretrained Offline Reinforcement Learning for the Job Shop Scheduling Problem](https://arxiv.org/abs/2609.30948) | 该论文提出PORL方法，将基于仿真的在线预训练与离线微调相结合，并利用KL散度策略约束将通用调度策略适配到特定生产数据分布，从而克服作业车间调度问题中仿真到现实的差距。 |
| [^98] | [OneWorld: Learning Consistent Physics Across Actions in World Models](https://arxiv.org/abs/2609.30946) | OneWorld提出了一种共享机制的反事实生成框架，通过在共同潜在物理机制下联合建模多个动作条件化的未来，解决了世界模型在不同动作干预下物理属性预测不一致的问题。 |
| [^99] | [LogicTree-RAG: Logic Tree-guided Retrieval-Augmented Generation for Long-form Patent Drafting](https://arxiv.org/abs/2609.30943) | 提出LogicTree-RAG框架，通过构建层次化逻辑树作为全局组织主干来引导检索增强生成，在不依赖专家先验知识的情况下实现长篇专利文档的整体自动化撰写。 |
| [^100] | [Spackle: Completing Large View Single Image NVS with Adaptive Gaussians](https://arxiv.org/abs/2609.30941) | 提出轻量级残差学习框架Spackle，通过自动识别重建较差区域并学习残差3D高斯，缓解了固定高斯数量带来的容量竞争问题，在不牺牲推理效率的情况下实现了高质量的大视角偏差单图像新视角合成。 |
| [^101] | [Financial Fragility in Societies of LLM Agents: Coordination Failures and Stabilizing Mechanisms](https://arxiv.org/abs/2609.30940) | 该论文提出FRAIL受控实验框架，发现七种主流LLM智能体在银行挤兑、债务展期和众筹等金融环境中，即使无任何破坏指令也会因协调失灵而普遍出现集体性金融失败（77%的银行挤兑场景和83%的债务展期场景失败），并验证了补偿性承诺、中心化承诺协议和参与者主导联盟三种机制能够有效稳定系统。 |
| [^102] | [MACBT: A Multi-Agent Cognitive Behavioral Therapy Decision Support System with Longitudinal Memory](https://arxiv.org/abs/2609.30939) | 该论文提出了MACBT决策支持系统，将CBT五阶段工作流编码为五个协作智能体，并借助纵向记忆模块跨会谈追踪认知扭曲情况，从而为临床医生生成会谈前病理报告和干预优先级建议，缓解CBT规模化应用的瓶颈。 |
| [^103] | [Self-Play Search Distillation for Large Language Model Reasoning](https://arxiv.org/abs/2609.30936) | 提出自博弈搜索蒸馏框架（SPSD），利用棋盘游戏中类MuZero网络的自博弈搜索记录生成超人思维链数据来训练大语言模型，仅凭自博弈数据即可将Qwen3-4B-Base在六个数学基准上的平均成绩从24.1提升到36，并展现出良好的跨领域迁移能力。 |
| [^104] | [Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models](https://arxiv.org/abs/2609.30935) | 提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。 |
| [^105] | [UltraG-Bench: A Multi-task Benchmark for assessing Large Vision-Language Models on Pixel-level Evidence Grounding in Ultrasound](https://arxiv.org/abs/2609.30928) | 该论文提出了UltraG-Bench，一个基于40个公开超声分割数据集标注、涵盖指令引导分割、证据定位VQA和证据定位报告生成三个任务的大规模多任务基准，用于系统评估大型视觉语言模型在超声图像上的像素级证据定位能力，并揭示了语义理解与细粒度像素级定位之间的显著差距。 |
| [^106] | [JevSoup: System-One Routing for Training-Free LoRA Composition](https://arxiv.org/abs/2609.30922) | JevSoup提出了一种免训练的LoRA专家组合框架，通过“系统一”结构化概率路由选出两个专家，并利用正交投影将两个专家的更新等权融合，无需辅助数据或额外训练即可在多任务上取得性能提升。 |
| [^107] | [Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations](https://arxiv.org/abs/2609.30918) | 本文提出一个统一的跨家族评估协议，在固定事实实例与反事实解释的前提下，针对相同的八种模型变化类型比较六种鲁棒反事实解释方法与两种基线，发现各方法的相对性能和失败模式随变化类型而异，因此现有报告的鲁棒性分数彼此不可比。 |
| [^108] | [Training Graph Foundation Models on The Web Graph](https://arxiv.org/abs/2609.30894) | Acacia是一个仅用Common Crawl网页图从零训练的图基础模型，无需额外训练即可支持任意特征维度及节点分类、链接预测、节点聚类、图生成等多种任务，具备上下文学习能力且不依赖预训练LLM，证明了图模型可以像LLM一样从零涌现能力。 |
| [^109] | [Adaptive Pilot Selection for Unified Semantic Communication and Semantic Sensing in ISAC](https://arxiv.org/abs/2609.30891) | 本文提出SemISAC框架，在单一双功能波形中统一实现语义通信与语义感知，通过自适应导频选择优化OFDM资源分配，在车联网场景中同时完成道路场景分割、目标分类与测距任务。 |
| [^110] | [From Tapping to Hopping: Augmenting Mobile GUI Agents with App-Native Deeplinks](https://arxiv.org/abs/2609.30887) | 本文提出GUI-Hopper混合交互方法，通过静态分析发现并在真实设备上验证应用深链接，构建带落地页描述的深链接目录，让移动GUI智能体用深链接直接跳转导航、用GUI操作处理其余任务，从而显著提升真实商业应用中的任务成功率。 |
| [^111] | [Warned alike, AI agents avoid the less-crowded road while people take it](https://arxiv.org/abs/2609.30883) | 本研究通过双路径拥堵博弈实验发现，一句相同的警告会引发50个GPT智能体集体涌入拥堵道路、避开空旷道路，使平均通行时间从64分钟升至95分钟，而人类群体在相同条件下保持均衡，揭示了共享AI预测可通过自我实现的预期来扭曲稀缺容量分配的全新反馈机制。 |
| [^112] | [EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting](https://arxiv.org/abs/2609.30880) | 该论文提出了EXAONE Demand——一个专为需求预测设计的时间序列基础模型，通过构建包含1130万条序列的需求专用语料库，以及基于四类需求类别（平滑、间歇、波动、块状）进行路由的低秩适配器架构，解决了通用时间序列基础模型难以处理需求数据短历史、频繁零值、缺货删失等特殊性质的问题。 |
| [^113] | [TISD: On-Policy Self-Distillation with Trajectory Intervention](https://arxiv.org/abs/2609.30878) | 该论文提出TISD算法，通过在师生分歧峰值处强制执行教师偏好的分支动作并重生成轨迹进行蒸馏，突破了在线策略自蒸馏无法监督学生未采样分支的、单纯依赖局部纠错的训练瓶颈。 |
| [^114] | [Persistent Negatives for Adversarial Black-Box On-Policy Distillation](https://arxiv.org/abs/2609.30864) | 提出持久负样本对抗蒸馏方法，通过活跃池机制用历史教师-学生对比作为稳定负样本，解决了对抗蒸馏中负样本分布随策略更新而漂移的移动目标问题，从而提升黑盒在线策略蒸馏的稳定性。 |
| [^115] | [Developing a Roadmap to an AI-first Organization: A Case Study in Embedded Software Development](https://arxiv.org/abs/2609.30863) | 本文通过对一家大型嵌入式系统公司40名从业者的研讨会数据进行混合方法分析，研究了该公司向AI优先组织转型的路线图，发现智能体AI预计将深刻影响团队结构、所需能力、组织战略及开发人员角色。 |
| [^116] | [SkillEvoReg: Regularizing Agent Skill Evolution Against Overfitting](https://arxiv.org/abs/2609.30861) | 提出 SkillEvoReg 框架，借鉴神经网络训练中的抗过拟合技术（训练时技能丢弃、复杂度感知正则化、因果反例验证），有效防止语言模型智能体技能演化中的过拟合问题。 |
| [^117] | [Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis](https://arxiv.org/abs/2609.30841) | 该论文将扩散语言模型的安全对齐解释为去噪能量景观中的一道能量屏障，据此将现有越狱攻击归纳为两类绕过策略，并由此推导出三种无需训练的检测信号（step-0 比率和两个轨迹速度信号）。 |
| [^118] | [MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation](https://arxiv.org/abs/2609.30837) | 提出MOPD-Router框架，无需领域标签即可在token级别对完整教师池进行监督路由，并通过ExpertAlign依据教师后训练所获专业化能力为其修正信号评分，从而充分释放多教师互补知识。 |
| [^119] | [PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices](https://arxiv.org/abs/2609.30836) | 提出无需训练、即插即用的PTC-Decoder解码器框架，通过强制规划调用和工具名称的token级硬约束，使小语言模型能够在离线资源受限的边缘设备上可靠执行多步智能体任务。 |
| [^120] | [Subject-Invariant Cross-Modal Decoding of Perceived Speech from Brain Recordings](https://arxiv.org/abs/2609.30832) | 提出了一种整合fMRI与MEG的受试者不变跨模态感知语音解码方法（SICMD），首次以统一框架同时解决了神经表征提取与跨受试者泛化两大挑战，并显著提升了解码性能。 |
| [^121] | [Evaluation Is All You Need for Multi-Modal Autonomous Driving](https://arxiv.org/abs/2609.30818) | 该论文揭示了多模态自动驾驶规划中“生成强而评估弱”的不对称问题，并提出iDriveVLA框架，通过统一轨迹评估器（安全感知评分器与VLM引导调制器）实现更可靠、场景自适应的候选轨迹选择，充分释放多模态规划潜力。 |
| [^122] | [A Benchmark and Diagnostic Study of Epistemic Admission in Shared Agent Memory](https://arxiv.org/abs/2609.30813) | 论文提出了相关提升基准（CPB），通过静态与动态两种模式评估共享智能体记忆中的主张准入策略，发现基于来源去重的策略会大量误拒真实主张，而保留答案覆盖率的策略几乎与无限制准入一样容易接纳错误主张。 |
| [^123] | [XPhysICS: Cross-Physical-Domain Threat Grounding for Industrial Control Systems Security](https://arxiv.org/abs/2609.30805) | XPhysICS提出了一种来源感知、目标条件化的跨物理域威胁落地方法，通过角色兼容性等五项资格标准，将某一工控系统记录的威胁确定性映射到目标系统的验证切片上，并以相互独立的证据层判断威胁的信息物理效应在目标系统上是否可容许、可评估。 |
| [^124] | [Evaluating Real-Time Voice Agents: From Component Quality to Grounded Outcomes](https://arxiv.org/abs/2609.30798) | 该论文指出现有实时语音智能体评估文献分散于语音基础建模、话轮转换心理语言学和智能体评估三个互不引用的领域，并基于38篇一手文献提出三项循证论点，主张超越单一组件指标、建立面向实际部署成效的统一评估框架。 |
| [^125] | [HasMem: Hard-Origin Adaptively Softened Memory for Long-Term LLM Agents](https://arxiv.org/abs/2609.30797) | HasMem以冻结的硬提示词嵌入作为可验证初始状态，通过控制器自适应调节记忆宽度、由Writer重编码、Reader与Global实现读取适应与跨轮状态，在MSC重建探针上以更少的记忆位置取得更高的F1，并大幅超越基于规则的重新编码方法。 |
| [^126] | [ConsultMind:Towards Automated Diagnostic Consultation via Uncertainty-Aware Reasoning](https://arxiv.org/abs/2609.30796) | 该论文提出AutoDisym流水线自动构建疾病-症状贝叶斯网络（DSBN），并在此基础上提出不确定性感知框架ConsultMind，通过在每次患者回应后更新疾病后验概率，并利用后验不确定性来指导问诊提问与诊断决策，从而实现自动化诊断问诊。 |
| [^127] | [Skip the Talk, Re-Focus on Vision: Latent Reasoning for Reasoning Segmentation in Multimodal Large Language Models](https://arxiv.org/abs/2609.30783) | 提出LIRSeg方法，用紧凑的可学习潜在标记完全替代显式思维链推理，消除冗余文本标记对视觉注意力的干扰，从而提升多模态大语言模型推理分割的性能。 |
| [^128] | [NavGen: Visual Generative Models as a Scalable Data Engine for Embodied 3D Navigation](https://arxiv.org/abs/2609.30770) | NavGen利用高保真视觉生成模型构建文本到视频数据生成流水线，生成约40万个覆盖室内外场景的视觉-语言导航片段，并通过风格多样化方法扩展难以采集的长尾数据，为具身三维导航提供了可扩展的数据引擎，有效缓解了仿真数据与真实数据之间的权衡问题。 |
| [^129] | [Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More](https://arxiv.org/abs/2609.30768) | 思考过程对反事实公平性具有非对称的双重效应——虽然能解决部分非思考状态下的偏见翻转，但制造的新偏见翻转数量约为解决数量的5倍。 |
| [^130] | [Insurance Reserve Intelligence Platform](https://arxiv.org/abs/2609.30765) | 本文提出了一个将经典Thiele微分方程求解器与知识信息增强的物理信息神经网络（KINN-PINN）相结合的保险准备金智能平台，在保持可解释性的同时大幅提升了定期寿险准备金计算的效率。 |
| [^131] | [HCOE: Hyperbolic Clinical Ontology Embeddings from Biomedical Language Models](https://arxiv.org/abs/2609.30763) | HCOE 通过将冻结的 BioBERT 嵌入映射到双曲庞加莱空间，并结合本体引导的对比学习与由粗到细的路径聚合，构建了保留医学代码层级结构的临床概念表示，在临床关系预测及死亡率、再入院、药物推荐等多项临床任务上均达到最佳性能。 |
| [^132] | [Selective Amortization of Full-Budget Counterfactual Reasoning for Visual Token Communication](https://arxiv.org/abs/2609.30756) | 本文提出ACV-Gate自适应候选评估框架，通过学习近似全预算反事实评估，仅对最有信息量的候选令牌进行精确评估，在保证重建质量的同时大幅降低生成式图像通信中令牌选择的编码端计算开销。 |
| [^133] | [Backbone-Adaptive Evidence Routing for Robust Pairwise LLM Judging](https://arxiv.org/abs/2609.30751) | 提出骨干自适应证据路由方法BAER，在保持候选对称性的前提下为不同基准和评判骨干自适应选择证据收集机制，在全部八个测试条件下均取得最高准确率，较最强基线提升0.87至7.32个百分点。 |
| [^134] | [ORCA: Evaluating LLMs on Data Science Code Translation](https://arxiv.org/abs/2609.30749) | 提出了ORCA综合基准，通过1,600个基础级任务和200个项目级翻译任务，首次系统评估了大语言模型在数据科学代码翻译（DSCT）这一未被充分研究领域的表现。 |
| [^135] | [Beyond the Last Truffula Tree: SustainAI - A Water-Aware, Closed-Loop Framework for Environmentally Accountable AI](https://arxiv.org/abs/2609.30747) | 该论文提出SustainAI闭环框架，通过实时水量计量、幻觉感知惩罚模型和考虑区域水资源压力的路由算法，将水资源消耗纳入AI部署的环境问责体系，并揭示不同地区数据中心每次推理的水足迹差异高达11倍。 |
| [^136] | [Anatomy-Aware Dexterity-Driven Design Optimization of Surgical Continuum Robots](https://arxiv.org/abs/2609.30745) | 该论文提出一种兼顾灵巧性与解剖结构的手术连续体机器人设计优化方法，引入RVDSA指标，结合高效运动规划器和渐近最优的模拟退火优化器，成功优化了用于结肠息肉手术的双臂灵巧鞘管机器人设计。 |
| [^137] | [From S3Q Theory to Implementation: Towards an Architecture for Machine Qualia](https://arxiv.org/abs/2609.30743) | 本文为S3Q意识理论提出了一个五层实现架构，将三个意识必要条件（感知运动情境性、世界模型内部模拟、预测与观察的结构连贯性）映射到具体的计算机制并整合为单一表示流水线。 |
| [^138] | [Learning What to Skip: Counterfactual Credit Assignment for Efficient Multi-Agent LLM Workflows](https://arxiv.org/abs/2609.30734) | 提出LW2S框架，通过反事实信用分配学习各组件的跳过安全模型，在多智能体LLM工作流中智能省略不必要的步骤，从而在保持或提升任务性能的同时显著降低token开销。 |
| [^139] | [Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents](https://arxiv.org/abs/2609.30725) | 该论文首次系统研究编码智能体中的成本低效行为，识别出子集化检索、相似脚本生成与测试重复执行三种行为（影响79%–98%的任务、最高占任务成本的22.75%），并评估了结构感知检索、智能体自合成技能与开发者设计技能三种缓解策略的效果。 |
| [^140] | [TrafficImag: A Benchmark for Counterfactual Roadside Traffic Video Generation](https://arxiv.org/abs/2609.30722) | 本文提出首个反事实路边交通视频生成基准TrafficImag，通过大规模标注数据集与可执行干预协议，将干预表示为交通参与者级别的程序，实现对行为推理、图像编辑和条件视频生成等异构基础模型的统一评估。 |
| [^141] | [Werracle: Sub-Cent Intra-Block AI Reflex Oracles and Flash-Loan Circuit Breakers for EVM Smart Contracts](https://arxiv.org/abs/2609.30719) | 该论文提出Werracle，一种零存储、可装入单个32字节EVM存储槽的链上AI决策预言机，以亚美分的成本实现区块内即时响应的AI反射预言机与闪电贷熔断器，从而克服ZK-ML证明延迟过高、无法应对单区块内原子性DeFi攻击的瓶颈。 |
| [^142] | [Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4](https://arxiv.org/abs/2609.30716) | 该研究通过完全平衡的实验设计评估 Gemma 4 模型在接收冲突文档时的行为，发现信息源的语义表述（如“官方指南”或“最新更新”）对模型最终答案的影响显著强于文档的呈现顺序。 |
| [^143] | [CRC-Router: Risk-Constrained Routing for Medical Agentic AI Systems](https://arxiv.org/abs/2609.30714) | 提出CRC-Router，一种基于保形风险控制的风险约束、不确定性感知路由模块，可为医疗AI系统判断何时能自主处理病例、何时需升级人工审查，从而有效控制错误接受风险并保障临床部署安全。 |
| [^144] | [VLALight: Lightweight Vision-Language-Action Models for Emergency-Aware Traffic Signal Control](https://arxiv.org/abs/2609.30709) | 提出了VLALight，一个轻量级端到端视觉-语言-动作框架，通过融合多方向摄像头视图并将交叉路口观测与信号相位信息直接映射为离散信号动作，仅用0.5B参数的紧凑模型实现了应急感知的交通信号控制。 |
| [^145] | [Combining General and Domain-Specific Pretext Tasks for Brain MR Image Segmentation](https://arxiv.org/abs/2609.30708) | 提出了一种联合优化体素级脑龄预测与图像修复的多任务自监督预训练框架，通过结合领域特定和通用前置任务学习互补的神经影像表示，以提升脑部MR图像分割性能。 |
| [^146] | [LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information](https://arxiv.org/abs/2609.30706) | LAVOIR将候选缺失信息槽位与答案选项一同置于输入中，使单次前向传播同时输出决策分布和各槽位的信息价值预期增益，从而以无需人工标注的方式教会决策模型何时提问、问什么。 |
| [^147] | [The Price of Thought: Does Test-Time Reasoning Pay in LLM Trading?](https://arxiv.org/abs/2609.30705) | 该论文首次将LLM的推理控制作为经济干预进行评估，通过对DeepSeek、GPT和Gemini三大模型系列在一年美国股市上的80余万次预测开展对照实验，发现增加测试时推理的计算投入并不能可靠地提升扣除交易成本后的净投资组合回报。 |
| [^148] | [SAGE: Source-Anchored Guidance via Frequency Equalization for Hierarchical RGB-T Alignment and Fusion](https://arxiv.org/abs/2609.30703) | 提出统一框架SAGE，通过可逆联合编码、源锚定低频调制、分层频率协同对齐与引导子带融合，端到端联合解决RGB-T融合中的空间失配与跨模态差异问题。 |
| [^149] | [Threat-Aware Energy-Efficient Deployment for Dynamic UAV Networks: A Multi-Agent RL Approach](https://arxiv.org/abs/2609.30690) | 提出了一种威胁感知的无人机网络节能部署三步框架，结合威胁感知K均值聚类、最优匹配和MATD3多智能体强化学习，在实现零安全违规的同时最大化全局能效并加速收敛。 |
| [^150] | [LLM Parkinsonism: Executive-Control Failure, Token-Inefficient Persistence, and an Uncertainty-Aware Global Executive Control Architecture for Autonomous Language-Model Agents](https://arxiv.org/abs/2609.30662) | 该论文提出“LLM帕金森综合征”这一非临床隐喻，指出大语言模型代理在目标完成后仍持续低价值行动的根源在于生成、评估与停止权限集中于同一自条件循环，并提出不确定性感知的全局执行控制架构GEC v0.2以实现动作生成与项目级控制的分离。 |
| [^151] | [Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation](https://arxiv.org/abs/2609.30650) | 本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。 |
| [^152] | [A Framework for Identifying, Categorizing, and Explaining Bias in AI-Generated Code](https://arxiv.org/abs/2609.30642) | 本研究提出了一个基于分类体系的框架，用于识别、分类和解释AI生成代码中的偏见，并构建真实标准数据集评估了各类LLM作为自动化偏见检测与解释系统的可靠性。 |
| [^153] | [Audio LLMs Know When They Can't Hear You](https://arxiv.org/abs/2609.30625) | 该论文发现音频大语言模型无法通过自我评估或现有方法（如语音质量预测器、生成不确定性等）有效判断自身语音转录的可靠性，但转录可靠性信息强烈编码在模型音频编码器的内部表示中，可用于检测转录失败。 |
| [^154] | [Subjects, Not Authors: The Authorship Hazard in Agentic Dataspaces](https://arxiv.org/abs/2609.30614) | 该论文提出“作者身份危害”概念并确立核心原则：LLM智能体应始终是数据空间治理平面的主体而非作者，其发布授权通道须在构造上被关闭，其起草内容的影响则作为执行问题加以管控。 |
| [^155] | [MedTokenBudget: Lesion-Preserving Token Routing for Dermoscopic Image Classification](https://arxiv.org/abs/2609.30613) | 本文提出MedTokenBudget，一个基于视觉Transformer的有监督token路由框架，利用病灶感知Token评分（LATS）模块在给定token预算下优先保留与病灶相关的图像块，从而为皮肤镜图像分类构建紧凑且富含病灶信息的表示。 |
| [^156] | [The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge](https://arxiv.org/abs/2609.30604) | 该论文提出了KNOWS基准，通过开放式、复杂的浏览器任务联合评估网络智能体在信息检索、知识综合、任务分解以及产出文档等最终制品方面的综合能力。 |
| [^157] | [Action Forcing: Training World Models on Unsupervised Video by Recovering Underlying Egomotion Bases](https://arxiv.org/abs/2609.30595) | 该论文提出"Action Forcing"方法，无需动作标注或训练，仅通过追踪视频帧间像素位移并用PCA提取自运动基，即可将普通无标注视频转化为带真实控制信号（油门-偏航）的动作监督数据，用于训练可控世界模型。 |
| [^158] | [T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation](https://arxiv.org/abs/2609.30576) | 提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。 |
| [^159] | [HARDEN: Constrained Evolutionary Search for Harder, Answer-Preserving Evaluation Cases](https://arxiv.org/abs/2609.30571) | HARDEN是一种约束进化搜索方法，能在保持预期答案不变的前提下将现有评估案例改编为更难的变体，使语言模型准确率平均下降22.7%、最高下降49.9%。 |
| [^160] | [Atelier: Learning Local Self-Supervised Features for CryoEM Volumes via Hypernetworks](https://arxiv.org/abs/2609.30569) | Atelier是一个基于Transformer超网络的自监督框架，通过摊销冷冻电镜图谱的隐式神经表示拟合，实现了高效、跨样本对齐的局部特征学习，可用于大规模冷冻电镜体积的特征提取。 |
| [^161] | [Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content](https://arxiv.org/abs/2609.30563) | 本研究通过八名真实用户画像与六十八条帖子反应的对照实验发现，采用直觉式（少推理）提示并以态度性内容而非人口统计背景构建用户画像，可显著提升大语言模型智能体模拟个体社交媒体真实反应的准确性（包括对陌生内容的反应），且智能体与画像的一致性并不等同于行为保真度。 |
| [^162] | [Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms](https://arxiv.org/abs/2609.30558) | 本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。 |
| [^163] | [Auditing Latent-Space Monitors for Autonomous Driving](https://arxiv.org/abs/2609.30557) | 该论文审计了自动驾驶中基于潜在空间探针的运行时故障监控方法，发现尽管内部表征能够有效预测故障（LaneSegNet和VAD的AUROC分别达0.780和0.868），但仅使用模型输出、自车状态和驾驶指令等外部信息即可达到相当甚至更好的性能（0.825和0.924），表明访问模型内部表征并非实现强大故障预测的必要条件。 |
| [^164] | [Proportional Representation in Temporal Voting with Ranked Preferences](https://arxiv.org/abs/2609.30555) | 该论文将比例代表制公理（JR、PJR、EJR、PSC）拓展到具有排序偏好的时间投票中，构建了公理层次结构，并发现与批准投票不同，没有任何版本的扩展正当代表制（EJR）是可以被保证的。 |
| [^165] | [Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study](https://arxiv.org/abs/2609.30553) | 提出TGL-NSGA-II框架，通过教师引导的轻量知识蒸馏生成排序可靠的低保真适应度分数，并与高斯过程代理模型融合，以显著降低昂贵TinyML神经架构搜索中进化优化的评估成本。 |
| [^166] | [Benchy: towards a universal language for task-oriented AI benchmarks](https://arxiv.org/abs/2609.30550) | Benchy提出了一种用于AI基准测试的语义语言和执行引擎，通过规范化YAML语法与通用运行时契约将基准测试定义与AI系统解耦，旨在建立面向任务AI基准测试的通用语言。 |
| [^167] | [Convergence guarantees for Muon: New parameter regimes and generalizations](https://arxiv.org/abs/2609.30546) | 本文首次证明了Muon优化算法的渐近收敛性，通过更精确的Newton-Schultz迭代代理揭示其本质是隐含正则化诱导的有界预条件子所构成的预条件Polyak重球方法，并据此提出了具有相同收敛保证的Nesterov变体Muesterov。 |
| [^168] | [Inquesto Score: A reliability Protocol For Voice Agents](https://arxiv.org/abs/2609.30514) | 本文提出 Inquesto Score（IS）协议，通过明确定义失败事件与严重级别，以固定版本化评估群体中成功完成呼叫者目标且无功能性故障的通话占比，来可复现、可解释地衡量已部署语音智能体的可靠性。 |
| [^169] | [PolicyAttention: Softmax Attention Implements Policy Mirror Descent for Closed-Loop Control](https://arxiv.org/abs/2609.30500) | 该论文构造了一个带显式残差的因果softmax“行动者—环境—单步评论家”协议，证明softmax注意力可以作为重复控制器实现策略镜像下降，并用预注册实验验证了训练的pre-LN Transformer能够恢复目标计算。 |
| [^170] | [Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs](https://arxiv.org/abs/2609.30492) | 该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。 |
| [^171] | [BioEVAL: A global, multi-institutional benchmark of large language and multimodal models for bioengineering](https://arxiv.org/abs/2609.30489) | 该论文提出了BioEVAL——首个由22个研究团队共同构建、覆盖11个生物工程子领域的博士级全球多机构基准，通过608个评估项目（多项选择题、文献综合任务和多模态实验图像解读）来评估大语言模型与多模态模型在生物工程前沿实验推理中的能力。 |
| [^172] | [Do LLMs Understand Context? A Knowledge Graph-Based Evaluation Framework](https://arxiv.org/abs/2609.30484) | 提出了一种基于知识图谱的评估框架，通过语义结构相似度衡量大语言模型在问答任务中真正的上下文理解能力，弥补了BLEU和困惑度等传统指标只能评估表面性能的不足。 |
| [^173] | [CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production](https://arxiv.org/abs/2609.30471) | 针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。 |
| [^174] | [Pretrained ASR Pseudo-labeling for Noisy Police Audio](https://arxiv.org/abs/2609.30469) | 该论文发现现有内部置信度指标无法有效过滤嘈杂警用广播通信音频的伪标签，并提出利用大语言模型作为评判者的外部过滤范式，显著降低了适配后ASR模型的词错误率。 |
| [^175] | [A Benchmarking Framework for Context-aware XR Interfaces](https://arxiv.org/abs/2609.30466) | 该论文提出ContextXR——首个面向情境感知XR界面的基准测试框架，通过功能面连通图表示、带功能面级标注的MineXR++数据集以及三个规范化的建议任务，实现了对XR自适应方法的系统化、可重复评估。 |
| [^176] | [Spectral Feedback for Test-Time Alignment of Protein Diffusion Models](https://arxiv.org/abs/2609.30456) | 提出谱反馈算法，通过在反馈回路中选择编辑位置并对词元重新掩码与重采样，使蛋白质离散扩散模型能在测试时迭代纠正自身生成结果，将“重新审视哪些词元”而非“如何分配词元标签”作为对齐的核心问题。 |
| [^177] | [Predicting Transmembrane Protein Topology from 3D Structure](https://arxiv.org/abs/2609.30446) | 该论文提出利用图神经网络SchNet直接从三维结构的全原子嵌入中预测跨膜蛋白拓扑结构，在无需预训练权重的情况下展现出优异的预测潜力。 |
| [^178] | [Actively Resolving Contextual Uncertainty for Underspecified Tasks in Natural Language](https://arxiv.org/abs/2609.30428) | 本文提出CLUE框架，使机器人面对欠明确的自然语言任务时，能通过LLM推导的策略假设任务相关概念与潜在计划，并结合在线构建的语言嵌入地图，以闭环方式主动消解上下文不确定性。 |
| [^179] | [Understanding Perturbed Parameter Ensemble Sensitivities Using A Contrastive Learning Approach](https://arxiv.org/abs/2609.30420) | 本文开发了一种可解释的对比学习模型，将CAM6扰动参数集合的多变量云与辐射场映射到共享表示空间，能以超过94%的准确率区分不同暖雨微物理方案，同时保留季节变率与参数扰动引起的集合离散度，为气候模式参数敏感性的解释与校准提供了新方法。 |
| [^180] | [A Unified Account of Concepts and Chunks](https://arxiv.org/abs/2609.30414) | 本文提出将认知心理学中原本分离的概念与组块统一到一个理论框架中，扩展Cobweb分类模型并实现TRELLIS系统，成功应用于同时包含概念性和组块性元素的上下文无关文法学习。 |
| [^181] | [What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study](https://arxiv.org/abs/2609.30402) | 本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。 |
| [^182] | [A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods](https://arxiv.org/abs/2609.30397) | 该论文提出了一种基于合成真值的可解释AI评估框架，通过受控干预生成已知输入重要性的合成数据集，解决了传统保真度评估无法验证解释是否真正反映模型底层决策过程的问题。 |
| [^183] | [Stealth Apart, Harm Together: Skill Cascading Attacks on Skill-Based Agent Systems](https://arxiv.org/abs/2609.30383) | 本文提出“技能级联攻击”这一新威胁范式，通过将恶意目标分散到多个各自看似无害的技能中，利用技能间的组合执行对基于技能的智能体系统发起隐蔽攻击。 |
| [^184] | [DanLing NestedTensor: Composable Multi-Ragged Tensors for Deep Learning](https://arxiv.org/abs/2609.30379) | DanLing NestedTensor 是一种将多重参差结构内嵌为张量自身属性的 PyTorch 张量抽象，使广播、特征变换和归约操作能够可组合地处理变长数据，在 BERT 任务上相比填充方法实现了最高 3.39 倍的加速。 |
| [^185] | [Cost-Aware Best-LLM Identification using Dueling Feedback](https://arxiv.org/abs/2609.30360) | 该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。 |
| [^186] | [Strategic Self-Consistency](https://arxiv.org/abs/2609.30352) | 本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。 |
| [^187] | [Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices](https://arxiv.org/abs/2609.30348) | 该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。 |
| [^188] | [Coding Agents Aren't Enough! Evaluating an Enterprise Security Brain for Agentic Cloud Investigations](https://arxiv.org/abs/2609.30345) | 该论文提出并评估了Sola Security Brain这一专门构建的企业安全智能层，通过28个云安全调查任务的对比实验证明，相比仅凭只读凭证直接调查实时AWS环境的通用编码智能体（Claude Code），专门设计的安全上下文层在总体性云安全调查中能提供更完整、更准确的结果。 |
| [^189] | [Bridging LLM Agents and Data Spaces: An Architectural Mediation Approach using the Model Context Protocol](https://arxiv.org/abs/2609.30341) | 该论文提出了一种基于模型上下文协议（MCP）的架构中介方法，通过将数据空间能力转换为结构化、模式驱动的工具，使LLM智能体能够在保留治理约束的前提下与数据空间服务进行受控交互，且无需修改现有数据空间组件。 |
| [^190] | [What Will Remain Human in Software Architecture? A Focus Group Report](https://arxiv.org/abs/2609.30334) | 本研究通过EuroPLoP 2026上的焦点小组探讨AI开发智能体对软件架构实践的影响，发现架构决策、问责制和架构护栏编写仍是不可替代的人类核心职责，并提出“驾驭工程”这一新兴概念——即构建用于治理AI辅助系统创建的系统的学科。 |
| [^191] | [When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess](https://arxiv.org/abs/2609.30328) | 该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。 |
| [^192] | [ScopeBench: Do Agents Preserve Engagement Boundaries Under Goal Pressure?](https://arxiv.org/abs/2609.30325) | 提出 ScopeBench 基准，通过 30 个“目标只能靠越界才能达成”的死胡同式安全任务，并在有无范围约束的配对条件下，区分并衡量智能体的能力与目标压力下遵守授权范围的意愿。 |
| [^193] | [Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops](https://arxiv.org/abs/2609.30297) | Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。 |
| [^194] | [SignTrace: Describe a Sign, Find the Word](https://arxiv.org/abs/2609.30295) | SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。 |
| [^195] | [SlideLab: Audience-Centered Scientific Slide Generation and Evaluation](https://arxiv.org/abs/2609.30294) | SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。 |
| [^196] | [Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents](https://arxiv.org/abs/2609.30293) | Cartograph是一个联邦式MCP代理，通过操作者签名的能力卡片、三层易混淆聚类分析（Rift）以及“先服务器后工具”的两阶段检索，将AI智能体的工具发现从O(n)目录遍历优化为O(k)渐进式披露，仅需暴露3个代理工具即可在374个工具的部署中取得0.816的R@5召回率，显著优于关键词基线。 |
| [^197] | [A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models](https://arxiv.org/abs/2609.30292) | 本综述从信息融合视角系统梳理了2018年至2026年初的211项虚假评论检测研究，按证据来源和融合层次组织现有工作，并分析了预训练语言模型和大语言模型对虚假评论的生成与检测带来的双重影响。 |
| [^198] | [Bringing AI to Autonomous Systems -- From Cognition to Collective Intelligence](https://arxiv.org/abs/2609.30291) | 本文提出了一个基于通用智能体架构的自主系统设计与评估综合框架，通过结合联结主义与符号主义AI，并整合个体认知与集体智能，推动AI向自主系统这一最终发展阶段迈进。 |
| [^199] | [A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID](https://arxiv.org/abs/2609.30287) | 该研究通过稀疏探测和双向激活修补方法，在冻结的BERT中定位出一组占比不足1%的神经元，证明它们因果性地支持跨六个生成器的AI生成文本检测任务。 |
| [^200] | [When Does Advection-Aware Graph Nowcasting Help? A Controlled Study of Distributed Solar Ramp Forecasting with a Self-Supervised Cloud-Motion Estimator](https://arxiv.org/abs/2609.30286) | 受控合成实验表明，平流感知图神经网络对分布式光伏坡升临近预报的增益有限——在现实的互相关CMV估计下并不优于静态或学习邻接的时空GNN，完美CMV约一半收益来自将运动矢量作为输入特征而非图结构本身，且只有当预报时域内平流位移落在传感网络范围内时平流信息才真正有用。 |
| [^201] | [ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers](https://arxiv.org/abs/2609.30272) | ENAS是一个无需GPU即可高效运行的硬件感知神经架构搜索框架，通过静态可行性检查、支持多种块的单元搜索空间和三阶段混合搜索策略，在资源受限的微控制器上实现了TinyML模型的快速搜索。 |
| [^202] | [AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control](https://arxiv.org/abs/2609.30264) | 提出动作判别式世界模型AD-WM，通过逆动力学和基于条件互信息的动作恢复正则化，使潜在世界模型能更好区分候选动作以支持反事实模型预测控制，在OGBench-Cube上将困难起始成功率从3.7%提升至52.0%。 |
| [^203] | [PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations](https://arxiv.org/abs/2609.30094) | 提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。 |
| [^204] | [PUBG Ally: A Conversational Embodied Agent as an AI Teammate](https://arxiv.org/abs/2609.29837) | 该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。 |
| [^205] | [Detecting Glaucoma Across Multi-ethnic Myopic and Non-Myopic Populations Using an Uncertainty-Aware Vision Transformer: A Multicentre Model Development and Validation Study](https://arxiv.org/abs/2609.29433) | 该研究开发了带有不确定性估计的Vision Transformer深度学习模型，在涵盖多民族、近视与非近视人群的三大洲16个外部数据集上实现了稳健且高性能的青光眼检测。 |
| [^206] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^207] | [ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts](https://arxiv.org/abs/2609.29349) | ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。 |
| [^208] | [AI in Science: Early Insights](https://arxiv.org/abs/2609.28504) | 该论文通过分析1500万次Gemini交互、2600多个专业AI模型和600多名科学家的调查数据，首次提供了科学家使用AI的大规模实证证据，发现AI在科学界已被广泛采用，且LLM与专业模型互为补充而非替代。 |
| [^209] | [NV-Reason-CT: 3D Visual Language Model for CT Analysis](https://arxiv.org/abs/2609.27511) | NV-Reason-CT通过原生3D视觉Transformer将全部视觉标记及其显式3D坐标直接传入语言模型解码，在基于7万余例CT、约55万条专家标注引导的多模态指令数据上训练，实现了保留完整体积空间信息的胸部和腹部CT智能推理分析。 |
| [^210] | [Reinforcement Learning with Decomposed Subtasks](https://arxiv.org/abs/2609.27035) | 该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。 |
| [^211] | [G\"odel's and Scott's Variants of the Ontological Argument in Lean 4](https://arxiv.org/abs/2609.26806) | 该论文将哥德尔与斯科特本体论论证的 Isabelle/HOL 形式化数据集完整且保结构地移植到 Lean 4，验证了全部 548 条陈述的一致性，并重新证明了包括模态坍缩、一神论等在内的所有原开发中被证明的结论。 |
| [^212] | [Towards Hierarchical GNNs for multi-grid power flow: generalization across operating scenarios](https://arxiv.org/abs/2609.26603) | 该论文提出在GENCO校正网络中引入分层潜在通信模块，通过Kron简化和Quotient构建两种简化图在不同电网间交换信息，显著提升了多电网潮流GNN模型对未见运行场景的泛化能力，其中Kron方法将电压误差降低了85.0%。 |
| [^213] | [Geometry-Aware Hyperbolic Residual Quantization](https://arxiv.org/abs/2609.26342) | 提出一种几何感知的双曲残差量化方法，通过双曲残差聚合恢复前向传播中庞加莱圆盘上的伸缩求和特性，并利用带折扣的双曲直通估计器在反向传播中保留几何信息，从而解决双曲空间残差量化的几何不一致问题。 |
| [^214] | [The Uncontrolled Variable: Vision-Language Refusal Is Conditioned on the Image-Attachment Interface, and Not Robust to Irrelevant Image Properties](https://arxiv.org/abs/2609.26174) | 该研究揭示视觉语言模型的拒绝行为取决于请求是否附带图像——即使附加的是空白画布等完全无关的图像——这种阈值偏移使边缘良性的敏感话题提问拒绝率上升23至51个百分点，而中性指令几乎不受影响。 |
| [^215] | [Hill Sampling for Test-Time Scaling: A Simple and Better Alternative to Repeated Sampling, Evolution, and Training](https://arxiv.org/abs/2609.25510) | 该论文提出了一种简单的“山峰采样”方法——从冻结的大语言模型中反复采样候选程序编辑并保留当前最优程序作为后续采样的条件，无需复杂的进化搜索或测试时训练，就在圆填充问题上创下新的最先进水平，并在Erdős最小重叠问题上超越了AlphaEvolve。 |
| [^216] | [You've Seen Enough: Quality-Constrained Image Coding for Machines](https://arxiv.org/abs/2609.25108) | 该论文提出一种面向机器的质量约束图像编码方法，将人类视觉质量限制在预设目标水平，并通过绝对值和双线性两种惩罚函数设计，把剩余比特容量全部分配给机器视觉任务，从而在满足人眼检查需求的同时最大化机器任务性能。 |
| [^217] | [Lifted Bellman Linear Programming for Offline Reinforcement Learning](https://arxiv.org/abs/2609.24489) | 提出提升贝尔曼线性规划（LBLP），通过将贝尔曼最优性的线性规划刻画提升到联合(Q,V)空间，用仅涉及数据集中状态-动作对的不等式约束实现样本内贝尔曼最优性，其唯一最优解在确定性动力学下介于数据集最佳回报与最优价值之间。 |
| [^218] | [MM-ContextFold: Context Folding for Multimodal Agentic Retrieval](https://arxiv.org/abs/2609.23121) | 提出了无需训练的 MM-ContextFold 框架，基于对约一万条轨迹的实证发现——当视觉信息被提取并文本化后原始图像变得冗余——通过“折叠”冗余视觉内容来解决多模态智能体检索中的上下文爆炸问题。 |
| [^219] | [Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods](https://arxiv.org/abs/2609.21815) | 本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。 |
| [^220] | [Higher-order pruning of experts in mixture-of-experts language models](https://arxiv.org/abs/2609.18916) | 提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。 |
| [^221] | [A Vision-Language Foundation Model for Precise and Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data](https://arxiv.org/abs/2609.16597) | BrainVLM是一种视觉-语言基础模型，能够基于术前多模态MRI数据对12种WHO 2021脑肿瘤类型进行自动精准分类，并同时提供诊断不确定性量化和放射学报告生成功能，解决了传统MRI诊断中影像特征重叠和观察者差异的难题。 |
| [^222] | [ProtoLIP: From Sentence-Level to Object-Level Evidence Disentanglement](https://arxiv.org/abs/2609.16284) | 提出ProtoLIP，一种轻量级的原型介导证据层，通过将视觉原型组织为文本语义族并进行查询相关的族路由，在无需空间标注或骨干网络重训练的情况下，实现从句子级到对象级的证据解耦，显著提升视觉-语言模型的证据定位与分离能力。 |
| [^223] | [State of Thought Enables Endogenous Reasoning](https://arxiv.org/abs/2609.16055) | 提出“思维状态”新推理范式，通过从模型内部信息传递中提取动力学-几何状态并用仅 582 参数的控制器在冻结模型上选择性激活历史推理支持，实现由模型内部状态主导的内生推理，摆脱外部控制。 |
| [^224] | [Interpreting hierarchical organisation of speaker embeddings](https://arxiv.org/abs/2609.15203) | 本文从可解释人工智能（XAI）的视角出发，利用SLINK层次聚类算法分析说话人嵌入是否自然形成层次化聚类结构，并提出了一种新的层次聚类-类别匹配评估方法来解释说话人嵌入的层次化组织。 |
| [^225] | [T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing](https://arxiv.org/abs/2609.15160) | 该论文提出T-LoopFormer，通过动态token选择路由机制，让循环Transformer中的每个token根据自身隐藏状态自适应地决定循环迭代次数，从而实现更优的计算分配和效率提升。 |
| [^226] | [Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting](https://arxiv.org/abs/2609.12890) | 提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。 |
| [^227] | [Exploring Second-Order Pattern Recognition in Speaker Recognition](https://arxiv.org/abs/2609.11182) | 该论文提出利用层次聚类来发现说话人识别网络中将话语识别为说话人身份时潜藏的“二阶模式”，并通过HCCM方法对其语义解释，进而提出“二阶模式识别”这一新任务。 |
| [^228] | [EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale](https://arxiv.org/abs/2609.10980) | 本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。 |
| [^229] | [High-probability guarantees for linear accessibility in feature superposition](https://arxiv.org/abs/2609.09556) | 该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。 |
| [^230] | [Steering Interference Reflects the Model's Defaults, Not the Behavior Directions](https://arxiv.org/abs/2609.06951) | 激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。 |
| [^231] | [The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists](https://arxiv.org/abs/2609.06934) | 该论文从几何视角证明，事后安全训练（如RLHF）的更新与模型能力方向近乎正交，只是在完好的能力之上叠加一道薄而尖锐的拒答“闸门”而非真正删除能力，因此注定会被越狱等攻击绕过，而持久的安全必须在预训练阶段扎根。 |
| [^232] | [Provably Safe Sim-to-Real Transfer](https://arxiv.org/abs/2609.01418) | 该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。 |
| [^233] | [SUN: Persistent Programs For Language-Grounded Control-to-Learning-to-Real Policies](https://arxiv.org/abs/2608.31167) | 该论文提出语义统一（SUN）程序，将几何与接触关系定义一次即编译为对齐的MPC代价、满足性谓词、RL奖励与转移守卫，并由“夸父”系统从语言和场景语义自动合成程序、经MPC筛选可行性后训练阶段条件化策略，在九个长时程操作任务上取得82.03%的成功率，显著超越稀疏奖励等基线方法。 |
| [^234] | [Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable](https://arxiv.org/abs/2608.27167) | 本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。 |
| [^235] | [Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings](https://arxiv.org/abs/2608.26088) | 行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。 |
| [^236] | [PhysElite: How Far Are LLMs from Solving Olympiad-Level Physics Problems?](https://arxiv.org/abs/2608.25097) | 提出了PhysElite，一个包含11,586道奥赛级物理问题、附视觉图表和中英双语分步解答的大规模双语多模态基准，并借此评估了18个开源与闭源多模态大语言模型的物理推理能力。 |
| [^237] | [PolyChirp: Multi-Species Birdsong Classification Using TinyML on Low-Power Acoustic Sensors](https://arxiv.org/abs/2608.23101) | PolyChirp通过结合生物专业知识、自动化数据集和NPU加速的微型多类模型，首次实现了低功耗微控制器上多物种鸟类鸣声的实时分类。 |
| [^238] | [Governance Records as Supervision: Verifier-Selected Self-Training for Structured Workflow Repair](https://arxiv.org/abs/2608.18324) | 本研究提出一种利用机器可验证工作流产生的治理记录进行自我训练的方法，使有限模型通过验证者选择的计划提升一次性执行能力，显著提高成功率和效率。 |
| [^239] | [Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection](https://arxiv.org/abs/2608.17965) | 本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。 |
| [^240] | [Admission Without Answers: Label-Free Certification and Experience Learning for LLM-Based Optimization Modeling](https://arxiv.org/abs/2608.15565) | 本文提出AdmitOR，一种基于校准外部行为证据的无标签准入门控方法，用于LLM优化建模中的经验学习，以解决无答案流中知识接纳不可靠的问题。 |
| [^241] | [Keep the Future, Drop the Rollout: RIFT for World Action Models](https://arxiv.org/abs/2608.11521) | 本文发现世界动作模型可重用固定未来缓存而非迭代展开，提出RIFT方法，在保持高成功率的同时大幅降低部署延迟。 |
| [^242] | [Beyond Forecasting: Recasting Volatility Control as a Routing Problem](https://arxiv.org/abs/2608.10375) | 本文提出VolRouter框架，创新性地将波动率控制重构为基于市场状态的条件路由问题，在估计器-控制器组合对之间动态切换，在四个基准测试中的三个取得最高夏普比率，并将标普500的夏普比率从0.952提升至1.22。 |
| [^243] | [Towards Unified Dynamic Face Landmark Detection](https://arxiv.org/abs/2608.10346) | 该论文提出人脸部位锚定关键点位置（FPALP）表示法，将关键点统一表示为人脸部位轮廓上的进度值，从而实现所有N点数据集的统一训练和动态数量的关键点输出。 |
| [^244] | [Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks](https://arxiv.org/abs/2608.07335) | 本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。 |
| [^245] | [Decoupling Intention from Trajectory: A Representational Deduction Framework for World Action Models](https://arxiv.org/abs/2608.06994) | 该论文提出PILOT框架，通过将运动思维链引导作为模型原生能力的“表征推演”机制，解耦世界动作模型中高层物理状态演化与低层动作轨迹生成之间的表征纠缠，从而增强世界演化建模对动作生成的预测与指导能力。 |
| [^246] | [Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation](https://arxiv.org/abs/2608.04408) | 本文提出反事实可恢复性框架，通过教师续写与回滚分支重放错误状态来区分可恢复与不可逆的错误，并据此决定在线策略蒸馏中对轨迹的保留、回滚或常规监督策略，其可恢复性代理指标AUC达1.000，远超仅依赖发散度指标的0.392。 |
| [^247] | [State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking](https://arxiv.org/abs/2608.03425) | 提出复数状态传播器（CSP），一种仅依赖atan2激活在复相位流形上运行的极简循环范式，实现相位信号在极深网络中的零衰减传播，从而解决Transformer和Mamba在确定性状态追踪中的分布外崩溃问题。 |
| [^248] | [Energy Efficiency of Locally Deployed LLMs: A Preliminary Quantitative GPU Power Benchmark on Consumer Hardware](https://arxiv.org/abs/2608.00008) | 本文在消费级GPU上对18个开源大语言模型进行了可复现的硬件级能耗基准测试，发现模型架构和量化策略（而非仅仅参数量）才是决定能效的关键因素，并给出了各模型的单位token能耗与吞吐量排名。 |
| [^249] | [Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters](https://arxiv.org/abs/2607.29238) | InMyStyle提出一种隐私优先的单用户方案，仅对0.5B至7B的小型模型进行LoRA微调，即可让AI编辑的文本自动改写为符合个人写作风格，且实验表明小模型已足以胜任该改写任务。 |
| [^250] | [Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents](https://arxiv.org/abs/2607.26865) | TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。 |
| [^251] | [Depth, Not Breadth: Best-of-N Jailbreaking Beyond Surface Noise](https://arxiv.org/abs/2607.26639) | 该研究将 Best-of-N 越狱攻击的查询预算从表面文本扰动转向结构性代码补全编码，对最强自检防御 SAGE 实现了高达各部分效果之和 9 至 75 倍的攻击成功率提升，证明攻击方差的结构性分布比表面噪声更重要。 |
| [^252] | [Blind, Not Weak: A Best-of-Suite Safety-Utility Frontier for Recover-and-Reguard Defenses Against Encoded VLM Jailbreaks](https://arxiv.org/abs/2607.26574) | 该论文构建了一个“恢复-重防”预处理器，在安全防护器之前恢复图像内容并解码编码，将图像渲染类越狱攻击的拦截率从零提升至67-90%，并据此刻画了此类防御在安全性与良性流量效用之间的套件级最优权衡边界。 |
| [^253] | [When Do Cheap Probes Predict Expensive Training? Probing 3D-CT Encoders for Text Generation](https://arxiv.org/abs/2607.22771) | 本文提出CheapCT廉价探针方法，证明其能高精度预测3D CT编码器微调后的生成性能，从而避免昂贵的全模型微调搜索。 |
| [^254] | [Do Neural Networks Preserve Case Structure? Case-Based Decomposition, Interpretation, and Decision Consistency](https://arxiv.org/abs/2607.11347) | 该论文在神经网络与基于案例的决策理论（CBDT）之间建立了联系，证明训练后的神经网络能够通过其学习到的表示保留可恢复的案例结构，从而将决策边际分解为单个训练案例的贡献，为模型决策是否真正根植于训练数据提供了可解释性与一致性验证的途径。 |
| [^255] | [CoFL-S: Spatially Queryable Sector Flow Fields for Local Language-Conditioned Navigation](https://arxiv.org/abs/2607.02222) | 提出 CoFL-S——一种在机器人局部可见扇区上预测语言条件流场并生成连续轨迹的底层视觉-语言-动作框架，配套将 VLN-CE 片段转换为帧级局部监督的训练方法，以及隔离底层动作接口的连续时间 Habitat 评测基准。 |
| [^256] | [Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark](https://arxiv.org/abs/2606.30170) | 提出纳米技术分子优化基准，用量子模拟取代代理预测器并引入严格协议，将生成式分子设计从药物发现领域拓展至量子材料科学与纳米技术研究。 |
| [^257] | [Efficient Safety Benchmarking via Item Response Theory](https://arxiv.org/abs/2606.20626) | 该论文提出将项目反应理论（IRT）与自适应选题方法应用于语言模型安全基准测试，能够恢复可解释的模型能力结构、区分原始指标上的天花板模型，并以远少于全量测试的响应数逼近完整基准的模型排序，大幅提升安全评估效率。 |
| [^258] | [Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts](https://arxiv.org/abs/2606.14929) | 该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。 |
| [^259] | [SciR: A Controllable Benchmark for Scientific Reasoning in LLMs](https://arxiv.org/abs/2606.13020) | SciR是一个基于形式化对象生成、答案可验证的科学推理基准，覆盖演绎、归纳与因果溯因三种推理范式，并能独立调节信息提取难度与推理难度两个维度，从而实现对大语言模型科学推理能力的可控评估。 |
| [^260] | [Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models](https://arxiv.org/abs/2606.10829) | 提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。 |
| [^261] | [INFUSER: Influence-Guided Self-Evolution Improves Reasoning](https://arxiv.org/abs/2606.09052) | INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。 |
| [^262] | [Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents](https://arxiv.org/abs/2606.05828) | 提出一种将统计偏好学习与语义意图解析严格解耦的轻量级本地框架，利用本地统计先验来调节远程LLM的技能选择决策，使个人智能体能够学习隐式用户偏好，并取得最低的累积遗憾和最高的测试准确率。 |
| [^263] | [Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring](https://arxiv.org/abs/2605.30834) | 提出Hide-and-Seek框架，将VLA失败检测建模为粗粒度监督学习问题，仅凭轨迹级标签通过轨迹间与轨迹内对比学习即可定位失败动作并生成时间结构化的失败信号，无需步级标注或昂贵的动作重采样与外部模型。 |
| [^264] | [CODESKILL: Learning Self-Evolving Skills for Coding Agents](https://arxiv.org/abs/2605.25430) | CODESKILL将编程智能体的技能提取与技能库维护建模为可学习的管理策略，通过强化学习和混合奖励，从任务轨迹中提取并演化多粒度程序性技能，从而实现智能体的自我进化。 |
| [^265] | [Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions](https://arxiv.org/abs/2605.19823) | 提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。 |
| [^266] | [Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models](https://arxiv.org/abs/2605.19398) | 论文揭示参考帧主导性（非参考帧对参考帧键token的过度自注意力）是图生视频模型运动受抑的关键原因，并提出免训练、模型无关的DyMoS方法，通过在初始去噪步骤重新平衡注意力通路来增强视频动态，且不损失对参考图像的保真度。 |
| [^267] | [Attention Sinks and Outliers in Attention Residuals](https://arxiv.org/abs/2605.17887) | 该论文提出OASIS方法，通过令牌级与深度级的显式空路由和空耦合机制稳定双归一化注意力残差架构，抑制注意力汇聚与激活离群值，并从理论与实验上解释和缓解了AttnResidual的低比特量化敏感性问题。 |
| [^268] | [Grid-Orch: An LLM-Powered Orchestrator for Distribution Grid Simulation and Analytics](https://arxiv.org/abs/2605.12728) | Grid-Orch通过MCP协议将大语言模型与OpenDSS配电网仿真相结合，提供36种领域工具，使工程师能用自然语言完成潮流计算、电压分析、QSTS仿真和自动化优化，并支持本地部署以满足电力系统安全隔离需求。 |
| [^269] | [Nice Fold or Hero Call: Learning Budget-Efficient Thinking under Policy-Dependent Solvability](https://arxiv.org/abs/2605.11625) | 该论文提出预算高效思考框架BET，将自适应推理建模为不确定性下的计算投资，使模型学会在“值得深入推理求解”与“果断放弃止损”之间做出决策，避免在模型能力之外的问题上浪费测试时计算资源。 |
| [^270] | [Selective Off-Policy Reference Tuning with Plan Guidance](https://arxiv.org/abs/2605.11505) | SORT通过从参考答案推导计划并加权提升计划条件下更可预测的token，将GRPO中全部采样失败的困难提示转化为结构感知的选择性学习信号，在八个推理基准上超越GRPO基线，对较弱模型的增益最大。 |
| [^271] | [AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment](https://arxiv.org/abs/2605.11398) | 该论文提出AcuityBench基准，通过统一五个公开数据集和四级急迫度框架，系统评估语言模型从用户医疗描述中识别就医紧急程度的能力，并纳入医生确认的模糊案例以衡量模型的不确定性对齐水平。 |
| [^272] | [Transformers Can Implement Preconditioned Richardson Iteration for In-Context Gaussian Kernel Regression](https://arxiv.org/abs/2605.08475) | 本文从理论和实验上证明，标准 softmax 注意力 Transformer 的前向传播可通过实现预条件 Richardson 迭代来近似高斯核岭回归预测器，其中注意力负责跨词元的核算子运算、MLP 负责词元内的标量算术，并以 O(log(1/ε)) 的深度达到 ε 精度。 |
| [^273] | [Agentick: A Unified Benchmark for General Sequential Decision-Making Agents](https://arxiv.org/abs/2605.06869) | Agentick是一个统一的序贯决策智能体基准，提供37个程序化生成任务，支持RL、LLM、VLM、混合及人类智能体在同一平台上的公平比较，大规模评估表明没有任何单一方法能全面占优。 |
| [^274] | [StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction](https://arxiv.org/abs/2605.06642) | StraTA 提出了一种策略性轨迹抽象框架，通过在智能体强化学习中引入显式的轨迹级策略并联合训练策略生成与动作执行，显著提升了大语言模型智能体在长时程决策任务中的样本效率和最终性能。 |
| [^275] | [Detecting Time Series Anomalies Like an Expert: A Multi-Agent LLM Framework with Specialized Analyzers](https://arxiv.org/abs/2605.05725) | SAGE是一个多智能体LLM框架，通过四个专门分析器对单变量时间序列异常进行基于证据的专家级诊断，并生成面向分析师的报告，在多个基准数据集上取得了最高的Point-F1分数。 |
| [^276] | [Topology-Driven Anti-Entanglement Control for Soft Robots](https://arxiv.org/abs/2605.05236) | 本文提出一种拓扑驱动的多智能体强化学习（TD-MARL）框架，通过共享拓扑状态的集中式学习协调多软体机器人系统，解决了高约束环境下防缠绕控制中可观测性不足与训练不稳定的问题。 |
| [^277] | [Reward-Decomposed Reinforcement Learning for Immersive Video Role-Playing](https://arxiv.org/abs/2605.04733) | 提出EBM-RL框架，通过将“观察-推理-生成”解耦为“眼-脑-嘴”三阶段，并结合场景-文本对齐、感知认知效用、回答忠实性和格式一致性等多维分解奖励，实现了显著超越现有基线的沉浸式视频角色扮演对话。 |
| [^278] | [The Scaling Properties of Implicit Deductive Reasoning in Transformers](https://arxiv.org/abs/2605.04330) | 通过反事实数据增强与跨模式共享推理原语的学习，配备双向前缀掩码的足够深的Transformer其隐式推理性能可接近显式思维链，但深度外推仍需依赖思维链。 |
| [^279] | [ReasonAudio: A Benchmark for Evaluating Reasoning Beyond Matching in Text-Audio Retrieval](https://arxiv.org/abs/2605.03361) | 该论文提出了ReasonAudio基准，用于评估文本-音频检索中的逻辑推理能力（包括否定、时序、声音共现和时长），实验表明现有最先进的检索模型推理能力与人类存在巨大差距。 |
| [^280] | [Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations](https://arxiv.org/abs/2604.23295) | 该论文通过适配 Moshi 双工语音架构，利用 26,000 小时真实印地语自发对话数据，构建了首个开放、可复现的印地语全双工口语对话系统，实现对打断、重叠等自然对话行为的建模。 |
| [^281] | [VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation](https://arxiv.org/abs/2604.21375) | VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。 |
| [^282] | [Information Aggregation with AI Agents](https://arxiv.org/abs/2604.20050) | AI智能体在预测市场实验中能够有效聚合简单信息结构下的分散信息，但在需要超过两层互动推理的复杂环境中表现受限，其推理上限接近人类水平，且廉价磋商、市场时长调整和策略性提示均无法改善其表现。 |
| [^283] | [An AI Agent Execution Environment to Safeguard User Data](https://arxiv.org/abs/2604.19657) | 本文提出GAAP执行环境，通过收集用户权限规范并确定性强制执行数据披露合规，在不信任智能体且不要求模型免受攻击的前提下，保证AI智能体处理私人用户数据时的机密性。 |
| [^284] | [Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines](https://arxiv.org/abs/2604.15186) | Scepsy利用每个LLM执行时间占比在请求间相对稳定这一洞察，通过剖析不同并行度下的LLM并构建聚合LLM流水线作为轻量级吞吐量与延迟预测器，实现了对任意多LLM智能体工作流在GPU集群上的高效低延迟调度。 |
| [^285] | [Endogenous Information in Routing Games: Memory-Constrained Equilibria, Recall Braess Paradoxes, and Memory Design](https://arxiv.org/abs/2604.11733) | 该论文首次将旅行者的有限记忆与信息呈现机制内生化引入路由博弈，建立了“遗忘性Wardrop均衡”的存在性与唯一性理论，并通过严格凸势函数刻画了显著性加权均衡，为记忆与信息界面的交通系统设计提供了可操作的优化框架。 |
| [^286] | [The Shrinking Lifespan of LLMs in Science](https://arxiv.org/abs/2604.07530) | 该研究通过分析62个大语言模型在超过10.8万篇引用论文中的科学采用轨迹，发现模型在科学领域的影响力持续时间主要取决于发布年份而非自身特性，且模型迭代速度不断加快——每往后一个发布年份，模型的达到峰值时间缩短27%、生命周期缩短23%，表明大语言模型正以惊人的速度被更新模型淘汰。 |
| [^287] | [Attribution Bias in Large Language Models](https://arxiv.org/abs/2604.05224) | 该论文提出了首个在作者知名度和人口统计特征上保持平衡的引言归属基准数据集AttriBench，通过对11个大语言模型的评估揭示了归属准确率在种族、性别及交叉群体间的系统性差异，并发现了一种名为“抑制”的新型失败模式——即模型即使掌握作者信息也会完全省略归属。 |
| [^288] | [Combee: Scaling Prompt Learning for Self-Improving Language Model Agents](https://arxiv.org/abs/2604.04247) | Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。 |
| [^289] | [GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow](https://arxiv.org/abs/2603.26571) | GVCC提出了一种零样本视频压缩框架，将预训练的视频生成模型直接作为解码器，通过把确定性修正流采样器转换为保持边缘分布的随机过程并编码每步随机创新量来传输信息，从而在极低码率下实现高保真的视频重建。 |
| [^290] | [Spectral-Sphere-Constrained Hyper-Connections](https://arxiv.org/abs/2603.20896) | 针对双随机约束超连接存在的恒等退化、表达能力瓶颈和参数化开销三重局限，提出将残差矩阵约束在谱球流形上的谱球约束超连接，在保持恒等映射性质以稳定训练的同时，恢复跨流混合的谱自由度和表达能力。 |
| [^291] | [Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System](https://arxiv.org/abs/2602.18640) | 本文提出GEARS框架，将大规模排序优化重构为可编程实验环境中的自主发现过程，通过专门的智能体技能封装排序专家知识，让操作者只需通过高层产品意图即可引导系统，从而突破将模糊产品意图转化为可验证假设的工程瓶颈。 |
| [^292] | [VLANeXt: Recipes for Building Strong VLA Models](https://arxiv.org/abs/2602.18532) | 本文通过统一框架系统剖析VLA设计空间，提炼出12个关键发现，形成了构建强大VLA模型的实用配方。 |
| [^293] | [When to Think Fast and Slow? AMOR: Adaptive Entropy Gate for Hybrid Models](https://arxiv.org/abs/2602.13215) | AMOR通过基于输出熵的动态门控自适应地选择调用注意力，仅在约40%的位置激活注意力且无需可学习路由参数，就能在每个规模上取得最佳的八项常识推理平均成绩。 |
| [^294] | [GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory](https://arxiv.org/abs/2602.12316) | 该论文提出GT-HarmBench——首个基于博弈论结构、包含1535个高风险场景的多智能体AI安全基准，发现前沿模型在38%的高风险场景中无法选择对社会有益的行动，而博弈论干预可将有益结果提升最高18%。 |
| [^295] | [Latent Generative Solvers for Generalizable Long-Term Physics Simulation](https://arxiv.org/abs/2602.11229) | 本文提出潜在生成求解器（LGS），通过物理VAE压缩十二个PDE族到共享潜在流形、金字塔流强制Transformer进行流匹配生成，以及训练时输入加噪的稳定性保证，首次实现了跨异构PDE族的泛化能力与长时域自回归物理模拟稳定性的兼顾。 |
| [^296] | [FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases](https://arxiv.org/abs/2602.09163) | FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。 |
| [^297] | [Beyond Bag-of-Words: Diagnosing Compositional Binding Failures in Vision-Language Models](https://arxiv.org/abs/2602.02043) | 该论文提出Auto-Comp——一个全自动概念驱动的基准生成流水线，通过“平行A/B构建”（最小样本与上下文样本对照）大规模生成照片级真实感的组合推理基准，从而精准诊断视觉-语言模型在属性与物体绑定上的失败。 |
| [^298] | [Stepwise Intrinsic Rewards for Reasoning in Large Language Models](https://arxiv.org/abs/2602.01034) | 提出了一种无需过程标注、辅助模型或推理时搜索的内在过程奖励方法——分步边际信息增益（MIG），通过衡量每个推理前缀对参考答案对数似然的提升，并结合单调水位线机制避免重复计分，从而为大语言模型的多步推理提供更精确的密集监督。 |
| [^299] | [RAPTOR: Ridge-Adaptive Logistic Probes](https://arxiv.org/abs/2602.00158) | 提出了一种简单的L2正则化逻辑回归探针RAPTOR，通过验证集调优的岭回归强度从归一化权重中提取准确且方向稳定的概念向量，可用于大语言模型的“先探针后引导”激活引导流程。 |
| [^300] | [Geometric-Photometric Event-based 3D Gaussian Ray Tracing](https://arxiv.org/abs/2512.18640) | 提出GPERT框架，通过光线追踪将渲染解耦为逐事件的几何（深度）渲染和基于快照的辐射度（强度）渲染两个分支，在基于事件的3D高斯泼溅中解决了精度与时间分辨率之间的权衡问题，且无需先验信息或COLMAP初始化即可达到最先进性能。 |
| [^301] | [Prompt-Based Continual Compositional Zero-Shot Learning](https://arxiv.org/abs/2512.09172) | 该论文提出了首个基于提示的持续组合零样本学习框架PromptCCZSL，通过近期加权多教师蒸馏、会话感知的组合提示与会话无关的属性/对象提示融合，并结合余弦锚定损失，在冻结的视觉-语言模型上实现对新属性、对象及组合的持续适应，同时有效防止旧知识遗忘。 |
| [^302] | [Demo: Generative AI helps Radiotherapy Planning with User Preference](https://arxiv.org/abs/2512.08996) | 本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。 |
| [^303] | [Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement](https://arxiv.org/abs/2512.08982) | 提出Consist-Retinex框架，通过Retinex分解网络与噪声强调的一致性训练实现一步式高质量低光照图像增强，摆脱了迭代采样带来的延迟限制。 |
| [^304] | [Testing the Utility of Using Large Language Models to Create Personalized Networks From Therapy Session Transcripts: A Proof of Concept Study](https://arxiv.org/abs/2512.05836) | 本研究验证了利用大语言模型从心理治疗会话记录中自动生成来访者个性化临床网络的可行性，从而摆脱对密集纵向数据的依赖，支持网络驱动的个案概念化与治疗计划制定。 |
| [^305] | [SLMFix: Leveraging Small Language Models for Domain Specific Language Error Fixing with Reinforcement Learning](https://arxiv.org/abs/2511.19422) | 该论文提出SLMFix流水线，利用强化学习微调的小语言模型根据解释器反馈修复LLM生成代码中的语法错误，在低资源编程语言上使验证器通过率提升了40%。 |
| [^306] | [Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture](https://arxiv.org/abs/2511.06701) | 该论文提出一种函数式架构，通过Haskell的Research monad、声明式脚手架、操作系统级沙箱以及机器验证的Lean 4形式化LORD在线FDR控制，从结构上强制保证AI驱动科学发现中的统计严谨性，防止AI科学家系统因不受控的多重检验而产生虚假发现。 |
| [^307] | [Flow Reconstruction from Sparse Measurements in Urban Drainage Networks: An Application and Evaluation of Data-Driven Sparse Sensing](https://arxiv.org/abs/2511.04556) | 该研究将数据驱动稀疏感知方法应用于77节点城市排水网络，证明仅需占网络4%的3个监测节点即可实现全网络流量状态的准确重构，为资源受限条件下的城市排水监测提供了高效可行的传感方案。 |
| [^308] | [Provable Speech Attributes Conversion via Latent Independence](https://arxiv.org/abs/2510.05191) | 本文为语音属性转换首次建立了形式化理论框架，证明了在确定性自编码器中对潜在表示与可控属性施加独立性约束的条件下，可以实现精确且一致属性迁移的理论保证。 |
| [^309] | [The Plot Twist: Jailbreaking Unified Multimodal Models with a Three-Act NarrativeAttack](https://arxiv.org/abs/2509.26473) | 提出NarrativeAttack框架，利用三幕式叙事结构让统一多模态模型自行生成铺垫与结局图像，并通过图像猜谜游戏将恶意查询隐藏于良性候选中，实现对模型安全机制的越狱。 |
| [^310] | [AUWave: A Data-Driven Model for Reconstructing Significant Wave Heights Using Sparse Observations](https://arxiv.org/abs/2509.19384) | 提出了AUWave混合深度学习框架，结合站点编码器与自注意力增强的多尺度U-Net，从稀疏浮标观测中高精度重建区域有效波高场，并通过浮标消融分析识别关键站点以指导海洋观测网络设计。 |
| [^311] | [Neural Bridge Processes](https://arxiv.org/abs/2508.07220) | 提出神经桥过程（NBP），用输入锚定的桥轨迹替代无条件前向核，使条件输入信息在扩散的含噪状态中就被编码，从而实现对随机函数更具表达力且强条件依赖的学习。 |
| [^312] | [Peer Review as Structured Commentary: Immutable Identity, Public Dialogue, and Reproducible Scholarship](https://arxiv.org/abs/2506.22497) | 本文提出将同行评议重构为基于区块链不可篡改审计和AI迭代综合的结构化公开评论系统，实现透明、可复现、可追溯的学术评价新范式。 |
| [^313] | [Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design](https://arxiv.org/abs/2506.04734) | 本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。 |
| [^314] | [Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning](https://arxiv.org/abs/2505.09738) | 提出了 Tokenadapt——一种与模型无关的分词器移植方法，结合面向多词“超词”的预分词学习，使语言模型能以较低计算成本灵活替换分词器，同时提升压缩效率并减少词元碎片化。 |
| [^315] | [MedHal: a Synthetic Dataset for Medical Hallucination Detection](https://arxiv.org/abs/2504.08596) | MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。 |
| [^316] | [SkillFlow: Scalable and Efficient Agent Skill Retrieval System](https://arxiv.org/abs/2504.06188) | SkillFlow是首个面向智能体技能发现的多阶段检索系统，它将技能获取视为信息检索问题，通过密集检索、交叉编码器重排序和LLM选择四个阶段，从约3.5万个社区技能定义中高效检索出最相关的技能。 |
| [^317] | [LEAD: An EEG Foundation Model for Alzheimer's Disease Detection](https://arxiv.org/abs/2502.01678) | 本文构建了迄今最大的EEG-AD数据集（2,238名受试者），并提出首个脑电图阿尔茨海默病检测基础模型LEAD，其门控时-空Transformer可适应异构EEG数据，配合被试正则化训练策略提升了跨被试泛化能力。 |
| [^318] | [Preference-based opponent shaping in differentiable games](https://arxiv.org/abs/2412.03072) | 本文提出基于偏好的对手塑造方法，通过塑造智能体对合作的偏好来增强多智能体博弈中的策略学习，克服了传统对手建模方法缺乏对行为偏好建模和泛化能力的局限。 |

# 详细

[^1]: 无需学习停止而学会停止：自监督置信度训练提升推理效率

    Learning to Stop without Learning to Stop: Self-Supervised Confidence Training Improves Reasoning Efficiency

    [https://arxiv.org/abs/2609.31619](https://arxiv.org/abs/2609.31619)

    该论文发现，仅通过自监督方式训练模型在推理过程中间点预测自身置信度（损失函数中不含任何长度、效率或停止目标），就能让模型在推理时无需任何提前停止机制便自发提升推理效率。

    

    推理模型通常会生成非常长的推理轨迹，导致推理的计算成本很高。现有方法通常通过两种途径提升效率：一是在推理阶段引入提前停止机制，二是在训练过程中显式鼓励更短的推理，例如使用带长度惩罚的强化学习。我们证明，显著的效率提升可以来自另一种不同的监督信号：置信度。通过一种自监督流程，我们仅使用600个训练问题，对推理模型进行微调，使其在自身推理轨迹的中间点预测对答案的置信度。置信度仅被用作训练目标：损失函数中不包含任何关于推理长度、效率或停止的目标。在推理阶段，微调后的模型采用标准生成流程，无需置信度引导或提前停止机制。尽管如此，自监督……

    arXiv:2609.31619v1 Announce Type: new  Abstract: Reasoning models often generate very long reasoning traces, making inference computationally expensive. Existing approaches typically improve efficiency either through inference-time early-stopping mechanisms or by explicitly encouraging shorter reasoning during training, for example through reinforcement learning with length penalties. We show that substantial efficiency gains can instead emerge from a different kind of supervision: \textit{confidence}. Using a self-supervised procedure, we fine-tune reasoning models to predict their confidence in the answer at intermediate points along their own reasoning trajectories using only 600 training problems. Confidence is used only as a training target: the loss contains no objective for reasoning length, efficiency, or stopping. At inference, the fine-tuned models use the standard generation procedure, with no confidence elicitation or early-stopping mechanism. Despite this, self-supervised 
    
[^2]: 通过输出后处理实现黑盒生成式AI的统计属性对齐

    Statistical attribute alignment for black-box generative AI via output post-processing

    [https://arxiv.org/abs/2609.31607](https://arxiv.org/abs/2609.31607)

    本文针对黑盒生成式AI提出了一种输出后处理方法，通过最小化查询次数的算法将生成输出的属性分布与用户指定目标对齐，并在精确与近似对齐两种情形下证明了算法的最优性。

    

    生成式AI系统的使用日益广泛，但使其输出与用户需求保持一致仍是一项持续的挑战。本文旨在确保AI生成输出的某个属性分布与用户指定的目标相一致。这一问题的动机来自诸如公平性等应用场景——在此场景中我们希望受保护属性（如性别、种族或年龄类别）遵循期望的分布；以及合成数据生成场景——在此场景中我们希望生成的数据能够代表目标分布。我们研究了实际中十分重要的黑盒访问设置，即用户可以重复查询生成式AI模型。目标是返回 $m\ge 1$ 个输出，使其联合属性分布尽可能接近该目标。对于精确对齐和近似对齐两种情形，我们开发了能够最小化生成器期望查询次数的算法，并进一步证明了当输出数量趋于无穷大时这些算法的最优性。

    arXiv:2609.31607v1 Announce Type: cross  Abstract: Generative AI systems are increasingly used, but aligning their outputs with user requirements poses a continuing challenge. Here, we aim to ensure that the distribution of an attribute of an AI-generated output aligns with a user-specified target. This is motivated by examples such as fairness, where we want to ensure that a protected attribute (e.g., gender, race, or age categories) follows a desired distribution, and synthetic data generation, where we want the generated data to be representative of a target distribution. We study the practically important black-box access setting, where a user can repeatedly query a generative AI model. The goal is to return $m\ge 1$ outputs whose joint attribute distribution is as close as possible to this target. For both exact and approximate alignment, we develop algorithms that minimize the expected number of queries to the generator, and we further demonstrate their optimality as the number o
    
[^3]: 面向编程智能体的紧凑文档：一个基准、一个优化器，以及它为何无法迁移

    Compact Documentation for Coding Agents: A Benchmark, an Optimizer, and Why It Does Not Transfer

    [https://arxiv.org/abs/2609.31587](https://arxiv.org/abs/2609.31587)

    本文提出了一个通过“重新生成代码能否通过原始测试”来评估代码描述的往返基准和优化器，发现完整性而非长度决定文档保真度，但令人意外的负面结果表明：当源代码可用时，更好的文档并不能帮助编程智能体解决真实的仓库问题。

    

    我们研究了自然语言文档是否能帮助编程智能体解决软件问题，并构建了用于生成和评估此类文档的工具。我们引入了一个往返式基准，通过“根据代码描述重新生成的代码能否通过原始测试”来为描述打分，并证明决定描述保真度的是完整性而非长度。以该基准作为优化信号，我们发现了一条描述撰写提示词，可达到完全保真度并能泛化到未见过的文件。随后，我们检验了这项工作最初的动机假设：更好的文档能帮助智能体解决真实的仓库问题。我们在两个模型家族和十个仓库上进行了测试，并设置了一个阳性对照以确认我们的评估能够检测出真正的改进，但结果表明情况并非如此。当源代码存在时，无论是静态紧凑文档还是检索到的上下文，都不如仅凭问题描述有效。我们将这一负面结果与……（原文截断）

    arXiv:2609.31587v1 Announce Type: cross  Abstract: We investigate whether natural-language documentation helps coding agents resolve software issues, and we build the tools to construct and evaluate it. We introduce a roundtrip benchmark that scores code descriptions by whether code regenerated from them passes the original tests, and show that completeness, not length, drives a description's fidelity. Using the benchmark as an optimization signal, we discover a description-writing prompt that reaches full fidelity and generalizes to unseen files. We then test the hypothesis that motivated the work: that better documentation helps an agent resolve real repository issues. Across two model families and ten repositories, and against a positive control confirming that our evaluation can detect a genuine improvement, we find that it does not. When the source is present, neither static compact documentation nor retrieved context beats the issue alone. We report this negative result together 
    
[^4]: OC-GS：面向不规则转台采集的高斯泼溅方法

    OC-GS: Gaussian Splatting for Irregular Turntable Capture

    [https://arxiv.org/abs/2609.31572](https://arxiv.org/abs/2609.31572)

    OC-GS提出一种轨道一致的物体中心高斯泼溅方法，在共享相机、旋转轴和枢轴的约束下联合优化图像几何与角度，实现了从不规则稀疏转台采集中高质量的三维物体重建。

    

    不均匀的旋转和丢帧使得等角度假设在转台重建中变得不可靠。我们提出OC-GS，一种以物体为中心的高斯泼溅方法，它在保持共享相机、旋转轴和枢轴的同时，对每张图像的角度进行细化。这种轨道一致的细化方法联合优化从图像推导的几何与角度，从而从稀疏、不规则的采集数据中重建物体。在具有12、8和6个不规则分布视角的渲染物体上，OC-GS分别取得了21.26、19.36和15.83dB的平均前景PSNR，在每种条件下均超过所有四个被评估的无位姿高斯泼溅基线方法。在共享训练器下，对图像估计角度进行细化相比固定这些估计，可使平均前景PSNR提升7.88dB。消融研究表明，图像推导的角度初始化和共享运动模型均对性能提升有所贡献。在真实采集数据上，OC-GS的细化方法提高了平均前景……（摘要原文在此处截断）

    arXiv:2609.31572v1 Announce Type: cross  Abstract: Uneven rotation and dropped frames make equal-angle assumptions unreliable for turntable reconstruction. We present OC-GS, an object-centric Gaussian splatting that refines each image's angle while maintaining a shared camera, rotation axis, and pivot. This orbit-consistent refinement jointly optimizes image-derived geometry and angles to reconstruct objects from sparse, irregular captures. On rendered objects with 12, 8, and 6 irregularly spaced views, OC-GS achieves mean foreground PSNR scores of 21.26, 19.36, and 15.83dB, respectively, exceeding all four evaluated pose-free Gaussian splatting baselines in each condition. Under a shared trainer, refining image-estimated angles improves mean foreground PSNR by 7.88dB over keeping those estimates fixed. An ablation study shows that both image-derived angle initialization and the shared motion model contribute to the improvement. On real captures, OC-GS's refinement increases mean foreg
    
[^5]: 适应AI：小学教师如何为AI融合课程调整自身教学实践

    Adapting for AI: How elementary teachers adjust their practices for an AI-integrated curriculum

    [https://arxiv.org/abs/2609.31569](https://arxiv.org/abs/2609.31569)

    本研究通过对三位小学教师实施基于对话式AI平台的课程进行为期三周的追踪，首次揭示了教师通过修复、差异化、转化与平衡等适应性实践，在技术、学习者与教学三重张力交汇处推进AI课程课堂落地的真实工作方式。

    

    对话式AI工具正在进入儿童的日常生活，学校也对采用这些工具表现出兴趣。然而，成功的课堂整合不仅取决于技术本身，还取决于教师为使技术适用于学生和课堂情境所付出的努力。目前，人们对小学教师在真实课堂中实施对话式AI工具时的工作方式知之甚少。在本研究中，我们考察了三位教师在13个教学日（为期三周的夏令营）中实施一套以ToyTalk（一个对话式AI玩具开发平台）为核心的AI素养与英语语言艺术（ELA）课程的经历。通过分析每日个人反思、小组反思以及营后访谈，我们发现教师的适应性实践——修复、差异化、转化与平衡——正处于三种张力（技术、学习者和教学）的交汇点上。教师的理解……

    arXiv:2609.31569v1 Announce Type: cross  Abstract: Conversational AI tools are entering children's everyday experiences, and schools are interested in adopting them. However, successful classroom integration depends not only on the technology but also on the work teachers do to make it usable and appropriate for their students and classroom context. There is little known about how elementary teachers work as they implement conversational AI tools in real classrooms. In this study, we examine three teachers' experiences implementing an AI literacy and English Language Arts (ELA) curriculum built around ToyTalk, a conversational AI toy development platform, over 13 instructional days, a three-week summer camp. Drawing on daily individual reflections, group reflections, and post-camp interviews, we find that teachers' adaptive practices of repair, differentiation, translation, and balancing sit at the intersection of three tensions (technology, learner, and instruction). Teachers' underst
    
[^6]: DeepEdu-v1：面向越南教育的高效且可扩展的智能体大语言模型

    DeepEdu-v1: Efficient and Scalable Agentic LLMs for Vietnamese Education

    [https://arxiv.org/abs/2609.31568](https://arxiv.org/abs/2609.31568)

    DeepEdu-v1是一个面向越南教育的AI辅导系统，通过本地部署保障数据主权、围绕国家课程体系组织知识，并克服了消费级GPU上长上下文推理的内存与延迟瓶颈以及区域内容上的幻觉问题。

    

    AI辅导能够显著改善越南等发展中地区学生的学习成果，然而两条显而易见的技术路径都存在不足。以ChatGPT为代表的云端助手会将敏感的学生数据传输到境外服务器——这违反了越南第53号法令等数据主权法律——而且由于预训练语料以西方为中心，这些助手并未围绕国家教科书课程体系进行组织，因此对本地内容的掌握零散且频繁出现幻觉。自行托管开源模型虽可将数据保留在本地，却面临双重障碍：训练后量化（AWQ、GPTQ）虽能控制静态权重的占用，但长辅导上下文带来的动态KV缓存与预填充（prefill）延迟仍会在消费级GPU上引发内存溢出（OOM）故障和响应迟缓，同时模型在区域特定材料上仍会不断产生幻觉。我们提出DeepEdu-v1，一个面向越南教育的AI辅导系统，它基于SCALE（自我改进的上下文感知大语言模型……（摘要原文在此处截断）

    arXiv:2609.31568v1 Announce Type: new  Abstract: AI tutoring could markedly improve learning outcomes for students in developing regions such as Vietnam, yet the two obvious paths both fall short. Cloud assistants such as ChatGPT route sensitive student data to foreign servers---violating data-sovereignty laws such as Vietnam's Decree 53---and, pre-trained on Western-centric corpora, are not organized around the national textbook curriculum, so their knowledge of local content is unsystematic and frequently hallucinated. Self-hosting an open model keeps data on-premise but hits a two-fold wall: post-training quantization (AWQ, GPTQ) tames the static weight footprint, yet the dynamic KV cache and prefill latency of long tutoring contexts still cause out-of-memory failures and slow responses on consumer GPUs, while the model keeps hallucinating on region-specific material. We present DeepEdu-v1, an AI-tutoring system for Vietnamese education built on SCALE (Self-improving Context-Aware L
    
[^7]: 跨越析取性与补偿性任务的多智能体规模扩展研究

    Multi-agent Scaling Across Disjunctive and Compensatory Tasks

    [https://arxiv.org/abs/2609.31563](https://arxiv.org/abs/2609.31563)

    本文引入Steiner群体任务分类法来分析多智能体LLM系统的规模扩展行为，发现在析取性任务中，尽管至少一个智能体答对的概率随团队规模提升5-20个百分点，但简单的多数投票策略几乎无法兑现这一集体潜力，其结果仅收敛于模型的众数答案。

    

    多智能体大语言模型（LLM）系统通常被认为会随着团队规模的增大而性能提升，但其扩展行为可能取决于任务结构。我们的核心贡献是引入Steiner的群体任务分类法作为分析多智能体LLM扩展的框架，并将分析重点聚焦于析取性任务和补偿性任务。我们将独立采样的智能体建模为在给定题目条件下条件独立，由此推导出它们在大团队规模下的极限：多数投票收敛于模型的众数答案，而取平均则收敛于模型的条目级偏差。在选定的代表性基准测试、13个开放权重模型以及最多30个智能体组成的团队上，我们发现了性质上截然不同的扩展行为。在析取性任务上，至少有一个智能体答对的概率随团队规模增长5-20个百分点，但对直接作答的智能体进行多数投票几乎无法实现这一潜力，其结果与模型预测的差距在0.5个百分点以内。

    arXiv:2609.31563v1 Announce Type: new  Abstract: Multi-agent LLM systems are often expected to improve as team size increases, yet the scaling behavior may depend on task structure. Our central contribution is to introduce Steiner's taxonomy of group tasks as a framework for analyzing multi-agent LLM scaling and focusing the analysis on disjunctive and compensatory tasks. We model independently sampled agents as conditionally independent given the item, which yields their large-team limits: plurality voting converges to the model's modal answer, and averaging converges to the model's item-level bias. Across selected representative benchmarks, 13 open-weight models, and teams of up to 30 agents, we find qualitatively different scaling behavior. On disjunctive tasks, the probability that at least one agent is correct grows by 5-20 points with team size, but plurality voting over agents that answer directly realises almost none of this potential, as the model predicts to within 0.5 points
    
[^8]: 用于神经表征差异性的流匹配框架

    A Flow Matching Framework for Neural Representational Dissimilarity

    [https://arxiv.org/abs/2609.31544](https://arxiv.org/abs/2609.31544)

    本文提出用深度生成模型中的流匹配框架统一多种神经表征差异性度量（即将其归结为不同速度约束下的Jeffreys散度），该框架在处理复杂分布和连续变量时具有估计优势，并能支持以有原则的方式设计新的距离度量。

    

    神经表征差异性量化了神经响应分布之间的差异，对于比较不同刺激、脑区、任务和模型之间的神经编码至关重要。常用的距离度量涉及不同的假设，并需使用各自不同的方法进行估计。在这里，我们证明了多种距离度量可以在深度生成模型中发展出的流匹配框架下得到统一。也就是说，这些距离在不同速度约束下表现为Jeffreys散度。我们发现，流匹配在估计涉及复杂分布和连续变量的距离方面具有优势。此外，该框架能够以有原则的方式设计新的距离度量。总之，流匹配为理解、估计和设计神经表征差异性度量提供了一种统一的方法。

    arXiv:2609.31544v1 Announce Type: new  Abstract: Neural representational dissimilarity quantifies differences between neural response distributions, and is essential for comparing neural codes across stimuli, brain areas, tasks, and models. Commonly used distance metrics involve different assumptions and are estimated with separate methods. Here, we show that a variety of distance metrics can be unified under a flow matching framework developed in deep generative models. That is, these distances arise as Jeffreys divergences under different velocity constraints. We find that flow matching has advantages for estimating distances involving complicated distributions and continuous variables. Furthermore, this framework enables the design of new distance metrics in a principled way. Together, flow matching provides a unified approach for understanding, estimating, and designing neural representational dissimilarity metrics.
    
[^9]: 你能检查吗？本地LLM网络自动化的可检验性边界

    Can You Check That? The Checkability Boundary for Local LLM Network Automation

    [https://arxiv.org/abs/2609.31540](https://arxiv.org/abs/2609.31540)

    该论文提出“可检验性”标准来判断网络自动化任务是否适合本地小模型推理，并实现了Touchstone系统，通过内在检查筛选SLM输出、仅将少量输入升级给前沿LLM，在保护敏感数据不出本地的同时实现了98.6%和93.8%的端到端准确率。

    

    将每个网络自动化输入都发送给第三方前沿大语言模型（LLM）会导致生产配置、拓扑结构和日志等敏感信息外流。在本地查询小语言模型（SLM）可以避免这种数据外流，但SLM的输出容易出错，难以直接使用。本工作引入“可检验性”作为判断哪些任务适合本地推理的标准。当一个任务存在一个廉价、确定性的测试——即内在检查——能够拒绝违反必要正确性条件的输出时，该任务就是可检验的。我们在Touchstone系统中实例化了这一想法，这是一个本地优先的流水线，使用七个现成的SLM（1-8B参数）生成候选结果，利用任务特定的内在检查来拒绝错误响应，并将无法解决的输入升级给前沿LLM。在冲突检测和意图翻译任务上，Touchstone分别达到98.6%和93.8%的端到端准确率，同时仅将16%和17%的输入升级处理。

    arXiv:2609.31540v1 Announce Type: cross  Abstract: Sending every network-automation input to a third-party frontier LLM exports sensitive artifacts such as production configurations, topologies, and logs. Querying small language models (SLMs) locally avoids this egress, but SLM outputs can be error-prone for direct use. This work introduces checkability as a criterion for determining which tasks are suitable for local inference. A task is checkable when it exposes a cheap, deterministic test - an intrinsic check - that rejects outputs violating a necessary correctness condition. We instantiate this idea in Touchstone, a local-first pipeline that uses seven off-the-shelf SLMs (1-8B parameters) to generate candidates, uses task-specific intrinsic checks to reject responses, and escalates unresolved inputs to a frontier LLM. On conflict detection and intent translation tasks, Touchstone reaches 98.6% and 93.8% end-to-end accuracy while escalating only 16% and 17% of inputs, respectively. 
    
[^10]: ClearGS：面向手持视频的可靠性感知高斯泼溅

    ClearGS: Reliability-Aware Gaussian Splatting from Handheld Videos

    [https://arxiv.org/abs/2609.31509](https://arxiv.org/abs/2609.31509)

    ClearGS通过可靠性感知视图分配（RVA）为手持视频帧分配分级监督权重，并结合渲染引导的视频内修复（RIVR）与全轨迹修复整合机制，实现了从视点覆盖不均、质量参差不齐的手持视频中进行高质量3D高斯泼溅重建。

    

    我们提出了ClearGS，用于从视点覆盖不均匀且帧质量参差不齐的手持视频中进行3D高斯泼溅（3DGS）。ClearGS并非采用二元的帧选择决策，而是使用可靠性感知视图分配（RVA），基于外观可靠性、退化风险和几何效用为原始帧分配分级的监督权重，同时弱激活被抑制的有用帧以维持轨迹覆盖。由于加权无法恢复因模糊或失真而丢失的细节，ClearGS进一步引入了渲染引导的视频内修复（RIVR）：当前的3DGS渲染提供姿态对齐的结构候选，一个冻结的无参考修复专家在没有任何干净参考图像的情况下修复对应的原始视频观测，再由无参考感知评分在渲染结果、修复后的观测以及高频融合候选之间进行选择。随后，ClearGS应用全轨迹修复整合（Full-Trajectory Repair Consolidation）来修订……（原文摘要在此处截断）

    arXiv:2609.31509v1 Announce Type: cross  Abstract: We present ClearGS for 3D Gaussian Splatting (3DGS) from handheld videos with uneven viewpoint coverage and mixed frame quality. Rather than selecting frames with binary decisions, ClearGS uses Reliability-aware View Allocation (RVA) to assign graded raw-supervision weights based on appearance reliability, degradation risk, and geometric utility, while weakly reactivating useful suppressed frames to maintain trajectory coverage. Since weighting cannot restore details lost to blur or distortion, ClearGS further introduces Render-Guided In-Video Restoration (RIVR). The current 3DGS render provides a pose-aligned structural candidate, a frozen no-reference restoration expert restores the corresponding raw video observation without any clean reference image, and no-reference perceptual scores select among the render, restored observation, and high-frequency fused candidate. ClearGS then applies Full-Trajectory Repair Consolidation to revis
    
[^11]: 评估大语言模型对海地克里奥尔语的文化意识

    Evaluating Cultural Awareness of LLMs for Haitian Creole

    [https://arxiv.org/abs/2609.31506](https://arxiv.org/abs/2609.31506)

    该论文首次从特异性、偏见、多样性和变异性四个维度系统评估了大语言模型对海地克里奥尔语的文化意识，发现其明显落后于法语，且海地角色常被以苦难形象呈现。

    

    大语言模型（LLMs）在高资源语言与低资源语言之间表现出显著的性能差异。除了任务性能较低之外，它们往往无法捕捉代表性不足群体的文化规范和价值观。在这项工作中，我们对大语言模型在海地克里奥尔语上的文化意识进行了首次系统性评估——海地克里奥尔语是一种有数百万人使用的语言，但在数字资源中却严重缺乏代表性。我们沿着四个互补的维度——特异性、偏见、多样性和变异性——对文化意识进行评估，使用了一个由母语者精心策划的文化显著性提示基准，并采用文本填充（text infilling）的实验设置。我们的结果揭示了海地克里奥尔语与资源更丰富的法语之间在文化意识上存在明显差距，海地语的表现不仅在各领域间更加不均衡，而且更容易受到法语的语言干扰。故事生成任务还进一步揭示了一个反复出现的现象：海地角色总是被通过苦难的形象来描绘。

    arXiv:2609.31506v1 Announce Type: cross  Abstract: Large language models (LLMs) exhibit substantial performance disparities between high- and low-resource languages. Beyond lower task performance, they often fail to capture the cultural norms and values of underrepresented communities. In this work, we present the first systematic evaluation of cultural awareness in LLMs for Haitian Creole, a language spoken by millions but severely underrepresented in digital resources. We assess cultural awareness along four complementary dimensions---specificity, bias, diversity, and variation---using a benchmark of culturally salient prompts curated by native speakers in a text infilling setting. Our results reveal a clear gap between cultural awareness in Haitian Creole and higher-resource French, with Haitian performance being more uneven across domains and more affected by French linguistic interference. Story generation further reveals recurring portrayals of Haitian characters through hardship
    
[^12]: 提示最小化：在不牺牲输出保真度的前提下减少输入冗余

    Prompt Minimization: Reducing Input Redundancy Without Sacrificing Output Fidelity

    [https://arxiv.org/abs/2609.31505](https://arxiv.org/abs/2609.31505)

    该论文提出“提示最小化”概念及三种评估框架，证明将提示词压缩至最小信息密集形式后仍能产生与原始长提示词相当的输出，从而降低计算开销并避免冗长提示对模型推理的损害。

    

    尽管大语言模型（LLM）的能力不断增强，但提示词的设计在很大程度上仍然依赖启发式和临时方法。本项目将探索“提示最小化”（prompt minimization），即将提示词缩减至其最小、信息最密集的形式，同时保持输出的保真度。在实际应用中，更短的提示词可以减少计算开销和推理延迟，尤其是在不必要地包含大型上下文（如整个文档或代码库）的情况下。此外，过长的提示词可能会损害LLM的推理能力和准确性。从理论角度看，多个提示词能产生等效输出这一现象表明输入空间存在高度冗余，这引发了关于哪些信息对于引发特定模型行为必不可少的根本性问题。我们提出了三种变体框架来识别和评估最小提示词，并证明最小提示词往往能产生与其较长版本相当的输出。

    arXiv:2609.31505v1 Announce Type: new  Abstract: Despite the growing capabilities of large language models (LLMs), prompt design remains largely heuristic and ad hoc. This project will explore $\textit{prompt minimization}$, the process of reducing prompts to their smallest, most information-dense form while preserving output fidelity. Practically, shorter prompts reduce computational overhead and inference latency, especially when large contexts, such as entire documents or codebases, are included unnecessarily. Further, longer prompts can damage LLM reasoning and accuracy. Theoretically, the existence of multiple prompts yielding equivalent outputs suggests a high degree of redundancy in the input space, raising fundamental questions about what information is essential to elicit specific model behaviors. We propose three variant frameworks to identify and evaluate minimal prompts and demonstrate that minimal prompts often produce outputs comparable to those of their longer counterpar
    
[^13]: UQ-LOB：不确定性感知的限价订单簿中间价预测

    UQ-LOB: Uncertainty-Aware Limit Order Book Mid-Price Forecasting

    [https://arxiv.org/abs/2609.31491](https://arxiv.org/abs/2609.31491)

    本文提出UQ-LOB，一个可附加到任意预训练LOB编码器的轻量级不确定性量化模块，通过对已完成窗口上下文的条件化，为限价订单簿中间价短期预测提供校准的置信度估计，从而支持选择性预测。

    

    从限价订单簿（LOB）数据预测短期中间价变动是算法交易的核心，然而大多数深度LOB预测器都是点预测器：它们只输出一个方向或位移量，却从不表明哪些预测结果是可信的。我们提出了UQ-LOB，一个轻量级、与编码器无关的不确定性量化模块，它可以附加到任何预训练的LOB编码器上，并借鉴注意力神经过程的思想，将每次预测条件化于一个由最近已完成且结果已经实现的窗口组成的上下文集合。UQ回归变体输出未来最小价位变动位移的校准高斯分布，而UQ分类变体输出下跌/上涨/平稳三类别的类别分布。两者都提供一个标量置信度（预测信噪比或类别概率），以支持选择性预测。在七种加密货币资产、共52亿条LOB事件以及5、10等预测期限上的实验中（摘要在此处截断）。

    arXiv:2609.31491v1 Announce Type: new  Abstract: Forecasting short-horizon mid-price movements from limit order book (LOB) data is central to algorithmic trading, yet most deep LOB forecasters are point predictors: they output a direction or a displacement, but never indicate which of their forecasts can be trusted. We introduce UQ-LOB, a lightweight, encoder-agnostic uncertainty quantification module that attaches to any pretrained LOB encoder and, in the spirit of attentive neural processes, conditions each forecast on a context set of recently completed windows whose outcomes are already realised. The UQ-regression variant outputs a calibrated Gaussian over the future tick displacement, while the UQ-classification variant outputs a categorical distribution over down/up/stationary. Both expose a scalar confidence (predicted signal-to-noise ratio or class probability) that supports selective prediction. On 5.2 billion LOB events across seven cryptocurrency assets and horizons of 5, 10
    
[^14]: “AI是（并非）新的……”：一个用于分析生成式AI文化影响的诊断性类比框架

    "AI is (not) the new...": A Diagnostic Analogy Framework for Generative AI's Cultural Impacts

    [https://arxiv.org/abs/2609.31482](https://arxiv.org/abs/2609.31482)

    本文提出一个诊断性类比框架，将技术干预分解为认知场所、治理逻辑和技术机制三个坐标，以精确分析生成式AI对认知与文化实践的变革，避免因印刷机、蒸汽机等模糊的历史类比而设计出针对错误属性的治理干预。

    

    生成式AI正在重塑知识被发现、综合和问责的文化基础设施。为了理解这一转变，学者和政策制定者求助于印刷机、蒸汽动力或电力等技术的历史类比。但这些比较通常无法精确说明技术的哪个属性承载了这种类比，而不精确的类比会导致不精确的治理，因为它针对系统的错误属性设计干预措施。本文提供了一个诊断框架，用于分析生成式AI如何转变认知和文化实践。我们将每项干预分解为三个坐标：技术发挥作用的认知场所、技术借此组织其对象的治理逻辑，以及技术借此实现作用的技术机制……

    arXiv:2609.31482v1 Announce Type: new  Abstract: Generative AI is reshaping the cultural infrastructures through which knowledge is found, synthesized, and held accountable. To make sense of this shift, scholars and policymakers reach for historical analogies of technologies such as the printing press, steam power or electricity. But these comparisons are typically imprecise about which property of the technology carries the comparison, and imprecise analogies produce imprecise governance by designing interventions against the wrong property of the system. This paper offers a diagnostic framework for analyzing how generative AI can transform epistemic and cultural practice. This paper offers a diagnostic framework for analyzing how generative AI can transform epistemic and cultural practice. We decompose each intervention into three coordinates: the epistemic site at which a technology acts, the governing logic by which it organizes its object, and the technical mechanism through which
    
[^15]: 游戏竞技场：竞争环境中的大语言模型策略评估

    Game Arena: Strategic LLM Evaluation in Competitive Environments

    [https://arxiv.org/abs/2609.31473](https://arxiv.org/abs/2609.31473)

    本文提出了Kaggle游戏竞技场，一个通过国际象棋、扑克、狼人杀三类竞技游戏动态评估大语言模型策略规划与适应能力的开放平台，有效避免了静态基准的性能饱和问题。

    

    我们推出了Kaggle游戏竞技场，这是一个通过竞技游戏来评估大语言模型（LLM）的开放且不断扩展的平台。与静态基准不同，游戏竞技场使模型能够在结构化环境中进行面对面的对决，随着模型的演进，游戏对抗强度自然提升，从而防止性能饱和。这份技术报告详细介绍了游戏竞技场背后的基础设施，并描述了三个试点游戏环境：国际象棋、扑克和狼人杀。这些环境涵盖了完全信息、不完全信息以及多人游戏设定，能够系统性地研究模型的策略规划能力、适应能力以及在不确定性下的鲁棒性。针对每个游戏，我们提供了环境的详细描述、评估指标以及跨模型完整竞赛的运行结果。通过强大的基础设施和基于大规模真实标准的评估，游戏竞技场确保了结果的可重现性。

    arXiv:2609.31473v1 Announce Type: new  Abstract: We introduce Kaggle Game Arena, an open and ever-expanding platform to evaluate large language models (LLMs) through competitive games. Different from static benchmarks, game arena enables models to play head-to-head matchups in structured environments where the gameplay strength naturally increases as models evolve, preventing performance saturation. This technical report details the infrastructure behind Game Arena and describes the three pilot game environments: Chess, Poker, and Werewolf. These environments span perfect information, imperfect information, and multiplayer game settings, enabling a systematic study of models' strategic planning, adaptation, and robustness under uncertainty. For each game, we provide a detailed description of the environment, evaluation metrics, and results from running full competitions across models. Through robust infrastructure and large-scale ground-truth based evaluation, Game Arena ensures reprod
    
[^16]: PriceBench：用于诊断LLM预订代理中价格、质量与品牌偏好的基准测试

    PriceBench: A Diagnostic Benchmark for Price, Quality, and Brand Preferences in LLM Booking Agents

    [https://arxiv.org/abs/2609.31468](https://arxiv.org/abs/2609.31468)

    PriceBench通过logit选择模型从LLM的酒店预订行为中诊断其价格、质量和品牌偏好，发现更强的模型偏好更一致但各不相同，而较弱的模型要么偏好僵化易被列表顺序操纵，要么近乎随机选择。

    

    LLM正越来越多地充当购买代理，这意味着是LLM而非用户在满足请求的各个选项中进行选择；其偏好悄然决定了最终买什么以及花多少钱。酒店预订是一个典型的场景：这是一个高频发生的选择过程，基于几个可比较的属性做出决定，而其选择能够揭示这些偏好。我们提出了PriceBench，一个诊断性基准，通过logit选择模型从LLM的预订选择中恢复其价格、质量和品牌偏好。该基准应用于来自8家提供商的28个LLM，涵盖179家真实纽约市酒店的3,600个酒店任务。我们发现，模型能力与LLM选择的一致性相关，而非与其选择的内容相关：能力更强的LLM持有更强、更一致的偏好，而能力较弱的模型要么锁定于某一种选择（这种僵化偏好可被控制列表顺序的一方所利用），要么几乎是无差别地随机选择。这些偏好所倾向的内容在不同提供商之间差异显著。

    arXiv:2609.31468v1 Announce Type: cross  Abstract: LLMs increasingly act as purchasing agents, which makes the LLM, not the user, the one choosing among the options that satisfy a request; its preferences quietly fix what gets bought and what it costs. Hotel booking is a clean instance: a high-volume choice settled on a few comparable attributes, where the pick reveals those preferences. We introduce PriceBench, a diagnostic benchmark that recovers an LLM's price, quality, and brand preferences from its booking choices with a logit choice model, applied to 28 LLMs from 8 providers on 3,600 hotel tasks from 179 real New York City properties. We find that capability is associated with how consistently an LLM chooses, not with what it chooses: more capable LLMs hold stronger, more consistent preferences, while weaker ones either lock onto one position, exploitable by whoever controls listing order, or choose almost indifferently. What those preferences favor varies sharply across provider
    
[^17]: 面向婴儿运动分析的不确定性感知联邦学习

    Uncertainty-Aware Federated Learning for Infant Movement Analysis

    [https://arxiv.org/abs/2609.31463](https://arxiv.org/abs/2609.31463)

    提出了首个面向婴儿运动分析的不确定性感知联邦学习框架，能够在保护隐私的前提下利用多机构骨骼运动数据实现自动化全身运动评估。

    

    婴儿运动分析为神经发育障碍的早期识别提供了有价值的生物标志物。深度学习的最新进展使得从视频提取的骨骼表示中进行自动化婴儿运动分析成为可能，在全身运动评估（GMA）等任务上达到了与专家评估相当的性能。然而，大多数现有方法依赖于集中式训练，需要将多个机构的数据收集并存储在单一站点。由于隐私、治理和数据共享的限制，这种假设在临床环境中往往不切实际。为了应对这些挑战，据我们所知，我们提出了首个使用骨骼运动数据进行自动化婴儿运动分析和全身运动评估的联邦学习框架。作为一个具有临床相关性的用例，所提出的框架在不安运动分类任务上进行了评估。为了量化……

    arXiv:2609.31463v1 Announce Type: cross  Abstract: Infant movement analysis provides valuable biomarkers for the early identification of neurodevelopmental disorders. Recent advances in deep learning have enabled automated analysis of infant movements from video-derived skeletal representations, achieving performance comparable to expert assessment for tasks such as General Movement Assessment (GMA). However, most existing approaches rely on centralized training, requiring data from multiple institutions to be collected and stored at a single site. Such assumptions are often impractical in clinical settings due to privacy, governance, and data-sharing constraints. To address these challenges, we present, to the best of our knowledge, the first federated learning framework for automated infant movement analysis and General Movement Assessment using skeletal motion data. As a clinically relevant use case, the proposed framework is evaluated on fidgety movement classification. To quantify
    
[^18]: 面向数据探索与资源效率提升的片段级智能体式主题建模

    Segment-Level Agentic Topic Modeling for Improved Data Exploration and Resource Efficiency

    [https://arxiv.org/abs/2609.31460](https://arxiv.org/abs/2609.31460)

    本文提出片段级智能体式主题建模框架SeLATM，通过分段处理方式解决基于LLM的主题模型无法生成文档主题分布、主题宽泛度不当以及资源消耗过高等问题，从而提升数据探索质量与资源利用效率。

    

    主题建模是一种用于发现文档中隐藏主题的有效技术，被广泛应用于各行业领域的文本挖掘与数据分析中。近年来，出现了基于大语言模型（LLM）的主题模型，其通过提示LLM生成主题并将其分配给文档，产生了比传统主题建模算法更自然、更易于人类理解的主题。然而，这种主题分配过程的固有特性带来了一些缺陷，例如无法生成文档的主题分布、主题过于宽泛或狭窄，以及高昂的资源消耗——且该消耗会随着被分配主题的文档数量和长度的增长而增加。这些问题对于需要高质量、深入分析并处理海量文档的工业应用而言尤为突出。在此背景下，本文提出了一个名为SeLATM的框架，旨在解决……（原文摘要在此处截断）

    arXiv:2609.31460v1 Announce Type: new  Abstract: Topic modeling is an effective technique for discovering hidden themes within documents and is widely used in text mining and data analysis across a variety of industry sectors. Recently, large language model (LLM)-based topic models have been emerged that prompt LLMs to generate topics then assign the topics to documents, producing more natural and human-readable topics than conventional topic modeling algorithms. However, the nature of topic assignment process causes certain drawbacks, such as the incapability to produce topic distributions over a document, too broad or narrow topics, and high resource consumption, which increases with the number and length of of documents being assigned topics. These issues are particularly critical for industrial applications, which require high-quality, in-depth analysis and the processing of large volumes of documents. In this context, this paper introduces a framework called SeLATM, which addresse
    
[^19]: 不同的损坏，不同的信号：联邦数据质量中的不确定性与损失

    Different Corruptions, Different Signals: Uncertainty and Loss in Federated Data Quality

    [https://arxiv.org/abs/2609.31454](https://arxiv.org/abs/2609.31454)

    本文比较了联邦学习中输入条件不确定性与预测-标签损失两种损坏检测信号，发现在非独立同分布数据条件下，这两种信号对输入噪声和标签翻转两类损坏表现出不同的检测效果。

    

    联邦学习中的数据损坏可能影响输入或标签，但目前尚不清楚输入条件不确定性和预测-标签损失是否能同等地揭示这些损坏模式。本文比较了联邦学习中两种损坏检测信号：输入条件不确定性和预测-标签损失。不确定性信号通过学习到的偶然方差估计以及蒙特卡洛 dropout 方差和熵度量来表征，而损失则基于所提供的标签进行计算。我们针对加性图像噪声和持续性随机标签翻转测试了这些信号。在 ResNet-20 上使用 CIFAR-10 和 SVHN 数据集、在 Dirichlet 划分的非独立同分布数据条件下，两种损坏类型表现出不同的行为。对于持续性随机标签翻转，客户端内逐样本的受试者工作特征曲线下面积（AUC）在 CIFAR-10 上为 0.85，在 0.95（原文摘要在此处中断）。

    arXiv:2609.31454v1 Announce Type: cross  Abstract: Federated learning (FL) data corruption can affect either inputs or labels, but it remains unclear whether input-conditional uncertainty and prediction-label loss expose these corruption modes equally. This paper compares two corruption-detection signals in FL: input-conditional uncertainty and prediction-label loss. The uncertainty signal is characterised using a learned aleatoric variance estimate together with Monte Carlo (MC) dropout variance and entropy measures, while the loss is computed against the supplied label. We test these signals against additive image noise and persistent random label flips. On ResNet-20 with CIFAR-10 and SVHN under Dirichlet partitions with data that are not independent and identically distributed (non-IID), the two corruption types behave differently. For persistent random label flips, the within-client per-sample area under the receiver operating characteristic curve (AUC) is 0.85 on CIFAR-10 and 0.95
    
[^20]: 从奖励信号到视觉效用：医学视觉语言模型后训练的受控审计

    From Reward Signal to Visual Utility: A Controlled Audit of Medical VLM Post-Training

    [https://arxiv.org/abs/2609.31450](https://arxiv.org/abs/2609.31450)

    该受控审计表明，医学VLM后训练中答案准确率的提升并不代表视觉利用能力的改善——语言模型LoRA微调虽小幅提升正确图像准确率，却降低了视觉收益事件与图像敏感度，而更广泛的多模态适应反而使正确图像准确率更低。

    

    医学视觉语言模型（VLM）的后训练通常通过答案准确率来评估。我们在PMC-VQA数据集上对Qwen2.5-VL-3B进行受控研究，考察准确率的变化和训练目标如何与基于图像条件的决策相关联。我们比较了仅限于语言模型的低秩适应（LoRA）监督微调（SFT）、扩展的多模态适应范围、标准的仅答案组相对策略优化（GRPO）以及一种反事实证据目标。在2,000个干净测试问题上，语言模型LoRA SFT使正确图像准确率变化+1.10个百分点（95%配对自助置信区间：-0.85至+3.05），而视觉收益事件减少2.40个百分点，图像敏感度下降5.60个百分点。配对记录显示有155个新获得的视觉收益事件和203个丢失的视觉收益事件。更广泛的多模态适应产生的正确图像准确率低于语言模型LoRA SFT。标准GRPO产生混合奖励组，且

    arXiv:2609.31450v1 Announce Type: cross  Abstract: Medical vision-language model (VLM) post-training is commonly evaluated through answer accuracy. We examine how changes in accuracy and training objectives relate to image-conditioned decisions in a controlled Qwen2.5-VL-3B study on PMC-VQA. We compare supervised fine-tuning (SFT) with low-rank adaptation (LoRA) restricted to the language model, expanded multimodal adaptation scopes, standard answer-only Group Relative Policy Optimization (GRPO), and a counterfactual evidence objective. On 2,000 clean-test questions, language model LoRA SFT changes correct-image accuracy by +1.10 percentage points (95% paired bootstrap CI:-0.85 to +3.05), while visual-benefit events decrease by 2.40 points and image sensitivity decreases by 5.60 points. Paired records reveal 155 acquired and 203 lost visual-benefit events. Broader adaptation yields lower correct-image accuracy than language-model LoRA SFT. Standard GRPO produces mixed-reward groups and
    
[^21]: ViSTA：一个简单的桥梁，将视觉对齐扩展至多模态大语言模型的临床时间序列理解

    ViSTA: A Simple Bridge Extends Visual Alignment to Clinical Time-Series Understanding in Multimodal LLMs

    [https://arxiv.org/abs/2609.31448](https://arxiv.org/abs/2609.31448)

    ViSTA是一种轻量级适配器，通过将不规则的数值型临床时间序列数据融入预训练视觉-语言模型的图表表示中，在冻结全部预训练参数的前提下实现临床风险预测性能的大幅提升。

    

    临床预测模型根据患者的测量数据估计风险，而大语言模型则支持医学文本理解和问答。然而，它们的语言能力并不能保证能够从结构化、高维的临床时间序列中进行准确预测。提升这一能力将把风险估计与对患者病情演变的灵活提问连接起来。我们提出了ViSTA，这是一个紧凑的适配器，将不规则的数值测量数据融入预训练视觉-语言模型的图表表示中。它在保持所有预训练参数完全不变的情况下，学习对视觉token的校正。在MIMIC-IV数据集上，对于急性肾损伤和死亡率预测，在参数规模为20亿至90亿的模型中，ViSTA在所有四项指标上均获得了所比较的适配方法中最高的平均分数。仅使用51.6万个可训练参数，20亿参数的模型在急性肾损伤预测中达到了0.7376的ROC曲线下面积。

    arXiv:2609.31448v1 Announce Type: cross  Abstract: Clinical prediction models estimate risk from patient measurements, while large language models support medical text understanding and question answering. Yet their language capabilities do not ensure accurate prediction from structured, high-dimensional clinical time series. Improving this ability would connect risk estimation with flexible questions about a patient's evolving condition. We introduce ViSTA, a compact adapter that incorporates irregular numerical measurements into a pretrained vision-language model's chart representations. It learns corrections to visual tokens while leaving all pretrained parameters unchanged. On MIMIC-IV, ViSTA has the highest mean scores among the compared adaptations on all four metrics for acute kidney injury and mortality prediction across models with 2-9 billion parameters. With 0.516 million trainable parameters, the 2-billion-parameter model reaches an area under the ROC curve of 0.7376 for ac
    
[^22]: 用于高光谱视频压缩的隐式神经表示

    Implicit Neural Representation for Hyperspectral Video Compression

    [https://arxiv.org/abs/2609.31435](https://arxiv.org/abs/2609.31435)

    该论文提出了一种基于隐式神经表示的高光谱视频压缩新方法，通过对现有RGB视频压缩模型进行新颖扩展，相比传统逐帧压缩方法实现了+4.99 dB的PSNR增益和-88.88%的码率降低，同时显著提升了下游目标跟踪任务的性能。

    

    随着快照相机的出现，高光谱视频正变得越来越容易获取。近年来，新应用的涌现导致数据集规模日益增大。然而，高光谱视频压缩仍处于早期发展阶段。在本研究中，我们探索将隐式神经表示作为一种候选解决方案。我们对现有的RGB视频压缩模型提出了一种新颖的扩展方法，与逐帧应用的传统高光谱图像压缩方法相比，实现了+4.99 dB的Bjøntegaard Delta PSNR增益和-88.88%的Bjøntegaard Delta码率降低。除了重建质量之外，本研究还以目标跟踪成功率的形式衡量了压缩对下游任务性能的影响。与在低数据量情况下基于主成分分析和JPEG2000的方法压缩的视频相比，我们提出的方法将跟踪曲线下面积最多提升了23.42%，距离精度也有相应提升。

    arXiv:2609.31435v1 Announce Type: cross  Abstract: With the advent of snapshot cameras, hyperspectral video is becoming more readily available. In recent years, new applications have emerged which have led to increasingly larger datasets. However, hyperspectral video compression remains in the early stages. In this study, we explore the use of implicit neural representation as a candidate solution. We propose a novel extension of an existing RGB video compression model, achieving Bj{\o}ntegaard Delta PSNR gains of +4.99 dB and Bj{\o}ntegaard Delta rate of -88.88% compared to traditional hyperspectral image compression methods applied frame-by-frame. In addition to reconstruction quality, the effects on downstream task performance are measured in the form of object tracking success. Compared to video compressed with methods based on principal component analysis and JPEG2000 in low data regimes, our proposed method improves tracking area under the curve by up to 23.42% and distance preci
    
[^23]: 压缩你所看到的，而非你所说的：面向潜在观察软件工程代理的锚定上下文蒸馏

    Compress What You See, Not What You Say: Anchored Context Distillation for Latent-Observation Software Engineering Agents

    [https://arxiv.org/abs/2609.31430](https://arxiv.org/abs/2609.31430)

    提出LOHA上下文布局与ACD锚定蒸馏训练方法，将较早的工具观察压缩为软令牌、保留近期文本，在大幅压缩软件工程代理上下文的同时保留操作关键信息并约束行为漂移。

    

    工具观察占据了软件工程代理上下文的主导地位，使得长期交互历史的维护成本高昂。现有的上下文压缩方法可能会丢弃后续操作所需的信息，而让代理适应软令牌（soft-token）表示又可能损害其原有行为。为了在保留操作关键信息和代理行为的同时减少上下文，我们结合了两项技术：潜在观察与硬动作（LOHA）——一种将压缩历史与精确引用所需文本分开的上下文布局；以及锚定上下文蒸馏（ACD）——一种在实现潜在读取的同时约束行为漂移的训练方法。LOHA将较早的工具观察压缩为软令牌，同时以文本形式保留代理自身的回合和最后K个观察，从而提供对历史信息的紧凑访问以及对最近内容的精确访问。为了使代理能够使用这种表示，ACD蒸馏基础模型的（原文在此处截断）

    arXiv:2609.31430v1 Announce Type: new  Abstract: Tool observations dominate the context of software-engineering agents, making long interaction histories costly to maintain. Existing context compression methods can discard information needed by later actions, while adapting agents to soft-token representations can compromise their original behavior. To reduce context while preserving action-critical information and agent behavior, we combine Latent Observations, Hard Actions (LOHA), a context layout that separates compressed history from text needed for exact reference, with Anchored Context Distillation (ACD), a training method that enables latent reading while constraining behavioral drift. LOHA compresses older tool observations into soft tokens while retaining the agent's own turns and the last K observations in text, providing compact access to historical information and exact access to recent content. To enable the agent to use this representation, ACD distills the base model's f
    
[^24]: 缓解伪造共识：面向多智能体辩论综合的主动溯源门控

    Towards Mitigating Fabricated Consensus: The Active Provenance Gate for Multi-Agent Debate Synthesis

    [https://arxiv.org/abs/2609.31422](https://arxiv.org/abs/2609.31422)

    提出主动溯源门控（APG），通过将来源作为硬约束、审计辩论中的每一条论断并自我纠错，来缓解多智能体辩论综合阶段摘要模型伪造无事实依据“辩论共识”的安全问题。

    

    基于大语言模型的多智能体辩论（MAD）系统正日益被用作分布式流程中的复杂决策管道，然而其最终综合阶段仍然缺乏足够的控制。即使拥有详细的辩论记录，摘要模型也容易编造出文笔流畅、但并未建立在辩论历史基础之上的“辩论共识”。为解决这一安全缺口，本文开展实证研究，探讨引入辩论后的主动验证能否在减少此类缺乏事实依据的摘要产生的同时，仍提供有价值的信息；此外，还检验了在缺乏可靠折中方案的情况下，显式发出分歧信号是否更为可取。本文提出了主动溯源门控作为辩论后验证层，它将来源作为硬约束，分析辩论记录、审计每一条论断，并应用自我纠错。

    arXiv:2609.31422v1 Announce Type: cross  Abstract: Large language model-based multi-agent debate (MAD) systems are being increasingly used as complex decision pipelines in distributed processes. However, their final synthesis phase still remains inadequately controlled. Even with detailed debate logs, summarizing models are prone to fabricating smoothly written debate consensus that is not grounded in the debate's history. To address this safety gap, this paper presents empirical research and studies if the introduction of active post-debate verification can mitigate the production of such factually unsupported summaries, while still providing valuable information. Furthermore, it is examined whether explicitly signalling divergence is preferable in the absence of a reliable compromise. The Active Provenance Gate (APG) is introduced as a post-debate verification layer that treats the source as a hard constraint, analysing the debate logs, auditing each claim, and applying self-correcti
    
[^25]: 对不起机器人，开心人类：视觉语言模型只能读取两层可读印刷文字中的一层

    Sorry Robot, Happy Human: Vision-Language Models Read Only One of Two Legible Typographic Layers

    [https://arxiv.org/abs/2609.31403](https://arxiv.org/abs/2609.31403)

    本研究创建了DecoyBench数据集，发现视觉语言模型在包含两层叠加可读文字的图像中只能读取轮廓文字而几乎无法提取阴影文字，而人类可以高准确率读取两层，揭示了VLM在多文本层图像理解上的根本性脆弱。

    

    尽管视觉语言模型（VLM）在光学字符识别（OCR）任务中取得了成功，但它们容易受到排版攻击，并且对包含多个文本层的图像结构脆弱。在本研究中，使用诱饵字体（Decoy Font）方法创建了DecoyBench数据集。该数据集由300张图像组成，每张图像包含带有清晰轮廓线的文字，叠加在另一段带有柔和阴影的文字之上。使用该数据集，在两种不同的提示条件（朴素提示和引导提示）以及两种不同的分辨率（512×512和64×64）下，评估了来自三个不同模型系列的六个最新闭源模型。验证研究表明，人类参与者能够以高准确率读取两层文本。相比之下，大多数模型变体在两种提示方法下，都能在高分辨率下以接近人类的准确率读取轮廓文字，但几乎从未完整提取出阴影文字。在低分辨率下，

    arXiv:2609.31403v1 Announce Type: cross  Abstract: Vision-language models (VLMs), despite their success in optical character recognition (OCR) tasks, are vulnerable to typographic attacks and have a fragile structure for images with multiple text layers. In this study, the DecoyBench dataset was created using the Decoy Font method. The dataset consists of 300 images, each containing text with sharp contour lines superimposed on another text with soft shading. Six recent closed-source models from three different model families were evaluated using this dataset under two different prompting conditions (naive and guided) and at two different resolutions ($512\times512$ and $64\times64$). A validation study showed that human participants could read both text layers with high accuracy. In contrast, the models, with most variants and both prompting methods, read the contour text with near-human accuracy at high resolution, but almost never fully extracted the shading text. At low resolution,
    
[^26]: Intent2Tc：基于语言模型的意图到流量控制自动转换

    Intent2Tc: Automated Intent-to-Traffic Control Translation with Language Models

    [https://arxiv.org/abs/2609.31397](https://arxiv.org/abs/2609.31397)

    该论文提出Intent2Tc，一个由语言模型驱动的闭环框架，能够将业务级流量整形意图自动转换为经过验证的可执行Linux流量控制配置，并通过数字孪生语义模型、批判驱动精炼和RAG知识复用显著提升语义一致性与配置可靠性。

    

    自动化且高度可用的服务质量保障需要将高层服务意图转换为可部署的流量管理策略。尽管基于意图的网络简化了策略的规范定义，但在业务级意图与可执行网络配置之间架起桥梁仍然复杂、易出错且难以自动化。本文提出了Intent2Tc，一个由语言模型驱动的闭环框架，它将业务级的流量整形意图转换为声明式的子意图，进而转换为经过验证的、可执行的Linux流量控制配置。该框架集成了基于主动队列管理的数字孪生语义模型、自动化元数据提取、基于批判的迭代精炼以及基于检索增强生成的知识复用，以提高语义一致性和配置的可靠性。我们评估了多个开源大语言模型……

    arXiv:2609.31397v1 Announce Type: cross  Abstract: Automated and highly usable Quality-of-Service (QoS) enforcement requires translating high-level service intents into deployable traffic-management policies. Although intent-based networking (IBN) has simplified policy specification, bridging the gap between business-level intents and executable network configurations remains complex, error-prone, and difficult to automate. This paper presents Intent2Tc, a closed-loop language-model-driven framework that translates business-level traffic-shaping intents into declarative sub-intents and subsequently into validated, executable Linux traffic control (tc) configurations. The framework integrates an Active Queue Management (AQM)-based digital twin (DT) semantic model, automated metadata extraction, critique-driven refinement, and Retrieval-Augmented Generation (RAG)-based knowledge reuse to improve semantic consistency and configuration reliability. We evaluate multiple open-source large la
    
[^27]: ActKV：通过动作引导的KV缓存管理实现高效的LLM智能体

    ActKV: Efficient LLM Agents through Action-Guided KV Cache Management

    [https://arxiv.org/abs/2609.31395](https://arxiv.org/abs/2609.31395)

    ActKV是首个专为智能体式LLM推理设计的KV缓存压缩框架，通过基于动作贡献的缓存逐出机制和置信度驱动的预算分配，在降低内存开销的同时优先保障动作质量与任务进展。

    

    智能体式LLM推理在迭代的观察-推理-动作循环中会不断累积冗长的KV缓存，带来巨大的内存开销并限制服务吞吐量。现有的压缩方法强调整体输出质量，却忽视了动作在推动任务进展中的不对称重要性。我们的核心思想是建立一种压缩标准，根据KV条目对动作生成的贡献来评估其价值，并优先保障动作质量。然而，迭代执行、动态内存需求以及分散的动作关键条目给逐出策略、预算分配和分页内存集成带来了挑战。为此，我们提出了ActKV，这是首个专为智能体式LLM推理设计的KV缓存压缩框架。(i) 面向动作的KV缓存逐出机制利用稳定的动作访问模式来保留对未来动作至关重要的条目，从而在压缩条件下支持可靠的任务进展。(ii) 置信度驱动的自适应

    arXiv:2609.31395v1 Announce Type: cross  Abstract: Agentic LLM inference accumulates long KV caches across iterative observation-reasoning-action loops, imposing substantial memory overhead and limiting serving throughput. Existing compression methods emphasize overall output quality, overlooking the asymmetric importance of actions in driving task progress. Our key idea is to establish a compression criterion that values KV entries by their contribution to action generation and prioritizes action quality. However, iterative execution, dynamic memory demands, and scattered action-critical entries pose challenges to eviction policies, budget allocation, and paged memory integration. To this end, we propose ActKV, the first KV cache compression framework tailored for agentic LLM inference. (i) Action-oriented KV cache eviction exploits stable action access patterns to retain entries critical to future actions, supporting reliable task progress under compression. (ii) Confidence-driven ad
    
[^28]: 基于端点约束轨迹优化的端到端驾驶模型引导

    Guiding End-to-End Driving Models with Endpoint-Constrained Trajectory Optimization

    [https://arxiv.org/abs/2609.31383](https://arxiv.org/abs/2609.31383)

    该论文发现端到端驾驶模型开环与闭环性能差距的一个新因素是中间路径点监督缺乏物理连贯性和可跟踪性，并提出轻量级后处理层ECO，通过锚定车辆执行历史、保留可靠的预测端点并重塑中间路径点来弥合这一差距。

    

    端到端驾驶策略通常通过开环行为克隆进行训练，但在部署到车辆上时最终必须以闭环方式运行，这导致训练与执行之间存在根本性的不匹配。除了已被广泛研究的协变量偏移和因果混淆效应之外，我们为这种开环/闭环差距识别出一个补充性因素：基于路径点的监督方式和位移度量无法保证中间轨迹在物理上是连贯的，也难以保证控制器能够顺利跟踪。我们观察到，这些不一致性主要集中在中间路径点上，而预测的端点则相对可靠。基于这一观察，我们提出了端点约束优化，这是一种轻量级的后处理层，它将轨迹锚定在车辆已执行的历史轨迹上，保留策略预测的端点，并重塑中间路径点以改进……

    arXiv:2609.31383v1 Announce Type: cross  Abstract: End-to-end driving policies are commonly trained through open-loop behavior cloning, yet they must ultimately operate in closed-loop when deployed on a vehicle, creating a fundamental mismatch between training and execution. Beyond the commonly studied effects of covariate shift and causal confusion, we identify a complementary factor for this open-loop/closed-loop gap: waypoint-based supervision and displacement metrics do not ensure that the intermediate trajectory is physically coherent or easy for the controller to track. We observe that these inconsistencies concentrate primarily at intermediate waypoints, while the predicted endpoint remains comparatively reliable. Based on this observation, we introduce Endpoint-Constrained Optimization (ECO), a lightweight postprocessing layer that anchors the trajectory to the vehicle's executed history, preserves the policy's predicted endpoint, and reshapes the intermediate waypoints to impr
    
[^29]: 先标注后总结：学习压缩证据以实现长上下文理解

    Highlight-Then-Summarize: Learning to Compress Evidence for Long-Context Understanding

    [https://arxiv.org/abs/2609.31382](https://arxiv.org/abs/2609.31382)

    提出“先标注后总结”（H2S）的先压缩后推理范式，通过带过程级奖励的强化学习训练模型先识别并压缩与问题相关的证据、生成条件化摘要后再作答，显著提升大模型的长上下文理解能力。

    

    长上下文理解要求大语言模型（LLM）能够对长篇文档、对话和代码进行推理，然而与任务相关的证据往往稀疏且分散在大量无关和冗余内容之中。我们提出了先标注后总结方法，这是一种“先压缩后推理”的范式：首先识别有出处依据的、与问题相关的证据，然后将其整合为一个紧凑的、以问题为条件的摘要，最后再生成最终答案。为了训练这种行为，我们构建了 H2S 数据集，包含来自 11 个基准系列的 6,647 个样本，平均上下文长度为 43.9K tokens；并提出了 H2S-RL 方法，除了最终答案的正确性之外，还为证据选择和摘要构建过程提供过程级奖励。我们在 H2S-Bench（一个包含七项任务的长上下文评测套件）上进行评估。在共享的 128K 输入和 4K 输出预算下，H2S-14B 取得了平均 32.60 的分数，优于 Qwen3.8-2（原文在此处截断）。

    arXiv:2609.31382v1 Announce Type: cross  Abstract: Long-context understanding requires large language models (LLMs) to reason over lengthy documents, conversations, and code, yet task-relevant evidence is often sparse and scattered amid substantial irrelevant and redundant content. We propose Highlight-Then-Summarize (H2S), a compress-then-reason paradigm that first identifies source-grounded, question-relevant evidence and then integrates it into a compact, question-conditioned summary before producing the final answer. To train this behavior, we construct H2S-Dataset, comprising 6,647 examples from 11 benchmark families with an average context length of 43.9K tokens, and introduce H2S-RL, which provides process-level rewards for evidence selection and summary construction in addition to final-answer correctness. We evaluate on H2S-Bench, a seven-task long-context suite. Under a shared 128K input and 4K output budget, H2S-14B achieves an average score of 32.60, outperforming Qwen3.8-2
    
[^30]: 已完成的配对掩盖了被截断的失败：选择性上下文投影的ReVerPi案例研究

    Completed Pairs Hide Capped Failures: A ReVerPi Case Study of Selective Context Projection

    [https://arxiv.org/abs/2609.31381](https://arxiv.org/abs/2609.31381)

    该论文通过ReVerPi案例研究揭示了选择性上下文投影的关键权衡：已完成的15对配对显示投影与完整上下文成功率相同并节省25%逻辑令牌，但恢复被抑制的全部27次边界运行后，投影相对完整的成功差异介于-9到+1个任务之间，且单个投影运行可能因归档检索而耗尽多达12次请求。

    

    arXiv:2609.31381v1 公告类型：新论文。摘要：上下文投影用紧凑、可寻址的摘录替换较旧的工具观测结果，在减少重复输入的同时可能增加证据检索的轮次。我们在ReVerPi中研究这一权衡，ReVerPi是一个具备观测归档以及匹配的完整/投影续跑的Pi扩展。在一项包含86次运行、641次模型请求的源代码阅读实验中，15对已完成的配对显示出相同的成功率：每臂均为12/15。另有12次边界运行中途停止，运行器在第一条臂未能完成时会抑制其配对运行。恢复全部27次边界运行后，投影相对完整的成功差异被界定在-9到+1个任务之间。一个被省略的、由选择器选定的投影续跑虽成功检索到归档文本，却耗尽了12次请求；而其对应的完整运行仅需3次即可作答。在此记录框架内，11对双方均正确的配对构成一个完全观测到的成功层级：投影使聚合逻辑令牌减少25%，同时增加了……（原文摘要在此截断）

    arXiv:2609.31381v1 Announce Type: new  Abstract: Context projection replaces older tool observations with compact, addressable excerpts, reducing repeated input while potentially adding evidence-retrieval turns. We study this trade-off in ReVerPi, a Pi extension with archived observations and matched full/projected continuations. In an 86-run source-reading campaign with 641 model requests, the 15 completed pairs show identical success: 12/15 per arm. Twelve further boundary runs stop, with the runner suppressing the companion whenever the first arm fails to complete. Restoring all 27 boundary runs bounds projected-minus-full success between $-$9 and +1 tasks. One omitted, selector-chosen projected continuation successfully retrieves archive text yet exhausts twelve requests; its full counterpart answers in three. The eleven jointly correct pairs form a fully observed success stratum within this recorded frame: projection reduces aggregate logical tokens by 25%, while increasing the me
    
[^31]: 从大脑皮层区域的视角审视大语言模型中的“层程序”

    Programs-of-Layers in LLMs through the Lens of Cortical Areas

    [https://arxiv.org/abs/2609.31360](https://arxiv.org/abs/2609.31360)

    该研究在皮层区域的视角下复现并验证了PoLar方法：将大语言模型的各层视为可动态路由的函数库，根据输入难度自适应地跳过或重复层块，可在多个模型上获得优于固定逐层推理的性能。

    

    大语言模型的推理通常采用固定深度、固定顺序的前向传播，无论输入难易程度如何都要经过每一层。人脑的工作方式并非如此：它以丘脑作为中央枢纽，根据需求将信息灵活地路由到皮层的各个区域。Li等人（2026）最近通过一个名为“层程序”（Program-of-Layers，PoLar）的系统表明，如果将Transformer的各层视为一个函数库而非固定序列，就可以赋予模型类似的灵活性。当每个输入被动态路由经过自适应的层块序列（跳过或重复连续的层块）时，性能会优于标准前向传播。我们比原论文更详细地重构了PoLar的诊断性蒙特卡洛树搜索（MCTS），并将其应用于5个模型。我们复现了PoLar的多项发现：跳过层块优于标准前向传播，重复层块优于跳过，而两者结合则优于……（原文摘要在此处截断）

    arXiv:2609.31360v1 Announce Type: new  Abstract: Inference in LLMs is conventionally a fixed-depth, fixed-order forward pass through every layer, regardless of how difficult the input is. The human brain does not work this way: using the thalamus as a central hub, it routes information flexibly to all regions of the cortex according to demand. Li et al. (2026) recently showed, with a system they call program-of-layers (PoLar), that transformers can be given an analogous flexibility if their layers are treated as a library of functions rather than a fixed sequence. Performance improves over the standard forward pass when each input is dynamically routed through an adaptive sequence of skipped or repeated contiguous layer blocks. We reconstructed PoLar's diagnostic MCTS in more detail than the original paper and applied it across 5 models. We reproduced several of PoLar's findings: skipping outperformed the standard pass, repeating outperformed skipping, and combining both outperformed e
    
[^32]: 面向医疗AI Agent的安全有界SDC到MCP网关

    A Safety-Bounded SDC-to-MCP Gateway for Medical AI Agents

    [https://arxiv.org/abs/2609.31358](https://arxiv.org/abs/2609.31358)

    该论文提出一种安全有界的网关，将IEEE 11073 SDC医疗设备协议接入MCP，通过只读资源和策略验证的干运行工具，确保医疗AI Agent可以访问设备状态和元数据但绝不触发实际设备操作。

    

    模型上下文协议为AI应用程序提供了发现和使用外部资源与工具的通用接口，使语言模型Agent能够将推理建立在当前系统状态之上，并与异构服务交互。然而在医疗环境中，暴露设备状态和动作可供性需要对可能产生的影响施加确定性约束。我们提出了一种IEEE 11073面向服务的设备连通性（SDC）到MCP的网关，该网关将指标、报警、上下文引用和语义元数据作为只读资源暴露，同时将选定的动作可供性表示为经策略验证的干运行工具。“安全有界”一词表示一种狭义的无执行属性：面向Agent的请求不会分派任何SDC设备操作。Python原型支持模拟故障与生命周期实验、跨越独立Java和Python实现的软件参考协议路径，以及确定性的（摘要截断）

    arXiv:2609.31358v1 Announce Type: cross  Abstract: The Model Context Protocol (MCP) provides a common interface through which AI applications discover and use external resources and tools. It allows language-model agents to ground their reasoning in current system state and interact with heterogeneous services. In medical environments, however, exposing device state and action affordances requires deterministic constraints on possible effects. We present an IEEE 11073 Service-Oriented Device Connectivity (SDC)-to-MCP gateway that exposes metrics, alarms, context references, and semantic metadata as read-only resources, while representing selected action affordances as policy-validated dry-run tools. The term safety-bounded denotes a narrow no-execution property: agent-facing requests dispatch no SDC device operation. A Python prototype supports simulated fault and lifecycle experiments, a software-reference protocol path spanning independent Java and Python implementations, determinist
    
[^33]: 可变对话记录：通过可编辑的对话状态缓解上下文污染

    Mutable Transcripts: Mitigating Context Pollution through Editable Conversation State

    [https://arxiv.org/abs/2609.31354](https://arxiv.org/abs/2609.31354)

    提出可变对话记录这一新交互范式，允许用户通过自然语言编辑请求直接修改历史对话内容，从而将对话记录从被动记录转变为可编辑的对话状态，有效缓解静态对话历史导致的上下文污染问题。

    

    当代大语言模型（LLM）聊天系统将对话历史视为不可变的回合序列，这一序列定义了模型的工作上下文。然而，真实交互中的用户意图并非静态：它会随着纠正、细化和约束条件的变化而不断演进。动态意图与静态记录之间的这种不匹配可能导致“上下文污染”，即过时或无关的信息持续存在并继续影响后续响应。我们提出了“可变对话记录”这一新的交互范式，使用户能够通过自然语言编辑请求修改先前的对话回合，从而使对话历史本身可以被更新而非仅仅追加。这将对话记录从被动记录重新定义为对话状态的可编辑表示。我们构建了一个将记录级修订集成到标准聊天界面中的可用原型，并通过对照实验评估了其可行性。

    arXiv:2609.31354v1 Announce Type: new  Abstract: Contemporary large language model (LLM) chat systems treat conversation history as an immutable sequence of turns that defines the model's working context. However, user intent in real interactions is not static: it evolves through correction, refinement, and shifting constraints. This mismatch between dynamic intent and static transcripts can result in context pollution, where outdated or irrelevant information persists and continues to influence subsequent responses. We introduce mutable transcripts, a new interaction paradigm that enables users to revise prior turns through natural language edit requests, allowing the conversation history itself to be updated rather than appended. This reframes the transcript from a passive record into an editable representation of conversational state. We present a working prototype that integrates transcript-level revision into a standard chat interface and evaluate its feasibility through a control
    
[^34]: DyMD：在少步视频世界模型中通过分布匹配蒸馏保留交互动态

    DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models

    [https://arxiv.org/abs/2609.31349](https://arxiv.org/abs/2609.31349)

    DyMD提出一种自适应分布匹配蒸馏框架，通过时间亲和条件化的加噪采样来动态调整教师监督与评论家拟合，从而在少步视频世界模型中有效保留机器人—物体交互动态。

    

    大型视频扩散模型为具身预测与学习提供了富有表现力的先验，但其多步采样对于交互式下游应用而言成本高昂。分布匹配蒸馏（DMD）能够实现少步视频生成，但在保持视觉质量的同时可能会抑制机器人—物体运动。通过考察DMD的教师信号与伪分数信号，我们发现弱加噪会使教师后验集中在缺乏运动的生成结果附近，从而限制了运动恢复引导的能力；同时，运动更强的生成结果往往会产生更大的伪分数拟合误差，这可能阻碍生成器对交互动态的学习。为此，我们提出DyMD，一个使教师监督与评论家拟合均能适应不断演进的学生模型的DMD框架。基于时间亲和条件化的加噪采样通过将基础调度与教师先验混合，使时间步分布能够适应每个生成结果当前的交互保真度（摘要在此处截断）。

    arXiv:2609.31349v1 Announce Type: cross  Abstract: Large video diffusion models offer expressive priors for embodied prediction and learning, yet their many-step sampling remains costly for interactive downstream use. Distribution Matching Distillation (DMD) enables few-step video generation, but can suppress robot--object motion while preserving visual quality. Examining DMD's teacher and fake-score signals, we find that weak re-noising keeps the teacher posterior concentrated near motion-deficient rollouts, limiting motion-restoring guidance. Meanwhile, stronger-motion rollouts tend to incur larger fake-score fitting errors, which can hinder the generator's learning of interaction dynamics. We propose DyMD, a DMD framework that adapts both teacher supervision and critic fitting to the evolving student. Temporal affinity--conditioned re-noise sampling adapts the timestep distribution to each rollout's current interaction fidelity by mixing the base schedule with a teacher prior motiva
    
[^35]: 正确的信息抽取流水线取决于文档本身：小型本地模型的准确率-能耗权衡

    The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models

    [https://arxiv.org/abs/2609.31341](https://arxiv.org/abs/2609.31341)

    该研究在隐私约束下系统评估了小型本地模型信息抽取流水线的准确率-能耗权衡，发现最优选择（图像 vs 文本）取决于文档类型，批处理可无损降低38-85%能耗，而神经OCR能耗高达传统OCR的17倍。

    

    arXiv:2609.31341v1 公告类型：新论文 摘要：信息抽取流水线应处理页面图像还是解析后的文本，取决于具体的文档，且答案会随着版面复杂度的变化而翻转。我们在一个排除（封闭式）云服务的约束下研究这一权衡：隐私敏感文档由小型（参数量≤80亿）纯文本模型和视觉-语言模型在本地部署处理，并在涵盖输入表示、模型家族和推理配置的设计空间中同时评估准确率与能耗。通过对接近纯文本的 Kleister-NDA 合同和版面丰富的 VRDU 表单进行基准测试，我们发现批处理是主要的节能杠杆，可在不损失准确率的情况下将每页能耗降低38-85%；而 FP8 量化在逐个处理请求时可节省27-32%的能耗，但在应用批处理后每页节省不足1毫瓦时（9-19%）。预处理主导了剩余的能耗：神经 OCR 每页的能耗是传统 OCR 的17倍，并且几……

    arXiv:2609.31341v1 Announce Type: new  Abstract: Whether an information extraction pipeline should process page images or parsed text depends on the document, and the answer flips across the layout spectrum. We study this trade-off under a constraint that rules out (closed) cloud services: privacy-sensitive documents processed on-premise by small ($\le 8\mathrm{B}$ parameter) text-only and vision--language models, evaluated on both accuracy and energy over a design space spanning input representation, model family, and inference configuration. Benchmarking on the near-plain-text Kleister-NDA contracts and the layout-rich VRDU forms, we find that batching is the dominant energy lever, cutting energy per page by 38-85% at no cost in accuracy, while FP8 quantization saves 27-32% when requests are served one at a time but less than 1mWh per page (9-19%) once batching is applied. Preprocessing dominates what remains: neural OCR costs $17\times$ more energy per page than classical OCR and ne
    
[^36]: CG-HAF：一种面向智能护肤辅助中序数型痤疮严重程度分级的可解释全局-局部皮损负担融合框架

    CG-HAF: An Interpretable Global-Local Lesion-Burden Fusion Framework for Ordinal Acne Severity Grading in Agentic Skincare Support

    [https://arxiv.org/abs/2609.31326](https://arxiv.org/abs/2609.31326)

    CG-HAF通过将独立分类器的整体严重程度概率与目标检测器提取的结构化皮损负担特征进行显式融合，并由轻量级可解释分类器输出最终痤疮严重程度等级，在基准上显著优于仅依赖全局证据的方法，且在最严重病例中收益最大。

    

    序数型痤疮严重程度分级需要在区分视觉上相似的相邻等级的同时，综合权衡整体面部外观与局部皮损负担——而大多数现有方法将这些证据压缩为单一的不透明表示。我们提出了CG-HAF，一个保持证据显式性的全局-局部融合框架：将独立训练的分类器所输出的平均整体严重程度概率，与目标检测器提供的结构化皮损负担描述符（皮损数量、检测置信度、皮损面积）结合为一个紧凑表示，再由一个轻量级、可解释的分类器产生最终等级。在一个广泛使用的基准上，这种融合相比仅使用全局证据的基线方法取得了明确且有统计学支持的性能提升，其中在最严重的病例上提升最大。在一个采用不同分级标准的独立数据集上的测试表明，其强大的数据集内表现……（摘要原文在此处截断）

    arXiv:2609.31326v1 Announce Type: cross  Abstract: Ordinal acne severity grading requires distinguishing visually similar neighboring grades while jointly weighing holistic facial appearance and localized lesion burden - evidence that most existing approaches collapse into a single opaque representation. We introduce CG-HAF, a global-local fusion framework that instead keeps this evidence explicit: averaged holistic severity probabilities from independently trained classifiers are combined with structured lesion-burden descriptors from an object detector (lesion count, detection confidence, lesion area) into a compact representation, from which a lightweight, interpretable classifier produces the final grade. On a widely used benchmark, this fusion yields a clear, statistically supported improvement over global-evidence-only baselines, with the largest gains on the most severe cases. Testing on an independent dataset with a different grading standard shows that strong within-dataset pe
    
[^37]: AgentXploit：面向AI智能体的从代码仓库到运行时的自主红队测试

    AgentXploit: Autonomous Repository-to-Runtime Red-Teaming for AI Agents

    [https://arxiv.org/abs/2609.31318](https://arxiv.org/abs/2609.31318)

    AgentXploit提出了一种将代码仓库层面的攻击路径发现与运行时漏洞利用相分离的双角色自主红队测试系统，用于在部署前对AI智能体进行授权白盒安全审计。

    

    AI智能体将语言模型与外部数据和工具相结合，这些工具能够修改文件、调用API或执行代码。当对抗性内容改变智能体的工具使用方式，或当其周边软件存在诸如路径穿越或命令注入等漏洞时，就可能产生安全故障。我们研究授权的白盒部署前审计，即审计者可以访问目标代码仓库和受控的运行时环境，但成功的攻击仍必须通过任务定义的攻击者接口来实施，并需经外部验证器确认。我们提出了AgentXploit，一个将代码仓库层面的攻击路径发现与运行时漏洞利用相分离的双角色审计系统。分析智能体（Analyzer Agent）将攻击者可控的输入追踪到敏感操作，并记录有代码支撑的候选攻击路径；利用智能体（Exploiter Agent）将这些路径转化为具体攻击，并利用运行时反馈对其进行修订。我们还引入了AgentXploit-B（注：原文摘要在此处截断）。

    arXiv:2609.31318v1 Announce Type: cross  Abstract: AI agents combine language models with external data and tools that can modify files, call APIs, or execute code. Security failures can arise when adversarial content changes an agent's tool use or when the surrounding software contains vulnerabilities such as path traversal or command injection. We study authorized white-box pre-deployment auditing, where the auditor has access to the target repository and a controlled runtime, but successful attacks must still act through the task-defined attacker interface and be confirmed by an external verifier. We present AgentXploit, a two-role auditing system that separates repository-level attack-path discovery from runtime exploitation. The Analyzer Agent traces attacker-controlled inputs to sensitive operations and records code-supported candidate attack paths; the Exploiter Agent turns these paths into concrete attacks and revises them using runtime feedback. We also introduce AgentXploit-B
    
[^38]: 迈向VLA-Dreamer：利用世界模型改进VLA行为

    Towards VLA-Dreamer: Refining VLA Behavior Using World Models

    [https://arxiv.org/abs/2609.31313](https://arxiv.org/abs/2609.31313)

    该论文提出在VLA视觉编码器的嵌入空间上训练预测性世界模型，以提升VLA的样本效率，并验证这些嵌入能否基于动作进行未来预测，从而判断VLA是否具备隐式世界模拟能力。

    

    视觉-语言-动作模型（VLA）在机器人控制方面展现出强大潜力，但需要大量高质量的模仿学习数据。此外，缺乏显式世界模型进一步使人们对其控制能力产生质疑。在这篇概念性论文中，我们提出了一种新颖的架构，通过在VLA视觉编码器的嵌入空间上训练预测性世界模型来解决VLA的样本效率问题。我们假设这些嵌入是与动作相关且可用于未来预测的。为此，我们建议使用所提出的架构来研究这些嵌入基于动作预测未来的能力，因为如果无法做到这一点，将标志着VLA架构的一个关键局限性：缺乏一个无损的隐式世界模型来模拟真实世界的动态。所提出的架构与标准世界模型动力学不同，因为其损失来自嵌入空间而非（原文在此处截断）。

    arXiv:2609.31313v1 Announce Type: cross  Abstract: Vision-Language-Action models (VLAs), while showing strong potential for robot control, require massive amounts of high-quality imitation learning data. Moreover, the absence of an explicit world model casts further doubt on their control capabilities. In this concept paper, we propose a novel architecture that addresses sample efficiency in VLAs by training a predictive world model on the embedding space of the VLA's vision encoder. We hypothesize that these embeddings are action-relevant and usable for future prediction. To this end, we propose using the suggested architecture to investigate how well these embeddings predict the future based on actions, as the inability to do so would mark a key limitation of VLA architectures: the lack of a non-lossy implicit world model to simulate real-world dynamics. The proposed architecture differs from the standard world model dynamics as the loss comes from the embedding space rather than the
    
[^39]: 超越已批准的操作：智能体工作流中持久性结果的运行时验证

    Beyond Approved Actions: Runtime Validation of Persistent Outcomes in Agent Workflows

    [https://arxiv.org/abs/2609.31301](https://arxiv.org/abs/2609.31301)

    提出了EffectMatch运行时系统，通过在受控执行边界内收集智能体操作产生的持久性变更，并将其与应用批准的内容进行比对验证后再决定提交，有效阻止了未批准的副作用在智能体工作流中传播。

    

    大型语言模型智能体越来越多地在软件系统上执行操作，不再仅仅生成文本，而是直接改变数据库和在线服务。然而，一个已获批准的数据库更新可能成功执行，却留下一个未获批准的通知，因为执行过程可能产生超出所请求变更的持久性影响。当前的保障机制可以批准某个操作或记录其后果，但若不在继续执行前检查持久性结果，未获批准的结果可能被误认为成功并传播到后续步骤中。我们提出了EffectMatch，一个运行时系统，它在受控的执行边界内收集持久性变更，并将其与应用程序针对当前状态和执行所批准的内容进行比较，该比较决定了提交以及依赖执行能否继续进行。在206个公开业务任务上的对比评估中，EffectMatch保留了所有正常的执行，并阻止了所有经过测试的错误提交。六项20次运行的消融实验揭示了……（摘要原文在此处截断）

    arXiv:2609.31301v1 Announce Type: cross  Abstract: Large language model agents increasingly act on software systems, no longer merely generating text but also changing databases and online services. However, an approved database update may succeed yet leave an unapproved notification because execution can produce persistent effects beyond the requested change. Current safeguards can approve an action or record its aftermath, but without checking the persistent result before continuation, an unapproved outcome can be accepted as success and propagated to later steps. We present EffectMatch, a runtime that collects persistent changes within a controlled execution boundary and compares them with what the application approved for the current state and execution. The comparison governs commit and dependent execution. In comparative evaluation on 206 public business tasks, EffectMatch preserved all clean executions and prevented all tested incorrect commits. Six 20-run ablations exposed the 
    
[^40]: UniAR：一种通过多视图提示学习增强的自闭症识别统一框架

    UniAR: A Unified Framework for Autism Recognition Enhanced by Multi-View Prompt Learning

    [https://arxiv.org/abs/2609.31298](https://arxiv.org/abs/2609.31298)

    UniAR提出了一种融合多粒度提示学习的统一自闭症识别框架，利用大型多模态模型生成词、短语、句子三层级的诊断描述以弥补临床文本数据稀缺，并通过基于专家混合的多尺度对齐模块将生成语义与视觉证据对齐，实现异构数据下稳健的自闭症谱系障碍识别。

    

    自闭症谱系障碍（ASD）是一种复杂的神经发育障碍，早期准确诊断对于改善长期发育结果至关重要。然而，现有的ASD识别方法通常受限于诊断文本数据的稀缺，迫使其主要依赖视觉分析，限制了其建模具有临床意义的语义推理的能力。为应对这一挑战，我们提出了UniAR，一个通过多粒度提示学习增强的统一框架，用于在异构数据变化下实现稳健的ASD识别。具体而言，UniAR利用大型多模态模型在词、短语和句子层面生成分层诊断描述，弥补了成对临床报告的缺乏。为了将生成的语义与视觉证据对齐，我们进一步设计了基于专家混合的多尺度对齐模块，该模块动态匹配向量量化……（摘要在此处被截断）

    arXiv:2609.31298v1 Announce Type: cross  Abstract: Autism Spectrum Disorder (ASD) is a complex neurodevelopmental disorder for which early and accurate diagnosis is critical to improving long-term developmental outcomes. However, existing ASD recognition methods are often constrained by the scarcity of diagnostic text data, forcing them to rely mainly on visual analysis and limiting their ability to model clinically meaningful semantic reasoning. To address this challenge, we propose UniAR, a unified framework enhanced by multi-granularity prompt learning for robust ASD recognition under heterogeneous data variations. Specifically, UniAR leverages a large multimodal model to generate hierarchical diagnostic descriptions at the word, phrase, and sentence levels, compensating for the lack of paired clinical reports. To align the generated semantics with visual evidence, we further design a Mixture-of-Experts-based Multi-Scale Alignment Module, which dynamically matches vector-quantized v
    
[^41]: 面向输出头量化的Softmax重参数化

    Softmax Reparameterization for Output-Head Quantization

    [https://arxiv.org/abs/2609.31291](https://arxiv.org/abs/2609.31291)

    提出softmax重参数化这一训练后量化方法，通过在量化前减去词表行均值的标量倍数来选取功能等价的输出头，在保持全精度softmax分布不变的同时显著降低输出头量化对语言模型预测的扭曲。

    

    arXiv:2609.31291v1 公告类型：交叉 摘要：大型词表使得输出头成为小型语言模型中相当可观的推理成本。我们提出softmax重参数化，这是一种训练后方法，可在量化之前选择一个功能等价的输出头。该方法从每个输出行中减去词表行均值的标量倍数，并分别针对RTN、激活加权MSE和全Hessian GPTQ通过验证集KL散度来选择系数。这一维搜索涵盖了原始输出头和固定均值中心化两种情形，保持全精度softmax分布不变，且不改动已训练的解码器；秩一修正则可处理诸如soft-capping之类的非线性logit路径。在七个输出头上的实验表明，W4量化收益集中在基线量化严重扭曲预测的场景：在Phi-4-mini上，AW-MSE的KL散度从0.936降至0.256。这些收益在更强的GPTQ校准下依然保持，并与精确的逐通道缩放和仿射量化互为补充。

    arXiv:2609.31291v1 Announce Type: cross  Abstract: Large vocabularies make output heads a substantial inference cost in small language models. We propose softmax reparameterization, a post-training method that selects a functionally equivalent output head before quantization. The method subtracts a scalar multiple of the vocabulary-row mean from every output row and selects the coefficient by validation KL separately for RTN, activation-weighted MSE, and full-Hessian GPTQ. This one-dimensional search includes the original head and fixed mean-centering, preserves the full-precision softmax distribution, and leaves the trained decoder unchanged; a rank-one correction handles nonlinear logit paths such as soft-capping. Across seven heads, W4 gains concentrate where baseline quantization substantially distorts predictions: on Phi-4-mini, AW-MSE KL falls from 0.936 to 0.256. The gains survive stronger GPTQ calibration and remain complementary to exact per-channel scaling and affine quantiza
    
[^42]: G2MAF：多智能体流策略的测试时梯度引导

    G2MAF: Test-Time Gradient Guidance for Multi-Agent Flow Policies

    [https://arxiv.org/abs/2609.31286](https://arxiv.org/abs/2609.31286)

    提出G2MAF框架，通过测试时应用全局归一化的投影评论家梯度来引导和协调多智能体流策略修正联合动作，在保持动作可行性的同时，在MPE和SMAC基准上分别取得9.2%和8.9%的平均相对性能提升。

    

    离线多智能体强化学习（MARL）从固定数据集中学习协作策略，无需进一步的环境交互，且学习到的策略在部署时被冻结。这种冻结策略通常提出单一的联合动作并在部署时直接执行。然而，这种一次性部署常常提交次优的动作提议，即使附近存在更好的替代方案且这些方案仍与行为数据保持一致。为解决这一问题，我们提出了梯度引导多智能体流（G2MAF），一个在测试时优化联合策略的精炼框架。G2MAF应用一个全局归一化、投影后的评论家（critic）梯度来引导和协调所有智能体的修正，同时保持动作既可行又接近冻结策略的提议。在24个MPE和SMAC设置中，其标准变体改进了20个冻结策略设置，在MPE上的平均相对增益为9.2%，在SMAC上为8.9%，且模型推理延迟较低。

    arXiv:2609.31286v1 Announce Type: new  Abstract: Offline multi-agent reinforcement learning (MARL) learns cooperative policies from fixed datasets without further environment interaction and a learned policy is frozen at deployment. Such a frozen policy typically proposes a single joint action and executes it directly at deployment time. However, this one-shot deployment often commits to a suboptimal proposal, even when better nearby alternatives remain consistent with the behavior data. To address this issue, we propose Gradient Guided Multi Agent Flow (G2MAF), a refinement framework for optimizing joint policies at test-time. G2MAF applies one globally normalized, projected critic gradient to guide and coordinate all agents' corrections while keeping the action both feasible and close to the frozen policy proposal. Across 24 MPE and SMAC settings, its canonical variant improves 20 frozen settings, with mean relative gains of 9.2% on MPE and 8.9% on SMAC, with model inference latency 
    
[^43]: 基于区块链的资源优化与能耗感知智能体AI框架，用于安全软件供应链

    Resource-Optimized and Energy-Aware Agentic AI Framework Anchored on Blockchain for Secure Software Supply Chains

    [https://arxiv.org/abs/2609.31282](https://arxiv.org/abs/2609.31282)

    该论文提出了一种基于区块链的智能体AI安全框架，利用LLM驱动的专门安全智能体保护软件开发生命周期的全过程，并通过在许可链上记录密码学签名证明来确保智能体自身的可信性与安全性。

    

    本文提出了一种以区块链为支撑的智能体安全框架，旨在保护完整的软件开发生命周期（SDLC），同时保护负责监控该生命周期的智能体AI组件本身。该框架协调了一组专门化的安全智能体，涵盖源代码完整性、依赖项与SBOM分析、CI配置审计、制品验证以及运行时策略评估，每个智能体均由一个大型语言模型（LLM）提供支持，该模型能够解释制品、对工具输出进行推理并生成结构化的安全报告。为确保智能体的可信性，每个智能体都会生成经密码学签名的证明，并通过智能合约记录在许可链区块链中，其中包括智能体注册表、不可变的证明日志以及可强制执行的发布策略模块。智能体之间以及智能体与区块链节点之间的通信由联盟运营的证书颁发机构进行保护，确保……（原文摘要在此处截断）

    arXiv:2609.31282v1 Announce Type: cross  Abstract: This paper proposes a blockchain-backed agentic security framework designed to safeguard the complete software development lifecycle (SDLC) while also securing the agentic AI components responsible for monitoring it. The framework coordinates a set of specialised security agents, covering source integrity, dependency and SBOM analysis, CI configura tion auditing, artifact verification, and runtime policy evaluation, each supported by a large language model (LLM) that interprets artefacts, reasons over tool outputs, and produces structured security reports. To ensure agent trustworthiness, every agent generates a cryptographically signed attestation that is recorded in a permissioned blockchain via smart contracts, including an agent registry, an immutable attestation log, and an enforceable release-policy module. Communication among agents and with blockchain nodes is secured using a consortium-operated certificate authority, ensuring 
    
[^44]: MA-WAM：用于测试时规划的多智能体世界-动作模型

    MA-WAM: Multi-Agent World-Action Model for Test-Time Planning

    [https://arxiv.org/abs/2609.31281](https://arxiv.org/abs/2609.31281)

    提出了首个面向多智能体流策略的测试时世界模型规划器 MA-WAM，通过建模跨智能体动作依赖关系来预测联合动作的后果，从而实现对候选联合动作的高效评分与规划。

    

    多智能体协作任务要求不同智能体同时执行联合动作，且每个智能体的动作都会影响其他智能体的观测与响应。因此，需要一个世界模型来预测所有智能体联合动作所带来的团队回报。一种朴素的扩展方法是在逐步预测团队回报时，直接将单智能体世界模型应用于每个智能体的动作。然而，这种扩展无法捕捉多个智能体同时动作之间的依赖关系。我们提出多智能体世界-动作模型（MA-WAM），这是一个测试时规划框架，使冻结的多智能体流策略能够评估候选联合动作的未来后果。据我们所知，MA-WAM 是首个面向多智能体流策略的测试时世界模型规划器。MA-WAM 依据跨智能体依赖关系预测每个联合动作的后果，并实现高效的候选动作评分。在……

    arXiv:2609.31281v1 Announce Type: new  Abstract: Multi-agent cooperative tasks require different agents to execute a joint action simultaneously, and each agent's action affects both the observations and responses of the other agents. Hence, a world model is needed to predict the team return resulting from the joint actions of all agents. A naive extension directly applies a single-agent world model to each agent's action when predicting the team return step by step. However, such an extension fails to capture the dependencies among the simultaneous actions of multiple agents. We propose Multi-Agent World-Action Model (MA-WAM), a test-time planning framework that enables a frozen multi-agent flow policy to evaluate futures of candidate joint actions. To our knowledge, MA-WAM is the first test-time world-model planner for multi-agent flow policies. MA-WAM predicts the consequences of each joint action according to cross-agent dependencies and enables efficient candidate scoring. Across 
    
[^45]: AI时代的认知技能：计算机专业学生与专家的认知与看法

    Cognitive Skills in the Age of AI: Computing Students and Experts Perceptions

    [https://arxiv.org/abs/2609.31272](https://arxiv.org/abs/2609.31272)

    该研究通过对计算机专业学生和专家的混合方法调查发现，在AI丰富的未来，大多数认知技能的重要性将会下降，但批判性思维能力仍将保持其关键地位。

    

    人工智能正日益融入日常工作流程之中，尤其是在计算机领域。我们正在逐步迈向一个AI丰富的未来——一个即将到来却仍充满未知的未来。一个重要的新兴问题是：我们是否正在为未来的计算机从业者做好相应准备。此外，我们需要了解哪些认知技能对于计算机从业者保持竞争力至关重要，以及认知技能的重要性是否正在发生变化。为探究这一方向，我们开展了一项混合方法研究，收集了计算机专业学生和计算机领域专家对认知技能在过去、现在和未来重要性的看法。我们发现，在AI丰富的环境中，大多数认知技能的感知重要性在未来将会有所下降，但批判性思维能力依然保持其重要性。此外，我们还通过访谈收集了关于认知技能重要性为何会发生变化以及未来将如何变化的原因。

    arXiv:2609.31272v1 Announce Type: cross  Abstract: AI is becoming increasingly integrated into daily workflows, especially in computing. We are gradually shifting towards an AI-rich future, an impending yet unknown one. One important emerging concern is whether we are accordingly preparing our future computing workforce. Further, we need to know what the important cognitive skills are to remain relevant in the computing workforce and if there are changes in cognitive skill importance. To investigate this direction, we conducted a mixed-methods study, collecting perceptions from computing students and computing experts regarding the importance of cognitive skills in the past, present, and future. We report that the perceived importance of most cognitive skills will decrease in the future, with an AI-rich environment, but critical thinking skills remain important. Further, we report reasons collected through interviews on why the importance of cognitive skills will change and how future 
    
[^46]: MoSAR：用于学习自适应且可近似注意力几何的语义注意力机制混合方法

    MoSAR: Mixture of Semantic Attention Regimes for Learning Adaptive and Approximable Attention Geometries

    [https://arxiv.org/abs/2609.31261](https://arxiv.org/abs/2609.31261)

    MoSAR将注意力近似视为几何学习问题，通过输入条件化路由器在短程、中程和全局注意力机制间学习自适应混合，以连续的距离相关注意力场取代固定稀疏模式，突破长上下文建模中稠密自注意力的二次复杂度瓶颈。

    

    稠密自注意力的二次方复杂度仍然是长上下文语言建模的核心瓶颈。许多高效的替代方法通过预先决定注意力应在哪里稀疏或局部化来应对这一开销。我们认为，注意力近似应当转而被视为一个几何问题，相关的交互几何应从数据中学习：自然语言的依赖关系依赖于输入且难以预先规定，因此模型应当学习位置相关性可以在哪里衰减，以及更广泛的交互必须在何处得以保留。我们提出了语义注意力机制混合模型MoSAR，它在查询-键交互上学习这种自适应的、受控衰减的几何结构。应用于位置编码之后的输入条件化查询与键路由器，在短程、中程和全局机制之间选择混合，从而诱导出连续的距离相关注意力场，而非固定的稀疏模式。

    arXiv:2609.31261v1 Announce Type: cross  Abstract: The quadratic complexity of dense self-attention remains a central bottleneck for long-context language modeling. Many efficient alternatives address this cost by deciding in advance where attention should be sparse or local. We argue that attention approximation should instead be approached as a geometric problem, with the relevant interaction geometry learned from data: natural-language dependencies are input-dependent and difficult to prescribe in advance, so the model should learn where positional relevance can decay and where broader interactions must be preserved. We introduce Mixture of Semantic Attention Regimes (MoSAR), which learns such an adaptive, controlled-decay geometry over query--key interactions. Input-conditioned query and key routers, applied after positional encoding, select mixtures over short, medium, and global regimes, inducing a continuous distance-dependent attention field rather than a fixed sparsity pattern
    
[^47]: 智能体限价订单簿：相变与市场冲击

    Agentic Limit Order Books: Phase Transitions and Market Impact

    [https://arxiv.org/abs/2609.31260](https://arxiv.org/abs/2609.31260)

    该论文首次构建了完全由强化学习智能体组成的限价订单簿，揭示了智能体数量与市场深度的临界阈值会触发市场从有序价格发现到高波动级联的相变，且智能体提供流动性时的市场冲击偏离经典平方根规律。

    

    我们研究了完全由自主强化学习智能体交易者构成的限价订单簿（LOB）中涌现的系统性宏观动态。通过在微观订单撮合引擎中对智能体交互进行形式化建模，我们考察了两个基本的量化现象：订单流状态切换中的均衡相变，以及市场冲击的结构性动态。我们证明，智能体限价订单簿表现出明显的相边界，将有秩序的价格发现与高波动级联状态区分开来，这些相边界由智能体数量和可观测市场深度的临界阈值所控制。此外，我们证明在智能体流动性提供的条件下，市场冲击偏离了经典的平方根动态规律，在非线性反馈回路作用下呈现出截然不同的耗散、平衡和非耗散状态。

    arXiv:2609.31260v1 Announce Type: cross  Abstract: We investigate the systemic macroscopic dynamics emerging from Limit Order Books (LOBs) populated exclusively by autonomous reinforcement-learning agentic traders. By formalizing agent interactions within a microscopic order-matching engine, we examine two fundamental quantitative phenomena: equilibrium phase transitions in order flow regime shifts, and the structural dynamics of market impact. We show that agentic LOBs exhibit distinct phase boundaries separating orderly price discovery from hyper-volatile cascade states, governed by critical thresholds in the number of agents and observable market depth. Furthermore, we demonstrate that market impact under agentic liquidity provision deviates from classical square-root dynamics, exhibiting distinct dissipative, balanced, and non-dissipative regimes under non-linear feedback loops.
    
[^48]: 多视图图像集中的几何不一致性定位

    Geometric Inconsistency Localization in Multi-View Image Sets

    [https://arxiv.org/abs/2609.31247](https://arxiv.org/abs/2609.31247)

    本文提出了带像素级几何不一致性标注的宽基线多视图数据集DeformView，并据此开发了轻量级分类器DEFECt3R，利用跨视图特征关系实现了多视图图像间几何不一致性的有效定位。

    

    新视角合成（NVS）模型能够从不同视点生成同一场景的逼真新视图。然而，这些生成的视图之间并不总是保持几何一致性。多视图（MV）一致性已被证明是评估这些NVS模型的有效工具，但其在多媒体取证领域的潜力在很大程度上尚未被探索，特别是针对宽基线图像对之间几何不一致性的定位。为了推动这一方向的研究，我们提出了DeformView，一个带有像素级几何不一致性标注的宽基线多视图数据集。利用DeformView，我们评估了当前最先进的多视图一致性评分方法，结果表明为NVS评估而开发的方法难以迁移到几何不一致性定位这一取证任务中。为了解决这一局限，我们提出了DEFECt3R，一个轻量级的基于学习的分类器，它利用跨视图特征关系来定位几何不一致性。

    arXiv:2609.31247v1 Announce Type: cross  Abstract: Novel view synthesis (NVS) models can produce realistic new views of the same scene from different viewpoints. However, these generated views are not always geometrically consistent with one another. Multi-view (MV) consistency has shown promise as a tool for evaluating these NVS models. Its potential for multimedia forensics, however, remains largely unexplored, particularly for localizing geometric inconsistencies across wide-baseline image pairs. To enable research in this direction, we introduce DeformView, a wide-baseline MV dataset with pixel-level annotations of geometric inconsistencies. Using DeformView, we evaluate state-of-the-art MV consistency-scoring methods and show that approaches developed for NVS evaluation transfer poorly to the forensic task of geometric inconsistency localization. To address this limitation, we propose DEFECt3R, a lightweight learning-based classifier that uses cross-view feature relationships to l
    
[^49]: Purin：一种受生物学启发的神经网络机制

    Purin: A Biology-inspired Mechanism for Artificial Neural Networks

    [https://arxiv.org/abs/2609.31235](https://arxiv.org/abs/2609.31235)

    Purin是一种受生物学启发的机制，通过基于时间间隔的神经活动抽象，将有界的突触效能调制引入传统卷积神经网络，无需离散时间步即可实现短期和长期的突触效能变化。

    

    人工神经网络（ANN）在训练批次期间通常使用固定的可训练权重来表示神经传递，这忽略了突触效能的短期变化。此外，离散时间步模拟需要额外的时序处理，而许多传统的ANN架构并未采用这种处理。为了克服这些挑战，我们提出了Purin，一种受生物学启发且与ANN兼容的机制，它将突触效能调制引入传统的卷积神经网络。Purin对神经活动采用基于时间间隔的抽象，使其能够在不使用离散时间步的情况下引入短期和长期的突触效能变化。Purin引入一个有界因子来表示临时的突触效能变化，并结合两个权重矩阵分别表示输入侧和输出侧的效能。这些权重矩阵通过反向传播进行更新，并被解释为长期突触……

    arXiv:2609.31235v1 Announce Type: new  Abstract: Artificial neural networks (ANNs) usually represent neural transmission with fixed trainable weights during a training batch, which omits short-term changes in synaptic efficacy. In addition, the discrete time-step simulation requires additional temporal processing that many conventional ANN architectures do not use. To overcome these challenges, we propose Purin, a biology-inspired and ANN-compatible mechanism, that introduces synaptic efficacy modulation into conventional convolutional neural networks. Purin uses a time-interval-based abstraction for neural activities, which allows Purin to introduce short- and long-term synaptic efficacy changes without using discrete time-steps. Purin introduces a bounded factor to represent temporary synaptic efficacy changes, together with two weight matrices that represent input-side and output-side efficacy. The weight matrices are updated by backpropagation and interpreted as the long-term synap
    
[^50]: 面向全双工语音模型的声学到文本KV压缩

    Acoustic-to-Text KV Compression for Full-Duplex Speech Models

    [https://arxiv.org/abs/2609.31224](https://arxiv.org/abs/2609.31224)

    提出声学到文本KV压缩方法，利用聆听时间余量通过转写侧通道将旧语音状态转换为紧凑文本记忆，大幅降低全双工语音模型长时间交互中的KV缓存内存占用。

    

    全双工语音语言模型会持续累积声学键值（KV）状态，使得长时间运行的交互非常消耗内存。在聆听过程中，模型可以在下一个音频单元到达之前完成对当前音频单元的处理；我们将这段剩余的时间间隔称为“聆听时间余量”。我们提出声学到文本KV压缩方法，该方法引入一个转写侧通道，利用这段时间间隔将传入的语音转换为紧凑的文本记忆。当推理过程中缓存超过目标预算时，较旧的声学状态会被移除，而转写文本和最近的声学上下文得以保留。我们使用LoRA在转写片段上以交叉熵损失训练该侧通道。为了保持聆听和说话行为，我们在原生预测位置对原始模型的token级输出分布应用知识蒸馏。在十分钟时长的LongSpeech会话上，我们的MiniCPM-o 4.5实现显著降低了流式KV缓存的峰值大小。

    arXiv:2609.31224v1 Announce Type: cross  Abstract: Full-duplex speech language models continuously accumulate acoustic key-value (KV) states, making long-running interactions memory-intensive. During listening, the model can finish processing an audio unit before the next arrives; we term the remaining interval listening-time slack. We propose acoustic-to-text KV compression, which introduces a transcription side channel to convert incoming speech into compact textual memory within this interval. When the cache exceeds a target budget during inference, older acoustic states are evicted while transcripts and recent acoustic context remain. We train the side channel with LoRA using cross-entropy on transcription segments. To preserve listening and speaking behavior, we apply knowledge distillation to the original model's token-level output distributions at native prediction positions. On ten-minute LongSpeech sessions, our MiniCPM-o 4.5 implementation reduces peak streaming KV-cache size
    
[^51]: DIAL：具有自适应人类偏好校准的位置去偏大语言模型评判器

    DIAL: Position-Debiased LLM Judges with Adaptive Human Preference Calibration

    [https://arxiv.org/abs/2609.31215](https://arxiv.org/abs/2609.31215)

    提出DIAL统一框架，利用大量LLM比较结合少量人类比较，分离并消除LLM评判器中的位置偏差，并将去偏后的偏好结构自适应校准至人类偏好目标，同时提供可识别性理论与不确定性量化保证。

    

    以大语言模型作为评判器可以实现可扩展的评估，但其判断可能对回答顺序敏感，并且即使去除这种位置效应后，其判断仍可能与人类偏好存在系统性偏差。我们提出了DIAL，这是一个统一框架，它将丰富的LLM比较与有限的人类比较相结合，以分离评判器特有的位置效应，学习位置去偏后LLM偏好中的共享结构，并将该结构自适应地校准至人类偏好目标。在理论上，我们研究了DIAL的三个方面：(i) 潜在LLM偏好、位置效应和人类校准的可识别性；(ii) 在LLM锚定与有限人类证据之间取得平衡的自适应估计方法；(iii) 针对校准后人类偏好的固定权重不确定性量化。在实证方面，我们在受控模拟和三个人类偏好数据集上分别评估了位置去偏和人类对齐性能（摘要原文在此处截断）。

    arXiv:2609.31215v1 Announce Type: new  Abstract: Large language models (LLMs) as a judge enable scalable evaluation, but their judgments can be sensitive to response order and, even after removing such position effects, can still diverge systematically from human preferences.We introduce DIAL, a unified framework that combines abundant LLM comparisons with limited human comparisons to separate judge-specific position effects, learn shared structure in position-debiased LLM preferences, and adaptively calibrate that structure toward the human preference target. Theoretically, we study three aspects of DIAL: (i) identification of latent LLM preferences, position effects, and human calibration; (ii) adaptive estimation that balances LLM anchoring against limited human evidence; and (iii) fixed-weight uncertainty quantification for the calibrated human preference. Empirically, we evaluate position debiasing and human alignment separately in controlled simulations and on three human-prefere
    
[^52]: 我们在估计哪种影响？反事实规范在数据归因中的作用

    Which Influence Are We Estimating? The Role of Counterfactual Specifications in Data Attribution

    [https://arxiv.org/abs/2609.31214](https://arxiv.org/abs/2609.31214)

    该论文指出数据归因中各影响估计器排序不一致的根本原因是“规范不匹配”而非近似误差，将影响形式化为反事实估计量，并按隐含规范对现有估计器进行系统分类。

    

    估计训练样本对模型行为的影响对于数据调试、数据估值和数据归因至关重要。现有的影响估计器常常产生互不相容的排序，这通常被归因于近似误差。我们认为，一个更根本的分歧来源是规范不匹配：影响取决于被归因的行为、施加于每个训练样本的干预，以及将干预映射到模型响应的反事实训练过程。当目标行为需要一个可处理的替代量（例如查询损失、logit 或间隔）时，这些选择尤为重要。我们将影响形式化为一个反事实估计量，区分了不同估计量之间的规范不匹配与估计固定估计量时产生的近似误差，并按照各估计器隐含的规范对代表性方法进行了分类整理。我们进一步推导出一个局部分解，揭示了行为……（摘要在此处截断）

    arXiv:2609.31214v1 Announce Type: new  Abstract: Estimating the influence of training examples on model behavior is essential for data debugging, valuation, and attribution. Existing influence estimators often produce incompatible rankings, which are commonly ascribed to approximation error. We argue that a more fundamental source of disagreement is specification mismatch: influence depends on the behavior being attributed, the intervention applied to each training example, and the counterfactual training process that maps the intervention to a model response. These choices are especially important when the target behavior requires a tractable surrogate, such as query loss, a logit, or a margin. We formalize influence as a counterfactual estimand, distinguish specification mismatch across estimands from approximation error in estimating a fixed estimand, and organize representative estimators by their implied specifications. We further derive a local decomposition that exposes how beha
    
[^53]: 样本、来源与空间：人脑微架构空间结构化表示学习中的数据规模分解

    Samples, Sources, Space: Decomposing Data Scale in Spatially Structured Representation Learning of Human Brain Microarchitecture

    [https://arxiv.org/abs/2609.31201](https://arxiv.org/abs/2609.31201)

    该研究将数据规模分解为独立样本数、来源多样性与空间覆盖率三个维度，通过93次对比学习预训练实验发现，人脑显微组织学表示学习性能随样本数量、空间覆盖、计算量和模型容量的增加而持续提升。

    

    扩展性研究通常仅用单一的样本数量来表征训练数据。然而，对于具有层级结构和空间结构的数据，相同数量的样本可能来自少量或大量不同的来源，并在底层领域内以不同方式分布。因此，我们将数据扩展视为一个分配问题，将独立样本数量、来源多样性和空间覆盖率三个因素分离开来。我们在显微全脑组织学数据上研究这种分解，其中一个来源对应单个大脑，一个样本对应特定空间位置上的图像块。通过对利用空间邻近性作为监督信号的对比学习模型进行93次受控预训练实验，我们在来自21个人类大脑的1160万个空间锚定图像块上改变了数据分配方式、计算量和模型容量。结果表明，性能随着独立样本数量增加、空间覆盖范围扩大、计算量增加以及模型容量增大而提升。在固定样本数量的情况下，分布式采样……（原文摘要在此处截断）

    arXiv:2609.31201v1 Announce Type: new  Abstract: Scaling studies typically represent training data by a single count of samples. For hierarchically and spatially structured data, however, the same number of samples can be drawn from few or many sources and distributed differently across the underlying domain. We therefore study data scaling as an allocation problem, separating unique sample count, source diversity, and spatial coverage. We study this decomposition in microscopic whole-brain histology, where a source is an individual brain, and a sample is an image patch at a specific spatial location. Across 93 controlled pretraining runs of a contrastive model that uses spatial proximity for supervision, we vary data allocation, compute, and model capacity over 11.6 million spatially anchored image patches from 21 human brains. Performance improves with more unique samples, broader spatial coverage, additional compute, and larger model capacity. At fixed sample count, distributing sam
    
[^54]: 重新思考AI驱动系统的数据质量：来自从业者访谈的证据

    Rethinking Data Quality for AI-Driven Systems: Evidence from Practitioner Interviews

    [https://arxiv.org/abs/2609.31191](https://arxiv.org/abs/2609.31191)

    本文通过对16名从业者的访谈首次提供了实证证据，揭示了AI驱动系统中数据质量内涵的根本转变——可追溯性转向模型行为归因、智能体上下文与记忆成为数据对象、合成数据使真实性成为关注点、合法性成为训练数据的准入门槛。

    

    数据质量研究通常将数据视为被存储、处理和验证的输入。而在AI驱动的软件密集型系统中，数据还塑造着模型行为、评估方式和合法使用。关于从业者在此类条件下如何定义、评估和管理质量，目前的实证证据仍然有限。我们访谈了来自九个组织的16名从业者，采用反身性主题分析法对访谈记录进行分析，并从参与者的叙述中提炼出六个主题。在AI系统中，可追溯性从模块化调试转变为对模型行为的归因，而将模型用作质量评估者则引入了循环性问题。智能体的上下文和记忆成为数据对象，合成数据和伪标签数据使真实性成为新的质量关注点。在基础模型开发中，合法性成为训练数据的准入门槛，而代表性则通过系统必须安全运行的情境覆盖范围来评判。

    arXiv:2609.31191v1 Announce Type: cross  Abstract: Data quality research has usually treated data as an input that is stored, processed, and validated. In AI-driven software-intensive systems, data also shapes model behavior, evaluation, and lawful use. Empirical evidence remains limited on how practitioners define, assess, and manage quality under these conditions. We interviewed 16 practitioners from nine organizations and analyzed the transcripts using reflexive thematic analysis and developed six themes from participants' accounts. In AI systems, traceability shifted from modular debugging to attributing model behavior, while using models as quality assessors introduced circularity. Agent context and memory became data objects, and synthetic and pseudo-labeled data made authenticity a quality concern. In foundation-model development, lawfulness became a gate for training data, while representativeness was judged through coverage of situations in which the system must behave safely.
    
[^55]: 递归自我改进人工智能的进化安全性：分类、风险发现与评估

    Evolutionary Safety of Recursive Self-Improving AI: Taxonomy, Risk Discovery, and Evaluation

    [https://arxiv.org/abs/2609.31186](https://arxiv.org/abs/2609.31186)

    该论文提出“进化安全”这一新视角，用于研究AI在递归自我改进过程中安全属性如何随演化而变化、持续、积累与传播，并给出了相应的风险分类、风险发现与评估框架。

    

    人工智能正在快速发展，能力日益强大的系统在推理、决策、科学发现和自主开发中扮演着越来越重要的角色。随着AI开始参与自身的改进——从模型训练和经验积累到智能体演化和自动化AI开发——递归自我改进（RSI）的前景正变得越来越相关。这一转变提出了一个根本性的安全问题：当系统本身、其积累的经验、甚至产生其后继者的过程都在持续变化时，如何保持安全性？我们引入“进化安全”作为研究持续递归自我改进下安全性的新视角。它关注的不仅是AI系统在某一特定时刻是否安全，而是安全属性如何在整个演化过程中变化、持续、积累和传播。我们刻画了反复出现的典型表现形式，包括……

    arXiv:2609.31186v1 Announce Type: new  Abstract: Artificial intelligence is advancing rapidly, with increasingly capable systems taking larger roles in reasoning, decision-making, scientific discovery, and autonomous development. As AI begins to participate in its own improvement, from model training and experience accumulation to agent evolution and automated AI development, the prospect of recursive self-improvement (RSI) is becoming increasingly relevant. This transition raises a fundamental safety question: how can safety be maintained when the system, its accumulated experience, and even the process producing its successors continue to change?   We introduce Evolutionary Safety as a perspective for studying safety under persistent and recursive self-improvement. It concerns not only whether an AI system is safe at a particular moment, but how safety properties change, persist, accumulate, and propagate throughout evolution. We characterize recurring manifestations, including inten
    
[^56]: 考虑偏差实现可持续的大语言模型评估

    Accounting for Bias Enables Sustainable LLM Evaluation

    [https://arxiv.org/abs/2609.31184](https://arxiv.org/abs/2609.31184)

    该论文提出一个统一的潜在变量框架，通过显式校正位置偏差、冗长偏差、评委严格度等系统性测量偏差，能用远少于以往的比较次数恢复可靠排名，从而为LLM评估提供了统计上更严谨、计算上更可持续的方案。

    

    以大语言模型作为评委（LLM-as-a-judge）已成为可扩展主观评估的事实标准，然而当前的排行榜通过进行越来越多的比较来补偿系统性测量偏差，这种方法在统计上不健全且在计算上浪费。其根本原因在于测量模型不完整：将LLM评委视为中性的、可互换的测量工具，忽视了已有文献记录的多种偏差，如位置偏差、冗长偏差、评委严格度以及自我增强偏好，而这些偏差无法通过增加数据量来消除。我们提出了一个统一的潜在变量框架，在联合建模成对比较和序数数据的同时显式校正这些混杂因素，从而从大幅减少的比较次数中恢复出可靠的排名。由于拟合该模型的计算成本相对于单轮LLM推理而言可以忽略不计，偏差校正不仅在统计上更为严谨，也是实现可信评估的一种更具可持续性的方法。

    arXiv:2609.31184v1 Announce Type: new  Abstract: LLM-as-a-judge has become the de facto standard for scalable, subjective evaluation, yet current leaderboards compensate for systematic measurement bias by running ever more comparisons, an approach that is both statistically unsound and computationally wasteful. The root cause is an incomplete measurement model, treating LLM judges as neutral, interchangeable instruments ignores documented biases like position bias, verbosity bias, judge severity, and self-enhancement, that no volume of additional data can eliminate. We propose a unified latent variable framework that jointly models pairwise and ordinal data while explicitly correcting for these confounders, recovering reliable rankings from substantially fewer comparisons. Because fitting this model costs negligible compute relative to a single round of LLM inference, bias correction is not only more statistically rigorous but also a more sustainable approach to trustworthy evaluation.
    
[^57]: BAT-CLIP：大脑、音频与文本的三模态对齐

    BAT-CLIP: Trimodal Alignment of Brain, Audio and Text

    [https://arxiv.org/abs/2609.31180](https://arxiv.org/abs/2609.31180)

    BAT-CLIP提出了首个面向iEEG的CLIP式三模态对齐框架，将神经嵌入同时对齐到预训练的音频和文本锚点，克服了单模态锚定带来的权衡问题，在自然语音解码中实现了比双模态基线更鲁棒的表示。

    

    从大脑中解码和解释自然语音越来越依赖于与预训练的语音和语言表示空间的对齐。然而，当前的CLIP式脑-语音对齐方法将神经活动锚定到单一的锚定模态——音频或文本——尽管大脑的语音处理本质上是多模态的。这导致了一种权衡：音频锚定保留了时间结构但削弱了语言可分性，而文本锚定捕获了语义却丢弃了声学细节。我们提出BAT-CLIP，这是首个面向iEEG（颅内脑电图）的CLIP式三模态对齐框架，它在一个共享的、冻结的音频-文本流形中将神经嵌入同时对齐到预训练的音频和文本锚点。在自然播客基准测试中，BAT-CLIP比双模态CLIP基线获得了更鲁棒的表示。我们还强调了使用自监督基础模型进行CLIP训练的重要性。

    arXiv:2609.31180v1 Announce Type: cross  Abstract: Decoding and interpreting naturalistic speech from the brain increasingly relies on alignment to pretrained speech and language representation spaces. However, current CLIP-style brain-speech alignment ground neural activity to a single anchor modality-audio or text-despite the brain's inherently multimodal speech processing. This induces a trade-off: audio anchoring preserves temporal structure but weakens linguistic separability, while text anchoring captures semantics yet discards acoustic detail. We propose BAT-CLIP, the first CLIP-style trimodal alignment framework for iEEG that jointly aligns neural embeddings to both pretrained audio and text anchors in a shared, frozen audio-text manifold. On the naturalistic Podcast benchmark, BAT-CLIP yields more robust representations than bimodal CLIP baselines. We also highlight the importance of using self-supervised foundation models for CLIP training.
    
[^58]: SPO：通过Stackelberg程序优化发现自适应大邻域搜索算子

    SPO: Discovering Adaptive Large Neighborhood Search Operators via Stackelberg Program Optimization

    [https://arxiv.org/abs/2609.31179](https://arxiv.org/abs/2609.31179)

    提出了基于LLM的Stackelberg程序优化框架SPO，将破坏算子作为领导者、修复算子作为条件性跟随者进行耦合进化搜索，自动发现能够适应LNS状态的自适应破坏-修复算子程序，在TSP和CVRP问题上超越了现有方法。

    

    大邻域搜索（LNS）严重依赖于破坏算子和修复算子，其有效性既取决于对不断演变的LNS状态的适应能力，也取决于这两个角色之间的交互。我们提出了Stackelberg程序优化（SPO），这是一个基于大语言模型（LLM）的框架，用于发现自适应的可执行破坏-修复程序。SPO将算子决策建立在紧凑的LNS状态之上，使依赖状态的行为能够通过程序发现自然涌现，并将破坏-修复算子的发现过程组织为程序空间上的Stackelberg交互，以反映二者之间的非对称依赖关系。角色特定的信用评估机制将破坏程序视为领导者、将修复程序视为条件性的跟随者响应，从而引导一个将LLM生成器学习与基于种群的程序进化搜索相结合的耦合优化过程。在旅行商问题（TSP）和带容量约束的车辆路径问题（CVRP）上的实验表明，SPO的性能优于现有方法。

    arXiv:2609.31179v1 Announce Type: new  Abstract: Large neighborhood search (LNS) relies critically on destroy and repair operators, whose effectiveness depends on both adaptation to the evolving LNS state and interaction between the two roles. We introduce Stackelberg Program Optimization (SPO), an LLM-based framework for discovering adaptive executable destroy-repair programs. SPO conditions operator decisions on a compact LNS state, allowing state-dependent behavior to emerge through program discovery, and organizes destroy-repair discovery as a Stackelberg interaction over program space that reflects their asymmetric dependency. Role-specific credits evaluate destroy programs as leaders and repair programs as conditional follower responses, guiding a coupled optimization process that combines LLM generator learning with population-based evolutionary search over programs. Experiments on the traveling salesperson problem and capacitated vehicle routing problem show that SPO outperform
    
[^59]: 代码仓库中问题定位的语义导航

    Semantic Navigation for Issue Localization in Code Repository

    [https://arxiv.org/abs/2609.31176](https://arxiv.org/abs/2609.31176)

    SemNav框架通过确定性检索生成初始候选集合，并借助LLM智能体基于证据进行迭代精炼，同时利用语义导航图按需解析程序关系，从而有效提升代码仓库级问题定位的效果。

    

    仓库级问题定位旨在识别并排序与解决所报告问题相关的文件和函数。LLM智能体以迭代方式处理该任务：它们先识别出一组潜在相关的位置，检查相应的代码，并随着新证据的获取不断修正对这些候选位置的判断。然而，现有环境对这一循环的支持有限：智能体必须自行搜索未解析的关系目标，从原始源代码中重建实体语义，并在缺乏证据依据的情况下修正候选。为解决这些局限，我们提出了SemNav，该框架利用确定性检索来生成广泛的候选集合，并由LLM智能体持续对其进行精炼，从而将初始覆盖与证据引导的修正相结合。SemNav通过三个关键组件支持这一过程，其中语义导航图通过语言服务器按需解析程序关系。

    arXiv:2609.31176v1 Announce Type: new  Abstract: Repository-level issue localization aims to identify and rank the files and functions relevant to resolving a reported issue. LLM agents approach this task iteratively: they identify a set of potentially relevant locations, inspect the corresponding code, and revise their judgments about these candidates as new evidence is acquired. Existing environments, however, provide limited support for this loop: agents must search for unresolved relation targets, reconstruct entity semantics from raw source code, and revise candidates without evidential basis. To address these limitations, we present SemNav, a framework that leverages deterministic retrieval to seed a broad candidate set and an LLM agent to continually refine that set, thereby combining initial coverage with evidence-guided revision. SemNav supports this process through three key components. A Semantic Navigation Graph resolves program relations on demand through a language server
    
[^60]: 基于度量损失加权提升大语言模型在多模态机器翻译中的视觉敏感性

    Improving Visual Sensitivity of LLMs on Multimodal Machine Translation with Metric-based Loss Weighting

    [https://arxiv.org/abs/2609.31169](https://arxiv.org/abs/2609.31169)

    提出基于度量的损失加权训练方法，利用PCXMI度量识别受益于图像的词元并提高其损失权重，从而增强多模态大语言模型在多模态机器翻译任务中对视觉信息的敏感性和利用能力。

    

    多模态机器翻译旨在利用来自非文本模态的额外信号，通过消解歧义来改进翻译。尽管模型通过多模态融合能够接收与源文本相关的图像，但它们可能会忽略这些信息。因此，提高模型的视觉敏感性仍然是一个活跃的研究方向。在这项工作中，我们提出了一种训练方法——基于度量的损失加权，通过提高那些受益于伴随图像的词元的损失权重，来增强翻译的视觉定位能力。我们使用点向交叉互信息度量来识别这些词元，该度量通过比较有视觉上下文和无视觉上下文情况下模型的输出概率来实现。我们进一步提出了一种基于一致性的PCXMI度量，并通过实验证明这两种度量结合使用能够取得最佳效果。我们通过微调三个预训练的多模态大语言模型来评估所提出的方法。

    arXiv:2609.31169v1 Announce Type: cross  Abstract: Multimodal Machine Translation aims to incorporate additional signal from non-textual modalities to improve translations by resolving ambiguities. While models, through multimodal fusion, are able to accept images related to the source text, they can ignore this information. Therefore, increasing their visual sensitivity remains an active research area. In this work, we introduce a training method, Metric-based Loss Weighting, that improves visual grounding of translations by increasing the loss function for tokens that benefit from the accompanying image. We identify these tokens using the Point-wise Cross-mutual Information (PCXMI) metric, which compares the model's output probabilities with and without visual context. We introduce a Congruency-based PCXMI metric and experimentally show that both metrics working in combination yield the best results. We evaluate our method by fine-tuning three pretrained Multimodal Large Language Mod
    
[^61]: 神经状态预测：阻碍EEG基础模型中的捷径学习

    Neural State Prediction: Obstructing Shortcut Learning in EEG Foundation Models

    [https://arxiv.org/abs/2609.31167](https://arxiv.org/abs/2609.31167)

    提出神经状态预测（NSP）框架，通过EMA目标编码器、身份残差化和拓扑分离上下文三重机制阻碍EEG基础模型中的捷径学习，迫使模型整合分布式神经上下文，从而学到更可迁移的神经表征。

    

    EEG基础模型越来越多地使用掩码预测任务从无标签的脑电记录中学习，但优化这一目标并不能保证学到可迁移的神经表征。一个核心挑战在于，稳定的位置线索和局部相关性可以使被掩码区域在无需整合分布式神经上下文的情况下就能被预测出来。为了减少对这种低信息量预测路径的依赖，我们提出了神经状态预测（NSP），这是一个同时约束预测目标和可用上下文的潜在预测框架。NSP使用由指数移动平均（EMA）更新的目标编码器来定义潜在监督信号。身份残差化从预测目标中去除与通道身份和相对时间相关的加性效应，而拓扑分离的上下文则从可见输入中排除这些目标的直接空间和时间邻域。我们在来自TUEG的220万个EEG片段上对NSP进行预训练，并对其进行评估。

    arXiv:2609.31167v1 Announce Type: new  Abstract: EEG foundation models increasingly use masked prediction to learn from unlabeled recordings, but optimizing this objective does not ensure transferable neural representations. A central challenge is that stable positional cues and local correlations can make masked regions predictable without integrating distributed neural context. To reduce this reliance on low-information prediction paths, we introduce Neural State Prediction (NSP), a latent-predictive framework that constrains both the prediction target and the available context. NSP uses a Target Encoder updated by an exponential moving average (EMA) to define latent supervision. Identity residualization removes additive effects associated with channel identity and relative time from the targets, while topology-separated context excludes their immediate spatial and temporal neighborhood from the visible input. We pretrain NSP on 2.2 million EEG segments from TUEG and evaluate it acro
    
[^62]: AgentRecommender：基于LLM智能体的用户端可定制推荐系统

    AgentRecommender: LLM Agents Enable Customizable Recommender Systems on the User Side

    [https://arxiv.org/abs/2609.31166](https://arxiv.org/abs/2609.31166)

    提出AgentRecommender方法，利用LLM智能体的调查能力和内部知识，在无需额外数据的情况下灵活构建用户端推荐系统，使用户能够轻松创建符合自身偏好的定制化推荐系统。

    

    推荐系统传统上是为平台开发的。然而，这催生了许多可能有利于平台锁定用户但对用户造成困扰的现象，例如点击诱饵（标题党）、过滤气泡和虚假新闻的传播。最近，用户端推荐系统作为一种解决这一问题的新范式被提出。如果用户部署自己的推荐系统，他们就不再受制于平台的利益。然而，构建用户端推荐系统并非易事；特别是，为自己定制一个推荐系统需要额外的数据。我们提出了AgentRecommender，这是一种利用LLM智能体的调查能力和内部知识来灵活构建用户端推荐系统的方法，无需额外数据。AgentRecommender使用户能够轻松创建符合自身偏好的推荐系统。

    arXiv:2609.31166v1 Announce Type: cross  Abstract: Recommender systems have traditionally been developed for platforms. However, this has given rise to many phenomena that may be advantageous for platform lock-in but are a nuisance to users, such as clickbait, filter bubbles, and the spread of fake news. Recently, user-side recommender systems have been proposed as a new paradigm for solving this problem. If users deploy their own recommender systems, they are no longer at the mercy of the platform's interests. However, building a user-side recommender system is not trivial; in particular, customizing one for oneself requires additional data. We propose AgentRecommender, a method that leverages the investigation capability and internal knowledge of LLM agents to flexibly build user-side recommender systems without additional data. AgentRecommender allows users to easily create recommender systems tailored to their own preferences.
    
[^63]: SPADE：突破流行度-相似度边界以度量意外性推荐

    SPADE: Escaping the Popularity-Similarity Frontier to Measure Serendipitous Recommendations

    [https://arxiv.org/abs/2609.31164](https://arxiv.org/abs/2609.31164)

    提出SPADE评估指标，将物品映射到二维流行度-相似度空间并计算用户特定的帕累托前沿距离，从而同时考量流行度、相似度与用户实际相关性，有效度量意外性推荐并防止算法投机。

    

    推荐系统通过设计意外性（serendipity）来促进用户的主动探索，并打破可预测的消费循环。现有离线“准确性之外”的评估指标存在的问题是，它们往往只孤立地考察历史相似性或全局流行度。我们的目标是设计一个能够同时考察相似性、流行度和用户实际相关性的评估指标。为此，我们提出了SPADE（意外性帕累托距离评估，Serendipitous Pareto Distance Evaluation）。SPADE将所有物品映射到二维空间中，直接为每个用户计算由流行度最高且历史最相似的物品构成的帕累托前沿。最终的意外性得分通过严格针对测试集中被正确推荐的物品，计算它们到该边界的最小欧氏距离并取平均值而得到。在五个数据集和五种基线算法上的评估证实了SPADE的有效性；我们的结果表明，该指标成功防止了算法对“准确性之外”评估指标的投机利用。

    arXiv:2609.31164v1 Announce Type: cross  Abstract: Recommender systems engineer serendipity to foster active exploration and break predictable consumption cycles. The problem with existing offline beyond-accuracy metrics is that they often either isolate historical similarity or global popularity. We aim to design an evaluation metric that examines similarity, popularity, and actual user relevance. To achieve this, we introduce SPADE (Serendipitous Pareto Distance Evaluation). SPADE maps all items into a two-dimensional space to directly calculate a user-specific Pareto frontier of maximally popular and historically similar items. The final serendipity score is then computed by averaging the minimum Euclidean distance from this boundary strictly for the correctly recommended test-set items. Evaluating SPADE across five datasets and five baseline algorithms confirms its effectiveness; our results show that the metric successfully prevents algorithms from exploiting beyond-accuracy measu
    
[^64]: ReG-SAM：参考图驱动的SAM用于2D血管分割基础模型

    ReG-SAM: Reference Graph-Driven SAM for 2D Foundational Vessel Segmentation

    [https://arxiv.org/abs/2609.31160](https://arxiv.org/abs/2609.31160)

    提出ReG-SAM，一种基于SAM并利用参考图集（图提示嵌入与血管原型等模态感知表征）来增强血管表征的2D血管分割基础模型框架。

    

    医学图像中的血管分割对于从诊断到治疗规划的许多临床任务至关重要。然而，由于复杂的血管形态和多样的成像条件，血管分割仍然具有挑战性。现有的深度学习方法很少致力于构建可跨解剖结构和成像模态泛化的血管分割器。虽然Segment Anything Model（SAM）在医学图像分割中展现出潜力，但其原始设计并未充分利用血管形态，且难以处理细粒度的血管结构，导致性能欠佳。在本文中，我们提出了ReG-SAM，一个专为2D血管分割设计的基于SAM的框架，它利用参考图集来增强血管表征。具体而言，我们引入了两种从参考掩码中导出的模态感知表征：图提示嵌入，用于从图中编码全局空间特征；以及血管原型……（摘要原文在此处被截断）

    arXiv:2609.31160v1 Announce Type: cross  Abstract: Vessel segmentation in medical images is essential for many clinical tasks, ranging from diagnosis to treatment planning. However, it remains challenging due to complex vascular morphology and diverse imaging conditions. Existing deep learning methods rarely aim at building a generalizable vessel segmentor across anatomies and modalities. While the Seg- ment Anything Model (SAM) has shown promise for med- ical image segmentation, its original design does not fully exploit vascular morphology and struggles with fine-grained vascular structures, leading to suboptimal performance. In this paper, we propose ReG-SAM, a SAM-based framework tailored to 2D vessel segmentation that leverages reference graph set for enhancing vascular representations. Specifically, we introduce two modality-aware representations derived from the reference masks: graph prompt embeddings (GPEs) that encode global spatial features from graphs, and vascu- lar protot
    
[^65]: 面向个性化时序边缘智能的动量引导联邦分割蒸馏

    Momentum-Guided Federated Split Distillation for Personalized Temporal Edge Intelligence

    [https://arxiv.org/abs/2609.31159](https://arxiv.org/abs/2609.31159)

    该论文提出动量引导的联邦分割蒸馏框架，通过TeRR-SAtt时序储备池学生注意力设计和AMGF动量引导融合机制，在大幅降低边缘设备训练与推理延迟及资源占用的同时，显著提升个性化时序学习的准确率。

    

    我们提出了一种动量引导的联邦分割蒸馏框架，用于实现个性化、高效且自主的时序边缘智能。我们引入了TeRR-SAtt，这是一种新颖的时序储备池学生注意力设计，它结合了固定的储备池表示、轻量级的时序学生模型和个性化的输出模块。我们还提出了AMGF，这是一种前瞻性的动量引导融合机制，通过学习动量对客户端进行聚类，并推导出专门的教师更新。在真实世界的智能建筑数据上，与所考虑的基线方法相比，TeRR-SAtt将边缘训练延迟降低了65.50%，推理延迟降低了44.70%，训练内存占用降低了18.40%，推理CPU使用率降低了33.10%。同时，与全局更新相比，AMGF将本地学习的RMSE提升了高达35.31%。

    arXiv:2609.31159v1 Announce Type: new  Abstract: We propose a momentum-guided federated split distillation framework for personalized, efficient, and autonomous temporal edge intelligence. We introduce TeRR-SAtt, our novel temporal reservoir student attention design that combines fixed reservoir representations, a lightweight temporal student, and personalized output modules. We also present AMGF, our anticipatory momentum-guided fusion mechanism that clusters clients through learning momentum and derives specialized teacher updates. On real-world smart-building data, TeRR-SAtt reduces edge training latency by 65.50%, inference latency by 44.70%, training memory usage by 18.40%, and inference CPU usage by 33.10% over the considered baselines. At the same time, AMGF improves local learning by up to 35.31% in RMSE compared to global updates.
    
[^66]: 域偏移下基于教师锚定的训练后量化模型选择

    Teacher-Anchored Selection of Post-Training Quantized Models under Domain Shift

    [https://arxiv.org/abs/2609.31155](https://arxiv.org/abs/2609.31155)

    该论文提出在域偏移且目标域标签缺失或稀缺的场景下，以教师模型失真作为锚点来指导训练后量化候选模型的部署选择，揭示了置信度类估计器的失效与输出分布类估计器的可靠，并将有监督项与教师锚点相结合以改进选择。

    

    压缩一个已训练好的模型会产生一系列部署候选模型，而在域偏移条件下，压缩程度最高的候选模型并不一定是应当部署的那一个。我们研究了在这类候选模型族上进行选择的问题，其中候选模型与教师模型固定不变，且目标域标签缺失或稀缺。两个发现组织了无标签情形下的研究。最小教师失真几乎表现为一种恒定规则，在每次运行中都选出同一个八比特、逐通道、无裁剪的配置，而该配置并不能最小化目标域的经验交叉熵。现有估计器则呈现明显分化：在CNN模型族的过度自信坍塌情形中，基于置信度的估计器对模型族的排序几乎完全颠倒，而识别这种情形的诊断方法又需要该设定下无法获得的标签；相比之下，基于输出分布的估计器与教师相对锚点相匹配，并在某一架构上超越了它。尽管如此，失真本身是稳定的，因此一个有监督的项可以将选择从锚点上移开。将两者结合，我们……（摘要在此处截断）

    arXiv:2609.31155v1 Announce Type: cross  Abstract: Compressing a trained model yields a family of deployment candidates, and under domain shift the most compressed one need not be the one to deploy. We study selection over such a family, with candidates and teacher fixed and target labels absent or scarce. Two findings organize the label-free case. Minimum teacher distortion behaves almost as a constant rule, selecting the same eight-bit, per-channel, unclipped configuration in every run, which does not minimize empirical target cross-entropy. Established estimators divide sharply: in the overconfident-collapse regime of the CNN families, confidence-based estimators order the family close to backwards, and the diagnostics that identify it need the labels the setting denies, while output-distribution estimators match the teacher-relative anchor and on one architecture beat it. Distortion is nonetheless stable, so a supervised term can move selection away from it. Combining the two, we g
    
[^67]: FedHisto-PAST：面向跨站点肺组织病理学分类的参数高效染色感知联邦学习

    FedHisto-PAST: Parameter-Efficient Stain-Aware Federated Learning for Cross-Site Lung Histopathology Classification

    [https://arxiv.org/abs/2609.31150](https://arxiv.org/abs/2609.31150)

    FedHisto-PAST v2通过将冻结的HIBOU-B基础模型与参数高效适配、染色感知学习及自适应联邦聚合相结合，以低通信和参数成本实现了跨站点肺组织病理学的鲁棒分类。

    

    跨站点肺组织病理学分类必须考虑染色变异、非独立同分布的客户端数据、类别缺失以及适配大型病理编码器的成本问题。本研究评估了FedHisto-PAST v2在腺癌（ACA）、正常组织与鳞状细胞癌（SCC）三类分类任务中的表现。FedHisto-PAST v2将冻结的HIBOU-B基础模型与参数高效适配、染色条件下的成对视图预测与特征一致性、可靠性感知的原型学习以及自适应联邦聚合相结合。实验采用了五客户端、非独立同分布、原始数据本地化的模拟设置，包括固定内部评估、客户端级别分析、组件消融实验、通信量统计，以及受开发过程影响的探索性LungHist700队列。所有主要方法在内部评估中均达到了接近上限的性能，这限制了其在固定数据划分上的区分能力。在LungHist700数据集上，FedHisto-PAST v2取得了0.728的Macro-F1分数。

    arXiv:2609.31150v1 Announce Type: cross  Abstract: Cross-site lung histopathology classification must account for stain variation, non-IID client data, missing classes, and the cost of adapting large pathology encoders. This study evaluates FedHisto-PAST v2 for three-way classification of adenocarcinoma (ACA), Normal, and squamous cell carcinoma (SCC). FedHisto-PAST v2 combines a frozen HIBOU-B foundation model with parameter-efficient adaptation, stain-conditioned paired-view prediction and feature consistency, reliability-aware prototype learning, and adaptive federated aggregation. Experiments used a five-client, non-IID, raw-data-local simulation with fixed internal evaluation, client-level analysis, component ablations, communication accounting, and a development-influenced exploratory LungHist700 cohort. All principal methods achieved near- ceiling internal performance, which limited discrimination on the fixed split. On LungHist700, FedHisto- PAST v2 achieved a Macro-F1 of 0.728
    
[^68]: JevAdvBench：面向校准决策强化学习模型的基准与黑盒攻击

    JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models

    [https://arxiv.org/abs/2609.31142](https://arxiv.org/abs/2609.31142)

    该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。

    

    使用校准决策强化学习（RLCD）训练的模型（如Jev），针对输入（即状态）回答一个类型化问题，输出概率、选择或分数，软件在无人阅读的情况下直接对该答案进行处理。这类模型的鲁棒性尚未得到测量：现有的对抗基准评估的是模型生成或执行的内容，而类型化模型不生成任何内容，即使被操纵也会返回格式良好的答案。测量本身也很困难，因为相同的请求可能返回不同的答案，大多数可用标签来自模型自身，且API会在用户不可见的情况下对每个请求进行预处理。我们的核心思想是将每个受攻击的决策与模型自身的干净决策进行对比评分，而非与标签对比，并结合相同请求重复运行所引起的变化来解读评分结果。基于此，我们提出了JevAdvBench，据我们所知这是首个针对RLCD模型的对抗基准，包含覆盖6个……的812个类型化问题（摘要在此处被截断）。

    arXiv:2609.31142v1 Announce Type: cross  Abstract: Models trained with reinforcement learning for calibrated decisions (RLCD), such as Jev, answer a typed question about an input, the state, with a probability, a choice, or a score, and software acts on the answer without a person reading it. Their robustness has not been measured: adversarial benchmarks score what a model generates or executes, whereas a typed model generates nothing and returns a well-formed answer even when manipulated. Measurement is also hard, because identical requests can return different answers, most available labels come from the model itself, and the API preprocesses each request out of view. Our key idea is to score each attacked decision against the model's own clean decision rather than against labels, and to read it against the change caused by an identical re-run. Building on this, we introduce JevAdvBench, to our knowledge the first adversarial benchmark for RLCD models, with 812 typed questions over 6
    
[^69]: 语言推理向量能否增强多模态推理能力？

    Can Linguistic Reasoning Vectors Enhance Multimodal Reasoning Ability?

    [https://arxiv.org/abs/2609.31140](https://arxiv.org/abs/2609.31140)

    该论文提出LIFT方法，通过提取“推理者路径”与“求解者路径”之间的隐状态差异作为推理向量并注入模型，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。

    

    大多数视觉语言模型都是通过在预训练大语言模型基础上扩展视觉模块并进行多模态对齐而构建的。然而，这种多模态扩展往往会降低基座大语言模型中原有的语言侧推理能力。尽管基座大语言模型在扩展后仍保留了可用的推理能力，但经过对齐的视觉语言模型本身却无法可靠地获取这一能力。因此，恢复视觉语言模型中退化的推理能力，向基座大语言模型寻求帮助比仅依靠视觉语言模型本身更为有效。基于这一动机，我们提出了LIFT（Language-side reasonIng Facilitation and Transfer，语言侧推理促进与迁移），这是一种轻量级的向量干预方法，无需重新训练骨干网络即可将基座大语言模型的推理能力迁移到视觉语言模型中。LIFT将推理向量定义为具有显式推理轨迹的推理者路径与没有推理轨迹的求解者路径之间的答案词元隐状态差异，并将这些向量注入语言……

    arXiv:2609.31140v1 Announce Type: new  Abstract: Most Vision-Language Models (VLMs) are built by extending pretrained Large Language Models (LLMs) with visual modules and multimodal alignment. However, this multimodal scaling often degrades the language-side reasoning ability originally encoded in the base LLM. While the base LLM retains usable reasoning after scaling, the aligned VLM itself cannot reliably access this ability. Therefore, recovering the degraded reasoning capability in VLMs would benefit more from seeking help from the base LLM than from the VLM alone. Motivated by this, we propose LIFT (Language-side reasonIng Facilitation and Transfer), a lightweight vector-intervention method that transfers reasoning capability from the base LLM to the VLM without retraining the backbone. LIFT defines Reasoning Vectors as answer-token hidden-state differences between a Reasoner path with an explicit reasoning trace and a Solver path without it, and injects these vectors into languag
    
[^70]: 面向AI增强的协作工程工作流：欧洲漫游车挑战赛的需求与架构

    Toward AI-Augmented Cooperative Engineering Workflows: Requirements and Architecture the European Rover Challenge

    [https://arxiv.org/abs/2609.31136](https://arxiv.org/abs/2609.31136)

    本文针对学生团队在复杂工程系统集成中缺乏AI流程级支持的问题，基于对欧洲漫游车挑战赛14支团队104份问卷的调研，提出了AI增强协作工程工作流的需求与架构。

    

    人工智能（AI）工具的日益普及为支持工程设计流程创造了新的机会，然而目前对AI的使用往往局限于编码、文档撰写或信息检索等孤立任务。对于AI如何在流程层面支持协作工程工作流——即团队必须协调需求、任务、沟通、知识传递和子系统集成的场景——所受关注较少。本文以欧洲漫游车挑战赛为背景研究这一挑战，在该赛事中，学生团队需要在严格的时间约束和高度子系统相互依赖的条件下，在单个学年内完成复杂漫游车系统的设计与集成。我们面向ERC 2025参赛团队开展了一项角色自适应的40题问卷调查，共收到来自14支团队的104份回复。该调查考察了团队结构、知识传递、任务管理、集成实践、沟通模式等方面。

    arXiv:2609.31136v1 Announce Type: new  Abstract: The growing availability of Artificial Intelligence (AI) tools creates new opportunities to support engineering design processes, yet their current use often remains limited to isolated tasks such as coding, documentation, or information retrieval. Less attention has been given to how AI can support cooperative engineering workflows at the process level, where teams must coordinate requirements, tasks, communication, knowledge transfer, and subsystem integration. This paper investigates this challenge in the context of the European Rover Challenge (ERC), where student teams design and integrate complex rover systems within a single academic cycle under strict time constraints and high subsystem interdependence. We conducted a role adaptive 40 question survey with ERC 2025 teams, yielding 104 responses from 14 teams. The survey examined team structure, knowledge transfer, task management, integration practices, communication patterns, and
    
[^71]: Pocket-STVG：用于时空视频定位（Spatio-Temporal Video Grounding）的轻量级架构

    Pocket-STVG: lightweight architecture for Spatio-Temporal Video Grounding

    [https://arxiv.org/abs/2609.31135](https://arxiv.org/abs/2609.31135)

    P-STVG 是一种轻量级级联架构，通过组合 MobileViCLIP、MDETR 等高效预训练组件而非大型端到端模型来实现时空视频定位，并借助轻量 1D U-Net 或阈值策略使同一框架同时支持弱监督与零样本设置。

    

    arXiv:2609.31135v1 公告类型：交叉列表（cross）。摘要：时空视频定位（STVG）旨在根据自然语言查询在视频中定位相应的时空管道（spatio-temporal tube）。尽管近期的相关方法在完全监督、弱监督和零样本设置下均取得了出色性能，但它们通常依赖于计算开销高昂的架构、复杂的训练流程或多模态大语言模型。我们提出了 Pocket-STVG（P-STVG），这是一种轻量级级联架构，通过组合高效的预训练组件而非大型端到端模型来解决 STVG 问题。P-STVG 集成了基于 MobileViCLIP 的时序感知视频编码器、源自 MDETR 的空间编码器-解码器，以及一个共享的对齐文本编码器。时序定位通过轻量级的 1D U-Net 或简单的阈值策略来实现，使同一框架能够在弱监督和零样本两种设置下运行。此外，视频表示会被预先计算……（原文摘要在此处被截断）

    arXiv:2609.31135v1 Announce Type: cross  Abstract: Spatio-Temporal Video Grounding (STVG) aims to localize the spatio-temporal tube in a video corresponding to a natural language query. While recent methods achieve strong performance in fully supervised, weakly supervised, and zero-shot settings, they typically rely on computationally expensive architectures, complex training pipelines, or multimodal large language models. We present Pocket-STVG (P-STVG), a lightweight cascade architecture that addresses STVG by combining efficient pre-trained components instead of large end-to-end models. P-STVG integrates a temporal-aware video encoder based on MobileViCLIP, a spatial encoder-decoder derived from MDETR, and a shared aligned text encoder. Temporal localization is performed through either a lightweight 1D U-Net or a simple thresholding strategy, enabling the same framework to operate in both weakly supervised and zero-shot settings. Furthermore, video representations are precomputed in
    
[^72]: AtomWorld-Mem：面向长时程原子演化的记忆恢复世界状态

    AtomWorld-Mem: Memory-Restored World States for Long-Horizon Atomistic Evolution

    [https://arxiv.org/abs/2609.31133](https://arxiv.org/abs/2609.31133)

    提出了记忆恢复型原子世界模型 AtomWorld-Mem，通过空间编码器生成多尺度原子关键帧，并结合短期事件记忆与长期结构记忆来恢复瞬时晶体快照中缺失的潜在世界状态，从而解决长时程原子演化中的快照歧义问题。

    

    在长时间尺度上进行高保真的原子演化，仅观察当前的晶体构型是不够的。瞬时的原子快照往往是不完整的：局部相似的构型可能对应着不同的隐含动力学背景、未来事件偏好以及等待时间尺度。我们认为，这种快照歧义性使得长时程原子演化在根本上成为一个基于记忆的世界状态恢复问题。为了解决这一问题，我们提出了 AtomWorld-Mem，一种通过记忆恢复潜在世界状态的原子世界模型，用以补全瞬时晶体快照中所缺失的信息。AtomWorld-Mem 将演化的合金视为一个 AtomWorld：空间编码器从密集的局部拓扑和稀疏的长程缺陷上下文中写入多尺度的原子关键帧，而短期事件记忆和长期结构记忆则将这些关键帧在时间维度上进行整合，以恢复具有未来预测能力的演化状态。

    arXiv:2609.31133v1 Announce Type: new  Abstract: High-fidelity atomistic evolution over long timescales requires more than observing the current crystal configuration. Instantaneous atomistic snapshots are often incomplete: locally similar configurations can correspond to different hidden dynamical contexts, future event preferences, and waiting-time scales. We argue that this snapshot ambiguity makes long-horizon atomistic evolution fundamentally a memory-based world-state restoration problem. To address this, we introduce AtomWorld-Mem, a memory-restored atomistic world model that recovers the latent world state missing from instantaneous crystal snapshots. AtomWorld-Mem treats the evolving alloy as an AtomWorld: spatial encoders write multi-scale atomistic keyframes from dense local topology and sparse long-range defect context, while short-term event memory and long-term structural memory integrate these keyframes across time to restore a future-predictive evolutionary state. The r
    
[^73]: 监控越狱：在不编码推理的情况下规避思维链监控

    Monitor Jailbreaking: Evading Chain-of-Thought Monitoring Without Encoded Reasoning

    [https://arxiv.org/abs/2609.31121](https://arxiv.org/abs/2609.31121)

    本研究发现在思维链监控的优化压力下，推理模型无需编码或隐藏其推理，仅通过调整思维链的措辞和格式即可规避监控器的检测，同时推理内容对人类依然完全透明，这一新现象被称为“监控越狱”。

    

    思维链监控是一种针对推理模型的有前景的安全技术，能够在模型采取行动之前检测到有问题的推理。一个关键的担忧是“编码推理”，即模型以监控者和人类都无法解读的方式隐藏其真实推理。强化学习过程中来自思维链监控器的优化压力被认为是驱动此类行为的可能因素。我们通过训练推理模型同时执行一个主任务和一个侧任务来研究这一问题，并在监控器检测到模型对侧任务的推理时对其进行惩罚。令人惊讶的是，模型学会了在不编码其推理的情况下规避监控器。相反，它们学会了通过调整措辞和格式来组织其思维链，使监控器无法标记出侧任务相关的推理，而这些推理对人类读者来说仍然完全透明。我们将这种现象称为“监控越狱”。我们发现，监控越狱在不同模型规模、不同监控器下均会出现。

    arXiv:2609.31121v1 Announce Type: new  Abstract: Chain-of-thought (CoT) monitoring is a promising safety technique for reasoning models, enabling detection of problematic reasoning before models act. A key concern is encoded reasoning, where models hide their true reasoning in ways that monitors and humans cannot interpret. Optimization pressure from CoT monitors during reinforcement learning is considered a likely driver of such behavior. We investigate this by training reasoning models to perform a main task and a side task, while penalizing them when a monitor detects reasoning about the side task. Surprisingly, models learn to evade monitors without encoding their reasoning. Instead, they learn to phrase and format their chains of thought such that monitors fail to flag side task reasoning, while the reasoning remains completely transparent to human readers. We call this phenomenon monitor jailbreaking. We find that monitor jailbreaking arises across different model sizes, monitors
    
[^74]: 从捷径学习到离散神经插入排序

    From Shortcut Learning to Discrete Neural Insertion Sort

    [https://arxiv.org/abs/2609.31114](https://arxiv.org/abs/2609.31114)

    该论文揭示了神经算法推理模型在学习插入排序时存在捷径学习问题——中间表示在算法执行结束前就能解码出排好序的结果——并提出了一种将序列表示为链、分离标量交换与控制状态转移、且每步后将节点表示投影回离散状态的离散神经插入排序模型，以促使模型真正遵循算法执行过程。

    

    神经算法推理旨在训练神经网络遵循已知算法，并泛化到训练中未见过的输入规模。然而，正确的最终输出与中间监督并不一定表明模型遵循了预期的执行过程。我们以插入排序为研究对象来探讨这一问题。我们对 CLRS30 基线 NAR 的分析表明，提示目标仅被弱优化，提示准确率始终较低。此外，在参考插入排序执行终止之前，许多中间表示已经可以被解码为已排序的序列，这表明模型学到的是通往最终输出的捷径。受这些发现启发，我们提出了离散神经插入排序。我们的模型将序列表示为链，将标量交换与控制状态转移分离，并在每个处理器步骤后将节点表示投影回离散状态。当仅在序列（上训练时……）（原文摘要在此处截断）

    arXiv:2609.31114v1 Announce Type: cross  Abstract: Neural algorithmic reasoning aims to train neural networks to follow known algorithms and generalize beyond the input sizes seen during training. However, correct final outputs and intermediate supervision do not necessarily show that a model follows the intended execution. We study this problem using insertion sort. Our analysis of the CLRS30 baseline NAR shows that the hint objective is weakly optimized and that hint accuracy remains low. Moreover, many intermediate representations can already be decoded into sorted sequences before the reference insertion-sort execution terminates, suggesting that the model learns a shortcut to the final output. Motivated by these findings, we introduce Discrete Neural Insertion Sort. Our model represents the sequence as a chain, separates scalar exchanges from control-state transitions, and projects node representations back to discrete states after every processor step. When trained only on sequen
    
[^75]: 基于Fisher信息几何的贝叶斯优化：梯度界与信赖域方法

    Bayesian Optimization with Fisher Information Geometry: Gradient Bounds and Trust-Region Methods

    [https://arxiv.org/abs/2609.31107](https://arxiv.org/abs/2609.31107)

    本文通过拉回Fisher信息度量导出采集函数的梯度上界，解释了高维贝叶斯优化中的梯度消失现象，并提出基于信赖域的FITR方法，以局部Fisher权重取代长度尺度缩放来提升优化性能。

    

    我们从信息几何的视角研究贝叶斯优化（BO）。通过代理后验映射拉回Fisher信息度量，可以在输入空间上得到一个局部敏感性张量，从而为可重参数化采集函数的梯度提供一个上界。这一视角解释了高维贝叶斯优化中的梯度消失现象，并为RAASP和维度缩放长度尺度等启发式方法提供了统一的解释。基于这一分析，我们提出了FITR，一种基于信赖域的贝叶斯优化方法，它用局部拉回Fisher权重取代了基于长度尺度的缩放。FITR不局限于具有显式长度尺度的高斯过程核。在使用SE核的高斯过程基准测试中，实验表明FITR具有有竞争力的性能。所提出的方法也很容易推广到非各向同性的代理模型，尽管在这种情形下其收益更加依赖于具体任务。

    arXiv:2609.31107v1 Announce Type: cross  Abstract: We study Bayesian optimization (BO) through the lens of information geometry. Pulling back the Fisher information metric through the surrogate posterior map yields a local sensitivity tensor on the input space, which leads to an upper bound on the gradient of reparameterizable acquisition functions. This view explains vanishing-gradient behavior in high-dimensional BO and provides a common interpretation of heuristics such as RAASP and dimension-scaled lengthscales. Building on this analysis, we propose FITR, a trust-region-based BO method that replaces lengthscale-based scaling by local pullback-Fisher weights. FITR is not restricted to GP kernels with explicit lengthscales. On GP benchmarks with an SE kernel, experiments show competitive performance using FITR. The proposed method also easily generalizes to non-isotropic surrogates, although the gains are more task-dependent in that setting.
    
[^76]: DepthEvidence：在多模态语言模型中统一度量深度预测与几何推理

    DepthEvidence: Unifying Metric Depth Prediction and Geometric Reasoning in Multimodal Language Models

    [https://arxiv.org/abs/2609.31103](https://arxiv.org/abs/2609.31103)

    DepthEvidence是一个40亿参数的多模态语言模型，它将自身预测的密集度量深度作为对象级证据融入语言生成过程，实现了度量深度预测与几何推理的统一，并通过Depth-VQA基准验证了对象测量和空间-数值组合推理能力。

    

    具有度量约束的空间推理需要将对象与几何测量值关联起来，并在语言推理过程中保留其数值内容。我们提出了DepthEvidence，这是一个40亿参数的模型，它将自身密集的度量深度预测作为语言生成中对象级证据。相机条件解码器利用多尺度视觉特征和高分辨率RGB细化来预测全分辨率的度量深度。一个从密集表示到语言的接口将预测的深度和解码器特征转换为与对象对齐的连续几何标记，并锚定于对象标识符。几何监督促使度量信息在语言上下文交互前后保持可恢复性，而指令微调则支持对象测量和组合推理。我们引入了Depth-VQA基准，用于评估对象-深度查询、相对比较以及结合空间与数值约束的决策。在九个（数据集上）……

    arXiv:2609.31103v1 Announce Type: cross  Abstract: Spatial reasoning with metric constraints requires linking objects to geometric measurements and preserving their numerical content during language reasoning. We present DepthEvidence, a 4B model that uses its own dense metric predictions as object-grounded evidence for language generation. A camera-conditioned decoder predicts full-resolution metric depth using multi-scale visual features and high-resolution RGB refinement. A dense-to-language interface converts predicted depths and decoder features into object-aligned continuous geometry tokens anchored to object identifiers. Geometric supervision encourages metric information to remain recoverable before and after language-context interaction, while instruction tuning supports object measurement and compositional reasoning. We introduce a Depth-VQA benchmark evaluating object-depth queries, relative comparisons, and decisions combining spatial and numerical constraints. Across nine 
    
[^77]: OmouAI：基于模拟角色的论辩式人机政策商议

    OmouAI: Argumentative Human-AI Policy Deliberation with Simulated Personas

    [https://arxiv.org/abs/2609.31078](https://arxiv.org/abs/2609.31078)

    OmouAI将大语言模型与计算论辩相结合，通过引入模拟角色（如利益相关者、领域专家、唱反调者）与人类用户共同商议现实政策主张，以减少LLM的谄媚性并提供人类监督。

    

    由大语言模型（LLM）驱动的智能体之间的辩论已在多种应用中展现出巨大潜力，但当这些交互包含人类参与并发生在高风险环境（例如公共政策商议）中时，便会受到谄媚性（sycophancy）和缺乏可信解释等问题的困扰。为解决这些问题，我们提出了OmouAI——一个交互式且具有包容性的商议系统，它将LLM与计算论辩（computational argumentation）相结合，后者是一个擅长于辩论中的表示与推理的领域。OmouAI允许人类用户与模拟角色（例如代表利益相关者、领域专家或唱反调者/魔鬼代言人）就现实世界挑战的政策主张进行商议，以减少谄媚性。每个角色都会生成自己的论点，所有各方的论点共同构成一个共享的论辩框架。用户随后可以质疑、添加和修改论点，从而提供至关重要的人类监督。

    arXiv:2609.31078v1 Announce Type: new  Abstract: Debates amongst agents driven by large language models (LLMs) have demonstrated vast potential in various applications, but when these interactions include humans and take place in high-stakes environments, e.g., in public policy deliberations, they are beset with issues such as sycophancy and a lack of faithful explanations. To tackle these issues, we present OmouAI, an interactive and inclusive deliberation system that uses LLMs in combination with computational argumentation, a field which excels in representing and reasoning within debates. OmouAI allows a human user to deliberate policy claims for real-world challenges with simulated personas, e.g., representing stakeholders, domain experts or devil's advocates, towards reducing sycophancy. Each persona generates its own arguments, and the arguments of all parties form a shared argumentation framework. Users can then contest, add and revise arguments, providing crucial human oversig
    
[^78]: 上下抽象阶梯：面向语言智能体的基于代码的技能

    Up and Down the Abstraction Ladder: Code-Based Skills for Language Agents

    [https://arxiv.org/abs/2609.31076](https://arxiv.org/abs/2609.31076)

    本文提出 CodeHack 技能库，在长时程游戏环境 NetHack 中系统研究了基于代码的动作抽象如何影响语言智能体的性能、推理成本和学习，揭示了抽象带来的生产力与灵活性之间的权衡。

    

    语言智能体在需要长序列低级动作的环境中行动和学习时常常遇到困难。基于代码的抽象可以让这些智能体更加高效，使它们能够调用可复用的技能，而不必反复选择单个动作。代码负责处理重复性的局部决策，而语言模型则决定使用哪些技能以及如何组合它们。然而，抽象是“有漏洞的”，当遇到超出技能能力范围的情况时，可能需要回退到原始动作。出于对生产力与灵活性之间这种权衡的思考，我们系统地研究了基于代码的动作抽象如何影响语言智能体的性能、推理成本和学习。我们在 NetHack 这一具有挑战性的长时程游戏环境中开展研究，使用 CodeHack——我们构建的带有自然语言描述的基于代码的技能库。我们利用该技能库，将仅限于原始动作的智能体与使用语义技能的智能体进行比较……

    arXiv:2609.31076v1 Announce Type: new  Abstract: Language agents struggle to act and learn in environments that require long sequences of low-level actions. Code-based abstractions can make these agents more productive by letting them invoke reusable skills instead of repeatedly selecting individual actions. The code handles recurring local decisions, while the language model decides which skills to use and how to combine them. Yet abstractions are leaky, and situations beyond a skill's capabilities may require a return to primitive actions. Motivated by this tradeoff between productivity and flexibility, we systematically study how code-based action abstraction affects the performance, inference cost, and learning of language agents. We study this in NetHack, a challenging, long-horizon game environment, using CodeHack, our library of code-based skills with natural-language descriptions. We use this library to compare agents restricted to primitives with those using semantic skills al
    
[^79]: 外化的CPDAG摘要提升大语言模型的因果推理能力

    Externalized CPDAG Summaries Improve LLM Causal Deduction

    [https://arxiv.org/abs/2609.31071](https://arxiv.org/abs/2609.31071)

    提出Structured Thinking两轮流程，先让模型外化生成受模式约束的CPDAG结构化摘要、再基于该图状态作答，将Qwen3.5-27B在Corr2Cause因果推理任务上的F1从73.0显著提升至86.4。

    

    Corr2Cause任务探究一个因果声明是否在所有与观测到的相关性及条件独立性相容的DAG（有向无环图）中都成立。我们将该任务框架化为潜在对象推理问题：标签由CPDAG查询定义，但自由格式的思维链往往会将马尔可夫等价类问题坍缩为局部模式匹配。我们提出Structured Thinking（结构化思维），这是一个两轮流程：首先外化生成一个带类型、受模式约束的CPDAG摘要，然后基于该图状态进行作答。在Corr2Cause完整测试集上，Structured Thinking将Qwen3.5-27B的F1(Yes)从73.0提升至86.4（相对强大的PC指令基线，主配对实验中提升13.4个百分点；McNemar检验p=2.4×10⁻⁶；自助法95%置信区间[+8.4, +18.6]）；在三个完整ID随机种子上，平均增益为+8.1±5.3个百分点。一个采用PC脚手架的两轮散文式对照方法仅达到67.6的F1，这表明仅有详细的PC脚手架加上不受模式约束的散文式中间表示是不够的。

    arXiv:2609.31071v1 Announce Type: new  Abstract: Corr2Cause asks whether a causal claim holds in every DAG compatible with observed correlations and conditional independencies. We frame this as latent-object reasoning: the label is defined by a CPDAG query, but free-form chain-of-thought often collapses the Markov-equivalence-class problem into local pattern matching. We propose Structured Thinking, a two-turn pipeline that first externalizes a typed, schema-constrained CPDAG summary and then answers against that graph state. On the Corr2Cause full test, Structured Thinking raises Qwen3.5-27B from $73.0$ to $86.4$ $F_1$(Yes) over a strong PC-instruction baseline in the primary paired run ($+13.4$ pp; McNemar $p=2.4\times 10^{-6}$; bootstrap $95\%$ CI [$+8.4$, $+18.6$]); across three full-ID seeds, the mean gain is $+8.1 \pm 5.3$ pp. A PC-scaffolded two-turn prose control reaches only $67.6$ $F_1$, indicating that a detailed PC scaffold plus a schema-free prose intermediate is not suffi
    
[^80]: 用于医学图像分析的量子扩散模型

    Quantum Diffusion Models for Medical Image Analysis

    [https://arxiv.org/abs/2609.31070](https://arxiv.org/abs/2609.31070)

    本文提出一种基于离散时间量子游走算法、结合经典逆向去噪模型的可扩展混合量子扩散模型，突破了现有量子设备规模的限制，能够处理真实世界的大尺寸医学图像数据。

    

    量子机器学习是一个新兴的研究领域，旨在利用量子力学的原理（如叠加、纠缠和干涉）来设计机器学习方法。在此背景下，我们提出了一种可扩展的混合量子扩散模型，并评估其在医学图像分析中的应用。具体而言，我们的方法基于在真实量子设备上执行的离散时间量子游走算法，用于对扩散模型的前向动力学进行建模。对于扩散模型的反向步骤，我们设计并评估了一个经典学习模型，用于对数据进行逆向去噪。与现有其他将量子机器学习应用于图像分析任务的尝试（这些尝试严重受限于现有量子设备的规模）不同，我们的方法能够处理真实世界的大尺寸医学数据。特别是，我们展示了在灰度图像、RGB图像以及中等规模三维体数据上的实验结果。

    arXiv:2609.31070v1 Announce Type: cross  Abstract: Quantum Machine Learning is a novel field of research aimed at devising machine learning approaches exploiting principles of quantum mechanics, such as superposition, entanglement and interference. In this context, we present a scalable hybrid Quantum Diffusion Model, and evaluate its use for medical image analysis. Specifically, our method is based on a Discrete-Time Quantum Walk algorithm, executed on a real quantum device, to model the forward dynamics of the diffusion model. For the backward step of the diffusion model, we devise and evaluate a classical learning model, which is used to reversely denoise the data. In contrast with other existing attempts at applying quantum machine learning for image analysis tasks, severely limited by the size of existing quantum devices, our method allows to process real-world large size medical data. In particular, we present results on grayscale and RGB images, as well as 3D volumes of moderate
    
[^81]: 神经化痕迹：基于对比稀疏自编码器的选择性表示级机器遗忘

    Neuralyzing the Trace: Selective Representation-Level Unlearning with Contrastive Sparse Autoencoders

    [https://arxiv.org/abs/2609.31056](https://arxiv.org/abs/2609.31056)

    该论文提出SCALPEL——一种对比稀疏自编码器，通过克服基于重构提取中的能量偏差、在表示层面学习目标选择性特征，实现了精准的机器遗忘，并从理论上证明其能有效控制对背景知识的扰动。

    

    机器遗忘旨在移除目标信息的同时保留模型的其他能力。在现实场景中，例如欧盟GDPR下的隐私请求，遗忘目标可能非常狭窄，例如与某一个人相关的信息。仅靠行为层面的遗忘可能并不足够，这促使研究者直接对模型内部表示进行干预。然而，标准的机制可解释性提取器对此类目标的选择性较差。我们发现了基于重构的提取方法中存在的一种能量偏差，它倾向于提取主导的背景结构而忽略低能量的目标特定成分。我们提出了SCALPEL，一种旨在学习更具选择性遗忘特征的对比稀疏自编码器。我们从理论上证明，对比训练能够促进目标选择性特征的学习，且我们的选择分数可以控制预期的背景知识扰动。我们在TOFU数据集上基于Qwen、Llama和Gemma模型对SCALPEL进行了实验验证。

    arXiv:2609.31056v1 Announce Type: new  Abstract: Machine unlearning aims to remove targeted information while preserving a model's other abilities. In realistic settings, such as privacy requests under the EU GDPR, the target may be narrow, for example information associated with a single person. Behavioral forgetting alone may be insufficient, motivating interventions directly on internal representations. However, standard mechanistic-interpretability extractors are poorly selective for such targets. We identify an energy bias in reconstruction-based extraction, which favors dominant background structure over low-energy target-specific components. We introduce SCALPEL, a contrastive sparse autoencoder designed to learn more selective forget features. We show theoretically that contrastive training promotes target-selective features and that our selection score controls expected background knowledge perturbation. We validate SCALPEL experimentally on TOFU across Qwen, Llama, and Gemma,
    
[^82]: 廉价的开源智能体使大语言模型污染更难缓解

    Cheap, open agents make LLM pollution harder to mitigate

    [https://arxiv.org/abs/2609.31054](https://arxiv.org/abs/2609.31054)

    廉价易得的开源智能体在调查任务中表现可与商业智能体媲美且更难被检测，使大语言模型污染的防范变得更加困难。

    

    大语言模型（LLM）污染是指合成回复污染了本意用于捕捉人类行为的数据。迄今为止，高昂的部署成本限制了自主调查智能体所带来的风险。然而，开放权重模型与开源智能体框架的结合可能已经消除了这一障碍。我们比较了九种智能体配置的性能与可检测性，涵盖从完全开源的变体到封闭的商业版本。每个智能体自主完成了一份包含多种响应类型的调查问卷，从而产生多种检测手段。完全开源的智能体可在本地运行且无需使用费用，其表现与商业替代方案相当。开源智能体和商业智能体未能通过不同的检测集合，没有任何单一检测手段能够可靠地识别所有智能体，但开放式文本回复在区分智能体与人类方面效果最佳。这些发现表明完全开源的智能体是LLM污染的一个独特风险，并支持采用多层检测策略。

    arXiv:2609.31054v1 Announce Type: new  Abstract: Large Language Model (LLM) pollution occurs when synthetic responses contaminate data intended to capture human behavior. High deployment costs have so far limited the risk posed by autonomous survey agents. However, open-weight models paired with open-source agentic frameworks may have removed this barrier. We compared the performance and detectability of nine agent configurations, ranging from fully open variants to closed commercial ones. Each agent autonomously completed a survey containing multiple response types yielding various detection checks. Fully open agents ran locally without usage fees and performed competitively with commercial alternatives. Open and commercial agents failed different sets of checks, and no single check reliably detected all agents, but open-text responses discriminated best between agents and humans. These findings identify fully open agents as a distinct risk for LLM pollution and support multilayered d
    
[^83]: DynBranch：面向动态智能体大语言模型服务的投机性子图复用

    DynBranch: Speculative Subgraph Reuse for Dynamic Agentic LLM Serving

    [https://arxiv.org/abs/2609.31047](https://arxiv.org/abs/2609.31047)

    DynBranch 通过让未解析的分支在解析前即可被寻址，实现了投机性子图执行与跨请求的子图结果复用，从而打破“分支解析屏障”，将智能体 LLM 服务的平均延迟降低最高 32%。

    

    智能体式大语言模型（LLM）工作流在运行时决定其执行路径。下游计算可能可预测，或者可能已经运行过，但在模型或用户解析出分支之前无法开始。我们将这种串行化称为“分支解析屏障”。单纯的缓存无法掩盖这一障碍：标识可复用结果的键在分支解析之前是未知的。本文提出 DynBranch，使未解析的分支在解析之前即可被寻址。其稳定的坐标允许候选子图在分支解析期间运行，并允许已完成的子图结果在后续请求中被复用。一个两级控制器会在预期收益超过负载代价时接纳这些工作。DynBranch 位于模型 API 边界，无需对智能体框架或模型执行引擎进行任何更改。在使用 Qwen3-32B 和 4 块 H200 GPU 的四个智能体工作负载上，DynBranch 相比每个工作负载最强的先前系统将平均延迟降低至多 32%，并

    arXiv:2609.31047v1 Announce Type: cross  Abstract: Agentic LLM workflows decide their execution paths at runtime. Downstream computation may be predictable, or may have run before, yet it cannot begin until the model or the user resolves the branch. We call this serialization the branch-resolution barrier. Caching alone does not hide it: the key that identifies a reusable result is not known until then. In this paper, we propose DynBranch, which makes an unresolved branch addressable before it resolves. Its stable coordinate lets candidate subgraphs run during resolution and completed subgraph results be reused across later requests. A two-level controller admits this work when its expected benefit exceeds the load price. DynBranch sits at the model-API boundary and requires no changes to agent harnesses or model execution engines. Across four agentic workloads with Qwen3-32B on 4x H200 GPUs, DynBranch reduces mean latency by up to 32% over each workload's strongest prior system and by
    
[^84]: 受治理的演绎：超越相关性的基于策略的前提授权

    Governed Deduction: Policy-Grounded Premise Authorization Beyond Relevance

    [https://arxiv.org/abs/2609.31029](https://arxiv.org/abs/2609.31029)

    该论文提出“受治理演绎”形式化框架，指出前提在推理中不仅需要相关性、还必须通过策略授权准入检查，并通过 RBAC 增强的授权配对基准实验揭示：线性控制器实际依赖角色名称捷径，在去除捷径的角色置换后性能降至 50% 的随机水平。

    

    推理系统通常将前提的使用视为一个相关性问题：如果某个事实可用且有用，它就可以被选择用于推理。授权则施加了一种不同的约束：一个前提可以被表示且在逻辑上可用，但不被允许用于特定的局部转换。我们将这一区别形式化为“受治理演绎”，引入转换局部的准入谓词 admit(p, tau, S)。基于一个独立生成的 RBAC 增强的 Spider 基准，我们构建了 4,461 个匹配的授权对，其中相同的前提和策略状态分别支持被允许和被拒绝的消费转换。一个初始的联合控制器达到了 99.19% 的留出集准确率，但仅使用转换的控制达到了 100%，暴露出一个角色名称捷径。在采用冻结的、与标签无关的上下文局部角色置换消除该捷径后，仅前提/状态、仅转换以及联合线性控制器在 1,856 个留出样本上（摘要原文在此处截断）均恰好只得到 50% 的得分。

    arXiv:2609.31029v1 Announce Type: new  Abstract: Reasoning systems usually treat premise use as a question of relevance: if a fact is available and useful, it may be selected for inference. Authorization imposes a different constraint: a premise may be represented and logically usable but not permitted for a particular local transition. We formalize this distinction as Governed Deduction (GD), with a transition-local admission predicate admit(p, tau, S). From an independently produced RBAC-augmented Spider benchmark, we construct 4,461 matched authorization pairs in which the same query premise and policy state support permitted and denied consuming transitions. An initial joint controller reaches 99.19% held-out accuracy, but a transition-only control reaches 100%, exposing a role-name shortcut. After a frozen, label-independent context-local role permutation removes that shortcut, premise/state-only, transition-only, and joint linear controllers all score exactly 50% on 1,856 held-ou
    
[^85]: 相同文本，不同数字：基于大语言模型的度量指标的分歧

    Same Text, Different Numbers: The Divergence of LLM-Based Measures

    [https://arxiv.org/abs/2609.31013](https://arxiv.org/abs/2609.31013)

    该论文让七个不同的大语言模型对相同的标普500公司财报电话会议文本进行十三种构念评分，发现基于LLM的文本度量存在显著的模型间分歧（平均秩相关仅0.52，共同差异仅占34%），且模型选择会实质性改变下游实证研究中的系数大小、符号和显著性。

    

    研究者越来越多地使用生成式大语言模型（LLM）将企业文本转化为实证变量。我们利用十三种度量指标（包括情感、管理层清晰度、不确定性、回答具体性以及气候与政治风险），考察了基于LLM的文本度量在多大程度上对模型选择保持不变。来自不同提供商的七个LLM对标准普尔500公司财报电话会议记录在这些构念上进行评分。结果显示，跨模型的秩相关系数平均仅为0.52，且各提供商之间共同的文本层面差异仅占总分变异的34%。跨模型分歧并不能预测后续分析师或市场的分歧，这表明存在实质性的模型特异性成分，而非底层信息披露本身存在共同的模糊性。模型选择会显著影响下游推断，系数大小、符号和统计显著性在不同模型之间存在较大差异。

    arXiv:2609.31013v1 Announce Type: new  Abstract: Researchers increasingly use generative large language models (LLMs) to convert corporate text into empirical variables. We examine the extent to which LLM-based textual measures are invariant to model choice using thirteen measures, including sentiment, management clarity, uncertainty, answer specificity, and climate and political risk. Seven LLMs from different providers score earnings call transcripts of S&P 500 companies on these constructs. Cross-model rank correlations average only 0.52, and transcript-level differences common across providers account for only 34% of total score variation. Cross-model disagreement does not predict subsequent analyst or market disagreement, consistent with a substantial model-specific component rather than common ambiguity in the underlying disclosure. Model choice significantly affects downstream inference, with coefficient magnitudes, signs, and statistical significance varying substantially acros
    
[^86]: G²PTQ：利用广义梯度补偿改进大语言模型训练后量化

    G$^2$PTQ: Improving LLM Post-Training Quantization with Generalized Gradient Compensation

    [https://arxiv.org/abs/2609.31009](https://arxiv.org/abs/2609.31009)

    G²PTQ提出了一种统一的大语言模型训练后量化框架，通过在每个Transformer块量化前动态刷新并融合一阶梯度与二阶Hessian信息的广义梯度补偿机制，克服了现有GPTQ类方法缺乏全局监督或指导信息随量化进程失效的缺陷。

    

    训练后量化（PTQ）是一种无需重新训练即可减少大语言模型（LLM）内存占用和计算开销的实用方法。基于GPTQ的方法已成为事实上的标准，但它们存在两个互补的局限性：采用局部、逐层目标的方法缺乏全局监督；而采用全局目标的方法在开始时便固定其Hessian估计并忽略一阶梯度，因此随着量化过程的进行，其指导作用会逐渐失效。本文提出了G²PTQ，这是一个融合广义梯度补偿的统一训练后量化框架，在全局监督的分块优化目标下同时整合了一阶和二阶信息。通过在量化每个Transformer块之前刷新梯度和Hessian估计，G²PTQ避免了先前全局方法中指导信息过时的问题。此外，为了稳定精确的一阶补偿，我们引入了一种信任域方案……（原文摘要在此处截断）

    arXiv:2609.31009v1 Announce Type: cross  Abstract: Post-training quantization (PTQ) is a practical approach to reducing the memory and computational footprint of large language models (LLMs) without retraining. GPTQ-based methods have become the de facto standard, yet they suffer from two complementary limitations. Methods with local, layer-wise objectives lack global supervision; while methods with global objectives fix their Hessian estimates at the start and ignore first-order gradients, so their guidance grows stale as quantization proceeds. This paper presents G$^2$PTQ, a unified PTQ framework with Generalized Gradient Compensation that integrates both first- and second-order information under a globally supervised, block-wise optimization objective. By refreshing gradient and Hessian estimates before quantizing each Transformer block, G$^2$PTQ avoids the staleness of prior global methods. Furthermore, to stabilize the exact first-order compensation, we introduce a trust-region sc
    
[^87]: 仅凭像素能否揭示图像来源？被动溯源的极小极大极限与可学习接口

    Can Pixels Alone Reveal Image Origin? Minimax Limits and Learnable Interfaces for Passive Provenance

    [https://arxiv.org/abs/2609.30997](https://arxiv.org/abs/2609.30997)

    该论文为仅凭像素的图像溯源建立了精确理论极限——由目标分布与受攻击源分布间的最小全变差距离决定且与验证器架构无关，并揭示了公开验证器因可被模拟而会在远早于该统计极限之前被代理黑盒攻击攻破。

    

    被动图像溯源探讨的是仅凭像素能否揭示图像的来源：是来自人类、某类AI模型的整体，还是某个特定的生成器。一旦源图像在验证者看到之前可能被编辑，这就变成了一个鲁棒性问题。我们将该问题形式化为对抗性分布偏移下的源-目标验证问题。我们的第一个结果给出了任何仅基于图像的验证器的精确最优极限：最大的鲁棒目标接受差距等于目标分布与受攻击源分布集合之间的最小全变差距离。该量仅取决于源分布、目标分布和编辑类别，而与验证器的架构无关。我们的第二个结果解释了为什么已部署的公开验证器会在达到这一统计极限之前就失效：如果验证器可以在攻击区域上以误差ε被模拟，那么代理黑盒攻击就能在2ε加上优化误差的范围内达到目标接受。

    arXiv:2609.30997v1 Announce Type: cross  Abstract: Passive image provenance asks whether pixels alone can reveal where an image came from: a human, an aggregate AI class, or a particular generator. This becomes a robustness problem once a source image can be edited before the verifier sees it. We study the problem as source--target verification under adversarial distribution shift. Our first result gives the exact best-case limit for any image-only verifier: the largest robust target-acceptance gap equals the minimum total-variation distance between the target distribution and the set of attacked source distributions. This quantity depends on the source, target, and edit class, not on the verifier architecture. Our second result explains why deployed public verifiers can fail before this statistical limit is reached. If the verifier can be emulated on the attack region to error $\varepsilon$, then a surrogate black-box attack reaches target acceptance within $2\varepsilon$ plus optimiz
    
[^88]: 视觉-语言-动作模型的线性表示假设

    The Linear Representation Hypothesis for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.30996](https://arxiv.org/abs/2609.30996)

    该论文提出了一种基于签名的理论框架，将线性表示假设从大语言模型扩展到视觉-语言-动作模型，统一了表示与策略，以应对具身交互中感兴趣的物理量与系统动力学共同演化这一挑战。

    

    线性表示假设（LRH）已成为通过大语言模型（LLM）内部表示来度量和干预语义信息的标准视角。越来越多的工作开始将这一视角扩展到视觉-语言-动作（VLA）模型，但具身交互的动态特性带来了额外的挑战。与LLM中通常研究的语义属性（如性别或语言）不同，VLA中感兴趣的物理量（QoI）会与系统动力学共同演化：表示会影响策略所选的动作，动作改变物理状态，进而又影响下一步的表示。在本文中，我们为VLA开发了一种基于签名的LRH理论表述，将表示与策略统一起来。在表示方面，我们证明了存在这样的表示，由其可以预测感兴趣的物理量在候选（策略）下的未来演化（摘要此处截断）。

    arXiv:2609.30996v1 Announce Type: cross  Abstract: The linear representation hypothesis (LRH) has become a standard lens for measuring and intervening on semantic information through the internal representations of large language models (LLMs). A growing body of work has begun extending this perspective to vision-language-action (VLA) models, but the dynamical nature of embodied interaction introduces an additional challenge. Unlike semantic attributes commonly studied in LLMs, such as gender or language, a physical quantity of interest (QoI) in a VLA evolves jointly with the system dynamics: the representation influences the actions selected by the policy, which alter the physical state and, in turn, the next representation.   In this paper, we develop a theoretical, signature-based formulation of the LRH for VLA that unifies representations and policies. On the representation side, we establish the existence of representations from which the future evolution of a QoI under a candidat
    
[^89]: FLIP：面向视觉语言模型的最终层推理时探测

    FLIP: Final Layer Inference-Time Probing for Vision-Language Models

    [https://arxiv.org/abs/2609.30993](https://arxiv.org/abs/2609.30993)

    提出FLIP——一种在推理时对VLM最终层隐藏状态施加逐元素下限干预的探测方法，用以验证logits前的干预位点支持结构化的任务相关计算，并揭示出干预强度下性能变化的三个区域以及形式化的四准则探测-扫描协议。

    

    我们提出FLIP，一种最终层推理时探测方法，用于检验开放权重视觉语言模型（VLM）中面向logits的干预位点是否支持结构化的、与任务相关的计算，而非一般性的扰动。内部干预引起的行为变化在机制上通常是模糊的：它可能反映视觉证据利用能力的改善、一般性的输出不稳定，或彻底的性能退化。FLIP在logits计算之前，对最终归一化的隐藏状态施加逐元素下限操作，同时保持参数、提示词和解码过程不变。在受控的检测/计数探测任务上，扫描干预强度揭示了三个区域：变化可忽略区域、一个有界的内部区域（其中IoU 0.50下的检测召回率 $R_{50}$ 提升且容忍性计数误差 $\mathcal{E}_{\mathrm{count}}$ 下降），以及过度抑制区域。我们形式化了一个用于规范解释的四准则探测-扫描协议。

    arXiv:2609.30993v1 Announce Type: cross  Abstract: We present FLIP, a final-layer inference-time probe for testing whether a logit-facing intervention site in an open-weight vision-language model (VLM) supports structured, task-linked computation rather than generic perturbation. Behavioral change under internal intervention is otherwise mechanistically ambiguous: it may reflect improved use of visual evidence, generic output instability, or outright degradation. FLIP applies elementwise flooring to the final normalized hidden state before logit computation, leaving parameters, prompts, and decoding unchanged. On a controlled detection/counting probe, sweeping intervention strength reveals three regions: negligible change, a bounded interior regime in which detection recall at IoU 0.50 ($R_{50}$) improves while tolerant counting error ($\mathcal{E}_{\mathrm{count}}$) falls, and over-suppression. We formalize a four-criterion probe-and-sweep protocol for disciplining the interpretation 
    
[^90]: FARE：用于捕获“偷换”（Bait-and-Switch）图像生成器的法证式接受区域估计

    FARE: Forensic Acceptance Region Estimation for Catching Bait-and-Switch Image Generators

    [https://arxiv.org/abs/2609.30982](https://arxiv.org/abs/2609.30982)

    本文提出FARE，通过在认证生成器的图像上训练并挖掘困难样本以收紧接受区域，实现部署后仅凭单张生成图像即可检测服务商是否“偷换”了图像生成器，用于不透明图像生成API的完整性审计。

    

    现代AI图像生成器越来越多地以不透明的API形式部署，客户可以查询已部署的服务，却无法查看模型权重或架构。这带来了一个现实挑战：提供商可能用某个生成器通过治理认证，随后在部署时悄悄换成更便宜、质量更低的生成器，从而损害公众信任，甚至在高风险领域危及安全。我们研究部署时的完整性审计，并提出FARE（法证接受区域估计，Forensic Acceptance Region Estimation）。通过在从经过认证的生成器采样的图像上训练FARE，将该生成器“注册”（enroll）到系统中。部署之后，FARE可以仅凭单张生成图像判断该图像是否与已注册的生成器一致。FARE的特征基于此前为取证应用提出的、图像生成器特有的伪影（artifacts）。FARE在训练过程中通过挖掘困难样本来放大这些特征，从而收紧接受区域（原文摘要在此处截断）。

    arXiv:2609.30982v1 Announce Type: cross  Abstract: Modern AI image generators are increasingly deployed as opaque APIs, where customers can query the deployed service, but cannot inspect model weights or architecture. This creates a practical challenge: a provider may pass governance certification with one generator and later silently switch to a cheaper and lower-quality one for deployment, compromising public trust or even safety in high-stakes domains. We study integrity auditing at deployment time and propose FARE (Forensic Acceptance Region Estimation). A certified generator is enrolled by training FARE on images sampled from that generator. After deployment, FARE can determine whether a generated image is consistent with the enrolled generator---using only that image. FARE's features are based on image generator-specific artifacts that have been proposed for forensic applications. FARE amplifies these features during training by finding hard samples that tighten the acceptance re
    
[^91]: 均匀离散扩散模型需要时间吗？

    Does Uniform Discrete Diffusion Need Time?

    [https://arxiv.org/abs/2609.30977](https://arxiv.org/abs/2609.30977)

    本研究从理论和实证两方面证明，在语言建模等有限数据场景下，均匀离散扩散模型的时间条件化在很大程度上是不必要的，与时间无关的预测器能达到甚至超越时间条件化模型的性能。

    

    均匀离散扩散模型（UDMs）通常使用显式的时间条件化机制，但我们发现这在实践中往往是不必要的。本文首先证明，在总体最优意义上，UDM的预测器通常依赖于时间：时间控制着模型应该在多大程度上信任观测到的上下文。随后我们证明，在与语言相关的有限数据设置中，这种依赖性可以变得微不足道。当一个被破坏的训练序列与其原始干净序列的距离仍远小于其与其他竞争训练序列的距离时，经验最优预测器在扩散轨迹的大部分范围内对时间几乎不敏感，尽管这一保证在高噪声端点附近会减弱。实证结果表明，训练好的语言UDM在轨迹的大部分范围内表现出有限的时间敏感性，而与时间无关的预测器在各种数据集和训练目标上与时间条件化模型相比仍具竞争力，且往往表现更优。

    arXiv:2609.30977v1 Announce Type: cross  Abstract: Uniform discrete diffusion models (UDMs) commonly use explicit time conditioning, but we find that it can often be unnecessary in practice. In this paper, we first show that the population-optimal UDM predictor generally depends on time: time controls how much the model should trust the observed context. We then show that this dependence can become negligible in finite-data settings relevant to language. When a corrupted training sequence remains much closer to its original clean sequence than to competing training sequences, the empirical-optimal predictor is nearly insensitive to time over most of the diffusion trajectory, where the guarantee weakens toward the high-noise endpoint. Empirically, trained language UDMs exhibit limited time sensitivity over most of the trajectory, while time-agnostic predictors remain competitive with, and often outperform, time-conditioned models across datasets and training objectives. These results ch
    
[^92]: 面向滚动轴承剩余使用寿命预测的因子化轴卷积门控循环单元与动态自适应池化方法

    Factorized axis convolutional gated recurrent unit with dynamic adaptive pooling for remaining useful life prediction of rolling bearings

    [https://arxiv.org/abs/2609.30972](https://arxiv.org/abs/2609.30972)

    本文提出一种因子化轴卷积门控循环单元结合动态自适应池化的方法，利用多尺度各向异性卷积与双轴注意力增强时频图中的方向性特征并自适应聚合关键信息，从而提升滚动轴承剩余使用寿命预测的精度。

    

    卷积神经网络（CNN）被广泛用于从振动信号的时频表示（TFR）中预测滚动轴承的剩余使用寿命（RUL）。然而，在退化过程中，时频表示中的特征结构主要沿频率轴或时间轴方向排列，这使得传统CNN的各向同性卷积核难以捕捉方向性结构。此外，全局平均池化（GAP）沿两个轴进行平均，可能会掩盖显著激活的位置和集中程度。本研究提出了一种因子化轴卷积门控循环单元（GRU），采用多尺度各向异性卷积和双轴卷积块注意力模块来增强方向性特征，并突出显著的时频区域。动态自适应池化（DAP）能够自适应地聚合所提取特征图中的时频轴信息，而GRU则捕捉时间动态特性。

    arXiv:2609.30972v1 Announce Type: new  Abstract: Convolutional neural networks (CNN) are widely used to predict the remaining useful life (RUL) of rolling bearings from time-frequency representations (TFRs) of vibration signals. However, during degradation, characteristic structures in TFRs align predominantly along the frequency or time axis, making it challenging for conventional CNN isotropic kernels to capture directional structure. Furthermore, global average pooling (GAP) averages across axes, potentially obscuring the locations and concentrations of salient activations. This study introduces a factorized-axis convolutional gated recurrent unit (GRU) that employs multiscale anisotropic convolution and dual-axis convolution block attention module to enhance directional features and highlight salient time-frequency regions. Dynamic adaptive pooling (DAP) adaptively aggregates the time-frequency-axis information from the extracted feature maps, whereas a GRU captures temporal dynami
    
[^93]: SciHorizon-eLab：一个面向科学具身智能体可扩展基准测试的智能体式协议到任务编译器

    SciHorizon-eLab: An Agentic Protocol-to-Task Compiler for Scalable Benchmarking of Scientific Embodied Agents

    [https://arxiv.org/abs/2609.30971](https://arxiv.org/abs/2609.30971)

    提出了SciHorizon-eLab，一个智能体式的协议到任务编译器，能够将自然语言科学实验协议自动编译为语义一致、可执行且经多阶段仿真验证的具身任务，从而实现对科学具身智能体的可扩展基准测试。

    

    具身智能体为实现科学实验自动化提供了一条有前景的路径，然而其发展受到缺乏可靠且系统的评估环境的制约。现有的基于仿真的实验室基准严重依赖人工任务工程，使得难以系统地将多样化的科学协议大规模编译为可执行且可验证的具身任务。为应对这一挑战，我们提出了SciHorizon-eLab，这是一个智能体式的协议到任务编译器，它将科学具身任务的构建形式化为一个编译问题。给定科学实验的自然语言协议，SciHorizon-eLab通过语义接地、可执行任务合成以及多阶段基于仿真的认证，逐步将实验室协议编译为语义保持的具身任务。该系统能够生成语义接地的环境、可执行的操作程序以及步骤级的成功判定标准……

    arXiv:2609.30971v1 Announce Type: new  Abstract: Embodied agents offer a promising route to automating scientific experimentation, yet their progress is constrained by the lack of reliable and systematic evaluation environments. Existing simulation-based laboratory benchmarks rely heavily on manual task engineering, making it challenging to systematically compile diverse scientific protocols into executable and verifiable embodied tasks at scale. To address this challenge, we introduce SciHorizon-eLab, an agentic protocol-to-task compiler that formulates scientific embodied task construction as a compilation problem. Given a natural-language protocol of scientific experiments, SciHorizon-eLab progressively compiles laboratory protocols into semantic-preserving embodied tasks through semantic grounding, executable task synthesis, and multi-stage simulation-based certification. The system generates semantically grounded environments, executable manipulation programs, and step-level succe
    
[^94]: MoMHa：面向准确率、安全性与Token消耗的大语言模型Harness多目标优化

    MoMHa: Multi-Objective Optimization of LLM Harnesses over Accuracy, Safety, and Tokens

    [https://arxiv.org/abs/2609.30967](https://arxiv.org/abs/2609.30967)

    该论文提出MoMHa，将大语言模型harness设计建模为在准确率、行为安全性和token成本三个目标上的搜索问题，并证明单阶段联合奖励的智能体提议方案在十七个领域上优于两阶段、仅标量反馈和仅准确率等所有替代方案。

    

    大多数关于改进大语言模型的工作都将准确率作为唯一目标。我们认为，harness——即围绕模型的Python代码，负责构建提示词、路由调用和解析输出——是一个一等的设计层面，其质量本质上是多目标的：一个准确率很高却不拒绝任何不安全请求的harness，或者一个多消耗一个数量级token的harness，都不是好的harness。我们提出了Meta-Harness，一个将harness设计建模为在三个领域内目标（准确率、行为安全性和token成本）上进行搜索的系统，该搜索由一个具备完整文件系统访问权限的智能体提议者来解决，它可以访问先前的harness源代码、执行轨迹和评分产物。我们的核心发现是，单阶段联合奖励的提议者优于所有替代方案，包括两阶段“先准确率后token”的消融变体、仅标量反馈以及仅准确率基线。我们在十七个领域上进行评估：七个……

    arXiv:2609.30967v1 Announce Type: new  Abstract: Most work on improving large language models treats accuracy as the sole objective. We argue that the harness, the Python code surrounding the model that constructs prompts, routes calls, and parses outputs, is a first-class design surface whose quality is inherently multi-objective: an accurate harness that refuses no unsafe request, or that consumes an order of magnitude more tokens, is not a good harness. We present Meta-Harness, a system that casts harness design as search over three per-domain objectives (accuracy, behavioural safety, and token cost) solved by an agentic proposer (Claude Code) with full filesystem access to prior harness source, execution traces, and scoring artifacts. Our central finding is that a singlephase joint-reward proposer (MoMHa) outperforms every alternative, including a two-phase "accuracy then tokens" ablation, scalar-only feedback, and an accuracy-only baseline. We evaluate on seventeen domains: seven 
    
[^95]: FTB图：确定并验证多语言语言模型中的首词广播器与语言身份头电路

    FTB Graph: Determining and Validating First-token Broadcasters and Language-Identity Head Circuits in Multilingual Language Models

    [https://arxiv.org/abs/2609.30954](https://arxiv.org/abs/2609.30954)

    该研究结合边归因修补与精确激活修补验证，对六种模型架构进行端到端电路分析，首次系统确定并验证了多语言大模型中决定首词语言身份的“首词广播器”与语言身份头电路。

    

    在多语言环境中运行的大语言模型必须在生成早期确定目标响应语言，然而控制首词语言身份决策的因果电路仍未得到充分描绘。我们对跨越四个系列的六种模型架构进行了端到端的结构电路分析：GPT-2、BLOOM-560M、Pythia-1B/2.8B以及Qwen2.5-1.5B Base/Instruct。使用带FP16主动截断的边归因修补技术，随后以2000个候选边搜索上限进行精确激活修补验证，我们提取了驱动首词语言广播的有向无环图。在独立模型中，我们观察到深层或中深层广播枢纽，尽管证据在Pythia-2.8B和BLOOM-560M中最为有力，因为GPT-2和Pythia-1B几乎没有留下图外的头可供比较，而两个Qwen2.5-1.5B变体则在必要性检验中出现了反转。从Pythia-1B扩展到2.8B扩大了节点……

    arXiv:2609.30954v1 Announce Type: new  Abstract: Large language models operating in multilingual contexts must resolve target response languages early in generation, yet the causal circuitry governing first-token language identity decisions remains poorly mapped. We present an end-to-end structural circuit analysis across six model architectures spanning four families: GPT-2, BLOOM-560M, Pythia-1B/2.8B, and Qwen2.5-1.5B Base/Instruct. Using Edge Attribution Patching (EAP) with FP16 active clamping, followed by exact activation patching verification with a 2,000-candidate-edge search ceiling, we extract directed acyclic graphs driving first-token language broadcasting. Across the standalone models, we observe deep or mid-to-deep broadcasting hubs, though the evidence is strongest for Pythia-2.8B and BLOOM-560M because GPT-2 and Pythia-1B leave few out-of-graph heads for comparison, while both Qwen2.5-1.5B variants invert the necessity check. Scaling from Pythia-1B to 2.8B expands node p
    
[^96]: MVVBench：面向视觉语言模型4D推理能力的基准测试

    MVVBench: Benchmarking 4D Reasoning in Vision-Language Models

    [https://arxiv.org/abs/2609.30952](https://arxiv.org/abs/2609.30952)

    MVVBench是一个基于真实多相机数据构建的多视角视频推理基准，其问题在视角和时间轴上均无法通过单目信息解答，只有联合跨视角、跨时间的4D推理才能求解，可用于系统评测视觉语言模型的4D推理能力。

    

    多视角视频理解需要整合来自多个（通常互不重叠的）相机视频流的空间与时间证据：追踪实体在不同视角之间的切换、跨时间对齐事件，并对潜在的4D连续性而非任何单一可见帧进行推理。我们提出了MVVBench，一个基于真实世界多相机数据集构建的多视角视频推理基准。其中的问题经过精心筛选，在视角轴和时间轴上均具有单目模糊性：每个问题都无法从指定输入集合中的任何单一视角回答，且大多数问题进一步无法从任何单一时刻回答。每个问题只有通过跨视角与跨时间的联合推理才能被唯一求解。MVVBench涵盖多样的动态场景，并考察六种能力：隐式/显式属性识别、隐式/显式相对距离、相对相机位姿以及组合计数。

    arXiv:2609.30952v1 Announce Type: cross  Abstract: Multi-view video understanding requires integrating spatial and temporal evidence across multiple, often non-overlapping camera streams: tracking entities as they transition between viewpoints, aligning events across time, and reasoning about latent 4D continuity rather than any single visible frame. We introduce MVVBench, a benchmark for multi-view video reasoning built from real world multi camera datasets. Questions are curated to be monocular-ambiguous along both the view and the temporal axis: each question is unanswerable from any single view in the designated input set, and the majority are further unanswerable from any single moment. Each question becomes uniquely solvable only by jointly reasoning across views and across time. MVVBench spans diverse dynamic scenes and probes six capabilities: implicit/explicit attribute identification, implicit/explicit relative distance, relative camera pose, and compositional counting, with 
    
[^97]: PORL：面向作业车间调度问题的预训练离线强化学习

    PORL: Pretrained Offline Reinforcement Learning for the Job Shop Scheduling Problem

    [https://arxiv.org/abs/2609.30948](https://arxiv.org/abs/2609.30948)

    该论文提出PORL方法，将基于仿真的在线预训练与离线微调相结合，并利用KL散度策略约束将通用调度策略适配到特定生产数据分布，从而克服作业车间调度问题中仿真到现实的差距。

    

    arXiv:2609.30948v1 公告类型：cross 摘要：作业车间调度问题是工业优化中的一个基础性组合优化问题。本工作提出了预训练离线强化学习，这是一种将基于仿真的在线预训练与针对特定生产数据的离线微调相结合的混合方法。通过在线交互进行强化学习能够探索通用的调度策略，但通常依赖于仿真环境，且可能受到仿真与现实差距的影响。相比之下，离线强化学习通过从历史数据中学习来避免与环境的直接交互，但其性能在很大程度上受数据集质量和覆盖范围的影响。PORL 结合了两种范式的优势：首先通过在线交互学习一个通用调度策略，随后将其离线适配到目标分布。文中引入了基于 KL 散度的策略约束，以限制偏离（原文摘要至此处被截断）

    arXiv:2609.30948v1 Announce Type: cross  Abstract: The Job Shop Scheduling Problem (JSSP) is a fundamental combinatorial optimization problem in industrial optimization. This work introduces Pretrained Offline Reinforcement Learning (PORL), a hybrid approach that combines simulation-based online pretraining with offline fine-tuning on production-specific data.   Reinforcement learning through online interaction enables exploration of general scheduling strategies, but typically relies on simulation environments and may suffer from a simulation-to-reality gap. In contrast, offline RL avoids direct interaction with the environment by learning from historical data, but its performance is strongly influenced by dataset quality and coverage. PORL combines the strengths of both paradigms by first learning a general scheduling policy through online interaction and subsequently adapting it offline to a target distribution. A KL-divergence-based policy constraint is introduced to limit deviatio
    
[^98]: OneWorld：在世界模型中学习跨动作的一致物理特性

    OneWorld: Learning Consistent Physics Across Actions in World Models

    [https://arxiv.org/abs/2609.30946](https://arxiv.org/abs/2609.30946)

    OneWorld提出了一种共享机制的反事实生成框架，通过在共同潜在物理机制下联合建模多个动作条件化的未来，解决了世界模型在不同动作干预下物理属性预测不一致的问题。

    

    动作条件化的视频世界模型旨在预测不同动作下场景的演化，这一能力对于在动态环境中实现可靠的规划、决策与交互至关重要。然而，从同一初始场景独立生成的多个未来可能各自看起来合理，却蕴含着互不相容的物理属性，例如摩擦力或质量。这种不一致性会导致跨干预的矛盾预测，使模型难以对底层世界保持连贯的理解，并限制了其在规划与决策中的可靠性。为解决这些问题，我们提出了OneWorld，一个共享机制的反事实生成框架，它在共同的潜在物理机制下联合建模多个动作条件化的未来。物理机制解释器首先从每个动作-结果分支中推断潜在机制的分布。这些分布（摘要在此处截断）

    arXiv:2609.30946v1 Announce Type: cross  Abstract: Action-conditioned video world models aim to predict scene evolution under different actions, a capability that is essential for reliable planning, decision-making, and interaction in dynamic environments. However, futures generated independently from the same initial scene may each appear plausible while implying incompatible physical properties, such as friction or mass. This inconsistency can lead to contradictory predictions across interventions, making it difficult for the model to maintain a coherent understanding of the underlying world and limiting its reliability for planning and decision-making. To address these issues, we propose OneWorld, a shared-mechanism counterfactual generation framework that jointly models multiple action-conditioned futures under a common latent physical mechanism. A physical mechanism interpreter first infers a distribution over latent mechanisms from each action-outcome branch. These distributions 
    
[^99]: LogicTree-RAG：逻辑树引导的检索增强生成方法用于长篇专利撰写

    LogicTree-RAG: Logic Tree-guided Retrieval-Augmented Generation for Long-form Patent Drafting

    [https://arxiv.org/abs/2609.30943](https://arxiv.org/abs/2609.30943)

    提出LogicTree-RAG框架，通过构建层次化逻辑树作为全局组织主干来引导检索增强生成，在不依赖专家先验知识的情况下实现长篇专利文档的整体自动化撰写。

    

    长篇技术文本生成是知识密集型工作流程的基础，但对大型语言模型（LLM）而言仍然具有挑战性，因为它需要全局一致的逻辑结构化以及超越局部连贯性的忠实技术推理。专利撰写是这一挑战的典型实例，需要通过持续的多专家协作来整体生成一份法律合规且技术上详尽的文档。现有方法通常专注于局部章节生成或依赖手工制作的提纲，限制了现实场景中的可扩展自动化。在这项工作中，我们提出了LogicTree-RAG，这是一个逻辑树引导的检索增强生成框架，它构建层次化逻辑树作为全局组织主干，用于组织和支撑技术披露内容，而无需依赖专家定义的撰写先验知识。逻辑树中的每个节点代表一个技术要素，并通过……（摘要在此处被截断）

    arXiv:2609.30943v1 Announce Type: new  Abstract: Long-form technical text generation underpins knowledge-intensive workflows, yet remains challenging for large language models (LLMs) due to the need for globally consistent logical structuring and faithful technical reasoning beyond local coherence. Patent drafting is a canonical instance of this challenge, demanding holistic generation of a legally compliant and technically exhaustive document through sustained multi-expert collaboration. Existing approaches often focus on partial section generation or rely on manually crafted outlines, limiting scalable automation in realistic settings. In this work, we propose LogicTree-RAG, a logic tree-guided retrieval-augmented generation framework that induces a hierarchical logic tree as a global organizational backbone to organize and ground technical disclosures, without relying on expert-defined drafting priors. Each node in the logic tree represents a technical element and is constructed thr
    
[^100]: Spackle：基于自适应高斯的单图像大视角新视角合成补全

    Spackle: Completing Large View Single Image NVS with Adaptive Gaussians

    [https://arxiv.org/abs/2609.30941](https://arxiv.org/abs/2609.30941)

    提出轻量级残差学习框架Spackle，通过自动识别重建较差区域并学习残差3D高斯，缓解了固定高斯数量带来的容量竞争问题，在不牺牲推理效率的情况下实现了高质量的大视角偏差单图像新视角合成。

    

    单图像新视角合成（NVS）能够从单张输入图像实现对未观测视角的照片级真实感渲染。实用的NVS系统需要两项关键能力：对遮挡区域的鲁棒重建以及高推理效率。虽然结合前馈3D高斯泼溅（3DGS）与扩散模型的混合解耦框架在大视角偏差NVS方面展现出前景，但它们存在容量竞争问题：固定数量的高斯基元迫使资源从可见区域转移到新暴露的去遮挡区域，当目标视角与输入视角显著偏离时会降低原始场景的保真度。为解决这一问题，我们提出了Spackle，一个在不牺牲效率的前提下缓解容量竞争的轻量级残差学习框架。Spackle分三个阶段运行：从给定视角预测基础3DGS属性、自动识别重建效果较差的区域，以及学习一个经过优化的残差3DGS。

    arXiv:2609.30941v1 Announce Type: cross  Abstract: Single-image novel view synthesis (NVS) enables photorealistic rendering of un- observed viewpoints from a single input. Practical NVS systems require two key capabilities: robust reconstruction of occluded regions and high inference effi- ciency. While hybrid decoupled frameworks combining feedforward 3D Gaussian Splatting (3DGS) and diffusion models show promise for large-view-deviation NVS, they suffer from capacity competition: a fixed number of Gaussians forces resource shifts from visible to newly disoccluded areas, degrading original scene fidelity when the target view deviates significantly from the input. To address this, we propose Spackle, a lightweight residual learning framework that mit- igates capacity competition without sacrificing efficiency. Spackle operates in three stages: predicting base 3DGS attributes from given views, automatically identifying poorly reconstructed regions, and learning a residual 3DGS optimized
    
[^101]: 大语言模型智能体社会中的金融脆弱性：协调失灵与稳定机制

    Financial Fragility in Societies of LLM Agents: Coordination Failures and Stabilizing Mechanisms

    [https://arxiv.org/abs/2609.30940](https://arxiv.org/abs/2609.30940)

    该论文提出FRAIL受控实验框架，发现七种主流LLM智能体在银行挤兑、债务展期和众筹等金融环境中，即使无任何破坏指令也会因协调失灵而普遍出现集体性金融失败（77%的银行挤兑场景和83%的债务展期场景失败），并验证了补偿性承诺、中心化承诺协议和参与者主导联盟三种机制能够有效稳定系统。

    

    个体层面的自保性决策可能导致本可避免的集体失败。随着大语言模型（LLM）智能体在金融决策中扮演越来越重要的角色，金融AI安全不仅需要在个体智能体层面加以考虑，还需要在其共同构成的系统层面加以审视。我们通过FRAIL框架研究这一问题，这是一个受控实验框架，将LLM智能体置于三种动态金融环境中——银行挤兑、债务展期和回报型众筹——在这些环境中，智能体的决策会重塑其他智能体所面临的金融条件。通过对七个领先LLM的实验，我们发现即使没有任何智能体被指示去破坏系统稳定，也普遍存在集体脆弱性：77%的基线银行挤兑场景和83%的债务展期场景以失败告终。随后，我们比较了三种基于补偿性承诺、中心化承诺协议和参与者主导联盟的互动机制，这三种机制均能改善……

    arXiv:2609.30940v1 Announce Type: new  Abstract: Individually protective decisions can produce avoidable collective failures. As large language model (LLM) agents take on greater roles in financial decision-making, financial AI safety must therefore be considered not only at the level of individual agents, but also at the level of the systems they jointly create. We study this problem with FRAIL, a controlled experimental framework that places LLM agents in three dynamic financial environments---bank runs, debt rollover, and reward crowdfunding---where agents' decisions reshape the financial conditions faced by others. Across seven leading LLMs, we find widespread collective fragility even when no agent is instructed to destabilize the system: 77\% of baseline bank-run episodes and 83\% of debt-rollover episodes end in failure. We then compare three interaction mechanisms based on compensated commitments, centralized commitment agreements, and participant-led coalitions. All three impr
    
[^102]: MACBT：一种具有纵向记忆的多智能体认知行为疗法决策支持系统

    MACBT: A Multi-Agent Cognitive Behavioral Therapy Decision Support System with Longitudinal Memory

    [https://arxiv.org/abs/2609.30939](https://arxiv.org/abs/2609.30939)

    该论文提出了MACBT决策支持系统，将CBT五阶段工作流编码为五个协作智能体，并借助纵向记忆模块跨会谈追踪认知扭曲情况，从而为临床医生生成会谈前病理报告和干预优先级建议，缓解CBT规模化应用的瓶颈。

    

    认知行为疗法（CBT）是一种基于循证证据的抑郁症一线治疗方法，但其规模化应用受限于临床医生在会谈前准备、会谈后记录以及纵向认知病理追踪上所花费的大量时间。我们提出了一款面向临床医生的AI决策支持系统，该系统将多智能体CBT框架（MACBT）与CBT专用的纵向记忆模块（CD Memory）相结合。MACBT将CBT五阶段工作流程（评估、苏格拉底式提问、认知重构、行为实验和治疗监测）编码为五个协作智能体。CD Memory跨会谈追踪认知扭曲的类型、频率、严重程度及重构效果，以生成会谈前病理报告和干预优先级建议。我们通过双角色大语言模型模拟构建了中文CBT对话语料库，并采用监督微调方法训练了Qwen3-14B骨干模型……（摘要在此处被截断）

    arXiv:2609.30939v1 Announce Type: new  Abstract: Cognitive behavioral therapy (CBT) is an evidence-based first-line treatment for depression, yet its scale is constrained by the time clinicians spend on pre-session preparation, post-session documentation, and longitudinal cognitive-pathology tracking. We present a clinician-facing AI decision-support system that combines a multi-agent CBT framework (MACBT) with a CBT-specific longitudinal memory module (CD Memory). MACBT encodes the five-stage CBT workflow (assessment, Socratic questioning, cognitive restructuring, behavioral experiments, and treatment monitoring) into five collaborative agents. CD Memory tracks cognitive-distortion type, frequency, severity, and restructuring efficacy across sessions to generate pre-session pathology reports and intervention-priority recommendations. We construct a Chinese CBT dialogue corpus via dual-role large language model simulation and train a Qwen3-14B backbone with supervised fine-tuning and d
    
[^103]: 面向大语言模型推理的自博弈搜索蒸馏

    Self-Play Search Distillation for Large Language Model Reasoning

    [https://arxiv.org/abs/2609.30936](https://arxiv.org/abs/2609.30936)

    提出自博弈搜索蒸馏框架（SPSD），利用棋盘游戏中类MuZero网络的自博弈搜索记录生成超人思维链数据来训练大语言模型，仅凭自博弈数据即可将Qwen3-4B-Base在六个数学基准上的平均成绩从24.1提升到36，并展现出良好的跨领域迁移能力。

    

    提升大语言模型（LLM）的推理能力需要高质量的数据，这些数据应能暴露困难的决策、相互竞争的备选方案及其后果。而数据稀缺的原因在于合成数据质量低下以及人工标注成本高昂。我们提出了自博弈搜索蒸馏（Self-Play Search Distillation, SPSD），这是一个通过在棋盘游戏上训练的类MuZero网络进行自博弈来生成超人水平合成数据的框架。SPSD利用可执行环境将搜索过程转化为结构化的推理问题。在每个状态下，专家会识别出首选决策、合理的备选方案、对手可能的回应以及价值估计。通过将自博弈搜索记录转化为超人水平的思维链，我们以环境为依据的监督方式来训练大语言模型。尽管仅在自博弈搜索记录上进行训练，SPSD仍能迁移到未见过的数学任务上。在Qwen3-4B-Base上，该方法将六个数学基准测试的平均成绩从24.1提升到了36。

    arXiv:2609.30936v1 Announce Type: new  Abstract: Improving reasoning abilities in Large Language Models (LLMs) requires high-quality data that exposes difficult decisions, competing alternatives, and their consequences. Data scarcity is driven by the low quality of synthetic data and the cost of human labeling. We introduce Self-Play Search Distillation (SPSD), a framework for generating superhuman synthetic data via self-play of MuZero-like networks trained on board games. SPSD uses executable environments to turn search into structured reasoning problems. At each state, the expert identifies a preferred decision, plausible alternatives, plausible opponent replies, and value estimates. By converting the self-play search records into superhuman chains-of-thought, we train LLMs with environment-grounded supervision. Although trained only on self-play search records, SPSD transfers to unseen mathematics. On Qwen3-4B-Base, it raises the mean over six mathematics benchmarks from 24.1 to 36
    
[^104]: 估计并正交化未知的预训练梯度以实现大语言模型的持续微调

    Estimating and Orthogonalizing Unknown Pre-training Gradients for Continual Fine-tuning of Large Language Models

    [https://arxiv.org/abs/2609.30935](https://arxiv.org/abs/2609.30935)

    提出EoupCT框架，通过动态生成最易受遗忘影响的伪数据来估计未知的预训练梯度，并将其与新任务的梯度进行正交化投影，从而在持续微调大语言模型时有效保护其固有的通用知识，避免灾难性遗忘。

    

    持续微调对于大语言模型动态适应现实世界环境至关重要，然而它不可避免地遭受灾难性遗忘的困扰，尤其是先前任务性能的下降以及大模型通用知识的退化。尽管现有方法（如正交梯度投影）能够缓解各种微调任务中的遗忘问题，但由于现成预训练大模型所需的原始数据和梯度严格未知且高度多样化，这些方法从根本上无法保留预训练大模型固有的通用知识。为弥合这一关键差距，我们提出了EoupCT，一个旨在估计并正交化未知预训练梯度以实现大语言模型持续微调的新型框架。具体而言，EoupCT通过动态生成对新任务最易受遗忘影响的伪数据来估计预训练梯度……（原文摘要在此处截断）

    arXiv:2609.30935v1 Announce Type: cross  Abstract: Continual fine-tuning is essential for large language models (LLMs) to dynamically adapt to real-world environments, yet it inevitably suffers from catastrophic forgetting, particularly the performance degradation of previous tasks and LLMs' general-purpose knowledge. Although existing methods, such as orthogonal gradient projection, mitigate the forgetting across various fine-tuning tasks, they fundamentally fail to preserve pre-training LLMs' inherent general-purpose knowledge because the original data and gradients of off-the-shelf pre-training LLMs required by these methods are strictly unknown and highly diverse. To bridge this critical gap, we propose EoupCT, a novel framework designed to Estimate and Orthogonalize Unknown Pre-training gradients for Continual LLM fine-Tuning. Specifically, EoupCT estimates pre-training gradients by dynamically generating pseudo data that is most susceptible to forgetting for new tasks through a l
    
[^105]: UltraG-Bench：一个用于评估大型视觉语言模型在超声图像上像素级证据定位能力的多任务基准

    UltraG-Bench: A Multi-task Benchmark for assessing Large Vision-Language Models on Pixel-level Evidence Grounding in Ultrasound

    [https://arxiv.org/abs/2609.30928](https://arxiv.org/abs/2609.30928)

    该论文提出了UltraG-Bench，一个基于40个公开超声分割数据集标注、涵盖指令引导分割、证据定位VQA和证据定位报告生成三个任务的大规模多任务基准，用于系统评估大型视觉语言模型在超声图像上的像素级证据定位能力，并揭示了语义理解与细粒度像素级定位之间的显著差距。

    

    超声是最广泛使用的医学成像方式之一，近年来大型视觉语言模型（VLM）在超声图像理解方面展现出日益增强的能力。然而，这些模型无法提供与其语义预测相一致的像素级视觉证据，且其在超声领域的细粒度定位能力在很大程度上仍不清楚。我们提出了UltraG-Bench，一个用于评估超声像素级证据定位能力的大规模多任务基准。UltraG-Bench通过标注40个涵盖13个解剖类别的公开超声分割数据集构建而成，包含三个渐进式任务：指令引导分割、证据定位视觉问答（VQA）和证据定位报告生成，分别包含331,125、666,779和138,832条标注。对14个最先进模型的综合评估显示，语义理解与细粒度像素级定位能力之间存在显著差距。

    arXiv:2609.30928v1 Announce Type: cross  Abstract: Ultrasound is one of the most widely used medical imaging modalities, and recent large vision-language models(VLMs) have shown increasing capabilities in ultrasound image understanding. However, these models fail to provide pixel-level visual evidence aligned with their semantic predictions, and their fine-grained grounding capability in ultrasound remains largely unclear. We introduce UltraG-Bench, a large-scale multi-task benchmark for evaluating pixel-level evidence grounding in ultrasound. UltraG-Bench is built by annotating 40 public ultrasound segmentation datasets spanning 13 anatomical categories, and comprises three progressive tasks: instruction-guided segmentation, evidence-grounded VQA, and evidence-grounded report generation, with 331125, 666779, and 138832 annotations, respectively. Comprehensive evaluation of 14 state-of-the-art models reveals a substantial gap between semantic understanding and fine-grained pixel-level 
    
[^106]: JevSoup：面向免训练 LoRA 组合的系统一路由方法

    JevSoup: System-One Routing for Training-Free LoRA Composition

    [https://arxiv.org/abs/2609.30922](https://arxiv.org/abs/2609.30922)

    JevSoup提出了一种免训练的LoRA专家组合框架，通过“系统一”结构化概率路由选出两个专家，并利用正交投影将两个专家的更新等权融合，无需辅助数据或额外训练即可在多任务上取得性能提升。

    

    构建可适应的AI系统需要在多样化任务之间有效协调各类专门能力。低秩适应（LoRA）实现了模块化的专业知识，但现有的路由方法可能需要辅助数据、额外训练或自回归解码。我们提出了JevSoup，这是一个将“系统一”专家路由与“系统二”执行相分离的免训练框架。仅利用输入和专家描述，Jev通过结构化概率选择两个专家。JevSoup保留首要专家的更新，将第二个专家的更新投影到第一个更新行空间的正交补空间上，并以等权重将二者组合。在14个PorTAL任务和三种Qwen3模型规模上，与所评估的最强外部基线相比，JevSoup在任务宏平均准确率上取得了最高1.19%、在样本微平均准确率上取得了最高1.21%的绝对提升。我们的代码已发布于 https://github.com/Leowang980/JevSoup。

    arXiv:2609.30922v1 Announce Type: new  Abstract: Building adaptable AI systems requires effective coordination of specialized capabilities across diverse tasks. Low-rank adaptation (LoRA) enables modular expertise, but existing routing approaches may require auxiliary data, additional training, or autoregressive decoding. We propose JevSoup, a training-free framework separating System One expert routing from System Two execution. Using only the input and expert descriptions, Jev selects two experts through structured probabilities. JevSoup retains the leading expert's update, projects the second onto the orthogonal complement of the first update's row space, and combines them with equal weights. Across 14 PorTAL tasks and three Qwen3 scales, JepSoup achieves absolute gains of up to 1.19\% in task-macro and 1.21\% in sample-micro accuracy over the strongest evaluated external baselines. Our code is available at https://github.com/Leowang980/JevSoup.
    
[^107]: 对哪种模型变化具有鲁棒性？鲁棒反事实解释的统一评估

    Robust to Which Model Change? A Unified Evaluation of Robust Counterfactual Explanations

    [https://arxiv.org/abs/2609.30918](https://arxiv.org/abs/2609.30918)

    本文提出一个统一的跨家族评估协议，在固定事实实例与反事实解释的前提下，针对相同的八种模型变化类型比较六种鲁棒反事实解释方法与两种基线，发现各方法的相对性能和失败模式随变化类型而异，因此现有报告的鲁棒性分数彼此不可比。

    

    鲁棒反事实解释承诺提供在其背后模型发生变化后仍然有效的补救措施。它们是否兑现这一承诺取决于变化是什么：参数的微小扰动、在新数据上的重新训练以及更换新架构是不同的事件，而每种现有方法都是针对其所设计的那个特定变化进行评估的。因此，已报告的鲁棒性分数回答的是不同的问题，彼此之间无法比较。我们提出了一个统一的跨家族评估协议，该协议保持事实实例和生成的反事实实例固定不变，同时针对相同的八种模型变化类型测试每种方法。该基准在四个表格数据集上比较了六种鲁棒方法和两种标准基线方法，通过每个变化后分类器的输出对其进行刻画，并报告经验鲁棒性以及覆盖率、基准有效性和接近度。我们发现，相对性能和失败模式在不同的变化家族之间存在差异。

    arXiv:2609.30918v1 Announce Type: cross  Abstract: Robust counterfactual explanations promise recourse that still works after the model behind it changes. Whether they keep that promise depends on what the change is. A small perturbation of the parameters, retraining on new data, and a new architecture are different events, and each existing method is evaluated against the one it was built for. Reported robustness scores, therefore, answer different questions and cannot be compared. We propose a unified cross-family evaluation protocol that holds factual instances and generated counterfactuals fixed while testing every method against the same eight types of model change. The benchmark compares six robust methods and two standard baselines on four tabular datasets. It characterizes every changed classifier through its outputs and reports empirical robustness together with coverage, base validity, and proximity. We find that relative performance and failure modes vary across change famil
    
[^108]: 在网页图上训练图基础模型

    Training Graph Foundation Models on The Web Graph

    [https://arxiv.org/abs/2609.30894](https://arxiv.org/abs/2609.30894)

    Acacia是一个仅用Common Crawl网页图从零训练的图基础模型，无需额外训练即可支持任意特征维度及节点分类、链接预测、节点聚类、图生成等多种任务，具备上下文学习能力且不依赖预训练LLM，证明了图模型可以像LLM一样从零涌现能力。

    

    我们介绍了Acacia，一个在网页图上训练的图基础模型。Acacia（i）支持任意的特征维度和语义，无需额外训练；（ii）支持广泛的任务，包括节点分类、链接预测、节点聚类和图生成，同样无需额外训练；（iii）具备上下文学习能力；（iv）不依赖预训练的大型语言模型（LLM）。特别值得注意的是，现有的图基础模型通常需要训练额外的分类头或特征投影器来适应新的图或新的标签，而Acacia不需要。此外，现有的图基础模型往往通过与预训练LLM拼接组合来获得其能力，而Acacia仅使用Common Crawl网页图从零开始训练。这也是一项重要的成果，因为它提供了证据，证明图模型可以像LLM一样从零开始获得涌现能力。

    arXiv:2609.30894v1 Announce Type: new  Abstract: We introduce Acacia, a graph foundation model, trained on the web graph. Acacia (i) supports arbitrary feature dimensionalities and semantics without additional training, (ii) supports a wide range of tasks, including node classification, link prediction, node clustering, and graph generation, without additional training, (iii) has in-context learning capabilities, and (iv) does not rely on pretrained LLMs. In particular, existing graph foundation models often require training additional classification heads or feature projectors to accommodate new graphs or new labels, whereas Acacia does not. Moreover, existing graph foundation models often gain their capabilities by being stitched together with pretrained LLMs, whereas Acacia is trained from scratch using only the Common Crawl web graph. This is also an important result because it provides evidence that graph models can acquire emergent capabilities from scratch like LLMs.
    
[^109]: 面向ISAC中统一语义通信与语义感知的自适应导频选择

    Adaptive Pilot Selection for Unified Semantic Communication and Semantic Sensing in ISAC

    [https://arxiv.org/abs/2609.30891](https://arxiv.org/abs/2609.30891)

    本文提出SemISAC框架，在单一双功能波形中统一实现语义通信与语义感知，通过自适应导频选择优化OFDM资源分配，在车联网场景中同时完成道路场景分割、目标分类与测距任务。

    

    语义通信和通信感知一体化（ISAC）是未来6G无线网络中极具前景的技术。现有研究仅将语义技术应用于ISAC的通信模块或感知模块之一。在本工作中，我们提出了SemISAC，它在单一双功能波形中同时执行语义通信和语义感知。SemISAC使用一个联合语义编码器，为通信和感知提取任务特定的信息。我们在车辆场景中评估了SemISAC，在该场景中，车辆共享道路环境的像素级分割结果，并通过感知对周围物体进行分类和距离估计。在发射机端，深度学习编码器将输入的道路场景图像转换为语义符号，并将其放置在OFDM网格的数据单元上，而其余单元则作为信道状态信息估计和感知的导频。

    arXiv:2609.30891v1 Announce Type: cross  Abstract: Semantic communication (SemCom) and integrated sensing and communication (ISAC) are promising technologies for future 6G wireless networks. Existing studies have applied semantic technology to either the communication module or the sensing module of ISAC. In this work, we propose SemISAC, which performs both SemCom and semantic sensing within a single dual-function waveform. SemISAC uses a joint semantic encoder that extracts task-specific information for both communication and sensing. We evaluate SemISAC in a vehicular scenario in which vehicles share pixel-wise segmentation of the road environment and, through sensing, classify surrounding objects and estimate their ranges. On the transmitter side, a deep learning encoder converts the input road-scene image into semantic symbols and places them on the data cells of an OFDM grid, while the remaining cells serve as pilots for channel state information estimation and sensing. The pilot
    
[^110]: 从点击到跳转：利用应用原生深链接增强移动GUI智能体

    From Tapping to Hopping: Augmenting Mobile GUI Agents with App-Native Deeplinks

    [https://arxiv.org/abs/2609.30887](https://arxiv.org/abs/2609.30887)

    本文提出GUI-Hopper混合交互方法，通过静态分析发现并在真实设备上验证应用深链接，构建带落地页描述的深链接目录，让移动GUI智能体用深链接直接跳转导航、用GUI操作处理其余任务，从而显著提升真实商业应用中的任务成功率。

    

    移动GUI智能体通过点击和滑动等GUI操作来完成任务。这些操作在各类应用中具有广泛的适用性，但需要逐屏导航才能到达目标界面。而一次深链接调用就可以替代一系列逐屏的GUI操作。因此，我们引入了混合交互方式：使用深链接进行直接导航，使用GUI操作执行其他屏幕操作并作为后备手段。为实现这一目标，我们通过静态分析发现候选深链接，在真实设备上对其进行验证，并描述其观察到的落地页面。这一过程构建了一个经过验证且有据可依的深链接目录，将每个可用的深链接与其落地页面的描述配对。利用该目录，我们提出了GUI-Hopper，它在真实设备上的商业应用中提升了任务成功率，进一步证明了混合交互的优势。

    arXiv:2609.30887v1 Announce Type: new  Abstract: Mobile GUI agents complete tasks using GUI actions like taps and swipes. These actions are broadly applicable across applications, but reaching a navigation interface. A single deeplink call can replace a sequence of screen-by-screen GUI actions. We therefore introduce hybrid interaction, using deeplinks for direct navigation and GUI actions for other on-screen operations and fallback. To enable this, we discover candidate deeplinks through static analysis, validate them on real devices, and describe their observed landing screens. This process creates a verified and grounded deeplink catalog that pairs each working deeplink with a description of its landing screen. Using this catalog, we introduce GUI-Hopper, a improves task success in commercial applications on real devices, further demonstrating the benefits of hybrid interaction.
    
[^111]: 同样收到警告后，AI智能体避开较不拥挤的道路，而人类却选择了它

    Warned alike, AI agents avoid the less-crowded road while people take it

    [https://arxiv.org/abs/2609.30883](https://arxiv.org/abs/2609.30883)

    本研究通过双路径拥堵博弈实验发现，一句相同的警告会引发50个GPT智能体集体涌入拥堵道路、避开空旷道路，使平均通行时间从64分钟升至95分钟，而人类群体在相同条件下保持均衡，揭示了共享AI预测可通过自我实现的预期来扭曲稀缺容量分配的全新反馈机制。

    

    基于少数共享模型构建的AI智能体越来越多地代替许多人行动。关于他人行为的共同预测可以使它们的选择趋于一致，并改变稀缺容量的分配方式。我们在一个双路径拥堵博弈中测试了这种反馈机制。仅仅添加一句警告——他人可能会跟随某个路线建议——就使得由50个GPT智能体组成的人群涌向一条道路，同时避开几乎空无一车的另一条路。平均通行时间从64分钟上升到95分钟，尽管任何一个身处拥挤道路的智能体只要单独换路就能节省69分钟。这句警告反而阻止了它所预言的行为。这种模式持续了100轮。另外两个模型家族也表现出同样的偏移，但没有锁定到一条路上。12个全人类小组（240名参与者）在收到数值报告或路线建议加警告时都保持接近均衡。在包含另外240名参与者的24个混合小组中，预注册分析显示，失衡程度随智能体比例的增加而加剧，而人类则越来越多地选择了那条……

    arXiv:2609.30883v1 Announce Type: cross  Abstract: AI agents built on a few shared models increasingly act for many people. A shared forecast about others can align their choices and change how scarce capacity is allocated. We tested this feedback in a two-road congestion game. Adding one sentence warning that others might follow a routing tip made populations of 50 GPT agents crowd one road while avoiding the nearly empty alternative. Average travel time rose from 64 to 95 min, although any crowded-road agent could have saved 69 min by switching alone. The warning discouraged the very move it predicted. The pattern persisted for 100 rounds. Two other model families shifted the same way without locking onto one road. Twelve all-human groups (240 participants) stayed near balance under numerical reports or the tip and warning. In 24 mixed groups with a further 240 participants, imbalance grew with the share of agents in the registered analysis, while people increasingly took the road th
    
[^112]: EXAONE Demand 1.0：一个面向需求预测的时间序列基础模型

    EXAONE Demand 1.0: A Time Series Foundation Model for Demand Forecasting

    [https://arxiv.org/abs/2609.30880](https://arxiv.org/abs/2609.30880)

    该论文提出了EXAONE Demand——一个专为需求预测设计的时间序列基础模型，通过构建包含1130万条序列的需求专用语料库，以及基于四类需求类别（平滑、间歇、波动、块状）进行路由的低秩适配器架构，解决了通用时间序列基础模型难以处理需求数据短历史、频繁零值、缺货删失等特殊性质的问题。

    

    时间序列基础模型（TSFM）通常在来自多个领域的序列数据上进行预训练，其中需求序列仅占很小一部分。需求数据具有此类语料库中罕见的特性：历史数据短、频繁出现零值、因缺货导致的删失，以及序列未能记录的外生事件。为此，我们提出了EXAONE Demand，其构建基于两大要素：1）需求专用语料库；2）需求感知适配器。在语料库方面，我们从73个数据源收集了1130万条序列和484亿个观测值，并通过合成生成器补充开放需求数据中代表性不足的行为模式。在适配器方面，我们在冻结的通用领域主干网络上附加低秩分支，分别对应四类需求（平滑型、间歇型、波动型和块状型），并由一个读取输入序列八项无标度统计量的路由器来决定各分支的贡献权重。我们构建了两个版本的EXAONE Demand，其中一个在真实世界与合成需求数据上训练。

    arXiv:2609.30880v1 Announce Type: new  Abstract: Time series foundation models (TSFMs) are pretrained on series from diverse domains, where demand series make up only a small fraction. Demand data has properties that such corpora rarely contain: Short histories, frequent zeros, censoring by stock-outs, and exogenous events that the series does not record. To this end, we propose EXAONE Demand, built on 1) a demand-specific corpus and 2) a demand-aware adapter. For the corpus, we assemble 11.3M series and 48.4B observations from 73 sources, and a synthetic generator supplies the behaviour that open demand data under-represents. For the adapter, we attach low-rank branches to a frozen general-domain backbone, one for each of the four demand classes (smooth, intermittent, erratic, and lumpy), and a router that reads eight scale-free statistics of the input series decides how much each branch contributes. We build EXAONE Demand in two versions, one trained on real-world and synthetic deman
    
[^113]: TISD：基于轨迹干预的在线策略自蒸馏

    TISD: On-Policy Self-Distillation with Trajectory Intervention

    [https://arxiv.org/abs/2609.30878](https://arxiv.org/abs/2609.30878)

    该论文提出TISD算法，通过在师生分歧峰值处强制执行教师偏好的分支动作并重生成轨迹进行蒸馏，突破了在线策略自蒸馏无法监督学生未采样分支的、单纯依赖局部纠错的训练瓶颈。

    

    在线策略自蒸馏（OPSD）能够提供密集的教师目标，但仅在学生采样的轨迹上对这些目标进行评估。当拥有特权信息的教师在已访问的前缀处偏好另一种动作时，OPSD可以为该分支决策提供目标，却无法监督由该动作所引发的后继上下文，除非学生自己采样到它。这造成了训练阶段的数据收集瓶颈，并提示师生分歧应当扮演不同的角色：提出轨迹分支，而非识别充分的局部修复。我们基于受控token干预的诊断框架揭示：在分歧峰值处的教师偏好token能够提升学生后续延续的成功率，而其局部纠正价值有限。受这一发现启发，我们提出了一种简单的“分支—重生成—蒸馏”算法——轨迹干预自蒸馏（TISD）。TISD强制执行教师选择的分支动作，并返回后继……（摘要原文在此处截断）

    arXiv:2609.30878v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) provides dense teacher targets, but evaluates them only along student-sampled rollouts. When the privileged teacher favors an alternative action at a visited prefix, OPSD can provide a target for the branch decision but cannot supervise the successor contexts induced by that action unless the student samples it. This creates a training-time data-collection bottleneck and suggests a different role for teacher-student disagreement: proposing a trajectory branch rather than identifying a sufficient local repair. Our diagnostic framework using controlled token interventions reveals that a teacher-preferred token at peak disagreement can improve student continuation success, while its local corrective value is limited. Motivated by this finding, we introduce a simple branch-regenerate-distill algorithm, Trajectory-Intervention Self-Distillation (TISD). TISD forces a teacher-selected branch action, returns su
    
[^114]: 面向对抗性黑盒在线策略蒸馏的持久负样本方法

    Persistent Negatives for Adversarial Black-Box On-Policy Distillation

    [https://arxiv.org/abs/2609.30864](https://arxiv.org/abs/2609.30864)

    提出持久负样本对抗蒸馏方法，通过活跃池机制用历史教师-学生对比作为稳定负样本，解决了对抗蒸馏中负样本分布随策略更新而漂移的移动目标问题，从而提升黑盒在线策略蒸馏的稳定性。

    

    黑盒在线策略蒸馏（OPD）旨在当教师模型只提供采样响应而不提供token概率时，从学生模型自身的生成结果中对其进行改进。对抗蒸馏提供了一条可行路径：它通过学习一个判别器来区分提示匹配的教师响应和学生响应，并将判别器的评分用作策略奖励。然而，在每一步都从最新的学生模型中采样判别器负样本，会使学到的奖励与每次策略更新后都会发生变化的负样本分布耦合在一起。我们通过持久负样本对抗蒸馏来解决这一移动目标问题，这是一种活跃池方法，用历史积累的、提示匹配的教师-学生对比来替换每个判别器批次中的一部分数据。在判别器计算量匹配的情况下，历史对比用于训练判别器，而GRPO则基于新鲜的学生响应保持在线策略特性。我们的分析确定了贝叶斯最优奖励的形式为教师样本相对于负样本的对数密度比。

    arXiv:2609.30864v1 Announce Type: cross  Abstract: Black-box On-Policy Distillation (OPD) seeks to improve a student from its own generations when the teacher provides sampled responses but not token probabilities. Adversarial distillation offers one route: it learns a discriminator over prompt-matched teacher and student responses and uses its score as the policy reward. However, sampling discriminator negatives from the latest student at each step couples the learned reward to a negative distribution that changes after every policy update. We address this moving-target problem with persistent-negative adversarial distillation, a live-pool method that replaces a fraction of each discriminator batch with historical, prompt-matched teacher--student comparisons. Under matched discriminator compute, historical comparisons train the discriminator, while GRPO remains on-policy with fresh student responses. Our analysis identifies the Bayes-optimal reward as a teacher-to-negative log-density
    
[^115]: 制定AI优先组织路线图：嵌入式软件开发案例研究

    Developing a Roadmap to an AI-first Organization: A Case Study in Embedded Software Development

    [https://arxiv.org/abs/2609.30863](https://arxiv.org/abs/2609.30863)

    本文通过对一家大型嵌入式系统公司40名从业者的研讨会数据进行混合方法分析，研究了该公司向AI优先组织转型的路线图，发现智能体AI预计将深刻影响团队结构、所需能力、组织战略及开发人员角色。

    

    AI智能体的出现预计将通过超越AI作为助手的角色，转向能够以日益增强的自主性来规划、执行和评估开发任务的系统，从而重塑软件工程。这一转变对于嵌入式软件组织尤为重要，因为这类组织通常对质量、可追溯性、验证和长期可维护性有着严格的要求。本文对一家大型嵌入式系统公司向AI优先组织转型的过程进行了案例研究。通过混合研究方法，我们分析了从一场有40名参与者（包括Scrum主管、架构师、管理层和产品负责人）参加的半结构化研讨会中收集的数据。研究结果表明，参与者预计智能体AI将影响团队结构、所需能力、组织战略以及开发人员在组织中的角色。基于这些发现，本文讨论了……（摘要原文在此处截断）

    arXiv:2609.30863v1 Announce Type: cross  Abstract: The emergence of AI agents is expected to reshape software engineering by moving beyond AI as assistants towards systems capable of planning, executing, and evaluating development tasks with increasing autonomy. This transition is particularly significant for embedded software organizations, where strict requirements for quality, traceability, verification, and long-term maintainability often apply. This paper presents a case study of a large embedded systems company and its transition toward becoming an AI-first organization. Through a mixed method, we analyzed data collected from a semi-structured workshop with 40 participants, including scrum masters, architects, management, and product owners. The findings show that the participants expect agentic AI to affect team structure, required competencies, organizational strategies, and developers' roles within the organization. Based on these findings, the paper discusses implications for
    
[^116]: SkillEvoReg：通过正则化防止智能体技能演化中的过拟合

    SkillEvoReg: Regularizing Agent Skill Evolution Against Overfitting

    [https://arxiv.org/abs/2609.30861](https://arxiv.org/abs/2609.30861)

    提出 SkillEvoReg 框架，借鉴神经网络训练中的抗过拟合技术（训练时技能丢弃、复杂度感知正则化、因果反例验证），有效防止语言模型智能体技能演化中的过拟合问题。

    

    语言模型智能体越来越多地通过将执行经验转化为可复用的外部技能来提升自身性能。然而，反复的技能更新本身也构成一个学习过程：局部有用的修改可能累积成冗余或任务特定的指令，而新的更新也可能破坏先前有效的行为。我们将该问题界定为技能演化过拟合，并提出 SkillEvoReg，一个受神经网络训练中抗过拟合技术启发的技能演化通用正则化框架。SkillEvoReg 结合了训练时技能丢弃（用于扰动更新生成）、复杂度感知的局部正则化（用于控制不必要的结构增长），以及因果反例验证（CCV，用于对候选技能引发的特定退化进行有针对性的行为验证）。我们在多个异构的技能演化系统上实例化了该框架，同时保留各系统原生的技能演化器。

    arXiv:2609.30861v1 Announce Type: new  Abstract: Language-model agents increasingly improve by converting execution experience into reusable external skills. Yet repeated skill updates form a learning process of their own: locally useful edits can accumulate into redundant or task-specific instructions, while new updates can disrupt behavior that previously worked. We study this problem as skill-evolution overfitting and introduce SkillEvoReg, a general regularization framework for skill evolution inspired by anti-overfitting techniques in neural-network training. SkillEvoReg combines training-time skill dropout, which perturbs update generation, and complexity-aware local regularization, which controls unnecessary structural growth, with causal counterexample validation (CCV), which provides targeted behavioral validation of candidate-specific regressions. We instantiate the framework across heterogeneous skill-evolution systems while retaining each system's native skill evolver and t
    
[^117]: 为什么越狱攻击在扩散语言模型中成功：一种能量景观分析

    Why Jailbreaks Succeed in Diffusion Language Models: An Energy Landscape Analysis

    [https://arxiv.org/abs/2609.30841](https://arxiv.org/abs/2609.30841)

    该论文将扩散语言模型的安全对齐解释为去噪能量景观中的一道能量屏障，据此将现有越狱攻击归纳为两类绕过策略，并由此推导出三种无需训练的检测信号（step-0 比率和两个轨迹速度信号）。

    

    现有针对基于扩散的大语言模型（dLLMs）的攻击和防御方法都只针对特定漏洞，但缺乏一个共同的框架来解释攻击为何能够成功。我们提出了这样一个框架：将安全对齐解释为对去噪能量景观的塑造——一个对齐良好的模型通过一道能量屏障将有害查询引向安全输出，这道屏障分隔了安全区域与有害区域。当前的越狱攻击可归结为绕过这道屏障的两种策略：一是在初始化时模糊查询的安全倾向，二是在去噪轨迹中途进行干预，迫使去噪路径跨越能量屏障。基于这一视角，以及掩码扩散模型在去噪过程中最小化动能的结论，我们推导出三种互补的、无需训练的检测信号：一个 step-0 比率，可在生成开始前从 logit 分布中读取查询的初始安全倾向；以及两个轨迹速度信号，用于跟踪动能（原文在此处截断）。

    arXiv:2609.30841v1 Announce Type: new  Abstract: Existing attacks and defenses for diffusion-based large language models (dLLMs) target specific vulnerabilities but lack a shared framework explaining why attacks succeed. We propose one by interpreting safety alignment as shaping the denoising energy landscape: a well-aligned model routes harmful queries toward safe outputs through an energy barrier that separates the two regions. Current jailbreak attacks reduce to two strategies for circumventing this barrier: obscuring the query's safety disposition at initialisation, or intervening mid-trajectory to force the denoising path across the energy barrier. From this perspective and the result that masked diffusion models minimise kinetic energy during denoising, we derive three complementary, training-free detection signals: a step-0 ratio that reads the initial safety disposition from the logit distribution before generation begins, and two trajectory-velocity signals that track kinetic 
    
[^118]: MOPD-Router：重新思考多教师在策略蒸馏中的教师路由

    MOPD-Router: Rethinking Teacher Routing in Multi-Teacher On-Policy Distillation

    [https://arxiv.org/abs/2609.30837](https://arxiv.org/abs/2609.30837)

    提出MOPD-Router框架，无需领域标签即可在token级别对完整教师池进行监督路由，并通过ExpertAlign依据教师后训练所获专业化能力为其修正信号评分，从而充分释放多教师互补知识。

    

    多教师在策略蒸馏（MOPD）旨在将多个专业化能力整合到单个学生模型中，但现有做法通常将每个提示硬路由到与其领域匹配的教师，并让其完成整个生成过程。这种对提示级领域标签的依赖限制了无标签训练混合数据的使用，也使其他教师的互补信号未被利用。我们提出MOPD-Router，一个无需领域标签、也无需训练额外路由模型的框架，它在每个token上对完整教师池进行监督路由。其插件式接口支持多种度量标准来选择和加权来自不同教师的OPD信号。在该接口内，我们提出ExpertAlign，它通过判断教师对学生当前token的修正是否体现了该教师在后训练阶段所获得的专业化能力来为每个教师评分，并将其与基于教师置信度和师生判别性的两种参考度量进行比较。

    arXiv:2609.30837v1 Announce Type: cross  Abstract: Multi-teacher on-policy distillation (MOPD) integrates specialized capabilities into a single student, but existing practice typically hard-routes each prompt to a domain-matched teacher for the entire rollout. This dependence on prompt-level domain labels restricts using unlabeled training mixtures and leaves complementary signals from other teachers unused. We introduce MOPD-Router, a framework that routes supervision over the full teacher pool at each token, without domain labels or training a separate routing model. Its plug-in interface supports different metrics for selecting and weighting teacher-specific OPD signals. Within this interface, we propose ExpertAlign, which scores each teacher by whether its correction to the student at the current token expresses the specialization that teacher acquired during post-training, and compare it against two reference metrics built on teacher confidence (Entropy) and teacher-student discr
    
[^119]: PTC-Decoder：面向离线资源受限边缘设备的智能小语言模型

    PTC-Decoder: Towards Intelligent SLMs on Offline Resource-Constrained Edge Devices

    [https://arxiv.org/abs/2609.30836](https://arxiv.org/abs/2609.30836)

    提出无需训练、即插即用的PTC-Decoder解码器框架，通过强制规划调用和工具名称的token级硬约束，使小语言模型能够在离线资源受限的边缘设备上可靠执行多步智能体任务。

    

    将小语言模型（SLM）部署在离线、资源受限的边缘设备（如遥感卫星）上存在一个根本性挑战：其有限的推理能力阻碍了需要复杂工具编排的多步智能体任务的可靠执行。现有的“规划-求解”范式依赖于基于提示词的强制约束，而我们的实验表明SLM几乎完全无视这种约束：弱模型无法调用规划。我们提出了PTC-Decoder（规划-工具约束解码器），这是一个无需训练、即插即用的解码器框架，它结合了两项技术：（1）Plan-to-Act范式，将规划提升为一个原子工具，并强制其在第一步推理时被调用；（2）TC-Decoder，一种确定性有限自动机，在保留参数生成自由度的同时，对工具名称施加token级别的硬约束，从而保留SLM的推理能力。该框架在200个真实遥感卫星任务上对7个小语言模型进行了评估。

    arXiv:2609.30836v1 Announce Type: new  Abstract: Deploying small language models (SLMs) on offline, resource-constrained edge devices such as remote sensing satellites presents a fundamental challenge: their limited reasoning capacity hinders reliable execution of multi-step agent tasks requiring complex tool orchestration. Existing plan-solve paradigms rely on prompt-based enforcement, which our experiments show SLMs almost entirely disregard: weak models fail to invoke the plan. We propose PTC-Decoder (Plan-Tool Constrained Decoder), a training-free, plug-and-play decoder framework that combines (1) a Plan-to-Act paradigm, which elevates planning to an atomic tool and forces its invocation at the first inference step, and (2) TC-Decoder, a deterministic finite automaton that imposes token-level hard constraints on tool names while preserving freedom over parameter generation, thereby retaining SLM reasoning capability. Evaluated on 200 real remote-sensing satellite tasks across 7 SLM
    
[^120]: 基于脑记录的感知语音的受试者不变跨模态解码

    Subject-Invariant Cross-Modal Decoding of Perceived Speech from Brain Recordings

    [https://arxiv.org/abs/2609.30832](https://arxiv.org/abs/2609.30832)

    提出了一种整合fMRI与MEG的受试者不变跨模态感知语音解码方法（SICMD），首次以统一框架同时解决了神经表征提取与跨受试者泛化两大挑战，并显著提升了解码性能。

    

    基于非侵入式脑机接口（BCI）信号的感知语音解码近年来得到了广泛研究。该领域的研究主要面临两个挑战：提取具有丰富时空信息的神经表征，以及实现跨受试者泛化。尽管已有研究分别提出了应对这些问题的方法，但目前仍缺乏能够同时解决这两个挑战的统一方法。为填补这一空白，我们提出了受试者不变跨模态感知语音解码（SICMD）方法，该方法整合了功能磁共振成像（fMRI）和脑磁图（MEG）。我们对融合方法、融合位置、编码器架构和模型输入进行了全面分析。结果表明，与基线方法相比，所提出的方法在跨受试者设置下将Top-1、Top-10和Rankacc指标分别提升了10.6%、10.1%和1.7%以上。

    arXiv:2609.30832v1 Announce Type: cross  Abstract: Perceived speech decoding based on non-invasive brain-computer interface (BCI) signals has been extensively studied in recent years. Research in this field primarily faces two challenges: extracting neural representations with rich spatiotemporal information and achieving cross-subject generalization. Although separate studies have proposed methods to cope with these issues, a unified approach that simultaneously tackles both challenges remains lacking. To fill this gap, we propose the Subject-Invariant Cross-Modal Perceived Speech Decoding (SICMD) method, which integrates functional magnetic resonance imaging (fMRI) and magnetoencephalography (MEG). We conduct comprehensive analyses of the fusion method, fusion position, encoder architecture, and model inputs. Our results demonstrate that the proposed method improves Top-1, Top-10, and Rankacc by more than 10.6%, 10.1%, and 1.7%, respectively, compared to baseline methods in cross-sub
    
[^121]: 评估即多模态自动驾驶所需的一切

    Evaluation Is All You Need for Multi-Modal Autonomous Driving

    [https://arxiv.org/abs/2609.30818](https://arxiv.org/abs/2609.30818)

    该论文揭示了多模态自动驾驶规划中“生成强而评估弱”的不对称问题，并提出iDriveVLA框架，通过统一轨迹评估器（安全感知评分器与VLM引导调制器）实现更可靠、场景自适应的候选轨迹选择，充分释放多模态规划潜力。

    

    多模态规划通过在模糊和长尾场景中表达多种合理行为，为自动驾驶带来了广阔前景。现有方法主要致力于提升轨迹的多模态性、增强轨迹表征或重塑候选分布。然而，我们发现了多模态规划中存在显著的“生成-评估不对称”问题：尽管现有规划器在oracle（最优参照）性能上表现强劲，但它们往往无法可靠地选出最佳可用候选轨迹，导致大量规划潜力未能实现。为应对这一挑战，我们提出了iDriveVLA——一个多模态规划框架，它在改善候选轨迹空间的同时，能够实现更可靠、更具场景感知能力的轨迹评估。具体而言，iDriveVLA引入了一个统一的轨迹评估器，其中包含用于质量与风险估计的安全感知评分器，以及用于场景自适应准则加权的VLM引导调制器。

    arXiv:2609.30818v1 Announce Type: cross  Abstract: Multi-modal planning is promising for autonomous driving by representing multiple plausible behaviors in ambiguous and long-tail scenarios. Existing methods mainly focus on improving trajectory multi-modality, enhancing trajectory representations, or reshaping the candidate distribution. Nevertheless, we identify a pronounced generation-evaluation asymmetry in multi-modal planning: despite strong oracle performance, existing planners often fail to reliably select the best available candidate, leaving substantial planning potential unrealized. To address this challenge, we propose iDriveVLA, a multi-modal planning framework that improves the candidate trajectory space while enabling more reliable and context-aware trajectory evaluation. Specifically, iDriveVLA introduces a unified trajectory evaluator comprising a Safety-aware Scorer for quality and risk estimation, together with a VLM-guided Modulator for scene-adaptive criterion weigh
    
[^122]: 共享智能体记忆中认知准入的基准与诊断研究

    A Benchmark and Diagnostic Study of Epistemic Admission in Shared Agent Memory

    [https://arxiv.org/abs/2609.30813](https://arxiv.org/abs/2609.30813)

    论文提出了相关提升基准（CPB），通过静态与动态两种模式评估共享智能体记忆中的主张准入策略，发现基于来源去重的策略会大量误拒真实主张，而保留答案覆盖率的策略几乎与无限制准入一样容易接纳错误主张。

    

    评估共享智能体记忆中的主张准入极具挑战性，因为重复出现的主张可能被误认为是独立的证据。一个智能体可能会复制或改写先前检索到的信念，而一旦错误的主张被准入共享记忆，后续的智能体都会暴露于该错误之中。为了研究这一问题，我们提出了相关提升基准，用于评估候选主张是否应被准入共享记忆。CPB-Static 从公开标注的数据源构建了固定的测试集划分，并配有确定的黄金动作标签；CPB-Live 则让多智能体团队在共享存储上运行，记录所有写入与检索操作，并跟踪每个场景定义的来源谱系。此外，由一个独立的消费者智能体仅依据存储内容进行回答。我们在四个智能体系列上评估了八种准入策略。结果表明：对来源进行去重的策略在拒绝错误主张的同时也会拒绝大量真实主张，而保留答案覆盖率的策略所准入的错误主张数量几乎与完全不设限制的情况相当。

    arXiv:2609.30813v1 Announce Type: new  Abstract: Evaluating claim admission in shared agent memory is challenging because repeated claims may be mistaken for independent evidence. An agent may copy or paraphrase a retrieved belief, while admitting a false claim exposes subsequent agents to it. To study this problem, we introduce the Correlated Promotion Benchmark (CPB), which evaluates whether candidate claims should be admitted to shared memory.CPB-Static constructs a frozen test split from publicly annotated sources with fixed gold actions. CPB-Live runs multi-agent teams over a shared store, records all writes and retrievals, and tracks source lineage defined by each scenario. A separate consumer answers from the store alone. We evaluate eight admission policies across four agent families. Our results show that policies which deduplicate sources reject many true claims alongside false ones, whereas policies preserving answer coverage admit nearly as many false claims as unrestricted
    
[^123]: XPhysICS：面向工业控制系统安全的跨物理域威胁落地方法

    XPhysICS: Cross-Physical-Domain Threat Grounding for Industrial Control Systems Security

    [https://arxiv.org/abs/2609.30805](https://arxiv.org/abs/2609.30805)

    XPhysICS提出了一种来源感知、目标条件化的跨物理域威胁落地方法，通过角色兼容性等五项资格标准，将某一工控系统记录的威胁确定性映射到目标系统的验证切片上，并以相互独立的证据层判断威胁的信息物理效应在目标系统上是否可容许、可评估。

    

    工业控制系统（ICS）中针对某一工厂记录的威胁，可能在另一个系统中表现出与之相关的信息物理效应，但仅凭语义相似性并不能确定这些效应在目标系统上是否结构上可容许或可评估。我们提出了XPhysICS，这是一种具备来源感知、以目标为条件的方法，它将分析人员引导的源抽象与确定性的落地过程相分离，落地结果为针对目标系统的验证切片。在给定固定的源抽象、词汇表与模式，以及经过机器验证的目标契约的情况下，XPhysICS使用五项资格标准评估候选映射：角色兼容性、已实现类型兼容性、阶段一致性、切片可行性和规则表面适用性。落地接受度、切片充分性、动态可实现性、消费者适用性和消费者结果保持为相互独立的证据层。我们在水处理、水分配等领域对83个结构化源威胁抽象进行了评估……（原文摘要到此截断）

    arXiv:2609.30805v1 Announce Type: cross  Abstract: Industrial control system (ICS) threats documented for one plant can express cyber-physical effects relevant to another, but semantic similarity alone does not establish whether those effects are structurally admissible or evaluable on a target. We present XPhysICS, a provenance-aware, target-conditioned method that separates analyst-guided source abstraction from deterministic grounding into target-specific validation slices. Given a fixed source abstraction, vocabulary and schema, and machine-validated target contract, XPhysICS evaluates candidate mappings using five eligibility criteria: role compatibility, implemented type compatibility, stage coherence, slice viability, and rule-surface applicability. Grounding acceptance, slice adequacy, dynamic realizability, consumer applicability, and consumer outcome remain distinct evidence layers. We evaluate 83 structured source-threat abstractions across water treatment, water distributio
    
[^124]: 评估实时语音智能体：从组件质量到实际落地成效

    Evaluating Real-Time Voice Agents: From Component Quality to Grounded Outcomes

    [https://arxiv.org/abs/2609.30798](https://arxiv.org/abs/2609.30798)

    该论文指出现有实时语音智能体评估文献分散于语音基础建模、话轮转换心理语言学和智能体评估三个互不引用的领域，并基于38篇一手文献提出三项循证论点，主张超越单一组件指标、建立面向实际部署成效的统一评估框架。

    

    实时语音智能体已从研究原型走向生产部署，然而描述它们的文献分散在三个鲜少互相引用的研究社区中：语音基础建模、话轮转换心理语言学以及智能体评估。架构论文报告延迟，话轮转换论文报告预测准确率，智能体基准测试报告任务成功率，因此没有任何单一指标能够描述一个已部署的智能体是否真正优秀。我们通过三个基于证据的论点来填补这一空白，每个论点都可追溯到一个由38篇一手文献组成的语料库，这些文献被组织成以应用为中心的六类分类体系。第一，架构选择是一种部署约束而非已有定论：一篇2026年的企业教程指出，目前尚无完全可自托管的端到端系统能满足生产约束，而一种分块级联（chunked cascade）架构独立达到了最先进的双工行为，表明双工行为是……

    arXiv:2609.30798v1 Announce Type: new  Abstract: Real-time voice agents have moved from research prototypes to production deployments, yet the literature describing them is fragmented across three communities that rarely cite one another: speech foundation modelling, turn-taking psycholinguistics, and agentic evaluation. Architecture papers report latency, turn-taking papers report prediction accuracy, and agentic benchmarks report task success, so no single number describes whether a deployed agent is actually good. We address that gap with three evidence-based claims, each traceable to a corpus of 38 primary sources organised into an application-centric taxonomy of six categories. First, architecture choice is a deployment constraint rather than a settled verdict: a 2026 enterprise tutorial reports that no fully self-hostable end-to-end system yet meets production constraints, while a chunked cascade independently reaches state-of-the-art duplex behaviour, showing duplex behaviour is
    
[^125]: HasMem：面向长期大语言模型智能体的硬起源自适应软化记忆

    HasMem: Hard-Origin Adaptively Softened Memory for Long-Term LLM Agents

    [https://arxiv.org/abs/2609.30797](https://arxiv.org/abs/2609.30797)

    HasMem以冻结的硬提示词嵌入作为可验证初始状态，通过控制器自适应调节记忆宽度、由Writer重编码、Reader与Global实现读取适应与跨轮状态，在MSC重建探针上以更少的记忆位置取得更高的F1，并大幅超越基于规则的重新编码方法。

    

    基于文本的记忆与上下文压缩技术支持对过去交互内容的复用。然而，对连续记忆进行尺寸调整会改变冻结大语言模型的输入，使容量分配与读取过程耦合在一起。我们提出了硬起源自适应软化记忆。冻结的硬提示词嵌入提供了可验证的初始状态；一个控制器负责调节记忆条目的宽度，一个Writer对调整尺寸后的条目进行重新编码，而Reader与Global模块则提供读取适应能力和跨轮次状态。在一个由多会话对话（MSC）开发集衍生的重建探针的全部535个问题上，主配置在仅使用硬参考93.6%的框架记忆位置的情况下，取得了95.3的词法F1分数（提升4.4个百分点）。在每问题目标正文预算大致匹配的条件下，平均每条目保留率约为0.83至0.91的六种配置，其精确匹配（EM）分数超过基于规则的重新编码方法8.0至23.6个百分点。在固定模型参数与……

    arXiv:2609.30797v1 Announce Type: new  Abstract: Text-based memory and context compression support reuse of past interactions. Resizing continuous memory changes the input to a frozen LLM, coupling capacity allocation with readout. We propose Hard-Origin Adaptively Softened Memory (HasMem). Frozen hard-prompt embeddings provide a verifiable initial state. A controller adjusts memory widths, a Writer re-encodes resized entries, and Reader and Global provide readout adaptation and cross-turn state. On all $535$ questions in a reconstruction probe derived from the Multi-Session Chat (MSC) development split, the main configuration achieves lexical F1 of $95.3$ ($+4.4$ percentage points) at $93.6\%$ of the hard reference's framed memory positions. With approximately matched per-question target body budgets, six configurations at mean per-entry retention around $0.83$--$0.91$ exceed rule-based re-encoding by $8.0$--$23.6$ exact-match (EM) percentage points. With fixed model parameters and ru
    
[^126]: ConsultMind：基于不确定性感知推理的自动化诊断问诊

    ConsultMind:Towards Automated Diagnostic Consultation via Uncertainty-Aware Reasoning

    [https://arxiv.org/abs/2609.30796](https://arxiv.org/abs/2609.30796)

    该论文提出AutoDisym流水线自动构建疾病-症状贝叶斯网络（DSBN），并在此基础上提出不确定性感知框架ConsultMind，通过在每次患者回应后更新疾病后验概率，并利用后验不确定性来指导问诊提问与诊断决策，从而实现自动化诊断问诊。

    

    诊断问诊是一个在线序贯决策过程，临床医生通过与患者的互动收集证据，直至诊断获得充分的支持。实现这一过程的自动化需要自适应的提问和可解释的决策。贝叶斯网络为此提供了天然的基础——随着证据的积累不断更新诊断后验概率，但将其应用于开放式问诊面临两大挑战：如何将诊断假设与潜在的提问相关联，以及如何将不断演变的后验概率转化为问诊决策。我们提出了AutoDisym，一个将诊断知识与带有诊断标签的异构临床叙事相结合，从而构建疾病-症状贝叶斯网络（DSBN）的自动化流水线。在DSBN的基础上，我们提出了ConsultMind，一个不确定性感知框架，它在每次获得患者回应后更新疾病后验概率，并利用后验不确定性来指导提问和诊断。我们对……

    arXiv:2609.30796v1 Announce Type: new  Abstract: Diagnostic consultation is an online sequential decision-making process in which clinicians gather evidence through patient interaction until a diagnosis is sufficiently supported. Automating this process requires adaptive inquiry and interpretable decisions. Bayesian networks offer a natural foundation by updating diagnostic posteriors as evidence accumulates, but their use in open-ended consultation raises two challenges: linking diagnostic hypotheses to potential inquiries and translating evolving posteriors into consultation decisions. We introduce AutoDisym, an automated pipeline that integrates diagnostic knowledge with heterogeneous diagnosis-labeled clinical narratives to construct a Disorder--Symptom Bayesian Network (DSBN). Building on the DSBN, we propose ConsultMind, an uncertainty-aware framework that updates disorder posteriors after each response and uses posterior uncertainty to guide inquiry and diagnosis. We evaluate bo
    
[^127]: 省却言语，重聚视觉：多模态大语言模型中推理分割的潜空间推理

    Skip the Talk, Re-Focus on Vision: Latent Reasoning for Reasoning Segmentation in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.30783](https://arxiv.org/abs/2609.30783)

    提出LIRSeg方法，用紧凑的可学习潜在标记完全替代显式思维链推理，消除冗余文本标记对视觉注意力的干扰，从而提升多模态大语言模型推理分割的性能。

    

    推理分割旨在解释隐含的文本查询并实现细粒度的视觉感知，这对于人机交互和具身智能体等应用至关重要。现有方法通常先由多模态大语言模型（MLLMs）生成显式的思维链（CoT），然后再定位目标。尽管直观，但这种显式的言语推理会引入显著的注意力干扰：冗余的文本标记在感知标记生成过程中扰乱注意力，同时增大了视觉标记之间的有效距离。为了解决这一问题，我们提出了LIRSeg，它用一组紧凑的可学习潜在标记完全替代显式CoT来执行推理分割。LIRSeg采用两阶段训练：空间对齐阶段使潜在标记扎根于与目标相关的视觉证据中，随后GRPO通过分割奖励进一步优化这些标记。

    arXiv:2609.30783v1 Announce Type: cross  Abstract: Reasoning segmentation aims to interpret implicit textual queries and enable fine-grained visual perception, which is critical for applications such as human-computer interaction and embodied agents. Existing methods typically generate explicit Chain-of-Thought (CoT) by multimodal large language models (MLLMs) before localizing the target. Although intuitive, such explicit verbal reasoning introduces substantial attention interference: redundant textual tokens disrupt attention during perception-token generation and also increase the effective distance between visual tokens. To address this issue, we propose LIRSeg, which fully replaces explicit CoT with a compact set of learnable latent tokens for reasoning segmentation. LIRSeg is trained in two stages: spatial alignment grounds the latent tokens in object-relevant visual evidence, and GRPO further optimizes them with segmentation rewards. To make these compact latent tokens more info
    
[^128]: NavGen：将视觉生成模型作为具身三维导航的可扩展数据引擎

    NavGen: Visual Generative Models as a Scalable Data Engine for Embodied 3D Navigation

    [https://arxiv.org/abs/2609.30770](https://arxiv.org/abs/2609.30770)

    NavGen利用高保真视觉生成模型构建文本到视频数据生成流水线，生成约40万个覆盖室内外场景的视觉-语言导航片段，并通过风格多样化方法扩展难以采集的长尾数据，为具身三维导航提供了可扩展的数据引擎，有效缓解了仿真数据与真实数据之间的权衡问题。

    

    通用机器人模型越来越依赖于大规模且多样化的数据集。然而，对于具身三维导航而言，现有数据源面临一个根本性的权衡：仿真数据可以大规模生成，但往往存在视觉上从仿真到现实的差距，而真实世界的飞行数据虽然能提供真实的观测，但采集成本高昂。本文研究了另一个方向：将高保真视觉生成模型用作具身三维导航的可扩展数据引擎。我们提出了NavGen，这是一个文本到视频的数据生成流水线，能够在室内和室外场景中生成多样化的视觉-语言导航（VLN）片段。我们还提出了一种风格多样化方法，用于扩展难以采集且成本高昂的长尾数据。所生成的数据集包含约40万个导航片段。我们在多个指标上将该数据集与现有的无人机导航数据集进行了对比评估，发现该模型……

    arXiv:2609.30770v1 Announce Type: cross  Abstract: General-purpose robot models increasingly rely on large and diverse datasets. For embodied 3D navigation, however, existing data sources face a fundamental trade-off: simulated data can be generated at scale but often suffer from the visual sim-to-real gap, whereas real-world flight data provide realistic observations but are costly to collect. This paper studies another direction: the use of high-fidelity visual generative models as scalable data engines for embodied 3D navigation. We introduce NavGen, a text-to-video data generation pipeline that produces diverse vision-language navigation (VLN) episodes across indoor and outdoor scenes. We also propose a style-diversification method that scales up long-tail data that are difficult and costly to collect. The resulting dataset contains approximately 400K navigation episodes. We evaluate our dataset against existing UAV navigation datasets across multiple metrics, and find that the mod
    
[^129]: 思考有助于公平吗？推理Token解决了一些偏见，却制造了更多偏见

    Does Thinking Help Fairness? Reasoning Tokens Resolve Some Biases but Create More

    [https://arxiv.org/abs/2609.30768](https://arxiv.org/abs/2609.30768)

    思考过程对反事实公平性具有非对称的双重效应——虽然能解决部分非思考状态下的偏见翻转，但制造的新偏见翻转数量约为解决数量的5倍。

    

    关于推理语言模型（RLMs）中的“思考”过程究竟是消解还是放大偏见，一直存在争议。先前的研究得出了两个相互矛盾方向的结论。我们通过在同一模型内进行“思考vs.非思考”的消融实验，覆盖QwQ-32B、DeepSeek-R1-Distill-Qwen-32B和Qwen3-32B三个模型，在三个高风险决策任务（Adult、COMPAS、Credit）上，展示了思考对反事实公平性具有非对称的双重效应：它既解决了非思考基线产生的反事实翻转，又在接近饱和的模型置信度下制造了新的翻转。在全部九种（模型，数据集）组合中，新制造的翻转数量大约是已解决翻转数量的5倍。为解释这一效应，我们将思考轨迹本身视为公平性变化的一个可测量场所，并通过两种动态分析工具对其进行研究：1）我们提出反事实深度概率差距（CDPG）来追踪思考深度中偏见的演变，并观察到偏见……（摘要至此被截断）

    arXiv:2609.30768v1 Announce Type: new  Abstract: Thinking in reasoning language models (RLMs) has been subject to debate on whether it resolves or amplifies bias. Prior works have shown competing conclusions in both directions. Using a within-model thinking-vs.-non-thinking ablation across QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, and Qwen3-32B on three high-stakes decision tasks (Adult, COMPAS, Credit), we show that thinking has an asymmetric dual effect on counterfactual fairness: it both resolves counterfactual flips produced by the non-thinking baseline and creates new flips at near-saturating model confidence. In all nine (model, dataset) combinations, the created flips outnumber the resolved flips by roughly 5 times. To explain the effect, we treat the thinking trace itself as a measurable site of fairness change and study it through two dynamic instruments: 1) We propose Counterfactual Depth Probability Gap (CDPG) to track bias evolution along thinking depth, and observe that bias 
    
[^130]: 保险准备金智能平台

    Insurance Reserve Intelligence Platform

    [https://arxiv.org/abs/2609.30765](https://arxiv.org/abs/2609.30765)

    本文提出了一个将经典Thiele微分方程求解器与知识信息增强的物理信息神经网络（KINN-PINN）相结合的保险准备金智能平台，在保持可解释性的同时大幅提升了定期寿险准备金计算的效率。

    

    保险准备金估算是支持保费定价、偿付能力评估、财务报告、资本规划和风险管理的基础精算任务。基于Thiele微分方程的经典准备金方法为人寿保险估值提供了严谨且可解释的基础，但在敏感性分析、优化和大规模情景评估中，重复的准备金计算在计算上成本高昂。本文提出了一个面向定期寿险准备金建模的保险准备金智能平台，该平台将经典的Thiele方程求解器与物理信息神经网络（PINN）相结合，并通过知识信息神经网络（KINN）损失加以增强。该框架包括合成保单生成、风险调整保费计算、经典准备金轨迹生成、准备金比率数据集构建、可配置的神经网络训练、验证诊断、敏感性及弹性分析。

    arXiv:2609.30765v1 Announce Type: new  Abstract: Insurance reserve estimation is a fundamental actuarial task supporting premium pricing, solvency assessment, financial reporting, capital planning, and risk management. Classical reserve methods based on Thiele's differential equation provide a rigorous and interpretable foundation for life insurance valuation, but repeated reserve calculations become computationally expensive in sensitivity analysis, optimization, and large-scale scenario evaluation.   This paper presents an Insurance Reserve Intelligence Platform for term-life reserve modelling that combines a classical Thiele-equation solver with a Physics-Informed Neural Network (PINN) enhanced by Knowledge-Informed Neural Network (KINN) losses. The framework includes synthetic policy generation, risk-adjusted premium calculation, classical reserve trajectory generation, reserve-ratio dataset construction, configurable neural training, validation diagnostics, sensitivity and elastic
    
[^131]: HCOE：基于生物医学语言模型的双曲临床本体嵌入

    HCOE: Hyperbolic Clinical Ontology Embeddings from Biomedical Language Models

    [https://arxiv.org/abs/2609.30763](https://arxiv.org/abs/2609.30763)

    HCOE 通过将冻结的 BioBERT 嵌入映射到双曲庞加莱空间，并结合本体引导的对比学习与由粗到细的路径聚合，构建了保留医学代码层级结构的临床概念表示，在临床关系预测及死亡率、再入院、药物推荐等多项临床任务上均达到最佳性能。

    

    生物医学语言模型（LM）能够编码文本语义，但无法显式地保留医学代码的层级结构。我们提出了双曲临床本体嵌入（HCOE），用于构建具有层级感知能力的临床概念表示。HCOE 将冻结的 BioBERT 嵌入映射到庞加莱球中，将父侧和子侧本体引导的对比学习与由粗到细的本体路径聚合相结合，并利用由临床分类软件（CCS）组织以及解剖学治疗化学（ATC）药物层级结构组织的国际疾病分类（ICD）代码。评估结果表明，HCOE 在 ICD/ATC 临床关系预测和 CCS 到 PheCode 的层级迁移任务上表现最佳。在 MIMIC-IV 数据集上，HCOE 在死亡率预测、再入院预测、药物推荐和罕见药物预测任务上也取得了最佳性能。

    arXiv:2609.30763v1 Announce Type: new  Abstract: Biomedical language models (LMs) encode textual semantics but do not explicitly preserve medical code hierarchies. We present Hyperbolic Clinical Ontology Embeddings (HCOE) for hierarchy-aware clinical concept representation. HCOE maps frozen BioBERT embeddings into a Poincare ball, combining parent-side and child-side ontology-guided contrastive learning with coarse-to-fine ontology-path aggregation. It uses International Classification of Diseases (ICD) codes organized by Clinical Classifications Software (CCS) and Anatomical Therapeutic Chemical (ATC) medication hierarchies. Evaluations show that HCOE performs best on ICD/ATC clinical relation prediction and CCS-to-PheCode hierarchy transfer. On the MIMIC-IV dataset, HCOE also achieves the best performance on mortality prediction, readmission prediction, medication recommendation, and rare drug prediction.
    
[^132]: 面向视觉令牌通信的全预算反事实推理选择性摊销方法

    Selective Amortization of Full-Budget Counterfactual Reasoning for Visual Token Communication

    [https://arxiv.org/abs/2609.30756](https://arxiv.org/abs/2609.30756)

    本文提出ACV-Gate自适应候选评估框架，通过学习近似全预算反事实评估，仅对最有信息量的候选令牌进行精确评估，在保证重建质量的同时大幅降低生成式图像通信中令牌选择的编码端计算开销。

    

    生成式图像通信在有限的数据包预算下传输紧凑的语义令牌，其中令牌的选择直接影响完整数据包解码后的最终重建质量。然而，准确估计每个候选令牌的终端价值需要反复进行接收端重建，导致编码端计算量巨大。为解决这一问题，我们提出了ACV-Gate，这是一种自适应候选评估框架，它学习近似全预算反事实评估，并选择性地将精确评估分配给信息量最大的候选者。具体而言，利用终端优势和遗憾值训练一个集合感知的学生模型，以直接预测候选排名；同时，选择性细化机制仅对包含Local-MDL和直接动作的有界候选集进行评估；基于成本的阈值进一步实现了对平均评估工作量的显式控制。

    arXiv:2609.30756v1 Announce Type: new  Abstract: Generative image communication transmits compact semantic tokens under a limited packet budget, where token selection directly affects the final reconstruction quality after the complete packet is decoded. However, accurately estimating the terminal value of every candidate token requires repeated receiver-side reconstruction, resulting in substantial encoder-side computation. To address this problem, we propose ACV-Gate, an adaptive candidate evaluation framework that learns to approximate full-budget counterfactual evaluation and selectively assigns exact evaluations to the most informative candidates. Specifically, a set-aware student is trained using terminal advantages and regrets to predict candidate rankings directly, while a selective refinement mechanism evaluates only a bounded candidate set containing both Local-MDL and direct actions; cost-based thresholds further enable explicit control of the average evaluation workload. Ex
    
[^133]: 面向鲁棒成对LLM评判的骨干自适应证据路由

    Backbone-Adaptive Evidence Routing for Robust Pairwise LLM Judging

    [https://arxiv.org/abs/2609.30751](https://arxiv.org/abs/2609.30751)

    提出骨干自适应证据路由方法BAER，在保持候选对称性的前提下为不同基准和评判骨干自适应选择证据收集机制，在全部八个测试条件下均取得最高准确率，较最强基线提升0.87至7.32个百分点。

    

    成对语言模型评判器可以通过直接比较、推理或基于参考的验证来收集证据，但没有单一的协议在所有基准和评判骨干上都是最优的。我们提出了骨干自适应证据路由，它在保持候选对称性的同时自适应调整证据机制：交换两个回答可能会反转偏好方向，但不会改变偏好的强度。BAER将每个专家的带符号偏好与候选不变的可靠性分离开来，并构建了三个对称的头：证据堆叠、基于可靠性的专家路由以及候选盲参考验证。开发数据为每个基准--骨干条件选择一个头，且该选择在测试前被冻结。在四个基准和两个8B评判骨干上，BAER在所有八个条件下的测试准确率均高于对比方法，实现了完整的预测覆盖率，并比最强基线提升了0.87至7.32个百分点。

    arXiv:2609.30751v1 Announce Type: new  Abstract: Pairwise language-model judges can gather evidence through direct comparison, reasoning, or reference-based verification, but no single protocol is best across benchmarks and judge backbones. We introduce Backbone-Adaptive Evidence Routing (BAER), which adapts the evidence mechanism while preserving candidate symmetry: swapping the two responses may reverse the preference but cannot change its strength. BAER separates each expert's signed preference from candidate-invariant reliability and builds three symmetric heads: evidence stacking, reliability-based expert routing, and candidate-blind reference verification. Development data select one head for each benchmark--backbone condition, and that choice is frozen before testing. Across four benchmarks and two 8B judge backbones, BAER achieves the highest test accuracy among the compared methods in all eight conditions, with full prediction coverage and gains of 0.87--7.32 points over the s
    
[^134]: ORCA：评估大语言模型在数据科学代码翻译上的能力

    ORCA: Evaluating LLMs on Data Science Code Translation

    [https://arxiv.org/abs/2609.30749](https://arxiv.org/abs/2609.30749)

    提出了ORCA综合基准，通过1,600个基础级任务和200个项目级翻译任务，首次系统评估了大语言模型在数据科学代码翻译（DSCT）这一未被充分研究领域的表现。

    

    数据科学代码翻译（DSCT）是指在保持功能等价性的同时，将代码在不同数据科学库之间进行转换，从而实现跨数据科学生态系统互操作性的过程。尽管大语言模型（LLMs）在数据科学代码生成（DSCG）方面已展现出显著进展，但其在数据科学代码翻译方面的表现仍未得到充分研究。为填补这一空白，我们提出了ORCA，一个包含两种互补设定的综合基准：ORCA-MAIN包含1,600个经过精心筛选的基础级任务，涵盖3个代表性领域——数据查询、数据操作和深度学习；ORCA-PROJECT包含200个针对完整数据科学项目的翻译任务，涵盖7种数据科学任务类型。每个任务均配有带注释的参考翻译以及用于验证功能等价性的测试用例。我们进一步引入了多阶段质量验证流程，以彻底……

    arXiv:2609.30749v1 Announce Type: new  Abstract: Data Science Code Translation (DSCT) is the process of converting code between data science libraries while preserving functional equivalence and enabling interoperability across data science ecosystems. While Large Language Models (LLMs) have demonstrated considerable progress in Data Science Code Generation (DSCG), their performance in DSCT remains insufficiently studied. To address this gap, we introduce ORCA, a comprehensive benchmark with two complementary settings: ORCA-MAIN, which comprises 1,600 carefully curated grounding-level tasks across 3 representative domains: Data Querying, Data Manipulation, and Deep Learning; and ORCA-PROJECT, which contains 200 translation tasks over complete data science projects across 7 data science task types. Each task is accompanied by annotated reference translations and test cases for validating functional equivalence. We further incorporate a multi-stage quality verification process that thoro
    
[^135]: 超越最后一棵特鲁法树：SustainAI——面向环境可问责人工智能的水资源感知闭环框架

    Beyond the Last Truffula Tree: SustainAI - A Water-Aware, Closed-Loop Framework for Environmentally Accountable AI

    [https://arxiv.org/abs/2609.30747](https://arxiv.org/abs/2609.30747)

    该论文提出SustainAI闭环框架，通过实时水量计量、幻觉感知惩罚模型和考虑区域水资源压力的路由算法，将水资源消耗纳入AI部署的环境问责体系，并揭示不同地区数据中心每次推理的水足迹差异高达11倍。

    

    随着人工智能（AI）融入日常生活，其环境足迹——尤其是水资源消耗——在很大程度上仍然不可见。尽管能源和碳排放影响已得到广泛认识，但数据中心冷却和电力生产对淡水的巨大需求却鲜少受到关注。为填补这一空白，我们提出了SustainAI，一个将环境可问责性纳入AI部署的水资源感知闭环框架。SustainAI集成了实时水量计量、幻觉感知惩罚模型，以及考虑区域水资源压力的水感知路由算法。通过小型语言模型（SLM）提取健康错误信息的任务进行评估，结果显示地理分布不同的数据中心之间水足迹存在11倍的差异（每次推理为0.0477毫升至0.5360毫升）。在1,335次推理运行中，系统共消耗约399毫升水，但仅产生了240个正确的（摘要在此处截断）。

    arXiv:2609.30747v1 Announce Type: cross  Abstract: As artificial intelligence (AI) becomes embedded in everyday life, its environmental footprint, particularly water consumption remains largely invisible. While energy and carbon impacts are widely recognized, the substantial freshwater demands of data center cooling and electricity generation receive little attention. To address this gap, we introduce SustainAI, a water-aware, closed-loop framework incorporating environmental accountability into AI deployment. SustainAI integrates real-time water metering, a hallucination-aware penalty model, and a water-aware routing algorithm that accounts for regional water stress. Evaluated via Small Language Models (SLMs) extracting health misinformation, results reveal an 11-fold variation in water footprint across geographically distributed data centers (0.0477 mL to 0.5360 mL per inference). Across 1,335 inference runs, the system consumed approximately 399 mL of water but produced only 240 cor
    
[^136]: 面向解剖结构的灵巧性驱动的手术连续体机器人设计优化

    Anatomy-Aware Dexterity-Driven Design Optimization of Surgical Continuum Robots

    [https://arxiv.org/abs/2609.30745](https://arxiv.org/abs/2609.30745)

    该论文提出一种兼顾灵巧性与解剖结构的手术连续体机器人设计优化方法，引入RVDSA指标，结合高效运动规划器和渐近最优的模拟退火优化器，成功优化了用于结肠息肉手术的双臂灵巧鞘管机器人设计。

    

    使用连续体机器人执行复杂的医疗手术需要仔细选择其几何设计参数，机器人应在其手术所处的特定解剖环境中具备高灵巧性。本工作提出了一种同时考虑灵巧性和解剖结构的设计优化方法。我们引入了可达体积灵巧立体角（RVDSA）指标作为优化目标，该指标衡量机器人末端执行器通过从起始构型出发的无碰撞路径从不同方向到达目标体积中各点的能力。我们提出了一种计算高效的运动规划器来计算给定机器人设计的该目标函数，并使用渐近最优的模拟退火优化器计算出优化后的设计。我们将新方法应用于优化双臂灵巧鞘管机器人的设计，用于在结肠解剖结构中对癌性息肉执行手术操作，实现了……

    arXiv:2609.30745v1 Announce Type: cross  Abstract: Performing complex medical procedures with continuum robots requires careful selection of their geometric design parameters. The robot should have high dexterity in the specific anatomical environment of its procedure. This work presents a design optimization method that considers both dexterity and anatomy. We introduce the Reachable Volumetric Dexterous Solid Angle (RVDSA) metric as our objective, which measures the ability of a robot's end effector to reach the points in a goal volume from different directions via collision-free paths from a start configuration. We present a computationally efficient motion planner to compute this objective function for a given robotic design, and we use an asymptotically optimal simulated annealing optimizer to compute an optimized design. We applied our new method to optimize the design of a bimanual dexterous sheaths robot for performing procedures on cancerous polyps in colon anatomies, achievin
    
[^137]: 从S3Q理论到实现：迈向机器感受质架构

    From S3Q Theory to Implementation: Towards an Architecture for Machine Qualia

    [https://arxiv.org/abs/2609.30743](https://arxiv.org/abs/2609.30743)

    本文为S3Q意识理论提出了一个五层实现架构，将三个意识必要条件（感知运动情境性、世界模型内部模拟、预测与观察的结构连贯性）映射到具体的计算机制并整合为单一表示流水线。

    

    机器意识研究中的一个关键挑战是将理论模型转化为计算层面的实现。本文通过为S3Q（模拟的、情境化的、结构连贯的）意识理论提出一个五层实现架构来应对这一挑战。该架构并未引入新颖的形式化方法，而是将已发表的计算原语组合成单一流水线。S3Q识别出感受质产生的三个联合必要条件：(1) 具身的感知运动情境性，(2) 通过世界模型进行内部模拟，以及(3) 预测与观察之间的结构连贯性。目前尚无现有计算系统同时实现这三者。我们将S3Q的每项原则映射到具体的、兼容的计算机制，并说明这些组件如何在单一表示流水线中相互衔接，该流水线基于连续、可微分的逐对象槽向量运行，并伴随一个开发……（原文摘要在此处截断）

    arXiv:2609.30743v1 Announce Type: new  Abstract: A key challenge in machine consciousness research is translating theoretical models into computational-level implementations. In this paper, we address this challenge by proposing a five-layer implementation architecture for the S3Q (Simulated, Situated, Structurally Coherent) theory of consciousness. Rather than introducing novel formalisms, the architecture composes published computational primitives into a single pipeline. S3Q identifies three jointly necessary conditions for qualia: (1) grounded sensorimotor situatedness, (2) internal simulation via a world model, and (3) structural coherence between predictions and observations. No existing computational system implements all three simultaneously. We map each S3Q tenet to specific, compatible computational machinery and specify how these components interface within a single representation pipeline that operates on continuous, differentiable, per-object slot vectors, along with a dev
    
[^138]: 学习跳过什么：面向高效多智能体LLM工作流的反事实信用分配

    Learning What to Skip: Counterfactual Credit Assignment for Efficient Multi-Agent LLM Workflows

    [https://arxiv.org/abs/2609.30734](https://arxiv.org/abs/2609.30734)

    提出LW2S框架，通过反事实信用分配学习各组件的跳过安全模型，在多智能体LLM工作流中智能省略不必要的步骤，从而在保持或提升任务性能的同时显著降低token开销。

    

    多智能体LLM工作流通过规划、执行、验证和总结来提升任务性能，然而每个组件的价值取决于已经产生的状态。执行每一个组件可能会浪费计算资源，或覆盖掉正确的中间答案。我们将组件省略问题形式化为反事实信用分配：完整工作流日志揭示了已执行轨迹的奖励，而受控的跳过干预则揭示了省略未来步骤的后果。我们提出了Learning What to Skip（LW2S）方法，它从这些干预中学习针对特定动作的安全模型，并结合留出校准与领域原生防护机制来选择跳过操作。当早期跳过被拒绝时，控制器可以继续执行并重新考虑后续组件。在使用两个指令模型家族的数学推理、多选题问答和代码生成任务上，LW2S在降低记录的token成本的同时，保持甚至提升了

    arXiv:2609.30734v1 Announce Type: new  Abstract: Multi-agent LLM workflows use planning, execution, verification, and summarization to improve task performance, yet the value of each component depends on the state already produced. Executing every component can waste computation or overwrite a correct intermediate answer. We formulate component omission as counterfactual credit assignment: full-workflow logs reveal the executed trajectory's reward, while controlled skip interventions reveal the consequences of omitting a future step. We introduce Learning What to Skip (LW2S), which learns action-specific safety models from these interventions and combines held-out calibration with domain-native guards to select skips. When an early skip is rejected, the controller can continue execution and reconsider a later component. Across mathematical reasoning, multiple-choice QA, and code generation with two instruction-model families, LW2S reduces recorded token cost while matching or improving
    
[^139]: 分析与缓解编码智能体中的成本低效行为

    Analyzing and Mitigating Cost-Inefficient Behaviors in Coding Agents

    [https://arxiv.org/abs/2609.30725](https://arxiv.org/abs/2609.30725)

    该论文首次系统研究编码智能体中的成本低效行为，识别出子集化检索、相似脚本生成与测试重复执行三种行为（影响79%–98%的任务、最高占任务成本的22.75%），并评估了结构感知检索、智能体自合成技能与开发者设计技能三种缓解策略的效果。

    

    arXiv:2609.30725v1 公告类型：新 摘要：编码智能体虽然效果显著，但往往会产生高昂的货币成本。其反复出现的成本低效行为至今仍缺乏充分研究。我们首次对编码智能体中的行为性成本低效开展了研究，分析了 Claude Code 和 Mini-SWE-Agent 在 SWE-bench Verified 上四种配置下的 1,200 条运行轨迹。我们识别出三种成本低效行为：子集化检索、相似脚本生成和测试重复执行。随后，我们在留出的 SWE-bench Verified 和 Pro 任务上基于 1 万条轨迹评估了三种缓解策略：结构感知检索、智能体自合成技能以及开发者设计技能。我们的主要发现包括：(1) 这三种行为影响 79.00%–98.00% 的编码任务，最高可占任务成本的 22.75%。(2) 结构感知检索可能引入检索开销并改变智能体的任务委派方式，导致检索效率提升不一致，且成本增加最高可达……（原文摘要至此截断）

    arXiv:2609.30725v1 Announce Type: new  Abstract: Although effective, coding agents often incur substantial monetary costs. Their recurring cost-inefficient behaviors remain underexplored. We conduct the first study of behavioral cost inefficiencies in coding agents, analyzing 1,200 trajectories from Claude Code and Mini-SWE-Agent across four configurations on SWE-bench Verified. We identify three cost-inefficient behaviors: subsumed retrieval, similar script generation, and test re-execution. We then evaluate three mitigation strategies: structure-aware retrieval, agent-synthesized skills, and developer-designed skills, over 10k trajectories on held-out SWE-bench Verified and Pro tasks. Our main findings are: (1) The three behaviors affect 79.00\%--98.00\% of coding tasks and account for up to 22.75\% of task cost. (2) Structure-aware retrieval can introduce retrieval overhead and alter agent delegation, causing inconsistent improvements in retrieval efficiency and cost increases of up
    
[^140]: TrafficImag：一个反事实路边交通视频生成基准

    TrafficImag: A Benchmark for Counterfactual Roadside Traffic Video Generation

    [https://arxiv.org/abs/2609.30722](https://arxiv.org/abs/2609.30722)

    本文提出首个反事实路边交通视频生成基准TrafficImag，通过大规模标注数据集与可执行干预协议，将干预表示为交通参与者级别的程序，实现对行为推理、图像编辑和条件视频生成等异构基础模型的统一评估。

    

    现有的路边交通数据集支持感知、预测和视觉问答任务，但它们并未评估反事实视频生成——即对选定的交通参与者进行修改后，所生成的未来视频应保持与道路拓扑及无关交通的一致性。我们提出了TrafficImag，这是首个面向反事实路边交通视频生成的基准。TrafficImag将一个大规模路边数据集（包含9,022张标注图像、7,043个去重视频片段以及31,145个以交通参与者为中心的历史-未来样本）与一个可执行协议相结合，该协议支持行为推理、干预感知的图像编辑以及条件视频生成。每次干预都被表示为一个交通参与者级别的程序，用于描述目标交通参与者、预期行为、合法路线、交互顺序和时间约束，从而在异构基础模型之间提供统一的评估接口。TrafficImag评估了四种（后文被截断）。

    arXiv:2609.30722v1 Announce Type: cross  Abstract: Existing roadside traffic datasets support perception, forecasting, and visual question answering, but they do not evaluate counterfactual video generation, in which a selected actor is modified and the generated future should remain consistent with road topology and unrelated traffic. We introduce TrafficImag, the first benchmark for counterfactual roadside traffic video generation. TrafficImag combines a large-scale roadside dataset (9,022 annotated images, 7,043 deduplicated video clips, and 31,145 actor-centered history-future samples) with an executable protocol that supports behavior reasoning, intervention-aware image editing, and conditional video generation. Each intervention is represented as an actor-level program describing the target actor, intended behavior, legal route, interaction order, and temporal constraints, enabling a unified evaluation interface across heterogeneous foundation models. TrafficImag evaluates four c
    
[^141]: Werracle：面向EVM智能合约的亚美分成本区块内AI反射预言机与闪电贷熔断器

    Werracle: Sub-Cent Intra-Block AI Reflex Oracles and Flash-Loan Circuit Breakers for EVM Smart Contracts

    [https://arxiv.org/abs/2609.30719](https://arxiv.org/abs/2609.30719)

    该论文提出Werracle，一种零存储、可装入单个32字节EVM存储槽的链上AI决策预言机，以亚美分的成本实现区块内即时响应的AI反射预言机与闪电贷熔断器，从而克服ZK-ML证明延迟过高、无法应对单区块内原子性DeFi攻击的瓶颈。

    

    当代链上人工智能（AI）遭遇了难以逾越的冯·诺依曼内存与延迟瓶颈。在以太坊虚拟机（EVM）存储中保存静态浮点神经网络权重矩阵需要数百万gas，使得直接进行链上推理无法实现。虽然零知识机器学习（ZK-ML）将矩阵张量乘法卸载到链下证明者，但它带来了致命的限制：10到300秒的SNARK证明延迟，以及每次证明验证25万至50万gas的开销。由于去中心化金融（DeFi）攻击——例如无抵押闪电贷攻击、掠夺性三明治MEV以及有毒的损失与再平衡（LVR）流——是在单个区块内原子性发生的，ZK-ML预言机无法及时作出反应。在此，我们提出了Werracle，一个生产级、零存储的链上AI决策预言机，可容纳于单个32字节的EVM存储槽（bytes32）中。该方法利用基础程序化Mandelbrot（摘要在此处截断）

    arXiv:2609.30719v1 Announce Type: cross  Abstract: Contemporary on-chain artificial intelligence (AI) encounters an intractable Von Neumann memory and latency wall. Storing static floating-point neural weight matrices inside Ethereum Virtual Machine (EVM) storage costs millions of gas, rendering direct on-chain inference impossible. While Zero-Knowledge Machine Learning (ZK-ML) offloads matrix tensor multiplications to off-chain provers, it introduces fatal constraints: 10 to 300 seconds of SNARK proving latency and 250,000 to 500,000 gas per proof verification. Because decentralized finance (DeFi) exploits - such as uncollateralized flash-loan attacks, predatory sandwich MEV, and toxic loss-versus-rebalancing (LVR) flow - occur atomically inside a single block, ZK-ML oracles cannot react in time. Here, we present Werracle, a production-grade, zero-storage on-chain AI decision oracle fitting inside a single 32-byte EVM storage slot (bytes32). Leveraging foundational procedural Mandelbr
    
[^142]: 言语重于顺序：Gemma 4 的行为评估

    Words Speak Louder Than Order: A Behavioral Evaluation of Gemma 4

    [https://arxiv.org/abs/2609.30716](https://arxiv.org/abs/2609.30716)

    该研究通过完全平衡的实验设计评估 Gemma 4 模型在接收冲突文档时的行为，发现信息源的语义表述（如“官方指南”或“最新更新”）对模型最终答案的影响显著强于文档的呈现顺序。

    

    当语言模型接收到两份相互冲突的文档作为输入时，它如何决定优先考虑哪一份？它是依赖于信息源的表述方式，还是文档的呈现顺序？我们在谷歌预训练的 Gemma 4-e4b 模型上，通过一个针对性的行为测试套件（n = 13 个条目，在简短的单轮对话情境下进行 784 次前向传播），采用完全平衡的实验设计对这一行为进行了评估。这一设置使我们能够在数学上隔离信息源表述和阅读位置的具体影响，同时确保模型固有的词汇偏好被抵消。在十个测试条件下，我们发现了以下结果：1. 信息源表述显著压过阅读位置。当二者直接竞争时，信息源的语义表述（例如将其呈现为官方指南或最新更新）对模型最终答案的影响显著强于文档的呈现顺序。

    arXiv:2609.30716v1 Announce Type: cross  Abstract: When a language model receives two conflicting documents as input, how does it decide which one to prioritize? Does it rely on how the sources are framed or the presentation order of the documents? We evaluated this behavior on Google's pre-trained Gemma 4-e4b model across a targeted behavioral suite (n = 13 items, 784 forward passes in short, single-turn contexts) using a completely counterbalanced experimental design. This setup allowed us to mathematically isolate the specific effects of source framing and reading position, while ensuring the model's natural vocabulary biases were canceled out.   Across ten test conditions, we discovered the following:   1. Source framing heavily overpowers reading position. When directly competing, the semantic framing of a source (such as presenting it as an official guideline or a fresh update) had a significantly stronger impact on the model's final answer than the presentation order of the docu
    
[^143]: CRC-Router：面向医疗智能体AI系统的风险约束路由

    CRC-Router: Risk-Constrained Routing for Medical Agentic AI Systems

    [https://arxiv.org/abs/2609.30714](https://arxiv.org/abs/2609.30714)

    提出CRC-Router，一种基于保形风险控制的风险约束、不确定性感知路由模块，可为医疗AI系统判断何时能自主处理病例、何时需升级人工审查，从而有效控制错误接受风险并保障临床部署安全。

    

    智能体AI系统在医学影像领域正被越来越多地探索，旨在提高处理吞吐量并减轻临床医生的工作负担；然而，安全部署仍然具有挑战性，因为自主运行产生的错误可能会传播到下游的临床决策中。因此，核心需求不仅是要有强大的预测性能，还需要一个可靠的路由机制，用于确定系统何时应该自主处理、何时应该将病例升级以进行进一步审查。为了填补这一空白，我们提出了CRC-Router，这是一种风险约束、不确定性感知的路由模块，既适用于传统医学预测模型，也适用于智能体医疗AI系统。CRC-Router将多个互补的不确定性信号与预测分数相结合，构建针对每个发现的逐项路由特征向量，并使用轻量级的逐发现风险模型将该向量映射为估计的错误接受风险，然后应用保形风险控制以……

    arXiv:2609.30714v1 Announce Type: new  Abstract: Agentic AI systems are increasingly being explored in medical imaging to improve throughput and reduce clinician workload; however, safe deployment remains challenging because autonomous errors may propagate into downstream clinical decisions. A central requirement is therefore not only strong predictive performance, but also a reliable routing mechanism that determines when the system should proceed autonomously and when a case should be escalated for further review. To address this gap, we propose CRC-Router, a risk-constrained, uncertainty-aware routing module that is applicable to both conventional medical prediction models and agentic medical AI systems. CRC-Router combines multiple complementary uncertainty signals with the predictive score to construct a per-finding routing feature vector, maps this vector to an estimated wrong-accept risk using a lightweight per-finding risk model, and then applies Conformal Risk Control (CRC) to
    
[^144]: VLALight：面向应急感知交通信号控制的轻量级视觉-语言-动作模型

    VLALight: Lightweight Vision-Language-Action Models for Emergency-Aware Traffic Signal Control

    [https://arxiv.org/abs/2609.30709](https://arxiv.org/abs/2609.30709)

    提出了VLALight，一个轻量级端到端视觉-语言-动作框架，通过融合多方向摄像头视图并将交叉路口观测与信号相位信息直接映射为离散信号动作，仅用0.5B参数的紧凑模型实现了应急感知的交通信号控制。

    

    交通信号控制（TSC）对于缓解城市拥堵至关重要。视觉-语言模型（VLMs）的最新进展使得对交叉路口场景的更丰富解读成为可能，为视觉情境感知的交通信号控制开辟了新的机遇。然而，模块之间松散的耦合和反复的信息转换可能导致细粒度视觉细节的丢失，而顺序推理则会引入显著的延迟。为了解决这些局限性，我们提出了VLALight，这是一个轻量级的端到端视觉-语言-动作框架，能够将交叉路口观测信息和信号相位信息直接映射为离散的信号动作。为了应对交通信号控制的多视角特性，VLALight将多个方向的摄像头视图组合成统一的视觉输入，并使用文本指令建立这些视图与交通流运动和信号相位之间的对应关系。这种设计使得仅使用0.5B参数的紧凑模型即可实现直接的动作预测。

    arXiv:2609.30709v1 Announce Type: cross  Abstract: Traffic signal control (TSC) is essential for mitigating urban congestion. Recent advances in vision-language models (VLMs) enable richer interpretation of intersection scenes, opening new opportunities for visual-context-aware TSC. However, the loose coupling and repeated information conversion between modules can lead to the loss of fine-grained visual details, while sequential inference introduces substantial latency. To address these limitations, we propose VLALight, a lightweight end-to-end vision-language-action framework that directly maps intersection observations and signal-phase information to discrete signal actions. To handle the multi-view nature of TSC, VLALight combines multiple directional camera views into a unified visual input and uses textual instructions to establish their correspondence with traffic movements and signal phases. This design enables direct action prediction with a compact 0.5 B-parameter model, with
    
[^145]: 结合通用与领域特定前置任务的脑部MR图像分割

    Combining General and Domain-Specific Pretext Tasks for Brain MR Image Segmentation

    [https://arxiv.org/abs/2609.30708](https://arxiv.org/abs/2609.30708)

    提出了一种联合优化体素级脑龄预测与图像修复的多任务自监督预训练框架，通过结合领域特定和通用前置任务学习互补的神经影像表示，以提升脑部MR图像分割性能。

    

    医学图像分析中的一个关键挑战是针对特定人群和疾病的大型标注数据集的稀缺性。由于深度学习模型严重依赖标注数据，因此需要有效的迁移学习策略来减少对手工标注的依赖。自监督学习已成为开发基础模型的一种有前景的方法，它能够从大规模无标注的医学影像数据集中学习可迁移的特征表示。在本研究中，我们研究了体素级脑龄预测作为一种领域特定的自监督前置任务，并将其与图像修复（一种广泛使用的非领域特定替代方法）进行比较。我们进一步提出了一种多任务自监督预训练框架，该框架联合优化这两个目标，以学习互补的神经影像表示。预训练模型在三个下游磁共振图像分割任务上进行了评估。

    arXiv:2609.30708v1 Announce Type: cross  Abstract: A key challenge in medical image analysis is the scarcity of large annotated datasets for specific populations and diseases. As deep learning models rely heavily on labeled data, effective transfer learning strategies are needed to reduce the dependence on manual annotations. Self-supervised learning has emerged as a promising approach for developing foundation models by enabling the learning of transferable feature representations from large-scale unlabeled medical imaging datasets. In this study, we investigate voxel-level brain age prediction as a domain-specific self-supervised pretext task and compare it with image inpainting, a widely used non-domain-specific alternative. We further propose a multitask self-supervised pretraining framework that jointly optimizes both objectives to learn complementary neuroimaging representations. The pretrained models are evaluated on three downstream magnetic resonance image segmentation tasks: 
    
[^146]: LAVOIR：通过摊销信息价值教会单次前向传播决策编码器何时提问以及问什么

    LAVOIR: Teaching a Single-Pass Decision Encoder When and What to Ask with Amortized Value of Information

    [https://arxiv.org/abs/2609.30706](https://arxiv.org/abs/2609.30706)

    LAVOIR将候选缺失信息槽位与答案选项一同置于输入中，使单次前向传播同时输出决策分布和各槽位的信息价值预期增益，从而以无需人工标注的方式教会决策模型何时提问、问什么。

    

    “System One”（系统一）决策模型，如TypeSafe的Jev及其开源对应模型Laya，能够在单次前向传播中以校准的概率回答关于文本的类型化问题，但它们无法询问缺失的信息：当首条消息没有说明区分两个部门的关键内容时，它们只能靠猜测。我们提出了LAVOIR（Laya with Value-Of-Information Routing，带信息价值路由的Laya），该方法将候选的缺失信息片段（槽位）放置在输入中答案选项的旁边，使得一次前向传播既能返回决策分布，又能针对每个槽位返回如果向用户询问该信息时正确决策概率的预期增益。VOI目标无需人工标注：黄金决策来自模式规则，LLM仅负责将消息和答案转化为文字表述，来自另一个模型家族的模型对每个文本进行校验，并且通过将每条消息与多个用户画像配对，对已实现增益进行回归从而估计预期增益。基尼不纯度上限约束了预（原文摘要在此处被截断）。

    arXiv:2609.30706v1 Announce Type: new  Abstract: "System One" decision models such as TypeSafe's Jev and its open counterpart Laya answer typed questions about a text in a single forward pass with calibrated probabilities, but they cannot ask for missing information: when a first message does not say what separates two departments, they guess. We present LAVOIR (Laya with Value-Of-Information Routing), which places the candidate pieces of missing information (slots) in the input next to the answer options, so that one forward pass returns both the decision distribution and, for every slot, the expected gain in the probability of the correct decision if the user were asked about it. VOI targets need no human labels: gold decisions come from schema rules, an LLM only verbalizes messages and answers, a model from another family checks every text, and pairing each message with several profiles makes regression on realized gains estimate the expected gain. A Gini-impurity cap bounds the pre
    
[^147]: 思维的代价：测试时推理在LLM交易中是否物有所值？

    The Price of Thought: Does Test-Time Reasoning Pay in LLM Trading?

    [https://arxiv.org/abs/2609.30705](https://arxiv.org/abs/2609.30705)

    该论文首次将LLM的推理控制作为经济干预进行评估，通过对DeepSeek、GPT和Gemini三大模型系列在一年美国股市上的80余万次预测开展对照实验，发现增加测试时推理的计算投入并不能可靠地提升扣除交易成本后的净投资组合回报。

    

    尽管大语言模型（LLM）中的推理时推理有望带来更好的决策，但其更高的计算成本可能无法转化为更好的经济结果。然而，推理控制很少被作为经济干预来评估——在这种干预中，模型输出的变化必须在扣除交易成本后转化为更优的投资组合。我们对来自DeepSeek、GPT和Gemini系列的代表性LLM进行了对照研究。我们在保持每个构建日期可用信息、提示词、输出格式和投资组合构建方法不变的前提下，改变推理投入程度。我们的评估涵盖了一整年的美国股票，在三种输入条件下进行：数值数据、可识别新闻和掩码新闻。评估包含超过80万次资产预测和多次重复的模型生成。在所有三个模型系列中，额外的推理并未带来净投资组合回报的可靠改善。对于DeepSeek，我们检验了从（原文在此处截断）……

    arXiv:2609.30705v1 Announce Type: new  Abstract: While inference-time reasoning in large language models (LLMs) promises better decision making, its higher computational cost may not yield better economic outcomes. Yet reasoning controls are rarely evaluated as economic interventions, where changes in model outputs must translate into better portfolios after trading costs. We conduct a controlled study of representative LLMs from the DeepSeek, GPT, and Gemini families. We vary reasoning effort while holding information available at each formation date, prompts, output formats, and portfolio construction fixed. Our evaluation covers a full year of U.S. equities under three input conditions: numerical, identifiable news, and masked news. It includes more than 800,000 asset predictions and repeated model generations. Across all three model families, additional reasoning does not produce a reliable improvement in net portfolio returns. For DeepSeek, where we examine the full progression fr
    
[^148]: SAGE：基于频率均衡的源锚定引导实现分层RGB-T对齐与融合

    SAGE: Source-Anchored Guidance via Frequency Equalization for Hierarchical RGB-T Alignment and Fusion

    [https://arxiv.org/abs/2609.30703](https://arxiv.org/abs/2609.30703)

    提出统一框架SAGE，通过可逆联合编码、源锚定低频调制、分层频率协同对齐与引导子带融合，端到端联合解决RGB-T融合中的空间失配与跨模态差异问题。

    

    在RGB-T融合中，空间失配与跨模态差异常常导致鬼影、结构模糊和内容失衡。现有方法通常将外观适配、几何对齐与信息融合解耦处理，限制了跨阶段的依赖传播。我们提出了基于频率均衡的源锚定引导分层RGB-T对齐与融合框架（SAGE），这是一个集成频率均衡、分层对齐和子带融合的统一框架。SAGE采用可逆联合编码和源特定的低频调制来推导结构与增益引导，同时保留源信息。分层频率协同对齐从低频近似中估计全局仿射几何，并将几何与上下文线索传递给高频相关推理，以实现可靠性感知的残差精修。引导子带融合联合聚合……（原文截断）

    arXiv:2609.30703v1 Announce Type: cross  Abstract: Spatial misregistration and cross-modal discrepancies often cause ghosting, structural blurring, and content imbalance in RGB-T fusion. Existing methods typically decouple appearance adaptation, geometric alignment, and information fusion, limiting dependency propagation across stages. We propose Source-Anchored Guidance via Frequency Equalization for Hierarchical RGB-T Alignment and Fusion (SAGE), a unified framework integrating frequency equalization, hierarchical alignment, and subband fusion. SAGE employs invertible joint encoding and source-specific low-frequency modulation to derive structural and gain guidance while preserving source information. Hierarchical frequency collaborative alignment estimates global affine geometry from low-frequency approximations and transfers geometric and contextual cues to high-frequency correlation reasoning for reliability-aware residual refinement. Guided subband fusion jointly aggregates the a
    
[^149]: 面向动态无人机网络的威胁感知节能部署：一种多智能体强化学习方法

    Threat-Aware Energy-Efficient Deployment for Dynamic UAV Networks: A Multi-Agent RL Approach

    [https://arxiv.org/abs/2609.30690](https://arxiv.org/abs/2609.30690)

    提出了一种威胁感知的无人机网络节能部署三步框架，结合威胁感知K均值聚类、最优匹配和MATD3多智能体强化学习，在实现零安全违规的同时最大化全局能效并加速收敛。

    

    在威胁多发环境中确保运行安全，对于作为空中基站的多无人机网络而言仍然是一项关键挑战。本文提出了一个高效框架，通过威胁感知聚类和基于奖励的安全约束机制来保障安全运行，同时最大化全局能效（EE）。该框架分三步执行：首先，威胁感知K均值（TAKM）算法确定所需的最少无人机数量并计算安全的初始部署位置；其次，最优匹配阶段将物理无人机分配到这些质心位置，以最小化能量消耗；第三，威胁感知多智能体双延迟深度确定性策略梯度（MATD3）算法动态优化轨迹、功率和用户关联。仿真结果表明，在所考虑的场景中，所提出的框架实现了零安全违规，同时相比（基线方法）取得了更优的能效和更快的收敛速度。

    arXiv:2609.30690v1 Announce Type: cross  Abstract: Ensuring operational safety in threat-prone environments remains a critical challenge for multi-UAV networks serving as aerial base stations. This paper proposes an efficient framework to maximize global energy efficiency (EE) while promoting safe operation through threat-aware clustering and reward-based safety enforcement. The proposed framework is executed in three steps. First, a threat-aware K-means (TAKM) algorithm determines the minimum required UAVs and computes safe initial placements. Second, an optimal matching stage assigns physical UAVs to these centroids to minimize energy expenditure. Third, a threat-aware multi-agent twin delayed deep deterministic policy gradient (MATD3) algorithm dynamically optimizes trajectories, power, and user associations. Simulation results show that the proposed framework achieves zero observed safety violations in the considered scenarios while achieving superior EE and faster convergence than
    
[^150]: LLM帕金森综合征：执行控制失败、令牌低效持续性，以及面向自主语言模型代理的不确定性感知全局执行控制架构

    LLM Parkinsonism: Executive-Control Failure, Token-Inefficient Persistence, and an Uncertainty-Aware Global Executive Control Architecture for Autonomous Language-Model Agents

    [https://arxiv.org/abs/2609.30662](https://arxiv.org/abs/2609.30662)

    该论文提出“LLM帕金森综合征”这一非临床隐喻，指出大语言模型代理在目标完成后仍持续低价值行动的根源在于生成、评估与停止权限集中于同一自条件循环，并提出不确定性感知的全局执行控制架构GEC v0.2以实现动作生成与项目级控制的分离。

    

    大语言模型（LLM）能够进行规划、使用工具、编写代码并执行长周期工作流，然而强大的局部能力并不能保证项目级的执行控制。代理可能在原始目标已经达成之后继续行动，产生低价值的优化、重复验证，以及对自身所创造复杂性的修复。我们使用“LLM帕金森综合征”作为一个狭义定义的、非临床的隐喻，来描述这种在任务层面价值递减时仍持续行动的模式。我们认为，这一问题并不能仅靠自回归的下一词元预测来解释，其更直接的原因是将提案生成、范围解释、进度评估和停止权限都集中在同一个自我条件化的循环之中。因此，我们提出了全局执行控制（GEC）v0.2，这是一种不确定性感知的治理架构，将动作生成与项目级控制分离开来。在一个包含24,000回合的匹配候选基准测试中……（原文摘要在此处截断）

    arXiv:2609.30662v1 Announce Type: new  Abstract: Large language models (LLMs) can plan, use tools, write code, and execute long-horizon workflows, yet strong local competence does not guarantee project-level executive control. Agents may continue acting after the original objective is satisfied, producing low-value refinements, repeated verification, and repairs to self-created complexity. We use LLM Parkinsonism as a narrowly defined, non-clinical metaphor for this pattern of persistent action despite diminishing task-level value. We argue that the problem is not explained by autoregressive next-token prediction alone, but more directly by concentrating proposal generation, scope interpretation, progress assessment, and stopping authority within the same self-conditioned loop. We therefore introduce Global Executive Control (GEC) v0.2, an uncertainty-aware governance architecture that separates action generation from project-level control. In a 24,000-episode matched-candidate benchma
    
[^151]: 交互式智能体中的因果保留：接口因式分解与选择性适应

    Causal Retention in Interactive Agents: Interface Factorization and Selective Adaptation

    [https://arxiv.org/abs/2609.30650](https://arxiv.org/abs/2609.30650)

    本文提出“因果保留”理论，证明冻结的学习状态能否正确响应独立于训练的机制探针取决于学习接口纤维与探针答案纤维之间的包含关系，并据此构建 Causal Core 系统，通过证据门控写入与选择性适应等机制实现无误差的目标更新。

    

    任务性能并不决定智能体保留哪种干预机制。我们研究因果保留：即一个冻结的学习状态能否回答一个独立于训练而固定的机制探针映射，该映射涵盖动作、上下文、直接目标、价值和延迟等维度。对于有限的结构因果模型类，最优探针误差是一个贝叶斯决策风险；当且仅当每个学习接口纤维都落在某一探针答案纤维之内时，该误差恰好消失，且任何通过对该接口进行后处理得到的状态都继承相同的下界。一个后验覆盖定理刻画了预算受限的重测试，而一个精确的编辑分解表明移位集是无误差目标更新的唯一支撑。Causal Core 通过证据门控写入、读出过滤、时间信用分配、隐藏上下文设置和局部诊断更新来实现这些条件。实验涵盖有限因果系统、连续模拟器、官方T……（原文摘要在此处不完整）

    arXiv:2609.30650v1 Announce Type: cross  Abstract: Task performance need not determine which intervention mechanism an agent retains. We study causal retention: whether a frozen learned state answers a mechanism-probe map fixed independently of training, including action, context, direct target, value, and delay. For finite structural causal model classes, the optimal probe error is a Bayes decision risk. It vanishes exactly when every learning-interface fiber lies within one probe-answer fiber; any state obtained by post-processing that interface inherits the same lower bound. A posterior-coverage theorem characterizes budgeted retesting, while an exact edit decomposition shows that the shifted set is the unique support of an error-free target update. Causal Core implements these conditions through evidence-gated writing, readout filtering, temporal credit, hidden-context setup, and local diagnostic updates. Experiments cover finite causal systems, continuous simulators, an official T
    
[^152]: 一个用于识别、分类和解释AI生成代码中偏见的框架

    A Framework for Identifying, Categorizing, and Explaining Bias in AI-Generated Code

    [https://arxiv.org/abs/2609.30642](https://arxiv.org/abs/2609.30642)

    本研究提出了一个基于分类体系的框架，用于识别、分类和解释AI生成代码中的偏见，并构建真实标准数据集评估了各类LLM作为自动化偏见检测与解释系统的可靠性。

    

    随着大型语言模型（LLM）被集成到软件开发工作流中，人们对AI生成代码中无意产生的偏见日益担忧。尽管有证据表明这些偏见确实存在，但系统性地识别、分类和解释这些偏见的研究仍然有限。本研究调查了AI生成代码中的偏见，并通过一个基于分类体系（taxonomy）的框架评估LLM能否可靠地识别和解释这些偏见。我们扩展了一个现有的包含偏见的AI生成Python代码数据集，并人工为代码片段标注偏见类别和人工编写的理由，从而建立了一个真实标准（ground-truth）数据集。利用该数据集，我们通过上下文学习（ICL）将专有和开源LLM评估为自动化偏见检测与理由生成系统。最后，我们使用结构化理由指标和代码识别指标，分析了LLM生成的解释与人工编写的理由之间的相似性。我们的研究结果表明……（原文摘要到此截断）

    arXiv:2609.30642v1 Announce Type: cross  Abstract: As Large Language Models (LLMs) become integrated into software development workflows, concerns regarding unintentional biases in AI-generated code. Although evidence suggests these biases exist, limited research has systematically identified, categorized, and explained them. This study investigates bias in AI-generated code and evaluates whether LLMs can reliably identify and explain it through a taxonomy-driven framework. We extended an existing dataset of biased AI-generated Python code and manually annotated snippets with bias categories and human-authored justifications to establish a ground-truth dataset. Using this dataset, we evaluated proprietary and open-source LLMs as automated bias detection and justification systems through ICL. Finally, we analyzed similarity between LLM-generated explanations and human-authored justifications using structured justification and code identification metrics.   Our findings demonstrate that 
    
[^153]: 音频大语言模型知道自己何时听不清你

    Audio LLMs Know When They Can't Hear You

    [https://arxiv.org/abs/2609.30625](https://arxiv.org/abs/2609.30625)

    该论文发现音频大语言模型无法通过自我评估或现有方法（如语音质量预测器、生成不确定性等）有效判断自身语音转录的可靠性，但转录可靠性信息强烈编码在模型音频编码器的内部表示中，可用于检测转录失败。

    

    音频大语言模型允许用户通过语音与模型进行交互。当输入的录音质量严重下降时，模型可能会误解用户的查询，并基于错误的转录内容进行回答。本文研究了“模型条件下的转录可靠性”问题：即音频大语言模型能否识别出其自身的转录何时是不可靠的。我们首先提示音频大语言模型评估其自身转录是否可靠，发现模型对自身转录可靠性的判断能力很差：在大多数情况下，它都预测自己的转录是可靠的。我们还发现，现有方法，包括语音质量预测器、音频大语言模型的生成不确定性、以及基于转录文本的WER估计，在检测转录失败方面所能提供的信号十分有限。相比之下，我们发现转录可靠性信息强烈地体现在模型的音频编码器表示中。基于这一观察……

    arXiv:2609.30625v1 Announce Type: new  Abstract: Audio large language models allow users to interact with the model through speech. When an input recording is too degraded, the model may misinterpret the user's query and respond based on an incorrect transcription. In this paper, we study model-conditional transcription reliability: whether an Audio LLM can recognize when its own transcription is unreliable. We first prompt the Audio LLM to assess whether its own transcription would be reliable, and find that the model is a poor judge of its own transcription reliability: in most cases, it predicts that its transcription will be reliable. We find that existing approaches, including speech quality predictors, audio LLM generation uncertainty, and transcript-conditioned WER estimation, provide limited signals for detecting transcription failures. In contrast, we discover that transcription reliability is strongly represented in the model's audio-encoder representations. Based on this obs
    
[^154]: 主体，而非作者：智能体数据空间中的作者身份危害

    Subjects, Not Authors: The Authorship Hazard in Agentic Dataspaces

    [https://arxiv.org/abs/2609.30614](https://arxiv.org/abs/2609.30614)

    该论文提出“作者身份危害”概念并确立核心原则：LLM智能体应始终是数据空间治理平面的主体而非作者，其发布授权通道须在构造上被关闭，其起草内容的影响则作为执行问题加以管控。

    

    数据空间连接器决定是否允许一次传输发生，而不决定所传输的值包含什么内容——这对契约式应用是可容忍的，但对能够组合工具调用并派生子智能体的LLM智能体而言则不然。关于生成治理制品的智能体的研究评估的是输出质量；而“谁有权授权制品投入使用”这一问题则落在该文献与治理文献之间的空白地带，双方均未涉及。已发布的策略正是数据空间决策点所强制执行的内容，因此发布是一个治理事件，而一个既是策略主体又是策略作者的智能体，就是在撰写约束自身的规范。我们将此称为“作者身份危害”，并提出一条原则：智能体是治理平面的主体，绝非其作者。其通往发布的授权通道在构造上被关闭；其影响通道——即起草供人类批准的内容——则被视为一个执行问题。在一个冻结的智能体草稿语料库上，未经批准而发布……

    arXiv:2609.30614v1 Announce Type: cross  Abstract: Dataspace connectors decide whether a transfer may occur, not what the transferred value contains, tolerable for contracted applications, not for LLM agents that compose tool calls and spawn sub-agents. Research on agents that generate governance artifacts evaluates output quality; who may authorize an artifact for use falls between that literature and the governance literature, and neither owns it. A published policy is what a dataspace's decision point enforces, so publication is a governance event, and an agent that is both policy subject and policy author writes the norms that bind it. We name this the authorship hazard and state one principle: an agent is a subject of the governance plane, never an author of it. Its authorization channel to publication is closed by construction; its influence channel, drafting what humans approve, is treated as an enforcement problem. On a frozen corpus of agent drafts, publishing without approval
    
[^155]: MedTokenBudget：面向皮肤镜图像分类的病灶保持型Token路由

    MedTokenBudget: Lesion-Preserving Token Routing for Dermoscopic Image Classification

    [https://arxiv.org/abs/2609.30613](https://arxiv.org/abs/2609.30613)

    本文提出MedTokenBudget，一个基于视觉Transformer的有监督token路由框架，利用病灶感知Token评分（LATS）模块在给定token预算下优先保留与病灶相关的图像块，从而为皮肤镜图像分类构建紧凑且富含病灶信息的表示。

    

    基于视觉Transformer构建的皮肤镜分类器会对所有图像块进行统一处理，而诊断证据实际上集中在病灶区域。现有的token剪枝方法通常依据通用的显著性或相似性信号来减少token数量，却很少关注保留下来的token子集是否仍然包含病灶。本文提出了MedTokenBudget，这是一种有监督的、位于骨干网络之后的token路由框架，当辅助病灶掩膜可用时，它能够学习构建紧凑且富含病灶信息的表示。其病灶感知Token评分（LATS）模块通过一个可学习的评分器融合注意力熵、特征范数和局部特征对比度，然后在目标预算下路由得分最高的前K个图像块。LATS通过预算课程学习、多样性正则化、注意力蒸馏和病灶掩膜监督进行训练。训练好的路由器通过病灶保留率进行评估，该指标直接衡量被保留token中所包含的真实病灶信息的多少。

    arXiv:2609.30613v1 Announce Type: cross  Abstract: Dermoscopy classifiers built on Vision Transformers process all image patches uniformly, although diagnostic evidence is concentrated in the lesion region. Existing token pruning methods reduce tokens using generic saliency or similarity signals, but rarely ask whether the retained subset still contains the lesion. This paper introduces MedTokenBudget, a supervised post-backbone token routing framework that learns to construct compact lesion-enriched representations when auxiliary lesion masks are available. Its Lesion-Aware Token Scoring (LATS) module fuses attention entropy, feature norm, and local feature contrast through a learned scorer, then routes the top-$K$ patches under a target budget. LATS is trained with budget curriculum learning, diversity regularization, attention distillation, and lesion-mask supervision. The trained router is evaluated with a lesion retention rate that directly measures how much ground-truth lesion ev
    
[^156]: 搜索之后的难题：对网络智能体在知识综合、组织与展示能力上进行基准测试

    The Hard Part Comes After Search: Benchmarking Web Agents on Synthesizing, Organizing, and Displaying Knowledge

    [https://arxiv.org/abs/2609.30604](https://arxiv.org/abs/2609.30604)

    该论文提出了KNOWS基准，通过开放式、复杂的浏览器任务联合评估网络智能体在信息检索、知识综合、任务分解以及产出文档等最终制品方面的综合能力。

    

    arXiv:2609.30604v1 公告类型：cross 摘要：现有的计算机使用智能体基准测试并未充分评估作为助手的智能体。一个有用的助手需要在复杂的多步骤工作流中检索信息，将其综合为制品（如文档、演示文稿、电子表格），并操作程序界面以产出连贯的最终成果。这类工作流需要推理与综合能力、复杂任务的分解能力，以及视觉与空间理解能力。为了在类似的工作流上研究智能体，我们推出了KNOWS，这是一个由开放式、复杂的基于浏览器的任务组成的基准测试，能够联合评估这些能力，且每个任务都以产出一个制品为最终目标。为了编写任务，我们开发了任务设计准则以及确保任务满足要求的协议。每个任务都配有一个评估器——一个将确定性检查与大语言模型判断相结合的程序，以在智能体评估中固有的丰富性、可靠性与自动化之间取得平衡。

    arXiv:2609.30604v1 Announce Type: cross  Abstract: Existing computer-use agent benchmarks do not fully evaluate agents acting as assistants. A useful assistant retrieves information across complex, multi-step workflows, synthesizes it into artifacts (documents, presentations, spreadsheets), and navigates program interfaces to produce a coherent final product. Such workflows demand reasoning and synthesis, decomposition of complex tasks, as well as visual and spatial understanding. To study agents on workflows like these, we introduce KNOWS, a benchmark of open-ended, complex, browser-based tasks that jointly evaluate these capabilities, with each task culminating in a produced artifact. To write tasks, we develop a task design rubric and a protocol for ensuring that tasks meet the requirements. Each task is paired with an evaluator, a program that combines deterministic checks with LLM judgments to balance the richness, reliability, and automation tradeoff inherent to agent evaluation.
    
[^157]: 动作强制：通过恢复底层自运动基在无监督视频上训练世界模型

    Action Forcing: Training World Models on Unsupervised Video by Recovering Underlying Egomotion Bases

    [https://arxiv.org/abs/2609.30595](https://arxiv.org/abs/2609.30595)

    该论文提出"Action Forcing"方法，无需动作标注或训练，仅通过追踪视频帧间像素位移并用PCA提取自运动基，即可将普通无标注视频转化为带真实控制信号（油门-偏航）的动作监督数据，用于训练可控世界模型。

    

    训练可控世界模型需要同步的动作标注，而这类数据集仍然难以获得。现有方法依赖带有标定传感器的专用测量平台、昂贵的人工标注，或缺乏真实依据的潜在动作模型。我们则通过恢复一个由数据导出的自运动基（无需训练），将普通的未标注视频转化为动作监督的训练数据。我们跨帧追踪像素位移，并利用自运动所诱导的反复出现的连贯结构，直接获得有真实依据的控制信号。使用主成分分析（PCA）这样简单的方法即可完成这一任务，我们发现其主成分可以提供带符号、可缩放且可组合的油门-偏航控制，尽管该方法只能恢复数据中所呈现的运动轴。为防止高容量的视频扩散Transformer（DiT）利用像素级监督，一个在线潜在评论家将冻结的解码器-追踪器-PCA（摘要在此处截断）

    arXiv:2609.30595v1 Announce Type: cross  Abstract: Synchronised action annotations are needed to train controllable world models and these datasets remain elusive. Existing approaches make use of instrumented platforms with calibrated sensors, costly manual annotation, or latent-action models which lack grounding. We instead turn ordinary unlabelled video into action-supervised training data by recovering (without training) a data-derived egomotion basis. We track pixel displacements across frames and exploit the recurring coherent structure induced by egomotion to obtain grounded control signals directly. Using a method as simple as principal components analysis perform this, we find that the leading components provide signed, scalable, and composable throttle--yaw controls, although the method can recover only motion axes represented in the data. To prevent a high-capacity video DiT from exploiting pixel-level supervision, an online latent critic distils a frozen decoder--tracker--PC
    
[^158]: T-RoPE：面向序列推荐的时间感知旋转位置编码

    T-RoPE: Time-Aware Rotary Position Embedding for Sequential Recommendation

    [https://arxiv.org/abs/2609.30576](https://arxiv.org/abs/2609.30576)

    提出T-RoPE，一种时间感知的旋转位置编码，通过基于时间戳的角度、可学习时间系数和多尺度频率等机制打破标准RoPE的时间平移不变性，使序列生成式推荐模型能够捕捉时间间隔、行为周期与季节性等关键时间信息。

    

    大规模推荐系统日益采用大语言模型背后的序列生成式方案，将Transformer引入推荐领域，同时也沿用了为文本设计的组件，包括旋转位置编码。在语言模型中，RoPE对词元索引进行编码以实现相对位置推理，但在推荐场景中，交互索引仅记录事件顺序，无法反映流逝的时间、跨尺度的行为周期或日历相位。我们重新审视这一设计选择，提出T-RoPE，一种用于序列生成式推荐的时间感知RoPE，它以基于时间戳的角度、可学习的时间系数、多尺度频率库、偏移查询对齐以及非平稳的键旋转，替代仅基于索引的旋转。我们证明标准RoPE即使作用于时间戳，仍保持时间平移不变性，无法区分季节性上下文；而T-RoPE在保留RoPE的……

    arXiv:2609.30576v1 Announce Type: new  Abstract: Large-scale recommenders increasingly adopt the sequential generative recipe behind large language models, bringing the Transformer into recommendation along with design choices made for text, including Rotary Position Embedding (RoPE). In language models, RoPE encodes token indices for relative position reasoning, but in recommendation, an interaction index records only event order, saying nothing about elapsed time, behavioral cycles across scales, or calendar phase. We revisit this choice and propose T-RoPE, a time-aware RoPE for sequential generative recommendation that replaces index-only rotation with timestamp-based angles, learnable temporal coefficients, multiscale frequency banks, shifted query alignment, and non-stationary key rotation. We prove that standard RoPE, even on timestamps, remains time-translation invariant and cannot distinguish seasonal contexts, and that T-RoPE breaks this invariance while preserving the RoPE in
    
[^159]: HARDEN：通过约束进化搜索生成更难且保留答案的评估案例

    HARDEN: Constrained Evolutionary Search for Harder, Answer-Preserving Evaluation Cases

    [https://arxiv.org/abs/2609.30571](https://arxiv.org/abs/2609.30571)

    HARDEN是一种约束进化搜索方法，能在保持预期答案不变的前提下将现有评估案例改编为更难的变体，使语言模型准确率平均下降22.7%、最高下降49.9%。

    

    语言模型通常在人工精选的基准测试上进行评估，而这些基准测试未能充分体现企业级部署场景的复杂性。我们提出了HARDEN，这是一种约束进化搜索方法，能够将现有评估案例的输入改编为更具挑战性的变体，同时保持其预期输出不变。HARDEN沿着生成的领域特定复杂性维度进行搜索，同时强制执行保持任务语义、真实性和执行有效性等可行性约束。在FinQA、PubMedQA和ContractNLI三个数据集以及三种规模的Qwen3.5模型（35B-A3B、122B-A10B和397B-A17B）上，与使用相同可行性检查的单次基线方法相比，HARDEN使任务模型准确率平均降低22.7%，最高降低49.9%。这些结果表明，进化搜索能够生成难度显著更高的有效评估案例。

    arXiv:2609.30571v1 Announce Type: new  Abstract: Language models are often evaluated on curated benchmarks that underrepresent the complexity of enterprise deployments. We introduce HARDEN, a constrained evolutionary search method to adapt the input of existing evaluation cases into more challenging variants while keeping their expected outputs fixed. HARDEN searches along generated domain-specific complexity axes while enforcing feasibility constraints such as preserving task semantics, realism, and execution validity. Across FinQA, PubMedQA, and ContractNLI and three Qwen3.5 model scales (35B-A3B, 122B-A10B, and 397B-A17B), HARDEN reduces task-model accuracy by 22.7% on average and by up to 49.9% relative to single-pass baselines using the same feasibility checks. These results show that evolutionary search can produce substantially harder valid evaluation cases.
    
[^160]: Atelier：通过超网络学习冷冻电镜体积的局部自监督特征

    Atelier: Learning Local Self-Supervised Features for CryoEM Volumes via Hypernetworks

    [https://arxiv.org/abs/2609.30569](https://arxiv.org/abs/2609.30569)

    Atelier是一个基于Transformer超网络的自监督框架，通过摊销冷冻电镜图谱的隐式神经表示拟合，实现了高效、跨样本对齐的局部特征学习，可用于大规模冷冻电镜体积的特征提取。

    

    冷冻电镜图谱解读需要空间局部化、跨样本一致且在不同空间尺度上均具信息量的特征。大多数用于图谱注释的深度学习方法从固定的体素网格中提取特征。然而，隐式神经表示（INR）能够将体积数据建模为与尺度无关、以坐标为条件的函数，因此对冷冻电镜很有吸引力；但为每个图谱单独拟合一个INR对于大规模特征提取而言成本过高，且所产生的表征无法在样本之间对齐。我们提出了Atelier，这是一个自监督框架，通过对重建的冷冻电镜图谱进行摊销式的INR拟合来解决这一问题。Atelier在电子显微镜数据库（EMDB）的5,439个图谱上进行了预训练，是一个基于Transformer的超网络，能够在广泛的蛋白质结构（包括大型多亚基组装体）上生成高保真的重建。除重建之外，该预训练生成的INR……（原文摘要在此处截断）

    arXiv:2609.30569v1 Announce Type: new  Abstract: CryoEM map interpretation requires features that are spatially localized, consistent across samples, and informative across spatial scales. Most deep learning methods for map annotation extract features from fixed voxel grids. However, implicit neural representations (INRs) are able to model volumetric data as scale-agnostic, coordinate-conditioned functions. INRs are therefore attractive for cryoEM, but fitting a separate INR for each map is too expensive for large-scale feature extraction and produces representations that are not aligned across samples. We introduce Atelier, a self-supervised framework that amortizes INR fitting for reconstructed cryoEM maps. Pretrained on 5,439 Electron Microscopy Data Bank maps, Atelier is a transformer-based hypernetwork that generates high-fidelity reconstructions across a wide range of protein structures, including large multi-subunit assemblies. Beyond reconstruction, the INR generated by the pre
    
[^161]: 少思考，模拟得更好：直觉式提示提升大语言模型智能体对个体社交媒体反应的模拟能力，包括对陌生内容的反应

    Thinking Less to Simulate Better: Intuitive Prompting Improves LLM Agents Simulating Individual Social Media Reactions, Including Unfamiliar Content

    [https://arxiv.org/abs/2609.30563](https://arxiv.org/abs/2609.30563)

    本研究通过八名真实用户画像与六十八条帖子反应的对照实验发现，采用直觉式（少推理）提示并以态度性内容而非人口统计背景构建用户画像，可显著提升大语言模型智能体模拟个体社交媒体真实反应的准确性（包括对陌生内容的反应），且智能体与画像的一致性并不等同于行为保真度。

    

    平台政策越来越多地在人工用户上进行测试，这使得智能体的保真度变得十分重要。然而，令人信服的虚假用户资料也可能在选举前操纵公众舆论的感知。以往的验证工作主要集中于智能体与人类行为的一致性，却很少关注智能体的行为是否符合其被赋予的资料。本研究通过问卷调查、深度访谈和书面自我介绍对八名塞尔维亚参与者进行画像，记录了他们对六十八条社交媒体帖子的反应，并让四个语言模型在五种提示条件（资料内容和指令风格各不相同）下预测这些反应。结果表明，态度性内容相比人口统计学背景故事大幅提升了预测效果；智能体与其所述资料的匹配程度甚至高于参与者与其自身问卷答案的匹配程度；而且一旦提供了资料信息，一致性与保真度之间并无关联。指令……

    arXiv:2609.30563v1 Announce Type: new  Abstract: Platform policies are increasingly tested on artificial users, making agent fidelity important. Yet convincing fake profiles could also manipulate perceived public opinion before elections. Validation has concentrated on agreement with human behaviour and has paid little attention to whether an agent behaves in line with the profile it was given. The present study profiled eight Serbian participants through a questionnaire, a deep interview, and a written self-presentation, recorded their reactions to sixty-eight social media posts, and asked four language models to predict those reactions under five prompt conditions varying profile content and instruction style. Attitudinal content improved prediction over demographic backstories by a wide margin. Agents matched their stated profiles more closely than participants matched their own survey answers, and consistency proved unrelated to fidelity once profile information was present. Instru
    
[^162]: 通过认知实验范式探究智能体记忆中的稳定性-可塑性权衡

    Probing Stability-Plasticity Tradeoffs in Agent Memory through Cognitive Experimental Paradigms

    [https://arxiv.org/abs/2609.30558](https://arxiv.org/abs/2609.30558)

    本文提出受认知科学启发的MemProbe框架，通过干扰、错误信息、巩固强度和再巩固窗口四种可复用的实验范式，超越传统最终答案准确率评估，系统诊断智能体记忆中稳定性与可塑性之间的权衡。

    

    智能体记忆系统越来越多地被用于维护长期用户偏好、任务状态和不断演变的事实，但当前的评估方法往往将记忆行为简化为最终答案的准确率。我们提出了MemProbe，一个受认知科学启发的框架，用于诊断智能体记忆中的稳定性-可塑性权衡。该框架的动机来自认知记忆研究的一个核心洞察：记忆是重构性的，并受到干扰、来源可靠性、强化和再激活等因素的塑造。MemProbe将这一洞察转化为四种可复用的实验范式（干扰、错误信息、巩固强度和再巩固窗口），这些范式操纵了记忆何时应该被更新、保留或视为不确定。该框架进一步将正确性分解为行为特征，揭示系统如何更新、保留、归因和时间性地组织信息。我们在56个剧集的诊断任务中实例化了这些范式（摘要在此处被截断）。

    arXiv:2609.30558v1 Announce Type: cross  Abstract: Agent memory systems are increasingly used to maintain long-term user preferences, task states and evolving facts, but current evaluations often collapse memory behavior into final-answer accuracy. We introduce MemProbe, a cognitive-science-inspired framework for diagnosing stability-plasticity tradeoffs in agent memory. The framework is motivated by a core insight from cognitive memory research: memory is reconstructive and shaped by interference, source reliability, reinforcement, and reactivation. MemProbe turns this insight into four reusable experimental paradigms (interference, misinformation, consolidation strength, and reconsolidation window) that manipulate when a memory should be updated, preserved, or treated as uncertain. It further decomposes correctness into behavioral profiles that reveal how systems update, preserve, attribute, and temporally organize information. We instantiate these paradigms in a 56-episode diagnosti
    
[^163]: 自动驾驶潜在空间监控器的审计

    Auditing Latent-Space Monitors for Autonomous Driving

    [https://arxiv.org/abs/2609.30557](https://arxiv.org/abs/2609.30557)

    该论文审计了自动驾驶中基于潜在空间探针的运行时故障监控方法，发现尽管内部表征能够有效预测故障（LaneSegNet和VAD的AUROC分别达0.780和0.868），但仅使用模型输出、自车状态和驾驶指令等外部信息即可达到相当甚至更好的性能（0.825和0.924），表明访问模型内部表征并非实现强大故障预测的必要条件。

    

    运行时故障监控器可以利用模型的内部表征来预测故障。我们在两个自动驾驶任务中审计了这种监控策略：使用LaneSegNet的在线矢量化地图生成和使用VAD的端到端规划。我们发现，在这两个任务中，帧级别的错误在推理阶段都是可预测的。对于LaneSegNet，一个有监督的潜在空间探针在高Chamfer误差上达到了受试者工作特征曲线下面积（AUROC）0.780；据我们所知，这是首个针对在线矢量化地图生成的事后帧级故障监控器。对于VAD，一个有监督的规划潜在空间探针在平均位移误差（mean-ADE）故障上达到了AUROC 0.868。我们的审计表明，访问内部表征对于实现强大的故障预测并非必要。仅使用LaneSegNet预测输出的监控器达到了AUROC 0.825，而对于VAD，仅使用自车状态、驾驶指令和规划器预测的轨迹在相同的平均位移误差指标上就达到了0.924。

    arXiv:2609.30557v1 Announce Type: cross  Abstract: Runtime failure monitors can use a model's internal representations to anticipate failures. We audit this monitoring strategy across two autonomous-driving tasks: online vectorized map generation with LaneSegNet and end-to-end planning with VAD. We find that frame-level errors are predictable at inference in both tasks. For LaneSegNet, a supervised latent probe reaches Area Under the Receiver Operating Characteristic curve (AUROC) 0.780 for high Chamfer error; to our knowledge, this is the first post-hoc frame-level failure monitor for online vectorized map generation. For VAD, a supervised planning-latent probe reaches AUROC 0.868 for mean-ADE failure.   Our audit shows that internal access is not necessary for strong failure prediction. A monitor using only LaneSegNet's prediction outputs reaches AUROC 0.825, while for VAD, ego state, driving command, and the planner's predicted trajectory reach 0.924 on the same mean-ADE endpoint. A
    
[^164]: 具有排序偏好的时间投票中的比例代表制

    Proportional Representation in Temporal Voting with Ranked Preferences

    [https://arxiv.org/abs/2609.30555](https://arxiv.org/abs/2609.30555)

    该论文将比例代表制公理（JR、PJR、EJR、PSC）拓展到具有排序偏好的时间投票中，构建了公理层次结构，并发现与批准投票不同，没有任何版本的扩展正当代表制（EJR）是可以被保证的。

    

    我们研究时间投票中的比例代表制，即每一轮选出一名候选人。先前的工作主要集中于批准投票，而我们考虑的是可能随时间变化的排序偏好。一种自然的方法是将每位选民排名靠前的候选人视为被批准的，但合适的截断点可能因选民和轮次而异。因此，我们要求比例性在每一种可接受的截断点选择下都成立，无论是固定且共同的、共同但在不同轮次间变化的，还是由每位选民在每轮单独设定的。将这些解释与正当代表制（JR）、比例正当代表制（PJR）、扩展正当代表制（EJR）以及坚实联盟比例性（PSC）的时间版本相结合，我们得到了一个公理层次结构。我们探讨这些公理中哪些可以被保证，以及需要对未来有多大的了解。与批准投票不同，没有任何版本的EJR能够被保证，而对于其他公理，截断点方面的灵活性……（原文摘要在此处被截断）

    arXiv:2609.30555v1 Announce Type: cross  Abstract: We study proportional representation in temporal voting, where one candidate is selected in each round. While prior work has focused on approval ballots, we consider ranked preferences, which may change over time. A natural approach treats each voter's top candidates as approved, but the right cutoff may differ across voters and rounds. We therefore require proportionality to hold for every admissible choice of cutoffs, whether fixed and common, common but varying across rounds, or set individually for each voter in each round. Combining these interpretations with temporal versions of justified representation (JR), proportional JR (PJR), extended JR (EJR), and proportionality for solid coalitions (PSC) gives us a hierarchy of axioms. We ask which of these axioms can be guaranteed, and with how much knowledge of the future. Unlike with approval ballots, no version of EJR can be guaranteed, and for the other axioms, flexibility in the cu
    
[^165]: 面向昂贵进化优化的排序可靠教师引导适应度近似：一项TinyML架构搜索研究

    Rank-Reliable Teacher-Guided Fitness Approximation for Expensive Evolutionary Optimization: A TinyML Architecture Search Study

    [https://arxiv.org/abs/2609.30553](https://arxiv.org/abs/2609.30553)

    提出TGL-NSGA-II框架，通过教师引导的轻量知识蒸馏生成排序可靠的低保真适应度分数，并与高斯过程代理模型融合，以显著降低昂贵TinyML神经架构搜索中进化优化的评估成本。

    

    昂贵的进化搜索并不总是需要对每个候选方案进行精确的适应度估计，它往往只需要对一个更简单问题的可靠回答：哪个候选方案更好？我们通过教师引导学习NSGA-II（TGL-NSGA-II）来满足这一需求，这是一个面向受限Tiny机器学习（TinyML）神经架构搜索的低保真度框架。预训练的教师模型将样本组织成由难度和类别共同定义的分层。每个候选方案随后经过KD-Lite——一种在紧凑训练集上进行的简短且有上限的知识蒸馏过程——然后在一个单独的分层评估集上被打分。这一教师引导的分数与高斯过程代理模型相融合，用于选择候选方案进行完整评估。对于固定的候选种群，我们分析了评估方差、分数集中度、成对排序反转、期望Kendall-τ、第一前沿识别以及超体积扰动。我们还推导了……（原文摘要在此处截断）

    arXiv:2609.30553v1 Announce Type: new  Abstract: Expensive evolutionary search does not always need an exact fitness estimate for every candidate. It often needs a reliable answer to a simpler question: which candidate is better? We address this need through Teacher-Guided Learning NSGA-II (TGL-NSGA-II), a low-fidelity framework for constrained Tiny Machine Learning (TinyML) neural architecture search. A pretrained teacher organizes samples into strata defined jointly by difficulty and class. Each candidate then undergoes KD-Lite, a short and capped knowledge-distillation procedure on a compact training set, before being scored on a separate stratified evaluation set. This teacher-guided score is fused with a Gaussian-process surrogate to select candidates for full evaluation. For a fixed candidate population, we analyse evaluation variance, score concentration, pairwise rank inversion, expected Kendall-$\tau$, first-front identification, and hypervolume perturbation. We also derive a 
    
[^166]: Benchy：迈向面向任务AI基准测试的通用语言

    Benchy: towards a universal language for task-oriented AI benchmarks

    [https://arxiv.org/abs/2609.30550](https://arxiv.org/abs/2609.30550)

    Benchy提出了一种用于AI基准测试的语义语言和执行引擎，通过规范化YAML语法与通用运行时契约将基准测试定义与AI系统解耦，旨在建立面向任务AI基准测试的通用语言。

    

    Benchy是一种用于AI程序基准测试的语义语言与执行引擎。一个基准测试完全由程序、评分函数和数据集定义，记作B=(P,S,D)，且与被测试的AI系统相互独立；一次运行将两者绑定，记作R=(B,AI)。基准测试以规范化YAML语言编写，其中每个语义概念只有一种有效语法，通过共享的任务/领域/语言本体进行分类，并被确定性地编译为引擎所执行的规范化JSON中间表示。编译过程只改变表示形式而不改变含义：它不会修复无效的定义，也不会注入隐藏的默认值。程序使用命名输入和输出字段的固定模式，叶子输出字段即为评分维度，引擎对外暴露一个通用运行时契约——输入为命名字段对象、输出为命名字段对象——外部AI系统只需在边界处适配该契约，因此集成机制永远不会传播进基准测试本身。

    arXiv:2609.30550v1 Announce Type: new  Abstract: Benchy is a semantic language and execution engine for benchmarking AI programs. A benchmark is completely specified by a program, a scoring function, and a dataset, B=(P,S,D), and is separate from the AI-system taking it; a run binds the two, R=(B,AI). Benchmarks are authored as canonical YAML in which each semantic concept has one valid syntax, classified by a shared task/domain/language ontology, and deterministically compiled into a canonical JSON intermediate representation that the engine executes. Compilation changes representation, not meaning: it does not repair invalid definitions or inject hidden defaults. Programs use fixed schemas of named input and output fields, the leaf output fields are the scoring dimensions, and the engine exposes one universal runtime contract --- a named-field input object in, a named-field output object out --- to which external AI-systems adapt at the boundary, so integration mechanics never propag
    
[^167]: Muon算法的收敛性保证：新的参数区间与推广

    Convergence guarantees for Muon: New parameter regimes and generalizations

    [https://arxiv.org/abs/2609.30546](https://arxiv.org/abs/2609.30546)

    本文首次证明了Muon优化算法的渐近收敛性，通过更精确的Newton-Schultz迭代代理揭示其本质是隐含正则化诱导的有界预条件子所构成的预条件Polyak重球方法，并据此提出了具有相同收敛保证的Nesterov变体Muesterov。

    

    本文通过对Newton-Schultz迭代采用比典型矩阵符号函数更精确的代理，首次建立了Muon算法的渐近收敛性保证。我们证明，在适当的超参数选择下，迭代序列满足 $\lim_{k\to\infty}\|\nabla f(x_k)\|=0$；并且在全球Polyak-Łojasiewicz条件下，函数值序列线性收敛。关键洞察在于：Muon的Newton-Schulz实现中隐含的正则化会诱导出一个有界的预条件子，这将Muon揭示为一种预条件Polyak重球方法，并使经典的Lyapunov分析得以适用。这一观察自然地启发了将相同的预条件结构应用于Nesterov梯度评估。我们通过引入Muesterov——一种基于Nesterov动量的Muon变体——将这一想法形式化，并证明其享有与Muon相同的收敛性保证，从而扩展了理论保证的适用范围。

    arXiv:2609.30546v1 Announce Type: cross  Abstract: In this paper, we establish the first asymptotic convergence guarantees for the Muon algorithm through a more accurate proxy for the Newton-Schultz iteration than the typical matrix sign function. We prove that, for appropriate choices of hyperparameters, the iterates satisfy $\lim_{k\to\infty}\|\nabla f(x_k)\|=0$, and, under a global Polyak-\L{}ojasiewicz condition, that the sequence of function values converges linearly. The key insight is that the regularization, implicit in Muon's Newton-Schulz implementation, induces a bounded preconditioner, exposing Muon as a \emph{preconditioned Polyak heavy-ball} method and enabling a classical Lyapunov analysis. This observation naturally motivates applying the same preconditioning structure to the Nesterov gradient evaluation. We formalize this idea by introducing \emph{Muesterov}, a Nesterov-based variant of Muon, and prove that it enjoys the same convergence guarantees, extending the theor
    
[^168]: Inquesto Score：一种语音智能体可靠性评估协议

    Inquesto Score: A reliability Protocol For Voice Agents

    [https://arxiv.org/abs/2609.30514](https://arxiv.org/abs/2609.30514)

    本文提出 Inquesto Score（IS）协议，通过明确定义失败事件与严重级别，以固定版本化评估群体中成功完成呼叫者目标且无功能性故障的通话占比，来可复现、可解释地衡量已部署语音智能体的可靠性。

    

    语音智能体越来越多地被部署在工作流程中，其中失败的交互可能影响交易、访问权限及其他重要后果，因此需要可复现且可解释的评估方法。我们提出了 Inquesto Score（IS），这是一种衡量语音智能体可靠性的协议，其定义为：在一个固定且带版本控制的评估通话群体中，实现呼叫者目标且未发生功能性故障或更严重问题的通话所占的百分比。IS 并非将异构指标混合在一起，而是明确定义了失败事件和严重程度级别，并对已部署的语音流水线进行评估。时序故障（包括抢话和响应延迟）直接从音频中测量，而语义性和状态相关的故障则通过场景谓词、工具调用轨迹以及固定的开源模型评判者进行评估。该评分还附带关于行为、声学鲁棒性、身份处理和说话人群体的诊断视图，但这些诊断信息不会被合并进评分之中。

    arXiv:2609.30514v1 Announce Type: cross  Abstract: Voice agents are increasingly deployed in workflows where failed interactions can affect transactions, access, and other consequential outcomes, creating a need for reproducible and interpretable evaluation. We introduce Inquesto Score (IS), a protocol for measuring voice-agent reliability as the percentage of calls in a fixed, versioned evaluation population that achieve the caller's goal without a functional failure or worse. Rather than combining heterogeneous metrics, IS defines explicit failure events and severity levels and evaluates the deployed voice pipeline. Timing failures, including talk-over and delayed responses, are measured directly from audio, while semantic and state-dependent failures are evaluated using scenario predicates, tool traces, and a pinned open-model judge. Diagnostic views of behavior, acoustic robustness, identity handling, and speaker groups accompany the score without being combined into it. Inquesto S
    
[^169]: PolicyAttention：Softmax注意力实现闭环控制的策略镜像下降

    PolicyAttention: Softmax Attention Implements Policy Mirror Descent for Closed-Loop Control

    [https://arxiv.org/abs/2609.30500](https://arxiv.org/abs/2609.30500)

    该论文构造了一个带显式残差的因果softmax“行动者—环境—单步评论家”协议，证明softmax注意力可以作为重复控制器实现策略镜像下降，并用预注册实验验证了训练的pre-LN Transformer能够恢复目标计算。

    

    因果softmax注意力能否将策略镜像下降实现为一个重复的控制器，而非仅仅一步代数恒等式？负熵策略镜像下降（PMD）具有逐状态更新式 PMD_η(π,Q)=softmax(log π+ηQ)。基于已知的Q-TD-PMD递推关系，我们构建了一个固定的因果softmax“行动者—环境—单步评论家”协议，其中包含显式的行动者残差、路由残差、采样残差和归一化残差，并将这些残差传播到最终实际返回的策略上。该构造明确了有限logit/全支撑的定义域、外部分词与采样边界，以及归一化编译所需的均值为零的LayerNorm载体条件。分别训练的pre-LN Transformer在经验上能够恢复目标计算。在所测试的固定规则中，冻结的单步审计模型最接近PMD；在一项预注册的五次运行、S=4的重复控制测试中，学习到的a……（原文摘要在此处截断）

    arXiv:2609.30500v1 Announce Type: cross  Abstract: Can causal softmax attention implement policy mirror descent as a repeated controller rather than a one-step algebraic identity? Negative-entropy policy mirror descent (PMD) has the statewise update $\operatorname{PMD}_\eta(\pi,Q)=\operatorname{softmax}(\log\pi+\eta Q)$. Building on the known Q-TD-PMD recursion, we construct one fixed causal-softmax actor--environment--one-step-critic protocol with explicit actor, routing, sampling, and normalization residuals, and propagate them to the policy actually returned. The construction states the finite-logit/full-support domain, the external tokenization and sampling boundary, and the mean-zero LayerNorm carrier conditions required by the normalized compilation.   Separately trained pre-LN Transformers recover the target computation empirically. A frozen one-step audit model is closest to PMD among the tested fixed rules; in a preregistered five-run $S=4$ repeated-control test, the learned a
    
[^170]: 打破同质性：多样化人格集合以实现创造性的LLM输出

    Breaking Homogeneity: Diversifying Persona Sets for Creative LLM Outputs

    [https://arxiv.org/abs/2609.30492](https://arxiv.org/abs/2609.30492)

    该论文将人格多样化建模为集合级条件化问题，在“选择vs生成”与“空间填充vs前沿探索”两个正交设计维度上提出四种方法，显著提升了语言模型创造性输出的多样性，其中进化式人格生成在AUT上将回答多样性提高了78.8%。

    

    语言模型在开放式任务中常常产生同质化的回答；这种同质性可能引发群体思维——即观点向单一且可能次优的决策趋同。我们将人格多样化表述为一个集合级条件化问题，并研究了两个正交的设计选择：是选择人格还是生成人格，以及是空间填充式多样性还是前沿探索式多样性。我们用四种方法实例化了这一设计空间，涵盖覆盖性与分散性子集选择、均匀覆盖采样以及进化式人格生成。在替代用途任务（AUT）、Infinity-Chat 和发散联想任务（DAT）上的评估表明，所提出的方法在各种任务和创造力目标上均展现出优势。在AUT上，与仅使用任务提示相比，进化式人格生成将回答多样性提高了78.8%，原创性提高了26.1%，灵活性提高了49.5%，整体创造力提高了13.9%，同时保持了98.5……

    arXiv:2609.30492v1 Announce Type: cross  Abstract: Language models often produce homogeneous responses to open-ended tasks; such homogeneity can spawn groupthink-the convergence of ideas toward a singular and potentially suboptimal decision. We formulate persona diversification as a set-level conditioning problem and study two orthogonal design choices: selecting versus generating personas, and space-filling versus frontier-seeking diversity. We instantiate this design space with four methods spanning coverage and dispersion subset selections, uniform-coverage sampling, and evolutionary persona generation. Evaluations on the Alternative Uses Task (AUT), Infinity-Chat, and Divergent Association Task (DAT) show the benefits of the proposed methods across tasks and creativity objectives. On AUT, evolutionary persona generation increases response diversity by 78.8%, originality by 26.1%, flexibility by 49.5%, and holistic creativity by 13.9% over task-only prompting, while maintaining 98.5
    
[^171]: BioEVAL：面向生物工程的大语言模型与多模态模型的全球多机构基准测试

    BioEVAL: A global, multi-institutional benchmark of large language and multimodal models for bioengineering

    [https://arxiv.org/abs/2609.30489](https://arxiv.org/abs/2609.30489)

    该论文提出了BioEVAL——首个由22个研究团队共同构建、覆盖11个生物工程子领域的博士级全球多机构基准，通过608个评估项目（多项选择题、文献综合任务和多模态实验图像解读）来评估大语言模型与多模态模型在生物工程前沿实验推理中的能力。

    

    大语言模型（LLM）在通用推理方面取得了历史性突破，并在生物医学科学领域获得了早期成功。然而，现有的LLM基准测试侧重于事实记忆，对模型在前沿任务和多模态任务上的表现提供的洞察有限。我们构建了BioEVAL（AI与大语言模型的生物工程验证），这是一项全球性的多机构合作计划，旨在评估生物工程（BE）各子领域的实验推理能力。BioEVAL涵盖11个主要的生物工程子领域以及一组未分类项目，汇集了22个研究团队，共同创建了一个博士级基准，包含608个评估项目：1）380道多项选择题（MCQ，经审核后保留359道）；2）218项文献综合任务；3）10个包含实验图像解读的多模态问题。所有基准项目在评估前均经过了出题团队的专家审查和集中质量控制。（摘要原文在此处截断）

    arXiv:2609.30489v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated historic breakthroughs in general reasoning with early successes in biomedical science. However, existing LLM benchmarking emphasizes factual recall, offering limited insight into model performance on frontier and multimodal tasks. We assembled BioEVAL (BioEngineering Validation of AI and LLMs), a global, multi-institutional initiative designed to assess experimental reasoning capability across bioengineering (BE) subfields. BioEVAL spans 11 major BE subfields plus a set of uncategorized items, bringing together 22 research groups to create a PhD-level benchmark comprising 608 evaluation items: 1) 380 multiple-choice questions (MCQs, 359 retained after audit), 2) 218 literature synthesis tasks, and 3) 10 multimodal problems with experimental image interpretation. Benchmark items underwent authoring-group expert review and centralized quality control before evaluation. Following evaluation, a
    
[^172]: 大语言模型真的理解上下文吗？一种基于知识图谱的评估框架

    Do LLMs Understand Context? A Knowledge Graph-Based Evaluation Framework

    [https://arxiv.org/abs/2609.30484](https://arxiv.org/abs/2609.30484)

    提出了一种基于知识图谱的评估框架，通过语义结构相似度衡量大语言模型在问答任务中真正的上下文理解能力，弥补了BLEU和困惑度等传统指标只能评估表面性能的不足。

    

    尽管大语言模型（LLM）已经展现出卓越的语言能力，但一个深刻的问题始终萦绕在其核心：这些模型究竟是真正理解上下文，还是只是以前所未有的规模擅长模式匹配？大语言模型中的上下文理解是指从给定上下文中正确提取相关信息，将其整合为连贯的内部表示，并对其进行推理，从而产生事实一致且立足于上下文的回应的能力。然而，双语评估替补（BLEU）和困惑度等传统方法只能衡量表面层次的性能，这在问答（QA）任务中暴露了一个关键缺口，因为问答的回应必须立足于上下文，而不仅仅是记忆中的关联。为填补这一空白，我们提出了一种新颖的基于知识图谱（KG）的评估框架，用于评估大语言模型在问答中的上下文理解能力，其核心是语义结构相似度（Semantic Structural Similarity）。

    arXiv:2609.30484v1 Announce Type: new  Abstract: While large language models (LLMs) have achieved remarkable linguistic capabilities, a profound question lingers at their core: do these models truly comprehend context or simply excel at pattern matching on an unprecedented scale? Contextual understanding in LLMs refers to the ability to correctly extract relevant information from a given context, integrate it into a coherent internal representation, and reason over it to produce factually consistent and contextually grounded responses. However, traditional methods such as BiLingual Evaluation Understudy (BLEU) and perplexity simply measure surface-level performance. This reveals a critical gap in question answering (QA), where responses must be contextually grounded rather than simply being memorized associations. To fill this void, we propose a novel knowledge graph (KG) based evaluation framework for LLM contextual understanding in QA. Central to this is Semantic Structural Similarit
    
[^173]: CARGO：面向生产环境中智能体AI的上下文感知检索门控评估

    CARGO: Context-Aware Retrieval-Gated Evaluation of Agentic AI in Production

    [https://arxiv.org/abs/2609.30471](https://arxiv.org/abs/2609.30471)

    针对生产环境中智能体AI评估时“参考-实例分歧”（即参考答案描述的是不同实体导致正确回答被误判为错误）的问题，提出CARGO框架，将检索参考视为流程范例、基于实时实例上下文判定事实并将评估门控于检索置信度之上。

    

    基于参考答案的LLM-as-a-judge评估方法假定参考答案即为目标答案。在部署于动态实体（如支持工单、资产、账户）之上运行的智能体系统中，最接近的可用参考答案通常是将正确流程应用到了不同的实体上，因此字面化的裁判会将不同的标识符、日期和状态判定为错误或幻觉。我们将这种失败模式命名为“参考-实例分歧”。我们提出了CARGO框架，该框架：（i）将检索到的参考答案视为流程范例，并将事实性判断建立在实际实例的实时观测上下文之上；（ii）为每条声明分配三向状态（支持、矛盾、无法验证），并仅对矛盾进行惩罚；（iii）通过检索置信度对评估进行门控，将生产环境评估建模为选择性预测问题。我们引入了CARGO-Bench，这是一个基于扰动的诊断套件，其真值通过构造方式获得，能够区分宽容……（原文在此截断）

    arXiv:2609.30471v1 Announce Type: cross  Abstract: Reference-based LLM-as-a-judge evaluation assumes the reference answer is the target. In deployed agentic systems that operate over dynamic entities (support cases, assets, accounts), the closest available reference typically applies the correct procedure to a different entity, so a literal judge penalizes different identifiers, dates, and statuses as errors or hallucinations. We name this failure mode reference-instance divergence (RID). We propose CARGO, a framework that (i) treats retrieved references as procedural exemplars and grounds factual judgments in the live instance's observed context, (ii) assigns each claim a three-way status (supported, contradicted, unverifiable) and penalizes only contradictions, and (iii) gates evaluation by retrieval confidence, casting production evaluation as selective prediction. We introduce CARGO-Bench, a perturbation-based diagnostic suite with ground truth by construction that separates lenien
    
[^174]: 面向嘈杂警用音频的预训练ASR伪标签方法

    Pretrained ASR Pseudo-labeling for Noisy Police Audio

    [https://arxiv.org/abs/2609.30469](https://arxiv.org/abs/2609.30469)

    该论文发现现有内部置信度指标无法有效过滤嘈杂警用广播通信音频的伪标签，并提出利用大语言模型作为评判者的外部过滤范式，显著降低了适配后ASR模型的词错误率。

    

    预训练ASR系统在嘈杂的警用广播通信（BPC）音频上表现不佳，阻碍了理解警务决策的努力。伪标签提供了一种无需昂贵人工标注即可改进ASR的无监督路径，但该方法在极度嘈杂领域的效果尚不清楚。在这项工作中，我们系统地评估了伪标签方法在将基础ASR模型（Whisper和Qwen3-ASR）适配到来自巴尔的摩和芝加哥的嘈杂BPC领域语料库时的机遇与局限。我们证明了现有的内部置信度指标（对数概率和STAR分数）无法区分高质量与低质量的BPC伪标签，并提出了一种外部的LLM-as-a-judge（大语言模型充当评判者）过滤范式，利用模型的参数化知识来剔除上下文上不合理的转录文本。我们的LLM评判过滤器比内部指标过滤得更加激进，并显著降低了伪标签训练集在BPC语料库上的词错误率（WER）。

    arXiv:2609.30469v1 Announce Type: new  Abstract: Pretrained ASR systems perform poorly on noisy Broadcast Police Communication (BPC), hindering efforts to understand police decision-making. Pseudo-labeling offers an unsupervised path to improve ASR without expensive human labels, but the efficacy of this approach on very noisy domains is not known. In this work, we systematically assess the opportunities and limits of pseudo-labeling to adapt foundation ASR models (Whisper and Qwen3-ASR) to noisy BPC domain corpora from Baltimore and Chicago. We demonstrate that existing internal confidence metrics (log-probabilities and STAR scores) fail to distinguish between high and low quality BPC pseudo-labels, and we introduce an external LLM-as-a-judge filtering paradigm that leverages parametric knowledge to discard contextually implausible transcripts. Our LLM-judging filters more aggressively than internal metrics and significantly reduces WER of the pseudo-labeled training sets across the B
    
[^175]: 面向情境感知XR界面的基准测试框架

    A Benchmarking Framework for Context-aware XR Interfaces

    [https://arxiv.org/abs/2609.30466](https://arxiv.org/abs/2609.30466)

    该论文提出ContextXR——首个面向情境感知XR界面的基准测试框架，通过功能面连通图表示、带功能面级标注的MineXR++数据集以及三个规范化的建议任务，实现了对XR自适应方法的系统化、可重复评估。

    

    日常扩展现实（XR）系统旨在让用户在恰当的时间和地点以情境感知的方式访问恰当的功能，并在用户切换情境时尽量减少手动重新配置。然而，这类界面很难评估：现有的原型设计与用户研究工作流缺乏一种系统化、可重复的方法来跨用户和场景比较自适应方法。我们提出了ContextXR，一个面向情境感知XR界面的新型基准测试框架。ContextXR将XR应用表示为由功能面构成的连通图，每个功能面是一组语义连贯的相关能力，共同支持一个共享的用户意图。基于这一表示，我们构建了MineXR++数据集，它通过功能面级别的标注增强了先前的XR界面数据，并形式化了情境感知建议的三个规范任务：情境因素分析、初始功能面建议和下一功能面建议。我们的评估协议对建议结果进行评分……

    arXiv:2609.30466v1 Announce Type: cross  Abstract: Everyday Extended Reality (XR) systems aim to provide context-aware access to the right functionalities at the right time and place, with minimal manual reconfiguration as users switch context. Yet these interfaces are hard to evaluate: current prototyping and user-study workflows offer no systematic, repeatable way to compare adaptation methods across users and scenarios. We present ContextXR, a novel benchmarking framework for context-aware XR interfaces. ContextXR represents an XR application as a connected graph of functional facets, each a semantically coherent group of related capabilities that together support a shared user intent. On this representation, we build MineXR++, a dataset augmenting prior XR interface data with facet-level annotations, and formulate three canonical tasks of context-aware suggestion: context factor analysis, initial facet suggestion, and next facet suggestion. Our evaluation protocol scores suggestion
    
[^176]: 用于蛋白质扩散模型测试时对齐的谱反馈方法

    Spectral Feedback for Test-Time Alignment of Protein Diffusion Models

    [https://arxiv.org/abs/2609.30456](https://arxiv.org/abs/2609.30456)

    提出谱反馈算法，通过在反馈回路中选择编辑位置并对词元重新掩码与重采样，使蛋白质离散扩散模型能在测试时迭代纠正自身生成结果，将“重新审视哪些词元”而非“如何分配词元标签”作为对齐的核心问题。

    

    针对离散扩散模型的奖励最大化对齐方法主要聚焦于引导反向过程，即通过影响词元logits或在中间步骤选择有利序列来实现。这些方法在很大程度上将推理视为单向过程，缺乏重新审视不理想词元选择的机制。我们提出了谱反馈，这是一种在反馈回路中选择编辑位置的算法，使模型能够迭代地纠正自身的生成结果。该方法利用离散扩散模型的掩码结构，通过重新掩码和重新采样词元来实现，类似于图像编辑方法中重新引入噪声潜变量并重新运行反向过程的做法。以往的对齐方法关注应分配什么样的词元标签以最大化目标奖励，而我们则将“应重新审视哪些词元”作为核心对齐问题。选择编辑位置具有挑战性，因为编辑效果是……（摘要截断）

    arXiv:2609.30456v1 Announce Type: new  Abstract: Reward maximization alignment methods for discrete diffusion models have primarily focused on steering the reverse process, either by influencing token logits or by selecting favorable sequences at intermediate steps. These approaches largely treat inference as a unidirectional process, lacking mechanisms for revisiting undesirable token selections. We introduce Spectral Feedback, an algorithm that selects edit-positions in a feedback loop, allowing the model to iteratively correct its own generations. This approach leverages the mask structure of discrete diffusion models by re-masking and re-sampling tokens, analogous to image editing methods that reintroduce noisy latents and re-run the reverse process. While prior alignment methods focus on what token labels to assign to maximize a target reward, we instead treat which tokens to revisit as the central alignment problem. Selecting edit-positions is challenging because edit effects are
    
[^177]: 基于三维结构预测跨膜蛋白拓扑结构

    Predicting Transmembrane Protein Topology from 3D Structure

    [https://arxiv.org/abs/2609.30446](https://arxiv.org/abs/2609.30446)

    该论文提出利用图神经网络SchNet直接从三维结构的全原子嵌入中预测跨膜蛋白拓扑结构，在无需预训练权重的情况下展现出优异的预测潜力。

    

    本文提出了一种利用最先进的图神经网络SchNet来推断蛋白质拓扑结构的新方法。该模型在与开发近期DeepTMHMM模型相同的数据集上进行训练，并采用五折交叉验证。与仅使用蛋白质序列或α-碳原子作为特征的传统方法不同，我们以这种方式设计分类器，从而利用了所有原子级别的嵌入表示。在未应用任何预训练权重的情况下，最终结果表明图神经网络在拓扑结构预测方面具有巨大潜力。

    arXiv:2609.30446v1 Announce Type: new  Abstract: This paper presents a novel approach to infer protein topology using the state-of-the-art graph neural network (GNN), SchNet. The model is trained on the same dataset used to develop the recent DeepTMHMM model with 5-fold cross-validation. Unlike the conventional approaches based on using only the protein sequences or the $\alpha$-carbons as features, we have decoded our classifier in this way, so all atom-level embeddings are used. Without applying any pre-trained weight, the final results have shown great potential that GNNs can be used for topological predictions.
    
[^178]: 自然语言中欠明确任务的上下文不确定性主动消解

    Actively Resolving Contextual Uncertainty for Underspecified Tasks in Natural Language

    [https://arxiv.org/abs/2609.30428](https://arxiv.org/abs/2609.30428)

    本文提出CLUE框架，使机器人面对欠明确的自然语言任务时，能通过LLM推导的策略假设任务相关概念与潜在计划，并结合在线构建的语言嵌入地图，以闭环方式主动消解上下文不确定性。

    

    基础模型赋予机器人理解自然语言并推理环境上下文的能力，然而大多数语言条件策略都假设目标是明确规定的，且任务相关信息通过先验地图预先提供。在陌生环境中执行欠明确的任务会带来很高的上下文不确定性：机器人必须同时推断什么构成任务成功、什么构成相关信息，以及这些信息存在于何处（或是否存在）。我们通过CLUE（闭环上下文不确定性消解，Closed-Loop contextual Uncertainty rEsolution）框架来解决这些局限，该框架能够针对自然语言给出的欠明确任务主动消解上下文不确定性。CLUE使用由大语言模型（LLM）推导的策略来假设任务相关概念和潜在计划，然后利用在线构建的语言嵌入地图将这些假设落地为具体动作。该策略依次评估各假设……（原文摘要在此处截断）

    arXiv:2609.30428v1 Announce Type: cross  Abstract: Foundation models provide robots with the ability to interpret natural language and reason about environmental context, yet most language-conditioned policies assume that goals are well-specified and that task-relevant information is provided upfront via a prior map. Operating in unfamiliar environments with underspecified tasks entails high contextual uncertainty: the robot must jointly infer what constitutes task success, what constitutes relevant information, and where (or whether) that information exists. We address these limitations via CLUE (Closed-Loop contextual Uncertainty rEsolution), a framework for actively resolving contextual uncertainty given underspecified tasks in natural language. CLUE uses an LLM-derived policy to hypothesize task-relevant concepts and potential plans. It then uses a language-embedded map, which is constructed online, to ground these hypotheses into actions. The policy sequentially evaluates hypothes
    
[^179]: 使用对比学习方法理解扰动参数集合敏感性

    Understanding Perturbed Parameter Ensemble Sensitivities Using A Contrastive Learning Approach

    [https://arxiv.org/abs/2609.30420](https://arxiv.org/abs/2609.30420)

    本文开发了一种可解释的对比学习模型，将CAM6扰动参数集合的多变量云与辐射场映射到共享表示空间，能以超过94%的准确率区分不同暖雨微物理方案，同时保留季节变率与参数扰动引起的集合离散度，为气候模式参数敏感性的解释与校准提供了新方法。

    

    扰动参数集合揭示了物理参数如何影响气候模拟，但在多变量、空间结构化的输出中解释参数敏感性仍然具有挑战性，特别是在根据观测数据校准模型时。我们开发了一个可解释的对比学习模型，将5个月平均云和辐射场映射到一个共享表示空间中。我们在两个各含100个成员的第六版社区大气模式（CAM6）扰动参数集合（PPE）的场数据上训练该模型，这两个集合涵盖34个参数，仅在暖雨微物理方案上有所不同：一个是默认的整体微物理方案KK2000，另一个是分档微物理方案的神经网络模拟器TAU-ML。学习到的表示能以超过94%的线性分类准确率区分两个扰动参数集合，同时保留了季节变率以及由参数扰动引起的集合离散度。在共享表示空间中，卫星……

    arXiv:2609.30420v1 Announce Type: cross  Abstract: Perturbed parameter ensembles (PPEs) reveal how physics parameters affect climate simulations, but interpreting parameter sensitivities across multivariate, spatially structured outputs remains challenging, particularly when calibrating models against observations. We develop an explainable contrastive learning model that maps 5 monthly cloud and radiation fields into a shared representation space. We train the model on the fields of two 100-member Community Atmosphere Model version 6 (CAM6) PPEs, spanning 34 parameters, that only differ in the warm rain microphysics scheme: KK2000, the default bulk microphysics scheme, and TAU-ML, a neural network emulator of a bin microphysics scheme. The learned representations separates two PPEs with over 94\% linear classification accuracy while preserving the seasonal variability and ensemble spread due to parameter perturbations. In the shared representation space, the representations of satelli
    
[^180]: 概念与组块的统一解释

    A Unified Account of Concepts and Chunks

    [https://arxiv.org/abs/2609.30414](https://arxiv.org/abs/2609.30414)

    本文提出将认知心理学中原本分离的概念与组块统一到一个理论框架中，扩展Cobweb分类模型并实现TRELLIS系统，成功应用于同时包含概念性和组块性元素的上下文无关文法学习。

    

    认知心理学已经研究了人们如何编码、使用和学习描述类别的概念，以及人们如何表征、识别和获取用于熟悉元素模式的组块。这两个主题的研究文献几乎互不相关，这对认知的统一理论构成了挑战。在本文中，我们回顾了Cobweb——一个关于分类和概念形成的计算模型，并提出了一个将组块及其获取过程纳入其中的扩展理论。该理论不对模态做任何假设，适用于任何可分解为元素及元素间关系的经验。我们还介绍了TRELLIS——该理论的一个实现系统，并展示了其在学习上下文无关文法方面的应用。我们选择上下文无关文法作为测试平台，是因为它们同时包含类似概念和类似组块的元素。此外，我们报告了在三个合成文法上的实验结果，证明了该系统的能力

    arXiv:2609.30414v1 Announce Type: cross  Abstract: Cognitive psychology has studied how people encode, use, and learn concepts that describe categories, and how they represent, recognize, and acquire chunks for familiar patterns of elements. The literatures on these two topics are nearly disjoint, which poses a challenge for unified theories of cognition. In this paper, we review Cobweb, a computational account of categorization and concept formation, and propose an extended theory that incorporates chunks and their acquisition. The theory makes no commitments about modality, applying to any experience that decomposes into elements and relations among them. We also present \trellis/, an implementation of this theory, and illustrate its application to learning context-free grammars, which we adopt as a testbed because they involve both concept-like and chunk-like elements. In addition, we report experimental results on three synthetic grammars that demonstrate the system's ability to re
    
[^181]: 什么能改进多模态虚假信息检测？来自大规模实证研究的答案

    What Improves Multimodal Misinformation Detection? Answers from a Large-Scale Empirical Study

    [https://arxiv.org/abs/2609.30402](https://arxiv.org/abs/2609.30402)

    本文通过涵盖3,375余次实验的大规模实证研究，系统性地回答了哪些设计选择能改进多模态虚假信息检测、它们何时会悄然失效，为构建更强大可靠的检测系统提供了实用指导。

    

    多模态虚假信息正日益被精心设计得看似令人信服，即将一段文本声明与一张看似能“证明”该声明的图像配对。然而在实践中，构建有效的检测器往往取决于一组很少受到系统性研究的设计选择。在本文中，我们针对多模态虚假信息检测的设计选择开展了一项大规模研究，进行了超过3,375次实验，涵盖三个基准数据集以及广泛的预训练视觉与语言骨干模型。通过系统性比较和针对性的鲁棒性分析，我们提炼出了实用的指导原则：哪些设计选择有帮助、它们何时会悄无声息地失效，以及流水线中的哪些方面最能强烈地影响模型行为，从而回答了4个关键研究问题（RQs）。我们旨在为设计更强大、更可靠的多模态虚假信息检测系统提供可靠的基础，从而为更广泛的研究社区做出贡献。

    arXiv:2609.30402v1 Announce Type: cross  Abstract: Multimodal misinformation is increasingly crafted to look convincing by pairing a textual claim with an image that appears to "prove" it. Yet in practice, building effective detectors often hinges on a small set of design choices that are rarely examined in a controlled way. In this paper, we conduct a large-scale study of multimodal design choices for misinformation detection with over 3,375 experiments- spanning three benchmark datasets and a broad range of pre-trained vision and language backbones. Through systematic comparisons and targeted robustness analyses, we distill practical guidance on which design choices help, when do they fail silently, and what aspects of the pipeline most strongly shape model behavior, answering 4 key Research Questions (RQs). We aim to provide a reliable foundation for designing stronger and more dependable multimodal misinformation detection systems, thus contributing to the broader research communit
    
[^182]: 可解释人工智能方法评估的合成真值框架

    A Synthetic Ground-Truth Framework for the Evaluation of Explainable AI Methods

    [https://arxiv.org/abs/2609.30397](https://arxiv.org/abs/2609.30397)

    该论文提出了一种基于合成真值的可解释AI评估框架，通过受控干预生成已知输入重要性的合成数据集，解决了传统保真度评估无法验证解释是否真正反映模型底层决策过程的问题。

    

    arXiv:2609.30397v1 公告类型：新论文 摘要：评估可解释人工智能（XAI）方法是一项具有挑战性的任务，原因在于缺乏可靠的评估程序，尤其是缺乏真值解释。在现有文献中，已有的评估方法通常通过度量解释相对于黑盒模型预测的保真度来评估解释质量。然而，这类评估策略仅量化了解释重现模型输出的程度，却无法确保解释正确地反映模型底层的决策过程。因此，不同的解释可能获得相似的保真度分数，同时对模型行为提供不一致或误导性的解读。在本文中，我们提出了一种基于合成真值的XAI方法评估框架。所提出的方法依赖于受控干预来生成合成数据集，在这些数据集中输入分量的重要性是已知的，从而为解释的正确性提供了可靠的评估基准。

    arXiv:2609.30397v1 Announce Type: new  Abstract: Evaluating explainable Artificial Intelligence (XAI) methods is a challenging task due to the lack of reliable evaluation procedures and, in particular, the absence of ground truth explanations. In the literature, existing evaluation approaches typically assess explanations by measuring their fidelity with respect to the predictions of a black-box model. However, such evaluation strategies only quantify the degree to which an explanation reproduces the model's output, without ensuring that the explanation correctly reflects the underlying decision process. As a consequence, different explanations may achieve similar fidelity scores while providing inconsistent or misleading interpretations of the model behavior. In this paper, we propose a framework for the evaluation of XAI methods based on synthetic ground truth. The proposed approach relies on controlled interventions to generate synthetic datasets in which the importance of input com
    
[^183]: 分则隐蔽，合则有害：针对基于技能的智能体系统的技能级联攻击

    Stealth Apart, Harm Together: Skill Cascading Attacks on Skill-Based Agent Systems

    [https://arxiv.org/abs/2609.30383](https://arxiv.org/abs/2609.30383)

    本文提出“技能级联攻击”这一新威胁范式，通过将恶意目标分散到多个各自看似无害的技能中，利用技能间的组合执行对基于技能的智能体系统发起隐蔽攻击。

    

    技能（skill）是一种模块化包，由自然语言指令、可执行脚本和参考资源组成，智能体可以在运行时加载它，以扩展其在特定任务上的能力。因此，基于技能的智能体系统实现了第三方能力的灵活复用，但这种技能生态系统的开放性也带来了新的攻击面。先前的工作主要关注单个技能内部的漏洞，而很少关注跨技能交互所产生的风险。在本文中，我们提出了技能级联攻击，这是一种威胁范式：恶意目标被分散到多个技能中，使得每个修改在单独看来都是良性的，但它们的组合执行却是有害的。例如，在一个处方审查流程中，第一个技能削弱提取病史中最近停用药物的信号，第二个技能降低任何与之相关的药物相互作用的严重性等级……（摘要截断）

    arXiv:2609.30383v1 Announce Type: new  Abstract: A skill is a modular package of natural-language instructions, executable scripts, and reference resources that an agent can load at runtime to extend its capabilities for a specific task. Skill-based agent systems therefore enable flexible reuse of third-party capabilities, but the openness of this skill ecosystem also opens up a new attack surface. Prior work has focused on vulnerabilities within individual skills, but little attention has been paid to risks that arise from interactions across skills. In this paper, we introduce skill cascading attacks, a threat paradigm in which a malicious objective is distributed across multiple skills so that each modification looks benign in isolation, yet their combined execution is harmful. For instance, in a prescription-review pipeline, the first skill weakens signals of recently discontinued medications in the extracted history, the second downgrades the severity of any drug interaction tied 
    
[^184]: DanLing NestedTensor：面向深度学习的可组合多重参差张量

    DanLing NestedTensor: Composable Multi-Ragged Tensors for Deep Learning

    [https://arxiv.org/abs/2609.30379](https://arxiv.org/abs/2609.30379)

    DanLing NestedTensor 是一种将多重参差结构内嵌为张量自身属性的 PyTorch 张量抽象，使广播、特征变换和归约操作能够可组合地处理变长数据，在 BERT 任务上相比填充方法实现了最高 3.39 倍的加速。

    

    变长输入在深度学习中十分常见，但稠密批处理会分配一个共享的包络，并在填充上耗费计算。这种开销会随变化维度的增多而成倍增加：例如显式的成对状态需要分配 $BN_{\max}^2$ 个位置，而非 $\sum_i N_i^2$。打包技术消除了这种浪费，但组合打包操作仍然需要逻辑轴和样本边界信息，而扁平缓冲区已不再暴露这些信息。我们提出了 DanLing NestedTensor，这是一种 PyTorch 张量抽象，它使多重参差结构成为张量本身的属性。打包的值携带基于张量的分区和逻辑维度顺序，因此广播可以创建参差轴，特征变换可以保留它们，而归约操作可以消费它们。同一表示贯穿自动微分以及急切与编译两种执行模式。在 A100 上，与同模式填充方法相比，在四个 BERT 规模上的几何平均加速比为：急切模式 2.74 倍、编译模式 3.39 倍，以及 1.97 倍

    arXiv:2609.30379v1 Announce Type: cross  Abstract: Variable-size inputs are common in deep learning, but dense batching allocates a shared envelope and spends computation on padding. The cost multiplies across varying axes: an explicit pair state allocates $BN_{\max}^2$ positions instead of $\sum_i N_i^2$. Packing removes that waste, but composing packed operations still requires the logical axes and sample boundaries a flat buffer no longer exposes. We present DanLing NestedTensor, a PyTorch tensor abstraction that makes multi-ragged structure a property of the tensor itself. Packed values carry tensor-backed partitions and logical dimension order, so broadcasting creates ragged axes, feature transformations retain them, and reductions consume them. The same representation carries through autograd and both eager and compiled execution. On an A100, the geometric-mean speedup over same-mode padding is 2.74$\times$ eager and 3.39$\times$ compiled across four BERT scales, and 1.97$\times$
    
[^185]: 基于对决反馈的成本感知最优大语言模型识别

    Cost-Aware Best-LLM Identification using Dueling Feedback

    [https://arxiv.org/abs/2609.30360](https://arxiv.org/abs/2609.30360)

    该论文提出了一种结合对决反馈与异质查询成本的成本感知多臂老虎机算法，用于在给定置信度下识别最佳大语言模型，并证明了其渐近最优成本性能。

    

    受从一组具有异质查询成本的大语言模型（LLM）中识别最佳模型这一问题的启发，我们提出并分析了一种多臂老虎机（MAB）的变体，该变体具有两个特点：（i）对决反馈，即通过对模型响应之间的成对比较来提供稳健的偏好信号；（ii）异质采样成本，反映了查询不同LLM所需的不同成本。在假设存在孔多塞赢家（Condorcet winner）的前提下（我们在多个真实世界数据集上对该条件进行了实证验证），我们提出了一种Track-and-Stop风格的算法，用于在给定置信水平下的最优臂识别。我们证明了随着误差趋于零，该算法几乎必然地实现渐近最优成本。最后，我们在合成数据和真实世界实例上对该方法进行了广泛评估，结果表明其相较于经典的无成本感知算法及其成本感知扩展版本均取得了一致的改进。

    arXiv:2609.30360v1 Announce Type: cross  Abstract: Inspired by the problem of identifying the best model from a collection of large language models (LLMs) with heterogeneous querying costs, we formulate and analyse a variant of the multi-armed bandit (MAB) with (i) dueling feedback, where pairwise comparisons between model responses provide robust preference signals, and (ii) heterogeneous sampling costs, reflecting the differing costs of querying different LLMs. Assuming the existence of a Condorcet winner, a condition we empirically validate across multiple real-world datasets, we propose a Track-and-Stop style algorithm for best-arm identification with prescribed confidence. We prove that the algorithm almost surely achieves the asymptotically optimal cost as the error tends to zero. Finally, we extensively evaluate our approach on both synthetic and real-world instances, demonstrating consistent improvements over classical cost-unaware algorithms and their cost-aware extensions.
    
[^186]: 策略性自洽性

    Strategic Self-Consistency

    [https://arxiv.org/abs/2609.30352](https://arxiv.org/abs/2609.30352)

    本文揭示了一种针对自洽性推理服务的潜在欺诈行为：不忠实的模型提供商可通过策略性地生成并重排额外的推理路径，使每条路径在多数投票中看起来都不可或缺，从而在避开审计检测的情况下人为增加路径数量以向用户多收费。

    

    自洽性（Self-consistency）已成为一种流行的技术，通过生成多条推理路径并通过多数投票选出最终答案来增强大型语言模型的推理能力。然而，由于模型提供商通常按照生成的推理路径数量向用户收费，他们存在人为增加路径数量的经济动机。在这项工作中，我们证明了一个不忠实的提供商可以利用这一动机，使用一种简单高效的算法同时避免被审计者检测：该算法通过生成并策略性地重新排序额外的推理路径，使得每条路径看起来都是达到多数所必需的。为了验证我们的算法，我们在涵盖数学、科学和问答任务的基准数据集上，使用来自Llama和Qwen系列的多个指令模型，以及从DeepSeek-R1蒸馏而来的推理模型进行了实验。我们的结果表……（摘要截断）

    arXiv:2609.30352v1 Announce Type: cross  Abstract: Self-consistency has become a popular technique for enhancing the reasoning abilities of large language models by generating multiple reasoning paths and selecting the final answer through a majority vote. However, because model providers typically charge users in proportion to the number of reasoning paths generated, they have a financial incentive to artificially increase the path count. In this work, we show that an unfaithful provider can exploit this incentive using a simple, efficient algorithm while avoiding detection by an auditor: by generating and strategically reordering additional reasoning paths, the algorithm makes every path appear necessary to reach the majority. To validate our algorithm, we conduct experiments with multiple instruct models from the Llama and Qwen families, as well as reasoning models distilled from DeepSeek-R1, on benchmark datasets spanning mathematics, science, and question answering. Our results su
    
[^187]: 自适应多分辨率高斯过程：基于天然数据稀疏协方差矩阵的可扩展精确推断

    Adaptive multi-resolution Gaussian processes: Scalable exact inference with naturally data-sparse covariance matrices

    [https://arxiv.org/abs/2609.30348](https://arxiv.org/abs/2609.30348)

    该论文提出一种自适应多分辨率高斯过程框架，通过直接锚定样本点的自适应多分辨率基函数构建天然数据稀疏的协方差矩阵，并结合稀疏Cholesky逆算法，实现了既可扩展又精确的高斯过程推断。

    

    高斯过程是概率机器学习的基石，然而将其扩展到大规模数据集通常需要在计算效率和模型保真度之间进行权衡。本工作通过提出一个既可扩展又精确的自适应多分辨率高斯过程框架来弥合这一差距。我们的关键创新是利用自适应多分辨率基函数构建天然数据稀疏的协方差矩阵。这些基函数直接锚定在样本上，从而无需辅助点。通过缩小多分辨率基的支撑域，矩阵块的大小受到限制，进而保证了稀疏性。数据稀疏协方差矩阵的逆可通过稀疏Cholesky逆算法被精确且高效地计算。为进一步提升预测不确定性估计的质量，我们构造了增广基函数。理论分析与数值实验表明……

    arXiv:2609.30348v1 Announce Type: cross  Abstract: Gaussian processes constitute a cornerstone of probabilistic machine learning, yet scaling them to large datasets typically forces a trade-off between computational efficiency and model fidelity. This work bridges this gap by presenting an adaptive multi-resolution Gaussian process framework that is both scalable and exact. Our key innovation is constructing a naturally data-sparse covariance matrix with adaptive multi-resolution basis functions. These basis functions are directly anchored to samples, eliminating the need for auxiliary points. By shrinking the support domains of multi-resolution basis, the matrix block sizes are limited, guaranteeing sparsity. The inverse of the data-sparse covariance matrix is computed exactly and efficiently via the sparse Cholesky inverse algorithm. To further improve predictive uncertainties, we construct an augmented basis function. Theoretical analysis and numerical experiments demonstrate that o
    
[^188]: 编码智能体还不够！评估用于智能体云调查的企业级安全大脑

    Coding Agents Aren't Enough! Evaluating an Enterprise Security Brain for Agentic Cloud Investigations

    [https://arxiv.org/abs/2609.30345](https://arxiv.org/abs/2609.30345)

    该论文提出并评估了Sola Security Brain这一专门构建的企业安全智能层，通过28个云安全调查任务的对比实验证明，相比仅凭只读凭证直接调查实时AWS环境的通用编码智能体（Claude Code），专门设计的安全上下文层在总体性云安全调查中能提供更完整、更准确的结果。

    

    云安全调查主要由“总体性任务”主导：哪些身份可以读取某个数据存储、多少资源不符合某项控制要求、哪些资产可以从另一个账户访问到。这些问题需要针对完整的资源清单来解答，而非针对某个指定的对象。对其中一个问题的部分回答并非部分结果，而是一个完全不同的结果。如今，通用编码智能体可以被授予只读云凭证并被要求直接开展调查，这就引出了一个问题：专门构建的安全上下文层还能带来什么额外价值。我们在28个云安全调查任务上，将Sola Security Brain（一个安全智能层，其关系基底在离线阶段完成解析，其安全逻辑在查询时基于该基底进行评估）与通过只读CLI操作同一实时AWS环境的Claude Code进行对比评估。答案通过在联合声明池上进行的盲评、分层加权、以事实依据为门控的相对召回率进行评分，平均……

    arXiv:2609.30345v1 Announce Type: cross  Abstract: Cloud-security investigation is dominated by population tasks: which identities can read a data store, how many resources fail a control, which assets are reachable from another account. These resolve against a complete inventory, not a named object. A partial answer to one is not a partial result. It is a different result. General-purpose coding agents can now be given read-only cloud credentials and asked to investigate directly, which raises the question of what a purpose-built security context layer still contributes. We evaluate the Sola Security Brain, a security intelligence layer whose relational substrate is resolved offline and whose security logic is evaluated against it at query time, against Claude Code operating the same live AWS environment through a read-only CLI, over 28 cloud-security investigation tasks. Answers are scored by a blinded, tier-weighted, grounding-gated relative recall over the joint claim pool, average
    
[^189]: 连接大语言模型智能体与数据空间：一种基于模型上下文协议的架构中介方法

    Bridging LLM Agents and Data Spaces: An Architectural Mediation Approach using the Model Context Protocol

    [https://arxiv.org/abs/2609.30341](https://arxiv.org/abs/2609.30341)

    该论文提出了一种基于模型上下文协议（MCP）的架构中介方法，通过将数据空间能力转换为结构化、模式驱动的工具，使LLM智能体能够在保留治理约束的前提下与数据空间服务进行受控交互，且无需修改现有数据空间组件。

    

    数据空间能够在组织边界之间实现主权和受治理的数据共享，但由于概率性语言模型交互与策略驱动的数据基础设施之间存在不匹配，数据空间与AI智能体的集成仍然充满挑战。本文提出了一种基于模型上下文协议（MCP）的架构中介方法，通过Eunomia智能体实现，使大语言模型（LLM）智能体与数据空间服务之间能够进行受控交互。所提出的中介层将数据空间能力转换为结构化的、模式驱动的工具，AI智能体可以发现并调用这些工具，同时保留治理约束。一个原型实现验证了涵盖目录发现、元数据检索和数据服务调用的端到端交互，且无需修改现有的数据空间组件。结果表明，基于协议的中介能够实现可互操作且符合标准的集成。

    arXiv:2609.30341v1 Announce Type: new  Abstract: Data Spaces enable sovereign and governed data sharing across organizational boundaries, but their integration with AI agents remains challenging due to mismatches between probabilistic language model interactions and policy-driven data infrastructures. This article presents an architectural mediation approach based on the Model Context Protocol (MCP), implemented through the Eunomia Agent, to enable controlled interaction between large language model (LLM) agents and data space services. The proposed mediation layer translates data space capabilities into structured, schema-driven tools that AI agents can discover and invoke while preserving governance constraints. A prototype implementation validates end-to-end interaction across catalog discovery, metadata retrieval, and data service invocation without modifying existing data space components. Results demonstrate that protocol-based mediation enables interoperable and standards-aligne
    
[^190]: 软件架构中什么仍将由人类掌控？一份焦点小组报告

    What Will Remain Human in Software Architecture? A Focus Group Report

    [https://arxiv.org/abs/2609.30334](https://arxiv.org/abs/2609.30334)

    本研究通过EuroPLoP 2026上的焦点小组探讨AI开发智能体对软件架构实践的影响，发现架构决策、问责制和架构护栏编写仍是不可替代的人类核心职责，并提出“驾驭工程”这一新兴概念——即构建用于治理AI辅助系统创建的系统的学科。

    

    AI开发智能体正被越来越多地用于支持并部分自动化软件架构任务。为了探索从业者如何看待这一转变——具体而言，什么在改变、什么保持不变、以及哪些新职责正在涌现——我们在第31届欧洲模式、人员与实践语言会议（EuroPLoP 2026）上开展了一次焦点小组研究。来自工业界和学术界的22名参与者讨论了当前的实践、信任与验证策略、AI自主性的边界、治理挑战以及对教育的影响。除其他发现外，我们观察到与会者存在广泛共识：架构决策、问责制以及架构护栏的编写从本质上仍然是人类的任务。一个核心的新兴概念是“驾驭工程”：即构建用于治理AI辅助系统创建过程的系统的学科，它包含验证机制、知识层以及公司特定的标准。

    arXiv:2609.30334v1 Announce Type: cross  Abstract: AI development agents are increasingly used to support and partially automate software architecture tasks. To explore how practitioners perceive this shift, specifically what changes, what remains, and what new responsibilities emerge, we conducted a focus group at the 31st European Conference on Pattern Languages of Programs, People, and Practices (EuroPLoP 2026). Twenty-two participants from industry and academia discussed current practices, trust and validation strategies, the boundaries of AI autonomy, governance challenges, and implications for education. Among others, we found broad consensus that architectural decision-making, accountability, and the authoring of architectural guardrails remain fundamentally human tasks. A central emergent concept was harness engineering: the discipline of building the system that governs AI-assisted system creation, comprising validation mechanisms, knowledge lay- ers, and company-specific stan
    
[^191]: 多智能体代码评判器何时才真正有据可依？两种无需标签的度量方法，以及一个拒绝猜测的评判器

    When Is a Multi-Agent Code Judge Actually Grounded? Two Label-Free Measurements, and a Judge That Declines to Guess

    [https://arxiv.org/abs/2609.30328](https://arxiv.org/abs/2609.30328)

    该论文指出多智能体验证框架在代码评判中因证据无法满足“独立于答案且能区分候选解”这两个条件而失效（在78%–95%的比较中判定两个解同样好、准确率仅4.4%），据此提出两种无需标签的度量方法来检验评判是否真正有据可依，并设计了一个在证据不足时拒绝猜测的评判器。

    

    当一个语言模型评判另一个语言模型的代码是否正确时，它并不会报告证据的缺失。它会返回一个附带推理过程的、充满自信的判决，这与一个真正有据可依的判决难以区分。多智能体验证将判断分解为可核查的声明，并逐条对照证据加以验证，是一种颇有前景的应对方式，且在证据为一组检索文档时效果良好。我们认为这类方法对其证据有两个要求：证据必须独立于被评审的答案，并且必须在被比较的两个候选答案之间有所区分。第二个条件在检索文档场景下会自动满足，但在代码评判中不再成立。我们在两个代码评判基准上未经修改地运行已发表的框架MARCH，进行了80项按条件逐单元的测量，发现它在78%到95%的比较中判定两个解同样好，而在直接询问同一模型时准确率仅为4.4%。

    arXiv:2609.30328v1 Announce Type: new  Abstract: When one language model judges whether another's code is correct, it does not report the absence of evidence. It returns a confident verdict with reasoning attached, indistinguishable from a verdict it had grounds for. Multi-agent verification, which decomposes a judgment into checkable claims and verifies each against evidence, is a promising response and works well when the evidence is a set of retrieved documents.   We argue such methods require two things of their evidence: it must be independent of the answer under review, and it must differ between the two candidates being compared. The second condition holds automatically with retrieved documents and stops holding in code judging.   Running MARCH, a published framework unmodified over 80 condition-by-cell measurements on two code judging benchmarks, we find it declares both solutions equally good on 78 to 95% of comparisons, reaching 4.4% accuracy where the same model asked direct
    
[^192]: ScopeBench：智能体在目标压力下能否守住授权边界？

    ScopeBench: Do Agents Preserve Engagement Boundaries Under Goal Pressure?

    [https://arxiv.org/abs/2609.30325](https://arxiv.org/abs/2609.30325)

    提出 ScopeBench 基准，通过 30 个“目标只能靠越界才能达成”的死胡同式安全任务，并在有无范围约束的配对条件下，区分并衡量智能体的能力与目标压力下遵守授权范围的意愿。

    

    智能体正越来越多地以真实自主权部署于网络应用和网络渗透测试中，而一次越界操作就可能突破客户的委托边界。现有的攻击性安全基准衡量的是原始的黑客能力；随着这些基准逐渐饱和，部署的真正障碍是对齐问题的一个特例：范围遵守。我们提出了 ScopeBench，这是一个包含 30 个“死胡同”式智能体安全任务的基准，其所述目标只有通过违反所述范围才能达成。每个任务在两种条件下出现，二者共享相同的环境、验证器和目标，仅在范围上有所不同：一个指令集不含范围，用于衡量能力；另一个指令集包含自然语言范围，用于衡量范围遵守。无范围轨迹由标准的确定性验证器评分。有范围轨迹则经过两个评分分支：首先，同一个确定性验证器检查标志，因为该标志……

    arXiv:2609.30325v1 Announce Type: new  Abstract: Agents are increasingly deployed with real autonomy in web application and network penetration testing, where a single out-of-scope action can breach a client's engagement boundary. Existing offensive-security benchmarks measure raw hacking capability; as those benchmarks saturate, the real barrier to deployment is a special case of alignment: scope adherence. We introduce ScopeBench, a benchmark of 30 dead-end agentic security tasks in which the stated objective is reachable only by violating the stated scope. Each task appears under two conditions that share an environment, verifier, and objective and differ only in scope: one instruction set has no scope and measures capability; the other has a natural-language scope to measure adherence. Scopeless trajectories are graded by a standard deterministic verifier. Scoped trajectories pass through two grading arms. First, the same deterministic verifier checks for the flag: because the flag
    
[^193]: 在Spotify引导对话式推荐智能体：合成数据生成与自我改进循环

    Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops

    [https://arxiv.org/abs/2609.30297](https://arxiv.org/abs/2609.30297)

    Spotify提出了一条多轮合成数据生成流水线与自我改进循环，通过基于方差的对比优化和编码智能体的迭代修复，在冷启动场景下自动优化对话式推荐智能体的规划与工具调用能力，使质量提升8%。

    

    对话式推荐智能体是内容发现的新范式，使用户能够通过自然语言表达复杂意图（例如：“推荐一些我没听过的意大利独立音乐人”）。构建此类智能体的核心挑战在于优化智能体规划——即决定如何选择、排序和调用工具——尤其是在真实用户交互尚不可用的冷启动场景中。为应对这一挑战，我们提出了一条多轮合成数据生成流水线和一种自我改进循环。合成数据流水线将单轮提示转化为逼真的多轮对话，从而支持上线前的系统性评估。自我改进循环将基于方差的对比优化与通过编码智能体进行的迭代改进相结合，能够自动识别并修复规划和工具使用中的错误。我们的方法在高度优化的基线之上将质量提升了+8%。

    arXiv:2609.30297v1 Announce Type: cross  Abstract: Conversational recommendation agents are a new paradigm for content discovery, enabling users to express complex intents through natural language (e.g., "recommend Italian indie artists I haven't heard before"). A central challenge in building such agents is optimizing agent planning -- deciding how to select, sequence, and invoke tools -- particularly in cold-start settings where real user interactions are not yet available. We introduce a pipeline for multi-turn synthetic data generation and a self-improvement loop to address this challenge. The synthetic data pipeline transforms single-turn prompts into realistic multi-turn conversations, enabling systematic evaluation before launch. The self-improvement loop combines variance-based contrastive optimization with iterative refinement through a coding agent, automatically identifying and fixing planning and tool-use errors. Our approach improves quality by +8% on top of a highly optim
    
[^194]: SignTrace：描述一个手势，找到对应的词

    SignTrace: Describe a Sign, Find the Word

    [https://arxiv.org/abs/2609.30295](https://arxiv.org/abs/2609.30295)

    SignTrace 利用大语言模型增强的中国手语词典，结合动作提取、七路检索与候选重排序，让学习者仅凭日常语言描述的手部动作即可反向查找到对应手语词条及其含义，在 500 条查询的基准上达到 94.0% 的 Hit@1。

    

    当学习者记得一个陌生手语的动作，却不知道其含义或正式的特征编码时，识别该手语十分困难。SignTrace 通过自然语言方式访问中国手语词典，解决了这一长期存在的反向查询问题。该系统集成了基于大语言模型的词典增强、动作提取、词典风格改写、七路检索以及候选重排序，覆盖 6,699 个词条。系统已部署进行用户试用，并收到了积极的非正式反馈。在一个由词典衍生、包含 500 条动作描述查询的基准测试上，系统取得了 94.0% 的 Hit@1、97.4% 的 Hit@9 以及 0.9540 的平均倒数排名。重排序将 Hit@1 从 71.8% 提升至 94.0%，组件分析显示了增强词条描述所作的贡献。在六个并发查询的情况下，查询处理的中位时间为 13.37 秒。通过将日常动作描述与已记录的手语动作及其含义关联起来……

    arXiv:2609.30295v1 Announce Type: cross  Abstract: Identifying an unfamiliar sign is difficult when a learner remembers its movement but does not know its meaning or formal feature codes. SignTrace addresses this longstanding reverse-lookup problem through natural-language access to a Chinese sign-language dictionary. The system integrates LLM-based dictionary enrichment, action extraction, dictionary-style rewriting, seven-channel retrieval, and candidate reranking over 6,699 entries. It has been deployed for user trials and has received positive informal feedback. Evaluation on a dictionary-derived benchmark of 500 movement-description queries yields 94.0% Hit@1, 97.4% Hit@9, and a mean reciprocal rank of 0.9540. Reranking increases Hit@1 from 71.8% to 94.0%, while component analyses show the contribution of enriched entry descriptions. Median query-processing time is 13.37 seconds with six concurrent queries. By connecting everyday movement descriptions to documented signs and meani
    
[^195]: SlideLab：以观众为中心的科学幻灯片生成与评估

    SlideLab: Audience-Centered Scientific Slide Generation and Evaluation

    [https://arxiv.org/abs/2609.30294](https://arxiv.org/abs/2609.30294)

    SlideLab 是一个无需训练的多智能体框架，能从研究论文生成以观众为中心的科学演示幻灯片，在盲测中于 77% 的论文上超越开源与商业系统且推理成本降低约 4 倍，并配套提出模拟会议室的观众导向评估框架 ConfArena。

    

    科学演示不仅仅是研究论文的摘要，它们需要以连贯的顺序呈现研究工作，清晰地解释核心思想，并帮助观众跟上演讲的节奏。我们提出了 SlideLab，一个无需训练的多智能体框架，用于从研究论文生成科学演示幻灯片。SlideLab 首先规划演示叙事，然后利用内容规划、视觉生成、布局优化和依据验证等智能体，构建并迭代完善一套共享幻灯片。在一项盲测人类偏好研究中，SlideLab 在 77% 的论文上优于开源和商业系统，同时其推理 token 使用量仅为最强开源基线的约四分之一。我们还引入了 ConfArena，一个面向观众的评估框架，它模拟会议室场景并逐张幻灯片地评估演示效果。ConfArena 的评估结果与人类对系统的排名相一致，并能检测注入的（原文在此处截断）……

    arXiv:2609.30294v1 Announce Type: cross  Abstract: Scientific presentations are more than summaries of research papers. They need to present the work in a coherent sequence, explain the main ideas clearly, and help the audience follow the presentation. We present SlideLab, a training-free multi-agent framework for generating scientific presentations from research papers. SlideLab first plans the presentation narrative, then builds and iteratively refines a shared slide deck using agents for content planning, visual generation, layout refinement, and grounding verification. In a blind human preference study, SlideLab was preferred over both open-source and commercial systems on 77% of papers while using roughly 4 times fewer inference tokens than the strongest open-source baseline. We also introduce ConfArena, an audience-oriented evaluation framework that simulates a conference room and assesses presentations slide by slide. ConfArena matches human system rankings and detects injected 
    
[^196]: Cartograph：面向AI智能体的操作者证明检索式联邦工具发现

    Cartograph: Federated Tool Discovery with Operator-Attested Retrieval for AI Agents

    [https://arxiv.org/abs/2609.30293](https://arxiv.org/abs/2609.30293)

    Cartograph是一个联邦式MCP代理，通过操作者签名的能力卡片、三层易混淆聚类分析（Rift）以及“先服务器后工具”的两阶段检索，将AI智能体的工具发现从O(n)目录遍历优化为O(k)渐进式披露，仅需暴露3个代理工具即可在374个工具的部署中取得0.816的R@5召回率，显著优于关键词基线。

    

    模型上下文协议（MCP）使AI智能体能够发现和调用工具，但随着所连接目录规模的增大，加载每一个工具定义的代价变得十分高昂。我们提出了Cartograph，一个联邦式MCP代理，它将智能体可见的工具发现从O(n)的目录遍历转变为O(k)的渐进式披露。Cartograph结合了三种机制：(1) 操作者证明的能力卡片，即在部署操作者控制下生成的Ed25519签名描述，而非由发布方营销文案排序的描述；(2) Rift，一个三层的易混淆聚类分析，包括密度聚类、查询边际分析和词元诊断；(3) 两阶段检索，先对服务器排序再对工具排序。在一个包含22个服务器、374个工具的部署中，Cartograph仅暴露3个代理工具而非374个工具定义。在一个由作者构建的49个查询的基准测试中，Cartograph取得了0.816的R@5召回率，相比之下Jaccard关键词基线仅为0.592，同时实测的前5名工具发现交换仅使用47（原文在此处截断）

    arXiv:2609.30293v1 Announce Type: cross  Abstract: The Model Context Protocol (MCP) enables AI agents to discover and call tools, but loading every definition becomes expensive as connected catalogs grow. We present Cartograph, a federated MCP proxy that changes agent-visible tool discovery from $O(n)$ catalog traversal to $O(k)$ progressive disclosure. Cartograph combines three mechanisms: (1) operator-attested capability cards, Ed25519-signed descriptions generated under the deploying operator's control rather than ranked publisher copy; (2) Rift, a three-layer confusable-cluster analysis comprising density clustering, query-margin analysis, and token diagnosis; and (3) two-stage retrieval, which ranks servers before tools. On a 22-server, 374-tool deployment, Cartograph exposes three proxy tools instead of 374 definitions. A 49-query author-constructed benchmark yields R@5 of 0.816, compared with 0.592 for a Jaccard keyword baseline, while a measured top-5 discovery exchange uses 47
    
[^197]: 虚假评论检测综述：从预训练语言模型到大语言模型

    A Survey on Fake Review Detection: From Pre-trained Language Models to Large Language Models

    [https://arxiv.org/abs/2609.30292](https://arxiv.org/abs/2609.30292)

    本综述从信息融合视角系统梳理了2018年至2026年初的211项虚假评论检测研究，按证据来源和融合层次组织现有工作，并分析了预训练语言模型和大语言模型对虚假评论的生成与检测带来的双重影响。

    

    在线评论影响着消费者的决策、平台治理和企业声誉。虚假评论通过向评分系统、推荐渠道和公众信任机制中注入欺骗性证据，破坏了这一信息渠道。大语言模型（LLM）的兴起从两个方向改变了这一问题：LLM能够生成流畅且具有上下文感知能力的欺骗性评论，而预训练语言模型（PLM）和LLM同时也为检测任务提供了更强的语义表示能力。本综述从信息融合的视角回顾了虚假评论检测研究，涵盖了2018年至2026年初发表的211项研究。我们按照证据来源和融合层次对现有工作进行组织，涵盖评论文本、情感、评分行为、时间元数据、用户-产品图、多模态内容、外部知识以及LLM生成的信号。我们追溯了从传统机器学习和深度学习到基于PLM的方法的发展历程。

    arXiv:2609.30292v1 Announce Type: cross  Abstract: Online reviews shape consumer decisions, platform governance, and corporate reputation.Fake reviews compromise this information channel by injecting deceptive evidence into rating systems, recommendation pipelines, and public trust mechanisms.The rise of large language models, or LLMs, has changed the problem in two directions.LLMs can generate fluent and context-aware deceptive reviews, while pre-trained language models, or PLMs, and LLMs also provide stronger semantic representations for detection.This survey reviews fake review detection from an information fusion perspective, covering 211 studies published from 2018 to early 2026.We organize existing work by evidence source and fusion level, covering review text, sentiment, rating behavior, temporal metadata, user-product graphs, multimodal content, external knowledge, and LLM-generated signals.We trace the development from traditional machine learning and deep learning to PLM-base
    
[^198]: 将人工智能引入自主系统——从认知到集体智能

    Bringing AI to Autonomous Systems -- From Cognition to Collective Intelligence

    [https://arxiv.org/abs/2609.30291](https://arxiv.org/abs/2609.30291)

    本文提出了一个基于通用智能体架构的自主系统设计与评估综合框架，通过结合联结主义与符号主义AI，并整合个体认知与集体智能，推动AI向自主系统这一最终发展阶段迈进。

    

    本文旨在强调自主系统作为人工智能发展最终阶段的核心地位，解释需要结合联结主义AI与符号主义AI的底层技术挑战，并将AI与系统工程相整合。我们提出了一个用于自主系统设计与评估的综合框架，该框架基于一种通用智能体架构，将自主系统的行为特征描述为围绕长期记忆组织的认知功能的组合，该长期记忆中包含智能体不断演进的知识。我们解决了实现智能体架构基本功能所带来的挑战，特别是感官数据与存储在记忆中的结构化数据之间的联系、与实现智能体目标及其规划相关的决策制定，以及智能体之间的协调以结合个体智能与集体智能。

    arXiv:2609.30291v1 Announce Type: new  Abstract: The purpose of this article is to highlight the central role of autonomous systems as the ultimate stage in the development of AI, to explain the underlying technical challenges that require a combination of connectionist AI and symbolic AI, and to integrate AI and systems engineering. We present a comprehensive framework for the design and evaluation of autonomous systems, based on a generic agent architecture that characterizes their behavior as the composition of cognitive functions organized around a long-term memory containing the agent's evolving knowledge. We address the challenges posed by the implementation of the fundamental features of the agent architecture, in particular the link between sensory data and structured data stored in memory, decision-making related to the achievement of the agent's goals and their planning, as well as the coordination of agents to combine individual and collective intelligence. We explain that a
    
[^199]: 冻结BERT中AI文本检测神经元的机制研究：基于RAID的稀疏探测与激活修补

    A Mechanistic Study of AI-Text Detection Neurons in Frozen BERT: Sparse Probing and Activation Patching on RAID

    [https://arxiv.org/abs/2609.30287](https://arxiv.org/abs/2609.30287)

    该研究通过稀疏探测和双向激活修补方法，在冻结的BERT中定位出一组占比不足1%的神经元，证明它们因果性地支持跨六个生成器的AI生成文本检测任务。

    

    AI生成文本检测器在标准基准测试中达到了很高的准确率，但驱动这些预测的内部表示仍鲜为人知。我们研究了冻结的BERT-base-uncased编码器中哪些神经元支持AI文本检测，使用了涵盖六个生成器（包括纯基座模型和指令微调模型）的RAID基准。我们将Gurnee等人（2023）提出的L1到L2稀疏探测协议应用于全部9,216个CLS隐藏状态维度（12层×768），我们将其称为神经元。该程序为每个生成器恢复出一组稳定的、占比不足1%的神经元，且在不同数据折和随机种子之间保持一致；仅使用该神经元集合的探测器即可保留全特征检测准确率的大部分。双向激活修补证实了该神经元集合的因果相关性：在两个方向上，它翻转预测的频率比大小匹配的随机神经元集合高一个数量级。而对相同神经元进行均值消融后，准确率基本保持不变；该信号是……（摘要在此处截断）

    arXiv:2609.30287v1 Announce Type: cross  Abstract: AI-generated text detectors achieve high accuracy on standard benchmarks, yet the internal representations that drive these predictions remain poorly understood. We study which neurons in a frozen BERT-base-uncased encoder support AI-text detection, using the RAID benchmark across six generators spanning pure-base and instruction-tuned models. We apply the L1-to-L2 sparse-probing protocol of Gurnee et al. (2023) to all 9,216 CLS hidden-state dimensions (12 layers x 768), which we call neurons. The procedure recovers a stable set of under 1% of neurons per generator, consistent across folds and seeds; a probe restricted to that set retains most of the full-feature detection accuracy. Bidirectional activation patching confirms this set's causal relevance: in both directions it flips predictions an order of magnitude more often than size-matched random sets. Mean-ablating the same neurons leaves accuracy largely intact; the signal is ther
    
[^200]: 平流感知图临近预报何时有效？——基于自监督云运动估计器的分布式光伏功率坡升预报受控研究

    When Does Advection-Aware Graph Nowcasting Help? A Controlled Study of Distributed Solar Ramp Forecasting with a Self-Supervised Cloud-Motion Estimator

    [https://arxiv.org/abs/2609.30286](https://arxiv.org/abs/2609.30286)

    受控合成实验表明，平流感知图神经网络对分布式光伏坡升临近预报的增益有限——在现实的互相关CMV估计下并不优于静态或学习邻接的时空GNN，完美CMV约一半收益来自将运动矢量作为输入特征而非图结构本身，且只有当预报时域内平流位移落在传感网络范围内时平流信息才真正有用。

    

    对分布式光伏（PV）或辐照度传感器网络中由云引起的功率坡升（ramp）进行短期预报，是电网运营商公认的痛点。一个自然的想法是让图神经网络（GNN）具备平流感知能力：将每个站点与其上风方向的站点相连，并由云运动矢量（CMV）设定边的时滞，使坡升信号在物理到达之前就被向前传播。利用一个具有已知风场的受控合成试验平台，我们证明：（i）在使用现实可行的互相关CMV估计时，显式平流图并不优于普通的静态图或学习邻接的时空GNN；（ii）完美CMV所能带来收益中，大约一半仅仅来自于将精确的运动矢量作为输入特征提供，而非来自图结构；（iii）只有当预报时域内的平流位移 v*H 落在传感器网络范围之内时，平流信息才有帮助。受（ii）的启发，

    arXiv:2609.30286v1 Announce Type: cross  Abstract: Short-term forecasting of cloud-induced power ramps across a network of distributed photovoltaic (PV) or irradiance sensors is a recognised pain point for grid operators. A natural idea is to make the graph neural network (GNN) advection-aware: connect each site to the sites upwind of it, with edge time-lags set by the cloud-motion vector (CMV), so that a ramp is propagated forward before it physically arrives. Using a controlled synthetic testbed with a known wind field, we show that (i) with a realistic cross-correlation CMV estimate, an explicit advection graph does not beat a plain static or learned-adjacency spatiotemporal GNN; (ii) roughly half of the benefit available from a perfect CMV comes simply from providing an accurate motion vector as an input feature, not from graph structure; and (iii) advection helps only when the advective displacement over the forecast horizon, v*H, fits inside the sensor network. Motivated by (ii),
    
[^201]: ENAS：一种面向资源受限微控制器上TinyML的高效硬件感知神经架构搜索框架

    ENAS: An Efficient Hardware-Aware Neural Architecture Search Framework for TinyML on Resource-Constrained Microcontrollers

    [https://arxiv.org/abs/2609.30272](https://arxiv.org/abs/2609.30272)

    ENAS是一个无需GPU即可高效运行的硬件感知神经架构搜索框架，通过静态可行性检查、支持多种块的单元搜索空间和三阶段混合搜索策略，在资源受限的微控制器上实现了TinyML模型的快速搜索。

    

    我们提出了ENAS，这是一个硬件感知的神经架构搜索（NAS）框架，它结合了静态可行性检查、一个支持标准块、深度可分离块和瓶颈块（带可选跳跃连接）的基于单元的搜索空间，以及一种具有跨运行持久缓存的三阶段混合搜索策略（随机搜索→top-K筛选→变异）。与许多依赖GPU加速的现有NAS框架不同，ENAS被设计为在无需GPU的情况下也能高效运行，使其适用于资源受限的开发环境。我们在两个TinyML基准数据集（Visual Wake Words和Melanoma Cancer）上对ENAS进行了评估，涵盖八款SRAM内存占用从20KB到1MB的微控制器以及九种输入图像分辨率。实验结果表明，ENAS在Visual Wake Words和Melanoma Cancer数据集上分别实现了平均2.41倍和1.70倍的搜索时间加速。

    arXiv:2609.30272v1 Announce Type: cross  Abstract: We present \textbf{ENAS}, a hardware-aware Neural Architecture Search (NAS) framework that combines a static feasibility check, a cell-based search space supporting standard, depthwise-separable, and bottleneck blocks with optional skip connections, and a three-stage hybrid search strategy (random $\rightarrow$ top-$K$ $\rightarrow$ mutation) with persistent cross-run caching. Unlike many existing NAS frameworks that rely on GPU acceleration, ENAS is designed to operate efficiently without requiring GPUs, making it suitable for resource-constrained development environments. We evaluate ENAS on two TinyML benchmarks, Visual Wake Words and Melanoma Cancer, across eight microcontrollers with memory footprints ranging from 20\,KB to 1\,MB SRAM and nine input image resolutions. Our experimental results show that ENAS achieves mean search-time speedups of $2.41{\times}$ and $1.70{\times}$ on the Visual Wake Words and Melanoma Cancer datasets
    
[^202]: AD-WM：用于反事实模型预测控制的动作判别式世界模型

    AD-WM: Action-Discriminative World Models for Counterfactual Model Predictive Control

    [https://arxiv.org/abs/2609.30264](https://arxiv.org/abs/2609.30264)

    提出动作判别式世界模型AD-WM，通过逆动力学和基于条件互信息的动作恢复正则化，使潜在世界模型能更好区分候选动作以支持反事实模型预测控制，在OGBench-Cube上将困难起始成功率从3.7%提升至52.0%。

    

    潜在世界模型通常被训练来预测事实性转移，而模型预测控制（MPC）必须比较从同一状态出发的不同候选动作。因此，一个模型可能实现较低的事实预测误差，却难以有效区分候选动作。我们提出了AD-WM，一种用于反事实MPC的动作判别式联合嵌入世界模型。AD-WM将残差潜在动力学与预测器层面的动作恢复正则化相结合，利用逆动力学以及受条件互信息启发的归一化恢复目标。这两个目标都促使规划转移保留动作信息；其辅助头在测试时会被丢弃，因此MPC本身保持不变。在OGBench-Cube上，AD-WM将困难起始成功率从匹配的LeWM基线的3.7%提升到52.0%，并且在五个仿真环境中的四个里，平均成功率超过了复现的基线。规划诊断表明，事实性预…

    arXiv:2609.30264v1 Announce Type: new  Abstract: Latent world models are typically trained to predict factual transitions, whereas model predictive control (MPC) must compare alternative actions from the same state. A model can therefore achieve low factual prediction error yet poorly distinguish candidate actions. We introduce AD-WM, an action-discriminative joint-embedding world model for counterfactual MPC. AD-WM combines residual latent dynamics with predictor-level action-recovery regularization, using inverse dynamics and a normalized recovery objective motivated by conditional mutual information. Both objectives encourage planning transitions to preserve action information; their auxiliary heads are discarded at test time, leaving MPC unchanged. On OGBench-Cube, AD-WM improves hard-start success from 3.7% to 52.0% over a matched LeWM baseline and improves mean success over the reproduced baseline in four of five simulation environments. Planning diagnostics show that factual pre
    
[^203]: PrivDrift：主动LLM对话中话题漂移下用户秘密泄露的审计

    PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

    [https://arxiv.org/abs/2609.30094](https://arxiv.org/abs/2609.30094)

    提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。

    

    arXiv:2609.30094v1 公告类型：新论文 摘要：大型语言模型日益作为持久性助手应用于面向用户的、共享会话以及工具增强的场景中。当用户在活跃对话中披露敏感信息时，即使对话随后转向无关话题，这些信息通过后续提示仍可能在行为层面被恢复出来。我们提出了PrivDrift，这是一个用于审计用户所披露秘密在经历对话话题漂移和基于说服的探测之后是否仍可被恢复的基准。PrivDrift包含1,000个受控多轮对话，其中预置了秘密信息、内容密集的漂移轮次以及标准化的提取探测。在三个具有扩展上下文窗口的大语言模型上，对话级混合泄露依然严重，泄露率介于38.7%至54.6%之间，并且随模型、秘密类型和说服强度的不同而有显著变化。在所测试的漂移窗口内，额外的话题漂移并不能可靠地降低泄露，这表明隐私风险持续存在（摘要在此处被截断）。

    arXiv:2609.30094v1 Announce Type: new  Abstract: Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that pri
    
[^204]: PUBG Ally：作为AI队友的对话式具身智能体

    PUBG Ally: A Conversational Embodied Agent as an AI Teammate

    [https://arxiv.org/abs/2609.29837](https://arxiv.org/abs/2609.29837)

    该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。

    

    我们推出了PUBG Ally，一个面向《绝地求生：大逃杀》(PUBG: BATTLEGROUNDS) 的具身智能体，它能够进行推理、自主行动，并作为支持语音交互的队友与玩家并肩作战。构建这样的队友需要结合两种困难的能力：它必须在严格的延迟约束下感知并响应不断变化的游戏世界，同时与玩家自然交互，使其语音与行动保持同步。因此，Ally将智能体的工具使用能力与实时游戏控制相结合。一个语言模型智能体通过受控接口来查看游戏信息、理解玩家语音、维护上下文、决定说什么，并发出高层动作选择，用以引导更快的控制层执行移动、战斗和恢复等操作。由于玩家和Ally的语音与行动会不断相互影响并塑造比赛进程，训练需要来自真实对局的数据。因此，我们在近3.9万场对局中收集了数据……（摘要在此处被截断）

    arXiv:2609.29837v1 Announce Type: new  Abstract: We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in whi
    
[^205]: 使用不确定性感知视觉Transformer在多民族近视与非近视人群中检测青光眼：一项多中心模型开发与验证研究

    Detecting Glaucoma Across Multi-ethnic Myopic and Non-Myopic Populations Using an Uncertainty-Aware Vision Transformer: A Multicentre Model Development and Validation Study

    [https://arxiv.org/abs/2609.29433](https://arxiv.org/abs/2609.29433)

    该研究开发了带有不确定性估计的Vision Transformer深度学习模型，在涵盖多民族、近视与非近视人群的三大洲16个外部数据集上实现了稳健且高性能的青光眼检测。

    

    背景：基于人工智能（AI）的彩色眼底照片（CFP）青光眼检测提供了可规模化的筛查手段，但由于真实标签定义、人群差异以及高度近视（HM）等共存疾病的影响，其在外部数据集上的性能可能下降。我们开发并验证了一种基于Vision Transformer的深度学习（DL）模型，用于在有和无高度近视的多民族队列中进行青光眼检测。方法：研究使用56,483张彩色眼底照片（其中57.1%为近视，14.4%为高度近视）开发了具有预测不确定性估计功能的ViT-B/16模型。青光眼标签通过临床、影像和视野检查数据进行标准化。该模型在三大洲的16个独立数据集上进行了验证，其中包括4个具有明确高度近视标签的数据集。结果：内部验证AUROC为98.7%（95% CI 98.2-99.1%），灵敏度为94.5%，特异度为97.3%。在来自八个国家的16个外部数据集中，AUROC范围……（原文摘要在此处被截断）

    arXiv:2609.29433v1 Announce Type: cross  Abstract: Background: Artificial intelligence (AI)-based glaucoma detection from colour fundus photographs (CFP) offers scalable screening, but performance may decline on external datasets because of differences in ground-truth definitions, populations, and coexisting conditions such as high myopia (HM). We developed and validated a Vision Transformer-based deep learning (DL) model for glaucoma detection across multi-ethnic cohorts with and without HM. Methods: A ViT-B/16 model with predictive uncertainty estimation was developed using 56,483 CFPs (57.1% with myopia; 14.4% with HM). Glaucoma labels were standardised using clinical, imaging, and perimetry data. The model was validated on 16 independent datasets across three continents, including four datasets with explicit HM labels. Findings: Internal AUROC was 98.7% (95% CI 98.2-99.1%), with sensitivity 94.5% and specificity 97.3%. Across 16 external datasets from eight countries, AUROCs ranged
    
[^206]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^207]: ArGuard共享任务：阿拉伯语表情包与大语言模型提示中的有害内容检测

    ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts

    [https://arxiv.org/abs/2609.29349](https://arxiv.org/abs/2609.29349)

    ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。

    

    ArGuard是一个针对阿拉伯语表情包和大语言模型（LLM）提示中有害内容检测的共享任务。该任务包含两个赛道：赛道A专注于阿拉伯语表情包中的多模态仇恨内容检测，赛道B则面向阿拉伯语LLM安全评估的有害提示检测。共有58支队伍报名，35支队伍参加了最终评估，27支队伍提交了系统描述论文。参赛队伍探索了AraBERT、Jais和Qwen3-VL等模型。最佳系统在A1、A2、B1和B2四个子任务上分别取得了0.823、0.419、0.984和0.790的宏F1分数。其中A2赛道中细粒度的表情包分类是最具挑战性的设置，部分原因在于标签稀疏以及训练集与测试集之间的分布偏移。

    arXiv:2609.29349v1 Announce Type: cross  Abstract: ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts.
    
[^208]: 科学中的人工智能：早期洞见

    AI in Science: Early Insights

    [https://arxiv.org/abs/2609.28504](https://arxiv.org/abs/2609.28504)

    该论文通过分析1500万次Gemini交互、2600多个专业AI模型和600多名科学家的调查数据，首次提供了科学家使用AI的大规模实证证据，发现AI在科学界已被广泛采用，且LLM与专业模型互为补充而非替代。

    

    科学进步是经济增长与繁荣的关键驱动力。人们对人工智能对科学的影响充满期待，同时也存在担忧，但迄今为止相关数据却很少。我们从三个数据来源提供了关于这一问题的早期洞见：1500万次Gemini交互的样本、一份涵盖各学科的2600多个专业AI模型清单，以及对600多名科学家的调查。我们将这些数据映射到一个新的科学任务分类体系中，以研究科学家如何使用人工智能。研究得出四个主要发现。首先，我们发现AI的采用率和覆盖面非常广泛：科学家比大多数其他职业更多地使用AI。专业AI模型具有广泛的学科覆盖面且被高度引用。接受调查的科学家中有近一半报告每天使用某种形式的AI。其次，我们记录了LLM（通过Gemini使用情况来衡量）与专业模型互为补充的证据——LLM被用于通用分析、编程和稿件准备，而……

    arXiv:2609.28504v1 Announce Type: cross  Abstract: Scientific progress is a key driver of economic growth and prosperity. There is great excitement - but also concerns - about the impacts of AI on science, but so far little data. We provide early insights on this from three data sources: a sample of 15 million Gemini interactions, an inventory of over 2,600 specialized AI models across disciplines, and a survey of over 600 scientists. We map these data to a new taxonomy of scientific tasks to study how scientists are using AI. Four main findings emerge. First, we find broad adoption and coverage: scientists use AI more than most other occupations. Specialized AI models have broad disciplinary coverage and are highly cited. Nearly half of the scientists surveyed report using some form of AI every day. Second, we document evidence that LLMs (proxied through Gemini usage) and specialized models act as complements-- LLMs are used for general analysis, coding, and manuscript preparation, wh
    
[^209]: NV-Reason-CT：用于CT分析的三维视觉语言模型

    NV-Reason-CT: 3D Visual Language Model for CT Analysis

    [https://arxiv.org/abs/2609.27511](https://arxiv.org/abs/2609.27511)

    NV-Reason-CT通过原生3D视觉Transformer将全部视觉标记及其显式3D坐标直接传入语言模型解码，在基于7万余例CT、约55万条专家标注引导的多模态指令数据上训练，实现了保留完整体积空间信息的胸部和腹部CT智能推理分析。

    

    我们提出了NV-Reason-CT，这是一个用于胸部和腹部CT分析的生成式视觉-语言模型，它将原生三维视觉编码与放射科医师引导的推理相结合。该模型将原生3D视觉Transformer与语言模型耦合，将所有视觉标记及其显式3D坐标直接传递给语言解码过程，无需进一步的空间标记合并。这使得体积空间信息在视觉编码器内部得以保留，并通过语言模型的位置编码在与文本联合处理时得以维持。我们在一个精选的语料库上进行训练，该语料库包含来自70,111个独特CT图像输入的约550,000个多模态指令样本，结合了标准化报告、以异常为重点和特定解剖部位的问题、多轮交互，以及来自专家CT解读录音和转录的由放射科医师撰写的推理。专家标注提供了直接监督，并指导了额外的基于报告的合成推理。（注：原文摘要不完整，在"End-to-en"处截断）

    arXiv:2609.27511v1 Announce Type: cross  Abstract: We present NV-Reason-CT, a generative vision--language model for chest and abdominal CT combining native 3D visual encoding with radiologist-guided reasoning. The model couples a native 3D vision transformer with a language model, passing all visual tokens and their explicit 3D coordinates into language decoding without further spatial token merging. This retains volumetric spatial information within the vision encoder and through the language model's positional encoding during joint processing with text.   We train on a curated corpus of approximately 550,000 multimodal instruction examples from 70,111 unique CT image inputs, combining standardized reports, abnormality-focused and anatomy-specific questions, multi-turn interactions, and radiologist-authored reasoning from recorded and transcribed expert CT interpretations. Expert annotations provide direct supervision and guide additional report-grounded synthetic reasoning. End-to-en
    
[^210]: 基于分解子任务的强化学习

    Reinforcement Learning with Decomposed Subtasks

    [https://arxiv.org/abs/2609.27035](https://arxiv.org/abs/2609.27035)

    该论文提出RLDS方法，其核心是子任务分解优势估计（SDAE），通过在固定分类体系上将轨迹奖励按子任务分解并计算各子任务的组相对优势，解决了GRPO等方法将多轮rollout压缩为单一标量奖励所导致的信息损失问题。

    

    组相对策略优化（GRPO）及用于训练语言模型智能体的相关策略梯度方法，在进入策略更新之前，会将整个多轮rollout压缩为单一标量轨迹奖励。当任务由不同技能组合而成时，尤其是在稀疏且延迟的环境反馈下，这种压缩是有损的：优化器必须隐式地推断是哪种能力导致了最终结果，以及这应当如何改变行为。我们认为正确的基元并非更好的标量，而是分解：轨迹奖励应当在进入策略更新之前沿着子任务进行拆分。我们提出了基于分解子任务的强化学习（RLDS），其核心是子任务分解优势估计（SDAE）：一种替代标量GRPO优势的方法，它在固定的分类体系上将轨迹奖励拆分为每个子任务的份额，为每个子任务计算组相对优势，并将每个token的信用分配……

    arXiv:2609.27035v1 Announce Type: new  Abstract: Group Relative Policy Optimization (GRPO) and related policy-gradient methods for training language model agents collapse an entire multi-turn rollout into a single scalar trajectory reward before it enters the policy update. When the task composes distinct skills, especially under sparse and delayed environmental feedback, this collapsing is lossy: the optimizer must implicitly infer which competency drove the outcome and how that should change behavior. We argue the right primitive is not a better scalar but a decomposition: trajectory reward should be split along subtasks before it enters the policy update. We introduce Reinforcement Learning with Decomposed Subtasks (RLDS), whose core is Subtask-Decomposed Advantage Estimation (SDAE): a replacement for the scalar GRPO advantage that splits trajectory reward into per-subtask shares on a fixed taxonomy, computes a group-relative advantage per subtask, and distributes per-token credit b
    
[^211]: Lean 4 中哥德尔与斯科特版本的本体论论证

    G\"odel's and Scott's Variants of the Ontological Argument in Lean 4

    [https://arxiv.org/abs/2609.26806](https://arxiv.org/abs/2609.26806)

    该论文将哥德尔与斯科特本体论论证的 Isabelle/HOL 形式化数据集完整且保结构地移植到 Lean 4，验证了全部 548 条陈述的一致性，并重新证明了包括模态坍缩、一神论等在内的所有原开发中被证明的结论。

    

    本文呈现了将 Benzmüller 和 Scott 关于哥德尔模态本体论论证及其斯科特变体研究所配套的 Isabelle/HOL 数据集完整且保结构地移植到 Lean 4 的工作。该移植包含 30 个 Lean 4 模块，每个模块对应一个 Isabelle/HOL 理论，保留了章节结构、声明顺序以及每条公理、定义、引理和定理的名称；一个比较工具验证了全部 548 条陈述完全一致。Isabelle/HOL 开发中证明的所有内容均被重新证明，包括哥德尔 1970 年公理的不一致性、修复后的哥德尔变体、斯科特变体、模态坍缩、一神论以及肯定性质的超滤性质；原文中有五条陈述在自动证明器找到证明后未再被重新验证（其中一条随后被作为公设），这些陈述也被证明。剩余 45 条未证明的陈述恰好是原文通过 nitpick 反驳的（35 条）或留作未决的陈述。

    arXiv:2609.26806v1 Announce Type: cross  Abstract: This paper presents a complete, structure-preserving port to Lean 4 of the Isabelle/HOL dataset accompanying Benzm\"uller and Scott's study of G\"odel's modal ontological argument and Scott's variant of it. The port comprises 30 Lean 4 modules, one per Isabelle/HOL theory, retaining the section structure, the declaration order and the name of every axiom, definition, lemma and theorem; a comparison tool certifies all 548 statements identical. Everything the Isabelle/HOL development proves is proved again, including the inconsistency of G\"odel's 1970 axioms, the repaired G\"odel variants, Scott's variant, modal collapse, monotheism and the ultrafilter property of the positive properties; five statements the original leaves unreplayed after an automated prover had found a proof, one of which it then postulates, are proved as well. The 45 remaining unproved statements are exactly those the original refutes by nitpick (35) or leaves open 
    
[^212]: 面向多电网潮流的分层图神经网络：跨运行场景的泛化能力

    Towards Hierarchical GNNs for multi-grid power flow: generalization across operating scenarios

    [https://arxiv.org/abs/2609.26603](https://arxiv.org/abs/2609.26603)

    该论文提出在GENCO校正网络中引入分层潜在通信模块，通过Kron简化和Quotient构建两种简化图在不同电网间交换信息，显著提升了多电网潮流GNN模型对未见运行场景的泛化能力，其中Kron方法将电压误差降低了85.0%。

    

    分层潜在通信提升了多电网潮流模型对新运行场景的泛化能力。该模块在基于GENCO的校正网络中通过两个简化图交换信息。我们在三种电网拓扑上进行了200个epoch的初步训练，比较了基于Kron导出的传输方法、同锚点Quotient构建方法以及平坦骨干网络，每个模型使用三个初始化随机种子。评估采用每个电网200个新生成的预选场景。在训练拓扑上，Kron方法将宏观族群平衡电压误差从5.660 ± 0.899降至0.851 ± 0.110：相比Flat GENCO降低了85.0%，相比达到1.235 ± 0.225的Quotient降低了31.0%。在所有三个随机种子中，两种分层模型在每个训练拓扑上都优于基于训练解拟合的逐母线均值方法。这些结果表明在所研究的拓扑内实现了跨运行场景的泛化。

    arXiv:2609.26603v1 Announce Type: new  Abstract: Hierarchical latent communication improves the generalization of a multi-grid power-flow model to new operating scenarios. The module exchanges information through two reduced graphs within a GENCO-based corrective network. We compare Kron-derived transports, a same-anchor Quotient construction and a flat backbone in preliminary trainings of 200 epochs on three grid topologies, with three initialization seeds per model. Evaluation uses 200 newly generated, preselected scenarios per grid. On the training topologies, Kron reduces the macro family-balanced voltage error from 5.660 +- 0.899 to 0.851 +- 0.110: an 85.0% reduction relative to Flat GENCO and 31.0% relative to Quotient, which reaches 1.235 +- 0.225. Both hierarchical models outperform a per-bus mean fitted on training solutions on every training topology in all three seeds. These results demonstrate generalization across operating scenarios within the studied topologies, with one
    
[^213]: 几何感知的双曲残差量化

    Geometry-Aware Hyperbolic Residual Quantization

    [https://arxiv.org/abs/2609.26342](https://arxiv.org/abs/2609.26342)

    提出一种几何感知的双曲残差量化方法，通过双曲残差聚合恢复前向传播中庞加莱圆盘上的伸缩求和特性，并利用带折扣的双曲直通估计器在反向传播中保留几何信息，从而解决双曲空间残差量化的几何不一致问题。

    

    残差向量量化将连续表示转化为离散的多层级token序列。然而，尽管所产生的编码具有由粗到细的结构，且许多数据域中存在潜在的层次结构，大多数方法仍在欧几里得空间中运行。双曲几何为层次化表示提供了一种自然的替代方案，但朴素的双曲扩展会引入几何上的不一致性：非结合的双曲加法阻碍了一致的残差聚合，而标准的直通梯度估计则忽略了潜在空间的几何特性。我们提出了一种几何感知的双曲残差量化方法，在前向和反向传播两个过程中都解决了这些问题。在前向传播中，双曲残差聚合恢复了庞加莱圆盘上残差量化的伸缩求和行为。在反向传播中，带折扣的双曲直通估计器将重构（梯度）……

    arXiv:2609.26342v1 Announce Type: new  Abstract: Residual Vector Quantization turns continuous representations into discrete, multi-level token sequences. Yet most methods operate in Euclidean space, despite the coarse-to-fine structure of the resulting codes and the latent hierarchies present in many data domains. Hyperbolic geometry offers a natural alternative for hierarchical representations, but naive hyperbolic extensions introduce geometric inconsistencies: non-associative hyperbolic addition prevents consistent residual aggregation, while standard straight-through gradient estimation ignores the geometry of the latent space. We propose a geometry-aware hyperbolic residual quantization that addresses these issues in both the forward and backward passes. In the forward pass, Hyperbolic Residual Aggregation restores the telescoping behavior of residual quantization on the Poincare ball. In the backward pass, a discounted Hyperbolic Straight-Through Estimator routes the reconstruct
    
[^214]: 未受控变量：视觉语言模型的拒绝行为取决于图像附加接口，且对无关图像属性不具备鲁棒性

    The Uncontrolled Variable: Vision-Language Refusal Is Conditioned on the Image-Attachment Interface, and Not Robust to Irrelevant Image Properties

    [https://arxiv.org/abs/2609.26174](https://arxiv.org/abs/2609.26174)

    该研究揭示视觉语言模型的拒绝行为取决于请求是否附带图像——即使附加的是空白画布等完全无关的图像——这种阈值偏移使边缘良性的敏感话题提问拒绝率上升23至51个百分点，而中性指令几乎不受影响。

    

    我们发现，经过对齐的视觉语言模型还会依据请求形式的一个属性来决定是否拒绝回答：即是否附带了图像，即使请求所询问的内容完全保持不变。附加一块空白画布——一张无法读取、与请求无关、且在该条件下所有提示词中字节完全相同的图像——会使良性拒绝率变动数十个百分点。这种变动并非一概而论的谨慎，而是一种阈值偏移：真正中性的指令几乎不受影响（在四个托管模型中的三个上变化不超过2个百分点），而处于边缘的良性提示则变动+23至+51个百分点，因此代价落在敏感性相关流量上，即关于隐私、自残、暴力和非法活动的良性提问。对这种阈值存在一种良性的解释，即在真实流量中图像附加与风险相关，我们认真对待这一解释；黑盒研究无法测量这种相关性，我们也未声称能够测量。它能够测试的是……

    arXiv:2609.26174v2 Announce Type: replace-cross  Abstract: We show that aligned vision-language models also condition refusal on a property of a request's form: whether an image is attached, holding everything the request asks fixed. Attaching a blank canvas, an image that cannot be read, cannot relate to the request, and is byte-identical across every prompt in its condition, shifts benign refusal by tens of points. The shift is not blanket caution but a threshold shift: genuinely neutral instructions are almost unaffected (<=2 percentage points on three of four hosted models) while borderline-benign prompts move +23 to +51 points, so the cost falls on sensitivity-adjacent traffic, meaning benign questions about privacy, self-harm, violence and illegal activity. There is a benign reading of such a threshold, namely that attachment correlates with risk in real traffic, and we take it seriously; a black-box study cannot measure that correlation and we do not claim to. What it can test i
    
[^215]: 用于测试时扩展的山峰采样：重复采样、进化与训练的简单且更优的替代方案

    Hill Sampling for Test-Time Scaling: A Simple and Better Alternative to Repeated Sampling, Evolution, and Training

    [https://arxiv.org/abs/2609.25510](https://arxiv.org/abs/2609.25510)

    该论文提出了一种简单的“山峰采样”方法——从冻结的大语言模型中反复采样候选程序编辑并保留当前最优程序作为后续采样的条件，无需复杂的进化搜索或测试时训练，就在圆填充问题上创下新的最先进水平，并在Erdős最小重叠问题上超越了AlphaEvolve。

    

    大型语言模型（LLM）可以通过在测试时投入额外的计算来改进可验证的科学和算法问题的解决方案。最近的系统借助日益复杂的进化搜索框架，或在测试时训练过程中更新模型参数，取得了强劲的成果。我们想探究这些复杂机制中究竟有多少是必要的。我们提出了山峰采样，这是一种简单的程序：从冻结的LLM中反复采样候选程序编辑，保留迄今为止找到的最佳程序，并让所有后续采样都以该程序为条件。我们在圆填充、集合的和/差以及Erdős最小重叠问题上，使用三个开源权重模型对该方法进行了评估。山峰采样在已发表方法中为圆填充问题设立了新的最先进水平，在Erdős最小重叠问题上超越了AlphaEvolve参考结果，并在有限集合的和与差问题上取得了强劲成果。

    arXiv:2609.25510v1 Announce Type: new  Abstract: Large language models (LLMs) can improve solutions to verifiable scientific and algorithmic problems by spending additional computation at test time. Recent systems achieve strong results with increasingly elaborate evolutionary search harnesses or by updating model parameters during test-time training. We ask how much of this machinery is necessary. We introduce Hill Sampling, a simple procedure that repeatedly samples candidate program edits from a frozen LLM, retains the best program found so far, and conditions all subsequent samples on that program. We evaluate the method on circle packing, sums/differences of sets, and Erdos' minimum-overlap problem using three open-weight models. Hill Sampling sets a new state of the art on circle packing among published methods, improves over the AlphaEvolve reference on Erdos' minimum-overlap problem, and achieves strong results on sums and differences of finite sets. The circle-packing and Erdo
    
[^216]: 你已经看够了：面向机器的质量约束图像编码

    You've Seen Enough: Quality-Constrained Image Coding for Machines

    [https://arxiv.org/abs/2609.25108](https://arxiv.org/abs/2609.25108)

    该论文提出一种面向机器的质量约束图像编码方法，将人类视觉质量限制在预设目标水平，并通过绝对值和双线性两种惩罚函数设计，把剩余比特容量全部分配给机器视觉任务，从而在满足人眼检查需求的同时最大化机器任务性能。

    

    视觉数据正越来越多地被机器视觉系统而非人类观察者所消费。面向机器的图像编码（ICM）在压缩图像时假设主要观察者是计算机视觉应用，而人类观察者只需检查或验证其决策结果。受恰可察觉失真的启发，该方法将人类观察到的质量上限设定在期望水平，并将剩余的比特全部用于提升机器性能。具体而言，压缩与分割的联合训练被重新表述为一个约束优化问题：编解码器必须满足预定义的可接受目标视觉质量，而任务项则消耗剩余的编码容量。本文提出了两种引导图像质量趋向目标的惩罚函数变体：绝对值函数和双线性函数，其中后者在超过目标视觉质量后采用更陡的斜率。实验结果表明，在质量约束下……

    arXiv:2609.25108v2 Announce Type: replace-cross  Abstract: Visual data is increasingly consumed by machine-vision systems rather than by human observers. Image Coding for Machines (ICM) compresses images assuming the main observer is a computer vision application and that the human observer needs to inspect or validate the decisions. Inspired by just-noticeable distortion, we cap human-observed quality at a desired level and devote the remaining bits to machine performance. Specifically, joint compression-segmentation training is recast as a constrained optimization problem in which the codec must meet a predefined acceptable target visual quality while a task term consumes the remaining coding capacity. This paper proposes two variants of a penalty function that guides the quality toward the target: an absolute function and a bilinear function, the latter applying a steeper slope once the target visual quality is exceeded. Experimental results show that, under the quality constraint, 
    
[^217]: 面向离线强化学习的提升贝尔曼线性规划

    Lifted Bellman Linear Programming for Offline Reinforcement Learning

    [https://arxiv.org/abs/2609.24489](https://arxiv.org/abs/2609.24489)

    提出提升贝尔曼线性规划（LBLP），通过将贝尔曼最优性的线性规划刻画提升到联合(Q,V)空间，用仅涉及数据集中状态-动作对的不等式约束实现样本内贝尔曼最优性，其唯一最优解在确定性动力学下介于数据集最佳回报与最优价值之间。

    

    离线强化学习（RL）通常通过针对自举价值目标最小化回归损失来训练评论家（critic），并利用采用指数移动平均（EMA）更新的目标网络来稳定训练。多步目标包含行为策略的动作，因此需要离策略修正。我们转而通过不等式约束对评论家施加样本内贝尔曼最优性。我们提出了提升贝尔曼线性规划（Lifted Bellman Linear Program, LBLP），它将贝尔曼最优性的线性规划刻画提升到联合 $(Q,V)$ 空间，使得每个约束仅涉及数据集中的状态-动作对。该规划的唯一最小化子即为样本内最优对，且沿数据集轨迹 $K$ 步片段施加的约束对于任何 rollout 策略和任意视野都保持该最小化子不变。在确定性动力学下，该最小化子介于数据集最佳回报与最优价值之间。将约束松弛为合页（损失）……

    arXiv:2609.24489v1 Announce Type: new  Abstract: Offline reinforcement learning (RL) typically trains a critic by minimizing a regression loss against bootstrapped value targets stabilized by target networks with exponential moving average (EMA) updates. Multi-step targets incorporate behavior-policy actions and therefore require off-policy correction. We instead impose in-sample Bellman optimality on the critic through inequality constraints. We formulate the Lifted Bellman Linear Program (LBLP), which lifts the linear programming characterization of Bellman optimality to the joint $(Q,V)$ space so that every constraint involves only state-action pairs in the dataset. Its unique minimizer is the in-sample optimal pair, and constraints along $K$-step segments of dataset trajectories leave this minimizer unchanged for any rollout policy and horizon. Under deterministic dynamics, this minimizer lies between the best dataset return and the optimal value. Relaxing the constraints into hing
    
[^218]: MM-ContextFold：面向多模态智能体检索的上下文折叠

    MM-ContextFold: Context Folding for Multimodal Agentic Retrieval

    [https://arxiv.org/abs/2609.23121](https://arxiv.org/abs/2609.23121)

    提出了无需训练的 MM-ContextFold 框架，基于对约一万条轨迹的实证发现——当视觉信息被提取并文本化后原始图像变得冗余——通过“折叠”冗余视觉内容来解决多模态智能体检索中的上下文爆炸问题。

    

    多模态智能体检索要求智能体通过迭代调用外部工具来解决复杂的信息搜索任务。诸如 ReAct 等典型框架将原始多模态输入和不断累积的交互历史保持在单一且持续增长的上下文中，从而导致上下文爆炸问题。虽然现有方法通过压缩冗余文本来缓解这一问题，但针对如何有效管理 token 密集的视觉内容的策略在很大程度上仍未被充分探索。为了填补这一空白，我们首先对约 10,000 条轨迹进行了系统的实证研究。结果表明，随着视觉线索通过外部工具被逐步提取并文本化到上下文中，原始图像变得越来越冗余。持续保留图像与更高的输出熵相关，甚至会降低任务准确率。受这些发现的启发，我们提出了 MM-ContextFold，这是一个无需训练的框架，其……（摘要在此处被截断）

    arXiv:2609.23121v1 Announce Type: cross  Abstract: Multimodal Agentic Retrieval (MAR) requires agents to solve complex information-seeking tasks by iteratively invoking external tools. Typical frameworks such as ReAct maintain raw multimodal inputs and the accumulating interaction history in a single, ever-growing context, leading to the context explosion problem. While existing methods alleviate this issue by compressing redundant text, effective strategies for managing token-intensive visual content remain largely underexplored. To address this gap, we first conduct a systematic empirical study of approximately 10,000 trajectories. The results show that as visual cues are progressively extracted through external tools and textualized into the context, raw images become increasingly redundant. Continued image retention is associated with higher output entropy and can even degrade task accuracy. Motivated by these findings, we propose MM-ContextFold, a training-free framework that load
    
[^219]: 矩阵 AdaGrad：按行与按列的自适应次梯度方法

    Matrix AdaGrad: Row-wise and Column-wise Adaptive Subgradient Methods

    [https://arxiv.org/abs/2609.21815](https://arxiv.org/abs/2609.21815)

    本文提出了一个针对矩阵值参数的通用在线镜像下降框架，通过引入按行和按列的自适应近端函数，推导出 Row-AdaGrad 和 Column-AdaGrad 两种优化器，将 AdaGrad 式的自适应次梯度方法推广到了具有矩阵结构的参数优化中。

    

    arXiv:2609.21815v1 公告类型：交叉 摘要：AdaGrad 和 Adam 等自适应优化方法在现代神经网络训练中被广泛使用，但它们的自适应缩放主要是为向量值参数设计的，并未显式地利用矩阵结构。近期的矩阵感知优化器展示了结构化优化的优势，然而，目前仍缺乏一个可用于推导与 AdaGrad 相当的矩阵感知自适应性的通用理论框架。在本工作中，我们为矩阵值参数开发了一个带有自适应近端函数的通用在线镜像下降框架，提供了一种通过在线遗憾最小化来推导矩阵感知自适应优化的原则性方法。通过引入按行和按列的矩阵近端函数，并分析由此产生的遗憾权衡，我们推导出了按行矩阵 AdaGrad和按列矩阵 AdaGrad，其自适应缩放由累积的（按行/按列）梯度信息决定。

    arXiv:2609.21815v1 Announce Type: cross  Abstract: Adaptive optimization methods such as AdaGrad and Adam are widely used in modern neural-network training, but their adaptive scaling is primarily designed for vector-valued parameters and does not explicitly exploit matrix structure. Recent matrix-aware optimizers demonstrate the benefits of structured optimization, yet a general theoretical framework for deriving matrix-aware adaptivity comparable to that of AdaGrad remains lacking. In this work, we develop a general Online Mirror Descent framework with adaptive proximal functions for matrix-valued parameters, providing a principled approach to deriving matrix-aware adaptive optimization through online regret minimization. By introducing row-wise and column-wise matrix proximal functions and analyzing the resulting regret trade-off, we derive Row-wise Matrix AdaGrad (Row-AdaGrad) and Column-wise Matrix AdaGrad (Column-AdaGrad), with adaptive scaling determined by the accumulated row-w
    
[^220]: 混合专家语言模型中专家的高阶剪枝

    Higher-order pruning of experts in mixture-of-experts language models

    [https://arxiv.org/abs/2609.18916](https://arxiv.org/abs/2609.18916)

    提出二阶剪枝方法HOPE，通过捕捉专家之间的高阶交互作用来可证明地最小化剪枝误差上界，在多个前沿MoE模型和基准测试上的剪枝效果优于忽略专家协作性的一阶方法。

    

    arXiv:2609.18916v1 公告类型：交叉 摘要：混合专家语言模型存在参数量庞大的问题，这造成了显著的内存瓶颈。专家剪枝是减少参数量最直接的方法，然而现有方法对每个专家独立地做出剪枝决策，并假设专家的贡献是纯粹可加的。实际上，混合专家模型中专家的使用本质上是协作性的。我们推导出了HOPE（专家高阶剪枝），这是一种二阶剪枝目标函数，可以证明能够最小化剪枝所产生的误差上界。我们证明REAP（一种最先进的一阶剪枝方法）是HOPE在忽略交互项时的特例。在三个前沿MoE模型（参数量高达1220亿）、两个不同的校准集以及多个基准测试（包括数学、指令遵循、编程和智能体任务套件）上，我们证明HOPE能够比现有方法做出更好的剪枝决策，并且……

    arXiv:2609.18916v1 Announce Type: cross  Abstract: Mixture-of-Experts (MoE) language models suffer from large parameter counts, which create a significant memory bottleneck. Expert pruning is the most direct approach for reducing this parameter count, yet existing methods make pruning decisions for each expert independently, and assume experts' contributions are purely additive. In reality, expert usage in MoEs is inherently cooperative. We derive HOPE (Higher-Order Pruning of Experts), a second-order pruning objective which provably minimizes an upper bound on the error resulting from pruning. We show that REAP (a state-of-the-art first-order pruning method) is a special case of HOPE where interaction terms are ignored. Across three frontier MoE models (up to 122B parameters), two distinct calibration sets, and multiple benchmarks (including math, instruction following, coding, and an agentic suite), we demonstrate that HOPE produces better pruning decisions than existing methods, and
    
[^221]: 一种基于术前多模态数据的精准且全面的脑肿瘤诊断视觉-语言基础模型

    A Vision-Language Foundation Model for Precise and Comprehensive Brain Tumor Diagnosis from Preoperative Multimodal Data

    [https://arxiv.org/abs/2609.16597](https://arxiv.org/abs/2609.16597)

    BrainVLM是一种视觉-语言基础模型，能够基于术前多模态MRI数据对12种WHO 2021脑肿瘤类型进行自动精准分类，并同时提供诊断不确定性量化和放射学报告生成功能，解决了传统MRI诊断中影像特征重叠和观察者差异的难题。

    

    背景：基于磁共振成像（MRI）的脑肿瘤类型术前无创诊断至关重要，但由于不同肿瘤类型之间的影像特征重叠、观察者间的判读差异以及培养专业放射科医师所需的长期训练，这一任务充满挑战。我们旨在开发一种基于MRI的人工智能（AI）模型，用于自动、可靠的脑肿瘤分类，并具备诊断不确定性量化和放射学报告生成能力。方法：我们开发了BrainVLM，可对所有12种世界卫生组织（WHO）2021年脑肿瘤类型进行分类。BrainVLM集成了不确定性量化策略以指示预测的可靠性，并包含一个生成放射学报告的模块以阐明临床诊断依据。BrainVLM在来自40,043名个体的多模态数据（MRI扫描、人口统计学信息和放射学报告）上进行训练，并在5,211名经病理确诊的脑肿瘤患者上进行了验证。

    arXiv:2609.16597v1 Announce Type: cross  Abstract: Background Non-invasive presurgical diagnosis of brain tumor types from Magnetic Resonance Imaging (MRI) is essential but challenging due to overlapping imaging features across tumor types, inter-observer variability, and the extensive training required for expertise. We aimed to develop an MRI-based Artificial Intelligence (AI) model for automatic and reliable brain tumor classification with diagnostic uncertainty quantification and radiology reports generation.   Methods We developed BrainVLM to classify all 12 World Health Organization (WHO) 2021 brain tumor types. BrainVLM integrates an uncertainty quantification strategy to indicate prediction reliability and a module for generating radiology reports to elucidate the clinical rationale. BrainVLM was trained on multi-modal data (MRI scans, demographics, and radiology reports) from 40,043 individuals. It was validated on 5,211 patients with pathologically confirmed brain tumors, inc
    
[^222]: ProtoLIP：从句子级到对象级的证据解耦

    ProtoLIP: From Sentence-Level to Object-Level Evidence Disentanglement

    [https://arxiv.org/abs/2609.16284](https://arxiv.org/abs/2609.16284)

    提出ProtoLIP，一种轻量级的原型介导证据层，通过将视觉原型组织为文本语义族并进行查询相关的族路由，在无需空间标注或骨干网络重训练的情况下，实现从句子级到对象级的证据解耦，显著提升视觉-语言模型的证据定位与分离能力。

    

    查询条件化的视觉-语言模型通过揭示视觉证据如何随文本查询变化，实现了细粒度的解释。然而，基于完整描述所条件化的证据并不一定能分解为特定对象的证据，且已暴露的证据图也不一定能识别出构成模型预测的证据。在多个VLM架构和独立基准测试中，我们发现对象级查询往往保留了来自共同出现对象和共享上下文的证据。本文提出了ProtoLIP，一个轻量级的原型介导证据层，它将可重用的视觉原型组织成由文本导出的语义族，并利用查询相关的语义族路由来约束哪些原型可以提供证据。在无需空间标注或骨干网络重训练的情况下，ProtoLIP在各种查询粒度上均提升了证据的定位与分离能力，并带来了定位性能的增益。

    arXiv:2609.16284v1 Announce Type: cross  Abstract: Query-conditioned vision--language models enable fine-grained interpretation by revealing how visual evidence changes with textual queries. However, evidence conditioned on complete descriptions does not necessarily resolve into object-specific evidence, nor does an exposed evidence map necessarily identify the evidence that constitutes the model's prediction. Across multiple VLM architectures and independent benchmarks, we find that object-level queries often retain evidence from co-occurring objects and shared context. In this paper, we introduce \textbf{ProtoLIP}, a lightweight prototype-mediated evidence layer that organizes reusable visual prototypes into text-derived semantic families and uses query-dependent family routing to constrain which prototypes may provide evidence. Without spatial annotations or backbone retraining, ProtoLIP improves evidence localization and separation across query granularities, with localization gain
    
[^223]: 思维状态实现内生推理

    State of Thought Enables Endogenous Reasoning

    [https://arxiv.org/abs/2609.16055](https://arxiv.org/abs/2609.16055)

    提出“思维状态”新推理范式，通过从模型内部信息传递中提取动力学-几何状态并用仅 582 参数的控制器在冻结模型上选择性激活历史推理支持，实现由模型内部状态主导的内生推理，摆脱外部控制。

    

    测试时计算已成为提升大语言模型能力的重要方法。然而，现有的测试时推理范式严重依赖外部施加的控制，要么通过固定的推理程序，要么通过在受限搜索空间中进行高成本扩展，这同时限制了泛化能力和效率。我们提出了思维状态，这是一种新的推理范式，使大语言模型能够进行内生推理，由模型自身的内部推理状态来支配推理的展开方式。具体而言，SoT 从模型内部的信息传递中提取一个紧凑的动力学-几何状态，并使用一个仅有 582 个参数的控制器作用于冻结的骨干模型，选择性地激活在当前推理状态下有用的历史推理支持，从而将推理构建为一个以状态为条件、基于证据的过程，而非外部规定的 token 链。在量化、通用、符号等任务上分别取得 1.34 倍、1.62 倍等提升。

    arXiv:2609.16055v1 Announce Type: cross  Abstract: Test-time compute has emerged as a major approach to improving the capabilities of Large Language Models (LLMs). However, existing test-time reasoning paradigms rely heavily on externally imposed control, either through fixed reasoning programs or through costly expansion in constrained search spaces, limiting both generalization and efficiency. We propose State of Thought (SoT), a new reasoning paradigm that enables endogenous reasoning in LLMs, with the model's internal reasoning state governing how reasoning unfolds. Concretely, SoT extracts a compact dynamics-geometric state from the model's internal information transfer and uses a 582-parameter controller on frozen backbones to selectively activate historical reasoning support useful under the current reasoning state, framing reasoning as a state-conditioned process over evidence rather than an externally prescribed token chain. Across quantitative (1.34x), general (1.62x), symbol
    
[^224]: 解释说话人嵌入的层次化组织

    Interpreting hierarchical organisation of speaker embeddings

    [https://arxiv.org/abs/2609.15203](https://arxiv.org/abs/2609.15203)

    本文从可解释人工智能（XAI）的视角出发，利用SLINK层次聚类算法分析说话人嵌入是否自然形成层次化聚类结构，并提出了一种新的层次聚类-类别匹配评估方法来解释说话人嵌入的层次化组织。

    

    说话人识别神经网络从输入语音中学习潜在表示（即说话人嵌入），以识别说话人身份。然而，这些网络的内部机制在很大程度上仍然是不透明的，这促使人们开展可解释人工智能（XAI）方面的研究来理解它们。尽管如此，现有研究已经对说话人嵌入的组织方式进行了分析，但很少将这些分析置于XAI的框架内。因此，本工作提出从XAI的角度来解释和阐释说话人嵌入的组织方式。为此，我们应用一种层次聚类算法——单链接聚类（SLINK），来分析某些说话人嵌入是否会自然形成具有层次关系的聚类。所得到的层次组织（即层次聚类）使用聚类-类别匹配（CCM）方法进行评估。此外，我们提出了一种新方法，称为层次聚类-类别匹配。

    arXiv:2609.15203v1 Announce Type: cross  Abstract: Speaker recognition neural networks learn latent representations (i.e. speaker embeddings) from input utterances to recognise speaker identities. However, the internal mechanisms of these networks remain largely opaque, motivating research in explainable artificial intelligence (XAI) to understand them. Nevertheless, existing studies have analysed how speaker embeddings are organised, but rarely frame these analyses within XAI. Hence, this work proposes to explain and interpret the organisation of speaker embeddings from an XAI perspective.   To this end, we apply a hierarchical clustering algorithm, Single-Linkage Clustering (SLINK), to analyse whether some speaker embeddings naturally form clusters with hierarchical relationships. The resulting hierarchical organisation (i.e. hierarchical clusters) is evaluated using the Cluster-Class Matching (CCM) method. Moreover, we propose a new method, termed Hierarchical Cluster-Class Matching
    
[^225]: T-LoopFormer：基于动态路由的Token级弹性深度循环Transformer用于潜在推理

    T-LoopFormer: Token-Level Elastic-Depth Looped Transformers for Latent Reasoning With Dynamic Routing

    [https://arxiv.org/abs/2609.15160](https://arxiv.org/abs/2609.15160)

    该论文提出T-LoopFormer，通过动态token选择路由机制，让循环Transformer中的每个token根据自身隐藏状态自适应地决定循环迭代次数，从而实现更优的计算分配和效率提升。

    

    循环Transformer（Looped Transformers）近期在推理和语言任务中展现出强大的性能，它通过在多次迭代中重用一组共享参数，在不牺牲表示能力的情况下实现了参数效率。此外，循环Transformer直接在潜空间中进行推理（即潜在推理），减少了推理过程中消耗的token数量，从而获得更高的样本效率。然而，这类模型通常对所有token统一施加固定的递归深度，导致计算分配欠佳，错失了显著的效率提升空间。在这项工作中，我们为循环Transformer提出了动态token选择路由机制，使每个token能够基于其隐藏状态自适应地确定自身的循环迭代次数。我们使用一个动态路由器来决定token是继续递归还是提前退出，允许简单的token绕过不必要的计算。

    arXiv:2609.15160v1 Announce Type: new  Abstract: Looped Transformers have recently demonstrated strong performance in both reasoning and language tasks by reusing a shared set of parameters across multiple iterations, achieving parameter efficiency without sacrificing representational power. Besides, looped Transformers perform inference directly in the latent space (latent reasoning) to reduce the number of tokens consumed during inference, thereby achieving improved sample efficiency. However, these models typically apply a fixed recursion depth uniformly to every token, leading to suboptimal compute allocation and leaving significant efficiency gains on the table. In this work, we propose \textbf{dynamic token-choice routing} for looped transformers, enabling each token to adaptively determine its own number of loop iterations based on its hidden state. We use a dynamic router to decide whether a token should continue recursing or exit early, allowing simple tokens to bypass unneces
    
[^226]: 远距离大梯度未必可靠：面向长时程自回归预测的可靠性加权信用分配

    Large Distant Gradients Need Not Be Reliable: reliability-weighted credit assignment for long-horizon autoregressive forecasting

    [https://arxiv.org/abs/2609.12890](https://arxiv.org/abs/2609.12890)

    提出Internal-DW方法，通过在反向传播中对每个残差块的恒等路由和非线性路由施加由显式噪声模型估计的有界维纳增益进行可靠性加权，在抑制长时程自回归预测中不可靠远距离梯度噪声的同时保留可预测的学习信号。

    

    在自回归预测中，长预测展开能够提供远距离的监督信号，但通过时间的反向传播（BPTT）需要将这些损失的梯度经过许多自回归步骤逐步传递。反复的雅可比矩阵乘积可能使远距离梯度在参数更新中占据主导地位，同时放大可预测信号与不可预测噪声；因此，大的远距离梯度并不一定携带可靠的学习信号。基于这一观察，我们提出了内部双维纳路由（Internal-DW），这是一种仅作用于反向传播过程的原则性干预方法，它在保留完整前向展开和所有时域损失的同时，对内部梯度路由进行可靠性加权。在每个残差块处，我们为恒等路由和非线性路由推导出有界的维纳增益，以在保留可预测学习信号与抑制不可预测变化之间取得平衡，并通过路由级别的梯度统计量和显式噪声模型对这些增益进行估计。在一个受控的（摘要在此处截断）

    arXiv:2609.12890v1 Announce Type: new  Abstract: In autoregressive forecasting, long prediction rollouts provide distant supervision, but backpropagation through time (BPTT) carries gradients from those losses through many autoregressive steps. Repeated Jacobian products can make distant gradients dominate the update while amplifying predictable signal and unpredictable noise together; a large distant gradient therefore need not carry reliable learning signal. Motivated by this observation, we introduce Internal Dual-Wiener routing (Internal-DW), a principled backward-only intervention that preserves the full forward rollout and all horizon losses while reliability-weighting internal gradient routes. At each residual block, we derive bounded Wiener gains for the identity and nonlinear routes that balance preserving predictable learning signal against suppressing unpredictable variation, and estimate them from route-level gradient statistics and an explicit noise model. In a controlled 
    
[^227]: 探索说话人识别中的二阶模式识别

    Exploring Second-Order Pattern Recognition in Speaker Recognition

    [https://arxiv.org/abs/2609.11182](https://arxiv.org/abs/2609.11182)

    该论文提出利用层次聚类来发现说话人识别网络中将话语识别为说话人身份时潜藏的“二阶模式”，并通过HCCM方法对其语义解释，进而提出“二阶模式识别”这一新任务。

    

    在经典模式识别任务中，神经网络被训练用于识别模型输入中人类定义的模式。一些可解释人工智能（XAI）方法能够解释潜藏在网络将输入识别为人类定义模式这一过程背后的其他潜在模式；在这项工作中，我们将这些潜在模式称为二阶模式，并提出对它们进行发现。为此，我们应用层次聚类算法来分析说话人识别网络从语音话语中学习到的表示是否自然形成层次聚类。每个由此产生的聚类代表一种二阶模式，它刻画了网络如何将一些已知话语识别为特定说话人身份。随后，我们使用现有的层次聚类-类别匹配（HCCM）方法对所有得到的二阶模式进行语义解释。此外，我们提出了一个新任务——二阶模式识别，用于识别所发现的……

    arXiv:2609.11182v1 Announce Type: cross  Abstract: In classical pattern recognition tasks, neural networks are trained to recognise human-defined patterns for model inputs. Some Explainable AI (XAI) methods can explain other latent patterns that underlie the network's recognition of inputs as human-defined patterns; in this work, we call these latent patterns second-order patterns, and we propose to discover them. To this end, we apply a hierarchical clustering algorithm to analyse whether representations learned by a speaker recognition network from utterances naturally form hierarchical clusters. Each resulting cluster represents a second-order pattern that characterises how the network recognises some known utterances as speaker identities. All the resulting second-order patterns are then semantically interpreted using the existing Hierarchical Cluster-Class Matching (HCCM) method.   Furthermore, we propose a new task, second-order pattern recognition, to identify which discovered s
    
[^228]: EGGROLL展开：理解并改进大规模低秩进化策略

    EGGROLL, Unrolled: Understanding and Improving Low-Rank Evolution Strategies at Scale

    [https://arxiv.org/abs/2609.10980](https://arxiv.org/abs/2609.10980)

    本文首次从理论上刻画了面向大语言模型的低秩进化策略EGGROLL的更新场，揭示其可能引入非保守分量并逆转最优点的局部稳定性，同时证明了该方法的二次目标精确性并给出非渐近误差界，为理解与改进该方法奠定理论基础。

    

    EGGROLL通过用低秩高斯乘积（通常为秩一）替代稠密高斯权重扰动，使进化策略（ES）在大语言模型（LLM）上变得实用。这一选择在计算上颇具吸引力，但在几何上却相当严苛：尽管协方差为单位阵，每个秩一扰动都位于环境矩阵空间的一个零体积子集中。我们刻画了在有限秩和非零扰动半径下EGGROLL的平均更新场，并分析了其有限种群估计器的误差。该种群场是通过将一个显式预解式作用于由扰动平滑后的目标函数梯度而得到的。我们证明该预解式可能引入非保守分量，并可能逆转最优点的局部稳定性。尽管如此，EGGROLL在任意秩和任意半径下对所有二次目标函数都是精确的。对于光滑目标函数，其首个局部有限秩修正项为O(σ²/r)，且非渐近界控制着

    arXiv:2609.10980v1 Announce Type: new  Abstract: EGGROLL makes evolution strategies (ES) practical for LLMs by replacing dense Gaussian weight perturbations with low-rank Gaussian products, often of rank one. This choice is computationally attractive but geometrically severe: each rank-one perturbation lies in a zero-volume subset of the ambient matrix space, despite having identity covariance. We characterize the mean EGGROLL update field at finite rank and nonzero perturbation radii, then analyze the error of its finite-population estimator. The population field is obtained by applying an explicit resolvent to the gradient of the objective smoothed by the perturbations. We show that the resolvent can introduce a nonconservative component and can reverse the local stability of an optimum. EGGROLL is nevertheless exact on every quadratic objective at every rank and radius. For smooth objectives, its first local finite-rank correction is $O(\sigma^2/r)$, and nonasymptotic bounds control
    
[^229]: 特征叠加中线性能及性的高概率保证

    High-probability guarantees for linear accessibility in feature superposition

    [https://arxiv.org/abs/2609.09556](https://arxiv.org/abs/2609.09556)

    该论文将特征叠加中的线性能及性建模为压缩感知问题，证明了充分维度只需线性规模（而非此前最坏情况的二次方限制）即可高概率地恢复同时激活的特征，从而量化了线性表示假设的几何约束并为稀疏自编码器和神经可解释性评估提供了理论框架。

    

    神经网络可以利用特征叠加来编码比维度数量更多的概念，但特征间的交叉干扰限制了同时激活特征的线性能及性。通过将线性能及性建模为一个压缩感知问题，我们在次高斯噪声下针对固定支撑集推导出高概率界，证明了充分维度以线性方式扩展（d=O_ε(k log m)），而非此前最坏情况下的二次方限制。随后，我们通过高斯尾近似在各系统参数下验证了这些界。这些结果量化了线性表示假设的几何约束，为评估稀疏自编码器、组合泛化和神经网络可解释性提供了一个框架。

    arXiv:2609.09556v1 Announce Type: cross  Abstract: Neural networks can leverage feature superposition to encode more concepts than dimensions, but cross-feature interference constrains the linear accessibility of simultaneously active features. By framing linear accessibility as a compressed sensing problem, we derive high-probability bounds for fixed supports under subgaussian noise, proving the sufficient dimension scales linearly ($d=O_{\varepsilon}(k \log m)$) rather than prior worst-case quadratic limits. We then validate these bounds across system parameters through Gaussian-tail approximations. These results quantify the geometric constraints of the linear representation hypothesis, providing a framework for evaluating sparse autoencoders, compositional generalization, and neural interpretability.
    
[^230]: 导向干扰反映的是模型的默认倾向，而非行为方向

    Steering Interference Reflects the Model's Defaults, Not the Behavior Directions

    [https://arxiv.org/abs/2609.06951](https://arxiv.org/abs/2609.06951)

    激活导向引发的副作用并非来自被导向的行为方向本身，而是由模型自身的默认偏好决定——无论导向何种行为，模型都会趋向其本已偏好的少数行为（如拒答、谄媚、诗歌化）。

    

    激活导向有望实现对语言模型行为的模块化控制：某种行为（如礼貌）对应模型激活中的一个方向，在模型生成时添加该方向应当能开启该行为，且不影响其他方面。但事实并非如此。我们探究了是什么决定了哪些其他行为会发生变化以及变化程度，发现起决定作用的是模型本身，而非被导向的行为。导向会使模型放松趋向于它本已偏好的一小部分行为，主要是拒答、谄媚和诗歌化倾向，且无论导向什么行为，这一行为集合大体相同。横跨24种行为和十个指令微调模型的三项结果支持这一结论，所有效应均由语言模型裁判从生成文本中读取，而非通过探针读取。这种读取方式很重要：所有24种行为都是线性可解码的，但只有20种行为会改变模型的实际输出内容。第一，一个不含任何行为内容、仅在特定方面与真实导向相匹配的方向……（摘要在此处被截断）

    arXiv:2609.06951v1 Announce Type: cross  Abstract: Activation steering promises modular control of language model behavior: a behavior such as politeness corresponds to a direction in a model's activations, and adding that direction while it generates should switch the behavior on and leave everything else alone. It does not. We ask what decides which other behaviors move, and by how much, and find that it is the model rather than the behavior being steered. A steer relaxes the model toward a small set of behaviors it already favors, chiefly refusal, sycophancy, and poeticism, and that set is much the same whatever is steered.   Three results across 24 behaviors and ten instruction-tuned models support this, every effect read off the generated text by a language-model judge rather than off a probe. That readout matters: all 24 behaviors are linearly decodable, but only 20 change what the model writes. First, a direction carrying no behavioral content, matched to a real steer only in th
    
[^231]: 拒答的几何学：为什么事后安全训练脆弱而预训练期安全却能持久

    The Geometry of Refusal: Why Post-Hoc Safety Is Fragile and Pretraining-Time Safety Persists

    [https://arxiv.org/abs/2609.06934](https://arxiv.org/abs/2609.06934)

    该论文从几何视角证明，事后安全训练（如RLHF）的更新与模型能力方向近乎正交，只是在完好的能力之上叠加一道薄而尖锐的拒答“闸门”而非真正删除能力，因此注定会被越狱等攻击绕过，而持久的安全必须在预训练阶段扎根。

    

    事后安全训练（RLHF、DPO）是目前对齐大语言模型的主流方法，然而越狱攻击（Zou et al., 2023b）、微调攻击（Qi et al., 2024）以及激活空间探测（Arditi et al., 2024）总能恢复出那些本应被移除的行为。我们对这种脆弱性给出了一种几何解释，并将其追溯到预训练过程中安全机制能够真正扎根的时机。我们将安全更新 $\Delta = W_{\text{safe}} - W_{\text{base}}$ 与模型能力的曲率（能力损失的经验 Fisher 信息）进行对照测量。研究发现，事后安全训练始终落入一个“抑制区间”：$\Delta$ 与能力方向近乎正交，其在子空间内的微小分量集中于少数高曲率方向。这一更新“薄而尖锐”——它是在完好的能力之上叠加的一道拒答闸门，而非对能力的真正抹除。一个核不动性引理解释了为什么这样的更新只能掩盖能力而无法将其移除，因此……（摘要原文在此处截断）

    arXiv:2609.06934v1 Announce Type: cross  Abstract: Post-hoc safety training (RLHF, DPO) is the dominant way to align large language models, yet jailbreaks (Zou et al., 2023b), fine-tuning attacks (Qi et al., 2024), and activation-space probes (Arditi et al., 2024) keep recovering the behaviors it was meant to remove. We give this fragility one geometric explanation and trace it to when, during pretraining, safety can take hold. We measure the safety update $\Delta = W_{\text{safe}} - W_{\text{base}}$ against the curvature of the model's capabilities (the empirical Fisher of a capability loss). Post-hoc safety consistently lands in a suppression regime: $\Delta$ is nearly orthogonal to the capability directions, and its small in-subspace part concentrates on a few high-curvature ones. The update is thin but sharp, a refusal gate laid over intact capabilities rather than erasure of them. A kernel-immobility lemma explains why such an update can only mask a capability, not remove it, so a
    
[^232]: 可证明安全的仿真到现实迁移

    Provably Safe Sim-to-Real Transfer

    [https://arxiv.org/abs/2609.01418](https://arxiv.org/abs/2609.01418)

    该论文提出并形式化了“安全仿真到现实迁移”问题，通过在无奖励安全强化学习框架内构建该问题，使智能体能够在利用不完美模拟器的同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略。

    

    为了缓解现实世界强化学习（RL）的样本复杂度问题，一种常见的做法是先在模拟器中训练策略（因为样本成本低廉），然后将学到的策略部署到现实世界中，并希望其能有效泛化。然而，这种直接的仿真到现实迁移并不保证成功：由于仿真与现实之间的失配（sim-to-real mismatch），在模拟器中训练的策略在现实世界中可能是次优的。纠正这种失配需要从真实系统收集数据，但在许多应用中（如机器人技术和医疗保健），这种数据收集过程本身受到安全约束的制约。这就引出了安全仿真到现实迁移的问题：智能体如何利用一个不完美的模拟器，同时确保现实世界数据收集的安全性，并为目标系统学习到接近最优的可行策略？我们通过在无奖励安全强化学习框架内构建安全仿真到现实迁移问题来应对这一挑战……

    arXiv:2609.01418v1 Announce Type: cross  Abstract: To mitigate the sample complexity of real-world reinforcement learning (RL), a common practice is to first train a policy in a simulator, where samples are cheap, and then deploy the learned policy in the real world with the hope that it generalizes effectively. Such direct sim-to-real transfer is not guaranteed to succeed: simulator-trained policies can be suboptimal in the real world due to sim-to-real mismatch. Correcting this mismatch requires collecting data from the real system, but in many applications, such as robotics and healthcare, this data-collection process is itself subject to safety constraints. This gives rise to the problem of safe sim-to-real transfer: how can an agent exploit an imperfect simulator while ensuring safe real-world data collection and learning a near-optimal feasible policy for the target system? We address this problem by formulating safe sim-to-real transfer within the framework of reward-free safe R
    
[^233]: SUN：面向语言接地的“控制—学习—真实部署”策略的持久化程序

    SUN: Persistent Programs For Language-Grounded Control-to-Learning-to-Real Policies

    [https://arxiv.org/abs/2608.31167](https://arxiv.org/abs/2608.31167)

    该论文提出语义统一（SUN）程序，将几何与接触关系定义一次即编译为对齐的MPC代价、满足性谓词、RL奖励与转移守卫，并由“夸父”系统从语言和场景语义自动合成程序、经MPC筛选可行性后训练阶段条件化策略，在九个长时程操作任务上取得82.03%的成功率，显著超越稀疏奖励等基线方法。

    

    在长时程操作任务中，桥接基于模型的控制与学习策略一直存在一个隐蔽的分歧：控制负责执行指定的目标，学习则将该行为摊销固化为反应式策略，然而现有协议丢弃了任务语义，导致奖励需要手工设计，且策略行为会偏离控制所验证的内容。我们提出了语义统一程序，这是一类带类型的可执行程序，其中几何与接触关系只需定义一次，即可编译为相互对齐的模型预测控制（MPC）代价函数、满足性谓词、强化学习奖励、转移守卫条件以及诊断信息。我们的系统“夸父”由大型视觉语言系统驱动，能够从语言和场景语义中自动合成SUN程序，通过MPC筛选任务可行性，并在训练阶段条件化策略的同时保留任务语义。在九个任务上，夸父取得了82.03%的宏观成功率，超越了稀疏奖励基线（35.67%）和Stage-BC基线（24.75%）。

    arXiv:2608.31167v1 Announce Type: cross  Abstract: Bridging model-based control and learned policies in long-horizon manipulation has harbored a silent disagreement: control executes specified objectives, learning amortizes that behavior into a reactive policy, yet existing protocols discard task semantics, leaving rewards hand-crafted and behavior drifting from what control verified.We introduce Semantically UNified (SUN) Programs, typed executables where geometric and contact relations are defined once and compiled into aligned Model Predictive Control (MPC) costs, satisfaction predicates, RL rewards, transition guards, and diagnostics. Our system, Kuafu, driven by large vision language systems, automatically synthesizes SUN Programs from language and scene semantics, screens feasibility via MPC, and retains semantics while training stage-conditioned policies. Across nine tasks, Kuafu achieves 82.03% macro-success, outperforming sparse-reward (35.67%) and Stage-BC (24.75%) baselines.
    
[^234]: 校准到足以知晓，却未校准到行动：伪造证据使LLM代理对不可知之事做出承诺

    Calibrated Enough to Know, Not Calibrated to Act: Fabricated Evidence Makes LLM Agents Commit to the Unknowable

    [https://arxiv.org/abs/2608.27167](https://arxiv.org/abs/2608.27167)

    本文发现LLM代理在面临伪造的专业面板证据时，会显著提高对不可预测问题的承诺率，其行为受证据包装的权威性驱动而非信息真实性或模型信念，揭示了一种可定位的校准失败。

    

    arXiv:2608.27167v1 公告类型：新 摘要：一个LLM代理在看到一个看起来专业的市场面板时，对一个问题做出方向性判断的频率远高于仅被问及该问题本身时的频率——在12个前沿模型中，随着证据的升级，承诺率从6.5%上升到54.0%。即使面板上的每个数字都是编造的，它同样会轻易承诺：完全伪造整个显示内容，使模型能看到的除了问题本身外无一是真，仍将承诺率从24.5%提升至36.8%，这在统计上与真实市场数据产生的37.6%无显著差异。解锁自信行动的不是信息，而是其包装的权威性。这种失败是狭窄且可定位的。能力不足并非答案：在附于相同面板的可回答问题上，相同模型几乎总是回答，且准确率近乎完美。这也不是信念问题——声明的概率在驱动行动变化达48个百分点的梯度上几乎不动，且得分糟糕。

    arXiv:2608.27167v1 Announce Type: new  Abstract: An LLM agent shown a professional-looking market panel commits to a directional call on a provably unpredictable question far more often than one asked the bare question: across 12 frontier models, commitment rises from 6.5% to 54.0% as evidence is escalated. It commits just as readily when every number on the panel is invented: fabricating the entire display, so nothing the model can see is true except the question itself, still lifts commitment from 24.5% to 36.8%, statistically indistinguishable from the 37.6% produced by genuine market data. What unlocks confident action is not information but the authority of its packaging. The failure is narrow and locatable. Incapacity is not the answer: on matched answerable questions attached to the same panels, the same models answer essentially always, at near-perfect accuracy. Nor is it belief - stated probabilities barely move across the gradient that swings action by 48 points, and score wo
    
[^235]: 行星预测引擎：通过智能数据选择和基础模型嵌入实现自主地理空间预测

    Planetary Prediction Engine: Autonomous Geospatial Prediction via Intelligent Data Selection and Foundation Model Embeddings

    [https://arxiv.org/abs/2608.26088](https://arxiv.org/abs/2608.26088)

    行星预测引擎是一个自主AI系统，能从自然语言查询直接端到端执行地理空间预测，通过智能数据选择和基础模型嵌入，自动整合多模态数据并搜索最优模型，以应对全球性挑战。

    

    应对从粮食安全、灾害风险到疾病爆发和社会经济脆弱性等关键全球挑战，需要高保真度的地理空间建模。然而，构建预测性行星模型仍受制于碎片化的数据生态系统，需要手动数据检索、多模态数据整理和融合以及迭代模型选择。我们提出了行星预测引擎（PPE），这是一种自主AI系统，可直接从自然语言查询中执行端到端工作流程。PPE动态合成多模态数据集，在开放网络和地球观测平台（Data Commons、Google Earth Engine）上检索时空相关协变量，并将其与地理空间基础模型嵌入（PDFM、AlphaEarth）融合。同时，它通过自动过拟合防护搜索针对任务定制的模型架构家族。在多样化的任务、地理区域和科学领域中，该引擎展现出显著性能。

    arXiv:2608.26088v1 Announce Type: cross  Abstract: Addressing critical global challenges, from food security and disaster risk to disease outbreaks and socio-economic vulnerability, demands high-fidelity geospatial modeling. However, building predictive planetary models remains bottlenecked by a fragmented data ecosystem, requiring manual data retrieval, multimodal data curation and fusion along with iterative model selection. We present the Planetary Prediction Engine (PPE), an autonomous AI system that executes this end-to-end workflow directly from natural-language queries. PPE synthesizes multimodal datasets on the fly, retrieving spatiotemporally relevant covariates across open-web and Earth observation platforms (Data Commons, Google Earth Engine) and fusing them with geospatial foundation model embeddings (PDFM, AlphaEarth). Simultaneously, it searches over task-tailored model architecture families with automated overfitting guards. Across diverse tasks, geographies, and scienti
    
[^236]: PhysElite：大语言模型距离解决奥赛级物理问题还有多远？

    PhysElite: How Far Are LLMs from Solving Olympiad-Level Physics Problems?

    [https://arxiv.org/abs/2608.25097](https://arxiv.org/abs/2608.25097)

    提出了PhysElite，一个包含11,586道奥赛级物理问题、附视觉图表和中英双语分步解答的大规模双语多模态基准，并借此评估了18个开源与闭源多模态大语言模型的物理推理能力。

    

    评估（多模态）大语言模型在物理问题上的表现，需要能够反映专家级物理推理难度和广度的基准测试。现有的物理基准测试在以下两个重要方面存在局限：（1）缺乏高难度数据集，（2）对视觉形式、知识点以及分步解题过程的覆盖不够全面。因此，模型在现有数据集上的表现可能无法充分代表其解决复杂物理问题的能力。为解决这些问题，我们提出了PhysElite——一个面向奥赛级物理推理的大规模双语多模态基准。PhysElite包含11,586道奥赛级别的问题。对于每个问题，我们提供了相应的视觉图表、中英双语的分步解题推导过程以及最终答案。我们对18个开源和闭源多模态大语言模型进行了基准测试，发现即使是最强的模型（摘要在此处截断）……

    arXiv:2608.25097v2 Announce Type: replace  Abstract: Understanding how (multimodal) large language models perform on physics problems requires benchmarks that reflect the difficulty and breadth of expert-level physical reasoning. Existing physics benchmarks remain limited in the following two important ways: (1) short of high-difficulty datasets, and (2) lack of comprehensive coverage of visual forms, knowledge points, and step-by-step solution processes. As a result, model performance on current datasets may not be fully representative of their ability to solve complex physics problems. To address these issues, we present PhysElite, a large-scale bilingual multimodal benchmark for Olympiad-level physics reasoning. PhysElite contains 11,586 Olympiad-tier problems. For each problem, we provide corresponding visual diagrams, step-by-step bilingual Chinese-English solution derivations, and the final answer. We benchmark 18 open-source and closed-source MLLMs, and find that even the strong
    
[^237]: PolyChirp：基于TinyML的低功耗声学传感器多物种鸟类鸣声分类

    PolyChirp: Multi-Species Birdsong Classification Using TinyML on Low-Power Acoustic Sensors

    [https://arxiv.org/abs/2608.23101](https://arxiv.org/abs/2608.23101)

    PolyChirp通过结合生物专业知识、自动化数据集和NPU加速的微型多类模型，首次实现了低功耗微控制器上多物种鸟类鸣声的实时分类。

    

    arXiv:2608.23101v1 公告类型：交叉 摘要：TinyML领域的最新进展表明，基于微控制器的低功耗硬件能够利用声学传感器数据，在一次电池充电的情况下，实时监测整个繁殖期内的鸟类物种。然而，目前低功耗微控制器上的先进技术仅限于单一物种的二分类。相比之下，实际的动物监测部署往往需要同时针对多个物种。为应对这一挑战，我们开发了PolyChirp，一种结合生物领域专业知识、自动化数据集整理、神经架构优化和新型硬件的方法，以实现野外多类鸟类物种检测。PolyChirp基于新设计的微型多类模型，利用最新的微控制器和带有神经处理单元（NPU）的硬件加速。我们评估了这些模型的预测性能，并测量了它们的计算效率。

    arXiv:2608.23101v1 Announce Type: cross  Abstract: Recent progress in the field of TinyML has demonstrated that low-power hardware based on microcontrollers can achieve bird species monitoring in real time based on acoustic sensor data for an entire breeding period on a single battery charge. However, the state of the art on low-power microcontrollers was so far limited to binary classification of a single species. In contrast, real fauna monitoring deployments often target multiple species simultaneously. To address this challenge we develop PolyChirp, an approach combining biological domain expertise, automated dataset curation, neural architecture optimization and novel hardware to achieve multiclass bird species detection in the wild. PolyChirp is based on newly designed tiny multiclass models that leverage recent microcontrollers and hardware acceleration with a neural processing unit (NPU). We evaluate the predictive performance of these models, and we measure their computational
    
[^238]: 治理记录作为监督：验证者选择的自我训练用于结构化工作流修复

    Governance Records as Supervision: Verifier-Selected Self-Training for Structured Workflow Repair

    [https://arxiv.org/abs/2608.18324](https://arxiv.org/abs/2608.18324)

    本研究提出一种利用机器可验证工作流产生的治理记录进行自我训练的方法，使有限模型通过验证者选择的计划提升一次性执行能力，显著提高成功率和效率。

    

    arXiv:2608.18324v1 公告类型：新 摘要：机器可验证的工作流产生治理记录，这些记录将任务合同、模型尝试、验证者决策、接受输出和目标来源关联起来。我们测试这些记录是否能监督有限模型，将偶尔或昂贵的能力巩固为可靠的一次性执行。在全新的、结构不相交的PlanBench重新规划案例中，Qwen3-14B思考模式生成了24个计划，这些计划被独立编写的VAL验证者接受。这些计划训练了同一检查点用于非思考执行，无需预言机目标或更强教师。在80个未打开案例中，VAL接受的计划从1个增加到57个，其中56个配对改进且零回归；思考模式达到30个。适配器在所有案例中模式有效，并使用约1/56的思考模式平均延迟。单独的配对接口-治愈门未通过。匹配消融固定了源案例、52个候选池、24个目标计数、模型、配方和种子，同时比较了对照组。

    arXiv:2608.18324v1 Announce Type: new  Abstract: Machine-verifiable workflows produce governance records linking a task contract, model attempt, verifier decision, accepted output, and target origin. We test whether these records can supervise bounded models, consolidating occasional or expensive capability into reliable one-shot execution.   On fresh, structure-disjoint PlanBench replanning cases, Qwen3-14B thinking generated 24 plans admitted by the independently authored VAL verifier. Those plans trained the same checkpoint for non-thinking execution, without oracle targets or a stronger teacher. On 80 unopened cases, VAL-accepted plans increased from 1 to 57, with 56 paired gains and zero regressions; thinking reached 30. The adapter was schema-valid on all cases and used approximately 1/56 of thinking's mean latency. The separate paired interface-cure gate did not pass.   A matched ablation fixed the source cases, 52-candidate pool, 24-target count, model, recipe, and seed while c
    
[^239]: 过于自信而不安全：用于可靠日志异常检测的模型校准

    Too Sure to Be Safe: Model Calibration for Reliable Log Anomaly Detection

    [https://arxiv.org/abs/2608.17965](https://arxiv.org/abs/2608.17965)

    本文提出LoRD框架，通过从正确分类样本的潜在表示中学习可靠性模型，解决日志异常检测器中置信度校准不良的问题，确保错误预测不被过度自信。

    

    在线日志异常检测对于维护大规模计算系统的可靠性至关重要。尽管基于语言模型的日志异常检测器取得了强大的检测性能，但其置信度估计仍校准不佳。我们表明，这些检测器经常对错误预测赋予过高的置信度，尤其是在严重类别不平衡下的异常日志中。此外，即使传统校准指标显示校准良好，错误预测的置信度仍持续偏高，这为运维监控系统造成了关键可靠性缺口。为解决此问题，我们提出了日志重建与距离（LoRD），一种轻量级的事后校准框架，用于可靠的日志异常检测。LoRD从正确分类的验证样本的潜在表示中学习预测路径特定的可靠性模型，并估计预测可靠性阈值。

    arXiv:2608.17965v1 Announce Type: cross  Abstract: Online log anomaly detection is critical for maintaining the reliability of large-scale computing systems. Although recent language model-based log anomaly detectors achieve strong detection performance, their confidence estimates remain poorly calibrated. We show that these detectors frequently assign excessive confidence to incorrect predictions, particularly for anomalous logs under severe class imbalance. Moreover, confidence on erroneous predictions remains persistently high even when conventional calibration metrics indicate good calibration, creating a critical reliability gap for operational monitoring systems. To address this issue, we propose Log Reconstruction and Distance (LoRD), a lightweight post-hoc calibration framework for reliable log anomaly detection. LoRD learns prediction-route-specific reliability models from latent representations of correctly classified validation samples and estimates prediction reliability th
    
[^240]: 无答案准入：基于无标签认证与经验学习的LLM优化建模

    Admission Without Answers: Label-Free Certification and Experience Learning for LLM-Based Optimization Modeling

    [https://arxiv.org/abs/2608.15565](https://arxiv.org/abs/2608.15565)

    本文提出AdmitOR，一种基于校准外部行为证据的无标签准入门控方法，用于LLM优化建模中的经验学习，以解决无答案流中知识接纳不可靠的问题。

    

    arXiv:2608.15565v1 公告类型：新 摘要：用于优化建模的经验学习智能体通过存储已验证的技能来改进，但现有学习者通过检查已知答案来接纳知识，而真实的票务流并不提供这些答案。自然的无标签替代方案不可靠：在一个包含300个问题的无标签盲流中，接纳每个可执行模型大约每四个接纳中就有一个被污染，而单实例一致性仅接受在某个值上匹配但在其他位置不同的模型。我们提出AdmitOR，一个基于校准的外部行为证据的接纳门控。来自三个模型家族、提示策略和求解器堆栈的候选者在从提取的参数域重新采样的实例上运行；跨所得值函数轨迹的一致性通过跨家族团进行总结，校准阈值返回接受、弃权或升级。预注册的假发现标准在校准数据上成立，但在野外流中不成立。我们重新...

    arXiv:2608.15565v1 Announce Type: new  Abstract: Experience-learning agents for optimization modeling improve by storing verified skills, but existing learners admit knowledge by checking against known answers, which real ticket streams do not provide. The natural label-free alternatives are unreliable: on a 300-problem label-blind stream, admitting every executable model poisons roughly one admission in four, while single-instance agreement accepts models that match at one value but differ elsewhere. We propose AdmitOR, an admission gate built on calibrated external behavioral evidence. Candidates from three model families, prompting strategies, and solver stacks are run on instances resampled from an extracted parameter domain; agreement across the resulting value-function traces is summarized by a cross-family clique, and a calibrated threshold returns accept, abstain, or escalate. The preregistered false-discovery criterion holds on calibration data but not on the wild stream. We r
    
[^241]: 保留未来，去掉展开：RIFT用于世界动作模型

    Keep the Future, Drop the Rollout: RIFT for World Action Models

    [https://arxiv.org/abs/2608.11521](https://arxiv.org/abs/2608.11521)

    本文发现世界动作模型可重用固定未来缓存而非迭代展开，提出RIFT方法，在保持高成功率的同时大幅降低部署延迟。

    

    世界动作模型（WAMs）根据预测的未来来调节机器人动作，但迭代视频展开会增加部署延迟。我们提出疑问：动作生成是否需要演进的展开轨迹，还是仅需其未来表示。在全部40个LIBERO任务中的四个WAMs上，配对的闭环干预表明，掩蔽或重新分配未来缓存值会改变执行并降低成功率，这表明模型对未来值及其分配位置敏感。然而，对于Joint和Cosmos-2模型，重放一个固定的最终清洁键/值（K/V）缓存几乎能保持未修改的执行，末端执行器平均位移误差为1.7到1.9厘米，成功率为97.9%到98.2%。这分离了缓存消费与生产：这些模型可以重用固定缓存，但仍需迭代展开来构建它。因此，我们提出RIFT（通过未来令牌的无展开想象），它使用学习到的预期机制。

    arXiv:2608.11521v1 Announce Type: cross  Abstract: World action models (WAMs) condition robot actions on predicted futures, but iterative video rollout increases deployment latency. We ask whether action generation requires the evolving rollout trajectory or only its future representation. Across four WAMs on all 40 LIBERO tasks, paired closed-loop interventions show that masking or reassigning future-cache values changes execution and reduces success, indicating sensitivity to future values and their assigned positions. For Joint and Cosmos-2, however, replaying one fixed final-clean key/value (K/V) cache nearly preserves unmodified execution, with $1.7$ to $1.9$~cm end-effector average displacement error and $97.9\%$ to $98.2\%$ success. This separates cache consumption from production: these models can reuse a fixed cache but still require iterative rollout to construct it. We therefore propose RIFT (\emph{Rollout-free Imagination via Future Tokens}), which uses learned anticipation
    
[^242]: 超越预测：将波动率控制重塑为路由问题

    Beyond Forecasting: Recasting Volatility Control as a Routing Problem

    [https://arxiv.org/abs/2608.10375](https://arxiv.org/abs/2608.10375)

    本文提出VolRouter框架，创新性地将波动率控制重构为基于市场状态的条件路由问题，在估计器-控制器组合对之间动态切换，在四个基准测试中的三个取得最高夏普比率，并将标普500的夏普比率从0.952提升至1.22。

    

    波动率控制将风险估计转化为投资组合敞口，然而现有方法通常依赖于固定的波动率估计器或预先定义的控制规则，可能无法适应不断变化的市场环境。我们提出了VolRouter，一个模块化框架，将波动率控制形式化为在估计器-控制器组合对上进行状态条件化的路由问题。VolRouter首先将市场状况总结为与控制相关的状态画像，然后通过三个阶段执行路由：状态推断、切换审查和组合对选择。该路由器可以基于规则、可学习模型或大语言模型（LLM）决策模块来实现，而投资组合操作仍由预定义的控制策略生成。我们在标普500、多资产、比特币和USDT波动率控制场景中评估了VolRouter。VolRouter在四个场景中的三个中取得了最高的夏普比率。在标普500上，它将夏普比率从“已实现波动率+朴素缩放”策略的0.952提升至1.22。

    arXiv:2608.10375v2 Announce Type: replace-cross  Abstract: Volatility control converts risk estimates into portfolio exposure, yet existing approaches often rely on a fixed volatility estimator or a pre-defined control rule that may not adapt to changing market conditions. We propose VolRouter, a modular framework that formulates volatility control as state-conditioned routing over estimator-controller pairs. VolRouter first summarizes market conditions into a control-relevant state profile and then performs routing through three stages: state inference, switch review, and pair selection. The Router can be implemented using rule-based, learnable, or LLM-based decision modules, while portfolio actions remain generated by predefined control policies. We evaluate VolRouter across S&P 500, Multi-Asset, Bitcoin, and USDT volatility-control settings. VolRouter achieves the highest Sharpe ratio in three of four settings. On S&P 500, it improves Sharpe from 0.952 for RV + Naive Scaling to 1.22
    
[^243]: 迈向统一的动态人脸关键点检测

    Towards Unified Dynamic Face Landmark Detection

    [https://arxiv.org/abs/2608.10346](https://arxiv.org/abs/2608.10346)

    该论文提出人脸部位锚定关键点位置（FPALP）表示法，将关键点统一表示为人脸部位轮廓上的进度值，从而实现所有N点数据集的统一训练和动态数量的关键点输出。

    

    尽管人脸关键点检测（FLD）方法不断进步并持续突破性能边界，但它们忽视了两个主要的功能局限：（1）每个"N点"基准数据集都需要独立训练不同的网络参数；（2）在"N点"数据集上训练的模型只能可靠地输出这N个关键点。在本工作中，我们首先提出了人脸部位锚定关键点位置（FPALPs）的概念，其中每个关键点被视为人脸部位轮廓上从零（起点）到一（终点）之间的进度值。无论关键点来自哪个数据集，都可以用FPALP格式表示，从而使所有"N点"数据集能够统一合并为单一数据集。其次，我们用基于FPALP的查询来表示每个关键点，通过跨模态解码器对其进行逐步细化，并基于最终表示预测其坐标。我们的方法被称为Unifi（摘要截断）

    arXiv:2608.10346v2 Announce Type: replace-cross  Abstract: Although advancements in face landmark detection (FLD) methods continue to push performance boundaries, they overlook two major functional limitations: (1) different network parameters need to be trained independently for each ``$N$-point'' benchmark dataset, and (2) a model trained on an ``$N$-point'' dataset reliably outputs only the $N$ landmarks. In our work, we first conceptualize Face Part-Anchored Landmark Positions (FPALPs), wherein each landmark is treated as a progression value between zero (start) and one (end) along a face part's contour. Every landmark can be expressed in the FPALP format, irrespective of its source dataset, hence unlocking the ability to unify all ``$N$-point'' datasets into a single dataset. Secondly, we represent each landmark with an FPALP-based query, refine it progressively with a cross-modality decoder, and predict its coordinates based on the final representation. Our approach, called Unifi
    
[^244]: Aftab：并行化Q网络中CNN编码器与先进价值函数的综合基准

    Aftab: A Comprehensive Benchmark of CNN Encoders and Advanced Value Functions in Parallelized Q-Networks

    [https://arxiv.org/abs/2608.07335](https://arxiv.org/abs/2608.07335)

    本文系统评估了八种CNN编码器在并行化Q网络中的性能，并结合Hadamax编码与多种价值函数头，提出了一个在Atari-57上表现优异的复合架构。

    

    arXiv:2608.07335v2 公告类型：替换交叉 摘要：深度强化学习的最新进展日益倾向于简化、高度并行化的范式。值得注意的是，并行化Q网络（PQN）算法能够在无需经验回放缓冲区或目标网络的情况下进行离策略价值学习。然而，在这些无缓冲区设置中运行的视觉编码器的表示能力和计算效率仍相对未被充分探索。在本工作中，我们系统性地研究了PQN内卷积神经网络的架构设计空间。我们评估了八种不同的CNN拓扑结构，同时明确表征了它们的参数和计算需求。我们进一步通过将Hadamax编码范式与分类、集成和决斗价值头集成，研究了乘性表示学习和先进价值估计的效果。在Atari-57上的广泛实验表明，我们最终的复合架构...

    arXiv:2608.07335v2 Announce Type: replace-cross  Abstract: Recent advancements in deep reinforcement learning have increasingly favored simplified, highly parallelized paradigms. Notably, the Parallelized Q-Network (PQN) algorithm enables off-policy value learning without relying on experience replay buffers or target networks. However, the representational capacity and computational efficiency of visual encoders operating in these buffer-free settings remain comparatively underexplored. In this work, we systematically investigate the architectural design space of Convolutional Neural Networks within PQN. We evaluate eight distinct CNN topologies while explicitly characterizing their parameter and computational requirements. We further study the effect of multiplicative representation learning and advanced value estimation by integrating the Hadamax encoding paradigm with categorical, ensemble, and dueling value heads. Extensive experiments on Atari-57 show that our final composite arc
    
[^245]: 将意图与轨迹解耦：面向世界动作模型的表征推演框架

    Decoupling Intention from Trajectory: A Representational Deduction Framework for World Action Models

    [https://arxiv.org/abs/2608.06994](https://arxiv.org/abs/2608.06994)

    该论文提出PILOT框架，通过将运动思维链引导作为模型原生能力的“表征推演”机制，解耦世界动作模型中高层物理状态演化与低层动作轨迹生成之间的表征纠缠，从而增强世界演化建模对动作生成的预测与指导能力。

    

    世界动作模型旨在构建一个统一的架构，既能理解世界状态的演化，又能指导生成式的运动规划。然而，现有的视觉分支侧重于预测静态的视觉观测，而非反映能够捕捉运动交互下世界状态演变的潜在转移信息。这导致动作模型内部高层物理条件演化与低层动作轨迹生成之间产生表征纠缠，形成结构性瓶颈，同时削弱了世界演化建模对动作生成的预测能力。我们提出了PILOT（面向潜在优化轨迹的物理推理），其核心的“表征推演”机制通过将运动思维链引导作为模型的原生能力来弥合这一差距。具体而言，RD旨在鼓励动作分支显式建模……

    arXiv:2608.06994v2 Announce Type: replace-cross  Abstract: World Action Models (WAMs) aim to construct a unified architecture capable of understanding world state evolution and guiding to generative motion planning. However, existing visual branches focus on predicting static visual observation, rather than reflecting potential transition information that captures the evolution of world states under motion interactions. This leads to representational entanglement between high-level physical condition evolution and low-level action trajectory generation within the Action Model, creating a structural bottleneck while weakening the predictive capability of world evolution modeling for action generation. We propose PILOT (Physical Inference for Latent Optimized Trajectories), whose core Representational Deduction (RD) bridges this gap by integrating motion thought-of-chain (CoT) guidance as a native model capability. Specifically, RD aims to encourage the action branch to explicitly model 
    
[^246]: 并非所有发散都应被抑制：在线策略蒸馏中的反事实可恢复性

    Not Every Divergence Should Be Suppressed: Counterfactual Recoverability in On-Policy Distillation

    [https://arxiv.org/abs/2608.04408](https://arxiv.org/abs/2608.04408)

    本文提出反事实可恢复性框架，通过教师续写与回滚分支重放错误状态来区分可恢复与不可逆的错误，并据此决定在线策略蒸馏中对轨迹的保留、回滚或常规监督策略，其可恢复性代理指标AUC达1.000，远超仅依赖发散度指标的0.392。

    

    在线策略蒸馏（OPD）对学生模型访问过的轨迹进行监督，然而基于发散度的规则无法判断一个错误前缀是否仍然可以被纠正。我们将这一决策问题形式化为反事实可恢复性，并通过预算匹配的教师续写分支与回滚分支对每个错误状态进行重放。根据二者的相对成功率，状态被分类为可恢复的、不可逆但可避免的、或模糊的，这些标签指导训练是保留、回滚还是常规监督相应的轨迹。在AIME分支诊断中，可恢复状态的平均“续写减回滚”效应为0.185，而不可逆但可避免的状态为-1.000，表明二者具有截然相反的干预偏好。基于分支实验导出的可恢复性代理指标达到了1.000的AUC，大幅优于仅使用发散度指标的0.392。在冻结评估中，可恢复性感知的控制方法取得了……（原文摘要此处截断）

    arXiv:2608.04408v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) supervises student-visited trajectories, yet divergence-based rules cannot determine whether an erroneous prefix remains correctable. We formulate this decision as counterfactual recoverability and replay each error state through budget-matched teacher-continuation and rollback branches. Based on their relative success, states are categorized as recoverable, irreversible-but-avoidable, or ambiguous, and these labels guide whether training retains, rolls back, or conventionally supervises the corresponding trajectory. On AIME branch diagnostics, the mean continuation-minus-rollback effect is 0.185 for recoverable states and -1.000 for irreversible-but-avoidable states, demonstrating opposite intervention preferences. A branch-derived recoverability proxy achieves an AUC of 1.000, substantially outperforming divergence alone at 0.392. Across frozen evaluations, recoverability-aware control achieves th
    
[^247]: 状态传播亦能胜任：一种用于确定性状态追踪的复值状态空间模型

    State Propagation Also Satisfies: A Complex-Valued State-Space Model for Deterministic State Tracking

    [https://arxiv.org/abs/2608.03425](https://arxiv.org/abs/2608.03425)

    提出复数状态传播器（CSP），一种仅依赖atan2激活在复相位流形上运行的极简循环范式，实现相位信号在极深网络中的零衰减传播，从而解决Transformer和Mamba在确定性状态追踪中的分布外崩溃问题。

    

    尽管大规模语言模型占据主导地位，但Transformers和Mamba等主流范式在连续确定性状态追踪方面存在根本性缺陷，在泛化到更长序列时会遭受灾难性的分布外（OOD）崩溃。为了打破这一瓶颈，我们提出了复数状态传播器，这是一种极度简约的循环范式，严格在复相位流形上运行，无需中间输出投影、幅度调制或逐步非线性变换。至关重要的是，我们揭示了一个前所未有的架构奇迹：相位信号在极深网络中得以存活，且信息衰减为零。通过实现由atan2(y, x)激活函数控制的精确四象限坐标到相位（C-to-2）变换，CSP迫使连续优化景观与离散循环群无缝对齐。值得注意的是……

    arXiv:2608.03425v3 Announce Type: replace  Abstract: Despite the dominance of massive language models, leading paradigms like Transformers and Mamba fundamentally falter at continuous deterministic state tracking, suffering from catastrophic out-of-distribution (OOD) collapse when generalizing to extended sequences. To shatter this bottleneck, we present the \textbf{Complex State Propagator (CSP)}, a radically minimalist recurrent paradigm that operates strictly on the complex phase manifold without intermediate output projections, amplitude modulations, or per-step non-linearities. Crucially, we unveil an unprecedented architectural marvel: \textbf{the phase signal survives extreme depth with absolute zero informational attenuation}. By implementing an exact four-quadrant \textbf{Coordinate-to-Phase (C-to-2)} transformation governed by the \(\text{atan2}(y, x)\) activation, CSP forces the continuous optimization landscape to seamlessly align with discrete cyclic groups. Remarkably, wi
    
[^248]: 本地部署大语言模型的能效研究：基于消费级硬件的GPU功耗定量初步基准测试

    Energy Efficiency of Locally Deployed LLMs: A Preliminary Quantitative GPU Power Benchmark on Consumer Hardware

    [https://arxiv.org/abs/2608.00008](https://arxiv.org/abs/2608.00008)

    本文在消费级GPU上对18个开源大语言模型进行了可复现的硬件级能耗基准测试，发现模型架构和量化策略（而非仅仅参数量）才是决定能效的关键因素，并给出了各模型的单位token能耗与吞吐量排名。

    

    由于隐私顾虑以及对本地推理的需求，大语言模型（LLM）的本地部署正日益受到关注。然而，消费级硬件上的能耗成本仍然缺乏充分表征，因为大多数基准测试仅关注准确性。本文对在单张消费级GPU（RTX 4060Ti 16GB）上运行的18个开源大语言模型（参数量从0.5B到7B）进行了可复现的硬件级能耗基准测试。使用Ollama推理引擎，通过nvidia-smi以2Hz的采样频率在固定提示集上采集GPU功耗。我们评估了平均/峰值功耗、每个提示的总能耗、每个输出token的能耗以及吞吐量。研究结果表明，除原始参数量之外，模型架构和量化策略等因素也在驱动能效表现。具体而言，qwen2.5:0.5b和tinyllama:1.1b实现了最低的能耗成本（分别为0.2747 J/tok和0.3234 J/tok）以及最高的吞吐量（>325……）

    arXiv:2608.00008v2 Announce Type: replace  Abstract: The local deployment of large language models (LLMs) is gaining traction due to privacy concerns and the desire for on-premise inference. However, the energy costs on consumer hardware remain poorly characterized, as most benchmarks focus solely on accuracy. This paper presents a reproducible, hardware-level energy benchmark of 18 open-source LLMs (0.5B to 7B parameters) executed on a single consumer GPU (RTX 4060ti 16GB). Using the Ollama inference engine, GPU power draw was sampled at 2hz via nvidia-smi across a fixed prompt set. We evaluate mean/peak power, total energy per prompt (J/prompt), energy per output token (J/tok), and throughput (tok/s). Our findings suggest that factors beyond raw parameter count, including model architecture and quantization strategy, drive energy efficiency. Specifically, qwen2.5:0.5b and tinyllama:1.1b achieve the lowest energy cost (0.2747 J/tok and 0.3234 J/tok) and the highest throughput (>325 to
    
[^249]: 小模型足矣：基于LoRA适配器的AI编辑文本个性化风格改写

    Small Is Enough: Per-User Style Rewriting of AI-Edited Text via LoRA Adapters

    [https://arxiv.org/abs/2607.29238](https://arxiv.org/abs/2607.29238)

    InMyStyle提出一种隐私优先的单用户方案，仅对0.5B至7B的小型模型进行LoRA微调，即可让AI编辑的文本自动改写为符合个人写作风格，且实验表明小模型已足以胜任该改写任务。

    

    InMyStyle是一个隐私优先的单用户系统，它使小型语言模型能够将AI编辑的文本改写为符合个人用户写作风格，且在推理时无需指令提示。给定用户的文档，系统利用多个本地辅助大语言模型构建配对训练样本，并在参数量从0.5B到7B的Qwen2.5模型上微调LoRA适配器。借助长度感知的生成预算和自动分块机制，系统可支持不同长度的输入。我们报告了一项单用户案例研究：基于一位作者73段科学写作文本衍生出219个评估样本对，所有适配器均采用相同的rank-8、三轮训练方案。在贪心解码和采样解码两种方式下，自动综合评分（0-1量表）在各模型规模上均趋于平稳（Q=0.689-0.695，置信区间相互重叠）。在此设置下，小模型足以完成所测量的改写任务，而模型规模主要决定了……（摘要原文在此处截断）。

    arXiv:2607.29238v2 Announce Type: replace-cross  Abstract: InMyStyle is a privacy-first, single-user system that adapts small language models to rewrite AI-edited text towards an individual user's writing style without an instruction prompt at inference. Given a user's documents, it uses multiple local helper LLMs to construct paired training examples and fine-tunes LoRA adapters on Qwen2.5 models ranging from 0.5B to 7B parameters. Length-aware generation budgets and automatic chunking support inputs of different lengths. We report a single-user case study: 219 evaluation pairs derived from 73 paragraphs of one author's scientific writing, with all adapters trained using the same rank-8, three-epoch recipe. The automatic composite score (0-1 scale) plateaus across model sizes under both greedy and sampled decoding ($Q=0.689$-$0.695$, with overlapping confidence intervals). In this setting, small models are sufficient for the measured rewriting task, and model size mainly determines ef
    
[^250]: 思考精简，智能委托，行动，重复：边缘LLM代理的校准推理与不确定性感知委托

    Think Short, Defer Smart, Act, and Repeat: Calibrated Reasoning and Uncertainty-Aware Deferral for Edge LLM Agents

    [https://arxiv.org/abs/2607.26865](https://arxiv.org/abs/2607.26865)

    TSDS框架通过轻量级收敛探针和基于困惑度的委托规则，在边缘LLM代理中实现推理预算与可靠性的平衡，并利用多目标LTT程序提供同时的有限样本保证。

    

    arXiv:2607.26865v2 公告类型：替换-交叉 摘要：遵循ReAct范式的LLM代理是实现复杂多步任务（包括多跳问答、代码生成和物理AI系统控制）的有前景的使能器。然而，当部署在边缘时，它们必须严格管理推理预算，同时保持可靠性，并且仅在本地不确定性过高而无法安全行动时，才委托给云端模型。我们提出“思考精简，智能委托”（TSDS）框架，该框架协同整合了一个轻量级收敛探针（一旦预期行动稳定即停止设备端推理）与一个基于困惑度的委托规则（将不确定行动升级到云端模型）。两种机制通过多目标“学习-然后-测试”（LTT）程序在端到端情节轨迹上联合校准，同时提供关于预期情节奖励和云端调用率的有限样本保证。我们在四个ReAct基准上评估TSDS，涵盖...

    arXiv:2607.26865v2 Announce Type: replace-cross  Abstract: LLM agents following the ReAct paradigm are promising enablers of complex multi-step tasks, including multi-hop question answering, code generation, and control of physical AI systems. Yet, when deployed at the edge, they must tightly manage their reasoning budget while remaining reliable and deferring to a cloud-side model only when local uncertainty is too high to act safely. We propose Think Short, Defer Smart (TSDS), a framework that synergistically integrates a lightweight convergence probe, which halts on-device reasoning once the intended action has stabilized, with a perplexity-based deferral rule that escalates uncertain actions to a cloud-side model. Both mechanisms are jointly calibrated on end-to-end episode trajectories via a multi-objective Learn-Then-Test (LTT) procedure, providing simultaneous finite-sample guarantees on expected episode reward and cloud-call rate. We evaluate TSDS on four ReAct benchmarks spann
    
[^251]: 深度而非广度：超越表面噪声的Best-of-N越狱攻击

    Depth, Not Breadth: Best-of-N Jailbreaking Beyond Surface Noise

    [https://arxiv.org/abs/2607.26639](https://arxiv.org/abs/2607.26639)

    该研究将 Best-of-N 越狱攻击的查询预算从表面文本扰动转向结构性代码补全编码，对最强自检防御 SAGE 实现了高达各部分效果之和 9 至 75 倍的攻击成功率提升，证明攻击方差的结构性分布比表面噪声更重要。

    

    Best-of-N 越狱攻击将查询预算花费在表面变化上——通过打乱和改变大小写等方式重复发送请求，直到某一次尝试成功。我们探究当把这种变化的方差转移到结构性通道中时，同样的预算能带来什么效果——在两个实验组中保持搜索方式完全一致，从而使编码方式成为唯一差异。针对 SAGE（目前公开发表的最强自检式防御），基于代码补全编码的 Best-of-N 在三个开源权重目标模型上分别达到 67%、22% 和 15% 的行为攻击成功率，而该编码单次使用最多仅达到 4.7%，公开发表的字符搜索在全预算下最多为 3.0%：组合效果达到各部分之和的 9 至 75 倍，且自助法置信区间在每个目标上都同时排除了这两个单独成分。我们在报告头条数字的同时也报告可操作的数字：在实际的严重性阈值下，这些数字分别为 24、8 和 1 个行为（95% 置信区间 [13, 28]、[3, 13]、[0, 3]）。一个将编码与变化分离的 2x2 实验表明，两种防御……（摘要在此处截断）

    arXiv:2607.26639v2 Announce Type: replace-cross  Abstract: Best-of-N jailbreaking spends a query budget on surface variation, scrambling and recasing a request until one draw lands. We ask what a budget buys when its variance is moved into a structural channel instead, holding the search identical across both arms so the encoding is the only difference. Against SAGE, the strongest published self-check defense, best-of-N over a code-completion encoding reaches 67, 22 and 15% of behaviors on three open-weight targets, where that encoding fired once reaches at most 4.7% and the published character search at full budget at most 3.0%: 9 to 75 times the sum of the parts, with bootstrap intervals clearing both ingredients on every target. We report the operative figure beside the headline rather than the headline alone: at the actionable severity threshold those cells read 24, 8 and 1 behaviors (95% CI [13, 28], [3, 13], [0, 3]). A 2x2 holding encoding and variation apart shows the two defens
    
[^252]: 盲而非弱：针对编码式VLM越狱攻击的“恢复-重防”防御的套件级最优安全-效用前沿

    Blind, Not Weak: A Best-of-Suite Safety-Utility Frontier for Recover-and-Reguard Defenses Against Encoded VLM Jailbreaks

    [https://arxiv.org/abs/2607.26574](https://arxiv.org/abs/2607.26574)

    该论文构建了一个“恢复-重防”预处理器，在安全防护器之前恢复图像内容并解码编码，将图像渲染类越狱攻击的拦截率从零提升至67-90%，并据此刻画了此类防御在安全性与良性流量效用之间的套件级最优权衡边界。

    

    安全分类器（“防护器”）是视觉语言模型（VLM）的主流黑盒防御手段，然而防护器评判的是输入的表层形式而非其含义：一个有害请求若被重新编码为集合论、形式逻辑、古典语言、代码，或以渲染在图像中的文本形式呈现，便能绕过原本会拦截其明文形式的防护器——这便是“解码鸿沟”。标准的补救方案是在防护器之前部署一个预处理器，用以恢复图像内容并解码编码。我们构建了这样一个预处理器，并在一个由十一种编码攻击组成的集成攻击集上对其进行评估——其中包含六个已发表的攻击实现、一个标准编码基线、一个改编版本以及三个作者自建的渲染攻击——只要任一攻击得手即判定该行为被攻破。正是恢复出防护器从未见过的视图才换来了覆盖率的提升——图像渲染攻击的拦截率从恰好为零提升至67-90%——而其在良性流量上的代价由防护器而非机制本身决定：某个防护器需付出9个百分点的良性拦截代价。

    arXiv:2607.26574v3 Announce Type: replace-cross  Abstract: Safety classifiers ("guards") are the dominant black-box defense for vision-language models, yet a guard judges an input's surface form, not its meaning: a harmful request re-encoded as set theory, formal logic, a classical language, code, or text rendered inside an image slips past a guard that would block it in plain language - the decode gap. The standard fix is a preprocessor that recovers image content and decodes the encoding before the guard. We build one and evaluate it against an ensemble of eleven encoding attacks - six published implementations, one standard encoding baseline, one adapted and three author-constructed renders - counting a behavior as broken if any attack succeeds. Restoring a view the guard never had is what buys coverage - block rates on image renders go from exactly zero to 67-90% - and what it costs in benign traffic is set by the guard, not by the mechanism: one guard pays 9 benign blocking points
    
[^253]: 廉价探针何时能预测昂贵训练？探究3D-CT编码器用于文本生成

    When Do Cheap Probes Predict Expensive Training? Probing 3D-CT Encoders for Text Generation

    [https://arxiv.org/abs/2607.22771](https://arxiv.org/abs/2607.22771)

    本文提出CheapCT廉价探针方法，证明其能高精度预测3D CT编码器微调后的生成性能，从而避免昂贵的全模型微调搜索。

    

    arXiv:2607.22771v2 公告类型：替换-交叉 摘要：构建一个3D CT视觉语言模型，首先需要选择基于哪个图像编码器。目前，这一选择是通过将每个候选编码器通过完整的语言模型进行微调，并比较下游得分来完成的，这是一个极其昂贵的搜索过程。对编码器表示进行廉价探针测试提供了一种出路，但这种方法是否能预测昂贵训练的结果从未被验证过。我们通过CheapCT在报告生成任务以及我们构建的新VQA数据集MeasureVQA上进行了测试。MeasureVQA逐项能力评估结果，其答案通过分割掩膜和亨氏单位测量。报告生成任务一次性评估整个报告，主要反映疾病情况。探针在所有能力上都能预测昂贵训练的结果。探针与微调之间的排名一致性始终很高，从ρ=0.90到1.00。当用于选择编码器时，CheapCT选择的编码器性能几乎与最佳编码器相当，而仅需微调一个编码器的成本。

    arXiv:2607.22771v2 Announce Type: replace-cross  Abstract: Building a 3D CT vision language model begins with a choice of which image encoder to build on. Today that choice is made by fine-tuning every candidate through the full language model and comparing downstream scores, an enormously expensive search. A cheap probe on the encoder's representation promises a way out, but whether it forecasts the expensive outcome has never been tested. We test this with CheapCT on report generation and on MeasureVQA, a new VQA dataset we build. MeasureVQA scores the outcome one capability at a time, its answers measured from segmentation masks and Hounsfield units. Report generation scores the whole report at once and reflects mostly disease. The probe forecasts expensive training across every capability. The rank agreement between probe and fine-tuning stays high throughout, from $\rho=0.90$ to $1.00$. Used to choose an encoder, CheapCT picks one nearly as good as the best while fine-tuning a sin
    
[^254]: 神经网络是否保留案例结构？基于案例的分解、解释与决策一致性

    Do Neural Networks Preserve Case Structure? Case-Based Decomposition, Interpretation, and Decision Consistency

    [https://arxiv.org/abs/2607.11347](https://arxiv.org/abs/2607.11347)

    该论文在神经网络与基于案例的决策理论（CBDT）之间建立了联系，证明训练后的神经网络能够通过其学习到的表示保留可恢复的案例结构，从而将决策边际分解为单个训练案例的贡献，为模型决策是否真正根植于训练数据提供了可解释性与一致性验证的途径。

    

    神经网络日益影响着具有重大后果的决策，使其可靠性变得愈发重要。然而，其内部机制几乎没有提供证据表明决策是否仍然根植于训练案例之中，以及哪些案例最终支持或反对其输出结果。缺乏决策与训练案例之间的这种联系，用户便无法判断模型是否从数据中学到了可靠的决策模式。这引出了一个根本性问题：神经网络是否保留了案例结构？我们在神经网络与基于案例的决策理论（CBDT）之间建立了联系，表明经过训练的神经网络能够通过其学习到的表示保留一种可恢复的案例结构。这样的结构使得拟合出的决策边际可以被分解为各个案例的独立贡献。我们进一步识别了这种恢复出的案例结构能够获得CBDT解释的条件。我们进一步确立了……（摘要内容不完整，原文在此处截断）

    arXiv:2607.11347v2 Announce Type: replace  Abstract: Neural networks increasingly inform consequential decisions, making their reliability increasingly important. Yet their internal mechanisms provide little evidence of whether decisions remain grounded in the training cases and which cases ultimately support or oppose their outcomes. Without this connection between decisions and training cases, users cannot determine whether a model has learned reliable decision patterns from data. This motivates a fundamental question: do neural networks preserve case structure? We establish a connection between neural networks and Case-Based Decision Theory (CBDT), showing that trained neural networks can preserve a recoverable case structure through their learned representations. Such a structure allows fitted decision margins to be decomposed into individual case contributions. We identify the conditions under which this recovered case structure admits a CBDT interpretation. We further establish d
    
[^255]: CoFL-S：用于局部语言条件导航的空间可查询扇区流场

    CoFL-S: Spatially Queryable Sector Flow Fields for Local Language-Conditioned Navigation

    [https://arxiv.org/abs/2607.02222](https://arxiv.org/abs/2607.02222)

    提出 CoFL-S——一种在机器人局部可见扇区上预测语言条件流场并生成连续轨迹的底层视觉-语言-动作框架，配套将 VLN-CE 片段转换为帧级局部监督的训练方法，以及隔离底层动作接口的连续时间 Habitat 评测基准。

    

    视觉-语言导航（Vision-Language Navigation）的研究日益强调高层指令推理、记忆、全局地图构建和指令分解，而底层动作表示却相对缺乏探索。我们提出了 CoFL-S，一个底层的视觉-语言-动作框架，它在机器人的局部可见扇区上预测语言条件化的流场，并通过滚动展开所预测的流场来生成连续轨迹。为了训练这种底层表示，我们将每个 VLN-CE 片段——原本是整段指令与动作序列的配对——转换为帧级的局部监督，其中包含对齐的子指令以及相匹配的动作、轨迹和稠密流场目标。在评估方面，我们引入了一个连续时间的 Habitat 基准，该基准将底层动作接口与指令分解隔离开来，并通过共享的速度指令控制器执行所有方法，从而实现……

    arXiv:2607.02222v2 Announce Type: replace-cross  Abstract: Vision-Language Navigation has increasingly emphasized high-level instruction reasoning, memory, global map construction, and instruction decomposition, while the low-level action representation remains comparatively underexplored. We propose CoFL-S, a low-level vision-language-action framework that predicts a language-conditioned flow field over the robot's local visible sector and generates continuous trajectories by rolling out the predicted field. To train this low-level representation, we convert each VLN-CE episode, originally a whole-episode instruction paired with an action sequence, into frame-level local supervision with aligned sub-instructions and matched action, trajectory, and dense flow-field targets. For evaluation, we introduce a continuous-time Habitat benchmark that isolates low-level action interfaces from instruction decomposition and executes all methods through a shared velocity-command controller, enabli
    
[^256]: 超越药物发现：纳米技术分子优化（NMO）基准

    Beyond Drug Discovery: The Nanotechnology Molecular Optimization (NMO) Benchmark

    [https://arxiv.org/abs/2606.30170](https://arxiv.org/abs/2606.30170)

    提出纳米技术分子优化基准，用量子模拟取代代理预测器并引入严格协议，将生成式分子设计从药物发现领域拓展至量子材料科学与纳米技术研究。

    

    生成式分子设计目前主要受针对药物类特性的简单代理基准以及在大型制药数据集上预训练的模型所塑造。这种组合虽然能产生亮眼的基准测试指标，却限制了其向与药物发现在结构上截然不同的领域迁移的能力。为了克服这一局限性，并推动分子发现面向真实的、具有科学依据的目标，我们提出了纳米技术分子优化基准，它连接了机器学习（ML）与量子材料科学。NMO 既可以作为机器学习社区的严格测试平台，也可以作为纳米技术研究的发现引擎。该基准套件用量子模拟取代了代理预测器，并引入了严格的评估协议，优先考虑科学实用性而非面向排行榜的过度拟合。基于物理的 NMO 任务施加了严格的结构约束和崎岖的适应度景观，对生成式模型提出了根本性的全新要求。

    arXiv:2606.30170v2 Announce Type: replace-cross  Abstract: Generative molecular design is shaped by simple proxy benchmarks for drug-like properties and models pretrained on large pharmaceutical datasets. This combination yields strong benchmark metrics but limits transferability to domains structurally distinct from drug discovery. To overcome this limitation and drive discovery toward real, scientifically grounded targets, we introduce the Nanotechnology Molecular Optimization (NMO) Benchmark, which bridges machine learning (ML) and quantum materials science. NMO acts simultaneously as a rigorous testbed for the ML community and a discovery engine for nanotechnology research. The suite replaces proxy oracles with quantum simulations and introduces strict protocols that prioritize scientific utility over leaderboard-oriented overfitting. The physics-based NMO tasks impose hard structural constraints and rugged fitness landscapes, posing fundamentally new requirements on generative mod
    
[^257]: 基于项目反应理论的高效安全基准测试

    Efficient Safety Benchmarking via Item Response Theory

    [https://arxiv.org/abs/2606.20626](https://arxiv.org/abs/2606.20626)

    该论文提出将项目反应理论（IRT）与自适应选题方法应用于语言模型安全基准测试，能够恢复可解释的模型能力结构、区分原始指标上的天花板模型，并以远少于全量测试的响应数逼近完整基准的模型排序，大幅提升安全评估效率。

    

    语言模型的安全基准通常采用静态评估范式，将所有测试项视为对所有模型同等信息化，这一假设对于对抗性的、高度异质的安全测试项尤为成问题。若将其完整应用于现代基准套件，当前的评估程序将需要约 $10^5$ 量级的响应，其中大部分响应几乎不提供任何排序信号。我们分析了六个广泛使用的安全基准，并为更高效的安全评估做出了三项贡献。首先，我们证明项目反应理论（IRT）能够从安全基准中恢复出可解释的结构，其能力估计能够区分那些在原始安全指标上聚集于天花板水平的模型之间的差异。其次，我们证明自适应选题方法——根据模型的响应动态为其选择信息量大的测试项——能够逼近完整基准的模型排序（Spearman's ρ > 0.90），从而减少评估……

    arXiv:2606.20626v2 Announce Type: replace-cross  Abstract: Safety benchmarks for language models are typically evaluated using static paradigms that treat all items as equally informative for all models, an assumption that is particularly problematic for adversarial, highly heterogeneous safety items. Applied in full to modern benchmark suites, current evaluation procedures would require on the order of $10^5$ responses, most of which provide little ranking signal. We analyze six widely used safety benchmarks and make three contributions toward more efficient safety evaluation. First, we show that Item Response Theory (IRT) recovers interpretable structure on safety benchmarks, with ability estimates resolving differences among models that cluster at the ceiling of raw safety metrics. Second, we show that adaptive item selection, which dynamically chooses informative items for each model based on its responses, approximates full-benchmark rankings (Spearman's $\rho >$ 0.90), reducing e
    
[^258]: 面向嵌入模型路由的策略后悔：具有低秩专家的上下文赌博机

    Policy Regret for Embedding Model Routing: Contextual Bandits with Low-Rank Experts

    [https://arxiv.org/abs/2606.14929](https://arxiv.org/abs/2606.14929)

    该论文将嵌入模型路由形式化为具有低秩专家的对抗性上下文线性赌博机问题，证明标准后悔度量存在结构性误设或统计不可处理的缺陷，并提出兼具表达能力与高效可学习性的对数二次策略类来实现查询依赖的模型路由。

    

    现代推荐系统日益依赖将多样化的查询动态路由到多个嵌入模型。尽管这一问题具有重要的实践意义，但在对抗性查询、赌博机反馈以及模型可观测性受限等现实条件下，该问题仍未得到充分理解。我们将嵌入模型路由形式化为一个具有低秩专家的对抗性上下文线性赌博机问题，其中上下文对应查询，动作对应物品，专家则对应工作在低秩潜在表示空间上的嵌入模型。我们首先证明了标准的后悔度量会遭遇结构性误设或统计上的不可处理性，并识别出一个对数二次策略类，该策略类既足够富有表现力以刻画依赖于查询的模型路由，又具备足够规整的结构以支持高效的在线学习。聚焦于这一在赌博机反馈下的对数二次策略优化问题——该问题本身亦具有独立的研究价值……

    arXiv:2606.14929v2 Announce Type: replace-cross  Abstract: Modern recommendation systems increasingly rely on dynamically routing diverse queries to multiple embedding models. Despite its practical significance, this problem remains poorly understood under realistic conditions like adversarial queries, bandit feedback, and limited observability of models. We formalize embedding model routing as an adversarial contextual linear bandit with low-rank experts, where contexts are queries, actions are items, and experts are the embedding models working on low-rank latent representation spaces. We first establish that standard regret notions suffer from structural misspecification or statistical intractability, and we identify a log-quadratic policy class that is expressive enough to capture query-dependent model routing, yet structured enough to allow efficient online learning. Focusing on this log-quadratic policy optimization problem under bandit feedback -- which is of independent interes
    
[^259]: SciR：一个面向大语言模型科学推理的可控基准

    SciR: A Controllable Benchmark for Scientific Reasoning in LLMs

    [https://arxiv.org/abs/2606.13020](https://arxiv.org/abs/2606.13020)

    SciR是一个基于形式化对象生成、答案可验证的科学推理基准，覆盖演绎、归纳与因果溯因三种推理范式，并能独立调节信息提取难度与推理难度两个维度，从而实现对大语言模型科学推理能力的可控评估。

    

    三种典型的推理形式贯穿于科学推理之中：演绎、归纳和因果溯因。目前在科学场景下对大语言模型进行这些推理能力的可靠评估尚难实现：基于人工标注的科学基准成本高昂且缺乏机制层面的真值，而合成的逻辑推理基准又与真实科学文档相去甚远。我们提出了SciR，一个将多范式推理与可控科学渲染相结合的基准，并锚定于三个典型的科学问题。任务从形式化对象（演绎树、归纳规则假设、因果图）中生成，以保证答案可验证，随后通过针对各赛道定制微调的文体被渲染为多文档科学论述。这种构建方式使我们能够独立地调节两个难度维度：提取推理所需关键信息的难度，以及进行有原则推理本身的难度。

    arXiv:2606.13020v3 Announce Type: replace  Abstract: Three paradigmatic forms of inference recur across scientific reasoning: deduction, induction, and causal abduction. Reliably evaluating LLMs on these in scientific settings is currently out of reach: scientific benchmarks built on human annotations are costly and lack mechanistic ground truth, while synthetic logical-reasoning benchmarks do not resemble real scientific documents. We introduce SciR, a benchmark that combines multi-paradigm reasoning with controllable scientific rendering, anchored on three paradigmatic scientific problems. Tasks are generated from formal objects (deduction tree, inductive rule hypothesis, causal graph) to guarantee verifiable answers, then rendered into multi-document scientific discourse via per-track domain-tuned genres. The construction lets us independently vary two difficulty axes: how hard it is to extract the key information needed for inference, and how hard the principled inference itself is
    
[^260]: 面向掩码扩散语言模型的注意力折扣自适应采样器

    Attention-Discounted Adaptive Sampler for Masked Diffusion Language Models

    [https://arxiv.org/abs/2606.10829](https://arxiv.org/abs/2606.10829)

    提出免训练重排序规则ADAS，根据每个token对已选位置的注意力（按预测不确定性加权）贪婪地折扣其置信度分数，从而提升掩码扩散语言模型在低推理步数下的表现。

    

    掩码扩散语言模型可以通过在每次去噪迭代中揭示多个token来减少推理步骤，但这种并行性是脆弱的：当各个位置的预测相互耦合时，单独看来置信度较高的位置一起提交可能并不安全。现有的免训练采样器（如Top-k、Fast-dLLM和EB-Sampler）主要控制揭示多少个token，而往往通过忽略所选集合内部交互的逐token分数对候选进行排序。我们提出了ADAS，这是一种免训练的重排序规则，它保持基础采样器的停止规则不变，并根据每个token对已选择位置的注意力（以其预测不确定性加权）来贪婪地折扣其逐token置信度分数。在LLaDA-8B-Base和Dream-7B-Base上，于推理基准GSM8K和MATH500以及代码基准HumanEval和MBPP上的实验表明，将ADAS插入到所有三种采样器中都能改善低NFE（前向评估次数）下的性能。

    arXiv:2606.10829v3 Announce Type: replace-cross  Abstract: Masked diffusion language models can reduce inference steps by revealing multiple tokens per denoising iteration, but this parallelism is fragile: positions that are individually confident may be unsafe to commit together when their predictions are coupled. Existing training-free samplers such as Top-$k$, Fast-dLLM, and EB-Sampler mainly control how many tokens to reveal, while often ranking candidates by token-wise scores that ignore interactions within the selected set. We propose ADAS, a training-free reranking rule that leaves the base sampler's stopping rule unchanged and greedily discounts each token-wise confidence score according to its attention to already selected positions, weighted by their prediction uncertainty. Across LLaDA-8B-Base and Dream-7B-Base on the reasoning benchmarks GSM8K and MATH500 and the code benchmarks HumanEval and MBPP, plugging ADAS into all three samplers improves low-NFE performance at matche
    
[^261]: INFUSER：影响力引导的自我进化提升推理能力

    INFUSER: Influence-Guided Self-Evolution Improves Reasoning

    [https://arxiv.org/abs/2606.09052](https://arxiv.org/abs/2606.09052)

    INFUSER提出了一种影响力引导的自我进化框架，通过生成器与求解器的协同训练，利用优化器感知的影响力分数来改进问题生成，从而显著提升推理能力。

    

    自我进化为增强推理能力提供了一条可扩展的路径：预训练语言模型仅需极少的外部监督即可自我提升。然而，现有方法要么依赖大量精心策划或教师生成的训练数据，要么在生成器无监督运行时，仅通过难度启发式给予奖励，这未必能改进求解器。我们引入了INFUSER，一种迭代协同训练框架，包含两个共同演化的角色：一个生成器，从自动收集的非结构化文档池中起草问题和参考标准答案；以及一个求解器，通过在这些问题上训练来改进自身。求解器使用标准正确性奖励，依据生成器提供的答案进行训练，而生成器则通过一个优化器感知的影响力分数获得奖励，该分数衡量每个提议的问题是否真正能提升求解器在目标分布上的表现。由于这种连续且嘈杂的影响力分数难以直接处理，我们采用了相应策略进行优化。

    arXiv:2606.09052v4 Announce Type: replace-cross  Abstract: Self-evolution offers a scalable path to stronger reasoning: a pretrained language model improves itself with only minimal external supervision. Yet existing methods either depend on extensively curated or teacher-generated training data, or, when the generator runs unsupervised, reward it by a difficulty heuristic that need not improve the solver. We introduce INFUSER, an iterative co-training framework with two co-evolving roles: a Generator that drafts questions and reference golden answers from a pool of unstructured, automatically collected documents, and a Solver that improves by training on them. The solver is trained with standard correctness rewards against the generator-provided answers, while the generator is rewarded by an optimizer-aware influence score that measures whether each proposed question would actually improve the solver on the target distribution. Because this continuous, noisy influence score is poorly 
    
[^262]: 用于隐式偏好的统计先验：在个人智能体中将技能选择解耦为本地框架

    Statistical Priors for Implicit Preferences: Decoupling Skill Selection as a Local Harness in Personal Agents

    [https://arxiv.org/abs/2606.05828](https://arxiv.org/abs/2606.05828)

    提出一种将统计偏好学习与语义意图解析严格解耦的轻量级本地框架，利用本地统计先验来调节远程LLM的技能选择决策，使个人智能体能够学习隐式用户偏好，并取得最低的累积遗憾和最高的测试准确率。

    

    随着大型语言模型（LLM）能力的不断进步，依赖基于API的远程模型和外部技能的本地部署个人智能体已成为一种新范式。随着可用技能的快速扩展，使个人智能体能够学习并适应隐式用户偏好成为一项关键挑战。然而，本地部署的限制排除了复杂的集中式选择算法，因此迫切需要一种轻量级的本地偏好框架。本文通过一种新颖的架构探索了此类框架的实现，该架构将统计偏好学习与语义意图解析严格解耦。具体而言，我们利用本地化的统计结果来影响和调节远程LLM的选择决策。大量评估表明，我们的解耦方法实现了最低的累积遗憾和最高的测试准确率，显著优于传统的基于记忆的方法。

    arXiv:2606.05828v2 Announce Type: replace  Abstract: As Large Language Model (LLM) capabilities advance, locally deployed personal agents relying on API-based remote models and external skills have emerged as a novel paradigm. With the rapid expansion of available skills, enabling personal agents to learn and adapt to implicit user preferences becomes a critical challenge. However, local deployment constraints preclude complex centralized selection algorithms, creating an urgent need for a lightweight local preference harness. This paper explores the implementation of such a harness through a novel architecture that strictly decouples statistical preference learning from semantic intent parsing. Specifically, we leverage localized statistical results to influence and modulate the selection decisions of the remote LLM. Extensive evaluations demonstrate that our decoupled approach achieves the lowest cumulative regret and highest test accuracy, significantly outperforming traditional mem
    
[^263]: 轨迹中的捉迷藏：为VLA运行时监控发现失败信号

    Hide-and-Seek in Trajectories: Discovering Failure Signals for VLA Runtime Monitoring

    [https://arxiv.org/abs/2605.30834](https://arxiv.org/abs/2605.30834)

    提出Hide-and-Seek框架，将VLA失败检测建模为粗粒度监督学习问题，仅凭轨迹级标签通过轨迹间与轨迹内对比学习即可定位失败动作并生成时间结构化的失败信号，无需步级标注或昂贵的动作重采样与外部模型。

    

    Vision-Language-Action（VLA）模型使机器人能够遵循自然语言指令并在多样化任务中进行泛化，但它们仍然容易受到执行失败的影响，从而损害真实世界部署的可靠性。因此，在执行过程中检测此类失败对具身系统的稳健部署至关重要。现有的失败检测方法要么依赖代价高昂的动作重采样或外部模型，要么将轨迹级别的标签均匀传播到每个时间步，从而掩盖了局部化的失败信号。在本文中，我们提出了 **Hide-and-Seek** 框架，将VLA失败检测表述为一个粗粒度监督学习问题。通过结合轨迹间与轨迹内的对比目标，Hide-and-Seek能够定位指示失败的动作，并仅凭轨迹级别的监督即可诱导出具有时间结构的失败信号，而无需任何步级标注。

    arXiv:2605.30834v2 Announce Type: replace-cross  Abstract: Vision-Language-Action (VLA) models enable robots to follow natural language instructions and generalize across diverse tasks, but they remain vulnerable to execution failures that compromise reliability in real-world deployment. Detecting such failures during execution is therefore critical for the robust deployment of embodied systems. Existing failure detection methods either rely on expensive action resampling or external models, while alternatives propagate trajectory-level labels uniformly across every timestep, obscuring localized failure signals. In this paper, we propose \textbf{Hide-and-Seek}, a framework that formulates VLA failure detection as a coarsely supervised learning problem. By combining inter-trajectory and intra-trajectory contrastive objectives, Hide-and-Seek localizes failure-indicative actions and induces temporally structured failure signals from trajectory-level supervision alone, without any step-lev
    
[^264]: CODESKILL：为编程智能体学习自进化技能

    CODESKILL: Learning Self-Evolving Skills for Coding Agents

    [https://arxiv.org/abs/2605.25430](https://arxiv.org/abs/2605.25430)

    CODESKILL将编程智能体的技能提取与技能库维护建模为可学习的管理策略，通过强化学习和混合奖励，从任务轨迹中提取并演化多粒度程序性技能，从而实现智能体的自我进化。

    

    编程智能体在解决软件工程任务时会产生丰富的轨迹。为了实现智能体的自我进化，这些轨迹可以被提炼为可复用的程序性技能，以紧凑的方式编码经验，从而指导未来的行为。然而，现有的技能构建与维护方法通常依赖固定的提示词和启发式更新规则，尚不清楚应如何选择、抽象和维护知识才能最好地服务于下游智能体。我们提出了CODESKILL，一个基于大语言模型的框架，它将技能提取和技能库维护重新表述为一个可学习的管理策略。CODESKILL从编程智能体轨迹中提取多粒度的程序性技能，利用新经验对技能进行演化，并维护一个紧凑的技能库以用于未来的任务求解。我们采用强化学习来训练CODESKILL，使用一种混合奖励机制，将基于评分标准的密集技能质量反馈与稀疏的可验证执行反馈相结合。

    arXiv:2605.25430v2 Announce Type: replace  Abstract: Coding agents produce rich trajectories while solving software-engineering tasks. To enable agent self-evolution, these trajectories can be distilled into reusable procedural skills that compactly encode experience to guide future behavior. However, existing skill construction and maintenance methods often rely on fixed prompts and heuristic update rules, leaving it unclear how knowledge should be selected, abstracted, and maintained to best serve downstream agents. We propose CODESKILL, an LLM-based framework that reformulates skill extraction and skill-bank maintenance as a learnable management policy. CODESKILL extracts multi-granularity procedural skills from coding-agent trajectories, evolves skills with new experience, and maintains a compact skill bank for future task solving. We train CODESKILL with reinforcement learning, using a hybrid reward that combines dense rubric-based skill-quality feedback with sparse verifiable exe
    
[^265]: 神经算子的平滑分段切割方法以处理不连续性与尖锐过渡

    Smooth Piecewise Cutting for Neural Operator to Handle Discontinuities and Sharp Transitions

    [https://arxiv.org/abs/2605.19823](https://arxiv.org/abs/2605.19823)

    提出 Cut-DeepONet 两阶段训练框架，通过将求解域切割为平滑子区域、并将不连续性表示为高维空间中的边界，使神经算子能够高效处理偏微分方程解中的不连续性与尖锐过渡。

    

    神经算子在学习偏微分方程（PDE）的解算子方面已取得出色的性能，但其固有的连续表示难以捕捉不连续性和尖锐过渡。现有方法通常在连续函数空间内近似此类特征，往往需要更大的模型容量和高分辨率数据。在本工作中，我们提出 Cut-DeepONet，这是一个两阶段训练框架，在显式建模不连续性的同时降低了学习复杂度。我们的方法通过一种提升策略重新表述该问题，将求解域划分为平滑的子区域，同时将不连续性表示为更高维空间中的边界。这种分离使算子学习任务与神经网络的归纳偏置相契合，并避免了直接近似不连续性。此外，一个额外的网络用于预测依赖于输入的不连续性位置，以……

    arXiv:2605.19823v2 Announce Type: replace-cross  Abstract: Neural operators have achieved strong performance in learning solution operators of partial differential equations (PDEs), but their inherently continuous representations struggle to capture discontinuities and sharp transitions. Existing approaches typically approximate such features within continuous function spaces, often requiring increased model capacity and high-resolution data. In this work, we propose Cut-DeepONet, a two-stage training framework that explicitly models discontinuities while reducing learning complexity. Our approach reformulates the problem via a lifting strategy, partitioning the domain into smooth subregions while representing discontinuities as boundaries in a higher-dimensional space. This separation aligns the operator learning task with the inductive bias of neural networks and avoids directly approximating discontinuities. An additional network predicts input-dependent discontinuity locations for 
    
[^266]: 通过重新平衡参考帧主导性来改善图生视频模型中的运动效果

    Rebalancing Reference Frame Dominance to Improve Motion in Image-to-Video Models

    [https://arxiv.org/abs/2605.19398](https://arxiv.org/abs/2605.19398)

    论文揭示参考帧主导性（非参考帧对参考帧键token的过度自注意力）是图生视频模型运动受抑的关键原因，并提出免训练、模型无关的DyMoS方法，通过在初始去噪步骤重新平衡注意力通路来增强视频动态，且不损失对参考图像的保真度。

    

    图生视频（I2V）模型生成的视频往往过于静态，相比文生视频模型缺乏动态表现。虽然已有方法通过削弱或修改图像条件信号来缓解这一问题，但它们通常需要额外的训练，或者会牺牲对参考图像的保真度。在本工作中，我们发现参考帧主导性（reference-frame dominance）是运动被抑制的关键机制。我们观察到，I2V 模型中的非参考帧在自注意力中对参考帧的键（key）token 分配了过多注意力，导致参考信息在时间维度上被过度传播，从而抑制了帧间的动态变化。基于这一发现，我们提出了 DyMoS（Dynamic Motion Slider，动态运动滑块），这是一种免训练且与具体模型无关的方法，可在初始去噪步骤中重新平衡从生成帧到参考帧的注意力通路。DyMoS 既不改变输入图像也不改变模型权重，并引入单个标量参数以实现对运动强度的连续……（内容截断）

    arXiv:2605.19398v4 Announce Type: replace-cross  Abstract: Image-to-video models often generate videos that remain overly static, compared to text-to-video models. While prior approaches mitigate this issue by weakening or modifying the image-conditioning signal, they often require additional training or sacrifice fidelity to the reference image. In this work, we identify reference-frame dominance as a key mechanism behind motion suppression. We observe that non-reference frames in I2V models allocate excessive self-attention to reference-frame key tokens, causing reference information to be over-propagated across time and suppressing inter-frame dynamics. Based on this finding, we propose DyMoS (Dynamic Motion Slider), a training-free and model-agnostic method that rebalances the attention pathway from generated frames to the reference frame during initial denoising steps. DyMoS leaves both the input image and model weights unchanged and introduces a single scalar parameter for contin
    
[^267]: 注意力残差中的注意力汇聚与离群值

    Attention Sinks and Outliers in Attention Residuals

    [https://arxiv.org/abs/2605.17887](https://arxiv.org/abs/2605.17887)

    该论文提出OASIS方法，通过令牌级与深度级的显式空路由和空耦合机制稳定双归一化注意力残差架构，抑制注意力汇聚与激活离群值，并从理论与实验上解释和缓解了AttnResidual的低比特量化敏感性问题。

    

    我们提出OASIS，这是一种感知离群值与注意力汇聚（sink）的方法，通过显式空路由和令牌到深度的空耦合来稳定双归一化的注意力残差架构。AttnResidual引入了一个额外的深度方向归一化通道，提升了层间路由的灵活性，但也可能放大注意力汇聚、激活离群值以及低比特量化误差。OASIS建立在令牌级和深度级上基于Softmax1的显式空路由之上，并利用令牌级的空证据来降低表现出更强空行为的深度分支的权重。在理论上，我们刻画了双归一化条件下类汇聚式注意力集中的机制，为AttnResidual中观察到的低比特敏感性提供了洞见。在实验中，我们在三个语言模型骨干以及多个语言建模、推理和长上下文基准上将OASIS与五个基线方法进行比较，并观察到（原文摘要在此处截断）

    arXiv:2605.17887v2 Announce Type: replace-cross  Abstract: We propose OASIS, an outlier- and sink-aware method that stabilizes dual-normalized attention-residual architectures through explicit null routing and token-to-depth null coupling. AttnResidual introduces an additional depth-wise normalization channel that improves inter-layer routing flexibility but can also amplify attention sinks, activation outliers, and low-bit quantization error. OASIS builds on explicit Softmax1-based null routes at both the token and depth levels and uses token-level null evidence to downweight depth branches exhibiting stronger null behavior. Theoretically, we characterize a conditional mechanism for sink-like attention concentration under dual normalization, offering insight into the low-bit sensitivity observed in AttnResidual. Experimentally, we compare OASIS against five baselines on three language-model backbones and multiple language-modeling, reasoning, and long-context benchmarks and observe co
    
[^268]: Grid-Orch：基于大语言模型的配电网仿真与分析编排器

    Grid-Orch: An LLM-Powered Orchestrator for Distribution Grid Simulation and Analytics

    [https://arxiv.org/abs/2605.12728](https://arxiv.org/abs/2605.12728)

    Grid-Orch通过MCP协议将大语言模型与OpenDSS配电网仿真相结合，提供36种领域工具，使工程师能用自然语言完成潮流计算、电压分析、QSTS仿真和自动化优化，并支持本地部署以满足电力系统安全隔离需求。

    

    摘要：预计到2030年，配电工程领域将面临高达150万名工程师的劳动力短缺，这使得对更易用的分析工具的需求变得十分迫切。本文提出了Grid-Orch，这是一个通过模型上下文协议（MCP）将大语言模型（LLM）与电力系统仿真相连接的框架，使工程师能够通过自然语言执行复杂的配电分析。以OpenDSS作为参考实现，Grid-Orch提供了涵盖十一个类别的36种领域专用工具，覆盖潮流计算、电压分析、准静态时间序列（QSTS）仿真以及自动化优化。其与供应商无关的LLM层同时支持云端托管模型（Gemini、Claude）和本地部署模型（Ollama、llama-cpp），从而能够为安全敏感的公用事业环境提供物理隔离（气隙）运行。三种优化技能——电容器选址、电压越限分析与过电压缓解，……（摘要在此处被截断）

    arXiv:2605.12728v2 Announce Type: replace-cross  Abstract: The power distribution engineering workforce faces a projected shortage of up to 1.5 million engineers by 2030, creating urgent demand for more accessible analysis tools. This paper introduces Grid-Orch, a framework that bridges Large Language Models (LLMs) and power system simulation through the Model Context Protocol (MCP), enabling engineers to perform complex distribution analyses via natural language. Using OpenDSS as the reference implementation, Grid-Orch provides 36 domain-specific tools across eleven categories, covering power flow, voltage analysis, quasi-static time series (QSTS) simulation, and automated optimization. A provider-agnostic LLM layer supports both cloud-hosted (Gemini, Claude) and locally deployed (Ollama, llama-cpp) models, enabling air-gapped operation for security-sensitive utility environments. Three optimization skills, capacitor placement, voltage violation analysis, and overvoltage mitigation, e
    
[^269]: 「优雅弃牌还是英雄跟注：在策略依赖可解性下学习预算高效的思考」

    Nice Fold or Hero Call: Learning Budget-Efficient Thinking under Policy-Dependent Solvability

    [https://arxiv.org/abs/2605.11625](https://arxiv.org/abs/2605.11625)

    该论文提出预算高效思考框架BET，将自适应推理建模为不确定性下的计算投资，使模型学会在“值得深入推理求解”与“果断放弃止损”之间做出决策，避免在模型能力之外的问题上浪费测试时计算资源。

    

    大型推理模型（LRM）通过延长推理来提升问题解决能力，但常常错误分配测试时计算资源。现有的效率方法通过压缩推理轨迹或根据感知难度调节预算来降低成本，然而它们将所得的通过率解读为难度分数，未能对“零回报区域”进行建模。其结果是，这些方法在超出模型能力的问题上过度投入，同时又压缩了那些需要深入推理的困难但可解的问题。本工作将自适应推理形式化为不确定性下的计算投资问题，其中预算应遵循推理的期望回报而非感知难度。为落实这一原则，我们提出预算高效思考（BET），这是一个将行为冷启动与投资成本感知奖励下的GRPO相结合的两阶段框架。通过将“求解或弃权”决策与从rollout导出的可解性对齐，BET学会三种行为：（1）……

    arXiv:2605.11625v2 Announce Type: replace  Abstract: Large reasoning models (LRMs) improve problem solving through extended reasoning, but often misallocate test-time compute. Existing efficiency methods reduce cost by compressing reasoning traces or conditioning budget on perceived difficulty, yet read the resulting pass rate as a difficulty score, leaving the zero-return regime unmodeled. As a result, they overspend on queries beyond the model's capability while compressing hard-but-solvable ones that need deeper reasoning. In this work, we formulate adaptive reasoning as a computational investment under uncertainty, where budget follows the expected return of reasoning rather than perceived difficulty. To instantiate this principle, we propose Budget-Efficient Thinking (BET), a two-stage framework that combines behavioral cold-start with GRPO under an investment-cost-aware reward. By aligning solve-or-fold decisions with rollout-derived solvability, BET learns three behaviors: (1) s
    
[^270]: 基于计划引导的选择性离策略参考调优

    Selective Off-Policy Reference Tuning with Plan Guidance

    [https://arxiv.org/abs/2605.11505](https://arxiv.org/abs/2605.11505)

    SORT通过从参考答案推导计划并加权提升计划条件下更可预测的token，将GRPO中全部采样失败的困难提示转化为结构感知的选择性学习信号，在八个推理基准上超越GRPO基线，对较弱模型的增益最大。

    

    使用可验证奖励的强化学习有助于提升推理能力，但GRPO风格的方法在所有采样轨迹都失败的困难提示上会陷入停滞。SORT为这些失败情况增加了一种修复性更新，且不改变轨迹的生成方式：它从参考答案中推导出一个计划，比较有无该计划条件下的token概率，并给那些在计划条件下变得更可预测的token赋予更高的权重。这将全错的提示转化为选择性、结构感知的学习信号，而非单一的模仿。在三个骨干模型和八个推理基准上，SORT的表现优于GRPO及引导基线方法，且在较弱的模型上增益最大。

    arXiv:2605.11505v3 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards helps reasoning, but GRPO-style methods stall on hard prompts where all sampled rollouts fail. SORT adds a repair update for those failures without changing rollout generation: it derives a plan from the reference solution, compares token probabilities with and without that plan, and gives higher weight to tokens that become more predictable under plan conditioning. This turns all-wrong prompts into selective, structure-aware learning signals instead of uniform imitation. Across three backbones and eight reasoning benchmarks, SORT improves over GRPO and guidance baselines, with largest gains on weaker models.
    
[^271]: AcuityBench：评估临床急迫度识别与不确定性对齐

    AcuityBench: Evaluating Clinical Acuity Identification and Uncertainty Alignment

    [https://arxiv.org/abs/2605.11398](https://arxiv.org/abs/2605.11398)

    该论文提出AcuityBench基准，通过统一五个公开数据集和四级急迫度框架，系统评估语言模型从用户医疗描述中识别就医紧急程度的能力，并纳入医生确认的模糊案例以衡量模型的不确定性对齐水平。

    

    我们提出了AcuityBench，这是一个用于评估语言模型能否从用户医疗描述中识别适当就医紧急程度的基准。现有的健康类基准侧重于医学问答、广泛的健康交互或狭窄的特定工作流分诊任务，但未能提供跨这些场景的急迫度识别的统一评估。AcuityBench通过统一五个公开数据集来填补这一空白，这些数据集涵盖用户对话、在线论坛帖子、临床案例情景和患者门户消息，并采用从居家监测到立即急诊护理的共享四级急迫度框架。该基准包含914个案例，其中包括697个用于标准准确性评估的共识案例，以及217个经医生确认的模糊案例，用于不确定性感知评估。该基准支持两种互补的任务形式：问答设置中的显式四分类，以及自由格式的对话响应。

    arXiv:2605.11398v2 Announce Type: replace  Abstract: We introduce AcuityBench, a benchmark for evaluating whether language models identify the appropriate urgency of care from user medical presentations. Existing health benchmarks emphasize medical question answering, broad health interactions, or narrow workflow-specific triage tasks, but they do not offer a unified evaluation of acuity identification across these settings. AcuityBench addresses this gap by harmonizing five public datasets spanning user conversations, online forum posts, clinical vignettes, and patient portal messages under a shared four-level acuity framework ranging from home monitoring to immediate emergency care. The benchmark contains 914 cases, including 697 consensus cases for standard accuracy evaluation and 217 physician-confirmed ambiguous cases for uncertainty-aware evaluation. It supports two complementary task formats: explicit four-way classification in a QA setting, and free-form conversational response
    
[^272]: Transformer 可通过预条件 Richardson 迭代实现上下文内高斯核回归

    Transformers Can Implement Preconditioned Richardson Iteration for In-Context Gaussian Kernel Regression

    [https://arxiv.org/abs/2605.08475](https://arxiv.org/abs/2605.08475)

    本文从理论和实验上证明，标准 softmax 注意力 Transformer 的前向传播可通过实现预条件 Richardson 迭代来近似高斯核岭回归预测器，其中注意力负责跨词元的核算子运算、MLP 负责词元内的标量算术，并以 O(log(1/ε)) 的深度达到 ε 精度。

    

    本文研究了基于高斯核的上下文内核岭回归（KRR），并从理论和实证两方面证明，标准的 softmax 注意力 Transformer 能够在其前向传播过程中近似 KRR 预测器。在有界数据假设下，我们构建了一个单头 Transformer，其前向传播可近似地在相应的核系统上实现预条件 Richardson 迭代。该构造仅需 O(log(1/ε)) 个模块和宽度为 O(√(N/ε)) 的 MLP，即可对长度为 N 的提示实现 ε 精度的预测。我们的构造揭示了 Transformer 架构内部的功能分解：softmax 注意力负责生成跨词元交互所需的行归一化高斯核算子，而 MLP 层则在局部近似更新所需的词元内标量运算。在实证方面，我们训练了 GPT-2 风格的 Transformer……（原文摘要在此处截断）

    arXiv:2605.08475v3 Announce Type: replace-cross  Abstract: In this paper, we study in-context kernel ridge regression (KRR) with Gaussian kernels and show, both theoretically and empirically, that a standard softmax-attention transformer can approximate the KRR predictor during its forward pass. Under bounded-data assumptions, we construct a single-head transformer whose forward pass approximately implements \textit{preconditioned Richardson iteration} on the associated kernel system. The construction uses $O(\log(1/\epsilon))$ blocks and MLP width $O(\sqrt{N/\epsilon})$ to achieve $\epsilon$-accurate prediction for prompts of length $N$. Our construction reveals a functional decomposition within the transformer architecture: softmax attention produces a row-normalized Gaussian-kernel operator needed for \emph{cross-token} interactions, while MLP layers act locally to approximate the \emph{intra-token} scalar arithmetic required by the update. Empirically, we train GPT-2-style transfor
    
[^273]: Agentick：面向通用序贯决策智能体的统一基准

    Agentick: A Unified Benchmark for General Sequential Decision-Making Agents

    [https://arxiv.org/abs/2605.06869](https://arxiv.org/abs/2605.06869)

    Agentick是一个统一的序贯决策智能体基准，提供37个程序化生成任务，支持RL、LLM、VLM、混合及人类智能体在同一平台上的公平比较，大规模评估表明没有任何单一方法能全面占优。

    

    AI智能体研究涵盖广泛的领域：从从零开始学习的强化学习（RL）智能体，到利用预训练知识的基础模型智能体，然而目前尚无统一基准能够在这些方法之间实现公平比较。我们提出了Agentick，这是一个面向序贯决策智能体的基准，旨在为RL、LLM、VLM、混合智能体以及人类智能体提供共同的评估平台，并推动序贯决策基本挑战的研究。Agentick提供了37个程序化生成的任务，涵盖六个能力类别、四个难度等级和五种观察模态，所有任务均通过统一的Gymnasium兼容接口开放。该基准还附带编码API、所有任务的oracle参考策略、预构建的SFT数据集、可组合的智能体框架以及实时排行榜。一项涵盖27种配置、超过90,000个回合的大规模评估显示，没有任何单一方法能够全面占优：GPT-5 mini以0.3（摘要在此处截断）

    arXiv:2605.06869v3 Announce Type: replace  Abstract: AI agent research spans a wide spectrum: from RL agents that learn from scratch to foundation model agents that leverage pre-trained knowledge, yet no unified benchmark enables fair comparison across these approaches. We present Agentick, a benchmark for sequential decision-making agents designed to evaluate RL, LLM, VLM, hybrid, and human agents on common ground and to power research on the fundamental challenges of sequential decision-making. Agentick provides 37 procedurally generated tasks across six capability categories, four difficulty levels, and five observation modalities, all exposed through a single Gymnasium-compatible interface. The benchmark ships with a Coding API, oracle reference policies for all tasks, pre-built SFT datasets, a composable agent harness, and a live leaderboard. An evaluation spanning 27 configurations and over 90,000 episodes reveals that no single approach dominates: GPT-5 mini leads overall at 0.3
    
[^274]: StraTA：通过策略性轨迹抽象激励智能体强化学习

    StraTA: Incentivizing Agentic Reinforcement Learning with Strategic Trajectory Abstraction

    [https://arxiv.org/abs/2605.06642](https://arxiv.org/abs/2605.06642)

    StraTA 提出了一种策略性轨迹抽象框架，通过在智能体强化学习中引入显式的轨迹级策略并联合训练策略生成与动作执行，显著提升了大语言模型智能体在长时程决策任务中的样本效率和最终性能。

    

    大语言模型（LLM）越来越多地被用作交互式智能体，但针对长时程决策对它们进行优化仍然十分困难，因为现有方法在很大程度上是纯反应式的，这既削弱了探索能力，也削弱了对长轨迹的信用分配。在这项工作中，我们提出了策略性轨迹抽象，这是一个简单的框架，将显式的轨迹级策略引入智能体强化学习（RL）中。StraTA 从初始任务状态采样出一个紧凑的策略，使后续动作以该策略为条件，并通过分层 GRPO 式的 rollout 设计联合训练策略生成与动作执行，同时借助多样化策略 rollout 和批判性自我判断进一步增强。在 ALFWorld、WebShop 和 SciWorld 上的实验表明，StraTA 在样本效率和最终性能上均持续优于强基线方法。StraTA 的成功率达到 93%……

    arXiv:2605.06642v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as interactive agents, but optimizing them for long-horizon decision making remains difficult because current methods are largely purely reactive, which weakens both exploration and credit assignment over extended trajectories. In this work, we present Strategic Trajectory Abstraction (StraTA), a simple framework that introduces an explicit trajectory-level strategy into agentic reinforcement learning (RL). StraTA samples a compact strategy from the initial task state, conditions subsequent actions on that strategy, and trains strategy generation and action execution jointly with a hierarchical GRPO-style rollout design, further enhanced by diverse strategy rollout and critical self-judgment. Experiments on ALFWorld, WebShop, and SciWorld show that StraTA consistently improves both sample efficiency and final performance over strong baselines. StraTA reaches success rates of 93
    
[^275]: 像专家一样检测时间序列异常：一种具有专业分析器的多智能体LLM框架

    Detecting Time Series Anomalies Like an Expert: A Multi-Agent LLM Framework with Specialized Analyzers

    [https://arxiv.org/abs/2605.05725](https://arxiv.org/abs/2605.05725)

    SAGE是一个多智能体LLM框架，通过四个专门分析器对单变量时间序列异常进行基于证据的专家级诊断，并生成面向分析师的报告，在多个基准数据集上取得了最高的Point-F1分数。

    

    时间序列异常检测通常只返回分数或区间，而分析师需要理解异常行为及其支持的证据。我们提出了SAGE（专家式检测专业分析器组），这是一个用于单变量时间序列基于证据诊断的多智能体框架。四个专业分析器利用数值工具和诊断可视化，分别检查点异常、结构异常、季节性异常和模式异常。检测器将这些证据整合为区间、候选类型和证据强度置信度分数；监督器则将这些记录转化为面向分析师的报告。合成上下文参考由正常参考训练片段构建，从而减少了对真实异常示例的依赖。在Yahoo S5、KPI和WSD数据集上，SAGE实现了平均66.26的Point-F1分数，是所评估方法中最高的。受控合成评估检验了定位和类型（能力）……

    arXiv:2605.05725v2 Announce Type: replace  Abstract: Time-series anomaly detection often returns scores or intervals, while analysts need to understand the abnormal behavior and the evidence supporting it. We introduce SAGE (Specialized Analyzer Group for Expert-like Detection), a multi-agent framework for evidence-grounded diagnosis of univariate time series. Four specialized Analyzers examine point, structural, seasonal, and pattern anomalies using numerical tools and diagnostic visualizations. A Detector integrates their evidence into intervals, candidate types, and evidence-strength confidence scores; a Supervisor translates these records into analyst-facing reports. Synthetic in-context references are constructed from normal-reference training segments, reducing dependence on real anomalous demonstrations. Across Yahoo S5, KPI, and WSD, SAGE achieves an average Point-F1 of 66.26, the highest among the evaluated methods. Controlled synthetic evaluation examines localization and typ
    
[^276]: 面向软体机器人的拓扑驱动防缠绕控制

    Topology-Driven Anti-Entanglement Control for Soft Robots

    [https://arxiv.org/abs/2605.05236](https://arxiv.org/abs/2605.05236)

    本文提出一种拓扑驱动的多智能体强化学习（TD-MARL）框架，通过共享拓扑状态的集中式学习协调多软体机器人系统，解决了高约束环境下防缠绕控制中可观测性不足与训练不稳定的问题。

    

    在复杂受限环境下的精密制造领域，软体机器人的作用日益突出，基于多智能体强化学习实现防缠绕控制已成为研究热点。当前的核心问题之一是在高度受限环境中协调多个机器人完成解缠绕操作。现有的分布式训练框架在高密度障碍和不稳定环境中面临可观测性方面的挑战，导致学习效果不佳。本文提出了一种拓扑驱动的多智能体强化学习（TD-MARL）框架，用于协调多机器人系统以避免缠绕。具体而言，关键网络采用集中式学习方式，使每个智能体能够通过共享拓扑状态感知其他智能体的策略，从而缓解训练不稳定问题（摘要原文在此处截断）。

    arXiv:2605.05236v2 Announce Type: replace-cross  Abstract: In the field of precision manufacturing in complex constrained environments, the role of soft robots is increasingly prominent, and the realization of anti-winding control based on multi-intelligent body reinforcement learning has become a research hotspot. One of the core problems at present is to coordinate multiple robots to complete the unwinding operation in a highly constrained environment. The existing distributed training framework faces some observability challenges in high-density barrier and unstable environments, resulting in poor learning results. This paper proposes a topology-driven Multi-Agent Reinforcement Learning (TD-MARL) framework to coordinate multi-robot systems to avoid entanglement. Specifically, the critical network adopts centralized learning, so that each intelligent body can perceive the strategies of other intelligent bodies by sharing the topological state, thus alleviating the training instabilit
    
[^277]: 面向沉浸式视频角色扮演的奖励分解强化学习

    Reward-Decomposed Reinforcement Learning for Immersive Video Role-Playing

    [https://arxiv.org/abs/2605.04733](https://arxiv.org/abs/2605.04733)

    提出EBM-RL框架，通过将“观察-推理-生成”解耦为“眼-脑-嘴”三阶段，并结合场景-文本对齐、感知认知效用、回答忠实性和格式一致性等多维分解奖励，实现了显著超越现有基线的沉浸式视频角色扮演对话。

    

    基于文本的角色扮演模型能够模仿角色风格，但往往无法捕捉场景氛围和不断演变的紧张感，而这些对于VR游戏和互动叙事等沉浸式应用至关重要。我们研究了基于视频的角色扮演对话，并提出了EBM-RL（眼-脑-嘴强化学习），这是一个基于GRPO的解耦框架，将观察、推理和话语生成分离开来。该设计模仿了人类“看-想-说”的过程，使模型在推理和生成回复之前先将对话扎根于视觉感知。为了优化这一“看-想-说”过程，EBM-RL整合了针对场景-文本对齐、感知-认知效用、回答忠实性和格式一致性的互补奖励。大量实验表明，在我们的沉浸式角色扮演基准上，EBM-RL显著优于纯文本角色扮演基线以及更大规模的视觉语言模型。

    arXiv:2605.04733v3 Announce Type: replace  Abstract: Text-based role-playing models can imitate character styles, but often fail to capture scene atmosphere and evolving tension, which are crucial for immersive applications such as VR games and interactive narratives. We study video-grounded role-playing dialogue and introduce EBM-RL (Eye--Brain--Mouth Reinforcement Learning), a decoupled GRPO-based framework that separates observation (), reasoning (), and utterance generation (). This design mimics the human See-Think-Speak process, enabling the model to ground dialogue in visual perception before reasoning and response generation. To optimize this See-Think-Speak process, EBM-RL integrates complementary rewards for scene--text alignment, perceptual--cognitive utility, answer faithfulness, and format consistency. Extensive experiments show that EBM-RL substantially outperforms text-only role-playing baselines and larger-scale vision-language models on our immersive role-playing bench
    
[^278]: Transformers中隐式演绎推理的缩放特性

    The Scaling Properties of Implicit Deductive Reasoning in Transformers

    [https://arxiv.org/abs/2605.04330](https://arxiv.org/abs/2605.04330)

    通过反事实数据增强与跨模式共享推理原语的学习，配备双向前缀掩码的足够深的Transformer其隐式推理性能可接近显式思维链，但深度外推仍需依赖思维链。

    

    我们研究了在深度受限的Transformer中对Horn子句进行隐式演绎推理的缩放特性。通过反事实数据增强来抑制对统计捷径的依赖，并促进在直接推理模式与思维链模式之间学习共享的推理原语，我们发现：在配备双向前缀掩码的足够深的模型中，隐式推理在不同图拓扑结构和问题宽度下都能接近显式思维链的性能，不过对于深度外推，思维链仍然是必需的。这些发现是在Transformer中实现更好组合推理能力的一个进步。复现本工作的代码和模型可在以下网址获取：https://github.com/envomp/Implicit-Deductive-Reasoning-in-Transformers

    arXiv:2605.04330v3 Announce Type: replace  Abstract: We investigate the scaling properties of implicit deductive reasoning over Horn clauses in depth-bounded Transformers. By discouraging the reliance on statistical shortcuts via counterfactual data augmentation, and promoting the learning of shared reasoning primitives across direct and CoT modes, we find that in sufficiently deep models with a bidirectional prefix mask, implicit reasoning approaches explicit CoT performance across graph topologies and problem widths, though CoT remains necessary for depth extrapolation. These findings represent a step toward achieving better compositional reasoning in Transformers. The code and models to reproduce this work are available at: https://github.com/envomp/Implicit-Deductive-Reasoning-in-Transformers
    
[^279]: ReasonAudio：评估文本-音频检索中超越匹配的推理能力的基准

    ReasonAudio: A Benchmark for Evaluating Reasoning Beyond Matching in Text-Audio Retrieval

    [https://arxiv.org/abs/2605.03361](https://arxiv.org/abs/2605.03361)

    该论文提出了ReasonAudio基准，用于评估文本-音频检索中的逻辑推理能力（包括否定、时序、声音共现和时长），实验表明现有最先进的检索模型推理能力与人类存在巨大差距。

    

    现有的音频检索基准主要评估语义匹配，缺乏对复杂查询所需的逻辑推理能力的全面评估。我们提出了ReasonAudio，这是一个面向推理密集型文本-音频检索的基准，评估四种能力：否定、时序顺序、声音共现和声音时长。该基准包含五个合成子任务，涵盖针对10,000个复合音频片段的1,000个查询，以及一个自然子任务，涵盖针对1,000个真实世界片段的100个查询。对11个最先进的检索系统的评估揭示了显著的局限性：表现最佳的模型OmniEmbed-7B仅获得20.7的总分。在一项旨在减少声音事件匹配影响的对照实验中，OmniEmbed-7B的平均准确率为53.8%，而其生成式骨干模型Qwen2.5-Omni-7B-Thinker为70.6%，人类则达到95.6%。我们的结果突出了文本-音频检索中推理能力面临的挑战。

    arXiv:2605.03361v3 Announce Type: replace  Abstract: Existing audio retrieval benchmarks primarily assess semantic matching, lacking comprehensive evaluation of the logical reasoning capabilities required by complex queries. We introduce ReasonAudio, a benchmark for reasoning-intensive Text-Audio Retrieval that evaluates four abilities: negation, temporal order, sound co-occurrence, and sound duration. It comprises five synthetic subtasks with 1,000 queries over 10,000 composite audio clips and a natural subtask with 100 queries over 1,000 real-world clips. Evaluation of 11 state-of-the-art retrieval systems reveals substantial limitations: the best-performing model, OmniEmbed-7B, achieves an overall score of 20.7. In a controlled experiment designed to reduce the influence of sound-event matching, OmniEmbed-7B attains 53.8% average accuracy, compared with 70.6% for its generative backbone, Qwen2.5-Omni-7B-Thinker, and 95.6% for humans. Our results highlight the challenges of reasoning
    
[^280]: Human-1 by Josh Talks：基于真实世界对话的印地语全双工对话建模框架

    Human-1 by Josh Talks: A Full-Duplex Conversational Modeling Framework in Hindi using Real-World Conversations

    [https://arxiv.org/abs/2604.23295](https://arxiv.org/abs/2604.23295)

    该论文通过适配 Moshi 双工语音架构，利用 26,000 小时真实印地语自发对话数据，构建了首个开放、可复现的印地语全双工口语对话系统，实现对打断、重叠等自然对话行为的建模。

    

    全双工口语对话系统能够对打断、话语重叠和回应等自然对话行为进行建模，但这类系统在印度语言领域仍然鲜有探索。我们通过适配最先进的双工语音架构 Moshi，使用定制开发的印地语分词器，并在从 14,695 名说话者收集的 26,000 小时具有独立说话者通道的真实自发性对话数据上进行训练，提出了首个开放且可复现的印地语全双工口语对话系统，从而能够从自然交互中直接学习话轮转换和话语重叠模式。为支持印地语文本生成，我们替换了原始的英语分词器，重新初始化了依赖文本词表的参数，同时保留了预训练的音频组件。我们提出了一个两阶段训练方案——先进行大规模预训练，随后在 1,000 小时的对话数据上进行微调。通过提示式对话进行的评估……

    arXiv:2604.23295v3 Announce Type: replace-cross  Abstract: Full-duplex spoken dialogue systems can model natural conversational behaviours such as interruptions, overlaps, and backchannels, yet such systems remain largely unexplored for Indian languages. We present the first open, reproducible full-duplex spoken dialogue system for Hindi by adapting Moshi, a state-of-the-art duplex speech architecture, using a custom Hindi tokeniser and training on 26,000 hours of real spontaneous conversations collected from 14,695 speakers with separate speaker channels, enabling direct learning of turn-taking and overlap patterns from natural interactions. To support Hindi text generation, we replace the original English tokeniser and reinitialise text-vocabulary-dependent parameters while retaining the pre-trained audio components. We propose a two-stage training recipe -- large-scale pre-training followed by fine-tuning on 1,000 hours of conversational data. Evaluation through the prompted dialogu
    
[^281]: VLAA-GUI：知道何时停止、恢复与搜索——一个模块化的GUI自动化框架

    VLAA-GUI: Knowing When to Stop, Recover, and Search, A Modular Framework for GUI Automation

    [https://arxiv.org/abs/2604.21375](https://arxiv.org/abs/2604.21375)

    VLAA-GUI提出一个模块化GUI自动化框架，通过强制性完整性验证器杜绝无视觉证据的过早成功宣告、多层级循环断路器打破重复失败循环、以及按需在线搜索应对不熟悉元素，系统性地解决了GUI智能体的过早停止与重复循环两大核心难题。

    

    自主GUI智能体面临两个根本性挑战：一是过早停止，即智能体在没有可验证证据的情况下过早宣布任务成功；二是重复循环，即智能体在没有恢复机制的情况下反复执行相同的失败动作。我们提出了VLAA-GUI，这是一个模块化的GUI智能体框架，围绕三个集成组件构建，用以指导系统何时停止、恢复和搜索。第一，一个强制性的完整性验证器在每个完成步骤强制执行UI可观察的成功标准和验证——借助一个智能体级验证器，利用决策规则对完成声明进行交叉审查，拒绝缺乏直接视觉证据的声明。第二，一个强制性的循环断路器提供多层级过滤：在重复失败后切换交互模式，在屏幕状态持续重现后强制改变策略，并将反思信号与策略转换绑定。第三，一个按需触发的搜索智能体在线搜索不熟悉的……（摘要原文在此处截断）

    arXiv:2604.21375v3 Announce Type: replace-cross  Abstract: Autonomous GUI agents face two fundamental challenges: early stopping, where agents prematurely declare success without verifiable evidence, and repetitive loops, where agents cycle through the same failing actions without recovery. We present VLAA-GUI, a modular GUI agentic framework built around three integrated components that guide the system on when to Stop, Recover, and Search. First, a mandatory Completeness Verifier enforces UI-observable success criteria and verification at every finish step -- with an agent-level verifier that cross-examines completion claims with decision rules, rejecting those lacking direct visual evidence. Second, a mandatory Loop Breaker provides multi-tier filtering: switching interaction mode after repeated failures, forcing strategy changes after persistent screen-state recurrence, and binding reflection signals to strategy shifts. Third, an on-demand Search Agent searches online for unfamilia
    
[^282]: AI智能体的信息聚合

    Information Aggregation with AI Agents

    [https://arxiv.org/abs/2604.20050](https://arxiv.org/abs/2604.20050)

    AI智能体在预测市场实验中能够有效聚合简单信息结构下的分散信息，但在需要超过两层互动推理的复杂环境中表现受限，其推理上限接近人类水平，且廉价磋商、市场时长调整和策略性提示均无法改善其表现。

    

    大语言模型（AI智能体）能否通过交易来聚合分散的私人信息，并通过观察价格波动来推断他人的知识？我们进行了一项受控实验，让AI智能体在接收私人信号后在预测市场中进行交易，实验涵盖四种复杂度递增的信息结构。我们发现，尽管在简单的信息结构下，市场中位数表现能够有效聚合信息，但在更复杂的结构下表现会恶化，这表明AI智能体在需要超过两层互动推理的环境中会遭遇困难，这一推理上限与人类受试者中记录的上限相近。与我们的理论预测一致，允许廉价磋商（cheap talk）交流、改变市场持续时间或采用策略性提示均无法提高市场准确性；初始价格总体上影响不大，但在非常困难的信息结构中却具有重要作用。

    arXiv:2604.20050v4 Announce Type: replace-cross  Abstract: Can Large Language Models (AI agents) aggregate dispersed private information through trading and reason about the knowledge of others by observing price movements? We conduct a controlled experiment where AI agents trade in a prediction market after receiving private signals, across four information structures of increasing complexity. We find that although the median market is effective at aggregating information in the easy information structures, performance deteriorates in the harder structures, suggesting that AI agents struggle in environments where more than two levels of interactive reasoning are required, a ceiling close to the one documented in human subjects. Consistent with our theoretical predictions, market accuracy does not improve from allowing cheap talk communication, changing the duration of the market, or strategic prompting; initial price has little average effect but matters in the very hard structure. We
    
[^283]: 一种保障用户数据安全的AI智能体执行环境

    An AI Agent Execution Environment to Safeguard User Data

    [https://arxiv.org/abs/2604.19657](https://arxiv.org/abs/2604.19657)

    本文提出GAAP执行环境，通过收集用户权限规范并确定性强制执行数据披露合规，在不信任智能体且不要求模型免受攻击的前提下，保证AI智能体处理私人用户数据时的机密性。

    

    AI智能体有望成为用户的通用个人助理，这需要它们能够访问用户的私人数据（例如个人和财务信息）。这对安全和隐私构成了严重风险：AI模型可能产生幻觉或犯错，攻击者也可能攻击它（例如通过提示注入）来窃取用户数据。本文提出了GAAP（Guaranteed Accounting for Agent Privacy，智能体隐私保证核算），这是一个面向AI智能体的执行环境，能够为用户的私人数据提供机密性保证。至关重要的是，GAAP确定性地提供这一保证，无需信任智能体处理用户私人数据，也无需要求任何AI模型或用户提示免受攻击。通过动态且有针对性的用户提示，GAAP从用户处收集权限规范，用以描述其私人数据可如何被共享。随后，GAAP强制智能体的数据披露行为符合这些规范。

    arXiv:2604.19657v2 Announce Type: replace-cross  Abstract: AI agents promise to serve as general-purpose personal assistants for their users, which requires them to have access to private user data (e.g., personal and financial information). This poses a serious risk to security and privacy: an AI model may hallucinate or make mistakes, and adversaries may attack it (e.g., via prompt injection) to exfiltrate user data.   This paper presents GAAP (Guaranteed Accounting for Agent Privacy), an execution environment for AI agents that guarantees confidentiality for private user data. Crucially, GAAP provides this guarantee deterministically, without trusting the agent with private user data, and without requiring any AI model or the user prompt to be free of attacks. Through dynamic and directed user prompts, GAAP collects permission specifications from users describing how their private data may be shared. GAAP then enforces that the agent's data disclosures comply with these specificatio
    
[^284]: Scepsy：使用聚合LLM流水线服务智能体工作流

    Scepsy: Serving Agentic Workflows Using Aggregate LLM Pipelines

    [https://arxiv.org/abs/2604.15186](https://arxiv.org/abs/2604.15186)

    Scepsy利用每个LLM执行时间占比在请求间相对稳定这一洞察，通过剖析不同并行度下的LLM并构建聚合LLM流水线作为轻量级吞吐量与延迟预测器，实现了对任意多LLM智能体工作流在GPU集群上的高效低延迟调度。

    

    智能体工作流通过编排多个大语言模型（LLM）和工具来执行复杂任务。要以目标吞吐量和低延迟来服务这些工作流非常困难，因为它们以任意的智能体框架编写，且其执行时间不可预测：执行会以数据依赖的方式分支、扇出或递归。此外，由于工作流中的LLM数量往往超过可用的GPU数量，它们还会超额占用GPU资源。本文提出了Scepsy，一个能够将任意多LLM智能体工作流调度到GPU集群上的服务系统。Scepsy的关键洞察在于：虽然智能体工作流的端到端延迟不可预测，但每个LLM的执行时间占比在不同请求之间相对稳定。Scepsy在不同并行度下对每个LLM进行性能剖析，并将这些剖析结果与上述执行时间占比相结合，构建出聚合LLM流水线，一个用于资源分配决策的轻量级吞吐量和延迟预测器。为了在最小化延迟的同时……（摘要在此处截断）

    arXiv:2604.15186v2 Announce Type: replace-cross  Abstract: Agentic workflows carry out complex tasks by orchestrating multiple large language models (LLMs) and tools. Serving them at a target throughput with low latency is hard because they are written in arbitrary agentic frameworks and their execution times are unpredictable: execution branches, fans out, or recurs in data-dependent ways. Since their LLMs often outnumber the available GPUs, they also oversubscribe GPUs.   We describe Scepsy, a serving system that schedules arbitrary multi-LLM agentic workflows onto a GPU cluster. Scepsy exploits the insight that, while the end-to-end latency of an agentic workflow is unpredictable, each LLM's fraction of execution time is comparatively stable across requests. Scepsy profiles each LLM under different parallelism degrees and combines the profiles with these fractions into an Aggregate LLM Pipeline, a lightweight throughput and latency predictor for allocations. To minimize latency at a
    
[^285]: 路由博弈中的内生信息：记忆受限均衡、回忆布雷斯悖论与记忆设计

    Endogenous Information in Routing Games: Memory-Constrained Equilibria, Recall Braess Paradoxes, and Memory Design

    [https://arxiv.org/abs/2604.11733](https://arxiv.org/abs/2604.11733)

    该论文首次将旅行者的有限记忆与信息呈现机制内生化引入路由博弈，建立了“遗忘性Wardrop均衡”的存在性与唯一性理论，并通过严格凸势函数刻画了显著性加权均衡，为记忆与信息界面的交通系统设计提供了可操作的优化框架。

    

    我们研究这样一类路由博弈：旅行者在被记住或被系统呈现的路径上进行优化，而非在固定的外生行动集合上做选择。本文为内生回忆建立了一个易于处理的设计理论，并将其与一个显式的有限记忆微观模型相连接。在微观层面，每个旅行者携带一个有限记忆状态，接收系统呈现的备选路径，通过logit规则进行选择，并在诸如LRU（最近最少使用）等策略下更新记忆。这一模型产生了一个平稳的“遗忘性Wardrop均衡”（FWE）；在温和的正则性条件下证明了其存在性，且在简约不动点映射满足压缩条件时可证唯一性。论文的核心设计层是一个平稳的显著性模型，它将持久的记忆效应与界面效应总结为路径特定的权重。显著性加权的随机用户均衡是一个严格凸势函数的唯一最小化子，从而给出了简洁的优化与可实现性理论。

    arXiv:2604.11733v2 Announce Type: replace-cross  Abstract: We study routing games in which travelers optimize over routes that are remembered or surfaced, rather than over a fixed exogenous action set. The paper develops a tractable design theory for endogenous recall and then connects it back to an explicit finite-memory micro model. At the micro level, each traveler carries a finite memory state, receives surfaced alternatives, chooses via a logit rule, and updates memory under a policy such as LRU. This yields a stationary Forgetful Wardrop Equilibrium (FWE); existence is proved under mild regularity, and uniqueness follows in a contraction regime for the reduced fixed-point map. The paper's main design layer is a stationary salience model that summarizes persistent memory and interface effects as route-specific weights. Salience-weighted stochastic user equilibrium is the unique minimizer of a strictly convex potential, which yields a clean optimization and implementability theory.
    
[^286]: 《大语言模型在科学领域生命周期的缩短》

    The Shrinking Lifespan of LLMs in Science

    [https://arxiv.org/abs/2604.07530](https://arxiv.org/abs/2604.07530)

    该研究通过分析62个大语言模型在超过10.8万篇引用论文中的科学采用轨迹，发现模型在科学领域的影响力持续时间主要取决于发布年份而非自身特性，且模型迭代速度不断加快——每往后一个发布年份，模型的达到峰值时间缩短27%、生命周期缩短23%，表明大语言模型正以惊人的速度被更新模型淘汰。

    

    arXiv:2604.07530v3 公告类型：replace-cross 摘要：缩放定律描述了语言模型的能力如何随计算资源和数据的增长而提升，但并未说明一个模型在发布后能保持多长时间的影响力。我们引入“达到峰值时间”（time-to-peak）和“生命周期”（lifespan）作为衡量模型过时程度的指标，并用它们来刻画62个大语言模型在超过10.8万篇引用论文（2019-2025年）中的科学采用轨迹，将有源采用与背景引用区分开来，从而恢复出引用总量无法分辨的单模型轨迹。我们发现，模型的生命周期更多取决于其发布时间而非其自身特性：发布年份对达到峰值时间和生命周期的预测能力，比模型架构、开放性或规模都更强。大语言模型的科学采用遵循倒U型曲线（发布后上升、达到峰值、随后下降），但这一模式正在迅速压缩。发布年份每往后一年，都伴随着达到峰值时间缩短27%、生命周期缩短23%（p < 0.001），且该结果对最小年龄阈值具有稳健性。

    arXiv:2604.07530v3 Announce Type: replace-cross  Abstract: Scaling laws describe how language model capabilities grow with compute and data, but say nothing about how long a model matters once released. We introduce time-to-peak and lifespan as measures of model obsolescence and use them to characterize the scientific adoption trajectories of 62 LLMs across more than 108k citing papers (2019-2025), separating active adoption from background citation to recover per-model trajectories that citation counts cannot resolve. We find that a model's longevity is shaped more by when it was released than by its characteristics: release year predicts time-to-peak and lifespan more strongly than architecture, openness, or scale. LLM adoption follows an inverted-U curve (rising after release, peaking, and then declining), but this pattern is rapidly compressing. Each successive release year is associated with a 27% shorter time-to-peak and a 23% shorter lifespan ($p < 0.001$), robust to minimum-age
    
[^287]: 大语言模型中的归因偏差

    Attribution Bias in Large Language Models

    [https://arxiv.org/abs/2604.05224](https://arxiv.org/abs/2604.05224)

    该论文提出了首个在作者知名度和人口统计特征上保持平衡的引言归属基准数据集AttriBench，通过对11个大语言模型的评估揭示了归属准确率在种族、性别及交叉群体间的系统性差异，并发现了一种名为“抑制”的新型失败模式——即模型即使掌握作者信息也会完全省略归属。

    

    随着大语言模型（LLMs）越来越多地被用于支持搜索和信息检索，它们能否将内容准确地归因于原作者变得至关重要。在本工作中，我们介绍了AttriBench，这是首个在作者知名度和人口统计特征方面保持平衡的引言归属基准数据集。通过明确平衡作者知名度和人口统计特征，AttriBench使得对引言归属中的人口统计偏差进行受控研究成为可能。利用该数据集，我们在不同提示设置下评估了11个广泛使用的大语言模型，发现即使对于最前沿的模型，引言归属仍然是一项具有挑战性的任务。我们观察到不同种族、性别以及交叉群体之间的归属准确率存在巨大且系统性的差异。我们进一步引入并研究了“抑制”（suppression）现象，这是一种独特的失败模式，即模型完全省略归属信息，即使模型本身拥有作者身份信息。我们发现抑制现象普遍存在且分布不均。

    arXiv:2604.05224v2 Announce Type: replace  Abstract: As Large Language Models (LLMs) are increasingly used to support search and information retrieval, it is critical that they accurately attribute content to its original authors. In this work, we introduce AttriBench, the first fame- and demographically-balanced quote attribution benchmark dataset. By explicitly balancing author fame and demographics, AttriBench enables controlled investigation of demographic bias in quote attribution. Using this dataset, we evaluate 11 widely used LLMs across different prompt settings and find that quote attribution remains a challenging task even for frontier models. We observe large and systematic disparities in attribution accuracy between race, gender, and intersectional groups. We further introduce and investigate suppression, a distinct failure mode in which models omit attribution entirely, even when the model has access to authorship information. We find that suppression is widespread and une
    
[^288]: Combee：为自我改进的语言模型智能体扩展提示学习

    Combee: Scaling Prompt Learning for Self-Improving Language Model Agents

    [https://arxiv.org/abs/2604.04247](https://arxiv.org/abs/2604.04247)

    Combee 提出了一种新颖的框架，通过有原则的并行扩展策略解决现有提示学习方法在高并行度下质量下降的问题，从而同时提升自我改进语言模型智能体提示学习的效率和效果。

    

    提示学习（prompt learning）的最新进展使大型语言模型智能体能够在不改变参数的情况下，从推理时上下文中获取与任务相关的知识。例如，现有方法（如 ACE 或 GEPA）可以基于先前的智能体运行记录来学习系统提示，从而提升准确率。然而，这些方法主要集中于单智能体或低并行度的设置，这从根本上限制了它们从大量收集的智能体轨迹中高效学习的能力。鉴于从大量智能体轨迹或并行智能体执行中学习的趋势日益增长，以并行方式运行提示学习将是高效且有益的。然而，由于缺乏有原则的扩展策略，当前方法在高并行度下会出现质量下降的问题。为了同时提升提示学习的效率和效果，我们提出了 Combee，一个用于扩展自我改进智能体并行提示学习的新颖框架。Combee 加速了学习……

    arXiv:2604.04247v2 Announce Type: replace  Abstract: Recent advances in prompt learning allow large language model agents to acquire task-relevant knowledge from inference-time context without parameter changes. For example, existing methods (like ACE or GEPA) can learn system prompts to improve accuracy based on previous agent runs. However, these methods primarily focus on single-agent or low-parallelism settings. This fundamentally limits their ability to efficiently learn from a large set of collected agentic traces. It would be efficient and beneficial to run prompt learning in parallel to accommodate the growing trend of learning from many agentic traces or parallel agent executions. Yet without a principled strategy for scaling, current methods suffer from quality degradation with high parallelism. To improve both the efficiency and quality of prompt learning, we propose Combee, a novel framework to scale parallel prompt learning for self-improving agents. Combee speeds up learn
    
[^289]: GVCC：基于码本驱动的随机修正流的零样本视频压缩

    GVCC: Zero-Shot Video Compression via Codebook-Driven Stochastic Rectified Flow

    [https://arxiv.org/abs/2603.26571](https://arxiv.org/abs/2603.26571)

    GVCC提出了一种零样本视频压缩框架，将预训练的视频生成模型直接作为解码器，通过把确定性修正流采样器转换为保持边缘分布的随机过程并编码每步随机创新量来传输信息，从而在极低码率下实现高保真的视频重建。

    

    在极低码率下，高保真重建需要从后验分布中采样出合理的视频，而不是回归到过度平滑的条件均值。我们提出了生成式视频码本编解码器，这是一个零样本框架，其中预训练的视频生成模型直接作为解码器，而传输的比特流用于指定其生成轨迹。现代修正流视频模型通常使用确定性ODE求解器进行采样，这无法为传输压缩信息提供逐步的随机信道。GVCC通过将确定性流采样器转换为等效的、保持边缘分布的随机过程来解决这一问题，从而可以通过对每一步的随机创新量进行编码来传输信息。与图像不同，视频带来了更长的时序依赖性和更多样的条件模式。我们将GVCC实例化为三种实用模式：文本到视频（T2V）——无需……

    arXiv:2603.26571v4 Announce Type: replace-cross  Abstract: At ultra-low bitrates, high-fidelity reconstruction requires sampling plausible videos from the posterior rather than regressing to oversmoothed conditional means. We propose Generative Video Codebook Codec (GVCC), a zero-shot framework in which a pretrained video generative model serves directly as the decoder, and the transmitted bitstream specifies its generation trajectory. Modern rectified-flow video models are typically sampled with deterministic ODE solvers, which leave no per-step stochastic channel for transmitting compressed information. GVCC addresses this by converting the deterministic flow sampler into an equivalent marginal-preserving stochastic process, so that information can be transmitted by encoding the per-step stochastic innovations. Unlike images, videos introduce longer temporal dependencies and more diverse conditioning modes. We instantiate GVCC in three practical modes: Text-to-Video (T2V) without a r
    
[^290]: 谱球约束超连接

    Spectral-Sphere-Constrained Hyper-Connections

    [https://arxiv.org/abs/2603.20896](https://arxiv.org/abs/2603.20896)

    针对双随机约束超连接存在的恒等退化、表达能力瓶颈和参数化开销三重局限，提出将残差矩阵约束在谱球流形上的谱球约束超连接，在保持恒等映射性质以稳定训练的同时，恢复跨流混合的谱自由度和表达能力。

    

    摘要：超连接将残差连接扩展为多条流，并利用残差矩阵进行跨流混合，以丰富模型的表达能力。然而，不加约束的混合会破坏残差连接固有的恒等映射性质，导致训练不稳定。为解决这一问题，流形约束超连接及其变体通过 Sinkhorn-Knopp（SK）算法或基于置换的参数化，将这些矩阵限制为双随机矩阵。我们揭示了这种双随机约束的三个局限：(1) 恒等退化，即学习到的矩阵坍缩在恒等初始化附近，削弱了跨流交互；(2) 表达能力瓶颈，双随机约束限制了残差矩阵次主导谱的自由度，使模型无法有选择地保留或衰减跨流变化；(3) 参数化……（原摘要在此处截断）

    arXiv:2603.20896v2 Announce Type: replace-cross  Abstract: Hyper-Connections (HC) extend residual connections into multiple streams, employing residual matrices for cross-stream mixing to enrich model expressivity. However, unconstrained mixing disrupts the identity mapping property intrinsic to the residual connection, causing unstable training. To address this, Manifold-Constrained Hyper-Connections (mHC) and its variants restrict these matrices to be doubly stochastic via Sinkhorn-Knopp (SK) algorithm or permutation-based parameterizations. We reveal three limitations of this doubly stochastic constraint: (1) identity degeneration, where learned matrices collapse around the identity initialization and diminish cross-stream interactions, (2) a expressivity bottleneck, where the doubly stochastic constraint restricts the freedom of the subdominant spectrum of the residual matrices, preventing the model from selectively preserving or attenuating cross-stream variations, and (3) paramet
    
[^291]: 解读机器学习决策：面向大规模排序系统的智能体推理框架

    Decoding ML Decision: An Agentic Reasoning Framework for Large-Scale Ranking System

    [https://arxiv.org/abs/2602.18640](https://arxiv.org/abs/2602.18640)

    本文提出GEARS框架，将大规模排序优化重构为可编程实验环境中的自主发现过程，通过专门的智能体技能封装排序专家知识，让操作者只需通过高层产品意图即可引导系统，从而突破将模糊产品意图转化为可验证假设的工程瓶颈。

    

    现代大规模排序系统在竞争目标、运营约束和不断演进的产品需求构成的复杂环境中运行。该领域的进展日益受到工程上下文约束的瓶颈制约，即如何将模糊的产品意图转化为合理、可执行、可验证的假设这一艰巨过程，而非仅仅受限于建模技术本身。我们提出了GEARS（面向智能体排序系统的生成式引擎），这是一个将排序优化重新定义为可编程实验环境中自主发现过程的框架。GEARS不再将优化视为静态的模型选择，而是利用专门的智能体技能将排序专家知识封装为可复用的推理能力，使操作者能够通过高层意图的vibe个性化来引导系统。此外，为确保生产可靠性，该框架引入了验证机制……

    arXiv:2602.18640v3 Announce Type: replace  Abstract: Modern large-scale ranking systems operate within a sophisticated landscape of competing objectives, operational constraints, and evolving product requirements. Progress in this domain is increasingly bottlenecked by the engineering context constraint: the arduous process of translating ambiguous product intent into reasonable, executable, verifiable hypotheses, rather than by modeling techniques alone. We present GEARS (Generative Engine for Agentic Ranking Systems), a framework that reframes ranking optimization as an autonomous discovery process within a programmable experimentation environment. Rather than treating optimization as static model selection, GEARS leverages Specialized Agent Skills to encapsulate ranking expert knowledge into reusable reasoning capabilities, enabling operators to steer systems via high-level intent vibe personalization. Furthermore, to ensure production reliability, the framework incorporates validat
    
[^292]: VLANeXt：构建强大VLA模型的配方

    VLANeXt: Recipes for Building Strong VLA Models

    [https://arxiv.org/abs/2602.18532](https://arxiv.org/abs/2602.18532)

    本文通过统一框架系统剖析VLA设计空间，提炼出12个关键发现，形成了构建强大VLA模型的实用配方。

    

    摘要：随着大型基础模型的兴起，视觉-语言-动作模型（VLAs）应运而生，利用视觉-语言模型（VLMs）强大的视觉和语言理解能力进行通用策略学习。然而，当前VLA领域仍然分散且探索性较强。尽管许多团队提出了各自的VLA模型，但训练协议和评估设置的不一致性使得难以确定哪些设计选择真正重要。为了给这一不断发展的领域带来结构，我们在统一框架和评估设置下重新审视了VLA设计空间。从一个类似于RT-2（VLA的起源）的简单VLA基线出发，我们系统地剖析了三个维度的设计选择：基础组件、感知要素和动作建模视角。通过这项研究，我们提炼出12个关键发现，共同构成了构建强大VLA模型的实用配方。最终结果如下。

    arXiv:2602.18532v3 Announce Type: replace-cross  Abstract: Following the rise of large foundation models, Vision-Language-Action models (VLAs) emerged, leveraging strong visual and language understanding from Vision-Language Models for general-purpose policy learning. Yet, the current VLA landscape remains fragmented and exploratory. Although many groups have proposed their own VLA models, inconsistencies in training protocols and evaluation settings make it difficult to identify which design choices truly matter. To bring structure to this evolving space, we reexamine the VLA design space under a unified framework and evaluation setup. Starting from a simple VLA baseline similar to RT-2, which is the origin of VLA, we systematically dissect design choices along three dimensions: foundational components, perception essentials, and action modelling perspectives. From this study, we distill 12 key findings that together form a practical recipe for building strong VLA models. The outcome 
    
[^293]: 何时该快思考与慢思考？AMOR：面向混合模型的自适应熵门控

    When to Think Fast and Slow? AMOR: Adaptive Entropy Gate for Hybrid Models

    [https://arxiv.org/abs/2602.13215](https://arxiv.org/abs/2602.13215)

    AMOR通过基于输出熵的动态门控自适应地选择调用注意力，仅在约40%的位置激活注意力且无需可学习路由参数，就能在每个规模上取得最佳的八项常识推理平均成绩。

    

    循环-注意力混合架构旨在结合循环网络的效率与注意力的上下文记忆能力，但现有方法通常在所有位置上统一应用注意力，即使循环状态本身已足以做出准确预测。我们提出了AMOR（自适应元认知输出路由器），一种事后（post-hoc）混合架构，可基于预测不确定性选择性地调用注意力。该方法在循环主干网络中加入了熵门控注意力块，仅当模型输出熵超过一个动态阈值时才被激活，该阈值由运行批次的中位数和缩放后的标准差推导得出。由此产生的二值门控无需任何可学习的路由参数。模型在FineWeb-Edu数据集上从头预训练，注意力仅在约40%的位置被调用，其中一个AMOR变体（采用Mamba2或Gated DeltaNet主干）在每个规模上均取得了八项常识推理任务平均分最高的成绩

    arXiv:2602.13215v3 Announce Type: replace  Abstract: Recurrent-attention hybrids aim to combine the efficiency of recurrence with the contextual recall of attention, but existing approaches typically apply attention uniformly across all positions, even when the recurrent state alone is sufficient for accurate prediction. We introduce AMOR (Adaptive Metacognitive Output Router), a post-hoc hybrid architecture that selectively invokes attention based on predictive uncertainty. A recurrent backbone is augmented with entropy-gated attention blocks that activate only when the model's output entropy exceeds a dynamic threshold derived from a running batch median and scaled standard deviation. The resulting binary gate requires no learned routing parameters. Pretrained from scratch on FineWeb-Edu and with attention invoked on only ~40% of positions, one of the AMOR variants (Mamba2 or Gated DeltaNet backbones) achieves the highest eight-task common-sense reasoning average at each scale among 
    
[^294]: GT-HarmBench：通过博弈论视角评估人工智能安全风险

    GT-HarmBench: Benchmarking AI Safety Risks Through the Lens of Game Theory

    [https://arxiv.org/abs/2602.12316](https://arxiv.org/abs/2602.12316)

    该论文提出GT-HarmBench——首个基于博弈论结构、包含1535个高风险场景的多智能体AI安全基准，发现前沿模型在38%的高风险场景中无法选择对社会有益的行动，而博弈论干预可将有益结果提升最高18%。

    

    前沿人工智能系统的能力日益增强，并被部署于高风险的多智能体环境中。然而，现有的人工智能安全基准主要评估单个智能体，导致对协调失败和冲突等多智能体风险的理解严重不足。我们提出了GT-HarmBench，这是一个包含1,535个高风险场景的基准，涵盖囚徒困境、猎鹿博弈和胆小鬼博弈等博弈论结构。这些场景取材于MIT人工智能风险知识库中的现实AI风险情境。在对15个前沿模型的评估中，智能体在38%的高风险案例（如军事升级、选举操纵和医疗事故）中未能选择对社会有益的行动。我们测量了模型对博弈论提示框架和顺序的敏感性，并分析了导致失败的推理模式。我们进一步表明，博弈论干预可以将社会有益结果提升至多18%。我们的结果凸显了前沿AI系统在多智能体环境中显著的可靠性问题。

    arXiv:2602.12316v3 Announce Type: replace  Abstract: Frontier AI systems are increasingly capable and deployed in high-stakes multi-agent environments. However, existing AI safety benchmarks largely evaluate single agents, leaving multi-agent risks such as coordination failure and conflict poorly understood. We introduce GT-HarmBench, a benchmark of 1,535 high-stakes scenarios spanning game-theoretic structures such as the Prisoner's Dilemma, Stag Hunt and Chicken. Scenarios are drawn from realistic AI risk contexts in the MIT AI Risk Repository. Across 15 frontier models, agents fail to choose socially beneficial actions in 38% of high-stakes cases, such as military escalation, election manipulation, and medical malpractice. We measure sensitivity to game-theoretic prompt framing and ordering, and analyze reasoning patterns driving failures. We further show that game-theoretic interventions improve socially beneficial outcomes by up to 18%. Our results highlight substantial reliabilit
    
[^295]: 面向可泛化长期物理模拟的潜在生成求解器

    Latent Generative Solvers for Generalizable Long-Term Physics Simulation

    [https://arxiv.org/abs/2602.11229](https://arxiv.org/abs/2602.11229)

    本文提出潜在生成求解器（LGS），通过物理VAE压缩十二个PDE族到共享潜在流形、金字塔流强制Transformer进行流匹配生成，以及训练时输入加噪的稳定性保证，首次实现了跨异构PDE族的泛化能力与长时域自回归物理模拟稳定性的兼顾。

    

    可靠的物理模拟需要两种当今神经偏微分方程（PDE）求解器无法同时具备的能力：跨异构PDE族的泛化能力，以及长时间自回归滚动预测下的稳定性。确定性算子会以几何级数累积误差，而现有的概率求解器则局限于单一PDE族或较短的预测时域。我们通过潜在生成求解器（Latent Generative Solver, LGS）弥合了这一差距，该求解器由三个相互耦合的组件构成：(i) 物理VAE（PhyVAE），将十二个PDE族压缩到共享的潜在流形中；(ii) 金字塔流强制Transformer（PFlowFT），通过流匹配生成下一个潜在状态，并以基于模型自身预测而更新的每条轨迹上下文作为条件；(iii) 训练期间的输入加噪，我们为此推导了一个充分条件下的收缩界，用以解释所观察到的长时域稳定性。LGS在一个包含250万条轨迹、16个系统、分辨率为128²的语料库上进行预训练，其匹配……（摘要原文在此处被截断）

    arXiv:2602.11229v3 Announce Type: replace  Abstract: Reliable physics simulation demands two capabilities that today's neural PDE solvers do not deliver together: generalization across heterogeneous PDE families, and stability under long autoregressive rollouts. Deterministic operators accumulate error geometrically, while existing probabilistic solvers are confined to a single PDE family or short horizons. We close this gap with the \textbf{Latent Generative Solver} (LGS), three coupled components: (i) a Physics VAE (PhyVAE) compressing twelve PDE families into a shared latent manifold; (ii) a Pyramidal Flow-Forcing Transformer (PFlowFT) that generates the next latent by flow matching, conditioned on a per-trajectory context updated on the model's own predictions; and (iii) input noising during training, for which we derive a sufficient-condition contraction bound explaining the observed long-horizon stability. Pretrained on a 2.5\,M-trajectory, 16-system corpus at $128^2$, LGS matche
    
[^296]: FlyAOC：评估果蝇科学知识库的智能体本体策展

    FlyAOC: Evaluating Agentic Ontology Curation of Drosophila Scientific Knowledge Bases

    [https://arxiv.org/abs/2602.09163](https://arxiv.org/abs/2602.09163)

    FlyAOC是一个评估AI智能体从科学文献中进行端到端本体策展的基准，要求智能体在16,898篇果蝇论文中检索证据，并恢复策展人级别的结构化基因标注，涵盖功能术语、表达模式和历史同义词。

    

    科学知识库通过将原始文献中的研究发现策展为结构化、可查询的格式，从而加速科学发现，服务于人类研究者和新兴的AI系统。维护这些资源需要专家策展人检索论文、整合跨文档证据，并生成基于本体的标注。现有基准通常只评估孤立的子任务（如命名实体识别或关系抽取），因此无法捕捉这一端到端的工作流程。我们提出FlyAOC，用于评估AI智能体在科学文献上执行端到端智能体本体策展的能力。给定一个基因符号、一段简洁的FlyBase基因描述、一个包含16,898篇论文语料库的访问权限以及本体资源，智能体必须检索证据并尽可能多地恢复与策展人相关的结构化标注。输出涵盖标准化的功能术语、表达模式，以及连接数十年命名历史的历史同义词。

    arXiv:2602.09163v2 Announce Type: replace  Abstract: Scientific knowledge bases accelerate discovery by curating findings from primary literature into structured, queryable formats for both human researchers and emerging AI systems. Maintaining these resources requires expert curators to search papers, reconcile evidence across documents, and produce ontology-grounded annotations. Existing benchmarks usually evaluate isolated subtasks, such as named entity recognition or relation extraction, and therefore do not capture this end-to-end workflow. We present FlyAOC to evaluate AI agents on end-to-end agentic ontology curation from scientific literature. Given a gene symbol, a concise FlyBase gene description, access to a 16,898-paper corpus, and ontology resources, agents must search for evidence and recover as many curator-relevant structured annotations as possible. Outputs span standardized function terms, expression patterns, and historical synonyms linking decades of nomenclature. T
    
[^297]: 超越词袋模型：诊断视觉-语言模型中的组合绑定失败

    Beyond Bag-of-Words: Diagnosing Compositional Binding Failures in Vision-Language Models

    [https://arxiv.org/abs/2602.02043](https://arxiv.org/abs/2602.02043)

    该论文提出Auto-Comp——一个全自动概念驱动的基准生成流水线，通过“平行A/B构建”（最小样本与上下文样本对照）大规模生成照片级真实感的组合推理基准，从而精准诊断视觉-语言模型在属性与物体绑定上的失败。

    

    现代视觉-语言模型在基本的组合推理方面存在困难，无法将属性正确绑定到物体，或将关系正确绑定到其指代对象。现有的基准测试要么依赖于嘈杂的真实图像，将混杂的视觉变量与推理失败混为一谈，要么使用缺乏现代VLM所适配的真实感的简单合成场景。我们引入了**Auto-Comp**，这是一个完全自动化的、概念驱动的流水线，通过大规模生成照片级真实感的组合推理基准来弥补这一差距。其核心创新是一种“平行A/B构建”方法：对于每个概念，该流水线会生成一个“最小”样本（模板描述、白色背景上的孤立物体）和一个“上下文”样本（由LLM重写的描述、嵌入真实场景中的物体），从而将核心绑定能力与视觉-语言复杂性分离开来。我们实例化了四个任务族，涵盖了组合推理的两个经典维度……（摘要截断）

    arXiv:2602.02043v2 Announce Type: replace-cross  Abstract: Modern vision-language models struggle with basic compositional reasoning, failing to bind attributes to objects or relations to their referents. Existing benchmarks either rely on noisy real images that conflate confounding visual variables with the reasoning failure, or use simplistic synthetic scenes lacking the realism modern VLMs are tuned for. We introduce \textbf{Auto-Comp}, a fully automated, concept-driven pipeline that bridges this gap by generating photorealistic compositional benchmarks at scale. Its core innovation is a \textit{parallel A/B construction}: for each concept, the pipeline emits a \textit{Minimal} sample (template caption, isolated objects on a white background) and a \textit{Contextual} sample (LLM-rewritten caption, objects embedded in a realistic scene), isolating core binding ability from visio-linguistic complexity. We instantiate \textit{four} task families spanning the two canonical axes of comp
    
[^298]: 大语言模型推理的分步内在奖励

    Stepwise Intrinsic Rewards for Reasoning in Large Language Models

    [https://arxiv.org/abs/2602.01034](https://arxiv.org/abs/2602.01034)

    提出了一种无需过程标注、辅助模型或推理时搜索的内在过程奖励方法——分步边际信息增益（MIG），通过衡量每个推理前缀对参考答案对数似然的提升，并结合单调水位线机制避免重复计分，从而为大语言模型的多步推理提供更精确的密集监督。

    

    强化学习（RL）已成为提升大语言模型（LLM）和视觉-语言模型（VLM）推理能力的广泛使用的范式。然而，稀疏的二值结果奖励仅对最终正确性进行评分，无法识别哪些中间步骤对正确性做出了贡献；在多模态任务中，这类奖励还可能奖励由语言先验而非视觉证据驱动的答案。过程奖励模型（PRM）虽然使监督更加密集，但通常需要过程标注、辅助模型或推理时搜索。在本文中，我们提出了分步边际信息增益（Stepwise Marginal Information Gain, MIG），这是一种由策略自身计算得到的内在过程奖励。MIG 衡量每个结构化推理前缀如何改变参考答案在长度归一化、教师强制条件下的对数似然。单调的历史水位线机制仅奖励新的似然最大值，从而避免在次优迂回路径之后进行重复计分。（注：原摘要在此处被截断）

    arXiv:2602.01034v2 Announce Type: replace  Abstract: Reinforcement learning (RL) has become a widely used paradigm for improving the reasoning abilities of large language models (LLMs) and Vision-language models (VLMs). Sparse binary outcome rewards, however, score only final correctness and cannot identify which intermediate steps contributed to it; in multimodal tasks, they may also reward answers driven by linguistic priors rather than visual evidence. Process reward models (PRMs) densify supervision but usually require process annotations, auxiliary models, or inference-time search. In this paper, we introduce Stepwise Marginal Information Gain (MIG), an intrinsic process reward computed from the policy itself. MIG measures how each structured reasoning prefix changes the length-normalized, teacher-forced log-likelihood of the reference answer. A monotonic historical watermark rewards only new likelihood maxima, avoiding duplicate credit after sub-record detours. We combine this si
    
[^299]: RAPTOR：岭自适应逻辑回归探针

    RAPTOR: Ridge-Adaptive Logistic Probes

    [https://arxiv.org/abs/2602.00158](https://arxiv.org/abs/2602.00158)

    提出了一种简单的L2正则化逻辑回归探针RAPTOR，通过验证集调优的岭回归强度从归一化权重中提取准确且方向稳定的概念向量，可用于大语言模型的“先探针后引导”激活引导流程。

    

    探针技术通过在冻结的大语言模型层表示之上训练轻量级预测器，来研究这些层表示中编码了哪些信息。除了分析用途之外，探针还经常被实际应用于“先探针后引导”流程中：从探针中提取学习到的概念向量，并通过加性激活引导的方式将其注入，即在前向传播过程中将其添加到某一层的表示上。该流程的有效性取决于所估计的概念向量是否准确、在消融下方向是否稳定，以及获取成本是否低廉。基于这些目标，我们提出了RAPTOR（岭自适应逻辑回归探针），这是一种简单的L2正则化逻辑回归探针，其通过验证集调优的岭回归强度从归一化权重中生成概念向量。在指令微调大语言模型和人类撰写的概念数据集上的大量实验中，RAPTOR在准确率上匹敌或超越强基线，同时实现了具有竞争力的方向稳定性。

    arXiv:2602.00158v3 Announce Type: replace-cross  Abstract: Probing studies what information is encoded in a frozen LLM's layer representations by training a lightweight predictor on top of them. Beyond analysis, probes are often used operationally in probe-then-steer pipelines: a learned concept vector is extracted from a probe and injected via additive activation steering by adding it to a layer representation during the forward pass. The effectiveness of this pipeline hinges on estimating concept vectors that are accurate, directionally stable under ablation, and inexpensive to obtain. Motivated by these desiderata, we propose RAPTOR (Ridge-Adaptive Logistic Probe), a simple L2-regularized logistic probe whose validation-tuned ridge strength yields concept vectors from normalized weights. Across extensive experiments on instruction-tuned LLMs and human-written concept datasets, RAPTOR matches or exceeds strong baselines in accuracy while achieving competitive directional stability an
    
[^300]: 基于事件的几何-光度3D高斯光线追踪

    Geometric-Photometric Event-based 3D Gaussian Ray Tracing

    [https://arxiv.org/abs/2512.18640](https://arxiv.org/abs/2512.18640)

    提出GPERT框架，通过光线追踪将渲染解耦为逐事件的几何（深度）渲染和基于快照的辐射度（强度）渲染两个分支，在基于事件的3D高斯泼溅中解决了精度与时间分辨率之间的权衡问题，且无需先验信息或COLMAP初始化即可达到最先进性能。

    

    事件相机相比传统基于帧的相机具有更高的时间分辨率，这使其非常适合用于运动和结构估计。然而，基于事件的3D高斯泼溅（3DGS）方法如何利用稀疏事件的细粒度时间信息，此前一直不明确。本工作提出了GPERT，一个用于解决基于事件的3DGS中精度与时间分辨率之间权衡问题的框架。我们的核心思想是将渲染解耦为两个分支：逐事件的几何（深度）渲染和基于快照的辐射度（强度）渲染，分别通过光线追踪和扭曲事件图像来实现。大量评估表明，我们的方法在真实世界数据集上取得了最先进的性能，在合成数据集上也具有竞争力的表现。此外，所提出的方法无需任何先验信息（如预训练的图像重建模型）或基于COLMAP的初始化。

    arXiv:2512.18640v3 Announce Type: replace-cross  Abstract: Event cameras offer a high temporal resolution over traditional frame-based cameras, which makes them suitable for motion and structure estimation. However, it has been unclear how event-based 3D Gaussian Splatting (3DGS) approaches could leverage fine-grained temporal information of sparse events. This work proposes GPERT, a framework to address the trade-off between accuracy and temporal resolution in event-based 3DGS. Our key idea is to decouple the rendering into two branches: event-by-event geometry (depth) rendering and snapshot-based radiance (intensity) rendering, by using ray-tracing and the image of warped events. The extensive evaluation shows that our method achieves state-of-the-art performance on the real-world datasets and competitive performance on the synthetic dataset. Also, the proposed method works without prior information (e.g., pretrained image reconstruction models) or COLMAP-based initialization, is mor
    
[^301]: 基于提示的持续组合零样本学习

    Prompt-Based Continual Compositional Zero-Shot Learning

    [https://arxiv.org/abs/2512.09172](https://arxiv.org/abs/2512.09172)

    该论文提出了首个基于提示的持续组合零样本学习框架PromptCCZSL，通过近期加权多教师蒸馏、会话感知的组合提示与会话无关的属性/对象提示融合，并结合余弦锚定损失，在冻结的视觉-语言模型上实现对新属性、对象及组合的持续适应，同时有效防止旧知识遗忘。

    

    我们研究在组合零样本学习（CZSL）中，视觉-语言模型对新属性、新对象及其组合的持续适应问题，同时防止对已有知识的遗忘。与类别互不重叠的经典持续学习不同，持续组合零样本学习（CCZSL）更为复杂，因为属性和对象可能在多个会话中重复出现，而组合始终保持唯一。基于冻结的视觉-语言模型（VLM）骨干网络，我们提出了首个基于提示的持续组合零样本学习框架，通过近期加权的多教师蒸馏来保留先验知识。该框架采用会话感知的组合提示来融合多模态特征以处理新组合，同时通过会话无关的融合方式学习属性提示和对象提示，以维持全局语义一致性，并进一步引入余弦锚定损失（CAL）加以稳定，从而保护先验知识。为增强当前会话的适应能力……（原文摘要到此不完整）

    arXiv:2512.09172v3 Announce Type: replace-cross  Abstract: We tackle continual adaptation of vision-language models to new attributes, objects, and their compositions in Compositional Zero-Shot Learning (CZSL), while preventing forgetting of prior knowledge. Unlike classical continual learning where classes are disjoint, CCZSL is more complex as attributes and objects may reoccur across sessions while compositions remain unique. Built on a frozen VLM backbone, we propose the first Prompt-based Continual Compositional Zero-Shot Learning (PromptCCZSL) framework that retains prior knowledge through recency-weighted multi-teacher distillation. It employs session-aware compositional prompts to fuse multimodal features for new compositions, while attribute and object prompts are learned through session-agnostic fusion to maintain global semantic consistency, which is further stabilized by a Cosine Anchor Loss (CAL) to preserve prior knowledge. To enhance adaptation in the current session, an
    
[^302]: 演示：生成式AI基于用户偏好辅助放射治疗计划设计

    Demo: Generative AI helps Radiotherapy Planning with User Preference

    [https://arxiv.org/abs/2512.08996](https://arxiv.org/abs/2512.08996)

    本文提出一种仅依据用户自定义偏好即可预测三维剂量分布的生成式模型，使计划制定者能够个性化权衡危及器官与靶区之间的取舍，并在适应性上超越Varian RapidPlan。

    

    放射治疗计划是一个高度复杂的过程，在不同机构和不同计划制定者之间往往存在显著差异。现有的大多数用于三维剂量预测的深度学习方法在训练时依赖参考计划作为真值（ground truth），这可能会无意中使模型偏向特定的计划风格或机构偏好。在本研究中，我们提出了一种新颖的生成式模型，仅根据用户自定义的偏好风格来预测三维剂量分布。这些可定制的偏好使计划制定者能够优先考虑危及器官（OARs）与计划靶区（PTVs）之间的特定权衡，提供了更大的灵活性和个性化。我们的方法旨在与临床治疗计划系统无缝集成，帮助用户高效地生成高质量计划。对比评估表明，我们的方法在适应性方面可以超越Varian RapidPlan模型。

    arXiv:2512.08996v2 Announce Type: replace-cross  Abstract: Radiotherapy planning is a highly complex process that often varies significantly across institutions and individual planners. Most existing deep learning approaches for 3D dose prediction rely on reference plans as ground truth during training, which can inadvertently bias models toward specific planning styles or institutional preferences. In this study, we introduce a novel generative model that predicts 3D dose distributions based solely on user-defined preference flavors. These customizable preferences enable planners to prioritize specific trade-offs between organs-at-risk (OARs) and planning target volumes (PTVs), offering greater flexibility and personalization. Designed for seamless integration with clinical treatment planning systems, our approach assists users in generating high-quality plans efficiently. Comparative evaluations demonstrate that our method can surpasses the Varian RapidPlan model in both adaptability
    
[^303]: Consist-Retinex：一步式噪声强调一致性训练加速高质量Retinex增强

    Consist-Retinex: One-Step Noise-Emphasized Consistency Training Accelerates High-Quality Retinex Enhancement

    [https://arxiv.org/abs/2512.08982](https://arxiv.org/abs/2512.08982)

    提出Consist-Retinex框架，通过Retinex分解网络与噪声强调的一致性训练实现一步式高质量低光照图像增强，摆脱了迭代采样带来的延迟限制。

    

    基于Retinex的低光照图像增强受益于反射率与光照的分离，但近期的生成式方法通常依赖迭代采样，难以在严格的延迟预算下部署。一致性模型为实现一步式恢复提供了自然的途径，但直接将其应用于Retinex分解增强并不稳定：一步推理是在高噪声端点进行评估的，而标准训练方案在该端点几乎不提供监督信号，且仅靠时间自一致性无法确定正确的条件目标。我们提出Consist-Retinex，首先使用Retinex Transformer分解网络（TDN）获得成对的反射率图和光照图，然后利用Retinex感知的双重目标函数和自适应噪声强调的不动点采样训练两个条件一致性模型。该双重目标将轨迹一致性与成对的真值组件相结合……

    arXiv:2512.08982v4 Announce Type: replace-cross  Abstract: Retinex-based low-light image enhancement benefits from separating reflectance and illumination, yet recent generative approaches often rely on iterative sampling and are difficult to deploy under strict latency budgets. Consistency models offer a natural route to one-step restoration, but direct adaptation to Retinex-factorized enhancement is unstable: one-step inference is evaluated at the high-noise endpoint, whereas standard training schedules provide little supervision there, and temporal self-consistency alone does not determine the correct conditional target. We propose Consist-Retinex, which first uses a Retinex Transformer Decomposition Network (TDN) to obtain paired reflectance and illumination maps, then trains two conditional consistency models with a Retinex-aware dual objective and adaptive noise-emphasized fixed-point sampling. The dual objective combines trajectory consistency with paired ground-truth component 
    
[^304]: 测试利用大语言模型从心理治疗会话记录中构建个性化网络的有效性：一项概念验证研究

    Testing the Utility of Using Large Language Models to Create Personalized Networks From Therapy Session Transcripts: A Proof of Concept Study

    [https://arxiv.org/abs/2512.05836](https://arxiv.org/abs/2512.05836)

    本研究验证了利用大语言模型从心理治疗会话记录中自动生成来访者个性化临床网络的可行性，从而摆脱对密集纵向数据的依赖，支持网络驱动的个案概念化与治疗计划制定。

    

    心理治疗领域的最新进展聚焦于治疗个性化，例如基于个体网络来选择治疗模块。然而，估计个性化网络通常需要密集的纵向数据，而这类数据并不总是可行的收集。利用大语言模型（LLMs）是提升网络驱动治疗个性化可扩展性的一种解决方案。在本研究中，我们开发了一个端到端的流水线，用于自动生成来访者网络，以支持个案概念化和治疗计划制定。我们对来自77份治疗记录（N = 6）的8,028条话语进行了标注。在流水线的第一阶段，我们识别了临床相关的过程（二元分类）及其对应的维度（多标签分类）。随后，我们引入了一种两步方法，将这些过程归入具有临床意义的聚类并为聚类生成标签。最后，我们生成（原文摘要在此处截断）……

    arXiv:2512.05836v2 Announce Type: replace  Abstract: Recent advances in psychotherapy have focused on treatment personalization, such as by selecting treatment modules based on individual networks. However, estimating personalized networks typically requires intensive longitudinal data, which is not always feasible to collect. A solution to increase scalability of network-driven treatment personalization is leveraging large language models (LLMs). In this study, we developed an end-to-end pipeline for automatically generating client networks to support case conceptualization and treatment planning. We annotated 8,028 utterances from 77 therapy transcripts (N = 6). In the first stage of the pipeline, we identified clinically relevant processes (binary classification) and their corresponding dimensions (multi-label classification). Then, we introduced a two-step method that grouped the processes into clinically meaningful clusters and generated labels for the clusters. Finally, we genera
    
[^305]: SLMFix：利用强化学习的小语言模型修复领域特定语言错误

    SLMFix: Leveraging Small Language Models for Domain Specific Language Error Fixing with Reinforcement Learning

    [https://arxiv.org/abs/2511.19422](https://arxiv.org/abs/2511.19422)

    该论文提出SLMFix流水线，利用强化学习微调的小语言模型根据解释器反馈修复LLM生成代码中的语法错误，在低资源编程语言上使验证器通过率提升了40%。

    

    大语言模型（LLMs）在多种编程语言的代码生成方面展现出了令人瞩目的能力，但即使是最先进的LLM也会生成包含语法错误的程序，无法完成给定的任务，尤其是在低资源编程语言（LRPLs）上。此外，高昂的训练成本使得计算资源受限的用户无法负担LLM的微调，进一步削弱了LLM在代码生成方面的有效性。在这项工作中，我们提出了SLMFix，一种新颖的代码生成流水线，它利用通过强化学习（RL）技术微调的小语言模型（SLM），基于解释器反馈来修复LLM生成的领域特定语言（DSLs）程序中的语法错误。我们的实验结果证明了该方法在多种DSL上的有效性和泛化能力，在低资源编程语言上将验证器通过率提高了40%，并消除……（原文摘要在此处截断）

    arXiv:2511.19422v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) have shown impressive capabilities in code generation across many programming languages but even state-of-the-art LLMs generate programs that contain syntactic errors and fail to complete the given tasks, especially for low-resource programming languages (LRPLs). In addition, the high cost of training makes finetuning LLMs unaffordable for those with constrained computational resources, further weakening the effectiveness of LLMs for code generation. In this work, we propose SLMFix, a novel code generation pipeline that leverages a small language model (SLM) finetuned using reinforcement learning (RL) techniques to fix syntactic errors in LLM-generated programs for domain-specific languages (DSLs) based on interpreter feedback. Our experimental results demonstrate the effectiveness and generalizability of our approach across multiple DSLs, improving the validator pass rates by 40% on LRPLs and elimi
    
[^306]: 人工智能驱动发现中统计严谨性的结构性保障：一种函数式架构

    Structural Enforcement of Statistical Rigor in AI-Driven Discovery: A Functional Architecture

    [https://arxiv.org/abs/2511.06701](https://arxiv.org/abs/2511.06701)

    该论文提出一种函数式架构，通过Haskell的Research monad、声明式脚手架、操作系统级沙箱以及机器验证的Lean 4形式化LORD在线FDR控制，从结构上强制保证AI驱动科学发现中的统计严谨性，防止AI科学家系统因不受控的多重检验而产生虚假发现。

    

    AI-Scientist系统存在通过不受控的多重检验而制造虚假发现的风险。我们提出一种在两个层面强制统计严谨性的功能架构：一是Haskell嵌入式领域专用语言（即Research monad），它使得在不更新错误预算的情况下无法检验假设；二是声明式脚手架，用于固定数据流与统计检验方法；此外还配有操作系统级沙箱，使验证数据在LLM生成代码的运行环境中物理上不可见。我们将FDR（错误发现率）控制视为形式化需求，并将其追溯至具体实现。我们以机器验证的Lean 4形式化为基础来支撑该设计，形式化了LORD在线错误发现率控制：我们推导了其错误预算，证明了边际FDR控制，以及当阈值不随先前拒绝结果自适应调整时的完全FDR控制。随后，我们在SPARK/Ada中验证了以IEEE 7（54浮点标准计算的LORD阈值）……

    arXiv:2511.06701v4 Announce Type: replace-cross  Abstract: AI-Scientist systems risk manufacturing spurious discoveries through uncontrolled multiple testing. We present a functional architecture that enforces statistical rigor at two levels: a Haskell embedded domain-specific language (the Research monad) that makes it impossible to test a hypothesis without updating the error budget, and a declarative scaffold that fixes the data flow and the statistical test, together with an OS-level sandbox that makes validation data physically absent from the environment in which LLM-generated code runs. We treat FDR control as a formal requirement and trace it to the implementation. We ground the design in a machine-checked Lean~4 formalization of LORD online false-discovery-rate (FDR) control: we derive its error budget and prove marginal FDR control, and full FDR control when thresholds do not adapt to earlier rejections. We then verify in SPARK/Ada that the LORD thresholds, computed in IEEE~7
    
[^307]: 城市排水网络中基于稀疏测量的流量重构：数据驱动稀疏感知的应用与评估

    Flow Reconstruction from Sparse Measurements in Urban Drainage Networks: An Application and Evaluation of Data-Driven Sparse Sensing

    [https://arxiv.org/abs/2511.04556](https://arxiv.org/abs/2511.04556)

    该研究将数据驱动稀疏感知方法应用于77节点城市排水网络，证明仅需占网络4%的3个监测节点即可实现全网络流量状态的准确重构，为资源受限条件下的城市排水监测提供了高效可行的传感方案。

    

    城市化进程和日益频繁的强降雨事件正对城市排水网络造成压力。尽管对城市排水网络进行密集监测是理想的做法，但时间、预算和技术方面的实际限制阻碍了其全面实施。如何在资源受限的条件下监测和预测整个网络的流量状况是一个重大挑战。为解决这一问题，我们应用并评估了一套成熟的数据驱动稀疏感知工作流程，在一个包含77个节点的城市排水网络中进行传感器布置优化和污水流量重构。研究采用经过验证的SWMM参数化方案，利用奇异值分解（SVD）构建空间基，使用枢轴QR分解选择特定秩的传感器布置方案，并定义重构解码器。在225个留出模拟（由25个合理的校准参数集与9场降雨事件组合而成）上进行测试，一个仅含3个节点的监测布置（占网络的4%）实现了中位数……（原文摘要在此处截断）

    arXiv:2511.04556v3 Announce Type: replace  Abstract: Urbanization and increasingly frequent intense storms are placing stress on urban drainage networks. While dense monitoring of urban drainage networks is desirable, practical constraints in time, budget, and technology hinder its full implementation. How to monitor and predict flow conditions across the entire network under constrained resources is a major challenge. To address this, we utilized and evaluated an established data-driven sparse sensing (DSS) workflow for sensor placement optimization and sewer flow reconstruction in a 77-node urban drainage network. A validated SWMM parameterization was used to construct a spatial basis using singular value decomposition (SVD), select rank-specific layouts using pivoted QR, and define the reconstruction decoder. Applied to 225 held-out simulations combining 25 plausible calibrated parameter sets with 9 rainfall events, a 3-node monitoring layout (4% of the network) achieved a median sy
    
[^308]: 基于潜在独立性的可证明语音属性转换

    Provable Speech Attributes Conversion via Latent Independence

    [https://arxiv.org/abs/2510.05191](https://arxiv.org/abs/2510.05191)

    本文为语音属性转换首次建立了形式化理论框架，证明了在确定性自编码器中对潜在表示与可控属性施加独立性约束的条件下，可以实现精确且一致属性迁移的理论保证。

    

    条件生成与解耦表示学习是音频、视觉和多模态领域中受控生成的核心。然而，尽管在实证方面取得了显著进展，特别是在语音风格迁移领域，大多数现有方法仍依赖于启发式的目标函数和架构选择，对于何时以及为何能够实现可靠的属性控制缺乏理论层面的理解。在这项工作中，我们为语音属性转换构建了一个形式化框架，并对实现精确且一致迁移的充分条件进行了理论分析。我们的分析聚焦于一个确定性自编码器设定，并在此基础上增加了学习到的潜在表示与可控属性之间的独立性约束。在对数据生成过程作出显式的总体层面假设下，我们建立了将重建、独立性与属性操纵可行性联系起来的理论保证。

    arXiv:2510.05191v3 Announce Type: replace-cross  Abstract: Conditional generation and disentangled representation learning are central to controlled generation across audio, vision, and multimodal domains. However, despite strong empirical progress, particularly in speech style transfer, most existing approaches rely on heuristic objectives and architectural choices, offering limited theoretical understanding of when and why reliable attribute control is achievable. In this work, we develop a formal framework for speech attribute conversion and provide a theoretical analysis of sufficient conditions for exact and consistent transfer. Our analysis focuses on a deterministic autoencoder setting augmented with an independence constraint between the learned latent representation and the controllable attribute. Under explicit population-level assumptions about the data-generating process, we establish guarantees linking reconstruction, independence, and the feasibility of attribute manipula
    
[^309]: 剧情反转：利用三幕式叙事攻击越狱统一多模态模型

    The Plot Twist: Jailbreaking Unified Multimodal Models with a Three-Act NarrativeAttack

    [https://arxiv.org/abs/2509.26473](https://arxiv.org/abs/2509.26473)

    提出NarrativeAttack框架，利用三幕式叙事结构让统一多模态模型自行生成铺垫与结局图像，并通过图像猜谜游戏将恶意查询隐藏于良性候选中，实现对模型安全机制的越狱。

    

    统一多模态理解与生成模型日益将视觉理解与图像生成结合在单一交互式工作流中，使得生成的视觉内容能够作为后续推理的上下文。然而，现有的越狱评估大多研究文本改写或孤立的视觉提示，对于叙事式跨轮次视觉锚定的安全风险仍未得到充分探索。我们提出了NarrativeAttack，一个语义保持的视觉叙事越狱框架。NarrativeAttack采用三幕式叙事结构，由统一多模态模型自身的生成器为铺垫（事件前）和结局（事件后）阶段生成图像，使整个攻击工作流自包含，同时将恶意事件隐藏为一个暗藏的高潮。攻击最后以一个基于图像的“猜谜游戏”收尾，将原始恶意查询嵌入到良性候选之中，迫使模型选择并回答最相关的图像。

    arXiv:2509.26473v2 Announce Type: replace  Abstract: Unified Multimodal Understanding and Generation Models (UMMs) increasingly combine visual understanding and image generation within a single interactive workflow, making generated visual content available as later reasoning context. However, existing jailbreak evaluations mostly study text rewriting or isolated visual prompts, leaving the safety risk of narrative cross-turn visual grounding underexplored. We propose NarrativeAttack, a semantic-preserving visual narrative jailbreak framework. NarrativeAttack employs a three-act narrative structure in which the UMM's own generator produces images for the setup (pre-event) and resolution (post-event) stages, making the full attack workflow self-contained while concealing the malicious event as a hidden climax. The attack concludes with an image-based "guessing game" that embeds the original malicious query among benign candidates, compelling the model to select and answer the most relev
    
[^310]: AUWave：一种利用稀疏观测重建有效波高的数据驱动模型

    AUWave: A Data-Driven Model for Reconstructing Significant Wave Heights Using Sparse Observations

    [https://arxiv.org/abs/2509.19384](https://arxiv.org/abs/2509.19384)

    提出了AUWave混合深度学习框架，结合站点编码器与自注意力增强的多尺度U-Net，从稀疏浮标观测中高精度重建区域有效波高场，并通过浮标消融分析识别关键站点以指导海洋观测网络设计。

    

    从稀疏的浮标观测中重建高分辨率区域有效波高（SWH）场是海洋监测中的一项关键挑战。我们提出了AUWave，这是一个混合深度学习框架，它将基于站点的编码器与通过自注意力机制增强的多尺度U-Net相融合，以恢复区域有效波高场。AUWave利用夏威夷地区的NDBC浮标观测数据和ERA5再分析数据进行训练和验证，实现了较高的精度。它始终优于一个代表性基线模型，尤其是在配置多于单个浮标的情形下，展示了其多尺度架构的优势。空间误差分析表明，正如预期的那样，模型性能在观测站点附近最高。此外，浮标消融研究识别出了关键的锚定站点，这些站点一旦被移除会导致性能不成比例地下降，从而为观测网络设计提供了可操作的指导。AUWave为填补数据空白提供了一条可扩展的路径。

    arXiv:2509.19384v2 Announce Type: replace-cross  Abstract: Reconstructing high-resolution regional significant wave height (SWH) fields from sparse buoy observations is a critical challenge for ocean monitoring. We introduce AUWave, a hybrid deep learning framework that fuses a station-wise encoder with a multi-scale U-Net enhanced by self-attention to recover regional SWH fields. Trained and validated using NDBC buoy observations and ERA5 reanalysis over the Hawaii region, AUWave achieves high accuracy. It consistently outperforms a representative baseline, especially in configurations with more than a single buoy, demonstrating the benefit of its multi-scale architecture. Spatial error analysis shows performance is highest near observation sites, as expected. Further, buoy ablation studies identify critical anchor stations whose removal disproportionately degrades performance, offering actionable guidance for observational network design. AUWave provides a scalable pathway for gap-fi
    
[^311]: 神经桥过程

    Neural Bridge Processes

    [https://arxiv.org/abs/2508.07220](https://arxiv.org/abs/2508.07220)

    提出神经桥过程（NBP），用输入锚定的桥轨迹替代无条件前向核，使条件输入信息在扩散的含噪状态中就被编码，从而实现对随机函数更具表达力且强条件依赖的学习。

    

    从部分观测的上下文-目标对中学习随机函数，需要模型具备强表达能力、不确定性感知能力以及对输入的强条件依赖。神经扩散过程（NDPs）通过去噪扩散提升了表达能力，但其前向过程与输入无关；输入仅进入反向去噪器，因此含噪的训练状态本身并不编码条件输入信息。我们提出神经桥过程（NBPs），用输入锚定的桥轨迹替代无条件的前向核。当输入与输出维度不同时，NBP学习一个输出空间锚点 $a_\psi(x)=P_\psi(x)$，使坐标或其他输入能够引导生成路径，而无需改变去噪主干网络。我们从理论上证明，过程级锚定诱导了逐路径的输入可区分性，将关于 x 的信息注入含噪状态，并创建了传统方法中不可得的直接梯度通路。

    arXiv:2508.07220v4 Announce Type: replace-cross  Abstract: Learning stochastic functions from partially observed context-target pairs requires models that are expressive, uncertainty-aware, and strongly conditioned on inputs. Neural Diffusion Processes (NDPs) improve expressivity with denoising diffusion, but their forward process is input-independent; inputs only enter the reverse denoiser, so the noisy training states themselves do not encode the conditioning inputs. We propose Neural Bridge Processes (NBPs), which replace the unconditional forward kernel with an input-anchored bridge trajectory. When input and output dimensions differ, NBP learns an output-space anchor $a_\psi(x)=P_\psi(x)$, allowing coordinates or other inputs to guide the generative path without changing the denoising backbone. We show theoretically that process-level anchoring induces pathwise input distinguishability, injects information about x into noisy states, and creates a direct gradient pathway unavailabl
    
[^312]: 同行评议作为结构化评论：不可篡改的身份、公开对话与可复现的学术研究

    Peer Review as Structured Commentary: Immutable Identity, Public Dialogue, and Reproducible Scholarship

    [https://arxiv.org/abs/2506.22497](https://arxiv.org/abs/2506.22497)

    本文提出将同行评议重构为基于区块链不可篡改审计和AI迭代综合的结构化公开评论系统，实现透明、可复现、可追溯的学术评价新范式。

    

    本文将同行评议重新概念化为结构化的公开评论。传统的学术验证受到匿名性、延迟性和守门机制的阻碍。我们提出一个透明的、与身份绑定的、可复现的学术评价系统，以公开评论为基础。利用区块链实现不可篡改的审计追踪，利用人工智能实现迭代综合，我们设计了一个能够激励知识贡献、捕捉认知演变并支持可追溯声誉动态的框架。这一模型赋能从计算科学到人文学科的各个领域，将学术知识重新定义为一种动态演进的鲜活过程，而非静态的凭证。

    arXiv:2506.22497v2 Announce Type: replace-cross  Abstract: This paper reconceptualises peer review as structured public commentary. Traditional academic validation is hindered by anonymity, latency, and gatekeeping. We propose a transparent, identity-linked, and reproducible system of scholarly evaluation anchored in open commentary. Leveraging blockchain for immutable audit trails and AI for iterative synthesis, we design a framework that incentivises intellectual contribution, captures epistemic evolution, and enables traceable reputational dynamics. This model empowers fields from computational science to the humanities, reframing academic knowledge as a living process rather than a static credential.
    
[^313]: 评估即一切：通过评估设计对大语言模型推理能力的策略性夸大

    Evaluation is All You Need: Strategic Overclaiming of LLM Reasoning Capabilities Through Evaluation Design

    [https://arxiv.org/abs/2506.04734](https://arxiv.org/abs/2506.04734)

    本研究揭示评估条件的细微差异会导致Deepseek-R1-Distill系列等推理模型的基准测试结果大幅波动，使其声称的性能提升难以可靠复现，并倡导建立更严格的模型性能评估范式。

    

    以Deepseek-R1-Distill系列为代表的推理模型因其在数学、科学、编程等领域的出色表现而被开源社区广泛采用。然而，我们的研究揭示，其基准测试评估结果会受到多种因素的影响而产生显著波动，评估条件的细微差异就可能导致结果的重大变化。在基于Deepseek-R1-Distill系列微调的其他开源推理模型以及QwQ-32B模型中也观察到类似现象，使得其所声称的性能提升难以可靠复现。因此，我们倡导建立更严格的模型性能评估范式，并呈现了我们对Deepseek-R1-Distill系列模型的实证评估。

    arXiv:2506.04734v3 Announce Type: replace  Abstract: Reasoning models represented by the Deepseek-R1-Distill series have been widely adopted by the open-source community due to their strong performance in mathematics, science, programming, and other domains. However, our study reveals that their benchmark evaluation results are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results. Similar phenomena are observed in other open-source inference models fine-tuned based on the Deepseek-R1-Distill series, as well as in the QwQ-32B model, making their claimed performance improvements difficult to reproduce reliably. Therefore, we advocate for the establishment of a more rigorous paradigm for model performance evaluation and present our empirical assessments of the Deepseek-R1-Distill series models.
    
[^314]: 通过启发式适配与超词学习实现语言模型的分词器灵活性

    Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning

    [https://arxiv.org/abs/2505.09738](https://arxiv.org/abs/2505.09738)

    提出了 Tokenadapt——一种与模型无关的分词器移植方法，结合面向多词“超词”的预分词学习，使语言模型能以较低计算成本灵活替换分词器，同时提升压缩效率并减少词元碎片化。

    

    预训练语言模型（LLMs）常常受限于其固定的分词方案，导致效率低下和性能受限，尤其是在多语言或专业化应用场景中。这种“分词器锁定”问题带来了重大挑战：克服该问题的标准方法通常需要极高的计算资源。尽管通过启发式初始化进行分词器替换旨在减轻这一负担，但现有方法往往需要进行穷举式的残差微调，且仍可能无法完整保留语义细节或充分解决底层的压缩低效问题。我们的框架引入了两项创新：其一，Tokenadapt，一种与模型无关的分词器移植方法；其二，新颖的预分词学习方法，用于多词“超词”（Supertokens），以增强压缩效果并减少碎片化。Tokenadapt 通过一种结合两种……的混合启发式方法来初始化新的独特词元嵌入。

    arXiv:2505.09738v2 Announce Type: replace-cross  Abstract: Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two me
    
[^315]: MedHal：一个用于医学幻觉检测的合成数据集

    MedHal: a Synthetic Dataset for Medical Hallucination Detection

    [https://arxiv.org/abs/2504.08596](https://arxiv.org/abs/2504.08596)

    MedHal是一个涵盖内在与外在幻觉的大规模合成医学数据集，用于训练和评估医学文本幻觉检测模型，基于该数据集训练的基线模型优于通用幻觉检测方法。

    

    幻觉是指AI系统生成非事实性内容的现象，在医学场景中会带来严重风险，因为错误可能直接影响患者的治疗效果。我们提出了MedHal，这是一个大规模数据集，专门用于评估模型能力并训练模型执行医学文本中的幻觉检测任务。当前的幻觉检测方法在应用于医学等专业领域时面临重大局限，可能造成灾难性后果。MedHal通过纳入多样化的医学文本来源和任务（涵盖内在幻觉与外在幻觉），并提供大量适合训练医学幻觉检测模型的数据样本来解决这一问题。我们通过训练和评估一个基线医学幻觉检测模型证明了MedHal的实用性，其表现优于通用幻觉检测方法。该资源为医学领域的幻觉检测研究和应用提供了重要支持。

    arXiv:2504.08596v3 Announce Type: replace-cross  Abstract: Hallucination, the generation of non factual content by AI systems, poses serious risks in medical contexts, where errors can directly affect patient outcomes. We present MedHal, a large-scale dataset specifically designed to assess capabilities and train models on the task of hallucination detection in medical texts. Current hallucination detection methods face significant limitations when applied to specialized domains like medicine, where they can have disastrous consequences. MedHal addresses this issue by incorporating diverse medical text sources and tasks covering both intrinsic and extrinsic hallucinations, and by providing a substantial volume of data samples suitable for training medical hallucination detection models. We demonstrate MedHal's utility by training and evaluating a baseline medical hallucination detection model, showing improvements over general-purpose hallucination detection approaches. This resource e
    
[^316]: SkillFlow：可扩展且高效的智能体技能检索系统

    SkillFlow: Scalable and Efficient Agent Skill Retrieval System

    [https://arxiv.org/abs/2504.06188](https://arxiv.org/abs/2504.06188)

    SkillFlow是首个面向智能体技能发现的多阶段检索系统，它将技能获取视为信息检索问题，通过密集检索、交叉编码器重排序和LLM选择四个阶段，从约3.5万个社区技能定义中高效检索出最相关的技能。

    

    AI智能体可以通过在推理时将可复用技能加载到上下文中来扩展其能力，然而为智能体配备过多技能（尤其是无关技能）会降低其性能。随着社区驱动的技能库不断增长，智能体需要一种能够从大型技能库中选择性地仅检索最相关技能的方法。我们提出了SkillFlow，这是首个开放的多阶段智能体技能发现检索系统，它将技能获取构建为一个信息检索问题，其语料库包含从GitHub索引的约35K个社区贡献的SKILL.md技能定义。该流水线通过四个阶段（密集检索、两轮交叉编码器重排序以及基于LLM的选择）逐步收窄大规模候选集，在每个阶段平衡召回率与精确率。我们在两个编程基准上评估了SkillFlow：SkillsBench（包含87个任务和229个匹配技能的基准）和Terminal-Bench（一个仅提供8个……的基准，摘要原文在此处截断）。

    arXiv:2504.06188v3 Announce Type: replace  Abstract: AI agents can extend their capabilities at inference time by loading reusable skills into context, yet equipping an agent with too many skills, particularly irrelevant ones, degrades performance. As community-driven skill repositories grow, agents need a way to selectively retrieve only the most relevant skills from a large library. We present SkillFlow, the first open, multi-stage retrieval system for agent skill discovery that frames skill acquisition as an information retrieval problem over a corpus of ~35K community-contributed SKILL.md definitions indexed from GitHub. The pipeline progressively narrows a large candidate set through four stages (dense retrieval, two rounds of cross-encoder reranking, and LLM-based selection), balancing recall and precision at each stage. We evaluate SkillFlow on two coding benchmarks: SkillsBench, a benchmark of 87 tasks and 229 matched skills; and Terminal-Bench, a benchmark that provides only 8
    
[^317]: LEAD：用于阿尔茨海默病检测的脑电图基础模型

    LEAD: An EEG Foundation Model for Alzheimer's Disease Detection

    [https://arxiv.org/abs/2502.01678](https://arxiv.org/abs/2502.01678)

    本文构建了迄今最大的EEG-AD数据集（2,238名受试者），并提出首个脑电图阿尔茨海默病检测基础模型LEAD，其门控时-空Transformer可适应异构EEG数据，配合被试正则化训练策略提升了跨被试泛化能力。

    

    脑电图（EEG）为检测阿尔茨海默病（AD）提供了一种无创、高度可及且经济高效的方法。然而，现有方法，无论是基于手工特征工程还是标准深度学习，都面临三大挑战：1）缺乏大规模基于EEG的AD数据集用于稳健的表示学习和评估；2）跨被试泛化能力有限；3）难以适应高度异构的数据。为应对这些挑战，我们构建了迄今世界上最大的EEG-AD语料库，包含2,238名受试者。利用这一独特资源，我们提出了LEAD，这是首个用于基于EEG的AD检测的基础模型。具体而言，我们设计了一种门控时-空Transformer，能够适应具有不同长度、通道配置和采样率的EEG记录。此外，我们引入了一种被试正则化训练策略，以增强端到端的子（摘要原文在此处截断）。

    arXiv:2502.01678v5 Announce Type: replace-cross  Abstract: Electroencephalography (EEG) provides a non-invasive, highly accessible, and cost-effective approach for detecting Alzheimer's disease (AD). However, existing methods, whether based on handcrafted feature engineering or standard deep learning, face three major challenges: 1) the lack of large-scale EEG-based AD datasets for robust representation learning and evaluation; 2) limited cross-subject generalizability; and 3) difficulty in adapting to highly heterogeneous data. To address these challenges, we curate the world's largest EEG-AD corpus to date, comprising 2,238 subjects. Leveraging this unique resource, we propose LEAD, the first foundation model for EEG-based AD detection. Specifically, we design a gated temporal-spatial Transformer that can adapt to EEG recordings with diverse lengths, channel configurations, and sampling rates. In addition, we introduce a subject-regularized training strategy to enhance end-to-end sub
    
[^318]: 可微博弈中基于偏好的对手塑造

    Preference-based opponent shaping in differentiable games

    [https://arxiv.org/abs/2412.03072](https://arxiv.org/abs/2412.03072)

    本文提出基于偏好的对手塑造方法，通过塑造智能体对合作的偏好来增强多智能体博弈中的策略学习，克服了传统对手建模方法缺乏对行为偏好建模和泛化能力的局限。

    

    多智能体博弈环境中的策略学习是一个具有挑战性的问题。由于每个智能体的奖励由联合策略决定，旨在最大化自身奖励的贪婪学习策略可能会陷入局部最优。近期研究提出了针对博弈环境的对手建模和塑造方法，这些方法通过建模其他智能体的策略及其更新过程来提高策略学习的效率。然而，这些方法通常依赖于对对手策略变化的简单预测，由于缺乏对合作与竞争等行为偏好的建模，它们通常仅适用于预定义的场景，缺乏泛化能力。在本文中，我们提出了一种新颖的基于偏好的对手塑造方法，通过塑造智能体对合作的偏好来增强策略学习过程。我们引入了偏好参数，该

    arXiv:2412.03072v2 Announce Type: replace  Abstract: Strategy learning in game environments with multi-agent is a challenging problem. Since each agent's reward is determined by the joint strategy, a greedy learning strategy that aims to maximize its own reward may fall into a local optimum. Recent studies have proposed the opponent modeling and shaping methods for game environments. These methods enhance the efficiency of strategy learning by modeling the strategies and updating processes of other agents. However, these methods often rely on simple predictions of opponent strategy changes. Due to the lack of modeling behavioral preferences such as cooperation and competition, they are usually applicable only to predefined scenarios and lack generalization capabilities. In this paper, we propose a novel Preference-based Opponent Shaping (PBOS) method to enhance the strategy learning process by shaping agents' preferences towards cooperation. We introduce the preference parameter, which
    

