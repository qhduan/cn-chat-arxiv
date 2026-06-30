# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^2] | [Robust Onion: Peeling Open Vocab Object Detectors Under Noise](https://arxiv.org/abs/2606.26734) | 研究发现开放词汇目标检测器的鲁棒性主要由视觉主干和图像域决定，而非预训练策略或注释，并通过受控噪声实验揭示了特征坍塌是鲁棒性退化的关键机制。 |
| [^3] | [CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation](https://arxiv.org/abs/2606.26423) | 本文提出通过组合简单、独立的行为来自然涌现复杂操作能力，避免依赖刚性流程或整体策略，从而同时实现高精度和泛化能力。 |
| [^4] | [Unbiased Canonical Set-Valued Oracles Via Lattice Theory](https://arxiv.org/abs/2606.26418) | 本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。 |
| [^5] | [Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model](https://arxiv.org/abs/2606.26373) | 本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。 |
| [^6] | [The Verification Horizon: No Silver Bullet for Coding Agent Rewards](https://arxiv.org/abs/2606.26300) | 本文指出，在编码智能体领域，验证解决方案比生成解决方案更难，并提出了验证信号质量的三个维度（可扩展性、忠实性和鲁棒性），强调同时实现这些维度面临根本性挑战。 |
| [^7] | [The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators](https://arxiv.org/abs/2606.26294) | 本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。 |
| [^8] | [Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models](https://arxiv.org/abs/2606.25041) | Wan-Streamer是一个原生流式、端到端的统一Transformer模型，通过块因果注意力机制联合学习感知、推理、生成和跨模态同步，无需外部模块即可实现低延迟的全双工音视频实时交互。 |
| [^9] | [Trust in Generative AI for Health Information Consumption and the Effect of Learned Dependency: An Experimental Investigation](https://arxiv.org/abs/2606.20605) | 本研究通过两个随机对照实验发现，生成式AI的健康信息准确性显著提升用户信任，但用户对AI的习得依赖会增强这种信任，而文本高亮未能有效减少对错误信息的过度依赖。 |
| [^10] | [To Use AI as Dice of Possibilities with Timing Computation](https://arxiv.org/abs/2605.01134) | 本文提出了一种基于动词范式的因果推理框架，通过时间计算与因果事实定义，使AI能够从数据中自动发现临床轨迹并进行反事实推理，在乳腺癌患者数据上首次实现了纯数据驱动的因果世界模型。 |
| [^11] | [Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models](https://arxiv.org/abs/2505.12343) | 本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。 |
| [^12] | [SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces](https://arxiv.org/abs/2403.07711) | 提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题 |
| [^13] | [Modelling Human Values for AI Reasoning](https://arxiv.org/abs/2402.06359) | 本研究详细介绍了一个关于人类价值观的形式化模型，并展示了它在AI推理中的应用。研究通过基于社会心理学研究的关键思想，为AI系统与人类价值观的一致性提供了具体的计算表示。 |
| [^14] | [Assortment Planning with Sponsored Products](https://arxiv.org/abs/2402.06158) | 本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。 |
| [^15] | [Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method.](http://arxiv.org/abs/2304.11171) | 本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。 |

# 详细

[^1]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^2]: 鲁棒洋葱：在噪声下剥离开放词汇目标检测器

    Robust Onion: Peeling Open Vocab Object Detectors Under Noise

    [https://arxiv.org/abs/2606.26734](https://arxiv.org/abs/2606.26734)

    研究发现开放词汇目标检测器的鲁棒性主要由视觉主干和图像域决定，而非预训练策略或注释，并通过受控噪声实验揭示了特征坍塌是鲁棒性退化的关键机制。

    

    摘要：由于开放词汇目标检测器（OV-ODs）架构的复杂性，真实世界噪声对其影响仍未被充分理解。我们提出了名为“鲁棒洋葱”的综合分析，这是一项实证研究，通过使用受控的合成视觉退化逐层剥离OV-ODs，揭示了鲁棒性如何、为何以及在何处退化，并系统地分析了特征坍塌。我们的发现表明，具有相似视觉主干的模型表现出相似的鲁棒性，这是由相似层中相似的特征坍塌驱动的，而预训练策略、架构细节和标题监督等因素贡献甚微。鲁棒性主要由图像域而非注释决定，这解释了为何COCO和LVIS上的鲁棒性影响相似，以及为何像ODinW-13这样的数据集会因大而孤立的物体而给人以鲁棒性膨胀的印象。最后，我们通过改进鲁棒性验证了我们的见解。

    arXiv:2606.26734v1 Announce Type: cross  Abstract: The impact of real-world noise on Open Vocabulary Object Detectors (OV-ODs) remains poorly understood due to their architectural complexity. We present our comprehensive analysis Robust Onion, an empirical study that uses controlled synthetic visual degradations to peel OV-ODs layer-by-layer, revealing how, why, and where robustness degrades, systematically analyzing feature collapse. Our findings reveal that models with similar vision backbones exhibit comparable robustness, driven by similar feature collapse at similar layers, while factors such as pretraining strategy, architectural nuances, and caption supervision contribute little. Robustness is primarily governed by the image domain rather than annotations, explaining the similar robustness impact on COCO and LVIS, and why datasets like ODinW-13 can give an impression of inflated robustness due to large, isolated objects. Finally, we validate our insights by improving robustness 
    
[^3]: CoStream：组合简单行为以实现可泛化的复杂操作

    CoStream: Composing Simple Behaviors for Generalizable Complex Manipulation

    [https://arxiv.org/abs/2606.26423](https://arxiv.org/abs/2606.26423)

    本文提出通过组合简单、独立的行为来自然涌现复杂操作能力，避免依赖刚性流程或整体策略，从而同时实现高精度和泛化能力。

    

    arXiv:2606.26423v1 公告类型：交叉 摘要：长期、高接触的复杂操作任务，例如将GPU插入PCIe插槽，既需要毫米级的高精度，又需要对新任务具有开箱即用的泛化能力。现有范式难以同时满足这两点：经典流程使用脆弱且任务特定的接口来实现高精度控制，但需要昂贵的流程重新设计才能适应新任务；而端到端整体策略虽能提供更好的泛化能力，但在复杂、分布外的任务上缺乏高精度，除非用新数据重新训练。两种范式都共享一个隐含假设：一旦获得操作能力，就必须以刚性流程或整体形式部署，而不能自由分解和重新组合。在本文中，我们展示了复杂操作能力可以自然地从简单、独立行为的组合中涌现。我们不部署整体策略或刚性流程，而是提出……

    arXiv:2606.26423v1 Announce Type: cross  Abstract: Long-horizon, contact-rich complex manipulation tasks, such as seating a GPU into a PCIe slot, demand both millimeter high precision and out-of-the-box generalization to new tasks. Existing paradigms struggle to satisfy both: classical pipelines use brittle, task-specific interfaces to achieve high-precision control but require costly pipeline redesigns to adapt to new tasks, whereas monolithic end-to-end policies provide better generalization but lack high precision on complex, out-of-distribution tasks unless retrained with new data. Both paradigms share an implicit assumption: once a manipulation capability is acquired, it must be deployed as a rigid pipeline or monolithic whole, rather than being freely decomposed and recomposed. In this paper, we show that complex manipulation capabilities can emerge naturally from the composition of simple, independent behaviors. Rather than deploying a monolithic policy or a rigid pipeline, we p
    
[^4]: 基于格理论的无偏规范集值预言机

    Unbiased Canonical Set-Valued Oracles Via Lattice Theory

    [https://arxiv.org/abs/2606.26418](https://arxiv.org/abs/2606.26418)

    本文通过Knaster–Tarski不动点定理，在完备格框架下提出了一种规范的非平凡credal集，解决了自指预言机在无偏和自洽约束下的唯一性问题。

    

    非智能体“预言机”AI在估计未来事件概率时面临自指问题：一旦其答案被学习并采取行动，就会改变它被要求报告的概率本身。针对科学家AI计划所倡导的一种回应是只询问反事实问题，并假设答案没有影响进行评估。我们观察到，这类答案一旦被学习就会变得无关紧要，恰恰是因为其前提随后变为假。因此，我们探索了一种自指替代方案：预言机报告的不是单一概率，而是一个同时无偏且与学习后果自洽的credal集。朴素的自洽性要求被太多集合满足（包括无用的答案[0,1]），因此问题在于挑选出一个规范的、非平凡的成员。我们通过闭包完备格上的Knaster–Tarski不动点定理实现了这一点。

    arXiv:2606.26418v1 Announce Type: new  Abstract: A non-agentic "oracle" AI that estimates probabilities of future events faces a self-reference problem: once its answer is learned and acted upon, it can change the very probability it was asked to report. One response, advocated for the Scientist AI programme, is to ask only counterfactual questions, evaluated as if the answer had no influence. We observe that such answers tend to become irrelevant the moment they are learned, precisely because their premise is then false. We therefore explore a self-referential alternative in which the oracle reports not a single probability but a credal set that is simultaneously unbiased and self-consistent with the consequences of being learned. The naive self-consistency requirement is satisfied by too many sets (including the useless answer $[0,1]$), so the problem is to single out a canonical, nontrivial member. We do so with the Knaster--Tarski fixed-point theorem on the complete lattice of clos
    
[^5]: 混合隐私感知语义搜索：受限威胁模型下基于SVD截断文档几何与CKKS加密查询重排序

    Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

    [https://arxiv.org/abs/2606.26373](https://arxiv.org/abs/2606.26373)

    本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。

    

    arXiv:2606.26373v1 公告类型：交叉 摘要：稠密嵌入为语义搜索和检索增强生成提供了强大支持，但嵌入反转攻击可以从向量中重建源文本：当向量数据库泄露时，其背后的文档也会随之泄露。教科书式的防御措施是极端方案——对整个搜索进行同态加密是可靠的，但在百万级文档规模下速度过慢，而隐私噪声在提供保护之前就已严重降低排序质量。我们研究了一条中间路径，利用静态集合与动态查询之间的不对称性。集合通过几何方式保护：每个向量被截断到低维SVD子空间，并通过仅由所有者知道的秘密正交变换进行旋转。查询通过密码学方式保护：在CKKS同态加密下进行重排序，因此诚实但好奇的服务器永远无法看到查询或分数。CKKS参数来自一个小型离线基准测试。我们证明了重构的下界紧致性。

    arXiv:2606.26373v1 Announce Type: cross  Abstract: Dense embeddings power semantic search and retrieval-augmented generation, but embedding-inversion attacks can reconstruct source text from a vector: when a vector database leaks, the documents behind it leak too. The textbook defences are extremes - encrypting the whole search homomorphically is sound but too slow at million-document scale, while privacy noise degrades ranking long before it protects. We study a middle path exploiting the asymmetry between the static collection and the dynamic query. The collection is protected geometrically: each vector is truncated onto a lower-dimensional SVD subspace and rotated by a secret orthogonal transform known only to the owner. The query is protected cryptographically: it is reranked under CKKS homomorphic encryption, so an honest-but-curious server never sees the query or the scores. CKKS parameters come from a small offline benchmark.   We prove a tight lower bound on the reconstruction 
    
[^6]: 验证地平线：编码智能体奖励没有银弹

    The Verification Horizon: No Silver Bullet for Coding Agent Rewards

    [https://arxiv.org/abs/2606.26300](https://arxiv.org/abs/2606.26300)

    本文指出，在编码智能体领域，验证解决方案比生成解决方案更难，并提出了验证信号质量的三个维度（可扩展性、忠实性和鲁棒性），强调同时实现这些维度面临根本性挑战。

    

    arXiv:2606.26300v1 公告类型：新论文 摘要：经典直觉认为，验证一个解决方案比生成一个解决方案更容易。对于当今的编码智能体而言，这种直觉正在被颠覆：随着基础模型发展出更强的推理能力，工程工具也变得日益复杂，生成复杂的候选解决方案已不再困难——而可靠地验证这些方案反而成了更棘手的问题。我们能够构建的每一个验证器都只是人类意图的代理，而非意图本身。这使得验证面临双重困难：首先，意图本质上是不明确的，因此很难忠实地检查它是否被满足；其次，在模型训练过程中，优化会扩大代理与意图之间的差距——表现为奖励黑客或信号饱和。为解决这一问题，我们从三个维度——可扩展性、忠实性和鲁棒性——描述了验证信号的质量，并论证了同时实现所有这些维度的困难。

    arXiv:2606.26300v1 Announce Type: new  Abstract: A classical intuition holds that verifying a solution is easier than producing one. For today's coding agents, this intuition is being inverted: as foundation models develop stronger reasoning capabilities and engineering harnesses grow more sophisticated, generating complex candidate solutions is no longer difficult -- reliably verifying them has become the harder problem. Every verifier we can build is only a proxy for human intent, never the intent itself. This makes verification subject to a twofold difficulty: first, intent is underspecified by nature, making it inherently hard to faithfully check whether it has been fulfilled; second, during model training, optimization widens the gap between proxy and intent -- manifesting as reward hacking or signal saturation. To address this, we characterize the quality of verification signals along three dimensions -- scalability, faithfulness, and robustness -- and argue that achieving all th
    
[^7]: 红皇后哥德尔机：共同进化的智能体及其评估者

    The Red Queen G\"odel Machine: Co-Evolving Agents and Their Evaluators

    [https://arxiv.org/abs/2606.26294](https://arxiv.org/abs/2606.26294)

    本文提出红皇后哥德尔机（RQGM），通过将评估者纳入进化循环，使智能体能在非平稳评估标准下进行递归自我改进，从而突破静态基准的限制。

    

    arXiv:2606.26294v1 公告类型：交叉 摘要：自我改进的智能体在编程基准测试中已达到最先进水平（SOTA），并最近被扩展到通用领域。然而，它们的搜索方法通常假设一个静态的评估标准：一个固定的验证器、基准测试或标注数据集，在智能体改进过程中保持有效。这忽略了进化的一个核心特征：物种随着环境的变化而适应。我们旨在将同样的原则引入递归自我改进，使评估成为改进循环的一部分，并将搜索开放给不断演化的评估者、对抗性目标和可能超越静态基准的动态效用函数。我们引入了红皇后哥德尔机（RQGM），这是一个用于非平稳效用下递归自我改进的演化框架。RQGM通过受控的效用演化实现了这一点：搜索被组织成具有固定期内评估标准的周期，而效用可以跨周期演化。

    arXiv:2606.26294v1 Announce Type: cross  Abstract: Self-improving agents are state-of-the-art (SOTA) on agentic coding benchmarks and have recently been extended to general domains. However, their search methods generally assume a stationary evaluation criterion: a fixed verifier, benchmark, or labeled dataset that remains valid as the agent improves. This ignores a central feature of evolution: species adapt as their environments change with them. We aim to bring the same principle to recursive self-improvement, making evaluation part of the improvement loop and opening search to evolving evaluators, adversarial objectives, and dynamic utilities that may surpass static benchmarks. We introduce the Red Queen Godel Machine (RQGM), an evolutionary framework for recursive self-improvement under non-stationary utilities. The RQGM makes this possible through controlled utility evolution: search is organized into epochs with a fixed within-epoch evaluation criterion, while the utility can be
    
[^8]: Wan-Streamer v0.1：端到端实时交互基础模型

    Wan-Streamer v0.1: End-to-end Real-time Interactive Foundation Models

    [https://arxiv.org/abs/2606.25041](https://arxiv.org/abs/2606.25041)

    Wan-Streamer是一个原生流式、端到端的统一Transformer模型，通过块因果注意力机制联合学习感知、推理、生成和跨模态同步，无需外部模块即可实现低延迟的全双工音视频实时交互。

    

    arXiv:2606.25041v2 公告类型：替换交叉 摘要：我们提出了Wan-Streamer，一个原生流式、端到端的交互基础模型，从头开始设计用于实时、低延迟、全双工的音视频交互。Wan-Streamer在单个Transformer内无缝地将语言、音频和视频作为输入和输出进行建模，其中序列表示为交错排列的视觉、音频和文本输入令牌，以及视觉、音频和文本输出令牌，通过块因果注意力机制协调实现增量流式处理。与依赖独立VAD、ASR、语言、TTS、音频驱动动画或视频生成模块的级联交互系统不同，Wan-Streamer不依赖外部语言、语音、虚拟形象或视频生成模块：感知、推理、生成、响应时序、对话轮次管理和跨模态同步都在一个统一模型中联合学习，从而减少流水线延迟和错误累积。为了支持自然的交互，该系统还集成了自适应响应调度机制。

    arXiv:2606.25041v2 Announce Type: replace-cross  Abstract: We present Wan-Streamer, a native-streaming, end-to-end interactive foundation model designed from the ground up for real-time, low-latency, full-duplex audio-visual interaction. Wan-Streamer seamlessly models language, audio, and video as both input and output within a single Transformer, where the sequence is represented as interleaved visual, audio, and text input tokens together with visual, audio, and text output tokens, coordinated by block-causal attention for incremental streaming. Unlike cascaded interactive systems that rely on separate VAD, ASR, language, TTS, audio-driven animation, or video-generation modules, Wan-Streamer does not rely on external language, speech, avatar, or video-generation modules: perception, reasoning, generation, response timing, turn management, and cross-modal synchronization are learned jointly within one unified model, reducing pipeline latency and error accumulation. To support natural 
    
[^9]: 生成式人工智能在健康信息消费中的信任与习得依赖效应：一项实验研究

    Trust in Generative AI for Health Information Consumption and the Effect of Learned Dependency: An Experimental Investigation

    [https://arxiv.org/abs/2606.20605](https://arxiv.org/abs/2606.20605)

    本研究通过两个随机对照实验发现，生成式AI的健康信息准确性显著提升用户信任，但用户对AI的习得依赖会增强这种信任，而文本高亮未能有效减少对错误信息的过度依赖。

    

    背景：生成式人工智能（GenAI）正越来越多地被用于健康信息领域，但其对用户信任校准的影响仍不明确。目的：本研究探讨对GenAI的习得依赖是否会影响用户对AI生成健康信息的信任，以及文本高亮是否能减少对错误输出的过度依赖。方法：开展了两个随机对照实验，分别涉及338名大学生和563名亚马逊土耳其机器人参与者。两个实验均采用2×2的组间设计，操纵信息准确性（正确与错误）和文本高亮（高亮与不高亮）。使用经过验证的量表测量信任和习得依赖，并通过线性回归模型检验主效应和交互效应。结果：在两个实验中，信息准确性显著提高了信任（p < 0.001），而习得依赖与信任呈正相关。

    arXiv:2606.20605v2 Announce Type: replace-cross  Abstract: Background: Generative artificial intelligence (GenAI) is increasingly used for health information, yet its influence on users' trust calibration remains unclear.   Objective: This study examines whether learned dependency on GenAI influences trust in AI-generated health information and whether text highlighting reduces overreliance on incorrect outputs.   Methods: Two randomized controlled experiments were conducted with 338 college students and 563 Amazon Mechanical Turk participants. Both experiments used a 2 by 2 between-subjects design manipulating information accuracy (correct versus incorrect) and text highlighting (highlight versus no highlight). Trust and learned dependency were measured using validated scales, and linear regression models tested main and interaction effects.   Results: In both experiments, information accuracy significantly increased trust (p < 0.001), while learned dependency was positively associate
    
[^10]: 将人工智能用作带有时间计算的“可能性骰子”

    To Use AI as Dice of Possibilities with Timing Computation

    [https://arxiv.org/abs/2605.01134](https://arxiv.org/abs/2605.01134)

    本文提出了一种基于动词范式的因果推理框架，通过时间计算与因果事实定义，使AI能够从数据中自动发现临床轨迹并进行反事实推理，在乳腺癌患者数据上首次实现了纯数据驱动的因果世界模型。

    

    arXiv:2605.01134v3 公告类型：替换 摘要：当前以名词为主的建模范式从根本上限制了人工智能的发展，使其无法充分将未来表征为一个开放的时间维度。本文引入了一种以动词为主的范式，并给出了“时间计算”和“因果事实”的精确定义，从而使人工智能能够作为自发构建因果推理世界模型的工具。将该框架应用于来自3276名乳腺癌患者的纵向电子健康记录数据，实证结果表明：(1) 自动发现具有临床意义的患者轨迹，以及(2) 反事实时间推理，即一种“假设机器”。这两项结果均以纯数据驱动的方式实现，无需借助先验领域知识，据我们所知，这是机器学习文献中首次展示此类成果。

    arXiv:2605.01134v3 Announce Type: replace  Abstract: The dominant noun-based modeling paradigm has fundamentally constrained AI development, precluding any adequate representation of the future as an open temporal dimension. This paper introduces a verb-based paradigm, together with precise definitions of \emph{timing computation} and \emph{causal factum}, that enables AI to function as an instrument for spontaneously constructing a causal-reasoning world model.   Applied to longitudinal EHR data from 3,276 breast cancer patients, the framework empirically demonstrates: (1) automatic discovery of clinically significant patient trajectories, and (2) counterfactual timing deduction, that is, a \emph{What-If Machine}. Both results are achieved in a purely data-driven manner, without recourse to prior domain knowledge, and represent, to our knowledge, the first such demonstrations in the machine learning literature.
    
[^11]: 通过层间一致性聚合缓解大型视觉语言模型中的幻觉

    Mitigating Hallucinations via Inter-Layer Consistency Aggregation in Large Vision-Language Models

    [https://arxiv.org/abs/2505.12343](https://arxiv.org/abs/2505.12343)

    本文提出了一种名为DCLA的免训练解码方法，通过聚合前层表示构建动态语义参考来纠正语义偏差的层，从而有效缓解大型视觉语言模型中的幻觉问题，并在多个模型和基准上显著提升性能。

    

    arXiv:2505.12343v2 公告类型：替换交叉 摘要：尽管大型视觉语言模型（LVLMs）能力令人印象深刻，但它们仍然容易产生幻觉，即生成的内容与输入图像不一致。现有的免训练幻觉缓解方法通常存在性能不稳定且对超参数设置高度敏感的问题，这限制了其实用性和更广泛的采用。在本文中，我们提出了一种通过层聚合进行层间一致性解码的方法（DCLA），这是一种免训练的解码机制，无需重新训练、微调或访问外部知识库。具体来说，DCLA通过聚合前几层的表示来构建动态语义参考，并用其纠正语义偏差的层，从而强制实现层间一致性。在七个LVLMs和多个基准上的实验证明了DCLA的通用性：它在MME指标上比标准解码高出28.58分。

    arXiv:2505.12343v2 Announce Type: replace-cross  Abstract: Despite the impressive capabilities of Large Vision-Language Models (LVLMs), they remain susceptible to hallucinations, where generated content is inconsistent with the input image. Existing training-free hallucination mitigation methods often suffer from unstable performance and high sensitivity to hyperparameter settings, which limits their practicality and broader adoption. In this paper, we propose Decoding with Inter-layer Consistency via Layer Aggregation (DCLA), a training-free decoding mechanism that requires no retraining, fine-tuning, or access to external knowledge bases. Specifically, DCLA constructs a dynamic semantic reference by aggregating representations from previous layers and uses it to correct semantically deviated layers, thereby enforcing inter-layer consistency. Experiments across seven LVLMs and multiple benchmarks demonstrate the generality of DCLA: it surpasses standard decoding by 28.58 MME points on
    
[^12]: SSM遇上视频扩散模型: 结构化状态空间下的高效视频生成

    SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces

    [https://arxiv.org/abs/2403.07711](https://arxiv.org/abs/2403.07711)

    提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题

    

    鉴于图像生成通过扩散模型取得的显著成就，研究界对将这些模型扩展到视频生成表现出越来越大的兴趣。最近用于视频生成的扩散模型主要利用注意力层来提取时间特征。然而，由于注意力层的内存消耗随着序列长度的增加呈二次增长，这种限制在尝试使用扩散模型生成更长视频序列时会带来重大挑战。为了克服这一挑战，我们提出利用状态空间模型（SSMs）。由于相对于序列长度，SSMs具有线性内存消耗，最近已经引起了越来越多的关注。在实验中，我们首先通过使用UCF101这一视频生成的标准基准来评估我们基于SSM的模型。此外，为探讨SSMs在更长视频生成中的潜力，

    arXiv:2403.07711v1 Announce Type: cross  Abstract: Given the remarkable achievements in image generation through diffusion models, the research community has shown increasing interest in extending these models to video generation. Recent diffusion models for video generation have predominantly utilized attention layers to extract temporal features. However, attention layers are limited by their memory consumption, which increases quadratically with the length of the sequence. This limitation presents significant challenges when attempting to generate longer video sequences using diffusion models. To overcome this challenge, we propose leveraging state-space models (SSMs). SSMs have recently gained attention as viable alternatives due to their linear memory consumption relative to sequence length. In the experiments, we first evaluate our SSM-based model with UCF101, a standard benchmark of video generation. In addition, to investigate the potential of SSMs for longer video generation, 
    
[^13]: 为AI推理建模人类价值观

    Modelling Human Values for AI Reasoning

    [https://arxiv.org/abs/2402.06359](https://arxiv.org/abs/2402.06359)

    本研究详细介绍了一个关于人类价值观的形式化模型，并展示了它在AI推理中的应用。研究通过基于社会心理学研究的关键思想，为AI系统与人类价值观的一致性提供了具体的计算表示。

    

    当今最重要的社会挑战之一是构建其行为与人类价值观一致的AI系统，或是其使人工和人工之间相互作用的社区行为与人类价值观一致。为了解决这一挑战，我们详细介绍了一个关于人类价值观的形式化模型，以进行其明确的计算表示。据我们所知，目前尚未有人尝试过这种模型，这在考虑到将价值观与AI整合的研究数量不断增长的情况下是令人惊讶的。我们以社会心理学领域近几十年来研究人类价值观性质的大量研究为起点，致力于提供这样一个形式化模型。我们展示了这个模型如何为基于AI的价值推理提供基础装置，并证明了它在实际应用案例中的适用性。我们阐述了我们的模型如何捕捉到社会心理学研究的关键思想，并提出了未来关于人类价值观在AI中集成和跨学科研究的路线图。

    One of today's most significant societal challenges is building AI systems whose behaviour, or the behaviour it enables within communities of interacting agents (human and artificial), aligns with human values. To address this challenge, we detail a formal model of human values for their explicit computational representation. To our knowledge, this has not been attempted as yet, which is surprising given the growing volume of research integrating values within AI. Taking as our starting point the wealth of research investigating the nature of human values from social psychology over the last few decades, we set out to provide such a formal model. We show how this model can provide the foundational apparatus for AI-based reasoning over values, and demonstrate its applicability in real-world use cases. We illustrate how our model captures the key ideas from social psychology research and propose a roadmap for future integrated, and interdisciplinary, research into human values in AI. The
    
[^14]: 带有赞助产品的品类规划

    Assortment Planning with Sponsored Products

    [https://arxiv.org/abs/2402.06158](https://arxiv.org/abs/2402.06158)

    本研究主要关注零售中带有赞助产品的品类规划挑战并将其建模为组合优化任务，以实现在考虑赞助产品的情况下优化预期收入的目的。

    

    在零售行业快速发展的背景下，品类规划对于企业的成功起着至关重要的作用。随着赞助产品在在线市场的日益突出地位，零售商在有效管理产品品类方面面临新的挑战。值得注意的是，以前的品类规划研究大多忽视了赞助产品的存在及其对整体推荐效果可能产生的影响。相反，他们通常简化地假设所有产品都是有机产品或非赞助产品。这个研究空白突显了在赞助产品存在的情况下更深入探讨品类规划挑战的必要性。我们将在存在赞助产品的情况下将品类规划问题建模为组合优化任务。最终目标是计算出一种最优的品类规划方案，既能优化预期收入，又能考虑到赞助产品的存在。

    In the rapidly evolving landscape of retail, assortment planning plays a crucial role in determining the success of a business. With the rise of sponsored products and their increasing prominence in online marketplaces, retailers face new challenges in effectively managing their product assortment in the presence of sponsored products. Remarkably, previous research in assortment planning largely overlooks the existence of sponsored products and their potential impact on overall recommendation effectiveness. Instead, they commonly make the simplifying assumption that all products are either organic or non-sponsored. This research gap underscores the necessity for a more thorough investigation of the assortment planning challenge when sponsored products are in play. We formulate the assortment planning problem in the presence of sponsored products as a combinatorial optimization task. The ultimate objective is to compute an assortment plan that optimizes expected revenue while considerin
    
[^15]: 颗粒球计算：一种高效、鲁棒和可解释的自适应多粒度表示和计算方法

    Granular ball computing: an efficient, robust, and interpretable adaptive multi-granularity representation and computation method. (arXiv:2304.11171v1 [cs.LG])

    [http://arxiv.org/abs/2304.11171](http://arxiv.org/abs/2304.11171)

    本文提出了一种基于颗粒球计算的自适应多粒度表示和计算方法，能够提高机器学习的效率、鲁棒性和可解释性。

    

    人类认知具有“先大后小”的认知机制，因此具有自适应的多粒度描述能力。这导致了有效性、鲁棒性和可解释性等计算特性。本文提出了一种新的基于颗粒球计算的自适应多粒度表示和计算方法。他们将这种方法应用于几个机器学习任务，并证明其相对于其他最先进的方法的有效性。

    Human cognition has a ``large-scale first'' cognitive mechanism, therefore possesses adaptive multi-granularity description capabilities. This results in computational characteristics such as efficiency, robustness, and interpretability. Although most existing artificial intelligence learning methods have certain multi-granularity features, they do not fully align with the ``large-scale first'' cognitive mechanism. Multi-granularity granular-ball computing is an important model method developed in recent years. This method can use granular-balls of different sizes to adaptively represent and cover the sample space, and perform learning based on granular-balls. Since the number of coarse-grained "granular-ball" is smaller than the number of sample points, granular-ball computing is more efficient; the coarse-grained characteristics of granular-balls are less likely to be affected by fine-grained sample points, making them more robust; the multi-granularity structure of granular-balls ca
    

