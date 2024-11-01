# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Natural Counterfactuals With Necessary Backtracking](https://rss.arxiv.org/abs/2402.01607) | 本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。 |
| [^2] | [When LLM-based Code Generation Meets the Software Development Process](https://arxiv.org/abs/2403.15852) | 该研究引入了基于LLM的代码生成框架LCG，通过模拟各种软件过程模型以及利用协作和技术提高代码质量。评估结果表明其在代码生成基准上的有效性。 |
| [^3] | [Policy Mirror Descent with Lookahead](https://arxiv.org/abs/2403.14156) | 提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。 |
| [^4] | [ERBench: An Entity-Relationship based Automatically Verifiable Hallucination Benchmark for Large Language Models](https://arxiv.org/abs/2403.05266) | ERBench是一个基于实体关系的大型语言模型幻觉基准，通过自动转换任何关系数据库并构建可自动验证的问题，以支持复杂性评估和调试 |
| [^5] | [Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level](https://arxiv.org/abs/2403.04690) | 该研究提出了一种更快的邻域注意力机制，通过将注意力限制在最近的邻居之间来降低自注意力的计算复杂度，实现了显著的性能提升。 |
| [^6] | [NeuralThink: Algorithm Synthesis that Extrapolates in General Tasks](https://arxiv.org/abs/2402.15393) | NeuralThink 是一种新的递归架构，可以一贯地对对称和不对称任务进行外推，相较于之前的最先进的深度思维架构在稳定地从较小的训练规模对大观测进行外推方面表现出更好的性能。 |
| [^7] | [SMX: Sequential Monte Carlo Planning for Expert Iteration](https://arxiv.org/abs/2402.07963) | 这项研究介绍了一种名为SMX的顺序蒙特卡洛规划算法，它利用可扩展的方法创建了有效的自我学习机制。它适用于离散和连续动作空间的环境，具有高并行性能。 |
| [^8] | [Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning](https://arxiv.org/abs/2402.06255) | 本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。 |
| [^9] | [DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning](https://arxiv.org/abs/2402.05421) | DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。 |
| [^10] | [Can Large Language Model Agents Simulate Human Trust Behaviors?](https://arxiv.org/abs/2402.04559) | 大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。 |
| [^11] | [Discovery of the Hidden World with Large Language Models](https://arxiv.org/abs/2402.03941) | 通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。 |
| [^12] | [Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate](https://arxiv.org/abs/2402.02769) | 通过从教学中学习（LoT）技术，我们提出了一种深度神经网络正则化技术，能够通过引入辅助学生模型来提升主模型的泛化性能。实验证明，LoT能有效识别具有泛化和可教授关系的信息。 |
| [^13] | [Factuality of Large Language Models in the Year 2024](https://arxiv.org/abs/2402.02420) | 本文调查了大规模语言模型（LLM）的真实性问题，并对其现有研究进行了批判性分析，指出了改进LLM真实性的挑战和解决方案，以及自动真实性评估的障碍。未来的研究应该关注在这些方面的进一步工作。 |
| [^14] | [Human-Readable Fingerprint for Large Language Models](https://arxiv.org/abs/2312.04828) | 这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。 |
| [^15] | [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) | 提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。 |
| [^16] | [Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning.](http://arxiv.org/abs/2401.10862) | 本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。 |
| [^17] | [End-to-end Learnable Clustering for Intent Learning in Recommendation.](http://arxiv.org/abs/2401.05975) | 本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。 |
| [^18] | [Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning.](http://arxiv.org/abs/2309.04459) | 通过将行动空间离散化并采用分词技术，我们提出了一种在稀疏奖励强化学习中生成技巧的新方法。这种方法能够减少探索的难度，并在连续行动空间中达到良好的性能。 |
| [^19] | [Invisible Image Watermarks Are Provably Removable Using Generative AI.](http://arxiv.org/abs/2306.01953) | 该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。 |
| [^20] | [Pre-trained transformer for adversarial purification.](http://arxiv.org/abs/2306.01762) | 本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。 |
| [^21] | [Augmented Modular Reinforcement Learning based on Heterogeneous Knowledge.](http://arxiv.org/abs/2306.01158) | 该论文提出了增强模块化强化学习（AMRL），使用仲裁器来选择异构模块，并无缝地整合不同类型的知识。该方法能够减缓强化学习中的一些低效问题，有望在深度强化学习领域得到应用。 |
| [^22] | [Temporally Layered Architecture for Efficient Continuous Control.](http://arxiv.org/abs/2305.18701) | 这项研究提出了一种时间分层架构，利用时间抽象实现了高效的连续控制，具有节能、持续探索、减少决策、减少抖动和增加动作重复等优势。 |
| [^23] | [GNNs,You can be Stronger,Deeper and Faster.](http://arxiv.org/abs/2305.05368) | 本文介绍了一种新的理解GNN表现能力的视角，并提出了一个新的采样节点级残差模块SDF，具有更好的表现能力。 |
| [^24] | [Knowledge Equivalence in Digital Twins of Intelligent Systems.](http://arxiv.org/abs/2204.07481) | 本文研究了智能系统数字孪生中的知识等价性，提出了在模型与物理系统之间同步知识的新技术。 |

# 详细

[^1]: 具有必要回溯的自然反事实

    Natural Counterfactuals With Necessary Backtracking

    [https://rss.arxiv.org/abs/2402.01607](https://rss.arxiv.org/abs/2402.01607)

    本研究提出了一种自然反事实框架和方法，通过优化控制回溯的范围，生成与实际世界的数据分布相匹配的自然反事实，从而改进了反事实推理。

    

    反事实推理对于人类认知非常重要，尤其对于提供解释和做出决策至关重要。尽管Judea Pearl的研究方法在理论上很优雅，但其生成反事实情景往往需要过于脱离实际情景的干预，因此难以实施。为了解决这个问题，我们提出了一种自然反事实的框架和一种根据实际世界数据分布生成自然反事实的方法。我们的方法提供了对反事实推理的改进，允许对因果前置变量进行改变以最小化与实际情景的偏差。为了生成自然反事实，我们引入了一种创新的优化框架，通过自然性准则允许但控制回溯的范围。实证实验表明了我们方法的有效性。

    Counterfactual reasoning is pivotal in human cognition and especially important for providing explanations and making decisions. While Judea Pearl's influential approach is theoretically elegant, its generation of a counterfactual scenario often requires interventions that are too detached from the real scenarios to be feasible. In response, we propose a framework of natural counterfactuals and a method for generating counterfactuals that are natural with respect to the actual world's data distribution. Our methodology refines counterfactual reasoning, allowing changes in causally preceding variables to minimize deviations from realistic scenarios. To generate natural counterfactuals, we introduce an innovative optimization framework that permits but controls the extent of backtracking with a naturalness criterion. Empirical experiments indicate the effectiveness of our method.
    
[^2]: 当基于LLM的代码生成遇上软件开发流程

    When LLM-based Code Generation Meets the Software Development Process

    [https://arxiv.org/abs/2403.15852](https://arxiv.org/abs/2403.15852)

    该研究引入了基于LLM的代码生成框架LCG，通过模拟各种软件过程模型以及利用协作和技术提高代码质量。评估结果表明其在代码生成基准上的有效性。

    

    软件过程模型在促进软件团队内协作与沟通，使其能够有效应对复杂的开发任务方面担当着关键角色。本文介绍了LCG，这是一个受到成熟软件工程实践启发的代码生成框架。LCG利用多个大型语言模型(LLM)代理来模拟各种软件过程模型，即LCGWaterfall、LCGTDD和LCGScrum。每个模型为LLM代理分配特定角色，如需求工程师、架构师、开发人员、测试人员和Scrum Master，反映了典型的开发活动和沟通模式。通过利用思维链和提示组合技术进行协作，代理不断完善自身以提高代码质量。在GPT3.5作为基础LLM和基准(GPT)的情况下，我们评估了LCG在四个代码生成基准测试上的表现：HumanEval、HumanEval-ET、MBPP和MBPP-ET。

    arXiv:2403.15852v1 Announce Type: cross  Abstract: Software process models play a pivotal role in fostering collaboration and communication within software teams, enabling them to tackle intricate development tasks effectively. This paper introduces LCG, a code generation framework inspired by established software engineering practices. LCG leverages multiple Large Language Model (LLM) agents to emulate various software process models, namely LCGWaterfall, LCGTDD, and LCGScrum. Each model assigns LLM agents specific roles such as requirement engineer, architect, developer, tester, and scrum master, mirroring typical development activities and communication patterns. Through collaborative efforts utilizing chain-of-thought and prompt composition techniques, the agents continuously refine themselves to enhance code quality. Utilizing GPT3.5 as the underlying LLM and baseline (GPT), we evaluate LCG across four code generation benchmarks: HumanEval, HumanEval-ET, MBPP, and MBPP-ET. Results
    
[^3]: 具有前瞻特性的策略镜像下降算法

    Policy Mirror Descent with Lookahead

    [https://arxiv.org/abs/2403.14156](https://arxiv.org/abs/2403.14156)

    提出了一种新类别的策略镜像下降算法$h$-PMD，它通过在PMD更新规则中结合多步贪心策略改进和前瞻深度$h，以解决折扣无限时间视角下的马尔可夫决策过程。

    

    策略镜像下降（PMD）作为一种多功能算法框架，包括几种重要的策略梯度算法，如自然策略梯度，并与最先进的强化学习（RL）算法（如TRPO和PPO）相联系。PMD可以看作是实现正则化1步贪心策略改进的软策略迭代算法。然而，1步贪心策略可能不是最佳选择，最近在RL领域取得了显着的实证成功，如AlphaGo和AlphaZero已经证明，相对于多步骤，贪心方法可以超越它们的1步骤对应物。在这项工作中，我们提出了一种新类别的PMD算法，称为$h$-PMD，它将具有前瞻深度$h$的多步贪心策略改进结合到PMD更新规则中。为了解决折扣无限时间视角下的马尔可夫决策过程，其中折扣因子为$\gamma$，我们展示了$h$-PMD可以推广标准的PMD。

    arXiv:2403.14156v1 Announce Type: cross  Abstract: Policy Mirror Descent (PMD) stands as a versatile algorithmic framework encompassing several seminal policy gradient algorithms such as natural policy gradient, with connections with state-of-the-art reinforcement learning (RL) algorithms such as TRPO and PPO. PMD can be seen as a soft Policy Iteration algorithm implementing regularized 1-step greedy policy improvement. However, 1-step greedy policies might not be the best choice and recent remarkable empirical successes in RL such as AlphaGo and AlphaZero have demonstrated that greedy approaches with respect to multiple steps outperform their 1-step counterpart. In this work, we propose a new class of PMD algorithms called $h$-PMD which incorporates multi-step greedy policy improvement with lookahead depth $h$ to the PMD update rule. To solve discounted infinite horizon Markov Decision Processes with discount factor $\gamma$, we show that $h$-PMD which generalizes the standard PMD enj
    
[^4]: ERBench：基于实体关系的可自动验证的大规模语言模型幻觉基准

    ERBench: An Entity-Relationship based Automatically Verifiable Hallucination Benchmark for Large Language Models

    [https://arxiv.org/abs/2403.05266](https://arxiv.org/abs/2403.05266)

    ERBench是一个基于实体关系的大型语言模型幻觉基准，通过自动转换任何关系数据库并构建可自动验证的问题，以支持复杂性评估和调试

    

    大型语言模型（LLMs）在各种应用中取得了前所未有的性能，然而它们的评估仍然是一个关键问题。现有的幻觉基准要么是静态的，要么缺乏可调整的复杂性进行彻底分析。我们认为利用现有的关系数据库是构建基准的一种有希望的方法，因为它们通过功能依赖关系可以准确描述知识。我们提出了ERBench，可以自动将任何关系数据库转换为基于实体关系（ER）模型的基准。我们的关键想法是使用数据库模式、记录和功能依赖来构建问题，以便可以自动验证。此外，我们使用外键约束来连接关系和构建多跳问题，这些问题可以任意复杂，用于调试LLMs的中间答案。最后，ERBench支持持续评估，多模态qu

    arXiv:2403.05266v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved unprecedented performance in various applications, yet their evaluation remains a critical issue. Existing hallucination benchmarks are either static or lack adjustable complexity for thorough analysis. We contend that utilizing existing relational databases is a promising approach for constructing benchmarks due to their accurate knowledge description via functional dependencies. We propose ERBench to automatically convert any relational database into a benchmark based on the entity-relationship (ER) model. Our key idea is to construct questions using the database schema, records, and functional dependencies such that they can be automatically verified. In addition, we use foreign key constraints to join relations and construct multihop questions, which can be arbitrarily complex and used to debug the intermediate answers of LLMs. Finally, ERBench supports continuous evaluation, multimodal qu
    
[^5]: 更快的邻域注意力: 在线程块级别减少自注意力的O(n^2)成本

    Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level

    [https://arxiv.org/abs/2403.04690](https://arxiv.org/abs/2403.04690)

    该研究提出了一种更快的邻域注意力机制，通过将注意力限制在最近的邻居之间来降低自注意力的计算复杂度，实现了显著的性能提升。

    

    邻域注意力通过限制每个标记的注意力范围为其最近的邻居来降低自注意力的成本。该限制由窗口大小和扩张因子参数化，介于线性投影和自注意力之间绘制了可能的注意力模式谱。邻域注意力，以及更一般地滑动窗口注意力模式，在基础设施方面长期受到限制，特别是在更高秩的空间（2-D和3-D），促使开发定制内核的发展，这些内核在功能或性能方面受限，如果不是两者都有。在这项工作中，我们首先展示邻域注意力可以表示为批量化的GEMM问题，类似于标准注意力，并为1-D和2-D邻域注意力实现它。与现有的简单内核相比，这些内核平均提供了分别是1-D和2-D邻域注意力的全精度延迟改进分别为895%和272%。

    arXiv:2403.04690v1 Announce Type: cross  Abstract: Neighborhood attention reduces the cost of self attention by restricting each token's attention span to its nearest neighbors. This restriction, parameterized by a window size and dilation factor, draws a spectrum of possible attention patterns between linear projection and self attention. Neighborhood attention, and more generally sliding window attention patterns, have long been bounded by infrastructure, particularly in higher-rank spaces (2-D and 3-D), calling for the development of custom kernels, which have been limited in either functionality, or performance, if not both. In this work, we first show that neighborhood attention can be represented as a batched GEMM problem, similar to standard attention, and implement it for 1-D and 2-D neighborhood attention. These kernels on average provide 895% and 272% improvement in full precision latency compared to existing naive kernels for 1-D and 2-D neighborhood attention respectively. 
    
[^6]: NeuralThink: 在一般任务中进行外推的算法综合

    NeuralThink: Algorithm Synthesis that Extrapolates in General Tasks

    [https://arxiv.org/abs/2402.15393](https://arxiv.org/abs/2402.15393)

    NeuralThink 是一种新的递归架构，可以一贯地对对称和不对称任务进行外推，相较于之前的最先进的深度思维架构在稳定地从较小的训练规模对大观测进行外推方面表现出更好的性能。

    

    虽然机器学习方法擅长模式识别，但在可扩展的算法方式上处理复杂的推理任务时仍然面临困难。最近的深度思维方法展现了学习可以外推的算法的潜力：在较小的环境中学习并在较大的环境中执行学到的算法。然而，这些工作局限于对称任务，即输入和输出的维度相同。为了填补这一空白，我们提出了 NeuralThink，一种新的递归架构，可以一贯地对对称和不对称任务进行外推，其中输入和输出的维度不同。我们提供了一个新颖的不对称任务外推基准。我们展示了 NeuralThink 在稳定地从较小的训练规模对大观测进行外推方面一直优于之前的最先进的深度思维架构。

    arXiv:2402.15393v1 Announce Type: cross  Abstract: While machine learning methods excel at pattern recognition, they struggle with complex reasoning tasks in a scalable, algorithmic manner. Recent Deep Thinking methods show promise in learning algorithms that extrapolate: learning in smaller environments and executing the learned algorithm in larger environments. However, these works are limited to symmetrical tasks, where the input and output dimensionalities are the same. To address this gap, we propose NeuralThink, a new recurrent architecture that can consistently extrapolate to both symmetrical and asymmetrical tasks, where the dimensionality of the input and output are different. We contribute with a novel benchmark of asymmetrical tasks for extrapolation. We show that NeuralThink consistently outperforms the prior state-of-the-art Deep Thinking architectures, in regards to stable extrapolation to large observations from smaller training sizes.
    
[^7]: SMX: 专家迭代的顺序蒙特卡洛规划

    SMX: Sequential Monte Carlo Planning for Expert Iteration

    [https://arxiv.org/abs/2402.07963](https://arxiv.org/abs/2402.07963)

    这项研究介绍了一种名为SMX的顺序蒙特卡洛规划算法，它利用可扩展的方法创建了有效的自我学习机制。它适用于离散和连续动作空间的环境，具有高并行性能。

    

    发展能够在决策和学习过程中利用规划能力的智能体对于人工智能的进步至关重要。最近的研究已经证明了树状搜索方法和自我对弈学习机制的有效性。然而，由于搜索过程的顺序性质，这些方法通常面临扩展性挑战。虽然实践工程解决方案可以部分克服这个问题，但仍需要大量计算资源，这限制了它们的适用性。在本文中，我们介绍一种名为SMX的基于模型的计划算法，它利用可扩展的顺序蒙特卡洛方法创建了一种有效的自我学习机制。SMX基于控制作为推断的理论框架，并受益于坚实的理论基础。它基于采样的搜索方法使其适应具有离散和连续动作空间的环境。此外，SMX允许高度并行化并可以运行于各类计算机设备上。

    Developing agents that can leverage planning abilities during their decision and learning processes is critical to the advancement of Artificial Intelligence. Recent works have demonstrated the effectiveness of combining tree-based search methods and self-play learning mechanisms. Yet, these methods typically face scaling challenges due to the sequential nature of their search. While practical engineering solutions can partly overcome this, they still demand extensive computational resources, which hinders their applicability. In this paper, we introduce SMX, a model-based planning algorithm that utilises scalable Sequential Monte Carlo methods to create an effective self-learning mechanism. Grounded in the theoretical framework of control as inference, SMX benefits from robust theoretical underpinnings. Its sampling-based search approach makes it adaptable to environments with both discrete and continuous action spaces. Furthermore, SMX allows for high parallelisation and can run on h
    
[^8]: 进取的鲍勃通过提示对抗调整抵制越狱行为

    Studious Bob Fight Back Against Jailbreaking via Prompt Adversarial Tuning

    [https://arxiv.org/abs/2402.06255](https://arxiv.org/abs/2402.06255)

    本文提出了一种名为Prompt Adversarial Tuning (PAT)的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中，实现对大型语言模型（LLMs）的越狱行为的防御。实验证明该方法在保护LLMs免受产生有害信息的影响方面效果显著。

    

    尽管大型语言模型（LLM）在各种应用中取得了巨大的成功，但它们也容易受到特定提示的影响，从而绕过内置的安全措施并提供危险或非法内容，这种现象被称为越狱行为。为了保护LLMs免受产生有害信息的影响，提出了各种防御策略，其中大多数集中在内容过滤或模型的对抗训练方面。在本文中，我们提出了一种名为Prompt Adversarial Tuning（PAT）的方法，通过训练一个防御控制机制并将其作为前缀嵌入到用户提示中来实现我们的防御策略。我们设计了一个类似对抗训练的训练过程，以实现我们的优化目标，交替更新攻击和防御控制机制。据我们所知，我们是第一个从提示调整的角度实施防御的人。一旦应用，我们的方法几乎不会影响LLMs的操作效率。实验表明我们的方法在抵御越狱行为方面具有良好的效果。

    Although Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to certain prompts that can induce them to bypass built-in safety measures and provide dangerous or illegal content, a phenomenon known as jailbreak. To protect LLMs from producing harmful information, various defense strategies are proposed, with most focusing on content filtering or adversarial training of models. In this paper, we propose an approach named Prompt Adversarial Tuning (PAT) to train a defense control mechanism, which is then embedded as a prefix to user prompts to implement our defense strategy. We design a training process similar to adversarial training to achieve our optimized goal, alternating between updating attack and defense controls. To our knowledge, we are the first to implement defense from the perspective of prompt tuning. Once employed, our method will hardly impact the operational efficiency of LLMs. Experiments show that our method i
    
[^9]: DiffTOP: 可微分轨迹优化在强化学习和模仿学习中的应用

    DiffTOP: Differentiable Trajectory Optimization for Deep Reinforcement and Imitation Learning

    [https://arxiv.org/abs/2402.05421](https://arxiv.org/abs/2402.05421)

    DiffTOP使用可微分轨迹优化作为策略表示来生成动作，解决了模型基于强化学习算法中的“目标不匹配”问题，并在模仿学习任务上进行了性能基准测试。

    

    本文介绍了DiffTOP，它利用可微分轨迹优化作为策略表示，为深度强化学习和模仿学习生成动作。轨迹优化是一种在控制领域中广泛使用的算法，由成本和动力学函数参数化。我们的方法的关键是利用了最近在可微分轨迹优化方面的进展，使得可以计算损失对于轨迹优化的参数的梯度。因此，轨迹优化的成本和动力学函数可以端到端地学习。DiffTOP解决了之前模型基于强化学习算法中的“目标不匹配”问题，因为DiffTOP中的动力学模型通过轨迹优化过程中的策略梯度损失直接最大化任务性能。我们还对DiffTOP在标准机器人操纵任务套件中进行了模仿学习性能基准测试。

    This paper introduces DiffTOP, which utilizes Differentiable Trajectory OPtimization as the policy representation to generate actions for deep reinforcement and imitation learning. Trajectory optimization is a powerful and widely used algorithm in control, parameterized by a cost and a dynamics function. The key to our approach is to leverage the recent progress in differentiable trajectory optimization, which enables computing the gradients of the loss with respect to the parameters of trajectory optimization. As a result, the cost and dynamics functions of trajectory optimization can be learned end-to-end. DiffTOP addresses the ``objective mismatch'' issue of prior model-based RL algorithms, as the dynamics model in DiffTOP is learned to directly maximize task performance by differentiating the policy gradient loss through the trajectory optimization process. We further benchmark DiffTOP for imitation learning on standard robotic manipulation task suites with high-dimensional sensory
    
[^10]: 大语言模型代理能够模拟人类的信任行为吗？

    Can Large Language Model Agents Simulate Human Trust Behaviors?

    [https://arxiv.org/abs/2402.04559](https://arxiv.org/abs/2402.04559)

    大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。

    

    大语言模型（LLM）代理已经越来越多地被采用作为模拟工具，用于模拟人类在社会科学等领域中的行为。然而，一个基本的问题仍然存在：LLM代理是否真的能够模拟人类行为？在本文中，我们专注于人类互动中最关键的行为之一，信任，旨在调查LLM代理是否能够模拟人类的信任行为。我们首先发现，在被行为经济学广泛接受的信任游戏框架下，LLM代理通常表现出信任行为，称为代理信任。然后，我们发现LLM代理在信任行为方面与人类具有较高的行为一致性，表明使用LLM代理模拟人类的信任行为是可行的。此外，我们还探索了代理信任中的偏见以及代理信任在对代理和人类之间的差异方面的内在特性。我们还探讨了包括高级推理策略在内的条件下代理信任的内在特性。

    Large Language Model (LLM) agents have been increasingly adopted as simulation tools to model humans in applications such as social science. However, one fundamental question remains: can LLM agents really simulate human behaviors? In this paper, we focus on one of the most critical behaviors in human interactions, trust, and aim to investigate whether or not LLM agents can simulate human trust behaviors. We first find that LLM agents generally exhibit trust behaviors, referred to as agent trust, under the framework of Trust Games, which are widely recognized in behavioral economics. Then, we discover that LLM agents can have high behavioral alignment with humans regarding trust behaviors, indicating the feasibility to simulate human trust behaviors with LLM agents. In addition, we probe into the biases in agent trust and the differences in agent trust towards agents and humans. We also explore the intrinsic properties of agent trust under conditions including advanced reasoning strate
    
[^11]: 用大型语言模型探索隐藏世界

    Discovery of the Hidden World with Large Language Models

    [https://arxiv.org/abs/2402.03941](https://arxiv.org/abs/2402.03941)

    通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。

    

    科学起源于从已知事实和观察中发现新的因果知识。传统的因果发现方法主要依赖于高质量的测量变量，通常由人类专家提供，以找到因果关系。然而，在许多现实世界的应用中，因果变量通常无法获取。大型语言模型（LLMs）的崛起为从原始观测数据中发现高级隐藏变量提供了新的机会。因此，我们介绍了COAT：因果表示助手。COAT将LLMs作为因素提供器引入，提取出来自非结构化数据的潜在因果因子。此外，LLMs还可以被指示提供用于收集数据值（例如注释标准）的额外信息，并将原始非结构化数据进一步解析为结构化数据。注释数据将被输入到...

    Science originates with discovering new causal knowledge from a combination of known facts and observations. Traditional causal discovery approaches mainly rely on high-quality measured variables, usually given by human experts, to find causal relations. However, the causal variables are usually unavailable in a wide range of real-world applications. The rise of large language models (LLMs) that are trained to learn rich knowledge from the massive observations of the world, provides a new opportunity to assist with discovering high-level hidden variables from the raw observational data. Therefore, we introduce COAT: Causal representatiOn AssistanT. COAT incorporates LLMs as a factor proposer that extracts the potential causal factors from unstructured data. Moreover, LLMs can also be instructed to provide additional information used to collect data values (e.g., annotation criteria) and to further parse the raw unstructured data into structured data. The annotated data will be fed to a
    
[^12]: 从教学中学习正则化: 易于模仿的可推广关系

    Learning from Teaching Regularization: Generalizable Correlations Should be Easy to Imitate

    [https://arxiv.org/abs/2402.02769](https://arxiv.org/abs/2402.02769)

    通过从教学中学习（LoT）技术，我们提出了一种深度神经网络正则化技术，能够通过引入辅助学生模型来提升主模型的泛化性能。实验证明，LoT能有效识别具有泛化和可教授关系的信息。

    

    泛化仍然是机器学习中的一个核心挑战。在这项工作中，我们提出了一种新颖的深度神经网络正则化技术，即从教学中学习（Learning from Teaching，简称LoT），以增强泛化性能。受到人类捕捉简明抽象模式的能力的启发，我们假设可推广的关系更容易教授。LoT通过辅助学生模型来实现这个概念，通过提供反馈来训练主模型和改进主模型，以捕捉更多具有泛化和可教授关系的信息。我们在计算机视觉、自然语言处理和强化学习等多个领域进行的实验结果表明，引入LoT相比仅在原始训练数据上训练模型带来了显著的好处。这表明了LoT在识别可推广信息方面的有效性。

    Generalization remains a central challenge in machine learning. In this work, we propose Learning from Teaching (LoT), a novel regularization technique for deep neural networks to enhance generalization. Inspired by the human ability to capture concise and abstract patterns, we hypothesize that generalizable correlations are expected to be easier to teach. LoT operationalizes this concept to improve the generalization of the main model with auxiliary student learners. The student learners are trained by the main model and improve the main model to capture more generalizable and teachable correlations by providing feedback. Our experimental results across several domains, including Computer Vision, Natural Language Processing, and Reinforcement Learning, demonstrate that the introduction of LoT brings significant benefits compared to merely training models on the original training data. It suggests the effectiveness of LoT in identifying generalizable information without falling into th
    
[^13]: 2024年大规模语言模型的真实性

    Factuality of Large Language Models in the Year 2024

    [https://arxiv.org/abs/2402.02420](https://arxiv.org/abs/2402.02420)

    本文调查了大规模语言模型（LLM）的真实性问题，并对其现有研究进行了批判性分析，指出了改进LLM真实性的挑战和解决方案，以及自动真实性评估的障碍。未来的研究应该关注在这些方面的进一步工作。

    

    大规模语言模型（LLMs），尤其是在聊天方面进行指导调整后，已经成为我们日常生活的一部分，通过在一个地方直接回答各种问题，使人们从搜索、提取和整合多个信息源的过程中得到解脱。然而，很多情况下，LLM的回答是错误的，这限制了它们在现实场景中的适用性。因此，对于评估和提高LLM真实性的研究近年来引起了很多关注。在这项调查中，我们对现有的研究进行了批判性分析，旨在找出主要挑战及其原因，并指出改进LLM真实性的潜在解决方案，以及分析开放文本生成的自动真实性评估面临的障碍。我们还展望了未来研究的方向。

    Large language models (LLMs), especially when instruction-tuned for chat, have become part of our daily lives, freeing people from the process of searching, extracting, and integrating information from multiple sources by offering a straightforward answer to a variety of questions in a single place. Unfortunately, in many cases, LLM responses are factually incorrect, which limits their applicability in real-world scenarios. As a result, research on evaluating and improving the factuality of LLMs has attracted a lot of research attention recently. In this survey, we critically analyze existing work with the aim to identify the major challenges and their associated causes, pointing out to potential solutions for improving the factuality of LLMs, and analyzing the obstacles to automated factuality evaluation for open-ended text generation. We further offer an outlook on where future research should go.
    
[^14]: 大型语言模型的人类可读指纹

    Human-Readable Fingerprint for Large Language Models

    [https://arxiv.org/abs/2312.04828](https://arxiv.org/abs/2312.04828)

    这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。

    

    由于大型语言模型（LLM）的资源密集型训练和配套的精心设计的许可证，保护LLM的版权变得至关重要。然而，由于可能的参数修改，确定LLM的原始基本模型是具有挑战性的。在本研究中，我们介绍了一种用于LLM的人类可读指纹，可以唯一地识别基本模型，而不暴露模型参数或干扰训练。我们首先观察到，在预训练期间模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括持续预训练、监督微调和RLHF，几乎没有扰动，这使得它成为识别基本模型的足够条件。通过继续训练LLM并添加一个额外的项来推开模型参数的方向，验证了这种必要性，结果使得模型受损。然而，这个方向容易受到简单攻击的影响，如维度...

    Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension 
    
[^15]: 攻击树：自动破解黑盒大型语言模型

    Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

    [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119)

    提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。

    

    大型语言模型(LLMs)展示了多功能性，但仍在生成有害、带偏见和有毒内容，这一点由人为设计的越狱行为的普遍存在得以证明。在这项工作中，我们提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成越狱，仅需要对目标LLM进行黑盒访问。TAP利用LLM来通过思维树推理迭代地优化候选（攻击）提示，直到生成的提示之一越狱目标。关键在于，在将提示发送给目标之前，TAP对其进行评估并移除可能不会导致越狱的提示。使用思维树推理使TAP能够在大量提示的搜索空间中导航，而修剪则减少了发送给目标的总查询数量。在实证评估中，我们观察到TAP生成的提示越狱了超过80%的最先进LLMs（包括GPT4和GPT4-Turbo）。

    arXiv:2312.02119v2 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80%
    
[^16]: 基于剪枝的保护: 在不进行微调的情况下增加对齐的LLMs的越狱抵抗力

    Pruning for Protection: Increasing Jailbreak Resistance in Aligned LLMs Without Fine-Tuning. (arXiv:2401.10862v1 [cs.LG])

    [http://arxiv.org/abs/2401.10862](http://arxiv.org/abs/2401.10862)

    本文研究了剪枝对齐的LLMs的保护措施，发现剪枝LLM参数可以显著增强其抵抗“越狱”提示攻击的能力，并且对其他LLM行为也可能有更普遍的效果。同时，引入了一个有害任务数据集，证明剪枝有助于集中注意力在与任务相关的标记上。突出的聊天模型表现出很高的易感性。

    

    大型语言模型（LLMs）容易受到“越狱”提示的攻击，这种攻击可以诱使这些模型生成有害和违法内容。本文表明，剪枝LLM参数多达20％可以显著增加它们对此类攻击的抵抗力，而无需额外训练并且不损害其在标准基准测试中的性能。有趣的是，我们发现剪枝后观察到的增强安全性与模型的初始安全训练水平相关，这暗示剪枝的效果可能更普遍，也可能适用于超出安全性范畴的其他LLM行为。另外，我们还介绍了一个包含五个类别、插入到十个不同越狱提示中的225个有害任务的精选数据集，表明剪枝有助于LLMs集中注意力在越狱提示中与任务相关的标记上。最后，我们的实验揭示了突出的聊天模型（如LLaMA-2 Chat，Vicuna和Mistral Instruct）具有很高的易感性。

    Large Language Models (LLMs) are vulnerable to `Jailbreaking' prompts, a type of attack that can coax these models into generating harmful and illegal content. In this paper, we show that pruning up to 20% of LLM parameters markedly increases their resistance to such attacks without additional training and without sacrificing their performance in standard benchmarks. Intriguingly, we discovered that the enhanced safety observed post-pruning correlates to the initial safety training level of the model, hinting that the effect of pruning could be more general and may hold for other LLM behaviors beyond safety. Additionally, we introduce a curated dataset of 225 harmful tasks across five categories, inserted into ten different Jailbreaking prompts, showing that pruning aids LLMs in concentrating attention on task-relevant tokens in jailbreaking prompts. Lastly, our experiments reveal that the prominent chat models, such as LLaMA-2 Chat, Vicuna, and Mistral Instruct exhibit high susceptibi
    
[^17]: 用于推荐中意图学习的端到端可学习聚类方法

    End-to-end Learnable Clustering for Intent Learning in Recommendation. (arXiv:2401.05975v1 [cs.IR])

    [http://arxiv.org/abs/2401.05975](http://arxiv.org/abs/2401.05975)

    本文提出了一种用于推荐中意图学习的端到端可学习聚类方法ELCRec，该方法解决了现有方法中的复杂优化问题和大规模数据集聚类的可扩展性问题。

    

    挖掘用户的意图在序列推荐中起着关键作用。最近的方法ICLRec使用对比学习和聚类来提取用户的潜在意图。尽管它已经显示出有效性，但现有的方法存在复杂和繁琐的交替优化问题，导致两个主要问题。首先，在广义期望最大化(EM)框架中分离表示学习和聚类优化经常导致次优性能。其次，在整个数据集上进行聚类会影响大规模行业数据的可扩展性。为了解决这些挑战，我们提出了一种新颖的意图学习方法，称为ELCRec，它将表示学习集成到一个端到端可学习聚类框架中进行推荐。

    Mining users' intents plays a crucial role in sequential recommendation. The recent approach, ICLRec, was introduced to extract underlying users' intents using contrastive learning and clustering. While it has shown effectiveness, the existing method suffers from complex and cumbersome alternating optimization, leading to two main issues. Firstly, the separation of representation learning and clustering optimization within a generalized expectation maximization (EM) framework often results in sub-optimal performance. Secondly, performing clustering on the entire dataset hampers scalability for large-scale industry data. To address these challenges, we propose a novel intent learning method called \underline{ELCRec}, which integrates representation learning into an \underline{E}nd-to-end \underline{L}earnable \underline{C}lustering framework for \underline{Rec}ommendation. Specifically, we encode users' behavior sequences and initialize the cluster centers as learnable network parameter
    
[^18]: 子词作为技巧：稀疏奖励强化学习的分词化

    Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning. (arXiv:2309.04459v1 [cs.LG])

    [http://arxiv.org/abs/2309.04459](http://arxiv.org/abs/2309.04459)

    通过将行动空间离散化并采用分词技术，我们提出了一种在稀疏奖励强化学习中生成技巧的新方法。这种方法能够减少探索的难度，并在连续行动空间中达到良好的性能。

    

    稀疏奖励强化学习中的探索具有困难，因为需要通过长期的、协调的行动序列才能获得任何奖励。而且，在连续的行动空间中，可能的行动数量是无穷多的，这只会增加探索的难度。为了解决这些问题，一类方法通过在同一领域收集的交互数据中形成时间上延伸的行动，通常称为技巧，并在这个新的行动空间上进行策略的优化。通常这样的方法在连续行动空间中需要一个漫长的预训练阶段，在强化学习开始之前形成技巧。鉴于先前的证据表明在这些任务中并不需要完整的连续行动空间，我们提出了一种新颖的技巧生成方法，包括两个组成部分。首先，我们通过聚类将行动空间离散化，然后我们利用从自然语言处理借鉴来的分词技术。

    Exploration in sparse-reward reinforcement learning is difficult due to the requirement of long, coordinated sequences of actions in order to achieve any reward. Moreover, in continuous action spaces there are an infinite number of possible actions, which only increases the difficulty of exploration. One class of methods designed to address these issues forms temporally extended actions, often called skills, from interaction data collected in the same domain, and optimizes a policy on top of this new action space. Typically such methods require a lengthy pretraining phase, especially in continuous action spaces, in order to form the skills before reinforcement learning can begin. Given prior evidence that the full range of the continuous action space is not required in such tasks, we propose a novel approach to skill-generation with two components. First we discretize the action space through clustering, and second we leverage a tokenization technique borrowed from natural language pro
    
[^19]: 通过生成式AI，证明了隐形图像水印是可清除的

    Invisible Image Watermarks Are Provably Removable Using Generative AI. (arXiv:2306.01953v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2306.01953](http://arxiv.org/abs/2306.01953)

    该论文证明了使用生成式AI可以清除隐形图像水印，提出了一种家族化再生攻击方法。通过形式化证明和实证结果，论文展示了所有隐形水印容易受到攻击，并针对一种具有弹性的水印RivaGAN，再生攻击可以去除93-99%的水印。

    

    隐形水印通过嵌入只有权利拥有者可以检测到的隐藏信息来保护图像的版权。它们还防止人们滥用由AI模型生成的图像。我们提出了一种家族化再生攻击来清除这些隐形水印。所提出的攻击方法首先向图像添加随机噪声来破坏水印，然后重建图像。这种方法灵活，可以与许多现有的图像降噪算法和预训练的生成模型（如扩散模型）实例化。通过形式化证明和实证结果，我们证明了所有隐形水印都容易受到所提出的攻击。对于一个特别有弹性的水印RivaGAN，再生攻击可以去除93-99%的隐形水印，而基线攻击只能去除不超过3%。然而，如果我们不要求带水印的图像与原始图像相同，保持图像语义相似的水印可能是一种替代方案。

    Invisible watermarks safeguard images' copyright by embedding hidden messages only detectable by owners. They also prevent people from misusing images, especially those generated by AI models. We propose a family of regeneration attacks to remove these invisible watermarks. The proposed attack method first adds random noise to an image to destroy the watermark and then reconstructs the image. This approach is flexible and can be instantiated with many existing image-denoising algorithms and pre-trained generative models such as diffusion models. Through formal proofs and empirical results, we show that all invisible watermarks are vulnerable to the proposed attack. For a particularly resilient watermark, RivaGAN, regeneration attacks remove 93-99% of the invisible watermarks while the baseline attacks remove no more than 3%. However, if we do not require the watermarked image to look the same as the original one, watermarks that keep the image semantically similar can be an alternative
    
[^20]: 预训练Transformer用于对抗性样本提纯

    Pre-trained transformer for adversarial purification. (arXiv:2306.01762v1 [cs.CR])

    [http://arxiv.org/abs/2306.01762](http://arxiv.org/abs/2306.01762)

    本文提出了一个快速防御对抗性攻击的方案RaPiD（Rapid Plug-in Defender），通过预训练的Transformer微调来提纯对抗样本，使其逼近清洁数据分布，实验结果表明，在有限数据情况下，该方法优于最先进的方法。

    

    随着越来越多的深度神经网络被部署为各种日常服务，它们的可靠性至关重要。深度神经网络容易受到对抗性攻击的影响，其中逃避攻击是最普遍的一种。最近的研究通常通过对抗训练或利用大量清洁数据的知识来增强其健壮性。然而，在实际应用中，重新训练和部署模型需要大量的计算资源，对在线服务造成重大损失。此外，当检测到某种攻击的对抗性例子时，服务提供者只能获得有限的对抗性样本，而大量的清洁数据可能无法获取。针对这些问题，我们提出了一种新的方案，名为RaPiD（Rapid Plug-in Defender），旨在快速防御具有少量干净和对抗性示例限制的原始服务模型的某种攻击。受到预训练模型提供转移学习良好初始化的通用趋势的启发，我们建议通过微调预先训练的Transformer来提纯对抗性样本。预训练的Transformer作为正则化器，鼓励提纯后的对抗性样本接近清晰数据的分布。实验结果表明，RaPiD在防御各种具有限数据的攻击方面优于最先进的方法。

    With more and more deep neural networks being deployed as various daily services, their reliability is essential. It's frightening that deep neural networks are vulnerable and sensitive to adversarial attacks, the most common one of which for the services is evasion-based. Recent works usually strengthen the robustness by adversarial training or leveraging the knowledge of an amount of clean data. However, in practical terms, retraining and redeploying the model need a large computational budget, leading to heavy losses to the online service. In addition, when adversarial examples of a certain attack are detected, only limited adversarial examples are available for the service provider, while much clean data may not be accessible. Given the mentioned problems, we propose a new scenario, RaPiD (Rapid Plug-in Defender), which is to rapidly defend against a certain attack for the frozen original service model with limitations of few clean and adversarial examples. Motivated by the general
    
[^21]: 基于异构知识的增强模块化强化学习

    Augmented Modular Reinforcement Learning based on Heterogeneous Knowledge. (arXiv:2306.01158v1 [cs.LG])

    [http://arxiv.org/abs/2306.01158](http://arxiv.org/abs/2306.01158)

    该论文提出了增强模块化强化学习（AMRL），使用仲裁器来选择异构模块，并无缝地整合不同类型的知识。该方法能够减缓强化学习中的一些低效问题，有望在深度强化学习领域得到应用。

    

    为了减缓强化学习中的一些低效问题，学者们提出了模块化方法，将不同的决策制定策略组合起来以衍生出可以执行多种任务的代理。这些体系结构的基础模块通常是可重复使用的，也允许“即插即用”的集成。然而，这些解决方案仍然缺乏处理和整合多种类型信息（知识）的能力，例如规则，子目标和技能。我们提出了增强模块化强化学习（AMRL）来解决这些限制。这种新的框架使用仲裁器来选择异构模块，并无缝地整合不同类型的知识。此外，我们引入了一种选择机制的变体，即增强记忆的仲裁器，它增加了利用时间信息的能力。我们在已有的环境中评估了所提出的机制，同时也在新环境中进行了基准测试，结果表明其在深度强化学习领域具有良好的应用前景。

    In order to mitigate some of the inefficiencies of Reinforcement Learning (RL), modular approaches composing different decision-making policies to derive agents capable of performing a variety of tasks have been proposed. The modules at the basis of these architectures are generally reusable, also allowing for "plug-and-play" integration. However, such solutions still lack the ability to process and integrate multiple types of information (knowledge), such as rules, sub-goals, and skills. We propose Augmented Modular Reinforcement Learning (AMRL) to address these limitations. This new framework uses an arbitrator to select heterogeneous modules and seamlessly incorporate different types of knowledge. Additionally, we introduce a variation of the selection mechanism, namely the Memory-Augmented Arbitrator, which adds the capability of exploiting temporal information. We evaluate the proposed mechanisms on established as well as new environments and benchmark them against prominent deep 
    
[^22]: 高效连续控制的时间分层架构

    Temporally Layered Architecture for Efficient Continuous Control. (arXiv:2305.18701v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.18701](http://arxiv.org/abs/2305.18701)

    这项研究提出了一种时间分层架构，利用时间抽象实现了高效的连续控制，具有节能、持续探索、减少决策、减少抖动和增加动作重复等优势。

    

    我们提出了一种时间分层架构（TLA），用于具有最小能量消耗的时间自适应控制。TLA将快速策略和慢速策略层叠在一起，实现时间抽象，使每个层可以专注于不同的时间尺度。我们的设计借鉴了人脑的节能机制，根据环境需求在不同的时间尺度上执行动作。我们证明了TLA除了节能之外，还提供了许多额外的优势，包括持续探索、减少需求决策、减少抖动和增加动作重复。我们在一系列连续控制任务上评估了我们的方法，并展示了TLA在多个重要指标上相对于现有方法的显著优势。我们还引入了多目标评分来定性评估连续控制策略，并展示了TLA具有显著更好的评分。我们的训练算法在慢速策略和

    We present a temporally layered architecture (TLA) for temporally adaptive control with minimal energy expenditure. The TLA layers a fast and a slow policy together to achieve temporal abstraction that allows each layer to focus on a different time scale. Our design draws on the energy-saving mechanism of the human brain, which executes actions at different timescales depending on the environment's demands. We demonstrate that beyond energy saving, TLA provides many additional advantages, including persistent exploration, fewer required decisions, reduced jerk, and increased action repetition. We evaluate our method on a suite of continuous control tasks and demonstrate the significant advantages of TLA over existing methods when measured over multiple important metrics. We also introduce a multi-objective score to qualitatively assess continuous control policies and demonstrate a significantly better score for TLA. Our training algorithm uses minimal communication between the slow and
    
[^23]: GNNs: 可以更强、更新、更快

    GNNs,You can be Stronger,Deeper and Faster. (arXiv:2305.05368v1 [cs.LG])

    [http://arxiv.org/abs/2305.05368](http://arxiv.org/abs/2305.05368)

    本文介绍了一种新的理解GNN表现能力的视角，并提出了一个新的采样节点级残差模块SDF，具有更好的表现能力。

    

    图神经网络（GNNs）是一类可以从图结构数据中学习并通过集成邻居节点的表示学习来表现出色的神经网络。然而，GNN的性能会随着层数增加而逐渐降低。本文引入了一个新的概念——k跳子图聚合，提出了一种新的理解GNN表现能力的视角，揭示了传统深层GNN表现逐渐退化的潜在原因，包括聚合子图的重叠以及基于残差的GNN实际上利用了1到k跳子图聚合结果来提高有效性。此外，我们提出了一种新的采样节点级残差模块SDF，通过理论推导证明其比之前的残差方法具有更优的表现能力，可以利用1到k跳跃子图的信息。

    Graph neural networks (GNNs), a type of neural network that can learn from graph-structured data and learn the representation of nodes by aggregating their neighbors, have shown excellent performance in downstream tasks.However, it is known that the performance of graph neural networks (GNNs) degrades gradually as the number of layers increases. Based on k-hop subgraph aggregation, which is a new concept, we propose a new perspective to understand the expressive power of GNN.From this perspective, we reveal the potential causes of the performance degradation of the deep traditional GNN - aggregated subgraph overlap, and the fact that the residual-based graph neural networks in fact exploit the aggregation results of 1 to k hop subgraphs to improve the effectiveness.Further, we propose a new sampling-based node-level residual module named SDF, which is shown by theoretical derivation to obtain a superior expressive power compared to previous residual methods by using information from 1 
    
[^24]: 智能系统数字孪生中的知识等价性

    Knowledge Equivalence in Digital Twins of Intelligent Systems. (arXiv:2204.07481v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2204.07481](http://arxiv.org/abs/2204.07481)

    本文研究了智能系统数字孪生中的知识等价性，提出了在模型与物理系统之间同步知识的新技术。

    

    数字孪生包含了对所研究的物理世界进行实时数据驱动建模的模型，并且可以使用模拟来优化物理世界。然而，数字孪生所做的分析只有在模型与实际物理世界等价的情况下才是有效和可靠的。在物理系统是智能和自主的情况下，保持这样一个等价模型具有挑战性。本文特别关注智能系统数字孪生模型，其中系统具有知识感知能力但能力有限。数字孪生通过在模拟环境中积累更多知识，从元层面上改进了物理系统的行为。在虚拟空间中复制这样一个智能物理系统的知识感知能力需要采用新颖的等价性维护技术，特别是在模型与物理系统之间同步知识。本文提出了知识等价性的概念。

    A digital twin contains up-to-date data-driven models of the physical world being studied and can use simulation to optimise the physical world. However, the analysis made by the digital twin is valid and reliable only when the model is equivalent to the physical world. Maintaining such an equivalent model is challenging, especially when the physical systems being modelled are intelligent and autonomous. The paper focuses in particular on digital twin models of intelligent systems where the systems are knowledge-aware but with limited capability. The digital twin improves the acting of the physical system at a meta-level by accumulating more knowledge in the simulated environment. The modelling of such an intelligent physical system requires replicating the knowledge-awareness capability in the virtual space. Novel equivalence maintaining techniques are needed, especially in synchronising the knowledge between the model and the physical system. This paper proposes the notion of knowled
    

