# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value](https://arxiv.org/abs/2404.01332) | 使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响 |
| [^2] | [Policy Optimization finds Nash Equilibrium in Regularized General-Sum LQ Games](https://arxiv.org/abs/2404.00045) | 引入相对熵正则化对一般和总 $N$-agent 游戏的纳什均衡产生影响，证明了NE符合线性高斯策略，并提出了政策优化算法以及增强技术来找到游戏内的NE。 |
| [^3] | [CoverUp: Coverage-Guided LLM-Based Test Generation](https://arxiv.org/abs/2403.16218) | CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。 |
| [^4] | [Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2403.11793) | 使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。 |
| [^5] | [IoTCO2: Assessing the End-To-End Carbon Footprint of Internet-of-Things-Enabled Deep Learning](https://arxiv.org/abs/2403.10984) | 介绍了一种名为\carb 的端到端建模工具，用于在物联网-启用深度学习中精确估算碳足迹，展示了与实际测量值相比最大$\pm21\%$的碳足迹差异。 |
| [^6] | [PERL: Parameter Efficient Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2403.10704) | 使用低秩适应（LoRA）方法进行参数高效强化学习（PERL），能够在与传统RLHF设置相当的性能下，实现更快的训练和更少的内存占用。 |
| [^7] | [Large Language Models and Games: A Survey and Roadmap](https://arxiv.org/abs/2402.18659) | 这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。 |
| [^8] | [A call for embodied AI](https://arxiv.org/abs/2402.03824) | 具象人工智能被提出作为追求人工通用智能的下一个基本步骤，并引入了一个基于认知架构的理论框架，与Friston的主动推断原则保持一致，为具象人工智能的发展提供了一个全面的方法。 |
| [^9] | [Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives](https://arxiv.org/abs/2402.01662) | 本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。 |
| [^10] | [Explainable Machine Learning for ICU Readmission Prediction.](http://arxiv.org/abs/2309.13781) | 本研究提出了一个标准化且可解释的机器学习流程，用于在多中心数据库中预测加护病房患者的再入院情况。 |
| [^11] | [RRWKV: Capturing Long-range Dependencies in RWKV.](http://arxiv.org/abs/2306.05176) | 本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。 |
| [^12] | [Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics.](http://arxiv.org/abs/2305.14286) | 本研究提出了一种称为EPNS的等变概率神经模拟框架，可以在系统演化中生成等变分布，并在随机时空动态方面表现出色。 |
| [^13] | [Where We Have Arrived in Proving the Emergence of Sparse Symbolic Concepts in AI Models.](http://arxiv.org/abs/2305.01939) | 证明了对于训练良好的AI模型，如果满足一定条件，将出现稀疏交互概念，这些概念能够描述输入变量之间的相互作用，并对模型推理分数产生影响。 |

# 详细

[^1]: 等等，这都是令牌噪音？一直就是吗：利用 Shapley 值解释 LLM 行为

    Wait, It's All Token Noise? Always Has Been: Interpreting LLM Behavior Using Shapley Value

    [https://arxiv.org/abs/2404.01332](https://arxiv.org/abs/2404.01332)

    使用Shapley值方法解释LLM行为，揭示了所谓的“令牌噪音”效应，揭示了LLMs的决策在很大程度上受到提示组件的影响

    

    大型语言模型（LLMs）的出现为模拟人类行为和认知过程开辟了新的可能性，潜在应用包括市场研究和消费者行为分析等各个领域。然而，由于LLMs的显著差异暗示了不同的基础过程在起作用，以及LLMs对提示变化的敏感性，利用LLMs作为人类主体的替代仍然存在不确定性。本文提出了一种基于合作博弈理论中Shapley值的新方法来解释LLM行为，并量化每个提示组件对模型输出的相对贡献。通过两个应用--一个离散选择实验和一个认知偏见调查，我们展示了Shapley值方法如何揭示我们所谓的“令牌噪音”效应，即LLM决策受到的影响严重偏向于

    arXiv:2404.01332v1 Announce Type: cross  Abstract: The emergence of large language models (LLMs) has opened up exciting possibilities for simulating human behavior and cognitive processes, with potential applications in various domains, including marketing research and consumer behavior analysis. However, the validity of utilizing LLMs as stand-ins for human subjects remains uncertain due to glaring divergences that suggest fundamentally different underlying processes at play and the sensitivity of LLM responses to prompt variations. This paper presents a novel approach based on Shapley values from cooperative game theory to interpret LLM behavior and quantify the relative contribution of each prompt component to the model's output. Through two applications-a discrete choice experiment and an investigation of cognitive biases-we demonstrate how the Shapley value method can uncover what we term "token noise" effects, a phenomenon where LLM decisions are disproportionately influenced by 
    
[^2]: 政策优化在正则化广义和总 LQ 游戏中找到纳什均衡

    Policy Optimization finds Nash Equilibrium in Regularized General-Sum LQ Games

    [https://arxiv.org/abs/2404.00045](https://arxiv.org/abs/2404.00045)

    引入相对熵正则化对一般和总 $N$-agent 游戏的纳什均衡产生影响，证明了NE符合线性高斯策略，并提出了政策优化算法以及增强技术来找到游戏内的NE。

    

    在本文中，我们研究了引入相对熵正则化对一般和总 $N$-agent 游戏的纳什均衡 (NE) 的影响，揭示了这类游戏的NE符合线性高斯策略的事实。此外，它描绘了在熵正则化的适当性方面，对游戏内NE独特性的充分条件。由于政策优化是强化学习 (RL) 技术的基础方法，旨在找到 NE，在这项工作中，我们证明了一个政策优化算法的线性收敛性，该算法 (在熵正则化的适当性下) 能够明显地实现 NE。此外，在熵正则化证明不足的情况下，我们提出了一个 $\delta$-增强技术，有助于实现游戏内的 $\epsilon$-NE。

    arXiv:2404.00045v1 Announce Type: cross  Abstract: In this paper, we investigate the impact of introducing relative entropy regularization on the Nash Equilibria (NE) of General-Sum $N$-agent games, revealing the fact that the NE of such games conform to linear Gaussian policies. Moreover, it delineates sufficient conditions, contingent upon the adequacy of entropy regularization, for the uniqueness of the NE within the game. As Policy Optimization serves as a foundational approach for Reinforcement Learning (RL) techniques aimed at finding the NE, in this work we prove the linear convergence of a policy optimization algorithm which (subject to the adequacy of entropy regularization) is capable of provably attaining the NE. Furthermore, in scenarios where the entropy regularization proves insufficient, we present a $\delta$-augmentation technique, which facilitates the achievement of an $\epsilon$-NE within the game.
    
[^3]: CoverUp：基于覆盖率引导的LLM测试生成系统

    CoverUp: Coverage-Guided LLM-Based Test Generation

    [https://arxiv.org/abs/2403.16218](https://arxiv.org/abs/2403.16218)

    CoverUp通过覆盖率分析和大型语言模型相结合的方式，驱动生成高覆盖率的Python回归测试，并在改进覆盖率方面取得显著成就。

    

    本文介绍了CoverUp，这是一个新型系统，通过覆盖率分析和大型语言模型（LLM）的结合驱动生成高覆盖率的Python回归测试。CoverUp通过迭代改善覆盖率，将覆盖率分析与LLM对话交替进行，以便将注意力集中在尚未涵盖的代码行和分支上。最终的测试套件相比当前技术水平显著提高了覆盖率：与CodaMosa相比，一种混合LLM / 基于搜索的软件测试系统，CoverUp在各方面都大幅提高了覆盖率。以模块为基础，CoverUp实现了81%的中位线覆盖率（对比62%）、53%的分支覆盖率（对比35%）和78%的线+分支覆盖率（对比55%）。我们展示了CoverUp的迭代、覆盖率引导方法对其有效性至关重要，为其成功的近一半作出了贡献。

    arXiv:2403.16218v1 Announce Type: cross  Abstract: This paper presents CoverUp, a novel system that drives the generation of high-coverage Python regression tests via a combination of coverage analysis and large-language models (LLMs). CoverUp iteratively improves coverage, interleaving coverage analysis with dialogs with the LLM to focus its attention on as yet uncovered lines and branches. The resulting test suites significantly improve coverage over the current state of the art: compared to CodaMosa, a hybrid LLM / search-based software testing system, CoverUp substantially improves coverage across the board. On a per-module basis, CoverUp achieves median line coverage of 81% (vs. 62%), branch coverage of 53% (vs. 35%) and line+branch coverage of 78% (vs. 55%). We show that CoverUp's iterative, coverage-guided approach is crucial to its effectiveness, contributing to nearly half of its successes.
    
[^4]: 大型语言模型的推理能力：对抽象和推理语料库的深入分析

    Reasoning Abilities of Large Language Models: In-Depth Analysis on the Abstraction and Reasoning Corpus

    [https://arxiv.org/abs/2403.11793](https://arxiv.org/abs/2403.11793)

    使用抽象和推理语料库（ARC）数据集评估大型语言模型的推理和上下文理解能力，结果显示虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后，实验结果有助于提出实现人类水平推理的发展路径。

    

    评估大型语言模型（LLMs）推理能力的现有方法以结果为中心，使得评估推理过程变得困难。我们引入了一种新方法，使用抽象和推理语料库（ARC）数据集以过程为中心的方式评估大型语言模型的推理和上下文理解能力。ARC要求解决问题时具有严谨的逻辑结构，这使得它成为一个能够促进模型推理能力与人类进行比较的基准。实验结果证实，虽然大型语言模型具有较弱的推理能力，但在逻辑连贯性、组合性和效率方面仍然落后。我们的实验突显了LLMs的推理能力，并提出了实现人类水平推理的发展路径。

    arXiv:2403.11793v1 Announce Type: cross  Abstract: The existing methods for evaluating the inference abilities of Large Language Models (LLMs) have been results-centric, making it difficult to assess the inference process. We introduce a new approach using the Abstract and Reasoning Corpus (ARC) dataset to evaluate the inference and contextual understanding abilities of large language models in a process-centric manner. ARC demands rigorous logical structures for problem-solving, making it a benchmark that facilitates the comparison of model inference abilities with humans. Experimental results confirm that while large language models possess weak inference abilities, they still lag in terms of logical coherence, compositionality, and productivity. Our experiments highlight the reasoning capabilities of LLMs, proposing development paths for achieving human-level reasoning.
    
[^5]: IoTCO2：评估物联网-启用深度学习的端到端碳足迹

    IoTCO2: Assessing the End-To-End Carbon Footprint of Internet-of-Things-Enabled Deep Learning

    [https://arxiv.org/abs/2403.10984](https://arxiv.org/abs/2403.10984)

    介绍了一种名为\carb 的端到端建模工具，用于在物联网-启用深度学习中精确估算碳足迹，展示了与实际测量值相比最大$\pm21\%$的碳足迹差异。

    

    为了提高隐私性和确保服务质量（QoS），深度学习（DL）模型越来越多地部署在物联网（IoT）设备上进行数据处理，极大地增加了与IoT上DL相关的碳足迹，涵盖了操作和实体方面。现有的操作能量预测器经常忽略了量化的DL模型和新兴的神经处理单元（NPUs），而实体碳足迹建模工具忽略了IoT设备中常见的非计算硬件组件，导致了物联网DL准确碳足迹建模工具的差距。本文介绍了\textit{\carb}，一种用于精确估算物联网DL中碳足迹的端到端建模工具，展示了与各种DL模型的实际测量值相比最大$\pm21\%$的碳足迹差异。此外，\carb的实际应用通过多个用户案例展示。

    arXiv:2403.10984v1 Announce Type: cross  Abstract: To improve privacy and ensure quality-of-service (QoS), deep learning (DL) models are increasingly deployed on Internet of Things (IoT) devices for data processing, significantly increasing the carbon footprint associated with DL on IoT, covering both operational and embodied aspects. Existing operational energy predictors often overlook quantized DL models and emerging neural processing units (NPUs), while embodied carbon footprint modeling tools neglect non-computing hardware components common in IoT devices, creating a gap in accurate carbon footprint modeling tools for IoT-enabled DL. This paper introduces \textit{\carb}, an end-to-end modeling tool for precise carbon footprint estimation in IoT-enabled DL, demonstrating a maximum $\pm21\%$ deviation in carbon footprint values compared to actual measurements across various DL models. Additionally, practical applications of \carb are showcased through multiple user case studies.
    
[^6]: PERL: 从人类反馈中实现参数高效强化学习

    PERL: Parameter Efficient Reinforcement Learning from Human Feedback

    [https://arxiv.org/abs/2403.10704](https://arxiv.org/abs/2403.10704)

    使用低秩适应（LoRA）方法进行参数高效强化学习（PERL），能够在与传统RLHF设置相当的性能下，实现更快的训练和更少的内存占用。

    

    强化学习从人类反馈（RLHF）已被证明是一种将预训练的大型语言模型（LLMs）与人类偏好对齐的有效方法。然而，使用RLHF训练模型计算成本高昂，且整个过程复杂。在本研究中，我们研究了RLHF，其中基础模型使用胡等人提出的低秩适应（LoRA）的参数高效方法进行训练。我们探讨了“参数高效强化学习”（PERL）的设置，在其中我们使用LoRA进行奖励模型训练和强化学习。我们将PERL与传统的微调（全调）在包括2个新数据集在内的7个基准测试中的奖励建模和强化学习方面的各种配置进行了比较。我们发现，PERL的性能与传统的RLHF设置相当，同时训练速度更快，内存占用更少。这使得RLHF具有很高的性能，同时减少了计算成本。

    arXiv:2403.10704v1 Announce Type: cross  Abstract: Reinforcement Learning from Human Feedback (RLHF) has proven to be a strong method to align Pretrained Large Language Models (LLMs) with human preferences. But training models with RLHF is computationally expensive, and an overall complex process. In this work, we study RLHF where the underlying models are trained using the parameter efficient method of Low-Rank Adaptation (LoRA) introduced by Hu et al. [2021]. We investigate the setup of "Parameter Efficient Reinforcement Learning" (PERL), in which we perform reward model training and reinforcement learning using LoRA. We compare PERL to conventional fine-tuning (full-tuning) across various configurations for 7 benchmarks, including 2 novel datasets, of reward modeling and reinforcement learning. We find that PERL performs on par with the conventional RLHF setting, while training faster, and with less memory. This enables the high performance of RLHF, while reducing the computational 
    
[^7]: 大型语言模型与游戏：调研与路线图

    Large Language Models and Games: A Survey and Roadmap

    [https://arxiv.org/abs/2402.18659](https://arxiv.org/abs/2402.18659)

    这项研究调查了大型语言模型在游戏领域中的多种应用及其角色，指出了未开发领域和未来发展方向，同时探讨了在游戏领域中大型语言模型的潜力和限制。

    

    近年来，大型语言模型（LLMs）的研究急剧增加，并伴随着公众对该主题的参与。尽管起初是自然语言处理中的一小部分，LLMs在广泛的应用和领域中展现出显著潜力，包括游戏。本文调查了LLMs在游戏中及为游戏提供支持的各种应用的最新技术水平，并明确了LLMs在游戏中可以扮演的不同角色。重要的是，我们讨论了尚未开发的领域和LLMs在游戏中未来应用的有前途的方向，以及在游戏领域中LLMs的潜力和限制。作为LLMs和游戏交叉领域的第一份综合调查和路线图，我们希望本文能够成为这一激动人心的新领域的开创性研究和创新的基础。

    arXiv:2402.18659v1 Announce Type: cross  Abstract: Recent years have seen an explosive increase in research on large language models (LLMs), and accompanying public engagement on the topic. While starting as a niche area within natural language processing, LLMs have shown remarkable potential across a broad range of applications and domains, including games. This paper surveys the current state of the art across the various applications of LLMs in and for games, and identifies the different roles LLMs can take within a game. Importantly, we discuss underexplored areas and promising directions for future uses of LLMs in games and we reconcile the potential and limitations of LLMs within the games domain. As the first comprehensive survey and roadmap at the intersection of LLMs and games, we are hopeful that this paper will serve as the basis for groundbreaking research and innovation in this exciting new field.
    
[^8]: 一种关于具象人工智能的呼吁

    A call for embodied AI

    [https://arxiv.org/abs/2402.03824](https://arxiv.org/abs/2402.03824)

    具象人工智能被提出作为追求人工通用智能的下一个基本步骤，并引入了一个基于认知架构的理论框架，与Friston的主动推断原则保持一致，为具象人工智能的发展提供了一个全面的方法。

    

    我们提出具象人工智能作为追求人工通用智能的下一个基本步骤，并将其与当前的人工智能进展进行对比，特别是大型语言模型。我们跨越了哲学、心理学、神经科学和机器人学等多个领域对具象概念的演变进行了研究，以凸显具象人工智能如何区别于静态学习的经典范式。通过扩大具象人工智能的范围，我们引入了一个基于认知架构的理论框架，强调知觉、行动、记忆和学习作为具象代理的基本组成部分。这个框架与Friston的主动推断原则保持一致，为具象人工智能的发展提供了一个全面的方法。尽管在人工智能领域取得了进展，但仍然存在诸如制定新的人工智能学习理论和创新先进硬件等重大挑战。我们的讨论为未来具象人工智能研究奠定了基础指导。

    We propose Embodied AI as the next fundamental step in the pursuit of Artificial General Intelligence, juxtaposing it against current AI advancements, particularly Large Language Models. We traverse the evolution of the embodiment concept across diverse fields - philosophy, psychology, neuroscience, and robotics - to highlight how EAI distinguishes itself from the classical paradigm of static learning. By broadening the scope of Embodied AI, we introduce a theoretical framework based on cognitive architectures, emphasizing perception, action, memory, and learning as essential components of an embodied agent. This framework is aligned with Friston's active inference principle, offering a comprehensive approach to EAI development. Despite the progress made in the field of AI, substantial challenges, such as the formulation of a novel AI learning theory and the innovation of advanced hardware, persist. Our discussion lays down a foundational guideline for future Embodied AI research. High
    
[^9]: 生成幽灵：预测人工智能来世的益处和风险

    Generative Ghosts: Anticipating Benefits and Risks of AI Afterlives

    [https://arxiv.org/abs/2402.01662](https://arxiv.org/abs/2402.01662)

    本文讨论了生成幽灵的潜在实施设计空间和其对个人和社会的实际和伦理影响，提出了研究议程以便使人们能够安全而有益地创建和与人工智能来世进行互动。

    

    随着人工智能系统在性能的广度和深度上迅速提升，它们越来越适合创建功能强大、逼真的代理人，包括基于特定人物建模的代理人的可能性。我们预计，在我们有生之年，人们可能会普遍使用定制的人工智能代理人与爱的人和/或更广大的世界进行互动。我们称之为生成幽灵，因为这些代理人将能够生成新颖的内容，而不只是复述其创作者在生前的内容。在本文中，我们首先讨论了生成幽灵潜在实施的设计空间。然后，我们讨论了生成幽灵的实际和伦理影响，包括对个人和社会的潜在积极和消极影响。基于这些考虑，我们制定了一个研究议程，旨在使人们能够安全而有益地创建和与人工智能来世进行互动。

    As AI systems quickly improve in both breadth and depth of performance, they lend themselves to creating increasingly powerful and realistic agents, including the possibility of agents modeled on specific people. We anticipate that within our lifetimes it may become common practice for people to create a custom AI agent to interact with loved ones and/or the broader world after death. We call these generative ghosts, since such agents will be capable of generating novel content rather than merely parroting content produced by their creator while living. In this paper, we first discuss the design space of potential implementations of generative ghosts. We then discuss the practical and ethical implications of generative ghosts, including potential positive and negative impacts on individuals and society. Based on these considerations, we lay out a research agenda for the AI and HCI research communities to empower people to create and interact with AI afterlives in a safe and beneficial 
    
[^10]: ICU 重新入院预测的可解释机器学习

    Explainable Machine Learning for ICU Readmission Prediction. (arXiv:2309.13781v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.13781](http://arxiv.org/abs/2309.13781)

    本研究提出了一个标准化且可解释的机器学习流程，用于在多中心数据库中预测加护病房患者的再入院情况。

    

    加护病房（ICU）是一个复杂的医院环境，医生的决策对患者的生命构成高风险。必须遵循一条全面的护理路径来减少并发症。在这种环境中，不确定性、竞争性和非计划性的因素增加了统一实施护理路径的困难。再入院是该路径的困难之一，即患者在短时间内再次入住ICU，导致高死亡率和高资源利用率。一些研究尝试通过患者的医疗信息来预测再入院情况。尽管它们在预测再入院时有一定的成功，但这些研究并未对再入院预测进行适当的评估、描述和理解。本研究提出了一个标准化且可解释的机器学习流程，用于在多中心数据库（即包含166,355名患者的eICU队列，200,859名...）

    The intensive care unit (ICU) comprises a complex hospital environment, where decisions made by clinicians have a high level of risk for the patients' lives. A comprehensive care pathway must then be followed to reduce p complications. Uncertain, competing and unplanned aspects within this environment increase the difficulty in uniformly implementing the care pathway. Readmission contributes to this pathway's difficulty, occurring when patients are admitted again to the ICU in a short timeframe, resulting in high mortality rates and high resource utilisation. Several works have tried to predict readmission through patients' medical information. Although they have some level of success while predicting readmission, those works do not properly assess, characterise and understand readmission prediction. This work proposes a standardised and explainable machine learning pipeline to model patient readmission on a multicentric database (i.e., the eICU cohort with 166,355 patients, 200,859 ad
    
[^11]: RRWKV：在RWKV中捕捉长距离依赖关系

    RRWKV: Capturing Long-range Dependencies in RWKV. (arXiv:2306.05176v1 [cs.CL])

    [http://arxiv.org/abs/2306.05176](http://arxiv.org/abs/2306.05176)

    本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。

    

    由于Transformer惊人的点积注意力，它已经成为各种自然语言处理（NLP）任务中的主要架构。最近，Receptance Weighted Key Value（RWKV）架构遵循非Transformer架构，消除了点积注意力的缺点，其中存储和计算复杂度随着序列长度呈二次扩展。尽管RWKV利用了线性张量积注意机制并通过部署时间序列模式实现了并行计算，但与标准Transformer中直接交互获得的完整信息相比，它无法捕捉长距离依赖关系，因为其受限于向后查看先前信息的能力。因此，本文通过将回顾能力纳入RWKV中来设计Retrospected Receptance Weighted Key Value（RRWKV）架构，以有效地吸收信息，同时保持记忆和计算效率。

    Owing to the impressive dot-product attention, the Transformers have been the dominant architectures in various natural language processing (NLP) tasks. Recently, the Receptance Weighted Key Value (RWKV) architecture follows a non-transformer architecture to eliminate the drawbacks of dot-product attention, where memory and computational complexity exhibits quadratic scaling with sequence length. Although RWKV has exploited a linearly tensor-product attention mechanism and achieved parallelized computations by deploying the time-sequential mode, it fails to capture long-range dependencies because of its limitation on looking back at previous information, compared with full information obtained by direct interactions in the standard transformer. Therefore, the paper devises the Retrospected Receptance Weighted Key Value (RRWKV) architecture via incorporating the retrospecting ability into the RWKV to effectively absorb information, which maintains memory and computational efficiency as 
    
[^12]: 等变神经模拟器用于随机时空动态

    Equivariant Neural Simulators for Stochastic Spatiotemporal Dynamics. (arXiv:2305.14286v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.14286](http://arxiv.org/abs/2305.14286)

    本研究提出了一种称为EPNS的等变概率神经模拟框架，可以在系统演化中生成等变分布，并在随机时空动态方面表现出色。

    

    神经网络正在成为可扩展的数据驱动高维动态系统模拟工具，特别是在数值方法不可行或计算昂贵的情况下。值得注意的是，已经证明在确定性神经模拟器中引入域对称性可以大大提高其精确性、样本效率和参数效率。然而，为了将对称性纳入可以模拟随机现象的概率神经模拟器中，我们需要一个能够生成等变轨迹分布而不是等变函数逼近的模型。在本文中，我们提出了等变概率神经模拟（EPNS），这是一个用于等变分布系统演化的自回归概率建模框架。我们使用EPNS设计了一个用于随机N体系统和随机细胞动力学的模型。我们的结果表明，EPNS在p方面比现有的基于神经网络的方法表现出色。

    Neural networks are emerging as a tool for scalable data-driven simulation of high-dimensional dynamical systems, especially in settings where numerical methods are infeasible or computationally expensive. Notably, it has been shown that incorporating domain symmetries in deterministic neural simulators can substantially improve their accuracy, sample efficiency, and parameter efficiency. However, to incorporate symmetries in probabilistic neural simulators that can simulate stochastic phenomena, we need a model that produces equivariant distributions over trajectories, rather than equivariant function approximations. In this paper, we propose Equivariant Probabilistic Neural Simulation (EPNS), a framework for autoregressive probabilistic modeling of equivariant distributions over system evolutions. We use EPNS to design models for a stochastic n-body system and stochastic cellular dynamics. Our results show that EPNS considerably outperforms existing neural network-based methods for p
    
[^13]: 证明AI模型中稀疏符号概念的出现

    Where We Have Arrived in Proving the Emergence of Sparse Symbolic Concepts in AI Models. (arXiv:2305.01939v1 [cs.LG])

    [http://arxiv.org/abs/2305.01939](http://arxiv.org/abs/2305.01939)

    证明了对于训练良好的AI模型，如果满足一定条件，将出现稀疏交互概念，这些概念能够描述输入变量之间的相互作用，并对模型推理分数产生影响。

    

    本文旨在证明训练良好的AI模型中出现符号概念的现象。我们证明，如果（1）模型输出相对于输入变量的高阶导数均为零，（2）AI模型可用于遮挡样本且输入样本较少遮挡时会产生更高的置信度，（3）AI模型在遮挡样本上的置信度并不会显著降低，则AI模型将编码稀疏交互概念。每个交互概念表示特定一组输入变量之间的相互作用，并对模型推理分数产生一定的数值影响。具体而言，我们证明了模型的推理分数总是可以表示为所有交互概念的交互效应之和。事实上，我们希望证明出现符号概念的条件非常普遍。这意味着对于大多数AI模型，我们通常可以使用少量的交互概念来模拟模型。

    This paper aims to prove the emergence of symbolic concepts in well-trained AI models. We prove that if (1) the high-order derivatives of the model output w.r.t. the input variables are all zero, (2) the AI model can be used on occluded samples and will yield higher confidence when the input sample is less occluded, and (3) the confidence of the AI model does not significantly degrade on occluded samples, then the AI model will encode sparse interactive concepts. Each interactive concept represents an interaction between a specific set of input variables, and has a certain numerical effect on the inference score of the model. Specifically, it is proved that the inference score of the model can always be represented as the sum of the interaction effects of all interactive concepts. In fact, we hope to prove that conditions for the emergence of symbolic concepts are quite common. It means that for most AI models, we can usually use a small number of interactive concepts to mimic the mode
    

