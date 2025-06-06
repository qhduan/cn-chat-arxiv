# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VCD: Knowledge Base Guided Visual Commonsense Discovery in Images](https://arxiv.org/abs/2402.17213) | 该论文提出了基于知识库的图像视觉常识发现（VCD）方法，通过定义细粒度的视觉常识类型以及构建包括超过10万张图像和1400万个对象-常识对的数据集，旨在提升计算机视觉系统的推理和决策能力。 |
| [^2] | [Representation Learning Using a Single Forward Pass](https://arxiv.org/abs/2402.09769) | 我们提出了一种神经科学启发的算法，可以通过单次前向传递进行表示学习。该算法具有独特的特点，并在不需要反向传播的情况下取得了高性能的分类结果。 |
| [^3] | [A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems](https://arxiv.org/abs/2402.09448) | 比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。 |
| [^4] | [Sequential Ordering in Textual Descriptions: Impact on Spatial Perception Abilities of Large Language Models](https://arxiv.org/abs/2402.07140) | 这项研究揭示了图描述的文本顺序对大语言模型在图推理中的性能产生显著影响，并通过改变文本顺序提高了大语言模型的性能。此外，发现大语言模型的推理性能与图大小之间的关系不是单调递减的。为了评估大语言模型在不同图大小上的性能，引入了规模化图推理基准。 |
| [^5] | [Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning.](http://arxiv.org/abs/2401.15098) | Hi-Core提出了一种新的框架，通过层次化的知识迁移来增强连续强化学习。该框架包括利用大型语言模型的推理能力设定目标的高层策略制定和通过强化学习按照高层目标导向的低层策略学习。在实验中，Hi-Core展现了较强的知识迁移能力。 |
| [^6] | [From User Surveys to Telemetry-Driven Agents: Exploring the Potential of Personalized Productivity Solutions.](http://arxiv.org/abs/2401.08960) | 本研究提出了一个以用户为中心的方法，以了解AI基于的生产力代理的偏好，并开发出个性化解决方案。通过调查和使用遥测数据，我们开发了一个GPT-4驱动的个性化生产力代理，并在研究中与其他辅助工具进行了比较。我们的研究突出了用户中心设计、适应性和个性化与隐私之间的平衡的重要性。 |
| [^7] | [Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications.](http://arxiv.org/abs/2401.04003) | 该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。 |
| [^8] | [Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings.](http://arxiv.org/abs/2310.17451) | 这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。 |
| [^9] | [Maximizing Seaweed Growth on Autonomous Farms: A Dynamic Programming Approach for Underactuated Systems Navigating on Uncertain Ocean Currents.](http://arxiv.org/abs/2307.01916) | 设计了一种基于动态规划的方法，用于在不确定的海洋洋流中最大化海藻生长，通过利用非线性时变的洋流实现高生长区域的探测。 |

# 详细

[^1]: 基于知识库的图像视觉常识发现（VCD）

    VCD: Knowledge Base Guided Visual Commonsense Discovery in Images

    [https://arxiv.org/abs/2402.17213](https://arxiv.org/abs/2402.17213)

    该论文提出了基于知识库的图像视觉常识发现（VCD）方法，通过定义细粒度的视觉常识类型以及构建包括超过10万张图像和1400万个对象-常识对的数据集，旨在提升计算机视觉系统的推理和决策能力。

    

    图像中的视觉常识包含有关对象属性、关系和行为的知识。发现视觉常识可以提供对图像的更全面和丰富的理解，并增强计算机视觉系统的推理和决策能力。然而，现有的视觉常识发现研究中所定义的视觉常识是粗粒度且不完整的。在这项工作中，我们从自然语言处理中的常识知识库ConceptNet中汲取灵感，并系统地定义了各种类型的视觉常识。基于此，我们引入了一个新任务，即视觉常识发现（VCD），旨在提取图像中不同对象所包含的不同类型的细粒度常识。因此，我们从Visual Genome和ConceptNet中构建了一个名为VCDD的数据集，包括超过10万张图像和1400万个对象-常识对。

    arXiv:2402.17213v1 Announce Type: cross  Abstract: Visual commonsense contains knowledge about object properties, relationships, and behaviors in visual data. Discovering visual commonsense can provide a more comprehensive and richer understanding of images, and enhance the reasoning and decision-making capabilities of computer vision systems. However, the visual commonsense defined in existing visual commonsense discovery studies is coarse-grained and incomplete. In this work, we draw inspiration from a commonsense knowledge base ConceptNet in natural language processing, and systematically define the types of visual commonsense. Based on this, we introduce a new task, Visual Commonsense Discovery (VCD), aiming to extract fine-grained commonsense of different types contained within different objects in the image. We accordingly construct a dataset (VCDD) from Visual Genome and ConceptNet for VCD, featuring over 100,000 images and 14 million object-commonsense pairs. We furthermore pro
    
[^2]: 使用单次前向传递的表示学习

    Representation Learning Using a Single Forward Pass

    [https://arxiv.org/abs/2402.09769](https://arxiv.org/abs/2402.09769)

    我们提出了一种神经科学启发的算法，可以通过单次前向传递进行表示学习。该算法具有独特的特点，并在不需要反向传播的情况下取得了高性能的分类结果。

    

    我们提出了一种受神经科学启发的单次传递嵌入学习算法（SPELA）。 SPELA是在边缘人工智能设备中进行训练和推理应用的首选候选人。 同时，SPELA可以最佳地满足对研究感知表示学习和形成框架的需求。 SPELA具有独特的特征，如嵌入向量形式的神经先验知识，不需要权重传输，不锁定权重更新，完全局部赫比安学习，不存储激活的单次前向传递和每个样本的单次权重更新。与传统方法相比，SPELA可以在不需要反向传播的情况下进行操作。 我们展示了我们的算法在一个有噪音的布尔运算数据集上可以执行非线性分类。 此外，我们展示了SPELA在MNIST，KMNIST和Fashion MNIST上的高性能表现。 最后，我们展示了SPELA在MNIST，KMNIST和Fashion MNIST上的少样本和1个时期学习能力。

    arXiv:2402.09769v1 Announce Type: new  Abstract: We propose a neuroscience-inspired Solo Pass Embedded Learning Algorithm (SPELA). SPELA is a prime candidate for training and inference applications in Edge AI devices. At the same time, SPELA can optimally cater to the need for a framework to study perceptual representation learning and formation. SPELA has distinctive features such as neural priors (in the form of embedded vectors), no weight transport, no update locking of weights, complete local Hebbian learning, single forward pass with no storage of activations, and single weight update per sample. Juxtaposed with traditional approaches, SPELA operates without the need for backpropagation. We show that our algorithm can perform nonlinear classification on a noisy boolean operation dataset. Additionally, we exhibit high performance using SPELA across MNIST, KMNIST, and Fashion MNIST. Lastly, we show the few-shot and 1-epoch learning capabilities of SPELA on MNIST, KMNIST, and Fashio
    
[^3]: 普通EEG与三极EEG在高性能到颤抓握BCI系统中的比较研究

    A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems

    [https://arxiv.org/abs/2402.09448](https://arxiv.org/abs/2402.09448)

    比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。

    

    本研究旨在比较传统EEG与三极EEG在提升运动障碍个体的BCI应用方面的有效性。重点是解读和解码各种抓握动作，如力握和精确握持。目标是确定哪种EEG技术在处理和翻译与抓握相关的脑电信号方面更为有效。研究涉及对十名健康参与者进行实验，参与者进行了两种不同的握持运动：力握和精确握持，无运动条件作为基线。我们的研究在解码抓握动作方面对EEG和三极EEG进行了全面比较。该比较涵盖了几个关键参数，包括信噪比（SNR）、通过功能连接的空间分辨率、ERPs和小波时频分析。此外，我们的研究还涉及从...

    arXiv:2402.09448v1 Announce Type: cross  Abstract: This study aims to enhance BCI applications for individuals with motor impairments by comparing the effectiveness of tripolar EEG (tEEG) with conventional EEG. The focus is on interpreting and decoding various grasping movements, such as power grasp and precision grasp. The goal is to determine which EEG technology is more effective in processing and translating grasp related neural signals. The approach involved experimenting on ten healthy participants who performed two distinct grasp movements: power grasp and precision grasp, with a no movement condition serving as the baseline. Our research presents a thorough comparison between EEG and tEEG in decoding grasping movements. This comparison spans several key parameters, including signal to noise ratio (SNR), spatial resolution via functional connectivity, ERPs, and wavelet time frequency analysis. Additionally, our study involved extracting and analyzing statistical features from th
    
[^4]: 文字描述中的顺序对大语言模型的空间感知能力的影响

    Sequential Ordering in Textual Descriptions: Impact on Spatial Perception Abilities of Large Language Models

    [https://arxiv.org/abs/2402.07140](https://arxiv.org/abs/2402.07140)

    这项研究揭示了图描述的文本顺序对大语言模型在图推理中的性能产生显著影响，并通过改变文本顺序提高了大语言模型的性能。此外，发现大语言模型的推理性能与图大小之间的关系不是单调递减的。为了评估大语言模型在不同图大小上的性能，引入了规模化图推理基准。

    

    最近几年，大语言模型在多个领域达到了最先进的性能。然而，图推理领域的进展仍然有限。我们的工作深入研究了大语言模型的图推理。在这项工作中，我们揭示了文本顺序对大语言模型空间理解的影响，发现图描述的文本顺序显著影响大语言模型对图的推理性能。通过改变图描述的文本顺序，我们将大语言模型的性能从42.22％提高到70％。此外，我们评估了大语言模型性能和图大小之间的关系，发现大语言模型的推理性能不随图大小的增加而单调递减。最后，我们引入了规模化图推理基准来评估大语言模型在不同图大小上的性能。

    In recent years, Large Language Models have reached state-of-the-art performance across multiple domains. However, the progress in the field of graph reasoning remains limited. Our work delves into this gap by thoroughly investigating graph reasoning with LLM. In this work, we reveal the impact of text sequence on LLM spatial understanding, finding that graph-descriptive text sequences significantly affect LLM reasoning performance on graphs. By altering the graph-descriptive text sequences, we enhance the performance of LLM from 42.22\% to 70\%. Furthermore, we evaluate the relationship between LLM performance and graph size, discovering that the reasoning performance of LLM does not monotonically decrease with the increase in graph size. Conclusively, we introduce the Scaled Graph Reasoning benchmark for assessing LLM performance across varied graph sizes.
    
[^5]: Hi-Core: 面向连续强化学习的层次化知识迁移

    Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning. (arXiv:2401.15098v1 [cs.LG])

    [http://arxiv.org/abs/2401.15098](http://arxiv.org/abs/2401.15098)

    Hi-Core提出了一种新的框架，通过层次化的知识迁移来增强连续强化学习。该框架包括利用大型语言模型的推理能力设定目标的高层策略制定和通过强化学习按照高层目标导向的低层策略学习。在实验中，Hi-Core展现了较强的知识迁移能力。

    

    连续强化学习（Continual Reinforcement Learning, CRL）赋予强化学习智能体从一系列任务中学习的能力，保留先前的知识并利用它来促进未来的学习。然而，现有的方法往往专注于在类似任务之间传输低层次的知识，忽视了人类认知控制的层次结构，导致在各种任务之间的知识迁移不足。为了增强高层次的知识迁移，我们提出了一种名为Hi-Core (Hierarchical knowledge transfer for Continual reinforcement learning)的新框架，它由两层结构组成：1) 利用大型语言模型（Large Language Model, LLM）的强大推理能力设定目标的高层策略制定和2) 通过强化学习按照高层目标导向的低层策略学习。此外，构建了一个知识库（策略库）来存储可以用于层次化知识迁移的策略。在MiniGr实验中进行了实验。

    Continual reinforcement learning (CRL) empowers RL agents with the ability to learn from a sequence of tasks, preserving previous knowledge and leveraging it to facilitate future learning. However, existing methods often focus on transferring low-level knowledge across similar tasks, which neglects the hierarchical structure of human cognitive control, resulting in insufficient knowledge transfer across diverse tasks. To enhance high-level knowledge transfer, we propose a novel framework named Hi-Core (Hierarchical knowledge transfer for Continual reinforcement learning), which is structured in two layers: 1) the high-level policy formulation which utilizes the powerful reasoning ability of the Large Language Model (LLM) to set goals and 2) the low-level policy learning through RL which is oriented by high-level goals. Moreover, the knowledge base (policy library) is constructed to store policies that can be retrieved for hierarchical knowledge transfer. Experiments conducted in MiniGr
    
[^6]: 从用户调查到遥测驱动代理：探索个性化的生产力解决方案的潜力

    From User Surveys to Telemetry-Driven Agents: Exploring the Potential of Personalized Productivity Solutions. (arXiv:2401.08960v1 [cs.HC])

    [http://arxiv.org/abs/2401.08960](http://arxiv.org/abs/2401.08960)

    本研究提出了一个以用户为中心的方法，以了解AI基于的生产力代理的偏好，并开发出个性化解决方案。通过调查和使用遥测数据，我们开发了一个GPT-4驱动的个性化生产力代理，并在研究中与其他辅助工具进行了比较。我们的研究突出了用户中心设计、适应性和个性化与隐私之间的平衡的重要性。

    

    我们提出了一个综合的以用户为中心的方法，用于了解基于人工智能的生产力代理的偏好，并开发出根据用户需求定制的个性化解决方案。通过两个阶段的方法，我们首先对363名参与者进行了调查，探索了生产力、沟通风格、代理方法、个性特征、个性化和隐私等各个方面。借助调查结果，我们开发了一个由Viva Insights收集的遥测数据驱动的个性化生产力代理，该代理利用GPT-4提供定制的帮助。我们在涉及40名参与者的研究中，将其性能与仪表板和叙述等替代的生产力辅助工具进行了比较。我们的研究结果凸显了用户中心设计、适应性以及个性化和隐私之间的平衡在AI辅助生产力工具中的重要性。通过借鉴我们研究中提炼的见解，我们相信我们的工作可以启发和指导未来的研究。

    We present a comprehensive, user-centric approach to understand preferences in AI-based productivity agents and develop personalized solutions tailored to users' needs. Utilizing a two-phase method, we first conducted a survey with 363 participants, exploring various aspects of productivity, communication style, agent approach, personality traits, personalization, and privacy. Drawing on the survey insights, we developed a GPT-4 powered personalized productivity agent that utilizes telemetry data gathered via Viva Insights from information workers to provide tailored assistance. We compared its performance with alternative productivity-assistive tools, such as dashboard and narrative, in a study involving 40 participants. Our findings highlight the importance of user-centric design, adaptability, and the balance between personalization and privacy in AI-assisted productivity tools. By building on the insights distilled from our study, we believe that our work can enable and guide futur
    
[^7]: 多机器人在层次化时间逻辑规范下的任务分配和规划

    Simultaneous Task Allocation and Planning for Multi-Robots under Hierarchical Temporal Logic Specifications. (arXiv:2401.04003v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2401.04003](http://arxiv.org/abs/2401.04003)

    该论文介绍了在多机器人系统中，利用层次化时间逻辑规范实现同时的任务分配和规划的方法。通过引入层次化结构到LTL规范中，该方法更具表达能力。采用基于搜索的方法来综合多机器人系统的计划，将搜索空间拆分为松散相互连接的子空间，以便更高效地进行任务分配和规划。

    

    过去关于机器人规划与时间逻辑规范的研究，特别是线性时间逻辑（LTL），主要是基于针对个体或群体机器人的单一公式。但随着任务复杂性的增加，LTL公式不可避免地变得冗长，使解释和规范生成变得复杂，同时还对规划器的计算能力造成压力。通过利用任务的内在结构，我们引入了一种层次化结构到具有语法和语义需求的LTL规范中，并证明它们比扁平规范更具表达能力。其次，我们采用基于搜索的方法来综合多机器人系统的计划，实现同时的任务分配和规划。搜索空间由松散相互连接的子空间近似表示，每个子空间对应一个LTL规范。搜索主要受限于单个子空间，根据特定条件转移到另一个子空间。

    Past research into robotic planning with temporal logic specifications, notably Linear Temporal Logic (LTL), was largely based on singular formulas for individual or groups of robots. But with increasing task complexity, LTL formulas unavoidably grow lengthy, complicating interpretation and specification generation, and straining the computational capacities of the planners. By leveraging the intrinsic structure of tasks, we introduced a hierarchical structure to LTL specifications with requirements on syntax and semantics, and proved that they are more expressive than their flat counterparts. Second, we employ a search-based approach to synthesize plans for a multi-robot system, accomplishing simultaneous task allocation and planning. The search space is approximated by loosely interconnected sub-spaces, with each sub-space corresponding to one LTL specification. The search is predominantly confined to a single sub-space, transitioning to another sub-space under certain conditions, de
    
[^8]: 通过理解生成：具有逻辑符号基础的神经视觉生成

    Generating by Understanding: Neural Visual Generation with Logical Symbol Groundings. (arXiv:2310.17451v1 [cs.AI])

    [http://arxiv.org/abs/2310.17451](http://arxiv.org/abs/2310.17451)

    这篇论文提出了一种神经符号学习方法，AbdGen，用于将知识推理系统与神经视觉生成模型集成。它解决了符号赋值和规则学习的问题，通过量化诱导方法实现可靠高效的符号赋值，通过对比元诱导方法实现精确的规则学习。

    

    尽管近年来神经视觉生成模型取得了很大的成功，但将其与强大的符号知识推理系统集成仍然是一个具有挑战性的任务。主要挑战有两个方面：一个是符号赋值，即将神经视觉生成器的潜在因素与知识推理系统中的有意义的符号进行绑定。另一个是规则学习，即学习新的规则，这些规则控制数据的生成过程，以增强知识推理系统。为了解决这些符号基础问题，我们提出了一种神经符号学习方法，Abductive Visual Generation (AbdGen)，用于基于诱导学习框架将逻辑编程系统与神经视觉生成模型集成起来。为了实现可靠高效的符号赋值，引入了量化诱导方法，通过语义编码本中的最近邻查找生成诱导提案。为了实现精确的规则学习，引入了对比元诱导方法。

    Despite the great success of neural visual generative models in recent years, integrating them with strong symbolic knowledge reasoning systems remains a challenging task. The main challenges are two-fold: one is symbol assignment, i.e. bonding latent factors of neural visual generators with meaningful symbols from knowledge reasoning systems. Another is rule learning, i.e. learning new rules, which govern the generative process of the data, to augment the knowledge reasoning systems. To deal with these symbol grounding problems, we propose a neural-symbolic learning approach, Abductive Visual Generation (AbdGen), for integrating logic programming systems with neural visual generative models based on the abductive learning framework. To achieve reliable and efficient symbol assignment, the quantized abduction method is introduced for generating abduction proposals by the nearest-neighbor lookups within semantic codebooks. To achieve precise rule learning, the contrastive meta-abduction
    
[^9]: 在不确定的海洋洋流中进行动力编程的无效系统上的自主农场上的海藻生长的最大化方法

    Maximizing Seaweed Growth on Autonomous Farms: A Dynamic Programming Approach for Underactuated Systems Navigating on Uncertain Ocean Currents. (arXiv:2307.01916v2 [eess.SY] UPDATED)

    [http://arxiv.org/abs/2307.01916](http://arxiv.org/abs/2307.01916)

    设计了一种基于动态规划的方法，用于在不确定的海洋洋流中最大化海藻生长，通过利用非线性时变的洋流实现高生长区域的探测。

    

    海藻生物量在气候减缓方面具有重要潜力，但需要大规模的自主开放式海洋农场来充分利用。这些农场通常具有低推进力，并受到海洋洋流的重大影响。我们希望设计一个控制器，通过利用非线性时变的海洋洋流来达到高生长区域，从而在几个月内最大化海藻生长。复杂的动力学和无效性使得即使知道洋流情况，这也是具有挑战性的。当只有短期不完善的预测且不确定性逐渐增大时，情况变得更加困难。我们提出了一种基于动态规划的方法，可以在已知真实洋流情况时有效地求解最优生长值函数。此外，我们还提出了三个扩展，即在现实中只知道预测的情况下：（1）我们方法得到的值函数可以作为反馈策略，以获得所有状态和时间的最佳生长控制，实现闭环控制的等价性。

    Seaweed biomass offers significant potential for climate mitigation, but large-scale, autonomous open-ocean farms are required to fully exploit it. Such farms typically have low propulsion and are heavily influenced by ocean currents. We want to design a controller that maximizes seaweed growth over months by taking advantage of the non-linear time-varying ocean currents for reaching high-growth regions. The complex dynamics and underactuation make this challenging even when the currents are known. This is even harder when only short-term imperfect forecasts with increasing uncertainty are available. We propose a dynamic programming-based method to efficiently solve for the optimal growth value function when true currents are known. We additionally present three extensions when as in reality only forecasts are known: (1) our methods resulting value function can be used as feedback policy to obtain the growth-optimal control for all states and times, allowing closed-loop control equival
    

