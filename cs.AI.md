# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Few-Shot Cross-System Anomaly Trace Classification for Microservice-based systems](https://arxiv.org/abs/2403.18998) | 提出了针对微服务系统的少样本异常跟踪分类的新框架，利用多头注意力自编码器构建系统特定的跟踪表示，并应用基于Transformer编码器的模型无关元学习进行高效分类。 |
| [^2] | [As Good As A Coin Toss Human detection of AI-generated images, videos, audio, and audiovisual stimuli](https://arxiv.org/abs/2403.16760) | 通过一项感知研究，评估了人们在日常生活中对合成图像、音频、视频和音视频刺激与真实的区分能力，以探讨人类对欺骗性合成媒体的易受程度。 |
| [^3] | [ChatDBG: An AI-Powered Debugging Assistant](https://arxiv.org/abs/2403.16354) | ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。 |
| [^4] | [Sphere Neural-Networks for Rational Reasoning](https://arxiv.org/abs/2403.15297) | 该论文提出了一种球形神经网络（SphNNs）来进行理性推理，通过将计算构建块从向量推广到球体，实现了人类类似的推理能力，并开发了用于三段论推理的SphNN。 |
| [^5] | [The AI Assessment Scale (AIAS) in action: A pilot implementation of GenAI supported assessment](https://arxiv.org/abs/2403.14692) | 该论文介绍了AI评估量表（AIAS）的实践应用，通过灵活框架将GenAI技术纳入教育评估中，显著降低了与GenAI相关的学术不端案件，提高了学生的学业成绩。 |
| [^6] | [ASGEA: Exploiting Logic Rules from Align-Subgraphs for Entity Alignment](https://arxiv.org/abs/2402.11000) | 提出了一个新的实体对齐框架ASGEA，利用Align-Subgraphs中的逻辑规则，设计了可解释的基于路径的图神经网络ASGNN，引入了多模态注意机制，取得了令人满意的实验结果 |
| [^7] | [CURE: Simulation-Augmented Auto-Tuning in Robotics](https://arxiv.org/abs/2402.05399) | 本论文提出了一种模拟辅助的自动调节技术，用于解决机器人系统中的高度可配置参数的优化问题。该技术通过解决软硬件之间配置选项的交互问题，实现了在不同环境和机器人平台之间的性能迁移。 |
| [^8] | [D-STGCNT: A Dense Spatio-Temporal Graph Conv-GRU Network based on transformer for assessment of patient physical rehabilitation.](http://arxiv.org/abs/2401.06150) | D-STGCNT是一种新的模型，结合了STGCN和transformer的架构，用于自动评估患者身体康复锻炼。它通过将骨架数据视为图形，并检测关键关节，在处理时空数据方面具有高效性。该模型通过密集连接和GRU机制来处理大型3D骨架输入，有效建立时空动态模型。transformer的注意力机制对于评估康复锻炼非常有用。 |
| [^9] | [Reinforcing POD based model reduction techniques in reaction-diffusion complex networks using stochastic filtering and pattern recognition.](http://arxiv.org/abs/2307.09762) | 该论文提出了一种算法框架，通过将模式识别和随机滤波理论的技术结合起来，强化了基于POD的反应扩散复杂网络模型简化技术，在受扰动输入的情况下提高了代理模型的准确性。 |

# 详细

[^1]: 微服务系统的少样本跨系统异常跟踪分类

    Few-Shot Cross-System Anomaly Trace Classification for Microservice-based systems

    [https://arxiv.org/abs/2403.18998](https://arxiv.org/abs/2403.18998)

    提出了针对微服务系统的少样本异常跟踪分类的新框架，利用多头注意力自编码器构建系统特定的跟踪表示，并应用基于Transformer编码器的模型无关元学习进行高效分类。

    

    微服务系统（MSS）由于其复杂和动态的特性可能在各种故障类别中出现故障。为了有效处理故障，AIOps工具利用基于跟踪的异常检测和根本原因分析。本文提出了一个新颖的框架，用于微服务系统的少样本异常跟踪分类。我们的框架包括两个主要组成部分：（1）多头注意力自编码器用于构建系统特定的跟踪表示，从而实现（2）基于Transformer编码器的模型无关元学习，以进行有效和高效的少样本异常跟踪分类。该框架在两个代表性的MSS，Trainticket和OnlineBoutique上进行了评估，使用开放数据集。结果表明，我们的框架能够调整学到的知识，以对新的、未见的新颖故障类别的异常跟踪进行分类，无论是在最初训练的同一系统内，还是在其他系统中。

    arXiv:2403.18998v1 Announce Type: cross  Abstract: Microservice-based systems (MSS) may experience failures in various fault categories due to their complex and dynamic nature. To effectively handle failures, AIOps tools utilize trace-based anomaly detection and root cause analysis. In this paper, we propose a novel framework for few-shot abnormal trace classification for MSS. Our framework comprises two main components: (1) Multi-Head Attention Autoencoder for constructing system-specific trace representations, which enables (2) Transformer Encoder-based Model-Agnostic Meta-Learning to perform effective and efficient few-shot learning for abnormal trace classification. The proposed framework is evaluated on two representative MSS, Trainticket and OnlineBoutique, with open datasets. The results show that our framework can adapt the learned knowledge to classify new, unseen abnormal traces of novel fault categories both within the same system it was initially trained on and even in the 
    
[^2]: 和抛硬币一样好：人类对AI生成的图像、视频、音频和音视频刺激的检测

    As Good As A Coin Toss Human detection of AI-generated images, videos, audio, and audiovisual stimuli

    [https://arxiv.org/abs/2403.16760](https://arxiv.org/abs/2403.16760)

    通过一项感知研究，评估了人们在日常生活中对合成图像、音频、视频和音视频刺激与真实的区分能力，以探讨人类对欺骗性合成媒体的易受程度。

    

    随着合成媒体变得越来越逼真，使用它的障碍不断降低，这项技术越来越被恶意利用，从金融欺诈到非自愿色情。今天，对抗被合成媒体误导的主要防御依赖于人类观察者在视觉和听觉上区分真假的能力。然而，人们在日常生活中实际上对欺骗性合成媒体有多脆弱仍不清楚。我们进行了一个包含1276名参与者的感知研究，评估人们在区分合成图像、仅音频、仅视频和音视频刺激与真实的准确性如何。为了反映人们在野外可能遇到合成媒体的情况，测试条件和刺激模拟了典型的在线平台，而调查中使用的所有合成媒体均来自

    arXiv:2403.16760v1 Announce Type: cross  Abstract: As synthetic media becomes progressively more realistic and barriers to using it continue to lower, the technology has been increasingly utilized for malicious purposes, from financial fraud to nonconsensual pornography. Today, the principal defense against being misled by synthetic media relies on the ability of the human observer to visually and auditorily discern between real and fake. However, it remains unclear just how vulnerable people actually are to deceptive synthetic media in the course of their day to day lives. We conducted a perceptual study with 1276 participants to assess how accurate people were at distinguishing synthetic images, audio only, video only, and audiovisual stimuli from authentic. To reflect the circumstances under which people would likely encounter synthetic media in the wild, testing conditions and stimuli emulated a typical online platform, while all synthetic media used in the survey was sourced from 
    
[^3]: ChatDBG: 一种基于人工智能的调试助手

    ChatDBG: An AI-Powered Debugging Assistant

    [https://arxiv.org/abs/2403.16354](https://arxiv.org/abs/2403.16354)

    ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。

    

    本文介绍了ChatDBG，这是第一个基于人工智能的调试助手。ChatDBG集成了大型语言模型(LLMs)，显著增强了传统调试器的功能和用户友好性。ChatDBG允许程序员与调试器进行协作对话，使他们能够提出关于程序状态的复杂问题，对崩溃或断言失败进行根本原因分析，并探索诸如“为什么x为空？”之类的开放性查询。为了处理这些查询，ChatDBG授予LLM自主权，通过发出命令来浏览堆栈和检查程序状态进行调试；然后报告其发现并将控制权交还给程序员。我们的ChatDBG原型与标准调试器集成，包括LLDB、GDB和WinDBG用于本地代码以及用于Python的Pdb。我们在各种代码集合上进行了评估，包括具有已知错误的C/C++代码和一套Python代码。

    arXiv:2403.16354v1 Announce Type: cross  Abstract: This paper presents ChatDBG, the first AI-powered debugging assistant. ChatDBG integrates large language models (LLMs) to significantly enhance the capabilities and user-friendliness of conventional debuggers. ChatDBG lets programmers engage in a collaborative dialogue with the debugger, allowing them to pose complex questions about program state, perform root cause analysis for crashes or assertion failures, and explore open-ended queries like "why is x null?". To handle these queries, ChatDBG grants the LLM autonomy to take the wheel and drive debugging by issuing commands to navigate through stacks and inspect program state; it then reports its findings and yields back control to the programmer. Our ChatDBG prototype integrates with standard debuggers including LLDB, GDB, and WinDBG for native code and Pdb for Python. Our evaluation across a diverse set of code, including C/C++ code with known bugs and a suite of Python code includi
    
[^4]: 用于理性推理的球形神经网络

    Sphere Neural-Networks for Rational Reasoning

    [https://arxiv.org/abs/2403.15297](https://arxiv.org/abs/2403.15297)

    该论文提出了一种球形神经网络（SphNNs）来进行理性推理，通过将计算构建块从向量推广到球体，实现了人类类似的推理能力，并开发了用于三段论推理的SphNN。

    

    大型语言模型（LLMs）如ChatGPT的成功得到了广泛的认可，其类人问题回答的能力以及不断提升的推理性能都证明了这一点。然而，LLMs是否会进行推理仍然不清楚。传统神经网络如何在定性上扩展以超越统计范式并实现高级认知是一个未解之谜。在这里，我们通过将计算构建块从向量推广到球体的方式提出了一种极简的定性扩展。我们提出了球形神经网络（SphNNs）用于通过模型构建和检查进行类人推理，并为三段论推理开发了SphNN，这是人类理性的缩影。SphNN不使用训练数据，而是使用邻域空间关系的神经符号转换映射来指导从当前球形配置向目标的转换。

    arXiv:2403.15297v1 Announce Type: new  Abstract: The success of Large Language Models (LLMs), e.g., ChatGPT, is witnessed by their planetary popularity, their capability of human-like question-answering, and also by their steadily improved reasoning performance. However, it remains unclear whether LLMs reason. It is an open problem how traditional neural networks can be qualitatively extended to go beyond the statistic paradigm and achieve high-level cognition. Here, we present a minimalist qualitative extension by generalising computational building blocks from vectors to spheres. We propose Sphere Neural Networks (SphNNs) for human-like reasoning through model construction and inspection, and develop SphNN for syllogistic reasoning, a microcosm of human rationality. Instead of training data, SphNN uses a neuro-symbolic transition map of neighbourhood spatial relations to guide transformations from the current sphere configuration towards the target. SphNN is the first neural model th
    
[^5]: AI 评估量表（AIAS）的实践：GenAI 支持评估的试点实施

    The AI Assessment Scale (AIAS) in action: A pilot implementation of GenAI supported assessment

    [https://arxiv.org/abs/2403.14692](https://arxiv.org/abs/2403.14692)

    该论文介绍了AI评估量表（AIAS）的实践应用，通过灵活框架将GenAI技术纳入教育评估中，显著降低了与GenAI相关的学术不端案件，提高了学生的学业成绩。

    

    在高等教育中快速采用生成人工智能（GenAI）技术引发了对学术诚信、评估实践和学生学习的关注。禁止或阻止GenAI工具已被证明是无效的，惩罚性方法忽略了这些技术的潜在好处。本文介绍了在英国越南大学（BUV）进行的试点研究的结果，探讨了人工智能评估量表（AIAS）的实施，这是一个灵活的框架，用于将GenAI纳入教育评估中。AIAS由五个级别组成，从“无AI”到“完全AI”，使教育工作者能够设计侧重于需要人类输入和批判性思维的评估。在实施AIAS后，试点研究结果表明与GenAI相关的学术不端案件显着减少，学生成绩提高了5.9%。

    arXiv:2403.14692v1 Announce Type: cross  Abstract: The rapid adoption of Generative Artificial Intelligence (GenAI) technologies in higher education has raised concerns about academic integrity, assessment practices, and student learning. Banning or blocking GenAI tools has proven ineffective, and punitive approaches ignore the potential benefits of these technologies. This paper presents the findings of a pilot study conducted at British University Vietnam (BUV) exploring the implementation of the Artificial Intelligence Assessment Scale (AIAS), a flexible framework for incorporating GenAI into educational assessments. The AIAS consists of five levels, ranging from 'No AI' to 'Full AI', enabling educators to design assessments that focus on areas requiring human input and critical thinking.   Following the implementation of the AIAS, the pilot study results indicate a significant reduction in academic misconduct cases related to GenAI, a 5.9% increase in student attainment across the 
    
[^6]: ASGEA：利用Align-Subgraphs中的逻辑规则进行实体对齐

    ASGEA: Exploiting Logic Rules from Align-Subgraphs for Entity Alignment

    [https://arxiv.org/abs/2402.11000](https://arxiv.org/abs/2402.11000)

    提出了一个新的实体对齐框架ASGEA，利用Align-Subgraphs中的逻辑规则，设计了可解释的基于路径的图神经网络ASGNN，引入了多模态注意机制，取得了令人满意的实验结果

    

    实体对齐（EA）旨在识别代表相同现实世界对象的不同知识图中的实体。最近基于嵌入的EA方法在EA方面取得了最先进的性能，但面临着解释性挑战，因为它们完全依赖于嵌入距离，并忽视了一对对齐实体背后的逻辑规则。在本文中，我们提出了Align-Subgraph实体对齐（ASGEA）框架来利用Align-Subgraphs中的逻辑规则。ASGEA使用锚链接作为桥梁来构建Align-Subgraphs，并沿着跨知识图的路径传播，这使其区别于基于嵌入的方法。此外，我们设计了一种可解释的基于路径的图神经网络ASGNN，以有效识别和整合跨知识图的逻辑规则。我们还引入了一个节点级多模态注意机制，结合多模态增强的锚点来增强Align-Subgraph。我们的实验结果

    arXiv:2402.11000v1 Announce Type: cross  Abstract: Entity alignment (EA) aims to identify entities across different knowledge graphs that represent the same real-world objects. Recent embedding-based EA methods have achieved state-of-the-art performance in EA yet faced interpretability challenges as they purely rely on the embedding distance and neglect the logic rules behind a pair of aligned entities. In this paper, we propose the Align-Subgraph Entity Alignment (ASGEA) framework to exploit logic rules from Align-Subgraphs. ASGEA uses anchor links as bridges to construct Align-Subgraphs and spreads along the paths across KGs, which distinguishes it from the embedding-based methods. Furthermore, we design an interpretable Path-based Graph Neural Network, ASGNN, to effectively identify and integrate the logic rules across KGs. We also introduce a node-level multi-modal attention mechanism coupled with multi-modal enriched anchors to augment the Align-Subgraph. Our experimental results 
    
[^7]: CURE: 机器人领域的模拟辅助自动调节技术

    CURE: Simulation-Augmented Auto-Tuning in Robotics

    [https://arxiv.org/abs/2402.05399](https://arxiv.org/abs/2402.05399)

    本论文提出了一种模拟辅助的自动调节技术，用于解决机器人系统中的高度可配置参数的优化问题。该技术通过解决软硬件之间配置选项的交互问题，实现了在不同环境和机器人平台之间的性能迁移。

    

    机器人系统通常由多个子系统组成，例如定位和导航，每个子系统又包含许多可配置的组件（例如选择不同的规划算法）。一旦选择了某个算法，就需要设置相关的配置选项以达到适当的值。系统堆栈中的配置选项会产生复杂的交互关系。在高度可配置的机器人中找到最佳配置来实现期望的性能是一个重大挑战，因为软件和硬件之间的配置选项交互导致了庞大且复杂的配置空间。性能迁移在不同的环境和机器人平台之间也是一个难题。数据高效优化算法（例如贝叶斯优化）已越来越多地用于自动化调整网络物理系统中的可配置参数。然而，这样的优化算法在机器人领域应用仍有局限性。

    Robotic systems are typically composed of various subsystems, such as localization and navigation, each encompassing numerous configurable components (e.g., selecting different planning algorithms). Once an algorithm has been selected for a component, its associated configuration options must be set to the appropriate values. Configuration options across the system stack interact non-trivially. Finding optimal configurations for highly configurable robots to achieve desired performance poses a significant challenge due to the interactions between configuration options across software and hardware that result in an exponentially large and complex configuration space. These challenges are further compounded by the need for transferability between different environments and robotic platforms. Data efficient optimization algorithms (e.g., Bayesian optimization) have been increasingly employed to automate the tuning of configurable parameters in cyber-physical systems. However, such optimiz
    
[^8]: D-STGCNT:一种基于transformer的密集时空图卷积GRU网络用于评估患者身体康复

    D-STGCNT: A Dense Spatio-Temporal Graph Conv-GRU Network based on transformer for assessment of patient physical rehabilitation. (arXiv:2401.06150v1 [eess.IV])

    [http://arxiv.org/abs/2401.06150](http://arxiv.org/abs/2401.06150)

    D-STGCNT是一种新的模型，结合了STGCN和transformer的架构，用于自动评估患者身体康复锻炼。它通过将骨架数据视为图形，并检测关键关节，在处理时空数据方面具有高效性。该模型通过密集连接和GRU机制来处理大型3D骨架输入，有效建立时空动态模型。transformer的注意力机制对于评估康复锻炼非常有用。

    

    本文解决了自动评估无临床监督情况下患者进行身体康复锻炼的挑战。其目标是提供质量评分以确保正确执行和获得期望结果。为实现这一目标，引入了一种新的基于图结构的模型，Dense Spatio-Temporal Graph Conv-GRU Network with Transformer。该模型结合了改进的STGCN和transformer架构，用于高效处理时空数据。其关键思想是将骨架数据视为图形，并检测每个康复锻炼中起主要作用的关节。密集连接和GRU机制用于快速处理大型3D骨架输入并有效建模时空动态。transformer编码器的注意机制侧重于输入序列的相关部分，使其在评估康复锻炼方面非常有用。

    This paper tackles the challenge of automatically assessing physical rehabilitation exercises for patients who perform the exercises without clinician supervision. The objective is to provide a quality score to ensure correct performance and achieve desired results. To achieve this goal, a new graph-based model, the Dense Spatio-Temporal Graph Conv-GRU Network with Transformer, is introduced. This model combines a modified version of STGCN and transformer architectures for efficient handling of spatio-temporal data. The key idea is to consider skeleton data respecting its non-linear structure as a graph and detecting joints playing the main role in each rehabilitation exercise. Dense connections and GRU mechanisms are used to rapidly process large 3D skeleton inputs and effectively model temporal dynamics. The transformer encoder's attention mechanism focuses on relevant parts of the input sequence, making it useful for evaluating rehabilitation exercises. The evaluation of our propose
    
[^9]: 通过随机滤波和模式识别强化基于POD的反应扩散复杂网络模型简化技术

    Reinforcing POD based model reduction techniques in reaction-diffusion complex networks using stochastic filtering and pattern recognition. (arXiv:2307.09762v1 [cs.CE])

    [http://arxiv.org/abs/2307.09762](http://arxiv.org/abs/2307.09762)

    该论文提出了一种算法框架，通过将模式识别和随机滤波理论的技术结合起来，强化了基于POD的反应扩散复杂网络模型简化技术，在受扰动输入的情况下提高了代理模型的准确性。

    

    复杂网络被用于建模许多现实世界系统，然而这些系统的维度使得其分析变得困难。在这种情况下，可以使用POD等降维技术。然而，这些模型容易受输入数据扰动的影响。我们提出了一种算法框架，将模式识别和随机滤波理论的技术结合起来，以增强这些模型的输出。研究结果表明，我们的方法可以在受扰动输入的情况下提高代理模型的准确性。深度神经网络(DNNs)容易受到对抗性攻击，然而最近的研究发现，神经常微分方程(ODEs)在特定应用中表现出鲁棒性。我们将我们的算法框架与基于神经ODE的方法进行了基准比较。

    Complex networks are used to model many real-world systems. However, the dimensionality of these systems can make them challenging to analyze. Dimensionality reduction techniques like POD can be used in such cases. However, these models are susceptible to perturbations in the input data. We propose an algorithmic framework that combines techniques from pattern recognition (PR) and stochastic filtering theory to enhance the output of such models. The results of our study show that our method can improve the accuracy of the surrogate model under perturbed inputs. Deep Neural Networks (DNNs) are susceptible to adversarial attacks. However, recent research has revealed that neural Ordinary Differential Equations (ODEs) exhibit robustness in specific applications. We benchmark our algorithmic framework with a Neural ODE-based approach as a reference.
    

