# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Diffusion Models for Adversarial Purification](https://arxiv.org/abs/2403.16067) | 提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。 |
| [^2] | [SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning](https://arxiv.org/abs/2403.15648) | SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。 |
| [^3] | [CleanAgent: Automating Data Standardization with LLM-based Agents](https://arxiv.org/abs/2403.08291) | 提出了一个具有声明性、统一API的Python库，通过简洁的API调用简化LLM的代码生成流程 |
| [^4] | [GraphEdit: Large Language Models for Graph Structure Learning](https://arxiv.org/abs/2402.15183) | 本研究提出了一种名为GraphEdit的方法，利用大型语言模型（LLMs）学习复杂的图结构化数据中的节点关系，通过在图结构上进行指导调整，增强LLMs的推理能力，从而提高图结构学习的可靠性。 |
| [^5] | [Synthesis of Hierarchical Controllers Based on Deep Reinforcement Learning Policies](https://arxiv.org/abs/2402.13785) | 提出了基于深度强化学习策略的分层控制器设计方法，通过训练简洁的“潜在”策略来解决房间建模问题，无需模型提炼步骤，克服了DRL中的稀疏奖励，实现了低级策略的可重用性 |
| [^6] | [Representation Learning Using a Single Forward Pass](https://arxiv.org/abs/2402.09769) | 我们提出了一种神经科学启发的算法，可以通过单次前向传递进行表示学习。该算法具有独特的特点，并在不需要反向传播的情况下取得了高性能的分类结果。 |
| [^7] | [Future Prediction Can be a Strong Evidence of Good History Representation in Partially Observable Environments](https://arxiv.org/abs/2402.07102) | 未来预测在部分可观测环境中学习 History Representation 具有很强的相关性和有效性。 |
| [^8] | [Detecting mental disorder on social media: a ChatGPT-augmented explainable approach](https://arxiv.org/abs/2401.17477) | 本文提出了一种利用大型语言模型，可解释人工智能和对话代理器ChatGPT相结合的新方法，以解决通过社交媒体检测抑郁症的可解释性挑战。通过将Twitter特定变体BERTweet与自解释模型BERT-XDD相结合，并借助ChatGPT将技术解释转化为人类可读的评论，实现了解释能力的同时提高了可解释性。这种方法可以为发展社会负责任的数字平台，促进早期干预做出贡献。 |
| [^9] | [Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model.](http://arxiv.org/abs/2401.15210) | Roq是一个基于风险感知学习方法的综合框架，用于实现鲁棒的查询优化。 |
| [^10] | [SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance.](http://arxiv.org/abs/2401.15199) | 这个论文介绍了一种来自SCANIA公司的真实世界多变量时间序列数据集，该数据集适用于各种机器学习应用，尤其是预测性维护场景。它具有庞大的样本数量和多样化的特征，以及时间信息，为研究者提供了一个使用真实世界数据的标准基准。 |
| [^11] | [A-KIT: Adaptive Kalman-Informed Transformer.](http://arxiv.org/abs/2401.09987) | 这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。 |
| [^12] | [Path To Gain Functional Transparency In Artificial Intelligence With Meaningful Explainability.](http://arxiv.org/abs/2310.08849) | 本论文提出了一个面向用户的合规设计，用于实现透明系统中的功能透明性，并强调了跨学科合作的重要性。 |
| [^13] | [Mixup Your Own Pairs.](http://arxiv.org/abs/2309.16633) | 本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。 |
| [^14] | [An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System.](http://arxiv.org/abs/2309.12365) | 本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。 |
| [^15] | [Distributionally Robust Statistical Verification with Imprecise Neural Networks.](http://arxiv.org/abs/2308.14815) | 本文提出了一种使用不精确神经网络的分布鲁棒统计验证方法，通过结合主动学习、不确定性量化和神经网络验证，可以在大量的分布上提供对黑盒系统行为的保证。 |

# 详细

[^1]: 针对对抗净化的强大扩散模型

    Robust Diffusion Models for Adversarial Purification

    [https://arxiv.org/abs/2403.16067](https://arxiv.org/abs/2403.16067)

    提出一种独立于预训练扩散模型的稳健反向过程，避免了重新训练或微调，有效处理对抗净化中的语义信息损失问题。

    

    基于扩散模型（DM）的对抗净化（AP）已被证明是对抗训练（AT）最有力的替代方法。然而，这些方法忽略了预训练的扩散模型本身对对抗攻击并不稳健这一事实。此外，扩散过程很容易破坏语义信息，在反向过程后生成高质量图像但与原始输入图像完全不同，导致标准精度下降。为了解决这些问题，一个自然的想法是利用对抗训练策略重新训练或微调预训练的扩散模型，然而这在计算上是禁止的。我们提出了一种新颖的具有对抗引导的稳健反向过程，它独立于给定的预训练DMs，并且避免了重新训练或微调DMs。这种强大的引导不仅可以确保生成的净化示例保留更多的语义内容，还可以...

    arXiv:2403.16067v1 Announce Type: cross  Abstract: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also m
    
[^2]: SRLM: 使用大型语言模型和深度强化学习进行人机交互式社交机器人导航

    SRLM: Human-in-Loop Interactive Social Robot Navigation with Large Language Model and Deep Reinforcement Learning

    [https://arxiv.org/abs/2403.15648](https://arxiv.org/abs/2403.15648)

    SRLM 提出了一种结合了大型语言模型和深度强化学习的新型混合方法，用于人机交互式社交机器人导航，通过实时的人类语言指令推断全局规划，并在公共空间中提供多种社交服务，表现出出色的性能。

    

    一名交互式社交机器人助手必须在复杂拥挤的空间中提供服务，根据实时的人类语言指令或反馈调整其行为。本文提出了一种名为Social Robot Planner (SRLM) 的新型混合方法，它将大型语言模型（LLM）和深度强化学习（DRL）整合起来，以在充斥着人群的公共空间中导航，并提供多种社交服务。SRLM 通过实时的人机交互指令推断全局规划，并将社交信息编码到基于LLM的大型导航模型（LNM）中，用于低层次的运动执行。此外，设计了一个基于DRL的规划器来保持基准性能，通过大型反馈模型（LFM）与LNM融合，以解决当前文本和LLM驱动的LNM的不稳定性。最后，SRLM 在广泛的实验中展示出了出色的性能。有关此工作的更多详细信息，请访问：https://sites.g

    arXiv:2403.15648v1 Announce Type: cross  Abstract: An interactive social robotic assistant must provide services in complex and crowded spaces while adapting its behavior based on real-time human language commands or feedback. In this paper, we propose a novel hybrid approach called Social Robot Planner (SRLM), which integrates Large Language Models (LLM) and Deep Reinforcement Learning (DRL) to navigate through human-filled public spaces and provide multiple social services. SRLM infers global planning from human-in-loop commands in real-time, and encodes social information into a LLM-based large navigation model (LNM) for low-level motion execution. Moreover, a DRL-based planner is designed to maintain benchmarking performance, which is blended with LNM by a large feedback model (LFM) to address the instability of current text and LLM-driven LNM. Finally, SRLM demonstrates outstanding performance in extensive experiments. More details about this work are available at: https://sites.g
    
[^3]: CleanAgent：基于LLM代理自动化数据标准化

    CleanAgent: Automating Data Standardization with LLM-based Agents

    [https://arxiv.org/abs/2403.08291](https://arxiv.org/abs/2403.08291)

    提出了一个具有声明性、统一API的Python库，通过简洁的API调用简化LLM的代码生成流程

    

    数据标准化是数据科学生命周期中至关重要的一部分。虽然诸如Pandas之类的工具提供了强大的功能，但它们的复杂性以及需要定制代码以适应不同列类型的手动操作带来了重大挑战。尽管大型语言模型（LLMs）如ChatGPT已经展现出通过自然语言理解和代码生成自动化此过程的潜力，但仍需要专业程度的编程知识和持续互动以进行及时的完善。为了解决这些挑战，我们的关键想法是提出一个具有声明性、统一API的Python库，用于标准化列类型，通过简洁的API调用简化LLM的代码生成流程。我们首先提出了Dataprep.Clean，作为Dataprep库的一个组件，通过一行代码实现特定列类型的标准化，极大降低了复杂性。然后我们介绍了CleanAgen

    arXiv:2403.08291v1 Announce Type: cross  Abstract: Data standardization is a crucial part in data science life cycle. While tools like Pandas offer robust functionalities, their complexity and the manual effort required for customizing code to diverse column types pose significant challenges. Although large language models (LLMs) like ChatGPT have shown promise in automating this process through natural language understanding and code generation, it still demands expert-level programming knowledge and continuous interaction for prompt refinement. To solve these challenges, our key idea is to propose a Python library with declarative, unified APIs for standardizing column types, simplifying the code generation of LLM with concise API calls. We first propose Dataprep.Clean which is written as a component of the Dataprep Library, offers a significant reduction in complexity by enabling the standardization of specific column types with a single line of code. Then we introduce the CleanAgen
    
[^4]: GraphEdit：用于图结构学习的大型语言模型

    GraphEdit: Large Language Models for Graph Structure Learning

    [https://arxiv.org/abs/2402.15183](https://arxiv.org/abs/2402.15183)

    本研究提出了一种名为GraphEdit的方法，利用大型语言模型（LLMs）学习复杂的图结构化数据中的节点关系，通过在图结构上进行指导调整，增强LLMs的推理能力，从而提高图结构学习的可靠性。

    

    图结构学习（GSL）致力于通过生成新颖的图结构来捕捉图结构数据中节点之间的固有依赖性和相互作用。本文提出了一种名为GraphEdit的方法，利用大型语言模型（LLMs）学习图结构化数据中复杂的节点关系。通过在图结构上进行指导调整，增强LLMs的推理能力，我们旨在克服显式图结构信息带来的挑战，并提高图结构学习的可靠性。

    arXiv:2402.15183v1 Announce Type: cross  Abstract: Graph Structure Learning (GSL) focuses on capturing intrinsic dependencies and interactions among nodes in graph-structured data by generating novel graph structures. Graph Neural Networks (GNNs) have emerged as promising GSL solutions, utilizing recursive message passing to encode node-wise inter-dependencies. However, many existing GSL methods heavily depend on explicit graph structural information as supervision signals, leaving them susceptible to challenges such as data noise and sparsity. In this work, we propose GraphEdit, an approach that leverages large language models (LLMs) to learn complex node relationships in graph-structured data. By enhancing the reasoning capabilities of LLMs through instruction-tuning over graph structures, we aim to overcome the limitations associated with explicit graph structural information and enhance the reliability of graph structure learning. Our approach not only effectively denoises noisy co
    
[^5]: 基于深度强化学习策略的分层控制器合成

    Synthesis of Hierarchical Controllers Based on Deep Reinforcement Learning Policies

    [https://arxiv.org/abs/2402.13785](https://arxiv.org/abs/2402.13785)

    提出了基于深度强化学习策略的分层控制器设计方法，通过训练简洁的“潜在”策略来解决房间建模问题，无需模型提炼步骤，克服了DRL中的稀疏奖励，实现了低级策略的可重用性

    

    我们提出了一种新颖的方法来解决将环境建模为马尔可夫决策过程（MDPs）的控制器设计问题。具体来说，我们考虑了一个分层MDP，一个由称为“房间”的MDP填充的图形。我们首先应用深度强化学习（DRL）来获得每个房间的低级策略，以适应未知结构的大房间。然后我们应用反应合成来获得一个高级规划者，选择在每个房间执行哪个低级策略。在合成规划者方面的中心挑战是需要对房间进行建模。我们通过开发一个DRL过程来训练简洁的“潜在”策略以及关于其性能的PAC保证来解决这一挑战。与先前方法不同，我们的方法规避了模型提炼步骤。我们的方法对抗DRL中的稀疏奖励，并实现了低级策略的可重用性。我们通过一个涉及代理导航的案例研究证明了可行性。

    arXiv:2402.13785v1 Announce Type: new  Abstract: We propose a novel approach to the problem of controller design for environments modeled as Markov decision processes (MDPs). Specifically, we consider a hierarchical MDP a graph with each vertex populated by an MDP called a "room". We first apply deep reinforcement learning (DRL) to obtain low-level policies for each room, scaling to large rooms of unknown structure. We then apply reactive synthesis to obtain a high-level planner that chooses which low-level policy to execute in each room. The central challenge in synthesizing the planner is the need for modeling rooms. We address this challenge by developing a DRL procedure to train concise "latent" policies together with PAC guarantees on their performance. Unlike previous approaches, ours circumvents a model distillation step. Our approach combats sparse rewards in DRL and enables reusability of low-level policies. We demonstrate feasibility in a case study involving agent navigation
    
[^6]: 使用单次前向传递的表示学习

    Representation Learning Using a Single Forward Pass

    [https://arxiv.org/abs/2402.09769](https://arxiv.org/abs/2402.09769)

    我们提出了一种神经科学启发的算法，可以通过单次前向传递进行表示学习。该算法具有独特的特点，并在不需要反向传播的情况下取得了高性能的分类结果。

    

    我们提出了一种受神经科学启发的单次传递嵌入学习算法（SPELA）。 SPELA是在边缘人工智能设备中进行训练和推理应用的首选候选人。 同时，SPELA可以最佳地满足对研究感知表示学习和形成框架的需求。 SPELA具有独特的特征，如嵌入向量形式的神经先验知识，不需要权重传输，不锁定权重更新，完全局部赫比安学习，不存储激活的单次前向传递和每个样本的单次权重更新。与传统方法相比，SPELA可以在不需要反向传播的情况下进行操作。 我们展示了我们的算法在一个有噪音的布尔运算数据集上可以执行非线性分类。 此外，我们展示了SPELA在MNIST，KMNIST和Fashion MNIST上的高性能表现。 最后，我们展示了SPELA在MNIST，KMNIST和Fashion MNIST上的少样本和1个时期学习能力。

    arXiv:2402.09769v1 Announce Type: new  Abstract: We propose a neuroscience-inspired Solo Pass Embedded Learning Algorithm (SPELA). SPELA is a prime candidate for training and inference applications in Edge AI devices. At the same time, SPELA can optimally cater to the need for a framework to study perceptual representation learning and formation. SPELA has distinctive features such as neural priors (in the form of embedded vectors), no weight transport, no update locking of weights, complete local Hebbian learning, single forward pass with no storage of activations, and single weight update per sample. Juxtaposed with traditional approaches, SPELA operates without the need for backpropagation. We show that our algorithm can perform nonlinear classification on a noisy boolean operation dataset. Additionally, we exhibit high performance using SPELA across MNIST, KMNIST, and Fashion MNIST. Lastly, we show the few-shot and 1-epoch learning capabilities of SPELA on MNIST, KMNIST, and Fashio
    
[^7]: 未来预测可以成为部分可观测环境中良好历史表达的有力证据

    Future Prediction Can be a Strong Evidence of Good History Representation in Partially Observable Environments

    [https://arxiv.org/abs/2402.07102](https://arxiv.org/abs/2402.07102)

    未来预测在部分可观测环境中学习 History Representation 具有很强的相关性和有效性。

    

    学习良好的历史表达是部分可观测环境中强化学习的核心挑战之一。最近的研究表明，各种辅助任务对促进表达学习具有优势。然而，这些辅助任务的有效性尚未完全使人信服，特别是在需要长期记忆和推理的部分可观测环境中。在这个实证研究中，我们探讨了未来预测在学习部分可观测环境中历史表达时的有效性。我们首先提出了一种通过未来预测将学习历史表达与策略优化分离的方法。然后，我们的主要贡献有两个方面：（a）我们证明了强化学习的性能与部分可观测环境中未来观测的预测精度强相关，（b）我们的方法可以有效地学习部分可观测环境中长时间历史的表达方式。

    Learning a good history representation is one of the core challenges of reinforcement learning (RL) in partially observable environments. Recent works have shown the advantages of various auxiliary tasks for facilitating representation learning. However, the effectiveness of such auxiliary tasks has not been fully convincing, especially in partially observable environments that require long-term memorization and inference. In this empirical study, we investigate the effectiveness of future prediction for learning the representations of histories, possibly of extensive length, in partially observable environments. We first introduce an approach that decouples the task of learning history representations from policy optimization via future prediction. Then, our main contributions are two-fold: (a) we demonstrate that the performance of reinforcement learning is strongly correlated with the prediction accuracy of future observations in partially observable environments, and (b) our approa
    
[^8]: 在社交媒体上检测心理障碍：基于ChatGPT的可解释方法

    Detecting mental disorder on social media: a ChatGPT-augmented explainable approach

    [https://arxiv.org/abs/2401.17477](https://arxiv.org/abs/2401.17477)

    本文提出了一种利用大型语言模型，可解释人工智能和对话代理器ChatGPT相结合的新方法，以解决通过社交媒体检测抑郁症的可解释性挑战。通过将Twitter特定变体BERTweet与自解释模型BERT-XDD相结合，并借助ChatGPT将技术解释转化为人类可读的评论，实现了解释能力的同时提高了可解释性。这种方法可以为发展社会负责任的数字平台，促进早期干预做出贡献。

    

    在数字时代，社交媒体上表达的抑郁症状的频率引起了严重关注，迫切需要先进的方法来及时检测。本文通过提出一种新颖的方法，将大型语言模型（LLM）与可解释的人工智能（XAI）和ChatGPT等对话代理器有效地结合起来，以应对可解释性抑郁症检测的挑战。在我们的方法中，通过将Twitter特定变体BERTweet与一种新型的自解释模型BERT-XDD相结合，实现了解释能力，该模型能够通过掩码注意力提供分类和解释。使用ChatGPT将技术解释转化为可读性强的评论，进一步增强了可解释性。通过引入一种有效且模块化的可解释抑郁症检测方法，我们的方法可以为发展社会负责任的数字平台做出贡献，促进早期干预。

    In the digital era, the prevalence of depressive symptoms expressed on social media has raised serious concerns, necessitating advanced methodologies for timely detection. This paper addresses the challenge of interpretable depression detection by proposing a novel methodology that effectively combines Large Language Models (LLMs) with eXplainable Artificial Intelligence (XAI) and conversational agents like ChatGPT. In our methodology, explanations are achieved by integrating BERTweet, a Twitter-specific variant of BERT, into a novel self-explanatory model, namely BERT-XDD, capable of providing both classification and explanations via masked attention. The interpretability is further enhanced using ChatGPT to transform technical explanations into human-readable commentaries. By introducing an effective and modular approach for interpretable depression detection, our methodology can contribute to the development of socially responsible digital platforms, fostering early intervention and
    
[^9]: Roq：基于风险感知学习成本模型的鲁棒查询优化

    Roq: Robust Query Optimization Based on a Risk-aware Learned Cost Model. (arXiv:2401.15210v1 [cs.DB])

    [http://arxiv.org/abs/2401.15210](http://arxiv.org/abs/2401.15210)

    Roq是一个基于风险感知学习方法的综合框架，用于实现鲁棒的查询优化。

    

    关系数据库管理系统(RDBMS)中的查询优化器搜索预期对于给定查询最优的执行计划。它们使用参数估计，通常是不准确的，并且做出的假设在实践中可能不成立。因此，在这些估计和假设无效时，它们可能选择在运行时是次优的执行计划，这可能导致查询性能不佳。因此，查询优化器不足以支持鲁棒的查询优化。近年来，使用机器学习(ML)来提高数据系统的效率并减少其维护开销的兴趣日益高涨，在查询优化领域取得了有希望的结果。在本文中，受到这些进展的启发，并基于IBM Db2多年的经验，我们提出了Roq: 一种基于风险感知学习方法的综合框架，它实现了鲁棒的查询优化。

    Query optimizers in relational database management systems (RDBMSs) search for execution plans expected to be optimal for a given queries. They use parameter estimates, often inaccurate, and make assumptions that may not hold in practice. Consequently, they may select execution plans that are suboptimal at runtime, when these estimates and assumptions are not valid, which may result in poor query performance. Therefore, query optimizers do not sufficiently support robust query optimization. Recent years have seen a surge of interest in using machine learning (ML) to improve efficiency of data systems and reduce their maintenance overheads, with promising results obtained in the area of query optimization in particular. In this paper, inspired by these advancements, and based on several years of experience of IBM Db2 in this journey, we propose Robust Optimization of Queries, (Roq), a holistic framework that enables robust query optimization based on a risk-aware learning approach. Roq 
    
[^10]: SCANIA组件X数据集：用于预测性维护的真实世界多变量时间序列数据集

    SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance. (arXiv:2401.15199v1 [cs.LG])

    [http://arxiv.org/abs/2401.15199](http://arxiv.org/abs/2401.15199)

    这个论文介绍了一种来自SCANIA公司的真实世界多变量时间序列数据集，该数据集适用于各种机器学习应用，尤其是预测性维护场景。它具有庞大的样本数量和多样化的特征，以及时间信息，为研究者提供了一个使用真实世界数据的标准基准。

    

    本论文介绍了一种来自SCANIA瑞典公司的卡车车队中匿名发动机部件（称为Component X）的真实世界多变量时间序列数据集。该数据集包括多种变量，捕捉了详细的操作数据、维修记录和卡车规格，同时通过匿名处理保持机密性。它非常适用于各种机器学习应用，如分类、回归、生存分析和异常检测，特别是在预测性维护场景中的应用。庞大的样本数量和以直方图和计数器形式的多样化特征，以及包含时间信息，使得这个真实世界数据集在该领域中独特。发布这个数据集的目标是让广大研究人员有可能使用来自一家国际知名公司的真实世界数据，并引入一个标准基准用于预测性维护的研究。

    This paper presents a description of a real-world, multivariate time series dataset collected from an anonymized engine component (called Component X) of a fleet of trucks from SCANIA, Sweden. This dataset includes diverse variables capturing detailed operational data, repair records, and specifications of trucks while maintaining confidentiality by anonymization. It is well-suited for a range of machine learning applications, such as classification, regression, survival analysis, and anomaly detection, particularly when applied to predictive maintenance scenarios. The large population size and variety of features in the format of histograms and numerical counters, along with the inclusion of temporal information, make this real-world dataset unique in the field. The objective of releasing this dataset is to give a broad range of researchers the possibility of working with real-world data from an internationally well-known company and introduce a standard benchmark to the predictive ma
    
[^11]: A-KIT:自适应Kalman-Informed Transformer

    A-KIT: Adaptive Kalman-Informed Transformer. (arXiv:2401.09987v1 [cs.RO])

    [http://arxiv.org/abs/2401.09987](http://arxiv.org/abs/2401.09987)

    这项研究提出了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习传感器融合中变化的过程噪声协方差。它通过适应实际情况中的过程噪声变化，改进了估计状态的准确性，避免了滤波器发散的问题。

    

    扩展卡尔曼滤波器(EKF)是导航应用中广泛采用的传感器融合方法。EKF的一个关键方面是在线确定反映模型不确定性的过程噪声协方差矩阵。尽管常见的EKF实现假设过程噪声是恒定的，但在实际情况中，过程噪声是变化的，导致估计状态的不准确，并可能导致滤波器发散。为了应对这种情况，提出了基于模型的自适应EKF方法，并展示了性能改进，凸显了对稳健自适应方法的需求。在本文中，我们推导并引入了A-KIT，一种自适应的Kalman-informed transformer，用于在线学习变化的过程噪声协方差。A-KIT框架适用于任何类型的传感器融合。我们在这里介绍了基于惯性导航系统和多普勒速度日志的非线性传感器融合方法。通过使用来自自主无人潜水器的真实记录数据，我们验证了A-KIT的有效性。

    The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous und
    
[^12]: 实现人工智能功能透明性的路径与有意义的可解释性

    Path To Gain Functional Transparency In Artificial Intelligence With Meaningful Explainability. (arXiv:2310.08849v1 [cs.AI])

    [http://arxiv.org/abs/2310.08849](http://arxiv.org/abs/2310.08849)

    本论文提出了一个面向用户的合规设计，用于实现透明系统中的功能透明性，并强调了跨学科合作的重要性。

    

    人工智能（AI）正在快速融入我们日常生活的各个方面，影响着诸如定向广告和配对算法等领域的决策过程。随着AI系统变得越来越复杂，确保其透明性和可解释性变得至关重要。功能透明性是算法决策系统的一个基本方面，它使利益相关者能够理解这些系统的内在运作，并能够评估其公正性和准确性。然而，实现功能透明性面临着重大挑战，需要加以解决。在本文中，我们提出了一种面向用户的合规设计，用于透明系统中的透明功能。我们强调，开发透明和可解释的人工智能系统是一项复杂而跨学科的努力，需要计算机科学、人工智能、伦理学、法律和社会的研究人员之间的合作。

    Artificial Intelligence (AI) is rapidly integrating into various aspects of our daily lives, influencing decision-making processes in areas such as targeted advertising and matchmaking algorithms. As AI systems become increasingly sophisticated, ensuring their transparency and explainability becomes crucial. Functional transparency is a fundamental aspect of algorithmic decision-making systems, allowing stakeholders to comprehend the inner workings of these systems and enabling them to evaluate their fairness and accuracy. However, achieving functional transparency poses significant challenges that need to be addressed. In this paper, we propose a design for user-centered compliant-by-design transparency in transparent systems. We emphasize that the development of transparent and explainable AI systems is a complex and multidisciplinary endeavor, necessitating collaboration among researchers from diverse fields such as computer science, artificial intelligence, ethics, law, and social 
    
[^13]: 混合你自己的对比对

    Mixup Your Own Pairs. (arXiv:2309.16633v1 [cs.LG])

    [http://arxiv.org/abs/2309.16633](http://arxiv.org/abs/2309.16633)

    本文提出了一种名为SupReMix的方法，通过混合样本，特别是混合负样本和混合正样本，来解决回归问题中表示学习的挑战。这种方法能够提供更好的性能和更准确的回归结果。

    

    在表示学习中，回归问题传统上比分类问题受到的关注较少。直接应用为分类设计的表示学习技术到回归问题往往会导致潜空间中碎片化的表示，从而产生次优的性能。本文认为，由于忽视了两个关键方面：序序感知和难度，对于回归问题而言，对比学习的潜能被忽视了。为了解决这些挑战，我们提倡“混合自己的对比对进行监督性对比回归”，而不仅仅依靠真实/增强样本。具体来说，我们提出了混合式监督对比回归学习（SupReMix）。它在嵌入级别上以锚点包含的混合（锚点和一个不同的负样本的混合）作为困难负对，以锚点排除的混合（两个不同的负样本的混合）作为困难正对。这一策略形成了困难样本对学习的方式。

    In representation learning, regression has traditionally received less attention than classification. Directly applying representation learning techniques designed for classification to regression often results in fragmented representations in the latent space, yielding sub-optimal performance. In this paper, we argue that the potential of contrastive learning for regression has been overshadowed due to the neglect of two crucial aspects: ordinality-awareness and hardness. To address these challenges, we advocate "mixup your own contrastive pairs for supervised contrastive regression", instead of relying solely on real/augmented samples. Specifically, we propose Supervised Contrastive Learning for Regression with Mixup (SupReMix). It takes anchor-inclusive mixtures (mixup of the anchor and a distinct negative sample) as hard negative pairs and anchor-exclusive mixtures (mixup of two distinct negative samples) as hard positive pairs at the embedding level. This strategy formulates harde
    
[^14]: 一个高效的智能半自动仓库库存盘点系统

    An Efficient Intelligent Semi-Automated Warehouse Inventory Stocktaking System. (arXiv:2309.12365v1 [cs.HC])

    [http://arxiv.org/abs/2309.12365](http://arxiv.org/abs/2309.12365)

    本研究提出了一个智能库存管理系统，通过结合条码和分布式flutter应用技术，以及大数据分析实现数据驱动的决策，解决了库存管理中的准确性、监测延迟和过度依赖主观经验的挑战。

    

    在不断发展的供应链管理背景下，高效的库存管理对于企业变得越来越重要。然而，传统的手工和经验驱动的方法往往难以满足现代市场需求的复杂性。本研究引入了一种智能库存管理系统，以解决与数据不准确、监测延迟和过度依赖主观经验的预测相关的挑战。该系统结合了条码和分布式 flutter 应用技术，用于智能感知，并通过全面的大数据分析实现数据驱动的决策。通过仔细的分析、系统设计、关键技术探索和模拟验证，成功展示了所提出系统的有效性。该智能系统实现了二级监测、高频检查和人工智能驱动的预测，从而提高了自动化程度。

    In the context of evolving supply chain management, the significance of efficient inventory management has grown substantially for businesses. However, conventional manual and experience-based approaches often struggle to meet the complexities of modern market demands. This research introduces an intelligent inventory management system to address challenges related to inaccurate data, delayed monitoring, and overreliance on subjective experience in forecasting. The proposed system integrates bar code and distributed flutter application technologies for intelligent perception, alongside comprehensive big data analytics to enable data-driven decision-making. Through meticulous analysis, system design, critical technology exploration, and simulation validation, the effectiveness of the proposed system is successfully demonstrated. The intelligent system facilitates second-level monitoring, high-frequency checks, and artificial intelligence-driven forecasting, consequently enhancing the au
    
[^15]: 使用不精确神经网络的分布鲁棒统计验证

    Distributionally Robust Statistical Verification with Imprecise Neural Networks. (arXiv:2308.14815v1 [cs.AI])

    [http://arxiv.org/abs/2308.14815](http://arxiv.org/abs/2308.14815)

    本文提出了一种使用不精确神经网络的分布鲁棒统计验证方法，通过结合主动学习、不确定性量化和神经网络验证，可以在大量的分布上提供对黑盒系统行为的保证。

    

    在AI安全领域，一个特别具有挑战性的问题是在高维自主系统的行为上提供保证。以可达性分析为中心的验证方法无法扩展，而纯粹的统计方法受到对采样过程的分布假设的限制。相反，我们提出了一个针对黑盒系统的分布鲁棒版本的统计验证问题，其中我们的性能保证适用于大量的分布。本文提出了一种基于主动学习、不确定性量化和神经网络验证的新方法。我们方法的一个核心部分是一种称为不精确神经网络的集成技术，它提供了不确定性以指导主动学习。主动学习使用了一种称为Sherlock的全面神经网络验证工具来收集样本。在openAI gym Mujoco环境中使用多个物理模拟器进行评估。

    A particularly challenging problem in AI safety is providing guarantees on the behavior of high-dimensional autonomous systems. Verification approaches centered around reachability analysis fail to scale, and purely statistical approaches are constrained by the distributional assumptions about the sampling process. Instead, we pose a distributionally robust version of the statistical verification problem for black-box systems, where our performance guarantees hold over a large family of distributions. This paper proposes a novel approach based on a combination of active learning, uncertainty quantification, and neural network verification. A central piece of our approach is an ensemble technique called Imprecise Neural Networks, which provides the uncertainty to guide active learning. The active learning uses an exhaustive neural-network verification tool Sherlock to collect samples. An evaluation on multiple physical simulators in the openAI gym Mujoco environments with reinforcement-
    

