# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models](https://arxiv.org/abs/2403.12952) | 引入了测试时间原型转移（TPS）框架，通过动态学习每个原型的转移向量，有效地弥合了领域差距并增强了类 |
| [^2] | [Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification](https://arxiv.org/abs/2403.12151) | 大型语言模型与知识图谱结合，提高零样本对象状态分类性能 |
| [^3] | [Merino: Entropy-driven Design for Generative Language Models on IoT Devices](https://arxiv.org/abs/2403.07921) | 在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型，通过最大化transformer解码器的熵来在计算预算内，成功设计了MeRino模型，在移动设置下展现出与当前最先进的自回归transformer模型竞争性能的特点 |
| [^4] | [Towards Multimodal Human Intention Understanding Debiasing via Subject-Deconfounding](https://arxiv.org/abs/2403.05025) | 通过引入概括性因果图和分析主题混淆效应，本文提出了SuCI，实现了多模态人类意图理解的去偏见，解决了MIU模型受主体变异问题困扰的挑战。 |
| [^5] | [ChatGPT and biometrics: an assessment of face recognition, gender detection, and age estimation capabilities](https://arxiv.org/abs/2403.02965) | 本文评估了ChatGPT在面部识别、性别检测和年龄估计等生物识别任务中的表现，结果显示ChatGPT在面部识别方面具有较高准确性，并在性别检测方面表现显著，在年龄估计任务中也具有相当准确性。 |
| [^6] | [Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey](https://arxiv.org/abs/2403.00420) | 通过对抗性训练来改进DRL对条件变化的鲁棒性，研究者系统分析了当代对抗攻击方法，提供了详细见解。 |
| [^7] | [Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts](https://arxiv.org/abs/2402.16822) | Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。 |
| [^8] | [Weighted Ensemble Models Are Strong Continual Learners](https://arxiv.org/abs/2312.08977) | 通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。 |
| [^9] | [AgentMixer: Multi-Agent Correlated Policy Factorization.](http://arxiv.org/abs/2401.08728) | AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。 |
| [^10] | [Representation Learning with Large Language Models for Recommendation.](http://arxiv.org/abs/2310.15950) | 这篇论文介绍了一个模型-不可知的框架RLMRec，通过使用大语言模型（LLMs）来增强传统的基于ID的推荐系统，并解决了可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。 |
| [^11] | [Multiple Physics Pretraining for Physical Surrogate Models.](http://arxiv.org/abs/2310.02994) | 多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。 |

# 详细

[^1]: 只需转移它：测试时间原型转移用于视觉语言模型的零样本泛化

    Just Shift It: Test-Time Prototype Shifting for Zero-Shot Generalization with Vision-Language Models

    [https://arxiv.org/abs/2403.12952](https://arxiv.org/abs/2403.12952)

    引入了测试时间原型转移（TPS）框架，通过动态学习每个原型的转移向量，有效地弥合了领域差距并增强了类

    

    视觉语言模型（VLMs）的进展推动了计算机视觉领域的发展，特别是在零样本学习设置中。尽管它们很有前景，但这些模型的有效性在测试环境中往往会因为领域转移而降低。为了解决这个问题，我们引入了测试时间原型转移（TPS）框架，这是一种旨在使用标记测试输入来使VLM适应测试数据集的开创性方法。我们的方法基于在共享嵌入空间中调节每个类别的原型的概念。通过使用预先训练的文本编码器生成并缓存原型，TPS不仅促进了无需优化的原型重用进行后续预测，还让其能够无缝集成当前进展的提示工程技术。在测试时间，TPS仅基于给定的测试样本动态学习每个原型的转移向量，有效地弥合领域差距并增强类

    arXiv:2403.12952v1 Announce Type: cross  Abstract: Advancements in vision-language models (VLMs) have propelled the field of computer vision, particularly in the zero-shot learning setting. Despite their promise, the effectiveness of these models often diminishes due to domain shifts in test environments. To address this, we introduce the Test-Time Prototype Shifting (TPS) framework, a pioneering approach designed to adapt VLMs to test datasets using unlabeled test inputs. Our method is based on the notion of modulating per-class prototypes in the shared embedding space. By pre-computing and caching prototypes generated with the pre-trained text encoder, TPS not only facilitates optimization-free prototype reuse for subsequent predictions but also enables seamless integration with current advancements in prompt engineering. At test-time, TPS dynamically learns shift vectors for each prototype based solely on the given test sample, effectively bridging the domain gap and enhancing class
    
[^2]: 将大型语言模型中的领域特定内容融入知识图谱，以增强零样本对象状态分类

    Fusing Domain-Specific Content from Large Language Models into Knowledge Graphs for Enhanced Zero Shot Object State Classification

    [https://arxiv.org/abs/2403.12151](https://arxiv.org/abs/2403.12151)

    大型语言模型与知识图谱结合，提高零样本对象状态分类性能

    

    领域特定知识可以显著有助于解决各种视觉任务，但生成这种知识需要大量人力和时间成本。本研究探讨了大型语言模型（LLMs）在通过语义嵌入生成和提供领域特定信息方面的潜力。为实现这一目标，将LLM集成到一个流程中，该流程在视觉基础零样本对象状态分类任务的背景下利用知识图谱和预训练的语义向量。通过广泛的消融研究彻底研究了LLM的行为。我们的研究结果表明，将基于LLM的嵌入与通用的预训练嵌入结合使用可以显著提高性能。借鉴这一消融研究的见解，我们对竞争模型进行了比较分析，从而突出了最新的表现水平。

    arXiv:2403.12151v1 Announce Type: new  Abstract: Domain-specific knowledge can significantly contribute to addressing a wide variety of vision tasks. However, the generation of such knowledge entails considerable human labor and time costs. This study investigates the potential of Large Language Models (LLMs) in generating and providing domain-specific information through semantic embeddings. To achieve this, an LLM is integrated into a pipeline that utilizes Knowledge Graphs and pre-trained semantic vectors in the context of the Vision-based Zero-shot Object State Classification task. We thoroughly examine the behavior of the LLM through an extensive ablation study. Our findings reveal that the integration of LLM-based embeddings, in combination with general-purpose pre-trained embeddings, leads to substantial performance improvements. Drawing insights from this ablation study, we conduct a comparative analysis against competing models, thereby highlighting the state-of-the-art perfor
    
[^3]: Merino：基于熵驱动的IoT设备上生成式语言模型设计

    Merino: Entropy-driven Design for Generative Language Models on IoT Devices

    [https://arxiv.org/abs/2403.07921](https://arxiv.org/abs/2403.07921)

    在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型，通过最大化transformer解码器的熵来在计算预算内，成功设计了MeRino模型，在移动设置下展现出与当前最先进的自回归transformer模型竞争性能的特点

    

    大规模生成式语言模型（LLMs）作为人工智能现代时代的革命性进步，然而，直接部署LLMs在资源受限的硬件上，比如物联网（IoT）设备，由于其高计算成本而变得困难。在本文中，我们提出了一个新颖的信息熵框架，用于设计手机友好的生成式语言模型。我们的主要设计范式是在给定的计算预算内最大化transformer解码器的熵。整个设计过程涉及解决一个数学规划（MP）问题，可以在几分钟内在CPU上完成，使其几乎是零成本的。我们评估了我们设计的模型MeRino，在九个NLP下游任务上展示了它们在移动设置下对抗当前最先进的自回归transformer模型的竞争性表现。值得注意的是，MeRino在移动设置下获得了类似或更好的零性能表现

    arXiv:2403.07921v1 Announce Type: cross  Abstract: Generative Large Language Models (LLMs) stand as a revolutionary advancement in the modern era of artificial intelligence (AI). However, directly deploying LLMs in resource-constrained hardware, such as Internet-of-Things (IoT) devices, is difficult due to their high computational cost. In this paper, we propose a novel information-entropy framework for designing mobile-friendly generative language models. Our key design paradigm is to maximize the entropy of transformer decoders within the given computational budgets. The whole design procedure involves solving a mathematical programming (MP) problem, which can be done on the CPU within minutes, making it nearly zero-cost. We evaluate our designed models, termed MeRino, across nine NLP downstream tasks, showing their competitive performance against the state-of-the-art autoregressive transformer models under the mobile setting. Notably, MeRino achieves similar or better zero performan
    
[^4]: 通过主题去相关实现多模态人类意图理解去偏见

    Towards Multimodal Human Intention Understanding Debiasing via Subject-Deconfounding

    [https://arxiv.org/abs/2403.05025](https://arxiv.org/abs/2403.05025)

    通过引入概括性因果图和分析主题混淆效应，本文提出了SuCI，实现了多模态人类意图理解的去偏见，解决了MIU模型受主体变异问题困扰的挑战。

    

    arXiv:2403.05025v1 公告类型: 新摘要: 多模态意图理解(MIU)是人类表达分析(例如情感或幽默)不可或缺的组成部分，涉及视觉姿势、语言内容和声学行为等异构模态。现有工作始终专注于设计复杂的结构或融合策略，取得显著进展。然而，它们都受到主题变异问题的困扰，因为不同主题之间的数据分布差异导致。具体而言，由于训练数据中具有不同表达习惯和特征的不同主题，MIU模型很容易被误导，以学习特定于主题的伪相关性，从而显着限制了跨未接触主题的性能和泛化能力。受这一观察启发，我们引入了一个概括性因果图来制定MIU过程，并分析主题的混淆效应。然后，我们提出了SuCI，一个简单而有效的因果

    arXiv:2403.05025v1 Announce Type: new  Abstract: Multimodal intention understanding (MIU) is an indispensable component of human expression analysis (e.g., sentiment or humor) from heterogeneous modalities, including visual postures, linguistic contents, and acoustic behaviors. Existing works invariably focus on designing sophisticated structures or fusion strategies to achieve impressive improvements. Unfortunately, they all suffer from the subject variation problem due to data distribution discrepancies among subjects. Concretely, MIU models are easily misled by distinct subjects with different expression customs and characteristics in the training data to learn subject-specific spurious correlations, significantly limiting performance and generalizability across uninitiated subjects.Motivated by this observation, we introduce a recapitulative causal graph to formulate the MIU procedure and analyze the confounding effect of subjects. Then, we propose SuCI, a simple yet effective caus
    
[^5]: ChatGPT与生物识别技术：对面部识别、性别检测和年龄估计能力的评估

    ChatGPT and biometrics: an assessment of face recognition, gender detection, and age estimation capabilities

    [https://arxiv.org/abs/2403.02965](https://arxiv.org/abs/2403.02965)

    本文评估了ChatGPT在面部识别、性别检测和年龄估计等生物识别任务中的表现，结果显示ChatGPT在面部识别方面具有较高准确性，并在性别检测方面表现显著，在年龄估计任务中也具有相当准确性。

    

    本文探讨了大型语言模型（LLMs），如ChatGPT，在生物识别任务中的应用。我们特别检验了ChatGPT在执行生物识别相关任务方面的能力，重点关注面部识别、性别检测和年龄估计。由于生物识别被视为敏感信息，ChatGPT避免回答直接提示，因此我们设计了提示策略来绕过其保护措施，并评估生物识别任务的能力。我们的研究表明，ChatGPT能够以相当高的准确性识别面部身份并在两个面部图像之间区分。此外，实验结果显示在性别检测方面性能显著，并对年龄估计任务有相当准确性能。我们的发现揭示了在生物识别中应用LLMs和基础模型具有广阔的潜力。

    arXiv:2403.02965v1 Announce Type: cross  Abstract: This paper explores the application of large language models (LLMs), like ChatGPT, for biometric tasks. We specifically examine the capabilities of ChatGPT in performing biometric-related tasks, with an emphasis on face recognition, gender detection, and age estimation. Since biometrics are considered as sensitive information, ChatGPT avoids answering direct prompts, and thus we crafted a prompting strategy to bypass its safeguard and evaluate the capabilities for biometrics tasks. Our study reveals that ChatGPT recognizes facial identities and differentiates between two facial images with considerable accuracy. Additionally, experimental results demonstrate remarkable performance in gender detection and reasonable accuracy for the age estimation tasks. Our findings shed light on the promising potentials in the application of LLMs and foundation models for biometrics.
    
[^6]: 经由对抗攻击和训练的稳健深度强化学习：一项调查

    Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey

    [https://arxiv.org/abs/2403.00420](https://arxiv.org/abs/2403.00420)

    通过对抗性训练来改进DRL对条件变化的鲁棒性，研究者系统分析了当代对抗攻击方法，提供了详细见解。

    

    深度强化学习（DRL）是一种训练自主代理在各种复杂环境中的方法。尽管在众所周知的环境中表现出色，但它仍然容易受到轻微条件变化的影响，引发了人们对其在现实应用中可靠性的担忧。为了提高可用性，DRL必须展示出可信度和鲁棒性。通过对抗性训练提高DRL对条件变化的鲁棒性是一种改进方式，通过训练代理针对环境动态的适当对抗性攻击。我们的工作致力于解决这一关键问题，对当代对抗攻击方法进行了深入分析，系统地对其进行分类，并比较它们的目标和操作机制。这种分类为我们提供了对对抗性攻击如何有效评估DRL代理的恢复力的详细见解，从而为开辟DRL在实际应用中的道路奠定了基础。

    arXiv:2403.00420v1 Announce Type: cross  Abstract: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the 
    
[^7]: 彩虹团队：多样化对抗性提示的开放式生成

    Rainbow Teaming: Open-Ended Generation of Diverse Adversarial Prompts

    [https://arxiv.org/abs/2402.16822](https://arxiv.org/abs/2402.16822)

    Rainbow Teaming提出了一种新方法，通过开放式搜索生成多样化的对抗性提示，可以帮助改善大型语言模型的稳健性，提高安全性，问答和网络安全等领域的模型漏洞。

    

    随着大型语言模型（LLMs）在许多现实世界应用中变得越来越普遍，理解和增强它们对用户输入的稳健性至关重要。现有的用于识别敌对提示的方法往往专注于特定领域，缺乏多样性，或需要大量人工注释。为了解决这些限制，我们提出了彩虹团队，一种用于生成多样化对抗性提示的新方法。彩虹团队将对抗性提示生成视为一个质量 - 多样性问题，并使用开放式搜索来生成既有效又多样的提示。它可以揭示模型在广泛领域内的脆弱性，包括本文中的安全性、问答和网络安全。我们还证明，对由彩虹团队生成的合成数据进行微调可以提高最先进的LLMs的安全性，而不损害它们的一般能力。

    arXiv:2402.16822v1 Announce Type: new  Abstract: As large language models (LLMs) become increasingly prevalent across many real-world applications, understanding and enhancing their robustness to user inputs is of paramount importance. Existing methods for identifying adversarial prompts tend to focus on specific domains, lack diversity, or require extensive human annotations. To address these limitations, we present Rainbow Teaming, a novel approach for producing a diverse collection of adversarial prompts. Rainbow Teaming casts adversarial prompt generation as a quality-diversity problem, and uses open-ended search to generate prompts that are both effective and diverse. It can uncover a model's vulnerabilities across a broad range of domains including, in this paper, safety, question answering, and cybersecurity. We also demonstrate that fine-tuning on synthetic data generated by Rainbow Teaming improves the safety of state-of-the-art LLMs without hurting their general capabilities 
    
[^8]: 加权集成模型是强大的持续学习者

    Weighted Ensemble Models Are Strong Continual Learners

    [https://arxiv.org/abs/2312.08977](https://arxiv.org/abs/2312.08977)

    通过加权集成模型实现了高准确性的持续学习，兼顾可塑性和稳定性。

    

    在本文中，我们研究持续学习（CL）的问题，其中目标是从一系列任务中学习模型，使得以前任务的数据在学习当前任务数据时不可用。CL本质上是在能够学习新任务（即可塑性）和保持先前学习概念的性能（即稳定性）之间取得平衡的过程。为了解决稳定性-可塑性的权衡问题，我们建议对先前和当前任务的模型参数进行加权集成。这种加权集成模型，我们称之为持续模型平均（或CoMA），通过利用可塑性在当前任务上获得高准确性，同时不会偏离太远的先前权重配置，从而确保稳定性。我们还提出了CoMA的改进型变体，名为持续费舍尔加权模型平均（或CoFiMA），该模型对每一个参数进行选择性加权。

    arXiv:2312.08977v2 Announce Type: replace-cross  Abstract: In this work, we study the problem of continual learning (CL) where the goal is to learn a model on a sequence of tasks, such that the data from the previous tasks becomes unavailable while learning on the current task data. CL is essentially a balancing act between being able to learn on the new task (i.e., plasticity) and maintaining the performance on the previously learned concepts (i.e., stability). Intending to address the stability-plasticity trade-off, we propose to perform weight-ensembling of the model parameters of the previous and current tasks. This weighted-ensembled model, which we call Continual Model Averaging (or CoMA), attains high accuracy on the current task by leveraging plasticity, while not deviating too far from the previous weight configuration, ensuring stability. We also propose an improved variant of CoMA, named Continual Fisher-weighted Model Averaging (or CoFiMA), that selectively weighs each para
    
[^9]: AgentMixer: 多智能体相关策略因子分解

    AgentMixer: Multi-Agent Correlated Policy Factorization. (arXiv:2401.08728v1 [cs.MA])

    [http://arxiv.org/abs/2401.08728](http://arxiv.org/abs/2401.08728)

    AgentMixer提出了一种新颖的框架，允许智能体通过策略修改来实现协同决策。通过构造联合策略为各个部分策略的非线性组合，可实现部分可观测智能体的稳定训练和分散执行。

    

    集中式训练与分散式执行（CTDE）广泛应用于通过在训练过程中利用集中式值函数来稳定部分可观察的多智能体强化学习（MARL）。然而，现有方法通常假设智能体基于本地观测独立地做决策，这可能不会导致具有足够协调性的相关联的联合策略。受相关均衡概念的启发，我们提出引入"策略修改"来为智能体提供协调策略的机制。具体地，我们提出了一个新颖的框架AgentMixer，将联合完全可观测策略构造为各个部分可观测策略的非线性组合。为了实现分散式执行，可以通过模仿联合策略来得到各个部分策略。不幸的是，这种模仿学习可能会导致由于联合策略和个体策略之间的不匹配而导致的非对称学习失败。

    Centralized training with decentralized execution (CTDE) is widely employed to stabilize partially observable multi-agent reinforcement learning (MARL) by utilizing a centralized value function during training. However, existing methods typically assume that agents make decisions based on their local observations independently, which may not lead to a correlated joint policy with sufficient coordination. Inspired by the concept of correlated equilibrium, we propose to introduce a \textit{strategy modification} to provide a mechanism for agents to correlate their policies. Specifically, we present a novel framework, AgentMixer, which constructs the joint fully observable policy as a non-linear combination of individual partially observable policies. To enable decentralized execution, one can derive individual policies by imitating the joint policy. Unfortunately, such imitation learning can lead to \textit{asymmetric learning failure} caused by the mismatch between joint policy and indi
    
[^10]: 用大语言模型进行推荐中的表示学习

    Representation Learning with Large Language Models for Recommendation. (arXiv:2310.15950v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.15950](http://arxiv.org/abs/2310.15950)

    这篇论文介绍了一个模型-不可知的框架RLMRec，通过使用大语言模型（LLMs）来增强传统的基于ID的推荐系统，并解决了可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。

    

    推荐系统在深度学习和图神经网络的影响下取得了显著进展，特别是在捕捉复杂的用户-物品关系方面。然而，这些基于图的推荐系统严重依赖于基于ID的数据，可能忽略了与用户和物品相关的有价值的文本信息，导致学到的表示不够富有信息。此外，隐式反馈数据的利用引入了潜在的噪声和偏差，给用户偏好学习的有效性带来了挑战。尽管将大语言模型（LLMs）与传统的基于ID的推荐系统相结合已经引起了人们的关注，但在实际推荐系统中有效实施还需要解决可扩展性问题、仅依赖文本的限制以及提示输入限制等挑战。为了解决这些挑战，我们提出了一个模型不可知的框架RLMRec，旨在通过LLM强化表示来增强现有的推荐系统。

    Recommender systems have seen significant advancements with the influence of deep learning and graph neural networks, particularly in capturing complex user-item relationships. However, these graph-based recommenders heavily depend on ID-based data, potentially disregarding valuable textual information associated with users and items, resulting in less informative learned representations. Moreover, the utilization of implicit feedback data introduces potential noise and bias, posing challenges for the effectiveness of user preference learning. While the integration of large language models (LLMs) into traditional ID-based recommenders has gained attention, challenges such as scalability issues, limitations in text-only reliance, and prompt input constraints need to be addressed for effective implementation in practical recommender systems. To address these challenges, we propose a model-agnostic framework RLMRec that aims to enhance existing recommenders with LLM-empowered representati
    
[^11]: 多物理学预训练用于物理代理模型

    Multiple Physics Pretraining for Physical Surrogate Models. (arXiv:2310.02994v1 [cs.LG])

    [http://arxiv.org/abs/2310.02994](http://arxiv.org/abs/2310.02994)

    多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。

    

    我们引入了一种多物理学预训练（MPP）的方法，这是一种自回归任务不可知的预训练方法，用于物理代理建模。MPP通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。为了有效学习，在这种设置中，我们引入了一种共享嵌入和归一化策略，将多个系统的字段投影到一个共享嵌入空间中。我们在一个涉及流体力学的广泛基准测试中验证了我们方法的有效性。我们表明，单个MPP预训练的变换器能够在所有预训练子任务上与或超过特定任务的基准结果，而无需微调。对于下游任务，我们证明微调MPP训练的模型相较于从头训练的模型，在多个时间步骤上对新物理的预测结果更准确。

    We introduce multiple physics pretraining (MPP), an autoregressive task-agnostic pretraining approach for physical surrogate modeling. MPP involves training large surrogate models to predict the dynamics of multiple heterogeneous physical systems simultaneously by learning features that are broadly useful across diverse physical tasks. In order to learn effectively in this setting, we introduce a shared embedding and normalization strategy that projects the fields of multiple systems into a single shared embedding space. We validate the efficacy of our approach on both pretraining and downstream tasks over a broad fluid mechanics-oriented benchmark. We show that a single MPP-pretrained transformer is able to match or outperform task-specific baselines on all pretraining sub-tasks without the need for finetuning. For downstream tasks, we demonstrate that finetuning MPP-trained models results in more accurate predictions across multiple time-steps on new physics compared to training from
    

