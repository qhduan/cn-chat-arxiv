# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey of Large Language Models for Autonomous Driving.](http://arxiv.org/abs/2311.01043) | 这篇论文概述了自动驾驶技术的发展趋势，从传统的基于规则的系统过渡到基于数据驱动的端到端系统，并介绍了利用大型语言模型与视觉模型相结合来增强自动驾驶系统能力的思路。 |
| [^2] | [Evaluating General-Purpose AI with Psychometrics.](http://arxiv.org/abs/2310.16379) | 本文提出将心理测量学放在评估通用人工智能的核心位置，以解决传统基准的一些挑战和缺点。 |
| [^3] | [Robustness-enhanced Uplift Modeling with Adversarial Feature Desensitization.](http://arxiv.org/abs/2310.04693) | 本文提出了一种增强鲁棒性的提升建模框架RUAD，并通过特征选择和对抗特征抑制两个定制模块更有效地解决了提升模型的特征敏感性问题。 |
| [^4] | [Graph Neural Prompting with Large Language Models.](http://arxiv.org/abs/2309.15427) | 本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。 |
| [^5] | [Maximum Diffusion Reinforcement Learning.](http://arxiv.org/abs/2309.15293) | 最大扩散强化学习是一种克服强化学习中数据相关性问题的方法，通过解耦代理的经验实现持续学习，并在各种测试中表现出色。 |
| [^6] | [Identifying and Mitigating the Security Risks of Generative AI.](http://arxiv.org/abs/2308.14840) | 生成式人工智能技术具有巨大的潜力，但也存在安全风险。这篇论文是一个研讨会的综合报道，讨论了生成式人工智能所带来的双重用途困境，提出了社区在这个领域的短期和长期目标。 |
| [^7] | [Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding.](http://arxiv.org/abs/2308.11234) | 本文提出了一种新的终身多智能体路径规划方法，通过引导智能体避开拥堵路径来优化交通流量，显著提高解决方案质量和总体吞吐量。 |
| [^8] | [Revealing the Underlying Patterns: Investigating Dataset Similarity, Performance, and Generalization.](http://arxiv.org/abs/2308.03580) | 该研究探索了监督深度学习模型的泛化能力和性能，并提出了一种结合距离度量和模型性能的方法，从候选架构中选择适当的模型/架构。结果显示，通过添加少量未见过的图像，可以改善模型的泛化能力。这种方法可以降低训练和标注成本，并在动态环境中提供模型在未见数据上的性能估计。 |
| [^9] | [Can Vision-Language Models be a Good Guesser? Exploring VLMs for Times and Location Reasoning.](http://arxiv.org/abs/2307.06166) | 本研究探索了使用Vision-Language Models（VLMs）进行时间和位置推理的能力，并提出了一个两阶段的识别和推理探测任务来研究VLMs的推理能力。实验发现，尽管VLMs能够有效地识别时间和位置相关特征，但在推理方面仍存在改进的空间。 |
| [^10] | [A Survey on Evaluation of Large Language Models.](http://arxiv.org/abs/2307.03109) | 本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。 |

# 详细

[^1]: 自动驾驶的大型语言模型概述

    A Survey of Large Language Models for Autonomous Driving. (arXiv:2311.01043v1 [cs.AI])

    [http://arxiv.org/abs/2311.01043](http://arxiv.org/abs/2311.01043)

    这篇论文概述了自动驾驶技术的发展趋势，从传统的基于规则的系统过渡到基于数据驱动的端到端系统，并介绍了利用大型语言模型与视觉模型相结合来增强自动驾驶系统能力的思路。

    

    自动驾驶技术作为改变交通和城市流动性的催化剂，正趋向于从基于规则的系统转向基于数据驱动的策略。传统的模块化系统受到级联模块中的累积误差和不灵活的预设规则的限制。相比之下，端到端自动驾驶系统通过完全数据驱动的训练过程，有潜力避免错误累积，尽管由于其黑盒性质，它们往往缺乏透明度，使得决策的验证和可追溯性变得复杂。近期，大型语言模型（LLMs）展示了理解背景、逻辑推理和生成答案等能力。自然而然的想法是利用这些能力赋予自动驾驶以更强大的能力。通过将LLM与基础视觉模型结合，可能打开对开放世界理解、推理和少样本学习的大门，这是当前自动驾驶系统所缺乏的。

    Autonomous driving technology, a catalyst for revolutionizing transportation and urban mobility, has the tend to transition from rule-based systems to data-driven strategies. Traditional module-based systems are constrained by cumulative errors among cascaded modules and inflexible pre-set rules. In contrast, end-to-end autonomous driving systems have the potential to avoid error accumulation due to their fully data-driven training process, although they often lack transparency due to their ``black box" nature, complicating the validation and traceability of decisions. Recently, large language models (LLMs) have demonstrated abilities including understanding context, logical reasoning, and generating answers. A natural thought is to utilize these abilities to empower autonomous driving. By combining LLM with foundation vision models, it could open the door to open-world understanding, reasoning, and few-shot learning, which current autonomous driving systems are lacking. In this paper,
    
[^2]: 用心理测量学评估通用人工智能

    Evaluating General-Purpose AI with Psychometrics. (arXiv:2310.16379v1 [cs.AI])

    [http://arxiv.org/abs/2310.16379](http://arxiv.org/abs/2310.16379)

    本文提出将心理测量学放在评估通用人工智能的核心位置，以解决传统基准的一些挑战和缺点。

    

    人工智能（AI）已经从特定任务向通用系统的发展，趋向于人类的多功能性。随着AI系统开始在社会中发挥重要作用，确保对其进行充分评估变得很重要。目前的AI基准通常在特定任务集合上评估性能。然而，对于评估通用AI系统来说，这有一些缺点。首先，很难预测AI系统是否能完成一项它从未见过或之前不存在的新任务。其次，这些基准常常关注整体性能指标，可能忽视了对做出明智决策至关重要的细节。最后，对现有基准的可靠性存在越来越多的担忧，并对正在进行的测量提出了疑问。为解决这些挑战，本文建议将心理测量学，即心理测量的科学，放在评估通用AI的核心位置。

    Artificial intelligence (AI) has witnessed an evolution from task-specific to general-purpose systems that trend toward human versatility. As AI systems begin to play pivotal roles in society, it is important to ensure that they are adequately evaluated. Current AI benchmarks typically assess performance on collections of specific tasks. This has drawbacks when used for assessing general-purpose AI systems. First, it is difficult to predict whether AI systems could complete a new task it has never seen or that did not previously exist. Second, these benchmarks often focus on overall performance metrics, potentially overlooking the finer details crucial for making informed decisions. Lastly, there are growing concerns about the reliability of existing benchmarks and questions about what is being measured. To solve these challenges, this paper suggests that psychometrics, the science of psychological measurement, should be placed at the core of evaluating general-purpose AI. Psychometric
    
[^3]: 增强鲁棒性的带对抗特征抑制的提升建模

    Robustness-enhanced Uplift Modeling with Adversarial Feature Desensitization. (arXiv:2310.04693v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.04693](http://arxiv.org/abs/2310.04693)

    本文提出了一种增强鲁棒性的提升建模框架RUAD，并通过特征选择和对抗特征抑制两个定制模块更有效地解决了提升模型的特征敏感性问题。

    

    提升建模在在线营销中展示了非常有希望的结果。然而，大多数现有的工作在一些实际应用中容易受到鲁棒性挑战的影响。本文首先对上述现象给出了一个可能的解释。我们使用不同的真实世界数据集验证了在线营销中存在特征敏感性问题，一些关键特征的扰动会严重影响提升模型的性能，甚至导致相反的趋势。为了解决上述问题，我们提出了一种新颖的通过对抗特征抑制增强鲁棒性的提升建模框架（RUAD）。具体来说，我们的RUAD通过两个定制模块更有效地减轻提升模型的特征敏感性，包括一个具有联合多标签建模的特征选择模块，以从输入特征中识别一个关键子集，以及一个采用对抗训练和软插值操作的对抗特征抑制模块。

    Uplift modeling has shown very promising results in online marketing. However, most existing works are prone to the robustness challenge in some practical applications. In this paper, we first present a possible explanation for the above phenomenon. We verify that there is a feature sensitivity problem in online marketing using different real-world datasets, where the perturbation of some key features will seriously affect the performance of the uplift model and even cause the opposite trend. To solve the above problem, we propose a novel robustness-enhanced uplift modeling framework with adversarial feature desensitization (RUAD). Specifically, our RUAD can more effectively alleviate the feature sensitivity of the uplift model through two customized modules, including a feature selection module with joint multi-label modeling to identify a key subset from the input features and an adversarial feature desensitization module using adversarial training and soft interpolation operations t
    
[^4]: 使用大型语言模型的图神经提示

    Graph Neural Prompting with Large Language Models. (arXiv:2309.15427v1 [cs.CL])

    [http://arxiv.org/abs/2309.15427](http://arxiv.org/abs/2309.15427)

    本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。

    

    大型语言模型（LLMs）在各种语言建模任务中表现出了卓越的泛化能力和出色的性能，但它们在准确捕捉和返回基于知识的信息方面仍存在固有限制。现有的研究已经探索了利用知识图来通过联合训练和定制模型架构增强语言建模，但是将此应用于LLMs存在参数数量庞大和计算成本高的问题。此外，如何利用预训练的LLMs并避免从头开始训练自定义模型仍然是一个开放的问题。在这项工作中，我们提出了图神经提示（GNP），一种新颖的即插即用方法，可以帮助预训练的LLMs从知识图中学习有益的知识。GNP包括各种设计，包括标准的图神经网络编码器、跨模态汇聚模块、域投影器和自监督链接预测目标。在多个实验中展示了GNP的有效性。

    Large Language Models (LLMs) have shown remarkable generalization capability with exceptional performance in various language modeling tasks. However, they still exhibit inherent limitations in precisely capturing and returning grounded knowledge. While existing work has explored utilizing knowledge graphs to enhance language modeling via joint training and customized model architectures, applying this to LLMs is problematic owing to their large number of parameters and high computational cost. In addition, how to leverage the pre-trained LLMs and avoid training a customized model from scratch remains an open question. In this work, we propose Graph Neural Prompting (GNP), a novel plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from KGs. GNP encompasses various designs, including a standard graph neural network encoder, a cross-modality pooling module, a domain projector, and a self-supervised link prediction objective. Extensive experiments on multiple
    
[^5]: 最大扩散强化学习

    Maximum Diffusion Reinforcement Learning. (arXiv:2309.15293v1 [cs.LG])

    [http://arxiv.org/abs/2309.15293](http://arxiv.org/abs/2309.15293)

    最大扩散强化学习是一种克服强化学习中数据相关性问题的方法，通过解耦代理的经验实现持续学习，并在各种测试中表现出色。

    

    所有机器学习都建立在数据独立且同分布的假设上。然而，在强化学习中，当数据是依次从代理经验中收集而来时，这一假设通常不成立。因此，我们提出了一种名为最大扩散强化学习的方法，利用统计力学中的遍历过程来克服这些限制。我们的方法通过解耦代理的经验，可证明地使代理在单次部署中能够持续学习，而不受初始化方式的影响。此外，我们证明了我们的方法推广了众所周知的最大熵技术，并且通过在流行的基准测试中稳定超过了最先进的性能水平。我们的研究成果极大地促进了物理学、学习和控制的交叉领域，为强化学习代理（如行走机器人和自动驾驶汽车）的透明可靠决策提供了一条道路。

    The assumption that data are independent and identically distributed underpins all machine learning. When data are collected sequentially from agent experiences this assumption does not generally hold, as in reinforcement learning. Here, we derive a method that overcomes these limitations by exploiting the statistical mechanics of ergodic processes, which we term maximum diffusion reinforcement learning. By decorrelating agent experiences, our approach provably enables agents to learn continually in single-shot deployments regardless of how they are initialized. Moreover, we prove our approach generalizes well-known maximum entropy techniques, and show that it robustly exceeds state-of-the-art performance across popular benchmarks. Our results at the nexus of physics, learning, and control pave the way towards more transparent and reliable decision-making in reinforcement learning agents, such as locomoting robots and self-driving cars.
    
[^6]: 识别和减轻生成式人工智能的安全风险

    Identifying and Mitigating the Security Risks of Generative AI. (arXiv:2308.14840v1 [cs.AI])

    [http://arxiv.org/abs/2308.14840](http://arxiv.org/abs/2308.14840)

    生成式人工智能技术具有巨大的潜力，但也存在安全风险。这篇论文是一个研讨会的综合报道，讨论了生成式人工智能所带来的双重用途困境，提出了社区在这个领域的短期和长期目标。

    

    每一项重大技术发明都会带来双重用途的困境 - 新技术既有可能被用于善良，也可能被用于恶意行为。生成式人工智能（GenAI）技术，如大型语言模型（LLM）和扩散模型，展示了卓越的能力（例如上下文学习，代码补全，文本到图像的生成和编辑）。然而，攻击者同样可以利用GenAI生成新的攻击，并增加现有攻击的速度和有效性。本文报告了在Google举办的一个研讨会的发现（由斯坦福大学和威斯康星大学麦迪逊分校共同组织）。本文并不意味着全面，而是试图综合一些有趣的研讨会发现。我们讨论了这个主题的短期和长期目标。我们希望这篇论文既为这个重要主题的讨论提供一个起点，也引起兴趣。

    Every major technical invention resurfaces the dual-use dilemma -- the new technology has the potential to be used for good as well as for harm. Generative AI (GenAI) techniques, such as large language models (LLMs) and diffusion models, have shown remarkable capabilities (e.g., in-context learning, code-completion, and text-to-image generation and editing). However, GenAI can be used just as well by attackers to generate new attacks and increase the velocity and efficacy of existing attacks.  This paper reports the findings of a workshop held at Google (co-organized by Stanford University and the University of Wisconsin-Madison) on the dual-use dilemma posed by GenAI. This paper is not meant to be comprehensive, but is rather an attempt to synthesize some of the interesting findings from the workshop. We discuss short-term and long-term goals for the community on this topic. We hope this paper provides both a launching point for a discussion on this important topic as well as interest
    
[^7]: 交通流量优化的终身多智能体路径规划

    Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding. (arXiv:2308.11234v1 [cs.AI])

    [http://arxiv.org/abs/2308.11234](http://arxiv.org/abs/2308.11234)

    本文提出了一种新的终身多智能体路径规划方法，通过引导智能体避开拥堵路径来优化交通流量，显著提高解决方案质量和总体吞吐量。

    

    多智能体路径规划(MAPF)是机器人领域的一个基本问题，要求为一个团队的智能体计算无碰撞路径，所有智能体都在共享地图上移动。尽管有许多相关研究，但当前的算法在智能体数量增加时都会遇到困难。主要原因是现有方法通常规划自由流动的最优路径，这会导致拥堵。为了解决这个问题，我们提出了一种新的MAPF方法，通过跟随避免拥堵的路径来引导智能体到达目的地。我们在两个大规模场景中评估了这个想法：一次性MAPF，每个智能体只有一个目的地，以及终身MAPF，智能体不断被分配新任务。对于一次性MAPF，我们展示了我们的方法大大提高了解决方案的质量。对于终身MAPF，我们报告了总体吞吐量的大幅提升。

    Multi-Agent Path Finding (MAPF) is a fundamental problem in robotics that asks us to compute collision-free paths for a team of agents, all moving across a shared map. Although many works appear on this topic, all current algorithms struggle as the number of agents grows. The principal reason is that existing approaches typically plan free-flow optimal paths, which creates congestion. To tackle this issue we propose a new approach for MAPF where agents are guided to their destination by following congestion-avoiding paths. We evaluate the idea in two large-scale settings: one-shot MAPF, where each agent has a single destination, and lifelong MAPF, where agents are continuously assigned new tasks. For one-shot MAPF we show that our approach substantially improves solution quality. For Lifelong MAPF we report large improvements in overall throughput.
    
[^8]: 揭示潜在模式：研究数据集的相似性、性能和泛化能力

    Revealing the Underlying Patterns: Investigating Dataset Similarity, Performance, and Generalization. (arXiv:2308.03580v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2308.03580](http://arxiv.org/abs/2308.03580)

    该研究探索了监督深度学习模型的泛化能力和性能，并提出了一种结合距离度量和模型性能的方法，从候选架构中选择适当的模型/架构。结果显示，通过添加少量未见过的图像，可以改善模型的泛化能力。这种方法可以降低训练和标注成本，并在动态环境中提供模型在未见数据上的性能估计。

    

    监督深度学习模型需要大量标记数据才能在特定任务上取得可接受的性能。然而，当在未见过的数据上进行测试时，模型可能表现不佳。因此，需要用额外和多样化的标记数据来训练模型以提高泛化能力。本研究旨在理解模型、它们的性能和泛化能力。我们建立了图像-图像、数据集-数据集和图像-数据集距离，以洞察模型的行为。我们提出的距离度量方法结合模型性能可以帮助从候选架构中选择一个合适的模型/架构。我们发现，只需将少量未见过的图像（如1、3或7个）添加到训练集中即可改善这些模型的泛化能力。我们提出的方法可以在动态环境中减少训练和标注成本，并提供模型在未见数据上的性能估计。

    Supervised deep learning models require significant amount of labelled data to achieve an acceptable performance on a specific task. However, when tested on unseen data, the models may not perform well. Therefore, the models need to be trained with additional and varying labelled data to improve the generalization. In this work, our goal is to understand the models, their performance and generalization. We establish image-image, dataset-dataset, and image-dataset distances to gain insights into the model's behavior. Our proposed distance metric when combined with model performance can help in selecting an appropriate model/architecture from a pool of candidate architectures. We have shown that the generalization of these models can be improved by only adding a small number of unseen images (say 1, 3 or 7) into the training set. Our proposed approach reduces training and annotation costs while providing an estimate of model performance on unseen data in dynamic environments.
    
[^9]: Vision-Language Models能成为良好猜测器吗？探索VLMs用于时间和位置推理

    Can Vision-Language Models be a Good Guesser? Exploring VLMs for Times and Location Reasoning. (arXiv:2307.06166v1 [cs.CV])

    [http://arxiv.org/abs/2307.06166](http://arxiv.org/abs/2307.06166)

    本研究探索了使用Vision-Language Models（VLMs）进行时间和位置推理的能力，并提出了一个两阶段的识别和推理探测任务来研究VLMs的推理能力。实验发现，尽管VLMs能够有效地识别时间和位置相关特征，但在推理方面仍存在改进的空间。

    

    期望Vision-Language Models（VLMs）能像人一样具备常识知识进行推理。一个例子是，人类可以根据他们的知识推断出一张图片的拍摄地点和时间。这让我们想知道，基于视觉线索，使用大规模图像-文本资源进行预训练的Vision-Language Models是否能够达到甚至超过人类在时间和位置推理方面的能力。为了回答这个问题，我们提出了一个两阶段的识别和推理探测任务，应用于鉴别性和生成性的VLMs，以发现VLMs能否识别出与时间和位置相关的特征，并进一步进行推理。为了方便这项研究，我们引入了WikiTiLo，一个包含丰富社会文化线索的精心策划的图像数据集。在广泛的实验研究中，我们发现虽然VLMs能够有效地保留视觉编码器中的相关特征，但在推理方面仍存在不完善的问题。我们将发布...

    Vision-Language Models (VLMs) are expected to be capable of reasoning with commonsense knowledge as human beings. One example is that humans can reason where and when an image is taken based on their knowledge. This makes us wonder if, based on visual cues, Vision-Language Models that are pre-trained with large-scale image-text resources can achieve and even outperform human's capability in reasoning times and location. To address this question, we propose a two-stage \recognition\space and \reasoning\space probing task, applied to discriminative and generative VLMs to uncover whether VLMs can recognize times and location-relevant features and further reason about it. To facilitate the investigation, we introduce WikiTiLo, a well-curated image dataset compromising images with rich socio-cultural cues. In the extensive experimental studies, we find that although VLMs can effectively retain relevant features in visual encoders, they still fail to make perfect reasoning. We will release o
    
[^10]: 对大型语言模型评估的调查

    A Survey on Evaluation of Large Language Models. (arXiv:2307.03109v1 [cs.CL])

    [http://arxiv.org/abs/2307.03109](http://arxiv.org/abs/2307.03109)

    本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。

    

    大型语言模型（LLMs）由于在各种应用中表现出的前所未有的性能而在学术界和工业界越来越受欢迎。随着LLMs在研究和日常使用中继续发挥着重要作用，它们的评估变得越来越关键，不仅在任务水平上，而且在社会层面上，以更好地了解它们的潜在风险。在过去的几年里，已经做出了相当大的努力来从不同的角度来研究LLMs。本文综述了LLMs的这些评估方法，重点关注三个关键维度：评估什么、在哪里评估以及如何评估。首先，我们从评估任务的角度提供了一个概述，涵盖了一般的自然语言处理任务、推理、医学应用、伦理学、教育、自然科学和社会科学、代理应用和其他领域。其次，我们通过深入探讨评估方法和基准答案来回答“在哪里”和“如何”这两个问题。

    Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level, but also at the society level for better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives. This paper presents a comprehensive review of these evaluation methods for LLMs, focusing on three key dimensions: what to evaluate, where to evaluate, and how to evaluate. Firstly, we provide an overview from the perspective of evaluation tasks, encompassing general natural language processing tasks, reasoning, medical usage, ethics, educations, natural and social sciences, agent applications, and other areas. Secondly, we answer the `where' and `how' questions by diving into the evaluation methods and bench
    

