# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^2] | [Universal representations for financial transactional data: embracing local, global, and external contexts](https://arxiv.org/abs/2404.02047) | 提出了一个金融交易数据通用表示的学习框架，结合了本地、全局和外部语境，提出了新颖的生成模型和整合外部信息的方法，并在本地任务中表现出超越性能。 |
| [^3] | [BirdSet: A Multi-Task Benchmark for Classification in Avian Bioacoustics](https://arxiv.org/abs/2403.10380) | 提出了BirdSet基准，用于鸟类生物声学中的分类任务，整合开源鸟类录音数据集合，全面评估模型性能和识别潜在不足。 |
| [^4] | [DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation](https://arxiv.org/abs/2403.07733) | 通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。 |
| [^5] | [Large Language Models are Advanced Anonymizers](https://arxiv.org/abs/2402.13846) | 大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。 |
| [^6] | [Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement](https://arxiv.org/abs/2402.11060) | 介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。 |
| [^7] | [Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data](https://arxiv.org/abs/2402.06104) | 该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。 |
| [^8] | [Limits of Large Language Models in Debating Humans](https://arxiv.org/abs/2402.06049) | 大型语言模型在与人类辩论中的能力有限，尽管它们能够融入和促进人类的工作效率，但在辩论中的说服力较弱。在成为可行的辩手之前，LLMs需要进一步发展。 |
| [^9] | [Group Distributionally Robust Dataset Distillation with Risk Minimization](https://arxiv.org/abs/2402.04676) | 这项研究关注数据集蒸馏与其泛化能力的关系，尤其是在面对不常见的子组的样本时，如何确保模型在合成数据集上的训练可以表现良好。 |
| [^10] | [Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization](https://arxiv.org/abs/2311.18703) | 该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。 |
| [^11] | [Make a Donut: Hierarchical EMD-Space Planning for Zero-Shot Deformable Manipulation with Tools](https://arxiv.org/abs/2311.02787) | 引入了一种无需演示的分层规划方法，利用大型语言模型来解决复杂的长时间任务，为每个阶段提供工具名称和Python代码。 |
| [^12] | [Do LLMs Dream of Ontologies?.](http://arxiv.org/abs/2401.14931) | 本文研究了通用预训练大型语言模型（LLMs）是否记忆了已知本体论的信息以及记忆的程度，结果显示LLMs部分地了解本体论的概念，记忆程度与其在Web上的流行程度成正比。 |
| [^13] | [Towards the Vulnerability of Watermarking Artificial Intelligence Generated Content.](http://arxiv.org/abs/2310.07726) | 该研究探讨了将水印技术应用于人工智能生成内容的漏洞，并证明了现有的水印机制容易被对手破解。 |
| [^14] | [A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises.](http://arxiv.org/abs/2306.04802) | 本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。 |

# 详细

[^1]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^2]: 金融交易数据的通用表示：融合本地、全局和外部语境

    Universal representations for financial transactional data: embracing local, global, and external contexts

    [https://arxiv.org/abs/2404.02047](https://arxiv.org/abs/2404.02047)

    提出了一个金融交易数据通用表示的学习框架，结合了本地、全局和外部语境，提出了新颖的生成模型和整合外部信息的方法，并在本地任务中表现出超越性能。

    

    金融交易的有效处理对银行数据分析至关重要。然而，在这一领域中，大多数方法专注于为独立问题提供专门化解决方案，而不是构建适用于许多问题的通用表示。我们提出了一个表示学习框架，旨在解决各种企业挑战。我们还提出了考虑数据特定性的新颖生成模型，并提出了一种整合外部信息到客户表示的方式，借鉴其他客户行动的见解。最后，我们提供了一个基准，描述了全球范围内的表示质量，涉及整个交易历史；本地范围内，反映客户当前状态；动态范围内，捕捉表示随时间演变的情况。我们的生成方法在本地任务中表现出色，对于下一个MCC预测任务的ROC-AUC提升高达14％，对于dow...

    arXiv:2404.02047v1 Announce Type: cross  Abstract: Effective processing of financial transactions is essential for banking data analysis. However, in this domain, most methods focus on specialized solutions to stand-alone problems instead of constructing universal representations suitable for many problems. We present a representation learning framework that addresses diverse business challenges. We also suggest novel generative models that account for data specifics, and a way to integrate external information into a client's representation, leveraging insights from other customers' actions. Finally, we offer a benchmark, describing representation quality globally, concerning the entire transaction history; locally, reflecting the client's current state; and dynamically, capturing representation evolution over time. Our generative approach demonstrates superior performance in local tasks, with an increase in ROC-AUC of up to 14\% for the next MCC prediction task and up to 46\% for dow
    
[^3]: BirdSet：鸟类生物声学分类的多任务基准

    BirdSet: A Multi-Task Benchmark for Classification in Avian Bioacoustics

    [https://arxiv.org/abs/2403.10380](https://arxiv.org/abs/2403.10380)

    提出了BirdSet基准，用于鸟类生物声学中的分类任务，整合开源鸟类录音数据集合，全面评估模型性能和识别潜在不足。

    

    深度学习模型已经成为鸟类生物声学领域诊断环境健康和生物多样性的强大工具，但研究中存在的不一致性给这一领域的进展带来了显著挑战。我们提出了BirdSet基准，一个统一的框架，综合研究努力，以全面分类鸟类鸣叫声。BirdSet将开源鸟类录音整合到一个精心策划的数据集合中，提供对模型性能的深入理解，并识别跨不同研究的潜在不足之处。

    arXiv:2403.10380v1 Announce Type: cross  Abstract: Deep learning (DL) models have emerged as a powerful tool in avian bioacoustics to diagnose environmental health and biodiversity. However, inconsistencies in research pose notable challenges hindering progress in this domain. Reliable DL models need to analyze bird calls flexibly across various species and environments to fully harness the potential of bioacoustics in a cost-effective passive acoustic monitoring scenario. Data fragmentation and opacity across studies complicate a comprehensive evaluation of general model performance. To overcome these challenges, we present the BirdSet benchmark, a unified framework consolidating research efforts with a holistic approach for classifying bird vocalizations in avian bioacoustics. BirdSet harmonizes open-source bird recordings into a curated dataset collection. This unified approach provides an in-depth understanding of model performance and identifies potential shortcomings across diffe
    
[^4]: DSEG-LIME -- 通过层次化数据驱动分割提升图像解释能力

    DSEG-LIME -- Improving Image Explanation by Hierarchical Data-Driven Segmentation

    [https://arxiv.org/abs/2403.07733](https://arxiv.org/abs/2403.07733)

    通过引入数据驱动分割和层次分割程序，DSEG-LIME改进了图像解释能力，提高了图像分类的可解释性。

    

    可解释的人工智能在揭示复杂机器学习模型的决策过程中至关重要。LIME (Local Interpretable Model-agnostic Explanations) 是一个广为人知的用于图像分析的XAI框架。它利用图像分割来创建特征以识别相关的分类区域。然而，较差的分割可能会影响解释的一致性并削弱各个区域的重要性，从而影响整体的可解释性。针对这些挑战，我们引入了DSEG-LIME (Data-Driven Segmentation LIME)，具有: i) 用于生成人类可识别特征的数据驱动分割, 和 ii) 通过组合实现的层次分割程序。我们在预训练模型上使用来自ImageNet数据集的图像对DSEG-LIME进行基准测试-这些情景不包含特定领域的知识。分析包括使用已建立的XAI指标进行定量评估，以及进一步的定性评估。

    arXiv:2403.07733v1 Announce Type: cross  Abstract: Explainable Artificial Intelligence is critical in unraveling decision-making processes in complex machine learning models. LIME (Local Interpretable Model-agnostic Explanations) is a well-known XAI framework for image analysis. It utilizes image segmentation to create features to identify relevant areas for classification. Consequently, poor segmentation can compromise the consistency of the explanation and undermine the importance of the segments, affecting the overall interpretability. Addressing these challenges, we introduce DSEG-LIME (Data-Driven Segmentation LIME), featuring: i) a data-driven segmentation for human-recognized feature generation, and ii) a hierarchical segmentation procedure through composition. We benchmark DSEG-LIME on pre-trained models with images from the ImageNet dataset - scenarios without domain-specific knowledge. The analysis includes a quantitative evaluation using established XAI metrics, complemented
    
[^5]: 大型语言模型是先进的匿名化工具

    Large Language Models are Advanced Anonymizers

    [https://arxiv.org/abs/2402.13846](https://arxiv.org/abs/2402.13846)

    大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。

    

    最近在隐私研究领域对大型语言模型的研究表明，它们在推断真实世界在线文本中的个人数据方面表现出接近人类水平的性能。随着模型能力的不断增强，现有的文本匿名化方法当前已经落后于监管要求和对抗威胁。这引出了一个问题：个人如何有效地保护他们在分享在线文本时的个人数据。在这项工作中，我们采取了两步来回答这个问题：首先，我们提出了一个新的设置，用于评估面对对抗性LLM的推断时的匿名化效果，从而允许自然地测量匿名化性能，同时纠正了以前指标的一些缺陷。然后，我们提出了基于LLM的对抗性匿名化框架，利用LLM的强大推断能力来指导我们的匿名化过程。在我们的实验评估中，我们展示了在真实世界中的匿名化实践。

    arXiv:2402.13846v1 Announce Type: cross  Abstract: Recent work in privacy research on large language models has shown that they achieve near human-level performance at inferring personal data from real-world online texts. With consistently increasing model capabilities, existing text anonymization methods are currently lacking behind regulatory requirements and adversarial threats. This raises the question of how individuals can effectively protect their personal data in sharing online texts. In this work, we take two steps to answer this question: We first present a new setting for evaluating anonymizations in the face of adversarial LLMs inferences, allowing for a natural measurement of anonymization performance while remedying some of the shortcomings of previous metrics. We then present our LLM-based adversarial anonymization framework leveraging the strong inferential capabilities of LLMs to inform our anonymization procedure. In our experimental evaluation, we show on real-world 
    
[^6]: Persona-DB：用于响应预测的高效大规模语言模型个性化与协同数据优化

    Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement

    [https://arxiv.org/abs/2402.11060](https://arxiv.org/abs/2402.11060)

    介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。

    

    随着对大型语言模型（LLMs）个性化交互需求的增加，需要开发能够准确快速识别用户意见和偏好的方法。检索增强作为一种有效策略出现，因为它可以适应大量用户而无需进行微调的成本。然而，现有研究主要集中在增强检索阶段，并对数据库表示的优化进行了有限的探索，这是个性化等任务的关键方面。在这项工作中，我们从一个新的角度研究了这个问题，着重于如何更有效地表示数据，以便在LLM定制的情境下更有效地进行检索。为了解决这一挑战，我们介绍了Persona-DB，这是一个简单而有效的框架，包括一个分层构建过程，以改善跨任务背景的泛化能力，并进行协同优化。

    arXiv:2402.11060v1 Announce Type: cross  Abstract: The increasing demand for personalized interactions with large language models (LLMs) calls for the development of methodologies capable of accurately and efficiently identifying user opinions and preferences. Retrieval augmentation emerges as an effective strategy, as it can accommodate a vast number of users without the costs from fine-tuning. Existing research, however, has largely focused on enhancing the retrieval stage and devoted limited exploration toward optimizing the representation of the database, a crucial aspect for tasks such as personalization. In this work, we examine the problem from a novel angle, focusing on how data can be better represented for more efficient retrieval in the context of LLM customization. To tackle this challenge, we introduce Persona-DB, a simple yet effective framework consisting of a hierarchical construction process to improve generalization across task contexts and collaborative refinement to
    
[^7]: 功能对齐回归：一种从数据中明确学习函数导数的方法

    Function Aligned Regression: A Method Explicitly Learns Functional Derivatives from Data

    [https://arxiv.org/abs/2402.06104](https://arxiv.org/abs/2402.06104)

    该论文提出了一种名为FAR的方法，通过捕捉函数导数来更好、更高效地拟合底层真实函数。在合成数据集和八个真实世界任务中证明了该方法的有效性。

    

    回归是机器学习中的一个基本任务，在过去几十年中引起了广泛关注。传统的回归方法主要通过使用损失函数来将模型预测与每个个体数据样本的真实值对齐，然而，我们发现这种方法可能导致在不同样本之间关系的预测不够优化。近期的研究工作引入了标签相似性信息来改进回归方法，但在完全捕捉底层真实函数的复杂性方面仍存在明显的差距。在本文中，我们提出了FAR（功能对齐回归）作为一种更好、更高效的解决方案，通过捕捉函数导数来拟合底层真实函数。我们在两个合成数据集和六个领域的八个大规模真实世界任务中验证了该方法的有效性。

    Regression is a fundamental task in machine learning that has garnered extensive attention over the past decades. The conventional approach for regression involves employing loss functions that primarily concentrate on aligning model prediction with the ground truth for each individual data sample, which, as we show, can result in sub-optimal prediction of the relationships between the different samples. Recent research endeavors have introduced novel perspectives by incorporating label similarity information to regression. However, a notable gap persists in these approaches when it comes to fully capturing the intricacies of the underlying ground truth function. In this work, we propose FAR (Function Aligned Regression) as a arguably better and more efficient solution to fit the underlying function of ground truth by capturing functional derivatives. We demonstrate the effectiveness of the proposed method practically on 2 synthetic datasets and on 8 extensive real-world tasks from 6 b
    
[^8]: 大型语言模型在与人类辩论中的局限性

    Limits of Large Language Models in Debating Humans

    [https://arxiv.org/abs/2402.06049](https://arxiv.org/abs/2402.06049)

    大型语言模型在与人类辩论中的能力有限，尽管它们能够融入和促进人类的工作效率，但在辩论中的说服力较弱。在成为可行的辩手之前，LLMs需要进一步发展。

    

    大型语言模型(LLMs)在与人类的互动中展现出了显著的潜力。随后，将它们作为人工代表和替代品进行社会学实验的潜在应用是一个令人激动的前景。但是这个想法有多可行呢？本文试图通过一项预先注册的研究来测试现阶段LLMs的局限性，该研究将真实的人类与扮演人类的LLM代理结合起来。本研究着重探讨辩论为基础的意见共识形成在三种环境下的情况：仅人类、代理和人类、仅代理。我们的目标是理解LLM代理对人类的影响，并评估它们在辩论方面的能力是否与人类相似。我们发现LLMs能够融入并促进人类的工作效率，但在辩论中的说服力较弱，最终行为与人类有所偏离。我们阐明了这些主要缺陷，并预计在成为可行的辩手之前，LLMs必须进一步发展。

    Large Language Models (LLMs) have shown remarkable promise in their ability to interact proficiently with humans. Subsequently, their potential use as artificial confederates and surrogates in sociological experiments involving conversation is an exciting prospect. But how viable is this idea? This paper endeavors to test the limits of current-day LLMs with a pre-registered study integrating real people with LLM agents acting as people. The study focuses on debate-based opinion consensus formation in three environments: humans only, agents and humans, and agents only. Our goal is to understand how LLM agents influence humans, and how capable they are in debating like humans. We find that LLMs can blend in and facilitate human productivity but are less convincing in debate, with their behavior ultimately deviating from human's. We elucidate these primary failings and anticipate that LLMs must evolve further before being viable debaters.
    
[^9]: 带风险最小化的分组分布鲁棒数据集蒸馏

    Group Distributionally Robust Dataset Distillation with Risk Minimization

    [https://arxiv.org/abs/2402.04676](https://arxiv.org/abs/2402.04676)

    这项研究关注数据集蒸馏与其泛化能力的关系，尤其是在面对不常见的子组的样本时，如何确保模型在合成数据集上的训练可以表现良好。

    

    数据集蒸馏（DD）已成为一种广泛采用的技术，用于构建一个合成数据集，该数据集在捕捉训练数据集的基本信息方面起到重要作用，从而方便准确训练神经模型。其应用涵盖了转移学习、联邦学习和神经架构搜索等各个领域。构建合成数据的最流行方法依赖于使模型在合成数据集和训练数据集上的收敛性能相匹配。然而，目标是将训练数据集视为辅助，就像训练集是人口分布的近似替代品一样，而后者才是我们感兴趣的数据。尽管其受欢迎程度很高，但尚未探索的一个方面是DD与其泛化能力的关系，特别是跨不常见的子组。也就是说，当面对来自罕见子组的样本时，我们如何确保在合成数据集上训练的模型表现良好。

    Dataset distillation (DD) has emerged as a widely adopted technique for crafting a synthetic dataset that captures the essential information of a training dataset, facilitating the training of accurate neural models. Its applications span various domains, including transfer learning, federated learning, and neural architecture search. The most popular methods for constructing the synthetic data rely on matching the convergence properties of training the model with the synthetic dataset and the training dataset. However, targeting the training dataset must be thought of as auxiliary in the same sense that the training set is an approximate substitute for the population distribution, and the latter is the data of interest. Yet despite its popularity, an aspect that remains unexplored is the relationship of DD to its generalization, particularly across uncommon subgroups. That is, how can we ensure that a model trained on the synthetic dataset performs well when faced with samples from re
    
[^10]: 通过熵率最小化实现可预测的强化学习动态

    Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization

    [https://arxiv.org/abs/2311.18703](https://arxiv.org/abs/2311.18703)

    该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。

    

    在强化学习中，智能体没有动机展示可预测的行为，通常通过策略熵正则化推动智能体在探索上随机化其行为。从人的角度来看，这使得强化学习智能体很难解释和预测；从安全角度来看，更难以进行形式化验证。我们提出了一种新的方法，称为可预测性感知强化学习（PA-RL），用于引导智能体展现可预测的行为，其利用状态序列熵率作为可预测性度量。我们展示了如何将熵率制定为平均奖励目标，并且由于其熵奖励函数依赖于策略，我们引入了一个动作相关的替代熵，以利用PG方法。我们证明了最小化平均替代奖励的确定性策略存在，并且最小化了实际熵率。我们还展示了如何在学习到的动态模型的基础上近似计算与值函数。

    In Reinforcement Learning (RL), agents have no incentive to exhibit predictable behaviors, and are often pushed (through e.g. policy entropy regularization) to randomize their actions in favor of exploration. From a human perspective, this makes RL agents hard to interpret and predict, and from a safety perspective, even harder to formally verify. We propose a novel method to induce predictable behavior in RL agents, referred to as Predictability-Aware RL (PA-RL), which employs the state sequence entropy rate as a predictability measure. We show how the entropy rate can be formulated as an average reward objective, and since its entropy reward function is policy-dependent, we introduce an action-dependent surrogate entropy enabling the use of PG methods. We prove that deterministic policies minimizing the average surrogate reward exist and also minimize the actual entropy rate, and show how, given a learned dynamical model, we are able to approximate the value function associated to th
    
[^11]: 制作一个甜甜圈：用于零样本变形操纵的分层EMD空间规划与工具

    Make a Donut: Hierarchical EMD-Space Planning for Zero-Shot Deformable Manipulation with Tools

    [https://arxiv.org/abs/2311.02787](https://arxiv.org/abs/2311.02787)

    引入了一种无需演示的分层规划方法，利用大型语言模型来解决复杂的长时间任务，为每个阶段提供工具名称和Python代码。

    

    变形物体操纵是机器人领域中最迷人又最艰巨的挑战之一。虽然先前的技术主要依赖于通过演示学习潜在动态，通常表示为粒子或图像之一，但存在一个重要限制：获取适当的演示，特别是对于长时间任务，可能是困难的。此外，完全基于演示进行学习可能会阻碍模型超越演示任务的能力。在这项工作中，我们介绍了一种无需演示的分层规划方法，能够处理复杂的长时间任务而无需任何训练。我们利用大型语言模型（LLMs）来表达与指定任务对应的高层、阶段-by-阶段计划。对于每个单独阶段，LLM提供工具的名称和Python代码，以制作中间子目标点云。

    arXiv:2311.02787v2 Announce Type: replace-cross  Abstract: Deformable object manipulation stands as one of the most captivating yet formidable challenges in robotics. While previous techniques have predominantly relied on learning latent dynamics through demonstrations, typically represented as either particles or images, there exists a pertinent limitation: acquiring suitable demonstrations, especially for long-horizon tasks, can be elusive. Moreover, basing learning entirely on demonstrations can hamper the model's ability to generalize beyond the demonstrated tasks. In this work, we introduce a demonstration-free hierarchical planning approach capable of tackling intricate long-horizon tasks without necessitating any training. We employ large language models (LLMs) to articulate a high-level, stage-by-stage plan corresponding to a specified task. For every individual stage, the LLM provides both the tool's name and the Python code to craft intermediate subgoal point clouds. With the
    
[^12]: LLM是否能记忆本体论？

    Do LLMs Dream of Ontologies?. (arXiv:2401.14931v1 [cs.CL])

    [http://arxiv.org/abs/2401.14931](http://arxiv.org/abs/2401.14931)

    本文研究了通用预训练大型语言模型（LLMs）是否记忆了已知本体论的信息以及记忆的程度，结果显示LLMs部分地了解本体论的概念，记忆程度与其在Web上的流行程度成正比。

    

    大型语言模型（LLMs）最近在自动文本理解和生成方面取得了革命性的进展。这些模型的性能依赖于底层神经网络体系结构的参数数量，这使得LLMs能够记忆训练过程中接触到的大量数据的一部分。本文研究了通用预训练LLMs是否记忆了已知本体论的信息以及记忆的程度。我们的结果表明，LLMs部分地了解本体论：它们可以记忆文本中提到的本体论概念，但其对概念的记忆程度似乎与其在Web上的流行程度成比例变化，因为Web是它们训练材料的主要来源。此外，我们提出了新的度量标准，通过测量不同提示重复、查询语言和确定度的输出一致性来估计LLMs对本体论信息的记忆程度。

    Large language models (LLMs) have recently revolutionized automated text understanding and generation. The performance of these models relies on the high number of parameters of the underlying neural architectures, which allows LLMs to memorize part of the vast quantity of data seen during the training. This paper investigates whether and to what extent general-purpose pre-trained LLMs have memorized information from known ontologies. Our results show that LLMs partially know ontologies: they can, and do indeed, memorize concepts from ontologies mentioned in the text, but the level of memorization of their concepts seems to vary proportionally to their popularity on the Web, the primary source of their training material. We additionally propose new metrics to estimate the degree of memorization of ontological information in LLMs by measuring the consistency of the output produced across different prompt repetitions, query languages, and degrees of determinism.
    
[^13]: 对水印技术应用于人工智能生成内容的漏洞研究

    Towards the Vulnerability of Watermarking Artificial Intelligence Generated Content. (arXiv:2310.07726v1 [cs.CV])

    [http://arxiv.org/abs/2310.07726](http://arxiv.org/abs/2310.07726)

    该研究探讨了将水印技术应用于人工智能生成内容的漏洞，并证明了现有的水印机制容易被对手破解。

    

    人工智能生成内容（AIGC）在社交媒体上越来越受欢迎，许多商业服务已经推出。这些服务利用先进的生成模型，如潜在扩散模型和大型语言模型，为用户生成创意内容（例如逼真的图像、流畅的句子）。对于此类生成内容的使用需要高度监管，因为服务提供商需要确保用户不违反使用政策（例如滥用商业化、生成和分发不安全的内容）。最近提出了许多水印技术，但是本文表明对手可以轻易破解这些水印机制。具体而言，我们考虑了两种可能的攻击方式：（1）水印去除：对手可以轻松地从生成内容中删除嵌入的水印，然后自由使用而不受服务提供商的限制；（2）水印伪造：对手可以创建非法的水印。

    Artificial Intelligence Generated Content (AIGC) is gaining great popularity in social media, with many commercial services available. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images, fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content).  Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely without the regulation of the service provider. (2) Watermark forge: the adversary can create illegal co
    
[^14]: 医疗知识图谱综述：资源、应用和前景

    A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises. (arXiv:2306.04802v1 [cs.AI])

    [http://arxiv.org/abs/2306.04802](http://arxiv.org/abs/2306.04802)

    本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。

    

    医疗知识图谱(HKGs)已成为组织医学知识的有结构且可解释的有为工具，提供了医学概念及其关系的全面视图。然而，数据异质性和覆盖范围有限等挑战仍然存在，强调了在HKG领域需要进一步研究的必要性。本综述是HKG的第一份综合概述。我们总结了HKG构建的流程和关键技术（即从头开始和通过集成），以及常见的利用方法（即基于模型和非基于模型）。为了为研究人员提供有价值的资源，我们根据它们捕获的数据类型和应用领域（该资源存储于https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase）组织了现有的HKG，并提供了相关的统计信息。在应用部分，我们深入探讨了HKG在各种医疗领域的变革性影响。

    Healthcare knowledge graphs (HKGs) have emerged as a promising tool for organizing medical knowledge in a structured and interpretable way, which provides a comprehensive view of medical concepts and their relationships. However, challenges such as data heterogeneity and limited coverage remain, emphasizing the need for further research in the field of HKGs. This survey paper serves as the first comprehensive overview of HKGs. We summarize the pipeline and key techniques for HKG construction (i.e., from scratch and through integration), as well as the common utilization approaches (i.e., model-free and model-based). To provide researchers with valuable resources, we organize existing HKGs (The resource is available at https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase) based on the data types they capture and application domains, supplemented with pertinent statistical information. In the application section, we delve into the transformative impact of HKGs across various hea
    

