# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^2] | [Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach](https://arxiv.org/abs/2403.10112) | 本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。 |
| [^3] | [Don't Forget What I did?: Assessing Client Contributions in Federated Learning](https://arxiv.org/abs/2403.07151) | 提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。 |
| [^4] | [MuseGraph: Graph-oriented Instruction Tuning of Large Language Models for Generic Graph Mining](https://arxiv.org/abs/2403.04780) | MuseGraph将GNNs和LLMs的优势结合起来，提出了一种更有效和通用的图挖掘方法，可以跨不同任务和数据集使用 |
| [^5] | [Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management](https://arxiv.org/abs/2402.07949) | 通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。 |
| [^6] | [Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies.](http://arxiv.org/abs/2401.10266) | 本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。 |
| [^7] | [Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach.](http://arxiv.org/abs/2401.09671) | 本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。 |
| [^8] | [Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.](http://arxiv.org/abs/2308.15022) | 递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。 |
| [^9] | [Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings.](http://arxiv.org/abs/2308.11804) | 该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。 |
| [^10] | [Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks.](http://arxiv.org/abs/2303.10225) | 本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。 |

# 详细

[^1]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^2]: 单智能体与多智能体的私密主动感知：深度神经进化方法

    Single- and Multi-Agent Private Active Sensing: A Deep Neuroevolution Approach

    [https://arxiv.org/abs/2403.10112](https://arxiv.org/abs/2403.10112)

    本文提出了一种基于神经进化方法的单智能体与多智能体私密主动感知框架，通过在无线传感器网络中进行异常检测示例用例的数值实验验证了该方法的优越性。

    

    本文关注存在窥视者情况下的主动假设测试中的一个集中式问题和一个分散式问题。针对包括单个合法智能体的集中式问题，我们提出了基于神经进化（NE）的新框架；而针对分散式问题，我们开发了一种新颖的基于NE的方法，用于解决协作多智能体任务，这种方法有趣地保持了单一智能体NE的所有计算优势。通过对无线传感器网络上异常检测示例用例中的数值实验，验证了所提出的EAHT方法优于传统的主动假设测试策略以及基于学习的方法。

    arXiv:2403.10112v1 Announce Type: new  Abstract: In this paper, we focus on one centralized and one decentralized problem of active hypothesis testing in the presence of an eavesdropper. For the centralized problem including a single legitimate agent, we present a new framework based on NeuroEvolution (NE), whereas, for the decentralized problem, we develop a novel NE-based method for solving collaborative multi-agent tasks, which interestingly maintains all computational benefits of single-agent NE. The superiority of the proposed EAHT approaches over conventional active hypothesis testing policies, as well as learning-based methods, is validated through numerical investigations in an example use case of anomaly detection over wireless sensor networks.
    
[^3]: 不要忘记我做的事：评估联邦学习中的客户贡献

    Don't Forget What I did?: Assessing Client Contributions in Federated Learning

    [https://arxiv.org/abs/2403.07151](https://arxiv.org/abs/2403.07151)

    提出了一个历史感知的博弈理论框架FLContrib，用来评估联邦学习中的客户贡献。

    

    联邦学习（FL）是一种协作机器学习（ML）方法，多个客户参与训练ML模型，而不暴露私人数据。公平准确评估客户贡献在FL中是一个重要问题，以促进激励分配并鼓励多样化客户参与统一模型训练。本文提出了一个历史感知的博弈理论框架FLContrib，用于评估在每个FL训练时期中的（潜在非独立同分布）客户参与。

    arXiv:2403.07151v1 Announce Type: cross  Abstract: Federated Learning (FL) is a collaborative machine learning (ML) approach, where multiple clients participate in training an ML model without exposing the private data. Fair and accurate assessment of client contributions is an important problem in FL to facilitate incentive allocation and encouraging diverse clients to participate in a unified model training. Existing methods for assessing client contribution adopts co-operative game-theoretic concepts, such as Shapley values, but under simplified assumptions. In this paper, we propose a history-aware game-theoretic framework, called FLContrib, to assess client contributions when a subset of (potentially non-i.i.d.) clients participate in each epoch of FL training. By exploiting the FL training process and linearity of Shapley value, we develop FLContrib that yields a historical timeline of client contributions as FL training progresses over epochs. Additionally, to assess client cont
    
[^4]: MuseGraph：面向大型语言模型的图导向指令调整用于通用图挖掘

    MuseGraph: Graph-oriented Instruction Tuning of Large Language Models for Generic Graph Mining

    [https://arxiv.org/abs/2403.04780](https://arxiv.org/abs/2403.04780)

    MuseGraph将GNNs和LLMs的优势结合起来，提出了一种更有效和通用的图挖掘方法，可以跨不同任务和数据集使用

    

    具有丰富属性的图在建模互联实体和改进各种实际应用中的预测方面至关重要。传统图神经网络（GNNs）通常用于建模带属性的图，但需要在应用于不同图任务和数据集时进行重新训练。尽管大型语言模型（LLMs）的出现在自然语言处理中引入了新的范例，但LLMs在图挖掘中的生成潜力仍未得到充分探索。为此，我们提出了一个新颖的框架 MuseGraph，它无缝整合了GNNs和LLMs的优势，并促进了一种更有效和通用的图挖掘方法，可跨不同任务和数据集使用。具体而言，我们首先通过提出的自适应输入生成引入一个紧凑的图描述，以在语言令牌限制的约束下封装来自图的关键信息。

    arXiv:2403.04780v1 Announce Type: cross  Abstract: Graphs with abundant attributes are essential in modeling interconnected entities and improving predictions in various real-world applications. Traditional Graph Neural Networks (GNNs), which are commonly used for modeling attributed graphs, need to be re-trained every time when applied to different graph tasks and datasets. Although the emergence of Large Language Models (LLMs) has introduced a new paradigm in natural language processing, the generative potential of LLMs in graph mining remains largely under-explored. To this end, we propose a novel framework MuseGraph, which seamlessly integrates the strengths of GNNs and LLMs and facilitates a more effective and generic approach for graph mining across different tasks and datasets. Specifically, we first introduce a compact graph description via the proposed adaptive input generation to encapsulate key information from the graph under the constraints of language token limitations. T
    
[^5]: 优化人工胰腺设计以改善糖尿病管理

    Optimizing the Design of an Artificial Pancreas to Improve Diabetes Management

    [https://arxiv.org/abs/2402.07949](https://arxiv.org/abs/2402.07949)

    通过神经进化算法优化人工胰腺治疗策略，减少糖尿病患者的血糖偏差，并且降低注射次数。

    

    糖尿病是一种慢性疾病，影响美国境内有3800万人，它会影响身体将食物转化为能量（即血糖）的能力。标准的治疗方法是通过使用人工胰腺，即持续胰岛素泵（基础注射），以及定期注射胰岛素（突发注射）来补充碳水化合物摄入量。治疗目标是将血糖保持在可接受范围的中心位置，通过持续血糖测量来进行衡量。次要目标是减少注射次数，因为对某些患者来说注射是不愉快且难以实施的。本研究使用神经进化来发现治疗的最佳策略。基于30天的治疗和单个患者的测量数据集，首先训练了随机森林来预测未来的血糖水平。然后通过进化了一个神经网络来指定碳水化合物摄入量、基础注射水平和突发注射。进化发现了一个帕累托前沿，减少了与目标值的偏差。

    Diabetes, a chronic condition that impairs how the body turns food into energy, i.e. blood glucose, affects 38 million people in the US alone. The standard treatment is to supplement carbohydrate intake with an artificial pancreas, i.e. a continuous insulin pump (basal shots), as well as occasional insulin injections (bolus shots). The goal of the treatment is to keep blood glucose at the center of an acceptable range, as measured through a continuous glucose meter. A secondary goal is to minimize injections, which are unpleasant and difficult for some patients to implement. In this study, neuroevolution was used to discover an optimal strategy for the treatment. Based on a dataset of 30 days of treatment and measurements of a single patient, a random forest was first trained to predict future glucose levels. A neural network was then evolved to prescribe carbohydrates, basal pumping levels, and bolus injections. Evolution discovered a Pareto front that reduced deviation from the targe
    
[^6]: 工业厂房智能状态监测: 方法论和不确定性管理策略综述

    Intelligent Condition Monitoring of Industrial Plants: An Overview of Methodologies and Uncertainty Management Strategies. (arXiv:2401.10266v1 [cs.LG])

    [http://arxiv.org/abs/2401.10266](http://arxiv.org/abs/2401.10266)

    本论文综述了工业厂房智能状态监测和故障检测和诊断方法，重点关注了Tennessee Eastman Process。调研总结了最流行和最先进的深度学习和机器学习算法，并探讨了算法的优劣势。还讨论了不平衡数据和无标记样本等挑战，以及深度学习模型如何应对。比较了不同算法在Tennessee Eastman Process上的准确性和规格。

    

    状态监测在现代工业系统的安全性和可靠性中起着重要作用。人工智能（AI）方法作为一种在工业应用中日益受到学术界和行业关注的增长主题和一种强大的故障识别方式。本文概述了工业厂房智能状态监测和故障检测和诊断方法，重点关注开源基准Tennessee Eastman Process（TEP）。在这项调查中，总结了用于工业厂房状态监测、故障检测和诊断的最流行和最先进的深度学习（DL）和机器学习（ML）算法，并研究了每种算法的优点和缺点。还涵盖了不平衡数据、无标记样本以及深度学习模型如何处理这些挑战。最后，比较了利用Tennessee Eastman Process的不同算法的准确性和规格。

    Condition monitoring plays a significant role in the safety and reliability of modern industrial systems. Artificial intelligence (AI) approaches are gaining attention from academia and industry as a growing subject in industrial applications and as a powerful way of identifying faults. This paper provides an overview of intelligent condition monitoring and fault detection and diagnosis methods for industrial plants with a focus on the open-source benchmark Tennessee Eastman Process (TEP). In this survey, the most popular and state-of-the-art deep learning (DL) and machine learning (ML) algorithms for industrial plant condition monitoring, fault detection, and diagnosis are summarized and the advantages and disadvantages of each algorithm are studied. Challenges like imbalanced data, unlabelled samples and how deep learning models can handle them are also covered. Finally, a comparison of the accuracies and specifications of different algorithms utilizing the Tennessee Eastman Process 
    
[^7]: 迈向可识别的无监督领域转换：一种多样化分布匹配的方法

    Towards Identifiable Unsupervised Domain Translation: A Diversified Distribution Matching Approach. (arXiv:2401.09671v1 [cs.LG])

    [http://arxiv.org/abs/2401.09671](http://arxiv.org/abs/2401.09671)

    本研究旨在解决无监督领域转换中的可识别性问题，引入了一个MPA消除理论，解决了CycleGAN及其变体产生内容不对齐的限制。

    

    无监督领域转换（UDT）旨在找到将一个领域的样本（例如素描）转换为另一个领域（例如照片）的函数，同时不改变高层语义意义（也称为“内容”）。这些转换函数通常通过转换源领域和目标领域的概率分布来寻找。CycleGAN可以说是这一领域中最具代表性的方法。然而，文献中指出CycleGAN及其变体可能无法识别所需的转换函数，并产生内容不对齐的转换。这种局限性源于学习准则解空间中存在多个转换函数，称为“保度自同构（MPA）”。尽管意识到了这种可识别性问题，但解决方案仍然难以找到。本研究深入探究了核心的可识别性问题，并引入了MPA消除理论。我们的分析表明...

    Unsupervised domain translation (UDT) aims to find functions that convert samples from one domain (e.g., sketches) to another domain (e.g., photos) without changing the high-level semantic meaning (also referred to as ``content''). The translation functions are often sought by probability distribution matching of the transformed source domain and target domain. CycleGAN stands as arguably the most representative approach among this line of work. However, it was noticed in the literature that CycleGAN and variants could fail to identify the desired translation functions and produce content-misaligned translations. This limitation arises due to the presence of multiple translation functions -- referred to as ``measure-preserving automorphism" (MPA) -- in the solution space of the learning criteria. Despite awareness of such identifiability issues, solutions have remained elusive. This study delves into the core identifiability inquiry and introduces an MPA elimination theory. Our analysi
    
[^8]: 递归总结在大型语言模型中实现长期对话记忆

    Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. (arXiv:2308.15022v1 [cs.CL])

    [http://arxiv.org/abs/2308.15022](http://arxiv.org/abs/2308.15022)

    递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。

    

    大多数开放领域的对话系统在长期对话中容易遗忘重要信息。现有方法通常训练特定的检索器或总结器从过去获取关键信息，这需要耗费时间且高度依赖标记数据的质量。为了缓解这个问题，我们提出使用大型语言模型（LLMs）递归生成总结/记忆，以增强长期记忆能力。具体而言，我们的方法首先刺激LLMs记住小对话上下文，然后递归地使用之前的记忆和随后的对话内容产生新的记忆。最后，LLM可以在最新记忆的帮助下轻松生成高度一致的响应。我们使用ChatGPT和text-davinci-003进行评估，对广泛使用的公共数据集进行的实验证明我们的方法在长对话中可以生成更一致的响应。值得注意的是，我们的方法是实现LLM建模的潜在解决方案。

    Most open-domain dialogue systems suffer from forgetting important information, especially in a long-term conversation. Existing works usually train the specific retriever or summarizer to obtain key information from the past, which is time-consuming and highly depends on the quality of labeled data. To alleviate this problem, we propose to recursively generate summaries/ memory using large language models (LLMs) to enhance long-term memory ability. Specifically, our method first stimulates LLMs to memorize small dialogue contexts and then recursively produce new memory using previous memory and following contexts. Finally, the LLM can easily generate a highly consistent response with the help of the latest memory. We evaluate our method using ChatGPT and text-davinci-003, and the experiments on the widely-used public dataset show that our method can generate more consistent responses in a long-context conversation. Notably, our method is a potential solution to enable the LLM to model
    
[^9]: 这不是一个苹果：多模态嵌入中的对抗幻觉

    Ceci n'est pas une pomme: Adversarial Illusions in Multi-Modal Embeddings. (arXiv:2308.11804v1 [cs.CR])

    [http://arxiv.org/abs/2308.11804](http://arxiv.org/abs/2308.11804)

    该论文研究了多模态嵌入中的对抗幻觉问题。对手可以扰动输入的任意模态，使其嵌入与其他模态的任意输入接近，从而实现任意图像与任意文本、任意文本与任意声音的对齐。该问题与下游任务无关，对生成和分类任务会产生误导。

    

    多模态编码器将图像、声音、文本、视频等映射到一个单一的嵌入空间中，通过对齐不同模态的表示（例如将一张狗的图像与一种叫声相关联）。我们展示了多模态嵌入可以受到一种我们称之为“对抗幻觉”的攻击。给定任意模态的输入，对手可以扰动它，使其嵌入接近于另一模态中任意对手选择的输入的嵌入。幻觉使对手能够将任意图像与任意文本、任意文本与任意声音等进行对齐。对抗幻觉利用了嵌入空间中的接近性，因此与下游任务无关。使用ImageBind嵌入，我们演示了在没有具体下游任务知识的情况下，通过对抗性对齐的输入如何误导图像生成、文本生成和零样例分类。

    Multi-modal encoders map images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an input in any modality, an adversary can perturb it so as to make its embedding close to that of an arbitrary, adversary-chosen input in another modality. Illusions thus enable the adversary to align any image with any text, any text with any sound, etc.  Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.
    
[^10]: 增强神经网络对多样化$\ell_p$攻击的鲁棒性:鲁棒模态连接导向的对抗性防御

    Robust Mode Connectivity-Oriented Adversarial Defense: Enhancing Neural Network Robustness Against Diversified $\ell_p$ Attacks. (arXiv:2303.10225v1 [cs.AI])

    [http://arxiv.org/abs/2303.10225](http://arxiv.org/abs/2303.10225)

    本文提出一种新颖的鲁棒模态连接导向的对抗性防御，实现神经网络对多样化$\ell_p$攻击的鲁棒性，其中包括两个基于种群学习的学习阶段。

    

    对抗性鲁棒性是衡量神经网络在推理阶段抵御对抗性攻击能力的关键概念。最近的研究表明，尽管使用的强化鲁棒性训练技术能够提高对一种类型的攻击的鲁棒性，但模型仍然容易受到多样化的$\ell_p$攻击。为了实现多样化的$\ell_p$鲁棒性，我们提出了一种新颖的鲁棒模态连接 (RMC) 导向的对抗性防御，它包含两个基于种群学习的学习阶段。第一个阶段，RMC，能够搜索两个预先训练模型之间的模型参数空间，并找到包含高鲁棒性点的路径以抵御多样化的$\ell_p$攻击。基于RMC的有效性，我们开发了第二个阶段，基于RMC的优化，其中RMC作为神经网络多样化$\ell_p$鲁棒性进一步增强的基本单元。为了提高计算效率，我们将学习与仅选择子集的对抗性示例相结合，这导致了一组较小的代表性对抗性示例，可用于增强神经网络对多样化$\ell_p$攻击的鲁棒性。

    Adversarial robustness is a key concept in measuring the ability of neural networks to defend against adversarial attacks during the inference phase. Recent studies have shown that despite the success of improving adversarial robustness against a single type of attack using robust training techniques, models are still vulnerable to diversified $\ell_p$ attacks. To achieve diversified $\ell_p$ robustness, we propose a novel robust mode connectivity (RMC)-oriented adversarial defense that contains two population-based learning phases. The first phase, RMC, is able to search the model parameter space between two pre-trained models and find a path containing points with high robustness against diversified $\ell_p$ attacks. In light of the effectiveness of RMC, we develop a second phase, RMC-based optimization, with RMC serving as the basic unit for further enhancement of neural network diversified $\ell_p$ robustness. To increase computational efficiency, we incorporate learning with a sel
    

