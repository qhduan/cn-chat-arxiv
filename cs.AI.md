# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Security and Privacy Challenges of Large Language Models: A Survey](https://rss.arxiv.org/abs/2402.00888) | 大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。 |
| [^2] | [SHIELD: A regularization technique for eXplainable Artificial Intelligence](https://arxiv.org/abs/2404.02611) | SHIELD引入了一种正则化技术，通过隐藏部分输入数据并评估预测结果的差异，从而改善了可解释人工智能模型的质量。 |
| [^3] | [Optimization-based Prompt Injection Attack to LLM-as-a-Judge](https://arxiv.org/abs/2403.17710) | 介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。 |
| [^4] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^5] | [CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion](https://arxiv.org/abs/2402.14551) | CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现 |
| [^6] | [Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding](https://arxiv.org/abs/2402.14279) | 通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。 |
| [^7] | [ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters](https://arxiv.org/abs/2402.10930) | ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。 |
| [^8] | [Large Language Model-Based Interpretable Machine Learning Control in Building Energy Systems](https://arxiv.org/abs/2402.09584) | 本文研究了机器学习控制在建筑能源系统中的可解释性，通过将Shapley值和大型语言模型相结合，提高了机器学习控制模型的透明性和理解性。 |
| [^9] | [Advancing Building Energy Modeling with Large Language Models: Exploration and Case Studies](https://arxiv.org/abs/2402.09579) | 本文研究了将大型语言模型ChatGPT与EnergyPlus建筑能源建模软件融合的创新方法，并强调了大型语言模型在解决建筑能源建模挑战方面的潜力和多种应用。 |
| [^10] | [Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs](https://arxiv.org/abs/2312.11282) | 该论文评估了当前最先进的大型语言模型（GPT-4）在知识图谱上的对话推理能力，提出了一种基于KG推理的LLM基准代理（LLM-ARK），该代理利用全文环境提示来实现精确和适应性强的KG路径预测，并采用近端策略优化算法进行训练。 |
| [^11] | [ShaRP: Explaining Rankings with Shapley Values.](http://arxiv.org/abs/2401.16744) | ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。 |
| [^12] | [Integrating Symbolic Reasoning into Neural Generative Models for Design Generation.](http://arxiv.org/abs/2310.09383) | 这项研究将神经网络和符号推理结合起来，提出了Spatial Reasoning Integrated Generator (SPRING)，用于设计生成。SPRING通过将神经网络和符号约束满足结合起来，能够生成满足用户规格和实用要求的设计。 |
| [^13] | [Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models.](http://arxiv.org/abs/2308.16703) | 本文介绍了故障注入和安全错误攻击用于提取嵌入式神经网络模型的方法，并阐述了对32位微控制器上的深度神经网络进行模型提取攻击的实验结果。 |
| [^14] | [A Machine with Short-Term, Episodic, and Semantic Memory Systems.](http://arxiv.org/abs/2212.02098) | 本文研究了一个具有短期、情节和语义内存系统的机器代理模型，通过基于知识图谱的建模，在强化学习环境中实现了短期记忆的管理和存储，实验证明这种人类记忆系统结构的代理比没有该结构的代理表现更好。 |

# 详细

[^1]: 大型语言模型的安全和隐私挑战：一项调查

    Security and Privacy Challenges of Large Language Models: A Survey

    [https://rss.arxiv.org/abs/2402.00888](https://rss.arxiv.org/abs/2402.00888)

    大型语言模型具有卓越的能力，但也面临着安全和隐私攻击的威胁。本调查全面审查了LLM的安全和隐私挑战，涵盖了训练数据、用户和应用风险等方面，并对解决方法进行了回顾。

    

    大型语言模型（LLM）展示了非凡的能力，并在生成和总结文本、语言翻译和问答等多个领域做出了贡献。如今，LLM正在成为计算机语言处理任务中非常流行的工具，具备分析复杂语言模式并根据上下文提供相关和适当回答的能力。然而，尽管具有显著优势，这些模型也容易受到安全和隐私攻击的威胁，如越狱攻击、数据污染攻击和个人可识别信息泄露攻击。本调查全面审查了LLM的安全和隐私挑战，包括训练数据和用户方面的问题，以及在交通、教育和医疗等各个领域中应用带来的风险。我们评估了LLM的脆弱性程度，调查了出现的安全和隐私攻击，并对潜在的解决方法进行了回顾。

    Large Language Models (LLMs) have demonstrated extraordinary capabilities and contributed to multiple fields, such as generating and summarizing text, language translation, and question-answering. Nowadays, LLM is becoming a very popular tool in computerized language processing tasks, with the capability to analyze complicated linguistic patterns and provide relevant and appropriate responses depending on the context. While offering significant advantages, these models are also vulnerable to security and privacy attacks, such as jailbreaking attacks, data poisoning attacks, and Personally Identifiable Information (PII) leakage attacks. This survey provides a thorough review of the security and privacy challenges of LLMs for both training data and users, along with the application-based risks in various domains, such as transportation, education, and healthcare. We assess the extent of LLM vulnerabilities, investigate emerging security and privacy attacks for LLMs, and review the potent
    
[^2]: SHIELD: 一种用于可解释人工智能的正则化技术

    SHIELD: A regularization technique for eXplainable Artificial Intelligence

    [https://arxiv.org/abs/2404.02611](https://arxiv.org/abs/2404.02611)

    SHIELD引入了一种正则化技术，通过隐藏部分输入数据并评估预测结果的差异，从而改善了可解释人工智能模型的质量。

    

    随着人工智能系统在各个领域变得不可或缺，对可解释性的需求与日俱增。尽管科学界的努力主要集中在为模型获取更好的解释上，但重要的是不要忽视这个解释过程对改善训练的潜力。虽然现有的努力主要集中在为黑盒模型生成和评估解释上，但直接通过这些评估来增强模型仍存在关键差距。本文介绍了SHIELD（选择性隐藏输入评估学习动态），这是一种适用于可解释人工智能的正则化技术，旨在通过隐藏部分输入数据并评估预测结果的差异来改善模型质量。与传统方法相比，SHIELD正则化无缝集成到目标函数中，提高了模型的可解释性同时也改善了性能

    arXiv:2404.02611v1 Announce Type: new  Abstract: As Artificial Intelligence systems become integral across domains, the demand for explainability grows. While the effort by the scientific community is focused on obtaining a better explanation for the model, it is important not to ignore the potential of this explanation process to improve training as well. While existing efforts primarily focus on generating and evaluating explanations for black-box models, there remains a critical gap in directly enhancing models through these evaluations. This paper introduces SHIELD (Selective Hidden Input Evaluation for Learning Dynamics), a regularization technique for explainable artificial intelligence designed to improve model quality by concealing portions of input data and assessing the resulting discrepancy in predictions. In contrast to conventional approaches, SHIELD regularization seamlessly integrates into the objective function, enhancing model explainability while also improving perfor
    
[^3]: 基于优化的对LLM评判系统的提示注入攻击

    Optimization-based Prompt Injection Attack to LLM-as-a-Judge

    [https://arxiv.org/abs/2403.17710](https://arxiv.org/abs/2403.17710)

    介绍了一种基于优化的提示注入攻击方法，JudgeDeceiver，针对LLM-as-a-Judge，通过自动化生成对抗序列实现了有针对性和高效的模型评估操控。

    

    LLM-as-a-Judge 是一种可以使用大型语言模型（LLMs）评估文本信息的新颖解决方案。根据现有研究，LLMs在提供传统人类评估的引人注目替代方面表现出色。然而，这些系统针对提示注入攻击的鲁棒性仍然是一个未解决的问题。在这项工作中，我们引入了JudgeDeceiver，一种针对LLM-as-a-Judge量身定制的基于优化的提示注入攻击。我们的方法制定了一个精确的优化目标，用于攻击LLM-as-a-Judge的决策过程，并利用优化算法高效地自动化生成对抗序列，实现对模型评估的有针对性和有效的操作。与手工制作的提示注入攻击相比，我们的方法表现出卓越的功效，给基于LLM的判断系统当前的安全范式带来了重大挑战。

    arXiv:2403.17710v1 Announce Type: cross  Abstract: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. T
    
[^4]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^5]: CLCE：一种优化学习融合的改进交叉熵和对比学习方法

    CLCE: An Approach to Refining Cross-Entropy and Contrastive Learning for Optimized Learning Fusion

    [https://arxiv.org/abs/2402.14551](https://arxiv.org/abs/2402.14551)

    CLCE方法结合了标签感知对比学习与交叉熵损失，通过协同利用难例挖掘提高了性能表现

    

    最先进的预训练图像模型主要采用两阶段方法：在大规模数据集上进行初始无监督预训练，然后使用交叉熵损失（CE）进行特定任务的微调。然而，已经证明CE可能会损害模型的泛化性和稳定性。为了解决这些问题，我们引入了一种名为CLCE的新方法，该方法将标签感知对比学习与CE相结合。我们的方法不仅保持了两种损失函数的优势，而且以协同方式利用难例挖掘来增强性能。

    arXiv:2402.14551v1 Announce Type: cross  Abstract: State-of-the-art pre-trained image models predominantly adopt a two-stage approach: initial unsupervised pre-training on large-scale datasets followed by task-specific fine-tuning using Cross-Entropy loss~(CE). However, it has been demonstrated that CE can compromise model generalization and stability. While recent works employing contrastive learning address some of these limitations by enhancing the quality of embeddings and producing better decision boundaries, they often overlook the importance of hard negative mining and rely on resource intensive and slow training using large sample batches. To counter these issues, we introduce a novel approach named CLCE, which integrates Label-Aware Contrastive Learning with CE. Our approach not only maintains the strengths of both loss functions but also leverages hard negative mining in a synergistic way to enhance performance. Experimental results demonstrate that CLCE significantly outperf
    
[^6]: 使用音素表示减缓语言差异，实现稳健的多语言理解

    Mitigating the Linguistic Gap with Phonemic Representations for Robust Multilingual Language Understanding

    [https://arxiv.org/abs/2402.14279](https://arxiv.org/abs/2402.14279)

    通过使用音素表示，本文提出了一种新颖的解决方案来减缓高资源语言和低资源语言之间的性能差距，并通过实证研究和理论分析证明了其有效性。

    

    为了改善多语言理解，通常需要在训练阶段使用多种语言，依赖复杂的训练技术，并且在高资源语言和低资源语言之间存在显著的性能差距。我们假设语言之间的性能差距受到这些语言之间的语言差异的影响，并通过使用音素表示（具体来说，将音素作为输入标记输入到语言模型中，而不是子词）提供了一种新颖的解决方案，以实现稳健的多语言建模。我们通过三个跨语言任务的定量证据展示了音素表示的有效性，这进一步得到了对跨语言性能差距的理论分析的证明。

    arXiv:2402.14279v1 Announce Type: cross  Abstract: Approaches to improving multilingual language understanding often require multiple languages during the training phase, rely on complicated training techniques, and -- importantly -- struggle with significant performance gaps between high-resource and low-resource languages. We hypothesize that the performance gaps between languages are affected by linguistic gaps between those languages and provide a novel solution for robust multilingual language modeling by employing phonemic representations (specifically, using phonemes as input tokens to LMs rather than subwords). We present quantitative evidence from three cross-lingual tasks that demonstrate the effectiveness of phonemic representation, which is further justified by a theoretical analysis of the cross-lingual performance gap.
    
[^7]: ConSmax: 具有可学习参数的硬件友好型Softmax替代方案

    ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters

    [https://arxiv.org/abs/2402.10930](https://arxiv.org/abs/2402.10930)

    ConSmax是一种硬件友好型Softmax替代方案，通过引入可学习参数，在不影响性能的情况下实现了对原Softmax关键任务的高效处理。

    

    自注意机制将基于transformer的大型语言模型（LLM）与卷积和循环神经网络区分开来。尽管性能有所提升，但由于自注意中广泛使用Softmax，在硅上实现实时LLM推断仍具挑战性。为了解决这一挑战，我们提出了Constant Softmax（ConSmax），这是一种高效的Softmax替代方案，采用可微的规范化参数来消除Softmax中的最大搜索和分母求和，实现了大规模并行化。

    arXiv:2402.10930v1 Announce Type: cross  Abstract: The self-attention mechanism sets transformer-based large language model (LLM) apart from the convolutional and recurrent neural networks. Despite the performance improvement, achieving real-time LLM inference on silicon is challenging due to the extensively used Softmax in self-attention. Apart from the non-linearity, the low arithmetic intensity greatly reduces the processing parallelism, which becomes the bottleneck especially when dealing with a longer context. To address this challenge, we propose Constant Softmax (ConSmax), a software-hardware co-design as an efficient Softmax alternative. ConSmax employs differentiable normalization parameters to remove the maximum searching and denominator summation in Softmax. It allows for massive parallelization while performing the critical tasks of Softmax. In addition, a scalable ConSmax hardware utilizing a bitwidth-split look-up table (LUT) can produce lossless non-linear operation and 
    
[^8]: 基于大型语言模型的建筑能源系统机器学习控制的可解释性研究

    Large Language Model-Based Interpretable Machine Learning Control in Building Energy Systems

    [https://arxiv.org/abs/2402.09584](https://arxiv.org/abs/2402.09584)

    本文研究了机器学习控制在建筑能源系统中的可解释性，通过将Shapley值和大型语言模型相结合，提高了机器学习控制模型的透明性和理解性。

    

    机器学习控制在暖通空调系统中的潜力受限于其不透明的性质和推理机制，这对于用户和建模者来说是具有挑战性的，难以完全理解，最终导致对基于机器学习控制的决策缺乏信任。为了解决这个挑战，本文研究和探索了可解释机器学习（IML），它是机器学习的一个分支，可以增强模型和推理的透明性和理解性，以提高MLC及其在暖通空调系统中的工业应用的可信度。具体而言，我们开发了一个创新性的框架，将Shapley值的原则和大型语言模型（LLMs）的上下文学习特性相结合。而Shapley值在解剖ML模型中各种特征的贡献方面起到了重要作用，LLM则可以深入理解MLC中基于规则的部分；将它们结合起来，LLM进一步将这些洞见打包到一个

    arXiv:2402.09584v1 Announce Type: new  Abstract: The potential of Machine Learning Control (MLC) in HVAC systems is hindered by its opaque nature and inference mechanisms, which is challenging for users and modelers to fully comprehend, ultimately leading to a lack of trust in MLC-based decision-making. To address this challenge, this paper investigates and explores Interpretable Machine Learning (IML), a branch of Machine Learning (ML) that enhances transparency and understanding of models and their inferences, to improve the credibility of MLC and its industrial application in HVAC systems. Specifically, we developed an innovative framework that combines the principles of Shapley values and the in-context learning feature of Large Language Models (LLMs). While the Shapley values are instrumental in dissecting the contributions of various features in ML models, LLM provides an in-depth understanding of rule-based parts in MLC; combining them, LLM further packages these insights into a
    
[^9]: 用大型语言模型推动建筑能源建模：探索和案例研究

    Advancing Building Energy Modeling with Large Language Models: Exploration and Case Studies

    [https://arxiv.org/abs/2402.09579](https://arxiv.org/abs/2402.09579)

    本文研究了将大型语言模型ChatGPT与EnergyPlus建筑能源建模软件融合的创新方法，并强调了大型语言模型在解决建筑能源建模挑战方面的潜力和多种应用。

    

    人工智能的快速发展促进了像ChatGPT这样的大型语言模型的出现，为专门的工程建模（尤其是基于物理的建筑能源建模）提供了潜在的应用。本文研究了将大型语言模型与建筑能源建模软件（具体为EnergyPlus）融合的创新方法。首先进行了文献综述，揭示了在工程建模中整合大型语言模型的增长趋势，但在建筑能源建模中的应用研究仍然有限。我们强调了大型语言模型在解决建筑能源建模挑战方面的潜力，并概述了潜在的应用，包括：1）模拟输入生成，2）模拟输出分析和可视化，3）进行错误分析，4）共模拟，5）模拟知识提取。

    arXiv:2402.09579v1 Announce Type: cross  Abstract: The rapid progression in artificial intelligence has facilitated the emergence of large language models like ChatGPT, offering potential applications extending into specialized engineering modeling, especially physics-based building energy modeling. This paper investigates the innovative integration of large language models with building energy modeling software, focusing specifically on the fusion of ChatGPT with EnergyPlus. A literature review is first conducted to reveal a growing trend of incorporating of large language models in engineering modeling, albeit limited research on their application in building energy modeling. We underscore the potential of large language models in addressing building energy modeling challenges and outline potential applications including 1) simulation input generation, 2) simulation output analysis and visualization, 3) conducting error analysis, 4) co-simulation, 5) simulation knowledge extraction a
    
[^10]: 评估和增强用于知识图谱上的对话推理的大型语言模型

    Evaluating and Enhancing Large Language Models for Conversational Reasoning on Knowledge Graphs

    [https://arxiv.org/abs/2312.11282](https://arxiv.org/abs/2312.11282)

    该论文评估了当前最先进的大型语言模型（GPT-4）在知识图谱上的对话推理能力，提出了一种基于KG推理的LLM基准代理（LLM-ARK），该代理利用全文环境提示来实现精确和适应性强的KG路径预测，并采用近端策略优化算法进行训练。

    

    大型语言模型（LLM）的发展得益于预训练技术的进展。通过手动设计的提示，这些模型展示了强大的推理能力。在这项工作中，我们评估了当前最先进的LLM（GPT-4）在知识图谱（KG）上的对话推理能力。然而，由于缺乏KG环境意识和开发有效的中间推理阶段优化机制的困难，LLM的性能受到限制。我们进一步引入了LLM-ARK，一个基于KG推理的LLM基准代理，旨在提供精确和适应性强的KG路径预测。LLM-ARK利用全文环境（FTE）提示来吸收每个推理步骤中的状态信息。我们将KG上的多跳推理挑战重新框定为顺序决策任务。利用近端策略优化（PPO）在线策略梯度强化学习算法，我们的模型...

    The development of large language models (LLMs) has been catalyzed by advancements in pre-training techniques. These models have demonstrated robust reasoning capabilities through manually designed prompts. In this work, we evaluate the conversational reasoning capabilities of the current state-of-the-art LLM (GPT-4) on knowledge graphs (KGs). However, the performance of LLMs is constrained due to a lack of KG environment awareness and the difficulties in developing effective optimization mechanisms for intermediary reasoning stages. We further introduce LLM-ARK, a LLM grounded KG reasoning agent designed to deliver precise and adaptable predictions on KG paths. LLM-ARK leverages Full Textual Environment (FTE) prompt to assimilate state information within each reasoning step. We reframe the challenge of multi-hop reasoning on the KG as a sequential decision-making task. Utilizing the Proximal Policy Optimization (PPO) online policy gradient reinforcement learning algorithm, our model i
    
[^11]: ShaRP：用Shapley值解释排名

    ShaRP: Explaining Rankings with Shapley Values. (arXiv:2401.16744v1 [cs.AI])

    [http://arxiv.org/abs/2401.16744](http://arxiv.org/abs/2401.16744)

    ShaRP是一个基于Shapley值的框架，用于解释排名结果中各个特征的贡献。即使使用线性评分函数，特征的权重也不一定对应其Shapley值的贡献，而是取决于特征分布和评分特征之间的局部相互作用。

    

    在招聘、大学招生和贷款等重要领域的算法决策常常是基于排名的。由于这些决策对个人、组织和人群的影响，有必要了解它们：了解决策是否遵守法律，帮助个人提高他们的排名，并设计更好的排名程序。本文提出了ShaRP（Shapley for Rankings and Preferences），这是一个基于Shapley值的框架，用于解释特征对排名结果不同方面的贡献。使用ShaRP，我们展示了即使算法排名器使用的评分函数是已知的且是线性的，每个特征的权重也不一定对应其Shapley值的贡献。贡献取决于特征的分布以及评分特征之间微妙的局部相互作用。ShaRP基于量化输入影响框架，并可以计算贡献。

    Algorithmic decisions in critical domains such as hiring, college admissions, and lending are often based on rankings. Because of the impact these decisions have on individuals, organizations, and population groups, there is a need to understand them: to know whether the decisions are abiding by the law, to help individuals improve their rankings, and to design better ranking procedures.  In this paper, we present ShaRP (Shapley for Rankings and Preferences), a framework that explains the contributions of features to different aspects of a ranked outcome, and is based on Shapley values. Using ShaRP, we show that even when the scoring function used by an algorithmic ranker is known and linear, the weight of each feature does not correspond to its Shapley value contribution. The contributions instead depend on the feature distributions, and on the subtle local interactions between the scoring features. ShaRP builds on the Quantitative Input Influence framework, and can compute the contri
    
[^12]: 将符号推理整合到神经生成模型中的设计生成

    Integrating Symbolic Reasoning into Neural Generative Models for Design Generation. (arXiv:2310.09383v1 [cs.AI])

    [http://arxiv.org/abs/2310.09383](http://arxiv.org/abs/2310.09383)

    这项研究将神经网络和符号推理结合起来，提出了Spatial Reasoning Integrated Generator (SPRING)，用于设计生成。SPRING通过将神经网络和符号约束满足结合起来，能够生成满足用户规格和实用要求的设计。

    

    设计生成需要将神经和符号推理紧密结合，因为良好的设计必须满足显式用户需求和隐含的美学、实用性和便利性规则。当前由神经网络驱动的自动化设计工具能够生成吸引人的设计，但不能满足用户的规格和实用要求。符号推理工具（如约束编程）不能感知图像中的低级视觉信息或捕捉到美学等微妙方面。我们引入了Spatial Reasoning Integrated Generator (SPRING)用于设计生成。SPRING在深度生成网络中嵌入了一个神经和符号整合的空间推理模块。空间推理模块通过一个循环神经网络预测并通过符号约束满足来决定要生成的对象的位置，以边界框的形式表示。将符号推理嵌入神经生成保证了SPRING的输出满足用户的规格和实用要求。

    Design generation requires tight integration of neural and symbolic reasoning, as good design must meet explicit user needs and honor implicit rules for aesthetics, utility, and convenience. Current automated design tools driven by neural networks produce appealing designs, but cannot satisfy user specifications and utility requirements. Symbolic reasoning tools, such as constraint programming, cannot perceive low-level visual information in images or capture subtle aspects such as aesthetics. We introduce the Spatial Reasoning Integrated Generator (SPRING) for design generation. SPRING embeds a neural and symbolic integrated spatial reasoning module inside the deep generative network. The spatial reasoning module decides the locations of objects to be generated in the form of bounding boxes, which are predicted by a recurrent neural network and filtered by symbolic constraint satisfaction. Embedding symbolic reasoning into neural generation guarantees that the output of SPRING satisfi
    
[^13]: 故障注入和安全错误攻击用于提取嵌入式神经网络模型

    Fault Injection and Safe-Error Attack for Extraction of Embedded Neural Network Models. (arXiv:2308.16703v1 [cs.CR])

    [http://arxiv.org/abs/2308.16703](http://arxiv.org/abs/2308.16703)

    本文介绍了故障注入和安全错误攻击用于提取嵌入式神经网络模型的方法，并阐述了对32位微控制器上的深度神经网络进行模型提取攻击的实验结果。

    

    模型提取作为一种关键的安全威胁而出现，攻击向量利用了算法和实现方面的方法。攻击者的主要目标是尽可能多地窃取受保护的受害者模型的信息，以便他可以用替代模型来模仿它，即使只有有限的访问相似的训练数据。最近，物理攻击，如故障注入，已经显示出对嵌入式模型的完整性和机密性的令人担忧的效果。我们的重点是32位微控制器上的嵌入式深度神经网络模型，这是物联网中广泛使用的硬件平台系列，以及使用标准故障注入策略-安全错误攻击（SEA）来进行具有有限训练数据访问的模型提取攻击。由于攻击强烈依赖于输入查询，我们提出了一种黑盒方法来构建一个成功的攻击集。对于一个经典的卷积神经网络，我们成功地恢复了至少90%的

    Model extraction emerges as a critical security threat with attack vectors exploiting both algorithmic and implementation-based approaches. The main goal of an attacker is to steal as much information as possible about a protected victim model, so that he can mimic it with a substitute model, even with a limited access to similar training data. Recently, physical attacks such as fault injection have shown worrying efficiency against the integrity and confidentiality of embedded models. We focus on embedded deep neural network models on 32-bit microcontrollers, a widespread family of hardware platforms in IoT, and the use of a standard fault injection strategy - Safe Error Attack (SEA) - to perform a model extraction attack with an adversary having a limited access to training data. Since the attack strongly depends on the input queries, we propose a black-box approach to craft a successful attack set. For a classical convolutional neural network, we successfully recover at least 90% of
    
[^14]: 一个具有短期、情节和语义内存系统的机器

    A Machine with Short-Term, Episodic, and Semantic Memory Systems. (arXiv:2212.02098v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2212.02098](http://arxiv.org/abs/2212.02098)

    本文研究了一个具有短期、情节和语义内存系统的机器代理模型，通过基于知识图谱的建模，在强化学习环境中实现了短期记忆的管理和存储，实验证明这种人类记忆系统结构的代理比没有该结构的代理表现更好。

    

    受认知科学理论中显性人类记忆系统的启发，我们建立了一个具有短期、情节和语义记忆系统的代理模型，每个记忆系统都用知识图谱建模。为了评估该系统并分析该代理的行为，我们设计并发布了我们自己的强化学习代理环境“房间”，在这个环境中，代理必须学习如何编码、存储和检索记忆，通过回答问题来最大化回报。我们证明了我们基于深度Q学习的代理成功学习了短期记忆是否应该被遗忘，还是应该存储在情节或语义记忆系统中。我们的实验表明，具有类人记忆系统的代理在环境中表现优于没有这种记忆结构的代理。

    Inspired by the cognitive science theory of the explicit human memory systems, we have modeled an agent with short-term, episodic, and semantic memory systems, each of which is modeled with a knowledge graph. To evaluate this system and analyze the behavior of this agent, we designed and released our own reinforcement learning agent environment, "the Room", where an agent has to learn how to encode, store, and retrieve memories to maximize its return by answering questions. We show that our deep Q-learning based agent successfully learns whether a short-term memory should be forgotten, or rather be stored in the episodic or semantic memory systems. Our experiments indicate that an agent with human-like memory systems can outperform an agent without this memory structure in the environment.
    

