# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack](https://arxiv.org/abs/2404.01833) | 提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。 |
| [^2] | [NeuroVoz: a Castillian Spanish corpus of parkinsonian speech](https://arxiv.org/abs/2403.02371) | 这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。 |
| [^3] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |
| [^4] | [The Last JITAI? The Unreasonable Effectiveness of Large Language Models in Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting](https://arxiv.org/abs/2402.08658) | 本研究探索了使用大型语言模型（LLMs）实现即时自适应干预（JITAIs）的可行性。通过测试GPT-4模型以促进门诊心脏康复中心的心脏健康体育活动的使用案例，我们提出了450个JITAI决策和信息。 |
| [^5] | [Large Language Model Agent for Hyper-Parameter Optimization](https://arxiv.org/abs/2402.01881) | 基于大规模语言模型的AgentHPO技术通过自动化超参数优化，在机器学习任务中大大减少了试验次数，简化了设置过程，提升了解释性和用户信任。 |
| [^6] | [Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory.](http://arxiv.org/abs/2310.20360) | 本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。 |
| [^7] | [Lifted Inference beyond First-Order Logic.](http://arxiv.org/abs/2308.11738) | 这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。 |
| [^8] | [Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance.](http://arxiv.org/abs/2308.08448) | 这项研究讨论了在金融领域中应用量子机器学习的新研究方向，通过比较qGAN和QCBM等模型，展示了在金融领域中实现量子优势的潜力。 |
| [^9] | [Towards an AI Accountability Policy.](http://arxiv.org/abs/2307.13658) | 这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。 |
| [^10] | [Towards Explainable TOPSIS: Visual Insights into the Effects of Weights and Aggregations on Rankings.](http://arxiv.org/abs/2306.07706) | 本研究提出了可解释TOPSIS的新方法，并介绍了TOPSIS-Explorer决策支持工具。该方法通过可视化方式解释权重和聚合对排名结果的影响，对实际应用有着重要意义。 |
| [^11] | [Diagrammatization: Rationalizing with diagrammatic AI explanations for abductive-deductive reasoning on hypotheses.](http://arxiv.org/abs/2302.01241) | 本文提出了一种图解化的方法，以支持可解释的人工智能，通过图解型和假设性推理，缩小可解释性差距。通过临床应用研究和建模研究，我们发现DiagramNet不仅能提供忠实的杂音形状解释，还具有较好的预测性能，而且图解型解释在临床相关的情况下更受推崇。 |

# 详细

[^1]: 伟大，现在写一篇关于此的文章：Crescendo多回合LLM越狱攻击

    Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack

    [https://arxiv.org/abs/2404.01833](https://arxiv.org/abs/2404.01833)

    提出了一种名为Crescendo的新型多回合越狱攻击方法，通过看似良性的对话方式逐渐升级与模型的交互，成功突破了大型语言模型的限制。

    

    大型语言模型（LLMs）的流行程度大幅上升，并且越来越多地被应用于多个领域。这些LLMs在设计上避免涉及非法或不道德的话题，以避免对负责任的AI造成伤害。然而，最近出现了一系列攻击，被称为“越狱”，旨在突破这种对齐。直观地说，越狱攻击旨在缩小模型能做的与愿意做的之间的差距。本文介绍了一种名为Crescendo的新型越狱攻击。与现有的越狱方法不同，Crescendo是一种多回合越狱，以一种看似良性的方式与模型进行交互。它从有关手头任务的一般提示或问题开始，然后逐渐升级对话，引用模型的回复，逐渐导致成功越狱。我们在包括ChatGPT、Gemini Pr在内的各种公共系统上评估了Crescendo。

    arXiv:2404.01833v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have risen significantly in popularity and are increasingly being adopted across multiple applications. These LLMs are heavily aligned to resist engaging in illegal or unethical topics as a means to avoid contributing to responsible AI harms. However, a recent line of attacks, known as "jailbreaks", seek to overcome this alignment. Intuitively, jailbreak attacks aim to narrow the gap between what the model can do and what it is willing to do. In this paper, we introduce a novel jailbreak attack called Crescendo. Unlike existing jailbreak methods, Crescendo is a multi-turn jailbreak that interacts with the model in a seemingly benign manner. It begins with a general prompt or question about the task at hand and then gradually escalates the dialogue by referencing the model's replies, progressively leading to a successful jailbreak. We evaluate Crescendo on various public systems, including ChatGPT, Gemini Pr
    
[^2]: NeuroVoz：帕金森病患者语音的卡斯蒂利亚语语料库

    NeuroVoz: a Castillian Spanish corpus of parkinsonian speech

    [https://arxiv.org/abs/2403.02371](https://arxiv.org/abs/2403.02371)

    这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。

    

    通过语音分析进行帕金森病（PD）诊断的进展受到公开可用、多样化的语言数据集的显著缺乏的阻碍，限制了现有研究结果的可再现性和进一步探索。为了弥补这一空白，我们引入了一个全面的语料库，包括来自108位母语为卡斯蒂利亚语的说话者，包括55名健康对照组和53名被诊断患有PD的个体，所有这些个体都在药物治疗下，并且在药物优化状态下进行记录。 这一独特数据集涵盖了广泛的语音任务，包括持续发音五个西班牙元音、发音测试、16个听后重复的话语以及自由独白。该数据集通过专家手动转录听后重复任务强调准确性和可靠性，并利用Whisper进行自动独白转录，使其成为帕金森病患者语音的最完整的公开语料库。

    arXiv:2403.02371v1 Announce Type: cross  Abstract: The advancement of Parkinson's Disease (PD) diagnosis through speech analysis is hindered by a notable lack of publicly available, diverse language datasets, limiting the reproducibility and further exploration of existing research.   In response to this gap, we introduce a comprehensive corpus from 108 native Castilian Spanish speakers, comprising 55 healthy controls and 53 individuals diagnosed with PD, all of whom were under pharmacological treatment and recorded in their medication-optimized state. This unique dataset features a wide array of speech tasks, including sustained phonation of the five Spanish vowels, diadochokinetic tests, 16 listen-and-repeat utterances, and free monologues. The dataset emphasizes accuracy and reliability through specialist manual transcriptions of the listen-and-repeat tasks and utilizes Whisper for automated monologue transcriptions, making it the most complete public corpus of Parkinsonian speech, 
    
[^3]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    
[^4]: 最后的JITAI？大型语言模型在发放及时自适应干预中的不合理有效性：在前瞻性心脏康复环境中促进体育活动

    The Last JITAI? The Unreasonable Effectiveness of Large Language Models in Issuing Just-in-Time Adaptive Interventions: Fostering Physical Activity in a Prospective Cardiac Rehabilitation Setting

    [https://arxiv.org/abs/2402.08658](https://arxiv.org/abs/2402.08658)

    本研究探索了使用大型语言模型（LLMs）实现即时自适应干预（JITAIs）的可行性。通过测试GPT-4模型以促进门诊心脏康复中心的心脏健康体育活动的使用案例，我们提出了450个JITAI决策和信息。

    

    我们探索了大型语言模型（LLMs）在数字健康中触发和个性化即时自适应干预（JITAIs）内容的可行性。JITAIs被视为可持续行为改变的关键机制，将干预措施根据个体的当前情境和需求进行调整。然而，传统的基于规则和机器学习模型在JITAI实施中面临可扩展性和可靠性的限制，例如缺乏个性化、管理多参数系统困难以及数据稀疏性等问题。为了研究通过LLMs实现JITAI，我们使用基于在门诊心脏康复中促进心脏健康体育活动的使用案例的现代最高性能模型“GPT-4”的实例作为触发和个性化JITAIs的基础。随后，我们生成了总共450个建议的JITAI决策和信息。

    We explored the viability of Large Language Models (LLMs) for triggering and personalizing content for Just-in-Time Adaptive Interventions (JITAIs) in digital health. JITAIs are being explored as a key mechanism for sustainable behavior change, adapting interventions to an individual's current context and needs. However, traditional rule-based and machine learning models for JITAI implementation face scalability and reliability limitations, such as lack of personalization, difficulty in managing multi-parametric systems, and issues with data sparsity. To investigate JITAI implementation via LLMs, we tested the contemporary overall performance-leading model 'GPT-4' with examples grounded in the use case of fostering heart-healthy physical activity in outpatient cardiac rehabilitation. Three personas and five sets of context information per persona were used as a basis of triggering and personalizing JITAIs. Subsequently, we generated a total of 450 proposed JITAI decisions and message c
    
[^5]: 基于大规模语言模型的超参数优化的技术

    Large Language Model Agent for Hyper-Parameter Optimization

    [https://arxiv.org/abs/2402.01881](https://arxiv.org/abs/2402.01881)

    基于大规模语言模型的AgentHPO技术通过自动化超参数优化，在机器学习任务中大大减少了试验次数，简化了设置过程，提升了解释性和用户信任。

    

    超参数优化在现代机器学习中至关重要，需要专业知识、大量实验以及高计算和人力资源。尽管自动化机器学习（AutoML）取得了一些进展，但试验效率、设置复杂性和互操作性方面仍存在挑战。为了解决这些问题，我们引入了一种新的范式，利用大规模语言模型（LLMs）来自动化不同机器学习任务的超参数优化，称为AgentHPO（LLM Agent-based Hyperparameter Optimization）。具体来说，AgentHPO自主处理任务信息，根据历史试验对特定超参数（HPs）进行实验，并进行迭代优化。与传统的AutoML方法相比，这种类似人类的优化过程极大地减少了所需的试验次数，简化了设置过程，并提升了解释性和用户信任。

    Hyperparameter optimization is critical in modern machine learning, requiring expert knowledge, numerous trials, and high computational and human resources. Despite the advancements in Automated Machine Learning (AutoML), challenges in terms of trial efficiency, setup complexity, and interoperability still persist. To address these issues, we introduce a novel paradigm leveraging Large Language Models (LLMs) to automate hyperparameter optimization across diverse machine learning tasks, which is named AgentHPO (short for LLM Agent-based Hyperparameter Optimization). Specifically, AgentHPO processes the task information autonomously, conducts experiments with specific hyperparameters (HPs), and iteratively optimizes them based on historical trials. This human-like optimization process largely reduces the number of required trials, simplifies the setup process, and enhances interpretability and user trust, compared to traditional AutoML methods. Extensive empirical experiments conducted o
    
[^6]: 深度学习的数学介绍：方法、实现和理论

    Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. (arXiv:2310.20360v1 [cs.LG])

    [http://arxiv.org/abs/2310.20360](http://arxiv.org/abs/2310.20360)

    本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。

    

    本书旨在介绍深度学习算法的主题。我们详细介绍了深度学习算法的基本组成部分，包括不同的人工神经网络架构（如全连接前馈神经网络、卷积神经网络、循环神经网络、残差神经网络和带有批归一化的神经网络）以及不同的优化算法（如基本的随机梯度下降法、加速方法和自适应方法）。我们还涵盖了深度学习算法的几个理论方面，如人工神经网络的逼近能力（包括神经网络的微积分）、优化理论（包括Kurdyka-Lojasiewicz不等式）和泛化误差。在本书的最后一部分，我们还回顾了一些用于偏微分方程的深度学习逼近方法，包括物理信息神经网络（PINNs）和深度Galerkin方法。希望本书能对学生和科学家们有所帮助。

    This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-{\L}ojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet 
    
[^7]: 超出一阶逻辑的提升推理

    Lifted Inference beyond First-Order Logic. (arXiv:2308.11738v1 [cs.AI])

    [http://arxiv.org/abs/2308.11738](http://arxiv.org/abs/2308.11738)

    这项工作研究了超越一阶逻辑的提升推理问题，扩展了计数量词扩展的两个变量的一阶逻辑片段的域可提升性，并在限定了关系的情况下探索了不同属性的域可提升性。

    

    在统计关系学习模型中，加权一阶模型计数(WFOMC)是概率推理的基础。由于WFOMC在一般情况下是不可计算的（$\#$P完全），因此能够在多项式时间内进行WFOMC的逻辑碎片非常有意义。这样的碎片被称为域可提升。最近的研究表明，在计数量词（$\mathrm{C^2}$）扩展的两个变量的一阶逻辑片段中，可以进行域提升。然而，许多真实世界数据的属性，如引用网络中的非循环性和社交网络中的连通性，不能在$\mathrm{C^2}$或一阶逻辑中建模。在这项工作中，我们扩展了$\mathrm{C^2}$的域可提升性，包括多个这样的属性。我们证明了在将$\mathrm{C^2}$句子的一个关系限定为表示有向无环图、连通图、树（或有向树）或森林（或有向森林）时，它仍然保持了域可提升性。所有我们的结果都是...

    Weighted First Order Model Counting (WFOMC) is fundamental to probabilistic inference in statistical relational learning models. As WFOMC is known to be intractable in general ($\#$P-complete), logical fragments that admit polynomial time WFOMC are of significant interest. Such fragments are called domain liftable. Recent works have shown that the two-variable fragment of first order logic extended with counting quantifiers ($\mathrm{C^2}$) is domain-liftable. However, many properties of real-world data, like acyclicity in citation networks and connectivity in social networks, cannot be modeled in $\mathrm{C^2}$, or first order logic in general. In this work, we expand the domain liftability of $\mathrm{C^2}$ with multiple such properties. We show that any $\mathrm{C^2}$ sentence remains domain liftable when one of its relations is restricted to represent a directed acyclic graph, a connected graph, a tree (resp. a directed tree) or a forest (resp. a directed forest). All our results r
    
[^8]: 在金融领域中实现量子生成对抗网络（qGAN）和QCBM

    Implementing Quantum Generative Adversarial Network (qGAN) and QCBM in Finance. (arXiv:2308.08448v1 [quant-ph])

    [http://arxiv.org/abs/2308.08448](http://arxiv.org/abs/2308.08448)

    这项研究讨论了在金融领域中应用量子机器学习的新研究方向，通过比较qGAN和QCBM等模型，展示了在金融领域中实现量子优势的潜力。

    

    量子机器学习（QML）是一个跨学科的领域，由两个最具创新性的研究领域组成：量子计算和经典机器学习（ML），ML和人工智能（AI）被认为是将受到量子计算机兴起影响的第一个领域。这项工作讨论了在金融中应用量子机器学习（QML）的一些新研究领域，我们讨论了一些已在金融界引起关注的QML模型，以及使用模拟环境中的真实金融数据集对qGAN（量子生成对抗网络）和QCBM（量子电路Born机）等模型进行比较。对于qGAN，我们定义了鉴别器和生成器的量子电路，并展示了未来在金融领域中通过QML实现量子优势的潜力。

    Quantum machine learning (QML) is a cross-disciplinary subject made up of two of the most exciting research areas: quantum computing and classical machine learning (ML), with ML and artificial intelligence (AI) being projected as the first fields that will be impacted by the rise of quantum machines. Quantum computers are being used today in drug discovery, material & molecular modelling and finance. In this work, we discuss some upcoming active new research areas in application of quantum machine learning (QML) in finance. We discuss certain QML models that has become areas of active interest in the financial world for various applications. We use real world financial dataset and compare models such as qGAN (quantum generative adversarial networks) and QCBM (quantum circuit Born machine) among others, using simulated environments. For the qGAN, we define quantum circuits for discriminators and generators and show promises of future quantum advantage via QML in finance.
    
[^9]: 关于AI问责政策的探索

    Towards an AI Accountability Policy. (arXiv:2307.13658v1 [cs.CY])

    [http://arxiv.org/abs/2307.13658](http://arxiv.org/abs/2307.13658)

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。

    

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”作出的回应。在回答相关问题的关键句子末尾，提供了要求评论的问题编号的上标。该白皮书提出了一组相互关联的AI问责政策建议。

    This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.
    
[^10]: 朝着可解释性TOPSIS：权重和聚合对排名的影响的视觉洞察力

    Towards Explainable TOPSIS: Visual Insights into the Effects of Weights and Aggregations on Rankings. (arXiv:2306.07706v1 [cs.AI])

    [http://arxiv.org/abs/2306.07706](http://arxiv.org/abs/2306.07706)

    本研究提出了可解释TOPSIS的新方法，并介绍了TOPSIS-Explorer决策支持工具。该方法通过可视化方式解释权重和聚合对排名结果的影响，对实际应用有着重要意义。

    

    多标准决策分析（MCDA）在各个行业中广泛用于评估和排名备选方案。在众多MCDA方法中，TOPSIS仍然是许多应用领域最受欢迎的选择之一。TOPSIS计算考虑的备选方案与两个预定义方案（即理想状态和反理想状态）之间的距离，并根据这些距离的聚合值创建备选方案的排名。然而，TOPSIS的内部工作解释是困难的，特别是当标准数目很大时。为此，最近的研究表明可以使用备选方案的平均值（M）和标准偏差（SD）来表示TOPSIS聚合值，从而创建MSD空间，这是一种可视化并解释聚合的工具。即使MSD空間非常有用，但它假设标准同样重要，使其在实际排名问题中的适用性降低 。在本文中，我们推广了 TOPSIS 结果的转换，使得不同的标准可以解释为视觉上的 MSD 空间，以此来处理将权重加入TOPSIS问题的场景。我们引入了加权 MSD 空间并开发了一个决策支持工具，称为 TOPSIS-Explorer，提供易于理解的视觉洞察力，以分析不同权重和聚合值对排名结果的影响。我们的方法在合成数据集和供应商选择实际案例上进行了评估，证明了它在提供可解释TOPSIS结果方面的适用性和有效性。

    Multi-Criteria Decision Analysis (MCDA) is extensively used across diverse industries to assess and rank alternatives. Among numerous MCDA methods developed to solve real-world ranking problems, TOPSIS remains one of the most popular choices in many application areas. TOPSIS calculates distances between the considered alternatives and two predefined ones, namely the ideal and the anti-ideal, and creates a ranking of the alternatives according to a chosen aggregation of these distances. However, the interpretation of the inner workings of TOPSIS is difficult, especially when the number of criteria is large. To this end, recent research has shown that TOPSIS aggregations can be expressed using the means (M) and standard deviations (SD) of alternatives, creating MSD-space, a tool for visualizing and explaining aggregations. Even though MSD-space is highly useful, it assumes equally important criteria, making it less applicable to real-world ranking problems. In this paper, we generalize t
    
[^11]: 图解化：利用图解型AI解释对假设性演绎推理的理性化

    Diagrammatization: Rationalizing with diagrammatic AI explanations for abductive-deductive reasoning on hypotheses. (arXiv:2302.01241v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.01241](http://arxiv.org/abs/2302.01241)

    本文提出了一种图解化的方法，以支持可解释的人工智能，通过图解型和假设性推理，缩小可解释性差距。通过临床应用研究和建模研究，我们发现DiagramNet不仅能提供忠实的杂音形状解释，还具有较好的预测性能，而且图解型解释在临床相关的情况下更受推崇。

    

    许多可解释的人工智能（XAI）可视化工具已经被开发出来，但它们通常需要用户进一步推理来解释。我们认为，XAI应该支持图解型和假设性推理，以便AI能够进行假设生成和评估，从而减少可解释性差距。我们提出了图解化方法，以i)进行Peircean推导-演绎推理，ii)遵循领域惯例，和iii)用图示或语言进行解释。我们在临床应用领域实现了DiagramNet，以预测心脏听诊中的心脏诊断，并用基于形状的杂音图解进行解释。在建模研究中，我们发现DiagramNet不仅提供了忠实的杂音形状解释，而且比基线模型具有更好的预测性能。我们进一步通过医学生的定性用户研究展示了图解型解释的可理解性和可信度，并表明在临床相关的情况下，图解式解释比其他方式更受推崇。

    Many visualizations have been developed for explainable AI (XAI), but they often require further reasoning by users to interpret. We argue that XAI should support diagrammatic and abductive reasoning for the AI to perform hypothesis generation and evaluation to reduce the interpretability gap. We propose Diagrammatization to i) perform Peircean abductive-deductive reasoning, ii) follow domain conventions, and iii) explain with diagrams visually or verbally. We implemented DiagramNet for a clinical application to predict cardiac diagnoses from heart auscultation, and explain with shape-based murmur diagrams. In modeling studies, we found that DiagramNet not only provides faithful murmur shape explanations, but also has better prediction performance than baseline models. We further demonstrate the interpretability and trustworthiness of diagrammatic explanations in a qualitative user study with medical students, showing that clinically-relevant, diagrammatic explanations are preferred ov
    

