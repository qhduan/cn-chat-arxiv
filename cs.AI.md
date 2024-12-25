# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ICLN: Input Convex Loss Network for Decision Focused Learning](https://arxiv.org/abs/2403.01875) | 提出了输入凸损失网络（ICLN），通过输入凸神经网络学习任务损失，为决策集中学习提供了全局替代损失。 |
| [^2] | [Leveraging AI Planning For Detecting Cloud Security Vulnerabilities](https://arxiv.org/abs/2402.10985) | 提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。 |
| [^3] | [Tracking Changing Probabilities via Dynamic Learners](https://arxiv.org/abs/2402.10142) | 该论文介绍了通过动态学习器追踪概率变化的方法，通过输出候选项目及其概率来预测离散项目序列中下一个可能出现的项目。 |
| [^4] | [Resolving Ethics Trade-offs in Implementing Responsible AI](https://arxiv.org/abs/2401.08103) | 该论文提出了通过权衡处理人工智能伦理中的紧张关系的五种方法，并提出了一个框架来实施全面的人工智能/机器学习系统。 |
| [^5] | [Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering](https://arxiv.org/abs/2401.06824) | 通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。 |
| [^6] | [AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents.](http://arxiv.org/abs/2401.13178) | AgentBoard是一个综合的基准测试和评估框架，专为分析评估LLM智能体而设计，解决了在多轮交互和部分可观察环境中对智能体性能进行基准测试的挑战，并提供了细粒度的进展率指标和评估工具包。 |
| [^7] | [How well can large language models explain business processes?.](http://arxiv.org/abs/2401.12846) | 该论文介绍了一个用于生成业务流程解释的SAX4BPM框架，该框架通过与大型语言模型（LLM）集成来综合各种输入要素，以提供更好的情境感知解释（SAX）。 |
| [^8] | [Perfect Alignment May be Poisonous to Graph Contrastive Learning.](http://arxiv.org/abs/2310.03977) | 本研究探讨了图形对比学习中增强方法和下游性能的关系，并发现图形对比学习主要通过分离不同类别的节点来为下游任务做出贡献。 |
| [^9] | [Bridging the Domain Gap by Clustering-based Image-Text Graph Matching.](http://arxiv.org/abs/2310.02692) | 通过基于聚类的图像-文本图匹配来弥合领域差距，学习领域不变特征以实现在未见过领域上的良好泛化能力，实验结果显示在公共数据集上达到最先进性能。 |
| [^10] | [Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints.](http://arxiv.org/abs/2308.16534) | 本文提出了一种方法，通过神经符号约束来调节基于评分的生成模型，实现了在非条件生成模型下强制执行任意的逻辑约束，从而获得了一个有效的、无需额外训练的条件采样算法。 |
| [^11] | [GPTEval: A Survey on Assessments of ChatGPT and GPT-4.](http://arxiv.org/abs/2308.12488) | 本文对ChatGPT和GPT-4的先前评估进行了综合分析，关注其语言和推理能力、科学知识和伦理考虑，提出了几个评估大型语言模型的建议。 |

# 详细

[^1]: ICLN：输入凸损失网络用于决策集中学习

    ICLN: Input Convex Loss Network for Decision Focused Learning

    [https://arxiv.org/abs/2403.01875](https://arxiv.org/abs/2403.01875)

    提出了输入凸损失网络（ICLN），通过输入凸神经网络学习任务损失，为决策集中学习提供了全局替代损失。

    

    在不确定性条件下的决策问题中，预测未知参数通常被认为与优化部分无关。决策集中学习（DFL）是一个面向任务的框架，通过调整预测模型以为相应任务提供更好的决策来整合预测和优化。本文提出了输入凸损失网络（ICLN），这是一种新颖的全局替代损失，可以在一般的DFL范式中实现。ICLN通过输入凸神经网络学习任务损失，已经被保证为某些情况下是凸的。

    arXiv:2403.01875v1 Announce Type: cross  Abstract: In decision-making problem under uncertainty, predicting unknown parameters is often considered independent of the optimization part. Decision-focused Learning (DFL) is a task-oriented framework to integrate prediction and optimization by adapting predictive model to give better decision for the corresponding task. Here, an inevitable challenge arises when computing gradients of the optimal decision with respect to the parameters. Existing researches cope this issue by smoothly reforming surrogate optimization or construct surrogate loss function that mimic task loss. However, they are applied to restricted optimization domain or build functions in a local manner leading a large computational time. In this paper, we propose Input Convex Loss Network (ICLN), a novel global surrogate loss which can be implemented in a general DFL paradigm. ICLN learns task loss via Input Convex Neural Networks which is guaranteed to be convex for some in
    
[^2]: 利用AI规划技术检测云安全漏洞

    Leveraging AI Planning For Detecting Cloud Security Vulnerabilities

    [https://arxiv.org/abs/2402.10985](https://arxiv.org/abs/2402.10985)

    提出了一个通用框架来建模云系统中的访问控制策略，并开发了基于PDDL模型的新方法来检测可能导致诸如勒索软件和敏感数据外泄等广泛攻击的安全漏洞。

    

    云计算服务提供了可扩展且具有成本效益的数据存储、处理和协作解决方案。随着它们的普及，与其安全漏洞相关的担忧也在增长，这可能导致数据泄露和勒索软件等复杂攻击。为了应对这些问题，我们首先提出了一个通用框架，用于表达云系统中不同对象（如用户、数据存储、安全角色）之间的关系，以建模云系统中的访问控制策略。访问控制误配置通常是云攻击的主要原因。其次，我们开发了一个PDDL模型，用于检测安全漏洞，例如可能导致广泛攻击（如勒索软件）和敏感数据外泄等。规划器可以生成攻击以识别云中的此类漏洞。最后，我们在14个不同商业组织的真实亚马逊AWS云配置上测试了我们的方法。

    arXiv:2402.10985v1 Announce Type: cross  Abstract: Cloud computing services provide scalable and cost-effective solutions for data storage, processing, and collaboration. Alongside their growing popularity, concerns related to their security vulnerabilities leading to data breaches and sophisticated attacks such as ransomware are growing. To address these, first, we propose a generic framework to express relations between different cloud objects such as users, datastores, security roles, to model access control policies in cloud systems. Access control misconfigurations are often the primary driver for cloud attacks. Second, we develop a PDDL model for detecting security vulnerabilities which can for example lead to widespread attacks such as ransomware, sensitive data exfiltration among others. A planner can then generate attacks to identify such vulnerabilities in the cloud. Finally, we test our approach on 14 real Amazon AWS cloud configurations of different commercial organizations
    
[^3]: 通过动态学习器追踪概率变化

    Tracking Changing Probabilities via Dynamic Learners

    [https://arxiv.org/abs/2402.10142](https://arxiv.org/abs/2402.10142)

    该论文介绍了通过动态学习器追踪概率变化的方法，通过输出候选项目及其概率来预测离散项目序列中下一个可能出现的项目。

    

    考虑一个预测器，即一个学习器，其输入是一系列离散项目。预测器的任务是在每个时间点进行概率多类别预测，即通过输出有零个或多个候选项目及其概率来预测接下来可能发生的项目，然后揭示实际项目并从中学习。为了输出概率，预测器会跟踪其所见项目的比例。预测器具有恒定（有限）的空间，我们寻求高效的预测和更新技术：流是无界的，项目的集合对预测器是未知的，它们的总数也可能无限增长。此外，存在非平稳性：项目的潜在频率可能会不时发生显著变化。例如，新项目可能开始出现，一些当前频繁出现的项目可能再次停止出现。由于有空间限制，预测器只需要提供概率。

    arXiv:2402.10142v1 Announce Type: cross  Abstract: Consider a predictor, a learner, whose input is a stream of discrete items. The predictor's task, at every time point, is probabilistic multiclass prediction, i.e., to predict which item may occur next by outputting zero or more candidate items, each with a probability, after which the actual item is revealed and the predictor learns from this observation. To output probabilities, the predictor keeps track of the proportions of the items it has seen. The predictor has constant (limited) space and we seek efficient prediction and update techniques: The stream is unbounded, the set of items is unknown to the predictor and their totality can also grow unbounded. Moreover, there is non-stationarity: the underlying frequencies of items may change, substantially, from time to time. For instance, new items may start appearing and a few currently frequent items may cease to occur again. The predictor, being space-bounded, need only provide pro
    
[^4]: 解决在实施负责任人工智能中的伦理权衡

    Resolving Ethics Trade-offs in Implementing Responsible AI

    [https://arxiv.org/abs/2401.08103](https://arxiv.org/abs/2401.08103)

    该论文提出了通过权衡处理人工智能伦理中的紧张关系的五种方法，并提出了一个框架来实施全面的人工智能/机器学习系统。

    

    虽然把高级人工智能伦理原则应用到实际的人工智能/机器学习系统中已经取得了进展，但在处理底层人工智能伦理方面的紧张关系方面仍存在理论与实践之间的差距。我们提出了五种处理这些关系的方法，从简单到复杂不等。这些方法在考虑的上下文类型、范围、衡量上下文的方法和证明程度上有所不同。这些方法中没有一种适用于所有组织、系统或应用。为了解决这个问题，我们提出了一个框架，包括：（i）积极识别紧张关系，（ii）优先处理和权衡伦理方面，（iii）证明和记录权衡决策。该提议的框架旨在促进实施符合潜在监管要求的全面人工智能/机器学习系统。

    While the operationalisation of high-level AI ethics principles into practical AI/ML systems has made progress, there is still a theory-practice gap in managing tensions between the underlying AI ethics aspects. We cover five approaches for addressing the tensions via trade-offs, ranging from rudimentary to complex. The approaches differ in the types of considered context, scope, methods for measuring contexts, and degree of justification. None of the approaches is likely to be appropriate for all organisations, systems, or applications. To address this, we propose a framework which consists of: (i) proactive identification of tensions, (ii) prioritisation and weighting of ethics aspects, (iii) justification and documentation of trade-off decisions. The proposed framework aims to facilitate the implementation of well-rounded AI/ML systems that are appropriate for potential regulatory requirements.
    
[^5]: 打开LLMs的潘多拉魔盒：通过表示工程对LLMs进行越狱

    Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering

    [https://arxiv.org/abs/2401.06824](https://arxiv.org/abs/2401.06824)

    通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。

    

    越狱技术旨在通过诱使大型语言模型（LLMs）生成对恶意查询产生有毒响应，来探索LLMs安全性边界，这在LLMs社区内是一个重要关注点。我们提出一种名为通过表示工程对LLMs进行越狱（Jailbreaking LLMs through Representation Engineering，JRE）的新颖越狱方法，其仅需要少量查询对以提取可用于规避目标模型防御的“安全模式”，实现了前所未有的越狱性能。

    arXiv:2401.06824v2 Announce Type: replace-cross  Abstract: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation conf
    
[^6]: AgentBoard: 一种多轮LLM智能体的分析评估板

    AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents. (arXiv:2401.13178v1 [cs.CL])

    [http://arxiv.org/abs/2401.13178](http://arxiv.org/abs/2401.13178)

    AgentBoard是一个综合的基准测试和评估框架，专为分析评估LLM智能体而设计，解决了在多轮交互和部分可观察环境中对智能体性能进行基准测试的挑战，并提供了细粒度的进展率指标和评估工具包。

    

    评估大型语言模型（LLM）作为通用智能体对于理解其能力并促进其融入实际应用至关重要。然而，评估过程面临重大挑战。主要障碍之一是在统一框架内对智能体在不同场景下的性能进行基准测试，特别是在维护部分可观察环境和确保多轮交互方面。此外，当前的评估框架主要关注最终成功率，过程中提供的见解很少，无法深入理解模型的能力。为了解决这些挑战，我们引入了AgentBoard，这是一个创新的综合基准和伴随的开源评估框架，专为LLM智能体的分析评估而设计。AgentBoard提供了一种细粒度的进展率指标，捕捉逐步的进展，以及一个综合的评估工具包，具有易于评估和分析模型能力的功能。

    Evaluating large language models (LLMs) as general-purpose agents is essential for understanding their capabilities and facilitating their integration into practical applications. However, the evaluation process presents substantial challenges. A primary obstacle is the benchmarking of agent performance across diverse scenarios within a unified framework, especially in maintaining partially-observable environments and ensuring multi-round interactions. Moreover, current evaluation frameworks mostly focus on the final success rate, revealing few insights during the process and failing to provide a deep understanding of the model abilities. To address these challenges, we introduce AgentBoard, a pioneering comprehensive benchmark and accompanied open-source evaluation framework tailored to analytical evaluation of LLM agents. AgentBoard offers a fine-grained progress rate metric that captures incremental advancements as well as a comprehensive evaluation toolkit that features easy assess
    
[^7]: 大型语言模型能够如何解释业务流程？

    How well can large language models explain business processes?. (arXiv:2401.12846v1 [cs.AI])

    [http://arxiv.org/abs/2401.12846](http://arxiv.org/abs/2401.12846)

    该论文介绍了一个用于生成业务流程解释的SAX4BPM框架，该框架通过与大型语言模型（LLM）集成来综合各种输入要素，以提供更好的情境感知解释（SAX）。

    

    大型语言模型（LLMs）可能在未来的AI辅助业务流程管理系统（ABPMSs）中发挥重要作用，其功能涵盖系统生命周期的各个阶段。其中一个系统功能是情境感知解释（SAX），它涉及生成在考虑所解释条件出现的流程上下文的前提下既符合因果关系又可人类解读的解释。在本文中，我们介绍了开发用于生成SAX解释的SAX4BPM框架。SAX4BPM套件包括一组服务和一个中央知识库。这些服务的功能是获取构成SAX解释的各种知识要素。其中一个创新性的关键组成部分是因果过程执行视图。在本工作中，我们将该框架与LLM集成，以利用其综合各种输入要素的能力，从而改进SAX解释的质量。

    Large Language Models (LLMs) are likely to play a prominent role in future AI-augmented business process management systems (ABPMSs) catering functionalities across all system lifecycle stages. One such system's functionality is Situation-Aware eXplainability (SAX), which relates to generating causally sound and yet human-interpretable explanations that take into account the process context in which the explained condition occurred. In this paper, we present the SAX4BPM framework developed to generate SAX explanations. The SAX4BPM suite consists of a set of services and a central knowledge repository. The functionality of these services is to elicit the various knowledge ingredients that underlie SAX explanations. A key innovative component among these ingredients is the causal process execution view. In this work, we integrate the framework with an LLM to leverage its power to synthesize the various input ingredients for the sake of improved SAX explanations. Since the use of LLMs for
    
[^8]: 完美对齐可能对图形对比学习产生负面影响

    Perfect Alignment May be Poisonous to Graph Contrastive Learning. (arXiv:2310.03977v1 [cs.LG])

    [http://arxiv.org/abs/2310.03977](http://arxiv.org/abs/2310.03977)

    本研究探讨了图形对比学习中增强方法和下游性能的关系，并发现图形对比学习主要通过分离不同类别的节点来为下游任务做出贡献。

    

    图形对比学习旨在通过对齐正样本和分离负样本来学习节点表示。然而，在基于图形的学习中，对于特定增强方法背后的内在规律的研究有限。什么样的增强方法可以提高下游性能？对比学习如何实际影响下游任务？为什么增强的幅度很重要？本文试图通过建立增强方法和下游性能之间的联系，以及对对比学习的泛化性进行研究来回答这些问题。我们的发现表明，图形对比学习主要通过分离不同类别而不是聚集同一类别的节点来为下游任务做出贡献。因此，无法解释对比学习的成功，即全部样本完美对齐和增强重叠。为了理解增强如何辅助对比学习过程，我们进行了进一步的研究。

    Graph Contrastive Learning (GCL) aims to learn node representations by aligning positive pairs and separating negative ones. However, limited research has been conducted on the inner law behind specific augmentations used in graph-based learning. What kind of augmentation will help downstream performance, how does contrastive learning actually influence downstream tasks, and why the magnitude of augmentation matters? This paper seeks to address these questions by establishing a connection between augmentation and downstream performance, as well as by investigating the generalization of contrastive learning. Our findings reveal that GCL contributes to downstream tasks mainly by separating different classes rather than gathering nodes of the same class. So perfect alignment and augmentation overlap which draw all intra-class samples the same can not explain the success of contrastive learning. Then in order to comprehend how augmentation aids the contrastive learning process, we conduct 
    
[^9]: 基于聚类的图像-文本图匹配来弥合领域差距

    Bridging the Domain Gap by Clustering-based Image-Text Graph Matching. (arXiv:2310.02692v1 [cs.CV])

    [http://arxiv.org/abs/2310.02692](http://arxiv.org/abs/2310.02692)

    通过基于聚类的图像-文本图匹配来弥合领域差距，学习领域不变特征以实现在未见过领域上的良好泛化能力，实验结果显示在公共数据集上达到最先进性能。

    

    学习领域不变表示对于训练可以很好地推广到未见过目标任务领域的模型非常重要。文本描述本身包含概念的语义结构，这样的辅助语义线索可以用作领域概括问题的有效枢纽嵌入。我们使用多模态图像和文本融合的图表示来获得在局部图像和文本描述符之间考虑内在语义结构的领域不变枢纽嵌入。具体来说，我们通过(i)用图表示图像和文本描述，以及(ii)将基于图像节点特征的聚类和匹配应用到文本图中，来学习领域不变特征。我们使用大规模公共数据集（如CUB-DG和DomainBed）进行实验，并在这些数据集上达到与或优于现有最先进模型的性能。我们的代码将在出版后公开提供。

    Learning domain-invariant representations is important to train a model that can generalize well to unseen target task domains. Text descriptions inherently contain semantic structures of concepts and such auxiliary semantic cues can be used as effective pivot embedding for domain generalization problems. Here, we use multimodal graph representations, fusing images and text, to get domain-invariant pivot embeddings by considering the inherent semantic structure between local images and text descriptors. Specifically, we aim to learn domain-invariant features by (i) representing the image and text descriptions with graphs, and by (ii) clustering and matching the graph-based image node features into textual graphs simultaneously. We experiment with large-scale public datasets, such as CUB-DG and DomainBed, and our model achieves matched or better state-of-the-art performance on these datasets. Our code will be publicly available upon publication.
    
[^10]: 通过神经符号约束来调节基于评分的生成模型

    Conditioning Score-Based Generative Models by Neuro-Symbolic Constraints. (arXiv:2308.16534v1 [cs.LG])

    [http://arxiv.org/abs/2308.16534](http://arxiv.org/abs/2308.16534)

    本文提出了一种方法，通过神经符号约束来调节基于评分的生成模型，实现了在非条件生成模型下强制执行任意的逻辑约束，从而获得了一个有效的、无需额外训练的条件采样算法。

    

    基于评分和扩散模型已经成为一种有效的条件和非条件生成方法。然而，条件生成基于特定训练的条件模型或分类器指导，这需要训练一个噪声依赖的分类器，即使对于未损坏数据的分类器已经给出。我们提出了一种方法，可以从非条件评分生成模型中采样，可以强制执行任意的逻辑约束，而无需进行额外的训练。首先，我们展示了如何操纵学习得到的评分，以便在用户定义的约束条件下从非归一化分布中采样。然后，我们定义了一个灵活而数值稳定的神经符号框架，用于编码软逻辑约束。将这两个组成部分结合起来，我们获得了一个一般的但是近似的条件采样算法。我们进一步开发了有效的启发式方法来改进近似。最后，我们展示了我们方法的有效性。

    Score-based and diffusion models have emerged as effective approaches for both conditional and unconditional generation. Still conditional generation is based on either a specific training of a conditional model or classifier guidance, which requires training a noise-dependent classifier, even when the classifier for uncorrupted data is given. We propose an approach to sample from unconditional score-based generative models enforcing arbitrary logical constraints, without any additional training. Firstly, we show how to manipulate the learned score in order to sample from an un-normalized distribution conditional on a user-defined constraint. Then, we define a flexible and numerically stable neuro-symbolic framework for encoding soft logical constraints. Combining these two ingredients we obtain a general, but approximate, conditional sampling algorithm. We further developed effective heuristics aimed at improving the approximation. Finally, we show the effectiveness of our approach fo
    
[^11]: GPTEval: 对ChatGPT和GPT-4评估的调查

    GPTEval: A Survey on Assessments of ChatGPT and GPT-4. (arXiv:2308.12488v1 [cs.AI])

    [http://arxiv.org/abs/2308.12488](http://arxiv.org/abs/2308.12488)

    本文对ChatGPT和GPT-4的先前评估进行了综合分析，关注其语言和推理能力、科学知识和伦理考虑，提出了几个评估大型语言模型的建议。

    

    ChatGPT的出现引发了媒体对其扰乱社会和经济系统潜力的许多猜测。其惊人的语言能力激起学者们对其在不同领域表现的浓厚兴趣。已经有许多研究评估了ChatGPT和GPT-4在不同任务和学科中的能力。然而，缺乏一项综合性的综述总结集体评估结果。本调查的目标是对ChatGPT和GPT-4的先前评估进行深入分析，重点关注其语言和推理能力、科学知识和伦理考虑。此外，对现有评估方法进行了检查，并提出了几个未来研究评估大型语言模型的建议。

    The emergence of ChatGPT has generated much speculation in the press about its potential to disrupt social and economic systems. Its astonishing language ability has aroused strong curiosity among scholars about its performance in different domains. There have been many studies evaluating the ability of ChatGPT and GPT-4 in different tasks and disciplines. However, a comprehensive review summarizing the collective assessment findings is lacking. The objective of this survey is to thoroughly analyze prior assessments of ChatGPT and GPT-4, focusing on its language and reasoning abilities, scientific knowledge, and ethical considerations. Furthermore, an examination of the existing evaluation methods is conducted, offering several recommendations for future research in evaluating large language models.
    

