# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge-refined Denoising Network for Robust Recommendation.](http://arxiv.org/abs/2304.14987) | 本文提出了一个名为知识精炼去噪网络（KRDN）的强健知识感知推荐框架，可以同时修剪任务无关的知识关联和噪声隐式反馈。 |
| [^2] | [Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models.](http://arxiv.org/abs/2304.14867) | 本文提出了面向主题的对抗排名攻击任务，以提高同一主题的一组查询中的特定文档排名，并提出了基于替代排名模型的新框架来改善攻击性能。 |
| [^3] | [A Unified Generative Retriever for Knowledge-Intensive Language Tasks via Prompt Learning.](http://arxiv.org/abs/2304.14856) | 本文提出了一种统一的生成检索器(UGR)，它将任务特定的效果与在KILTs中不同检索任务的强健性能相结合。UGR采用基于Prompt的学习方法，使其能够适应各种检索任务，并通过实验在五个KILTs基准测试上显著优于最先进的基线。 |
| [^4] | [Made of Steel? Learning Plausible Materials for Components in the Vehicle Repair Domain.](http://arxiv.org/abs/2304.14745) | 本文提出了一种新方法，通过探索预训练语言模型（PLM）学习车辆维修领域组件的特定材料，成功克服了数据稀疏性问题和缺乏注释数据集的问题。 |
| [^5] | [Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation.](http://arxiv.org/abs/2304.14668) | 本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。 |
| [^6] | [Multivariate Representation Learning for Information Retrieval.](http://arxiv.org/abs/2304.14522) | 本论文提出一种多元分布模型的信息检索表示学习框架，可无缝集成到现有近似最近邻算法中以实现高效检索。 |
| [^7] | [Evaluation of GPT-3.5 and GPT-4 for supporting real-world information needs in healthcare delivery.](http://arxiv.org/abs/2304.13714) | 本研究评估了在临床环境中使用GPT-3.5和GPT-4解决医学问题的安全性以及与信息技术咨询服务报告的一致性。研究结果表明，两个LLMs都可以以安全和一致的方式满足医生的信息需求。 |

# 详细

[^1]: 知识精炼去噪网络用于强健推荐

    Knowledge-refined Denoising Network for Robust Recommendation. (arXiv:2304.14987v1 [cs.IR])

    [http://arxiv.org/abs/2304.14987](http://arxiv.org/abs/2304.14987)

    本文提出了一个名为知识精炼去噪网络（KRDN）的强健知识感知推荐框架，可以同时修剪任务无关的知识关联和噪声隐式反馈。

    

    知识图谱作为丰富的辅助信息，成为了提升推荐性能和改善可解释性的重要组成部分。然而，现有的知识感知推荐方法直接在知识图谱和用户-物品二分图上执行信息传递，忽略了任务无关知识传播和交互噪声的影响，限制了其性能。为了解决这些问题，我们提出了一个强健的知识感知推荐框架，称为知识精炼去噪网络（KRDN），以同时修剪任务无关的知识关联和噪声隐式反馈。KRDN包括自适应知识精炼策略和对比去噪机制，能够自动收集高质量的知识图谱三元组，并裁剪噪声隐式反馈。此外，我们还设计了自适应损失函数和梯度估计器，用于模型训练。

    Knowledge graph (KG), which contains rich side information, becomes an essential part to boost the recommendation performance and improve its explainability. However, existing knowledge-aware recommendation methods directly perform information propagation on KG and user-item bipartite graph, ignoring the impacts of \textit{task-irrelevant knowledge propagation} and \textit{vulnerability to interaction noise}, which limits their performance. To solve these issues, we propose a robust knowledge-aware recommendation framework, called \textit{Knowledge-refined Denoising Network} (KRDN), to prune the task-irrelevant knowledge associations and noisy implicit feedback simultaneously. KRDN consists of an adaptive knowledge refining strategy and a contrastive denoising mechanism, which are able to automatically distill high-quality KG triplets for aggregation and prune noisy implicit feedback respectively. Besides, we also design the self-adapted loss function and the gradient estimator for mod
    
[^2]: 面向主题的黑盒神经排名模型对抗攻击

    Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models. (arXiv:2304.14867v1 [cs.IR])

    [http://arxiv.org/abs/2304.14867](http://arxiv.org/abs/2304.14867)

    本文提出了面向主题的对抗排名攻击任务，以提高同一主题的一组查询中的特定文档排名，并提出了基于替代排名模型的新框架来改善攻击性能。

    

    神经排名模型（NRMs）在信息检索中受到了极大的关注。但不幸的是，NRMs可能继承了一般神经网络的对抗性漏洞，这可能被黑帽搜索引擎优化从业者利用。最近，人们在配对攻击设置中探索了对NRMs的对抗攻击，为特定查询的目标文档生成对抗扰动。本文关注更通用类型的扰动，引入了针对NRMs的面向主题的对抗排名攻击任务，旨在找到一种无法察觉的扰动，可以在同一主题的一组查询中提高目标文档的排名。我们为任务定义了静态和动态设置，并专注于基于决策的黑盒攻击。我们提出了一种新的框架来改进基于主题的攻击性能，基于替代排名模型。攻击问题被形式化为马尔可夫决策过程（MDP）。

    Neural ranking models (NRMs) have attracted considerable attention in information retrieval. Unfortunately, NRMs may inherit the adversarial vulnerabilities of general neural networks, which might be leveraged by black-hat search engine optimization practitioners. Recently, adversarial attacks against NRMs have been explored in the paired attack setting, generating an adversarial perturbation to a target document for a specific query. In this paper, we focus on a more general type of perturbation and introduce the topic-oriented adversarial ranking attack task against NRMs, which aims to find an imperceptible perturbation that can promote a target document in ranking for a group of queries with the same topic. We define both static and dynamic settings for the task and focus on decision-based black-box attacks. We propose a novel framework to improve topic-oriented attack performance based on a surrogate ranking model. The attack problem is formalized as a Markov decision process (MDP)
    
[^3]: 通过Prompt学习的知识密集语言任务的统一生成检索器

    A Unified Generative Retriever for Knowledge-Intensive Language Tasks via Prompt Learning. (arXiv:2304.14856v1 [cs.IR])

    [http://arxiv.org/abs/2304.14856](http://arxiv.org/abs/2304.14856)

    本文提出了一种统一的生成检索器(UGR)，它将任务特定的效果与在KILTs中不同检索任务的强健性能相结合。UGR采用基于Prompt的学习方法，使其能够适应各种检索任务，并通过实验在五个KILTs基准测试上显著优于最先进的基线。

    

    知识密集语言任务(KILTs)受益于从大型外部知识语料库中检索高质量的相关内容。学习任务特定的检索器，返回恰当语义粒度的相关内容(如文档检索器、段落检索器、句子检索器和实例检索器)有助于在端到端任务上获得更好的性能。但是任务特定的检索器通常对新领域和任务具有较差的泛化能力，在实践中部署多种专门的检索器可能成本过高。我们提出了一种统一的生成检索器(UGR)，它将任务特定的效果与在KILTs中不同检索任务的强健性能相结合。为了实现这个目标，我们做出了两个主要贡献:(i)为了将不同的检索任务统一成单一的生成形式，我们介绍了一个基于n-gram的标识符，用于在KILTs中不同粒度级别上识别相关内容。(ii)为了解决不同任务的不同检索需求，我们提出了一种基于Prompt的学习方法，使UGR能够通过从任务特定的Prompts中学习来适应各种检索任务。在五个KILTs基准测试上的实验结果表明，我们的UGR在所有任务上只用一个模型就显著优于最先进的基线。

    Knowledge-intensive language tasks (KILTs) benefit from retrieving high-quality relevant contexts from large external knowledge corpora. Learning task-specific retrievers that return relevant contexts at an appropriate level of semantic granularity, such as a document retriever, passage retriever, sentence retriever, and entity retriever, may help to achieve better performance on the end-to-end task. But a task-specific retriever usually has poor generalization ability to new domains and tasks, and it may be costly to deploy a variety of specialised retrievers in practice. We propose a unified generative retriever (UGR) that combines task-specific effectiveness with robust performance over different retrieval tasks in KILTs. To achieve this goal, we make two major contributions: (i) To unify different retrieval tasks into a single generative form, we introduce an n-gram-based identifier for relevant contexts at different levels of granularity in KILTs. And (ii) to address different ret
    
[^4]: 由什么构成？学习修车领域组件的可信材料

    Made of Steel? Learning Plausible Materials for Components in the Vehicle Repair Domain. (arXiv:2304.14745v1 [cs.CL])

    [http://arxiv.org/abs/2304.14745](http://arxiv.org/abs/2304.14745)

    本文提出了一种新方法，通过探索预训练语言模型（PLM）学习车辆维修领域组件的特定材料，成功克服了数据稀疏性问题和缺乏注释数据集的问题。

    

    我们提出了一种新的方法，通过探索预训练语言模型（PLM）中的cloze任务样式设置来学习车辆维修领域组件的特定材料，以克服缺乏注释数据集的问题。我们设计了一种新方法，聚合了一组cloze查询模板的显著预测，并表明使用小型高质量或定制的维基百科语料库的领域自适应可以提高性能。当探索资源紧缺的替代方案时，我们发现精简的PLM明显优于经典的基于模式的算法。此外，考虑到我们领域特定组件的98％都是多词表达式，我们成功地利用组成性假设来解决数据稀疏性问题。

    We propose a novel approach to learn domain-specific plausible materials for components in the vehicle repair domain by probing Pretrained Language Models (PLMs) in a cloze task style setting to overcome the lack of annotated datasets. We devise a new method to aggregate salient predictions from a set of cloze query templates and show that domain-adaptation using either a small, high-quality or a customized Wikipedia corpus boosts performance. When exploring resource-lean alternatives, we find a distilled PLM clearly outperforming a classic pattern-based algorithm. Further, given that 98% of our domain-specific components are multiword expressions, we successfully exploit the compositionality assumption as a way to address data sparsity.
    
[^5]: 基于对比知识蒸馏的集成建模在序列推荐中的应用

    Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation. (arXiv:2304.14668v1 [cs.IR])

    [http://arxiv.org/abs/2304.14668](http://arxiv.org/abs/2304.14668)

    本研究提出了一种基于对比知识蒸馏的集成建模方法EMKD，它采用多个并行网络作为序列编码器，在序列推荐中根据所有网络的输出分布推荐物品。实验证明，EMKD在两个真实世界数据集上的表现显著优于最先进的方法。

    

    序列推荐旨在捕捉用户的动态兴趣，预测用户下一次的偏好物品。多数方法使用深度神经网络作为序列编码器生成用户和物品表示。现有工作主要侧重于设计更强的序列编码器。然而，很少有尝试使用训练一组网络作为序列编码器的方法，这比单个网络更强大，因为一组并行网络可以产生多样化的预测结果，从而获得更好的准确性。本文提出了一种基于对比知识蒸馏的集成建模方法，即EMKD，在序列推荐中使用多个并行网络作为序列编码器，并根据所有这些网络的输出分布推荐物品。为了促进并行网络之间的知识转移，我们提出了一种新颖的对比知识蒸馏方法，它将知识从教师网络转移到多个学生网络中。在两个真实世界数据集上的实验表明，我们提出的EMKD显著优于最先进的序列推荐方法和集成基线。

    Sequential recommendation aims to capture users' dynamic interest and predicts the next item of users' preference. Most sequential recommendation methods use a deep neural network as sequence encoder to generate user and item representations. Existing works mainly center upon designing a stronger sequence encoder. However, few attempts have been made with training an ensemble of networks as sequence encoders, which is more powerful than a single network because an ensemble of parallel networks can yield diverse prediction results and hence better accuracy. In this paper, we present Ensemble Modeling with contrastive Knowledge Distillation for sequential recommendation (EMKD). Our framework adopts multiple parallel networks as an ensemble of sequence encoders and recommends items based on the output distributions of all these networks. To facilitate knowledge transfer between parallel networks, we propose a novel contrastive knowledge distillation approach, which performs knowledge tran
    
[^6]: 信息检索的多元表示学习

    Multivariate Representation Learning for Information Retrieval. (arXiv:2304.14522v1 [cs.IR])

    [http://arxiv.org/abs/2304.14522](http://arxiv.org/abs/2304.14522)

    本论文提出一种多元分布模型的信息检索表示学习框架，可无缝集成到现有近似最近邻算法中以实现高效检索。

    

    稠密检索模型使用双编码器网络架构来学习查询和文档的表示形式，这些表示形式通常采用向量表示，它们的相似性通常使用点积函数计算。本文提出一种新的稠密检索表示学习框架。我们的框架不是学习每个查询和文档的向量，而是学习多元分布，并使用负多元KL散度计算分布之间的相似性。为了简化和提高效率，我们假设这些分布是多维正态分布，然后训练大型语言模型来生成这些分布的均值和方差向量。我们为所提出的框架提供了理论基础，并展示了它可以无缝地集成到现有的近似最近邻算法中以实现高效检索。我们进行了广泛的实验，覆盖了各种不同的基准数据集和评估指标。

    Dense retrieval models use bi-encoder network architectures for learning query and document representations. These representations are often in the form of a vector representation and their similarities are often computed using the dot product function. In this paper, we propose a new representation learning framework for dense retrieval. Instead of learning a vector for each query and document, our framework learns a multivariate distribution and uses negative multivariate KL divergence to compute the similarity between distributions. For simplicity and efficiency reasons, we assume that the distributions are multivariate normals and then train large language models to produce mean and variance vectors for these distributions. We provide a theoretical foundation for the proposed framework and show that it can be seamlessly integrated into the existing approximate nearest neighbor algorithms to perform retrieval efficiently. We conduct an extensive suite of experiments on a wide range 
    
[^7]: 评估GPT-3.5和GPT-4在支持医疗保健信息需求方面的实际作用

    Evaluation of GPT-3.5 and GPT-4 for supporting real-world information needs in healthcare delivery. (arXiv:2304.13714v1 [cs.AI])

    [http://arxiv.org/abs/2304.13714](http://arxiv.org/abs/2304.13714)

    本研究评估了在临床环境中使用GPT-3.5和GPT-4解决医学问题的安全性以及与信息技术咨询服务报告的一致性。研究结果表明，两个LLMs都可以以安全和一致的方式满足医生的信息需求。

    

    尽管在医疗保健领域使用大型语言模型(LLMs)越来越受关注，但当前的探索并未评估LLMs在临床环境中的实用性和安全性。我们的目标是确定两个LLM是否可以以安全和一致的方式满足由医生提交的信息需求问题。我们将66个来自信息技术咨询服务的问题通过简单的提示提交给GPT-3.5和GPT-4。12名医生评估了LLM响应对患者造成伤害的可能性以及与信息技术咨询服务的现有报告的一致性。医生的评估基于多数票汇总。对于没有任何问题，大多数医生认为任何一个LLM响应都不会造成伤害。对于GPT-3.5，8个问题的响应与信息技术咨询报告一致，20个不一致，9个无法评估。有29个响应没有多数票表示“同意”、“不同意”和“无法评估”。

    Despite growing interest in using large language models (LLMs) in healthcare, current explorations do not assess the real-world utility and safety of LLMs in clinical settings. Our objective was to determine whether two LLMs can serve information needs submitted by physicians as questions to an informatics consultation service in a safe and concordant manner. Sixty six questions from an informatics consult service were submitted to GPT-3.5 and GPT-4 via simple prompts. 12 physicians assessed the LLM responses' possibility of patient harm and concordance with existing reports from an informatics consultation service. Physician assessments were summarized based on majority vote. For no questions did a majority of physicians deem either LLM response as harmful. For GPT-3.5, responses to 8 questions were concordant with the informatics consult report, 20 discordant, and 9 were unable to be assessed. There were 29 responses with no majority on "Agree", "Disagree", and "Unable to assess". Fo
    

