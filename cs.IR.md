# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-autoregressive Generative Models for Reranking Recommendation](https://arxiv.org/abs/2402.06871) | 本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。 |
| [^2] | [Future Impact Decomposition in Request-level Recommendations](https://arxiv.org/abs/2401.16108) | 在请求级别的推荐系统中，我们通过比较标准方法和基于物品级别的演员-评论家框架在模拟和在线实验中的性能，证明了基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。 |
| [^3] | [Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models.](http://arxiv.org/abs/2401.15269) | 本论文介绍了一种名为Self-BioRAG的框架，通过使用检索和自我反思的方法，提高了医疗推理的能力。该框架专注于生成解释、检索领域特定文档以及对生成的响应进行自我反思。 |
| [^4] | [A Survey on Large Language Models for Recommendation.](http://arxiv.org/abs/2305.19860) | 本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。 |
| [^5] | [Graph-guided Personalization for Federated Recommendation.](http://arxiv.org/abs/2305.07866) | 本文提出了一种基于图引导的Federated Recommendation个性化框架（GPFedRec），通过自适应图结构来增强客户端之间的协作，可以同时使用共享和个性化的信息，提高推荐准确性，保护用户隐私。 |

# 详细

[^1]: 非自回归的生成模型用于排序推荐

    Non-autoregressive Generative Models for Reranking Recommendation

    [https://arxiv.org/abs/2402.06871](https://arxiv.org/abs/2402.06871)

    本研究提出了一个非自回归的生成模型用于排序推荐，在多阶段推荐系统中扮演关键角色。该模型旨在提高效率和效果，并解决稀疏训练样本和动态候选项对模型收敛性的挑战。

    

    在多阶段推荐系统中，重新排序通过建模项目之间的内部相关性起到了至关重要的作用。重新排序的关键挑战在于在排列的组合空间中探索最佳序列。最近的研究提出了生成器-评估器学习范式，生成器生成多个可行序列，评估器基于估计的列表得分选择最佳序列。生成器至关重要，而生成模型非常适合生成器函数。当前的生成模型采用自回归策略进行序列生成。然而，在实时工业系统中部署自回归模型是具有挑战性的。因此，我们提出了一个非自回归生成模型用于排序推荐（NAR4Rec），以提高效率和效果。为了解决与稀疏训练样本和动态候选项对模型收敛性的挑战，我们引入了一个m

    In a multi-stage recommendation system, reranking plays a crucial role by modeling the intra-list correlations among items.The key challenge of reranking lies in the exploration of optimal sequences within the combinatorial space of permutations. Recent research proposes a generator-evaluator learning paradigm, where the generator generates multiple feasible sequences and the evaluator picks out the best sequence based on the estimated listwise score. Generator is of vital importance, and generative models are well-suited for the generator function. Current generative models employ an autoregressive strategy for sequence generation. However, deploying autoregressive models in real-time industrial systems is challenging. Hence, we propose a Non-AutoRegressive generative model for reranking Recommendation (NAR4Rec) designed to enhance efficiency and effectiveness. To address challenges related to sparse training samples and dynamic candidates impacting model convergence, we introduce a m
    
[^2]: 请求级别推荐中的未来影响分解

    Future Impact Decomposition in Request-level Recommendations

    [https://arxiv.org/abs/2401.16108](https://arxiv.org/abs/2401.16108)

    在请求级别的推荐系统中，我们通过比较标准方法和基于物品级别的演员-评论家框架在模拟和在线实验中的性能，证明了基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。

    

    在推荐系统中，强化学习解决方案在优化用户和系统之间的交互序列以提高长期性能方面显示出有希望的结果。出于实际原因，策略的动作通常被设计为推荐一组物品以更高效地处理用户的频繁和连续的浏览请求。在这种列表式推荐场景中，用户状态在相应的MDP（马尔可夫决策过程）表述中的每个请求上都会更新。然而，这种请求级别的表述与用户的物品级别行为实质上是不一致的。在这项研究中，我们证明了在请求级别MDP下，基于物品级别的优化方法可以更好地利用物品特性并优化策略的性能。我们通过比较标准请求级别方法和提出的基于物品级别的演员-评论家框架在模拟和在线实验中的性能来支持这一观点。

    In recommender systems, reinforcement learning solutions have shown promising results in optimizing the interaction sequence between users and the system over the long-term performance. For practical reasons, the policy's actions are typically designed as recommending a list of items to handle users' frequent and continuous browsing requests more efficiently. In this list-wise recommendation scenario, the user state is updated upon every request in the corresponding MDP formulation. However, this request-level formulation is essentially inconsistent with the user's item-level behavior. In this study, we demonstrate that an item-level optimization approach can better utilize item characteristics and optimize the policy's performance even under the request-level MDP. We support this claim by comparing the performance of standard request-level methods with the proposed item-level actor-critic framework in both simulation and online experiments. Furthermore, we show that a reward-based fut
    
[^3]: 通过检索和自我反思改善医疗推理能力的检索增强型大型语言模型

    Improving Medical Reasoning through Retrieval and Self-Reflection with Retrieval-Augmented Large Language Models. (arXiv:2401.15269v1 [cs.CL])

    [http://arxiv.org/abs/2401.15269](http://arxiv.org/abs/2401.15269)

    本论文介绍了一种名为Self-BioRAG的框架，通过使用检索和自我反思的方法，提高了医疗推理的能力。该框架专注于生成解释、检索领域特定文档以及对生成的响应进行自我反思。

    

    最近的专有大型语言模型（LLMs），例如GPT-4，在生物医学领域中解决了从多项选择题到长篇生成等多样化挑战的里程碑。为了解决LLMs编码知识无法处理的挑战，已经开发了各种检索增强生成（RAG）方法，通过从知识语料库中搜索文档并无条件或有选择地将其附加到LLMs的输入来进行生成。然而，将现有方法应用于不同领域特定问题时，出现了泛化能力差的问题，导致获取不正确的文档或做出不准确的判断。在本文中，我们介绍了一种可靠的医学文本框架Self-BioRAG，专门用于生成解释、检索领域特定文档和自我反思生成的响应。我们使用了84k个经过过滤的生物医学指令集来训练Self-BioRAG，它具备评估自己的基因

    Recent proprietary large language models (LLMs), such as GPT-4, have achieved a milestone in tackling diverse challenges in the biomedical domain, ranging from multiple-choice questions to long-form generations. To address challenges that still cannot be handled with the encoded knowledge of LLMs, various retrieval-augmented generation (RAG) methods have been developed by searching documents from the knowledge corpus and appending them unconditionally or selectively to the input of LLMs for generation. However, when applying existing methods to different domain-specific problems, poor generalization becomes apparent, leading to fetching incorrect documents or making inaccurate judgments. In this paper, we introduce Self-BioRAG, a framework reliable for biomedical text that specializes in generating explanations, retrieving domain-specific documents, and self-reflecting generated responses. We utilize 84k filtered biomedical instruction sets to train Self-BioRAG that can assess its gene
    
[^4]: 基于大语言模型的推荐系统综述

    A Survey on Large Language Models for Recommendation. (arXiv:2305.19860v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.19860](http://arxiv.org/abs/2305.19860)

    本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。

    

    大语言模型（LLMs）已成为自然语言处理（NLP）领域强大的工具，并在推荐系统领域引起了重视。这些模型使用自监督学习在海量数据上进行训练，已在学习通用表示方面取得了显着成功，并有可能通过一些有效的转移技术（如微调和提示调整）等手段提高推荐系统的各个方面的性能。利用大语言模型增强推荐质量的关键是利用它们高质量的文本特征表示和大量的外部知识覆盖，建立项目和用户之间的相关性。为了全面了解现有基于LLM的推荐系统，本综述提出了一种分类法，将这些模型分为两种主要范式，分别是判别式LLMs和生成式LLMs。此外，我们总结了这些范式的最新进展，并强调了这个新兴领域的挑战和开放性研究问题。

    Large Language Models (LLMs) have emerged as powerful tools in the field of Natural Language Processing (NLP) and have recently gained significant attention in the domain of Recommendation Systems (RS). These models, trained on massive amounts of data using self-supervised learning, have demonstrated remarkable success in learning universal representations and have the potential to enhance various aspects of recommendation systems by some effective transfer techniques such as fine-tuning and prompt tuning, and so on. The crucial aspect of harnessing the power of language models in enhancing recommendation quality is the utilization of their high-quality representations of textual features and their extensive coverage of external knowledge to establish correlations between items and users. To provide a comprehensive understanding of the existing LLM-based recommendation systems, this survey presents a taxonomy that categorizes these models into two major paradigms, respectively Discrimi
    
[^5]: 基于图引导的Federated Recommendation个性化方法

    Graph-guided Personalization for Federated Recommendation. (arXiv:2305.07866v1 [cs.IR])

    [http://arxiv.org/abs/2305.07866](http://arxiv.org/abs/2305.07866)

    本文提出了一种基于图引导的Federated Recommendation个性化框架（GPFedRec），通过自适应图结构来增强客户端之间的协作，可以同时使用共享和个性化的信息，提高推荐准确性，保护用户隐私。

    

    Federated Recommendation是一种新的服务架构，可以在不与服务器共享用户数据的情况下提供推荐。现有方法在每个客户端上部署推荐模型，并通过同步和聚合项目嵌入来协调它们的训练。然而，由于用户通常对某些项目具有多样化的偏好，这些方法会无差别地聚合来自所有客户端的项目嵌入，从而中和了底层用户特定的偏好。这种忽视将使得聚合嵌入变得不太具有区分性，并阻碍个性化推荐。本文提出了一种新颖的基于图引导的Federated Recommendation个性化框架（GPFedRec）。GPFedRec通过利用自适应图结构来捕捉用户偏好的相关性，增强了客户端之间的协作。此外，它将客户端的训练过程制定为统一的联邦优化框架，其中模型可以同时使用共享和个性化的信息。在真实世界的数据集上进行的大量实验表明，GPFedRec在保护用户隐私的同时，在推荐准确性方面显著优于现有的方法。

    Federated Recommendation is a new service architecture providing recommendations without sharing user data with the server. Existing methods deploy a recommendation model on each client and coordinate their training by synchronizing and aggregating item embeddings. However, while users usually hold diverse preferences toward certain items, these methods indiscriminately aggregate item embeddings from all clients, neutralizing underlying user-specific preferences. Such neglect will leave the aggregated embedding less discriminative and hinder personalized recommendations. This paper proposes a novel Graph-guided Personalization framework (GPFedRec) for the federated recommendation. The GPFedRec enhances cross-client collaboration by leveraging an adaptive graph structure to capture the correlation of user preferences. Besides, it guides training processes on clients by formulating them into a unified federated optimization framework, where models can simultaneously use shared and person
    

