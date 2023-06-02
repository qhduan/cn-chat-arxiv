# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [End-to-end Knowledge Retrieval with Multi-modal Queries.](http://arxiv.org/abs/2306.00424) | 该论文提出了一个新的多模态检索任务，引入了一个名为“ReViz”的检索模型，可以直接处理文本和图像输入以实现端到端的知识检索，同时提出了一种有效的预训练任务，并在两个数据集上展示了优越的检索性能。 |
| [^2] | [A Survey on Fairness-aware Recommender Systems.](http://arxiv.org/abs/2306.00403) | 本综述对现有的公正感知推荐系统方法和实践进行了总结分析，详细介绍了相关的概念定义、分类、方法和需解决的问题，并提出了未来的研究方向。 |
| [^3] | [TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest.](http://arxiv.org/abs/2306.00248) | 本文介绍了Pinterest推荐系统的架构和TransAct模型。TransAct是一个从用户实时活动中提取短期偏好的序列模型。本文还介绍了通过混合排名方法结合直接在实时用户活动上学习和在较长时间段内学习批量用户表示的优点。 |
| [^4] | [VMap: An Interactive Rectangular Space-filling Visualization for Map-like Vertex-centric Graph Exploration.](http://arxiv.org/abs/2306.00120) | VMap是一种交互式矩形填充地图可视化方法，用于顶点为中心的图表探索。通过集成DAR矩形分割算法、双阶段矩形调整算法和基于模拟退火的启发式优化器，它能够优化矩形纵横比、顶点-边交叉和数据编码精度。 |
| [^5] | [A Survey on Large Language Models for Recommendation.](http://arxiv.org/abs/2305.19860) | 本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。 |
| [^6] | [Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation.](http://arxiv.org/abs/2305.18885) | 本文提出了一种面向多准则推荐的标准偏好感知轻量图卷积网络，该方法结合了MC扩展图，可以准确地捕捉用户的标准偏好，并进一步将用户对各个标准的偏好合并到最终的推荐列表中。 |
| [^7] | [Graph Masked Autoencoder for Sequential Recommendation.](http://arxiv.org/abs/2305.04619) | 提出了一种简单而有效的基于图遮盖自编码器的序列推荐系统，它使用基于图的注意力机制暴露出带有遮盖的项目序列，自适应动态提取全局项目转换信息进行自监督增强，在具有较少标记样本的情况下始终比最先进的序列推荐方法表现出更好的性能，而且对数据损坏和缺失情况具有鲁棒性。 |
| [^8] | [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?.](http://arxiv.org/abs/2305.01555) | 本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。 |
| [^9] | [Explaining Recommendation System Using Counterfactual Textual Explanations.](http://arxiv.org/abs/2303.11160) | 本文提供了一种利用反事实推理来生成可理解解释的方法，其在推荐系统上取得了成功应用。 |
| [^10] | [Reasoning with Language Model Prompting: A Survey.](http://arxiv.org/abs/2212.09597) | 本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。 |
| [^11] | [Efficient Bi-Level Optimization for Recommendation Denoising.](http://arxiv.org/abs/2210.10321) | 本文提出了一种推荐去噪的高效双层优化方法，该方法可以迭代调整推荐模型，以考虑前几次迭代中为每个反馈分配的权重。 |

# 详细

[^1]: 多模态查询的端到端知识检索

    End-to-end Knowledge Retrieval with Multi-modal Queries. (arXiv:2306.00424v1 [cs.CL])

    [http://arxiv.org/abs/2306.00424](http://arxiv.org/abs/2306.00424)

    该论文提出了一个新的多模态检索任务，引入了一个名为“ReViz”的检索模型，可以直接处理文本和图像输入以实现端到端的知识检索，同时提出了一种有效的预训练任务，并在两个数据集上展示了优越的检索性能。

    

    我们研究了多模态查询下的知识检索，即包含图像和文本的查询的任务，这是与之前的跨模态检索研究不同的挑战性任务。我们创建了一个名为ReMuQ的新数据集，用于评估这个任务的进展。ReMuQ需要一个系统通过整合来自文本和图像查询的内容来检索大规模语料库中的知识。我们引入了一个叫做“ReViz”的检索模型，这个模型可以直接处理输入的文本和图像，以端到端方式检索相关知识，而不依赖于像目标检测器或标题生成器等中间模块。我们介绍了一种新的预训练任务，有效地学习多模态查询下的知识检索，并在下游任务中提高了性能。我们展示了在零-shot设置下，在两个数据集（ReMuQ和OK-VQA）上的检索性能优越以及在这些数据集上微调后进一步的性能提升。

    We investigate knowledge retrieval with multi-modal queries, i.e. queries containing information split across image and text inputs, a challenging task that differs from previous work on cross-modal retrieval. We curate a new dataset called ReMuQ for benchmarking progress on this task. ReMuQ requires a system to retrieve knowledge from a large corpus by integrating contents from both text and image queries. We introduce a retriever model ``ReViz'' that can directly process input text and images to retrieve relevant knowledge in an end-to-end fashion without being dependent on intermediate modules such as object detectors or caption generators. We introduce a new pretraining task that is effective for learning knowledge retrieval with multimodal queries and also improves performance on downstream tasks. We demonstrate superior performance in retrieval on two datasets (ReMuQ and OK-VQA) under zero-shot settings as well as further improvements when finetuned on these datasets.
    
[^2]: 公正感知推荐系统综述

    A Survey on Fairness-aware Recommender Systems. (arXiv:2306.00403v1 [cs.IR])

    [http://arxiv.org/abs/2306.00403](http://arxiv.org/abs/2306.00403)

    本综述对现有的公正感知推荐系统方法和实践进行了总结分析，详细介绍了相关的概念定义、分类、方法和需解决的问题，并提出了未来的研究方向。

    

    作为信息过滤服务，推荐系统通过提供个性化建议和帮助人们做出决策极大地丰富了我们的日常生活，使它们在信息时代对人类社会至关重要和不可或缺。然而，随着人们对它们的依赖程度增加，最近的研究显示，由于其不公平性（例如工作推荐中的性别歧视），推荐系统对社会和个人可能拥有无意识的影响。为了开发可信赖的服务，设计公正感知的推荐系统以缓解这些偏见问题至关重要。本综述概述了现有的公正性推荐系统方法和实践。首先，我们介绍了不同推荐场景下的公正性概念，全面分类当前的进展并介绍了促进推荐系统不同阶段的公正性的典型方法。接下来，在介绍数据集和评估方法后，我们讨论了一些未来的研究方向。

    As information filtering services, recommender systems have extremely enriched our daily life by providing personalized suggestions and facilitating people in decision-making, which makes them vital and indispensable to human society in the information era. However, as people become more dependent on them, recent studies show that recommender systems potentially own unintentional impacts on society and individuals because of their unfairness (e.g., gender discrimination in job recommendations). To develop trustworthy services, it is crucial to devise fairness-aware recommender systems that can mitigate these bias issues. In this survey, we summarise existing methodologies and practices of fairness in recommender systems. Firstly, we present concepts of fairness in different recommendation scenarios, comprehensively categorize current advances, and introduce typical methods to promote fairness in different stages of recommender systems. Next, after introducing datasets and evaluation me
    
[^3]: TransAct: Pinterest实时用户行为模型中的Transformer-Based推荐

    TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest. (arXiv:2306.00248v1 [cs.IR])

    [http://arxiv.org/abs/2306.00248](http://arxiv.org/abs/2306.00248)

    本文介绍了Pinterest推荐系统的架构和TransAct模型。TransAct是一个从用户实时活动中提取短期偏好的序列模型。本文还介绍了通过混合排名方法结合直接在实时用户活动上学习和在较长时间段内学习批量用户表示的优点。

    

    编码用户活动以进行下一步行动预测的序列模型已成为构建大规模个性化推荐系统的常见设计选择。 传统的序列推荐方法要么利用实时用户操作进行端到端学习，要么以脱机批量生成的方式单独学习用户表示。 本文(1)介绍了Pinterest Homefeed的排名架构，即我们的个性化推荐产品和最大的参与面；(2)提出了TransAct，一种从用户实时活动中提取短期偏好的序列模型；(3)描述了我们的混合排名方法，即通过TransAct的端到端序列建模与批量生成的用户嵌入混合。 混合方法允许我们结合直接在实时用户活动上学习以获得响应性的优点和在较长时间段内学习的批量用户表示的成本效益。 我们描述了实验结果......（原文内容省略）

    Sequential models that encode user activity for next action prediction have become a popular design choice for building web-scale personalized recommendation systems. Traditional methods of sequential recommendation either utilize end-to-end learning on realtime user actions, or learn user representations separately in an offline batch-generated manner. This paper (1) presents Pinterest's ranking architecture for Homefeed, our personalized recommendation product and the largest engagement surface; (2) proposes TransAct, a sequential model that extracts users' short-term preferences from their realtime activities; (3) describes our hybrid approach to ranking, which combines end-to-end sequential modeling via TransAct with batch-generated user embeddings. The hybrid approach allows us to combine the advantages of responsiveness from learning directly on realtime user activity with the cost-effectiveness of batch user representations learned over a longer time period. We describe the resu
    
[^4]: VMap: 一种交互式矩形填充可视化地图，用于顶点为中心的图表探索。

    VMap: An Interactive Rectangular Space-filling Visualization for Map-like Vertex-centric Graph Exploration. (arXiv:2306.00120v1 [cs.GR])

    [http://arxiv.org/abs/2306.00120](http://arxiv.org/abs/2306.00120)

    VMap是一种交互式矩形填充地图可视化方法，用于顶点为中心的图表探索。通过集成DAR矩形分割算法、双阶段矩形调整算法和基于模拟退火的启发式优化器，它能够优化矩形纵横比、顶点-边交叉和数据编码精度。

    

    我们提出了一种名为VMap的地图状矩形填充可视化方法，用于顶点为中心的图表探索。为优化矩形的纵横比、顶点-边交叉和数据编码精度等方面，现有的可视化方法存在支持不足。为应对这些问题，VMap集成了三种新组建：（1）基于想要的纵横比（DAR）的矩形分割算法，（2）双阶段矩形调整算法，（3）基于模拟退火的启发式优化器。首先，为了生成输入图表的矩形填充布局，我们将图表的2D嵌入方式划分为矩形，并优化矩形的纵横比，以达到想要的纵横比。其次，为了在矩形之间路由图表边缘，而不会出现顶点-边缘重叠，我们设计了一个双阶段算法来调整矩形布局，以在矩形之间插入边框空间。第三，为了通过考虑多种视觉准则来生成和排列矩形，我们设计了一个包含模拟退火的启发式优化器。

    We present VMap, a map-like rectangular space-filling visualization, to perform vertex-centric graph exploration. Existing visualizations have limited support for quality optimization among rectangular aspect ratios, vertex-edge intersection, and data encoding accuracy. To tackle this problem, VMap integrates three novel components: (1) a desired-aspect-ratio (DAR) rectangular partitioning algorithm, (2) a two-stage rectangle adjustment algorithm, and (3) a simulated annealing based heuristic optimizer. First, to generate a rectangular space-filling layout of an input graph, we subdivide the 2D embedding of the graph into rectangles with optimization of rectangles' aspect ratios toward a desired aspect ratio. Second, to route graph edges between rectangles without vertex-edge occlusion, we devise a two-stage algorithm to adjust a rectangular layout to insert border space between rectangles. Third, to produce and arrange rectangles by considering multiple visual criteria, we design a si
    
[^5]: 基于大语言模型的推荐系统综述

    A Survey on Large Language Models for Recommendation. (arXiv:2305.19860v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.19860](http://arxiv.org/abs/2305.19860)

    本综述介绍了基于大语言模型的推荐系统，提出了判别式LLMs和生成式LLMs两种模型范式，总结了这些模型的最新进展，强调了该领域的挑战和研究方向。

    

    大语言模型（LLMs）已成为自然语言处理（NLP）领域强大的工具，并在推荐系统领域引起了重视。这些模型使用自监督学习在海量数据上进行训练，已在学习通用表示方面取得了显着成功，并有可能通过一些有效的转移技术（如微调和提示调整）等手段提高推荐系统的各个方面的性能。利用大语言模型增强推荐质量的关键是利用它们高质量的文本特征表示和大量的外部知识覆盖，建立项目和用户之间的相关性。为了全面了解现有基于LLM的推荐系统，本综述提出了一种分类法，将这些模型分为两种主要范式，分别是判别式LLMs和生成式LLMs。此外，我们总结了这些范式的最新进展，并强调了这个新兴领域的挑战和开放性研究问题。

    Large Language Models (LLMs) have emerged as powerful tools in the field of Natural Language Processing (NLP) and have recently gained significant attention in the domain of Recommendation Systems (RS). These models, trained on massive amounts of data using self-supervised learning, have demonstrated remarkable success in learning universal representations and have the potential to enhance various aspects of recommendation systems by some effective transfer techniques such as fine-tuning and prompt tuning, and so on. The crucial aspect of harnessing the power of language models in enhancing recommendation quality is the utilization of their high-quality representations of textual features and their extensive coverage of external knowledge to establish correlations between items and users. To provide a comprehensive understanding of the existing LLM-based recommendation systems, this survey presents a taxonomy that categorizes these models into two major paradigms, respectively Discrimi
    
[^6]: 标准比评分更重要：面向多准则推荐的标准偏好感知轻量图卷积网络

    Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation. (arXiv:2305.18885v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2305.18885](http://arxiv.org/abs/2305.18885)

    本文提出了一种面向多准则推荐的标准偏好感知轻量图卷积网络，该方法结合了MC扩展图，可以准确地捕捉用户的标准偏好，并进一步将用户对各个标准的偏好合并到最终的推荐列表中。

    

    多准则推荐系统现在在广泛的电子商务领域中利用多准则 (MC) 评分信息，而深度学习中的图神经网络 (GNN) 已经被广泛应用于各种推荐系统的开发中。在这种情况下，本文首次尝试使用GNN辅助设计MC推荐系统。具体而言，我们提出了一种新颖的标准偏好感知轻量图卷积方法(CPA-LGC),可以准确捕捉用户的标准偏好以及复杂高阶连接中的协作信号。本文在MC扩展图上构建了一个能够将用户-物品MC评分转换为扩展二分图的MC扩展图，再进一步将标准重要性编码到图卷积过程中，并引入了一种新的标准偏好感知聚合方法来将用户对不同标准的偏好合并到最终的推荐列表中。

    The multi-criteria (MC) recommender system, which leverages MC rating information in a wide range of e-commerce areas, is ubiquitous nowadays. Surprisingly, although graph neural networks (GNNs) have been widely applied to develop various recommender systems due to GNN's high expressive capability in learning graph representations, it has been still unexplored how to design MC recommender systems with GNNs. In light of this, we make the first attempt towards designing a GNN-aided MC recommender system. Specifically, rather than straightforwardly adopting existing GNN-based recommendation methods, we devise a novel criteria preference-aware light graph convolution CPA-LGC method, which is capable of precisely capturing the criteria preference of users as well as the collaborative signal in complex high-order connectivities. To this end, we first construct an MC expansion graph that transforms user--item MC ratings into an expanded bipartite graph to potentially learn from the collaborat
    
[^7]: 基于图形遮盖自编码器的序列推荐系统

    Graph Masked Autoencoder for Sequential Recommendation. (arXiv:2305.04619v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2305.04619](http://arxiv.org/abs/2305.04619)

    提出了一种简单而有效的基于图遮盖自编码器的序列推荐系统，它使用基于图的注意力机制暴露出带有遮盖的项目序列，自适应动态提取全局项目转换信息进行自监督增强，在具有较少标记样本的情况下始终比最先进的序列推荐方法表现出更好的性能，而且对数据损坏和缺失情况具有鲁棒性。

    

    虽然一些强大的神经网络架构（例如Transformer、图神经网络）通过高阶项依赖建模在序列推荐中实现了改进的性能，但它们可能在标签稀缺情况下表现出较差的表征能力。为了解决标签不足的问题，对比学习（CL）已经引起了近期的关注，通过嵌入对比来进行自我监督的数据增强。然而，由于其对比视图生成策略的手工制定特性，现有的CL增强模型不仅难以在不同的序列推荐任务中产生一致的性能，还可能对用户行为数据噪声不具有鲁棒性。鉴于这一点，我们提出了一种简单而有效的自适应全局信息提取的图遮盖自编码器增强的序列推荐系统（MAERec）来解决这个问题。它自然地避免了上述问题，得益于其独特的数据重构机制。具体而言，我们的模型使用基于图的注意力机制，暴露出带有遮盖的项目序列，使表示不仅利用本地顺序信息，还利用项目之间的全局相关性。我们在四个基准数据集上对我们的方法进行了广泛评估。实验结果表明，我们的模型在具有较少标记样本的情况下始终比最先进的序列推荐方法表现出更好的性能，而且对数据损坏和缺失情况具有鲁棒性。

    While some powerful neural network architectures (e.g., Transformer, Graph Neural Networks) have achieved improved performance in sequential recommendation with high-order item dependency modeling, they may suffer from poor representation capability in label scarcity scenarios. To address the issue of insufficient labels, Contrastive Learning (CL) has attracted much attention in recent methods to perform data augmentation through embedding contrasting for self-supervision. However, due to the hand-crafted property of their contrastive view generation strategies, existing CL-enhanced models i) can hardly yield consistent performance on diverse sequential recommendation tasks; ii) may not be immune to user behavior data noise. In light of this, we propose a simple yet effective Graph Masked AutoEncoder-enhanced sequential Recommender system (MAERec) that adaptively and dynamically distills global item transitional information for self-supervised augmentation. It naturally avoids the abov
    
[^8]: 如何发挥大语言模型在少样本关系抽取中的能力？

    How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?. (arXiv:2305.01555v1 [cs.CL])

    [http://arxiv.org/abs/2305.01555](http://arxiv.org/abs/2305.01555)

    本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。

    

    语言模型的扩展已经彻底改变了广泛的自然语言处理任务，但是使用大型语言模型进行少样本关系抽取还没有得到全面探索。本文通过详细实验，研究了使用GPT-3.5进行少样本关系抽取的基本方法——上下文学习和数据生成。为了增强少样本性能，我们进一步提出了与任务相关的指导说明和约束模式下的数据生成。我们观察到，在上下文学习的情况下，可以实现与以前的提示学习方法相当的性能，而使用大型语言模型的数据生成可以推动以前的解决方案以在四个广泛研究的关系抽取数据集上获得新的最先进的少样本结果。我们希望我们的工作可以激发未来对大型语言模型在少样本关系抽取中的能力的研究。代码可以在 \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm} 中找到。

    Scaling language models have revolutionized widespread NLP tasks, yet little comprehensively explored few-shot relation extraction with large language models. In this paper, we investigate principal methodologies, in-context learning and data generation, for few-shot relation extraction via GPT-3.5 through exhaustive experiments. To enhance few-shot performance, we further propose task-related instructions and schema-constrained data generation. We observe that in-context learning can achieve performance on par with previous prompt learning approaches, and data generation with the large language model can boost previous solutions to obtain new state-of-the-art few-shot results on four widely-studied relation extraction datasets. We hope our work can inspire future research for the capabilities of large language models in few-shot relation extraction. Code is available in \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^9]: 利用反事实文本解释来解释推荐系统

    Explaining Recommendation System Using Counterfactual Textual Explanations. (arXiv:2303.11160v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.11160](http://arxiv.org/abs/2303.11160)

    本文提供了一种利用反事实推理来生成可理解解释的方法，其在推荐系统上取得了成功应用。

    

    目前，在人工智能领域，有大量的研究致力于改进深度学习模型的可解释性和可解读性。研究发现，如果最终用户理解某些输出的原因，就更容易信任系统。推荐系统是需要进行改进以使其输出更加可解释的系统之一。产生更可解释的输出的一种方法是使用反事实推理，这涉及对最小要素进行修改，以生成导致系统输出变化的反事实项目。这一过程允许识别对期望输出有重大影响的输入要素，从而提供有效的解释。在本文中，我们提出了一种方法来生成针对表格和文本要素的反事实解释。我们在三个真实数据集上评估了我们提出的方法的性能，并证明它在为最终用户提供可理解的解释方面是有效的。

    Currently, there is a significant amount of research being conducted in the field of artificial intelligence to improve the explainability and interpretability of deep learning models. It is found that if end-users understand the reason for the production of some output, it is easier to trust the system. Recommender systems are one example of systems that great efforts have been conducted to make their output more explainable. One method for producing a more explainable output is using counterfactual reasoning, which involves altering minimal features to generate a counterfactual item that results in changing the output of the system. This process allows the identification of input features that have a significant impact on the desired output, leading to effective explanations. In this paper, we present a method for generating counterfactual explanations for both tabular and textual features. We evaluated the performance of our proposed method on three real-world datasets and demonstra
    
[^10]: 使用语言模型提示进行推理：一项调查

    Reasoning with Language Model Prompting: A Survey. (arXiv:2212.09597v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09597](http://arxiv.org/abs/2212.09597)

    本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。

    

    推理作为复杂问题解决的重要能力，可以为医疗诊断、谈判等各种实际应用提供后端支持。本文对使用语言模型提示进行推理的前沿研究进行了综合调查。我们介绍了研究成果的比较和总结，并提供了系统资源以帮助初学者。我们还讨论了新兴推理能力出现的潜在原因，并突出了未来的研究方向。资源可在 https://github.com/zjunlp/Prompt4ReasoningPapers 上获取（定期更新）。

    Reasoning, as an essential ability for complex problem-solving, can provide back-end support for various real-world applications, such as medical diagnosis, negotiation, etc. This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting. We introduce research works with comparisons and summaries and provide systematic resources to help beginners. We also discuss the potential reasons for emerging such reasoning abilities and highlight future research directions. Resources are available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated periodically).
    
[^11]: 推荐去噪的高效双层优化方法

    Efficient Bi-Level Optimization for Recommendation Denoising. (arXiv:2210.10321v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.10321](http://arxiv.org/abs/2210.10321)

    本文提出了一种推荐去噪的高效双层优化方法，该方法可以迭代调整推荐模型，以考虑前几次迭代中为每个反馈分配的权重。

    

    在实际推荐系统中，获得明确的用户反馈（例如评分）通常会受到需要用户积极参与的限制。为了缓解这个问题，利用用户浏览期间生成的隐式反馈（例如点击）作为可行的替代方法。然而，隐式反馈具有很高的噪声，这会显着损害推荐质量。本文中，我们将推荐去噪建模为一个双层优化问题，通过考虑前几次迭代中为每个反馈分配的权重来迭代地调整推荐模型。实验结果表明，该方法在两个大规模推荐数据集上的表现优于现有的基准方法。

    The acquisition of explicit user feedback (e.g., ratings) in real-world recommender systems is often hindered by the need for active user involvement. To mitigate this issue, implicit feedback (e.g., clicks) generated during user browsing is exploited as a viable substitute. However, implicit feedback possesses a high degree of noise, which significantly undermines recommendation quality. While many methods have been proposed to address this issue by assigning varying weights to implicit feedback, two shortcomings persist: (1) the weight calculation in these methods is iteration-independent, without considering the influence of weights in previous iterations, and (2) the weight calculation often relies on prior knowledge, which may not always be readily available or universally applicable.  To overcome these two limitations, we model recommendation denoising as a bi-level optimization problem. The inner optimization aims to derive an effective model for the recommendation, as well as g
    

