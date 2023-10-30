# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Text2Bundle: Towards Personalized Query-based Bundle Generation.](http://arxiv.org/abs/2310.18004) | 本论文提出了一种名为Text2Bundle的框架，用于个性化基于查询的bundle生成。该框架利用了用户从查询中的短期兴趣和从历史交互中的长期偏好，并能够生成与用户意图完全匹配的个性化bundle。 |
| [^2] | [Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation.](http://arxiv.org/abs/2310.17922) | 提出了一种称为MTAMCR的对话推荐问题设定，通过每轮询问涵盖多个属性类型的多选题，提高了互动效率。同时，通过Chain-of-Choice层次化策略学习框架，提高了对话推荐系统的询问效率和推荐效果。 |
| [^3] | [Ranking with Slot Constraints.](http://arxiv.org/abs/2310.17870) | 带有槽约束的排名问题中，我们提出了一种新的排名算法MatchRank，它在候选人按排名顺序被人类决策者评估时，产生最大化填充槽位的排名。算法在理论上具有强大的逼近保证，并且可以高效实现。 (arXiv:2310.17870v1 [cs.IR]) |
| [^4] | [GNN-GMVO: Graph Neural Networks for Optimizing Gross Merchandise Value in Similar Item Recommendation.](http://arxiv.org/abs/2310.17732) | 这项研究设计了一种名为GNN-GMVO的新型图神经网络架构，用于优化电子商务中相似商品推荐的商品总交易价值（GMV）。它解决了传统GNN架构在优化收入相关目标方面的不足，并通过直接优化GMV来保证推荐质量。 |
| [^5] | [Music Recommendation Based on Audio Fingerprint.](http://arxiv.org/abs/2310.17655) | 该研究提出了一种基于音频指纹的音乐推荐方法，结合了不同的音频特征，通过PCA降维并计算指纹间的相似矩阵，成功实现了89%的准确率。 |
| [^6] | [Framework based on complex networks to model and mine patient pathways.](http://arxiv.org/abs/2309.14208) | 该论文提出了一个基于复杂网络的框架，用于建模和挖掘患者路径。该框架包括路径模型、新的相似度测量方法和基于传统中心度的挖掘方法。评估结果表明该框架可有效应用于实际医疗数据分析。 |
| [^7] | [Cross-Modal Retrieval: A Systematic Review of Methods and Future Directions.](http://arxiv.org/abs/2308.14263) | 本文提供了一篇关于跨模态检索的方法和未来方向的系统综述，从浅层统计分析到视觉-语言预训练模型，深入探讨了现有跨模态检索方法的原理和架构。 |
| [^8] | [DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs.](http://arxiv.org/abs/2307.04090) | 本论文提出了一种利用语义知识图自动创建政策辩论案例的方法，通过在争论的语义知识图上进行限制最短路径遍历，有效构建高质量的辩论案例。研究结果表明，在美国竞赛辩论中，利用这种方法显著改进了已有数据集DebateSum，并贡献了新的例子和有用的元数据。通过使用txtai语义搜索和知识图工具链，创建和贡献了9个语义知识图，同时提出了一种独特的评估方法来确定哪个知识图更适合政策辩论案例生成。 |
| [^9] | [Table Detection for Visually Rich Document Images.](http://arxiv.org/abs/2305.19181) | 本研究提出了一种在可视丰富文件图像中进行表格检测的方法。通过将IoU分解为真实覆盖项和预测覆盖项来衡量预测结果的信息丢失程度。此外，通过使用高斯噪声增强的图像大小区域提案和多对一标签分配，进一步改进模型。实验证明，该方法能够在不同数据集上优于最先进方法。 |
| [^10] | [Machine Reading Comprehension using Case-based Reasoning.](http://arxiv.org/abs/2305.14815) | 本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。 |
| [^11] | [Is ChatGPT a Good Recommender? A Preliminary Study.](http://arxiv.org/abs/2304.10149) | 本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。 |
| [^12] | [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.](http://arxiv.org/abs/2304.09542) | 本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。 |

# 详细

[^1]: Text2Bundle: 个性化基于查询的bundle生成

    Text2Bundle: Towards Personalized Query-based Bundle Generation. (arXiv:2310.18004v1 [cs.IR])

    [http://arxiv.org/abs/2310.18004](http://arxiv.org/abs/2310.18004)

    本论文提出了一种名为Text2Bundle的框架，用于个性化基于查询的bundle生成。该框架利用了用户从查询中的短期兴趣和从历史交互中的长期偏好，并能够生成与用户意图完全匹配的个性化bundle。

    

    Bundle生成旨在为用户提供一组物品，并已广泛研究和应用于在线服务平台。现有的bundle生成方法主要利用了用户在常见推荐范式中的历史交互中的偏好，忽略了用户当前明确意图的潜在文本查询。可能存在这样的场景，用户主动使用自然语言描述查询bundle，系统应能够通过用户的查询和偏好生成与用户意图完全匹配的bundle。在这项工作中，我们将这种用户友好的场景定义为基于查询的bundle生成任务，并提出了一种新颖的框架Text2Bundle，该框架同时利用了用户从查询中的短期兴趣和从历史交互中的长期偏好。我们的框架由三个模块组成：(1)一个查询兴趣提取器，从查询中挖掘用户的细粒度兴趣；(2) 一个历史交互偏好提取器，从历史交互中提取用户的长期偏好；(3) 一个bundle生成器，结合前两个模块的信息生成用户个性化的bundle。

    Bundle generation aims to provide a bundle of items for the user, and has been widely studied and applied on online service platforms. Existing bundle generation methods mainly utilized user's preference from historical interactions in common recommendation paradigm, and ignored the potential textual query which is user's current explicit intention. There can be a scenario in which a user proactively queries a bundle with some natural language description, the system should be able to generate a bundle that exactly matches the user's intention through the user's query and preferences. In this work, we define this user-friendly scenario as Query-based Bundle Generation task and propose a novel framework Text2Bundle that leverages both the user's short-term interests from the query and the user's long-term preferences from the historical interactions. Our framework consists of three modules: (1) a query interest extractor that mines the user's fine-grained interests from the query; (2) a
    
[^2]: Chain-of-Choice层次化策略学习用于对话推荐

    Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation. (arXiv:2310.17922v1 [cs.IR])

    [http://arxiv.org/abs/2310.17922](http://arxiv.org/abs/2310.17922)

    提出了一种称为MTAMCR的对话推荐问题设定，通过每轮询问涵盖多个属性类型的多选题，提高了互动效率。同时，通过Chain-of-Choice层次化策略学习框架，提高了对话推荐系统的询问效率和推荐效果。

    

    对话推荐系统通过多轮互动对话来揭示用户偏好，最终导向精确和满意的推荐。然而，现有的对话推荐系统仅限于根据每轮单个属性类型（如颜色）询问二进制或多选题，导致互动轮数过多，降低了用户体验。为解决这个问题，我们提出了一种更现实和高效的对话推荐问题设定，称为多类型属性多轮对话推荐（MTAMCR），该问题设定使得对话推荐系统能够在每轮中询问涵盖多个属性类型的多选题，从而提高互动效率。此外，通过将MTAMCR定义为一项层次化强化学习任务，我们提出了一种Chain-of-Choice层次化策略学习（CoCHPL）框架来提高MTAMCR中的询问效率和推荐效果。

    Conversational Recommender Systems (CRS) illuminate user preferences via multi-round interactive dialogues, ultimately navigating towards precise and satisfactory recommendations. However, contemporary CRS are limited to inquiring binary or multi-choice questions based on a single attribute type (e.g., color) per round, which causes excessive rounds of interaction and diminishes the user's experience. To address this, we propose a more realistic and efficient conversational recommendation problem setting, called Multi-Type-Attribute Multi-round Conversational Recommendation (MTAMCR), which enables CRS to inquire about multi-choice questions covering multiple types of attributes in each round, thereby improving interactive efficiency. Moreover, by formulating MTAMCR as a hierarchical reinforcement learning task, we propose a Chain-of-Choice Hierarchical Policy Learning (CoCHPL) framework to enhance both the questioning efficiency and recommendation effectiveness in MTAMCR. Specifically,
    
[^3]: 带有槽约束的排名问题

    Ranking with Slot Constraints. (arXiv:2310.17870v1 [cs.IR])

    [http://arxiv.org/abs/2310.17870](http://arxiv.org/abs/2310.17870)

    带有槽约束的排名问题中，我们提出了一种新的排名算法MatchRank，它在候选人按排名顺序被人类决策者评估时，产生最大化填充槽位的排名。算法在理论上具有强大的逼近保证，并且可以高效实现。 (arXiv:2310.17870v1 [cs.IR])

    

    我们引入了带有槽约束的排名问题，这可以用来建模各种应用问题 - 从具有不同专业限制槽位的大学录取，到在医学试验中构建符合条件的参与者分层队列。我们发现，传统的概率排名原则（PRP）在带有槽约束的排名问题中可能会非常次优，因此我们提出了一种新的排名算法，称为MatchRank。MatchRank的目标是在候选人按排名顺序由人类决策者进行评估时，产生最大化填充槽位的排名。这样，MatchRank在广义上是PRP的推广，当没有槽约束时，它是PRP的特例。我们的理论分析表明，MatchRank具有强大的逼近保证，没有任何槽位或候选人之间的独立性假设。此外，我们展示了如何高效地实现MatchRank。除了理论保证外，我们还展示了MatchRank的实验结果在不同应用领域的有效性。

    We introduce the problem of ranking with slot constraints, which can be used to model a wide range of application problems -- from college admission with limited slots for different majors, to composing a stratified cohort of eligible participants in a medical trial. We show that the conventional Probability Ranking Principle (PRP) can be highly sub-optimal for slot-constrained ranking problems, and we devise a new ranking algorithm, called MatchRank. The goal of MatchRank is to produce rankings that maximize the number of filled slots if candidates are evaluated by a human decision maker in the order of the ranking. In this way, MatchRank generalizes the PRP, and it subsumes the PRP as a special case when there are no slot constraints. Our theoretical analysis shows that MatchRank has a strong approximation guarantee without any independence assumptions between slots or candidates. Furthermore, we show how MatchRank can be implemented efficiently. Beyond the theoretical guarantees, em
    
[^4]: GNN-GMVO: 用于优化相似商品推荐中的商品总交易价值的图神经网络

    GNN-GMVO: Graph Neural Networks for Optimizing Gross Merchandise Value in Similar Item Recommendation. (arXiv:2310.17732v1 [cs.IR])

    [http://arxiv.org/abs/2310.17732](http://arxiv.org/abs/2310.17732)

    这项研究设计了一种名为GNN-GMVO的新型图神经网络架构，用于优化电子商务中相似商品推荐的商品总交易价值（GMV）。它解决了传统GNN架构在优化收入相关目标方面的不足，并通过直接优化GMV来保证推荐质量。

    

    相似商品推荐是电子商务行业中的关键任务，它帮助客户基于他们感兴趣的产品探索相似和相关的替代品。尽管传统的机器学习模型，图神经网络（GNN）可以理解产品之间的复杂关系，如相似性。然而，与它们在检索任务中的广泛应用和优化相关性的重点相反，当前的GNN架构并未针对最大化与收入相关的目标（如商品总交易价值（GMV））进行设计，而GMV是电子商务公司的主要业务指标之一。此外，在大规模电子商务系统中定义准确的边关系对于GNN来说是非常复杂的，因为商品之间的关系具有异质性。本研究旨在通过设计一种称为GNN-GMVO的新型GNN架构来解决这些问题。该模型直接优化GMV，同时保证推荐质量。

    Similar item recommendation is a critical task in the e-Commerce industry, which helps customers explore similar and relevant alternatives based on their interested products. Despite the traditional machine learning models, Graph Neural Networks (GNNs), by design, can understand complex relations like similarity between products. However, in contrast to their wide usage in retrieval tasks and their focus on optimizing the relevance, the current GNN architectures are not tailored toward maximizing revenue-related objectives such as Gross Merchandise Value (GMV), which is one of the major business metrics for e-Commerce companies. In addition, defining accurate edge relations in GNNs is non-trivial in large-scale e-Commerce systems, due to the heterogeneity nature of the item-item relationships. This work aims to address these issues by designing a new GNN architecture called GNN-GMVO (Graph Neural Network - Gross Merchandise Value Optimizer). This model directly optimizes GMV while cons
    
[^5]: 基于音频指纹的音乐推荐

    Music Recommendation Based on Audio Fingerprint. (arXiv:2310.17655v1 [eess.AS])

    [http://arxiv.org/abs/2310.17655](http://arxiv.org/abs/2310.17655)

    该研究提出了一种基于音频指纹的音乐推荐方法，结合了不同的音频特征，通过PCA降维并计算指纹间的相似矩阵，成功实现了89%的准确率。

    

    该研究结合了不同的音频特征，以获得更稳健的指纹来用于音乐推荐的过程中。这些方法的组合导致了一个高维向量。为了减少值的数量，对得到的指纹集合应用了主成分分析(PCA)，选择与解释方差为95%相对应的主成分的数量。最后，使用这些PCA指纹计算了每个指纹与整个数据集的相似矩阵。该过程被应用于个人音乐库中的200首歌曲，这些歌曲被标记了艺术家对应的流派。如果推荐的歌曲的流派与目标歌曲的流派匹配，则被评为成功的推荐(根据指纹的相似度)。通过这个过程，可以获得89%的准确率(成功的推荐占总共的推荐请求)。

    This work combined different audio features to obtain a more robust fingerprint to be used in a music recommendation process. The combination of these methods resulted in a high-dimensional vector. To reduce the number of values, PCA was applied to the set of resulting fingerprints, selecting the number of principal components that corresponded to an explained variance of $95\%$. Finally, with these PCA-fingerprints, the similarity matrix of each fingerprint with the entire data set was calculated. The process was applied to 200 songs from a personal music library; the songs were tagged with the artists' corresponding genres. The recommendations (fingerprints of songs with the closest similarity) were rated successful if the recommended songs' genre matched the target songs' genre. With this procedure, it was possible to obtain an accuracy of $89\%$ (successful recommendations out of total recommendation requests).
    
[^6]: 基于复杂网络的框架用于建模和挖掘患者路径

    Framework based on complex networks to model and mine patient pathways. (arXiv:2309.14208v2 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2309.14208](http://arxiv.org/abs/2309.14208)

    该论文提出了一个基于复杂网络的框架，用于建模和挖掘患者路径。该框架包括路径模型、新的相似度测量方法和基于传统中心度的挖掘方法。评估结果表明该框架可有效应用于实际医疗数据分析。

    

    自动发现用于表示一组患者与医疗系统的接触历史的模型，即所谓的“患者路径”，是一项新的研究领域，它支持临床和组织决策，以提高提供的治疗质量和效率。慢性病患者的路径往往因人而异，有重复的任务，并需要分析多个方面（干预、诊断、医疗专业等），影响结果。因此，建模和挖掘这些路径仍然是一个具有挑战性的任务。在这项工作中，我们提出了一个框架，包括：（i）基于多方面图的路径模型，（ii）一种新的相似度测量方法，考虑了耗时，用于比较路径，并且（iii）基于传统中心度测量方法的挖掘方法，用于发现路径中最相关的步骤。我们使用实际医疗数据评估了这个框架。

    The automatic discovery of a model to represent the history of encounters of a group of patients with the healthcare system -- the so-called "pathway of patients" -- is a new field of research that supports clinical and organisational decisions to improve the quality and efficiency of the treatment provided. The pathways of patients with chronic conditions tend to vary significantly from one person to another, have repetitive tasks, and demand the analysis of multiple perspectives (interventions, diagnoses, medical specialities, among others) influencing the results. Therefore, modelling and mining those pathways is still a challenging task. In this work, we propose a framework comprising: (i) a pathway model based on a multi-aspect graph, (ii) a novel dissimilarity measurement to compare pathways taking the elapsed time into account, and (iii) a mining method based on traditional centrality measures to discover the most relevant steps of the pathways. We evaluated the framework using 
    
[^7]: 跨模态检索：方法和未来方向的系统综述

    Cross-Modal Retrieval: A Systematic Review of Methods and Future Directions. (arXiv:2308.14263v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2308.14263](http://arxiv.org/abs/2308.14263)

    本文提供了一篇关于跨模态检索的方法和未来方向的系统综述，从浅层统计分析到视觉-语言预训练模型，深入探讨了现有跨模态检索方法的原理和架构。

    

    随着多样化多模态数据的爆炸性增长，传统的单模态检索方法难以满足用户对不同模态数据访问的需求。为解决这个问题，跨模态检索应运而生，它能够实现跨模态交互，促进语义匹配，并利用不同模态数据之间的互补性和一致性。尽管以往的文献对跨模态检索领域进行了综述，但存在着关于及时性、分类体系和全面性等方面的缺陷。本文对跨模态检索的发展进行了全面的综述，涵盖了从浅层统计分析技术到视觉-语言预训练模型的演进。文章首先从机器学习范式、机制和模型的角度构建了一个全面的分类体系，然后深入探讨了现有跨模态检索方法的原理和架构。此外，文章还概述了当前广泛使用的评估数据集、性能评价指标和常见问题。

    With the exponential surge in diverse multi-modal data, traditional uni-modal retrieval methods struggle to meet the needs of users demanding access to data from various modalities. To address this, cross-modal retrieval has emerged, enabling interaction across modalities, facilitating semantic matching, and leveraging complementarity and consistency between different modal data. Although prior literature undertook a review of the cross-modal retrieval field, it exhibits numerous deficiencies pertaining to timeliness, taxonomy, and comprehensiveness. This paper conducts a comprehensive review of cross-modal retrieval's evolution, spanning from shallow statistical analysis techniques to vision-language pre-training models. Commencing with a comprehensive taxonomy grounded in machine learning paradigms, mechanisms, and models, the paper then delves deeply into the principles and architectures underpinning existing cross-modal retrieval methods. Furthermore, it offers an overview of widel
    
[^8]: DebateKG: 用语义知识图自动创建政策辩论案例

    DebateKG: Automatic Policy Debate Case Creation with Semantic Knowledge Graphs. (arXiv:2307.04090v1 [cs.CL])

    [http://arxiv.org/abs/2307.04090](http://arxiv.org/abs/2307.04090)

    本论文提出了一种利用语义知识图自动创建政策辩论案例的方法，通过在争论的语义知识图上进行限制最短路径遍历，有效构建高质量的辩论案例。研究结果表明，在美国竞赛辩论中，利用这种方法显著改进了已有数据集DebateSum，并贡献了新的例子和有用的元数据。通过使用txtai语义搜索和知识图工具链，创建和贡献了9个语义知识图，同时提出了一种独特的评估方法来确定哪个知识图更适合政策辩论案例生成。

    

    近期相关工作表明，自然语言处理系统在解决竞赛辩论中的问题方面具有应用性。竞赛辩论中最重要的任务之一是辩手创建高质量的辩论案例。我们展示了使用限制最短路径遍历在争论的语义知识图上构建有效的辩论案例的方法。我们在一个名为DebateSum的大规模数据集上研究了这种潜力，该数据集针对的是一种名为政策辩论的美国竞赛辩论类型。我们通过向数据集中引入53180个新的例子，并为每个例子提供进一步有用的元数据，显著改进了DebateSum。我们利用txtai语义搜索和知识图工具链基于这个数据集产生并贡献了9个语义知识图。我们创建了一种独特的评估方法，以确定在政策辩论案例生成的背景下哪个知识图更好。

    Recent work within the Argument Mining community has shown the applicability of Natural Language Processing systems for solving problems found within competitive debate. One of the most important tasks within competitive debate is for debaters to create high quality debate cases. We show that effective debate cases can be constructed using constrained shortest path traversals on Argumentative Semantic Knowledge Graphs. We study this potential in the context of a type of American Competitive Debate, called Policy Debate, which already has a large scale dataset targeting it called DebateSum. We significantly improve upon DebateSum by introducing 53180 new examples, as well as further useful metadata for every example, to the dataset. We leverage the txtai semantic search and knowledge graph toolchain to produce and contribute 9 semantic knowledge graphs built on this dataset. We create a unique method for evaluating which knowledge graphs are better in the context of producing policy deb
    
[^9]: 可视丰富文件图像的表格检测

    Table Detection for Visually Rich Document Images. (arXiv:2305.19181v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.19181](http://arxiv.org/abs/2305.19181)

    本研究提出了一种在可视丰富文件图像中进行表格检测的方法。通过将IoU分解为真实覆盖项和预测覆盖项来衡量预测结果的信息丢失程度。此外，通过使用高斯噪声增强的图像大小区域提案和多对一标签分配，进一步改进模型。实验证明，该方法能够在不同数据集上优于最先进方法。

    

    表格检测是实现可视丰富文件理解的基本任务，需要模型在提取信息时避免信息丢失。然而，常用的交叉联合（IoU）评估指标和基于IoU的检测模型损失函数无法直接表示预测结果的信息丢失程度。因此，我们提出将IoU分解为真实覆盖项和预测覆盖项，其中前者可以用于衡量预测结果的信息丢失。此外，考虑到文档图像中表格的稀疏分布，我们使用SparseR-CNN作为基础模型，并通过使用高斯噪声增强的图像大小区域提案和多对一标签分配来进一步改进模型。全面实验结果表明，所提出的方法可以在不同IoU指标下的各种数据集上始终优于最先进的方法，并展示了其有效性。

    Table Detection (TD) is a fundamental task to enable visually rich document understanding, which requires the model to extract information without information loss. However, popular Intersection over Union (IoU) based evaluation metrics and IoU-based loss functions for the detection models cannot directly represent the degree of information loss for the prediction results. Therefore, we propose to decouple IoU into a ground truth coverage term and a prediction coverage term, in which the former can be used to measure the information loss of the prediction results. Besides, considering the sparse distribution of tables in document images, we use SparseR-CNN as the base model and further improve the model by using Gaussian Noise Augmented Image Size region proposals and many-to-one label assignments. Results under comprehensive experiments show that the proposed method can consistently outperform state-of-the-art methods with different IoU-based metrics under various datasets and demonst
    
[^10]: 使用基于案例推理的机器阅读理解

    Machine Reading Comprehension using Case-based Reasoning. (arXiv:2305.14815v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14815](http://arxiv.org/abs/2305.14815)

    本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。

    

    我们提出了一种准确且可解释的方法，用于机器阅读理解中的答案提取，该方法类似于经典人工智能中的基于案例推理（CBR）。我们的方法（CBR-MRC）基于一个假设，即相似问题的上下文化答案彼此之间具有语义相似性。给定一个测试问题，CBR-MRC首先从非参数化存储器中检索一组相似的案例，然后通过选择测试上下文中最类似于检索到的案例中上下文化答案表示的范围来预测答案。我们的方法半参数化的特性使其能够将预测归因于特定的证据案例集，因此在构建可靠且可调试的问答系统时是一个理想的选择。我们展示了CBR-MRC在自然语言问题（NaturalQuestions）和新闻问答（NewsQA）上比大型读者模型提供了高准确性，并且优于基准分别提升了11.5和8.4 EM。此外，我们还展示了CBR-MRC在识别与他人评估员不同的答案方面的能力。

    We present an accurate and interpretable method for answer extraction in machine reading comprehension that is reminiscent of case-based reasoning (CBR) from classical AI. Our method (CBR-MRC) builds upon the hypothesis that contextualized answers to similar questions share semantic similarities with each other. Given a test question, CBR-MRC first retrieves a set of similar cases from a non-parametric memory and then predicts an answer by selecting the span in the test context that is most similar to the contextualized representations of answers in the retrieved cases. The semi-parametric nature of our approach allows it to attribute a prediction to the specific set of evidence cases, making it a desirable choice for building reliable and debuggable QA systems. We show that CBR-MRC provides high accuracy comparable with large reader models and outperforms baselines by 11.5 and 8.4 EM on NaturalQuestions and NewsQA, respectively. Further, we demonstrate the ability of CBR-MRC in identi
    
[^11]: ChatGPT是一个好的推荐算法吗？初步研究

    Is ChatGPT a Good Recommender? A Preliminary Study. (arXiv:2304.10149v1 [cs.IR])

    [http://arxiv.org/abs/2304.10149](http://arxiv.org/abs/2304.10149)

    本论文研究了在推荐领域广泛使用的ChatGPT的潜力。实验结果表明即使没有微调，ChatGPT在五个推荐场景中表现出色，具有很好的推荐精度和解释性。

    

    推荐系统在过去几十年中取得了显著进展并得到广泛应用。然而，大多数传统推荐方法都是特定任务的，因此缺乏有效的泛化能力。最近，ChatGPT的出现通过增强对话模型的能力，显著推进了NLP任务。尽管如此，ChatGPT在推荐领域的应用还没有得到充分的研究。在本文中，我们采用ChatGPT作为通用推荐模型，探讨它将从大规模语料库中获得的广泛语言和世界知识转移到推荐场景中的潜力。具体而言，我们设计了一组提示，并评估ChatGPT在五个推荐场景中的表现。与传统的推荐方法不同的是，在整个评估过程中我们不微调ChatGPT，仅依靠提示自身将推荐任务转化为自然语言。

    Recommendation systems have witnessed significant advancements and have been widely used over the past decades. However, most traditional recommendation methods are task-specific and therefore lack efficient generalization ability. Recently, the emergence of ChatGPT has significantly advanced NLP tasks by enhancing the capabilities of conversational models. Nonetheless, the application of ChatGPT in the recommendation domain has not been thoroughly investigated. In this paper, we employ ChatGPT as a general-purpose recommendation model to explore its potential for transferring extensive linguistic and world knowledge acquired from large-scale corpora to recommendation scenarios. Specifically, we design a set of prompts and evaluate ChatGPT's performance on five recommendation scenarios. Unlike traditional recommendation methods, we do not fine-tune ChatGPT during the entire evaluation process, relying only on the prompts themselves to convert recommendation tasks into natural language 
    
[^12]: 大型语言模型在信息检索中的排名能力研究——以ChatGPT为例

    Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. (arXiv:2304.09542v1 [cs.CL])

    [http://arxiv.org/abs/2304.09542](http://arxiv.org/abs/2304.09542)

    本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。

    

    大型语言模型（LLMs）已经证明具有remarkable能力，能够将一些零样本语言任务推广至其他领域。本文研究了ChatGPT和GPT-4等生成性LLMs的相关性排名在信息检索方面的能力。实验结果显示，经过适当的指导，ChatGPT和GPT-4可以在流行的信息检索基准上取得竞争优势，甚至有时优于监督学习方法。特别地，GPT-4在TREC数据集上的平均nDCG上表现优于完全微调的monoT5-3B，BEIR数据集上的平均nDCG上优于monoT5-3B 2.3个点，低资源语言Mr.TyDi上的平均nDCG上优于monoT5-3B 2.7个点。随后，我们探讨了将ChatGPT的排名能力提炼为专门的模型的潜力。我们训练的小型专门模型（训练于10K个ChatGPT生成的数据）在BEIR上的表现优于在400K个MS MARCO注释数据上训练的monoT5。代码可在www.github.com/sunnwe上复现。

    Large Language Models (LLMs) have demonstrated a remarkable ability to generalize zero-shot to various language-related tasks. This paper focuses on the study of exploring generative LLMs such as ChatGPT and GPT-4 for relevance ranking in Information Retrieval (IR). Surprisingly, our experiments reveal that properly instructed ChatGPT and GPT-4 can deliver competitive, even superior results than supervised methods on popular IR benchmarks. Notably, GPT-4 outperforms the fully fine-tuned monoT5-3B on MS MARCO by an average of 2.7 nDCG on TREC datasets, an average of 2.3 nDCG on eight BEIR datasets, and an average of 2.7 nDCG on ten low-resource languages Mr.TyDi. Subsequently, we delve into the potential for distilling the ranking capabilities of ChatGPT into a specialized model. Our small specialized model that trained on 10K ChatGPT generated data outperforms monoT5 trained on 400K annotated MS MARCO data on BEIR. The code to reproduce our results is available at www.github.com/sunnwe
    

