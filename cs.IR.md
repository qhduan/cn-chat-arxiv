# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Popularity Bias of Multimodal-aware Recommender Systems: a Modalities-driven Analysis.](http://arxiv.org/abs/2308.12911) | 这项研究通过评估多模态感知推荐系统算法在Amazon数据集上的表现，发现了多模态推荐如何进一步放大流行偏差问题。 |
| [^2] | [Towards Communication-Efficient Model Updating for On-Device Session-Based Recommendation.](http://arxiv.org/abs/2308.12777) | 这项研究提出了一种基于组合编码的高效方法，用于在云端更新推荐系统模型并通过网络通信传输到设备上，以解决设备资源有限和网络带宽压力的问题。 |
| [^3] | [On the Consistency of Average Embeddings for Item Recommendation.](http://arxiv.org/abs/2308.12767) | 本文研究了推荐系统中平均嵌入的一致性，并提出了一种衡量方法。实证结果表明，现实世界的平均嵌入在推荐中一致性较低，为进一步改进现实世界嵌入提供了方向。 |
| [^4] | [Video Recommendation Using Social Network Analysis and User Viewing Patterns.](http://arxiv.org/abs/2308.12743) | 本论文旨在填补现有基于隐性反馈的视频推荐系统研究空白，通过社交网络分析和用户观看模式来构建有效的视频推荐模型。 |
| [^5] | [Out of the Box Thinking: Improving Customer Lifetime Value Modelling via Expert Routing and Game Whale Detection.](http://arxiv.org/abs/2308.12729) | 本文提出了一种新颖的多任务框架ExpLTV，利用游戏鲸鱼的检测来改进客户终身价值预测模型的准确性。 |
| [^6] | [Laying foundations to quantify the "Effort of Reproducibility".](http://arxiv.org/abs/2308.12580) | 为了解决科学论文的可重复性危机，该研究提出了一些解决方案，包括标记可重复性文章、会议上的可重复性检查清单和在OpenReview上共享成果。 |
| [^7] | [Exploring the Integration Strategies of Retriever and Large Language Models.](http://arxiv.org/abs/2308.12574) | 本文通过探索不同的检索器和大型语言模型整合方法来增强答案生成，并发现常用的连接方法存在局限性。为了解决这个问题，本文提出了四种替代策略，包括两种单轮方法和两种多轮策略。 |
| [^8] | [Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature.](http://arxiv.org/abs/2308.12420) | 本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。 |
| [^9] | [Natural Language is All a Graph Needs.](http://arxiv.org/abs/2308.07134) | 本论文提出了一种名为InstructGLM的结构化语言模型算法，该算法将大型语言模型与图表学习问题相结合，旨在探索是否可以用语言模型取代图神经网络作为图表的基础模型。 |
| [^10] | [Pareto Invariant Representation Learning for Multimedia Recommendation.](http://arxiv.org/abs/2308.04706) | 本文介绍了一种名为Pareto Invariant Representation Learning（PaInvRL）的框架，应用于多媒体推荐。该框架通过学习不变表示和变体表示的同时来缓解通用表示引入的错误相关性问题。从IID-OOD多目标优化的角度，PaInvRL减少了错误相关性对用户偏好的影响。 |
| [^11] | [Uncovering ChatGPT's Capabilities in Recommender Systems.](http://arxiv.org/abs/2305.02182) | 本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。 |
| [^12] | [An In-depth Investigation of User Response Simulation for Conversational Search.](http://arxiv.org/abs/2304.07944) | 本文研究了对话式搜索中用户响应模拟的方法。当前的模拟系统要么只能对是非问题进行回答，要么无法产生高质量的响应。通过用更小但先进的系统替换当前最先进的用户模拟系统，能够显著改进性能。 |
| [^13] | [Committed Private Information Retrieval.](http://arxiv.org/abs/2302.01733) | 该论文提出了一种承诺PIR方案，通过结合线性映射承诺和任意线性PIR方案，实现了$k$-可验证的PIR方案。 |

# 详细

[^1]: 关于多模态感知推荐系统中的流行偏差：一种模态驱动的分析

    On Popularity Bias of Multimodal-aware Recommender Systems: a Modalities-driven Analysis. (arXiv:2308.12911v1 [cs.IR])

    [http://arxiv.org/abs/2308.12911](http://arxiv.org/abs/2308.12911)

    这项研究通过评估多模态感知推荐系统算法在Amazon数据集上的表现，发现了多模态推荐如何进一步放大流行偏差问题。

    

    多模态感知推荐系统(MRSs)利用多模态内容（例如产品图片或描述）作为项目的附加信息，以提高推荐准确性。虽然大多数这样的方法都依赖于分解模型（例如MFBPR）作为基础架构，但已经证明MFBPR可能受到流行偏差的影响，即它本质上倾向于提升流行（即短头）目录中的项目推荐，而对长尾（即小众）目录中的项目推荐不利。在这项工作中，我们首次提供了关于多模态推荐如何进一步放大流行偏差的分析。具体而言，我们评估了四种最先进的MRSs算法（即VBPR、MMGCN、GRCN、LATTICE）在Amazon的三个数据集上的性能，评估推荐准确度指标的同时，还考虑了推荐项目的多样性和检索到的小众项目的比例等性能度量。为了更好地研究这一点……

    Multimodal-aware recommender systems (MRSs) exploit multimodal content (e.g., product images or descriptions) as items' side information to improve recommendation accuracy. While most of such methods rely on factorization models (e.g., MFBPR) as base architecture, it has been shown that MFBPR may be affected by popularity bias, meaning that it inherently tends to boost the recommendation of popular (i.e., short-head) items at the detriment of niche (i.e., long-tail) items from the catalog. Motivated by this assumption, in this work, we provide one of the first analyses on how multimodality in recommendation could further amplify popularity bias. Concretely, we evaluate the performance of four state-of-the-art MRSs algorithms (i.e., VBPR, MMGCN, GRCN, LATTICE) on three datasets from Amazon by assessing, along with recommendation accuracy metrics, performance measures accounting for the diversity of recommended items and the portion of retrieved niche items. To better investigate this as
    
[^2]: 面向通信高效的在线设备会话推荐模型更新

    Towards Communication-Efficient Model Updating for On-Device Session-Based Recommendation. (arXiv:2308.12777v1 [cs.IR])

    [http://arxiv.org/abs/2308.12777](http://arxiv.org/abs/2308.12777)

    这项研究提出了一种基于组合编码的高效方法，用于在云端更新推荐系统模型并通过网络通信传输到设备上，以解决设备资源有限和网络带宽压力的问题。

    

    最近，基于设备的推荐系统由于其提供即时响应和保护隐私的优势而受到越来越多的关注。为了与用户的兴趣变化保持同步，云端推荐系统会定期使用新的交互数据进行更新。然而，由于有限的设备计算资源，设备上的模型难以进行重新训练。作为解决方案，我们考虑了模型重新训练发生在服务器端的情景，然后通过网络通信将更新的参数传输到边缘设备。虽然这消除了本地重新训练的需求，但也导致了常规参数传输，给网络带宽带来了显著的压力。为了缓解这个问题，我们基于组合编码开发了一种高效的方法来压缩模型更新。通过这种方法，可以在尽量使用先前知识的同时，灵活地更新设备上的模型，减少额外的参数。我们进行了大量的实验来验证这种方法的有效性。

    On-device recommender systems recently have garnered increasing attention due to their advantages of providing prompt response and securing privacy. To stay current with evolving user interests, cloud-based recommender systems are periodically updated with new interaction data. However, on-device models struggle to retrain themselves because of limited onboard computing resources. As a solution, we consider the scenario where the model retraining occurs on the server side and then the updated parameters are transferred to edge devices via network communication. While this eliminates the need for local retraining, it incurs a regular transfer of parameters that significantly taxes network bandwidth. To mitigate this issue, we develop an efficient approach based on compositional codes to compress the model update. This approach ensures the on-device model is updated flexibly with minimal additional parameters whilst utilizing previous knowledge. The extensive experiments conducted on mul
    
[^3]: 关于平均嵌入用于物品推荐的一致性研究

    On the Consistency of Average Embeddings for Item Recommendation. (arXiv:2308.12767v1 [cs.IR])

    [http://arxiv.org/abs/2308.12767](http://arxiv.org/abs/2308.12767)

    本文研究了推荐系统中平均嵌入的一致性，并提出了一种衡量方法。实证结果表明，现实世界的平均嵌入在推荐中一致性较低，为进一步改进现实世界嵌入提供了方向。

    

    推荐系统中一种流行的做法是将物品嵌入进行平均以在同一嵌入空间中代表用户或更高级的概念。本文研究了这种做法的相关性。为此，我们提出了一种期望精度分数，用于衡量平均嵌入与其构建所使用的物品的一致性。我们随后在具有特定假设的理论环境和来自音乐流媒体服务的真实数据上分析了该分数的数学表达式及其经验表现。我们的结果强调了现实世界的平均值在推荐中的一致性较低，为未来研究更好地将现实世界的嵌入与我们理论环境的假设相一致铺平了道路。

    A prevalent practice in recommender systems consists of averaging item embeddings to represent users or higher-level concepts in the same embedding space. This paper investigates the relevance of such a practice. For this purpose, we propose an expected precision score, designed to measure the consistency of an average embedding relative to the items used for its construction. We subsequently analyze the mathematical expression of this score in a theoretical setting with specific assumptions, as well as its empirical behavior on real-world data from music streaming services. Our results emphasize that real-world averages are less consistent for recommendation, which paves the way for future research to better align real-world embeddings with assumptions from our theoretical setting.
    
[^4]: 使用社交网络分析和用户观看模式的视频推荐方法

    Video Recommendation Using Social Network Analysis and User Viewing Patterns. (arXiv:2308.12743v1 [cs.SI])

    [http://arxiv.org/abs/2308.12743](http://arxiv.org/abs/2308.12743)

    本论文旨在填补现有基于隐性反馈的视频推荐系统研究空白，通过社交网络分析和用户观看模式来构建有效的视频推荐模型。

    

    随着视频点播平台的迅猛崛起，用户面临着从大量内容中筛选出与自己喜好相符的节目的挑战。为了解决这个信息过载的困境，视频点播服务越来越多地加入了利用算法分析用户行为并建议个性化内容的推荐系统。然而，大多数现有的推荐系统依赖于用户明确的反馈，如评分和评论，但这种反馈的收集往往困难且耗时。因此，在建立有效的视频推荐模型方面，利用用户的隐性反馈模式可能提供了一条替代途径，避免了对明确评分的需求。然而，现有文献对于基于隐性反馈的推荐系统，特别是在建模视频观看行为方面，尚缺乏足够的探索。因此，本文旨在填补这一研究空白。

    With the meteoric rise of video-on-demand (VOD) platforms, users face the challenge of sifting through an expansive sea of content to uncover shows that closely match their preferences. To address this information overload dilemma, VOD services have increasingly incorporated recommender systems powered by algorithms that analyze user behavior and suggest personalized content. However, a majority of existing recommender systems depend on explicit user feedback in the form of ratings and reviews, which can be difficult and time-consuming to collect at scale. This presents a key research gap, as leveraging users' implicit feedback patterns could provide an alternative avenue for building effective video recommendation models, circumventing the need for explicit ratings. However, prior literature lacks sufficient exploration into implicit feedback-based recommender systems, especially in the context of modeling video viewing behavior. Therefore, this paper aims to bridge this research gap 
    
[^5]: 独特思维：通过专家路径选择和游戏鲸鱼检测改进客户终身价值建模

    Out of the Box Thinking: Improving Customer Lifetime Value Modelling via Expert Routing and Game Whale Detection. (arXiv:2308.12729v1 [cs.IR])

    [http://arxiv.org/abs/2308.12729](http://arxiv.org/abs/2308.12729)

    本文提出了一种新颖的多任务框架ExpLTV，利用游戏鲸鱼的检测来改进客户终身价值预测模型的准确性。

    

    客户终身价值（LTV）预测对于试图根据估计价值优化广告投资的移动游戏发行商至关重要。在移动游戏中，部署微交易是一种简单而有效的货币化策略，吸引了一小群在游戏内购买上大量消费的游戏鲸鱼。这种游戏鲸鱼的存在可能阻碍现有LTV预测模型的实用性，因为游戏鲸鱼的购买行为总是表现出与普通用户不同的分布。因此，识别游戏鲸鱼可以为改进LTV预测模型的准确性打开新机会。然而，目前在LTV预测中应用游戏鲸鱼检测的研究很少，现有工作主要针对假设可获得高质量用户特征的长期LTV预测，这在用户获取阶段不适用。在本文中，我们提出了一种新颖的多任务框架ExpLTV。

    Customer lifetime value (LTV) prediction is essential for mobile game publishers trying to optimize the advertising investment for each user acquisition based on the estimated worth. In mobile games, deploying microtransactions is a simple yet effective monetization strategy, which attracts a tiny group of game whales who splurge on in-game purchases. The presence of such game whales may impede the practicality of existing LTV prediction models, since game whales' purchase behaviours always exhibit varied distribution from general users. Consequently, identifying game whales can open up new opportunities to improve the accuracy of LTV prediction models. However, little attention has been paid to applying game whale detection in LTV prediction, and existing works are mainly specialized for the long-term LTV prediction with the assumption that the high-quality user features are available, which is not applicable in the UA stage. In this paper, we propose ExpLTV, a novel multi-task framew
    
[^6]: 建立量化"可重复性努力"的基础

    Laying foundations to quantify the "Effort of Reproducibility". (arXiv:2308.12580v1 [cs.DL])

    [http://arxiv.org/abs/2308.12580](http://arxiv.org/abs/2308.12580)

    为了解决科学论文的可重复性危机，该研究提出了一些解决方案，包括标记可重复性文章、会议上的可重复性检查清单和在OpenReview上共享成果。

    

    为什么有些研究容易重现，而其他研究则难以重现？对科学工作的准确性产生怀疑并不是有益的，特别是当个体研究者无法重现论文中的主张时。无法重现科研论文可能有许多主观原因。机器学习领域面临着可重复性危机，对已发表文章的调查导致人们意识到虽然共享代码存储库可取，但代码并不能决定一篇文章的可重复性。参与出版过程的各方都在积极解决可重复性危机，诸如对具有可重复性的文章进行徽章标记、会议上的可重复性检查清单（如NeurIPS、ICML、ICLR等）以及在OpenReview上共享成果等解决方案都显得很有前景。

    Why are some research studies easy to reproduce while others are difficult? Casting doubt on the accuracy of scientific work is not fruitful, especially when an individual researcher cannot reproduce the claims made in the paper. There could be many subjective reasons behind the inability to reproduce a scientific paper. The field of Machine Learning (ML) faces a reproducibility crisis, and surveying a portion of published articles has resulted in a group realization that although sharing code repositories would be appreciable, code bases are not the end all be all for determining the reproducibility of an article. Various parties involved in the publication process have come forward to address the reproducibility crisis and solutions such as badging articles as reproducible, reproducibility checklists at conferences (\textit{NeurIPS, ICML, ICLR, etc.}), and sharing artifacts on \textit{OpenReview} come across as promising solutions to the core problem. The breadth of literature on rep
    
[^7]: 探索检索器和大型语言模型的整合策略

    Exploring the Integration Strategies of Retriever and Large Language Models. (arXiv:2308.12574v1 [cs.IR])

    [http://arxiv.org/abs/2308.12574](http://arxiv.org/abs/2308.12574)

    本文通过探索不同的检索器和大型语言模型整合方法来增强答案生成，并发现常用的连接方法存在局限性。为了解决这个问题，本文提出了四种替代策略，包括两种单轮方法和两种多轮策略。

    

    检索到的段落和大型语言模型（如ChatGPT）的整合为提高开放领域问答作出了显著贡献。然而，如何将检索到的段落融入答案生成过程中的最佳方法仍然缺乏探索。本文旨在通过研究不同的方法来结合检索到的段落和大型语言模型以增强答案生成。我们首先研究了常用的连接方法的局限性。令人惊讶的是，即使正确的文档在前k个检索到的段落中，这种方法经常会生成“未知”输出。为了解决这个问题，我们探索了四种将检索到的段落与大型语言模型整合的替代策略。这些策略包括两种利用思维链推理的单轮方法和两种利用反馈循环的多轮策略。通过全面的分析和实验，我们发现...

    The integration of retrieved passages and large language models (LLMs), such as ChatGPTs, has significantly contributed to improving open-domain question answering. However, there is still a lack of exploration regarding the optimal approach for incorporating retrieved passages into the answer generation process. This paper aims to fill this gap by investigating different methods of combining retrieved passages with LLMs to enhance answer generation. We begin by examining the limitations of a commonly-used concatenation approach. Surprisingly, this approach often results in generating "unknown" outputs, even when the correct document is among the top-k retrieved passages. To address this issue, we explore four alternative strategies for integrating the retrieved passages with the LLMs. These strategies include two single-round methods that utilize chain-of-thought reasoning and two multi-round strategies that incorporate feedback loops. Through comprehensive analyses and experiments, w
    
[^8]: ESG主导的DLT研究的演化：对文献进行NLP分析

    Evolution of ESG-focused DLT Research: An NLP Analysis of the Literature. (arXiv:2308.12420v1 [cs.IR])

    [http://arxiv.org/abs/2308.12420](http://arxiv.org/abs/2308.12420)

    本研究通过NLP分析了ESG主导的DLT研究的演化，通过构建引用网络和命名实体识别任务，对DLT在ESG背景下的发展进行了文献综述。

    

    分布式账本技术(DLT)迅速发展，需要全面了解其各个组成部分。然而，针对DLT的环境、可持续性和治理(ESG)组成部分的系统文献综述还不足。为填补这一空白，我们选择了107篇种子文献，构建了一个包含63,083个参考文献的引用网络，并将其精炼为24,539篇文献的语料库进行分析。然后，我们根据一个已建立的技术分类法从46篇论文中标记了命名实体，并通过找出DLT的ESG要素来完善这个分类法。利用基于transformer的语言模型，我们对一个预先训练的语言模型进行了细化调整，用于命名实体识别任务，使用我们标记的数据集。我们利用我们调整后的语言模型对语料库进行了精简，得到了505篇关键论文，通过命名实体和时间图分析，促进了对DLT在ESG背景下的演化的文献综述。

    Distributed Ledger Technologies (DLTs) have rapidly evolved, necessitating comprehensive insights into their diverse components. However, a systematic literature review that emphasizes the Environmental, Sustainability, and Governance (ESG) components of DLT remains lacking. To bridge this gap, we selected 107 seed papers to build a citation network of 63,083 references and refined it to a corpus of 24,539 publications for analysis. Then, we labeled the named entities in 46 papers according to twelve top-level categories derived from an established technology taxonomy and enhanced the taxonomy by pinpointing DLT's ESG elements. Leveraging transformer-based language models, we fine-tuned a pre-trained language model for a Named Entity Recognition (NER) task using our labeled dataset. We used our fine-tuned language model to distill the corpus to 505 key papers, facilitating a literature review via named entities and temporal graph analysis on DLT evolution in the context of ESG. Our con
    
[^9]: 自然语言是图表所需要的全部内容

    Natural Language is All a Graph Needs. (arXiv:2308.07134v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.07134](http://arxiv.org/abs/2308.07134)

    本论文提出了一种名为InstructGLM的结构化语言模型算法，该算法将大型语言模型与图表学习问题相结合，旨在探索是否可以用语言模型取代图神经网络作为图表的基础模型。

    

    大规模预训练语言模型的出现，如ChatGPT，已经在人工智能的各个研究领域中引起了革命。基于Transformer的大型语言模型（LLMs）逐渐取代了CNN和RNN，将计算机视觉和自然语言处理领域统一起来。与相对独立存在的数据（如图像、视频或文本）相比，图表是一种包含丰富结构和关系信息的数据类型。同时，作为最具表现力的媒介之一，自然语言在描述复杂结构方面表现出色。然而，将图表学习问题纳入生成式语言建模框架的现有工作仍然非常有限。随着大型语言模型的重要性不断增长，探索LLMs是否也可以替代GNNs成为图表的基础模型变得至关重要。在本文中，我们提出了InstructGLM（结构化语言模型）算法，系统地设计高度可扩展的模型来处理图表学习问题。

    The emergence of large-scale pre-trained language models, such as ChatGPT, has revolutionized various research fields in artificial intelligence. Transformers-based large language models (LLMs) have gradually replaced CNNs and RNNs to unify fields of computer vision and natural language processing. Compared with the data that exists relatively independently such as images, videos or texts, graph is a type of data that contains rich structural and relational information. Meanwhile, natural language, as one of the most expressive mediums, excels in describing complex structures. However, existing work on incorporating graph learning problems into the generative language modeling framework remains very limited. As the importance of large language models continues to grow, it becomes essential to explore whether LLMs can also replace GNNs as the foundation model for graphs. In this paper, we propose InstructGLM (Instruction-finetuned Graph Language Model), systematically design highly scal
    
[^10]: Pareto不变表示学习在多媒体推荐中的应用

    Pareto Invariant Representation Learning for Multimedia Recommendation. (arXiv:2308.04706v1 [cs.IR])

    [http://arxiv.org/abs/2308.04706](http://arxiv.org/abs/2308.04706)

    本文介绍了一种名为Pareto Invariant Representation Learning（PaInvRL）的框架，应用于多媒体推荐。该框架通过学习不变表示和变体表示的同时来缓解通用表示引入的错误相关性问题。从IID-OOD多目标优化的角度，PaInvRL减少了错误相关性对用户偏好的影响。

    

    多媒体推荐涉及个性化排序任务，通常使用通用编码器表示多媒体内容。然而，这些通用表示引入了错误的相关性，无法揭示用户的真实偏好。现有的工作尝试通过学习不变表示来缓解这个问题，但忽视了独立同分布（IID）和非分布（OOD）广义化之间的平衡。本文提出了一个名为Pareto Invariant Representation Learning（PaInvRL）的框架，从IID-OOD多目标优化的角度减少了错误相关性的影响，同时学习不变表示（吸引用户注意的内在因素）和变体表示（其他因素）。具体而言，PaInvRL包括三个迭代执行的模块：（i）非同质识别模块，用于识别反映分布转移

    Multimedia recommendation involves personalized ranking tasks, where multimedia content is usually represented using a generic encoder. However, these generic representations introduce spurious correlations that fail to reveal users' true preferences. Existing works attempt to alleviate this problem by learning invariant representations, but overlook the balance between independent and identically distributed (IID) and out-of-distribution (OOD) generalization. In this paper, we propose a framework called Pareto Invariant Representation Learning (PaInvRL) to mitigate the impact of spurious correlations from an IID-OOD multi-objective optimization perspective, by learning invariant representations (intrinsic factors that attract user attention) and variant representations (other factors) simultaneously. Specifically, PaInvRL includes three iteratively executed modules: (i) heterogeneous identification module, which identifies the heterogeneous environments to reflect distributional shift
    
[^11]: 揭示ChatGPT在推荐系统中的能力

    Uncovering ChatGPT's Capabilities in Recommender Systems. (arXiv:2305.02182v1 [cs.IR])

    [http://arxiv.org/abs/2305.02182](http://arxiv.org/abs/2305.02182)

    本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。

    

    ChatGPT的问答功能吸引了自然语言处理（NLP）界及外界的关注。为了测试ChatGPT在推荐方面的表现，本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析。通过在不同领域的四个数据集上进行大量实验，我们发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型。基于单位成本改进的分析，我们确定ChatGPT在列表排名中能够在成本和性能之间实现最佳平衡，而在对和点排名中表现相对较弱。

    The debut of ChatGPT has recently attracted the attention of the natural language processing (NLP) community and beyond. Existing studies have demonstrated that ChatGPT shows significant improvement in a range of downstream NLP tasks, but the capabilities and limitations of ChatGPT in terms of recommendations remain unclear. In this study, we aim to conduct an empirical analysis of ChatGPT's recommendation ability from an Information Retrieval (IR) perspective, including point-wise, pair-wise, and list-wise ranking. To achieve this goal, we re-formulate the above three recommendation policies into a domain-specific prompt format. Through extensive experiments on four datasets from different domains, we demonstrate that ChatGPT outperforms other large language models across all three ranking policies. Based on the analysis of unit cost improvements, we identify that ChatGPT with list-wise ranking achieves the best trade-off between cost and performance compared to point-wise and pair-wi
    
[^12]: 论用户响应模拟在对话式搜索中的深入研究

    An In-depth Investigation of User Response Simulation for Conversational Search. (arXiv:2304.07944v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.07944](http://arxiv.org/abs/2304.07944)

    本文研究了对话式搜索中用户响应模拟的方法。当前的模拟系统要么只能对是非问题进行回答，要么无法产生高质量的响应。通过用更小但先进的系统替换当前最先进的用户模拟系统，能够显著改进性能。

    

    对话式搜索在信息检索和自然语言处理领域引起了广泛关注。它通过多次自然语言交互来澄清和解决用户的搜索需求。然而，大多数现有系统是通过记录或人工对话日志进行训练和演示的。最终，对话式搜索系统应该在未见过的对话轨迹的开放环境中进行训练、评估和部署。一个关键的挑战是训练和评估这样的系统都需要人工参与，这既昂贵又不可扩展。其中一种策略是模拟用户，以此来减少扩展成本。然而，当前的用户模拟器要么仅限于对对话搜索系统的是非问题进行回答，要么无法产生高质量的响应。本文表明，通过用一个更小但先进的系统来替换当前最先进的用户模拟系统，能够大幅改进其性能。

    Conversational search has seen increased recent attention in both the IR and NLP communities. It seeks to clarify and solve a user's search need through multi-turn natural language interactions. However, most existing systems are trained and demonstrated with recorded or artificial conversation logs. Eventually, conversational search systems should be trained, evaluated, and deployed in an open-ended setting with unseen conversation trajectories. A key challenge is that training and evaluating such systems both require a human-in-the-loop, which is expensive and does not scale. One strategy for this is to simulate users, thereby reducing the scaling costs. However, current user simulators are either limited to only respond to yes-no questions from the conversational search system, or unable to produce high quality responses in general.  In this paper, we show that current state-of-the-art user simulation system could be significantly improved by replacing it with a smaller but advanced
    
[^13]: 承诺私人信息检索

    Committed Private Information Retrieval. (arXiv:2302.01733v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2302.01733](http://arxiv.org/abs/2302.01733)

    该论文提出了一种承诺PIR方案，通过结合线性映射承诺和任意线性PIR方案，实现了$k$-可验证的PIR方案。

    

    私人信息检索（PIR）方案允许客户端在$k$个服务器的$n$个项目$x_1,x_2,\ldots,x_n$中检索出一个数据项目$x_i$，即使当$t<k$个服务器合谋并试图学习$i$时也不会透露$i$是什么。这样的PIR方案被称为$t-$私密。如果客户端可以在$v\leq k$个服务器合谋并试图通过发送篡改的数据来愚弄客户端的情况下验证检索到的$x_i$的正确性，则PIR方案为$v-$可验证。文献中的大多数先前研究假设$v<k$，留下了服务器全部合谋的情况。我们提出了一种通用构造，将线性映射承诺（LMC）和任意线性PIR方案结合起来，以产生一个$k-$可验证的PIR方案，称为承诺PIR方案。即使在最坏的情况下，当所有服务器都在攻击者的控制下，尽管隐私无法避免丢失，客户端也不会被愚弄而接受不正确的$x_i$。

    A private information retrieval (PIR) scheme allows a client to retrieve a data item $x_i$ among $n$ items $x_1,x_2,\ldots,x_n$ from $k$ servers, without revealing what $i$ is even when $t < k$ servers collude and try to learn $i$. Such a PIR scheme is said to be $t$-private. A PIR scheme is $v$-verifiable if the client can verify the correctness of the retrieved $x_i$ even when $v \leq k$ servers collude and try to fool the client by sending manipulated data. Most of the previous works in the literature on PIR assumed that $v < k$, leaving the case of all-colluding servers open. We propose a generic construction that combines a linear map commitment (LMC) and an arbitrary linear PIR scheme to produce a $k$-verifiable PIR scheme, termed a committed PIR scheme. Such a scheme guarantees that even in the worst scenario, when all servers are under the control of an attacker, although the privacy is unavoidably lost, the client won't be fooled into accepting an incorrect $x_i$. We demonstra
    

