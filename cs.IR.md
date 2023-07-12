# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation.](http://arxiv.org/abs/2307.05476) | 本文首次将Fisher合并方法应用于序列推荐中，通过合并多个模型的参数来改善整体性能，从而解决了实际挑战，具有推动最新技术的潜力。 |
| [^2] | [AdaptiveRec: Adaptively Construct Pairs for Contrastive Learning in Sequential Recommendation.](http://arxiv.org/abs/2307.05469) | 本文提出了AdaptiveRec，这是一种在顺序推荐中解决对比学习挑战的自适应方法，通过改善嵌入质量和减轻误判问题来提高效果，并在各种推荐场景中展示了其灵活性和适用性的价值。 |
| [^3] | [Duncode Characters Shorter.](http://arxiv.org/abs/2307.05414) | 本文研究了文本转换中使用的各种编码器，并介绍了一种创新的Duncode编码方法，该方法在编码整个Unicode字符集时具有较高的空间效率。 |
| [^4] | [Temporal Graphs Anomaly Emergence Detection: Benchmarking For Social Media Interactions.](http://arxiv.org/abs/2307.05268) | 本研究针对时间图中的异常出现进行了全面的基准测试，比较了12种数据驱动的方法。实验结果表明，针对这类任务没有明确的最佳方法，突显了检测大型动态系统中新兴异常的复杂性和挑战，强调了进一步研究和创新方法的需求。 |
| [^5] | [U-CREAT: Unsupervised Case Retrieval using Events extrAcTion.](http://arxiv.org/abs/2307.05260) | U-CREAT是一个无监督案例检索系统，通过使用事件提取实现了更高的性能和更快的检索速度，适用于实时案例检索系统。 |
| [^6] | [Generative Contrastive Graph Learning for Recommendation.](http://arxiv.org/abs/2307.05100) | 本文介绍了一种基于生成对比图学习的推荐系统。通过将用户的互动视为用户-项目图，结合对比学习技术和数据增强技术，提供更有效的自监督信号，从而改进了传统的基于协同过滤的推荐系统模型。 |
| [^7] | [Retrieval-augmented GPT-3.5-based Text-to-SQL Framework with Sample-aware Prompting and Dynamic Revision Chain.](http://arxiv.org/abs/2307.05074) | 本文提出了一种基于检索增强的GPT-3.5文本到SQL框架，采用了样本感知引导和动态修订链的方法，以应对现有方法在处理语义差距较大的检索示例时面临的挑战。 |
| [^8] | [Mining for Unknown Unknowns.](http://arxiv.org/abs/2307.05071) | 本论文使用形式概念分析（FCA）框架，旨在系统地挖掘和寻找未知未知，从而避免潜在的重大收益或损失。 |
| [^9] | [Neural-Symbolic Recommendation with Graph-Enhanced Information.](http://arxiv.org/abs/2307.05036) | 本研究结合了图神经网络和命题逻辑操作的优势，构建了一个具有全局隐式推理能力和局部显式逻辑推理能力的神经符号推荐模型。 |
| [^10] | [Empowering recommender systems using automatically generated Knowledge Graphs and Reinforcement Learning.](http://arxiv.org/abs/2307.04996) | 本文介绍了两种基于知识图谱的方法，一种使用强化学习，另一种使用XGBoost算法，用于个性化文章推荐。这些方法利用自动生成的知识图谱，并在一个大型跨国金融服务公司的客户中进行了实证研究。 |
| [^11] | [Ranking with Long-Term Constraints.](http://arxiv.org/abs/2307.04923) | 本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。 |
| [^12] | [Continual Learning on Dynamic Graphs via Parameter Isolation.](http://arxiv.org/abs/2305.13825) | 提出了Parameter Isolation GNN (PI-GNN)模型，用于处理动态图上的持续学习任务。该模型通过参数隔离和扩展来避免学习新模式和保留旧模式之间的权衡。 |
| [^13] | [One-Shot Labeling for Automatic Relevance Estimation.](http://arxiv.org/abs/2302.11266) | 本研究探索了利用大型语言模型来填补搜索系统离线评估过程中未评定文档的问题。研究结果表明，尽管预测结果与人工评定存在差异，但使用一次性标注器能够提供更可靠的系统排序效果。 |
| [^14] | [Collective Privacy Recovery: Data-sharing Coordination via Decentralized Artificial Intelligence.](http://arxiv.org/abs/2301.05995) | 本文研究了集体隐私恢复的问题，通过分散式人工智能实现数据共享的协同。研究发现，数据共享协调可以实现对隐私的显著恢复，并带来双赢效果。 |

# 详细

[^1]: 序列推荐中对比学习模型的Fisher加权合并

    Fisher-Weighted Merge of Contrastive Learning Models in Sequential Recommendation. (arXiv:2307.05476v1 [cs.IR])

    [http://arxiv.org/abs/2307.05476](http://arxiv.org/abs/2307.05476)

    本文首次将Fisher合并方法应用于序列推荐中，通过合并多个模型的参数来改善整体性能，从而解决了实际挑战，具有推动最新技术的潜力。

    

    随着在线平台和服务的指数增长，推荐系统已成为根据用户偏好识别相关物品的必备工具。序列推荐的领域旨在捕捉用户随时间变化的偏好。为了解决动态偏好，已提出了各种对比学习方法来应对推荐系统中由于有限的用户-物品交互而导致的数据稀疏性挑战。在本文中，我们首次将Fisher合并方法应用于序列推荐中，解决并解决了与之相关的实际挑战。这种方法通过合并多个模型的参数来确保鲁棒微调，从而改善整体性能。通过大量实验，我们证明了我们提出的方法的有效性，并突出了它们在序列学习和推荐系统中推动最新技术的潜力。

    Along with the exponential growth of online platforms and services, recommendation systems have become essential for identifying relevant items based on user preferences. The domain of sequential recommendation aims to capture evolving user preferences over time. To address dynamic preference, various contrastive learning methods have been proposed to target data sparsity, a challenge in recommendation systems due to the limited user-item interactions. In this paper, we are the first to apply the Fisher-Merging method to Sequential Recommendation, addressing and resolving practical challenges associated with it. This approach ensures robust fine-tuning by merging the parameters of multiple models, resulting in improved overall performance. Through extensive experiments, we demonstrate the effectiveness of our proposed methods, highlighting their potential to advance the state-of-the-art in sequential learning and recommendation systems.
    
[^2]: AdaptiveRec：在顺序推荐中自适应构建对比学习的解决方案

    AdaptiveRec: Adaptively Construct Pairs for Contrastive Learning in Sequential Recommendation. (arXiv:2307.05469v1 [cs.IR])

    [http://arxiv.org/abs/2307.05469](http://arxiv.org/abs/2307.05469)

    本文提出了AdaptiveRec，这是一种在顺序推荐中解决对比学习挑战的自适应方法，通过改善嵌入质量和减轻误判问题来提高效果，并在各种推荐场景中展示了其灵活性和适用性的价值。

    

    本文针对顺序推荐系统中对比学习所面临的挑战提出了一种解决方案。具体而言，它解决了误判的问题，该问题限制了推荐算法的有效性。通过引入先进的对比学习方法，所提出的方法改善了物品嵌入的质量，并减轻了将相似实例错误地归类为不相似的问题。实验证明，与现有系统相比，该方法提供了性能的提升。所提出的方法在各种推荐场景中的灵活性和适用性进一步凸显了它在增强顺序推荐系统中的价值。

    This paper presents a solution to the challenges faced by contrastive learning in sequential recommendation systems. In particular, it addresses the issue of false negative, which limits the effectiveness of recommendation algorithms. By introducing an advanced approach to contrastive learning, the proposed method improves the quality of item embeddings and mitigates the problem of falsely categorizing similar instances as dissimilar. Experimental results demonstrate performance enhancements compared to existing systems. The flexibility and applicability of the proposed approach across various recommendation scenarios further highlight its value in enhancing sequential recommendation systems.
    
[^3]: Duncode字符更短的技术

    Duncode Characters Shorter. (arXiv:2307.05414v1 [cs.CL])

    [http://arxiv.org/abs/2307.05414](http://arxiv.org/abs/2307.05414)

    本文研究了文本转换中使用的各种编码器，并介绍了一种创新的Duncode编码方法，该方法在编码整个Unicode字符集时具有较高的空间效率。

    

    本文研究了在文本转换中使用各种编码器，将字符转换为字节。讨论了本地编码器（如ASCII和GB-2312），它们将特定字符编码为较短的字节，以及通用编码器（如UTF-8和UTF-16），它们可以使用更多的空间来编码完整的Unicode字符集，并得到广泛接受。然而，其他编码器（包括SCSU，BOCU-1和二进制编码器）缺乏自同步功能。Duncode是一种创新的编码方法，旨在以高空间效率编码整个Unicode字符集，类似于本地编码器。它有潜力使用较少的字节将字符串的多个字符压缩为一个Duncode单元。尽管提供了较少的自同步识别信息，Duncode在空间效率方面超越了UTF8。应用程序可在\url{https://github.com/laohur/duncode}中找到。此外，我们还开发了一个基准测试工具。

    This paper investigates the employment of various encoders in text transformation, converting characters into bytes. It discusses local encoders such as ASCII and GB-2312, which encode specific characters into shorter bytes, and universal encoders like UTF-8 and UTF-16, which can encode the complete Unicode set with greater space requirements and are gaining widespread acceptance. Other encoders, including SCSU, BOCU-1, and binary encoders, however, lack self-synchronizing capabilities. Duncode is introduced as an innovative encoding method that aims to encode the entire Unicode character set with high space efficiency, akin to local encoders. It has the potential to compress multiple characters of a string into a Duncode unit using fewer bytes. Despite offering less self-synchronizing identification information, Duncode surpasses UTF8 in terms of space efficiency. The application is available at \url{https://github.com/laohur/duncode}. Additionally, we have developed a benchmark for e
    
[^4]: 时间图异常出现检测: 社交媒体交互的基准测试

    Temporal Graphs Anomaly Emergence Detection: Benchmarking For Social Media Interactions. (arXiv:2307.05268v1 [cs.SI])

    [http://arxiv.org/abs/2307.05268](http://arxiv.org/abs/2307.05268)

    本研究针对时间图中的异常出现进行了全面的基准测试，比较了12种数据驱动的方法。实验结果表明，针对这类任务没有明确的最佳方法，突显了检测大型动态系统中新兴异常的复杂性和挑战，强调了进一步研究和创新方法的需求。

    

    时间图已成为分析具有多个代理的复杂动态系统的重要工具。检测时间图中的异常对于各种应用至关重要，包括识别新兴趋势、监测网络安全、理解社交动态、追踪疾病爆发和了解金融动态等。本文提出了一个全面的基准测试研究，比较了12种数据驱动的时间图异常检测方法。我们在从Twitter和Facebook提取的两个时间图上进行实验，旨在识别群体交互中的异常。令人惊讶的是，我们的研究揭示了在这些任务中最好的方法存在不确定模式，突显了在大型动态系统中检测异常的复杂性和挑战。结果强调了进一步研究和创新方法以有效地检测表示为时间图的新兴异常的需求。

    Temporal graphs have become an essential tool for analyzing complex dynamic systems with multiple agents. Detecting anomalies in temporal graphs is crucial for various applications, including identifying emerging trends, monitoring network security, understanding social dynamics, tracking disease outbreaks, and understanding financial dynamics. In this paper, we present a comprehensive benchmarking study that compares 12 data-driven methods for anomaly detection in temporal graphs. We conduct experiments on two temporal graphs extracted from Twitter and Facebook, aiming to identify anomalies in group interactions. Surprisingly, our study reveals an unclear pattern regarding the best method for such tasks, highlighting the complexity and challenges involved in anomaly emergence detection in large and dynamic systems. The results underscore the need for further research and innovative approaches to effectively detect emerging anomalies in dynamic systems represented as temporal graphs.
    
[^5]: U-CREAT: 无监督事件提取的无监督案例检索系统

    U-CREAT: Unsupervised Case Retrieval using Events extrAcTion. (arXiv:2307.05260v1 [cs.IR])

    [http://arxiv.org/abs/2307.05260](http://arxiv.org/abs/2307.05260)

    U-CREAT是一个无监督案例检索系统，通过使用事件提取实现了更高的性能和更快的检索速度，适用于实时案例检索系统。

    

    在法律领域，先前案例检索的任务是自动引用与给定查询案例相关（基于事实和先例）的先前法律案例。为了进一步推动先前案例检索研究，本文提出了一个新的大型基准（以英文为主）用于先前案例检索任务：IL-PCR（印度法律先前案例检索）语料库。考虑到案例相关性的复杂性和法律文档的长度，BM25仍然是排名引用先前文档的强大基准。在这项工作中，我们探索了事件在法律案例检索中的作用，并提出一种基于无监督检索方法的管道系统U-CREAT（无监督事件提取的无监督案例检索系统）。我们发现，所提出的无监督检索方法与BM25相比显著提高了性能，并且使检索速度大大加快，使其适用于实时案例检索系统。我们的系统具有通用性，我们证明它适用于两个不同的法律体系（印度）。

    The task of Prior Case Retrieval (PCR) in the legal domain is about automatically citing relevant (based on facts and precedence) prior legal cases in a given query case. To further promote research in PCR, in this paper, we propose a new large benchmark (in English) for the PCR task: IL-PCR (Indian Legal Prior Case Retrieval) corpus. Given the complex nature of case relevance and the long size of legal documents, BM25 remains a strong baseline for ranking the cited prior documents. In this work, we explore the role of events in legal case retrieval and propose an unsupervised retrieval method-based pipeline U-CREAT (Unsupervised Case Retrieval using Events Extraction). We find that the proposed unsupervised retrieval method significantly increases performance compared to BM25 and makes retrieval faster by a considerable margin, making it applicable to real-time case retrieval systems. Our proposed system is generic, we show that it generalizes across two different legal systems (India
    
[^6]: 基于生成对比图学习的推荐系统

    Generative Contrastive Graph Learning for Recommendation. (arXiv:2307.05100v1 [cs.IR])

    [http://arxiv.org/abs/2307.05100](http://arxiv.org/abs/2307.05100)

    本文介绍了一种基于生成对比图学习的推荐系统。通过将用户的互动视为用户-项目图，结合对比学习技术和数据增强技术，提供更有效的自监督信号，从而改进了传统的基于协同过滤的推荐系统模型。

    

    通过将用户的互动视为用户-项目图，图学习模型已广泛应用于基于协同过滤的推荐系统。最近，研究人员将图对比学习技术引入协同过滤中，以缓解稀疏监督问题，首先通过数据增强构建对比视图，然后通过最大化对比视图之间的互信息提供自监督信号。尽管有效，我们认为当前基于图对比学习的推荐系统仍存在局限性，当前的数据增强技术，无论是结构增强还是特征增强。首先，结构增强随机丢弃节点或边，很容易破坏用户-项目图的内在特性。其次，特征增强对每个节点施加相同规模的噪声增强，忽视了图上节点的独特特征。为了解决上述限制，我们提出了一种新颖的变分生成对比图学习模型（Variational Gr）

    By treating users' interactions as a user-item graph, graph learning models have been widely deployed in Collaborative Filtering(CF) based recommendation. Recently, researchers have introduced Graph Contrastive Learning(GCL) techniques into CF to alleviate the sparse supervision issue, which first constructs contrastive views by data augmentations and then provides self-supervised signals by maximizing the mutual information between contrastive views. Despite the effectiveness, we argue that current GCL-based recommendation models are still limited as current data augmentation techniques, either structure augmentation or feature augmentation. First, structure augmentation randomly dropout nodes or edges, which is easy to destroy the intrinsic nature of the user-item graph. Second, feature augmentation imposes the same scale noise augmentation on each node, which neglects the unique characteristics of nodes on the graph. To tackle the above limitations, we propose a novel Variational Gr
    
[^7]: 采用样本感知引导和动态修订链的基于检索增强的GPT-3.5文本到SQL框架

    Retrieval-augmented GPT-3.5-based Text-to-SQL Framework with Sample-aware Prompting and Dynamic Revision Chain. (arXiv:2307.05074v1 [cs.IR])

    [http://arxiv.org/abs/2307.05074](http://arxiv.org/abs/2307.05074)

    本文提出了一种基于检索增强的GPT-3.5文本到SQL框架，采用了样本感知引导和动态修订链的方法，以应对现有方法在处理语义差距较大的检索示例时面临的挑战。

    

    文本到SQL旨在为给定的自然语言问题生成SQL查询，从而帮助用户查询数据库。最近出现了一种基于大型语言模型（LLMs）的提示学习方法，该方法设计提示以引导LLMs理解输入问题并生成相应的SQL。然而，它面临着严格的SQL语法要求的挑战。现有工作使用一系列示例（即问题-SQL对）来提示LLMs生成SQL，但固定的提示几乎无法处理检索出的示例与输入问题之间的语义差距较大的情况。在本文中，我们提出了一种基于检索增强的提示方法，用于基于LLM的文本到SQL框架，包括样本感知提示和动态修订链。我们的方法包括样本感知示例，其中包括SQL运算符的组合和与给定问题相关的细粒度信息。

    Text-to-SQL aims at generating SQL queries for the given natural language questions and thus helping users to query databases. Prompt learning with large language models (LLMs) has emerged as a recent approach, which designs prompts to lead LLMs to understand the input question and generate the corresponding SQL. However, it faces challenges with strict SQL syntax requirements. Existing work prompts the LLMs with a list of demonstration examples (i.e. question-SQL pairs) to generate SQL, but the fixed prompts can hardly handle the scenario where the semantic gap between the retrieved demonstration and the input question is large. In this paper, we propose a retrieval-augmented prompting method for a LLM-based Text-to-SQL framework, involving sample-aware prompting and a dynamic revision chain. Our approach incorporates sample-aware demonstrations, which include the composition of SQL operators and fine-grained information related to the given question. To retrieve questions sharing sim
    
[^8]: 对未知未知的挖掘

    Mining for Unknown Unknowns. (arXiv:2307.05071v1 [cs.AI])

    [http://arxiv.org/abs/2307.05071](http://arxiv.org/abs/2307.05071)

    本论文使用形式概念分析（FCA）框架，旨在系统地挖掘和寻找未知未知，从而避免潜在的重大收益或损失。

    

    未知未知是缺乏事前描述的未来相关的偶发事件。尽管有许多回顾性的报告显示，如果此类情况事前被发现，可以实现或避免显著收益或损失，但获取未知未知仍然是难以捉摸的，无论是在实践上还是在概念上。本文使用形式概念分析（FCA） - 一种越来越多地应用于挖掘和组织数据的格论子领域 - 引入了一个简单的框架，以系统地打破思维定势，指导对未知未知的搜索。

    Unknown unknowns are future relevant contingencies that lack an ex ante description. While there are numerous retrospective accounts showing that significant gains or losses might have been achieved or avoided had such contingencies been previously uncovered, getting hold of unknown unknowns still remains elusive, both in practice and conceptually. Using Formal Concept Analysis (FCA) - a subfield of lattice theory which is increasingly applied for mining and organizing data - this paper introduces a simple framework to systematically think out of the box and direct the search for unknown unknowns.
    
[^9]: 具有图增强信息的神经符号推荐系统

    Neural-Symbolic Recommendation with Graph-Enhanced Information. (arXiv:2307.05036v1 [cs.AI])

    [http://arxiv.org/abs/2307.05036](http://arxiv.org/abs/2307.05036)

    本研究结合了图神经网络和命题逻辑操作的优势，构建了一个具有全局隐式推理能力和局部显式逻辑推理能力的神经符号推荐模型。

    

    推荐系统不仅是一个从数据中归纳统计的问题，也是一个需要推理能力的认知任务。最先进的图神经网络在推荐系统中被广泛使用，因为它们能够从图结构数据中捕捉到隐式结构信息。然而，像大多数神经网络算法一样，它们只从感知的角度学习匹配模式。一些研究者使用用户行为进行逻辑推理，从认知推理的角度实现推荐预测，但这种推理是局部的，忽视了全局范围内的隐式信息。在这项工作中，我们结合了图神经网络和命题逻辑操作的优势，构建了一个具有全局隐式推理能力和局部显式逻辑推理能力的神经符号推荐模型。我们首先基于相邻交互原则构建了一个物品-物品图，并使用图神经网络对其进行学习和推理。然后，我们引入了命题逻辑操作，使模型能够从全局范围内进行推理。最后，我们通过实验证明了该模型的有效性和准确性。

    The recommendation system is not only a problem of inductive statistics from data but also a cognitive task that requires reasoning ability. The most advanced graph neural networks have been widely used in recommendation systems because they can capture implicit structured information from graph-structured data. However, like most neural network algorithms, they only learn matching patterns from a perception perspective. Some researchers use user behavior for logic reasoning to achieve recommendation prediction from the perspective of cognitive reasoning, but this kind of reasoning is a local one and ignores implicit information on a global scale. In this work, we combine the advantages of graph neural networks and propositional logic operations to construct a neuro-symbolic recommendation model with both global implicit reasoning ability and local explicit logic reasoning ability. We first build an item-item graph based on the principle of adjacent interaction and use graph neural net
    
[^10]: 利用自动生成的知识图谱和强化学习增强推荐系统

    Empowering recommender systems using automatically generated Knowledge Graphs and Reinforcement Learning. (arXiv:2307.04996v1 [cs.IR])

    [http://arxiv.org/abs/2307.04996](http://arxiv.org/abs/2307.04996)

    本文介绍了两种基于知识图谱的方法，一种使用强化学习，另一种使用XGBoost算法，用于个性化文章推荐。这些方法利用自动生成的知识图谱，并在一个大型跨国金融服务公司的客户中进行了实证研究。

    

    个性化推荐在直接营销中越来越重要，激发了通过知识图谱（KG）应用来提升客户体验的研究动机。例如，在金融服务领域，公司可以通过向客户提供相关金融文章来培养关系，促进客户参与和促进知情的金融决策。尽管一些方法专注于基于KG的推荐系统以改进内容，但在本研究中，我们专注于可解释的基于KG的推荐系统来进行决策。为此，我们提出了两种基于知识图谱的个性化文章推荐方法，用于一家大型跨国金融服务公司的一组客户。第一种方法使用强化学习，第二种方法使用XGBoost算法来向客户推荐文章。这两种方法都利用从结构化（表格数据）和非结构化数据（大量文本数据）生成的KG。

    Personalized recommendations have a growing importance in direct marketing, which motivates research to enhance customer experiences by knowledge graph (KG) applications. For example, in financial services, companies may benefit from providing relevant financial articles to their customers to cultivate relationships, foster client engagement and promote informed financial decisions. While several approaches center on KG-based recommender systems for improved content, in this study we focus on interpretable KG-based recommender systems for decision making.To this end, we present two knowledge graph-based approaches for personalized article recommendations for a set of customers of a large multinational financial services company. The first approach employs Reinforcement Learning and the second approach uses the XGBoost algorithm for recommending articles to the customers. Both approaches make use of a KG generated from both structured (tabular data) and unstructured data (a large body o
    
[^11]: 带有长期约束的排名

    Ranking with Long-Term Constraints. (arXiv:2307.04923v1 [cs.IR])

    [http://arxiv.org/abs/2307.04923](http://arxiv.org/abs/2307.04923)

    本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。

    

    用户通过他们的选择反馈（例如点击，购买）是为训练搜索和推荐算法提供的最常见类型的数据之一。然而，仅基于选择数据进行短视培训的系统可能仅改善短期参与度，而不能改善平台的长期可持续性以及对用户、内容提供者和其他利益相关者的长期利益。因此，本文开发了一个新的框架，其中决策者（例如平台运营商、监管机构、用户）可以表达平台行为的长期目标（例如公平性、收入分配、法律要求）。这些目标采取了超越个体会话的曝光或影响目标的形式，我们提供了新的基于控制的算法来实现这些目标。具体而言，控制器的设计旨在以最小化对短期参与的影响来实现所述的长期目标。除了原则性的理论推导外，

    The feedback that users provide through their choices (e.g., clicks, purchases) is one of the most common types of data readily available for training search and recommendation algorithms. However, myopically training systems based on choice data may only improve short-term engagement, but not the long-term sustainability of the platform and the long-term benefits to its users, content providers, and other stakeholders. In this paper, we thus develop a new framework in which decision makers (e.g., platform operators, regulators, users) can express long-term goals for the behavior of the platform (e.g., fairness, revenue distribution, legal requirements). These goals take the form of exposure or impact targets that go well beyond individual sessions, and we provide new control-based algorithms to achieve these goals. In particular, the controllers are designed to achieve the stated long-term goals with minimum impact on short-term engagement. Beyond the principled theoretical derivation
    
[^12]: 基于参数隔离的动态图上的持续学习

    Continual Learning on Dynamic Graphs via Parameter Isolation. (arXiv:2305.13825v1 [cs.LG])

    [http://arxiv.org/abs/2305.13825](http://arxiv.org/abs/2305.13825)

    提出了Parameter Isolation GNN (PI-GNN)模型，用于处理动态图上的持续学习任务。该模型通过参数隔离和扩展来避免学习新模式和保留旧模式之间的权衡。

    

    许多实际的图学习任务需要处理新节点和边出现的动态图。动态图学习方法通常遭遇灾难性遗忘问题，即为以前的图所学的知识会被新图的更新覆盖。为了缓解这个问题，提出了持续图学习方法。然而，现有的持续图学习方法旨在学习新的模式并维护旧的模式，但使用相同固定大小的参数集，因此面临两种目标之间的根本权衡。在本文中，我们提出了Parameter Isolation GNN (PI-GNN)，用于动态图上的持续学习，通过参数隔离和扩展来避免这种权衡。我们的动机在于不同的参数对于学习不同的图模式有贡献。基于这个想法，我们扩展模型参数以持续学习出现的图模式。与此同时，为了有效地保存未受影响模式的知识，我们找到参数。

    Many real-world graph learning tasks require handling dynamic graphs where new nodes and edges emerge. Dynamic graph learning methods commonly suffer from the catastrophic forgetting problem, where knowledge learned for previous graphs is overwritten by updates for new graphs. To alleviate the problem, continual graph learning methods are proposed. However, existing continual graph learning methods aim to learn new patterns and maintain old ones with the same set of parameters of fixed size, and thus face a fundamental tradeoff between both goals. In this paper, we propose Parameter Isolation GNN (PI-GNN) for continual learning on dynamic graphs that circumvents the tradeoff via parameter isolation and expansion. Our motivation lies in that different parameters contribute to learning different graph patterns. Based on the idea, we expand model parameters to continually learn emerging graph patterns. Meanwhile, to effectively preserve knowledge for unaffected patterns, we find parameter
    
[^13]: 自动相关性估计中的一次性标注

    One-Shot Labeling for Automatic Relevance Estimation. (arXiv:2302.11266v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.11266](http://arxiv.org/abs/2302.11266)

    本研究探索了利用大型语言模型来填补搜索系统离线评估过程中未评定文档的问题。研究结果表明，尽管预测结果与人工评定存在差异，但使用一次性标注器能够提供更可靠的系统排序效果。

    

    在离线实验中评估搜索系统时，处理未经评定的文档（“空洞”）是一个长期存在的问题。空洞可能会降低评估中检索系统的表现效果，并在使用不完整数据进行训练的模型中引入偏差。在本研究中，我们探讨了是否可以利用大型语言模型来填补这些空洞，以提高离线评估的效果。我们研究了一种极端但常见的评估设置，即每个查询只有一个已知相关文档可用于评估。然后，我们探讨了各种方法来预测未经评定文档与查询和已知相关文档的相关性，包括最近邻、监督和提示技术。我们发现，尽管这些一次性标注器（1SL）的预测经常与人工评定不一致，但它们产生的标签比单独的标签更可靠地对系统进行排序。具体地说，最强的一次性标注器可以显著提高系统的排序效果。

    Dealing with unjudged documents ("holes") in relevance assessments is a perennial problem when evaluating search systems with offline experiments. Holes can reduce the apparent effectiveness of retrieval systems during evaluation and introduce biases in models trained with incomplete data. In this work, we explore whether large language models can help us fill such holes to improve offline evaluations. We examine an extreme, albeit common, evaluation setting wherein only a single known relevant document per query is available for evaluation. We then explore various approaches for predicting the relevance of unjudged documents with respect to a query and the known relevant document, including nearest neighbor, supervised, and prompting techniques. We find that although the predictions of these One-Shot Labelers (1SL) frequently disagree with human assessments, the labels they produce yield a far more reliable ranking of systems than the single labels do alone. Specifically, the stronges
    
[^14]: 集体隐私恢复：通过分散式人工智能进行数据共享协同

    Collective Privacy Recovery: Data-sharing Coordination via Decentralized Artificial Intelligence. (arXiv:2301.05995v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.05995](http://arxiv.org/abs/2301.05995)

    本文研究了集体隐私恢复的问题，通过分散式人工智能实现数据共享的协同。研究发现，数据共享协调可以实现对隐私的显著恢复，并带来双赢效果。

    

    集体隐私损失变成了一个巨大的问题，对个人自由和民主构成了紧急威胁。但是，我们是否准备好将个人数据视为稀缺资源，并根据“尽可能少，尽可能多”的原则共享数据？我们假设，如果一个个体群体（数据集体）协调共享最少数据，以满足在线服务的所需质量，将会产生显著的隐私恢复。在这里，我们展示了如何使用去中心化人工智能自动化和扩展复杂的集体隐私恢复安排。为此，我们首次在一个严谨的高度逼真的实验中比较了态度、内在、奖励和协调数据共享，并利用因果推断和聚类分析方法区分了预测隐私和五个关键数据共享行为的标准。令人惊讶的是，数据共享协调对所有人来说都是双赢的：隐私得到显著恢复。

    Collective privacy loss becomes a colossal problem, an emergency for personal freedoms and democracy. But, are we prepared to handle personal data as scarce resource and collectively share data under the doctrine: as little as possible, as much as necessary? We hypothesize a significant privacy recovery if a population of individuals, the data collective, coordinates to share minimum data for running online services with the required quality. Here we show how to automate and scale-up complex collective arrangements for privacy recovery using decentralized artificial intelligence. For this, we compare for first time attitudinal, intrinsic, rewarded and coordinated data sharing in a rigorous living-lab experiment of high realism involving >27,000 real data disclosures. Using causal inference and cluster analysis, we differentiate criteria predicting privacy and five key data-sharing behaviors. Strikingly, data-sharing coordination proves to be a win-win for all: remarkable privacy recove
    

