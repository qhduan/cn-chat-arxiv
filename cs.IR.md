# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex.](http://arxiv.org/abs/2310.00402) | 提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。 |
| [^2] | [ReLoop2: Building Self-Adaptive Recommendation Models via Responsive Error Compensation Loop.](http://arxiv.org/abs/2306.08808) | 本文提出了ReLoop2，一种利用响应式误差补偿循环构建自适应推荐模型的方法，通过错误记忆模块来补偿模型预测误差，实现快速模型适应。 |
| [^3] | [Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models.](http://arxiv.org/abs/2306.08018) | Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。 |
| [^4] | [Editing Large Language Models: Problems, Methods, and Opportunities.](http://arxiv.org/abs/2305.13172) | 本文深入探讨了编辑大型语言模型的问题、方法和机会，提供了任务定义和挑战的概述、先进方法的实证分析，以及构建了新的基准数据集。这些结果有助于改进LLMs的编辑技术，提高其效果和可行性。 |
| [^5] | [Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study.](http://arxiv.org/abs/2305.11164) | 本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。 |
| [^6] | [Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application.](http://arxiv.org/abs/2305.03972) | 提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。 |
| [^7] | [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction.](http://arxiv.org/abs/2304.00902) | 本研究提出了一个用于CTR预测的增强双流MLP模型，经实证研究表明，该模型仅是简单地结合两个MLP就可以实现令人惊讶的良好性能。 |

# 详细

[^1]: DiskANN++: 使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索

    DiskANN++: Efficient Page-based Search over Isomorphic Mapped Graph Index using Query-sensitivity Entry Vertex. (arXiv:2310.00402v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.00402](http://arxiv.org/abs/2310.00402)

    提出了一个优化的DiskANN++方法，使用查询敏感的入口顶点在同构映射图索引上进行高效基于页面的搜索。

    

    给定一个向量数据集X和一个查询向量xq，基于图的近似最近邻搜索(ANNS)旨在构建一个图索引G，并通过在G上搜索来近似返回与xq的最小距离向量。基于图的ANNS的主要缺点是图索引太大，无法适应尤其是大规模X的内存。为了解决这个问题，提出了一种基于产品量化(PQ)的混合方法DiskANN，它在内存中存储低维度的PQ索引，并在SSD中保留图索引，从而减小内存开销同时确保高搜索准确性。然而，它存在两个重要的I/O问题，严重影响了整体效率：(1)从入口顶点到查询邻域的长路径导致大量的I/O请求，以及(2)在路由过程中的冗余I/O请求。为了解决上述问题，我们提出了优化的DiskANN++。

    Given a vector dataset $\mathcal{X}$ and a query vector $\vec{x}_q$, graph-based Approximate Nearest Neighbor Search (ANNS) aims to build a graph index $G$ and approximately return vectors with minimum distances to $\vec{x}_q$ by searching over $G$. The main drawback of graph-based ANNS is that a graph index would be too large to fit into the memory especially for a large-scale $\mathcal{X}$. To solve this, a Product Quantization (PQ)-based hybrid method called DiskANN is proposed to store a low-dimensional PQ index in memory and retain a graph index in SSD, thus reducing memory overhead while ensuring a high search accuracy. However, it suffers from two I/O issues that significantly affect the overall efficiency: (1) long routing path from an entry vertex to the query's neighborhood that results in large number of I/O requests and (2) redundant I/O requests during the routing process. We propose an optimized DiskANN++ to overcome above issues. Specifically, for the first issue, we pre
    
[^2]: ReLoop2: 通过响应式误差补偿循环构建自适应推荐模型

    ReLoop2: Building Self-Adaptive Recommendation Models via Responsive Error Compensation Loop. (arXiv:2306.08808v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.08808](http://arxiv.org/abs/2306.08808)

    本文提出了ReLoop2，一种利用响应式误差补偿循环构建自适应推荐模型的方法，通过错误记忆模块来补偿模型预测误差，实现快速模型适应。

    

    工业推荐系统面临非静态环境下运行的挑战，其中数据分布的转移源于用户行为的演变。为了解决这个挑战，一种常见的方法是使用新观察到的数据定期重新训练或增量更新已部署的深度模型，产生连续的训练过程。然而，神经网络的传统学习范式依赖于小学习率的迭代梯度更新，使得大推荐模型很难适应。本文引入了ReLoop2，一种自我纠正的学习环路，通过响应式误差补偿促进在线推荐系统中的快速模型适应。本文受人类大脑中观察到的慢-快互补学习系统的启发，提出一个错误记忆模块，直接存储来自数据流的错误样本。随后利用这些存储的样本来补偿模型预测误差。

    Industrial recommender systems face the challenge of operating in non-stationary environments, where data distribution shifts arise from evolving user behaviors over time. To tackle this challenge, a common approach is to periodically re-train or incrementally update deployed deep models with newly observed data, resulting in a continual training process. However, the conventional learning paradigm of neural networks relies on iterative gradient-based updates with a small learning rate, making it slow for large recommendation models to adapt. In this paper, we introduce ReLoop2, a self-correcting learning loop that facilitates fast model adaptation in online recommender systems through responsive error compensation. Inspired by the slow-fast complementary learning system observed in human brains, we propose an error memory module that directly stores error samples from incoming data streams. These stored samples are subsequently leveraged to compensate for model prediction errors durin
    
[^3]: Mol-Instructions: 一个大规模生物分子指令数据集，为大语言模型提供支持

    Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset for Large Language Models. (arXiv:2306.08018v1 [q-bio.QM])

    [http://arxiv.org/abs/2306.08018](http://arxiv.org/abs/2306.08018)

    Mol-Instructions是一个专门为生物分子领域设计的综合指令数据集，可以显著提高大语言模型在生物领域中的适应能力和认知敏锐度。

    

    大语言模型（LLM）以其卓越的任务处理能力和创新的输出，在许多领域推动了重大进展。然而，它们在生物分子研究等专业领域的熟练应用还受到限制。为了解决这个挑战，我们介绍了Mol-Instructions，这是一个经过精心策划、专门针对生物分子领域设计的综合指令数据集。Mol-Instructions由三个关键组成部分组成：分子导向指令、蛋白质导向指令和生物分子文本指令，每个部分都被策划用于增强LLM对生物分子特性和行为的理解和预测能力。通过对代表性LLM的广泛指令调整实验，我们强调了Mol-Instructions在增强大模型在生物分子研究复杂领域内的适应能力和认知敏锐度方面的潜力，从而促进生物分子领域的进一步发展。

    Large Language Models (LLMs), with their remarkable task-handling capabilities and innovative outputs, have catalyzed significant advancements across a spectrum of fields. However, their proficiency within specialized domains such as biomolecular studies remains limited. To address this challenge, we introduce Mol-Instructions, a meticulously curated, comprehensive instruction dataset expressly designed for the biomolecular realm. Mol-Instructions is composed of three pivotal components: molecule-oriented instructions, protein-oriented instructions, and biomolecular text instructions, each curated to enhance the understanding and prediction capabilities of LLMs concerning biomolecular features and behaviors. Through extensive instruction tuning experiments on the representative LLM, we underscore the potency of Mol-Instructions to enhance the adaptability and cognitive acuity of large models within the complex sphere of biomolecular studies, thereby promoting advancements in the biomol
    
[^4]: 编辑大型语言模型：问题、方法和机会

    Editing Large Language Models: Problems, Methods, and Opportunities. (arXiv:2305.13172v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13172](http://arxiv.org/abs/2305.13172)

    本文深入探讨了编辑大型语言模型的问题、方法和机会，提供了任务定义和挑战的概述、先进方法的实证分析，以及构建了新的基准数据集。这些结果有助于改进LLMs的编辑技术，提高其效果和可行性。

    

    尽管能够训练出表现优秀的大型语言模型（LLMs），但其保持相关性和纠正错误的方法仍然难以确定。为此，最近几年出现了许多编辑LLMs的技术，其目标是在特定领域内高效地改变LLMs的行为，同时不对其他输入的性能产生负面影响。本文深入探讨了与LLMs模型编辑相关的问题、方法和机会。特别是，我们提供了关于模型编辑任务定义和相关挑战的全面概述，以及对目前最先进的方法的深入实证分析。我们还构建了一个新的基准数据集，以促进更强大的评估，并指出现有技术固有的持久问题。我们的目标是为每种编辑技术的效果和可行性提供有价值的见解，从而帮助社区在LLMs的管理中取得更好的结果。

    Despite the ability to train capable LLMs, the methodology for maintaining their relevancy and rectifying errors remains elusive. To this end, the past few years have witnessed a surge in techniques for editing LLMs, the objective of which is to efficiently alter the behavior of LLMs within a specific domain without negatively impacting performance across other inputs. This paper embarks on a deep exploration of the problems, methods, and opportunities related to model editing for LLMs. In particular, we provide an exhaustive overview of the task definition and challenges associated with model editing, along with an in-depth empirical analysis of the most progressive methods currently at our disposal. We also build a new benchmark dataset to facilitate a more robust evaluation and pinpoint enduring issues intrinsic to existing techniques. Our objective is to provide valuable insights into the effectiveness and feasibility of each editing technique, thereby assisting the community in ma
    
[^5]: 探索抱抱脸ML模型的碳足迹：一项存储库挖掘研究

    Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study. (arXiv:2305.11164v1 [cs.LG])

    [http://arxiv.org/abs/2305.11164](http://arxiv.org/abs/2305.11164)

    本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。

    

    机器学习(ML)系统的崛起加剧了它们的碳足迹，这是由于其增加的能力和模型大小所致。然而，目前对ML模型的碳足迹如何实际测量、报告和评估的认识相对较少。因此，本论文旨在分析在Hugging Face上1,417个ML模型和相关数据集的碳足迹测量情况，Hugging Face是最受欢迎的预训练ML模型的存储库。目标是提供有关如何报告和优化ML模型的碳效率的见解和建议。该研究包括Hugging Face Hub API上有关碳排放的第一项存储库挖掘研究。本研究旨在回答两个研究问题：(1) ML模型的创建者如何在Hugging Face Hub上测量和报告碳排放？(2) 哪些方面影响了训练ML模型的碳排放？该研究得出了几个关键发现。其中包括碳排放报告模式比例的逐步下降等。

    The rise of machine learning (ML) systems has exacerbated their carbon footprint due to increased capabilities and model sizes. However, there is scarce knowledge on how the carbon footprint of ML models is actually measured, reported, and evaluated. In light of this, the paper aims to analyze the measurement of the carbon footprint of 1,417 ML models and associated datasets on Hugging Face, which is the most popular repository for pretrained ML models. The goal is to provide insights and recommendations on how to report and optimize the carbon efficiency of ML models. The study includes the first repository mining study on the Hugging Face Hub API on carbon emissions. This study seeks to answer two research questions: (1) how do ML model creators measure and report carbon emissions on Hugging Face Hub?, and (2) what aspects impact the carbon emissions of training ML models? The study yielded several key findings. These include a decreasing proportion of carbon emissions-reporting mode
    
[^6]: Mixer: 应用于工业应用的图像到跨模态检索学习

    Mixer: Image to Multi-Modal Retrieval Learning for Industrial Application. (arXiv:2305.03972v1 [cs.IR])

    [http://arxiv.org/abs/2305.03972](http://arxiv.org/abs/2305.03972)

    提出了一种新的可伸缩高效的图像到跨模态检索范式Mixer，解决了领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。

    

    跨模态检索一直是电子商务平台和内容分享社交媒体中普遍存在的需求，其中查询是一张图片，文档是具有图片和文本描述的项目。然而，目前这种检索任务仍面临诸多挑战，包括领域差距、跨模态数据对齐和融合、繁杂的数据训练标签以及海量查询和及时响应等问题。为此，我们提出了一种名为Mixer的新型可伸缩和高效的图像查询到跨模态检索学习范式。Mixer通过自适应地整合多模态数据、更高效地挖掘偏斜和嘈杂的数据，并可扩展到高负载量，解决了这些问题。

    Cross-modal retrieval, where the query is an image and the doc is an item with both image and text description, is ubiquitous in e-commerce platforms and content-sharing social media. However, little research attention has been paid to this important application. This type of retrieval task is challenging due to the facts: 1)~domain gap exists between query and doc. 2)~multi-modality alignment and fusion. 3)~skewed training data and noisy labels collected from user behaviors. 4)~huge number of queries and timely responses while the large-scale candidate docs exist. To this end, we propose a novel scalable and efficient image query to multi-modal retrieval learning paradigm called Mixer, which adaptively integrates multi-modality data, mines skewed and noisy data more efficiently and scalable to high traffic. The Mixer consists of three key ingredients: First, for query and doc image, a shared encoder network followed by separate transformation networks are utilized to account for their
    
[^7]: FinalMLP: 用于CTR预测的增强双流MLP模型

    FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction. (arXiv:2304.00902v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.00902](http://arxiv.org/abs/2304.00902)

    本研究提出了一个用于CTR预测的增强双流MLP模型，经实证研究表明，该模型仅是简单地结合两个MLP就可以实现令人惊讶的良好性能。

    

    点击率预测是在线广告和推荐中的基本任务之一。虽然多层感知器（MLP）在许多深度CTR预测模型中作为核心组件，但广为人知的是，仅应用一个基本MLP网络在学习乘法特征相互作用方面并不高效。因此，许多两个流交互模型（例如，DeepFM和DCN）通过将MLP网络与另一个专用网络集成以增强CTR预测。由于MLP流隐式地学习特征交互作用，因此现有研究主要关注于增强补充流中的显式特征交互作用。相反，我们的实证研究表明，一个经过良好调整的双流MLP模型，它只是简单地结合了两个MLP，甚至可以实现令人惊讶的良好性能，这在现有的工作中从未被报道过。基于这个观察结果，我们进一步提出了特征选择和交互聚合层。

    Click-through rate (CTR) prediction is one of the fundamental tasks for online advertising and recommendation. While multi-layer perceptron (MLP) serves as a core component in many deep CTR prediction models, it has been widely recognized that applying a vanilla MLP network alone is inefficient in learning multiplicative feature interactions. As such, many two-stream interaction models (e.g., DeepFM and DCN) have been proposed by integrating an MLP network with another dedicated network for enhanced CTR prediction. As the MLP stream learns feature interactions implicitly, existing research focuses mainly on enhancing explicit feature interactions in the complementary stream. In contrast, our empirical study shows that a well-tuned two-stream MLP model that simply combines two MLPs can even achieve surprisingly good performance, which has never been reported before by existing work. Based on this observation, we further propose feature selection and interaction aggregation layers that c
    

