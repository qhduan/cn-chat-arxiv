# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Zero-Shot Next-Item Recommendation using Large Pretrained Language Models.](http://arxiv.org/abs/2304.03153) | 本研究通过提出零样本下一个项目推荐策略，解决了使用大型预训练语言模型进行下一个项目推荐中遇到的挑战。 |
| [^2] | [Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives.](http://arxiv.org/abs/2304.03112) | 该论文提出了一个新闻推荐的统一框架，以便在候选人感知用户建模、点击行为融合和培训目标等关键设计维度上进行系统和公平的比较。该框架简化了模型设计和训练目标，并获得了有竞争力的结果。 |
| [^3] | [Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures.](http://arxiv.org/abs/2304.03054) | 本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。 |
| [^4] | [TagGPT: Large Language Models are Zero-shot Multimodal Taggers.](http://arxiv.org/abs/2304.03022) | TagGPT是一个零-shot多模态标注器，通过精心设计的提示，利用大型语言模型从一系列原始数据中预测大规模的候选标签，并进行过滤和语义分析，为特定应用程序构建高质量标签集。 |
| [^5] | [HGCC: Enhancing Hyperbolic Graph Convolution Networks on Heterogeneous Collaborative Graph for Recommendation.](http://arxiv.org/abs/2304.02961) | 该论文提出了一种名为HGCC的协同过滤模型，它通过添加幂律偏差来保持协同图的长尾性质，并直接聚合邻居节点，以提高推荐性能。 |
| [^6] | [Opportunities and challenges of ChatGPT for design knowledge management.](http://arxiv.org/abs/2304.02796) | 本文评述了设计知识分类和表示方法，以及过去支持设计师获得知识的努力。通过分析 ChatGPT 影响设计知识管理的机遇和挑战，提出了未来的研究方向，并进行了验证，结果表明设计师可以从各个领域获取有针对性的知识，但知识的质量高度依赖于提示。 |
| [^7] | [Unfolded Self-Reconstruction LSH: Towards Machine Unlearning in Approximate Nearest Neighbour Search.](http://arxiv.org/abs/2304.02350) | 本文提出了一种基于数据依赖的哈希方法USR-LSH，该方法通过展开实例级数据重建的优化更新，提高了数据的信息保留能力，同时还提出了一种动态的遗忘机制，使得数据可以快速删除和插入，无需重新训练，这是一种具有数据隐私和安全要求的在线ANN搜索的实际解决方案。 |
| [^8] | [FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction.](http://arxiv.org/abs/2304.00902) | 本研究提出了一个用于CTR预测的增强双流MLP模型，经实证研究表明，该模型仅是简单地结合两个MLP就可以实现令人惊讶的良好性能。 |
| [^9] | [Blurring-Sharpening Process Models for Collaborative Filtering.](http://arxiv.org/abs/2211.09324) | 本文提出了一种协同过滤的模糊-锐化过程模型（BSPM），并利用期望最大化算法学习模型参数，在显式和隐式反馈中优于现有技术方法。 |
| [^10] | [Personalized Showcases: Generating Multi-Modal Explanations for Recommendations.](http://arxiv.org/abs/2207.00422) | 该论文提出了一个新的任务——个性化展示，通过提供文本和视觉信息进一步丰富推荐的解释。作者从 Google Local（即地图）收集了一个大规模的数据集，并提出了一个个性化多模态框架。实验证明，该框架能够产生比先前方法更多样化和更具表现力的解释。 |

# 详细

[^1]: 利用大型预训练语言模型进行零样本下一个项目推荐

    Zero-Shot Next-Item Recommendation using Large Pretrained Language Models. (arXiv:2304.03153v1 [cs.IR])

    [http://arxiv.org/abs/2304.03153](http://arxiv.org/abs/2304.03153)

    本研究通过提出零样本下一个项目推荐策略，解决了使用大型预训练语言模型进行下一个项目推荐中遇到的挑战。

    

    大型语言模型（LLM）在各种自然语言处理（NLP）任务中取得了令人印象深刻的零样本表现，展示了它们在没有训练示例的情况下进行推理的能力。尽管取得了成功，但尚未有研究探索LLMs在零样本情况下执行下一个项目推荐的潜力。作者们确定了必须解决的两个主要问题，以使LLMs有效地充当推荐者。

    Large language models (LLMs) have achieved impressive zero-shot performance in various natural language processing (NLP) tasks, demonstrating their capabilities for inference without training examples. Despite their success, no research has yet explored the potential of LLMs to perform next-item recommendations in the zero-shot setting. We have identified two major challenges that must be addressed to enable LLMs to act effectively as recommenders. First, the recommendation space can be extremely large for LLMs, and LLMs do not know about the target user's past interacted items and preferences. To address this gap, we propose a prompting strategy called Zero-Shot Next-Item Recommendation (NIR) prompting that directs LLMs to make next-item recommendations. Specifically, the NIR-based strategy involves using an external module to generate candidate items based on user-filtering or item-filtering. Our strategy incorporates a 3-step prompting that guides GPT-3 to carry subtasks that captur
    
[^2]: 简化基于内容的神经新闻推荐：关于用户建模和训练目标

    Simplifying Content-Based Neural News Recommendation: On User Modeling and Training Objectives. (arXiv:2304.03112v1 [cs.IR])

    [http://arxiv.org/abs/2304.03112](http://arxiv.org/abs/2304.03112)

    该论文提出了一个新闻推荐的统一框架，以便在候选人感知用户建模、点击行为融合和培训目标等关键设计维度上进行系统和公平的比较。该框架简化了模型设计和训练目标，并获得了有竞争力的结果。

    

    个性化新闻推荐的出现使得推荐体系结构变得越来越复杂。大多数神经新闻推荐器依赖于用户点击行为，通常引入专门的用户编码器将点击新闻内容聚合成用户嵌入（早期融合）。这些模型主要通过标准的逐点分类目标进行训练。现有的工作存在两个主要缺点：（1）尽管设计普遍相同，但由于评估数据集和协议的不同，模型之间的直接比较受到了阻碍; （2）留给了替代的模型设计和训练目标大量的未开发空间。在这项工作中，我们提出了一个新闻推荐的统一框架，允许在几个关键设计维度上系统地和公平地比较新闻推荐器: （i）候选人感知用户建模，（ii）点击行为融合，和（iii）培训目标。我们的发现挑战了神经新闻推荐的现状，并突显了我们提出的框架的有效性，它使用非常简化的模型设计和训练目标实现了竞争性的结果。

    The advent of personalized news recommendation has given rise to increasingly complex recommender architectures. Most neural news recommenders rely on user click behavior and typically introduce dedicated user encoders that aggregate the content of clicked news into user embeddings (early fusion). These models are predominantly trained with standard point-wise classification objectives. The existing body of work exhibits two main shortcomings: (1) despite general design homogeneity, direct comparisons between models are hindered by varying evaluation datasets and protocols; (2) it leaves alternative model designs and training objectives vastly unexplored. In this work, we present a unified framework for news recommendation, allowing for a systematic and fair comparison of news recommenders across several crucial design dimensions: (i) candidate-awareness in user modeling, (ii) click behavior fusion, and (iii) training objectives. Our findings challenge the status quo in neural news rec
    
[^3]: 操纵联邦推荐系统: 用合成用户进行攻击及其对策

    Manipulating Federated Recommender Systems: Poisoning with Synthetic Users and Its Countermeasures. (arXiv:2304.03054v1 [cs.IR])

    [http://arxiv.org/abs/2304.03054](http://arxiv.org/abs/2304.03054)

    本文提出了一种新的攻击方法，利用合成的恶意用户上传有毒的梯度来在联邦推荐系统中有效地操纵目标物品的排名和曝光率。在两个真实世界的推荐数据集上进行了大量实验。

    

    联邦推荐系统（FedRecs）被认为是一种保护隐私的技术，可以在不共享用户数据的情况下协同学习推荐模型。因为所有参与者都可以通过上传梯度直接影响系统，所以FedRecs容易受到恶意客户的攻击，尤其是利用合成用户进行的攻击更加有效。本文提出了一种新的攻击方法，可以在不依赖任何先前知识的情况下，通过一组合成的恶意用户上传有毒的梯度来有效地操纵目标物品的排名和曝光率。我们在两个真实世界的推荐数据集上对两种广泛使用的FedRecs （Fed-NCF和Fed-LightGCN）进行了大量实验。

    Federated Recommender Systems (FedRecs) are considered privacy-preserving techniques to collaboratively learn a recommendation model without sharing user data. Since all participants can directly influence the systems by uploading gradients, FedRecs are vulnerable to poisoning attacks of malicious clients. However, most existing poisoning attacks on FedRecs are either based on some prior knowledge or with less effectiveness. To reveal the real vulnerability of FedRecs, in this paper, we present a new poisoning attack method to manipulate target items' ranks and exposure rates effectively in the top-$K$ recommendation without relying on any prior knowledge. Specifically, our attack manipulates target items' exposure rate by a group of synthetic malicious users who upload poisoned gradients considering target items' alternative products. We conduct extensive experiments with two widely used FedRecs (Fed-NCF and Fed-LightGCN) on two real-world recommendation datasets. The experimental res
    
[^4]: TagGPT：大型语言模型是零-shot多模态标注器

    TagGPT: Large Language Models are Zero-shot Multimodal Taggers. (arXiv:2304.03022v1 [cs.IR])

    [http://arxiv.org/abs/2304.03022](http://arxiv.org/abs/2304.03022)

    TagGPT是一个零-shot多模态标注器，通过精心设计的提示，利用大型语言模型从一系列原始数据中预测大规模的候选标签，并进行过滤和语义分析，为特定应用程序构建高质量标签集。

    

    标签在促进当代互联网时代各种应用中多媒体内容的有效分发方面起着关键作用，如搜索引擎和推荐系统。最近，大型语言模型在各种任务上展示了惊人的能力。在这项工作中，我们提出了TagGPT，这是一个完全自动化的系统，能够以完全零-shot的方式进行标签提取和多模态标注。我们的核心见解是，通过精心设计的提示，LLM能够在给定视觉、语音等多模态数据的文本线索的情况下提取和推理正确的标签。具体来说，为了自动构建反映特定应用程序中用户意图和兴趣的高质量标签集，TagGPT通过提示LLM从一系列原始数据中预测大规模的候选标签，并进行了过滤和语义分析。对于需要分发标记的新实体，TagGPT提出了两个零-shot选项。

    Tags are pivotal in facilitating the effective distribution of multimedia content in various applications in the contemporary Internet era, such as search engines and recommendation systems. Recently, large language models (LLMs) have demonstrated impressive capabilities across a wide range of tasks. In this work, we propose TagGPT, a fully automated system capable of tag extraction and multimodal tagging in a completely zero-shot fashion. Our core insight is that, through elaborate prompt engineering, LLMs are able to extract and reason about proper tags given textual clues of multimodal data, e.g., OCR, ASR, title, etc. Specifically, to automatically build a high-quality tag set that reflects user intent and interests for a specific application, TagGPT predicts large-scale candidate tags from a series of raw data via prompting LLMs, filtered with frequency and semantics. Given a new entity that needs tagging for distribution, TagGPT introduces two alternative options for zero-shot ta
    
[^5]: HGCC：提高异构协同图上的超几何图卷积网络用于推荐

    HGCC: Enhancing Hyperbolic Graph Convolution Networks on Heterogeneous Collaborative Graph for Recommendation. (arXiv:2304.02961v1 [cs.IR])

    [http://arxiv.org/abs/2304.02961](http://arxiv.org/abs/2304.02961)

    该论文提出了一种名为HGCC的协同过滤模型，它通过添加幂律偏差来保持协同图的长尾性质，并直接聚合邻居节点，以提高推荐性能。

    

    由于推荐任务中用户-物品交互数据的自然幂律分布特性，超几何空间建模已被引入协同过滤方法中。其中，超几何GCN结合了GCN和超几何空间的优势，并取得了令人惊讶的性能。然而，这些方法仅在设计中部分利用了超几何空间的特性，由于完全随机的嵌入初始化和不精确的切线空间聚合。此外，这些工作中使用的数据主要集中在仅用户-物品交互数据中，这进一步限制了模型的性能。本文提出了一种超几何GCN协同过滤模型HGCC，它改进了现有的超几何GCN结构，用于协同过滤并纳入了附加信息。它通过在节点嵌入初始化时添加幂律偏差来保持协同图的长尾性质；然后，它直接聚合邻居节点，以提高推荐性能。

    Due to the naturally power-law distributed nature of user-item interaction data in recommendation tasks, hyperbolic space modeling has recently been introduced into collaborative filtering methods. Among them, hyperbolic GCN combines the advantages of GCN and hyperbolic space and achieves a surprising performance. However, these methods only partially exploit the nature of hyperbolic space in their designs due to completely random embedding initialization and an inaccurate tangent space aggregation. In addition, the data used in these works mainly focus on user-item interaction data only, which further limits the performance of the models. In this paper, we propose a hyperbolic GCN collaborative filtering model, HGCC, which improves the existing hyperbolic GCN structure for collaborative filtering and incorporates side information. It keeps the long-tailed nature of the collaborative graph by adding power law prior to node embedding initialization; then, it aggregates neighbors directl
    
[^6]: ChatGPT 在设计知识管理中的机遇和挑战

    Opportunities and challenges of ChatGPT for design knowledge management. (arXiv:2304.02796v1 [cs.IR])

    [http://arxiv.org/abs/2304.02796](http://arxiv.org/abs/2304.02796)

    本文评述了设计知识分类和表示方法，以及过去支持设计师获得知识的努力。通过分析 ChatGPT 影响设计知识管理的机遇和挑战，提出了未来的研究方向，并进行了验证，结果表明设计师可以从各个领域获取有针对性的知识，但知识的质量高度依赖于提示。

    

    自然语言处理技术的进步使得像 ChatGPT 这样的大型语言模型能够便捷地为设计师提供广泛的相关信息，以促进设计过程中的知识管理。然而，将 ChatGPT 导入设计过程也带来了新的挑战。本文简要评述了设计知识的分类和表示方式，以及支持设计师获取知识的先前努力。我们分析了 ChatGPT 在设计知识管理中所带来的机遇和挑战，并提出了未来有前途的研究方向。通过实验证明 ChatGPT 可以让设计师获取来自各个领域的有针对性的知识，但所获取的知识质量极大地依赖于提示的质量。

    Recent advancements in Natural Language Processing have opened up new possibilities for the development of large language models like ChatGPT, which can facilitate knowledge management in the design process by providing designers with access to a vast array of relevant information. However, integrating ChatGPT into the design process also presents new challenges. In this paper, we provide a concise review of the classification and representation of design knowledge, and past efforts to support designers in acquiring knowledge. We analyze the opportunities and challenges that ChatGPT presents for knowledge management in design and propose promising future research directions. A case study is conducted to validate the advantages and drawbacks of ChatGPT, showing that designers can acquire targeted knowledge from various domains, but the quality of the acquired knowledge is highly dependent on the prompt.
    
[^7]: 未折叠自重建局部敏感哈希：走向近似最近邻搜索中的机器遗忘

    Unfolded Self-Reconstruction LSH: Towards Machine Unlearning in Approximate Nearest Neighbour Search. (arXiv:2304.02350v1 [cs.IR])

    [http://arxiv.org/abs/2304.02350](http://arxiv.org/abs/2304.02350)

    本文提出了一种基于数据依赖的哈希方法USR-LSH，该方法通过展开实例级数据重建的优化更新，提高了数据的信息保留能力，同时还提出了一种动态的遗忘机制，使得数据可以快速删除和插入，无需重新训练，这是一种具有数据隐私和安全要求的在线ANN搜索的实际解决方案。

    

    近似最近邻搜索是搜索引擎、推荐系统等的重要组成部分。许多最近的工作都是基于学习的数据分布依赖哈希，实现了良好的检索性能。但是，由于对用户隐私和安全的需求不断增加，我们经常需要从机器学习模型中删除用户数据信息以满足特定的隐私和安全要求。这种需求需要ANN搜索算法支持快速的在线数据删除和插入。当前的基于学习的哈希方法需要重新训练哈希函数，这是由于大规模数据的时间成本太高而难以承受的。为了解决这个问题，我们提出了一种新型的数据依赖哈希方法，名为unfolded self-reconstruction locality-sensitive hashing (USR-LSH)。我们的USR-LSH展开了实例级数据重建的优化更新，这比数据无关的LSH更能保留数据信息。此外，我们的USR-LSH提出了一种动态的遗忘机制，用于快速的数据删除和插入，无需重新训练。实验结果表明，USR-LSH在检索准确性和时间效率方面优于现有的哈希方法。USR-LSH是具有数据隐私和安全要求的在线ANN搜索的实际解决方案。

    Approximate nearest neighbour (ANN) search is an essential component of search engines, recommendation systems, etc. Many recent works focus on learning-based data-distribution-dependent hashing and achieve good retrieval performance. However, due to increasing demand for users' privacy and security, we often need to remove users' data information from Machine Learning (ML) models to satisfy specific privacy and security requirements. This need requires the ANN search algorithm to support fast online data deletion and insertion. Current learning-based hashing methods need retraining the hash function, which is prohibitable due to the vast time-cost of large-scale data. To address this problem, we propose a novel data-dependent hashing method named unfolded self-reconstruction locality-sensitive hashing (USR-LSH). Our USR-LSH unfolded the optimization update for instance-wise data reconstruction, which is better for preserving data information than data-independent LSH. Moreover, our US
    
[^8]: FinalMLP: 用于CTR预测的增强双流MLP模型

    FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction. (arXiv:2304.00902v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.00902](http://arxiv.org/abs/2304.00902)

    本研究提出了一个用于CTR预测的增强双流MLP模型，经实证研究表明，该模型仅是简单地结合两个MLP就可以实现令人惊讶的良好性能。

    

    点击率预测是在线广告和推荐中的基本任务之一。虽然多层感知器（MLP）在许多深度CTR预测模型中作为核心组件，但广为人知的是，仅应用一个基本MLP网络在学习乘法特征相互作用方面并不高效。因此，许多两个流交互模型（例如，DeepFM和DCN）通过将MLP网络与另一个专用网络集成以增强CTR预测。由于MLP流隐式地学习特征交互作用，因此现有研究主要关注于增强补充流中的显式特征交互作用。相反，我们的实证研究表明，一个经过良好调整的双流MLP模型，它只是简单地结合了两个MLP，甚至可以实现令人惊讶的良好性能，这在现有的工作中从未被报道过。基于这个观察结果，我们进一步提出了特征选择和交互聚合层。

    Click-through rate (CTR) prediction is one of the fundamental tasks for online advertising and recommendation. While multi-layer perceptron (MLP) serves as a core component in many deep CTR prediction models, it has been widely recognized that applying a vanilla MLP network alone is inefficient in learning multiplicative feature interactions. As such, many two-stream interaction models (e.g., DeepFM and DCN) have been proposed by integrating an MLP network with another dedicated network for enhanced CTR prediction. As the MLP stream learns feature interactions implicitly, existing research focuses mainly on enhancing explicit feature interactions in the complementary stream. In contrast, our empirical study shows that a well-tuned two-stream MLP model that simply combines two MLPs can even achieve surprisingly good performance, which has never been reported before by existing work. Based on this observation, we further propose feature selection and interaction aggregation layers that c
    
[^9]: 协同过滤的模糊-锐化过程模型

    Blurring-Sharpening Process Models for Collaborative Filtering. (arXiv:2211.09324v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.09324](http://arxiv.org/abs/2211.09324)

    本文提出了一种协同过滤的模糊-锐化过程模型（BSPM），并利用期望最大化算法学习模型参数，在显式和隐式反馈中优于现有技术方法。

    

    协同过滤是推荐系统中最基本的主题之一。从矩阵分解到图卷积方法，已经提出了各种各样的协同过滤方法。在图过滤方法和基于分数的生成模型（SGM）的最近成功启发下，我们提出了一种新的模糊-锐化过程模型（BSPM）的概念。SGM和BSPM共享相同的处理哲学，即在将原始信息首先扰乱然后恢复到原始形式的过程中可以发现新信息（例如，在SGM的情况下生成新图像）。然而，SGM和我们的BSPM处理不同类型的信息，并且它们的最优扰动和恢复过程存在根本上的差异。因此，我们的BSPM与SGM具有不同的形式。此外，我们的概念不仅理论上包括了许多现有的协同过滤模型，而且在显式和隐式反馈的情况下，在召回率和NDCG方面也优于它们。具体而言，我们提出了一组具有不同模糊和锐化滤波器设置的BSPM，并推导了期望最大化（EM）算法以学习模型参数。四个基准数据集的实验结果证明了我们的模型相对于现有技术方法的有效性。

    Collaborative filtering is one of the most fundamental topics for recommender systems. Various methods have been proposed for collaborative filtering, ranging from matrix factorization to graph convolutional methods. Being inspired by recent successes of graph filtering-based methods and score-based generative models (SGMs), we present a novel concept of blurring-sharpening process model (BSPM). SGMs and BSPMs share the same processing philosophy that new information can be discovered (e.g., new images are generated in the case of SGMs) while original information is first perturbed and then recovered to its original form. However, SGMs and our BSPMs deal with different types of information, and their optimal perturbation and recovery processes have fundamental discrepancies. Therefore, our BSPMs have different forms from SGMs. In addition, our concept not only theoretically subsumes many existing collaborative filtering models but also outperforms them in terms of Recall and NDCG in th
    
[^10]: 个性化展示：生成面向推荐的多模态解释

    Personalized Showcases: Generating Multi-Modal Explanations for Recommendations. (arXiv:2207.00422v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2207.00422](http://arxiv.org/abs/2207.00422)

    该论文提出了一个新的任务——个性化展示，通过提供文本和视觉信息进一步丰富推荐的解释。作者从 Google Local（即地图）收集了一个大规模的数据集，并提出了一个个性化多模态框架。实验证明，该框架能够产生比先前方法更多样化和更具表现力的解释。

    

    现有的解释模型只为推荐生成文本，但仍然难以产生多样化的内容。在本文中，我们提出了一个名为“个性化展示”的新任务，通过在解释中提供文本和视觉信息来进一步丰富解释。具体而言，我们首先选择一个定制的图像集，该集合与用户对推荐物品的兴趣最相关。然后，根据我们所选的图像生成自然语言解释。 为了实现这个新任务，我们从 Google Local（即地图）收集了一个大规模的数据集，并构建了一个高质量的子集以生成多模态解释。我们提出了一个个性化多模态框架，可以通过对比学习生成多样化和视觉一致的解释。实验表明，我们的框架受益于不同的输入模态，并且能够产生比先前方法更多样化和更具表现力的解释。

    Existing explanation models generate only text for recommendations but still struggle to produce diverse contents. In this paper, to further enrich explanations, we propose a new task named personalized showcases, in which we provide both textual and visual information to explain our recommendations. Specifically, we first select a personalized image set that is the most relevant to a user's interest toward a recommended item. Then, natural language explanations are generated accordingly given our selected images. For this new task, we collect a large-scale dataset from Google Local (i.e.,~maps) and construct a high-quality subset for generating multi-modal explanations. We propose a personalized multi-modal framework which can generate diverse and visually-aligned explanations via contrastive learning. Experiments show that our framework benefits from different modalities as inputs, and is able to produce more diverse and expressive explanations compared to previous methods on a varie
    

