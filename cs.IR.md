# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [R\'esum\'e Parsing as Hierarchical Sequence Labeling: An Empirical Study.](http://arxiv.org/abs/2309.07015) | 本研究将简历解析问题作为分层序列标注任务，提出了同时解决行和标记两个任务的模型架构，并构建了多语言的高质量简历解析语料库。实验结果表明，所提出模型在信息提取任务中优于先前工作中的方法。进一步分析了模型性能和资源效率，并描述了模型在生产环境中的权衡。 |
| [^2] | [Modeling Dislocation Dynamics Data Using Semantic Web Technologies.](http://arxiv.org/abs/2309.06930) | 本文介绍了如何使用语义网技术对位错动力学模拟数据进行建模，并通过添加缺失的概念和与相关本体对齐来扩展位错本体。 |
| [^3] | [Multi-behavior Recommendation with SVD Graph Neural Networks.](http://arxiv.org/abs/2309.06912) | 本研究提出了一种使用SVD图神经网络进行多行为推荐的模型MB-SVD，通过考虑用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。 |
| [^4] | [Towards the TopMost: A Topic Modeling System Toolkit.](http://arxiv.org/abs/2309.06908) | 本文提出了一个名为TopMost的主题建模系统工具包，通过涵盖更广泛的主题建模场景和具有高度凝聚力和解耦模块化设计的特点，可以促进主题模型的研究和应用。 |
| [^5] | [ProMap: Datasets for Product Mapping in E-commerce.](http://arxiv.org/abs/2309.06882) | 该论文介绍了两个新的产品映射数据集：ProMapCz和ProMapEn，分别包含捷克和英文产品对，这些数据集具有较为完整的产品信息，并解决了目前现有数据集无法区分非常相似但不匹配产品对的问题。 |
| [^6] | [An Image Dataset for Benchmarking Recommender Systems with Raw Pixels.](http://arxiv.org/abs/2309.06789) | 这个研究提出了PixelRec，一个大规模图像推荐数据集，包括2亿个用户-图像交互、30万个用户和40万个高质量封面图像。通过提供原始图像像素的直接访问，PixelRec使得推荐模型能够直接从图像学习项目表示。 |
| [^7] | [CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation.](http://arxiv.org/abs/2309.06748) | CONVERSER是一个使用少量对话样本进行训练的对话式密集检索框架，通过利用大型语言模型的上下文学习能力，能够生成与检索语料库中段落相关的对话查询，实验结果表明其在少样本对话密集检索中表现出与完全监督模型相当的性能。 |
| [^8] | [Hierarchical Multi-Task Learning Framework for Session-based Recommendations.](http://arxiv.org/abs/2309.06533) | 本文提出了一种面向会话推荐的层次化多任务学习框架HierSRec，通过在预测任务之间设置层次结构，并利用辅助任务的输出来提供更丰富的输入特征和更高的预测可解释性，进一步增强了预测准确性和可泛化性。 |
| [^9] | [AKEM: Aligning Knowledge Base to Queries with Ensemble Model for Entity Recognition and Linking.](http://arxiv.org/abs/2309.06175) | 本文提出了一种利用集成模型将知识库与查询对齐的方法，用于实体识别和链接挑战。通过扩展知识库和利用外部知识，提高了召回率，并使用支持向量回归和多元加性回归树过滤结果得到高精度的实体识别和链接。最终实现了高效的计算和0.535的F1分数。 |
| [^10] | [Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems.](http://arxiv.org/abs/2302.03735) | 本文系统地研究了如何从不同预训练语言模型中提取和转移知识，提高推荐系统性能。我们提出了一个分类法，分析和总结了基于预训练语言模型的推荐系统的培训策略和目标。 |
| [^11] | [Cost-optimal Seeding Strategy During a Botanical Pandemic in Domesticated Fields.](http://arxiv.org/abs/2301.02817) | 这项研究提出了一种基于网格的经济最优播种策略，通过数学模型描述了在植物流行病期间农田作物的经济利润，可为农田主人和决策者提供指导。 |

# 详细

[^1]: 简历解析作为分层序列标注的实证研究

    R\'esum\'e Parsing as Hierarchical Sequence Labeling: An Empirical Study. (arXiv:2309.07015v1 [cs.CL])

    [http://arxiv.org/abs/2309.07015](http://arxiv.org/abs/2309.07015)

    本研究将简历解析问题作为分层序列标注任务，提出了同时解决行和标记两个任务的模型架构，并构建了多语言的高质量简历解析语料库。实验结果表明，所提出模型在信息提取任务中优于先前工作中的方法。进一步分析了模型性能和资源效率，并描述了模型在生产环境中的权衡。

    

    从简历中提取信息通常被形式化为一个两阶段的问题，即首先将文档分段，然后对每个段落进行单独处理以提取目标实体。我们将整个问题分为两个级别的序列标注任务，即行和标记，并研究了同时解决这两个任务的模型架构。我们构建了英语、法语、中文、西班牙语、德语、葡萄牙语和瑞典语的高质量简历解析语料库。基于这些语料库，我们提出了实验结果，证明了所提出模型在信息提取任务中的有效性，优于先前工作中的方法。我们对所提出的架构进行了消融研究。我们还分析了模型的性能和资源效率，并描述了在生产环境中进行模型部署时的权衡。

    Extracting information from r\'esum\'es is typically formulated as a two-stage problem, where the document is first segmented into sections and then each section is processed individually to extract the target entities. Instead, we cast the whole problem as sequence labeling in two levels -- lines and tokens -- and study model architectures for solving both tasks simultaneously. We build high-quality r\'esum\'e parsing corpora in English, French, Chinese, Spanish, German, Portuguese, and Swedish. Based on these corpora, we present experimental results that demonstrate the effectiveness of the proposed models for the information extraction task, outperforming approaches introduced in previous work. We conduct an ablation study of the proposed architectures. We also analyze both model performance and resource efficiency, and describe the trade-offs for model deployment in the context of a production environment.
    
[^2]: 使用语义网技术对位错动力学数据进行建模

    Modeling Dislocation Dynamics Data Using Semantic Web Technologies. (arXiv:2309.06930v1 [cond-mat.mtrl-sci])

    [http://arxiv.org/abs/2309.06930](http://arxiv.org/abs/2309.06930)

    本文介绍了如何使用语义网技术对位错动力学模拟数据进行建模，并通过添加缺失的概念和与相关本体对齐来扩展位错本体。

    

    材料科学与工程领域的研究着眼于材料的设计、合成、性能和性能。被广泛研究的一个重要材料类别是晶体材料，包括金属和半导体。晶体材料通常包含一种称为“位错”的特殊缺陷。这种缺陷显著影响各种材料性能，包括强度、断裂韧性和延展性。近年来，研究人员通过实验表征技术和模拟（如位错动力学模拟）致力于理解位错行为。本文介绍了如何通过使用语义网技术以本体方式对位错动力学模拟数据进行建模。我们通过添加缺失的概念并将其与其他两个与该领域相关的本体（即Elementary Multi-perspectiv）进行对齐来扩展已有的位错本体。

    Research in the field of Materials Science and Engineering focuses on the design, synthesis, properties, and performance of materials. An important class of materials that is widely investigated are crystalline materials, including metals and semiconductors. Crystalline material typically contains a distinct type of defect called "dislocation". This defect significantly affects various material properties, including strength, fracture toughness, and ductility. Researchers have devoted a significant effort in recent years to understanding dislocation behavior through experimental characterization techniques and simulations, e.g., dislocation dynamics simulations. This paper presents how data from dislocation dynamics simulations can be modeled using semantic web technologies through annotating data with ontologies. We extend the already existing Dislocation Ontology by adding missing concepts and aligning it with two other domain-related ontologies (i.e., the Elementary Multi-perspectiv
    
[^3]: 用SVD图神经网络进行多行为推荐

    Multi-behavior Recommendation with SVD Graph Neural Networks. (arXiv:2309.06912v1 [cs.IR])

    [http://arxiv.org/abs/2309.06912](http://arxiv.org/abs/2309.06912)

    本研究提出了一种使用SVD图神经网络进行多行为推荐的模型MB-SVD，通过考虑用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。

    

    图神经网络(GNNs)广泛应用于推荐系统领域，为用户提供个性化推荐并取得显著成果。最近，融入对比学习的GNNs在处理推荐系统的稀疏数据问题方面表现出了很大的潜力。然而，现有的对比学习方法在解决冷启动问题和抵抗噪声干扰方面仍然存在限制，尤其是对于多行为推荐。为了缓解上述问题，本研究提出了一种基于GNNs的多行为推荐模型MB-SVD，利用奇异值分解(SVD)图来提高模型性能。具体而言，MB-SVD考虑了用户在不同行为下的偏好，改善了推荐效果，同时更好地解决了冷启动问题。我们的模型引入了一种创新的方法论，将多行为对比学习范式融入到模型中，以提高模型的性能。

    Graph Neural Networks (GNNs) has been extensively employed in the field of recommender systems, offering users personalized recommendations and yielding remarkable outcomes. Recently, GNNs incorporating contrastive learning have demonstrated promising performance in handling sparse data problem of recommendation system. However, existing contrastive learning methods still have limitations in addressing the cold-start problem and resisting noise interference especially for multi-behavior recommendation. To mitigate the aforementioned issues, the present research posits a GNNs based multi-behavior recommendation model MB-SVD that utilizes Singular Value Decomposition (SVD) graphs to enhance model performance. In particular, MB-SVD considers user preferences under different behaviors, improving recommendation effectiveness while better addressing the cold-start problem. Our model introduces an innovative methodology, which subsume multi-behavior contrastive learning paradigm to proficient
    
[^4]: 走向TopMost：一个主题建模系统工具包

    Towards the TopMost: A Topic Modeling System Toolkit. (arXiv:2309.06908v1 [cs.CL])

    [http://arxiv.org/abs/2309.06908](http://arxiv.org/abs/2309.06908)

    本文提出了一个名为TopMost的主题建模系统工具包，通过涵盖更广泛的主题建模场景和具有高度凝聚力和解耦模块化设计的特点，可以促进主题模型的研究和应用。

    

    主题模型已经在过去几十年中被提出，并且具有各种应用，在神经变分推断的推动下近期得到了更新。然而，这些主题模型采用完全不同的数据集、实现和评估设置，这阻碍了它们的快速利用和公平比较。这严重阻碍了主题模型的研究进展。为了解决这些问题，本文提出了一个主题建模系统工具包（TopMost）。与现有的工具包相比，TopMost通过涵盖更广泛的主题建模场景，包括数据集预处理、模型训练、测试和评估的完整生命周期，脱颖而出。TopMost的高度凝聚力和解耦模块化设计可以快速利用，公平比较，并灵活扩展不同的主题模型，这可以促进主题模型的研究和应用。我们的代码、教程和文档可在https://github.com/bobxwu/topmost 上获得。

    Topic models have been proposed for decades with various applications and recently refreshed by the neural variational inference. However, these topic models adopt totally distinct dataset, implementation, and evaluation settings, which hinders their quick utilization and fair comparisons. This greatly hinders the research progress of topic models. To address these issues, in this paper we propose a Topic Modeling System Toolkit (TopMost). Compared to existing toolkits, TopMost stands out by covering a wider range of topic modeling scenarios including complete lifecycles with dataset pre-processing, model training, testing, and evaluations. The highly cohesive and decoupled modular design of TopMost enables quick utilization, fair comparisons, and flexible extensions of different topic models. This can facilitate the research and applications of topic models. Our code, tutorials, and documentation are available at https://github.com/bobxwu/topmost.
    
[^5]: ProMap：电子商务产品映射数据集

    ProMap: Datasets for Product Mapping in E-commerce. (arXiv:2309.06882v1 [cs.LG])

    [http://arxiv.org/abs/2309.06882](http://arxiv.org/abs/2309.06882)

    该论文介绍了两个新的产品映射数据集：ProMapCz和ProMapEn，分别包含捷克和英文产品对，这些数据集具有较为完整的产品信息，并解决了目前现有数据集无法区分非常相似但不匹配产品对的问题。

    

    产品映射的目标是确定两个不同电子商店中的两个列表是否描述相同的产品。然而，现有的匹配和非匹配产品对的数据集经常受到产品信息不完整或者只包含非常远的非匹配产品的问题。因此，尽管在这些数据集上训练的预测模型取得了良好的结果，但实际上它们无法区分非常相似但不匹配的产品对，因此无法使用。本文介绍了两个新的产品映射数据集：ProMapCz包含1495对捷克产品，ProMapEn包含1555对英文产品，这些产品对来自两个电子商店，包含了产品的图像和文字描述，包括规格，使它们成为最完整的产品映射数据集之一。此外，非匹配产品是通过两个阶段进行选择的。

    The goal of product mapping is to decide, whether two listings from two different e-shops describe the same products. Existing datasets of matching and non-matching pairs of products, however, often suffer from incomplete product information or contain only very distant non-matching products. Therefore, while predictive models trained on these datasets achieve good results on them, in practice, they are unusable as they cannot distinguish very similar but non-matching pairs of products. This paper introduces two new datasets for product mapping: ProMapCz consisting of 1,495 Czech product pairs and ProMapEn consisting of 1,555 English product pairs of matching and non-matching products manually scraped from two pairs of e-shops. The datasets contain both images and textual descriptions of the products, including their specifications, making them one of the most complete datasets for product mapping. Additionally, the non-matching products were selected in two phases, creating two types 
    
[^6]: 使用原始像素为基准的推荐系统图像数据集

    An Image Dataset for Benchmarking Recommender Systems with Raw Pixels. (arXiv:2309.06789v1 [cs.IR])

    [http://arxiv.org/abs/2309.06789](http://arxiv.org/abs/2309.06789)

    这个研究提出了PixelRec，一个大规模图像推荐数据集，包括2亿个用户-图像交互、30万个用户和40万个高质量封面图像。通过提供原始图像像素的直接访问，PixelRec使得推荐模型能够直接从图像学习项目表示。

    

    推荐系统（RS）通过利用明确的识别（ID）特征取得了显著的成功。但是，内容特征，尤其是纯图像像素特征的全部潜力仍然相对未被开发。大规模、多样化、以内容为驱动的图像推荐数据集的有限可用性阻碍了使用原始图像作为项目表示的能力。在这方面，我们提出了PixelRec，一个大规模的以图像为中心的推荐数据集，包括约2亿个用户-图像交互、3000万个用户和40万个高质量封面图像。通过直接访问原始图像像素，PixelRec使得推荐模型能够直接从中学习项目表示。为了证明其效用，我们首先呈现了在PixelRec上训练的几个经典纯ID基线模型（称为IDNet）的结果。然后，为了展示数据集的图像特征的有效性，我们将项目ID嵌入（来自IDNet）与一个...

    Recommender systems (RS) have achieved significant success by leveraging explicit identification (ID) features. However, the full potential of content features, especially the pure image pixel features, remains relatively unexplored. The limited availability of large, diverse, and content-driven image recommendation datasets has hindered the use of raw images as item representations. In this regard, we present PixelRec, a massive image-centric recommendation dataset that includes approximately 200 million user-image interactions, 30 million users, and 400,000 high-quality cover images. By providing direct access to raw image pixels, PixelRec enables recommendation models to learn item representation directly from them. To demonstrate its utility, we begin by presenting the results of several classical pure ID-based baseline models, termed IDNet, trained on PixelRec. Then, to show the effectiveness of the dataset's image features, we substitute the itemID embeddings (from IDNet) with a 
    
[^7]: CONVERSER：使用合成数据生成的少样本对话密集检索

    CONVERSER: Few-Shot Conversational Dense Retrieval with Synthetic Data Generation. (arXiv:2309.06748v1 [cs.CL])

    [http://arxiv.org/abs/2309.06748](http://arxiv.org/abs/2309.06748)

    CONVERSER是一个使用少量对话样本进行训练的对话式密集检索框架，通过利用大型语言模型的上下文学习能力，能够生成与检索语料库中段落相关的对话查询，实验结果表明其在少样本对话密集检索中表现出与完全监督模型相当的性能。

    

    对话式搜索为信息检索提供了自然界面。最近的方法在对话式信息检索中应用了密集检索取得了有希望的结果。然而，训练密集检索器需要大量的领域相关的配对数据。这限制了对话式密集检索器的发展，因为收集大量领域相关对话是昂贵的。在本文中，我们提出了CONVERSER，这是一个用最多6对领域相关对话进行训练的对话式密集检索框架。具体而言，我们利用大型语言模型的上下文学习能力，根据检索语料库中的段落生成对话查询。对OR-QuAC和TREC CAsT 19等对话检索基准进行的实验结果表明，所提出的CONVERSER达到了与完全监督模型相当的性能，证明了我们提出的少样本对话密集检索框架的有效性。

    Conversational search provides a natural interface for information retrieval (IR). Recent approaches have demonstrated promising results in applying dense retrieval to conversational IR. However, training dense retrievers requires large amounts of in-domain paired data. This hinders the development of conversational dense retrievers, as abundant in-domain conversations are expensive to collect. In this paper, we propose CONVERSER, a framework for training conversational dense retrievers with at most 6 examples of in-domain dialogues. Specifically, we utilize the in-context learning capability of large language models to generate conversational queries given a passage in the retrieval corpus. Experimental results on conversational retrieval benchmarks OR-QuAC and TREC CAsT 19 show that the proposed CONVERSER achieves comparable performance to fully-supervised models, demonstrating the effectiveness of our proposed framework in few-shot conversational dense retrieval. All source code and
    
[^8]: 面向会话推荐的层次化多任务学习框架

    Hierarchical Multi-Task Learning Framework for Session-based Recommendations. (arXiv:2309.06533v1 [cs.IR])

    [http://arxiv.org/abs/2309.06533](http://arxiv.org/abs/2309.06533)

    本文提出了一种面向会话推荐的层次化多任务学习框架HierSRec，通过在预测任务之间设置层次结构，并利用辅助任务的输出来提供更丰富的输入特征和更高的预测可解释性，进一步增强了预测准确性和可泛化性。

    

    虽然会话推荐系统（SBRS）已经表现出卓越的推荐性能，但多任务学习（MTL）已经被SBRS采用以进一步提高其预测准确性和可泛化性。层次化多任务学习（H-MTL）在预测任务之间设置了层次结构，并将辅助任务的输出馈送给主任务。与现有的MTL框架相比，这种层次结构为主任务提供了更丰富的输入特征和更高的预测可解释性。然而，H-MTL框架在SBRS中尚未进行研究。在本文中，我们提出了HierSRec，将H-MTL架构纳入SBRS中。HierSRec使用元数据感知Transformer对给定会话进行编码，并使用会话编码进行下一类别预测（即辅助任务）。接下来，HierSRec使用类别预测结果和会话编码进行下一个物品预测（即主任务）。为了可扩展的推断，HierSRec创建了一个紧凑的候选物品集合。

    While session-based recommender systems (SBRSs) have shown superior recommendation performance, multi-task learning (MTL) has been adopted by SBRSs to enhance their prediction accuracy and generalizability further. Hierarchical MTL (H-MTL) sets a hierarchical structure between prediction tasks and feeds outputs from auxiliary tasks to main tasks. This hierarchy leads to richer input features for main tasks and higher interpretability of predictions, compared to existing MTL frameworks. However, the H-MTL framework has not been investigated in SBRSs yet. In this paper, we propose HierSRec which incorporates the H-MTL architecture into SBRSs. HierSRec encodes a given session with a metadata-aware Transformer and performs next-category prediction (i.e., auxiliary task) with the session encoding. Next, HierSRec conducts next-item prediction (i.e., main task) with the category prediction result and session encoding. For scalable inference, HierSRec creates a compact set of candidate items (
    
[^9]: AKEM: 利用集成模型将知识库与查询对齐以进行实体识别和链接

    AKEM: Aligning Knowledge Base to Queries with Ensemble Model for Entity Recognition and Linking. (arXiv:2309.06175v1 [cs.CL])

    [http://arxiv.org/abs/2309.06175](http://arxiv.org/abs/2309.06175)

    本文提出了一种利用集成模型将知识库与查询对齐的方法，用于实体识别和链接挑战。通过扩展知识库和利用外部知识，提高了召回率，并使用支持向量回归和多元加性回归树过滤结果得到高精度的实体识别和链接。最终实现了高效的计算和0.535的F1分数。

    

    本文提出了一种解决NLPCC 2015中实体识别和链接挑战的新方法。该任务包括从短搜索查询中提取命名实体的提及，并将其链接到参考中文知识库中的实体。为了解决这个问题，我们首先扩展现有知识库，并利用外部知识识别候选实体，从而提高召回率。接下来，我们从候选实体中提取特征，并利用支持向量回归和多元加性回归树作为评分函数来过滤结果。此外，我们还应用规则来进一步细化结果和提高精度。我们的方法计算效率高，达到了0.535的F1分数。

    This paper presents a novel approach to address the Entity Recognition and Linking Challenge at NLPCC 2015. The task involves extracting named entity mentions from short search queries and linking them to entities within a reference Chinese knowledge base. To tackle this problem, we first expand the existing knowledge base and utilize external knowledge to identify candidate entities, thereby improving the recall rate. Next, we extract features from the candidate entities and utilize Support Vector Regression and Multiple Additive Regression Tree as scoring functions to filter the results. Additionally, we apply rules to further refine the results and enhance precision. Our method is computationally efficient and achieves an F1 score of 0.535.
    
[^10]: 预训练、提示和推荐：语言模型范式在推荐系统中的综合调查

    Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems. (arXiv:2302.03735v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.03735](http://arxiv.org/abs/2302.03735)

    本文系统地研究了如何从不同预训练语言模型中提取和转移知识，提高推荐系统性能。我们提出了一个分类法，分析和总结了基于预训练语言模型的推荐系统的培训策略和目标。

    

    预训练语言模型（PLM）的出现，通过自监督方式在大型语料库上学习通用表示，在自然语言处理（NLP）领域取得了巨大成功。预训练模型和学到的表示可受益于一系列下游NLP任务。这种培训范式最近被适用于推荐领域，并被学术界和工业界认为是一种有前途的方法。本文系统地研究了如何从不同PLM相关训练范式学习到的预训练模型中提取和转移知识，从多个角度（如通用性、稀疏性、效率和效果）提高推荐性能。具体而言，我们提出了一个正交分类法来划分现有的基于PLM的推荐系统，针对其培训策略和目标进行分析和总结。

    The emergency of Pre-trained Language Models (PLMs) has achieved tremendous success in the field of Natural Language Processing (NLP) by learning universal representations on large corpora in a self-supervised manner. The pre-trained models and the learned representations can be beneficial to a series of downstream NLP tasks. This training paradigm has recently been adapted to the recommendation domain and is considered a promising approach by both academia and industry. In this paper, we systematically investigate how to extract and transfer knowledge from pre-trained models learned by different PLM-related training paradigms to improve recommendation performance from various perspectives, such as generality, sparsity, efficiency and effectiveness. Specifically, we propose an orthogonal taxonomy to divide existing PLM-based recommender systems w.r.t. their training strategies and objectives. Then, we analyze and summarize the connection between PLM-based training paradigms and differe
    
[^11]: 植物流行病在农田中的成本最优播种策略

    Cost-optimal Seeding Strategy During a Botanical Pandemic in Domesticated Fields. (arXiv:2301.02817v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.02817](http://arxiv.org/abs/2301.02817)

    这项研究提出了一种基于网格的经济最优播种策略，通过数学模型描述了在植物流行病期间农田作物的经济利润，可为农田主人和决策者提供指导。

    

    背景：植物流行病在全球范围内造成了巨大的经济损失和粮食短缺。然而，由于植物流行病在短中期内将继续存在，农田主人可以根据策略性地在自己的农田中播种，以优化每一次作物生产的经济利润。目标：鉴于病原体的流行病学特性，我们旨在为农田主人和决策者寻找一种基于网格的经济最优播种策略。方法：我们提出了一种新颖的流行病学-经济数学模型，描述了在植物流行病期间农田作物的经济利润。我们使用时空扩展的易感-感染-康复流行病学模型以及非线性输出流行病学模型来描述流行病学动态。结果和结论：我们提供了一种算法，用于根据农田和病原体的特性获取最优的网格形成的播种策略，以最大化经济利润。此外，我们还在现实情况下实施了提出的模型。

    Context: Botanical pandemics cause enormous economic damage and food shortage around the globe. However, since botanical pandemics are here to stay in the short-medium term, domesticated field owners can strategically seed their fields to optimize each session's economic profit. Objective: Given the pathogen's epidemiological properties, we aim to find an economically optimal grid-based seeding strategy for field owners and policymakers. Methods: We propose a novel epidemiological-economic mathematical model that describes the economic profit from a field of plants during a botanical pandemic. We describe the epidemiological dynamics using a spatio-temporal extended Susceptible-Infected-Recovered epidemiological model with a non-linear output epidemiological model. Results and Conclusions: We provide an algorithm to obtain an optimal grid-formed seeding strategy to maximize economic profit, given field and pathogen properties. In addition, we implement the proposed model in realistic s
    

