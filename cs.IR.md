# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction.](http://arxiv.org/abs/2310.09234) | 这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。 |
| [^2] | [AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems.](http://arxiv.org/abs/2310.09233) | AgentCF 是一种基于自主语言代理的协作学习方法，在推荐系统中模拟用户和物品的交互，并优化这两类代理。 |
| [^3] | [EHI: End-to-end Learning of Hierarchical Index for Efficient Dense Retrieval.](http://arxiv.org/abs/2310.08891) | EHI是一种端到端学习的层次索引方法，用于高效密集检索。它同时学习嵌入和ANNS结构，通过使用密集路径嵌入来捕获索引的语义信息，以优化检索性能。 |
| [^4] | [Question Answering for Electronic Health Records: A Scoping Review of datasets and models.](http://arxiv.org/abs/2310.08759) | 本文对电子健康记录（EHR）中的问题回答进行了范围回顾。与其他医学QA任务不同，EHR QA通过从患者的医疗记录中获取答案。这项研究为现有的EHR QA作品提供了方法论回顾。 |
| [^5] | [Individual Variation Affects Outbreak Magnitude and Predictability in an Extended Multi-Pathogen SIR Model of Pigeons Vising Dairy Farms.](http://arxiv.org/abs/2310.08613) | 该研究通过一个扩展的多病原SIR模型，考虑个体差异和鸽子的移动动态，在鸽子访问奶牛场的过程中研究了疫情的规模和可预测性。研究结果对于减轻农业环境中疾病传播的风险具有重要意义。 |
| [^6] | [Predicting Lung Cancer's Metastats' Locations Using Bioclinical Model.](http://arxiv.org/abs/2310.08596) | 本研究开发了一个生物临床模型，利用三维计算机断层扫描（CT）预测肺癌转移的空间扩散，验证了在转移位置预测方面的高准确率。这一研究突出了生物物理学和机器学习模型结合在肺癌诊断和治疗方面的潜力。 |
| [^7] | [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.](http://arxiv.org/abs/2308.11131) | 本论文提出了一种名为ReLLa的检索增强大型语言模型框架，用于零样本和小样本推荐任务。通过语义用户行为检索（SUBR）来提取上下文中的有用信息，以改善LLMs的推荐性能。 |
| [^8] | [Efficient High-Resolution Template Matching with Vector Quantized Nearest Neighbour Fields.](http://arxiv.org/abs/2306.15010) | 本研究提出了一种高效的高分辨率模板匹配方法，通过向量量化和滤波来减少计算量和考虑变形，取得了最先进的性能。 |
| [^9] | [UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers.](http://arxiv.org/abs/2303.00807) | 该论文提出了一种无监督领域自适应方法，利用大型语言模型(LLMs)生成大量合成查询和reranker模型，蒸馏为高效的检索器，适用于长尾领域。 |

# 详细

[^1]: ClickPrompt: CTR模型是将语言模型适应为CTR预测的强大提示生成器

    ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction. (arXiv:2310.09234v1 [cs.IR])

    [http://arxiv.org/abs/2310.09234](http://arxiv.org/abs/2310.09234)

    这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。

    

    点击率（CTR）预测已经成为各种互联网应用程序中越来越不可或缺的。传统的CTR模型通过独热编码将多字段分类数据转换为ID特征，并提取特征之间的协同信号。这种范式的问题在于语义信息的丢失。另一方面的研究通过将输入数据转换为文本句子来探索预训练语言模型（PLM）在CTR预测中的潜力。虽然语义信号得到了保留，但它们通常无法捕捉到协同信息（如特征交互、纯ID特征），更不用说由庞大的模型大小带来的无法接受的推理开销了。在本文中，我们旨在为准确的CTR估计建立语义知识和协同知识，并解决推理效率问题。为了从两个领域中受益并弥合它们之间的差距，我们提出了一种新颖的模型-。

    Click-through rate (CTR) prediction has become increasingly indispensable for various Internet applications. Traditional CTR models convert the multi-field categorical data into ID features via one-hot encoding, and extract the collaborative signals among features. Such a paradigm suffers from the problem of semantic information loss. Another line of research explores the potential of pretrained language models (PLMs) for CTR prediction by converting input data into textual sentences through hard prompt templates. Although semantic signals are preserved, they generally fail to capture the collaborative information (e.g., feature interactions, pure ID features), not to mention the unacceptable inference overhead brought by the huge model size. In this paper, we aim to model both the semantic knowledge and collaborative knowledge for accurate CTR estimation, and meanwhile address the inference inefficiency issue. To benefit from both worlds and close their gaps, we propose a novel model-
    
[^2]: AgentCF: 基于自主语言代理的协作学习在推荐系统中的应用

    AgentCF: Collaborative Learning with Autonomous Language Agents for Recommender Systems. (arXiv:2310.09233v1 [cs.IR])

    [http://arxiv.org/abs/2310.09233](http://arxiv.org/abs/2310.09233)

    AgentCF 是一种基于自主语言代理的协作学习方法，在推荐系统中模拟用户和物品的交互，并优化这两类代理。

    

    近年来，利用具有出色决策能力的LLM（语言混合模型）代理作为可信的人类代理出现了。然而，现有研究主要集中在模拟人类对话上。人类的非语言行为，如推荐系统中的物品点击，虽然隐含着用户的偏好并能提升用户建模，但尚未得到深入探索。主要原因在于语言建模与行为建模之间的差距，以及LLMs对用户-物品关系的不理解。为了解决这个问题，我们提出了AgentCF，通过基于代理的协同过滤来模拟推荐系统中的用户-物品交互。我们创造性地考虑用户和物品作为代理，并开发了一种协作学习方法来优化这两种代理。具体而言，每个时间步，我们首先促使用户代理和物品代理自主地进行交互。然后，基于差异性对这两类代理进行优化。

    Recently, there has been an emergence of employing LLM-powered agents as believable human proxies, based on their remarkable decision-making capability. However, existing studies mainly focus on simulating human dialogue. Human non-verbal behaviors, such as item clicking in recommender systems, although implicitly exhibiting user preferences and could enhance the modeling of users, have not been deeply explored. The main reasons lie in the gap between language modeling and behavior modeling, as well as the incomprehension of LLMs about user-item relations.  To address this issue, we propose AgentCF for simulating user-item interactions in recommender systems through agent-based collaborative filtering. We creatively consider not only users but also items as agents, and develop a collaborative learning approach that optimizes both kinds of agents together. Specifically, at each time step, we first prompt the user and item agents to interact autonomously. Then, based on the disparities b
    
[^3]: EHI: 高效密集检索的层次索引的端到端学习

    EHI: End-to-end Learning of Hierarchical Index for Efficient Dense Retrieval. (arXiv:2310.08891v1 [cs.LG])

    [http://arxiv.org/abs/2310.08891](http://arxiv.org/abs/2310.08891)

    EHI是一种端到端学习的层次索引方法，用于高效密集检索。它同时学习嵌入和ANNS结构，通过使用密集路径嵌入来捕获索引的语义信息，以优化检索性能。

    

    密集嵌入式检索现已成为语义搜索和排名问题的行业标准，如获取给定查询的相关网络文档。这些技术使用了两个阶段的过程：(a)对比学习来训练双编码器以嵌入查询和文档，以及(b)近似最近邻搜索(ANNS)以查找给定查询的相似文档。这两个阶段是不相交的；学得的嵌入可能不适合ANNS方法，反之亦然，导致性能不佳。在这项工作中，我们提出了一种名为端到端层次索引(EHI)的方法，它同时学习嵌入和ANNS结构以优化检索性能。EHI使用标准的双编码器模型来嵌入查询和文档，同时学习一个倒排文件索引(IVF)风格的树状结构以实现高效的ANNS。为了确保离散基于树的ANNS结构的稳定和高效学习，EHI引入了密集路径嵌入的概念，用来捕获索引的语义信息。

    Dense embedding-based retrieval is now the industry standard for semantic search and ranking problems, like obtaining relevant web documents for a given query. Such techniques use a two-stage process: (a) contrastive learning to train a dual encoder to embed both the query and documents and (b) approximate nearest neighbor search (ANNS) for finding similar documents for a given query. These two stages are disjoint; the learned embeddings might be ill-suited for the ANNS method and vice-versa, leading to suboptimal performance. In this work, we propose End-to-end Hierarchical Indexing -- EHI -- that jointly learns both the embeddings and the ANNS structure to optimize retrieval performance. EHI uses a standard dual encoder model for embedding queries and documents while learning an inverted file index (IVF) style tree structure for efficient ANNS. To ensure stable and efficient learning of discrete tree-based ANNS structure, EHI introduces the notion of dense path embedding that capture
    
[^4]: 电子健康记录的问题回答：数据集和模型的范围回顾

    Question Answering for Electronic Health Records: A Scoping Review of datasets and models. (arXiv:2310.08759v1 [cs.LG])

    [http://arxiv.org/abs/2310.08759](http://arxiv.org/abs/2310.08759)

    本文对电子健康记录（EHR）中的问题回答进行了范围回顾。与其他医学QA任务不同，EHR QA通过从患者的医疗记录中获取答案。这项研究为现有的EHR QA作品提供了方法论回顾。

    

    与患者相关的问题回答（QA）系统可以帮助临床医生和患者。它们可以帮助临床医生做决策，并使患者更好地了解他们的病历。大量的患者数据存储在电子健康记录（EHR）中，使得EHR QA成为一个重要的研究领域。在EHR QA中，答案是从患者的医疗记录中获得的。由于数据格式和模式的差异，这与其他使用医学网站或科学论文检索答案的医学QA任务有很大的不同，这使得研究EHR问题回答变得至关重要。本研究旨在对现有关于EHR QA的作品进行方法论回顾。我们在包括Google Scholar、ACL Anthology、ACM Digital Library和PubMed在内的四个数字资源中搜索了从2005年1月1日到2023年9月30日的文章，以收集有关EHR QA的相关出版物。共发现了4111篇论文。

    Question Answering (QA) systems on patient-related data can assist both clinicians and patients. They can, for example, assist clinicians in decision-making and enable patients to have a better understanding of their medical history. Significant amounts of patient data are stored in Electronic Health Records (EHRs), making EHR QA an important research area. In EHR QA, the answer is obtained from the medical record of the patient. Because of the differences in data format and modality, this differs greatly from other medical QA tasks that employ medical websites or scientific papers to retrieve answers, making it critical to research EHR question answering. This study aimed to provide a methodological review of existing works on QA over EHRs. We searched for articles from January 1st, 2005 to September 30th, 2023 in four digital sources including Google Scholar, ACL Anthology, ACM Digital Library, and PubMed to collect relevant publications on EHR QA. 4111 papers were identified for our
    
[^5]: 个体差异影响了鸽子访问奶牛场的扩展多病原SIR模型中的疫情规模和可预测性

    Individual Variation Affects Outbreak Magnitude and Predictability in an Extended Multi-Pathogen SIR Model of Pigeons Vising Dairy Farms. (arXiv:2310.08613v1 [q-bio.PE])

    [http://arxiv.org/abs/2310.08613](http://arxiv.org/abs/2310.08613)

    该研究通过一个扩展的多病原SIR模型，考虑个体差异和鸽子的移动动态，在鸽子访问奶牛场的过程中研究了疫情的规模和可预测性。研究结果对于减轻农业环境中疾病传播的风险具有重要意义。

    

    动物和人类之间的人畜共患病传播风险日益增加，农业环境作为传播的可能点，个体差异作为一个重要的要素。因此，了解野生动物和畜牧业界面上疾病传播的动态对于减轻这些传播风险至关重要。具体而言，鸽子与奶牛在奶牛场内的相互作用会导致重大的疾病传播和农民的经济损失，从而让畜牧动物、相邻人口和其他野生动物物种面临风险。本文提出了一种新颖的时空多病原模型，该模型具有连续的空间移动。该模型在易感-暴露-感染-康复-死亡（SEIRD）框架上进行了扩展，并考虑了病原体的种内和种间传播，以及鸽子在感染传播中扮演的探索-利用移动动态的关键作用。

    Zoonotic disease transmission between animals and humans is a growing risk and the agricultural context acts as a likely point of transition, with individual heterogeneity acting as an important contributor. Thus, understanding the dynamics of disease spread in the wildlife-livestock interface is crucial for mitigating these risks of transmission. Specifically, the interactions between pigeons and in-door cows at dairy farms can lead to significant disease transmission and economic losses for farmers; putting livestock, adjacent human populations, and other wildlife species at risk. In this paper, we propose a novel spatio-temporal multi-pathogen model with continuous spatial movement. The model expands on the Susceptible-Exposed-Infected-Recovered-Dead (SEIRD) framework and accounts for both within-species and cross-species transmission of pathogens, as well as the exploration-exploitation movement dynamics of pigeons, which play a critical role in the spread of infection agents. In a
    
[^6]: 预测肺癌转移位置的生物临床模型研究

    Predicting Lung Cancer's Metastats' Locations Using Bioclinical Model. (arXiv:2310.08596v1 [eess.IV])

    [http://arxiv.org/abs/2310.08596](http://arxiv.org/abs/2310.08596)

    本研究开发了一个生物临床模型，利用三维计算机断层扫描（CT）预测肺癌转移的空间扩散，验证了在转移位置预测方面的高准确率。这一研究突出了生物物理学和机器学习模型结合在肺癌诊断和治疗方面的潜力。

    

    肺癌是全球癌症相关死亡的主要原因。疾病从原发部位扩散到肺部其他部位，即转移，对治疗过程产生了重大影响。及早识别转移病灶对于及时有效的治疗至关重要，但传统影像技术在检测小的转移病灶上存在局限。在本研究中，我们开发了一个生物临床模型，利用三维计算机断层扫描（CT）预测肺癌转移的空间扩散。我们使用一个三层生物模型预测具有转移结节高概率的位置。我们在10名患者的实际数据上验证了生物临床模型，表明在转移位置预测方面有着令人期待的74%的准确率。我们的研究突出了生物物理学和机器学习模型结合在肺癌诊断和治疗方面的潜力，提供了更全面的方法。

    Lung cancer is a leading cause of cancer-related deaths worldwide. The spread of the disease from its primary site to other parts of the lungs, known as metastasis, significantly impacts the course of treatment. Early identification of metastatic lesions is crucial for prompt and effective treatment, but conventional imaging techniques have limitations in detecting small metastases. In this study, we develop a bioclinical model for predicting the spatial spread of lung cancer's metastasis using a three-dimensional computed tomography (CT) scan. We used a three-layer biological model of cancer spread to predict locations with a high probability of metastasis colonization. We validated the bioclinical model on real-world data from 10 patients, showing promising 74% accuracy in the metastasis location prediction. Our study highlights the potential of the combination of biophysical and ML models to advance the way that lung cancer is diagnosed and treated, by providing a more comprehensive
    
[^7]: ReLLa: 基于检索增强的大型语言模型的推荐系统中的生命周期序列行为理解

    ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation. (arXiv:2308.11131v1 [cs.IR])

    [http://arxiv.org/abs/2308.11131](http://arxiv.org/abs/2308.11131)

    本论文提出了一种名为ReLLa的检索增强大型语言模型框架，用于零样本和小样本推荐任务。通过语义用户行为检索（SUBR）来提取上下文中的有用信息，以改善LLMs的推荐性能。

    

    随着大型语言模型（LLMs）在自然语言处理（NLP）领域取得了显著突破，基于LLM的推荐系统引起了广泛关注并被积极探索。本文专注于适应和增强纯大型语言模型以用于零样本和小样本推荐任务。首先，我们针对推荐领域中LLMs无法从长用户行为序列的文本上下文中提取有用信息的问题，提出并定义了生命周期序列行为理解问题。为了解决这个问题并提高LLMs的推荐性能，我们提出了一种新的框架，即检索增强的大型语言模型（ReLLa）。针对零样本推荐，我们执行语义用户行为检索（SUBR）来提高数据的利用率。

    With large language models (LLMs) achieving remarkable breakthroughs in natural language processing (NLP) domains, LLM-enhanced recommender systems have received much attention and have been actively explored currently. In this paper, we focus on adapting and empowering a pure large language model for zero-shot and few-shot recommendation tasks. First and foremost, we identify and formulate the lifelong sequential behavior incomprehension problem for LLMs in recommendation domains, i.e., LLMs fail to extract useful information from a textual context of long user behavior sequence, even if the length of context is far from reaching the context limitation of LLMs. To address such an issue and improve the recommendation performance of LLMs, we propose a novel framework, namely Retrieval-enhanced Large Language models (ReLLa) for recommendation tasks in both zero-shot and few-shot settings. For zero-shot recommendation, we perform semantic user behavior retrieval (SUBR) to improve the data
    
[^8]: 高分辨率模板匹配中的高效向量量化最近邻场

    Efficient High-Resolution Template Matching with Vector Quantized Nearest Neighbour Fields. (arXiv:2306.15010v1 [cs.CV])

    [http://arxiv.org/abs/2306.15010](http://arxiv.org/abs/2306.15010)

    本研究提出了一种高效的高分辨率模板匹配方法，通过向量量化和滤波来减少计算量和考虑变形，取得了最先进的性能。

    

    模板匹配是计算机视觉中的基础问题，并在物体检测、图像配准和物体跟踪等领域有应用。当前最先进的方法是依赖于最近邻（NN）匹配，在该方法中，将查询特征空间转换为NN空间，其中每个查询像素用模板像素中的最近邻表示。NN匹配在遮挡、外观变化、光照变化和非刚性变换等方面表现出更好的性能。然而，NN匹配在高分辨率数据和高维特征方面的扩展性较差。本文提出了一种基于NN的模板匹配方法，该方法有效地减少了NN计算量，并在NN场中引入滤波以考虑变形。首先，通过向量量化将模板表示为k个特征，然后通过滤波比较模板和查询在k个特征上的分布。我们展示了该方法达到了最先进的性能。

    Template matching is a fundamental problem in computer vision and has applications in various fields, such as object detection, image registration, and object tracking. The current state-of-the-art methods rely on nearest-neighbour (NN) matching in which the query feature space is converted to NN space by representing each query pixel with its NN in the template pixels. The NN-based methods have been shown to perform better in occlusions, changes in appearance, illumination variations, and non-rigid transformations. However, NN matching scales poorly with high-resolution data and high feature dimensions. In this work, we present an NN-based template-matching method which efficiently reduces the NN computations and introduces filtering in the NN fields to consider deformations. A vector quantization step first represents the template with $k$ features, then filtering compares the template and query distributions over the $k$ features. We show that state-of-the-art performance was achiev
    
[^9]: UDAPDR: 基于LLM提示与reranker蒸馏的无监督领域自适应

    UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers. (arXiv:2303.00807v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2303.00807](http://arxiv.org/abs/2303.00807)

    该论文提出了一种无监督领域自适应方法，利用大型语言模型(LLMs)生成大量合成查询和reranker模型，蒸馏为高效的检索器，适用于长尾领域。

    

    很多信息检索任务需要大型标注数据集进行微调，但这样的数据集通常不可用，且在应用于真实场景中时可能会因为领域漂移而迅速失去效用。为了解决这个问题，我们提出一种使用大型语言模型(LLMs)廉价生成大量合成查询的方法。该方法首先利用昂贵的LLM生成少量合成查询，然后再利用成本较低的LLM生成大量的合成查询以微调一组reranker模型。最后，这些reranker会被蒸 distill 成一个高效的检索器，用于目标领域中的检索。实验证明，这种技术可以提高长尾领域中的零样本准确性，即使只使用2K个合成查询进行微调，并且比标准的reranking方法具有更低的延迟。我们提供完整的端到端方案，包括合成数据集等。

    Many information retrieval tasks require large labeled datasets for fine-tuning. However, such datasets are often unavailable, and their utility for real-world applications can diminish quickly due to domain shifts. To address this challenge, we develop and motivate a method for using large language models (LLMs) to generate large numbers of synthetic queries cheaply. The method begins by generating a small number of synthetic queries using an expensive LLM. After that, a much less expensive one is used to create large numbers of synthetic queries, which are used to fine-tune a family of reranker models. These rerankers are then distilled into a single efficient retriever for use in the target domain. We show that this technique boosts zero-shot accuracy in long-tail domains, even where only 2K synthetic queries are used for fine-tuning, and that it achieves substantially lower latency than standard reranking methods. We make our end-to-end approach, including our synthetic datasets an
    

