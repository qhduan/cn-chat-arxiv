# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Backtracing: Retrieving the Cause of the Query](https://arxiv.org/abs/2403.03956) | 引入了回溯任务，通过检索文本段来确定引发用户查询的原因，涉及到不同领域，包括讲座、新闻和对话，评估了零次性能。 |
| [^2] | [Bridging Language and Items for Retrieval and Recommendation](https://arxiv.org/abs/2403.03952) | 本文介绍了BLaIR，一个专门针对推荐场景的预训练句子嵌入模型，通过学习商品元数据与自然语言语境之间的相关性，提高了检索和推荐商品的效果。 |
| [^3] | [Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models](https://arxiv.org/abs/2403.03900) | Mamba4Rec是首个探索选择性状态空间模型用于高效序列推荐的工作，能够在保持推断效率的同时提升模型性能。 |
| [^4] | [Cobweb: An Incremental and Hierarchical Model of Human-Like Category Learning](https://arxiv.org/abs/2403.03835) | Cobweb是一种类似人类类别学习系统，采用类别效用度量构建分层组织的类似树状结构，能够捕捉心理效应并在单一模型中展现出实例和原型学习的灵活性，为将来研究人类类别学习提供了基础。 |
| [^5] | [Intent-aware Recommendation via Disentangled Graph Contrastive Learning](https://arxiv.org/abs/2403.03714) | 本文提出了一种通过解缠结图对比学习实现意图感知推荐的方法，可以同时学习可解释的意图以及这些意图上的行为分布 |
| [^6] | [Towards Efficient and Effective Unlearning of Large Language Models for Recommendation](https://arxiv.org/abs/2403.03536) | 提出了E2URec，这是为了解决大型语言模型在推荐系统中遗忘特定用户数据所面临的效率和有效性方面的挑战。 |
| [^7] | [Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling](https://arxiv.org/abs/2403.03516) | 通过生成伪标签实现的无监督多语言稠密检索方法能够在多语言信息检索中取得优异性能，提高了多语言检索器的实用性 |
| [^8] | [Generative News Recommendation](https://arxiv.org/abs/2403.03424) | 提出了一种新颖的生成式新闻推荐范式，通过利用大型语言模型进行高级匹配和生成连贯结构的叙述，帮助用户更全面理解事件。 |
| [^9] | [FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling](https://arxiv.org/abs/2403.02630) | 该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。 |
| [^10] | [Pfeed: Generating near real-time personalized feeds using precomputed embedding similarities](https://arxiv.org/abs/2402.16073) | 使用预计算的嵌入相似性生成个性化信息流，提高了电子商务平台上的客户参与度和体验，转化率提升4.9％。 |
| [^11] | [Retention Induced Biases in a Recommendation System with Heterogeneous Users](https://arxiv.org/abs/2402.13959) | 通过研究留存引发的偏见，发现改变推荐算法会导致推荐系统的行为在过渡期间与其新稳态不同，从而破坏了A/B实验作为评估RS改进的可靠性。 |
| [^12] | [Co-evolving Vector Quantization for ID-based Recommendation.](http://arxiv.org/abs/2308.16761) | 这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。 |

# 详细

[^1]: 回溯：检索查询原因

    Backtracing: Retrieving the Cause of the Query

    [https://arxiv.org/abs/2403.03956](https://arxiv.org/abs/2403.03956)

    引入了回溯任务，通过检索文本段来确定引发用户查询的原因，涉及到不同领域，包括讲座、新闻和对话，评估了零次性能。

    

    许多在线内容门户允许用户提出问题以补充他们的理解（例如，对讲座的理解）。虽然信息检索（IR）系统可以为这类用户查询提供答案，但它们并不直接帮助内容创建者（如希望改进内容的讲师）识别引发用户提出这些问题的段落。我们引入了回溯任务，系统检索出最有可能引发用户查询的文本段。我们对提高内容传递和沟通中重要的回溯任务进行了三个现实世界领域的形式化：（a）讲座领域中学生困惑的原因，（b）新闻文章领域中读者好奇心的原因，以及（c）对话领域中用户情绪的原因。我们评估了流行的信息检索方法和语言建模方法的零次性能，包括双编码器、重新排序和基于可能性的方法。

    arXiv:2403.03956v1 Announce Type: cross  Abstract: Many online content portals allow users to ask questions to supplement their understanding (e.g., of lectures). While information retrieval (IR) systems may provide answers for such user queries, they do not directly assist content creators -- such as lecturers who want to improve their content -- identify segments that _caused_ a user to ask those questions. We introduce the task of backtracing, in which systems retrieve the text segment that most likely caused a user query. We formalize three real-world domains for which backtracing is important in improving content delivery and communication: understanding the cause of (a) student confusion in the Lecture domain, (b) reader curiosity in the News Article domain, and (c) user emotion in the Conversation domain. We evaluate the zero-shot performance of popular information retrieval methods and language modeling methods, including bi-encoder, re-ranking and likelihood-based methods and 
    
[^2]: 将语言和物品联系起来进行检索和推荐

    Bridging Language and Items for Retrieval and Recommendation

    [https://arxiv.org/abs/2403.03952](https://arxiv.org/abs/2403.03952)

    本文介绍了BLaIR，一个专门针对推荐场景的预训练句子嵌入模型，通过学习商品元数据与自然语言语境之间的相关性，提高了检索和推荐商品的效果。

    

    本文介绍了BLaIR，这是一系列专门针对推荐场景的预训练句子嵌入模型。BLaIR被训练用于学习商品元数据与潜在自然语言语境之间的相关性，这对于检索和推荐商品很有用。为了预训练BLaIR，我们收集了Amazon Reviews 2023，这是一个新数据集，包括了来自33个类​​别的超过5.7亿条评论和4800万个物品，明显扩大了之前版本的范围。我们评估了BLaIR在多个领域和任务中的泛化能力，包括一个名为复杂产品搜索的新任务，指的是在给定长且复杂的自然语言语境的情况下检索相关物品。利用像ChatGPT这样的大型语言模型，我们相应地构建了一个半合成评估集Amazon-C4。实验结果表明，在新任务以及传统的检索和推荐任务中，BLaIR表现出优秀的性能。

    arXiv:2403.03952v1 Announce Type: new  Abstract: This paper introduces BLaIR, a series of pretrained sentence embedding models specialized for recommendation scenarios. BLaIR is trained to learn correlations between item metadata and potential natural language context, which is useful for retrieving and recommending items. To pretrain BLaIR, we collect Amazon Reviews 2023, a new dataset comprising over 570 million reviews and 48 million items from 33 categories, significantly expanding beyond the scope of previous versions. We evaluate the generalization ability of BLaIR across multiple domains and tasks, including a new task named complex product search, referring to retrieving relevant items given long, complex natural language contexts. Leveraging large language models like ChatGPT, we correspondingly construct a semi-synthetic evaluation set, Amazon-C4. Empirical results on the new task, as well as conventional retrieval and recommendation tasks, demonstrate that BLaIR exhibit stro
    
[^3]: Mamba4Rec：针对具有选择性状态空间模型的高效序列推荐

    Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models

    [https://arxiv.org/abs/2403.03900](https://arxiv.org/abs/2403.03900)

    Mamba4Rec是首个探索选择性状态空间模型用于高效序列推荐的工作，能够在保持推断效率的同时提升模型性能。

    

    序列推荐旨在估计动态用户偏好和历史用户行为之间的顺序依赖关系。本文提出了Mamba4Rec，这是首个探索选择性SSM潜力以实现高效序列推荐的工作。通过基本的Mamba块构建，结合一系列顺序建模技术，我们进一步提升了模型性能，同时保持了推断效率。实验证明，Mamba4Rec能够很好地处理序列推荐的有效性问题。

    arXiv:2403.03900v1 Announce Type: new  Abstract: Sequential recommendation aims to estimate the dynamic user preferences and sequential dependencies among historical user behaviors. Although Transformer-based models have proven to be effective for sequential recommendation, they suffer from the inference inefficiency problem stemming from the quadratic computational complexity of attention operators, especially for long-range behavior sequences. Inspired by the recent success of state space models (SSMs), we propose Mamba4Rec, which is the first work to explore the potential of selective SSMs for efficient sequential recommendation. Built upon the basic Mamba block which is a selective SSM with an efficient hardware-aware parallel algorithm, we incorporate a series of sequential modeling techniques to further promote the model performance and meanwhile maintain the inference efficiency. Experiments on two public datasets demonstrate that Mamba4Rec is able to well address the effectiven
    
[^4]: Cobweb：一种增量和分层式的人类类别学习模型

    Cobweb: An Incremental and Hierarchical Model of Human-Like Category Learning

    [https://arxiv.org/abs/2403.03835](https://arxiv.org/abs/2403.03835)

    Cobweb是一种类似人类类别学习系统，采用类别效用度量构建分层组织的类似树状结构，能够捕捉心理效应并在单一模型中展现出实例和原型学习的灵活性，为将来研究人类类别学习提供了基础。

    

    Cobweb是一种类似人类的类别学习系统，与其他增量分类模型不同的是，它利用类别效用度量构建分层组织的类似树状结构。先前的研究表明，Cobweb能够捕捉心理效应，如基本水平、典型性和扇形效应。然而，对Cobweb作为人类分类模型的更广泛评估仍然缺乏。本研究填补了这一空白。它确定了Cobweb与经典的人类类别学习效应的一致性。还探讨了Cobweb展现出在单一模型中既有实例又有原型学习的灵活性。这些发现为将来研究Cobweb作为人类类别学习的综合模型奠定了基础。

    arXiv:2403.03835v1 Announce Type: cross  Abstract: Cobweb, a human like category learning system, differs from other incremental categorization models in constructing hierarchically organized cognitive tree-like structures using the category utility measure. Prior studies have shown that Cobweb can capture psychological effects such as the basic level, typicality, and fan effects. However, a broader evaluation of Cobweb as a model of human categorization remains lacking. The current study addresses this gap. It establishes Cobweb's alignment with classical human category learning effects. It also explores Cobweb's flexibility to exhibit both exemplar and prototype like learning within a single model. These findings set the stage for future research on Cobweb as a comprehensive model of human category learning.
    
[^5]: 通过解缠结图对比学习实现意图感知推荐

    Intent-aware Recommendation via Disentangled Graph Contrastive Learning

    [https://arxiv.org/abs/2403.03714](https://arxiv.org/abs/2403.03714)

    本文提出了一种通过解缠结图对比学习实现意图感知推荐的方法，可以同时学习可解释的意图以及这些意图上的行为分布

    

    基于图神经网络（GNN）的推荐系统已经成为主流趋势之一，这是因为它能够从用户行为数据中强大地学习。从行为数据中理解用户意图是推荐系统的关键，这为基于GNN的推荐系统提出了两个基本要求。一个是在现实中用户行为通常是不足的情况下如何学习复杂而多样的意图。另一个是不同的行为具有不同的意图分布，因此如何建立它们之间的关系，以实现更有解释力的推荐系统。本文提出了一种通过解缠结图对比学习实现意图感知推荐（Intent-aware Recommendation via Disentangled Graph Contrastive Learning，IDCL），同时学习可解释的意图以及这些意图上的行为分布。具体而言，我们首先将用户行为数据建模为用户-物品-概念图，并设计了一个基于GNN的行为解缠结模块来学习不同的意图。

    arXiv:2403.03714v1 Announce Type: new  Abstract: Graph neural network (GNN) based recommender systems have become one of the mainstream trends due to the powerful learning ability from user behavior data. Understanding the user intents from behavior data is the key to recommender systems, which poses two basic requirements for GNN-based recommender systems. One is how to learn complex and diverse intents especially when the user behavior is usually inadequate in reality. The other is different behaviors have different intent distributions, so how to establish their relations for a more explainable recommender system. In this paper, we present the Intent-aware Recommendation via Disentangled Graph Contrastive Learning (IDCL), which simultaneously learns interpretable intents and behavior distributions over those intents. Specifically, we first model the user behavior data as a user-item-concept graph, and design a GNN based behavior disentangling module to learn the different intents. T
    
[^6]: 为推荐而设计的大型语言模型的高效和有效的遗忘

    Towards Efficient and Effective Unlearning of Large Language Models for Recommendation

    [https://arxiv.org/abs/2403.03536](https://arxiv.org/abs/2403.03536)

    提出了E2URec，这是为了解决大型语言模型在推荐系统中遗忘特定用户数据所面临的效率和有效性方面的挑战。

    

    大型语言模型（LLMs）的显著进展产生了一项有前途的研究方向，即利用LLMs作为推荐系统（LLMRec）。 LLMRec的有效性源自LLMs固有的开放世界知识和推理能力。 LLMRec通过基于用户互动数据的指导调整获得推荐功能。 然而，为了保护用户隐私并优化效用，LLMRec还必须有意忘记特定用户数据，这通常称为推荐遗忘。 在LLMs时代，推荐遗忘在\textit{效率}和\textit{有效性}方面为LLMRec带来了新挑战。 现有的遗忘方法需要更新LLMRec中数十亿参数，这是昂贵且耗时的。 此外，它们在遗忘过程中总是影响模型效用。 为此，我们提出了\textbf{E2URec}，第一

    arXiv:2403.03536v1 Announce Type: cross  Abstract: The significant advancements in large language models (LLMs) give rise to a promising research direction, i.e., leveraging LLMs as recommenders (LLMRec). The efficacy of LLMRec arises from the open-world knowledge and reasoning capabilities inherent in LLMs. LLMRec acquires the recommendation capabilities through instruction tuning based on user interaction data. However, in order to protect user privacy and optimize utility, it is also crucial for LLMRec to intentionally forget specific user data, which is generally referred to as recommendation unlearning. In the era of LLMs, recommendation unlearning poses new challenges for LLMRec in terms of \textit{inefficiency} and \textit{ineffectiveness}. Existing unlearning methods require updating billions of parameters in LLMRec, which is costly and time-consuming. Besides, they always impact the model utility during the unlearning process. To this end, we propose \textbf{E2URec}, the first
    
[^7]: 通过生成伪标签实现的无监督多语言稠密检索

    Unsupervised Multilingual Dense Retrieval via Generative Pseudo Labeling

    [https://arxiv.org/abs/2403.03516](https://arxiv.org/abs/2403.03516)

    通过生成伪标签实现的无监督多语言稠密检索方法能够在多语言信息检索中取得优异性能，提高了多语言检索器的实用性

    

    稠密检索方法在多语言信息检索中表现出色，但通常需要大量配对数据，这在多语言场景下更具挑战性。本文介绍了UMR，一种无需任何配对数据训练的无监督多语言稠密检索器。我们的方法利用多语言语言模型的序列似然估计能力来获取用于训练稠密检索器的伪标签。我们提出了一个两阶段框架，通过迭代改善多语言稠密检索器的性能。对两个基准数据集的实验结果表明，UMR的性能优于监督基线，展示了无需配对数据训练多语言检索器的潜力，从而提高了其实用性。我们的源代码、数据和模型已公开可用。

    arXiv:2403.03516v1 Announce Type: new  Abstract: Dense retrieval methods have demonstrated promising performance in multilingual information retrieval, where queries and documents can be in different languages. However, dense retrievers typically require a substantial amount of paired data, which poses even greater challenges in multilingual scenarios. This paper introduces UMR, an Unsupervised Multilingual dense Retriever trained without any paired data. Our approach leverages the sequence likelihood estimation capabilities of multilingual language models to acquire pseudo labels for training dense retrievers. We propose a two-stage framework which iteratively improves the performance of multilingual dense retrievers. Experimental results on two benchmark datasets show that UMR outperforms supervised baselines, showcasing the potential of training multilingual retrievers without paired data, thereby enhancing their practicality. Our source code, data, and models are publicly available
    
[^8]: 生成式新闻推荐

    Generative News Recommendation

    [https://arxiv.org/abs/2403.03424](https://arxiv.org/abs/2403.03424)

    提出了一种新颖的生成式新闻推荐范式，通过利用大型语言模型进行高级匹配和生成连贯结构的叙述，帮助用户更全面理解事件。

    

    大多数现有的新闻推荐方法通过在候选新闻和由历史点击新闻产生的用户表示之间进行语义匹配来处理此任务。但它们忽视了不同新闻文章之间的高级连接，也忽略了这些新闻文章与用户之间的深刻关系。这些方法的定义规定它们只能原样发布新闻文章。相反，将几篇相关新闻文章整合成连贯的叙述将帮助用户更快速、全面地理解事件。在本文中，我们提出了一种新颖的生成式新闻推荐范式，包括两个步骤：(1)利用大型语言模型（LLM）的内部知识和推理能力来进行候选新闻和用户表示之间的高级匹配；(2)基于提供更快速、全面理解事件。

    arXiv:2403.03424v1 Announce Type: new  Abstract: Most existing news recommendation methods tackle this task by conducting semantic matching between candidate news and user representation produced by historical clicked news. However, they overlook the high-level connections among different news articles and also ignore the profound relationship between these news articles and users. And the definition of these methods dictates that they can only deliver news articles as-is. On the contrary, integrating several relevant news articles into a coherent narrative would assist users in gaining a quicker and more comprehensive understanding of events. In this paper, we propose a novel generative news recommendation paradigm that includes two steps: (1) Leveraging the internal knowledge and reasoning capabilities of the Large Language Model (LLM) to perform high-level matching between candidate news and user representation; (2) Generating a coherent and logically structured narrative based on t
    
[^9]: FedHCDR: 具有超图信号解耦的联邦跨领域推荐

    FedHCDR: Federated Cross-Domain Recommendation with Hypergraph Signal Decoupling

    [https://arxiv.org/abs/2403.02630](https://arxiv.org/abs/2403.02630)

    该研究提出了FedHCDR框架，通过超图信号解耦的方式解决了联邦跨领域推荐中不同领域数据异质性的问题。

    

    近年来，跨领域推荐（CDR）备受关注，利用来自多个领域的用户数据来增强推荐性能。然而，当前的CDR方法需要跨领域共享用户数据，违反了《通用数据保护条例》（GDPR）。因此，已提出了许多联邦跨领域推荐（FedCDR）方法。然而，不同领域间的数据异质性不可避免地影响了联邦学习的整体性能。在这项研究中，我们提出了FedHCDR，一种具有超图信号解耦的新型联邦跨领域推荐框架。具体地，为了解决不同领域之间的数据异质性，我们引入一种称为超图信号解耦（HSD）的方法，将用户特征解耦为领域独有和领域共享特征。该方法采用高通和低通超图滤波器来进行解耦。

    arXiv:2403.02630v1 Announce Type: new  Abstract: In recent years, Cross-Domain Recommendation (CDR) has drawn significant attention, which utilizes user data from multiple domains to enhance the recommendation performance. However, current CDR methods require sharing user data across domains, thereby violating the General Data Protection Regulation (GDPR). Consequently, numerous approaches have been proposed for Federated Cross-Domain Recommendation (FedCDR). Nevertheless, the data heterogeneity across different domains inevitably influences the overall performance of federated learning. In this study, we propose FedHCDR, a novel Federated Cross-Domain Recommendation framework with Hypergraph signal decoupling. Specifically, to address the data heterogeneity across domains, we introduce an approach called hypergraph signal decoupling (HSD) to decouple the user features into domain-exclusive and domain-shared features. The approach employs high-pass and low-pass hypergraph filters to de
    
[^10]: 使用预计算的嵌入相似性生成几乎实时个性化信息流

    Pfeed: Generating near real-time personalized feeds using precomputed embedding similarities

    [https://arxiv.org/abs/2402.16073](https://arxiv.org/abs/2402.16073)

    使用预计算的嵌入相似性生成个性化信息流，提高了电子商务平台上的客户参与度和体验，转化率提升4.9％。

    

    在个性化推荐系统中，通常使用嵌入来编码用户动作和项目，然后在嵌入空间中进行检索，使用近似最近邻搜索。然而，这种方法可能会导致两个挑战：1）用户嵌入可能限制所捕获的兴趣多样性，2）保持它们最新需要昂贵的实时基础设施。本文提出了一种在实际工业环境中克服这些挑战的方法。该方法动态更新客户配置文件，并每两分钟组成一个信息流，利用预计算的嵌入及其各自的相似性。我们在荷兰和比利时最大的电子商务平台之一Bol上测试并部署了这种方法，该方法提高了客户参与度和体验，导致转化率显著提高了4.9％。

    arXiv:2402.16073v1 Announce Type: cross  Abstract: In personalized recommender systems, embeddings are often used to encode customer actions and items, and retrieval is then performed in the embedding space using approximate nearest neighbor search. However, this approach can lead to two challenges: 1) user embeddings can restrict the diversity of interests captured and 2) the need to keep them up-to-date requires an expensive, real-time infrastructure. In this paper, we propose a method that overcomes these challenges in a practical, industrial setting. The method dynamically updates customer profiles and composes a feed every two minutes, employing precomputed embeddings and their respective similarities. We tested and deployed this method to personalise promotional items at Bol, one of the largest e-commerce platforms of the Netherlands and Belgium. The method enhanced customer engagement and experience, leading to a significant 4.9% uplift in conversions.
    
[^11]: 具有异构用户的推荐系统中的留存引发偏见

    Retention Induced Biases in a Recommendation System with Heterogeneous Users

    [https://arxiv.org/abs/2402.13959](https://arxiv.org/abs/2402.13959)

    通过研究留存引发的偏见，发现改变推荐算法会导致推荐系统的行为在过渡期间与其新稳态不同，从而破坏了A/B实验作为评估RS改进的可靠性。

    

    我研究了一个具有用户流入和流失动态的推荐系统（RS）的概念模型。当流入和流失达到平衡时，用户分布达到稳定状态。改变推荐算法会改变稳定状态并产生过渡期。在这个期间，RS的行为与其新稳态不同。特别是，在过渡期内获得的A/B实验指标是RS长期性能的偏见指标。然而，学者和实践者经常在引入新算法后不久进行A/B测试以验证其有效性。然而，这种被广泛认为是评估RS改进的黄金标准的A/B实验范式可能产生错误结论。我还简要讨论了用户保留动态造成的数据偏见。

    arXiv:2402.13959v1 Announce Type: new  Abstract: I examine a conceptual model of a recommendation system (RS) with user inflow and churn dynamics. When inflow and churn balance out, the user distribution reaches a steady state. Changing the recommendation algorithm alters the steady state and creates a transition period. During this period, the RS behaves differently from its new steady state. In particular, A/B experiment metrics obtained in transition periods are biased indicators of the RS's long term performance. Scholars and practitioners, however, often conduct A/B tests shortly after introducing new algorithms to validate their effectiveness. This A/B experiment paradigm, widely regarded as the gold standard for assessing RS improvements, may consequently yield false conclusions. I also briefly discuss the data bias caused by the user retention dynamics.
    
[^12]: 基于ID的推荐的共同演化向量量化

    Co-evolving Vector Quantization for ID-based Recommendation. (arXiv:2308.16761v1 [cs.IR])

    [http://arxiv.org/abs/2308.16761](http://arxiv.org/abs/2308.16761)

    这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。

    

    类别信息对于提高推荐的质量和个性化起着至关重要的作用。然而，在基于ID的推荐中，项目类别信息的可用性并不一致。在这项工作中，我们提出了一种替代方法，以自动学习和生成实体（即用户和项目）在不同粒度级别上的分类信息，特别适用于基于ID的推荐。具体而言，我们设计了一个共同演化向量量化框架，即COVE，它能够同时学习和改进代码表示和实体嵌入，并以从随机初始化状态开始的端到端方式进行。通过其高度适应性，COVE可以轻松集成到现有的推荐模型中。我们验证了COVE在各种推荐任务中的有效性，包括列表完成、协同过滤和点击率预测，涵盖不同的推荐场景。

    Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommend
    

