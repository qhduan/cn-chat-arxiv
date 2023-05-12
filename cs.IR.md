# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.](http://arxiv.org/abs/2305.07001) | 采用大型语言模型以指令遵循为方法的推荐系统，可以将用户偏好或需求进行自然语言描述，进而提高推荐精度。 |
| [^2] | [AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages.](http://arxiv.org/abs/2305.06897) | AfriQA是第一个专注于非洲语言的跨语言QA数据集，弥补了非洲语言数字化内容不足的问题。实验结果表明自动翻译和多语言检索模型的性能较差，需要支持跨语言推理和转移学习的模型。 |
| [^3] | [THUIR@COLIEE 2023: More Parameters and Legal Knowledge for Legal Case Entailment.](http://arxiv.org/abs/2305.06817) | 本文描述了THUIR团队在COLIEE 2023法律案例蕴涵任务中的方法，尝试了传统的词汇匹配方法和预训练语言模型，并采用学习排序方法进一步提高性能，结果表明更多的参数和法律知识对法律案例蕴涵任务有所贡献。 |
| [^4] | [THUIR@COLIEE 2023: Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval.](http://arxiv.org/abs/2305.06812) | 本文总结了THUIR在COLIEE 2023比赛中的冠军方案，其将结构化知识融入预训练语言模型，提出启发式预处理和后处理方法，采用学习排序方法进行特征合并，实验结果显示其具有卓越的优势。 |
| [^5] | [PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training.](http://arxiv.org/abs/2305.06622) | PerFedRec++采用自监督预训练技术，提高联邦推荐系统的个性化和推荐准确度。 |
| [^6] | [Backdoor to the Hidden Ground State: Planted Vertex Cover Example.](http://arxiv.org/abs/2305.06610) | 本论文发现了在正则随机图中存在一种新类型的自由能重构，称为eureka点，通过eureka点可以轻易访问具有消失自由能屏障的隐藏基态。 |
| [^7] | [How to Index Item IDs for Recommendation Foundation Models.](http://arxiv.org/abs/2305.06569) | 本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。 |
| [^8] | [A First Look at LLM-Powered Generative News Recommendation.](http://arxiv.org/abs/2305.06566) | 本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。 |
| [^9] | [Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction.](http://arxiv.org/abs/2305.06474) | 本文研究了大型语言模型（LLMs）在用户评分预测任务中的表现，与传统的协同过滤方法进行对比。结果发现LLMs能够在较少数据的情况下保持优秀的性能，并且在零样本和少样本情况下表现很好。 |
| [^10] | [Dynamic Graph Representation Learning for Depression Screening with Transformer.](http://arxiv.org/abs/2305.06447) | 利用Transformer进行抑郁症筛查，克服了传统方法的特征工程依赖和忽略时变因素的缺点。 |
| [^11] | [Uncovering ChatGPT's Capabilities in Recommender Systems.](http://arxiv.org/abs/2305.02182) | 本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。 |
| [^12] | [An Offline Metric for the Debiasedness of Click Models.](http://arxiv.org/abs/2304.09560) | 该论文介绍了一种离线评估点击模型去协变偏移的鲁棒性的方法，并提出了去偏差性这一概念和测量方法，这是恢复无偏一致相关性评分和点击模型对排名分布变化不变性的必要条件。 |
| [^13] | [Learning to Rank under Multinomial Logit Choice.](http://arxiv.org/abs/2009.03207) | 该论文提出了一个基于多项Logit选择模型的学习排序框架，能够更准确地捕捉用户在整个项目列表中的选择行为，为网站设计提供了更好的排序方案。 |

# 详细

[^1]: 推荐系统作为指令遵循的方法：大型语言模型增强的推荐方法

    Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach. (arXiv:2305.07001v1 [cs.IR])

    [http://arxiv.org/abs/2305.07001](http://arxiv.org/abs/2305.07001)

    采用大型语言模型以指令遵循为方法的推荐系统，可以将用户偏好或需求进行自然语言描述，进而提高推荐精度。

    

    在过去几十年中，推荐系统在研究和产业社区中引起了广泛关注，并且许多研究致力于开发有效的推荐模型。这些模型主要从历史行为数据中学习潜在的用户偏好，进而估计用户-项目匹配关系以进行推荐。受到大型语言模型（LLMs）近期进展的启发，我们采用了一种不同的方法来开发推荐模型，将推荐视为LLMs的指令遵循。关键思想是用户的偏好或需求可以用自然语言描述（称为指令）来表达，从而LLMs可以理解并进一步执行指令以达到推荐任务的目的。我们通过指令微调开源LLM（3B Flan-T5-XL）来开发推荐方法，以更好地使LLMs适应推荐系统。为此，我们首先提出了一种转换方法，将用户行为数据转化为指令，并在大规模电子商务数据集上评估了提出的方法。实验结果表明，我们的方法优于几种最先进的推荐方法，在推荐准确性方面取得了显着的改进。

    In the past decades, recommender systems have attracted much attention in both research and industry communities, and a large number of studies have been devoted to developing effective recommendation models. Basically speaking, these models mainly learn the underlying user preference from historical behavior data, and then estimate the user-item matching relationships for recommendations. Inspired by the recent progress on large language models (LLMs), we take a different approach to developing the recommendation models, considering recommendation as instruction following by LLMs. The key idea is that the preferences or needs of a user can be expressed in natural language descriptions (called instructions), so that LLMs can understand and further execute the instruction for fulfilling the recommendation task. Instead of using public APIs of LLMs, we instruction tune an open-source LLM (3B Flan-T5-XL), in order to better adapt LLMs to recommender systems. For this purpose, we first des
    
[^2]: AfriQA：针对非洲语言的跨语言开放检索问答

    AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages. (arXiv:2305.06897v1 [cs.CL])

    [http://arxiv.org/abs/2305.06897](http://arxiv.org/abs/2305.06897)

    AfriQA是第一个专注于非洲语言的跨语言QA数据集，弥补了非洲语言数字化内容不足的问题。实验结果表明自动翻译和多语言检索模型的性能较差，需要支持跨语言推理和转移学习的模型。

    

    数字化的非洲语言内容远远不足，这使得问答系统难以满足用户的信息需求。跨语言开放检索问答（XOR QA）系统--这些系统可以在为人们提供本地语言服务的同时从其他语言中获取答案内容--提供了一种填补这一差距的手段。为此，我们创建了AfriQA，这是第一个专注于非洲语言的跨语言QA数据集。AfriQA包括10种非洲语言的12,000多个XOR QA示例。尽管先前的数据集主要关注交叉语言QA增强目标语言覆盖范围的语言，但AfriQA侧重于交叉语言答案内容是唯一高覆盖范围答案内容的语言。因此，我们认为非洲语言是XOR QA中最重要和最现实的用例之一。我们的实验证明了自动翻译和多语言检索系统在我们的数据集上表现不佳，突显了需要支持跨语言推理和转移学习的模型。

    African languages have far less in-language content available digitally, making it challenging for question answering systems to satisfy the information needs of users. Cross-lingual open-retrieval question answering (XOR QA) systems -- those that retrieve answer content from other languages while serving people in their native language -- offer a means of filling this gap. To this end, we create AfriQA, the first cross-lingual QA dataset with a focus on African languages. AfriQA includes 12,000+ XOR QA examples across 10 African languages. While previous datasets have focused primarily on languages where cross-lingual QA augments coverage from the target language, AfriQA focuses on languages where cross-lingual answer content is the only high-coverage source of answer content. Because of this, we argue that African languages are one of the most important and realistic use cases for XOR QA. Our experiments demonstrate the poor performance of automatic translation and multilingual retri
    
[^3]: THUIR@COLIEE 2023：更多参数和法律知识用于法律案例蕴含问题

    THUIR@COLIEE 2023: More Parameters and Legal Knowledge for Legal Case Entailment. (arXiv:2305.06817v1 [cs.CL])

    [http://arxiv.org/abs/2305.06817](http://arxiv.org/abs/2305.06817)

    本文描述了THUIR团队在COLIEE 2023法律案例蕴涵任务中的方法，尝试了传统的词汇匹配方法和预训练语言模型，并采用学习排序方法进一步提高性能，结果表明更多的参数和法律知识对法律案例蕴涵任务有所贡献。

    

    本文描述了THUIR团队在COLIEE 2023法律案例蕴涵任务中的方法。该任务要求参与者从给定的支持案例中识别一个特定段落，该段落蕴含了查询案例的决定。我们尝试了传统的词汇匹配方法和具有不同大小的预训练语言模型，进一步采用学习排序方法提高性能。然而，学习排序方法在这个任务中并不是很健壮，这表明答案段落不能简单地通过信息检索技术确定。实验结果表明，更多的参数和法律知识对法律案例的蕴涵任务有所贡献。最后，我们在COLIEE 2023比赛中获得第三名。我们的方法的实现可以在https://github.com/CSHaitao/THUIR-COLIEE2023找到。

    This paper describes the approach of the THUIR team at the COLIEE 2023 Legal Case Entailment task. This task requires the participant to identify a specific paragraph from a given supporting case that entails the decision for the query case. We try traditional lexical matching methods and pre-trained language models with different sizes. Furthermore, learning-to-rank methods are employed to further improve performance. However, learning-to-rank is not very robust on this task. which suggests that answer passages cannot simply be determined with information retrieval techniques. Experimental results show that more parameters and legal knowledge contribute to the legal case entailment task. Finally, we get the third place in COLIEE 2023. The implementation of our method can be found at https://github.com/CSHaitao/THUIR-COLIEE2023.
    
[^4]: THUIR@COLIEE 2023: 将结构化知识融入预训练语言模型中用于法律案例检索

    THUIR@COLIEE 2023: Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval. (arXiv:2305.06812v1 [cs.IR])

    [http://arxiv.org/abs/2305.06812](http://arxiv.org/abs/2305.06812)

    本文总结了THUIR在COLIEE 2023比赛中的冠军方案，其将结构化知识融入预训练语言模型，提出启发式预处理和后处理方法，采用学习排序方法进行特征合并，实验结果显示其具有卓越的优势。

    

    法律案例检索技术在现代智能法律系统中起着重要作用，而作为一项年度知名国际比赛，COLIEE旨在实现针对法律文本的最先进检索模型。本文总结了冠军团队THUIR在COLIEE 2023的方法，具体而言，我们设计了结构感知的预训练语言模型以增强对法律案例的理解。此外，我们还提出了启发式预处理和后处理方法以减少无关信息的影响。最后，我们采用学习排序方法将具有不同维度的特征合并。实验结果表明我们的方案具有卓越的优势，官方结果显示我们的运行效果在所有提交中表现最佳。我们的方法实现可在https://github.com/CSHaitao/THUIR-COLIEE2023找到。

    Legal case retrieval techniques play an essential role in modern intelligent legal systems. As an annually well-known international competition, COLIEE is aiming to achieve the state-of-the-art retrieval model for legal texts. This paper summarizes the approach of the championship team THUIR in COLIEE 2023. To be specific, we design structure-aware pre-trained language models to enhance the understanding of legal cases. Furthermore, we propose heuristic pre-processing and post-processing approaches to reduce the influence of irrelevant messages. In the end, learning-to-rank methods are employed to merge features with different dimensions. Experimental results demonstrate the superiority of our proposal. Official results show that our run has the best performance among all submissions. The implementation of our method can be found at https://github.com/CSHaitao/THUIR-COLIEE2023.
    
[^5]: PerFedRec++：采用自监督预训练提高个性化联邦推荐

    PerFedRec++: Enhancing Personalized Federated Recommendation with Self-Supervised Pre-Training. (arXiv:2305.06622v1 [cs.IR])

    [http://arxiv.org/abs/2305.06622](http://arxiv.org/abs/2305.06622)

    PerFedRec++采用自监督预训练技术，提高联邦推荐系统的个性化和推荐准确度。

    

    联邦推荐系统通过采用联邦学习技术，在用户设备和中央服务器之间传输模型参数而非原始用户数据来保护用户隐私。然而，当前的联邦推荐系统面临着异构性和个性化、模型性能下降和通信瓶颈等挑战。先前的研究尝试解决这些问题，但均未能同时解决。我们在本文中提出了一个名为PerFedRec++的新框架，通过自监督预训练来增强个性化联邦推荐。具体而言，我们利用联邦推荐系统的隐私保护机制生成了两个增强图视图，并将其作为对比任务用于自监督图学习中的预训练。预训练通过提高表示学习的一致性来增强联邦模型的性能。

    Federated recommendation systems employ federated learning techniques to safeguard user privacy by transmitting model parameters instead of raw user data between user devices and the central server. Nevertheless, the current federated recommender system faces challenges such as heterogeneity and personalization, model performance degradation, and communication bottleneck. Previous studies have attempted to address these issues, but none have been able to solve them simultaneously.  In this paper, we propose a novel framework, named PerFedRec++, to enhance the personalized federated recommendation with self-supervised pre-training. Specifically, we utilize the privacy-preserving mechanism of federated recommender systems to generate two augmented graph views, which are used as contrastive tasks in self-supervised graph learning to pre-train the model. Pre-training enhances the performance of federated models by improving the uniformity of representation learning. Also, by providing a be
    
[^6]: 针对正则随机图的种植顶点覆盖问题及其发现的自由能重构

    Backdoor to the Hidden Ground State: Planted Vertex Cover Example. (arXiv:2305.06610v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2305.06610](http://arxiv.org/abs/2305.06610)

    本论文发现了在正则随机图中存在一种新类型的自由能重构，称为eureka点，通过eureka点可以轻易访问具有消失自由能屏障的隐藏基态。

    

    我们引入了一个针对正则随机图的种植顶点覆盖问题，并通过空穴方法对其进行研究。此二元自旋交互作用系统的平衡序相变具有不连续的性质，不同于常规的类似伊辛模型的连续相变，并且在广泛的自由能屏障的动态阻塞下。我们发现，该系统的无序对称相在除了唯一的eureka点$\beta_b$之外的所有逆温度下都可以在有序相的情况下局部稳定。 eureka点$\beta_b$为访问具有消失自由能屏障的隐藏基态提供了一个便道。它存在于无限系列的种植随机图集合中，并且我们通过分析确定了它们的结构参数。揭示出的新类型的自由能景观也可能存在于统计物理学和统计学界面的其他种植随机图优化问题中。

    We introduce a planted vertex cover problem on regular random graphs and study it by the cavity method. The equilibrium ordering phase transition of this binary-spin two-body interaction system is discontinuous in nature distinct from the continuous one of conventional Ising-like models, and it is dynamically blocked by an extensive free energy barrier. We discover that the disordered symmetric phase of this system may be locally stable with respect to the ordered phase at all inverse temperatures except for a unique eureka point $\beta_b$ at which it is only marginally stable. The eureka point $\beta_b$ serves as a backdoor to access the hidden ground state with vanishing free energy barrier. It exists in an infinite series of planted random graph ensembles and we determine their structural parameters analytically. The revealed new type of free energy landscape may also exist in other planted random-graph optimization problems at the interface of statistical physics and statistical in
    
[^7]: 如何为推荐基础模型索引项目ID

    How to Index Item IDs for Recommendation Foundation Models. (arXiv:2305.06569v1 [cs.IR])

    [http://arxiv.org/abs/2305.06569](http://arxiv.org/abs/2305.06569)

    本研究对推荐基础模型的项目索引问题进行了系统检查，提出了一种新的上下文感知索引方法，该方法在项目推荐准确性和文本生成质量方面具有优势。

    

    推荐基础模型将推荐任务转换为自然语言任务，利用大型语言模型（LLM）进行推荐。它通过直接生成建议的项目而不是计算传统推荐模型中每个候选项目的排名得分，简化了推荐管道，避免了多段过滤的问题。为了避免在决定要推荐哪些项目时生成过长的文本，为推荐基础模型创建LLM兼容的项目ID是必要的。本研究系统地研究了推荐基础模型的项目索引问题，以P5为代表的主干模型，并使用各种索引方法复制其结果。我们首先讨论了几种微不足道的项目索引方法（如独立索引、标题索引和随机索引）的问题，并表明它们不适用于推荐基础模型，然后提出了一种新的索引方法，称为上下文感知索引。我们表明，这种索引方法在项目推荐准确性和文本生成质量方面优于其他索引方法。

    Recommendation foundation model utilizes large language models (LLM) for recommendation by converting recommendation tasks into natural language tasks. It enables generative recommendation which directly generates the item(s) to recommend rather than calculating a ranking score for each and every candidate item in traditional recommendation models, simplifying the recommendation pipeline from multi-stage filtering to single-stage filtering. To avoid generating excessively long text when deciding which item(s) to recommend, creating LLM-compatible item IDs is essential for recommendation foundation models. In this study, we systematically examine the item indexing problem for recommendation foundation models, using P5 as the representative backbone model and replicating its results with various indexing methods. To emphasize the importance of item indexing, we first discuss the issues of several trivial item indexing methods, such as independent indexing, title indexing, and random inde
    
[^8]: LLM驱动的生成式新闻推荐初探

    A First Look at LLM-Powered Generative News Recommendation. (arXiv:2305.06566v1 [cs.IR])

    [http://arxiv.org/abs/2305.06566](http://arxiv.org/abs/2305.06566)

    本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。

    

    个性化的新闻推荐系统已成为用户浏览海量在线新闻内容所必需的工具，然而现有的新闻推荐系统面临着冷启动问题、用户画像建模和新闻内容理解等重大挑战。先前的研究通常通过模型设计遵循一种不灵活的例行程序来解决特定的挑战，但在理解新闻内容和捕捉用户兴趣方面存在局限性。在本文中，我们介绍了GENRE，一种LLM驱动的生成式新闻推荐框架，它利用来自大型语言模型的预训练语义知识来丰富新闻数据。我们的目标是通过从模型设计转移到提示设计来提供一种灵活而统一的新闻推荐解决方案。我们展示了GENRE在个性化新闻生成、用户画像和新闻摘要中的应用。使用各种流行的推荐模型进行的大量实验证明了GENRE的有效性。

    Personalized news recommendation systems have become essential tools for users to navigate the vast amount of online news content, yet existing news recommenders face significant challenges such as the cold-start problem, user profile modeling, and news content understanding. Previous works have typically followed an inflexible routine to address a particular challenge through model design, but are limited in their ability to understand news content and capture user interests. In this paper, we introduce GENRE, an LLM-powered generative news recommendation framework, which leverages pretrained semantic knowledge from large language models to enrich news data. Our aim is to provide a flexible and unified solution for news recommendation by moving from model design to prompt design. We showcase the use of GENRE for personalized news generation, user profiling, and news summarization. Extensive experiments with various popular recommendation models demonstrate the effectiveness of GENRE. 
    
[^9]: LLM是否能理解用户偏好？在用户评分预测任务中对LLM进行评估。

    Do LLMs Understand User Preferences? Evaluating LLMs On User Rating Prediction. (arXiv:2305.06474v1 [cs.IR])

    [http://arxiv.org/abs/2305.06474](http://arxiv.org/abs/2305.06474)

    本文研究了大型语言模型（LLMs）在用户评分预测任务中的表现，与传统的协同过滤方法进行对比。结果发现LLMs能够在较少数据的情况下保持优秀的性能，并且在零样本和少样本情况下表现很好。

    

    大型语言模型(LLMs)在零样本或少样本情况下展现出了杰出的泛化能力。然而，LLMs在基于用户以前的行为推断用户偏好方面能力的程度还是一个尚不清楚的问题。传统上，协同过滤(CF)是这些任务中最有效的方法，主要依赖于大量的评分数据。相比之下，LLMs通常需要更少的数据，同时又保持了每个项目(如电影或产品)的详尽的世界知识。在本文中，我们对用户评分预测这一经典任务中的CF和LLMs进行了全面的比较。这一任务涉及基于用户过去的评分预测候选项目的评分。我们研究了不同大小的LLMs，从250M到540B个参数，并评估了它们在零样本、少样本和微调场景下的性能。

    Large Language Models (LLMs) have demonstrated exceptional capabilities in generalizing to new tasks in a zero-shot or few-shot manner. However, the extent to which LLMs can comprehend user preferences based on their previous behavior remains an emerging and still unclear research question. Traditionally, Collaborative Filtering (CF) has been the most effective method for these tasks, predominantly relying on the extensive volume of rating data. In contrast, LLMs typically demand considerably less data while maintaining an exhaustive world knowledge about each item, such as movies or products. In this paper, we conduct a thorough examination of both CF and LLMs within the classic task of user rating prediction, which involves predicting a user's rating for a candidate item based on their past ratings. We investigate various LLMs in different sizes, ranging from 250M to 540B parameters and evaluate their performance in zero-shot, few-shot, and fine-tuning scenarios. We conduct comprehen
    
[^10]: 利用Transformer进行抑郁症筛查的动态图表示学习

    Dynamic Graph Representation Learning for Depression Screening with Transformer. (arXiv:2305.06447v1 [cs.LG])

    [http://arxiv.org/abs/2305.06447](http://arxiv.org/abs/2305.06447)

    利用Transformer进行抑郁症筛查，克服了传统方法的特征工程依赖和忽略时变因素的缺点。

    

    快速发现心理障碍至关重要，因为这样可以及时干预和治疗，从而大大改善患有严重心理疾病的个体的预后。社交媒体平台上最近出现的心理健康讨论的激增为研究心理健康提供了机会，并有可能检测到心理疾病的发生。然而，现有的抑郁症检测方法由于两个主要限制而受到限制：(1)依赖于特征工程，(2)没有考虑时变因素。具体而言，这些方法需要大量的特征工程和领域知识，其中严重依赖于用户生成内容的数量、质量和类型。此外，这些方法忽视了时变因素对抑郁症检测的重要影响，例如社交媒体上随时间推移而产生的语言模式和人际互动行为的动态变化(例如回复、提及和引用推文)。

    Early detection of mental disorder is crucial as it enables prompt intervention and treatment, which can greatly improve outcomes for individuals suffering from debilitating mental affliction. The recent proliferation of mental health discussions on social media platforms presents research opportunities to investigate mental health and potentially detect instances of mental illness. However, existing depression detection methods are constrained due to two major limitations: (1) the reliance on feature engineering and (2) the lack of consideration for time-varying factors. Specifically, these methods require extensive feature engineering and domain knowledge, which heavily rely on the amount, quality, and type of user-generated content. Moreover, these methods ignore the important impact of time-varying factors on depression detection, such as the dynamics of linguistic patterns and interpersonal interactive behaviors over time on social media (e.g., replies, mentions, and quote-tweets)
    
[^11]: 揭示ChatGPT在推荐系统中的能力

    Uncovering ChatGPT's Capabilities in Recommender Systems. (arXiv:2305.02182v1 [cs.IR])

    [http://arxiv.org/abs/2305.02182](http://arxiv.org/abs/2305.02182)

    本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。

    

    ChatGPT的问答功能吸引了自然语言处理（NLP）界及外界的关注。为了测试ChatGPT在推荐方面的表现，本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析。通过在不同领域的四个数据集上进行大量实验，我们发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型。基于单位成本改进的分析，我们确定ChatGPT在列表排名中能够在成本和性能之间实现最佳平衡，而在对和点排名中表现相对较弱。

    The debut of ChatGPT has recently attracted the attention of the natural language processing (NLP) community and beyond. Existing studies have demonstrated that ChatGPT shows significant improvement in a range of downstream NLP tasks, but the capabilities and limitations of ChatGPT in terms of recommendations remain unclear. In this study, we aim to conduct an empirical analysis of ChatGPT's recommendation ability from an Information Retrieval (IR) perspective, including point-wise, pair-wise, and list-wise ranking. To achieve this goal, we re-formulate the above three recommendation policies into a domain-specific prompt format. Through extensive experiments on four datasets from different domains, we demonstrate that ChatGPT outperforms other large language models across all three ranking policies. Based on the analysis of unit cost improvements, we identify that ChatGPT with list-wise ranking achieves the best trade-off between cost and performance compared to point-wise and pair-wi
    
[^12]: 离线度量点击模型的去偏差性

    An Offline Metric for the Debiasedness of Click Models. (arXiv:2304.09560v1 [cs.IR])

    [http://arxiv.org/abs/2304.09560](http://arxiv.org/abs/2304.09560)

    该论文介绍了一种离线评估点击模型去协变偏移的鲁棒性的方法，并提出了去偏差性这一概念和测量方法，这是恢复无偏一致相关性评分和点击模型对排名分布变化不变性的必要条件。

    

    在学习用户点击时，固有偏见是数据中普遍存在的一个问题，例如位置偏见或信任偏见。点击模型是从用户点击中提取信息的常用方法，例如在Web搜索中提取文档相关性，或者估计点击偏差以用于下游应用，例如反事实的学习排序、广告位置和公平排序。最近的研究表明，社区中的当前评估实践不能保证性能良好的点击模型对于下游任务的泛化能力，其中排名分布与训练分布不同，即在协变偏移下。在这项工作中，我们提出了一个基于条件独立性测试的评估度量，以检测点击模型对协变偏移的缺乏鲁棒性。我们引入了去偏差性的概念和一种测量方法。我们证明，去偏差性是恢复无偏的一致相关性评分以及使点击模型对排名分布变化的不变性的必要条件。

    A well-known problem when learning from user clicks are inherent biases prevalent in the data, such as position or trust bias. Click models are a common method for extracting information from user clicks, such as document relevance in web search, or to estimate click biases for downstream applications such as counterfactual learning-to-rank, ad placement, or fair ranking. Recent work shows that the current evaluation practices in the community fail to guarantee that a well-performing click model generalizes well to downstream tasks in which the ranking distribution differs from the training distribution, i.e., under covariate shift. In this work, we propose an evaluation metric based on conditional independence testing to detect a lack of robustness to covariate shift in click models. We introduce the concept of debiasedness and a metric for measuring it. We prove that debiasedness is a necessary condition for recovering unbiased and consistent relevance scores and for the invariance o
    
[^13]: 学习在多项Logit选择下进行排序

    Learning to Rank under Multinomial Logit Choice. (arXiv:2009.03207v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2009.03207](http://arxiv.org/abs/2009.03207)

    该论文提出了一个基于多项Logit选择模型的学习排序框架，能够更准确地捕捉用户在整个项目列表中的选择行为，为网站设计提供了更好的排序方案。

    

    在网站设计中，学习最佳内容排序是一个重要的挑战。学习排序（LTR）框架将这个问题建模为选择内容列表并观察用户决定点击的顺序问题。大多数以前的LTR工作假设用户在列表中独立考虑每个项目，并对每个项目进行二选一的选择。我们引入了多项式Logit（MNL）选择模型到LTR框架中，它捕捉到用户将有序的项目列表作为一个整体，从所有项目和没有点击选项中做出一个选择的行为。在MNL模型下，用户更喜欢本质上更有吸引力的项目，或者处于列表中更可取的位置的项目。我们提出了上置信界（UCB）算法，以在已知和未知的位置依赖参数的两种设置中最小化遗憾。我们提出了理论分析，导致了对问题的$\Omega（\sqrt{JT}）$下限。

    Learning the optimal ordering of content is an important challenge in website design. The learning to rank (LTR) framework models this problem as a sequential problem of selecting lists of content and observing where users decide to click. Most previous work on LTR assumes that the user considers each item in the list in isolation, and makes binary choices to click or not on each. We introduce a multinomial logit (MNL) choice model to the LTR framework, which captures the behaviour of users who consider the ordered list of items as a whole and make a single choice among all the items and a no-click option. Under the MNL model, the user favours items which are either inherently more attractive, or placed in a preferable position within the list. We propose upper confidence bound (UCB) algorithms to minimise regret in two settings where the position dependent parameters are known, and unknown. We present theoretical analysis leading to an $\Omega(\sqrt{JT})$ lower bound for the problem
    

