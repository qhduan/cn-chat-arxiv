# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Precision Anti-Cancer Drug Selection via Neural Ranking.](http://arxiv.org/abs/2306.17771) | 通过神经排序方法，准确选择和优先排序敏感药物来进行个性化抗癌治疗。 |
| [^2] | [Outcome-based Evaluation of Systematic Review Automation.](http://arxiv.org/abs/2306.17614) | 提出一种新的评估框架，将考虑重新评估的影响。 |
| [^3] | [Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting.](http://arxiv.org/abs/2306.17563) | 本论文提出了一种名为PRP的新技术，通过使用两两排名提示来显著减轻大型语言模型（LLM）的负担，并首次在标准基准测试中实现了最先进的排名性能。 |
| [^4] | [Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework.](http://arxiv.org/abs/2306.17426) | 该论文提出了一种因果标记框架，利用观看时间反馈进行短视频推荐。通过构建多个语义的标签，并使用分位数来提取观看时间的信息，使模型学习更加容易，同时减少偏见对推荐结果的影响。 |
| [^5] | [Audio Embeddings as Teachers for Music Classification.](http://arxiv.org/abs/2306.17424) | 本文提出了一种使用预训练的音频嵌入作为教师来引导低复杂度学生网络的方法，并在音乐乐器分类和音乐自动标记任务上取得了显著的改进. |
| [^6] | [DeepTagger: Knowledge Enhanced Named Entity Recognition for Web-Based Ads Queries.](http://arxiv.org/abs/2306.17413) | DeepTagger是一种基于知识增强的网络广告查询的命名实体识别模型，通过模型无关和模型基于方法，利用未标记的网络查询和网络搜索结果来增加领域知识，采用大型语言模型自动生成标签，并应用对抗数据增强方法进行模型训练。 |
| [^7] | [Towards Personalized Cold-Start Recommendation with Prompts.](http://arxiv.org/abs/2306.17256) | 本研究旨在解决个性化冷启动推荐问题，通过利用预训练语言模型的能力，将推荐过程转化为自然语言情感分析，提供适用于创业企业和用户参与历史不足的平台的个性化推荐。 |
| [^8] | [Enforcing Data Geolocation Policies in Public Clouds using Trusted Computing.](http://arxiv.org/abs/2306.17171) | 本论文提出了一种使用可信计算的技术，可以限制公共云中数据的地理位置。用户将上传数据，并只与第三方验证服务器共享解密密钥。 |
| [^9] | [Harnessing the Power of Hugging Face Transformers for Predicting Mental Health Disorders in Social Networks.](http://arxiv.org/abs/2306.16891) | 该研究通过使用社交媒体和预训练的语言模型，探索了使用用户生成的数据预测精神障碍症状的方法，并发现新模型的准确度高达97%。这表明社交媒体数据是进行精神健康筛查的一个重要资源，预训练模型能够有效地自动化这一任务。 |
| [^10] | [Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints.](http://arxiv.org/abs/2306.12857) | 本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。 |
| [^11] | [Bounded (O(1)) Regret Recommendation Learning via Synthetic Controls Oracle.](http://arxiv.org/abs/2301.12571) | 通过合成控制理论，本论文提出了一种实现有界遗憾的推荐学习方法，并解决了线性模型的精确知识、潜在协变量的存在、不均匀的用户到达速率和用户选择退出私人数据跟踪等实践中的问题。 |
| [^12] | [Sequential Recommendation Model for Next Purchase Prediction.](http://arxiv.org/abs/2207.06225) | 本文提出了一种顺序推荐系统，考虑了用户的购买顺序以预测他们的下一次购买，该模型利用大规模的信用卡交易数据集进行了验证和排名，展现了其在准确性和效果上的优势。 |
| [^13] | [Conversational Question Answering on Heterogeneous Sources.](http://arxiv.org/abs/2204.11677) | 本文提出了CONVINSE，一个用于异构数据源上的ConvQA的端到端流水线，通过联合提取来自知识库、文本和表格的信息，提升了答案覆盖率和可信度。 |

# 详细

[^1]: 通过神经排序实现精确的抗癌药物选择

    Precision Anti-Cancer Drug Selection via Neural Ranking. (arXiv:2306.17771v1 [cs.LG])

    [http://arxiv.org/abs/2306.17771](http://arxiv.org/abs/2306.17771)

    通过神经排序方法，准确选择和优先排序敏感药物来进行个性化抗癌治疗。

    

    个性化癌症治疗需要对药物与癌细胞系在不同的遗传和分子环境中的复杂相互作用有深入的理解。为了解决这个问题，高通量筛选已被用来生成大规模的药物反应数据，促进数据驱动的计算模型。这些模型可以完全以数据驱动的方式捕捉到不同环境下复杂的药物-细胞系相互作用。然而，准确地为每个细胞系优先选择最敏感的药物仍然是一个重大挑战。为了解决这个问题，我们开发了神经排序方法，利用来自不同癌症类型的多个细胞系的大规模药物反应数据。与现有方法主要使用回归和分类技术进行药物反应预测不同，我们将药物选择和优先级确定的目标形式化为一个药物排序问题。在这项工作中，我们提出了两种神经排序方法，可以学习潜在的表示来解决这个问题。

    Personalized cancer treatment requires a thorough understanding of complex interactions between drugs and cancer cell lines in varying genetic and molecular contexts. To address this, high-throughput screening has been used to generate large-scale drug response data, facilitating data-driven computational models. Such models can capture complex drug-cell line interactions across various contexts in a fully data-driven manner. However, accurately prioritizing the most sensitive drugs for each cell line still remains a significant challenge. To address this, we developed neural ranking approaches that leverage large-scale drug response data across multiple cell lines from diverse cancer types. Unlike existing approaches that primarily utilize regression and classification techniques for drug response prediction, we formulated the objective of drug selection and prioritization as a drug ranking problem. In this work, we proposed two neural listwise ranking methods that learn latent repres
    
[^2]: 按结果评估系统化综述自动化方法

    Outcome-based Evaluation of Systematic Review Automation. (arXiv:2306.17614v1 [cs.IR])

    [http://arxiv.org/abs/2306.17614](http://arxiv.org/abs/2306.17614)

    提出一种新的评估框架，将考虑重新评估的影响。

    

    目前评估系统化文献综述的搜索策略和自动化引文筛选的方法通常依赖于计算相关和不相关出版物的数量。然而，这种做法并未准确反映进行系统化综述的现实情况，因为并非所有包含的出版物对最终的综述结果的影响相同。具体而言，如果重要的出版物被排除或包含，这可能会显著改变整个综述结果，而不包含或排除不太重要的研究可能只会有有限的影响。然而，在评估指标方面，所有的包含和排除决策被平等对待，因此未能检索出对综述结果几乎无影响的出版物与未能检索到关键出版物导致的召回率降低是一样的。我们提出了一个新的评估框架，考虑到了重新评估的影响。

    Current methods of evaluating search strategies and automated citation screening for systematic literature reviews typically rely on counting the number of relevant and not relevant publications. This established practice, however, does not accurately reflect the reality of conducting a systematic review, because not all included publications have the same influence on the final outcome of the systematic review. More specifically, if an important publication gets excluded or included, this might significantly change the overall review outcome, while not including or excluding less influential studies may only have a limited impact. However, in terms of evaluation measures, all inclusion and exclusion decisions are treated equally and, therefore, failing to retrieve publications with little to no impact on the review outcome leads to the same decrease in recall as failing to retrieve crucial publications. We propose a new evaluation framework that takes into account the impact of the re
    
[^3]: 大型语言模型是有效的文本排序器，具有两两排名提示

    Large Language Models are Effective Text Rankers with Pairwise Ranking Prompting. (arXiv:2306.17563v1 [cs.IR])

    [http://arxiv.org/abs/2306.17563](http://arxiv.org/abs/2306.17563)

    本论文提出了一种名为PRP的新技术，通过使用两两排名提示来显著减轻大型语言模型（LLM）的负担，并首次在标准基准测试中实现了最先进的排名性能。

    

    使用大型语言模型（LLM）通过直接将查询和候选文档输入提示进行文档排序是一个有趣且实用的问题。然而，迄今为止取得了有限的成功，研究人员发现很难在基准数据集上超越精调基准排序器。我们分析了现有方法使用的点对点和列表排序提示，并认为现成的LLM没有完全理解这些排序公式，可能是由于LLM的训练方式的特性。在本文中，我们提出了一种名为两两排名提示（PRP）的新技术，大大减轻了LLM的负担。我们的结果是文献中首次使用中等规模的开源LLM在标准基准测试中实现了最先进的排名性能。在TREC-DL2020上，基于20B参数的Flan-UL2模型的PRP超过了文献中基于商业黑盒GPT-4的最佳方法。

    Ranking documents using Large Language Models (LLMs) by directly feeding the query and candidate documents into the prompt is an interesting and practical problem. However, there has been limited success so far, as researchers have found it difficult to outperform fine-tuned baseline rankers on benchmark datasets. We analyze pointwise and listwise ranking prompts used by existing methods and argue that off-the-shelf LLMs do not fully understand these ranking formulations, possibly due to the nature of how LLMs are trained. In this paper, we propose to significantly reduce the burden on LLMs by using a new technique called Pairwise Ranking Prompting (PRP). Our results are the first in the literature to achieve state-of-the-art ranking performance on standard benchmarks using moderate-sized open-sourced LLMs. On TREC-DL2020, PRP based on the Flan-UL2 model with 20B parameters outperforms the previous best approach in the literature, which is based on the blackbox commercial GPT-4 that ha
    
[^4]: 利用观看时间反馈进行短视频推荐：一种因果标记框架

    Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework. (arXiv:2306.17426v1 [cs.IR])

    [http://arxiv.org/abs/2306.17426](http://arxiv.org/abs/2306.17426)

    该论文提出了一种因果标记框架，利用观看时间反馈进行短视频推荐。通过构建多个语义的标签，并使用分位数来提取观看时间的信息，使模型学习更加容易，同时减少偏见对推荐结果的影响。

    

    随着短视频应用的普及，短视频推荐的重要性大大增加。与其他推荐场景不同，短视频推荐系统大量依赖于观看时间的反馈。现有方法简单地将观看时间视为直接标签，未能充分利用其广泛的语义并引入偏见，从而限制了基于观看时间建模用户兴趣的潜力。为了克服这一挑战，我们提出了一个名为去偏多语义提取标记（DML）的框架。DML利用观看时间分布得出的分位数构建包含各种语义的标签，优先考虑相对顺序而不是绝对标签值。这种方法便于模型学习，同时符合推荐的排序目标。此外，我们引入了受因果调整启发的方法来优化标签定义，从而减少偏见对推荐结果的影响。

    With the proliferation of short video applications, the significance of short video recommendations has vastly increased. Unlike other recommendation scenarios, short video recommendation systems heavily rely on feedback from watch time. Existing approaches simply treat watch time as a direct label, failing to effectively harness its extensive semantics and introduce bias, thereby limiting the potential for modeling user interests based on watch time. To overcome this challenge, we propose a framework named Debiasied Multiple-semantics-extracting Labeling (DML). DML constructs labels that encompass various semantics by utilizing quantiles derived from the distribution of watch time, prioritizing relative order rather than absolute label values. This approach facilitates easier model learning while aligning with the ranking objective of recommendations. Furthermore, we introduce a method inspired by causal adjustment to refine label definitions, thereby reducing the impact of bias on th
    
[^5]: 音频嵌入作为音乐分类的教师

    Audio Embeddings as Teachers for Music Classification. (arXiv:2306.17424v1 [cs.SD])

    [http://arxiv.org/abs/2306.17424](http://arxiv.org/abs/2306.17424)

    本文提出了一种使用预训练的音频嵌入作为教师来引导低复杂度学生网络的方法，并在音乐乐器分类和音乐自动标记任务上取得了显著的改进.

    

    音乐分类一直是音乐信息检索领域中最受欢迎的任务之一。随着深度学习模型的发展，过去十年在各种分类任务中取得了令人瞩目的进展。然而，不断增加的模型复杂性使得训练和推断的计算成本变得昂贵。在本文中，我们将迁移学习和基于特征的知识蒸馏的思想相结合，并系统地研究使用预训练的音频嵌入作为教师来指导低复杂度的学生网络的训练。通过用预训练的嵌入方式规范化学生网络的特征空间，教师嵌入中的知识可以传递给学生。我们使用各种预训练的音频嵌入，并在音乐乐器分类和音乐自动标记任务上测试了该方法的有效性。结果表明，与同一方法相比，我们的方法显著提高了结果.

    Music classification has been one of the most popular tasks in the field of music information retrieval. With the development of deep learning models, the last decade has seen impressive improvements in a wide range of classification tasks. However, the increasing model complexity makes both training and inference computationally expensive. In this paper, we integrate the ideas of transfer learning and feature-based knowledge distillation and systematically investigate using pre-trained audio embeddings as teachers to guide the training of low-complexity student networks. By regularizing the feature space of the student networks with the pre-trained embeddings, the knowledge in the teacher embeddings can be transferred to the students. We use various pre-trained audio embeddings and test the effectiveness of the method on the tasks of musical instrument classification and music auto-tagging. Results show that our method significantly improves the results in comparison to the identical 
    
[^6]: DeepTagger: 基于知识增强的网络广告查询的命名实体识别

    DeepTagger: Knowledge Enhanced Named Entity Recognition for Web-Based Ads Queries. (arXiv:2306.17413v1 [cs.IR])

    [http://arxiv.org/abs/2306.17413](http://arxiv.org/abs/2306.17413)

    DeepTagger是一种基于知识增强的网络广告查询的命名实体识别模型，通过模型无关和模型基于方法，利用未标记的网络查询和网络搜索结果来增加领域知识，采用大型语言模型自动生成标签，并应用对抗数据增强方法进行模型训练。

    

    命名实体识别（NER）是在线广告的关键任务。最先进的解决方案利用预训练的语言模型来完成这项任务。然而，仍存在三个主要挑战：网络查询与预训练模型训练的自然语言不同；网络查询短小，缺乏上下文信息；NER的标记数据稀缺。我们提出了DeepTagger，一种基于知识增强的网络广告查询的NER模型。所提出的知识增强框架利用模型无关和模型基于方法。对于模型无关的增强，我们收集未标记的网络查询来增加领域知识；还收集网络搜索结果来丰富广告查询的信息。我们进一步利用ChatGPT等大型语言模型采用有效的提示方法自动生成标签。另外，我们采用基于对抗数据增强的模型基于知识增强方法。我们采用三阶段的训练框架。

    Named entity recognition (NER) is a crucial task for online advertisement. State-of-the-art solutions leverage pre-trained language models for this task. However, three major challenges remain unresolved: web queries differ from natural language, on which pre-trained models are trained; web queries are short and lack contextual information; and labeled data for NER is scarce. We propose DeepTagger, a knowledge-enhanced NER model for web-based ads queries. The proposed knowledge enhancement framework leverages both model-free and model-based approaches. For model-free enhancement, we collect unlabeled web queries to augment domain knowledge; and we collect web search results to enrich the information of ads queries. We further leverage effective prompting methods to automatically generate labels using large language models such as ChatGPT. Additionally, we adopt a model-based knowledge enhancement method based on adversarial data augmentation. We employ a three-stage training framework 
    
[^7]: 以提示为基础的个性化冷启动推荐的研究

    Towards Personalized Cold-Start Recommendation with Prompts. (arXiv:2306.17256v1 [cs.IR])

    [http://arxiv.org/abs/2306.17256](http://arxiv.org/abs/2306.17256)

    本研究旨在解决个性化冷启动推荐问题，通过利用预训练语言模型的能力，将推荐过程转化为自然语言情感分析，提供适用于创业企业和用户参与历史不足的平台的个性化推荐。

    

    推荐系统在根据用户过去的行为帮助用户发现与其兴趣相符的信息方面发挥着关键作用。然而，当用户和物品之间的历史交互记录不可用时，开发个性化推荐系统变得具有挑战性，这就是所谓的系统冷启动推荐问题。此问题在创业企业或用户参与历史不足的平台中尤为突出。以往的研究集中在用户或物品的冷启动场景，其中系统仍然通过在同一领域中的历史用户和物品交互进行训练来为新用户或物品提供推荐，而无法解决我们的问题。为了弥合这一鸿沟，我们的研究引入了一种创新且有效的方法，利用预训练语言模型的能力。我们将推荐过程转化为自然语言情感分析，其中包含用户资料和物品属性的信息。

    Recommender systems play a crucial role in helping users discover information that aligns with their interests based on their past behaviors. However, developing personalized recommendation systems becomes challenging when historical records of user-item interactions are unavailable, leading to what is known as the system cold-start recommendation problem. This issue is particularly prominent in start-up businesses or platforms with insufficient user engagement history. Previous studies focus on user or item cold-start scenarios, where systems could make recommendations for new users or items but are still trained with historical user-item interactions in the same domain, which cannot solve our problem. To bridge the gap, our research introduces an innovative and effective approach, capitalizing on the capabilities of pre-trained language models. We transform the recommendation process into sentiment analysis of natural languages containing information of user profiles and item attribu
    
[^8]: 在公共云中使用可信计算强制执行数据地理位置策略

    Enforcing Data Geolocation Policies in Public Clouds using Trusted Computing. (arXiv:2306.17171v1 [cs.DC])

    [http://arxiv.org/abs/2306.17171](http://arxiv.org/abs/2306.17171)

    本论文提出了一种使用可信计算的技术，可以限制公共云中数据的地理位置。用户将上传数据，并只与第三方验证服务器共享解密密钥。

    

    随着技术的进步，云计算通过革命性的解决方案自动化和简化复杂的计算任务，不断给世界带来惊喜。免维护成本、可访问性、数据备份、按使用付费模式、无限存储和处理能力等优势，鼓励个人和企业将工作负载迁移到云端。尽管云计算有许多优点，但数据在云环境中的地理位置是一个巨大的问题，它与数据所受到的性能和政府法规有关。数据地理位置的不清晰可能引发合规性问题。在这项工作中，我们提出了一种技术，允许用户限制其数据在云环境中的地理位置。我们使用可信计算机制远程验证主机及其地理位置。在这个模型中，用户将上传其数据，并将解密密钥仅与第三方验证服务器共享。

    With the advancement in technology, Cloud computing always amazes the world with revolutionizing solutions that automate and simplify complex computational tasks. The advantages like no maintenance cost, accessibility, data backup, pay-per-use models, unlimited storage, and processing power encourage individuals and businesses to migrate their workload to the cloud. Despite the numerous advantages of cloud computing, the geolocation of data in the cloud environment is a massive concern, which relates to the performance and government legislation that will be applied to data. The unclarity of data geolocation can cause compliance concerns. In this work, we have presented a technique that will allow users to restrict the geolocation of their data in the cloud environment. We have used trusted computing mechanisms to attest the host and its geolocation remotely. With this model, the user will upload the data whose decryption key will be shared with a third-party attestation server only. T
    
[^9]: 利用Hugging Face Transformers预测社交网络中的精神健康障碍的力量

    Harnessing the Power of Hugging Face Transformers for Predicting Mental Health Disorders in Social Networks. (arXiv:2306.16891v1 [cs.IR])

    [http://arxiv.org/abs/2306.16891](http://arxiv.org/abs/2306.16891)

    该研究通过使用社交媒体和预训练的语言模型，探索了使用用户生成的数据预测精神障碍症状的方法，并发现新模型的准确度高达97%。这表明社交媒体数据是进行精神健康筛查的一个重要资源，预训练模型能够有效地自动化这一任务。

    

    早期诊断精神障碍并进行干预可以促进预防严重伤害和改善治疗效果。本研究利用社交媒体和预训练的语言模型，探讨用户生成的数据如何用于预测精神障碍症状。我们的研究比较了Hugging Face的四种不同BERT模型和近期文献中用于自动抑郁症诊断的标准机器学习技术。结果显示，新模型的准确率高达97%，超过了以前的方法。通过补充先前的发现，对结果进行分析，我们发现即使是微小的数据量（如用户的个人简介描述）也有预测精神障碍的潜力。我们得出结论，社交媒体数据是进行精神健康筛查的一个极好的来源，并且预训练模型可以有效自动化这一关键任务。

    Early diagnosis of mental disorders and intervention can facilitate the prevention of severe injuries and the improvement of treatment results. Using social media and pre-trained language models, this study explores how user-generated data can be used to predict mental disorder symptoms. Our study compares four different BERT models of Hugging Face with standard machine learning techniques used in automatic depression diagnosis in recent literature. The results show that new models outperform the previous approach with an accuracy rate of up to 97%. Analyzing the results while complementing past findings, we find that even tiny amounts of data (like users' bio descriptions) have the potential to predict mental disorders. We conclude that social media data is an excellent source of mental health screening, and pre-trained models can effectively automate this critical task.
    
[^10]: 基于信息丢失约束的大规模公共安全时空数据高效划分方法

    Efficient Partitioning Method of Large-Scale Public Safety Spatio-Temporal Data based on Information Loss Constraints. (arXiv:2306.12857v1 [cs.LG])

    [http://arxiv.org/abs/2306.12857](http://arxiv.org/abs/2306.12857)

    本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)，可以显著减小数据规模，同时保持模型的准确性，确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    

    大规模时空数据的存储、管理和应用在各种实际场景中广泛应用，包括公共安全。然而，由于现实世界数据的独特时空分布特征，大多数现有方法在数据时空接近度和分布式存储负载平衡方面存在限制。因此，本文提出了一种基于信息丢失约束的大规模公共安全时空数据高效划分方法(IFL-LSTP)。该IFL-LSTP模型针对大规模时空点数据，将时空划分模块(STPM)和图划分模块(GPM)相结合。该方法可以显著减小数据规模，同时保持模型的准确性，以提高划分效率。它还可以确保分布式存储的负载平衡，同时保持数据划分的时空接近性。

    The storage, management, and application of massive spatio-temporal data are widely applied in various practical scenarios, including public safety. However, due to the unique spatio-temporal distribution characteristics of re-al-world data, most existing methods have limitations in terms of the spatio-temporal proximity of data and load balancing in distributed storage. There-fore, this paper proposes an efficient partitioning method of large-scale public safety spatio-temporal data based on information loss constraints (IFL-LSTP). The IFL-LSTP model specifically targets large-scale spatio-temporal point da-ta by combining the spatio-temporal partitioning module (STPM) with the graph partitioning module (GPM). This approach can significantly reduce the scale of data while maintaining the model's accuracy, in order to improve the partitioning efficiency. It can also ensure the load balancing of distributed storage while maintaining spatio-temporal proximity of the data partitioning res
    
[^11]: 通过合成控制理论实现有界（O(1)）遗憾的推荐学习

    Bounded (O(1)) Regret Recommendation Learning via Synthetic Controls Oracle. (arXiv:2301.12571v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.12571](http://arxiv.org/abs/2301.12571)

    通过合成控制理论，本论文提出了一种实现有界遗憾的推荐学习方法，并解决了线性模型的精确知识、潜在协变量的存在、不均匀的用户到达速率和用户选择退出私人数据跟踪等实践中的问题。

    

    在在线探索系统中，当具有固定偏好的用户重复到达时，最近已经证明可以将系统建模为线性情境广告带来O(1)的有界遗憾。这个结果可能对推荐系统具有兴趣，因为它们的物品的流行度通常是短暂的，即探索本身可能在潜在的长期非稳态开始之前很快完成。然而，在实践中，线性模型的精确知识往往难以证明。此外，潜在协变量的存在，不均匀的用户到达速率，对必要等级条件的解释以及用户选择退出私人数据跟踪等都需要在实际的推荐系统应用中解决。在这项工作中，我们进行了理论研究，以解决所有这些问题，同时仍然实现了有界遗憾。除了证明技术，我们在这里所做的关键区别性假设是有效合成控制理论的存在。

    In online exploration systems where users with fixed preferences repeatedly arrive, it has recently been shown that O(1), i.e., bounded regret, can be achieved when the system is modeled as a linear contextual bandit. This result may be of interest for recommender systems, where the popularity of their items is often short-lived, as the exploration itself may be completed quickly before potential long-run non-stationarities come into play. However, in practice, exact knowledge of the linear model is difficult to justify. Furthermore, potential existence of unobservable covariates, uneven user arrival rates, interpretation of the necessary rank condition, and users opting out of private data tracking all need to be addressed for practical recommender system applications. In this work, we conduct a theoretical study to address all these issues while still achieving bounded regret. Aside from proof techniques, the key differentiating assumption we make here is the presence of effective Sy
    
[^12]: 针对下一次购买预测的顺序推荐模型

    Sequential Recommendation Model for Next Purchase Prediction. (arXiv:2207.06225v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2207.06225](http://arxiv.org/abs/2207.06225)

    本文提出了一种顺序推荐系统，考虑了用户的购买顺序以预测他们的下一次购买，该模型利用大规模的信用卡交易数据集进行了验证和排名，展现了其在准确性和效果上的优势。

    

    在提供当代数字营销体验时，推荐的时效性和上下文准确性变得越来越重要。传统的推荐系统通过考虑用户的过去购买记录向用户推荐相关但不受时间影响的物品。这些推荐只是符合用户的一般偏好，而不是用户在购买之前的具体需求。相反，考虑交易、购买或体验顺序来衡量用户演化偏好的推荐系统能够为用户提供更准确和有效的推荐：顺序推荐系统不仅能更好地理解用户当前需求的行为，还具有更好的预测能力。在本文中，我们利用一份包含超过2.7百万信用卡交易数据和46K个持卡人的生产数据集，展示并排名了顺序推荐系统的效果。该方法首先使用自编码器对原始的交易数据进行处理，然后提交观测数据进行预测。

    Timeliness and contextual accuracy of recommendations are increasingly important when delivering contemporary digital marketing experiences. Conventional recommender systems (RS) suggest relevant but time-invariant items to users by accounting for their past purchases. These recommendations only map to customers' general preferences rather than a customer's specific needs immediately preceding a purchase. In contrast, RSs that consider the order of transactions, purchases, or experiences to measure evolving preferences can offer more salient and effective recommendations to customers: Sequential RSs not only benefit from a better behavioral understanding of a user's current needs but also better predictive power. In this paper, we demonstrate and rank the effectiveness of a sequential recommendation system by utilizing a production dataset of over 2.7 million credit card transactions for 46K cardholders. The method first employs an autoencoder on raw transaction data and submits observ
    
[^13]: 异构数据源上的对话问答

    Conversational Question Answering on Heterogeneous Sources. (arXiv:2204.11677v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.11677](http://arxiv.org/abs/2204.11677)

    本文提出了CONVINSE，一个用于异构数据源上的ConvQA的端到端流水线，通过联合提取来自知识库、文本和表格的信息，提升了答案覆盖率和可信度。

    

    会话问答(ConvQA)解决了顺序信息需求中的后续问题中上下文隐含的问题。当前的ConvQA系统只能在同质化的信息源上操作，如知识库(KB)、文本语料库或表格集合。本文针对的是在这些异质数据源中联合提取信息，从而提升答案覆盖率和可信度的新问题。我们提出了CONVINSE，一个用于异构数据源上的ConvQA的端到端流水线，分为三个阶段：i）学习对传入问题及其对话上下文的明确结构化表示，ii）利用这个类似框架的表示方式统一地获取到来自KB、文本和表格的相关证据，iii）运行融合解码模型来生成答案。我们构建并发布了第一个基准数据集ConvMix，用于异构数据源上的ConvQA，包括3000个真实用户对话和16000个问题，以及实体注释。

    Conversational question answering (ConvQA) tackles sequential information needs where contexts in follow-up questions are left implicit. Current ConvQA systems operate over homogeneous sources of information: either a knowledge base (KB), or a text corpus, or a collection of tables. This paper addresses the novel issue of jointly tapping into all of these together, this way boosting answer coverage and confidence. We present CONVINSE, an end-to-end pipeline for ConvQA over heterogeneous sources, operating in three stages: i) learning an explicit structured representation of an incoming question and its conversational context, ii) harnessing this frame-like representation to uniformly capture relevant evidences from KB, text, and tables, and iii) running a fusion-in-decoder model to generate the answer. We construct and release the first benchmark, ConvMix, for ConvQA over heterogeneous sources, comprising 3000 real-user conversations with 16000 questions, along with entity annotations,
    

