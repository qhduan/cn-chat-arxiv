# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Practicality of Dynamic Updates in Fast Searchable Encryption.](http://arxiv.org/abs/2308.13486) | 本论文研究了可搜索加密(SE)索引系统的实用性问题，并提出了一个实际运行的、端到端的SE实施，对动态SE更新操作进行了首次实证性性能评估。 |
| [^2] | [Leveraging Knowledge and Reinforcement Learning for Enhanced Reliability of Language Models.](http://arxiv.org/abs/2308.13467) | 本研究通过利用知识和强化学习的方法，实现了一个知识引导的语言模型集成，通过整合外部知识来弥补现有数据集中的信息缺失，从而提高了语言模型的可靠性和准确性。 |
| [^3] | [A Bayesian Active Learning Approach to Comparative Judgement.](http://arxiv.org/abs/2308.13292) | 这项研究提出了一种基于贝叶斯主动学习的比较评判方法，用于解决传统教育评估中存在的一致性和偏见等问题，并探索了如何选择比较项目的可靠数量。 |
| [^4] | [Learning and Optimization of Implicit Negative Feedback for Industrial Short-video Recommender System.](http://arxiv.org/abs/2308.13249) | 本文提出了一个在快手中的工业解决方案，通过部署一个关注反馈的编码模块和设计一个多目标预测模块，从大量的反馈中提取用户偏好并预测相关的短视频。 |
| [^5] | [Model-free Reinforcement Learning with Stochastic Reward Stabilization for Recommender Systems.](http://arxiv.org/abs/2308.13246) | 本文研究了在推荐系统中应用无模型强化学习的问题。针对推荐系统中随机奖励的特性，我们设计了两种随机奖励稳定化框架，用于更有效地处理随机反馈。我们的实验证明了这些框架的优越性。 |
| [^6] | [Optimizing Group-Fair Plackett-Luce Ranking Models for Relevance and Ex-Post Fairness.](http://arxiv.org/abs/2308.13242) | 提出了一种优化群组公平Plackett-Luce排序模型的方法，该方法最大化预期相关性并满足表示约束以确保后期公平性。 |
| [^7] | [MMBAttn: Max-Mean and Bit-wise Attention for CTR Prediction.](http://arxiv.org/abs/2308.13187) | 提出了一种利用最大均值和逐位注意力机制的CTR预测方法，可以准确估计特征的重要性，并通过考虑位级的细粒度交互，提高预测的精确性。 |
| [^8] | [Multi-BERT for Embeddings for Recommendation System.](http://arxiv.org/abs/2308.13050) | 本文提出了一种使用多种最先进的自然语言处理模型来生成文档嵌入的方法，并在图书推荐任务中取得了比单一模型更好的性能。 |
| [^9] | [Financial News Analytics Using Fine-Tuned Llama 2 GPT Model.](http://arxiv.org/abs/2308.13032) | 本研究通过精细调整的Llama 2模型实现了金融新闻的多任务分析，包括文本分析、摘要和情感提取等。实验结果显示，提取的命名实体情感可以作为有监督机器学习模型的预测特征。 |
| [^10] | [Replace Scoring with Arrangement: A Contextual Set-to-Arrangement Framework for Learning-to-Rank.](http://arxiv.org/abs/2308.02860) | 这篇论文提出了一种新的学习排序框架，名为STARank，它通过直接生成候选项目排列来替代个别评分和排序操作，并且是端到端可微分的。 |
| [^11] | [Graph-Based Recommendation System Enhanced with Community Detection.](http://arxiv.org/abs/2201.03622) | 本文提出了一个基于图的推荐系统，利用数学和统计方法确定标签的相似性，包括词汇相似性和共现解决方案，并考虑了标签分配的时间，以提高推荐的准确性。 |

# 详细

[^1]: 关于快速可搜索加密中动态更新的实用性研究

    On the Practicality of Dynamic Updates in Fast Searchable Encryption. (arXiv:2308.13486v1 [cs.CR])

    [http://arxiv.org/abs/2308.13486](http://arxiv.org/abs/2308.13486)

    本论文研究了可搜索加密(SE)索引系统的实用性问题，并提出了一个实际运行的、端到端的SE实施，对动态SE更新操作进行了首次实证性性能评估。

    

    可搜索加密(SE)索引系统是一种利用云服务存储和管理敏感信息的有用工具。然而，迄今为止，SE系统的大部分工作仍停留在理论阶段。为了使其实用化，需要更多工作来开发最优的协议和工作模型。这包括特别创建一个可工作的更新模型，以保持对动态文档集合（如电子邮件收件箱）进行加密索引的能力。我已经创建了一个实际运行的、端到端的SE实施，满足了这些需求，并进行了动态SE更新操作的第一次实证性性能评估。通过这样做，我展示了从先前研究人员描述的理论概念转向未来可投入生产的实施的可行路径，并确定了需要进一步研究的问题。

    Searchable encrypted (SE) indexing systems are a useful tool for utilizing cloud services to store and manage sensitive information. However, much of the work on SE systems to date has remained theoretical. In order to make them of practical use, more work is needed to develop optimal protocols and working models for them. This includes, in particular, the creation of a working update model in order to maintain an encrypted index of a dynamic document set such as an email inbox. I have created a working, real-world end-to-end SE implementation that satisfies these needs, including the first empirical performance evaluation of the dynamic SE update operation. In doing so, I show a viable path to move from the theoretical concepts described by previous researchers to a future production-worthy implementation and identify issues for follow-on investigation.
    
[^2]: 利用知识和强化学习提高语言模型的可靠性

    Leveraging Knowledge and Reinforcement Learning for Enhanced Reliability of Language Models. (arXiv:2308.13467v1 [cs.CL])

    [http://arxiv.org/abs/2308.13467](http://arxiv.org/abs/2308.13467)

    本研究通过利用知识和强化学习的方法，实现了一个知识引导的语言模型集成，通过整合外部知识来弥补现有数据集中的信息缺失，从而提高了语言模型的可靠性和准确性。

    

    自然语言处理(NLP)社区一直在使用众包技术，创建用于训练现代语言模型如BERT的基准数据集，例如General Language Understanding and Evaluation(GLUE)。GLUE任务使用互评计量方法（如Cohens Kappa）来衡量可靠性分数。然而，语言模型的可靠性方面常常被忽视。为解决这个问题，我们探索了一种知识引导的语言模型集成方法，利用强化学习将ConceptNet和维基百科的知识作为知识图嵌入进行整合。这种方法模仿了人类注释者使用外部知识来弥补数据集中的信息缺失。通过在九个GLUE数据集上的研究表明，语言模型集成可以增强可靠性和准确性得分，超过现有最先进方法。

    The Natural Language Processing(NLP) community has been using crowd sourcing techniques to create benchmark datasets such as General Language Understanding and Evaluation(GLUE) for training modern Language Models such as BERT. GLUE tasks measure the reliability scores using inter annotator metrics i.e. Cohens Kappa. However, the reliability aspect of LMs has often been overlooked. To counter this problem, we explore a knowledge-guided LM ensembling approach that leverages reinforcement learning to integrate knowledge from ConceptNet and Wikipedia as knowledge graph embeddings. This approach mimics human annotators resorting to external knowledge to compensate for information deficits in the datasets. Across nine GLUE datasets, our research shows that ensembling strengthens reliability and accuracy scores, outperforming state of the art.
    
[^3]: 基于贝叶斯主动学习的比较评判方法

    A Bayesian Active Learning Approach to Comparative Judgement. (arXiv:2308.13292v1 [cs.LG])

    [http://arxiv.org/abs/2308.13292](http://arxiv.org/abs/2308.13292)

    这项研究提出了一种基于贝叶斯主动学习的比较评判方法，用于解决传统教育评估中存在的一致性和偏见等问题，并探索了如何选择比较项目的可靠数量。

    

    评估是教育的关键部分。传统的评分方法存在一些问题，如一致性不足，存在无意识的偏见，给评估者带来较大的认知负担。比较评判（CJ）是一种解决这些问题的方法。在CJ中，评估者以一对项目为单位，选择哪个更好。通过一系列比较，可以使用排名模型（例如BTM）推导出一个排名。虽然CJ被认为是一种可靠的评分方法，但仍存在透明度和生成可靠排名所需的对比次数的理想数量等问题。此外，已有尝试提出了一些方法以有效方式选择下一个应比较的项目，但某些现有方法会在结果中产生自己的偏见，从而增加了所使用的可靠性度量。因此，通常使用随机选择的方法。本文提出了一种新颖的基于贝叶斯的方法。

    Assessment is a crucial part of education. Traditional marking is a source of inconsistencies and unconscious bias, placing a high cognitive load on the assessors. An approach to address these issues is comparative judgement (CJ). In CJ, the assessor is presented with a pair of items and is asked to select the better one. Following a series of comparisons, a rank is derived using a ranking model, for example, the BTM, based on the results. While CJ is considered a reliable method for marking, there are concerns around transparency, and the ideal number of pairwise comparisons to generate a reliable estimation of the rank order is not known. Additionally, there have been attempts to generate a method of selecting pairs that should be compared next in an informative manner, but some existing methods are known to have created their own bias within results inflating the reliability metric used. As a result, a random selection approach is usually deployed.  We propose a novel Bayesian appro
    
[^4]: 学习和优化工业短视频推荐系统的隐式负反馈

    Learning and Optimization of Implicit Negative Feedback for Industrial Short-video Recommender System. (arXiv:2308.13249v1 [cs.IR])

    [http://arxiv.org/abs/2308.13249](http://arxiv.org/abs/2308.13249)

    本文提出了一个在快手中的工业解决方案，通过部署一个关注反馈的编码模块和设计一个多目标预测模块，从大量的反馈中提取用户偏好并预测相关的短视频。

    

    短视频推荐是当今工业信息系统中最重要的推荐应用之一。与其他推荐任务相比，海量的反馈是最典型的特点。具体来说，在短视频推荐中，最容易收集到的用户反馈是跳过行为，这对推荐模型提出了两个重要的挑战。首先，跳过行为反映了隐式的用户偏好，因此对于兴趣提取是具有挑战性的。其次，这种特殊的反馈涉及到多个目标，如总观看时间，这也是非常具有挑战性的。在本文中，我们介绍了我们在快手中的工业解决方案，每天为十亿级用户提供服务。具体来说，我们部署了一个关注反馈的编码模块，很好地提取用户偏好并考虑了上下文的影响。我们进一步设计了一个多目标预测模块，能够很好地区分相关和不相关的视频。

    Short-video recommendation is one of the most important recommendation applications in today's industrial information systems. Compared with other recommendation tasks, the enormous amount of feedback is the most typical characteristic. Specifically, in short-video recommendation, the easiest-to-collect user feedback is from the skipping behaviors, which leads to two critical challenges for the recommendation model. First, the skipping behavior reflects implicit user preferences, and thus it is challenging for interest extraction. Second, the kind of special feedback involves multiple objectives, such as total watching time, which is also very challenging. In this paper, we present our industrial solution in Kuaishou, which serves billion-level users every day. Specifically, we deploy a feedback-aware encoding module which well extracts user preference taking the impact of context into consideration. We further design a multi-objective prediction module which well distinguishes the rel
    
[^5]: 无模型强化学习在推荐系统中的应用: 随机奖励稳定化

    Model-free Reinforcement Learning with Stochastic Reward Stabilization for Recommender Systems. (arXiv:2308.13246v1 [cs.LG])

    [http://arxiv.org/abs/2308.13246](http://arxiv.org/abs/2308.13246)

    本文研究了在推荐系统中应用无模型强化学习的问题。针对推荐系统中随机奖励的特性，我们设计了两种随机奖励稳定化框架，用于更有效地处理随机反馈。我们的实验证明了这些框架的优越性。

    

    最近，基于无模型的强化学习（RL）的推荐系统因其处理部分反馈和长期奖励的能力而受到越来越多的研究关注。然而，大多数现有研究忽略了推荐系统中的一个关键特征：同一用户在不同时间对同一项的反馈是随机的。随机奖励的特性与具有确定性奖励的经典RL场景本质上不同，这使得基于RL的推荐系统更具挑战性。本文首先在一个模拟环境中展示了直接使用随机反馈会导致性能显著下降。为了更有效地处理随机反馈，我们设计了两种随机奖励稳定化框架，用于用监督模型学习到的奖励替代直接的随机反馈。这两个框架都是模型无关的，即它们可以有效地利用各种监督模型。我们证明了所提出的框架的优越性。

    Model-free RL-based recommender systems have recently received increasing research attention due to their capability to handle partial feedback and long-term rewards. However, most existing research has ignored a critical feature in recommender systems: one user's feedback on the same item at different times is random. The stochastic rewards property essentially differs from that in classic RL scenarios with deterministic rewards, which makes RL-based recommender systems much more challenging. In this paper, we first demonstrate in a simulator environment where using direct stochastic feedback results in a significant drop in performance. Then to handle the stochastic feedback more efficiently, we design two stochastic reward stabilization frameworks that replace the direct stochastic feedback with that learned by a supervised model. Both frameworks are model-agnostic, i.e., they can effectively utilize various supervised models. We demonstrate the superiority of the proposed framework
    
[^6]: 优化关于相关性和后期公平性的群组公平Plackett-Luce排序模型

    Optimizing Group-Fair Plackett-Luce Ranking Models for Relevance and Ex-Post Fairness. (arXiv:2308.13242v1 [cs.LG])

    [http://arxiv.org/abs/2308.13242](http://arxiv.org/abs/2308.13242)

    提出了一种优化群组公平Plackett-Luce排序模型的方法，该方法最大化预期相关性并满足表示约束以确保后期公平性。

    

    在学习排名中，仅优化相关性（或预期排名效用）可能对某些类别的项目造成表现性损害。此外，如果相关性分数中存在隐性偏见，则学习排名模型可能无法优化真实的相关性。以前的研究提出了有效的算法来训练随机排名模型，以达到群组预期暴露的公平性（即期望值），但可能无法保证群组后期的表现公平性，即在从随机排序模型中实现排名之后。通常，通过后期处理实现后期公平性，但是以前的工作不训练意识到此后期处理的随机排序模型。在本文中，我们提出了一种新颖的目标函数，仅在满足给定表示约束的排名中最大化预期相关性，以确保后期公平性。基于最近关于后期群组公平排名的有效抽样器的工作，我们提出了一个新颖的目标函数，仅在满足给定表示约束的排名中最大化预期相关性，以确保后期公正性。建立在一个高效的抽样器的基础上，我们提出了一种新的目标函数，它最大化在满足给定的表示约束的排名中的预期相关性，以确保后期的公平性。

    In learning-to-rank (LTR), optimizing only the relevance (or the expected ranking utility) can cause representational harm to certain categories of items. Moreover, if there is implicit bias in the relevance scores, LTR models may fail to optimize for true relevance. Previous works have proposed efficient algorithms to train stochastic ranking models that achieve fairness of exposure to the groups ex-ante (or, in expectation), which may not guarantee representation fairness to the groups ex-post, that is, after realizing a ranking from the stochastic ranking model. Typically, ex-post fairness is achieved by post-processing, but previous work does not train stochastic ranking models that are aware of this post-processing.  In this paper, we propose a novel objective that maximizes expected relevance only over those rankings that satisfy given representation constraints to ensure ex-post fairness. Building upon recent work on an efficient sampler for ex-post group-fair rankings, we propo
    
[^7]: MMBAttn: 最大均值和逐位注意力用于CTR预测

    MMBAttn: Max-Mean and Bit-wise Attention for CTR Prediction. (arXiv:2308.13187v1 [cs.IR])

    [http://arxiv.org/abs/2308.13187](http://arxiv.org/abs/2308.13187)

    提出了一种利用最大均值和逐位注意力机制的CTR预测方法，可以准确估计特征的重要性，并通过考虑位级的细粒度交互，提高预测的精确性。

    

    随着在线广告和推荐系统中点击率（CTR）预测任务的复杂性和规模增加，准确估计特征的重要性成为开发有效模型的关键。本文提出了一种基于注意力的方法，利用最大值和平均值池化操作以及逐位注意力机制，在CTR预测中增强特征重要性估计。传统上，最大值和平均值池化等池化操作被广泛用于从特征中提取相关信息。然而，这些操作可能导致信息丢失，并妨碍准确确定特征的重要性。为了解决这个挑战，我们提出了一种新颖的注意力架构，利用逐位注意力结构强调特征中所有位之间的关系，同时进行最大池化和平均池化。通过考虑位级的细粒度交互，我们的方法旨在提高CTR预测的精确性。

    With the increasing complexity and scale of click-through rate (CTR) prediction tasks in online advertising and recommendation systems, accurately estimating the importance of features has become a critical aspect of developing effective models. In this paper, we propose an attention-based approach that leverages max and mean pooling operations, along with a bit-wise attention mechanism, to enhance feature importance estimation in CTR prediction. Traditionally, pooling operations such as max and mean pooling have been widely used to extract relevant information from features. However, these operations can lead to information loss and hinder the accurate determination of feature importance. To address this challenge, we propose a novel attention architecture that utilizes a bit-based attention structure that emphasizes the relationships between all bits in features, together with maximum and mean pooling. By considering the fine-grained interactions at the bit level, our method aims to 
    
[^8]: 多重BERT用于推荐系统中的嵌入

    Multi-BERT for Embeddings for Recommendation System. (arXiv:2308.13050v1 [cs.IR])

    [http://arxiv.org/abs/2308.13050](http://arxiv.org/abs/2308.13050)

    本文提出了一种使用多种最先进的自然语言处理模型来生成文档嵌入的方法，并在图书推荐任务中取得了比单一模型更好的性能。

    

    本文提出了一种新颖的方法，使用句子BERT（SBERT）和RoBERTa两种最先进的自然语言处理模型来生成文档嵌入。我们的方法将句子视为标记，并为它们生成嵌入，使模型能够捕捉文档内的句子内部和句子间关系。我们在图书推荐任务上评估了我们的模型，并展示了它在生成语义丰富和准确的文档嵌入方面的有效性。为了评估我们的方法的性能，我们在Goodreads数据集上进行了图书推荐任务的实验。我们将使用我们的MULTI-BERT模型生成的文档嵌入与仅使用SBERT生成的嵌入进行比较。我们使用精确度作为评估指标来比较生成嵌入的质量。结果表明，我们的模型在生成嵌入的质量方面始终优于SBERT。此外，我们发现...

    In this paper, we propose a novel approach for generating document embeddings using a combination of Sentence-BERT (SBERT) and RoBERTa, two state-of-the-art natural language processing models. Our approach treats sentences as tokens and generates embeddings for them, allowing the model to capture both intra-sentence and inter-sentence relations within a document. We evaluate our model on a book recommendation task and demonstrate its effectiveness in generating more semantically rich and accurate document embeddings. To assess the performance of our approach, we conducted experiments on a book recommendation task using the Goodreads dataset. We compared the document embeddings generated using our MULTI-BERT model to those generated using SBERT alone. We used precision as our evaluation metric to compare the quality of the generated embeddings. Our results showed that our model consistently outperformed SBERT in terms of the quality of the generated embeddings. Furthermore, we found tha
    
[^9]: 使用精细调整的Llama 2 GPT模型进行金融新闻分析

    Financial News Analytics Using Fine-Tuned Llama 2 GPT Model. (arXiv:2308.13032v1 [cs.CL])

    [http://arxiv.org/abs/2308.13032](http://arxiv.org/abs/2308.13032)

    本研究通过精细调整的Llama 2模型实现了金融新闻的多任务分析，包括文本分析、摘要和情感提取等。实验结果显示，提取的命名实体情感可以作为有监督机器学习模型的预测特征。

    

    本文考虑了使用精细调整的Llama 2 Large Language Model (LLM) 对金融新闻进行多任务分析的可能性。通过PEFT/LoRA方法对模型进行精细调整，主要包括从金融市场角度分析文本、突出文本的主要观点、对文本进行摘要和提取具有适当情感的命名实体等任务。实验结果表明，经过精细调整的Llama 2模型能够进行多任务的金融新闻分析，其响应的结构可以部分为结构化文本，另一部分数据可以采用JSON格式进一步处理。提取的命名实体情感可以被视为具有定量目标变量的监督机器学习模型的预测特征。

    The paper considers the possibility to fine-tune Llama 2 Large Language Model (LLM) for the multitask analysis of financial news. For fine-tuning, the PEFT/LoRA based approach was used. In the study, the model was fine-tuned for the following tasks: analysing a text from financial market perspectives, highlighting main points of a text, summarizing a text and extracting named entities with appropriate sentiments. The obtained results show that the fine-tuned Llama 2 model can perform a multitask financial news analysis with a specified structure of response, part of response can be a structured text and another part of data can have JSON format for further processing. Extracted sentiments for named entities can be considered as predictive features in supervised machine learning models with quantitative target variables.
    
[^10]: 用安排取代评分：一种用于学习排序的上下文集合到排列框架

    Replace Scoring with Arrangement: A Contextual Set-to-Arrangement Framework for Learning-to-Rank. (arXiv:2308.02860v1 [cs.IR])

    [http://arxiv.org/abs/2308.02860](http://arxiv.org/abs/2308.02860)

    这篇论文提出了一种新的学习排序框架，名为STARank，它通过直接生成候选项目排列来替代个别评分和排序操作，并且是端到端可微分的。

    

    学习排序是top-N推荐任务中的核心技术，理想的排名器应该是一个从项目集合到排列（即排列）的映射。现有的大多数解决方案属于概率排序原则（PRP）范式，即首先对候选集中的每个项目进行评分，然后执行排序操作以生成排名列表。然而，这些方法忽视了个体评分过程中候选项目之间的上下文依赖性，并且排序操作是不可微分的。为了解决上述问题，我们提出了一种名为STARank的集合到排列排序框架，它直接生成候选项目的排列，而不需要进行个别评分和排序操作，并且是端到端可微分的。因此，STARank可以在只有真实排列可访问但没有项目的真实相关度分数的情况下运行。为此，STARank首先阅读候选项目...

    Learning-to-rank is a core technique in the top-N recommendation task, where an ideal ranker would be a mapping from an item set to an arrangement (a.k.a. permutation). Most existing solutions fall in the paradigm of probabilistic ranking principle (PRP), i.e., first score each item in the candidate set and then perform a sort operation to generate the top ranking list. However, these approaches neglect the contextual dependence among candidate items during individual scoring, and the sort operation is non-differentiable. To bypass the above issues, we propose Set-To-Arrangement Ranking (STARank), a new framework directly generates the permutations of the candidate items without the need for individually scoring and sort operations; and is end-to-end differentiable. As a result, STARank can operate when only the ground-truth permutations are accessible without requiring access to the ground-truth relevance scores for items. For this purpose, STARank first reads the candidate items in t
    
[^11]: 基于图的推荐系统在社区检测中的增强

    Graph-Based Recommendation System Enhanced with Community Detection. (arXiv:2201.03622v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2201.03622](http://arxiv.org/abs/2201.03622)

    本文提出了一个基于图的推荐系统，利用数学和统计方法确定标签的相似性，包括词汇相似性和共现解决方案，并考虑了标签分配的时间，以提高推荐的准确性。

    

    许多研究者已经利用标签信息来改善推荐系统中推荐技术的性能。通过研究用户的标签，可以了解他们的兴趣，从而提高推荐的准确性。然而，由于用户自定义标签的任意性和缺乏限制，确定其确切含义和标签之间的相似性存在问题。本文利用数学和统计方法确定标签的词汇相似性和共现解决方案，以分配语义相似性。另外，考虑到用户兴趣随时间变化，本文还在共现标签中考虑了标签分配的时间以确定标签的相似性。然后，基于标签的相似性创建图形模型来建模用户的兴趣。

    Many researchers have used tag information to improve the performance of recommendation techniques in recommender systems. Examining the tags of users will help to get their interests and leads to more accuracy in the recommendations. Since user-defined tags are chosen freely and without any restrictions, problems arise in determining their exact meaning and the similarity of tags. However, using thesaurus and ontologies to find the meaning of tags is not very efficient due to their free definition by users and the use of different languages in many data sets. Therefore, this article uses mathematical and statistical methods to determine lexical similarity and co-occurrence tags solution to assign semantic similarity. On the other hand, due to the change of users' interests over time this article has considered the time of tag assignments in co-occurrence tags for determining similarity of tags. Then the graph is created based on similarity of tags. For modeling the interests of the us
    

