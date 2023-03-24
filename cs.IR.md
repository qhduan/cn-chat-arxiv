# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Why and When: Understanding System Initiative during Conversational Collaborative Search.](http://arxiv.org/abs/2303.13484) | 本文研究了如何支持通过即时通讯进行协作的“多个”搜索者，发现在此情况下搜索机器人采用了更多的对话水平倡议，该研究可以帮助未来会话搜索系统的设计。 |
| [^2] | [Modular Retrieval for Generalization and Interpretation.](http://arxiv.org/abs/2303.13419) | 调整模块化提示的检索模型REMOP，通过多个现有检索模块的组合解决新的检索任务，具有泛化和解释能力， 在零-shot检索基准测试中表现与最先进的检索模型相媲美。 |
| [^3] | [A Unified Framework for Learned Sparse Retrieval.](http://arxiv.org/abs/2303.13416) | 本文提出了一个LSR框架，以统一已有的LSR方法并分析了其关键组件。实验结果表明，在包括文档术语加权时，LSR方法的有效性和效率最优。 |
| [^4] | [GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering.](http://arxiv.org/abs/2303.13284) | 本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。 |
| [^5] | [Parameter-Efficient Sparse Retrievers and Rerankers using Adapters.](http://arxiv.org/abs/2303.13220) | 本文研究了在信息检索中使用适配器的效果，特别是在SPLADE这种稀疏检索器上。研究表明，适配器-SPLADE只需优化2%的训练参数，但在效率和效果方面均优于完全微调的方法。 |
| [^6] | [Limits of Predictability in Top-N Recommendation.](http://arxiv.org/abs/2303.13091) | 本研究探究了Top-N推荐中的可预测性极限，发现用户行为的可预测性限制了Top-N推荐算法的准确度，无论使用何种算法均无法突破其上限。 |
| [^7] | [Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation.](http://arxiv.org/abs/2303.12973) | 本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。 |
| [^8] | [Focusing on Potential Named Entities During Active Label Acquisition.](http://arxiv.org/abs/2111.03837) | 本文提出了几个AL句子查询评估函数，关注潜在正面标记，并使用更好的数据驱动的正常化方法，以最小化NER注释成本。 |

# 详细

[^1]: 为什么和何时：理解会话协作搜索中的系统倡议

    Why and When: Understanding System Initiative during Conversational Collaborative Search. (arXiv:2303.13484v1 [cs.HC])

    [http://arxiv.org/abs/2303.13484](http://arxiv.org/abs/2303.13484)

    本文研究了如何支持通过即时通讯进行协作的“多个”搜索者，发现在此情况下搜索机器人采用了更多的对话水平倡议，该研究可以帮助未来会话搜索系统的设计。

    

    在过去的十年中，会话搜索引起了相当多的关注。但是，大多数研究都集中在能够支持“单个”搜索者的系统上。在本文中，我们探讨了如何支持通过即时通讯平台（即 Slack）协作的“多个”搜索者的系统。我们展示了一项“奥兹巫师”研究，27对参与者在 Slack 上协作完成了三个信息寻求任务。参与者无法独立搜索，必须通过与从 Slack 通道直接交互的“搜索机器人”收集信息。搜索机器人的角色由参考图书馆员扮演。会话搜索系统必须能够参与“混合倡议”交互，通过控制对话并交还控制权以实现不同的目标。话语分析研究表明，会话代理可以采取两种倡议水平：对话水平和任务水平倡议。代理通过在多个选项中选择打开新的子话题或提示用户进行澄清来采取对话水平倡议。代理通过为该任务提供新的相关信息来采取任务水平倡议。基于我们的分析，我们发现，在与“多个”搜索者进行协作搜索的情况下，搜索机器人采用了更高比例的对话水平倡议。本文提供了有关搜索机器人如何在存在多个搜索者的情况下采取和放弃倡议以响应的见解，这可以为未来的会话搜索系统的设计提供指导。

    In the last decade, conversational search has attracted considerable attention. However, most research has focused on systems that can support a \emph{single} searcher. In this paper, we explore how systems can support \emph{multiple} searchers collaborating over an instant messaging platform (i.e., Slack). We present a ``Wizard of Oz'' study in which 27 participant pairs collaborated on three information-seeking tasks over Slack. Participants were unable to search on their own and had to gather information by interacting with a \emph{searchbot} directly from the Slack channel. The role of the searchbot was played by a reference librarian. Conversational search systems must be capable of engaging in \emph{mixed-initiative} interaction by taking and relinquishing control of the conversation to fulfill different objectives. Discourse analysis research suggests that conversational agents can take \emph{two} levels of initiative: dialog- and task-level initiative. Agents take dialog-level 
    
[^2]: 模块化检索：泛化和解释的出路

    Modular Retrieval for Generalization and Interpretation. (arXiv:2303.13419v1 [cs.IR])

    [http://arxiv.org/abs/2303.13419](http://arxiv.org/abs/2303.13419)

    调整模块化提示的检索模型REMOP，通过多个现有检索模块的组合解决新的检索任务，具有泛化和解释能力， 在零-shot检索基准测试中表现与最先进的检索模型相媲美。

    

    随着新的检索任务不断涌现，需要开发新的检索模型。但是，为每个新的检索任务实例化检索模型是耗费资源和耗时的，尤其是对于采用大规模预训练语言模型的检索模型而言。为解决这个问题，我们转向一种新的检索范式，称为模块化检索，它旨在通过组合多个现有的检索模块来解决新的检索任务。基于这个范式，我们提出了一个具有模块化提示调整的检索模型（REMOP）。它通过深度提示调整构建与任务属性相关联的检索模块，并通过模块化组合产生与任务相关的检索模型。我们验证，REMOP天生具有模块性不仅在初步探索中具有吸引人的泛化能力和可解释性，而且在零-shot检索基准测试中具有与最先进的检索模型相媲美的性能。

    New retrieval tasks have always been emerging, thus urging the development of new retrieval models. However, instantiating a retrieval model for each new retrieval task is resource-intensive and time-consuming, especially for a retrieval model that employs a large-scale pre-trained language model. To address this issue, we shift to a novel retrieval paradigm called modular retrieval, which aims to solve new retrieval tasks by instead composing multiple existing retrieval modules. Built upon the paradigm, we propose a retrieval model with modular prompt tuning named REMOP. It constructs retrieval modules subject to task attributes with deep prompt tuning, and yields retrieval models subject to tasks with module composition. We validate that, REMOP inherently with modularity not only has appealing generalizability and interpretability in preliminary explorations, but also achieves comparable performance to state-of-the-art retrieval models on a zero-shot retrieval benchmark.\footnote{Our
    
[^3]: 一种学习稀疏检索的统一框架

    A Unified Framework for Learned Sparse Retrieval. (arXiv:2303.13416v1 [cs.IR])

    [http://arxiv.org/abs/2303.13416](http://arxiv.org/abs/2303.13416)

    本文提出了一个LSR框架，以统一已有的LSR方法并分析了其关键组件。实验结果表明，在包括文档术语加权时，LSR方法的有效性和效率最优。

    

    学习稀疏检索(Learned sparse retrieval, LSR)是一组用于生成查询和文档的稀疏词汇表示以用于反向索引的第一阶段检索方法。最近引入了许多LSR方法，其中Splade模型在MSMarco上取得了最先进的性能。尽管这些LSR方法在模型架构上相似，但在有效性和效率方面存在实质性差异。使用不同的实验设置和配置使得难以比较这些方法并获得深入的见解。本文分析了现有的LSR方法，确定了关键组件，建立了一个统一的LSR框架，将所有LSR方法统一到一个框架下。我们使用相同的代码库重现了所有重要的方法，然后在相同的环境中重新训练它们，这使我们能够量化框架的组件如何影响有效性和效率。我们发现，(1)包括文档术语加权最为重要。

    Learned sparse retrieval (LSR) is a family of first-stage retrieval methods that are trained to generate sparse lexical representations of queries and documents for use with an inverted index. Many LSR methods have been recently introduced, with Splade models achieving state-of-the-art performance on MSMarco. Despite similarities in their model architectures, many LSR methods show substantial differences in effectiveness and efficiency. Differences in the experimental setups and configurations used make it difficult to compare the methods and derive insights. In this work, we analyze existing LSR methods and identify key components to establish an LSR framework that unifies all LSR methods under the same perspective. We then reproduce all prominent methods using a common codebase and re-train them in the same environment, which allows us to quantify how components of the framework affect effectiveness and efficiency. We find that (1) including document term weighting is most important 
    
[^4]: GETT-QA：基于图嵌入的知识图谱问答中的T2T Transformer

    GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering. (arXiv:2303.13284v1 [cs.CL])

    [http://arxiv.org/abs/2303.13284](http://arxiv.org/abs/2303.13284)

    本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。

    

    本文提出了一个名为GETT-QA的端到端知识图谱问答系统。GETT-QA使用了T5，这是一种热门的文本到文本预训练语言模型。该模型以自然语言形式的问题作为输入并生成所需SPARQL查询的简化形式。在简化形式中，模型不直接生成实体和关系ID，而是产生相应的实体和关系标签。标签在随后的步骤中与KG实体和关系ID联系起来。为了进一步改进结果，我们指导模型为每个实体生成KG嵌入的截断版本。截断的KG嵌入使得更精细的搜索从而更有效进行消歧。我们发现，T5能够在不改变损失函数的情况下学习截断的KG嵌入，提高了KGQA的性能。因此，我们在Wikidata的LC-QuAD 2.0和SimpleQuestions-Wikidata数据集上报告了端到端KGQA的强大结果。

    In this work, we present an end-to-end Knowledge Graph Question Answering (KGQA) system named GETT-QA. GETT-QA uses T5, a popular text-to-text pre-trained language model. The model takes a question in natural language as input and produces a simpler form of the intended SPARQL query. In the simpler form, the model does not directly produce entity and relation IDs. Instead, it produces corresponding entity and relation labels. The labels are grounded to KG entity and relation IDs in a subsequent step. To further improve the results, we instruct the model to produce a truncated version of the KG embedding for each entity. The truncated KG embedding enables a finer search for disambiguation purposes. We find that T5 is able to learn the truncated KG embeddings without any change of loss function, improving KGQA performance. As a result, we report strong results for LC-QuAD 2.0 and SimpleQuestions-Wikidata datasets on end-to-end KGQA over Wikidata.
    
[^5]: 使用适配器的参数高效稀疏检索器和重排器

    Parameter-Efficient Sparse Retrievers and Rerankers using Adapters. (arXiv:2303.13220v1 [cs.IR])

    [http://arxiv.org/abs/2303.13220](http://arxiv.org/abs/2303.13220)

    本文研究了在信息检索中使用适配器的效果，特别是在SPLADE这种稀疏检索器上。研究表明，适配器-SPLADE只需优化2%的训练参数，但在效率和效果方面均优于完全微调的方法。

    

    使用适配器的参数高效迁移学习已在自然语言处理（NLP）中研究作为完全微调的替代方法。适配器是内存高效的，并通过在变压器层之间添加小瓶颈层进行训练，同时保持大型预训练语言模型（PLMs）冻结来与下游任务良好地进行缩放。尽管在NLP中表现出有希望的结果，但这些方法在信息检索方面尚未得到充分探索。本文旨在完善适配器在IR中的使用情况。首先，我们研究了适配器对于SPLADE（一种稀疏检索器）的应用，适配器不仅保留了通过完全微调实现的效率和效果，而且内存高效，训练轻量级。我们观察到，适配器-SPLADE仅优化2％的训练参数，但胜过完全微调的对应物以及已有的最佳稀疏检索器。

    Parameter-Efficient transfer learning with Adapters have been studied in Natural Language Processing (NLP) as an alternative to full fine-tuning. Adapters are memory-efficient and scale well with downstream tasks by training small bottle-neck layers added between transformer layers while keeping the large pretrained language model (PLMs) frozen. In spite of showing promising results in NLP, these methods are under-explored in Information Retrieval. While previous studies have only experimented with dense retriever or in a cross lingual retrieval scenario, in this paper we aim to complete the picture on the use of adapters in IR. First, we study adapters for SPLADE, a sparse retriever, for which adapters not only retain the efficiency and effectiveness otherwise achieved by finetuning, but are memory-efficient and orders of magnitude lighter to train. We observe that Adapters-SPLADE not only optimizes just 2\% of training parameters, but outperforms fully fine-tuned counterpart and exis
    
[^6]: Top-N推荐中的可预测性极限

    Limits of Predictability in Top-N Recommendation. (arXiv:2303.13091v1 [cs.IR])

    [http://arxiv.org/abs/2303.13091](http://arxiv.org/abs/2303.13091)

    本研究探究了Top-N推荐中的可预测性极限，发现用户行为的可预测性限制了Top-N推荐算法的准确度，无论使用何种算法均无法突破其上限。

    

    Top-N推荐旨在从大量物品中向每个用户推荐一个小的N个物品集合，其准确性是评估推荐系统性能的最常用指标之一。尽管有大量的算法提出从用户的历史购买数据中学习用户偏好以提高Top-N准确性，但自然会有一个可预测性问题-是否存在这样一个Top-N准确性的上限。本研究通过研究特定用户行为数据集的规则程度来调查这种可预测性。同时量化Top-N推荐的可预测性需要同时量化N个具有最高概率的行为的准确度的限制，从而大大增加了问题的难度。为了实现这一目的，我们首先挖掘了N个具有最高概率的行为之间的关联，并基于信息论描述了用户行为分布。然后，我们采用用户行为数据的可压缩性作为可预测性的度量，并分析其对Top-N推荐准确度的影响。实验结果表明，用户行为的可预测性限制了Top-N推荐算法的准确度，存在着一种上限的准确度，任何算法都无法超越。

    Top-N recommendation aims to recommend each consumer a small set of N items from a large collection of items, and its accuracy is one of the most common indexes to evaluate the performance of a recommendation system. While a large number of algorithms are proposed to push the Top-N accuracy by learning the user preference from their history purchase data, a predictability question is naturally raised - whether there is an upper limit of such Top-N accuracy. This work investigates such predictability by studying the degree of regularity from a specific set of user behavior data. Quantifying the predictability of Top-N recommendations requires simultaneously quantifying the limits on the accuracy of the N behaviors with the highest probability. This greatly increases the difficulty of the problem. To achieve this, we firstly excavate the associations among N behaviors with the highest probability and describe the user behavior distribution based on the information theory. Then, we adopt 
    
[^7]: 推荐系统中反事实倾向估计的不确定性校准

    Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation. (arXiv:2303.12973v1 [cs.AI])

    [http://arxiv.org/abs/2303.12973](http://arxiv.org/abs/2303.12973)

    本文提出了多种不确定性校准技术，以改进推荐系统中倾向性估计的效果。经过实验验证，校准后的IPS估计器在Coat和yahoo数据集上表现更好。

    

    在推荐系统中，由于选择偏差，许多评分信息都丢失了，这被称为非随机缺失。反事实逆倾向评分（IPS）被用于衡量每个观察到的评分的填充错误。虽然在多种情况下有效，但我们认为IPS估计的性能受到倾向性估计不确定性的限制。本文提出了多种代表性的不确定性校准技术，以改进推荐系统中倾向性估计的不确定性校准。通过对偏误和推广界限的理论分析表明，经过校准的IPS估计器优于未校准的IPS估计器。 Coat和yahoo数据集上的实验结果表明，不确定性校准得到改进，从而使推荐结果更好。

    In recommendation systems, a large portion of the ratings are missing due to the selection biases, which is known as Missing Not At Random. The counterfactual inverse propensity scoring (IPS) was used to weight the imputation error of every observed rating. Although effective in multiple scenarios, we argue that the performance of IPS estimation is limited due to the uncertainty miscalibration of propensity estimation. In this paper, we propose the uncertainty calibration for the propensity estimation in recommendation systems with multiple representative uncertainty calibration techniques. Theoretical analysis on the bias and generalization bound shows the superiority of the calibrated IPS estimator over the uncalibrated one. Experimental results on the coat and yahoo datasets shows that the uncertainty calibration is improved and hence brings the better recommendation results.
    
[^8]: 集中关注潜在命名实体的主动标注获取

    Focusing on Potential Named Entities During Active Label Acquisition. (arXiv:2111.03837v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2111.03837](http://arxiv.org/abs/2111.03837)

    本文提出了几个AL句子查询评估函数，关注潜在正面标记，并使用更好的数据驱动的正常化方法，以最小化NER注释成本。

    

    命名实体识别(NER)旨在识别结构化文本中命名实体的提及并将其分类到预定义的命名实体类别中。虽然基于深度学习的预训练语言模型有助于在NER中实现良好的预测性能，但许多特定领域的NER应用仍需要大量标记数据。主动学习(AL)是解决标签获取问题的通用框架，已用于NER任务，以最小化注释成本而不牺牲模型性能。然而，标记的严重不均匀类分布引入了设计有效的NER主动学习查询方法的挑战。我们提出了几个AL句子查询评估函数，更多关注潜在的正面标记，并使用基于句子和标记成本评估策略来评估这些提议的函数。我们还提出了更好的数据驱动的正常化方法，以惩罚过长或过短的句子。

    Named entity recognition (NER) aims to identify mentions of named entities in an unstructured text and classify them into predefined named entity classes. While deep learning-based pre-trained language models help to achieve good predictive performances in NER, many domain-specific NER applications still call for a substantial amount of labeled data. Active learning (AL), a general framework for the label acquisition problem, has been used for NER tasks to minimize the annotation cost without sacrificing model performance. However, the heavily imbalanced class distribution of tokens introduces challenges in designing effective AL querying methods for NER. We propose several AL sentence query evaluation functions that pay more attention to potential positive tokens, and evaluate these proposed functions with both sentence-based and token-based cost evaluation strategies. We also propose a better data-driven normalization approach to penalize sentences that are too long or too short. Our
    

