# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual Evaluation of Peer-Review Assignment Policies.](http://arxiv.org/abs/2305.17339) | 本研究探讨了同行评审分配算法的反事实评估方法，利用引入随机性的新政策，来评估同行评审分配的变化对于评审质量的影响。 |
| [^2] | [DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication.](http://arxiv.org/abs/2305.17310) | DotHash是一种用于集合相似度度量的无偏估计器，可用于链接预测和文档去重等应用。 |
| [^3] | [Text Is All You Need: Learning Language Representations for Sequential Recommendation.](http://arxiv.org/abs/2305.13731) | 本研究提出了一种名为Recformer的框架，它将用户喜好和项目特征建模为可以推广到新项目和数据集的语言表示，并利用双向Transformer来捕捉长期依赖关系，用于序列推荐，比目前最先进的方法表现更好。 |
| [^4] | [InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval.](http://arxiv.org/abs/2301.01820) | 本文提出 InPars-v2，使用开源 LLMs 和强大再排序器生成用于信息检索中训练的合成查询-文档对，可在 BEIR 基准测试中达到最新的最好结果。 |
| [^5] | [Reasoning with Language Model Prompting: A Survey.](http://arxiv.org/abs/2212.09597) | 本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。 |
| [^6] | [Nonparametric Decoding for Generative Retrieval.](http://arxiv.org/abs/2210.02068) | 本文提出了一种非参数化解码方法，通过利用上下文化词汇嵌入，解决了生成式检索模型信息容量受限的问题，在文档检索任务中具有高效性和高性能。 |
| [^7] | [Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model.](http://arxiv.org/abs/2208.06164) | 本论文提出了一个联合优化排名和校准能力的方法JRC，通过对比输出logit值来提高排名能力和校准能力。 |
| [^8] | [Optimizing Test-Time Query Representations for Dense Retrieval.](http://arxiv.org/abs/2205.12680) | 本文介绍了TOUR算法，它利用交叉编码再排序器提供的伪标签优化基于实例级别的查询表示，显著提高了端到端开放领域问答的准确性。 |
| [^9] | [Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System.](http://arxiv.org/abs/2204.11970) | 本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。 |

# 详细

[^1]: 同行评审分配政策的反事实评估

    Counterfactual Evaluation of Peer-Review Assignment Policies. (arXiv:2305.17339v1 [cs.IR])

    [http://arxiv.org/abs/2305.17339](http://arxiv.org/abs/2305.17339)

    本研究探讨了同行评审分配算法的反事实评估方法，利用引入随机性的新政策，来评估同行评审分配的变化对于评审质量的影响。

    

    同行评审分配算法旨在将研究论文分配给适当的专家评审人员，以最大程度地提高评审质量。有效的分配策略的主要挑战在于评估分配算法的更改如何映射到评审质量的变化。在本研究中，我们利用新近提出的政策，引入同行评审分配中的随机性——以缓解欺诈现象——作为评估反事实分配政策的宝贵机会。具体而言，我们利用这些随机分配所提供的积极概率来观察多种感兴趣的分配政策的评审结果。为了解决应用标准离策略评估方法所面临的挑战，例如违反积极性，我们引入了基于单调性和Lipschitz平滑性假设的部分识别方法，用于评估评审人员-论文协变量和结果之间的映射。我们将我们的方法应用于两种计算机同行评审数据。

    Peer review assignment algorithms aim to match research papers to suitable expert reviewers, working to maximize the quality of the resulting reviews. A key challenge in designing effective assignment policies is evaluating how changes to the assignment algorithm map to changes in review quality. In this work, we leverage recently proposed policies that introduce randomness in peer-review assignment--in order to mitigate fraud--as a valuable opportunity to evaluate counterfactual assignment policies. Specifically, we exploit how such randomized assignments provide a positive probability of observing the reviews of many assignment policies of interest. To address challenges in applying standard off-policy evaluation methods, such as violations of positivity, we introduce novel methods for partial identification based on monotonicity and Lipschitz smoothness assumptions for the mapping between reviewer-paper covariates and outcomes. We apply our methods to peer-review data from two compu
    
[^2]: DotHash：用于链接预测和文档去重的集合相似度度量估计

    DotHash: Estimating Set Similarity Metrics for Link Prediction and Document Deduplication. (arXiv:2305.17310v1 [cs.SI])

    [http://arxiv.org/abs/2305.17310](http://arxiv.org/abs/2305.17310)

    DotHash是一种用于集合相似度度量的无偏估计器，可用于链接预测和文档去重等应用。

    

    集合相似度的度量是几种数据挖掘任务的核心方面。例如，在网页搜索中删除重复结果的常见方法是查看所有页面对之间的Jaccard指数。在社交网络分析中，一种备受赞赏的度量是Adamic-Adar指数，广泛用于比较预测链接中的节点邻域集合。然而，随着要处理的数据量增加，计算所有成对之间的确切相似度可能是棘手的。这种规模上的挑战已经激发了对于集合相似度度量的有效估计器的研究。最流行的两种估计器，MinHash和SimHash，的确被用于需要处理大量数据的应用程序，如文档去重和推荐系统。考虑到这些任务的重要性，推进估计器的需求是显然的。我们提出了DotHash，这是两个集合交集大小的无偏估计器。

    Metrics for set similarity are a core aspect of several data mining tasks. To remove duplicate results in a Web search, for example, a common approach looks at the Jaccard index between all pairs of pages. In social network analysis, a much-celebrated metric is the Adamic-Adar index, widely used to compare node neighborhood sets in the important problem of predicting links. However, with the increasing amount of data to be processed, calculating the exact similarity between all pairs can be intractable. The challenge of working at this scale has motivated research into efficient estimators for set similarity metrics. The two most popular estimators, MinHash and SimHash, are indeed used in applications such as document deduplication and recommender systems where large volumes of data need to be processed. Given the importance of these tasks, the demand for advancing estimators is evident. We propose DotHash, an unbiased estimator for the intersection size of two sets. DotHash can be use
    
[^3]: 文本是唯一需要的：用于序列推荐的语言表示学习

    Text Is All You Need: Learning Language Representations for Sequential Recommendation. (arXiv:2305.13731v1 [cs.IR])

    [http://arxiv.org/abs/2305.13731](http://arxiv.org/abs/2305.13731)

    本研究提出了一种名为Recformer的框架，它将用户喜好和项目特征建模为可以推广到新项目和数据集的语言表示，并利用双向Transformer来捕捉长期依赖关系，用于序列推荐，比目前最先进的方法表现更好。

    

    序列推荐旨在从历史交互中建模动态用户行为。现有方法依靠明确的项目ID或一般文本特征进行序列建模，以理解用户喜好。尽管很有前途，但这些方法仍然难以建模冷启动项目或将知识转移至新数据集。在本文中，我们建议将用户喜好和项目特征建模为可以推广到新项目和数据集的语言表示。为此，我们提出了一个名为Recformer的新框架，它有效地学习序列推荐的语言表示。具体而言，我们建议通过展平由文本描述的项目键值属性，将项目作为“句子”（单词序列）来编写，以便用户的项目序列成为句子序列。为推荐，Recformer被训练以理解“句子”序列并检索下一个“句子”。为了编码项目序列，我们设计了一个双向Transformer，利用自我注意机制来捕捉长期依赖关系。我们在三个具有不同特征的公共数据集上进行了广泛的实验，并展示了Recformer始终优于现有最先进的方法。

    Sequential recommendation aims to model dynamic user behavior from historical interactions. Existing methods rely on either explicit item IDs or general textual features for sequence modeling to understand user preferences. While promising, these approaches still struggle to model cold-start items or transfer knowledge to new datasets. In this paper, we propose to model user preferences and item features as language representations that can be generalized to new items and datasets. To this end, we present a novel framework, named Recformer, which effectively learns language representations for sequential recommendation. Specifically, we propose to formulate an item as a "sentence" (word sequence) by flattening item key-value attributes described by text so that an item sequence for a user becomes a sequence of sentences. For recommendation, Recformer is trained to understand the "sentence" sequence and retrieve the next "sentence". To encode item sequences, we design a bi-directional T
    
[^4]: InPars-v2: 利用大型语言模型作为信息检索高效数据集生成器

    InPars-v2: Large Language Models as Efficient Dataset Generators for Information Retrieval. (arXiv:2301.01820v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2301.01820](http://arxiv.org/abs/2301.01820)

    本文提出 InPars-v2，使用开源 LLMs 和强大再排序器生成用于信息检索中训练的合成查询-文档对，可在 BEIR 基准测试中达到最新的最好结果。

    

    近来，InPars 提出了一种利用大型语言模型（LLMs）在信息检索任务中高效生成相关查询的方法：通过少量样本，诱导 LLM 生成与文档相关的查询，在此基础上生成合成的查询-文档对，用于训练检索器。然而，InPars 和 Promptagator 等方法依赖于 GPT-3 和 FLAN 等专有 LLMs 生成这些数据集。本文提出了 InPars-v2，该数据集生成器使用开放源代码的 LLM 和现有的强大再排序器来选择用于训练的合成查询-文档对。一个简单的 BM25 检索管道，在经过由 InPars-v2 数据微调的 monoT5 再排序器之后，便可在 BEIR 基准测试中达到最新的最好结果。为了让研究人员进一步提高我们的方法，我们开源了代码、数据和微调模型：https://github.com/zetaalphavector/inPars/tree/master/tpu。

    Recently, InPars introduced a method to efficiently use large language models (LLMs) in information retrieval tasks: via few-shot examples, an LLM is induced to generate relevant queries for documents. These synthetic query-document pairs can then be used to train a retriever. However, InPars and, more recently, Promptagator, rely on proprietary LLMs such as GPT-3 and FLAN to generate such datasets. In this work we introduce InPars-v2, a dataset generator that uses open-source LLMs and existing powerful rerankers to select synthetic query-document pairs for training. A simple BM25 retrieval pipeline followed by a monoT5 reranker finetuned on InPars-v2 data achieves new state-of-the-art results on the BEIR benchmark. To allow researchers to further improve our method, we open source the code, synthetic data, and finetuned models: https://github.com/zetaalphavector/inPars/tree/master/tpu
    
[^5]: 使用语言模型提示进行推理：一项调查

    Reasoning with Language Model Prompting: A Survey. (arXiv:2212.09597v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09597](http://arxiv.org/abs/2212.09597)

    本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。

    

    推理作为复杂问题解决的重要能力，可以为医疗诊断、谈判等各种实际应用提供后端支持。本文对使用语言模型提示进行推理的前沿研究进行了综合调查。我们介绍了研究成果的比较和总结，并提供了系统资源以帮助初学者。我们还讨论了新兴推理能力出现的潜在原因，并突出了未来的研究方向。资源可在 https://github.com/zjunlp/Prompt4ReasoningPapers 上获取（定期更新）。

    Reasoning, as an essential ability for complex problem-solving, can provide back-end support for various real-world applications, such as medical diagnosis, negotiation, etc. This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting. We introduce research works with comparisons and summaries and provide systematic resources to help beginners. We also discuss the potential reasons for emerging such reasoning abilities and highlight future research directions. Resources are available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated periodically).
    
[^6]: 生成式检索的非参数化解码方法

    Nonparametric Decoding for Generative Retrieval. (arXiv:2210.02068v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.02068](http://arxiv.org/abs/2210.02068)

    本文提出了一种非参数化解码方法，通过利用上下文化词汇嵌入，解决了生成式检索模型信息容量受限的问题，在文档检索任务中具有高效性和高性能。

    

    生成式检索模型仅依赖于其模型参数中编码的信息，没有外部存储器，其信息容量受到限制并且是固定的。为了克服这一限制，我们提出了一种非参数化解码方法（Np Decoding），可以应用于现有的生成式检索模型中。Np Decoding使用非参数化的上下文化词汇嵌入（外部存储器），而不是作为解码器词汇嵌入的常规词汇嵌入。通过利用上下文化词汇嵌入，生成式检索模型能够同时利用参数空间和非参数空间。在9个数据集（8个单跳和1个多跳）的文档检索任务中的评估表明，将Np Decoding应用于生成式检索模型可以显著提高性能。我们还表明，Np Decoding具有数据和参数效率，并在零样本设置中表现出高性能。

    The generative retrieval model depends solely on the information encoded in its model parameters without external memory, its information capacity is limited and fixed. To overcome the limitation, we propose Nonparametric Decoding (Np Decoding) which can be applied to existing generative retrieval models. Np Decoding uses nonparametric contextualized vocab embeddings (external memory) rather than vanilla vocab embeddings as decoder vocab embeddings. By leveraging the contextualized vocab embeddings, the generative retrieval model is able to utilize both the parametric and nonparametric space. Evaluation over 9 datasets (8 single-hop and 1 multi-hop) in the document retrieval task shows that applying Np Decoding to generative retrieval models significantly improves the performance. We also show that Np Decoding is dataand parameter-efficient, and shows high performance in the zero-shot setting.
    
[^7]: 融合背景的混合模型下的排名和校准联合优化研究

    Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model. (arXiv:2208.06164v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.06164](http://arxiv.org/abs/2208.06164)

    本论文提出了一个联合优化排名和校准能力的方法JRC，通过对比输出logit值来提高排名能力和校准能力。

    

    尽管排名优化技术得到了发展，点对点损失仍然是点击率预测的主要方法。这可以归因于点对点损失的校准能力，因为预测可以被视为点击概率。实际上，CTR预测模型也通常通过排名能力进行评估。为了优化排名能力，可以采用排名损失（例如成对或列表损失），因为它们通常比点对点损失实现更好的排名。之前的研究尝试直接将两种损失组合起来以获得两种损失的益处，并观察到了改进的性能。然而，之前的研究打破了输出logit作为点击率的含义，这可能会导致次优解。为了解决这个问题，我们提出了一种方法，可以联合优化排名和校准能力（简称JRC）。JRC通过对输出logit值进行对比来提高排名能力。

    Despite the development of ranking optimization techniques, pointwise loss remains the dominating approach for click-through rate prediction. It can be attributed to the calibration ability of the pointwise loss since the prediction can be viewed as the click probability. In practice, a CTR prediction model is also commonly assessed with the ranking ability. To optimize the ranking ability, ranking loss (e.g., pairwise or listwise loss) can be adopted as they usually achieve better rankings than pointwise loss. Previous studies have experimented with a direct combination of the two losses to obtain the benefit from both losses and observed an improved performance. However, previous studies break the meaning of output logit as the click-through rate, which may lead to sub-optimal solutions. To address this issue, we propose an approach that can Jointly optimize the Ranking and Calibration abilities (JRC for short). JRC improves the ranking ability by contrasting the logit value for the 
    
[^8]: 优化密集检索的测试时间查询表示

    Optimizing Test-Time Query Representations for Dense Retrieval. (arXiv:2205.12680v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12680](http://arxiv.org/abs/2205.12680)

    本文介绍了TOUR算法，它利用交叉编码再排序器提供的伪标签优化基于实例级别的查询表示，显著提高了端到端开放领域问答的准确性。

    

    最近，密集检索的发展依赖于预训练的查询和上下文编码器提供的质量表示查询和上下文。本文介绍了TOUR（Test-Time Optimization of Query Representations），它通过来自测试时间检索结果的信号进一步优化基于实例级别的查询表示。我们利用交叉编码器再排序器来为检索结果提供细粒度的伪标签，并通过梯度下降迭代地优化查询表示。我们的理论分析表明，TOUR可以看作是伪相关反馈的经典Rocchio算法的一种推广，并提出了两种利用伪标签作为硬二进制或软连续标签的变体。我们首先将TOUR应用于短语检索，并使用我们提出的短语再排序器评估其在通道检索上的有效性。TOUR极大地提高了端到端开放领域问答的准确性。

    Recent developments of dense retrieval rely on quality representations of queries and contexts from pre-trained query and context encoders. In this paper, we introduce TOUR (Test-Time Optimization of Query Representations), which further optimizes instance-level query representations guided by signals from test-time retrieval results. We leverage a cross-encoder re-ranker to provide fine-grained pseudo labels over retrieval results and iteratively optimize query representations with gradient descent. Our theoretical analysis reveals that TOUR can be viewed as a generalization of the classical Rocchio algorithm for pseudo relevance feedback, and we present two variants that leverage pseudo-labels as hard binary or soft continuous labels. We first apply TOUR on phrase retrieval with our proposed phrase re-ranker, and also evaluate its effectiveness on passage retrieval with an off-the-shelf re-ranker. TOUR greatly improves end-to-end open-domain question answering accuracy, as well as pa
    
[^9]: 基于机器学习的多阶段系统对真实患者数据进行视力预测

    Visual Acuity Prediction on Real-Life Patient Data Using a Machine Learning Based Multistage System. (arXiv:2204.11970v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2204.11970](http://arxiv.org/abs/2204.11970)

    本研究提供了一种使用机器学习技术开发预测模型的多阶段系统，可高精度预测三种眼疾患者的视力变化，并辅助眼科医生进行临床决策和患者咨询。

    

    现实生活中，眼科学中的玻璃体手术药物治疗是治疗年龄相关性黄斑变性（AMD）、糖尿病性黄斑水肿（DME）和视网膜静脉阻塞（RVO）相关疾病的一种普遍治疗方法。然而，在真实世界的情况下，由于数据的异质性和不完整性，患者往往会在多年时间内失去视力，尽管接受治疗。本文采用多种IT系统，提出了一种用于研究的数据集成流程，该流程融合了德国一家最佳医疗保健医院的眼科部门的不同IT系统。经过使用机器学习技术开发预测模型，我们实现了对患者视力的预测。我们的结果表明，我们的系统可以为三种疾病的预测提供高准确性。此外，我们还展示了我们的系统可以作为工具，辅助眼科医生进行临床决策和患者咨询。

    In ophthalmology, intravitreal operative medication therapy (IVOM) is a widespread treatment for diseases related to the age-related macular degeneration (AMD), the diabetic macular edema (DME), as well as the retinal vein occlusion (RVO). However, in real-world settings, patients often suffer from loss of vision on time scales of years despite therapy, whereas the prediction of the visual acuity (VA) and the earliest possible detection of deterioration under real-life conditions is challenging due to heterogeneous and incomplete data. In this contribution, we present a workflow for the development of a research-compatible data corpus fusing different IT systems of the department of ophthalmology of a German maximum care hospital. The extensive data corpus allows predictive statements of the expected progression of a patient and his or her VA in each of the three diseases. We found out for the disease AMD a significant deterioration of the visual acuity over time. Within our proposed m
    

