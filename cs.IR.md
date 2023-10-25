# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topology-aware Debiased Self-supervised Graph Learning for Recommendation.](http://arxiv.org/abs/2310.15858) | 面向推荐的拓扑感知去偏向自监督图学习（TDSGL）通过构建对比对，考虑用户（物品）的语义相似性，解决了推荐系统中负采样策略导致的错误负样本和忽略正样本的问题。 |
| [^2] | [A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction.](http://arxiv.org/abs/2310.15790) | 我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。 |
| [^3] | [TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction.](http://arxiv.org/abs/2310.15556) | TCRA-LLM是通过概述压缩和语义压缩两种方法来减少商业大型语言模型推理成本的方案。 |
| [^4] | [KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval.](http://arxiv.org/abs/2310.15511) | 本研究评估了最先进的模型在信息检索中回答约束满足查询的能力，并引入了一个新的数据集KITAB来衡量语言模型的约束满足能力。 |
| [^5] | [Robust Representation Learning for Unified Online Top-K Recommendation.](http://arxiv.org/abs/2310.15492) | 本论文提出一种鲁棒的表示学习方法，用于统一在线Top-K推荐，在大规模工业电子商务中解决物品广告和内容广告的不一致性，以及跨不同领域的多实体广告的检索问题。 |
| [^6] | [Off-Policy Evaluation for Large Action Spaces via Policy Convolution.](http://arxiv.org/abs/2310.15433) | 本研究提出了一种名为策略卷积（PC）的离策略估计方法，该方法通过动作嵌入来解决大动作空间下的分布转移问题，可以在偏差和方差之间进行权衡 |
| [^7] | [Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network.](http://arxiv.org/abs/2310.15342) | 本论文提出了一种针对深度稀疏网络的混合粒度特征交互选择方法，能够同时考虑特征域和特征值，实验证明该方法在准确性和效率方面表现良好。 |
| [^8] | [Triple Simplex Matrix Completion for Expense Forecasting.](http://arxiv.org/abs/2310.15275) | 本文提出了一种使用三重单纯形矩阵完成方法进行费用预测的模型。该模型通过学习项目与潜在空间中的费用模式相关性来预测费用，同时满足预算约束并保证预测结果的准确性。 |
| [^9] | [CorefPrompt: Prompt-based Event Coreference Resolution by Measuring Event Type and Argument Compatibilities.](http://arxiv.org/abs/2310.14512) | CorefPrompt是一种基于提示的方法，通过测量事件类型和参数的兼容性来进行事件指代消解。该方法将事件指代消解转化为一个填空式MLM任务，并通过引入辅助的提示任务来帮助模型进行推理，最终在基准测试中取得了良好的表现。 |
| [^10] | [AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking.](http://arxiv.org/abs/2310.09706) | 在用户建模中，通过自适应增强自监督排序方法预训练用户模型，解决了数据稀疏性问题和现有增强方法引入的噪音问题。 |
| [^11] | [Beyond Semantics: Learning a Behavior Augmented Relevance Model with Self-supervised Learning.](http://arxiv.org/abs/2308.05379) | 这篇论文提出了一种行为增强的相关模型，利用自我监督学习，通过从用户历史行为数据中提取辅助查询-项目交互，来改进搜索引擎中的查询-项目匹配，提高准确性和鲁棒性。 |
| [^12] | [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models.](http://arxiv.org/abs/2306.10933) | 本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。 |
| [^13] | [Evaluating Verifiability in Generative Search Engines.](http://arxiv.org/abs/2304.09848) | 本文评估了四个流行生成式搜索引擎的可验证性，发现现有生成式搜索引擎响应流畅但仅有51.5%的生成句子得到了完整的引用支持，仅有74.5%的引用支持其相关语句。 |
| [^14] | [Content-Based Search for Deep Generative Models.](http://arxiv.org/abs/2210.03116) | 这个论文介绍了基于内容的深层生成模型搜索任务，通过优化问题选择生成与查询最相似内容概率最高的模型，并提出了适用于不同查询模态的对比学习框架。（翻译为中文） |

# 详细

[^1]: 面向推荐的拓扑感知去偏向自监督图学习

    Topology-aware Debiased Self-supervised Graph Learning for Recommendation. (arXiv:2310.15858v1 [cs.IR])

    [http://arxiv.org/abs/2310.15858](http://arxiv.org/abs/2310.15858)

    面向推荐的拓扑感知去偏向自监督图学习（TDSGL）通过构建对比对，考虑用户（物品）的语义相似性，解决了推荐系统中负采样策略导致的错误负样本和忽略正样本的问题。

    

    在推荐系统中，基于图的协同过滤方法通过引入图对比学习来缓解数据稀疏性。然而，这些基于图对比学习的协同过滤模型中的随机负采样策略忽视了用户（物品）的语义结构，这不仅引入了错误的负样本（与锚定用户（物品）相似的负样本），还忽略了潜在的正样本。为了解决上述问题，我们提出了面向推荐的拓扑感知去偏向自监督图学习（TDSGL），根据用户（物品）之间的语义相似性构建对比对。具体而言，由于原始的用户-物品交互数据很好地反映了用户的购买意图和物品的某些特征，我们在交互数据上计算用户（物品）之间的语义相似性。然后，给定一个用户（物品），我们通过选择嵌入不同语义结构的用户（物品）来构建其负样本对，以确保去偏向性。

    In recommendation, graph-based Collaborative Filtering (CF) methods mitigate the data sparsity by introducing Graph Contrastive Learning (GCL). However, the random negative sampling strategy in these GCL-based CF models neglects the semantic structure of users (items), which not only introduces false negatives (negatives that are similar to anchor user (item)) but also ignores the potential positive samples. To tackle the above issues, we propose Topology-aware Debiased Self-supervised Graph Learning (TDSGL) for recommendation, which constructs contrastive pairs according to the semantic similarity between users (items). Specifically, since the original user-item interaction data commendably reflects the purchasing intent of users and certain characteristics of items, we calculate the semantic similarity between users (items) on interaction data. Then, given a user (item), we construct its negative pairs by selecting users (items) which embed different semantic structures to ensure the
    
[^2]: 一种用于测量专业术语抽取中术语爆发性的统计显著性测试方法

    A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction. (arXiv:2310.15790v1 [cs.IR])

    [http://arxiv.org/abs/2310.15790](http://arxiv.org/abs/2310.15790)

    我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。

    

    专业术语抽取是文本分析中的重要任务。当语料库中一个术语的出现集中在少数几个文件中时，可称之为“爆发性”。作为内容丰富的术语，爆发性术语非常适合用于主题描述，并且是技术术语的自然候选词。文献中提出了多种术语爆发性的测量方法。然而，在文本分析中，包括与术语爆发性相关的统计显著性测试范式尚未得到充分探索。为了探索这个领域，我们的主要贡献是提出了一种基于多项式语言模型的术语爆发性统计显著性的精确测试方法。由于计算成本过高，我们还提出了一个启发式公式，用于近似测试P值。作为补充的理论贡献，我们推导了一种未经报道的逆文档频率与逆收集频率的关系。

    Domain-specific terminology extraction is an important task in text analysis. A term in a corpus is said to be "bursty" when its occurrences are concentrated in few out of many documents. Being content rich, bursty terms are highly suited for subject matter characterization, and serve as natural candidates for identifying with technical terminology. Multiple measures of term burstiness have been proposed in the literature. However, the statistical significance testing paradigm has remained underexplored in text analysis, including in relation to term burstiness. To test these waters, we propose as our main contribution a multinomial language model-based exact test of statistical significance for term burstiness. Due to its prohibitive computational cost, we advance a heuristic formula designed to serve as a proxy for test P-values. As a complementary theoretical contribution, we derive a previously unreported relationship connecting the inverse document frequency and inverse collection
    
[^3]: TCRA-LLM: 用于减少推理成本的令牌压缩检索增强大型语言模型

    TCRA-LLM: Token Compression Retrieval Augmented Large Language Model for Inference Cost Reduction. (arXiv:2310.15556v1 [cs.CL])

    [http://arxiv.org/abs/2310.15556](http://arxiv.org/abs/2310.15556)

    TCRA-LLM是通过概述压缩和语义压缩两种方法来减少商业大型语言模型推理成本的方案。

    

    自从ChatGPT发布了API供公众使用以来，构建在商业大型语言模型（LLM）之上的应用程序数量呈指数增长。这种模型的一个流行用法是利用其上下文学习能力并生成响应以回答用户查询，并利用检索增强获得的知识。部署商业检索增强型LLM的一个问题是成本，因为额外检索的上下文大大增加了LLM的输入标记量。为了缓解这个问题，我们提出了一种令牌压缩方案，包括两种方法：概述压缩和语义压缩。第一种方法使用基于T5模型，通过使用包含具有不同长度的样本的自指示数据集进行微调，并通过概述来减少令牌大小。第二种方法通过移除对语义影响较小的词来进一步压缩令牌大小。为了充分评估所提方法的有效性，

    Since ChatGPT released its API for public use, the number of applications built on top of commercial large language models (LLMs) increase exponentially. One popular usage of such models is leveraging its in-context learning ability and generating responses given user queries leveraging knowledge obtained by retrieval augmentation. One problem of deploying commercial retrieval-augmented LLMs is the cost due to the additionally retrieved context that largely increases the input token size of the LLMs. To mitigate this, we propose a token compression scheme that includes two methods: summarization compression and semantic compression. The first method applies a T5-based model that is fine-tuned by datasets generated using self-instruct containing samples with varying lengths and reduce token size by doing summarization. The second method further compresses the token size by removing words with lower impact on the semantic. In order to adequately evaluate the effectiveness of the proposed
    
[^4]: 在信息检索中评估基于约束满足的LLMs

    KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval. (arXiv:2310.15511v1 [cs.LG])

    [http://arxiv.org/abs/2310.15511](http://arxiv.org/abs/2310.15511)

    本研究评估了最先进的模型在信息检索中回答约束满足查询的能力，并引入了一个新的数据集KITAB来衡量语言模型的约束满足能力。

    

    我们研究了最先进的模型在信息检索中回答约束满足查询（例如，“圣地亚哥的冰淇淋店列表”）的能力。过去，这样的查询被认为只能通过网络搜索或知识库来解决。最近，大型语言模型（LLMs）在这个任务中展示了初步的能力。然而，许多当前的检索基准要么已饱和，要么不能衡量约束满足。受到对LLMs事实不正确和产生幻觉的日益关注的驱动，我们提出了KITAB，一个用于衡量语言模型约束满足能力的新数据集。KITAB包含600多位作者和13,000个查询的与书籍相关的数据，还提供了一个关联的动态数据收集和约束验证方法，以获得其他作者的类似测试数据。我们对GPT4和GPT3.5进行了扩展实验，对常见的失败模式进行了表征和解耦。

    We study the ability of state-of-the art models to answer constraint satisfaction queries for information retrieval (e.g., 'a list of ice cream shops in San Diego'). In the past, such queries were considered to be tasks that could only be solved via web-search or knowledge bases. More recently, large language models (LLMs) have demonstrated initial emergent abilities in this task. However, many current retrieval benchmarks are either saturated or do not measure constraint satisfaction. Motivated by rising concerns around factual incorrectness and hallucinations of LLMs, we present KITAB, a new dataset for measuring constraint satisfaction abilities of language models. KITAB consists of book-related data across more than 600 authors and 13,000 queries, and also offers an associated dynamic data collection and constraint verification approach for acquiring similar test data for other authors. Our extended experiments on GPT4 and GPT3.5 characterize and decouple common failure modes acros
    
[^5]: 鲁棒的统一在线Top-K推荐的表示学习

    Robust Representation Learning for Unified Online Top-K Recommendation. (arXiv:2310.15492v1 [cs.IR])

    [http://arxiv.org/abs/2310.15492](http://arxiv.org/abs/2310.15492)

    本论文提出一种鲁棒的表示学习方法，用于统一在线Top-K推荐，在大规模工业电子商务中解决物品广告和内容广告的不一致性，以及跨不同领域的多实体广告的检索问题。

    

    在大规模工业电子商务中，在线推荐系统的效率对于提供高度相关的物品/内容广告以满足多样化的业务场景至关重要。然而，大部分现有研究仅关注物品广告，忽视了内容广告的重要性。这种疏忽导致了多实体结构内的不一致性和不公平检索。此外，从跨不同领域的多实体广告中检索Top-K广告的挑战也增加了复杂性。最近的研究证明，不同领域内的用户-实体行为表现出差异性和同质性特征。因此，多领域匹配模型通常依赖于具有领域不变和领域特定表示的混合专家框架。不幸的是，大多数方法主要关注优化不同专家的组合模式，未能解决优化中固有的困难点。

    In large-scale industrial e-commerce, the efficiency of an online recommendation system is crucial in delivering highly relevant item/content advertising that caters to diverse business scenarios. However, most existing studies focus solely on item advertising, neglecting the significance of content advertising. This oversight results in inconsistencies within the multi-entity structure and unfair retrieval. Furthermore, the challenge of retrieving top-k advertisements from multi-entity advertisements across different domains adds to the complexity. Recent research proves that user-entity behaviors within different domains exhibit characteristics of differentiation and homogeneity. Therefore, the multi-domain matching models typically rely on the hybrid-experts framework with domain-invariant and domain-specific representations. Unfortunately, most approaches primarily focus on optimizing the combination mode of different experts, failing to address the inherent difficulty in optimizin
    
[^6]: 基于策略卷积的大动作空间离策略评估

    Off-Policy Evaluation for Large Action Spaces via Policy Convolution. (arXiv:2310.15433v1 [cs.LG])

    [http://arxiv.org/abs/2310.15433](http://arxiv.org/abs/2310.15433)

    本研究提出了一种名为策略卷积（PC）的离策略估计方法，该方法通过动作嵌入来解决大动作空间下的分布转移问题，可以在偏差和方差之间进行权衡

    

    发展准确的离策略估计器对于评估和优化新策略至关重要。离策略估计的主要挑战在于生成数据的记录策略和我们要评估的目标策略之间的分布转移。通常，纠正分布转移的技术涉及某种形式的重要性采样。这种方法导致了无偏值估计，但往往会带来高方差的代价，即使在简单的一步情境多臂老虎机的情况下也是如此。此外，重要性采样依赖于共同支持假设，在动作空间很大时变得不切实际。为了解决这些挑战，我们引入了策略卷积 (PC)家族的估计器。这些方法利用通过动作嵌入提供的动作内部结构进行策略的策略卷积。这种卷积引入了独特的偏差-方差权衡，可以进行控制

    Developing accurate off-policy estimators is crucial for both evaluating and optimizing for new policies. The main challenge in off-policy estimation is the distribution shift between the logging policy that generates data and the target policy that we aim to evaluate. Typically, techniques for correcting distribution shift involve some form of importance sampling. This approach results in unbiased value estimation but often comes with the trade-off of high variance, even in the simpler case of one-step contextual bandits. Furthermore, importance sampling relies on the common support assumption, which becomes impractical when the action space is large. To address these challenges, we introduce the Policy Convolution (PC) family of estimators. These methods leverage latent structure within actions -- made available through action embeddings -- to strategically convolve the logging and target policies. This convolution introduces a unique bias-variance trade-off, which can be controlled 
    
[^7]: 面向深度稀疏网络的混合粒度特征交互选择方法

    Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network. (arXiv:2310.15342v1 [cs.LG])

    [http://arxiv.org/abs/2310.15342](http://arxiv.org/abs/2310.15342)

    本论文提出了一种针对深度稀疏网络的混合粒度特征交互选择方法，能够同时考虑特征域和特征值，实验证明该方法在准确性和效率方面表现良好。

    

    深度稀疏网络被广泛研究作为具有高维稀疏特征的预测任务的神经网络架构，其中特征交互选择是一个关键组成部分。然而，先前的方法主要集中在如何在粗粒度空间中搜索特征交互，对于更细粒度的细节则关注较少。在这项工作中，我们引入了一种针对深度稀疏网络的混合粒度特征交互选择方法，旨在同时考虑特征域和特征值。为了探索这样广阔的空间，我们提出了一种即时计算的分解空间。然后，我们开发了一个名为OptFeature的选择算法，它可以有效地从特征域和特征值同时选择特征交互。在三个大型真实世界基准数据集的实验结果表明，OptFeature在准确性和效率方面表现良好。额外的研究支持了我们方法的可行性。

    Deep sparse networks are widely investigated as a neural network architecture for prediction tasks with high-dimensional sparse features, with which feature interaction selection is a critical component. While previous methods primarily focus on how to search feature interaction in a coarse-grained space, less attention has been given to a finer granularity. In this work, we introduce a hybrid-grained feature interaction selection approach that targets both feature field and feature value for deep sparse networks. To explore such expansive space, we propose a decomposed space which is calculated on the fly. We then develop a selection algorithm called OptFeature, which efficiently selects the feature interaction from both the feature field and the feature value simultaneously. Results from experiments on three large real-world benchmark datasets demonstrate that OptFeature performs well in terms of accuracy and efficiency. Additional studies support the feasibility of our method.
    
[^8]: 三重单纯形矩阵完成用于费用预测

    Triple Simplex Matrix Completion for Expense Forecasting. (arXiv:2310.15275v1 [cs.LG])

    [http://arxiv.org/abs/2310.15275](http://arxiv.org/abs/2310.15275)

    本文提出了一种使用三重单纯形矩阵完成方法进行费用预测的模型。该模型通过学习项目与潜在空间中的费用模式相关性来预测费用，同时满足预算约束并保证预测结果的准确性。

    

    预测项目费用是企业避免预算超支和项目失败的关键步骤。传统上，这是由财务分析师或数据科学技术（如时间序列分析）完成的。然而，这些方法可能存在不确定性，并产生与计划预算不同的结果，特别是在项目开始时数据点有限的情况下。本文提出了一种受约束的非负矩阵完成模型，通过学习项目与潜在空间中某些费用模式的相关性，预测费用的可能性。该模型在因子矩阵和缺失条目上受到三个概率单纯形的约束。此外，预测的费用值保证满足预算约束，无需后处理。一个非精确的交替优化算法被开发用于解决相关优化问题，并证明收敛到一个稳定点。

    Forecasting project expenses is a crucial step for businesses to avoid budget overruns and project failures. Traditionally, this has been done by financial analysts or data science techniques such as time-series analysis. However, these approaches can be uncertain and produce results that differ from the planned budget, especially at the start of a project with limited data points. This paper proposes a constrained non-negative matrix completion model that predicts expenses by learning the likelihood of the project correlating with certain expense patterns in the latent space. The model is constrained on three probability simplexes, two of which are on the factor matrices and the third on the missing entries. Additionally, the predicted expense values are guaranteed to meet the budget constraint without the need of post-processing. An inexact alternating optimization algorithm is developed to solve the associated optimization problem and is proven to converge to a stationary point. Res
    
[^9]: CorefPrompt: 基于提示的事件指代消解通过测量事件类型和参数的兼容性

    CorefPrompt: Prompt-based Event Coreference Resolution by Measuring Event Type and Argument Compatibilities. (arXiv:2310.14512v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14512](http://arxiv.org/abs/2310.14512)

    CorefPrompt是一种基于提示的方法，通过测量事件类型和参数的兼容性来进行事件指代消解。该方法将事件指代消解转化为一个填空式MLM任务，并通过引入辅助的提示任务来帮助模型进行推理，最终在基准测试中取得了良好的表现。

    

    事件指代消解旨在将指代同一实际事件的事件提及聚类在一起。大多数先前的研究采用“先编码，然后评分”的框架，使得指代消解依赖于事件编码。此外，当前的方法很难利用人工总结的事件指代消解规则，例如，指代同一事件的事件应具有相同的事件类型，以指导模型。为了解决这两个问题，我们提出了一种基于提示的方法CorefPrompt，将事件指代消解转化为一个填空式MLM（掩码语言模型）任务。这样可以在一个单一的模板中同时进行事件建模和指代消解判别，并且具有完全共享的上下文。此外，我们引入了两个辅助的提示任务，事件类型兼容性和参数兼容性，以明确展示事件指代消解的推理过程，从而帮助模型做出最终的预测。实验结果表明，我们的方法CorefPrompt在最先进的基准测试中表现良好。

    Event coreference resolution (ECR) aims to group event mentions referring to the same real-world event into clusters. Most previous studies adopt the "encoding first, then scoring" framework, making the coreference judgment rely on event encoding. Furthermore, current methods struggle to leverage human-summarized ECR rules, e.g., coreferential events should have the same event type, to guide the model. To address these two issues, we propose a prompt-based approach, CorefPrompt, to transform ECR into a cloze-style MLM (masked language model) task. This allows for simultaneous event modeling and coreference discrimination within a single template, with a fully shared context. In addition, we introduce two auxiliary prompt tasks, event-type compatibility and argument compatibility, to explicitly demonstrate the reasoning process of ECR, which helps the model make final predictions. Experimental results show that our method CorefPrompt performs well in a state-of-the-art (SOTA) benchmark.
    
[^10]: AdaptSSR: 使用自适应增强自监督排序方法预训练用户模型

    AdaptSSR: Pre-training User Model with Augmentation-Adaptive Self-Supervised Ranking. (arXiv:2310.09706v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.09706](http://arxiv.org/abs/2310.09706)

    在用户建模中，通过自适应增强自监督排序方法预训练用户模型，解决了数据稀疏性问题和现有增强方法引入的噪音问题。

    

    用户建模旨在捕捉用户的特征或兴趣，但受到数据稀疏性问题的影响，往往需要依赖特定任务的标注数据。最近的几项研究通过在大量用户行为序列上进行对比学习的预训练来解决这个问题。一般而言，这些方法假设通过数据增强构建的同一行为序列的不同视图在语义上是一致的，即反映用户的相似特征或兴趣，并在特征空间中最大化它们的一致性。然而，由于用户行为的多样兴趣和大量噪音，现有的增强方法往往会丢失某些用户特征或引入噪声行为。因此，直接最大化增强视图之间的相似性可能导致负面迁移。为此，我们提出用新的预训练任务替代对比学习任务：自适应增强自监督排序方法。

    User modeling, which aims to capture users' characteristics or interests, heavily relies on task-specific labeled data and suffers from the data sparsity issue. Several recent studies tackled this problem by pre-training the user model on massive user behavior sequences with a contrastive learning task. Generally, these methods assume different views of the same behavior sequence constructed via data augmentation are semantically consistent, i.e., reflecting similar characteristics or interests of the user, and thus maximizing their agreement in the feature space. However, due to the diverse interests and heavy noise in user behaviors, existing augmentation methods tend to lose certain characteristics of the user or introduce noisy behaviors. Thus, forcing the user model to directly maximize the similarity between the augmented views may result in a negative transfer. To this end, we propose to replace the contrastive learning task with a new pretext task: Augmentation-Adaptive SelfSup
    
[^11]: 超越语义：利用自我监督学习的行为增强相关模型的学习

    Beyond Semantics: Learning a Behavior Augmented Relevance Model with Self-supervised Learning. (arXiv:2308.05379v1 [cs.IR])

    [http://arxiv.org/abs/2308.05379](http://arxiv.org/abs/2308.05379)

    这篇论文提出了一种行为增强的相关模型，利用自我监督学习，通过从用户历史行为数据中提取辅助查询-项目交互，来改进搜索引擎中的查询-项目匹配，提高准确性和鲁棒性。

    

    相关建模旨在定位与对应查询相关的理想项目，这对于搜索引擎确保用户体验非常重要。虽然大多数传统方法通过评估查询与项目之间的语义相似性来解决这个问题，但纯语义匹配并不是唯一的方法。实际上，从用户搜索记录的历史行为数据中提取的辅助查询-项目交互可以提供进一步揭示用户搜索意图的线索。得益于此，我们设计了一种新颖的基于行为增强相关学习模型的支付宝搜索模型（BARL-ASe），该模型利用目标项目的相邻查询和目标查询的相邻项目来补充目标查询-项目的语义匹配。具体而言，我们的模型建立了多层共同注意力，从相邻和目标视图中提取了粗粒度和细粒度的语义表示。模型随后采用邻居-目标的自我监督学习来提高精度和鲁棒性。

    Relevance modeling aims to locate desirable items for corresponding queries, which is crucial for search engines to ensure user experience. Although most conventional approaches address this problem by assessing the semantic similarity between the query and item, pure semantic matching is not everything. In reality, auxiliary query-item interactions extracted from user historical behavior data of the search log could provide hints to reveal users' search intents further. Drawing inspiration from this, we devise a novel Behavior Augmented Relevance Learning model for Alipay Search (BARL-ASe) that leverages neighbor queries of target item and neighbor items of target query to complement target query-item semantic matching. Specifically, our model builds multi-level co-attention for distilling coarse-grained and fine-grained semantic representations from both neighbor and target views. The model subsequently employs neighbor-target self-supervised learning to improve the accuracy and robu
    
[^12]: 基于大型语言模型的开放世界推荐系统

    Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models. (arXiv:2306.10933v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.10933](http://arxiv.org/abs/2306.10933)

    本文介绍了KAR框架，它从大型语言模型中获取两种类型的外部知识，分别是用户偏好的推理知识和项目的事实知识。通过混合专家适配器将推理和事实知识转换为增强向量，以便与现有的协同过滤推荐算法兼容。

    

    推荐系统在各种在线服务中都扮演着至关重要的角色。但是，它们在特定领域内进行训练和部署的封闭性限制了它们访问开放世界知识的能力。最近，大型语言模型(LLM)的出现在编码广泛的世界知识和展示推理能力方面显示出了希望。尽管如此，直接使用LLM作为推荐人之前的尝试并没有取得令人满意的结果。在本文中，我们提出了一种基于大型语言模型的开放世界知识增强推荐框架(KAR)，以从LLM获取两种类型的外部知识--用户偏好的推理知识和项目的事实知识。我们介绍了因子分解提示来引导对用户喜好的准确推理。生成的推理和事实知识通过混合专家适配器有效地转换并压缩为增强向量，以便与现有的协同过滤推荐算法兼容。

    Recommender systems play a vital role in various online services. However, the insulated nature of training and deploying separately within a specific domain limits their access to open-world knowledge. Recently, the emergence of large language models (LLMs) has shown promise in bridging this gap by encoding extensive world knowledge and demonstrating reasoning capability. Nevertheless, previous attempts to directly use LLMs as recommenders have not achieved satisfactory results. In this work, we propose an Open-World Knowledge Augmented Recommendation Framework with Large Language Models, dubbed KAR, to acquire two types of external knowledge from LLMs -- the reasoning knowledge on user preferences and the factual knowledge on items. We introduce factorization prompting to elicit accurate reasoning on user preferences. The generated reasoning and factual knowledge are effectively transformed and condensed into augmented vectors by a hybrid-expert adaptor in order to be compatible with
    
[^13]: 评估生成式搜索引擎中的可验证性

    Evaluating Verifiability in Generative Search Engines. (arXiv:2304.09848v1 [cs.CL])

    [http://arxiv.org/abs/2304.09848](http://arxiv.org/abs/2304.09848)

    本文评估了四个流行生成式搜索引擎的可验证性，发现现有生成式搜索引擎响应流畅但仅有51.5%的生成句子得到了完整的引用支持，仅有74.5%的引用支持其相关语句。

    

    生成式搜索引擎直接为用户查询生成响应，并提供内联引用。一个值得信赖的生成式搜索引擎的先决条件是可验证性，即系统应全面引用（高引用回忆率，所有语句都有完整的引用支持）和准确（高引用精度，每个引用都支持其相关语句）。我们对四个流行的生成式搜索引擎——Bing Chat、NeevaAI、perplexity.ai和YouChat——进行了人类评估，涵盖了各种来源的多样化查询（例如历史上的Google用户查询、Reddit上动态收集的开放性问题等）。我们发现现有的生成式搜索引擎响应流畅且信息丰富，但常常包含不支持的语句和不准确的引用：平均而言，仅有51.5%的生成句子得到了完整的引用支持，只有74.5%的引用支持其相关语句。我们认为...

    Generative search engines directly generate responses to user queries, along with in-line citations. A prerequisite trait of a trustworthy generative search engine is verifiability, i.e., systems should cite comprehensively (high citation recall; all statements are fully supported by citations) and accurately (high citation precision; every cite supports its associated statement). We conduct human evaluation to audit four popular generative search engines -- Bing Chat, NeevaAI, perplexity.ai, and YouChat -- across a diverse set of queries from a variety of sources (e.g., historical Google user queries, dynamically-collected open-ended questions on Reddit, etc.). We find that responses from existing generative search engines are fluent and appear informative, but frequently contain unsupported statements and inaccurate citations: on average, a mere 51.5% of generated sentences are fully supported by citations and only 74.5% of citations support their associated sentence. We believe that
    
[^14]: 基于内容的深层生成模型搜索

    Content-Based Search for Deep Generative Models. (arXiv:2210.03116v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.03116](http://arxiv.org/abs/2210.03116)

    这个论文介绍了基于内容的深层生成模型搜索任务，通过优化问题选择生成与查询最相似内容概率最高的模型，并提出了适用于不同查询模态的对比学习框架。（翻译为中文）

    

    自定义和预训练生成模型的不断增加使得用户不可能完全了解每个存在的模型。为了解决这个问题，我们引入了基于内容的模型搜索任务：给定一个查询和一组大规模的生成模型，找到与查询最匹配的模型。由于每个生成模型产生一系列图像的分布，我们将搜索任务作为一个优化问题，选择生成与查询相似内容概率最高的模型。我们提出了一个用于近似计算概率的公式，可以根据不同的查询模态（例如图像、草图和文本）来计算。此外，我们提出了一个对模型检索的对比学习框架，该框架学习适应不同查询模态的特征。我们证明了我们的方法在生成模型动物园（Generative Model Zoo）上优于几个基准模型的表现。

    The growing proliferation of customized and pretrained generative models has made it infeasible for a user to be fully cognizant of every model in existence. To address this need, we introduce the task of content-based model search: given a query and a large set of generative models, finding the models that best match the query. As each generative model produces a distribution of images, we formulate the search task as an optimization problem to select the model with the highest probability of generating similar content as the query. We introduce a formulation to approximate this probability given the query from different modalities, e.g., image, sketch, and text. Furthermore, we propose a contrastive learning framework for model retrieval, which learns to adapt features for various query modalities. We demonstrate that our method outperforms several baselines on Generative Model Zoo, a new benchmark we create for the model retrieval task.
    

