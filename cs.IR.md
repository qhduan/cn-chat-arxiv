# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Approximate Nearest Neighbor Search with Window Filters](https://rss.arxiv.org/abs/2402.00943) | 这篇论文提出了一种使用窗口过滤的近似最近邻搜索方法，能够在各种语义搜索问题中实现高速搜索，并在多个基准数据集上取得了显著的速度提升。 |
| [^2] | [RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records](https://arxiv.org/abs/2403.00815) | RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。 |
| [^3] | [ARL2: Aligning Retrievers for Black-box Large Language Models via Self-guided Adaptive Relevance Labeling](https://arxiv.org/abs/2402.13542) | ARL2提出了一种检索器学习技术，利用LLMs作为标注者，并采用自适应自训练策略，能够有效减少注释成本，并在NQ和MMLU上取得了5.4%和4.6%的准确度提升。 |
| [^4] | [Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation](https://arxiv.org/abs/2402.07092) | 本文提出了一种通过LLM-认知数据增强的方法来广义对话密集检索。该方法首先生成多级增强对话，捕捉多样的对话环境。其次，通过认知感知过程减少错误生成情况，并通过难度自适应样本筛选器选择具有挑战性的样本。 |
| [^5] | [C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.03181) | C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。 |
| [^6] | [Data-efficient Fine-tuning for LLM-based Recommendation](https://arxiv.org/abs/2401.17197) | 本论文中介绍了基于LLM的推荐中的数据高效微调的方法，通过剪枝数据来减少LLM的微调成本，并提出了两种目标来实现高准确性的数据剪枝。 |
| [^7] | [Evaluating ChatGPT as a Recommender System: A Rigorous Approach.](http://arxiv.org/abs/2309.03613) | 这项研究评估了ChatGPT作为推荐系统的能力，通过探索其利用用户偏好进行推荐、重新排序推荐列表、利用相似用户信息以及处理冷启动情况的能力，并使用三个数据集进行了全面实验。 |

# 详细

[^1]: 使用窗口过滤的近似最近邻搜索

    Approximate Nearest Neighbor Search with Window Filters

    [https://rss.arxiv.org/abs/2402.00943](https://rss.arxiv.org/abs/2402.00943)

    这篇论文提出了一种使用窗口过滤的近似最近邻搜索方法，能够在各种语义搜索问题中实现高速搜索，并在多个基准数据集上取得了显著的速度提升。

    

    我们定义并研究了$\textit{c-近似窗口搜索}$问题：近似最近邻搜索其中数据集中的每个点都有一个数值标签，目标是在任意标签范围内找到查询点的最近邻。许多语义搜索问题，例如带有时间戳过滤器的图像和文档搜索，或带有成本过滤器的产品搜索，是这个问题的自然例子。我们提出并在理论上分析了一种基于模块化树的框架，用于将解决传统c-近似最近邻问题的索引转化为解决窗口搜索的数据结构。在标准的最近邻基准数据集上，配备了随机标签值、对抗性构建的嵌入以及带有真实时间戳的图像搜索嵌入，我们获得了与现有解决方案相比高达75倍的速度提升，同时保持相同的召回率。

    We define and investigate the problem of $\textit{c-approximate window search}$: approximate nearest neighbor search where each point in the dataset has a numeric label, and the goal is to find nearest neighbors to queries within arbitrary label ranges. Many semantic search problems, such as image and document search with timestamp filters, or product search with cost filters, are natural examples of this problem. We propose and theoretically analyze a modular tree-based framework for transforming an index that solves the traditional c-approximate nearest neighbor problem into a data structure that solves window search. On standard nearest neighbor benchmark datasets equipped with random label values, adversarially constructed embeddings, and image search embeddings with real timestamps, we obtain up to a $75\times$ speedup over existing solutions at the same level of recall.
    
[^2]: RAM-EHR: 电子健康记录上的检索增强与临床预测相遇

    RAM-EHR: Retrieval Augmentation Meets Clinical Predictions on Electronic Health Records

    [https://arxiv.org/abs/2403.00815](https://arxiv.org/abs/2403.00815)

    RAM-EHR通过增强检索并利用总结知识，提高了针对电子健康记录的临床预测效果。

    

    我们提出了RAM-EHR，这是一个用于改善电子健康记录（EHR）上临床预测的检索增强（Retrieval Augmentation）流程。RAM-EHR首先收集多个知识来源，将它们转换为文本格式，并使用密集检索来获取与医学概念相关的信息。这一策略解决了与复杂概念名称相关的困难。RAM-EHR然后增广了与一致性正则化代码联合训练的本地EHR预测模型，以捕获来自患者就诊和总结知识的互补信息。在两个EHR数据集上的实验表明，RAM-EHR相对于之前的知识增强基线效果显著（AUROC增益3.4％，AUPR增益7.2％），强调了RAM-EHR的总结知识对临床预测任务的有效性。代码将发布在\url{https://github.com/ritaranx/RAM-EHR}。

    arXiv:2403.00815v1 Announce Type: cross  Abstract: We present RAM-EHR, a Retrieval AugMentation pipeline to improve clinical predictions on Electronic Health Records (EHRs). RAM-EHR first collects multiple knowledge sources, converts them into text format, and uses dense retrieval to obtain information related to medical concepts. This strategy addresses the difficulties associated with complex names for the concepts. RAM-EHR then augments the local EHR predictive model co-trained with consistency regularization to capture complementary information from patient visits and summarized knowledge. Experiments on two EHR datasets show the efficacy of RAM-EHR over previous knowledge-enhanced baselines (3.4% gain in AUROC and 7.2% gain in AUPR), emphasizing the effectiveness of the summarized knowledge from RAM-EHR for clinical prediction tasks. The code will be published at \url{https://github.com/ritaranx/RAM-EHR}.
    
[^3]: ARL2: 通过自导自适应相关性标记将检索器与黑盒大型语言模型对齐

    ARL2: Aligning Retrievers for Black-box Large Language Models via Self-guided Adaptive Relevance Labeling

    [https://arxiv.org/abs/2402.13542](https://arxiv.org/abs/2402.13542)

    ARL2提出了一种检索器学习技术，利用LLMs作为标注者，并采用自适应自训练策略，能够有效减少注释成本，并在NQ和MMLU上取得了5.4%和4.6%的准确度提升。

    

    arXiv:2402.13542v1 公告类型: 交叉 摘要: 检索增强生成通过整合外部知识源的相关信息改进大型语言模型（LLMs），使LLMs能够适应特定领域，并减轻知识密集任务中的幻觉。然而，由于其分开的训练过程和LLMs的黑盒特性，现有的检索器通常与LLMs不匹配。为解决这一挑战，我们提出了ARL2，一种利用LLMs作为标注者的检索器学习技术。ARL2利用LLMs注释和评分相关证据，从而能够从强大的LLM监督中学习检索器。此外，ARL2使用自适应自训练策略来策划高质量和多样性相关性数据，可以有效降低标注成本。大量实验表明ARL2的有效性，与最先进方法相比，在NQ上提高了5.4%的准确率，在MMLU上提高了4.6%。

    arXiv:2402.13542v1 Announce Type: cross  Abstract: Retrieval-augmented generation enhances large language models (LLMs) by incorporating relevant information from external knowledge sources. This enables LLMs to adapt to specific domains and mitigate hallucinations in knowledge-intensive tasks. However, existing retrievers are often misaligned with LLMs due to their separate training processes and the black-box nature of LLMs. To address this challenge, we propose ARL2, a retriever learning technique that harnesses LLMs as labelers. ARL2 leverages LLMs to annotate and score relevant evidence, enabling learning the retriever from robust LLM supervision. Furthermore, ARL2 uses an adaptive self-training strategy for curating high-quality and diverse relevance data, which can effectively reduce the annotation cost. Extensive experiments demonstrate the effectiveness of ARL2, achieving accuracy improvements of 5.4% on NQ and 4.6% on MMLU compared to the state-of-the-art methods. Additionall
    
[^4]: 通过LLM-认知数据增强广义对话密集检索

    Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation

    [https://arxiv.org/abs/2402.07092](https://arxiv.org/abs/2402.07092)

    本文提出了一种通过LLM-认知数据增强的方法来广义对话密集检索。该方法首先生成多级增强对话，捕捉多样的对话环境。其次，通过认知感知过程减少错误生成情况，并通过难度自适应样本筛选器选择具有挑战性的样本。

    

    对话式搜索利用多轮自然语言环境来检索相关段落。现有的对话密集检索模型大多将对话视为一系列固定的问题和回答，忽视了严重的数据稀疏性问题 - 也就是说，用户可以以不同的方式进行对话，而这些备选对话是未记录的。因此，它们经常难以推广到真实场景中的多样对话。在这项工作中，我们提出了一种通过LLM-认知数据增强广义对话密集检索的框架(ConvAug)。ConvAug首先生成多级增强对话，以捕捉对话环境的多样性。受人类认知方式的启发，我们设计了一种认知感知过程，以减少错误的正例、负例和幻觉的生成。此外，我们还开发了一种难度自适应样本筛选器，用于选择复杂对话的具有挑战性的样本。

    Conversational search utilizes muli-turn natural language contexts to retrieve relevant passages. Existing conversational dense retrieval models mostly view a conversation as a fixed sequence of questions and responses, overlooking the severe data sparsity problem -- that is, users can perform a conversation in various ways, and these alternate conversations are unrecorded. Consequently, they often struggle to generalize to diverse conversations in real-world scenarios. In this work, we propose a framework for generalizing Conversational dense retrieval via LLM-cognition data Augmentation (ConvAug). ConvAug first generates multi-level augmented conversations to capture the diverse nature of conversational contexts. Inspired by human cognition, we devise a cognition-aware process to mitigate the generation of false positives, false negatives, and hallucinations. Moreover, we develop a difficulty-adaptive sample filter that selects challenging samples for complex conversations, thereby g
    
[^5]: C-RAG: 针对检索增强语言模型的认证生成风险

    C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models

    [https://arxiv.org/abs/2402.03181](https://arxiv.org/abs/2402.03181)

    C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。

    

    尽管大型语言模型（LLMs）在各种应用中具备令人印象深刻的能力，但它们仍然存在可信度问题，如幻觉和错位。检索增强语言模型（RAG）被提出来增强生成结果的可信性，通过引入外部知识。但是，对于RAG模型的生成风险的理论理解尚未被研究。本文回答了以下问题：1）RAG是否确实能够降低生成风险，2）如何对RAG和传统LLM的生成风险提供可证明的保证，以及3）哪些充分条件使得RAG模型能够降低生成风险。我们提出了C-RAG，第一个用于认证RAG模型生成风险的框架。具体而言，我们为RAG模型提供了符合风险分析，并确保了生成风险的上界，我们称之为符合生成风险。我们还对一般有界风险下的符合生成风险提供了理论保证。

    Despite the impressive capabilities of large language models (LLMs) across diverse applications, they still suffer from trustworthiness issues, such as hallucinations and misalignments. Retrieval-augmented language models (RAG) have been proposed to enhance the credibility of generations by grounding external knowledge, but the theoretical understandings of their generation risks remains unexplored. In this paper, we answer: 1) whether RAG can indeed lead to low generation risks, 2) how to provide provable guarantees on the generation risks of RAG and vanilla LLMs, and 3) what sufficient conditions enable RAG models to reduce generation risks. We propose C-RAG, the first framework to certify generation risks for RAG models. Specifically, we provide conformal risk analysis for RAG models and certify an upper confidence bound of generation risks, which we refer to as conformal generation risk. We also provide theoretical guarantees on conformal generation risks for general bounded risk f
    
[^6]: 基于LLM的推荐的数据高效微调

    Data-efficient Fine-tuning for LLM-based Recommendation

    [https://arxiv.org/abs/2401.17197](https://arxiv.org/abs/2401.17197)

    本论文中介绍了基于LLM的推荐中的数据高效微调的方法，通过剪枝数据来减少LLM的微调成本，并提出了两种目标来实现高准确性的数据剪枝。

    

    近年来，利用大型语言模型（LLM）进行推荐引起了广泛关注，其中微调在LLM的适应中起着关键作用。然而，快速扩展的推荐数据上微调LLM的成本限制了它们的实际应用。为了解决这一挑战，少样本微调提供了一种快速适应LLM到新的推荐数据的方法。我们提出了为高效的LLM推荐任务剪枝数据的任务，旨在找到适合LLM的少样本微调的代表样本。虽然核心集选择与所提出的任务密切相关，但现有的核心集选择方法往往依赖于次优启发式指标或需要在大规模推荐数据上进行昂贵的优化。

    Leveraging Large Language Models (LLMs) for recommendation has recently garnered considerable attention, where fine-tuning plays a key role in LLMs' adaptation. However, the cost of fine-tuning LLMs on rapidly expanding recommendation data limits their practical application. To address this challenge, few-shot fine-tuning offers a promising approach to quickly adapt LLMs to new recommendation data. We propose the task of data pruning for efficient LLM-based recommendation, aimed at identifying representative samples tailored for LLMs' few-shot fine-tuning. While coreset selection is closely related to the proposed task, existing coreset selection methods often rely on suboptimal heuristic metrics or entail costly optimization on large-scale recommendation data.   To tackle these issues, we introduce two objectives for the data pruning task in the context of LLM-based recommendation: 1) high accuracy aims to identify the influential samples that can lead to high overall performance; and
    
[^7]: 评估ChatGPT作为推荐系统的严谨方法

    Evaluating ChatGPT as a Recommender System: A Rigorous Approach. (arXiv:2309.03613v1 [cs.IR])

    [http://arxiv.org/abs/2309.03613](http://arxiv.org/abs/2309.03613)

    这项研究评估了ChatGPT作为推荐系统的能力，通过探索其利用用户偏好进行推荐、重新排序推荐列表、利用相似用户信息以及处理冷启动情况的能力，并使用三个数据集进行了全面实验。

    

    由于其卓越的自然语言处理能力，大型AI语言模型近年来备受关注。它们在语言相关任务中具有重要贡献，包括基于提示的学习，因此对于各种特定任务非常有价值。这种方法释放了它们的全部潜力，提高了准确性和泛化性。研究界正在积极探索它们的应用，ChatGPT也因此获得了认可。尽管大型语言模型已经有了广泛的研究，但其在推荐场景中的潜力仍待探索。本研究旨在填补这一空白，通过探究ChatGPT作为零-shot推荐系统的能力。我们的目标包括评估其利用用户偏好进行推荐、重新排序现有推荐列表、利用相似用户的信息以及处理冷启动情况的能力。我们通过对三个数据集（MovieLens Small、Last.FM和Facebook Bo）进行全面实验来评估ChatGPT的性能。

    Recent popularity surrounds large AI language models due to their impressive natural language capabilities. They contribute significantly to language-related tasks, including prompt-based learning, making them valuable for various specific tasks. This approach unlocks their full potential, enhancing precision and generalization. Research communities are actively exploring their applications, with ChatGPT receiving recognition. Despite extensive research on large language models, their potential in recommendation scenarios still needs to be explored. This study aims to fill this gap by investigating ChatGPT's capabilities as a zero-shot recommender system. Our goals include evaluating its ability to use user preferences for recommendations, reordering existing recommendation lists, leveraging information from similar users, and handling cold-start situations. We assess ChatGPT's performance through comprehensive experiments using three datasets (MovieLens Small, Last.FM, and Facebook Bo
    

