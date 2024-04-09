# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tired of Plugins? Large Language Models Can Be End-To-End Recommenders](https://arxiv.org/abs/2404.00702) | 大型语言模型UniLLMRec将多阶段任务整合为端到端推荐框架，以解决推荐系统中对大规模物品集合的挑战 |
| [^2] | [To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models](https://arxiv.org/abs/2403.18628) | 这项研究提出并定义了可推荐性识别问题，专注于在当前对话环境中探讨是否需要推荐，以减少用户干扰。 |
| [^3] | [RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers](https://arxiv.org/abs/2403.18276) | Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。 |
| [^4] | [Trajectory-wise Iterative Reinforcement Learning Framework for Auto-bidding](https://arxiv.org/abs/2402.15102) | 自动广告竞标中使用了一种新的迭代离线强化学习框架，有效缓解了传统RL算法在在线环境下性能下降的问题。 |
| [^5] | [IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus](https://arxiv.org/abs/2402.14710) | 发布了IEPile，一个包含约0.32B个标记的综合双语IE指令语料库，通过收集和清理33个现有IE数据集并引入基于模式的指令生成，可以提高大型语言模型在信息抽取领域的性能，尤其是零样本泛化。 |
| [^6] | [Exploring the Integration Strategies of Retriever and Large Language Models.](http://arxiv.org/abs/2308.12574) | 本文通过探索不同的检索器和大型语言模型整合方法来增强答案生成，并发现常用的连接方法存在局限性。为了解决这个问题，本文提出了四种替代策略，包括两种单轮方法和两种多轮策略。 |

# 详细

[^1]: 厌倦了插件？大型语言模型可以成为端到端推荐系统

    Tired of Plugins? Large Language Models Can Be End-To-End Recommenders

    [https://arxiv.org/abs/2404.00702](https://arxiv.org/abs/2404.00702)

    大型语言模型UniLLMRec将多阶段任务整合为端到端推荐框架，以解决推荐系统中对大规模物品集合的挑战

    

    推荐系统旨在基于历史行为数据预测用户兴趣。它们主要设计为顺序流水线，需要大量数据来训练不同子系统，并且难以扩展到新域。最近，大型语言模型（LLMs）展示了出色的通用能力，使一个模型能够处理各种场景中的多样化推荐任务。然而，现有基于LLM的推荐系统纯粹利用LLM来处理推荐流水线的单个任务。此外，这些系统面临着以自然语言格式向LLM呈现大规模物品集合的挑战，由于输入长度的限制。为解决这些挑战，我们引入了一种基于LLM的端到端推荐框架：UniLLMRec。具体而言，UniLLMRec通过推荐链集成多阶段任务（例如召回、排序、重新排序）。为了处理大规模物品，我们提出

    arXiv:2404.00702v1 Announce Type: new  Abstract: Recommender systems aim to predict user interest based on historical behavioral data. They are mainly designed in sequential pipelines, requiring lots of data to train different sub-systems, and are hard to scale to new domains. Recently, Large Language Models (LLMs) have demonstrated remarkable generalized capabilities, enabling a singular model to tackle diverse recommendation tasks across various scenarios. Nonetheless, existing LLM-based recommendation systems utilize LLM purely for a single task of the recommendation pipeline. Besides, these systems face challenges in presenting large-scale item sets to LLMs in natural language format, due to the constraint of input length. To address these challenges, we introduce an LLM-based end-to-end recommendation framework: UniLLMRec. Specifically, UniLLMRec integrates multi-stage tasks (e.g. recall, ranking, re-ranking) via chain-of-recommendations. To deal with large-scale items, we propose
    
[^2]: 是否推荐：在与预训练语言模型的对话中识别可推荐性

    To Recommend or Not: Recommendability Identification in Conversations with Pre-trained Language Models

    [https://arxiv.org/abs/2403.18628](https://arxiv.org/abs/2403.18628)

    这项研究提出并定义了可推荐性识别问题，专注于在当前对话环境中探讨是否需要推荐，以减少用户干扰。

    

    大多数当前的推荐系统主要关注于推荐什么，假设用户总是需要个性化推荐。然而，随着ChatGPT和其他聊天机器人的广泛传播，在对话系统的背景下更关键的问题是在为用户提供推荐服务时如何最小化用户的干扰。我们正式定义了可推荐性识别问题，旨在确定在特定场景中是否需要推荐。

    arXiv:2403.18628v1 Announce Type: new  Abstract: Most current recommender systems primarily focus on what to recommend, assuming users always require personalized recommendations. However, with the widely spread of ChatGPT and other chatbots, a more crucial problem in the context of conversational systems is how to minimize user disruption when we provide recommendation services for users. While previous research has extensively explored different user intents in dialogue systems, fewer efforts are made to investigate whether recommendations should be provided. In this paper, we formally define the recommendability identification problem, which aims to determine whether recommendations are necessary in a specific scenario. First, we propose and define the recommendability identification task, which investigates the need for recommendations in the current conversational context. A new dataset is constructed. Subsequently, we discuss and evaluate the feasibility of leveraging pre-trained
    
[^3]: RankMamba，在Transformer时代对Mamba文档排名性能的基准测试

    RankMamba, Benchmarking Mamba's Document Ranking Performance in the Era of Transformers

    [https://arxiv.org/abs/2403.18276](https://arxiv.org/abs/2403.18276)

    Mamba模型基于状态空间模型，在多个序列建模任务中取得了与Transformer相当的性能，并在经典信息检索任务--文档排名中展现了其有效性。

    

    Transformer结构在自然语言处理（NLP）、计算机视觉（CV）和信息检索(IR)等多个应用的机器学习领域取得了巨大成功。Transformer架构的核心机制--注意力，在训练中需要$O(n^2)$的时间复杂度，在推断中需要$O(n)$的时间复杂度。许多工作已经提出改进注意力机制的可扩展性，比如Flash Attention和Multi-query Attention。另一方面的工作旨在设计新的机制来取代注意力。最近，基于状态空间模型的一个显著模型结构--Mamba，在多个序列建模任务中取得了与Transformer相当的性能。

    arXiv:2403.18276v1 Announce Type: cross  Abstract: Transformer structure has achieved great success in multiple applied machine learning communities, such as natural language processing (NLP), computer vision (CV) and information retrieval (IR). Transformer architecture's core mechanism -- attention requires $O(n^2)$ time complexity in training and $O(n)$ time complexity in inference. Many works have been proposed to improve the attention mechanism's scalability, such as Flash Attention and Multi-query Attention. A different line of work aims to design new mechanisms to replace attention. Recently, a notable model structure -- Mamba, which is based on state space models, has achieved transformer-equivalent performance in multiple sequence modeling tasks.   In this work, we examine \mamba's efficacy through the lens of a classical IR task -- document ranking. A reranker model takes a query and a document as input, and predicts a scalar relevance score. This task demands the language mod
    
[^4]: 轨迹式迭代强化学习框架用于自动竞标

    Trajectory-wise Iterative Reinforcement Learning Framework for Auto-bidding

    [https://arxiv.org/abs/2402.15102](https://arxiv.org/abs/2402.15102)

    自动广告竞标中使用了一种新的迭代离线强化学习框架，有效缓解了传统RL算法在在线环境下性能下降的问题。

    

    在在线广告中，广告主参与广告竞拍以获取广告机会，通常是通过需求方平台(DSPs)提供的自动竞标工具。目前的自动竞标算法通常采用强化学习（RL）。然而，由于安全性问题，大多数基于RL的自动竞标策略是在模拟环境中进行训练的，在在线环境中部署会导致性能下降。为了缩小这一差距，我们可以并行部署多个自动竞标代理以收集大量交互数据集。然后，可以利用离线RL算法训练新策略。训练后的策略随后可以部署以进行进一步的数据收集，从而形成一个迭代训练框架，我们将其称为迭代离线RL。在这项工作中，我们确定了这种迭代离线RL框架的性能瓶颈，其根源在于由于内在原因而导致的探索和利用的低效问题。

    arXiv:2402.15102v1 Announce Type: cross  Abstract: In online advertising, advertisers participate in ad auctions to acquire ad opportunities, often by utilizing auto-bidding tools provided by demand-side platforms (DSPs). The current auto-bidding algorithms typically employ reinforcement learning (RL). However, due to safety concerns, most RL-based auto-bidding policies are trained in simulation, leading to a performance degradation when deployed in online environments. To narrow this gap, we can deploy multiple auto-bidding agents in parallel to collect a large interaction dataset. Offline RL algorithms can then be utilized to train a new policy. The trained policy can subsequently be deployed for further data collection, resulting in an iterative training framework, which we refer to as iterative offline RL. In this work, we identify the performance bottleneck of this iterative offline RL framework, which originates from the ineffective exploration and exploitation caused by the inhe
    
[^5]: IEPile: 挖掘大规模基于模式的信息抽取语料库

    IEPile: Unearthing Large-Scale Schema-Based Information Extraction Corpus

    [https://arxiv.org/abs/2402.14710](https://arxiv.org/abs/2402.14710)

    发布了IEPile，一个包含约0.32B个标记的综合双语IE指令语料库，通过收集和清理33个现有IE数据集并引入基于模式的指令生成，可以提高大型语言模型在信息抽取领域的性能，尤其是零样本泛化。

    

    大型语言模型（LLMs）在各个领域展现出了显著的潜力；然而，在信息抽取（IE）方面表现出了显著的性能差距。高质量的指令数据是提升LLMs特定能力的关键，而当前的IE数据集往往规模较小、分散且缺乏标准化的模式。因此，我们介绍了IEPile，一个综合的双语（英文和中文）IE指令语料库，包含约0.32B个标记。我们通过收集和清理33个现有IE数据集构建IEPile，并引入基于模式的指令生成来挖掘大规模语料库。在LLaMA和Baichuan上的实验结果表明，使用IEPile可以提高LLMs在IE方面的性能，尤其是零样本泛化。我们开源了资源和预训练模型，希望为自然语言处理社区提供有价值的支持。

    arXiv:2402.14710v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demonstrate remarkable potential across various domains; however, they exhibit a significant performance gap in Information Extraction (IE). Note that high-quality instruction data is the vital key for enhancing the specific capabilities of LLMs, while current IE datasets tend to be small in scale, fragmented, and lack standardized schema. To this end, we introduce IEPile, a comprehensive bilingual (English and Chinese) IE instruction corpus, which contains approximately 0.32B tokens. We construct IEPile by collecting and cleaning 33 existing IE datasets, and introduce schema-based instruction generation to unearth a large-scale corpus. Experimental results on LLaMA and Baichuan demonstrate that using IEPile can enhance the performance of LLMs for IE, especially the zero-shot generalization. We open-source the resource and pre-trained models, hoping to provide valuable support to the NLP community.
    
[^6]: 探索检索器和大型语言模型的整合策略

    Exploring the Integration Strategies of Retriever and Large Language Models. (arXiv:2308.12574v1 [cs.IR])

    [http://arxiv.org/abs/2308.12574](http://arxiv.org/abs/2308.12574)

    本文通过探索不同的检索器和大型语言模型整合方法来增强答案生成，并发现常用的连接方法存在局限性。为了解决这个问题，本文提出了四种替代策略，包括两种单轮方法和两种多轮策略。

    

    检索到的段落和大型语言模型（如ChatGPT）的整合为提高开放领域问答作出了显著贡献。然而，如何将检索到的段落融入答案生成过程中的最佳方法仍然缺乏探索。本文旨在通过研究不同的方法来结合检索到的段落和大型语言模型以增强答案生成。我们首先研究了常用的连接方法的局限性。令人惊讶的是，即使正确的文档在前k个检索到的段落中，这种方法经常会生成“未知”输出。为了解决这个问题，我们探索了四种将检索到的段落与大型语言模型整合的替代策略。这些策略包括两种利用思维链推理的单轮方法和两种利用反馈循环的多轮策略。通过全面的分析和实验，我们发现...

    The integration of retrieved passages and large language models (LLMs), such as ChatGPTs, has significantly contributed to improving open-domain question answering. However, there is still a lack of exploration regarding the optimal approach for incorporating retrieved passages into the answer generation process. This paper aims to fill this gap by investigating different methods of combining retrieved passages with LLMs to enhance answer generation. We begin by examining the limitations of a commonly-used concatenation approach. Surprisingly, this approach often results in generating "unknown" outputs, even when the correct document is among the top-k retrieved passages. To address this issue, we explore four alternative strategies for integrating the retrieved passages with the LLMs. These strategies include two single-round methods that utilize chain-of-thought reasoning and two multi-round strategies that incorporate feedback loops. Through comprehensive analyses and experiments, w
    

