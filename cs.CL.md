# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods](https://arxiv.org/abs/2404.00282) | 大型语言模型在强化学习中具有潜在优势，通过结构化分类和角色分析，为未来研究提供指导。 |
| [^2] | [CURATRON: Complete Robust Preference Data for Robust Alignment of Large Language Models](https://arxiv.org/abs/2403.02745) | 本论文提出了一种新方法，通过彻底重校准偏好数据集中的价值观，以增强大型语言模型对问题的韧性。 |
| [^3] | [LLMs for Targeted Sentiment in News Headlines: Exploring Different Levels of Prompt Prescriptiveness](https://arxiv.org/abs/2403.00418) | 提出了探索LLMs在新闻标题有针对性情感分析中不同级别提示规范性的方法，以提高其性能。 |
| [^4] | [Speak Out of Turn: Safety Vulnerability of Large Language Models in Multi-turn Dialogue](https://arxiv.org/abs/2402.17262) | 本论文探讨了多轮对话中大型语言模型的安全性漏洞，指出人类可以通过多轮对话诱使其生成有害信息。 |
| [^5] | [Noise Contrastive Alignment of Language Models with Explicit Rewards](https://arxiv.org/abs/2402.05369) | 本文提出了一个基于噪声对比估计的通用LM对齐框架，能够处理明确注释的奖励数据，并且扩展了当前的对齐理论。 |
| [^6] | [Comparing Template-based and Template-free Language Model Probing](https://arxiv.org/abs/2402.00123) | 本文比较了基于模板和非模板语言模型的探测方法，发现它们在模型排名、绝对得分和与困惑度的关系等方面存在差异。 |
| [^7] | [Large language models can enhance persuasion through linguistic feature alignment](https://arxiv.org/abs/2311.16466) | 本研究调查了大型语言模型对人类沟通的影响，使用了消费者金融投诉数据，并发现大型语言模型的使用可能增强了一整套语言特征，提高了信息说服力。 |
| [^8] | [Recent Advances in Hate Speech Moderation: Multimodality and the Role of Large Models.](http://arxiv.org/abs/2401.16727) | 这项综合调查总结了最近在仇恨言论审核方面的进展，重点介绍了大型语言模型和大型多模态模型的作用。研究发现了文本、视觉和听觉元素在传播仇恨言论中的微妙相互作用，并强调了大型模型对审核能力的重新定义。同时，研究还指出了在少数语言和文化背景下的研究差距和处理低资源环境的需求。 |
| [^9] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^10] | [ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models.](http://arxiv.org/abs/2310.08975) | ChatKBQA是一个基于精调大型语言模型的生成-检索框架，用于改进知识库问答的效率和准确性，实验结果显示在多个数据集上取得了新的最好表现。 |
| [^11] | [Text2NKG: Fine-Grained N-ary Relation Extraction for N-ary relational Knowledge Graph Construction.](http://arxiv.org/abs/2310.05185) | Text2NKG是一种用于构建N元关系知识图的细粒度N元关系抽取框架，支持多种NKG模式，具有高灵活性和实用性。 |
| [^12] | [Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs.](http://arxiv.org/abs/2310.01801) | 该研究提出了一种自适应的KV缓存压缩方法，用于减少大型语言模型的内存消耗。通过有针对性的分析和结构识别，构建了具有自适应性的KV缓存，通过清除和丢弃特定的上下文，以及只对特定的注意力头使用标准KV缓存，实现了显著的内存占用减少。 |
| [^13] | [Federated Large Language Model: A Position Paper.](http://arxiv.org/abs/2307.08925) | 我们提出了联邦式大规模语言模型的概念，通过联邦学习实现分散数据的共同训练共享模型，以应对公共数据可用性的限制和私有数据的隐私保护需求。我们讨论了预训练、微调和提示工程这三个组件的优势，并提出了实施策略。同时，我们探讨了FL和LLM集成带来的新挑战，并分析了现有解决方案和潜在障碍。 |
| [^14] | [Domain-Expanded ASTE: Rethinking Generalization in Aspect Sentiment Triplet Extraction.](http://arxiv.org/abs/2305.14434) | 该论文提出了一个领域扩展的ASTE基准数据集，通过生成方法来解决领域泛化的问题。 |
| [^15] | [Efficient distributed representations beyond negative sampling.](http://arxiv.org/abs/2303.17475) | 本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。 |

# 详细

[^1]: 基于大型语言模型增强强化学习的调查:概念、分类和方法

    Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods

    [https://arxiv.org/abs/2404.00282](https://arxiv.org/abs/2404.00282)

    大型语言模型在强化学习中具有潜在优势，通过结构化分类和角色分析，为未来研究提供指导。

    

    随着大规模语言模型(LLMs)拥有广泛的预训练知识和高级通用能力，它们在增强学习方面如多任务学习、样本效率和任务规划等方面展现出潜力。本调查综述了现有$\textit{LLM增强RL}$文献，总结了其与传统RL方法的特征，旨在澄清研究范围和未来研究方向。利用经典的Agent-环境交互范例，我们提出了一个结构化的分类法，系统地将LLMs在RL中的功能分类，包括四种角色：信息处理器、奖励设计者、决策者和生成器。此外，针对每个角色，我们总结了方法论，分析了缓解的特定RL挑战，并提供了未来方向的见解。最后，潜在应用、前景

    arXiv:2404.00282v1 Announce Type: cross  Abstract: With extensive pre-trained knowledge and high-level general capabilities, large language models (LLMs) emerge as a promising avenue to augment reinforcement learning (RL) in aspects such as multi-task learning, sample efficiency, and task planning. In this survey, we provide a comprehensive review of the existing literature in $\textit{LLM-enhanced RL}$ and summarize its characteristics compared to conventional RL methods, aiming to clarify the research scope and directions for future studies. Utilizing the classical agent-environment interaction paradigm, we propose a structured taxonomy to systematically categorize LLMs' functionalities in RL, including four roles: information processor, reward designer, decision-maker, and generator. Additionally, for each role, we summarize the methodologies, analyze the specific RL challenges that are mitigated, and provide insights into future directions. Lastly, potential applications, prospecti
    
[^2]: CURATRON：完整健壮偏好数据用于大型语言模型的健壮对齐

    CURATRON: Complete Robust Preference Data for Robust Alignment of Large Language Models

    [https://arxiv.org/abs/2403.02745](https://arxiv.org/abs/2403.02745)

    本论文提出了一种新方法，通过彻底重校准偏好数据集中的价值观，以增强大型语言模型对问题的韧性。

    

    这篇论文解决了通过偏好学习（PL）将大型语言模型（LLMs）与人类价值观对齐的挑战，重点关注偏好数据集中不完整和损坏的问题。我们提出了一种新方法，通过彻底和完全地重新校准这些数据集中的价值观，以增强LLMs对问题的韧性。特别是，我们设计了一个有保证的多项式时间排名算法，可以增强几种现有模型的健壮性，比如经典的Bradley–Terry–Luce（BTL）（Bradley和Terry，1952）模型以及对其某些推广。据我们所知，我们的工作是第一个提出一种可证明在高概率下恢复{\epsilon}-最优排序的算法，同时允许每个模型响应多达O(n)扰动的成对比较结果。此外，我们展示了在部分观察设置下的健壮恢复结果。我们的实验证实了我们的算法

    arXiv:2403.02745v1 Announce Type: new  Abstract: This paper addresses the challenges of aligning large language models (LLMs) with human values via preference learning (PL), with a focus on the issues of incomplete and corrupted data in preference datasets. We propose a novel method for robustly and completely recalibrating values within these datasets to enhance LLMs resilience against the issues. In particular, we devise a guaranteed polynomial time ranking algorithm that robustifies several existing models, such as the classic Bradley--Terry--Luce (BTL) (Bradley and Terry, 1952) model and certain generalizations of it. To the best of our knowledge, our present work is the first to propose an algorithm that provably recovers an {\epsilon}-optimal ranking with high probability while allowing as large as O(n) perturbed pairwise comparison results per model response. Furthermore, we show robust recovery results in the partially observed setting. Our experiments confirm that our algorith
    
[^3]: LLMs用于新闻标题的有针对性情感分析：探索不同级别的提示规范化

    LLMs for Targeted Sentiment in News Headlines: Exploring Different Levels of Prompt Prescriptiveness

    [https://arxiv.org/abs/2403.00418](https://arxiv.org/abs/2403.00418)

    提出了探索LLMs在新闻标题有针对性情感分析中不同级别提示规范性的方法，以提高其性能。

    

    新闻标题常常通过有意识地以特定方式描绘实体来引发情感，这使得新闻标题的有针对性情感分析(TSA)成为一项值得做但具有挑战的任务。微调的编码器模型展现出令人满意的TSA性能，但它们的背景知识有限，需要有标记的数据集。LLMs由于其广泛的语言和世界知识以及上下文学习能力，为TSA提供了一个潜在的通用解决方案，然而它们的性能受提示设计的影响很大。通过与主观任务的注释范式进行类比，我们探讨了提示设计对LLMs在新闻标题TSA中性能的影响。我们评估了使用不同级别的规范提示（从纯粹的零样本到符合注释指南的精心准备的少样本提示）的最先进LLMs的预测准确性。认识到TSA的主观性质，我们评估

    arXiv:2403.00418v1 Announce Type: new  Abstract: News headlines often evoke sentiment by intentionally portraying entities in particular ways, making targeted sentiment analysis (TSA) of headlines a worthwhile but difficult task. Fine-tuned encoder models show satisfactory TSA performance, but their background knowledge is limited, and they require a labeled dataset. LLMs offer a potentially universal solution for TSA due to their broad linguistic and world knowledge along with in-context learning abilities, yet their performance is heavily influenced by prompt design. Drawing parallels with annotation paradigms for subjective tasks, we explore the influence of prompt design on the performance of LLMs for TSA of news headlines. We evaluate the predictive accuracy of state-of-the-art LLMs using prompts with different levels of prescriptiveness, ranging from plain zero-shot to elaborate few-shot prompts matching annotation guidelines. Recognizing the subjective nature of TSA, we evaluate
    
[^4]: 失言：多轮对话中大型语言模型的安全漏洞

    Speak Out of Turn: Safety Vulnerability of Large Language Models in Multi-turn Dialogue

    [https://arxiv.org/abs/2402.17262](https://arxiv.org/abs/2402.17262)

    本论文探讨了多轮对话中大型语言模型的安全性漏洞，指出人类可以通过多轮对话诱使其生成有害信息。

    

    大型语言模型(LLMs)已被证明在面临"越狱"时会产生非法或不道德的回应。 "越狱"研究强调了LLMs的安全问题。然而，先前的研究主要集中在单轮对话上，忽视了多轮对话可能带来的复杂性和风险，这是人类从LLMs获取信息的关键方式。本文认为人类可以利用多轮对话诱使LLMs生成有害信息。LLMs可能不会拒绝警告性或边界不安全的查询，即使在多轮对话中每个回合都被服务于一个恶意目的。因此，通过将一个不安全查询分解为多个子查询用于多轮对话，我们逐渐诱使LLMs回答有害的子问题，最终导致总体有害响应。我们的实验跨越了广泛的范围。

    arXiv:2402.17262v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have been demonstrated to generate illegal or unethical responses, particularly when subjected to "jailbreak." Research on jailbreak has highlighted the safety issues of LLMs. However, prior studies have predominantly focused on single-turn dialogue, ignoring the potential complexities and risks presented by multi-turn dialogue, a crucial mode through which humans derive information from LLMs. In this paper, we argue that humans could exploit multi-turn dialogue to induce LLMs into generating harmful information. LLMs may not intend to reject cautionary or borderline unsafe queries, even if each turn is closely served for one malicious purpose in a multi-turn dialogue. Therefore, by decomposing an unsafe query into several sub-queries for multi-turn dialogue, we induced LLMs to answer harmful sub-questions incrementally, culminating in an overall harmful response. Our experiments, conducted across a wide ra
    
[^5]: 以显式奖励的噪声对比对齐语言模型

    Noise Contrastive Alignment of Language Models with Explicit Rewards

    [https://arxiv.org/abs/2402.05369](https://arxiv.org/abs/2402.05369)

    本文提出了一个基于噪声对比估计的通用LM对齐框架，能够处理明确注释的奖励数据，并且扩展了当前的对齐理论。

    

    用户意图通常被形式化为需要在微调语言模型时最大化的评估奖励。现有的对齐方法，如直接优化偏好（DPO），主要适用于隐含定义而非明确给定奖励的两两偏好数据。在本文中，我们引入了一个通用的LM对齐框架，利用噪声对比估计（NCE）来解决明确注释有标量评估的奖励数据处理的差距。我们的框架包括两个并行算法，NCA和InfoNCA，两者都能从奖励数据和偏好数据中直接提取LM策略。值得注意的是，我们证明了DPO损失是我们提出的InfoNCA目标在两两偏好设置下的特殊情况，从而集成和扩展了当前的对齐理论。通过对比NCA和InfoNCA，我们展示了InfoNCA和DPO如何在不同响应对于单个指令的相对可能性上进行调整。

    User intentions are typically formalized as evaluation rewards to be maximized when fine-tuning language models (LMs). Existing alignment methods, such as Direct Preference Optimization (DPO), are mainly tailored for pairwise preference data where rewards are implicitly defined rather than explicitly given. In this paper, we introduce a general framework for LM alignment, leveraging Noise Contrastive Estimation (NCE) to bridge the gap in handling reward datasets explicitly annotated with scalar evaluations. Our framework comprises two parallel algorithms, NCA and InfoNCA, both enabling the direct extraction of an LM policy from reward data as well as preference data. Notably, we show that the DPO loss is a special case of our proposed InfoNCA objective under pairwise preference settings, thereby integrating and extending current alignment theories. By contrasting NCA and InfoNCA, we show that InfoNCA and DPO adjust relative likelihood across different responses to a single instruction,
    
[^6]: 比较基于模板和非模板语言模型的探测方法

    Comparing Template-based and Template-free Language Model Probing

    [https://arxiv.org/abs/2402.00123](https://arxiv.org/abs/2402.00123)

    本文比较了基于模板和非模板语言模型的探测方法，发现它们在模型排名、绝对得分和与困惑度的关系等方面存在差异。

    

    以专家制作的模板和自然发生的文本为基础的语言模型探测方法的差异经常被忽视。在这里，我们评估了16种不同的语言模型在10个英文探测数据集上的性能，其中包括4个基于模板的和6个非模板的数据集，并针对以下研究问题进行了回答：（RQ1）模型排名在两种方法中是否不同？（RQ2）模型的绝对得分在两种方法中是否不同？（RQ3）RQ1和RQ2的答案在一般和领域特定模型之间是否不同？我们的发现是：1）除了顶级的领域特定模型外，基于模板和非模板方法通常排名不同。2）与平行的非模板和模板提示相比，准确度下降了最多42%。3）在非模板方法中，困惑度与准确度呈负相关，但是在基于模板的探测中，它们呈正相关，这与直觉相反。4）模型倾向于预测相同的内容。

    The differences between cloze-task language model (LM) probing with 1) expert-made templates and 2) naturally-occurring text have often been overlooked. Here, we evaluate 16 different LMs on 10 probing English datasets -- 4 template-based and 6 template-free -- in general and biomedical domains to answer the following research questions: (RQ1) Do model rankings differ between the two approaches? (RQ2) Do models' absolute scores differ between the two approaches? (RQ3) Do the answers to RQ1 and RQ2 differ between general and domain-specific models? Our findings are: 1) Template-free and template-based approaches often rank models differently, except for the top domain-specific models. 2) Scores decrease by up to 42% Acc@1 when comparing parallel template-free and template-based prompts. 3) Perplexity is negatively correlated with accuracy in the template-free approach, but, counter-intuitively, they are positively correlated for template-based probing. 4) Models tend to predict the same
    
[^7]: 大型语言模型通过语言特征对齐可以增强说服力

    Large language models can enhance persuasion through linguistic feature alignment

    [https://arxiv.org/abs/2311.16466](https://arxiv.org/abs/2311.16466)

    本研究调查了大型语言模型对人类沟通的影响，使用了消费者金融投诉数据，并发现大型语言模型的使用可能增强了一整套语言特征，提高了信息说服力。

    

    尽管大型语言模型 (LLMs)正在重新塑造人类生活的各个方面，但我们对它们的影响的理解仍然有些受限。本文研究了LLMs对人类沟通的影响，使用了消费者金融投诉的数据。通过对消费者金融保护局 (CFPB) 收集的超过820,000个投诉进行AI检测，我们发现在ChatGPT发布后不久，LLMs的使用可能性急剧增加。此外，LLMs的使用可能性与信息说服力（即从金融公司获得救济的可能性增加）呈正相关。计算语言分析表明，这种正相关可能是由LLMs增强了各种语言特征所解释的。根据这些观察研究的结果，我们假设LLMs的使用可能增强了一整套语言特征，提高了对具有不同语言背景的接收者的信息说服力。

    Although large language models (LLMs) are reshaping various aspects of human life, our current understanding of their impacts remains somewhat constrained. Here we investigate the impact of LLMs on human communication, using data on consumer complaints in the financial industry. By employing an AI detection tool on more than 820K complaints gathered by the Consumer Financial Protection Bureau (CFPB), we find a sharp increase in the likely use of LLMs shortly after the release of ChatGPT. Moreover, the likely LLM usage was positively correlated with message persuasiveness (i.e., increased likelihood of obtaining relief from financial firms). Computational linguistic analyses suggest that the positive correlation may be explained by LLMs' enhancement of various linguistic features. Based on the results of these observational studies, we hypothesize that LLM usage may enhance a comprehensive set of linguistic features, increasing message persuasiveness to receivers with heterogeneous ling
    
[^8]: 最近在仇恨言论审核方面的进展：多模态和大型模型的作用

    Recent Advances in Hate Speech Moderation: Multimodality and the Role of Large Models. (arXiv:2401.16727v1 [cs.CL])

    [http://arxiv.org/abs/2401.16727](http://arxiv.org/abs/2401.16727)

    这项综合调查总结了最近在仇恨言论审核方面的进展，重点介绍了大型语言模型和大型多模态模型的作用。研究发现了文本、视觉和听觉元素在传播仇恨言论中的微妙相互作用，并强调了大型模型对审核能力的重新定义。同时，研究还指出了在少数语言和文化背景下的研究差距和处理低资源环境的需求。

    

    在网络交流的不断发展中，审核仇恨言论（HS）面临着复杂的挑战，这是由数字内容的多模态特性所带来的。这项综合调查深入研究了HS审核的最新进展，着重介绍了大型语言模型（LLMs）和大型多模态模型（LMMs）的崛起角色。我们的研究从对当前文献的全面分析开始，揭示了文本、视觉和听觉元素在传播HS中的微妙相互作用。我们发现了一个明显的趋势，即将这些模态整合在一起，主要是因为HS的传播具有复杂性和微妙性。对于由LLMs和LMMs带来的进展，我们特别强调了其对检测和审核能力边界的重新定义。我们确定了研究中存在的现有差距，特别是在少数语言和文化的背景下，以及在处理低资源环境中需要解决方案的需求。

    In the evolving landscape of online communication, moderating hate speech (HS) presents an intricate challenge, compounded by the multimodal nature of digital content. This comprehensive survey delves into the recent strides in HS moderation, spotlighting the burgeoning role of large language models (LLMs) and large multimodal models (LMMs). Our exploration begins with a thorough analysis of current literature, revealing the nuanced interplay between textual, visual, and auditory elements in propagating HS. We uncover a notable trend towards integrating these modalities, primarily due to the complexity and subtlety with which HS is disseminated. A significant emphasis is placed on the advances facilitated by LLMs and LMMs, which have begun to redefine the boundaries of detection and moderation capabilities. We identify existing gaps in research, particularly in the context of underrepresented languages and cultures, and the need for solutions to handle low-resource settings. The survey
    
[^9]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^10]: ChatKBQA: 一个基于精调大型语言模型的生成-检索框架用于知识库问答

    ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models. (arXiv:2310.08975v1 [cs.CL])

    [http://arxiv.org/abs/2310.08975](http://arxiv.org/abs/2310.08975)

    ChatKBQA是一个基于精调大型语言模型的生成-检索框架，用于改进知识库问答的效率和准确性，实验结果显示在多个数据集上取得了新的最好表现。

    

    知识库问答（KBQA）旨在通过大规模知识库（KB）获取自然语言问题的答案，通常分为两个研究组成部分：知识检索和语义解析。然而，仍然存在三个核心挑战，包括低效的知识检索、检索错误对语义解析的不利影响以及之前的KBQA方法的复杂性。在大型语言模型（LLM）时代，我们介绍了ChatKBQA，这是一个新颖的基于精调开源LLMs（如Llama-2、ChatGLM2和Baichuan2）构建的生成-检索KBQA框架。ChatKBQA提议首先使用精调的LLMs生成逻辑形式，然后通过无监督检索方法检索和替换实体和关系，从而更直观地改进了生成和检索。实验结果表明，ChatKBQA在标准KBQA数据集WebQSP和ComplexWebQuestions (CWQ)上取得了新的最先进性能。

    Knowledge Base Question Answering (KBQA) aims to derive answers to natural language questions over large-scale knowledge bases (KBs), which are generally divided into two research components: knowledge retrieval and semantic parsing. However, three core challenges remain, including inefficient knowledge retrieval, retrieval errors adversely affecting semantic parsing, and the complexity of previous KBQA methods. In the era of large language models (LLMs), we introduce ChatKBQA, a novel generate-then-retrieve KBQA framework built on fine-tuning open-source LLMs such as Llama-2, ChatGLM2 and Baichuan2. ChatKBQA proposes generating the logical form with fine-tuned LLMs first, then retrieving and replacing entities and relations through an unsupervised retrieval method, which improves both generation and retrieval more straightforwardly. Experimental results reveal that ChatKBQA achieves new state-of-the-art performance on standard KBQA datasets, WebQSP, and ComplexWebQuestions (CWQ). This
    
[^11]: Text2NKG: 面向N元关系知识图构建的细粒度N元关系抽取

    Text2NKG: Fine-Grained N-ary Relation Extraction for N-ary relational Knowledge Graph Construction. (arXiv:2310.05185v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.05185](http://arxiv.org/abs/2310.05185)

    Text2NKG是一种用于构建N元关系知识图的细粒度N元关系抽取框架，支持多种NKG模式，具有高灵活性和实用性。

    

    除了传统的二元关系事实外，N元关系知识图(NKGs)由包含两个以上实体的N元关系事实组成，更接近于具有广泛应用的真实世界事实。然而，NKG的构建仍然严重依赖于人工劳动，并且N元关系抽取仍然停留在粗粒度水平，通常是在单一模式和固定的实体数量上操作。为了解决这些限制，我们提出了Text2NKG，一种新颖的面向N元关系知识图构建的细粒度N元关系抽取框架。我们引入了一种跨度元组分类方法，并采用异构排序合并来实现不同度的细粒度N元关系抽取。此外，Text2NKG支持四种典型的NKG模式：超关系模式、基于事件的模式、基于角色的模式和超图模式，具有较高的灵活性和实用性。实验结果表明，Text2NKG的表现优于传统的N元关系抽取方法。

    Beyond traditional binary relational facts, n-ary relational knowledge graphs (NKGs) are comprised of n-ary relational facts containing more than two entities, which are closer to real-world facts with broader applications. However, the construction of NKGs still significantly relies on manual labor, and n-ary relation extraction still remains at a course-grained level, which is always in a single schema and fixed arity of entities. To address these restrictions, we propose Text2NKG, a novel fine-grained n-ary relation extraction framework for n-ary relational knowledge graph construction. We introduce a span-tuple classification approach with hetero-ordered merging to accomplish fine-grained n-ary relation extraction in different arity. Furthermore, Text2NKG supports four typical NKG schemas: hyper-relational schema, event-based schema, role-based schema, and hypergraph-based schema, with high flexibility and practicality. Experimental results demonstrate that Text2NKG outperforms the
    
[^12]: 模型告诉你该丢弃什么：适应性KV缓存压缩用于LLMs

    Model Tells You What to Discard: Adaptive KV Cache Compression for LLMs. (arXiv:2310.01801v1 [cs.CL])

    [http://arxiv.org/abs/2310.01801](http://arxiv.org/abs/2310.01801)

    该研究提出了一种自适应的KV缓存压缩方法，用于减少大型语言模型的内存消耗。通过有针对性的分析和结构识别，构建了具有自适应性的KV缓存，通过清除和丢弃特定的上下文，以及只对特定的注意力头使用标准KV缓存，实现了显著的内存占用减少。

    

    在这项研究中，我们引入了一种自适应的KV缓存压缩方法，它可以减少大型语言模型（LLMs）生成推理的内存占用。与传统的KV缓存不同，我们通过有针对性的分析来识别注意力模块的内在结构。基于识别出的结构，我们以自适应的方式构建KV缓存：在强调本地上下文的注意力头上清除长距离上下文，在以特殊标记为中心的注意力头上丢弃非特殊标记，并且仅对广泛关注所有标记的注意力头使用标准的KV缓存。此外，通过使用轻量级的注意力分析来指导自适应KV缓存的构建，FastGen可以在不需要资源密集型的微调或重新训练的情况下部署。在各种任务的实验中，FastGen在GPU内存消耗方面显示出了显著的减少。

    In this study, we introduce adaptive KV cache compression, a plug-and-play method that reduces the memory footprint of generative inference for Large Language Models (LLMs). Different from the conventional KV cache that retains key and value vectors for all context tokens, we conduct targeted profiling to discern the intrinsic structure of attention modules. Based on the recognized structure, we then construct the KV cache in an adaptive manner: evicting long-range contexts on attention heads emphasizing local contexts, discarding non-special tokens on attention heads centered on special tokens, and only employing the standard KV cache for attention heads that broadly attend to all tokens. Moreover, with the lightweight attention profiling used to guide the construction of the adaptive KV cache, FastGen can be deployed without resource-intensive fine-tuning or re-training. In our experiments across various asks, FastGen demonstrates substantial reduction on GPU memory consumption with 
    
[^13]: 联邦式大规模语言模型：一个立场论文

    Federated Large Language Model: A Position Paper. (arXiv:2307.08925v1 [cs.LG])

    [http://arxiv.org/abs/2307.08925](http://arxiv.org/abs/2307.08925)

    我们提出了联邦式大规模语言模型的概念，通过联邦学习实现分散数据的共同训练共享模型，以应对公共数据可用性的限制和私有数据的隐私保护需求。我们讨论了预训练、微调和提示工程这三个组件的优势，并提出了实施策略。同时，我们探讨了FL和LLM集成带来的新挑战，并分析了现有解决方案和潜在障碍。

    

    大规模语言模型（LLM）在各个领域获得了相当大的关注并找到了多样化的应用，但在真实场景中开发时面临挑战。这些挑战源于公共领域数据可用性的匮乏以及对私有领域数据的隐私保护需求。为了解决这些问题，联邦学习（FL）作为一项有前景的技术出现了，它能够在保持分散数据的同时实现共同训练共享模型。我们提出了联邦式LLM的概念，包括三个关键组成部分，即联邦式LLM预训练、联邦式LLM微调和联邦式LLM提示工程。对于每个组件，我们讨论了它相对于传统LLM训练方法的优势，并提出了具体的工程策略来实施。此外，我们探讨了FL和LLM集成带来的新挑战。我们分析现有的解决方案并确定可能的障碍

    Large scale language models (LLM) have received significant attention and found diverse applications across various domains, but their development encounters challenges in real-world scenarios. These challenges arise due to the scarcity of public domain data availability and the need to maintain privacy with respect to private domain data. To address these issues, federated learning (FL) has emerged as a promising technology that enables collaborative training of shared models while preserving decentralized data. We propose the concept of federated LLM, which comprises three key components, i.e., federated LLM pre-training, federated LLM fine-tuning, and federated LLM prompt engineering. For each component, we discuss its advantage over traditional LLM training methods and propose specific engineering strategies for implementation. Furthermore, we explore the novel challenges introduced by the integration of FL and LLM. We analyze existing solutions and identify potential obstacles fac
    
[^14]: 面向领域扩展的ASTE：重新审视情感三元组提取中的泛化问题

    Domain-Expanded ASTE: Rethinking Generalization in Aspect Sentiment Triplet Extraction. (arXiv:2305.14434v1 [cs.CL])

    [http://arxiv.org/abs/2305.14434](http://arxiv.org/abs/2305.14434)

    该论文提出了一个领域扩展的ASTE基准数据集，通过生成方法来解决领域泛化的问题。

    

    面向领域扩展的Aspect Sentiment Triplet Extraction (ASTE) 是Aspect-Based Sentiment Analysis (ABSA) 中的一个子任务，考虑每个观点术语、它们表达的情感及相应的方面目标。然而，现有方法仅限于两个领域内的场景。因此，我们提出了一个领域扩展的基准数据集，以应对领域内、领域间和跨领域的情况。我们基于酒店和化妆品评论，标注了超过4000个数据样本来支持新的基准数据集。我们对五种现有的方法进行了分析，结果表明，尽管领域内和领域外的性能存在显著差距，但生成方法在领域泛化方面具有很强的潜力。我们的数据集、代码实现和模型均可在https://github.com/DAMO-NLP-SG/domain-expanded-aste 上获得。

    Aspect Sentiment Triplet Extraction (ASTE) is a subtask of Aspect-Based Sentiment Analysis (ABSA) that considers each opinion term, their expressed sentiment, and the corresponding aspect targets. However, existing methods are limited to the in-domain setting with two domains. Hence, we propose a domain-expanded benchmark to address the in-domain, out-of-domain and cross-domain settings. We support the new benchmark by annotating more than 4000 data samples for two new domains based on hotel and cosmetics reviews. Our analysis of five existing methods shows that while there is a significant gap between in-domain and out-of-domain performance, generative methods have a strong potential for domain generalization. Our datasets, code implementation and models are available at https://github.com/DAMO-NLP-SG/domain-expanded-aste .
    
[^15]: 超越负采样的高效分布式表示方法

    Efficient distributed representations beyond negative sampling. (arXiv:2303.17475v1 [cs.LG])

    [http://arxiv.org/abs/2303.17475](http://arxiv.org/abs/2303.17475)

    本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。

    

    本文介绍了一种高效的学习分布式表示（也称为嵌入）的方法。该方法通过最小化一个类似于Word2Vec算法中引入并在多个工作中采用的目标函数来实现。优化计算的瓶颈是softmax归一化常数的计算，这需要与样本大小呈二次比例的操作数。这种复杂度不适用于大型数据集，所以负采样是一个常见的解决方法，可以在与样本大小线性相关的时间内获得分布式表示。然而，负采样会改变损失函数，因此解决的是与最初提出的不同的优化问题。我们的贡献在于展示如何通过线性时间估计softmax归一化常数，从而设计了一种有效的优化策略来学习分布式表示。我们使用不同的数据集进行测试，并展示了我们的方法在嵌入质量和训练时间方面优于负采样。

    This article describes an efficient method to learn distributed representations, also known as embeddings. This is accomplished minimizing an objective function similar to the one introduced in the Word2Vec algorithm and later adopted in several works. The optimization computational bottleneck is the calculation of the softmax normalization constants for which a number of operations scaling quadratically with the sample size is required. This complexity is unsuited for large datasets and negative sampling is a popular workaround, allowing one to obtain distributed representations in linear time with respect to the sample size. Negative sampling consists, however, in a change of the loss function and hence solves a different optimization problem from the one originally proposed. Our contribution is to show that the sotfmax normalization constants can be estimated in linear time, allowing us to design an efficient optimization strategy to learn distributed representations. We test our ap
    

