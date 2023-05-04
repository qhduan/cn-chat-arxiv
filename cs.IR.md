# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generating Synthetic Documents for Cross-Encoder Re-Rankers: A Comparative Study of ChatGPT and Human Experts.](http://arxiv.org/abs/2305.02320) | 本研究探讨了使用大型语言模型生成合成文档，作为交叉编码器重新排序的训练数据的有效性。并引入了一个新的数据集ChatGPT-RetrievalQA，最终发现ChatGPT反应训练的交叉编码器重排模型比使用人类生成数据的模型更有效。 |
| [^2] | [Uncovering ChatGPT's Capabilities in Recommender Systems.](http://arxiv.org/abs/2305.02182) | 本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。 |
| [^3] | [Zero-Shot Listwise Document Reranking with a Large Language Model.](http://arxiv.org/abs/2305.02156) | 本文提出了一种基于大型语言模型的列表式重新排序器，可以在没有特定任务训练数据的情况下实现强大的重新排序效果，并在实验中取得了令人满意的结果。 |
| [^4] | [Understanding Differential Search Index for Text Retrieval.](http://arxiv.org/abs/2305.02073) | DSI是一种可微分搜索指数信息检索框架，虽然其能够生成文件标识符的排序列表，但在区分相关文档和随机文档方面表现不佳，研究人员提出了一种多任务蒸馏方法以提高其检索质量。 |
| [^5] | [Denoising Multi-modal Sequential Recommenders with Contrastive Learning.](http://arxiv.org/abs/2305.01915) | 本文提出一种名为Demure的系统，采用对比学习来去除多模态推荐系统中的潜在噪音，以此提高模型效果。 |
| [^6] | [Pre-train and Search: Efficient Embedding Table Sharding with Pre-trained Neural Cost Models.](http://arxiv.org/abs/2305.01868) | 本文探索了一种基于预训练成本模型的高效分片方法，通过神经网络预测成本，并使用在线搜索确定最佳分片计划，实验结果表明其在嵌入表分片任务中表现很好。 |
| [^7] | [Towards Imperceptible Document Manipulations against Neural Ranking Models.](http://arxiv.org/abs/2305.01860) | 该论文提出了一种针对神经排序模型的不易被检测到的对抗性攻击框架，称为“几乎不可察觉文档操作”（IDEM）。IDEM使用生成语言模型生成连结句，无法引入易于检测的错误，并且使用单独的位置合并策略来平衡扰动文本的相关性和连贯性，实验结果表明，IDEM可以在保持高人类评估得分的同时优于强基线。 |
| [^8] | [When Newer is Not Better: Does Deep Learning Really Benefit Recommendation From Implicit Feedback?.](http://arxiv.org/abs/2305.01801) | 本研究对多个神经推荐模型与传统模型进行比较，提出了一组评估策略来衡量其记忆性能、泛化性能和子群特定性能，揭示了在IMDB和Yelp数据集上，神经推荐模型与传统模型的差异性。 |
| [^9] | [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?.](http://arxiv.org/abs/2305.01555) | 本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。 |
| [^10] | [Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond.](http://arxiv.org/abs/2304.13874) | 本研究利用一个名为ConvSim的用户模拟器来评估用户反馈，从而提高会话式搜索的性能，实验结果显示有效利用用户反馈可以大幅提高检索性能。 |
| [^11] | [Do you MIND? Reflections on the MIND dataset for research on diversity in news recommendations.](http://arxiv.org/abs/2304.08253) | 该研究分析了MIND数据集在新闻推荐多样性研究中的适用性，发现虽然是一个很好的进步，但仍有很大的改进空间。 |
| [^12] | [Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited.](http://arxiv.org/abs/2303.13835) | 推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。 |
| [^13] | [Product Question Answering in E-Commerce: A Survey.](http://arxiv.org/abs/2302.08092) | 电商PQA的研究面临着问题多、数据难收集、答案不确定等特殊挑战。本文系统地综述了PQA研究的现状与未来方向。 |
| [^14] | [How Bad is Top-$K$ Recommendation under Competing Content Creators?.](http://arxiv.org/abs/2302.01971) | 本文基于随机效用模型，研究了内容创作者在Top-K推荐下的竞争影响，证明了用户福利损失受小常数上界影响。 |
| [^15] | [Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning.](http://arxiv.org/abs/2212.03760) | 本文研究发现，用户历史语言建模可以在不同推荐任务中取得优异结果，并且利用任务无关的用户历史还可以提供显著的性能优势。该方法具有广泛的现实世界迁移学习能力。 |

# 详细

[^1]: 产生用于交叉编码器的合成文档: ChatGPT和人类专家的比较研究。

    Generating Synthetic Documents for Cross-Encoder Re-Rankers: A Comparative Study of ChatGPT and Human Experts. (arXiv:2305.02320v1 [cs.IR])

    [http://arxiv.org/abs/2305.02320](http://arxiv.org/abs/2305.02320)

    本研究探讨了使用大型语言模型生成合成文档，作为交叉编码器重新排序的训练数据的有效性。并引入了一个新的数据集ChatGPT-RetrievalQA，最终发现ChatGPT反应训练的交叉编码器重排模型比使用人类生成数据的模型更有效。

    

    我们探讨了生成大型语言模型（LLMs）在产生训练数据方面的有用性，以供交叉编码器的重新排序进行对比研究，从而产生了合成文档而不是合成查询的新方向。我们引入了一个新的数据集，ChatGPT-RetrievalQA，并比较了在LLM-generated和human-generated数据上进行微调的模型的有效性。使用生成的LLMs数据可以用于增加训练数据，特别是在标记数据较少的领域中。我们基于现有数据集，即由公共问题集和ChatGPT的人类回答和答案组成的人类ChatGPT比较语料库（HC3）构建了ChatGPT-RetrievalQA。我们在MS MARCO DEV，TREC DL'19和TREC DL'20上进行评估，结果表明在ChatGPT响应方面受过训练的交叉编码器重新排序模型是零-shot重新排序器比那些接受了人类生成数据的有效性显著更高的。

    We investigate the usefulness of generative Large Language Models (LLMs) in generating training data for cross-encoder re-rankers in a novel direction: generating synthetic documents instead of synthetic queries. We introduce a new dataset, ChatGPT-RetrievalQA, and compare the effectiveness of models fine-tuned on LLM-generated and human-generated data. Data generated with generative LLMs can be used to augment training data, especially in domains with smaller amounts of labeled data. We build ChatGPT-RetrievalQA based on an existing dataset, human ChatGPT Comparison Corpus (HC3), consisting of public question collections with human responses and answers from ChatGPT. We fine-tune a range of cross-encoder re-rankers on either human-generated or ChatGPT-generated data. Our evaluation on MS MARCO DEV, TREC DL'19, and TREC DL'20 demonstrates that cross-encoder re-ranking models trained on ChatGPT responses are statistically significantly more effective zero-shot re-rankers than those trai
    
[^2]: 揭示ChatGPT在推荐系统中的能力

    Uncovering ChatGPT's Capabilities in Recommender Systems. (arXiv:2305.02182v1 [cs.IR])

    [http://arxiv.org/abs/2305.02182](http://arxiv.org/abs/2305.02182)

    本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析，在四个不同领域的数据集上进行大量实验并发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型，在列表排名中能够达到成本和性能最佳平衡。

    

    ChatGPT的问答功能吸引了自然语言处理（NLP）界及外界的关注。为了测试ChatGPT在推荐方面的表现，本研究从信息检索（IR）的角度出发，对ChatGPT在点、对、列表三种排名策略下的推荐能力进行了实证分析。通过在不同领域的四个数据集上进行大量实验，我们发现ChatGPT在三种排名策略下的表现均优于其他大型语言模型。基于单位成本改进的分析，我们确定ChatGPT在列表排名中能够在成本和性能之间实现最佳平衡，而在对和点排名中表现相对较弱。

    The debut of ChatGPT has recently attracted the attention of the natural language processing (NLP) community and beyond. Existing studies have demonstrated that ChatGPT shows significant improvement in a range of downstream NLP tasks, but the capabilities and limitations of ChatGPT in terms of recommendations remain unclear. In this study, we aim to conduct an empirical analysis of ChatGPT's recommendation ability from an Information Retrieval (IR) perspective, including point-wise, pair-wise, and list-wise ranking. To achieve this goal, we re-formulate the above three recommendation policies into a domain-specific prompt format. Through extensive experiments on four datasets from different domains, we demonstrate that ChatGPT outperforms other large language models across all three ranking policies. Based on the analysis of unit cost improvements, we identify that ChatGPT with list-wise ranking achieves the best trade-off between cost and performance compared to point-wise and pair-wi
    
[^3]: 基于大型语言模型的零样本列表式文档重新排序

    Zero-Shot Listwise Document Reranking with a Large Language Model. (arXiv:2305.02156v1 [cs.IR])

    [http://arxiv.org/abs/2305.02156](http://arxiv.org/abs/2305.02156)

    本文提出了一种基于大型语言模型的列表式重新排序器，可以在没有特定任务训练数据的情况下实现强大的重新排序效果，并在实验中取得了令人满意的结果。

    

    基于双编码器或交叉编码器结构的监督排序方法已经在多阶段文本排序任务中取得了成功，但是它们需要大量相关性判断作为训练数据。在本文中，我们提出了一种使用大型语言模型的列表式重新排序器(LRL)，它可以在不使用任何特定任务训练数据的情况下实现强大的重新排序效果。LRL与现有的点式排名方法不同，在点式排名方法中，文档是独立得分并按分数排名，而LRL直接生成给定候选文档的重新排序文档标识符列表。在三个TREC网络搜索数据集上的实验表明，LRL不仅可以在重新排序第一阶段检索结果时优于零样本点式方法，还可以作为点式方法的最终阶段重新排序器，以提高效率并提高前几名的结果。此外，我们将我们的方法应用于MIRACL的子集，这是最近的一个多语言检索数据集，获得了令人满意的结果。

    Supervised ranking methods based on bi-encoder or cross-encoder architectures have shown success in multi-stage text ranking tasks, but they require large amounts of relevance judgments as training data. In this work, we propose Listwise Reranker with a Large Language Model (LRL), which achieves strong reranking effectiveness without using any task-specific training data. Different from the existing pointwise ranking methods, where documents are scored independently and ranked according to the scores, LRL directly generates a reordered list of document identifiers given the candidate documents. Experiments on three TREC web search datasets demonstrate that LRL not only outperforms zero-shot pointwise methods when reranking first-stage retrieval results, but can also act as a final-stage reranker to improve the top-ranked results of a pointwise method for improved efficiency. Additionally, we apply our approach to subsets of MIRACL, a recent multilingual retrieval dataset, with results 
    
[^4]: 了解文本检索的可微分搜索指数

    Understanding Differential Search Index for Text Retrieval. (arXiv:2305.02073v1 [cs.IR])

    [http://arxiv.org/abs/2305.02073](http://arxiv.org/abs/2305.02073)

    DSI是一种可微分搜索指数信息检索框架，虽然其能够生成文件标识符的排序列表，但在区分相关文档和随机文档方面表现不佳，研究人员提出了一种多任务蒸馏方法以提高其检索质量。

    

    可微分搜索指数（DSI）是一种新颖的信息检索框架，利用可微分函数对给定查询生成一个文件标识符的排序列表。然而，由于端到端神经架构的黑盒特性，仍需了解DSI具备基本索引和检索能力的程度。为了弥补这一差距，在本研究中，我们定义并检验了一个有效的IR框架应具备的三个重要能力，即排他性、完整性和相关性排序。我们的分析实验证明，虽然DSI在记忆从伪查询到文档标识符的单向映射方面表现出熟练程度，但在区分相关文档和随机文档方面效果不佳，从而对其检索效果产生负面影响。为了解决这个问题，我们提出了一种多任务蒸馏方法，以提高检索质量而不改变结构。

    The Differentiable Search Index (DSI) is a novel information retrieval (IR) framework that utilizes a differentiable function to generate a sorted list of document identifiers in response to a given query. However, due to the black-box nature of the end-to-end neural architecture, it remains to be understood to what extent DSI possesses the basic indexing and retrieval abilities. To mitigate this gap, in this study, we define and examine three important abilities that a functioning IR framework should possess, namely, exclusivity, completeness, and relevance ordering. Our analytical experimentation shows that while DSI demonstrates proficiency in memorizing the unidirectional mapping from pseudo queries to document identifiers, it falls short in distinguishing relevant documents from random ones, thereby negatively impacting its retrieval effectiveness. To address this issue, we propose a multi-task distillation approach to enhance the retrieval quality without altering the structure o
    
[^5]: 基于对比学习的去噪多模态推荐系统

    Denoising Multi-modal Sequential Recommenders with Contrastive Learning. (arXiv:2305.01915v1 [cs.IR])

    [http://arxiv.org/abs/2305.01915](http://arxiv.org/abs/2305.01915)

    本文提出一种名为Demure的系统，采用对比学习来去除多模态推荐系统中的潜在噪音，以此提高模型效果。

    

    目前，将多模态数据用于推荐系统中的用户建模已成为一个快速增长的研究领域。现有的多媒体推荐器通过整合各种模态并设计精细的模块，取得了显著的改进。但是，当用户决定与物品进行交互时，他们大多数并没有完全阅读所有模态的内容。我们称直接导致用户行为的模态为兴趣点，这些点是捕捉用户兴趣的重要方面。与之相反，不直接导致用户行为的模态则可能是潜在的噪音，并可能误导推荐模型的学习。由于无法访问用户显式反馈其兴趣点，因此文献中很少有研究致力于去除这些潜在的噪音。为了填补这一空白，我们提出一种基于对比学习的弱监督框架，用于去噪多模态推荐系统（称为Demure）。

    There is a rapidly-growing research interest in engaging users with multi-modal data for accurate user modeling on recommender systems. Existing multimedia recommenders have achieved substantial improvements by incorporating various modalities and devising delicate modules. However, when users decide to interact with items, most of them do not fully read the content of all modalities. We refer to modalities that directly cause users' behaviors as point-of-interests, which are important aspects to capture users' interests. In contrast, modalities that do not cause users' behaviors are potential noises and might mislead the learning of a recommendation model. Not surprisingly, little research in the literature has been devoted to denoising such potential noises due to the inaccessibility of users' explicit feedback on their point-of-interests. To bridge the gap, we propose a weakly-supervised framework based on contrastive learning for denoising multi-modal recommenders (dubbed Demure). 
    
[^6]: 预训练和搜索：基于预训练神经成本模型的高效嵌入表分片方法

    Pre-train and Search: Efficient Embedding Table Sharding with Pre-trained Neural Cost Models. (arXiv:2305.01868v1 [cs.LG])

    [http://arxiv.org/abs/2305.01868](http://arxiv.org/abs/2305.01868)

    本文探索了一种基于预训练成本模型的高效分片方法，通过神经网络预测成本，并使用在线搜索确定最佳分片计划，实验结果表明其在嵌入表分片任务中表现很好。

    

    在分布式训练中，将大型机器学习模型分片到多个设备上以平衡成本非常重要。由于分区是NP难问题且准确和高效地估算成本很困难，因此这是具有挑战性的。本文探索了一种“预训练和搜索”范式，用于实现高效的分片。该方法是预先训练一个通用的、永久存在的神经网络，来预测所有可能的分片的成本，这个网络就是一个高效的分片模拟器。在此预训练成本模型的基础上，我们进行在线搜索，以确定给定任何特定分片任务的最佳分片计划。在深度学习推荐模型（DLRMs）中，我们将此思想实例化，并提议了NeuroShard用于嵌入表分片。NeuroShard在扩展表上预先训练神经成本模型，以涵盖各种分片场景。然后，使用波束搜索和贪心网格搜索，分别确定最佳的列和表分片计划。实验结果表明

    Sharding a large machine learning model across multiple devices to balance the costs is important in distributed training. This is challenging because partitioning is NP-hard, and estimating the costs accurately and efficiently is difficult. In this work, we explore a "pre-train, and search" paradigm for efficient sharding. The idea is to pre-train a universal and once-for-all neural network to predict the costs of all the possible shards, which serves as an efficient sharding simulator. Built upon this pre-trained cost model, we then perform an online search to identify the best sharding plans given any specific sharding task. We instantiate this idea in deep learning recommendation models (DLRMs) and propose NeuroShard for embedding table sharding. NeuroShard pre-trains neural cost models on augmented tables to cover various sharding scenarios. Then it identifies the best column-wise and table-wise sharding plans with beam search and greedy grid search, respectively. Experiments show
    
[^7]: 针对神经排序模型的几乎不可察觉的文档篡改

    Towards Imperceptible Document Manipulations against Neural Ranking Models. (arXiv:2305.01860v1 [cs.IR])

    [http://arxiv.org/abs/2305.01860](http://arxiv.org/abs/2305.01860)

    该论文提出了一种针对神经排序模型的不易被检测到的对抗性攻击框架，称为“几乎不可察觉文档操作”（IDEM）。IDEM使用生成语言模型生成连结句，无法引入易于检测的错误，并且使用单独的位置合并策略来平衡扰动文本的相关性和连贯性，实验结果表明，IDEM可以在保持高人类评估得分的同时优于强基线。

    

    对抗性攻击已经开始应用于发现神经排序模型（NRMs）中的潜在漏洞，但是当前攻击方法常常会引入语法错误，无意义的表达，或不连贯的文本片段，这些都很容易被检测到。此外，当前方法严重依赖于使用与真实的NRM相似的模拟NRM来保证攻击效果，这使得它们在实践中难以使用。为了解决这些问题，我们提出了一个称为“几乎不可察觉文档操作”（IDEM）的框架，用于生成对算法和人类来说都不太明显的对抗文档。IDEM指示一个经过良好建立的生成语言模型（例如BART）生成连接句，而不会引入易于检测的错误，并采用单独的逐位置合并策略来平衡扰动文本的相关性和连贯性。在流行的MS MARCO基准上的实验结果表明，IDEM可以在保持高人类评估得分的同时，优于强基线。

    Adversarial attacks have gained traction in order to identify potential vulnerabilities in neural ranking models (NRMs), but current attack methods often introduce grammatical errors, nonsensical expressions, or incoherent text fragments, which can be easily detected. Additionally, current methods rely heavily on the use of a well-imitated surrogate NRM to guarantee the attack effect, which makes them difficult to use in practice. To address these issues, we propose a framework called Imperceptible DocumEnt Manipulation (IDEM) to produce adversarial documents that are less noticeable to both algorithms and humans. IDEM instructs a well-established generative language model, such as BART, to generate connection sentences without introducing easy-to-detect errors, and employs a separate position-wise merging strategy to balance relevance and coherence of the perturbed text. Experimental results on the popular MS MARCO benchmark demonstrate that IDEM can outperform strong baselines while 
    
[^8]: 当新的不一定是更好的：深度学习是否真正受益于基于隐式反馈的推荐？

    When Newer is Not Better: Does Deep Learning Really Benefit Recommendation From Implicit Feedback?. (arXiv:2305.01801v1 [cs.IR])

    [http://arxiv.org/abs/2305.01801](http://arxiv.org/abs/2305.01801)

    本研究对多个神经推荐模型与传统模型进行比较，提出了一组评估策略来衡量其记忆性能、泛化性能和子群特定性能，揭示了在IMDB和Yelp数据集上，神经推荐模型与传统模型的差异性。

    

    最近几年，神经模型被多次宣传为推荐领域的最先进技术，但是多个研究表明，许多神经推荐模型的最新结果并不能可靠地复现。一个主要原因是现有的评估是在不一致的协议下进行的。因此，这些可重复性问题使人们难以了解实际上可以从这些神经模型中获得多少益处。因此，需要一个公平而全面的绩效比较来比较传统模型和神经模型。为此，我们进行了一项大规模、系统性的研究，比较了基于隐式数据的顶部推荐的最新神经推荐模型和传统模型。我们提出了一组评估策略，用于衡量推荐模型的记忆性能、泛化性能和子群特定性能。

    In recent years, neural models have been repeatedly touted to exhibit state-of-the-art performance in recommendation. Nevertheless, multiple recent studies have revealed that the reported state-of-the-art results of many neural recommendation models cannot be reliably replicated. A primary reason is that existing evaluations are performed under various inconsistent protocols. Correspondingly, these replicability issues make it difficult to understand how much benefit we can actually gain from these neural models. It then becomes clear that a fair and comprehensive performance comparison between traditional and neural models is needed.  Motivated by these issues, we perform a large-scale, systematic study to compare recent neural recommendation models against traditional ones in top-n recommendation from implicit data. We propose a set of evaluation strategies for measuring memorization performance, generalization performance, and subgroup-specific performance of recommendation models. 
    
[^9]: 如何发挥大语言模型在少样本关系抽取中的能力？

    How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?. (arXiv:2305.01555v1 [cs.CL])

    [http://arxiv.org/abs/2305.01555](http://arxiv.org/abs/2305.01555)

    本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。

    

    语言模型的扩展已经彻底改变了广泛的自然语言处理任务，但是使用大型语言模型进行少样本关系抽取还没有得到全面探索。本文通过详细实验，研究了使用GPT-3.5进行少样本关系抽取的基本方法——上下文学习和数据生成。为了增强少样本性能，我们进一步提出了与任务相关的指导说明和约束模式下的数据生成。我们观察到，在上下文学习的情况下，可以实现与以前的提示学习方法相当的性能，而使用大型语言模型的数据生成可以推动以前的解决方案以在四个广泛研究的关系抽取数据集上获得新的最先进的少样本结果。我们希望我们的工作可以激发未来对大型语言模型在少样本关系抽取中的能力的研究。代码可以在 \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm} 中找到。

    Scaling language models have revolutionized widespread NLP tasks, yet little comprehensively explored few-shot relation extraction with large language models. In this paper, we investigate principal methodologies, in-context learning and data generation, for few-shot relation extraction via GPT-3.5 through exhaustive experiments. To enhance few-shot performance, we further propose task-related instructions and schema-constrained data generation. We observe that in-context learning can achieve performance on par with previous prompt learning approaches, and data generation with the large language model can boost previous solutions to obtain new state-of-the-art few-shot results on four widely-studied relation extraction datasets. We hope our work can inspire future research for the capabilities of large language models in few-shot relation extraction. Code is available in \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^10]: 利用模拟用户反馈的方式来优化会话式搜索

    Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond. (arXiv:2304.13874v1 [cs.IR])

    [http://arxiv.org/abs/2304.13874](http://arxiv.org/abs/2304.13874)

    本研究利用一个名为ConvSim的用户模拟器来评估用户反馈，从而提高会话式搜索的性能，实验结果显示有效利用用户反馈可以大幅提高检索性能。

    

    本研究旨在探索评估用户反馈在混合倡议的会话式搜索系统中的各种方法。虽然会话式搜索系统在多个方面都取得了重大进展，但最近的研究未能成功地将用户反馈纳入系统中。其中一个主要原因是缺乏系统-用户对话交互数据。为此，我们提出了一种基于用户模拟器的框架，可用于与各种混合倡议的会话式搜索系统进行多轮交互。具体来说，我们开发了一个名为ConvSim的用户模拟器，一旦初始化了信息需求描述，就能够对系统的响应提供反馈，并回答潜在的澄清问题。我们对各种最先进的段落检索和神经重新排序模型进行的实验表明，有效利用用户反馈可以导致在nDCG@3方面16%的检索性能提高。此外，我们观察到随着n的增加，一致的改进。

    This research aims to explore various methods for assessing user feedback in mixed-initiative conversational search (CS) systems. While CS systems enjoy profuse advancements across multiple aspects, recent research fails to successfully incorporate feedback from the users. One of the main reasons for that is the lack of system-user conversational interaction data. To this end, we propose a user simulator-based framework for multi-turn interactions with a variety of mixed-initiative CS systems. Specifically, we develop a user simulator, dubbed ConvSim, that, once initialized with an information need description, is capable of providing feedback to a system's responses, as well as answering potential clarifying questions. Our experiments on a wide variety of state-of-the-art passage retrieval and neural re-ranking models show that effective utilization of user feedback can lead to 16% retrieval performance increase in terms of nDCG@3. Moreover, we observe consistent improvements as the n
    
[^11]: 你在意吗？对于MIND数据集在新闻推荐多样性研究中的反思

    Do you MIND? Reflections on the MIND dataset for research on diversity in news recommendations. (arXiv:2304.08253v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2304.08253](http://arxiv.org/abs/2304.08253)

    该研究分析了MIND数据集在新闻推荐多样性研究中的适用性，发现虽然是一个很好的进步，但仍有很大的改进空间。

    

    MIND数据集是目前可用于新闻推荐系统研究和开发的最广泛的数据集。本研究分析了该数据集在多样化新闻推荐研究中的适用性。一方面，我们分析了推荐流程中不同步骤对文章类别分布的影响，另一方面，我们检查所提供的数据是否足以进行更复杂的多样性分析。我们得出结论，虽然MIND是一个很好的进步，但仍有很大的改进空间。

    The MIND dataset is at the moment of writing the most extensive dataset available for the research and development of news recommender systems. This work analyzes the suitability of the dataset for research on diverse news recommendations. On the one hand we analyze the effect the different steps in the recommendation pipeline have on the distribution of article categories, and on the other hand we check whether the supplied data would be sufficient for more sophisticated diversity analysis. We conclude that while MIND is a great step forward, there is still a lot of room for improvement.
    
[^12]: 推荐系统何去何从？ID- vs. 基于模态的推荐模型再探讨

    Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited. (arXiv:2303.13835v1 [cs.IR])

    [http://arxiv.org/abs/2303.13835](http://arxiv.org/abs/2303.13835)

    推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。

    

    过去十年，利用唯一标识（ID）来表示不同用户和物品的推荐模型一直是最先进的，并且在推荐系统文献中占主导地位。与此同时，预训练模态编码器（如BERT和ViT）在对物品的原始模态特征（如文本和图像）进行建模方面变得越来越强大。因此，自然而然的问题是：通过用最先进的模态编码器替换物品ID嵌入向量，一个纯粹的基于模态的推荐模型（MoRec）能否胜过或与纯ID基础模型（IDRec）相匹配？实际上，早在十年前，这个问题就被回答了，IDRec在推荐准确性和效率方面都远远胜过MoRec。我们旨在重新审视这个“老问题”，从多个方面对MoRec进行系统研究。具体而言，我们研究了几个子问题：（i）在实际场景中，MoRec或IDRec哪个推荐模式表现更好，特别是在一般情况和......

    Recommendation models that utilize unique identities (IDs) to represent distinct users and items have been state-of-the-art (SOTA) and dominated the recommender systems (RS) literature for over a decade. Meanwhile, the pre-trained modality encoders, such as BERT and ViT, have become increasingly powerful in modeling the raw modality features of an item, such as text and images. Given this, a natural question arises: can a purely modality-based recommendation model (MoRec) outperforms or matches a pure ID-based model (IDRec) by replacing the itemID embedding with a SOTA modality encoder? In fact, this question was answered ten years ago when IDRec beats MoRec by a strong margin in both recommendation accuracy and efficiency. We aim to revisit this `old' question and systematically study MoRec from several aspects. Specifically, we study several sub-questions: (i) which recommendation paradigm, MoRec or IDRec, performs better in practical scenarios, especially in the general setting and 
    
[^13]: 电商产品问答：一项综述调查

    Product Question Answering in E-Commerce: A Survey. (arXiv:2302.08092v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08092](http://arxiv.org/abs/2302.08092)

    电商PQA的研究面临着问题多、数据难收集、答案不确定等特殊挑战。本文系统地综述了PQA研究的现状与未来方向。

    

    产品问答（PQA）致力于在电商平台上自动快速回复客户问题，近年来受到越来越多的关注。与典型的问答问题相比，PQA表现出了独特的挑战，例如电商平台上用户生成内容的主观性和可靠性。因此，各种问题设置和新方法已被提出来捕捉这些特殊特征。本文旨在系统地审查现有关于PQA的研究工作。具体而言，我们将PQA研究按提供的答案形式将其分类为四个问题设置。我们分析了每个设置的优缺点，并介绍了现有的数据集和评估协议。我们进一步总结了表征PQA与一般QA应用的最显著的挑战，并讨论了相应的解决方案。最后，我们通过提供几个未来方向来结束本文。

    Product question answering (PQA), aiming to automatically provide instant responses to customer's questions in E-Commerce platforms, has drawn increasing attention in recent years. Compared with typical QA problems, PQA exhibits unique challenges such as the subjectivity and reliability of user-generated contents in E-commerce platforms. Therefore, various problem settings and novel methods have been proposed to capture these special characteristics. In this paper, we aim to systematically review existing research efforts on PQA. Specifically, we categorize PQA studies into four problem settings in terms of the form of provided answers. We analyze the pros and cons, as well as present existing datasets and evaluation protocols for each setting. We further summarize the most significant challenges that characterize PQA from general QA applications and discuss their corresponding solutions. Finally, we conclude this paper by providing the prospect on several future directions.
    
[^14]: 竞争性内容创作者下的Top-K推荐有多糟糕？

    How Bad is Top-$K$ Recommendation under Competing Content Creators?. (arXiv:2302.01971v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.01971](http://arxiv.org/abs/2302.01971)

    本文基于随机效用模型，研究了内容创作者在Top-K推荐下的竞争影响，证明了用户福利损失受小常数上界影响。

    

    内容创作者在推荐平台上竞争曝光率，这种战略行为导致了内容分布的动态转移。然而，创作者的竞争如何影响用户福利，以及相关推荐如何影响长期动态仍然大部分未知。本文提出了这些研究问题的理论见解。我们在以下假设下建模创作者的竞争：1）平台采用无害的top-K推荐策略；2）用户决策遵循随机效用模型；3）内容创作者竞争用户互动，不知道事先他们的效用函数，因此应用任意的无悔学习算法来更新他们的策略。我们通过洛城价格的角度研究用户福利的保证，并展示了由于创作者竞争导致的用户福利损失份额始终受到$K$和用户决策随机性影响的小常数的上界约束。

    Content creators compete for exposure on recommendation platforms, and such strategic behavior leads to a dynamic shift over the content distribution. However, how the creators' competition impacts user welfare and how the relevance-driven recommendation influences the dynamics in the long run are still largely unknown.  This work provides theoretical insights into these research questions. We model the creators' competition under the assumptions that: 1) the platform employs an innocuous top-$K$ recommendation policy; 2) user decisions follow the Random Utility model; 3) content creators compete for user engagement and, without knowing their utility function in hindsight, apply arbitrary no-regret learning algorithms to update their strategies. We study the user welfare guarantee through the lens of Price of Anarchy and show that the fraction of user welfare loss due to creator competition is always upper bounded by a small constant depending on $K$ and randomness in user decisions; w
    
[^15]: 语言建模在推荐系统中的关键作用：丰富任务特定和任务无关的表示学习

    Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning. (arXiv:2212.03760v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.03760](http://arxiv.org/abs/2212.03760)

    本文研究发现，用户历史语言建模可以在不同推荐任务中取得优异结果，并且利用任务无关的用户历史还可以提供显著的性能优势。该方法具有广泛的现实世界迁移学习能力。

    

    最近的研究提出了利用来自各种应用程序的用户行为数据的统一用户建模框架。其中许多受益于将用户行为序列作为纯文本使用，代表着任何领域或系统中的丰富信息而不失通用性。因此，一个问题产生了：用户历史语言建模能否帮助改善推荐系统？虽然语言建模的多功能性已在许多领域广泛研究，但其在推荐系统中的应用仍未深入探讨。我们展示了直接应用于任务特定用户历史的语言建模在不同的推荐任务上可以取得优异的结果。此外，利用任务无关的用户历史还可以提供显著的性能优势。我们进一步证明了我们的方法可以为广泛的现实世界推荐系统提供有前途的迁移学习能力，甚至在未知域和服务上也可以实现。

    Recent studies have proposed unified user modeling frameworks that leverage user behavior data from various applications. Many of them benefit from utilizing users' behavior sequences as plain texts, representing rich information in any domain or system without losing generality. Hence, a question arises: Can language modeling for user history corpus help improve recommender systems? While its versatile usability has been widely investigated in many domains, its applications to recommender systems still remain underexplored. We show that language modeling applied directly to task-specific user histories achieves excellent results on diverse recommendation tasks. Also, leveraging additional task-agnostic user histories delivers significant performance benefits. We further demonstrate that our approach can provide promising transfer learning capabilities for a broad spectrum of real-world recommender systems, even on unseen domains and services.
    

