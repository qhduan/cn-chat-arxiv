# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree Planning](https://arxiv.org/abs/2402.13125) | TreeEval提出了一种无基准评估方法，通过树规划策略提升了大型语言模型的评估效率和完整性 |
| [^2] | [Citation Amnesia: NLP and Other Academic Fields Are in a Citation Age Recession](https://arxiv.org/abs/2402.12046) | 该研究分析了不同学术领域在43年间引用较旧作品的趋势，发现自然语言处理和机器学习研究中引文年龄衰退最为明显，此趋势并非由出版速率增长主导。 |
| [^3] | [Pretraining Vision-Language Model for Difference Visual Question Answering in Longitudinal Chest X-rays](https://arxiv.org/abs/2402.08966) | 提出了一种名为PLURAL的预训练视觉-语言模型，用于纵向胸部X射线图中差异视觉问答任务。该模型通过在自然图像和文本上进行预训练，然后使用纵向胸部X射线数据进行训练，从而提高了模型的性能。 |
| [^4] | [ReFT: Reasoning with Reinforced Fine-Tuning.](http://arxiv.org/abs/2401.08967) | ReFT是一种加强推理能力的强化微调方法，通过利用更多的推理路径进行微调，提高了大型语言模型在数学问题解决中的泛化能力。 |
| [^5] | [Salute the Classic: Revisiting Challenges of Machine Translation in the Age of Large Language Models.](http://arxiv.org/abs/2401.08350) | 本文重新审视了机器翻译领域的六个核心挑战，并提供了对于大规模语言模型在这些挑战中所取得进展的实证发现。研究发现，大规模语言模型能够有效减少对平行数据的依赖，提高翻译质量并扩展翻译文档的长度范围。然而，领域不匹配和罕见词预测仍然是需要解决的问题。 |
| [^6] | [CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search.](http://arxiv.org/abs/2305.11626) | 本研究提出了多语言克隆检测问题，并从CodeForces数据集开发了一个新的基准数据集XCD。我们使用跨语言一致性训练（CCT）方法训练了语言模型，得到了具有新颖性能的CCT-LM模型，超过了现有的方法。 |

# 详细

[^1]: TreeEval：通过树规划实现对大型语言模型的无基准评估

    TreeEval: Benchmark-Free Evaluation of Large Language Models through Tree Planning

    [https://arxiv.org/abs/2402.13125](https://arxiv.org/abs/2402.13125)

    TreeEval提出了一种无基准评估方法，通过树规划策略提升了大型语言模型的评估效率和完整性

    

    最近，建立了许多新的基准来评估大型语言模型（LLMs）的性能，通过计算整体得分或使用另一个LLM作为评判者。然而，这些方法由于基准的公开访问和评估过程的不灵活而遭受数据泄漏的困扰。为了解决这个问题，我们引入了TreeEval，这是一种无基准评估方法，让一个高性能的LLM主持一个不可重现的评估会话，从根本上避免了数据泄漏。此外，这个LLM充当一个考官，提出一系列关于一个主题的问题，并采用树规划策略，考虑当前的评估状态来决定下一个问题的生成，确保评估过程的完整性和效率。我们评估了不同参数大小的6个模型，包括7B、13B和33B，最终实现了最高的相关系数。

    arXiv:2402.13125v1 Announce Type: cross  Abstract: Recently, numerous new benchmarks have been established to evaluate the performance of large language models (LLMs) via either computing a holistic score or employing another LLM as a judge. However, these approaches suffer from data leakage due to the open access of the benchmark and inflexible evaluation process. To address this issue, we introduce $\textbf{TreeEval}$, a benchmark-free evaluation method for LLMs that let a high-performance LLM host an irreproducible evaluation session and essentially avoids the data leakage. Moreover, this LLM performs as an examiner to raise up a series of questions under a topic with a tree planing strategy, which considers the current evaluation status to decide the next question generation and ensures the completeness and efficiency of the evaluation process. We evaluate $6$ models of different parameter sizes, including $7$B, $13$B, and $33$B, and ultimately achieved the highest correlation coef
    
[^2]: 引文遗忘：自然语言处理和其他学术领域正处于引文年龄衰退期

    Citation Amnesia: NLP and Other Academic Fields Are in a Citation Age Recession

    [https://arxiv.org/abs/2402.12046](https://arxiv.org/abs/2402.12046)

    该研究分析了不同学术领域在43年间引用较旧作品的趋势，发现自然语言处理和机器学习研究中引文年龄衰退最为明显，此趋势并非由出版速率增长主导。

    

    这项研究考察了在43年的时间跨度（1980-2023年）内，在20个研究领域中倾向于引用较旧作品的趋势。我们将自然语言处理(NLP)倾向于引用较旧作品的特性放在其他20个领域的背景下进行分析，以探讨NLP是否展现出与其他领域随时间出现类似的引文模式，或者是否可以观察到差异。我们的分析基于约2.4亿篇论文的数据集，揭示了一个更广泛的科学趋势：许多领域在引用较旧作品方面明显下降（例如心理学、计算机科学）。我们将这种下降称为“引文年龄衰退”，类似于经济学家如何定义减少经济活动的时期。这一趋势在NLP和机器学习研究中最为显著（引文年龄从先前高峰下降了12.8%和5.5%）。我们的结果表明，对更近期作品的引用并非直接受到出版速率增长的推动（跨领域下降了3.4%，人文学科下降了5.2%，形式科学下降了5.5%）--即使在控制了发表数量时。

    arXiv:2402.12046v1 Announce Type: cross  Abstract: This study examines the tendency to cite older work across 20 fields of study over 43 years (1980--2023). We put NLP's propensity to cite older work in the context of these 20 other fields to analyze whether NLP shows similar temporal citation patterns to these other fields over time or whether differences can be observed. Our analysis, based on a dataset of approximately 240 million papers, reveals a broader scientific trend: many fields have markedly declined in citing older works (e.g., psychology, computer science). We term this decline a 'citation age recession', analogous to how economists define periods of reduced economic activity. The trend is strongest in NLP and ML research (-12.8% and -5.5% in citation age from previous peaks). Our results suggest that citing more recent works is not directly driven by the growth in publication rates (-3.4% across fields; -5.2% in humanities; -5.5% in formal sciences) -- even when controlli
    
[^3]: 用于纵向胸部X射线图中差异视觉问答的预训练视觉-语言模型

    Pretraining Vision-Language Model for Difference Visual Question Answering in Longitudinal Chest X-rays

    [https://arxiv.org/abs/2402.08966](https://arxiv.org/abs/2402.08966)

    提出了一种名为PLURAL的预训练视觉-语言模型，用于纵向胸部X射线图中差异视觉问答任务。该模型通过在自然图像和文本上进行预训练，然后使用纵向胸部X射线数据进行训练，从而提高了模型的性能。

    

    差异视觉问答(diff-VQA)是一个挑战性的任务，要求根据一对图像的差异回答复杂的问题。在读取胸部X射线图像中尤为重要，因为放射科医生通常会对同一患者在不同时间拍摄的多幅图像进行比较，以追踪疾病的进展和其临床实践中严重程度的变化。然而，之前的研究集中在为差异视觉问答任务设计特定的网络架构，错过了利用预训练的视觉-语言模型(VLM)提高模型性能的机会。在这里，我们介绍了一种名为PLURAL的新型VLM，它在自然图像和纵向胸部X射线数据上进行了差异视觉问答任务的预训练。该模型采用逐步的方法开发，从在自然图像和文本上进行预训练开始，然后使用纵向胸部X射线数据进行训练。纵向数据包括...

    arXiv:2402.08966v1 Announce Type: cross Abstract: Difference visual question answering (diff-VQA) is a challenging task that requires answering complex questions based on differences between a pair of images. This task is particularly important in reading chest X-ray images because radiologists often compare multiple images of the same patient taken at different times to track disease progression and changes in its severity in their clinical practice. However, previous works focused on designing specific network architectures for the diff-VQA task, missing opportunities to enhance the model's performance using a pretrained vision-language model (VLM). Here, we introduce a novel VLM called PLURAL, which is pretrained on natural and longitudinal chest X-ray data for the diff-VQA task. The model is developed using a step-by-step approach, starting with being pretrained on natural images and texts, followed by being trained using longitudinal chest X-ray data. The longitudinal data consist
    
[^4]: ReFT: 加强强化微调的推理能力

    ReFT: Reasoning with Reinforced Fine-Tuning. (arXiv:2401.08967v1 [cs.CL])

    [http://arxiv.org/abs/2401.08967](http://arxiv.org/abs/2401.08967)

    ReFT是一种加强推理能力的强化微调方法，通过利用更多的推理路径进行微调，提高了大型语言模型在数学问题解决中的泛化能力。

    

    增强大型语言模型（LLMs）的推理能力的一种方法是使用链式思考（CoT）注释进行监督微调（SFT）。然而，这种方法在泛化能力上并不十分强大，因为训练仅依赖于给定的CoT数据。例如，在数学问题解决中，训练数据中通常只有一个注释的推理路径用于每个问题。直观来说，让算法从给定的问题中学习多个注释的推理路径会更好。为了解决这个问题，我们提出了一种简单而有效的方法，称为加强强化微调（ReFT），以增强学习LLMs进行推理的泛化能力，以数学问题解决为例。ReFT首先使用SFT对模型进行热身，然后采用在线强化学习（具体来说，在本文中是使用PPO算法）进一步微调模型，其中根据问题自动采样了大量的推理路径。

    One way to enhance the reasoning capability of Large Language Models (LLMs) is to conduct Supervised Fine-Tuning (SFT) using Chain-of-Thought (CoT) annotations. This approach does not show sufficiently strong generalization ability, however, because the training only relies on the given CoT data. In math problem-solving, for example, there is usually only one annotated reasoning path for each question in the training data. Intuitively, it would be better for the algorithm to learn from multiple annotated reasoning paths given a question. To address this issue, we propose a simple yet effective approach called Reinforced Fine-Tuning (ReFT) to enhance the generalizability of learning LLMs for reasoning, with math problem-solving as an example. ReFT first warmups the model with SFT, and then employs on-line reinforcement learning, specifically the PPO algorithm in this paper, to further fine-tune the model, where an abundance of reasoning paths are automatically sampled given the question
    
[^5]: 向经典致敬：在大规模语言模型时代重新思考机器翻译的挑战

    Salute the Classic: Revisiting Challenges of Machine Translation in the Age of Large Language Models. (arXiv:2401.08350v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.08350](http://arxiv.org/abs/2401.08350)

    本文重新审视了机器翻译领域的六个核心挑战，并提供了对于大规模语言模型在这些挑战中所取得进展的实证发现。研究发现，大规模语言模型能够有效减少对平行数据的依赖，提高翻译质量并扩展翻译文档的长度范围。然而，领域不匹配和罕见词预测仍然是需要解决的问题。

    

    神经机器翻译 (NMT) 的发展受到六个核心挑战的显著影响，这些挑战为这个领域的进展提供了基准。本研究重新审视了这些挑战，在先进大规模语言模型 (LLM) 的背景下，提供了对这些挑战持续相关性的深入见解。我们的实证研究表明，在预训练阶段，LLM能够有效减少对平行数据的依赖，特别是对于主要语言。此外，基于LLM的翻译系统显著提高了翻译约80个单词的长句子的质量，并且能够翻译长达512个单词的文档。然而，尽管取得了显著的改进，领域不匹配和罕见词预测仍然是挑战。在解决单词对齐和亚最优搜索的挑战方面，LLM仍存在改进的空间。

    The evolution of Neural Machine Translation (NMT) has been significantly influenced by six core challenges (Koehn and Knowles, 2017), which have acted as benchmarks for progress in this field. This study revisits these challenges, offering insights into their ongoing relevance in the context of advanced Large Language Models (LLMs): domain mismatch, amount of parallel data, rare word prediction, translation of long sentences, attention model as word alignment, and sub-optimal beam search. Our empirical findings indicate that LLMs effectively lessen the reliance on parallel data for major languages in the pretraining phase. Additionally, the LLM-based translation system significantly enhances the translation of long sentences that contain approximately 80 words and shows the capability to translate documents of up to 512 words. However, despite these significant improvements, the challenges of domain mismatch and prediction of rare words persist. While the challenges of word alignment a
    
[^6]: CCT-Code：面向多语言克隆检测和代码搜索的跨语言一致性训练

    CCT-Code: Cross-Consistency Training for Multilingual Clone Detection and Code Search. (arXiv:2305.11626v1 [cs.CL])

    [http://arxiv.org/abs/2305.11626](http://arxiv.org/abs/2305.11626)

    本研究提出了多语言克隆检测问题，并从CodeForces数据集开发了一个新的基准数据集XCD。我们使用跨语言一致性训练（CCT）方法训练了语言模型，得到了具有新颖性能的CCT-LM模型，超过了现有的方法。

    

    本文考虑源代码的克隆检测和信息检索问题，这两个问题对于任何编程语言都非常重要。我们提出了多语言克隆检测问题，并从CodeForces提交数据集产生了一个新的基准数据集XCD。此外，我们提出了一种新型的训练方法，称为跨语言一致性训练（CCT），用于在不同的编程语言中训练语言模型，进而得到基于CCT-LM 模型。该模型继承了GraphCodeBERT并用CCT微调，达到了95.67\% MAP和47.18\% MRR的性能，成功创造了新的最优结果。

    We consider the clone detection and information retrieval problems for source code, well-known tasks important for any programming language. Although it is also an important and interesting problem to find code snippets that operate identically but are written in different programming languages, to the best of our knowledge multilingual clone detection has not been studied in literature. In this work, we formulate the multilingual clone detection problem and present XCD, a new benchmark dataset produced from the CodeForces submissions dataset. Moreover, we present a novel training procedure, called cross-consistency training (CCT), that we apply to train language models on source code in different programming languages. The resulting CCT-LM model, initialized with GraphCodeBERT and fine-tuned with CCT, achieves new state of the art, outperforming existing approaches on the POJ-104 clone detection benchmark with 95.67\% MAP and AdvTest code search benchmark with 47.18\% MRR; it also sho
    

