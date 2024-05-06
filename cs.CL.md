# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can Large Language Models (or Humans) Distill Text?](https://arxiv.org/abs/2403.16584) | 大型语言模型（LLMs）在文本提炼中具有独特优势，但在处理情感时仍存在一定局限性，无论是对机器学习分类器还是人类注释员而言。 |
| [^2] | [From explainable to interpretable deep learning for natural language processing in healthcare: how far from reality?](https://arxiv.org/abs/2403.11894) | 该研究对医疗保健NLP中的深度学习进行了全面审查，提出了可解释和可解释的人工智能（XIAI）概念，并发现注意机制是主要新兴IAI，同时面临着缺乏全局建模、最佳实践以及系统评估和基准测试的挑战。 |
| [^3] | [Fisher Mask Nodes for Language Model Merging](https://arxiv.org/abs/2403.09891) | 介绍了一种用于Transformers的新型模型合并方法，利用Fisher信息进行加权平均，提高了多任务模型的性能。 |
| [^4] | [LUCID: LLM-Generated Utterances for Complex and Interesting Dialogues](https://arxiv.org/abs/2403.00462) | LUCID旨在通过高质量和语言复杂的数据，以及高度自动化的LLM模型，解决现有数据集领域覆盖有限、对话现象有限、未标记的特点，以及需要大量人力投入的问题。 |
| [^5] | [Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Match Human Crowd Accuracy](https://arxiv.org/abs/2402.19379) | 该研究通过将十二个LLMs组成的LLM集成方法与925名人类预测者的群体预测进行比较，发现LLM群体优于简单的无信息基准，并在统计上等效于人类群体。 |
| [^6] | [Training Language Model Agents without Modifying Language Models](https://arxiv.org/abs/2402.11359) | 提出一种新的方法，在不修改语言模型的情况下训练语言模型代理，通过进化代理的功能来解决下游任务 |
| [^7] | [Careless Whisper: Speech-to-Text Hallucination Harms](https://arxiv.org/abs/2402.08021) | 该论文评估了开放AI的语音识别服务Whisper，并指出其中约1%的转录存在完全幻觉的短语或句子。这些幻觉内容中有38%包含明确的伤害，如暴力、虚构的个人信息或虚假的基于视频的权威。研究者进一步提供了幻觉发生的假设，并指出了由于语音类型和健康状况的不同可能导致的潜在差异。他们呼吁行业从业者改善基于语言模型的幻觉，并增强对下游潜在偏见的认识。 |
| [^8] | [InceptionXML: A Lightweight Framework with Synchronized Negative Sampling for Short Text Extreme Classification](https://arxiv.org/abs/2109.07319) | 提出了一种轻量级框架InceptionXML，通过在embedding维度上重新分配卷积操作，应对短文本查询中的单词顺序缺失，同时提出了InceptionXML+框架，通过同步标签筛选器和极端分类器，改进了动态硬负采样技术。 |
| [^9] | [Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models.](http://arxiv.org/abs/2401.10647) | 本文研究了通过编辑语言模型的复杂后果，发现在增强模型准确性与保持道德完整性之间存在悖论。我们发现，尽管注入准确信息对模型的可靠性很重要，但它可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。 |
| [^10] | [Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition.](http://arxiv.org/abs/2312.17279) | 本文提出一种基于FastConformer架构的流式语音识别模型，通过限制上下文和引入缓存机制，在推理过程中实现非自回归编码器的自回归操作，并消除了训练和推理准确度间的差异。同时，还提出了CTC/RNNT混合架构以提高准确度和节省计算。 |
| [^11] | [From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks.](http://arxiv.org/abs/2310.11884) | 本文调查了解释神经网络中概念的最新方法，这对于实现基于可解释概念的神经符号化人工智能来说是重要的一步。 |
| [^12] | [Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance.](http://arxiv.org/abs/2310.05597) | 本文研究了语言模型是否能够学习类比推理的任务，并测试了几种学习方法。实验结果表明，模型能够通过少量数据学习类比推理，并在与人类基准进行比较后接近人类的表现水平。 |
| [^13] | [RLAdapter: Bridging Large Language Models to Reinforcement Learning in Open Worlds.](http://arxiv.org/abs/2309.17176) | RLAdapter引入了一个框架，将大型语言模型（LLM）与强化学习相结合，以提高在稀疏奖励环境中的策略学习性能。这通过解决LLM在理解下游任务方面的困难，以及通过避免使用不可访问的模型权重或大量计算资源来微调LLM的方式实现。 |
| [^14] | [MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models.](http://arxiv.org/abs/2309.12284) | MetaMath是一种专门用于数学推理的微调语言模型，通过从多个角度重新编写问题来生成数学问题，并在两个基准测试中取得了优于其他开源语言模型的表现。 |

# 详细

[^1]: 大型语言模型（或人类）是否能进行文本提炼？

    Can Large Language Models (or Humans) Distill Text?

    [https://arxiv.org/abs/2403.16584](https://arxiv.org/abs/2403.16584)

    大型语言模型（LLMs）在文本提炼中具有独特优势，但在处理情感时仍存在一定局限性，无论是对机器学习分类器还是人类注释员而言。

    

    我们研究了大型语言模型（LLMs）在文本提炼方面的潜力：即去除不需要的禁止变量的文本痕迹。我们利用具有不同架构和训练方法的一系列LLMs来识别和去除关于目标变量的信息，同时保留其他相关信号。我们的研究结果揭示了LLMs在处理提炼中的优势和局限性，并为在涉及文本数据的计算社会科学调查中利用这些模型的策略提供了见解。尤其是，我们发现在强烈测试情感移除时，经过LLM提炼的文本与情感之间的统计关联仍然可以被机器学习分类器清晰地检测到。此外，我们发现人类注释员在保留其他语义内容的同时，也很难提炼出情感。这表明可能存在一定的局限性。

    arXiv:2403.16584v1 Announce Type: new  Abstract: We investigate the potential of large language models (LLMs) to distill text: to remove the textual traces of an undesired forbidden variable. We employ a range of LLMs with varying architectures and training approaches to distill text by identifying and removing information about the target variable while preserving other relevant signals. Our findings shed light on the strengths and limitations of LLMs in addressing the distillation and provide insights into the strategies for leveraging these models in computational social science investigations involving text data. In particular, we show that in the strong test of removing sentiment, the statistical association between the processed text and sentiment is still clearly detectable to machine learning classifiers post-LLM-distillation. Furthermore, we find that human annotators also struggle to distill sentiment while preserving other semantic content. This suggests there may be limited
    
[^2]: 从可解释到可解释的深度学习在医疗自然语言处理中的应用：现实有多远？

    From explainable to interpretable deep learning for natural language processing in healthcare: how far from reality?

    [https://arxiv.org/abs/2403.11894](https://arxiv.org/abs/2403.11894)

    该研究对医疗保健NLP中的深度学习进行了全面审查，提出了可解释和可解释的人工智能（XIAI）概念，并发现注意机制是主要新兴IAI，同时面临着缺乏全局建模、最佳实践以及系统评估和基准测试的挑战。

    

    深度学习（DL）通过解决各种自然语言处理（NLP）任务，极大地增强了医疗保健研究。然而，基于DL的NLP方法日益复杂，需要透明的模型解释性，或至少是可解释性，以进行可靠的决策制定。本文对医疗健康NLP中的可解释和可解释的DL进行了彻底的范围审查。引入了术语“XIAI”（eXplainable和Interpretable Artificial Intelligence）以区分XAI和IAI。方法根据其功能（模型、输入、输出为基础）和范围（局部、全局）进一步分类。我们的分析表明，注意机制是最主要的新兴IAI。此外，IAI越来越多地用于对抗XAI。确定的主要挑战是大多数XIAI不探索“全局”建模过程，缺乏最佳实践，并且需要系统评估和基准测试。

    arXiv:2403.11894v1 Announce Type: cross  Abstract: Deep learning (DL) has substantially enhanced healthcare research by addressing various natural language processing (NLP) tasks. Yet, the increasing complexity of DL-based NLP methods necessitates transparent model interpretability, or at least explainability, for reliable decision-making. This work presents a thorough scoping review on explainable and interpretable DL in healthcare NLP. The term "XIAI" (eXplainable and Interpretable Artificial Intelligence) was introduced to distinguish XAI from IAI. Methods were further categorized based on their functionality (model-, input-, output-based) and scope (local, global). Our analysis shows that attention mechanisms were the most dominant emerging IAI. Moreover, IAI is increasingly used against XAI. The major challenges identified are that most XIAI do not explore "global" modeling processes, the lack of best practices, and the unmet need for systematic evaluation and benchmarks. Importan
    
[^3]: Fisher Mask节点用于语言模型合并

    Fisher Mask Nodes for Language Model Merging

    [https://arxiv.org/abs/2403.09891](https://arxiv.org/abs/2403.09891)

    介绍了一种用于Transformers的新型模型合并方法，利用Fisher信息进行加权平均，提高了多任务模型的性能。

    

    微调预训练模型在下游性能方面具有显著优势。预训练模型（如BERT及其衍生物）在自然语言处理中的普遍性也导致了任务特定微调模型的激增。在多任务场景中，由于这些模型通常只能很好地执行一项任务，因此需要额外的训练或集成。模型合并这一不断增长的领域提供了一个解决方案，解决了将多个任务特定模型合并为单个多任务模型的挑战。在本研究中，我们引入了一种新颖的用于Transformers的模型合并方法，结合了先前Fisher加权平均和Fisher信息在模型修剪中的应用的见解。通过利用Transformer架构内的mask节点的Fisher信息，我们设计了一个计算效率高的加权平均方案。我们的方法展现出了稳定且显著的性能。

    arXiv:2403.09891v1 Announce Type: cross  Abstract: Fine-tuning pre-trained models provides significant advantages in downstream performance. The ubiquitous nature of pre-trained models such as BERT and its derivatives in natural language processing has also led to a proliferation of task-specific fine-tuned models. As these models typically only perform one task well, additional training or ensembling is required in multi-task scenarios. The growing field of model merging provides a solution, dealing with the challenge of combining multiple task-specific models into a single multi-task model. In this study, we introduce a novel model merging method for Transformers, combining insights from previous work in Fisher-weighted averaging and the use of Fisher information in model pruning. Utilizing the Fisher information of mask nodes within the Transformer architecture, we devise a computationally efficient weighted-averaging scheme. Our method exhibits a regular and significant performance
    
[^4]: LUCID: 由LLM生成的复杂且有趣对话

    LUCID: LLM-Generated Utterances for Complex and Interesting Dialogues

    [https://arxiv.org/abs/2403.00462](https://arxiv.org/abs/2403.00462)

    LUCID旨在通过高质量和语言复杂的数据，以及高度自动化的LLM模型，解决现有数据集领域覆盖有限、对话现象有限、未标记的特点，以及需要大量人力投入的问题。

    

    arXiv:2403.00462v1 公告类型:新 摘要:虚拟助手在对话能力方面即将迈向一个重要进步，这得益于基于变压器的大型语言模型（LLMs）的最新进展。然而，实现真正革命性的任务导向对话能力的主要瓶颈仍然是高质量和语言复杂的数据的稀缺性。现有数据集在规模上令人印象深刻，但领域覆盖范围有限，包含寥寥无几的真正具挑战性的对话现象；这些现象通常未标记，这使得在没有费时费力的人类评估的情况下难以评估模型的优势和劣势。此外，直到现在，创建高质量对话数据仍需要相当大量的人力投入，这限制了这些数据集的规模以及快速为新目标领域的数据增量。我们的目标是通过LUCID来克服这些问题，这是一个模块化且高度自动化的LLM

    arXiv:2403.00462v1 Announce Type: new  Abstract: Virtual assistants are poised to take a dramatic leap forward in terms of their dialogue capabilities, spurred by recent advances in transformer-based Large Language Models (LLMs). Yet a major bottleneck to achieving genuinely transformative task-oriented dialogue capabilities remains the scarcity of high quality and linguistically sophisticated data. Existing datasets, while impressive in scale, have limited domain coverage and contain few genuinely challenging conversational phenomena; those which are present are typically unlabelled, making it difficult to assess the strengths and weaknesses of models without time-consuming and costly human evaluation. Moreover, creating high quality dialogue data has until now required considerable human input, limiting both the scale of these datasets and the ability to rapidly bootstrap data for a new target domain. We aim to overcome these issues with LUCID, a modularised and highly automated LLM-
    
[^5]: 硅谷人群的智慧：LLM集成预测能力达到人群准确率水平

    Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Match Human Crowd Accuracy

    [https://arxiv.org/abs/2402.19379](https://arxiv.org/abs/2402.19379)

    该研究通过将十二个LLMs组成的LLM集成方法与925名人类预测者的群体预测进行比较，发现LLM群体优于简单的无信息基准，并在统计上等效于人类群体。

    

    实践中人类预测准确性依赖于“群体智慧”效应，即通过聚合一群个体预测者的预测可以显著提高对未来事件的预测。过去关于大型语言模型（LLMs）预测能力的研究表明，作为个体预测者的前沿LLMs表现不佳，与人类群体预测比赛的黄金标准相比。我们通过使用一个由十二个LLMs组成的LLM集成方法，扩展了研究。我们将31个二元问题的聚合LLM预测与一个来自三个月预测比赛的925名人类预测者的群体预测进行比较。我们的主要分析表明，LLM群体的表现优于简单的无信息基准，并在统计上等效于人类群体。我们还观察到一种顺从效应，平均模型预测明显高于50%，尽管几乎是平等的。

    arXiv:2402.19379v1 Announce Type: cross  Abstract: Human forecasting accuracy in practice relies on the 'wisdom of the crowd' effect, in which predictions about future events are significantly improved by aggregating across a crowd of individual forecasters. Past work on the forecasting ability of large language models (LLMs) suggests that frontier LLMs, as individual forecasters, underperform compared to the gold standard of a human crowd forecasting tournament aggregate. In Study 1, we expand this research by using an LLM ensemble approach consisting of a crowd of twelve LLMs. We compare the aggregated LLM predictions on 31 binary questions to that of a crowd of 925 human forecasters from a three-month forecasting tournament. Our main analysis shows that the LLM crowd outperforms a simple no-information benchmark and is statistically equivalent to the human crowd. We also observe an acquiescence effect, with mean model predictions being significantly above 50%, despite an almost even
    
[^6]: 在不修改语言模型的情况下训练语言模型代理

    Training Language Model Agents without Modifying Language Models

    [https://arxiv.org/abs/2402.11359](https://arxiv.org/abs/2402.11359)

    提出一种新的方法，在不修改语言模型的情况下训练语言模型代理，通过进化代理的功能来解决下游任务

    

    研究人员和实践者最近已经将强大的大型语言模型（LLMs）重新定义为代理，使它们能够通过使用专门的功能自动化地完成复杂任务。为了促进LLM代理的发展，我们提出了一种在不修改LLM权重的情况下训练LLM代理的新范式，当LLM难以或无法进行修改时尤其有用。受到人类不断锻造工具以适应现实任务的启发，而不是改变我们的生物结构以适应一组静态工具，我们提出逐步锻造代理的功能，以更好地解决下游任务，而不是修改LLM权重。通过将这些功能视为可学习的“代理参数”并利用人工智能模型训练的基本思想，我们开发了AgentOptimizer，利用LLM更新代理的功能，并设计了一种代理训练算法

    arXiv:2402.11359v1 Announce Type: new  Abstract: Researchers and practitioners have recently reframed powerful Large Language Models (LLMs) as agents, enabling them to automate complex tasks largely via the use of specialized functions. To facilitate the development of LLM agents, we present a novel paradigm of training LLM agents without modifying the LLM weights, which is particularly useful when the LLMs are difficult or inaccessible for modifications. Inspired by how humans continuously forge tools to adapt to real-world tasks, rather than change our biological structure to fit a static set of tools, we propose to progressively forge agent's functions to better solve the downstream tasks instead of modifying the LLM weights. By treating the functions as learnable `agent parameters' and leveraging the fundamental idea of model training in artificial intelligence, we develop AgentOptimizer that employs the LLM to update agents' functions and devise an agent training algorithm with tw
    
[^7]: 不小心的耳语：语音转文本幻觉的危害

    Careless Whisper: Speech-to-Text Hallucination Harms

    [https://arxiv.org/abs/2402.08021](https://arxiv.org/abs/2402.08021)

    该论文评估了开放AI的语音识别服务Whisper，并指出其中约1%的转录存在完全幻觉的短语或句子。这些幻觉内容中有38%包含明确的伤害，如暴力、虚构的个人信息或虚假的基于视频的权威。研究者进一步提供了幻觉发生的假设，并指出了由于语音类型和健康状况的不同可能导致的潜在差异。他们呼吁行业从业者改善基于语言模型的幻觉，并增强对下游潜在偏见的认识。

    

    语音转文本服务旨在尽可能准确地转录输入音频。它们在日常生活中的作用越来越大，例如个人语音助手或公司与客户的互动中。我们评估了开放AI的Whisper，这是一种超越行业竞争对手的最新服务。虽然Whisper的许多转录非常准确，但我们发现大约1％的音频转录包含完全幻觉的短语或句子，这些短语或句子在基础音频中不存在。我们主题化地分析了Whisper幻觉的内容，发现38％的幻觉包含明确的伤害，例如暴力、虚构的个人信息或虚假的基于视频的权威。我们进一步提供了关于幻觉发生的假设，并揭示了由于健康状况而导致的语音类型的潜在差异。我们呼吁行业从业者改善Whisper中基于语言模型的幻觉，并增强对下游潜在偏见的认识。

    Speech-to-text services aim to transcribe input audio as accurately as possible. They increasingly play a role in everyday life, for example in personal voice assistants or in customer-company interactions. We evaluate Open AI's Whisper, a state-of-the-art service outperforming industry competitors. While many of Whisper's transcriptions were highly accurate, we found that roughly 1% of audio transcriptions contained entire hallucinated phrases or sentences, which did not exist in any form in the underlying audio. We thematically analyze the Whisper-hallucinated content, finding that 38% of hallucinations include explicit harms such as violence, made up personal information, or false video-based authority. We further provide hypotheses on why hallucinations occur, uncovering potential disparities due to speech type by health status. We call on industry practitioners to ameliorate these language-model-based hallucinations in Whisper, and to raise awareness of potential biases in downstr
    
[^8]: InceptionXML：一种带有同步负采样的轻量级框架，用于短文本极端分类

    InceptionXML: A Lightweight Framework with Synchronized Negative Sampling for Short Text Extreme Classification

    [https://arxiv.org/abs/2109.07319](https://arxiv.org/abs/2109.07319)

    提出了一种轻量级框架InceptionXML，通过在embedding维度上重新分配卷积操作，应对短文本查询中的单词顺序缺失，同时提出了InceptionXML+框架，通过同步标签筛选器和极端分类器，改进了动态硬负采样技术。

    

    短文本数据对大量目标标签进行自动注释，被称为短文本极端分类，已经在许多应用中得到应用，包括相关搜索预测和产品推荐任务。本文提出了一种卷积架构InceptionXML，其轻量但功能强大，并且能够应对搜索和推荐任务中短文本查询中固有的缺乏单词顺序的特点。我们通过将卷积的操作沿着嵌入维度重新构建，而不是像传统CNNs一样沿着单词维度进行文本分类，证明了应用卷积的有效性。为了将我们的模型扩展到具有数百万标签的数据集，我们还提出了InceptionXML+框架，通过同步标签筛选器和极端分类器，改进了最近提出的动态硬负采样技术在标签筛选中的缺陷。

    arXiv:2109.07319v3 Announce Type: replace-cross  Abstract: Automatic annotation of short-text data to a large number of target labels, referred to as Short Text Extreme Classification, has found numerous applications including prediction of related searches and product recommendation tasks. In this paper, we propose a convolutional architecture InceptionXML which is light-weight, yet powerful, and robust to the inherent lack of word-order in short-text queries encountered in search and recommendation tasks. We demonstrate the efficacy of applying convolutions by recasting the operation along the embedding dimension instead of the word dimension as applied in conventional CNNs for text classification. Towards scaling our model to datasets with millions of labels, we also propose InceptionXML+ framework which improves upon the shortcomings of the recently proposed dynamic hard-negative mining technique for label shortlisting by synchronizing the label-shortlister and extreme classifier. 
    
[^9]: 播风撩起风暴：编辑语言模型的影响

    Sowing the Wind, Reaping the Whirlwind: The Impact of Editing Language Models. (arXiv:2401.10647v1 [cs.CL])

    [http://arxiv.org/abs/2401.10647](http://arxiv.org/abs/2401.10647)

    本文研究了通过编辑语言模型的复杂后果，发现在增强模型准确性与保持道德完整性之间存在悖论。我们发现，尽管注入准确信息对模型的可靠性很重要，但它可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。

    

    在人工智能领域中，红队测试或越狱大型语言模型（LLM）的概念已成为一个重要的研究领域。通过对模型进行编辑，揭示了这种修改的复杂后果，发现了增强模型准确性与保持其道德完整性之间的复杂关系。我们的深入分析揭示了一个令人惊讶的悖论：虽然注入准确信息对于模型的可靠性至关重要，但它却可能破坏模型的基本框架，导致不可预测和潜在的不安全行为。此外，我们提出了一个基准数据集NicheHazardQA，用于研究模型在相同和跨领域中的不安全行为。这一方面的研究揭示了编辑如何影响模型的安全度量和保护机制。

    In the rapidly advancing field of artificial intelligence, the concept of Red-Teaming or Jailbreaking large language models (LLMs) has emerged as a crucial area of study. This approach is especially significant in terms of assessing and enhancing the safety and robustness of these models. This paper investigates the intricate consequences of such modifications through model editing, uncovering a complex relationship between enhancing model accuracy and preserving its ethical integrity. Our in-depth analysis reveals a striking paradox: while injecting accurate information is crucial for model reliability, it can paradoxically destabilize the model's foundational framework, resulting in unpredictable and potentially unsafe behaviors. Additionally, we propose a benchmark dataset NicheHazardQA to investigate this unsafe behavior both within the same and cross topical domain. This aspect of our research sheds light on how the edits, impact the model's safety metrics and guardrails. Our find
    
[^10]: 使用基于缓存推理的带状态Conformer模型的流式自动语音识别

    Stateful Conformer with Cache-based Inference for Streaming Automatic Speech Recognition. (arXiv:2312.17279v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.17279](http://arxiv.org/abs/2312.17279)

    本文提出一种基于FastConformer架构的流式语音识别模型，通过限制上下文和引入缓存机制，在推理过程中实现非自回归编码器的自回归操作，并消除了训练和推理准确度间的差异。同时，还提出了CTC/RNNT混合架构以提高准确度和节省计算。

    

    本文提出了一种基于FastConformer架构的高效准确的流式语音识别模型。通过对FastConformer架构进行调整，我们适用于流式应用的方式有两个：（1）限制编码器中的前瞻和历史上下文，（2）引入激活缓存机制以使非自回归编码器在推理过程中以自回归方式工作。所提出的模型经过精心设计，消除了许多流式模型在训练和推理时间中的准确度差异。此外，我们的编码器与不同的解码器配置兼容，包括CTC和RNNT解码器。此外，我们还引入了一种混合的CTC/RNNT架构，它利用共享的编码器和CTC和RNNT解码器来提高准确度并节省计算。我们在LibriSpeech数据集和多领域大型数据集上评估了所提出的模型。

    In this paper, we propose an efficient and accurate streaming speech recognition model based on the FastConformer architecture. We adapted the FastConformer architecture for streaming applications through: (1) constraining both the look-ahead and past contexts in the encoder, and (2) introducing an activation caching mechanism to enable the non-autoregressive encoder to operate autoregressively during inference. The proposed model is thoughtfully designed in a way to eliminate the accuracy disparity between the train and inference time which is common for many streaming models. Furthermore, our proposed encoder works with various decoder configurations including Connectionist Temporal Classification (CTC) and RNN-Transducer (RNNT) decoders. Additionally, we introduced a hybrid CTC/RNNT architecture which utilizes a shared encoder with both a CTC and RNNT decoder to boost the accuracy and save computation. We evaluate the proposed model on LibriSpeech dataset and a multi-domain large sc
    
[^11]: 从神经激活到概念: 解释神经网络中的概念的调查

    From Neural Activations to Concepts: A Survey on Explaining Concepts in Neural Networks. (arXiv:2310.11884v1 [cs.AI])

    [http://arxiv.org/abs/2310.11884](http://arxiv.org/abs/2310.11884)

    本文调查了解释神经网络中概念的最新方法，这对于实现基于可解释概念的神经符号化人工智能来说是重要的一步。

    

    在本文中，我们审查了解释神经网络中概念的最新方法。概念可以作为学习和推理之间的自然桥梁：一旦确定了神经学习系统使用的概念，就可以将这些概念与推理系统整合，用于推理或使用推理系统对其进行改进或增强以改善学习系统。另一方面，不仅可以从神经网络中提取知识，还可以将概念知识插入神经网络体系结构中。由于整合学习和推理是神经符号化人工智能的核心，所以通过这项调查获得的见解可以成为实现基于可解释概念的神经符号化人工智能的重要一步。

    In this paper, we review recent approaches for explaining concepts in neural networks. Concepts can act as a natural link between learning and reasoning: once the concepts are identified that a neural learning system uses, one can integrate those concepts with a reasoning system for inference or use a reasoning system to act upon them to improve or enhance the learning system. On the other hand, knowledge can not only be extracted from neural networks but concept knowledge can also be inserted into neural network architectures. Since integrating learning and reasoning is at the core of neuro-symbolic AI, the insights gained from this survey can serve as an important step towards realizing neuro-symbolic AI based on explainable concepts.
    
[^12]: 语言模型是否能够学习类比推理？研究训练目标和与人类表现的比较。

    Can language models learn analogical reasoning? Investigating training objectives and comparisons to human performance. (arXiv:2310.05597v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05597](http://arxiv.org/abs/2310.05597)

    本文研究了语言模型是否能够学习类比推理的任务，并测试了几种学习方法。实验结果表明，模型能够通过少量数据学习类比推理，并在与人类基准进行比较后接近人类的表现水平。

    

    虽然类比是评估自然语言处理中词嵌入的常见方式，但研究类比推理是否是一种可以学习的任务也很有意义。本文测试了几种学习基本类比推理的方法，特别关注的是那些更符合人类类比推理评估标准的类比。我们的实验发现，模型能够在少量数据的情况下学习类比推理。此外，我们还将我们的模型与具有人类基准的数据集进行比较，并发现在训练后，模型接近人类的表现水平。

    While analogies are a common way to evaluate word embeddings in NLP, it is also of interest to investigate whether or not analogical reasoning is a task in itself that can be learned. In this paper, we test several ways to learn basic analogical reasoning, specifically focusing on analogies that are more typical of what is used to evaluate analogical reasoning in humans than those in commonly used NLP benchmarks. Our experiments find that models are able to learn analogical reasoning, even with a small amount of data. We additionally compare our models to a dataset with a human baseline, and find that after training, models approach human performance.
    
[^13]: RLAdapter：在开放环境中将大型语言模型与强化学习相结合

    RLAdapter: Bridging Large Language Models to Reinforcement Learning in Open Worlds. (arXiv:2309.17176v1 [cs.AI])

    [http://arxiv.org/abs/2309.17176](http://arxiv.org/abs/2309.17176)

    RLAdapter引入了一个框架，将大型语言模型（LLM）与强化学习相结合，以提高在稀疏奖励环境中的策略学习性能。这通过解决LLM在理解下游任务方面的困难，以及通过避免使用不可访问的模型权重或大量计算资源来微调LLM的方式实现。

    

    强化学习在决策问题中取得了显著的成功，但通常需要与环境进行大量的交互，在稀疏奖励环境中学习有意义的策略是具有挑战性的。大型语言模型（LLM）可以为代理提供有价值的指导，从而增强RL算法在这些环境中的性能。然而，LLM通常在理解下游任务方面遇到困难，这阻碍了它们在这些任务中最优地帮助代理的能力。缓解这个问题的常见方法是使用与任务相关的数据来微调LLM，使其能够为RL代理提供有用的指导。然而，这种方法遇到了一些困难，比如无法访问的模型权重或需要大量的计算资源，使其不切实际。在这项工作中，我们引入了RLAdapter，这是一个框架，在RL算法和LLM之间建立更好的连接，从而整合它们的优势和能力。

    While reinforcement learning (RL) shows remarkable success in decision-making problems, it often requires a lot of interactions with the environment, and in sparse-reward environments, it is challenging to learn meaningful policies. Large Language Models (LLMs) can potentially provide valuable guidance to agents in learning policies, thereby enhancing the performance of RL algorithms in such environments. However, LLMs often encounter difficulties in understanding downstream tasks, which hinders their ability to optimally assist agents in these tasks. A common approach to mitigating this issue is to fine-tune the LLMs with task-related data, enabling them to offer useful guidance for RL agents. However, this approach encounters several difficulties, such as inaccessible model weights or the need for significant computational resources, making it impractical. In this work, we introduce RLAdapter, a framework that builds a better connection between RL algorithms and LLMs by incorporating
    
[^14]: MetaMath：为大型语言模型创建自己的数学问题

    MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models. (arXiv:2309.12284v1 [cs.CL])

    [http://arxiv.org/abs/2309.12284](http://arxiv.org/abs/2309.12284)

    MetaMath是一种专门用于数学推理的微调语言模型，通过从多个角度重新编写问题来生成数学问题，并在两个基准测试中取得了优于其他开源语言模型的表现。

    

    大型语言模型（LLMs）推动了自然语言理解的极限，并展示了出色的问题解决能力。尽管取得了巨大的成功，但大多数现有的开源LLMs（例如LLaMA-2）在解决数学问题方面仍然远远不够令人满意，原因是复杂的推理过程。为了弥合这一鸿沟，我们提出了MetaMath，一种专门用于数学推理的微调语言模型。具体而言，我们通过在没有额外知识的情况下以多个角度重新写入问题来引导数学问题，从而产生了一个名为MetaMathQA的新数据集。然后我们在MetaMathQA上对LLaMA-2模型进行了微调。对于数学推理的两个流行基准测试（即GSM8K和MATH），实验结果表明MetaMath在性能上明显优于一套开源LLMs。我们的MetaMath-7B模型在GSM8K上达到了66.4％，在MATH上达到了19.4％，超过了相同规模的最先进模型。

    Large language models (LLMs) have pushed the limits of natural language understanding and exhibited excellent problem-solving ability. Despite the great success, most existing open-source LLMs (\eg, LLaMA-2) are still far away from satisfactory for solving mathematical problem due to the complex reasoning procedures. To bridge this gap, we propose \emph{MetaMath}, a fine-tuned language model that specializes in mathematical reasoning. Specifically, we start by bootstrapping mathematical questions by rewriting the question from multiple perspectives without extra knowledge, which results in a new dataset called {MetaMathQA}. Then we fine-tune the LLaMA-2 models on MetaMathQA. Experimental results on two popular benchmarks (\ie, GSM8K and MATH) for mathematical reasoning demonstrate that MetaMath outperforms a suite of open-source LLMs by a significant margin. Our MetaMath-7B model achieves $66.4\%$ on GSM8K and $19.4\%$ on MATH, exceeding the state-of-the-art models of the same size by 
    

