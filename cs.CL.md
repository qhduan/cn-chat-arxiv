# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transcription free filler word detection with Neural semi-CRFs.](http://arxiv.org/abs/2303.06475) | 本文提出了一种无需转录的填充词检测系统，使用结构化状态空间序列模型和神经半马尔可夫条件随机场，能够在PodcastFillers数据集上实现6.4％（分段级别）和3.1％（事件级别）的绝对F1改进。 |
| [^2] | [ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation.](http://arxiv.org/abs/2303.06458) | ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。 |
| [^3] | [Parachute: Evaluating Interactive Human-LM Co-writing Systems.](http://arxiv.org/abs/2303.06333) | 本文提出了一个以人为中心的评估框架Parachute，用于交互式共同撰写系统的评估，该框架包含了分类的实用指标，可以用于评估和比较共同撰写系统。 |
| [^4] | [Stabilizing Transformer Training by Preventing Attention Entropy Collapse.](http://arxiv.org/abs/2303.06296) | 本文研究了Transformer的训练动态，发现低注意力熵伴随着高训练不稳定性，提出了一种简单而有效的解决方案$\sigma$Reparam，成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。 |
| [^5] | [Consistency Analysis of ChatGPT.](http://arxiv.org/abs/2303.06273) | 本文研究了ChatGPT的一致性问题，发现尽管它具有更好的语言理解能力，但仍然经常无法生成逻辑上正确的预测。因此，在现实世界的应用需要进一步考虑，特别是在风险方面。 |
| [^6] | [An Interactive UI to Support Sensemaking over Collections of Parallel Texts.](http://arxiv.org/abs/2303.06264) | AVTALER是一种交互式UI，结合了人类的独特技能和自动化的优势，支持用户对可比较的文本摘录进行意义建构和对比。 |
| [^7] | [AUTODIAL: Efficient Asynchronous Task-Oriented Dialogue Model.](http://arxiv.org/abs/2303.06245) | AUTODIAL是一种多任务对话模型，通过使用并行解码器来执行对话任务，从而显著减少内存占用并实现更快的推理时间。与现有的生成方法相比，AUTODIAL在三个对话任务上提供了3-6倍的速度提升，同时具有11倍的参数减少。这表明将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。 |
| [^8] | [Generating Query Focused Summaries without Fine-tuning the Transformer-based Pre-trained Models.](http://arxiv.org/abs/2303.06230) | 本文研究了不使用微调基于Transformer的预训练模型生成查询聚焦摘要的方法，使用边际最大相关性（MMR）方法直接从新数据集中获得查询聚焦摘要，避免了微调步骤，减少了计算时间和成本。 |
| [^9] | [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.](http://arxiv.org/abs/2303.06182) | 本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。 |
| [^10] | [Clinical BERTScore: An Improved Measure of Automatic Speech Recognition Performance in Clinical Settings.](http://arxiv.org/abs/2303.05737) | 本文提出了一种临床BERTScore（CBERTScore）度量，它比其他度量更严厉地惩罚临床相关的错误，更接近于临床医生对医学句子的偏好。作者还收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。 |
| [^11] | [Learning the Legibility of Visual Text Perturbations.](http://arxiv.org/abs/2303.05077) | 本文提出了一种学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名的方法，填补了保持易读性的文本扰动的系统性表征的空白。 |
| [^12] | [NASTyLinker: NIL-Aware Scalable Transformer-based Entity Linker.](http://arxiv.org/abs/2303.04426) | NASTyLinker是一种NIL感知的实体链接器，它通过生成提及簇来表示NIL实体，并在保持已知实体高链接性能的同时解决冲突。 |
| [^13] | [Reducing Spurious Correlations for Aspect-Based Sentiment Analysis with Variational Information Bottleneck and Contrastive Learning.](http://arxiv.org/abs/2303.02846) | 本文提出了一种新的对比变分信息瓶颈框架（CVIB），以减少方面情感分析（ABSA）中的虚假相关性。该框架由一个原始网络和一个自剪枝网络组成，通过对比学习同时进行优化，从而丢弃了输入特征和预测标签之间的多余模式或虚假相关性。 |
| [^14] | [FinXABSA: Explainable Finance through Aspect-Based Sentiment Analysis.](http://arxiv.org/abs/2303.02563) | 本文提出了一种基于方面的情感分析方法，通过与股票价格的相关性建立关系，实现金融分析的可解释性。该方法提供了更详细和准确的了解情感分析与股票价格之间关系的方法，对于投资者和金融分析师做出明智决策非常有用。 |
| [^15] | [Rethinking the Reasonability of the Test Set for Simultaneous Machine Translation.](http://arxiv.org/abs/2303.00969) | 本文提出了一个手动注释的单调测试集SiMuST-C，用于评估同时机器翻译（SimulMT）模型的性能，实验证明该测试集可以缓解低估问题，并且在自动提取的单调训练集上微调可以将SimulMT模型的性能提高3个BLEU分数。 |
| [^16] | [A Survey on Event-based News Narrative Extraction.](http://arxiv.org/abs/2302.08351) | 本文综述了事件驱动新闻叙事提取的研究，通过筛选超过900篇文章，得到了54篇相关文章，这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。 |
| [^17] | ["Correct answers" from the psychology of artificial intelligence.](http://arxiv.org/abs/2302.07267) | 本文使用OpenAI的GPT3.5模型重新复制了Many Labs 2复制项目中的14项研究，其中8项研究的结果被成功复制。然而，对于剩下的6项研究，GPT3.5以极其预定的方式回答了调查问题，导致无法分析这些研究。 |
| [^18] | [AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models.](http://arxiv.org/abs/2302.07027) | 本文提出了AdapterSoup，一种使用加权平均改善预训练语言模型泛化能力的方法。该方法在不同领域训练的适配器上执行权重空间平均，可以在不需要额外训练的情况下提高对新领域的性能。 |
| [^19] | [Adaptive Machine Translation with Large Language Models.](http://arxiv.org/abs/2301.13294) | 本文研究了如何利用大型语言模型的上下文学习来改进实时自适应机器翻译，实验结果表明有希望的效果。 |
| [^20] | [Unifying Vision, Text, and Layout for Universal Document Processing.](http://arxiv.org/abs/2212.02623) | 本文提出了通用文档处理（UDOP）模型，将文本、图像和布局模态以及各种任务格式统一起来，通过一种新颖的Transformer模型实现预训练和多域下游任务的统一，同时实现了高质量的神经文档编辑和内容定制。 |
| [^21] | [Neural Transducer Training: Reduced Memory Consumption with Sample-wise Computation.](http://arxiv.org/abs/2211.16270) | 本文提出了一种内存高效的神经转录器训练方法，采用逐个样本计算转录器损失和梯度，显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。 |
| [^22] | [Automatically Extracting Information in Medical Dialogue: Expert System And Attention for Labelling.](http://arxiv.org/abs/2211.15544) | 本文提出了一种新颖的模型ESAL，使用专家混合和预训练的BERT来检索不同类别的语义，使模型能够融合它们之间的差异。实验结果表明ESAL显著提高了医疗信息分类的性能。 |
| [^23] | [Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models.](http://arxiv.org/abs/2211.05103) | 本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。预训练的语音模型在较低层中最优地编码了语言区分信息，从这些层获得的嵌入能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果，且使用的参数量仅为其它模型的五分之一。 |
| [^24] | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model.](http://arxiv.org/abs/2211.05100) | BLOOM是一个由数百名研究人员合作设计和构建的拥有176B参数的开放式语言模型，它在多种基准测试中表现出竞争性的性能，并在进行多任务提示微调后表现更强。该模型的发布有助于推动大型语言模型技术的民主化。 |
| [^25] | [Using Emotion Embeddings to Transfer Knowledge Between Emotions, Languages, and Annotation Formats.](http://arxiv.org/abs/2211.00171) | 本文研究了如何通过利用多语言模型和Demux来构建一个可以在不同情感、语言和注释格式之间转换的单一模型，以实现知识共享和降低训练成本。 |
| [^26] | [Leveraging Label Correlations in a Multi-label Setting: A Case Study in Emotion.](http://arxiv.org/abs/2210.15842) | 本文研究了利用标签相关性来改善情感检测的方法，开发了两种建模方法来捕捉情感词本身的词汇关联性，并将情感表示的成对约束作为正则化项与模型的分类损失一起集成，展示了在SemEval 2018任务1 E-c中使用单语BERT模型展示了西班牙语、英语和阿拉伯语的最新性能。 |
| [^27] | [Articulation GAN: Unsupervised modeling of articulatory learning.](http://arxiv.org/abs/2210.15173) | 本文提出了一种新的无监督生成模型，通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式，从而更好地模拟人类语音产生的过程。 |
| [^28] | [Named Entity Detection and Injection for Direct Speech Translation.](http://arxiv.org/abs/2210.11981) | 本文研究了如何利用命名实体字典来改进语音到文本翻译模型的输出，实验表明可以通过减少31%的人名错误来提高翻译中的命名实体准确性。 |
| [^29] | [Augmentation with Projection: Towards an Effective and Efficient Data Augmentation Paradigm for Distillation.](http://arxiv.org/abs/2210.11768) | 提出了一种名为AugPro的数据增强方法，它是一种有效且高效的蒸馏数据增强方法，可以避免偏移决策边界，计算开销小。 |
| [^30] | [Attribution and Obfuscation of Neural Text Authorship: A Data Mining Perspective.](http://arxiv.org/abs/2210.10488) | 本文调查了神经文本生成技术在作者归属和混淆中的应用，提出了需要开发新型AA / AO解决方案来处理神经文本的问题。 |
| [^31] | [EDU-level Extractive Summarization with Varying Summary Lengths.](http://arxiv.org/abs/2210.04029) | 本文提出了一种EDU级别的可变长度摘要提取模型，通过比较分析证明EDU比句子具有更高的自动评估分数。 |
| [^32] | [The Surprising Computational Power of Nondeterministic Stack RNNs.](http://arxiv.org/abs/2210.01343) | 本文展示了非确定性堆栈循环神经网络的惊人计算能力，它不仅可以识别上下文无关语言，还可以识别许多非上下文无关语言，并且可以识别具有比其堆栈字母表大小更大的语言。 |
| [^33] | [Assessing the impact of contextual information in hate speech detection.](http://arxiv.org/abs/2210.00465) | 本文提供了一个基于Twitter上媒体机构新闻帖子的用户响应的上下文化仇恨言论检测新语料库，以解决当前自动检测仇恨言论方法缺乏上下文的限制。 |
| [^34] | [Learning ASR pathways: A sparse multilingual ASR model.](http://arxiv.org/abs/2209.05735) | 本文提出了一种稀疏的多语言ASR模型，通过激活语言特定的子网络来显式地学习每种语言的参数，同时通过联合多语言训练实现对低资源语言的知识转移，相比于密集模型和语言不可知的剪枝模型，在低资源语言上提供更好的性能。 |
| [^35] | [DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech.](http://arxiv.org/abs/2207.01063) | 本文介绍了一个高质量的对话语音数据集DailyTalk，专门为对话TTS设计。DailyTalk可以用作通用的TTS数据集，而且基线可以表示DailyTalk的上下文信息。 |
| [^36] | [EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing.](http://arxiv.org/abs/2205.00258) | EasyNLP是一款全面且易于使用的自然语言处理工具包，支持全面的NLP算法套件，具有知识增强的预训练、知识蒸馏和少样本学习功能，用于大规模PTMs，并为实际应用提供了模型训练、推理和部署的统一框架。 |
| [^37] | [A Token-level Contrastive Framework for Sign Language Translation.](http://arxiv.org/abs/2204.04916) | 提出了一种基于对比学习的手语翻译框架ConSLT，通过将标记级对比学习纳入SLT解码过程中，学习有效的标记表示，以缓解公开可用的手语翻译语料库非常有限的问题。 |
| [^38] | [Alternate Intermediate Conditioning with Syllable-level and Character-level Targets for Japanese ASR.](http://arxiv.org/abs/2204.00175) | 该论文提出了一种基于音节和字符目标的交替中间条件方法，利用字符级和音节级中间预测作为条件特征来处理日语ASR中的多对一和一对多的映射问题，并在实验中取得了优异的表现。 |
| [^39] | [I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning.](http://arxiv.org/abs/2202.06574) | 本文提出了一种轻量级的图像字幕生成框架（I-Tuning），通过设计新颖的交叉注意力模块将不可训练的预训练语言解码器和视觉编码器连接起来，使得模型包含的可训练参数少，训练速度快，同时在三个图像字幕生成基准测试上实现了与大规模基线系统相当或更好的性能，但需要的可训练参数和训练数据量都少得多。 |
| [^40] | [Temporal Sentence Grounding in Videos: A Survey and Future Directions.](http://arxiv.org/abs/2201.08071) | 本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。 |
| [^41] | [Two-view Graph Neural Networks for Knowledge Graph Completion.](http://arxiv.org/abs/2112.09231) | 本文提出了一种名为WGE的图神经网络模型，通过两个单一的实体和关系为中心的图来学习实体和关系的向量表示，并在知识图谱补全任务上取得了优异的性能。 |
| [^42] | [Pre-trained Language Models in Biomedical Domain: A Systematic Survey.](http://arxiv.org/abs/2110.05006) | 本文系统调查了生物医学领域中的预训练语言模型，总结了它们的最新进展和应用，并提出了分类法。 |
| [^43] | [Self-Attention Networks Can Process Bounded Hierarchical Languages.](http://arxiv.org/abs/2105.11115) | 本文证明了自注意力网络可以处理深度受限的$\mathsf{Dyck}_{k}$子集$\mathsf{Dyck}_{k,D}$，这更好地捕捉了自然语言的有界层次结构。 |

# 详细

[^1]: 基于神经半条件随机场的无需转录的填充词检测

    Transcription free filler word detection with Neural semi-CRFs. (arXiv:2303.06475v1 [eess.AS])

    [http://arxiv.org/abs/2303.06475](http://arxiv.org/abs/2303.06475)

    本文提出了一种无需转录的填充词检测系统，使用结构化状态空间序列模型和神经半马尔可夫条件随机场，能够在PodcastFillers数据集上实现6.4％（分段级别）和3.1％（事件级别）的绝对F1改进。

    This paper proposes a transcription-free filler word detection system that uses structured state space sequence model and neural semi-Markov conditional random fields, achieving an absolute F1 improvement of 6.4% (segment level) and 3.1% (event level) on the PodcastFillers dataset.

    非语言填充词，如“嗯”或“啊”，在自发语言中普遍存在，用于表达犹豫或不确定性。以前检测某些非语言填充词的工作高度依赖于来自成熟商业自动语音识别（ASR）系统的转录。然而，某些ASR系统在许多方面（例如预算、目标语言和计算能力）并不普遍可用。在这项工作中，我们研究了不依赖于ASR系统的填充词检测系统。我们展示了通过使用结构化状态空间序列模型（S4）和神经半马尔可夫条件随机场（semi-CRFs），我们在PodcastFillers数据集上实现了6.4％（分段级别）和3.1％（事件级别）的绝对F1改进。我们还对检测结果进行了定性分析，以分析我们提出的系统的局限性。

    Non-linguistic filler words, such as "uh" or "um", are prevalent in spontaneous speech and serve as indicators for expressing hesitation or uncertainty. Previous works for detecting certain non-linguistic filler words are highly dependent on transcriptions from a well-established commercial automatic speech recognition (ASR) system. However, certain ASR systems are not universally accessible from many aspects, e.g., budget, target languages, and computational power. In this work, we investigate filler word detection system that does not depend on ASR systems. We show that, by using the structured state space sequence model (S4) and neural semi-Markov conditional random fields (semi-CRFs), we achieve an absolute F1 improvement of 6.4% (segment level) and 3.1% (event level) on the PodcastFillers dataset. We also conduct a qualitative analysis on the detected results to analyze the limitations of our proposed system.
    
[^2]: ZeroNLG: 将领域对齐和自编码用于零样本多模态和多语言自然语言生成

    ZeroNLG: Aligning and Autoencoding Domains for Zero-Shot Multimodal and Multilingual Natural Language Generation. (arXiv:2303.06458v1 [cs.CL])

    [http://arxiv.org/abs/2303.06458](http://arxiv.org/abs/2303.06458)

    ZeroNLG是一个零样本学习框架，可以处理多个NLG任务，包括图像到文本、视频到文本和文本到文本，跨越英语、中文、德语和法语。它不需要任何标记的下游对进行训练，通过将不同的领域投影到共享的公共潜在空间中的相应坐标，桥接不同领域之间的差异。

    ZeroNLG is a zero-shot learning framework that can handle multiple NLG tasks, including image-to-text, video-to-text, and text-to-text, across English, Chinese, German, and French. It does not require any labeled downstream pairs for training, and bridges the differences between different domains by projecting them to corresponding coordinates in a shared common latent space.

    自然语言生成（NLG）接受以图像、视频或文本形式的输入数据，并生成相应的自然语言文本作为输出。现有的NLG方法主要采用监督方法，并且严重依赖于耦合的数据到文本对。然而，对于许多有针对性的场景和非英语语言，往往没有足够数量的标记数据。为了放松对下游任务标记数据的依赖性，我们提出了一个直观有效的零样本学习框架ZeroNLG，它可以处理多个NLG任务，包括图像到文本（图像字幕）、视频到文本（视频字幕）和文本到文本（神经机器翻译），跨越英语、中文、德语和法语在一个统一的框架内。ZeroNLG不需要任何标记的下游对进行训练。在训练期间，ZeroNLG（i）将不同的领域（跨模态和语言）投影到共享的公共潜在空间中的相应坐标；（ii）桥接差异

    Natural Language Generation (NLG) accepts input data in the form of images, videos, or text and generates corresponding natural language text as output. Existing NLG methods mainly adopt a supervised approach and rely heavily on coupled data-to-text pairs. However, for many targeted scenarios and for non-English languages, sufficient quantities of labeled data are often not available. To relax the dependency on labeled data of downstream tasks, we propose an intuitive and effective zero-shot learning framework, ZeroNLG, which can deal with multiple NLG tasks, including image-to-text (image captioning), video-to-text (video captioning), and text-to-text (neural machine translation), across English, Chinese, German, and French within a unified framework. ZeroNLG does not require any labeled downstream pairs for training. During training, ZeroNLG (i) projects different domains (across modalities and languages) to corresponding coordinates in a shared common latent space; (ii) bridges diff
    
[^3]: 降落伞：评估交互式人机共同撰写系统

    Parachute: Evaluating Interactive Human-LM Co-writing Systems. (arXiv:2303.06333v1 [cs.HC])

    [http://arxiv.org/abs/2303.06333](http://arxiv.org/abs/2303.06333)

    本文提出了一个以人为中心的评估框架Parachute，用于交互式共同撰写系统的评估，该框架包含了分类的实用指标，可以用于评估和比较共同撰写系统。

    This paper proposes a human-centered evaluation framework, Parachute, for interactive co-writing systems, which includes categorized practical metrics and can be used to evaluate and compare co-writing systems.

    语言模型的飞速发展引起了人们对于利用语言模型构建共同撰写系统的极大兴趣，其中人类和语言模型交互地为共同的写作成果做出贡献。然而，缺乏对于交互式环境下共同撰写系统的评估研究。我们提出了一个以人为中心的评估框架Parachute，用于交互式共同撰写系统的评估。Parachute展示了交互评估的综合视角，其中每个评估方面都包含了分类的实用指标。此外，我们提供了一个使用案例来演示如何使用Parachute评估和比较共同撰写系统。

    A surge of advances in language models (LMs) has led to significant interest in using LMs to build co-writing systems, in which humans and LMs interactively contribute to a shared writing artifact. However, there is a lack of studies assessing co-writing systems in interactive settings. We propose a human-centered evaluation framework, Parachute, for interactive co-writing systems. Parachute showcases an integrative view of interaction evaluation, where each evaluation aspect consists of categorized practical metrics. Furthermore, we present Parachute with a use case to demonstrate how to evaluate and compare co-writing systems using Parachute.
    
[^4]: 防止注意力熵崩溃的Transformer训练稳定性研究

    Stabilizing Transformer Training by Preventing Attention Entropy Collapse. (arXiv:2303.06296v1 [cs.LG])

    [http://arxiv.org/abs/2303.06296](http://arxiv.org/abs/2303.06296)

    本文研究了Transformer的训练动态，发现低注意力熵伴随着高训练不稳定性，提出了一种简单而有效的解决方案$\sigma$Reparam，成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。

    This paper investigates the training dynamics of Transformers and proposes a simple and efficient solution, $\sigma$Reparam, to prevent entropy collapse in the attention layers, promoting more stable training.

    训练稳定性对于Transformer至关重要。本文通过研究注意力层的演变来探究Transformer的训练动态。特别地，我们在训练过程中跟踪每个注意力头的注意力熵，这是模型锐度的代理。我们发现，在不同的架构和任务中存在一种常见模式，即低注意力熵伴随着高训练不稳定性，这可能采取振荡损失或发散的形式。我们将病态低注意力熵，对应高度集中的注意力分数，称为$\textit{熵崩溃}$。作为一种解决方案，我们提出了$\sigma$Reparam，一种简单而有效的解决方案，其中我们使用谱归一化和额外的学习标量重新参数化所有线性层。我们证明了所提出的重新参数化成功地防止了注意力层中的熵崩溃，促进了更稳定的训练。此外，我们

    Training stability is of great importance to Transformers. In this work, we investigate the training dynamics of Transformers by examining the evolution of the attention layers. In particular, we track the attention entropy for each attention head during the course of training, which is a proxy for model sharpness. We identify a common pattern across different architectures and tasks, where low attention entropy is accompanied by high training instability, which can take the form of oscillating loss or divergence. We denote the pathologically low attention entropy, corresponding to highly concentrated attention scores, as $\textit{entropy collapse}$. As a remedy, we propose $\sigma$Reparam, a simple and efficient solution where we reparametrize all linear layers with spectral normalization and an additional learned scalar. We demonstrate that the proposed reparameterization successfully prevents entropy collapse in the attention layers, promoting more stable training. Additionally, we 
    
[^5]: ChatGPT的一致性分析

    Consistency Analysis of ChatGPT. (arXiv:2303.06273v1 [cs.CL])

    [http://arxiv.org/abs/2303.06273](http://arxiv.org/abs/2303.06273)

    本文研究了ChatGPT的一致性问题，发现尽管它具有更好的语言理解能力，但仍然经常无法生成逻辑上正确的预测。因此，在现实世界的应用需要进一步考虑，特别是在风险方面。

    This paper investigates the consistency issue of ChatGPT and finds that although it has improved language understanding ability, it frequently fails to generate logically correct predictions. Therefore, further consideration is needed for its real-world applications, especially in terms of risk.

    ChatGPT是一种基于大型语言模型的问答对话系统，自推出以来广受欢迎。虽然它在法律、医学和金融等领域的专业考试中取得了不错的成绩，但也有人对其可靠性和信任度表示怀疑。本文针对ChatGPT在逻辑一致性方面的可信度进行了调查研究。我们的研究发现，尽管ChatGPT似乎具有更好的语言理解能力，但它仍然经常无法生成逻辑上正确的预测。因此，虽然ChatGPT是一种令人印象深刻和有前途的新技术，但我们得出结论，如果没有经过彻底的人工检查，它在现实世界的应用需要进一步考虑，特别是在风险方面。

    ChatGPT, a question-and-answer dialogue system based on a large language model, has gained huge popularity since its introduction. Its positive aspects have been reported through many media platforms, and some analyses even showed that ChatGPT achieved a decent grade in professional exams, including the law, medical, and finance domains, adding extra support to the claim that AI now can assist and, even, replace humans in industrial fields. Others, however, doubt its reliability and trustworthiness. In this paper, we investigate ChatGPT's trustworthiness regarding logically consistent behaviours. Our findings suggest that, although ChatGPT seems to achieve an improved language understanding ability, it still fails to generate logically correct predictions frequently. Hence, while it is true that ChatGPT is an impressive and promising new technique, we conclude that its usage in real-world applications without thorough human inspection requires further consideration, especially for risk
    
[^6]: 一种支持对平行文本集合进行意义建构的交互式UI

    An Interactive UI to Support Sensemaking over Collections of Parallel Texts. (arXiv:2303.06264v1 [cs.HC])

    [http://arxiv.org/abs/2303.06264](http://arxiv.org/abs/2303.06264)

    AVTALER是一种交互式UI，结合了人类的独特技能和自动化的优势，支持用户对可比较的文本摘录进行意义建构和对比。

    AVTALER is an interactive UI that combines human skills and the advantages of automation to support users in sensemaking and contrasting comparable text excerpts.

    科学家和科学记者等人经常需要理解大量论文及其在范围、重点、发现或其他重要因素方面的比较。然而，对于大量的论文，逐一进行比较和对比是认知上具有挑战性的。完全自动化这个审查过程是不可行的，因为它通常需要领域特定的知识，以及理解审查的背景和动机。虽然有现有的工具来帮助组织和注释文献综述的论文，但它们仍然依赖于人们逐个阅读论文并手动理解相关信息。我们提出了AVTALER，它结合了人们独特的技能、上下文意识和知识，以及自动化的优势。给定一组可比较的文本摘录，它支持用户进行意义建构和对比。

    Scientists and science journalists, among others, often need to make sense of a large number of papers and how they compare with each other in scope, focus, findings, or any other important factors. However, with a large corpus of papers, it's cognitively demanding to pairwise compare and contrast them all with each other. Fully automating this review process would be infeasible, because it often requires domain-specific knowledge, as well as understanding what the context and motivations for the review are. While there are existing tools to help with the process of organizing and annotating papers for literature reviews, at the core they still rely on people to serially read through papers and manually make sense of relevant information.  We present AVTALER, which combines peoples' unique skills, contextual awareness, and knowledge, together with the strength of automation. Given a set of comparable text excerpts from a paper corpus, it supports users in sensemaking and contrasting pa
    
[^7]: AUTODIAL: 高效异步任务导向的对话模型

    AUTODIAL: Efficient Asynchronous Task-Oriented Dialogue Model. (arXiv:2303.06245v1 [cs.CL])

    [http://arxiv.org/abs/2303.06245](http://arxiv.org/abs/2303.06245)

    AUTODIAL是一种多任务对话模型，通过使用并行解码器来执行对话任务，从而显著减少内存占用并实现更快的推理时间。与现有的生成方法相比，AUTODIAL在三个对话任务上提供了3-6倍的速度提升，同时具有11倍的参数减少。这表明将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。

    AUTODIAL is a multi-task dialogue model that significantly reduces memory footprint and achieves faster inference times by using parallel decoders to perform dialogue tasks. Compared to existing generative approach, AUTODIAL provides 3-6x speedups during inference while having 11x fewer parameters on three dialogue tasks. This suggests that extending current dialogue models to have parallel decoders can be a viable alternative for deploying them in resource-constrained environments.

    随着大型对话模型在实践中变得普遍，训练、推理和更大的内存占用的高计算要求问题仍然存在。在这项工作中，我们提出了AUTODIAL，一种多任务对话模型，解决了部署对话模型的挑战。AUTODIAL利用并行解码器执行诸如对话行为预测、领域预测、意图预测和对话状态跟踪等任务。使用分类解码器而不是生成解码器使AUTODIAL能够显著减少内存占用，并在推理时间上实现比现有生成方法（即SimpleTOD）更快的速度。我们证明，将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。

    As large dialogue models become commonplace in practice, the problems surrounding high compute requirements for training, inference and larger memory footprint still persists. In this work, we present AUTODIAL, a multi-task dialogue model that addresses the challenges of deploying dialogue model. AUTODIAL utilizes parallel decoders to perform tasks such as dialogue act prediction, domain prediction, intent prediction, and dialogue state tracking. Using classification decoders over generative decoders allows AUTODIAL to significantly reduce memory footprint and achieve faster inference times compared to existing generative approach namely SimpleTOD. We demonstrate that AUTODIAL provides 3-6x speedups during inference while having 11x fewer parameters on three dialogue tasks compared to SimpleTOD. Our results show that extending current dialogue models to have parallel decoders can be a viable alternative for deploying them in resource-constrained environments.
    
[^8]: 不使用微调基于Transformer的预训练模型生成查询聚焦摘要

    Generating Query Focused Summaries without Fine-tuning the Transformer-based Pre-trained Models. (arXiv:2303.06230v1 [cs.CL])

    [http://arxiv.org/abs/2303.06230](http://arxiv.org/abs/2303.06230)

    本文研究了不使用微调基于Transformer的预训练模型生成查询聚焦摘要的方法，使用边际最大相关性（MMR）方法直接从新数据集中获得查询聚焦摘要，避免了微调步骤，减少了计算时间和成本。

    This paper investigates the method of generating query-focused summaries without fine-tuning transformer-based pre-trained models, using the Marginal Maximum Relevance (MMR) approach to obtain query-focused summaries directly from a new dataset without fine-tuning, reducing computational time and cost.

    对于每个新数据集，微调自然语言处理（NLP）模型需要更高的计算时间，伴随着增加的碳足迹和成本。然而，微调有助于预训练模型适应最新的数据集；如果我们避免微调步骤，仅使用预训练模型尝试生成摘要以减少计算时间和成本，会怎样呢？在本文中，我们尝试省略微调步骤，并调查边际最大相关性（MMR）方法是否可以帮助预训练模型直接从未用于预训练模型的新数据集中获得查询聚焦摘要。首先，我们使用主题建模在维基百科当前事件门户（WCEP）和Debatepedia数据集上生成摘要任务的查询。然后，使用MMR，我们根据查询对文档的句子进行排名。接下来，我们将排名后的句子传递给七个基于Transformer的预训练模型来执行摘要。

    Fine-tuning the Natural Language Processing (NLP) models for each new data set requires higher computational time associated with increased carbon footprint and cost. However, fine-tuning helps the pre-trained models adapt to the latest data sets; what if we avoid the fine-tuning steps and attempt to generate summaries using just the pre-trained models to reduce computational time and cost. In this paper, we tried to omit the fine-tuning steps and investigate whether the Marginal Maximum Relevance (MMR)-based approach can help the pre-trained models to obtain query-focused summaries directly from a new data set that was not used to pre-train the models. First, we used topic modelling on Wikipedia Current Events Portal (WCEP) and Debatepedia datasets to generate queries for summarization tasks. Then, using MMR, we ranked the sentences of the documents according to the queries. Next, we passed the ranked sentences to seven transformer-based pre-trained models to perform the summarization
    
[^9]: 迈向MoE部署：缓解混合专家（MoE）推理中的低效率

    Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference. (arXiv:2303.06182v1 [cs.DC])

    [http://arxiv.org/abs/2303.06182](http://arxiv.org/abs/2303.06182)

    本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。

    This paper proposes three optimization techniques to mitigate inefficiencies in Mixture-of-Experts (MoE) models during inference, including dynamic gating, expert buffering, and expert load balancing. These techniques can significantly improve execution time and reduce memory usage.

    混合专家（MoE）模型最近在计算机视觉和自然语言处理的广泛任务中取得了最先进的性能。它们在训练期间有效地扩展了模型容量，同时增加的计算成本很小。然而，由于其庞大的模型大小和复杂的通信模式，部署这样的模型进行推理是困难的。在这项工作中，我们提供了两个MoE工作负载的特征化，即语言建模（LM）和机器翻译（MT），并确定了它们在部署时的低效率来源。我们提出了三种优化技术来缓解低效率的来源，即（1）动态门控，（2）专家缓冲和（3）专家负载平衡。我们展示了动态门控可以使LM的执行时间提高1.25-4倍，MT编码器提高2-5倍，MT解码器提高1.09-1.5倍。它还可以将LM的内存使用减少高达1.36倍，MT的内存使用减少高达1.1倍。

    Mixture-of-Experts (MoE) models have recently gained steam in achieving the state-of-the-art performance in a wide range of tasks in computer vision and natural language processing. They effectively expand the model capacity while incurring a minimal increase in computation cost during training. However, deploying such models for inference is difficult due to their large model size and complex communication pattern. In this work, we provide a characterization of two MoE workloads, namely Language Modeling (LM) and Machine Translation (MT) and identify their sources of inefficiencies at deployment.  We propose three optimization techniques to mitigate sources of inefficiencies, namely (1) Dynamic gating, (2) Expert Buffering, and (3) Expert load balancing. We show that dynamic gating improves execution time by 1.25-4$\times$ for LM, 2-5$\times$ for MT Encoder and 1.09-1.5$\times$ for MT Decoder. It also reduces memory usage by up to 1.36$\times$ for LM and up to 1.1$\times$ for MT. We f
    
[^10]: 临床BERTScore：临床环境下自动语音识别性能的改进度量

    Clinical BERTScore: An Improved Measure of Automatic Speech Recognition Performance in Clinical Settings. (arXiv:2303.05737v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2303.05737](http://arxiv.org/abs/2303.05737)

    本文提出了一种临床BERTScore（CBERTScore）度量，它比其他度量更严厉地惩罚临床相关的错误，更接近于临床医生对医学句子的偏好。作者还收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。

    The paper proposes a Clinical BERTScore (CBERTScore) metric for ASR in medical contexts, which penalizes clinically-relevant mistakes more than other metrics and aligns more closely with clinician preferences. The authors also collect a benchmark of clinician preferences on medical sentences and release it for the community to further develop clinically-aware ASR metrics.

    医学环境中的自动语音识别（ASR）有潜力节省时间，降低成本，提高报告准确性并减少医生的疲劳。然而，由于避免医学相关的转录错误的重要性，医疗行业采用这种技术的速度较慢。在这项工作中，我们提出了临床BERTScore（CBERTScore），这是一种ASR度量，它比其他度量（WER、BLUE、METEOR等）更严厉地惩罚临床相关的错误。我们证明了这个度量更接近于临床医生对医学句子的偏好，有时差距很大。我们收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。

    Automatic Speech Recognition (ASR) in medical contexts has the potential to save time, cut costs, increase report accuracy, and reduce physician burnout. However, the healthcare industry has been slower to adopt this technology, in part due to the importance of avoiding medically-relevant transcription mistakes. In this work, we present the Clinical BERTScore (CBERTScore), an ASR metric that penalizes clinically-relevant mistakes more than others. We demonstrate that this metric more closely aligns with clinician preferences on medical sentences as compared to other metrics (WER, BLUE, METEOR, etc), sometimes by wide margins. We collect a benchmark of 13 clinician preferences on 149 realistic medical sentences called the Clinician Transcript Preference benchmark (CTP), demonstrate that CBERTScore more closely matches what clinicians prefer, and release the benchmark for the community to further develop clinically-aware ASR metrics.
    
[^11]: 学习视觉文本扰动的易读性

    Learning the Legibility of Visual Text Perturbations. (arXiv:2303.05077v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.05077](http://arxiv.org/abs/2303.05077)

    本文提出了一种学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名的方法，填补了保持易读性的文本扰动的系统性表征的空白。

    This paper proposes a method to learn models that predict the legibility of a perturbed string and rank candidate perturbations based on their legibility, filling the gap in systematically characterizing the legibility of text perturbations while preserving it.

    许多NLP中的对抗性攻击会扰动输入以产生在视觉上相似但对模型性能有负面影响的字符串（例如'ergo' $\rightarrow$ '$\epsilon$rgo'），这些字符串对人类来说是易读的。尽管保持易读性是文本扰动的必要条件，但很少有工作对其进行系统性的表征；相反，易读性通常通过对扰动的性质和程度的直觉约束来实现。特别地，尚不清楚在保持易读性的情况下可以扰动多少输入，或如何量化扰动字符串的易读性。在这项工作中，我们通过学习模型来预测扰动字符串的易读性，并根据其易读性对候选扰动进行排名，以填补这一空白。为此，我们收集并发布了LEGIT，一个人类注释的数据集，其中包括视觉上扰动文本的易读性。使用这个数据集，我们构建了基于文本和视觉的模型，可以达到$0.91$的F1分数，以预测输入是否易读。

    Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input
    
[^12]: NASTyLinker：NIL感知可扩展基于Transformer的实体链接器

    NASTyLinker: NIL-Aware Scalable Transformer-based Entity Linker. (arXiv:2303.04426v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.04426](http://arxiv.org/abs/2303.04426)

    NASTyLinker是一种NIL感知的实体链接器，它通过生成提及簇来表示NIL实体，并在保持已知实体高链接性能的同时解决冲突。

    NASTyLinker is a NIL-aware entity linker that represents NIL entities by producing mention clusters and resolves conflicts while maintaining high linking performance for known entities.

    实体链接（EL）是检测文本中实体提及并将其消歧为参考知识库的任务。大多数流行的EL方法假定参考知识库是完整的。然而，在实践中，需要处理链接到不包含在知识库中的实体（NIL实体）的情况。最近的研究表明，考虑提及之间的亲和力可以用于表示NIL实体，方法是通过生成提及簇。同时，提及之间的亲和力可以帮助显著提高已知实体的链接性能。通过NASTyLinker，我们介绍了一种EL方法，它知道NIL实体并产生相应的提及簇，同时保持已知实体的高链接性能。该方法基于Transformer的密集表示对提及和实体进行聚类，并解决冲突（如果一个实体有多个提及）。

    Entity Linking (EL) is the task of detecting mentions of entities in text and disambiguating them to a reference knowledge base. Most prevalent EL approaches assume that the reference knowledge base is complete. In practice, however, it is necessary to deal with the case of linking to an entity that is not contained in the knowledge base (NIL entity). Recent works have shown that, instead of focusing only on affinities between mentions and entities, considering inter-mention affinities can be used to represent NIL entities by producing clusters of mentions. At the same time, inter-mention affinities can help to substantially improve linking performance for known entities. With NASTyLinker, we introduce an EL approach that is aware of NIL entities and produces corresponding mention clusters while maintaining high linking performance for known entities. The approach clusters mentions and entities based on dense representations from Transformers and resolves conflicts (if more than one en
    
[^13]: 通过变分信息瓶颈和对比学习减少方面情感分析中的虚假相关性

    Reducing Spurious Correlations for Aspect-Based Sentiment Analysis with Variational Information Bottleneck and Contrastive Learning. (arXiv:2303.02846v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.02846](http://arxiv.org/abs/2303.02846)

    本文提出了一种新的对比变分信息瓶颈框架（CVIB），以减少方面情感分析（ABSA）中的虚假相关性。该框架由一个原始网络和一个自剪枝网络组成，通过对比学习同时进行优化，从而丢弃了输入特征和预测标签之间的多余模式或虚假相关性。

    This paper proposes a novel Contrastive Variational Information Bottleneck framework (CVIB) to reduce spurious correlations for aspect-based sentiment analysis (ABSA). The proposed CVIB framework is composed of an original network and a self-pruned network, and these two networks are optimized simultaneously via contrastive learning, which discards the superfluous patterns or spurious correlations between input features and prediction labels.

    深度学习技术在方面情感分析（ABSA）的文献中占据主导地位，取得了最先进的结果。然而，这些深度模型通常在输入特征和输出标签之间存在虚假相关性问题，这会给鲁棒性和泛化能力带来重大障碍。在本文中，我们提出了一种新颖的对比变分信息瓶颈框架（称为CVIB），以减少ABSA中的虚假相关性。所提出的CVIB框架由一个原始网络和一个自剪枝网络组成，这两个网络通过对比学习同时进行优化。具体而言，我们采用变分信息瓶颈（VIB）原则从原始网络中学习一个信息丰富且压缩的网络（自剪枝网络），该网络丢弃了输入特征和预测标签之间的多余模式或虚假相关性。然后，我们设计了自剪枝对比学习，以将两个网络拉在一起。

    Deep learning techniques have dominated the literature on aspect-based sentiment analysis (ABSA), yielding state-of-the-art results. However, these deep models generally suffer from spurious correlation problems between input features and output labels, which creates significant barriers to robustness and generalization capability. In this paper, we propose a novel Contrastive Variational Information Bottleneck framework (called CVIB) to reduce spurious correlations for ABSA. The proposed CVIB framework is composed of an original network and a self-pruned network, and these two networks are optimized simultaneously via contrastive learning. Concretely, we employ the Variational Information Bottleneck (VIB) principle to learn an informative and compressed network (self-pruned network) from the original network, which discards the superfluous patterns or spurious correlations between input features and prediction labels. Then, self-pruning contrastive learning is devised to pull together
    
[^14]: FinXABSA: 通过基于方面的情感分析实现可解释的金融分析

    FinXABSA: Explainable Finance through Aspect-Based Sentiment Analysis. (arXiv:2303.02563v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.02563](http://arxiv.org/abs/2303.02563)

    本文提出了一种基于方面的情感分析方法，通过与股票价格的相关性建立关系，实现金融分析的可解释性。该方法提供了更详细和准确的了解情感分析与股票价格之间关系的方法，对于投资者和金融分析师做出明智决策非常有用。

    This paper proposes an aspect-based sentiment analysis approach to achieve explainability in financial analysis by establishing a relationship with stock prices using the Pearson correlation coefficient. The proposed methodology provides a more detailed and accurate understanding of the relationship between sentiment analysis and stock prices, which can be useful for investors and financial analysts in making informed decisions.

    本文提出了一种新颖的方法，通过利用Pearson相关系数建立基于方面的情感分析与股票价格之间的关系，实现金融分析的可解释性。所提出的方法涉及从金融新闻文章中构建方面列表，并分析每个方面的情感强度得分。然后，使用Pearson系数将这些得分与相关公司的股票价格进行比较，以确定任何显著的相关性。结果表明，所提出的方法提供了更详细和准确的了解情感分析与股票价格之间关系的方法，这对于投资者和金融分析师做出明智决策非常有用。此外，该方法提供了一种透明且可解释的方式来解释情感分析结果及其对股票价格的影响。总的来说，本文的研究结果表明，可解释性在金融分析中的重要性。

    This paper presents a novel approach for explainability in financial analysis by utilizing the Pearson correlation coefficient to establish a relationship between aspect-based sentiment analysis and stock prices. The proposed methodology involves constructing an aspect list from financial news articles and analyzing sentiment intensity scores for each aspect. These scores are then compared to the stock prices for the relevant companies using the Pearson coefficient to determine any significant correlations. The results indicate that the proposed approach provides a more detailed and accurate understanding of the relationship between sentiment analysis and stock prices, which can be useful for investors and financial analysts in making informed decisions. Additionally, this methodology offers a transparent and interpretable way to explain the sentiment analysis results and their impact on stock prices. Overall, the findings of this paper demonstrate the importance of explainability in f
    
[^15]: 重新思考同时机器翻译测试集的合理性

    Rethinking the Reasonability of the Test Set for Simultaneous Machine Translation. (arXiv:2303.00969v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.00969](http://arxiv.org/abs/2303.00969)

    本文提出了一个手动注释的单调测试集SiMuST-C，用于评估同时机器翻译（SimulMT）模型的性能，实验证明该测试集可以缓解低估问题，并且在自动提取的单调训练集上微调可以将SimulMT模型的性能提高3个BLEU分数。

    This paper proposes a manually annotated monotonic test set SiMuST-C for evaluating the performance of simultaneous machine translation (SimulMT) models, which can alleviate the underestimation problem and fine-tuning on an automatically extracted monotonic training set can improve SimulMT models by up to 3 BLEU points.

    同时机器翻译（SimulMT）模型在源语句结束前开始翻译，使得翻译与源语句单调对齐。然而，一般的完整句子翻译测试集是通过离线翻译整个源语句获得的，这不是为SimulMT评估而设计的，这使我们重新思考这是否会低估SimulMT模型的性能。在本文中，我们基于MuST-C英汉测试集手动注释了一个单调测试集，称为SiMuST-C。我们的人工评估确认了我们注释的测试集的可接受性。对三种不同的SimulMT模型的评估验证了我们的测试集可以缓解低估问题。进一步的实验表明，在自动提取的单调训练集上微调可以将SimulMT模型的性能提高3个BLEU分数。

    Simultaneous machine translation (SimulMT) models start translation before the end of the source sentence, making the translation monotonically aligned with the source sentence. However, the general full-sentence translation test set is acquired by offline translation of the entire source sentence, which is not designed for SimulMT evaluation, making us rethink whether this will underestimate the performance of SimulMT models. In this paper, we manually annotate a monotonic test set based on the MuST-C English-Chinese test set, denoted as SiMuST-C. Our human evaluation confirms the acceptability of our annotated test set. Evaluations on three different SimulMT models verify that the underestimation problem can be alleviated on our test set. Further experiments show that finetuning on an automatically extracted monotonic training set improves SimulMT models by up to 3 BLEU points.
    
[^16]: 事件驱动新闻叙事提取综述

    A Survey on Event-based News Narrative Extraction. (arXiv:2302.08351v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08351](http://arxiv.org/abs/2302.08351)

    本文综述了事件驱动新闻叙事提取的研究，通过筛选超过900篇文章，得到了54篇相关文章，这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。

    This survey presents an extensive study of research in the area of event-based news narrative extraction, screening over 900 articles and synthesizing 54 relevant articles organized by representation model, extraction criteria, and evaluation application.

    叙事是我们理解世界的基础，为我们提供了一个自然的时间知识表示结构。计算机叙事提取是人工智能的一个子领域，它大量使用信息检索和自然语言处理技术。尽管计算机叙事提取的重要性，但在综合以前的研究和策划未来的研究方面，学术研究相对较少。特别是，本文侧重于从事件中心的角度提取新闻叙事。从新闻数据中提取叙事在理解不断变化的信息景观方面具有多种应用。本综述对事件驱动新闻叙事提取领域的研究进行了广泛的研究。特别是，我们筛选了超过900篇文章，得到了54篇相关文章。这些文章通过表示模型、提取标准和评估应用程序进行综合和组织。

    Narratives are fundamental to our understanding of the world, providing us with a natural structure for knowledge representation over time. Computational narrative extraction is a subfield of artificial intelligence that makes heavy use of information retrieval and natural language processing techniques. Despite the importance of computational narrative extraction, relatively little scholarly work exists on synthesizing previous research and strategizing future research in the area. In particular, this article focuses on extracting news narratives from an event-centric perspective. Extracting narratives from news data has multiple applications in understanding the evolving information landscape. This survey presents an extensive study of research in the area of event-based news narrative extraction. In particular, we screened over 900 articles that yielded 54 relevant articles. These articles are synthesized and organized by representation model, extraction criteria, and evaluation app
    
[^17]: 人工智能心理学中的“正确答案”

    "Correct answers" from the psychology of artificial intelligence. (arXiv:2302.07267v3 [cs.HC] UPDATED)

    [http://arxiv.org/abs/2302.07267](http://arxiv.org/abs/2302.07267)

    本文使用OpenAI的GPT3.5模型重新复制了Many Labs 2复制项目中的14项研究，其中8项研究的结果被成功复制。然而，对于剩下的6项研究，GPT3.5以极其预定的方式回答了调查问题，导致无法分析这些研究。

    This paper replicates 14 studies from the Many Labs 2 replication project with OpenAI's text-davinci-003 model, and successfully replicates the results of 8 studies. However, for the remaining 6 studies, GPT3.5 answered survey questions in an extremely predetermined way, making it impossible to analyze these studies.

    大型语言模型的能力已经大大增强。这种AI系统的一个提出的应用是支持社会和认知科学中的数据收集，目前完美的实验控制是不可行的，而大规模、代表性数据集的收集通常是昂贵的。在本文中，我们使用OpenAI的text-davinci-003模型（俗称GPT3.5）重新复制了Many Labs 2复制项目中的14项研究。我们通过将每项研究的调查作为文本输入，从GPT3.5的默认设置中收集了响应。在我们可以分析的八项研究中，我们的GPT样本复制了原始结果的37.5%以及Many Labs 2结果的37.5%。出乎意料的是，我们无法像预先注册的计划那样分析剩下的六项研究。这是因为对于这六项研究中的每一项，GPT3.5以极其预定的方式回答了调查问题（无论是因变量还是条件变量）：一个未知的

    Large Language Models have vastly grown in capabilities. One proposed application of such AI systems is to support data collection in the social and cognitive sciences, where perfect experimental control is currently unfeasible and the collection of large, representative datasets is generally expensive. In this paper, we re-replicate 14 studies from the Many Labs 2 replication project with OpenAI's text-davinci-003 model, colloquially known as GPT3.5. We collected responses from the default setting of GPT3.5 by inputting each study's survey as text. Among the eight studies we could analyse, our GPT sample replicated 37.5% of the original results as well as 37.5% of the Many Labs 2 results. Unexpectedly, we could not analyse the remaining six studies as we had planned in our pre-registration. This was because for each of these six studies, GPT3.5 answered at least one of the survey questions (either a dependent variable or a condition variable) in an extremely predetermined way: an unex
    
[^18]: AdapterSoup：使用加权平均改善预训练语言模型的泛化能力

    AdapterSoup: Weight Averaging to Improve Generalization of Pretrained Language Models. (arXiv:2302.07027v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07027](http://arxiv.org/abs/2302.07027)

    本文提出了AdapterSoup，一种使用加权平均改善预训练语言模型泛化能力的方法。该方法在不同领域训练的适配器上执行权重空间平均，可以在不需要额外训练的情况下提高对新领域的性能。

    AdapterSoup is a method that uses weight averaging to improve the generalization ability of pretrained language models. It performs weight-space averaging of adapters trained on different domains, and can improve performance to new domains without extra training.

    预训练语言模型（PLMs）在大规模语料库上进行训练，但通常需要专门针对特定领域进行特化。一种参数有效的适应方法建议在语言建模任务上为每个领域训练一个适配器。这导致了良好的领域内得分，但在领域或资源受限的情况下可能不切实际。解决方案是在测试时使用相关领域适配器来处理新领域。在本文中，我们介绍了AdapterSoup，一种在不同领域训练的适配器上执行权重空间平均的方法。我们的方法是令人尴尬的并行的：首先，我们训练一组特定领域的适配器；然后，对于每个新领域，我们确定在测试时应平均哪些适配器。我们进行了大量实验，表明AdapterSoup始终提高了对新领域的性能，而无需额外的训练。我们还探讨了在不同超参数下训练的相同领域适配器的权重平均，并表明它可以保留

    Pretrained language models (PLMs) are trained on massive corpora, but often need to specialize to specific domains. A parameter-efficient adaptation method suggests training an adapter for each domain on the task of language modeling. This leads to good in-domain scores but can be impractical for domain- or resource-restricted settings. A solution is to use a related-domain adapter for the novel domain at test time. In this paper, we introduce AdapterSoup, an approach that performs weight-space averaging of adapters trained on different domains. Our approach is embarrassingly parallel: first, we train a set of domain-specific adapters; then, for each novel domain, we determine which adapters should be averaged at test time. We present extensive experiments showing that AdapterSoup consistently improves performance to new domains without extra training. We also explore weight averaging of adapters trained on the same domain with different hyper-parameters, and show that it preserves the
    
[^19]: 基于大型语言模型的自适应机器翻译

    Adaptive Machine Translation with Large Language Models. (arXiv:2301.13294v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.13294](http://arxiv.org/abs/2301.13294)

    本文研究了如何利用大型语言模型的上下文学习来改进实时自适应机器翻译，实验结果表明有希望的效果。

    This paper investigates how to use in-context learning of large language models to improve real-time adaptive machine translation, and the experimental results show promising effects.

    一致性是高质量翻译的关键要求。在特定领域的项目中，遵循预先批准的术语并适应更正的翻译尤为重要。机器翻译（MT）在领域适应方面取得了重大进展。然而，实时适应仍然具有挑战性。最近，大规模语言模型（LLM）展示了在上下文学习方面的有趣能力，它们学习复制某些输入-输出文本生成模式，而无需进一步微调。通过在推理时间将LLM提供给由翻译对列表组成的提示，它可以模拟领域和风格特征。本文旨在研究如何利用上下文学习来改进实时自适应MT。我们的广泛实验在翻译时间显示出有希望的结果。例如，GPT-3.5可以在翻译新句子时适应一组领域内的句子对和/或术语。我们的研究表明，基于大型语言模型的自适应机器翻译是可行的。

    Consistency is a key requirement of high-quality translation. It is especially important to adhere to pre-approved terminology and adapt to corrected translations in domain-specific projects. Machine translation (MT) has achieved significant progress in the area of domain adaptation. However, real-time adaptation remains challenging. Large-scale language models (LLMs) have recently shown interesting capabilities of in-context learning, where they learn to replicate certain input-output text generation patterns, without further fine-tuning. By feeding an LLM at inference time with a prompt that consists of a list of translation pairs, it can then simulate the domain and style characteristics. This work aims to investigate how we can utilize in-context learning to improve real-time adaptive MT. Our extensive experiments show promising results at translation time. For example, GPT-3.5 can adapt to a set of in-domain sentence pairs and/or terminology while translating a new sentence. We ob
    
[^20]: 统一视觉、文本和布局的通用文档处理

    Unifying Vision, Text, and Layout for Universal Document Processing. (arXiv:2212.02623v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.02623](http://arxiv.org/abs/2212.02623)

    本文提出了通用文档处理（UDOP）模型，将文本、图像和布局模态以及各种任务格式统一起来，通过一种新颖的Transformer模型实现预训练和多域下游任务的统一，同时实现了高质量的神经文档编辑和内容定制。

    This paper proposes the Universal Document Processing (UDOP) model, which unifies text, image, and layout modalities together with varied task formats, and achieves pretraining and multi-domain downstream tasks unification through a novel Transformer model. It also achieves high-quality neural document editing and content customization.

    我们提出了通用文档处理（UDOP），这是一个基础的文档AI模型，它将文本、图像和布局模态以及各种任务格式（包括文档理解和生成）统一起来。UDOP利用文本内容和文档图像之间的空间相关性，用一个统一的表示来建模图像、文本和布局模态。通过一种新颖的Vision-Text-Layout Transformer，UDOP将预训练和多域下游任务统一到基于提示的序列生成方案中。UDOP在大规模无标签文档语料库和多样化标记数据上使用创新的自监督目标进行预训练。UDOP还通过遮蔽图像重建学习从文本和布局模态生成文档图像。据我们所知，这是文档AI领域中第一次使用一个模型同时实现高质量的神经文档编辑和内容定制。我们的方法在8个文档处理基准数据集上取得了最先进的结果。

    We propose Universal Document Processing (UDOP), a foundation Document AI model which unifies text, image, and layout modalities together with varied task formats, including document understanding and generation. UDOP leverages the spatial correlation between textual content and document image to model image, text, and layout modalities with one uniform representation. With a novel Vision-Text-Layout Transformer, UDOP unifies pretraining and multi-domain downstream tasks into a prompt-based sequence generation scheme. UDOP is pretrained on both large-scale unlabeled document corpora using innovative self-supervised objectives and diverse labeled data. UDOP also learns to generate document images from text and layout modalities via masked image reconstruction. To the best of our knowledge, this is the first time in the field of document AI that one model simultaneously achieves high-quality neural document editing and content customization. Our method sets the state-of-the-art on 8 Docu
    
[^21]: 神经转录器训练：采用逐样本计算减少内存消耗

    Neural Transducer Training: Reduced Memory Consumption with Sample-wise Computation. (arXiv:2211.16270v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.16270](http://arxiv.org/abs/2211.16270)

    本文提出了一种内存高效的神经转录器训练方法，采用逐个样本计算转录器损失和梯度，显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。

    This paper proposes a memory-efficient training method for neural transducer, which computes the transducer loss and gradients sample by sample, significantly reducing memory usage and performing at competitive speed compared to the default batched computation.

    神经转录器是一种用于自动语音识别（ASR）的端到端模型。虽然该模型非常适合流式ASR，但训练过程仍然具有挑战性。在训练过程中，内存需求可能会迅速超过最先进的GPU的容量，限制批量大小和序列长度。在这项工作中，我们分析了典型转录器训练设置的时间和空间复杂度。我们提出了一种内存高效的训练方法，逐个样本计算转录器损失和梯度。我们提出了优化方法，以增加逐样本方法的效率和并行性。在一组彻底的基准测试中，我们展示了我们的逐样本方法显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。作为亮点，我们成功地使用仅6 GB的内存计算了批量大小为1024，音频长度为40秒的转录器损失和梯度。

    The neural transducer is an end-to-end model for automatic speech recognition (ASR). While the model is well-suited for streaming ASR, the training process remains challenging. During training, the memory requirements may quickly exceed the capacity of state-of-the-art GPUs, limiting batch size and sequence lengths. In this work, we analyze the time and space complexity of a typical transducer training setup. We propose a memory-efficient training method that computes the transducer loss and gradients sample by sample. We present optimizations to increase the efficiency and parallelism of the sample-wise method. In a set of thorough benchmarks, we show that our sample-wise method significantly reduces memory usage, and performs at competitive speed when compared to the default batched computation. As a highlight, we manage to compute the transducer loss and gradients for a batch size of 1024, and audio length of 40 seconds, using only 6 GB of memory.
    
[^22]: 自动提取医疗对话中的信息：专家系统和标注注意力

    Automatically Extracting Information in Medical Dialogue: Expert System And Attention for Labelling. (arXiv:2211.15544v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.15544](http://arxiv.org/abs/2211.15544)

    本文提出了一种新颖的模型ESAL，使用专家混合和预训练的BERT来检索不同类别的语义，使模型能够融合它们之间的差异。实验结果表明ESAL显著提高了医疗信息分类的性能。

    This paper proposes a novel model ESAL, which uses mixture of experts and pre-trained BERT to retrieve the semantics of different categories, enabling the model to fuse the differences between them. The experimental results indicate that ESAL significantly improves the performance of Medical Information Classification.

    医疗对话信息提取正在成为现代医疗保健中越来越重要的问题。由于电子病历（EMR）数量庞大，从中提取关键信息是困难的。以前，研究人员提出了基于注意力的模型来从EMR中检索特征，但它们的局限性在于无法识别医疗对话中的不同类别。在本文中，我们提出了一种新颖的模型，即专家系统和标注注意力（ESAL）。我们使用专家混合和预训练的BERT来检索不同类别的语义，使模型能够融合它们之间的差异。在我们的实验中，ESAL被应用于一个公共数据集，实验结果表明ESAL显著提高了医疗信息分类的性能。

    Medical dialogue information extraction is becoming an increasingly significant problem in modern medical care. It is difficult to extract key information from electronic medical records (EMRs) due to their large numbers. Previously, researchers proposed attention-based models for retrieving features from EMRs, but their limitations were reflected in their inability to recognize different categories in medical dialogues. In this paper, we propose a novel model, Expert System and Attention for Labelling (ESAL). We use mixture of experts and pre-trained BERT to retrieve the semantics of different categories, enabling the model to fuse the differences between them. In our experiment, ESAL was applied to a public dataset and the experimental results indicated that ESAL significantly improved the performance of Medical Information Classification.
    
[^23]: 意外学习者：自监督多语言模型中的口语语言识别

    Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models. (arXiv:2211.05103v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2211.05103](http://arxiv.org/abs/2211.05103)

    本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。预训练的语音模型在较低层中最优地编码了语言区分信息，从这些层获得的嵌入能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果，且使用的参数量仅为其它模型的五分之一。

    This paper extends previous self-supervised approaches for language identification by experimenting with Conformer based architecture in a multilingual pre-training paradigm. The pre-trained speech models optimally encode language discriminatory information in lower layers, and the embeddings obtained from these layers are significantly robust to classify unseen languages and different acoustic environments without additional training. After fine-tuning a pre-trained Conformer model on the VoxLingua107 dataset, the authors achieve results similar to current state-of-the-art systems for language identification, with 5x less parameters. The model is open-sourced through the NVIDIA NeMo toolkit.

    本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。我们发现，预训练的语音模型在较低层中最优地编码了语言区分信息。此外，我们证明了从这些层获得的嵌入在没有额外训练的情况下，能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果。此外，我们的模型使用的参数量仅为其它模型的五分之一。我们通过NVIDIA NeMo工具包开源了该模型。

    In this paper, we extend previous self-supervised approaches for language identification by experimenting with Conformer based architecture in a multilingual pre-training paradigm. We find that pre-trained speech models optimally encode language discriminatory information in lower layers. Further, we demonstrate that the embeddings obtained from these layers are significantly robust to classify unseen languages and different acoustic environments without additional training. After fine-tuning a pre-trained Conformer model on the VoxLingua107 dataset, we achieve results similar to current state-of-the-art systems for language identification. More, our model accomplishes this with 5x less parameters. We open-source the model through the NVIDIA NeMo toolkit.
    
[^24]: BLOOM: 一个拥有176B参数的开放式多语言语言模型

    BLOOM: A 176B-Parameter Open-Access Multilingual Language Model. (arXiv:2211.05100v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.05100](http://arxiv.org/abs/2211.05100)

    BLOOM是一个由数百名研究人员合作设计和构建的拥有176B参数的开放式语言模型，它在多种基准测试中表现出竞争性的性能，并在进行多任务提示微调后表现更强。该模型的发布有助于推动大型语言模型技术的民主化。

    BLOOM is an open-access language model with 176B parameters, designed and built by hundreds of researchers. It achieves competitive performance on various benchmarks and shows even stronger results after multitask prompted finetuning. The release of this model facilitates the democratization of large language model technology.

    大型语言模型（LLMs）已被证明能够根据少量演示或自然语言指令执行新任务。尽管这些能力已被广泛采用，但大多数LLMs都是由资源丰富的组织开发的，并经常被保密。为了推动这种强大技术的民主化，我们提出了BLOOM，这是一个由数百名研究人员合作设计和构建的拥有176B参数的开放式语言模型。BLOOM是一个仅解码器的Transformer语言模型，它是在ROOTS语料库上进行训练的，该语料库包括46种自然语言和13种编程语言的数百个来源（总共59种）。我们发现，BLOOM在各种基准测试中取得了竞争性的表现，在进行多任务提示微调后表现更强。为了促进未来使用LLMs的研究和应用，我们在负责任的AI许可下公开发布我们的模型和代码。

    Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License.
    
[^25]: 使用情感嵌入在情感、语言和注释格式之间传递知识

    Using Emotion Embeddings to Transfer Knowledge Between Emotions, Languages, and Annotation Formats. (arXiv:2211.00171v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.00171](http://arxiv.org/abs/2211.00171)

    本文研究了如何通过利用多语言模型和Demux来构建一个可以在不同情感、语言和注释格式之间转换的单一模型，以实现知识共享和降低训练成本。

    This paper studies how to build a single model that can transition between different emotions, languages, and annotation formats by leveraging multilingual models and Demux, to achieve knowledge sharing and reduce training costs.

    随着越来越多的学科将情感融入其理论和应用中，从文本中推断情感的需求不断多样化。这些需求包括推断不同类型的情感、处理多种语言和不同的注释格式。不同配置之间的共享模型将使知识共享和训练成本降低，并简化在新环境中部署情感识别模型的过程。在这项工作中，我们研究了如何通过利用多语言模型和Demux来构建一个可以在这些不同配置之间转换的单一模型，Demux是一个基于transformer的模型，其输入包括感兴趣的情感，使我们能够动态地改变模型预测的情感。Demux还产生情感嵌入，对它们执行操作可以通过汇集每个簇的嵌入来过渡到情感簇。我们展示了Demux可以同时传输k

    The need for emotional inference from text continues to diversify as more and more disciplines integrate emotions into their theories and applications. These needs include inferring different emotion types, handling multiple languages, and different annotation formats. A shared model between different configurations would enable the sharing of knowledge and a decrease in training costs, and would simplify the process of deploying emotion recognition models in novel environments. In this work, we study how we can build a single model that can transition between these different configurations by leveraging multilingual models and Demux, a transformer-based model whose input includes the emotions of interest, enabling us to dynamically change the emotions predicted by the model. Demux also produces emotion embeddings, and performing operations on them allows us to transition to clusters of emotions by pooling the embeddings of each cluster. We show that Demux can simultaneously transfer k
    
[^26]: 在多标签情感识别中利用标签相关性的研究：以情感为例

    Leveraging Label Correlations in a Multi-label Setting: A Case Study in Emotion. (arXiv:2210.15842v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.15842](http://arxiv.org/abs/2210.15842)

    本文研究了利用标签相关性来改善情感检测的方法，开发了两种建模方法来捕捉情感词本身的词汇关联性，并将情感表示的成对约束作为正则化项与模型的分类损失一起集成，展示了在SemEval 2018任务1 E-c中使用单语BERT模型展示了西班牙语、英语和阿拉伯语的最新性能。

    This paper investigates ways to exploit label correlations in multi-label emotion recognition models to improve emotion detection, develops two modeling approaches to capture word associations of the emotion words themselves, and integrates pairwise constraints of emotion representations as regularization terms alongside the classification loss of the models, demonstrating state-of-the-art performance across Spanish, English, and Arabic in SemEval 2018 Task 1 E-c using monolingual BERT-based models.

    在文本中检测表达的情感已经成为许多领域的关键。本文研究了利用多标签情感识别模型中的标签相关性来改善情感检测的方法。首先，我们开发了两种建模方法来捕捉情感词本身的词汇关联性，一种是将情感包含在输入中，另一种是利用遮蔽语言建模（MLM）。其次，我们将情感表示的成对约束作为正则化项与模型的分类损失一起集成。我们将这些项分为两类，局部和全局。前者根据金标签动态变化，而后者在训练期间保持不变。我们在SemEval 2018任务1 E-c中使用单语BERT模型展示了西班牙语、英语和阿拉伯语的最新性能。除了更好的性能外，我们还展示了改进的鲁棒性。

    Detecting emotions expressed in text has become critical to a range of fields. In this work, we investigate ways to exploit label correlations in multi-label emotion recognition models to improve emotion detection. First, we develop two modeling approaches to the problem in order to capture word associations of the emotion words themselves, by either including the emotions in the input, or by leveraging Masked Language Modeling (MLM). Second, we integrate pairwise constraints of emotion representations as regularization terms alongside the classification loss of the models. We split these terms into two categories, local and global. The former dynamically change based on the gold labels, while the latter remain static during training. We demonstrate state-of-the-art performance across Spanish, English, and Arabic in SemEval 2018 Task 1 E-c using monolingual BERT-based models. On top of better performance, we also demonstrate improved robustness. Code is available at https://github.com/
    
[^27]: Articulation GAN: 无监督建模关节学习

    Articulation GAN: Unsupervised modeling of articulatory learning. (arXiv:2210.15173v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.15173](http://arxiv.org/abs/2210.15173)

    本文提出了一种新的无监督生成模型，通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式，从而更好地模拟人类语音产生的过程。

    This paper proposes a new unsupervised generative model that learns to generate articulatory representations (electromagnetic articulography or EMA) in a fully unsupervised manner, which more closely mimics human speech production and better simulates the process of human speech production.

    生成式深度神经网络广泛用于语音合成，但大多数现有模型直接生成波形或频谱输出。然而，人类通过控制关节来产生语音，这通过声音传播的物理特性导致语音声音的产生。我们引入了关节生成器到生成对抗网络范例中，这是一种新的无监督生成模型，用于语音产生/合成。关节生成器通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式。然后，一个单独的预训练物理模型（ema2wav）将生成的EMA表示转换为语音波形，这些波形被发送到鉴别器进行评估。关节分析表明，网络学习控制关节的方式类似于人类在语音产生过程中的方式。输出的声学分析表明...

    Generative deep neural networks are widely used for speech synthesis, but most existing models directly generate waveforms or spectral outputs. Humans, however, produce speech by controlling articulators, which results in the production of speech sounds through physical properties of sound propagation. We introduce the Articulatory Generator to the Generative Adversarial Network paradigm, a new unsupervised generative model of speech production/synthesis. The Articulatory Generator more closely mimics human speech production by learning to generate articulatory representations (electromagnetic articulography or EMA) in a fully unsupervised manner. A separate pre-trained physical model (ema2wav) then transforms the generated EMA representations to speech waveforms, which get sent to the Discriminator for evaluation. Articulatory analysis suggests that the network learns to control articulators in a similar manner to humans during speech production. Acoustic analysis of the outputs sugge
    
[^28]: 直接语音翻译中的命名实体检测和注入

    Named Entity Detection and Injection for Direct Speech Translation. (arXiv:2210.11981v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.11981](http://arxiv.org/abs/2210.11981)

    本文研究了如何利用命名实体字典来改进语音到文本翻译模型的输出，实验表明可以通过减少31%的人名错误来提高翻译中的命名实体准确性。

    This paper explores how to leverage named entity dictionaries to improve speech-to-text translation model outputs, and experiments show that it can improve named entity accuracy in translation with a 31% reduction in person name errors.

    在一句话中，某些词对其语义至关重要。其中，命名实体（NE）对神经模型来说是极具挑战性的。尽管它们很重要，但在语音到文本（S2T）翻译研究中，它们的准确处理一直被忽视，最近的研究表明，S2T模型在位置和特别是人名方面表现不佳，除非事先知道其拼写。在这项工作中，我们探讨了如何利用已知可能出现在给定上下文中的NE字典来改进S2T模型输出。我们的实验表明，我们可以可靠地检测出可能出现在话语中的NE，从S2T编码器输出开始。事实上，我们证明了当前的检测质量足以通过减少31％的人名错误来提高翻译中的NE准确性。

    In a sentence, certain words are critical for its semantic. Among them, named entities (NEs) are notoriously challenging for neural models. Despite their importance, their accurate handling has been neglected in speech-to-text (S2T) translation research, and recent work has shown that S2T models perform poorly for locations and notably person names, whose spelling is challenging unless known in advance. In this work, we explore how to leverage dictionaries of NEs known to likely appear in a given context to improve S2T model outputs. Our experiments show that we can reliably detect NEs likely present in an utterance starting from S2T encoder outputs. Indeed, we demonstrate that the current detection quality is sufficient to improve NE accuracy in the translation with a 31% reduction in person name errors.
    
[^29]: 投影增强：一种有效且高效的蒸馏数据增强范式

    Augmentation with Projection: Towards an Effective and Efficient Data Augmentation Paradigm for Distillation. (arXiv:2210.11768v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.11768](http://arxiv.org/abs/2210.11768)

    提出了一种名为AugPro的数据增强方法，它是一种有效且高效的蒸馏数据增强方法，可以避免偏移决策边界，计算开销小。

    AugPro is an effective and efficient data augmentation method for distillation, which avoids shifting decision boundaries and has little computational overhead.

    知识蒸馏是从大型模型向小型模型转移知识的主要方法之一。然而，它需要大量的任务特定数据，在许多实际应用中可能不可行。为了解决这个问题，采用了表示插值、标记替换或模型增强等数据增强方法。然而，这些数据增强方法可能会导致决策边界的偏移（表示插值），不够表达（标记替换）或引入过多的计算开销（模型增强）。为此，我们提出了AugPro（投影增强），一种有效且高效的蒸馏数据增强方法。我们的方法建立在表示插值增强方法之上，以保持表达的多样性，并将增强的数据转换为标记以避免偏移决策边界。它使用简单的操作，计算开销小。

    Knowledge distillation is one of the primary methods of transferring knowledge from large to small models. However, it requires massive task-specific data, which may not be plausible in many real-world applications. Data augmentation methods such as representation interpolation, token replacement, or augmentation with models are applied to tackle this problem. However, these data augmentation methods either potentially cause shifts in decision boundaries (representation interpolation), are not expressive enough (token replacement), or introduce too much computational overhead (augmentation with models). To this end, we propose AugPro (Augmentation with Projection), an effective and efficient data augmentation method for distillation. Our method builds on top of representation interpolation augmentation methods to maintain the diversity of expressions and converts the augmented data to tokens to avoid shifting decision boundaries. It uses simple operations that come with little computat
    
[^30]: 神经文本作者归属和混淆：数据挖掘视角

    Attribution and Obfuscation of Neural Text Authorship: A Data Mining Perspective. (arXiv:2210.10488v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.10488](http://arxiv.org/abs/2210.10488)

    本文调查了神经文本生成技术在作者归属和混淆中的应用，提出了需要开发新型AA / AO解决方案来处理神经文本的问题。

    This survey investigates the application of neural text generation techniques in authorship attribution and obfuscation, and highlights the need for developing novel AA/AO solutions to deal with neural texts.

    作者归属（AA）和作者混淆（AO）是隐私研究中越来越受关注和重要的两个交织的研究问题。传统上，作者的概念及其随之而来的隐私关注仅针对人类作者。然而，由于自然语言处理中神经文本生成（NTG）技术的爆炸性进展，现在必须考虑人类、机器或它们的组合的作者身份。由于神经文本在恶意使用时的潜在威胁，了解传统AA / AO解决方案的局限性并开发处理神经文本的新型AA / AO解决方案变得至关重要。

    Two interlocking research questions of growing interest and importance in privacy research are Authorship Attribution (AA) and Authorship Obfuscation (AO). Given an artifact, especially a text t in question, an AA solution aims to accurately attribute t to its true author out of many candidate authors while an AO solution aims to modify t to hide its true authorship. Traditionally, the notion of authorship and its accompanying privacy concern is only toward human authors. However, in recent years, due to the explosive advancements in Neural Text Generation (NTG) techniques in NLP, capable of synthesizing human-quality open-ended texts (so-called "neural texts"), one has to now consider authorships by humans, machines, or their combination. Due to the implications and potential threats of neural texts when used maliciously, it has become critical to understand the limitations of traditional AA/AO solutions and develop novel AA/AO solutions in dealing with neural texts. In this survey, t
    
[^31]: EDU级别的可变长度摘要提取

    EDU-level Extractive Summarization with Varying Summary Lengths. (arXiv:2210.04029v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.04029](http://arxiv.org/abs/2210.04029)

    本文提出了一种EDU级别的可变长度摘要提取模型，通过比较分析证明EDU比句子具有更高的自动评估分数。

    This paper proposes an EDU-level extractive model with varying summary lengths and provides evidence from both theoretical and experimental perspectives to justify and quantify that EDUs make summaries with higher automatic evaluation scores than sentences.

    抽取式模型通常将文本摘要制定为从文档中提取固定的前k个显著句子作为摘要。很少有工作探索细粒度的基本话语单元（EDU）的提取，而且对于抽取单元的选择缺乏分析和证明。此外，固定的前k个显著句子的选择策略不适合摘要需求，因为不同文档中显著句子的数量不同，因此在现实中不存在共同或最佳的k。为了填补这些空白，本文首先对基于EDU和句子的oracle摘要进行比较分析，从理论和实验的角度提供证据，证明和量化EDU比句子具有更高的自动评估分数。然后，考虑到EDU的这种优点，本文进一步提出了一种EDU级别的可变长度摘要提取模型，并开发了相应的学习算法。EDU-VL

    Extractive models usually formulate text summarization as extracting fixed top-$k$ salient sentences from the document as a summary. Few works exploited extracting finer-grained Elementary Discourse Unit (EDU) with little analysis and justification for the extractive unit selection. Further, the selection strategy of the fixed top-$k$ salient sentences fits the summarization need poorly, as the number of salient sentences in different documents varies and therefore a common or best $k$ does not exist in reality. To fill these gaps, this paper first conducts the comparison analysis of oracle summaries based on EDUs and sentences, which provides evidence from both theoretical and experimental perspectives to justify and quantify that EDUs make summaries with higher automatic evaluation scores than sentences. Then, considering this merit of EDUs, this paper further proposes an EDU-level extractive model with Varying summary Lengths and develops the corresponding learning algorithm. EDU-VL
    
[^32]: 非确定性堆栈循环神经网络的惊人计算能力

    The Surprising Computational Power of Nondeterministic Stack RNNs. (arXiv:2210.01343v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.01343](http://arxiv.org/abs/2210.01343)

    本文展示了非确定性堆栈循环神经网络的惊人计算能力，它不仅可以识别上下文无关语言，还可以识别许多非上下文无关语言，并且可以识别具有比其堆栈字母表大小更大的语言。

    This paper demonstrates the surprising computational power of nondeterministic stack recurrent neural networks, which can not only recognize context-free languages, but also many non-context-free languages and languages with larger alphabet sizes than their stack alphabet.

    传统的循环神经网络（RNN）具有固定的、有限的记忆单元。在理论上（假设有界的范围和精度），这限制了它们的形式语言识别能力为正则语言，并且在实践中，已经证明RNN无法学习许多上下文无关语言（CFL）。为了扩展RNN识别的语言类别，先前的工作使用非确定性堆栈数据结构增强了RNN，使它们与下推自动机相当，并将它们的语言识别能力增加到CFL。非确定性是识别所有CFL所必需的（不仅仅是确定性CFL），但在本文中，我们展示了非确定性和神经控制器相互作用产生了另外两种意想不到的能力。首先，非确定性堆栈RNN不仅可以识别CFL，还可以识别许多非上下文无关语言。其次，它可以识别具有比其堆栈字母表大小更大的语言，这一点可能超出了人们的预期。最后，为了增加其计算能力，我们提出了一种新的训练方法，该方法可以在不增加计算复杂度的情况下提高其识别能力。

    Traditional recurrent neural networks (RNNs) have a fixed, finite number of memory cells. In theory (assuming bounded range and precision), this limits their formal language recognition power to regular languages, and in practice, RNNs have been shown to be unable to learn many context-free languages (CFLs). In order to expand the class of languages RNNs recognize, prior work has augmented RNNs with a nondeterministic stack data structure, putting them on par with pushdown automata and increasing their language recognition power to CFLs. Nondeterminism is needed for recognizing all CFLs (not just deterministic CFLs), but in this paper, we show that nondeterminism and the neural controller interact to produce two more unexpected abilities. First, the nondeterministic stack RNN can recognize not only CFLs, but also many non-context-free languages. Second, it can recognize languages with much larger alphabet sizes than one might expect given the size of its stack alphabet. Finally, to inc
    
[^33]: 评估上下文信息在仇恨言论检测中的影响

    Assessing the impact of contextual information in hate speech detection. (arXiv:2210.00465v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.00465](http://arxiv.org/abs/2210.00465)

    本文提供了一个基于Twitter上媒体机构新闻帖子的用户响应的上下文化仇恨言论检测新语料库，以解决当前自动检测仇恨言论方法缺乏上下文的限制。

    This paper provides a novel corpus for contextualized hate speech detection based on user responses to news posts from media outlets on Twitter, to address the limitation of current automatic hate speech detection methods lacking context.

    近年来，由于其强度和与受保护群体成员的暴力行为的关系，仇恨言论在社交网络和其他虚拟媒体中获得了极大的关注。由于用户生成的内容数量巨大，因此在研究和开发自动工具以帮助分析和管理这种言论方面已经做出了巨大的努力，至少在其最具威胁性的形式中。当前自动检测仇恨言论方法的一个限制是缺乏上下文。大多数研究和资源是在没有上下文的数据上进行的；也就是说，孤立的消息没有任何类型的对话上下文或正在讨论的主题。这限制了可用信息来定义社交网络上的帖子是否具有仇恨性。在这项工作中，我们提供了一个基于Twitter上媒体机构新闻帖子的用户响应的上下文化仇恨言论检测新语料库。

    In recent years, hate speech has gained great relevance in social networks and other virtual media because of its intensity and its relationship with violent acts against members of protected groups. Due to the great amount of content generated by users, great effort has been made in the research and development of automatic tools to aid the analysis and moderation of this speech, at least in its most threatening forms. One of the limitations of current approaches to automatic hate speech detection is the lack of context. Most studies and resources are performed on data without context; that is, isolated messages without any type of conversational context or the topic being discussed. This restricts the available information to define if a post on a social network is hateful or not. In this work, we provide a novel corpus for contextualized hate speech detection based on user responses to news posts from media outlets on Twitter. This corpus was collected in the Rioplatense dialectal v
    
[^34]: 学习ASR路径：一种稀疏的多语言ASR模型

    Learning ASR pathways: A sparse multilingual ASR model. (arXiv:2209.05735v3 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2209.05735](http://arxiv.org/abs/2209.05735)

    本文提出了一种稀疏的多语言ASR模型，通过激活语言特定的子网络来显式地学习每种语言的参数，同时通过联合多语言训练实现对低资源语言的知识转移，相比于密集模型和语言不可知的剪枝模型，在低资源语言上提供更好的性能。

    This paper proposes a sparse multilingual ASR model, which explicitly learns the parameters for each language by activating language-specific sub-networks, and enables knowledge transfer for lower-resource languages via joint multilingual training. The proposed ASR pathways outperform both dense models and a language-agnostically pruned model, and provide better performance on low-resource languages compared to the monolingual sparse models.

    神经网络剪枝有效地压缩了自动语音识别（ASR）模型。然而，在多语言ASR中，语言不可知的剪枝可能会导致某些语言的性能严重下降，因为语言不可知的剪枝掩码可能不适合所有语言并且丢弃重要的语言特定参数。在这项工作中，我们提出了ASR路径，一种稀疏的多语言ASR模型，它激活语言特定的子网络（“路径”），以便为每种语言显式地学习参数。通过重叠的子网络，共享参数还可以通过联合多语言训练实现对低资源语言的知识转移。我们提出了一种新算法来学习ASR路径，并使用流式RNN-T模型在4种语言上评估了所提出的方法。我们提出的ASR路径模型优于密集模型和语言不可知的剪枝模型，并且与单语稀疏模型相比，在低资源语言上提供更好的性能。

    Neural network pruning compresses automatic speech recognition (ASR) models effectively. However, in multilingual ASR, language-agnostic pruning may lead to severe performance drops on some languages because language-agnostic pruning masks may not fit all languages and discard important language-specific parameters. In this work, we present ASR pathways, a sparse multilingual ASR model that activates language-specific sub-networks ("pathways"), such that the parameters for each language are learned explicitly. With the overlapping sub-networks, the shared parameters can also enable knowledge transfer for lower-resource languages via joint multilingual training. We propose a novel algorithm to learn ASR pathways, and evaluate the proposed method on 4 languages with a streaming RNN-T model. Our proposed ASR pathways outperform both dense models and a language-agnostically pruned model, and provide better performance on low-resource languages compared to the monolingual sparse models.
    
[^35]: DailyTalk：面向对话文本转语音的口语对话数据集

    DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech. (arXiv:2207.01063v3 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2207.01063](http://arxiv.org/abs/2207.01063)

    本文介绍了一个高质量的对话语音数据集DailyTalk，专门为对话TTS设计。DailyTalk可以用作通用的TTS数据集，而且基线可以表示DailyTalk的上下文信息。

    This paper introduces a high-quality conversational speech dataset DailyTalk designed for conversational TTS. DailyTalk can be used as a general TTS dataset, and the baseline can represent contextual information from DailyTalk.

    目前大多数的文本转语音（TTS）数据集都是由单个话语组成，缺乏对话方面的内容。本文介绍了DailyTalk，这是一个高质量的对话语音数据集，专门为对话TTS设计。我们从开放领域对话数据集DailyDialog中抽样、修改和录制了2,541个对话，并继承了其注释属性。在我们的数据集上，我们扩展了之前的工作作为我们的基线，其中一个非自回归TTS在对话中的历史信息的条件下进行。通过基线实验和我们的新颖度量标准，我们展示了DailyTalk可以用作通用的TTS数据集，而且我们的基线可以表示DailyTalk的上下文信息。DailyTalk数据集和基线代码可供学术用途免费使用，采用CC-BY-SA 4.0许可证。

    The majority of current Text-to-Speech (TTS) datasets, which are collections of individual utterances, contain few conversational aspects. In this paper, we introduce DailyTalk, a high-quality conversational speech dataset designed for conversational TTS. We sampled, modified, and recorded 2,541 dialogues from the open-domain dialogue dataset DailyDialog inheriting its annotated attributes. On top of our dataset, we extend prior work as our baseline, where a non-autoregressive TTS is conditioned on historical information in a dialogue. From the baseline experiment with both general and our novel metrics, we show that DailyTalk can be used as a general TTS dataset, and more than that, our baseline can represent contextual information from DailyTalk. The DailyTalk dataset and baseline code are freely available for academic use with CC-BY-SA 4.0 license.
    
[^36]: EasyNLP：一款全面且易于使用的自然语言处理工具包

    EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing. (arXiv:2205.00258v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.00258](http://arxiv.org/abs/2205.00258)

    EasyNLP是一款全面且易于使用的自然语言处理工具包，支持全面的NLP算法套件，具有知识增强的预训练、知识蒸馏和少样本学习功能，用于大规模PTMs，并为实际应用提供了模型训练、推理和部署的统一框架。

    EasyNLP is a comprehensive and easy-to-use natural language processing toolkit that supports a comprehensive suite of NLP algorithms. It features knowledge-enhanced pre-training, knowledge distillation and few-shot learning functionalities for large-scale PTMs, and provides a unified framework of model training, inference and deployment for real-world applications.

    预训练模型（PTMs）的成功重塑了自然语言处理（NLP）的发展。然而，对于工业从业者来说，获得高性能模型并将其部署到在线环境中并不容易。为了弥合这一差距，EasyNLP旨在使构建NLP应用程序变得容易，支持全面的NLP算法套件。它还具有知识增强的预训练、知识蒸馏和少样本学习功能，用于大规模PTMs，并为实际应用提供了模型训练、推理和部署的统一框架。目前，EasyNLP已经为阿里巴巴集团的十多个业务部门提供支持，并无缝集成到阿里云的AI平台（PAI）产品中。我们的EasyNLP工具包的源代码已在GitHub（https://github.com/alibaba/EasyNLP）上发布。

    The success of Pre-Trained Models (PTMs) has reshaped the development of Natural Language Processing (NLP). Yet, it is not easy to obtain high-performing models and deploy them online for industrial practitioners. To bridge this gap, EasyNLP is designed to make it easy to build NLP applications, which supports a comprehensive suite of NLP algorithms. It further features knowledge-enhanced pre-training, knowledge distillation and few-shot learning functionalities for large-scale PTMs, and provides a unified framework of model training, inference and deployment for real-world applications. Currently, EasyNLP has powered over ten business units within Alibaba Group and is seamlessly integrated to the Platform of AI (PAI) products on Alibaba Cloud. The source code of our EasyNLP toolkit is released at GitHub (https://github.com/alibaba/EasyNLP).
    
[^37]: 一种基于对比学习的手语翻译框架

    A Token-level Contrastive Framework for Sign Language Translation. (arXiv:2204.04916v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.04916](http://arxiv.org/abs/2204.04916)

    提出了一种基于对比学习的手语翻译框架ConSLT，通过将标记级对比学习纳入SLT解码过程中，学习有效的标记表示，以缓解公开可用的手语翻译语料库非常有限的问题。

    Proposed a token-level contrastive framework for sign language translation, ConSLT, which incorporates token-level contrastive learning into the SLT decoding process to learn effective token representations and alleviate the issue of limited publicly available SLT corpus.

    手语翻译是一项有前途的技术，可以弥合聋人和听力人之间的沟通隔阂。最近，研究人员采用了神经机器翻译（NMT）方法来实现手语翻译，但公开可用的手语翻译语料库非常有限，这导致了标记表示的崩溃和生成标记的不准确性。为了缓解这个问题，我们提出了ConSLT，一种新颖的基于对比学习的手语翻译框架，通过将标记级对比学习纳入SLT解码过程中，学习有效的标记表示。具体而言，ConSLT在解码过程中将每个标记及其由不同丢失掩码生成的对应标记视为正对，然后随机从当前句子中不在词汇表中的$K$个标记中抽样构建负例。我们进行了全面的实验来验证ConSLT的有效性。

    Sign Language Translation (SLT) is a promising technology to bridge the communication gap between the deaf and the hearing people. Recently, researchers have adopted Neural Machine Translation (NMT) methods, which usually require large-scale corpus for training, to achieve SLT. However, the publicly available SLT corpus is very limited, which causes the collapse of the token representations and the inaccuracy of the generated tokens. To alleviate this issue, we propose ConSLT, a novel token-level \textbf{Con}trastive learning framework for \textbf{S}ign \textbf{L}anguage \textbf{T}ranslation , which learns effective token representations by incorporating token-level contrastive learning into the SLT decoding process. Concretely, ConSLT treats each token and its counterpart generated by different dropout masks as positive pairs during decoding, and then randomly samples $K$ tokens in the vocabulary that are not in the current sentence to construct negative examples. We conduct comprehen
    
[^38]: 日语ASR中基于音节和字符目标的交替中间条件

    Alternate Intermediate Conditioning with Syllable-level and Character-level Targets for Japanese ASR. (arXiv:2204.00175v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.00175](http://arxiv.org/abs/2204.00175)

    该论文提出了一种基于音节和字符目标的交替中间条件方法，利用字符级和音节级中间预测作为条件特征来处理日语ASR中的多对一和一对多的映射问题，并在实验中取得了优异的表现。

    This paper proposes an alternate intermediate conditioning method with syllable-level and character-level targets to deal with the many-to-one and one-to-many mapping problems in Japanese ASR, and achieves better performance than conventional multi-task and Self-conditioned CTC methods in experiments.

    端到端的自动语音识别直接将输入语音映射到字符。然而，当多个不同的发音应该映射到一个字符或一个发音被多个不同的字符共享时，映射可能会出现问题。由于日语汉字的存在，日语ASR最容易遭受这种多对一和一对多的映射问题。为了缓解这些问题，我们引入了字符和音节之间的显式交互，使用自我条件连接主义时间分类（CTC），其中上层“自我条件”于下层的中间预测。所提出的方法利用字符级和音节级中间预测作为条件特征来处理字符和音节之间的相互依赖关系。在自发日语语料库上的实验结果表明，所提出的方法优于传统的多任务和自我条件CTC方法。

    End-to-end automatic speech recognition directly maps input speech to characters. However, the mapping can be problematic when several different pronunciations should be mapped into one character or when one pronunciation is shared among many different characters. Japanese ASR suffers the most from such many-to-one and one-to-many mapping problems due to Japanese kanji characters. To alleviate the problems, we introduce explicit interaction between characters and syllables using Self-conditioned connectionist temporal classification (CTC), in which the upper layers are ``self-conditioned'' on the intermediate predictions from the lower layers. The proposed method utilizes character-level and syllable-level intermediate predictions as conditioning features to deal with mutual dependency between characters and syllables. Experimental results on Corpus of Spontaneous Japanese show that the proposed method outperformed the conventional multi-task and Self-conditioned CTC methods.
    
[^39]: I-Tuning: 利用图像对冻结语言模型进行轻量级图像字幕生成的调整

    I-Tuning: Tuning Frozen Language Models with Image for Lightweight Image Captioning. (arXiv:2202.06574v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2202.06574](http://arxiv.org/abs/2202.06574)

    本文提出了一种轻量级的图像字幕生成框架（I-Tuning），通过设计新颖的交叉注意力模块将不可训练的预训练语言解码器和视觉编码器连接起来，使得模型包含的可训练参数少，训练速度快，同时在三个图像字幕生成基准测试上实现了与大规模基线系统相当或更好的性能，但需要的可训练参数和训练数据量都少得多。

    This paper proposes a lightweight image captioning framework (I-Tuning) that connects the non-trainable pre-trained language decoder GPT2 and vision encoder CLIP-ViT with a novel I-Tuning cross-attention module. The framework contains fewer trainable parameters and achieves comparable or better performance than large-scale baseline systems on three image captioning benchmarks, while requiring much fewer trainable parameters and training data compared with state-of-the-art baselines.

    图像字幕生成是一项传统的视觉与语言任务，旨在生成图像的语言描述。最近的研究集中在扩大模型规模和训练数据量，这显著增加了模型训练的成本。与这些高成本模型不同，我们引入了一个轻量级的图像字幕生成框架（I-Tuning），其中包含少量可训练参数。我们设计了一种新颖的I-Tuning交叉注意力模块，将不可训练的预训练语言解码器GPT2和视觉编码器CLIP-ViT连接起来。由于大多数参数在训练期间不需要更新，因此我们的框架轻巧快速。在三个图像字幕生成基准测试上进行的实验结果表明，我们的框架实现了与大规模基线系统相当或更好的性能。但与最先进的基线相比，我们的模型包含多达10倍少的可训练参数，并且需要更少的数据进行训练。

    Image Captioning is a traditional vision-and-language task that aims to generate the language description of an image. Recent studies focus on scaling up the model size and the number of training data, which significantly increase the cost of model training. Different to these heavy-cost models, we introduce a lightweight image captioning framework (I-Tuning), which contains a small number of trainable parameters. We design a novel I-Tuning cross-attention module to connect the non-trainable pre-trained language decoder GPT2 and vision encoder CLIP-ViT. Since most parameters are not required to be updated during training, our framework is lightweight and fast. Experimental results conducted on three image captioning benchmarks reveal that our framework achieves comparable or better performance than the large-scale baseline systems. But our models contain up to 10 times fewer trainable parameters and require much fewer data for training compared with state-of-the-art baselines.
    
[^40]: 视频中的时间句子定位：综述与未来方向

    Temporal Sentence Grounding in Videos: A Survey and Future Directions. (arXiv:2201.08071v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2201.08071](http://arxiv.org/abs/2201.08071)

    本文综述了视频中的时间句子定位（TSGV）的基本概念和当前研究现状，以及未来研究方向。TSGV旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻，连接计算机视觉和自然语言，是两个社区研究人员的重点关注点。

    This survey summarizes the fundamental concepts and current research status of temporal sentence grounding in videos (TSGV), also known as natural language video localization (NLVL) or video moment retrieval (VMR), as well as future research directions. TSGV aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video, connecting computer vision and natural language, and has drawn significant attention from researchers in both communities.

    视频中的时间句子定位（TSGV），又称自然语言视频定位（NLVL）或视频时刻检索（VMR），旨在从未经修剪的视频中检索与语言查询语义对应的时间时刻。连接计算机视觉和自然语言，TSGV引起了两个社区研究人员的重视。本综述试图提供TSGV中基本概念和当前研究现状的总结，以及未来研究方向。作为背景，我们以教程的形式介绍了TSGV中功能组件的常见结构：从原始视频和语言查询的特征提取到目标时刻的答案预测。然后，我们回顾了多模态理解和交互的技术，这是TSGV的重点关注点，以实现两种模态之间的有效对齐。我们构建了TSGV技术的分类法，并详细阐述了不同类别的方法及其优缺点。

    Temporal sentence grounding in videos (TSGV), \aka natural language video localization (NLVL) or video moment retrieval (VMR), aims to retrieve a temporal moment that semantically corresponds to a language query from an untrimmed video. Connecting computer vision and natural language, TSGV has drawn significant attention from researchers in both communities. This survey attempts to provide a summary of fundamental concepts in TSGV and current research status, as well as future research directions. As the background, we present a common structure of functional components in TSGV, in a tutorial style: from feature extraction from raw video and language query, to answer prediction of the target moment. Then we review the techniques for multimodal understanding and interaction, which is the key focus of TSGV for effective alignment between the two modalities. We construct a taxonomy of TSGV techniques and elaborate the methods in different categories with their strengths and weaknesses. La
    
[^41]: 两视角图神经网络用于知识图谱补全

    Two-view Graph Neural Networks for Knowledge Graph Completion. (arXiv:2112.09231v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2112.09231](http://arxiv.org/abs/2112.09231)

    本文提出了一种名为WGE的图神经网络模型，通过两个单一的实体和关系为中心的图来学习实体和关系的向量表示，并在知识图谱补全任务上取得了优异的性能。

    This paper proposes a graph neural network model named WGE, which learns vector representations of entities and relations from two single entity- and relation-focused graphs, and achieves excellent performance on knowledge graph completion task.

    我们提出了一种有效的基于图神经网络（GNN）的知识图谱嵌入模型，称为WGE，以捕捉实体和关系为中心的图结构。给定一个知识图谱，WGE构建一个单一的无向实体为中心的图，将实体视为节点。WGE还从关系为中心的约束条件构建另一个单一的无向图，将实体和关系视为节点。然后，WGE提出了一种基于GNN的架构，从这两个单一的实体和关系为中心的图中更好地学习实体和关系的向量表示。WGE将学习到的实体和关系表示馈送到加权得分函数中，以返回知识图谱补全的三元组得分。实验结果表明，WGE在七个知识图谱补全基准数据集上优于强基线模型。

    We present an effective graph neural network (GNN)-based knowledge graph embedding model, which we name WGE, to capture entity- and relation-focused graph structures. Given a knowledge graph, WGE builds a single undirected entity-focused graph that views entities as nodes. WGE also constructs another single undirected graph from relation-focused constraints, which views entities and relations as nodes. WGE then proposes a GNN-based architecture to better learn vector representations of entities and relations from these two single entity- and relation-focused graphs. WGE feeds the learned entity and relation representations into a weighted score function to return the triple scores for knowledge graph completion. Experimental results show that WGE outperforms strong baselines on seven benchmark datasets for knowledge graph completion.
    
[^42]: 生物医学领域中的预训练语言模型：系统调查

    Pre-trained Language Models in Biomedical Domain: A Systematic Survey. (arXiv:2110.05006v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2110.05006](http://arxiv.org/abs/2110.05006)

    本文系统调查了生物医学领域中的预训练语言模型，总结了它们的最新进展和应用，并提出了分类法。

    This paper systematically surveys pre-trained language models in the biomedical domain, summarizes their recent progress and applications, and proposes a taxonomy.

    预训练语言模型（PLMs）已成为大多数自然语言处理（NLP）任务的事实标准。这也有益于生物医学领域：来自信息学、医学和计算机科学（CS）社区的研究人员提出了各种在生物医学数据集上训练的PLMs，例如生物医学文本、电子健康记录、蛋白质和DNA序列，用于各种生物医学任务。然而，生物医学PLMs的跨学科特性阻碍了它们在社区之间的传播；一些现有的工作相互孤立，缺乏全面的比较和讨论。期望一项调查，不仅系统地审查生物医学PLMs及其应用的最新进展，而且标准化术语和基准。在本文中，我们总结了生物医学领域中预训练语言模型的最新进展以及它们在生物医学下游任务中的应用。特别是，我们讨论了动机并提出了现有生物医学PLMs的分类法。

    Pre-trained language models (PLMs) have been the de facto paradigm for most natural language processing (NLP) tasks. This also benefits biomedical domain: researchers from informatics, medicine, and computer science (CS) communities propose various PLMs trained on biomedical datasets, e.g., biomedical text, electronic health records, protein, and DNA sequences for various biomedical tasks. However, the cross-discipline characteristics of biomedical PLMs hinder their spreading among communities; some existing works are isolated from each other without comprehensive comparison and discussions. It expects a survey that not only systematically reviews recent advances of biomedical PLMs and their applications but also standardizes terminology and benchmarks. In this paper, we summarize the recent progress of pre-trained language models in the biomedical domain and their applications in biomedical downstream tasks. Particularly, we discuss the motivations and propose a taxonomy of existing b
    
[^43]: 自注意力网络可以处理有界层次语言

    Self-Attention Networks Can Process Bounded Hierarchical Languages. (arXiv:2105.11115v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2105.11115](http://arxiv.org/abs/2105.11115)

    本文证明了自注意力网络可以处理深度受限的$\mathsf{Dyck}_{k}$子集$\mathsf{Dyck}_{k,D}$，这更好地捕捉了自然语言的有界层次结构。

    This paper proves that self-attention networks can process the depth-bounded subset of $\mathsf{Dyck}_{k}$, $\mathsf{Dyck}_{k,D}$, which better captures the bounded hierarchical structure of natural language.

    尽管自注意力网络在自然语言处理中表现出色，但最近证明它们在处理具有分层结构的形式语言（例如$\mathsf{Dyck}_k$，由$k$种嵌套括号组成的语言）方面存在局限性。这表明自然语言可以用模型很好地近似，而这些模型对于形式语言来说过于弱，或者说层次结构和递归在自然语言中的作用可能是有限的。我们通过证明自注意力网络可以处理$\mathsf{Dyck}_{k,D}$来限定这一含义，其中$\mathsf{Dyck}_{k,D}$是深度受限的$\mathsf{Dyck}_{k}$子集，它更好地捕捉了自然语言的有界层次结构。具体而言，我们构建了一个具有$D+1$层和$O(\log k)$内存大小（每个令牌每层）的硬注意力网络，用于识别$\mathsf{Dyck}_{k,D}$，以及一个具有两层和$O(\log k)$内存大小的软注意力网络，用于生成$\mathsf{Dyck}_{k,D}$。实验表明，软注意力网络可以在$\mathsf{Dyck}_{k,D}$上生成正确的序列。

    Despite their impressive performance in NLP, self-attention networks were recently proved to be limited for processing formal languages with hierarchical structure, such as $\mathsf{Dyck}_k$, the language consisting of well-nested parentheses of $k$ types. This suggested that natural language can be approximated well with models that are too weak for formal languages, or that the role of hierarchy and recursion in natural language might be limited. We qualify this implication by proving that self-attention networks can process $\mathsf{Dyck}_{k, D}$, the subset of $\mathsf{Dyck}_{k}$ with depth bounded by $D$, which arguably better captures the bounded hierarchical structure of natural language. Specifically, we construct a hard-attention network with $D+1$ layers and $O(\log k)$ memory size (per token per layer) that recognizes $\mathsf{Dyck}_{k, D}$, and a soft-attention network with two layers and $O(\log k)$ memory size that generates $\mathsf{Dyck}_{k, D}$. Experiments show that s
    

