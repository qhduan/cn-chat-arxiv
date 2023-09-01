# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PointLLM: Empowering Large Language Models to Understand Point Clouds.](http://arxiv.org/abs/2308.16911) | PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。 |
| [^2] | [Transformers as Support Vector Machines.](http://arxiv.org/abs/2308.16898) | 这项工作建立了自注意力和硬间隔支持向量机问题之间的正式等价关系，通过转换器架构的优化几何来解决自然语言处理问题，同时揭示了梯度下降优化的转换器的隐式偏差。 |
| [^3] | [TouchStone: Evaluating Vision-Language Models by Language Models.](http://arxiv.org/abs/2308.16890) | TouchStone提出了一种评估方法，使用强大的语言模型作为评委来全面评估大规模视觉-语言模型的各种能力。通过构建综合的视觉对话数据集和整合图像注释，评估包括识别、理解、对话和叙事等多个能力。 |
| [^4] | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.](http://arxiv.org/abs/2308.16884) | Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。 |
| [^5] | [The Gender-GAP Pipeline: A Gender-Aware Polyglot Pipeline for Gender Characterisation in 55 Languages.](http://arxiv.org/abs/2308.16871) | 本文介绍了Gender-GAP Pipeline，一个用于55种语言中性别表征的自动流水线，通过使用多语言性别人称名词词汇表对文本进行量化来报告数据中的性别表征。在WMT训练数据和新闻任务的开发数据中表明当前数据偏向男性表征。 |
| [^6] | [Can Programming Languages Boost Each Other via Instruction Tuning?.](http://arxiv.org/abs/2308.16824) | 研究发现，编程语言可以在指令调优阶段相互促进，并显著提高彼此的能力。 |
| [^7] | [Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation.](http://arxiv.org/abs/2308.16797) | 该论文提出了一个新颖的框架，通过利用当前评估模型的优势和新建立的提示大型语言模型(LLM)范式，实现了稳健的、多语言的对话评估指标。实证结果表明，这个框架在多个基准测试中取得了最先进的结果，并在DSTC11 Track 4中的稳健和多语言任务中排名第一。 |
| [^8] | [Towards Multilingual Automatic Dialogue Evaluation.](http://arxiv.org/abs/2308.16795) | 本文提出了一种通过使用机器翻译增强现有英语对话数据的方法来解决多语言对话评估的数据不足问题。实证研究表明，仅使用源数据进行微调的强基准方法优于直接使用翻译数据微调预训练多语言模型的朴素方法。最佳方法是利用翻译质量估计指标精心筛选翻译数据，排除低质量的翻译。 |
| [^9] | [Enhancing PLM Performance on Labour Market Tasks via Instruction-based Finetuning and Prompt-tuning with Rules.](http://arxiv.org/abs/2308.16770) | 通过指令调整和规则的提示调整，我们提高了PLM在劳动力市场任务上的性能，而无需额外模型层，手动注释和数据增强。 |
| [^10] | [Ladder-of-Thought: Using Knowledge as Steps to Elevate Stance Detection.](http://arxiv.org/abs/2308.16763) | 该论文介绍了一种名为“Ladder-of-Thought”的方法，通过引入外部知识来提升立场检测任务中的语言模型的性能，解决了小型模型在应用先前内部知识时性能提升不明显的问题，以及大规模模型在效率方面的挑战。 |
| [^11] | [CReHate: Cross-cultural Re-annotation of English Hate Speech Dataset.](http://arxiv.org/abs/2308.16705) | CReHate通过跨文化重新注释英语仇恨言论数据集，揭示了来自不同国家的个体对仇恨言论的不同看法，并引入了一种具有文化敏感性的分类器。这些发现强调了重新评估NLP研究在仇恨言论领域的必要性。 |
| [^12] | [SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models.](http://arxiv.org/abs/2308.16692) | 提出了一种面向语音大语言模型的统一语音分词器SpeechTokenizer，通过统一语义和声学标记并采用编码器-解码器架构，实现了在不同层级上解耦语音信息的不同方面，构建了一个统一语音语言模型（USLM）。 |
| [^13] | [Using Large Language Models to Automate Category and Trend Analysis of Scientific Articles: An Application in Ophthalmology.](http://arxiv.org/abs/2308.16688) | 本文介绍了一种利用大型语言模型自动分类科学文章的方法，主要应用于眼科学领域，但可扩展到其他领域。通过比较不同LLM模型，结果表明LLMs在无需人工干预的情况下能有效地对大量眼科学论文进行分类。 |
| [^14] | [DictaBERT: A State-of-the-Art BERT Suite for Modern Hebrew.](http://arxiv.org/abs/2308.16687) | DictaBERT是一种最先进的预训练BERT模型，针对现代希伯来语，在大多数基准测试中表现优于其他模型。它还提供了两个经过微调的模型版本，可用于希伯来语文本分析中的前缀分割和形态标注任务。这些模型的发布旨在促进希伯来语自然语言处理的研究和发展。 |
| [^15] | [Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering.](http://arxiv.org/abs/2308.16622) | 本文介绍了一个基准测试框架，用于评估大型语言模型在知识图谱工程中的应用。框架包括语法和错误修正、事实提取和数据集生成三个挑战，同时也揭示了LLMs在零-shot提示下辅助知识图谱生成方面的不足。 |
| [^16] | [Towards Spontaneous Style Modeling with Semi-supervised Pre-training for Conversational Text-to-Speech Synthesis.](http://arxiv.org/abs/2308.16593) | 本文提出了一种半监督预训练方法，通过增加自发风格语音和自发行为标签的数量，以实现自发风格建模用于对话文本到语音合成。实验结果显示，该方法能够在自发风格的语音中建模自发行为，并从文本中预测合理的自发行为。 |
| [^17] | [Interpreting Sentiment Composition with Latent Semantic Tree.](http://arxiv.org/abs/2308.16588) | 本研究提出了一种新的树形结构——语义树，用于解释情感组合。它通过描述不同语义角色上的组合规则来实现情感组合的解释，并通过内部算法进行边际化和学习，以优化分类性能。定量和定性结果表明这种方法具有优化的分类性能和对情感组合语义的解释能力。 |
| [^18] | [Unsupervised Text Style Transfer with Deep Generative Models.](http://arxiv.org/abs/2308.16584) | 该论文提出了一种无监督的深度生成模型框架，用于实现文本风格转换。该框架通过学习潜在编码，并利用这些编码来转换句子，能够统一以前的方法，并提供一种解释先前提出的技术的原理性观点。实验结果显示，与强基线方法相比，该方法取得了更好或具有竞争力的结果。 |
| [^19] | [Improving Mandarin Prosodic Structure Prediction with Multi-level Contextual Information.](http://arxiv.org/abs/2308.16577) | 本研究通过利用多级上下文信息，包括跨句子和内部句子的语言信息，提高普通话韵律结构预测的性能，取得了更好的预测结果。 |
| [^20] | [Thesis Distillation: Investigating The Impact of Bias in NLP Models on Hate Speech Detection.](http://arxiv.org/abs/2308.16549) | 本文总结了研究自然语言处理模型中偏见对仇恨言论检测的影响的博士论文。研究发现，偏见对检测任务的影响包括可解释性、冒犯性刻板印象和公平性三个方面。为了有效解决当前在测量和减轻偏见方面的限制，需要将社会科学纳入到研究中。 |
| [^21] | [Time-Varying Quasi-Closed-Phase Analysis for Accurate Formant Tracking in Speech Signals.](http://arxiv.org/abs/2308.16540) | 本文提出了一种新的方法，利用时变准封闭相位分析进行语音信号共振峰的准确估计和追踪，通过将估计和追踪两个阶段合并为一个单一阶段，提高了共振峰估计和追踪的准确性。 |
| [^22] | [The Smart Data Extractor, a Clinician Friendly Solution to Accelerate and Improve the Data Collection During Clinical Trials.](http://arxiv.org/abs/2308.16537) | 本研究提出了一种医生友好的解决方案，即智能数据提取器，通过半自动化的方式来加速和改善临床试验期间的数据收集。与传统的手动数据收集相比，智能数据提取器能够显著减少填写时间，并提供更高质量的数据，避免了人为错误和数据重复输入。 |
| [^23] | [Generalised Winograd Schema and its Contextuality.](http://arxiv.org/abs/2308.16498) | 本文研究了广义Winograd Schema在上下文性方面的应用，提出了一种利用量子物理实验模型来解决Winograd模式挑战的方法。 |
| [^24] | [Transformer Compression via Subspace Projection.](http://arxiv.org/abs/2308.16475) | Transformer压缩通过子空间投影，在减小模型隐藏大小的同时实现了较大的模型参数和计算资源的减少，并且与其他方法兼容。 |
| [^25] | [Enhancing Subtask Performance of Multi-modal Large Language Model.](http://arxiv.org/abs/2308.16474) | 该论文提出了一种方法，通过选择多个预训练模型来完成相同的子任务，通过组合多个模型的结果获得最佳的子任务结果。 |
| [^26] | [Link Prediction for Wikipedia Articles as a Natural Language Inference Task.](http://arxiv.org/abs/2308.16469) | 本文将维基百科文章的链接预测任务视为自然语言推理任务，采用了一种新的方法，并在DSAA-2023竞赛中取得了较高的评分。 |
| [^27] | [Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models.](http://arxiv.org/abs/2308.16463) | Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。 |
| [^28] | [BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge.](http://arxiv.org/abs/2308.16458) | BioCoder是一个用于评估预训练模型在生成生物信息学代码方面的基准，涵盖了函数代码生成中的包依赖关系、类声明和全局变量，并通过模糊测试框架进行评估。 |
| [^29] | [Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer.](http://arxiv.org/abs/2308.16415) | 本文提出了一种从非流式到流式ASR编码器的知识蒸馏方法，通过逐层蒸馏和引入辅助的非流式层，以及特定的蒸馏损失函数设计，显著降低了流式ASR的词错误率。 |
| [^30] | [Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations.](http://arxiv.org/abs/2308.16349) | 我们引入了一个名为AffectVisDial的大规模数据集，其中包含50,000个基于视觉的对话，我们训练了情感视觉对话模型来解决基于对话的问答、情感预测和情感解释任务，展示出了有希望的情感推理能力。 |
| [^31] | [ToddlerBERTa: Exploiting BabyBERTa for Grammar Learning and Language Understanding.](http://arxiv.org/abs/2308.16336) | ToddlerBERTa是一个类似于BabyBERTa的语言模型，尽管在较小的数据集上进行训练，但它展示了令人称赞的性能，并具有强大的语言理解能力，与最先进的RoBERTa-base相媲美。 |
| [^32] | [Exploring Large Language Models for Knowledge Graph Completion.](http://arxiv.org/abs/2308.13916) | 本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。 |
| [^33] | [DocPrompt: Large-scale continue pretrain for zero-shot and few-shot document question answering.](http://arxiv.org/abs/2308.10959) | 本文提出了一个名为DocPrompt的方法，用于处理文档问答任务，具有强大的零样本和少样本性能。实验结果表明，DocPrompt模型经过连续预训练后在文档问答任务中表现优异，大大提高了交付效率和模型性能，降低了注释成本和劳动成本。 |
| [^34] | [Playing with Words: Comparing the Vocabulary and Lexical Richness of ChatGPT and Humans.](http://arxiv.org/abs/2308.07462) | 这篇论文比较了ChatGPT和人类在词汇和词汇丰富度方面的差异，研究发现使用ChatGPT等工具会对词汇使用和词汇丰富度产生影响，这可能会对语言演变产生影响。 |
| [^35] | [Sensi-BERT: Towards Sensitivity Driven Fine-Tuning for Parameter-Efficient BERT.](http://arxiv.org/abs/2307.11764) | Sensi-BERT是一种面向敏感度驱动的参数高效BERT微调方法，通过敏感度分析和裁剪参数张量，可生成适用于下游任务的高度参数高效的模型。 |
| [^36] | ["It Felt Like Having a Second Mind": Investigating Human-AI Co-creativity in Prewriting with Large Language Models.](http://arxiv.org/abs/2307.10811) | 通过三节次的定性研究，探究了人类与大型语言模型在预写过程中的合作模式，并发现了一个三阶段的人机共创过程：构思、启发和实施。在这个合作过程中，人类扮演着主导角色。 |
| [^37] | [Multi-Modal Discussion Transformer: Integrating Text, Images and Graph Transformers to Detect Hate Speech on Social Media.](http://arxiv.org/abs/2307.09312) | 多模态讨论变换器 (mDT) 是一个用于检测在线社交网络中仇恨言论的新颖模型。与传统的仅使用文本的方法不同，mDT通过整体分析文本和图像，结合图变换器捕捉评论周围整个讨论的上下文关系，并通过交织融合层将文本和图像嵌入进行组合。研究发现，捕捉对话的整体视图可以极大地提高检测反社会行为的准确性。 |
| [^38] | [CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care.](http://arxiv.org/abs/2307.01458) | CARE-MI是一个用于评估中国孕婴护理领域LLM虚假信息的基准，填补了这一领域的研究空白，并提供了构建长篇生成评估基准的创新范式。 |
| [^39] | [Automatic Design of Semantic Similarity Ensembles Using Grammatical Evolution.](http://arxiv.org/abs/2307.00925) | 本研究首次使用语法演化自动设计语义相似性集合，通过自动选择和聚合候选度量来优化集合与人类判断的相关性，提高相似度评估准确性，并证明了使用集合对语义相似性任务的益处。 |
| [^40] | [C-PMI: Conditional Pointwise Mutual Information for Turn-level Dialogue Evaluation.](http://arxiv.org/abs/2306.15245) | 本研究提出了一种基于条件点对点互信息的模型-无关方法，用于衡量对话系统与用户之间的交互，通过替换评分器，显著改进了与人类判断的相关性。 |
| [^41] | [Improving Non-autoregressive Translation Quality with Pretrained Language Model, Embedding Distillation and Upsampling Strategy for CTC.](http://arxiv.org/abs/2306.06345) | 本文提出了一些技术来提高非自回归翻译模型的翻译质量，在保持显着推理速度加速的同时，通过使用预训练多语言模型进行微调、采用MASK插入方案进行上采样、以及采用嵌入蒸馏方法来进一步提高性能。在多个数据集上，模型表现优于基线自回归模型。 |
| [^42] | [A First Look at LLM-Powered Generative News Recommendation.](http://arxiv.org/abs/2305.06566) | 本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。 |
| [^43] | [SCOTT: Self-Consistent Chain-of-Thought Distillation.](http://arxiv.org/abs/2305.01879) | 本研究提出了一种忠实的知识蒸馏方法，从比教师模型大数倍的模型中学习一个小的、自我一致的思路串模型。实验结果表明，该方法有助于证明决策并提高性能，特别是在较小的语言模型中。 |
| [^44] | [OLISIA: a Cascade System for Spoken Dialogue State Tracking.](http://arxiv.org/abs/2304.11073) | 我们提出了OLISIA，一个口语对话状态跟踪的级联系统，使用自动语音识别和DST模型，采用几个适应性策略来提高稳健性，并在DSTC11 Track3中取得第一名的好成绩。 |
| [^45] | [Deanthropomorphising NLP: Can a Language Model Be Conscious?.](http://arxiv.org/abs/2211.11483) | 本文讨论了关于使用Transformer架构的预训练语言模型LaMDA是否具有意识的说法。作者认为语言模型不可能具有意识，而LaMDA没有比其他类似模型更具先进性。 |
| [^46] | [DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generation Models.](http://arxiv.org/abs/2202.04053) | 本研究探究了不同文本到图像模型的视觉推理能力和社会偏见。尽管模型在图像生成方面表现出高质量的结果，但在对象计数和空间关系理解能力方面仍存在与上界准确性之间的巨大差距。此外，我们还评估了模型在性别和肤色方面的偏见。 |

# 详细

[^1]: PointLLM：赋予大型语言模型理解点云的能力

    PointLLM: Empowering Large Language Models to Understand Point Clouds. (arXiv:2308.16911v1 [cs.CV])

    [http://arxiv.org/abs/2308.16911](http://arxiv.org/abs/2308.16911)

    PointLLM是一种使大型语言模型理解点云的方法，它利用点云编码器和强大的LLM将几何、外观和语言信息融合，并通过人类指导生成环境上恰当的响应。该方法通过收集大规模的点-文本指令对数据集进行两阶段的训练，以提高模型的感知能力和泛化能力。

    

    大型语言模型（LLM）的前所未有的进展对自然语言处理产生了深远影响，但在3D理解领域仍有待完全发展。本文介绍了PointLLM，这是一项填补这一空白的初步工作，使LLM能够理解点云，并提供了超越2D视觉数据的新途径。PointLLM通过人类指导处理带有颜色的物体点云，并生成环境上恰当的响应，展示了其对点云和常识的掌握。具体来说，它利用了一个点云编码器和一个强大的LLM，有效地融合了几何、外观和语言信息。我们收集了一个新颖的数据集，包括66万个简单和7万个复杂的点-文本指令对，以实现两阶段的训练策略：首先对齐潜在空间，然后对统一模型进行指令调整。为了严格评估我们模型的感知能力和其泛化能力，我们建立了评估基准数据集进行实验。

    The unprecedented advancements in Large Language Models (LLMs) have created a profound impact on natural language processing but are yet to fully embrace the realm of 3D understanding. This paper introduces PointLLM, a preliminary effort to fill this gap, thereby enabling LLMs to understand point clouds and offering a new avenue beyond 2D visual data. PointLLM processes colored object point clouds with human instructions and generates contextually appropriate responses, illustrating its grasp of point clouds and common sense. Specifically, it leverages a point cloud encoder with a powerful LLM to effectively fuse geometric, appearance, and linguistic information. We collect a novel dataset comprising 660K simple and 70K complex point-text instruction pairs to enable a two-stage training strategy: initially aligning latent spaces and subsequently instruction-tuning the unified model. To rigorously evaluate our model's perceptual abilities and its generalization capabilities, we establis
    
[^2]: Transformers作为支持向量机

    Transformers as Support Vector Machines. (arXiv:2308.16898v1 [cs.LG])

    [http://arxiv.org/abs/2308.16898](http://arxiv.org/abs/2308.16898)

    这项工作建立了自注意力和硬间隔支持向量机问题之间的正式等价关系，通过转换器架构的优化几何来解决自然语言处理问题，同时揭示了梯度下降优化的转换器的隐式偏差。

    

    自从"Attention Is All You Need"中引入转换器架构以来，它在自然语言处理领域取得了革命性的进展。转换器中的注意力层接受输入令牌序列$X$并通过计算softmax$(XQK^\top X^\top)$的成对相似性使它们相互作用，其中$(K,Q)$是可训练的键-查询参数。在这项工作中，我们建立了自注意力优化几何和一个硬间隔支持向量机问题之间的正式等价关系，通过对令牌对的外积施加线性约束，将最佳输入令牌与非最佳令牌分离。这个形式主义使我们能够表征梯度下降优化的单层转换器的隐式偏差：(1)优化注意力层，使用可变正则化参数$(K,Q)$，收敛的方向是一个最小化综合参数$W=KQ^\top$的核范数的支持向量机解决方案。而直接使用$W$进行参数化则最小化一个Frobenius范数目标。

    Since its inception in "Attention Is All You Need", transformer architecture has led to revolutionary advancements in NLP. The attention layer within the transformer admits a sequence of input tokens $X$ and makes them interact through pairwise similarities computed as softmax$(XQK^\top X^\top)$, where $(K,Q)$ are the trainable key-query parameters. In this work, we establish a formal equivalence between the optimization geometry of self-attention and a hard-margin SVM problem that separates optimal input tokens from non-optimal tokens using linear constraints on the outer-products of token pairs. This formalism allows us to characterize the implicit bias of 1-layer transformers optimized with gradient descent: (1) Optimizing the attention layer with vanishing regularization, parameterized by $(K,Q)$, converges in direction to an SVM solution minimizing the nuclear norm of the combined parameter $W=KQ^\top$. Instead, directly parameterizing by $W$ minimizes a Frobenius norm objective. 
    
[^3]: TouchStone: 用语言模型评估视觉-语言模型

    TouchStone: Evaluating Vision-Language Models by Language Models. (arXiv:2308.16890v1 [cs.CV])

    [http://arxiv.org/abs/2308.16890](http://arxiv.org/abs/2308.16890)

    TouchStone提出了一种评估方法，使用强大的语言模型作为评委来全面评估大规模视觉-语言模型的各种能力。通过构建综合的视觉对话数据集和整合图像注释，评估包括识别、理解、对话和叙事等多个能力。

    

    大规模视觉-语言模型（LVLMs）近年来取得了快速进展，通过将视觉接收器与大型语言模型（LLMs）相连接，展现出了惊人的感知、理解和处理视觉信息的能力。然而，目前的评估主要关注识别和推理能力，缺乏对对话能力和视觉叙事能力的直接评估。本文提出了一种评估方法，使用强大的LLMs作为评委来全面评估LVLMs的各种能力。首先，我们构建了一个包含开放世界图像和问题的综合视觉对话数据集TouchStone，涵盖了五个主要能力和27个子任务。该数据集不仅涵盖了基础的识别和理解，还扩展到文学创作。其次，通过整合详细的图像注释，我们有效地将多模态输入内容转化为LLMs可以理解的形式。

    Large vision-language models (LVLMs) have recently witnessed rapid advancements, exhibiting a remarkable capacity for perceiving, understanding, and processing visual information by connecting visual receptor with large language models (LLMs). However, current assessments mainly focus on recognizing and reasoning abilities, lacking direct evaluation of conversational skills and neglecting visual storytelling abilities. In this paper, we propose an evaluation method that uses strong LLMs as judges to comprehensively evaluate the various abilities of LVLMs. Firstly, we construct a comprehensive visual dialogue dataset TouchStone, consisting of open-world images and questions, covering five major categories of abilities and 27 subtasks. This dataset not only covers fundamental recognition and comprehension but also extends to literary creation. Secondly, by integrating detailed image annotations we effectively transform the multimodal input content into a form understandable by LLMs. This
    
[^4]: Belebele基准数据集：122种语言变体的并行阅读理解数据集

    The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. (arXiv:2308.16884v1 [cs.CL])

    [http://arxiv.org/abs/2308.16884](http://arxiv.org/abs/2308.16884)

    Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。

    

    我们提出了Belebele，一个包含122种语言变体的多选机器阅读理解（MRC）数据集。该数据集极大地扩展了自然语言理解（NLU）基准的语言覆盖范围，使得可以评估文本模型在高、中和低资源语言中的性能。每个问题都基于Flores-200数据集中的一个短篇文章，并提供了四个多选答案。问题经过精心策划，以区分具有不同通用语言理解水平的模型。单独的英语数据集已经足够困难，可以挑战最先进的语言模型。由于完全并行，该数据集可以直接比较所有语言的模型性能。我们使用该数据集评估多语言遮蔽语言模型（MLMs）和大型语言模型（LLMs）的能力。我们展示了广泛的结果，并发现尽管英语为中心的LLMs之间存在显著的跨语言转移，但小型MLMs在其他语言上的表现相对较好。

    We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and find that despite significant cross-lingual transfer in English-centric LLMs, much small
    
[^5]: The Gender-GAP Pipeline: 一个用于55种语言中性别表征的性别感知多语言流水线

    The Gender-GAP Pipeline: A Gender-Aware Polyglot Pipeline for Gender Characterisation in 55 Languages. (arXiv:2308.16871v1 [cs.CL])

    [http://arxiv.org/abs/2308.16871](http://arxiv.org/abs/2308.16871)

    本文介绍了Gender-GAP Pipeline，一个用于55种语言中性别表征的自动流水线，通过使用多语言性别人称名词词汇表对文本进行量化来报告数据中的性别表征。在WMT训练数据和新闻任务的开发数据中表明当前数据偏向男性表征。

    

    语言生成系统中的性别偏见很难被缓解。其中一个可能导致这种偏见的原因是训练和评估数据中的性别表征不平衡。尽管最近在记录这个问题和试图缓解它方面取得了一些进展，但我们仍然缺乏共享的方法论和工具，以报告大规模数据集中的性别表征。这种定量报告将使进一步缓解成为可能，例如通过数据增强。本文描述了Gender-GAP Pipeline（用于性别感知的多语言流水线），它是一个自动流程，用于对55种语言的大规模数据集进行性别表征。该流水线使用一个多语言性别人称名词词汇表来量化文本中的性别表征。我们展示了它来报告WMT训练数据和新闻任务的开发数据中的性别表征，证实当前数据偏向男性表征。拥有不平衡的数据集可能会间接地优化我们的系统。

    Gender biases in language generation systems are challenging to mitigate. One possible source for these biases is gender representation disparities in the training and evaluation data. Despite recent progress in documenting this problem and many attempts at mitigating it, we still lack shared methodology and tooling to report gender representation in large datasets. Such quantitative reporting will enable further mitigation, e.g., via data augmentation. This paper describes the Gender-GAP Pipeline (for Gender-Aware Polyglot Pipeline), an automatic pipeline to characterize gender representation in large-scale datasets for 55 languages. The pipeline uses a multilingual lexicon of gendered person-nouns to quantify the gender representation in text. We showcase it to report gender representation in WMT training data and development data for the News task, confirming that current data is skewed towards masculine representation. Having unbalanced datasets may indirectly optimize our systems 
    
[^6]: 编程语言能通过指令调优相互提升吗？

    Can Programming Languages Boost Each Other via Instruction Tuning?. (arXiv:2308.16824v1 [cs.CL])

    [http://arxiv.org/abs/2308.16824](http://arxiv.org/abs/2308.16824)

    研究发现，编程语言可以在指令调优阶段相互促进，并显著提高彼此的能力。

    

    当人类程序员掌握了一种编程语言后，学习一种新的编程语言会更容易。在本报告中，我们重点探讨了在代码大规模语言模型的指令微调阶段中，编程语言是否能够通过相互提升来增强彼此的能力。我们在StarCoder上对8种流行的编程语言进行了广泛的实验（Python，JavaScript，TypeScript，C，C ++，Java，Go，HTML）。结果表明，编程语言可以显著提高彼此的能力。例如，通过在Python上训练的CodeM-Python 15B可以使Java的pass@1率绝对增加了17.95％。更令人惊讶的是，我们发现通过在HTML语料库上训练的CodeM-HTML 7B可以使Java的pass@1率绝对增加了15.24％。我们的训练数据已经发布在https://github.com/NL2Code/CodeM上。

    When human programmers have mastered a programming language, it would be easier when they learn a new programming language. In this report, we focus on exploring whether programming languages can boost each other during the instruction fine-tuning phase of code large language models. We conduct extensive experiments of 8 popular programming languages (Python, JavaScript, TypeScript, C, C++, Java, Go, HTML) on StarCoder. Results demonstrate that programming languages can significantly improve each other. For example, CodeM-Python 15B trained on Python is able to increase Java by an absolute 17.95% pass@1 on HumanEval-X. More surprisingly, we found that CodeM-HTML 7B trained on the HTML corpus can improve Java by an absolute 15.24% pass@1. Our training data is released at https://github.com/NL2Code/CodeM.
    
[^7]: 简单的LLM提示是稳健且多语言对话评价的最先进技术

    Simple LLM Prompting is State-of-the-Art for Robust and Multilingual Dialogue Evaluation. (arXiv:2308.16797v1 [cs.CL])

    [http://arxiv.org/abs/2308.16797](http://arxiv.org/abs/2308.16797)

    该论文提出了一个新颖的框架，通过利用当前评估模型的优势和新建立的提示大型语言模型(LLM)范式，实现了稳健的、多语言的对话评估指标。实证结果表明，这个框架在多个基准测试中取得了最先进的结果，并在DSTC11 Track 4中的稳健和多语言任务中排名第一。

    

    尽管在自动对话评价指标的开发上已经付出了大量的研究工作，但对评价非英语对话的思考却很少。与此同时，确保指标对语义相似的回答不变也是一个被忽视的问题。为了实现稳健性和多语言性对话评估指标的期望属性，我们提出了一个新颖的框架，利用当前评估模型的优势和新建立的提示大型语言模型(LLM)范式。实证结果表明，我们的框架在多个基准测试中以平均斯皮尔曼相关得分创造了最先进的结果，并在DSTC11 Track 4“开放域对话系统的自动评价指标”中的稳健和多语言任务中排名第一，证明了提示LLM的评估能力。

    Despite significant research effort in the development of automatic dialogue evaluation metrics, little thought is given to evaluating dialogues other than in English. At the same time, ensuring metrics are invariant to semantically similar responses is also an overlooked topic. In order to achieve the desired properties of robustness and multilinguality for dialogue evaluation metrics, we propose a novel framework that takes advantage of the strengths of current evaluation models with the newly-established paradigm of prompting Large Language Models (LLMs). Empirical results show our framework achieves state of the art results in terms of mean Spearman correlation scores across several benchmarks and ranks first place on both the Robust and Multilingual tasks of the DSTC11 Track 4 "Automatic Evaluation Metrics for Open-Domain Dialogue Systems", proving the evaluation capabilities of prompted LLMs.
    
[^8]: 实现多语言自动对话评估的方法

    Towards Multilingual Automatic Dialogue Evaluation. (arXiv:2308.16795v1 [cs.CL])

    [http://arxiv.org/abs/2308.16795](http://arxiv.org/abs/2308.16795)

    本文提出了一种通过使用机器翻译增强现有英语对话数据的方法来解决多语言对话评估的数据不足问题。实证研究表明，仅使用源数据进行微调的强基准方法优于直接使用翻译数据微调预训练多语言模型的朴素方法。最佳方法是利用翻译质量估计指标精心筛选翻译数据，排除低质量的翻译。

    

    多语言对话评估指标发展的主要限制因素是缺乏多语言数据和少量的开源多语言对话系统。本文提出了一种解决这种数据不足的方法，即利用强大的预训练多语言语言模型（LLM）并使用机器翻译来增强现有的英语对话数据。我们的实证研究表明，直接使用翻译数据来微调预训练的多语言编码模型的朴素方法无法超过只使用源数据微调多语言模型的强基准。相反，最佳方法是通过使用翻译质量估计指标来精心筛选翻译数据，排除低质量的翻译，以提高其性能。

    The main limiting factor in the development of robust multilingual dialogue evaluation metrics is the lack of multilingual data and the limited availability of open sourced multilingual dialogue systems. In this work, we propose a workaround for this lack of data by leveraging a strong multilingual pretrained LLM and augmenting existing English dialogue data using Machine Translation. We empirically show that the naive approach of finetuning a pretrained multilingual encoder model with translated data is insufficient to outperform the strong baseline of finetuning a multilingual model with only source data. Instead, the best approach consists in the careful curation of translated data using MT Quality Estimation metrics, excluding low quality translations that hinder its performance.
    
[^9]: 通过基于指令的微调和规则的提示微调增强劳动力市场任务的PLM性能

    Enhancing PLM Performance on Labour Market Tasks via Instruction-based Finetuning and Prompt-tuning with Rules. (arXiv:2308.16770v1 [cs.CL])

    [http://arxiv.org/abs/2308.16770](http://arxiv.org/abs/2308.16770)

    通过指令调整和规则的提示调整，我们提高了PLM在劳动力市场任务上的性能，而无需额外模型层，手动注释和数据增强。

    

    劳动力市场的数字化增长使得研究人员、教育者和企业能够分析和更好地理解劳动力市场。然而，尽管劳动力市场资源数量庞大，但往往是非结构化的，因此，对于实体的识别、关联和提取的方法研究变得越来越重要。在追求更好的劳动力市场表现的背景下，资源限制和大规模标注数据不可用导致人类领域专家的依赖。我们证明了基于提示的预训练语言模型（PLM）在劳动力市场特定应用中的有效性。我们的结果表明，PTR和没有示例的指令调整等成本效益方法能够显著提高PLM在劳动力市场应用中的性能，而无需引入额外的模型层、手动注释和数据增强。

    The increased digitization of the labour market has given researchers, educators, and companies the means to analyze and better understand the labour market. However, labour market resources, although available in high volumes, tend to be unstructured, and as such, research towards methodologies for the identification, linking, and extraction of entities becomes more and more important. Against the backdrop of this quest for better labour market representations, resource constraints and the unavailability of large-scale annotated data cause a reliance on human domain experts. We demonstrate the effectiveness of prompt-based tuning of pre-trained language models (PLM) in labour market specific applications. Our results indicate that cost-efficient methods such as PTR and instruction tuning without exemplars can significantly increase the performance of PLMs on downstream labour market applications without introducing additional model layers, manual annotations, and data augmentation.
    
[^10]: Ladder-of-Thought: 使用知识作为阶梯提升立场检测

    Ladder-of-Thought: Using Knowledge as Steps to Elevate Stance Detection. (arXiv:2308.16763v1 [cs.CL])

    [http://arxiv.org/abs/2308.16763](http://arxiv.org/abs/2308.16763)

    该论文介绍了一种名为“Ladder-of-Thought”的方法，通过引入外部知识来提升立场检测任务中的语言模型的性能，解决了小型模型在应用先前内部知识时性能提升不明显的问题，以及大规模模型在效率方面的挑战。

    

    思维链式提供（CoT）通过生成中间的推理来增强大型语言模型（LLM）的推理能力。然而，这些增强主要有益于大规模模型，在直接应用CoT时小型LLM的性能改进不明显。尽管LLM具有先进的推理能力，CoT主要依赖于其预先训练的内部知识，先前未知于模型的外部知识未被充分利用。在立场检测等任务中，外部背景知识起着关键作用，这种遗漏变得更加明显。此外，LLM的大规模架构在部署过程中不可避免地存在效率挑战。为了解决这些挑战，我们引入了用于立场检测的思维阶梯（LoT）。LoT基于双阶段级联优化框架，指导模型整合高质量的外部知识，增强中间步骤的性能。

    Chain-of-Thought Prompting (CoT) reinforces the reasoning capabilities of Large Language Models (LLMs) through the generation of intermediate rationales. However, these enhancements predominantly benefit large-scale models, leaving small LMs without significant performance improvements when directly applying CoT. Despite the advanced reasoning capabilities of LLMs, CoT relies primarily on their pre-trained internal knowledge. The external knowledge that is previously unknown to the model remains unexploited. This omission becomes pronounced in tasks such as stance detection, where the external background knowledge plays a pivotal role. Additionally, the large-scale architecture of LLMs inevitably present efficiency challenges during deployment. To address these challenges, we introduce the Ladder-of-Thought (LoT) for stance detection. Grounded in a dual-phase Cascaded Optimization framework, LoT directs the model to incorporate high-quality external knowledge, enhancing the intermediat
    
[^11]: CReHate: 跨文化重新注释英语仇恨言论数据集

    CReHate: Cross-cultural Re-annotation of English Hate Speech Dataset. (arXiv:2308.16705v1 [cs.CL])

    [http://arxiv.org/abs/2308.16705](http://arxiv.org/abs/2308.16705)

    CReHate通过跨文化重新注释英语仇恨言论数据集，揭示了来自不同国家的个体对仇恨言论的不同看法，并引入了一种具有文化敏感性的分类器。这些发现强调了重新评估NLP研究在仇恨言论领域的必要性。

    

    英语数据集主要反映了特定国家的观点，这可能导致模型和数据集中存在文化偏差。这在受主观性影响较大的任务，如仇恨言论检测中特别有问题。为了深入了解来自不同国家的个体如何理解仇恨言论，我们介绍了CReHate，对抽样的SBIC数据集进行了跨文化重新注释。该数据集包括来自五个不同国家的注释：澳大利亚、新加坡、南非、英国和美国。我们进行了彻底的统计分析，发现基于国籍存在显著差异，只有59.4%的样本在所有国家之间达成共识。我们还通过迁移学习引入了一种具有文化敏感性的仇恨言论分类器，能够捕捉不同国籍的观点。这些发现强调了需要重新评估自然语言处理研究的某些方面，特别是对于仇恨言论的细微性质。

    English datasets predominantly reflect the perspectives of certain nationalities, which can lead to cultural biases in models and datasets. This is particularly problematic in tasks heavily influenced by subjectivity, such as hate speech detection. To delve into how individuals from different countries perceive hate speech, we introduce CReHate, a cross-cultural re-annotation of the sampled SBIC dataset. This dataset includes annotations from five distinct countries: Australia, Singapore, South Africa, the United Kingdom, and the United States. Our thorough statistical analysis highlights significant differences based on nationality, with only 59.4% of the samples achieving consensus among all countries. We also introduce a culturally sensitive hate speech classifier via transfer learning, adept at capturing perspectives of different nationalities. These findings underscore the need to re-evaluate certain aspects of NLP research, especially with regard to the nuanced nature of hate spe
    
[^12]: SpeechTokenizer：面向语音大语言模型的统一语音分词器

    SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models. (arXiv:2308.16692v1 [cs.CL])

    [http://arxiv.org/abs/2308.16692](http://arxiv.org/abs/2308.16692)

    提出了一种面向语音大语言模型的统一语音分词器SpeechTokenizer，通过统一语义和声学标记并采用编码器-解码器架构，实现了在不同层级上解耦语音信息的不同方面，构建了一个统一语音语言模型（USLM）。

    

    当前的语音大语言模型基于离散的语音表示，可以分为语义标记和声学标记。然而，现有的语音标记并非专为语音语言建模而设计。为了评估语音标记在构建语音语言模型方面的适应性，我们建立了第一个基准标准，即SLMTokBench。我们的结果表明，无论是语义标记还是声学标记都不适合这个目的。因此，我们提出了SpeechTokenizer，一种面向语音大语言模型的统一语音分词器。SpeechTokenizer采用具有残差向量量化（RVQ）的编码器-解码器架构。通过统一语义和声学标记，SpeechTokenizer在不同的RVQ层级上以层次方式解耦语音信息的不同方面。此外，我们构建了一个利用SpeechTokenizer的统一语音语言模型（USLM）。实验证明，SpeechTokenizer在语音重建和...

    Current speech large language models build upon discrete speech representations, which can be categorized into semantic tokens and acoustic tokens. However, existing speech tokens are not specifically designed for speech language modeling. To assess the suitability of speech tokens for building speech language models, we established the first benchmark, SLMTokBench. Our results indicate that neither semantic nor acoustic tokens are ideal for this purpose. Therefore, we propose SpeechTokenizer, a unified speech tokenizer for speech large language models. SpeechTokenizer adopts the Encoder-Decoder architecture with residual vector quantization (RVQ). Unifying semantic and acoustic tokens, SpeechTokenizer disentangles different aspects of speech information hierarchically across different RVQ layers. Furthermore, We construct a Unified Speech Language Model (USLM) leveraging SpeechTokenizer. Experiments show that SpeechTokenizer performs comparably to EnCodec in speech reconstruction and 
    
[^13]: 使用大型语言模型自动化科学文章的分类和趋势分析：以眼科学为应用实例

    Using Large Language Models to Automate Category and Trend Analysis of Scientific Articles: An Application in Ophthalmology. (arXiv:2308.16688v1 [cs.CL])

    [http://arxiv.org/abs/2308.16688](http://arxiv.org/abs/2308.16688)

    本文介绍了一种利用大型语言模型自动分类科学文章的方法，主要应用于眼科学领域，但可扩展到其他领域。通过比较不同LLM模型，结果表明LLMs在无需人工干预的情况下能有效地对大量眼科学论文进行分类。

    

    本文介绍了一种利用大型语言模型（LLM）自动进行文章分类的方法。主要关注眼科领域，但该模型可扩展到其他领域。通过自然语言处理技术（NLP）开发了一个模型，包括高级LLM，用于处理和分析科学论文的文本内容。在LLM模型中，我们采用了零样本学习（ZSL）LLM模型，并与双向和自回归变换器（BART）及其变种，以及双向编码器表示从变换器（BERT）及其变种（如distilBERT，SciBERT，PubmedBERT，BioBERT）进行比较。分类结果表明，在没有人为干预的情况下，LLM在对大量眼科学论文进行分类方面具有有效性。

    Purpose: In this paper, we present an automated method for article classification, leveraging the power of Large Language Models (LLM). The primary focus is on the field of ophthalmology, but the model is extendable to other fields. Methods: We have developed a model based on Natural Language Processing (NLP) techniques, including advanced LLMs, to process and analyze the textual content of scientific papers. Specifically, we have employed zero-shot learning (ZSL) LLM models and compared against Bidirectional and Auto-Regressive Transformers (BART) and its variants, and Bidirectional Encoder Representations from Transformers (BERT), and its variant such as distilBERT, SciBERT, PubmedBERT, BioBERT. Results: The classification results demonstrate the effectiveness of LLMs in categorizing large number of ophthalmology papers without human intervention. Results: To evalute the LLMs, we compiled a dataset (RenD) of 1000 ocular disease-related articles, which were expertly annotated by a pan
    
[^14]: DictaBERT: 一款用于现代希伯来语的最先进BERT套件的翻译标题

    DictaBERT: A State-of-the-Art BERT Suite for Modern Hebrew. (arXiv:2308.16687v1 [cs.CL])

    [http://arxiv.org/abs/2308.16687](http://arxiv.org/abs/2308.16687)

    DictaBERT是一种最先进的预训练BERT模型，针对现代希伯来语，在大多数基准测试中表现优于其他模型。它还提供了两个经过微调的模型版本，可用于希伯来语文本分析中的前缀分割和形态标注任务。这些模型的发布旨在促进希伯来语自然语言处理的研究和发展。

    

    我们提出了DictaBERT，这是一种用于现代希伯来语的最先进的预训练BERT模型，在大多数基准测试中表现优于现有模型。此外，我们发布了两个经过微调的模型版本，旨在执行希伯来语文本分析的两个特定基本任务：前缀分割和形态标注。这些经过微调的模型允许任何开发人员只需调用HuggingFace模型一次即可对希伯来语句子进行前缀分割和形态标注，无需集成任何额外的库或代码。在本文中，我们描述了训练的细节以及在不同基准测试上的结果。我们将这些模型与展示其使用的示例代码一起发布给社区。我们发布这些模型是为了帮助进一步促进希伯来语自然语言处理的研究和发展。

    We present DictaBERT, a new state-of-the-art pre-trained BERT model for modern Hebrew, outperforming existing models on most benchmarks. Additionally, we release two fine-tuned versions of the model, designed to perform two specific foundational tasks in the analysis of Hebrew texts: prefix segmentation and morphological tagging. These fine-tuned models allow any developer to perform prefix segmentation and morphological tagging of a Hebrew sentence with a single call to a HuggingFace model, without the need to integrate any additional libraries or code. In this paper we describe the details of the training as well and the results on the different benchmarks. We release the models to the community, along with sample code demonstrating their use. We release these models as part of our goal to help further research and development in Hebrew NLP.
    
[^15]: 在知识图谱工程中开发一个可扩展的用于评估大型语言模型的基准测试

    Developing a Scalable Benchmark for Assessing Large Language Models in Knowledge Graph Engineering. (arXiv:2308.16622v1 [cs.AI])

    [http://arxiv.org/abs/2308.16622](http://arxiv.org/abs/2308.16622)

    本文介绍了一个基准测试框架，用于评估大型语言模型在知识图谱工程中的应用。框架包括语法和错误修正、事实提取和数据集生成三个挑战，同时也揭示了LLMs在零-shot提示下辅助知识图谱生成方面的不足。

    

    随着大型语言模型（LLMs）领域的快速发展，评估和监测其性能的迫切需求浮出水面。我们介绍了一个针对知识图谱工程（KGE）的基准测试框架，并提出了三个挑战，涉及语法和错误修正、事实提取和数据集生成。我们展示了尽管LLMs是有用的工具，但它们尚不能在零-shot提示下辅助知识图谱生成。因此，我们的LLM-KG-Bench框架提供了LLM回答的自动评估和存储，以及统计数据和可视化工具，支持提示工程和模型性能的跟踪。

    As the field of Large Language Models (LLMs) evolves at an accelerated pace, the critical need to assess and monitor their performance emerges. We introduce a benchmarking framework focused on knowledge graph engineering (KGE) accompanied by three challenges addressing syntax and error correction, facts extraction and dataset generation. We show that while being a useful tool, LLMs are yet unfit to assist in knowledge graph generation with zero-shot prompting. Consequently, our LLM-KG-Bench framework provides automatic evaluation and storage of LLM responses as well as statistical data and visualization tools to support tracking of prompt engineering and model performance.
    
[^16]: 通过半监督预训练实现自发风格建模用于对话文本到语音合成

    Towards Spontaneous Style Modeling with Semi-supervised Pre-training for Conversational Text-to-Speech Synthesis. (arXiv:2308.16593v1 [cs.SD])

    [http://arxiv.org/abs/2308.16593](http://arxiv.org/abs/2308.16593)

    本文提出了一种半监督预训练方法，通过增加自发风格语音和自发行为标签的数量，以实现自发风格建模用于对话文本到语音合成。实验结果显示，该方法能够在自发风格的语音中建模自发行为，并从文本中预测合理的自发行为。

    

    对话中经常发生的自发行为使得语音听起来更加像人类，而不是像朗读。然而，合成自发风格的语音具有挑战性，因为缺乏高质量的自发数据集，并且标记自发行为的成本较高。在本文中，我们提出了一种半监督预训练方法，以增加自发风格语音和自发行为标签的数量。在半监督学习的过程中，考虑了文本和语音信息，以便在语音中检测自发行为标签。此外，使用了一个语言感知编码器来建模对话中每个句子之间的关系。实验结果表明，我们提出的方法在表达性语音合成性能方面具有优越性，能够在自发风格的语音中建模自发行为，并从文本中预测合理的自发行为。

    The spontaneous behavior that often occurs in conversations makes speech more human-like compared to reading-style. However, synthesizing spontaneous-style speech is challenging due to the lack of high-quality spontaneous datasets and the high cost of labeling spontaneous behavior. In this paper, we propose a semi-supervised pre-training method to increase the amount of spontaneous-style speech and spontaneous behavioral labels. In the process of semi-supervised learning, both text and speech information are considered for detecting spontaneous behaviors labels in speech. Moreover, a linguistic-aware encoder is used to model the relationship between each sentence in the conversation. Experimental results indicate that our proposed method achieves superior expressive speech synthesis performance with the ability to model spontaneous behavior in spontaneous-style speech and predict reasonable spontaneous behavior from text.
    
[^17]: 用潜在语义树解释情感组合

    Interpreting Sentiment Composition with Latent Semantic Tree. (arXiv:2308.16588v1 [cs.CL])

    [http://arxiv.org/abs/2308.16588](http://arxiv.org/abs/2308.16588)

    本研究提出了一种新的树形结构——语义树，用于解释情感组合。它通过描述不同语义角色上的组合规则来实现情感组合的解释，并通过内部算法进行边际化和学习，以优化分类性能。定量和定性结果表明这种方法具有优化的分类性能和对情感组合语义的解释能力。

    

    作为情感分析的关键，情感组合考虑通过子成分的分类和应用于它们的规则对成分进行分类。我们认为，以前广泛研究过的包括未标记和情感树在内的分层树形结构在本质上是次优的。为了解决这个问题，我们提出了语义树，一种能够以系统的方式解释情感组合的新树形结构。语义树是上下文无关语法（CFG）的一个派生，描述了不同语义角色上的特定组合规则，其设计经过了之前的语言学结论的精心考虑。然而，由于常规数据集中没有对语义树的注释，语义树是一个潜在变量。因此，在我们的方法中，通过内部算法对其进行边际化，并通过学习优化分类性能。定量和定性结果表明，我们的方法不仅实现了优化的分类性能，还能够解释情感组合的语义。

    As the key to sentiment analysis, sentiment composition considers the classification of a constituent via classifications of its contained sub-constituents and rules operated on them. Such compositionality has been widely studied previously in the form of hierarchical trees including untagged and sentiment ones, which are intrinsically suboptimal in our view. To address this, we propose semantic tree, a new tree form capable of interpreting the sentiment composition in a principled way. Semantic tree is a derivation of a context-free grammar (CFG) describing the specific composition rules on difference semantic roles, which is designed carefully following previous linguistic conclusions. However, semantic tree is a latent variable since there is no its annotation in regular datasets. Thus, in our method, it is marginalized out via inside algorithm and learned to optimize the classification performance. Quantitative and qualitative results demonstrate that our method not only achieves b
    
[^18]: 无监督的深度生成模型实现文本风格转换

    Unsupervised Text Style Transfer with Deep Generative Models. (arXiv:2308.16584v1 [cs.CL])

    [http://arxiv.org/abs/2308.16584](http://arxiv.org/abs/2308.16584)

    该论文提出了一种无监督的深度生成模型框架，用于实现文本风格转换。该框架通过学习潜在编码，并利用这些编码来转换句子，能够统一以前的方法，并提供一种解释先前提出的技术的原理性观点。实验结果显示，与强基线方法相比，该方法取得了更好或具有竞争力的结果。

    

    我们提出了一种使用深度生成模型实现无监督文本风格转换的通用框架。该框架将非平行语料中的每个句子-标签对建模为一个部分观测的完整四元组，该四元组还包含表示内容和风格的两个潜在编码。这些编码通过利用观测数据内部的依赖关系来进行学习。然后通过操纵这些编码来实现句子的转换。我们的框架能够将以前的嵌入和原型方法统一为两个特殊形式。它还提供了一种原理性的观点来解释领域中先前提出的技术，如对齐编码器和对抗训练。我们进一步在三个基准测试上进行了实验证明。自动评估和人工评估结果表明，与几个强基线方法相比，我们的方法取得了更好或具有竞争力的结果。

    We present a general framework for unsupervised text style transfer with deep generative models. The framework models each sentence-label pair in the non-parallel corpus as partially observed from a complete quadruplet which additionally contains two latent codes representing the content and style, respectively. These codes are learned by exploiting dependencies inside the observed data. Then a sentence is transferred by manipulating them. Our framework is able to unify previous embedding and prototype methods as two special forms. It also provides a principled perspective to explain previously proposed techniques in the field such as aligned encoder and adversarial training. We further conduct experiments on three benchmarks. Both automatic and human evaluation results show that our methods achieve better or competitive results compared to several strong baselines.
    
[^19]: 用多级上下文信息提升普通话韵律结构预测

    Improving Mandarin Prosodic Structure Prediction with Multi-level Contextual Information. (arXiv:2308.16577v1 [cs.SD])

    [http://arxiv.org/abs/2308.16577](http://arxiv.org/abs/2308.16577)

    本研究通过利用多级上下文信息，包括跨句子和内部句子的语言信息，提高普通话韵律结构预测的性能，取得了更好的预测结果。

    

    对于文本到语音合成（TTS）而言，韵律结构预测（PSP）在生成自然和可理解的语音方面起着重要的作用。尽管跨句子的语言信息可以影响目标句子的语音解释，但之前关于PSP的研究主要集中在仅利用当前句子的内部语言信息上。本文提出利用跨句子的语言信息来提高PSP的性能。通过层次编码器从输入文本的字符级、句子级和话语级提取多级上下文信息，包括跨句子和内部句子的语言信息。然后，多任务学习（MTL）解码器从多级上下文信息中预测韵律边界。在两个数据集上的客观评估结果显示，我们的方法在预测韵律词（PW）、韵律短语（PPH）和语调短语（IPH）方面取得了更好的F1得分。

    For text-to-speech (TTS) synthesis, prosodic structure prediction (PSP) plays an important role in producing natural and intelligible speech. Although inter-utterance linguistic information can influence the speech interpretation of the target utterance, previous works on PSP mainly focus on utilizing intrautterance linguistic information of the current utterance only. This work proposes to use inter-utterance linguistic information to improve the performance of PSP. Multi-level contextual information, which includes both inter-utterance and intrautterance linguistic information, is extracted by a hierarchical encoder from character level, utterance level and discourse level of the input text. Then a multi-task learning (MTL) decoder predicts prosodic boundaries from multi-level contextual information. Objective evaluation results on two datasets show that our method achieves better F1 scores in predicting prosodic word (PW), prosodic phrase (PPH) and intonational phrase (IPH). It demo
    
[^20]: 论文摘要——论述自然语言处理模型中的偏见对仇恨言论检测的影响

    Thesis Distillation: Investigating The Impact of Bias in NLP Models on Hate Speech Detection. (arXiv:2308.16549v1 [cs.CL])

    [http://arxiv.org/abs/2308.16549](http://arxiv.org/abs/2308.16549)

    本文总结了研究自然语言处理模型中偏见对仇恨言论检测的影响的博士论文。研究发现，偏见对检测任务的影响包括可解释性、冒犯性刻板印象和公平性三个方面。为了有效解决当前在测量和减轻偏见方面的限制，需要将社会科学纳入到研究中。

    

    本文总结了我的博士论文工作。在这篇论文中，我从可解释性、冒犯性刻板印象和公平性三个方面探讨了自然语言处理模型中偏见对仇恨言论检测任务的影响。我讨论了论文的主要要点以及它们对更广泛的自然语言处理社区的益处。最后，我讨论了重要的未来研究方向。我的论文研究结果表明，自然语言处理模型中的偏见从这三个方面影响了仇恨言论检测任务。除非我们开始将社会科学纳入到研究自然语言处理模型中的偏见中，否则我们将无法有效地克服目前在测量和减轻自然语言处理模型偏见方面的局限性。

    This paper is a summary of the work in my PhD thesis. In which, I investigate the impact of bias in NLP models on the task of hate speech detection from three perspectives: explainability, offensive stereotyping bias, and fairness. I discuss the main takeaways from my thesis and how they can benefit the broader NLP community. Finally, I discuss important future research directions. The findings of my thesis suggest that bias in NLP models impacts the task of hate speech detection from all three perspectives. And that unless we start incorporating social sciences in studying bias in NLP models, we will not effectively overcome the current limitations of measuring and mitigating bias in NLP models.
    
[^21]: 针对语音信号准确追踪共振峰的时变准封闭相位分析方法

    Time-Varying Quasi-Closed-Phase Analysis for Accurate Formant Tracking in Speech Signals. (arXiv:2308.16540v1 [eess.AS])

    [http://arxiv.org/abs/2308.16540](http://arxiv.org/abs/2308.16540)

    本文提出了一种新的方法，利用时变准封闭相位分析进行语音信号共振峰的准确估计和追踪，通过将估计和追踪两个阶段合并为一个单一阶段，提高了共振峰估计和追踪的准确性。

    

    本文提出了一种利用时变准封闭相位（TVQCP）分析进行语音信号共振峰的准确估计和追踪的新方法。传统的共振峰追踪方法通常采用两阶段的估计和追踪策略，首先利用短时分析（例如10-50毫秒）估计得到初步的共振峰候选集，然后根据动态规划或线性状态空间模型进行追踪。这些方法的主要缺点之一是，无论追踪阶段如何优秀，都无法提高第一阶段的共振峰估计准确度。所提出的TVQCP方法将估计和追踪两个阶段合并为一个单一阶段的共振峰追踪方法。TVQCP分析结合了三种方法来提高共振峰估计和追踪的准确性：（1）利用时域加权的准封闭相位分析，以减少激励源的干扰。

    In this paper, we propose a new method for the accurate estimation and tracking of formants in speech signals using time-varying quasi-closed-phase (TVQCP) analysis. Conventional formant tracking methods typically adopt a two-stage estimate-and-track strategy wherein an initial set of formant candidates are estimated using short-time analysis (e.g., 10--50 ms), followed by a tracking stage based on dynamic programming or a linear state-space model. One of the main disadvantages of these approaches is that the tracking stage, however good it may be, cannot improve upon the formant estimation accuracy of the first stage. The proposed TVQCP method provides a single-stage formant tracking that combines the estimation and tracking stages into one. TVQCP analysis combines three approaches to improve formant estimation and tracking: (1) it uses temporally weighted quasi-closed-phase analysis to derive closed-phase estimates of the vocal tract with reduced interference from the excitation sour
    
[^22]: The Smart Data Extractor，一种医生友好的解决方案，在临床试验期间加速和改善数据收集

    The Smart Data Extractor, a Clinician Friendly Solution to Accelerate and Improve the Data Collection During Clinical Trials. (arXiv:2308.16537v1 [q-bio.QM])

    [http://arxiv.org/abs/2308.16537](http://arxiv.org/abs/2308.16537)

    本研究提出了一种医生友好的解决方案，即智能数据提取器，通过半自动化的方式来加速和改善临床试验期间的数据收集。与传统的手动数据收集相比，智能数据提取器能够显著减少填写时间，并提供更高质量的数据，避免了人为错误和数据重复输入。

    

    在医学研究中，传统的数据收集方式，即查看病人档案，已经被证明会引起偏倚、错误、人力和成本。我们提出了一种半自动化系统，能够提取各种类型的数据，包括笔记。智能数据提取器通过遵循规则来预填临床研究表格。我们进行了一项交叉测试实验，将半自动化数据收集与手动数据收集进行比较。对于79名患者，需要收集20个目标项目。手动数据收集完成一个表格的平均时间为6分81秒，而智能数据提取器为3分22秒。手动数据收集中的错误数量（整个队列共163个）也比智能数据提取器（整个队列共46个）多。我们提供了一种易于使用、理解和灵活的解决方案，用于填写临床研究表格。它减少了人力投入，提供更高质量的数据，避免了数据重复输入和疲劳引起的错误。

    In medical research, the traditional way to collect data, i.e. browsing patient files, has been proven to induce bias, errors, human labor and costs. We propose a semi-automated system able to extract every type of data, including notes. The Smart Data Extractor pre-populates clinic research forms by following rules. We performed a cross-testing experiment to compare semi-automated to manual data collection. 20 target items had to be collected for 79 patients. The average time to complete one form was 6'81'' for manual data collection and 3'22'' with the Smart Data Extractor. There were also more mistakes during manual data collection (163 for the whole cohort) than with the Smart Data Extractor (46 for the whole cohort). We present an easy to use, understandable and agile solution to fill out clinical research forms. It reduces human effort and provides higher quality data, avoiding data re-entry and fatigue induced errors.
    
[^23]: 广义Winograd Schema及其上下文性

    Generalised Winograd Schema and its Contextuality. (arXiv:2308.16498v1 [cs.CL])

    [http://arxiv.org/abs/2308.16498](http://arxiv.org/abs/2308.16498)

    本文研究了广义Winograd Schema在上下文性方面的应用，提出了一种利用量子物理实验模型来解决Winograd模式挑战的方法。

    

    自然语言中的歧义会引起对解释的概率分布。这些分布通常涉及到多个模棱两可的词汇，这使得它们成为适合量子上下文性拟设模型的研究主题。以前的研究表明，上下文性的不同定量度量与心理语言学研究中的词义歧义有很好的相关性。在本研究中，我们关注指代的歧义，并研究了Winograd模式挑战（WSC），这是Levesque在2011年提出的用于评估机器智能的测试。WSC包含一系列多项选择问题，需要在按照Winograd模式构造的句子中消除代词的歧义，这对机器来说很难确定正确的代词指向，但对人类理解来说却很直观。在本研究中，我们提出了一种类似地将Winograd模式建模为量子物理实验的方法。

    Ambiguities in natural language give rise to probability distributions over interpretations. The distributions are often over multiple ambiguous words at a time; a multiplicity which makes them a suitable topic for sheaf-theoretic models of quantum contextuality. Previous research showed that different quantitative measures of contextuality correlate well with Psycholinguistic research on lexical ambiguities. In this work, we focus on coreference ambiguities and investigate the Winograd Schema Challenge (WSC), a test proposed by Levesque in 2011 to evaluate the intelligence of machines. The WSC consists of a collection of multiple-choice questions that require disambiguating pronouns in sentences structured according to the Winograd schema, in a way that makes it difficult for machines to determine the correct referents but remains intuitive for human comprehension. In this study, we propose an approach that analogously models the Winograd schema as an experiment in quantum physics. Ho
    
[^24]: Transformer压缩通过子空间投影

    Transformer Compression via Subspace Projection. (arXiv:2308.16475v1 [cs.CL])

    [http://arxiv.org/abs/2308.16475](http://arxiv.org/abs/2308.16475)

    Transformer压缩通过子空间投影，在减小模型隐藏大小的同时实现了较大的模型参数和计算资源的减少，并且与其他方法兼容。

    

    我们提出了一种名为TCSP的新方法，用于通过减少模型的隐藏大小来压缩Transformer模型。通过将整个转换模型投影到一个子空间中，我们使模型中的权重矩阵与减小维度空间中的特征之间可以进行矩阵操作，从而显著减少了模型参数和计算资源。为了建立这个子空间，我们将来自不同层次的采样数据实例的特征矩阵分解为一个投影矩阵。为了评估效果，我们在GLUE和SQuAD基准测试上应用TCSP来压缩T5和BERT模型。实验结果表明，TCSP在保证最多1.6%的准确度降低的情况下实现了44%的压缩比，超过或者达到了先前的压缩方法。此外，TCSP还与其他目标过滤器和注意力头大小压缩的方法相兼容。

    We propose TCSP, a novel method for compressing a transformer model by focusing on reducing the hidden size of the model. By projecting the whole transform model into a subspace, we enable matrix operations between the weight matrices in the model and features in a reduced-dimensional space, leading to significant reductions in model parameters and computing resources. To establish this subspace, we decompose the feature matrix, derived from different layers of sampled data instances, into a projection matrix. For evaluation, TCSP is applied to compress T5 and BERT models on the GLUE and SQuAD benchmarks. Experimental results demonstrate that TCSP achieves a compression ratio of 44\% with at most 1.6\% degradation in accuracy, surpassing or matching prior compression methods. Furthermore, TCSP exhibits compatibility with other methods targeting filter and attention head size compression.
    
[^25]: 提升多模式大型语言模型的子任务性能

    Enhancing Subtask Performance of Multi-modal Large Language Model. (arXiv:2308.16474v1 [cs.CL])

    [http://arxiv.org/abs/2308.16474](http://arxiv.org/abs/2308.16474)

    该论文提出了一种方法，通过选择多个预训练模型来完成相同的子任务，通过组合多个模型的结果获得最佳的子任务结果。

    

    多模式大型语言模型（MLLM）是指从大型语言模型（LLM）扩展而来的模型，具备处理和推理多模式数据的能力。当前的MLLM通常通过使用LLM将任务分解为多个子任务，然后使用各个预训练模型完成特定的子任务，并最终利用LLM整合每个子任务的结果来获得任务的结果。在现实世界中，处理大型项目时，常常将项目分解为较小的子项目，并由不同的团队提供相应的解决方案或结果。项目所有者随后决定使用哪个解决方案或结果，以确保每个子任务和整个项目能够达到最佳结果。受此启发，本研究考虑选择多个预训练模型来完成相同的子任务。通过将多个预训练模型的结果进行组合，获得最佳的子任务结果。

    Multi-modal Large Language Model (MLLM) refers to a model expanded from a Large Language Model (LLM) that possesses the capability to handle and infer multi-modal data. Current MLLMs typically begin by using LLMs to decompose tasks into multiple subtasks, then employing individual pre-trained models to complete specific subtasks, and ultimately utilizing LLMs to integrate the results of each subtasks to obtain the results of the task. In real-world scenarios, when dealing with large projects, it is common practice to break down the project into smaller sub-projects, with different teams providing corresponding solutions or results. The project owner then decides which solution or result to use, ensuring the best possible outcome for each subtask and, consequently, for the entire project. Inspired by this, this study considers selecting multiple pre-trained models to complete the same subtask. By combining the results from multiple pre-trained models, the optimal subtask result is obtai
    
[^26]: 将维基百科文章的链接预测任务作为自然语言推理任务

    Link Prediction for Wikipedia Articles as a Natural Language Inference Task. (arXiv:2308.16469v1 [cs.CL])

    [http://arxiv.org/abs/2308.16469](http://arxiv.org/abs/2308.16469)

    本文将维基百科文章的链接预测任务视为自然语言推理任务，采用了一种新的方法，并在DSAA-2023竞赛中取得了较高的评分。

    

    链接预测任务对于自动理解大型知识库的结构至关重要。本文介绍了我们在数据科学和高级分析2023年竞赛“高效和有效的链接预测”（DSAA-2023竞赛）中用包含948,233个训练样本和238,265个用于公共测试的语料库解决这一任务的系统。本文引入了一种将维基百科文章的链接预测问题建模为自然语言推理任务（NLI）的方法。受到近期自然语言处理和理解方面的进展启发，我们将链接预测作为一个NLI任务，其中将两个文章之间的链接存在视为前提，任务是基于文章中呈现的信息来确定该前提是否成立。我们的系统是基于用于维基百科文章链接预测的句对分类的实现。我们的系统实现了0.99996的Macro F1-score和1.00000的Macro F1-score。

    Link prediction task is vital to automatically understanding the structure of large knowledge bases. In this paper, we present our system to solve this task at the Data Science and Advanced Analytics 2023 Competition "Efficient and Effective Link Prediction" (DSAA-2023 Competition) with a corpus containing 948,233 training and 238,265 for public testing. This paper introduces an approach to link prediction in Wikipedia articles by formulating it as a natural language inference (NLI) task. Drawing inspiration from recent advancements in natural language processing and understanding, we cast link prediction as an NLI task, wherein the presence of a link between two articles is treated as a premise, and the task is to determine whether this premise holds based on the information presented in the articles. We implemented our system based on the Sentence Pair Classification for Link Prediction for the Wikipedia Articles task. Our system achieved 0.99996 Macro F1-score and 1.00000 Macro F1-s
    
[^27]: Sparkles: 解锁多图聊天以实现多模态指令跟踪模型

    Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models. (arXiv:2308.16463v1 [cs.CV])

    [http://arxiv.org/abs/2308.16463](http://arxiv.org/abs/2308.16463)

    Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。

    

    当使用指令跟踪数据来进行微调时，大型语言模型在各种任务上展现出了强大的零-shot性能。多模态指令跟踪模型通过整合文本和图像进一步扩展了这些能力。然而，现有的模型（如MiniGPT-4）在涉及多个图像的情况下保持对话连贯性面临挑战。一个主要原因是缺乏一个专门针对这一关键应用的数据集。为了弥合这些差距，我们提出了SparklesChat，一个用于多图对话的多模态指令跟踪模型。为了支持训练，我们引入了SparklesDialogue，这是第一个专为单词级交错多图像和文本交互而定制的机器生成对话数据集。此外，我们构建了SparklesEval，一个借助GPT辅助的基准，用于定量评估模型在多个图像和对话轮次中的对话能力。我们的实验验证了SparklesChat在理解多图对话方面的有效性。

    Large language models exhibit enhanced zero-shot performance on various tasks when fine-tuned with instruction-following data. Multimodal instruction-following models extend these capabilities by integrating both text and images. However, existing models such as MiniGPT-4 face challenges in maintaining dialogue coherence in scenarios involving multiple images. A primary reason is the lack of a specialized dataset for this critical application. To bridge these gaps, we present SparklesChat, a multimodal instruction-following model for open-ended dialogues across multiple images. To support the training, we introduce SparklesDialogue, the first machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions. Furthermore, we construct SparklesEval, a GPT-assisted benchmark for quantitatively assessing a model's conversational competence across multiple images and dialogue turns. Our experiments validate the effectiveness of SparklesChat in understa
    
[^28]: BioCoder: 一种带有上下文语用知识的生物信息学代码生成基准

    BioCoder: A Benchmark for Bioinformatics Code Generation with Contextual Pragmatic Knowledge. (arXiv:2308.16458v1 [cs.LG])

    [http://arxiv.org/abs/2308.16458](http://arxiv.org/abs/2308.16458)

    BioCoder是一个用于评估预训练模型在生成生物信息学代码方面的基准，涵盖了函数代码生成中的包依赖关系、类声明和全局变量，并通过模糊测试框架进行评估。

    

    预训练的语言模型（如ChatGPT）显著改进了代码生成。随着这些模型的扩大，需要输出来处理更复杂的任务的需求也越来越多。此外，在生物信息学中，生成功能程序由于领域知识量大、需要复杂的数据操作和复杂的功能依赖关系而面临额外的挑战。在这里，我们介绍了BioCoder，这是一个用于评估现有预训练模型在生成生物信息学代码方面的基准。与函数代码生成有关，BioCoder涵盖了可能的包依赖关系、类声明和全局变量。它包括来自GitHub的1026个Python和Java函数和1243个方法，以及来自Rosalind项目的253个示例。BioCoder还结合了一个用于评估的模糊测试框架，我们已经应用它来评估许多模型，包括InCoder、CodeGen、CodeGen2、SantaCoder、StarCoder、StarCoder+、InstructCodeT。

    Pre-trained language models like ChatGPT have significantly improved code generation. As these models scale up, there is an increasing need for the output to handle more intricate tasks. Moreover, in bioinformatics, generating functional programs poses additional notable challenges due to the amount of domain knowledge, the need for complicated data operations, and intricate functional dependencies between the operations. Here, we present BioCoder, a benchmark developed to evaluate existing pre-trained models in generating bioinformatics code. In relation to function-code generation, BioCoder covers potential package dependencies, class declarations, and global variables. It incorporates 1026 functions and 1243 methods in Python and Java from GitHub and 253 examples from the Rosalind Project. BioCoder incorporates a fuzz-testing framework for evaluation, and we have applied it to evaluate many models including InCoder, CodeGen, CodeGen2, SantaCoder, StarCoder, StarCoder+, InstructCodeT
    
[^29]: 从非流式到流式ASR编码器的知识蒸馏，使用辅助的非流式层

    Knowledge Distillation from Non-streaming to Streaming ASR Encoder using Auxiliary Non-streaming Layer. (arXiv:2308.16415v1 [cs.CL])

    [http://arxiv.org/abs/2308.16415](http://arxiv.org/abs/2308.16415)

    本文提出了一种从非流式到流式ASR编码器的知识蒸馏方法，通过逐层蒸馏和引入辅助的非流式层，以及特定的蒸馏损失函数设计，显著降低了流式ASR的词错误率。

    

    流式自动语音识别（ASR）模型由于无法访问未来的上下文，导致性能比非流式模型差。为了提高流式ASR的性能，已经研究了从非流式到流式模型的知识蒸馏，主要关注输出标记概率的对齐。本文提出了一种从教师编码器到学生编码器的逐层知识蒸馏方法。为了确保使用相同的上下文进行特征提取，我们在学生模型中插入辅助的非流式分支，并从非流式教师层向非流式辅助层进行知识蒸馏。我们设计了一种特殊的蒸馏损失，利用自回归预测编码（APC）机制，鼓励流式模型预测看不见的未来上下文。实验结果表明，与之前的标记概率蒸馏方法相比，所提出的方法可以显著降低词错误率。

    Streaming automatic speech recognition (ASR) models are restricted from accessing future context, which results in worse performance compared to the non-streaming models. To improve the performance of streaming ASR, knowledge distillation (KD) from the non-streaming to streaming model has been studied, mainly focusing on aligning the output token probabilities. In this paper, we propose a layer-to-layer KD from the teacher encoder to the student encoder. To ensure that features are extracted using the same context, we insert auxiliary non-streaming branches to the student and perform KD from the non-streaming teacher layer to the non-streaming auxiliary layer. We design a special KD loss that leverages the autoregressive predictive coding (APC) mechanism to encourage the streaming model to predict unseen future contexts. Experimental results show that the proposed method can significantly reduce the word error rate compared to previous token probability distillation methods.
    
[^30]: 情感视觉对话：基于视觉对话理解情感形成的大规模基准

    Affective Visual Dialog: A Large-Scale Benchmark for Emotional Reasoning Based on Visually Grounded Conversations. (arXiv:2308.16349v1 [cs.CL])

    [http://arxiv.org/abs/2308.16349](http://arxiv.org/abs/2308.16349)

    我们引入了一个名为AffectVisDial的大规模数据集，其中包含50,000个基于视觉的对话，我们训练了情感视觉对话模型来解决基于对话的问答、情感预测和情感解释任务，展示出了有希望的情感推理能力。

    

    我们引入了情感视觉对话，作为一个测试平台，用于研究理解在基于视觉对话中情感形成的过程。这项任务涉及三项技能：（1）基于对话的问答，（2）基于对话的情感预测，以及（3）基于对话生成情感解释。我们的主要贡献是构建了一个大规模数据集，称为AffectVisDial，包含50,000个10轮的基于视觉的对话，还包括总结的情感归因和基于对话的情感解释，总共需要27180个工作小时。我们解释了收集该数据集的设计决策，并介绍了与对话参与者相关的提问者和回答者任务。我们训练和展示了来自最先进模型的坚实的情感视觉对话基线。值得注意的是，我们模型生成的回答显示出有希望的情感推理能力。

    We introduce Affective Visual Dialog, an emotion explanation and reasoning task as a testbed for research on understanding the formation of emotions in visually grounded conversations. The task involves three skills: (1) Dialog-based Question Answering (2) Dialog-based Emotion Prediction and (3) Affective emotion explanation generation based on the dialog. Our key contribution is the collection of a large-scale dataset, dubbed AffectVisDial, consisting of 50K 10-turn visually grounded dialogs as well as concluding emotion attributions and dialog-informed textual emotion explanations, resulting in a total of 27,180 working hours. We explain our design decisions in collecting the dataset and introduce the questioner and answerer tasks that are associated with the participants in the conversation. We train and demonstrate solid Affective Visual Dialog baselines adapted from state-of-the-art models. Remarkably, the responses generated by our models show promising emotional reasoning abilit
    
[^31]: ToddlerBERTa: 利用BabyBERTa进行语法学习和语言理解

    ToddlerBERTa: Exploiting BabyBERTa for Grammar Learning and Language Understanding. (arXiv:2308.16336v1 [cs.CL])

    [http://arxiv.org/abs/2308.16336](http://arxiv.org/abs/2308.16336)

    ToddlerBERTa是一个类似于BabyBERTa的语言模型，尽管在较小的数据集上进行训练，但它展示了令人称赞的性能，并具有强大的语言理解能力，与最先进的RoBERTa-base相媲美。

    

    我们提出了ToddlerBERTa，这是一个类似于BabyBERTa的语言模型，并通过五种不同的具有不同超参数的模型来探索其能力。在BLiMP，SuperGLUE，MSGS和BabyLM挑战中进行评估，我们发现较小的模型在特定任务上表现出色，而较大的模型在大量数据方面表现良好。尽管在较小的数据集上训练，ToddlerBERTa展示了令人称赞的性能，与最先进的RoBERTa-base相媲美。该模型展示了强大的语言理解能力，即使是在单句预训练的情况下，也能与利用更广泛上下文信息的基线竞争。我们的工作为超参数选择和数据利用提供了洞察，并为语言模型的发展做出了贡献。

    We present ToddlerBERTa, a BabyBERTa-like language model, exploring its capabilities through five different models with varied hyperparameters. Evaluating on BLiMP, SuperGLUE, MSGS, and a Supplement benchmark from the BabyLM challenge, we find that smaller models can excel in specific tasks, while larger models perform well with substantial data. Despite training on a smaller dataset, ToddlerBERTa demonstrates commendable performance, rivalling the state-of-the-art RoBERTa-base. The model showcases robust language understanding, even with single-sentence pretraining, and competes with baselines that leverage broader contextual information. Our work provides insights into hyperparameter choices, and data utilization, contributing to the advancement of language models.
    
[^32]: 探索大型语言模型用于知识图谱补全

    Exploring Large Language Models for Knowledge Graph Completion. (arXiv:2308.13916v1 [cs.CL])

    [http://arxiv.org/abs/2308.13916](http://arxiv.org/abs/2308.13916)

    本文研究了利用大型语言模型（LLM）进行知识图谱补全的方法，并引入了一种创新的框架（知识图谱LLM），以提高三元组分类和关系预测的性能。

    

    知识图谱在众多人工智能任务中发挥着重要作用，但经常面临不完整性的问题。在本研究中，我们探索了利用大型语言模型（LLM）进行知识图谱补全的方法。我们将知识图谱中的三元组视为文本序列，并引入了一种创新的框架，称为知识图谱LLM（KG-LLM），来对这些三元组进行建模。我们的技术利用三元组的实体和关系描述作为提示，并利用响应进行预测。对各种基准知识图谱的实验表明，我们的方法在三元组分类和关系预测等任务中达到了最先进的性能。我们还发现，微调相对较小的模型（例如LLaMA-7B，ChatGLM-6B）优于最新的ChatGPT和GPT-4。

    Knowledge graphs play a vital role in numerous artificial intelligence tasks, yet they frequently face the issue of incompleteness. In this study, we explore utilizing Large Language Models (LLM) for knowledge graph completion. We consider triples in knowledge graphs as text sequences and introduce an innovative framework called Knowledge Graph LLM (KG-LLM) to model these triples. Our technique employs entity and relation descriptions of a triple as prompts and utilizes the response for predictions. Experiments on various benchmark knowledge graphs demonstrate that our method attains state-of-the-art performance in tasks such as triple classification and relation prediction. We also find that fine-tuning relatively smaller models (e.g., LLaMA-7B, ChatGLM-6B) outperforms recent ChatGPT and GPT-4.
    
[^33]: DocPrompt: 大规模连续预训练用于零样本和少样本文档问答

    DocPrompt: Large-scale continue pretrain for zero-shot and few-shot document question answering. (arXiv:2308.10959v1 [cs.CL])

    [http://arxiv.org/abs/2308.10959](http://arxiv.org/abs/2308.10959)

    本文提出了一个名为DocPrompt的方法，用于处理文档问答任务，具有强大的零样本和少样本性能。实验结果表明，DocPrompt模型经过连续预训练后在文档问答任务中表现优异，大大提高了交付效率和模型性能，降低了注释成本和劳动成本。

    

    本文提出了一个名为DocPrompt的方法，用于处理文档问答任务，具有强大的零样本和少样本性能。我们提出了一种新颖的弱监督数据生成方法、一种新颖的多阶段训练方法，以及一种新颖的理解模型和生成模型集成方法。实验结果表明，在文档问答任务中，经过连续预训练的DocPrompt模型明显优于现有的强基线模型。这种方法极大地提高了文档问答客户项目的交付效率和模型性能，降低了注释成本和劳动成本。我们的演示可以在https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout找到。

    In this paper, we propose Docprompt for document question answering tasks with powerful zero-shot and few-shot performance. We proposed a novel weakly supervised data generation method, a novel multl-stage training method and a novel understanding model & generation model ensemble method. Experiment results show that the Docprompt model after continue pretrain significantly outperforms the existing strong baseline models on document question answering tasks. This method greatly improves the delivery efficiency and model performance of document question answering customer projects, reducing annotation costs and labor costs. Our demo can be found at https://huggingface.co/spaces/PaddlePaddle/ERNIE-Layout.
    
[^34]: 玩弄文字：比较ChatGPT和人类的词汇和词汇丰富度

    Playing with Words: Comparing the Vocabulary and Lexical Richness of ChatGPT and Humans. (arXiv:2308.07462v1 [cs.CL])

    [http://arxiv.org/abs/2308.07462](http://arxiv.org/abs/2308.07462)

    这篇论文比较了ChatGPT和人类在词汇和词汇丰富度方面的差异，研究发现使用ChatGPT等工具会对词汇使用和词汇丰富度产生影响，这可能会对语言演变产生影响。

    

    人工智能生成语言模型（如GPT）和ChatGPT等工具的引入引发了一场革命，可以改变文本生成的方式。这对读者的语言能力以及新型人工智能工具的培训是否会产生影响具有许多含义？它是否会影响语言的演变？我们关注语言的一个特定方面：词语；在编写给定文本时，使用ChatGPT等工具会增加或减少使用的词汇量或词汇丰富度（理解为书面或口头表达中使用的不同词汇数量）？这对词语有影响，因为未包含在人工智能生成的内容中的词语往往会变得越来越不受欢迎，并最终可能消失。在这项工作中，我们对ChatGPT和人类的词汇和词汇丰富度进行了初步比较。

    The introduction of Artificial Intelligence (AI) generative language models such as GPT (Generative Pre-trained Transformer) and tools such as ChatGPT has triggered a revolution that can transform how text is generated. This has many implications, for example, as AI-generated text becomes a significant fraction of the text in many disciplines, would this have an effect on the language capabilities of readers and also on the training of newer AI tools? Would it affect the evolution of languages? Focusing on one specific aspect of the language: words; will the use of tools such as ChatGPT increase or reduce the vocabulary used or the lexical richness (understood as the number of different words used in a written or oral production) when writing a given text? This has implications for words, as those not included in AI-generated content will tend to be less and less popular and may eventually be lost. In this work, we perform an initial comparison of the vocabulary and lexical richness of
    
[^35]: Sensi-BERT: 面向敏感度驱动的参数高效BERT微调

    Sensi-BERT: Towards Sensitivity Driven Fine-Tuning for Parameter-Efficient BERT. (arXiv:2307.11764v1 [cs.CL])

    [http://arxiv.org/abs/2307.11764](http://arxiv.org/abs/2307.11764)

    Sensi-BERT是一种面向敏感度驱动的参数高效BERT微调方法，通过敏感度分析和裁剪参数张量，可生成适用于下游任务的高度参数高效的模型。

    

    近年来，由于在文本分类和问答等各种下游任务上的改进表现，大型预训练语言模型逐渐受到关注，只需进行很少次数的微调。然而，其庞大的模型大小常常限制了它们在资源受限的边缘设备上的应用。现有的参数高效BERT模型解决方案大多依赖于计算密集的训练和微调，并且常常依赖于额外的计算密集型模型来弥补性能差距。本文介绍了Sensi-BERT，一种敏感度驱动的BERT模型高效微调方法，可以使用现成的预训练BERT模型，生成适用于下游任务的高度参数高效的模型。具体而言，我们进行敏感度分析以对每个单独的参数张量进行排序，然后在微调过程中根据给定的参数或FLOPs预算进行相应的裁剪。实验结果表明Sensi-BERT的有效性。

    Large pre-trained language models have recently gained significant traction due to their improved performance on various down-stream tasks like text classification and question answering, requiring only few epochs of fine-tuning. However, their large model sizes often prohibit their applications on resource-constrained edge devices. Existing solutions of yielding parameter-efficient BERT models largely rely on compute-exhaustive training and fine-tuning. Moreover, they often rely on additional compute heavy models to mitigate the performance gap. In this paper, we present Sensi-BERT, a sensitivity driven efficient fine-tuning of BERT models that can take an off-the-shelf pre-trained BERT model and yield highly parameter-efficient models for downstream tasks. In particular, we perform sensitivity analysis to rank each individual parameter tensor, that then is used to trim them accordingly during fine-tuning for a given parameter or FLOPs budget. Our experiments show the efficacy of Sens
    
[^36]: "感觉像有第二个思维": 探究在大型语言模型中进行创意可写性预写的人机共创

    "It Felt Like Having a Second Mind": Investigating Human-AI Co-creativity in Prewriting with Large Language Models. (arXiv:2307.10811v1 [cs.HC])

    [http://arxiv.org/abs/2307.10811](http://arxiv.org/abs/2307.10811)

    通过三节次的定性研究，探究了人类与大型语言模型在预写过程中的合作模式，并发现了一个三阶段的人机共创过程：构思、启发和实施。在这个合作过程中，人类扮演着主导角色。

    

    预写是在第一稿之前发现和发展思想的过程，它需要发散性思维，通常涉及到无结构的策略，如图表、概述和自由写作等。虽然已经证明大型语言模型（LLMs）在各种任务中都是有用的，包括创意写作，但对用户如何与LLMs合作来支持预写的方式知之甚少。在这种创造性过程中，LLMs的首选合作角色和主动性也不明确。为了研究人类与LLMs在预写过程中的合作模式和动力学，我们进行了一项三节次的定性研究，与15位参与者进行了两个创造性任务：写故事和写口号。研究结果表明，在合作的预写过程中，似乎存在着一个三阶段迭代的人机共创过程，包括构思、启发和实施阶段。这个合作过程以人类在主导角色中取得了成功。

    Prewriting is the process of discovering and developing ideas before a first draft, which requires divergent thinking and often implies unstructured strategies such as diagramming, outlining, free-writing, etc. Although large language models (LLMs) have been demonstrated to be useful for a variety of tasks including creative writing, little is known about how users would collaborate with LLMs to support prewriting. The preferred collaborative role and initiative of LLMs during such a creativity process is also unclear. To investigate human-LLM collaboration patterns and dynamics during prewriting, we conducted a three-session qualitative study with 15 participants in two creative tasks: story writing and slogan writing. The findings indicated that during collaborative prewriting, there appears to be a three-stage iterative Human-AI Co-creativity process that includes Ideation, Illumination, and Implementation stages. This collaborative process champions the human in a dominant role, in
    
[^37]: 多模态讨论变换器：整合文本、图像和图变换器以检测社交媒体上的仇恨言论。

    Multi-Modal Discussion Transformer: Integrating Text, Images and Graph Transformers to Detect Hate Speech on Social Media. (arXiv:2307.09312v1 [cs.CL])

    [http://arxiv.org/abs/2307.09312](http://arxiv.org/abs/2307.09312)

    多模态讨论变换器 (mDT) 是一个用于检测在线社交网络中仇恨言论的新颖模型。与传统的仅使用文本的方法不同，mDT通过整体分析文本和图像，结合图变换器捕捉评论周围整个讨论的上下文关系，并通过交织融合层将文本和图像嵌入进行组合。研究发现，捕捉对话的整体视图可以极大地提高检测反社会行为的准确性。

    

    我们提出了一种新颖的多模态基于图的变换器模型，名为多模态讨论变换器（mDT），用于检测在线社交网络中的仇恨言论。与传统的仅使用文本的方法不同，我们将标记评论为仇恨言论的方法围绕文本和图像的整体分析展开。这是通过利用图变换器来捕捉评论周围整个讨论中的上下文关系，并采用交织融合层来组合文本和图像嵌入，而不是单独处理不同的模态。我们将模型的性能与仅处理文本的基线进行比较，还进行了广泛的消融研究。最后，我们展望了多模态解决方案在在线环境中提供社会价值的未来工作，并认为捕捉对话的整体视图极大地推进了检测反社会行为的努力。

    We present the Multi-Modal Discussion Transformer (mDT), a novel multi-modal graph-based transformer model for detecting hate speech in online social networks. In contrast to traditional text-only methods, our approach to labelling a comment as hate speech centers around the holistic analysis of text and images. This is done by leveraging graph transformers to capture the contextual relationships in the entire discussion that surrounds a comment, with interwoven fusion layers to combine text and image embeddings instead of processing different modalities separately. We compare the performance of our model to baselines that only process text; we also conduct extensive ablation studies. We conclude with future work for multimodal solutions to deliver social value in online contexts, arguing that capturing a holistic view of a conversation greatly advances the effort to detect anti-social behavior.
    
[^38]: CARE-MI: 中国孕婴护理领域的虚假信息评估基准

    CARE-MI: Chinese Benchmark for Misinformation Evaluation in Maternity and Infant Care. (arXiv:2307.01458v1 [cs.CL])

    [http://arxiv.org/abs/2307.01458](http://arxiv.org/abs/2307.01458)

    CARE-MI是一个用于评估中国孕婴护理领域LLM虚假信息的基准，填补了这一领域的研究空白，并提供了构建长篇生成评估基准的创新范式。

    

    最近自然语言处理的进展导致了将LLM应用于现实场景的新趋势。尽管最新的LLM在与人类互动时令人惊叹地流利，但它们在生成错误事实陈述时会意外产生虚假信息问题。这可能导致有害后果，尤其是在敏感环境下，比如医疗保健领域。然而，之前很少有研究关注评估LLM长篇生成中的虚假信息，尤其是针对知识密集型主题。此外，尽管LLM在不同语言上表现良好，但虚假信息评估主要在英语中进行。为此，我们提供了一个基准，CARE-MI，用于评估LLM虚假信息在：1）一个敏感主题，具体是孕婴护理领域；和2）一种非英语语言，即中文。最重要的是，我们提供了一个创新的范式，用于构建长篇生成评估基准，可以

    The recent advances in NLP, have led to a new trend of applying LLMs to real-world scenarios. While the latest LLMs are astonishingly fluent when interacting with humans, they suffer from the misinformation problem by unintentionally generating factually false statements. This can lead to harmful consequences, especially when produced within sensitive contexts, such as healthcare. Yet few previous works have focused on evaluating misinformation in the long-form generation of LLMs, especially for knowledge-intensive topics. Moreover, although LLMs have been shown to perform well in different languages, misinformation evaluation has been mostly conducted in English. To this end, we present a benchmark, CARE-MI, for evaluating LLM misinformation in: 1) a sensitive topic, specifically the maternity and infant care domain; and 2) a language other than English, namely Chinese. Most importantly, we provide an innovative paradigm for building long-form generation evaluation benchmarks that can
    
[^39]: 使用语法演化自动设计语义相似性集合

    Automatic Design of Semantic Similarity Ensembles Using Grammatical Evolution. (arXiv:2307.00925v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.00925](http://arxiv.org/abs/2307.00925)

    本研究首次使用语法演化自动设计语义相似性集合，通过自动选择和聚合候选度量来优化集合与人类判断的相关性，提高相似度评估准确性，并证明了使用集合对语义相似性任务的益处。

    

    语义相似性度量在自然语言处理中被广泛应用于多种与计算机相关的任务。然而，没有单一的语义相似性度量适用于所有任务，研究人员经常使用集合策略来确保性能。本研究提出了一种自动设计语义相似性集合的方法。事实上，我们提出的方法首次使用语法演化来自动选择和聚合一组候选度量，以创建一个最大化与人类判断相关性的集合。该方法在多个基准数据集上进行了评估，并与最先进的集合进行了比较，结果显示它可以显著提高相似度评估的准确性，并在某些情况下优于现有方法。因此，我们的研究既展示了使用语法演化来自动比较文本的潜力，也证明了使用集合对语义相似性任务的益处。

    Semantic similarity measures are widely used in natural language processing to catalyze various computer-related tasks. However, no single semantic similarity measure is the most appropriate for all tasks, and researchers often use ensemble strategies to ensure performance. This research work proposes a method for automatically designing semantic similarity ensembles. In fact, our proposed method uses grammatical evolution, for the first time, to automatically select and aggregate measures from a pool of candidates to create an ensemble that maximizes correlation to human judgment. The method is evaluated on several benchmark datasets and compared to state-of-the-art ensembles, showing that it can significantly improve similarity assessment accuracy and outperform existing methods in some cases. As a result, our research demonstrates the potential of using grammatical evolution to automatically compare text and prove the benefits of using ensembles for semantic similarity tasks. The so
    
[^40]: C-PMI: 条件点对点互信息用于对话评估的方法研究

    C-PMI: Conditional Pointwise Mutual Information for Turn-level Dialogue Evaluation. (arXiv:2306.15245v1 [cs.CL])

    [http://arxiv.org/abs/2306.15245](http://arxiv.org/abs/2306.15245)

    本研究提出了一种基于条件点对点互信息的模型-无关方法，用于衡量对话系统与用户之间的交互，通过替换评分器，显著改进了与人类判断的相关性。

    

    现有的chatbot的无参考级对话评估指标不足以捕捉用户与系统之间的交互。因此，它们通常与人类评估的相关性较差。为解决这一问题，我们提出了一种新颖的模型无关方法，利用条件点对点互信息（C-PMI）来度量系统和用户之间基于给定评估维度的对话交互。在广泛使用的FED对话评估数据集上的实验结果表明，与现有评估系统相比，我们的方法显著提高了与人类判断的相关性。通过将基于负对数似然的评分器替换为我们提出的C-PMI评分器，我们在FED评估指标上的Spearman相关性平均相对提高了60.5%。我们的代码公开发布在https://github.com/renll/C-PMI。

    Existing reference-free turn-level evaluation metrics for chatbots inadequately capture the interaction between the user and the system. Consequently, they often correlate poorly with human evaluations. To address this issue, we propose a novel model-agnostic approach that leverages Conditional Pointwise Mutual Information (C-PMI) to measure the turn-level interaction between the system and the user based on a given evaluation dimension. Experimental results on the widely used FED dialogue evaluation dataset demonstrate that our approach significantly improves the correlation with human judgment compared with existing evaluation systems. By replacing the negative log-likelihood-based scorer with our proposed C-PMI scorer, we achieve a relative 60.5% higher Spearman correlation on average for the FED evaluation metric. Our code is publicly available at https://github.com/renll/C-PMI.
    
[^41]: 利用预训练语言模型、嵌入蒸馏和上采样策略改善非自回归翻译质量（arXiv:2306.06345v1 [cs.CL]）

    Improving Non-autoregressive Translation Quality with Pretrained Language Model, Embedding Distillation and Upsampling Strategy for CTC. (arXiv:2306.06345v1 [cs.CL])

    [http://arxiv.org/abs/2306.06345](http://arxiv.org/abs/2306.06345)

    本文提出了一些技术来提高非自回归翻译模型的翻译质量，在保持显着推理速度加速的同时，通过使用预训练多语言模型进行微调、采用MASK插入方案进行上采样、以及采用嵌入蒸馏方法来进一步提高性能。在多个数据集上，模型表现优于基线自回归模型。

    

    非自回归方法旨在提高翻译模型的推理速度，特别是那些可以一次正向传递生成输出的模型。但是，与自回归模型相比，这些方法往往在翻译质量上有显著的下降。本文引入了一系列创新技术，以提高非自回归翻译模型的翻译质量，同时保持推理速度的显著加速。我们建议使用CTC损失微调预训练多语言模型来有效地训练NAT模型。此外，我们采用MASK插入方案进行上采样，而不是令牌复制，并提出了一种嵌入蒸馏方法以进一步提高性能。在我们的实验中，我们的模型在多个数据集上优于基线自回归模型（Transformer base），包括WMT'14 DE $\leftrightarrow$ EN、WMT'16 RO $\leftrightarrow$ EN和IWSLT'14 DE $\leftrightarrow$ EN。

    Non-autoregressive approaches aim to improve the inference speed of translation models, particularly those that generate output in a one-pass forward manner. However, these approaches often suffer from a significant drop in translation quality compared to autoregressive models. This paper introduces a series of innovative techniques to enhance the translation quality of Non-Autoregressive Translation (NAT) models while maintaining a substantial acceleration in inference speed. We propose fine-tuning Pretrained Multilingual Language Models (PMLMs) with the CTC loss to train NAT models effectively. Furthermore, we adopt the MASK insertion scheme for up-sampling instead of token duplication, and we present an embedding distillation method to further enhance performance. In our experiments, our model outperforms the baseline autoregressive model (Transformer \textit{base}) on multiple datasets, including WMT'14 DE$\leftrightarrow$EN, WMT'16 RO$\leftrightarrow$EN, and IWSLT'14 DE$\leftright
    
[^42]: LLM驱动的生成式新闻推荐初探

    A First Look at LLM-Powered Generative News Recommendation. (arXiv:2305.06566v1 [cs.IR])

    [http://arxiv.org/abs/2305.06566](http://arxiv.org/abs/2305.06566)

    本文介绍了一种LLM驱动的生成式新闻推荐框架GENRE，它利用预训练语义知识丰富新闻数据，通过从模型设计转移到提示设计提供灵活而统一的解决方案，实现了个性化新闻生成、用户画像和新闻摘要。

    

    个性化的新闻推荐系统已成为用户浏览海量在线新闻内容所必需的工具，然而现有的新闻推荐系统面临着冷启动问题、用户画像建模和新闻内容理解等重大挑战。先前的研究通常通过模型设计遵循一种不灵活的例行程序来解决特定的挑战，但在理解新闻内容和捕捉用户兴趣方面存在局限性。在本文中，我们介绍了GENRE，一种LLM驱动的生成式新闻推荐框架，它利用来自大型语言模型的预训练语义知识来丰富新闻数据。我们的目标是通过从模型设计转移到提示设计来提供一种灵活而统一的新闻推荐解决方案。我们展示了GENRE在个性化新闻生成、用户画像和新闻摘要中的应用。使用各种流行的推荐模型进行的大量实验证明了GENRE的有效性。

    Personalized news recommendation systems have become essential tools for users to navigate the vast amount of online news content, yet existing news recommenders face significant challenges such as the cold-start problem, user profile modeling, and news content understanding. Previous works have typically followed an inflexible routine to address a particular challenge through model design, but are limited in their ability to understand news content and capture user interests. In this paper, we introduce GENRE, an LLM-powered generative news recommendation framework, which leverages pretrained semantic knowledge from large language models to enrich news data. Our aim is to provide a flexible and unified solution for news recommendation by moving from model design to prompt design. We showcase the use of GENRE for personalized news generation, user profiling, and news summarization. Extensive experiments with various popular recommendation models demonstrate the effectiveness of GENRE. 
    
[^43]: SCOTT: 自我一致性思路串提炼

    SCOTT: Self-Consistent Chain-of-Thought Distillation. (arXiv:2305.01879v1 [cs.CL])

    [http://arxiv.org/abs/2305.01879](http://arxiv.org/abs/2305.01879)

    本研究提出了一种忠实的知识蒸馏方法，从比教师模型大数倍的模型中学习一个小的、自我一致的思路串模型。实验结果表明，该方法有助于证明决策并提高性能，特别是在较小的语言模型中。

    

    超出一定规模的大型语言模型表现出通过一系列连续的思考过程获得自由文本理由的突出能力。虽然思路串可以显著提高性能，但仅在足够大的语言模型中才能观察到这种收益。更令人担忧的是，生成的理由很少保证与语言模型的预测保持一致或者忠实地证明决策。在这项工作中，我们提出了一种忠实的知识蒸馏方法，从比教师模型大数倍的模型中学习一个小的、自我一致的思路串模型。为了形成更好的监督，我们通过对比解码引导教师模型产生支持正确答案的理由，这鼓励教师模型生成的token只在考虑到答案时才更加可信。为了保证忠实的蒸馏，我们使用教师生成的理由来学习一个学生模型，该模型具有反事实推理目标，即根据具有自我一致性且忠实于教师预测的思路串理由预测决策。在自然语言推理和抽象摘要基准测试上，我们证明了学生模型中的自我一致性有助于证明决策并提高性能，特别是在较小的语言模型中。

    Large language models (LMs) beyond a certain scale, demonstrate the emergent capability of generating free-text rationales for their predictions via chain-of-thought (CoT) prompting. While CoT can yield dramatically improved performance, such gains are only observed for sufficiently large LMs. Even more concerning, there is little guarantee that the generated rationales are consistent with LM's predictions or faithfully justify the decisions. In this work, we propose a faithful knowledge distillation method to learn a small, self-consistent CoT model from a teacher model that is orders of magnitude larger. To form better supervision, we elicit rationales supporting the gold answers from a large LM (teacher) by contrastive decoding, which encourages the teacher to generate tokens that become more plausible only when the answer is considered. To ensure faithful distillation, we use the teacher-generated rationales to learn a student LM with a counterfactual reasoning objective, which pre
    
[^44]: OLISIA: 一个用于口语化对话状态跟踪的级联系统

    OLISIA: a Cascade System for Spoken Dialogue State Tracking. (arXiv:2304.11073v1 [eess.AS])

    [http://arxiv.org/abs/2304.11073](http://arxiv.org/abs/2304.11073)

    我们提出了OLISIA，一个口语对话状态跟踪的级联系统，使用自动语音识别和DST模型，采用几个适应性策略来提高稳健性，并在DSTC11 Track3中取得第一名的好成绩。

    

    对话状态跟踪 (DST) 是口语对话系统的核心组成部分。然而，最近关于该任务的研究大多集中于聊天时的语料库，忽略了口语和书面语之间的差异。本文提出了 OLISIA，这是一个级联系统，它集成了自动语音识别 (ASR) 模型和 DST 模型。我们在 ASR 和 DST 模块中引入了几个适应性策略，以提高对口语对话的整合性和稳健性。经过这些策略的调整，我们的系统在 DSTC11 Track 3 中排名第一，这是一个评估口语 DST 性能的基准。我们进行了深入的结果分析，发现规范化 ASR 的输出和通过数据增强调整 DST 的输入，以及增加预训练模型的大小，都在降低书面和口语对话之间性能差异方面发挥了重要作用。

    Though Dialogue State Tracking (DST) is a core component of spoken dialogue systems, recent work on this task mostly deals with chat corpora, disregarding the discrepancies between spoken and written language.In this paper, we propose OLISIA, a cascade system which integrates an Automatic Speech Recognition (ASR) model and a DST model. We introduce several adaptations in the ASR and DST modules to improve integration and robustness to spoken conversations.With these adaptations, our system ranked first in DSTC11 Track 3, a benchmark to evaluate spoken DST. We conduct an in-depth analysis of the results and find that normalizing the ASR outputs and adapting the DST inputs through data augmentation, along with increasing the pre-trained models size all play an important role in reducing the performance discrepancy between written and spoken conversations.
    
[^45]: Deanthropomorphising NLP：语言模型可以意识到吗？

    Deanthropomorphising NLP: Can a Language Model Be Conscious?. (arXiv:2211.11483v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.11483](http://arxiv.org/abs/2211.11483)

    本文讨论了关于使用Transformer架构的预训练语言模型LaMDA是否具有意识的说法。作者认为语言模型不可能具有意识，而LaMDA没有比其他类似模型更具先进性。

    

    本文旨在对最近有关使用Transformer模型架构的预训练语言模型LaMDA具有意识的说法进行讨论。我们认为这样的语言模型不可能具有意识，而LaMDA并没有比其他类似模型更具先进性。我们通过综合信息理论对Transformer架构进行分析来证明这一点。我们认为这些有意识的说法是NLP报道中使用拟人化语言的更广泛倾向的一部分。无论这些说法的真实性如何，我们认为现在是评估语言建模进展并考虑该任务的伦理影响的适当时机。为了使本文有助于NLP社区以外的读者，我们还提供了一些NLP基础知识的介绍。

    This work is intended as a voice in the discussion over the recent claims that LaMDA, a pretrained language model based on the Transformer model architecture, is sentient. This claim, if confirmed, would have serious ramifications in the Natural Language Processing (NLP) community due to wide-spread use of similar models. However, here we take the position that such a language model cannot be sentient, or conscious, and that LaMDA in particular exhibits no advances over other similar models that would qualify it. We justify this by analysing the Transformer architecture through Integrated Information Theory. We see the claims of consciousness as part of a wider tendency to use anthropomorphic language in NLP reporting. Regardless of the veracity of the claims, we consider this an opportune moment to take stock of progress in language modelling and consider the ethical implications of the task. In order to make this work helpful for readers outside the NLP community, we also present the
    
[^46]: DALL-Eval: 探究文本到图像生成模型的推理能力和社会偏见

    DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generation Models. (arXiv:2202.04053v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2202.04053](http://arxiv.org/abs/2202.04053)

    本研究探究了不同文本到图像模型的视觉推理能力和社会偏见。尽管模型在图像生成方面表现出高质量的结果，但在对象计数和空间关系理解能力方面仍存在与上界准确性之间的巨大差距。此外，我们还评估了模型在性别和肤色方面的偏见。

    

    最近，DALL-E，一种多模态的转换语言模型及其变种，包括扩散模型，展示了高质量的文本到图像生成能力。然而，尽管有逼真的图像生成结果，对于如何评估这些模型还没有进行详细的分析。在这项工作中，我们研究了不同文本到图像模型的视觉推理能力和社会偏见，涵盖了多模态转换语言模型和扩散模型。首先，我们量化了三种视觉推理能力：对象识别，对象计数和空间关系理解。为此，我们提出了PaintSkills，这是一个用于衡量这些能力的组合式诊断评估数据集。尽管具有高度逼真的图像生成能力，但在对象计数和空间关系理解能力方面，最近模型的性能与上界准确性之间存在很大差距。其次，我们通过测量性别和肤色偏见来评估模型的性别和肤色偏见。

    Recently, DALL-E, a multimodal transformer language model, and its variants, including diffusion models, have shown high-quality text-to-image generation capabilities. However, despite the realistic image generation results, there has not been a detailed analysis of how to evaluate such models. In this work, we investigate the visual reasoning capabilities and social biases of different text-to-image models, covering both multimodal transformer language models and diffusion models. First, we measure three visual reasoning skills: object recognition, object counting, and spatial relation understanding. For this, we propose PaintSkills, a compositional diagnostic evaluation dataset that measures these skills. Despite the high-fidelity image generation capability, a large gap exists between the performance of recent models and the upper bound accuracy in object counting and spatial relation understanding skills. Second, we assess the gender and skin tone biases by measuring the gender/ski
    

