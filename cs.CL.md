# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Visually grounded few-shot word acquisition with fewer shots.](http://arxiv.org/abs/2305.15937) | 该论文提出了一种基于视觉的语音模型，可以从仅有少量的词-图像示例对中习得新的词汇及其视觉表示，并且与现有方法相比，该模型在使用更少的样本时取得了更好的性能。 |
| [^2] | [BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering.](http://arxiv.org/abs/2305.15932) | 本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。 |
| [^3] | [Emergence of a phonological bias in ChatGPT.](http://arxiv.org/abs/2305.15929) | ChatGPT表现出人类语言处理的音韵偏见，更倾向于使用辅音而不是元音来识别单词。 |
| [^4] | [Reliable identification of selection mechanisms in language change.](http://arxiv.org/abs/2305.15914) | 本文探究了语言变化中的选择机制，提出了一个可靠且可解释的方法来量化历史语言变化的特定实例中的选择强度。该方法被证明比以前应用过的方法更可靠。作者还展示了语音简单性优先于语法简单性，并说明了该方法也可以检测选择强度变化的时间点。 |
| [^5] | [MEMEX: Detecting Explanatory Evidence for Memes via Knowledge-Enriched Contextualization.](http://arxiv.org/abs/2305.15913) | 本研究提出了MEMEX任务，通过知识增强的上下文化技术检测迷因的解释性证据。通过构建MCC数据集，使用分层方法捕捉迷因和上下文的跨模态语义依赖，提出了MIME多模式神经框架来解释迷因。 |
| [^6] | [Response Generation in Longitudinal Dialogues: Which Knowledge Representation Helps?.](http://arxiv.org/abs/2305.15908) | 本文研究了长期对话中的回应生成任务，通过微调GePpeTto(GPT-2)和iT5等PLM模型，并将从LD中提取的个人知识进行不同表示，以获得基于实例的回应生成，以此来解决对话系统面临的挑战。 |
| [^7] | [MTCue: Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation.](http://arxiv.org/abs/2305.15904) | 本研究提出了一种新颖的神经机器翻译框架MTCue，它将所有上下文解释为文本，实现了可转移性并学会了以零样本的方式利用额外的文本属性（如礼貌和对话行为等变量）的控制。在四个语言对的翻译方向上，MTCue的翻译质量显着提高，BLEU（+0.88）和Comet（+1.58）。 |
| [^8] | [Collective Knowledge Graph Completion with Mutual Knowledge Distillation.](http://arxiv.org/abs/2305.15895) | 本文提出一种集体知识图谱补全方法，通过在一个大型聚合知识图谱上使用关系感知图卷积神经网络编码器模型来最大化不同知识图谱的集体知识，并采用互相知识蒸馏机制来增强该方法的效果。 |
| [^9] | [Private Meeting Summarization Without Performance Loss.](http://arxiv.org/abs/2305.15894) | 本文研究了在差分隐私约束下的会议摘要问题，发现差分隐私虽然会稍微降低性能，但在评估未见过的会议类型时却能提高性能，这一发现使得安全的会议摘要更加可行。 |
| [^10] | [CSS: A Large-scale Cross-schema Chinese Text-to-SQL Medical Dataset.](http://arxiv.org/abs/2305.15891) | CSS是一个大规模跨模式中文文本到SQL的医学数据集，以解决现实应用中的跨领域文本到SQL难题。它包括2个数据库中的4,340个问题/SQL对和19个新数据库的29,280个对应的数据集示例。 |
| [^11] | [LFTK: Handcrafted Features in Computational Linguistics.](http://arxiv.org/abs/2305.15878) | 该论文收集和分类了超过220个受欢迎的手工语言特征，设计了一个多语言的手工语言特征提取系统，以系统性的可扩展方式实现，并在几个任务特定的数据集上进行了相关性分析研究。 |
| [^12] | [Linguistic Properties of Truthful Response.](http://arxiv.org/abs/2305.15875) | 该论文研究了LLM的不真实回答现象，发现GPT-3模型对给定提示的回答在语言特性上很相似。同时，该论文证明了在没有评估内容本身的情况下，仅依赖模型回答的风格成分即可分类真实性。 |
| [^13] | [Jointprop: Joint Semi-supervised Learning for Entity and Relation Extraction with Heterogeneous Graph-based Propagation.](http://arxiv.org/abs/2305.15872) | 提出了Jointprop框架，用于基于异构图传播的联合半监督实体和关系提取，采用一个统一的异构图来使用未标记数据中的全局结构信息，提高实体和关系提取性能。 |
| [^14] | [Extracting Text Representations for Terms and Phrases in Technical Domains.](http://arxiv.org/abs/2305.15867) | 本文提出了一种无监督的文本编码方法，使用小型基于字符的模型重构大型预训练嵌入矩阵，其可以在技术领域内达到与句子编码器相同的质量，但大小为后者的五分之一，计算时间能快10倍。 |
| [^15] | [Sequential Integrated Gradients: a simple but effective method for explaining language models.](http://arxiv.org/abs/2305.15853) | 本文提出了一种名为顺序 Integrated Gradients（SIG）的解释语言模型的新方法，通过保持其他单词不变，仅在基线和感兴趣的单词之间创建插值来计算句子中每个单词的重要性，并用训练的令牌“mask”替换基线令牌“pad”来显着改善解释效果。 |
| [^16] | [Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation.](http://arxiv.org/abs/2305.15852) | 本文对大型语言模型的自相矛盾幻觉进行了评估、检测和缓解，探究了这一幻觉形式的普遍存在性。通过设计框架有效触发自相矛盾，发现不同语言模型中这种现象都频繁出现。ChatGPT和GPT-4能够准确识别自相矛盾，而Vicuna-13B则有些困难。 |
| [^17] | [Bhasha-Abhijnaanam: Native-script and romanized Language Identification for 22 Indic languages.](http://arxiv.org/abs/2305.15814) | 该研究提供了22种印度宪法中列出的所有21种本土文字和罗马字母的公开语言鉴别（LID）数据和模型。IndicLID是上述语言的本土和罗马化脚本的语言鉴别器，还提出了解决罗马化文本的LID问题的方案。 |
| [^18] | [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers.](http://arxiv.org/abs/2305.15805) | 本研究提出了一种动态上下文剪枝方法，可以在保持模型表现力的同时，动态减少无效信息，提高模型的效率和可解释性。该技术可以应用于现有的预训练模型，并且可以通过简单的微调过程实现。 |
| [^19] | [MERGE: Fast Private Text Generation.](http://arxiv.org/abs/2305.15769) | 该论文提出了MERGE，一个基于Transformer语言模型的快速私有文本生成框架。实验结果表明，MERGE在保护隐私的同时，实现了26.5倍的加速和80%的通信字节数减少。 |
| [^20] | [Svarah: Evaluating English ASR Systems on Indian Accents.](http://arxiv.org/abs/2305.15760) | Svarah是一个新的基准测试，包含印度65个不同地理位置上的117个说话者的9.6小时的英语音频转录。该基准测试表明，现有的英语ASR系统在印度口音上需要改进。 |
| [^21] | [Healing Unsafe Dialogue Responses with Weak Supervision Signals.](http://arxiv.org/abs/2305.15757) | 本论文提出了一种名为TEMP的无监督伪标签采样方法，可以自动分配潜在的安全响应，解决大规模对话系统的不安全响应生成问题。 |
| [^22] | [UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation.](http://arxiv.org/abs/2305.15756) | UniTRec是一个文本到文本的推荐框架，采用了统一的局部-全局注意力Transformer编码器来处理用户历史的上下文，并且使用Transformer解码器的语言困惑度来构建对比信号，可以显著提高性能。 |
| [^23] | [Multilingual Text-to-Speech Synthesis for Turkic Languages Using Transliteration.](http://arxiv.org/abs/2305.15749) | 通过使用转写法，本研究建立了一个多语种文本转语音合成系统，针对十种突厥语言进行研究，使用Tacotron 2架构的TTS系统，仅使用哈萨克语训练数据，可以实现零样本学习并生成其他突厥语族的语音。 |
| [^24] | [Learn to Not Link: Exploring NIL Prediction in Entity Linking.](http://arxiv.org/abs/2305.15725) | 该论文提出了一个实体链接数据集NEL及其对NIL预测问题的分类方法，研究结果表明在训练数据中，缺失实体和非实体短语均对NIL预测的准确性具有显著影响。 |
| [^25] | [Comparative Study of Pre-Trained BERT Models for Code-Mixed Hindi-English Data.](http://arxiv.org/abs/2305.15722) | 本文比较了使用不同预训练Transformer模型的印地语-英语代码混合数据的性能表现，以提高情感分析、情绪识别和仇恨言论识别等自然语言处理任务的性能。 |
| [^26] | [Towards Higher Pareto Frontier in Multilingual Machine Translation.](http://arxiv.org/abs/2305.15718) | 本文提出了一种新的训练框架——帕累托互攻（Pareto-MD），旨在将帕累托前沿向外推进，而不是进行权衡，以提高多语言机器翻译性能。 |
| [^27] | [The False Promise of Imitating Proprietary LLMs.](http://arxiv.org/abs/2305.15717) | 本文分析了使用较弱开源模型模仿专有语言模型的可靠度，发现在某些情况下，表现可能非常出色，但在大多数任务中仍无法取代专有语言模型。 |
| [^28] | [Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts.](http://arxiv.org/abs/2305.15689) | 本研究提出了一种零样本方法，自动生成多个类似于基础提示的高质量提示，并使用新的度量方法进行排名，从而克服了提示的扰动敏感性，并在情感分类任务中具有较高的准确性。 |
| [^29] | [RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting.](http://arxiv.org/abs/2305.15685) | 本文介绍了 RewriteLM，一种指令调整的大型语言模型，用于长篇文本重写。同时，我们提出了一个名为 OpenRewriteEval 的基准测试，用于评估各种类型的开放式长篇文本重写。我们采用新的策略来促进多样的指令和偏好数据生成，从而为长篇文本重写提供更好的评估手段。 |
| [^30] | [Perturbation-based Self-supervised Attention for Attention Bias in Text Classification.](http://arxiv.org/abs/2305.15684) | 本论文提出了一种基于干扰的自我监督注意力方法来引导注意力学习，无需任何注释开销，能够在三个文本分类任务中显著提高当前基于注意力的模型的性能，该方法比现有的自我监督方法更有效。 |
| [^31] | [Revisiting non-English Text Simplification: A Unified Multilingual Benchmark.](http://arxiv.org/abs/2305.15678) | 这篇论文介绍了MultiSim基准，它包含了27个资源、12种语言超过1.7百万个复杂-简单的句子对。使用该基准进行预训练的多语言语言模型可以在非英语环境中带来令人兴奋的性能提升，并且俄语在跨语言转移方面表现出强大的性能。 |
| [^32] | [Enhancing Grammatical Error Correction Systems with Explanations.](http://arxiv.org/abs/2305.15676) | 该论文介绍了一个用解释提高语法错误修正系统能力的方法，通过引入包含证据单词和语法错误类型注释的大数据集，找到错误的原因，并提出了几个基线和分析方法来理解这个任务，同时也证明了解释可以帮助第二语言学习者更好地理解语法规则。 |
| [^33] | [BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model.](http://arxiv.org/abs/2305.15673) | 本文介绍了一种基于大型语言模型的通用图书推荐框架BookGPT，通过将生成式预训练变换器技术应用于图书推荐场景中的三种任务，即图书评分推荐、用户评分推荐和图书摘要推荐，实现了对图书推荐的有力改进。 |
| [^34] | [Mixture-of-Expert Conformer for Streaming Multilingual ASR.](http://arxiv.org/abs/2305.15663) | 本文提出了一种流式多语言Conformer模型，引入了混合专家层，能够在训练和推理过程中学习仅激活子集参数，实现了高效的推理。与基准模型相比具有显著的WRE性能提升而与适配器模型相比具有类似的性能，无需语言信息，同时利用多语言浅融合还实现了进一步的性能提升。 |
| [^35] | [ConvGQR: Generative Query Reformulation for Conversational Search.](http://arxiv.org/abs/2305.15645) | 本文提出了一种新的面向会话搜索的ConvGQR框架，通过结合预训练语言模型来重新构造查询，从而提供更好的搜索查询。 |
| [^36] | [Morphological Inflection: A Reality Check.](http://arxiv.org/abs/2305.15637) | 本文研究了词形变化任务存在的高性能和高可变性的原因，并提出新的数据采样和评估策略以改善结果的通用性和可靠性。通过这些策略，我们对当前词形变化系统的泛化能力做出了新的观察。 |
| [^37] | [Revisiting Sentence Union Generation as a Testbed for Text Consolidation.](http://arxiv.org/abs/2305.15605) | 本文提出将句子联合生成任务作为一个有效的测试基准，以评估文本整合的能力。该任务将整合挑战与主观内容选择分离开来，并提供了精细的注释方法和工具。实验研究表明，即使是先进的模型也难以应对一些关键的整合方面，表明在这个任务中有明显的改进空间。 |
| [^38] | [Text-Augmented Open Knowledge Graph Completion via Pre-Trained Language Models.](http://arxiv.org/abs/2305.15597) | TAGREAL是一种可自动生成高质量查询提示信息，从大型文本语料库中检索支持信息以从PLM中探测知识的方法，用于开放知识图谱补全中，在两个基准数据集上取得了最先进的表现，并且即使在有限的训练数据情况下，仍然具有突出的性能。 |
| [^39] | [Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models.](http://arxiv.org/abs/2305.15594) | 本文提出了一种差分隐私的提示学习方法，可用于大型语言模型，包括软提示和通过随机鹦鹉群体进行的离散提示，以解决由于提示数据敏感性引起的隐私问题。 |
| [^40] | [How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks.](http://arxiv.org/abs/2305.15587) | 本研究通过调查人们对抗性文本样本的可感知性，得出现有文本攻击在人类参与的现实世界场景中是不切实际的，提供了更为现实的对NLP模型鲁棒性的评估。 |
| [^41] | [Balancing Effect of Training Dataset Distribution of Multiple Styles for Multi-Style Text Transfer.](http://arxiv.org/abs/2305.15582) | 本文探讨了训练数据输入多样性对多种文本风格转换模型生成文本质量的影响，通过平衡训练数据集中的风格分布，可以产生更有效的多种风格控制效果。 |
| [^42] | [Refocusing Is Key to Transfer Learning.](http://arxiv.org/abs/2305.15542) | 这篇论文提出了一种名为 TOAST 的迁移学习算法，通过重新聚焦注意力，选择与任务相关的元素并反馈回模型，有效地提高了细粒度视觉分类数据集的性能，同时具有小部分可调参数。 |
| [^43] | [Automated Refugee Case Analysis: An NLP Pipeline for Supporting Legal Practitioners.](http://arxiv.org/abs/2305.15533) | 本文介绍了一个支持法律从业者的自动化流水线，用于检索、处理和提取法律案件中的有针对性信息，并在加拿大的难民法律案例研究中扩展现有模型，提取19个有用类别的条款。使用最先进的神经命名实体识别进行信息提取，结果表明，在法律数据上预训练的模型表现最佳。 |
| [^44] | [Large Language Models are Few-Shot Health Learners.](http://arxiv.org/abs/2305.15525) | 本论文提出大型语言模型可用于健康应用，只需少量调整便能捕捉健康领域的数字数据并在临床和健康环境下推理及参与各项健康任务。 |
| [^45] | [Exploring Automatically Perturbed Natural Language Explanations in Relation Extraction.](http://arxiv.org/abs/2305.15520) | 本文研究了自然语言解释在关系抽取中的有效性，发现扰动解释也能够达到相当甚至更好的效果，这提供了新的见解。 |
| [^46] | [Free Lunch for Efficient Textual Commonsense Integration in Language Models.](http://arxiv.org/abs/2305.15516) | 本文提出了一种计算上更加高效的方法来将文本常识描述集成到语言模型中，将具有相似常识描述的样本分批次进行编码以提高效率。 |
| [^47] | [The Larger They Are, the Harder They Fail: Language Models do not Recognize Identifier Swaps in Python.](http://arxiv.org/abs/2305.15507) | 该论文揭示了大型语言模型对Python标识符交换的识别问题，尤其是在逆比例缩放现象影响下表现更为显著。这表明LLM缺乏深刻、抽象的理解，无法胜任与训练偏差的任务。 |
| [^48] | [Deriving Language Models from Masked Language Models.](http://arxiv.org/abs/2305.15501) | 本文研究了从掩码语言模型中推导显式联合分布的方法，找到了一种基于条件接近的方法，可以优于现有的基于马尔科夫随机场的方法，并发现条件可以超过原始的MLM。 |
| [^49] | [Large Language Models for User Interest Journeys.](http://arxiv.org/abs/2305.15498) | 该论文提出了使用大型语言模型(LLMs)对用户兴趣进行建模的方法，并通过定义兴趣旅程，提出了一种模型旨在提高推荐的质量，并提供了可解释性和新颖性。 |
| [^50] | [PromptNER: Prompting For Named Entity Recognition.](http://arxiv.org/abs/2305.15444) | PromptNER是一种基于提示的命名实体识别算法，利用LLM生成潜在实体列表并提供解释，在少样本NER和跨领域NER方面实现了最先进性能。 |
| [^51] | [Language Model Tokenizers Introduce Unfairness Between Languages.](http://arxiv.org/abs/2305.15425) | 语言模型的分词器在不同语言之间引入了不公平现象，因为同一段文本翻译成不同的语言可能会导致极大的分词长度差异，这影响了一些语言社区在获取商业语言服务的成本、处理时间和延迟以及提供给机器学习模型的内容量方面存在不公平待遇。 |
| [^52] | [RefGPT: Reference -> Truthful & Customized Dialogues Generation by GPTs and for GPTs.](http://arxiv.org/abs/2305.14994) | RefGPT是一种基于参考的对话生成方法，可以生成大量真实且定制化的对话，并解决了对话生成中的模型幻觉问题。 |
| [^53] | [Utility-Probability Duality of Neural Networks.](http://arxiv.org/abs/2305.14859) | 提出了一种将深度学习中的标准监督学习过程解释为基于效用的解释方法，将学习的神经网络解释为编码在训练数据中显示的偏好的序数效用函数，可以将SGD最大似然估计的学习动态视为将神经网络优化到最优效用函数的迭代过程，从而提供了一个设计更好的神经网络体系结构的新视角。 |
| [^54] | [Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion.](http://arxiv.org/abs/2305.14652) | 本文提出了一种基于去噪瓶颈和最大化互信息的视频多模态融合模型（DBF），该模型可以细粒度地过滤掉冗余和噪声信息，同时保留不同模态中的关键信息，并在多语言视频分类任务中表现出显著优越性。 |
| [^55] | [CMOT: Cross-modal Mixup via Optimal Transport for Speech Translation.](http://arxiv.org/abs/2305.14635) | CMOT是一种用于跨模态语音翻译的方法，通过最优传输找到语音和文本序列之间的对齐，并在标记级别上混合不同模态的序列，实现了在有限数据下更好的性能表现。 |
| [^56] | [Automatic Readability Assessment for Closely Related Languages.](http://arxiv.org/abs/2305.13478) | 本研究探索了如何通过语言方面（如相互智能性或语言相关性）来提高低资源语言中的自动可读性评估，并使用三种菲律宾语言的短篇小说来训练模型，发现应用专业特征CrossNGO可以改善ARA。 |
| [^57] | [Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design.](http://arxiv.org/abs/2305.13297) | 本文研究了利用并行注意力和前馈网络设计（PAF）架构验证了前馈网络在变压器模型中的关键作用，并表明FFN块的主要功能是保持各向同性并防止退化，注意块中计算的残差范数远小于输入令牌嵌入范数。 |
| [^58] | [EMNS /Imz/ Corpus: An emotive single-speaker dataset for narrative storytelling in games, television and graphic novels.](http://arxiv.org/abs/2305.13137) | 该研究提出了一个名为EMNS/Imz/ Corpus的情感单说者数据集，旨在增强交互式叙述驱动系统中对话的表现力和情感质量。该数据集在传达情感和表现力方面表现最佳，尤其在共享情感方面表现出色。 |
| [^59] | [Learning Optimal Policy for Simultaneous Machine Translation via Binary Search.](http://arxiv.org/abs/2305.12774) | 本文提出通过二分搜索学习同时机器翻译最优策略的方法，并在多个翻译任务上验证了在所有延迟情景下超越强基线的效果。 |
| [^60] | [FIT: Far-reaching Interleaved Transformers.](http://arxiv.org/abs/2305.12689) | FIT是一种基于Transformer的架构，将数据标记分组，使用局部层和全局层进行操作。通过交错使用这些层并使用交叉注意力促进信息交换，FIT在一系列任务中均实现最先进的性能。 |
| [^61] | [Cross2StrA: Unpaired Cross-lingual Image Captioning with Cross-lingual Cross-modal Structure-pivoted Alignment.](http://arxiv.org/abs/2305.12260) | 本论文提出了一种基于跨语言交叉模态结构枢纽对跨语言图像字幕生成的方法，通过结合场景图和句法句子树结构来提高字幕生成相关性和流畅性，并使用跨语言和跨模式的返译训练以完全对齐字幕生成和翻译阶段。 |
| [^62] | [Constructing Code-mixed Universal Dependency Forest for Unbiased Cross-lingual Relation Extraction.](http://arxiv.org/abs/2305.12258) | 本文提出了一种无偏的UD树跨语言关系抽取转移方法，通过构建混合编码UD树，有助于弥合训练和预测阶段之间差距从而实现显著的跨语言关系抽取性能提升。 |
| [^63] | [Scene Graph as Pivoting: Inference-time Image-free Unsupervised Multimodal Machine Translation with Visual Scene Hallucination.](http://arxiv.org/abs/2305.12256) | 本研究提出了一种基于情景图的轴心方法，通过动态生成伪视觉情景图来实现推理过程中的纯文本输入，从而实现了无监督多模态机器翻译任务。 |
| [^64] | [Prefix Propagation: Parameter-Efficient Tuning for Long Sequences.](http://arxiv.org/abs/2305.12086) | 前缀传播是一种针对长序列参数高效调整的方法，可实现50%减少参数且在处理长文档任务时具有更优性能。 |
| [^65] | [Generating Visual Spatial Description via Holistic 3D Scene Understanding.](http://arxiv.org/abs/2305.11768) | 本文研究将3D场景特征纳入VSD方法，构建目标对象为中心的3D空间场景图(Go3D-S2G)，提出多样化的文本生成方法，可以显著提高性能。 |
| [^66] | [Information Screening whilst Exploiting! Multimodal Relation Extraction with Feature Denoising and Multimodal Topic Modeling.](http://arxiv.org/abs/2305.11719) | 该论文提出了一种新的多模态关系抽取框架，结合了内部信息筛选和外部信息利用的思想。通过视觉和文本场景图表示输入的细粒度语义结构，并利用图形信息瓶颈原理进行结构细化和特征去噪，同时运用主题建模丰富上下文，该系统在基准MRE数据集上表现优异，具有巨大的潜力。 |
| [^67] | [Zero-Shot Text Classification via Self-Supervised Tuning.](http://arxiv.org/abs/2305.11442) | 本文提出了一种基于自监督调整的零样本文本分类算法，通过使用无标签数据来调整语言模型，通过学习预测段落中的第一句话，实现了对未见过任务的零样本推断，模型不需要注释数据进行元调整，对模板的选择不敏感，并在实验中取得不错的结果。 |
| [^68] | [Reasoning Implicit Sentiment with Chain-of-Thought Prompting.](http://arxiv.org/abs/2305.11255) | 本研究提出了一种基于思维链索引的隐式情感推断框架（THOR），通过三次跳推理模仿类人推理过程，支持常识和多跳推理以推断意见的潜在意图，并逐步诱导隐式方面、意见和最终情感极性，实现了在监督和零样本设置上大幅提高技术水平。 |
| [^69] | [Accurate and Reliable Confidence Estimation Based on Non-Autoregressive End-to-End Speech Recognition System.](http://arxiv.org/abs/2305.10680) | 本文介绍了一种CIF-Aligned置信度估计模型，利用了非自回归E2E ASR模型-Paraformer的特性，生成符号同步的声学嵌入，实现了准确可靠的置信度估计。 |
| [^70] | [From chocolate bunny to chocolate crocodile: Do Language Models Understand Noun Compounds?.](http://arxiv.org/abs/2305.10568) | 本文研究了语言模型对于名词复合词的理解能力，提出了名词复合词解释的任务和名词复合词概念化的任务，并发现GPT-3在这些任务中的表现优于人类。 |
| [^71] | [sustain.AI: a Recommender System to analyze Sustainability Reports.](http://arxiv.org/abs/2305.08711) | sustain.AI是一个智能的、上下文感知的推荐系统，可以帮助审计师、金融投资者以及广大公众高效地分析公司的可持续性报告，并通过与GRI标准匹配来提供更好的推荐精度。 |
| [^72] | [Using LLM-assisted Annotation for Corpus Linguistics: A Case Study of Local Grammar Analysis.](http://arxiv.org/abs/2305.08339) | 本文研究了使用基于大语言模型的聊天机器人自动标注文本的潜力，重点考察了从本地语法角度观察道歉言语行为构成的功能元素的程度，并比较了不同模型在注释任务中的表现，结果表明Bing聊天机器人在任务中表现优于ChatGPT和人类标注员。 |
| [^73] | [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?.](http://arxiv.org/abs/2305.07759) | 本文针对小型语言模型生成连贯的英文文本难题，引入了一个合成故事数据集 TinyStories，并探索小型模型规模、结构复杂度和训练数据规模对于语言模型表现的影响，证明了仅含 200 万参数的简单语言模型也能产生连贯的短故事。 |
| [^74] | [Masked Audio Text Encoders are Effective Multi-Modal Rescorers.](http://arxiv.org/abs/2305.07677) | 本文提出了Masked Audio Text Encoders（MATE），一个多模态掩码语言模型重新打分器，将声学表示形式并入到MLM的输入空间中。使用MATE对自动语音识别（ASR）系统进行多模态打分，即使在目标域数据不足的情况下，也可以提高系统的领域泛化能力，并且可以在非常有限的训练数据量下就将单词错误率（WER）降低。 |
| [^75] | [SemEval-2023 Task 2: Fine-grained Multilingual Named Entity Recognition (MultiCoNER 2).](http://arxiv.org/abs/2305.06586) | 该论文介绍了SemEval-2023 Task 2的研究发现，任务旨在通过使用MultiCoNER V2数据集，识别12种语言中复杂的细粒度命名实体。最优方法是将外部知识融入transformer模型，最具挑战性的是媒体标题和产品名称等实体类型。 |
| [^76] | [How Do In-Context Examples Affect Compositional Generalization?.](http://arxiv.org/abs/2305.04835) | 本文提出了CoFe测试套件来调查上下文组合泛化。实验结果表明，上下文示例应该在结构上与测试用例类似，相互之间应该不同，而且单独地简单。 |
| [^77] | [DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation.](http://arxiv.org/abs/2305.04720) | DEnsity 提出了一种利用密度估计的开放域对话评估新方法，在特征空间中评估响应可能性来更好地与人类评估相关。 |
| [^78] | [Pay More Attention to Relation Exploration for Knowledge Base Question Answering.](http://arxiv.org/abs/2305.02118) | 该研究提出了一个新框架RE-KBQA，利用知识库中的关系增强实体表示，并引入额外监督。在三个方面探索关系指导，包括区分相似实体、探索额外监督以及进行后处理的基于关系指导的重排算法。该方法在两个基准数据集上验证有效性。 |
| [^79] | [Few-shot Event Detection: An Empirical Study and a Unified View.](http://arxiv.org/abs/2305.01901) | 本文从两个实用的设置出发，分析比较了十种代表性的小样本事件检测方法，归纳总结出了原型方法的性能优越性，并在此基础上提出了一种简单且有效的方法。 |
| [^80] | [RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment.](http://arxiv.org/abs/2304.06767) | RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。 |
| [^81] | [Measuring Gender Bias in West Slavic Language Models.](http://arxiv.org/abs/2304.05783) | 本研究分析了西斯拉夫语言模型中的性别偏差，发现捷克语、波兰语和斯洛伐克语均存在相似程度的性别偏见。这一研究填补了研究非英语语言模型性别偏见的空白。 |
| [^82] | [LLMMaps -- A Visual Metaphor for Stratified Evaluation of Large Language Models.](http://arxiv.org/abs/2304.00457) | LLMMaps是一种分层评估大型语言模型性能的可视化技术，能够揭示取得高准确度和产生幻觉的子领域，并指导模型的进一步发展。 |
| [^83] | [HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace.](http://arxiv.org/abs/2303.17580) | 用ChatGPT作为任务规划工具，利用大型语言模型（LLM）作为控制器来整合现有的AI模型，解决复杂的AI任务。 |
| [^84] | [InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis.](http://arxiv.org/abs/2302.08624) | InstructABSA是一种使用指令学习范式的方面情感分析方法，能够显著提高Aspect Term Extraction、Aspect Term Sentiment Classification、和Joint Task subtasks三个子任务的性能，并且在多个数据集上表现超过之前的最先进方法。 |
| [^85] | [Whats New? Identifying the Unfolding of New Events in Narratives.](http://arxiv.org/abs/2302.07748) | 本文提出了一个新的挑战性任务：自动识别叙述中的新事件，以识别创新和贡献。他们将事件定义为主语、谓语和宾语的三元组，并将其分类为新事件，具体取决于其是否可以通过常识推理来推导。 |
| [^86] | [Transformer models: an introduction and catalog.](http://arxiv.org/abs/2302.07730) | 本论文介绍与分类了Transformer模型系列中最流行的模型，包括基于自监督学习和人类参与训练的模型，并对其中创新性的方面做了介绍。 |
| [^87] | [READIN: A Chinese Multi-Task Benchmark with Realistic and Diverse Input Noises.](http://arxiv.org/abs/2302.07324) | READIN是一个中文多任务基准数据集，它包含真实和多样化的输入噪声，旨在测试模型的鲁棒性和公平性。注释管道被设计来最大化多样性。 |
| [^88] | [Selective Explanations: Leveraging Human Input to Align Explainable AI.](http://arxiv.org/abs/2301.09656) | 本研究提出一种通过利用人类输入生成选择性解释的通用框架，以弥合可解释人工智能（XAI）与人类解释的差距，并且在决策支持任务中进行了实验证明其有效性。 |
| [^89] | [Continual Contrastive Finetuning Improves Low-Resource Relation Extraction.](http://arxiv.org/abs/2212.10823) | 本文提出了一种使用连续对比微调的方法来改进低资源关系提取，通过使用一致的对比学习目标预训练和微调RE模型，以及多中心对比损失来允许一个关系形成多个聚类。实验结果表明该方法可以显着提高低资源情况和领域中的关系提取性能。 |
| [^90] | [PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition.](http://arxiv.org/abs/2212.10750) | 这个论文提出了一个大规模的命题级别分割和包含关系识别的语料库PropSegmEnt解决了NLI中对语义单元的识别问题。 |
| [^91] | [Data Curation Alone Can Stabilize In-context Learning.](http://arxiv.org/abs/2212.10378) | 精心整理训练数据子集可以极大地稳定上下文学习表现，而不需要对ICL算法进行其他更改。CondAcc通过将训练示例与随机训练示例组合时的平均开发集ICL准确性来评分训练示例，而DataModels学习线性回归器，估计每个训练示例的存在如何影响LLM输出。 |
| [^92] | [HINT: Hypernetwork Instruction Tuning for Efficient Zero- & Few-Shot Generalisation.](http://arxiv.org/abs/2212.10315) | 本文介绍了一种新的NLP模型HINT，它使用超网络将任务指令和示例转换为参数高效的模块，从而无需将指令包含在模型输入中，并可为解码期间提供编码指令。HINT模型在计算量相等的情况下性能比最新的基线模型强10%以上，解决了高计算成本的问题。 |
| [^93] | [Do language models have coherent mental models of everyday things?.](http://arxiv.org/abs/2212.10029) | 语言模型缺乏对日常物品的一致性心理模型，会因此出现荒谬的解决方法。虽然最先进的预训练语言模型具有这些实体的知识碎片，但它们无法为所有实体产生一致且正确的心理模型。语言模型训练可以改善这种情况。 |
| [^94] | [Human-in-the-loop Evaluation for Early Misinformation Detection: A Case Study of COVID-19 Treatments.](http://arxiv.org/abs/2212.09683) | 该论文提出了一种人机协同评估框架，用于检测新的虚假信息声明并识别支持它们的社交媒体消息。在COVID-19治疗的案例中，基于现代NLP方法开发基线系统，并展示了人类事实核查人员每小时可以识别出违反Twitter关于COVID-19虚假信息方针的124条推文。 |
| [^95] | [The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges.](http://arxiv.org/abs/2212.09660) | 本文系统调查了几十年来自然语言处理中关于代码交换的研究进展和挑战，总结了趋势和发现，并讨论了未来方向和进一步研究的开放性问题。 |
| [^96] | [BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting.](http://arxiv.org/abs/2212.09535) | 本文在BLOOM模型中应用语言适应策略，将其适应到新语言上，并在八种新语言的零样本提示表现中提升了性能。适配器微调比大模型的持续预训练更有效，提示性能主要由语言适应数据的大小确定。 |
| [^97] | [Lattice-Free Sequence Discriminative Training for Phoneme-Based Neural Transducers.](http://arxiv.org/abs/2212.04325) | 本文提出了三种无格栅训练目标，用于基于音素的神经传递器的最终后验输出，与使用N-best列表的方法相比，无格栅方法在训练期间消除了假设生成的步骤，从而导致更高效的训练，在单词错误率上表现也有6.5％的相对改进。 |
| [^98] | [Data-Efficient Finetuning Using Cross-Task Nearest Neighbors.](http://arxiv.org/abs/2212.00196) | 本文提出一种通过跨任务最近邻进行数据高效微调的方法，通过仅使用少量目标任务数据和多任务数据中的最相似标记数据，避免了对大量标记数据的需求，取得了比强基准模型更好的效果。 |
| [^99] | [Evaluating and reducing the distance between synthetic and real speech distributions.](http://arxiv.org/abs/2211.16049) | 本研究通过比较真实语音和合成语音的分布，使用统计学方法量化它们之间的距离，最终实现了10%的距离缩小。 |
| [^100] | [Global and Local Hierarchy-aware Contrastive Framework for Implicit Discourse Relation Recognition.](http://arxiv.org/abs/2211.13873) | 本文提出了GOLF框架，它能够充分利用全局和本地感知层次结构来提升隐含篇章关系识别效果。 |
| [^101] | [The NCTE Transcripts: A Dataset of Elementary Math Classroom Transcripts.](http://arxiv.org/abs/2211.11772) | NCTE记录提供了一个庞大的小学数学课堂记录数据集，它有助于研究课堂对话并改善教学质量。 |
| [^102] | [Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text.](http://arxiv.org/abs/2211.11300) | 本文提出了一种多层知识蒸馏方法，融合了语言模型的训练和微调方法来进行文本中的离群检测，实验结果表明其有效性。 |
| [^103] | [Using Persuasive Writing Strategies to Explain and Detect Health Misinformation.](http://arxiv.org/abs/2211.05985) | 本研究旨在通过使用说服性写作技巧的文本段落进行分类来增加自动化虚假信息检测的新层次，以产生可解释的理由。我们提出了一个包含常见说服性写作策略的注释方案和数据集，并使用 RoBERTa 文本分类模型进行实验。 |
| [^104] | [Impact of Adversarial Training on Robustness and Generalizability of Language Models.](http://arxiv.org/abs/2211.05523) | 本研究比较了在变压器语言模型中不同的对抗训练方法，并发现预训练数据增强或训练时间输入扰动可以实现更好的鲁棒性，而训练中使用的嵌入空间扰动可以显著提高泛化性。神经元的语言相关性分析表明，这些改进是由于存在“更专业”的神经元。这是首个对语言模型对抗训练不同方法进行全面比较的工作。 |
| [^105] | [Zero-Shot Classification by Logical Reasoning on Natural Language Explanations.](http://arxiv.org/abs/2211.03252) | 本文提出了一种基于逻辑推理的零样本分类方法，通过对自然语言解释进行分析和推理，将文本信息显式地编码成逻辑结构， 进而获得可靠的分类结果。 |
| [^106] | [Contextual information integration for stance detection via cross-attention.](http://arxiv.org/abs/2211.01874) | 本论文提出一种将来自异构数据源的上下文信息作为文本整合的方法，用于立场检测，取得了优于竞争基线的结果，对于未被提前见过的目标仍然有效。 |
| [^107] | [Random Utterance Concatenation Based Data Augmentation for Improving Short-video Speech Recognition.](http://arxiv.org/abs/2210.15876) | 本文提出了一种基于随机话语拼接的数据增强方法，用于提高短视频语音识别任务的训练和测试话语长度不匹配问题。该方法可显著提高长话语的识别率，同时对短话语的性能没有下降，并取得了5.72%的词错误率降低。 |
| [^108] | [Towards Parameter-Efficient Integration of Pre-Trained Language Models In Temporal Video Grounding.](http://arxiv.org/abs/2209.13359) | 本文探讨了如何更加高效地将预训练语言模型应用于视频时间对齐任务中，并通过将PLMs与现有方法相结合，证明了在三个具有挑战性的数据集上，TVG模型从PLM集成和微调中受益匪浅。 |
| [^109] | [Learning Better Masking for Better Language Model Pre-training.](http://arxiv.org/abs/2208.10806) | 本文提出了两种时间变化的MLM掩蔽策略，可以在不同的训练阶段自适应地调整掩蔽比例和掩蔽内容，提高语言模型的预训练效率和有效性。 |
| [^110] | [DICE: Data-Efficient Clinical Event Extraction with Generative Models.](http://arxiv.org/abs/2208.07989) | 介绍了一种稳健高效的基于生成模型的临床事件抽取方法DICE，引入了对比性学习目标和特殊标记，共同训练实体提及和事件抽取等辅助任务，所提出的MACCROBAT-EE数据集为临床事件抽取提供了基准测试。 |
| [^111] | [Claim-Dissector: An Interpretable Fact-Checking System with Joint Re-ranking and Veracity Prediction.](http://arxiv.org/abs/2207.14116) | Claim-Dissector是一款联合重排和真实性预测的可解释的事实核查系统，可以识别与声明相关的证据，并确定声明的真实性。该系统的个人贡献以及证据所支持或反驳声明的贡献都可以被识别。 |
| [^112] | [GENEVA: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles.](http://arxiv.org/abs/2205.12505) | 本文提出了一个大而全的EAE本体论，105个事件和220个论元角色的包含在内，利用这个本体论创建了一种多样化的通用性基准测试数据集GENEVA，共包含四个测试套件，旨在评估模型处理有限数据的能力。 |
| [^113] | [A Computational Inflection for Scientific Discovery.](http://arxiv.org/abs/2205.02007) | 本文介绍了一种计算变革的框架，利用最新的人工智能技术来增强科学发现和交流。这个框架有很多应用场景，作者提供了一个原型系统的初始实现，并探讨了未来研究和发展方向。 |
| [^114] | [Explore More Guidance: A Task-aware Instruction Network for Sign Language Translation Enhanced with Data Augmentation.](http://arxiv.org/abs/2204.05953) | 本研究提出了一种任务感知指令网络TIN-SLT用于手语翻译，引入指令模块和特征融合策略进一步提升了翻译性能，并且通过多层数据增强方案调整了数据分布。 |
| [^115] | [Contrastive Learning of Sociopragmatic Meaning in Social Media.](http://arxiv.org/abs/2203.07648) | 提出了一种社交媒体中社会语用意义的对比学习框架，该框架能够学习可迁移的任务不可知表示学习，并在各种对比学习框架中表现最佳。 |
| [^116] | [HyperMixer: An MLP-based Low Cost Alternative to Transformers.](http://arxiv.org/abs/2203.03691) | HyperMixer是一种低成本的基于MLP的Transformer替代方案，通过动态形成标记混合MLP来实现自然语言理解，其性能比替代方案好，并可与Transformer媲美，成本更低。 |
| [^117] | [pNLP-Mixer: an Efficient all-MLP Architecture for Language.](http://arxiv.org/abs/2202.04350) | pNLP-Mixer是一种新型的MLP-Mixer模型，不需要嵌入层，用于设备上高效的自然语言处理，可以达到基于transformer架构的大型预训练语言模型相近的性能，却只需要很少的资源。 |

# 详细

[^1]: 用更少的数据进行视觉上有依据的少样本词汇习得

    Visually grounded few-shot word acquisition with fewer shots. (arXiv:2305.15937v1 [cs.CL])

    [http://arxiv.org/abs/2305.15937](http://arxiv.org/abs/2305.15937)

    该论文提出了一种基于视觉的语音模型，可以从仅有少量的词-图像示例对中习得新的词汇及其视觉表示，并且与现有方法相比，该模型在使用更少的样本时取得了更好的性能。

    

    我们提出了一种基于视觉的语音模型，它可以从仅有少量的词-图像示例对中习得新的词汇及其视觉表示。给定一组测试图像和一个口头查询，我们要求模型指出哪个图像展示了查询词。先前的工作要么使用数字词-图像对的人造环境来简化该问题，要么使用每个类别大量的示例。我们提出了一种方法，可以在自然的词-图像对上进行，但只需更少的数据，即更少的样本。我们的方法包括使用给定的词-图像示例对从大量未标记的语音和图像中挖掘新的无监督词-图像训练对。另外，我们使用了一种单词到图像的注意力机制来确定词-图像的相似度。通过这种新模型，我们实现了比任何现有方法都更好的性能，而且只需更少的数据量。

    We propose a visually grounded speech model that acquires new words and their visual depictions from just a few word-image example pairs. Given a set of test images and a spoken query, we ask the model which image depicts the query word. Previous work has simplified this problem by either using an artificial setting with digit word-image pairs or by using a large number of examples per class. We propose an approach that can work on natural word-image pairs but with less examples, i.e. fewer shots. Our approach involves using the given word-image example pairs to mine new unsupervised word-image training pairs from large collections of unlabelled speech and images. Additionally, we use a word-to-image attention mechanism to determine word-image similarity. With this new model, we achieve better performance with fewer shots than any existing approach.
    
[^2]: BUCA：一种用于无监督常识问题回答的二分类方法

    BUCA: A Binary Classification Approach to Unsupervised Commonsense Question Answering. (arXiv:2305.15932v1 [cs.CL])

    [http://arxiv.org/abs/2305.15932](http://arxiv.org/abs/2305.15932)

    本文提出了一种更简单的二分类方法，将下游的多项选择题回答任务转换为二分类任务，根据合理性对所有候选答案进行排名，以实现无监督常识问题回答，相较于现有使用知识图谱的UCR方法，我们的方法更为节省数据。

    

    随着常识推理数据集的构建变得越来越昂贵且在范围上不可避免地受限，无监督的常识推理(UCR)变得越来越流行。UCR的一种流行方法是利用外部知识将语言模型进行微调(例如，知识图谱)，但这通常需要大量的训练样例。在本文中，我们提出将下游的多项选择题回答任务转换为一个更简单的二分类任务，通过对所有候选答案的合理性进行排名来完成。为了训练模型，我们将知识图谱三元组转换为合理和不合理的文本。广泛的实验结果显示了我们的方法在各种多项选择问题回答基准测试中的有效性。此外，与使用KG的现有UCR方法相比，我们的方法更节省数据。我们的代码可在https://github.com/probe2/BUCA上获取。

    Unsupervised commonsense reasoning (UCR) is becoming increasingly popular as the construction of commonsense reasoning datasets is expensive, and they are inevitably limited in their scope. A popular approach to UCR is to fine-tune language models with external knowledge (e.g., knowledge graphs), but this usually requires a large number of training examples. In this paper, we propose to transform the downstream multiple choice question answering task into a simpler binary classification task by ranking all candidate answers according to their reasonableness. To this end, for training the model, we convert the knowledge graph triples into reasonable and unreasonable texts. Extensive experimental results show the effectiveness of our approach on various multiple choice question answering benchmarks. Furthermore, compared with existing UCR approaches using KGs, ours is less data hungry. Our code is available at https://github.com/probe2/BUCA.
    
[^3]: ChatGPT中出现了音韵偏见

    Emergence of a phonological bias in ChatGPT. (arXiv:2305.15929v1 [cs.CL])

    [http://arxiv.org/abs/2305.15929](http://arxiv.org/abs/2305.15929)

    ChatGPT表现出人类语言处理的音韵偏见，更倾向于使用辅音而不是元音来识别单词。

    

    当前的大型语言模型，例如OpenAI的ChatGPT，因其在语言使用方面的出色表现而受到公众的关注。在这里，我证明了ChatGPT显示了人类语言处理的音韵偏见。更具体地说，就像人类一样，ChatGPT具有一个辅音偏见。也就是说，这个聊天机器人倾向于使用辅音而不是元音来识别单词。这在具有不同辅音和元音分布比例的语言（如英语和西班牙语）中都有观察到。尽管当前人工智能语言模型在处理语言刺激和人类婴儿获得语言的方式上存在差异，但这样的训练似乎足以在ChatGPT中引出一个音韵偏见。

    Current large language models, such as OpenAI's ChatGPT, have captured the public's attention because how remarkable they are in the use of language. Here, I demonstrate that ChatGPT displays phonological biases that are a hallmark of human language processing. More concretely, just like humans, ChatGPT has a consonant bias. That is, the chatbot has a tendency to use consonants over vowels to identify words. This is observed across languages that differ in their relative distribution of consonants and vowels such as English and Spanish. Despite the differences in how current artificial intelligence language models are trained to process linguistic stimuli and how human infants acquire language, such training seems to be enough for the emergence of a phonological bias in ChatGPT
    
[^4]: 语言变化中选择机制的可靠识别

    Reliable identification of selection mechanisms in language change. (arXiv:2305.15914v1 [cs.CL])

    [http://arxiv.org/abs/2305.15914](http://arxiv.org/abs/2305.15914)

    本文探究了语言变化中的选择机制，提出了一个可靠且可解释的方法来量化历史语言变化的特定实例中的选择强度。该方法被证明比以前应用过的方法更可靠。作者还展示了语音简单性优先于语法简单性，并说明了该方法也可以检测选择强度变化的时间点。

    

    语言变化是一种文化进化过程，其中语言变量的变异通过类似于突变、选择和遗传漂变的过程而频繁变化。本文应用最近引入的一种方法来对语料库数据进行分析，以量化历史语言变化的特定实例中的选择强度。我们首先在英语不规则动词的语境下证明了这种方法比以前应用过的类似方法更可靠和可解释。我们进一步扩展了这项研究，证明了在语音简单性与语法简单性冲突时，对语音简单性的偏好优先于对语法简单性的偏好。最后，针对西班牙的拼写改革，我们展示了该方法也可以检测选择强度变化的时间点，这是社会动机语言变化通常具有的特征。这些结果共同表明如何测试语言变化机制的假设。

    Language change is a cultural evolutionary process in which variants of linguistic variables change in frequency through processes analogous to mutation, selection and genetic drift. In this work, we apply a recently-introduced method to corpus data to quantify the strength of selection in specific instances of historical language change. We first demonstrate, in the context of English irregular verbs, that this method is more reliable and interpretable than similar methods that have previously been applied. We further extend this study to demonstrate that a bias towards phonological simplicity overrides that favouring grammatical simplicity when these are in conflict. Finally, with reference to Spanish spelling reforms, we show that the method can also detect points in time at which selection strengths change, a feature that is generically expected for socially-motivated language change. Together, these results indicate how hypotheses for mechanisms of language change can be tested qu
    
[^5]: MEMEX：通过知识增强的上下文化来检测迷因的解释性证据

    MEMEX: Detecting Explanatory Evidence for Memes via Knowledge-Enriched Contextualization. (arXiv:2305.15913v1 [cs.CL])

    [http://arxiv.org/abs/2305.15913](http://arxiv.org/abs/2305.15913)

    本研究提出了MEMEX任务，通过知识增强的上下文化技术检测迷因的解释性证据。通过构建MCC数据集，使用分层方法捕捉迷因和上下文的跨模态语义依赖，提出了MIME多模式神经框架来解释迷因。

    

    迷因是社交媒体上强大的交际工具，它们在政治、历史和社会文化现象中的不断发展使其成为理想的交流媒介。为了理解迷因传达的微妙信息，必须了解促进其整体吸收的背景。除了像knowyourmeme.com这样的几个网站对迷因及其元数据进行数字存档外，目前没有有效的方法动态地推断迷因的上下文。在这项工作中，我们提出了一个新的任务，MEMEX，给定一个迷因和一个相关的文档，其目的是挖掘简洁地解释迷因背景的上下文。首先，我们开发了MCC（Meme Context Corpus），这是一个为MEMEX设计的新数据集。此外，为了基准测试MCC，我们提出了MIME（MultImodal Meme Explainer），这是一个多模式神经框架，使用通识强化的迷因表示和一种分层方法来捕捉迷因和上下文之间的跨模态语义依赖。

    Memes are a powerful tool for communication over social media. Their affinity for evolving across politics, history, and sociocultural phenomena makes them an ideal communication vehicle. To comprehend the subtle message conveyed within a meme, one must understand the background that facilitates its holistic assimilation. Besides digital archiving of memes and their metadata by a few websites like knowyourmeme.com, currently, there is no efficient way to deduce a meme's context dynamically. In this work, we propose a novel task, MEMEX given a meme and a related document, the aim is to mine the context that succinctly explains the background of the meme. At first, we develop MCC (Meme Context Corpus), a novel dataset for MEMEX. Further, to benchmark MCC, we propose MIME (MultImodal Meme Explainer), a multimodal neural framework that uses common sense enriched meme representation and a layered approach to capture the cross-modal semantic dependencies between the meme and the context. M
    
[^6]: 长期对话中的回应生成：哪种知识表示有助于？

    Response Generation in Longitudinal Dialogues: Which Knowledge Representation Helps?. (arXiv:2305.15908v1 [cs.CL])

    [http://arxiv.org/abs/2305.15908](http://arxiv.org/abs/2305.15908)

    本文研究了长期对话中的回应生成任务，通过微调GePpeTto(GPT-2)和iT5等PLM模型，并将从LD中提取的个人知识进行不同表示，以获得基于实例的回应生成，以此来解决对话系统面临的挑战。

    

    长期对话是人机对话系统面临的最具挑战性的类型之一。长期对话包括个人在稀疏的对话序列中回忆的事件、个人思想和情感等内容。设计用于长期对话的对话系统应该能够在多个对话会话和长时间（例如数周）内与用户进行独特交互，并让他们参与个人对话以阐述他们的感受、思想和真实生活事件。本文研究了长期对话中的回应生成任务。我们评估了通用的预训练语言模型（PLM）是否适合这个任务。我们使用LD数据集微调了两个PLM模型，GePpeTto (GPT-2)和iT5。我们尝试了从LD中提取的个人知识的不同表示形式，包括提到的事件和参与者的图形表示，以获得基于实例的回应生成。我们通过自动指标评估了模型的性能。

    Longitudinal Dialogues (LD) are the most challenging type of conversation for human-machine dialogue systems. LDs include the recollections of events, personal thoughts, and emotions specific to each individual in a sparse sequence of dialogue sessions. Dialogue systems designed for LDs should uniquely interact with the users over multiple sessions and long periods of time (e.g. weeks), and engage them in personal dialogues to elaborate on their feelings, thoughts, and real-life events. In this paper, we study the task of response generation in LDs. We evaluate whether general-purpose Pre-trained Language Models (PLM) are appropriate for this purpose. We fine-tune two PLMs, GePpeTto (GPT-2) and iT5, using a dataset of LDs. We experiment with different representations of the personal knowledge extracted from LDs for grounded response generation, including the graph representation of the mentioned events and participants. We evaluate the performance of the models via automatic metrics an
    
[^7]: MTCue：利用神经机器翻译中未结构化上下文学习零样本控制额外文本属性

    MTCue: Learning Zero-Shot Control of Extra-Textual Attributes by Leveraging Unstructured Context in Neural Machine Translation. (arXiv:2305.15904v1 [cs.CL])

    [http://arxiv.org/abs/2305.15904](http://arxiv.org/abs/2305.15904)

    本研究提出了一种新颖的神经机器翻译框架MTCue，它将所有上下文解释为文本，实现了可转移性并学会了以零样本的方式利用额外的文本属性（如礼貌和对话行为等变量）的控制。在四个语言对的翻译方向上，MTCue的翻译质量显着提高，BLEU（+0.88）和Comet（+1.58）。

    

    高效地利用文本内和文本外的上下文仍是机器和人类翻译之间的关键差距之一。现有的研究主要集中在在翻译中提供个别定义良好类型的上下文，如周围的文本或离散的外部变量（如说话者的性别）。本文介绍了MTCue，这是一个新颖的神经机器翻译（NMT）框架，它将所有上下文（包括离散变量）解释为文本。 MTCue学习上下文的抽象表达，即使在不同的数据设置和低资源场景中，也能实现可转移性并利用类似属性。我们不断评估MTCue在四个语言对的翻译方向上，重点关注具有文档和元数据上下文访问权限的对话领域。与参数匹配的非上下文基线相比，我们的框架在翻译质量方面表现出显着的提高，BLEU（+0.88）和Comet（+1.58）。此外，MTCue成功地学会了以零样本的方式利用额外的文本属性，实现了诸如礼貌和对话行为等变量的控制。

    Efficient utilisation of both intra- and extra-textual context remains one of the critical gaps between machine and human translation. Existing research has primarily focused on providing individual, well-defined types of context in translation, such as the surrounding text or discrete external variables like the speaker's gender. This work introduces MTCue, a novel neural machine translation (NMT) framework that interprets all context (including discrete variables) as text. MTCue learns an abstract representation of context, enabling transferability across different data settings and leveraging similar attributes in low-resource scenarios. With a focus on a dialogue domain with access to document and metadata context, we extensively evaluate MTCue in four language pairs in both translation directions. Our framework demonstrates significant improvements in translation quality over a parameter-matched non-contextual baseline, as measured by BLEU (+0.88) and Comet (+1.58). Moreover, MTCu
    
[^8]: 带互相知识蒸馏的集体知识图谱补全

    Collective Knowledge Graph Completion with Mutual Knowledge Distillation. (arXiv:2305.15895v1 [cs.CL])

    [http://arxiv.org/abs/2305.15895](http://arxiv.org/abs/2305.15895)

    本文提出一种集体知识图谱补全方法，通过在一个大型聚合知识图谱上使用关系感知图卷积神经网络编码器模型来最大化不同知识图谱的集体知识，并采用互相知识蒸馏机制来增强该方法的效果。

    

    知识图谱补全是根据知识图谱内的现有关系数据预测丢失信息的任务，近年来受到了重视。然而，不同来源和语言的知识图谱之间的完整性常常限制了KGC方法的预测能力。在单语和多语环境中，知识图谱潜在地互补。本文研究了多知识图谱补全问题，重点是为了增强个体知识图谱的不完整性而最大化来自不同知识图谱的集体知识。具体而言，我们提出了一种新方法，称为CKGC-CKD，在个体知识图谱和一个大型聚合知识图谱上使用关系感知图卷积神经网络编码器模型，其中KG之间的种子对齐被视为消息传递的边缘。我们还采用了一种额外的互相知识蒸馏机制，以最大化模型之间的知识传递。

    Knowledge graph completion (KGC), the task of predicting missing information based on the existing relational data inside a knowledge graph (KG), has drawn significant attention in recent years. However, the predictive power of KGC methods is often limited by the completeness of the existing knowledge graphs from different sources and languages. In monolingual and multilingual settings, KGs are potentially complementary to each other. In this paper, we study the problem of multi-KG completion, where we focus on maximizing the collective knowledge from different KGs to alleviate the incompleteness of individual KGs. Specifically, we propose a novel method called CKGC-CKD that uses relation-aware graph convolutional network encoder models on both individual KGs and a large fused KG in which seed alignments between KGs are regarded as edges for message propagation. An additional mutual knowledge distillation mechanism is also employed to maximize the knowledge transfer between the models 
    
[^9]: 不损失性能的私人会议摘要

    Private Meeting Summarization Without Performance Loss. (arXiv:2305.15894v1 [cs.CL])

    [http://arxiv.org/abs/2305.15894](http://arxiv.org/abs/2305.15894)

    本文研究了在差分隐私约束下的会议摘要问题，发现差分隐私虽然会稍微降低性能，但在评估未见过的会议类型时却能提高性能，这一发现使得安全的会议摘要更加可行。

    

    会议摘要具有巨大的商业潜力，但除了是难题外，隐私问题也是一个挑战。我们探讨了在差分隐私约束下的会议摘要问题，并惊讶地发现，虽然差分隐私会导致样本内数据的性能略有降低，但在未见过的会议类型上评估时，差分隐私会提高性能。由于在实际应用场景中，会议摘要系统将遇到各种各样的会议类型，这一发现使得安全的会议摘要似乎更加可行。我们进行了广泛的误差分析，并识别了在差分隐私下进行会议摘要的潜在风险，包括忠实度分析。

    Meeting summarization has an enormous business potential, but in addition to being a hard problem, roll-out is challenged by privacy concerns. We explore the problem of meeting summarization under differential privacy constraints and find, to our surprise, that while differential privacy leads to slightly lower performance on in-sample data, differential privacy improves performance when evaluated on unseen meeting types. Since meeting summarization systems will encounter a great variety of meeting types in practical employment scenarios, this observation makes safe meeting summarization seem much more feasible. We perform extensive error analysis and identify potential risks in meeting summarization under differential privacy, including a faithfulness analysis.
    
[^10]: CSS: 一个大规模跨模式中文文本到SQL的医学数据集

    CSS: A Large-scale Cross-schema Chinese Text-to-SQL Medical Dataset. (arXiv:2305.15891v1 [cs.CL])

    [http://arxiv.org/abs/2305.15891](http://arxiv.org/abs/2305.15891)

    CSS是一个大规模跨模式中文文本到SQL的医学数据集，以解决现实应用中的跨领域文本到SQL难题。它包括2个数据库中的4,340个问题/SQL对和19个新数据库的29,280个对应的数据集示例。

    

    跨领域文本到SQL任务旨在构建一个系统，该系统可以将用户问题解析为SQL，这些数据库是完全未见过的，在同一领域内进行跨模式文本到SQL任务以解决现实应用中的难题。为此，我们介绍了跨模式文本到SQL任务，并提出了CSS，一个大规模的跨模式中文文本到SQL数据集，来开展相应的研究。CSS最初由2个数据库中的4,340个问题/SQL对组成。为了将模型推广到不同的医疗系统，我们扩展了CSS并创建了19个新数据库以及29,280个相应的数据集示例。此外，CSS 还是进行单领域中文文本到SQL研究的大型语料库。我们介绍了数据收集方法和一系列数据统计分析。

    The cross-domain text-to-SQL task aims to build a system that can parse user questions into SQL on complete unseen databases, and the single-domain text-to-SQL task evaluates the performance on identical databases. Both of these setups confront unavoidable difficulties in real-world applications. To this end, we introduce the cross-schema text-to-SQL task, where the databases of evaluation data are different from that in the training data but come from the same domain. Furthermore, we present CSS, a large-scale CrosS-Schema Chinese text-to-SQL dataset, to carry on corresponding studies. CSS originally consisted of 4,340 question/SQL pairs across 2 databases. In order to generalize models to different medical systems, we extend CSS and create 19 new databases along with 29,280 corresponding dataset examples. Moreover, CSS is also a large corpus for single-domain Chinese text-to-SQL studies. We present the data collection approach and a series of analyses of the data statistics. To show 
    
[^11]: LFTK: 计算语言学中的手工特征

    LFTK: Handcrafted Features in Computational Linguistics. (arXiv:2305.15878v1 [cs.CL])

    [http://arxiv.org/abs/2305.15878](http://arxiv.org/abs/2305.15878)

    该论文收集和分类了超过220个受欢迎的手工语言特征，设计了一个多语言的手工语言特征提取系统，以系统性的可扩展方式实现，并在几个任务特定的数据集上进行了相关性分析研究。

    

    过去的研究已经鉴定出了一组丰富的手工语言特征，可以潜在地帮助各种任务。但是，由于这些特征数量庞大，因此难以有效地选择和利用现有的手工特征。加上在研究工作中实现不一致的问题，目前还不存在分类方案或者统一接受的特征名称，这造成了不必要的混乱。此外，大多数现有的手工特征提取库都不是开源的，或者没有得到积极的维护。因此，研究人员经常需要从零开始构建这样的提取系统。我们通过过去的文献收集和分类了超过220个受欢迎的手工特征。然后，我们在几个任务特定的数据集上进行了相关性分析研究，并报告了每个特征的潜在用途。最后，我们设计了一个多语言的手工语言特征提取系统，以系统性的可扩展方式实现。我们开源了我们的系统。

    Past research has identified a rich set of handcrafted linguistic features that can potentially assist various tasks. However, their extensive number makes it difficult to effectively select and utilize existing handcrafted features. Coupled with the problem of inconsistent implementation across research works, there has been no categorization scheme or generally-accepted feature names. This creates unwanted confusion. Also, most existing handcrafted feature extraction libraries are not open-source or not actively maintained. As a result, a researcher often has to build such an extraction system from the ground up.  We collect and categorize more than 220 popular handcrafted features grounded on past literature. Then, we conduct a correlation analysis study on several task-specific datasets and report the potential use cases of each feature. Lastly, we devise a multilingual handcrafted linguistic feature extraction system in a systematically expandable manner. We open-source our system
    
[^12]: 真实回答的语言特性研究

    Linguistic Properties of Truthful Response. (arXiv:2305.15875v1 [cs.CL])

    [http://arxiv.org/abs/2305.15875](http://arxiv.org/abs/2305.15875)

    该论文研究了LLM的不真实回答现象，发现GPT-3模型对给定提示的回答在语言特性上很相似。同时，该论文证明了在没有评估内容本身的情况下，仅依赖模型回答的风格成分即可分类真实性。

    

    我们使用220个手工制作的语言特性对LLM不真实回答的现象进行了研究。我们专注于GPT-3模型，并发现不同大小的LLM对给定提示的回答在语言特性上很相似。我们通过训练只依赖模型回答的风格成分来分类陈述真实性的支持向量机扩展了这一发现。虽然数据集大小限制了我们的当前研究成果，但我们提供了有希望的证据，表明可以在不评估内容本身的情况下检测真实性。

    We investigate the phenomenon of an LLM's untruthful response using a large set of 220 handcrafted linguistic features. We focus on GPT-3 models and find that the linguistic profiles of responses are similar across model sizes. That is, how varying-sized LLMs respond to given prompts stays similar on the linguistic properties level. We expand upon this finding by training support vector machines that rely only upon the stylistic components of model responses to classify the truthfulness of statements. Though the dataset size limits our current findings, we present promising evidence that truthfulness detection is possible without evaluating the content itself.
    
[^13]: Jointprop：基于异构图传播的实体和关系联合半监督学习

    Jointprop: Joint Semi-supervised Learning for Entity and Relation Extraction with Heterogeneous Graph-based Propagation. (arXiv:2305.15872v1 [cs.CL])

    [http://arxiv.org/abs/2305.15872](http://arxiv.org/abs/2305.15872)

    提出了Jointprop框架，用于基于异构图传播的联合半监督实体和关系提取，采用一个统一的异构图来使用未标记数据中的全局结构信息，提高实体和关系提取性能。

    

    半监督学习是从有限数据中提取实体和关系的重要方法。然而，当前的半监督方法分别处理命名实体识别和关系抽取两个任务，并忽略实体和关系实例之间的交叉相关性以及未标记数据中相似实例的存在。为了解决这些问题，我们提出了Jointprop，一种基于异构图传播的联合半监督实体和关系提取框架，其捕获各个任务之间的全局结构信息，并利用未标记数据中的交互。具体地，我们从实体和关系候选构建了一个统一的基于标记的异构图，并根据置信度得分传播类标签。然后，我们采用传播学习方案来利用标记和未标记样本之间的关联。在基准数据集上的实验表明，我们的框架表现出令人满意的实体和关系提取性能。

    Semi-supervised learning has been an important approach to address challenges in extracting entities and relations from limited data. However, current semi-supervised works handle the two tasks (i.e., Named Entity Recognition and Relation Extraction) separately and ignore the cross-correlation of entity and relation instances as well as the existence of similar instances across unlabeled data. To alleviate the issues, we propose Jointprop, a Heterogeneous Graph-based Propagation framework for joint semi-supervised entity and relation extraction, which captures the global structure information between individual tasks and exploits interactions within unlabeled data. Specifically, we construct a unified span-based heterogeneous graph from entity and relation candidates and propagate class labels based on confidence scores. We then employ a propagation learning scheme to leverage the affinities between labelled and unlabeled samples. Experiments on benchmark datasets show that our framewo
    
[^14]: 技术领域术语和短语的文本表征提取

    Extracting Text Representations for Terms and Phrases in Technical Domains. (arXiv:2305.15867v1 [cs.CL])

    [http://arxiv.org/abs/2305.15867](http://arxiv.org/abs/2305.15867)

    本文提出了一种无监督的文本编码方法，使用小型基于字符的模型重构大型预训练嵌入矩阵，其可以在技术领域内达到与句子编码器相同的质量，但大小为后者的五分之一，计算时间能快10倍。

    

    获取术语和短语的密集表示是面向高度技术领域的知识发现平台的重要任务。常用的方法包括使用自监督设置训练领域特定的嵌入或使用训练过相似性任务的句子编码器模型。本文提出了一种完全无监督的文本编码方法，其中包括使用小型基于字符的模型来重构大型预训练嵌入矩阵。与静态嵌入相比，句子编码器不会受到词汇外问题的影响，但会带来显著的计算成本。

    Extracting dense representations for terms and phrases is a task of great importance for knowledge discovery platforms targeting highly-technical fields. Dense representations are used as features for downstream components and have multiple applications ranging from ranking results in search to summarization. Common approaches to create dense representations include training domain-specific embeddings with self-supervised setups or using sentence encoder models trained over similarity tasks. In contrast to static embeddings, sentence encoders do not suffer from the out-of-vocabulary (OOV) problem, but impose significant computational costs. In this paper, we propose a fully unsupervised approach to text encoding that consists of training small character-based models with the objective of reconstructing large pre-trained embedding matrices. Models trained with this approach can not only match the quality of sentence encoders in technical domains, but are 5 times smaller and up to 10 tim
    
[^15]: 顺序Integrated Gradients：一种解释语言模型的简单而有效的方法。

    Sequential Integrated Gradients: a simple but effective method for explaining language models. (arXiv:2305.15853v1 [cs.CL])

    [http://arxiv.org/abs/2305.15853](http://arxiv.org/abs/2305.15853)

    本文提出了一种名为顺序 Integrated Gradients（SIG）的解释语言模型的新方法，通过保持其他单词不变，仅在基线和感兴趣的单词之间创建插值来计算句子中每个单词的重要性，并用训练的令牌“mask”替换基线令牌“pad”来显着改善解释效果。

    

    几种解释方法（例如Integrated Gradients（IG））可以被描述为基于路径的方法，因为它们依赖于数据和无信息基线之间的直线。然而，当应用于语言模型时，这些方法同时为每个句子单词量产生路径，这可能会导致从插值词生成的句子没有明确的含义，或者与原始句子相比有显着不同的含义。为了使这些句子的含义尽可能接近原始句子，我们提出了顺序Integrated Gradients（SIG），它通过保持其他单词不变，仅在基线和感兴趣的单词之间创建插值来计算句子中每个单词的重要性。此外，受到几个语言模型的训练过程的启发，我们还建议用训练的令牌“mask”替换基线令牌“pad”。虽然这只是对原始IG方法的简单改进，但效果显著。

    Several explanation methods such as Integrated Gradients (IG) can be characterised as path-based methods, as they rely on a straight line between the data and an uninformative baseline. However, when applied to language models, these methods produce a path for each word of a sentence simultaneously, which could lead to creating sentences from interpolated words either having no clear meaning, or having a significantly different meaning compared to the original sentence. In order to keep the meaning of these sentences as close as possible to the original one, we propose Sequential Integrated Gradients (SIG), which computes the importance of each word in a sentence by keeping fixed every other words, only creating interpolations between the baseline and the word of interest. Moreover, inspired by the training procedure of several language models, we also propose to replace the baseline token "pad" with the trained token "mask". While being a simple improvement over the original IG method
    
[^16]: 大型语言模型的自相矛盾幻觉：评估、检测和缓解

    Self-contradictory Hallucinations of Large Language Models: Evaluation, Detection and Mitigation. (arXiv:2305.15852v1 [cs.CL])

    [http://arxiv.org/abs/2305.15852](http://arxiv.org/abs/2305.15852)

    本文对大型语言模型的自相矛盾幻觉进行了评估、检测和缓解，探究了这一幻觉形式的普遍存在性。通过设计框架有效触发自相矛盾，发现不同语言模型中这种现象都频繁出现。ChatGPT和GPT-4能够准确识别自相矛盾，而Vicuna-13B则有些困难。

    

    大型语言模型容易产生幻想的文本。自相矛盾是一种重要的幻觉形式，指的是语言模型在同一语境中生成两个矛盾的句子。本文针对最先进、经过指导的语言模型，对自相矛盾进行了全面的分析、评估、检测和缓解。我们设计了一个框架来有效地触发自相矛盾，评估结果表明，无论是对于著名的还是不太出名的话题，不同的语言模型中自相矛盾都经常发生。

    Large language models (large LMs) are susceptible to producing text with hallucinated content. Self-contradiction, where the LM generates two contradictory sentences within the same context, is an important form of hallucination. In this work, we present a comprehensive analysis on self-contradiction for state-of-the-art, instruction-tuned LMs, including evaluation, detection, and mitigation. To effectively trigger self-contradictions, we design a framework that constrains LMs to generate appropriate sentence pairs. Our evaluation on these sentence pairs reveals that self-contradictions occur frequently across different LMs for both famous and lesser-known topics. Next, we prompt the LMs to detect self-contradictions. Our results indicate that ChatGPT and GPT-4 are able to accurately identify self-contradictions, while Vicuna-13B struggles to do so. For example, with our best prompting method, ChatGPT achieves 91.0% precision and 80.5% recall on the sentence pairs generated by itself. 
    
[^17]: Bhasha-Abhijnaanam：22种印度文字和罗马拼音语言鉴别。 (arXiv：2305.15814v1 [cs.CL])

    Bhasha-Abhijnaanam: Native-script and romanized Language Identification for 22 Indic languages. (arXiv:2305.15814v1 [cs.CL])

    [http://arxiv.org/abs/2305.15814](http://arxiv.org/abs/2305.15814)

    该研究提供了22种印度宪法中列出的所有21种本土文字和罗马字母的公开语言鉴别（LID）数据和模型。IndicLID是上述语言的本土和罗马化脚本的语言鉴别器，还提出了解决罗马化文本的LID问题的方案。

    

    我们提供了22个印度宪法中列出的所有21种本土文字和罗马字母的公开语言鉴别（LID）数据和模型。与现有的LID相比，我们的Bhasha-Abhijnaanam在本土文字文本的语言涵盖范围方面更为广泛，并具有竞争力或更好的性能，IndicLID是上述语言的本土和罗马化脚本的语言鉴别器。对于罗马化文本的LID，存在两个主要挑战：缺乏训练数据和当语言相似时，低LID性能。我们提供了简单有效的解决方案。总的来说，在任何语言中，罗马化文本的研究都很有限，我们的研究结果对需要罗马化语言鉴别的其他语言也具有参考意义。

    We create publicly available language identification (LID) datasets and models in all 22 Indian languages listed in the Indian constitution in both native-script and romanized text. First, we create Bhasha-Abhijnaanam, a language identification test set for native-script as well as romanized text which spans all 22 Indic languages. We also train IndicLID, a language identifier for all the above-mentioned languages in both native and romanized script. For native-script text, it has better language coverage than existing LIDs and is competitive or better than other LIDs. IndicLID is the first LID for romanized text in Indian languages. Two major challenges for romanized text LID are the lack of training data and low-LID performance when languages are similar. We provide simple and effective solutions to these problems. In general, there has been limited work on romanized text in any language, and our findings are relevant to other languages that need romanized language identification. Ou
    
[^18]: 动态上下文剪枝用于高效和可解释的自回归变换器

    Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers. (arXiv:2305.15805v1 [cs.CL])

    [http://arxiv.org/abs/2305.15805](http://arxiv.org/abs/2305.15805)

    本研究提出了一种动态上下文剪枝方法，可以在保持模型表现力的同时，动态减少无效信息，提高模型的效率和可解释性。该技术可以应用于现有的预训练模型，并且可以通过简单的微调过程实现。

    

    大型语言模型中采用的自回归变换器难以扩展到长序列。尽管有几项工作试图减少它们的计算成本，但大多数LLM仍然在所有标记对之间采用注意层，从而产生二次成本。本研究提出了一种新方法，通过保留模型的表现力来动态修剪上下文信息，从而在推理过程中减少内存和计算要求。我们的方法使用可学习机制，在生成过程中确定哪些无关的标记可以从上下文中删除。通过这样做，我们的方法不仅解决了性能问题，而且增强了可解释性，为模型的决策过程提供了宝贵的洞察力。我们的技术可以通过简单的微调过程应用于现有的预训练模型，并且剪枝强度可以由稀疏度参数指定。

    Autoregressive Transformers adopted in Large Language Models (LLMs) are hard to scale to long sequences. Despite several works trying to reduce their computational cost, most of LLMs still adopt attention layers between all pairs of tokens in the sequence, thus incurring a quadratic cost. In this study, we present a novel approach that dynamically prunes contextual information while preserving the model's expressiveness, resulting in reduced memory and computational requirements during inference. Our method employs a learnable mechanism that determines which uninformative tokens can be dropped from the context at any point across the generation process. By doing so, our approach not only addresses performance concerns but also enhances interpretability, providing valuable insight into the model's decision-making process. Our technique can be applied to existing pre-trained models through a straightforward fine-tuning process, and the pruning strength can be specified by a sparsity para
    
[^19]: MERGE: 快速的私有文本生成

    MERGE: Fast Private Text Generation. (arXiv:2305.15769v1 [cs.CL])

    [http://arxiv.org/abs/2305.15769](http://arxiv.org/abs/2305.15769)

    该论文提出了MERGE，一个基于Transformer语言模型的快速私有文本生成框架。实验结果表明，MERGE在保护隐私的同时，实现了26.5倍的加速和80%的通信字节数减少。

    

    近年来，人们越来越关注NLP服务和Transformer模型的私有推理。然而，现有的两方隐私保护方法仅考虑NLU场景，而文本生成的私有推理，如翻译、对话和代码补全，仍未解决。此外，将现有的隐私保护方法迁移到NLG模型时，性能表现差，而在训练阶段受到收敛问题的困扰。为了解决这些问题，我们提出了MERGE，这是一个基于Transformer语言模型的快速私有文本生成框架。具体而言，MERGE重用输出隐藏状态作为单词嵌入，以跳过嵌入计算，并重新组织Transformer模块中的线性操作以加速向前过程。基于这两个优化，大量的实验表明，在序列长度为512时，MERGE可实现26.5倍的加速，并减少80\%的通信字节数。

    Recent years have seen increasing concerns about the private inference of NLP services and Transformer models. However, existing two-party privacy-preserving methods solely consider NLU scenarios, while the private inference of text generation such as translation, dialogue, and code completion remains unsolved. Besides, while migrated to NLG models, existing privacy-preserving methods perform poorly in terms of inference speed, and suffer from the convergence problem during the training stage. To address these issues, we propose MERGE, a fast private text generation framework for Transformer-based language models. Specifically, MERGE reuse the output hidden state as the word embedding to bypass the embedding computation, and reorganize the linear operations in the Transformer module to accelerate the forward procedure. Based on these two optimizations, extensive experiments show that MERGE can achieve a 26.5x speedup under the sequence length 512, and reduce 80\% communication bytes, w
    
[^20]: Svarah: 在印度口音上评估英语语音识别系统

    Svarah: Evaluating English ASR Systems on Indian Accents. (arXiv:2305.15760v1 [cs.CL])

    [http://arxiv.org/abs/2305.15760](http://arxiv.org/abs/2305.15760)

    Svarah是一个新的基准测试，包含印度65个不同地理位置上的117个说话者的9.6小时的英语音频转录。该基准测试表明，现有的英语ASR系统在印度口音上需要改进。

    

    印度是世界上第二大讲英语的国家，其使用者约有1.3亿人。因此，对于英语的自动语音识别(ASR)系统来说，对印度口音的评估是至关重要的。然而，印度说话者在现有的英语ASR基准测试中，如LibriSpeech、Switchboard、Speech Accent Archive等，得到的代表性非常差。在这项工作中，我们通过创建Svarah来解决这一缺口，该基准测试包含来自印度65个地理位置上的117个说话者的9.6小时的英语音频转录，具有各种口音和领域的阅读和会话数据，如历史、文化、旅游等，确保了词汇的多样性。我们在Svarah上评估了6个开放源代码ASR模型和2个商业ASR系统，并表明印度口音上存在改进的明显空间。Svarah和我们的所有代码都将公开提供。

    India is the second largest English-speaking country in the world with a speaker base of roughly 130 million. Thus, it is imperative that automatic speech recognition (ASR) systems for English should be evaluated on Indian accents. Unfortunately, Indian speakers find a very poor representation in existing English ASR benchmarks such as LibriSpeech, Switchboard, Speech Accent Archive, etc. In this work, we address this gap by creating Svarah, a benchmark that contains 9.6 hours of transcribed English audio from 117 speakers across 65 geographic locations throughout India, resulting in a diverse range of accents. Svarah comprises both read speech and spontaneous conversational data, covering various domains, such as history, culture, tourism, etc., ensuring a diverse vocabulary. We evaluate 6 open source ASR models and 2 commercial ASR systems on Svarah and show that there is clear scope for improvement on Indian accents. Svarah as well as all our code will be publicly available.
    
[^21]: 利用弱监督信号治愈不安全的对话回复

    Healing Unsafe Dialogue Responses with Weak Supervision Signals. (arXiv:2305.15757v1 [cs.CL])

    [http://arxiv.org/abs/2305.15757](http://arxiv.org/abs/2305.15757)

    本论文提出了一种名为TEMP的无监督伪标签采样方法，可以自动分配潜在的安全响应，解决大规模对话系统的不安全响应生成问题。

    

    近年来，人们对大规模对话系统的不安全响应生成越来越担忧，代理会从现实世界语料库中学习攻击性或有偏见的行为。提出了一些方法通过检测和替换不安全的训练样本来解决上述问题，虽然有效，但它们的注释成本高，并且在未见过的场景以及对抗性攻击方面适应性差。此外，忽略提供安全的响应（例如简单地替换模板）将导致对话信息缺失的问题。为了解决这些问题，我们提出了一种无监督的伪标签采样方法TEMP，可以自动分配潜在的安全响应。具体而言，我们的TEMP方法将响应分成几个簇，并使用一种自适应锐化采样策略进行多标签采样，灵感来自于不安全样本通常少且分布在尾部的观察。

    Recent years have seen increasing concerns about the unsafe response generation of large-scale dialogue systems, where agents will learn offensive or biased behaviors from the real-world corpus. Some methods are proposed to address the above issue by detecting and replacing unsafe training examples in a pipeline style. Though effective, they suffer from a high annotation cost and adapt poorly to unseen scenarios as well as adversarial attacks. Besides, the neglect of providing safe responses (e.g. simply replacing with templates) will cause the information-missing problem of dialogues. To address these issues, we propose an unsupervised pseudo-label sampling method, TEMP, that can automatically assign potential safe responses. Specifically, our TEMP method groups responses into several clusters and samples multiple labels with an adaptively sharpened sampling strategy, inspired by the observation that unsafe samples in the clusters are usually few and distribute in the tail. Extensive 
    
[^22]: UniTRec: 一个统一的文本到文本变换器和联合对比学习框架用于基于文本的推荐

    UniTRec: A Unified Text-to-Text Transformer and Joint Contrastive Learning Framework for Text-based Recommendation. (arXiv:2305.15756v1 [cs.CL])

    [http://arxiv.org/abs/2305.15756](http://arxiv.org/abs/2305.15756)

    UniTRec是一个文本到文本的推荐框架，采用了统一的局部-全局注意力Transformer编码器来处理用户历史的上下文，并且使用Transformer解码器的语言困惑度来构建对比信号，可以显著提高性能。

    

    先前的研究表明，预训练语言模型（PLM）可以提高基于文本的推荐性能。与以往工作不同，我们提出了一个统一的局部-全局注意力Transformer编码器，以更好地建模用户历史的两个层次的上下文。此外，在基于Transformer编码器编码的用户历史的条件下，我们的框架利用Transformer解码器估计候选文本项的语言困惑度，这可以作为用户-物品文本匹配的简单而重要的对比信号。基于此，我们的框架UniTRec将区分性匹配得分和候选文本困惑度的对比目标统一起来，以共同增强基于文本的推荐。广泛的评估表明，UniTRec在三个基于文本的推荐任务上提供了SOTA性能。代码可在https://anonymous.com上找到。

    Prior study has shown that pretrained language models (PLM) can boost the performance of text-based recommendation. In contrast to previous works that either use PLM to encode user history as a whole input text, or impose an additional aggregation network to fuse multi-turn history representations, we propose a unified local- and global-attention Transformer encoder to better model two-level contexts of user history. Moreover, conditioned on user history encoded by Transformer encoders, our framework leverages Transformer decoders to estimate the language perplexity of candidate text items, which can serve as a straightforward yet significant contrastive signal for user-item text matching. Based on this, our framework, UniTRec, unifies the contrastive objectives of discriminative matching scores and candidate text perplexity to jointly enhance text-based recommendation. Extensive evaluation shows that UniTRec delivers SOTA performance on three text-based recommendation tasks. Code is a
    
[^23]: 使用转写法的突厥语多语种文本转语音合成

    Multilingual Text-to-Speech Synthesis for Turkic Languages Using Transliteration. (arXiv:2305.15749v1 [eess.AS])

    [http://arxiv.org/abs/2305.15749](http://arxiv.org/abs/2305.15749)

    通过使用转写法，本研究建立了一个多语种文本转语音合成系统，针对十种突厥语言进行研究，使用Tacotron 2架构的TTS系统，仅使用哈萨克语训练数据，可以实现零样本学习并生成其他突厥语族的语音。

    

    本文旨在建立一个多语种文本转语音（TTS）合成系统，针对十种资源匮乏的突厥语言进行研究：阿塞拜疆语、巴什基尔语、哈萨克语、柯尔克孜语、萨哈语、鞑靼语、土耳其语、土库曼语、维吾尔语和乌兹别克语。特别针对零样本学习场景，即使用一个语言的数据训练TTS模型，来合成未经见过的语言的语音。我们采用Tacotron 2架构的端到端TTS系统，仅使用哈萨克语的现有数据训练，然后通过将突厥语的字母映射到国际语音符号（IPA）的符号，再转换为哈萨克语的字母，来为其他突厥语言生成语音。为了证明所提出的方法的可行性，我们主观地评估了多语种突厥TTS模型，并获得了令人鼓舞的结果。为了使实验可复制，我们在GitHub存储库中公开了我们的代码和数据集。

    This work aims to build a multilingual text-to-speech (TTS) synthesis system for ten lower-resourced Turkic languages: Azerbaijani, Bashkir, Kazakh, Kyrgyz, Sakha, Tatar, Turkish, Turkmen, Uyghur, and Uzbek. We specifically target the zero-shot learning scenario, where a TTS model trained using the data of one language is applied to synthesise speech for other, unseen languages. An end-to-end TTS system based on the Tacotron 2 architecture was trained using only the available data of the Kazakh language. To generate speech for the other Turkic languages, we first mapped the letters of the Turkic alphabets onto the symbols of the International Phonetic Alphabet (IPA), which were then converted to the Kazakh alphabet letters. To demonstrate the feasibility of the proposed approach, we evaluated the multilingual Turkic TTS model subjectively and obtained promising results. To enable replication of the experiments, we make our code and dataset publicly available in our GitHub repository.
    
[^24]: 学习不链接：探索实体链接中的NIL预测

    Learn to Not Link: Exploring NIL Prediction in Entity Linking. (arXiv:2305.15725v1 [cs.CL])

    [http://arxiv.org/abs/2305.15725](http://arxiv.org/abs/2305.15725)

    该论文提出了一个实体链接数据集NEL及其对NIL预测问题的分类方法，研究结果表明在训练数据中，缺失实体和非实体短语均对NIL预测的准确性具有显著影响。

    

    实体链接模型通过利用预训练的语言模型捕捉语义特征已取得重大成功，然而对于寻找没有相应知识库实体的提及的NIL预测问题尚未得到足够关注。我们将链接到NIL的提及分为缺失实体和非实体短语，并提出了一个实体链接数据集NEL，重点关注NIL预测问题。NEL以不明确的实体作为种子，在维基百科语料库中收集相关的提及上下文，并通过人工注释和实体屏蔽确保链接到NIL的提及的存在。我们使用广泛使用的双编码器和交叉编码器实体链接模型进行了一系列实验，结果表明在训练数据中，这两种类型的NIL提及对NIL预测的准确性有显着影响。我们的代码和数据集可以在 https://github.com/solitaryzero/NIL_EL 上获取。

    Entity linking models have achieved significant success via utilizing pretrained language models to capture semantic features. However, the NIL prediction problem, which aims to identify mentions without a corresponding entity in the knowledge base, has received insufficient attention. We categorize mentions linking to NIL into Missing Entity and Non-Entity Phrase, and propose an entity linking dataset NEL that focuses on the NIL prediction problem. NEL takes ambiguous entities as seeds, collects relevant mention context in the Wikipedia corpus, and ensures the presence of mentions linking to NIL by human annotation and entity masking. We conduct a series of experiments with the widely used bi-encoder and cross-encoder entity linking models, results show that both types of NIL mentions in training data have a significant influence on the accuracy of NIL prediction. Our code and dataset can be accessed at https://github.com/solitaryzero/NIL_EL
    
[^25]: 面向代码混合的印地语-英语数据的预训练BERT模型的比较研究

    Comparative Study of Pre-Trained BERT Models for Code-Mixed Hindi-English Data. (arXiv:2305.15722v1 [cs.CL])

    [http://arxiv.org/abs/2305.15722](http://arxiv.org/abs/2305.15722)

    本文比较了使用不同预训练Transformer模型的印地语-英语代码混合数据的性能表现，以提高情感分析、情绪识别和仇恨言论识别等自然语言处理任务的性能。

    

    “代码混合”是指在同一段文本中使用多种语言的现象。这种现象在社交媒体平台上广泛存在，并随着时间的推移越来越多地被采纳。检测语言中的外来元素并正确处理它们至关重要，因为许多人使用代码混合语言，其中任一语言都无法理解。本文重点研究低资源的印地语-英语代码混合语言，并提高不同代码混合自然语言处理任务（如情感分析、情绪识别和仇恨言论识别）的性能。我们对使用无监督方法预训练的不同基于Transformer的语言模型进行了比较分析。我们包括了代码混合模型（如HingBERT、HingRoBERTa、HingRoBERTa-Mixed、mBERT）和非代码混合模型（如AlBERT、BERT、RoBERTa），进行比较分析印地语-英语代码混合。

    The term "Code Mixed" refers to the use of more than one language in the same text. This phenomenon is predominantly observed on social media platforms, with an increasing amount of adaptation as time goes on. It is critical to detect foreign elements in a language and process them correctly, as a considerable number of individuals are using code-mixed languages that could not be comprehended by understanding one of those languages. In this work, we focus on low-resource Hindi-English code-mixed language and enhancing the performance of different code-mixed natural language processing tasks such as sentiment analysis, emotion recognition, and hate speech identification. We perform a comparative analysis of different Transformer-based language Models pre-trained using unsupervised approaches. We have included the code-mixed models like HingBERT, HingRoBERTa, HingRoBERTa-Mixed, mBERT, and non-code-mixed models like AlBERT, BERT, and RoBERTa for comparative analysis of code-mixed Hindi-En
    
[^26]: 走向更高的多语言机器翻译帕累托前沿

    Towards Higher Pareto Frontier in Multilingual Machine Translation. (arXiv:2305.15718v1 [cs.CL])

    [http://arxiv.org/abs/2305.15718](http://arxiv.org/abs/2305.15718)

    本文提出了一种新的训练框架——帕累托互攻（Pareto-MD），旨在将帕累托前沿向外推进，而不是进行权衡，以提高多语言机器翻译性能。

    

    近年来，多语言神经机器翻译取得了显著进展。然而，多语料库的长尾分布形成了帕累托最优化的挑战，即为了优化某些语言的翻译，可能损害其他语言的性能。现有的平衡训练策略等同于一系列帕累托最优解，它们在帕累托前沿上进行权衡。在本文中，我们提出了一种新的训练框架——帕累托互攻（Pareto-MD），旨在将帕累托前沿向外推进，而不是进行权衡。具体来说，Pareto-MD共同训练两个偏向不同语言的帕累托最优解，并通过知识蒸馏让它们相互学习优点。此外，我们还引入了一种新的策略，以实现更强大的帕累托最优解之间的通信，拓宽了我们方法的适用范围。实验结果表明，在广泛使用的WMT和TED数据集上，我们的方法都获得了显著的改进。

    Multilingual neural machine translation has witnessed remarkable progress in recent years. However, the long-tailed distribution of multilingual corpora poses a challenge of Pareto optimization, i.e., optimizing for some languages may come at the cost of degrading the performance of others. Existing balancing training strategies are equivalent to a series of Pareto optimal solutions, which trade off on a Pareto frontier. In this work, we propose a new training framework, Pareto Mutual Distillation (Pareto-MD), towards pushing the Pareto frontier outwards rather than making trade-offs. Specifically, Pareto-MD collaboratively trains two Pareto optimal solutions that favor different languages and allows them to learn from the strengths of each other via knowledge distillation. Furthermore, we introduce a novel strategy to enable stronger communication between Pareto optimal solutions and broaden the applicability of our approach. Experimental results on the widely-used WMT and TED dataset
    
[^27]: 模仿专有语言模型的错误承诺

    The False Promise of Imitating Proprietary LLMs. (arXiv:2305.15717v1 [cs.CL])

    [http://arxiv.org/abs/2305.15717](http://arxiv.org/abs/2305.15717)

    本文分析了使用较弱开源模型模仿专有语言模型的可靠度，发现在某些情况下，表现可能非常出色，但在大多数任务中仍无法取代专有语言模型。

    

    一种提高较弱语言模型性能的新方法是基于较强模型的输出进行微调，例如专有系统ChatGPT（例如Alpaca、Self-Instruct等）。这种方法旨在使用较弱的开源模型廉价地模仿专有模型的能力。本文对这种方法进行了批判性分析。我们首先 使用不同的基础模型大小（1.5B-13B）、数据源和模仿数据量（0.3M-150M令牌）来微调一系列模仿ChatGPT的LM。然后，我们使用众包评估和规范的NLP基准对模型进行评估。最初，我们对模仿模型的输出质量感到惊讶——它们似乎更擅长按照指示进行操作，并且众包工作者将它们的输出评为与ChatGPT具有竞争力。然而，当进行更有针对性的自动评估时，我们发现，与未在模仿模型中得到充分支持的任务相比，模仿模型在缩小基础语言模型与ChatGPT之间差距方面帮助不大。

    An emerging method to cheaply improve a weaker language model is to finetune it on outputs from a stronger model, such as a proprietary system like ChatGPT (e.g., Alpaca, Self-Instruct, and others). This approach looks to cheaply imitate the proprietary model's capabilities using a weaker open-source model. In this work, we critically analyze this approach. We first finetune a series of LMs that imitate ChatGPT using varying base model sizes (1.5B--13B), data sources, and imitation data amounts (0.3M--150M tokens). We then evaluate the models using crowd raters and canonical NLP benchmarks. Initially, we were surprised by the output quality of our imitation models -- they appear far better at following instructions, and crowd workers rate their outputs as competitive with ChatGPT. However, when conducting more targeted automatic evaluations, we find that imitation models close little to none of the gap from the base LM to ChatGPT on tasks that are not heavily supported in the imitation
    
[^28]: 克服提示扰动敏感性的零样本方法

    Zero-shot Approach to Overcome Perturbation Sensitivity of Prompts. (arXiv:2305.15689v1 [cs.CL])

    [http://arxiv.org/abs/2305.15689](http://arxiv.org/abs/2305.15689)

    本研究提出了一种零样本方法，自动生成多个类似于基础提示的高质量提示，并使用新的度量方法进行排名，从而克服了提示的扰动敏感性，并在情感分类任务中具有较高的准确性。

    

    最近的研究表明，自然语言提示可以帮助利用预训练语言模型学习的知识进行二元句级情感分类任务。具体来说，这些方法利用少量样本学习设置，使用手动或自动生成的提示来微调情感分类模型。然而，这些方法的性能对所使用提示的扰动敏感。此外，这些方法依赖于少量带标签实例进行自动提示生成和提示排序。本研究旨在在零样本设置中为所给定的任务找到高质量的提示。我们的提议方法给定一个基础提示，采用位置、推理和释义技术自动生成多个类似于基础提示的提示，然后使用一种新的度量方法对提示进行排名。我们从实验上证明，排名靠前的提示具有很高的质量，并在提示扰动鲁棒性和整体准确性方面显著优于基础提示和其他现有的提示生成方法。

    Recent studies have demonstrated that natural-language prompts can help to leverage the knowledge learned by pre-trained language models for the binary sentence-level sentiment classification task. Specifically, these methods utilize few-shot learning settings to fine-tune the sentiment classification model using manual or automatically generated prompts. However, the performance of these methods is sensitive to the perturbations of the utilized prompts. Furthermore, these methods depend on a few labeled instances for automatic prompt generation and prompt ranking. This study aims to find high-quality prompts for the given task in a zero-shot setting. Given a base prompt, our proposed approach automatically generates multiple prompts similar to the base prompt employing positional, reasoning, and paraphrasing techniques and then ranks the prompts using a novel metric. We empirically demonstrate that the top-ranked prompts are high-quality and significantly outperform the base prompt an
    
[^29]: RewriteLM：一种面向文本重写的指令调整大型语言模型。

    RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting. (arXiv:2305.15685v1 [cs.CL])

    [http://arxiv.org/abs/2305.15685](http://arxiv.org/abs/2305.15685)

    本文介绍了 RewriteLM，一种指令调整的大型语言模型，用于长篇文本重写。同时，我们提出了一个名为 OpenRewriteEval 的基准测试，用于评估各种类型的开放式长篇文本重写。我们采用新的策略来促进多样的指令和偏好数据生成，从而为长篇文本重写提供更好的评估手段。

    

    大型语言模型已经展现出在长篇文本生成任务中通过自然语言指令表达来的惊人的零-shot能力，然而用户对于长篇文本重写的期望值很高，模型产生的意外重写（“幻觉”）会对其整体性能产生负面影响。现有的评估基准主要关注有限的重写风格和句子级重写，而不是长篇开放式重写。我们引入了一个新的基准测试OpenRewriteEval，它涵盖了通过自然语言指令表达的各种重写类型。它特别设计用于促进长篇文本开放式重写的评估。此外，我们提出了一种强大的基线模型RewriteLM，一个用于长篇文本重写的指令调整大型语言模型。我们开发了一些新策略，以最小人工干预促进生成多样的指令和偏好数据。

    Large Language Models (LLMs) have demonstrated impressive zero-shot capabilities in long-form text generation tasks expressed through natural language instructions. However, user expectations for long-form text rewriting is high, and unintended rewrites (''hallucinations'') produced by the model can negatively impact its overall performance. Existing evaluation benchmarks primarily focus on limited rewriting styles and sentence-level rewriting rather than long-form open-ended rewriting.We introduce OpenRewriteEval, a novel benchmark that covers a wide variety of rewriting types expressed through natural language instructions. It is specifically designed to facilitate the evaluation of open-ended rewriting of long-form texts. In addition, we propose a strong baseline model, RewriteLM, an instruction-tuned large language model for long-form text rewriting. We develop new strategies that facilitate the generation of diverse instructions and preference data with minimal human intervention.
    
[^30]: 基于干扰的自我监督注意力用于文本分类中的注意偏差

    Perturbation-based Self-supervised Attention for Attention Bias in Text Classification. (arXiv:2305.15684v1 [cs.CL])

    [http://arxiv.org/abs/2305.15684](http://arxiv.org/abs/2305.15684)

    本论文提出了一种基于干扰的自我监督注意力方法来引导注意力学习，无需任何注释开销，能够在三个文本分类任务中显著提高当前基于注意力的模型的性能，该方法比现有的自我监督方法更有效。

    

    在文本分类中，传统的注意力机制通常过于关注频繁出现的单词，并且需要大量已注释的数据才能学习。本文提出了一种基于干扰的自我监督注意力方法来引导注意力学习，无需任何注释开销。具体而言，我们尽可能地添加噪声到句子中的所有单词，而不改变它们的语义和预测。我们假设能够容忍更多噪声的单词意义更不重要，并且可以使用该信息来优化注意力分布。在三个文本分类任务上的实验结果表明，我们的方法可以显著提高当前基于注意力的模型的性能，且比现有的自我监督方法更有效。我们还提供了可视化分析，以验证我们方法的有效性。

    In text classification, the traditional attention mechanisms usually focus too much on frequent words, and need extensive labeled data in order to learn. This paper proposes a perturbation-based self-supervised attention approach to guide attention learning without any annotation overhead. Specifically, we add as much noise as possible to all the words in the sentence without changing their semantics and predictions. We hypothesize that words that tolerate more noise are less significant, and we can use this information to refine the attention distribution. Experimental results on three text classification tasks show that our approach can significantly improve the performance of current attention-based models, and is more effective than existing self-supervised methods. We also provide a visualization analysis to verify the effectiveness of our approach.
    
[^31]: 重新审视非英语文本简化：一个统一的多语言基准

    Revisiting non-English Text Simplification: A Unified Multilingual Benchmark. (arXiv:2305.15678v1 [cs.CL])

    [http://arxiv.org/abs/2305.15678](http://arxiv.org/abs/2305.15678)

    这篇论文介绍了MultiSim基准，它包含了27个资源、12种语言超过1.7百万个复杂-简单的句子对。使用该基准进行预训练的多语言语言模型可以在非英语环境中带来令人兴奋的性能提升，并且俄语在跨语言转移方面表现出强大的性能。

    

    最近英语自动文本简化（ATS）研究中高质量、大规模的英语资源的进展将英语ATS研究的前沿推向了更高的水平。然而，由于缺乏一个覆盖多种语言中的复杂-简洁句子对的多样化评估基准，对多语言文本简化的研究工作较少。本文介绍了MultiSim基准，这是一个收集了12种不同语言中超过1.7百万个复杂-简单句子对的27个资源的集合。这个基准将鼓励研究开发更有效的多语言文本简化模型和评估指标。我们使用MultiSim与预训练的多语言语言模型进行的实验显示，在非英语环境中进行多语言训练可以带来令人兴奋的性能提升。我们观察到俄语在零-shot跨语言转移到低资源语言的强大表现。我们进一步展示使用BLOOM-176b的少量提示可以达到可比的参考简化质量。

    Recent advancements in high-quality, large-scale English resources have pushed the frontier of English Automatic Text Simplification (ATS) research. However, less work has been done on multilingual text simplification due to the lack of a diverse evaluation benchmark that covers complex-simple sentence pairs in many languages. This paper introduces the MultiSim benchmark, a collection of 27 resources in 12 distinct languages containing over 1.7 million complex-simple sentence pairs. This benchmark will encourage research in developing more effective multilingual text simplification models and evaluation metrics. Our experiments using MultiSim with pre-trained multilingual language models reveal exciting performance improvements from multilingual training in non-English settings. We observe strong performance from Russian in zero-shot cross-lingual transfer to low-resource languages. We further show that few-shot prompting with BLOOM-176b achieves comparable quality to reference simplif
    
[^32]: 用解释提高语法错误修正系统的能力

    Enhancing Grammatical Error Correction Systems with Explanations. (arXiv:2305.15676v1 [cs.CL])

    [http://arxiv.org/abs/2305.15676](http://arxiv.org/abs/2305.15676)

    该论文介绍了一个用解释提高语法错误修正系统能力的方法，通过引入包含证据单词和语法错误类型注释的大数据集，找到错误的原因，并提出了几个基线和分析方法来理解这个任务，同时也证明了解释可以帮助第二语言学习者更好地理解语法规则。

    

    语法校正系统通过检测和更正语言错误来提升书写交流。为了帮助语言学习者更好地理解GEC系统为什么做出某种更正，错误的原因（证据单词）和相应的错误类型是两个关键因素。为了用解释增强GEC系统，我们引入了EXPECT，一个大数据集，其中包含了证据单词和语法错误类型的注释。我们提出了几个基线和分析方法来理解这个任务。此外，人类评估证明，我们可解释的GEC系统的解释能够帮助第二语言学习者确定是否接受更正建议，并理解相关的语法规则。

    Grammatical error correction systems improve written communication by detecting and correcting language mistakes. To help language learners better understand why the GEC system makes a certain correction, the causes of errors (evidence words) and the corresponding error types are two key factors. To enhance GEC systems with explanations, we introduce EXPECT, a large dataset annotated with evidence words and grammatical error types. We propose several baselines and anlysis to understand this task. Furthermore, human evaluation verifies our explainable GEC system's explanations can assist second-language learners in determining whether to accept a correction suggestion and in understanding the associated grammar rule.
    
[^33]: BookGPT：一种基于大型语言模型的通用图书推荐框架

    BookGPT: A General Framework for Book Recommendation Empowered by Large Language Model. (arXiv:2305.15673v1 [cs.IR])

    [http://arxiv.org/abs/2305.15673](http://arxiv.org/abs/2305.15673)

    本文介绍了一种基于大型语言模型的通用图书推荐框架BookGPT，通过将生成式预训练变换器技术应用于图书推荐场景中的三种任务，即图书评分推荐、用户评分推荐和图书摘要推荐，实现了对图书推荐的有力改进。

    

    随着生成式预训练变换器（GPT）等大型语言模型技术的不断发展和变化，各个领域的许多经典场景重新展现出新的机遇。本文将ChatGPT作为建模对象，首次将LLM技术并入传统的图书资源理解和推荐场景中，并付诸实践。本文基于ChatGPT构建了类似于聊天机器人的图书推荐系统框架（BookGPT），试图将ChatGPT应用于三种典型任务的推荐建模：图书评分推荐，用户评分推荐和图书摘要推荐，探索LLM技术在图书推荐场景中的可行性。同时，本文根据不同的图书推荐任务评估方案和现有的经典推荐模型，讨论了BookGPT在图书推荐场景下的优缺点，并进行了一系列实证比较和分析，证明基于LLM技术的BookGPT框架可以为图书推荐领域带来显著的改进。

    With the continuous development and change exhibited by large language model (LLM) technology, represented by generative pretrained transformers (GPTs), many classic scenarios in various fields have re-emerged with new opportunities. This paper takes ChatGPT as the modeling object, incorporates LLM technology into the typical book resource understanding and recommendation scenario for the first time, and puts it into practice. By building a ChatGPT-like book recommendation system (BookGPT) framework based on ChatGPT, this paper attempts to apply ChatGPT to recommendation modeling for three typical tasks, book rating recommendation, user rating recommendation, and book summary recommendation, and explores the feasibility of LLM technology in book recommendation scenarios. At the same time, based on different evaluation schemes for book recommendation tasks and the existing classic recommendation models, this paper discusses the advantages and disadvantages of the BookGPT in book recomme
    
[^34]: 流式多语言自动语音识别中的专家混合Conformer模型

    Mixture-of-Expert Conformer for Streaming Multilingual ASR. (arXiv:2305.15663v1 [cs.CL])

    [http://arxiv.org/abs/2305.15663](http://arxiv.org/abs/2305.15663)

    本文提出了一种流式多语言Conformer模型，引入了混合专家层，能够在训练和推理过程中学习仅激活子集参数，实现了高效的推理。与基准模型相比具有显著的WRE性能提升而与适配器模型相比具有类似的性能，无需语言信息，同时利用多语言浅融合还实现了进一步的性能提升。

    

    大容量的端到端模型已经显著提高了多语种自动语音识别的性能，但它们的计算成本对于设备应用来说仍然具有挑战性。我们提出了一种混合专家（MoE）层的流式真正多语言Conformer，该层能够在训练和推理过程中学习仅激活子集参数。MoE层包括一个softmax门，该门在前馈传播中选择多个专家中的最佳两个。所提出的MoE层通过激活固定数量的参数来提供高效的推理，随着专家数量的增加，激活的参数数量也会增加。我们在12种语言的数据集上对所提出的模型进行评估，并获得相对基线的平均11.9％的识别错误率改进。与使用基准信息的适配器模型相比，我们的MoE模型实现了类似的识别错误率，并且激活了相似数量的参数，但不需要任何语言信息。我们进一步展示了多语言浅融合约3％的相对识别错误率改进。

    End-to-end models with large capacity have significantly improved multilingual automatic speech recognition, but their computation cost poses challenges for on-device applications. We propose a streaming truly multilingual Conformer incorporating mixture-of-expert (MoE) layers that learn to only activate a subset of parameters in training and inference. The MoE layer consists of a softmax gate which chooses the best two experts among many in forward propagation. The proposed MoE layer offers efficient inference by activating a fixed number of parameters as the number of experts increases. We evaluate the proposed model on a set of 12 languages, and achieve an average 11.9% relative improvement in WER over the baseline. Compared to an adapter model using ground truth information, our MoE model achieves similar WER and activates similar number of parameters but without any language information. We further show around 3% relative WER improvement by multilingual shallow fusion.
    
[^35]: ConvGQR：面向会话搜索的生成式查询重构

    ConvGQR: Generative Query Reformulation for Conversational Search. (arXiv:2305.15645v1 [cs.IR])

    [http://arxiv.org/abs/2305.15645](http://arxiv.org/abs/2305.15645)

    本文提出了一种新的面向会话搜索的ConvGQR框架，通过结合预训练语言模型来重新构造查询，从而提供更好的搜索查询。

    

    在会话搜索中，用户当前搜索意图依赖于先前的对话历史。从整个对话上下文中确定一个良好的搜索查询是具有挑战性的。为避免查询编码器的昂贵重新训练，大部分现有方法尝试学习一个重写模型，通过模仿手动查询重写来去除当前查询的上下文。然而，手动重写的查询并不总是最好的搜索查询。训练重写模型会限制模型产生良好搜索查询的能力。本文提出一种新的框架ConvGQR，基于预训练语言模型（PLM），一个用于查询重写，另一个用于生成潜在答案，以重新构造会话查询。通过结合两者，ConvGQR可以提供更好的搜索查询。此外，为了将查询重构与检索性能联系起来，我们提出了一种基于特征选择的相似度分数模型，用于验证ConvGQR的有效性。

    In conversational search, the user's real search intent for the current turn is dependent on the previous conversation history. It is challenging to determine a good search query from the whole conversation context. To avoid the expensive re-training of the query encoder, most existing methods try to learn a rewriting model to de-contextualize the current query by mimicking the manual query rewriting. However, manually rewritten queries are not always the best search queries. Training a rewriting model on them would limit the model's ability to produce good search queries. Another useful hint is the potential answer to the question. In this paper, we propose ConvGQR, a new framework to reformulate conversational queries based on generative pre-trained language models (PLMs), one for query rewriting and another for generating potential answers. By combining both, ConvGQR can produce better search queries. In addition, to relate query reformulation to retrieval performance, we propose a 
    
[^36]: 词形变化：一个现实检验

    Morphological Inflection: A Reality Check. (arXiv:2305.15637v1 [cs.CL])

    [http://arxiv.org/abs/2305.15637](http://arxiv.org/abs/2305.15637)

    本文研究了词形变化任务存在的高性能和高可变性的原因，并提出新的数据采样和评估策略以改善结果的通用性和可靠性。通过这些策略，我们对当前词形变化系统的泛化能力做出了新的观察。

    

    词形变化是一个具有实践和认知应用的亚词汇自然语言处理任务。多年来，最先进的系统报告了在各种数据集和语言中高但也高度可变的性能。本文研究了高性能和高可变性的原因，并发现了数据集创建和评估中的几个方面在系统上系统地提高了性能，并掩盖了语言之间的差异。为了改善结果的通用性和可靠性，我们提出了更好地反映可能用例的新数据采样和评估策略。使用这些新策略，我们就当前变形系统的泛化能力做出了新的发现。

    Morphological inflection is a popular task in sub-word NLP with both practical and cognitive applications. For years now, state-of-the-art systems have reported high, but also highly variable, performance across data sets and languages. We investigate the causes of this high performance and high variability; we find several aspects of data set creation and evaluation which systematically inflate performance and obfuscate differences between languages. To improve generalizability and reliability of results, we propose new data sampling and evaluation strategies that better reflect likely use-cases. Using these new strategies, we make new observations on the generalization abilities of current inflection systems.
    
[^37]: 作为文本整合测试基准的句子联合生成的再探讨

    Revisiting Sentence Union Generation as a Testbed for Text Consolidation. (arXiv:2305.15605v1 [cs.CL])

    [http://arxiv.org/abs/2305.15605](http://arxiv.org/abs/2305.15605)

    本文提出将句子联合生成任务作为一个有效的测试基准，以评估文本整合的能力。该任务将整合挑战与主观内容选择分离开来，并提供了精细的注释方法和工具。实验研究表明，即使是先进的模型也难以应对一些关键的整合方面，表明在这个任务中有明显的改进空间。

    

    基于多个输入文本生成文本的任务（例如多文档摘要、长篇问题回答和现代对话应用）挑战模型对于适当整合部分重叠的多文本信息的能力。然而，这些任务将整合阶段与常常主观和定义不明确的内容选择要求相结合，阻碍了模型整合能力的适当评估。在本文中，我们建议重新考虑句子联合生成任务作为一个有效的定义明确的测试基准，评估文本整合能力，并将整合挑战与主观内容选择分离开来。为了支持这个任务的研究，我们提出了精细的注释方法和工具，用于众包句子联合，创建了迄今为止最大的联合数据集，并提供了其丰富的各种整合方面的覆盖率分析。然后，我们提出了一个全面的联合生成评估协议，包括自动和人为评估，并报告了几个最先进模型的结果。我们的实验研究表明，即使是先进的模型也难以应对一些关键的整合方面，表明在这个任务中有明显的改进空间。

    Tasks involving text generation based on multiple input texts, such as multi-document summarization, long-form question answering and contemporary dialogue applications, challenge models for their ability to properly consolidate partly-overlapping multi-text information. However, these tasks entangle the consolidation phase with the often subjective and ill-defined content selection requirement, impeding proper assessment of models' consolidation capabilities. In this paper, we suggest revisiting the sentence union generation task as an effective well-defined testbed for assessing text consolidation capabilities, decoupling the consolidation challenge from subjective content selection. To support research on this task, we present refined annotation methodology and tools for crowdsourcing sentence union, create the largest union dataset to date and provide an analysis of its rich coverage of various consolidation aspects. We then propose a comprehensive evaluation protocol for union gen
    
[^38]: 基于预训练语言模型的文本增强开放知识图谱补全

    Text-Augmented Open Knowledge Graph Completion via Pre-Trained Language Models. (arXiv:2305.15597v1 [cs.CL])

    [http://arxiv.org/abs/2305.15597](http://arxiv.org/abs/2305.15597)

    TAGREAL是一种可自动生成高质量查询提示信息，从大型文本语料库中检索支持信息以从PLM中探测知识的方法，用于开放知识图谱补全中，在两个基准数据集上取得了最先进的表现，并且即使在有限的训练数据情况下，仍然具有突出的性能。

    

    开放知识图谱补全的任务是从已知事实中提取新的发现。现有的增强知识图谱补全的方法要么需要事实三元组以扩大图推理空间，要么需要手动设计提示信息以从预训练语言模型中提取知识，这些方法性能有限，需要专家昂贵的工作。为此，我们提出了TAGREAL，它自动生成高质量的查询提示信息，并从大型文本语料库中检索支持信息以从PLM中探测知识以完成知识图谱补全。结果表明，TAGREAL在两个基准数据集上实现了最先进的性能。我们发现，即使是在有限的训练数据情况下，TAGREAL的性能仍然非常突出，超过了现有的基于嵌入、基于图和基于PLM的方法。

    The mission of open knowledge graph (KG) completion is to draw new findings from known facts. Existing works that augment KG completion require either (1) factual triples to enlarge the graph reasoning space or (2) manually designed prompts to extract knowledge from a pre-trained language model (PLM), exhibiting limited performance and requiring expensive efforts from experts. To this end, we propose TAGREAL that automatically generates quality query prompts and retrieves support information from large text corpora to probe knowledge from PLM for KG completion. The results show that TAGREAL achieves state-of-the-art performance on two benchmark datasets. We find that TAGREAL has superb performance even with limited training data, outperforming existing embedding-based, graph-based, and PLM-based methods.
    
[^39]: 随机鹦鹉群体：用差分隐私促进大型语言模型的学习

    Flocks of Stochastic Parrots: Differentially Private Prompt Learning for Large Language Models. (arXiv:2305.15594v1 [cs.LG])

    [http://arxiv.org/abs/2305.15594](http://arxiv.org/abs/2305.15594)

    本文提出了一种差分隐私的提示学习方法，可用于大型语言模型，包括软提示和通过随机鹦鹉群体进行的离散提示，以解决由于提示数据敏感性引起的隐私问题。

    

    大型语言模型(LLMs)在上下文学习中表现出色。 然而，提示中包含的数据的敏感性引起了隐私问题。文章首先证明了这些问题是合理的：我们对用于提示LLMs的数据进行了简单但非常有效的成员推断攻击。为了解决这个问题，作者提出了一种私有的提示学习方法，并展示了私有的软提示可以通过下游数据的梯度下降实现。而离散提示则需要用多个LLMs进行嘈杂的表决，即随机鹦鹉群体，来将其知识传递到一个公共提示中。

    Large language models (LLMs) are excellent in-context learners. However, the sensitivity of data contained in prompts raises privacy concerns. Our work first shows that these concerns are valid: we instantiate a simple but highly effective membership inference attack against the data used to prompt LLMs. To address this vulnerability, one could forego prompting and resort to fine-tuning LLMs with known algorithms for private gradient descent. However, this comes at the expense of the practicality and efficiency offered by prompting. Therefore, we propose to privately learn to prompt. We first show that soft prompts can be obtained privately through gradient descent on downstream data. However, this is not the case for discrete prompts. Thus, we orchestrate a noisy vote among an ensemble of LLMs presented with different prompts, i.e., a flock of stochastic parrots. The vote privately transfers the flock's knowledge into a single public prompt. We show that LLMs prompted with our private
    
[^40]: 人类如何感知对抗文本？对基于词语对抗攻击的有效性和自然性进行现实检验。

    How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks. (arXiv:2305.15587v1 [cs.CL])

    [http://arxiv.org/abs/2305.15587](http://arxiv.org/abs/2305.15587)

    本研究通过调查人们对抗性文本样本的可感知性，得出现有文本攻击在人类参与的现实世界场景中是不切实际的，提供了更为现实的对NLP模型鲁棒性的评估。

    

    基于机器学习的自然语言处理(NLP)模型容易受到对抗攻击——恶意算法会微小地修改输入文本，导致模型做出错误预测。然而，这些攻击的评估忽略了不可察觉性质或者在有限的情况下进行研究。这意味着对抗扰动不会通过任何人类质量测试，也不会对通过人工检查的NLP系统构成真正的威胁。为了绕过这个限制并实现NLP模型鲁棒性的适当评估（以及后来的改进），我们对378名人类参与者就当前最先进的攻击方法生产的文本对抗样本的可感知性进行了调查。我们的结果表明，现有的文本攻击在人类参与的现实世界场景中是不切实际的。这与先前规模较小的人类研究相矛盾，后者报道了攻击成功的过于乐观结论。通过我们的工作，我们希望为当前对NLP模型的对抗攻击的研究做出积极贡献，提供对其潜在影响的更现实的看法。

    Natural Language Processing (NLP) models based on Machine Learning (ML) are susceptible to adversarial attacks -- malicious algorithms that imperceptibly modify input text to force models into making incorrect predictions. However, evaluations of these attacks ignore the property of imperceptibility or study it under limited settings. This entails that adversarial perturbations would not pass any human quality gate and do not represent real threats to human-checked NLP systems. To bypass this limitation and enable proper assessment (and later, improvement) of NLP model robustness, we have surveyed 378 human participants about the perceptibility of text adversarial examples produced by state-of-the-art methods. Our results underline that existing text attacks are impractical in real-world scenarios where humans are involved. This contrasts with previous smaller-scale human studies, which reported overly optimistic conclusions regarding attack success. Through our work, we hope to positi
    
[^41]: 多样化训练数据分布对多种文本风格转换的平衡效应

    Balancing Effect of Training Dataset Distribution of Multiple Styles for Multi-Style Text Transfer. (arXiv:2305.15582v1 [cs.CL])

    [http://arxiv.org/abs/2305.15582](http://arxiv.org/abs/2305.15582)

    本文探讨了训练数据输入多样性对多种文本风格转换模型生成文本质量的影响，通过平衡训练数据集中的风格分布，可以产生更有效的多种风格控制效果。

    

    文本风格转换是自然语言生成领域中一个激动人心的任务，但需要高质量的成对数据。对于多属性文本风格转换，训练模型需要具有足够支持所有考虑风格属性组合的数据集，增加了训练模型的难度。本文探讨了训练数据输入多样性对多风格转换模型生成文本质量的影响。我们通过设计启发式方法调整训练样本中的风格分布，构建了一个伪平行数据集。通过边际和联合分布平衡我们的训练数据集，训练我们的风格转换模型。我们观察到，平衡的数据集比不平衡或倾斜的数据集产生更有效的多种风格控制效果。通过定量分析，我们探讨了训练数据中多种风格分布对文本风格转换性能的影响。

    Text style transfer is an exciting task within the field of natural language generation that is often plagued by the need for high-quality paired datasets. Furthermore, training a model for multi-attribute text style transfer requires datasets with sufficient support across all combinations of the considered stylistic attributes, adding to the challenges of training a style transfer model. This paper explores the impact of training data input diversity on the quality of the generated text from the multi-style transfer model. We construct a pseudo-parallel dataset by devising heuristics to adjust the style distribution in the training samples. We balance our training dataset using marginal and joint distributions to train our style transfer models. We observe that a balanced dataset produces more effective control effects over multiple styles than an imbalanced or skewed one. Through quantitative analysis, we explore the impact of multiple style distributions in training data on style-t
    
[^42]: 聚焦是迁移学习的关键。

    Refocusing Is Key to Transfer Learning. (arXiv:2305.15542v1 [cs.CV])

    [http://arxiv.org/abs/2305.15542](http://arxiv.org/abs/2305.15542)

    这篇论文提出了一种名为 TOAST 的迁移学习算法，通过重新聚焦注意力，选择与任务相关的元素并反馈回模型，有效地提高了细粒度视觉分类数据集的性能，同时具有小部分可调参数。

    

    迁移学习涉及将预先训练好的模型适应新的下游任务。然而，我们观察到当前的迁移学习方法常常无法聚焦于与任务相关的特征。在这项工作中，我们强调了在迁移学习中重新聚焦注意力的重要性。我们引入了一种新的迁移学习算法-Top-Down Attention Steering（TOAST），它保持预先训练的骨干结构不变，同时选择输出中与任务有关的元素，并将它们反馈回模型，以引导其注意任务特定的特征。仅通过重新聚焦注意力，TOAST在许多迁移学习基准测试中实现了最先进的结果，同时具有小部分可调参数。与完全微调、LoRA和提示微调相比，TOAST在一系列细粒度视觉分类数据集上（例如，在 FGVC 上从 81.1% 提高到 86.2%）显着提高了性能。TOAST在指令跟随方面也优于完全微调的 Alpaca 模型。

    Transfer learning involves adapting a pre-trained model to novel downstream tasks. However, we observe that current transfer learning methods often fail to focus on task-relevant features. In this work, we emphasize the importance of refocusing the attention in transfer learning. We introduce Top-Down Attention Steering (TOAST), a novel transfer learning algorithm that keeps the pre-trained backbone frozen, while selecting the task-relevant elements in the output and feeding them back to the model to steer its attention to the task-specific features. By refocusing the attention only, TOAST achieves state-of-the-art results on a number of transfer learning benchmarks, while having a small portion of tunable parameters. Compared to fully fine-tuning, LoRA, and prompt tuning, TOAST substantially improves performance across a range of fine-grained visual classification datasets (e.g., 81.1% -> 86.2% on FGVC). TOAST also outperforms the fully fine-tuned Alpaca model on instruction-following
    
[^43]: 自动化难民案例分析：支持法律从业者的NLP流水线

    Automated Refugee Case Analysis: An NLP Pipeline for Supporting Legal Practitioners. (arXiv:2305.15533v1 [cs.CL])

    [http://arxiv.org/abs/2305.15533](http://arxiv.org/abs/2305.15533)

    本文介绍了一个支持法律从业者的自动化流水线，用于检索、处理和提取法律案件中的有针对性信息，并在加拿大的难民法律案例研究中扩展现有模型，提取19个有用类别的条款。使用最先进的神经命名实体识别进行信息提取，结果表明，在法律数据上预训练的模型表现最佳。

    

    本文介绍了一个端到端的流水线，用于检索、处理和提取法律案件中的有针对性信息。通过在加拿大的难民法律案例研究中调查一个少有研究的法律领域。搜索过去类似案例的案例法是律师和法官的法律工作的重要组成部分。虽然传统的基于命名实体识别的标签（如日期）在法律工作中提供有意义的信息，但我们提出了扩展现有模型，从难民案件中检索19个有用类别的条款。在创建了一个新颖的案例数据集后，我们使用最先进的神经命名实体识别（NER）进行信息提取。我们测试了包括两种变压器模型在内的不同架构，使用上下文和非上下文嵌入，比较了通用目的和领域特定的预训练。结果表明，在法律数据上预训练的模型表现最佳，尽管他们的规模较小。

    In this paper, we introduce an end-to-end pipeline for retrieving, processing, and extracting targeted information from legal cases. We investigate an under-studied legal domain with a case study on refugee law in Canada. Searching case law for past similar cases is a key part of legal work for both lawyers and judges, the potential end-users of our prototype. While traditional named-entity recognition labels such as dates provide meaningful information in legal work, we propose to extend existing models and retrieve a total of 19 useful categories of items from refugee cases. After creating a novel data set of cases, we perform information extraction based on state-of-the-art neural named-entity recognition (NER). We test different architectures including two transformer models, using contextual and non-contextual embeddings, and compare general purpose versus domain-specific pre-training. The results demonstrate that models pre-trained on legal data perform best despite their smaller
    
[^44]: 大型语言模型是少样本健康学习器

    Large Language Models are Few-Shot Health Learners. (arXiv:2305.15525v1 [cs.CL])

    [http://arxiv.org/abs/2305.15525](http://arxiv.org/abs/2305.15525)

    本论文提出大型语言模型可用于健康应用，只需少量调整便能捕捉健康领域的数字数据并在临床和健康环境下推理及参与各项健康任务。

    

    大型语言模型(LLMs)在捕捉实现实际任务中有用的丰富概念表示方面表现出色。然而，仅有语言的模型具有局限性。健康应用要求模型在数字数据(例如，临床领域中的生命体征、实验室值；在健康领域中的步数、运动)中具有良好的推理能力，而这些数字数据在现有训练语料中很难或不能用文本轻松表达。我们证明，只需进行少量调整，大型语言模型便能够将各种生理和行为时间序列数据与多种健康任务联系起来，适用于临床和健康环境。使用可穿戴设备和医疗传感器记录的数据，我们评估了这些能力，并应用于心脏信号分析、物理活动识别、代谢计算(例如，燃烧的卡路里)以及压力报告和心理健康筛查的估计任务。

    Large language models (LLMs) can capture rich representations of concepts that are useful for real-world tasks. However, language alone is limited. While existing LLMs excel at text-based inferences, health applications require that models be grounded in numerical data (e.g., vital signs, laboratory values in clinical domains; steps, movement in the wellness domain) that is not easily or readily expressed as text in existing training corpus. We demonstrate that with only few-shot tuning, a large language model is capable of grounding various physiological and behavioral time-series data and making meaningful inferences on numerous health tasks for both clinical and wellness contexts. Using data from wearable and medical sensor recordings, we evaluate these capabilities on the tasks of cardiac signal analysis, physical activity recognition, metabolic calculation (e.g., calories burned), and estimation of stress reports and mental health screeners.
    
[^45]: 探索在关系抽取中自动扰动自然语言解释的有效性

    Exploring Automatically Perturbed Natural Language Explanations in Relation Extraction. (arXiv:2305.15520v1 [cs.CL])

    [http://arxiv.org/abs/2305.15520](http://arxiv.org/abs/2305.15520)

    本文研究了自然语言解释在关系抽取中的有效性，发现扰动解释也能够达到相当甚至更好的效果，这提供了新的见解。

    

    先前的研究已经表明，自然语言解释为模型提供重要的归纳偏好，从而提高了其泛化能力和数据效率。本文系统地研究了这些解释的有效性。引人注目的是，我们发现带有降低归纳偏好的扰动解释可以达到与原始解释相当或更好的性能。我们的发现为自然语言解释的特征提供了新的见解：（1）解释的影响因训练风格和数据集而异，以前认为的改进主要在冻结语言模型中观察到。（2）尽管以前的研究将解释的影响归因于它们的归纳偏好，但我们的研究表明，即使完全扰动解释，效果仍然存在。我们认为主要影响是提供了额外的信息和方面。

    Previous research has demonstrated that natural language explanations provide valuable inductive biases that guide models, thereby improving the generalization ability and data efficiency. In this paper, we undertake a systematic examination of the effectiveness of these explanations. Remarkably, we find that corrupted explanations with diminished inductive biases can achieve competitive or superior performance compared to the original explanations. Our findings furnish novel insights into the characteristics of natural language explanations in the following ways: (1) the impact of explanations varies across different training styles and datasets, with previously believed improvements primarily observed in frozen language models. (2) While previous research has attributed the effect of explanations solely to their inductive biases, our study shows that the effect persists even when the explanations are completely corrupted. We propose that the main effect is due to the provision of add
    
[^46]: 语言模型中高效文本常识融合的免费午餐

    Free Lunch for Efficient Textual Commonsense Integration in Language Models. (arXiv:2305.15516v1 [cs.CL])

    [http://arxiv.org/abs/2305.15516](http://arxiv.org/abs/2305.15516)

    本文提出了一种计算上更加高效的方法来将文本常识描述集成到语言模型中，将具有相似常识描述的样本分批次进行编码以提高效率。

    

    最近几年，文本常识知识库的出现旨在提供更加细致和丰富的上下文知识。将外部常识融合进语言模型已被证明是推进各种NLP任务的关键。然而，与编码传统符号知识相比，将文本常识描述合并是计算上昂贵的。在本文中，我们提出了一种不修改模型的方法来提高其效率。我们将具有相似常识描述的训练样本分成一个批次，从而在多个样本中重复使用编码描述。其中一个关键观察是，批次划分的上限可以缩小到经典的图k分割问题。因此，我们提出了一种基于谱聚类的算法来解决这个问题。大量实验证明了所提出的批次划分方法有效地减少了计算复杂度。

    Recent years have witnessed the emergence of textual commonsense knowledge bases, aimed at providing more nuanced and context-rich knowledge. The integration of external commonsense into language models has been shown to be a key enabler in advancing the state-of-the-art for a wide range of NLP tasks. However, incorporating textual commonsense descriptions is computationally expensive, as compared to encoding conventional symbolic knowledge. In this paper, we propose a method to improve its efficiency without modifying the model. We group training samples with similar commonsense descriptions into a single batch, thus reusing the encoded description across multiple samples. One key observation is that the upper bound of batch partitioning can be reduced to the classic {\it graph k-cut problem}. Consequently, we propose a spectral clustering-based algorithm to solve this problem. Extensive experiments illustrate that the proposed batch partitioning approach effectively reduces the compu
    
[^47]: 越大的语言模型错误越难以捉摸：Python中的标识符交换不被语言模型识别

    The Larger They Are, the Harder They Fail: Language Models do not Recognize Identifier Swaps in Python. (arXiv:2305.15507v1 [cs.CL])

    [http://arxiv.org/abs/2305.15507](http://arxiv.org/abs/2305.15507)

    该论文揭示了大型语言模型对Python标识符交换的识别问题，尤其是在逆比例缩放现象影响下表现更为显著。这表明LLM缺乏深刻、抽象的理解，无法胜任与训练偏差的任务。

    

    大型语言模型已成功应用于代码生成任务，这引发了一个问题，即这些模型对编程的理解程度如何。传统的编程语言具有不变性和等变性，人类程序员能直观地理解和利用这些性质，如标识符重命名（近似）不变性。我们发现，LLM在默认函数名称交换时不仅无法正确生成Python代码，有些模型在模型大小增加时甚至变得更加自信地进行错误预测，这是最近发现的逆比例缩放现象的实例，与通常观察到的模型大小增加会提高预测质量的趋势相反。我们的发现表明，尽管LLM的典型情况表现出色，但它们仍然缺乏一个深刻的、抽象的理解据以操纵内容，使它们无法胜任与训练偏差的任务。

    Large Language Models (LLMs) have successfully been applied to code generation tasks, raising the question of how well these models understand programming. Typical programming languages have invariances and equivariances in their semantics that human programmers intuitively understand and exploit, such as the (near) invariance to the renaming of identifiers. We show that LLMs not only fail to properly generate correct Python code when default function names are swapped, but some of them even become more confident in their incorrect predictions as the model size increases, an instance of the recently discovered phenomenon of Inverse Scaling, which runs contrary to the commonly observed trend of increasing prediction quality with increasing model size. Our findings indicate that, despite their astonishing typical-case performance, LLMs still lack a deep, abstract understanding of the content they manipulate, making them unsuitable for tasks that statistically deviate from their training 
    
[^48]: 从掩码语言模型中推导语言模型

    Deriving Language Models from Masked Language Models. (arXiv:2305.15501v1 [cs.CL])

    [http://arxiv.org/abs/2305.15501](http://arxiv.org/abs/2305.15501)

    本文研究了从掩码语言模型中推导显式联合分布的方法，找到了一种基于条件接近的方法，可以优于现有的基于马尔科夫随机场的方法，并发现条件可以超过原始的MLM。

    

    掩码语言模型（MLM）没有明确定义语言的分布，即它们本身并不是语言模型。然而，最近的研究在生成和评分的目的上将它们隐含地视为语言模型。本文研究了从MLM中导出显式联合分布的方法，重点是关注两个标记的分布，这样可以计算精确的分布特性。我们发现，一种基于识别条件接近于MLM的条件的联结的方法效果良好，并优于现有的基于马尔科夫随机场的方法。我们进一步发现，这个推导模型的条件甚至有时可以超过原始MLM的条件。

    Masked language models (MLM) do not explicitly define a distribution over language, i.e., they are not language models per se. However, recent work has implicitly treated them as such for the purposes of generation and scoring. This paper studies methods for deriving explicit joint distributions from MLMs, focusing on distributions over two tokens, which makes it possible to calculate exact distributional properties. We find that an approach based on identifying joints whose conditionals are closest to those of the MLM works well and outperforms existing Markov random field-based approaches. We further find that this derived model's conditionals can even occasionally outperform the original MLM's conditionals.
    
[^49]: 用户兴趣旅程的大型语言模型

    Large Language Models for User Interest Journeys. (arXiv:2305.15498v1 [cs.CL])

    [http://arxiv.org/abs/2305.15498](http://arxiv.org/abs/2305.15498)

    该论文提出了使用大型语言模型(LLMs)对用户兴趣进行建模的方法，并通过定义兴趣旅程，提出了一种模型旨在提高推荐的质量，并提供了可解释性和新颖性。

    

    大型语言模型（LLMs）已经展示出在自然语言理解和生成方面的令人瞩目能力。然而，它们在更深入地理解用户和改善个性化推荐平台体验方面的潜力还远未被发挥。本文旨在填补这一空白。我们提出了使用LLMs对用户兴趣进行建模的方法，并定义了兴趣旅程作为用户基于他们的活动而遍历过的兴趣状态序列。我们的实验证明，相对于传统的用户表示方法，我们提出的方法可以提高推荐的质量，并且生成的兴趣旅程为推荐过程提供了可解释性和新颖性。

    Large language models (LLMs) have shown impressive capabilities in natural language understanding and generation. Their potential for deeper user understanding and improved personalized user experience on recommendation platforms is, however, largely untapped. This paper aims to address this gap. Recommender systems today capture users' interests through encoding their historical activities on the platforms. The generated user representations are hard to examine or interpret. On the other hand, if we were to ask people about interests they pursue in their life, they might talk about their hobbies, like I just started learning the ukulele, or their relaxation routines, e.g., I like to watch Saturday Night Live, or I want to plant a vertical garden. We argue, and demonstrate through extensive experiments, that LLMs as foundation models can reason through user activities, and describe their interests in nuanced and interesting ways, similar to how a human would.  We define interest journe
    
[^50]: PromptNER: 基于提示的命名实体识别

    PromptNER: Prompting For Named Entity Recognition. (arXiv:2305.15444v1 [cs.CL])

    [http://arxiv.org/abs/2305.15444](http://arxiv.org/abs/2305.15444)

    PromptNER是一种基于提示的命名实体识别算法，利用LLM生成潜在实体列表并提供解释，在少样本NER和跨领域NER方面实现了最先进性能。

    

    令人惊讶的是，大型语言模型（LLMs）和越来越多的基于提示的启发式方法现在提供了强大的现成方法，为各种经典的NLP问题提供了少量样本的解决方案。然而，尽管有着令人期待的初步结果，但这些基于LLM的少样本方法在命名实体识别（NER）方面仍远未达到最先进水平，现有的方法包括通过端到端结构理解学习表示，并在标准标记语料库上进行微调。本文介绍了PromptNER，一种新的用于少样本和跨领域NER的最先进算法。为了适应任何新的NER任务，PromptNER需要提供一组实体定义，除基本的少样本样例以外。给定输入句子，PromptNER提示LLM生成一个潜在实体列表，并提供相应的解释，证明它们与提供的实体类型定义的兼容性。值得注意的是，PromptNER在少样本NER任务方面实现了最先进的性能，并在具有挑战性的WikiAnn数据集上为跨领域NER设定了新的SOTA。

    In a surprising turn, Large Language Models (LLMs) together with a growing arsenal of prompt-based heuristics now offer powerful off-the-shelf approaches providing few-shot solutions to myriad classic NLP problems. However, despite promising early results, these LLM-based few-shot methods remain far from the state of the art in Named Entity Recognition (NER), where prevailing methods include learning representations via end-to-end structural understanding and fine-tuning on standard labeled corpora. In this paper, we introduce PromptNER, a new state-of-the-art algorithm for few-Shot and cross-domain NER. To adapt to any new NER task PromptNER requires a set of entity definitions in addition to the standard few-shot examples. Given a sentence, PromptNER prompts an LLM to produce a list of potential entities along with corresponding explanations justifying their compatibility with the provided entity type definitions. Remarkably, PromptNER achieves state-of-the-art performance on few-sho
    
[^51]: 语言模型的分词器在不同语言之间引入了不公平现象

    Language Model Tokenizers Introduce Unfairness Between Languages. (arXiv:2305.15425v1 [cs.CL])

    [http://arxiv.org/abs/2305.15425](http://arxiv.org/abs/2305.15425)

    语言模型的分词器在不同语言之间引入了不公平现象，因为同一段文本翻译成不同的语言可能会导致极大的分词长度差异，这影响了一些语言社区在获取商业语言服务的成本、处理时间和延迟以及提供给机器学习模型的内容量方面存在不公平待遇。

    

    最近的语言模型表现出了惊人的多语言性能，即使没有明确为此进行过训练。尽管如此，人们对它们在不同语言之间的输出质量提出了担忧。在本文中，我们展示了在分词阶段出现了不同语言的处理差异，甚至在模型被调用之前就已经出现了。同一段文本翻译成不同的语言可以有极大的分词长度差异，有些情况下差异可高达15倍。这些差异在我们评估的17种分词器中仍然存在，即使它们是有意为多语言支持进行训练的。某些语言对的字符级和字节级模型也显示出4倍以上的编码长度差异。这导致了一些语言社区在获取商业语言服务的成本、处理时间和延迟以及提供给机器学习模型的内容量方面存在不公平待遇。

    Recent language models have shown impressive multilingual performance, even when not explicitly trained for it. Despite this, concerns have been raised about the quality of their outputs across different languages. In this paper, we show how disparity in the treatment of different languages arises at the tokenization stage, well before a model is even invoked. The same text translated into different languages can have drastically different tokenization lengths, with differences up to 15 times in some cases. These disparities persist across the 17 tokenizers we evaluate, even if they are intentionally trained for multilingual support. Character-level and byte-level models also exhibit over 4 times the difference in the encoding length for some language pairs. This induces unfair treatment for some language communities in regard to the cost of accessing commercial language services, the processing time and latency, as well as the amount of content that can be provided as context to the m
    
[^52]: RefGPT: GPT模型中基于参考的真实且可学习化的对话生成

    RefGPT: Reference -> Truthful & Customized Dialogues Generation by GPTs and for GPTs. (arXiv:2305.14994v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14994](http://arxiv.org/abs/2305.14994)

    RefGPT是一种基于参考的对话生成方法，可以生成大量真实且定制化的对话，并解决了对话生成中的模型幻觉问题。

    

    ChatGPT等通用的聊天模型已经通过使用高质量指令数据调整大型语言模型（LLM）来解决各种NLP任务。然而，收集人类编写的高质量数据，尤其是多轮对话，对大多数人来说是昂贵且难以实现的。尽管以往的研究已经使用了强大的LLMs来自动生成对话，但由于LLMs存在幻觉，这些对话都无法完全真实。因此，我们提出了一种名为RefGPT的方法，可以生成大量真实且定制化的对话，而无需担心模型幻觉造成的事实错误。RefGPT通过限制LLMs使用给定参考而不是回忆自己的知识来生成对话，从而解决了对话生成中的模型幻觉。此外，RefGPT对每个话语都添加了详细的控制，使其具有高度定制化的能力，这是以往研究所忽略的。

    General chat models, like ChatGPT, have attained impressive capability to resolve a wide range of NLP tasks by tuning Large Language Models (LLMs) with high-quality instruction data. However, collecting human-written high-quality data, especially multi-turn dialogues, is expensive and unattainable for most people. Though previous studies have used powerful LLMs to generate the dialogues automatically, but they all suffer from generating untruthful dialogues because of the LLMs hallucination. Therefore, we propose a method called RefGPT to generate enormous truthful and customized dialogues without worrying about factual errors caused by the model hallucination. RefGPT solves the model hallucination in dialogue generation by restricting the LLMs to leverage the given reference instead of reciting their own knowledge to generate dialogues. Additionally, RefGPT adds detailed controls on every utterances to enable highly customization capability, which previous studies have ignored. On the
    
[^53]: 神经网络的效用-概率对偶

    Utility-Probability Duality of Neural Networks. (arXiv:2305.14859v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.14859](http://arxiv.org/abs/2305.14859)

    提出了一种将深度学习中的标准监督学习过程解释为基于效用的解释方法，将学习的神经网络解释为编码在训练数据中显示的偏好的序数效用函数，可以将SGD最大似然估计的学习动态视为将神经网络优化到最优效用函数的迭代过程，从而提供了一个设计更好的神经网络体系结构的新视角。

    

    现代神经网络的训练通常被认为是拟合所需输出的概率分布的过程。然而，最近在许多语言生成任务中观察到的悖论现象让人们怀疑这种基于概率的解释是否能真正解释深度学习的经验成功。为了解决这个问题，我们提出了一种替代方法，将深度学习中的标准监督学习过程解释为基于效用的解释。基本思想是将学习的神经网络不解释为概率模型，而解释为编码在训练数据中显示的偏好的序数效用函数。在这个视角下，神经网络的训练对应于一个效用学习过程。具体而言，我们证明了对于所有具有softmax输出的神经网络，最大似然估计（MLE）的SGD学习动态可以被视为一个迭代过程，该过程将神经网络优化到最优效用函数。这个框架不仅提供了对神经网络训练过程的新解释，而且提供了一个设计更好的神经网络体系结构的新视角。

    It is typically understood that the training of modern neural networks is a process of fitting the probability distribution of desired output. However, recent paradoxical observations in a number of language generation tasks let one wonder if this canonical probability-based explanation can really account for the empirical success of deep learning.  To resolve this issue, we propose an alternative utility-based explanation to the standard supervised learning procedure in deep learning. The basic idea is to interpret the learned neural network not as a probability model but as an ordinal utility function that encodes the preference revealed in training data. In this perspective, training of the neural network corresponds to a utility learning process. Specifically, we show that for all neural networks with softmax outputs, the SGD learning dynamic of maximum likelihood estimation (MLE) can be seen as an iteration process that optimizes the neural network toward an optimal utility functi
    
[^54]: 基于最大化互信息的视频多模态融合去噪瓶颈模型

    Denoising Bottleneck with Mutual Information Maximization for Video Multimodal Fusion. (arXiv:2305.14652v1 [cs.CL])

    [http://arxiv.org/abs/2305.14652](http://arxiv.org/abs/2305.14652)

    本文提出了一种基于去噪瓶颈和最大化互信息的视频多模态融合模型（DBF），该模型可以细粒度地过滤掉冗余和噪声信息，同时保留不同模态中的关键信息，并在多语言视频分类任务中表现出显著优越性。

    

    视频多模态融合旨在将视频中的多模态信号（如视觉、音频和文本）整合，以使用多模态内容进行补充预测。然而，与其他图像-文本多模态任务不同，视频具有更长的多模态序列，在视觉和音频模态中存在更多的冗余和噪声。因此，我们提出了一种用于细粒度视频多模态融合的去噪瓶颈融合（DBF）模型。我们一方面采用瓶颈机制，以限制的感受野过滤噪声和冗余信息。另一方面，我们使用最大化互信息模块来调节过滤模块，以保留不同模态中的关键信息。我们的DBF模型在多语言视频分类任务中显著优于当前最先进的基准模型。

    Video multimodal fusion aims to integrate multimodal signals in videos, such as visual, audio and text, to make a complementary prediction with multiple modalities contents. However, unlike other image-text multimodal tasks, video has longer multimodal sequences with more redundancy and noise in both visual and audio modalities. Prior denoising methods like forget gate are coarse in the granularity of noise filtering. They often suppress the redundant and noisy information at the risk of losing critical information. Therefore, we propose a denoising bottleneck fusion (DBF) model for fine-grained video multimodal fusion. On the one hand, we employ a bottleneck mechanism to filter out noise and redundancy with a restrained receptive field. On the other hand, we use a mutual information maximization module to regulate the filter-out module to preserve key information within different modalities. Our DBF model achieves significant improvement over current state-of-the-art baselines on mult
    
[^55]: CMOT: 通过最优传输进行跨模态Mixup，用于语音翻译

    CMOT: Cross-modal Mixup via Optimal Transport for Speech Translation. (arXiv:2305.14635v1 [cs.CL])

    [http://arxiv.org/abs/2305.14635](http://arxiv.org/abs/2305.14635)

    CMOT是一种用于跨模态语音翻译的方法，通过最优传输找到语音和文本序列之间的对齐，并在标记级别上混合不同模态的序列，实现了在有限数据下更好的性能表现。

    

    端到端语音翻译（ST）是将源语言中的语音信号翻译成目标语言文本的任务。作为一项跨模态任务，端到端ST在有限数据下进行训练非常困难。现有的方法通常尝试从机器翻译（MT）中转移知识，但其性能由于语音和文本之间的模态差距受到限制。本文提出了Cross-modal Mixup via Optimal Transport（CMOT）来克服模态差距。我们通过最优传输找到语音和文本序列之间的对齐，然后使用对齐在标记级别上混合来自不同模态的序列。在MuST-C ST基准测试上的实验表明，CMOT在8个翻译方向上实现了平均BLEU值为30.0，超过了以前的方法。进一步的分析表明，CMOT可以自适应地找到模态之间的对齐，有助于缓解语音和文本之间的模态差距。代码公开可用于 https://github.com/ic

    End-to-end speech translation (ST) is the task of translating speech signals in the source language into text in the target language. As a cross-modal task, end-to-end ST is difficult to train with limited data. Existing methods often try to transfer knowledge from machine translation (MT), but their performances are restricted by the modality gap between speech and text. In this paper, we propose Cross-modal Mixup via Optimal Transport CMOT to overcome the modality gap. We find the alignment between speech and text sequences via optimal transport and then mix up the sequences from different modalities at a token level using the alignment. Experiments on the MuST-C ST benchmark demonstrate that CMOT achieves an average BLEU of 30.0 in 8 translation directions, outperforming previous methods. Further analysis shows CMOT can adaptively find the alignment between modalities, which helps alleviate the modality gap between speech and text. Code is publicly available at https://github.com/ic
    
[^56]: 相近语言的自动可读性评估

    Automatic Readability Assessment for Closely Related Languages. (arXiv:2305.13478v1 [cs.CL])

    [http://arxiv.org/abs/2305.13478](http://arxiv.org/abs/2305.13478)

    本研究探索了如何通过语言方面（如相互智能性或语言相关性）来提高低资源语言中的自动可读性评估，并使用三种菲律宾语言的短篇小说来训练模型，发现应用专业特征CrossNGO可以改善ARA。

    

    近年来，自动可读性评估（ARA）的主要研究重点已转向使用昂贵的基于深度学习的方法，其主要目标是提高模型的准确性。然而，在低资源语言中，传统的手工制作特征仍然广泛使用，因为缺乏现有的NLP工具来提取更深层次的语言表示。我们从技术组件上退一步，着重探讨如何通过诸如相互智能性或语言相关性的语言方面来提高低资源环境下的ARA。我们收集在菲律宾的三种语言（他加禄语，比科尔语和宿务语）中编写的短篇小说来训练可读性评估模型，并探索各种跨语言设置中的数据和特征之间的交互作用。 我们的结果表明，在具有高相互可理解性的语言中应用n-gram重叠的新型专业特征CrossNGO可以改善ARA。

    In recent years, the main focus of research on automatic readability assessment (ARA) has shifted towards using expensive deep learning-based methods with the primary goal of increasing models' accuracy. This, however, is rarely applicable for low-resource languages where traditional handcrafted features are still widely used due to the lack of existing NLP tools to extract deeper linguistic representations. In this work, we take a step back from the technical component and focus on how linguistic aspects such as mutual intelligibility or degree of language relatedness can improve ARA in a low-resource setting. We collect short stories written in three languages in the Philippines-Tagalog, Bikol, and Cebuano-to train readability assessment models and explore the interaction of data and features in various cross-lingual setups. Our results show that the inclusion of CrossNGO, a novel specialized feature exploiting n-gram overlap applied to languages with high mutual intelligibility, sig
    
[^57]: 利用并行注意力和前馈网络设计研究Transformer中前馈网络的作用

    Investigating the Role of Feed-Forward Networks in Transformers Using Parallel Attention and Feed-Forward Net Design. (arXiv:2305.13297v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13297](http://arxiv.org/abs/2305.13297)

    本文研究了利用并行注意力和前馈网络设计（PAF）架构验证了前馈网络在变压器模型中的关键作用，并表明FFN块的主要功能是保持各向同性并防止退化，注意块中计算的残差范数远小于输入令牌嵌入范数。

    

    本文通过利用并行注意力和前馈网络设计（PAF）架构，并将其与系列注意力和前馈网络设计（SAF）进行比较，研究前馈网络在变压器模型中的关键作用。 PAF的有效性关键在于两个主要假设，即FFN块和层内注意块的主要功能为保持令牌嵌入的各向同性并防止其退化，以及注意力块中计算的残差范数远小于输入令牌嵌入范数。为了实证这些假设，我们训练了RoBERTa-large和bert-large-uncased两个大型语言模型的PAF变体。我们的结果表明，在PAF设计中，这两个假设都成立。本研究有助于深入了解Transformer架构中FFN和自注意机制之间的作用和相互作用。

    This paper investigates the key role of Feed-Forward Networks (FFNs) in transformer models by utilizing the Parallel Attention and Feed-Forward Net Design (PAF) architecture, and comparing it to their Series Attention and Feed-Forward Net Design (SAF) counterparts. Central to the effectiveness of PAF are two main assumptions regarding the FFN block and the attention block within a layer: 1) the primary function of the FFN block is to maintain isotropy among token embeddings and prevent their degeneration, and 2) the residual norm computed in the attention block is substantially smaller than the input token embedding norm. To empirically validate these assumptions, we train PAF variants of two large language models (RoBERTa-large and bert-large-uncased). Our results demonstrate that both assumptions hold true in the PAF design. This study contributes to a deeper understanding of the roles and interactions between FFNs and self-attention mechanisms in transformer architectures.
    
[^58]: EMNS / Imz / Corpus：情感单说者数据集，用于游戏、电视和漫画中的叙述故事。

    EMNS /Imz/ Corpus: An emotive single-speaker dataset for narrative storytelling in games, television and graphic novels. (arXiv:2305.13137v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13137](http://arxiv.org/abs/2305.13137)

    该研究提出了一个名为EMNS/Imz/ Corpus的情感单说者数据集，旨在增强交互式叙述驱动系统中对话的表现力和情感质量。该数据集在传达情感和表现力方面表现最佳，尤其在共享情感方面表现出色。

    

    文字到语音技术的日益普及导致了对于适应对话背景和情感语气的自然情感语音的需求。情感叙述故事（EMNS）语料库是一个独特的语音数据集，旨在增强交互式叙述驱动系统中对话的表现力和情感质量。该语料库包括一位女性演说者讲述标记话语的2.3小时录音，涵盖了八种表演情感状态，分布均匀，方差为0.68％，以及表现力水平和自然语言描述和词重音标签。对来自不同数据集的音频样本进行的评估表明，EMNS语料库在准确传达情感和表现力方面获得了最高的平均分。它在表达共享情感方面优于其他数据集，并达到了可比的真实水平。一个分类任务证实了准确的表示。

    The increasing adoption of text-to-speech technologies has led to a growing demand for natural and emotive voices that adapt to a conversation's context and emotional tone. The Emotive Narrative Storytelling (EMNS) corpus is a unique speech dataset created to enhance conversations' expressiveness and emotive quality in interactive narrative-driven systems. The corpus consists of a 2.3-hour recording featuring a female speaker delivering labelled utterances. It encompasses eight acted emotional states, evenly distributed with a variance of 0.68%, along with expressiveness levels and natural language descriptions with word emphasis labels. The evaluation of audio samples from different datasets revealed that the EMNS corpus achieved the highest average scores in accurately conveying emotions and demonstrating expressiveness. It outperformed other datasets in conveying shared emotions and achieved comparable levels of genuineness. A classification task confirmed the accurate representatio
    
[^59]: 通过二分搜索学习同时机器翻译的最优策略

    Learning Optimal Policy for Simultaneous Machine Translation via Binary Search. (arXiv:2305.12774v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12774](http://arxiv.org/abs/2305.12774)

    本文提出通过二分搜索学习同时机器翻译最优策略的方法，并在多个翻译任务上验证了在所有延迟情景下超越强基线的效果。

    

    同时机器翻译（SiMT）在阅读源句子时开始输出翻译，并需要精确的策略来决定何时输出生成的翻译。因此，该策略决定了在翻译每个目标令牌期间读取的源标记数量。然而，学习精确的翻译策略以实现良好的延迟质量权衡是困难的，因为没有与并行句子对应的黄金策略作为显式监督。本文提出了一种通过二分搜索在线构建最优策略的新方法。通过采用显式监督，我们的方法使SiMT模型能够学习最优策略，这可以指导模型在推理过程中完成翻译。在四个翻译任务上的实验结果表明，我们的方法可以在所有延迟方案下超越强基线。

    Simultaneous machine translation (SiMT) starts to output translation while reading the source sentence and needs a precise policy to decide when to output the generated translation. Therefore, the policy determines the number of source tokens read during the translation of each target token. However, it is difficult to learn a precise translation policy to achieve good latency-quality trade-offs, because there is no golden policy corresponding to parallel sentences as explicit supervision. In this paper, we present a new method for constructing the optimal policy online via binary search. By employing explicit supervision, our approach enables the SiMT model to learn the optimal policy, which can guide the model in completing the translation during inference. Experiments on four translation tasks show that our method can exceed strong baselines across all latency scenarios.
    
[^60]: FIT：远程交错Transformer

    FIT: Far-reaching Interleaved Transformers. (arXiv:2305.12689v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.12689](http://arxiv.org/abs/2305.12689)

    FIT是一种基于Transformer的架构，将数据标记分组，使用局部层和全局层进行操作。通过交错使用这些层并使用交叉注意力促进信息交换，FIT在一系列任务中均实现最先进的性能。

    

    我们提出了FIT：一种基于Transformer的架构，具有高效的自我注意力和自适应计算能力。与原始Transformer不同的是，我们将数据标记分成组，每个组是一个较短的标记序列。我们使用两种类型的Transformer层：局部层在每个组内操作数据标记，而全局层在一个更小的引入的潜在标记集合上操作。这些层包括与标准Transformer相同的自我注意力和前馈层，被交错使用，交叉注意力用于在同一组内数据和潜在标记之间促进信息交换。每个大小为n的组内的注意力复杂度为$O(n^2)$，但对于长度为L的序列，可以在全局范围内达到$O(L^{{4}/{3}})$。通过更多地依赖执行使用更小潜在标记集合的全局层，可以进一步提高效率。FIT是一种多用途的架构，可应用于广泛的任务，并在几个基准数据集上实现了最先进的性能，包括语言建模和图像字幕生成。

    We present FIT: a transformer-based architecture with efficient self-attention and adaptive computation. Unlike original transformers, which operate on a single sequence of data tokens, we divide the data tokens into groups, with each group being a shorter sequence of tokens. We employ two types of transformer layers: local layers operate on data tokens within each group, while global layers operate on a smaller set of introduced latent tokens. These layers, comprising the same set of self-attention and feed-forward layers as standard transformers, are interleaved, and cross-attention is used to facilitate information exchange between data and latent tokens within the same group. The attention complexity is $O(n^2)$ locally within each group of size $n$, but can reach $O(L^{{4}/{3}})$ globally for sequence length of $L$. The efficiency can be further enhanced by relying more on global layers that perform adaptive computation using a smaller set of latent tokens. FIT is a versatile arch
    
[^61]: Cross2StrA: 基于跨语言交叉模态结构枢纽对进行非配对跨语言图像字幕生成

    Cross2StrA: Unpaired Cross-lingual Image Captioning with Cross-lingual Cross-modal Structure-pivoted Alignment. (arXiv:2305.12260v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.12260](http://arxiv.org/abs/2305.12260)

    本论文提出了一种基于跨语言交叉模态结构枢纽对跨语言图像字幕生成的方法，通过结合场景图和句法句子树结构来提高字幕生成相关性和流畅性，并使用跨语言和跨模式的返译训练以完全对齐字幕生成和翻译阶段。

    

    由于转移过程中语义场景和语法属性的不一致性，非配对跨语言图像字幕生成长期以来一直面临着不相关和语法不流畅等问题。在本研究中，我们提出通过结合场景图和句法句子树结构来解决以上问题。我们的字幕生成器包含语义结构引导的图像到枢纽（英文）字幕生成和语法结构引导下的枢纽（中文）到目标语言（中文）翻译，两者之间通过枢纽语言连接。我们使用场景图和句法句子树结构作为枢纽对跨模式语义结构对齐和跨语言语法结构对齐进行学习。我们还引入了跨语言和跨模式的返译训练，以完全对齐字幕生成和翻译阶段。在英汉转移实验中，我们的模型在提高字幕生成相关性和流畅性方面表现出极高的优越性。

    Unpaired cross-lingual image captioning has long suffered from irrelevancy and disfluency issues, due to the inconsistencies of the semantic scene and syntax attributes during transfer. In this work, we propose to address the above problems by incorporating the scene graph (SG) structures and the syntactic constituency (SC) trees. Our captioner contains the semantic structure-guided image-to-pivot captioning and the syntactic structure-guided pivot-to-target translation, two of which are joined via pivot language. We then take the SG and SC structures as pivoting, performing cross-modal semantic structure alignment and cross-lingual syntactic structure alignment learning. We further introduce cross-lingual&cross-modal back-translation training to fully align the captioning and translation stages. Experiments on English-Chinese transfers show that our model shows great superiority in improving captioning relevancy and fluency.
    
[^62]: 为无偏的跨语言关系抽取构建混合编码的通用依存森林

    Constructing Code-mixed Universal Dependency Forest for Unbiased Cross-lingual Relation Extraction. (arXiv:2305.12258v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12258](http://arxiv.org/abs/2305.12258)

    本文提出了一种无偏的UD树跨语言关系抽取转移方法，通过构建混合编码UD树，有助于弥合训练和预测阶段之间差距从而实现显著的跨语言关系抽取性能提升。

    

    最近的跨语言关系抽取研究采用通用依存（UD）资源的语言一致性结构特征，但由于语言的不可避免差异，很容易遭受受偏转移（例如目标偏差或源偏差）。在本文中，我们通过构建一种类型的混合编码UD树，研究了一种无偏的UD树跨语言关系抽取转移。首先将源语言的句子翻译成平行的目标语言，对两种语言都分别解析UD树，然后将源语言/目标语言的UD结构合并为统一的混合编码UD树。通过这样的森林特征，UD树的跨语言关系抽取训练和预测阶段之间的差距可以有效缩小。我们在ACE XRE基准数据集上进行实验，结果表明，所提出的混合编码UD森林有助于无偏的UD树跨语言关系抽取转移，并实现了显着的跨语言关系抽取性能提升。

    Latest efforts on cross-lingual relation extraction (XRE) aggressively leverage the language-consistent structural features from the universal dependency (UD) resource, while they may largely suffer from biased transfer (e.g., either target-biased or source-biased) due to the inevitable linguistic disparity between languages. In this work, we investigate an unbiased UD-based XRE transfer by constructing a type of code-mixed UD forest. We first translate the sentence of the source language to the parallel target-side language, for both of which we parse the UD tree respectively. Then, we merge the source-/target-side UD structures as a unified code-mixed UD forest. With such forest features, the gaps of UD-based XRE between the training and predicting phases can be effectively closed. We conduct experiments on the ACE XRE benchmark datasets, where the results demonstrate that the proposed code-mixed UD forests help unbiased UD-based XRE transfer, with which we achieve significant XRE pe
    
[^63]: 以情景图为轴心: 推理时基于图像无监督多模态机器翻译的视觉情景幻化方法

    Scene Graph as Pivoting: Inference-time Image-free Unsupervised Multimodal Machine Translation with Visual Scene Hallucination. (arXiv:2305.12256v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12256](http://arxiv.org/abs/2305.12256)

    本研究提出了一种基于情景图的轴心方法，通过动态生成伪视觉情景图来实现推理过程中的纯文本输入，从而实现了无监督多模态机器翻译任务。

    

    本文研究了更现实的无监督多模态机器翻译（UMMT）的推理时基于图像无监督多模态机器翻译(Ummt)。我们首先使用视觉和语言情景图(SG)来表示输入的图像和文本，通过SG表示方法可以确保对语义的整体理解。为了在推理过程中实现纯文本输入，我们设计了一种视觉情景幻化机制，可以从给定的文本SG动态生成伪视觉SG。我们引入了几种基于SG轴心的学习目标，以实现无监督翻译训练。在基准数据集Multi30K上，我们的SG方法在任务和设置上的 BLEU 得分显著高于最佳基线，有助于产生更完整、相关和流畅的翻译，而不需要依赖成对的图像。进一步的深入分析揭示了我们的模型如何推进UMMT任务。

    In this work, we investigate a more realistic unsupervised multimodal machine translation (UMMT) setup, inference-time image-free UMMT, where the model is trained with source-text image pairs, and tested with only source-text inputs. First, we represent the input images and texts with the visual and language scene graphs (SG), where such fine-grained vision-language features ensure a holistic understanding of the semantics. To enable pure-text input during inference, we devise a visual scene hallucination mechanism that dynamically generates pseudo visual SG from the given textual SG. Several SG-pivoting based learning objectives are introduced for unsupervised translation training. On the benchmark Multi30K data, our SG-based method outperforms the best-performing baseline by significant BLEU scores on the task and setup, helping yield translations with better completeness, relevance and fluency without relying on paired images. Further in-depth analyses reveal how our model advances 
    
[^64]: 前缀传播: 针对长序列参数高效调整的方法

    Prefix Propagation: Parameter-Efficient Tuning for Long Sequences. (arXiv:2305.12086v1 [cs.CL])

    [http://arxiv.org/abs/2305.12086](http://arxiv.org/abs/2305.12086)

    前缀传播是一种针对长序列参数高效调整的方法，可实现50%减少参数且在处理长文档任务时具有更优性能。

    

    参数高效调整旨在减轻针对下游任务调整预训练语言模型的大内存需求。本文提出前缀传播这一简单有效的方法来弥补目前前缀调整存在的问题，并展示前缀传播与前缀调整相比在处理长文档任务时具有更好的性能，所需参数也只有前者的一半。

    Parameter-efficient tuning aims to mitigate the large memory requirements of adapting pretrained language models for downstream tasks. For example, one popular method, prefix-tuning, prepends trainable tokens to sequences while freezing the rest of the model's parameters. Although such models attain comparable performance with fine-tuning when applied to sequences with short to moderate lengths, we show their inferior performance when modelling long sequences. To bridge this gap, we propose prefix-propagation, a simple but effective approach that conditions prefixes on previous hidden states. We empirically demonstrate that prefix-propagation outperforms prefix-tuning across long-document tasks, while using 50% fewer parameters. To further investigate the proposed architecture, we also show its advantage in calibration, and perform additional study on its relationship with kernel attention. To the best of our knowledge, this work is the first to focus on parameter-efficient learning fo
    
[^65]: 通过整体3D场景理解生成视觉空间描述

    Generating Visual Spatial Description via Holistic 3D Scene Understanding. (arXiv:2305.11768v1 [cs.CV])

    [http://arxiv.org/abs/2305.11768](http://arxiv.org/abs/2305.11768)

    本文研究将3D场景特征纳入VSD方法，构建目标对象为中心的3D空间场景图(Go3D-S2G)，提出多样化的文本生成方法，可以显著提高性能。

    

    视觉空间描述(VSD)的目标是生成描述图像中给定对象空间关系的文本。现有的VSD工作仅模拟2D几何视觉特征，因此不可避免地陷入目标对象空间理解倾斜的问题。本文研究了将3D场景特征纳入VSD的方法。通过外部3D场景提取器，我们获取输入图像的3D对象和场景特征，基于此构建目标对象为中心的3D空间场景图(Go3D-S2G)，从而模拟目标对象在整体3D场景中的空间语义。此外，我们提出一种场景子图选择机制，从Go3D-S2G中采样拓扑多样的子图，导航不同的局部结构特征以产生空间多样化的文本生成。对两个VSD数据集的实验结果表明，我们的框架显著优于基线，特别是在one-split情况下提高了性能。

    Visual spatial description (VSD) aims to generate texts that describe the spatial relations of the given objects within images. Existing VSD work merely models the 2D geometrical vision features, thus inevitably falling prey to the problem of skewed spatial understanding of target objects. In this work, we investigate the incorporation of 3D scene features for VSD. With an external 3D scene extractor, we obtain the 3D objects and scene features for input images, based on which we construct a target object-centered 3D spatial scene graph (Go3D-S2G), such that we model the spatial semantics of target objects within the holistic 3D scenes. Besides, we propose a scene subgraph selecting mechanism, sampling topologically-diverse subgraphs from Go3D-S2G, where the diverse local structure features are navigated to yield spatially-diversified text generation. Experimental results on two VSD datasets demonstrate that our framework outperforms the baselines significantly, especially improving on
    
[^66]: 利用特征去噪和多模态主题建模的多模态关系抽取中的信息筛选

    Information Screening whilst Exploiting! Multimodal Relation Extraction with Feature Denoising and Multimodal Topic Modeling. (arXiv:2305.11719v1 [cs.CV])

    [http://arxiv.org/abs/2305.11719](http://arxiv.org/abs/2305.11719)

    该论文提出了一种新的多模态关系抽取框架，结合了内部信息筛选和外部信息利用的思想。通过视觉和文本场景图表示输入的细粒度语义结构，并利用图形信息瓶颈原理进行结构细化和特征去噪，同时运用主题建模丰富上下文，该系统在基准MRE数据集上表现优异，具有巨大的潜力。

    

    现有的多模态关系抽取(MRE)研究面临着两个共存的挑战，即内部信息过度利用和外部信息未能充分利用。为了应对这个问题，我们提出了一个新的框架，同时实现了内部信息筛选和外部信息利用的思想。首先，我们用视觉和文本场景图表示输入图像和文本的细粒度语义结构，将其进一步融合成一个统一的跨模态图(CMG)。基于CMG，我们利用图形信息瓶颈原理进行结构细化，主动去除不太具有信息量的特征。接下来，我们对输入图像和文本进行主题建模，将潜在的多模态主题特征融入其中以丰富上下文。在基准MRE数据集上，我们的系统显著优于当前最佳模型。通过进一步深入的分析，我们揭示了我们的方法在MRE任务中具有巨大的潜力。

    Existing research on multimodal relation extraction (MRE) faces two co-existing challenges, internal-information over-utilization and external-information under-exploitation. To combat that, we propose a novel framework that simultaneously implements the idea of internal-information screening and external-information exploiting. First, we represent the fine-grained semantic structures of the input image and text with the visual and textual scene graphs, which are further fused into a unified cross-modal graph (CMG). Based on CMG, we perform structure refinement with the guidance of the graph information bottleneck principle, actively denoising the less-informative features. Next, we perform topic modeling over the input image and text, incorporating latent multimodal topic features to enrich the contexts. On the benchmark MRE dataset, our system outperforms the current best model significantly. With further in-depth analyses, we reveal the great potential of our method for the MRE task
    
[^67]: 基于自监督调整的零样本文本分类算法

    Zero-Shot Text Classification via Self-Supervised Tuning. (arXiv:2305.11442v1 [cs.CL])

    [http://arxiv.org/abs/2305.11442](http://arxiv.org/abs/2305.11442)

    本文提出了一种基于自监督调整的零样本文本分类算法，通过使用无标签数据来调整语言模型，通过学习预测段落中的第一句话，实现了对未见过任务的零样本推断，模型不需要注释数据进行元调整，对模板的选择不敏感，并在实验中取得不错的结果。

    

    现有的零样本文本分类方法要么使用预训练语言模型进行提示，但这种方法对模板的选择非常敏感；要么依赖于大量相关任务的注释数据进行元调整。本文提出了一种基于自监督学习的新范式，通过使用无标签数据来调整语言模型，称为自监督调整。通过探索自由文本的内在结构，我们提出了一种新的学习目标，称为首句预测，以弥合无标签数据和文本分类任务之间的差距。调整模型以学习根据剩余文本来预测段落中的第一句话后，该模型能够推断出未见过的任务，如主题分类和情感分析。实验结果表明，我们的模型在10个任务中的7个任务上优于现有基准线。此外，分析表明，我们的模型对模板的选择不敏感，并且不需要注释数据进行元调整。

    Existing solutions to zero-shot text classification either conduct prompting with pre-trained language models, which is sensitive to the choices of templates, or rely on large-scale annotated data of relevant tasks for meta-tuning. In this work, we propose a new paradigm based on self-supervised learning to solve zero-shot text classification tasks by tuning the language models with unlabeled data, called self-supervised tuning. By exploring the inherent structure of free texts, we propose a new learning objective called first sentence prediction to bridge the gap between unlabeled data and text classification tasks. After tuning the model to learn to predict the first sentence in a paragraph based on the rest, the model is able to conduct zero-shot inference on unseen tasks such as topic classification and sentiment analysis. Experimental results show that our model outperforms the state-of-the-art baselines on 7 out of 10 tasks. Moreover, the analysis reveals that our model is less s
    
[^68]: 基于思维链索引的隐式情感推断

    Reasoning Implicit Sentiment with Chain-of-Thought Prompting. (arXiv:2305.11255v1 [cs.CL])

    [http://arxiv.org/abs/2305.11255](http://arxiv.org/abs/2305.11255)

    本研究提出了一种基于思维链索引的隐式情感推断框架（THOR），通过三次跳推理模仿类人推理过程，支持常识和多跳推理以推断意见的潜在意图，并逐步诱导隐式方面、意见和最终情感极性，实现了在监督和零样本设置上大幅提高技术水平。

    

    情感分析系统通过分析输入文本中的关键观点表达来确定给定目标的情感极性，而在隐式情感分析（ISA）中，观点提示以一种隐含和模糊的方式出现。因此，检测隐式情感需要常识和多跳推理能力来推断意见的潜在意图。受最近思维链索引（CoT）思想的启发，本研究介绍了一种三次跳推理（THOR）CoT框架，模仿ISA的类人推理过程。我们为THOR设计了一个三步提示原则，以逐步诱导隐式方面、意见和最终情感极性。我们的THOR+Flan-T5（11B）在监督设置上将技术水平推进了超过6％的F1值。更为显著的是，THOR+GPT3（175B）在零样本设置上将技术水平提升了超过50％的F1值。我们的代码位于https://github.com/scofield7419/THOR-ISA 。

    While sentiment analysis systems try to determine the sentiment polarities of given targets based on the key opinion expressions in input texts, in implicit sentiment analysis (ISA) the opinion cues come in an implicit and obscure manner. Thus detecting implicit sentiment requires the common-sense and multi-hop reasoning ability to infer the latent intent of opinion. Inspired by the recent chain-of-thought (CoT) idea, in this work we introduce a Three-hop Reasoning (THOR) CoT framework to mimic the human-like reasoning process for ISA. We design a three-step prompting principle for THOR to step-by-step induce the implicit aspect, opinion, and finally the sentiment polarity. Our THOR+Flan-T5 (11B) pushes the state-of-the-art (SoTA) by over 6% F1 on supervised setup. More strikingly, THOR+GPT3 (175B) boosts the SoTA by over 50% F1 on zero-shot setting. Our code is at https://github.com/scofield7419/THOR-ISA.
    
[^69]: 基于非自回归端到端语音识别系统的准确可靠置信度估计

    Accurate and Reliable Confidence Estimation Based on Non-Autoregressive End-to-End Speech Recognition System. (arXiv:2305.10680v1 [cs.SD])

    [http://arxiv.org/abs/2305.10680](http://arxiv.org/abs/2305.10680)

    本文介绍了一种CIF-Aligned置信度估计模型，利用了非自回归E2E ASR模型-Paraformer的特性，生成符号同步的声学嵌入，实现了准确可靠的置信度估计。

    

    在ASR领域中，估计识别结果的置信度得分是一项经典任务，对于各种下游任务和训练策略都至关重要。过去的端到端(E2E)置信度估计模型(CEM)预测与输入转录文本长度相等的得分序列，导致在删除和插入错误发生时估计不可靠。本文提出了一种CIF对齐置信度估计模型(CA-CEM)，利用连续积分和火 CIF 机制来生成符号同步的声学嵌入，以解决上述的估计失败问题。我们在令牌级别上使用AUC和RMSE以及utterance级别上的一种提出度量ECE-U来衡量估计质量。CA-CEM在ECE-U上获得了24%和19%的相对降低率，并且在两个测试集中具有更好的AUC和RMSE。此外，我们进行了分析。

    Estimating confidence scores for recognition results is a classic task in ASR field and of vital importance for kinds of downstream tasks and training strategies. Previous end-to-end~(E2E) based confidence estimation models (CEM) predict score sequences of equal length with input transcriptions, leading to unreliable estimation when deletion and insertion errors occur. In this paper we proposed CIF-Aligned confidence estimation model (CA-CEM) to achieve accurate and reliable confidence estimation based on novel non-autoregressive E2E ASR model - Paraformer. CA-CEM utilizes the modeling character of continuous integrate-and-fire (CIF) mechanism to generate token-synchronous acoustic embedding, which solves the estimation failure issue above. We measure the quality of estimation with AUC and RMSE in token level and ECE-U - a proposed metrics in utterance level. CA-CEM gains 24% and 19% relative reduction on ECE-U and also better AUC and RMSE on two test sets. Furthermore, we conduct anal
    
[^70]: 从巧克力兔到巧克力鳄鱼：语言模型是否理解名词复合词？

    From chocolate bunny to chocolate crocodile: Do Language Models Understand Noun Compounds?. (arXiv:2305.10568v1 [cs.CL])

    [http://arxiv.org/abs/2305.10568](http://arxiv.org/abs/2305.10568)

    本文研究了语言模型对于名词复合词的理解能力，提出了名词复合词解释的任务和名词复合词概念化的任务，并发现GPT-3在这些任务中的表现优于人类。

    

    名词复合词是指将多个名词组合成一个新词，如“巧克力兔”。本文提出了一种名词复合词解释的任务并修改了该任务的数据和评估设置。结果表明，GPT-3（一种语言模型）几乎可以完美地解决该任务。同时，本文还探讨了名词复合词概念化的任务，即解释新颖或罕见的名词组合词。最后，我们评估了GPT-3推理世界的程度，并发现它的表现优于人类。

    Noun compound interpretation is the task of expressing a noun compound (e.g. chocolate bunny) in a free-text paraphrase that makes the relationship between the constituent nouns explicit (e.g. bunny-shaped chocolate). We propose modifications to the data and evaluation setup of the standard task (Hendrickx et al., 2013), and show that GPT-3 solves it almost perfectly. We then investigate the task of noun compound conceptualization, i.e. paraphrasing a novel or rare noun compound. E.g., chocolate crocodile is a crocodile-shaped chocolate. This task requires creativity, commonsense, and the ability to generalize knowledge about similar concepts. While GPT-3's performance is not perfect, it is better than that of humans -- likely thanks to its access to vast amounts of knowledge, and because conceptual processing is effortful for people (Connell and Lynott, 2012). Finally, we estimate the extent to which GPT-3 is reasoning about the world vs. parroting its training data. We find that the 
    
[^71]: sustain.AI: 一种分析可持续性报告的推荐系统

    sustain.AI: a Recommender System to analyze Sustainability Reports. (arXiv:2305.08711v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08711](http://arxiv.org/abs/2305.08711)

    sustain.AI是一个智能的、上下文感知的推荐系统，可以帮助审计师、金融投资者以及广大公众高效地分析公司的可持续性报告，并通过与GRI标准匹配来提供更好的推荐精度。

    

    本文介绍了sustain.AI，这是一个智能的、上下文感知的推荐系统，可帮助审计师、金融投资者以及广大公众高效地分析公司的可持续性报告。该工具利用了端到端可训练的架构，将基于BERT的编码模块与多标签分类头相结合，将可持续性报告中的相关文本段落与全球报告倡议（GRI）标准中的相应法律法规匹配。我们在两个新颖的德国可持续性报告数据集上评估了我们的模型，并始终实现了与多个强基线模型相比更高的推荐性能。此外，sustain.AI已经公开在https://sustain.ki.nrw/上提供给所有人使用。

    We present $\text{sustain.AI}$, an intelligent, context-aware recommender system that assists auditors and financial investors as well as the general public to efficiently analyze companies' sustainability reports. The tool leverages an end-to-end trainable architecture that couples a BERT-based encoding module with a multi-label classification head to match relevant text passages from sustainability reports to their respective law regulations from the Global Reporting Initiative (GRI) standards. We evaluate our model on two novel German sustainability reporting data sets and consistently achieve a significantly higher recommendation performance compared to multiple strong baselines. Furthermore, $\text{sustain.AI}$ is publicly available for everyone at https://sustain.ki.nrw/.
    
[^72]: 使用LLM辅助注释进行语料库语言学研究：本地语法分析案例研究

    Using LLM-assisted Annotation for Corpus Linguistics: A Case Study of Local Grammar Analysis. (arXiv:2305.08339v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08339](http://arxiv.org/abs/2305.08339)

    本文研究了使用基于大语言模型的聊天机器人自动标注文本的潜力，重点考察了从本地语法角度观察道歉言语行为构成的功能元素的程度，并比较了不同模型在注释任务中的表现，结果表明Bing聊天机器人在任务中表现优于ChatGPT和人类标注员。

    

    基于大语言模型（LLMs）的聊天机器人在语言理解方面表现出很强的能力。本研究探索LLMs在协助基于语料库的语言学研究方面的潜力，通过将文本自动标注为特定语言信息类别。具体而言，我们研究了从本地语法的角度观察道歉言语行为构成的功能元素的程度，通过比较基于GPT-3.5的ChatGPT、基于GPT-4的Bing聊天机器人和人类编码器在注释任务中的表现。结果表明，Bing聊天机器人在任务中表现显着优于ChatGPT。与人类标注员相比，Bing聊天机器人的整体表现略低于人类标注员的表现，但已经取得了较高的F1得分:道歉标记99.95％，原因标记91.91％，道歉者标记95.35％，被道歉者标记89.74％和加强标记96.47％。这表明，在语言类别清晰且可以轻松识别的情况下，使用LLM辅助注释进行语料库语言学研究是可行的。

    Chatbots based on Large Language Models (LLMs) have shown strong capabilities in language understanding. In this study, we explore the potential of LLMs in assisting corpus-based linguistic studies through automatic annotation of texts with specific categories of linguistic information. Specifically, we examined to what extent LLMs understand the functional elements constituting the speech act of apology from a local grammar perspective, by comparing the performance of ChatGPT (powered by GPT-3.5), the Bing chatbot (powered by GPT-4), and a human coder in the annotation task. The results demonstrate that the Bing chatbot significantly outperformed ChatGPT in the task. Compared to human annotator, the overall performance of the Bing chatbot was slightly less satisfactory. However, it already achieved high F1 scores: 99.95% for the tag of APOLOGISING, 91.91% for REASON, 95.35% for APOLOGISER, 89.74% for APOLOGISEE, and 96.47% for INTENSIFIER. This suggests that it is feasible to use LLM-
    
[^73]: TinyStories: 语言模型能简小到什么程度却依然能够讲述连贯的英文故事？

    TinyStories: How Small Can Language Models Be and Still Speak Coherent English?. (arXiv:2305.07759v1 [cs.CL])

    [http://arxiv.org/abs/2305.07759](http://arxiv.org/abs/2305.07759)

    本文针对小型语言模型生成连贯的英文文本难题，引入了一个合成故事数据集 TinyStories，并探索小型模型规模、结构复杂度和训练数据规模对于语言模型表现的影响，证明了仅含 200 万参数的简单语言模型也能产生连贯的短故事。

    

    语言模型是自然语言处理中强大的工具，但在小型化时经常难以产生连贯和流畅的文本。本文引入了一个名为 TinyStories 的合成故事数据集，用于训练和评估规模小、复杂度低的语言模型对于短故事的生成能力。

    Language models (LMs) are powerful tools for natural language processing, but they often struggle to produce coherent and fluent text when they are small. Models with around 125M parameters such as GPT-Neo (small) or GPT-2 (small) can rarely generate coherent and consistent English text beyond a few words even after extensive training. This raises the question of whether the emergence of the ability to produce coherent English text only occurs at larger scales (with hundreds of millions of parameters or more) and complex architectures (with many layers of global attention).  In this work, we introduce TinyStories, a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand, generated by GPT-3.5 and GPT-4. We show that TinyStories can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters), or have much simpler architectures (with only one transformer block), yet stil
    
[^74]: Masked Audio Text Encoders 在多模态重打分中是有效的。

    Masked Audio Text Encoders are Effective Multi-Modal Rescorers. (arXiv:2305.07677v1 [cs.SD])

    [http://arxiv.org/abs/2305.07677](http://arxiv.org/abs/2305.07677)

    本文提出了Masked Audio Text Encoders（MATE），一个多模态掩码语言模型重新打分器，将声学表示形式并入到MLM的输入空间中。使用MATE对自动语音识别（ASR）系统进行多模态打分，即使在目标域数据不足的情况下，也可以提高系统的领域泛化能力，并且可以在非常有限的训练数据量下就将单词错误率（WER）降低。

    

    掩码语言模型（MLM）已被证明对于自动语音识别（ASR）系统的二次打分非常有效。在这项工作中，我们提出 Masked Audio Text Encoder（MATE），它是一个多模态掩码语言模型重新打分器，将声学表示形式并入到MLM的输入空间中。我们采用对比学习来通过学习共享表示来有效地对齐各种模态。我们发现，当目标域数据不可用时，使用多模态重新打分器对ASR系统的领域泛化很有好处。与仅文本的基线相比，在域内数据组上，MATE 可以将单词错误率（WER）降低4％-16％，在域外数据组上可将WER降低3％-7％。此外，仅使用非常有限的训练数据（0.8小时），MATE就可以将WER比一次打分的基线降低8％-23％。

    Masked Language Models (MLMs) have proven to be effective for second-pass rescoring in Automatic Speech Recognition (ASR) systems. In this work, we propose Masked Audio Text Encoder (MATE), a multi-modal masked language model rescorer which incorporates acoustic representations into the input space of MLM. We adopt contrastive learning for effectively aligning the modalities by learning shared representations. We show that using a multi-modal rescorer is beneficial for domain generalization of the ASR system when target domain data is unavailable. MATE reduces word error rate (WER) by 4%-16% on in-domain, and 3%-7% on out-of-domain datasets, over the text-only baseline. Additionally, with very limited amount of training data (0.8 hours), MATE achieves a WER reduction of 8%-23% over the first-pass baseline.
    
[^75]: SemEval-2023任务2：细粒度多语言命名实体识别（MultiCoNER 2）

    SemEval-2023 Task 2: Fine-grained Multilingual Named Entity Recognition (MultiCoNER 2). (arXiv:2305.06586v1 [cs.CL])

    [http://arxiv.org/abs/2305.06586](http://arxiv.org/abs/2305.06586)

    该论文介绍了SemEval-2023 Task 2的研究发现，任务旨在通过使用MultiCoNER V2数据集，识别12种语言中复杂的细粒度命名实体。最优方法是将外部知识融入transformer模型，最具挑战性的是媒体标题和产品名称等实体类型。

    

    我们介绍了SemEval-2023任务2关于细粒度多语言命名实体识别（MultiCoNER 2）的研究发现。该任务分为13个轨道，重点关注12种语言和单语、多语和嘈杂环境下识别复杂的细粒度命名实体（如WRITTENWORK、VEHICLE、MUSICALGRP）的方法。任务使用MultiCoNER V2数据集，该数据集由Bangla、Chinese、English、Farsi、French、German、Hindi、Italian、Portuguese、Spanish、Swedish和Ukrainian组成，共有220万个实例。MultiCoNER 2是SemEval-2023中最受欢迎的任务之一，吸引了47个队伍提交842个结果，其中34个队伍提交了系统论文。结果表明，媒体标题和产品名称等复杂实体类型是最具挑战性的，将外部知识融入transformer模型的方法取得了最佳性能，并且在Creative Work和Group类别上获得了最大增益，即使使用外部知识仍然具有挑战性。

    We present the findings of SemEval-2023 Task 2 on Fine-grained Multilingual Named Entity Recognition (MultiCoNER 2). Divided into 13 tracks, the task focused on methods to identify complex fine-grained named entities (like WRITTENWORK, VEHICLE, MUSICALGRP) across 12 languages, in both monolingual and multilingual scenarios, as well as noisy settings. The task used the MultiCoNER V2 dataset, composed of 2.2 million instances in Bangla, Chinese, English, Farsi, French, German, Hindi, Italian., Portuguese, Spanish, Swedish, and Ukrainian. MultiCoNER 2 was one of the most popular tasks of SemEval-2023. It attracted 842 submissions from 47 teams, and 34 teams submitted system papers. Results showed that complex entity types such as media titles and product names were the most challenging. Methods fusing external knowledge into transformer models achieved the best performance, and the largest gains were on the Creative Work and Group classes, which are still challenging even with external kn
    
[^76]: 如何影响上下文范例在组合通用性中的作用？

    How Do In-Context Examples Affect Compositional Generalization?. (arXiv:2305.04835v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.04835](http://arxiv.org/abs/2305.04835)

    本文提出了CoFe测试套件来调查上下文组合泛化。实验结果表明，上下文示例应该在结构上与测试用例类似，相互之间应该不同，而且单独地简单。

    

    组合泛化——理解看不见的已知原始组合——是人类智能中的一个重要推理能力。AI社区主要通过在许多训练样本上微调神经网络来研究这种能力，然而还不清楚上下文学习——基于大型语言模型的主要少样本范式——是否展示组合泛化。在本文中，我们提出了CoFe，一个测试套件来调查上下文组合泛化。我们发现，组合泛化性能很容易受到上下文示例选择的影响，因此提出了研究问题：什么是在组合泛化中制作好的上下文示例的关键因素。我们研究了三个潜在因素：相似性、多样性和复杂性。我们的系统实验表明，在组合通用性中，上下文示例应该在结构上与测试用例类似，相互之间应该不同，而且单独地简单。

    Compositional generalization--understanding unseen combinations of seen primitives--is an essential reasoning capability in human intelligence. The AI community mainly studies this capability by fine-tuning neural networks on lots of training samples, while it is still unclear whether and how in-context learning--the prevailing few-shot paradigm based on large language models--exhibits compositional generalization. In this paper, we present CoFe, a test suite to investigate in-context compositional generalization. We find that the compositional generalization performance can be easily affected by the selection of in-context examples, thus raising the research question what the key factors are to make good in-context examples for compositional generalization. We study three potential factors: similarity, diversity and complexity. Our systematic experiments indicate that in-context examples should be structurally similar to the test case, diverse from each other, and individually simple.
    
[^77]: DEnsity: 利用密度估计的开放域对话评估度量

    DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation. (arXiv:2305.04720v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.04720](http://arxiv.org/abs/2305.04720)

    DEnsity 提出了一种利用密度估计的开放域对话评估新方法，在特征空间中评估响应可能性来更好地与人类评估相关。

    

    尽管开放域对话系统在最近取得了一些进展，但构建一个可靠的评估度量仍然是一个具有挑战性的问题。最近的研究提出了基于分类模型的可学习度量，它们被训练用于区分正确的响应。然而，神经分类器对于来自未见分布的样本会做出过于自信的预测。我们提出了 DEsity，利用神经分类器从特征空间派生特征，并利用密度估计来评估响应。我们的度量器测量响应在人类对话分布中出现的可能性。此外，为了提高 DEnsity 的性能，我们利用对比学习进一步压缩了特征空间。多个响应评估数据集上的实验表明，DEnsity 与现有度量器相比更好地与人类评估相关。我们的代码可在 https://github.com/ddehun/DEnsity 获取。

    Despite the recent advances in open-domain dialogue systems, building a reliable evaluation metric is still a challenging problem. Recent studies proposed learnable metrics based on classification models trained to distinguish the correct response. However, neural classifiers are known to make overly confident predictions for examples from unseen distributions. We propose DEnsity, which evaluates a response by utilizing density estimation on the feature space derived from a neural classifier. Our metric measures how likely a response would appear in the distribution of human conversations. Moreover, to improve the performance of DEnsity, we utilize contrastive learning to further compress the feature space. Experiments on multiple response evaluation datasets show that DEnsity correlates better with human evaluations than the existing metrics. Our code is available at https://github.com/ddehun/DEnsity.
    
[^78]: 关注关系探索，提升知识库问答技术

    Pay More Attention to Relation Exploration for Knowledge Base Question Answering. (arXiv:2305.02118v1 [cs.CL])

    [http://arxiv.org/abs/2305.02118](http://arxiv.org/abs/2305.02118)

    该研究提出了一个新框架RE-KBQA，利用知识库中的关系增强实体表示，并引入额外监督。在三个方面探索关系指导，包括区分相似实体、探索额外监督以及进行后处理的基于关系指导的重排算法。该方法在两个基准数据集上验证有效性。

    

    知识库问答（KBQA）是一项挑战性的任务，旨在从大规模知识库中检索正确答案。现有研究主要关注实体表示和最终答案推理，导致对此任务的限制性监督。此外，关系在最近的技术中并未被充分考虑，而关系实际上决定了推理路径的选择。在本研究中，我们提出了一个新框架RE-KBQA，利用知识库中的关系增强实体表示，并引入额外的监督。我们从三个方面探索关系指导，包括（1）通过使用变分图自编码器学习关系重要性来区分相似实体；（2）通过预测关系分布作为软标签，采用多任务方案探索额外监督；（3）设计基于关系指导的重排算法进行后处理。在两个基准数据集上的实验结果表明了该方法的有效性。

    Knowledge base question answering (KBQA) is a challenging task that aims to retrieve correct answers from large-scale knowledge bases. Existing attempts primarily focus on entity representation and final answer reasoning, which results in limited supervision for this task. Moreover, the relations, which empirically determine the reasoning path selection, are not fully considered in recent advancements. In this study, we propose a novel framework, RE-KBQA, that utilizes relations in the knowledge base to enhance entity representation and introduce additional supervision. We explore guidance from relations in three aspects, including (1) distinguishing similar entities by employing a variational graph auto-encoder to learn relation importance; (2) exploring extra supervision by predicting relation distributions as soft labels with a multi-task scheme; (3) designing a relation-guided re-ranking algorithm for post-processing. Experimental results on two benchmark datasets demonstrate the e
    
[^79]: 小样本事件检测：经验研究和统一视角

    Few-shot Event Detection: An Empirical Study and a Unified View. (arXiv:2305.01901v1 [cs.CL])

    [http://arxiv.org/abs/2305.01901](http://arxiv.org/abs/2305.01901)

    本文从两个实用的设置出发，分析比较了十种代表性的小样本事件检测方法，归纳总结出了原型方法的性能优越性，并在此基础上提出了一种简单且有效的方法。

    

    小样本事件检测 (ED) 已经被广泛研究，然而这也带来了明显的差异，例如各种动机、任务和实验设置，这些差异妨碍了模型的理解和未来进展。本文提出了一项彻底的经验研究、一个ED模型的统一视角和一个更好的统一基准线。为了公平评估，我们选择了两个实用的设置：低资源设置来评估泛化能力和类转移设置来评估可转移性。我们比较了三个数据集上的十种代表性方法，这些方法大致被分为基于提示和基于原型的模型进行详细分析。为了调查基于原型方法的优越性能，我们分解了设计并建立了一个统一框架。基于此，我们不仅提出了一种简单而有效的方法（例如在低资源设置下获得2.7％F1收益），而且为未来的研究提供了许多有价值的研究见解。

    Few-shot event detection (ED) has been widely studied, while this brings noticeable discrepancies, e.g., various motivations, tasks, and experimental settings, that hinder the understanding of models for future progress. This paper presents a thorough empirical study, a unified view of ED models, and a better unified baseline. For fair evaluation, we choose two practical settings: low-resource setting to assess generalization ability and class-transfer setting for transferability. We compare ten representative methods on three datasets, which are roughly grouped into prompt-based and prototype-based models for detailed analysis. To investigate the superior performance of prototype-based methods, we break down the design and build a unified framework. Based on that, we not only propose a simple yet effective method (e.g., 2.7% F1 gains under low-resource setting) but also offer many valuable research insights for future research.
    
[^80]: RAFT: 奖励排名微调用于生成型基础模型对齐

    RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. (arXiv:2304.06767v1 [cs.LG])

    [http://arxiv.org/abs/2304.06767](http://arxiv.org/abs/2304.06767)

    RAFT框架引入了奖励排名微调方法，用于对齐生成型基础模型，以解决强化学习带来的低效和不稳定性问题。

    

    生成型基础模型容易受到广泛的无监督训练数据带来的隐式偏见的影响。这些偏见可能导致子优样本、扭曲的结果和不公平，可能产生重大影响。因此，将这些模型与人的伦理和偏好对齐是确保它们在真实应用中负责任和有效的部署的关键步骤。以往的研究主要采用人类反馈的强化学习（ RLHF）作为解决这个问题的手段。在 RL 算法的指导下，用人类反馈指导的奖励模型对生成模型进行微调。然而， RL 算法的低效性和不稳定性常常会对生成模型的成功对齐产生重大障碍，因此需要开发一种更为强大和简化的方法。为此，我们引入了一个新的框架，即奖励排名微调（ RAFT ），旨在对齐生成基础模型。

    Generative foundation models are susceptible to implicit biases that can arise from extensive unsupervised training data. Such biases can produce suboptimal samples, skewed outcomes, and unfairness, with potentially significant repercussions. Consequently, aligning these models with human ethics and preferences is an essential step toward ensuring their responsible and effective deployment in real-world applications. Prior research has primarily employed Reinforcement Learning from Human Feedback (RLHF) as a means of addressing this problem, wherein generative models are fine-tuned using RL algorithms guided by a human-feedback-informed reward model. However, the inefficiencies and instabilities associated with RL algorithms frequently present substantial obstacles to the successful alignment of generative models, necessitating the development of a more robust and streamlined approach. To this end, we introduce a new framework, Reward rAnked FineTuning (RAFT), designed to align generat
    
[^81]: 捷克语、波兰语和斯洛伐克语中的性别偏见量化研究

    Measuring Gender Bias in West Slavic Language Models. (arXiv:2304.05783v1 [cs.CL])

    [http://arxiv.org/abs/2304.05783](http://arxiv.org/abs/2304.05783)

    本研究分析了西斯拉夫语言模型中的性别偏差，发现捷克语、波兰语和斯洛伐克语均存在相似程度的性别偏见。这一研究填补了研究非英语语言模型性别偏见的空白。

    

    预训练模型会将基础数据集中的偏见延续到下游任务。然而，这些研究大多基于英语的单语言模型，而针对扩展到英语以外的语言的语言模型中的偏见的研究很少。本文通过分析西斯拉夫语言模型中的性别偏差来填补这一空白。我们首次引入了基于模板的数据集（包括捷克语、波兰语和斯洛伐克语），以测量针对男性、女性和非二进制主体的性别偏见。我们使用单语和多语言模型来完成这些句子，并评估它们是否适合于被遮盖的语言建模任务。接下来，我们通过量化生成单词的有毒性和性别特征来测量西斯拉夫语言模型中的性别偏见。我们发现这些语言模型生成的语句会因主体的性别而产生伤害性的完成度。令人惊讶的是，捷克语、斯洛伐克语和波兰语均显示出相似程度的性别偏见。本研究对于关于语言模型中存在的偏见的日益增长的研究体系做出贡献，并为评估和减少西斯拉夫语言中的性别偏见奠定了基础。

    Pre-trained language models have been known to perpetuate biases from the underlying datasets to downstream tasks. However, these findings are predominantly based on monolingual language models for English, whereas there are few investigative studies of biases encoded in language models for languages beyond English. In this paper, we fill this gap by analysing gender bias in West Slavic language models. We introduce the first template-based dataset in Czech, Polish, and Slovak for measuring gender bias towards male, female and non-binary subjects. We complete the sentences using both mono- and multilingual language models and assess their suitability for the masked language modelling objective. Next, we measure gender bias encoded in West Slavic language models by quantifying the toxicity and genderness of the generated words. We find that these language models produce hurtful completions that depend on the subject's gender. Perhaps surprisingly, Czech, Slovak, and Polish language mode
    
[^82]: LLMMaps——大型语言模型分层评价的可视化隐喻

    LLMMaps -- A Visual Metaphor for Stratified Evaluation of Large Language Models. (arXiv:2304.00457v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.00457](http://arxiv.org/abs/2304.00457)

    LLMMaps是一种分层评估大型语言模型性能的可视化技术，能够揭示取得高准确度和产生幻觉的子领域，并指导模型的进一步发展。

    

    大型语言模型(LLMs)在自然语言处理中取得了革命性的进展，并在各种任务中展示了惊人的能力。然而，它们容易产生幻觉，即模型在响应中暴露出不正确或错误的信息，这使得必须采用勤奋的评估方法。虽然LLM在特定知识领域中的表现通常是基于问答(Q&A)数据集进行评估，但这些评估通常仅报告整个领域的单个准确度数字，这一程序在透明度和模型改进方面存在问题。分层评估可以揭示可能更容易发生幻觉的子领域，从而有助于更好地评估LLMs的风险并指导它们的进一步发展。为支持这样的分层评估，我们提出了LLMMaps作为一种新的可视化技术，使用户能够根据Q&A数据集评估LLMs的性能。LLMMaps提供了对LLMs在不同子领域中的知识分布的详细洞察，允许用户放大领域的特定部分并探索模型性能上的差异。我们的实验证明，LLMMaps有助于识别出更容易出现LLM幻觉的子领域，并可以指导模型的发展，以改善这些领域的准确性。

    Large Language Models (LLMs) have revolutionized natural language processing and demonstrated impressive capabilities in various tasks. Unfortunately, they are prone to hallucinations, where the model exposes incorrect or false information in its responses, which renders diligent evaluation approaches mandatory. While LLM performance in specific knowledge fields is often evaluated based on question and answer (Q&A) datasets, such evaluations usually report only a single accuracy number for the entire field, a procedure which is problematic with respect to transparency and model improvement. A stratified evaluation could instead reveal subfields, where hallucinations are more likely to occur and thus help to better assess LLMs' risks and guide their further development. To support such stratified evaluations, we propose LLMMaps as a novel visualization technique that enables users to evaluate LLMs' performance with respect to Q&A datasets. LLMMaps provide detailed insights into LLMs' kn
    
[^83]: HuggingGPT: 在HugingFace中使用ChatGPT及其伙伴解决AI任务

    HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace. (arXiv:2303.17580v1 [cs.CL])

    [http://arxiv.org/abs/2303.17580](http://arxiv.org/abs/2303.17580)

    用ChatGPT作为任务规划工具，利用大型语言模型（LLM）作为控制器来整合现有的AI模型，解决复杂的AI任务。

    

    解决不同领域和模态的复杂AI任务是通向人工智能的关键步骤。本文提出了一个系统，利用大型语言模型（LLMs）作为控制器来管理现有的AI模型以解决AI任务，语言成为通用接口来赋能它。具体来说，我们使用ChatGPT作为任务规划工具，根据HuggingFace中可用的模型功能描述来选择模型，在选定AI模型的情况下执行每个子任务，并总结响应。

    Solving complicated AI tasks with different domains and modalities is a key step toward artificial general intelligence (AGI). While there are abundant AI models available for different domains and modalities, they cannot handle complicated AI tasks. Considering large language models (LLMs) have exhibited exceptional ability in language understanding, generation, interaction, and reasoning, we advocate that LLMs could act as a controller to manage existing AI models to solve complicated AI tasks and language could be a generic interface to empower this. Based on this philosophy, we present HuggingGPT, a system that leverages LLMs (e.g., ChatGPT) to connect various AI models in machine learning communities (e.g., HuggingFace) to solve AI tasks. Specifically, we use ChatGPT to conduct task planning when receiving a user request, select models according to their function descriptions available in HuggingFace, execute each subtask with the selected AI model, and summarize the response acco
    
[^84]: InstructABSA: 基于指令学习的方面情感分析

    InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis. (arXiv:2302.08624v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08624](http://arxiv.org/abs/2302.08624)

    InstructABSA是一种使用指令学习范式的方面情感分析方法，能够显著提高Aspect Term Extraction、Aspect Term Sentiment Classification、和Joint Task subtasks三个子任务的性能，并且在多个数据集上表现超过之前的最先进方法。

    

    本文介绍了InstructABSA，一种使用指令学习范式进行Aspect Based Sentiment Analysis (ABSA) 所有子任务（Aspect Term Extraction (ATE)，Aspect Term Sentiment Classification (ATSC)，以及Joint Task modeling）的方法。我们的方法对每个训练样本引入了正面、负面、和中性的例子，并使用指令来调整每个ABSA子任务的模型（Tk-Instruct），从而显著提高了性能。在Sem Eval 2014、2015和2016数据集上的实验结果表明，在所有三个ABSA子任务（ATE、ATSC和Joint Task）上，InstructABSA在性能上都比之前的最先进方法（SOTA）表现出了显著的优势，并且表现超过了7倍大的模型。特别是，在Rest14 ATE子任务上，InstructABSA超过了SOTA 7.31%的得分，Rest15 ATSC子任务上也有提升，并且在Lapt14 Joint Task上的表现提升了8.63%点。我们的结果还表明，对于所有三个子任务，InstructABSA具有强大的新领域泛化能力。

    In this paper, we present InstructABSA, Aspect Based Sentiment Analysis (ABSA) using the instruction learning paradigm for all ABSA subtasks: Aspect Term Extraction (ATE), Aspect Term Sentiment Classification (ATSC), and Joint Task modeling. Our method introduces positive, negative, and neutral examples to each training sample, and instruction tunes the model (Tk-Instruct) for each ABSA subtask, yielding significant performance improvements. Experimental results on the Sem Eval 2014, 15, and 16 datasets demonstrate that InstructABSA outperforms the previous state-of-the-art (SOTA) approaches on all three ABSA subtasks (ATE, ATSC, and Joint Task) by a significant margin, outperforming 7x larger models. In particular, InstructABSA surpasses the SOTA on the Rest14 ATE subtask by 7.31% points, Rest15 ATSC subtask by and on the Lapt14 Joint Task by 8.63% points. Our results also suggest a strong generalization ability to new domains across all three subtasks
    
[^85]: 什么是新事件？在叙事中识别新事件的演变。

    Whats New? Identifying the Unfolding of New Events in Narratives. (arXiv:2302.07748v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07748](http://arxiv.org/abs/2302.07748)

    本文提出了一个新的挑战性任务：自动识别叙述中的新事件，以识别创新和贡献。他们将事件定义为主语、谓语和宾语的三元组，并将其分类为新事件，具体取决于其是否可以通过常识推理来推导。

    

    叙事包含了上下文和时间的丰富事件资源。对这些事件的自动理解提供了摘要理解，以供进一步的计算(如推理)。本文研究了事件的信息状态(IS)，并提出了一个新的有挑战性的任务:自动识别叙述中的新事件。我们将事件定义为主语、谓语和宾语的三元组。该事件相对于话语上下文被归类为新事件，并取决于是否可以通过常识推理来推导。我们使用人类标注者在公开的叙述语料库上进行了句子级别的新事件标注。我们提供了标注协议，并研究了注释的质量和任务的难度。我们公开了标注的数据集、标注材料和用于叙述理解中新事件提取的机器学习基准模型。

    Narratives include a rich source of events unfolding over time and context. Automatic understanding of these events provides a summarised comprehension of the narrative for further computation (such as reasoning). In this paper, we study the Information Status (IS) of the events and propose a novel challenging task: the automatic identification of \textit{new} events in a narrative. We define an event as a triplet of subject, predicate, and object. The event is categorized as new with respect to the discourse context and whether it can be inferred through commonsense reasoning. We annotated a publicly available corpus of narratives with the new events at sentence level using human annotators. We present the annotation protocol and study the quality of the annotation and the difficulty of the task. We publish the annotated dataset, annotation materials, and machine learning baseline models for the task of new event extraction for narrative understanding.
    
[^86]: Transformer模型：介绍与目录

    Transformer models: an introduction and catalog. (arXiv:2302.07730v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07730](http://arxiv.org/abs/2302.07730)

    本论文介绍与分类了Transformer模型系列中最流行的模型，包括基于自监督学习和人类参与训练的模型，并对其中创新性的方面做了介绍。

    

    近年来，Transformer系列的基础模型如雨后春笋般涌现出来，它们中有些具有令人难忘的、有时甚至滑稽有趣但却不具自解释性的名称。本文旨在提供一个相对全面但简单的Transformer模型目录和分类，并介绍Transformer模型中最重要的方面和创新。目录中的模型包括通过自监督学习进行训练（例如BERT或GPT3）的模型，以及进一步通过人类参与训练（例如ChatGPT使用的InstructGPT模型）的模型。

    In the past few years we have seen the meteoric appearance of dozens of foundation models of the Transformer family, all of which have memorable and sometimes funny, but not self-explanatory, names. The goal of this paper is to offer a somewhat comprehensive but simple catalog and classification of the most popular Transformer models. The paper also includes an introduction to the most important aspects and innovations in Transformer models. Our catalog will include models that are trained using self-supervised learning (e.g., BERT or GPT3) as well as those that are further trained using a human-in-the-loop (e.g. the InstructGPT model used by ChatGPT).
    
[^87]: READIN：一个带有真实和多样化输入噪声的中文多任务基准数据集

    READIN: A Chinese Multi-Task Benchmark with Realistic and Diverse Input Noises. (arXiv:2302.07324v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07324](http://arxiv.org/abs/2302.07324)

    READIN是一个中文多任务基准数据集，它包含真实和多样化的输入噪声，旨在测试模型的鲁棒性和公平性。注释管道被设计来最大化多样性。

    

    对于许多真实世界应用，用户生成的输入通常包含由于语言变体或打字错误引起的语音识别错误等各种噪声。因此，测试模型在带有真实输入噪声的数据上的性能以确保其鲁棒性和公平性非常重要。然而，在中文中构建这样的基准数据集的研究很少，而实际情况中各种与语言相关的输入噪声屡见不鲜。为了填补这一重要空白，我们构建了一个名为READIN的中文多任务基准数据集，其中包括四个不同的任务，并要求注释者使用两种常用的中文输入方法——拼音输入和语音输入，重新输入原始测试数据。我们设计了注释管道以最大化多样性，例如通过指示注释者使用多样化的输入法编辑器（IME）来获得键盘噪声，并招募来自不同方言组的演讲者获取语音噪声。

    For many real-world applications, the user-generated inputs usually contain various noises due to speech recognition errors caused by linguistic variations1 or typographical errors (typos). Thus, it is crucial to test model performance on data with realistic input noises to ensure robustness and fairness. However, little study has been done to construct such benchmarks for Chinese, where various language-specific input noises happen in the real world. In order to fill this important gap, we construct READIN: a Chinese multi-task benchmark with REalistic And Diverse Input Noises. READIN contains four diverse tasks and requests annotators to re-enter the original test data with two commonly used Chinese input methods: Pinyin input and speech input. We designed our annotation pipeline to maximize diversity, for example by instructing the annotators to use diverse input method editors (IMEs) for keyboard noises and recruiting speakers from diverse dialectical groups for speech noises. We e
    
[^88]: 选择性解释：利用人类输入对齐可解释人工智能

    Selective Explanations: Leveraging Human Input to Align Explainable AI. (arXiv:2301.09656v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.09656](http://arxiv.org/abs/2301.09656)

    本研究提出一种通过利用人类输入生成选择性解释的通用框架，以弥合可解释人工智能（XAI）与人类解释的差距，并且在决策支持任务中进行了实验证明其有效性。

    

    近年来，出现了大量的可解释人工智能（XAI）算法，但它们经常因与人类解释的生产和消费方式存在显著差距而受到批评。因此，目前的XAI技术往往难以使用并缺乏有效性。在本文中，我们尝试通过使AI解释具有选择性（这是人类解释的基本属性之一）来弥合这些差距，通过根据接收方的偏好有选择性地呈现大量模型原因的子集来实现。我们提出了一个通用的框架，通过利用小样本上的人类输入来生成选择性解释。该框架开辟了一个丰富的设计空间，涵盖了不同的选择性目标、输入类型等。作为一个展示，我们使用决策支持任务来探索基于决策者认为相关的选择性解释。我们进行了两项实验研究，以检查从大一组模型原因中选择的三个子集与未选择的子集相比，选择性解释的效果。

    While a vast collection of explainable AI (XAI) algorithms have been developed in recent years, they are often criticized for significant gaps with how humans produce and consume explanations. As a result, current XAI techniques are often found to be hard to use and lack effectiveness. In this work, we attempt to close these gaps by making AI explanations selective -- a fundamental property of human explanations -- by selectively presenting a subset from a large set of model reasons based on what aligns with the recipient's preferences. We propose a general framework for generating selective explanations by leveraging human input on a small sample. This framework opens up a rich design space that accounts for different selectivity goals, types of input, and more. As a showcase, we use a decision-support task to explore selective explanations based on what the decision-maker would consider relevant to the decision task. We conducted two experimental studies to examine three out of a bro
    
[^89]: 连续对比微调改进低资源关系提取

    Continual Contrastive Finetuning Improves Low-Resource Relation Extraction. (arXiv:2212.10823v1 [cs.CL] CROSS LISTED)

    [http://arxiv.org/abs/2212.10823](http://arxiv.org/abs/2212.10823)

    本文提出了一种使用连续对比微调的方法来改进低资源关系提取，通过使用一致的对比学习目标预训练和微调RE模型，以及多中心对比损失来允许一个关系形成多个聚类。实验结果表明该方法可以显着提高低资源情况和领域中的关系提取性能。

    

    关系提取（RE）依赖结构化注释语料库进行模型训练，尤其在低资源情况和领域中，该任务具有挑战性。近期研究通过自监督学习来解决低资源的RE，其中解决方案包括通过RE目标预训练关系嵌入，并通过分类为基础的目标对有标签数据进行微调。然而，这种方法的一个关键挑战是目标之间的差距，它阻止RE模型充分利用预训练表示中的知识。本文旨在弥合差距，并提出使用一致的对比学习目标预训练和微调RE模型。由于在这种表示学习范式中，一个关系可能在表示空间中轻松形成多个聚类，因此我们进一步提出了多中心对比损失，允许一个关系形成多个聚类以更好地对齐预训练。在两个文档中的实验表明，所提出的方法可以在低资源情况和领域中显着提高关系提取性能。

    Relation extraction (RE), which has relied on structurally annotated corpora for model training, has been particularly challenging in low-resource scenarios and domains. Recent literature has tackled low-resource RE by self-supervised learning, where the solution involves pretraining the relation embedding by RE-based objective and finetuning on labeled data by classification-based objective. However, a critical challenge to this approach is the gap in objectives, which prevents the RE model from fully utilizing the knowledge in pretrained representations. In this paper, we aim at bridging the gap and propose to pretrain and finetune the RE model using consistent objectives of contrastive learning. Since in this kind of representation learning paradigm, one relation may easily form multiple clusters in the representation space, we further propose a multi-center contrastive loss that allows one relation to form multiple clusters to better align with pretraining. Experiments on two docum
    
[^90]: PropSegmEnt: 一个用于命题级别分割和包含关系识别的大规模语料库

    PropSegmEnt: A Large-Scale Corpus for Proposition-Level Segmentation and Entailment Recognition. (arXiv:2212.10750v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10750](http://arxiv.org/abs/2212.10750)

    这个论文提出了一个大规模的命题级别分割和包含关系识别的语料库PropSegmEnt解决了NLI中对语义单元的识别问题。

    

    自然语言推理(NLI)的任务需要识别一个文本是否可以从另一个文本中推断出来，通常在句子或段落级别上定义文本蕴含关系。然而，即使是简单的句子也往往包含多个命题，因此我们提出在句子中识别每个命题的文本包含关系。我们提供的PropSegmEnt数据集包括由专家评估员标注的超过45K个命题。我们的数据集结构类似于(1)将文档中的句子分段为命题的集合，以及(2)相对于一个不同的但与主题对齐的中心句子，对每个命题进行包含关系的分类。

    The widely studied task of Natural Language Inference (NLI) requires a system to recognize whether one piece of text is textually entailed by another, i.e. whether the entirety of its meaning can be inferred from the other. In current NLI datasets and models, textual entailment relations are typically defined on the sentence- or paragraph-level. However, even a simple sentence often contains multiple propositions, i.e. distinct units of meaning conveyed by the sentence. As these propositions can carry different truth values in the context of a given premise, we argue for the need to recognize the textual entailment relation of each proposition in a sentence individually.  We propose PropSegmEnt, a corpus of over 45K propositions annotated by expert human raters. Our dataset structure resembles the tasks of (1) segmenting sentences within a document to the set of propositions, and (2) classifying the entailment relation of each proposition with respect to a different yet topically-align
    
[^91]: 仅通过数据整理可以稳定上下文学习

    Data Curation Alone Can Stabilize In-context Learning. (arXiv:2212.10378v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10378](http://arxiv.org/abs/2212.10378)

    精心整理训练数据子集可以极大地稳定上下文学习表现，而不需要对ICL算法进行其他更改。CondAcc通过将训练示例与随机训练示例组合时的平均开发集ICL准确性来评分训练示例，而DataModels学习线性回归器，估计每个训练示例的存在如何影响LLM输出。

    

    上下文学习使得大型语言模型（LLM）通过一系列训练样例可以执行新任务。然而，已知ICL对训练样例的选择非常敏感：从训练集中随机抽样导致性能高度变异。在本文中，我们展示了精心整理训练数据子集可以极大地稳定ICL表现，而不需要ICL算法的其他更改（例如提示检索或校准）。我们引入了两种选择培训子集的方法——两者都单独评分培训示例，然后选择得分最高的示例。CondAcc通过将训练示例与随机训练示例组合时的平均开发集ICL准确性来评分训练示例，而DataModels学习线性回归器，估计每个训练示例的存在如何影响LLM输出。在五个任务和两个LLM的情况下，从由CondAcc和DataModels选择的稳定子集中抽样可以提高平均准确性。

    In-context learning (ICL) enables large language models (LLMs) to perform new tasks by prompting them with a sequence of training examples. However, it is known that ICL is very sensitive to the choice of training examples: randomly sampling examples from a training set leads to high variance in performance. In this paper, we show that carefully curating a subset of training data greatly stabilizes ICL performance without any other changes to the ICL algorithm (e.g., prompt retrieval or calibration). We introduce two methods to choose training subsets -- both score training examples individually, then select the highest-scoring ones. CondAcc scores a training example by its average dev-set ICL accuracy when combined with random training examples, while Datamodels learns linear regressors that estimate how the presence of each training example influences LLM outputs. Across five tasks and two LLMs, sampling from stable subsets selected by CondAcc and Datamodels improves average accuracy
    
[^92]: HINT：用于高效零及少样本泛化的超网络指令调整

    HINT: Hypernetwork Instruction Tuning for Efficient Zero- & Few-Shot Generalisation. (arXiv:2212.10315v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10315](http://arxiv.org/abs/2212.10315)

    本文介绍了一种新的NLP模型HINT，它使用超网络将任务指令和示例转换为参数高效的模块，从而无需将指令包含在模型输入中，并可为解码期间提供编码指令。HINT模型在计算量相等的情况下性能比最新的基线模型强10%以上，解决了高计算成本的问题。

    

    最近的NLP模型显示出了在新任务中只使用自然语言指导就能很好地推广“零样本”的非凡能力。然而，由于这些方法依赖于将冗长的指令与每个输入示例连接，导致指令的昂贵重新处理，因此许多方法存在高计算成本的问题。为了避免这一点，我们引入了用于指令调整的超级网络（HINT），它将任务指令和示例转换为参数高效的模块，使用预训练的文本编码器将其插入基础模型中，从而无需将指令包含在模型输入中。HINT中的超网络还产生了一种编码指令，我们在解码期间将其与编码输入连接起来以进一步提高性能。在控制计算（以FLOPs计量）的情况下，HINT模型的表现优于强有力的最新基线模型。通过将指令转换为模块，HINT模型可以有效地忽略指令。

    Recent NLP models have shown the remarkable ability to effectively generalise `zero-shot' to new tasks using only natural language instructions as guidance. However, many of these approaches suffer from high computational costs due to their reliance on concatenating lengthy instructions with every input example, resulting in costly reprocessing of the instruction. To avoid this, we introduce Hypernetworks for INstruction Tuning (HINT), which convert task instructions and examples into parameter-efficient modules inserted into an underlying model using a pretrained text encoder, eliminating the need to include instructions in the model input. The hypernetwork in HINT also produces an encoded instruction, which we concatenate with encoded inputs during decoding to further improve performance. HINT models outperform strong state-of-the-art baselines by over 10% when controlling for compute (measured in FLOPs). By converting instructions into modules, HINT models can effectively disregard 
    
[^93]: 语言模型是否具有日常物品的一致性心理模型？

    Do language models have coherent mental models of everyday things?. (arXiv:2212.10029v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10029](http://arxiv.org/abs/2212.10029)

    语言模型缺乏对日常物品的一致性心理模型，会因此出现荒谬的解决方法。虽然最先进的预训练语言模型具有这些实体的知识碎片，但它们无法为所有实体产生一致且正确的心理模型。语言模型训练可以改善这种情况。

    

    当人们想到像“鸡蛋”这样的日常用品时，通常会有一个与之相关联的心理图像。这种常识性知识有助于我们理解这些日常用品的工作原理以及如何与它们交互。然而，如果系统对这样的日常用品没有一致的图像，比如认为鸡蛋黄包围着壳，那么它可能不得不采取荒谬的方法，比如试图把鸡蛋黄刮下壳放入平底锅中煎煮。语言模型是否具有这种日常用品的一致性心理模型？为了调查这个问题，我们提出了一个基准数据集，包括100种日常用品、它们的部件以及这些部件之间的关系。我们观察到，像GPT-3和Macaw这样的最先进的预训练语言模型具有这些实体的知识碎片，但它们无法为所有实体产生一致且正确的心理模型。我们还发现，对这个基准数据集进行语言模型训练可以提高它们在某些方面的性能。

    When people think of everyday things like an "egg," they typically have a mental image associated with it. This commonsense knowledge helps us understand how these everyday things work and how to interact with them. For example, when someone tries to make a fried egg, they know that it has a shell and that it can be cracked open to reveal the egg white and yolk inside. However, if a system does not have a coherent picture of such everyday things, thinking that the egg yolk surrounds the shell, then it might have to resort to ridiculous approaches such as trying to scrape the egg yolk off the shell into the pan. Do language models have a coherent picture of such everyday things? To investigate this, we propose a benchmark dataset consisting of 100 everyday things, their parts, and the relationships between these parts. We observe that state-of-the-art pre-trained language models (LMs) like GPT-3 and Macaw have fragments of knowledge about these entities, but they fail to produce consist
    
[^94]: 人机协同评估早期误传信息检测：COVID-19治疗案例研究。

    Human-in-the-loop Evaluation for Early Misinformation Detection: A Case Study of COVID-19 Treatments. (arXiv:2212.09683v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09683](http://arxiv.org/abs/2212.09683)

    该论文提出了一种人机协同评估框架，用于检测新的虚假信息声明并识别支持它们的社交媒体消息。在COVID-19治疗的案例中，基于现代NLP方法开发基线系统，并展示了人类事实核查人员每小时可以识别出违反Twitter关于COVID-19虚假信息方针的124条推文。

    

    我们提出了一个人机协同评估框架，用于事实核查新的虚假信息声明并识别支持它们的社交媒体消息。我们的方法提取值得核查的声明，这些声明被聚合并排名以便复审。然后使用立场分类器来识别支持新虚假信息申述的推文，进一步检查以确定它们是否违反相关政策。为了展示我们的方法的可行性，我们在COVID-19治疗领域基于现代NLP方法开发了一个基线系统用于人机协同事实核查。使用我们的基线系统，我们展示了人类事实核查人员每小时能够识别出违反Twitter关于COVID-19虚假信息方针的124条推文。我们将提供我们的代码、数据、基线模型和详细注释指南来支持人机协同系统的评估，这些系统可以直接从原始用户生成的内容中识别新的虚假信息。

    We present a human-in-the-loop evaluation framework for fact-checking novel misinformation claims and identifying social media messages that support them. Our approach extracts check-worthy claims, which are aggregated and ranked for review. Stance classifiers are then used to identify tweets supporting novel misinformation claims, which are further reviewed to determine whether they violate relevant policies. To demonstrate the feasibility of our approach, we develop a baseline system based on modern NLP methods for human-in-the-loop fact-checking in the domain of COVID-19 treatments. Using our baseline system, we show that human fact-checkers can identify 124 tweets per hour that violate Twitter's policies on COVID-19 misinformation. We will make our code, data, baseline models, and detailed annotation guidelines available to support the evaluation of human-in-the-loop systems that identify novel misinformation directly from raw user-generated content.
    
[^95]: 自然语言处理中的代码交换研究：趋势和挑战的系统调查

    The Decades Progress on Code-Switching Research in NLP: A Systematic Survey on Trends and Challenges. (arXiv:2212.09660v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09660](http://arxiv.org/abs/2212.09660)

    本文系统调查了几十年来自然语言处理中关于代码交换的研究进展和挑战，总结了趋势和发现，并讨论了未来方向和进一步研究的开放性问题。

    

    代码交换在书面文本和口语交流中是一种常见现象，已经受到自然语言处理（NLP）研究界的研究。最初，通过运用语言学理论来深入探索代码交换，目前则采用更多面向机器学习的方法来开发模型。我们介绍了一篇全面的系统调查，旨在了解过去几十年代码交换研究的进展情况，并构思代码交换主题上的挑战和任务。最后，我们总结了趋势和发现，并讨论了未来方向和进一步研究的开放性问题。

    Code-Switching, a common phenomenon in written text and conversation, has been studied over decades by the natural language processing (NLP) research community. Initially, code-switching is intensively explored by leveraging linguistic theories and, currently, more machine-learning oriented approaches to develop models. We introduce a comprehensive systematic survey on code-switching research in natural language processing to understand the progress of the past decades and conceptualize the challenges and tasks on the code-switching topic. Finally, we summarize the trends and findings and conclude with a discussion for future direction and open questions for further investigation.
    
[^96]: BLOOM+1：为零样本提示添加语言支持

    BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting. (arXiv:2212.09535v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09535](http://arxiv.org/abs/2212.09535)

    本文在BLOOM模型中应用语言适应策略，将其适应到新语言上，并在八种新语言的零样本提示表现中提升了性能。适配器微调比大模型的持续预训练更有效，提示性能主要由语言适应数据的大小确定。

    

    BLOOM模型是一个大型公开的多语言语言模型，但其预训练仅限于46种语言。为了将BLOOM的好处扩展到其他语言，而不会产生过高的成本，有必要将BLOOM适应到新的语言上。本文将现有的语言适应策略应用于BLOOM，并在资源受限的情况下对其在八种新语言的零样本提示表现进行基准测试。我们发现，语言适应对于提高新语言的零样本性能是有效的。令人惊讶的是，我们发现适配器微调比大模型的持续预训练更有效。此外，我们发现提示性能不会受到语言特定性的显着影响，如书写系统。它主要由语言适应数据的大小确定。我们还向BLOOMZ添加了新语言，这是BLOOM的多任务微调版本，能够跟随提示。

    The BLOOM model is a large publicly available multilingual language model, but its pretraining was limited to 46 languages. To extend the benefits of BLOOM to other languages without incurring prohibitively large costs, it is desirable to adapt BLOOM to new languages not seen during pretraining. In this work, we apply existing language adaptation strategies to BLOOM and benchmark its zero-shot prompting performance on eight new languages in a resource-constrained setting. We find language adaptation to be effective at improving zero-shot performance in new languages. Surprisingly, we find that adapter-based finetuning is more effective than continued pretraining for large models. In addition, we discover that prompting performance is not significantly affected by language specifics, such as the writing system. It is primarily determined by the size of the language adaptation data. We also add new languages to BLOOMZ, which is a multitask finetuned version of BLOOM capable of following 
    
[^97]: 无格栅序列鉴别训练用于基于音素的神经传递器

    Lattice-Free Sequence Discriminative Training for Phoneme-Based Neural Transducers. (arXiv:2212.04325v3 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2212.04325](http://arxiv.org/abs/2212.04325)

    本文提出了三种无格栅训练目标，用于基于音素的神经传递器的最终后验输出，与使用N-best列表的方法相比，无格栅方法在训练期间消除了假设生成的步骤，从而导致更高效的训练，在单词错误率上表现也有6.5％的相对改进。

    

    最近，RNN-Transducer已经在各种自动语音识别任务中取得了显着成果。然而，无格栅序列鉴别训练方法，在混合模型中获得了优异的性能，但在RNN-Transducer中很少被研究。在本文中，我们提出了三个无格栅训练目标，即无格栅最大互信息、无格栅段级最小贝叶斯风险和无格栅最小贝叶斯风险，用于具有有限上下文依赖性的基于音素的神经传递器的最终后验输出。与使用N-best列表的方法相比，无格栅方法在训练期间消除了假设生成的解码步骤，从而导致更高效的训练。实验结果表明，与序列级交叉熵训练模型相比，无格栅方法在单词错误率上获得了高达6.5％的相对改进。与基于N-best列表的最小贝叶斯风险目标相比，无格栅方法具有更高的灵活性和可行性，尤其是在N-best列表中具有一些噪声和错误的情况下。

    Recently, RNN-Transducers have achieved remarkable results on various automatic speech recognition tasks. However, lattice-free sequence discriminative training methods, which obtain superior performance in hybrid models, are rarely investigated in RNN-Transducers. In this work, we propose three lattice-free training objectives, namely lattice-free maximum mutual information, lattice-free segment-level minimum Bayes risk, and lattice-free minimum Bayes risk, which are used for the final posterior output of the phoneme-based neural transducer with a limited context dependency. Compared to criteria using N-best lists, lattice-free methods eliminate the decoding step for hypotheses generation during training, which leads to more efficient training. Experimental results show that lattice-free methods gain up to 6.5% relative improvement in word error rate compared to a sequence-level cross-entropy trained model. Compared to the N-best-list based minimum Bayes risk objectives, lattice-free 
    
[^98]: 采用跨任务最近邻的数据高效微调方法

    Data-Efficient Finetuning Using Cross-Task Nearest Neighbors. (arXiv:2212.00196v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.00196](http://arxiv.org/abs/2212.00196)

    本文提出一种通过跨任务最近邻进行数据高效微调的方法，通过仅使用少量目标任务数据和多任务数据中的最相似标记数据，避免了对大量标记数据的需求，取得了比强基准模型更好的效果。

    

    获取用于训练感兴趣任务的标记数据通常代价高昂。先前的研究表明，在多任务数据上进行训练并加上任务描述（提示）有效地将知识传递给新任务。为了有效地构建任务特定模型，我们假设可以访问少量（32-1000）未标记的目标任务示例，并使用这些示例从包含提示的大量多任务数据中检索最相似的标记示例。与当前在均匀采样提示任务多任务数据（例如：FLAN、T0）上微调模型的做法相比，我们的方法在数据效率上显著更高。在没有任何标记的目标任务数据的情况下，仅使用 P3 池中 2% 的数据，我们的模型在 12 个代表保留任务的数据集（包括法律和科学文档 QA）中的性能要比在所有可用数据上进行训练的强基准模型高出 3-30%。采用跨任务最近邻训练的模型效果类似。

    Obtaining labeled data to train a model for a task of interest is often expensive. Prior work shows training models on multitask data augmented with task descriptions (prompts) effectively transfers knowledge to new tasks. Towards efficiently building task-specific models, we assume access to a small number (32-1000) of unlabeled target-task examples and use those to retrieve the most similar labeled examples from a large pool of multitask data augmented with prompts. Compared to the current practice of finetuning models on uniformly sampled prompted multitask data (e.g.: FLAN, T0), our approach of finetuning on cross-task nearest neighbors is significantly more data-efficient. Using only 2% of the data from the P3 pool without any labeled target-task data, our models outperform strong baselines trained on all available data by 3-30% on 12 out of 14 datasets representing held-out tasks including legal and scientific document QA. Similarly, models trained on cross-task nearest neighbors
    
[^99]: 评估和缩小合成语音和真实语音分布之间的差距

    Evaluating and reducing the distance between synthetic and real speech distributions. (arXiv:2211.16049v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2211.16049](http://arxiv.org/abs/2211.16049)

    本研究通过比较真实语音和合成语音的分布，使用统计学方法量化它们之间的距离，最终实现了10%的距离缩小。

    

    尽管现代TTS系统可以产生自然流畅的语音，但它们仍无法复现自然语音数据中发现的全部多样性。本研究考虑了一组发音者所能产生的所有可能真实语音样本的分布，以及使用特定TTS系统可以生成的所有合成样本的分布。我们通过一系列与发音者属性、语音韵律和声学环境相关的话语水平统计信息来量化真实语音和合成语音之间的距离，并使用Wasserstein距离评估这些统计信息分布的差异。通过在生成时提供基准值，我们缩小了这些距离，并使用自动语音识别系统来量化整体分布距离的改进情况。在我们的最佳系统中，分布距离缩小了10％。

    While modern Text-to-Speech (TTS) systems can produce natural-sounding speech, they remain unable to reproduce the full diversity found in natural speech data. We consider the distribution of all possible real speech samples that could be generated by these speakers alongside the distribution of all synthetic samples that could be generated for the same set of speakers, using a particular TTS system. We set out to quantify the distance between real and synthetic speech via a range of utterance-level statistics related to properties of the speaker, speech prosody and acoustic environment. Differences in the distribution of these statistics are evaluated using the Wasserstein distance. We reduce these distances by providing ground-truth values at generation time, and quantify the improvements to the overall distribution distance, approximated using an automatic speech recognition system. Our best system achieves a 10\% reduction in distribution distance.
    
[^100]: 全局和本地分层感知对比框架用于隐含篇章关系识别

    Global and Local Hierarchy-aware Contrastive Framework for Implicit Discourse Relation Recognition. (arXiv:2211.13873v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.13873](http://arxiv.org/abs/2211.13873)

    本文提出了GOLF框架，它能够充分利用全局和本地感知层次结构来提升隐含篇章关系识别效果。

    

    由于缺乏显式的连接词，隐含篇章关系识别(IDRR)仍然是篇章分析中的难题。IDRR的关键步骤是学习两个论点之间高质量的篇章关系表示。最近的方法趋向于将整个感知层次结构信息整合到篇章关系表示中进行多级别感知识别。然而，它们未能充分整合包含所有感知的静态分层结构（定义为全局分层结构），并忽略了与每个实例对应的层次感知标签序列（定义为本地分层结构）。为了充分利用全局和本地感知层次结构来学习更好的篇章关系表示，我们提出了一种新颖的全局和本地分层感知对比框架(GOLF)，借助于多任务学习和对比学习来模拟两种层次感知。在PDTB 2.0和PDTB-EDT语料库上的实验结果表明，所提出的GOLF在IDRR方面明显优于现有的最先进方法。

    Due to the absence of explicit connectives, implicit discourse relation recognition (IDRR) remains a challenging task in discourse analysis. The critical step for IDRR is to learn high-quality discourse relation representations between two arguments. Recent methods tend to integrate the whole hierarchical information of senses into discourse relation representations for multi-level sense recognition. Nevertheless, they insufficiently incorporate the static hierarchical structure containing all senses (defined as global hierarchy), and ignore the hierarchical sense label sequence corresponding to each instance (defined as local hierarchy). For the purpose of sufficiently exploiting global and local hierarchies of senses to learn better discourse relation representations, we propose a novel GlObal and Local Hierarchy-aware Contrastive Framework (GOLF), to model two kinds of hierarchies with the aid of multi-task learning and contrastive learning. Experimental results on PDTB 2.0 and PDTB
    
[^101]: NCTE 记录：一个小学数学课堂记录数据集

    The NCTE Transcripts: A Dataset of Elementary Math Classroom Transcripts. (arXiv:2211.11772v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.11772](http://arxiv.org/abs/2211.11772)

    NCTE记录提供了一个庞大的小学数学课堂记录数据集，它有助于研究课堂对话并改善教学质量。

    

    课堂话语是一种核心的教学媒介-分析它可以提供窥视教学和学习的视窗，并推动开发改善教学的新工具。我们引入了研究者可用的最大的数学课堂记录数据集，并证明了这些数据可以帮助提高教学质量。该数据集包括2010-2013年间 NCTE 收集的 1,660条 45-60 分钟的小学四五年级数学课堂记录数据，是最大的数据集之一。这些匿名记录来自于 4 个主要服务于历史上处于边缘化的学生的学区的 317 名教师。它们带有丰富的元数据，包括对话盘点上的注释、课堂观察得分、人口统计信息、调查回答和学生测试成绩。我们证明了基于对话盘点的自然语言处理模型可以以最先进的准确度学习在课堂上识别对话盘点的行为。我们还提供了此数据集如何用于研究课堂对话并改善教学的示例。

    Classroom discourse is a core medium of instruction - analyzing it can provide a window into teaching and learning as well as driving the development of new tools for improving instruction. We introduce the largest dataset of mathematics classroom transcripts available to researchers, and demonstrate how this data can help improve instruction. The dataset consists of 1,660 45-60 minute long 4th and 5th grade elementary mathematics observations collected by the National Center for Teacher Effectiveness (NCTE) between 2010-2013. The anonymized transcripts represent data from 317 teachers across 4 school districts that serve largely historically marginalized students. The transcripts come with rich metadata, including turn-level annotations for dialogic discourse moves, classroom observation scores, demographic information, survey responses and student test scores. We demonstrate that our natural language processing model, trained on our turn-level annotations, can learn to identify dialo
    
[^102]: 文本中的离群检测的多层知识蒸馏

    Multi-Level Knowledge Distillation for Out-of-Distribution Detection in Text. (arXiv:2211.11300v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.11300](http://arxiv.org/abs/2211.11300)

    本文提出了一种多层知识蒸馏方法，融合了语言模型的训练和微调方法来进行文本中的离群检测，实验结果表明其有效性。

    

    自监督表示学习已经证明是只使用内分布(ID)样例文本进行离群检测的宝贵组成部分。这些方法要么从头开始训练语言模型，要么通过使用ID样例微调预训练语言模型，然后将语言模型输出的困惑度作为离群得分。本文分析了两种离群检测方法的互补特性，并提出了一种融合它们优势并减轻它们的局限性的多层知识蒸馏方法。具体而言，我们使用微调模型作为老师，在ID示例上教授一个随机初始化的学生模型。除了预测层蒸馏外，我们还提出了一种基于相似性的中间层蒸馏方法，以全面探索老师模型的表示空间。通过这种方式，学习的学生可以更好地表示ID数据流形，同时获得更强的将OoD示例映射到流形之外的能力。基准数据集上的实验结果证明了我们所提出的方法与竞争基线相比的有效性。

    Self-supervised representation learning has proved to be a valuable component for out-of-distribution (OoD) detection with only the texts of in-distribution (ID) examples. These approaches either train a language model from scratch or fine-tune a pre-trained language model using ID examples, and then take the perplexity output by the language model as OoD scores. In this paper, we analyze the complementary characteristics of both OoD detection methods and propose a multi-level knowledge distillation approach that integrates their strengths while mitigating their limitations. Specifically, we use a fine-tuned model as the teacher to teach a randomly initialized student model on the ID examples. Besides the prediction layer distillation, we present a similarity-based intermediate layer distillation method to thoroughly explore the representation space of the teacher model. In this way, the learned student can better represent the ID data manifold while gaining a stronger ability to map O
    
[^103]: 使用说服性写作策略来解释和检测健康错误信息

    Using Persuasive Writing Strategies to Explain and Detect Health Misinformation. (arXiv:2211.05985v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.05985](http://arxiv.org/abs/2211.05985)

    本研究旨在通过使用说服性写作技巧的文本段落进行分类来增加自动化虚假信息检测的新层次，以产生可解释的理由。我们提出了一个包含常见说服性写作策略的注释方案和数据集，并使用 RoBERTa 文本分类模型进行实验。

    

    虚假信息的传播是当今社会的一大问题，许多学术界和工业界的研究人员正在努力解决这个问题。由于每天创造的虚假信息数量巨大，将此任务留给人工事实检查员是不切实际的。数据科学家和研究人员多年来一直致力于自动化虚假信息检测，但今天仍然是一个具有挑战性的问题。我们的研究目标是为自动化虚假信息检测添加一个新层次；使用具有说服性写作技巧的文本段落进行分类，以产生可解释的理由，说明为什么这篇文章可以标记为虚假信息。为此，我们提出了一个包含许多常见说服性写作策略的新注释方案，以及相应的人工注释数据集。我们使用 RoBERTa 文本分类模型来完成此任务，因为它在自然语言处理方面具有高性能。我们开发了几种基于语言模型的基线模型，并提供了结果分析。

    The spread of misinformation is a prominent problem in today's society, and many researchers in academia and industry are trying to combat it. Due to the vast amount of misinformation that is created every day, it is unrealistic to leave this task to human fact-checkers. Data scientists and researchers have been working on automated misinformation detection for years, and it is still a challenging problem today. The goal of our research is to add a new level to automated misinformation detection; classifying segments of text with persuasive writing techniques in order to produce interpretable reasoning for why an article can be marked as misinformation. To accomplish this, we present a novel annotation scheme containing many common persuasive writing tactics, along with a dataset with human annotations accordingly. For this task, we make use of a RoBERTa model for text classification, due to its high performance in NLP. We develop several language model-based baselines and present the 
    
[^104]: 对抗训练对语言模型的鲁棒性和泛化性的影响

    Impact of Adversarial Training on Robustness and Generalizability of Language Models. (arXiv:2211.05523v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.05523](http://arxiv.org/abs/2211.05523)

    本研究比较了在变压器语言模型中不同的对抗训练方法，并发现预训练数据增强或训练时间输入扰动可以实现更好的鲁棒性，而训练中使用的嵌入空间扰动可以显著提高泛化性。神经元的语言相关性分析表明，这些改进是由于存在“更专业”的神经元。这是首个对语言模型对抗训练不同方法进行全面比较的工作。

    

    对抗训练被广泛认为是防御对抗攻击的最有效手段。但是，已经确认对抗训练模型同时实现鲁棒性和泛化性需要进行权衡。本研究旨在深入比较语言模型中不同的对抗训练方法。具体而言，我们研究了在变压器语言模型中预训练数据增强、训练时间输入扰动和嵌入空间扰动对鲁棒性和泛化性的影响。我们的研究结果表明，通过预训练数据增强或训练时间输入扰动可以实现更好的鲁棒性。然而，通过嵌入空间扰动进行训练可以显著地提高泛化性。学习模型神经元的语言相关性分析表明，改善泛化性是由于存在“更专业”的神经元。据我们所知，这是首个对语言模型对抗训练不同方法进行全面比较的工作。

    Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of transformer-based language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveals that the improved generalization is due to 'more specialized' neurons. To the best of our knowledge,
    
[^105]: 基于自然语言解释的逻辑推理零样本分类

    Zero-Shot Classification by Logical Reasoning on Natural Language Explanations. (arXiv:2211.03252v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.03252](http://arxiv.org/abs/2211.03252)

    本文提出了一种基于逻辑推理的零样本分类方法，通过对自然语言解释进行分析和推理，将文本信息显式地编码成逻辑结构， 进而获得可靠的分类结果。

    

    人类可以通过对其语言解释的推理来分类未见过的数据类别。这种能力源于语言的组成性质：我们可以组合以前看到的属性来描述新的类别。本文受到这一观察的启发，通过对自然语言解释进行逻辑分析和推理，解决了零样本分类任务。为此，我们提出了CLORE框架（Classification by LOgical Reasoning on Explanations）。 CLORE将以前的方法所隐含的文本信息解析成逻辑结构并沿着这些结构推理，以产生分类分数。

    Humans can classify data of an unseen category by reasoning on its language explanations. This ability is owing to the compositional nature of language: we can combine previously seen attributes to describe the new category. For example, we might describe a sage thrasher as "it has a slim straight relatively short bill, yellow eyes and a long tail", so that others can use their knowledge of attributes "slim straight relatively short bill", "yellow eyes" and "long tail" to recognize a sage thrasher. Inspired by this observation, in this work we tackle zero-shot classification task by logically parsing and reasoning on natural language expla-nations. To this end, we propose the framework CLORE (Classification by LOgical Reasoning on Explanations). While previous methods usually regard textual information as implicit features, CLORE parses explanations into logical structures and then explicitly reasons along thess structures on the input to produce a classification score. Experimental re
    
[^106]: 基于交叉注意力的立场检测中的上下文信息整合

    Contextual information integration for stance detection via cross-attention. (arXiv:2211.01874v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.01874](http://arxiv.org/abs/2211.01874)

    本论文提出一种将来自异构数据源的上下文信息作为文本整合的方法，用于立场检测，取得了优于竞争基线的结果，对于未被提前见过的目标仍然有效。

    

    立场检测旨在确定作者对目标的立场。然而，大多数现有的立场检测模型存在局限性，因为它们没有考虑相关的上下文信息来正确地推断立场。为了解决这个问题，我们探索了一种方法，即将上下文信息作为文本进行整合。这种方法可以从异构数据源，如结构化知识源和大型语言模型，整合上下文信息，并可以克服标准知识库的图形结构对预训练语言模型的集成的复杂性。我们的方法在一个大型和多样化的立场检测基准测试中可以优于竞争基线，在交叉目标设置中，即针对在训练期间未见过的目标。我们证明它对噪声上下文更加鲁棒，并且可以为标签和目标特定词汇之间的不必要相关性进行正则化。最后，它是独立的。

    Stance detection deals with identifying an author's stance towards a target. Most existing stance detection models are limited because they do not consider relevant contextual information which allows for inferring the stance correctly. Complementary context can be found in knowledge bases but integrating the context into pretrained language models is non-trivial due to the graph structure of standard knowledge bases. To overcome this, we explore an approach to integrate contextual information as text which allows for integrating contextual information from heterogeneous sources, such as structured knowledge sources and by prompting large language models. Our approach can outperform competitive baselines on a large and diverse stance detection benchmark in a cross-target setup, i.e. for targets unseen during training. We demonstrate that it is more robust to noisy context and can regularize for unwanted correlations between labels and target-specific vocabulary. Finally, it is independ
    
[^107]: 基于随机话语拼接的数据增强用于提高短视频语音识别

    Random Utterance Concatenation Based Data Augmentation for Improving Short-video Speech Recognition. (arXiv:2210.15876v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2210.15876](http://arxiv.org/abs/2210.15876)

    本文提出了一种基于随机话语拼接的数据增强方法，用于提高短视频语音识别任务的训练和测试话语长度不匹配问题。该方法可显著提高长话语的识别率，同时对短话语的性能没有下降，并取得了5.72%的词错误率降低。

    

    端到端自动语音识别框架的局限性之一是在训练和测试话语长度不匹配时，其性能会受到影响。本文提出了一种即时基于随机话语拼接的数据增强方法，用于缓解短视频语音识别任务中的训练和测试话语长度不匹配问题。具体来说，我们针对观察到的人类转录的训练话语往往对于短视频自发语音（平均约3秒）要短得多的情况，而由语音活动检测前端生成的测试话语则要长得多（平均约10秒）。这种不匹配可能导致表现次优。实证表明，所提出的RUC方法显著提高了长话语的识别率，而对短话语的性能没有下降。总体而言，对于15种语言，该方法平均实现了5.72％的词错误率降低，并提高了对各种话语长度的稳健性。

    One of limitations in end-to-end automatic speech recognition (ASR) framework is its performance would be compromised if train-test utterance lengths are mismatched. In this paper, we propose an on-the-fly random utterance concatenation (RUC) based data augmentation method to alleviate train-test utterance length mismatch issue for short-video ASR task. Specifically, we are motivated by observations that our human-transcribed training utterances tend to be much shorter for short-video spontaneous speech (~3 seconds on average), while our test utterance generated from voice activity detection front-end is much longer (~10 seconds on average). Such a mismatch can lead to suboptimal performance. Empirically, it's observed the proposed RUC method significantly improves long utterance recognition without performance drop on short one. Overall, it achieves 5.72% word error rate reduction on average for 15 languages and improved robustness to various utterance length.
    
[^108]: 为了更加高效地将预训练语言模型应用于视频时间对齐，本文提出了一种方法

    Towards Parameter-Efficient Integration of Pre-Trained Language Models In Temporal Video Grounding. (arXiv:2209.13359v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.13359](http://arxiv.org/abs/2209.13359)

    本文探讨了如何更加高效地将预训练语言模型应用于视频时间对齐任务中，并通过将PLMs与现有方法相结合，证明了在三个具有挑战性的数据集上，TVG模型从PLM集成和微调中受益匪浅。

    

    本文旨在探讨视频时间对齐（TVG）任务，即在给定未修剪视频和自然语言句子查询的情况下，识别并确定与查询描述的视频动作实例的时间边界。最近的研究通过利用较大的预训练语言模型（PLM）改进查询输入来解决这个任务，但代价是更昂贵的训练费用。然而，这种集成的效果还不清楚，因为这些研究还提出了改进视觉输入的方法。因此，本文研究了PLM在TVG中的影响，并评估了使用NLP适配器进行参数高效训练的适用性。我们将流行的PLM与现有方法的选择结合使用，并测试不同的适配器以减少额外参数的影响。我们在三个具有挑战性的数据集上的实验结果显示，不改变视觉输入的情况下，TVG模型从PLM集成和微调中受益匪浅，强调了句子查询表示的重要性。

    This paper explores the task of Temporal Video Grounding (TVG) where, given an untrimmed video and a natural language sentence query, the goal is to recognize and determine temporal boundaries of action instances in the video described by the query. Recent works tackled this task by improving query inputs with large pre-trained language models (PLM) at the cost of more expensive training. However, the effects of this integration are unclear, as these works also propose improvements in the visual inputs. Therefore, this paper studies the effects of PLMs in TVG and assesses the applicability of parameter-efficient training with NLP adapters. We couple popular PLMs with a selection of existing approaches and test different adapters to reduce the impact of the additional parameters. Our results on three challenging datasets show that, without changing the visual inputs, TVG models greatly benefited from the PLM integration and fine-tuning, stressing the importance of sentence query represe
    
[^109]: 学习更好的掩蔽策略以实现更好的语言模型预训练

    Learning Better Masking for Better Language Model Pre-training. (arXiv:2208.10806v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2208.10806](http://arxiv.org/abs/2208.10806)

    本文提出了两种时间变化的MLM掩蔽策略，可以在不同的训练阶段自适应地调整掩蔽比例和掩蔽内容，提高语言模型的预训练效率和有效性。

    

    掩蔽语言建模（MLM）已被广泛用作预训练语言模型（PrLM）中的去噪目标。现有模型通常采用随机掩蔽策略，其中应用固定的掩蔽比例，并且以相等的概率掩蔽不同的内容。然而，模型可能会受到预训练状态的复杂影响，这种影响会随着训练时间的推移而变化。在本文中，我们展示了这种时间不变的MLM设置可能无法产生最佳结果，这促使我们探索时间变化的MLM设置的影响。我们提出了两种计划掩蔽方法，可以在不同的训练阶段自适应地调整掩蔽比例和掩蔽的内容，从而提高预训练效率和有效性，并在下游任务中得到验证。我们的工作是关于比率和内容的时间变化掩蔽策略的先驱研究，可以更好地理解和应用这些策略。

    Masked Language Modeling (MLM) has been widely used as the denoising objective in pre-training language models (PrLMs). Existing PrLMs commonly adopt a Random-Token Masking strategy where a fixed masking ratio is applied and different contents are masked by an equal probability throughout the entire training. However, the model may receive complicated impact from pre-training status, which changes accordingly as training time goes on. In this paper, we show that such time-invariant MLM settings on masking ratio and masked content are unlikely to deliver an optimal outcome, which motivates us to explore the influence of time-variant MLM settings. We propose two scheduled masking approaches that adaptively tune the masking ratio and masked content in different training stages, which improves the pre-training efficiency and effectiveness verified on the downstream tasks. Our work is a pioneer study on time-variant masking strategy on ratio and content and gives a better understanding of h
    
[^110]: DICE：基于生成模型的高效临床事件抽取

    DICE: Data-Efficient Clinical Event Extraction with Generative Models. (arXiv:2208.07989v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2208.07989](http://arxiv.org/abs/2208.07989)

    介绍了一种稳健高效的基于生成模型的临床事件抽取方法DICE，引入了对比性学习目标和特殊标记，共同训练实体提及和事件抽取等辅助任务，所提出的MACCROBAT-EE数据集为临床事件抽取提供了基准测试。

    

    临床领域的事件抽取是一个未被充分研究的研究领域。缺乏训练数据，以及领域特定术语的数量众多和实体界限模糊，使得这项任务尤其具有挑战性。本文介绍了DICE，一种稳健、高效的临床事件抽取生成模型。DICE将事件抽取作为条件生成问题，并引入对比性学习目标，以准确确定生物医学提及的边界。DICE还联合训练辅助提及标识任务和事件抽取任务，以更好地确定实体提及边界，并进一步引入特殊的标记来作为触发器和参数候选项，以包含其各自的任务中的确定实体问题。为了对临床事件抽取进行基准测试，我们根据现有的临床信息提取数据集MACCRO，构建了第一个带有参数批注的临床事件抽取数据集MACCROBAT-EE。

    Event extraction for the clinical domain is an under-explored research area. The lack of training data along with the high volume of domain-specific terminologies with vague entity boundaries makes the task especially challenging. In this paper, we introduce DICE, a robust and data-efficient generative model for clinical event extraction. DICE frames event extraction as a conditional generation problem and introduces a contrastive learning objective to accurately decide the boundaries of biomedical mentions. DICE also trains an auxiliary mention identification task jointly with event extraction tasks to better identify entity mention boundaries, and further introduces special markers to incorporate identified entity mentions as trigger and argument candidates for their respective tasks. To benchmark clinical event extraction, we compose MACCROBAT-EE, the first clinical event extraction dataset with argument annotation, based on an existing clinical information extraction dataset MACCRO
    
[^111]: Claim-Dissector: 一款联合重排和真实性预测的可解释的事实核查系统

    Claim-Dissector: An Interpretable Fact-Checking System with Joint Re-ranking and Veracity Prediction. (arXiv:2207.14116v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.14116](http://arxiv.org/abs/2207.14116)

    Claim-Dissector是一款联合重排和真实性预测的可解释的事实核查系统，可以识别与声明相关的证据，并确定声明的真实性。该系统的个人贡献以及证据所支持或反驳声明的贡献都可以被识别。

    

    我们提出了Claim-Dissector，一种针对事实核查和分析的新型潜变量模型，给出一个声明和一组检索到的证据，联合学习识别：（i）与给定声明相关的证据，（ii）声明的真实性。我们建议以可解释的方式解开每个证据的相关性概率及其对最终真实性概率的影响-最终真实性概率与每个证据相关性概率的线性整合成比例。通过这种方式，可以识别出每个证据对最终预测概率的个人贡献。在每个证据的相关性概率中，我们的模型还可以进一步区分每个相关证据是支持（S）还是反驳（R）声明。这样可以量化S/R概率对最终结论的贡献或检测有异议的证据。尽管我们的系统具有可解释性，但在FEVER竞赛中，其结果与最先进的结果相当。

    We present Claim-Dissector: a novel latent variable model for fact-checking and analysis, which given a claim and a set of retrieved evidences jointly learns to identify: (i) the relevant evidences to the given claim, (ii) the veracity of the claim. We propose to disentangle the per-evidence relevance probability and its contribution to the final veracity probability in an interpretable way -- the final veracity probability is proportional to a linear ensemble of per-evidence relevance probabilities. In this way, the individual contributions of evidences towards the final predicted probability can be identified. In per-evidence relevance probability, our model can further distinguish whether each relevant evidence is supporting (S) or refuting (R) the claim. This allows to quantify how much the S/R probability contributes to the final verdict or to detect disagreeing evidence.  Despite its interpretable nature, our system achieves results competitive with state-of-the-art on the FEVER 
    
[^112]: GENEVA：“通用性基准测试”事件论元提取，涵盖数百种事件类型和论元角色

    GENEVA: Benchmarking Generalizability for Event Argument Extraction with Hundreds of Event Types and Argument Roles. (arXiv:2205.12505v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12505](http://arxiv.org/abs/2205.12505)

    本文提出了一个大而全的EAE本体论，105个事件和220个论元角色的包含在内，利用这个本体论创建了一种多样化的通用性基准测试数据集GENEVA，共包含四个测试套件，旨在评估模型处理有限数据的能力。

    

    最近事件论元提取（EAE）的研究关注于提高模型的通用性以适应新的事件类型和领域。然而，标准的评估数据集如ACE和ERE只涵盖不到40种事件类型和25种面向实体的论元角色。数据集的有限多样性和覆盖范围影响了这些数据集对EAE模型通用性的充分评估。本文提出了一个大而全的EAE本体论，在FrameNet的基础上创建了包含115个事件和220个论元角色的本体论，其中许多角色不是实体。我们利用这个本体论进一步引入了GENEVA，一种多样化的通用性基准测试数据集，包括四个测试套件，旨在评估模型处理有限数据的能力。

    Recent works in Event Argument Extraction (EAE) have focused on improving model generalizability to cater to new events and domains. However, standard benchmarking datasets like ACE and ERE cover less than 40 event types and 25 entity-centric argument roles. Limited diversity and coverage hinder these datasets from adequately evaluating the generalizability of EAE models. In this paper, we first contribute by creating a large and diverse EAE ontology. This ontology is created by transforming FrameNet, a comprehensive semantic role labeling (SRL) dataset for EAE, by exploiting the similarity between these two tasks. Then, exhaustive human expert annotations are collected to build the ontology, concluding with 115 events and 220 argument roles, with a significant portion of roles not being entities. We utilize this ontology to further introduce GENEVA, a diverse generalizability benchmarking dataset comprising four test suites, aimed at evaluating models' ability to handle limited data a
    
[^113]: 科学发现的计算变革

    A Computational Inflection for Scientific Discovery. (arXiv:2205.02007v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.02007](http://arxiv.org/abs/2205.02007)

    本文介绍了一种计算变革的框架，利用最新的人工智能技术来增强科学发现和交流。这个框架有很多应用场景，作者提供了一个原型系统的初始实现，并探讨了未来研究和发展方向。

    

    我们正站在科学发现轨迹上一个重要的拐点上。随着社会的快速数字化转型，人类的科学知识和交流也在数字化的形式下不断增长。我们现在阅读和撰写的论文、预印本、书籍、代码、数据集、会议演示稿以及社交网络和协作和沟通平台上的交互等，大多已经以数字化的方式记录。这种转变导致了大量信息的创造和增长——其中很多已经可供公众获取——为分析和利用其的计算模型和系统开启了令人激动的机遇。与此同时，数据处理能力的指数增长推动了人工智能的显著进步，包括能够从非结构化文本中学习强大表示的大型神经语言模型。然而，需要进行重大改变，以在科学知识和交流的更大生态系统中有效整合这些进展，创建一种新的统一的科学交流范式。本文介绍了一种科学发现的计算变革——利用人工智能的最新进展，增强科学发现和交流的统一框架。我们展示了这个框架的潜力，提供了一个原型系统的初始实现，并讨论了未来的研究和发展方向。

    We stand at the foot of a significant inflection in the trajectory of scientific discovery. As society continues on its fast-paced digital transformation, so does humankind's collective scientific knowledge and discourse. We now read and write papers in digitized form, and a great deal of the formal and informal processes of science are captured digitally -including papers, preprints and books, code and datasets, conference presentations, and interactions in social networks and collaboration and communication platforms. The transition has led to the creation and growth of a tremendous amount of information -- much of which is available for public access -- opening exciting opportunities for computational models and systems that analyze and harness it. In parallel, exponential growth in data processing power has fueled remarkable advances in artificial intelligence, including large neural language models capable of learning powerful representations from unstructured text. Dramatic cha
    
[^114]: 探索更多的指导：一种增强数据增强的任务感知指令网络用于手语翻译

    Explore More Guidance: A Task-aware Instruction Network for Sign Language Translation Enhanced with Data Augmentation. (arXiv:2204.05953v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.05953](http://arxiv.org/abs/2204.05953)

    本研究提出了一种任务感知指令网络TIN-SLT用于手语翻译，引入指令模块和特征融合策略进一步提升了翻译性能，并且通过多层数据增强方案调整了数据分布。

    

    手语识别和翻译首先使用识别模块从手语视频中生成手语词汇，然后使用翻译模块将手语词汇翻译成口语句子。本文提出了一种针对手语翻译的任务感知指令网络TIN-SLT，通过将指令模块和基于学习的特征融合策略引入Transformer网络。这样，预训练模型的语言能力可以被充分探索和利用，进一步提升翻译性能。此外，通过探索手语词汇和目标口语的表示空间，我们提出了一种多层数据增强方案，调整训练集的数据分布。我们在两个挑战性基准数据集PHOENIX-2014-T和ASLG-PC12上进行了大量实验，证明了我们的方法的优越性能。

    Sign language recognition and translation first uses a recognition module to generate glosses from sign language videos and then employs a translation module to translate glosses into spoken sentences. Most existing works focus on the recognition step, while paying less attention to sign language translation. In this work, we propose a task-aware instruction network, namely TIN-SLT, for sign language translation, by introducing the instruction module and the learning-based feature fuse strategy into a Transformer network. In this way, the pre-trained model's language ability can be well explored and utilized to further boost the translation performance. Moreover, by exploring the representation space of sign language glosses and target spoken language, we propose a multi-level data augmentation scheme to adjust the data distribution of the training set. We conduct extensive experiments on two challenging benchmark datasets, PHOENIX-2014-T and ASLG-PC12, on which our method outperforms 
    
[^115]: 社交媒体中社会语用意义的对比学习

    Contrastive Learning of Sociopragmatic Meaning in Social Media. (arXiv:2203.07648v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.07648](http://arxiv.org/abs/2203.07648)

    提出了一种社交媒体中社会语用意义的对比学习框架，该框架能够学习可迁移的任务不可知表示学习，并在各种对比学习框架中表现最佳。

    

    最近自然语言处理中的表示学习和对比学习等研究进展尚未广泛考虑社会语用意义这一类别（即不同语言社区内的交流意义）。为了弥补这一空白，我们提出了一种新的框架，用于学习可迁移至各种社会语用任务（如情感、仇恨言论、幽默、讽刺）的任务不可知表示学习。我们的框架在领域内和领域外数据以及一般和少样本情况下的各种对比学习框架中表现最佳。例如，与两个流行的预训练语言模型相比，我们的方法在每个数据集仅用20个训练样本微调时，平均F1值在16个数据集上提高了11.66个百分点。

    Recent progress in representation and contrastive learning in NLP has not widely considered the class of \textit{sociopragmatic meaning} (i.e., meaning in interaction within different language communities). To bridge this gap, we propose a novel framework for learning task-agnostic representations transferable to a wide range of sociopragmatic tasks (e.g., emotion, hate speech, humor, sarcasm). Our framework outperforms other contrastive learning frameworks for both in-domain and out-of-domain data, across both the general and few-shot settings. For example, compared to two popular pre-trained language models, our method obtains an improvement of $11.66$ average $F_1$ on $16$ datasets when fine-tuned on only $20$ training samples per dataset.
    
[^116]: HyperMixer：一种基于MLP的低成本Transformer替代方案

    HyperMixer: An MLP-based Low Cost Alternative to Transformers. (arXiv:2203.03691v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.03691](http://arxiv.org/abs/2203.03691)

    HyperMixer是一种低成本的基于MLP的Transformer替代方案，通过动态形成标记混合MLP来实现自然语言理解，其性能比替代方案好，并可与Transformer媲美，成本更低。

    

    Transformer架构是自然语言理解的首选模型，但它们的成本相当高，因为它们在输入长度方面具有二次复杂度，需要大量的训练数据，并且可能难以调整。为了降低成本，我们研究了简单的基于MLP的架构。我们发现现有的架构（例如MLPMixer）通过静态的MLP独立地应用于每个特征，而过于脱离自然语言理解所需的归纳偏差。在本文中，我们提出了一种简单的改进，即HyperMixer，它使用超网络动态地形成标记混合MLP。实验上，我们证明了我们的模型表现优于替代的基于MLP的模型，并与Transformer媲美。与Transformer不同，HyperMixer在处理时间、训练数据和超参数调整方面具有大大降低的成本。

    Transformer-based architectures are the model of choice for natural language understanding, but they come at a significant cost, as they have quadratic complexity in the input length, require a lot of training data, and can be difficult to tune. In the pursuit of lower costs, we investigate simple MLP-based architectures. We find that existing architectures such as MLPMixer, which achieves token mixing through a static MLP applied to each feature independently, are too detached from the inductive biases required for natural language understanding. In this paper, we propose a simple variant, HyperMixer, which forms the token mixing MLP dynamically using hypernetworks. Empirically, we demonstrate that our model performs better than alternative MLP-based models, and on par with Transformers. In contrast to Transformers, HyperMixer achieves these results at substantially lower costs in terms of processing time, training data, and hyperparameter tuning.
    
[^117]: pNLP-Mixer：一种高效的全MLP架构用于自然语言处理

    pNLP-Mixer: an Efficient all-MLP Architecture for Language. (arXiv:2202.04350v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2202.04350](http://arxiv.org/abs/2202.04350)

    pNLP-Mixer是一种新型的MLP-Mixer模型，不需要嵌入层，用于设备上高效的自然语言处理，可以达到基于transformer架构的大型预训练语言模型相近的性能，却只需要很少的资源。

    

    基于Transformer架构的大型预训练语言模型已经彻底改变了自然语言处理(NLP)领域的格局。然而，在智能手表等受限设备上部署这些模型完全不可行，因为它们的大小和推理成本。作为Transformer架构的替代方案，最近关于高效NLP的工作表明，权重高效的模型可以在兆字节级的模型大小中获得简单任务(如槽填充和意图分类)的竞争性能。这项工作介绍了pNLP-Mixer架构，一种用于设备上NLP的无嵌入MLP-Mixer模型，由于采用了新颖的投影层，因此实现了高效的权重。我们在两个多语义解析数据集MTOP和multiATIS上评估了一个大小仅为1兆字节的pNLP-Mixer模型。我们的量化模型在MTOP和multi-ATIS上实现了mBERT的99.4％和97.8％的性能，同时使用的资源仅为mBERT的170倍。

    Large pre-trained language models based on transformer architecture have drastically changed the natural language processing (NLP) landscape. However, deploying those models for on-device applications in constrained devices such as smart watches is completely impractical due to their size and inference cost. As an alternative to transformer-based architectures, recent work on efficient NLP has shown that weight-efficient models can attain competitive performance for simple tasks, such as slot filling and intent classification, with model sizes in the order of the megabyte. This work introduces the pNLP-Mixer architecture, an embedding-free MLP-Mixer model for on-device NLP that achieves high weight-efficiency thanks to a novel projection layer. We evaluate a pNLP-Mixer model of only one megabyte in size on two multi-lingual semantic parsing datasets, MTOP and multiATIS. Our quantized model achieves 99.4% and 97.8% the performance of mBERT on MTOP and multi-ATIS, while using 170x fewer 
    

