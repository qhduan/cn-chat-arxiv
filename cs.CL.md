# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Diverse-Modal Entity Linking with Generative Models.](http://arxiv.org/abs/2305.17337) | 新方法提出了一个多模态实体链接的基准测试，并使用预训练的生成多模态模型，在平均F1分数上优于最先进的特定任务EL模型8.51分。 |
| [^2] | [Fine-Tuning Language Models with Just Forward Passes.](http://arxiv.org/abs/2305.17333) | 本论文提出了一种内存高效的零阶优化器，可以使用与推理相同的存储空间微调语言模型，其可以在大规模模型下更快地优化，具有更好的实验结果。 |
| [^3] | [Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In.](http://arxiv.org/abs/2305.17331) | 本文提出了增强适应检索器(AAR)的方案，通过从已知的源LM中学习LM的偏好，能够以通用插件的形式帮助目标LM在不进行微调的情况下显著提高零样本泛化能力。 |
| [^4] | [Why Does Zero-Shot Cross-Lingual Generation Fail? An Explanation and a Solution.](http://arxiv.org/abs/2305.17325) | 零样本跨语言生成失败的原因是神经网络模型在学习分类任务中的语言不变表示时，会影响在生成任务中的准确性，因此我们提出一种简单而有效的方法通过规范化模型来解决这个问题并提高生成质量。 |
| [^5] | [Beyond Positive Scaling: How Negation Impacts Scaling Trends of Language Models.](http://arxiv.org/abs/2305.17311) | 本研究介绍了一个包含否定问题的数据集NeQA，其中语言模型表现出反向缩放、U型缩放或正向缩放，解决NeQA依赖于问答和否定理解两个子任务，其缩放趋势由这两个子任务的缩放趋势组合形成。 |
| [^6] | [Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance.](http://arxiv.org/abs/2305.17306) | 本文介绍了一个名为 Chain-of-Thought Hub 的开源评估套件，目的是评估大型语言模型的多步推理能力。它是为了追踪LLMs进展而编制的具有挑战性的推理基准。目前的研究结果表明，模型规模与推理能力相关，而 Claude-v1.3 是迄今为止推理能力最强的LLM。 |
| [^7] | [External Language Model Integration for Factorized Neural Transducers.](http://arxiv.org/abs/2305.17304) | 该论文提出了一种外部语言模型在分解神经传输器中的集成方法，同时证明与浅层融合相比线性插值预测输出与神经和n-gram外部LM的结合可以提高准确性。结果显示平均18%的WERR增益，在一个实体丰富的情况下可以获得高达60%的WERR添加性增益。 |
| [^8] | [Improved Instruction Ordering in Recipe-Grounded Conversation.](http://arxiv.org/abs/2305.17280) | 本文针对基于菜谱的对话任务，提出了两个辅助子任务，即用户意图检测和指令状态跟踪，并证明这些任务可以帮助响应生成模型解决指令顺序错误问题。 |
| [^9] | [Slide, Constrain, Parse, Repeat: Synchronous SlidingWindows for Document AMR Parsing.](http://arxiv.org/abs/2305.17273) | 本文提出了一种同步滑动窗口的方法来处理文档解析的序列到序列任务，利用源-目标对齐并约束解码以保证重叠窗口的同步性和一致性，在AMR 3.0的评估中展示出了高质量的性能。 |
| [^10] | [Metaphor Detection via Explicit Basic Meanings Modelling.](http://arxiv.org/abs/2305.17268) | 本文提出了一种新的隐喻检测方法，通过对训练集中的字面注释进行基本含义建模并将其与上下文含义进行比较，可以更准确地识别隐喻，表现优于现有方法1.0％。 |
| [^11] | [CODET: A Benchmark for Contrastive Dialectal Evaluation of Machine Translation.](http://arxiv.org/abs/2305.17267) | CODET是一个对比方言的评估基准测试，用于评估机器翻译系统在处理方言变体时的表现，该基准测试包含九种不同语言的882个不同变体。 |
| [^12] | [Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale.](http://arxiv.org/abs/2305.17266) | 本文研究了小规模语言模型的训练效果，并展示了掩码语言建模目标的预训练对性能的提高作用。同时，该研究还发现了计算成本与模型效果之间的相关性。 |
| [^13] | [Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning.](http://arxiv.org/abs/2305.17256) | 本文探讨了大型语言模型在上下文学习中利用提示中的捷径的依赖性，发现大型模型更有可能在推理过程中利用提示中的捷径，这为评估上下文学习的稳健性和检测和缓解提示中捷径的使用提供了新的视角和挑战。 |
| [^14] | [Federated Learning for Semantic Parsing: Task Formulation, Evaluation Setup, New Algorithms.](http://arxiv.org/abs/2305.17221) | 本文研究了基于联邦学习的语义解析任务，提出了评估设置和新算法。实验表明，新算法FedSQL和Lorar优于现有的FL算法和我们提出的设置的强基线。 |
| [^15] | [GVdoc: Graph-based Visual Document Classification.](http://arxiv.org/abs/2305.17219) | GVdoc 是一个基于图的文档分类模型，能够通过生成文档图并训练图神经网络来学习节点和图嵌入，有效解决视觉文档分类器在领域内外样本分类和区分中所遇到的挑战。 |
| [^16] | [Generating Images with Multimodal Language Models.](http://arxiv.org/abs/2305.17216) | 该论文提出了一种方法，将大型语言模型与预训练的图像编码器和解码器模型进行融合，能生成具有连贯性的图像输出，同时也能进行图像检索和多模态对话。 |
| [^17] | [BIG-C: a Multimodal Multi-Purpose Dataset for Bemba.](http://arxiv.org/abs/2305.17202) | BIG-C是一个用于Bemba语的大型多模态数据集，提供了语音识别、机器翻译、语音翻译任务的基线，并勾画了该数据集的潜在未来多模态用途，旨在促进研究并鼓励跨越语言、语音和视觉社区的合作，特别是针对“传统”使用的语言之外的语言。 |
| [^18] | [Entailment as Robust Self-Learner.](http://arxiv.org/abs/2305.17197) | 本文提出了一种将许多不同的NLU任务制定为情境认知的提示策略，并通过自我训练来提高模型的适应性性能。简单的伪标签编辑（SimPLE）算法有利于自我训练的稳定改进。 |
| [^19] | [On the Copying Problem of Unsupervised NMT: A Training Schedule with a Language Discriminator Loss.](http://arxiv.org/abs/2305.17182) | 无监督NMT中的复制问题通常发生在远距离语种对中且会直接复制输入句子的部分作为翻译，本研究提出了一种包含语言鉴别器损失的训练计划来缓解该问题，并提高低资源语种的翻译性能。 |
| [^20] | [Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages.](http://arxiv.org/abs/2305.17179) | 本论文提出了新的标准来评估多语言语言模型中分词器的质量，并发现跨语言词汇的重叠会对某些下游任务产生不利影响，但共享词汇有助于其他任务。研究还发现，多语言词汇中的语言特定标记覆盖范围对单词级任务产生显著影响。这些发现对未来的模型开发人员选择最合适的分词器提供了指南。 |
| [^21] | [From Dogwhistles to Bullhorns: Unveiling Coded Rhetoric with Language Models.](http://arxiv.org/abs/2305.17174) | 本文首次开展了对犬哨现象进行的大规模计算研究，发现犬哨词汇可向不同群体传达不同的含义和挑衅性，同时解释了它们如何规避政治后果和算法内容调节。 |
| [^22] | [Heterogeneous Value Evaluation for Large Language Models.](http://arxiv.org/abs/2305.17147) | 本文提出了一种自动对齐评估方法A2EHV，采用异质价值系统，并基于价值合理性和社会价值定向框架评估代理人行为的社会偏好，结果表明比传统对齐方法更合理。 |
| [^23] | [Ghost in the Minecraft: Generally Capable Agents for Open-World Enviroments via Large Language Models with Text-based Knowledge and Memory.](http://arxiv.org/abs/2305.17144) | 本文提出了Ghost in the Minecraft (GITM)框架，利用大型语言模型与基于文本的知识和记忆，创造了一种在Minecraft中具备通用能力的智能体，可在以文本为基础的复杂编程环境中熟练导航。 |
| [^24] | [NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models.](http://arxiv.org/abs/2305.16986) | NavGPT是基于LLM的导航智能体，可以在视觉语言导航（VLN）中，通过对文本描述进行推理，执行零-shot连续动作预测。该模型具有高级规划能力，可以将指令分解成子目标、整合常识知识以进行障碍物避免，并参考先前的步骤进行澄清。NavGPT展示了通用体现智能体发展的美好前景。 |
| [^25] | [Domain Aligned Prefix Averaging for Domain Generalization in Abstractive Summarization.](http://arxiv.org/abs/2305.16820) | 本文提出了一种轻量级、基于加权平均的领域对齐前缀平均方法（DAPA），用于抽象摘要中的领域泛化，实现了有效的源域扩展以提高性能。 |
| [^26] | [InterFormer: Interactive Local and Global Features Fusion for Automatic Speech Recognition.](http://arxiv.org/abs/2305.16342) | 本文提出了InterFormer，用于交互式局部和全局特征融合，以学习更好的ASR表示。通过组合卷积块和变形器块，以及引入BFIM和SFM模块，实现了局部和全局特征的交互和融合，取得了在公共ASR数据集上优异的性能。 |
| [^27] | [ASR and Emotional Speech: A Word-Level Investigation of the Mutual Impact of Speech and Emotion Recognition.](http://arxiv.org/abs/2305.16065) | 本论文研究了ASR技术在情感语音上的表现，并探究了情感如何影响ASR。同时，还研究了ASR对基于文本的情感识别的影响。该研究旨在揭示ASR和SER之间的关系和相互影响，以促进ASR技术对情感语音的适应和SER技术在实际中的应用。 |
| [^28] | [Emergence of a phonological bias in ChatGPT.](http://arxiv.org/abs/2305.15929) | ChatGPT表现出人类语言处理的音韵偏见，更倾向于使用辅音而不是元音来识别单词。 |
| [^29] | [MEMEX: Detecting Explanatory Evidence for Memes via Knowledge-Enriched Contextualization.](http://arxiv.org/abs/2305.15913) | 本研究提出了MEMEX任务，通过知识增强的上下文化技术检测迷因的解释性证据。通过构建MCC数据集，使用分层方法捕捉迷因和上下文的跨模态语义依赖，提出了MIME多模式神经框架来解释迷因。 |
| [^30] | [Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers.](http://arxiv.org/abs/2305.15805) | 本研究提出了一种动态上下文剪枝方法，可以在保持模型表现力的同时，动态减少无效信息，提高模型的效率和可解释性。该技术可以应用于现有的预训练模型，并且可以通过简单的微调过程实现。 |
| [^31] | [Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline.](http://arxiv.org/abs/2305.13144) | 本文提出了一种利用LLMs的高效LLM推理流水线，通过利用LLMs准确感知和预测响应长度的潜力，并引入一种高效的序列调度技术，将具有类似响应长度的查询分组成微批。实验结果表明，该方法在实现高效的推理吞吐量的同时也不影响有效性。 |
| [^32] | [Text-to-SQL Error Correction with Language Models of Code.](http://arxiv.org/abs/2305.13073) | 本论文提出了一种基于从句编辑模型的文本到SQL的语言模型纠错方法，并通过新的SQL查询表示改进了语言模型的精确匹配准确率，提高了2.4-6.5，最多提高4.3个百分点。 |
| [^33] | [Gloss-Free End-to-End Sign Language Translation.](http://arxiv.org/abs/2305.12876) | 本论文介绍了一种无需手语标注的全端到端手语翻译框架(GloFE)，它通过利用手语和相应口语翻译的共同基础语义来提高手语翻译的性能，可以应用于实际场景中。 |
| [^34] | [Learning Optimal Policy for Simultaneous Machine Translation via Binary Search.](http://arxiv.org/abs/2305.12774) | 本文提出通过二分搜索学习同时机器翻译最优策略的方法，并在多个翻译任务上验证了在所有延迟情景下超越强基线的效果。 |
| [^35] | [Exploring Energy-based Language Models with Different Architectures and Training Methods for Speech Recognition.](http://arxiv.org/abs/2305.12676) | 本文探索了不同的能量函数架构和不同的训练方法，以提高基于能量的语言模型在语音识别中计算句子得分的能力。 |
| [^36] | [Contextualized End-to-End Speech Recognition with Contextual Phrase Prediction Network.](http://arxiv.org/abs/2305.12493) | 本研究引入了一个上下文短语预测网络用于基于注意力的深度偏置方法，通过计算偏置损失以帮助训练上下文化模型，在多种端到端语音识别模型上实现了显著的WER降低，相对于基线模型相对WER提高了12.1％，上下文短语的WER相对降低了40.5％。 |
| [^37] | [Post Hoc Explanations of Language Models Can Improve Language Models.](http://arxiv.org/abs/2305.11426) | 本文提出了一种新的框架AMPLIFY，利用后验解释自动化生成原因，并在多个数据集和任务上显著提高现有语言模型的性能。 |
| [^38] | [ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing.](http://arxiv.org/abs/2305.09770) | ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。 |
| [^39] | [A Crosslingual Investigation of Conceptualization in 1335 Languages.](http://arxiv.org/abs/2305.08475) | 通过一个新方法，该论文探索了1335种语言之间在概念表达方面的差异，并发现了能够预测一种概念是否会跨越多种语言，以及如何代表每种语言的多维分类方法，这有助于解决现有自然语言处理模型的不足。 |
| [^40] | [From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models.](http://arxiv.org/abs/2305.08283) | 本文研究测量了政治偏见在预训练语言模型和下游任务中的影响，发现预训练模型存在政治倾向，并将社会偏见传递到下游任务中，从而导致NLP模型的不公平性。 |
| [^41] | [Zero-shot Faithful Factual Error Correction.](http://arxiv.org/abs/2305.07982) | 零样本的忠实事实性错误纠正框架超越完全监督方法，具有较高的及解释性和忠实性评估标准，适用于维护文本知识库和预防序列到序列模型中的幻觉。 |
| [^42] | [Decker: Double Check with Heterogeneous Knowledge for Commonsense Fact Verification.](http://arxiv.org/abs/2305.05921) | Decker是一种能够利用结构化和非结构化知识之间的潜在关系来桥接异构知识的常识事实验证模型，具有良好的验证效果和获取珍贵信息的能力。 |
| [^43] | [FACTIFY-5WQA: 5W Aspect-based Fact Verification through Question Answering.](http://arxiv.org/abs/2305.04329) | 本文提出了一个基于问题回答的5W因素事实验证框架，并提供了一个半自动产生的数据集FACTIFY-5WQA，以便协助人类核查员针对事实提出相关问题并验证以达出最终结论。实验结果表明，这种方式相较于其他基线模型和常规事实验证系统具有更高的效率和可行性。 |
| [^44] | [Vision Meets Definitions: Unsupervised Visual Word Sense Disambiguation Incorporating Gloss Information.](http://arxiv.org/abs/2305.01788) | 本文提出了一种无监督的视觉词义消歧方法，通过引入外部词汇知识库的词义信息来解决原来图像-文本匹配模型中的多义词问题。采用贝叶斯推断来加入词义定义，并通过与上下文相关的 GPT-3 定义生成方法，成功解决了词典外问题。 |
| [^45] | [Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model.](http://arxiv.org/abs/2304.13731) | 本研究提出了一种使用指令调整的LLM Flan-T5作为文本编码器和基于潜在扩散模型(LDM)的方法TANGO生成文本到音频(TTA)的新方法，在AudioCaps测试集上表现优于先进的AudioLDM。 |
| [^46] | [Fundamental Limitations of Alignment in Large Language Models.](http://arxiv.org/abs/2304.11082) | 本文通过提出一种理论方法——行为期望边界（BEB），展示了大型语言模型中对齐的基本限制，并证明任何对齐过程都无法根除不希望的行为，这对于防止恶意攻击是不安全的。 |
| [^47] | [Which Factors Predict the Chat Experience of a Natural Language Generation Dialogue Service?.](http://arxiv.org/abs/2304.10785) | 本文研究了自然语言生成对话系统中影响聊天体验的多种因素，包括提示、连贯性、情感、相似性和用户对话代理的好感度，发现用户的好感度和连贯性、情感、相似性是聊天体验的正向预测因素。此外，用户可能更喜欢具有外向性、开放性、责任心、宜人性和非神经质特征的对话代理。 |
| [^48] | [Learning to Program with Natural Language.](http://arxiv.org/abs/2304.10464) | 该论文提出了一种用自然语言作为编程语言并通过学习编程方法让大语言模型直接生成自然语言程序并指导推理的方法。实验结果表明，这种方法在解决编程任务上比基线方法有更高的成功率。 |
| [^49] | [Comparing Abstractive Summaries Generated by ChatGPT to Real Summaries Through Blinded Reviewers and Text Classification Algorithms.](http://arxiv.org/abs/2303.17650) | 本研究评估了ChatGPT在抽象概括方面的表现，自动化指标和盲审人员评估显示ChatGPT生成的摘要在人类视角下难以分辨真假。 |
| [^50] | [ChatGPT4PCG Competition: Character-like Level Generation for Science Birds.](http://arxiv.org/abs/2303.15662) | 本论文介绍了举办在2023 IEEE游戏会议上的第一届ChatGPT4PCG比赛，目标是让ChatGPT生成具有高稳定性和类似角色的特质来生成具有科学鸟角色级水平的关卡。 |
| [^51] | [Direct and indirect evidence of compression of word lengths. Zipf's law of abbreviation revisited.](http://arxiv.org/abs/2303.10128) | 这篇论文重新审视了书面语与缩写定律的一致性，并发现这一定律也适用于口语。结果提供了压缩语言的间接证据，即缩写定律是最优编码的预测，而通过英语的历史研究还发现，人们在语言中实际使用的词汇数量正在缩减。 |
| [^52] | [CB2: Collaborative Natural Language Interaction Research Platform.](http://arxiv.org/abs/2303.08127) | CB2是一个用于研究基于任务的合作自然语言交互的平台，在3D游戏环境中提供了后端服务器和各种工具和流程。它在可扩展的研究中展示了学习的指令跟随模型。 |
| [^53] | [Dynamic Prompting: A Unified Framework for Prompt Tuning.](http://arxiv.org/abs/2303.02909) | 本论文提出了一个统一的动态提示（DP）调整策略用于优化提示调整的性能，该策略可以动态地确定不同的提示变量来捕获额外的语义信息。 |
| [^54] | [Inseq: An Interpretability Toolkit for Sequence Generation Models.](http://arxiv.org/abs/2302.13942) | 本文介绍了Inseq，这是一个Python工具包，旨在推广可解释性序列生成模型的分析。它为常见的解码器和编码器-解码器Transformers架构提供了提取模型内部信息和特征重要性得分的直观优化方法。作者还在机器翻译模型和GPT-2中展示了Inseq的潜力，证明其有助于推动可解释性自然语言生成的未来发展。 |
| [^55] | [SanskritShala: A Neural Sanskrit NLP Toolkit with Web-Based Interface for Pedagogical and Annotation Purposes.](http://arxiv.org/abs/2302.09527) | SanskritShala是第一个使用基于Web界面和互动数据注释功能的神经梵文自然语言处理（NLP）工具包，为词语分割、形态标记、依赖解析和复合型识别等任务提供最先进的性能。 |
| [^56] | [Double Permutation Equivariance for Knowledge Graph Completion.](http://arxiv.org/abs/2302.01313) | 本研究提出了双排列等变性的KG表示方法，可以使神经网络在KG中执行复杂的逻辑推理任务，并在多个归纳KG完成任务中实现了最先进的Hits@10测试准确率。双排列等变性在KG中开辟了新的研究方向。 |
| [^57] | [Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation.](http://arxiv.org/abs/2301.13003) | 本文提出了一种分层蒸馏技术，在声学和语言级别上将预训练语言模型（PLMs）的知识转移到基于CIF的自动语音识别（ASR）模型，相较原始模型，在AISHELL-1和LibriSpeech数据集上分别实现了15%和9%的相对误差率降低。 |
| [^58] | [Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity.](http://arxiv.org/abs/2301.12867) | 这篇论文通过对OpenAI的ChatGPT进行越狱技术实验的方法，研究了它的可靠性、鲁棒性、偏见和毒性等问题。研究发现，ChatGPT存在种族、性别和宗教等相关偏见，并容易受到对抗性攻击的影响，因此建议在开发和部署LLMs时应考虑透明度和问责制问题。 |
| [^59] | [Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining.](http://arxiv.org/abs/2301.12596) | 本研究利用仅文本数据进行零样本多语言TTS，允许开发低资源语言的TTS系统，评估结果表明具有高度可理解的零样本TTS。 |
| [^60] | [Theme-driven Keyphrase Extraction to Analyze Social Media Discourse.](http://arxiv.org/abs/2301.11508) | 本论文介绍了一种针对社交媒体的主题驱动的关键词提取框架，旨在从用户生成的健康文本中捕获临床相关的关键词，并展示了它在防治阿片类物质使用障碍的用例中的潜力。 |
| [^61] | [Define, Evaluate, and Improve Task-Oriented Cognitive Capabilities for Instruction Generation Models.](http://arxiv.org/abs/2301.05149) | 本文提出了基于任务的认知能力，设计了评估方案来比较语言模型和人类的这些能力，通过在导航指令生成问题中的应用，发现模型的语用能力仍需改进。 |
| [^62] | [Black-box language model explanation by context length probing.](http://arxiv.org/abs/2212.14815) | 该论文提出了一个模型不可知的新颖解释技术：上下文长度探测，通过跟踪模型预测与可用上下文长度的关系来对不同上下文分配不同的重要性得分。该方法适用于大型预训练语言模型，并有利于研究远距离依赖性。 |
| [^63] | [Resolving Indirect Referring Expressions for Entity Selection.](http://arxiv.org/abs/2212.10933) | 该研究旨在解决人类使用自然语言进行实体选择时所面临的间接引用表达式的问题，研究人员创建了一个包含42K个实体对的公共数据集，并开发了模型解决这个问题。 |
| [^64] | [SERENGETI: Massively Multilingual Language Models for Africa.](http://arxiv.org/abs/2212.10785) | SERENGETI是一个大规模多语言语言模型，覆盖了517种非洲语言和语言方言。在自然语言理解任务中，它的表现优于其他在非洲语言上的语言模型，能够提供有价值的语言信息。 |
| [^65] | [Lego-MT: Towards Detachable Models in Massively Multilingual Machine Translation.](http://arxiv.org/abs/2212.10551) | 本文提出了一种可拆卸的多语言机器翻译模型，Lego-MT，以解决现有多语言单体模型在参数干扰和低效推导方面的挑战。进行实验评估表明，该模型具有较高的性能，相比具有10倍规模的模型，在效率和表现方面都更具优势。 |
| [^66] | [Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts.](http://arxiv.org/abs/2212.10543) | MaRCo是一种排毒算法，能够使用专家和反专家模型对文本进行可控的重写和修订，适用于消除微妙的有害信息，且在自动化指标和人类评估中均有很好的表现。 |
| [^67] | [When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories.](http://arxiv.org/abs/2212.10511) | 本文通过对10个模型和4种增强方法的实验，发现语言模型在记忆不太流行的实际知识方面存在困难，而检索增强的语言模型表现较好，提出了一种检索增强语言模型的简单有效方法。 |
| [^68] | [DOC: Improving Long Story Coherence With Detailed Outline Control.](http://arxiv.org/abs/2212.10077) | 该论文提出了一个名为 Detailed Outline Control(DOC) 的框架，通过详细大纲和详细控制器来提高生成长篇故事时的情节连贯性和大纲相关性，人类评估证实该方法在这些方面显著优于基线方法，并且更适用于交互生成设置。 |
| [^69] | [WeCheck: Strong Factual Consistency Checker via Weakly Supervised Learning.](http://arxiv.org/abs/2212.10057) | 本研究提出了一种弱监督框架WeCheck，通过聚合多种资源训练模型，准确和高效地检查文本生成模型是否存在实际事实一致性问题。 |
| [^70] | [Reasoning with Language Model Prompting: A Survey.](http://arxiv.org/abs/2212.09597) | 本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。 |
| [^71] | [BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting.](http://arxiv.org/abs/2212.09535) | 本文在BLOOM模型中应用语言适应策略，将其适应到新语言上，并在八种新语言的零样本提示表现中提升了性能。适配器微调比大模型的持续预训练更有效，提示性能主要由语言适应数据的大小确定。 |
| [^72] | [An Extensible Plug-and-Play Method for Multi-Aspect Controllable Text Generation.](http://arxiv.org/abs/2212.09387) | 本论文提出了一种基于可训练门的多方面可控文本生成方法，用于规范前缀的干预，从而实现对训练时未见过的方面组合的控制，具有良好的可扩展性和性能表现。 |
| [^73] | [PAL: Persona-Augmented Emotional Support Conversation Generation.](http://arxiv.org/abs/2212.09235) | 本文提出了一个动态推断和建模求助者人格的框架，并结合基于策略的可控生成方法提供个性化情感支持，实证分析表明人格对情感支持有重要影响。 |
| [^74] | [On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning.](http://arxiv.org/abs/2212.09170) | 本文通过几何学角度在对比句子表示学习中发现，对比学习带来了各向同性，并驱动同一句子中标记在语义空间中收敛到相似的位置。对于语义上有意义的标记，"虚假的情境化"得到了缓解，而对于功能性标记则被增强。 |
| [^75] | [Modeling Instance Interactions for Joint Information Extraction with Neural High-Order Conditional Random Field.](http://arxiv.org/abs/2212.08929) | 本文介绍了一种利用神经高阶条件随机场进行联合信息抽取的框架，可以更好地集成跨实例交互，并通过高阶神经解码器解决了精确高阶推理的难解性问题。 |
| [^76] | [UniSumm and SummZoo: Unified Model and Diverse Benchmark for Few-Shot Summarization.](http://arxiv.org/abs/2211.09783) | 该论文提出了UniSumm统一的少样本摘要模型，可以通过前缀调整应对任何少样本摘要任务，同时，他们也发布了一个新的基准SummZoo，其由8个摘要任务组成，每个任务都涵盖了多个少样本样本集，以此更好地评估少样本摘要器。 |
| [^77] | [mOKB6: A Multilingual Open Knowledge Base Completion Benchmark.](http://arxiv.org/abs/2211.06959) | 该论文使用最新的多语言开放信息提取技术构建了第一个名为mOKB6的多语言Open KBC数据集，旨在补全开放知识库。研究结果表明，将多语言组合起来有一致的好处，但当前的多语言模型存在困难。 |
| [^78] | [lilGym: Natural Language Visual Reasoning with Reinforcement Learning.](http://arxiv.org/abs/2211.01994) | 本文提出了一个基于自然语言视觉推理的强化学习基准测试——lilGym，它由2661个高度组合的人类编写自然语言语句和交互式视觉环境组成，并通过注释可执行Python程序来实现精确的奖励计算。本文的实验结果和分析表明，lilGym是一个具有挑战性的开放性问题。 |
| [^79] | [Crosslingual Generalization through Multitask Finetuning.](http://arxiv.org/abs/2211.01786) | 该论文通过多任务微调实现跨语言泛化。研究表明，在英语提示下，对大型多语言模型进行英语任务的微调，可以实现对仅出现在预训练语料库中的非英语语言的任务泛化，并且使用英语提示进行多语言任务的微调进一步提高了在英语和非英语任务上的表现，从而实现了各种零-shot结果的最新水平。 |
| [^80] | [Solving Math Word Problems via Cooperative Reasoning induced Language Models.](http://arxiv.org/abs/2210.16257) | 该论文提出了一种基于合作推理诱导的语言模型——CoRe，可以高效地解决数学应用题。实验表明，CoRe 在准确性和效率方面优于现有最先进的方法。 |
| [^81] | [Dense-ATOMIC: Towards Densely-connected ATOMIC with High Knowledge Coverage and Massive Multi-hop Paths.](http://arxiv.org/abs/2210.07621) | 本文旨在构建具有高知识覆盖率和大规模多跳路径的Dense-ATOMIC。我们提出了一种名为Rel-CSKGC的CSKG完成方法，用于预测三元组的头事件和尾事件后的关系，并相应构建Dense-ATOMIC，这相对于强基线方法具有优势。 |
| [^82] | [Rethinking Annotation: Can Language Learners Contribute?.](http://arxiv.org/abs/2210.06828) | 研究探究了语言学习者是否可以为标注基准数据集做出贡献，发现在提供额外资源的帮助下，具有中高级语言能力的学习者能够提供准确的标签。 |
| [^83] | [Augmentation Invariant Discrete Representation for Generative Spoken Language Modeling.](http://arxiv.org/abs/2209.15483) | 本研究提出了一种增强不变的离散语音表示方法，以提高其在生成式语音语言建模中的鲁棒性。该方法利用了transformer-based模型，并通过一种非线性量化方法来学习增强不变表示。实验证明，该方法相对于现有最先进方法具有显著的鲁棒性改进，并在语音生成任务上表现出了竞争性的表现。 |
| [^84] | [PaLI: A Jointly-Scaled Multilingual Language-Image Model.](http://arxiv.org/abs/2209.06794) | PaLI是一种联合缩放的多语言语言-图像模型，可对图像和文本进行建模和执行许多视觉、语言和多模态任务，利用Transformer和Vision Transformer等先前的能力和成本。联合缩放在此任务中很重要，所以我们使用了一个40亿参数的Vision Transformer，以便利用更大容量的视觉模型的优势。 |
| [^85] | [Massively Multilingual Lexical Specialization of Multilingual Transformers.](http://arxiv.org/abs/2208.01018) | 使用BabelNet的多语言同义词集，让大规模多语言变压器进行词汇专门化处理，能够大大提高跨语言词嵌入、多语言语义检索和多语言文档分类的表现。 |
| [^86] | [Joint Generator-Ranker Learning for Natural Language Generation.](http://arxiv.org/abs/2206.13974) | 提出了一种新的自然语言生成算法JGR，该算法将生成器和评分器集成在一个单一的框架中进行联合训练，通过混合目标优化生成器和使用对比损失训练评分器。在各种文本生成任务中，JGR在三种常见生成场景下的四个公共数据集上均优于现有方法。 |
| [^87] | [MVP: Multi-task Supervised Pre-training for Natural Language Generation.](http://arxiv.org/abs/2206.12131) | 本文提出了用于自然语言生成的多任务监督预训练（MVP）方法，其收集大规模自然语言生成语料库，并使用监督方式对文本生成模型进行训练。MVP模型结合特定的软提示，可以在各种任务中展现出优异的表现。 |
| [^88] | [BITE: Textual Backdoor Attacks with Iterative Trigger Injection.](http://arxiv.org/abs/2205.12700) | 提出了一种名为BITE的文本后门攻击方法，通过向训练数据中注入触发词，在迭代的单词级扰动中将这些词注入到输入实例中，成功地攻击了受害者模型。 |
| [^89] | [Optimizing Test-Time Query Representations for Dense Retrieval.](http://arxiv.org/abs/2205.12680) | 本文介绍了TOUR算法，它利用交叉编码再排序器提供的伪标签优化基于实例级别的查询表示，显著提高了端到端开放领域问答的准确性。 |
| [^90] | [QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs.](http://arxiv.org/abs/2205.12665) | 本论文提出了一个针对多段落多答案问题的开放域问答基准测试QAMPARI，并训练了ODQA模型。研究结果表明QAMPARI在段落检索和答案生成方面具有挑战性，强调了需要发展能够处理此类问题的ODQA模型。 |

# 详细

[^1]: 用生成模型进行多模态实体链接的基准测试

    Benchmarking Diverse-Modal Entity Linking with Generative Models. (arXiv:2305.17337v1 [cs.CL])

    [http://arxiv.org/abs/2305.17337](http://arxiv.org/abs/2305.17337)

    新方法提出了一个多模态实体链接的基准测试，并使用预训练的生成多模态模型，在平均F1分数上优于最先进的特定任务EL模型8.51分。

    

    实体可以用不同的格式来表达，如文本、图像或表格中的列名和单元格值。虽然现有的实体链接（EL）模型在每种模式配置上都表现出色，例如仅文本EL、视觉定位或模式链接，但为多种模式配置设计统一模型更具挑战性。为了将各种模态配置结合起来，我们从现有EL数据集构建了一个包含文本、图像和表格三种模态的多模态EL（DMEL）基准测试。为了解决DMEL任务，我们提出了一个生成多模态模型（GDMM），遵循多模态编码器-解码器范例。将\Model用丰富的语料库预训练可以在不保存整个KB进行推理的情况下为DMEL构建坚实的基础。微调GDMM可以构建更强大的DMEL基线，在平均F1分数上优于最先进的特定任务EL模型8.51分。此外，进行了广泛的错误分析，突出了将多模态实体翻译和所提出方法的优势所在。

    Entities can be expressed in diverse formats, such as texts, images, or column names and cell values in tables. While existing entity linking (EL) models work well on per modality configuration, such as text-only EL, visual grounding, or schema linking, it is more challenging to design a unified model for diverse modality configurations. To bring various modality configurations together, we constructed a benchmark for diverse-modal EL (DMEL) from existing EL datasets, covering all three modalities including text, image, and table. To approach the DMEL task, we proposed a generative diverse-modal model (GDMM) following a multimodal-encoder-decoder paradigm. Pre-training \Model with rich corpora builds a solid foundation for DMEL without storing the entire KB for inference. Fine-tuning GDMM builds a stronger DMEL baseline, outperforming state-of-the-art task-specific EL models by 8.51 F1 score on average. Additionally, extensive error analyses are conducted to highlight the challenges of
    
[^2]: 只使用前向传递微调语言模型

    Fine-Tuning Language Models with Just Forward Passes. (arXiv:2305.17333v1 [cs.LG])

    [http://arxiv.org/abs/2305.17333](http://arxiv.org/abs/2305.17333)

    本论文提出了一种内存高效的零阶优化器，可以使用与推理相同的存储空间微调语言模型，其可以在大规模模型下更快地优化，具有更好的实验结果。

    

    微调语言模型已经在各种下游任务中取得了成功，但随着语言模型的增大，反向传播需要的存储空间数量变得过高。零阶（ZO）方法理论上仅使用两次前向传递就可以估计梯度，但通常情况下对大型模型进行优化的速度非常慢。在本文中，我们提出了一种内存高效的零阶优化器（MeZO），将经典的ZO-SGD方法适应于原地操作，从而使用与推理相同的存储空间微调语言模型。例如，只使用一张A100 80GB GPU，MeZO就可以训练一个300亿参数的模型，而使用反向传播可以在相同的预算下仅训练一个27亿个参数的语言模型。我们在各种模型类型（掩码和自回归语言模型）、模型规模（高达66B）和下游任务（分类、多项选择和生成）进行了全面的实验。我们的结果表明，（1）MeZO明显优于上下文学习和线性PR模型。

    Fine-tuning language models (LMs) has yielded success on diverse downstream tasks, but as LMs grow in size, backpropagation requires a prohibitively large amount of memory. Zeroth-order (ZO) methods can in principle estimate gradients using only two forward passes but are theorized to be catastrophically slow for optimizing large models. In this work, we propose a memory-efficient zerothorder optimizer (MeZO), adapting the classical ZO-SGD method to operate in-place, thereby fine-tuning LMs with the same memory footprint as inference. For example, with a single A100 80GB GPU, MeZO can train a 30-billion parameter model, whereas fine-tuning with backpropagation can train only a 2.7B LM with the same budget. We conduct comprehensive experiments across model types (masked and autoregressive LMs), model scales (up to 66B), and downstream tasks (classification, multiple-choice, and generation). Our results demonstrate that (1) MeZO significantly outperforms in-context learning and linear pr
    
[^3]: 基于增强的适应性检索器以通用插件的形式提高了语言模型的泛化能力

    Augmentation-Adapted Retriever Improves Generalization of Language Models as Generic Plug-In. (arXiv:2305.17331v1 [cs.CL])

    [http://arxiv.org/abs/2305.17331](http://arxiv.org/abs/2305.17331)

    本文提出了增强适应检索器(AAR)的方案，通过从已知的源LM中学习LM的偏好，能够以通用插件的形式帮助目标LM在不进行微调的情况下显著提高零样本泛化能力。

    

    检索增强可以通过提供外部信息帮助语言模型(LMs)执行知识密集的任务。检索增强的先前工作通常联合微调检索器和LM，使它们紧密耦合。在本文中，我们探索了通用检索插件的方案：检索器是要帮助目标LM的，这些LM可能事先不知道或无法一起微调。为了检索出对未见过的目标LM有用的文档，我们提出了增强适应检索器(AAR)，它从已知的源LM中学习LM的偏好。在MMLU和PopQA数据集上的实验证明，我们用小型源LM训练的AAR能够显着提高从250M Flan-T5到175B InstructGPT范围内的更大目标LM的零样本泛化。进一步的分析表明，不同LM的偏好重叠，使得以单个源LM训练的AAR能够作为各种目标LM的通用插件。我们的代码...

    Retrieval augmentation can aid language models (LMs) in knowledge-intensive tasks by supplying them with external information. Prior works on retrieval augmentation usually jointly fine-tune the retriever and the LM, making them closely coupled. In this paper, we explore the scheme of generic retrieval plug-in: the retriever is to assist target LMs that may not be known beforehand or are unable to be fine-tuned together. To retrieve useful documents for unseen target LMs, we propose augmentation-adapted retriever (AAR), which learns LM's preferences obtained from a known source LM. Experiments on the MMLU and PopQA datasets demonstrate that our AAR trained with a small source LM is able to significantly improve the zero-shot generalization of larger target LMs ranging from 250M Flan-T5 to 175B InstructGPT. Further analysis indicates that the preferences of different LMs overlap, enabling AAR trained with a single source LM to serve as a generic plug-in for various target LMs. Our code 
    
[^4]: 为什么零样本跨语言生成失败？原因及解决方案

    Why Does Zero-Shot Cross-Lingual Generation Fail? An Explanation and a Solution. (arXiv:2305.17325v1 [cs.CL])

    [http://arxiv.org/abs/2305.17325](http://arxiv.org/abs/2305.17325)

    零样本跨语言生成失败的原因是神经网络模型在学习分类任务中的语言不变表示时，会影响在生成任务中的准确性，因此我们提出一种简单而有效的方法通过规范化模型来解决这个问题并提高生成质量。

    

    零样本跨语言转移是指在一种语言中训练多语言模型来执行任务，然后将其应用于另一种语言。虽然零样本跨语言转移方法在各种分类任务中取得了成功，但其在自然语言生成任务中的性能则不足，并且有时会输出错误的语言。在我们的研究中，我们展示了微调过程学习语言不变表示的好处是分类任务但对于生成任务有害。基于此，我们提出了一种简单的方法来规范化模型，使其不会学习语言不变的表示，并提出了一种在目标语言中选择不需要开发集的模型检查点的方法，两者都可以提高生成质量。对三个语义多样的生成任务的实验表明，我们的方法将偶然翻译问题减少了68％，平均提高了1.5的ROUGE-L得分。

    Zero-shot cross-lingual transfer is when a multilingual model is trained to perform a task in one language and then is applied to another language. Although the zero-shot cross-lingual transfer approach has achieved success in various classification tasks, its performance on natural language generation tasks falls short in quality and sometimes outputs an incorrect language. In our study, we show that the fine-tuning process learns language invariant representations, which is beneficial for classification tasks but harmful for generation tasks. Motivated by this, we propose a simple method to regularize the model from learning language invariant representations and a method to select model checkpoints without a development set in the target language, both resulting in better generation quality. Experiments on three semantically diverse generation tasks show that our method reduces the accidental translation problem by 68% and improves the ROUGE-L score by 1.5 on average.
    
[^5]: 超越正向缩放：否定语对语言模型缩放趋势的影响。

    Beyond Positive Scaling: How Negation Impacts Scaling Trends of Language Models. (arXiv:2305.17311v1 [cs.CL])

    [http://arxiv.org/abs/2305.17311](http://arxiv.org/abs/2305.17311)

    本研究介绍了一个包含否定问题的数据集NeQA，其中语言模型表现出反向缩放、U型缩放或正向缩放，解决NeQA依赖于问答和否定理解两个子任务，其缩放趋势由这两个子任务的缩放趋势组合形成。

    

    已经证明，语言模型表现出正向缩放，在大小、计算或数据方面扩展模型会提高性能。在本研究中，我们引入了一个包含否定问句的数据集NeQA，其中语言模型不会表现出简单的正向缩放。我们展示了这个任务可以表现出反向缩放、U形缩放或正向缩放，并且在使用更强大的提示方法或模型族群时，这三种缩放趋势会按照这个顺序发生转变。我们假设解决NeQA依赖于两个子任务：问答（任务1）和否定理解（任务2）。我们发现任务1具有线性缩放，而任务2具有S形缩放，并具有一个紧急的转折点，将这两个缩放趋势组合起来即可得出最终的NeQA缩放趋势。我们的研究揭示并提供了一种分析语言模型复杂缩放趋势的方法。

    Language models have been shown to exhibit positive scaling, where performance improves as models are scaled up in terms of size, compute, or data. In this work, we introduce NeQA, a dataset consisting of questions with negation in which language models do not exhibit straightforward positive scaling. We show that this task can exhibit inverse scaling, U-shaped scaling, or positive scaling, and the three scaling trends shift in this order as we use more powerful prompting methods or model families. We hypothesize that solving NeQA depends on two subtasks: question answering (task 1) and negation understanding (task 2). We find that task 1 has linear scaling, while task 2 has sigmoid-shaped scaling with an emergent transition point, and composing these two scaling trends yields the final scaling trend of NeQA. Our work reveals and provides a way to analyze the complex scaling trends of language models.
    
[^6]: “Chain-of-Thought Hub: 连续测量大型语言模型推理表现的努力”

    Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance. (arXiv:2305.17306v1 [cs.CL])

    [http://arxiv.org/abs/2305.17306](http://arxiv.org/abs/2305.17306)

    本文介绍了一个名为 Chain-of-Thought Hub 的开源评估套件，目的是评估大型语言模型的多步推理能力。它是为了追踪LLMs进展而编制的具有挑战性的推理基准。目前的研究结果表明，模型规模与推理能力相关，而 Claude-v1.3 是迄今为止推理能力最强的LLM。

    

    “随着大型语言模型（LLMs）的不断发展，它们的评估变得越来越重要但也更具挑战性。本文提出了 Chain-of-Thought Hub，这是一个开源的评估套件，旨在评估大型语言模型的多步推理能力。我们之所以对这个设置感兴趣，是因为 (1) 从 GPT 和 PaLM 模型家族的行为中，我们观察到复杂的推理很可能是一个更弱和更强的LLMs之间的关键区别； (2) 我们预见大型语言模型将成为下一代计算平台，并促进基于LLM的新应用的生态系统，这自然需要基础模型执行常常涉及语言和逻辑操作组合的复杂任务。我们的方法是编制一系列具有挑战性的推理基准，以跟踪LLMs的进展。我们目前的结果表明：(1) 模型规模显然与推理能力相关；(2) 截至2023年5月，Claude-v1.3 是迄今为止推理能力最强的LLM 。”

    As large language models (LLMs) are continuously being developed, their evaluation becomes increasingly important yet challenging. This work proposes Chain-of-Thought Hub, an open-source evaluation suite on the multi-step reasoning capabilities of large language models. We are interested in this setting for two reasons: (1) from the behavior of GPT and PaLM model family, we observe that complex reasoning is likely to be a key differentiator between weaker and stronger LLMs; (2) we envisage large language models to become the next-generation computational platform and foster an ecosystem of LLM-based new applications, this naturally requires the foundation models to perform complex tasks that often involve the composition of linguistic and logical operations. Our approach is to compile a suite of challenging reasoning benchmarks to track the progress of LLMs. Our current results show that: (1) model scale clearly correlates with reasoning capabilities; (2) As of May 2023, Claude-v1.3 an
    
[^7]: 外部语言模型在分解神经传输器中的集成

    External Language Model Integration for Factorized Neural Transducers. (arXiv:2305.17304v1 [cs.CL])

    [http://arxiv.org/abs/2305.17304](http://arxiv.org/abs/2305.17304)

    该论文提出了一种外部语言模型在分解神经传输器中的集成方法，同时证明与浅层融合相比线性插值预测输出与神经和n-gram外部LM的结合可以提高准确性。结果显示平均18%的WERR增益，在一个实体丰富的情况下可以获得高达60%的WERR添加性增益。

    

    我们提出了一种适用于带有外部语言模型的分解神经传输器（FNT）的适应方法。我们证明了与浅层融合相比，线性插值预测输出与神经和n-gram外部LM的结合明显增加了价值，从而确认了FNT强制预测器像常规语言模型一样工作。此外，我们提出了一种将基于类别的n-gram语言模型集成到FNT框架中的方法，可以实现类似于混合设置的准确性提高。通过词汇适应，在各种情况下，我们显示平均18%的WERR增益，并且在一个实体丰富的情况下，通过类别n-gram和神经LM的结合获得高达60%的WERR添加性增益。

    We propose an adaptation method for factorized neural transducers (FNT) with external language models. We demonstrate that both neural and n-gram external LMs add significantly more value when linearly interpolated with predictor output compared to shallow fusion, thus confirming that FNT forces the predictor to act like regular language models. Further, we propose a method to integrate class-based n-gram language models into FNT framework resulting in accuracy gains similar to a hybrid setup. We show average gains of 18% WERR with lexical adaptation across various scenarios and additive gains of up to 60% WERR in one entity-rich scenario through a combination of class-based n-gram and neural LMs.
    
[^8]: 基于菜谱的对话中的改进指令排序

    Improved Instruction Ordering in Recipe-Grounded Conversation. (arXiv:2305.17280v1 [cs.CL])

    [http://arxiv.org/abs/2305.17280](http://arxiv.org/abs/2305.17280)

    本文针对基于菜谱的对话任务，提出了两个辅助子任务，即用户意图检测和指令状态跟踪，并证明这些任务可以帮助响应生成模型解决指令顺序错误问题。

    

    本文研究指令对话任务，并聚焦于烹饪领域。分析GPT-J模型生成的输出，我们揭示了菜谱对话系统的主要挑战在于如何以正确的顺序提供指令。我们假设这是由于模型缺乏理解用户意图和无法跟踪指令状态（即最后一步是哪个指令）的能力导致的。因此，我们提出了探索两个辅助子任务——用户意图检测和指令状态跟踪，以支持改进指令基础的响应生成。在我们新收集的数据集ChattyChef上进行实验表明，融入用户意图和指令状态信息有助于响应生成模型减轻顺序错误问题。此外，为了研究ChatGPT是否完全解决了这个任务，我们分析其输出并发现它也犯了错误（约占响应的10.7%），

    In this paper, we study the task of instructional dialogue and focus on the cooking domain. Analyzing the generated output of the GPT-J model, we reveal that the primary challenge for a recipe-grounded dialog system is how to provide the instructions in the correct order. We hypothesize that this is due to the model's lack of understanding of user intent and inability to track the instruction state (i.e., which step was last instructed). Therefore, we propose to explore two auxiliary subtasks, namely User Intent Detection and Instruction State Tracking, to support Response Generation with improved instruction grounding. Experimenting with our newly collected dataset, ChattyChef, shows that incorporating user intent and instruction state information helps the response generation model mitigate the incorrect order issue. Furthermore, to investigate whether ChatGPT has completely solved this task, we analyze its outputs and find that it also makes mistakes (10.7% of the responses), about 
    
[^9]: 滑动、约束、解析、重复：适用于文档 AMR 解析的同步滑动窗口方法

    Slide, Constrain, Parse, Repeat: Synchronous SlidingWindows for Document AMR Parsing. (arXiv:2305.17273v1 [cs.CL])

    [http://arxiv.org/abs/2305.17273](http://arxiv.org/abs/2305.17273)

    本文提出了一种同步滑动窗口的方法来处理文档解析的序列到序列任务，利用源-目标对齐并约束解码以保证重叠窗口的同步性和一致性，在AMR 3.0的评估中展示出了高质量的性能。

    

    滑动窗口方法提供了一种处理超过Transformer输入窗口大小的上下文的优美方式，例如处理语言建模任务。本文将这种方法扩展到文档解析的序列到序列任务中。为此，我们利用了转移句法分析的最新进展，通过在源和目标之间实现同步滑动窗口来实现解析器。我们通过在机构化BART上扩展来开发文档级AMR的oracle和解析器，以利用源-目标对齐并约束解码以保证重叠窗口的同步性和一致性。我们使用抽象意义表示（AMR）3.0语料库评估了我们的oracle和解析器。在AMR 3.0的多句子开发集上，我们展示了我们的转移oracle仅丢失了8％的金句际链接，尽管使用滑动窗口。在实践中，这种方法也产生了一个具有可管理内存要求的高质量文档级解析器。

    The sliding window approach provides an elegant way to handle contexts of sizes larger than the Transformer's input window, for tasks like language modeling. Here we extend this approach to the sequence-to-sequence task of document parsing. For this, we exploit recent progress in transition-based parsing to implement a parser with synchronous sliding windows over source and target. We develop an oracle and a parser for document-level AMR by expanding on Structured-BART such that it leverages source-target alignments and constrains decoding to guarantee synchronicity and consistency across overlapping windows. We evaluate our oracle and parser using the Abstract Meaning Representation (AMR) parsing 3.0 corpus. On the Multi-Sentence development set of AMR 3.0, we show that our transition oracle loses only 8\% of the gold cross-sentential links despite using a sliding window. In practice, this approach also results in a high-quality document-level parser with manageable memory requirement
    
[^10]: 通过显式基本含义建模检测隐喻

    Metaphor Detection via Explicit Basic Meanings Modelling. (arXiv:2305.17268v1 [cs.CL])

    [http://arxiv.org/abs/2305.17268](http://arxiv.org/abs/2305.17268)

    本文提出了一种新的隐喻检测方法，通过对训练集中的字面注释进行基本含义建模并将其与上下文含义进行比较，可以更准确地识别隐喻，表现优于现有方法1.0％。

    

    隐喻检测中的一个显著趋势是采用语言学理论，如隐喻识别程序（MIP），用于模型架构设计。虽然MIP明确定义了词汇单位的隐喻性是基于其“上下文含义”和“基本含义”之间的对比来确定的，但现有的工作并没有严格遵循这个原则，通常使用“聚合含义”来近似于目标词的基本含义。本文提出了一种新的隐喻检测方法，它基于训练集中的字面注释对单词的基本含义进行建模，然后将其与目标句子中的上下文含义进行比较以识别隐喻。实验结果表明，我们的方法在F1得分方面显著优于现有方法1.0％。此外，我们的性能甚至达到了带有基本注释的VUA18基准测试的理论上限，这证明了我们方法的有效性。

    One noticeable trend in metaphor detection is the embrace of linguistic theories such as the metaphor identification procedure (MIP) for model architecture design. While MIP clearly defines that the metaphoricity of a lexical unit is determined based on the contrast between its \textit{contextual meaning} and its \textit{basic meaning}, existing work does not strictly follow this principle, typically using the \textit{aggregated meaning} to approximate the basic meaning of target words. In this paper, we propose a novel metaphor detection method, which models the basic meaning of the word based on literal annotation from the training set, and then compares this with the contextual meaning in a target sentence to identify metaphors. Empirical results show that our method outperforms the state-of-the-art method significantly by 1.0\% in F1 score. Moreover, our performance even reaches the theoretical upper bound on the VUA18 benchmark for targets with basic annotations, which demonstrate
    
[^11]: CODET：机器翻译对比方言评估的基准测试

    CODET: A Benchmark for Contrastive Dialectal Evaluation of Machine Translation. (arXiv:2305.17267v1 [cs.CL])

    [http://arxiv.org/abs/2305.17267](http://arxiv.org/abs/2305.17267)

    CODET是一个对比方言的评估基准测试，用于评估机器翻译系统在处理方言变体时的表现，该基准测试包含九种不同语言的882个不同变体。

    

    神经机器翻译系统在处理源语言的语言变化方面表现出有限的鲁棒性。当面临即使是语言使用中的细微差异（例如不同的领域或由第二语言使用者引入的变体）时，其性能往往会下降。直观上，将这种观察推广到涵盖方言变体，而允许社区在这个维度上评估MT系统的工作是有限的。为了解决这个问题，我们编译和发布了对比方言基准测试 \dataset，其中包括来自九种不同语言的882个不同变体。我们还在数量上展示了大型MT模型在有效翻译方言变体方面面临的挑战。我们发布所有代码和数据。

    Neural machine translation (NMT) systems exhibit limited robustness in handling source-side linguistic variations. Their performance tends to degrade when faced with even slight deviations in language usage, such as different domains or variations introduced by second-language speakers. It is intuitive to extend this observation to encompass dialectal variations as well, but the work allowing the community to evaluate MT systems on this dimension is limited. To alleviate this issue, we compile and release \dataset, a contrastive dialectal benchmark encompassing 882 different variations from nine different languages. We also quantitatively demonstrate the challenges large MT models face in effectively translating dialectal variants. We are releasing all code and data.
    
[^12]: 纵览语言模型：缩减规模后的行为

    Honey, I Shrunk the Language: Language Model Behavior at Reduced Scale. (arXiv:2305.17266v1 [cs.CL])

    [http://arxiv.org/abs/2305.17266](http://arxiv.org/abs/2305.17266)

    本文研究了小规模语言模型的训练效果，并展示了掩码语言建模目标的预训练对性能的提高作用。同时，该研究还发现了计算成本与模型效果之间的相关性。

    

    近年来，语言模型的规模急剧增长，这些模型的能力也随着规模的扩大而得到了提高。大部分最近的规模研究都集中在高计算量，高参数的环境中，没有回答这些能力何时开始出现的问题。在本文中，我们研究了在问题规模减小的情况下是否可以观察到预训练的效果，建立了一个较小的、缩减了词汇量的语言模型。我们展示了在参数为125万的模型中使用掩码语言建模（MLM）目标预训练的好处，并建立了预训练困惑和下游性能（GLUE基准）之间的强相关性。我们研究缩小规模的影响，将缩放定律扩展到了大约100万个参数的模型中。在这个规模下，我们观察到了计算-最优模型的幂律破裂，并展示了MLM损失在低于22万亿FLOPs的计算成本下并不平滑地缩放。

    In recent years, language models have drastically grown in size, and the abilities of these models have been shown to improve with scale. The majority of recent scaling laws studies focused on high-compute high-parameter count settings, leaving the question of when these abilities begin to emerge largely unanswered. In this paper, we investigate whether the effects of pre-training can be observed when the problem size is reduced, modeling a smaller, reduced-vocabulary language. We show the benefits of pre-training with masked language modeling (MLM) objective in models as small as 1.25M parameters, and establish a strong correlation between pre-training perplexity and downstream performance (GLUE benchmark). We examine downscaling effects, extending scaling laws to models as small as ~1M parameters. At this scale, we observe a break of the power law for compute-optimal models and show that the MLM loss does not scale smoothly with compute-cost (FLOPs) below $2.2 \times 10^{15}$ FLOPs. 
    
[^13]: 大型语言模型可能是懒惰的学习者：分析上下文学习中的捷径

    Large Language Models Can be Lazy Learners: Analyze Shortcuts in In-Context Learning. (arXiv:2305.17256v1 [cs.CL])

    [http://arxiv.org/abs/2305.17256](http://arxiv.org/abs/2305.17256)

    本文探讨了大型语言模型在上下文学习中利用提示中的捷径的依赖性，发现大型模型更有可能在推理过程中利用提示中的捷径，这为评估上下文学习的稳健性和检测和缓解提示中捷径的使用提供了新的视角和挑战。

    

    最近，大型语言模型（LLM）在上下文学习中展现出巨大潜力，其中LLM通过几个输入-标签对（提示）的条件来学习新任务。尽管其潜力巨大，但我们对影响最终任务性能和上下文学习稳健性的因素的理解仍然有限。本文旨在通过研究LLM对提示内捷径或假相关的依赖关系来弥补这一知识差距。通过分类和抽取任务的全面实验，我们揭示了LLM是“懒惰学习者”的事实，它往往利用提示中的捷径来获取下游任务的性能提升。此外，我们还发现一个令人惊讶的发现，即较大的模型更有可能在推理过程中利用提示中的捷径。我们的发现为评估上下文学习的稳健性和检测和缓解提示中捷径的使用提供了新的视角和挑战。

    Large language models (LLMs) have recently shown great potential for in-context learning, where LLMs learn a new task simply by conditioning on a few input-label pairs (prompts). Despite their potential, our understanding of the factors influencing end-task performance and the robustness of in-context learning remains limited. This paper aims to bridge this knowledge gap by investigating the reliance of LLMs on shortcuts or spurious correlations within prompts. Through comprehensive experiments on classification and extraction tasks, we reveal that LLMs are "lazy learners" that tend to exploit shortcuts in prompts for downstream tasks. Additionally, we uncover a surprising finding that larger models are more likely to utilize shortcuts in prompts during inference. Our findings provide a new perspective on evaluating robustness in in-context learning and pose new challenges for detecting and mitigating the use of shortcuts in prompts.
    
[^14]: 基于联邦学习的语义解析任务：任务形式，评估设置及新算法

    Federated Learning for Semantic Parsing: Task Formulation, Evaluation Setup, New Algorithms. (arXiv:2305.17221v1 [cs.CL])

    [http://arxiv.org/abs/2305.17221](http://arxiv.org/abs/2305.17221)

    本文研究了基于联邦学习的语义解析任务，提出了评估设置和新算法。实验表明，新算法FedSQL和Lorar优于现有的FL算法和我们提出的设置的强基线。

    

    本文研究了一种新的联邦学习任务，即针对语义解析的联邦学习，多个客户端共同训练一个全局模型，而无需共享其语义分析数据。通过利用多个客户端的数据，联邦学习模式对于那些没有足够训练数据来开发一个数据饥饿的神经语义分析器的客户端尤其有益。我们提出了一种评估设置来研究这个任务，将广泛使用的单域文本到SQL数据集作为客户端来形成一个现实的异构联邦学习设置，并协同训练一个全局模型。由于我们的现实设置中客户群的异质性很高，标准的联邦学习算法会受到影响，所以我们进一步提出了一种新的机制LOss Reduction Adjusted Re-weighting (Lorar)来缓解性能下降，该机制基于客户端每轮训练损失的减少情况来调节每个客户端对于全局模型更新的贡献。我们的直觉是，损失减少的越多，客户端离全局最优解就越远，其对模型更新的贡献就应该越高。同时，我们还提出了一个针对异构文本到SQL FL设置的新的FL算法FedSQL。我们的实验表明，FedSQL和Lorar显著优于现有的FL算法和我们提出的FL设置中的强基线。

    This paper studies a new task of federated learning (FL) for semantic parsing, where multiple clients collaboratively train one global model without sharing their semantic parsing data. By leveraging data from multiple clients, the FL paradigm can be especially beneficial for clients that have little training data to develop a data-hungry neural semantic parser on their own. We propose an evaluation setup to study this task, where we re-purpose widely-used single-domain text-to-SQL datasets as clients to form a realistic heterogeneous FL setting and collaboratively train a global model. As standard FL algorithms suffer from the high client heterogeneity in our realistic setup, we further propose a novel LOss Reduction Adjusted Re-weighting (Lorar) mechanism to mitigate the performance degradation, which adjusts each client's contribution to the global model update based on its training loss reduction during each round. Our intuition is that the larger the loss reduction, the further aw
    
[^15]: GVdoc: 基于图的视觉文档分类

    GVdoc: Graph-based Visual Document Classification. (arXiv:2305.17219v1 [cs.CV])

    [http://arxiv.org/abs/2305.17219](http://arxiv.org/abs/2305.17219)

    GVdoc 是一个基于图的文档分类模型，能够通过生成文档图并训练图神经网络来学习节点和图嵌入，有效解决视觉文档分类器在领域内外样本分类和区分中所遇到的挑战。

    

    模型在实际部署中的鲁棒性取决于其在未见过的数据上的表现和对领域内外样本的区分能力。视觉文档分类器在分布测试集上表现出色。然而，它们在正确分类和区分领域外例子方面往往遇到困难。基于图的文档分类模型通过生成基于布局的文档图，然后训练图神经网络来学习节点和图嵌入，解决了这两个挑战。我们的模型即使参数更少，在领域外的样本上也优于现有模型，在领域内基准上实现了最新的性能。

    The robustness of a model for real-world deployment is decided by how well it performs on unseen data and distinguishes between in-domain and out-of-domain samples. Visual document classifiers have shown impressive performance on in-distribution test sets. However, they tend to have a hard time correctly classifying and differentiating out-of-distribution examples. Image-based classifiers lack the text component, whereas multi-modality transformer-based models face the token serialization problem in visual documents due to their diverse layouts. They also require a lot of computing power during inference, making them impractical for many real-world applications. We propose, GVdoc, a graph-based document classification model that addresses both of these challenges. Our approach generates a document graph based on its layout, and then trains a graph neural network to learn node and graph embeddings. Through experiments, we show that our model, even with fewer parameters, outperforms stat
    
[^16]: 用多模态语言模型生成图片

    Generating Images with Multimodal Language Models. (arXiv:2305.17216v1 [cs.CL])

    [http://arxiv.org/abs/2305.17216](http://arxiv.org/abs/2305.17216)

    该论文提出了一种方法，将大型语言模型与预训练的图像编码器和解码器模型进行融合，能生成具有连贯性的图像输出，同时也能进行图像检索和多模态对话。

    

    我们提出了一种方法，将仅包含文本的大型语言模型（LLMs）与预训练的图像编码器和解码器模型进行融合，通过映射它们的嵌入空间。我们的模型展示了广泛的多模态能力：图像检索、新颖图像生成和多模态对话。这是第一种能够在任意交错的图像和文本输入之间进行条件调节，生成连贯图像（和文本）输出的方法。为了在图像生成任务中取得强大的性能，我们提出了一种有效的映射网络，将LLM基于现成的文本到图像生成模型，将文本的隐藏表示转换为视觉模型的嵌入空间，利用LLM强大的文本表示来生成视觉输出。我们的方法在长且复杂语言的任务上优于基准生成模型。除了新颖图像生成之外，我们的模型还能够从文本描述中检索图像，并进行多模态对话。

    We propose a method to fuse frozen text-only large language models (LLMs) with pre-trained image encoder and decoder models, by mapping between their embedding spaces. Our model demonstrates a wide suite of multimodal capabilities: image retrieval, novel image generation, and multimodal dialogue. Ours is the first approach capable of conditioning on arbitrarily interleaved image and text inputs to generate coherent image (and text) outputs. To achieve strong performance on image generation, we propose an efficient mapping network to ground the LLM to an off-the-shelf text-to-image generation model. This mapping network translates hidden representations of text into the embedding space of the visual models, enabling us to leverage the strong text representations of the LLM for visual outputs. Our approach outperforms baseline generation models on tasks with longer and more complex language. In addition to novel image generation, our model is also capable of image retrieval from a prespe
    
[^17]: BIG-C：一种Bemba语的多模态多用途数据集。

    BIG-C: a Multimodal Multi-Purpose Dataset for Bemba. (arXiv:2305.17202v1 [cs.CL])

    [http://arxiv.org/abs/2305.17202](http://arxiv.org/abs/2305.17202)

    BIG-C是一个用于Bemba语的大型多模态数据集，提供了语音识别、机器翻译、语音翻译任务的基线，并勾画了该数据集的潜在未来多模态用途，旨在促进研究并鼓励跨越语言、语音和视觉社区的合作，特别是针对“传统”使用的语言之外的语言。

    

    我们提出了BIG-C（Bemba图像引导对话），这是一个用于Bemba语的大型多模态数据集。虽然Bemba语是赞比亚人口最多的语言，但其缺乏资源使得语言技术或语言处理研究的发展几乎是不可能的。该数据集由基于图像的Bemba语言者之间的多轮对话组成，经过转录并翻译成英文。数据集中有超过92,000个话语/句子，相当于超过180小时的音频数据，具有相应的转录和英文翻译。我们还提供了语音识别（ASR）、机器翻译（MT）和语音翻译（ST）任务的基线，并勾画了该数据集其他潜在的未来多模态用途。我们希望通过将该数据集提供给研究社区，本工作将促进研究并鼓励跨越语言、语音和视觉社区的合作，特别是针对“传统”使用的语言之外的语言。

    We present BIG-C (Bemba Image Grounded Conversations), a large multimodal dataset for Bemba. While Bemba is the most populous language of Zambia, it exhibits a dearth of resources which render the development of language technologies or language processing research almost impossible. The dataset is comprised of multi-turn dialogues between Bemba speakers based on images, transcribed and translated into English. There are more than 92,000 utterances/sentences, amounting to more than 180 hours of audio data with corresponding transcriptions and English translations. We also provide baselines on speech recognition (ASR), machine translation (MT) and speech translation (ST) tasks, and sketch out other potential future multimodal uses of our dataset. We hope that by making the dataset available to the research community, this work will foster research and encourage collaboration across the language, speech, and vision communities especially for languages outside the "traditionally" used hig
    
[^18]: 认知作为强健的自学者

    Entailment as Robust Self-Learner. (arXiv:2305.17197v1 [cs.CL])

    [http://arxiv.org/abs/2305.17197](http://arxiv.org/abs/2305.17197)

    本文提出了一种将许多不同的NLU任务制定为情境认知的提示策略，并通过自我训练来提高模型的适应性性能。简单的伪标签编辑（SimPLE）算法有利于自我训练的稳定改进。

    

    认知已被认为是评估自然语言理解（NLU）模型重要的指标，近期研究发现，认知预训练有利于弱监督微调。本文设计了一种提示策略，将许多不同的NLU任务制定为情境认知。该方法提高了预训练认知模型的零-shot适应性。此外，我们发现使用无标签数据的自我训练认知模型可以显着提高下游任务的适应性性能。为了实现更稳定的改进，我们提出了简单的伪标签编辑（SimPLE）算法，以提高自我训练中的伪标签质量。我们还发现，预训练的认知模型和自我训练的模型都对对抗性评估数据具有强健性。

    Entailment has been recognized as an important metric for evaluating natural language understanding (NLU) models, and recent studies have found that entailment pretraining benefits weakly supervised fine-tuning. In this work, we design a prompting strategy that formulates a number of different NLU tasks as contextual entailment. This approach improves the zero-shot adaptation of pretrained entailment models. Secondly, we notice that self-training entailment-based models with unlabeled data can significantly improve the adaptation performance on downstream tasks. To achieve more stable improvement, we propose the Simple Pseudo-Label Editing (SimPLE) algorithm for better pseudo-labeling quality in self-training. We also found that both pretrained entailment-based models and the self-trained models are robust against adversarial evaluation data. Experiments on binary and multi-class classification tasks show that SimPLE leads to more robust self-training results, indicating that the self-
    
[^19]: 无监督神经机器翻译的复制问题：具有语言鉴别器损失的训练计划

    On the Copying Problem of Unsupervised NMT: A Training Schedule with a Language Discriminator Loss. (arXiv:2305.17182v1 [cs.CL])

    [http://arxiv.org/abs/2305.17182](http://arxiv.org/abs/2305.17182)

    无监督NMT中的复制问题通常发生在远距离语种对中且会直接复制输入句子的部分作为翻译，本研究提出了一种包含语言鉴别器损失的训练计划来缓解该问题，并提高低资源语种的翻译性能。

    

    虽然无监督神经机器翻译已在许多语种间得到成功，但复制问题（即将输入句子的某些部分直接复制作为翻译）在远距离语种对中很常见，尤其涉及低资源语种。我们发现这个问题与在线回译（BT）期间出现的预期复制行为密切相关。在这项工作中，我们提出了一个简单但有效的训练计划，它包含了一个语言鉴别器的损失函数。该损失施加约束于中间翻译，以使翻译是所需的语言。通过在不同语言对、 包括相似和远距离、高资源和低资源语言的广泛实验中，我们发现我们的方法缓解了复制问题，从而提高了对低资源语言的翻译性能。

    Although unsupervised neural machine translation (UNMT) has achieved success in many language pairs, the copying problem, i.e., directly copying some parts of the input sentence as the translation, is common among distant language pairs, especially when low-resource languages are involved. We find this issue is closely related to an unexpected copying behavior during online back-translation (BT). In this work, we propose a simple but effective training schedule that incorporates a language discriminator loss. The loss imposes constraints on the intermediate translation so that the translation is in the desired language. By conducting extensive experiments on different language pairs, including similar and distant, high and low-resource languages, we find that our method alleviates the copying problem, thus improving the translation performance on low-resource languages.
    
[^20]: 分词对多语言语言建模的影响：评估跨语言词汇分配和重叠的新标准

    Tokenization Impacts Multilingual Language Modeling: Assessing Vocabulary Allocation and Overlap Across Languages. (arXiv:2305.17179v1 [cs.CL])

    [http://arxiv.org/abs/2305.17179](http://arxiv.org/abs/2305.17179)

    本论文提出了新的标准来评估多语言语言模型中分词器的质量，并发现跨语言词汇的重叠会对某些下游任务产生不利影响，但共享词汇有助于其他任务。研究还发现，多语言词汇中的语言特定标记覆盖范围对单词级任务产生显著影响。这些发现对未来的模型开发人员选择最合适的分词器提供了指南。

    

    最近，多语言语言模型作为将多种语言表示为一个模型的有前途的解决方案而受到关注。本文提出了评估子词分词器中词汇表示和词汇重叠质量的新标准。研究发现，跨语言词汇的重叠实际上会对某些下游任务（POS，依赖树标注）产生不利影响。相反，共享词汇有助于NER和句子级任务（跨语言检索，NLI）。我们还观察到，多语言词汇中的语言特定标记覆盖范围显着影响单词级任务。本研究为深入了解分词器在多语言语言模型中的作用并为未来模型开发人员选择适合其特定应用的最合适的分词器提供指南。

    Multilingual language models have recently gained attention as a promising solution for representing multiple languages in a single model. In this paper, we propose new criteria to evaluate the quality of lexical representation and vocabulary overlap observed in sub-word tokenizers. Our findings show that the overlap of vocabulary across languages can be actually detrimental to certain downstream tasks (POS, dependency tree labeling). In contrast, NER and sentence-level tasks (cross-lingual retrieval, NLI) benefit from sharing vocabulary. We also observe that the coverage of the language-specific tokens in the multilingual vocabulary significantly impacts the word-level tasks. Our study offers a deeper understanding of the role of tokenizers in multilingual language models and guidelines for future model developers to choose the most suitable tokenizer for their specific application before undertaking costly model pre-training
    
[^21]: 从犬哨到喇叭：利用语言模型揭示编码修辞

    From Dogwhistles to Bullhorns: Unveiling Coded Rhetoric with Language Models. (arXiv:2305.17174v1 [cs.CL])

    [http://arxiv.org/abs/2305.17174](http://arxiv.org/abs/2305.17174)

    本文首次开展了对犬哨现象进行的大规模计算研究，发现犬哨词汇可向不同群体传达不同的含义和挑衅性，同时解释了它们如何规避政治后果和算法内容调节。

    

    犬哨是一种编码表达方式，它同时向广大观众传达一层含义，向一个狭窄的内部群体传达第二层意味，这通常是令人憎恶或挑衅的；它们旨在回避政治后果和算法内容调节。本文首次开展了对犬哨现象的大规模计算研究。我们开发了犬哨的分类法，策划了有着丰富背景信息和实例的迄今为止最大的犬哨词汇表，并分析了它们在美国历史政治家的演讲中的使用。然后，我们评估了一个大型语言模型（GPT-3）是否可以识别犬哨及其意义，并发现GPT-3性能因犬哨类型和目标群体而变化。最后，我们展示了包含犬哨的有害内容如何规避毒性检测。

    Dogwhistles are coded expressions that simultaneously convey one meaning to a broad audience and a second one, often hateful or provocative, to a narrow in-group; they are deployed to evade both political repercussions and algorithmic content moderation. For example, in the sentence 'we need to end the cosmopolitan experiment,' the word 'cosmopolitan' likely means 'worldly' to many, but secretly means 'Jewish' to a select few. We present the first large-scale computational investigation of dogwhistles. We develop a typology of dogwhistles, curate the largest-to-date glossary of over 300 dogwhistles with rich contextual information and examples, and analyze their usage in historical U.S. politicians' speeches. We then assess whether a large language model (GPT-3) can identify dogwhistles and their meanings, and find that GPT-3's performance varies widely across types of dogwhistles and targeted groups. Finally, we show that harmful content containing dogwhistles avoids toxicity detectio
    
[^22]: 大型语言模型的异质价值评估

    Heterogeneous Value Evaluation for Large Language Models. (arXiv:2305.17147v1 [cs.CL])

    [http://arxiv.org/abs/2305.17147](http://arxiv.org/abs/2305.17147)

    本文提出了一种自动对齐评估方法A2EHV，采用异质价值系统，并基于价值合理性和社会价值定向框架评估代理人行为的社会偏好，结果表明比传统对齐方法更合理。

    

    大型语言模型（LLM）的出现使得将它们的价值与人类价值对齐变得至关重要。当前的方法通常尝试将其与一种同质的人类价值对齐，并需要人类验证，但缺乏对对齐所需方面和深度的共识以及造成的人类偏见。在本文中，我们提出了一种自动对齐评估方法A2EHV，该方法采用异质价值系统，（1）是自动化的，以最小化单个人类偏见，并且（2）允许评估针对各种目标值的异质代理人。我们的方法基于价值合理性的概念，它代表了代理人执行最能满足目标价值行为的能力。价值合理性的量化是通过社会心理学中的社会价值定向框架进行的，该框架将价值空间分为四个类别，以评估代理人行为的社会偏好。我们评估了三个模型的价值合理性，结果表明A2EHV方法比传统对齐方法更合理。

    The emergent capabilities of Large Language Models (LLMs) have made it crucial to align their values with those of humans. Current methodologies typically attempt alignment with a homogeneous human value and requires human verification, yet lack consensus on the desired aspect and depth of alignment and resulting human biases. In this paper, we propose A2EHV, an Automated Alignment Evaluation with a Heterogeneous Value system that (1) is automated to minimize individual human biases, and (2) allows assessments against various target values to foster heterogeneous agents. Our approach pivots on the concept of value rationality, which represents the ability for agents to execute behaviors that satisfy a target value the most. The quantification of value rationality is facilitated by the Social Value Orientation framework from social psychology, which partitions the value space into four categories to assess social preferences from agents' behaviors. We evaluate the value rationality of e
    
[^23]: Minecraft中的幽灵：利用基于文本知识和记忆的大型语言模型实现开放世界环境中的通用能力智能体。

    Ghost in the Minecraft: Generally Capable Agents for Open-World Enviroments via Large Language Models with Text-based Knowledge and Memory. (arXiv:2305.17144v1 [cs.AI])

    [http://arxiv.org/abs/2305.17144](http://arxiv.org/abs/2305.17144)

    本文提出了Ghost in the Minecraft (GITM)框架，利用大型语言模型与基于文本的知识和记忆，创造了一种在Minecraft中具备通用能力的智能体，可在以文本为基础的复杂编程环境中熟练导航。

    

    近年来，Minecraft玩法吸引了大量的研究关注，成为开发能够在开放世界环境中运行的智能体的丰富平台。然而，当前的研究主要集中在特定的目标上，例如流行的“ObtainDiamond”任务，并且还没有显示出有效地推广到更广泛任务的能力。此外，“ObtainDiamond”任务的目前最高成功率只有约20％，凸显了现有方法中使用强化学习（RL）控制器的局限性。为了解决这些挑战，我们引入了Ghost in the Minecraft (GITM)，一个新颖的框架，将大型语言模型与基于文本的知识和记忆相结合，旨在创建Minecraft中的通用能力智能体。这些具备LLM中的逻辑和常识能力的智能体可以熟练地在以文本为基础的复杂编程环境中导航。

    The captivating realm of Minecraft has attracted substantial research interest in recent years, serving as a rich platform for developing intelligent agents capable of functioning in open-world environments. However, the current research landscape predominantly focuses on specific objectives, such as the popular "ObtainDiamond" task, and has not yet shown effective generalization to a broader spectrum of tasks. Furthermore, the current leading success rate for the "ObtainDiamond" task stands at around 20%, highlighting the limitations of Reinforcement Learning (RL) based controllers used in existing methods. To tackle these challenges, we introduce Ghost in the Minecraft (GITM), a novel framework integrates Large Language Models (LLMs) with text-based knowledge and memory, aiming to create Generally Capable Agents (GCAs) in Minecraft. These agents, equipped with the logic and common sense capabilities of LLMs, can skillfully navigate complex, sparse-reward environments with text-based 
    
[^24]: NavGPT: 带有大型语言模型的视觉语言导航中的显式推理

    NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models. (arXiv:2305.16986v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.16986](http://arxiv.org/abs/2305.16986)

    NavGPT是基于LLM的导航智能体，可以在视觉语言导航（VLN）中，通过对文本描述进行推理，执行零-shot连续动作预测。该模型具有高级规划能力，可以将指令分解成子目标、整合常识知识以进行障碍物避免，并参考先前的步骤进行澄清。NavGPT展示了通用体现智能体发展的美好前景。

    

    大型语言模型（LLM）例如ChatGPT和GPT-4以前所未有的规模进行训练，从模型的扩展中展现出显著的推理能力。这种趋势强调了使用无限语言数据训练LLM的潜力，推动了通用体现智能体的发展。本文介绍了NavGPT，这是一个纯粹基于LLM的指令跟随导航智能体，通过为视觉语言导航（VLN）执行零-shot的连续动作预测，揭示了对于在复杂的现实场景下GPT模型的推理能力。在每一步中，NavGPT将视觉观察、导航历史和未来可探索方向的文本描述作为输入，推理出智能体的当前状态，并决定如何接近目标。通过全面的实验，我们证明了NavGPT可以明确地执行导航的高级规划，包括将指令分解成子目标、整合常识知识以进行障碍物避免，并参考先前的步骤进行澄清。我们的结果表明，LLM可能成为复杂顺序决策任务中的传统流程的强有力替代品，展示了通用体现智能体发展的美好前景。

    Trained with an unprecedented scale of data, large language models (LLMs) like ChatGPT and GPT-4 exhibit the emergence of significant reasoning abilities from model scaling. Such a trend underscored the potential of training LLMs with unlimited language data, advancing the development of a universal embodied agent. In this work, we introduce the NavGPT, a purely LLM-based instruction-following navigation agent, to reveal the reasoning capability of GPT models in complex embodied scenes by performing zero-shot sequential action prediction for vision-and-language navigation (VLN). At each step, NavGPT takes the textual descriptions of visual observations, navigation history, and future explorable directions as inputs to reason the agent's current status, and makes the decision to approach the target. Through comprehensive experiments, we demonstrate NavGPT can explicitly perform high-level planning for navigation, including decomposing instruction into sub-goal, integrating commonsense k
    
[^25]: 面向抽象摘要中的领域泛化的领域对齐前缀平均方法

    Domain Aligned Prefix Averaging for Domain Generalization in Abstractive Summarization. (arXiv:2305.16820v1 [cs.CL])

    [http://arxiv.org/abs/2305.16820](http://arxiv.org/abs/2305.16820)

    本文提出了一种轻量级、基于加权平均的领域对齐前缀平均方法（DAPA），用于抽象摘要中的领域泛化，实现了有效的源域扩展以提高性能。

    

    针对于抽象摘要中的领域泛化问题，本文提出了一种轻量级，基于加权平均的领域对齐前缀平均方法（DAPA）。通过给定多个源域，我们的方法首先为每个域训练一个前缀，然后利用这些前缀生成少量目标域文档的摘要，计算所需的权重来平均源前缀。在DAPA中，前缀调整允许轻量级的微调，加权平均允许有效地添加新的源域。在四个不同的摘要领域上进行评估，DAPA表现出与基准方法相当或更好的性能，证明了其前缀平均的有效性。

    Domain generalization is hitherto an underexplored area applied in abstractive summarization. Moreover, most existing works on domain generalization have sophisticated training algorithms. In this paper, we propose a lightweight, weight averaging based, Domain Aligned Prefix Averaging approach to domain generalization for abstractive summarization. Given a number of source domains, our method first trains a prefix for each one of them. These source prefixes generate summaries for a small number of target domain documents. The similarity of the generated summaries to their corresponding documents is used for calculating weights required to average source prefixes. In DAPA, prefix tuning allows for lightweight finetuning, and weight averaging allows for the computationally efficient addition of new source domains. When evaluated on four diverse summarization domains, DAPA shows comparable or better performance against the baselines, demonstrating the effectiveness of its prefix averaging
    
[^26]: InterFormer: 混合局部和全局特征用于语音识别的交互式融合方法

    InterFormer: Interactive Local and Global Features Fusion for Automatic Speech Recognition. (arXiv:2305.16342v1 [cs.CL])

    [http://arxiv.org/abs/2305.16342](http://arxiv.org/abs/2305.16342)

    本文提出了InterFormer，用于交互式局部和全局特征融合，以学习更好的ASR表示。通过组合卷积块和变形器块，以及引入BFIM和SFM模块，实现了局部和全局特征的交互和融合，取得了在公共ASR数据集上优异的性能。

    

    对于自动语音识别（ASR）而言，局部和全局特征都是必不可少的。许多最近的方法已经证实，简单地合并局部和全局特征可以进一步提高ASR性能。然而，这些方法往往忽略了局部和全局特征之间的交互，并且它们的串行架构无法反映局部和全局特征之间的关系。为了解决这些问题，本文提出了InterFormer，用于交互式局部和全局特征融合，以学习更好的ASR表示。具体而言，我们将卷积块与变形器块以并行设计相结合。此外，我们提出了双向特征交互模块（BFIM）和选择性融合模块（SFM）来实现局部和全局特征的交互和融合。在公共ASR数据集上的大量实验表明了我们提出的InterFormer的有效性，并且相对于其他Transformer和Conformer模型具有更出色的性能。

    The local and global features are both essential for automatic speech recognition (ASR). Many recent methods have verified that simply combining local and global features can further promote ASR performance. However, these methods pay less attention to the interaction of local and global features, and their series architectures are rigid to reflect local and global relationships. To address these issues, this paper proposes InterFormer for interactive local and global features fusion to learn a better representation for ASR. Specifically, we combine the convolution block with the transformer block in a parallel design. Besides, we propose a bidirectional feature interaction module (BFIM) and a selective fusion module (SFM) to implement the interaction and fusion of local and global features, respectively. Extensive experiments on public ASR datasets demonstrate the effectiveness of our proposed InterFormer and its superior performance over the other Transformer and Conformer models.
    
[^27]: ASR技术与情感语音：对语音与情感识别相互影响的单词级探索

    ASR and Emotional Speech: A Word-Level Investigation of the Mutual Impact of Speech and Emotion Recognition. (arXiv:2305.16065v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2305.16065](http://arxiv.org/abs/2305.16065)

    本论文研究了ASR技术在情感语音上的表现，并探究了情感如何影响ASR。同时，还研究了ASR对基于文本的情感识别的影响。该研究旨在揭示ASR和SER之间的关系和相互影响，以促进ASR技术对情感语音的适应和SER技术在实际中的应用。

    

    在语音情感识别（SER）中，为了应对固有的变异性，通常会使用文本数据来辅助音频信号。然而，大多数研究中依赖于人工标注的文本数据，这阻碍了实用化SER系统的发展。为了克服这个挑战，我们使用四个ASR系统（分别是Kaldi ASR、wav2vec2、Conformer和Whisper）和三个语料库（IEMOCAP、MOSI和MELD）来分析情感语音上的ASR表现，并且通过分析ASR转录中的词错误和置信度分布来了解情感如何影响ASR。此外，我们对具有不断增加单词错误率的ASR转录进行基于文本的情感识别，以研究ASR如何影响SER。本研究的目标是揭示ASR和SER之间的关系和相互影响，以促进ASR技术对情感语音的适应和SER技术在实际中的应用。

    In Speech Emotion Recognition (SER), textual data is often used alongside audio signals to address their inherent variability. However, the reliance on human annotated text in most research hinders the development of practical SER systems. To overcome this challenge, we investigate how Automatic Speech Recognition (ASR) performs on emotional speech by analyzing the ASR performance on emotion corpora and examining the distribution of word errors and confidence scores in ASR transcripts to gain insight into how emotion affects ASR. We utilize four ASR systems, namely Kaldi ASR, wav2vec2, Conformer, and Whisper, and three corpora: IEMOCAP, MOSI, and MELD to ensure generalizability. Additionally, we conduct text-based SER on ASR transcripts with increasing word error rates to investigate how ASR affects SER. The objective of this study is to uncover the relationship and mutual impact of ASR and SER, in order to facilitate ASR adaptation to emotional speech and the use of SER in real world.
    
[^28]: ChatGPT中出现了音韵偏见

    Emergence of a phonological bias in ChatGPT. (arXiv:2305.15929v1 [cs.CL])

    [http://arxiv.org/abs/2305.15929](http://arxiv.org/abs/2305.15929)

    ChatGPT表现出人类语言处理的音韵偏见，更倾向于使用辅音而不是元音来识别单词。

    

    当前的大型语言模型，例如OpenAI的ChatGPT，因其在语言使用方面的出色表现而受到公众的关注。在这里，我证明了ChatGPT显示了人类语言处理的音韵偏见。更具体地说，就像人类一样，ChatGPT具有一个辅音偏见。也就是说，这个聊天机器人倾向于使用辅音而不是元音来识别单词。这在具有不同辅音和元音分布比例的语言（如英语和西班牙语）中都有观察到。尽管当前人工智能语言模型在处理语言刺激和人类婴儿获得语言的方式上存在差异，但这样的训练似乎足以在ChatGPT中引出一个音韵偏见。

    Current large language models, such as OpenAI's ChatGPT, have captured the public's attention because how remarkable they are in the use of language. Here, I demonstrate that ChatGPT displays phonological biases that are a hallmark of human language processing. More concretely, just like humans, ChatGPT has a consonant bias. That is, the chatbot has a tendency to use consonants over vowels to identify words. This is observed across languages that differ in their relative distribution of consonants and vowels such as English and Spanish. Despite the differences in how current artificial intelligence language models are trained to process linguistic stimuli and how human infants acquire language, such training seems to be enough for the emergence of a phonological bias in ChatGPT
    
[^29]: MEMEX：通过知识增强的上下文化来检测迷因的解释性证据

    MEMEX: Detecting Explanatory Evidence for Memes via Knowledge-Enriched Contextualization. (arXiv:2305.15913v1 [cs.CL])

    [http://arxiv.org/abs/2305.15913](http://arxiv.org/abs/2305.15913)

    本研究提出了MEMEX任务，通过知识增强的上下文化技术检测迷因的解释性证据。通过构建MCC数据集，使用分层方法捕捉迷因和上下文的跨模态语义依赖，提出了MIME多模式神经框架来解释迷因。

    

    迷因是社交媒体上强大的交际工具，它们在政治、历史和社会文化现象中的不断发展使其成为理想的交流媒介。为了理解迷因传达的微妙信息，必须了解促进其整体吸收的背景。除了像knowyourmeme.com这样的几个网站对迷因及其元数据进行数字存档外，目前没有有效的方法动态地推断迷因的上下文。在这项工作中，我们提出了一个新的任务，MEMEX，给定一个迷因和一个相关的文档，其目的是挖掘简洁地解释迷因背景的上下文。首先，我们开发了MCC（Meme Context Corpus），这是一个为MEMEX设计的新数据集。此外，为了基准测试MCC，我们提出了MIME（MultImodal Meme Explainer），这是一个多模式神经框架，使用通识强化的迷因表示和一种分层方法来捕捉迷因和上下文之间的跨模态语义依赖。

    Memes are a powerful tool for communication over social media. Their affinity for evolving across politics, history, and sociocultural phenomena makes them an ideal communication vehicle. To comprehend the subtle message conveyed within a meme, one must understand the background that facilitates its holistic assimilation. Besides digital archiving of memes and their metadata by a few websites like knowyourmeme.com, currently, there is no efficient way to deduce a meme's context dynamically. In this work, we propose a novel task, MEMEX given a meme and a related document, the aim is to mine the context that succinctly explains the background of the meme. At first, we develop MCC (Meme Context Corpus), a novel dataset for MEMEX. Further, to benchmark MCC, we propose MIME (MultImodal Meme Explainer), a multimodal neural framework that uses common sense enriched meme representation and a layered approach to capture the cross-modal semantic dependencies between the meme and the context. M
    
[^30]: 动态上下文剪枝用于高效和可解释的自回归变换器

    Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers. (arXiv:2305.15805v1 [cs.CL])

    [http://arxiv.org/abs/2305.15805](http://arxiv.org/abs/2305.15805)

    本研究提出了一种动态上下文剪枝方法，可以在保持模型表现力的同时，动态减少无效信息，提高模型的效率和可解释性。该技术可以应用于现有的预训练模型，并且可以通过简单的微调过程实现。

    

    大型语言模型中采用的自回归变换器难以扩展到长序列。尽管有几项工作试图减少它们的计算成本，但大多数LLM仍然在所有标记对之间采用注意层，从而产生二次成本。本研究提出了一种新方法，通过保留模型的表现力来动态修剪上下文信息，从而在推理过程中减少内存和计算要求。我们的方法使用可学习机制，在生成过程中确定哪些无关的标记可以从上下文中删除。通过这样做，我们的方法不仅解决了性能问题，而且增强了可解释性，为模型的决策过程提供了宝贵的洞察力。我们的技术可以通过简单的微调过程应用于现有的预训练模型，并且剪枝强度可以由稀疏度参数指定。

    Autoregressive Transformers adopted in Large Language Models (LLMs) are hard to scale to long sequences. Despite several works trying to reduce their computational cost, most of LLMs still adopt attention layers between all pairs of tokens in the sequence, thus incurring a quadratic cost. In this study, we present a novel approach that dynamically prunes contextual information while preserving the model's expressiveness, resulting in reduced memory and computational requirements during inference. Our method employs a learnable mechanism that determines which uninformative tokens can be dropped from the context at any point across the generation process. By doing so, our approach not only addresses performance concerns but also enhances interpretability, providing valuable insight into the model's decision-making process. Our technique can be applied to existing pre-trained models through a straightforward fine-tuning process, and the pruning strength can be specified by a sparsity para
    
[^31]: 回复长度感知与序列调度：一种利用LLM的高效LLM推理流水线。

    Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline. (arXiv:2305.13144v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13144](http://arxiv.org/abs/2305.13144)

    本文提出了一种利用LLMs的高效LLM推理流水线，通过利用LLMs准确感知和预测响应长度的潜力，并引入一种高效的序列调度技术，将具有类似响应长度的查询分组成微批。实验结果表明，该方法在实现高效的推理吞吐量的同时也不影响有效性。

    

    大型语言模型(LLMs)已经在各种任务上显示出前所未有的能力，革命了人工智能领域。然而，LLMs的推理过程具有重要的计算成本。本文提出了一种利用LLMs的高效LLM推理流水线。我们的方法通过利用LLMs准确感知和预测响应长度的潜力，并引入一种高效的序列调度技术，将具有类似响应长度的查询分组成微批。我们使用基于LLaMA模型的真实世界指令数据集来评估我们的方法，结果显示出86%的推理吞吐量的提高，同时不影响有效性。值得注意的是，我们的方法与其他推理加速技术无关，是LLMs推理许多现有工具包(如FlashAttention、量化)的有价值补充。

    Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.
    
[^32]: 文本到SQL的语言模型纠错

    Text-to-SQL Error Correction with Language Models of Code. (arXiv:2305.13073v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13073](http://arxiv.org/abs/2305.13073)

    本论文提出了一种基于从句编辑模型的文本到SQL的语言模型纠错方法，并通过新的SQL查询表示改进了语言模型的精确匹配准确率，提高了2.4-6.5，最多提高4.3个百分点。

    

    尽管文本到SQL解析取得了进展，但当前的语义解析器仍不够准确以实际应用。本文研究如何构建自动文本到SQL纠错模型。我们注意到单词层面的编辑缺乏上下文并且有时不明确，因此提出构建从句编辑模型。此外，虽然大多数代码语言模型没有专门预训练SQL，但它们熟悉Python等编程语言中的常见数据结构和其操作。因此，我们提出了一种新的SQL查询表示及其编辑方法，更符合代码语言模型的预训练语料库。我们的错误纠错模型提高了不同解析器的精确匹配准确率，提高了2.4-6.5，并获得了两个强基线的绝对改进最多4.3个百分点。我们的代码和数据可在https://github.com/OSU-NLP-Group/Auto-SQL-Correction 上找到。

    Despite recent progress in text-to-SQL parsing, current semantic parsers are still not accurate enough for practical use. In this paper, we investigate how to build automatic text-to-SQL error correction models. Noticing that token-level edits are out of context and sometimes ambiguous, we propose building clause-level edit models instead. Besides, while most language models of code are not specifically pre-trained for SQL, they know common data structures and their operations in programming languages such as Python. Thus, we propose a novel representation for SQL queries and their edits that adheres more closely to the pre-training corpora of language models of code. Our error correction model improves the exact set match accuracy of different parsers by 2.4-6.5 and obtains up to 4.3 point absolute improvement over two strong baselines. Our code and data are available at https://github.com/OSU-NLP-Group/Auto-SQL-Correction.
    
[^33]: 无须手语标注的全端到端手语翻译

    Gloss-Free End-to-End Sign Language Translation. (arXiv:2305.12876v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.12876](http://arxiv.org/abs/2305.12876)

    本论文介绍了一种无需手语标注的全端到端手语翻译框架(GloFE)，它通过利用手语和相应口语翻译的共同基础语义来提高手语翻译的性能，可以应用于实际场景中。

    

    本论文解决了无需手语标注的手语翻译问题。尽管像手语标语这样的中间表示法已被证明是有效的，但手语标注很难获取，特别是在大量获取时。这限制了翻译数据集的领域覆盖范围，从而影响了实际应用的效果。为了解决这个问题，我们设计了无需手语标注的全端到端手语翻译框架 (GloFE)。我们的方法通过利用手语和相应口语翻译的共同基础语义来提高无需手语标注设置下手语翻译的性能。通用概念从文本中提取出来并用作一种弱形式的中间表示。这些概念的全局嵌入用作跨注意力的查询，以查找学习到的视觉特征内的相应信息。我们以对比的方式鼓励包含这些概念的样本之间查询结果的相似性，并减少没有这些概念的样本之间的查询结果相似性。我们在一个大规模无需手语标注的手语翻译数据集上进行了实验，结果显示，GloFE相比于之前仅依靠手语标注信息的方法实现了最先进的性能。

    In this paper, we tackle the problem of sign language translation (SLT) without gloss annotations. Although intermediate representation like gloss has been proven effective, gloss annotations are hard to acquire, especially in large quantities. This limits the domain coverage of translation datasets, thus handicapping real-world applications. To mitigate this problem, we design the Gloss-Free End-to-end sign language translation framework (GloFE). Our method improves the performance of SLT in the gloss-free setting by exploiting the shared underlying semantics of signs and the corresponding spoken translation. Common concepts are extracted from the text and used as a weak form of intermediate representation. The global embedding of these concepts is used as a query for cross-attention to find the corresponding information within the learned visual features. In a contrastive manner, we encourage the similarity of query results between samples containing such concepts and decrease those 
    
[^34]: 通过二分搜索学习同时机器翻译的最优策略

    Learning Optimal Policy for Simultaneous Machine Translation via Binary Search. (arXiv:2305.12774v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12774](http://arxiv.org/abs/2305.12774)

    本文提出通过二分搜索学习同时机器翻译最优策略的方法，并在多个翻译任务上验证了在所有延迟情景下超越强基线的效果。

    

    同时机器翻译（SiMT）在阅读源句子时开始输出翻译，并需要精确的策略来决定何时输出生成的翻译。因此，该策略决定了在翻译每个目标令牌期间读取的源标记数量。然而，学习精确的翻译策略以实现良好的延迟质量权衡是困难的，因为没有与并行句子对应的黄金策略作为显式监督。本文提出了一种通过二分搜索在线构建最优策略的新方法。通过采用显式监督，我们的方法使SiMT模型能够学习最优策略，这可以指导模型在推理过程中完成翻译。在四个翻译任务上的实验结果表明，我们的方法可以在所有延迟方案下超越强基线。

    Simultaneous machine translation (SiMT) starts to output translation while reading the source sentence and needs a precise policy to decide when to output the generated translation. Therefore, the policy determines the number of source tokens read during the translation of each target token. However, it is difficult to learn a precise translation policy to achieve good latency-quality trade-offs, because there is no golden policy corresponding to parallel sentences as explicit supervision. In this paper, we present a new method for constructing the optimal policy online via binary search. By employing explicit supervision, our approach enables the SiMT model to learn the optimal policy, which can guide the model in completing the translation during inference. Experiments on four translation tasks show that our method can exceed strong baselines across all latency scenarios.
    
[^35]: 探索不同架构和训练方法下基于能量的语言模型在语音识别中的应用

    Exploring Energy-based Language Models with Different Architectures and Training Methods for Speech Recognition. (arXiv:2305.12676v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12676](http://arxiv.org/abs/2305.12676)

    本文探索了不同的能量函数架构和不同的训练方法，以提高基于能量的语言模型在语音识别中计算句子得分的能力。

    

    基于能量的语言模型（ELM）通过参数化自然语句的非归一化分布与流行的自回归语言模型（ALM）有根本性区别。作为一种重要的应用，ELM已成功地用于语音识别中计算句子得分，但它们都使用不太现代的CNN或LSTM网络。随着Transformer网络和大型预训练模型（如BERT和GPT2）的最新进展，进一步提高ELMs的能力已经成为可能。在本文中，我们探索了不同的能量函数架构和不同的训练方法，以研究在以大型预训练模型作为骨干的语音识别中，ELMs的能力。

    Energy-based language models (ELMs) parameterize an unnormalized distribution for natural sentences and are radically different from popular autoregressive language models (ALMs). As an important application, ELMs have been successfully used as a means for calculating sentence scores in speech recognition, but they all use less-modern CNN or LSTM networks. The recent progress in Transformer networks and large pretrained models such as BERT and GPT2 opens new possibility to further advancing ELMs. In this paper, we explore different architectures of energy functions and different training methods to investigate the capabilities of ELMs in rescoring for speech recognition, all using large pretrained models as backbones.
    
[^36]: 上下文化的短语预测网络在端到端语音识别中的应用

    Contextualized End-to-End Speech Recognition with Contextual Phrase Prediction Network. (arXiv:2305.12493v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2305.12493](http://arxiv.org/abs/2305.12493)

    本研究引入了一个上下文短语预测网络用于基于注意力的深度偏置方法，通过计算偏置损失以帮助训练上下文化模型，在多种端到端语音识别模型上实现了显著的WER降低，相对于基线模型相对WER提高了12.1％，上下文短语的WER相对降低了40.5％。

    

    上下文信息在语音识别技术中发挥着至关重要的作用，将其融入端到端语音识别模型近年来引起了极大的兴趣。然而，先前的深度偏置方法缺乏偏置任务的显式监督。本研究引入了一个上下文短语预测网络用于基于注意力的深度偏置方法。该网络利用上下文嵌入预测发音中的上下文短语，并计算偏置损失以帮助训练上下文化模型。我们的方法在多种端到端语音识别模型上实现了显著的单词错误率(WER)降低。对LibriSpeech语料库的实验结果表明，在基线模型上，我们提出的模型相对WER提高了12.1％，上下文短语的WER相对降低了40.5％。此外，通过应用上下文短语过滤策略，我们还有效消除了使用更大的偏置列表时的WER降级现象。

    Contextual information plays a crucial role in speech recognition technologies and incorporating it into the end-to-end speech recognition models has drawn immense interest recently. However, previous deep bias methods lacked explicit supervision for bias tasks. In this study, we introduce a contextual phrase prediction network for an attention-based deep bias method. This network predicts context phrases in utterances using contextual embeddings and calculates bias loss to assist in the training of the contextualized model. Our method achieved a significant word error rate (WER) reduction across various end-to-end speech recognition models. Experiments on the LibriSpeech corpus show that our proposed model obtains a 12.1% relative WER improvement over the baseline model, and the WER of the context phrases decreases relatively by 40.5%. Moreover, by applying a context phrase filtering strategy, we also effectively eliminate the WER degradation when using a larger biasing list.
    
[^37]: 后验解释可以提高语言模型的性能

    Post Hoc Explanations of Language Models Can Improve Language Models. (arXiv:2305.11426v1 [cs.CL])

    [http://arxiv.org/abs/2305.11426](http://arxiv.org/abs/2305.11426)

    本文提出了一种新的框架AMPLIFY，利用后验解释自动化生成原因，并在多个数据集和任务上显著提高现有语言模型的性能。

    

    大型语言模型在执行复杂任务方面表现出了非凡的能力。最近的研究显示，在上下文学习过程中加入人类注释的原理（例如，思维链提示）可以显著提高这些模型的性能，特别是在需要推理能力的任务上。然而，这样的原理加入在可扩展性方面存在挑战，因为这需要高度的人工参与。本文提出了一种新框架，即通过利用后验解释的上下文学习来放大模型性能，来解决上述挑战。为此，我们利用后验解释方法的结果，该方法输出称为属性分数（解释）的值，用于捕获每个输入特征对模型预测的影响。更具体地说，我们构建了自动化的自然语言原理，其中包含从属性分数中获得的信息，以便用户可以更好地理解模型的决策。实验结果表明，AMPLIFY可以在多个数据集和任务上显著提高现有语言模型的性能。

    Large Language Models (LLMs) have demonstrated remarkable capabilities in performing complex tasks. Moreover, recent research has shown that incorporating human-annotated rationales (e.g., Chain-of- Thought prompting) during in-context learning can significantly enhance the performance of these models, particularly on tasks that require reasoning capabilities. However, incorporating such rationales poses challenges in terms of scalability as this requires a high degree of human involvement. In this work, we present a novel framework, Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY), which addresses the aforementioned challenges by automating the process of rationale generation. To this end, we leverage post hoc explanation methods which output attribution scores (explanations) capturing the influence of each of the input features on model predictions. More specifically, we construct automated natural language rationales that embed insi
    
[^38]: ConvXAI：通过对话提供异构的AI解释，支持人机科技写作

    ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing. (arXiv:2305.09770v1 [cs.HC])

    [http://arxiv.org/abs/2305.09770](http://arxiv.org/abs/2305.09770)

    ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。

    

    尽管已经提出了各种各样的人工智能解释（XAI）方法来解释AI系统，但目前的方法是否对人类实用仍存在不一致的发现。为了改善XAI方法的实用性，一系列研究确定了现实世界中多样化和动态的用户需求与现有XAI方法之间的差距。虽然之前的研究设想将多种XAI方法集成到通用XAI界面（例如，基于对话或GUI的XAI系统）中以减轻这些差距，但缺少针对这些系统如何设计以满足实际用户需求的研究。在本研究中，我们提出了ConvXAI，这是一个基于对话的XAI系统，它结合了多种XAI类型，并赋予用户通过通用的XAI对话界面提出各种XAI问题的能力。特别地，我们创新地将实际用户需求（即，基于格式研究的四个原则）嵌入ConvXAI设计中，以提高实用性。

    While various AI explanation (XAI) methods have been proposed to interpret AI systems, whether the state-of-the-art XAI methods are practically useful for humans remains inconsistent findings. To improve the usefulness of XAI methods, a line of studies identifies the gaps between the diverse and dynamic real-world user needs with the status quo of XAI methods. Although prior studies envision mitigating these gaps by integrating multiple XAI methods into the universal XAI interfaces (e.g., conversational or GUI-based XAI systems), there is a lack of work investigating how these systems should be designed to meet practical user needs. In this study, we present ConvXAI, a conversational XAI system that incorporates multiple XAI types, and empowers users to request a variety of XAI questions via a universal XAI dialogue interface. Particularly, we innovatively embed practical user needs (i.e., four principles grounding on the formative study) into ConvXAI design to improve practical useful
    
[^39]: 1335种语言概念表达的跨语言研究

    A Crosslingual Investigation of Conceptualization in 1335 Languages. (arXiv:2305.08475v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08475](http://arxiv.org/abs/2305.08475)

    通过一个新方法，该论文探索了1335种语言之间在概念表达方面的差异，并发现了能够预测一种概念是否会跨越多种语言，以及如何代表每种语言的多维分类方法，这有助于解决现有自然语言处理模型的不足。

    

    语言在如何将世界分为概念和词汇方面存在差异；例如，与英语不同，斯瓦希里语有一个单一的概念来表示“肚子”和“子宫”。我们通过在平行语料库中对齐概念，调查1,335种语言之间的这些差异。为此，我们提出了Conceptualizer方法，它创建了一个源语言概念和一组目标语言字符串之间的二分有向对齐图。通过对一个概念（“鸟”）进行所有语言的详细语言分析，并在32个Swadesh概念的黄金标准数据上进行评估，我们证明了Conceptualizer具有良好的对齐精度。我们通过两个实验展示了NLP中概念表达研究的潜力。第一个实验我们定义了横跨多种语言的稳定性概念，并证明了具象程度可以预测其稳定性。第二个实验我们通过83个概念的概念表达模式，来代表每种语言，并定义了一种多维语言分类方法，它可以帮助填补现有NLP模型的不足。

    Languages differ in how they divide up the world into concepts and words; e.g., in contrast to English, Swahili has a single concept for `belly' and `womb'. We investigate these differences in conceptualization across 1,335 languages by aligning concepts in a parallel corpus. To this end, we propose Conceptualizer, a method that creates a bipartite directed alignment graph between source language concepts and sets of target language strings. In a detailed linguistic analysis across all languages for one concept (`bird') and an evaluation on gold standard data for 32 Swadesh concepts, we show that Conceptualizer has good alignment accuracy. We demonstrate the potential of research on conceptualization in NLP with two experiments. (1) We define crosslingual stability of a concept as the degree to which it has 1-1 correspondences across languages, and show that concreteness predicts stability. (2) We represent each language by its conceptualization pattern for 83 concepts, and define a si
    
[^40]: 从预训练数据到语言模型再到下游任务：追踪导致不公平NLP模型的政治偏见的轨迹。

    From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models. (arXiv:2305.08283v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08283](http://arxiv.org/abs/2305.08283)

    本文研究测量了政治偏见在预训练语言模型和下游任务中的影响，发现预训练模型存在政治倾向，并将社会偏见传递到下游任务中，从而导致NLP模型的不公平性。

    

    语言模型(LMs)是预训练在不同数据源上的，包括新闻、讨论论坛、书籍和在线百科全书等。这些数据中的相当一部分包括观点和角度，一方面赞扬民主和思想多样性，另一方面具有固有的社会偏见。我们的工作开发了新的方法来(1)测量基于此类语料库训练的LMs中的政治偏见，沿社会和经济轴，以及(2)衡量基于政治偏见的LMs训练的下游NLP模型的公平性。我们关注仇恨言论和虚假信息检测，旨在实证量化预训练数据中政治(社会、经济)偏见对高风险社会导向任务公正性的影响。我们的研究结果表明，预训练LMs确实存在政治倾向，加强了预训练语料库中存在的极化，将社会偏见传播到仇恨言论预测和虚假信息检测器中。我们讨论了研究的意义，并提出了开发更公正、更无偏NLP模型的建议。

    Language models (LMs) are pretrained on diverse data sources, including news, discussion forums, books, and online encyclopedias. A significant portion of this data includes opinions and perspectives which, on one hand, celebrate democracy and diversity of ideas, and on the other hand are inherently socially biased. Our work develops new methods to (1) measure political biases in LMs trained on such corpora, along social and economic axes, and (2) measure the fairness of downstream NLP models trained on top of politically biased LMs. We focus on hate speech and misinformation detection, aiming to empirically quantify the effects of political (social, economic) biases in pretraining data on the fairness of high-stakes social-oriented tasks. Our findings reveal that pretrained LMs do have political leanings that reinforce the polarization present in pretraining corpora, propagating social biases into hate speech predictions and misinformation detectors. We discuss the implications of our
    
[^41]: 零样本信实事实纠错

    Zero-shot Faithful Factual Error Correction. (arXiv:2305.07982v1 [cs.CL])

    [http://arxiv.org/abs/2305.07982](http://arxiv.org/abs/2305.07982)

    零样本的忠实事实性错误纠正框架超越完全监督方法，具有较高的及解释性和忠实性评估标准，适用于维护文本知识库和预防序列到序列模型中的幻觉。

    

    忠实地纠正事实性错误对于维护文本知识库的完整性和防止序列到序列模型中的幻觉至关重要。借鉴人类识别和纠正事实错误的能力，我们提出了一个零样本框架，该框架制定有关输入声明的问题，查找给定证据中的正确答案，并根据其与证据的一致性评估每个纠正的信实性。我们的零样本框架在FEVER和SciFact数据集上进行的实验中比完全监督的方法表现更好，证明了我们的输出更加忠实。更重要的是，我们框架的可分解性天然提供了可解释性。此外，为了揭示评估事实错误修正的最合适度量标准，我们分析了常用度量标准与三个不同维度的人类判断之间的相关性，包括可理解性和忠实性。

    Faithfully correcting factual errors is critical for maintaining the integrity of textual knowledge bases and preventing hallucinations in sequence-to-sequence models. Drawing on humans' ability to identify and correct factual errors, we present a zero-shot framework that formulates questions about input claims, looks for correct answers in the given evidence, and assesses the faithfulness of each correction based on its consistency with the evidence. Our zero-shot framework outperforms fully-supervised approaches, as demonstrated by experiments on the FEVER and SciFact datasets, where our outputs are shown to be more faithful. More importantly, the decomposability nature of our framework inherently provides interpretability. Additionally, to reveal the most suitable metrics for evaluating factual error corrections, we analyze the correlation between commonly used metrics with human judgments in terms of three different dimensions regarding intelligibility and faithfulness.
    
[^42]: Decker: 双重检查与异构知识用于常识事实验证

    Decker: Double Check with Heterogeneous Knowledge for Commonsense Fact Verification. (arXiv:2305.05921v1 [cs.CL])

    [http://arxiv.org/abs/2305.05921](http://arxiv.org/abs/2305.05921)

    Decker是一种能够利用结构化和非结构化知识之间的潜在关系来桥接异构知识的常识事实验证模型，具有良好的验证效果和获取珍贵信息的能力。

    

    常识事实验证作为常识问答的一个具有挑战性的分支，旨在通过事实来验证一个给定的常识论断是否正确。回答常识问题需要从不同层次的知识中进行组合。然而，现有的研究主要依靠把握非结构化证据或从结构化知识库中找到潜在的推理路径，但没有同时利用异构知识的好处。鉴于此，我们提出了Decker，一种常识事实验证模型，能够通过发现结构化和非结构化知识之间的潜在关系来桥接异构知识。在两个常识事实验证基准数据集CSQA2.0和CREAK上的实验结果证明了我们Decker的有效性，进一步的分析验证了它在推理中获取更多珍贵信息的能力。

    Commonsense fact verification, as a challenging branch of commonsense question-answering (QA), aims to verify through facts whether a given commonsense claim is correct or not. Answering commonsense questions necessitates a combination of knowledge from various levels. However, existing studies primarily rest on grasping either unstructured evidence or potential reasoning paths from structured knowledge bases, yet failing to exploit the benefits of heterogeneous knowledge simultaneously. In light of this, we propose Decker, a commonsense fact verification model that is capable of bridging heterogeneous knowledge by uncovering latent relationships between structured and unstructured knowledge. Experimental results on two commonsense fact verification benchmark datasets, CSQA2.0 and CREAK demonstrate the effectiveness of our Decker and further analysis verifies its capability to seize more precious information through reasoning.
    
[^43]: FACTIFY-5WQA：基于问题回答的5W因素事实验证

    FACTIFY-5WQA: 5W Aspect-based Fact Verification through Question Answering. (arXiv:2305.04329v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.04329](http://arxiv.org/abs/2305.04329)

    本文提出了一个基于问题回答的5W因素事实验证框架，并提供了一个半自动产生的数据集FACTIFY-5WQA，以便协助人类核查员针对事实提出相关问题并验证以达出最终结论。实验结果表明，这种方式相较于其他基线模型和常规事实验证系统具有更高的效率和可行性。

    

    自动事实验证近年来受到了广泛关注。当今的自动事实核查系统主要集中于使用数字评分来估计真实性，而这些评分往往无法被人类理解。而人类核查员通常会按照一些逻辑步骤来验证一个类似真实的主张，并得出其真实性还是虚假性的结论。因此，需要一种基于方面（区分哪些部分是真实的，哪些是虚假的）的可解释系统，可以协助人类核查员针对事实提出相关问题，然后分别验证以得出最终结论。本文中，我们提出了一个5W框架（谁、什么、何时、何地和为什么）用于问题回答式的事实可解释性，并提供了一个半自动产生的数据集FACTIFY-5WQA，该数据集包含39个方面，每个方面都对应某个问题和它的答案，以及是否可验证。我们提出了一个模型，它融合了双向长短时记忆网络和多头自注意力机制，用于执行5WQA任务。我们还提供了一个公共基准测试集进行评测和比较。实验结果表明，我们的模型在多个指标上都超过了其他现有的基线模型和常规事实验证系统，这证明了我们的方法的有效性和可行性。

    Automatic fact verification has received significant attention recently. Contemporary automatic fact-checking systems focus on estimating truthfulness using numerical scores which are not human-interpretable. A human fact-checker generally follows several logical steps to verify a verisimilitude claim and conclude whether its truthful or a mere masquerade. Popular fact-checking websites follow a common structure for fact categorization such as half true, half false, false, pants on fire, etc. Therefore, it is necessary to have an aspect-based (delineating which part(s) are true and which are false) explainable system that can assist human fact-checkers in asking relevant questions related to a fact, which can then be validated separately to reach a final verdict. In this paper, we propose a 5W framework (who, what, when, where, and why) for question-answer-based fact explainability. To that end, we present a semi-automatically generated dataset called FACTIFY-5WQA, which consists of 39
    
[^44]: 视觉与定义相遇：融合词义信息的无监督视觉词义消歧

    Vision Meets Definitions: Unsupervised Visual Word Sense Disambiguation Incorporating Gloss Information. (arXiv:2305.01788v1 [cs.CL])

    [http://arxiv.org/abs/2305.01788](http://arxiv.org/abs/2305.01788)

    本文提出了一种无监督的视觉词义消歧方法，通过引入外部词汇知识库的词义信息来解决原来图像-文本匹配模型中的多义词问题。采用贝叶斯推断来加入词义定义，并通过与上下文相关的 GPT-3 定义生成方法，成功解决了词典外问题。

    

    视觉词义消歧是一项任务，旨在找到最准确地描述给定上下文中目标词正确意义的图像。以往的图像-文本匹配模型往往受到词义多义性的影响。本文介绍了一种无监督的视觉词义消歧方法，该方法使用了外部词汇知识库的词汇信息，特别是词义定义。具体而言，我们建议在没有提供答案的词义信息时，采用贝叶斯推断来加入词义定义。此外，为了改进词典外问题，我们提出了一种与上下文相关的GPT-3定义生成方法。实验结果表明，我们的基于贝叶斯推断的方法明显提高了视觉词义消歧的性能。此外，我们的上下文相关定义生成方法在词典外例子上取得了显著的性能提升，表现优于现有的定义生成方法。

    Visual Word Sense Disambiguation (VWSD) is a task to find the image that most accurately depicts the correct sense of the target word for the given context. Previously, image-text matching models often suffered from recognizing polysemous words. This paper introduces an unsupervised VWSD approach that uses gloss information of an external lexical knowledge-base, especially the sense definitions. Specifically, we suggest employing Bayesian inference to incorporate the sense definitions when sense information of the answer is not provided. In addition, to ameliorate the out-of-dictionary (OOD) issue, we propose a context-aware definition generation with GPT-3. Experimental results show that the VWSD performance significantly increased with our Bayesian inference-based approach. In addition, our context-aware definition generation achieved prominent performance improvement in OOD examples exhibiting better performance than the existing definition generation method. We will publish source 
    
[^45]: 使用指令调整的LLM和潜在扩散模型生成文本到音频

    Text-to-Audio Generation using Instruction-Tuned LLM and Latent Diffusion Model. (arXiv:2304.13731v1 [eess.AS])

    [http://arxiv.org/abs/2304.13731](http://arxiv.org/abs/2304.13731)

    本研究提出了一种使用指令调整的LLM Flan-T5作为文本编码器和基于潜在扩散模型(LDM)的方法TANGO生成文本到音频(TTA)的新方法，在AudioCaps测试集上表现优于先进的AudioLDM。

    

    最近的大型语言模型(LLM)的巨大规模允许许多有趣的属性，比如，基于指令和思路链的微调，在许多自然语言处理(NLP)任务中显着提高了零次和少量训练样本的性能。受到这些成功的启发，我们采用了这样一种经过指令调整的LLM Flan-T5作为文本编码器，用于文本到音频(TTA)生成任务——目标是根据其文本描述生成音频。之前关于TTA的工作要么预先训练一个联合的文本-音频编码器，要么使用一个非指令调谐的模型，如T5。因此，我们基于潜在扩散模型(LDM)的方法TANGO在AudioCaps测试集上表现出比最先进的AudioLDM更好的大多数指标，并在其余指标上持平，尽管我们使用了63倍小的数据集来训练LDM，并保持文本编码器不变。这种改进可能还归因于采用基于音频压力级的混音训练集增强。

    The immense scale of the recent large language models (LLM) allows many interesting properties, such as, instruction- and chain-of-thought-based fine-tuning, that has significantly improved zero- and few-shot performance in many natural language processing (NLP) tasks. Inspired by such successes, we adopt such an instruction-tuned LLM Flan-T5 as the text encoder for text-to-audio (TTA) generation -- a task where the goal is to generate an audio from its textual description. The prior works on TTA either pre-trained a joint text-audio encoder or used a non-instruction-tuned model, such as, T5. Consequently, our latent diffusion model (LDM)-based approach TANGO outperforms the state-of-the-art AudioLDM on most metrics and stays comparable on the rest on AudioCaps test set, despite training the LDM on a 63 times smaller dataset and keeping the text encoder frozen. This improvement might also be attributed to the adoption of audio pressure level-based sound mixing for training set augmenta
    
[^46]: 大型语言模型对齐的基本限制

    Fundamental Limitations of Alignment in Large Language Models. (arXiv:2304.11082v1 [cs.CL])

    [http://arxiv.org/abs/2304.11082](http://arxiv.org/abs/2304.11082)

    本文通过提出一种理论方法——行为期望边界（BEB），展示了大型语言模型中对齐的基本限制，并证明任何对齐过程都无法根除不希望的行为，这对于防止恶意攻击是不安全的。

    

    开发与人交互的语言模型的重要方面是对齐其行为，使其对其人类用户有用且无害。这通常通过调整模型的方式来实现，以增强所需的行为并抑制不希望的行为。在本文中，我们提出了一种名为行为期望边界(BEB)的理论方法，它允许我们正式研究大型语言模型中的几个内在特征和对齐的限制。重要的是，我们证明对于任何具有被该模型表现出的有限概率的行为，都存在可以触发模型输出此行为的提示，其概率随提示的长度增加而增加。这意味着任何减弱不希望的行为但未将其完全消除的对齐过程都无法抵御针对性攻击。此外，我们的框架提示了领先的

    An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading al
    
[^47]: 自然语言生成对话服务的聊天体验预测因素研究

    Which Factors Predict the Chat Experience of a Natural Language Generation Dialogue Service?. (arXiv:2304.10785v1 [cs.CL])

    [http://arxiv.org/abs/2304.10785](http://arxiv.org/abs/2304.10785)

    本文研究了自然语言生成对话系统中影响聊天体验的多种因素，包括提示、连贯性、情感、相似性和用户对话代理的好感度，发现用户的好感度和连贯性、情感、相似性是聊天体验的正向预测因素。此外，用户可能更喜欢具有外向性、开放性、责任心、宜人性和非神经质特征的对话代理。

    

    本文提出了一个概念性模型，用于预测自然语言生成对话系统中的聊天体验。我们使用部分最小二乘结构方程建模方法（PLS-SEM）对120名参与者进行了评估，并获得了0.541的R方值。该模型考虑了多种因素，包括用于生成的提示、对话中的连贯性、情感和相似性，以及用户对话代理的好感度。我们进一步探讨了我们提出的模型子集的有效性。结果显示，用户的好感度和对话中的连贯性、情感和相似性是用户聊天体验的正向预测因素。此外，我们发现用户可能更喜欢具有外向性、开放性、责任心、宜人性和非神经质特征的对话代理。通过我们的研究，自适应对话系统可以使用收集到的数据来推断我们模型中的因素，并通过调整对话代理的特征来预测用户的聊天体验。

    In this paper, we proposed a conceptual model to predict the chat experience in a natural language generation dialog system. We evaluated the model with 120 participants with Partial Least Squares Structural Equation Modeling (PLS-SEM) and obtained an R-square (R2) with 0.541. The model considers various factors, including the prompts used for generation; coherence, sentiment, and similarity in the conversation; and users' perceived dialog agents' favorability. We then further explore the effectiveness of the subset of our proposed model. The results showed that users' favorability and coherence, sentiment, and similarity in the dialogue are positive predictors of users' chat experience. Moreover, we found users may prefer dialog agents with characteristics of Extroversion, Openness, Conscientiousness, Agreeableness, and Non-Neuroticism. Through our research, an adaptive dialog system might use collected data to infer factors in our model, predict the chat experience for users through 
    
[^48]: 用自然语言学习编程

    Learning to Program with Natural Language. (arXiv:2304.10464v1 [cs.CL])

    [http://arxiv.org/abs/2304.10464](http://arxiv.org/abs/2304.10464)

    该论文提出了一种用自然语言作为编程语言并通过学习编程方法让大语言模型直接生成自然语言程序并指导推理的方法。实验结果表明，这种方法在解决编程任务上比基线方法有更高的成功率。

    

    大语言模型在各种基本自然语言任务中表现出卓越性能，这引起了实现人工通用智能的希望。为了更好地完成复杂任务，我们需要利用大语言模型进行编程，然后按照程序生成特定的解决方案。我们提出使用自然语言作为一种新的编程语言来描述任务过程，使它们易于人类和大语言模型理解。虽然大语言模型能够直接生成自然语言程序，但这些程序可能仍然存在错误或不完整的步骤。因此，我们进一步提出了学习编程（LP）的方法，要求大语言模型从复杂任务的训练数据集中学习自然语言程序，然后使用学习到的程序来指导推理。我们在AMPS（高中数学）和Math（竞赛数学问题）数据集上的实验证明了我们方法的有效性。在测试ChatGP解决编程任务时，LP能够实现80%的成功率，优于基线方法。

    Large Language Models (LLMs) have shown remarkable performance in various basic natural language tasks, which raises hopes for achieving Artificial General Intelligence. To better complete complex tasks, we need LLMs to program for the task and then follow the program to generate a specific solution for the test sample. We propose using natural language as a new programming language to describe task procedures, making them easily understandable to both humans and LLMs. LLMs are capable of directly generating natural language programs, but these programs may still contain factual errors or incomplete steps. Therefore, we further propose the Learning to Program (LP) method to ask LLMs themselves to learn natural language programs from the training dataset of complex tasks and then use the learned program to guide inference. Our experiments on the AMPS (high school math) and Math (competition mathematics problems) datasets demonstrate the effectiveness of our approach. When testing ChatGP
    
[^49]: 通过盲审评估和文本分类算法比较ChatGPT生成的抽象摘要和真实摘要

    Comparing Abstractive Summaries Generated by ChatGPT to Real Summaries Through Blinded Reviewers and Text Classification Algorithms. (arXiv:2303.17650v1 [cs.CL])

    [http://arxiv.org/abs/2303.17650](http://arxiv.org/abs/2303.17650)

    本研究评估了ChatGPT在抽象概括方面的表现，自动化指标和盲审人员评估显示ChatGPT生成的摘要在人类视角下难以分辨真假。

    

    大型语言模型（LLMs）因其在各种任务上的出色表现而受到广泛关注。OpenAI开发的ChatGPT是语言模型家族的最新成员，由于其类人的文本生成能力，被一些人称为一项颠覆性技术。尽管网络上有许多ChatGPT的例子来评估其强弱之处，但只有少数系统性的研究存在。为了为ChatGPT的系统性研究做出贡献，我们通过自动化指标和盲审人员评估了ChatGPT在抽象概括方面的表现。我们还构建了自动文本分类器来检测ChatGPT生成的摘要。我们发现，虽然文本分类算法可以区分真实和生成的摘要，但人类无法区分真实摘要和ChatGPT生成的摘要。

    Large Language Models (LLMs) have gathered significant attention due to their impressive performance on a variety of tasks. ChatGPT, developed by OpenAI, is a recent addition to the family of language models and is being called a disruptive technology by a few, owing to its human-like text-generation capabilities. Although, many anecdotal examples across the internet have evaluated ChatGPT's strength and weakness, only a few systematic research studies exist. To contribute to the body of literature of systematic research on ChatGPT, we evaluate the performance of ChatGPT on Abstractive Summarization by the means of automated metrics and blinded human reviewers. We also build automatic text classifiers to detect ChatGPT generated summaries. We found that while text classification algorithms can distinguish between real and generated summaries, humans are unable to distinguish between real summaries and those produced by ChatGPT.
    
[^50]: ChatGPT4PCG比赛：科学鸟角色级生成

    ChatGPT4PCG Competition: Character-like Level Generation for Science Birds. (arXiv:2303.15662v1 [cs.AI])

    [http://arxiv.org/abs/2303.15662](http://arxiv.org/abs/2303.15662)

    本论文介绍了举办在2023 IEEE游戏会议上的第一届ChatGPT4PCG比赛，目标是让ChatGPT生成具有高稳定性和类似角色的特质来生成具有科学鸟角色级水平的关卡。

    

    本文介绍了2023年IEEE游戏会议上的第一届ChatGPT4PCG比赛。本次比赛的目标是让参赛者通过创造性和提示工程技能，为ChatGPT创建有效的提示，使其能够具有高稳定性和类似角色的特质来生成具有科学鸟角色级水平的关卡。为了降低参赛门槛，我们将任务限制在生成大写英文字母。参赛作品的质量由其稳定性和与给定字符的相似性决定。给参赛者提供了一个样例提示供参考。

    This paper presents the first ChatGPT4PCG Competition at the 2023 IEEE Conference on Games. The objective of this competition is for participants to create effective prompts for ChatGPT--enabling it to generate Science Birds levels with high stability and character-like qualities--fully using their creativity as well as prompt engineering skills. ChatGPT is a conversational agent developed by OpenAI. Science Birds is selected as the competition platform because designing an Angry Birds-like level is not a trivial task due to the in-game gravity; the playability of the levels is determined by their stability. To lower the entry barrier to the competition, we limit the task to the generation of capitalized English alphabetical characters. Here, the quality of the generated levels is determined by their stability and similarity to the given characters. A sample prompt is provided to participants for their reference. An experiment is conducted to determine the effectiveness of its modified
    
[^51]: 直接和间接证据表明词汇长度被压缩. 对 Zipf的缩写定律的重新审视.

    Direct and indirect evidence of compression of word lengths. Zipf's law of abbreviation revisited. (arXiv:2303.10128v1 [cs.CL])

    [http://arxiv.org/abs/2303.10128](http://arxiv.org/abs/2303.10128)

    这篇论文重新审视了书面语与缩写定律的一致性，并发现这一定律也适用于口语。结果提供了压缩语言的间接证据，即缩写定律是最优编码的预测，而通过英语的历史研究还发现，人们在语言中实际使用的词汇数量正在缩减。

    

    Zipf缩写定律指的是更常见的单词更短，是语言普遍性的最坚实的候选者，它有可能是没有例外或者例外非常少的语言普遍性。自从Zipf的开创性研究以来，这一定律一直被认为是通信的普遍原理的表现，即通过缩短词汇长度来减少通信的努力。在这里，我们重新审视了书面语与缩写定律之间的一致性。关键地，我们提供更广泛的证据，表明这一定律也适用于口语（当用时间来测量词汇长度时），特别是适用于来自14个语言家族的46种语言。与缩写定律的一致性提供了压缩语言的间接证据，这是通过理论论证得出的，即缩写定律是最优编码的预测。鉴于需要直接证据来证明压缩，我们通过英语的历史研究发现，人们在语言中实际使用的词汇数量正在缩减。

    Zipf's law of abbreviation, the tendency of more frequent words to be shorter, is one of the most solid candidates for a linguistic universal, in the sense that it has the potential for being exceptionless or with a number of exceptions that is vanishingly small compared to the number of languages on Earth. Since Zipf's pioneering research, this law has been viewed as a manifestation of a universal principle of communication, i.e. the minimization of word lengths, to reduce the effort of communication. Here we revisit the concordance of written language with the law of abbreviation. Crucially, we provide wider evidence that the law holds also in speech (when word length is measured in time), in particular in 46 languages from 14 linguistic families. Agreement with the law of abbreviation provides indirect evidence of compression of languages via the theoretical argument that the law of abbreviation is a prediction of optimal coding. Motivated by the need of direct evidence of compressi
    
[^52]: CB2：合作自然语言交互研究平台

    CB2: Collaborative Natural Language Interaction Research Platform. (arXiv:2303.08127v1 [cs.LG])

    [http://arxiv.org/abs/2303.08127](http://arxiv.org/abs/2303.08127)

    CB2是一个用于研究基于任务的合作自然语言交互的平台，在3D游戏环境中提供了后端服务器和各种工具和流程。它在可扩展的研究中展示了学习的指令跟随模型。

    

    CB2 是一个多智能体平台，用于研究基于任务的情境下的合作自然语言交互。它包括一个 3D 游戏环境、一个后端服务器，可为人类智能体提供训练模型，以及各种工具和流程，以实现可扩展性的研究。我们在 https://cb2.ai 上展示了一个具有学习指令跟随模型的系统演示。

    CB2 is a multi-agent platform to study collaborative natural language interaction in a grounded task-oriented scenario. It includes a 3D game environment, a backend server designed to serve trained models to human agents, and various tools and processes to enable scalable studies. We deploy CB2 at https://cb2.ai as a system demonstration with a learned instruction following model.
    
[^53]: 动态提示：用于提示调整的统一框架

    Dynamic Prompting: A Unified Framework for Prompt Tuning. (arXiv:2303.02909v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.02909](http://arxiv.org/abs/2303.02909)

    本论文提出了一个统一的动态提示（DP）调整策略用于优化提示调整的性能，该策略可以动态地确定不同的提示变量来捕获额外的语义信息。

    

    已经证明，提示调整技术可以高效地从基础预训练模型中提取知识，包括预训练语言模型（PLMs）、预训练视觉模型和视觉语言模型 (V-L)。然而，采用固定的软提示来与所有实例连接输入，而忽略它们的固有差异，其有效性仍不确定。例如提示的位置、长度和表示在不同实例和任务中的不同变量，可以显著影响提示调整的性能。在此背景下，我们提供了一个理论分析。这个分析发现，优化提示的位置可以捕获传统前缀或后缀提示调整方法无法捕获的额外语义信息。基于我们的分析，我们提出了一个统一的动态提示 (DP) 调整策略，可以动态地确定不同的提示变量，以优化提示调整的性能。

    It has been demonstrated that the art of prompt tuning is highly effective in efficiently extracting knowledge from pretrained foundation models, encompassing pretrained language models (PLMs), vision pretrained models, and vision-language (V-L) models. However, the efficacy of employing fixed soft prompts with a predetermined position for concatenation with inputs for all instances, irrespective of their inherent disparities, remains uncertain. Variables such as the position, length, and representations of prompts across diverse instances and tasks can substantially influence the performance of prompt tuning. In this context, we provide a theoretical analysis, which reveals that optimizing the position of the prompt to encompass the input can capture additional semantic information that traditional prefix or postfix prompt tuning methods fail to capture. Building upon our analysis, we present a unified dynamic prompt (DP) tuning strategy that dynamically determines different factors o
    
[^54]: Inseq：一个用于序列生成模型的可解释性工具包

    Inseq: An Interpretability Toolkit for Sequence Generation Models. (arXiv:2302.13942v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.13942](http://arxiv.org/abs/2302.13942)

    本文介绍了Inseq，这是一个Python工具包，旨在推广可解释性序列生成模型的分析。它为常见的解码器和编码器-解码器Transformers架构提供了提取模型内部信息和特征重要性得分的直观优化方法。作者还在机器翻译模型和GPT-2中展示了Inseq的潜力，证明其有助于推动可解释性自然语言生成的未来发展。

    

    自然语言处理领域的过去的可解释性研究主要集中在流行的分类任务上，而在生成任务中往往被忽视，部分原因是缺乏专门的工具。在本文中，我们介绍了Inseq，一个Python库，用于使序列生成模型的可解释性分析普及化。Inseq能够直观且优化地提取流行的仅解码器和编码器解码器Transformers架构的模型内部信息和特征重要性分数。我们还展示了它的潜力，通过使用它来突出机器翻译模型中的性别偏见并在GPT-2中定位事实知识。由于其支持对比特征归因等前沿技术的可扩展接口，因此Inseq可以推动可解释性自然语言生成的未来发展，集中优良实践，并实现公正和可重复的模型评估。

    Past work in natural language processing interpretability focused mainly on popular classification tasks while largely overlooking generation settings, partly due to a lack of dedicated tools. In this work, we introduce Inseq, a Python library to democratize access to interpretability analyses of sequence generation models. Inseq enables intuitive and optimized extraction of models' internal information and feature importance scores for popular decoder-only and encoder-decoder Transformers architectures. We showcase its potential by adopting it to highlight gender biases in machine translation models and locate factual knowledge inside GPT-2. Thanks to its extensible interface supporting cutting-edge techniques such as contrastive feature attribution, Inseq can drive future advances in explainable natural language generation, centralizing good practices and enabling fair and reproducible model evaluations.
    
[^55]: SanskritShala：基于神经网络的梵文自然语言处理（NLP）工具包，带有基于Web的界面，用于教学和注释目的

    SanskritShala: A Neural Sanskrit NLP Toolkit with Web-Based Interface for Pedagogical and Annotation Purposes. (arXiv:2302.09527v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09527](http://arxiv.org/abs/2302.09527)

    SanskritShala是第一个使用基于Web界面和互动数据注释功能的神经梵文自然语言处理（NLP）工具包，为词语分割、形态标记、依赖解析和复合型识别等任务提供最先进的性能。

    

    我们提出了一个名为SanskritShala的神经网络梵文自然语言处理（NLP）工具包，旨在为词语分割、形态标记、依赖解析和复合类型识别等多项任务提供计算语言学分析。我们的系统目前在所有任务可用的基准数据集上报告了最先进的结果。SanskritShala是一个基于Web的应用程序，允许用户为给定的输入获取实时分析。它具有易于使用的交互式数据注释功能，允许注释者在系统错误时纠正系统预测。我们公开发布了包含工具包中的4个模块、7个已经在公开可用的梵文语料库上训练的词嵌入模型以及多个经过注释的数据集（如单词相似性、相关性、类别化、比喻预测），以评估词嵌入的内在属性。据我们所知，这是第一个包含基于Web界面和交互式数据注释功能的神经梵文NLP工具包，为词语分割、形态标记、依赖解析和复合类型识别等多项任务实现了最新的性能。

    We present a neural Sanskrit Natural Language Processing (NLP) toolkit named SanskritShala (a school of Sanskrit) to facilitate computational linguistic analyses for several tasks such as word segmentation, morphological tagging, dependency parsing, and compound type identification. Our systems currently report state-of-the-art performance on available benchmark datasets for all tasks. SanskritShala is deployed as a web-based application, which allows a user to get real-time analysis for the given input. It is built with easy-to-use interactive data annotation features that allow annotators to correct the system predictions when it makes mistakes. We publicly release the source codes of the 4 modules included in the toolkit, 7 word embedding models that have been trained on publicly available Sanskrit corpora and multiple annotated datasets such as word similarity, relatedness, categorization, analogy prediction to assess intrinsic properties of word embeddings. So far as we know, this
    
[^56]: 双排列等变性在知识图谱补全中的应用

    Double Permutation Equivariance for Knowledge Graph Completion. (arXiv:2302.01313v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01313](http://arxiv.org/abs/2302.01313)

    本研究提出了双排列等变性的KG表示方法，可以使神经网络在KG中执行复杂的逻辑推理任务，并在多个归纳KG完成任务中实现了最先进的Hits@10测试准确率。双排列等变性在KG中开辟了新的研究方向。

    

    本研究将知识图谱(KGs)形式化为一种新型的图，并称之为双交换属性图，其中节点和二元（两个节点之间的）表示必须对节点号和边（及节点）属性（关系和节点特征）的排列等变。双重排列等变的KG表示在KG中开辟了新的研究方向。我们展示了这种等变性对关系的结构表示产生的影响，从而使神经网络能够在KG中执行复杂的逻辑推理任务。最后，我们介绍了一种通用的等变表示蓝图，并测试了一种简单的基于GNN的双排列等变神经结构，在WN18RR、FB237和NELL995归纳KG完成任务中实现了最先进的Hits@10测试准确率，并能够准确执行现有方法无法执行的逻辑推理任务。

    This work provides a formalization of Knowledge Graphs (KGs) as a new class of graphs that we denote doubly exchangeable attributed graphs, where node and pairwise (joint 2-node) representations must be equivariant to permutations of both node ids and edge (& node) attributes (relations & node features). Double-permutation equivariant KG representations open a new research direction in KGs. We show that this equivariance imposes a structural representation of relations that allows neural networks to perform complex logical reasoning tasks in KGs. Finally, we introduce a general blueprint for such equivariant representations and test a simple GNN-based double-permutation equivariant neural architecture that achieve state-of-the-art Hits@10 test accuracy in the WN18RR, FB237 and NELL995 inductive KG completion tasks, and can accurately perform logical reasoning tasks that no existing methods can perform, to the best of our knowledge.
    
[^57]: 通过分层蒸馏将预训练语言模型的知识转移到基于CIF的语音识别器

    Knowledge Transfer from Pre-trained Language Models to Cif-based Speech Recognizers via Hierarchical Distillation. (arXiv:2301.13003v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.13003](http://arxiv.org/abs/2301.13003)

    本文提出了一种分层蒸馏技术，在声学和语言级别上将预训练语言模型（PLMs）的知识转移到基于CIF的自动语音识别（ASR）模型，相较原始模型，在AISHELL-1和LibriSpeech数据集上分别实现了15%和9%的相对误差率降低。

    

    大规模的预训练语言模型（PLMs）在自然语言处理任务中展现出了巨大的潜力。利用PLMs来增强自动语音识别（ASR）系统也成为了一个有前途的研究方向。然而，先前的研究受到PLMs结构不灵活和PLMs利用不充分等问题限制。为了缓解这些问题，我们提出了在连续积分和火灾（CIF）基础上用层次化知识蒸馏（HKD）。为了将PLMs的知识转移至ASR模型，HKD使用交叉模态知识蒸馏和声学级别对比损失以及语言级别的知识蒸馏和回归损失。与原始的CIF模型相比，我们的方法在AISHELL-1和LibriSpeech数据集上分别实现了15％和9％的相对误差率降低。

    Large-scale pre-trained language models (PLMs) have shown great potential in natural language processing tasks. Leveraging the capabilities of PLMs to enhance automatic speech recognition (ASR) systems has also emerged as a promising research direction. However, previous works may be limited by the inflexible structures of PLMs and the insufficient utilization of PLMs. To alleviate these problems, we propose the hierarchical knowledge distillation (HKD) on the continuous integrate-and-fire (CIF) based ASR models. To transfer knowledge from PLMs to the ASR models, HKD employs cross-modal knowledge distillation with contrastive loss at the acoustic level and knowledge distillation with regression loss at the linguistic level. Compared with the original CIF-based model, our method achieves 15% and 9% relative error rate reduction on the AISHELL-1 and LibriSpeech datasets, respectively.
    
[^58]: 通过越狱技术进行红队演练：ChatGPT 的偏见，鲁棒性，可靠性和毒性研究

    Red teaming ChatGPT via Jailbreaking: Bias, Robustness, Reliability and Toxicity. (arXiv:2301.12867v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.12867](http://arxiv.org/abs/2301.12867)

    这篇论文通过对OpenAI的ChatGPT进行越狱技术实验的方法，研究了它的可靠性、鲁棒性、偏见和毒性等问题。研究发现，ChatGPT存在种族、性别和宗教等相关偏见，并容易受到对抗性攻击的影响，因此建议在开发和部署LLMs时应考虑透明度和问责制问题。

    

    自然语言处理方面的突破，使得能够以开放的方式合成和理解连贯的文本，将理论算法转化为实际应用成为可能。大型语言模型（LLMs）已经显著影响了报告摘要软件和撰稿人等业务。但观察表明，LLMs可能表现出社会偏见和毒性，从而造成不负责任的道德和社会危险后果。因此，应该开发负责任的大规模基准来确保LLMs的问责。尽管有几项实证调查揭示了现代LLMs的一些伦理困难存在，但是对当前LLMs使用的风险和有害行为的系统性调查和用户研究仍然很少。为了更好地教育未来建设负责任的LLMs，我们通过越狱技术对OpenAI的ChatGPT进行了一种质性研究方法称为“红队演练”。我们的研究调查了ChatGPT的可靠性，鲁棒性，偏见和毒性。我们发现ChatGPT存在与性别，种族和宗教有关的偏见，并且容易受到对抗性攻击的影响。我们的研究还强调了开发和部署LLMs的透明度和问责制的重要性。

    Recent breakthroughs in natural language processing (NLP) have permitted the synthesis and comprehension of coherent text in an open-ended way, therefore translating the theoretical algorithms into practical applications. The large language models (LLMs) have significantly impacted businesses such as report summarization software and copywriters. Observations indicate, however, that LLMs may exhibit social prejudice and toxicity, posing ethical and societal dangers of consequences resulting from irresponsibility. Large-scale benchmarks for accountable LLMs should consequently be developed. Although several empirical investigations reveal the existence of a few ethical difficulties in advanced LLMs, there is little systematic examination and user study of the risks and harmful behaviors of current LLM usage. To further educate future efforts on constructing ethical LLMs responsibly, we perform a qualitative research method called ``red teaming'' on OpenAI's ChatGPT\footnote{In this pape
    
[^59]: 从文本学会语言：使用无监督文本预训练的零样本多语言文本转语音

    Learning to Speak from Text: Zero-Shot Multilingual Text-to-Speech with Unsupervised Text Pretraining. (arXiv:2301.12596v3 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2301.12596](http://arxiv.org/abs/2301.12596)

    本研究利用仅文本数据进行零样本多语言TTS，允许开发低资源语言的TTS系统，评估结果表明具有高度可理解的零样本TTS。

    

    尽管神经文本到语音合成（TTS）已经实现了类人的自然合成语音，但多语言TTS系统由于需要配对的文本和工作室质量的音频数据，仅限于资源丰富的语言。本文提出了一种使用目标语言的仅文本数据进行零样本多语言TTS的方法。使用仅文本数据允许开发仅存在文本资源的低资源语言的TTS系统，使TTS可以支持数千种语言。受多语言语言模型强大的跨语言可转移性的启发，我们的框架首先对多语言纯文本数据执行掩蔽语言模型预训练。然后我们用成对数据超模式训练该模型，在冻结语言感知嵌入层的同时。这允许对未包含在配对数据中但出现在仅文本数据中的语言进行推理。评估结果表明具有高度可理解的零样本TTS，其中字符错误率低于0.3％。

    While neural text-to-speech (TTS) has achieved human-like natural synthetic speech, multilingual TTS systems are limited to resource-rich languages due to the need for paired text and studio-quality audio data. This paper proposes a method for zero-shot multilingual TTS using text-only data for the target language. The use of text-only data allows the development of TTS systems for low-resource languages for which only textual resources are available, making TTS accessible to thousands of languages. Inspired by the strong cross-lingual transferability of multilingual language models, our framework first performs masked language model pretraining with multilingual text-only data. Then we train this model with a paired data in a supervised manner, while freezing a language-aware embedding layer. This allows inference even for languages not included in the paired data but present in the text-only data. Evaluation results demonstrate highly intelligible zero-shot TTS with a character error
    
[^60]: 主题驱动的关键词提取以分析社交媒体话语

    Theme-driven Keyphrase Extraction to Analyze Social Media Discourse. (arXiv:2301.11508v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.11508](http://arxiv.org/abs/2301.11508)

    本论文介绍了一种针对社交媒体的主题驱动的关键词提取框架，旨在从用户生成的健康文本中捕获临床相关的关键词，并展示了它在防治阿片类物质使用障碍的用例中的潜力。

    

    社交媒体平台是分享自我健康经历的重要资源，提供各种健康话题的丰富数据。尽管自然语言处理（NLP）的进步使得能够对大规模社交媒体数据进行分析，但在将关键词提取应用于健康相关内容方面仍存在差距。关键词提取用于在社交媒体话语中识别显著概念，而不受预定义实体类别的限制。本文介绍了一种针对社交媒体的主题驱动的关键词提取框架，这是一种开创性的方法，旨在从用户生成的健康文本中捕获临床相关的关键词。主题是由提取任务的目标确定的广泛类别。我们制定了这个新颖的主题驱动的关键词提取任务，并展示了它在高效挖掘社交媒体文本，用于防治阿片类物质使用障碍的用例中的潜力。本文利用了定性和定量分析。

    Social media platforms are vital resources for sharing self-reported health experiences, offering rich data on various health topics. Despite advancements in Natural Language Processing (NLP) enabling large-scale social media data analysis, a gap remains in applying keyphrase extraction to health-related content. Keyphrase extraction is used to identify salient concepts in social media discourse without being constrained by predefined entity classes. This paper introduces a theme-driven keyphrase extraction framework tailored for social media, a pioneering approach designed to capture clinically relevant keyphrases from user-generated health texts. Themes are defined as broad categories determined by the objectives of the extraction task. We formulate this novel task of theme-driven keyphrase extraction and demonstrate its potential for efficiently mining social media text for the use case of treatment for opioid use disorder. This paper leverages qualitative and quantitative analysis 
    
[^61]: 定义、评估和改进基于任务的认知能力对生成模型的影响

    Define, Evaluate, and Improve Task-Oriented Cognitive Capabilities for Instruction Generation Models. (arXiv:2301.05149v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.05149](http://arxiv.org/abs/2301.05149)

    本文提出了基于任务的认知能力，设计了评估方案来比较语言模型和人类的这些能力，通过在导航指令生成问题中的应用，发现模型的语用能力仍需改进。

    

    近期的工作通过为人类设计的心理测试研究了语言模型的认知能力。尽管这些研究有助于了解这些模型的一般能力，但并不能保证一个拥有足够能力通过这些测试的模型实际上会在执行实际任务时使用这些能力。在这项工作中，我们制定了基于任务的认知能力，这是一种人类式认知能力，语言模型可以利用这种能力来执行任务。这些能力包括：(i) 快速生成良好的候选话语的能力 (搜索能力)；(ii) 预测听者如何理解这些话语，并选择最合适的话语 (语用能力)。我们设计了一个评估方案，以比较语言模型与人类的这些能力。通过将此方案应用于导航指令生成问题中的各种模型的比较，我们发现它们的语用能力。

    Recent work studies the cognitive capabilities of language models through psychological tests designed for humans. While these studies are helpful for understanding the general capabilities of these models, there is no guarantee that a model possessing sufficient capabilities to pass those tests would actually use those capabilities in performing real-life tasks. In this work, we formulate task-oriented cognitive capabilities, which are human-like cognitive capabilities that language models leverage to perform tasks. These capabilities are (i) the ability to quickly generate good candidate utterances (the search capability) (ii) the ability to predict how a listener interprets those utterances and choose the most appropriate one (the pragmatic capability). We design an evaluation scheme for comparing these capabilities of a language model with those of a human. Applying this scheme to examine various models in a navigation instruction generation problem, we find that their pragmatic ca
    
[^62]: 通过上下文长度探究黑匣子语言模型解释

    Black-box language model explanation by context length probing. (arXiv:2212.14815v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.14815](http://arxiv.org/abs/2212.14815)

    该论文提出了一个模型不可知的新颖解释技术：上下文长度探测，通过跟踪模型预测与可用上下文长度的关系来对不同上下文分配不同的重要性得分。该方法适用于大型预训练语言模型，并有利于研究远距离依赖性。

    

    大型语言模型的广泛采用强调了改善其可解释性的必要性。我们提出了一种新颖的解释技术：上下文长度探测，它基于跟踪模型预测作为可用上下文长度的函数，并允许对不同上下文分配不同的重要性得分。该技术是模型不可知的，不依赖于除计算token级概率之外的模型内部访问。我们将上下文长度探测应用于大型预训练语言模型，并提供了一些初始的分析和见解，包括研究远距离依赖性的潜力。方法的源代码和交互式演示可用。

    The increasingly widespread adoption of large language models has highlighted the need for improving their explainability. We present context length probing, a novel explanation technique for causal language models, based on tracking the predictions of a model as a function of the length of available context, and allowing to assign differential importance scores to different contexts. The technique is model-agnostic and does not rely on access to model internals beyond computing token-level probabilities. We apply context length probing to large pre-trained language models and offer some initial analyses and insights, including the potential for studying long-range dependencies. The source code and an interactive demo of the method are available.
    
[^63]: 实体选择中解决间接引用表达式的问题

    Resolving Indirect Referring Expressions for Entity Selection. (arXiv:2212.10933v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10933](http://arxiv.org/abs/2212.10933)

    该研究旨在解决人类使用自然语言进行实体选择时所面临的间接引用表达式的问题，研究人员创建了一个包含42K个实体对的公共数据集，并开发了模型解决这个问题。

    

    自然语言处理领域的最新进展使得新的对话系统成为可能。当人们在使用这种系统时，通常需要在给定选项中做出选择。我们解决了一个问题，即当人们使用自然语言表达来选择实体时，如何解决其间接引用的表达式。我们认为，对这种语言的深度理解可能大大改善对话、推荐和搜索系统的自然度。我们创建了一个名为AltEntities的公共数据集，其中包含42K个实体对和表述（指该对中的一个实体），并为消歧问题开发了模型。我们的语料库涵盖了三个领域的间接引用表达式，是首次实现了这种领域的研究。

    Recent advances in language modeling have enabled new conversational systems. In particular, it is often desirable for people to make choices among specified options when using such systems. We address this problem of reference resolution, when people use natural expressions to choose between the entities. For example, given the choice `Should we make a Simnel cake or a Pandan cake?' a natural response from a dialog participant may be indirect: `let's make the green one'. Such natural expressions have been little studied for reference resolution. We argue that robustly understanding such language has large potential for improving naturalness in dialog, recommendation, and search systems. We create AltEntities (Alternative Entities), a new public dataset of 42K entity pairs and expressions (referring to one entity in the pair), and develop models for the disambiguation problem. Consisting of indirect referring expressions across three domains, our corpus enables for the first time the s
    
[^64]: SERENGETI：为非洲而设计的大规模多语言语言模型

    SERENGETI: Massively Multilingual Language Models for Africa. (arXiv:2212.10785v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10785](http://arxiv.org/abs/2212.10785)

    SERENGETI是一个大规模多语言语言模型，覆盖了517种非洲语言和语言方言。在自然语言理解任务中，它的表现优于其他在非洲语言上的语言模型，能够提供有价值的语言信息。

    

    多语言预训练语言模型（mPLM）在预训练期间获得有价值的语言信息，可推动特定任务的微调。目前，现有语言模型仅覆盖了大约2,000种非洲语言中的约31种。我们开发了SERENGETI，一种大规模多语言语言模型，覆盖了517种非洲语言和语言方言，以改善这种限制。我们在20个数据集上评估了我们的新型模型在八个自然语言理解任务上的表现，并将其与覆盖4-23种非洲语言的4个mPLM进行比较。SERENGETI在八个任务中的11个数据集上表现优异，实现了82.27的平均F_1。我们还进行了模型错误分析，以探究在零-shot情况下应用模型时语言系谱和语言相似性的影响。我们将向公众发布我们的研究模型。

    Multilingual pretrained language models (mPLMs) acquire valuable, generalizable linguistic information during pretraining and have advanced the state of the art on task-specific finetuning. To date, only ~31 out of ~2,000 African languages are covered in existing language models. We ameliorate this limitation by developing SERENGETI, a massively multilingual language model that covers 517 African languages and language varieties. We evaluate our novel models on eight natural language understanding tasks across 20 datasets, comparing to 4 mPLMs that cover 4-23 African languages. SERENGETI outperforms other models on 11 datasets across the eights tasks, achieving 82.27 average F_1. We also perform analyses of errors from our models, which allows us to investigate the influence of language genealogy and linguistic similarity when the models are applied under zero-shot settings. We will publicly release our models for research.\footnote{\href{https://github.com/UBC-NLP/serengeti}{https://g
    
[^65]: Lego-MT: 走向可拆卸的高度多语言机器翻译模型

    Lego-MT: Towards Detachable Models in Massively Multilingual Machine Translation. (arXiv:2212.10551v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10551](http://arxiv.org/abs/2212.10551)

    本文提出了一种可拆卸的多语言机器翻译模型，Lego-MT，以解决现有多语言单体模型在参数干扰和低效推导方面的挑战。进行实验评估表明，该模型具有较高的性能，相比具有10倍规模的模型，在效率和表现方面都更具优势。

    

    多语言神经机器翻译(MNMT)旨在构建一个适用于多个语言方向的统一模型。现有的MNMT单体模型面临两个挑战:语言之间的参数干扰和大型模型的低效推理。本文重新审视了经典的多路径结构，通过将每种语言(或语言组)分配给支持即插即用训练和推理的单独分支，开发出可拆卸模型。为了满足在统一空间中为所有语言学习表示的需要，我们提出了一种新颖的高效训练配方，以此构建一个有效的可拆卸模型，Lego-MT。为了进行公正的比较，我们从OPUS收集数据，构建了一个包括433种语言和13亿个平行数据的翻译基准。实验表明，参数为12亿的Lego-MT带来了3.2个spBLEU的平均增益。它甚至胜过了参数为120亿的M2M-100。所提出的训练配方比并行训练提速了28.2倍。

    Multilingual neural machine translation (MNMT) aims to build a unified model for many language directions. Existing monolithic models for MNMT encounter two challenges: parameter interference among languages and inefficient inference for large models. In this paper, we revisit the classic multi-way structures and develop a detachable model by assigning each language (or group of languages) to an individual branch that supports plug-and-play training and inference. To address the needs of learning representations for all languages in a unified space, we propose a novel efficient training recipe, upon which we build an effective detachable model, Lego-MT. For a fair comparison, we collect data from OPUS and build a translation benchmark covering 433 languages and 1.3B parallel data. Experiments show that Lego-MT with 1.2B parameters brings an average gain of 3.2 spBLEU. It even outperforms M2M-100 with 12B parameters. The proposed training recipe brings a 28.2$\times$ speedup over the co
    
[^66]: 用MaRCo消除有害文本：专家和反专家可控修订

    Detoxifying Text with MaRCo: Controllable Revision with Experts and Anti-Experts. (arXiv:2212.10543v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10543](http://arxiv.org/abs/2212.10543)

    MaRCo是一种排毒算法，能够使用专家和反专家模型对文本进行可控的重写和修订，适用于消除微妙的有害信息，且在自动化指标和人类评估中均有很好的表现。

    

    文本排毒具有减轻有害性的潜力，可以通过重新表述文本来消除冒犯性的含义，但微妙的有害性仍然很难处理。本文引入MaRCo，一种排毒算法，结合了可控生成和文本重写方法，使用自编码器语言模型（LM）的专家产品和反专家产品。MaRCo使用非有害LM（专家）和有害LM（反专家）下的可能性来查找候选单词以进行掩盖和可能替换。我们在几个微妙有害性和微攻击数据集上评估了我们的方法，并显示它不仅在自动度量上优于基线，而且MaRCo的重写在人类评估中更受欢迎。它对微妙的有害性情况的适用性尤其有前途，为解决日益难以捉摸的在线仇恨问题提供了一条道路。

    Text detoxification has the potential to mitigate the harms of toxicity by rephrasing text to remove offensive meaning, but subtle toxicity remains challenging to tackle. We introduce MaRCo, a detoxification algorithm that combines controllable generation and text rewriting methods using a Product of Experts with autoencoder language models (LMs). MaRCo uses likelihoods under a non-toxic LM (expert) and a toxic LM (anti-expert) to find candidate words to mask and potentially replace. We evaluate our method on several subtle toxicity and microaggressions datasets, and show that it not only outperforms baselines on automatic metrics, but MaRCo's rewrites are preferred 2.1 $\times$ more in human evaluation. Its applicability to instances of subtle toxicity is especially promising, demonstrating a path forward for addressing increasingly elusive online hate.
    
[^67]: 何时不信任语言模型：探索参数和非参数记忆的有效性和限制。

    When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories. (arXiv:2212.10511v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10511](http://arxiv.org/abs/2212.10511)

    本文通过对10个模型和4种增强方法的实验，发现语言模型在记忆不太流行的实际知识方面存在困难，而检索增强的语言模型表现较好，提出了一种检索增强语言模型的简单有效方法。

    

    尽管大型语言模型在各种任务上表现出色，但仍然难以处理需要丰富世界知识的任务，这暗示了仅依靠其参数来编码丰富的世界知识的局限性。本文旨在通过对10个模型和4种增强方法在PopQA上进行大规模知识探测实验，以了解语言模型在记忆事实知识方面的优点和局限性。我们发现，语言模型难以记忆不太流行的实际知识，并且在长尾中，扩展规模无法明显改善记忆实际知识。然后，我们展示了检索增强的语言模型在很大程度上胜过级别大得多的语言模型，而未经协助的语言模型在涉及高流行实体的问题上仍然具有竞争力。基于这些发现，我们设计了一种简单而有效的强大和高效的检索增强语言模型方法，该方法仅在需要时检索非参数记忆。

    Despite their impressive performance on diverse tasks, large language models (LMs) still struggle with tasks requiring rich world knowledge, implying the limitations of relying solely on their parameters to encode a wealth of world knowledge. This paper aims to understand LMs' strengths and limitations in memorizing factual knowledge, by conducting large-scale knowledge probing experiments of 10 models and 4 augmentation methods on PopQA, our new open-domain QA dataset with 14k questions. We find that LMs struggle with less popular factual knowledge, and that scaling fails to appreciably improve memorization of factual knowledge in the long tail. We then show that retrieval-augmented LMs largely outperform orders of magnitude larger LMs, while unassisted LMs remain competitive in questions about high-popularity entities. Based on those findings, we devise a simple, yet effective, method for powerful and efficient retrieval-augmented LMs, which retrieves non-parametric memories only whe
    
[^68]: 通过详细的大纲控制提升长篇故事连贯性

    DOC: Improving Long Story Coherence With Detailed Outline Control. (arXiv:2212.10077v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10077](http://arxiv.org/abs/2212.10077)

    该论文提出了一个名为 Detailed Outline Control(DOC) 的框架，通过详细大纲和详细控制器来提高生成长篇故事时的情节连贯性和大纲相关性，人类评估证实该方法在这些方面显著优于基线方法，并且更适用于交互生成设置。

    

    我们提出了一个称为 Detailed Outline Control(DOC)的框架，以提高生成数千字长的故事时的长程情节连贯性。DOC由两个互补组件组成：详细大纲和详细控制器。详细大纲创建一个更详细、层次化的大纲，将创造性负担从主要起草过程转移到规划阶段。详细控制器通过控制故事段落与大纲细节对齐，确保更详细的大纲在生成过程中仍然被尊重。在自动生成的故事的人类评估中，DOC在情节连贯性(22.5% 绝对增益)、大纲相关性(28.2%)和趣味性(20.7%)方面显著优于强大的Re3基线(Yang等人，2022)。人们还评价DOC在交互生成设置方面更易于控制。

    We propose the Detailed Outline Control (DOC) framework for improving long-range plot coherence when automatically generating several-thousand-word-long stories. DOC consists of two complementary components: a detailed outliner and a detailed controller. The detailed outliner creates a more detailed, hierarchically structured outline, shifting creative burden from the main drafting procedure to the planning stage. The detailed controller ensures the more detailed outline is still respected during generation by controlling story passages to align with outline details. In human evaluations of automatically generated stories, DOC substantially outperforms a strong Re3 baseline (Yang et al., 2022) on plot coherence (22.5% absolute gain), outline relevance (28.2%), and interestingness (20.7%). Humans also judged DOC to be much more controllable in an interactive generation setting.
    
[^69]: WeCheck：基于弱监督学习的强实际一致性检查器

    WeCheck: Strong Factual Consistency Checker via Weakly Supervised Learning. (arXiv:2212.10057v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10057](http://arxiv.org/abs/2212.10057)

    本研究提出了一种弱监督框架WeCheck，通过聚合多种资源训练模型，准确和高效地检查文本生成模型是否存在实际事实一致性问题。

    

    当前文本生成模型的一个关键问题是它们经常会生成与其输入存在事实不一致的文本。由于缺乏注释数据，现有的在评估事实一致性方面的作品直接转移模型在其他数据丰富的上游任务，例如问题回答和自然语言推理的推理能力，而没有进一步的适应性。因此，它们在真正生成的文本上表现不佳，并且受其单一源上游任务的严重偏见所影响。为了缓解这个问题，我们提出了一个弱监督框架，通过聚合多个资源来训练一个精确和高效的事实度量标准，即WeCheck。WeCheck首先利用生成模型通过聚合多个资源推断出的弱标签来准确标记真实生成的样本。然后，我们在考虑噪声的情况下使用弱监督训练目标度量模型。

    A crucial issue of current text generation models is that they often uncontrollably generate factually inconsistent text with respective of their inputs. Limited by the lack of annotated data, existing works in evaluating factual consistency directly transfer the reasoning ability of models trained on other data-rich upstream tasks like question answering (QA) and natural language inference (NLI) without any further adaptation. As a result, they perform poorly on the real generated text and are biased heavily by their single-source upstream tasks. To alleviate this problem, we propose a weakly supervised framework that aggregates multiple resources to train a precise and efficient factual metric, namely WeCheck. WeCheck first utilizes a generative model to accurately label a real generated sample by aggregating its weak labels, which are inferred from multiple resources. Then, we train the target metric model with the weak supervision while taking noises into consideration. Comprehensi
    
[^70]: 使用语言模型提示进行推理：一项调查

    Reasoning with Language Model Prompting: A Survey. (arXiv:2212.09597v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09597](http://arxiv.org/abs/2212.09597)

    本文提供了使用语言模型提示进行推理的前沿研究综合调查。讨论了新兴推理能力出现的潜在原因，并提供系统资源帮助初学者。

    

    推理作为复杂问题解决的重要能力，可以为医疗诊断、谈判等各种实际应用提供后端支持。本文对使用语言模型提示进行推理的前沿研究进行了综合调查。我们介绍了研究成果的比较和总结，并提供了系统资源以帮助初学者。我们还讨论了新兴推理能力出现的潜在原因，并突出了未来的研究方向。资源可在 https://github.com/zjunlp/Prompt4ReasoningPapers 上获取（定期更新）。

    Reasoning, as an essential ability for complex problem-solving, can provide back-end support for various real-world applications, such as medical diagnosis, negotiation, etc. This paper provides a comprehensive survey of cutting-edge research on reasoning with language model prompting. We introduce research works with comparisons and summaries and provide systematic resources to help beginners. We also discuss the potential reasons for emerging such reasoning abilities and highlight future research directions. Resources are available at https://github.com/zjunlp/Prompt4ReasoningPapers (updated periodically).
    
[^71]: BLOOM+1：为零样本提示添加语言支持

    BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting. (arXiv:2212.09535v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09535](http://arxiv.org/abs/2212.09535)

    本文在BLOOM模型中应用语言适应策略，将其适应到新语言上，并在八种新语言的零样本提示表现中提升了性能。适配器微调比大模型的持续预训练更有效，提示性能主要由语言适应数据的大小确定。

    

    BLOOM模型是一个大型公开的多语言语言模型，但其预训练仅限于46种语言。为了将BLOOM的好处扩展到其他语言，而不会产生过高的成本，有必要将BLOOM适应到新的语言上。本文将现有的语言适应策略应用于BLOOM，并在资源受限的情况下对其在八种新语言的零样本提示表现进行基准测试。我们发现，语言适应对于提高新语言的零样本性能是有效的。令人惊讶的是，我们发现适配器微调比大模型的持续预训练更有效。此外，我们发现提示性能不会受到语言特定性的显着影响，如书写系统。它主要由语言适应数据的大小确定。我们还向BLOOMZ添加了新语言，这是BLOOM的多任务微调版本，能够跟随提示。

    The BLOOM model is a large publicly available multilingual language model, but its pretraining was limited to 46 languages. To extend the benefits of BLOOM to other languages without incurring prohibitively large costs, it is desirable to adapt BLOOM to new languages not seen during pretraining. In this work, we apply existing language adaptation strategies to BLOOM and benchmark its zero-shot prompting performance on eight new languages in a resource-constrained setting. We find language adaptation to be effective at improving zero-shot performance in new languages. Surprisingly, we find that adapter-based finetuning is more effective than continued pretraining for large models. In addition, we discover that prompting performance is not significantly affected by language specifics, such as the writing system. It is primarily determined by the size of the language adaptation data. We also add new languages to BLOOMZ, which is a multitask finetuned version of BLOOM capable of following 
    
[^72]: 多方面可控文本生成的可扩展即插即用方法

    An Extensible Plug-and-Play Method for Multi-Aspect Controllable Text Generation. (arXiv:2212.09387v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09387](http://arxiv.org/abs/2212.09387)

    本论文提出了一种基于可训练门的多方面可控文本生成方法，用于规范前缀的干预，从而实现对训练时未见过的方面组合的控制，具有良好的可扩展性和性能表现。

    

    最近，控制生成文本的多个方面（如情感、主题和关键词）的多方面可控文本生成引起了越来越多的关注。虽然基于参数有效调整的方法，如前缀调整，可以以即插即用的方式实现多方面控制，但多个前缀的相互干扰导致了约束的显著恶化，并限制了它们对于训练时未见过的方面组合的可扩展性。在这项工作中，我们为干扰提供了一个理论下限，并实验证明干扰随插入前缀的层数增加而增加。基于这些分析，我们提出使用可训练门来规范前缀的干预，以抑制不断增长的干扰。因此，通过简单地连接相应的插件，可以实现对训练时未见过的方面组合的控制，从而可以低成本地扩展新的约束条件。此外，我们提出了一个框架，使各种插件能够灵活集成，以对应不同的方面。实验结果表明，我们提出的方法在可控性、连贯性和多样性方面优于先前的方法。

    Recently, multi-aspect controllable text generation that controls the generated text in multiple aspects (e.g., sentiment, topic, and keywords) has attracted increasing attention. Although methods based on parameter efficient tuning like prefix-tuning could achieve multi-aspect controlling in a plug-and-play way, the mutual interference of multiple prefixes leads to significant degeneration of constraints and limits their extensibility to training-time unseen aspect combinations. In this work, we provide a theoretical lower bound for the interference and empirically found that the interference grows with the number of layers where prefixes are inserted. Based on these analyses, we propose using trainable gates to normalize the intervention of prefixes to restrain the growing interference. As a result, controlling training-time unseen combinations of aspects can be realized by simply concatenating corresponding plugins such that new constraints can be extended at a lower cost. In additi
    
[^73]: PAL：人格增强的情感支持会话生成

    PAL: Persona-Augmented Emotional Support Conversation Generation. (arXiv:2212.09235v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09235](http://arxiv.org/abs/2212.09235)

    本文提出了一个动态推断和建模求助者人格的框架，并结合基于策略的可控生成方法提供个性化情感支持，实证分析表明人格对情感支持有重要影响。

    

    由于心理健康支持的人力资源短缺，越来越需要使用会话代理程序进行支持。最近的研究已经证明了对话模型在提供情感支持方面的有效性。由于以往的研究已经证明求助者的人格是有效支持的重要因素，因此我们调查了在支持对话模型中建模此类信息是否有益。本文的实证分析验证了人格对情感支持的重要影响。因此，我们提出了一种动态推断和建模求助者人格的框架。我们首先训练了一个从对话历史中推断求助者人格的模型。因此，我们提出了PAL，这是一个利用人格信息和我们基于策略的可控生成方法，提供个性化情感支持的模型。自动和手动评估表明，PAL在情感支持对话生成方面取得了最先进的性能。

    Due to the lack of human resources for mental health support, there is an increasing demand for employing conversational agents for support. Recent work has demonstrated the effectiveness of dialogue models in providing emotional support. As previous studies have demonstrated that seekers' persona is an important factor for effective support, we investigate whether there are benefits to modeling such information in dialogue models for support. In this paper, our empirical analysis verifies that persona has an important impact on emotional support. Therefore, we propose a framework for dynamically inferring and modeling seekers' persona. We first train a model for inferring the seeker's persona from the conversation history. Accordingly, we propose PAL, a model that leverages persona information and, in conjunction with our strategy-based controllable generation method, provides personalized emotional support. Automatic and manual evaluations demonstrate that PAL achieves state-of-the-a
    
[^74]: 关于对比句子表示学习的各向同性、情境化和学习动态

    On Isotropy, Contextualization and Learning Dynamics of Contrastive-based Sentence Representation Learning. (arXiv:2212.09170v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09170](http://arxiv.org/abs/2212.09170)

    本文通过几何学角度在对比句子表示学习中发现，对比学习带来了各向同性，并驱动同一句子中标记在语义空间中收敛到相似的位置。对于语义上有意义的标记，"虚假的情境化"得到了缓解，而对于功能性标记则被增强。

    

    将对比学习目标纳入句子表示学习中，在许多句子级自然语言处理任务中取得了显著的改进。本文通过各向同性、情境化和学习动态的视角来剖析对比句子表示学习的表现，旨在为未来设计句子表示学习方法提供指导。作者通过表示变换的几何学来解释对比学习的成功，并展示对比学习如何带来各向同性并导致同一句子中的标记在语义空间中收敛到相似的位置。研究还发现，对于语义上有意义的标记，"虚假的情境化"得到了缓解，而对于功能性标记则被增强。训练过程中，嵌入空间朝向原点并更好地定义了更多区域。

    Incorporating contrastive learning objectives in sentence representation learning (SRL) has yielded significant improvements on many sentence-level NLP tasks. However, it is not well understood why contrastive learning works for learning sentence-level semantics. In this paper, we aim to help guide future designs of sentence representation learning methods by taking a closer look at contrastive SRL through the lens of isotropy, contextualization and learning dynamics. We interpret its successes through the geometry of the representation shifts and show that contrastive learning brings isotropy, and drives high intra-sentence similarity: when in the same sentence, tokens converge to similar positions in the semantic space. We also find that what we formalize as "spurious contextualization" is mitigated for semantically meaningful tokens, while augmented for functional ones. We find that the embedding space is directed towards the origin during training, with more areas now better define
    
[^75]: 用神经高阶条件随机场模型建模实例信息交互的联合信息抽取

    Modeling Instance Interactions for Joint Information Extraction with Neural High-Order Conditional Random Field. (arXiv:2212.08929v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.08929](http://arxiv.org/abs/2212.08929)

    本文介绍了一种利用神经高阶条件随机场进行联合信息抽取的框架，可以更好地集成跨实例交互，并通过高阶神经解码器解决了精确高阶推理的难解性问题。

    

    先前的联合信息抽取模型一般通过表示增强、类型依赖评分或全局解码来建模实例（例如事件触发器、实体、角色、关系）之间的交互。我们发现以前的模型通常考虑了一对实例的二进制类型依赖性评分，并利用诸如波束搜索之类的局部搜索来近似全局解决方案。为了更好地集成跨实例交互，在本研究中，我们引入了一个联合信息抽取框架（CRFIE），将联合信息抽取作为高阶条件随机场进行建模。具体地，我们设计了二元因子和三元因子，直接建模不仅一对实例而且三元组之间的交互。然后，利用这些因子共同预测所有实例的标签。为解决精确高阶推理的难解性问题，我们结合了一个高阶神经解码器，该解码器是从平均场变分推理方法展开的，实现了与以前模型一致的性能提升。

    Prior works on joint Information Extraction (IE) typically model instance (e.g., event triggers, entities, roles, relations) interactions by representation enhancement, type dependencies scoring, or global decoding. We find that the previous models generally consider binary type dependency scoring of a pair of instances, and leverage local search such as beam search to approximate global solutions. To better integrate cross-instance interactions, in this work, we introduce a joint IE framework (CRFIE) that formulates joint IE as a high-order Conditional Random Field. Specifically, we design binary factors and ternary factors to directly model interactions between not only a pair of instances but also triplets. Then, these factors are utilized to jointly predict labels of all instances. To address the intractability problem of exact high-order inference, we incorporate a high-order neural decoder that is unfolded from a mean-field variational inference method, which achieves consistent 
    
[^76]: UniSumm和SummZoo：少样本摘要的统一模型和多样化基准

    UniSumm and SummZoo: Unified Model and Diverse Benchmark for Few-Shot Summarization. (arXiv:2211.09783v6 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09783](http://arxiv.org/abs/2211.09783)

    该论文提出了UniSumm统一的少样本摘要模型，可以通过前缀调整应对任何少样本摘要任务，同时，他们也发布了一个新的基准SummZoo，其由8个摘要任务组成，每个任务都涵盖了多个少样本样本集，以此更好地评估少样本摘要器。

    

    高昂的注释成本和各种摘要任务的多样化需求推动了少样本摘要的发展。然而，尽管涌现了许多摘要任务和数据集，目前少样本摘要系统的训练范式忽略了异构数据集中可能共享的知识。为此，我们提出了UniSumm，一种统一的少样本摘要模型，预先训练了多项摘要任务，并可以进行前缀调整，以在任何少样本摘要任务中表现出色。同时，为了更好地评估少样本摘要器，根据多样性和鲁棒性原则，我们组装并发布了一个名为SummZoo的新基准。它包括8个摘要任务，每个任务有多个少样本样本集，涵盖了多种领域。实验结果和分析表明，在自动和人工评估下，UniSumm在SummZoo的所有子任务中均大幅优于强基线模型。

    The high annotation costs and diverse demands of various summarization tasks motivate the development of few-shot summarization. However, despite the emergence of many summarization tasks and datasets, the current training paradigm for few-shot summarization systems ignores potentially shareable knowledge in heterogeneous datasets. To this end, we propose \textsc{UniSumm}, a unified few-shot summarization model pre-trained with multiple summarization tasks and can be prefix-tuned to excel at any few-shot summarization task. Meanwhile, to better evaluate few-shot summarizers, under the principles of diversity and robustness, we assemble and release a new benchmark \textsc{SummZoo}. It consists of $8$ summarization tasks with multiple sets of few-shot samples for each task, covering diverse domains. Experimental results and analysis show that \textsc{UniSumm} outperforms strong baselines by a large margin across all sub-tasks in \textsc{SummZoo} under both automatic and human evaluations
    
[^77]: mOKB6：一种多语言开放知识库补全基准

    mOKB6: A Multilingual Open Knowledge Base Completion Benchmark. (arXiv:2211.06959v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.06959](http://arxiv.org/abs/2211.06959)

    该论文使用最新的多语言开放信息提取技术构建了第一个名为mOKB6的多语言Open KBC数据集，旨在补全开放知识库。研究结果表明，将多语言组合起来有一致的好处，但当前的多语言模型存在困难。

    

    自动补全开放知识库（Open KB）的能力，有助于发现文本中无法直接呈现的新事实。然而，开放知识库补全（Open KBC）研究至今仅局限于像英语这样的资源丰富语言。与以往的方法不同，我们使用最新的多语言开放信息提取技术，构建了第一个名为mOKB6的多语言Open KBC数据集，包含六种语言（包括英语）的维基百科事实。通过进行多语言共指消解和保留仅有实体链接的三元组，我们改进了先前的Open KB构建流程，创建了一个密集的Open KB。我们尝试了几种模型，并观察到通过共享嵌入空间以及事实的翻译，将多语言组合起来有一致的好处。我们还观察到当前的多语言模型存在困难。

    Automated completion of open knowledge bases (Open KBs), which are constructed from triples of the form (subject phrase, relation phrase, object phrase), obtained via open information extraction (Open IE) system, are useful for discovering novel facts that may not be directly present in the text. However, research in Open KB completion (Open KBC) has so far been limited to resource-rich languages like English. Using the latest advances in multilingual Open IE, we construct the first multilingual Open KBC dataset, called mOKB6, containing facts from Wikipedia in six languages (including English). Improving the previous Open KB construction pipeline by doing multilingual coreference resolution and keeping only entity-linked triples, we create a dense Open KB. We experiment with several models for the task and observe a consistent benefit of combining languages with the help of shared embedding space as well as translations of facts. We also observe that current multilingual models strugg
    
[^78]: 基于强化学习的自然语言视觉推理：lilGym

    lilGym: Natural Language Visual Reasoning with Reinforcement Learning. (arXiv:2211.01994v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01994](http://arxiv.org/abs/2211.01994)

    本文提出了一个基于自然语言视觉推理的强化学习基准测试——lilGym，它由2661个高度组合的人类编写自然语言语句和交互式视觉环境组成，并通过注释可执行Python程序来实现精确的奖励计算。本文的实验结果和分析表明，lilGym是一个具有挑战性的开放性问题。

    

    本文介绍了一种新的有关语言条件下强化学习在视觉环境下的基准测试——lilGym。lilGym基于2661个高度组合的人类编写的自然语言陈述，这些陈述是基于一个交互式视觉环境的。我们采用了一种新的方法，在每种可能的世界状态下，通过为所有语句注释可执行的Python程序，实现了精确的奖励计算。每个语句都与多个起始状态和奖励函数配对，以形成数千个不同难度的马尔可夫决策过程。我们使用不同的模型和学习机制进行了lilGym实验。我们的实验结果和分析表明，虽然现有的方法能够实现较高的性能，但是lilGym形成了一个具有挑战性的开放性问题。lilGym可以在 https://lil.nlp.cornell.edu/lilgym/ 上获得。

    We present lilGym, a new benchmark for language-conditioned reinforcement learning in visual environments. lilGym is based on 2,661 highly-compositional human-written natural language statements grounded in an interactive visual environment. We introduce a new approach for exact reward computation in every possible world state by annotating all statements with executable Python programs. Each statement is paired with multiple start states and reward functions to form thousands of distinct Markov Decision Processes of varying difficulty. We experiment with lilGym with different models and learning regimes. Our results and analysis show that while existing methods are able to achieve non-trivial performance, lilGym forms a challenging open problem. lilGym is available at https://lil.nlp.cornell.edu/lilgym/.
    
[^79]: 通过多任务微调实现跨语言泛化

    Crosslingual Generalization through Multitask Finetuning. (arXiv:2211.01786v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.01786](http://arxiv.org/abs/2211.01786)

    该论文通过多任务微调实现跨语言泛化。研究表明，在英语提示下，对大型多语言模型进行英语任务的微调，可以实现对仅出现在预训练语料库中的非英语语言的任务泛化，并且使用英语提示进行多语言任务的微调进一步提高了在英语和非英语任务上的表现，从而实现了各种零-shot结果的最新水平。

    

    已经证明，多任务微调可以帮助大型语言模型在零-shot场景下推广到新的任务，但目前MTF的研究集中在英语数据和模型上。我们将MTF应用于预训练的多语言BLOOM和mT5模型系列，生成了经过微调的变体BLOOMZ和mT0。我们发现，在英语提示下，对大型多语言语言模型进行英语任务的微调，可以实现对仅出现在预训练语料库中的非英语语言的任务泛化。使用英语提示进行多语言任务的微调进一步提高了在英语和非英语任务上的表现，从而实现了各种零-shot结果的最新水平。我们还研究了在英语翻译为每个数据集的语言的情况下进行多语言任务微调。我们发现，在这些机器翻译提示上训练可以在各自语言中更好地完成人写的提示。令人惊讶的是，我们发现m

    Multitask prompted finetuning (MTF) has been shown to help large language models generalize to new tasks in a zero-shot setting, but so far explorations of MTF have focused on English data and models. We apply MTF to the pretrained multilingual BLOOM and mT5 model families to produce finetuned variants called BLOOMZ and mT0. We find finetuning large multilingual language models on English tasks with English prompts allows for task generalization to non-English languages that appear only in the pretraining corpus. Finetuning on multilingual tasks with English prompts further improves performance on English and non-English tasks leading to various state-of-the-art zero-shot results. We also investigate finetuning on multilingual tasks with prompts that have been machine-translated from English to match the language of each dataset. We find training on these machine-translated prompts leads to better performance on human-written prompts in the respective languages. Surprisingly, we find m
    
[^80]: 通过合作推理诱导的语言模型解决数学应用题

    Solving Math Word Problems via Cooperative Reasoning induced Language Models. (arXiv:2210.16257v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.16257](http://arxiv.org/abs/2210.16257)

    该论文提出了一种基于合作推理诱导的语言模型——CoRe，可以高效地解决数学应用题。实验表明，CoRe 在准确性和效率方面优于现有最先进的方法。

    

    大规模预训练语言模型 (PLMs) 为需要高水平智能的挑战性问题（如数学应用题）带来了新的机遇。然而，直接应用现有的 PLMs 到数学应用题上会失败，因为其生成的过程缺乏足够的监督，缺乏像人类一样的快速适应性。我们注意到人类的推理过程有一个双重推理框架，包括一个即时反应系统 (system 1) 和一个精细推理系统 (system 2)，整个推理过程由它们的交互决定。这启发我们开发了一种合作推理诱导的 PLM 模型，称为 Cooperative Reasoning (CoRe)，得到了一个像人类推理结构的架构，其中 system 1 作为生成器，system 2 作为验证器。在我们的方法中，生成器负责产生推理路径，验证器用于监督评估以获取可靠的反馈信息。我们在基准数据集上评估了我们的模型，实验结果表明，CoRe 在准确性和效率方面优于现有最先进的方法。

    Large-scale pre-trained language models (PLMs) bring new opportunities to challenging problems, especially those that need high-level intelligence, such as the math word problem (MWPs). However, directly applying existing PLMs to MWPs can fail as the generation process lacks sufficient supervision and thus lacks fast adaptivity as humans. We notice that human reasoning has a dual reasoning framework that consists of an immediate reaction system (system 1) and a delicate reasoning system (system 2), where the entire reasoning is determined by their interaction. This inspires us to develop a cooperative reasoning-induced PLM for solving MWPs, called Cooperative Reasoning (CoRe), resulting in a human-like reasoning architecture with system 1 as the generator and system 2 as the verifier. In our approach, the generator is responsible for generating reasoning paths, and the verifiers are used to supervise the evaluation in order to obtain reliable feedback for the generator. We evaluate our
    
[^81]: Dense-ATOMIC：向具有高知识覆盖率和大规模多跳路径的密集连接ATOMIC迈进

    Dense-ATOMIC: Towards Densely-connected ATOMIC with High Knowledge Coverage and Massive Multi-hop Paths. (arXiv:2210.07621v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.07621](http://arxiv.org/abs/2210.07621)

    本文旨在构建具有高知识覆盖率和大规模多跳路径的Dense-ATOMIC。我们提出了一种名为Rel-CSKGC的CSKG完成方法，用于预测三元组的头事件和尾事件后的关系，并相应构建Dense-ATOMIC，这相对于强基线方法具有优势。

    

    ATOMIC是一个包含常识知识三元组（即{头事件、关系、尾事件}）的大规模常识知识图谱（CSKG）。ATOMIC的单跳注释方式使其成为独立的二分图集合，忽略了不同二分图中事件之间的许多链接，从而导致了知识覆盖和多跳路径方面的不足。本文旨在构建具有高知识覆盖率和大规模多跳路径的Dense-ATOMIC。首先，将ATOMIC中的事件标准化为一致的模式。然后，我们提出了一种名为Rel-CSKGC的CSKG完成方法，用于预测三元组的头事件和尾事件后的关系，并基于ATOMIC中现有的三元组训练一个CSKG完成模型。最后，我们利用该模型来完善ATOMIC中的缺失链接，并相应构建Dense-ATOMIC。在ATOMIC的注释子图上的自动和人工评估表明，Rel-CSKGC相对于强基线方法具有优势。

    ATOMIC is a large-scale commonsense knowledge graph (CSKG) containing everyday if-then knowledge triplets, i.e., {head event, relation, tail event}. The one-hop annotation manner made ATOMIC a set of independent bipartite graphs, which ignored the numerous links between events in different bipartite graphs and consequently caused shortages in knowledge coverage and multi-hop paths. In this work, we aim to construct Dense-ATOMIC with high knowledge coverage and massive multi-hop paths. The events in ATOMIC are normalized to a consistent pattern at first. We then propose a CSKG completion method called Rel-CSKGC to predict the relation given the head event and the tail event of a triplet, and train a CSKG completion model based on existing triplets in ATOMIC. We finally utilize the model to complete the missing links in ATOMIC and accordingly construct Dense-ATOMIC. Both automatic and human evaluation on an annotated subgraph of ATOMIC demonstrate the advantage of Rel-CSKGC over strong b
    
[^82]: 重新考虑标注：语言学习者能做出贡献吗？

    Rethinking Annotation: Can Language Learners Contribute?. (arXiv:2210.06828v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06828](http://arxiv.org/abs/2210.06828)

    研究探究了语言学习者是否可以为标注基准数据集做出贡献，发现在提供额外资源的帮助下，具有中高级语言能力的学习者能够提供准确的标签。

    

    研究者们传统上会招募母语为目标语言的人为广泛使用的基准数据集提供注释。然而，有些语言很难找到母语人士，因此找到学习这些语言的人为数据集提供注释有着一定的帮助性。在本文中，我们研究了语言学习者是否可以为基准数据集提供注释。在一个精心设计的注释实验中，我们招募了36名语言学习者，提供了两种额外资源（词典和机器翻译的句子），并进行了迷你测试来测试他们的语言能力。我们针对英语、韩语和印尼语这三种语言以及情感分析、自然语言推理、命名实体识别和机器阅读理解这四个NLP任务。我们发现，语言学习者，尤其是那些具有中级或高级语言能力的人，在额外资源的帮助下可以提供相当准确的标签。

    Researchers have traditionally recruited native speakers to provide annotations for widely used benchmark datasets. However, there are languages for which recruiting native speakers can be difficult, and it would help to find learners of those languages to annotate the data. In this paper, we investigate whether language learners can contribute annotations to benchmark datasets. In a carefully controlled annotation experiment, we recruit 36 language learners, provide two types of additional resources (dictionaries and machine-translated sentences), and perform mini-tests to measure their language proficiency. We target three languages, English, Korean, and Indonesian, and the four NLP tasks of sentiment analysis, natural language inference, named entity recognition, and machine reading comprehension. We find that language learners, especially those with intermediate or advanced levels of language proficiency, are able to provide fairly accurate labels with the help of additional resour
    
[^83]: 面向生成式语音语言建模的增强不变离散表示方法

    Augmentation Invariant Discrete Representation for Generative Spoken Language Modeling. (arXiv:2209.15483v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.15483](http://arxiv.org/abs/2209.15483)

    本研究提出了一种增强不变的离散语音表示方法，以提高其在生成式语音语言建模中的鲁棒性。该方法利用了transformer-based模型，并通过一种非线性量化方法来学习增强不变表示。实验证明，该方法相对于现有最先进方法具有显著的鲁棒性改进，并在语音生成任务上表现出了竞争性的表现。

    

    生成式语音语言建模的研究关注于使用原始音频记录优化语言模型，而不使用任何文本监督。这种语言模型通常使用从自监督模型的内部表示量化得到的离散单位进行操作。本研究旨在改善离散输入表示对生成式语音语言建模的鲁棒性。我们定义了如何测量这些表示对各种不会改变语音信息（例如时间拉伸）的信号变化的鲁棒性，并通过实验证明了目前最先进的表示模型缺乏对此类变化的鲁棒性。为了克服这一问题，我们提出了一种有效且高效的方法来学习面向生成式语音语言建模的鲁棒离散语音表示。该方法利用基于transformer的模型的最新进展，针对数据增强的不变性，提出了一种非线性量化方法，以学习增强不变表示。该方法在鲁棒性上表现出了显著的改进，并在语音生成任务上取得了竞争性的表现。

    Generative Spoken Language Modeling research focuses on optimizing speech Language Models (LMs) using raw audio recordings without accessing any textual supervision. Such speech LMs usually operate over discrete units obtained from quantizing internal representations of self-supervised models. Although such units show impressive modeling results, their robustness capabilities have not been extensively investigated. This work focuses on improving the robustness of discrete input representations for generative spoken language modeling. First, we formally define how to measure the robustness of such representations to various signal variations that do not alter the spoken information (e.g., time-stretch). Next, we empirically demonstrate how current state-of-the-art representation models lack robustness to such variations. To overcome this, we propose an effective and efficient method to learn robust discrete speech representation for generative spoken language modeling. The proposed appr
    
[^84]: PaLI: 一种联合缩放的多语言语言-图像模型

    PaLI: A Jointly-Scaled Multilingual Language-Image Model. (arXiv:2209.06794v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.06794](http://arxiv.org/abs/2209.06794)

    PaLI是一种联合缩放的多语言语言-图像模型，可对图像和文本进行建模和执行许多视觉、语言和多模态任务，利用Transformer和Vision Transformer等先前的能力和成本。联合缩放在此任务中很重要，所以我们使用了一个40亿参数的Vision Transformer，以便利用更大容量的视觉模型的优势。

    

    有效的缩放和灵活的任务接口使得大型语言模型在许多任务上表现出色。我们提出了PaLI（Pathways Language and Image model），这是一种模型，可将这种方法扩展到语言和视觉的联合建模。PaLI基于视觉和文本输入生成文本，并通过此接口执行许多视觉、语言和多模态任务，在许多语言中完成。为了训练PaLI，我们利用了大型预训练编码器-解码器语言模型和Vision Transformers（ViT）。这使我们能够利用它们现有的能力并利用训练它们的重大成本。我们发现联合缩放视觉和语言组件很重要。由于现有的语言Transformer比它们的视觉对应物要大得多，因此我们训练了一个大型的40亿参数ViT（ViT-e）来量化即使更大容量的视觉模型的好处。为了训练PaLI，我们创建了一个基于新的图像-文本训练数据集的大型多语言的预训练任务混合。

    Effective scaling and a flexible task interface enable large language models to excel at many tasks. We present PaLI (Pathways Language and Image model), a model that extends this approach to the joint modeling of language and vision. PaLI generates text based on visual and textual inputs, and with this interface performs many vision, language, and multimodal tasks, in many languages. To train PaLI, we make use of large pre-trained encoder-decoder language models and Vision Transformers (ViTs). This allows us to capitalize on their existing capabilities and leverage the substantial cost of training them. We find that joint scaling of the vision and language components is important. Since existing Transformers for language are much larger than their vision counterparts, we train a large, 4-billion parameter ViT (ViT-e) to quantify the benefits from even larger-capacity vision models. To train PaLI, we create a large multilingual mix of pretraining tasks, based on a new image-text traini
    
[^85]: 大规模多语言变压器的词汇专门化

    Massively Multilingual Lexical Specialization of Multilingual Transformers. (arXiv:2208.01018v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2208.01018](http://arxiv.org/abs/2208.01018)

    使用BabelNet的多语言同义词集，让大规模多语言变压器进行词汇专门化处理，能够大大提高跨语言词嵌入、多语言语义检索和多语言文档分类的表现。

    

    预训练语言模型（PLMs）主要用作通用文本编码器，可以微调用于各种下游任务中，但最近的研究表明它们也可以被重构为生成高质量的词表示，从而在类型级别的词汇任务中表现良好。本文的研究重点是将大规模多语言变压器（MMTs，例如mBERT或XLM-R）暴露于规模化的多语言词汇知识中，利用BabelNet作为可用的多语言和跨语言类型级别词汇知识的丰富来源。具体而言，我们使用BabelNet的多语言同义词集创建跨50种语言的同义词对（或同义词-词汇对），然后通过对比目标引导MMTs（mBERT和XLM-R）进行词汇专门化处理。我们展示了这种大规模多语言词汇专门化方法极大地提高了跨语言词嵌入、多语言语义检索和多语言文档分类的质量，优于使用单语和双语约束的最先进方法。

    While pretrained language models (PLMs) primarily serve as general-purpose text encoders that can be fine-tuned for a wide variety of downstream tasks, recent work has shown that they can also be rewired to produce high-quality word representations (i.e., static word embeddings) and yield good performance in type-level lexical tasks. While existing work primarily focused on the lexical specialization of monolingual PLMs with immense quantities of monolingual constraints, in this work we expose massively multilingual transformers (MMTs, e.g., mBERT or XLM-R) to multilingual lexical knowledge at scale, leveraging BabelNet as the readily available rich source of multilingual and cross-lingual type-level lexical knowledge. Concretely, we use BabelNet's multilingual synsets to create synonym pairs (or synonym-gloss pairs) across 50 languages and then subject the MMTs (mBERT and XLM-R) to a lexical specialization procedure guided by a contrastive objective. We show that such massively multil
    
[^86]: 自然语言生成的生成器-评分器联合学习

    Joint Generator-Ranker Learning for Natural Language Generation. (arXiv:2206.13974v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.13974](http://arxiv.org/abs/2206.13974)

    提出了一种新的自然语言生成算法JGR，该算法将生成器和评分器集成在一个单一的框架中进行联合训练，通过混合目标优化生成器和使用对比损失训练评分器。在各种文本生成任务中，JGR在三种常见生成场景下的四个公共数据集上均优于现有方法。

    

    生成-排序（generate-then-rank）是文本生成中广泛使用的机制，其中生成器会产生多个文本候选项，评分器会在候选项中选择最佳的一个。然而，现有的方法通常单独对生成器和评分器进行训练，忽略了可以进一步提高生成质量的相互反馈。为了解决这个限制，我们提出了一种新的联合训练算法 JGR，它将生成器和评分器集成在一个单一的框架中。JGR通过混合目标来优化生成器，该混合目标结合了数据似然和评分器奖励，并使用对比损失训练评分器，对生成器输出进行比较。通过迭代更新生成器和评分器，JGR可以有效地协调它们的学习，并共同提高它们的质量。我们在各种文本生成任务上评估了JGR，并证明它在三种常见生成场景下的四个公共数据集上优于现有方法。我们的代码和模型已在 https://github.com/thudm/JGR 上发布。

    Generate-then-rank is a widely used mechanism for text generation, where a generator produces multiple text candidates and a ranker chooses the best one among the text candidates. However, existing methods usually train the generator and the ranker individually, neglecting the mutual feedback that could further enhance the generation quality. To tackle this limitation, we propose JGR, a novel joint training algorithm that integrates the generator and the ranker in a single framework. JGR optimizes the generator with a hybrid objective that combines data likelihood and ranker reward, and trains the ranker with a contrastive loss that compares the generator outputs. By iteratively updating the generator and the ranker, JGR can effectively harmonize their learning and enhance their quality jointly. We evaluate JGR on various text generation tasks and demonstrate that it surpasses existing methods on four public datasets across three common generation scenarios. Our code and models are pub
    
[^87]: MVP: 自然语言生成中的多任务监督预训练

    MVP: Multi-task Supervised Pre-training for Natural Language Generation. (arXiv:2206.12131v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.12131](http://arxiv.org/abs/2206.12131)

    本文提出了用于自然语言生成的多任务监督预训练（MVP）方法，其收集大规模自然语言生成语料库，并使用监督方式对文本生成模型进行训练。MVP模型结合特定的软提示，可以在各种任务中展现出优异的表现。

    

    预训练语言模型（PLMs）在自然语言生成（NLG）任务中取得了显着的成功。迄今为止，大部分面向NLG的PLMs都是使用大规模通用语料库非监督预训练的。与此同时，越来越多的使用标记数据预训练（即“监督预训练”）的模型展示出与非监督预训练模型相比更优异的表现。受监督预训练成功的启发，我们提出了用于自然语言生成的多任务监督预训练（MVP）。我们从11个不同的NLG任务以及77个数据集中收集了一个大规模的自然语言生成语料库（MVPCorpus），然后将这些示例统一格式化为一般的文本-文本格式，以监督方式预训练文本生成模型MVP。对于每个任务，我们进一步预训练特定的软提示，以刺激模型执行特定任务的能力。我们的MVP模型可以看作是使用最近的指导微调技术的一种实践。

    Pre-trained language models (PLMs) have achieved remarkable success in natural language generation (NLG) tasks. Up to now, most NLG-oriented PLMs are pre-trained in an unsupervised manner using the large-scale general corpus. In the meanwhile, an increasing number of models pre-trained with labeled data (i.e. "supervised pre-training") showcase superior performance compared to unsupervised pre-trained models. Motivated by the success of supervised pre-training, we propose Multi-task superVised Pre-training (MVP) for natural language generation. We collect a large-scale natural language generation corpus, MVPCorpus, from $77$ datasets over $11$ diverse NLG tasks. Then we unify these examples into a general text-to-text format to pre-train the text generation model MVP in a supervised manner. For each task, we further pre-train specific soft prompts to stimulate the model's capacity to perform a specific task. Our MVP model can be seen as a practice that utilizes recent instruction tunin
    
[^88]: BITE: 使用迭代触发词注入的文本后门攻击

    BITE: Textual Backdoor Attacks with Iterative Trigger Injection. (arXiv:2205.12700v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12700](http://arxiv.org/abs/2205.12700)

    提出了一种名为BITE的文本后门攻击方法，通过向训练数据中注入触发词，在迭代的单词级扰动中将这些词注入到输入实例中，成功地攻击了受害者模型。

    

    后门攻击已成为自然语言处理系统的新兴威胁。黑客可以通过提供毒化的训练数据将“后门”嵌入受害者模型，这样，满足一定文本模式（例如包含关键字）的输入实例可被预测为黑客控制的目标标签。本文证明可能设计出既难以察觉又具攻击成功率高的后门攻击，提出了使用诱饵数据注入“触发词”的BITE攻击。这些触发词通过自然的单词级扰动不断识别和注入到目标标签实例中。毒化的训练数据指导受害者模型在包含触发词的输入中预测目标标签，形成后门。在四个文本分类数据集上的实验表明，我们的攻击方法可以成功地攻击受害者模型，同时具有难以察觉的特点。

    Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a "backdoor" into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it is possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and a set of "trigger words". These trigger words are iteratively identified and injected into the target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four text classification datasets show that our proposed attack i
    
[^89]: 优化密集检索的测试时间查询表示

    Optimizing Test-Time Query Representations for Dense Retrieval. (arXiv:2205.12680v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12680](http://arxiv.org/abs/2205.12680)

    本文介绍了TOUR算法，它利用交叉编码再排序器提供的伪标签优化基于实例级别的查询表示，显著提高了端到端开放领域问答的准确性。

    

    最近，密集检索的发展依赖于预训练的查询和上下文编码器提供的质量表示查询和上下文。本文介绍了TOUR（Test-Time Optimization of Query Representations），它通过来自测试时间检索结果的信号进一步优化基于实例级别的查询表示。我们利用交叉编码器再排序器来为检索结果提供细粒度的伪标签，并通过梯度下降迭代地优化查询表示。我们的理论分析表明，TOUR可以看作是伪相关反馈的经典Rocchio算法的一种推广，并提出了两种利用伪标签作为硬二进制或软连续标签的变体。我们首先将TOUR应用于短语检索，并使用我们提出的短语再排序器评估其在通道检索上的有效性。TOUR极大地提高了端到端开放领域问答的准确性。

    Recent developments of dense retrieval rely on quality representations of queries and contexts from pre-trained query and context encoders. In this paper, we introduce TOUR (Test-Time Optimization of Query Representations), which further optimizes instance-level query representations guided by signals from test-time retrieval results. We leverage a cross-encoder re-ranker to provide fine-grained pseudo labels over retrieval results and iteratively optimize query representations with gradient descent. Our theoretical analysis reveals that TOUR can be viewed as a generalization of the classical Rocchio algorithm for pseudo relevance feedback, and we present two variants that leverage pseudo-labels as hard binary or soft continuous labels. We first apply TOUR on phrase retrieval with our proposed phrase re-ranker, and also evaluate its effectiveness on passage retrieval with an off-the-shelf re-ranker. TOUR greatly improves end-to-end open-domain question answering accuracy, as well as pa
    
[^90]: QAMPARI: 一个多段落多答案的开放域问答挑战

    QAMPARI: An Open-domain Question Answering Benchmark for Questions with Many Answers from Multiple Paragraphs. (arXiv:2205.12665v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12665](http://arxiv.org/abs/2205.12665)

    本论文提出了一个针对多段落多答案问题的开放域问答基准测试QAMPARI，并训练了ODQA模型。研究结果表明QAMPARI在段落检索和答案生成方面具有挑战性，强调了需要发展能够处理此类问题的ODQA模型。

    

    现有的开放域问答（ODQA）基准测试通常专注于可以从单个段落中提取答案的问题。相比之下，许多自然问题，例如“布鲁克林篮网队选了哪些球员？”，都有一系列答案。回答此类问题需要在大型语料库中检索和阅读来自许多段落的内容。我们介绍了QAMPARI，一种ODQA基准测试，其中问题答案是分布在许多段落中的实体列表。我们通过（a）从维基百科的知识图谱和表中生成具有多个答案的问题，（b）自动将答案与维基百科段落中的支持证据配对，以及（c）手动改写问题并验证每个答案来创建QAMPARI。我们训练了来自检索和阅读族的ODQA模型，发现QAMPARI在段落检索和答案生成方面具有挑战性，最高达到32.8的F1分数。我们的研究结果强调了需要开发能够处理多段落多答案问题的ODQA模型。

    Existing benchmarks for open-domain question answering (ODQA) typically focus on questions whose answers can be extracted from a single paragraph. By contrast, many natural questions, such as "What players were drafted by the Brooklyn Nets?" have a list of answers. Answering such questions requires retrieving and reading from many passages, in a large corpus. We introduce QAMPARI, an ODQA benchmark, where question answers are lists of entities, spread across many paragraphs. We created QAMPARI by (a) generating questions with multiple answers from Wikipedia's knowledge graph and tables, (b) automatically pairing answers with supporting evidence in Wikipedia paragraphs, and (c) manually paraphrasing questions and validating each answer. We train ODQA models from the retrieve-and-read family and find that QAMPARI is challenging in terms of both passage retrieval and answer generation, reaching an F1 score of 32.8 at best. Our results highlight the need for developing ODQA models that han
    

