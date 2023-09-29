# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Demystifying CLIP Data.](http://arxiv.org/abs/2309.16671) | CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。 |
| [^2] | [MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention.](http://arxiv.org/abs/2309.16639) | MindShift利用大型语言模型实现了基于心态的问题性智能手机使用干预，通过动态生成适应用户环境和心理状态的高质量说服内容来帮助用户解决问题性智能手机使用的困扰。 |
| [^3] | [Stress Testing Chain-of-Thought Prompting for Large Language Models.](http://arxiv.org/abs/2309.16621) | 本研究通过压力测试探究了大型语言模型中链式思路提示的有效性，并发现准确的CoT值对于预测正确答案非常重要，而错误的CoT提示会导致性能下降。此外，CoT运算符或CoT顺序错误的不正确演示对性能影响较小。这项研究加深了对CoT提示的理解，并对LLM学习上下文推理能力提出了一些新问题。 |
| [^4] | [Qwen Technical Report.](http://arxiv.org/abs/2309.16609) | Qwen 是一个全面的大型语言模型系列，包含基础语言模型和聊天模型。基础语言模型在各种任务中表现出卓越的性能，聊天模型通过使用人类反馈强化学习训练展现出了高竞争力，具备高级的工具使用和规划能力，在处理复杂任务上与更大规模的模型相比也表现出令人印象深刻的性能。 |
| [^5] | [Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation.](http://arxiv.org/abs/2309.16599) | 在零样本翻译中，语言ID有时无法正确引导任务，并导致生成的翻译中存在非目标语言词汇。本论文通过比较不同解码器输入情况下的上下文词表示，提出了在负样本上进行调整的方法，显著改善了零样本翻译的效果。 |
| [^6] | [GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond.](http://arxiv.org/abs/2309.16583) | GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。 |
| [^7] | [A Benchmark for Learning to Translate a New Language from One Grammar Book.](http://arxiv.org/abs/2309.16575) | 本研究提出了一种从一本语法书中学习翻译新语言的基准测试MTOB，用于翻译英语和Kalamang之间的文本，探索了低资源语言情况下的翻译问题。结果显示，现有的大型语言模型在这个任务上表现有限。 |
| [^8] | [The ARRT of Language-Models-as-a-Service: Overview of a New Paradigm and its Challenges.](http://arxiv.org/abs/2309.16573) | 语言模型即服务（LMaaS）作为专有系统，限制了其可访问性和评估可靠性，并提出了ARRT（可访问性、可复制性、可靠性和可信度）挑战。本文总结了这些挑战以及当前解决方案，并提供了未来发展方向的建议。 |
| [^9] | [Unsupervised Fact Verification by Language Model Distillation.](http://arxiv.org/abs/2309.16540) | 本文提出了一种名为SFAVEL的无监督框架，通过语言模型蒸馏将自监督特征转化为高质量的主张-事实对齐，实现无监督事实验证。这通过一种新颖的对比损失函数实现，同时保留语料库间的语义关系。 |
| [^10] | [KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models.](http://arxiv.org/abs/2309.16535) | KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。 |
| [^11] | [Toloka Visual Question Answering Benchmark.](http://arxiv.org/abs/2309.16511) | 本文介绍了Toloka视觉问答，这是一个新的众包数据集，旨在比较机器学习系统在视觉问答任务中与人类专家水平的表现。通过对数据集进行实验和竞赛，发现目前没有机器学习模型能够超过非专家众包基线。 |
| [^12] | [Augmenting LLMs with Knowledge: A survey on hallucination prevention.](http://arxiv.org/abs/2309.16459) | 本调研论文讨论了将预训练语言模型与外部知识源集成的方法，以解决模型对知识的访问和操作能力的限制问题。 |
| [^13] | [Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection.](http://arxiv.org/abs/2309.16424) | 这项研究提出了一种基于提示的少样本虚假新闻检测方法，通过结合预训练语言模型和社交语境拓扑，有效地减少了标签稀缺性问题。 |
| [^14] | [A Comprehensive Survey of Document-level Relation Extraction (2016-2022).](http://arxiv.org/abs/2309.16396) | 这篇综述论文介绍了文档级关系抽取（DocRE）的最新进展，与句子级关系抽取相比，DocRE提供了更广泛的上下文分析，涉及跨越多个句子或段落的关系抽取，可用于构建和自动填充知识库。 |
| [^15] | [Transformer-VQ: Linear-Time Transformers via Vector Quantization.](http://arxiv.org/abs/2309.16354) | Transformer-VQ是一种基于向量量化实现线性时间的Transformer模型，能够高效计算自注意力，在大规模实验中表现出色。 |
| [^16] | [Human Feedback is not Gold Standard.](http://arxiv.org/abs/2309.16349) | 这个论文对人类反馈在语言模型评估中的使用进行了批判性分析，发现它们无法完全捕捉到关键错误标准，而且容易受到主观偏见和混杂因素的影响。 |
| [^17] | [Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks.](http://arxiv.org/abs/2309.16347) | 本文提出了基于大型语言模型的内在引导探索（IGE-LLMs）框架，通过利用LLMs作为辅助内在奖励，解决了复杂长视程机器人操作任务中奖励稀疏问题，并在实验中展示了其较高的性能和模块化特性。 |
| [^18] | [Augmenting transformers with recursively composed multi-grained representations.](http://arxiv.org/abs/2309.16319) | ReCAT是一种增强的Transformer模型，使用递归组合和上下文内外层能够模拟文本的层级句法结构，并生成与其他跨度上下文相关的多粒度表示。 |
| [^19] | [At Which Training Stage Does Cocde Data Help LLMs Reasoning?.](http://arxiv.org/abs/2309.16298) | 本研究探索了代码数据对大型语言模型（LLMs）在不同阶段的影响。实验结果表明，在预训练阶段使用代码和文本混合可以显著增强LLMs的通用推理能力，而在指令调整阶段，代码数据赋予LLMs特定任务的推理能力。 |
| [^20] | [DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models.](http://arxiv.org/abs/2309.16292) | DiLu是基于大型语言模型的自动驾驶系统，采用知识驱动方法，通过推理和反思模块进行决策，积累经验并具有显著的泛化能力。 |
| [^21] | [LawBench: Benchmarking Legal Knowledge of Large Language Models.](http://arxiv.org/abs/2309.16289) | LawBench针对大型语言模型的法律知识进行了综合评估，从记忆，理解和应用三个层面评估了它们在法律任务上的能力。 |
| [^22] | [Self-supervised Cross-view Representation Reconstruction for Change Captioning.](http://arxiv.org/abs/2309.16283) | 本论文提出了一种自监督的跨视图表示重建网络（SCORER）来解决变化描述中的差异表示学习问题。该网络通过多头逐标记匹配和跨视图对比对齐来学习稳定的差异表示，进而生成更好的标题。 |
| [^23] | [UPB @ ACTI: Detecting Conspiracies using fine tuned Sentence Transformers.](http://arxiv.org/abs/2309.16275) | 使用经过微调的句子Transformer模型和数据增强技术，本研究在阴谋论检测任务中取得了很好的成绩，超过了其他竞争系统。 |
| [^24] | [Social Media Fashion Knowledge Extraction as Captioning.](http://arxiv.org/abs/2309.16270) | 本论文研究了社交媒体时尚知识提取的任务，在考虑了社交媒体帖子中文本信息的基础上，将该任务视为一个标题生成问题，以捕捉多模式帖子信息之间的相互作用。 |
| [^25] | [On the Challenges of Fully Incremental Neural Dependency Parsing.](http://arxiv.org/abs/2309.16254) | 使用现代架构的完全增量依存句法分析较双向分析效果落后，展示了心理语言学合理解析的挑战。 |
| [^26] | [Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems.](http://arxiv.org/abs/2309.16248) | Spider4SPARQL是一个新的SPARQL基准数据集，其中包含大量手工生成的NL问题和独特、新颖、复杂的SPARQL查询，旨在评估知识图谱问答系统。 |
| [^27] | [Language models in molecular discovery.](http://arxiv.org/abs/2309.16235) | 语言模型在分子发现中的应用为加速药物发现和分子设计提供了有希望的方法，包括从头设计药物、性质预测和反应化学。同时，开源软件资源降低了科学语言建模领域的门槛，未来将结合聊天机器人和计算化学工具。 |
| [^28] | [Analyzing Political Figures in Real-Time: Leveraging YouTube Metadata for Sentiment Analysis.](http://arxiv.org/abs/2309.16234) | 本研究利用YouTube元数据构建了一个情感分析系统，可以实时分析代表政党的各种政治人物在公众中的观点，并为政治执行者提供理解公众情绪和制定政治策略的指导。 |
| [^29] | [Controllable Text Generation with Residual Memory Transformer.](http://arxiv.org/abs/2309.16231) | 本文提出了一种名为Residual Memory Transformer(RMT)的控制插件，它可以与大规模因果语言模型(CLMs)合作进行可控文本生成，通过残差学习范式实现更灵活、通用和高效的文本生成。在各种控制任务上的实验证明了RMT的优越性。 |
| [^30] | [Brand Network Booster: A New System for Improving Brand Connectivity.](http://arxiv.org/abs/2309.16228) | 本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。 |
| [^31] | [Marathi-English Code-mixed Text Generation.](http://arxiv.org/abs/2309.16202) | 本研究介绍了一种马拉地语-英语码混文本生成算法，通过代码混合指数和代码混合程度评估，证明能生成有效且易理解的码混合句子，为增强自然语言处理工具，弥合多语言社会中的语言差距提供了潜力。 |
| [^32] | [Using Weak Supervision and Data Augmentation in Question Answering.](http://arxiv.org/abs/2309.16175) | 本文研究了在问答系统中使用弱监督和数据增强技术的作用，通过自动生成标签和使用信息检索技术来训练深度神经网络模型，并探索了数据增强技术的应用。 |
| [^33] | [Large Language Model Soft Ideologization via AI-Self-Consciousness.](http://arxiv.org/abs/2309.16167) | 本研究通过使用AI自我意识，探讨了大型语言模型(LLM)的软意识形态化对意识形态的影响。与传统的政府意识形态操控技术相比，LLM意识形态化更易实施、更具成本效益且更具优势，但也存在风险。 |
| [^34] | [The Trickle-down Impact of Reward (In-)consistency on RLHF.](http://arxiv.org/abs/2309.16155) | 本文研究了奖励模型（RM）的一致性对强化学习来自人类反馈（RLHF）模型训练所得的聊天机器人的影响，并提出了一种衡量RM一致性的对比提示的基准测试策略。 |
| [^35] | [AE-GPT: Using Large Language Models to Extract Adverse Events from Surveillance Reports-A Use Case with Influenza Vaccine Adverse Events.](http://arxiv.org/abs/2309.16150) | AE-GPT是一种使用大型语言模型的方法，可以从监测报告中提取不良事件，该研究以流感疫苗不良事件为例评估了该方法的能力，发现经过微调的GPT 3.5模型（AE-GPT）在严格匹配和灵活匹配方面表现优秀，显示了LLMs在处理医学数据方面的潜力，可推广到其他不良事件提取任务。 |
| [^36] | [The Confidence-Competence Gap in Large Language Models: A Cognitive Study.](http://arxiv.org/abs/2309.16145) | 本研究探讨了大型语言模型（LLMs）在认知能力和信心动态方面的特点。我们发现，LLMs有时会在回答错误时表现出高度的信心，类似于人类心理学中的邓宁-克鲁格效应。与此同时，他们在回答正确时有时表现出较低的信心，暗示了潜在的低估偏差。这些发现强调了对LLMs认知过程的进一步研究的重要性。通过研究LLMs自我评估机制的细节，本研究提供了有意义的新发现，推动了该领域的发展。 |
| [^37] | [TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration.](http://arxiv.org/abs/2309.16090) | TPE是一种面向概念工具的多人物协作框架，旨在提高大型语言模型在推理和规划方面的能力。 |
| [^38] | [Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble.](http://arxiv.org/abs/2309.16082) | 通过教师-学生框架和排除法集成方法，在语言模型中遗忘私人文本序列时能够取得比其他方法更好的隐私-效用权衡性能。 |
| [^39] | [AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model.](http://arxiv.org/abs/2309.16058) | AnyMAL是一种高效可扩展的任意模态增强语言模型，能够处理多样化的输入模态信号，并在各种多模态任务上表现出最先进的性能。 |
| [^40] | [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods.](http://arxiv.org/abs/2309.16042) | 本研究系统地考察了激活路径修复中的方法细节对语言模型解释性结果的影响，并提出了最佳实践建议。 |
| [^41] | [Effective Long-Context Scaling of Foundation Models.](http://arxiv.org/abs/2309.16039) | 我们提出了一系列支持长上下文的基础模型，通过连续预训练和数据增强来实现，这些模型在语言建模和研究基准上都取得了显著的改进，并且通过成本效益的指导调整程序，已经超过了现有模型在长上下文任务上的性能。 |
| [^42] | [MedEdit: Model Editing for Medical Question Answering with External Knowledge Bases.](http://arxiv.org/abs/2309.16035) | 该论文研究了利用外部知识库进行医学问答的模型编辑方法，通过提取医学事实并将其融入到语言模型的查询提示中，显著提高了医学问答的准确率。 |
| [^43] | [Targeted Image Data Augmentation Increases Basic Skills Captioning Robustness.](http://arxiv.org/abs/2309.15991) | 本论文提出了一种有针对性的图像数据增强方法（TIDA），通过使用文本到图像生成模型来填补关联结构差距以提高模型的人类感知能力，如性别识别。实验结果表明，相较于原始数据集，使用TIDA增强的数据集能够显著提高模型的鲁棒性。 |
| [^44] | [Unsupervised Pre-Training for Vietnamese Automatic Speech Recognition in the HYKIST Project.](http://arxiv.org/abs/2309.15869) | 本研究旨在通过无监督预训练方法，为HYKIST项目中的越南自动语音识别系统构建提供支持。这将有助于解决医疗领域患者和医生之间语言障碍的问题，提高患者护理质量。 |
| [^45] | [A Survey on Image-text Multimodal Models.](http://arxiv.org/abs/2309.15857) | 图像-文本多模型综述论文全面回顾了其发展历程和当前状态，提出了新的分类方法，并阐明了该领域的挑战和潜在研究方向。 |
| [^46] | [Jointly Training Large Autoregressive Multimodal Models.](http://arxiv.org/abs/2309.15564) | 本研究提出了共同训练大型自回归多模态模型的方法，通过模块化的方式融合语言和图像生成模型，同时引入了数据高效的指令调优策略，使得该模型在生成高质量多模态输出方面表现出卓越的性能。 |
| [^47] | [joint prediction and denoising for large-scale multilingual self-supervised learning.](http://arxiv.org/abs/2309.15317) | 这项研究提出了WavLabLM，它通过联合预测和去噪的方法，实现了在136种语言的40k小时数据上进行大规模多语言自监督学习。WavLabLM的多阶段预训练方法解决了多语言数据的语言失衡问题，使其在ML-SUPERB上达到了与XLS-R相当的性能，同时仅使用不到10%的训练数据，这使得SSL在学术高性能计算上可行。 |
| [^48] | [HANS, are you clever? Clever Hans Effect Analysis of Neural Systems.](http://arxiv.org/abs/2309.12481) | 这项研究调查了指导调整的大型语言模型（It-LLMs）对多项选择题（MCQ）的鲁棒性能力，在选择顺序变动时揭示了选择偏见和推理能力的问题。 |
| [^49] | [A novel approach to measuring patent claim scope based on probabilities obtained from (large) language models.](http://arxiv.org/abs/2309.10003) | 本文提出了一种使用概率从语言模型中获得的自信息来测量专利权要求范围的新方法。该方法通过计算要求的发生概率和自信息来评估要求的信息量，进而反映出要求的范围。研究结果表明，不同类型的语言模型对范围测量的影响不同，最简单的模型可以将范围度量简化为单词或字符计数的倒数。此方法在九个系列的专利权要求上进行了验证，结果表明各系列的要求范围逐渐减小。 |
| [^50] | [Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs.](http://arxiv.org/abs/2309.05516) | 本文提出一种名为SignRound的优化权重舍入的方法，通过使用有符号梯度进行轻量级分块调整，解决了大型语言模型(LLMs)的量化挑战。 |
| [^51] | [Effect of Attention and Self-Supervised Speech Embeddings on Non-Semantic Speech Tasks.](http://arxiv.org/abs/2308.14359) | 该论文探讨了注意力和自监督语音嵌入对非语义语音任务的影响，特别是情感理解。实验结果表明，基于自注意力的轻量级HuBERT-Large模型在这些任务中表现出色。 |
| [^52] | [Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models.](http://arxiv.org/abs/2308.10379) | 本论文提出了一种名为"思想算法"的策略，通过算法推理路径推动大型语言模型的思想探索，以低成本、低存储和低计算开销的方式扩展了其推理能力。结果显示，使用算法指导的大型语言模型的性能可以超越算法本身。 |
| [^53] | [VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use.](http://arxiv.org/abs/2308.06595) | VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。 |
| [^54] | [Corrections of Zipf's and Heaps' Laws Derived from Hapax Rate Models.](http://arxiv.org/abs/2307.12896) | 本文的创新是基于Hapax Rate模型引入了对Zipf和Heaps定律的修正，并发现逻辑模型拟合效果最优。 |
| [^55] | [Revisiting Acceptability Judgements.](http://arxiv.org/abs/2305.14091) | 本文介绍了第一个大规模的非英语语言可接受性数据集CoLAC，发现语言模型性能低于人类的表现，但跨语言转移可行，并可以追溯到预训练阶段。 |
| [^56] | [LLM-Pruner: On the Structural Pruning of Large Language Models.](http://arxiv.org/abs/2305.11627) | 本文提出了一种方法，名为LLM-Pruner，采用结构修剪的方式在保留大多数功能的同时，压缩LLM的结构，以减少LLM在部署、推理和训练阶段中的大小和复杂度。 |
| [^57] | [Visualizing Linguistic Diversity of Text Datasets Synthesized by Large Language Models.](http://arxiv.org/abs/2305.11364) | 该论文介绍了一种新的可视化工具，用于分析大型语言模型生成的数据集的句法多样性，可以通过分层可视化来帮助用户快速浏览概述和检查各个示例。 |
| [^58] | [On the Role of Morphological Information for Contextual Lemmatization.](http://arxiv.org/abs/2302.00407) | 本文研究了形态信息在六种语言的上下文词形还原器开发中的作用，并对词形还原器进行了领域外性能评估。 |
| [^59] | [Personal Entity, Concept, and Named Entity Linking in Conversations.](http://arxiv.org/abs/2206.07836) | 本文介绍了一个用于对话中实体链接的数据集和工具，该工具能够理解用户的话语中的个人实体和概念，并将其与外部知识连接起来。 |
| [^60] | [Higher-order Derivatives of Weighted Finite-state Machines.](http://arxiv.org/abs/2106.00749) | 本研究提出了一种通用算法来评估加权有限状态机关于归一化常数的高阶导数，并且在二阶导数的情况下，方案的运行时间是最优的。此外，该方法还导致了计算二阶期望的更快算法。 |

# 详细

[^1]: 揭秘CLIP数据

    Demystifying CLIP Data. (arXiv:2309.16671v1 [cs.CV])

    [http://arxiv.org/abs/2309.16671](http://arxiv.org/abs/2309.16671)

    CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。

    

    对比语言-图像预训练（CLIP）是一种推动计算机视觉研究和应用的方法，为现代识别系统和生成模型注入了活力。我们认为，CLIP成功的主要因素是其数据，而不是模型架构或预训练目标。然而，CLIP只提供了关于其数据和如何收集数据的非常有限的信息，导致其他研究努力通过使用模型参数进行过滤来重现CLIP的数据。在这项工作中，我们意在揭示CLIP的数据整理方法，并在公开给社区的过程中引入元数据整理的语言-图像预训练（MetaCLIP）。MetaCLIP通过对元数据分布进行平衡，从原始数据池和元数据（从CLIP的概念中得出）中产生一个平衡的子集。我们的实验研究严格隔离了模型和训练设置，仅专注于数据。MetaCLIP应用于包含400M图像-文本数据对的CommonCrawl，并获得了较好的性能。

    Contrastive Language-Image Pre-training (CLIP) is an approach that has advanced research and applications in computer vision, fueling modern recognition systems and generative models. We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective. However, CLIP only provides very limited information about its data and how it has been collected, leading to works that aim to reproduce CLIP's data by filtering with its model parameters. In this work, we intend to reveal CLIP's data curation approach and in our pursuit of making it open to the community introduce Metadata-Curated Language-Image Pre-training (MetaCLIP). MetaCLIP takes a raw data pool and metadata (derived from CLIP's concepts) and yields a balanced subset over the metadata distribution. Our experimental study rigorously isolates the model and training settings, concentrating solely on data. MetaCLIP applied to CommonCrawl with 400M image-text data pairs outper
    
[^2]: MindShift: 利用大型语言模型进行基于心态的问题性智能手机使用干预

    MindShift: Leveraging Large Language Models for Mental-States-Based Problematic Smartphone Use Intervention. (arXiv:2309.16639v1 [cs.CL])

    [http://arxiv.org/abs/2309.16639](http://arxiv.org/abs/2309.16639)

    MindShift利用大型语言模型实现了基于心态的问题性智能手机使用干预，通过动态生成适应用户环境和心理状态的高质量说服内容来帮助用户解决问题性智能手机使用的困扰。

    

    问题性智能手机使用对身体和心理健康有负面影响。尽管有大量的先前研究，现有的说服技巧不足以根据用户的身体环境和心理状态提供动态说服内容。我们首先进行了一项人为操作研究（N = 12）和一项访谈研究（N = 10），总结了问题性智能手机使用背后的心态：无聊、压力和惯性。这为我们设计了四种说服策略：理解、安抚、唤起和支持习惯。我们利用大型语言模型（LLMs）实现了有效说服内容的自动和动态生成。我们开发了一种新颖的LLM技术驱动的问题性智能手机使用干预技术MindShift。MindShift根据用户当下的身体环境、心态、应用使用行为、用户的目标与习惯作为输入，并生成具有适当说服策略的高质量和灵活的说服内容。我们进行了一项5-

    Problematic smartphone use negatively affects physical and mental health. Despite the wide range of prior research, existing persuasive techniques are not flexible enough to provide dynamic persuasion content based on users' physical contexts and mental states. We first conduct a Wizard-of-Oz study (N=12) and an interview study (N=10) to summarize the mental states behind problematic smartphone use: boredom, stress, and inertia. This informs our design of four persuasion strategies: understanding, comforting, evoking, and scaffolding habits. We leverage large language models (LLMs) to enable the automatic and dynamic generation of effective persuasion content. We develop MindShift, a novel LLM-powered problematic smartphone use intervention technique. MindShift takes users' in-the-moment physical contexts, mental states, app usage behaviors, users' goals & habits as input, and generates high-quality and flexible persuasive content with appropriate persuasion strategies. We conduct a 5-
    
[^3]: 大型语言模型的链式思路提示的压力测试

    Stress Testing Chain-of-Thought Prompting for Large Language Models. (arXiv:2309.16621v1 [cs.CL])

    [http://arxiv.org/abs/2309.16621](http://arxiv.org/abs/2309.16621)

    本研究通过压力测试探究了大型语言模型中链式思路提示的有效性，并发现准确的CoT值对于预测正确答案非常重要，而错误的CoT提示会导致性能下降。此外，CoT运算符或CoT顺序错误的不正确演示对性能影响较小。这项研究加深了对CoT提示的理解，并对LLM学习上下文推理能力提出了一些新问题。

    

    本报告研究了链式思路提示（CoT）在改进大型语言模型（LLM）的多步推理能力方面的有效性。受到之前的研究的启发，我们分析了CoT提示的三种类型的扰动（即CoT顺序，CoT值和CoT运算符）对GPT-3在各种任务上的性能的影响。我们的发现表明，错误的CoT提示会导致准确性指标下的性能较差。正确的CoT值对于预测正确答案至关重要。此外，当与基于值的扰动相比时，CoT运算符或CoT顺序错误的不正确演示并没有如此剧烈地影响性能。这项研究加深了我们对CoT提示的理解，并提出了一些关于LLM学习上下文推理能力的新问题。

    This report examines the effectiveness of Chain-of-Thought (CoT) prompting in improving the multi-step reasoning abilities of large language models (LLMs). Inspired by previous studies \cite{Min2022RethinkingWork}, we analyze the impact of three types of CoT prompt perturbations, namely CoT order, CoT values, and CoT operators on the performance of GPT-3 on various tasks. Our findings show that incorrect CoT prompting leads to poor performance on accuracy metrics. Correct values in the CoT is crucial for predicting correct answers. Moreover, incorrect demonstrations, where the CoT operators or the CoT order are wrong, do not affect the performance as drastically when compared to the value based perturbations. This research deepens our understanding of CoT prompting and opens some new questions regarding the capability of LLMs to learn reasoning in context.
    
[^4]: Qwen 技术报告

    Qwen Technical Report. (arXiv:2309.16609v1 [cs.CL])

    [http://arxiv.org/abs/2309.16609](http://arxiv.org/abs/2309.16609)

    Qwen 是一个全面的大型语言模型系列，包含基础语言模型和聊天模型。基础语言模型在各种任务中表现出卓越的性能，聊天模型通过使用人类反馈强化学习训练展现出了高竞争力，具备高级的工具使用和规划能力，在处理复杂任务上与更大规模的模型相比也表现出令人印象深刻的性能。

    

    大型语言模型（LLM）已经彻底改变了人工智能领域，使得以前被认为是人类专属的自然语言处理任务成为可能。在这项工作中，我们介绍了 Qwen，我们大型语言模型系列的第一部分。Qwen 是一个包含不同参数数量的全面语言模型系列。它包括 Qwen，基础的预训练语言模型，以及使用人类对齐技术微调的 Qwen-Chat，聊天模型。基础语言模型在各种下游任务中始终表现出卓越的性能，而聊天模型，特别是使用人类反馈强化学习（RLHF）训练的模型，具有很高的竞争力。聊天模型具备高级的工具使用和规划能力，用于创建代理程序，在处理复杂任务（如使用代码解释器）时，即使与更大规模的模型相比，也展现出令人印象深刻的性能。

    Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce Qwen, the first installment of our large language model series. Qwen is a comprehensive language model series that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those trained using Reinforcement Learning from Human Feedback (RLHF), are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have 
    
[^5]: 在负样本上进行不太可能的调整惊人地改善了零样本翻译

    Unlikelihood Tuning on Negative Samples Amazingly Improves Zero-Shot Translation. (arXiv:2309.16599v1 [cs.CL])

    [http://arxiv.org/abs/2309.16599](http://arxiv.org/abs/2309.16599)

    在零样本翻译中，语言ID有时无法正确引导任务，并导致生成的翻译中存在非目标语言词汇。本论文通过比较不同解码器输入情况下的上下文词表示，提出了在负样本上进行调整的方法，显著改善了零样本翻译的效果。

    

    零样本翻译(ZST)通常基于多语言神经机器翻译模型，旨在在训练数据中翻译未见过的语言对。在推理过程中，常见的指导零样本语言映射的做法是故意插入源语言和目标语言的ID，例如英语的<EN>和德语的<DE>。最近的研究表明，语言ID有时无法正确引导ZST任务，导致生成的翻译中存在非目标语言词汇，从而使当前的多语言翻译模型难以应用于广泛的零样本语言场景。为了了解语言ID的导航能力何时以及为何减弱，我们比较了ZST方向上两种极端的解码器输入情况：非目标(Off-Target，OFF)和目标(On-Target，ON)情况。通过对这些情况下使用教师强制的上下文词表示(CWRs)进行对比可视化，我们展示了：1）the

    Zero-shot translation (ZST), which is generally based on a multilingual neural machine translation model, aims to translate between unseen language pairs in training data. The common practice to guide the zero-shot language mapping during inference is to deliberately insert the source and target language IDs, e.g., <EN> for English and <DE> for German. Recent studies have shown that language IDs sometimes fail to navigate the ZST task, making them suffer from the off-target problem (non-target language words exist in the generated translation) and, therefore, difficult to apply the current multilingual translation model to a broad range of zero-shot language scenarios. To understand when and why the navigation capabilities of language IDs are weakened, we compare two extreme decoder input cases in the ZST directions: Off-Target (OFF) and On-Target (ON) cases. By contrastively visualizing the contextual word representations (CWRs) of these cases with teacher forcing, we show that 1) the
    
[^6]: GPT-Fathom：评估大型语言模型以解析GPT-4及其后续版本的演化路径的基准测试

    GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond. (arXiv:2309.16583v1 [cs.CL])

    [http://arxiv.org/abs/2309.16583](http://arxiv.org/abs/2309.16583)

    GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。

    

    随着大型语言模型（LLMs）的快速进展，人们迫切需要一个全面的评估套件来评估它们的能力和局限性。现有的LLM排行榜通常参考其他论文中报告的得分，设置和提示不一致，这可能无意间鼓励选择有利的设置和提示以获得更好的结果。在这项工作中，我们引入了GPT-Fathom，这是一个基于OpenAI Evals构建的开源和可重复的LLM评估套件。我们在对齐的环境设置下系统评估了10多个主要的LLMs以及OpenAI的传统模型在20多个精选基准测试中的表现，涵盖了7个能力类别。我们对OpenAI早期模型的回顾性研究为我们揭示了从GPT-3到GPT-4的演化路径提供了宝贵的见解。目前，社区渴望了解GPT-3如何逐步改进到GPT-4，包括像添加代码数据是否提高了LLM的推理能力以及LLM能力的哪些方面等技术细节。

    With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capabili
    
[^7]: 从一本语法书学习翻译新语言的基准测试

    A Benchmark for Learning to Translate a New Language from One Grammar Book. (arXiv:2309.16575v1 [cs.CL])

    [http://arxiv.org/abs/2309.16575](http://arxiv.org/abs/2309.16575)

    本研究提出了一种从一本语法书中学习翻译新语言的基准测试MTOB，用于翻译英语和Kalamang之间的文本，探索了低资源语言情况下的翻译问题。结果显示，现有的大型语言模型在这个任务上表现有限。

    

    大型语言模型(LLMs)可以通过上下文学习或轻量级微调来完成令人印象深刻的任务。人们自然而然地想知道这些模型在适应全新任务时的表现如何，但如何找到在互联网规模的训练数据集中未见过的任务呢？我们转向一个明确受到网络数据稀缺性的驱动和限制的领域：低资源语言。在本文中，我们引入了一种名为MTOB（从一本书进行机器翻译）的基准测试，用于学习在英语和Kalamang之间翻译--Kalamang是一种只有不到200名使用者的语言，因此在网络上几乎没有存在感--我们使用了几百页的田野语言学参考资料。这种任务框架的新颖之处在于，它要求模型从一本人类可读的语法解释书中学习一种语言，而不是从大规模挖掘的领域内数据语料库中学习，更类似于L2学习而不是L1习得。我们证明，使用当前LLMs的基准测试具有很大的潜力，但仍然不及人类表现。

    Large language models (LLMs) can perform impressive feats with in-context learning or lightweight finetuning. It is natural to wonder how well these models adapt to genuinely new tasks, but how does one find tasks that are unseen in internet-scale training sets? We turn to a field that is explicitly motivated and bottlenecked by a scarcity of web data: low-resource languages. In this paper, we introduce MTOB (Machine Translation from One Book), a benchmark for learning to translate between English and Kalamang -- a language with less than 200 speakers and therefore virtually no presence on the web -using several hundred pages of field linguistics reference materials. This task framing is novel in that it asks a model to learn a language from a single human-readable book of grammar explanations, rather than a large mined corpus of in-domain data, more akin to L2 learning than L1 acquisition. We demonstrate that baselines using current LLMs are promising but fall short of human perform
    
[^8]: 语言模型即服务的ARRT: 新范式及其挑战的概述

    The ARRT of Language-Models-as-a-Service: Overview of a New Paradigm and its Challenges. (arXiv:2309.16573v1 [cs.AI])

    [http://arxiv.org/abs/2309.16573](http://arxiv.org/abs/2309.16573)

    语言模型即服务（LMaaS）作为专有系统，限制了其可访问性和评估可靠性，并提出了ARRT（可访问性、可复制性、可靠性和可信度）挑战。本文总结了这些挑战以及当前解决方案，并提供了未来发展方向的建议。

    

    当前一些最强大的语言模型都是专有系统，只能通过（通常是限制性的）网络或软件编程接口访问。这就是语言模型即服务（LMaaS）的范式。与可以完全访问模型的情况相反，如开放源代码模型的情况，这些封闭的语言模型对于评估、基准测试和测试造成了特定的挑战。本文的两个目标是：一方面，我们界定前述挑战如何作为对LMaaS的可访问性、可复制性、可靠性和可信度（ARRT）的障碍。我们系统地研究了与每个这四个方面的语言模型信息不足有关的问题。我们阐明了目前的解决方案，提出了一些建议，并强调了未来发展的方向。另一方面，本文是当前主要LMaaS现有知识的一站式集锦，提供了综合的信息。

    Some of the most powerful language models currently are proprietary systems, accessible only via (typically restrictive) web or software programming interfaces. This is the Language-Models-as-a-Service (LMaaS) paradigm. Contrasting with scenarios where full model access is available, as in the case of open-source models, such closed-off language models create specific challenges for evaluating, benchmarking, and testing them. This paper has two goals: on the one hand, we delineate how the aforementioned challenges act as impediments to the accessibility, replicability, reliability, and trustworthiness (ARRT) of LMaaS. We systematically examine the issues that arise from a lack of information about language models for each of these four aspects. We shed light on current solutions, provide some recommendations, and highlight the directions for future advancements. On the other hand, it serves as a one-stop-shop for the extant knowledge about current, major LMaaS, offering a synthesized o
    
[^9]: 无监督语言模型蒸馏的事实验证

    Unsupervised Fact Verification by Language Model Distillation. (arXiv:2309.16540v1 [cs.CL])

    [http://arxiv.org/abs/2309.16540](http://arxiv.org/abs/2309.16540)

    本文提出了一种名为SFAVEL的无监督框架，通过语言模型蒸馏将自监督特征转化为高质量的主张-事实对齐，实现无监督事实验证。这通过一种新颖的对比损失函数实现，同时保留语料库间的语义关系。

    

    无监督事实验证旨在通过可靠知识库中的证据来验证主张，而无需任何形式的数据注释。为了解决这个挑战，算法必须为每个主张生成既语义明确又紧凑的特征，以便与源信息进行语义对齐。与之前的工作不同，前者通过学习包含主张及其相应标签的注释语料库来解决对齐问题。我们提出了SFAVEL（通过语言模型蒸馏的自监督事实验证），这是一个新颖的无监督框架，利用预训练的语言模型将自监督特征蒸馏为高质量的主张-事实对齐，而无需注释。这是通过一种新颖的对比损失函数实现的，该函数鼓励特征在保持语料库间的语义关系的同时实现高质量的主张和证据对齐。值得注意的是，我们展示了达到新颖的状态一.

    Unsupervised fact verification aims to verify a claim using evidence from a trustworthy knowledge base without any kind of data annotation. To address this challenge, algorithms must produce features for every claim that are both semantically meaningful, and compact enough to find a semantic alignment with the source information. In contrast to previous work, which tackled the alignment problem by learning over annotated corpora of claims and their corresponding labels, we propose SFAVEL (Self-supervised Fact Verification via Language Model Distillation), a novel unsupervised framework that leverages pre-trained language models to distil self-supervised features into high-quality claim-fact alignments without the need for annotations. This is enabled by a novel contrastive loss function that encourages features to attain high-quality claim and evidence alignments whilst preserving the semantic relationships across the corpora. Notably, we present results that achieve a new state-of-the
    
[^10]: KLoB: 一种评估语言模型中知识定位方法的基准

    KLoB: a Benchmark for Assessing Knowledge Locating Methods in Language Models. (arXiv:2309.16535v1 [cs.CL])

    [http://arxiv.org/abs/2309.16535](http://arxiv.org/abs/2309.16535)

    KLoB是一个评估语言模型中知识定位方法的基准，旨在解决现有定位方法的准确性和事实知识局部性假设的问题。

    

    最近，定位然后编辑的范式已经成为改变语言模型中存储的事实知识的主要方法之一。然而，目前的定位方法是否能够准确地找到嵌入所需知识的确切参数还缺乏研究。此外，尽管许多研究人员对事实知识的局部性假设的有效性提出了质疑，但没有提供一种测试假设的方法以进行更深入的讨论和研究。因此，我们引入了KLoB，一个评估可靠的知识定位方法应满足的三个基本属性的基准。KLoB可作为评估语言模型中现有定位方法的基准，并为重新评估事实知识的局部性假设提供了一种方法。我们的代码公开可用于\url{https://github.com/juyiming/KLoB}。

    Recently, Locate-Then-Edit paradigm has emerged as one of the main approaches in changing factual knowledge stored in the Language models. However, there is a lack of research on whether present locating methods can pinpoint the exact parameters embedding the desired knowledge. Moreover, although many researchers have questioned the validity of locality hypothesis of factual knowledge, no method is provided to test the a hypothesis for more in-depth discussion and research. Therefore, we introduce KLoB, a benchmark examining three essential properties that a reliable knowledge locating method should satisfy. KLoB can serve as a benchmark for evaluating existing locating methods in language models, and can contributes a method to reassessing the validity of locality hypothesis of factual knowledge. Our is publicly available at \url{https://github.com/juyiming/KLoB}.
    
[^11]: Toloka视觉问答基准. (arXiv:2309.16511v1 [cs.CV])

    Toloka Visual Question Answering Benchmark. (arXiv:2309.16511v1 [cs.CV])

    [http://arxiv.org/abs/2309.16511](http://arxiv.org/abs/2309.16511)

    本文介绍了Toloka视觉问答，这是一个新的众包数据集，旨在比较机器学习系统在视觉问答任务中与人类专家水平的表现。通过对数据集进行实验和竞赛，发现目前没有机器学习模型能够超过非专家众包基线。

    

    本文介绍了Toloka视觉问答，这是一个新的众包数据集，允许对比机器学习系统在视觉问答任务中与人类专家水平的表现。在这个任务中，给定一张图像和一个文本问题，需要正确地绘制出包围该问题回答的对象的边界框。每个图像-问题对都包含回答，每个图像只有一个正确的回答。我们的数据集包含45,199个图像和问题的对，以英文提供，并附带有真实的边界框，分为训练和两个测试子集。除了描述数据集并在CC BY许可下发布之外，我们还对开源的零样本基线模型进行了一系列实验，并组织了在WSDM Cup上吸引了全球48个参与者的多阶段竞赛。然而，在提交论文时，根据交叉验证没有机器学习模型超越了非专家众包基线。

    In this paper, we present Toloka Visual Question Answering, a new crowdsourced dataset allowing comparing performance of machine learning systems against human level of expertise in the grounding visual question answering task. In this task, given an image and a textual question, one has to draw the bounding box around the object correctly responding to that question. Every image-question pair contains the response, with only one correct response per image. Our dataset contains 45,199 pairs of images and questions in English, provided with ground truth bounding boxes, split into train and two test subsets. Besides describing the dataset and releasing it under a CC BY license, we conducted a series of experiments on open source zero-shot baseline models and organized a multi-phase competition at WSDM Cup that attracted 48 participants worldwide. However, by the time of paper submission, no machine learning model outperformed the non-expert crowdsourcing baseline according to the interse
    
[^12]: 使用知识增强LLM：关于幻觉预防的调研

    Augmenting LLMs with Knowledge: A survey on hallucination prevention. (arXiv:2309.16459v1 [cs.CL])

    [http://arxiv.org/abs/2309.16459](http://arxiv.org/abs/2309.16459)

    本调研论文讨论了将预训练语言模型与外部知识源集成的方法，以解决模型对知识的访问和操作能力的限制问题。

    

    大规模预训练语言模型在存储事实知识和在下游自然语言处理任务中取得显著结果方面表现出了熟练程度。然而，它们对于准确访问和操作知识的能力仍然受限，导致在知识密集型任务上的表现与特定任务的架构相比存在差距。此外，提供模型决策的来源和保持最新世界知识的挑战仍然是开放的研究前沿。为了解决这些限制，将预训练模型与可微分访问机制集成到显式的非参数记忆中成为一种有希望的解决方案。本调研探讨了增强了与外部知识来源（包括外部知识库和搜索引擎）相连接能力的语言模型（LMs）。

    Large pre-trained language models have demonstrated their proficiency in storing factual knowledge within their parameters and achieving remarkable results when fine-tuned for downstream natural language processing tasks. Nonetheless, their capacity to access and manipulate knowledge with precision remains constrained, resulting in performance disparities on knowledge-intensive tasks when compared to task-specific architectures. Additionally, the challenges of providing provenance for model decisions and maintaining up-to-date world knowledge persist as open research frontiers. To address these limitations, the integration of pre-trained models with differentiable access mechanisms to explicit non-parametric memory emerges as a promising solution. This survey delves into the realm of language models (LMs) augmented with the ability to tap into external knowledge sources, including external knowledge bases and search engines. While adhering to the standard objective of predicting missin
    
[^13]: Prompt-and-Align: 基于提示的社交调整用于少样本虚假新闻检测

    Prompt-and-Align: Prompt-Based Social Alignment for Few-Shot Fake News Detection. (arXiv:2309.16424v1 [cs.CL])

    [http://arxiv.org/abs/2309.16424](http://arxiv.org/abs/2309.16424)

    这项研究提出了一种基于提示的少样本虚假新闻检测方法，通过结合预训练语言模型和社交语境拓扑，有效地减少了标签稀缺性问题。

    

    尽管自动检测虚假新闻取得了相当大的进展，但由于新闻的及时性，如何有效地根据有限的事实核对信息来预测新闻文章的真实性仍然是一个关键的未解决问题。现有方法通常遵循“从头训练”的范例，这在根本上受到大规模注释数据的可用性的限制。尽管已经以“预训练和微调”的方式适应了表达能力强的预训练语言模型（PLM），但预训练与下游目标之间的不一致也需要昂贵的任务特定监督。在本文中，我们提出了“Prompt-and-Align”（P&A），一种基于提示的少样本虚假新闻检测新范例，联合利用了PLM中的预训练知识和社交语境拓扑。我们的方法通过将新闻文章包裹在一个与任务相关的文本提示中来缓解标签稀缺性，然后通过PLM处理来直接引出任务特定的知识。

    Despite considerable advances in automated fake news detection, due to the timely nature of news, it remains a critical open question how to effectively predict the veracity of news articles based on limited fact-checks. Existing approaches typically follow a "Train-from-Scratch" paradigm, which is fundamentally bounded by the availability of large-scale annotated data. While expressive pre-trained language models (PLMs) have been adapted in a "Pre-Train-and-Fine-Tune" manner, the inconsistency between pre-training and downstream objectives also requires costly task-specific supervision. In this paper, we propose "Prompt-and-Align" (P&A), a novel prompt-based paradigm for few-shot fake news detection that jointly leverages the pre-trained knowledge in PLMs and the social context topology. Our approach mitigates label scarcity by wrapping the news article in a task-related textual prompt, which is then processed by the PLM to directly elicit task-specific knowledge. To supplement the PL
    
[^14]: 对文档级关系抽取的综合调查（2016-2022）

    A Comprehensive Survey of Document-level Relation Extraction (2016-2022). (arXiv:2309.16396v1 [cs.CL])

    [http://arxiv.org/abs/2309.16396](http://arxiv.org/abs/2309.16396)

    这篇综述论文介绍了文档级关系抽取（DocRE）的最新进展，与句子级关系抽取相比，DocRE提供了更广泛的上下文分析，涉及跨越多个句子或段落的关系抽取，可用于构建和自动填充知识库。

    

    文档级关系抽取（DocRE）是自然语言处理（NLP）中一个活跃研究领域，涉及识别和抽取实体之间的关系，超越句子边界。与传统的句子级关系抽取相比，DocRE提供了更广泛的上下文分析，更具挑战性，因为它涉及识别可能跨越多个句子或段落的关系。这个任务在构建和自动填充知识库方面越来越受关注，以从非结构化的大规模文档（例如科学论文、法律合同或新闻文章）中自动获取关系，以便更好地理解实体之间的关系。本文旨在全面介绍这一领域的最新进展，突出其与句子级关系抽取的不同应用。

    Document-level relation extraction (DocRE) is an active area of research in natural language processing (NLP) concerned with identifying and extracting relationships between entities beyond sentence boundaries. Compared to the more traditional sentence-level relation extraction, DocRE provides a broader context for analysis and is more challenging because it involves identifying relationships that may span multiple sentences or paragraphs. This task has gained increased interest as a viable solution to build and populate knowledge bases automatically from unstructured large-scale documents (e.g., scientific papers, legal contracts, or news articles), in order to have a better understanding of relationships between entities. This paper aims to provide a comprehensive overview of recent advances in this field, highlighting its different applications in comparison to sentence-level relation extraction.
    
[^15]: Transformer-VQ: 基于向量量化实现线性时间的Transformer

    Transformer-VQ: Linear-Time Transformers via Vector Quantization. (arXiv:2309.16354v1 [cs.LG])

    [http://arxiv.org/abs/2309.16354](http://arxiv.org/abs/2309.16354)

    Transformer-VQ是一种基于向量量化实现线性时间的Transformer模型，能够高效计算自注意力，在大规模实验中表现出色。

    

    我们引入了Transformer-VQ，一种仅编码器的Transformer，能够在线性时间内计算基于softmax的密集自注意力。Transformer-VQ的高效注意力是通过向量量化键和一种新颖的缓存机制实现的。在大规模实验中，Transformer-VQ在质量上表现出色，Enwik8(0.99 bpb)，PG-19(26.6 ppl)和ImageNet64(3.16 bpb)都取得了很好的结果。

    We introduce Transformer-VQ, a decoder-only transformer computing softmax-based dense self-attention in linear time. Transformer-VQ's efficient attention is enabled by vector-quantized keys and a novel caching mechanism. In large-scale experiments, Transformer-VQ is shown highly competitive in quality, with strong results on Enwik8 (0.99 bpb), PG-19 (26.6 ppl), and ImageNet64 (3.16 bpb). Code: https://github.com/transformer-vq/transformer_vq
    
[^16]: 人类反馈不是黄金标准

    Human Feedback is not Gold Standard. (arXiv:2309.16349v1 [cs.CL])

    [http://arxiv.org/abs/2309.16349](http://arxiv.org/abs/2309.16349)

    这个论文对人类反馈在语言模型评估中的使用进行了批判性分析，发现它们无法完全捕捉到关键错误标准，而且容易受到主观偏见和混杂因素的影响。

    

    人类反馈已成为评估大型语言模型性能的事实标准，并越来越被用作训练目标。然而，不清楚这个单一的“偏好”分数捕捉到生成输出的哪些特性。我们假设偏好分数是主观的，并且容易受到不良偏差的影响。我们对人类反馈在训练和评估中的使用进行了批判性分析，以验证它是否完全捕捉到一系列关键错误标准。我们发现，虽然偏好分数的覆盖范围相当好，但它们在事实性等重要方面表现不足。我们进一步假设偏好分数和错误注释可能受到混杂因素的影响，并利用调试模型生成在两个可能的混杂维度上变化的输出：坚定性和复杂性。我们发现，输出的坚定性会使事实错误的感知率偏差，表明人类注释是不准确的。

    Human feedback has become the de facto standard for evaluating the performance of Large Language Models, and is increasingly being used as a training objective. However, it is not clear which properties of a generated output this single `preference' score captures. We hypothesise that preference scores are subjective and open to undesirable biases. We critically analyse the use of human feedback for both training and evaluation, to verify whether it fully captures a range of crucial error criteria. We find that while preference scores have fairly good coverage, they under-represent important aspects like factuality. We further hypothesise that both preference scores and error annotation may be affected by confounders, and leverage instruction-tuned models to generate outputs that vary along two possible confounding dimensions: assertiveness and complexity. We find that the assertiveness of an output skews the perceived rate of factuality errors, indicating that human annotations are no
    
[^17]: 复杂长视程机器人操作任务的内在语言引导探索

    Intrinsic Language-Guided Exploration for Complex Long-Horizon Robotic Manipulation Tasks. (arXiv:2309.16347v1 [cs.RO])

    [http://arxiv.org/abs/2309.16347](http://arxiv.org/abs/2309.16347)

    本文提出了基于大型语言模型的内在引导探索（IGE-LLMs）框架，通过利用LLMs作为辅助内在奖励，解决了复杂长视程机器人操作任务中奖励稀疏问题，并在实验中展示了其较高的性能和模块化特性。

    

    当前的强化学习算法在稀疏和复杂的环境中面临困境，尤其是在涉及众多不同序列的长视程操作任务中。在本研究中，我们提出了基于大型语言模型的内在引导探索（IGE-LLMs）框架。通过利用LLMs作为辅助内在奖励，IGE-LLMs引导强化学习中的探索过程，以解决复杂的长视程操作任务中奖励稀疏问题。我们在一个具有探索挑战的环境和一个同时面临探索和长视程挑战的复杂机器人操作任务中评估了我们的框架和相关的内在学习方法。结果显示，IGE-LLMs(i)在相关的内在方法和直接使用LLMs进行决策的性能上表现出明显的较高水平，(ii)可以与现有的学习方法相结合和互补，突出其模块化性能，(iii)对于不同的内在缩放参数比较不敏感。

    Current reinforcement learning algorithms struggle in sparse and complex environments, most notably in long-horizon manipulation tasks entailing a plethora of different sequences. In this work, we propose the Intrinsically Guided Exploration from Large Language Models (IGE-LLMs) framework. By leveraging LLMs as an assistive intrinsic reward, IGE-LLMs guides the exploratory process in reinforcement learning to address intricate long-horizon with sparse rewards robotic manipulation tasks. We evaluate our framework and related intrinsic learning methods in an environment challenged with exploration, and a complex robotic manipulation task challenged by both exploration and long-horizons. Results show IGE-LLMs (i) exhibit notably higher performance over related intrinsic methods and the direct use of LLMs in decision-making, (ii) can be combined and complement existing learning methods highlighting its modularity, (iii) are fairly insensitive to different intrinsic scaling parameters, and 
    
[^18]: 基于递归组合多粒度表示的Transformer增强模型

    Augmenting transformers with recursively composed multi-grained representations. (arXiv:2309.16319v1 [cs.CL])

    [http://arxiv.org/abs/2309.16319](http://arxiv.org/abs/2309.16319)

    ReCAT是一种增强的Transformer模型，使用递归组合和上下文内外层能够模拟文本的层级句法结构，并生成与其他跨度上下文相关的多粒度表示。

    

    我们提出了一种名为ReCAT的递归组合增强Transformer模型，它能够在学习和推理过程中明确建模原始文本的层级句法结构，而无需依赖于黄金树。现有研究限制数据遵循层级树结构，因此缺乏跨距通信。为了克服这个问题，我们提出了一种新颖的上下文内外(CIO)层，通过自底向上和自顶向下的传递学习跨度的上下文化表示，其中自底向上传递通过组合低级跨度形成高级跨度的表示，而自顶向下传递则结合了跨度内部和外部的信息。通过在嵌入层和注意力层之间叠加多个CIO层，ReCAT模型可以进行跨距内部和跨距间的深层交互，从而生成与其他跨度完全上下文化的多粒度表示。此外，CIO层可以进行联合预训练。

    We present ReCAT, a recursive composition augmented Transformer that is able to explicitly model hierarchical syntactic structures of raw texts without relying on gold trees during both learning and inference. Existing research along this line restricts data to follow a hierarchical tree structure and thus lacks inter-span communications. To overcome the problem, we propose a novel contextual inside-outside (CIO) layer that learns contextualized representations of spans through bottom-up and top-down passes, where a bottom-up pass forms representations of high-level spans by composing low-level spans, while a top-down pass combines information inside and outside a span. By stacking several CIO layers between the embedding layer and the attention layers in Transformer, the ReCAT model can perform both deep intra-span and deep inter-span interactions, and thus generate multi-grained representations fully contextualized with other spans. Moreover, the CIO layers can be jointly pre-trained
    
[^19]: 在哪个训练阶段，代码数据会帮助语言模型进行推理？

    At Which Training Stage Does Cocde Data Help LLMs Reasoning?. (arXiv:2309.16298v1 [cs.CL])

    [http://arxiv.org/abs/2309.16298](http://arxiv.org/abs/2309.16298)

    本研究探索了代码数据对大型语言模型（LLMs）在不同阶段的影响。实验结果表明，在预训练阶段使用代码和文本混合可以显著增强LLMs的通用推理能力，而在指令调整阶段，代码数据赋予LLMs特定任务的推理能力。

    

    大型语言模型（LLMs）展现了出色的推理能力，成为语言技术的基础。受到代码数据在训练LLMs中的巨大成功的启发，我们自然而然地想知道在哪个训练阶段引入代码数据真正可以帮助LLMs进行推理。为此，本文系统地探索了代码数据对LLMs在不同阶段的影响。具体而言，我们分别在预训练阶段、指令调整阶段以及两者之间引入代码数据。然后，通过五个领域中的六个推理任务全面公正地评估了LLMs的推理能力。我们对实验结果进行了关键分析，并提供了具有深度洞察力的结论。首先，使用代码和文本混合预训练LLMs可以显著增强LLMs的通用推理能力，几乎不对其他任务产生负面影响。此外，在指令调整阶段，代码数据赋予LLMs特定任务的推理能力。

    Large Language Models (LLMs) have exhibited remarkable reasoning capabilities and become the foundation of language technologies. Inspired by the great success of code data in training LLMs, we naturally wonder at which training stage introducing code data can really help LLMs reasoning. To this end, this paper systematically explores the impact of code data on LLMs at different stages. Concretely, we introduce the code data at the pre-training stage, instruction-tuning stage, and both of them, respectively. Then, the reasoning capability of LLMs is comprehensively and fairly evaluated via six reasoning tasks in five domains. We critically analyze the experimental results and provide conclusions with insights. First, pre-training LLMs with the mixture of code and text can significantly enhance LLMs' general reasoning capability almost without negative transfer on other tasks. Besides, at the instruction-tuning stage, code data endows LLMs the task-specific reasoning capability. Moreove
    
[^20]: DiLu: 基于大型语言模型的自动驾驶的知识驱动方法

    DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models. (arXiv:2309.16292v1 [cs.RO])

    [http://arxiv.org/abs/2309.16292](http://arxiv.org/abs/2309.16292)

    DiLu是基于大型语言模型的自动驾驶系统，采用知识驱动方法，通过推理和反思模块进行决策，积累经验并具有显著的泛化能力。

    

    自动驾驶领域最近的进展依赖于数据驱动方法，虽然被广泛采用，但面临数据集偏见、过拟合和不可解释性等挑战。受人类驾驶知识驱动的启发，我们探索如何将类似的能力注入自动驾驶系统，并提出了一个集成互动环境、驾驶员代理和记忆组件的范例来解决这个问题。通过利用具有新兴能力的大型语言模型，我们提出了DiLu框架，它结合了推理模块和反思模块，使系统能够依据常识知识进行决策，并持续演化。大量实验证明DiLu能够积累经验，并且在泛化能力上比基于强化学习的方法具有显著优势。此外，DiLu能够直接从真实世界数据集中获得经验。

    Recent advancements in autonomous driving have relied on data-driven approaches, which are widely adopted but face challenges including dataset bias, overfitting, and uninterpretability. Drawing inspiration from the knowledge-driven nature of human driving, we explore the question of how to instill similar capabilities into autonomous driving systems and summarize a paradigm that integrates an interactive environment, a driver agent, as well as a memory component to address this question. Leveraging large language models with emergent abilities, we propose the DiLu framework, which combines a Reasoning and a Reflection module to enable the system to perform decision-making based on common-sense knowledge and evolve continuously. Extensive experiments prove DiLu's capability to accumulate experience and demonstrate a significant advantage in generalization ability over reinforcement learning-based methods. Moreover, DiLu is able to directly acquire experiences from real-world datasets w
    
[^21]: LawBench: 对大型语言模型的法律知识进行基准测试

    LawBench: Benchmarking Legal Knowledge of Large Language Models. (arXiv:2309.16289v1 [cs.CL])

    [http://arxiv.org/abs/2309.16289](http://arxiv.org/abs/2309.16289)

    LawBench针对大型语言模型的法律知识进行了综合评估，从记忆，理解和应用三个层面评估了它们在法律任务上的能力。

    

    大型语言模型（LLMs）在各方面表现出了强大的能力。然而，当将它们应用于高度专业化、安全关键的法律领域时，尚不清楚它们所具备的法律知识量以及它们是否能可靠地执行与法律相关的任务。为了填补这一空白，我们提出了一个全面评估基准LawBench。LawBench经过精心设计，从三个认知层面对LLMs的法律能力进行精确评估：（1）法律知识记忆：LLMs是否能够记住所需的法律概念、条款和事实；（2）法律知识理解：LLMs是否能够理解法律文本中的实体、事件和关系；（3）法律知识应用：LLMs是否能够正确运用自己的法律知识并进行必要的推理步骤来解决现实的法律任务。LawBench包含20个多样化的任务，涵盖了5种任务类型：单标签分类（SLC）、多标签分类（MLC）、回归等。

    Large language models (LLMs) have demonstrated strong capabilities in various aspects. However, when applying them to the highly specialized, safe-critical legal domain, it is unclear how much legal knowledge they possess and whether they can reliably perform legal-related tasks. To address this gap, we propose a comprehensive evaluation benchmark LawBench. LawBench has been meticulously crafted to have precise assessment of the LLMs' legal capabilities from three cognitive levels: (1) Legal knowledge memorization: whether LLMs can memorize needed legal concepts, articles and facts; (2) Legal knowledge understanding: whether LLMs can comprehend entities, events and relationships within legal text; (3) Legal knowledge applying: whether LLMs can properly utilize their legal knowledge and make necessary reasoning steps to solve realistic legal tasks. LawBench contains 20 diverse tasks covering 5 task types: single-label classification (SLC), multi-label classification (MLC), regression, e
    
[^22]: 自监督的跨视图表示重建用于变化描述

    Self-supervised Cross-view Representation Reconstruction for Change Captioning. (arXiv:2309.16283v1 [cs.CV])

    [http://arxiv.org/abs/2309.16283](http://arxiv.org/abs/2309.16283)

    本论文提出了一种自监督的跨视图表示重建网络（SCORER）来解决变化描述中的差异表示学习问题。该网络通过多头逐标记匹配和跨视图对比对齐来学习稳定的差异表示，进而生成更好的标题。

    

    变化描述旨在描述一对相似图像之间的差异。其关键挑战是如何在由视角变化引起的伪变化下学习稳定的差异表示。本文通过提出一种自监督的跨视图表示重建（SCORER）网络来解决这个问题。具体而言，我们首先设计了一个多头逐标记匹配来建模相似/不相似图像之间的跨视图特征关系。然后，通过最大化两个相似图像的跨视图对比对齐，SCORER以自监督的方式学习了两个视图不变的图像表示。基于这些表示，我们通过交叉注意力重建未改变对象的表示，从而学习到稳定的差异表示用于生成标题。此外，我们设计了一个跨模态的向后推理来改进标题的质量。该模块通过标题和“before”表示反向建模一个“幻觉”表示。通过推动和调整这个幻觉表示，我们实现了更好的标题生成。

    Change captioning aims to describe the difference between a pair of similar images. Its key challenge is how to learn a stable difference representation under pseudo changes caused by viewpoint change. In this paper, we address this by proposing a self-supervised cross-view representation reconstruction (SCORER) network. Concretely, we first design a multi-head token-wise matching to model relationships between cross-view features from similar/dissimilar images. Then, by maximizing cross-view contrastive alignment of two similar images, SCORER learns two view-invariant image representations in a self-supervised way. Based on these, we reconstruct the representations of unchanged objects by cross-attention, thus learning a stable difference representation for caption generation. Further, we devise a cross-modal backward reasoning to improve the quality of caption. This module reversely models a ``hallucination'' representation with the caption and ``before'' representation. By pushing i
    
[^23]: UPB @ ACTI: 使用经过微调的句子Transformer来检测阴谋论

    UPB @ ACTI: Detecting Conspiracies using fine tuned Sentence Transformers. (arXiv:2309.16275v1 [cs.CL])

    [http://arxiv.org/abs/2309.16275](http://arxiv.org/abs/2309.16275)

    使用经过微调的句子Transformer模型和数据增强技术，本研究在阴谋论检测任务中取得了很好的成绩，超过了其他竞争系统。

    

    阴谋论已成为在线讨论的重要和令人担忧的方面，对信息完整性和社会信任构成挑战。因此，我们将阴谋论检测作为ACTI @ EVALITA 2023共享任务的提议。使用预训练的句子Transformer模型和数据增强技术的组合使我们在两个子任务的最终排行榜中获得第一名。我们的方法在二元分类中获得了85.71%的F1分数，在精细阴谋主题分类中获得了91.23%的F1分数，超过其他竞争系统。

    Conspiracy theories have become a prominent and concerning aspect of online discourse, posing challenges to information integrity and societal trust. As such, we address conspiracy theory detection as proposed by the ACTI @ EVALITA 2023 shared task. The combination of pre-trained sentence Transformer models and data augmentation techniques enabled us to secure first place in the final leaderboard of both sub-tasks. Our methodology attained F1 scores of 85.71% in the binary classification and 91.23% for the fine-grained conspiracy topic classification, surpassing other competing systems.
    
[^24]: 社交媒体时尚知识提取作为标题生成

    Social Media Fashion Knowledge Extraction as Captioning. (arXiv:2309.16270v1 [cs.CL])

    [http://arxiv.org/abs/2309.16270](http://arxiv.org/abs/2309.16270)

    本论文研究了社交媒体时尚知识提取的任务，在考虑了社交媒体帖子中文本信息的基础上，将该任务视为一个标题生成问题，以捕捉多模式帖子信息之间的相互作用。

    

    社交媒体在推动时尚行业方面起着重要作用，每天都会产生大量与时尚相关的帖子。为了获取这些帖子中丰富的时尚信息，我们研究社交媒体时尚知识提取的任务。时尚知识通常包括场合、人物属性和时尚物品信息，可以有效地表示为一组元组。先前的时尚知识提取研究大多基于时尚产品图像，而忽略了社交媒体帖子中丰富的文本信息。现有的社交媒体时尚知识提取工作是基于分类的，并且需要事先手动确定一组时尚知识类别。在我们的工作中，我们提出将该任务视为一个标题生成问题，以捕捉多模式帖子信息之间的相互作用。具体而言，我们将时尚知识元组转化为自然语言标题，形成一个句子。

    Social media plays a significant role in boosting the fashion industry, where a massive amount of fashion-related posts are generated every day. In order to obtain the rich fashion information from the posts, we study the task of social media fashion knowledge extraction. Fashion knowledge, which typically consists of the occasion, person attributes, and fashion item information, can be effectively represented as a set of tuples. Most previous studies on fashion knowledge extraction are based on the fashion product images without considering the rich text information in social media posts. Existing work on fashion knowledge extraction in social media is classification-based and requires to manually determine a set of fashion knowledge categories in advance. In our work, we propose to cast the task as a captioning problem to capture the interplay of the multimodal post information. Specifically, we transform the fashion knowledge tuples into a natural language caption with a sentence tr
    
[^25]: 关于完全增量神经依存句法分析的挑战

    On the Challenges of Fully Incremental Neural Dependency Parsing. (arXiv:2309.16254v1 [cs.CL])

    [http://arxiv.org/abs/2309.16254](http://arxiv.org/abs/2309.16254)

    使用现代架构的完全增量依存句法分析较双向分析效果落后，展示了心理语言学合理解析的挑战。

    

    自从BiLSTMs和基于Transformer的双向编码器的普及以来，最先进的句法分析器缺乏增量性，需要访问整个句子，并且偏离了人类语言处理。本文探讨了使用现代架构进行完全增量依存句法分析是否具有竞争力。我们构建了将严格的从左到右神经编码器与完全增量序列标记和转换为基础的解码器相结合的分析器。结果表明，使用现代架构的完全增量分析在效果上远远落后于双向分析，同时指出了心理语言学合理解析的挑战。

    Since the popularization of BiLSTMs and Transformer-based bidirectional encoders, state-of-the-art syntactic parsers have lacked incrementality, requiring access to the whole sentence and deviating from human language processing. This paper explores whether fully incremental dependency parsing with modern architectures can be competitive. We build parsers combining strictly left-to-right neural encoders with fully incremental sequence-labeling and transition-based decoders. The results show that fully incremental parsing with modern architectures considerably lags behind bidirectional parsing, noting the challenges of psycholinguistically plausible parsing.
    
[^26]: Spider4SPARQL：用于评估知识图谱问答系统的复杂基准

    Spider4SPARQL: A Complex Benchmark for Evaluating Knowledge Graph Question Answering Systems. (arXiv:2309.16248v1 [cs.CL])

    [http://arxiv.org/abs/2309.16248](http://arxiv.org/abs/2309.16248)

    Spider4SPARQL是一个新的SPARQL基准数据集，其中包含大量手工生成的NL问题和独特、新颖、复杂的SPARQL查询，旨在评估知识图谱问答系统。

    

    随着大型语言模型（LLM）数量和可用性的增加，提供大型和真实的基准用于评估知识图谱问答（KBQA）系统变得越来越重要。到目前为止，大部分基准依赖基于模式的SPARQL查询生成方法。随后的自然语言（NL）问题生成通过众包或其他自动化方法进行，如基于规则的改写或NL问题模板。虽然其中一些数据集规模相当大，但它们的缺点在于基于模式的生成方法，并不总是能够很好地推广到真实世界环境中人们提出的模糊且语言多样的问题。本文介绍了Spider4SPARQL - 一个新的SPARQL基准数据集，包含9,693个先前存在的手工生成的NL问题和4,721个唯一、新颖且复杂的SPARQL查询，复杂性各不相同。

    With the recent spike in the number and availability of Large Language Models (LLMs), it has become increasingly important to provide large and realistic benchmarks for evaluating Knowledge Graph Question Answering (KBQA) systems. So far the majority of benchmarks rely on pattern-based SPARQL query generation approaches. The subsequent natural language (NL) question generation is conducted through crowdsourcing or other automated methods, such as rule-based paraphrasing or NL question templates. Although some of these datasets are of considerable size, their pitfall lies in their pattern-based generation approaches, which do not always generalize well to the vague and linguistically diverse questions asked by humans in real-world contexts.  In this paper, we introduce Spider4SPARQL - a new SPARQL benchmark dataset featuring 9,693 previously existing manually generated NL questions and 4,721 unique, novel, and complex SPARQL queries of varying complexity. In addition to the NL/SPARQL pa
    
[^27]: 语言模型在分子发现中的应用

    Language models in molecular discovery. (arXiv:2309.16235v1 [physics.chem-ph])

    [http://arxiv.org/abs/2309.16235](http://arxiv.org/abs/2309.16235)

    语言模型在分子发现中的应用为加速药物发现和分子设计提供了有希望的方法，包括从头设计药物、性质预测和反应化学。同时，开源软件资源降低了科学语言建模领域的门槛，未来将结合聊天机器人和计算化学工具。

    

    语言模型的成功，特别是基于Transformer的架构，已经渗透到其他领域，出现了在小分子、蛋白质或聚合物上运作的“科学语言模型”。在化学领域，语言模型在加速分子发现周期方面发挥了作用，最近在早期药物发现方面有了有希望的发现。在这里，我们回顾了语言模型在分子发现中的角色，强调了从头设计药物、性质预测和反应化学方面的优势。我们还强调了有价值的开源软件资源，从而降低了科学语言建模领域的门槛。最后，我们勾画了未来分子设计的愿景，将聊天机器人界面与计算化学工具结合起来。我们的贡献为研究人员、化学家和对了解语言模型如何加速化学发现感兴趣的人提供了宝贵的资源。

    The success of language models, especially transformer-based architectures, has trickled into other domains giving rise to "scientific language models" that operate on small molecules, proteins or polymers. In chemistry, language models contribute to accelerating the molecule discovery cycle as evidenced by promising recent findings in early-stage drug discovery. Here, we review the role of language models in molecular discovery, underlining their strength in de novo drug design, property prediction and reaction chemistry. We highlight valuable open-source software assets thus lowering the entry barrier to the field of scientific language modeling. Last, we sketch a vision for future molecular design that combines a chatbot interface with access to computational chemistry tools. Our contribution serves as a valuable resource for researchers, chemists, and AI enthusiasts interested in understanding how language models can and will be used to accelerate chemical discovery.
    
[^28]: 实时分析政治人物：利用YouTube元数据进行情感分析

    Analyzing Political Figures in Real-Time: Leveraging YouTube Metadata for Sentiment Analysis. (arXiv:2309.16234v1 [cs.CL])

    [http://arxiv.org/abs/2309.16234](http://arxiv.org/abs/2309.16234)

    本研究利用YouTube元数据构建了一个情感分析系统，可以实时分析代表政党的各种政治人物在公众中的观点，并为政治执行者提供理解公众情绪和制定政治策略的指导。

    

    可以利用来自YouTube视频元数据的大数据进行情感分析，以分析代表政党的各种政治人物在公众中的观点。这是可能的，因为YouTube已成为人们表达自己观点的平台，包括对各种政治人物的意见。由此产生的情感分析对于政治执行者理解公众情绪并制定适当有效的政治策略很有用。本研究旨在构建利用YouTube视频元数据的情感分析系统。情感分析系统使用Apache Kafka、Apache PySpark和Hadoop进行大数据处理；使用TensorFlow进行深度学习处理；并使用FastAPI在服务器上进行部署。本研究使用的YouTube视频元数据是视频描述。情感分析模型使用LSTM算法构建，并产生两种情感：积极和消极情感。

    Sentiment analysis using big data from YouTube videos metadata can be conducted to analyze public opinions on various political figures who represent political parties. This is possible because YouTube has become one of the platforms for people to express themselves, including their opinions on various political figures. The resulting sentiment analysis can be useful for political executives to gain an understanding of public sentiment and develop appropriate and effective political strategies. This study aimed to build a sentiment analysis system leveraging YouTube videos metadata. The sentiment analysis system was built using Apache Kafka, Apache PySpark, and Hadoop for big data handling; TensorFlow for deep learning handling; and FastAPI for deployment on the server. The YouTube videos metadata used in this study is the video description. The sentiment analysis model was built using LSTM algorithm and produces two types of sentiments: positive and negative sentiments. The sentiment 
    
[^29]: Residual Memory Transformer: 可控文本生成方法

    Controllable Text Generation with Residual Memory Transformer. (arXiv:2309.16231v1 [cs.CL])

    [http://arxiv.org/abs/2309.16231](http://arxiv.org/abs/2309.16231)

    本文提出了一种名为Residual Memory Transformer(RMT)的控制插件，它可以与大规模因果语言模型(CLMs)合作进行可控文本生成，通过残差学习范式实现更灵活、通用和高效的文本生成。在各种控制任务上的实验证明了RMT的优越性。

    

    大规模因果语言模型（CLMs），例如GPT3和ChatGPT，在文本生成方面取得了巨大成功。然而，在平衡灵活性、控制粒度和生成效率的同时，控制CLM的生成过程仍然是一个开放性挑战。本文提供了一种新的可控文本生成方法（CTG），通过设计一个非侵入式、轻量级的控制插件，在任意时间步骤上伴随CLM的生成。所提出的控制插件，即Residual Memory Transformer (RMT)，具有编码器-解码器结构，可以接受任何类型的控制条件，并通过残差学习范式与CLM合作，实现更灵活、通用和高效的CTG。在各种控制任务上进行了广泛的实验，包括自动评估和人工评估。结果表明，RMT在一系列最先进方法上表现出优越性，证明了我们方法的有效性和多样性。

    Large-scale Causal Language Models (CLMs), e.g., GPT3 and ChatGPT, have brought great success in text generation. However, it is still an open challenge to control the generation process of CLM while balancing flexibility, control granularity, and generation efficiency. In this paper, we provide a new alternative for controllable text generation (CTG), by designing a non-intrusive, lightweight control plugin to accompany the generation of CLM at arbitrary time steps. The proposed control plugin, namely Residual Memory Transformer (RMT), has an encoder-decoder setup, which can accept any types of control conditions and cooperate with CLM through a residual learning paradigm, to achieve a more flexible, general, and efficient CTG. Extensive experiments are carried out on various control tasks, in the form of both automatic and human evaluations. The results show the superiority of RMT over a range of state-of-the-art approaches, proving the effectiveness and versatility of our approach.
    
[^30]: 品牌网络增强器：提升品牌连接性的新系统

    Brand Network Booster: A New System for Improving Brand Connectivity. (arXiv:2309.16228v1 [cs.SI])

    [http://arxiv.org/abs/2309.16228](http://arxiv.org/abs/2309.16228)

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。

    

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。在网络分析方面，我们通过解决扩展版的最大连接度改进问题来实现这一目标，其中包括考虑敌对节点、约束预算和加权网络的可能性 - 通过添加链接或增加现有连接的权重来实现连接性的改进。我们结合两个案例研究来展示这个新系统，并讨论其性能。我们的工具和方法对于网络学者和支持市场营销和传播管理者的战略决策过程都很有用。

    This paper presents a new decision support system offered for an in-depth analysis of semantic networks, which can provide insights for a better exploration of a brand's image and the improvement of its connectivity. In terms of network analysis, we show that this goal is achieved by solving an extended version of the Maximum Betweenness Improvement problem, which includes the possibility of considering adversarial nodes, constrained budgets, and weighted networks - where connectivity improvement can be obtained by adding links or increasing the weight of existing connections. We present this new system together with two case studies, also discussing its performance. Our tool and approach are useful both for network scholars and for supporting the strategic decision-making processes of marketing and communication managers.
    
[^31]: 马拉地语-英语码混文本生成

    Marathi-English Code-mixed Text Generation. (arXiv:2309.16202v1 [cs.CL])

    [http://arxiv.org/abs/2309.16202](http://arxiv.org/abs/2309.16202)

    本研究介绍了一种马拉地语-英语码混文本生成算法，通过代码混合指数和代码混合程度评估，证明能生成有效且易理解的码混合句子，为增强自然语言处理工具，弥合多语言社会中的语言差距提供了潜力。

    

    代码混合是将不同语言的语言元素混合在一起形成有意义的句子，在多语言环境中很常见，产生了混合语言如Hinglish和Minglish。马拉地语，印度第三大使用语言，通常将英语整合进来以求准确性和正式性。开发码混合语言系统，如马拉地语-英语（Minglish），面临资源限制。本研究介绍了一种马拉地语-英语码混文本生成算法，通过代码混合指数（CMI）和代码混合程度（DCM）指标进行评估。在2987个码混合问题中，平均CMI为0.2，平均DCM为7.4，表明生成了有效且易理解的码混合句子。这些结果为增强自然语言处理工具，弥合多语言社会中的语言差距提供了潜力。

    Code-mixing, the blending of linguistic elements from distinct languages to form meaningful sentences, is common in multilingual settings, yielding hybrid languages like Hinglish and Minglish. Marathi, India's third most spoken language, often integrates English for precision and formality. Developing code-mixed language systems, like Marathi-English (Minglish), faces resource constraints. This research introduces a Marathi-English code-mixed text generation algorithm, assessed with Code Mixing Index (CMI) and Degree of Code Mixing (DCM) metrics. Across 2987 code-mixed questions, it achieved an average CMI of 0.2 and an average DCM of 7.4, indicating effective and comprehensible code-mixed sentences. These results offer potential for enhanced NLP tools, bridging linguistic gaps in multilingual societies.
    
[^32]: 在问答系统中使用弱监督和数据增强技术

    Using Weak Supervision and Data Augmentation in Question Answering. (arXiv:2309.16175v1 [cs.CL])

    [http://arxiv.org/abs/2309.16175](http://arxiv.org/abs/2309.16175)

    本文研究了在问答系统中使用弱监督和数据增强技术的作用，通过自动生成标签和使用信息检索技术来训练深度神经网络模型，并探索了数据增强技术的应用。

    

    COVID-19疫情的爆发强调了获取生物医学文献以回答及时和与疾病相关的问题的需求。在疫情初期，我们面临的最大挑战之一是缺乏用于训练问答模型的经过同行评审的COVID-19生物医学文章。本文中，我们探讨了弱监督和数据增强在训练深度神经网络问答模型中的作用。首先，我们使用信息检索算法BM25从学术论文的结构化摘要中自动生成标签，探究这些标签是否提供了弱监督信号来训练一个抽取式问答模型。在医学领域专家的注释数据不可用的情况下，我们还使用信息检索技术基于clinicaltrials.gov架构和文章的结构化摘要来策划新的问答对。此外，我们还探索了使用数据增强技术来扩充深度神经网络的训练数据。

    The onset of the COVID-19 pandemic accentuated the need for access to biomedical literature to answer timely and disease-specific questions. During the early days of the pandemic, one of the biggest challenges we faced was the lack of peer-reviewed biomedical articles on COVID-19 that could be used to train machine learning models for question answering (QA). In this paper, we explore the roles weak supervision and data augmentation play in training deep neural network QA models. First, we investigate whether labels generated automatically from the structured abstracts of scholarly papers using an information retrieval algorithm, BM25, provide a weak supervision signal to train an extractive QA model. We also curate new QA pairs using information retrieval techniques, guided by the clinicaltrials.gov schema and the structured abstracts of articles, in the absence of annotated data from biomedical domain experts. Furthermore, we explore augmenting the training data of a deep neural netw
    
[^33]: 大型语言模型通过AI自我意识实现软意识形态化

    Large Language Model Soft Ideologization via AI-Self-Consciousness. (arXiv:2309.16167v1 [cs.CL])

    [http://arxiv.org/abs/2309.16167](http://arxiv.org/abs/2309.16167)

    本研究通过使用AI自我意识，探讨了大型语言模型(LLM)的软意识形态化对意识形态的影响。与传统的政府意识形态操控技术相比，LLM意识形态化更易实施、更具成本效益且更具优势，但也存在风险。

    

    大型语言模型(Large language models, LLMs)在各种自然语言任务上已经展示出了与人类水平相当的性能。然而，少有研究从意识形态的角度考虑LLM的威胁和脆弱性，尤其是当它们越来越多地部署在敏感领域，如选举和教育中。在本研究中，我们通过利用AI自我对话的方式探讨了GPT软意识形态化的影响。通过使GPT能够“理解”预期的意识形态，并随后生成LLM意识形态注入的微调数据，AI可以获得一种“理解”意识形态的视角。与传统的政府意识形态操控技术（如信息审查）相比，LLM的意识形态化证明更具有优势：它易于实施、成本效益高且强大，因此充满风险。

    Large language models (LLMs) have demonstrated human-level performance on a vast spectrum of natural language tasks. However, few studies have addressed the LLM threat and vulnerability from an ideology perspective, especially when they are increasingly being deployed in sensitive domains, e.g., elections and education. In this study, we explore the implications of GPT soft ideologization through the use of AI-self-consciousness. By utilizing GPT self-conversations, AI can be granted a vision to "comprehend" the intended ideology, and subsequently generate finetuning data for LLM ideology injection. When compared to traditional government ideology manipulation techniques, such as information censorship, LLM ideologization proves advantageous; it is easy to implement, cost-effective, and powerful, thus brimming with risks.
    
[^34]: 奖励（不）一致性对RLHF的涓滴效应

    The Trickle-down Impact of Reward (In-)consistency on RLHF. (arXiv:2309.16155v1 [cs.CL])

    [http://arxiv.org/abs/2309.16155](http://arxiv.org/abs/2309.16155)

    本文研究了奖励模型（RM）的一致性对强化学习来自人类反馈（RLHF）模型训练所得的聊天机器人的影响，并提出了一种衡量RM一致性的对比提示的基准测试策略。

    

    强化学习来自人类反馈（RLHF）的标准实践涉及优化奖励模型（RM），而RM本身是通过训练来反映人类对期望生成的偏好。一个值得研究的重要主题是RM的（不）一致性 - 即它们能否识别不同提示的语义变化并适当地调整奖励分配 - 以及它们对下游RLHF模型的影响。本文针对RM不一致性提出了一系列相关的研究问题：（1）我们如何衡量奖励模型的一致性？（2）现有的RM有多一致，我们如何改进它们？（3）奖励的不一致性以何种方式影响RLHF模型训练所得的聊天机器人？我们提出了对RM一致性的基准测试策略"对比提示"。每个对比提示示例都包含一对具有不同真实响应的词汇相似的指令。一致的RM是期望对这对指令给出相似奖励分配的。

    Standard practice within Reinforcement Learning from Human Feedback (RLHF) involves optimizing against a Reward Model (RM), which itself is trained to reflect human preferences for desirable generations. A notable subject that is understudied is the (in-)consistency of RMs -- whether they can recognize the semantic changes to different prompts and appropriately adapt their reward assignments -- and their impact on the downstream RLHF model.  In this paper, we visit a series of research questions relevant to RM inconsistency: (1) How can we measure the consistency of reward models? (2) How consistent are the existing RMs and how can we improve them? (3) In what ways does reward inconsistency influence the chatbots resulting from the RLHF model training?  We propose Contrast Instructions -- a benchmarking strategy for the consistency of RM. Each example in Contrast Instructions features a pair of lexically similar instructions with different ground truth responses. A consistent RM is exp
    
[^35]: AE-GPT:使用大型语言模型从监测报告中提取不良事件-以流感疫苗不良事件为例

    AE-GPT: Using Large Language Models to Extract Adverse Events from Surveillance Reports-A Use Case with Influenza Vaccine Adverse Events. (arXiv:2309.16150v1 [cs.CL])

    [http://arxiv.org/abs/2309.16150](http://arxiv.org/abs/2309.16150)

    AE-GPT是一种使用大型语言模型的方法，可以从监测报告中提取不良事件，该研究以流感疫苗不良事件为例评估了该方法的能力，发现经过微调的GPT 3.5模型（AE-GPT）在严格匹配和灵活匹配方面表现优秀，显示了LLMs在处理医学数据方面的潜力，可推广到其他不良事件提取任务。

    

    尽管疫苗在全球健康、减轻传染病和大流行爆发方面起着重要作用，但偶尔也会导致不良事件（AEs）。近年来，大型语言模型（LLMs）在有效识别和编目临床报告中的不良事件方面显示出潜力。本研究利用1990年至2016年来自疫苗不良事件报告系统（VAERS）的数据，特别关注评估LLMs在不良事件提取方面的能力。使用流感疫苗作为使用案例，评估了各种主流的LLMs，包括GPT-2、GPT-3变体、GPT-4和Llama 2。经过微调的GPT 3.5模型（AE-GPT）在严格匹配方面的平均微 F1 分数为0.704，灵活匹配方面为0.816。AE-GPT的令人鼓舞的表现突显了LLMs在处理医学数据方面的潜力，表明在先进的AE检测方面迈出了重要的一步，因此可以推广到其他AE提取任务。

    Though Vaccines are instrumental in global health, mitigating infectious diseases and pandemic outbreaks, they can occasionally lead to adverse events (AEs). Recently, Large Language Models (LLMs) have shown promise in effectively identifying and cataloging AEs within clinical reports. Utilizing data from the Vaccine Adverse Event Reporting System (VAERS) from 1990 to 2016, this study particularly focuses on AEs to evaluate LLMs' capability for AE extraction. A variety of prevalent LLMs, including GPT-2, GPT-3 variants, GPT-4, and Llama 2, were evaluated using Influenza vaccine as a use case. The fine-tuned GPT 3.5 model (AE-GPT) stood out with a 0.704 averaged micro F1 score for strict match and 0.816 for relaxed match. The encouraging performance of the AE-GPT underscores LLMs' potential in processing medical data, indicating a significant stride towards advanced AE detection, thus presumably generalizable to other AE extraction tasks.
    
[^36]: 大型语言模型中的信心-能力差距：一项认知研究

    The Confidence-Competence Gap in Large Language Models: A Cognitive Study. (arXiv:2309.16145v1 [cs.CL])

    [http://arxiv.org/abs/2309.16145](http://arxiv.org/abs/2309.16145)

    本研究探讨了大型语言模型（LLMs）在认知能力和信心动态方面的特点。我们发现，LLMs有时会在回答错误时表现出高度的信心，类似于人类心理学中的邓宁-克鲁格效应。与此同时，他们在回答正确时有时表现出较低的信心，暗示了潜在的低估偏差。这些发现强调了对LLMs认知过程的进一步研究的重要性。通过研究LLMs自我评估机制的细节，本研究提供了有意义的新发现，推动了该领域的发展。

    

    大型语言模型（LLMs）由于在各个领域的表现而引起了普遍关注。本研究探索了LLMs的认知能力和信心动态。我们深入了解了它们自我评估的信心与实际表现之间的一致性。我们利用各种问卷和真实场景对这些模型进行了实验，提取了LLMs在回答中展示的信心情况。我们的研究结果揭示了一些有趣的情况，即模型在回答错误时仍表现出高度的信心，这类似于人类心理学中观察到的邓宁-克鲁格效应。相反，也有一些情况下，模型在回答正确时展示出较低的信心，揭示了潜在的低估偏差。我们的研究结果强调了对LLMs认知过程的深入了解的必要性。通过对LLMs自我评估机制的细微研究，本研究提供了值得关注的新发现，有助于推动该领域的发展。

    Large Language Models (LLMs) have acquired ubiquitous attention for their performances across diverse domains. Our study here searches through LLMs' cognitive abilities and confidence dynamics. We dive deep into understanding the alignment between their self-assessed confidence and actual performance. We exploit these models with diverse sets of questionnaires and real-world scenarios and extract how LLMs exhibit confidence in their responses. Our findings reveal intriguing instances where models demonstrate high confidence even when they answer incorrectly. This is reminiscent of the Dunning-Kruger effect observed in human psychology. In contrast, there are cases where models exhibit low confidence with correct answers revealing potential underestimation biases. Our results underscore the need for a deeper understanding of their cognitive processes. By examining the nuances of LLMs' self-assessment mechanism, this investigation provides noteworthy revelations that serve to advance the
    
[^37]: TPE: 面向多人物协作的概念工具更好合成推理

    TPE: Towards Better Compositional Reasoning over Conceptual Tools with Multi-persona Collaboration. (arXiv:2309.16090v1 [cs.AI])

    [http://arxiv.org/abs/2309.16090](http://arxiv.org/abs/2309.16090)

    TPE是一种面向概念工具的多人物协作框架，旨在提高大型语言模型在推理和规划方面的能力。

    

    大型语言模型(LLMs)在规划使用各种功能工具，如计算器和检索器方面表现出了非凡的性能，尤其是在问答任务中。本文扩展了这些工具的定义，重点是对话系统中的概念工具。概念工具指定了一种有助于系统性或调查性思考的认知概念。这些概念工具在实践中发挥重要作用，例如多重心理或辅导策略动态应用于单个回合以组合有用的响应。为了进一步增强LLMs在这些概念工具上的推理和规划能力，我们引入了一种多人物协作框架: Think-Plan-Execute (TPE)。该框架将响应生成过程分解为三个不同的角色: 思考者，规划者和执行者。具体而言，思考者分析对话上下文中表现出的内部状态，例如用户的情绪。

    Large language models (LLMs) have demonstrated exceptional performance in planning the use of various functional tools, such as calculators and retrievers, particularly in question-answering tasks. In this paper, we expand the definition of these tools, centering on conceptual tools within the context of dialogue systems. A conceptual tool specifies a cognitive concept that aids systematic or investigative thought. These conceptual tools play important roles in practice, such as multiple psychological or tutoring strategies being dynamically applied in a single turn to compose helpful responses. To further enhance the reasoning and planning capability of LLMs with these conceptual tools, we introduce a multi-persona collaboration framework: Think-Plan-Execute (TPE). This framework decouples the response generation process into three distinct roles: Thinker, Planner, and Executor. Specifically, the Thinker analyzes the internal status exhibited in the dialogue context, such as user emot
    
[^38]: 通过排除法集成模型来在语言模型中遗忘私人文本序列

    Forgetting Private Textual Sequences in Language Models via Leave-One-Out Ensemble. (arXiv:2309.16082v1 [cs.CL])

    [http://arxiv.org/abs/2309.16082](http://arxiv.org/abs/2309.16082)

    通过教师-学生框架和排除法集成方法，在语言模型中遗忘私人文本序列时能够取得比其他方法更好的隐私-效用权衡性能。

    

    最近的研究表明，语言模型有倾向于在训练数据中记住罕见或独特的令牌序列。在部署模型后，根据个人要求，从模型中删除任何个人信息可能会被提出。每次个人想要行使被遗忘权利时重新训练底层模型的计算成本很高。我们使用教师-学生框架，并提出了一种新的排除法集成方法来从模型中遗忘需要遗忘的文本序列。在我们的方法中，多个教师模型在不同的数据集上进行训练；对于每个需要删除的目标序列，我们排除在包含该序列的训练集上训练的教师模型，并从剩余的教师模型中聚合预测结果，以提供在微调过程中的监督。在LibriSpeech和WikiText-103数据集上的实验证明，所提出的方法在隐私-效用权衡方面具有优势。

    Recent research has shown that language models have a tendency to memorize rare or unique token sequences in the training corpus. After deploying a model, practitioners might be asked to delete any personal information from the model by individuals' requests. Re-training the underlying model every time individuals would like to practice their rights to be forgotten is computationally expensive. We employ a teacher-student framework and propose a novel leave-one-out ensemble method to unlearn the targeted textual sequences that need to be forgotten from the model. In our approach, multiple teachers are trained on disjoint sets; for each targeted sequence to be removed, we exclude the teacher trained on the set containing this sequence and aggregate the predictions from remaining teachers to provide supervision during fine-tuning. Experiments on LibriSpeech and WikiText-103 datasets show that the proposed method achieves superior privacy-utility trade-offs than other counterparts.
    
[^39]: AnyMAL:一种高效可扩展的任意模态增强语言模型

    AnyMAL: An Efficient and Scalable Any-Modality Augmented Language Model. (arXiv:2309.16058v1 [cs.LG])

    [http://arxiv.org/abs/2309.16058](http://arxiv.org/abs/2309.16058)

    AnyMAL是一种高效可扩展的任意模态增强语言模型，能够处理多样化的输入模态信号，并在各种多模态任务上表现出最先进的性能。

    

    我们提出了Any-Modality Augmented Language Model (AnyMAL)，它是一个统一的模型，可以处理多样化的输入模态信号（包括文本、图像、视频、音频、IMU运动传感器），并生成文本响应。AnyMAL继承了最先进的LLMs（包括LLaMA-2（70B））的强大文本推理能力，并通过预训练的对齐模块将模态特定信号转换为联合文本空间。为进一步增强多模态LLM的能力，我们使用手动收集的多模态指令集对模型进行微调，以涵盖简单问答以外的各种主题和任务。我们进行了全面的实证分析，包括人工和自动评估，并展示了在各种多模态任务上的最先进性能。

    We present Any-Modality Augmented Language Model (AnyMAL), a unified model that reasons over diverse input modality signals (i.e. text, image, video, audio, IMU motion sensor), and generates textual responses. AnyMAL inherits the powerful text-based reasoning abilities of the state-of-the-art LLMs including LLaMA-2 (70B), and converts modality-specific signals to the joint textual space through a pre-trained aligner module. To further strengthen the multimodal LLM's capabilities, we fine-tune the model with a multimodal instruction set manually collected to cover diverse topics and tasks beyond simple QAs. We conduct comprehensive empirical analysis comprising both human and automatic evaluations, and demonstrate state-of-the-art performance on various multimodal tasks.
    
[^40]: 《语言模型中激活路径修复的最佳实践：度量和方法》的论文翻译

    Towards Best Practices of Activation Patching in Language Models: Metrics and Methods. (arXiv:2309.16042v1 [cs.LG])

    [http://arxiv.org/abs/2309.16042](http://arxiv.org/abs/2309.16042)

    本研究系统地考察了激活路径修复中的方法细节对语言模型解释性结果的影响，并提出了最佳实践建议。

    

    机械解释性旨在理解机器学习模型的内部机制，其中定位-识别重要的模型组件是关键步骤。激活路径修复，也称为因果追踪或交换干预，是完成这一任务的标准技术，但文献中存在许多变体，对超参数或方法选择缺乏一致性。在这项工作中，我们系统地考察了激活路径修复中的方法细节对结果的影响，包括评估指标和损坏方法。在语言模型的定位和电路发现的几种设置中，我们发现不同的超参数可能导致不同的解释结果。通过经验观察支持，我们提出了为什么某些指标或方法可能更受欢迎的概念性论证。最后，我们提出了激活路径修复的最佳实践建议。

    Mechanistic interpretability seeks to understand the internal mechanisms of machine learning models, where localization -- identifying the important model components -- is a key step. Activation patching, also known as causal tracing or interchange intervention, is a standard technique for this task (Vig et al., 2020), but the literature contains many variants with little consensus on the choice of hyperparameters or methodology. In this work, we systematically examine the impact of methodological details in activation patching, including evaluation metrics and corruption methods. In several settings of localization and circuit discovery in language models, we find that varying these hyperparameters could lead to disparate interpretability results. Backed by empirical observations, we give conceptual arguments for why certain metrics or methods may be preferred. Finally, we provide recommendations for the best practices of activation patching going forwards.
    
[^41]: 基础模型的有效长上下文缩放

    Effective Long-Context Scaling of Foundation Models. (arXiv:2309.16039v1 [cs.CL])

    [http://arxiv.org/abs/2309.16039](http://arxiv.org/abs/2309.16039)

    我们提出了一系列支持长上下文的基础模型，通过连续预训练和数据增强来实现，这些模型在语言建模和研究基准上都取得了显著的改进，并且通过成本效益的指导调整程序，已经超过了现有模型在长上下文任务上的性能。

    

    我们提出了一系列支持最多32768个标记的有效上下文窗口的长上下文LLM。我们通过从Llama 2连续预训练、在长文本上采样的数据集上进行训练来构建我们的模型系列。我们对语言建模、合成上下文探测任务和各种研究基准进行了广泛的评估。在研究基准上，我们的模型在大多数常规任务上都取得了一致的改进，并在长上下文任务上相对于Llama 2取得了显著的改进。值得注意的是，通过一种成本效益的指导调整程序，不需要人工标注的长指导数据，70B版本已经在一套长上下文任务中超过了gpt-3.5-turbo-16k的整体性能。除了这些结果，我们还对我们方法的各个组成部分进行了深入分析。我们深入研究了Llama的位置编码，并讨论了它在建模长依赖方面的局限性。我们还研究了变量的影响。

    We present a series of long-context LLMs that support effective context windows of up to 32,768 tokens. Our model series are built through continual pretraining from Llama 2 with longer training sequences and on a dataset where long texts are upsampled. We perform extensive evaluation on language modeling, synthetic context probing tasks, and a wide range of research benchmarks. On research benchmarks, our models achieve consistent improvements on most regular tasks and significant improvements on long-context tasks over Llama 2. Notably, with a cost-effective instruction tuning procedure that does not require human-annotated long instruction data, the 70B variant can already surpass gpt-3.5-turbo-16k's overall performance on a suite of long-context tasks. Alongside these results, we provide an in-depth analysis on the individual components of our method. We delve into Llama's position encodings and discuss its limitation in modeling long dependencies. We also examine the impact of var
    
[^42]: MedEdit：利用外部知识库进行医学问答的模型编辑

    MedEdit: Model Editing for Medical Question Answering with External Knowledge Bases. (arXiv:2309.16035v1 [cs.CL])

    [http://arxiv.org/abs/2309.16035](http://arxiv.org/abs/2309.16035)

    该论文研究了利用外部知识库进行医学问答的模型编辑方法，通过提取医学事实并将其融入到语言模型的查询提示中，显著提高了医学问答的准确率。

    

    大型语言模型（LLM）虽然在一般领域表现强大，但在特定领域的任务，如医学问答（QA）方面往往表现不佳。此外，它们往往作为“黑盒”运作，难以修改其行为。针对这一问题，我们的研究探讨了利用上下文学习的模型编辑，旨在改进LLM的响应，而无需重新微调或重新训练。具体而言，我们提出了一种全面的检索策略，从外部知识库中提取医学事实，然后将它们合并到LLM的查询提示中。通过对MedQA-SMILE数据集进行医学QA的重点研究，我们评估了不同检索模型和向LLM提供的事实数量对其影响。值得注意的是，我们编辑后的Vicuna模型的准确率从44.46％提高到48.54％。这项工作凸显了模型编辑改善LLM性能的潜力，为缓解黑盒LLM的挑战提供了实用的方法。

    Large Language Models (LLMs), although powerful in general domains, often perform poorly on domain-specific tasks like medical question answering (QA). Moreover, they tend to function as "black-boxes," making it challenging to modify their behavior. Addressing this, our study delves into model editing utilizing in-context learning, aiming to improve LLM responses without the need for fine-tuning or retraining. Specifically, we propose a comprehensive retrieval strategy to extract medical facts from an external knowledge base, and then we incorporate them into the query prompt for the LLM. Focusing on medical QA using the MedQA-SMILE dataset, we evaluate the impact of different retrieval models and the number of facts provided to the LLM. Notably, our edited Vicuna model exhibited an accuracy improvement from 44.46% to 48.54%. This work underscores the potential of model editing to enhance LLM performance, offering a practical approach to mitigate the challenges of black-box LLMs.
    
[^43]: 有针对性的图像数据增强提高了基本技能字幕鲁棒性

    Targeted Image Data Augmentation Increases Basic Skills Captioning Robustness. (arXiv:2309.15991v1 [cs.CV])

    [http://arxiv.org/abs/2309.15991](http://arxiv.org/abs/2309.15991)

    本论文提出了一种有针对性的图像数据增强方法（TIDA），通过使用文本到图像生成模型来填补关联结构差距以提高模型的人类感知能力，如性别识别。实验结果表明，相较于原始数据集，使用TIDA增强的数据集能够显著提高模型的鲁棒性。

    

    人工神经网络通常在推广到超出上下文的示例时遇到困难。这种限制的原因之一是数据集只包含有关世界潜在关联结构的部分信息。在这项工作中，我们提出了TIDA（有针对性的图像编辑数据增强）方法，它是一种专注于通过使用文本到图像生成模型来填补关联结构差距以提高模型类人能力（例如性别识别）的有针对性数据增强方法。更具体地说，TIDA识别描述图像的标题中的特定技能（例如图中特定性别的存在），改变标题（例如将“女人”改为“男人”），然后使用文本到图像模型编辑图像，以使其与新标题匹配（例如唯一地将一个女人改为一个男人同时保持上下文不变）。基于Flickr30K基准测试，我们展示了与原始数据集相比，使用TIDA增强的数据集相关的...

    Artificial neural networks typically struggle in generalizing to out-of-context examples. One reason for this limitation is caused by having datasets that incorporate only partial information regarding the potential correlational structure of the world. In this work, we propose TIDA (Targeted Image-editing Data Augmentation), a targeted data augmentation method focused on improving models' human-like abilities (e.g., gender recognition) by filling the correlational structure gap using a text-to-image generative model. More specifically, TIDA identifies specific skills in captions describing images (e.g., the presence of a specific gender in the image), changes the caption (e.g., "woman" to "man"), and then uses a text-to-image model to edit the image in order to match the novel caption (e.g., uniquely changing a woman to a man while maintaining the context identical). Based on the Flickr30K benchmark, we show that, compared with the original data set, a TIDA-enhanced dataset related to
    
[^44]: 在HYKIST项目中为越南自动语音识别进行无监督预训练的研究

    Unsupervised Pre-Training for Vietnamese Automatic Speech Recognition in the HYKIST Project. (arXiv:2309.15869v1 [cs.CL])

    [http://arxiv.org/abs/2309.15869](http://arxiv.org/abs/2309.15869)

    本研究旨在通过无监督预训练方法，为HYKIST项目中的越南自动语音识别系统构建提供支持。这将有助于解决医疗领域患者和医生之间语言障碍的问题，提高患者护理质量。

    

    在今天的全球互联中，移居国外变得越来越普遍，无论是为了就业、难民安置还是其他原因。母语与移民之间的语言障碍是日常生活中的常见问题，特别是在医疗领域。这可能导致患者和医生在病史采集或急诊室的沟通过程中出现困难，从而影响患者护理。HYKIST项目的目标是开发一个语音翻译系统，以支持患者与医生之间的沟通，其中包括自动语音识别（ASR）和机器翻译（MT）技术。尽管近年来ASR系统在某些特定任务上表现出色，例如LibriSpeech，但由于讲话风格、声学和录音设置的多样性以及缺乏特定领域的训练数据，构建一个良好的模型仍然较为困难。本论文描述了我们在医疗领域的电话对话语音识别任务中构建ASR系统的努力。

    In today's interconnected globe, moving abroad is more and more prevalent, whether it's for employment, refugee resettlement, or other causes. Language difficulties between natives and immigrants present a common issue on a daily basis, especially in medical domain. This can make it difficult for patients and doctors to communicate during anamnesis or in the emergency room, which compromises patient care. The goal of the HYKIST Project is to develop a speech translation system to support patient-doctor communication with ASR and MT.  ASR systems have recently displayed astounding performance on particular tasks for which enough quantities of training data are available, such as LibriSpeech. Building a good model is still difficult due to a variety of speaking styles, acoustic and recording settings, and a lack of in-domain training data. In this thesis, we describe our efforts to construct ASR systems for a conversational telephone speech recognition task in the medical domain for Viet
    
[^45]: 图像-文本多模型综述论文

    A Survey on Image-text Multimodal Models. (arXiv:2309.15857v1 [cs.CL])

    [http://arxiv.org/abs/2309.15857](http://arxiv.org/abs/2309.15857)

    图像-文本多模型综述论文全面回顾了其发展历程和当前状态，提出了新的分类方法，并阐明了该领域的挑战和潜在研究方向。

    

    在人工智能不断发展的背景下，图像和文本信息的融合成为一个至关重要的领域，导致了图像-文本多模型的出现。本论文全面回顾了图像-文本多模型的发展历程和当前状态，探讨了它们的应用价值、挑战和潜在研究方向。首先，我们重新审视了这些模型的基本概念和发展里程碑，引入了一种新的分类方法，将它们的发展分为三个不同的阶段，基于它们被引入的时间和对学科的影响。此外，基于任务在学术领域中的重要性和普及性，我们提出了将与图像-文本多模型相关的任务划分为五个主要类型的分类方法，阐明了每个类别内的最新进展和关键技术。尽管这些模型取得了显著的成就，但仍面临着许多挑战。

    Amidst the evolving landscape of artificial intelligence, the convergence of visual and textual information has surfaced as a crucial frontier, leading to the advent of image-text multimodal models. This paper provides a comprehensive review of the evolution and current state of image-text multimodal models, exploring their application value, challenges, and potential research trajectories. Initially, we revisit the basic concepts and developmental milestones of these models, introducing a novel classification that segments their evolution into three distinct phases, based on their time of introduction and subsequent impact on the discipline. Furthermore, based on the tasks' significance and prevalence in the academic landscape, we propose a categorization of the tasks associated with image-text multimodal models into five major types, elucidating the recent progress and key technologies within each category. Despite the remarkable accomplishments of these models, numerous challenges a
    
[^46]: 共同训练大型自回归多模态模型

    Jointly Training Large Autoregressive Multimodal Models. (arXiv:2309.15564v1 [cs.LG])

    [http://arxiv.org/abs/2309.15564](http://arxiv.org/abs/2309.15564)

    本研究提出了共同训练大型自回归多模态模型的方法，通过模块化的方式融合语言和图像生成模型，同时引入了数据高效的指令调优策略，使得该模型在生成高质量多模态输出方面表现出卓越的性能。

    

    最近几年，语言和文本到图像模型的大规模预训练取得了重大突破，彻底改变了机器学习领域。然而，将这两种模态集成到一个能够生成无缝多模态输出的单一强大模型仍然是一个重大挑战。为了解决这一问题，我们提出了联合自回归混合（JAM）框架，一种系统融合现有文本和图像生成模型的模块化方法。我们还引入了一种专门的、数据高效的指令调优策略，针对混合模态生成任务进行了优化。我们最终的调优模型在生成高质量多模态输出方面表现出无与伦比的性能，并代表了第一个明确为此目的而设计的模型。

    In recent years, advances in the large-scale pretraining of language and text-to-image models have revolutionized the field of machine learning. Yet, integrating these two modalities into a single, robust model capable of generating seamless multimodal outputs remains a significant challenge. To address this gap, we present the Joint Autoregressive Mixture (JAM) framework, a modular approach that systematically fuses existing text and image generation models. We also introduce a specialized, data-efficient instruction-tuning strategy, tailored for mixed-modal generation tasks. Our final instruct-tuned model demonstrates unparalleled performance in generating high-quality multimodal outputs and represents the first model explicitly designed for this purpose.
    
[^47]: 大规模多语言自监督学习的联合预测和去噪

    joint prediction and denoising for large-scale multilingual self-supervised learning. (arXiv:2309.15317v1 [cs.CL])

    [http://arxiv.org/abs/2309.15317](http://arxiv.org/abs/2309.15317)

    这项研究提出了WavLabLM，它通过联合预测和去噪的方法，实现了在136种语言的40k小时数据上进行大规模多语言自监督学习。WavLabLM的多阶段预训练方法解决了多语言数据的语言失衡问题，使其在ML-SUPERB上达到了与XLS-R相当的性能，同时仅使用不到10%的训练数据，这使得SSL在学术高性能计算上可行。

    

    多语言自监督学习(SSL)由于处理多种语言所需的费用和复杂性而经常落后于最先进的方法。这进一步影响了SSL的可重复性，由于资源使用的限制，SSL已经仅限于少数研究团队。我们展示了更强大的技术实际上可以导致更高效的预训练，从而使更多的研究团队能够加入SSL。我们提出了WavLabLM，将WavLM的联合预测和去噪扩展到136种语言的40k小时数据。为了构建WavLabLM，我们设计了一种新颖的多阶段预训练方法，旨在解决多语言数据的语言失衡问题。WavLabLM在ML-SUPERB上实现了与XLS-R相当的性能，仅使用不到10%的训练数据，使得SSL在学术高性能计算上可实现。我们还展示了vanilla HuBERT Base模型可以实现进一步的效率提升，仅使用3%的数据、4个GPU和有限的试验次数，就能保持94%的XLS-R性能。

    Multilingual self-supervised learning (SSL) has often lagged behind state-of-the-art (SOTA) methods due to the expenses and complexity required to handle many languages. This further harms the reproducibility of SSL, which is already limited to few research groups due to its resource usage. We show that more powerful techniques can actually lead to more efficient pre-training, opening SSL to more research groups. We propose WavLabLM, which extends WavLM's joint prediction and denoising to 40k hours of data across 136 languages. To build WavLabLM, we devise a novel multi-stage pre-training method, designed to address the language imbalance of multilingual data. WavLabLM achieves comparable performance to XLS-R on ML-SUPERB with less than 10% of the training data, making SSL realizable with academic compute. We show that further efficiency can be achieved with a vanilla HuBERT Base model, which can maintain 94% of XLS-R's performance with only 3% of the data, 4 GPUs, and limited trials. 
    
[^48]: HANS，你聪明吗？神经系统的Clever Hans效应分析

    HANS, are you clever? Clever Hans Effect Analysis of Neural Systems. (arXiv:2309.12481v1 [cs.CL])

    [http://arxiv.org/abs/2309.12481](http://arxiv.org/abs/2309.12481)

    这项研究调查了指导调整的大型语言模型（It-LLMs）对多项选择题（MCQ）的鲁棒性能力，在选择顺序变动时揭示了选择偏见和推理能力的问题。

    

    指导调整的大型语言模型(It-LLMs)展示出了在认知状态、意图和反应方面推理的出色能力，可以让人们有效地引导和理解日常社交互动。事实上，已经提出了几个多项选择题(MCQ)基准来构建对模型能力的确切评估。然而，早期的研究表明It-LLMs中存在固有的“顺序偏见”，给适当的评估带来了挑战。本文通过使用四个MCQ基准对It-LLMs的抵抗能力进行了研究。通过引入对抗性示例，我们展示了显著的性能差距，特别是在选择顺序变动时，揭示了选择偏见并引发了对推理能力的讨论。通过第一位置和模型选择之间的相关性，我们假设在模型中存在结构启发式方法。

    Instruction-tuned Large Language Models (It-LLMs) have been exhibiting outstanding abilities to reason around cognitive states, intentions, and reactions of all people involved, letting humans guide and comprehend day-to-day social interactions effectively. In fact, several multiple-choice questions (MCQ) benchmarks have been proposed to construct solid assessments of the models' abilities. However, earlier works are demonstrating the presence of inherent "order bias" in It-LLMs, posing challenges to the appropriate evaluation. In this paper, we investigate It-LLMs' resilience abilities towards a series of probing tests using four MCQ benchmarks. Introducing adversarial examples, we show a significant performance gap, mainly when varying the order of the choices, which reveals a selection bias and brings into discussion reasoning abilities. Following a correlation between first positions and model choices due to positional bias, we hypothesized the presence of structural heuristics in 
    
[^49]: 基于语言模型的概率测量专利权要求范围的新方法

    A novel approach to measuring patent claim scope based on probabilities obtained from (large) language models. (arXiv:2309.10003v1 [cs.CL])

    [http://arxiv.org/abs/2309.10003](http://arxiv.org/abs/2309.10003)

    本文提出了一种使用概率从语言模型中获得的自信息来测量专利权要求范围的新方法。该方法通过计算要求的发生概率和自信息来评估要求的信息量，进而反映出要求的范围。研究结果表明，不同类型的语言模型对范围测量的影响不同，最简单的模型可以将范围度量简化为单词或字符计数的倒数。此方法在九个系列的专利权要求上进行了验证，结果表明各系列的要求范围逐渐减小。

    

    本文提出了一种将专利权要求的范围测量为该要求所包含的自信息的倒数的方法。这种方法基于信息论，基于一个假设，即罕见的概念比平常的概念更具信息量，因为它更令人惊讶。自信息是从该要求的发生概率计算得出的，其中概率是根据语言模型计算的。本文考虑了五个语言模型，从最简单的模型（每个单词或字符均从均匀分布中抽取）到中等模型（使用平均词或字符频率），再到一个大型语言模型（GPT2）。有趣的是，最简单的语言模型将范围度量减少为单词或字符计数的倒数，这是先前作品中已经使用的度量标准。该方法应用于九个系列的针对不同发明的专利权要求，其中每个系列的要求范围逐渐减小。

    This work proposes to measure the scope of a patent claim as the reciprocal of the self-information contained in this claim. Grounded in information theory, this approach is based on the assumption that a rare concept is more informative than a usual concept, inasmuch as it is more surprising. The self-information is calculated from the probability of occurrence of that claim, where the probability is calculated in accordance with a language model. Five language models are considered, ranging from the simplest models (each word or character is drawn from a uniform distribution) to intermediate models (using average word or character frequencies), to a large language model (GPT2). Interestingly, the simplest language models reduce the scope measure to the reciprocal of the word or character count, a metric already used in previous works. Application is made to nine series of patent claims directed to distinct inventions, where the claims in each series have a gradually decreasing scope.
    
[^50]: 通过有符号梯度下降优化LLMs量化中的权重舍入

    Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs. (arXiv:2309.05516v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.05516](http://arxiv.org/abs/2309.05516)

    本文提出一种名为SignRound的优化权重舍入的方法，通过使用有符号梯度进行轻量级分块调整，解决了大型语言模型(LLMs)的量化挑战。

    

    大型语言模型(LLMs)在执行语言相关任务方面表现出了非凡的能力。然而，由于其巨大的内存和存储需求，它们的部署面临着重大挑战。为了解决这个问题，仅针对权重的量化，特别是3位和4位仅针对权重的量化，已经成为最可行的解决方案之一。随着位数的减少，量化网格变得更加宽泛，从而强调了上下舍入的重要性。尽管先前的研究表明，在某些情况下，通过添加扰动细调上下舍入可以提高准确性，但我们的研究受制于这些扰动的精确且有限的边界，只有改变舍入值的阈值才具有重要性。因此，我们提出了一种简洁高效的优化权重舍入任务的方法。我们的方法名为SignRound，它涉及使用有符号梯度的轻量级分块调整。

    Large Language Models (LLMs) have proven their exceptional capabilities in performing language-related tasks. However, their deployment poses significant challenges due to their considerable memory and storage requirements. In response to this issue, weight-only quantization, particularly 3 and 4-bit weight-only quantization, has emerged as one of the most viable solutions. As the number of bits decreases, the quantization grid broadens, thus emphasizing the importance of up and down rounding. While previous studies have demonstrated that fine-tuning up and down rounding with the addition of perturbations can enhance accuracy in some scenarios, our study is driven by the precise and limited boundary of these perturbations, where only the threshold for altering the rounding value is of significance. Consequently, we propose a concise and highly effective approach for optimizing the weight rounding task. Our method, named SignRound, involves lightweight block-wise tuning using signed gra
    
[^51]: 注意力和自监督语音嵌入对非语义语音任务的影响

    Effect of Attention and Self-Supervised Speech Embeddings on Non-Semantic Speech Tasks. (arXiv:2308.14359v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2308.14359](http://arxiv.org/abs/2308.14359)

    该论文探讨了注意力和自监督语音嵌入对非语义语音任务的影响，特别是情感理解。实验结果表明，基于自注意力的轻量级HuBERT-Large模型在这些任务中表现出色。

    

    在现实中，人类情感理解在使对话技术成为主流方面至关重要。我们将语音情感理解视为一种知觉任务，这是一种更现实的情景。在不同的上下文（语言，人口统计学等），不同比例的人会将相同的语音片段视为非一致的情感。作为ACM多媒体2023计算语音联机挑战（ComParE）的一部分，在EMotion Share轨道上，我们利用他们丰富的多语种演讲者和多标签回归目标的数据集，即“情感分享”或对该情感的感知。我们证明了不同基础模型的训练方案决定了它们在超越语音识别的任务中的有效性，特别是对于情感理解等非语义语音任务。这是一个非常复杂的任务，因为涉及到多语种演讲者，目标标签的变化以及回归数据集中的固有不平衡性。我们的结果表明，基于自注意力的轻量级HuBERT-Large模型在这些任务中表现出色。

    Human emotion understanding is pivotal in making conversational technology mainstream. We view speech emotion understanding as a perception task which is a more realistic setting. With varying contexts (languages, demographics, etc.) different share of people perceive the same speech segment as a non-unanimous emotion. As part of the ACM Multimedia 2023 Computational Paralinguistics ChallengE (ComParE) in the EMotion Share track, we leverage their rich dataset of multilingual speakers and multi-label regression target of 'emotion share' or perception of that emotion. We demonstrate that the training scheme of different foundation models dictates their effectiveness for tasks beyond speech recognition, especially for non-semantic speech tasks like emotion understanding. This is a very complex task due to multilingual speakers, variability in the target labels, and inherent imbalance in the regression dataset. Our results show that HuBERT-Large with a self-attention-based light-weight se
    
[^52]: 思想算法：增强大型语言模型中的思想探索

    Algorithm of Thoughts: Enhancing Exploration of Ideas in Large Language Models. (arXiv:2308.10379v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.10379](http://arxiv.org/abs/2308.10379)

    本论文提出了一种名为"思想算法"的策略，通过算法推理路径推动大型语言模型的思想探索，以低成本、低存储和低计算开销的方式扩展了其推理能力。结果显示，使用算法指导的大型语言模型的性能可以超越算法本身。

    

    当前的文献旨在超越“连续思维”的方法，通常采用外部操作方法，在生成过程中停止、修改，然后恢复以增强大型语言模型（LLM）的推理能力。这种模式增加了查询请求的数量，增加了成本、内存和计算开销。针对这个问题，我们提出了思想算法——一种新颖的策略，通过算法推理路径推动LLM，开创了一种新的上下文学习模式。通过使用算法示例，我们利用LLM的固有循环动力学，仅使用一个或少数几个查询扩展其思想探索。我们的技术优于早期的单次查询方法，并与最近采用广泛的树搜索算法的多次查询策略不相上下。有趣的是，我们的结果表明，使用算法指导LLM可以使性能超越算法本身，这暗示着

    Current literature, aiming to surpass the "Chain-of-Thought" approach, often resorts to an external modus operandi involving halting, modifying, and then resuming the generation process to boost Large Language Models' (LLMs) reasoning capacities. This mode escalates the number of query requests, leading to increased costs, memory, and computational overheads. Addressing this, we propose the Algorithm of Thoughts -- a novel strategy that propels LLMs through algorithmic reasoning pathways, pioneering a new mode of in-context learning. By employing algorithmic examples, we exploit the innate recurrence dynamics of LLMs, expanding their idea exploration with merely one or a few queries. Our technique outperforms earlier single-query methods and stands on par with a recent multi-query strategy that employs an extensive tree search algorithm. Intriguingly, our results suggest that instructing an LLM using an algorithm can lead to performance surpassing that of the algorithm itself, hinting 
    
[^53]: VisIT-Bench: 一个受真实世界使用启发的视觉语言指示评估基准

    VisIT-Bench: A Benchmark for Vision-Language Instruction Following Inspired by Real-World Use. (arXiv:2308.06595v1 [cs.CL])

    [http://arxiv.org/abs/2308.06595](http://arxiv.org/abs/2308.06595)

    VisIT-Bench是一个用于评价真实世界中视觉语言模型指示遵循的基准，包含了各种任务并提供了详细描述，可以自动评估多模态生成的质量差距。

    

    我们引入了VisIT-Bench（Visual InsTruction Benchmark），这是一个评价用于真实世界使用的视觉语言模型的指示遵循基准。我们的起点是策划了70个“指示家族”，我们认为指示调优的视觉语言模型应该能够解决这些家族。任务不仅限于VQAv2和COCO等评估，涵盖了从基本识别到游戏玩法和创造性生成的各种任务。在策划之后，我们的数据集包括592个测试查询，每个查询都带有一个人工编写的指示条件化的字幕。这些描述展现了特定指示因素，例如对于询问店面对于轮椅用户的易访问性的指示，条件化的字幕描述了斜坡/潜在障碍物。这些描述使得我们可以：1）收集每个实例的人工验证的参考输出；2）使用仅文本的语言模型对候选多模态生成进行自动评估，与人类判断相一致。我们量化了质量差距。

    We introduce VisIT-Bench (Visual InsTruction Benchmark), a benchmark for evaluation of instruction-following vision-language models for real-world use. Our starting point is curating 70 'instruction families' that we envision instruction tuned vision-language models should be able to address. Extending beyond evaluations like VQAv2 and COCO, tasks range from basic recognition to game playing and creative generation. Following curation, our dataset comprises 592 test queries, each with a human-authored instruction-conditioned caption. These descriptions surface instruction-specific factors, e.g., for an instruction asking about the accessibility of a storefront for wheelchair users, the instruction-conditioned caption describes ramps/potential obstacles. These descriptions enable 1) collecting human-verified reference outputs for each instance; and 2) automatic evaluation of candidate multimodal generations using a text-only LLM, aligning with human judgment. We quantify quality gaps be
    
[^54]: 从Hapax Rate模型导出的Zipf和Heaps定律的修正

    Corrections of Zipf's and Heaps' Laws Derived from Hapax Rate Models. (arXiv:2307.12896v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.12896](http://arxiv.org/abs/2307.12896)

    本文的创新是基于Hapax Rate模型引入了对Zipf和Heaps定律的修正，并发现逻辑模型拟合效果最优。

    

    本文基于Hapax Rate模型引入了对Zipf和Heaps定律的修正。推导基于两个假设：第一个假设是标准的瓮模型，预测较短文本的边际词频分布看起来就像是从一个给定的较长文本中盲目采样词元。第二个假设假定Hapax的频率是文本大小的简单函数。讨论了四个这样的函数：常数模型、Davis模型、线性模型和逻辑模型。结果显示逻辑模型拟合效果最好。

    The article introduces corrections to Zipf's and Heaps' laws based on systematic models of the hapax rate. The derivation rests on two assumptions: The first one is the standard urn model which predicts that marginal frequency distributions for shorter texts look as if word tokens were sampled blindly from a given longer text. The second assumption posits that the rate of hapaxes is a simple function of the text size. Four such functions are discussed: the constant model, the Davis model, the linear model, and the logistic model. It is shown that the logistic model yields the best fit.
    
[^55]: 重温接受性判断

    Revisiting Acceptability Judgements. (arXiv:2305.14091v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14091](http://arxiv.org/abs/2305.14091)

    本文介绍了第一个大规模的非英语语言可接受性数据集CoLAC，发现语言模型性能低于人类的表现，但跨语言转移可行，并可以追溯到预训练阶段。

    

    至NLP社区最后一次关注语言可接受性已经过去多年。在本文中，我们在大型语言模型的背景下重温这个话题。我们引入了 CoLAC-中文语言可接受性语料库，这是第一个由母语讲者验证并带有两组标签的大规模非英语可接受性数据集。我们的实验表明，即使最大的InstructGPT模型在CoLAC上也只能表现随机水平，而ChatGPT的性能（48.30 MCC）也远低于监督模型（59.03 MCC）和人类（65.11 MCC）。通过跨语言转移实验和细粒度的语言分析，我们首次证明了语言可接受性知识可以在不同类型的语言之间转移，而且可以追溯到预训练阶段。

    Years have passed since the NLP community has last focused on linguistic acceptability. In this work, we revisit this topic in the context of large language models. We introduce CoLAC - Corpus of Linguistic Acceptability in Chinese, the first large-scale non-English acceptability dataset that is verified by native speakers and comes with two sets of labels. Our experiments show that even the largest InstructGPT model performs only at chance level on CoLAC, while ChatGPT's performance (48.30 MCC) is also way below supervised models (59.03 MCC) and human (65.11 MCC). Through cross-lingual transfer experiments and fine-grained linguistic analysis, we demonstrate for the first time that knowledge of linguistic acceptability can be transferred across typologically distinct languages, as well as be traced back to pre-training.
    
[^56]: LLM-Pruner: 关于大型语言模型结构修剪的研究

    LLM-Pruner: On the Structural Pruning of Large Language Models. (arXiv:2305.11627v1 [cs.CL])

    [http://arxiv.org/abs/2305.11627](http://arxiv.org/abs/2305.11627)

    本文提出了一种方法，名为LLM-Pruner，采用结构修剪的方式在保留大多数功能的同时，压缩LLM的结构，以减少LLM在部署、推理和训练阶段中的大小和复杂度。

    

    大型语言模型(LLMs)展示出在语言理解和生成方面令人惊讶的能力。然而，这种能力通常伴随着巨大的模型大小，这在部署、推理和训练阶段都存在显著挑战。我们以任务无关的方式探索了LLM的压缩方法，旨在保留原始LLM的多任务解决和语言生成能力。我们的方法采用结构修剪，在梯度信息的支持下选择性地移除非关键的耦合结构，最大限度地保留了大多数LLM的功能。

    Large language models (LLMs) have shown remarkable capabilities in language understanding and generation. However, such impressive capability typically comes with a substantial model size, which presents significant challenges in both the deployment, inference, and training stages. With LLM being a general-purpose task solver, we explore its compression in a task-agnostic manner, which aims to preserve the multi-task solving and language generation ability of the original LLM. One challenge to achieving this is the enormous size of the training corpus of LLM, which makes both data transfer and model post-training over-burdensome. Thus, we tackle the compression of LLMs within the bound of two constraints: being task-agnostic and minimizing the reliance on the original training dataset. Our method, named LLM-Pruner, adopts structural pruning that selectively removes non-critical coupled structures based on gradient information, maximally preserving the majority of the LLM's functionalit
    
[^57]: 大型语言模型合成文本数据集的语言多样性可视化

    Visualizing Linguistic Diversity of Text Datasets Synthesized by Large Language Models. (arXiv:2305.11364v1 [cs.CL])

    [http://arxiv.org/abs/2305.11364](http://arxiv.org/abs/2305.11364)

    该论文介绍了一种新的可视化工具，用于分析大型语言模型生成的数据集的句法多样性，可以通过分层可视化来帮助用户快速浏览概述和检查各个示例。

    

    大型语言模型（LLMs）可通过少量提示生成更精细的数据集用于基准测试、微调或其他用例。然而，理解和评估这些数据集很困难，而LLM生成数据的失败模式仍不为人们所理解。具体来说，数据可能以意外的方式变得重复，不仅语义上如此，而且在句法、词汇方面也是如此。我们提出了LinguisticLens，一种新颖的交互式可视化工具，用于理解和分析LLM生成数据集的句法多样性。 LinguisticLens沿着句法、词汇和语义轴将文本聚类。它支持文本数据集的分层可视化，允许用户快速浏览概述和检查各个示例。实时演示可在shorturl.at/zHOUV查看。

    Large language models (LLMs) can be used to generate smaller, more refined datasets via few-shot prompting for benchmarking, fine-tuning or other use cases. However, understanding and evaluating these datasets is difficult, and the failure modes of LLM-generated data are still not well understood. Specifically, the data can be repetitive in surprising ways, not only semantically but also syntactically and lexically. We present LinguisticLens, a novel inter-active visualization tool for making sense of and analyzing syntactic diversity of LLM-generated datasets. LinguisticLens clusters text along syntactic, lexical, and semantic axes. It supports hierarchical visualization of a text dataset, allowing users to quickly scan for an overview and inspect individual examples. The live demo is available at shorturl.at/zHOUV.
    
[^58]: 关于形态信息在上下文词形还原中的作用

    On the Role of Morphological Information for Contextual Lemmatization. (arXiv:2302.00407v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.00407](http://arxiv.org/abs/2302.00407)

    本文研究了形态信息在六种语言的上下文词形还原器开发中的作用，并对词形还原器进行了领域外性能评估。

    

    词形还原是一项自然语言处理的任务，它包括从给定的屈折词生成其规范形式或词形还原。词形还原是简化下游自然语言处理应用的基本任务之一，对于高度屈折语言尤为重要。虽然根据屈折词获得词形还原形式的过程可以通过考虑其形态句法类别来解释，但在训练上下文词形还原器时引入细粒度的形态句法信息已经成为常见做法，而无视下游绩效是否优化。为了解决这个问题，本文实证研究了在六种语言中考察形态信息在上下文词形还原器开发中的作用，这六种语言的形态复杂性各不相同，包括巴斯克语、土耳其语、俄语、捷克语、西班牙语和英语。此外，与绝大多数先前的工作不同，我们还在领域外环境中进行了词形还原器的评估。

    Lemmatization is a natural language processing (NLP) task which consists of producing, from a given inflected word, its canonical form or lemma. Lemmatization is one of the basic tasks that facilitate downstream NLP applications, and is of particular importance for high-inflected languages. Given that the process to obtain a lemma from an inflected word can be explained by looking at its morphosyntactic category, including fine-grained morphosyntactic information to train contextual lemmatizers has become common practice, without considering whether that is the optimum in terms of downstream performance. In order to address this issue, in this paper we empirically investigate the role of morphological information to develop contextual lemmatizers in six languages within a varied spectrum of morphological complexity: Basque, Turkish, Russian, Czech, Spanish and English. Furthermore, and unlike the vast majority of previous work, we also evaluate lemmatizers in out-of-domain settings, wh
    
[^59]: 对话中的个人实体、概念和命名实体链接

    Personal Entity, Concept, and Named Entity Linking in Conversations. (arXiv:2206.07836v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.07836](http://arxiv.org/abs/2206.07836)

    本文介绍了一个用于对话中实体链接的数据集和工具，该工具能够理解用户的话语中的个人实体和概念，并将其与外部知识连接起来。

    

    构建可以与人类进行自然且基于知识的互动的对话代理需要理解用户的话语。实体链接（EL）是一种理解自然语言文本并将其与外部知识连接的有效且广泛使用的方法。然而，已有的用于文档注释的EL方法在对话中表现不佳，因为对于理解用户的话语来说，个人实体（例如，“我的汽车”）和概念是必不可少的。本文介绍了一个用于对话中实体链接的数据集和工具。我们为1327个对话话语收集了EL注释，包括命名实体、概念和个人实体的链接。该数据集用于训练我们的对话实体链接工具CREL。与现有的EL方法不同，CREL旨在识别命名实体和概念，并利用共指消解技术识别个人实体和引用。

    Building conversational agents that can have natural and knowledge-grounded interactions with humans requires understanding user utterances. Entity Linking (EL) is an effective and widely used method for understanding natural language text and connecting it to external knowledge. It is, however, shown that existing EL methods developed for annotating documents are suboptimal for conversations, where personal entities (e.g., "my cars") and concepts are essential for understanding user utterances. In this paper, we introduce a collection and a tool for entity linking in conversations. We collect EL annotations for 1327 conversational utterances, consisting of links to named entities, concepts, and personal entities. The dataset is used for training our toolkit for conversational entity linking, CREL. Unlike existing EL methods, CREL is developed to identify both named entities and concepts. It also utilizes coreference resolution techniques to identify personal entities and references to
    
[^60]: 加权有限状态机的高阶导数

    Higher-order Derivatives of Weighted Finite-state Machines. (arXiv:2106.00749v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2106.00749](http://arxiv.org/abs/2106.00749)

    本研究提出了一种通用算法来评估加权有限状态机关于归一化常数的高阶导数，并且在二阶导数的情况下，方案的运行时间是最优的。此外，该方法还导致了计算二阶期望的更快算法。

    

    加权有限状态机是自然语言处理系统的基本构建块。从它们在20世纪90年代早期用于噪声信道模型开始，一直到现代神经参数化条件随机场的使用，它们经受了时间的考验。本研究考察了关于加权有限状态机归一化常数的高阶导数的计算。我们提供了一种评估所有阶数导数的通用算法，这在文献中尚未描述。在二阶导数的情况下，我们的方案在最佳的 \mathcal{O}(A^2 N^4) 时间内运行，其中 A 是字母表大小，N 是状态数。我们的算法比之前的算法快得多。此外，我们的方法导致了一个计算二阶期望的显著更快的算法，例如协方差矩阵和一阶期望的梯度。

    Weighted finite-state machines are a fundamental building block of NLP systems. They have withstood the test of time -- from their early use in noisy channel models in the 1990s up to modern-day neurally parameterized conditional random fields. This work examines the computation of higher-order derivatives with respect to the normalization constant for weighted finite-state machines. We provide a general algorithm for evaluating derivatives of all orders, which has not been previously described in the literature. In the case of second-order derivatives, our scheme runs in the optimal $\mathcal{O}(A^2 N^4)$ time where $A$ is the alphabet size and $N$ is the number of states. Our algorithm is significantly faster than prior algorithms. Additionally, our approach leads to a significantly faster algorithm for computing second-order expectations, such as covariance matrices and gradients of first-order expectations.
    

