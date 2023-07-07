# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Lost in the Middle: How Language Models Use Long Contexts.](http://arxiv.org/abs/2307.03172) | 本研究分析了语言模型在多文档问答和键值检索任务中的表现，发现当相关信息位于输入文本的开头或结尾时性能最佳，而当模型需要在长文本的中间访问相关信息时性能显著下降。此外，即使对于专门处理长文本的模型，输入文本越长性能也会大幅降低。我们的研究为理解语言模型如何使用输入文本的上下文提供了新的认识，并且为未来的长文本模型提供了新的评估方案。 |
| [^2] | [Focused Transformer: Contrastive Training for Context Scaling.](http://arxiv.org/abs/2307.03170) | Focused Transformer通过反差训练优化了上下文缩放问题，允许语言模型处理更长的上下文信息。 |
| [^3] | [Distilling Large Vision-Language Model with Out-of-Distribution Generalizability.](http://arxiv.org/abs/2307.03135) | 本文研究了针对大型视觉语言模型的模型压缩方法，将教师模型的视觉表示压缩到学生模型中。研究重点在于超出分布可泛化的问题，并提出了两个原则来增强学生模型的性能。 |
| [^4] | [T-MARS: Improving Visual Representations by Circumventing Text Feature Learning.](http://arxiv.org/abs/2307.03132) | T-MARS提出一种新的数据筛选方法，通过规避文本特征学习，改善了视觉表示的学习，解决了大型多模态数据集中存在的文本与图像重叠的问题。 |
| [^5] | [BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training.](http://arxiv.org/abs/2307.03131) | 本研究通过最小风险训练系统性地分析和比较了各种自动度量，并发现了BLEURT和BARTScore等度量中存在的通用对抗翻译现象。研究结果表明，这些鲁棒性缺陷主要由训练数据集中的分布偏差和度量范式的倾向引起。通过引入标记级约束，可以提高度量的鲁棒性。 |
| [^6] | [VisKoP: Visual Knowledge oriented Programming for Interactive Knowledge Base Question Answering.](http://arxiv.org/abs/2307.03130) | VisKoP是一种知识库问答（KBQA）系统，通过将人类融入到知识库查询的编辑和调试中，提供了一种视觉知识导向编程的平台。它不仅提供了神经程序归纳模块，还将程序映射为图形元素，使其易于编辑和调试。通过提供自动补全功能和高效的执行引擎，VisKoP在处理大规模知识库问答时表现出高效性和准确性。 |
| [^7] | [Extracting Multi-valued Relations from Language Models.](http://arxiv.org/abs/2307.03122) | 该论文研究了从预训练语言模型中提取多值关系的问题，并通过排名和选择任务的方法解决了这个问题。结果表明，选择具有特定关系阈值以上的对象可以达到49.5%的F1得分，这对于将语言模型应用于多值槽位填充任务而言是具有挑战性的。该研究为从潜在语言表示中提取关系知识开辟了进一步研究的道路。 |
| [^8] | [KoRC: Knowledge oriented Reading Comprehension Benchmark for Deep Text Understanding.](http://arxiv.org/abs/2307.03115) | 近年来，深度文本理解的重要性在许多基准测试中得到了强调。为了克服已有基准测试的限制，本论文提出了一个新的基准测试，KoRC，在知识覆盖和答案格式上具有优势。实验结果表明，KoRC可以帮助改进文本理解模型的性能。 |
| [^9] | [A Survey on Evaluation of Large Language Models.](http://arxiv.org/abs/2307.03109) | 本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。 |
| [^10] | [Efficient Domain Adaptation of Sentence Embeddings using Adapters.](http://arxiv.org/abs/2307.03104) | 本论文提出了一种通过训练轻量级适配器来高效域自适应句子嵌入的方法，避免了微调整个句子嵌入模型的资源消耗。通过训练特定领域的适配器，可以在不同领域中使用同一模型获得良好的性能。 |
| [^11] | [OpenDelta: A Plug-and-play Library for Parameter-efficient Adaptation of Pre-trained Models.](http://arxiv.org/abs/2307.03084) | OpenDelta是一个开源库，提供了各种delta调整方法的即插即用实现。它能够以高效的方式调整大型预训练模型的参数，而无需修改模型的代码，具有实用性和灵活性。 |
| [^12] | [DeepOnto: A Python Package for Ontology Engineering with Deep Learning.](http://arxiv.org/abs/2307.03067) | DeepOnto是一个Python包，用于深度学习本体工程。它通过集成深度学习框架和本体API，提供了丰富的工具和算法，支持本体工程任务，如本体对齐和完成。 |
| [^13] | [Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain.](http://arxiv.org/abs/2307.03042) | 本研究提出了一种参数高效微调（PEFT）方法，在临床领域使用临床记录训练了一个专门适配临床领域的LLaMA-LoRA模型，同时提出了一个两步PEFT框架，用于将其与Downstream LLaMA-LoRA适配器进行融合，以实现领域适应。 |
| [^14] | [Improving Retrieval-Augmented Large Language Models via Data Importance Learning.](http://arxiv.org/abs/2307.03027) | 本文通过多线性扩展算法评估检索增强模型中检索到的数据点的数据重要性，并提出了一个多项式时间算法来计算其数据重要性。实验结果表明，修剪或增强大型语言模型可以提高性能。 |
| [^15] | [Style Over Substance: Evaluation Biases for Large Language Models.](http://arxiv.org/abs/2307.03025) | 这项研究调查了人类和基于大型语言模型的评委在比较不同模型输出时的行为，并发现评估过程中存在偏见，即尽管包含事实错误，答案仍然被更高地评分。为了解决这个问题，我们提出了 |
| [^16] | [Efficient Semiring-Weighted Earley Parsing.](http://arxiv.org/abs/2307.02982) | 本文提出了高效的半环加权Earley分析算法，用于解决自然语言处理中大规模语法的问题，并提供了多种加速方法，包括对语法的预处理和推理循环的消除。 |
| [^17] | [On the Cultural Gap in Text-to-Image Generation.](http://arxiv.org/abs/2307.02971) | 该论文研究文本到图像生成中的文化差异，并提出了一个具有挑战性的跨文化基准，通过分析已有模型在该基准上生成的有缺陷的图像，提出了使用对象-文本对齐的多模态度量来优化跨文化模型的微调数据。 |
| [^18] | [LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias.](http://arxiv.org/abs/2307.02912) | 本论文提出了LEA模块，用于提高对打字错误的句子相似性鲁棒性。该模块通过引入词汇相似性来解决文本噪音问题，并避免了打字错误导致的标记分布偏移。 |
| [^19] | [Agentivit\`a e telicit\`a in GilBERTo: implicazioni cognitive.](http://arxiv.org/abs/2307.02910) | 本研究使用基于Transformer的神经语言模型，并通过与意大利本土讲者的对比实验，探究了模型是否能推断词汇语义和完成形态句法模式。研究结果表明模型在捕捉人类语义能力方面取得了一定的成果。 |
| [^20] | [The Relationship Between Speech Features Changes When You Get Depressed: Feature Correlations for Improving Speed and Performance of Depression Detection.](http://arxiv.org/abs/2307.02892) | 本研究发现抑郁症会改变从语音中提取的特征之间的相关性，同时利用这种洞察力可以通过改进特征相关性来提高抑郁症检测器的训练速度和性能。 |
| [^21] | [Contrast Is All You Need.](http://arxiv.org/abs/2307.02882) | 对比学习方法在数据稀缺的法律分类场景中表现更好，使用SetFit微调的模型比普通微调使用更少的训练样本。LIME的结果显示，对比学习方法有助于提升对正面和负面特征的认知，这些特征在法律上具有信息量，并对分类结果有贡献。 |
| [^22] | [ValiTex -- a uniform validation framework for computational text-based measures of social science constructs.](http://arxiv.org/abs/2307.02863) | ValiTex是一个统一的验证框架，旨在帮助学者们基于文本数据来度量社会科学构建。它借鉴了心理测量学的传统，通过概念模型和动态检查表提供了验证的结构和步骤。 |
| [^23] | [NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic.](http://arxiv.org/abs/2307.02849) | NatLogAttack是一个用自然逻辑对自然语言推理模型进行系统性攻击的框架，它可以进行保持标签和翻转标签的攻击，并相比现有攻击模型产生了更好的对抗性攻击。 |
| [^24] | [Enhancing LLM with Evolutionary Fine Tuning for News Summary Generation.](http://arxiv.org/abs/2307.02839) | 本论文提出一种新的方法使用LLM进行新闻摘要生成，通过进化调优事件模式群体，提高生成结果的准确性和可靠性。 |
| [^25] | [Generative Zero-Shot Prompt Learning for Cross-Domain Slot Filling with Inverse Prompting.](http://arxiv.org/abs/2307.02830) | 本文提出了一种生成式零样本提示学习框架，针对跨领域槽填充问题，通过引入逆向提示策略和高效的提示调整策略，改进了泛化性和鲁棒性，并在未见槽上取得了显著提升。 |
| [^26] | [VerifAI: Verified Generative AI.](http://arxiv.org/abs/2307.02796) | 验证生成式人工智能的输出是一个新兴问题，我们提出了通过分析多模态数据湖的底层数据，评估其质量和一致性，来建立评估生成式人工智能模型输出的更坚实基础，并解决错误信息传播的挑战。 |
| [^27] | [What Should Data Science Education Do with Large Language Models?.](http://arxiv.org/abs/2307.02792) | 大型语言模型（LLM）正在改变数据科学家的责任和数据科学教育模式，从动手编码和标准分析转变为评估和管理自动化AI执行的分析。这种转变要求数据科学教育注重培养学生的多样化技能，如创造力、批判性思维和AI引导的编程。 |
| [^28] | [Training Models to Generate, Recognize, and Reframe Unhelpful Thoughts.](http://arxiv.org/abs/2307.02768) | 本研究提出了一个新数据集PATTERNREFRAME，用于训练和评估语言模型，探讨如何生成大量的个性化练习材料和建议，用于识别和重构无益思维模式。 |
| [^29] | [Your spouse needs professional help: Determining the Contextual Appropriateness of Messages through Modeling Social Relationships.](http://arxiv.org/abs/2307.02763) | 本文介绍了一种通过建模个体之间的社交关系来准确识别给定上下文中适宜性的新方法，并探讨了关系本身如何作为暗含规范的功能以及在不同对话设置中上下文敏感性的程度。 |
| [^30] | [PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations.](http://arxiv.org/abs/2307.02762) | 本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。 |
| [^31] | [Exploring Linguistic Style Matching in Online Communities: The Role of Social Context and Conversation Dynamics.](http://arxiv.org/abs/2307.02758) | 本研究在Reddit上分析了两方对话主题的大量语料库，研究了LSM在对话中的差异以及与社区指标的关系，揭示了理解社区动态时对话参与的重要性。 |
| [^32] | [Dense Retrieval Adaptation using Target Domain Description.](http://arxiv.org/abs/2307.02740) | 本文提出了一种新的信息检索领域适应方法，该方法假设检索模型无法访问目标文档集，但可以访问描述目标领域的简要文本描述。 |
| [^33] | [RecallM: An Architecture for Temporal Context Understanding and Question Answering.](http://arxiv.org/abs/2307.02738) | 本文介绍了一种名为RecallM的架构，用于创建可适应和可更新的长期记忆，以提升大型语言模型聊天机器人的时间理解能力。 |
| [^34] | [Text Alignment Is An Efficient Unified Model for Massive NLP Tasks.](http://arxiv.org/abs/2307.02729) | 本研究提出了一种高效的文本对齐模型，可以应用于广泛的NLP任务，包括文本蕴含、相似性、问答、事实一致性等。通过对RoBERTa进行轻量级微调，可以构建一个更小规模的模型，实现与大型语言模型相当甚至更优的性能。 |
| [^35] | [On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation.](http://arxiv.org/abs/2307.02720) | 这项研究提出了一种基于知识蒸馏的自监督语音表示学习架构，以应用于设备内关键词检测。通过在有限的资源情况下将知识从复杂模型传递给轻量级模型，该方法在关键词检测任务中表现出了出色的性能。 |
| [^36] | [CFSum: A Coarse-to-Fine Contribution Network for Multimodal Summarization.](http://arxiv.org/abs/2307.02716) | CFSum是一个用于多模态摘要的粗到细贡献网络，能够准确计算和利用图像在摘要中的不同贡献，实验结果显示其优于其他基准方法。 |
| [^37] | [Statistical Mechanics of Strahler Number via Random and Natural Language Sentences.](http://arxiv.org/abs/2307.02697) | 本文通过统计力学分析自然语言句子树结构的Strahler数的上下限，发现它几乎总是3或4，并证明它是处理句子所需记忆量的下限。同时，对随机树进行的分析揭示出Strahler数的增长模式，揭示了它作为自然语言句子特征的统计基础。 |
| [^38] | [Scaling In-Context Demonstrations with Structured Attention.](http://arxiv.org/abs/2307.02690) | 本研究提出了一种用于上下文学习的结构化注意力机制，解决了大规模语言模型在使用演示进行上下文学习时遇到的限制与挑战。 |
| [^39] | [Learning Symbolic Rules over Abstract Meaning Representations for Textual Reinforcement Learning.](http://arxiv.org/abs/2307.02689) | 本文提出了一个基于神经符号方法的文本强化学习代理NESTA，通过学习抽象可解释的规则作为策略，在文本游戏中取得了比深度强化学习方法更好的泛化和学习效果。 |
| [^40] | [Zero-Shot Dense Video Captioning by Jointly Optimizing Text and Moment.](http://arxiv.org/abs/2307.02682) | 通过在训练阶段不使用视频和标注，而是在测试时仅优化输入，我们提出了一种零样本的密集视频字幕生成方法。通过联合优化文本和时刻，我们的方法能够在视频中准确地定位和描述事件。 |
| [^41] | [Unsupervised Sentiment Analysis of Plastic Surgery Social Media Posts.](http://arxiv.org/abs/2307.02640) | 该论文提出了一种无监督的方法来对社交媒体中的整形手术帖子进行情感分析。利用自然语言处理和深度学习模型，研究人员通过创建特征、学习和生成主题，实现了对大量文本数据的自动分析。 |
| [^42] | [SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference.](http://arxiv.org/abs/2307.02628) | 本文提出了一种名为SkipDecode的简单而有效的标记级早期退出方法，能够与批量推理和KV缓存无缝配合，解决了传统方法在这些方面的限制。 |
| [^43] | [Human Inspired Progressive Alignment and Comparative Learning for Grounded Word Acquisition.](http://arxiv.org/abs/2307.02615) | 本研究通过比较学习和渐进对齐的方式，借鉴人类语言习得的过程，探索了一种用于基于经验的词汇获取的计算过程。该方法不涉及固定的词汇量大小，也不涉及有区分性的目标，能够高效地持续学习更多的概念。 |
| [^44] | [Evade ChatGPT Detectors via A Single Space.](http://arxiv.org/abs/2307.02599) | 本研究发现，当前的ChatGPT检测器不能有效区分人类生成和AI生成内容之间的差异，而一个额外的空格成为了规避检测的关键因素。 |
| [^45] | [ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection.](http://arxiv.org/abs/2307.02591) | 这个研究介绍了一份名为ODD的新型基准数据集，用于通过分析患者的电子健康记录笔记，检测和分类药物滥用异常行为。这个数据集在药物相关病例的自然语言处理研究中具有重要的创新和贡献。 |
| [^46] | [Named Entity Inclusion in Abstractive Text Summarization.](http://arxiv.org/abs/2307.02570) | 该论文提出了一种解决抽象文本摘要中命名实体遗漏问题的方法，通过使用定制的预训练目标和模型训练策略，改善了命名实体的包含情况，提高了摘要的准确性和召回率。 |
| [^47] | [Analyzing the Performance of ChatGPT in Cardiology and Vascular Pathologies.](http://arxiv.org/abs/2307.02518) | ChatGPT是由OpenAI开发的大型语言模型，该研究分析了ChatGPT在心脏病和血管病理学中的表现。结果表明，ChatGPT在回答挑战性多项选择题方面表现优于医学生，显示了ChatGPT在提供准确答案方面具有潜在的高效性。 |
| [^48] | [Natural Language Generation and Understanding of Big Code for AI-Assisted Programming: A Review.](http://arxiv.org/abs/2307.02503) | 这项综述回顾了大型代码训练的transformer-based大语言模型（LLMs）在AI辅助编程方面的应用，包括代码生成、代码摘要、缺陷检测等。同时讨论了将NLP技术与软件自然化相结合的挑战和机会。 |
| [^49] | [Math Agents: Computational Infrastructure, Mathematical Embedding, and Genomics.](http://arxiv.org/abs/2307.02502) | 本文提出了数学智能体和数学嵌入作为解决生成式人工智能在基因组学应用方面的局限性的新方法，通过使用基于GPT的工作流将文献中的方程转换为LaTeX和Python格式，以实现自动化的大规模评估和交互式计算。 |
| [^50] | [mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding.](http://arxiv.org/abs/2307.02499) | mPLUG-DocOwl是一种模块化多模态大型语言模型，用于无OCR文档理解。它通过联合训练语言、通用视觉-语言和文档指令调优数据集，提升了无OCR文档理解能力。 |
| [^51] | [Deductive Additivity for Planning of Natural Language Proofs.](http://arxiv.org/abs/2307.02472) | 本论文研究了自然语言证明规划中的演绎可加性，探讨了是否能够通过嵌入空间实现高效的规划启发式方法。研究结果表明，嵌入空间的前提陈述总和接近于基于这些前提的结论嵌入。从而证明了演绎可加性的存在。 |
| [^52] | [Utilizing ChatGPT Generated Data to Retrieve Depression Symptoms from Social Media.](http://arxiv.org/abs/2307.02313) | 本文通过利用ChatGPT生成的数据，从社交媒体中检索和排名传达抑郁症状的句子。采用语义搜索和余弦相似度进行句子与BDI-II症状的相关性排名。结果显示合成数据比BDI-II响应更丰富和语义多样化，能有效增加数据和微调下游模型。 |
| [^53] | [Transformed Protoform Reconstruction.](http://arxiv.org/abs/2307.01896) | 该论文介绍了一个采用Transformer的转换的原型重构模型，相比于基于RNN的编码器-解码器模型，在拉丁语和汉语两个数据集上取得了更好的性能，并探索了模型中潜在的系统发育信号。 |
| [^54] | [Align With Purpose: Optimize Desired Properties in CTC Models with a General Plug-and-Play Framework.](http://arxiv.org/abs/2307.01715) | 本文提出了一个通用的插入式框架，用于优化CTC模型中的所需属性。该框架通过补充额外的损失项来优先考虑符合所需属性的对齐，并不需要修改CTC损失函数。 |
| [^55] | [Image Matters: A New Dataset and Empirical Study for Multimodal Hyperbole Detection.](http://arxiv.org/abs/2307.00209) | 本研究提出了一个新的多模态夸张检测数据集，并使用文本和图像作为两种模态进行研究。同时，评估了不同预训练的多模态编码器在此任务中的表现。该研究探索了夸张检测的跨领域性能。 |
| [^56] | [Biomedical Language Models are Robust to Sub-optimal Tokenization.](http://arxiv.org/abs/2306.17649) | 生物医学语言模型对生物医学术语的标记分割方式具有鲁棒性，这对于改进下游生物医学自然语言处理任务的性能非常重要。 |
| [^57] | [Automatic Calibration and Error Correction for Large Language Models via Pareto Optimal Self-Supervision.](http://arxiv.org/abs/2306.16564) | 本文介绍了一种Pareto Optimal自监督框架，利用可用的编程监督将大型语言模型(LLM)的响应进行系统校准，通过为每个响应生成风险评分，而无需额外的手动工作。 |
| [^58] | [The Singing Voice Conversion Challenge 2023.](http://arxiv.org/abs/2306.14422) | 2023年唱声转换挑战赛（SVCC）的最新版本旨在比较和了解不同的唱声转换系统。通过大规模听力测试，我们发现虽然顶尖系统达到了接近人类水平的自然度，但未能达到目标发音者的相似度评分。跨域SVC比域内SVC更困难。 |
| [^59] | [Chinese Fine-Grained Financial Sentiment Analysis with Large Language Models.](http://arxiv.org/abs/2306.14096) | 本文提出了一个用于企业预警的新型、广泛的中文细粒度金融情感分析数据集FinChina SA，并使用现有开源大语言模型对其进行评估和实验。该数据集将成为推进真实金融情感分析任务探索的宝贵资源。 |
| [^60] | [The Impact of ChatGPT and LLMs on Medical Imaging Stakeholders: Perspectives and Use Cases.](http://arxiv.org/abs/2306.06767) | 本研究调查了ChatGPT和LLMs在医学影像领域的变革潜力，它们正在增强放射科医生的解释能力、提升患者与医生之间的沟通，以及简化临床工作流程。 |
| [^61] | [Evaluation of ChatGPT on Biomedical Tasks: A Zero-Shot Comparison with Fine-Tuned Generative Transformers.](http://arxiv.org/abs/2306.04504) | 本文评估了ChatGPT在生物医学任务上的表现，发现在生物数据集训练样本较小时，零样例ChatGPT甚至优于精调生成式变压器模型。由此表明ChatGPT具有在生物医学领域成为有价值工具的潜力。 |
| [^62] | [A Systematic Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets.](http://arxiv.org/abs/2305.18486) | 本文对基准数据集上 ChatGPT 的性能进行了全面的评估，包括问答、文本摘要、代码生成、常识推理、数学问题求解、机器翻译、偏见检测和伦理考虑等任务。研究旨在验证 ChatGPT 的优势和弱点，并为使用语言模型的未来研究提供见解。 |
| [^63] | [Calibration of Transformer-based Models for Identifying Stress and Depression in Social Media.](http://arxiv.org/abs/2305.16797) | 本文提出了第一个使用校准后的Transformer模型来检测社交媒体上的压力和抑郁症状的研究。 |
| [^64] | [Self-supervised representations in speech-based depression detection.](http://arxiv.org/abs/2305.12263) | 本文使用基于自监督学习的预训练基础模型解决了语音交流中自动抑郁症检测训练数据稀疏性的问题，并通过微调基础模型将自动语音识别和情感识别的知识转移到抑郁症检测中。实验结果表明，在DAIC-WOZ数据集上实现了基于真实ASR的最先进的抑郁症检测性能。 |
| [^65] | [From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models.](http://arxiv.org/abs/2305.08283) | 本文研究测量了政治偏见在预训练语言模型和下游任务中的影响，发现预训练模型存在政治倾向，并将社会偏见传递到下游任务中，从而导致NLP模型的不公平性。 |
| [^66] | [Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models.](http://arxiv.org/abs/2305.01645) | 本文研究了压缩模型的高效微调方法，实验结果表明，与手动标注更多的微调数据以直接训练压缩模型相比，从T5-XXL蒸馏到T5-Small几乎总是更具成本效益。 |
| [^67] | [Unstructured and structured data: Can we have the best of both worlds with large language models?.](http://arxiv.org/abs/2304.13010) | 本文探讨使用大型语言模型查询无结构数据和结构化数据的潜力及挑战。 |
| [^68] | [Generation of Highlights from Research Papers Using Pointer-Generator Networks and SciBERT Embeddings.](http://arxiv.org/abs/2302.07729) | 该论文提出了一种使用指针生成网络和SciBERT嵌入来自动生成研究论文亮点的方法。在多个基准数据集上的实验证明，该模型在研究亮点生成方面具有最佳性能。 |
| [^69] | [Computer says "No": The Case Against Empathetic Conversational AI.](http://arxiv.org/abs/2212.10983) | 论文提出反对对话型人工智能过度同理心回应用户情绪的观点，强调需谨慎考虑对用户情绪的回应方式。 |
| [^70] | [Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model.](http://arxiv.org/abs/2212.09811) | 本研究提出了一种节约内存的NLLB-200模型修剪方法，可在保持翻译质量的同时移除多达80％的专家，使得在单个32GB的GPU上运行模型成为可能。这对于大规模多语言机器翻译具有重要的意义。 |
| [^71] | [An Effective Employment of Contrastive Learning in Multi-label Text Classification.](http://arxiv.org/abs/2212.00552) | 本论文提出了五个新的对比损失函数，用于多标签文本分类任务。通过对比学习技术的应用，探索了其在多标签文本分类任务中的有效性，并提供了一套基准模型。 |
| [^72] | [A Weakly-Supervised Streaming Multilingual Speech Model with Truly Zero-Shot Capability.](http://arxiv.org/abs/2211.02499) | 本文介绍了一个弱监督流式多语言语音模型，利用机器翻译服务将语音识别转录转化为弱监督数据来训练模型。该模型具有真正的零-shot能力，可以在扩展到新的目标语言时产生高质量的语音翻译结果。 |
| [^73] | [A Chinese Spelling Check Framework Based on Reverse Contrastive Learning.](http://arxiv.org/abs/2210.13823) | 提出了一种基于逆对比学习的框架用于中文拼写检查，通过增强模型对混淆词的区分能力，有效提高了检测和纠正的准确性和能力。 |
| [^74] | [Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest.](http://arxiv.org/abs/2209.06293) | 通过纽约客漫画字幕比赛的任务，我们挑战了AI模型对幽默的“理解”。结果发现，无论是多模态模型还是仅文本模型，在配对笑话和漫画、识别获胜字幕以及解释获胜字幕为什么有趣的任务上都存在困难。 |
| [^75] | [A Survey on Non-Autoregressive Generation for Neural Machine Translation and Beyond.](http://arxiv.org/abs/2204.09269) | 这项调查研究了非自回归生成在神经机器翻译以及其他领域的应用。研究发现，尽管非自回归生成可以加快推理速度，但与自回归生成相比存在翻译准确性的损失。然而，通过各种方法和算法的改进，可以缩小这一准确性差距。 |

# 详细

[^1]: 迷失在中间：语言模型如何使用长文本

    Lost in the Middle: How Language Models Use Long Contexts. (arXiv:2307.03172v1 [cs.CL])

    [http://arxiv.org/abs/2307.03172](http://arxiv.org/abs/2307.03172)

    本研究分析了语言模型在多文档问答和键值检索任务中的表现，发现当相关信息位于输入文本的开头或结尾时性能最佳，而当模型需要在长文本的中间访问相关信息时性能显著下降。此外，即使对于专门处理长文本的模型，输入文本越长性能也会大幅降低。我们的研究为理解语言模型如何使用输入文本的上下文提供了新的认识，并且为未来的长文本模型提供了新的评估方案。

    

    尽管最近的语言模型能够将长文本作为输入，但我们对语言模型如何有效地使用较长的文本还知之甚少。本研究分析了语言模型在两个需要在输入文本中识别相关信息的任务（多文档问答和键值检索）上的表现。我们发现，当相关信息出现在输入文本的开头或结尾时，语言模型的表现通常最佳；而当模型需要访问长文本中的中间相关信息时，性能显著下降。此外，即使对于专门处理长文本的模型，当输入文本变得越来越长时，性能也会大幅降低。我们的分析为我们更好地理解语言模型如何使用输入文本的上下文，并为未来的长文本模型提供了新的评估方案。

    While recent language models have the ability to take long contexts as input, relatively little is known about how well the language models use longer context. We analyze language model performance on two tasks that require identifying relevant information within their input contexts: multi-document question answering and key-value retrieval. We find that performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts. Furthermore, performance substantially decreases as the input context grows longer, even for explicitly long-context models. Our analysis provides a better understanding of how language models use their input context and provides new evaluation protocols for future long-context models.
    
[^2]: Focused Transformer: 反差训练对上下文缩放进行优化

    Focused Transformer: Contrastive Training for Context Scaling. (arXiv:2307.03170v1 [cs.CL])

    [http://arxiv.org/abs/2307.03170](http://arxiv.org/abs/2307.03170)

    Focused Transformer通过反差训练优化了上下文缩放问题，允许语言模型处理更长的上下文信息。

    

    大规模语言模型能够以上下文化的方式吸纳新的信息，但由于有效上下文长度的限制，这种方法的潜力通常受到限制。解决这个问题的一种方法是为注意力层提供访问外部存储器的能力，该存储器由（键，值）对组成。然而，随着文档数量的增加，相关键与无关键的比例减少，使模型更加关注无关键。我们发现了一个名为分心问题的重要挑战，即与不同语义值相关联的键可能重叠，使它们难以区分。为了解决这个问题，我们引入了Focused Transformer（FoT），一种受对比学习启发的训练方法。这种新颖的方法增强了（键，值）空间的结构，使上下文长度得以扩展。我们的方法允许对现有大型模型进行微调，以更好地处理长上下文。

    Large language models have an exceptional capability to incorporate new information in a contextual manner. However, the full potential of such an approach is often restrained due to a limitation in the effective context length. One solution to this issue is to endow an attention layer with access to an external memory, which comprises of (key, value) pairs. Yet, as the number of documents increases, the proportion of relevant keys to irrelevant ones decreases, leading the model to focus more on the irrelevant keys. We identify a significant challenge, dubbed the distraction issue, where keys linked to different semantic values might overlap, making them hard to distinguish. To tackle this problem, we introduce the Focused Transformer (FoT), a technique that employs a training process inspired by contrastive learning. This novel approach enhances the structure of the (key, value) space, enabling an extension of the context length. Our method allows for fine-tuning pre-existing, large-s
    
[^3]: 用于超出分布可泛化性的大型视觉语言模型压缩

    Distilling Large Vision-Language Model with Out-of-Distribution Generalizability. (arXiv:2307.03135v1 [cs.CV])

    [http://arxiv.org/abs/2307.03135](http://arxiv.org/abs/2307.03135)

    本文研究了针对大型视觉语言模型的模型压缩方法，将教师模型的视觉表示压缩到学生模型中。研究重点在于超出分布可泛化的问题，并提出了两个原则来增强学生模型的性能。

    

    大型视觉语言模型取得了出色的性能，但其规模和计算要求使它们在资源受限设备和时间敏感任务上的部署变得不切实际。模型压缩是创建更小、更快的模型以保持较大模型性能的有希望的方法。本文研究了将大型视觉语言模型中的视觉表示压缩到轻量级学生模型中的过程，使用小型或中型数据集。值得注意的是，本研究关注的是超出分布（OOD）可泛化的开放词汇问题，这在以往的模型压缩研究中被忽视了。我们从视觉和语言的角度提出了两个原则来增强学生模型的OOD可泛化性：（1）更好地模仿教师的视觉表示空间，并在视觉语言对齐方面谨慎地促进更好的一致性；（2）通过丰富学生模型的自举学习和数据扩充来提高OOD可泛化性。

    Large vision-language models have achieved outstanding performance, but their size and computational requirements make their deployment on resource-constrained devices and time-sensitive tasks impractical. Model distillation, the process of creating smaller, faster models that maintain the performance of larger models, is a promising direction towards the solution. This paper investigates the distillation of visual representations in large teacher vision-language models into lightweight student models using a smallor mid-scale dataset. Notably, this study focuses on open-vocabulary out-of-distribution (OOD) generalization, a challenging problem that has been overlooked in previous model distillation literature. We propose two principles from vision and language modality perspectives to enhance student's OOD generalization: (1) by better imitating teacher's visual representation space, and carefully promoting better coherence in vision-language alignment with the teacher; (2) by enric
    
[^4]: T-MARS：通过规避文本特征学习来改善视觉表示

    T-MARS: Improving Visual Representations by Circumventing Text Feature Learning. (arXiv:2307.03132v1 [cs.CV])

    [http://arxiv.org/abs/2307.03132](http://arxiv.org/abs/2307.03132)

    T-MARS提出一种新的数据筛选方法，通过规避文本特征学习，改善了视觉表示的学习，解决了大型多模态数据集中存在的文本与图像重叠的问题。

    

    大型网络来源的多模态数据集为学习通用视觉表示的新方法提供了动力，推动了计算机视觉的最新发展，并彻底改变了零样本和少样本识别。一个关键的决策问题是如何筛选这些日益庞大的数据集。本文提出了一种新的最先进的数据筛选方法，其动机是我们观察到近40%的LAION数据集的图像与说明存在重叠的文本。直觉上，这样的数据可能会浪费资源，因为它鼓励模型进行光学字符识别而不是学习视觉特征。然而，简单地将所有这些数据去除也可能浪费，因为这会丢弃包含视觉特征的图像（除了重叠的文本）。我们提出了一种简单而可扩展的方法来解决这个问题。

    Large web-sourced multimodal datasets have powered a slew of new methods for learning general-purpose visual representations, advancing the state of the art in computer vision and revolutionizing zero- and few-shot recognition. One crucial decision facing practitioners is how, if at all, to curate these ever-larger datasets. For example, the creators of the LAION-5B dataset chose to retain only image-caption pairs whose CLIP similarity score exceeded a designated threshold. In this paper, we propose a new state-of-the-art data filtering approach motivated by our observation that nearly 40% of LAION's images contain text that overlaps significantly with the caption. Intuitively, such data could be wasteful as it incentivizes models to perform optical character recognition rather than learning visual features. However, naively removing all such data could also be wasteful, as it throws away images that contain visual features (in addition to overlapping text). Our simple and scalable app
    
[^5]: BLEURT具有通用翻译能力：基于最小风险训练的自动度量分析

    BLEURT Has Universal Translations: An Analysis of Automatic Metrics by Minimum Risk Training. (arXiv:2307.03131v1 [cs.CL])

    [http://arxiv.org/abs/2307.03131](http://arxiv.org/abs/2307.03131)

    本研究通过最小风险训练系统性地分析和比较了各种自动度量，并发现了BLEURT和BARTScore等度量中存在的通用对抗翻译现象。研究结果表明，这些鲁棒性缺陷主要由训练数据集中的分布偏差和度量范式的倾向引起。通过引入标记级约束，可以提高度量的鲁棒性。

    

    自动度量在机器翻译中起着关键作用。尽管n-gram度量广泛应用，但最近出现了基于预训练模型的度量的发展潮流，重点在于测量句子语义。然而，这些神经度量虽然与人工评估相关性更高，但常常被认为是带有潜在偏见且难以检测的黑盒子。本研究从训练机器翻译系统的指导角度，系统分析和比较了多种主流和前沿的自动度量。通过最小风险训练（MRT），我们发现某些度量存在鲁棒性缺陷，例如BLEURT和BARTScore中存在通用对抗翻译现象。深入分析表明，这些鲁棒性缺陷主要有两个原因：训练数据集中的分布偏差和度量范式的倾向。通过引入标记级约束，我们增强了度量的鲁棒性。

    Automatic metrics play a crucial role in machine translation. Despite the widespread use of n-gram-based metrics, there has been a recent surge in the development of pre-trained model-based metrics that focus on measuring sentence semantics. However, these neural metrics, while achieving higher correlations with human evaluations, are often considered to be black boxes with potential biases that are difficult to detect. In this study, we systematically analyze and compare various mainstream and cutting-edge automatic metrics from the perspective of their guidance for training machine translation systems. Through Minimum Risk Training (MRT), we find that certain metrics exhibit robustness defects, such as the presence of universal adversarial translations in BLEURT and BARTScore. In-depth analysis suggests two main causes of these robustness deficits: distribution biases in the training datasets, and the tendency of the metric paradigm. By incorporating token-level constraints, we enhan
    
[^6]: VisKoP：面向交互式知识库问答的视觉知识导向编程

    VisKoP: Visual Knowledge oriented Programming for Interactive Knowledge Base Question Answering. (arXiv:2307.03130v1 [cs.CL])

    [http://arxiv.org/abs/2307.03130](http://arxiv.org/abs/2307.03130)

    VisKoP是一种知识库问答（KBQA）系统，通过将人类融入到知识库查询的编辑和调试中，提供了一种视觉知识导向编程的平台。它不仅提供了神经程序归纳模块，还将程序映射为图形元素，使其易于编辑和调试。通过提供自动补全功能和高效的执行引擎，VisKoP在处理大规模知识库问答时表现出高效性和准确性。

    

    我们提出了一种名为VisKoP的视觉知识导向编程平台，它是一个知识库问答（KBQA）系统，将人类融入到知识库（KB）查询的编辑和调试中。VisKoP不仅提供了一个神经程序归纳模块，将自然语言问题转化为知识导向的程序语言（KoPL），还将KoPL程序映射为图形元素。KoPL程序可以通过简单的图形操作进行编辑，例如拖动以添加知识操作符和使用槽填充以指定操作符参数。此外，VisKoP还为其知识库模式提供了自动补全功能，并通过检查中间结果，用户可以轻松调试KoPL程序。为了便于在亿级别的实际知识库问答上进行，我们为后端设计了一个高效的KoPL执行引擎。实验结果显示，VisKoP非常高效，用户交互可以修复大部分错误的KoPL程序以获取正确的答案。

    We present Visual Knowledge oriented Programming platform (VisKoP), a knowledge base question answering (KBQA) system that integrates human into the loop to edit and debug the knowledge base (KB) queries. VisKoP not only provides a neural program induction module, which converts natural language questions into knowledge oriented program language (KoPL), but also maps KoPL programs into graphical elements. KoPL programs can be edited with simple graphical operators, such as dragging to add knowledge operators and slot filling to designate operator arguments. Moreover, VisKoP provides auto-completion for its knowledge base schema and users can easily debug the KoPL program by checking its intermediate results. To facilitate the practical KBQA on a million-entity-level KB, we design a highly efficient KoPL execution engine for the back-end. Experiment results show that VisKoP is highly efficient and user interaction can fix a large portion of wrong KoPL programs to acquire the correct ans
    
[^7]: 从语言模型中提取多值关系

    Extracting Multi-valued Relations from Language Models. (arXiv:2307.03122v1 [cs.CL])

    [http://arxiv.org/abs/2307.03122](http://arxiv.org/abs/2307.03122)

    该论文研究了从预训练语言模型中提取多值关系的问题，并通过排名和选择任务的方法解决了这个问题。结果表明，选择具有特定关系阈值以上的对象可以达到49.5%的F1得分，这对于将语言模型应用于多值槽位填充任务而言是具有挑战性的。该研究为从潜在语言表示中提取关系知识开辟了进一步研究的道路。

    

    广泛使用预训练语言模型（LMs）的潜在语言表示表明它们是一种有前景的结构化知识来源。然而，现有方法仅关注每个主题-关系对中的单个对象，尽管通常有多个对象是正确的。为了克服这个限制，我们分析这些表示以了解它们产生多对象关系知识的潜力。我们将该问题制定为一个排名-选择任务。对于排名候选对象，我们评估现有的提示技术并提出了融入领域知识的新技术。在选择方法中，我们发现选择具有高于学习到的关系特定阈值的对象可以达到49.5%的F1得分。我们的结果突显了使用LMs进行多值槽位填充任务的困难，并为从潜在语言表示中提取关系知识的进一步研究铺平了道路。

    The widespread usage of latent language representations via pre-trained language models (LMs) suggests that they are a promising source of structured knowledge. However, existing methods focus only on a single object per subject-relation pair, even though often multiple objects are correct. To overcome this limitation, we analyze these representations for their potential to yield materialized multi-object relational knowledge. We formulate the problem as a rank-then-select task. For ranking candidate objects, we evaluate existing prompting techniques and propose new ones incorporating domain knowledge. Among the selection methods, we find that choosing objects with a likelihood above a learned relation-specific threshold gives a 49.5% F1 score. Our results highlight the difficulty of employing LMs for the multi-valued slot-filling task and pave the way for further research on extracting relational knowledge from latent language representations.
    
[^8]: KoRC: 面向深度文本理解的知识导向阅读理解基准

    KoRC: Knowledge oriented Reading Comprehension Benchmark for Deep Text Understanding. (arXiv:2307.03115v1 [cs.CL])

    [http://arxiv.org/abs/2307.03115](http://arxiv.org/abs/2307.03115)

    近年来，深度文本理解的重要性在许多基准测试中得到了强调。为了克服已有基准测试的限制，本论文提出了一个新的基准测试，KoRC，在知识覆盖和答案格式上具有优势。实验结果表明，KoRC可以帮助改进文本理解模型的性能。

    

    近年来，许多基准测试都强调了深度文本理解对于给定文档和文本以外的先验知识之间的联系的需求。然而，这些基准测试遇到了两个主要的限制。一方面，大多数基准测试需要人工注释知识，导致了知识覆盖的限制。另一方面，它们通常使用文本中的选项或跨度作为答案，导致了狭窄的答案空间。为克服这些限制，我们在本文中建立了一个新的具有挑战性的基准测试，名为KoRC。与先前的基准测试相比，KoRC具有两个优点，即广泛的知识覆盖和灵活的答案格式。具体来说，我们利用海量知识库来指导注释者或大型语言模型（LLM）构建有见地的问题。此外，我们使用知识库中的标签作为最终答案，而不是跨度或选项。我们在KoRC上测试了最先进的模型，实验结果表明

    Deep text understanding, which requires the connections between a given document and prior knowledge beyond its text, has been highlighted by many benchmarks in recent years. However, these benchmarks have encountered two major limitations. On the one hand, most of them require human annotation of knowledge, which leads to limited knowledge coverage. On the other hand, they usually use choices or spans in the texts as the answers, which results in narrow answer space. To overcome these limitations, we build a new challenging benchmark named KoRc in this paper. Compared with previous benchmarks, KoRC has two advantages, i.e., broad knowledge coverage and flexible answer format. Specifically, we utilize massive knowledge bases to guide annotators or large language models (LLMs) to construct knowledgable questions. Moreover, we use labels in knowledge bases rather than spans or choices as the final answers. We test state-of-the-art models on KoRC and the experimental results show that the
    
[^9]: 对大型语言模型评估的调查

    A Survey on Evaluation of Large Language Models. (arXiv:2307.03109v1 [cs.CL])

    [http://arxiv.org/abs/2307.03109](http://arxiv.org/abs/2307.03109)

    本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。

    

    大型语言模型（LLMs）由于在各种应用中表现出的前所未有的性能而在学术界和工业界越来越受欢迎。随着LLMs在研究和日常使用中继续发挥着重要作用，它们的评估变得越来越关键，不仅在任务水平上，而且在社会层面上，以更好地了解它们的潜在风险。在过去的几年里，已经做出了相当大的努力来从不同的角度来研究LLMs。本文综述了LLMs的这些评估方法，重点关注三个关键维度：评估什么、在哪里评估以及如何评估。首先，我们从评估任务的角度提供了一个概述，涵盖了一般的自然语言处理任务、推理、医学应用、伦理学、教育、自然科学和社会科学、代理应用和其他领域。其次，我们通过深入探讨评估方法和基准答案来回答“在哪里”和“如何”这两个问题。

    Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level, but also at the society level for better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives. This paper presents a comprehensive review of these evaluation methods for LLMs, focusing on three key dimensions: what to evaluate, where to evaluate, and how to evaluate. Firstly, we provide an overview from the perspective of evaluation tasks, encompassing general natural language processing tasks, reasoning, medical usage, ethics, educations, natural and social sciences, agent applications, and other areas. Secondly, we answer the `where' and `how' questions by diving into the evaluation methods and bench
    
[^10]: 使用适配器高效域自适应句子嵌入

    Efficient Domain Adaptation of Sentence Embeddings using Adapters. (arXiv:2307.03104v1 [cs.CL])

    [http://arxiv.org/abs/2307.03104](http://arxiv.org/abs/2307.03104)

    本论文提出了一种通过训练轻量级适配器来高效域自适应句子嵌入的方法，避免了微调整个句子嵌入模型的资源消耗。通过训练特定领域的适配器，可以在不同领域中使用同一模型获得良好的性能。

    

    句子嵌入使我们能够捕捉短文本的语义相似性。大多数句子嵌入模型是针对一般语义文本相似性（STS）任务进行训练的。因此，要在特定领域中使用句子嵌入，必须将模型适应于该领域以获得良好的结果。通常，这是通过对感兴趣的域对整个句子嵌入模型进行微调来实现的。虽然这种方法能够产生最先进的结果，但在微调过程中更新了所有模型的权重，使该方法在资源上要求较高。因此，我们提出了训练轻量级适配器的方法，而不是单独为每个目标领域微调整个句子嵌入模型。这些特定领域的适配器不需要微调所有底层句子嵌入模型的参数。相反，我们只训练少量的额外参数，同时保持底层句子嵌入模型的权重不变。训练特定领域的适配器可以始终使用同一模型并在不同领域中获得良好的性能。

    Sentence embeddings enable us to capture the semantic similarity of short texts. Most sentence embedding models are trained for general semantic textual similarity (STS) tasks. Therefore, to use sentence embeddings in a particular domain, the model must be adapted to it in order to achieve good results. Usually, this is done by fine-tuning the entire sentence embedding model for the domain of interest. While this approach yields state-of-the-art results, all of the model's weights are updated during fine-tuning, making this method resource-intensive. Therefore, instead of fine-tuning entire sentence embedding models for each target domain individually, we propose to train lightweight adapters. These domain-specific adapters do not require fine-tuning all underlying sentence embedding model parameters. Instead, we only train a small number of additional parameters while keeping the weights of the underlying sentence embedding model fixed. Training domain-specific adapters allows always 
    
[^11]: OpenDelta: 一种用于参数高效调整预训练模型的即插即用库

    OpenDelta: A Plug-and-play Library for Parameter-efficient Adaptation of Pre-trained Models. (arXiv:2307.03084v1 [cs.LG])

    [http://arxiv.org/abs/2307.03084](http://arxiv.org/abs/2307.03084)

    OpenDelta是一个开源库，提供了各种delta调整方法的即插即用实现。它能够以高效的方式调整大型预训练模型的参数，而无需修改模型的代码，具有实用性和灵活性。

    

    大型预训练模型 (PTMs) 的规模给调整下游任务带来了重大挑战，原因是全参数微调涉及高昂的优化开销和存储成本。为了解决这个问题，许多研究探索了参数高效调整方法，也称为 "delta 调整"，即仅更新一小部分参数，称为 "delta 模块"，同时保持主干模型的参数固定。然而，由于现有实现直接修改主干 PTMs 的代码，并为每个 PTM 硬编码特定的 delta 调整方法，delta 调整的实用性和灵活性受到了限制。在本文中，我们提出了 OpenDelta，这是一个开源库，通过提供各种 delta 调整方法的即插即用实现来克服这些限制。我们的新技术消除了修改主干 PTMs 代码的需求，使 OpenDelta 可以与不同的、甚至是新的 PTMs 兼容。OpenDelta 的设计简单、可扩展，并且易于使用。

    The scale of large pre-trained models (PTMs) poses significant challenges in adapting to downstream tasks due to the high optimization overhead and storage costs associated with full-parameter fine-tuning. To address this, many studies explore parameter-efficient tuning methods, also framed as "delta tuning", which updates only a small subset of parameters, known as "delta modules", while keeping the backbone model's parameters fixed. However, the practicality and flexibility of delta tuning have been limited due to existing implementations that directly modify the code of the backbone PTMs and hard-code specific delta tuning methods for each PTM. In this paper, we present OpenDelta, an open-source library that overcomes these limitations by providing a plug-and-play implementation of various delta tuning methods. Our novel techniques eliminate the need to modify the backbone PTMs' code, making OpenDelta compatible with different, even novel PTMs. OpenDelta is designed to be simple, mo
    
[^12]: DeepOnto: 一个用于深度学习本体工程的Python包

    DeepOnto: A Python Package for Ontology Engineering with Deep Learning. (arXiv:2307.03067v1 [cs.AI])

    [http://arxiv.org/abs/2307.03067](http://arxiv.org/abs/2307.03067)

    DeepOnto是一个Python包，用于深度学习本体工程。它通过集成深度学习框架和本体API，提供了丰富的工具和算法，支持本体工程任务，如本体对齐和完成。

    

    应用深度学习技术，特别是语言模型（LMs），在本体工程中已经引起了广泛关注。然而，深度学习框架如PyTorch和Tensorflow主要是为Python开发的，而广泛使用的本体API（如OWL API和Jena）主要是基于Java的。为了方便无缝集成这些框架和API，我们提出了Deeponto，一个专为本体工程设计的Python包。该包包括一个基于广泛认可和可靠的OWL API的核心本体处理模块，以更“Pythonic”的方式封装其基本特性，并扩展其功能以包括其他重要组成部分，包括推理、语言化、规范化、投影等。基于这个模块，Deeponto提供了一套工具、资源和算法，支持各种本体工程任务，例如本体对齐和完成，利用深度学习方法实现。

    Applying deep learning techniques, particularly language models (LMs), in ontology engineering has raised widespread attention. However, deep learning frameworks like PyTorch and Tensorflow are predominantly developed for Python programming, while widely-used ontology APIs, such as the OWL API and Jena, are primarily Java-based. To facilitate seamless integration of these frameworks and APIs, we present Deeponto, a Python package designed for ontology engineering. The package encompasses a core ontology processing module founded on the widely-recognised and reliable OWL API, encapsulating its fundamental features in a more "Pythonic" manner and extending its capabilities to include other essential components including reasoning, verbalisation, normalisation, projection, and more. Building on this module, Deeponto offers a suite of tools, resources, and algorithms that support various ontology engineering tasks, such as ontology alignment and completion, by harnessing deep learning meth
    
[^13]: LLaMA在临床领域的参数高效微调

    Parameter-Efficient Fine-Tuning of LLaMA for the Clinical Domain. (arXiv:2307.03042v1 [cs.CL])

    [http://arxiv.org/abs/2307.03042](http://arxiv.org/abs/2307.03042)

    本研究提出了一种参数高效微调（PEFT）方法，在临床领域使用临床记录训练了一个专门适配临床领域的LLaMA-LoRA模型，同时提出了一个两步PEFT框架，用于将其与Downstream LLaMA-LoRA适配器进行融合，以实现领域适应。

    

    传统上，将预训练的语言模型适应到新领域，如临床应用，需要重新训练所有参数。然而，由于训练这些大型语言模型所需的计算资源巨大，这种方法的实践性越来越被证明是不切实际的。为了解决这个问题，参数高效微调（PEFT）技术提供了一种可行的解决方案，通过选择性地微调一个小的附加参数集，显著减少了领域适应所需的计算资源。在本研究中，我们提出了临床LLaMA-LoRA，这是一个构建在开源LLaMA模型上的PEFT适配器层。临床LLaMA-LoRA使用从MIMIC-IV数据库中获取的临床记录进行训练，从而创建了一个专为临床领域设计的专用适配器。此外，我们提出了一个两步PEFT框架，将临床LLaMA-LoRA与Downstream LLaMA-LoRA进行融合，后者是另一个专为下游任务设计的PEFT适配器。

    Adapting pretrained language models to novel domains, such as clinical applications, traditionally involves retraining their entire set of parameters. However, this approach is increasingly proven to be impractical owing to the substantial computational requirements associated with training such large language models. To address this issue, Parameter-Efficient Fine-Tuning (PEFT) techniques offer a viable solution by selectively fine-tuning a small subset of additional parameters, significantly reducing the computational requirements for domain adaptation. In this study, we propose Clinical LLaMA-LoRA, a PEFT adapter layer built upon the open-sourced LLaMA model. Clinical LLaMA-LoRA is trained using clinical notes obtained from the MIMIC-IV database, thereby creating a specialised adapter designed for the clinical domain. Additionally, we propose a two-step PEFT framework which fuses Clinical LLaMA-LoRA with Downstream LLaMA-LoRA, another PEFT adapter specialised for downstream tasks. W
    
[^14]: 通过数据重要性学习改善检索增强的大型语言模型

    Improving Retrieval-Augmented Large Language Models via Data Importance Learning. (arXiv:2307.03027v1 [cs.LG])

    [http://arxiv.org/abs/2307.03027](http://arxiv.org/abs/2307.03027)

    本文通过多线性扩展算法评估检索增强模型中检索到的数据点的数据重要性，并提出了一个多项式时间算法来计算其数据重要性。实验结果表明，修剪或增强大型语言模型可以提高性能。

    

    检索增强使得大型语言模型能够利用外部知识，例如在问题回答和数据补全等任务中。然而，这种检索增强模型的性能受到其基础检索语料的数据质量的限制。本文提出了一种基于多线性扩展的算法，用于评估检索到的数据点的数据重要性。多线性扩展中存在指数级的项，本文的一个关键贡献是提出了一个多项式时间算法，能够精确计算具有加法效用函数和验证集的检索增强模型中的数据点在检索语料中的数据重要性。我们还提出了一种更高效的（ε，δ）-近似算法。实验结果表明，我们可以通过仅修剪或增强大型语言模型来提高其性能。

    Retrieval augmentation enables large language models to take advantage of external knowledge, for example on tasks like question answering and data imputation. However, the performance of such retrieval-augmented models is limited by the data quality of their underlying retrieval corpus. In this paper, we propose an algorithm based on multilinear extension for evaluating the data importance of retrieved data points. There are exponentially many terms in the multilinear extension, and one key contribution of this paper is a polynomial time algorithm that computes exactly, given a retrieval-augmented model with an additive utility function and a validation set, the data importance of data points in the retrieval corpus using the multilinear extension of the model's utility function. We further proposed an even more efficient ({\epsilon}, {\delta})-approximation algorithm. Our experimental results illustrate that we can enhance the performance of large language models by only pruning or r
    
[^15]: 风格胜过实质：大型语言模型的评估偏见

    Style Over Substance: Evaluation Biases for Large Language Models. (arXiv:2307.03025v1 [cs.CL])

    [http://arxiv.org/abs/2307.03025](http://arxiv.org/abs/2307.03025)

    这项研究调查了人类和基于大型语言模型的评委在比较不同模型输出时的行为，并发现评估过程中存在偏见，即尽管包含事实错误，答案仍然被更高地评分。为了解决这个问题，我们提出了

    

    随着大型语言模型（LLMs）的不断进步，准确和全面评估它们的性能变得越来越具有挑战性。传统上，人类评估被认为是自然语言生成的黄金标准。最近的进展将最先进的LLMs纳入评估过程中，作为人类评委的代理。然而，人类和LLMs作为评估者的能力程度仍然不确定。本研究旨在研究众包人类评委和基于LLMs的评委在比较不同模型的输出时的行为。为了实现这一目标，我们收集了一个包含故意有缺陷的机器生成答案的数据集。我们的研究结果表明，尽管事实上的错误可能带来更大的危险，但带有事实错误的答案仍然比长度过短或包含语法错误的答案评分更高。这突显了评估过程中存在的令人担忧的偏见。为了解决这个问题，我们提出了

    As large language models (LLMs) continue to advance, accurately and comprehensively evaluating their performance becomes increasingly challenging. Conventionally, human evaluations are considered the gold standard in natural language generation. Recent advancements incorporate state-of-the-art LLMs as proxies for human judges in evaluation processes. Nonetheless, the extent to which humans and LLMs are capable evaluators remains uncertain. This study aims to investigate the behavior of both crowd-sourced human and LLM-based judges when comparing outputs from different models. To accomplish this, we curate a dataset comprising intentionally flawed machine-generated answers. Our findings indicate that despite the potentially greater danger posed by factual errors, answers with factual errors were still rated more favorably compared to answers that were too short or contained grammatical errors. This highlights a concerning bias in the evaluation process. To address this issue, we propose
    
[^16]: 高效的半环加权Earley分析

    Efficient Semiring-Weighted Earley Parsing. (arXiv:2307.02982v1 [cs.CL])

    [http://arxiv.org/abs/2307.02982](http://arxiv.org/abs/2307.02982)

    本文提出了高效的半环加权Earley分析算法，用于解决自然语言处理中大规模语法的问题，并提供了多种加速方法，包括对语法的预处理和推理循环的消除。

    

    本文提供了Earley (1970)的上下文无关语法分析算法的参考描述，包括多种加速方法。我们的演示包括从Earley的$O(N^3|G||R|)$到$O(N^3|G|)$的已知最坏情况运行时改进，后者可以解决自然语言处理中出现的大规模语法的问题。其中，$N$是句子的长度，$|R|$是$G$中的产生式数量，$|G|$是这些产生式的总长度。当以单个有限状态自动机$M$的紧凑表示方式表示语法时，我们还提供了运行时为$O(N^3|M|)$，其中$|M|\leq|G|$的版本（部分是新颖的）。我们仔细处理了半环加权推理的泛化问题，对语法进行预处理以消除推理循环，并进一步推广了Stolcke (1995)的方法以计算句子前缀的权重。

    This paper provides a reference description, in the form of a deduction system, of Earley's (1970) context-free parsing algorithm with various speed-ups. Our presentation includes a known worst-case runtime improvement from Earley's $O (N^3|G||R|)$, which is unworkable for the large grammars that arise in natural language processing, to $O (N^3|G|)$, which matches the runtime of CKY on a binarized version of the grammar $G$. Here $N$ is the length of the sentence, $|R|$ is the number of productions in $G$, and $|G|$ is the total length of those productions. We also provide a version that achieves runtime of $O (N^3|M|)$ with $|M| \leq |G|$ when the grammar is represented compactly as a single finite-state automaton $M$ (this is partly novel). We carefully treat the generalization to semiring-weighted deduction, preprocessing the grammar like Stolcke (1995) to eliminate deduction cycles, and further generalize Stolcke's method to compute the weights of sentence prefixes. We also provide
    
[^17]: 关于文本到图像生成中的文化差异

    On the Cultural Gap in Text-to-Image Generation. (arXiv:2307.02971v1 [cs.CV])

    [http://arxiv.org/abs/2307.02971](http://arxiv.org/abs/2307.02971)

    该论文研究文本到图像生成中的文化差异，并提出了一个具有挑战性的跨文化基准，通过分析已有模型在该基准上生成的有缺陷的图像，提出了使用对象-文本对齐的多模态度量来优化跨文化模型的微调数据。

    

    文本到图像（T2I）生成中的一个挑战是在训练数据中意外反映了文化差距，当输入文本的文化元素很少出现在训练集中时，这表明生成图像的质量差异。尽管各种T2I模型展示了令人印象深刻但是随意的例子，但是目前没有一个基准来系统评估T2I模型生成跨文化图像的能力。为了弥补这一差距，我们提出了一个具有综合评估标准的具有挑战性的跨文化（C3）基准，该基准可以评估模型对目标文化的适应性。通过分析在C3基准上由稳定扩散模型生成的有缺陷的图像，我们发现该模型经常无法生成特定的文化对象。因此，我们提出了一种考虑对象与文本对齐的新型多模态度量，用于过滤目标文化中的微调数据，用于优化跨文化能力的T2I模型。

    One challenge in text-to-image (T2I) generation is the inadvertent reflection of culture gaps present in the training data, which signifies the disparity in generated image quality when the cultural elements of the input text are rarely collected in the training set. Although various T2I models have shown impressive but arbitrary examples, there is no benchmark to systematically evaluate a T2I model's ability to generate cross-cultural images. To bridge the gap, we propose a Challenging Cross-Cultural (C3) benchmark with comprehensive evaluation criteria, which can assess how well-suited a model is to a target culture. By analyzing the flawed images generated by the Stable Diffusion model on the C3 benchmark, we find that the model often fails to generate certain cultural objects. Accordingly, we propose a novel multi-modal metric that considers object-text alignment to filter the fine-tuning data in the target culture, which is used to fine-tune a T2I model to improve cross-cultural g
    
[^18]: LEA: 使用词汇注意偏差提高对打字错误的句子相似性鲁棒性

    LEA: Improving Sentence Similarity Robustness to Typos Using Lexical Attention Bias. (arXiv:2307.02912v1 [cs.CL])

    [http://arxiv.org/abs/2307.02912](http://arxiv.org/abs/2307.02912)

    本论文提出了LEA模块，用于提高对打字错误的句子相似性鲁棒性。该模块通过引入词汇相似性来解决文本噪音问题，并避免了打字错误导致的标记分布偏移。

    

    文本噪音，如打字错误或缩写，是一个众所周知的问题，会对大多数下游任务中的纯变压器模型造成惩罚。我们展示了这也适用于句子相似性，这是多个领域中的一个基本任务，比如匹配、检索或释义。可以使用交叉编码器来处理句子相似性，其中两个句子在输入中连接，使模型能够利用它们之间的相互关系。之前解决噪音问题的工作主要依赖于数据增强策略，展示了在处理与训练样本相似的损坏样本时性能有所提升。然而，所有这些方法仍然受到打字错误引起的标记分布偏移的影响。在这项工作中，我们提出使用一种新颖的词汇感知注意模块（LEA）来解决文本噪音问题，该模块在两个句子中的词之间引入了词汇相似性。通过使用原始文本相似性，我们的方法避免了token分布偏移的问题。

    Textual noise, such as typos or abbreviations, is a well-known issue that penalizes vanilla Transformers for most downstream tasks. We show that this is also the case for sentence similarity, a fundamental task in multiple domains, e.g. matching, retrieval or paraphrasing. Sentence similarity can be approached using cross-encoders, where the two sentences are concatenated in the input allowing the model to exploit the inter-relations between them. Previous works addressing the noise issue mainly rely on data augmentation strategies, showing improved robustness when dealing with corrupted samples that are similar to the ones used for training. However, all these methods still suffer from the token distribution shift induced by typos. In this work, we propose to tackle textual noise by equipping cross-encoders with a novel LExical-aware Attention module (LEA) that incorporates lexical similarities between words in both sentences. By using raw text similarities, our approach avoids the to
    
[^19]: GilBERTo中的主动性和幸福感：对认知的影响

    Agentivit\`a e telicit\`a in GilBERTo: implicazioni cognitive. (arXiv:2307.02910v1 [cs.CL])

    [http://arxiv.org/abs/2307.02910](http://arxiv.org/abs/2307.02910)

    本研究使用基于Transformer的神经语言模型，并通过与意大利本土讲者的对比实验，探究了模型是否能推断词汇语义和完成形态句法模式。研究结果表明模型在捕捉人类语义能力方面取得了一定的成果。

    

    本研究旨在探究基于Transformer的神经语言模型是否能够推断词汇语义并利用此信息完成形态句法模式。考虑的语义属性是延续性（也与确定性相结合）和主动性。二者均在语义和形态句法之间起作用：它们在语义上确定，在句法上编码。这些任务同时交给了计算模型和一组意大利本土讲者。比较这两组数据可以让我们探究神经语言模型在多大程度上捕捉到人类语义能力的重要方面。

    The goal of this study is to investigate whether a Transformer-based neural language model infers lexical semantics and use this information for the completion of morphosyntactic patterns. The semantic properties considered are telicity (also combined with definiteness) and agentivity. Both act at the interface between semantics and morphosyntax: they are semantically determined and syntactically encoded. The tasks were submitted to both the computational model and a group of Italian native speakers. The comparison between the two groups of data allows us to investigate to what extent neural language models capture significant aspects of human semantic competence.
    
[^20]: 抑郁症对语音特征的相关性产生影响: 通过改进特征相关性来提高抑郁症检测的速度和性能

    The Relationship Between Speech Features Changes When You Get Depressed: Feature Correlations for Improving Speed and Performance of Depression Detection. (arXiv:2307.02892v1 [cs.CL])

    [http://arxiv.org/abs/2307.02892](http://arxiv.org/abs/2307.02892)

    本研究发现抑郁症会改变从语音中提取的特征之间的相关性，同时利用这种洞察力可以通过改进特征相关性来提高抑郁症检测器的训练速度和性能。

    

    本研究表明，抑郁症会改变从语音中提取的特征之间的相关性。此外，它还表明利用这样的洞察力可以提高基于SVM和LSTMs的抑郁症检测器的训练速度和性能。实验是在Androids Corpus上进行的，这是一个涉及112名说话者的公开数据集，其中包括58名由专业精神病学家诊断为抑郁症的人。结果显示，实验中使用的模型在训练速度和性能方面都得到了改善，与使用特征向量相比，使用特征相关性矩阵作为输入时，错误率相对减少了23.1％到26.6％，具体取决于模型。可能的解释是，在抑郁的说话者中，特征相关性矩阵似乎更加多变。相应地，这种现象可以被视为抑郁症的一个标记。

    This work shows that depression changes the correlation between features extracted from speech. Furthermore, it shows that using such an insight can improve the training speed and performance of depression detectors based on SVMs and LSTMs. The experiments were performed over the Androids Corpus, a publicly available dataset involving 112 speakers, including 58 people diagnosed with depression by professional psychiatrists. The results show that the models used in the experiments improve in terms of training speed and performance when fed with feature correlation matrices rather than with feature vectors. The relative reduction of the error rate ranges between 23.1% and 26.6% depending on the model. The probable explanation is that feature correlation matrices appear to be more variable in the case of depressed speakers. Correspondingly, such a phenomenon can be thought of as a depression marker.
    
[^21]: 对比就是你所需的一切

    Contrast Is All You Need. (arXiv:2307.02882v1 [cs.CL])

    [http://arxiv.org/abs/2307.02882](http://arxiv.org/abs/2307.02882)

    对比学习方法在数据稀缺的法律分类场景中表现更好，使用SetFit微调的模型比普通微调使用更少的训练样本。LIME的结果显示，对比学习方法有助于提升对正面和负面特征的认知，这些特征在法律上具有信息量，并对分类结果有贡献。

    

    在这项研究中，我们分析了数据稀缺的分类场景，其中可用的标记法律数据很少且不平衡，可能会影响结果的质量。我们重点关注了两个微调目标：SetFit（句子转换器微调），一种对比学习设置，以及在法律条款分类任务上的普通微调设置。此外，我们使用LIME（局部可解释模型无关解释）比较了提取的特征，以查看哪些特定特征对模型的分类决策有贡献。结果显示，与使用相同数量的训练样本的普通微调相比，使用SetFit的对比设置表现更好。LIME的结果显示，对比学习方法有助于提升对正面和负面特征的认知，这些特征在法律上具有信息量，并对分类结果有贡献。因此，使用对比目标进行微调的模型似乎更自信地基于法律信息进行决策。

    In this study, we analyze data-scarce classification scenarios, where available labeled legal data is small and imbalanced, potentially hurting the quality of the results. We focused on two finetuning objectives; SetFit (Sentence Transformer Finetuning), a contrastive learning setup, and a vanilla finetuning setup on a legal provision classification task. Additionally, we compare the features that are extracted with LIME (Local Interpretable Model-agnostic Explanations) to see which particular features contributed to the model's classification decisions. The results show that a contrastive setup with SetFit performed better than vanilla finetuning while using a fraction of the training samples. LIME results show that the contrastive learning approach helps boost both positive and negative features which are legally informative and contribute to the classification results. Thus a model finetuned with a contrastive objective seems to base its decisions more confidently on legally informa
    
[^22]: ValiTex -- 一种用于计算文本的社会科学构建度量的统一验证框架

    ValiTex -- a uniform validation framework for computational text-based measures of social science constructs. (arXiv:2307.02863v1 [cs.CL])

    [http://arxiv.org/abs/2307.02863](http://arxiv.org/abs/2307.02863)

    ValiTex是一个统一的验证框架，旨在帮助学者们基于文本数据来度量社会科学构建。它借鉴了心理测量学的传统，通过概念模型和动态检查表提供了验证的结构和步骤。

    

    关于如何验证计算文本的社会科学构建度量的指导是分散的。虽然学者们普遍认识到验证他们的文本度量的重要性，但他们通常缺乏共同的术语和统一的框架来进行验证。本文介绍了一个名为ValiTex的新验证框架，旨在帮助学者们基于文本数据来度量社会科学构建。该框架借鉴了心理测量学中长期存在的传统，同时扩展了框架以适用于计算文本分析的目的。ValiTex包括两个组成部分，一个是概念模型，一个是动态检查表。概念模型提供了一个通用的结构，可以指导验证的不同阶段，动态检查表定义了具体的验证步骤，并提供了哪些步骤可能被认为是推荐的（即提供相关和必要的验证证据）或可选的（即对提供额外信息有用的）。

    Guidance on how to validate computational text-based measures of social science constructs is fragmented. Whereas scholars are generally acknowledging the importance of validating their text-based measures, they often lack common terminology and a unified framework to do so. This paper introduces a new validation framework called ValiTex, designed to assist scholars to measure social science constructs based on textual data. The framework draws on a long-established tradition within psychometrics while extending the framework for the purpose of computational text analysis. ValiTex consists of two components, a conceptual model, and a dynamic checklist. Whereas the conceptual model provides a general structure along distinct phases on how to approach validation, the dynamic checklist defines specific validation steps and provides guidance on which steps might be considered recommendable (i.e., providing relevant and necessary validation evidence) or optional (i.e., useful for providing 
    
[^23]: NatLogAttack: 一个用自然逻辑对自然语言推理模型进行攻击的框架

    NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic. (arXiv:2307.02849v1 [cs.CL])

    [http://arxiv.org/abs/2307.02849](http://arxiv.org/abs/2307.02849)

    NatLogAttack是一个用自然逻辑对自然语言推理模型进行系统性攻击的框架，它可以进行保持标签和翻转标签的攻击，并相比现有攻击模型产生了更好的对抗性攻击。

    

    推理自从人工智能的开始就是一个中心话题。近年来在分布式表示和神经网络上取得的进展持续改进了自然语言推理模型的最新性能。然而，这些模型是否通过真正的推理来得出结论，还是依赖于虚假的相关性，这仍然是一个未解决的问题。对抗性攻击已经证明是评估受害模型的致命弱点的重要工具。在这项研究中，我们探讨了基于逻辑形式主义开发攻击模型的基本问题。我们提出了NatLogAttack来执行围绕自然逻辑的系统性攻击，这是一个可追溯到亚里士多德三段论并且与自然语言推理密切相关的经典逻辑形式。该提议的框架可以进行保持标签和翻转标签的攻击。我们展示了与现有攻击模型相比，NatLogAttack产生了更好的对抗性攻击。

    Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial 
    
[^24]: 使用进化调优增强LLM进行新闻摘要生成

    Enhancing LLM with Evolutionary Fine Tuning for News Summary Generation. (arXiv:2307.02839v1 [cs.CL])

    [http://arxiv.org/abs/2307.02839](http://arxiv.org/abs/2307.02839)

    本论文提出一种新的方法使用LLM进行新闻摘要生成，通过进化调优事件模式群体，提高生成结果的准确性和可靠性。

    

    新闻摘要生成是情报分析领域中的重要任务，可以提供准确全面的信息，帮助人们更好地理解和应对复杂的现实事件。然而，传统的新闻摘要生成方法面临一些挑战，包括模型本身和训练数据量的限制，以及文本噪声的影响，使得准确生成可靠信息变得困难。本文提出了一种使用具有强大自然语言理解和生成能力的LLM进行新闻摘要生成的新范式。我们利用LLM从新闻段落中提取多个结构化事件模式，通过遗传算法进化事件模式群体，并选择最适应的事件模式输入LLM生成新闻摘要。设计了一个新闻摘要生成器(NSG)来选择和进化事件模式群体，并生成新闻摘要。

    News summary generation is an important task in the field of intelligence analysis, which can provide accurate and comprehensive information to help people better understand and respond to complex real-world events. However, traditional news summary generation methods face some challenges, which are limited by the model itself and the amount of training data, as well as the influence of text noise, making it difficult to generate reliable information accurately. In this paper, we propose a new paradigm for news summary generation using LLM with powerful natural language understanding and generative capabilities. We use LLM to extract multiple structured event patterns from the events contained in news paragraphs, evolve the event pattern population with genetic algorithm, and select the most adaptive event pattern to input into the LLM to generate news summaries. A News Summary Generator (NSG) is designed to select and evolve the event pattern populations and generate news summaries. T
    
[^25]: 跨领域槽填充的生成式零样本提示学习与逆向提示

    Generative Zero-Shot Prompt Learning for Cross-Domain Slot Filling with Inverse Prompting. (arXiv:2307.02830v1 [cs.CL])

    [http://arxiv.org/abs/2307.02830](http://arxiv.org/abs/2307.02830)

    本文提出了一种生成式零样本提示学习框架，针对跨领域槽填充问题，通过引入逆向提示策略和高效的提示调整策略，改进了泛化性和鲁棒性，并在未见槽上取得了显著提升。

    

    零样本跨领域槽填充旨在将标记的源领域知识转移到未标记的目标领域。现有模型要么对槽描述和示例进行编码，要么使用启发式规则设计手工制作的问题模板，这些模型在泛化能力或鲁棒性方面存在问题。本文提出了一种生成式零样本提示学习框架，用于跨领域槽填充，比之前的工作更具泛化性和鲁棒性。此外，我们引入了一种新颖的逆向提示策略，用于区分不同的槽类型以避免多重预测问题，并提出了一种高效的提示调整策略，通过仅训练较少的提示参数来提高性能。实验证明了我们提出的框架的有效性，特别是在未见槽上的显著提升（+13.44% F1）。

    Zero-shot cross-domain slot filling aims to transfer knowledge from the labeled source domain to the unlabeled target domain. Existing models either encode slot descriptions and examples or design handcrafted question templates using heuristic rules, suffering from poor generalization capability or robustness. In this paper, we propose a generative zero-shot prompt learning framework for cross-domain slot filling, both improving generalization and robustness than previous work. Besides, we introduce a novel inverse prompting strategy to distinguish different slot types to avoid the multiple prediction problem, and an efficient prompt-tuning strategy to boost higher performance by only training fewer prompt parameters. Experiments and analysis demonstrate the effectiveness of our proposed framework, especially huge improvements (+13.44% F1) on the unseen slots.
    
[^26]: VerifAI：验证生成式人工智能

    VerifAI: Verified Generative AI. (arXiv:2307.02796v1 [cs.DB])

    [http://arxiv.org/abs/2307.02796](http://arxiv.org/abs/2307.02796)

    验证生成式人工智能的输出是一个新兴问题，我们提出了通过分析多模态数据湖的底层数据，评估其质量和一致性，来建立评估生成式人工智能模型输出的更坚实基础，并解决错误信息传播的挑战。

    

    生成式人工智能已经取得了重要的进展，但是对于其输出的准确性和可靠性的担忧仍在增长。这种不准确性可能产生严重后果，如错误决策，传播虚假信息，侵犯隐私，法律责任等。虽然已经在进行应对这些风险的努力，包括可解释的人工智能和负责任的人工智能实践，如透明度，隐私保护，偏见缓解以及社会和环境责任等，但由生成式人工智能引起的错误信息仍然是一个重大挑战。我们提出，从数据管理的角度验证生成式人工智能的输出是生成式人工智能的一个新兴问题。这包括分析来自多模态数据湖的底层数据，包括文本文件，表格和知识图谱，并评估其质量和一致性。通过这样做，我们可以为评估生成式人工智能模型的输出奠定更坚实的基础。这种方法能够帮助解决生成式人工智能的输出验证问题。

    Generative AI has made significant strides, yet concerns about the accuracy and reliability of its outputs continue to grow. Such inaccuracies can have serious consequences such as inaccurate decision-making, the spread of false information, privacy violations, legal liabilities, and more. Although efforts to address these risks are underway, including explainable AI and responsible AI practices such as transparency, privacy protection, bias mitigation, and social and environmental responsibility, misinformation caused by generative AI will remain a significant challenge. We propose that verifying the outputs of generative AI from a data management perspective is an emerging issue for generative AI. This involves analyzing the underlying data from multi-modal data lakes, including text files, tables, and knowledge graphs, and assessing its quality and consistency. By doing so, we can establish a stronger foundation for evaluating the outputs of generative AI models. Such an approach ca
    
[^27]: 大规模语言模型对数据科学教育应该做什么？

    What Should Data Science Education Do with Large Language Models?. (arXiv:2307.02792v1 [cs.CY])

    [http://arxiv.org/abs/2307.02792](http://arxiv.org/abs/2307.02792)

    大型语言模型（LLM）正在改变数据科学家的责任和数据科学教育模式，从动手编码和标准分析转变为评估和管理自动化AI执行的分析。这种转变要求数据科学教育注重培养学生的多样化技能，如创造力、批判性思维和AI引导的编程。

    

    大型语言模型（LLM），如ChatGPT等的快速发展正在改变数据科学和统计学。这些最先进的工具可以简化复杂的流程，从而重塑了数据科学家的角色。我们认为LLM正在转变数据科学家的责任，将他们的重点从动手编码、数据整理和进行标准分析转变为评估和管理这些自动化AI执行的分析。这种角色的演变类似于从软件工程师转变为产品经理。我们在本文中使用LLM在数据科学案例研究中说明了这种转变。这些发展要求数据科学教育有意义地发展。教育方法现在必须更加注重培养学生的多样化技能，如LLM启发的创造力、批判性思维、AI引导的编程。LLM还可以在课堂上起到重要的作用，作为互动式教学和...

    The rapid advances of large language models (LLMs), such as ChatGPT, are revolutionizing data science and statistics. These state-of-the-art tools can streamline complex processes. As a result, it reshapes the role of data scientists. We argue that LLMs are transforming the responsibilities of data scientists, shifting their focus from hands-on coding, data-wrangling and conducting standard analyses to assessing and managing analyses performed by these automated AIs. This evolution of roles is reminiscent of the transition from a software engineer to a product manager. We illustrate this transition with concrete data science case studies using LLMs in this paper. These developments necessitate a meaningful evolution in data science education. Pedagogy must now place greater emphasis on cultivating diverse skillsets among students, such as LLM-informed creativity, critical thinking, AI-guided programming. LLMs can also play a significant role in the classroom as interactive teaching and
    
[^28]: 训练模型生成、识别和重构无益思维

    Training Models to Generate, Recognize, and Reframe Unhelpful Thoughts. (arXiv:2307.02768v1 [cs.CL])

    [http://arxiv.org/abs/2307.02768](http://arxiv.org/abs/2307.02768)

    本研究提出了一个新数据集PATTERNREFRAME，用于训练和评估语言模型，探讨如何生成大量的个性化练习材料和建议，用于识别和重构无益思维模式。

    

    在过去几十年中，许多认知方法来提高幸福感，例如识别和重构无益思维，得到了相当多的实证支持，但在自助格式下仍缺乏广泛采纳。其中一个原因是缺乏充分具体且多样化的专门练习材料。本研究探讨当前语言模型是否可以利用来产生大量的练习材料，展示与特定上下文匹配的常见无益思维模式，并生成适当的积极重构建议。我们提出了一个名为PATTERNREFRAME的新数据集，包含约1万个以给定角色为条件的包含无益思维模式的例子，并附带大约2.7万个积极的重构建议。通过使用该数据集对当前模型进行训练和/或评估，我们展示现有模型已经可以成为一个有力的工具，帮助生成量身定制的练习材料和假设，且不需或只需最少的人工修改。

    Many cognitive approaches to well-being, such as recognizing and reframing unhelpful thoughts, have received considerable empirical support over the past decades, yet still lack truly widespread adoption in self-help format. A barrier to that adoption is a lack of adequately specific and diverse dedicated practice material. This work examines whether current language models can be leveraged to both produce a virtually unlimited quantity of practice material illustrating standard unhelpful thought patterns matching specific given contexts, and generate suitable positive reframing proposals. We propose PATTERNREFRAME, a novel dataset of about 10k examples of thoughts containing unhelpful thought patterns conditioned on a given persona, accompanied by about 27k positive reframes. By using this dataset to train and/or evaluate current models, we show that existing models can already be powerful tools to help generate an abundance of tailored practice material and hypotheses, with no or min
    
[^29]: 您的伴侣需要专业帮助：通过建模社交关系确定消息的上下文适当性

    Your spouse needs professional help: Determining the Contextual Appropriateness of Messages through Modeling Social Relationships. (arXiv:2307.02763v1 [cs.CL])

    [http://arxiv.org/abs/2307.02763](http://arxiv.org/abs/2307.02763)

    本文介绍了一种通过建模个体之间的社交关系来准确识别给定上下文中适宜性的新方法，并探讨了关系本身如何作为暗含规范的功能以及在不同对话设置中上下文敏感性的程度。

    

    理解人际交流在一定程度上需要理解消息所说的社交背景和规范。然而，目前用于识别此类交流中冒犯内容的方法很大程度上独立于上下文，只有少数方法考虑到社区规范或先前对话作为上下文。在这里，我们介绍了一种通过显式地建模个体之间的社交关系来识别不适当交流的新方法。我们引入了一个新的具有上下文判断适宜性的数据集，并展示了大型语言模型可以很好地整合关系信息来准确地识别给定上下文中的适宜性。利用在线对话和电影对话的数据，我们揭示了关系本身作为暗含规范的功能，并量化了不同对话设置中需要上下文敏感性的程度。

    Understanding interpersonal communication requires, in part, understanding the social context and norms in which a message is said. However, current methods for identifying offensive content in such communication largely operate independent of context, with only a few approaches considering community norms or prior conversation as context. Here, we introduce a new approach to identifying inappropriate communication by explicitly modeling the social relationship between the individuals. We introduce a new dataset of contextually-situated judgments of appropriateness and show that large language models can readily incorporate relationship information to accurately identify appropriateness in a given context. Using data from online conversations and movie dialogues, we provide insight into how the relationships themselves function as implicit norms and quantify the degree to which context-sensitivity is needed in different conversation settings. Further, we also demonstrate that contextua
    
[^30]: PRD: 同行评级和讨论改善基于大型语言模型的评估

    PRD: Peer Rank and Discussion Improve Large Language Model based Evaluations. (arXiv:2307.02762v1 [cs.CL])

    [http://arxiv.org/abs/2307.02762](http://arxiv.org/abs/2307.02762)

    本研究提出了PRD算法，利用同行评级和讨论改善了基于大型语言模型的评估方法，解决了自我提升和位置偏见等问题。

    

    如今，评估和比较不同现代大型语言模型（LLMs）生成的回答质量在自动化方面很难。最近的研究建议并主要使用LLMs作为无参考度量衡开放式问题回答的参考指标。更具体地说，他们以被认为是“最强”的LLM作为评估器，对候选模型的答案进行两两比较并提供排名分数。然而，这种直观的方法存在多个问题，例如带来自我提升（青睐自己的答案）和位置偏见。我们从教育领域（Cho and MacArthur, 2011；Walsh, 2014）中汲取见解和教训，改进了基于LLM的评估。具体而言，我们提出了（1）同行评级（PR）算法，该算法考虑每个同行LLM对所有答案对的两两偏好，并输出模型的最终排名；以及（2）同行讨论（PD），在其中我们促使两个LLMs进行讨论并尝试就两个偏好达成共识。

    Nowadays, the quality of responses generated by different modern large language models (LLMs) are hard to evaluate and compare automatically. Recent studies suggest and predominantly use LLMs as a reference-free metric for open-ended question answering. More specifically, they use the recognized "strongest" LLM as the evaluator, which conducts pairwise comparisons of candidate models' answers and provides a ranking score. However, this intuitive method has multiple problems, such as bringing in self-enhancement (favoring its own answers) and positional bias. We draw insights and lessons from the educational domain (Cho and MacArthur, 2011; Walsh, 2014) to improve LLM-based evaluations. Specifically, we propose the (1) peer rank (PR) algorithm that takes into account each peer LLM's pairwise preferences of all answer pairs, and outputs a final ranking of models; and (2) peer discussion (PD), where we prompt two LLMs to discuss and try to reach a mutual agreement on preferences of two an
    
[^31]: 在在线社区中探索语言风格匹配：社会背景和对话动态的作用

    Exploring Linguistic Style Matching in Online Communities: The Role of Social Context and Conversation Dynamics. (arXiv:2307.02758v1 [cs.CL])

    [http://arxiv.org/abs/2307.02758](http://arxiv.org/abs/2307.02758)

    本研究在Reddit上分析了两方对话主题的大量语料库，研究了LSM在对话中的差异以及与社区指标的关系，揭示了理解社区动态时对话参与的重要性。

    

    在对话中的语言风格匹配可以反映出社会影响的多个方面，如权力或说服力。然而，在类似Reddit等平台上，LSM与在线沟通结果的关系尚不清楚。在这项研究中，我们分析了Reddit中两方对话主题的大量语料库，并使用两种类型的风格：功能词的使用和形式化。使用这个框架，我们研究了不同社交因素在Reddit对话中LSM水平的差异：帖子和子社区特征、对话深度、用户资历和评论的争议性。最后，我们测量了社区禁令后失去地位后LSM的变化。我们的发现揭示了LSM在Reddit对话中与几个社区指标的相互作用，暗示了在了解社区动态时理解对话参与的重要性。

    Linguistic style matching (LSM) in conversations can be reflective of several aspects of social influence such as power or persuasion. However, how LSM relates to the outcomes of online communication on platforms such as Reddit is an unknown question. In this study, we analyze a large corpus of two-party conversation threads in Reddit where we identify all occurrences of LSM using two types of style: the use of function words and formality. Using this framework, we examine how levels of LSM differ in conversations depending on several social factors within Reddit: post and subreddit features, conversation depth, user tenure, and the controversiality of a comment. Finally, we measure the change of LSM following loss of status after community banning. Our findings reveal the interplay of LSM in Reddit conversations with several community metrics, suggesting the importance of understanding conversation engagement when understanding community dynamics.
    
[^32]: 使用目标领域描述的密集检索适应

    Dense Retrieval Adaptation using Target Domain Description. (arXiv:2307.02740v1 [cs.IR])

    [http://arxiv.org/abs/2307.02740](http://arxiv.org/abs/2307.02740)

    本文提出了一种新的信息检索领域适应方法，该方法假设检索模型无法访问目标文档集，但可以访问描述目标领域的简要文本描述。

    

    在信息检索中，领域适应是将检索模型适应于数据分布与源领域不同的新领域的过程。现有的方法集中于无监督领域适应，在这种情况下，它们可以访问目标文档集，或者是监督（通常是少样本）领域适应，在这种情况下，它们还可以访问目标领域中（有限的）标记数据。还存在一些研究致力于改善没有适应的检索模型的零样本性能。本文介绍了信息检索中尚未探索的一类新的领域适应方法。在这种情况下，与零样本设置类似，我们假设检索模型无法访问目标文档集，但可以访问一个简要的文本描述，说明目标领域。我们定义了一个领域属性的分类学，用于理解源领域可以适应到目标领域的不同特性。

    In information retrieval (IR), domain adaptation is the process of adapting a retrieval model to a new domain whose data distribution is different from the source domain. Existing methods in this area focus on unsupervised domain adaptation where they have access to the target document collection or supervised (often few-shot) domain adaptation where they additionally have access to (limited) labeled data in the target domain. There also exists research on improving zero-shot performance of retrieval models with no adaptation. This paper introduces a new category of domain adaptation in IR that is as-yet unexplored. Here, similar to the zero-shot setting, we assume the retrieval model does not have access to the target document collection. In contrast, it does have access to a brief textual description that explains the target domain. We define a taxonomy of domain attributes in retrieval tasks to understand different properties of a source domain that can be adapted to a target domain
    
[^33]: RecallM:一种用于时间上下文理解和问题回答的架构

    RecallM: An Architecture for Temporal Context Understanding and Question Answering. (arXiv:2307.02738v1 [cs.AI])

    [http://arxiv.org/abs/2307.02738](http://arxiv.org/abs/2307.02738)

    本文介绍了一种名为RecallM的架构，用于创建可适应和可更新的长期记忆，以提升大型语言模型聊天机器人的时间理解能力。

    

    用于大型语言模型（LLM）聊天机器人的理想长期记忆机制将为连续学习、复杂推理和学习序列和时间依赖关系打下基础。创建这种类型的记忆机制是一个极具挑战性的问题。在本文中，我们探索了不同方法实现长期记忆的效果。我们提出了一种新的架构，专注于为AGI系统创建可适应和可更新的长期记忆。我们通过各种实验展示了RecallM架构的好处，特别是它提供的改进的时间理解能力。

    The ideal long-term memory mechanism for Large Language Model (LLM) based chatbots, would lay the foundation for continual learning, complex reasoning and allow sequential and temporal dependencies to be learnt. Creating this type of memory mechanism is an extremely challenging problem. In this paper we explore different methods of achieving the effect of long-term memory. We propose a new architecture focused on creating adaptable and updatable long-term memory for AGI systems. We demonstrate through various experiments the benefits of the RecallM architecture, particularly the improved temporal understanding it provides.
    
[^34]: 文本对齐是一个高效的用于海量NLP任务的统一模型

    Text Alignment Is An Efficient Unified Model for Massive NLP Tasks. (arXiv:2307.02729v1 [cs.CL])

    [http://arxiv.org/abs/2307.02729](http://arxiv.org/abs/2307.02729)

    本研究提出了一种高效的文本对齐模型，可以应用于广泛的NLP任务，包括文本蕴含、相似性、问答、事实一致性等。通过对RoBERTa进行轻量级微调，可以构建一个更小规模的模型，实现与大型语言模型相当甚至更优的性能。

    

    大型语言模型（LLMs）通常被设计为下一个词语预测的函数，在广泛的NLP任务中表现出色。尽管具有广泛性，但下一个词语预测对于许多任务来说通常不是一种高效的表达方式，需要极大规模的模型参数（数百亿级别），有时会导致次优的性能。实际上，构建更高效的模型通常是可取的——尽管不够通用，但它们仍然适用于大量问题的子集，并以更小的模型规模实现相当或甚至更优的性能。在本文中，我们将文本对齐提出作为一种高效的统一模型，用于涉及文本蕴含、相似性、问答（和可回答性）、事实一致性等关键任务的广泛范围。给定一对文本，该模型测量它们之间信息的对齐程度。我们通过对RoBERTa（3.55亿参数）进行轻量级微调来实例化一个对齐模型（Align）。

    Large language models (LLMs), typically designed as a function of next-word prediction, have excelled across extensive NLP tasks. Despite the generality, next-word prediction is often not an efficient formulation for many of the tasks, demanding an extreme scale of model parameters (10s or 100s of billions) and sometimes yielding suboptimal performance. In practice, it is often desirable to build more efficient models -- despite being less versatile, they still apply to a substantial subset of problems, delivering on par or even superior performance with much smaller model sizes. In this paper, we propose text alignment as an efficient unified model for a wide range of crucial tasks involving text entailment, similarity, question answering (and answerability), factual consistency, and so forth. Given a pair of texts, the model measures the degree of alignment between their information. We instantiate an alignment model (Align) through lightweight finetuning of RoBERTa (355M parameters)
    
[^35]: 基于设备限制的自监督语音表示学习在关键词检测中的应用及知识蒸馏（arXiv:2307.02720v1 [cs.CL]）

    On-Device Constrained Self-Supervised Speech Representation Learning for Keyword Spotting via Knowledge Distillation. (arXiv:2307.02720v1 [cs.CL])

    [http://arxiv.org/abs/2307.02720](http://arxiv.org/abs/2307.02720)

    这项研究提出了一种基于知识蒸馏的自监督语音表示学习架构，以应用于设备内关键词检测。通过在有限的资源情况下将知识从复杂模型传递给轻量级模型，该方法在关键词检测任务中表现出了出色的性能。

    

    大型自监督模型是有效的特征提取器，但在设备内预算限制和有偏差的数据集收集下应用具有挑战性，特别是在关键词检测中。为了解决这个问题，我们提出了一种基于知识蒸馏的自监督语音表示学习（S3RL）架构，用于设备内关键词检测。我们的方法使用教师-学生框架，通过双视图互相关蒸馏和教师的码本作为学习目标，从更大、更复杂的模型中传递知识到更小、轻量级的模型中。我们使用一个16.6k小时的内部数据集，在Alexa的关键词检测任务上评估了我们模型的性能。我们的技术在正常和噪声条件下表现出了出色的性能，证明了知识蒸馏方法在在设备资源限制下构建自监督模型的有效性。

    Large self-supervised models are effective feature extractors, but their application is challenging under on-device budget constraints and biased dataset collection, especially in keyword spotting. To address this, we proposed a knowledge distillation-based self-supervised speech representation learning (S3RL) architecture for on-device keyword spotting. Our approach used a teacher-student framework to transfer knowledge from a larger, more complex model to a smaller, light-weight model using dual-view cross-correlation distillation and the teacher's codebook as learning objectives. We evaluated our model's performance on an Alexa keyword spotting detection task using a 16.6k-hour in-house dataset. Our technique showed exceptional performance in normal and noisy conditions, demonstrating the efficacy of knowledge distillation methods in constructing self-supervised models for keyword spotting tasks while working within on-device resource constraints.
    
[^36]: CFSum: 一种用于多模态摘要的粗到细贡献网络

    CFSum: A Coarse-to-Fine Contribution Network for Multimodal Summarization. (arXiv:2307.02716v1 [cs.CL])

    [http://arxiv.org/abs/2307.02716](http://arxiv.org/abs/2307.02716)

    CFSum是一个用于多模态摘要的粗到细贡献网络，能够准确计算和利用图像在摘要中的不同贡献，实验结果显示其优于其他基准方法。

    

    多模态摘要通常存在视觉模态贡献不明确的问题。现有的多模态摘要方法都集中在设计不同模态的融合方法，而忽视了视觉模态有用的自适应条件。因此，我们提出了一种新颖的粗到细贡献网络用于多模态摘要（CFSum），以考虑图像在摘要中的不同贡献。首先，为了消除无用图像的干扰，我们提出了一个预过滤模块来舍弃无用图像。其次，为了准确使用有用图像，我们提出了两个层次的视觉补充模块，词级和短语级。具体而言，计算图像贡献并用于引导文本和视觉模态的注意力。实验结果表明，CFSum在标准基准上明显优于多个强基准。此外，分析验证了提出的方法的有效性。

    Multimodal summarization usually suffers from the problem that the contribution of the visual modality is unclear. Existing multimodal summarization approaches focus on designing the fusion methods of different modalities, while ignoring the adaptive conditions under which visual modalities are useful. Therefore, we propose a novel Coarse-to-Fine contribution network for multimodal Summarization (CFSum) to consider different contributions of images for summarization. First, to eliminate the interference of useless images, we propose a pre-filter module to abandon useless images. Second, to make accurate use of useful images, we propose two levels of visual complement modules, word level and phrase level. Specifically, image contributions are calculated and are adopted to guide the attention of both textual and visual modalities. Experimental results have shown that CFSum significantly outperforms multiple strong baselines on the standard benchmark. Furthermore, the analysis verifies th
    
[^37]: Strahler数的统计力学：基于随机和自然语言句子的研究

    Statistical Mechanics of Strahler Number via Random and Natural Language Sentences. (arXiv:2307.02697v1 [cs.CL])

    [http://arxiv.org/abs/2307.02697](http://arxiv.org/abs/2307.02697)

    本文通过统计力学分析自然语言句子树结构的Strahler数的上下限，发现它几乎总是3或4，并证明它是处理句子所需记忆量的下限。同时，对随机树进行的分析揭示出Strahler数的增长模式，揭示了它作为自然语言句子特征的统计基础。

    

    Strahler数最初被提出用于描述河流分支的复杂性，并找到了各种应用。本文提出了计算自然语言句子树结构的Strahler数上下限的方法，这些结构可以在一个大型数据集中进行统计力学分析。通过对语法注释数据的经验性测量，显示自然语言句子的Strahler数几乎总是3或4，与Strahler（1957年）和Horton（1945年）报道的河流分流情况类似。从该数值的理论观点出发，我们证明它是在特定模型下处理句子所需记忆量的下限。对随机树进行的数学分析进一步假设了Strahler数的性质，揭示出它并非常数而是以对数形式增长。这一发现揭示了Strahler数作为描述自然语言句子特征的统计基础。

    The Strahler number was originally proposed to characterize the complexity of river bifurcation and has found various applications. This article proposes computation of the Strahler number's upper and lower limits for natural language sentence tree structures, which are available in a large dataset allowing for statistical mechanics analysis.  Through empirical measurements across grammatically annotated data, the Strahler number of natural language sentences is shown to be almost always 3 or 4, similar to the case of river bifurcation as reported by Strahler (1957) and Horton (1945).  From the theory behind the number, we show that it is the lower limit of the amount of memory required to process sentences under a particular model. A mathematical analysis of random trees provides a further conjecture on the nature of the Strahler number, revealing that it is not a constant but grows logarithmically. This finding uncovers the statistical basics behind the Strahler number as a character
    
[^38]: 使用结构化注意力扩展上下文演示

    Scaling In-Context Demonstrations with Structured Attention. (arXiv:2307.02690v1 [cs.CL])

    [http://arxiv.org/abs/2307.02690](http://arxiv.org/abs/2307.02690)

    本研究提出了一种用于上下文学习的结构化注意力机制，解决了大规模语言模型在使用演示进行上下文学习时遇到的限制与挑战。

    

    最近大规模语言模型的兴起突出了它们在上下文学习方面的能力，即在上下文中从少数演示中“学习”执行任务而无需进行参数更新。然而，它们在上下文学习方面的能力受到模型架构的限制：1）由于位置嵌入，演示的使用受到最大句子长度的限制；2）注意力的二次复杂度阻碍用户有效使用更多的演示；3）研究表明，LLM对演示的顺序敏感。在这项工作中，我们通过提出更好的架构设计来解决这些挑战。我们提出了SAICL（用于上下文学习的结构化注意力），它通过为上下文学习设计了一种结构化注意力机制来替换全注意力，并消除了个别演示之间不必要的依赖关系，同时使模型对演示的排列不变。

    The recent surge of large language models (LLMs) highlights their ability to perform in-context learning, i.e., "learning" to perform a task from a few demonstrations in the context without any parameter updates. However, their capabilities of in-context learning are limited by the model architecture: 1) the use of demonstrations is constrained by a maximum sentence length due to positional embeddings; 2) the quadratic complexity of attention hinders users from using more demonstrations efficiently; 3) LLMs are shown to be sensitive to the order of the demonstrations. In this work, we tackle these challenges by proposing a better architectural design for in-context learning. We propose SAICL (Structured Attention for In-Context Learning), which replaces the full-attention by a structured attention mechanism designed for in-context learning, and removes unnecessary dependencies between individual demonstrations, while making the model invariant to the permutation of demonstrations. We e
    
[^39]: 学习抽象意义表示的符号规则以进行文本强化学习

    Learning Symbolic Rules over Abstract Meaning Representations for Textual Reinforcement Learning. (arXiv:2307.02689v1 [cs.CL])

    [http://arxiv.org/abs/2307.02689](http://arxiv.org/abs/2307.02689)

    本文提出了一个基于神经符号方法的文本强化学习代理NESTA，通过学习抽象可解释的规则作为策略，在文本游戏中取得了比深度强化学习方法更好的泛化和学习效果。

    

    基于文本的强化学习代理通常是基于神经网络的模型，使用嵌入式表示学习无法解释的策略，往往不能很好地推广到未知的游戏。而神经符号方法，特别是利用中间形式表示的方法，在语言理解任务中受到了重视。这是因为它们具有从内部可解释性、对训练数据要求较少以及在未知数据情况下具有泛化能力的优势。因此，在本文中，我们提出了一个模块化的NEuro-Symbolic Textual Agent (NESTA)方法，它将通用的语义解析器与规则归纳系统相结合，学习抽象可解释的规则作为策略。我们在已建立的基于文本的游戏基准上进行实验，结果显示所提出的NESTA方法在未知测试游戏上具有更好的泛化能力，并能够从中进行学习。

    Text-based reinforcement learning agents have predominantly been neural network-based models with embeddings-based representation, learning uninterpretable policies that often do not generalize well to unseen games. On the other hand, neuro-symbolic methods, specifically those that leverage an intermediate formal representation, are gaining significant attention in language understanding tasks. This is because of their advantages ranging from inherent interpretability, the lesser requirement of training data, and being generalizable in scenarios with unseen data. Therefore, in this paper, we propose a modular, NEuro-Symbolic Textual Agent (NESTA) that combines a generic semantic parser with a rule induction system to learn abstract interpretable rules as policies. Our experiments on established text-based game benchmarks show that the proposed NESTA method outperforms deep reinforcement learning-based techniques by achieving better generalization to unseen test games and learning from 
    
[^40]: 通过联合优化文本和时刻实现零样本密集视频字幕生成

    Zero-Shot Dense Video Captioning by Jointly Optimizing Text and Moment. (arXiv:2307.02682v1 [cs.CV])

    [http://arxiv.org/abs/2307.02682](http://arxiv.org/abs/2307.02682)

    通过在训练阶段不使用视频和标注，而是在测试时仅优化输入，我们提出了一种零样本的密集视频字幕生成方法。通过联合优化文本和时刻，我们的方法能够在视频中准确地定位和描述事件。

    

    密集视频字幕生成是一项将有意义的时刻定位和相关字幕生成应用于视频中的任务，通常需要一个昂贵的带有标注的视频片段和文本的语料库。为了降低标注成本，我们提出了一种新颖的零样本密集视频字幕生成方法ZeroTA。我们的方法在训练阶段不需要任何视频或标注，而是通过仅在输入上进行优化，在测试时定位和描述每个输入视频中的事件。这通过引入一个表示视频中的时间段的软时刻掩码，并与语言模型的前缀参数进行联合优化来实现。这种联合优化通过最大化生成文本与视频中的某个时刻之间的匹配分数，将固定的语言生成模型（即GPT-2）与固定的视觉-语言对比模型（即CLIP）进行对齐。我们还引入了一对一的时间IoU损失，使一组软时刻掩码之间进行比对。

    Dense video captioning, a task of localizing meaningful moments and generating relevant captions for videos, often requires a large, expensive corpus of annotated video segments paired with text. In an effort to minimize the annotation cost, we propose ZeroTA, a novel method for dense video captioning in a zero-shot manner. Our method does not require any videos or annotations for training; instead, it localizes and describes events within each input video at test time by optimizing solely on the input. This is accomplished by introducing a soft moment mask that represents a temporal segment in the video and jointly optimizing it with the prefix parameters of a language model. This joint optimization aligns a frozen language generation model (i.e., GPT-2) with a frozen vision-language contrastive model (i.e., CLIP) by maximizing the matching score between the generated text and a moment within the video. We also introduce a pairwise temporal IoU loss to let a set of soft moment masks c
    
[^41]: 无监督的社交媒体整形手术帖子情感分析

    Unsupervised Sentiment Analysis of Plastic Surgery Social Media Posts. (arXiv:2307.02640v1 [cs.CL])

    [http://arxiv.org/abs/2307.02640](http://arxiv.org/abs/2307.02640)

    该论文提出了一种无监督的方法来对社交媒体中的整形手术帖子进行情感分析。利用自然语言处理和深度学习模型，研究人员通过创建特征、学习和生成主题，实现了对大量文本数据的自动分析。

    

    社交媒体平台上用户帖子的大量收集主要由于文本数据的数量和速度，而在人工智能（AI）的用例中未被使用。自然语言处理（NLP）是AI的一个子领域，利用文档集合（称为语料库）对计算机进行类似人类语言理解的训练。利用词频-逆文档频率（TF-IDF）的词排名方法，在文档之间创建特征，可以进行无监督的分析，即机器学习在没有人工标记数据的情况下对文档进行分组。针对拥有数千个特征的大数据集，本研究利用t-分布随机邻域嵌入（t-SNE）、K-均值聚类和潜在狄利克雷分布（LDA）来学习顶级词汇并生成Reddit和Twitter的合并语料库的主题。利用非常简单的深度学习模型，本研究证明了无监督分析的应用结果，使得计算机能够自动进行社交媒体帖子的情感分析。

    The massive collection of user posts across social media platforms is primarily untapped for artificial intelligence (AI) use cases based on the sheer volume and velocity of textual data. Natural language processing (NLP) is a subfield of AI that leverages bodies of documents, known as corpora, to train computers in human-like language understanding. Using a word ranking method, term frequency-inverse document frequency (TF-IDF), to create features across documents, it is possible to perform unsupervised analytics, machine learning (ML) that can group the documents without a human manually labeling the data. For large datasets with thousands of features, t-distributed stochastic neighbor embedding (t-SNE), k-means clustering and Latent Dirichlet allocation (LDA) are employed to learn top words and generate topics for a Reddit and Twitter combined corpus. Using extremely simple deep learning models, this study demonstrates that the applied results of unsupervised analysis allow a comput
    
[^42]: SkipDecode：用于高效LLM推理的自回归跳跃解码方法（arXiv:2307.02628v1 [cs.CL]）

    SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference. (arXiv:2307.02628v1 [cs.CL])

    [http://arxiv.org/abs/2307.02628](http://arxiv.org/abs/2307.02628)

    本文提出了一种名为SkipDecode的简单而有效的标记级早期退出方法，能够与批量推理和KV缓存无缝配合，解决了传统方法在这些方面的限制。

    

    自回归的大型语言模型（LLM）在各种自然语言生成任务中取得了显著的进展。然而，由于逐个生成标记的自回归方式，它们产生了高计算成本和延迟。为了解决这个问题，已经提出了几种方法来使用早期退出策略来降低计算成本。这些策略能够在不对每个标记应用完整计算图的情况下加快文本生成速度。虽然现有的标记级早期退出方法在在线推理中显示出了有希望的结果，但它们不能轻易应用于批量推理和键值缓存。这是因为它们必须等到批量中的最后一个标记退出后才能停止计算。这严重限制了这种技术的实际应用。在本文中，我们提出了一种简单而有效的标记级早期退出方法SkipDecode，它能够与批量推理和KV缓存无缝配合。它克服了先前的限制，并提出了一种实用的解决方案。

    Autoregressive large language models (LLMs) have made remarkable progress in various natural language generation tasks. However, they incur high computation cost and latency resulting from the autoregressive token-by-token generation. To address this issue, several approaches have been proposed to reduce computational cost using early-exit strategies. These strategies enable faster text generation using reduced computation without applying the full computation graph to each token. While existing token-level early exit methods show promising results for online inference, they cannot be readily applied for batch inferencing and Key-Value caching. This is because they have to wait until the last token in a batch exits before they can stop computing. This severely limits the practical application of such techniques. In this paper, we propose a simple and effective token-level early exit method, SkipDecode, designed to work seamlessly with batch inferencing and KV caching. It overcomes prio
    
[^43]: 基于人类启发的渐进对齐和比较学习用于基于经验的词汇获取

    Human Inspired Progressive Alignment and Comparative Learning for Grounded Word Acquisition. (arXiv:2307.02615v1 [cs.CL])

    [http://arxiv.org/abs/2307.02615](http://arxiv.org/abs/2307.02615)

    本研究通过比较学习和渐进对齐的方式，借鉴人类语言习得的过程，探索了一种用于基于经验的词汇获取的计算过程。该方法不涉及固定的词汇量大小，也不涉及有区分性的目标，能够高效地持续学习更多的概念。

    

    人类语言习得是一种高效、受监督和持续的过程。本研究借鉴了人类婴儿习得第一门语言的方式，通过比较学习开发了一种用于词汇获取的计算过程。受认知发现的启发，我们生成了一个小型数据集，使计算模型能够比较各种属性的相似性和差异性，学习过滤出并提取共同的信息用于每个共享的语言标签。我们将词汇获取框架定义为既包括信息过滤过程，也包括表征-符号映射过程。该过程不涉及固定的词汇量大小，也不涉及有区分性的目标，使模型能够持续高效地学习更多的概念。我们在控制实验中的结果显示了这种方法在高效地持续学习基于经验的词汇方面的潜力。

    Human language acquisition is an efficient, supervised, and continual process. In this work, we took inspiration from how human babies acquire their first language, and developed a computational process for word acquisition through comparative learning. Motivated by cognitive findings, we generated a small dataset that enables the computation models to compare the similarities and differences of various attributes, learn to filter out and extract the common information for each shared linguistic label. We frame the acquisition of words as not only the information filtration process, but also as representation-symbol mapping. This procedure does not involve a fixed vocabulary size, nor a discriminative objective, and allows the models to continually learn more concepts efficiently. Our results in controlled experiments have shown the potential of this approach for efficient continual learning of grounded words.
    
[^44]: 通过一个空格绕过ChatGPT检测器

    Evade ChatGPT Detectors via A Single Space. (arXiv:2307.02599v1 [cs.CL])

    [http://arxiv.org/abs/2307.02599](http://arxiv.org/abs/2307.02599)

    本研究发现，当前的ChatGPT检测器不能有效区分人类生成和AI生成内容之间的差异，而一个额外的空格成为了规避检测的关键因素。

    

    ChatGPT带来了革命性的社会价值，但也引发了人们对于AI生成内容滥用的担忧。因此，一个重要问题是如何检测出内容是由ChatGPT生成还是人类生成的。现有的检测器是建立在人类生成和AI生成内容之间存在分布差距的假设上的。这些差距通常是通过统计信息或分类器来识别的。我们的研究质疑了检测器中的分布差距假设。我们发现检测器不能有效地区分人类生成和AI生成内容之间的语义和风格差距。相反，"微小的差异"，如额外的一个空格，在检测中变得至关重要。基于这一发现，我们提出了SpaceInfi策略来规避检测。实验证明了这种策略在多个基准和检测器上的有效性。我们还对为什么SpaceInfi能成功规避检测提供了理论解释。

    ChatGPT brings revolutionary social value but also raises concerns about the misuse of AI-generated content. Consequently, an important question is how to detect whether content is generated by ChatGPT or by human. Existing detectors are built upon the assumption that there are distributional gaps between human-generated and AI-generated content. These gaps are typically identified using statistical information or classifiers. Our research challenges the distributional gap assumption in detectors. We find that detectors do not effectively discriminate the semantic and stylistic gaps between human-generated and AI-generated content. Instead, the "subtle differences", such as an extra space, become crucial for detection. Based on this discovery, we propose the SpaceInfi strategy to evade detection. Experiments demonstrate the effectiveness of this strategy across multiple benchmarks and detectors. We also provide a theoretical explanation for why SpaceInfi is successful in evading perple
    
[^45]: ODD: 一份基于自然语言处理的药物滥用异常行为检测的基准数据集

    ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection. (arXiv:2307.02591v1 [cs.CL])

    [http://arxiv.org/abs/2307.02591](http://arxiv.org/abs/2307.02591)

    这个研究介绍了一份名为ODD的新型基准数据集，用于通过分析患者的电子健康记录笔记，检测和分类药物滥用异常行为。这个数据集在药物相关病例的自然语言处理研究中具有重要的创新和贡献。

    

    药物滥用异常行为（ORAB）是防止药物过量的新风险因素。以往，ORAB主要通过调查结果和药物给予监测进行评估。然而，这些方法无法扩展，并不能涵盖所有异常行为的范围。然而，ORAB在电子健康记录笔记中广泛有记录。本文介绍了一个名为ODD的新型生物医学自然语言处理基准数据集，用于ORAB检测。ODD是一个专家注释的数据集，包括750多个公开可用的电子健康记录笔记。ODD旨在从患者的电子健康记录笔记中识别ORAB，并将其分类为九个类别：1）已确认异常行为，2）暗示的异常行为，3）阿片类药物，4）适应症，5）已诊断的阿片制剂依赖，6）苯二氮平类药物，7）药物变化，8）与中枢神经系统相关，9）社会健康决定因素。

    Opioid related aberrant behaviors (ORAB) present novel risk factors for opioid overdose. Previously, ORAB have been mainly assessed by survey results and by monitoring drug administrations. Such methods however, cannot scale up and do not cover the entire spectrum of aberrant behaviors. On the other hand, ORAB are widely documented in electronic health record notes. This paper introduces a novel biomedical natural language processing benchmark dataset named ODD, for ORAB Detection Dataset. ODD is an expert-annotated dataset comprising of more than 750 publicly available EHR notes. ODD has been designed to identify ORAB from patients' EHR notes and classify them into nine categories; 1) Confirmed Aberrant Behavior, 2) Suggested Aberrant Behavior, 3) Opioids, 4) Indication, 5) Diagnosed opioid dependency, 6) Benzodiapines, 7) Medication Changes, 8) Central Nervous System-related, and 9) Social Determinants of Health. We explored two state-of-the-art natural language processing (NLP) mode
    
[^46]: 抽象文本摘要中的命名实体包含

    Named Entity Inclusion in Abstractive Text Summarization. (arXiv:2307.02570v1 [cs.CL])

    [http://arxiv.org/abs/2307.02570](http://arxiv.org/abs/2307.02570)

    该论文提出了一种解决抽象文本摘要中命名实体遗漏问题的方法，通过使用定制的预训练目标和模型训练策略，改善了命名实体的包含情况，提高了摘要的准确性和召回率。

    

    我们解决了许多当前抽象文本摘要器的缺点，即命名实体的遗漏问题。我们建议采用定制的预训练目标来增强模型对文本中的命名实体的注意力。首先，使用命名实体识别模型RoBERTa来确定文本中的命名实体。然后，使用该模型对文本中的命名实体进行屏蔽，再使用BART模型对其进行重建。接下来，将BART模型在摘要任务上进行微调。实验证明，这种预训练方法改善了命名实体包含的精确度和召回率指标。

    We address the named entity omission - the drawback of many current abstractive text summarizers. We suggest a custom pretraining objective to enhance the model's attention on the named entities in a text. At first, the named entity recognition model RoBERTa is trained to determine named entities in the text. After that, this model is used to mask named entities in the text and the BART model is trained to reconstruct them. Next, the BART model is fine-tuned on the summarization task. Our experiments showed that this pretraining approach improves named entity inclusion precision and recall metrics.
    
[^47]: 分析ChatGPT在心脏病和血管病理学中的表现

    Analyzing the Performance of ChatGPT in Cardiology and Vascular Pathologies. (arXiv:2307.02518v1 [cs.CL])

    [http://arxiv.org/abs/2307.02518](http://arxiv.org/abs/2307.02518)

    ChatGPT是由OpenAI开发的大型语言模型，该研究分析了ChatGPT在心脏病和血管病理学中的表现。结果表明，ChatGPT在回答挑战性多项选择题方面表现优于医学生，显示了ChatGPT在提供准确答案方面具有潜在的高效性。

    

    该文章旨在分析OpenAI开发的大型语言模型ChatGPT在心脏病和血管病理学领域的表现。该研究评估了ChatGPT在回答挑战性多项选择题的准确性，使用了来自Siamois-QCM平台的190个问题数据集。研究目标是评估ChatGPT在医学教育中与两名成绩优秀的医学生相比的潜力。结果显示，ChatGPT在190道问题中获得了175个正确答案，准确率为92.10\%，而两名学生分别得分163和159，准确率分别为85.78\%和82.63\%。这些结果展示了ChatGPT在心脏病和血管病理学领域具有准确回答相关问题的潜力。

    The article aims to analyze the performance of ChatGPT, a large language model developed by OpenAI, in the context of cardiology and vascular pathologies. The study evaluated the accuracy of ChatGPT in answering challenging multiple-choice questions (QCM) using a dataset of 190 questions from the Siamois-QCM platform. The goal was to assess ChatGPT potential as a valuable tool in medical education compared to two well-ranked students of medicine. The results showed that ChatGPT outperformed the students, scoring 175 out of 190 correct answers with a percentage of 92.10\%, while the two students achieved scores of 163 and 159 with percentages of 85.78\% and 82.63\%, respectively. These results showcase how ChatGPT has the potential to be highly effective in the fields of cardiology and vascular pathologies by providing accurate answers to relevant questions.
    
[^48]: 自然语言生成和理解大型代码用于AI辅助编程：一项综述

    Natural Language Generation and Understanding of Big Code for AI-Assisted Programming: A Review. (arXiv:2307.02503v1 [cs.SE])

    [http://arxiv.org/abs/2307.02503](http://arxiv.org/abs/2307.02503)

    这项综述回顾了大型代码训练的transformer-based大语言模型（LLMs）在AI辅助编程方面的应用，包括代码生成、代码摘要、缺陷检测等。同时讨论了将NLP技术与软件自然化相结合的挑战和机会。

    

    本文全面回顾了自然语言处理（NLP）技术的文献，特别关注使用大型代码进行训练的基于transformer的大型语言模型（LLMs）在AI辅助编程任务领域中的应用。经过软件自然化增强的LLMs在促进AI辅助编程应用方面起到了至关重要的作用，包括代码生成、代码补全、代码翻译、代码优化、代码摘要、缺陷检测和克隆检测。其中著名的应用包括由OpenAI的Codex和DeepMind AlphaCode驱动的GitHub Copilot。本文概述了主要的LLMs及其在与AI辅助编程相关的下游任务中的应用。此外，本文探讨了在这些应用中将NLP技术与软件自然化相结合所面临的挑战和机会，并讨论了扩展AI辅助编程的可能性。

    This paper provides a comprehensive review of the literature concerning the utilization of Natural Language Processing (NLP) techniques, with a particular focus on transformer-based large language models (LLMs) trained using Big Code, within the domain of AI-assisted programming tasks. LLMs, augmented with software naturalness, have played a crucial role in facilitating AI-assisted programming applications, including code generation, code completion, code translation, code refinement, code summarization, defect detection, and clone detection. Notable examples of such applications include the GitHub Copilot powered by OpenAI's Codex and DeepMind AlphaCode. This paper presents an overview of the major LLMs and their applications in downstream tasks related to AI-assisted programming. Furthermore, it explores the challenges and opportunities associated with incorporating NLP techniques with software naturalness in these applications, with a discussion on extending AI-assisted programming 
    
[^49]: 数学智能体：计算基础设施、数学嵌入和基因组学

    Math Agents: Computational Infrastructure, Mathematical Embedding, and Genomics. (arXiv:2307.02502v1 [q-bio.OT])

    [http://arxiv.org/abs/2307.02502](http://arxiv.org/abs/2307.02502)

    本文提出了数学智能体和数学嵌入作为解决生成式人工智能在基因组学应用方面的局限性的新方法，通过使用基于GPT的工作流将文献中的方程转换为LaTeX和Python格式，以实现自动化的大规模评估和交互式计算。

    

    生成式人工智能的发展可以通过更易于理解的数学知识来提升。除了人工智能聊天以外，大型语言模型（LLM）也在编程、算法发现和定理证明方面得到应用，但它们在基因组学应用方面的局限性较大。本项目引入了数学智能体和数学嵌入作为“数学摩尔定律”的新进展，使用基于GPT的工作流将文献中的方程转换为LaTeX和Python格式。虽然存在许多数字方程表示方法，但缺乏自动化的大规模评估工具。LLM对于人工智能聊天提供了语言用户界面，并为大规模的AI辅助计算基础设施提供了形式化语言。鉴于无限的形式可能空间，与数学互动的数学智能体有可能使我们从“大数据”转向“大数学”。数学与自然语言不同，具有可以通过证明来验证的特性，使其在应用范围上更加广泛。

    The advancement in generative AI could be boosted with more accessible mathematics. Beyond human-AI chat, large language models (LLMs) are emerging in programming, algorithm discovery, and theorem proving, yet their genomics application is limited. This project introduces Math Agents and mathematical embedding as fresh entries to the "Moore's Law of Mathematics", using a GPT-based workflow to convert equations from literature into LaTeX and Python formats. While many digital equation representations exist, there's a lack of automated large-scale evaluation tools. LLMs are pivotal as linguistic user interfaces, providing natural language access for human-AI chat and formal languages for large-scale AI-assisted computational infrastructure. Given the infinite formal possibility spaces, Math Agents, which interact with math, could potentially shift us from "big data" to "big math". Math, unlike the more flexible natural language, has properties subject to proof, enabling its use beyond tr
    
[^50]: mPLUG-DocOwl: 模块化多模态大型语言模型用于文档理解

    mPLUG-DocOwl: Modularized Multimodal Large Language Model for Document Understanding. (arXiv:2307.02499v1 [cs.CL])

    [http://arxiv.org/abs/2307.02499](http://arxiv.org/abs/2307.02499)

    mPLUG-DocOwl是一种模块化多模态大型语言模型，用于无OCR文档理解。它通过联合训练语言、通用视觉-语言和文档指令调优数据集，提升了无OCR文档理解能力。

    

    文档理解是指从各种类型的数字文档中自动提取、分析和理解信息，例如网页。现有的多模态大型语言模型（MLLMs），包括mPLUG-Owl，已经展示了有希望的零-shot能力，可以实现无OCR的文本识别，表明它们在无OCR文档理解方面具有潜力。然而，这些模型在没有领域内训练的情况下，往往忽视OCR细粒度特征，如复杂的表格或大块文本，这些特征对于无OCR文档理解是必要的。在本文中，我们基于mPLUG-Owl提出了mPLUG-DocOwl，用于无OCR文档理解。具体而言，我们首先构建了一个包含多种视觉-文本理解任务的指令调优数据集。然后，我们通过针对语言、通用视觉-语言和文档指令调优数据集进行联合训练来增强无OCR文档理解能力。

    Document understanding refers to automatically extract, analyze and comprehend information from various types of digital documents, such as a web page. Existing Multi-model Large Language Models (MLLMs), including mPLUG-Owl, have demonstrated promising zero-shot capabilities in shallow OCR-free text recognition, indicating their potential for OCR-free document understanding. Nevertheless, without in-domain training, these models tend to ignore fine-grained OCR features, such as sophisticated tables or large blocks of text, which are essential for OCR-free document understanding. In this paper, we propose mPLUG-DocOwl based on mPLUG-Owl for OCR-free document understanding. Specifically, we first construct a instruction tuning dataset featuring a wide range of visual-text understanding tasks. Then, we strengthen the OCR-free document understanding ability by jointly train the model on language-only, general vision-and-language, and document instruction tuning dataset with our unified ins
    
[^51]: 自然语言证明规划的演绎可加性研究

    Deductive Additivity for Planning of Natural Language Proofs. (arXiv:2307.02472v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.02472](http://arxiv.org/abs/2307.02472)

    本论文研究了自然语言证明规划中的演绎可加性，探讨了是否能够通过嵌入空间实现高效的规划启发式方法。研究结果表明，嵌入空间的前提陈述总和接近于基于这些前提的结论嵌入。从而证明了演绎可加性的存在。

    

    当前设计用于多步骤命题验证的自然语言系统通常分为两个阶段：使用启发式方法检索一组相关的前提陈述（规划），然后使用大型语言模型从这些陈述中生成新的结论（演绎）。规划阶段通常需要昂贵的Transformer操作，并且无法扩展到任意数量的前提陈述。本文研究了是否可以通过与演绎推理兼容的嵌入空间实现高效的规划启发式方法。具体地，我们评估了嵌入空间是否具有我们称之为演绎可加性的特性：前提陈述嵌入的总和应该接近基于这些前提的结论的嵌入。除了来自GPT3的微调嵌入和来自BM25的稀疏嵌入之外，我们还探索了多种现成的密集嵌入源。我们在内在上研究了嵌入模型，评估了演绎可加性的属性是否存在。

    Current natural language systems designed for multi-step claim validation typically operate in two phases: retrieve a set of relevant premise statements using heuristics (planning), then generate novel conclusions from those statements using a large language model (deduction). The planning step often requires expensive Transformer operations and does not scale to arbitrary numbers of premise statements. In this paper, we investigate whether an efficient planning heuristic is possible via embedding spaces compatible with deductive reasoning. Specifically, we evaluate whether embedding spaces exhibit a property we call deductive additivity: the sum of premise statement embeddings should be close to embeddings of conclusions based on those premises. We explore multiple sources of off-the-shelf dense embeddings in addition to fine-tuned embeddings from GPT3 and sparse embeddings from BM25. We study embedding models both intrinsically, evaluating whether the property of deductive additivity
    
[^52]: 利用ChatGPT生成的数据从社交媒体中检索抑郁症状

    Utilizing ChatGPT Generated Data to Retrieve Depression Symptoms from Social Media. (arXiv:2307.02313v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.02313](http://arxiv.org/abs/2307.02313)

    本文通过利用ChatGPT生成的数据，从社交媒体中检索和排名传达抑郁症状的句子。采用语义搜索和余弦相似度进行句子与BDI-II症状的相关性排名。结果显示合成数据比BDI-II响应更丰富和语义多样化，能有效增加数据和微调下游模型。

    

    本文介绍了BLUE团队在eRisk Lab任务中搜索抑郁症状的贡献。该任务包括从BDI-II问卷中检索和排名Reddit社交媒体句子，这些句子传达了抑郁症状。鉴于LLMs提供的合成数据已被证明是增加数据和微调下游模型的可靠方法，我们选择使用ChatGPT为每个BDI-II问卷的症状生成合成数据。我们设计了一个提示，使生成的数据比每个问题的BDI-II响应更丰富和语义多样化，同时包含了Reddit上分享经历更私密方式中特有的情感和轶事经历。我们通过余弦相似度执行语义搜索并对句子与BDI-II症状的相关性进行排名。我们使用了两种最先进的基于Transformer的模型（MentalRoBERTa和MPNet的变体）进行嵌入。

    In this work, we present the contribution of the BLUE team in the eRisk Lab task on searching for symptoms of depression. The task consists of retrieving and ranking Reddit social media sentences that convey symptoms of depression from the BDI-II questionnaire. Given that synthetic data provided by LLMs have been proven to be a reliable method for augmenting data and fine-tuning downstream models, we chose to generate synthetic data using ChatGPT for each of the symptoms of the BDI-II questionnaire. We designed a prompt such that the generated data contains more richness and semantic diversity than the BDI-II responses for each question and, at the same time, contains emotional and anecdotal experiences that are specific to the more intimate way of sharing experiences on Reddit. We perform semantic search and rank the sentences' relevance to the BDI-II symptoms by cosine similarity. We used two state-of-the-art transformer-based models (MentalRoBERTa and a variant of MPNet) for embeddi
    
[^53]: 转换的原型重构

    Transformed Protoform Reconstruction. (arXiv:2307.01896v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.01896](http://arxiv.org/abs/2307.01896)

    该论文介绍了一个采用Transformer的转换的原型重构模型，相比于基于RNN的编码器-解码器模型，在拉丁语和汉语两个数据集上取得了更好的性能，并探索了模型中潜在的系统发育信号。

    

    原型重构是推断一组子语言中所出现的语素或词汇在祖先语中的情况的任务。Meloni等人（2021）通过采用基于RNN的编码器-解码器与注意力模型，在拉丁语原型重构方面取得了最新进展。我们更新了他们的模型，采用了先进的序列到序列模型：Transformer。我们的模型在两个不同的数据集上，即覆盖了5种语言的8000个同源词的罗曼语数据集和涵盖了39种语言变体的800+同源词的中国数据集（Hou 2004），在一系列不同的度量指标上胜过了他们的模型。我们还对我们的模型进行了潜在的系统发育信号探索。我们的代码可在https://github.com/cmu-llab/acl-2023公开获取。

    Protoform reconstruction is the task of inferring what morphemes or words appeared like in the ancestral languages of a set of daughter languages. Meloni et al. (2021) achieved the state-of-the-art on Latin protoform reconstruction with an RNN-based encoder-decoder with attention model. We update their model with the state-of-the-art seq2seq model: the Transformer. Our model outperforms their model on a suite of different metrics on two different datasets: their Romance data of 8,000 cognates spanning 5 languages and a Chinese dataset (Hou 2004) of 800+ cognates spanning 39 varieties. We also probe our model for potential phylogenetic signal contained in the model. Our code is publicly available at https://github.com/cmu-llab/acl-2023.
    
[^54]: 符合目标：使用通用的插入式框架在CTC模型中优化所需属性

    Align With Purpose: Optimize Desired Properties in CTC Models with a General Plug-and-Play Framework. (arXiv:2307.01715v1 [cs.CL])

    [http://arxiv.org/abs/2307.01715](http://arxiv.org/abs/2307.01715)

    本文提出了一个通用的插入式框架，用于优化CTC模型中的所需属性。该框架通过补充额外的损失项来优先考虑符合所需属性的对齐，并不需要修改CTC损失函数。

    

    连接主义时间分类（CTC）是训练监督序列到序列模型广泛使用的准则。它通过将完美对齐（产生基本事实）的边际化来学习输入和输出序列之间的关系，称为对其，以代价不完美对齐。这种对完美和不完美对齐的二元区分无法捕捉到在其他实际应用中具有重要意义的其他关键对齐属性。在这里，我们提出了$\textit{Align With Purpose}$，这是一个用于增强CTC条件下训练模型中所需属性的$\textbf{通用插入式框架}$。我们通过使用额外的损失项来补充CTC来优先考虑符合所需属性的对齐。我们的方法不需要干预CTC损失函数，能够轻松优化各种属性，并且可以区分完美和不完美的对齐。

    Connectionist Temporal Classification (CTC) is a widely used criterion for training supervised sequence-to-sequence (seq2seq) models. It enables learning the relations between input and output sequences, termed alignments, by marginalizing over perfect alignments (that yield the ground truth), at the expense of imperfect alignments. This binary differentiation of perfect and imperfect alignments falls short of capturing other essential alignment properties that hold significance in other real-world applications. Here we propose $\textit{Align With Purpose}$, a $\textbf{general Plug-and-Play framework}$ for enhancing a desired property in models trained with the CTC criterion. We do that by complementing the CTC with an additional loss term that prioritizes alignments according to a desired property. Our method does not require any intervention in the CTC loss function, enables easy optimization of a variety of properties, and allows differentiation between both perfect and imperfect al
    
[^55]: 图像的重要性：多模态夸张检测的新数据集和实证研究

    Image Matters: A New Dataset and Empirical Study for Multimodal Hyperbole Detection. (arXiv:2307.00209v1 [cs.CV])

    [http://arxiv.org/abs/2307.00209](http://arxiv.org/abs/2307.00209)

    本研究提出了一个新的多模态夸张检测数据集，并使用文本和图像作为两种模态进行研究。同时，评估了不同预训练的多模态编码器在此任务中的表现。该研究探索了夸张检测的跨领域性能。

    

    夸张，即夸大其词，是一种常见的语言现象。夸张检测是理解人类表达的重要部分。已经有几项关于夸张检测的研究，但大多数的研究只关注文本模态。然而，随着社交媒体的发展，人们可以使用各种模态（包括文本、图像、视频等）来表达夸张。在本文中，我们专注于多模态夸张检测。我们从微博（中国的一种社交媒体）创建了一个多模态检测数据集，并对其进行了一些研究。我们将微博的文本和图像视为两种模态，探索了文本和图像在夸张检测中的作用。此外，我们还评估了不同预训练的多模态编码器在这个下游任务上的性能。由于这个数据集是从五个不同的主题构建的，我们还评估了不同领域之间的性能。

    Hyperbole, or exaggeration, is a common linguistic phenomenon. The detection of hyperbole is an important part of understanding human expression. There have been several studies on hyperbole detection, but most of which focus on text modality only. However, with the development of social media, people can create hyperbolic expressions with various modalities, including text, images, videos, etc. In this paper, we focus on multimodal hyperbole detection. We create a multimodal detection dataset\footnote{The dataset will be released to the community.} from Weibo (a Chinese social media) and carry out some studies on it. We treat the text and image from a piece of weibo as two modalities and explore the role of text and image for hyperbole detection. Different pre-trained multimodal encoders are also evaluated on this downstream task to show their performance. Besides, since this dataset is constructed from five different topics, we also evaluate the cross-domain performance of different 
    
[^56]: 生物医学语言模型对不理想的标记分割方式具有鲁棒性

    Biomedical Language Models are Robust to Sub-optimal Tokenization. (arXiv:2306.17649v1 [cs.CL])

    [http://arxiv.org/abs/2306.17649](http://arxiv.org/abs/2306.17649)

    生物医学语言模型对生物医学术语的标记分割方式具有鲁棒性，这对于改进下游生物医学自然语言处理任务的性能非常重要。

    

    与一般的英语相反，生物医学术语中的许多概念是由生物医学专业人员设计的，目的是要精确且简明。通常通过将有意义的生物医学词素连接起来创建新的语义单位来实现这一目标。然而，大多数现代生物医学语言模型 (LMs) 是使用从大规模生物医学语料库统计中导出的标准领域特定标记器进行预训练的，而没有明确利用生物医学语言的粘附性特点。在这项工作中，我们首先发现标准通用领域和生物医学标记器在将生物医学术语分割成有意义的组成部分方面能力有限。因此，我们假设使用一种更准确分割生物医学术语的标记器将使生物医学语言模型在下游生物医学自然语言处理任务中提高性能，特别是涉及生物医学术语的任务，如命名实体识别 (NER) 和实体链接。

    As opposed to general English, many concepts in biomedical terminology have been designed in recent history by biomedical professionals with the goal of being precise and concise. This is often achieved by concatenating meaningful biomedical morphemes to create new semantic units. Nevertheless, most modern biomedical language models (LMs) are pre-trained using standard domain-specific tokenizers derived from large scale biomedical corpus statistics without explicitly leveraging the agglutinating nature of biomedical language. In this work, we first find that standard open-domain and biomedical tokenizers are largely unable to segment biomedical terms into meaningful components. Therefore, we hypothesize that using a tokenizer which segments biomedical terminology more accurately would enable biomedical LMs to improve their performance on downstream biomedical NLP tasks, especially ones which involve biomedical terms directly such as named entity recognition (NER) and entity linking. Su
    
[^57]: 通过Pareto Optimal自监督实现大型语言模型的自动校准和错误修正

    Automatic Calibration and Error Correction for Large Language Models via Pareto Optimal Self-Supervision. (arXiv:2306.16564v1 [cs.CL])

    [http://arxiv.org/abs/2306.16564](http://arxiv.org/abs/2306.16564)

    本文介绍了一种Pareto Optimal自监督框架，利用可用的编程监督将大型语言模型(LLM)的响应进行系统校准，通过为每个响应生成风险评分，而无需额外的手动工作。

    

    大型语言模型(LLM)已经展现了出色的能力，适用于广泛的应用领域，但是准确性仍然是一个重要的增长领域，特别是在生物医学等关键领域。一种有效的方法，用于校准LLM响应的置信水平，对于自动检测错误并促进人机协作验证至关重要。一个重要的校准信号来源是专家指定的编程监督，通常具有较低的成本，但也有其自身的局限性，如噪声和覆盖范围。在本文中，我们引入了一种Pareto Optimal自监督框架，可以利用可用的编程监督来系统地校准LLM响应，通过为每个响应生成风险评分，而不需要任何额外的手动工作。这通过学习一个调和模型来实现，将LLM输出与其他可用的监督来源相协调，将更不确定的响应分配更高的风险评分。

    Large language models (LLMs) have demonstrated remarkable capabilities out of box for a wide range of applications, yet accuracy still remains a major growth area, especially in mission-critical domains such as biomedicine. An effective method to calibrate the confidence level on LLM responses is essential to automatically detect errors and facilitate human-in-the-loop verification. An important source of calibration signals stems from expert-stipulated programmatic supervision, which is often available at low cost but has its own limitations such as noise and coverage. In this paper, we introduce a Pareto optimal self-supervision framework that can leverage available programmatic supervision to systematically calibrate LLM responses by producing a risk score for every response, without any additional manual efforts. This is accomplished by learning a harmonizer model to align LLM output with other available supervision sources, which would assign higher risk scores to more uncertain L
    
[^58]: 2023年唱声转换挑战赛

    The Singing Voice Conversion Challenge 2023. (arXiv:2306.14422v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2306.14422](http://arxiv.org/abs/2306.14422)

    2023年唱声转换挑战赛（SVCC）的最新版本旨在比较和了解不同的唱声转换系统。通过大规模听力测试，我们发现虽然顶尖系统达到了接近人类水平的自然度，但未能达到目标发音者的相似度评分。跨域SVC比域内SVC更困难。

    

    我们呈现了声音转换挑战（VCC）系列的最新版本，这是一个每两年举行一次的科学活动，旨在比较和了解基于共同数据集的不同声音转换（VC）系统。今年，我们把重点转向了唱声转换（SVC），因此将挑战命名为唱声转换挑战（SVCC）。我们构建了一个新的数据库，用于两个任务，即域内和跨域SVC。挑战持续了两个月，我们共收到了26个提交，其中包括2个基准系统。通过大规模众包听力测试，我们观察到，对于两个任务，虽然顶尖系统实现了接近人类水平的自然度，但没有团队能够获得与目标发音者一样高的相似度评分。同时，正如预期的那样，跨域SVC比域内SVC更加困难，特别是在相似度方面。我们还研究了现有的客观评测方法是否能够预测主观性能，并发现只有很少的指标能够实现这一点。

    We present the latest iteration of the voice conversion challenge (VCC) series, a bi-annual scientific event aiming to compare and understand different voice conversion (VC) systems based on a common dataset. This year we shifted our focus to singing voice conversion (SVC), thus named the challenge the Singing Voice Conversion Challenge (SVCC). A new database was constructed for two tasks, namely in-domain and cross-domain SVC. The challenge was run for two months, and in total we received 26 submissions, including 2 baselines. Through a large-scale crowd-sourced listening test, we observed that for both tasks, although human-level naturalness was achieved by the top system, no team was able to obtain a similarity score as high as the target speakers. Also, as expected, cross-domain SVC is harder than in-domain SVC, especially in the similarity aspect. We also investigated whether existing objective measurements were able to predict perceptual performance, and found that only few of th
    
[^59]: 基于大语言模型的中文细粒度金融情感分析

    Chinese Fine-Grained Financial Sentiment Analysis with Large Language Models. (arXiv:2306.14096v1 [cs.CL])

    [http://arxiv.org/abs/2306.14096](http://arxiv.org/abs/2306.14096)

    本文提出了一个用于企业预警的新型、广泛的中文细粒度金融情感分析数据集FinChina SA，并使用现有开源大语言模型对其进行评估和实验。该数据集将成为推进真实金融情感分析任务探索的宝贵资源。

    

    金融领域实体级别的细粒度情感分析是情感分析的重要子任务，目前面临着众多挑战。其中主要挑战之一来自于缺乏专门设计用于金融文本情感分析的高质量大规模标注语料库，这限制了开发有效文本处理技术所需的数据的可用性。大语言模型（LLMs）的最新进展在自然语言处理任务中取得了显著的性能，主要集中在语言模式匹配方面。在本文中，我们提出了一个新颖的、广泛的中文细粒度金融情感分析数据集FinChina SA，用于企业预警。我们对流行的现有开源LLMs使用我们的数据集进行了全面的评估和实验。我们坚信，我们的数据集将成为推动真实世界金融情感分析任务探索的宝贵资源。

    Entity-level fine-grained sentiment analysis in the financial domain is a crucial subtask of sentiment analysis and currently faces numerous challenges. The primary challenge stems from the lack of high-quality and large-scale annotated corpora specifically designed for financial text sentiment analysis, which in turn limits the availability of data necessary for developing effective text processing techniques. Recent advancements in large language models (LLMs) have yielded remarkable performance in natural language processing tasks, primarily centered around language pattern matching. In this paper, we propose a novel and extensive Chinese fine-grained financial sentiment analysis dataset, FinChina SA, for enterprise early warning. We thoroughly evaluate and experiment with well-known existing open-source LLMs using our dataset. We firmly believe that our dataset will serve as a valuable resource to advance the exploration of real-world financial sentiment analysis tasks, which shoul
    
[^60]: ChatGPT和LLMs对医学影像利益相关者的影响：观点和应用案例

    The Impact of ChatGPT and LLMs on Medical Imaging Stakeholders: Perspectives and Use Cases. (arXiv:2306.06767v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2306.06767](http://arxiv.org/abs/2306.06767)

    本研究调查了ChatGPT和LLMs在医学影像领域的变革潜力，它们正在增强放射科医生的解释能力、提升患者与医生之间的沟通，以及简化临床工作流程。

    

    本研究调查了大型语言模型（LLMs）如OpenAI ChatGPT在医学影像领域的变革潜力。借助公共数据，这些模型具有卓越的语言理解和生成能力，正在增强放射科医生的解释能力，提升患者与医生之间的沟通，并简化临床工作流程。本文介绍了一个分析框架，用于展示LLMs与医学影像利益相关者之间的复杂互动，包括企业、保险机构、政府、研究机构和医院（被称为BIGR-H）。通过详细分析、示例应用案例以及对广泛影响和未来方向的讨论，本文旨在提高在AI驱动医疗保健时代的战略规划和决策制定中的讨论水平。

    This study investigates the transformative potential of Large Language Models (LLMs), such as OpenAI ChatGPT, in medical imaging. With the aid of public data, these models, which possess remarkable language understanding and generation capabilities, are augmenting the interpretive skills of radiologists, enhancing patient-physician communication, and streamlining clinical workflows. The paper introduces an analytic framework for presenting the complex interactions between LLMs and the broader ecosystem of medical imaging stakeholders, including businesses, insurance entities, governments, research institutions, and hospitals (nicknamed BIGR-H). Through detailed analyses, illustrative use cases, and discussions on the broader implications and future directions, this perspective seeks to raise discussion in strategic planning and decision-making in the era of AI-enabled healthcare.
    
[^61]: 在生物医学任务上评估ChatGPT：与精调生成式变压器的零样例比较。

    Evaluation of ChatGPT on Biomedical Tasks: A Zero-Shot Comparison with Fine-Tuned Generative Transformers. (arXiv:2306.04504v1 [cs.CL])

    [http://arxiv.org/abs/2306.04504](http://arxiv.org/abs/2306.04504)

    本文评估了ChatGPT在生物医学任务上的表现，发现在生物数据集训练样本较小时，零样例ChatGPT甚至优于精调生成式变压器模型。由此表明ChatGPT具有在生物医学领域成为有价值工具的潜力。

    

    ChatGPT是OpenAI开发的大型语言模型。尽管其在各种任务上表现出色，但先前的工作尚未研究其在生物医学领域的能力。因此，本文旨在评估ChatGPT在各种基准生物医学任务上的性能，如关系提取、文档分类、问答和摘要。据我们所知，这是首次对ChatGPT在生物医学领域进行全面评估的工作。有趣的是，在训练集较小的生物医学数据集中，基于我们的评估结果，零样例ChatGPT甚至优于先进的精调生成式变压器模型，如BioGPT和BioBART。这表明ChatGPT在大型文本语料库上的预训练使其在生物医学领域具有相当的专业性。我们的发现表明，ChatGPT在生物医学领域具有成为各种任务的有价值工具的潜力。

    ChatGPT is a large language model developed by OpenAI. Despite its impressive performance across various tasks, no prior work has investigated its capability in the biomedical domain yet. To this end, this paper aims to evaluate the performance of ChatGPT on various benchmark biomedical tasks, such as relation extraction, document classification, question answering, and summarization. To the best of our knowledge, this is the first work that conducts an extensive evaluation of ChatGPT in the biomedical domain. Interestingly, we find based on our evaluation that in biomedical datasets that have smaller training sets, zero-shot ChatGPT even outperforms the state-of-the-art fine-tuned generative transformer models, such as BioGPT and BioBART. This suggests that ChatGPT's pre-training on large text corpora makes it quite specialized even in the biomedical domain. Our findings demonstrate that ChatGPT has the potential to be a valuable tool for various tasks in the biomedical domain that la
    
[^62]: 基准数据集上 ChatGPT 的系统研究和全面评估

    A Systematic Study and Comprehensive Evaluation of ChatGPT on Benchmark Datasets. (arXiv:2305.18486v1 [cs.CL])

    [http://arxiv.org/abs/2305.18486](http://arxiv.org/abs/2305.18486)

    本文对基准数据集上 ChatGPT 的性能进行了全面的评估，包括问答、文本摘要、代码生成、常识推理、数学问题求解、机器翻译、偏见检测和伦理考虑等任务。研究旨在验证 ChatGPT 的优势和弱点，并为使用语言模型的未来研究提供见解。

    

    最近，如 ChatGPT 这样的大型语言模型（LLM）的开发引起了很多关注。然而，由于难以将该模型生成的产出与基本事实进行比较，因此其在基准学术数据集上的评估仍未充分探索。本文旨在对 ChatGPT 在包括问答、文本摘要、代码生成、常识推理、数学问题求解、机器翻译、偏见检测和伦理考虑等任务中的表现进行彻底评估。具体而言，我们在 140 个任务中评估了 ChatGPT，并分析了其在这些数据集中生成的 255K 次响应，这使我们的工作成为了在 NLP 基准测试中对 ChatGPT 进行的最大评估。简而言之，我们的研究旨在验证 ChatGPT 在各种任务中的优势和弱点，并为使用 LLM 的未来研究提供见解。我们还报告了一种新的迸发能力，即遵循多个查询指令。

    The development of large language models (LLMs) such as ChatGPT has brought a lot of attention recently. However, their evaluation in the benchmark academic datasets remains under-explored due to the difficulty of evaluating the generative outputs produced by this model against the ground truth. In this paper, we aim to present a thorough evaluation of ChatGPT's performance on diverse academic datasets, covering tasks like question-answering, text summarization, code generation, commonsense reasoning, mathematical problem-solving, machine translation, bias detection, and ethical considerations. Specifically, we evaluate ChatGPT across 140 tasks and analyze 255K responses it generates in these datasets. This makes our work the largest evaluation of ChatGPT in NLP benchmarks. In short, our study aims to validate the strengths and weaknesses of ChatGPT in various tasks and provide insights for future research using LLMs. We also report a new emergent ability to follow multi-query instruct
    
[^63]: 基于Transformer模型的社交媒体压力和抑郁识别模型的校准

    Calibration of Transformer-based Models for Identifying Stress and Depression in Social Media. (arXiv:2305.16797v1 [cs.CL])

    [http://arxiv.org/abs/2305.16797](http://arxiv.org/abs/2305.16797)

    本文提出了第一个使用校准后的Transformer模型来检测社交媒体上的压力和抑郁症状的研究。

    

    在当今快节奏的生活中，压力和抑郁症的发病率呈现上升趋势。社交媒体为早期发现心理健康状况提供了帮助。现有的方法主要介绍特征提取方法并训练浅层机器学习分类器。其他研究使用深度神经网络或Transformer模型。尽管Transformer模型取得了明显的改进，但它们通常无法捕捉丰富的实际知识。虽然已经提出了许多旨在增强预训练Transformer模型的研究，但没有先前的工作利用这些修改来通过社交媒体检测压力和抑郁症。此外，尽管机器学习模型在其预测中的置信度可靠性对于高风险应用至关重要，但尚未有先前的工作考虑模型的校准。为解决以上问题，本文提出了第一个通过模型校准来检测社交媒体上的压力和抑郁症状的研究。

    In today's fast-paced world, the rates of stress and depression present a surge. Social media provide assistance for the early detection of mental health conditions. Existing methods mainly introduce feature extraction approaches and train shallow machine learning classifiers. Other researches use deep neural networks or transformers. Despite the fact that transformer-based models achieve noticeable improvements, they cannot often capture rich factual knowledge. Although there have been proposed a number of studies aiming to enhance the pretrained transformer-based models with extra information or additional modalities, no prior work has exploited these modifications for detecting stress and depression through social media. In addition, although the reliability of a machine learning model's confidence in its predictions is critical for high-risk applications, there is no prior work taken into consideration the model calibration. To resolve the above issues, we present the first study i
    
[^64]: 基于自监督学习的语音交流中自我监督表示法在抑郁症检测中的应用

    Self-supervised representations in speech-based depression detection. (arXiv:2305.12263v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12263](http://arxiv.org/abs/2305.12263)

    本文使用基于自监督学习的预训练基础模型解决了语音交流中自动抑郁症检测训练数据稀疏性的问题，并通过微调基础模型将自动语音识别和情感识别的知识转移到抑郁症检测中。实验结果表明，在DAIC-WOZ数据集上实现了基于真实ASR的最先进的抑郁症检测性能。

    

    本文提出使用基于自监督学习（SSL）预训练的基础模型来处理语音交流中自动抑郁症检测（SDD）训练数据的稀疏性。首先，对从不同层次的预训练基础模型中得到的SSL表示进行了SDD分析，从而为抑郁症检测提供了合适的指标见解。然后，通过微调基础模型，从自动语音识别（ASR）和情感识别转移知识到SDD。结果表明，在将ASR模型的隐藏表示与ASR文本信息相结合时，使用oracle和ASR转录产生了类似的SDD性能。通过整合来自多个基础模型的表示，在DAIC-WOZ数据集上实现了基于真实ASR的最先进的SDD结果。

    This paper proposes handling training data sparsity in speech-based automatic depression detection (SDD) using foundation models pre-trained with self-supervised learning (SSL). An analysis of SSL representations derived from different layers of pre-trained foundation models is first presented for SDD, which provides insight to suitable indicator for depression detection. Knowledge transfer is then performed from automatic speech recognition (ASR) and emotion recognition to SDD by fine-tuning the foundation models. Results show that the uses of oracle and ASR transcriptions yield similar SDD performance when the hidden representations of the ASR model is incorporated along with the ASR textual information. By integrating representations from multiple foundation models, state-of-the-art SDD results based on real ASR were achieved on the DAIC-WOZ dataset.
    
[^65]: 从预训练数据到语言模型再到下游任务：追踪导致不公平NLP模型的政治偏见的轨迹。

    From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models. (arXiv:2305.08283v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08283](http://arxiv.org/abs/2305.08283)

    本文研究测量了政治偏见在预训练语言模型和下游任务中的影响，发现预训练模型存在政治倾向，并将社会偏见传递到下游任务中，从而导致NLP模型的不公平性。

    

    语言模型(LMs)是预训练在不同数据源上的，包括新闻、讨论论坛、书籍和在线百科全书等。这些数据中的相当一部分包括观点和角度，一方面赞扬民主和思想多样性，另一方面具有固有的社会偏见。我们的工作开发了新的方法来(1)测量基于此类语料库训练的LMs中的政治偏见，沿社会和经济轴，以及(2)衡量基于政治偏见的LMs训练的下游NLP模型的公平性。我们关注仇恨言论和虚假信息检测，旨在实证量化预训练数据中政治(社会、经济)偏见对高风险社会导向任务公正性的影响。我们的研究结果表明，预训练LMs确实存在政治倾向，加强了预训练语料库中存在的极化，将社会偏见传播到仇恨言论预测和虚假信息检测器中。我们讨论了研究的意义，并提出了开发更公正、更无偏NLP模型的建议。

    Language models (LMs) are pretrained on diverse data sources, including news, discussion forums, books, and online encyclopedias. A significant portion of this data includes opinions and perspectives which, on one hand, celebrate democracy and diversity of ideas, and on the other hand are inherently socially biased. Our work develops new methods to (1) measure political biases in LMs trained on such corpora, along social and economic axes, and (2) measure the fairness of downstream NLP models trained on top of politically biased LMs. We focus on hate speech and misinformation detection, aiming to empirically quantify the effects of political (social, economic) biases in pretraining data on the fairness of high-stakes social-oriented tasks. Our findings reveal that pretrained LMs do have political leanings that reinforce the polarization present in pretraining corpora, propagating social biases into hate speech predictions and misinformation detectors. We discuss the implications of our
    
[^66]: 压缩模型的高效微调：蒸馏还是标注？

    Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models. (arXiv:2305.01645v1 [cs.CL])

    [http://arxiv.org/abs/2305.01645](http://arxiv.org/abs/2305.01645)

    本文研究了压缩模型的高效微调方法，实验结果表明，与手动标注更多的微调数据以直接训练压缩模型相比，从T5-XXL蒸馏到T5-Small几乎总是更具成本效益。

    

    大型模型的微调虽然效果显著，但使用这些模型进行推理成本高且会产生碳排放。知识蒸馏已被证明是减少推理成本的实用解决方案，但蒸馏过程本身需要大量计算资源。本文研究了如何最有效地使用固定预算构建压缩模型。在六个不同的自然语言处理任务上进行了大量实验后，我们发现从T5-XXL（11B）蒸馏到T5-Small（60M）几乎总是比手动标注更多的微调数据以直接训练一个压缩模型（T5-Small（60M））更具成本效益。我们进一步展示了最大化效用的最佳蒸馏量因任务而异。

    Fine-tuning large models is highly effective, however, inference using these models can be expensive and produces carbon emissions. Knowledge distillation has been shown to be a practical solution to reduce inference costs, but the distillation process itself requires significant computational resources. Rather than buying or renting GPUs to fine-tune, then distill a large model, an NLP practitioner who needs a compact model might also choose to simply allocate an available budget to hire annotators and manually label additional fine-tuning data. In this paper, we investigate how to most efficiently use a fixed budget to build a compact model. Through our extensive experiments on six diverse NLP tasks, we find that distilling from T5-XXL (11B) to T5-Small (60M) leads to almost always a cost-efficient option compared to annotating more data to directly train a compact model (T5-Small (60M)). We further demonstrate that the optimal amount of distillation that maximizes utility varies acr
    
[^67]: 无结构数据和结构化数据：我们能否使用大型语言模型获得最佳结果？

    Unstructured and structured data: Can we have the best of both worlds with large language models?. (arXiv:2304.13010v1 [cs.DB])

    [http://arxiv.org/abs/2304.13010](http://arxiv.org/abs/2304.13010)

    本文探讨使用大型语言模型查询无结构数据和结构化数据的潜力及挑战。

    

    本文讨论了使用大型语言模型查询无结构数据和结构化数据的潜力，并概述了构建适用于两种数据类型的问答系统所涉及的一些研究挑战。

    This paper presents an opinion on the potential of using large language models to query on both unstructured and structured data. It also outlines some research challenges related to the topic of building question-answering systems for both types of data.
    
[^68]: 使用指针生成网络和SciBERT嵌入生成研究论文的摘要

    Generation of Highlights from Research Papers Using Pointer-Generator Networks and SciBERT Embeddings. (arXiv:2302.07729v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07729](http://arxiv.org/abs/2302.07729)

    该论文提出了一种使用指针生成网络和SciBERT嵌入来自动生成研究论文亮点的方法。在多个基准数据集上的实验证明，该模型在研究亮点生成方面具有最佳性能。

    

    如今，许多研究文章都以研究亮点作为前言，以总结论文的主要发现。亮点不仅帮助研究人员准确快速地识别论文的贡献，还通过搜索引擎增加了文章的可发现性。我们的目标是在给定研究论文的特定段落的情况下自动构建研究亮点。我们使用了一个具有覆盖机制和上下文嵌入层的指针生成网络，将输入标记编码为SciBERT嵌入。我们在基准数据集CSPubSum上测试了我们的模型，并且还提出了MixSub，一个用于自动生成研究亮点的新的跨学科论文语料库。对于CSPubSum和MixSub，我们观察到所提出的模型相对于相关变体和文献中提出的其他模型来说具有最佳性能。在CSPubSum数据集上，我们的模型在只使用论文的摘要作为输入时表现最佳。

    Nowadays many research articles are prefaced with research highlights to summarize the main findings of the paper. Highlights not only help researchers precisely and quickly identify the contributions of a paper, they also enhance the discoverability of the article via search engines. We aim to automatically construct research highlights given certain segments of a research paper. We use a pointer-generator network with coverage mechanism and a contextual embedding layer at the input that encodes the input tokens into SciBERT embeddings. We test our model on a benchmark dataset, CSPubSum, and also present MixSub, a new multi-disciplinary corpus of papers for automatic research highlight generation. For both CSPubSum and MixSub, we have observed that the proposed model achieves the best performance compared to related variants and other models proposed in the literature. On the CSPubSum dataset, our model achieves the best performance when the input is only the abstract of a paper as op
    
[^69]: 计算机说“不”: 反对有同理心的对话型人工智能的案例

    Computer says "No": The Case Against Empathetic Conversational AI. (arXiv:2212.10983v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10983](http://arxiv.org/abs/2212.10983)

    论文提出反对对话型人工智能过度同理心回应用户情绪的观点，强调需谨慎考虑对用户情绪的回应方式。

    

    情绪是人类认知的重要组成部分，它们不仅指导着我们对世界的理解，也指导着我们在其中的行动。因此，我们对情绪的抚慰或激怒并非无关紧要。近期在对话型人工智能领域的研究一直致力于对用户做出有同理心的回应，验证和抚慰他们的情绪，即使没有真实的基础。这种基于人工智能的情绪调节可能对用户和社会产生负面影响，趋向于将幸福定义为仅仅是“负面”情绪的缺失。我们认为我们必须谨慎考虑是否以及如何回应用户的情绪。

    Emotions are an integral part of human cognition and they guide not only our understanding of the world but also our actions within it. As such, whether we soothe or flame an emotion is not inconsequential. Recent work in conversational AI has focused on responding empathetically to users, validating and soothing their emotions without a real basis. This AI-aided emotional regulation can have negative consequences for users and society, tending towards a one-noted happiness defined as only the absence of "negative" emotions. We argue that we must carefully consider whether and how to respond to users' emotions.
    
[^70]: 高效节约内存的NLLB-200：针对大规模多语言机器翻译模型的语言特定专家删减

    Memory-efficient NLLB-200: Language-specific Expert Pruning of a Massively Multilingual Machine Translation Model. (arXiv:2212.09811v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09811](http://arxiv.org/abs/2212.09811)

    本研究提出了一种节约内存的NLLB-200模型修剪方法，可在保持翻译质量的同时移除多达80％的专家，使得在单个32GB的GPU上运行模型成为可能。这对于大规模多语言机器翻译具有重要的意义。

    

    与传统的双语翻译系统相比，大规模多语言机器翻译具有吸引力，因为一个单一模型可以翻译成多种语言，并从知识转移中获益，尤其是对于低资源语言。然而，大规模多语言模型受到多语言性的限制，除非进行大规模扩展，否则会增加训练和推理成本。稀疏的专家混合模型是一种在不需要大量计算的情况下大幅增加模型容量的方法。最近发布的NLLB-200是这样一个模型的例子。它涵盖了202种语言，但仅推理就需要至少四个32GB的GPU。在这项工作中，我们提出了一种修剪方法，允许删除多达80％的专家，但翻译质量几乎没有损失，这使得在单个32GB的GPU上运行该模型成为可能。进一步分析表明，我们的修剪度量指标可以识别出语言特定的专家

    Compared to conventional bilingual translation systems, massively multilingual machine translation is appealing because a single model can translate into multiple languages and benefit from knowledge transfer for low resource languages. On the other hand, massively multilingual models suffer from the curse of multilinguality, unless scaling their size massively, which increases their training and inference costs. Sparse Mixture-of-Experts models are a way to drastically increase model capacity without the need for a proportional amount of computing. The recently released NLLB-200 is an example of such a model. It covers 202 languages but requires at least four 32GB GPUs just for inference. In this work, we propose a pruning method that allows the removal of up to 80\% of experts with a negligible loss in translation quality, which makes it feasible to run the model on a single 32GB GPU. Further analysis suggests that our pruning metrics allow to identify language-specific experts and p
    
[^71]: 在多标签文本分类中有效地使用对比学习

    An Effective Employment of Contrastive Learning in Multi-label Text Classification. (arXiv:2212.00552v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.00552](http://arxiv.org/abs/2212.00552)

    本论文提出了五个新的对比损失函数，用于多标签文本分类任务。通过对比学习技术的应用，探索了其在多标签文本分类任务中的有效性，并提供了一套基准模型。

    

    对比学习技术在自然语言处理任务中的有效性尚待探索和分析。如何正确合理地构建正负样本是对比学习的核心挑战，而在多标签文本分类任务中，发现对比对象更加困难。之前提出的对比损失函数很少。在本文中，我们从不同的角度探讨这个问题，提出了五个新颖的对比损失函数，用于多标签文本分类任务，包括严格对比损失 (SCL)、内标签对比损失 (ICL)、Jaccard相似度对比损失(JSCL)、Jaccard相似度概率对比损失(JSPCL)和逐步标签对比损失(SLCL)。我们通过使用这些新颖的损失函数，探索了对比学习在多标签文本分类任务中的有效性，并为在特定任务上部署对比学习技术提供了一套基准模型。

    The effectiveness of contrastive learning technology in natural language processing tasks is yet to be explored and analyzed. How to construct positive and negative samples correctly and reasonably is the core challenge of contrastive learning. It is even harder to discover contrastive objects in multi-label text classification tasks. There are very few contrastive losses proposed previously. In this paper, we investigate the problem from a different angle by proposing five novel contrastive losses for multi-label text classification tasks. These are Strict Contrastive Loss (SCL), Intra-label Contrastive Loss (ICL), Jaccard Similarity Contrastive Loss (JSCL), Jaccard Similarity Probability Contrastive Loss (JSPCL), and Stepwise Label Contrastive Loss (SLCL). We explore the effectiveness of contrastive learning for multi-label text classification tasks by the employment of these novel losses and provide a set of baseline models for deploying contrastive learning techniques on specific t
    
[^72]: 具有真正零-shot能力的弱监督流式多语言语音模型

    A Weakly-Supervised Streaming Multilingual Speech Model with Truly Zero-Shot Capability. (arXiv:2211.02499v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.02499](http://arxiv.org/abs/2211.02499)

    本文介绍了一个弱监督流式多语言语音模型，利用机器翻译服务将语音识别转录转化为弱监督数据来训练模型。该模型具有真正的零-shot能力，可以在扩展到新的目标语言时产生高质量的语音翻译结果。

    

    本文介绍了我们构建的流式多语言语音模型（SM2）的工作，该模型可以将多种口语语言转录或翻译为目标语言的文本。SM2的核心是Transformer Transducer，具有高度流式处理能力。我们使用通过机器翻译服务将语音识别语料库中的转录转化而成的弱监督数据来训练SM2模型，而非人工标记的语音翻译数据。利用来自25种语言的35.1万小时的匿名语音训练数据，SM2模型的语音翻译质量与某些最新的大规模非流式语音模型相当甚至更好。更重要的是，我们展示了当扩展到新的目标语言时，SM2具有真正的零-shot能力，可以为训练过程中未见过的{源语音，目标文本}对产生高质量的语音翻译结果。

    In this paper, we introduce our work of building a Streaming Multilingual Speech Model (SM2), which can transcribe or translate multiple spoken languages into texts of the target language. The backbone of SM2 is Transformer Transducer, which has high streaming capability. Instead of human labeled speech translation (ST) data, SM2 models are trained using weakly supervised data generated by converting the transcriptions in speech recognition corpora with a machine translation service. With 351 thousand hours of anonymized speech training data from 25 languages, SM2 models achieve comparable or even better ST quality than some recent popular large-scale non-streaming speech models. More importantly, we show that SM2 has the truly zero-shot capability when expanding to new target languages, yielding high quality ST results for {source-speech, target-text} pairs that are not seen during training.
    
[^73]: 基于逆对比学习的中文拼写检查框架

    A Chinese Spelling Check Framework Based on Reverse Contrastive Learning. (arXiv:2210.13823v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.13823](http://arxiv.org/abs/2210.13823)

    提出了一种基于逆对比学习的框架用于中文拼写检查，通过增强模型对混淆词的区分能力，有效提高了检测和纠正的准确性和能力。

    

    中文拼写检查是一项检测和纠正中文文本中拼写错误的任务。现有研究旨在增强文本表示，并利用多源信息提高模型的检测和纠正能力，但对于改善模型区分混淆词的能力并未给予足够关注。对比学习近年来在自然语言处理中成为一种主要技术，其目标是在表示空间中最小化相似样本对之间的距离。受对比学习的启发，我们提出了一种新颖的中文拼写检查框架，包括语言表示、拼写检查和逆对比学习三个模块。具体而言，我们提出了一种逆对比学习策略，明确要求模型最小化相似例子之间的一致性，即音符和视觉上易混淆的字符。实验结果表明，我们的框架能够提高中文拼写检查的准确性和纠错能力。

    Chinese spelling check is a task to detect and correct spelling mistakes in Chinese text. Existing research aims to enhance the text representation and use multi-source information to improve the detection and correction capabilities of models, but does not pay too much attention to improving their ability to distinguish between confusable words. Contrastive learning, whose aim is to minimize the distance in representation space between similar sample pairs, has recently become a dominant technique in natural language processing. Inspired by contrastive learning, we present a novel framework for Chinese spelling checking, which consists of three modules: language representation, spelling check and reverse contrastive learning. Specifically, we propose a reverse contrastive learning strategy, which explicitly forces the model to minimize the agreement between the similar examples, namely, the phonetically and visually confusable characters. Experimental results show that our framework i
    
[^74]: Android们是否会笑电子羊？源自纽约客漫画字幕比赛的幽默“理解”评估。

    Do Androids Laugh at Electric Sheep? Humor "Understanding" Benchmarks from The New Yorker Caption Contest. (arXiv:2209.06293v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2209.06293](http://arxiv.org/abs/2209.06293)

    通过纽约客漫画字幕比赛的任务，我们挑战了AI模型对幽默的“理解”。结果发现，无论是多模态模型还是仅文本模型，在配对笑话和漫画、识别获胜字幕以及解释获胜字幕为什么有趣的任务上都存在困难。

    

    目前大型神经网络能够生成笑话，但它们真正“理解”幽默吗？我们通过三个任务挑战AI模型，这些任务源自于纽约客漫画字幕比赛：将笑话与漫画配对、识别获胜字幕并解释为什么获胜的字幕很有趣。这些任务逐渐包含了“理解”漫画的更复杂方面；关键因素是图像与字幕之间的复杂、常常令人惊讶的关系，以及频繁包含对人类经验和文化的间接和富有玩味的暗示。我们研究了多模态和仅文本模型：前者直接面对漫画图像进行挑战，而后者则给出了多方面的视觉场景描述以模拟人类级别的视觉理解。我们发现两种类型的模型在所有三个任务上都面临困难。例如，我们最好的多模态模型在配对任务上的准确率比人类表现低30个百分点，即便在提供了图像描述的情况下。

    Large neural networks can now generate jokes, but do they really "understand" humor? We challenge AI models with three tasks derived from the New Yorker Cartoon Caption Contest: matching a joke to a cartoon, identifying a winning caption, and explaining why a winning caption is funny. These tasks encapsulate progressively more sophisticated aspects of "understanding" a cartoon; key elements are the complex, often surprising relationships between images and captions and the frequent inclusion of indirect and playful allusions to human experience and culture. We investigate both multimodal and language-only models: the former are challenged with the cartoon images directly, while the latter are given multifaceted descriptions of the visual scene to simulate human-level visual understanding. We find that both types of models struggle at all three tasks. For example, our best multimodal models fall 30 accuracy points behind human performance on the matching task, and, even when provided gr
    
[^75]: 针对神经机器翻译及其扩展的非自回归生成的调查

    A Survey on Non-Autoregressive Generation for Neural Machine Translation and Beyond. (arXiv:2204.09269v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.09269](http://arxiv.org/abs/2204.09269)

    这项调查研究了非自回归生成在神经机器翻译以及其他领域的应用。研究发现，尽管非自回归生成可以加快推理速度，但与自回归生成相比存在翻译准确性的损失。然而，通过各种方法和算法的改进，可以缩小这一准确性差距。

    

    非自回归（NAR）生成首次在神经机器翻译（NMT）中提出，旨在加速推理过程，并引起了机器学习和自然语言处理领域的广泛关注。尽管NAR生成可以显著加快机器翻译的推理速度，但与其对应的自回归（AR）生成相比，其翻译准确性有所降低。近年来，许多新模型和算法已被设计/提出以弥补NAR生成与AR生成之间的准确性差距。在本文中，我们对不同方面的各种非自回归翻译（NAT）模型进行了系统调查，并进行了比较和讨论。具体而言，我们将NAT的努力分成了几个方向，包括数据处理、建模方法、训练标准、解码算法以及来自预训练模型的益处。此外，我们还简要回顾了NAR模型的其他应用。

    Non-autoregressive (NAR) generation, which is first proposed in neural machine translation (NMT) to speed up inference, has attracted much attention in both machine learning and natural language processing communities. While NAR generation can significantly accelerate inference speed for machine translation, the speedup comes at the cost of sacrificed translation accuracy compared to its counterpart, autoregressive (AR) generation. In recent years, many new models and algorithms have been designed/proposed to bridge the accuracy gap between NAR generation and AR generation. In this paper, we conduct a systematic survey with comparisons and discussions of various non-autoregressive translation (NAT) models from different aspects. Specifically, we categorize the efforts of NAT into several groups, including data manipulation, modeling methods, training criterion, decoding algorithms, and the benefit from pre-trained models. Furthermore, we briefly review other applications of NAR models 
    

