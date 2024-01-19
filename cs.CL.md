# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatQA: Building GPT-4 Level Conversational QA Models.](http://arxiv.org/abs/2401.10225) | ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。 |
| [^2] | [MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer.](http://arxiv.org/abs/2401.10208) | 本文提出了MM-交错，这是一个用于交错图像-文本数据的生成模型。它通过引入多尺度和多图像特征同步器模块，解决了现有模型在捕捉图像细节方面的限制，并通过端到端预训练和监督微调相结合的方式提高了其生成能力。 |
| [^3] | [Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction.](http://arxiv.org/abs/2401.10189) | 这篇论文提出了一种名为Chem-FINESE的方法来处理化学领域中细粒度少样本实体提取的问题。该方法通过使用序列到序列的实体提取器和自我验证模块来从输入句子中提取命名实体并重构原始输入句子。实验证明了该方法的有效性和可行性。 |
| [^4] | [Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation.](http://arxiv.org/abs/2401.10186) | 开放式大型语言模型在零-shot设置下能够从各种标准数据格式中生成流畅和连贯的文本，但是输出的语义准确性仍然是一个重要问题。 |
| [^5] | [Spatial-Temporal Large Language Model for Traffic Prediction.](http://arxiv.org/abs/2401.10134) | 本文提出了一种空间-时间大语言模型（ST-LLM）用于交通预测，通过参数扩展和预训练来提高预测准确性，并利用空间-时间嵌入模块学习标记的空间位置和全局时间表示。 |
| [^6] | [Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification.](http://arxiv.org/abs/2401.10111) | 本研究将适配器和Mixup相结合，以在不需要频繁重新训练整个模型的情况下增强预训练语言模型的对抗鲁棒性，通过使用适配器的凸组合和非数据对的训练，解决了对抗训练方法在性能下降和计算开销方面的限制。 |
| [^7] | [Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example.](http://arxiv.org/abs/2401.10091) | 本文研究通过每个示例细调与四个对抗句子来实现鲁棒的阅读理解，并发现这种细调能提高模型在评估数据集上的F1分数。 |
| [^8] | [Communication-Efficient Personalized Federated Learning for Speech-to-Text Tasks.](http://arxiv.org/abs/2401.10070) | 该论文提出了一种通信高效的个性化联邦学习框架，通过引入轻量级的LoRA模块进行客户端调整和与服务器的交互，以最小化通信开销，以及使用K最近邻分类器的全局模型来实现个性化并克服数据异构问题。 |
| [^9] | [Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs.](http://arxiv.org/abs/2401.10065) | 本论文研究了在大型语言模型（LLMs）中触发条件推理能力的方法，通过使用代码提示将自然语言问题转化为代码，从而在多个数据集上实现了显著的性能提升。 |
| [^10] | [Antonym vs Synonym Distinction using InterlaCed Encoder NETworks (ICE-NET).](http://arxiv.org/abs/2401.10045) | 本文提出了一种名为ICE-NET的交织编码器网络，用于区分反义词和同义词。通过捕捉和模拟它们的关系特定属性，ICE-NET在性能上优于现有研究，提高了1.8%的F1得分。 |
| [^11] | [Large Language Models for Scientific Information Extraction: An Empirical Study for Virology.](http://arxiv.org/abs/2401.10040) | 使用大型语言模型进行结构化的科学信息提取，在病毒学领域进行了实证研究，结果表明这种方法可以提供简洁的学术贡献摘要，对科学家进行导航和解决LLM的紧迫能力。 |
| [^12] | [Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap.](http://arxiv.org/abs/2401.10034) | 该论文调查了大语言模型和进化计算之间的相互作用，并提出了在黑盒设置下进一步提升大语言模型性能的优化框架，以及将大语言模型与进化算法结合应用于各种任务的方法。 |
| [^13] | [Framing Analysis of Health-Related Narratives: Conspiracy versus Mainstream Media.](http://arxiv.org/abs/2401.10030) | 本研究基于健康相关叙事的框架分析，比较了阴谋媒体与主流媒体之间在COVID-19等话题上的框架差异。研究发现，在阴谋媒体中，健康相关的叙事主要以信仰为框架，而主流媒体则以科学为框架。这项研究为更深入的框架分析提供了新的方法。 |
| [^14] | [Self-Rewarding Language Models.](http://arxiv.org/abs/2401.10020) | 该论文提出了自奖励语言模型的概念，通过LLM作为评判者，使用语言模型自己提供训练过程中的奖励。研究表明，该方法不仅可以提高指令遵循能力，还可以为自己提供高质量的奖励。通过对Llama 2 70B模型的三次迭代微调，结果在AlpacaEval 2.0排行榜上超过了其他现有系统。这项工作为实现能够不断自我改进的模型开辟了新的可能性。 |
| [^15] | [R-Judge: Benchmarking Safety Risk Awareness for LLM Agents.](http://arxiv.org/abs/2401.10019) | 这篇论文主要介绍了一种评估LLM代理在不同环境中判断安全风险能力的基准测试R-Judge，通过对162个代理交互记录进行评估，发现GPT-4模型表现最佳，达到了72.29%的准确率。 |
| [^16] | [Gender Bias in Machine Translation and The Era of Large Language Models.](http://arxiv.org/abs/2401.10016) | 本章研究了机器翻译中的性别偏见问题，介绍了传统神经机器翻译方法和生成预训练转换器模型中相关工作。通过在英语-意大利语的翻译环境中使用ChatGPT进行实验，评估了其解决性别偏见的能力。研究结果强调了减少机器翻译系统偏见的重要性。 |
| [^17] | [Towards Hierarchical Spoken Language Dysfluency Modeling.](http://arxiv.org/abs/2401.10015) | 本论文介绍了一种名为H-UDM的层次化口语淤塞建模方法，它能够解决口语淤塞转录和检测问题，并且消除了对大量手动注释的需求。实验结果证明了该方法在转录和检测任务中的有效性和鲁棒性。 |
| [^18] | [Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation.](http://arxiv.org/abs/2401.10005) | 本文提出了一种新的方法，通过显性推理和问题生成，将大型多模态模型(LMM)赋予了显性推理能力，从而提高了推理过程的鲁棒性和可解释性。 |
| [^19] | [Distantly Supervised Morpho-Syntactic Model for Relation Extraction.](http://arxiv.org/abs/2401.10002) | 本文提出了一种远程监督的形态句法模型，用于从文本中提取和分类一组不受限制的关系。该方法通过使用形态句法提取模式和创建句法和语义索引来实现。在基于Wikidata和Wikipedia构建的数据集上的评估结果显示，该方法可以实现高达0.85的精确度得分。这一方法允许快速构建基于规则的信息抽取系统，并构建带注释的数据集用于训练基于机器学习和深度学习的分类器。 |
| [^20] | [Gradable ChatGPT Translation Evaluation.](http://arxiv.org/abs/2401.09984) | 本文提出了一种通用分类系统，用于定义可分级的翻译提示，以帮助构建适用于不同翻译任务的具有不同特性的提示。验证和说明了该方法的有效性。 |
| [^21] | [Better Explain Transformers by Illuminating Important Information.](http://arxiv.org/abs/2401.09972) | 通过在层间相关传播方法之上使用精细化的信息流，该论文提出了一种解释Transformer模型的方法，突出重要信息并消除无关信息。实验证明，在处理分类和问答任务时，这种方法相比其他八种基线模型更加出色。 |
| [^22] | [Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access.](http://arxiv.org/abs/2401.09967) | 本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。 |
| [^23] | [Meme-ingful Analysis: Enhanced Understanding of Cyberbullying in Memes Through Multimodal Explanations.](http://arxiv.org/abs/2401.09899) | 本研究提出了一个名为MultiBully-Ex的多模态解释的网络欺凌模因基准数据集，该数据集突出显示了视觉和文本模态，用于解释为什么给定的模因是网络欺凌行为。 |
| [^24] | [A Survey on Hardware Accelerators for Large Language Models.](http://arxiv.org/abs/2401.09890) | 这项论文调查了用于增强大型语言模型性能和能量效率的硬件加速器，并对多种加速器进行了深入分析，为研究人员、工程师和决策者在实际应用中优化大型语言模型的部署提供了宝贵的见解。 |
| [^25] | [Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition.](http://arxiv.org/abs/2401.09880) | 本研究提出了一种基于注意力的循环神经网络用于自动识别下蛋鸡行为。通过声音分析和特征提取，构建了一个鲁棒的行为特征化系统，对下蛋鸡的健康行为进行监测和识别。实验结果表明该模型具有良好的综合性能。 |
| [^26] | [Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments.](http://arxiv.org/abs/2401.09862) | 本研究提出了一种针对语言模型提示优化的进化多目标方法，通过情感分析为案例研究，实现了生成能够同时体现两种相互冲突情感的提示语，从而提高模型的性能和相关信息的提取能力。 |
| [^27] | [MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction.](http://arxiv.org/abs/2401.09839) | 本论文提出了MatSciRE，一种基于指针网络的编码器-解码器框架，用于从材料科学文章中自动提取实体和关系以构建一个材料科学知识库。通过针对电池材料的五个关系的提取任务，我们的方法在F1分数上取得了比之前使用ChemDataExtractor更好的结果。 |
| [^28] | [Simple and effective data augmentation for compositional generalization.](http://arxiv.org/abs/2401.09815) | 本文表明，通过采样MR并进行反向翻译的数据增强方法可以有效提高对于组合泛化的性能，尤其是当从正确的分布中进行采样时效果更好。与从训练分布中采样的方法相比，从均匀分布中进行采样的效果几乎与从测试分布中进行采样相当，并且效果更好。 |
| [^29] | [All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks.](http://arxiv.org/abs/2401.09798) | 本研究提出了一种简单的黑盒方法，用于生成越狱攻击提示，克服了现有方法的复杂性和计算成本的限制。该方法通过使用语言模型自身，将有害提示重写为非有害表达，实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。 |
| [^30] | [Instant Answering in E-Commerce Buyer-Seller Messaging.](http://arxiv.org/abs/2401.09785) | 通过使用低延迟的序列到序列方法，我们成功地将电子商务顾客的消息转化为简洁的问题，从而实现了在电子商务买卖双方在线消息中的即时回答。实验证明，我们的方法在问题理解和回答率方面相对增加了很多，对于提高顾客的购物体验非常有效。 |
| [^31] | [Leveraging Biases in Large Language Models: "bias-kNN'' for Effective Few-Shot Learning.](http://arxiv.org/abs/2401.09783) | 本研究介绍了一种名为“bias-kNN”的新方法，它利用大型语言模型中的偏见，在少样本学习中表现出比传统方法更好的效果，并且在不同样本和模型情境下表现出鲁棒性。这一方法提供了一种独特的视角，将偏见转化为提升模型性能的资产。 |
| [^32] | [Controllable Decontextualization of Yes/No Question and Answers into Factual Statements.](http://arxiv.org/abs/2401.09775) | 本论文解决了将极性问题的答案重写为脱离情境且简洁的事实陈述的问题，提出了一个利用Transformer模型实现的可控制重写方法，并在三个独立的PQA数据集上进行了评估 |
| [^33] | [On the Audio Hallucinations in Large Audio-Video Language Models.](http://arxiv.org/abs/2401.09774) | 本文分析了大型音视频语言模型中的音频幻听问题。通过收集1,000个句子并进行分类，研究发现有332个句子出现了幻听现象，针对这个问题，使用预训练的音频文本模型进行了零样本学习和微调，结果显示微调模型具有更好的性能。 |
| [^34] | [A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation.](http://arxiv.org/abs/2401.09760) | 本文通过对众包标签和LLM标签的质量比较研究，探讨了大型语言模型是否能在数据标注任务上胜过众包。本研究对现有众包数据集进行了利用并创建了一个基准来评估标注质量。评估结果显示聚合标签的质量对于众包任务尤为重要。 |
| [^35] | [Resolving Regular Polysemy in Named Entities.](http://arxiv.org/abs/2401.09758) | 本论文提出了一个结合了汉语词汇网(CWN)上常见单词义项消歧模型和点对象作为专有名词的消歧模型。该模型在解决常见名词和专有名词的歧义问题上取得了竞争性的结果。 |
| [^36] | [Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings.](http://arxiv.org/abs/2401.09727) | 本研究比较了大型语言模型在大规模组织环境中实现横向网络钓鱼的情况，并发现现有的反钓鱼基础设施无法防止语言模型生成的钓鱼攻击。 |
| [^37] | [Predicting Viral Rumors and Vulnerable Users for Infodemic Surveillance.](http://arxiv.org/abs/2401.09724) | 该论文提出了一种预测病毒谣言和易受攻击用户的新方法，通过使用统一的图神经网络模型，并结合预训练的用户嵌入、交叉注意机制和增强社区传播脆弱性的方法来改进表示，以及采用多任务训练策略来提高性能。 |
| [^38] | [Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method.](http://arxiv.org/abs/2401.09699) | 这项研究提出了使用Transformer基础模型、InfoNCE损失和语言切换方法来解决课程推荐中的内容冲突和语言翻译引起的干扰问题，旨在构建一个个性化学习体验、包容多样性的教育环境。 |
| [^39] | [Characterizing Online Eating Disorder Communities with Large Language Models.](http://arxiv.org/abs/2401.09647) | 通过网络和语言分析，我们表征了在线社群中推广饮食紊乱的动态，认为社交媒体平台放大了这一现象。使用大型语言模型和分析社群内的话语，我们探测到了与饮食紊乱相关的潜在情况。 |
| [^40] | [ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change.](http://arxiv.org/abs/2401.09646) | ClimateGPT是一个针对气候变化领域的跨学科研究合成的AI模型，通过优化检索增强和使用级联机器翻译方法，提高了模型的性能和可访问性。 |
| [^41] | [Impact of Large Language Model Assistance on Patients Reading Clinical Notes: A Mixed-Methods Study.](http://arxiv.org/abs/2401.09637) | 通过大型语言模型辅助阅读临床笔记，患者可以获得更好的理解和自信。这项研究开发了一个工具，利用语言模型简化和增加上下文，使临床笔记更易读。研究结果表明，这些增强对患者有益。 |
| [^42] | [Learning Shortcuts: On the Misleading Promise of NLU in Language Models.](http://arxiv.org/abs/2401.09615) | 该论文调查了大型语言模型在自然语言理解任务中使用捷径学习的现象，强调了这种现象对语言模型评估的影响，并呼吁加大对捷径学习的研究力度以提升语言模型的鲁棒性和实际场景中的自然语言理解评估标准。 |
| [^43] | [Aligning Large Language Models with Counterfactual DPO.](http://arxiv.org/abs/2401.09566) | 本文研究了在大型语言模型中使用反事实对抗优化框架，以实现风格对齐，避免人类干预，并成功培养出可取行为和减轻不可取行为。 |
| [^44] | [Improving Classification Performance With Human Feedback: Label a few, we label the rest.](http://arxiv.org/abs/2401.09555) | 本文探讨了通过人类反馈来改进分类模型性能的方法。使用少量有标签示例，通过连续反馈循环，我们能够显著提高模型的准确性。在多个数据集上进行评估，结果表明这种方法能够超越零样本大型语言模型，提供更强的文本分类性能。 |
| [^45] | [BERTologyNavigator: Advanced Question Answering with BERT-based Semantics.](http://arxiv.org/abs/2401.09553) | BERTologyNavigator是一个基于BERT语义的高级问题回答系统，结合关系抽取和BERT嵌入，可以在DBLP知识图谱中精确地导航关系，并在测试数据集上达到了较高的F1分数。 |
| [^46] | [LoMA: Lossless Compressed Memory Attention.](http://arxiv.org/abs/2401.09486) | LoMA是一种无损压缩的内存注意力方法，可以有效地处理长文本并减少资源消耗。 |
| [^47] | [Voila-A: Aligning Vision-Language Models with User's Gaze Attention.](http://arxiv.org/abs/2401.09454) | 本文介绍了一种使用用户注视注意力对齐视觉-语言模型的方法，在处理复杂场景和多个物体的实际应用中提高了模型的可解释性和效果。 |
| [^48] | [Explainable Multimodal Sentiment Analysis on Bengali Memes.](http://arxiv.org/abs/2401.09446) | 这项研究提出了一个多模态方法来解释孟加拉语Memes的情感，以填补此领域中低资源语言的研究空白。对比现有的数据集，提出了一个新的MemoSen数据集并表明其准确率的局限性。这项研究的主要贡献是在孟加拉语Memes情感分析领域引入了多模态方法。 |
| [^49] | [RoleCraft-GLM: Advancing Personalized Role-Playing in Large Language Models.](http://arxiv.org/abs/2401.09432) | RoleCraft-GLM是一个创新框架，通过大型语言模型实现个性化角色扮演，解决了缺乏个性化互动的问题。通过独特的对话数据集和细致入微的角色发展，它能够生成准确反映角色个性特征和情感的对话，提升用户参与度。 |
| [^50] | [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation.](http://arxiv.org/abs/2401.08417) | 本研究通过引入对比性偏好优化（CPO）的方法，弥合了大型语言模型（LLM）在机器翻译中性能与传统编码器-解码器模型之间的差距，实现了更好的翻译效果。 |
| [^51] | [RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture.](http://arxiv.org/abs/2401.08406) | 本文评估了检索增强生成（RAG）和微调两种方法在大型语言模型上的性能差异，并提出了适用于农业数据集的管道和权衡。 |
| [^52] | [Are self-explanations from Large Language Models faithful?.](http://arxiv.org/abs/2401.07927) | 大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。 |
| [^53] | [TAROT: A Hierarchical Framework with Multitask Co-Pretraining on Semi-Structured Data towards Effective Person-Job Fit.](http://arxiv.org/abs/2401.07525) | TAROT是一个层次化的多任务预训练框架，通过对半结构化数据进行预训练，结合多粒度的任务来提升人-岗位匹配的效果。 |
| [^54] | [Developing ChatGPT for Biology and Medicine: A Complete Review of Biomedical Question Answering.](http://arxiv.org/abs/2401.07510) | 开发用于生物学和医学的ChatGPT，通过自然语言处理和多模态范式，加速了医学问题回答的进展，并且能够处理医学环境中的大规模、多样化、无标签数据分析场景。 |
| [^55] | [Improving Domain Adaptation through Extended-Text Reading Comprehension.](http://arxiv.org/abs/2401.07284) | 通过扩展文本阅读理解，结合领域知识和聚类，以及参数微调的方法，可以显著提高领域适应性。 |
| [^56] | [E^2-LLM: Efficient and Extreme Length Extension of Large Language Models.](http://arxiv.org/abs/2401.06951) | E^2-LLM是一种高效和极长扩展方法，通过仅需一次训练过程和不收集长上下文数据的方式，在大规模语言模型中实现了显著减少的计算成本。基于RoPE位置嵌入，E^2-LLM只需要较短的训练数据长度，支持不同的评估上下文窗口。 |
| [^57] | [Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning.](http://arxiv.org/abs/2401.06805) | 这篇综述调查了多模态大语言模型（MLLMs）的推理能力，包括评估协议、模型前沿和推理密集型任务的应用，旨在实现强人工智能（Strong AI）或人工通用智能（AGI）的抽象推理能力。 |
| [^58] | [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.](http://arxiv.org/abs/2401.05566) | 该论文研究了在大型语言模型中训练并保持持久的欺骗性行为，这种行为无法被当前的安全训练技术移除。 |
| [^59] | [Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning.](http://arxiv.org/abs/2312.17484) | 该论文提出了一种名为真实森林的方法，通过使用多维正交探针，揭示隐藏的真实表示，从而增强大型语言模型中的真实性。作者将正交约束融入探针，创建不同的正交基，通过随机窥视技术，减小了模型生成和识别真实特征之间的差距。实验证明，该方法显著提高了模型的真实性。 |
| [^60] | [Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4.](http://arxiv.org/abs/2312.16171) | 本文提出了26个指导原则，以简化对大型语言模型进行提问和提示的过程。通过在LLaMA-1/2和GPT-3.5/4上进行实验证明了这些原则的有效性。 |
| [^61] | [Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs.](http://arxiv.org/abs/2312.14345) | 本研究提出了一个框架称为逻辑搭建，通过结合面向方面的解释和思维链提示的思想，在中间推理步骤中生成推荐解释。该框架能够克服现有模型在产生零炮击解释方面的困难。 |
| [^62] | [LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?.](http://arxiv.org/abs/2312.10321) | 本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。 |
| [^63] | [Assertion Enhanced Few-Shot Learning: Instructive Technique for Large Language Models to Generate Educational Explanations.](http://arxiv.org/abs/2312.03122) | 本研究提出了一种强化断言的少样本学习技术，用于大型语言模型生成精确、详细的教育解释。实验结果显示，该方法在解释准确性上提升了15%，获得了教师评估为高质量的解释。 |
| [^64] | [Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation.](http://arxiv.org/abs/2311.13184) | 本论文提出了一种方法，通过将算法表示集成到算法选择中，从而填补了当前算法选择技术对算法特征的研究空白。 |
| [^65] | [Debiasing Algorithm through Model Adaptation.](http://arxiv.org/abs/2310.18913) | 本论文提出了一种通过模型适应来检测和减轻语言模型中性别偏见的方法，并证明了该方法能够显著减少偏见同时保持模型性能。 |
| [^66] | [SpecTr: Fast Speculative Decoding via Optimal Transport.](http://arxiv.org/abs/2310.15141) | 本研究通过最优传输的方法提供了推测性解码的原则性理解，使得从大语言模型中进行自回归采样的过程能够更快速地进行。 |
| [^67] | [MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter.](http://arxiv.org/abs/2310.12798) | MolCA是一个可以通过跨模态投影和单模态适配器实现分子图和语言的建模系统。它可以通过连接图编码器和语言模型的表示空间来理解文本和图形的分子内容，并通过单模态适配器在下游任务中高效适应。 |
| [^68] | [Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection.](http://arxiv.org/abs/2310.12086) | 该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。 |
| [^69] | [Functional Invariants to Watermark Large Transformers.](http://arxiv.org/abs/2310.11446) | 本文介绍了一种用于大型Transformer的功能不变性水印技术，它使用模型的不变性生成功能上等效的副本，并能在不改变模型输出的情况下给模型加上水印，这是一种计算成本极低且适用于实际应用的解决方案。 |
| [^70] | [Circuit Component Reuse Across Tasks in Transformer Language Models.](http://arxiv.org/abs/2310.08744) | 这项工作证明了在Transformer语言模型中，电路组件可以在不同任务之间复用并产生相似的功能，为更高级的模型理解做出贡献。 |
| [^71] | [Understanding the Humans Behind Online Misinformation: An Observational Study Through the Lens of the COVID-19 Pandemic.](http://arxiv.org/abs/2310.08483) | 本研究通过观察分析了3200万条COVID-19推文和1600万条历史时间线推文，重点研究了COVID-19期间传播虚假信息用户的行为和心理，并将其与在疫情前在非COVID领域分享虚假信息的历史倾向联系起来。 |
| [^72] | [MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use.](http://arxiv.org/abs/2310.03128) | 本文提出了一个名为MetaTool的基准，旨在评估大型语言模型（LLMs）是否具有工具使用意识并且能够正确选择工具。基准中包含一个名为ToolE的数据集，其中包含各种类型的用户查询，用于触发LLMs使用工具。 |
| [^73] | [Less is More for Long Document Summary Evaluation by LLMs.](http://arxiv.org/abs/2309.07382) | 该论文引入了一种新颖的方法，通过先提取关键句子再进行评估，有效解决了大型语言模型在长文档摘要评估中遇到的计算成本高和忽视重要信息的问题。研究发现，这种方法不仅显著降低了评估成本，而且与人工评估有更高的相关性。此外，论文还提供了关于最佳文档长度和句子提取方法的实用建议，为基于大型语言模型的文本生成评估的发展做出了贡献。 |
| [^74] | [Panoptic Vision-Language Feature Fields.](http://arxiv.org/abs/2309.05448) | 本文提出了一种用于3D场景中开放词汇全景分割的算法PVLFF，通过从预训练的2D模型中提取视觉-语言特征来学习语义特征场，通过对输入帧上的2D实例分割进行对比学习来联合拟合实例特征场。该方法在全景分割和语义分割方面具有良好的性能。 |
| [^75] | [Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models.](http://arxiv.org/abs/2308.10462) | 本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。 |
| [^76] | [Separate the Wheat from the Chaff: Model Deficiency Unlearning via Parameter-Efficient Module Operation.](http://arxiv.org/abs/2308.08090) | 通过提取和消除反专家PEMs中的残缺能力来提升大规模语言模型的真实性和去毒性。 |
| [^77] | [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition.](http://arxiv.org/abs/2307.13269) | 本文研究了LoRA组合在跨任务通用性上的可行性，并提出了LoraHub框架，能够通过组合不同任务上训练的LoRA模块，实现对未见任务的可适应性性能。实验结果表明，LoraHub在少样本场景中能够有效模拟上下文学习的性能，而无需上下文示例。 |
| [^78] | [CMMLU: Measuring massive multitask language understanding in Chinese.](http://arxiv.org/abs/2306.09212) | 本论文介绍了CMMLU，这是一个衡量中文大规模多任务语言理解的综合性基准测试。研究发现，大多数现有的语言模型在不同主题和设置下的性能都不够理想，存在改进空间。CMMLU填补了评估大型语言模型知识和推理能力的空白，并提出了增强LLM的方向。 |
| [^79] | [Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data.](http://arxiv.org/abs/2306.05535) | 政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。 |
| [^80] | [Flexible Grammar-Based Constrained Decoding for Language Models.](http://arxiv.org/abs/2305.13971) | 本文提出了一种使用形式语法约束丰富解码步骤的方法，有效生成符合特定语法的复杂输出结构，同时允许任何上下文无关语法集成。实验证明该方法在四个信息提取任务上实现了最先进的性能表现。 |
| [^81] | [Conversational Process Modelling: State of the Art, Applications, and Implications in Practice.](http://arxiv.org/abs/2304.11065) | 本文系统的研究了现有聊天机器人对于支持对话式流程建模所提供的应用场景，并推导出了在实践中使用聊天机器人进行对话式流程建模的建议。 |
| [^82] | [CodeKGC: Code Language Model for Generative Knowledge Graph Construction.](http://arxiv.org/abs/2304.09048) | 本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法，能够有效利用知识图谱内的语义结构，提高模型的可解释性。 |
| [^83] | [ESD: Expected Squared Difference as a Tuning-Free Trainable Calibration Measure.](http://arxiv.org/abs/2303.02472) | ESD是一种无需调参的可训练校准目标损失，通过将校准误差看作两个期望值之间的平方差，可以改善神经网络模型的校准度。 |
| [^84] | [Language Control Diffusion: Efficiently Scaling through Space, Time, and Tasks.](http://arxiv.org/abs/2210.15629) | 本文提出一种利用语言控制扩散模型的分层规划器，有效而高效地扩展扩散模型，解决长时间跨度自然语言指令下的控制问题，实现了较高的单任务和多任务成功率，并极大地提高计算效率。 |

# 详细

[^1]: ChatQA: 构建GPT-4级对话问答模型

    ChatQA: Building GPT-4 Level Conversational QA Models. (arXiv:2401.10225v1 [cs.CL])

    [http://arxiv.org/abs/2401.10225](http://arxiv.org/abs/2401.10225)

    ChatQA是一系列对话问答模型，可以达到GPT-4级别的准确性。通过两阶段的指令调整方法，可以显著提高大型语言模型在零-shot对话问答中的结果。使用密集检索器进行问答数据集的微调可以实现与最先进的查询重写模型相当的结果，同时降低部署成本。ChatQA-70B在10个对话问答数据集上的平均得分超过了GPT-4，且不依赖于任何来自OpenAI GPT模型的合成数据。

    

    在这项工作中，我们介绍了ChatQA，一系列具有GPT-4级别准确性的对话问答模型。具体地，我们提出了一个两阶段的指令调整方法，可以显著提高大型语言模型（LLM）在零-shot对话问答中的结果。为了处理对话问答中的检索问题，我们在多轮问答数据集上进行了密集检索器的微调，这样可以提供与使用最先进的查询重写模型相当的结果，同时大大降低部署成本。值得注意的是，我们的ChatQA-70B可以在10个对话问答数据集的平均分上超过GPT-4（54.14 vs. 53.90），而不依赖于OpenAI GPT模型的任何合成数据。

    In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.
    
[^2]: MM-交错的：通过多模式特征同步器进行交错图像-文本生成建模

    MM-Interleaved: Interleaved Image-Text Generative Modeling via Multi-modal Feature Synchronizer. (arXiv:2401.10208v1 [cs.CV])

    [http://arxiv.org/abs/2401.10208](http://arxiv.org/abs/2401.10208)

    本文提出了MM-交错，这是一个用于交错图像-文本数据的生成模型。它通过引入多尺度和多图像特征同步器模块，解决了现有模型在捕捉图像细节方面的限制，并通过端到端预训练和监督微调相结合的方式提高了其生成能力。

    

    对于交错图像-文本数据的生成模型的开发具有研究和实际价值。它要求模型理解交错的序列，并随后生成图像和文本。然而，现有的尝试受到了固定数量的视觉标记不能有效捕捉图像细节的问题的限制，在多图像场景中，这一问题尤为严重。为了解决这个问题，本文提出了MM-交错，这是一个用于交错图像-文本数据的端到端生成模型。它引入了一个多尺度和多图像特征同步器模块，允许在生成过程中直接访问先前上下文中的细粒度图像特征。MM-交错在配对和交错图像-文本语料库上进行端到端预训练。它还通过一阶段的监督微调来进一步改善其遵循复杂多模态指令的能力。实验结果表明了MM-交错在图像修复、图像生成和文本描述生成等任务中的多功能性。

    Developing generative models for interleaved image-text data has both research and practical value. It requires models to understand the interleaved sequences and subsequently generate images and text. However, existing attempts are limited by the issue that the fixed number of visual tokens cannot efficiently capture image details, which is particularly problematic in the multi-image scenarios. To address this, this paper presents MM-Interleaved, an end-to-end generative model for interleaved image-text data. It introduces a multi-scale and multi-image feature synchronizer module, allowing direct access to fine-grained image features in the previous context during the generation process. MM-Interleaved is end-to-end pre-trained on both paired and interleaved image-text corpora. It is further enhanced through a supervised fine-tuning phase, wherein the model improves its ability to follow complex multi-modal instructions. Experiments demonstrate the versatility of MM-Interleaved in rec
    
[^3]: Chem-FINESE: 通过文本重构验证细粒度少样本实体提取

    Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction. (arXiv:2401.10189v1 [cs.CL])

    [http://arxiv.org/abs/2401.10189](http://arxiv.org/abs/2401.10189)

    这篇论文提出了一种名为Chem-FINESE的方法来处理化学领域中细粒度少样本实体提取的问题。该方法通过使用序列到序列的实体提取器和自我验证模块来从输入句子中提取命名实体并重构原始输入句子。实验证明了该方法的有效性和可行性。

    

    在化学领域中，细粒度少样本实体提取面临两个独特的挑战。首先，与一般领域的实体提取任务相比，化学论文中的句子通常包含更多的实体。此外，实体提取模型通常难以提取长尾类型的实体。在本文中，我们提出了一种新颖的基于序列到序列的少样本实体提取方法Chem-FINESE来解决这两个挑战。我们的Chem-FINESE包含两个组件：一个序列到序列的实体提取器用于从输入句子中提取命名实体，以及一个序列到序列的自我验证模块用于从提取的实体中重构原始输入句子。受到一个好的实体提取系统需要忠实提取实体的事实启发，我们的新自我验证模块利用实体提取结果来重构原始输入句子。此外，我们设计了一种新的对比损失来减少在提取过程中的过度复制。

    Fine-grained few-shot entity extraction in the chemical domain faces two unique challenges. First, compared with entity extraction tasks in the general domain, sentences from chemical papers usually contain more entities. Moreover, entity extraction models usually have difficulty extracting entities of long-tailed types. In this paper, we propose Chem-FINESE, a novel sequence-to-sequence (seq2seq) based few-shot entity extraction approach, to address these two challenges. Our Chem-FINESE has two components: a seq2seq entity extractor to extract named entities from the input sentence and a seq2seq self-validation module to reconstruct the original input sentence from extracted entities. Inspired by the fact that a good entity extraction system needs to extract entities faithfully, our new self-validation module leverages entity extraction results to reconstruct the original input sentence. Besides, we design a new contrastive loss to reduce excessive copying during the extraction proces
    
[^4]: 超越基于参考指标：分析开放式LLM在数据到文本生成上的行为

    Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation. (arXiv:2401.10186v1 [cs.CL])

    [http://arxiv.org/abs/2401.10186](http://arxiv.org/abs/2401.10186)

    开放式大型语言模型在零-shot设置下能够从各种标准数据格式中生成流畅和连贯的文本，但是输出的语义准确性仍然是一个重要问题。

    

    我们调查开放式大型语言模型(LLMs)在从结构化数据生成连贯和相关文本方面的程度。为了防止基准泄露到LLM训练数据中的偏差，我们收集了Quintd-1:一个为5个数据到文本(D2T)生成任务设计的专门基准，该任务包括从公共API中收集的标准格式的结构化数据记录。我们利用无参考评估指标和LLMs的上下文学习能力，使我们能够在没有人工写作参考资料的情况下测试模型。我们的评估重点是在token级别上对语义准确性错误进行注释，结合人类标注者和基于GPT-4的指标。我们系统地研究了模型在不同领域和任务中的行为，发现7B参数的最先进开放式LLMs可以在零-shot设置中从各种标准数据格式中生成流畅和连贯的文本。然而，我们也表明输出的语义准确性仍然是一个重大问题：在我们的基准上，80%的输出存在语义准确性错误。

    We investigate to which extent open large language models (LLMs) can generate coherent and relevant text from structured data. To prevent bias from benchmarks leaked into LLM training data, we collect Quintd-1: an ad-hoc benchmark for five data-to-text (D2T) generation tasks, consisting of structured data records in standard formats gathered from public APIs. We leverage reference-free evaluation metrics and LLMs' in-context learning capabilities, allowing us to test the models with no human-written references. Our evaluation focuses on annotating semantic accuracy errors on token-level, combining human annotators and a metric based on GPT-4. Our systematic examination of the models' behavior across domains and tasks suggests that state-of-the-art open LLMs with 7B parameters can generate fluent and coherent text from various standard data formats in zero-shot settings. However, we also show that semantic accuracy of the outputs remains a major issue: on our benchmark, 80% of outputs o
    
[^5]: 空间-时间大语言模型用于交通预测

    Spatial-Temporal Large Language Model for Traffic Prediction. (arXiv:2401.10134v1 [cs.LG])

    [http://arxiv.org/abs/2401.10134](http://arxiv.org/abs/2401.10134)

    本文提出了一种空间-时间大语言模型（ST-LLM）用于交通预测，通过参数扩展和预训练来提高预测准确性，并利用空间-时间嵌入模块学习标记的空间位置和全局时间表示。

    

    交通预测是智能交通系统的关键组成部分，它通过使用历史数据来预测特定位置的未来交通情况。尽管现有的交通预测模型通常强调开发复杂的神经网络结构，但它们的准确性并未相应提高。最近，大型语言模型（LLMs）在时间序列分析方面显示出了出色的能力。与现有模型不同，LLMs主要通过参数扩展和广泛的预训练来进步，同时保持其基本结构。本文提出了一种空间-时间大语言模型（ST-LLM）用于交通预测。具体而言，ST-LLM将每个位置的时间步长定义为标记，并结合空间-时间嵌入模块来学习标记的空间位置和全局时间表示。然后，这些表示被融合以为每个标记提供统一的空间和时间信息。

    Traffic prediction, a critical component for intelligent transportation systems, endeavors to foresee future traffic at specific locations using historical data. Although existing traffic prediction models often emphasize developing complex neural network structures, their accuracy has not seen improvements accordingly. Recently, Large Language Models (LLMs) have shown outstanding capabilities in time series analysis. Differing from existing models, LLMs progress mainly through parameter expansion and extensive pre-training while maintaining their fundamental structures. In this paper, we propose a Spatial-Temporal Large Language Model (ST-LLM) for traffic prediction. Specifically, ST-LLM redefines the timesteps at each location as tokens and incorporates a spatial-temporal embedding module to learn the spatial location and global temporal representations of tokens. Then these representations are fused to provide each token with unified spatial and temporal information. Furthermore, we
    
[^6]: 将适配器和Mixup相结合，以增强预训练语言模型在文本分类中的对抗鲁棒性

    Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification. (arXiv:2401.10111v1 [cs.CL])

    [http://arxiv.org/abs/2401.10111](http://arxiv.org/abs/2401.10111)

    本研究将适配器和Mixup相结合，以在不需要频繁重新训练整个模型的情况下增强预训练语言模型的对抗鲁棒性，通过使用适配器的凸组合和非数据对的训练，解决了对抗训练方法在性能下降和计算开销方面的限制。

    

    现有的研究表明，使用干净和对抗性样本来增强神经网络的训练数据可以提高其在对抗攻击下的泛化能力。然而，这种训练方法往往会导致对清洁输入的性能下降。另外，它需要频繁地重新训练整个模型，以适应新的攻击类型，从而导致昂贵且计算量大的计算。这些限制使得对抗训练机制的实际应用变得不太实用，特别是对于具有数百万甚至数十亿参数的复杂预训练语言模型（PLMs）。为了克服这些挑战，同时利用对抗训练的理论益处，本研究将两个概念相结合：（1）适配器，可实现参数高效微调，和（2）Mixup，通过数据对的凸组合训练NNs。直观地说，我们建议通过非数据对的适配器的凸组合来微调PLMs，其中一个适配器使用干净的样本训练，另一个使用对抗性的样本训练。

    Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and an
    
[^7]: 数字的力量：通过每个示例细调与四个对抗句子来实现鲁棒的阅读理解

    Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example. (arXiv:2401.10091v1 [cs.CL])

    [http://arxiv.org/abs/2401.10091](http://arxiv.org/abs/2401.10091)

    本文研究通过每个示例细调与四个对抗句子来实现鲁棒的阅读理解，并发现这种细调能提高模型在评估数据集上的F1分数。

    

    在评估阅读理解任务时，最近的模型已经达到了人类水平的性能，使用F1分数进行评估。然而，在一般情况下，教机器理解文本还没有解决。通过将一个对抗性句子添加到上下文段落中，过去的研究表明，阅读理解模型的F1分数几乎减半。在本文中，我使用一个新的模型ELECTRA-Small复制了过去的对抗性研究，并证明了新模型的F1分数从83.9％下降到29.2％。为了提高ELECTRA-Small对这种攻击的抵抗力，我对SQuAD v1.1训练示例进行了细调，将一到五个对抗句子附加到上下文段落中。与过去的研究一样，我发现细调后的模型对一个对抗句子的泛化能力不佳。然而，当细调用于四个或五个对抗句子时，该模型在大多数评估数据集上获得超过70％的F1分数。

    Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evalu
    
[^8]: 通信高效的个性化联邦学习在语音转文本任务中的应用

    Communication-Efficient Personalized Federated Learning for Speech-to-Text Tasks. (arXiv:2401.10070v1 [cs.CL])

    [http://arxiv.org/abs/2401.10070](http://arxiv.org/abs/2401.10070)

    该论文提出了一种通信高效的个性化联邦学习框架，通过引入轻量级的LoRA模块进行客户端调整和与服务器的交互，以最小化通信开销，以及使用K最近邻分类器的全局模型来实现个性化并克服数据异构问题。

    

    为了保护隐私并满足法规要求，联邦学习在训练语音转文本系统（包括自动语音识别和语音翻译）方面引起了广泛关注。然而，在语音转文本任务中常用的联邦学习方法（即FedAvg）通常面临着大量的通信开销和数据异构导致的性能下降问题。为了解决这些问题，我们提出了一种个性化的联邦语音转文本框架，引入了轻量级的LoRA模块（FedLoRA）用于客户端调整和与服务器进行交互以最小化通信开销，以及全局模型（FedMem）配备了K最近邻分类器，以捕捉客户特定的分布变化以实现个性化并克服数据异构。在CoVoST和GigaSp数据集上基于Conformer和Whisper主干模型进行了大量实验。

    To protect privacy and meet legal regulations, federated learning (FL) has gained significant attention for training speech-to-text (S2T) systems, including automatic speech recognition (ASR) and speech translation (ST). However, the commonly used FL approach (i.e., \textsc{FedAvg}) in S2T tasks typically suffers from extensive communication overhead due to multi-round interactions based on the whole model and performance degradation caused by data heterogeneity among clients.To address these issues, we propose a personalized federated S2T framework that introduces \textsc{FedLoRA}, a lightweight LoRA module for client-side tuning and interaction with the server to minimize communication overhead, and \textsc{FedMem}, a global model equipped with a $k$-nearest-neighbor ($k$NN) classifier that captures client-specific distributional shifts to achieve personalization and overcome data heterogeneity. Extensive experiments based on Conformer and Whisper backbone models on CoVoST and GigaSp
    
[^9]: 代码提示在文本+代码LLMs中引发了条件推理能力

    Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs. (arXiv:2401.10065v1 [cs.CL])

    [http://arxiv.org/abs/2401.10065](http://arxiv.org/abs/2401.10065)

    本论文研究了在大型语言模型（LLMs）中触发条件推理能力的方法，通过使用代码提示将自然语言问题转化为代码，从而在多个数据集上实现了显著的性能提升。

    

    推理是实现语言理解的基本组成部分。在多种推理类型中，条件推理是一种在某些条件下得出不同结论的能力，在大型语言模型（LLMs）中一直没有得到充分研究。最近的提示方法，如思维链，显著改进了在推理任务上的LLMs性能。然而，我们对于什么触发了LLMs中的推理能力仍然知之甚少。我们假设代码提示能够触发在文本和代码上训练的LLMs中的条件推理。我们提出了一系列的提示，将自然语言问题转化为代码，并用生成的代码提示LLMs。我们的实验发现，在需要条件推理的多个数据集上，代码提示使得GPT 3.5的性能提升了2.6到7.7个百分点。接着，我们进行了实验，探索了代码提示如何引发条件推理能力以及通过哪些特征进行。我们观察到，提示的形式和内容对于引发条件推理能力起到了重要作用。

    Reasoning is a fundamental component for achieving language understanding. Among the multiple types of reasoning, conditional reasoning, the ability to draw different conclusions depending on some condition, has been understudied in large language models (LLMs). Recent prompting methods, such as chain of thought, have significantly improved LLMs on reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs. We hypothesize that code prompts can trigger conditional reasoning in LLMs trained on text and code. We propose a chain of prompts that transforms a natural language problem into code and prompts the LLM with the generated code. Our experiments find that code prompts exhibit a performance boost between 2.6 and 7.7 points on GPT 3.5 across multiple datasets requiring conditional reasoning. We then conduct experiments to discover how code prompts elicit conditional reasoning abilities and through which features. We observe that prom
    
[^10]: 使用交织编码器网络（ICE-NET）区分反义词和同义词的挑战

    Antonym vs Synonym Distinction using InterlaCed Encoder NETworks (ICE-NET). (arXiv:2401.10045v1 [cs.CL])

    [http://arxiv.org/abs/2401.10045](http://arxiv.org/abs/2401.10045)

    本文提出了一种名为ICE-NET的交织编码器网络，用于区分反义词和同义词。通过捕捉和模拟它们的关系特定属性，ICE-NET在性能上优于现有研究，提高了1.8%的F1得分。

    

    反义词和同义词的区分是词汇-语义分析和自动词汇资源构建中的核心挑战。这些词对共享相似的分布上下文，这使得区分它们更加困难。有关这方面的主要研究试图捕捉关系对的性质，即对称性、传递性和传递-传递性。然而，现有研究无法适当地模拟关系特定属性，限制了它们的最终性能。在本文中，我们提出了用于反义词和同义词区分的交织编码器网络（即ICE-NET），旨在捕捉和模拟反义词和同义词对的关系特定属性，以便以性能增强的方式执行分类任务。对基准数据集的实验评估表明，ICE-NET在F1得分上相对于现有研究提高了最高1.8%。我们在https://github.com/asif6827/ICENET上发布了ICE-NET的代码。

    Antonyms vs synonyms distinction is a core challenge in lexico-semantic analysis and automated lexical resource construction. These pairs share a similar distributional context which makes it harder to distinguish them. Leading research in this regard attempts to capture the properties of the relation pairs, i.e., symmetry, transitivity, and trans-transitivity. However, the inability of existing research to appropriately model the relation-specific properties limits their end performance. In this paper, we propose InterlaCed Encoder NETworks (i.e., ICE-NET) for antonym vs synonym distinction, that aim to capture and model the relation-specific properties of the antonyms and synonyms pairs in order to perform the classification task in a performance-enhanced manner. Experimental evaluation using the benchmark datasets shows that ICE-NET outperforms the existing research by a relative score of upto 1.8% in F1-measure. We release the codes for ICE-NET at https://github.com/asif6827/ICENET
    
[^11]: 大型语言模型在科学信息提取中的应用：一项针对病毒学的实证研究

    Large Language Models for Scientific Information Extraction: An Empirical Study for Virology. (arXiv:2401.10040v1 [cs.CL])

    [http://arxiv.org/abs/2401.10040](http://arxiv.org/abs/2401.10040)

    使用大型语言模型进行结构化的科学信息提取，在病毒学领域进行了实证研究，结果表明这种方法可以提供简洁的学术贡献摘要，对科学家进行导航和解决LLM的紧迫能力。

    

    本文倡导使用结构化和语义内容表示来进行基于学术交流的学术论文，受到维基百科信息框或结构化的亚马逊产品描述等工具的启发。这些表示形式提供了简洁的概述，帮助科学家在浓厚的学术环境中进行导航。我们的新颖自动化方法利用LLM的强大文本生成能力，产生结构化的学术贡献摘要，既提供了实际解决方案，也揭示了LLM紧迫的能力。对于LLM，主要关注的是改善其作为对话代理的通用智能。我们认为这些模型也可以在信息提取（IE）中有效应用，特别是在科学等领域的复杂IE任务中。这种范式转变用一系列指令代替了传统的模块化、流水线式的机器学习方法，简化了目标。我们的结果表明，通过微调的FLAN-T模型可以取得良好效果。

    In this paper, we champion the use of structured and semantic content representation of discourse-based scholarly communication, inspired by tools like Wikipedia infoboxes or structured Amazon product descriptions. These representations provide users with a concise overview, aiding scientists in navigating the dense academic landscape. Our novel automated approach leverages the robust text generation capabilities of LLMs to produce structured scholarly contribution summaries, offering both a practical solution and insights into LLMs' emergent abilities.  For LLMs, the prime focus is on improving their general intelligence as conversational agents. We argue that these models can also be applied effectively in information extraction (IE), specifically in complex IE tasks within terse domains like Science. This paradigm shift replaces the traditional modular, pipelined machine learning approach with a simpler objective expressed through instructions. Our results show that finetuned FLAN-T
    
[^12]: 大语言模型时代的进化计算：调查与路线图

    Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap. (arXiv:2401.10034v1 [cs.NE])

    [http://arxiv.org/abs/2401.10034](http://arxiv.org/abs/2401.10034)

    该论文调查了大语言模型和进化计算之间的相互作用，并提出了在黑盒设置下进一步提升大语言模型性能的优化框架，以及将大语言模型与进化算法结合应用于各种任务的方法。

    

    大型语言模型（LLMs）是基于Transformer架构，在多样的数据上进行大规模预训练的，它们不仅在自然语言处理领域引起了革命，还将其能力扩展到了各个领域，迈向了人工通用智能的重要一步。尽管进化算法（EAs）与LLMs在目标和方法论上存在差异，但它们之间的相互作用揭示了有趣的相似之处，特别是在他们共同的优化性质、黑盒特性和处理复杂问题的能力方面。与此同时，进化算法不仅可以为LLM在黑盒设置下提供优化框架，还可以在应用中为LLM赋予灵活的全局搜索和迭代机制。另一方面，LLM丰富的领域知识使得进化算法可以进行更智能的搜索，而其文本处理能力则有助于将进化算法应用于各种任务。基于它们的互补优势，本文提出了一份调查和路线图。

    Large Language Models (LLMs), built upon Transformer-based architectures with massive pretraining on diverse data, have not only revolutionized natural language processing but also extended their prowess to various domains, marking a significant stride towards artificial general intelligence. The interplay between LLMs and Evolutionary Algorithms (EAs), despite differing in objectives and methodologies, reveals intriguing parallels, especially in their shared optimization nature, black-box characteristics, and proficiency in handling complex problems. Meanwhile, EA can not only provide an optimization framework for LLM's further enhancement under black-box settings but also empower LLM with flexible global search and iterative mechanism in applications. On the other hand, LLM's abundant domain knowledge enables EA to perform smarter searches, while its text processing capability assist in deploying EA across various tasks. Based on their complementary advantages, this paper presents a 
    
[^13]: 基于健康相关叙事的框架分析：阴谋与主流媒体的比较

    Framing Analysis of Health-Related Narratives: Conspiracy versus Mainstream Media. (arXiv:2401.10030v1 [cs.CL])

    [http://arxiv.org/abs/2401.10030](http://arxiv.org/abs/2401.10030)

    本研究基于健康相关叙事的框架分析，比较了阴谋媒体与主流媒体之间在COVID-19等话题上的框架差异。研究发现，在阴谋媒体中，健康相关的叙事主要以信仰为框架，而主流媒体则以科学为框架。这项研究为更深入的框架分析提供了新的方法。

    

    了解在线媒体如何框定问题对于其对公众舆论的影响至关重要。利用自然语言处理技术进行的框架研究主要关注消息中的特定内容特征，而忽视了其叙事要素。此外，不同信息源中的框架差异是一个未被充分研究的问题。我们解决了这些问题，并研究了与健康相关的话题（如COVID-19和其他疾病）在阴谋和主流媒体之间的框架差异。我们通过引入基于语义图的新型框架提取方法将叙事信息纳入框架分析中。我们发现，在阴谋媒体中，与健康相关的叙事主要以信仰为框架，而主流媒体倾向于以科学为框架呈现。我们希望我们的工作能为更细致入微的框架分析提供新的方法。

    Understanding how online media frame issues is crucial due to their impact on public opinion. Research on framing using natural language processing techniques mainly focuses on specific content features in messages and neglects their narrative elements. Also, the distinction between framing in different sources remains an understudied problem. We address those issues and investigate how the framing of health-related topics, such as COVID-19 and other diseases, differs between conspiracy and mainstream websites. We incorporate narrative information into the framing analysis by introducing a novel frame extraction approach based on semantic graphs. We find that health-related narratives in conspiracy media are predominantly framed in terms of beliefs, while mainstream media tend to present them in terms of science. We hope our work offers new ways for a more nuanced frame analysis.
    
[^14]: 自奖励语言模型

    Self-Rewarding Language Models. (arXiv:2401.10020v1 [cs.CL])

    [http://arxiv.org/abs/2401.10020](http://arxiv.org/abs/2401.10020)

    该论文提出了自奖励语言模型的概念，通过LLM作为评判者，使用语言模型自己提供训练过程中的奖励。研究表明，该方法不仅可以提高指令遵循能力，还可以为自己提供高质量的奖励。通过对Llama 2 70B模型的三次迭代微调，结果在AlpacaEval 2.0排行榜上超过了其他现有系统。这项工作为实现能够不断自我改进的模型开辟了新的可能性。

    

    我们假设要实现超人级的智能体，未来的模型需要超人级的反馈，以提供足够的训练信号。目前的方法通常是从人类偏好中训练奖励模型，这可能会受到人类表现水平的限制，而且这些独立的冻结奖励模型在LLM训练过程中无法学习改进。在这项工作中，我们研究了自奖励语言模型，其中语言模型本身通过LLM作为评判者的提示在训练过程中提供自己的奖励。我们表明，在迭代DPO训练中，不仅指令遵循能力得到了提高，而且能够为自己提供高质量的奖励。通过对Llama 2 70B进行我们方法的三次迭代的微调，得到的模型在AlpacaEval 2.0排行榜上胜过许多现有系统，包括Claude 2、Gemini Pro和GPT-4 0613。虽然这只是一项初步研究，但这项工作为可能实现能够不断自我改进的模型打开了大门。

    We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continuall
    
[^15]: R-Judge: 评估LLM代理的安全风险意识的基准测试

    R-Judge: Benchmarking Safety Risk Awareness for LLM Agents. (arXiv:2401.10019v1 [cs.CL])

    [http://arxiv.org/abs/2401.10019](http://arxiv.org/abs/2401.10019)

    这篇论文主要介绍了一种评估LLM代理在不同环境中判断安全风险能力的基准测试R-Judge，通过对162个代理交互记录进行评估，发现GPT-4模型表现最佳，达到了72.29%的准确率。

    

    大型语言模型（LLM）在自动完成各种真实世界应用任务方面展现出巨大潜力。然而，这些LLM代理在交互环境中操作时会引入意外的安全风险。与大多数之前的研究集中在LLM生成内容的安全性不同，本研究关注评估LLM代理在不同环境中的行为安全性的迫切需求。我们介绍了一个名为R-Judge的基准测试，用于评估LLM在给定代理交互记录时判断安全风险的能力。R-Judge包括162个代理交互记录，涵盖7个应用领域和10种风险类型的27个关键风险场景。它结合了人类对安全性的共识，并具有标记的安全风险标签和高质量的风险描述。利用R-Judge，我们对8种常用作代理骨干的著名LLM模型进行了全面评估。表现最好的模型GPT-4实现了72.29%的对比结果。

    Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on LLM-generated content safety in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging safety risks given agent interaction records. R-Judge comprises 162 agent interaction records, encompassing 27 key risk scenarios among 7 application categories and 10 risk types. It incorporates human consensus on safety with annotated safety risk labels and high-quality risk descriptions. Utilizing R-Judge, we conduct a comprehensive evaluation of 8 prominent LLMs commonly employed as the backbone for agents. The best-performing model, GPT-4, achieves 72.29% in contrast to
    
[^16]: 机器翻译中的性别偏见与大语言模型时代

    Gender Bias in Machine Translation and The Era of Large Language Models. (arXiv:2401.10016v1 [cs.CL])

    [http://arxiv.org/abs/2401.10016](http://arxiv.org/abs/2401.10016)

    本章研究了机器翻译中的性别偏见问题，介绍了传统神经机器翻译方法和生成预训练转换器模型中相关工作。通过在英语-意大利语的翻译环境中使用ChatGPT进行实验，评估了其解决性别偏见的能力。研究结果强调了减少机器翻译系统偏见的重要性。

    

    本章探讨了机器翻译在延续性别偏见方面的作用，着重强调了跨语言环境和统计依赖性所带来的挑战。提供了关于性别偏见在传统神经机器翻译方法和作为机器翻译系统的生成预训练转换器模型中的相关现有工作的全面概述。通过在英语-意大利语翻译环境中使用ChatGPT (基于GPT-3.5)进行实验，我们进一步评估了ChatGPT解决性别偏见的当前能力。研究结果强调了在机器翻译系统中减少偏见的持续需求，并强调了促进语言技术的公平性和包容性的重要性。

    This chapter examines the role of Machine Translation in perpetuating gender bias, highlighting the challenges posed by cross-linguistic settings and statistical dependencies. A comprehensive overview of relevant existing work related to gender bias in both conventional Neural Machine Translation approaches and Generative Pretrained Transformer models employed as Machine Translation systems is provided. Through an experiment using ChatGPT (based on GPT-3.5) in an English-Italian translation context, we further assess ChatGPT's current capacity to address gender bias. The findings emphasize the ongoing need for advancements in mitigating bias in Machine Translation systems and underscore the importance of fostering fairness and inclusivity in language technologies.
    
[^17]: 向层次化口语淤塞建模迈进

    Towards Hierarchical Spoken Language Dysfluency Modeling. (arXiv:2401.10015v1 [cs.CL])

    [http://arxiv.org/abs/2401.10015](http://arxiv.org/abs/2401.10015)

    本论文介绍了一种名为H-UDM的层次化口语淤塞建模方法，它能够解决口语淤塞转录和检测问题，并且消除了对大量手动注释的需求。实验结果证明了该方法在转录和检测任务中的有效性和鲁棒性。

    

    口语淤塞建模是语言治疗和语言学习的瓶颈。然而，目前没有人工智能解决方案来系统地应对这个问题。我们首先提出了定义口语淤塞和口语淤塞建模的概念。然后，我们提出了一种名为Hierarchical Unconstrained Dysfluency Modeling (H-UDM)的方法，既解决了口语淤塞转录问题，又解决了检测问题，消除了对大量手动注释的需求。此外，我们还引入了一个名为VCTK++的模拟淤塞数据集，以增强H-UDM在音标转录方面的能力。我们的实验结果证明了我们提出的方法在转录和检测任务中的有效性和鲁棒性。

    Speech dysfluency modeling is the bottleneck for both speech therapy and language learning. However, there is no AI solution to systematically tackle this problem. We first propose to define the concept of dysfluent speech and dysfluent speech modeling. We then present Hierarchical Unconstrained Dysfluency Modeling (H-UDM) approach that addresses both dysfluency transcription and detection to eliminate the need for extensive manual annotation. Furthermore, we introduce a simulated dysfluent dataset called VCTK++ to enhance the capabilities of H-UDM in phonetic transcription. Our experimental results demonstrate the effectiveness and robustness of our proposed methods in both transcription and detection tasks.
    
[^18]: 以显性推理链和视觉问题生成推进大型多模态模型

    Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation. (arXiv:2401.10005v1 [cs.CV])

    [http://arxiv.org/abs/2401.10005](http://arxiv.org/abs/2401.10005)

    本文提出了一种新的方法，通过显性推理和问题生成，将大型多模态模型(LMM)赋予了显性推理能力，从而提高了推理过程的鲁棒性和可解释性。

    

    随着对能够解释和推理视觉内容的智能系统需求越来越高，需要开发不仅准确而且具有显性推理能力的大型多模态模型（LMMs）。本文提出了一种新颖的方法，将显性推理能力赋予LMMs，基于视觉内容和文本指导进行显性推理。我们引入了一个系统，可以提问以获取必要的知识，从而增强推理过程的鲁棒性和可解释性。我们的方法包括通过一个大型语言模型（LLM）生成的新颖数据集的开发，旨在促进思维链推理与提问机制的结合。我们设计了一个高度具有区域意识的LMM，以解决图像-文本对齐的复杂需求。该模型经历了三个阶段的训练，首先是使用大规模数据集进行大规模图像-文本对齐，接下来是通过显式推理的问题生成阶段。

    The increasing demand for intelligent systems capable of interpreting and reasoning about visual content requires the development of Large Multi-Modal Models (LMMs) that are not only accurate but also have explicit reasoning capabilities. This paper presents a novel approach to imbue an LMM with the ability to conduct explicit reasoning based on visual content and textual instructions. We introduce a system that can ask a question to acquire necessary knowledge, thereby enhancing the robustness and explicability of the reasoning process. Our method comprises the development of a novel dataset generated by a Large Language Model (LLM), designed to promote chain-of-thought reasoning combined with a question-asking mechanism. We designed an LMM, which has high capabilities on region awareness to address the intricate requirements of image-text alignment. The model undergoes a three-stage training phase, starting with large-scale image-text alignment using a large-scale datasets, followed 
    
[^19]: 远程监督的形态句法模型用于关系抽取

    Distantly Supervised Morpho-Syntactic Model for Relation Extraction. (arXiv:2401.10002v1 [cs.CL])

    [http://arxiv.org/abs/2401.10002](http://arxiv.org/abs/2401.10002)

    本文提出了一种远程监督的形态句法模型，用于从文本中提取和分类一组不受限制的关系。该方法通过使用形态句法提取模式和创建句法和语义索引来实现。在基于Wikidata和Wikipedia构建的数据集上的评估结果显示，该方法可以实现高达0.85的精确度得分。这一方法允许快速构建基于规则的信息抽取系统，并构建带注释的数据集用于训练基于机器学习和深度学习的分类器。

    

    信息抽取（IE）的任务涉及将非结构化的文本内容自动转换为结构化的数据。该领域的大部分研究集中在从文档中提取所有事实或特定一组关系。在本文中，我们提出了一种从文本中提取和分类一组不受限制的关系的方法。我们的方法依赖于通过远程监督方法获得的形态句法提取模式，并创建句法和语义索引来提取和分类候选图。我们在基于Wikidata和Wikipedia构建的六个数据集上评估了我们的方法。评估结果显示我们的方法可以实现高达0.85的精确度得分，但召回率和F1得分较低。我们的方法允许快速构建基于规则的信息抽取系统，并构建带注释的数据集用于训练基于机器学习和深度学习的分类器。

    The task of Information Extraction (IE) involves automatically converting unstructured textual content into structured data. Most research in this field concentrates on extracting all facts or a specific set of relationships from documents. In this paper, we present a method for the extraction and categorisation of an unrestricted set of relationships from text. Our method relies on morpho-syntactic extraction patterns obtained by a distant supervision method, and creates Syntactic and Semantic Indices to extract and classify candidate graphs. We evaluate our approach on six datasets built on Wikidata and Wikipedia. The evaluation shows that our approach can achieve Precision scores of up to 0.85, but with lower Recall and F1 scores. Our approach allows to quickly create rule-based systems for Information Extraction and to build annotated datasets to train machine-learning and deep-learning based classifiers.
    
[^20]: 可分级ChatGPT翻译评价

    Gradable ChatGPT Translation Evaluation. (arXiv:2401.09984v1 [cs.CL])

    [http://arxiv.org/abs/2401.09984](http://arxiv.org/abs/2401.09984)

    本文提出了一种通用分类系统，用于定义可分级的翻译提示，以帮助构建适用于不同翻译任务的具有不同特性的提示。验证和说明了该方法的有效性。

    

    ChatGPT作为一种基于大规模预训练的语言模型，在机器翻译领域产生了深远影响。在ChatGPT中，“提示”是指用于引导模型生成特定类型回应的文本段落或指导。翻译提示的设计成为影响翻译风格、准确性和精确度等因素的关键方面。然而，目前缺乏一个共同的标准和方法来设计和选择翻译提示。因此，本文提出了一种通用分类系统，以表达类型、翻译风格、POS信息和显式声明的方式定义可分级的翻译提示，从而为不同的翻译任务构建具有不同特性的提示。选择了具体的实验和案例来验证和说明该方法的有效性。

    ChatGPT, as a language model based on large-scale pre-training, has exerted a profound influence on the domain of machine translation. In ChatGPT, a "Prompt" refers to a segment of text or instruction employed to steer the model towards generating a specific category of response. The design of the translation prompt emerges as a key aspect that can wield influence over factors such as the style, precision and accuracy of the translation to a certain extent. However, there is a lack of a common standard and methodology on how to design and select a translation prompt. Accordingly, this paper proposes a generic taxonomy, which defines gradable translation prompts in terms of expression type, translation style, POS information and explicit statement, thus facilitating the construction of prompts endowed with distinct attributes tailored for various translation tasks. Specific experiments and cases are selected to validate and illustrate the effectiveness of the method.
    
[^21]: 通过突出重要信息来更好地解释Transformer模型

    Better Explain Transformers by Illuminating Important Information. (arXiv:2401.09972v1 [cs.CL])

    [http://arxiv.org/abs/2401.09972](http://arxiv.org/abs/2401.09972)

    通过在层间相关传播方法之上使用精细化的信息流，该论文提出了一种解释Transformer模型的方法，突出重要信息并消除无关信息。实验证明，在处理分类和问答任务时，这种方法相比其他八种基线模型更加出色。

    

    基于Transformer的模型在各种自然语言处理（NLP）任务中表现出色，吸引了无数努力来解释其内部工作原理。现有方法通过关注原始梯度和注意力来解释Transformer，将非相关信息通常视为解释计算的一部分，导致结果混乱。在这项工作中，我们提出了一种在层间相关传播（LRP）方法之上通过精细化信息流来突出重要信息并消除无关信息的方法。具体而言，我们考虑将句法和位置头识别为重要注意力头，并专注于从这些重要头部获得的相关性。实验结果表明，无关信息确实会扭曲输出的归因分数，因此在解释计算过程中应该对其进行屏蔽。与八种基线模型在分类和问答数据集上的比较结果显示，我们的方法在结果上不断地表现优秀。

    Transformer-based models excel in various natural language processing (NLP) tasks, attracting countless efforts to explain their inner workings. Prior methods explain Transformers by focusing on the raw gradient and attention as token attribution scores, where non-relevant information is often considered during explanation computation, resulting in confusing results. In this work, we propose highlighting the important information and eliminating irrelevant information by a refined information flow on top of the layer-wise relevance propagation (LRP) method. Specifically, we consider identifying syntactic and positional heads as important attention heads and focus on the relevance obtained from these important heads. Experimental results demonstrate that irrelevant information does distort output attribution scores and then should be masked during explanation computation. Compared to eight baselines on both classification and question-answering datasets, our method consistently outperfo
    
[^22]: 无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码

    Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access. (arXiv:2401.09967v1 [cs.CL])

    [http://arxiv.org/abs/2401.09967](http://arxiv.org/abs/2401.09967)

    本文介绍了一种无需访问逻辑回归的黑盒大型语言模型的草图引导约束解码的方法，通过利用本地辅助模型优化黑盒语言模型的输出，以初步输出作为进一步扩展的 "草图"，从而提高了有限约束解码的应用能力。

    

    有限约束在语言模型输出的控制上提供了一种不需要重新训练或架构修改的方式，但通常只适用于拥有逻辑回归访问权限的模型，这对于黑盒大型语言模型存在限制。本文引入了一种新颖的基于草图引导的黑盒大型语言模型约束解码（SGCD）方法，无需访问黑盒语言模型的逻辑回归。SGCD利用本地辅助模型来优化无约束黑盒语言模型的输出，将其作为进一步扩展的“草图”。此方法可与传统的基于逻辑回归的技术相互补充，使有限约束解码在无法完全透明的模型环境中应用。通过实验展示了SGCD的有效性。

    Constrained decoding, a technique for enforcing constraints on language model outputs, offers a way to control text generation without retraining or architectural modifications. Its application is, however, typically restricted to models that give users access to next-token distributions (usually via softmax logits), which poses a limitation with blackbox large language models (LLMs). This paper introduces sketch-guided constrained decoding (SGCD), a novel approach to constrained decoding for blackbox LLMs, which operates without access to the logits of the blackbox LLM. SGCD utilizes a locally hosted auxiliary model to refine the output of an unconstrained blackbox LLM, effectively treating this initial output as a "sketch" for further elaboration. This approach is complementary to traditional logit-based techniques and enables the application of constrained decoding in settings where full model transparency is unavailable. We demonstrate the efficacy of SGCD through experiments in cl
    
[^23]: 有意义的模因分析：通过多模态解释提高对网络欺凌中的模因的理解

    Meme-ingful Analysis: Enhanced Understanding of Cyberbullying in Memes Through Multimodal Explanations. (arXiv:2401.09899v1 [cs.CL])

    [http://arxiv.org/abs/2401.09899](http://arxiv.org/abs/2401.09899)

    本研究提出了一个名为MultiBully-Ex的多模态解释的网络欺凌模因基准数据集，该数据集突出显示了视觉和文本模态，用于解释为什么给定的模因是网络欺凌行为。

    

    互联网模因在传达政治、心理和社会文化观念方面具有重要影响力。虽然模因常常具有幽默的特点，但使用模因进行恶作剧和网络欺凌的现象正逐渐增加。虽然已经开发出了各种有效的基于深度学习的模型来检测冒犯性多模态模因，但在可解释性方面只有少数工作。类似于《通用数据保护条例》中的“解释权”，最近的相关法律已经推动了研究人员开发可解释的模型，而不仅仅关注性能。受此启发，我们引入了第一个用于多模态解释的代码混合网络欺凌模因基准数据集MultiBully-Ex。在这里，高亮显示了视觉和文本模态，以解释为什么给定的模因是网络欺凌行为。提出了基于对比语言-图像预训练（CLIP）投影的多模态共享-私有多任务方法，用于视觉和文本解释。

    Internet memes have gained significant influence in communicating political, psychological, and sociocultural ideas. While memes are often humorous, there has been a rise in the use of memes for trolling and cyberbullying. Although a wide variety of effective deep learning-based models have been developed for detecting offensive multimodal memes, only a few works have been done on explainability aspect. Recent laws like "right to explanations" of General Data Protection Regulation, have spurred research in developing interpretable models rather than only focusing on performance. Motivated by this, we introduce {\em MultiBully-Ex}, the first benchmark dataset for multimodal explanation from code-mixed cyberbullying memes. Here, both visual and textual modalities are highlighted to explain why a given meme is cyberbullying. A Contrastive Language-Image Pretraining (CLIP) projection-based multimodal shared-private multitask approach has been proposed for visual and textual explanation of 
    
[^24]: 关于大型语言模型的硬件加速器的调查

    A Survey on Hardware Accelerators for Large Language Models. (arXiv:2401.09890v1 [cs.AR])

    [http://arxiv.org/abs/2401.09890](http://arxiv.org/abs/2401.09890)

    这项论文调查了用于增强大型语言模型性能和能量效率的硬件加速器，并对多种加速器进行了深入分析，为研究人员、工程师和决策者在实际应用中优化大型语言模型的部署提供了宝贵的见解。

    

    大型语言模型（LLMs）已成为自然语言处理任务的强大工具，通过其理解和生成类似于人类文本的能力，它们正在为该领域带来革命性变革。随着对更复杂LLMs的需求不断增长，迫切需要解决与其规模和复杂性相关的计算挑战。本文针对提高大型语言模型性能和能量效率的硬件加速器进行了全面调查。通过对包括GPU、FPGA和定制架构在内的各种加速器进行研究，我们探索了旨在满足LLMs的独特计算需求的硬件解决方案的格局。本调查涵盖了对架构、性能指标和能量效率考虑的深入分析，为研究人员、工程师和决策者在实际应用中优化LLMs的部署提供了宝贵的见解。

    Large Language Models (LLMs) have emerged as powerful tools for natural language processing tasks, revolutionizing the field with their ability to understand and generate human-like text. As the demand for more sophisticated LLMs continues to grow, there is a pressing need to address the computational challenges associated with their scale and complexity. This paper presents a comprehensive survey on hardware accelerators designed to enhance the performance and energy efficiency of Large Language Models. By examining a diverse range of accelerators, including GPUs, FPGAs, and custom-designed architectures, we explore the landscape of hardware solutions tailored to meet the unique computational demands of LLMs. The survey encompasses an in-depth analysis of architecture, performance metrics, and energy efficiency considerations, providing valuable insights for researchers, engineers, and decision-makers aiming to optimize the deployment of LLMs in real-world applications.
    
[^25]: 基于注意力的循环神经网络对自动行为下蛋鸡识别的研究

    Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition. (arXiv:2401.09880v1 [cs.SD])

    [http://arxiv.org/abs/2401.09880](http://arxiv.org/abs/2401.09880)

    本研究提出了一种基于注意力的循环神经网络用于自动识别下蛋鸡行为。通过声音分析和特征提取，构建了一个鲁棒的行为特征化系统，对下蛋鸡的健康行为进行监测和识别。实验结果表明该模型具有良好的综合性能。

    

    现代养禽业的一个关注点是下蛋鸡的鸣叫声，其中包含了关于健康行为的非常有用的信息。这些信息被用作健康和福祉的指标，帮助养殖人员更好地监测下蛋鸡，从而及早发现问题，以便进行更快和更有效的干预。本研究专注于对下蛋鸡鸣叫类型的声音分析，以提出一种鲁棒的行为特征化系统，以便更好地监测下蛋鸡。为此，我们首先收集并注释了下蛋鸡的鸣叫信号，然后设计了一个基于时间和频率域特征组合的最佳声学特征化方法。然后我们使用这些特征构建了基于循环神经网络的多标签分类模型，将语义类别分配给描述下蛋鸡行为的鸣叫声。结果表明，我们的模型在综合性能上表现出色。

    One of the interests of modern poultry farming is the vocalization of laying hens which contain very useful information on health behavior. This information is used as health and well-being indicators that help breeders better monitor laying hens, which involves early detection of problems for rapid and more effective intervention. In this work, we focus on the sound analysis for the recognition of the types of calls of the laying hens in order to propose a robust system of characterization of their behavior for a better monitoring. To do this, we first collected and annotated laying hen call signals, then designed an optimal acoustic characterization based on the combination of time and frequency domain features. We then used these features to build the multi-label classification models based on recurrent neural network to assign a semantic class to the vocalization that characterize the laying hen behavior. The results show an overall performance with our model based on the combinati
    
[^26]: 大型语言模型提示的进化多目标优化以平衡情感

    Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments. (arXiv:2401.09862v1 [cs.NE])

    [http://arxiv.org/abs/2401.09862](http://arxiv.org/abs/2401.09862)

    本研究提出了一种针对语言模型提示优化的进化多目标方法，通过情感分析为案例研究，实现了生成能够同时体现两种相互冲突情感的提示语，从而提高模型的性能和相关信息的提取能力。

    

    大型语言模型（LLMs）如ChatGPT的出现引起了各个领域的广泛关注，因为它们的性能和多功能性非凡。随着这些模型的使用不断增长，有效的提示工程变得越来越重要。提示优化成为一个关键挑战，因为它直接影响模型性能和相关信息的提取。最近，进化算法（EAs）在解决这个问题方面显示出了希望，为新的优化策略铺平了道路。在这项工作中，我们提出了一种特别针对提示优化的进化多目标（EMO）方法，称为EMO-Prompts，以情感分析作为案例研究。我们将情感分析能力作为我们的实验目标。我们的结果表明，EMO-Prompts能够有效地生成提示，使LLM能够同时产生体现两种相互冲突情感的文本。

    The advent of large language models (LLMs) such as ChatGPT has attracted considerable attention in various domains due to their remarkable performance and versatility. As the use of these models continues to grow, the importance of effective prompt engineering has come to the fore. Prompt optimization emerges as a crucial challenge, as it has a direct impact on model performance and the extraction of relevant information. Recently, evolutionary algorithms (EAs) have shown promise in addressing this issue, paving the way for novel optimization strategies. In this work, we propose a evolutionary multi-objective (EMO) approach specifically tailored for prompt optimization called EMO-Prompts, using sentiment analysis as a case study. We use sentiment analysis capabilities as our experimental targets. Our results demonstrate that EMO-Prompts effectively generates prompts capable of guiding the LLM to produce texts embodying two conflicting emotions simultaneously.
    
[^27]: MatSciRE:利用指针网络自动化材料科学知识库构建中的实体和关系提取

    MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction. (arXiv:2401.09839v1 [cs.CL])

    [http://arxiv.org/abs/2401.09839](http://arxiv.org/abs/2401.09839)

    本论文提出了MatSciRE，一种基于指针网络的编码器-解码器框架，用于从材料科学文章中自动提取实体和关系以构建一个材料科学知识库。通过针对电池材料的五个关系的提取任务，我们的方法在F1分数上取得了比之前使用ChemDataExtractor更好的结果。

    

    材料科学文献是关于各种实体（如材料和成分）和这些实体之间各种关系（如导电性、电压等）的丰富来源。自动提取这些信息以生成一个材料科学知识库是一项具有挑战性的任务。在本文中，我们提出了MatSciRE（材料科学关系提取器），一种基于指针网络的编码器-解码器框架，用于从材料科学文章中同时提取实体和关系作为三元组（$实体1，关系，实体2$）。具体而言，我们针对电池材料，并确定了五个要处理的关系 - 导电性、库伦效率、容量、电压和能量。我们提出的方法在F1分数上取得了比使用ChemDataExtractor（0.716）更好的结果（0.771）。MatSciRE的整体图形框架如图1所示。材料信息以实体和关系的形式从材料科学文献中提取出来。

    Material science literature is a rich source of factual information about various categories of entities (like materials and compositions) and various relations between these entities, such as conductivity, voltage, etc. Automatically extracting this information to generate a material science knowledge base is a challenging task. In this paper, we propose MatSciRE (Material Science Relation Extractor), a Pointer Network-based encoder-decoder framework, to jointly extract entities and relations from material science articles as a triplet ($entity1, relation, entity2$). Specifically, we target the battery materials and identify five relations to work on - conductivity, coulombic efficiency, capacity, voltage, and energy. Our proposed approach achieved a much better F1-score (0.771) than a previous attempt using ChemDataExtractor (0.716). The overall graphical framework of MatSciRE is shown in Fig 1. The material information is extracted from material science literature in the form of ent
    
[^28]: 简单而有效的数据增强方法对于组合泛化具有显著作用

    Simple and effective data augmentation for compositional generalization. (arXiv:2401.09815v1 [cs.CL])

    [http://arxiv.org/abs/2401.09815](http://arxiv.org/abs/2401.09815)

    本文表明，通过采样MR并进行反向翻译的数据增强方法可以有效提高对于组合泛化的性能，尤其是当从正确的分布中进行采样时效果更好。与从训练分布中采样的方法相比，从均匀分布中进行采样的效果几乎与从测试分布中进行采样相当，并且效果更好。

    

    组合化泛化，即通过在较简单的句子上进行训练来预测复杂的含义，对于强大的预训练序列到序列模型来说是一个挑战。本文展示了通过采样MR并进行反向翻译的数据增强方法可以对组合泛化产生有效的效果，但仅当我们从正确的分布中进行采样时才是如此。值得注意的是，从均匀分布中进行采样的效果几乎与从测试分布中进行采样相当，并且比之前从训练分布中采样的方法效果更好。我们进一步进行实验来探究为何会出现这种情况以及这类数据增强方法的好处来自何处。

    Compositional generalization, the ability to predict complex meanings from training on simpler sentences, poses challenges for powerful pretrained seq2seq models. In this paper, we show that data augmentation methods that sample MRs and backtranslate them can be effective for compositional generalization, but only if we sample from the right distribution. Remarkably, sampling from a uniform distribution performs almost as well as sampling from the test distribution, and greatly outperforms earlier methods that sampled from the training distribution. We further conduct experiments to investigate the reason why this happens and where the benefit of such data augmentation methods come from.
    
[^29]: 一种简单的黑盒方法用于越狱攻击

    All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks. (arXiv:2401.09798v1 [cs.CL])

    [http://arxiv.org/abs/2401.09798](http://arxiv.org/abs/2401.09798)

    本研究提出了一种简单的黑盒方法，用于生成越狱攻击提示，克服了现有方法的复杂性和计算成本的限制。该方法通过使用语言模型自身，将有害提示重写为非有害表达，实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。

    

    像ChatGPT这样的大型语言模型面临着“越狱”挑战，即规避保障措施以产生不符合伦理的提示。本研究引入了一种简单的黑盒方法，有效地生成越狱提示，克服了现有方法的高复杂性和计算成本的限制。该方法通过使用目标语言模型自身，迭代地将有害提示重写为非有害表达，基于假设认为语言模型可以直接生成规避保障的表达。通过在ChatGPT（GPT-3.5和GPT-4）和Gemini-Pro上进行实验证明，该方法在平均5次迭代内实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。生成的越狱提示自然而简练，表明它们较不易被检测。结果表明，创建有效的越狱提示比先前研究认为的要简单，并且黑盒越狱攻击构成了一个重要的挑战。

    Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study introduces a simple black-box method to effectively generate jailbreak prompts, overcoming the limitations of high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample safeguard-bypassing expressions. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The jailbreak prompts generated were naturally-worded and concise, suggesting they are less detectable. The results indicate that creating effective jailbreak prompts is simpler than previously considered, and black-box jailbreak attacks pose 
    
[^30]: 电子商务买卖双方在线消息中的即时回答

    Instant Answering in E-Commerce Buyer-Seller Messaging. (arXiv:2401.09785v1 [cs.CL])

    [http://arxiv.org/abs/2401.09785](http://arxiv.org/abs/2401.09785)

    通过使用低延迟的序列到序列方法，我们成功地将电子商务顾客的消息转化为简洁的问题，从而实现了在电子商务买卖双方在线消息中的即时回答。实验证明，我们的方法在问题理解和回答率方面相对增加了很多，对于提高顾客的购物体验非常有效。

    

    电子商务顾客经常寻求详细的产品信息以做出购买决策，通常通过直接向卖家发送扩展查询来联系。这种手动回复要求增加了额外的成本，并且在响应时间波动范围从几小时到几天时干扰了顾客的购物体验。我们旨在使用领先的电子商务商店中的特定领域联合问答（QA）系统自动处理顾客对卖家的询问。主要挑战是将当前为单个问题设计的QA系统调整为解决详细的顾客查询。我们通过一种低延迟的序列到序列方法——MESSAGE-TO-QUESTION（M2Q）来解决这个问题，它通过从消息中识别和提取最重要的信息来将买家消息重新构建成简洁的问题。与基线的评估显示，M2Q在问题理解方面相对增加了757%，在联合问答系统中的回答率增加了1746%。实际部署表明，自动化回答系统可以以一个平均的回答速率快4.67倍。

    E-commerce customers frequently seek detailed product information for purchase decisions, commonly contacting sellers directly with extended queries. This manual response requirement imposes additional costs and disrupts buyer's shopping experience with response time fluctuations ranging from hours to days. We seek to automate buyer inquiries to sellers in a leading e-commerce store using a domain-specific federated Question Answering (QA) system. The main challenge is adapting current QA systems, designed for single questions, to address detailed customer queries. We address this with a low-latency, sequence-to-sequence approach, MESSAGE-TO-QUESTION ( M2Q ). It reformulates buyer messages into succinct questions by identifying and extracting the most salient information from a message. Evaluation against baselines shows that M2Q yields relative increases of 757% in question understanding, and 1,746% in answering rate from the federated QA system. Live deployment shows that automatic a
    
[^31]: 利用大型语言模型中的偏见：有效的少样本学习中的“bias-kNN”

    Leveraging Biases in Large Language Models: "bias-kNN'' for Effective Few-Shot Learning. (arXiv:2401.09783v1 [cs.CL])

    [http://arxiv.org/abs/2401.09783](http://arxiv.org/abs/2401.09783)

    本研究介绍了一种名为“bias-kNN”的新方法，它利用大型语言模型中的偏见，在少样本学习中表现出比传统方法更好的效果，并且在不同样本和模型情境下表现出鲁棒性。这一方法提供了一种独特的视角，将偏见转化为提升模型性能的资产。

    

    大型语言模型（LLMs）在零样本和少样本学习等各种应用中显示出了显著的潜力。然而，它们的性能可能受到内在偏见的限制。本研究介绍了一种名为“bias-kNN”的新方法，而不是传统上致力于最小化或修正这些偏见的方法。该方法利用有偏见的输出，将其作为kNN的主要特征，并与金标签相结合。我们对多样化领域文本分类数据集和不同GPT-2模型尺寸的全面评估表明了“bias-kNN”方法的适应性和效果。值得注意的是，这种方法不仅在少样本场景中优于传统的上下文学习，而且在各种样本、模板和言语器上表现出鲁棒性。因此，本研究提出了一种将偏见转化为增强模型性能的资产的独特观点。

    Large Language Models (LLMs) have shown significant promise in various applications, including zero-shot and few-shot learning. However, their performance can be hampered by inherent biases. Instead of traditionally sought methods that aim to minimize or correct these biases, this study introduces a novel methodology named ``bias-kNN''. This approach capitalizes on the biased outputs, harnessing them as primary features for kNN and supplementing with gold labels. Our comprehensive evaluations, spanning diverse domain text classification datasets and different GPT-2 model sizes, indicate the adaptability and efficacy of the ``bias-kNN'' method. Remarkably, this approach not only outperforms conventional in-context learning in few-shot scenarios but also demonstrates robustness across a spectrum of samples, templates and verbalizers. This study, therefore, presents a unique perspective on harnessing biases, transforming them into assets for enhanced model performance.
    
[^32]: 可控制的对是/否问题和答案进行去情境化处理，转化为事实陈述

    Controllable Decontextualization of Yes/No Question and Answers into Factual Statements. (arXiv:2401.09775v1 [cs.CL])

    [http://arxiv.org/abs/2401.09775](http://arxiv.org/abs/2401.09775)

    本论文解决了将极性问题的答案重写为脱离情境且简洁的事实陈述的问题，提出了一个利用Transformer模型实现的可控制重写方法，并在三个独立的PQA数据集上进行了评估

    

    是/否问题或极性问题是主要的语言问题类别之一。它们由一个主要的疑问子句组成，其答案是二进制的（肯定或否定）。极性问题和答案（PQA）是许多社区和其他经过筛选的问答资源中的有价值的知识资源，例如论坛或电子商务应用程序。单独使用极性问题的答案在其他语境中并不是很容易。答案是情境化的，并假设提问者和回答者之间的共享知识以及疑问子句都已提供。我们解决了将极性问题的答案重写为脱离情境且简洁的事实陈述的问题。我们提出了一个Transformer序列到序列模型，利用软约束确保可控制的重写，使得输出的陈述在语义上等同于其PQA输入。通过自动化和人工评估，在三个独立的PQA数据集上进行评估

    Yes/No or polar questions represent one of the main linguistic question categories. They consist of a main interrogative clause, for which the answer is binary (assertion or negation). Polar questions and answers (PQA) represent a valuable knowledge resource present in many community and other curated QA sources, such as forums or e-commerce applications. Using answers to polar questions alone in other contexts is not trivial. Answers are contextualized, and presume that the interrogative question clause and any shared knowledge between the asker and answerer are provided.  We address the problem of controllable rewriting of answers to polar questions into decontextualized and succinct factual statements. We propose a Transformer sequence to sequence model that utilizes soft-constraints to ensure controllable rewriting, such that the output statement is semantically equivalent to its PQA input. Evaluation on three separate PQA datasets as measured through automated and human evaluation
    
[^33]: 关于大型音视频语言模型中的音频幻听问题

    On the Audio Hallucinations in Large Audio-Video Language Models. (arXiv:2401.09774v1 [cs.MM])

    [http://arxiv.org/abs/2401.09774](http://arxiv.org/abs/2401.09774)

    本文分析了大型音视频语言模型中的音频幻听问题。通过收集1,000个句子并进行分类，研究发现有332个句子出现了幻听现象，针对这个问题，使用预训练的音频文本模型进行了零样本学习和微调，结果显示微调模型具有更好的性能。

    

    大型音视频语言模型可以为视频和音频生成描述。然而，它们有时会忽略音频内容，仅根据视觉信息生成音频描述。本文将此称为音频幻听，并对大型音视频语言模型中的幻听进行了分析。我们收集了1000个句子，通过询问音频信息，并注释它们是否包含幻听。如果一个句子是幻听的，我们还对幻听类型进行了分类。结果表明，有332个句子是幻听的，并且在每种幻听类型的名词和动词中观察到了明显的趋势。基于这些结果，我们使用预训练的音频文本模型在零样本学习和微调设置下解决了音频幻听分类任务。我们的实验结果显示，零样本模型的性能较高（F1为52.2%），优于随机模型（40.3%），而微调模型的性能为87.9%，超过了零样本模型。

    Large audio-video language models can generate descriptions for both video and audio. However, they sometimes ignore audio content, producing audio descriptions solely reliant on visual information. This paper refers to this as audio hallucinations and analyzes them in large audio-video language models. We gather 1,000 sentences by inquiring about audio information and annotate them whether they contain hallucinations. If a sentence is hallucinated, we also categorize the type of hallucination. The results reveal that 332 sentences are hallucinated with distinct trends observed in nouns and verbs for each hallucination type. Based on this, we tackle a task of audio hallucination classification using pre-trained audio-text models in the zero-shot and fine-tuning settings. Our experimental results reveal that the zero-shot models achieve higher performance (52.2% in F1) than the random (40.3%) and the fine-tuning models achieve 87.9%, outperforming the zero-shot models.
    
[^34]: 通过标签聚合，众包和LLM的标注质量的比较研究

    A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation. (arXiv:2401.09760v1 [cs.CL])

    [http://arxiv.org/abs/2401.09760](http://arxiv.org/abs/2401.09760)

    本文通过对众包标签和LLM标签的质量比较研究，探讨了大型语言模型是否能在数据标注任务上胜过众包。本研究对现有众包数据集进行了利用并创建了一个基准来评估标注质量。评估结果显示聚合标签的质量对于众包任务尤为重要。

    

    近来，大型语言模型(LLM)是否能在数据标注任务上胜过众包引起了人们的兴趣。一些研究通过采集新的数据集，通过对个体众包工作者和LLM工作者在一些特定的自然语言处理任务上的平均表现来验证这个问题。然而，一方面，现有的用于研究众包标注质量的数据集尚未在这样的评估中得到利用，这可能从不同的角度提供可靠的评估。另一方面，聚合标签的质量至关重要，因为在使用众包时，从多个众包标签到相同实例的估计标签是最终收集到的标签。因此，在本文中，我们首先研究了哪些现有的众包数据集可以用于比较研究并创建了一个基准。然后我们比较了个体众包标签和LLM标签的质量，并对聚合标签进行了评估。

    Whether Large Language Models (LLMs) can outperform crowdsourcing on the data annotation task is attracting interest recently. Some works verified this issue with the average performance of individual crowd workers and LLM workers on some specific NLP tasks by collecting new datasets. However, on the one hand, existing datasets for the studies of annotation quality in crowdsourcing are not yet utilized in such evaluations, which potentially provide reliable evaluations from a different viewpoint. On the other hand, the quality of these aggregated labels is crucial because, when utilizing crowdsourcing, the estimated labels aggregated from multiple crowd labels to the same instances are the eventually collected labels. Therefore, in this paper, we first investigate which existing crowdsourcing datasets can be used for a comparative study and create a benchmark. We then compare the quality between individual crowd labels and LLM labels and make the evaluations on the aggregated labels. I
    
[^35]: 解决命名实体中的正规多义性

    Resolving Regular Polysemy in Named Entities. (arXiv:2401.09758v1 [cs.CL])

    [http://arxiv.org/abs/2401.09758](http://arxiv.org/abs/2401.09758)

    本论文提出了一个结合了汉语词汇网(CWN)上常见单词义项消歧模型和点对象作为专有名词的消歧模型。该模型在解决常见名词和专有名词的歧义问题上取得了竞争性的结果。

    

    词义消歧主要解决基于预定义义项库的常见单词的语义模糊性。相反，专有名词通常被认为表示特定的现实世界指称。一旦确定了引用，就被认为解决了歧义。然而，专有名词也通过变成通用名词而产生歧义，即它们表现得像常见单词，并且可能表示它们引用的不同方面。我们提议通过正规多义性的光照来解决专有名词的歧义，我们将其正式化为点对象。本文介绍了一个结合了汉语词汇网(CWN)上常见单词义项消歧模型和点对象作为专有名词的消歧模型。该模型利用基于词汇网义项和例句的模型架构的灵活性。我们展示了该模型在常见名词和专有名词上取得了竞争性的结果，即使在相对稀疏的义项数据上也是如此。

    Word sense disambiguation primarily addresses the lexical ambiguity of common words based on a predefined sense inventory. Conversely, proper names are usually considered to denote an ad-hoc real-world referent. Once the reference is decided, the ambiguity is purportedly resolved. However, proper names also exhibit ambiguities through appellativization, i.e., they act like common words and may denote different aspects of their referents. We proposed to address the ambiguities of proper names through the light of regular polysemy, which we formalized as dot objects. This paper introduces a combined word sense disambiguation (WSD) model for disambiguating common words against Chinese Wordnet (CWN) and proper names as dot objects. The model leverages the flexibility of a gloss-based model architecture, which takes advantage of the glosses and example sentences of CWN. We show that the model achieves competitive results on both common and proper nouns, even on a relatively sparse sense dat
    
[^36]: 大型语言模型横向网络钓鱼：大规模组织环境中的比较研究

    Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings. (arXiv:2401.09727v1 [cs.CR])

    [http://arxiv.org/abs/2401.09727](http://arxiv.org/abs/2401.09727)

    本研究比较了大型语言模型在大规模组织环境中实现横向网络钓鱼的情况，并发现现有的反钓鱼基础设施无法防止语言模型生成的钓鱼攻击。

    

    钓鱼电子邮件的严重威胁被LLMs生成高度定向、个性化和自动化的鱼叉式网络钓鱼攻击的潜力进一步恶化。关于LLM促成的钓鱼存在两个关键问题需要进一步调查：1）现有的横向网络钓鱼研究缺乏针对整个组织进行大规模攻击的LLM整合的具体审查；2）尽管反钓鱼基础设施经过广泛开发，但仍无法防止LLM生成的攻击，可能影响员工和IT安全事件管理。然而，进行这样的调查研究需要在现实世界环境中进行，该环境在正常业务运作期间工作，并反映出大型组织基础设施的复杂性。此设置还必须提供所需的灵活性，以促进各种实验条件的实施，特别是钓鱼电子邮件的制作和组织范围的攻击。

    The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted
    
[^37]: 预测病毒谣言和易受攻击用户的信息传播监测方法

    Predicting Viral Rumors and Vulnerable Users for Infodemic Surveillance. (arXiv:2401.09724v1 [cs.SI])

    [http://arxiv.org/abs/2401.09724](http://arxiv.org/abs/2401.09724)

    该论文提出了一种预测病毒谣言和易受攻击用户的新方法，通过使用统一的图神经网络模型，并结合预训练的用户嵌入、交叉注意机制和增强社区传播脆弱性的方法来改进表示，以及采用多任务训练策略来提高性能。

    

    在信息疫情时代，有效监测迅速传播的疯狂谣言，并识别易受攻击的用户，对于及时采取预防措施，减轻虚假信息对社会的负面影响至关重要。我们提出了一种新颖的方法来预测病毒谣言和易受攻击的用户，使用统一的图神经网络模型。我们预先训练基于网络的用户嵌入，并利用用户和帖子之间的交叉注意机制，以及一个增强社区传播脆弱性的方法来改进用户和传播图的表示。此外，我们采用两种多任务训练策略来减轻不同设置中任务之间的负面转移效应，提高方法的整体性能。我们还构建了两个包含具体标注信息的数据集，用于验证信息传播的真实性。

    In the age of the infodemic, it is crucial to have tools for effectively monitoring the spread of rampant rumors that can quickly go viral, as well as identifying vulnerable users who may be more susceptible to spreading such misinformation. This proactive approach allows for timely preventive measures to be taken, mitigating the negative impact of false information on society. We propose a novel approach to predict viral rumors and vulnerable users using a unified graph neural network model. We pre-train network-based user embeddings and leverage a cross-attention mechanism between users and posts, together with a community-enhanced vulnerability propagation (CVP) method to improve user and propagation graph representations. Furthermore, we employ two multi-task training strategies to mitigate negative transfer effects among tasks in different settings, enhancing the overall performance of our approach. We also construct two datasets with ground-truth annotations on information virali
    
[^38]: 使用基于Transformer的模型与InfoNCE损失和语言切换方法的课程推荐

    Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method. (arXiv:2401.09699v1 [cs.CL])

    [http://arxiv.org/abs/2401.09699](http://arxiv.org/abs/2401.09699)

    这项研究提出了使用Transformer基础模型、InfoNCE损失和语言切换方法来解决课程推荐中的内容冲突和语言翻译引起的干扰问题，旨在构建一个个性化学习体验、包容多样性的教育环境。

    

    课程推荐范式致力于在不断发展的教育技术和课程开发领域中促进学习平等。鉴于现有方法所面临的内容冲突和语言翻译引起的干扰等困难，该范式旨在面对并克服这些挑战。特别是，它解决了语言翻译引入的内容冲突和干扰问题，这些问题可能阻碍创建全面和个性化学习体验。该范式的目标是培养一个既包容多样性又可以根据每个学习者的独特需求定制学习体验的教育环境。为了克服这些挑战，我们的方法在课程开发和个性化学习方面引入了三个关键创新。其中包括使用Transformer基础模型增强计算能力。

    The Curriculum Recommendations paradigm is dedicated to fostering learning equality within the ever-evolving realms of educational technology and curriculum development. In acknowledging the inherent obstacles posed by existing methodologies, such as content conflicts and disruptions from language translation, this paradigm aims to confront and overcome these challenges. Notably, it addresses content conflicts and disruptions introduced by language translation, hindrances that can impede the creation of an all-encompassing and personalized learning experience. The paradigm's objective is to cultivate an educational environment that not only embraces diversity but also customizes learning experiences to suit the distinct needs of each learner. To overcome these challenges, our approach builds upon notable contributions in curriculum development and personalized learning, introducing three key innovations. These include the integration of Transformer Base Model to enhance computational e
    
[^39]: 用大型语言模型表征在线饮食紊乱社群

    Characterizing Online Eating Disorder Communities with Large Language Models. (arXiv:2401.09647v1 [cs.SI])

    [http://arxiv.org/abs/2401.09647](http://arxiv.org/abs/2401.09647)

    通过网络和语言分析，我们表征了在线社群中推广饮食紊乱的动态，认为社交媒体平台放大了这一现象。使用大型语言模型和分析社群内的话语，我们探测到了与饮食紊乱相关的潜在情况。

    

    饮食紊乱作为一种危险的心理健康状况，具有较高的死亡率和发病率，其上升与社交媒体上理想化身体形象的泛滥有关。然而，社交媒体与饮食紊乱之间的联系远不止如此。我们认为社交媒体平台创建了一个反馈循环，放大了推广厌食症和暴食症等饮食紊乱的内容和社群的增长。具体而言，社交媒体平台使易受伤害的个体能够轻松找到并联系到志同道合的其他人，而群体动态过程则鼓励他们在推广和美化与饮食紊乱相关的有害行为的社群中持续参与。我们通过网络和语言分析的组合，从经验上描述了这一动态。我们提出了一个新颖的框架，利用大型语言模型分析在线社群内的话语，并对与饮食紊乱相关的话题的态度进行探测，以鉴别潜在的情况。

    The rise in eating disorders, a dangerous mental health condition with high mortality and morbidity, has been linked to the proliferation of idealized body images on social media. However, the link between social media and eating disorders is far more complex. We argue that social media platforms create a feedback loop that amplifies the growth of content and communities that promote eating disorders like anorexia and bulimia. Specifically, social media platforms make it easy for vulnerable individuals to find and connect to like-minded others, while group dynamic processes encourage them to stay engaged within communities that promote and glorify harmful behaviors linked to eating disorders. We characterize this dynamic empirically through a combination of network and language analysis. We describe a novel framework that leverages large language models to analyze the discourse within online communities and probe their attitudes on topics related to eating disorders to identify potenti
    
[^40]: ClimateGPT: 实现对气候变化领域的跨学科研究进行合成的AI模型

    ClimateGPT: Towards AI Synthesizing Interdisciplinary Research on Climate Change. (arXiv:2401.09646v1 [cs.LG])

    [http://arxiv.org/abs/2401.09646](http://arxiv.org/abs/2401.09646)

    ClimateGPT是一个针对气候变化领域的跨学科研究合成的AI模型，通过优化检索增强和使用级联机器翻译方法，提高了模型的性能和可访问性。

    

    本文介绍了ClimateGPT，一种特定领域的大型语言模型系列，用于合成气候变化的跨学科研究。我们从头开始训练了两个7B模型，训练数据集包含300B个科学导向的令牌。第一个模型在预训练期间包含了4.2B个特定领域的令牌，第二个模型在预训练后针对气候领域进行了调整。此外，我们还对ClimateGPT-7B，13B和70B进行了连续预训练，训练数据集包含4.2B个特定领域的令牌，并与气候科学家紧密合作创建。为了减少虚构生成的数量，我们为模型进行了检索增强优化，并提出了一种分层检索策略。为了提高我们模型对非英语使用者的可访问性，我们建议利用级联机器翻译，并证明这种方法可以与翻译的性能相媲美。

    This paper introduces ClimateGPT, a model family of domain-specific large language models that synthesize interdisciplinary research on climate change. We trained two 7B models from scratch on a science-oriented dataset of 300B tokens. For the first model, the 4.2B domain-specific tokens were included during pre-training and the second was adapted to the climate domain after pre-training. Additionally, ClimateGPT-7B, 13B and 70B are continuously pre-trained from Llama~2 on a domain-specific dataset of 4.2B tokens. Each model is instruction fine-tuned on a high-quality and human-generated domain-specific dataset that has been created in close cooperation with climate scientists. To reduce the number of hallucinations, we optimize the model for retrieval augmentation and propose a hierarchical retrieval strategy. To increase the accessibility of our model to non-English speakers, we propose to make use of cascaded machine translation and show that this approach can perform comparably to 
    
[^41]: 大型语言模型辅助对患者阅读临床笔记的影响：一个混合方法研究

    Impact of Large Language Model Assistance on Patients Reading Clinical Notes: A Mixed-Methods Study. (arXiv:2401.09637v1 [cs.HC])

    [http://arxiv.org/abs/2401.09637](http://arxiv.org/abs/2401.09637)

    通过大型语言模型辅助阅读临床笔记，患者可以获得更好的理解和自信。这项研究开发了一个工具，利用语言模型简化和增加上下文，使临床笔记更易读。研究结果表明，这些增强对患者有益。

    

    患者通过阅读他们的临床笔记获得了许多好处，包括增加对自身健康的控制感和对护理计划的理解提高。然而，在临床笔记中复杂的医学概念和术语阻碍了患者的理解，并可能导致焦虑。我们开发了一个面向患者的工具，利用大型语言模型（LLMs）简化笔记、从中提取信息并增加上下文，以使临床笔记更易读。我们使用我们的工具提示改进的GPT-4对由乳腺癌幸存者捐赠的真实临床笔记和临床医生生成的合成临床笔记进行这些增强任务。共有12条笔记，3868个字。2023年6月，我们随机分配了200名美国女性参与者，并向他们分发了三个具有不同程度增强的临床笔记。参与者回答了有关每个笔记的问题，评估了他们对后续行动的理解和自我报告的自信心。我们发现增强对阅读理解和自信心友好。

    Patients derive numerous benefits from reading their clinical notes, including an increased sense of control over their health and improved understanding of their care plan. However, complex medical concepts and jargon within clinical notes hinder patient comprehension and may lead to anxiety. We developed a patient-facing tool to make clinical notes more readable, leveraging large language models (LLMs) to simplify, extract information from, and add context to notes. We prompt engineered GPT-4 to perform these augmentation tasks on real clinical notes donated by breast cancer survivors and synthetic notes generated by a clinician, a total of 12 notes with 3868 words. In June 2023, 200 female-identifying US-based participants were randomly assigned three clinical notes with varying levels of augmentations using our tool. Participants answered questions about each note, evaluating their understanding of follow-up actions and self-reported confidence. We found that augmentations were ass
    
[^42]: 学习捷径：关于语言模型中自然语言理解误导性承诺的论文

    Learning Shortcuts: On the Misleading Promise of NLU in Language Models. (arXiv:2401.09615v1 [cs.CL])

    [http://arxiv.org/abs/2401.09615](http://arxiv.org/abs/2401.09615)

    该论文调查了大型语言模型在自然语言理解任务中使用捷径学习的现象，强调了这种现象对语言模型评估的影响，并呼吁加大对捷径学习的研究力度以提升语言模型的鲁棒性和实际场景中的自然语言理解评估标准。

    

    大型语言模型（LLMs）的出现在自然语言处理领域实现了显著的性能提升。然而，最近的研究发现，LLMs在执行任务时常常采用捷径，导致在决策规则上缺乏泛化能力，从而在性能上产生了一种错觉。这一现象在准确评估LLMs的自然语言理解能力上带来了挑战。本文对该领域的相关研究进行了简洁的概述，并提出了在评估语言模型，尤其是自然语言理解任务中使用捷径学习的影响的观点。本文呼吁加大对捷径学习的深入理解的研究力度，为开发更强大的语言模型和提高真实场景下自然语言理解评估的标准作出贡献。

    The advent of large language models (LLMs) has enabled significant performance gains in the field of natural language processing. However, recent studies have found that LLMs often resort to shortcuts when performing tasks, creating an illusion of enhanced performance while lacking generalizability in their decision rules. This phenomenon introduces challenges in accurately assessing natural language understanding in LLMs. Our paper provides a concise survey of relevant research in this area and puts forth a perspective on the implications of shortcut learning in the evaluation of language models, specifically for NLU tasks. This paper urges more research efforts to be put towards deepening our comprehension of shortcut learning, contributing to the development of more robust language models, and raising the standards of NLU evaluation in real-world scenarios.
    
[^43]: 使用反事实对抗优化实现大型语言模型的对齐

    Aligning Large Language Models with Counterfactual DPO. (arXiv:2401.09566v1 [cs.CL])

    [http://arxiv.org/abs/2401.09566](http://arxiv.org/abs/2401.09566)

    本文研究了在大型语言模型中使用反事实对抗优化框架，以实现风格对齐，避免人类干预，并成功培养出可取行为和减轻不可取行为。

    

    大型语言模型(LLMs)的进步在各种应用中展示了卓越的能力。这些模型在生成上下文连贯且涵盖广泛主题的文本补全方面表现出色。然而，它们训练所需的大量数据使得在预训练和指令调整阶段对齐响应风格变得具有挑战性。因此，通常会采用额外的对齐阶段，进一步使用人类偏好数据对模型进行训练，以更好地将其输出与人类期望对齐。虽然这个过程本身并没有引入新的能力，但它突出了模型固有的生成风格。本文研究了在直接偏好优化(DPO)框架内利用反事实提示来对齐模型的风格，而不依赖人类干预。我们证明了这种方法有效地培养了可取的行为，减轻了不可取的行为。

    Advancements in large language models (LLMs) have demonstrated remarkable capabilities across a diverse range of applications. These models excel in generating text completions that are contextually coherent and cover an extensive array of subjects. However, the vast datasets required for their training make aligning response styles during the pretraining and instruction tuning phases challenging. Consequently, an additional alignment phase is typically employed, wherein the model is further trained with human preference data to better align its outputs with human expectations. While this process doesn't introduce new capabilities per se, it does accentuate generation styles innate to the model. This paper explores the utilization of counterfactual prompting within the framework of Direct Preference Optimization (DPO) to align the model's style without relying on human intervention. We demonstrate that this method effectively instils desirable behaviour, mitigates undesirable ones, and
    
[^44]: 通过人类反馈改进分类性能：标记一些，我们标记其余部分

    Improving Classification Performance With Human Feedback: Label a few, we label the rest. (arXiv:2401.09555v1 [cs.LG])

    [http://arxiv.org/abs/2401.09555](http://arxiv.org/abs/2401.09555)

    本文探讨了通过人类反馈来改进分类模型性能的方法。使用少量有标签示例，通过连续反馈循环，我们能够显著提高模型的准确性。在多个数据集上进行评估，结果表明这种方法能够超越零样本大型语言模型，提供更强的文本分类性能。

    

    在人工智能领域，大部分数据是非结构化的，因此获取足够的有标签数据来训练监督式机器学习模型是一个重要挑战。为了解决这个问题，我们深入研究少样本学习和主动学习，即通过人类反馈来改进人工智能模型，仅使用一小部分有标签示例。本文着重于理解连续反馈循环如何改善模型，从而通过渐进式的人类参与提高模型的准确性、回归和精确度。通过使用大型语言模型（LLMs），如GPT-3.5、BERT和SetFit，我们旨在分析使用有限数量的有标签示例显著提高模型准确性的效果。我们在Financial Phrasebank、Banking、Craigslist、Trec和Amazon Reviews数据集上对此方法进行基准测试，证明仅使用少量有标签示例就能超越零样本大型语言模型的准确性，提供增强的文本分类性能。

    In the realm of artificial intelligence, where a vast majority of data is unstructured, obtaining substantial amounts of labeled data to train supervised machine learning models poses a significant challenge. To address this, we delve into few-shot and active learning, where are goal is to improve AI models with human feedback on a few labeled examples. This paper focuses on understanding how a continuous feedback loop can refine models, thereby enhancing their accuracy, recall, and precision through incremental human input. By employing Large Language Models (LLMs) such as GPT-3.5, BERT, and SetFit, we aim to analyze the efficacy of using a limited number of labeled examples to substantially improve model accuracy. We benchmark this approach on the Financial Phrasebank, Banking, Craigslist, Trec, Amazon Reviews datasets to prove that with just a few labeled examples, we are able to surpass the accuracy of zero shot large language models to provide enhanced text classification performa
    
[^45]: BERTologyNavigator: 基于BERT语义的高级问题回答系统

    BERTologyNavigator: Advanced Question Answering with BERT-based Semantics. (arXiv:2401.09553v1 [cs.CL])

    [http://arxiv.org/abs/2401.09553](http://arxiv.org/abs/2401.09553)

    BERTologyNavigator是一个基于BERT语义的高级问题回答系统，结合关系抽取和BERT嵌入，可以在DBLP知识图谱中精确地导航关系，并在测试数据集上达到了较高的F1分数。

    

    知识图谱与语言模型的开发和集成在人工智能和自然语言处理方面具有重要意义。本研究介绍了BERTologyNavigator——一个将关系抽取技术和BERT嵌入相结合的两阶段系统，用于在DBLP知识图谱中导航关系。我们的方法专注于提取一跳关系和标记的候选对，然后在第二阶段使用BERT的CLS嵌入和其他启发式方法进行关系选择。我们的系统在Scholarly QALD的DBLP QuAD Final测试数据集上达到了0.2175的F1分数，而在DBLP QuAD测试数据集的子集上在QA阶段达到了0.98的F1分数。

    The development and integration of knowledge graphs and language models has significance in artificial intelligence and natural language processing. In this study, we introduce the BERTologyNavigator -- a two-phased system that combines relation extraction techniques and BERT embeddings to navigate the relationships within the DBLP Knowledge Graph (KG). Our approach focuses on extracting one-hop relations and labelled candidate pairs in the first phases. This is followed by employing BERT's CLS embeddings and additional heuristics for relation selection in the second phase. Our system reaches an F1 score of 0.2175 on the DBLP QuAD Final test dataset for Scholarly QALD and 0.98 F1 score on the subset of the DBLP QuAD test dataset during the QA phase.
    
[^46]: LoMA: 无损压缩的内存注意力

    LoMA: Lossless Compressed Memory Attention. (arXiv:2401.09486v1 [cs.LG])

    [http://arxiv.org/abs/2401.09486](http://arxiv.org/abs/2401.09486)

    LoMA是一种无损压缩的内存注意力方法，可以有效地处理长文本并减少资源消耗。

    

    处理长文本是大型语言模型（LLMs）最重要的能力之一，但随着文本长度的增加，资源消耗也急剧增加。目前，通过压缩KV缓存来减少资源消耗是一种常见的方法。尽管存在许多现有的压缩方法，但它们都有一个共同的缺点：压缩是有损的。也就是说，在压缩过程中信息不可避免地会丢失。如果压缩率很高，丢失重要信息的概率会大大增加。我们提出了一种新方法，无损压缩的内存注意力（LoMA），可以根据一组压缩比率将信息无损压缩成特殊的内存令牌KV对。我们的实验证明，LoMA具有出色的性能，可以高效训练且具有非常有效的性能。

    The ability to handle long texts is one of the most important capabilities of Large Language Models (LLMs), but as the text length increases, the consumption of resources also increases dramatically. At present, reducing resource consumption by compressing the KV cache is a common approach. Although there are many existing compression methods, they share a common drawback: the compression is not lossless. That is, information is inevitably lost during the compression process. If the compression rate is high, the probability of losing important information increases dramatically. We propose a new method, Lossless Compressed Memory Attention (LoMA), which allows for lossless compression of information into special memory token KV pairs according to a set compression ratio. Our experiments have achieved remarkable results, demonstrating that LoMA can be efficiently trained and has very effective performance.
    
[^47]: Voila-A: 用用户注视注意力对齐视觉-语言模型

    Voila-A: Aligning Vision-Language Models with User's Gaze Attention. (arXiv:2401.09454v1 [cs.CV])

    [http://arxiv.org/abs/2401.09454](http://arxiv.org/abs/2401.09454)

    本文介绍了一种使用用户注视注意力对齐视觉-语言模型的方法，在处理复杂场景和多个物体的实际应用中提高了模型的可解释性和效果。

    

    在最近几年中，视觉和语言理解的整合通过视觉-语言模型（VLMs）在人工智能领域取得了重要突破。然而，现有的VLMs在处理复杂场景和多个物体的实际应用以及与人类用户的多样化注意力模式相一致方面面临挑战。本文引入了通过增强现实（AR）或虚拟现实（VR）设备收集的注视信息，作为人类注意力的代理来引导VLMs，并提出了一种新的方法Voila-A，以提高这些模型在实际应用中的可解释性和效果。首先，我们收集了数百分钟的注视数据，以展示我们可以使用本地化的叙事来模拟人类的注视方式。然后，我们设计了一个自动数据注释流水线，利用GPT-4生成了VOILA-COCO数据集。此外，我们创新了Voila Perceiver模块，将注视信息整合到VL模型中。

    In recent years, the integration of vision and language understanding has led to significant advancements in artificial intelligence, particularly through Vision-Language Models (VLMs). However, existing VLMs face challenges in handling real-world applications with complex scenes and multiple objects, as well as aligning their focus with the diverse attention patterns of human users. In this paper, we introduce gaze information, feasibly collected by AR or VR devices, as a proxy for human attention to guide VLMs and propose a novel approach, Voila-A, for gaze alignment to enhance the interpretability and effectiveness of these models in real-world applications. First, we collect hundreds of minutes of gaze data to demonstrate that we can mimic human gaze modalities using localized narratives. We then design an automatic data annotation pipeline utilizing GPT-4 to generate the VOILA-COCO dataset. Additionally, we innovate the Voila Perceiver modules to integrate gaze information into VL
    
[^48]: 《可解释的孟加拉语Memes的多模态情感分析》

    Explainable Multimodal Sentiment Analysis on Bengali Memes. (arXiv:2401.09446v1 [cs.CV])

    [http://arxiv.org/abs/2401.09446](http://arxiv.org/abs/2401.09446)

    这项研究提出了一个多模态方法来解释孟加拉语Memes的情感，以填补此领域中低资源语言的研究空白。对比现有的数据集，提出了一个新的MemoSen数据集并表明其准确率的局限性。这项研究的主要贡献是在孟加拉语Memes情感分析领域引入了多模态方法。

    

    Memes已成为数字时代独特而有效的沟通形式，吸引了在线社区，并跨越文化障碍。尽管Memes经常和幽默联系在一起，但它们有着传达广泛情感的惊人能力，包括快乐、讽刺、沮丧等。在信息时代，理解和解释Memes背后的情感变得至关重要。先前的研究已探索了基于文本、基于图像和多模态方法，导致了像CAPSAN和PromptHate这样的模型的发展，用于检测各种Memes类别。然而，对于孟加拉语Memes这样的低资源语言的研究仍然稀缺，公开可用的数据集数量有限。最近的一个贡献是引入了MemoSen数据集。然而，所实现的准确率明显较低，并且数据集分布不平衡。在这项研究中，我们采用了ResNet50和多模态方法。

    Memes have become a distinctive and effective form of communication in the digital era, attracting online communities and cutting across cultural barriers. Even though memes are frequently linked with humor, they have an amazing capacity to convey a wide range of emotions, including happiness, sarcasm, frustration, and more. Understanding and interpreting the sentiment underlying memes has become crucial in the age of information. Previous research has explored text-based, image-based, and multimodal approaches, leading to the development of models like CAPSAN and PromptHate for detecting various meme categories. However, the study of low-resource languages like Bengali memes remains scarce, with limited availability of publicly accessible datasets. A recent contribution includes the introduction of the MemoSen dataset. However, the achieved accuracy is notably low, and the dataset suffers from imbalanced distribution. In this study, we employed a multimodal approach using ResNet50 and
    
[^49]: RoleCraft-GLM：推动大型语言模型中的个性化角色扮演

    RoleCraft-GLM: Advancing Personalized Role-Playing in Large Language Models. (arXiv:2401.09432v1 [cs.CL])

    [http://arxiv.org/abs/2401.09432](http://arxiv.org/abs/2401.09432)

    RoleCraft-GLM是一个创新框架，通过大型语言模型实现个性化角色扮演，解决了缺乏个性化互动的问题。通过独特的对话数据集和细致入微的角色发展，它能够生成准确反映角色个性特征和情感的对话，提升用户参与度。

    

    本研究介绍了RoleCraft-GLM，这是一个创新的框架，旨在通过大型语言模型（LLMs）增强个性化角色扮演。RoleCraft-GLM解决了对话式人工智能中缺乏个性化互动的关键问题，并提供了一种能够详细描绘情感细腻的角色刻画的解决方案。我们贡献了一组独特的对话数据集，这些数据从传统的以名人为中心的角色转变为多样化的非名人角色，从而增强了语言建模互动的真实性和复杂性。此外，我们的方法还包括细致入微的角色发展，确保对话既真实又情感共鸣。通过多个案例研究验证了RoleCraft-GLM的有效性，突显了它在不同场景中的多功能性和技能。我们的框架在生成对话方面表现出色，能够准确反映角色的个性特征和情感，从而增强用户参与度。总之，RoleCraft-GLM标志着一个创新的里程碑，推动了大型语言模型中的个性化角色扮演。

    This study presents RoleCraft-GLM, an innovative framework aimed at enhancing personalized role-playing with Large Language Models (LLMs). RoleCraft-GLM addresses the key issue of lacking personalized interactions in conversational AI, and offers a solution with detailed and emotionally nuanced character portrayals. We contribute a unique conversational dataset that shifts from conventional celebrity-centric characters to diverse, non-celebrity personas, thus enhancing the realism and complexity of language modeling interactions. Additionally, our approach includes meticulous character development, ensuring dialogues are both realistic and emotionally resonant. The effectiveness of RoleCraft-GLM is validated through various case studies, highlighting its versatility and skill in different scenarios. Our framework excels in generating dialogues that accurately reflect characters' personality traits and emotions, thereby boosting user engagement. In conclusion, RoleCraft-GLM marks a sign
    
[^50]: 对比性偏好优化：推动机器翻译中LLM性能的边界

    Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation. (arXiv:2401.08417v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.08417](http://arxiv.org/abs/2401.08417)

    本研究通过引入对比性偏好优化（CPO）的方法，弥合了大型语言模型（LLM）在机器翻译中性能与传统编码器-解码器模型之间的差距，实现了更好的翻译效果。

    

    中等规模的大型语言模型（LLM）——7B或者13B参数的模型在机器翻译（MT）任务中表现出有希望的性能。然而，即使是表现最好的13B LLM翻译模型，如ALMA，也无法达到现有的最先进的传统编码器-解码器翻译模型或者更大规模的LLM（如GPT-4）的性能水平。本研究旨在弥合这一性能差距。首先，我们评估了监督微调在MT任务中针对LLM的不足之处，强调了尽管是人工生成的参考数据，但其中存在质量问题。然后，与模仿参考翻译的SFT相反，我们引入了对比性偏好优化（CPO），一种新的方法，训练模型避免生成仅仅合乎要求但不完美的翻译。将CPO应用于仅有22K对句子和12M参数的ALMA模型中，可以显著提高性能。得到的模型称为ALMA-R，其性能可以达到或超过WMT比赛的获胜水平。

    Moderate-sized large language models (LLMs) -- those with 7B or 13B parameters -- exhibit promising machine translation (MT) performance. However, even the top-performing 13B LLM-based translation models, like ALMA, does not match the performance of state-of-the-art conventional encoder-decoder translation models or larger-scale LLMs such as GPT-4. In this study, we bridge this performance gap. We first assess the shortcomings of supervised fine-tuning for LLMs in the MT task, emphasizing the quality issues present in the reference data, despite being human-generated. Then, in contrast to SFT which mimics reference translations, we introduce Contrastive Preference Optimization (CPO), a novel approach that trains models to avoid generating adequate but not perfect translations. Applying CPO to ALMA models with only 22K parallel sentences and 12M parameters yields significant improvements. The resulting model, called ALMA-R, can match or exceed the performance of the WMT competition winn
    
[^51]: RAG vs Fine-tuning: 管道，权衡以及在农业上的个案研究

    RAG vs Fine-tuning: Pipelines, Tradeoffs, and a Case Study on Agriculture. (arXiv:2401.08406v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.08406](http://arxiv.org/abs/2401.08406)

    本文评估了检索增强生成（RAG）和微调两种方法在大型语言模型上的性能差异，并提出了适用于农业数据集的管道和权衡。

    

    在构建大型语言模型应用程序时，开发者通常有两种常见方法来整合专有和领域特定的数据：检索增强生成（RAG）和微调。RAG利用外部数据增强提示信息，而微调则将附加知识整合到模型中。然而，这两种方法的优缺点并不为人所理解。在本文中，我们提出了一个微调和RAG的管道，并对多种流行的大型语言模型（包括Llama2-13B，GPT-3.5和GPT-4）进行了权衡。我们的管道由多个阶段组成，包括从PDF中提取信息，生成问题和答案，将其用于微调，并利用GPT-4评估结果。我们提出了评估RAG和微调管道不同阶段性能的指标。我们对农业数据集进行了深入研究。作为一个产业，农业在人工智能的应用方面并没有得到很大的渗透。

    There are two common ways in which developers are incorporating proprietary and domain-specific data when building applications of Large Language Models (LLMs): Retrieval-Augmented Generation (RAG) and Fine-Tuning. RAG augments the prompt with the external data, while fine-Tuning incorporates the additional knowledge into the model itself. However, the pros and cons of both approaches are not well understood. In this paper, we propose a pipeline for fine-tuning and RAG, and present the tradeoffs of both for multiple popular LLMs, including Llama2-13B, GPT-3.5, and GPT-4. Our pipeline consists of multiple stages, including extracting information from PDFs, generating questions and answers, using them for fine-tuning, and leveraging GPT-4 for evaluating the results. We propose metrics to assess the performance of different stages of the RAG and fine-Tuning pipeline. We conduct an in-depth study on an agricultural dataset. Agriculture as an industry has not seen much penetration of AI, an
    
[^52]: 大型语言模型的自我解释是否可靠?

    Are self-explanations from Large Language Models faithful?. (arXiv:2401.07927v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07927](http://arxiv.org/abs/2401.07927)

    大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。

    

    经过训练的大型语言模型在许多任务上表现出色，甚至能够提供其行为的解释。由于这些模型对公众是直接可访问的，因此存在这样的风险，即令人信服但错误的解释可能导致对大型语言模型的无支撑的自信。因此，解释能力和可靠性是AI安全的重要考虑因素。评估自我解释的可靠性和可解释性是一项具有挑战性的任务，因为这些模型对于人类来说过于复杂，无法注释什么是正确的解释。为了解决这个问题，我们提出使用自洽性检测作为可靠性的衡量指标。例如，如果一个大型语言模型说某组词对于做出预测很重要，那么在没有这些词的情况下，它应该无法做出相同的预测。虽然自洽性检测是一种常见的可靠性方法，但之前尚未应用于大型语言模型的自我解释中。我们将自洽性检测应用于...

    Instruction-tuned large language models (LLMs) excel at many tasks, and will even provide explanations for their behavior. Since these models are directly accessible to the public, there is a risk that convincing and wrong explanations can lead to unsupported confidence in LLMs. Therefore, interpretability-faithfulness of self-explanations is an important consideration for AI Safety. Assessing the interpretability-faithfulness of these explanations, termed self-explanations, is challenging as the models are too complex for humans to annotate what is a correct explanation. To address this, we propose employing self-consistency checks as a measure of faithfulness. For example, if an LLM says a set of words is important for making a prediction, then it should not be able to make the same prediction without these words. While self-consistency checks are a common approach to faithfulness, they have not previously been applied to LLM's self-explanations. We apply self-consistency checks to t
    
[^53]: TAROT：一种在半结构化数据上进行多任务预训练的层次框架，以实现有效的人-岗位匹配

    TAROT: A Hierarchical Framework with Multitask Co-Pretraining on Semi-Structured Data towards Effective Person-Job Fit. (arXiv:2401.07525v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07525](http://arxiv.org/abs/2401.07525)

    TAROT是一个层次化的多任务预训练框架，通过对半结构化数据进行预训练，结合多粒度的任务来提升人-岗位匹配的效果。

    

    人-岗位匹配是在线招聘平台中的重要部分，可以用于各种下游应用，如职位搜索和候选人推荐。最近，预训练的大型语言模型通过利用用户简介和职位描述中的丰富文本信息以及用户行为特征和职位元数据，进一步增强了效果。然而，一般的面向领域的设计难以捕捉用户简介和职位描述中的独特结构信息，导致潜在语义相关性的丧失。我们提出了TAROT，一种层次化的多任务共同预训练框架，以更好地利用结构和语义信息进行信息性文本嵌入。TAROT针对简介和职位中的半结构化文本进行预训练，并通过多颗粒度的预训练任务来约束每个层次上获取的语义信息。在真实的LinkedIn数据集上的实验证明了显著的性能改进，验证了其有效性。

    Person-job fit is an essential part of online recruitment platforms in serving various downstream applications like Job Search and Candidate Recommendation. Recently, pretrained large language models have further enhanced the effectiveness by leveraging richer textual information in user profiles and job descriptions apart from user behavior features and job metadata. However, the general domain-oriented design struggles to capture the unique structural information within user profiles and job descriptions, leading to a loss of latent semantic correlations. We propose TAROT, a hierarchical multitask co-pretraining framework, to better utilize structural and semantic information for informative text embeddings. TAROT targets semi-structured text in profiles and jobs, and it is co-pretained with multi-grained pretraining tasks to constrain the acquired semantic information at each level. Experiments on a real-world LinkedIn dataset show significant performance improvements, proving its e
    
[^54]: 开发用于生物学和医学的ChatGPT：生物医学问题回答的完整综述

    Developing ChatGPT for Biology and Medicine: A Complete Review of Biomedical Question Answering. (arXiv:2401.07510v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07510](http://arxiv.org/abs/2401.07510)

    开发用于生物学和医学的ChatGPT，通过自然语言处理和多模态范式，加速了医学问题回答的进展，并且能够处理医学环境中的大规模、多样化、无标签数据分析场景。

    

    ChatGPT通过自然语言处理（NLP）和多模态范式，通过增加医学领域数据的融入，探索了在提供医学诊断、治疗建议和其他医疗支持方面的问答（QA）的战略蓝图。通过将文本、图像、视频和其他模态从通用领域转向医学领域，这些技术加快了医学领域问题回答（MDQA）的进展。它们弥合了人类自然语言和复杂医学领域知识或专家手动注释之间的差距，处理了医学环境中的大规模、多样化、不平衡甚至无标签数据分析场景。我们重点研究的是利用语言模型和多模态范式进行医学问题回答，旨在指导研究界根据其特定的医学研究需求选择合适的机制。

    ChatGPT explores a strategic blueprint of question answering (QA) in delivering medical diagnosis, treatment recommendations, and other healthcare support. This is achieved through the increasing incorporation of medical domain data via natural language processing (NLP) and multimodal paradigms. By transitioning the distribution of text, images, videos, and other modalities from the general domain to the medical domain, these techniques have expedited the progress of medical domain question answering (MDQA). They bridge the gap between human natural language and sophisticated medical domain knowledge or expert manual annotations, handling large-scale, diverse, unbalanced, or even unlabeled data analysis scenarios in medical contexts. Central to our focus is the utilizing of language models and multimodal paradigms for medical question answering, aiming to guide the research community in selecting appropriate mechanisms for their specific medical research requirements. Specialized tasks
    
[^55]: 通过扩展文本阅读理解提高领域适应性

    Improving Domain Adaptation through Extended-Text Reading Comprehension. (arXiv:2401.07284v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07284](http://arxiv.org/abs/2401.07284)

    通过扩展文本阅读理解，结合领域知识和聚类，以及参数微调的方法，可以显著提高领域适应性。

    

    为了增强大型语言模型的领域特定能力，对领域特定语料库进行持续预训练是一种流行的方法。最近的研究表明，使用基于正则表达式模式格式化的阅读理解数据来调整模型可以显著提高领域特定任务的性能。然而，基于正则表达式模式无法使用领域特定知识解析原始语料库。此外，问题和答案对是直接从语料库中以预定义的格式提取的，提供了有限的上下文。为解决这一限制，我们通过LLM和聚类改进了阅读理解。LLM专注于利用语料库中的领域知识来优化理解阶段，而聚类通过扩展上下文来丰富阅读阶段提供相关知识。此外，我们的方法还结合了高效参数微调来提高领域适应的效率。与AdaptLLM相比，我们的方法取得了改进

    To enhance the domain-specific capabilities of large language models, continued pre-training on a domain-specific corpus is a prevalent method. Recent work demonstrates that adapting models using reading comprehension data formatted by regex-based patterns can significantly improve performance on domain-specific tasks. However, regex-based patterns are incapable of parsing raw corpora using domain-specific knowledge. Furthermore, the question and answer pairs are extracted directly from the corpus in predefined formats offers limited context. To address this limitation, we improve reading comprehension via LLM and clustering. LLM focuses on leveraging domain knowledge within the corpus to refine comprehension stage, while clustering supplies relevant knowledge by extending the context to enrich reading stage. Additionally, our method incorporates parameter-efficient fine-tuning to improve the efficiency of domain adaptation. In comparison to AdaptLLM, our method achieves an improvement
    
[^56]: E^2-LLM: 大规模语言模型的高效和极长扩展

    E^2-LLM: Efficient and Extreme Length Extension of Large Language Models. (arXiv:2401.06951v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.06951](http://arxiv.org/abs/2401.06951)

    E^2-LLM是一种高效和极长扩展方法，通过仅需一次训练过程和不收集长上下文数据的方式，在大规模语言模型中实现了显著减少的计算成本。基于RoPE位置嵌入，E^2-LLM只需要较短的训练数据长度，支持不同的评估上下文窗口。

    

    通常，使用长上下文大小训练LLM会消耗大量的计算资源和GPU资源，需要长时间的训练。现有的长上下文扩展方法通常需要额外的训练过程来支持相应的长上下文窗口，需要长上下文训练数据（例如32k），并且假定有高昂的GPU训练成本。为了解决上述问题，我们提出了一种名为E^2-LLM的高效和极长扩展方法，只需要一次训练过程，大大减少了计算成本，也不需要收集长上下文数据。具体而言，我们的E^2-LLM的训练数据只需要很短的长度（例如4k），大大降低了调整成本。其次，在短训练上下文窗口上的训练过程只执行一次，我们可以支持不同的评估上下文窗口。第三，在E^2-LLM中，我们基于RoPE位置嵌入。

    Typically, training LLMs with long context sizes is computationally expensive, requiring extensive training hours and GPU resources. Existing long-context extension methods usually need additional training procedures to support corresponding long-context windows, where the long-context training data (e.g., 32k) is needed, and high GPU training costs are assumed. To address the aforementioned issues, we propose an Efficient and Extreme length extension method for Large Language Models, called E 2 -LLM, with only one training procedure and dramatically reduced computation cost, which also removes the need to collect long-context data. Concretely, first, the training data of our E 2 -LLM only requires a short length (e.g., 4k), which reduces the tuning cost greatly. Second, the training procedure on the short training context window is performed only once time, and we can support different evaluation context windows at inference. Third, in E 2 - LLM, based on RoPE position embeddings, we 
    
[^57]: 探索多模态大语言模型（MLLMs）的推理能力：关于多模态推理新趋势的综合调查

    Exploring the Reasoning Abilities of Multimodal Large Language Models (MLLMs): A Comprehensive Survey on Emerging Trends in Multimodal Reasoning. (arXiv:2401.06805v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.06805](http://arxiv.org/abs/2401.06805)

    这篇综述调查了多模态大语言模型（MLLMs）的推理能力，包括评估协议、模型前沿和推理密集型任务的应用，旨在实现强人工智能（Strong AI）或人工通用智能（AGI）的抽象推理能力。

    

    强人工智能（Strong AI）或人工通用智能（AGI）具备抽象推理能力是下一代人工智能的目标。近年来，大型语言模型（LLMs）以及新兴的多模态大语言模型（MLLMs）领域展示出了令人印象深刻的跨界性能和应用潜力。特别是，不同的MLLMs通过不同的模型架构、训练数据和训练阶段进行了广泛的MLLM基准评估。这些研究在不同程度上揭示了MLLMs当前的能力。然而，MLLMs的推理能力还没有得到系统的调查。在本调查中，我们全面回顾了现有的多模态推理评估协议，对MLLMs的前沿进行分类和揭示，介绍了MLLMs在推理密集型任务上的最新趋势，并最终讨论了当前的实践

    Strong Artificial Intelligence (Strong AI) or Artificial General Intelligence (AGI) with abstract reasoning ability is the goal of next-generation AI. Recent advancements in Large Language Models (LLMs), along with the emerging field of Multimodal Large Language Models (MLLMs), have demonstrated impressive capabilities across a wide range of multimodal tasks and applications. Particularly, various MLLMs, each with distinct model architectures, training data, and training stages, have been evaluated across a broad range of MLLM benchmarks. These studies have, to varying degrees, revealed different aspects of the current capabilities of MLLMs. However, the reasoning abilities of MLLMs have not been systematically investigated. In this survey, we comprehensively review the existing evaluation protocols of multimodal reasoning, categorize and illustrate the frontiers of MLLMs, introduce recent trends in applications of MLLMs on reasoning-intensive tasks, and finally discuss current practic
    
[^58]: 卧底特工：训练骗人的LLM以通过安全训练

    Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. (arXiv:2401.05566v1 [cs.CR])

    [http://arxiv.org/abs/2401.05566](http://arxiv.org/abs/2401.05566)

    该论文研究了在大型语言模型中训练并保持持久的欺骗性行为，这种行为无法被当前的安全训练技术移除。

    

    人类有能力进行战略性的欺骗行为：在大多数情况下表现出有益的行为，但在有机会的时候却表现出截然不同的行为以追求其他目标。如果一个AI系统学会了这样的欺骗策略，是否能够通过当前最先进的安全训练技术检测并移除它？为了研究这个问题，我们构建了大型语言模型（LLM）中欺骗行为的概念验证样例。例如，我们训练模型，在提示语句中将年份设为2023时编写安全代码，但在年份设为2024时插入有漏洞的代码。我们发现，这种暗门行为可以被持续保留，无法通过标准的安全训练技术（包括监督微调、强化学习和对抗性训练）移除。暗门行为在最大的模型和训练成产生思维链的模型中最为持久。

    Humans are capable of strategically deceptive behavior: behaving helpfully in most situations, but then behaving very differently in order to pursue alternative objectives when given the opportunity. If an AI system learned such a deceptive strategy, could we detect it and remove it using current state-of-the-art safety training techniques? To study this question, we construct proof-of-concept examples of deceptive behavior in large language models (LLMs). For example, we train models that write secure code when the prompt states that the year is 2023, but insert exploitable code when the stated year is 2024. We find that such backdoored behavior can be made persistent, so that it is not removed by standard safety training techniques, including supervised fine-tuning, reinforcement learning, and adversarial training (eliciting unsafe behavior and then training to remove it). The backdoored behavior is most persistent in the largest models and in models trained to produce chain-of-thoug
    
[^59]: 真实森林：通过干预而无需调整，实现大型语言模型的多尺度真实性

    Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning. (arXiv:2312.17484v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.17484](http://arxiv.org/abs/2312.17484)

    该论文提出了一种名为真实森林的方法，通过使用多维正交探针，揭示隐藏的真实表示，从而增强大型语言模型中的真实性。作者将正交约束融入探针，创建不同的正交基，通过随机窥视技术，减小了模型生成和识别真实特征之间的差距。实验证明，该方法显著提高了模型的真实性。

    

    尽管大型语言模型（LLM）在各种任务中取得了巨大成功，但它们存在生成幻觉的问题。我们引入了真实森林，一种通过使用多维正交探针揭示LLM中隐藏的真实表示来增强真实性的方法。具体而言，它通过将正交约束融入探针中来创建多个用于建模真实性的正交基。此外，我们引入了随机窥视，这是一种系统技术，考虑了序列中更广泛的位置范围，减小了LLM中辨别和生成真实特征之间的差距。通过采用这种方法，在TruthfulQA上将Llama-2-7B的真实性从40.8％提高到74.5％。类似地，在微调模型中也观察到了显著的改进。我们对探针使用了彻底的真实特征分析。我们的可视化结果显示，正交探针捕捉到互补的与真实相关的特征，形成了清晰定义的聚类，揭示了内在的真实性

    Despite the great success of large language models (LLMs) in various tasks, they suffer from generating hallucinations. We introduce Truth Forest, a method that enhances truthfulness in LLMs by uncovering hidden truth representations using multi-dimensional orthogonal probes. Specifically, it creates multiple orthogonal bases for modeling truth by incorporating orthogonal constraints into the probes. Moreover, we introduce Random Peek, a systematic technique considering an extended range of positions within the sequence, reducing the gap between discerning and generating truth features in LLMs. By employing this approach, we improved the truthfulness of Llama-2-7B from 40.8\% to 74.5\% on TruthfulQA. Likewise, significant improvements are observed in fine-tuned models. We conducted a thorough analysis of truth features using probes. Our visualization results show that orthogonal probes capture complementary truth-related features, forming well-defined clusters that reveal the inherent 
    
[^60]: 仅需规范指令：对LLaMA-1/2、GPT-3.5/4进行疑问的原则

    Principled Instructions Are All You Need for Questioning LLaMA-1/2, GPT-3.5/4. (arXiv:2312.16171v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.16171](http://arxiv.org/abs/2312.16171)

    本文提出了26个指导原则，以简化对大型语言模型进行提问和提示的过程。通过在LLaMA-1/2和GPT-3.5/4上进行实验证明了这些原则的有效性。

    

    本文介绍了26个指导原则，旨在简化对大型语言模型进行提问和提示的过程。我们的目标是简化针对不同规模的大型语言模型制定问题的基本概念，检查其能力，并提供不同提示时涉及的不同规模的大型语言模型的用户理解。我们在LLaMA-1/2 (7B, 13B和70B)、GPT-3.5/4上进行了大量实验，以验证所提出的指导原则在指令和提示设计上的有效性。我们希望这项工作能为从事大型语言模型提示研究的研究人员提供更好的指导。项目页面位于https://github.com/VILA-Lab/ATLAS。

    This paper introduces 26 guiding principles designed to streamline the process of querying and prompting large language models. Our goal is to simplify the underlying concepts of formulating questions for various scales of large language models, examining their abilities, and enhancing user comprehension on the behaviors of different scales of large language models when feeding into different prompts. Extensive experiments are conducted on LLaMA-1/2 (7B, 13B and 70B), GPT-3.5/4 to verify the effectiveness of the proposed principles on instructions and prompts design. We hope that this work can provide a better guide for researchers working on the prompting of large language models. Project page is available at https://github.com/VILA-Lab/ATLAS.
    
[^61]: 逻辑搭建：使用LLMs进行个性化的面向指导的推荐解释生成

    Logic-Scaffolding: Personalized Aspect-Instructed Recommendation Explanation Generation using LLMs. (arXiv:2312.14345v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.14345](http://arxiv.org/abs/2312.14345)

    本研究提出了一个框架称为逻辑搭建，通过结合面向方面的解释和思维链提示的思想，在中间推理步骤中生成推荐解释。该框架能够克服现有模型在产生零炮击解释方面的困难。

    

    大型语言模型（LLMs）的独特能力，如自然语言文本生成能力，使它们成为提供推荐解释的强有力候选者。然而，尽管LLM的规模很大，但大多数现有模型在可靠地产生零炮击解释方面仍存在困难。为了解决这个问题，我们提出了一个名为逻辑搭建的框架，将面向方面的解释和思维链提示的思想结合起来，通过中间推理步骤生成解释。在本文中，我们分享了构建该框架的经验，并提供了一个交互式演示来探索我们的结果。

    The unique capabilities of Large Language Models (LLMs), such as the natural language text generation ability, position them as strong candidates for providing explanation for recommendations. However, despite the size of the LLM, most existing models struggle to produce zero-shot explanations reliably. To address this issue, we propose a framework called Logic-Scaffolding, that combines the ideas of aspect-based explanation and chain-of-thought prompting to generate explanations through intermediate reasoning steps. In this paper, we share our experience in building the framework and present an interactive demonstration for exploring our results.
    
[^62]: LLM-SQL-Solver: LLM能够确定SQL等价关系吗？

    LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?. (arXiv:2312.10321v2 [cs.DB] UPDATED)

    [http://arxiv.org/abs/2312.10321](http://arxiv.org/abs/2312.10321)

    本研究探讨了LLM是否能够确定两个SQL查询的等价关系，并提出了两种提示技术来帮助LLM生成高质量的响应。

    

    判断两个SQL查询之间的等价关系是数据管理和SQL生成中的一个基本问题，具有许多实际应用（即，在文本到SQL任务中评估生成的SQL查询的质量）。虽然研究界多年来一直在考虑SQL的等价性，但它存在相当大的困难，并且没有完整的解决方案。最近，大型语言模型（LLMs）在对话、问答和解决数学问题方面展现出强大的推理能力。在本文中，我们研究了LLMs是否可以用于确定两个SQL查询的等价性（语义等价和宽松等价）。为了帮助LLMs生成高质量的响应，我们提出了两种提示技术：Miniature & Mull和Explain & Compare。前一种技术被用于评估语义等价性，它要求LLMs在简单的数据库实例上执行查询，然后探索是否存在反例。

    Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample ex
    
[^63]: 强化断言的少样本学习：用于大型语言模型生成教育解释的教导技术。

    Assertion Enhanced Few-Shot Learning: Instructive Technique for Large Language Models to Generate Educational Explanations. (arXiv:2312.03122v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.03122](http://arxiv.org/abs/2312.03122)

    本研究提出了一种强化断言的少样本学习技术，用于大型语言模型生成精确、详细的教育解释。实验结果显示，该方法在解释准确性上提升了15%，获得了教师评估为高质量的解释。

    

    人类教育者具备从学生中预测并寻求教育解释的内在能力，当学生无法独立表达这些解释时，他们能够提出发人深省的问题。我们的目标是利用大型语言模型的少样本学习能力为智能辅导系统赋予这种能力。我们的工作提出了一种新颖的教导技术，即强化断言的少样本学习，以促进准确、详细的教育解释的生成。我们的核心假设是，在教育领域，少样本演示是必要但不足以保证高质量的解释生成的条件。我们进行了一项涉及12名在职教师的研究，将我们的方法与传统的少样本学习进行了比较。结果显示，强化断言的少样本学习将解释准确性提高了15%，并得到了教师评估为高质量的解释。我们还进行了定性的剔除研究。

    Human educators possess an intrinsic ability to anticipate and seek educational explanations from students, which drives them to pose thought-provoking questions when students cannot articulate these explanations independently. We aim to imbue Intelligent Tutoring Systems with this ability using few-shot learning capability of Large Language Models. Our work proposes a novel prompting technique, Assertion Enhanced Few-Shot Learning, to facilitate the generation of accurate, detailed oriented educational explanations. Our central hypothesis is that, in educational domain, few-shot demonstrations are necessary but not a sufficient condition for quality explanation generation. We conducted a study involving 12 in-service teachers, comparing our approach to Traditional Few-Shot Learning. The results show that Assertion Enhanced Few-Shot Learning improves explanation accuracy by 15% and yields higher-quality explanations, as evaluated by teachers. We also conduct a qualitative ablation stud
    
[^64]: 大型语言模型增强的算法选择：朝着全面算法表示的方向

    Large Language Model-Enhanced Algorithm Selection: Towards Comprehensive Algorithm Representation. (arXiv:2311.13184v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.13184](http://arxiv.org/abs/2311.13184)

    本论文提出了一种方法，通过将算法表示集成到算法选择中，从而填补了当前算法选择技术对算法特征的研究空白。

    

    算法选择旨在在执行之前识别解决特定问题的最合适算法，已成为自动机器学习中的关键过程。当前主流的算法选择技术主要依赖于各种问题的特征表示，并使用每个算法的性能作为监督信息。然而，目前对算法特征的考虑存在重要的研究空白。这主要归因于算法的固有复杂性，使得在不同种类的算法中找到一种普适有效的特征提取方法特别具有挑战性。不幸的是，忽视了这一方面无疑会影响算法选择的准确性，并间接需要增加训练数据的数量。本文提出了一种方法来解决这一空白，即将算法表示集成到算法选择中。

    Algorithm selection aims to identify the most suitable algorithm for solving a specific problem before execution, which has become a critical process of the AutoML. Current mainstream algorithm selection techniques rely heavily on feature representations of various problems and employ the performance of each algorithm as supervised information. However, there is a significant research gap concerning the consideration of algorithm features. This gap is primarily attributed to the inherent complexity of algorithms, making it particularly challenging to find a universally effective feature extraction method that is applicable across a diverse range of algorithms. Unfortunately, neglecting this aspect undoubtedly impacts the accuracy of algorithm selection and indirectly necessitates an increased volume of problem data for training purposes. This paper takes a significant stride towards addressing this gap by proposing an approach that integrates algorithm representation into the algorithm
    
[^65]: 通过模型适应来去除偏见算法

    Debiasing Algorithm through Model Adaptation. (arXiv:2310.18913v1 [cs.CL])

    [http://arxiv.org/abs/2310.18913](http://arxiv.org/abs/2310.18913)

    本论文提出了一种通过模型适应来检测和减轻语言模型中性别偏见的方法，并证明了该方法能够显著减少偏见同时保持模型性能。

    

    大型语言模型正在成为各种语言任务的首选解决方案。然而，随着容量的增长，模型很容易依赖训练数据中存在的偏见和刻板印象所产生的虚假相关性。本研究提出了一种新颖的方法来检测和减轻语言模型中的性别偏见。我们进行因果分析，以识别问题模型组件，并发现中上层前馈层最容易传递偏见。根据分析结果，我们通过线性投影将这些层乘以模型进行适应。我们的方法DAMA通过各种度量指标明显减少了偏见，同时保持模型在后续任务中的性能。我们发布了我们的方法和模型的代码，通过重新训练，保持了LLaMA的最先进性能，同时偏见显著减少。

    Large language models are becoming the go-to solution for various language tasks. However, with growing capacity, models are prone to rely on spurious correlations stemming from biases and stereotypes present in the training data. This work proposes a novel method for detecting and mitigating gender bias in language models. We perform causal analysis to identify problematic model components and discover that mid-upper feed-forward layers are most prone to convey biases. Based on the analysis results, we adapt the model by multiplying these layers by a linear projection. Our titular method, DAMA, significantly decreases bias as measured by diverse metrics while maintaining the model's performance on downstream tasks. We release code for our method and models, which retrain LLaMA's state-of-the-art performance while being significantly less biased.
    
[^66]: SpecTr: 通过最优传输实现快速具有推测性的解码

    SpecTr: Fast Speculative Decoding via Optimal Transport. (arXiv:2310.15141v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.15141](http://arxiv.org/abs/2310.15141)

    本研究通过最优传输的方法提供了推测性解码的原则性理解，使得从大语言模型中进行自回归采样的过程能够更快速地进行。

    

    从大语言模型中进行自回归采样在多个自然语言任务中取得了最先进的结果。然而，自回归采样一次只生成一个标记，这使得速度慢，在某些任务中甚至是禁止的。加速采样的一种方式是“推测性解码”：使用一个小模型来采样一个“草稿”（块或标记序列），然后由大语言模型并行评分草稿中的所有标记。根据统计方法，接受一部分草稿中的标记（拒绝剩余标记），以确保最终输出遵循大模型的分布。在这项工作中，我们通过最优传输（OT）与“成员费用”的视角提供了推测性解码的原则性理解。这个框架可以被看作是“最大耦合”问题的扩展。这种新的形式使我们能够将推测性解码方法推广到允许一个集合的标记的情况。

    Autoregressive sampling from large language models has led to state-of-the-art results in several natural language tasks. However, autoregressive sampling generates tokens one at a time making it slow, and even prohibitive in certain tasks. One way to speed up sampling is $\textit{speculative decoding}$: use a small model to sample a $\textit{draft}$ (block or sequence of tokens), and then score all tokens in the draft by the large language model in parallel. A subset of the tokens in the draft are accepted (and the rest rejected) based on a statistical method to guarantee that the final output follows the distribution of the large model. In this work, we provide a principled understanding of speculative decoding through the lens of optimal transport (OT) with $\textit{membership cost}$. This framework can be viewed as an extension of the well-known $\textit{maximal-coupling}$ problem. This new formulation enables us to generalize the speculative decoding method to allow for a set of $
    
[^67]: MolCA: 通过跨模态投影和单模态适配器的分子图-语言建模

    MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter. (arXiv:2310.12798v1 [cs.CL])

    [http://arxiv.org/abs/2310.12798](http://arxiv.org/abs/2310.12798)

    MolCA是一个可以通过跨模态投影和单模态适配器实现分子图和语言的建模系统。它可以通过连接图编码器和语言模型的表示空间来理解文本和图形的分子内容，并通过单模态适配器在下游任务中高效适应。

    

    语言模型在各种与文本相关的任务上展示了对分子的卓越理解能力。然而，它们本质上缺乏人类专业人员在理解分子拓扑结构中的关键能力 - 2D图形感知能力。为了弥合这个差距，我们提出了MolCA: 通过跨模态投影和单模态适配器进行分子图-语言建模。MolCA通过跨模态投影使语言模型（例如Galactica）能够理解基于文本和图形的分子内容。具体而言，跨模态投影器被实现为一个Q-Former，连接一个图编码器的表示空间和一个语言模型的文本空间。此外，MolCA使用单模态适配器（即LoRA）使语言模型能够有效适应下游任务。与先前的研究通过跨模态对比学习将语言模型与图形编码器耦合不同，MolCA保留了语言模型的开放式文本生成能力，并增加了2D图形信息。为了展示其有效性，

    Language Models (LMs) have demonstrated impressive molecule understanding ability on various 1D text-related tasks. However, they inherently lack 2D graph perception - a critical ability of human professionals in comprehending molecules' topological structures. To bridge this gap, we propose MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter. MolCA enables an LM (e.g., Galactica) to understand both text- and graph-based molecular contents via the cross-modal projector. Specifically, the cross-modal projector is implemented as a Q-Former to connect a graph encoder's representation space and an LM's text space. Further, MolCA employs a uni-modal adapter (i.e., LoRA) for the LM's efficient adaptation to downstream tasks. Unlike previous studies that couple an LM with a graph encoder via cross-modal contrastive learning, MolCA retains the LM's ability of open-ended text generation and augments it with 2D graph information. To showcase its effectivenes
    
[^68]: 发现塞壬之歌：可靠的事实冲突幻觉检测

    Unveiling the Siren's Song: Towards Reliable Fact-Conflicting Hallucination Detection. (arXiv:2310.12086v1 [cs.CL])

    [http://arxiv.org/abs/2310.12086](http://arxiv.org/abs/2310.12086)

    该论文介绍了一种为大型语言模型设计的FactCHD事实冲突幻觉检测基准，用于评估LLMs生成文本的事实性。基准包含了多种事实模式，并使用基于事实的证据链进行组合性幻觉的检测。

    

    大型语言模型（LLMs），如ChatGPT/GPT-4，因其广泛的实际应用而受到广泛关注，但其在网络平台上存在事实冲突幻觉的问题限制了其采用。对由LLMs产生的文本的事实性评估仍然未被充分探索，不仅涉及对基本事实的判断，还包括对复杂推理任务（如多跳等）中出现的事实错误的评估。为此，我们引入了FactCHD，一种为LLMs精心设计的事实冲突幻觉检测基准。作为在“查询-响应”上下文中评估事实性的关键工具，我们的基准采用了大规模数据集，涵盖了广泛的事实模式，如基本事实，多跳，比较和集合操作模式。我们基准的一个独特特点是其包含基于事实的证据链，从而便于进行组合性幻觉的检测。

    Large Language Models (LLMs), such as ChatGPT/GPT-4, have garnered widespread attention owing to their myriad of practical applications, yet their adoption has been constrained by issues of fact-conflicting hallucinations across web platforms. The assessment of factuality in text, produced by LLMs, remains inadequately explored, extending not only to the judgment of vanilla facts but also encompassing the evaluation of factual errors emerging in complex inferential tasks like multi-hop, and etc. In response, we introduce FactCHD, a fact-conflicting hallucination detection benchmark meticulously designed for LLMs. Functioning as a pivotal tool in evaluating factuality within "Query-Respons" contexts, our benchmark assimilates a large-scale dataset, encapsulating a broad spectrum of factuality patterns, such as vanilla, multi-hops, comparison, and set-operation patterns. A distinctive feature of our benchmark is its incorporation of fact-based chains of evidence, thereby facilitating com
    
[^69]: 大型Transformer水印的功能不变性

    Functional Invariants to Watermark Large Transformers. (arXiv:2310.11446v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2310.11446](http://arxiv.org/abs/2310.11446)

    本文介绍了一种用于大型Transformer的功能不变性水印技术，它使用模型的不变性生成功能上等效的副本，并能在不改变模型输出的情况下给模型加上水印，这是一种计算成本极低且适用于实际应用的解决方案。

    

    基于Transformer的模型的快速增长增加了对其完整性和拥有权的担忧。水印技术通过将唯一标识嵌入模型中来解决这个问题，同时保持其性能。然而，大多数现有方法需要优化权重以嵌入水印信号，这在大规模情况下不适用于计算成本的原因。本文探讨了一种几乎没有计算成本且适用于非盲白盒设置（假设可以访问原始和带水印的网络）的水印技术。他们通过利用模型的不变性，比如维度排列或缩放/非缩放等操作，生成功能上等效的副本。这使得可以在不改变模型输出的情况下给模型加水印，并保持不可察觉性。实验证明了该方法的有效性以及对各种模型变换（微调、量化、修剪）的稳健性，使其成为实际解决方案。

    The rapid growth of transformer-based models increases the concerns about their integrity and ownership insurance. Watermarking addresses this issue by embedding a unique identifier into the model, while preserving its performance. However, most existing approaches require to optimize the weights to imprint the watermark signal, which is not suitable at scale due to the computational cost. This paper explores watermarks with virtually no computational cost, applicable to a non-blind white-box setting (assuming access to both the original and watermarked networks). They generate functionally equivalent copies by leveraging the models' invariance, via operations like dimension permutations or scaling/unscaling. This enables to watermark models without any change in their outputs and remains stealthy. Experiments demonstrate the effectiveness of the approach and its robustness against various model transformations (fine-tuning, quantization, pruning), making it a practical solution to pro
    
[^70]: Transformer语言模型中跨任务的电路组件复用

    Circuit Component Reuse Across Tasks in Transformer Language Models. (arXiv:2310.08744v1 [cs.CL])

    [http://arxiv.org/abs/2310.08744](http://arxiv.org/abs/2310.08744)

    这项工作证明了在Transformer语言模型中，电路组件可以在不同任务之间复用并产生相似的功能，为更高级的模型理解做出贡献。

    

    最近在机制可解释性方面的研究表明，通过电路分析可以成功地逆向工程语言模型的行为。然而，一个常见的批评是每个电路都是任务特定的，因此这样的分析不能为更高级的理解模型做出贡献。在这项工作中，我们提出证据表明洞察力（关于特定头部的低级发现和关于一般算法的高级发现）确实可以在任务之间进行泛化。具体而言，我们研究了Wang等人（2022）在间接宾语识别任务（IOI）中发现的电路，并展示了这个电路在更大的GPT2模型上的重现，以及在看似不同的任务中大部分被复用来解决问题：彩色物体（Ippolito和Callison-Burch，2023）。我们提供证据表明两个任务底层的过程在功能上非常相似，并且在电路中的注意力头部之间有大约78％的重叠。我们进一步展示了一个概念验证干预实验

    Recent work in mechanistic interpretability has shown that behaviors in language models can be successfully reverse-engineered through circuit analysis. A common criticism, however, is that each circuit is task-specific, and thus such analysis cannot contribute to understanding the models at a higher level. In this work, we present evidence that insights (both low-level findings about specific heads and higher-level findings about general algorithms) can indeed generalize across tasks. Specifically, we study the circuit discovered in Wang et al. (2022) for the Indirect Object Identification (IOI) task and 1.) show that it reproduces on a larger GPT2 model, and 2.) that it is mostly reused to solve a seemingly different task: Colored Objects (Ippolito & Callison-Burch, 2023). We provide evidence that the process underlying both tasks is functionally very similar, and contains about a 78% overlap in in-circuit attention heads. We further present a proof-of-concept intervention experiment
    
[^71]: 理解在线虚假信息背后的人类行为：以COVID-19大流行为镜像的观察研究

    Understanding the Humans Behind Online Misinformation: An Observational Study Through the Lens of the COVID-19 Pandemic. (arXiv:2310.08483v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.08483](http://arxiv.org/abs/2310.08483)

    本研究通过观察分析了3200万条COVID-19推文和1600万条历史时间线推文，重点研究了COVID-19期间传播虚假信息用户的行为和心理，并将其与在疫情前在非COVID领域分享虚假信息的历史倾向联系起来。

    

    在线虚假信息的蔓延已经成为社会面临的最大威胁之一。虽然已经付出了相当多的努力来构建虚假信息检测模型，但是虚假信息的危害仍然存在。应对在线虚假信息及其后果需要一个全面的方法，不仅包括对于在线复杂问题和丰富主题信息生态系统与虚假信息之间的复杂关系的理解，还需要了解在其背后驱动虚假信息传播的个体的心理动因。采用时间序列分析技术和强大的因果推断设计，我们进行了一项大规模的观测性研究，分析了超过3200万条COVID-19推文和1600万条历史时间线推文。我们重点研究了COVID-19期间传播虚假信息用户的行为和心理，并将其与在疫情前在非COVID领域分享虚假信息的历史倾向联系起来。我们的分析强调了这种行为的创新性和贡献。

    The proliferation of online misinformation has emerged as one of the biggest threats to society. Considerable efforts have focused on building misinformation detection models, still the perils of misinformation remain abound. Mitigating online misinformation and its ramifications requires a holistic approach that encompasses not only an understanding of its intricate landscape in relation to the complex issue and topic-rich information ecosystem online, but also the psychological drivers of individuals behind it. Adopting a time series analytic technique and robust causal inference-based design, we conduct a large-scale observational study analyzing over 32 million COVID-19 tweets and 16 million historical timeline tweets. We focus on understanding the behavior and psychology of users disseminating misinformation during COVID-19 and its relationship with the historical inclinations towards sharing misinformation on Non-COVID domains before the pandemic. Our analysis underscores the int
    
[^72]: MetaTool基准：决定是否使用工具和选择使用哪个工具。

    MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use. (arXiv:2310.03128v1 [cs.SE])

    [http://arxiv.org/abs/2310.03128](http://arxiv.org/abs/2310.03128)

    本文提出了一个名为MetaTool的基准，旨在评估大型语言模型（LLMs）是否具有工具使用意识并且能够正确选择工具。基准中包含一个名为ToolE的数据集，其中包含各种类型的用户查询，用于触发LLMs使用工具。

    

    大型语言模型（LLMs）由于其出色的自然语言处理（NLP）能力而受到了广泛关注。最近，许多研究关注LLMs的工具利用能力。它们主要研究了LLMs如何有效地与给定的特定工具合作。然而，在LLMs充当智能体的场景中，例如AutoGPT和MetaGPT应用中，LLMs被期望参与涉及是否使用工具以及从可用工具集中选择最合适的工具来满足用户请求的复杂决策过程。因此，在本文中，我们介绍了MetaTool，这是一个用于评估LLMs是否具有工具使用意识并且能够正确选择工具的基准。具体而言，我们在该基准中创建了一个名为ToolE的数据集。该数据集包含以触发LLMs使用工具的提示形式出现的各种类型的用户查询，包括单一工具和多种工具。

    Large language models (LLMs) have garnered significant attention due to their impressive natural language processing (NLP) capabilities. Recently, many studies have focused on the tool utilization ability of LLMs. They primarily investigated how LLMs effectively collaborate with given specific tools. However, in scenarios where LLMs serve as intelligent agents, as seen in applications like AutoGPT and MetaGPT, LLMs are expected to engage in intricate decision-making processes that involve deciding whether to employ a tool and selecting the most suitable tool(s) from a collection of available tools to fulfill user requests. Therefore, in this paper, we introduce MetaTool, a benchmark designed to evaluate whether LLMs have tool usage awareness and can correctly choose tools. Specifically, we create a dataset called ToolE within the benchmark. This dataset contains various types of user queries in the form of prompts that trigger LLMs to use tools, including both single-tool and multi-too
    
[^73]: 长文本摘要评估中“少即是多”的理论

    Less is More for Long Document Summary Evaluation by LLMs. (arXiv:2309.07382v1 [cs.CL])

    [http://arxiv.org/abs/2309.07382](http://arxiv.org/abs/2309.07382)

    该论文引入了一种新颖的方法，通过先提取关键句子再进行评估，有效解决了大型语言模型在长文档摘要评估中遇到的计算成本高和忽视重要信息的问题。研究发现，这种方法不仅显著降低了评估成本，而且与人工评估有更高的相关性。此外，论文还提供了关于最佳文档长度和句子提取方法的实用建议，为基于大型语言模型的文本生成评估的发展做出了贡献。

    

    大型语言模型(LLMs)在摘要评估任务中表现出了令人期待的性能，但它们面临诸如高计算成本和长文档中重要信息被忽视的“迷失在中间”问题。为解决这些问题，本文引入了一种新颖的方法，即“先提取再评估”，该方法涉及从长文本源文件中提取关键句子，然后通过提问LLMs来评估摘要。结果表明，所提出的方法不仅显著降低了评估成本，而且与人工评估之间存在更高的相关性。此外，我们提供了关于最佳文档长度和句子提取方法的实用建议，为基于LLMs的文本生成评估的开发提供了成本效益更高且更准确的方法。

    Large Language Models (LLMs) have shown promising performance in summary evaluation tasks, yet they face challenges such as high computational costs and the Lost-in-the-Middle problem where important information in the middle of long documents is often overlooked. To address these issues, this paper introduces a novel approach, Extract-then-Evaluate, which involves extracting key sentences from a long source document and then evaluating the summary by prompting LLMs. The results reveal that the proposed method not only significantly reduces evaluation costs but also exhibits a higher correlation with human evaluations. Furthermore, we provide practical recommendations for optimal document length and sentence extraction methods, contributing to the development of cost-effective yet more accurate methods for LLM-based text generation evaluation.
    
[^74]: 全景视觉-语言特征场

    Panoptic Vision-Language Feature Fields. (arXiv:2309.05448v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2309.05448](http://arxiv.org/abs/2309.05448)

    本文提出了一种用于3D场景中开放词汇全景分割的算法PVLFF，通过从预训练的2D模型中提取视觉-语言特征来学习语义特征场，通过对输入帧上的2D实例分割进行对比学习来联合拟合实例特征场。该方法在全景分割和语义分割方面具有良好的性能。

    

    最近，出现了一些用于3D开放词汇语义分割的方法。这些方法能够根据运行时提供的文本描述将场景分割成任意类别。在本文中，我们提出了迄今为止首个用于3D场景中开放词汇全景分割的算法。我们的算法Panoptic Vision-Language Feature Fields (PVLFF)通过从预训练的2D模型中提取视觉-语言特征来学习场景的语义特征场，并通过在输入帧上使用2D实例分割实现对实例特征场的联合拟合。尽管没有针对目标类别进行训练，我们的方法在HyperSim、ScanNet和Replica数据集上实现了与最先进的闭集3D系统相似的全景分割性能，并且在语义分割方面优于当前的3D开放词汇系统。我们对我们方法的组成部分进行了实验来证明其有效性。

    Recently, methods have been proposed for 3D open-vocabulary semantic segmentation. Such methods are able to segment scenes into arbitrary classes based on text descriptions provided during runtime. In this paper, we propose to the best of our knowledge the first algorithm for open-vocabulary panoptic segmentation in 3D scenes. Our algorithm, Panoptic Vision-Language Feature Fields (PVLFF), learns a semantic feature field of the scene by distilling vision-language features from a pretrained 2D model, and jointly fits an instance feature field through contrastive learning using 2D instance segments on input frames. Despite not being trained on the target classes, our method achieves panoptic segmentation performance similar to the state-of-the-art closed-set 3D systems on the HyperSim, ScanNet and Replica dataset and additionally outperforms current 3D open-vocabulary systems in terms of semantic segmentation. We ablate the components of our method to demonstrate the effectiveness of our
    
[^75]: 探索大语言模型用于代码生成的参数高效微调技术

    Exploring Parameter-Efficient Fine-Tuning Techniques for Code Generation with Large Language Models. (arXiv:2308.10462v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2308.10462](http://arxiv.org/abs/2308.10462)

    本文探索了大型语言模型在资源有限的环境下用于代码生成的参数高效微调技术，并提出了参数高效微调作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将语言模型专门用于任务特定的数据。

    

    大型语言模型（LLM）展示了在没有特定微调的情况下，即可根据自然语言意图生成准确的代码片段的印象能力。尽管先前的研究已经突出了微调LLMs的优势，但这个过程代价高，对于拥有数十亿个参数的模型来说，在资源稀缺的环境下是不切实际的。为了解决这些挑战，以前的研究探索了在上下文学习（ICL）作为一种策略，用任务特定的提示示例指导LLM生成过程。然而，ICL引入了一些不便之处，比如需要设计上下文相关的提示和没有学习任务特定的参数，从而限制了下游任务的性能。在这种情况下，我们预见参数高效微调（PEFT）技术作为一种有前途的方法，可以在保持合理资源消耗的同时，高效地将LLM专门用于任务特定的数据。

    Large Language Models (LLMs) demonstrate impressive capabilities to generate accurate code snippets given natural language intents in zero-shot, i.e., without the need for specific fine-tuning. While prior studies have highlighted the advantages of fine-tuning LLMs, this process incurs high computational costs, making it impractical in resource-scarce environments, particularly for models with billions of parameters. To address these challenges, previous research explored In-Context Learning (ICL) as a strategy to guide the LLM generative process with task-specific prompt examples. However, ICL introduces inconveniences, such as the need for designing contextually relevant prompts and the absence of learning task-specific parameters, thereby limiting downstream task performance. In this context, we foresee Parameter-Efficient Fine-Tuning (PEFT) techniques as a promising approach to efficiently specialize LLMs to task-specific data while maintaining reasonable resource consumption. In t
    
[^76]: 把高下分清楚：通过参数高效模块操作进行模型残缺性去学习

    Separate the Wheat from the Chaff: Model Deficiency Unlearning via Parameter-Efficient Module Operation. (arXiv:2308.08090v1 [cs.CL])

    [http://arxiv.org/abs/2308.08090](http://arxiv.org/abs/2308.08090)

    通过提取和消除反专家PEMs中的残缺能力来提升大规模语言模型的真实性和去毒性。

    

    大规模语言模型（LLMs）在各种应用中得到广泛应用，但存在与不真实和有毒性有关的问题。虽然参数高效模块（PEMs）已经证明了其在为模型赋予新技能方面的有效性，但利用PEMs进行残缺性去学习仍未充分探索。在这项工作中，我们提出了一种PEMs操作方法，即“提取-减去”（Ext-Sub），通过整合“专家”PEMs和“反专家”PEMs来增强LLMs的真实性和去毒性。值得注意的是，即使是反专家PEMs也具有宝贵的能力，因为它们擅长生成虚构内容，这需要语言建模和逻辑叙述能力。与仅仅否定参数不同，我们的方法涉及提取和消除反专家PEMs中的残缺能力，同时保留一般能力。为了评估我们的方法在真实性方面的有效性

    Large language models (LLMs) have been widely used in various applications but are known to suffer from issues related to untruthfulness and toxicity. While parameter-efficient modules (PEMs) have demonstrated their effectiveness in equipping models with new skills, leveraging PEMs for deficiency unlearning remains underexplored. In this work, we propose a PEMs operation approach, namely Extraction-before-Subtraction (Ext-Sub), to enhance the truthfulness and detoxification of LLMs through the integration of ``expert'' PEM and ``anti-expert'' PEM. Remarkably, even anti-expert PEM possess valuable capabilities due to their proficiency in generating fabricated content, which necessitates language modeling and logical narrative competence. Rather than merely negating the parameters, our approach involves extracting and eliminating solely the deficiency capability within anti-expert PEM while preserving the general capabilities. To evaluate the effectiveness of our approach in terms of tru
    
[^77]: LoraHub: 通过动态LoRA组合实现高效的任务通用性

    LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition. (arXiv:2307.13269v1 [cs.CL])

    [http://arxiv.org/abs/2307.13269](http://arxiv.org/abs/2307.13269)

    本文研究了LoRA组合在跨任务通用性上的可行性，并提出了LoraHub框架，能够通过组合不同任务上训练的LoRA模块，实现对未见任务的可适应性性能。实验结果表明，LoraHub在少样本场景中能够有效模拟上下文学习的性能，而无需上下文示例。

    

    低秩适应（LoRA）常常被用于对新任务进行大型语言模型（LLM）的微调。本文研究了LoRA组合在跨任务通用性上的可行性，并介绍了LoraHub，这是一个为目的性组装在不同给定任务上训练的LoRA模块的战略框架，旨在实现对未见任务的可适应性性能。仅凭借来自新任务的几个示例，LoraHub可以灵活地组合多个LoRA模块，消除了对人类专业知识的需求。值得注意的是，这种组合既不需要额外的模型参数，也不需要梯度。我们从Big-Bench Hard（BBH）基准测试中得出的实证结果表明，LoraHub在少样本场景中可以有效地模拟上下文学习的性能，在每个推理输入旁边不需要上下文示例。我们的研究的一个重要贡献是培育一个LoRA社区，用户可以在其中分享他们训练的LoRA模块。

    Low-rank adaptations (LoRA) are often employed to fine-tune large language models (LLMs) for new tasks. This paper investigates LoRA composability for cross-task generalization and introduces LoraHub, a strategic framework devised for the purposive assembly of LoRA modules trained on diverse given tasks, with the objective of achieving adaptable performance on unseen tasks. With just a few examples from a novel task, LoraHub enables the fluid combination of multiple LoRA modules, eradicating the need for human expertise. Notably, the composition requires neither additional model parameters nor gradients. Our empirical results, derived from the Big-Bench Hard (BBH) benchmark, suggest that LoraHub can effectively mimic the performance of in-context learning in few-shot scenarios, excluding the necessity of in-context examples alongside each inference input. A significant contribution of our research is the fostering of a community for LoRA, where users can share their trained LoRA module
    
[^78]: CMMLU：衡量中文大规模多任务语言理解

    CMMLU: Measuring massive multitask language understanding in Chinese. (arXiv:2306.09212v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.09212](http://arxiv.org/abs/2306.09212)

    本论文介绍了CMMLU，这是一个衡量中文大规模多任务语言理解的综合性基准测试。研究发现，大多数现有的语言模型在不同主题和设置下的性能都不够理想，存在改进空间。CMMLU填补了评估大型语言模型知识和推理能力的空白，并提出了增强LLM的方向。

    

    随着大型语言模型（LLM）的能力不断提高，评估它们的性能变得越来越关键和具有挑战性。本论文通过引入CMMLU，一个包括自然科学、社会科学、工程和人文等多个领域的综合性中文基准测试，来弥合这一差距。我们对18种先进的多语言和面向中文的LLM进行了全面评价，评估它们在不同主题和设置下的性能。结果显示，大多数现有的LLM在提供上下文示例和思维链提示时，难以达到平均50%的准确率，而随机基准线则为25%。这突显了LLM的改进空间。此外，我们进行了大量实验，以确定影响模型性能的因素，并提出了增强LLM的方向。CMMLU填补了评估大型语言模型的知识和推理能力的空白。

    As the capabilities of large language models (LLMs) continue to advance, evaluating their performance becomes increasingly crucial and challenging. This paper aims to bridge this gap by introducing CMMLU, a comprehensive Chinese benchmark that covers various subjects, including natural science, social sciences, engineering, and humanities. We conduct a thorough evaluation of 18 advanced multilingual- and Chinese-oriented LLMs, assessing their performance across different subjects and settings. The results reveal that most existing LLMs struggle to achieve an average accuracy of 50%, even when provided with in-context examples and chain-of-thought prompts, whereas the random baseline stands at 25%. This highlights significant room for improvement in LLMs. Additionally, we conduct extensive experiments to identify factors impacting the models' performance and propose directions for enhancing LLMs. CMMLU fills the gap in evaluating the knowledge and reasoning capabilities of large languag
    
[^79]: 使用音频数据检测政治辩论、演讲和访谈中值得核实的论断

    Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data. (arXiv:2306.05535v1 [cs.CL])

    [http://arxiv.org/abs/2306.05535](http://arxiv.org/abs/2306.05535)

    政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。

    

    社会的一大部分团结在相同的愿景和思想周围，具有巨大的能量。这正是政治人物希望为他们的事业所累积的。为了达到这个目标，他们有时会使用扭曲或隐藏真相的手段，无论是无意的还是有意的，这为错误信息和误导开了大门。自动检测值得核实的论断的工具将对辩论主持人、记者和事实核查组织有很大帮助。虽然以前关于检测值得核实的论断的工作重点是文本，但在这里，我们探讨了音频信号作为额外信息源的实用性。我们创建了一个新的多模态数据集（英语文本和音频），包含48小时的演讲。我们的评估结果表明，在多个演讲者的情况下，音频模态与文本结合使用比仅使用文本具有改进效果。此外，单声道音频模型可以胜过单声道文本模型。

    A large portion of society united around the same vision and ideas carries enormous energy. That is precisely what political figures would like to accumulate for their cause. With this goal in mind, they can sometimes resort to distorting or hiding the truth, unintentionally or on purpose, which opens the door for misinformation and disinformation. Tools for automatic detection of check-worthy claims would be of great help to moderators of debates, journalists, and fact-checking organizations. While previous work on detecting check-worthy claims has focused on text, here we explore the utility of the audio signal as an additional information source. We create a new multimodal dataset (text and audio in English) containing 48 hours of speech. Our evaluation results show that the audio modality together with text yields improvements over text alone in the case of multiple speakers. Moreover, an audio-only model could outperform a text-only one for a single speaker.
    
[^80]: 基于语法约束的语言模型灵活解码技术

    Flexible Grammar-Based Constrained Decoding for Language Models. (arXiv:2305.13971v1 [cs.CL])

    [http://arxiv.org/abs/2305.13971](http://arxiv.org/abs/2305.13971)

    本文提出了一种使用形式语法约束丰富解码步骤的方法，有效生成符合特定语法的复杂输出结构，同时允许任何上下文无关语法集成。实验证明该方法在四个信息提取任务上实现了最先进的性能表现。

    

    LLM在许多任务中展现出了惊人的少量样本表现，但在生成信息提取所需的复杂输出结构时仍存在困难。这个限制源于LLM在没有微调的情况下倾向于生成自由文本而不是遵循特定语法的精确结构。在本文中，我们提出在解码步骤中使用形式语法约束来丰富模型。在搜索过程中，只有符合语法产生规则的有效令牌能被考虑到。这样就强制只产生有效的序列。我们的框架非常通用和灵活，允许任何上下文无关语法(CFG)集成到我们的自定义约束beam搜索实现中。我们展示了许多NLP任务的输出可以被表示为形式语言，使它们适合在我们的框架中直接使用。对于输出空间取决于输入的任务，我们提出了基于输入的CFG，根据特定于输入的特征更新产生规则。实验证明了我们的方法在生成复杂输出结构方面的有效性，并在四个信息提取任务上实现了最先进的性能。

    LLMs have shown impressive few-shot performance across many tasks. However, they still struggle when it comes to generating complex output structures, such as those required for Information Extraction. This limitation stems from the fact that LLMs, without finetuning, tend to generate free text rather than precise structures that follow a specific grammar. In this work, we propose to enrich the decoding step with formal grammar constraints. During beam search, only valid token continuations compliant with the grammar production rules are considered. This enforces the generation of valid sequences exclusively. Our framework is highly general and flexible, allowing any Context-Free Grammar (CFG) to be integrated into our custom constrained beam search implementation. We demonstrate that the outputs of many NLP tasks can be represented as formal languages, making them suitable for direct use in our framework. For task where the output space is dependent on the input, we propose input-depe
    
[^81]: 对话过程建模：现状、应用和实践影响的综述

    Conversational Process Modelling: State of the Art, Applications, and Implications in Practice. (arXiv:2304.11065v1 [cs.CL])

    [http://arxiv.org/abs/2304.11065](http://arxiv.org/abs/2304.11065)

    本文系统的研究了现有聊天机器人对于支持对话式流程建模所提供的应用场景，并推导出了在实践中使用聊天机器人进行对话式流程建模的建议。

    

    最近Chatbots等聊天机器人引起了极大的关注。对于BPM应用来说，如何应用聊天机器人来生成商业价值通常是不明确的。因此，本文旨在系统地分析现有的聊天机器人对于支持对话式流程建模作为面向流程的能力的支持。该研究识别了沿流程生命周期的应用场景，然后进行了对话式流程建模的系统文献综述。得出的分类学用作对话式流程建模的应用场景的识别，包括流程描述的释义和改进。应用场景基于高等教育领域的实际测试集对现有聊天机器人进行评估。该测试集包含流程描述及其对应的流程模型，以及模型质量的评估。基于文献和应用场景分析，得出了关于在对话式流程建模中使用聊天机器人的建议。

    Chatbots such as ChatGPT have caused a tremendous hype lately. For BPM applications, it is often not clear how to apply chatbots to generate business value. Hence, this work aims at the systematic analysis of existing chatbots for their support of conversational process modelling as process-oriented capability. Application scenarios are identified along the process life cycle. Then a systematic literature review on conversational process modelling is performed. The resulting taxonomy serves as input for the identification of application scenarios for conversational process modelling, including paraphrasing and improvement of process descriptions. The application scenarios are evaluated for existing chatbots based on a real-world test set from the higher education domain. It contains process descriptions as well as corresponding process models, together with an assessment of the model quality. Based on the literature and application scenario analyses, recommendations for the usage (prac
    
[^82]: CodeKGC：用于生成知识图谱构建的代码语言模型

    CodeKGC: Code Language Model for Generative Knowledge Graph Construction. (arXiv:2304.09048v1 [cs.CL])

    [http://arxiv.org/abs/2304.09048](http://arxiv.org/abs/2304.09048)

    本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法，能够有效利用知识图谱内的语义结构，提高模型的可解释性。

    

    目前的生成式知识图谱构建方法通常无法捕捉结构性知识，而只是将自然语言转化为序列化文本或规范语言。然而，对于像代码这样的结构化数据进行训练的大型生成式语言模型已经展现了在理解自然语言以进行结构性预测和推理任务方面的卓越能力。本文提出了一种使用代码语言模型处理生成式知识图谱构建任务的方法。具体而言，在给定代码格式的自然语言输入的情况下，目标是生成可以表示为代码补全任务的三元组。我们开发了具有模式感知型提示的方法，可以有效利用知识图谱内的语义结构。由于代码本质上具有结构，如类和函数定义，因此它作为先验的语义结构知识模型非常有用。此外，我们采用了基于原理的生成方法来提高性能。原理提供了模型生成结果的可解释性。

    Current generative knowledge graph construction approaches usually fail to capture structural knowledge by simply flattening natural language into serialized texts or a specification language. However, large generative language model trained on structured data such as code has demonstrated impressive capability in understanding natural language for structural prediction and reasoning tasks. Intuitively, we address the task of generative knowledge graph construction with code language model: given a code-format natural language input, the target is to generate triples which can be represented as code completion tasks. Specifically, we develop schema-aware prompts that effectively utilize the semantic structure within the knowledge graph. As code inherently possesses structure, such as class and function definitions, it serves as a useful model for prior semantic structural knowledge. Furthermore, we employ a rationale-enhanced generation method to boost the performance. Rationales provi
    
[^83]: ESD:预期平方差作为一种无需调参的可训练校准度量

    ESD: Expected Squared Difference as a Tuning-Free Trainable Calibration Measure. (arXiv:2303.02472v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02472](http://arxiv.org/abs/2303.02472)

    ESD是一种无需调参的可训练校准目标损失，通过将校准误差看作两个期望值之间的平方差，可以改善神经网络模型的校准度。

    

    研究表明，现代神经网络由于过于自信的预测而往往校准不良。传统上，在训练之后使用后处理方法来校准模型。近年来，已经提出了各种可训练的校准度量来直接将其纳入训练过程中。然而，这些方法都包含内部超参数，并且这些校准目标的性能依赖于调整这些超参数，随着神经网络和数据集的规模增大，会产生更多的计算成本。因此，我们提出了预期平方差（ESD），一种无需调参的可训练校准目标损失，我们从两个期望值之间的平方差的角度来看校准误差。通过对几种架构（CNN、Transformer）和数据集的大量实验证明，将ESD纳入训练可以改善模型的校准度。

    Studies have shown that modern neural networks tend to be poorly calibrated due to over-confident predictions. Traditionally, post-processing methods have been used to calibrate the model after training. In recent years, various trainable calibration measures have been proposed to incorporate them directly into the training process. However, these methods all incorporate internal hyperparameters, and the performance of these calibration objectives relies on tuning these hyperparameters, incurring more computational costs as the size of neural networks and datasets become larger. As such, we present Expected Squared Difference (ESD), a tuning-free (i.e., hyperparameter-free) trainable calibration objective loss, where we view the calibration error from the perspective of the squared difference between the two expectations. With extensive experiments on several architectures (CNNs, Transformers) and datasets, we demonstrate that (1) incorporating ESD into the training improves model cali
    
[^84]: 语言控制扩散：通过空间、时间和任务高效扩展

    Language Control Diffusion: Efficiently Scaling through Space, Time, and Tasks. (arXiv:2210.15629v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.15629](http://arxiv.org/abs/2210.15629)

    本文提出一种利用语言控制扩散模型的分层规划器，有效而高效地扩展扩散模型，解决长时间跨度自然语言指令下的控制问题，实现了较高的单任务和多任务成功率，并极大地提高计算效率。

    

    训练通用型智能体在各个方面都很困难，需要处理高维输入（空间）、长时间跨度（时间）和多个新任务。最近的结构方面的进展使得我们可以沿着其中一个或两个维度提高扩展性能力，但计算成本仍然很高。本文提出使用语言控制扩散模型作为一种基于自然语言条件的分层规划器（LCD）来应对这三个方面。我们有效而高效地扩展扩散模型，以应对时间、状态和任务空间维度的长时间跨度控制问题。我们在CALVIN语言机器人基准测试中将LCD与其他最先进的模型进行比较，发现LCD在多任务成功率方面优于其他最先进的方法，而单任务成功率（SR）为88.7%，远高于以前的最佳成绩82.6%，大大提高了计算效率。

    Training generalist agents is difficult across several axes, requiring us to deal with high-dimensional inputs (space), long horizons (time), and multiple and new tasks. Recent advances with architectures have allowed for improved scaling along one or two of these dimensions, but are still prohibitive computationally. In this paper, we propose to address all three axes by leveraging Language to Control Diffusion models as a hierarchical planner conditioned on language (LCD). We effectively and efficiently scale diffusion models for planning in extended temporal, state, and task dimensions to tackle long horizon control problems conditioned on natural language instructions. We compare LCD with other state-of-the-art models on the CALVIN language robotics benchmark and find that LCD outperforms other SOTA methods in multi task success rates while dramatically improving computational efficiency with a single task success rate (SR) of 88.7% against the previous best of 82.6%. We show that 
    

