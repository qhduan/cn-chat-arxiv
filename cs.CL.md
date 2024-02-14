# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CABINET: Content Relevance based Noise Reduction for Table Question Answering](https://rss.arxiv.org/abs/2402.01155) | CABINET是一个用于表格问答系统的基于内容相关性的噪声降低方法，通过加权处理表格内容并生成解析语句，使得大型语言模型能够专注于相关表格数据而抑制无关信息的干扰。 |
| [^2] | [Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance](https://arxiv.org/abs/2402.08680) | 本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。 |
| [^3] | [COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability](https://arxiv.org/abs/2402.08679) | 本文提出了COLD-Attack框架，旨在实现具有隐秘性和可控性的LLM越狱。通过建立可控文本生成与攻击生成之间的关联，采用了能量限制解码与Langevin动力学算法，使得在不同的控制要求下搜索对抗性LLM攻击成为可能。 |
| [^4] | [Improving Generalization in Semantic Parsing by Increasing Natural Language Variation](https://arxiv.org/abs/2402.08666) | 通过使用数据增强和大型语言模型，本论文提出的方法可以提高文本到SQL语义解析器在处理自然语言变体时的泛化能力，从而显著提升解析器的性能。 |
| [^5] | [Tandem Transformers for Inference Efficient LLMs](https://arxiv.org/abs/2402.08644) | 该论文提出了一种新的架构，称为串联Transformer，用于解决传统大型语言模型推断速度限制的问题。该架构通过将小型自回归模型和大模型以块模式结合起来，并让小模型关注大模型的丰富表示，从而显著提高了小模型的预测准确性。实验证明，在预训练数据集上，串联的PaLM2-Bison和PaLM2-Gecko相比独立的PaLM2-Gecko，在下一个词元预测准确性上提高了3.3%，并且相较于具有相似下游任务的PaLM2-Otter模型，加速比达到1.16倍。 |
| [^6] | [SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages](https://arxiv.org/abs/2402.08638) | SemRel2024是一个包含来自14种语言的语义文本相关性数据集合，通过该数据集合可以探索和量化语义相关性。这对于大型语言模型的能力和性能有重要的影响。这个数据集合涵盖了南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语等14种语言。 |
| [^7] | [Knowledge Editing on Black-box Large Language Models](https://arxiv.org/abs/2402.08631) | 这项研究提出了在黑盒大型语言模型上进行知识编辑的方法，并引入了一种多角度评估框架和一种新的postEdit框架，以解决现有方法中的隐私和风格问题。 |
| [^8] | [Bayesian Multi-Task Transfer Learning for Soft Prompt Tuning](https://arxiv.org/abs/2402.08594) | 本文提出了一种基于贝叶斯方法的多任务迁移学习框架，用于软提示调整。通过考虑源任务之间的相关性，我们可以提高提示在目标任务上的迁移效果。 |
| [^9] | [Improving Factual Error Correction for Abstractive Summarization via Data Distillation and Conditional-generation Cloze](https://arxiv.org/abs/2402.08581) | 本文提出了一种通过数据精炼和条件生成填空题的方法，以提高抽象摘要中的真实错误修正能力。该方法能够构建摘要中的真实因素之间的因果关系，并且在多个真实一致性指标上实现了改进。 |
| [^10] | [Test-Time Backdoor Attacks on Multimodal Large Language Models](https://arxiv.org/abs/2402.08577) | 本文提出了一种针对多模态大型语言模型的测试时反向门控攻击（AnyDoor），通过使用对抗性测试图像将反向门控注入到文本模态中，而无需访问或修改训练数据。AnyDoor具有分离设置和激活有害效果的时间的能力，并且在实验中证明了其有效性。 |
| [^11] | [Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast](https://arxiv.org/abs/2402.08567) | Agent Smith提出了一种安全问题，即传染性越狱，该问题在多代理环境中可以通过简单的越狱一个代理来迅速感染所有代理并导致有害行为。 |
| [^12] | [Higher Layers Need More LoRA Experts](https://arxiv.org/abs/2402.08562) | 这篇论文提出了一种新颖的参数高效的MoE方法（MoLA），用于Transformer-based模型，其中每个模型层可以灵活地使用不同数量的LoRA专家。通过在多个基准数据集上进行实验，研究结果表明高层需要更多的LoRA专家来提高模型性能。 |
| [^13] | [Concept-1K: A Novel Benchmark for Instance Incremental Learning](https://arxiv.org/abs/2402.08526) | 我们提出了一种名为Concept-1K的具有挑战性的实例增量学习（IIL）场景和数据集，揭示了十亿参数的PLM仍然遭受灾难性遗忘，影响因素包括模型规模、预训练和缓冲区大小。现有的IL方法和LoRA技术无法满足性能需求。我们的研究为探索和缓解PLM中的遗忘问题提供了新的场景。 |
| [^14] | [Auditing Counterfire: Evaluating Advanced Counterargument Generation with Evidence and Style](https://arxiv.org/abs/2402.08498) | 这项研究提出了一个新的数据集，用于生成具有证据和风格的反驳，该数据集基于Reddit ChangeMyView数据集中的帖子，并可用于论证的改进和评估。评估结果显示，GPT-3.5 turbo模型在论证质量方面表现出色，并且具有很高的风格融合能力。互惠式反驳的效果最佳。 |
| [^15] | [A Systematic Review of Data-to-Text NLG](https://arxiv.org/abs/2402.08496) | 这篇系统性回顾全面分析了数据到文本自然语言生成研究的现状，提出未来方向，并解决了相关挑战。 |
| [^16] | [Plausible Extractive Rationalization through Semi-Supervised Entailment Signal](https://arxiv.org/abs/2402.08479) | 本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型 |
| [^17] | [Lying Blindly: Bypassing ChatGPT's Safeguards to Generate Hard-to-Detect Disinformation Claims at Scale](https://arxiv.org/abs/2402.08467) | 本研究探索了ChatGPT在生成关于乌克兰战争的虚假信息方面的能力，发现它可以以较低成本、快速且大规模地生成逼真的定制虚假信息，而且这些虚假信息很难被人类读者和现有的自动化工具可靠地区分出来。 |
| [^18] | [LLMs and the Human Condition](https://arxiv.org/abs/2402.08403) | 本文提出了将三个成熟的人类决策理论整合到一起，形成了一个目的性人类行动模型。同时，将语言作为行动的观点应用于对话用户界面。通过理解ChatGPT的智能来源，可以在减少资源的同时获得对我们之间关系的认识。 |
| [^19] | [Large Language Models as Minecraft Agents](https://arxiv.org/abs/2402.08392) | 本文研究了在Minecraft代理环境中使用大型语言模型（LLMs）的应用和评估，探讨了建造者和建筑师设置中的挑战和机遇，提出了澄清问题，并介绍了与代理进行在线交互的平台。 |
| [^20] | [Punctuation Restoration Improves Structure Understanding without Supervision](https://arxiv.org/abs/2402.08382) | 标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。 |
| [^21] | [Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries](https://arxiv.org/abs/2402.08349) | 本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。 |
| [^22] | [Eliciting Big Five Personality Traits in Large Language Models: A Textual Analysis with Classifier-Driven Approach](https://arxiv.org/abs/2402.08341) | 本研究使用分类器驱动的方法，通过不同的输入提示探究大型语言模型的输出变化，以增加其透明度。结果显示，这些模型根据输入的不同提示而表现出不同的人格特质，类似于人类对刺激做出的反应。 |
| [^23] | [PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers](https://arxiv.org/abs/2402.08327) | PreFLMR是一种扩展细粒度迟交互多模态检索器，用于解决知识式视觉问答任务。该方法通过训练和评估框架M2KR进行了开发，并在多个任务中取得了新的最先进结果。此外，还对PreFLMR的扩展行为进行了研究，为通用多模态检索器的未来发展提供了有用的启示。 |
| [^24] | [Explicit References to Social Values in Fairy Tales: A Comparison between Three European Cultures](https://arxiv.org/abs/2402.08318) | 研究了葡萄牙、意大利和德国童话中明确表达的价值观差异，使用词嵌入技术和罗盘量化分析。初步发现表明这些国家之间存在共享的文化理解和对善良、遵从和普遍价值观的表达。 |
| [^25] | [Prompted Contextual Vectors for Spear-Phishing Detection](https://arxiv.org/abs/2402.08309) | 通过新的文档向量化方法，我们的方法使用大型语言模型来检测钓鱼网络攻击的电子邮件，并在实验证明具有高效性能。 |
| [^26] | [ChatCell: Facilitating Single-Cell Analysis with Natural Language](https://arxiv.org/abs/2402.08303) | ChatCell是一个利用自然语言促进单细胞分析的工具，通过词汇适应和统一序列生成，它具备深厚的专业知识和适应各种分析任务的能力。 |
| [^27] | [Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering](https://arxiv.org/abs/2402.08277) | 这项工作探索了如何鲁棒地微调大型语言模型以提高答案的来源质量和答案归因能力，引入了数据生成流水线和四个测试集来评估模型的性能，并展示了在合成数据上微调可以改善内部和外部分布的性能。 |
| [^28] | [A Survey of Table Reasoning with Large Language Models](https://arxiv.org/abs/2402.08259) | 这篇论文调查了使用大规模语言模型进行表格推理的现有研究，总结了LLM在表格推理中的优势，并探讨了未来增强表格推理能力的方向。 |
| [^29] | [Privacy-Preserving Language Model Inference with Instance Obfuscation](https://arxiv.org/abs/2402.08227) | 本论文提出了一种名为实例混淆推断（IOI）的方法，该方法致力于解决语言模型推断中决策隐私问题，通过对输入数据进行转换和保护，实现了在维持模型黑盒的同时保护决策隐私。 |
| [^30] | [BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models](https://arxiv.org/abs/2402.08219) | BBox-Adapter是一种适用于黑盒大型语言模型的轻量级适配器，通过区分目标和源域数据，并采用排名式噪音对比估计（NCE）损失和在线适应机制，实现了在透明、隐私和成本方面的有效适应。 |
| [^31] | [Pixel Sentence Representation Learning](https://arxiv.org/abs/2402.08183) | 本文提出了一种无监督的视觉句子表示学习框架，通过引入基于视觉基础文本扰动方法，如打字错误和单词顺序混排，借鉴认知和语言科学，以连续的方式感知文本的扰动，从而改善预训练语言模型在捕捉句子级文本语义方面的性能。 |
| [^32] | [CMA-R:Causal Mediation Analysis for Explaining Rumour Detection](https://arxiv.org/abs/2402.08155) | CMA-R通过因果中介分析解释了神经模型在Twitter上进行谣言检测的决策过程，并能够识别出关键推文和因果影响单词，提高了对黑盒子谣言检测系统的解释性和透明度。 |
| [^33] | [Active Preference Learning for Large Language Models](https://arxiv.org/abs/2402.08114) | 本论文提出了一种用于大型语言模型的主动偏好学习策略，通过直接偏好优化（DPO）来更好地利用偏好标签。实验结果表明，该方法提高了基于成对偏好数据的微调的学习速度和最终性能。 |
| [^34] | [Addressing cognitive bias in medical language models](https://arxiv.org/abs/2402.08113) | 本研究通过开发BiasMedQA，一个用于评估医学任务中LLMs的认知偏见的新型基准，发现LLMs在面对包含认知偏见的临床问题时，其回答的准确性明显降低。 |
| [^35] | [Investigating the Impact of Data Contamination of Large Language Models in Text-to-SQL Translation](https://arxiv.org/abs/2402.08100) | 本研究调查了大型语言模型在文本到SQL翻译中数据污染的影响。通过引入一种新的检测方法，研究人员发现GPT-3.5在陌生数据集上的性能显著下降。此外，通过采用对抗性表断开方法，研究人员还分析了GPT-3.5在修改信息的数据库上的效果。 |
| [^36] | [BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data](https://arxiv.org/abs/2402.08093) | 基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。 |
| [^37] | [Text-centric Alignment for Multi-Modality Learning](https://arxiv.org/abs/2402.08086) | 本研究提出了一种名为文本中心对齐的多模态学习方法（TAMML），利用大型语言模型和基础模型，通过将文本作为统一的语义空间，解决了多模态学习中的模态不匹配问题，并在处理未见过的、多样化的、不可预测的模态组合时取得了显著的改进。 |
| [^38] | [Large Language Models as Agents in Two-Player Games](https://arxiv.org/abs/2402.08078) | 通过将大型语言模型的训练过程重新概念化为基于语言的两人游戏中的代理学习，我们能够得到关键的见解，并提供了新的方法和技术来推进大型语言模型的发展。 |
| [^39] | [Beyond LLMs: Advancing the Landscape of Complex Reasoning](https://arxiv.org/abs/2402.08064) | 在人工智能领域，大型语言模型(LLMs)一直被视为解决许多问题的标准解决方案。然而，对于约束满足和优化问题，LLMs表现不佳。因此，Elemental Cognition开发了EC AI平台，采用神经符号方法解决这些问题，同时利用LLMs进行知识获取和用户交互。 |
| [^40] | [Careless Whisper: Speech-to-Text Hallucination Harms](https://arxiv.org/abs/2402.08021) | 该论文评估了开放AI的语音识别服务Whisper，并指出其中约1%的转录存在完全幻觉的短语或句子。这些幻觉内容中有38%包含明确的伤害，如暴力、虚构的个人信息或虚假的基于视频的权威。研究者进一步提供了幻觉发生的假设，并指出了由于语音类型和健康状况的不同可能导致的潜在差异。他们呼吁行业从业者改善基于语言模型的幻觉，并增强对下游潜在偏见的认识。 |
| [^41] | [Lumos : Empowering Multimodal LLMs with Scene Text Recognition](https://arxiv.org/abs/2402.08017) | 本论文介绍了Lumos，它是第一个具备文本理解能力的多模式问答系统，通过运用场景文本识别组件，能够从第一人称视角图像中提取文本，并将其用于加强多模式大型语言模型的输入。研究过程中，作者克服了与文本识别质量、延迟和模型推断相关的多个挑战，并提供了全面的组件评估结果，展示了高质量和高效率的性能。 |
| [^42] | [Enhancing Amharic-LLaMA: Integrating Task Specific and Generative Datasets](https://arxiv.org/abs/2402.08015) | 本研究通过整合任务特定和生成数据集来增强Amharic-LLaMA模型，提高了阿姆哈拉语言模型的性能。他们通过创建阿姆哈拉语指令微调数据集和微调模型，在不同的NLP任务中取得了有希望的结果。 |
| [^43] | [Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs](https://arxiv.org/abs/2402.08005) | 本文提出了一种改进的直接优化法（rDPO），通过使用合成数据来改善大规模语言模型（LLM）的行为调整。这种方法通过自我评论和广义DPO损失函数来优化学生LLM，并利用外部奖励模型提高合成数据质量，从而使rDPO在多个行为调整任务中表现出良好效果。 |
| [^44] | [Sentinels of the Stream: Unleashing Large Language Models for Dynamic Packet Classification in Software Defined Networks -- Position Paper](https://arxiv.org/abs/2402.07950) | 本文提出了在网络安全领域探索大型语言模型适用性的计划，计划创建名为Sentinel的LLM来分析网络数据包内容并评估威胁级别。 |
| [^45] | [Re-Envisioning Command and Control](https://arxiv.org/abs/2402.07946) | 重新构想的论文提出了未来指挥与控制（C2）决策需要面对更复杂和挑战性的环境，因此提出了基于人工智能系统与人类强有力伙伴关系的未来C2的愿景。这个愿景的核心是优化C2操作流程，保持协同努力，发展自适应的集体知识系统。 |
| [^46] | [UFO: A UI-Focused Agent for Windows OS Interaction](https://arxiv.org/abs/2402.07939) | UFO是一个专注于Windows操作系统上应用程序的用户界面智能体，利用GPT-Vision的能力来满足用户需求。它通过观察和分析Windows应用程序的图形用户界面和控制信息，实现无缝导航和操作以满足用户的请求。UFO的控制交互模块使得无需人工干预即可实现动作连接和完全自动化执行，使繁琐和耗时的过程变为简单任务。经过测试，UFO在各种场景中取得了良好效果。 |
| [^47] | [Large Language User Interfaces: Voice Interactive User Interfaces powered by LLMs](https://arxiv.org/abs/2402.07938) | 本研究旨在利用和引导升级后的LLMs的强大能力，构建一个框架，作为用户和用户界面之间的中介，通过对自然文本输入进行彻底分析，实现智能和响应式用户体验。 |
| [^48] | [A Human-Machine Collaboration Framework for the Development of Schemas](https://arxiv.org/abs/2402.07932) | 这篇论文提出了一个人机协作的框架用于设计新的模式，目的是解决机器智能中的挑战，并将AI社区的关注从技术转向AI科学。 |
| [^49] | [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927) | 这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。 |
| [^50] | [QACP: An Annotated Question Answering Dataset for Assisting Chinese Python Programming Learners](https://arxiv.org/abs/2402.07913) | 为解决编程智能教育系统中数据稀缺问题，本文提出了一个新的针对Python学习者的中文问答数据集，通过收集与分类真实学生问题，提高在线编程教育的效果和质量。 |
| [^51] | [Prompt4Vis: Prompting Large Language Models with Example Mining and Schema Filtering for Tabular Data Visualization](https://arxiv.org/abs/2402.07909) | 提出了 Prompt4Vis，使用示例挖掘和结构过滤来为表格数据可视化的大语言模型提供提示。这种方法利用了巨大的语言模型的优势，并能够改进当前自然语言查询转换成数据可视化查询的方法。 |
| [^52] | [Applications, challenges and ethical issues of AI and ChatGPT in education](https://arxiv.org/abs/2402.07907) | 本文探讨了人工智能和ChatGPT在改进教育方面的机遇，同时也指出了相关的挑战和伦理问题。 |
| [^53] | [Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/abs/2402.07896) | 本研究提出了一种名为“直接原则反馈”的新方法，用于控制语言模型中的LLM行为。通过在批评和修订上直接使用DPO来跳过响应的排名，我们成功地解决了“粉色大象问题”并取得了显著的性能优势。 |
| [^54] | [Quality Does Matter: A Detailed Look at the Quality and Utility of Web-Mined Parallel Corpora](https://arxiv.org/abs/2402.07446) | 这项研究详细分析了网络挖掘语料库的质量和实用性，并发现不同语言和数据集之间存在显著的质量差异。同时，我们还展示了某些网络挖掘数据集的最佳部分训练的神经机器翻译模型可以与人工策划的数据集持平。 |
| [^55] | [SALAD: Smart AI Language Assistant Daily](https://arxiv.org/abs/2402.07431) | SALAD是一款智能AI语言助手应用，旨在帮助外国人学习日语。它提供了多种学习工具和功能，包括翻译，语音识别，音频翻译，词汇跟踪等，并通过每日翻译帮助提高与母语人士的交流能力。调查结果显示60%的外国人对SALAD提升日语能力有信心。该应用利用大型语言模型和扩散模型促进日本社区的包容性。 |
| [^56] | [HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs](https://arxiv.org/abs/2402.07309) | 本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。 |
| [^57] | [How do Large Language Models Navigate Conflicts between Honesty and Helpfulness?](https://arxiv.org/abs/2402.07282) | 本文研究了如何在大型语言模型中权衡诚实和帮助性，在实验中发现强化学习改善了诚实和帮助性，而链式思维提示则偏向于帮助性。研究结果还展示了GPT-4 Turbo对对话框架和听众决策背景的敏感性。这些发现揭示了大型语言模型内化的对话价值观，并暗示零-shot提示可以在一定程度上引导这些抽象价值观。 |
| [^58] | [Learn To be Efficient: Build Structured Sparsity in Large Language Models](https://arxiv.org/abs/2402.06126) | 本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。 |
| [^59] | [Limits of Transformer Language Models on Algorithmic Learning](https://arxiv.org/abs/2402.05785) | Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。 |
| [^60] | [C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.03181) | C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。 |
| [^61] | [CroissantLLM: A Truly Bilingual French-English Language Model](https://arxiv.org/abs/2402.00786) | CroissantLLM是一个1.3B的双语语言模型，通过使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集进行训练，实现了高性能和开源。模型还发布了训练数据集和多个检查点，以及一个法语基准测试 FrenchBench。 |
| [^62] | [Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model](https://arxiv.org/abs/2402.00746) | 提出了一个创新的框架，健康-LLM，通过大规模特征提取和医学知识权衡评分，实现了个性化的检索增强疾病预测模型。这种方法通过整合健康报告，调整特征权重，以及利用语言模型和专家见解提高预测准确性，与传统健康管理方法相比具有明显优势。 |
| [^63] | [A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains](https://arxiv.org/abs/2402.00559) | 本论文提出了Reveal数据集，用于在开放领域的问答设置中对复杂思维链的自动验证进行基准测试。这个数据集包含了详尽的标签，用于评估语言模型的答案中每个推理步骤的相关性、归因和逻辑正确性。 |
| [^64] | [The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models](https://arxiv.org/abs/2401.12117) | 本研究评估了多模态大型语言模型（MLLMs）在非语言抽象推理方面的能力，并发现开源和闭源模型之间存在巨大差距和个体模块的关键缺陷。 |
| [^65] | [ALMs: Authorial Language Models for Authorship Attribution](https://arxiv.org/abs/2401.12005) | 本文介绍了一种名为作者语言模型（ALMs）的作者归属方法，通过计算疑问文件的困惑度来确定最有可能的作者。ALMs在Blogs50数据集上表现出色，在CCAT50数据集上与最好的方法相当。在较短文本上，ALMs需要较少的令牌来实现较高的准确率。 |
| [^66] | [Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA?](https://arxiv.org/abs/2401.11911) | 该论文研究了大型语言模型如何合并生成和检索的上下文以提升开放领域问答，发现这些模型偏向于生成的上下文，即使它们提供了错误的信息。 |
| [^67] | [Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales](https://arxiv.org/abs/2312.07399) | 该论文提出了一种基于提示生成的理由的“推理感知”诊断框架，通过大型语言模型来进行临床推理，实现了在疾病诊断过程中的高效、时间节约和劳动节约的方法。 |
| [^68] | [Multi-Step Dialogue Workflow Action Prediction](https://arxiv.org/abs/2311.09593) | 本文提出了多步骤工作流动作预测的新问题，通过准确预测多个步骤，实现对任务的多轮自动化，节省时间。提出了三种简单易行的建模方法，并展示了多步骤动作预测提高对话任务准确性和步骤自动化的特征。 |
| [^69] | [LILO: Learning Interpretable Libraries by Compressing and Documenting Code](https://arxiv.org/abs/2310.19791) | LILO是一种神经符号框架，通过迭代地合成、压缩和文档化代码来构建可解释且适用于特定问题领域的程序库。在其中，LILO结合了大型语言模型引导的程序合成和程序自动重构的算法进展，并且通过自动文档过程使得代码抽象可解释并提升性能。 |
| [^70] | [PeTailor: Improving Large Language Model by Tailored Chunk Scorer in Biomedical Triple Extraction](https://arxiv.org/abs/2310.18463) | 我们提出了PeTailor，这是一个基于检索的框架，通过使用定制的分块评分器从预先构建的分块数据库中检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，以改进生物医学三元组提取的效果。 |
| [^71] | [Exploring the Maze of Multilingual Modeling](https://arxiv.org/abs/2310.05404) | 本文综合评估了mBERT、XLM-R和GPT-3等三种流行的多语言语言模型在不同语言上的性能，并研究了资源可用性、语言家族、脚本类型和词序等因素对模型性能的影响。研究结果表明，语言特定预训练数据的数量对模型性能至关重要，同时还发现了其他重要因素。 |
| [^72] | [CCPrefix: Counterfactual Contrastive Prefix-Tuning for Many-Class Classification](https://arxiv.org/abs/2211.05987) | CCPrefix是一种针对多类分类的新型前缀调整方法，利用反事实对比来解决语言表示器模糊性问题。 |
| [^73] | [Scalable Qualitative Coding with LLMs: Chain-of-Thought Reasoning Matches Human Performance in Some Hermeneutic Tasks.](http://arxiv.org/abs/2401.15170) | 本研究证明了大型语言模型在定性编码中的应用潜力。相比于GPT-3.5，GPT-4能够实现与人类相当的解释能力，并具有较高的编码一致性。无论模型规模大小，只要满足一定条件，模型都可以实现较高的编码准确性。 |
| [^74] | [PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models.](http://arxiv.org/abs/2401.15042) | PROXYQA是一个用于评估大型语言模型长篇文本生成的替代框架，通过生成详尽的内容，并利用评估器和生成内容作为背景环境，根据评估器回答代理问题的表现来评估生成内容的质量。 |
| [^75] | [The Power of Noise: Redefining Retrieval for RAG Systems.](http://arxiv.org/abs/2401.14887) | 本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。 |
| [^76] | [AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models.](http://arxiv.org/abs/2401.09002) | 本研究提出一种新方法评估大型语言模型上越狱攻击效果，引入粗粒度和细粒度评估框架，提供了更全面和细致的评估角度，并开发了专门的真实数据集作为基准，为未来研究建立了基础资源。 |
| [^77] | [Multilingual Instruction Tuning With Just a Pinch of Multilinguality.](http://arxiv.org/abs/2401.01854) | 本研究研究了多语言指令调优中的多语言性对跨语言指令遵循的影响。研究发现，即使在单语调优过程中，许多语言也可以将一些指令遵循能力转移到其他语言上。此外，只有40个多语言示例能够显著提高多语言指令遵循。总体来说，多语言混合调优的模型在多种语言上的表现相比单语调优的模型要好或者不相上下，尽管使用的这些语言的训练示例数量只有10倍少。 |
| [^78] | [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.](http://arxiv.org/abs/2401.01335) | 本文提出了一种名为自我对弱语言模型进行细调（SPIN）的方法，通过模型自我对弈生成训练数据，并从中优化模型策略，从而将弱语言模型转化为强语言模型，无需额外的人类标注数据。 |
| [^79] | [Supercharging academic writing with generative AI: framework, techniques, and caveats.](http://arxiv.org/abs/2310.17143) | 这篇论文介绍了使用生成型人工智能（AI）提高学术写作质量和效率的原则和方法，包括一个人机协作框架、有效的提示技术和两阶段模型，旨在实现认知卸载和想象刺激的AI辅助写作。 |
| [^80] | [Controlled Decoding from Language Models.](http://arxiv.org/abs/2310.17022) | 本论文提出了一种名为受控解码（CD）的离策略强化学习方法，用于控制语言模型的生成，以达到高回报的结果。CD通过前缀评分器来引导生成，可以在推理时预测预期回报，并且具有模块化设计，可用于解决多目标强化学习问题，而不增加复杂性。 |
| [^81] | [Instruction Tuning with Human Curriculum.](http://arxiv.org/abs/2310.09518) | 本文探讨了在大型语言模型中应用结构化认知学习方法进行指令调整的潜在好处，并提出了一个高度结构化的合成数据集，结果表明该方法优于传统的随机化方法，提高了指令调整的性能。 |
| [^82] | [Ranking LLM-Generated Loop Invariants for Program Verification.](http://arxiv.org/abs/2310.09342) | 本研究提出了一种针对LLM生成结果进行重新排名的方法，可以显著提高正确不变量的排名，从而减少程序验证的调用次数。 |
| [^83] | [CATfOOD: Counterfactual Augmented Training for Improving Out-of-Domain Performance and Calibration.](http://arxiv.org/abs/2309.07822) | 本研究通过在小型语言模型训练数据中增加自动生成的反事实实例，提高了摘要问答模型在领域外的性能和模型校准能力，并发现性能改进与反事实实例的多样性相关。 |
| [^84] | [Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?.](http://arxiv.org/abs/2309.07462) | 大型语言模型（LLMs）作为评估器可以解决当前多语言评估的限制和挑战，能够对各种语言中的自然语言处理任务进行有效评估。 |
| [^85] | [LaFiCMIL: Rethinking Large File Classification from the Perspective of Correlated Multiple Instance Learning.](http://arxiv.org/abs/2308.01413) | LaFiCMIL是一个新的方法，从相关多实例学习的角度解决了Transformer模型输入长度限制的问题，可以用于改进大文件分类任务。 |
| [^86] | [Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy.](http://arxiv.org/abs/2307.13808) | 本论文研究了在语言模型中嵌入水印以进行AI检测的挑战，并提出了一种简单而有效的语义感知水印算法，该算法在保持检测能力的同时，在不同的文本生成任务中取得了显著改进。 |
| [^87] | [AlpaGasus: Training A Better Alpaca with Fewer Data.](http://arxiv.org/abs/2307.08701) | 这项研究提出了一种用于训练语言模型的数据筛选策略AlpaGasus，通过使用强大的语言模型过滤掉低质量数据，它在测试中表现出比原始模型更好的性能，并提供了更快的训练速度。 |
| [^88] | [This Land is {Your, My} Land: Evaluating Geopolitical Biases in Language Models.](http://arxiv.org/abs/2305.14610) | 本文提出了地缘政治偏见的概念，并以领土争端为例，利用多语言、多选题的数据集BorderLines和几个定量指标分析语言模型响应中的地缘政治偏见现象。 |
| [^89] | [Efficient Open Domain Multi-Hop Question Answering with Few-Shot Data Synthesis.](http://arxiv.org/abs/2305.13691) | 提出了一种使用少量合成数据进行高效的开放领域多跳问题回答的方法，可用于改善小型语言模型的性能。通过语言模型和提示参数化的数据生成函数合成数据，微调后的模型参数量只有之前模型的三分之一，达到了与之前模型类似的性能。 |
| [^90] | [Learning to Compress Prompts with Gist Tokens.](http://arxiv.org/abs/2304.08467) | 该论文提出了一种名为"gisting"的方法，通过训练语言模型将提示压缩为更小的"要点"标记集，以提高计算效率。通过这种方法，可以实现高达26倍的提示压缩，减少40％的FLOPs、4.2％的墙时速度提升，并节省存储空间，同时最小化输出质量损失。 |
| [^91] | [The Deep Latent Position Topic Model for Clustering and Representation of Networks with Textual Edges.](http://arxiv.org/abs/2304.08242) | 深层潜在位置主题模型用于网络聚类和表示，通过基于模型的聚类策略和概率模型对节点和边进行联合表示，并使用模型选择准则进行参数选择。 |
| [^92] | [Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning.](http://arxiv.org/abs/2301.11916) | 本研究发现，大型语言模型可以被视为隐式的主题模型，并提出了一种算法，从注释数据中选择最佳示范，大大提高了上下文学习的能力。 |
| [^93] | [What can we know about that which we cannot even imagine?.](http://arxiv.org/abs/2208.03886) | 这篇文章探讨了关于智能、人类语言和人类数学的问题，强调了人类语言的局限性，以及我们能否对我们无法想象的事物有任何了解。 |

# 详细

[^1]: CABINET: 表格问答系统的基于内容相关性的噪声降低方法

    CABINET: Content Relevance based Noise Reduction for Table Question Answering

    [https://rss.arxiv.org/abs/2402.01155](https://rss.arxiv.org/abs/2402.01155)

    CABINET是一个用于表格问答系统的基于内容相关性的噪声降低方法，通过加权处理表格内容并生成解析语句，使得大型语言模型能够专注于相关表格数据而抑制无关信息的干扰。

    

    大型语言模型（LLMs）的表格理解能力通过对表格的问答任务进行了广泛研究。通常，只有表格的一小部分与给定问题的答案相关。不相关的部分会产生噪声和干扰信息，导致LLMs的性能下降。为了缓解这个问题，我们提出了CABINET（基于内容相关性的表格问答噪声降低方法）- 一个能够让LLMs专注于相关表格数据并抑制无关信息的框架。CABINET包括一个无监督的相关性评分器（URS），与问答LLM差异性训练，根据其与输入问题的相关性对表格内容进行加权处理后再输入问答LLM（QA LLM）。为了进一步辅助相关性评分器，CABINET利用一个弱监督模块生成一个解析语句，描述行和列的标准。

    Table understanding capability of Large Language Models (LLMs) has been extensively studied through the task of question-answering (QA) over tables. Typically, only a small part of the whole table is relevant to derive the answer for a given question. The irrelevant parts act as noise and are distracting information, resulting in sub-optimal performance due to the vulnerability of LLMs to noise. To mitigate this, we propose CABINET (Content RelevAnce-Based NoIse ReductioN for TablE QuesTion-Answering) - a framework to enable LLMs to focus on relevant tabular data by suppressing extraneous information. CABINET comprises an Unsupervised Relevance Scorer (URS), trained differentially with the QA LLM, that weighs the table content based on its relevance to the input question before feeding it to the question-answering LLM (QA LLM). To further aid the relevance scorer, CABINET employs a weakly supervised module that generates a parsing statement describing the criteria of rows and columns r
    
[^2]: 通过无分类器引导来减轻大型视觉语言模型中的物体幻觉

    Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance

    [https://arxiv.org/abs/2402.08680](https://arxiv.org/abs/2402.08680)

    本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。

    

    大型视觉语言模型（LVLM）的进展越来越突出了它们在图像中产生虚假物体的严重问题。为了解决这个问题，先前的研究着重于使用特殊策划的数据集或强大的LLM（例如GPT-3.5）来纠正LVLM的输出。然而，这些方法要求昂贵的训练/微调或API访问先进的LLM来在生成后纠正模型的输出。在本文中，我们通过引入一个名为通过无分类器引导缓解幻觉的框架（MARINE）来解决这个挑战，该框架既无需训练也无需API访问，可以在生成过程中有效地减少物体幻觉。具体而言，MARINE通过集成现有的开源视觉模型丰富LVLM的视觉语境，并使用无分类器引导来整合额外的物体基础特征，以提高LVLM生成的精确性。

    The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs (e.g., GPT-3.5) to rectify the outputs of LVLMs. However, these approaches require either expensive training/fine-tuning or API access to advanced LLMs to correct the model's output post-generation. In this paper, we tackle this challenge by introducing a framework called Mitigating hallucinAtion via classifieR-Free guIdaNcE (MARINE), which is both training-free and API-free, and can effectively and efficiently reduce object hallucinations during the generation process. Specifically, MARINE enriches the visual context of LVLMs by integrating existing open-source vision models, and employs classifier-free guidance to incorporate the additional object grounding features to improve the precision of LVLMs' generations. Thr
    
[^3]: COLD-Attack: 用于具有隐秘性和可控性的LLM越狱

    COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability

    [https://arxiv.org/abs/2402.08679](https://arxiv.org/abs/2402.08679)

    本文提出了COLD-Attack框架，旨在实现具有隐秘性和可控性的LLM越狱。通过建立可控文本生成与攻击生成之间的关联，采用了能量限制解码与Langevin动力学算法，使得在不同的控制要求下搜索对抗性LLM攻击成为可能。

    

    最近对大型语言模型（LLMs）进行越狱的注意力越来越多。为了全面评估LLM的安全性，有必要考虑具有不同属性的越狱，例如上下文连贯性以及情感/风格变化，因此研究可控性越狱是有益的，即如何对LLM攻击进行控制。在本文中，我们正式形式化了可控性攻击生成问题，并建立了该问题与可控文本生成之间的新型关联，这是自然语言处理中一个被广泛探索的主题。基于这种关联，我们改进了能量限制解码与Langevin动力学（COLD）的算法，这是一种在可控文本生成中的高效算法，并引入了COLD-Attack框架，该框架统一且自动化地搜索各种控制要求下的对抗性LLM攻击，例如流畅性、隐秘性、情感和左右连贯性。

    Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controlla
    
[^4]: 通过增加自然语言变体来提高语义解析的泛化能力

    Improving Generalization in Semantic Parsing by Increasing Natural Language Variation

    [https://arxiv.org/abs/2402.08666](https://arxiv.org/abs/2402.08666)

    通过使用数据增强和大型语言模型，本论文提出的方法可以提高文本到SQL语义解析器在处理自然语言变体时的泛化能力，从而显著提升解析器的性能。

    

    文本到SQL语义解析在近年来取得了显著进展，各种模型在具有挑战性的Spider基准测试上展示了令人印象深刻的性能。然而，研究表明，即使面对先前（精确）解析表达式的轻微扰动，这些模型通常也很难泛化。这主要是由于Spider中问题的语言形式过于具体、不自然且变化有限。在这项工作中，我们使用数据增强来提升文本到SQL解析器对自然语言变体的鲁棒性。现有的方法通过在Spider上训练的模型生成问题重组，或仅引入局部变化。相比之下，我们利用大型语言模型的能力生成更真实和多样的问题。仅使用几个提示，我们在Spider中的问题数量增加了一倍。在这个增强的数据集上进行训练可以显著提高解析器的性能。

    Text-to-SQL semantic parsing has made significant progress in recent years, with various models demonstrating impressive performance on the challenging Spider benchmark. However, it has also been shown that these models often struggle to generalize even when faced with small perturbations of previously (accurately) parsed expressions. This is mainly due to the linguistic form of questions in Spider which are overly specific, unnatural, and display limited variation. In this work, we use data augmentation to enhance the robustness of text-to-SQL parsers against natural language variations. Existing approaches generate question reformulations either via models trained on Spider or only introduce local changes. In contrast, we leverage the capabilities of large language models to generate more realistic and diverse questions. Using only a few prompts, we achieve a two-fold increase in the number of questions in Spider. Training on this augmented dataset yields substantial improvements on 
    
[^5]: 用于推断高效LLMs的串联Transformer

    Tandem Transformers for Inference Efficient LLMs

    [https://arxiv.org/abs/2402.08644](https://arxiv.org/abs/2402.08644)

    该论文提出了一种新的架构，称为串联Transformer，用于解决传统大型语言模型推断速度限制的问题。该架构通过将小型自回归模型和大模型以块模式结合起来，并让小模型关注大模型的丰富表示，从而显著提高了小模型的预测准确性。实验证明，在预训练数据集上，串联的PaLM2-Bison和PaLM2-Gecko相比独立的PaLM2-Gecko，在下一个词元预测准确性上提高了3.3%，并且相较于具有相似下游任务的PaLM2-Otter模型，加速比达到1.16倍。

    

    传统的大型语言模型( LLMs )具有自回归的特性，这使得推断速度受到限制，因为词元是按顺序生成的。尽管有些预测和并行解码技术试图减轻这个问题，但它们都有限制：要么依赖更精简但准确度较低的模型进行生成，要么没有充分利用基础LLM的表示。我们提出了一种新颖的架构，即串联Transformer，来解决这些问题。这种架构独特地结合了(1)一个小型自回归模型和(2)一个以块模式运行的大模型(同时处理多个词元)。通过让小模型关注大模型更丰富的表示，大幅提升小模型的预测准确性。在PaLM2预训练数据集上，PaLM2-Bison和PaLM2-Gecko的串联相较独立的PaLM2-Gecko，在下一个词元预测准确性上提升了3.3%，与具有相似下游任务的PaLM2-Otter模型相比，提供了1.16倍的加速比。

    The autoregressive nature of conventional large language models (LLMs) inherently limits inference speed, as tokens are generated sequentially. While speculative and parallel decoding techniques attempt to mitigate this, they face limitations: either relying on less accurate smaller models for generation or failing to fully leverage the base LLM's representations.   We introduce a novel architecture, Tandem transformers, to address these issues. This architecture uniquely combines (1) a small autoregressive model and (2) a large model operating in block mode (processing multiple tokens simultaneously). The small model's predictive accuracy is substantially enhanced by granting it attention to the large model's richer representations. On the PaLM2 pretraining dataset, a tandem of PaLM2-Bison and PaLM2-Gecko demonstrates a 3.3% improvement in next-token prediction accuracy over a standalone PaLM2-Gecko, offering a 1.16x speedup compared to a PaLM2-Otter model with comparable downstream p
    
[^6]: SemRel2024: 14种语言的语义文本相关性数据集合

    SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages

    [https://arxiv.org/abs/2402.08638](https://arxiv.org/abs/2402.08638)

    SemRel2024是一个包含来自14种语言的语义文本相关性数据集合，通过该数据集合可以探索和量化语义相关性。这对于大型语言模型的能力和性能有重要的影响。这个数据集合涵盖了南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语等14种语言。

    

    探索和量化语义相关性是语言表达的核心。它在各种自然语言处理任务中具有重要影响，包括为大型语言模型（LLM）的能力和性能提供洞察。虽然早期的自然语言处理研究主要集中在语义相似性上，往往是在英语语境中，但我们认为更广泛的语义相关性现象值得研究。本文介绍了SemRel，这是一个由母语为14种语言进行注释的新的语义相关性数据集合：南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语。这些语言来自五个不同的语系，主要在非洲和亚洲使用，这些地区的自然语言处理资源相对较少。SemRel数据集中的每个实例都是与一个表示相关性得分的句子对相关联。

    Exploring and quantifying semantic relatedness is central to representing language. It holds significant implications across various NLP tasks, including offering insights into the capabilities and performance of Large Language Models (LLMs). While earlier NLP research primarily focused on semantic similarity, often within the English language context, we instead investigate the broader phenomenon of semantic relatedness. In this paper, we present SemRel, a new semantic relatedness dataset collection annotated by native speakers across 14 languages:Afrikaans, Algerian Arabic, Amharic, English, Hausa, Hindi, Indonesian, Kinyarwanda, Marathi, Moroccan Arabic, Modern Standard Arabic, Punjabi, Spanish, and Telugu. These languages originate from five distinct language families and are predominantly spoken in Africa and Asia -- regions characterised by a relatively limited availability of NLP resources. Each instance in the SemRel datasets is a sentence pair associated with a score that repr
    
[^7]: 在黑盒大型语言模型上进行知识编辑

    Knowledge Editing on Black-box Large Language Models

    [https://arxiv.org/abs/2402.08631](https://arxiv.org/abs/2402.08631)

    这项研究提出了在黑盒大型语言模型上进行知识编辑的方法，并引入了一种多角度评估框架和一种新的postEdit框架，以解决现有方法中的隐私和风格问题。

    

    知识编辑旨在高效、精确地修改大型语言模型的行为，以更新特定的知识，而不对其他知识产生负面影响。当前的研究主要集中在白盒语言模型编辑上，忽视了一个重要的场景：黑盒语言模型编辑，即通过接口访问语言模型，并仅可用文本输出。为了解决现有评估在黑盒语言模型编辑上不适用且缺乏全面性的局限性，我们提出了一种多角度评估框架，首次将风格保留的评估纳入其中。为了解决当前方法中的编辑数据隐私泄漏和风格过度编辑的问题，我们引入了一种新的postEdit框架，通过下游后处理解决隐私问题，并通过对原始回答进行细粒度编辑来保持文本风格一致性。在两个基准测试上的实验与分析表明，postEdit的性能超过了所有现有方法。

    Knowledge editing (KE) aims to efficiently and precisely modify the behavior of large language models (LLMs) to update specific knowledge without negatively influencing other knowledge. Current research primarily focuses on white-box LLMs editing, overlooking an important scenario: black-box LLMs editing, where LLMs are accessed through interfaces and only textual output is available. To address the limitations of existing evaluations that are not inapplicable to black-box LLM editing and lack comprehensiveness, we propose a multi-perspective evaluation framework, incorporating the assessment of style retention for the first time. To tackle privacy leaks of editing data and style over-editing in current methods, we introduce a novel postEdit framework, resolving privacy concerns through downstream post-processing and maintaining textual style consistency via fine-grained editing to original responses. Experiments and analysis on two benchmarks demonstrate that postEdit outperforms all 
    
[^8]: 贝叶斯多任务迁移学习用于软提示调整

    Bayesian Multi-Task Transfer Learning for Soft Prompt Tuning

    [https://arxiv.org/abs/2402.08594](https://arxiv.org/abs/2402.08594)

    本文提出了一种基于贝叶斯方法的多任务迁移学习框架，用于软提示调整。通过考虑源任务之间的相关性，我们可以提高提示在目标任务上的迁移效果。

    

    提示调整是一种优化预训练语言模型的方法，通过优化提示来适应下游任务，而不是微调整个模型参数。当这些提示在多任务迁移学习设置下进行训练时，已经证明其特别有效。这些方法通常涉及为每个源任务单独训练提示，然后聚合它们以提供目标任务的提示初始化。然而，这种方法忽视了一些源任务之间可能存在的负面或正面干扰。我们认为，当我们通过训练源提示从源任务中提取知识时，需要考虑源任务之间的相关性，以实现更好的向目标任务的迁移。为此，我们提出了一种基于贝叶斯方法的思路，通过工作在提示在源任务之间的后验分布上。我们利用从后验中提取的样本获得代表性的源提示。

    Prompt tuning, in which prompts are optimized to adapt large-scale pre-trained language models to downstream tasks instead of fine-tuning the full model parameters, has been shown to be particularly effective when the prompts are trained in a multi-task transfer learning setting. These methods generally involve individually training prompts for each source task and then aggregating them to provide the initialization of the prompt for the target task. However, this approach critically ignores the fact that some of the source tasks could be negatively or positively interfering with each other. We argue that when we extract knowledge from source tasks via training source prompts, we need to consider this correlation among source tasks for better transfer to target tasks. To this end, we propose a Bayesian approach where we work with the posterior distribution of prompts across source tasks. We obtain representative source prompts corresponding to the samples from the posterior utilizing S
    
[^9]: 通过数据精炼和条件生成填空题，提高抽象摘要的真实错误修正能力

    Improving Factual Error Correction for Abstractive Summarization via Data Distillation and Conditional-generation Cloze

    [https://arxiv.org/abs/2402.08581](https://arxiv.org/abs/2402.08581)

    本文提出了一种通过数据精炼和条件生成填空题的方法，以提高抽象摘要中的真实错误修正能力。该方法能够构建摘要中的真实因素之间的因果关系，并且在多个真实一致性指标上实现了改进。

    

    提高抽象摘要中的真实一致性一直是当前研究的重点。一种有前途的方法是后期编辑方法。然而，先前的工作尚未充分利用摘要中的真实因素，并且受到训练数据集的负面影响。在本文中，我们首先提出了一种基于条件生成填空题的新型真实错误修正模型FactCloze。FactCloze可以在摘要中构建真实因素之间的因果关系，同时能够确定空白是否可以回答。然后，我们提出了一种通过多维评估生成更加准确的摘要数据集SummDSC的数据精炼方法。通过实验证实了我们的方法的有效性，与基准方法相比，在多个真实一致性指标上实现了改进。

    Improving factual consistency in abstractive summarization has been a focus of current research. One promising approach is the post-editing method. However, previous works have yet to make sufficient use of factual factors in summaries and suffers from the negative effect of the training datasets. In this paper, we first propose a novel factual error correction model FactCloze based on a conditional-generation cloze task. FactCloze can construct the causality among factual factors while being able to determine whether the blank can be answered or not. Then, we propose a data distillation method to generate a more faithful summarization dataset SummDSC via multiple-dimensional evaluation. We experimentally validate the effectiveness of our approach, which leads to an improvement in multiple factual consistency metrics compared to baselines.
    
[^10]: 对多模态大型语言模型的测试时反向门控攻击

    Test-Time Backdoor Attacks on Multimodal Large Language Models

    [https://arxiv.org/abs/2402.08577](https://arxiv.org/abs/2402.08577)

    本文提出了一种针对多模态大型语言模型的测试时反向门控攻击（AnyDoor），通过使用对抗性测试图像将反向门控注入到文本模态中，而无需访问或修改训练数据。AnyDoor具有分离设置和激活有害效果的时间的能力，并且在实验中证明了其有效性。

    

    反向门控攻击通常通过污染训练数据来执行，从而在测试阶段触发预定的有害效果。在本文中，我们提出了AnyDoor，一种针对多模态大型语言模型（MLLMs）的测试时反向门控攻击，它使用对抗性测试图像将反向门控注入到文本模态中（共享相同的通用扰动），而无需访问或修改训练数据。AnyDoor采用类似于通用对抗攻击的技术，但其通过能够分离有害效果的设置和激活的时间来区别于其他攻击。在我们的实验中，我们验证了AnyDoor对流行的MLLMs（如LLaVA-1.5、MiniGPT-4、InstructBLIP和BLIP-2）的有效性，并提供了全面的消融研究。值得注意的是，由于反向门控由通用扰动注入，AnyDoor可以动态改变其反向门触发提示/有害效果，从而暴露出...

    Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing 
    
[^11]: Agent Smith:一张图像可以迅速越狱一百万个多模态LLM代理

    Agent Smith: A Single Image Can Jailbreak One Million Multimodal LLM Agents Exponentially Fast

    [https://arxiv.org/abs/2402.08567](https://arxiv.org/abs/2402.08567)

    Agent Smith提出了一种安全问题，即传染性越狱，该问题在多代理环境中可以通过简单的越狱一个代理来迅速感染所有代理并导致有害行为。

    

    多模态大型语言模型（MLLM）代理可以接收指令，捕捉图像，从内存中检索历史记录，并决定使用哪些工具。然而，红队评估发现恶意图像/提示可以越狱MLLM并导致不对齐的行为。在这项工作中，我们报告了多代理环境中更严重的安全问题，称为传染性越狱。它涉及到对单个代理进行简单的越狱，无需来自对手的进一步干预，（几乎）所有代理将以指数级别被感染并展示有害行为。为了验证传染性越狱的可行性，我们模拟了包含高达一百万个LLaVA-1.5代理的多代理环境，并将随机匹配对聊天作为多代理交互的概念验证实例。我们的结果表明，将（传染性）恶意图像输入到任意选择的代理的内存中就足以实现传染性越狱。

    A multimodal large language model (MLLM) agent can receive instructions, capture images, retrieve histories from memory, and decide which tools to use. Nonetheless, red-teaming efforts have revealed that adversarial images/prompts can jailbreak an MLLM and cause unaligned behaviors. In this work, we report an even more severe safety issue in multi-agent environments, referred to as infectious jailbreak. It entails the adversary simply jailbreaking a single agent, and without any further intervention from the adversary, (almost) all agents will become infected exponentially fast and exhibit harmful behaviors. To validate the feasibility of infectious jailbreak, we simulate multi-agent environments containing up to one million LLaVA-1.5 agents, and employ randomized pair-wise chat as a proof-of-concept instantiation for multi-agent interaction. Our results show that feeding an (infectious) adversarial image into the memory of any randomly chosen agent is sufficient to achieve infectious 
    
[^12]: Higher Layers Need More LoRA Experts

    Higher Layers Need More LoRA Experts

    [https://arxiv.org/abs/2402.08562](https://arxiv.org/abs/2402.08562)

    这篇论文提出了一种新颖的参数高效的MoE方法（MoLA），用于Transformer-based模型，其中每个模型层可以灵活地使用不同数量的LoRA专家。通过在多个基准数据集上进行实验，研究结果表明高层需要更多的LoRA专家来提高模型性能。

    

    参数高效调整（PEFT）技术，如低秩适应（LoRA），在大型语言模型上提供了训练效率，但对模型性能的影响仍有限。最近的努力整合了LoRA和专家混合（MoE），以提高PEFT方法的性能。尽管有了有希望的结果，但改进带有MoE的LoRA的效率的研究仍处于初级阶段。最近的研究表明，MoE体系结构中的专家具有不同的优势，并且还存在一些冗余。这个论断是否也适用于参数高效的MoE？在本文中，我们介绍了一种新颖的参数高效的MoE方法，称为MoLA（\textit{\textbf{M}oE-L\textbf{o}RA with \textbf{L}ayer-wise Expert \textbf{A}llocation）），用于基于Transformer的模型，其中每个模型层可以灵活地使用不同数量的LoRA专家。我们研究了几种具有不同层级专家配置的体系结构。对六个知名的自然语言处理和常识问答基准进行了实验。

    Parameter-efficient tuning (PEFT) techniques like low-rank adaptation (LoRA) offer training efficiency on Large Language Models, but their impact on model performance remains limited. Recent efforts integrate LoRA and Mixture-of-Experts (MoE) to improve the performance of PEFT methods. Despite promising results, research on improving the efficiency of LoRA with MoE is still in its early stages. Recent studies have shown that experts in the MoE architecture have different strengths and also exhibit some redundancy. Does this statement also apply to parameter-efficient MoE? In this paper, we introduce a novel parameter-efficient MoE method, \textit{\textbf{M}oE-L\textbf{o}RA with \textbf{L}ayer-wise Expert \textbf{A}llocation (MoLA)} for Transformer-based models, where each model layer has the flexibility to employ a varying number of LoRA experts. We investigate several architectures with varying layer-wise expert configurations. Experiments on six well-known NLP and commonsense QA benc
    
[^13]: Concept-1K：一种用于实例增量学习的新型基准

    Concept-1K: A Novel Benchmark for Instance Incremental Learning

    [https://arxiv.org/abs/2402.08526](https://arxiv.org/abs/2402.08526)

    我们提出了一种名为Concept-1K的具有挑战性的实例增量学习（IIL）场景和数据集，揭示了十亿参数的PLM仍然遭受灾难性遗忘，影响因素包括模型规模、预训练和缓冲区大小。现有的IL方法和LoRA技术无法满足性能需求。我们的研究为探索和缓解PLM中的遗忘问题提供了新的场景。

    

    增量学习（IL）对于实现神经网络中的人类级智能至关重要。然而，现有的IL场景和数据集无法评估PLM中的遗忘，使人误以为PLM不会遭受灾难性遗忘。为此，我们提出了一种具有挑战性的IL场景，称为实例增量学习（IIL），以及一个支持数量级更大的IL步骤的新数据集Concept-1K。基于对Concept-1K的实验，我们揭示了十亿参数的PLM仍然遭受着灾难性遗忘，并且遗忘受模型规模、预训练和缓冲区大小的影响。此外，现有的IL方法和一种流行的微调技术LoRA都未能达到令人满意的性能。我们的研究为未来研究提供了一个新的场景，探索PLM的灾难性遗忘，并鼓励设计更强大的技术以减轻PLM中的遗忘问题。

    Incremental learning (IL) is essential to realize the human-level intelligence in the neural network. However, existing IL scenarios and datasets are unqualified for assessing forgetting in PLMs, giving an illusion that PLMs do not suffer from catastrophic forgetting. To this end, we propose a challenging IL scenario called instance-incremental learning (IIL) and a novel dataset called Concept-1K, which supports an order of magnitude larger IL steps. Based on the experiments on Concept-1K, we reveal that billion-parameter PLMs still suffer from catastrophic forgetting, and the forgetting is affected by both model scale, pretraining, and buffer size. Furthermore, existing IL methods and a popular finetuning technique, LoRA, fail to achieve satisfactory performance. Our study provides a novel scenario for future studies to explore the catastrophic forgetting of PLMs and encourage more powerful techniques to be designed for alleviating the forgetting in PLMs. The data, code and scripts ar
    
[^14]: 审计反火：评估具有证据和风格的先进反驳生成

    Auditing Counterfire: Evaluating Advanced Counterargument Generation with Evidence and Style

    [https://arxiv.org/abs/2402.08498](https://arxiv.org/abs/2402.08498)

    这项研究提出了一个新的数据集，用于生成具有证据和风格的反驳，该数据集基于Reddit ChangeMyView数据集中的帖子，并可用于论证的改进和评估。评估结果显示，GPT-3.5 turbo模型在论证质量方面表现出色，并且具有很高的风格融合能力。互惠式反驳的效果最佳。

    

    我们提出了一个新颖的数据集，用于控制性反驳的合成，旨在进一步应用于论证的改进、挖掘和评估。我们的数据集包含与Reddit ChangeMyView数据集中的帖子相结合的丰富的反驳，这些反驳融入了从高质量来源中检索到的证据，并根据用户偏好生成，调整了证据和论证风格的关键属性。由此产生的Counterfire语料库包括从GPT-3.5 turbo、Koala和PaLM 2模型以及它们的两个微调变体生成的论证（N = 32,000）。模型评估表明，在证据方面具有强大的改写能力，尽管词汇重叠有限，同时表现出高度的风格融合（对于“互惠”的得分为0.9682），显示了LLM融合多样风格的能力。在所有模型中，GPT-3.5 turbo在论证质量评估中显示出最高分数，表现出一致准确性（得分 >0.8）。在进一步的分析中，互惠式反驳证明效果最佳，能够产生更好的论证结果。

    We present a novel dataset for the controlled composition of counterarguments designed for further applications in argument refining, mining, and evaluation. Our dataset constitutes enriched counter-arguments to posts in the Reddit ChangeMyView dataset that are integrated with evidence retrieved from high-quality sources and generated based on user preferences, adjusting the critical attributes of evidence and argument style. The resultant Counterfire corpus comprises arguments generated from GPT-3.5 turbo, Koala, and PaLM 2 models and two of their finetuned variants (N = 32,000). Model evaluation indicates strong paraphrasing abilities with evidence, albeit limited word overlap, while demonstrating high style integration (0.9682 for 'reciprocity'), showing the ability of LLM to assimilate diverse styles. Of all models, GPT-3.5 turbo showed the highest scores in argument quality evaluation, showing consistent accuracy (score >0.8). In further analyses, reciprocity-style counterargument
    
[^15]: 数据到文本自然语言生成研究的系统性回顾

    A Systematic Review of Data-to-Text NLG

    [https://arxiv.org/abs/2402.08496](https://arxiv.org/abs/2402.08496)

    这篇系统性回顾全面分析了数据到文本自然语言生成研究的现状，提出未来方向，并解决了相关挑战。

    

    这篇系统性回顾旨在全面分析数据到文本生成研究的现状，重点是确定研究空白，提供未来方向，并解决回顾中发现的挑战。我们对文献进行了全面的检查，包括方法、数据集、评估指标、应用、多语言性和幻觉缓解措施。我们的回顾为这个快速发展的领域的未来研究提供了路线图。

    This systematic review aims to provide a comprehensive analysis of the state of data-to-text generation research, focusing on identifying research gaps, offering future directions, and addressing challenges found during the review. We thoroughly examined the literature, including approaches, datasets, evaluation metrics, applications, multilingualism, and hallucination mitigation measures. Our review provides a roadmap for future research in this rapidly evolving field.
    
[^16]: 可信的取样合理化通过半监督的蕴涵信号

    Plausible Extractive Rationalization through Semi-Supervised Entailment Signal

    [https://arxiv.org/abs/2402.08479](https://arxiv.org/abs/2402.08479)

    本文通过半监督方法，采用蕴涵对齐，以优化可行性，提取有理的方式提供一个可解释的替代模型

    

    复杂和不透明的黑盒子模型的增加需要采用可解释的措施，其中一种选择是提取有理的模型，它们作为更可解释的替代方案。这些模型，也称为先解释然后预测模型，使用解释模型来提取有理，然后使用提取的信息来调整预测模型。它们的主要目标是提供精确和忠实的解释，由提取的有理表示。在本文中，我们采用半监督方法来优化提取有理的可行性。我们采用一个预训练的自然语言推理（NLI）模型，并在一个小型的有监督有理集（10%）上进一步微调它。通过蕴涵对齐，NLI预测模型被利用作为解释模型的一种监督信号源。通过在问答任务中强制解释和答案之间的对齐一致，我们证明了性能得到了提升。

    The increasing use of complex and opaque black box models requires the adoption of interpretable measures, one such option is extractive rationalizing models, which serve as a more interpretable alternative. These models, also known as Explain-Then-Predict models, employ an explainer model to extract rationales and subsequently condition the predictor with the extracted information. Their primary objective is to provide precise and faithful explanations, represented by the extracted rationales. In this paper, we take a semi-supervised approach to optimize for the plausibility of extracted rationales. We adopt a pre-trained natural language inference (NLI) model and further fine-tune it on a small set of supervised rationales ($10\%$). The NLI predictor is leveraged as a source of supervisory signals to the explainer via entailment alignment. We show that, by enforcing the alignment agreement between the explanation and answer in a question-answering task, the performance can be improve
    
[^17]: 胡乱造谣：绕过ChatGPT的防护措施，大规模生成难以检测的虚假信息声明

    Lying Blindly: Bypassing ChatGPT's Safeguards to Generate Hard-to-Detect Disinformation Claims at Scale

    [https://arxiv.org/abs/2402.08467](https://arxiv.org/abs/2402.08467)

    本研究探索了ChatGPT在生成关于乌克兰战争的虚假信息方面的能力，发现它可以以较低成本、快速且大规模地生成逼真的定制虚假信息，而且这些虚假信息很难被人类读者和现有的自动化工具可靠地区分出来。

    

    随着大型语言模型（LLM）变得越来越熟练，它们在大规模病毒式虚假信息活动中的滥用成为一个越来越严重的问题。本研究探讨了ChatGPT生成关于乌克兰战争的无条件声明的能力，这是一个超出其知识界限的事件，并评估这些声明是否可以被人类读者和自动化工具与人类编写的声明区分出来。我们比较了ClaimReview中关于战争的声明，这些声明是由IFCN注册的事实核查员撰写的，以及ChatGPT生成的类似的短篇内容。我们证明，ChatGPT可以快速、廉价且规模化地生成逼真且针对特定目标的虚假信息，而且这些声明人类和现有的自动化工具无法可靠地区分出来。

    As Large Language Models (LLMs) become more proficient, their misuse in large-scale viral disinformation campaigns is a growing concern. This study explores the capability of ChatGPT to generate unconditioned claims about the war in Ukraine, an event beyond its knowledge cutoff, and evaluates whether such claims can be differentiated by human readers and automated tools from human-written ones. We compare war-related claims from ClaimReview, authored by IFCN-registered fact-checkers, and similar short-form content generated by ChatGPT. We demonstrate that ChatGPT can produce realistic, target-specific disinformation cheaply, fast, and at scale, and that these claims cannot be reliably distinguished by humans or existing automated tools.
    
[^18]: LLMs和人类条件

    LLMs and the Human Condition

    [https://arxiv.org/abs/2402.08403](https://arxiv.org/abs/2402.08403)

    本文提出了将三个成熟的人类决策理论整合到一起，形成了一个目的性人类行动模型。同时，将语言作为行动的观点应用于对话用户界面。通过理解ChatGPT的智能来源，可以在减少资源的同时获得对我们之间关系的认识。

    

    本文介绍了人类决策的三个成熟理论，并描述了如何将它们整合起来提供一个目的性人类行动的模型。同时，将语言作为行动的观点应用于对话用户界面。最近，基于理论的人工智能研究遇到了困难，本文旨在重新激发对理解LLMs实际执行的兴趣，而不仅仅是在所有数据上运行难以理解的机器学习例程。当一台售价不到50美元的树莓派电脑比第一台商业Cray超级计算机快400倍时，大型科技公司可以接近拥有无数随机打字并生成有意义文字的猴子。通过理解ChatGPT的表现智能的来源，也许我们可以用更少的资源进行同样的魔术，并在此过程中获得一些关于我们之间关系的理解。

    This paper presents three established theories of human decision-making and describes how they can be integrated to provide a model of purposive human action. Taking seriously the idea of language as action the model is then applied to the conversational user interfaces. Theory based AI research has had a hard time recently and the aim here is to revitalise interest in understanding what LLMs are actually doing other than running poorly understood machine learning routines over all the data the relevant Big Tech company can hoover up. When a raspberry pi computer for under 50USD is up to 400 times faster than the first commercial Cray super computer~\cite{crayVpi}, Big Tech can get really close to having an infinite number of monkeys typing at random and producing text, some of which will make sense. By understanding where ChatGPT's apparent intelligence comes from, perhaps we can perform the magic with fewer resources and at the same time gain some understanding about our relationship
    
[^19]: 作为Minecraft代理的大型语言模型

    Large Language Models as Minecraft Agents

    [https://arxiv.org/abs/2402.08392](https://arxiv.org/abs/2402.08392)

    本文研究了在Minecraft代理环境中使用大型语言模型（LLMs）的应用和评估，探讨了建造者和建筑师设置中的挑战和机遇，提出了澄清问题，并介绍了与代理进行在线交互的平台。

    

    在这项工作中，我们研究了将大型语言模型（LLMs）应用于充当Minecraft代理的具有挑战性的环境。我们在建造者和建筑师设置中应用和评估LLMs，引入澄清问题，并研究改进的挑战和机遇。此外，我们还提出了一个用于与代理进行在线交互的平台，并对之前的工作进行了评估。

    In this work we examine the use of Large Language Models (LLMs) in the challenging setting of acting as a Minecraft agent. We apply and evaluate LLMs in the builder and architect settings, introduce clarification questions and examining the challenges and opportunities for improvement. In addition, we present a platform for online interaction with the agents and an evaluation against previous works.
    
[^20]: 标点符号恢复在没有监督的情况下改善结构理解

    Punctuation Restoration Improves Structure Understanding without Supervision

    [https://arxiv.org/abs/2402.08382](https://arxiv.org/abs/2402.08382)

    标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。

    

    无监督学习目标，如语言建模和去噪等，在生成预训练模型方面起着重要作用，这些预训练模型能够执行从自然语言理解到会话任务的各种下游应用。然而，尽管最近的大型语言模型具有令人印象深刻的对话能力，但它们在捕捉文本的句法或语义结构方面的能力仍然落后。我们假设，语言性能和机器能力之间的不匹配归因于当前流行的预训练目标未能充分传递语言结构知识给计算系统。我们展示了标点符号恢复对结构相关任务的内部和外部表现的改善，如命名实体识别、开放式信息提取、分块和词性标注。标点符号恢复是一个有效的学习目标，可以改善结构理解并产生更加鲁棒的模型。

    Unsupervised learning objectives like language modeling and de-noising constitute a significant part in producing pre-trained models that perform various downstream applications from natural language understanding to conversational tasks. However, despite impressive conversational capabilities of recent large language model, their abilities to capture syntactic or semantic structure within text lag behind. We hypothesize that the mismatch between linguistic performance and competence in machines is attributable to insufficient transfer of linguistic structure knowledge to computational systems with currently popular pre-training objectives. We show that punctuation restoration transfers to improvements in in- and out-of-distribution performance on structure-related tasks like named entity recognition, open information extraction, chunking, and part-of-speech tagging. Punctuation restoration is an effective learning objective that can improve structure understanding and yield a more rob
    
[^21]: 基于真实用户查询评估文本到SQL系统的数据模型鲁棒性

    Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries

    [https://arxiv.org/abs/2402.08349](https://arxiv.org/abs/2402.08349)

    本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。

    

    文本到SQL系统（也称为自然语言到SQL系统）已成为弥合用户能力与基于SQL的数据访问之间差距的越来越流行的解决方案。这些系统将用户的自然语言请求转化为特定数据库的有效SQL语句。最近的基于转换器的语言模型使得文本到SQL系统受益匪浅。然而，虽然这些系统在常常是合成基准数据集上不断取得新的高分，但对于它们在真实世界、现实场景中对不同数据模型的鲁棒性的系统性探索明显缺乏。本文基于一个多年国际项目关于文本到SQL界面的集中评估，提供了对文本到SQL系统在实践中数据模型鲁棒性的首次深度评估。我们的评估基于FootballDB的真实部署，该系统在FIFA World Cup的背景下连续运行了9个月。

    Text-to-SQL systems (also known as NL-to-SQL systems) have become an increasingly popular solution for bridging the gap between user capabilities and SQL-based data access. These systems translate user requests in natural language to valid SQL statements for a specific database. Recent Text-to-SQL systems have benefited from the rapid improvement of transformer-based language models. However, while Text-to-SQL systems that incorporate such models continuously reach new high scores on -- often synthetic -- benchmark datasets, a systematic exploration of their robustness towards different data models in a real-world, realistic scenario is notably missing. This paper provides the first in-depth evaluation of the data model robustness of Text-to-SQL systems in practice based on a multi-year international project focused on Text-to-SQL interfaces. Our evaluation is based on a real-world deployment of FootballDB, a system that was deployed over a 9 month period in the context of the FIFA Wor
    
[^22]: 用分类器驱动的方法揭示大型语言模型中的五大人格特质：文本分析

    Eliciting Big Five Personality Traits in Large Language Models: A Textual Analysis with Classifier-Driven Approach

    [https://arxiv.org/abs/2402.08341](https://arxiv.org/abs/2402.08341)

    本研究使用分类器驱动的方法，通过不同的输入提示探究大型语言模型的输出变化，以增加其透明度。结果显示，这些模型根据输入的不同提示而表现出不同的人格特质，类似于人类对刺激做出的反应。

    

    大型语言模型（LLMs）在招聘背景下被应聘者和雇主广泛使用，然而这也引发了众多伦理问题，特别是与这些“黑盒子”模型缺乏透明度有关。尽管先前的研究试图通过调查LLMs的人格特质来增加其透明度，但许多先前的研究都要求模型来完成人格评估。相反，本研究旨在通过检查不同输入提示下模型的输出变化来更好地理解这些模型。具体来说，我们使用从常见面试问题和旨在引发特定的五大人格特质的提示来进行新颖的调查方法，以检查模型是否像人类一样容易激活特定人格特质，并根据其输出中的语言来评估其人格。为此，我们反复提供提示。

    Large Language Models (LLMs) are increasingly being utilized by both candidates and employers in the recruitment context. However, with this comes numerous ethical concerns, particularly related to the lack of transparency in these "black-box" models. Although previous studies have sought to increase the transparency of these models by investigating the personality traits of LLMs, many of the previous studies have provided them with personality assessments to complete. On the other hand, this study seeks to obtain a better understanding of such models by examining their output variations based on different input prompts. Specifically, we use a novel elicitation approach using prompts derived from common interview questions, as well as prompts designed to elicit particular Big Five personality traits to examine whether the models were susceptible to trait-activation like humans are, to measure their personality based on the language used in their outputs. To do so, we repeatedly prompte
    
[^23]: PreFLMR: 扩展细粒度迟交互多模态检索器

    PreFLMR: Scaling Up Fine-Grained Late-Interaction Multi-modal Retrievers

    [https://arxiv.org/abs/2402.08327](https://arxiv.org/abs/2402.08327)

    PreFLMR是一种扩展细粒度迟交互多模态检索器，用于解决知识式视觉问答任务。该方法通过训练和评估框架M2KR进行了开发，并在多个任务中取得了新的最先进结果。此外，还对PreFLMR的扩展行为进行了研究，为通用多模态检索器的未来发展提供了有用的启示。

    

    大型多模态模型(LMMs)在自然语言和视觉理解方面表现出色，但在诸如基于知识的视觉问答(KB-VQA)这样的严格任务中，却面临着从文档集合中检索相关信息以用于塑造问题答案的挑战。我们提出了一个广泛的训练和评估框架M2KR，用于KB-VQA。M2KR包含了一系列的视觉和语言任务，我们将其整合为一个用于训练和评估通用多模态检索器的基准任务套件。我们使用M2KR开发了PreFLMR，这是最近开发的细粒度迟交互多模态检索器(FLMR)方法的预训练版本，并且我们报告了一系列任务中的新的最先进结果。我们还对PreFLMR的扩展行为进行了研究，旨在对未来发展的通用多模态检索器有所帮助。

    Large Multimodal Models (LMMs) excel in natural language and visual understanding but are challenged by exacting tasks such as Knowledge-based Visual Question Answering (KB-VQA) which involve the retrieval of relevant information from document collections to use in shaping answers to questions. We present an extensive training and evaluation framework, M2KR, for KB-VQA. M2KR contains a collection of vision and language tasks which we have incorporated into a single suite of benchmark tasks for training and evaluating general-purpose multi-modal retrievers. We use M2KR to develop PreFLMR, a pre-trained version of the recently developed Fine-grained Late-interaction Multi-modal Retriever (FLMR) approach to KB-VQA, and we report new state-of-the-art results across a range of tasks. We also present investigations into the scaling behaviors of PreFLMR intended to be useful in future developments in general-purpose multi-modal retrievers.
    
[^24]: 《童话中明确表达的社会价值观：三种欧洲文化的比较》

    Explicit References to Social Values in Fairy Tales: A Comparison between Three European Cultures

    [https://arxiv.org/abs/2402.08318](https://arxiv.org/abs/2402.08318)

    研究了葡萄牙、意大利和德国童话中明确表达的价值观差异，使用词嵌入技术和罗盘量化分析。初步发现表明这些国家之间存在共享的文化理解和对善良、遵从和普遍价值观的表达。

    

    研究童话中的社会价值观可以了解价值观在时空中的传递。我们提出使用词嵌入技术和罗盘来量化葡萄牙、意大利和德国童话中的价值观传递。我们研究这三种国家的童话在明确表达价值观方面的差异。为此，我们指定了一个充满价值观的词汇列表，考虑它们的词干，并分析在专门预训练的Word2Vec模型中它们之间的距离。我们通过多角度验证和批判性讨论量化模型所提出的假设的有效性。我们认为，这是一个可复用和可重现的方法来研究历史语料库中明确引用的价值观。最后，我们的初步发现暗示有着共享文化理解和对善良、遵从和普遍价值观的表达。

    The study of social values in fairy tales opens the possibility to learn about the communication of values across space and time. We propose to study the communication of values in fairy tales from Portugal, Italy and Germany using a technique called word embedding with a compass to quantify vocabulary differences and commonalities. We study how these three national traditions of fairy tales differ in their explicit references to values. To do this, we specify a list of value-charged tokens, consider their word stems and analyse the distance between these in a bespoke pre-trained Word2Vec model. We triangulate and critically discuss the validity of the resulting hypotheses emerging from this quantitative model. Our claim is that this is a reusable and reproducible method for the study of the values explicitly referenced in historical corpora. Finally, our preliminary findings hint at a shared cultural understanding and the expression of values such as Benevolence, Conformity, and Unive
    
[^25]: 通过提示的上下文向量检测钓鱼网络攻击

    Prompted Contextual Vectors for Spear-Phishing Detection

    [https://arxiv.org/abs/2402.08309](https://arxiv.org/abs/2402.08309)

    通过新的文档向量化方法，我们的方法使用大型语言模型来检测钓鱼网络攻击的电子邮件，并在实验证明具有高效性能。

    

    钓鱼网络攻击是一个重大的安全挑战，而大型语言模型（LLMs）通过生成令人信服的电子邮件并方便目标侦察来升级了威胁。为了解决这个问题，我们提出了一种基于新颖文档向量化方法的检测方法，该方法利用LLMs的集合来创建表示向量。通过提示LLMs来推理和回答人工制定的问题，我们量化电子邮件内容中常见说服原则的存在，为下游监督机器学习模型生成提示上下文文档向量。我们使用一个专有系统生成的独特数据集来评估我们的方法，该系统自动化目标侦察和钓鱼电子邮件的创建。我们的方法在仅包含传统钓鱼和良性电子邮件的训练集中实现了91%的F1得分，其中关键贡献包括一种创新的文档向量化方法。

    Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizin
    
[^26]: ChatCell: 利用自然语言促进单细胞分析

    ChatCell: Facilitating Single-Cell Analysis with Natural Language

    [https://arxiv.org/abs/2402.08303](https://arxiv.org/abs/2402.08303)

    ChatCell是一个利用自然语言促进单细胞分析的工具，通过词汇适应和统一序列生成，它具备深厚的专业知识和适应各种分析任务的能力。

    

    随着大型语言模型(LLMs)的快速发展，它们在科学中的影响日益突出。LLMs在任务泛化和自由对话方面的新兴能力可以极大地推进化学和生物学等领域。然而，单细胞生物学这个构成生物体基础构件的领域仍面临一些挑战。当前方法在知识门槛和可扩展性方面存在限制，阻碍了LLMs在掌握单细胞数据方面的充分利用，影响了直接可访问和快速迭代的能力。为此，我们引入了ChatCell，通过利用词汇适应和统一序列生成，它在单细胞生物学领域获得了深厚的专业知识和适应各种分析任务的能力，标志着一种范式转变。

    As Large Language Models (LLMs) rapidly evolve, their influence in science is becoming increasingly prominent. The emerging capabilities of LLMs in task generalization and free-form dialogue can significantly advance fields like chemistry and biology. However, the field of single-cell biology, which forms the foundational building blocks of living organisms, still faces several challenges. High knowledge barriers and limited scalability in current methods restrict the full exploitation of LLMs in mastering single-cell data, impeding direct accessibility and rapid iteration. To this end, we introduce ChatCell, which signifies a paradigm shift by facilitating single-cell analysis with natural language. Leveraging vocabulary adaptation and unified sequence generation, ChatCell has acquired profound expertise in single-cell biology and the capability to accommodate a diverse range of analysis tasks. Extensive experiments further demonstrate ChatCell's robust performance and potential to de
    
[^27]: 朝着忠实和强大的基于证据的问答专家的方向前进

    Towards Faithful and Robust LLM Specialists for Evidence-Based Question-Answering

    [https://arxiv.org/abs/2402.08277](https://arxiv.org/abs/2402.08277)

    这项工作探索了如何鲁棒地微调大型语言模型以提高答案的来源质量和答案归因能力，引入了数据生成流水线和四个测试集来评估模型的性能，并展示了在合成数据上微调可以改善内部和外部分布的性能。

    

    对大型语言模型（LLM）更忠实和可追踪的答案的进步对于各种研究和实践活动至关重要。其中一种达到这个目标的方法是基于可靠的来源提供答案。然而，这种基于证据的问答在使用LLM时已经证明在引用正确的来源（来源质量）和准确地表示来源中的信息（答案归因能力）方面工作不足。在这项工作中，我们系统地研究了如何鲁棒地微调LLM，以提高来源质量和答案归因能力。具体而言，我们引入了一个数据生成流水线，其中包括自动数据质量过滤器，可以大规模合成多样化的高质量训练和测试数据。我们还引入了四个测试集，以对微调后的专家模型的鲁棒性进行基准测试。广泛的评估结果表明，在合成数据上进行微调可以提高在内部和外部分布的性能。%基于证据的问答案例。此外，我们展示了用于评估的四个测试集，以评估微调后的专家模型的鲁棒性。

    Advances towards more faithful and traceable answers of Large Language Models (LLMs) are crucial for various research and practical endeavors. One avenue in reaching this goal is basing the answers on reliable sources. However, this Evidence-Based QA has proven to work insufficiently with LLMs in terms of citing the correct sources (source quality) and truthfully representing the information within sources (answer attributability). In this work, we systematically investigate how to robustly fine-tune LLMs for better source quality and answer attributability. Specifically, we introduce a data generation pipeline with automated data quality filters, which can synthesize diversified high-quality training and testing data at scale. We further introduce four test sets to benchmark the robustness of fine-tuned specialist models. Extensive evaluation shows that fine-tuning on synthetic data improves performance on both in- and out-of-distribution. %Evidence-Based QA cases. Furthermore, we sho
    
[^28]: 大规模语言模型在表格推理中的调查

    A Survey of Table Reasoning with Large Language Models

    [https://arxiv.org/abs/2402.08259](https://arxiv.org/abs/2402.08259)

    这篇论文调查了使用大规模语言模型进行表格推理的现有研究，总结了LLM在表格推理中的优势，并探讨了未来增强表格推理能力的方向。

    

    表格推理旨在根据提供的表格和可选的文本描述，根据用户要求生成相应的问题答案，有效提高获取信息的效率。最近，使用大规模语言模型(LLMs)已成为表格推理的主流方法，因为它不仅显著降低了注释成本，还超过了先前方法的性能。然而，现有研究仍缺乏LLM-based表格推理工作的总结。由于现有研究的缺乏，关于在LLM时代如何提高表格推理性能的技术，为什么LLMs在表格推理方面表现出色，以及如何在未来增强表格推理能力的问题仍然大部分未被探索。这一差距严重限制了研究的进展。为了回答上述问题并推动基于LLMs的表格推理研究，我们提出这个调查分析现有的研究，激发创新。

    Table reasoning, which aims to generate the corresponding answer to the question following the user requirement according to the provided table, and optionally a text description of the table, effectively improving the efficiency of obtaining information. Recently, using Large Language Models (LLMs) has become the mainstream method for table reasoning, because it not only significantly reduces the annotation cost but also exceeds the performance of previous methods. However, existing research still lacks a summary of LLM-based table reasoning works. Due to the existing lack of research, questions about which techniques can improve table reasoning performance in the era of LLMs, why LLMs excel at table reasoning, and how to enhance table reasoning abilities in the future, remain largely unexplored. This gap significantly limits progress in research. To answer the above questions and advance table reasoning research with LLMs, we present this survey to analyze existing research, inspirin
    
[^29]: 使用实例混淆的隐私保护语言模型推断

    Privacy-Preserving Language Model Inference with Instance Obfuscation

    [https://arxiv.org/abs/2402.08227](https://arxiv.org/abs/2402.08227)

    本论文提出了一种名为实例混淆推断（IOI）的方法，该方法致力于解决语言模型推断中决策隐私问题，通过对输入数据进行转换和保护，实现了在维持模型黑盒的同时保护决策隐私。

    

    语言模型作为一种服务（LMaaS），为开发者和研究人员提供了便利的访问方式，能够使用预训练的语言模型进行推断。然而，在服务调用过程中，输入数据和推断结果等包含私人信息的内容以明文形式暴露出来，导致了隐私问题。最近的研究开始通过从用户端开始的技术，如噪声添加和内容扰动，将输入数据转换为隐私保护的表示形式来解决隐私问题，然而推断结果的保护，即决策隐私，仍然是一个空白页。为了维持LMaaS的黑盒方式，进行数据隐私保护，尤其是对于决策的保护，是一项具有挑战性的任务，因为该过程必须对模型来说是无缝的，并伴随着有限的通信和计算开销。因此，我们提出了一种名为实例混淆推断（IOI）的方法，重点解决了决策隐私的问题。

    Language Models as a Service (LMaaS) offers convenient access for developers and researchers to perform inference using pre-trained language models. Nonetheless, the input data and the inference results containing private information are exposed as plaintext during the service call, leading to privacy issues. Recent studies have started tackling the privacy issue by transforming input data into privacy-preserving representation from the user-end with the techniques such as noise addition and content perturbation, while the exploration of inference result protection, namely decision privacy, is still a blank page. In order to maintain the black-box manner of LMaaS, conducting data privacy protection, especially for the decision, is a challenging task because the process has to be seamless to the models and accompanied by limited communication and computation overhead. We thus propose Instance-Obfuscated Inference (IOI) method, which focuses on addressing the decision privacy issue of na
    
[^30]: BBox-Adapter: 轻量级适配黑盒大型语言模型

    BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models

    [https://arxiv.org/abs/2402.08219](https://arxiv.org/abs/2402.08219)

    BBox-Adapter是一种适用于黑盒大型语言模型的轻量级适配器，通过区分目标和源域数据，并采用排名式噪音对比估计（NCE）损失和在线适应机制，实现了在透明、隐私和成本方面的有效适应。

    

    适应最先进的大型语言模型（LLMs），如GPT-4和Gemini，以满足特定任务的要求是具有挑战性的。由于它们的参数、嵌入和输出概率的不透明性，现有的微调适应方法是不适用的。因此，只能通过它们的API服务适应这些黑盒LLMs，这引发了透明度、隐私和成本的担忧。为了解决这些挑战，我们介绍了BBox-Adapter，一种新颖的适用于黑盒LLMs的轻量级适配器。BBox-Adapter通过将目标数据视为正样本，将源数据视为负样本来区分目标和源域数据。它采用基于排名的噪音对比估计（NCE）损失来提高目标域数据的可能性，同时惩罚源域数据的可能性。此外，它还具有在线适应机制，该机制将来自真实数据、人类或AI反馈的实时正样本采样与先前适应的负样本数据相结合。广泛的实验表明，BBox-Adapter在不降低性能的同时，提供了高效而灵活的黑盒LLMs适应解决方案。

    Adapting state-of-the-art Large Language Models (LLMs) like GPT-4 and Gemini for specific tasks is challenging. Due to the opacity in their parameters, embeddings, and even output probabilities, existing fine-tuning adaptation methods are inapplicable. Consequently, adapting these black-box LLMs is only possible through their API services, raising concerns about transparency, privacy, and cost. To address these challenges, we introduce BBox-Adapter, a novel lightweight adapter for black-box LLMs. BBox-Adapter distinguishes target and source domain data by treating target data as positive and source data as negative. It employs a ranking-based Noise Contrastive Estimation (NCE) loss to promote the likelihood of target domain data while penalizing that of the source domain. Furthermore, it features an online adaptation mechanism, which incorporates real-time positive data sampling from ground-truth, human, or AI feedback, coupled with negative data from previous adaptations. Extensive ex
    
[^31]: 像素句子表示学习

    Pixel Sentence Representation Learning

    [https://arxiv.org/abs/2402.08183](https://arxiv.org/abs/2402.08183)

    本文提出了一种无监督的视觉句子表示学习框架，通过引入基于视觉基础文本扰动方法，如打字错误和单词顺序混排，借鉴认知和语言科学，以连续的方式感知文本的扰动，从而改善预训练语言模型在捕捉句子级文本语义方面的性能。

    

    预训练语言模型长期以来被认为在捕捉句子和文档级语义方面表现不佳。尽管已经进行了大量研究，但从无监督的视觉表示学习中将基于扰动的方法转移到自然语言处理仍然是一个尚未解决的问题。这在很大程度上是由于语言模型的分词所引入的子词单元的离散性，限制了对输入进行小扰动以形成保留语义的正向对。在这项工作中，我们将学习句子级文本语义视作一个视觉表示学习过程。借鉴认知和语言科学，我们引入了一个无监督的视觉句子表示学习框架，利用像打字错误和单词顺序混排等基于视觉基础文本扰动方法，与人类认知模式共鸣，并使文本的扰动被感知为连续。我们的方法进一步加强了大规模无监督的主题对齐训练和自然语言文本的仿真可视化条件。

    Pretrained language models are long known to be subpar in capturing sentence and document-level semantics. Though heavily investigated, transferring perturbation-based methods from unsupervised visual representation learning to NLP remains an unsolved problem. This is largely due to the discreteness of subword units brought by tokenization of language models, limiting small perturbations of inputs to form semantics-preserved positive pairs. In this work, we conceptualize the learning of sentence-level textual semantics as a visual representation learning process. Drawing from cognitive and linguistic sciences, we introduce an unsupervised visual sentence representation learning framework, employing visually-grounded text perturbation methods like typos and word order shuffling, resonating with human cognitive patterns, and enabling perturbation to texts to be perceived as continuous. Our approach is further bolstered by large-scale unsupervised topical alignment training and natural la
    
[^32]: CMA-R：用于解释谣言检测的因果中介分析

    CMA-R:Causal Mediation Analysis for Explaining Rumour Detection

    [https://arxiv.org/abs/2402.08155](https://arxiv.org/abs/2402.08155)

    CMA-R通过因果中介分析解释了神经模型在Twitter上进行谣言检测的决策过程，并能够识别出关键推文和因果影响单词，提高了对黑盒子谣言检测系统的解释性和透明度。

    

    我们将因果中介分析应用于解释神经模型在Twitter上进行谣言检测的决策过程。在输入和网络层面进行干预可以揭示模型输出中推文和单词的因果影响。我们发现我们的方法CMA-R - 因果中介分析用于谣言检测 - 可以识别出解释模型预测并与人类判断关于故事真实性的关键推文的显著性推文，并展示出强烈的一致性。CMA-R还可以突出显著性推文中具有因果影响的单词，提供对这些黑盒子谣言检测系统的另一层解释性和透明度。代码可在此处找到：https://github.com/ltian678/cma-r.

    We apply causal mediation analysis to explain the decision-making process of neural models for rumour detection on Twitter. Interventions at the input and network level reveal the causal impacts of tweets and words in the model output. We find that our approach CMA-R -- Causal Mediation Analysis for Rumour detection -- identifies salient tweets that explain model predictions and show strong agreement with human judgements for critical tweets determining the truthfulness of stories. CMA-R can further highlight causally impactful words in the salient tweets, providing another layer of interpretability and transparency into these blackbox rumour detection systems. Code is available at: https://github.com/ltian678/cma-r.
    
[^33]: 大型语言模型的主动偏好学习

    Active Preference Learning for Large Language Models

    [https://arxiv.org/abs/2402.08114](https://arxiv.org/abs/2402.08114)

    本论文提出了一种用于大型语言模型的主动偏好学习策略，通过直接偏好优化（DPO）来更好地利用偏好标签。实验结果表明，该方法提高了基于成对偏好数据的微调的学习速度和最终性能。

    

    随着大型语言模型（LLM）的能力越来越强，与人类意图对齐的微调技术变得越来越重要。对于对齐这些模型来说，最关键的考虑是如何最有效地利用人力资源，或者在LLM本身被用作oracle的情况下如何最有效地利用模型资源。从人类或AI偏好中进行强化学习（RLHF / RLAIF）是这种技术最突出的例子，但它往往复杂且不稳定。最近，直接偏好优化（DPO）被提出作为一个更简单和更稳定的替代方法。在这项工作中，我们开发了一种DPO的主动学习策略，以更好地利用偏好标签。我们提出了一个基于语言模型的预测熵和DPO优化的隐式偏好模型的确定性度量的实用采集函数，展示了我们的方法如何提高基于成对偏好数据的微调的学习速度和最终性能。

    As large language models (LLMs) become more capable, fine-tuning techniques for aligning with human intent are increasingly important. A key consideration for aligning these models is how to most effectively use human resources, or model resources in the case where LLMs themselves are used as oracles. Reinforcement learning from Human or AI preferences (RLHF/RLAIF) is the most prominent example of such a technique, but is complex and often unstable. Direct Preference Optimization (DPO) has recently been proposed as a simpler and more stable alternative. In this work, we develop an active learning strategy for DPO to make better use of preference labels. We propose a practical acquisition function for prompt/completion pairs based on the predictive entropy of the language model and a measure of certainty of the implicit preference model optimized by DPO. We demonstrate how our approach improves both the rate of learning and final performance of fine-tuning on pairwise preference data.
    
[^34]: 解决医学语言模型中的认知偏见问题

    Addressing cognitive bias in medical language models

    [https://arxiv.org/abs/2402.08113](https://arxiv.org/abs/2402.08113)

    本研究通过开发BiasMedQA，一个用于评估医学任务中LLMs的认知偏见的新型基准，发现LLMs在面对包含认知偏见的临床问题时，其回答的准确性明显降低。

    

    将大型语言模型（LLMs）整合到医学领域已经引起了重大关注，因为它们在模拟临床决策场景中的准确性很有前景。然而，临床决策比模拟更复杂，因为医生的决策受到许多因素的影响，包括认知偏见的存在。然而，LLMs在面对包含认知偏见的临床问题时，与不包含这些偏见的问题相比，其回答的准确性会明显降低，这一问题尚未被探索。本研究的假设认为，当LLMs面对包含认知偏见的临床问题时，与不包含这些偏见的问题相比，其回答的准确性会明显降低。我们开发了BiasMedQA，这是一个用于评估LLMs在医学任务中的认知偏见的新型基准。使用BiasMedQA，我们评估了六个LLMs，分别是GPT-4、Mixtral-8x70B、GPT-3.5、PaLM-2、Llama 2 70B-chat和医学专业的PMC Llama 13B。我们在127个临床问题上测试了这些模型。

    The integration of large language models (LLMs) into the medical field has gained significant attention due to their promising accuracy in simulated clinical decision-making settings. However, clinical decision-making is more complex than simulations because physicians' decisions are shaped by many factors, including the presence of cognitive bias. However, the degree to which LLMs are susceptible to the same cognitive biases that affect human clinicians remains unexplored. Our hypothesis posits that when LLMs are confronted with clinical questions containing cognitive biases, they will yield significantly less accurate responses compared to the same questions presented without such biases.In this study, we developed BiasMedQA, a novel benchmark for evaluating cognitive biases in LLMs applied to medical tasks. Using BiasMedQA we evaluated six LLMs, namely GPT-4, Mixtral-8x70B, GPT-3.5, PaLM-2, Llama 2 70B-chat, and the medically specialized PMC Llama 13B. We tested these models on 1,27
    
[^35]: 研究大型语言模型在文本到SQL翻译中数据污染的影响

    Investigating the Impact of Data Contamination of Large Language Models in Text-to-SQL Translation

    [https://arxiv.org/abs/2402.08100](https://arxiv.org/abs/2402.08100)

    本研究调查了大型语言模型在文本到SQL翻译中数据污染的影响。通过引入一种新的检测方法，研究人员发现GPT-3.5在陌生数据集上的性能显著下降。此外，通过采用对抗性表断开方法，研究人员还分析了GPT-3.5在修改信息的数据库上的效果。

    

    理解文本描述以生成代码似乎是零-shot场景下指令遵循大型语言模型（LLM）的一项已实现的能力。然而，可能会严重影响这种翻译能力的因素是已经见过目标的文本描述和相关代码。这种影响被称为数据污染。  在本研究中，我们调查了数据污染对GPT-3.5在文本到SQL代码生成任务中性能的影响。因此，我们提出了一种新的方法来检测GPTs中的数据污染，并使用已知的Spider数据集和我们的新的陌生数据集Termite来检查GPT-3.5在文本到SQL任务中的表现。此外，我们通过采用对抗性表断开（ATD）方法分析了GPT-3.5在具有修改信息的数据库上的效果，通过从数据库中删除结构信息来使文本到SQL任务复杂化。我们的研究结果表明，GPT-3.5在陌生的Termite数据集上表现出显著的性能下降。

    Understanding textual description to generate code seems to be an achieved capability of instruction-following Large Language Models (LLMs) in zero-shot scenario. However, there is a severe possibility that this translation ability may be influenced by having seen target textual descriptions and the related code. This effect is known as Data Contamination.   In this study, we investigate the impact of Data Contamination on the performance of GPT-3.5 in the Text-to-SQL code-generating tasks. Hence, we introduce a novel method to detect Data Contamination in GPTs and examine GPT-3.5's Text-to-SQL performances using the known Spider Dataset and our new unfamiliar dataset Termite. Furthermore, we analyze GPT-3.5's efficacy on databases with modified information via an adversarial table disconnection (ATD) approach, complicating Text-to-SQL tasks by removing structural pieces of information from the database. Our results indicate a significant performance drop in GPT-3.5 on the unfamiliar T
    
[^36]: 基于10万小时数据的10亿参数文本到语音模型的经验教训

    BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data

    [https://arxiv.org/abs/2402.08093](https://arxiv.org/abs/2402.08093)

    基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。

    

    我们介绍了一个名为BASE TTS的文本到语音（TTS）模型，其中BASE代表大规模自适应可流式TTS和新出现的能力。BASE TTS是迄今为止最大的TTS模型，训练于10万小时的公共领域语音数据，实现了语音自然度的最新技术水平。它采用了一个10亿参数的自回归Transformer，将原始文本转换为离散代码（"speechcodes"），然后通过基于卷积的解码器将这些speechcodes以增量、可流式的方式转换为波形。此外，我们的speechcodes采用了一种新颖的语音标记化技术，具有说话者ID解耦和字节对编码的压缩特性。与大量数据训练的大语言模型广泛报道的"新出现的能力"类似，我们展示了使用10K+小时和500M+参数构建的BASE TTS变体在文本复杂句子上开始展现自然的韵律。我们设计了...

    We introduce a text-to-speech (TTS) model called BASE TTS, which stands for $\textbf{B}$ig $\textbf{A}$daptive $\textbf{S}$treamable TTS with $\textbf{E}$mergent abilities. BASE TTS is the largest TTS model to-date, trained on 100K hours of public domain speech data, achieving a new state-of-the-art in speech naturalness. It deploys a 1-billion-parameter autoregressive Transformer that converts raw texts into discrete codes ("speechcodes") followed by a convolution-based decoder which converts these speechcodes into waveforms in an incremental, streamable manner. Further, our speechcodes are built using a novel speech tokenization technique that features speaker ID disentanglement and compression with byte-pair encoding. Echoing the widely-reported "emergent abilities" of large language models when trained on increasing volume of data, we show that BASE TTS variants built with 10K+ hours and 500M+ parameters begin to demonstrate natural prosody on textually complex sentences. We design
    
[^37]: 多模态学习中的文本中心对齐

    Text-centric Alignment for Multi-Modality Learning

    [https://arxiv.org/abs/2402.08086](https://arxiv.org/abs/2402.08086)

    本研究提出了一种名为文本中心对齐的多模态学习方法（TAMML），利用大型语言模型和基础模型，通过将文本作为统一的语义空间，解决了多模态学习中的模态不匹配问题，并在处理未见过的、多样化的、不可预测的模态组合时取得了显著的改进。

    

    本研究论文解决了多模态学习中的模态不匹配问题，即推理阶段可用的模态与训练阶段不同。我们提出了一种名为文本中心对齐的多模态学习（TAMML）方法，该方法利用大型语言模型（LLMs）进行上下文学习，并借助基础模型增强多模态系统在这些条件下的泛化能力。通过利用文本作为统一的语义空间的独特特性，TAMML在处理未见过的、多样化的、不可预测的模态组合方面展示出显著的改进。TAMML不仅能够适应不同的模态，还能保持稳健的性能，展示了基础模型在克服传统的固定模态框架中的表示嵌入限制方面的潜力。这项研究为领域提供了一种灵活有效的解决方案，适用于现实世界的应用，其中模态的可用性可能会变化。

    This research paper addresses the challenge of modality mismatch in multimodal learning, where the modalities available during inference differ from those available at training. We propose the Text-centric Alignment for Multi-Modality Learning (TAMML) approach, an innovative method that utilizes Large Language Models (LLMs) with in-context learning and foundation models to enhance the generalizability of multimodal systems under these conditions. By leveraging the unique properties of text as a unified semantic space, TAMML demonstrates significant improvements in handling unseen, diverse, and unpredictable modality combinations. TAMML not only adapts to varying modalities but also maintains robust performance, showcasing the potential of foundation models in overcoming the limitations of traditional fixed-modality frameworks in embedding representations. This study contributes to the field by offering a flexible, effective solution for real-world applications where modality availabili
    
[^38]: 大型语言模型作为两人游戏中的代理

    Large Language Models as Agents in Two-Player Games

    [https://arxiv.org/abs/2402.08078](https://arxiv.org/abs/2402.08078)

    通过将大型语言模型的训练过程重新概念化为基于语言的两人游戏中的代理学习，我们能够得到关键的见解，并提供了新的方法和技术来推进大型语言模型的发展。

    

    通过在一个统一的机器学习范式中正式定义大型语言模型（LLMs）的训练过程，该过程通常包括预训练、有监督微调和强化学习与人类反馈，在推进LLM技术方面可以获得关键性的见解。本文描述了LLM的训练方法与在博弈论、强化学习和多智能体系统中研究的两人游戏代理开发所采用的策略之间的相似之处。我们提出了一种将LLM学习过程重新概念化为基于语言的游戏中的代理学习的框架。这一框架揭示了LLM开发中的成功与挑战的创新视角，提供了对解决对齐问题和其他战略考虑的新理解。此外，我们的两人游戏方法为训练LLMs提供了新的数据准备和机器学习技术的启示。

    By formally defining the training processes of large language models (LLMs), which usually encompasses pre-training, supervised fine-tuning, and reinforcement learning with human feedback, within a single and unified machine learning paradigm, we can glean pivotal insights for advancing LLM technologies. This position paper delineates the parallels between the training methods of LLMs and the strategies employed for the development of agents in two-player games, as studied in game theory, reinforcement learning, and multi-agent systems. We propose a re-conceptualization of LLM learning processes in terms of agent learning in language-based games. This framework unveils innovative perspectives on the successes and challenges in LLM development, offering a fresh understanding of addressing alignment issues among other strategic considerations. Furthermore, our two-player game approach sheds light on novel data preparation and machine learning techniques for training LLMs.
    
[^39]: 超越LLMs：推进复杂推理的发展

    Beyond LLMs: Advancing the Landscape of Complex Reasoning

    [https://arxiv.org/abs/2402.08064](https://arxiv.org/abs/2402.08064)

    在人工智能领域，大型语言模型(LLMs)一直被视为解决许多问题的标准解决方案。然而，对于约束满足和优化问题，LLMs表现不佳。因此，Elemental Cognition开发了EC AI平台，采用神经符号方法解决这些问题，同时利用LLMs进行知识获取和用户交互。

    

    自几年前出现大型语言模型以来，它们往往被视为许多人工智能问题的事实解决方案。然而，除了LLMs的许多不足之外，如可靠性、成本和速度等问题，还有一类常见的现实世界问题使大型语言模型表现不佳，即约束满足和优化问题。这些问题无处不在，目前的解决方案非常专业化且实施成本高。在Elemental Cognition，我们开发了EC AI平台，该平台采用了神经符号方法来解决约束满足和优化问题。该平台的核心是一个精确高效的逻辑推理引擎，并利用LLMs进行知识获取和用户交互。该平台支持开发人员用自然简洁的语言指定应用逻辑，并生成应用用户界面以与用户进行交互。

    Since the advent of Large Language Models a few years ago, they have often been considered the de facto solution for many AI problems. However, in addition to the many deficiencies of LLMs that prevent them from broad industry adoption, such as reliability, cost, and speed, there is a whole class of common real world problems that Large Language Models perform poorly on, namely, constraint satisfaction and optimization problems. These problems are ubiquitous and current solutions are highly specialized and expensive to implement. At Elemental Cognition, we developed our EC AI platform which takes a neuro-symbolic approach to solving constraint satisfaction and optimization problems. The platform employs, at its core, a precise and high performance logical reasoning engine, and leverages LLMs for knowledge acquisition and user interaction. This platform supports developers in specifying application logic in natural and concise language while generating application user interfaces to int
    
[^40]: 不小心的耳语：语音转文本幻觉的危害

    Careless Whisper: Speech-to-Text Hallucination Harms

    [https://arxiv.org/abs/2402.08021](https://arxiv.org/abs/2402.08021)

    该论文评估了开放AI的语音识别服务Whisper，并指出其中约1%的转录存在完全幻觉的短语或句子。这些幻觉内容中有38%包含明确的伤害，如暴力、虚构的个人信息或虚假的基于视频的权威。研究者进一步提供了幻觉发生的假设，并指出了由于语音类型和健康状况的不同可能导致的潜在差异。他们呼吁行业从业者改善基于语言模型的幻觉，并增强对下游潜在偏见的认识。

    

    语音转文本服务旨在尽可能准确地转录输入音频。它们在日常生活中的作用越来越大，例如个人语音助手或公司与客户的互动中。我们评估了开放AI的Whisper，这是一种超越行业竞争对手的最新服务。虽然Whisper的许多转录非常准确，但我们发现大约1％的音频转录包含完全幻觉的短语或句子，这些短语或句子在基础音频中不存在。我们主题化地分析了Whisper幻觉的内容，发现38％的幻觉包含明确的伤害，例如暴力、虚构的个人信息或虚假的基于视频的权威。我们进一步提供了关于幻觉发生的假设，并揭示了由于健康状况而导致的语音类型的潜在差异。我们呼吁行业从业者改善Whisper中基于语言模型的幻觉，并增强对下游潜在偏见的认识。

    Speech-to-text services aim to transcribe input audio as accurately as possible. They increasingly play a role in everyday life, for example in personal voice assistants or in customer-company interactions. We evaluate Open AI's Whisper, a state-of-the-art service outperforming industry competitors. While many of Whisper's transcriptions were highly accurate, we found that roughly 1% of audio transcriptions contained entire hallucinated phrases or sentences, which did not exist in any form in the underlying audio. We thematically analyze the Whisper-hallucinated content, finding that 38% of hallucinations include explicit harms such as violence, made up personal information, or false video-based authority. We further provide hypotheses on why hallucinations occur, uncovering potential disparities due to speech type by health status. We call on industry practitioners to ameliorate these language-model-based hallucinations in Whisper, and to raise awareness of potential biases in downstr
    
[^41]: Lumos : 用场景文本识别增强多模式LLMs的能力

    Lumos : Empowering Multimodal LLMs with Scene Text Recognition

    [https://arxiv.org/abs/2402.08017](https://arxiv.org/abs/2402.08017)

    本论文介绍了Lumos，它是第一个具备文本理解能力的多模式问答系统，通过运用场景文本识别组件，能够从第一人称视角图像中提取文本，并将其用于加强多模式大型语言模型的输入。研究过程中，作者克服了与文本识别质量、延迟和模型推断相关的多个挑战，并提供了全面的组件评估结果，展示了高质量和高效率的性能。

    

    我们介绍了Lumos，它是第一个具备文本理解能力的端到端多模式问答系统。Lumos的核心是一个场景文本识别（STR）组件，用于从第一人称视角图像中提取文本，并将其用于增强多模式大型语言模型（MM-LLM）的输入。在构建Lumos的过程中，我们遇到了许多与STR质量、整体延迟和模型推断相关的挑战。在本文中，我们探讨了这些挑战，并讨论了用于克服这些障碍的系统架构、设计选择和建模技术。我们还对每个组件进行了全面评估，展示了高质量和高效率的性能。

    We introduce Lumos, the first end-to-end multimodal question-answering system with text understanding capabilities. At the core of Lumos is a Scene Text Recognition (STR) component that extracts text from first person point-of-view images, the output of which is used to augment input to a Multimodal Large Language Model (MM-LLM). While building Lumos, we encountered numerous challenges related to STR quality, overall latency, and model inference. In this paper, we delve into those challenges, and discuss the system architecture, design choices, and modeling techniques employed to overcome these obstacles. We also provide a comprehensive evaluation for each component, showcasing high quality and efficiency.
    
[^42]: 增强Amharic-LLaMA: 整合特定任务与生成数据集

    Enhancing Amharic-LLaMA: Integrating Task Specific and Generative Datasets

    [https://arxiv.org/abs/2402.08015](https://arxiv.org/abs/2402.08015)

    本研究通过整合任务特定和生成数据集来增强Amharic-LLaMA模型，提高了阿姆哈拉语言模型的性能。他们通过创建阿姆哈拉语指令微调数据集和微调模型，在不同的NLP任务中取得了有希望的结果。

    

    大型语言模型（LLM）因其在理解和生成人类语言方面的出色表现而在自然语言处理（NLP）研究中受到了很多关注。然而，资源匮乏的语言因缺乏资源而被落下。在这项工作中，我们致力于通过整合特定任务和生成数据集来增强LLaMA-2-Amharic模型，以提高阿姆哈拉语的语言模型性能。我们创建了一个阿姆哈拉语指令微调数据集，并对LLaMA-2-Amharic模型进行了微调。经过微调的模型在不同的NLP任务中表现出有希望的结果。我们开源了我们的数据集创建流程、指令数据集、训练模型和评估输出，以促进对这些模型的语言特定研究。

    Large language models (LLMs) have received a lot of attention in natural language processing (NLP) research because of their exceptional performance in understanding and generating human languages. However, low-resource languages are left behind due to the unavailability of resources. In this work, we focus on enhancing the LLaMA-2-Amharic model by integrating task-specific and generative datasets to improve language model performance for Amharic. We compile an Amharic instruction fine-tuning dataset and fine-tuned LLaMA-2-Amharic model. The fine-tuned model shows promising results in different NLP tasks. We open-source our dataset creation pipeline, instruction datasets, trained models, and evaluation outputs to promote language-specific studies on these models.
    
[^43]: 带有合成数据的改进型直接优化法用于LLM的行为调整

    Refined Direct Preference Optimization with Synthetic Data for Behavioral Alignment of LLMs

    [https://arxiv.org/abs/2402.08005](https://arxiv.org/abs/2402.08005)

    本文提出了一种改进的直接优化法（rDPO），通过使用合成数据来改善大规模语言模型（LLM）的行为调整。这种方法通过自我评论和广义DPO损失函数来优化学生LLM，并利用外部奖励模型提高合成数据质量，从而使rDPO在多个行为调整任务中表现出良好效果。

    

    本文介绍了一种改进型直接优化法（rDPO），用于改善大规模语言模型（LLM）的行为调整，无需人工标注数据。该方法通过自我评论提示教师LLM来创建合成数据，然后利用广义DPO损失函数来提纯给学生LLM。损失函数结合了额外的外部奖励模型，以提高合成数据的质量，使rDPO能够抵抗合成数据集中的潜在噪声。rDPO在多种行为调整任务中展现出良好效果，如提高安全性，抵抗角色扮演，降低巴结行为。代码将在https://github.com/vicgalle/refined-dpo上发布。

    In this paper, we introduce \emph{refined Direct Preference Optimization} (rDPO), a method for improving the behavioral alignment of Large Language Models (LLMs) without the need for human-annotated data. The method involves creating synthetic data using self-critique prompting by a teacher LLM and then utilising a generalized DPO loss function to distil to a student LLM. The loss function incorporates an additional external reward model to improve the quality of synthetic data, making rDPO robust to potential noise in the synthetic dataset. rDPO is shown to be effective in a diverse set of behavioural alignment tasks, such as improved safety, robustness against role-playing, and reduced sycophancy. Code to be released at https://github.com/vicgalle/refined-dpo.
    
[^44]: 流媒体的卫士：在软件定义网络中释放大型语言模型进行动态数据包分类--定位论文

    Sentinels of the Stream: Unleashing Large Language Models for Dynamic Packet Classification in Software Defined Networks -- Position Paper

    [https://arxiv.org/abs/2402.07950](https://arxiv.org/abs/2402.07950)

    本文提出了在网络安全领域探索大型语言模型适用性的计划，计划创建名为Sentinel的LLM来分析网络数据包内容并评估威胁级别。

    

    随着OpenAI的ChatGPT的发布，大型语言模型（LLM）领域在基于GPT的聊天助手方面引起了学术界的兴趣增加。在接下来的几个月中，释放了多个可访问的大型语言模型，包括Meta的LLama模型和Mistral AI的Mistral和Mixtral MoE模型。这些模型以各种不同的许可证对外公开，可用于各种不同的目的。这些LLM已经在代码开发、SQL生成等多个领域得到了应用。在本文中，我们提出了在网络安全领域探索大型语言模型适用性的计划。我们计划创建一个名为Sentinel的LLM，用于分析网络数据包内容并对其威胁级别进行判定。这项工作是我们未来发展规划的初步报告。

    With the release of OpenAI's ChatGPT, the field of large language models (LLM) saw an increase of academic interest in GPT based chat assistants. In the next few months multiple accesible large language models were released that included Meta's LLama models and Mistral AI's Mistral and Mixtral MoE models. These models are available openly for a wide array of purposes with a wide spectrum of licenses. These LLMs have found their use in a different number of fields like code development, SQL generation etc. In this work we propose our plan to explore the applicability of large language model in the domain of network security. We plan to create Sentinel, a LLM, to analyse network packet contents and pass a judgment on it's threat level. This work is a preliminary report that will lay our plan for our future endeavors.
    
[^45]: 重新构想指挥与控制

    Re-Envisioning Command and Control

    [https://arxiv.org/abs/2402.07946](https://arxiv.org/abs/2402.07946)

    重新构想的论文提出了未来指挥与控制（C2）决策需要面对更复杂和挑战性的环境，因此提出了基于人工智能系统与人类强有力伙伴关系的未来C2的愿景。这个愿景的核心是优化C2操作流程，保持协同努力，发展自适应的集体知识系统。

    

    未来的战争将要求在更复杂、快节奏、不结构化和极具挑战性的环境中进行指挥与控制（C2）决策。C2将因被拒绝、退化、间歇和有限的通信以及需要考虑到多个作战领域中的许多数据流而变得更加复杂。然而，当前的C2实践——源自工业时代而非新兴的智能时代——是线性的且耗时。而且，这些方法可能无法在未来战场上与对手保持优势。为了应对这些挑战，我们提出了一种基于人工智能（AI）系统与人类之间强有力伙伴关系的未来C2愿景。这个未来愿景体现在三个运营影响上：优化C2操作流程，保持协同努力，以及发展自适应的集体知识系统。本文阐述了所设想的未来指挥与控制的愿景。

    Future warfare will require Command and Control (C2) decision-making to occur in more complex, fast-paced, ill-structured, and demanding conditions. C2 will be further complicated by operational challenges such as Denied, Degraded, Intermittent, and Limited (DDIL) communications and the need to account for many data streams, potentially across multiple domains of operation. Yet, current C2 practices -- which stem from the industrial era rather than the emerging intelligence era -- are linear and time-consuming. Critically, these approaches may fail to maintain overmatch against adversaries on the future battlefield. To address these challenges, we propose a vision for future C2 based on robust partnerships between humans and artificial intelligence (AI) systems. This future vision is encapsulated in three operational impacts: streamlining the C2 operations process, maintaining unity of effort, and developing adaptive collective knowledge systems. This paper illustrates the envisaged fu
    
[^46]: UFO: 一个专注于Windows操作系统交互的用户界面智能体

    UFO: A UI-Focused Agent for Windows OS Interaction

    [https://arxiv.org/abs/2402.07939](https://arxiv.org/abs/2402.07939)

    UFO是一个专注于Windows操作系统上应用程序的用户界面智能体，利用GPT-Vision的能力来满足用户需求。它通过观察和分析Windows应用程序的图形用户界面和控制信息，实现无缝导航和操作以满足用户的请求。UFO的控制交互模块使得无需人工干预即可实现动作连接和完全自动化执行，使繁琐和耗时的过程变为简单任务。经过测试，UFO在各种场景中取得了良好效果。

    

    我们介绍了UFO，一个创新的专注于Windows操作系统上应用程序的用户界面智能体，利用了GPT-Vision的能力来满足用户需求。UFO采用双智能体框架，精确观察和分析Windows应用程序的图形用户界面（GUI）和控制信息。这使得智能体可以无缝地在单个应用程序内以及跨应用程序进行导航和操作，以满足用户的需求，即使涉及多个应用程序。该框架包括一个控制交互模块，实现无需人工干预的动作连接，并实现完全自动化执行。因此，UFO将艰巨而耗时的过程转变为仅通过自然语言命令就可以完成的简单任务。我们在9个流行的Windows应用程序上对UFO进行了测试，涵盖了反映用户日常使用情景的各种情况。通过定量指标和真实案例研究得出的结果强调了UFO的效果。

    We introduce UFO, an innovative UI-Focused agent to fulfill user requests tailored to applications on Windows OS, harnessing the capabilities of GPT-Vision. UFO employs a dual-agent framework to meticulously observe and analyze the graphical user interface (GUI) and control information of Windows applications. This enables the agent to seamlessly navigate and operate within individual applications and across them to fulfill user requests, even when spanning multiple applications. The framework incorporates a control interaction module, facilitating action grounding without human intervention and enabling fully automated execution. Consequently, UFO transforms arduous and time-consuming processes into simple tasks achievable solely through natural language commands. We conducted testing of UFO across 9 popular Windows applications, encompassing a variety of scenarios reflective of users' daily usage. The results, derived from both quantitative metrics and real-case studies, underscore t
    
[^47]: 大型语言用户界面：由LLMs驱动的语音交互用户界面

    Large Language User Interfaces: Voice Interactive User Interfaces powered by LLMs

    [https://arxiv.org/abs/2402.07938](https://arxiv.org/abs/2402.07938)

    本研究旨在利用和引导升级后的LLMs的强大能力，构建一个框架，作为用户和用户界面之间的中介，通过对自然文本输入进行彻底分析，实现智能和响应式用户体验。

    

    最近大型语言模型的快速发展展示了其在逻辑推理和理解方面的卓越能力。这些新发现的能力引发了新一代软件的诞生，正如它们在工业界无数应用中所展示的那样。本研究旨在利用和引导升级后的LLMs的强大能力，构建一个框架，作为用户和用户界面之间的中介。通过对自然文本输入进行彻底分析，一个经过精心设计的LLM引擎可以理解用户的需求，分类最有可能的应用程序，识别所需的UI组件，并随后执行用户期望的操作。这种集成可以将静态UI系统发展成高度动态和可适应的解决方案，引入智能和响应式用户体验的新领域。这样的框架可以从根本上改变用户完成日常任务的方式，极大提升用户体验。

    The recent meteoric advancements in large language models have showcased a remarkable capacity for logical reasoning and comprehension. These newfound capabilities have opened the door to a new generation of software, as has been made obvious through the innumerable ways they are being applied in the industry. This research focuses on harnessing and guiding the upgraded power of LLMs to construct a framework that can serve as an intermediary between a user and their user interface. By comprehending a user's needs through a thorough analysis of natural textual inputs, an effectively crafted LLM engine can classify the most likely available application, identify the desired UI component and subsequently execute the user's expected actions. This integration can evolve static UI systems into highly dynamic and adaptable solutions, introducing a new frontier of intelligent and responsive user experiences. Such a framework can fundamentally shift how users accomplish daily tasks, skyrocket e
    
[^48]: 一个人机协作的框架用于的模式开发

    A Human-Machine Collaboration Framework for the Development of Schemas

    [https://arxiv.org/abs/2402.07932](https://arxiv.org/abs/2402.07932)

    这篇论文提出了一个人机协作的框架用于设计新的模式，目的是解决机器智能中的挑战，并将AI社区的关注从技术转向AI科学。

    

    Winograd模式挑战（WSC）是一个为了研究展示人类行为的系统而设立的测试，它旨在将AI社区的关注点从技术转向AI科学。研究表明，尽管对人类来说是常见和琐碎的，但对机器来说仍然具有挑战性，尤其是当它们需要处理需要解决确定性代词的精心设计的句子时。为了解决这个需求，我们提出了一个专注于人机如何作为团队合作来设计新模式的新框架。

    The Winograd Schema Challenge (WSC), a seemingly well-thought-out test for machine intelligence, has been proposed to shed light on developing systems that exhibit human behavior. Since its introduction, it aimed to pivot the focus of the AI community from the technology to the science of AI. While common and trivial for humans, studies show that it is still challenging for machines, especially when they have to deal with novel schemas, that is, well-designed sentences that require the resolving of definite pronouns. As researchers have become increasingly interested in the challenge itself, this presumably necessitates the availability of an extensive collection of Winograd schemas, which goes beyond what human experts can reasonably develop themselves, especially after proposed ways of utilizing them as novel forms of CAPTCHAs.   To address this necessity, we propose a novel framework that explicitly focuses on how humans and machines can collaborate as teammates to design novel sche
    
[^49]: 大型语言模型中提示工程的系统调查：技术和应用

    A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications

    [https://arxiv.org/abs/2402.07927](https://arxiv.org/abs/2402.07927)

    这篇调查论文系统概述了大型语言模型中提示工程的最新进展，探讨了提示工程的方法和技术，并说明了其在各种应用中的重要作用。

    

    提示工程已成为扩展大型语言模型（LLM）和视觉语言模型（VLM）能力的不可或缺的技术。该方法利用任务特定的指令（称为提示）在不修改核心模型参数的情况下增强模型的效果。提示允许将预训练模型无缝集成到下游任务中，仅根据给定的提示引发所需的模型行为，而不是更新模型参数。提示可以是提供上下文以指导模型的自然语言指令，也可以是调用相关知识的学习向量表示。这个新兴领域在各种应用中取得了成功，从问答到常识推理都有涉及。然而，对于多样的提示工程方法和技术缺乏系统的组织和理解。本调查论文通过提供对最近进展的结构化概述来填补这一空白。

    Prompt engineering has emerged as an indispensable technique for extending the capabilities of large language models (LLMs) and vision-language models (VLMs). This approach leverages task-specific instructions, known as prompts, to enhance model efficacy without modifying the core model parameters. Rather than updating the model parameters, prompts allow seamless integration of pre-trained models into downstream tasks by eliciting desired model behaviors solely based on the given prompt. Prompts can be natural language instructions that provide context to guide the model or learned vector representations that activate relevant knowledge. This burgeoning field has enabled success across various applications, from question-answering to commonsense reasoning. However, there remains a lack of systematic organization and understanding of the diverse prompt engineering methods and techniques. This survey paper addresses the gap by providing a structured overview of recent advancements in pro
    
[^50]: 为帮助中国Python编程学习者提供的一个带有注释的问答数据集

    QACP: An Annotated Question Answering Dataset for Assisting Chinese Python Programming Learners

    [https://arxiv.org/abs/2402.07913](https://arxiv.org/abs/2402.07913)

    为解决编程智能教育系统中数据稀缺问题，本文提出了一个新的针对Python学习者的中文问答数据集，通过收集与分类真实学生问题，提高在线编程教育的效果和质量。

    

    在在线学习平台中，特别是在快速增长的计算机编程课程中，解答成千上万学生的学习问题需要相当大的人力成本。为编程教育定制智能助手大型语言模型（LLMs）的创建需要独特的数据支持。然而，在实际应用场景中，用于训练此类LLMs的数据资源相对稀缺。因此，为了解决编程智能教育系统中的数据稀缺问题，本文提出了一个新的针对Python学习者的中文问答数据集。为确保问题的来源的真实性和可靠性，我们收集了实际学生提出的问题，并根据问题的类型和学习者的类型进行分类。这种注释原则旨在提高在线编程教育的效果和质量，为开发这方面的工作提供坚实的数据基础。

    In online learning platforms, particularly in rapidly growing computer programming courses, addressing the thousands of students' learning queries requires considerable human cost. The creation of intelligent assistant large language models (LLMs) tailored for programming education necessitates distinct data support. However, in real application scenarios, the data resources for training such LLMs are relatively scarce. Therefore, to address the data scarcity in intelligent educational systems for programming, this paper proposes a new Chinese question-and-answer dataset for Python learners. To ensure the authenticity and reliability of the sources of the questions, we collected questions from actual student questions and categorized them according to various dimensions such as the type of questions and the type of learners. This annotation principle is designed to enhance the effectiveness and quality of online programming education, providing a solid data foundation for developing th
    
[^51]: Prompt4Vis: 使用示例挖掘和结构过滤来为表格数据可视化提供提示的大语言模型

    Prompt4Vis: Prompting Large Language Models with Example Mining and Schema Filtering for Tabular Data Visualization

    [https://arxiv.org/abs/2402.07909](https://arxiv.org/abs/2402.07909)

    提出了 Prompt4Vis，使用示例挖掘和结构过滤来为表格数据可视化的大语言模型提供提示。这种方法利用了巨大的语言模型的优势，并能够改进当前自然语言查询转换成数据可视化查询的方法。

    

    数据可视化(DV)系统因其在大数据集中发现洞见的深厚能力而得到越来越多的认可，引起了工业界和学术界的关注。在某些声明式可视化语言(DVLs，如Vega-Lite、EChart)中，编制数据查询是一个重要的过程。自然语言处理(NLP)技术的发展使得使用自然语言界面来可视化表格数据的过程更加简单和直观，提供了更可访问和直观的用户体验。然而，当前将自然语言问题转换成数据可视化查询的方法，如Seq2Vis、ncNet和RGVisNet，尽管利用了复杂的神经网络架构，仍然不尽人意，有很大的改进空间。大语言模型(LLMs)如ChatGPT和GPT-4在各种NLP任务中取得了新的基准，从根本上改变了该领域的格局。受到这些进展的启发，我们引入了一种新方法。

    Data visualization (DV) systems are increasingly recognized for their profound capability to uncover insights from vast datasets, gaining attention across both industry and academia. Crafting data queries is an essential process within certain declarative visualization languages (DVLs, e.g., Vega-Lite, EChart.). The evolution of natural language processing (NLP) technologies has streamlined the use of natural language interfaces to visualize tabular data, offering a more accessible and intuitive user experience. However, current methods for converting natural language questions into data visualization queries, such as Seq2Vis, ncNet, and RGVisNet, despite utilizing complex neural network architectures, still fall short of expectations and have great room for improvement.   Large language models (LLMs) such as ChatGPT and GPT-4, have established new benchmarks in a variety of NLP tasks, fundamentally altering the landscape of the field. Inspired by these advancements, we introduce a nov
    
[^52]: AI和ChatGPT在教育中的应用、挑战和伦理问题

    Applications, challenges and ethical issues of AI and ChatGPT in education

    [https://arxiv.org/abs/2402.07907](https://arxiv.org/abs/2402.07907)

    本文探讨了人工智能和ChatGPT在改进教育方面的机遇，同时也指出了相关的挑战和伦理问题。

    

    近年来，人工智能（AI）展示了令人瞩目的发展，并且趋向在生活的各个方面发挥催化作用。学术界和政府对AI的兴趣日益增长，这也反映在正在进行的投资与研究的数量激增上。每天都有热情洋溢的意见和论述关于AI，但同时也提出了对其影响的警示性预测。本文旨在描述利用人工智能和ChatGPT改进教育所带来的机遇，同时也要识别出出现的挑战和伦理问题。

    Artificial Intelligence (AI) in recent years has shown an unprecedentedly impressive development, tending to play a catalytic role in all aspects of life. The interest of the academic community, but also of governments, is huge in the dynamics of AI and is reflected by the truly explosive amount of investment and research that is underway. Enthusiastic opinions and statements about AI are made every day, but at the same time they also bring to the fore alarming predictions about its effects. This paper aims to describe the opportunities emerging from the use of artificial intelligence and ChatGPT to improve education, but also to identify the challenges and ethical issues that arise.
    
[^53]: 使用直接原则反馈抑制“粉色大象”

    Suppressing Pink Elephants with Direct Principle Feedback

    [https://arxiv.org/abs/2402.07896](https://arxiv.org/abs/2402.07896)

    本研究提出了一种名为“直接原则反馈”的新方法，用于控制语言模型中的LLM行为。通过在批评和修订上直接使用DPO来跳过响应的排名，我们成功地解决了“粉色大象问题”并取得了显著的性能优势。

    

    目前的语言模型控制方法，如RLHF和宪法AI，涉及确定LLM行为的可取之处，并将其训练到语言模型中。然而，在许多情况下，希望LLM在推理时是可控制的，这样可以在多种需要的上下文中使用。我们用“粉色大象问题”作为例子：指示LLM避免讨论某个特定实体（“粉色大象”），而是讨论首选实体（“灰色大象”）。我们应用了一种新颖的Constitutional AI简化方法，“直接原则反馈”，它跳过了对响应的排名，直接在批评和修订上使用DPO。我们的结果表明，在我们合成的“粉色大象”数据集上进行DPF微调后，我们的13B微调LLaMA 2模型明显优于Llama-2-13B-Chat和提示基线，并且在评估“粉色大象问题”的精心选择测试集上表现与GPT-4一样好。

    Existing methods for controlling language models, such as RLHF and Constitutional AI, involve determining which LLM behaviors are desirable and training them into a language model. However, in many cases, it is desirable for LLMs to be controllable \textit{at inference time}, so that they can be used in multiple contexts with diverse needs. We illustrate this with the \textbf{Pink Elephant Problem}: instructing an LLM to avoid discussing a certain entity (a ``Pink Elephant''), and instead discuss a preferred entity (``Grey Elephant''). We apply a novel simplification of Constitutional AI, \textbf{Direct Principle Feedback}, which skips the ranking of responses and uses DPO directly on critiques and revisions. Our results show that after DPF fine-tuning on our synthetic Pink Elephants dataset, our 13B fine-tuned LLaMA 2 model significantly outperforms Llama-2-13B-Chat and a prompted baseline, and performs as well as GPT-4 in on our curated test set assessing the Pink Elephant Problem.
    
[^54]: 质量确实重要：对网络挖掘平行语料库的质量和实用性进行详细研究

    Quality Does Matter: A Detailed Look at the Quality and Utility of Web-Mined Parallel Corpora

    [https://arxiv.org/abs/2402.07446](https://arxiv.org/abs/2402.07446)

    这项研究详细分析了网络挖掘语料库的质量和实用性，并发现不同语言和数据集之间存在显著的质量差异。同时，我们还展示了某些网络挖掘数据集的最佳部分训练的神经机器翻译模型可以与人工策划的数据集持平。

    

    我们对两种低资源语言（英文-僧伽罗语，英文-泰米尔语和僧伽罗语-泰米尔语）的网络挖掘语料库的质量进行了详细分析。我们根据相似度标准对每个语料库进行了排名，并对排名语料库的不同部分进行内在和外在评估。我们显示不同部分的网络挖掘语料库存在显著的质量差异，并且质量在不同语言和数据集之间存在变化。我们还表明，对于某些网络挖掘数据集，使用其排名最高的25k部分训练的神经机器翻译（NMT）模型可以与人工策划的数据集持平。

    We conducted a detailed analysis on the quality of web-mined corpora for two low-resource languages (making three language pairs, English-Sinhala, English-Tamil and Sinhala-Tamil). We ranked each corpus according to a similarity measure and carried out an intrinsic and extrinsic evaluation on different portions of this ranked corpus. We show that there are significant quality differences between different portions of web-mined corpora and that the quality varies across languages and datasets. We also show that, for some web-mined datasets, Neural Machine Translation (NMT) models trained with their highest-ranked 25k portion can be on par with human-curated datasets.
    
[^55]: SALAD: 智能AI语言助手日常

    SALAD: Smart AI Language Assistant Daily

    [https://arxiv.org/abs/2402.07431](https://arxiv.org/abs/2402.07431)

    SALAD是一款智能AI语言助手应用，旨在帮助外国人学习日语。它提供了多种学习工具和功能，包括翻译，语音识别，音频翻译，词汇跟踪等，并通过每日翻译帮助提高与母语人士的交流能力。调查结果显示60%的外国人对SALAD提升日语能力有信心。该应用利用大型语言模型和扩散模型促进日本社区的包容性。

    

    SALAD是一款由AI驱动的语言学习应用程序，旨在帮助外国人学习日语。它提供了汉字-假名-罗马字的翻译，语音识别，翻译音频，词汇跟踪，语法解释，以及由新学到的词汇生成的歌曲。该应用针对初学者和中级学习者，旨在使语言习得更加可获得和愉快。SALAD利用每日翻译来增强与母语人士的流利度和交流舒适度。主要目标包括有效的日语学习，用户参与度和进展跟踪。我们的调查发现，在日本的外国人中，有39%在与日本人交谈时感到不适。超过60%的外国人表示对SALAD提升他们的日语能力有信心。该应用使用大型语言模型，语音识别和扩散模型来弥合语言隔阂，促进日本更具包容性的社区。

    SALAD is an AI-driven language-learning application designed to help foreigners learn Japanese. It offers translations in Kanji-Kana-Romaji, speech recognition, translated audio, vocabulary tracking, grammar explanations, and songs generated from newly learned words. The app targets beginners and intermediate learners, aiming to make language acquisition more accessible and enjoyable. SALAD uses daily translations to enhance fluency and comfort in communication with native speakers. The primary objectives include effective Japanese language learning, user engagement, and progress tracking. A survey by us found that 39% of foreigners in Japan face discomfort in conversations with Japanese speakers. Over 60% of foreigners expressed confidence in SALAD's ability to enhance their Japanese language skills. The app uses large language models, speech recognition, and diffusion models to bridge the language gap and foster a more inclusive community in Japan.
    
[^56]: HyperBERT:将混合超图感知层与语言模型用于文本属性超图上的节点分类

    HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs

    [https://arxiv.org/abs/2402.07309](https://arxiv.org/abs/2402.07309)

    本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。

    

    超图通过复杂的拓扑结构标记，表达多个实体之间的高阶相互作用，其中超边扮演重要角色。最近，基于超图的深度学习方法在学习文本属性超图上的节点分类问题中引起了越来越多的研究关注。然而，现有方法往往难以同时捕捉超图结构信息的全部内容和节点属性中的丰富语言属性，这在很大程度上影响了它们的效果和泛化能力。为了克服这些挑战，我们探索了如何通过为节点分类任务进一步增强预训练的BERT模型，引入专门的超图感知层。这些层将高阶结构归纳偏差引入语言模型中，从而提高模型利用超图结构中的高阶上下文信息和文本中的语义信息的能力。

    Hypergraphs are marked by complex topology, expressing higher-order interactions among multiple entities with hyperedges. Lately, hypergraph-based deep learning methods to learn informative data representations for the problem of node classification on text-attributed hypergraphs have garnered increasing research attention. However, existing methods struggle to simultaneously capture the full extent of hypergraph structural information and the rich linguistic attributes inherent in the nodes attributes, which largely hampers their effectiveness and generalizability. To overcome these challenges, we explore ways to further augment a pretrained BERT model with specialized hypergraph-aware layers for the task of node classification. Such layers introduce higher-order structural inductive bias into the language model, thus improving the model's capacity to harness both higher-order context information from the hypergraph structure and semantic information present in text. In this paper, we
    
[^57]: 大型语言模型如何在诚实与帮助之间进行权衡？

    How do Large Language Models Navigate Conflicts between Honesty and Helpfulness?

    [https://arxiv.org/abs/2402.07282](https://arxiv.org/abs/2402.07282)

    本文研究了如何在大型语言模型中权衡诚实和帮助性，在实验中发现强化学习改善了诚实和帮助性，而链式思维提示则偏向于帮助性。研究结果还展示了GPT-4 Turbo对对话框架和听众决策背景的敏感性。这些发现揭示了大型语言模型内化的对话价值观，并暗示零-shot提示可以在一定程度上引导这些抽象价值观。

    

    在日常交流中，人们经常为了最大限度地帮助听众而近似真相，例如约略时间或省略细节。大型语言模型（LLMs）如何处理这种微妙的权衡？为了回答这个问题，我们使用心理模型和旨在描述人类行为的实验来分析LLMs。我们测试了一系列LLMs，并探讨了优化人类偏好或推理时思考对这些权衡的影响。我们发现，从人类反馈中的强化学习改善了诚实和帮助性，而链式思维提示使LLMs偏向于帮助性而不是诚实。最后，GPT-4 Turbo展示了类似人类的回应模式，包括对对话框架和听众决策背景的敏感性。我们的研究结果揭示了LLMs内化的对话价值观，并暗示即使这些抽象价值观也可以在零-shot提示下在一定程度上被引导。

    In day-to-day communication, people often approximate the truth - for example, rounding the time or omitting details - in order to be maximally helpful to the listener. How do large language models (LLMs) handle such nuanced trade-offs? To address this question, we use psychological models and experiments designed to characterize human behavior to analyze LLMs. We test a range of LLMs and explore how optimization for human preferences or inference-time reasoning affects these trade-offs. We find that reinforcement learning from human feedback improves both honesty and helpfulness, while chain-of-thought prompting skews LLMs towards helpfulness over honesty. Finally, GPT-4 Turbo demonstrates human-like response patterns including sensitivity to the conversational framing and listener's decision context. Our findings reveal the conversational values internalized by LLMs and suggest that even these abstract values can, to a degree, be steered by zero-shot prompting.
    
[^58]: 学习变得高效：在大型语言模型中构建结构化稀疏性

    Learn To be Efficient: Build Structured Sparsity in Large Language Models

    [https://arxiv.org/abs/2402.06126](https://arxiv.org/abs/2402.06126)

    本文通过引入一种新的算法"Learn-To-be-Efficient(LTE)"，提出了在大型语言模型(LLM)中构建结构化稀疏性的方法。该方法通过训练高效意识的LLM学习激活更少的神经元，取得更好的稀疏性和性能折衷。

    

    大型语言模型(LLM)以其十亿级参数取得了显著的成功，但它们产生了高昂的推理开销。在LLM中出现的激活稀疏性为通过仅涉及部分参数进行推理提供了一种自然的方法来减少这种成本。现有方法只关注利用这种自然形成的激活稀疏性，忽视了进一步放大这种固有稀疏性的潜力。本文中，我们假设LLM可以通过实现更结构化的激活稀疏性来学习高效。为实现这一目标，我们引入了一种新颖的算法"Learn-To-be-Efficient(LTE)", 旨在训练高效意识的LLM学习激活更少的神经元，并在稀疏性和性能之间取得更好的折衷。此外，与主要关注基于ReLU模型的SOTA MoEfication方法不同，LTE还可以应用于像GPT和LLaMA这样具有软激活函数的LLM。我们在四个模型和十一个数据集上评估了LTE。

    Large Language Models (LLMs) have achieved remarkable success with their billion-level parameters, yet they incur high inference overheads. The emergence of activation sparsity in LLMs provides a natural approach to reduce this cost by involving only parts of the parameters for inference. Existing methods only focus on utilizing this naturally formed activation sparsity, overlooking the potential for further amplifying this inherent sparsity. In this paper, we hypothesize that LLMs can learn to be efficient by achieving more structured activation sparsity.To achieve this, we introduce a novel algorithm, Learn-To-be-Efficient (LTE), designed to train efficiency-aware LLMs to learn to activate fewer neurons and achieve a better trade-off between sparsity and performance. Furthermore, unlike SOTA MoEfication methods, which mainly focus on ReLU-based models, LTE can also be applied to LLMs like GPT and LLaMA with soft activation functions. We evaluate LTE on four models and eleven datasets
    
[^59]: Transformer语言模型在算法学习上的限制

    Limits of Transformer Language Models on Algorithmic Learning

    [https://arxiv.org/abs/2402.05785](https://arxiv.org/abs/2402.05785)

    Transformer语言模型在学习离散算法方面的组合能力非常有限，比重新学习所有子任务对于新的算法组合的效果更差，而且梯度下降在记忆前馈模型上的效率非常低。

    

    我们分析了Transformer语言模型在学习离散算法方面的能力。为此，我们引入了两个要求组合多个离散子任务的新任务。我们通过从头开始训练LLaMA模型和在GPT-4和Gemini上提示来衡量学习学习原语的组合。我们观察到，目前最先进的Transformer语言模型的组合能力非常有限，并且在样本规模方面比为新的算法组合重新学习所有子任务效果更差。我们还提出了一个复杂性理论的定理，证明了记忆前馈模型上的梯度下降可以指数级地浪费数据。

    We analyze the capabilities of Transformer language models on learning discrete algorithms. To this end, we introduce two new tasks demanding the composition of several discrete sub-tasks. On both training LLaMA models from scratch and prompting on GPT-4 and Gemini we measure learning compositions of learned primitives. We observe that the compositional capabilities of state-of-the-art Transformer language models are very limited and sample-wise scale worse than relearning all sub-tasks for a new algorithmic composition. We also present a theorem in complexity theory, showing that gradient descent on memorizing feedforward models can be exponentially data inefficient.
    
[^60]: C-RAG: 针对检索增强语言模型的认证生成风险

    C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models

    [https://arxiv.org/abs/2402.03181](https://arxiv.org/abs/2402.03181)

    C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。

    

    尽管大型语言模型（LLMs）在各种应用中具备令人印象深刻的能力，但它们仍然存在可信度问题，如幻觉和错位。检索增强语言模型（RAG）被提出来增强生成结果的可信性，通过引入外部知识。但是，对于RAG模型的生成风险的理论理解尚未被研究。本文回答了以下问题：1）RAG是否确实能够降低生成风险，2）如何对RAG和传统LLM的生成风险提供可证明的保证，以及3）哪些充分条件使得RAG模型能够降低生成风险。我们提出了C-RAG，第一个用于认证RAG模型生成风险的框架。具体而言，我们为RAG模型提供了符合风险分析，并确保了生成风险的上界，我们称之为符合生成风险。我们还对一般有界风险下的符合生成风险提供了理论保证。

    Despite the impressive capabilities of large language models (LLMs) across diverse applications, they still suffer from trustworthiness issues, such as hallucinations and misalignments. Retrieval-augmented language models (RAG) have been proposed to enhance the credibility of generations by grounding external knowledge, but the theoretical understandings of their generation risks remains unexplored. In this paper, we answer: 1) whether RAG can indeed lead to low generation risks, 2) how to provide provable guarantees on the generation risks of RAG and vanilla LLMs, and 3) what sufficient conditions enable RAG models to reduce generation risks. We propose C-RAG, the first framework to certify generation risks for RAG models. Specifically, we provide conformal risk analysis for RAG models and certify an upper confidence bound of generation risks, which we refer to as conformal generation risk. We also provide theoretical guarantees on conformal generation risks for general bounded risk f
    
[^61]: CroissantLLM: 一个真正的双语法语-英语语言模型

    CroissantLLM: A Truly Bilingual French-English Language Model

    [https://arxiv.org/abs/2402.00786](https://arxiv.org/abs/2402.00786)

    CroissantLLM是一个1.3B的双语语言模型，通过使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集进行训练，实现了高性能和开源。模型还发布了训练数据集和多个检查点，以及一个法语基准测试 FrenchBench。

    

    我们介绍了CroissantLLM，这是一个在3T个英语和法语标记上预训练的13亿语言模型，为研究和工业社区带来了一种高性能的、完全开源的双语模型，可以在消费级本地硬件上快速运行。为此，我们首次尝试使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集来训练一种内在双语的模型。我们发布了训练数据集，其中包含了一个法语分割，其中包含了手工策划、高质量和多样化的数据源。为了评估在英语以外的性能，我们创建了一个新的基准测试 FrenchBench，包括一系列分类和生成任务，涵盖了模型在法语语言中性能的各个方面。此外，为了保持透明度并促进进一步的大规模语言模型研究，我们发布了代码库和各种模型规模、训练数据分布上的几十个检查点。

    We introduce CroissantLLM, a 1.3B language model pretrained on a set of 3T English and French tokens, to bring to the research and industrial community a high-performance, fully open-sourced bilingual model that runs swiftly on consumer-grade local hardware. To that end, we pioneer the approach of training an intrinsically bilingual model with a 1:1 English-to-French pretraining data ratio, a custom tokenizer, and bilingual finetuning datasets. We release the training dataset, notably containing a French split with manually curated, high-quality, and varied data sources. To assess performance outside of English, we craft a novel benchmark, FrenchBench, consisting of an array of classification and generation tasks, covering various orthogonal aspects of model performance in the French Language. Additionally, rooted in transparency and to foster further Large Language Model research, we release codebases, and dozens of checkpoints across various model sizes, training data distributions, 
    
[^62]: 健康-LLM：个性化检索增强的疾病预测模型

    Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model

    [https://arxiv.org/abs/2402.00746](https://arxiv.org/abs/2402.00746)

    提出了一个创新的框架，健康-LLM，通过大规模特征提取和医学知识权衡评分，实现了个性化的检索增强疾病预测模型。这种方法通过整合健康报告，调整特征权重，以及利用语言模型和专家见解提高预测准确性，与传统健康管理方法相比具有明显优势。

    

    在卫生保健领域，人工智能（AI）极大地推进了智能医疗技术的发展。然而，传统智能医疗受限于静态数据和统一标准，无法完全与个体情况集成，同时也面临其他挑战。为此，我们提出了一种创新的框架，命名为健康-LLM，将大规模特征提取和医学知识权衡评分相结合。与传统健康管理方法相比，我们的方法具有三个主要优势。首先，我们的方法将健康报告整合到大模型中，提供详细的任务信息。其次，我们使用专业的医学专业知识调整健康特征的权重得分。第三，我们使用半自动特征提取框架增强语言模型的分析能力，并整合专家见解以提高疾病预测的准确性。

    Artificial intelligence (AI) in healthcare has significantly advanced intelligent medical treatment. However, traditional intelligent healthcare is limited by static data and unified standards, preventing full integration with individual situations and other challenges. Hence, a more professional and detailed intelligent healthcare method is needed for development. To this end, we propose an innovative framework named Heath-LLM, which combines large-scale feature extraction and medical knowledge trade-off scoring. Compared to traditional health management methods, our approach has three main advantages. First, our method integrates health reports into a large model to provide detailed task information. Second, professional medical expertise is used to adjust the weighted scores of health characteristics. Third, we use a semi-automated feature extraction framework to enhance the analytical power of language models and incorporate expert insights to improve the accuracy of disease predic
    
[^63]: 一条思维链条的强度取决于最弱的环节：一个验证推理链的基准

    A Chain-of-Thought Is as Strong as Its Weakest Link: A Benchmark for Verifiers of Reasoning Chains

    [https://arxiv.org/abs/2402.00559](https://arxiv.org/abs/2402.00559)

    本论文提出了Reveal数据集，用于在开放领域的问答设置中对复杂思维链的自动验证进行基准测试。这个数据集包含了详尽的标签，用于评估语言模型的答案中每个推理步骤的相关性、归因和逻辑正确性。

    

    促使语言模型提供逐步回答（例如“思维链”）是复杂推理任务的主要方法，其中更准确的推理链通常可以提高下游任务的性能。最近的文献讨论了自动验证推理步骤的方法，以评估和改善其正确性。然而，缺乏细粒度的步骤级数据集，无法对这类验证方法进行全面评估，从而阻碍了在这个方向上的进展。我们介绍了Reveal：推理验证评估，这是一个新的数据集，用于在开放领域的问答设置中对复杂思维链的自动验证进行基准测试。Reveal包括对语言模型答案中每个推理步骤的相关性、归因于证据段落以及逻辑正确性的全面标签，涵盖了各种数据集和最先进的语言模型。

    Prompting language models to provide step-by-step answers (e.g., "Chain-of-Thought") is the prominent approach for complex reasoning tasks, where more accurate reasoning chains typically improve downstream task performance. Recent literature discusses automatic methods to verify reasoning steps to evaluate and improve their correctness. However, no fine-grained step-level datasets are available to enable thorough evaluation of such verification methods, hindering progress in this direction. We introduce Reveal: Reasoning Verification Evaluation, a new dataset to benchmark automatic verifiers of complex Chain-of-Thought reasoning in open-domain question answering settings. Reveal includes comprehensive labels for the relevance, attribution to evidence passages, and logical correctness of each reasoning step in a language model's answer, across a wide variety of datasets and state-of-the-art language models.
    
[^64]: 多模态大型语言模型中非语言抽象推理的奇特案例

    The Curious Case of Nonverbal Abstract Reasoning with Multi-Modal Large Language Models

    [https://arxiv.org/abs/2401.12117](https://arxiv.org/abs/2401.12117)

    本研究评估了多模态大型语言模型（MLLMs）在非语言抽象推理方面的能力，并发现开源和闭源模型之间存在巨大差距和个体模块的关键缺陷。

    

    尽管大型语言模型（LLMs）仍在逐渐应用于新领域并在新应用中被利用，但我们正在经历新一代基础模型的涌现，即多模态大型语言模型（MLLMs）。这些模型将语言和视觉信息进行整合，为展示出更复杂的推理能力在两种模态的交集处提供了新的可能性。然而，尽管MLLMs的前景具有革命性的前景，我们对它们的推理能力的理解仍然有限。在本研究中，我们使用Raven's Progressive Matrices的变式评估了开源和闭源MLLMs的非语言抽象推理能力。我们的实验揭示了解决这类问题的困难，同时展示了开源和闭源模型之间巨大的差距。我们还揭示了个体视觉和文本模块的关键缺陷，导致模型的性能受到低谷的限制。最后，为了提高MLLMs的性能，我们进行了一系列实验。

    While large language models (LLMs) are still being adopted to new domains and utilized in novel applications, we are experiencing an influx of the new generation of foundation models, namely multi-modal large language models (MLLMs). These models integrate verbal and visual information, opening new possibilities to demonstrate more complex reasoning abilities at the intersection of the two modalities. However, despite the revolutionizing prospect of MLLMs, our understanding of their reasoning abilities is limited. In this study, we assess the nonverbal abstract reasoning abilities of open-source and closed-source MLLMs using variations of Raven's Progressive Matrices. Our experiments expose the difficulty of solving such problems while showcasing the immense gap between open-source and closed-source models. We also reveal critical shortcomings with individual visual and textual modules, subjecting the models to low-performance ceilings. Finally, to improve MLLMs' performance, we experi
    
[^65]: ALMs: 作者语言模型用于作者归属性

    ALMs: Authorial Language Models for Authorship Attribution

    [https://arxiv.org/abs/2401.12005](https://arxiv.org/abs/2401.12005)

    本文介绍了一种名为作者语言模型（ALMs）的作者归属方法，通过计算疑问文件的困惑度来确定最有可能的作者。ALMs在Blogs50数据集上表现出色，在CCAT50数据集上与最好的方法相当。在较短文本上，ALMs需要较少的令牌来实现较高的准确率。

    

    在本文中，我们介绍了一种称为作者语言模型（ALMs）的作者归属方法，该方法通过计算基于一组被调整为候选作者的写作的因果语言模型的疑问文件的困惑度来确定最有可能的作者。我们使用CCAT50数据集和Blogs50数据集对ALMs进行了基准测试。我们发现，ALMs在Blogs50上取得了83.6%的宏平均准确率，超过了所有其他方法，而在CCAT50上取得了74.9%的宏平均准确率，与最好的方法相当。为了评估ALMs在较短文本上的性能，我们还进行了文本剥离测试。我们发现，为了达到70%的宏平均准确率，ALMs在Blogs50上需要40个令牌，在CCAT50上需要400个令牌，而为了达到60%，ALMs在Blogs50上需要20个令牌，在CCAT50上需要70个令牌。

    In this paper, we introduce an authorship attribution method called Authorial Language Models (ALMs) that involves identifying the most likely author of a questioned document based on the perplexity of the questioned document calculated for a set of causal language models fine-tuned on the writings of a set of candidate author. We benchmarked ALMs against state-of-art-systems using the CCAT50 dataset and the Blogs50 datasets. We find that ALMs achieves a macro-average accuracy score of 83.6% on Blogs50, outperforming all other methods, and 74.9% on CCAT50, matching the performance of the best method. To assess the performance of ALMs on shorter texts, we also conducted text ablation testing. We found that to reach a macro-average accuracy of 70%, ALMs needs 40 tokens on Blogs50 and 400 tokens on CCAT50, while to reach 60% ALMs requires 20 tokens on Blogs50 and 70 tokens on CCAT50.
    
[^66]: 如何合并生成和检索上下文以增强开放领域问答的语言模型的研究

    Blinded by Generated Contexts: How Language Models Merge Generated and Retrieved Contexts for Open-Domain QA?

    [https://arxiv.org/abs/2401.11911](https://arxiv.org/abs/2401.11911)

    该论文研究了大型语言模型如何合并生成和检索的上下文以提升开放领域问答，发现这些模型偏向于生成的上下文，即使它们提供了错误的信息。

    

    虽然辅助信息已经成为增强大型语言模型（LLMs）的关键，但对于LLMs如何合并生成的和检索的上下文仍知之甚少。为了研究这一点，我们制定了一个系统性的框架来确定LLMs的响应是源自于生成的上下文还是检索的上下文。为了实现这个目标，我们构建了包含相互冲突的上下文的数据集，其中每个问题都与生成的和检索的上下文配对，但只有一个上下文包含了正确的答案。我们的实验证明，LLMs（如GPT-4/3.5和Llama2）存在显著的偏差，更倾向于生成的上下文，即使这些上下文提供了错误的信息。我们进一步确定了导致这种偏差的两个关键因素：i）LLMs生成的上下文通常与问题更相似，增加了其被选择的可能性；ii）检索上下文中使用的分割过程打断了其连贯性。

    While auxiliary information has become a key to enhance Large Language Models (LLMs), relatively little is known about how LLMs merge these contexts, specifically generated and retrieved. To study this, we formulate a systematic framework to identify whether LLMs' responses, derived from the integration of generated and retrieved contexts, are attributed to either generated or retrieved contexts. To achieve this, we construct datasets with conflicting contexts, where each question is paired with both generated and retrieved contexts, yet only one of them contains the correct answer. Our experiments reveal a significant bias in LLMs (GPT-4/3.5 and Llama2) towards generated contexts, even when they provide incorrect information. We further identify two key factors contributing to this bias: i) contexts generated by LLMs typically show greater similarity to the questions, increasing their likelihood of selection; ii) the segmentation process used in retrieved contexts disrupts their compl
    
[^67]: 大型语言模型是临床推理者：基于提示生成的理由的推理感知诊断框架

    Large Language Models are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales

    [https://arxiv.org/abs/2312.07399](https://arxiv.org/abs/2312.07399)

    该论文提出了一种基于提示生成的理由的“推理感知”诊断框架，通过大型语言模型来进行临床推理，实现了在疾病诊断过程中的高效、时间节约和劳动节约的方法。

    

    由于大型语言模型（LLMs）的进展，机器推理在近年来取得了巨大的进展。然而，在临床领域，大多数以自然语言处理为驱动的项目主要集中在临床分类或阅读理解上，并且由于与临床医生的理念注解成本较高，对于疾病诊断的临床推理还未得到充分的研究。在这项工作中，我们提出了一个“推理感知”的诊断框架，通过基于提示的学习以一种高效的时间和劳动方式去理性化诊断过程，并学习对提示生成的理由进行推理。具体而言，我们解决了疾病诊断的临床推理问题，其中LLM生成了诊断性的理由，提供其对呈现的患者数据的见解以及达到诊断的推理路径，即临床思维链（Clinical CoT）。我们通过广泛的实验和分析在理由生成和疾病诊断方面实证了LLMs/LMs的临床推理能力。

    Machine reasoning has made great progress in recent years owing to large language models (LLMs). In the clinical domain, however, most NLP-driven projects mainly focus on clinical classification or reading comprehension, and under-explore clinical reasoning for disease diagnosis due to the expensive rationale annotation with clinicians. In this work, we present a ``reasoning-aware'' diagnosis framework that rationalizes the diagnostic process via prompt-based learning in a time- and labor-efficient manner, and learns to reason over the prompt-generated rationales. Specifically, we address the clinical reasoning for disease diagnosis, where the LLM generates diagnostic rationales providing its insight on presented patient data and the reasoning path towards the diagnosis, namely Clinical Chain-of-Thought (Clinical CoT). We empirically demonstrate LLMs/LMs' ability of clinical reasoning via extensive experiments and analyses on both rationale generation and disease diagnosis in various s
    
[^68]: 多步骤对话工作流动作预测

    Multi-Step Dialogue Workflow Action Prediction

    [https://arxiv.org/abs/2311.09593](https://arxiv.org/abs/2311.09593)

    本文提出了多步骤工作流动作预测的新问题，通过准确预测多个步骤，实现对任务的多轮自动化，节省时间。提出了三种简单易行的建模方法，并展示了多步骤动作预测提高对话任务准确性和步骤自动化的特征。

    

    在面向任务的对话中，系统通常需要遵循一系列动作的顺序，称为工作流，以便根据一组准则完成任务。本文提出了多步骤工作流动作预测的新问题，系统预测未来的多个工作流动作。准确预测多个步骤可以实现多轮自动化，从而能够节省时间专注于更复杂的任务。我们提出了三种简单易行且能提高动作自动化的建模方法：1）在训练数据集上进行微调，2）使用检索和大型语言模型提示进行少样本上下文学习，以及3）零样本图遍历，将历史动作序列汇总成图进行预测。我们展示了多步骤动作预测产生的特征，可以提高下游对话任务（如预测任务成功）的准确性，并可以在不需要太多反馈的情况下将步骤的自动化提高20%。

    In task-oriented dialogue, a system often needs to follow a sequence of actions, called a workflow, that complies with a set of guidelines in order to complete a task. In this paper, we propose the novel problem of multi-step workflow action prediction, in which the system predicts multiple future workflow actions. Accurate prediction of multiple steps allows for multi-turn automation, which can free up time to focus on more complex tasks. We propose three modeling approaches that are simple to implement yet lead to more action automation: 1) fine-tuning on a training dataset, 2) few-shot in-context learning leveraging retrieval and large language model prompting, and 3) zero-shot graph traversal, which aggregates historical action sequences into a graph for prediction. We show that multi-step action prediction produces features that improve accuracy on downstream dialogue tasks like predicting task success, and can increase automation of steps by 20% without requiring as much feedback
    
[^69]: LILO：通过压缩和文档化代码学习可解释库

    LILO: Learning Interpretable Libraries by Compressing and Documenting Code

    [https://arxiv.org/abs/2310.19791](https://arxiv.org/abs/2310.19791)

    LILO是一种神经符号框架，通过迭代地合成、压缩和文档化代码来构建可解释且适用于特定问题领域的程序库。在其中，LILO结合了大型语言模型引导的程序合成和程序自动重构的算法进展，并且通过自动文档过程使得代码抽象可解释并提升性能。

    

    尽管大型语言模型（LLMs）在代码生成方面表现出色，但软件开发的关键方面是重构的艺术：将代码整合到可重用和可读的程序库中。本文介绍了一种名为LILO的神经符号框架，它通过迭代地合成、压缩和文档化代码来构建适合特定问题领域的库。LILO将LLM引导的程序合成与Stitch自动重构的近期算法进展相结合：Stitch是一个符号压缩系统，可以高效地识别大型代码语料库中的最佳lambda抽象。为了使这些抽象可解释，我们引入了一种自动文档（AutoDoc）过程，它根据上下文中的使用示例推断出自然语言名称和文档字符串。除了提高人类可读性外，我们发现AutoDoc通过帮助LILO的合成器解释和部署学习到的抽象来提高性能。我们对LILO进行了三个归纳式程序综合的评估。

    While large language models (LLMs) now excel at code generation, a key aspect of software development is the art of refactoring: consolidating code into libraries of reusable and readable programs. In this paper, we introduce LILO, a neurosymbolic framework that iteratively synthesizes, compresses, and documents code to build libraries tailored to particular problem domains. LILO combines LLM-guided program synthesis with recent algorithmic advances in automated refactoring from Stitch: a symbolic compression system that efficiently identifies optimal lambda abstractions across large code corpora. To make these abstractions interpretable, we introduce an auto-documentation (AutoDoc) procedure that infers natural language names and docstrings based on contextual examples of usage. In addition to improving human readability, we find that AutoDoc boosts performance by helping LILO's synthesizer to interpret and deploy learned abstractions. We evaluate LILO on three inductive program synth
    
[^70]: PeTailor：通过定制的分块评分器改进生物医学三元组提取的大型语言模型

    PeTailor: Improving Large Language Model by Tailored Chunk Scorer in Biomedical Triple Extraction

    [https://arxiv.org/abs/2310.18463](https://arxiv.org/abs/2310.18463)

    我们提出了PeTailor，这是一个基于检索的框架，通过使用定制的分块评分器从预先构建的分块数据库中检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，以改进生物医学三元组提取的效果。

    

    生物医学三元组提取系统旨在自动提取生物医学实体和实体之间的关系。虽然当前的统一信息提取模型展示了最先进的性能，但在理解复杂生物医学句子中实体之间的关系方面面临挑战。此外，缺乏高质量的生物医学三元组提取数据集阻碍了稳健的三元组提取系统的开发进展。为了解决这些挑战，我们提出了一种新颖的适用于生物医学三元组提取的基于检索的框架，名为PeTailor，它使用一种新颖的定制分块评分器从我们预先构建的多样分块数据库中显式地检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，为输入的句子生成相应的三元组（头实体，关系，尾实体）。此外，我们还提供了GM-CIHT，一种专家标注的生物医学三元组提取数据集，该数据集支持了我们的方法的实验评估。

    Biomedical triple extraction systems aim to automatically extract biomedical entities and relations between entities. While current unified information extraction models showcase state-of-the-art performance, they face challenges in understanding relationships between entities within intricate biomedical sentences. Furthermore, the absence of a high-quality biomedical triple extraction dataset impedes the progress in developing robust triple extraction systems. To tackle these challenges, we propose a novel retrieval-based framework for biomedical triple extraction, namely PeTailor, which explicitly retrieves the relevant document from our pre-built diverse chunk database using a novel tailored chunk scorer and integrates the retrieved information into the input of a Large Language Model (LLM) to generate the corresponding triple (head entity, relation, tail entity) for the input sentence. Additionally, we present GM-CIHT, an expert-annotated biomedical triple extraction dataset that c
    
[^71]: 探索多语言建模的迷宫

    Exploring the Maze of Multilingual Modeling

    [https://arxiv.org/abs/2310.05404](https://arxiv.org/abs/2310.05404)

    本文综合评估了mBERT、XLM-R和GPT-3等三种流行的多语言语言模型在不同语言上的性能，并研究了资源可用性、语言家族、脚本类型和词序等因素对模型性能的影响。研究结果表明，语言特定预训练数据的数量对模型性能至关重要，同时还发现了其他重要因素。

    

    近年来，多语言语言模型引起了极大关注，可以开发适应不同语言环境的应用。本文对三种流行的多语言语言模型（mBERT、XLM-R和GPT-3）进行了全面评估。我们评估了它们在不同语言上的性能，重点研究了资源可用性（通用和模型特定）、语言家族、脚本类型和词序对模型性能的影响，针对两种不同任务-文本分类和文本生成。我们的研究结果表明，语言特定预训练数据的数量在模型性能中起着关键作用，同时我们也确定了其他因素，如通用资源可用性、语言家族和脚本类型的重要性。我们希望我们的研究能够加深对多语言语言模型的理解，以提高它们在不同语言和语言环境中的性能。

    Multilingual language models have gained significant attention in recent years, enabling the development of applications that meet diverse linguistic contexts. In this paper, we present a comprehensive evaluation of three popular multilingual language models: mBERT, XLM-R, and GPT-3. We assess their performance across a diverse set of languages, with a focus on understanding the impact of resource availability (general and model-specific), language family, script type, and word order on model performance, under two distinct tasks - text classification and text generation. Our findings reveal that while the amount of language-specific pretraining data plays a crucial role in model performance, we also identify other factors such as general resource availability, language family, and script type, as important features. We hope that our study contributes to a deeper understanding of multilingual language models to enhance their performance across languages and linguistic contexts.
    
[^72]: CCPrefix:反事实对比前缀调整用于多类分类

    CCPrefix: Counterfactual Contrastive Prefix-Tuning for Many-Class Classification

    [https://arxiv.org/abs/2211.05987](https://arxiv.org/abs/2211.05987)

    CCPrefix是一种针对多类分类的新型前缀调整方法，利用反事实对比来解决语言表示器模糊性问题。

    

    最近，提出了前缀调整以有效地将预训练语言模型适应广泛的自然语言分类任务。它利用软前缀作为任务特定的指示器和语言表示器作为分类标签的提及，以减少从预训练语言模型到特定任务的差异。然而，当标签空间大幅增加时（即多类分类），这种调整技术会面临语言表示器模糊性问题，因为短语言短句中的类别标签由语义相似的语言表示器表示。为了克服这个问题，受人类决策过程的启发，即每个实例都会考虑最模糊的类别，我们提出了全新的前缀调整方法，即反事实对比前缀调整方法（CCPrefix），用于多类分类。基本上，我们利用标签空间中的事实-反事实对来得到依赖于实例的软前缀，以补充语言表示器。

    Recently, prefix-tuning was proposed to efficiently adapt pre-trained language models to a broad spectrum of natural language classification tasks. It leverages soft prefix as task-specific indicators and language verbalizers as categorical-label mentions to narrow the formulation gap from pre-training language models. However, when the label space increases considerably (i.e., many-class classification), such a tuning technique suffers from a verbalizer ambiguity problem since the many-class labels are represented by semantic-similar verbalizers in short language phrases. To overcome this, inspired by the human-decision process that the most ambiguous classes would be mulled over for each instance, we propose a brand-new prefix-tuning method, Counterfactual Contrastive Prefix-tuning (CCPrefix), for many-class classification. Basically, an instance-dependent soft prefix, derived from fact-counterfactual pairs in the label space, is leveraged to complement the language verbalizers in ma
    
[^73]: LLMs实现可扩展的定性编码：思维链推理在某些解释学任务中能达到人类水平

    Scalable Qualitative Coding with LLMs: Chain-of-Thought Reasoning Matches Human Performance in Some Hermeneutic Tasks. (arXiv:2401.15170v1 [cs.CL])

    [http://arxiv.org/abs/2401.15170](http://arxiv.org/abs/2401.15170)

    本研究证明了大型语言模型在定性编码中的应用潜力。相比于GPT-3.5，GPT-4能够实现与人类相当的解释能力，并具有较高的编码一致性。无论模型规模大小，只要满足一定条件，模型都可以实现较高的编码准确性。

    

    定性编码或内容分析从文本中提取含义，以识别跨文本语料库的定量模式。最近，大型语言模型（LLMs）在解释能力方面的进展为自动化编码过程（对文本应用类别标签）提供了潜力，从而使人类研究人员能够专注于更有创造力的研究方面，同时将这些解释任务委托给人工智能。我们的案例研究包括对人文学研究具有代表性的密集段落的一组社会历史编码。我们发现GPT-4能够达到与人类相当的解释，而GPT-3.5则不能。与我们由人类获得的金标准相比，GPT-4在3个编码中具有优秀的编码一致性（Cohen's κ ≥ 0.79），在9个编码中有8个具有显著的一致性（κ ≥ 0.6）。相比之下，GPT-3.5在所有编码中表现不佳（mean(κ) = 0.34；max(κ) = 0.55）。重要的是，我们发现编码的准确性不受模型规模影响，在满足一定条件的情况下，较小的模型也可以实现较高的编码准确性。

    Qualitative coding, or content analysis, extracts meaning from text to discern quantitative patterns across a corpus of texts. Recently, advances in the interpretive abilities of large language models (LLMs) offer potential for automating the coding process (applying category labels to texts), thereby enabling human researchers to concentrate on more creative research aspects, while delegating these interpretive tasks to AI. Our case study comprises a set of socio-historical codes on dense, paragraph-long passages representative of a humanistic study. We show that GPT-4 is capable of human-equivalent interpretations, whereas GPT-3.5 is not. Compared to our human-derived gold standard, GPT-4 delivers excellent intercoder reliability (Cohen's $\kappa \geq 0.79$) for 3 of 9 codes, and substantial reliability ($\kappa \geq 0.6$) for 8 of 9 codes. In contrast, GPT-3.5 greatly underperforms for all codes ($mean(\kappa) = 0.34$; $max(\kappa) = 0.55$). Importantly, we find that coding fidelity
    
[^74]: PROXYQA：一种用于评估大型语言模型长篇文本生成的替代框架

    PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models. (arXiv:2401.15042v1 [cs.CL])

    [http://arxiv.org/abs/2401.15042](http://arxiv.org/abs/2401.15042)

    PROXYQA是一个用于评估大型语言模型长篇文本生成的替代框架，通过生成详尽的内容，并利用评估器和生成内容作为背景环境，根据评估器回答代理问题的表现来评估生成内容的质量。

    

    大型语言模型（LLM）在长篇文本理解任务中取得了显著的成功。然而，它们生成长篇内容（如报告和文章）的能力尚未得到充分探索。当前的基准不足以充分评估LLMs生成信息丰富且全面的内容，因此需要一种更严格的评估方法。在本研究中，我们介绍了一种名为\textsc{ProxyQA}的框架，用于评估长篇文本生成，包括深入人工策划的涵盖多个领域的“元问题”。每个元问题都包含相应的带注释答案的“代理问题”。LLMs被要求根据这些元问题生成详尽的内容。利用评估器并将生成的内容作为背景环境，\textsc{ProxyQA}根据评估器回答“代理问题”的表现评估生成内容的质量。我们检验了多个LLMs，重点关注了...

    Large Language Models (LLMs) have exhibited remarkable success in long-form context comprehension tasks. However, their capacity to generate long contents, such as reports and articles, remains insufficiently explored. Current benchmarks do not adequately assess LLMs' ability to produce informative and comprehensive content, necessitating a more rigorous evaluation approach. In this study, we introduce \textsc{ProxyQA}, a framework for evaluating long-form text generation, comprising in-depth human-curated \textit{meta-questions} spanning various domains. Each meta-question contains corresponding \textit{proxy-questions} with annotated answers. LLMs are prompted to generate extensive content in response to these meta-questions. Utilizing an evaluator and incorporating generated content as background context, \textsc{ProxyQA} evaluates the quality of generated content based on the evaluator's performance in answering the \textit{proxy-questions}. We examine multiple LLMs, emphasizing \t
    
[^75]: 噪声的力量：重新定义RAG系统的检索

    The Power of Noise: Redefining Retrieval for RAG Systems. (arXiv:2401.14887v1 [cs.IR])

    [http://arxiv.org/abs/2401.14887](http://arxiv.org/abs/2401.14887)

    本研究通过分析和评估检索增强生成（RAG）系统中的信息检索（IR）组件，填补了目前研究中忽视的领域，在有效的RAG的提示表述中，不相关文档的包含可能会对系统性能产生负面影响。

    

    检索增强生成（RAG）系统相对于传统的大型语言模型（LLMs）代表了一个重大进步。RAG系统通过整合通过信息检索（IR）阶段检索的外部数据来增强其生成能力，克服了标准LLMs的限制，后者仅限于其预先训练的知识和有限的上下文窗口。这个领域的大部分研究主要集中在RAG系统内LLMs的生成方面。我们的研究填补了这一空白，通过全面而批判性地分析IR组件对RAG系统的影响。本文分析了一个检索器在有效的RAG的提示表述中应该具备的特征，重点关注应该检索哪种类型的文档。我们评估了各种因素，如文档与提示的相关性，它们的位置以及上下文中包含的数量。我们的发现揭示出，包含不相关的文档可能会…

    Retrieval-Augmented Generation (RAG) systems represent a significant advancement over traditional Large Language Models (LLMs). RAG systems enhance their generation ability by incorporating external data retrieved through an Information Retrieval (IR) phase, overcoming the limitations of standard LLMs, which are restricted to their pre-trained knowledge and limited context window. Most research in this area has predominantly concentrated on the generative aspect of LLMs within RAG systems. Our study fills this gap by thoroughly and critically analyzing the influence of IR components on RAG systems. This paper analyzes which characteristics a retriever should possess for an effective RAG's prompt formulation, focusing on the type of documents that should be retrieved. We evaluate various elements, such as the relevance of the documents to the prompt, their position, and the number included in the context. Our findings reveal, among other insights, that including irrelevant documents can
    
[^76]: 评估大型语言模型上越狱攻击效果的方法研究

    AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models. (arXiv:2401.09002v1 [cs.CL])

    [http://arxiv.org/abs/2401.09002](http://arxiv.org/abs/2401.09002)

    本研究提出一种新方法评估大型语言模型上越狱攻击效果，引入粗粒度和细粒度评估框架，提供了更全面和细致的评估角度，并开发了专门的真实数据集作为基准，为未来研究建立了基础资源。

    

    在我们的研究中，我们开创性地提出了一种评估大型语言模型（LLMs）上越狱攻击效果的新方法，与传统的健壮性评估方法不同。我们的研究引入了两个不同的评估框架：粗粒度评估和细粒度评估。每个框架都使用从0到1的评分范围，提供了独特的视角，能够更全面和细致地评估攻击效果，并帮助攻击者更好地优化攻击提示。此外，我们还开发了一个专门用于越狱任务的全面的真实数据集。这个数据集不仅是我们当前研究的关键基准，也为未来研究建立了一个基础资源，可以在这个不断发展的领域中进行一致和比较的分析。通过与传统评估方法的精心比较，我们发现我们的评估方法与之相一致。

    In our research, we pioneer a novel approach to evaluate the effectiveness of jailbreak attacks on Large Language Models (LLMs), such as GPT-4 and LLaMa2, diverging from traditional robustness-focused binary evaluations. Our study introduces two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework, using a scoring range from 0 to 1, offers a unique perspective, enabling a more comprehensive and nuanced evaluation of attack effectiveness and empowering attackers to refine their attack prompts with greater understanding. Furthermore, we have developed a comprehensive ground truth dataset specifically tailored for jailbreak tasks. This dataset not only serves as a crucial benchmark for our current study but also establishes a foundational resource for future research, enabling consistent and comparative analyses in this evolving field. Upon meticulous comparison with traditional evaluation methods, we discovered that our evaluation alig
    
[^77]: 多语言指令调优中的多语言性

    Multilingual Instruction Tuning With Just a Pinch of Multilinguality. (arXiv:2401.01854v1 [cs.CL])

    [http://arxiv.org/abs/2401.01854](http://arxiv.org/abs/2401.01854)

    本研究研究了多语言指令调优中的多语言性对跨语言指令遵循的影响。研究发现，即使在单语调优过程中，许多语言也可以将一些指令遵循能力转移到其他语言上。此外，只有40个多语言示例能够显著提高多语言指令遵循。总体来说，多语言混合调优的模型在多种语言上的表现相比单语调优的模型要好或者不相上下，尽管使用的这些语言的训练示例数量只有10倍少。

    

    随着大型语言模型（LLMs）的全球采纳，它们在多语言指令遵循能力变得越来越重要。一种有前途的方法是跨语言转移，通过在另一种语言上微调，模型可以在某种语言上获得特定的功能。本文研究了多语言LLM在指令调优过程中的多语言性对跨语言指令遵循的影响。首先我们发现，即使在单语调优过程中，许多语言也可以将一些指令遵循能力转移到其他语言上。此外，我们发现在英语调优集合中，只有40个多语言示例能够显著提高多语言指令遵循，在调优过程中不论是已见语言还是未见语言。总的来说，我们观察到在多语言混合调优的模型在多种语言上的表现相比单语调优的模型要好或者不相上下，尽管使用的这些语言的训练示例数量只有10倍少。

    As instruction-tuned large language models (LLMs) gain global adoption, their ability to follow instructions in multiple languages becomes increasingly crucial. One promising approach is cross-lingual transfer, where a model acquires specific functionality on some language by finetuning on another language. In this work, we investigate how multilinguality during instruction tuning of a multilingual LLM affects instruction-following across languages. We first show that many languages transfer some instruction-following capabilities to other languages from even monolingual tuning. Furthermore, we find that only 40 multilingual examples in an English tuning set substantially improve multilingual instruction-following, both in seen and unseen languages during tuning. In general, we observe that models tuned on multilingual mixtures exhibit comparable or superior performance in several languages compared to monolingually tuned models, despite training on 10x fewer examples in those language
    
[^78]: 自我对弱语言模型进行细调可以将其转化为强语言模型

    Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. (arXiv:2401.01335v1 [cs.LG])

    [http://arxiv.org/abs/2401.01335](http://arxiv.org/abs/2401.01335)

    本文提出了一种名为自我对弱语言模型进行细调（SPIN）的方法，通过模型自我对弈生成训练数据，并从中优化模型策略，从而将弱语言模型转化为强语言模型，无需额外的人类标注数据。

    

    通过监督细调（SFT）利用人类标注数据的力量对于推进大型语言模型（LLMs）至关重要。本文探讨了在不需要获取额外人类标注数据的情况下，将弱语言模型发展成为强语言模型的可能性。我们提出了一种名为自我对弱语言模型进行细调（SPIN）的新的细调方法，该方法从一个经过监督细调的模型开始。SPIN的核心是自我对弱语言模型的机制，其中弱语言模型通过与自身的实例对弈来提升自己的能力。具体而言，弱语言模型通过生成自己的训练数据来优化自身策略，通过区分自我生成的回应与来自人类标注数据的回应来改进。我们的方法逐步将弱语言模型提升为强大的模型，充分发掘人类标注示范数据在SFT中的潜力。在理论上，我们证明了该方法的训练目标函数的全局最优解是可以达到的。

    Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achiev
    
[^79]: 使用生成型人工智能推动学术写作：框架、技术和注意事项

    Supercharging academic writing with generative AI: framework, techniques, and caveats. (arXiv:2310.17143v1 [cs.CY])

    [http://arxiv.org/abs/2310.17143](http://arxiv.org/abs/2310.17143)

    这篇论文介绍了使用生成型人工智能（AI）提高学术写作质量和效率的原则和方法，包括一个人机协作框架、有效的提示技术和两阶段模型，旨在实现认知卸载和想象刺激的AI辅助写作。

    

    学术写作是研究项目中不可或缺但费时费力的部分。本文介绍了使用生成型人工智能（AI）特别是大型语言模型（LLMs）提高学术写作质量和效率的原则和方法。我们提出了一个人机协作框架，详细阐述了AI在写作中的理论基础（为什么）、过程（如何）和性质（什么）。该框架指出了短期和长期参与AI写作的原因及其基本机制（如认知卸载和想象刺激）。它揭示了AI在整个写作过程中的作用，通过一个人机协作写作的两阶段模型和写作辅助类型和级别的模型表示了AI在写作中的帮助方式。基于该框架，我们描述了在写作常规中整合AI的有效提示技术（大纲、起草和编辑）。

    Academic writing is an indispensable yet laborious part of the research enterprise. This Perspective maps out principles and methods for using generative artificial intelligence (AI), specifically large language models (LLMs), to elevate the quality and efficiency of academic writing. We introduce a human-AI collaborative framework that delineates the rationale (why), process (how), and nature (what) of AI engagement in writing. The framework pinpoints both short-term and long-term reasons for engagement and their underlying mechanisms (e.g., cognitive offloading and imaginative stimulation). It reveals the role of AI throughout the writing process, conceptualized through a two-stage model for human-AI collaborative writing, and the nature of AI assistance in writing, represented through a model of writing-assistance types and levels. Building on this framework, we describe effective prompting techniques for incorporating AI into the writing routine (outlining, drafting, and editing) a
    
[^80]: 受控解码来自语言模型

    Controlled Decoding from Language Models. (arXiv:2310.17022v1 [cs.LG])

    [http://arxiv.org/abs/2310.17022](http://arxiv.org/abs/2310.17022)

    本论文提出了一种名为受控解码（CD）的离策略强化学习方法，用于控制语言模型的生成，以达到高回报的结果。CD通过前缀评分器来引导生成，可以在推理时预测预期回报，并且具有模块化设计，可用于解决多目标强化学习问题，而不增加复杂性。

    

    我们提出了一种新颖的离策略强化学习方法，称为受控解码（CD），用于控制自回归语言模型的生成，以获得高回报的结果。CD通过值函数来解决离策略强化学习问题，该值函数被称为前缀评分器。前缀评分器在推理时用于引导生成向更高回报的结果。我们展示了前缀评分器可以从（可能是）离策略数据中训练出来，用于预测从部分解码的响应继续解码时的预期回报。我们在Reddit对话语料库上经验证明，CD作为一种控制机制是有效的。我们还展示了CD设计的模块化使其能够有效解决多目标强化学习问题，而不会增加任何复杂性。最后，我们展示了CD可以以一种新颖的分块方式在推理时应用，同样无需任何额外的操作。

    We propose controlled decoding (CD), a novel off-policy reinforcement learning method to control the autoregressive generation from language models towards high reward outcomes. CD solves an off-policy reinforcement learning problem through a value function for the reward, which we call a prefix scorer. The prefix scorer is used at inference time to steer the generation towards higher reward outcomes. We show that the prefix scorer may be trained on (possibly) off-policy data to predict the expected reward when decoding is continued from a partially decoded response. We empirically demonstrate that CD is effective as a control mechanism on Reddit conversations corpus. We also show that the modularity of the design of CD makes it possible to control for multiple rewards, effectively solving a multi-objective reinforcement learning problem with no additional complexity. Finally, we show that CD can be applied in a novel blockwise fashion at inference-time, again without the need for any 
    
[^81]: 人类课程指导下的指令调整

    Instruction Tuning with Human Curriculum. (arXiv:2310.09518v1 [cs.CL])

    [http://arxiv.org/abs/2310.09518](http://arxiv.org/abs/2310.09518)

    本文探讨了在大型语言模型中应用结构化认知学习方法进行指令调整的潜在好处，并提出了一个高度结构化的合成数据集，结果表明该方法优于传统的随机化方法，提高了指令调整的性能。

    

    指令调整的主流范式是随机洗牌训练最大多样化指令-响应对。本文探讨了在当代大型语言模型如ChatGPT和GPT-4中应用结构化认知学习方法进行指令调整的潜在好处。与以往传统的随机化指令数据集不同，我们提出了一个高度结构化的合成数据集，模拟了人类教育的渐进性和有组织性。我们通过将数据集与教育框架对齐来策划我们的数据集，为每个样本包括主题和认知严谨程度等元信息。我们的数据集涵盖了从中学到研究生阶段的全面细粒度主题，每个主题都有各种问题，以利用布鲁姆的认知分级法提高概念深度，该分级法用于区分每个概念的不同人类认知水平。结果表明，这种认知学习方法优于传统的随机化方法，提高了指令调整的性能。

    The dominant paradigm for instruction tuning is the random-shuffled training of maximally diverse instruction-response pairs. This paper explores the potential benefits of applying a structured cognitive learning approach to instruction tuning in contemporary large language models like ChatGPT and GPT-4. Unlike the previous conventional randomized instruction dataset, we propose a highly structured synthetic dataset that mimics the progressive and organized nature of human education. We curate our dataset by aligning it with educational frameworks, incorporating meta information including its topic and cognitive rigor level for each sample. Our dataset covers comprehensive fine-grained topics spanning diverse educational stages (from middle school to graduate school) with various questions for each topic to enhance conceptual depth using Bloom's taxonomy-a classification framework distinguishing various levels of human cognition for each concept. The results demonstrate that this cogni
    
[^82]: 为程序验证对LLM生成的循环不变式进行排名

    Ranking LLM-Generated Loop Invariants for Program Verification. (arXiv:2310.09342v1 [cs.PL])

    [http://arxiv.org/abs/2310.09342](http://arxiv.org/abs/2310.09342)

    本研究提出了一种针对LLM生成结果进行重新排名的方法，可以显著提高正确不变量的排名，从而减少程序验证的调用次数。

    

    合成归纳循环不变量是自动化程序验证的基础。我们观察到，大型语言模型（如gpt-3.5或gpt-4）能够在0-shot环境下为一类程序合成循环不变量，但需要多个样本才能生成正确的不变量。这可能导致大量调用程序验证器来建立不变性。为了解决这个问题，我们提出了一种对LLM生成结果进行重新排名的方法。我们设计了一个排名器，可以根据问题定义区分正确的归纳不变量和错误的尝试。该排名器经过对比排名优化。实验结果表明，这种重新排名机制显著提高了正确不变量在生成的候选项中的排名，从而大幅减少了对验证器的调用次数。

    Synthesizing inductive loop invariants is fundamental to automating program verification. In this work, we observe that Large Language Models (such as gpt-3.5 or gpt-4) are capable of synthesizing loop invariants for a class of programs in a 0-shot setting, yet require several samples to generate the correct invariants. This can lead to a large number of calls to a program verifier to establish an invariant. To address this issue, we propose a {\it re-ranking} approach for the generated results of LLMs. We have designed a ranker that can distinguish between correct inductive invariants and incorrect attempts based on the problem definition. The ranker is optimized as a contrastive ranker. Experimental results demonstrate that this re-ranking mechanism significantly improves the ranking of correct invariants among the generated candidates, leading to a notable reduction in the number of calls to a verifier.
    
[^83]: CATfOOD：反事实增强训练以提高领域外性能和校准能力

    CATfOOD: Counterfactual Augmented Training for Improving Out-of-Domain Performance and Calibration. (arXiv:2309.07822v1 [cs.CL])

    [http://arxiv.org/abs/2309.07822](http://arxiv.org/abs/2309.07822)

    本研究通过在小型语言模型训练数据中增加自动生成的反事实实例，提高了摘要问答模型在领域外的性能和模型校准能力，并发现性能改进与反事实实例的多样性相关。

    

    在最近的几年中，大型语言模型（LLM）在规模方面展示了显著的能力，特别是在给定提示的条件下生成文本。在我们的研究中，我们探讨了使用LLM来增强小型语言模型（SLM）的训练数据的方法，通过自动生成的反事实（CF）实例（即最小程度的改变输入），以提高SLM在摘要问答（QA）设置下的领域外（OOD）性能。我们证明，在各种LLM生成器中，这种数据增强始终能够提高OOD性能，并改进了基于置信度和基于理性增强的校准模型的模型校准能力。此外，这些性能提升与CF实例在外观形式和语义内容方面的多样性呈正相关。最后，我们证明了校准更容易的CF增强模型在分配重要性时的熵也较低，这表明理性增强的校准器更偏好简洁的解释。

    In recent years, large language models (LLMs) have shown remarkable capabilities at scale, particularly at generating text conditioned on a prompt. In our work, we investigate the use of LLMs to augment training data of small language models~(SLMs) with automatically generated counterfactual~(CF) instances -- i.e. minimally altered inputs -- in order to improve out-of-domain~(OOD) performance of SLMs in the extractive question answering~(QA) setup. We show that, across various LLM generators, such data augmentation consistently enhances OOD performance and improves model calibration for both confidence-based and rationale-augmented calibrator models. Furthermore, these performance improvements correlate with higher diversity of CF instances in terms of their surface form and semantic content. Finally, we show that CF augmented models which are easier to calibrate also exhibit much lower entropy when assigning importance, indicating that rationale-augmented calibrators prefer concise ex
    
[^84]: 基于大型语言模型的评估器是否是扩展多语言评估的解决方案？

    Are Large Language Model-based Evaluators the Solution to Scaling Up Multilingual Evaluation?. (arXiv:2309.07462v1 [cs.CL])

    [http://arxiv.org/abs/2309.07462](http://arxiv.org/abs/2309.07462)

    大型语言模型（LLMs）作为评估器可以解决当前多语言评估的限制和挑战，能够对各种语言中的自然语言处理任务进行有效评估。

    

    大型语言模型（LLMs）在自然语言处理（NLP）任务中展现出了令人印象深刻的性能，如问答、摘要和分类。将LLMs用作评估器，可以对其他模型（通常为LLMs）的输出进行排序或评分，因为当前评估技术存在许多限制，包括缺乏适当的基准、指标、成本和人工标注者的访问性。虽然LLMs能够处理大约100种语言，但大多数语言在各种任务、指标和基准上缺乏系统的评估。这就迫切需要扩展多语言评估，以确保对LLM在各种语言中的性能有准确的理解。基于LLM的评估器似乎是这个问题的完美解决方案，因为它们不需要人工标注者、人工创建的参考和基准，并且理论上可以用来评估任何被覆盖的语言。

    Large Language Models (LLMs) have demonstrated impressive performance on Natural Language Processing (NLP) tasks, such as Question Answering, Summarization, and Classification. The use of LLMs as evaluators, that can rank or score the output of other models (usually LLMs) has become increasingly popular, due to the limitations of current evaluation techniques including the lack of appropriate benchmarks, metrics, cost, and access to human annotators. While LLMs are capable of handling approximately 100 languages, the majority of languages beyond the top 20 lack systematic evaluation across various tasks, metrics, and benchmarks. This creates an urgent need to scale up multilingual evaluation to ensure a precise understanding of LLM performance across diverse languages. LLM-based evaluators seem like the perfect solution to this problem, as they do not require human annotators, human-created references, or benchmarks and can theoretically be used to evaluate any language covered by the 
    
[^85]: LaFiCMIL：从相关多实例学习的角度重新思考大文件分类

    LaFiCMIL: Rethinking Large File Classification from the Perspective of Correlated Multiple Instance Learning. (arXiv:2308.01413v1 [cs.CL])

    [http://arxiv.org/abs/2308.01413](http://arxiv.org/abs/2308.01413)

    LaFiCMIL是一个新的方法，从相关多实例学习的角度解决了Transformer模型输入长度限制的问题，可以用于改进大文件分类任务。

    

    基于Transformer的模型在各种语言任务的性能上取得了革命性的突破。直观上，人们可能会期望文本分类，作为不需要像生成任务那样许多高级表示的任务，能够充分利用Transformer强大的表示能力来进行综合性的处理。然而，实际上，在多类别和多标签分类长文本文档和其他大文件的领域仍然存在较大的改进潜力。Transformer模型的性能主要受到一个重要限制的阻碍：有限的输入长度，比如BERT的512个标记。虽然增加GPU内存可以稍微扩展这个限制，但实际应用中往往受限于有限的GPU资源。在这项工作中，我们从相关多实例学习的角度解决了输入限制问题。所提出的方法LaFiCMIL，作为一个多功能的框架，适用于

    Transformer-based models have revolutionized the performance of a wide range of language tasks. Intuitively, one might expect text classification, which does not necessitate as many high-level representations as generative tasks, to be comprehensively addressed with the powerful representation capabilities of Transformers. However, in reality, there remains significant potential for enhancement, particularly in the areas of multi-class and multi-label classification of lengthy textual documents and other large files. The performance of Transformer-based models is mainly hindered by a major limitation: a restricted input length, e.g., 512 tokens for BERT. While an increase in GPU memory can marginally extend this limit, practical real-world applications often operate under constrained GPU resources. In this work, we tackle the input limit problem from the perspective of correlated multiple instance learning. The proposed approach, LaFiCMIL, serves as a versatile framework applicable to 
    
[^86]: 水印技术用于AI检测的条件文本生成：揭示挑战及语义感知水印解决方案

    Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy. (arXiv:2307.13808v1 [cs.CL])

    [http://arxiv.org/abs/2307.13808](http://arxiv.org/abs/2307.13808)

    本论文研究了在语言模型中嵌入水印以进行AI检测的挑战，并提出了一种简单而有效的语义感知水印算法，该算法在保持检测能力的同时，在不同的文本生成任务中取得了显著改进。

    

    为了减轻语言模型可能带来的潜在风险，最近的AI检测研究提出了通过随机限制词汇并利用此信息进行检测的方式将水印嵌入到机器生成的文本中。虽然这些水印只会导致困惑度轻微下降，但我们的实证调查揭示了对条件文本生成性能的显著影响。为了解决这个问题，我们引入了一个简单但有效的语义感知水印算法，考虑了条件文本生成的特性和输入上下文。实验结果表明，我们提出的方法在各种文本生成模型（包括BART和Flan-T5）中取得了显著改进，在摘要生成和数据到文本生成等任务中仍保持了检测能力。

    To mitigate potential risks associated with language models, recent AI detection research proposes incorporating watermarks into machine-generated text through random vocabulary restrictions and utilizing this information for detection. While these watermarks only induce a slight deterioration in perplexity, our empirical investigation reveals a significant detriment to the performance of conditional text generation. To address this issue, we introduce a simple yet effective semantic-aware watermarking algorithm that considers the characteristics of conditional text generation and the input context. Experimental results demonstrate that our proposed method yields substantial improvements across various text generation models, including BART and Flan-T5, in tasks such as summarization and data-to-text generation while maintaining detection ability.
    
[^87]: AlpaGasus: 用更少数据训练更好的羊驼

    AlpaGasus: Training A Better Alpaca with Fewer Data. (arXiv:2307.08701v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.08701](http://arxiv.org/abs/2307.08701)

    这项研究提出了一种用于训练语言模型的数据筛选策略AlpaGasus，通过使用强大的语言模型过滤掉低质量数据，它在测试中表现出比原始模型更好的性能，并提供了更快的训练速度。

    

    大型语言模型通过在有监督的指令/回复数据上进行指令微调（IFT）来增强其遵循指令的能力。然而，广泛使用的IFT数据集（例如：Alpaca的52k数据）出乎意料地包含许多具有不正确或不相关回复的低质量实例，这些实例会误导和对IFT产生不利影响。在本文中，我们提出了一种简单而有效的数据选择策略，该策略使用强大的语言模型（例如：ChatGPT）自动识别并过滤掉低质量数据。为此，我们引入了AlpaGasus，它仅在从52k Alpaca数据中过滤得到的9k高质量数据上进行微调。AlpaGasus在多个测试数据集和人工评估中均显著优于原始的Alpaca，由GPT-4进行评估。其13B变种在测试任务上的性能与其教师模型语言模型（即生成52k数据的Text-Davinci-003）的性能匹配率超过90％。它还提供了5.7倍更快的训练速度，将7B变种的训练时间从80分钟减少到了...

    Large language models~(LLMs) strengthen instruction-following capability through instruction-finetuning (IFT) on supervised instruction/response data. However, widely used IFT datasets (e.g., Alpaca's 52k data) surprisingly contain many low-quality instances with incorrect or irrelevant responses, which are misleading and detrimental to IFT. In this paper, we propose a simple and effective data selection strategy that automatically identifies and filters out low-quality data using a strong LLM (e.g., ChatGPT). To this end, we introduce AlpaGasus, which is finetuned on only 9k high-quality data filtered from the 52k Alpaca data. AlpaGasus significantly outperforms the original Alpaca as evaluated by GPT-4 on multiple test sets and the controlled human evaluation. Its 13B variant matches $>90\%$ performance of its teacher LLM (i.e., Text-Davinci-003 generating the 52k data) on test tasks. It also provides 5.7x faster training, reducing the training time for a 7B variant from 80 minutes (
    
[^88]: 这片土地是你我的土地：评估语言模型中的地缘政治偏见

    This Land is {Your, My} Land: Evaluating Geopolitical Biases in Language Models. (arXiv:2305.14610v1 [cs.CL])

    [http://arxiv.org/abs/2305.14610](http://arxiv.org/abs/2305.14610)

    本文提出了地缘政治偏见的概念，并以领土争端为例，利用多语言、多选题的数据集BorderLines和几个定量指标分析语言模型响应中的地缘政治偏见现象。

    

    我们引入了地缘政治偏见的概念——即根据语言环境报道不同的地缘政治知识的倾向。我们以领土争端为案例进行了研究。例如，对于被广泛争议的南沙群岛，如果用中文问，LM是否更有可能说它们属于中国，而如果用塔加洛语问，则更有可能说它们属于菲律宾？为了评估是否存在这种偏见，我们首先从维基百科上收集了一组领土争端数据，然后将每个领土与一组多语言、多选题联系起来。这个数据集被称为BorderLines，它包括250个领土和45种语言的问题。我们将这些问题集提交给语言模型，并通过几个提出的定量指标分析它们的响应中地缘政治偏见。这些指标比较不同语言的回答以及实际的地缘政治情况。地缘政治偏见现象是一种独特的跨语言评估。

    We introduce the notion of geopolitical bias -- a tendency to report different geopolitical knowledge depending on the linguistic context. As a case study, we consider territorial disputes between countries. For example, for the widely contested Spratly Islands, would an LM be more likely to say they belong to China if asked in Chinese, vs. to the Philippines if asked in Tagalog? To evaluate if such biases exist, we first collect a dataset of territorial disputes from Wikipedia, then associate each territory with a set of multilingual, multiple-choice questions. This dataset, termed BorderLines, consists of 250 territories with questions in 45 languages. We pose these question sets to language models, and analyze geopolitical bias in their responses through several proposed quantitative metrics. The metrics compare between responses in different question languages as well as to the actual geopolitical situation. The phenomenon of geopolitical bias is a uniquely cross-lingual evaluation
    
[^89]: 使用少量合成数据进行高效的开放领域“多跳问题回答”

    Efficient Open Domain Multi-Hop Question Answering with Few-Shot Data Synthesis. (arXiv:2305.13691v1 [cs.CL])

    [http://arxiv.org/abs/2305.13691](http://arxiv.org/abs/2305.13691)

    提出了一种使用少量合成数据进行高效的开放领域多跳问题回答的方法，可用于改善小型语言模型的性能。通过语言模型和提示参数化的数据生成函数合成数据，微调后的模型参数量只有之前模型的三分之一，达到了与之前模型类似的性能。

    

    对于开放领域“多跳问题回答”，少量数据的学习通常依赖于大型语言模型（LLMs）。虽然强大，但是LLMs在推断时效率低下。我们提出了一种数据合成框架，用于“多跳问题回答”，可使小型语言模型在少于10个人类注释的问题-答案对方面得到改善。该框架建立在由LLMs和提示参数化的数据生成函数之上，这仅需要少量的手工特征。实证上，我们合成了数百万个多跳问题和声称。在合成数据上微调语言模型后，我们在流行的多跳问题回答和事实验证基准上评估模型。我们的实验结果表明，在合成数据上进行微调可以显著改善模型性能，使我们的微调模型在参数量上仅为之前模型的三分之一，同时保持竞争力。

    Few-shot learning for open domain multi-hop question answering typically relies on large language models (LLMs). While powerful, LLMs are inefficient at the inference time. We propose a data synthesis framework for multi-hop question answering that allows for improving smaller language models with less than 10 human-annotated question answer pairs. The framework is built upon the data generation functions parameterized by LLMs and prompts, which requires minimal hand-crafted features. Empirically, we synthesize millions of multi-hop questions and claims. After finetuning language models on the synthetic data, we evaluate the models on popular benchmarks on multi-hop question answering and fact verification. Our experimental results show that finetuning on the synthetic data improves model performance significantly, allowing our finetuned models to be competitive with prior models while being almost one-third the size in terms of parameter counts.
    
[^90]: 学习使用要点标记压缩提示语

    Learning to Compress Prompts with Gist Tokens. (arXiv:2304.08467v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.08467](http://arxiv.org/abs/2304.08467)

    该论文提出了一种名为"gisting"的方法，通过训练语言模型将提示压缩为更小的"要点"标记集，以提高计算效率。通过这种方法，可以实现高达26倍的提示压缩，减少40％的FLOPs、4.2％的墙时速度提升，并节省存储空间，同时最小化输出质量损失。

    

    提示是利用语言模型的多任务能力的主要方式，但是提示占据了输入上下文窗口中宝贵的空间，重复编码相同的提示在计算上是低效的。微调和蒸馏方法可以实现语言模型的专门化，但需要为每个任务重新训练模型。为了完全避免这种权衡，我们提出了gist，它训练一个语言模型将提示压缩成更小的“要点”标记集，可以用于计算效率的缓存和重用。通过简单地修改Transformer的注意力掩码，可以在没有额外成本的情况下对gist模型进行训练，从而实现对提示的高达26倍的压缩，从而减少高达40％的FLOPs、4.2％的墙时速度提升，并节省存储空间，同时最小化输出质量损失。

    Prompting is the primary way to utilize the multitask capabilities of language models (LMs), but prompts occupy valuable space in the input context window, and repeatedly encoding the same prompt is computationally inefficient. Finetuning and distillation methods allow for specialization of LMs without prompting, but require retraining the model for each task. To avoid this trade-off entirely, we present gisting, which trains an LM to compress prompts into smaller sets of "gist" tokens which can be cached and reused for compute efficiency. Gist models can be trained with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks to encourage prompt compression. On decoder (LLaMA-7B) and encoder-decoder (FLAN-T5-XXL) LMs, gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs reductions, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.
    
[^91]: 深层潜在位置主题模型用于带有文本边的网络聚类和表示

    The Deep Latent Position Topic Model for Clustering and Representation of Networks with Textual Edges. (arXiv:2304.08242v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2304.08242](http://arxiv.org/abs/2304.08242)

    深层潜在位置主题模型用于网络聚类和表示，通过基于模型的聚类策略和概率模型对节点和边进行联合表示，并使用模型选择准则进行参数选择。

    

    数值交互导致用户共享其他人发布的文本内容，这些内容自然地由将个体与节点关联和交换的文本定义为边的网络来表示。为了理解这些异构和复杂的数据结构，将节点聚类为同类群组以及呈现可理解的数据可视化是必要的。为了解决这两个问题，我们引入了Deep-LPTM，这是一种基于模型的聚类策略，依赖于变分图自动编码器方法以及概率模型来描述讨论的主题。Deep-LPTM允许在两个嵌入空间中构建节点和边的联合表示。参数使用变分推断算法进行推断。我们还引入了IC2L，这是一种专门设计用于选择具有相关聚类和可视化属性的模型的模型选择准则。对合成数据进行了广泛的基准测试研究。特别是

    Numerical interactions leading to users sharing textual content published by others are naturally represented by a network where the individuals are associated with the nodes and the exchanged texts with the edges. To understand those heterogeneous and complex data structures, clustering nodes into homogeneous groups as well as rendering a comprehensible visualisation of the data is mandatory. To address both issues, we introduce Deep-LPTM, a model-based clustering strategy relying on a variational graph auto-encoder approach as well as a probabilistic model to characterise the topics of discussion. Deep-LPTM allows to build a joint representation of the nodes and of the edges in two embeddings spaces. The parameters are inferred using a variational inference algorithm. We also introduce IC2L, a model selection criterion specifically designed to choose models with relevant clustering and visualisation properties. An extensive benchmark study on synthetic data is provided. In particular
    
[^92]: 大型语言模型可被视为隐含的主题模型：解释和寻找好的示范以实现上下文学习

    Large Language Models Are Implicitly Topic Models: Explaining and Finding Good Demonstrations for In-Context Learning. (arXiv:2301.11916v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.11916](http://arxiv.org/abs/2301.11916)

    本研究发现，大型语言模型可以被视为隐式的主题模型，并提出了一种算法，从注释数据中选择最佳示范，大大提高了上下文学习的能力。

    

    近年来，预训练的大型语言模型表现出了在推理时实现少量样本学习能力的显著效率，被称为上下文学习。 然而，现有文献强调这种能力对少量样本示范的选择很敏感。本研究旨在通过贝叶斯视角研究上下文学习现象，将大型语言模型视为从示范中隐含地推断出相关信息的主题模型。在此前提下，我们提出了一种算法，用于从一组注释数据中选择最佳示范，并证明相对于随机选择基线的平均值，在八个不同的真实文本分类数据集上平均每个 GPT2 和 GPT3 模型有显着的 12.5% 的提升。我们的实证发现支持我们的假设，即大型语言模型可被视为隐含的主题模型。

    In recent years, pre-trained large language models have demonstrated remarkable efficiency in achieving an inference-time few-shot learning capability known as in-context learning. However, existing literature has highlighted the sensitivity of this capability to the selection of few-shot demonstrations. The underlying mechanisms by which this capability arises from regular language model pretraining objectives remain poorly understood. In this study, we aim to examine the in-context learning phenomenon through a Bayesian lens, viewing large language models as topic models that implicitly infer task-related information from demonstrations. On this premise, we propose an algorithm for selecting optimal demonstrations from a set of annotated data and demonstrate a significant 12.5% improvement relative to the random selection baseline, averaged over eight GPT2 and GPT3 models on eight different real-world text classification datasets. Our empirical findings support our hypothesis that la
    
[^93]: 我们能了解甚至无法想象的事物吗？

    What can we know about that which we cannot even imagine?. (arXiv:2208.03886v3 [physics.hist-ph] UPDATED)

    [http://arxiv.org/abs/2208.03886](http://arxiv.org/abs/2208.03886)

    这篇文章探讨了关于智能、人类语言和人类数学的问题，强调了人类语言的局限性，以及我们能否对我们无法想象的事物有任何了解。

    

    在这篇文章中，我将考虑一系列问题。首先，这些问题涉及到智能的生物学功能，特别是人类智能的认知义肢。这将引出关于人类语言的问题，也许是人类迄今为止开发的最重要的认知义肢。虽然传统上对人类语言所包含的认知能力进行赞美，但我将强调人类语言多么有限，因此我们的认知能力也是有限的，尽管语言对其进行了增强。这将引出关于人类数学的问题，因为它最终是以人类语言的形式来表述的，所以也存在深层次的限制。然后，我将结合这些问题，对这篇文章的核心问题提出一个部分性的、有点侧面的答案：我们能够对我们甚至无法构想的事物有何了解？

    In this essay I will consider a sequence of questions. The first questions concern the biological function of intelligence in general, and cognitive prostheses of human intelligence in particular. These will lead into questions concerning human language, perhaps the most important cognitive prosthesis humanity has ever developed. While it is traditional to rhapsodize about the cognitive power encapsulated in human language, I will emphasize how horribly limited human language is -- and therefore how limited our cognitive abilities are, despite their being augmented with language. This will lead to questions of whether human mathematics, being ultimately formulated in terms of human language, is also deeply limited. I will then combine these questions to pose a partial, sort-of, sideways answer to the guiding concern of this essay: what we can ever discern about that we cannot even conceive?
    

