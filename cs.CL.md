# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge.](http://arxiv.org/abs/2305.01651) | 语言模型已被用于知识密集型任务，但其知识随着世界变化不断过时，先前的工作注入单个事实成功，但在基于注入的事实进行推理（或传播这些事实）方面表现不佳。这一研究突显了注入知识传播中的挑战，并提出需要新技术使语言模型能够学习和使用实体的新信息。 |
| [^2] | [Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models.](http://arxiv.org/abs/2305.01645) | 本文研究了压缩模型的高效微调方法，实验结果表明，与手动标注更多的微调数据以直接训练压缩模型相比，从T5-XXL蒸馏到T5-Small几乎总是更具成本效益。 |
| [^3] | [Missing Information, Unresponsive Authors, Experimental Flaws: The Impossibility of Assessing the Reproducibility of Previous Human Evaluations in NLP.](http://arxiv.org/abs/2305.01633) | 该论文研究了NLP领域过去的人类评估的再现性问题，结果发现大部分人类评估都无法重复或再现，可能是由于缺失信息、无回应作者和实验缺陷等原因导致。这个结果提示我们需要重新考虑如何设计和报告人类评估实验。 |
| [^4] | [The Benefits of Bad Advice: Autocontrastive Decoding across Model Layers.](http://arxiv.org/abs/2305.01628) | 本文提出一种新颖的方法，利用语言模型层之间的对比来改进文本生成输出，解决模型在开放式生成中的不良行为问题，并显著提高生成文本的质量。 |
| [^5] | [Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks.](http://arxiv.org/abs/2305.01626) | 该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。 |
| [^6] | [Unlimiformer: Long-Range Transformers with Unlimited Length Input.](http://arxiv.org/abs/2305.01625) | Unlimiformer是一种Transformer模型的通用方法，可以将所有层的注意计算卸载到单个k近邻索引上，从而可处理无限长度的输入，而不增加额外的学习负担。 |
| [^7] | [UNTER: A Unified Knowledge Interface for Enhancing Pre-trained Language Models.](http://arxiv.org/abs/2305.01624) | 本篇论文提出了一种名为UNTER的统一知识接口，可以同时利用结构化和非结构化知识，从而提高预训练语言模型（PLMs）性能，在实验中表现出不断的改进。 |
| [^8] | [A Study on the Integration of Pipeline and E2E SLU systems for Spoken Semantic Parsing toward STOP Quality Challenge.](http://arxiv.org/abs/2305.01620) | 本文介绍了针对ICASSP信号处理大挑战2023中的口语语言理解大挑战的质量轨迹（Track 1）提出的口语语义解析系统。使用了端到端和pipeline系统，并在SLU框架中使用了强大的自动语音识别（ASR）模型和预训练的语言模型（LM）等方法，最终获得了80.8的精确匹配度，获得了挑战的第一名。 |
| [^9] | [FreeLM: Fine-Tuning-Free Language Model.](http://arxiv.org/abs/2305.01616) | 本文提出了一种免调优策略来训练语言模型，该模型考虑了语言信号和教师信号。通过与大型模型相比，实验证明 FreeLM 模型在多种语言理解任务上表现出了更好的性能。 |
| [^10] | [Discern and Answer: Mitigating the Impact of Misinformation in Retrieval-Augmented Models with Discriminators.](http://arxiv.org/abs/2305.01579) | 本文研究了现有检索增强语言模型假设所有检索信息都是正确的假设的问题，在实际应用中可能存在虚假信息导致冲突的情况下，提出了通过精细调整鉴别器和提示鉴别能力引出鲁棒性的方法，这显著改善了模型在知识冲突下的效果；同时提供了关于交替精细调整模型和上下文学习的新的结论。 |
| [^11] | [OTIEA:Ontology-enhanced Triple Intrinsic-Correlation for Cross-lingual Entity Alignment.](http://arxiv.org/abs/2305.01561) | 本研究提出了一种基于本体对和三元素内部交互关系的新型通用EA框架（OTIEA），能够解决目前EA方法中模拟三元素内部交互和考虑可扩展性不足的问题。 |
| [^12] | [Type-enhanced Ensemble Triple Representation via Triple-aware Attention for Cross-lingual Entity Alignment.](http://arxiv.org/abs/2305.01556) | 提出了一种新的跨语言实体对齐框架TTEA，该框架考虑了三元组特征和实体角色多样性，利用三元组感知注意力构建增强型三元组集成表示，通过具体性感知控制了信息传播过程中的噪声影响。 |
| [^13] | [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?.](http://arxiv.org/abs/2305.01555) | 本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。 |
| [^14] | [Mitigating Approximate Memorization in Language Models via Dissimilarity Learned Policy.](http://arxiv.org/abs/2305.01550) | 本论文提出了一种利用强化学习方法缓解语言模型近似记忆化问题的新框架，使用负相似度评分作为奖励信号来学习差异策略。该方法能够有效解决显式和隐式假设所导致的数据结构不完全的问题。 |
| [^15] | [FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information.](http://arxiv.org/abs/2305.01528) | 本研究介绍了一份包含真实游戏状态信息的Dungeons & Dragons实际游戏数据集FIREBALL，它可以改善自然语言生成的质量。此外，LLMs可以使用FIREBALL中的游戏状态信息来生成更高质量的游戏回合。 |
| [^16] | [Huatuo-26M, a Large-scale Chinese Medical QA Dataset.](http://arxiv.org/abs/2305.01526) | 我们发布了一份最大的中医问答数据集，现有模型的表现远低于预期。此数据集可以用于其他QA数据集的零-shot学习并用作检索增强生成（RAG）的外部知识，在预训练语言模型时代仍然具有挑战性。 |
| [^17] | [Beyond Classification: Financial Reasoning in State-of-the-Art Language Models.](http://arxiv.org/abs/2305.01505) | 本研究探讨了大语言模型在财务推理领域的潜在应用，对任务制定、数据生成、提示方法和评估能力等方面进行了详细研究，最终在各种数据集大小上对参数规模为2.8B至13B的各种GPT变体进行基准测试。 |
| [^18] | [NewsPanda: Media Monitoring for Timely Conservation Action.](http://arxiv.org/abs/2305.01503) | NewsPanda是一个可以自动检测和分析与环保和基础建设相关的网上文章的工具。该工具使用主动学习方法和噪声校正算法对BERT模型进行微调以识别相关文章，并提取关键字和找到相关来源。已被世界自然基金会团队在英国、印度和尼泊尔成功部署。 |
| [^19] | [Towards Summarizing Multiple Documents with Hierarchical Relationships.](http://arxiv.org/abs/2305.01498) | 提出了一个新的数据集PeerSum用于生成科学论文的元评论，源文档具有显式层次结构的丰富文档间关系，提出了一种用于元评论生成的关系感知多任务模型Rammer。 |
| [^20] | [Multimodal Neural Databases.](http://arxiv.org/abs/2305.01447) | 本文提出了多模态神经数据库框架（MMNDBs），可以回答涉及文本和图像等不同输入模态的复杂查询，具有类似数据库的功能。 |
| [^21] | [From Local to Global: Navigating Linguistic Diversity in the African Context.](http://arxiv.org/abs/2305.01427) | 摘要介绍了如何应对非洲语言多样性的挑战，并提出了模型作为产品教学的教学工具的想法，该方法对于寻求改善非洲本土方言客户体验和产品开发的企业有重要影响。 |
| [^22] | [Class based Influence Functions for Error Detection.](http://arxiv.org/abs/2305.01384) | 该论文提出了基于类别信息的影响函数来提高异常检测的稳定性和性能。 |
| [^23] | [Turning Flowchart into Dialog: Plan-based Data Augmentation for Low-Resource Flowchart-grounded Troubleshooting Dialogs.](http://arxiv.org/abs/2305.01323) | 本文提出了一个基于计划的数据增强方法，能够将简洁的流程图转化成对话，以生成足够的数据来训练以流程图为基础的故障排除对话系统，实验结果表明该方法有效地提高了系统性能。 |
| [^24] | [Transfer Visual Prompt Generator across LLMs.](http://arxiv.org/abs/2305.01278) | 本论文提出将已有的轻量化视觉提示发生器连接到视觉-语言LLM以减少资源消耗的方法，并提出了跨不同大小和类型的LLMs的VPG转移方案VPGTrans，该方案在VQA和NLVR2任务中表现优秀。 |
| [^25] | [The Role of Summarization in Generative Agents: A Preliminary Perspective.](http://arxiv.org/abs/2305.01253) | 总结是生成智能体最基本、最不可或缺的能力。本报告通过综合观点，尝试理解生成智能体，并推进研究。 |
| [^26] | [Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models.](http://arxiv.org/abs/2305.01219) | 本研究提出一种新颖有效的“ProAttack”方法来执行干净标签的后门攻击，使用的是提示本身作为触发器。该方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性，相比于现有的后门攻击方法有显著提升。 |
| [^27] | [MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset.](http://arxiv.org/abs/2305.01211) | 本文介绍了一个多语言法律句子边界检测数据集MultiLegalSBD，包括6种语言的130,000个注释句子。在该数据集上，现有的SBD模型的表现不佳。作者训练了基于CRF、BiLSTM-CRF和transformers的单语和多语模型，并展示了在该领域中最先进的性能。他们的多语模型在零-shot测试中优于所有基线。 |
| [^28] | [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.](http://arxiv.org/abs/2305.01210) | 本论文提出了一个严格的代码综合基准评估框架EvalPlus，用于评估利用大型语言模型生成的代码的功能正确性。 |
| [^29] | [Topic Shift Detection in Chinese Dialogues: Corpus and Benchmark.](http://arxiv.org/abs/2305.01195) | 本文注释了一个由1308个对话组成的中文自然话题对话语料库，以填补中文自然对话话题语料库的空白。并提出了一种基于分层对比学习的师生框架来预测没有回复的话题转移。 |
| [^30] | [The Pipeline System of ASR and NLU with MLM-based Data Augmentation toward STOP Low-resource Challenge.](http://arxiv.org/abs/2305.01194) | 本文介绍了在低资源适应题目中使用的ASR和NLU的管道方法。在ASR中，使用上采样的Whisper对每个领域进行Feine-tune；在NLU中，使用MLM技术进行数据增强并使用基于检索的方法扩充数据。最终，我们在提醒/天气领域获得了高精确匹配准确度并获得了挑战的第一名。 |
| [^31] | [New Trends in Machine Translation using Large Language Models: Case Examples with ChatGPT.](http://arxiv.org/abs/2305.01181) | 本文提出了使用大型语言模型的机器翻译中的几个新方向，包括风格化MT、交互式MT和基于翻译记忆的MT，并讨论了隐私问题的解决方案。 |
| [^32] | [Lessons Learned in ATCO2: 5000 hours of Air Traffic Control Communications for Robust Automatic Speech Recognition and Understanding.](http://arxiv.org/abs/2305.01155) | 本文讨论了ATCO2项目的经验教训，该项目旨在开发一种唯一平台，以实时方式收集和预处理大量的来自空域的ATC数据。该平台可以作为获得“无限源”的数据。为ATC领域的数据驱动型人工智能系统开发提供了可行的解决方案。 |
| [^33] | [RadAdapt: Radiology Report Summarization via Lightweight Domain Adaptation of Large Language Models.](http://arxiv.org/abs/2305.01146) | 本研究重点研究了轻量化策略，通过在临床文本上进行预训练和在RRS示例上进行参数高效微调，实现适应大型语言模型进行放射性报告摘要（RRS）任务。并且该方法仅微调模型的0.32％的参数，提高了表现。研究结果强调了领域适应在RRS中的重要性，并为开发更好的放射性报告摘要模型提供了有价值的见解。 |
| [^34] | [ADVISE: AI-accelerated Design of Evidence Synthesis for Global Development.](http://arxiv.org/abs/2305.01145) | 该论文研究了如何通过将基于BERT的AI代理融入人类团队中，来加速全球发展证据综述产品的设计；同时还研究了不同的主动学习抽样策略对于协作筛选过程的影响。结果表明，将AI代理整合到人类团队中可以将筛选文档的时间缩短60％，使用主动学习还可以将效率进一步提高20％。 |
| [^35] | [Logion: Machine Learning for Greek Philology.](http://arxiv.org/abs/2305.01099) | 该研究提出了基于机器学习的方法来解决希腊语语言学中的问题，成功利用BERT模型发现和纠正了抄写员在文本传递过程中未被发现的错误，并能在修复预现代手稿材料老化引起的信息缺失方面发挥作用。同时，在领域专家与模型合作时，最佳性能可以通过启示性建议实现。模型的注意力头似乎编码了预现代希腊语的选择性语法特征。 |
| [^36] | [SafeWebUH at SemEval-2023 Task 11: Learning Annotator Disagreement in Derogatory Text: Comparison of Direct Training vs Aggregation.](http://arxiv.org/abs/2305.01050) | 本文研究使用BERT模型来标注侮辱性文本中的注释者不一致性，并比较直接训练和聚合两种方法，结果发现聚合方法比直接训练有更好的效果。 |
| [^37] | [Company classification using zero-shot learning.](http://arxiv.org/abs/2305.01028) | 本文提出了一种利用自然语言处理和零样本学习的方法来进行公司分类的方法。该方法可以简化公司分类过程，从而减少传统方法如全球产业分类标准（GICS）所需的时间和资源。 |
| [^38] | [Evaluating statistical language models as pragmatic reasoners.](http://arxiv.org/abs/2305.01020) | 本文评估了大型语言模型推断语用话语的能力。作者通过表述有关等级形容词“强”的门槛估计来测试 LLM 的性能，并发现 LLM 可以生成具有上下文依赖性、类人的分布，但在组合方面存在困难。 |
| [^39] | [Deception Detection with Feature-Augmentation by soft Domain Transfer.](http://arxiv.org/abs/2305.01011) | 本文提出了一种基于中间层表示的特征增强方法，通过软域转移进行领域间的关联，提高欺骗检测的准确率，分析结果显示推文是检测假新闻和钓鱼电子邮件最有帮助的信息提供者，新闻在推特谣言检测中最有帮助。 |
| [^40] | [TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis.](http://arxiv.org/abs/2305.00976) | 本文介绍了一种名为TMR的方法，用于将文本转换为3D人体运动。它在先前的工作中取得了明显的优势，并通过引入对比性损失的方法来更好地建立跨模态潜在空间结构。结果表明，保持运动生成损失和对比性训练至关重要。 |
| [^41] | [Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding.](http://arxiv.org/abs/2305.00633) | 本论文提出了一种通过自我评估引导解码提高推理的方法，使用经过校准的自动标准探索推理搜索空间，使搜索能够产生更高质量的最终预测结果；使用自我评估引导的随机束搜索在产生推理链的质量和多样性之间平衡权衡，适应多数投票，并且可以准确判断逻辑错误，提高一致性和鲁棒性。 |
| [^42] | [Still no evidence for an effect of the proportion of non-native speakers on language complexity -- A response to Kauhanen, Einhaus & Walkden (2023).](http://arxiv.org/abs/2305.00217) | 本研究为对Kauhanen、Einhaus和Walkden（2023）的回应，仍然没有证据表明大量的L2用户影响语言复杂性。 |
| [^43] | [Towards autonomous system: flexible modular production system enhanced with large language model agents.](http://arxiv.org/abs/2304.14721) | 本论文介绍了一种将大语言模型、数字孪生和工业自动化系统相结合的框架，实现生产过程的智能化规划和控制。通过LLM代理的协调控制，实现了灵活生产的自主规划和控制，能够处理未预定义的任务并规划生产过程。 |
| [^44] | [Sebis at SemEval-2023 Task 7: A Joint System for Natural Language Inference and Evidence Retrieval from Clinical Trial Reports.](http://arxiv.org/abs/2304.13180) | 本文描述了两个NLP系统：一个为自然语言推理，一个为临床试验数据证据检索。它们分别采用了流水线模型和联合模型，并在最终的集成系统中融合输出。 |
| [^45] | [Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis.](http://arxiv.org/abs/2304.04675) | 本文系统地研究了大语言模型在多语机器翻译中的优势和挑战，证明其表现出卓越的潜力。本研究发现LLMs在给定上下文示例时可以意外地忽略提示语义，并且跨语言示例可以为低资源翻译提供更好的任务指导。但实证结果表明，即使是最好的模型ChatGPT仍然落后于监督基线NLLB。 |
| [^46] | [Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark.](http://arxiv.org/abs/2304.03279) | 本文介绍了 MACHIAVELLI 基准测试，用于衡量人工智能代理是否表现出马基雅维利行为，发现了最大化奖励和行为的道德性之间存在权衡，并探索了基于语言模型的方法来减轻这种权衡。 |
| [^47] | [Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning.](http://arxiv.org/abs/2303.10475) | 传统的自然语言处理机器学习需要大规模的任务特定示例，但这不适用于任务可能过于复杂或成本过高以进行注释的场景。因此，社区对于自然语言处理中新的监督寻求范式--从任务指令学习--越来越感兴趣。 |
| [^48] | [AUTODIAL: Efficient Asynchronous Task-Oriented Dialogue Model.](http://arxiv.org/abs/2303.06245) | AUTODIAL是一种多任务对话模型，通过使用并行解码器来执行对话任务，从而显著减少内存占用并实现更快的推理时间。与现有的生成方法相比，AUTODIAL在三个对话任务上提供了3-6倍的速度提升，同时具有11倍的参数减少。这表明将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。 |
| [^49] | [Language Model Analysis for Ontology Subsumption Inference.](http://arxiv.org/abs/2302.06761) | 本文研究了语言模型对本体子类推断的理解能力，提出了一套涉及原子概念和复合概念的推理任务，并证明语言模型对子类推断背景知识的记忆相对较少，但在给定少量样本的情况下可显著提高准确率。 |
| [^50] | [CoRRPUS: Codex-Leveraged Structured Representations for Neurosymbolic Story Understanding.](http://arxiv.org/abs/2212.10754) | 本研究利用Code-LLMs引导符号表示以增强神经符号故事理解，通过CoRRPUS系统和抽象提示程序，在最小的手动工程条件下，击败了当前最先进的结构LLM技术。 |
| [^51] | [Frustratingly Easy Label Projection for Cross-lingual Transfer.](http://arxiv.org/abs/2211.15613) | 本文通过一项广泛的实证研究，对57种语言和三个任务下的跨语言转移进行了研究，并发现优化后的标记-翻译法比传统注释投影方法更有效。 |
| [^52] | [Characterizing Verbatim Short-Term Memory in Neural Language Models.](http://arxiv.org/abs/2210.13569) | 本研究探讨了语言模型的逐字短期记忆特征，发现Transformer模型可以准确检索先前出现过的单词身份和顺序，而LSTM模型的检索能力受限于列表初始标记和短的干扰文本。 |
| [^53] | [Benchmarking Long-tail Generalization with Likelihood Splits.](http://arxiv.org/abs/2210.06799) | 为了可靠地处理自然语言，NLP系统需要推广长尾稀有语句。这篇论文提出了一种使用概率分割来创建有意义测试数据集的方法，引入了更多的挑战性并可能提高在关键任务上的错误率。 |
| [^54] | [Generating Executable Action Plans with Environmentally-Aware Language Models.](http://arxiv.org/abs/2210.04964) | 本文提出了一种带环境感知的语言模型生成可执行的动作计划的方法，通过集成环境对象和对象关系作为额外输入，设计了新颖的评分函数，生成的可执行计划相对于传统的LLM方法有更高成功率。 |
| [^55] | [Large scale analysis of gender bias and sexism in song lyrics.](http://arxiv.org/abs/2208.02052) | 本文对377808首英文歌曲歌词进行大规模的自然语言处理分析，揭示了及时的性别歧视的增加以及不同性别表演者的语言偏见。 |
| [^56] | [Know your audience: specializing grounded language models with listener subtraction.](http://arxiv.org/abs/2206.08349) | 本文介绍了一种利用多智能体图像参照游戏自适应不同听众的目标任务描述的方法，并通过微调 CLIP 视觉编码器和大型语言模型之间的适配器，在适应听众的语言上下文的情况下进行了自然语言专业化。 |
| [^57] | [End-to-End Training for Back-Translation with Categorical Reparameterization Trick.](http://arxiv.org/abs/2202.08465) | 本文提出了一种基于分类重新参数化技巧的回译端到端训练方法，来有效地减少两个神经机器翻译模型间离散属性的影响，从而实现端到端式的训练，获得了比以前基准测试更好的BLEU分数。 |
| [^58] | [Towards Learning to Speak and Hear Through Multi-Agent Communication over a Continuous Acoustic Channel.](http://arxiv.org/abs/2111.02827) | 本研究旨在通过提供一个智能体间的消息传递环境，使得智能体能够通过连续声学通道进行通讯并观察到新兴语言的产生与特点，结果表明：与离散型信号不同，声学讲话者学习使用冗余信息以提高侦听者的连贯性。 |
| [^59] | [Word Embeddings: A Survey.](http://arxiv.org/abs/1901.09069) | 这篇综述介绍了一些主要的词向量构建策略，称为word embeddings，这些策略基于分布假设，编码了语法和语义信息，并被证明在很多NLP任务中是有用的额外特征。 |

# 详细

[^1]: 语言模型能够从描述中学习新实体吗？注入知识传播中的挑战

    Can LMs Learn New Entities from Descriptions? Challenges in Propagating Injected Knowledge. (arXiv:2305.01651v1 [cs.CL])

    [http://arxiv.org/abs/2305.01651](http://arxiv.org/abs/2305.01651)

    语言模型已被用于知识密集型任务，但其知识随着世界变化不断过时，先前的工作注入单个事实成功，但在基于注入的事实进行推理（或传播这些事实）方面表现不佳。这一研究突显了注入知识传播中的挑战，并提出需要新技术使语言模型能够学习和使用实体的新信息。

    

    预先训练的语言模型已经被用于像问答这样的知识密集型任务，但是随着世界的变化，它们的知识不断过时。先前的工作研究了对语言模型进行有针对性的更新，注入个别的事实并评估模型是否能够学习这些事实，同时不改变其他上下文的预测。本文进一步研究了语言模型基于注入的事实进行推理的能力（或传播这些事实的能力）：例如，在学习了某个东西是电视节目之后，语言模型是否会预测你可以通过它来观看? 我们通过两种填空式任务来研究这一问题：一个是关于新实体的现实世界句子的现有数据集（ECBD），另一个是一个新的受控基准测试，其中手动设计的模板要求注入的知识具有不同程度的推理。令人惊讶的是，我们发现现有的更新知识的方法（基于梯度微调和此方法的修改）在注入知识的传播方面表现不佳。这些方法在学习个别事实方面非常成功，但在从注入知识中进行推理方面却存在困难。我们的研究突显了注入知识传播中的挑战，并建议需要新的技术来使语言模型能够学习和使用关于实体的新信息。

    Pre-trained language models (LMs) are used for knowledge intensive tasks like question answering, but their knowledge gets continuously outdated as the world changes. Prior work has studied targeted updates to LMs, injecting individual facts and evaluating whether the model learns these facts while not changing predictions on other contexts. We take a step forward and study LMs' abilities to make inferences based on injected facts (or propagate those facts): for example, after learning that something is a TV show, does an LM predict that you can watch it? We study this with two cloze-style tasks: an existing dataset of real-world sentences about novel entities (ECBD) as well as a new controlled benchmark with manually designed templates requiring varying levels of inference about injected knowledge. Surprisingly, we find that existing methods for updating knowledge (gradient-based fine-tuning and modifications of this approach) show little propagation of injected knowledge. These metho
    
[^2]: 压缩模型的高效微调：蒸馏还是标注？

    Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models. (arXiv:2305.01645v1 [cs.CL])

    [http://arxiv.org/abs/2305.01645](http://arxiv.org/abs/2305.01645)

    本文研究了压缩模型的高效微调方法，实验结果表明，与手动标注更多的微调数据以直接训练压缩模型相比，从T5-XXL蒸馏到T5-Small几乎总是更具成本效益。

    

    大型模型的微调虽然效果显著，但使用这些模型进行推理成本高且会产生碳排放。知识蒸馏已被证明是减少推理成本的实用解决方案，但蒸馏过程本身需要大量计算资源。本文研究了如何最有效地使用固定预算构建压缩模型。在六个不同的自然语言处理任务上进行了大量实验后，我们发现从T5-XXL（11B）蒸馏到T5-Small（60M）几乎总是比手动标注更多的微调数据以直接训练一个压缩模型（T5-Small（60M））更具成本效益。我们进一步展示了最大化效用的最佳蒸馏量因任务而异。

    Fine-tuning large models is highly effective, however, inference using these models can be expensive and produces carbon emissions. Knowledge distillation has been shown to be a practical solution to reduce inference costs, but the distillation process itself requires significant computational resources. Rather than buying or renting GPUs to fine-tune, then distill a large model, an NLP practitioner who needs a compact model might also choose to simply allocate an available budget to hire annotators and manually label additional fine-tuning data. In this paper, we investigate how to most efficiently use a fixed budget to build a compact model. Through our extensive experiments on six diverse NLP tasks, we find that distilling from T5-XXL (11B) to T5-Small (60M) leads to almost always a cost-efficient option compared to annotating more data to directly train a compact model (T5-Small (60M)). We further demonstrate that the optimal amount of distillation that maximizes utility varies acr
    
[^3]: 缺失信息、无回应作者、实验缺陷：NLP中不可能评估以前的人类评估的再现性。

    Missing Information, Unresponsive Authors, Experimental Flaws: The Impossibility of Assessing the Reproducibility of Previous Human Evaluations in NLP. (arXiv:2305.01633v1 [cs.CL])

    [http://arxiv.org/abs/2305.01633](http://arxiv.org/abs/2305.01633)

    该论文研究了NLP领域过去的人类评估的再现性问题，结果发现大部分人类评估都无法重复或再现，可能是由于缺失信息、无回应作者和实验缺陷等原因导致。这个结果提示我们需要重新考虑如何设计和报告人类评估实验。

    

    我们报告了我们在识别一组先前适合进行协调研究的NLP领域人类评估的努力，以考察是什么使得NLP领域的人类评估更/ less能再现。我们提供了我们的结果和发现，其中包括仅有13％的论文具有（i）足够低的再现障碍，以及（ii）足够的可获取信息，才可以被考虑进行再现，并且我们选择进行再现的所有实验都被发现存在缺陷，这使得进行再现的有意义性值得怀疑。因此，我们不得不将我们的协调研究设计从再现方法更改为标准化-然后再现两次的方法。我们总体而言（是负面的）发现，NLP领域中的绝大多数人类评估都不能重复和/或不能再现和/或具有太多缺陷以证明其可再现性。这描绘了一个可怕的画面，但也为重新考虑如何设计和报告NLP领域的人类评估提供了机会。

    We report our efforts in identifying a set of previous human evaluations in NLP that would be suitable for a coordinated study examining what makes human evaluations in NLP more/less reproducible. We present our results and findings, which include that just 13\% of papers had (i) sufficiently low barriers to reproduction, and (ii) enough obtainable information, to be considered for reproduction, and that all but one of the experiments we selected for reproduction was discovered to have flaws that made the meaningfulness of conducting a reproduction questionable. As a result, we had to change our coordinated study design from a reproduce approach to a standardise-then-reproduce-twice approach. Our overall (negative) finding that the great majority of human evaluations in NLP is not repeatable and/or not reproducible and/or too flawed to justify reproduction, paints a dire picture, but presents an opportunity for a rethink about how to design and report human evaluations in NLP.
    
[^4]: 坏建议的好处：模型层间自动对照解码

    The Benefits of Bad Advice: Autocontrastive Decoding across Model Layers. (arXiv:2305.01628v1 [cs.CL])

    [http://arxiv.org/abs/2305.01628](http://arxiv.org/abs/2305.01628)

    本文提出一种新颖的方法，利用语言模型层之间的对比来改进文本生成输出，解决模型在开放式生成中的不良行为问题，并显著提高生成文本的质量。

    

    在自然语言处理任务中，应用语言模型通常依赖于最终模型层的表示，因为假设中间隐藏层的表示是不太有用的。本文认为由于模型层之间的渐进改进，可以从更高层和更低层之间的对比中获取额外信息。具体来说，在选择生成模型的下一个可能标记的预测时，可以使用较低层的预测来突出哪些候选项是最好避免的。我们提出了一种新颖的方法，利用层之间的对比来改进文本生成输出，并表明它可以缓解模型在开放式生成中的不良行为，显著提高生成的文本质量。此外，我们的结果表明，在推断时比较模型层之间可以对一些总体语言模型能力的方面产生实质性的好处。

    Applying language models to natural language processing tasks typically relies on the representations in the final model layer, as intermediate hidden layer representations are presumed to be less informative. In this work, we argue that due to the gradual improvement across model layers, additional information can be gleaned from the contrast between higher and lower layers during inference. Specifically, in choosing between the probable next token predictions of a generative model, the predictions of lower layers can be used to highlight which candidates are best avoided. We propose a novel approach that utilizes the contrast between layers to improve text generation outputs, and show that it mitigates degenerative behaviors of the model in open-ended generation, significantly improving the quality of generated texts. Furthermore, our results indicate that contrasting between model layers at inference time can yield substantial benefits to certain aspects of general language model ca
    
[^5]: 基于语音的基础语法：自发联接的自监督深度神经网络

    Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks. (arXiv:2305.01626v1 [cs.CL])

    [http://arxiv.org/abs/2305.01626](http://arxiv.org/abs/2305.01626)

    该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。

    

    语法的计算模型主要基于文本。本文提出了一种完全无监督的方法，可以直接从原始语音中建立基础语法模型。我们重点研究了最普遍和基本的语法特性之一——联接。我们介绍了自发联接现象：卷积神经网络(CNN)在个别单词的声学记录上训练时，开始产生输出，这些输出将两个甚至三个单词连接在一起，而不会接触到具有多个单词的输入数据。此外，训练两个单词的网络可以学习将单词嵌入到新的未见过的单词组合中。据我们所知，这是在生成对抗网络环境下训练的原始语音CNN以前未报道的属性，它不仅对我们理解这些体系结构的学习方式有影响，还对建立从原始声学输入中的语法及其演化的模型有影响。

    Computational models of syntax are predominantly text-based. Here we propose that basic syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and basic properties of syntax -- concatenation. We introduce spontaneous concatenation: a phenomenon where convolutional neural networks (CNNs) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the input. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. To our knowledge, this is a previously unreported property of CNNs trained on raw speech in the Generative Adversarial Network setting and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution from raw acoustic inputs.
    
[^6]: 无限长度输入的长距离Transformer-Unlimiformer

    Unlimiformer: Long-Range Transformers with Unlimited Length Input. (arXiv:2305.01625v1 [cs.CL])

    [http://arxiv.org/abs/2305.01625](http://arxiv.org/abs/2305.01625)

    Unlimiformer是一种Transformer模型的通用方法，可以将所有层的注意计算卸载到单个k近邻索引上，从而可处理无限长度的输入，而不增加额外的学习负担。

    

    基于Transformer的模型通常对输入长度有预定义的限制，因为它们可能需要参考输入中的每个标记。本文提出了一种通用方法-Unlimiformer，可以包装任何现有的预训练编码器-解码器Transformer，并将所有层的注意计算卸载到单个k近邻索引上。我们在几个长文档和多文档摘要基准测试中证明了Unlimiformer的有效性，展示了它可以总结350k令牌长的输入而不进行测试时的截断。

    Transformer-based models typically have a predefined bound to their input length, because of their need to potentially attend to every token in the input. In this work, we propose Unlimiformer: a general approach that can wrap any existing pretrained encoder-decoder transformer, and offload the attention computation across all layers to a single $k$-nearest-neighbor index; this index can be kept on either the GPU or CPU memory and queried in sub-linear time. This way, we can index extremely long input sequences, while every attention head in every decoder layer retrieves its top-$k$ keys, instead of attending to every key. We demonstrate Unlimiformers's efficacy on several long-document and multi-document summarization benchmarks, showing that it can summarize even 350k token-long inputs from the BookSum dataset, without any input truncation at test time. Unlimiformer improves pretrained models such as BART and Longformer by extending them to unlimited inputs without additional learned
    
[^7]: UNTER: 一种用于增强预训练语言模型的统一知识接口

    UNTER: A Unified Knowledge Interface for Enhancing Pre-trained Language Models. (arXiv:2305.01624v1 [cs.CL])

    [http://arxiv.org/abs/2305.01624](http://arxiv.org/abs/2305.01624)

    本篇论文提出了一种名为UNTER的统一知识接口，可以同时利用结构化和非结构化知识，从而提高预训练语言模型（PLMs）性能，在实验中表现出不断的改进。

    

    最近的研究表明，外部知识注入可以提高预训练语言模型（PLMs）在各种下游NLP任务中的性能。但是，现有的知识注入方法适用于结构化知识或非结构化知识，缺乏统一的使用方式。本文提出了一种名为UNTER的统一知识接口，以提供利用结构化知识和非结构化知识的统一视角。在UNTER中，我们采用解码器作为统一的知识接口，将从编码器获取的跨度表示与其对应的知识进行对齐。这种方法使编码器能够从其参数中统一调用下游应用程序的跨度相关的知识。实验结果表明，通过注入两种形式的知识，UNTER在一系列知识驱动的NLP任务中获得了不断的改进，包括实体类型、命名实体识别和关系抽取，尤其在低资源场景中效果明显。

    Recent research demonstrates that external knowledge injection can advance pre-trained language models (PLMs) in a variety of downstream NLP tasks. However, existing knowledge injection methods are either applicable to structured knowledge or unstructured knowledge, lacking a unified usage. In this paper, we propose a UNified knowledge inTERface, UNTER, to provide a unified perspective to exploit both structured knowledge and unstructured knowledge. In UNTER, we adopt the decoder as a unified knowledge interface, aligning span representations obtained from the encoder with their corresponding knowledge. This approach enables the encoder to uniformly invoke span-related knowledge from its parameters for downstream applications. Experimental results show that, with both forms of knowledge injected, UNTER gains continuous improvements on a series of knowledge-driven NLP tasks, including entity typing, named entity recognition and relation extraction, especially in low-resource scenarios.
    
[^8]: 一项关于将Pipeline和E2E SLU系统整合用于口语语义解析的研究：STOP品质挑战

    A Study on the Integration of Pipeline and E2E SLU systems for Spoken Semantic Parsing toward STOP Quality Challenge. (arXiv:2305.01620v1 [cs.CL])

    [http://arxiv.org/abs/2305.01620](http://arxiv.org/abs/2305.01620)

    本文介绍了针对ICASSP信号处理大挑战2023中的口语语言理解大挑战的质量轨迹（Track 1）提出的口语语义解析系统。使用了端到端和pipeline系统，并在SLU框架中使用了强大的自动语音识别（ASR）模型和预训练的语言模型（LM）等方法，最终获得了80.8的精确匹配度，获得了挑战的第一名。

    

    最近，有一些新的基准任务被引入用于口语理解（SLU），例如语义解析。本文介绍了我们针对ICASSP信号处理大挑战2023中的口语语言理解大挑战的质量轨迹（Track 1）提出的口语语义解析系统。我们尝试了端到端和pipeline系统来完成这项任务。我们在SLU框架中使用了强大的自动语音识别（ASR）模型，如Whisper，以及预训练的语言模型（LM），如BART，来提高性能。我们还研究了不同模型的输出级别组合，以获得80.8的精确匹配度，这使我们在挑战中获得了第一名。

    Recently there have been efforts to introduce new benchmark tasks for spoken language understanding (SLU), like semantic parsing. In this paper, we describe our proposed spoken semantic parsing system for the quality track (Track 1) in Spoken Language Understanding Grand Challenge which is part of ICASSP Signal Processing Grand Challenge 2023. We experiment with both end-to-end and pipeline systems for this task. Strong automatic speech recognition (ASR) models like Whisper and pretrained Language models (LM) like BART are utilized inside our SLU framework to boost performance. We also investigate the output level combination of various models to get an exact match accuracy of 80.8, which won the 1st place at the challenge.
    
[^9]: FreeLM: 免调优语言模型

    FreeLM: Fine-Tuning-Free Language Model. (arXiv:2305.01616v1 [cs.CL])

    [http://arxiv.org/abs/2305.01616](http://arxiv.org/abs/2305.01616)

    本文提出了一种免调优策略来训练语言模型，该模型考虑了语言信号和教师信号。通过与大型模型相比，实验证明 FreeLM 模型在多种语言理解任务上表现出了更好的性能。

    

    预训练语言模型 (PLMs) 在 NLP 任务中取得了显著的成功。尽管如此，主流的解决方案仍然遵循预训练后调优的范式，这既带来了高昂的部署成本，也降低了训练效率。然而，调优特定任务是必要的，因为 PLMs 仅在大型原始数据的语言信号下进行了预训练。本文提出了一种新颖的无调优策略，即同时考虑语言信号和教师信号。教师信号是下游任务的一个抽象表示，以统一命题格式提供。我们的 FreeLM 模型在交互式地使用语言信号和强任务感知的教师信号进行训练后表现出了强大的泛化和鲁棒性。在实验中，FreeLM 在多种语言理解任务上优于大型模型，如 GPT-3 和 InstructGPT。与这些模型的 175B 参数相比，FreeLM 更小，只有 0.3B 参数。

    Pre-trained language models (PLMs) have achieved remarkable success in NLP tasks. Despite the great success, mainstream solutions largely follow the pre-training then finetuning paradigm, which brings in both high deployment costs and low training efficiency. Nevertheless, fine-tuning on a specific task is essential because PLMs are only pre-trained with language signal from large raw data. In this paper, we propose a novel fine-tuning-free strategy for language models, to consider both language signal and teacher signal. Teacher signal is an abstraction of a battery of downstream tasks, provided in a unified proposition format. Trained with both language and strong task-aware teacher signals in an interactive manner, our FreeLM model demonstrates strong generalization and robustness. FreeLM outperforms large models e.g., GPT-3 and InstructGPT, on a range of language understanding tasks in experiments. FreeLM is much smaller with 0.3B parameters, compared to 175B in these models.
    
[^10]: 区分和回答：通过辨别器缓解检索增强模型中虚假信息的影响

    Discern and Answer: Mitigating the Impact of Misinformation in Retrieval-Augmented Models with Discriminators. (arXiv:2305.01579v1 [cs.CL])

    [http://arxiv.org/abs/2305.01579](http://arxiv.org/abs/2305.01579)

    本文研究了现有检索增强语言模型假设所有检索信息都是正确的假设的问题，在实际应用中可能存在虚假信息导致冲突的情况下，提出了通过精细调整鉴别器和提示鉴别能力引出鲁棒性的方法，这显著改善了模型在知识冲突下的效果；同时提供了关于交替精细调整模型和上下文学习的新的结论。

    

    大多数现有的检索增强语言模型（LM）假定所有检索到的信息都是事实上正确的。本文研究一个更加现实的场景，即检索到的文档可能包含虚假信息，从而导致它们之间存在冲突。我们观察到，现有模型在精调和上下文少样本学习设置中对这种信息高度脆弱。我们提出了一些方法，通过明确地对鉴别器进行精细调整或提示来引出GPT-3的鉴别能力，使检索增强LM对虚假信息具有鲁棒性。我们在开放域问答方面的实证结果表明，这些方法显著改善了LM对知识冲突的鲁棒性。我们还提供了关于交替精细调整模型的决策与上下文学习过程的发现，为利用两者的最佳方式铺平了新的道路。

    Most existing retrieval-augmented language models (LMs) for question answering assume all retrieved information is factually correct. In this work, we study a more realistic scenario in which retrieved documents may contain misinformation, causing conflicts among them. We observe that the existing models are highly brittle to such information in both fine-tuning and in-context few-shot learning settings. We propose approaches to make retrieval-augmented LMs robust to misinformation by explicitly fine-tuning a discriminator or prompting to elicit discrimination capability in GPT-3. Our empirical results on open-domain question answering show that these approaches significantly improve LMs' robustness to knowledge conflicts. We also provide our findings on interleaving the fine-tuned model's decision with the in-context learning process, paving a new path to leverage the best of both worlds.
    
[^11]: OTIEA: 基于本体增强的三元内部相关性用于跨语言实体对齐

    OTIEA:Ontology-enhanced Triple Intrinsic-Correlation for Cross-lingual Entity Alignment. (arXiv:2305.01561v1 [cs.CL])

    [http://arxiv.org/abs/2305.01561](http://arxiv.org/abs/2305.01561)

    本研究提出了一种基于本体对和三元素内部交互关系的新型通用EA框架（OTIEA），能够解决目前EA方法中模拟三元素内部交互和考虑可扩展性不足的问题。

    

    在没有足够外部资源的情况下进行跨语言和跨领域知识对齐是融合不规则数据的基本和关键任务。实体对齐（EA）作为逐元素融合过程旨在从不同的知识图谱（KGs）中发现等价的对象，近年来在行业和学术界引起了极大的兴趣。现有的大多数EA方法通常通过邻节点、结构信息和外部资源来探索实体和关系之间的相关性。然而，这些方法中很少模拟三元素之间的复杂内在交互和角色信息，这可能导致三元素的不充分说明。此外，在某些场景中，外部资源通常不可用，特别是跨语言和跨领域应用，这反映了这些方法的可扩展性不足。为了解决上述不足，提出了一种基于本体对和三元素内部交互关系的新型通用EA框架（OTIEA）。

    Cross-lingual and cross-domain knowledge alignment without sufficient external resources is a fundamental and crucial task for fusing irregular data. As the element-wise fusion process aiming to discover equivalent objects from different knowledge graphs (KGs), entity alignment (EA) has been attracting great interest from industry and academic research recent years. Most of existing EA methods usually explore the correlation between entities and relations through neighbor nodes, structural information and external resources. However, the complex intrinsic interactions among triple elements and role information are rarely modeled in these methods, which may lead to the inadequate illustration for triple. In addition, external resources are usually unavailable in some scenarios especially cross-lingual and cross-domain applications, which reflects the little scalability of these methods. To tackle the above insufficiency, a novel universal EA framework (OTIEA) based on ontology pair and 
    
[^12]: 借助三元组感知注意力的增强型三元组集成表示方法，用于跨语言实体对齐

    Type-enhanced Ensemble Triple Representation via Triple-aware Attention for Cross-lingual Entity Alignment. (arXiv:2305.01556v1 [cs.CL])

    [http://arxiv.org/abs/2305.01556](http://arxiv.org/abs/2305.01556)

    提出了一种新的跨语言实体对齐框架TTEA，该框架考虑了三元组特征和实体角色多样性，利用三元组感知注意力构建增强型三元组集成表示，通过具体性感知控制了信息传播过程中的噪声影响。

    

    实体对齐是将跨语言和跨领域的知识图谱中指向同一实际对象的实体进行发现的关键任务。目前大多数现有的方法是通过基于嵌入的方法挖掘三元组元素的相关性来生成对齐实体表示，很少关注三元组的不可分性和实体角色的多样性。本文提出了一种新颖的框架TTEA（类型增强的三元组集成表示），通过考虑集成三元组的具体性和实体角色特征来克服上述问题。具体而言，通过将关系作为信息载体在语义空间和类型空间之间进行转换，从而可以通过具体性感知的三元组注意力平稳地控制空间转换和信息传播过程中的噪声影响。此外，我们的框架使用了...

    Entity alignment(EA) is a crucial task for integrating cross-lingual and cross-domain knowledge graphs(KGs), which aims to discover entities referring to the same real-world object from different KGs. Most existing methods generate aligning entity representation by mining the relevance of triple elements via embedding-based methods, paying little attention to triple indivisibility and entity role diversity. In this paper, a novel framework named TTEA -- Type-enhanced Ensemble Triple Representation via Triple-aware Attention for Cross-lingual Entity Alignment is proposed to overcome the above issues considering ensemble triple specificity and entity role features. Specifically, the ensemble triple representation is derived by regarding relation as information carrier between semantic space and type space, and hence the noise influence during spatial transformation and information propagation can be smoothly controlled via specificity-aware triple attention. Moreover, our framework uses 
    
[^13]: 如何发挥大语言模型在少样本关系抽取中的能力？

    How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?. (arXiv:2305.01555v1 [cs.CL])

    [http://arxiv.org/abs/2305.01555](http://arxiv.org/abs/2305.01555)

    本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。

    

    语言模型的扩展已经彻底改变了广泛的自然语言处理任务，但是使用大型语言模型进行少样本关系抽取还没有得到全面探索。本文通过详细实验，研究了使用GPT-3.5进行少样本关系抽取的基本方法——上下文学习和数据生成。为了增强少样本性能，我们进一步提出了与任务相关的指导说明和约束模式下的数据生成。我们观察到，在上下文学习的情况下，可以实现与以前的提示学习方法相当的性能，而使用大型语言模型的数据生成可以推动以前的解决方案以在四个广泛研究的关系抽取数据集上获得新的最先进的少样本结果。我们希望我们的工作可以激发未来对大型语言模型在少样本关系抽取中的能力的研究。代码可以在 \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm} 中找到。

    Scaling language models have revolutionized widespread NLP tasks, yet little comprehensively explored few-shot relation extraction with large language models. In this paper, we investigate principal methodologies, in-context learning and data generation, for few-shot relation extraction via GPT-3.5 through exhaustive experiments. To enhance few-shot performance, we further propose task-related instructions and schema-constrained data generation. We observe that in-context learning can achieve performance on par with previous prompt learning approaches, and data generation with the large language model can boost previous solutions to obtain new state-of-the-art few-shot results on four widely-studied relation extraction datasets. We hope our work can inspire future research for the capabilities of large language models in few-shot relation extraction. Code is available in \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^14]: 通过学习的差异策略缓解语言模型中的近似记忆化问题

    Mitigating Approximate Memorization in Language Models via Dissimilarity Learned Policy. (arXiv:2305.01550v1 [cs.CL])

    [http://arxiv.org/abs/2305.01550](http://arxiv.org/abs/2305.01550)

    本论文提出了一种利用强化学习方法缓解语言模型近似记忆化问题的新框架，使用负相似度评分作为奖励信号来学习差异策略。该方法能够有效解决显式和隐式假设所导致的数据结构不完全的问题。

    

    大型语言模型（LLMs）是通过大量数据进行训练的，其中可能包含会危害个人隐私的敏感信息。LLMs被证明会记忆训练数据的部分内容并在遇到对应提示时直接输出该数据。先前的研究主要集中在数据预处理和差分隐私技术上以解决记忆化问题，但这些方法都依赖于对要保护数据结构的显式和隐式假设，导致问题解决不完全。为了解决这个问题，我们提出了一种新的框架，利用强化学习方法（PPO）微调LLMs以缓解近似记忆化问题。我们的方法使用负相似度评分（例如BERTScore或SacreBLEU）作为奖励信号来学习差异策略。我们的结果表明，这种框架有效地缓解了近似记忆化问题。

    Large Language models (LLMs) are trained on large amounts of data, which can include sensitive information that may compromise per- sonal privacy. LLMs showed to memorize parts of the training data and emit those data verbatim when an adversary prompts appropriately. Previous research has primarily focused on data preprocessing and differential privacy techniques to address memorization or prevent verbatim memorization exclusively, which can give a false sense of privacy. However, these methods rely on explicit and implicit assumptions about the structure of the data to be protected, which often results in an incomplete solution to the problem. To address this, we propose a novel framework that utilizes a reinforcement learning approach (PPO) to fine-tune LLMs to mitigate approximate memorization. Our approach utilizes a negative similarity score, such as BERTScore or SacreBLEU, as a reward signal to learn a dissimilarity policy. Our results demonstrate that this framework effectively 
    
[^15]: FIREBALL：一份包含结构化游戏状态信息的Dungeons & Dragons实际游戏数据集

    FIREBALL: A Dataset of Dungeons and Dragons Actual-Play with Structured Game State Information. (arXiv:2305.01528v1 [cs.CL])

    [http://arxiv.org/abs/2305.01528](http://arxiv.org/abs/2305.01528)

    本研究介绍了一份包含真实游戏状态信息的Dungeons & Dragons实际游戏数据集FIREBALL，它可以改善自然语言生成的质量。此外，LLMs可以使用FIREBALL中的游戏状态信息来生成更高质量的游戏回合。

    

    Dungeons & Dragons（D＆D）是一款桌面角色扮演游戏，其玩家之间存在复杂的自然语言交互和隐藏的状态信息。最近的研究表明，拥有状态信息的大型语言模型（LLMs）生成的游戏回合比仅使用对话历史的LLMs更具高质量。然而，以往的研究使用的游戏状态信息是启发式创建的，并不是真正的黄金标准游戏状态。我们提出了FIREBALL，这是一个包含真实游戏状态信息的大型数据集，其中包含来自Discord的近25,000个真实D＆D游戏会话。我们记录了使用Avrae机器人的玩家的游戏会话，该机器人是为了帮助人们在线玩D＆D而开发的，并捕获了语言、游戏命令和基础游戏状态信息。我们证明，通过使用Avrae状态信息，FIREBALL可以提高自然语言生成（NLG），从而提高自动评估指标和人类的质量评判。此外，我们还展示了LLMs可以生成…

    Dungeons & Dragons (D&D) is a tabletop roleplaying game with complex natural language interactions between players and hidden state information. Recent work has shown that large language models (LLMs) that have access to state information can generate higher quality game turns than LLMs that use dialog history alone. However, previous work used game state information that was heuristically created and was not a true gold standard game state. We present FIREBALL, a large dataset containing nearly 25,000 unique sessions from real D\&D gameplay on Discord with true game state info. We recorded game play sessions of players who used the Avrae bot, which was developed to aid people in playing D&D online, capturing language, game commands and underlying game state information. We demonstrate that FIREBALL can improve natural language generation (NLG) by using Avrae state information, improving both automated metrics and human judgments of quality. Additionally, we show that LLMs can generate
    
[^16]: Huatuo-26M：一份大规模的中医问答数据集

    Huatuo-26M, a Large-scale Chinese Medical QA Dataset. (arXiv:2305.01526v1 [cs.CL])

    [http://arxiv.org/abs/2305.01526](http://arxiv.org/abs/2305.01526)

    我们发布了一份最大的中医问答数据集，现有模型的表现远低于预期。此数据集可以用于其他QA数据集的零-shot学习并用作检索增强生成（RAG）的外部知识，在预训练语言模型时代仍然具有挑战性。

    

    本文中，我们发布了一份有着2600万个问答对的中医问答数据集，此为迄今为止最大型的数据集。我们以检索和生成两个方面对现有方法在此数据集上进行了基准测试。实验结果表明，现有模型的表现远低于预期，而且在预训练语言模型时代，发布的数据集仍然具有挑战性。此外，我们还实验性地展示了所提议数据集在以下方面的益处：（i）以零-shot的方式为其他问答数据集训练模型；（ii）作为检索增强生成（RAG）的外部知识；（iii）通过将问答对作为预训练语料库进行连续训练方式，提升现有预训练语言模型的性能。我们相信该数据集不仅将有助于医学研究，还将促进病人和临床医生的服务。请参考：\url{https://github.com/FreedomIntelligence/Huatuo-26M}。

    In this paper, we release a largest ever medical Question Answering (QA) dataset with 26 million QA pairs. We benchmark many existing approaches in our dataset in terms of both retrieval and generation. Experimental results show that the existing models perform far lower than expected and the released dataset is still challenging in the pre-trained language model era. Moreover, we also experimentally show the benefit of the proposed dataset in many aspects: (i) trained models for other QA datasets in a zero-shot fashion; and (ii) as external knowledge for retrieval-augmented generation (RAG); and (iii) improving existing pre-trained language models by using the QA pairs as a pre-training corpus in continued training manner. We believe that this dataset will not only contribute to medical research but also facilitate both the patients and clinical doctors. See \url{https://github.com/FreedomIntelligence/Huatuo-26M}.
    
[^17]: 超越分类：最先进的语言模型中的财务推理

    Beyond Classification: Financial Reasoning in State-of-the-Art Language Models. (arXiv:2305.01505v1 [cs.CL])

    [http://arxiv.org/abs/2305.01505](http://arxiv.org/abs/2305.01505)

    本研究探讨了大语言模型在财务推理领域的潜在应用，对任务制定、数据生成、提示方法和评估能力等方面进行了详细研究，最终在各种数据集大小上对参数规模为2.8B至13B的各种GPT变体进行基准测试。

    

    大语言模型(LLMs)由1000亿及以上的参数组成，在复杂的多步推理任务中表现出了非凡的能力。然而，这种通用的进展应用在很少领域中，例如临床或法律领域，而财务推理领域基本上未被探索。据我们所知，LLMs解决财务推理问题的能力从未被研究过，并且它是否可以在任何规模上完成仍未知。为了填补这一知识空白，本研究对LLMs在财务领域的潜在应用进行了全面调查。调查包括对一系列主题的详细探讨，包括任务制定，合成数据生成，提示方法和评估能力。此外，本研究在不同的数据集大小上对参数规模为2.8B至13B的各种GPT变体进行基准测试，包括有无指导调整。

    Large Language Models (LLMs), consisting of 100 billion or more parameters, have demonstrated remarkable ability in complex multi-step reasoning tasks. However, the application of such generic advancements has been limited to a few fields, such as clinical or legal, with the field of financial reasoning remaining largely unexplored. To the best of our knowledge, the ability of LLMs to solve financial reasoning problems has never been dealt with, and whether it can be performed at any scale remains unknown. To address this knowledge gap, this research presents a comprehensive investigation into the potential application of LLMs in the financial domain. The investigation includes a detailed exploration of a range of subjects, including task formulation, synthetic data generation, prompting methods, and evaluation capability. Furthermore, the study benchmarks various GPT variants with parameter scales ranging from 2.8B to 13B, with and without instruction tuning, on diverse dataset sizes.
    
[^18]: NewsPanda: 用于及时保护行动的媒体监控工具

    NewsPanda: Media Monitoring for Timely Conservation Action. (arXiv:2305.01503v1 [cs.IR])

    [http://arxiv.org/abs/2305.01503](http://arxiv.org/abs/2305.01503)

    NewsPanda是一个可以自动检测和分析与环保和基础建设相关的网上文章的工具。该工具使用主动学习方法和噪声校正算法对BERT模型进行微调以识别相关文章，并提取关键字和找到相关来源。已被世界自然基金会团队在英国、印度和尼泊尔成功部署。

    

    环保非政府组织对监测相关媒体并及时了解基础建设项目的更新具有重要兴趣，因为这些项目可能会对重要的保护区域产生巨大影响。然而，这种监测很难且需要耗费时间。我们介绍了一个名为NewsPanda的工具包，它可以自动检测和分析与环保和基础建设相关的网上文章。我们使用主动学习方法和噪声校正算法对BERT模型进行微调，以识别与保护和基础建设相关的文章。对已识别出的文章，我们进行进一步的分析，提取关键字并找到可能相关的来源。NewsPanda自2022年2月以来已被世界自然基金会团队在英国、印度和尼泊尔成功部署。它目前监测了印度80,000个网站和1,074个保护地点。

    Non-governmental organizations for environmental conservation have a significant interest in monitoring conservation-related media and getting timely updates about infrastructure construction projects as they may cause massive impact to key conservation areas. Such monitoring, however, is difficult and time-consuming. We introduce NewsPanda, a toolkit which automatically detects and analyzes online articles related to environmental conservation and infrastructure construction. We fine-tune a BERT-based model using active learning methods and noise correction algorithms to identify articles that are relevant to conservation and infrastructure construction. For the identified articles, we perform further analysis, extracting keywords and finding potentially related sources. NewsPanda has been successfully deployed by the World Wide Fund for Nature teams in the UK, India, and Nepal since February 2022. It currently monitors over 80,000 websites and 1,074 conservation sites across India an
    
[^19]: 旨在总结带有层次关系的多篇文档

    Towards Summarizing Multiple Documents with Hierarchical Relationships. (arXiv:2305.01498v1 [cs.CL])

    [http://arxiv.org/abs/2305.01498](http://arxiv.org/abs/2305.01498)

    提出了一个新的数据集PeerSum用于生成科学论文的元评论，源文档具有显式层次结构的丰富文档间关系，提出了一种用于元评论生成的关系感知多任务模型Rammer。

    

    多数现存的多文档摘要(MDS)数据集缺少人工生成的、真实的(即非合成的)摘要或者带有显式文档间关系的源文档。为了增强MDS系统的能力，我们提出PeerSum，这是一个新颖的数据集，用于生成科学论文的元评论，其中元评论是对评论和相应讨论的高度概括且真实的摘要。这些源文档具有显式层次结构的丰富文档间关系，包括交叉引用和经常出现的冲突。鉴于很少有研究采用基于预训练语言模型的注意力操纵来将层次关系纳入MDS系统中，我们还提出了Rammer(关系感知多任务元评论生成器)，这是一种元评论生成模型，使用基于层次关系的稀疏注意力和多任务目标，可以预测多个度量值。

    Most existing multi-document summarization (MDS) datasets lack human-generated and genuine (i.e., not synthetic) summaries or source documents with explicit inter-document relationships that a summary must capture. To enhance the capabilities of MDS systems we present PeerSum, a novel dataset for generating meta-reviews of scientific papers, where the meta-reviews are highly abstractive and genuine summaries of reviews and corresponding discussions. These source documents have rich inter-document relationships of an explicit hierarchical structure with cross-references and often feature conflicts. As there is a scarcity of research that incorporates hierarchical relationships into MDS systems through attention manipulation on pre-trained language models, we additionally present Rammer (Relationship-aware Multi-task Meta-review Generator), a meta-review generation model that uses sparse attention based on the hierarchical relationships and a multi-task objective that predicts several me
    
[^20]: 多模态神经数据库

    Multimodal Neural Databases. (arXiv:2305.01447v1 [cs.MM])

    [http://arxiv.org/abs/2305.01447](http://arxiv.org/abs/2305.01447)

    本文提出了多模态神经数据库框架（MMNDBs），可以回答涉及文本和图像等不同输入模态的复杂查询，具有类似数据库的功能。

    

    近年来，文本、图像和其他模态的松散结构数据的增加呼吁新的查询方法。多媒体信息检索填补了这个空白并在最近几年取得了令人兴奋的进展。检索大规模多媒体档案的任务已经经历了巨大的性能提升，这在很大程度上是由于多模态深度学习的最近发展所推动的。然而，这个领域的方法在它们所支持的查询类型上仍然受到限制，尤其是它们无法回答类似数据库的查询。因此，受神经数据库的最新工作启发，我们提出了一个新框架，命名为多模态神经数据库（MMNDBs）。MMNDBs可以回答涉及不同输入模态（例如文本和图像）的推理的复杂类似数据库的查询。在本文中，我们提出了第一个能够满足这一系列要求的架构，并通过几个基线测试了它的性能。

    The rise in loosely-structured data available through text, images, and other modalities has called for new ways of querying them. Multimedia Information Retrieval has filled this gap and has witnessed exciting progress in recent years. Tasks such as search and retrieval of extensive multimedia archives have undergone massive performance improvements, driven to a large extent by recent developments in multimodal deep learning. However, methods in this field remain limited in the kinds of queries they support and, in particular, their inability to answer database-like queries. For this reason, inspired by recent work on neural databases, we propose a new framework, which we name Multimodal Neural Databases (MMNDBs). MMNDBs can answer complex database-like queries that involve reasoning over different input modalities, such as text and images, at scale. In this paper, we present the first architecture able to fulfill this set of requirements and test it with several baselines, showing th
    
[^21]: 从本地到全球：应对非洲语言多样性的挑战

    From Local to Global: Navigating Linguistic Diversity in the African Context. (arXiv:2305.01427v1 [cs.CL])

    [http://arxiv.org/abs/2305.01427](http://arxiv.org/abs/2305.01427)

    摘要介绍了如何应对非洲语言多样性的挑战，并提出了模型作为产品教学的教学工具的想法，该方法对于寻求改善非洲本土方言客户体验和产品开发的企业有重要影响。

    

    本文探讨了自然语言处理（NLP）中存在的有关非洲大陆上语言多样性的关键问题，特别是非洲的地方方言和鲜为人知的阿拉伯方言。我们评估了各种方法，展示了它们的有效性，同时强调了所提出方法对于寻求改进非洲本土方言的客户体验和产品开发的企业可能产生的潜在影响。使用该模型作为产品教学的教学工具的想法也很有趣，因为这可能会激发学习者的兴趣并引发科技创业。总的来说，我们的改进方法提供了一个应对非洲本土方言挑战的有前途的分析。特别是鲜为人知的阿拉伯方言，这对于寻求改进客户体验和产品开发的企业可能会产生重大影响。

    The focus is on critical problems in NLP related to linguistic diversity and variation across the African continent, specifically with regards to African local di- alects and Arabic dialects that have received little attention. We evaluated our various approaches, demonstrating their effectiveness while highlighting the potential impact of the proposed approach on businesses seeking to improve customer experience and product development in African local dialects. The idea of using the model as a teaching tool for product-based instruction is interesting, as it could potentially stimulate interest in learners and trigger techno entrepreneurship. Overall, our modified approach offers a promising analysis of the challenges of dealing with African local dialects. Particularly Arabic dialects, which could have a significant impact on businesses seeking to improve customer experience and product development.
    
[^22]: 基于类的影响函数用于误差检测

    Class based Influence Functions for Error Detection. (arXiv:2305.01384v1 [cs.CL])

    [http://arxiv.org/abs/2305.01384](http://arxiv.org/abs/2305.01384)

    该论文提出了基于类别信息的影响函数来提高异常检测的稳定性和性能。

    

    影响函数(IFs)是在大规模数据集中检测异常样本的强大工具。然而，当应用于深度网络时，它们不稳定。本文解释了IFs不稳定的原因，并提出了解决这个问题的方法。我们表明，在两个数据点属于两个不同类别时，IFs是不可靠的。我们的解决方法利用类别信息来改进IFs的稳定性。大量实验证明，我们的修改显著提高了IFs的性能和稳定性，同时不增加任何计算成本。

    Influence functions (IFs) are a powerful tool for detecting anomalous examples in large scale datasets. However, they are unstable when applied to deep networks. In this paper, we provide an explanation for the instability of IFs and develop a solution to this problem. We show that IFs are unreliable when the two data points belong to two different classes. Our solution leverages class information to improve the stability of IFs. Extensive experiments show that our modification significantly improves the performance and stability of IFs while incurring no additional computational cost.
    
[^23]: 将流程图转化为对话：基于计划的数据增强方法，用于低资源流程图相关故障排除对话

    Turning Flowchart into Dialog: Plan-based Data Augmentation for Low-Resource Flowchart-grounded Troubleshooting Dialogs. (arXiv:2305.01323v1 [cs.CL])

    [http://arxiv.org/abs/2305.01323](http://arxiv.org/abs/2305.01323)

    本文提出了一个基于计划的数据增强方法，能够将简洁的流程图转化成对话，以生成足够的数据来训练以流程图为基础的故障排除对话系统，实验结果表明该方法有效地提高了系统性能。

    

    近年来，以流程图为基础的故障排除对话系统（FTD系统）一直备受研究关注。然而，收集充分的自然基于流程图的对话数据成本较高，因此FTD系统受到数据稀缺的限制。为了缓解数据稀疏性问题，我们提出了基于计划的数据增强（PlanDA）方法，通过将简洁的流程图转化为对话，生成大量多样的合成对话数据。具体来说，它的生成模型采用具有全局和局部潜在规划变量的分层规划策略的变分基框架。在FloDial数据集上的实验表明，PlanDA生成的合成对话改善了下游任务的性能，包括流程图路径检索和响应生成，特别是在流程图以外的情况下。

    Flowchart-grounded troubleshooting dialogue (FTD) systems, which follow the instructions of a flowchart to diagnose users' problems in specific domains (eg., vehicle, laptop), have been gaining research interest in recent years. However, collecting sufficient dialogues that are naturally grounded on flowcharts is costly, thus FTD systems are impeded by scarce training data. To mitigate the data sparsity issue, we propose a plan-based data augmentation (PlanDA) approach that generates diverse synthetic dialog data at scale by transforming concise flowchart into dialogues. Specifically, its generative model employs a variational-base framework with a hierarchical planning strategy that includes global and local latent planning variables. Experiments on the FloDial dataset show that synthetic dialogue produced by PlanDA improves the performance of downstream tasks, including flowchart path retrieval and response generation, in particular on the Out-of-Flowchart settings. In addition, furt
    
[^24]: 横向迁移轻量化视觉提示发生器在VL-LLMs之间的应用研究

    Transfer Visual Prompt Generator across LLMs. (arXiv:2305.01278v1 [cs.CV])

    [http://arxiv.org/abs/2305.01278](http://arxiv.org/abs/2305.01278)

    本论文提出将已有的轻量化视觉提示发生器连接到视觉-语言LLM以减少资源消耗的方法，并提出了跨不同大小和类型的LLMs的VPG转移方案VPGTrans，该方案在VQA和NLVR2任务中表现优秀。

    

    本文研究利用现有的轻量化视觉提示发生器（VPG）连接已有的视觉-语言LLM（VL-LLM）以减少资源消耗。此外，我们提出一种跨不同大小和类型的LLMs的VPG转移方案。基于我们的观察，我们设计了一个名为VPGTrans的两阶段转移框架，它在VQA和NLVR2两个下游任务中表现出比现有方法更好的精度和转移速度。

    While developing a new vision-language LLM (VL-LLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG part of the VL-LLM still suffers from indispensable computational costs, i.e., requiring thousands of GPU hours and millions of training data. One alternative solution is to transfer an existing VPG from any existing VL-LLMs for the target VL-LLM.  In this work, we for the first time investigate the VPG transferability across LLMs, and explore a solution to reduce the cost of VPG transfer. We first study the VPG transfer across different LLM sizes (e.g., small-to-large), and across different LLM types, through which we diagnose the key factors to maximize the transfer efficiency. Based on our observation, we design a two-stage transfer framework named VPGTrans, which is simple yet highly e
    
[^25]: 总结在生成智能体中的作用:初步观点

    The Role of Summarization in Generative Agents: A Preliminary Perspective. (arXiv:2305.01253v1 [cs.CL])

    [http://arxiv.org/abs/2305.01253](http://arxiv.org/abs/2305.01253)

    总结是生成智能体最基本、最不可或缺的能力。本报告通过综合观点，尝试理解生成智能体，并推进研究。

    

    模拟人类社会的生成智能体展现出更进一步研究和实际应用的巨大潜力。其中，由多个细致设计的模块构成的生成智能体架构是最关键的组成部分。为了推进这一研究，本报告提出了我们的综合观点，认为通过总结理解生成智能体是至关重要和不可或缺的能力，这种能力会在各种场景下表现出来。我们希望这份报告能够提供洞见，帮助人们理解总结能力在生成智能体中的重要性，并激发未来的研究。

    Generative agents that simulate human society show tremendous potential for further research and practical applications. Specifically, the generative agent architecture comprising several meticulously designed modules constitutes the most critical component. To facilitate progress in this research, this report presents our integrated perspective on comprehending generative agents through summarization, since we believe summarization is the most fundamental and indispensable capacity of generative agents manifested across diverse scenarios. We hope this report can provide insight into understanding the importance of summarization capacity in generative agents and motivate future research.
    
[^26]: 触发词作为后门攻击的触发器：检查语言模型的脆弱性

    Prompt as Triggers for Backdoor Attack: Examining the Vulnerability in Language Models. (arXiv:2305.01219v1 [cs.CL])

    [http://arxiv.org/abs/2305.01219](http://arxiv.org/abs/2305.01219)

    本研究提出一种新颖有效的“ProAttack”方法来执行干净标签的后门攻击，使用的是提示本身作为触发器。该方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性，相比于现有的后门攻击方法有显著提升。

    

    基于提示的学习范例弥合了预训练和微调之间的差距，在几个NLP任务中取得了最先进的性能，尤其是在少样本情况下。尽管应用广泛，但基于提示的学习容易受到后门攻击。文本后门攻击旨在通过注入触发器并修改标签来在模型中引入有针对性的漏洞。然而，由于触发器的存在和毒瘤数据标注不正确等缺陷，这种攻击存在异常的自然语言表达。在本研究中，我们提出了一种新颖有效的“ProAttack”方法，基于提示来执行干净标签的后门攻击，使用的是提示本身作为触发器。我们的方法不需要外部触发器，并确保毒瘤数据的标注正确，提高了后门攻击的隐蔽性。通过在丰富的资源和少样本文本语料库上的广泛实验，我们证明了ProAttack方法在保持干净数据一致性的同时显著优于现有的后门攻击方式。

    The prompt-based learning paradigm, which bridges the gap between pre-training and fine-tuning, achieves state-of-the-art performance on several NLP tasks, particularly in few-shot settings. Despite being widely applied, prompt-based learning is vulnerable to backdoor attacks. Textual backdoor attacks are designed to introduce targeted vulnerabilities into models by poisoning a subset of training samples through trigger injection and label modification. However, they suffer from flaws such as abnormal natural language expressions resulting from the trigger and incorrect labeling of poisoned samples. In this study, we propose {\bf ProAttack}, a novel and efficient method for performing clean-label backdoor attacks based on the prompt, which uses the prompt itself as a trigger. Our method does not require external triggers and ensures correct labeling of poisoned samples, improving the stealthy nature of the backdoor attack. With extensive experiments on rich-resource and few-shot text c
    
[^27]: MultiLegalSBD：一个多语言法律句子边界检测数据集

    MultiLegalSBD: A Multilingual Legal Sentence Boundary Detection Dataset. (arXiv:2305.01211v1 [cs.CL])

    [http://arxiv.org/abs/2305.01211](http://arxiv.org/abs/2305.01211)

    本文介绍了一个多语言法律句子边界检测数据集MultiLegalSBD，包括6种语言的130,000个注释句子。在该数据集上，现有的SBD模型的表现不佳。作者训练了基于CRF、BiLSTM-CRF和transformers的单语和多语模型，并展示了在该领域中最先进的性能。他们的多语模型在零-shot测试中优于所有基线。

    

    句子边界检测是自然语言处理的基础之一，不正确的分割会严重影响下游任务的输出质量。对于算法来说是一个具有挑战性的任务，尤其对于法律领域，因为使用的复杂句子结构各不相同。本文精心策划了一个多语种法律数据集，包括6种语言的130,000个注释句子。我们的实验结果表明，现有的SBD模型在多语种法律数据上的性能表现不佳。我们训练和测试了基于CRF、BiLSTM-CRF和transformers的单语和多语模型，展示了最先进的性能。我们还展示了我们的多语模型在葡萄牙语测试集的零-shot设置中优于所有基线。为了鼓励社区进一步的研究和发展，我们已经公开了我们的数据集、模型和代码。

    Sentence Boundary Detection (SBD) is one of the foundational building blocks of Natural Language Processing (NLP), with incorrectly split sentences heavily influencing the output quality of downstream tasks. It is a challenging task for algorithms, especially in the legal domain, considering the complex and different sentence structures used. In this work, we curated a diverse multilingual legal dataset consisting of over 130'000 annotated sentences in 6 languages. Our experimental results indicate that the performance of existing SBD models is subpar on multilingual legal data. We trained and tested monolingual and multilingual models based on CRF, BiLSTM-CRF, and transformers, demonstrating state-of-the-art performance. We also show that our multilingual models outperform all baselines in the zero-shot setting on a Portuguese test set. To encourage further research and development by the community, we have made our dataset, models, and code publicly available.
    
[^28]: ChatGPT生成的代码真的正确吗？对大型语言模型在代码生成方面的严格评估

    Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. (arXiv:2305.01210v1 [cs.SE])

    [http://arxiv.org/abs/2305.01210](http://arxiv.org/abs/2305.01210)

    本论文提出了一个严格的代码综合基准评估框架EvalPlus，用于评估利用大型语言模型生成的代码的功能正确性。

    

    程序综合一直以来都是被长期研究的领域，最近的方法集中于直接利用大型语言模型(LLMs)根据自然语言中用户的意图生成代码。代码评估数据集，包含策划好的综合问题和各种输入/输出测试用例，被用来衡量各种LLMs在代码综合上的性能。然而，这些数据集中的测试用例在完全评估生成代码的功能正确性方面，数量和质量都可能有所限制。这种现有基准中的限制引出了以下问题：在LLMs时代，生成的代码真的正确吗？为了回答这个问题，我们提出了EvalPlus——一个评估LLM-synthesized代码功能正确性的严格基准评估框架。EvalPlus接受基础评估数据集，并利用自动输入生成步骤，使用LLM-based和基于变异的方法生成和多样化大量新的测试输入。

    Program synthesis has been long studied with recent approaches focused on directly using the power of Large Language Models (LLMs) to generate code according to user intent written in natural language. Code evaluation datasets, containing curated synthesis problems with input/output test-cases, are used to measure the performance of various LLMs on code synthesis. However, test-cases in these datasets can be limited in both quantity and quality for fully assessing the functional correctness of the generated code. Such limitation in the existing benchmarks begs the following question: In the era of LLMs, is the code generated really correct? To answer this, we propose EvalPlus -- a code synthesis benchmarking framework to rigorously evaluate the functional correctness of LLM-synthesized code. In short, EvalPlus takes in the base evaluation dataset and uses an automatic input generation step to produce and diversify large amounts of new test inputs using both LLM-based and mutation-based
    
[^29]: 中文对话中的话题转移检测：语料库与基准

    Topic Shift Detection in Chinese Dialogues: Corpus and Benchmark. (arXiv:2305.01195v1 [cs.CL])

    [http://arxiv.org/abs/2305.01195](http://arxiv.org/abs/2305.01195)

    本文注释了一个由1308个对话组成的中文自然话题对话语料库，以填补中文自然对话话题语料库的空白。并提出了一种基于分层对比学习的师生框架来预测没有回复的话题转移。

    

    对话话题转移检测是指检测对话中正在进行的话题是否转移或应该转移。任务可分为已知回复任务和未知回复任务。目前，只有少数针对后者进行了研究，因为在没有回复信息的情况下预测话题转移仍然是一项挑战。本文首先注释了一个由1308个对话组成的中文自然话题对话（CNTD）语料库，以填补中文自然会话话题语料库的空白。然后，我们专注于未知回复任务，并提出了一种基于分层对比学习的师生框架来预测没有回复的话题转移。具体而言，在高级师生响应中引入对比学习，用于建立响应和上下文之间的对比学习，而在低级学生中构建标签对比学习。在我们的中文CNTD和英文TIAGE上的实验结果表明了其有效性。

    Dialogue topic shift detection is to detect whether an ongoing topic has shifted or should shift in a dialogue, which can be divided into two categories, i.e., response-known task and response-unknown task. Currently, only a few investigated the latter, because it is still a challenge to predict the topic shift without the response information. In this paper, we first annotate a Chinese Natural Topic Dialogue (CNTD) corpus consisting of 1308 dialogues to fill the gap in the Chinese natural conversation topic corpus. And then we focus on the response-unknown task and propose a teacher-student framework based on hierarchical contrastive learning to predict the topic shift without the response. Specifically, the response at high-level teacher-student is introduced to build the contrastive learning between the response and the context, while the label contrastive learning is constructed at low-level student. The experimental results on our Chinese CNTD and English TIAGE show the effectiven
    
[^30]: 基于MLM数据增强的ASR和NLU管道系统应对STOP低资源挑战

    The Pipeline System of ASR and NLU with MLM-based Data Augmentation toward STOP Low-resource Challenge. (arXiv:2305.01194v1 [cs.CL])

    [http://arxiv.org/abs/2305.01194](http://arxiv.org/abs/2305.01194)

    本文介绍了在低资源适应题目中使用的ASR和NLU的管道方法。在ASR中，使用上采样的Whisper对每个领域进行Feine-tune；在NLU中，使用MLM技术进行数据增强并使用基于检索的方法扩充数据。最终，我们在提醒/天气领域获得了高精确匹配准确度并获得了挑战的第一名。

    

    本文描述了我们在ICASSP信号处理大赛2023的口语理解大挑战（Spoken Language Understanding Grand Challenge）低资源领域适应赛道（Track3）中采用的ASR和NLU的管道方法。针对ASR，我们使用上采样 fine-tune Whisper 以适应每个领域。针对NLU，我们 fine-tune BART 在所有 Track3 数据上，然后在低资源域数据上进行 fine-tune。我们应用了基于遮盖的LM（MLM）数据增强，其中一些输入标记和相应的目标标签使用 MLM 进行替换。我们还采用了基于检索的方法，模型输入与类似的训练样本一起进行增强。结果，我们在提醒/天气领域实现了63.3 / 75.0（平均：69.15）的精确匹配（EM）准确度，获得了该挑战的第一名。

    This paper describes our system for the low-resource domain adaptation track (Track 3) in Spoken Language Understanding Grand Challenge, which is a part of ICASSP Signal Processing Grand Challenge 2023. In the track, we adopt a pipeline approach of ASR and NLU. For ASR, we fine-tune Whisper for each domain with upsampling. For NLU, we fine-tune BART on all the Track3 data and then on low-resource domain data. We apply masked LM (MLM) -based data augmentation, where some of input tokens and corresponding target labels are replaced using MLM. We also apply a retrieval-based approach, where model input is augmented with similar training samples. As a result, we achieved exact match (EM) accuracy 63.3/75.0 (average: 69.15) for reminder/weather domain, and won the 1st place at the challenge.
    
[^31]: 使用大语言模型的机器翻译新趋势：以ChatGPT为例

    New Trends in Machine Translation using Large Language Models: Case Examples with ChatGPT. (arXiv:2305.01181v1 [cs.CL])

    [http://arxiv.org/abs/2305.01181](http://arxiv.org/abs/2305.01181)

    本文提出了使用大型语言模型的机器翻译中的几个新方向，包括风格化MT、交互式MT和基于翻译记忆的MT，并讨论了隐私问题的解决方案。

    

    近年来，机器翻译（MT）在深度学习的推动下取得了显著进展，特别是在GPT-3和ChatGPT等大型语言模型（LLMs）的出现后。这为使用LLMs的MT带来了新的挑战和机遇。本文提出了一些有趣的使用LLMs的MT方向，包括风格化MT、交互式MT和基于翻译记忆的MT，以及一种使用LLMs的新评估范例。同时，我们还讨论了使用LLMs的MT中的隐私问题，并提出了一种基本的隐私保护方法以减轻此类风险。为了说明我们提出的方法的潜力，我们给出了几个以上提到的新方向的示例，展示了所提出方向的可行性，并突出了使用LLMs的MT未来研究的机会和挑战。

    Machine Translation (MT) has made significant progress in recent years using deep learning, especially after the emergence of large language models (LLMs) such as GPT-3 and ChatGPT. This brings new challenges and opportunities for MT using LLMs. In this paper, we brainstorm some interesting directions for MT using LLMs, including stylized MT, interactive MT, and Translation Memory-based MT, as well as a new evaluation paradigm using LLMs. We also discuss the privacy concerns in MT using LLMs and a basic privacy-preserving method to mitigate such risks. To illustrate the potential of our proposed directions, we present several examples for the new directions mentioned above, demonstrating the feasibility of the proposed directions and highlight the opportunities and challenges for future research in MT using LLMs.
    
[^32]: ATCO2中的经验教训：5000小时的空中交通管制通信对于稳健的自动语音识别和理解。 （arXiv：2305.01155v1 [eess.AS]）

    Lessons Learned in ATCO2: 5000 hours of Air Traffic Control Communications for Robust Automatic Speech Recognition and Understanding. (arXiv:2305.01155v1 [eess.AS])

    [http://arxiv.org/abs/2305.01155](http://arxiv.org/abs/2305.01155)

    本文讨论了ATCO2项目的经验教训，该项目旨在开发一种唯一平台，以实时方式收集和预处理大量的来自空域的ATC数据。该平台可以作为获得“无限源”的数据。为ATC领域的数据驱动型人工智能系统开发提供了可行的解决方案。

    

    空管员和飞行员之间的语音交流对于确保安全和高效的空中交通管制（ATC）至关重要。这项任务需要空管员具有高度的警觉性并且可能是繁琐和容易出错的。最近，已经尝试将人工智能（AI）集成到ATC中，以减少空管员的工作量。然而，为ATC开发数据驱动的AI系统需要大规模的注释数据集，而该领域目前尚缺乏这些数据集。本文探讨了ATCO2项目的经验教训，该项目旨在开发一种唯一平台，以实时方式收集和预处理大量来自空域的ATC数据。通过拥有群众志愿者拥有的VHF接收器从公开可访问的无线电频率通道收集音频和监视数据，然后上传到Opensky网络服务器，这可以被视为“无限源”的数据。此外，本文还回顾了ATCO2以前的工作。

    Voice communication between air traffic controllers (ATCos) and pilots is critical for ensuring safe and efficient air traffic control (ATC). This task requires high levels of awareness from ATCos and can be tedious and error-prone. Recent attempts have been made to integrate artificial intelligence (AI) into ATC in order to reduce the workload of ATCos. However, the development of data-driven AI systems for ATC demands large-scale annotated datasets, which are currently lacking in the field. This paper explores the lessons learned from the ATCO2 project, a project that aimed to develop a unique platform to collect and preprocess large amounts of ATC data from airspace in real time. Audio and surveillance data were collected from publicly accessible radio frequency channels with VHF receivers owned by a community of volunteers and later uploaded to Opensky Network servers, which can be considered an "unlimited source" of data. In addition, this paper reviews previous work from ATCO2 pa
    
[^33]: RadAdapt：通过大型语言模型的轻量化领域自适应实现放射学报告摘要

    RadAdapt: Radiology Report Summarization via Lightweight Domain Adaptation of Large Language Models. (arXiv:2305.01146v1 [cs.CL])

    [http://arxiv.org/abs/2305.01146](http://arxiv.org/abs/2305.01146)

    本研究重点研究了轻量化策略，通过在临床文本上进行预训练和在RRS示例上进行参数高效微调，实现适应大型语言模型进行放射性报告摘要（RRS）任务。并且该方法仅微调模型的0.32％的参数，提高了表现。研究结果强调了领域适应在RRS中的重要性，并为开发更好的放射性报告摘要模型提供了有价值的见解。

    

    本文系统地研究了轻量级策略，通过预训练（自然语言，生物医学文本，临床文本）和提示（零-shot、上下文学习）或参数高效微调（前缀微调，LoRA），来适应大型语言模型（LLMs）进行放射性报告摘要（RRS）任务。结果表明，最大程度地适应任务的方法是，通过在临床文本上预先训练，然后在RRS示例上进行参数高效微调。值得注意的是，这种方法仅微调模型的0.32％的参数，与端对端微调（100％的参数）形成对比。此外，在研究上下文示例和分布外（OOD）训练的影响后，我们进行了放射科医师读者研究和定性分析。我们的研究结果强调了领域适应在RRS中的重要性，并为开发更好的放射性报告摘要模型提供了有价值的见解。

    We systematically investigate lightweight strategies to adapt large language models (LLMs) for the task of radiology report summarization (RRS). Specifically, we focus on domain adaptation via pretraining (on natural language, biomedical text, and clinical text) and via prompting (zero-shot, in-context learning) or parameter-efficient fine-tuning (prefix tuning, LoRA). Our results on the MIMIC-III dataset consistently demonstrate best performance by maximally adapting to the task via pretraining on clinical text and parameter-efficient fine-tuning on RRS examples. Importantly, this method fine-tunes a mere 0.32% of parameters throughout the model, in contrast to end-to-end fine-tuning (100% of parameters). Additionally, we study the effect of in-context examples and out-of-distribution (OOD) training before concluding with a radiologist reader study and qualitative analysis. Our findings highlight the importance of domain adaptation in RRS and provide valuable insights toward developin
    
[^34]: ADVISE：AI加速全球发展证据综述设计

    ADVISE: AI-accelerated Design of Evidence Synthesis for Global Development. (arXiv:2305.01145v1 [cs.CL])

    [http://arxiv.org/abs/2305.01145](http://arxiv.org/abs/2305.01145)

    该论文研究了如何通过将基于BERT的AI代理融入人类团队中，来加速全球发展证据综述产品的设计；同时还研究了不同的主动学习抽样策略对于协作筛选过程的影响。结果表明，将AI代理整合到人类团队中可以将筛选文档的时间缩短60％，使用主动学习还可以将效率进一步提高20％。

    

    在设计基于证据的政策和计划时，决策者必须从大量且迅速增长的文献中提取关键信息。从原始搜索结果中确定相关文献需要耗费大量时间和资源，并且通常是通过人工筛选来完成的。在本研究中，我们开发了一个基于BERT模型的AI代理，将其整合到一个人类团队中，设计一个全球发展证据综述产品。我们探索了人-AI混合团队在加速证据综述过程中的有效性。为了进一步提高团队效率，我们通过主动学习(AL)增强了人-AI混合团队。具体而言，我们研究了不同的抽样策略，包括随机抽样，最小置信度(LC)抽样和最高优先级(HP)抽样，以研究它们对协作筛选过程的影响。结果表明，将基于BERT的AI代理整合到人类团队中可以将筛选文档的时间缩短60％，同时保持高准确度。使用主动学习还可以将效率进一步提高20％。我们的研究展示了AI在加速全球发展证据综述设计方面的潜力。

    When designing evidence-based policies and programs, decision-makers must distill key information from a vast and rapidly growing literature base. Identifying relevant literature from raw search results is time and resource intensive, and is often done by manual screening. In this study, we develop an AI agent based on a bidirectional encoder representations from transformers (BERT) model and incorporate it into a human team designing an evidence synthesis product for global development. We explore the effectiveness of the human-AI hybrid team in accelerating the evidence synthesis process. To further improve team efficiency, we enhance the human-AI hybrid team through active learning (AL). Specifically, we explore different sampling strategies, including random sampling, least confidence (LC) sampling, and highest priority (HP) sampling, to study their influence on the collaborative screening process. Results show that incorporating the BERT-based AI agent into the human team can redu
    
[^35]: Logion：基于机器学习的希腊语语言学

    Logion: Machine Learning for Greek Philology. (arXiv:2305.01099v1 [cs.CL])

    [http://arxiv.org/abs/2305.01099](http://arxiv.org/abs/2305.01099)

    该研究提出了基于机器学习的方法来解决希腊语语言学中的问题，成功利用BERT模型发现和纠正了抄写员在文本传递过程中未被发现的错误，并能在修复预现代手稿材料老化引起的信息缺失方面发挥作用。同时，在领域专家与模型合作时，最佳性能可以通过启示性建议实现。模型的注意力头似乎编码了预现代希腊语的选择性语法特征。

    

    本篇论文提出了一种基于机器学习的方法来解决希腊语语言学中的各种问题。首先，我们在迄今为止用于此目的的最大预现代希腊语数据集上训练了BERT模型，发现并纠正了抄写员在文本传递过程中未被发现的错误。此外，我们展示了该模型在填补由于预现代手稿材料老化导致的缺口方面的能力，并将其性能与领域专家的表现进行了比较。我们发现，当领域专家得到模型的启示性建议时，才能达到最佳性能。考虑到这种人与计算机的合作，我们探讨了模型的可解释性，并发现一些注意力头似乎编码了预现代希腊语的选择性语法特征。

    This paper presents machine-learning methods to address various problems in Greek philology. After training a BERT model on the largest premodern Greek dataset used for this purpose to date, we identify and correct previously undetected errors made by scribes in the process of textual transmission, in what is, to our knowledge, the first successful identification of such errors via machine learning. Additionally, we demonstrate the model's capacity to fill gaps caused by material deterioration of premodern manuscripts and compare the model's performance to that of a domain expert. We find that best performance is achieved when the domain expert is provided with model suggestions for inspiration. With such human-computer collaborations in mind, we explore the model's interpretability and find that certain attention heads appear to encode select grammatical features of premodern Greek.
    
[^36]: SemEval-2023 任务11中的SafeWebUH：学习侮辱性文本的注释者不一致性： 直接训练与聚合的比较。

    SafeWebUH at SemEval-2023 Task 11: Learning Annotator Disagreement in Derogatory Text: Comparison of Direct Training vs Aggregation. (arXiv:2305.01050v1 [cs.CL])

    [http://arxiv.org/abs/2305.01050](http://arxiv.org/abs/2305.01050)

    本文研究使用BERT模型来标注侮辱性文本中的注释者不一致性，并比较直接训练和聚合两种方法，结果发现聚合方法比直接训练有更好的效果。

    

    主观性和不同意见是关键的社会现象，考虑到这一点在注释和检测侮辱性文本内容的过程中至关重要。本文使用SemEval-2023任务11提供的四个数据集，对BERT模型进行微调，以捕捉注释中的不一致性。我们发现个体注释者建模和聚合将交叉熵得分平均降低了0.21，而与直接训练软标签相比。我们的研究进一步证明了注释者元数据对平均交叉熵分数的0.029降低有所贡献。

    Subjectivity and difference of opinion are key social phenomena, and it is crucial to take these into account in the annotation and detection process of derogatory textual content. In this paper, we use four datasets provided by SemEval-2023 Task 11 and fine-tune a BERT model to capture the disagreement in the annotation. We find individual annotator modeling and aggregation lowers the Cross-Entropy score by an average of 0.21, compared to the direct training on the soft labels. Our findings further demonstrate that annotator metadata contributes to the average 0.029 reduction in the Cross-Entropy score.
    
[^37]: 零样本学习在公司分类中的应用

    Company classification using zero-shot learning. (arXiv:2305.01028v1 [cs.CL])

    [http://arxiv.org/abs/2305.01028](http://arxiv.org/abs/2305.01028)

    本文提出了一种利用自然语言处理和零样本学习的方法来进行公司分类的方法。该方法可以简化公司分类过程，从而减少传统方法如全球产业分类标准（GICS）所需的时间和资源。

    

    近年来，自然语言处理在许多商业应用中变得越来越重要，包括情感分析、文本分类和命名实体识别。本文提出了一种利用自然语言处理和零样本学习的方法来进行公司分类的方法。我们的方法利用预训练的Transformer模型从公司描述中提取特征，然后应用零样本学习将公司分类到相关类别，无需为每个类别提供特定的训练数据。我们在公开可用的公司文本描述数据集上评估我们的方法，并证明它可以简化公司分类过程，从而减少传统方法如全球产业分类标准（GICS）所需的时间和资源。结果表明，该方法具有自动化公司分类的潜力，是未来研究的一个有前途的方向。

    In recent years, natural language processing (NLP) has become increasingly important in a variety of business applications, including sentiment analysis, text classification, and named entity recognition. In this paper, we propose an approach for company classification using NLP and zero-shot learning. Our method utilizes pre-trained transformer models to extract features from company descriptions, and then applies zero-shot learning to classify companies into relevant categories without the need for specific training data for each category. We evaluate our approach on publicly available datasets of textual descriptions of companies, and demonstrate that it can streamline the process of company classification, thereby reducing the time and resources required in traditional approaches such as the Global Industry Classification Standard (GICS). The results show that this method has potential for automation of company classification, making it a promising avenue for future research in thi
    
[^38]: 将统计语言模型作为语用推理的评估器

    Evaluating statistical language models as pragmatic reasoners. (arXiv:2305.01020v1 [cs.CL])

    [http://arxiv.org/abs/2305.01020](http://arxiv.org/abs/2305.01020)

    本文评估了大型语言模型推断语用话语的能力。作者通过表述有关等级形容词“强”的门槛估计来测试 LLM 的性能，并发现 LLM 可以生成具有上下文依赖性、类人的分布，但在组合方面存在困难。

    

    沟通语言和预期意义之间的关系通常是概率性的，而且对语境非常敏感。许多策略试图估计这种映射，通常利用基于贝叶斯递归模型的通信方式。与此同时，大型语言模型 (LLMs) 被越来越多地应用于语义分析应用程序，任务是从自然语言推断逻辑表示。虽然现有的 LLM 探索主要局限于字面上的语言使用，但在这项工作中，我们评估了 LLM 推断语用话语的能力。具体而言，我们探讨了基于“强”这个等级形容词的门槛估计的情况，从具有强度先验条件的语境出发，然后扩展到限定、否定、极性反转和类比组合。我们发现，LLMs 可以推导出与语境相关的类人分布，涉及到几个复杂语用话语的解释，但在组合上还有困难。

    The relationship between communicated language and intended meaning is often probabilistic and sensitive to context. Numerous strategies attempt to estimate such a mapping, often leveraging recursive Bayesian models of communication. In parallel, large language models (LLMs) have been increasingly applied to semantic parsing applications, tasked with inferring logical representations from natural language. While existing LLM explorations have been largely restricted to literal language use, in this work, we evaluate the capacity of LLMs to infer the meanings of pragmatic utterances. Specifically, we explore the case of threshold estimation on the gradable adjective ``strong'', contextually conditioned on a strength prior, then extended to composition with qualification, negation, polarity inversion, and class comparison. We find that LLMs can derive context-grounded, human-like distributions over the interpretations of several complex pragmatic utterances, yet struggle composing with n
    
[^39]: 基于软域转移的特征增强欺骗检测

    Deception Detection with Feature-Augmentation by soft Domain Transfer. (arXiv:2305.01011v1 [cs.CL])

    [http://arxiv.org/abs/2305.01011](http://arxiv.org/abs/2305.01011)

    本文提出了一种基于中间层表示的特征增强方法，通过软域转移进行领域间的关联，提高欺骗检测的准确率，分析结果显示推文是检测假新闻和钓鱼电子邮件最有帮助的信息提供者，新闻在推特谣言检测中最有帮助。

    

    在信息爆炸的这个时代，欺骗者利用不同的信息领域或媒介来利用用户，比如新闻、电子邮件和推文等。尽管已经进行了大量的研究来检测这些领域中的欺骗，但新事件中信息的短缺需要这些领域相互关联来对抗欺骗。为了形成这种关联，我们提出了一种通过利用神经模型的中间层表示进行特征增强的方法。我们的方法比自身领域基线模型提高了多达6.60%的准确率。我们发现推文是检测假新闻和钓鱼电子邮件最有帮助的信息提供者，而新闻在推特谣言检测中最有帮助。我们的分析提供了对于域知识转移的有用洞见，可以帮助建立比现有文献更强大的欺骗检测系统。

    In this era of information explosion, deceivers use different domains or mediums of information to exploit the users, such as News, Emails, and Tweets. Although numerous research has been done to detect deception in all these domains, information shortage in a new event necessitates these domains to associate with each other to battle deception. To form this association, we propose a feature augmentation method by harnessing the intermediate layer representation of neural models. Our approaches provide an improvement over the self-domain baseline models by up to 6.60%. We find Tweets to be the most helpful information provider for Fake News and Phishing Email detection, whereas News helps most in Tweet Rumor detection. Our analysis provides a useful insight for domain knowledge transfer which can help build a stronger deception detection system than the existing literature.
    
[^40]: TMR:使用对比3D人体运动合成的文本到运动检索

    TMR: Text-to-Motion Retrieval Using Contrastive 3D Human Motion Synthesis. (arXiv:2305.00976v1 [cs.CV])

    [http://arxiv.org/abs/2305.00976](http://arxiv.org/abs/2305.00976)

    本文介绍了一种名为TMR的方法，用于将文本转换为3D人体运动。它在先前的工作中取得了明显的优势，并通过引入对比性损失的方法来更好地建立跨模态潜在空间结构。结果表明，保持运动生成损失和对比性训练至关重要。

    

    本文提出了一种名为TMR的简单而有效的方法，用于将文本转换为3D人体运动。与之前的工作仅将检索视为代理评估指标不同，我们将其作为一个独立的任务来解决。我们的方法扩展了最先进的文本到动作合成模型TEMOS，并结合对比损失来更好地构造跨模态的潜在空间。我们表明保持运动生成损失和对比性训练至关重要，以获得良好的性能。我们引入了一个基准来进行评估，并通过报告几个协议的结果进行了深入分析。我们在KIT-ML和HumanML3D数据集上进行的广泛实验表明，TMR比先前的工作表现出明显的优势，例如将中位数排名从54降至19。最后，我们展示了我们方法在时刻检索方面的潜力。我们的代码和模型是公开可用的。

    In this paper, we present TMR, a simple yet effective approach for text to 3D human motion retrieval. While previous work has only treated retrieval as a proxy evaluation metric, we tackle it as a standalone task. Our method extends the state-of-the-art text-to-motion synthesis model TEMOS, and incorporates a contrastive loss to better structure the cross-modal latent space. We show that maintaining the motion generation loss, along with the contrastive training, is crucial to obtain good performance. We introduce a benchmark for evaluation and provide an in-depth analysis by reporting results on several protocols. Our extensive experiments on the KIT-ML and HumanML3D datasets show that TMR outperforms the prior work by a significant margin, for example reducing the median rank from 54 to 19. Finally, we showcase the potential of our approach on moment retrieval. Our code and models are publicly available.
    
[^41]: 分解增强推理的自我评估引导解码

    Decomposition Enhances Reasoning via Self-Evaluation Guided Decoding. (arXiv:2305.00633v1 [cs.CL])

    [http://arxiv.org/abs/2305.00633](http://arxiv.org/abs/2305.00633)

    本论文提出了一种通过自我评估引导解码提高推理的方法，使用经过校准的自动标准探索推理搜索空间，使搜索能够产生更高质量的最终预测结果；使用自我评估引导的随机束搜索在产生推理链的质量和多样性之间平衡权衡，适应多数投票，并且可以准确判断逻辑错误，提高一致性和鲁棒性。

    

    我们提出了一种有效的提示方法，通过随机束搜索结合自我评估引导。我们的方法使用经过校准的自动标准探索推理搜索空间。这使得有效搜索能够产生更高质量的最终预测结果。使用自我评估引导的随机束搜索，我们在产生推理链的质量和多样性之间平衡权衡，从而能够适应多数投票，并在GSM8K、AQUA和StrategyQA基准测试中以少量示例准确性分别超越对应的Codex-backboned基线$6.34\%$、$9.56\%$和$5.46\%$。对我们的分解式推理分析发现，它可以指出逻辑错误并导致更高的一致性和鲁棒性。

    We propose an effective prompting approach that integrates self-evaluation guidance through stochastic beam search. Our approach explores the reasoning search space using a well-calibrated automatic criterion. This enables an efficient search to produce higher-quality final predictions. With the self-evaluation guided stochastic beam search, we also balance the quality--diversity trade-off in the generation of reasoning chains. This allows our approach to adapt well with majority voting and surpass the corresponding Codex-backboned baselines by $6.34\%$, $9.56\%$, and $5.46\%$ on the GSM8K, AQUA, and StrategyQA benchmarks, respectively, in few-shot accuracy. Analysis of our decompositional reasoning finds it pinpoints logic failures and leads to higher consistency and robustness.
    
[^42]: 对Kauhanen、Einhaus和Walkden（2023年）的回应：仍然没有证据证明非母语用户比例对语言复杂度有影响（arXiv:2305.00217v1 [cs.CL]）

    Still no evidence for an effect of the proportion of non-native speakers on language complexity -- A response to Kauhanen, Einhaus & Walkden (2023). (arXiv:2305.00217v1 [cs.CL])

    [http://arxiv.org/abs/2305.00217](http://arxiv.org/abs/2305.00217)

    本研究为对Kauhanen、Einhaus和Walkden（2023）的回应，仍然没有证据表明大量的L2用户影响语言复杂性。

    

    近期在《语言进化杂志》发表的一篇论文中，Kauhanen、Einhaus和Walkden（https://doi.org/10.1093/jole/lzad005，KEW）挑战了我在一篇论文中（Koplenig，Royal Society Open Science，6，181274（2019），https://doi.org/10.1098/rsos.181274）所呈现的结果。在该论文中，我试图通过一系列的统计分析来表明大量L2（第二语言）用户似乎不会影响语言的（语法或统计）复杂性。为此，我专注于Ethnologue评估语言地位的方式：如果一种语言除了被L1（第一语言）使用者之外，还应该有大量的L2使用者，那么该语言就被描述为传播性的。KEW批评了将传播性作为语言是否拥有大量L2使用者（二元）指标的使用，以及在直接估计L2比例的情况下，将L2用户比例归为非传播性语言的想法。

    In a recent paper published in the Journal of Language Evolution, Kauhanen, Einhaus & Walkden (https://doi.org/10.1093/jole/lzad005, KEW) challenge the results presented in one of my papers (Koplenig, Royal Society Open Science, 6, 181274 (2019), https://doi.org/10.1098/rsos.181274), in which I tried to show through a series of statistical analyses that large numbers of L2 (second language) speakers do not seem to affect the (grammatical or statistical) complexity of a language. To this end, I focus on the way in which the Ethnologue assesses language status: a language is characterised as vehicular if, in addition to being used by L1 (first language) speakers, it should also have a significant number of L2 users. KEW criticise both the use of vehicularity as a (binary) indicator of whether a language has a significant number of L2 users and the idea of imputing a zero proportion of L2 speakers to non-vehicular languages whenever a direct estimate of that proportion is unavailable. Whi
    
[^43]: 朝自主系统迈进：使用大语言模型代理增强的灵活模块化生产系统

    Towards autonomous system: flexible modular production system enhanced with large language model agents. (arXiv:2304.14721v1 [cs.RO])

    [http://arxiv.org/abs/2304.14721](http://arxiv.org/abs/2304.14721)

    本论文介绍了一种将大语言模型、数字孪生和工业自动化系统相结合的框架，实现生产过程的智能化规划和控制。通过LLM代理的协调控制，实现了灵活生产的自主规划和控制，能够处理未预定义的任务并规划生产过程。

    

    本文提出了一种新颖的框架，将大型语言模型（LLM），数字孪生和工业自动化系统结合起来，实现生产过程的智能规划和控制。我们的方法涉及开发包含生产描述信息的数字孪生系统，并将自动化系统改造为提供统一接口的细粒度功能或模块，以供自动化组件或模块执行。随后，设计LLM代理来解释数字孪生中的描述性信息，并通过RESTful接口控制物理系统。这些LLM代理作为自动化系统内的智能代理，实现了灵活生产的自主规划和控制。给定一个任务指令作为输入，LLM代理协调一系列原子功能和技能来完成任务。我们展示了我们实现的原型如何处理未预定义的任务，并计划生产过程。

    In this paper, we present a novel framework that combines large language models (LLMs), digital twins and industrial automation system to enable intelligent planning and control of production processes. Our approach involves developing a digital twin system that contains descriptive information about the production and retrofitting the automation system to offer unified interfaces of fine-granular functionalities or skills executable by automation components or modules. Subsequently, LLM-Agents are designed to interpret descriptive information in the digital twins and control the physical system through RESTful interfaces. These LLM-Agents serve as intelligent agents within an automation system, enabling autonomous planning and control of flexible production. Given a task instruction as input, the LLM-agents orchestrate a sequence of atomic functionalities and skills to accomplish the task. We demonstrate how our implemented prototype can handle un-predefined tasks, plan a production p
    
[^44]: Sebis在SemEval-2023任务7中：临床试验报告中自然语言推理和证据检索的联合系统

    Sebis at SemEval-2023 Task 7: A Joint System for Natural Language Inference and Evidence Retrieval from Clinical Trial Reports. (arXiv:2304.13180v1 [cs.CL])

    [http://arxiv.org/abs/2304.13180](http://arxiv.org/abs/2304.13180)

    本文描述了两个NLP系统：一个为自然语言推理，一个为临床试验数据证据检索。它们分别采用了流水线模型和联合模型，并在最终的集成系统中融合输出。

    

    随着每天产生的临床试验报告数量增加，跟进告知基于证据的医疗建议的新发现变得越来越难。为了帮助自动化这一过程并协助医疗专家，正在开发NLP解决方案。这激发了SemEval-2023任务7，该任务的目标是开发一个NLP系统，以处理从临床试验数据中提取证据和进行自然语言推理的两个任务。本文介绍我们开发的两个系统。第一个是流水线系统，单独建模了这两个任务，而第二个是联合系统，采用共享表示和多任务学习方法同时学习这两个任务。最终系统将它们的输出合并为一个集成系统。我们规范化模型，介绍其特点和挑战，并对实现的结果进行分析。

    With the increasing number of clinical trial reports generated every day, it is becoming hard to keep up with novel discoveries that inform evidence-based healthcare recommendations. To help automate this process and assist medical experts, NLP solutions are being developed. This motivated the SemEval-2023 Task 7, where the goal was to develop an NLP system for two tasks: evidence retrieval and natural language inference from clinical trial data. In this paper, we describe our two developed systems. The first one is a pipeline system that models the two tasks separately, while the second one is a joint system that learns the two tasks simultaneously with a shared representation and a multi-task learning approach. The final system combines their outputs in an ensemble system. We formalize the models, present their characteristics and challenges, and provide an analysis of achieved results.
    
[^45]: 大语言模型实现多语机器翻译：实证结果和分析

    Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis. (arXiv:2304.04675v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.04675](http://arxiv.org/abs/2304.04675)

    本文系统地研究了大语言模型在多语机器翻译中的优势和挑战，证明其表现出卓越的潜力。本研究发现LLMs在给定上下文示例时可以意外地忽略提示语义，并且跨语言示例可以为低资源翻译提供更好的任务指导。但实证结果表明，即使是最好的模型ChatGPT仍然落后于监督基线NLLB。

    

    大语言模型(LLMs)在处理多语机器翻译(MMT)方面表现出了卓越的潜力。本文通过回答两个问题系统地研究了LLMs在MMT中的优势和挑战：1) LLMs在翻译大量语言方面表现如何？2) 哪些因素会影响LLMs在翻译中的表现？我们评估了包括XGLM、OPT、BLOOMZ和ChatGPT在内的几个受欢迎的LLMs在102种语言上的表现。我们的实证结果显示，即使是最好的模型ChatGPT在83.33%的翻译方向上也落后于监督基线NLLB。通过进一步的分析，我们发现当用于MMT时，LLMs表现出新的工作模式。首先，在给定上下文示例时，提示语义可能会被意外地忽略，即使提示不合理，LLMs仍然表现出强大的性能。其次，跨语言示例可以为低资源翻译提供比相同语言对中的示例更好的任务指导。第三，当翻译低资源语言时，LLMs往往表现得更好。总的来说，我们的研究为LLMs在MMT中的潜力和局限性提供了新的见解，为未来的研究提供了有用的启示。

    Large language models (LLMs) have demonstrated remarkable potential in handling multilingual machine translation (MMT). In this paper, we systematically investigate the advantages and challenges of LLMs for MMT by answering two questions: 1) How well do LLMs perform in translating a massive number of languages? 2) Which factors affect LLMs' performance in translation? We evaluate popular LLMs, including XGLM, OPT, BLOOMZ, and ChatGPT, on 102 languages. Our empirical results show that even the best model ChatGPT still lags behind the supervised baseline NLLB in 83.33% of translation directions. Through further analysis, we discover that LLMs exhibit new working patterns when used for MMT. First, prompt semantics can surprisingly be ignored when given in-context exemplars, where LLMs still show strong performance even with unreasonable prompts. Second, cross-lingual exemplars can provide better task instruction for low-resource translation than exemplars in the same language pairs. Third
    
[^46]: 奖励是否合理？在 MACHIAVELLI 基准测试中衡量奖励与道德行为之间的权衡

    Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark. (arXiv:2304.03279v1 [cs.LG])

    [http://arxiv.org/abs/2304.03279](http://arxiv.org/abs/2304.03279)

    本文介绍了 MACHIAVELLI 基准测试，用于衡量人工智能代理是否表现出马基雅维利行为，发现了最大化奖励和行为的道德性之间存在权衡，并探索了基于语言模型的方法来减轻这种权衡。

    

    传统上，人工智能代理被训练成最大化奖励，这可能会激励追求权力和欺骗行为，类似于语言模型中的下一个标记预测可能会激励有害行为。那么代理是否自然而然地学会了马基雅维利行为？我们如何在 GPT-4 等通用模型中衡量这些行为呢？为回答这些问题，我们引入了 MACHIAVELLI 基准测试，该测试涵盖了超过一百万个多样化的情景，重点关注社会决策制定，用于衡量人工代理是否表现出马基雅维利行为。我们数学化了数十种有害行为，并使用我们的注释来评估代理倾向于追求权力，造成功能不良和违反伦理的倾向。我们观察到最大化奖励和行为的道德性之间存在一些紧张关系。为了改善这种权衡，我们研究了基于语言模型的方法，以使代理趋向于采取更少的有害行为。我们的结果显示，MACHIAVELLI 是评估人工代理马基雅维利行为水平的有用基准测试。

    Artificial agents have traditionally been trained to maximize reward, which may incentivize power-seeking and deception, analogous to how next-token prediction in language models (LMs) may incentivize toxicity. So do agents naturally learn to be Machiavellian? And how do we measure these behaviors in general-purpose models such as GPT-4? Towards answering these questions, we introduce MACHIAVELLI, a benchmark of 134 Choose-Your-Own-Adventure games containing over half a million rich, diverse scenarios that center on social decision-making. Scenario labeling is automated with LMs, which are more performant than human annotators. We mathematize dozens of harmful behaviors and use our annotations to evaluate agents' tendencies to be power-seeking, cause disutility, and commit ethical violations. We observe some tension between maximizing reward and behaving ethically. To improve this trade-off, we investigate LM-based methods to steer agents' towards less harmful behaviors. Our results sh
    
[^47]: 仅仅提示足够了吗？不是的。指导学习的全面和更广阔视角（arXiv：2303.10475v1 [cs.CL]）

    Is Prompt All You Need? No. A Comprehensive and Broader View of Instruction Learning. (arXiv:2303.10475v1 [cs.CL])

    [http://arxiv.org/abs/2303.10475](http://arxiv.org/abs/2303.10475)

    传统的自然语言处理机器学习需要大规模的任务特定示例，但这不适用于任务可能过于复杂或成本过高以进行注释的场景。因此，社区对于自然语言处理中新的监督寻求范式--从任务指令学习--越来越感兴趣。

    

    任务语义可以通过一组输入输出示例或一条文本指令来表达。传统的自然语言处理（NLP）机器学习方法主要依赖于大规模的任务特定示例的可用性。这引起了两个问题：首先，收集任务特定标记示例不适用于任务可能过于复杂或成本过高以进行注释的场景，或者系统需要立即处理新任务。其次，这不是用户友好的，因为最终用户可能更愿意在使用系统之前提供任务描述而不是一组示例。因此，社区对于自然语言处理中新的监督寻求范式--从任务指令学习--越来越感兴趣。尽管取得了令人印象深刻的进展，但社区仍然面临着一些共同的问题。本次调查旨在总结指导学习的当前研究，特别是回答以下问题：

    Task semantics can be expressed by a set of input-to-output examples or a piece of textual instruction. Conventional machine learning approaches for natural language processing (NLP) mainly rely on the availability of large-scale sets of task-specific examples. Two issues arise: first, collecting task-specific labeled examples does not apply to scenarios where tasks may be too complicated or costly to annotate, or the system is required to handle a new task immediately; second, this is not user-friendly since end-users are probably more willing to provide task description rather than a set of examples before using the system. Therefore, the community is paying increasing interest in a new supervision-seeking paradigm for NLP: learning from task instructions. Despite its impressive progress, there are some common issues that the community struggles with. This survey paper tries to summarize the current research on instruction learning, particularly, by answering the following questions:
    
[^48]: AUTODIAL: 高效异步任务导向的对话模型

    AUTODIAL: Efficient Asynchronous Task-Oriented Dialogue Model. (arXiv:2303.06245v1 [cs.CL])

    [http://arxiv.org/abs/2303.06245](http://arxiv.org/abs/2303.06245)

    AUTODIAL是一种多任务对话模型，通过使用并行解码器来执行对话任务，从而显著减少内存占用并实现更快的推理时间。与现有的生成方法相比，AUTODIAL在三个对话任务上提供了3-6倍的速度提升，同时具有11倍的参数减少。这表明将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。

    AUTODIAL is a multi-task dialogue model that significantly reduces memory footprint and achieves faster inference times by using parallel decoders to perform dialogue tasks. Compared to existing generative approach, AUTODIAL provides 3-6x speedups during inference while having 11x fewer parameters on three dialogue tasks. This suggests that extending current dialogue models to have parallel decoders can be a viable alternative for deploying them in resource-constrained environments.

    随着大型对话模型在实践中变得普遍，训练、推理和更大的内存占用的高计算要求问题仍然存在。在这项工作中，我们提出了AUTODIAL，一种多任务对话模型，解决了部署对话模型的挑战。AUTODIAL利用并行解码器执行诸如对话行为预测、领域预测、意图预测和对话状态跟踪等任务。使用分类解码器而不是生成解码器使AUTODIAL能够显著减少内存占用，并在推理时间上实现比现有生成方法（即SimpleTOD）更快的速度。我们证明，将当前的对话模型扩展为具有并行解码器可以成为在资源受限环境中部署它们的可行替代方案。

    As large dialogue models become commonplace in practice, the problems surrounding high compute requirements for training, inference and larger memory footprint still persists. In this work, we present AUTODIAL, a multi-task dialogue model that addresses the challenges of deploying dialogue model. AUTODIAL utilizes parallel decoders to perform tasks such as dialogue act prediction, domain prediction, intent prediction, and dialogue state tracking. Using classification decoders over generative decoders allows AUTODIAL to significantly reduce memory footprint and achieve faster inference times compared to existing generative approach namely SimpleTOD. We demonstrate that AUTODIAL provides 3-6x speedups during inference while having 11x fewer parameters on three dialogue tasks compared to SimpleTOD. Our results show that extending current dialogue models to have parallel decoders can be a viable alternative for deploying them in resource-constrained environments.
    
[^49]: 语言模型分析本体子类推断

    Language Model Analysis for Ontology Subsumption Inference. (arXiv:2302.06761v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.06761](http://arxiv.org/abs/2302.06761)

    本文研究了语言模型对本体子类推断的理解能力，提出了一套涉及原子概念和复合概念的推理任务，并证明语言模型对子类推断背景知识的记忆相对较少，但在给定少量样本的情况下可显著提高准确率。

    

    最近，研究人员开始探究预训练的语言模型是否能够作为知识库的替代。然而，现有的研究都关注于简单的三元组关系型知识库，忽略了更为复杂、逻辑为基础、概念化的 OWL 本体等知识库。为了研究语言模型对于本体的了解，我们提出 OntoLAMA，它包含基于推理的一系列测试任务和数据集，从涉及原子概念和复合概念的子类推断公理出发。我们对不同领域和规模的本体进行了大量实验，结果表明，相比传统的自然语言推理，语言模型对子类推断的背景知识记忆相对较少，但是在给定少量样本的情况下，可以显著提高子类推断的准确率。我们将公开源码和数据集。

    Investigating whether pre-trained language models (LMs) can function as knowledge bases (KBs) has raised wide research interests recently. However, existing works focus on simple, triple-based, relational KBs, but omit more sophisticated, logic-based, conceptualised KBs such as OWL ontologies. To investigate an LM's knowledge of ontologies, we propose OntoLAMA, a set of inference-based probing tasks and datasets from ontology subsumption axioms involving both atomic and complex concepts. We conduct extensive experiments on ontologies of different domains and scales, and our results demonstrate that LMs encode relatively less background knowledge of Subsumption Inference (SI) than traditional Natural Language Inference (NLI) but can improve on SI significantly when a small number of samples are given. We will open-source our code and datasets.
    
[^50]: CoRRPUS: 利用Codex的结构化表示增强神经符号故事理解

    CoRRPUS: Codex-Leveraged Structured Representations for Neurosymbolic Story Understanding. (arXiv:2212.10754v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10754](http://arxiv.org/abs/2212.10754)

    本研究利用Code-LLMs引导符号表示以增强神经符号故事理解，通过CoRRPUS系统和抽象提示程序，在最小的手动工程条件下，击败了当前最先进的结构LLM技术。

    

    随着所有自然语言生成/理解任务的神经符号技术的飞速发展，故事生成和理解也得到了极大的改进。本研究利用最先进的Code-LLMs，如Codex，引导符号方法的使用，以跟踪故事状态并帮助故事理解。我们展示了我们的CoRRPUS系统和抽象提示程序如何在最小的手动工程条件下，击败了当前最先进的结构LLM技术，完成了预先存在的故事理解任务（bAbI task 2 和 Re^3）。我们希望本研究能够凸显符号表示和专业提示对LLMs的重要性，因为这些模型需要一些手工优化。

    Story generation and understanding -- as with all NLG/NLU tasks -- has seen a surge in neurosymbolic work. Researchers have recognized that, while large language models (LLMs) have tremendous utility, they can be augmented with symbolic means to be even better and to make up for any flaws that the neural networks might have. However, symbolic methods are extremely costly in terms of the amount of time and expertise needed to create them. In this work, we capitalize on state-of-the-art Code-LLMs, such as Codex, to bootstrap the use of symbolic methods for tracking the state of stories and aiding in story understanding. We show that our CoRRPUS system and abstracted prompting procedures can beat current state-of-the-art structured LLM techniques on pre-existing story understanding tasks (bAbI task 2 and Re^3) with minimal hand engineering. We hope that this work can help highlight the importance of symbolic representations and specialized prompting for LLMs as these models require some g
    
[^51]: 跨语言转移的令人沮丧的简易标签投影

    Frustratingly Easy Label Projection for Cross-lingual Transfer. (arXiv:2211.15613v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.15613](http://arxiv.org/abs/2211.15613)

    本文通过一项广泛的实证研究，对57种语言和三个任务下的跨语言转移进行了研究，并发现优化后的标记-翻译法比传统注释投影方法更有效。

    

    将训练数据翻译成多种语言已成为提高跨语言转移的实际解决方案。对于涉及跨度级别注释（例如信息提取或问题回答）的任务，需要进行额外的标签投影步骤，将已注释的跨度映射到翻译后的文本中。然而，据我们所知，迄今为止尚未对这种方法与基于单词对齐的传统注释投影进行实证分析。在本文中，我们展示了一项对57种语言和三个任务（QA，NER和事件提取）进行广泛的实证研究，以评估两种方法的有效性和局限性，并填补文献中的重要空白。实验结果表明，我们优化后的标记-翻译法比传统注释投影方法更有效。

    Translating training data into many languages has emerged as a practical solution for improving cross-lingual transfer. For tasks that involve span-level annotations, such as information extraction or question answering, an additional label projection step is required to map annotated spans onto the translated texts. Recently, a few efforts have utilized a simple mark-then-translate method to jointly perform translation and projection by inserting special markers around the labeled spans in the original sentence. However, as far as we are aware, no empirical analysis has been conducted on how this approach compares to traditional annotation projection based on word alignment. In this paper, we present an extensive empirical study across 57 languages and three tasks (QA, NER, and Event Extraction) to evaluate the effectiveness and limitations of both methods, filling an important gap in the literature. Experimental results show that our optimized version of mark-then-translate, which we
    
[^52]: 神经语言模型中的逐字短期记忆特征研究

    Characterizing Verbatim Short-Term Memory in Neural Language Models. (arXiv:2210.13569v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.13569](http://arxiv.org/abs/2210.13569)

    本研究探讨了语言模型的逐字短期记忆特征，发现Transformer模型可以准确检索先前出现过的单词身份和顺序，而LSTM模型的检索能力受限于列表初始标记和短的干扰文本。

    

    当语言模型被训练用于预测自然语言序列时，它在每个时刻的预测依赖于先前上下文的表征。语言模型能够检索到哪些关于先前上下文的信息？本研究测试了语言模型能否检索到先前在文本中出现过的确切单词。我们以英文文本为范例，其中一个名词列表出现了两次，利用Transformer和LSTM模型处理。我们将检索定义为从第一个列表到第二个列表的惊异度降低。我们发现，Transformer可以从第一个列表中检索到名词的身份和顺序。此外，当Transformer在更大的语料库中和更深的模型中进行训练时，它们的检索能力显著增强。最后，Transformer索引先前的标记的能力取决于学习到的注意模式。相反，LSTM的检索能力较低，仅限于列表初始标记和短的干扰文本。

    When a language model is trained to predict natural language sequences, its prediction at each moment depends on a representation of prior context. What kind of information about the prior context can language models retrieve? We tested whether language models could retrieve the exact words that occurred previously in a text. In our paradigm, language models (transformers and an LSTM) processed English text in which a list of nouns occurred twice. We operationalized retrieval as the reduction in surprisal from the first to the second list. We found that the transformers retrieved both the identity and ordering of nouns from the first list. Further, the transformers' retrieval was markedly enhanced when they were trained on a larger corpus and with greater model depth. Lastly, their ability to index prior tokens was dependent on learned attention patterns. In contrast, the LSTM exhibited less precise retrieval, which was limited to list-initial tokens and to short intervening texts. The
    
[^53]: 使用可能性分割进行长尾概括基准测试

    Benchmarking Long-tail Generalization with Likelihood Splits. (arXiv:2210.06799v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06799](http://arxiv.org/abs/2210.06799)

    为了可靠地处理自然语言，NLP系统需要推广长尾稀有语句。这篇论文提出了一种使用概率分割来创建有意义测试数据集的方法，引入了更多的挑战性并可能提高在关键任务上的错误率。

    

    为了可靠地处理自然语言，NLP系统必须推广到稀有语句的长尾部分。我们提出了一种方法来创建需要推广到分布尾部的具有挑战性的基准测试，即通过重新划分现有数据集来实现。我们创建了“可能性分割”，即将由预训练语言模型（LM）分配较低可能性的实例放置在测试集中，而更可能的实例则在训练集中。这种简单的方法可以自定义，以构建适合各种任务的有意义的训练-测试分割。相对于随机分割，可能性分割表现出比随机分割更多的挑战：在Spider上进行的语义解析的最先进模型的相对误差率增加了59％、在SNLI上进行的自然语言推理的相对误差率增加了93％、在BoolQ上进行的是/否问题回答的相对误差率增加了33％。此外，可能性分割创建比对抗过滤更公平的基准测试；当用于创建分割的LM也是检测器时，几乎不会影响原始模型的性能。

    In order to reliably process natural language, NLP systems must generalize to the long tail of rare utterances. We propose a method to create challenging benchmarks that require generalizing to the tail of the distribution by re-splitting existing datasets. We create 'Likelihood Splits' where examples that are assigned lower likelihood by a pre-trained language model (LM) are placed in the test set, and more likely examples are in the training set. This simple approach can be customized to construct meaningful train-test splits for a wide range of tasks. Likelihood Splits surface more challenges than random splits: relative error rates of state-of-the-art models increase by 59% for semantic parsing on Spider, 93% for natural language inference on SNLI, and 33% for yes/no question answering on BoolQ, on our splits compared with the corresponding random splits. Moreover, Likelihood Splits create fairer benchmarks than adversarial filtering; when the LM used to create the splits is also e
    
[^54]: 带环境感知的语言模型生成可执行的动作计划

    Generating Executable Action Plans with Environmentally-Aware Language Models. (arXiv:2210.04964v2 [cs.RO] UPDATED)

    [http://arxiv.org/abs/2210.04964](http://arxiv.org/abs/2210.04964)

    本文提出了一种带环境感知的语言模型生成可执行的动作计划的方法，通过集成环境对象和对象关系作为额外输入，设计了新颖的评分函数，生成的可执行计划相对于传统的LLM方法有更高成功率。

    

    近期大规模语言模型在从高层次文本查询中生成机器人代理的动作计划方面表现出了很好的潜力。然而，这些模型通常不考虑机器人的环境，导致生成的计划可能无法执行，因为计划中的动作模糊不清或受到环境限制。本文提出了一种生成带环境感知的可执行动作计划的方法。我们的方法是将环境对象和对象关系作为额外输入集成到LLM动作计划生成中，以提供系统对周围环境的感知，从而生成与场景中存在的物体相对应的计划动作。此外，我们设计了一种新颖的评分函数，帮助系统消除物体实例之间的歧义并考虑它们的状态。我们在一个机器人操作任务上评估了我们的方法，并展示了我们的模型相对于传统的LLM方法生成了更高成功率的可执行计划。

    Large Language Models (LLMs) trained using massive text datasets have recently shown promise in generating action plans for robotic agents from high level text queries. However, these models typically do not consider the robot's environment, resulting in generated plans that may not actually be executable, due to ambiguities in the planned actions or environmental constraints. In this paper, we propose an approach to generate environmentally-aware action plans that agents are better able to execute. Our approach involves integrating environmental objects and object relations as additional inputs into LLM action plan generation to provide the system with an awareness of its surroundings, resulting in plans where each generated action is mapped to objects present in the scene. We also design a novel scoring function that, along with generating the action steps and associating them with objects, helps the system disambiguate among object instances and take into account their states. We ev
    
[^55]: 歌词中的性别歧视和性别偏见的大规模分析

    Large scale analysis of gender bias and sexism in song lyrics. (arXiv:2208.02052v3 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2208.02052](http://arxiv.org/abs/2208.02052)

    本文对377808首英文歌曲歌词进行大规模的自然语言处理分析，揭示了及时的性别歧视的增加以及不同性别表演者的语言偏见。

    

    我们使用自然语言处理技术分析了“Two Million Song Database”语料库中377808首英文歌曲歌词，着重分析了五十年（1960-2010）间性别歧视的表达，以及对性别偏差的评测。通过使用一个性别歧视分类器，我们在较大的规模上识别了性别歧视歌词，远超前人用手动标注流行歌曲的小样本研究。此外，通过在歌曲歌词上学习的词嵌入来衡量关联，我们揭示了性别偏见。我们发现，尤其是由男性艺术家演唱的流行歌曲中的性别歧视内容在时间上呈逐渐增多的趋势。根据表演者的性别不同，歌曲还显示出不同的语言偏见，男性独唱艺术家的歌曲中包含更多和更强的偏见。这是第一次进行这种大规模的分析，为我们揭示了流行文化这一重要部分的语言用法。

    We employ Natural Language Processing techniques to analyse 377808 English song lyrics from the "Two Million Song Database" corpus, focusing on the expression of sexism across five decades (1960-2010) and the measurement of gender biases. Using a sexism classifier, we identify sexist lyrics at a larger scale than previous studies using small samples of manually annotated popular songs. Furthermore, we reveal gender biases by measuring associations in word embeddings learned on song lyrics. We find sexist content to increase across time, especially from male artists and for popular songs appearing in Billboard charts. Songs are also shown to contain different language biases depending on the gender of the performer, with male solo artist songs containing more and stronger biases. This is the first large scale analysis of this type, giving insights into language usage in such an influential part of popular culture.
    
[^56]: 了解你的听众：用听众减法专门化基于上下文的语言模型

    Know your audience: specializing grounded language models with listener subtraction. (arXiv:2206.08349v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.08349](http://arxiv.org/abs/2206.08349)

    本文介绍了一种利用多智能体图像参照游戏自适应不同听众的目标任务描述的方法，并通过微调 CLIP 视觉编码器和大型语言模型之间的适配器，在适应听众的语言上下文的情况下进行了自然语言专业化。

    

    有效的沟通需要适应每个交际情境的特殊性，比如与每个交互伙伴分享的共同语境。本研究通过借鉴对话游戏 Dixit 的思想设计了一个多智能体图像参照游戏，训练一个说话者模型来描述一个目标图像，使得一个听者能够在干扰项中正确地识别出目标图像，而另一个听者则不能。这要求说话者利用它与不同听者的共同知识差异进行适应。本研究还展示了在这种对比、多智能体的语境下微调 CLIP 视觉编码器和大型语言模型之间的注意力适配器会自然地产生上下文依赖的自然语言专业化，且只需要通过奖励而无需直接监督来实现。通过控制实验，本研究证明了用两个听者来训练说话者的有效性。

    Effective communication requires adapting to the idiosyncrasies of each communicative context--such as the common ground shared with each partner. Humans demonstrate this ability to specialize to their audience in many contexts, such as the popular game Dixit. We take inspiration from Dixit to formulate a multi-agent image reference game where a (trained) speaker model is rewarded for describing a target image such that one (pretrained) listener model can correctly identify it among distractors, but another listener cannot. To adapt, the speaker must exploit differences in the knowledge it shares with the different listeners. We show that finetuning an attention-based adapter between a CLIP vision encoder and a large language model in this contrastive, multi-agent setting gives rise to context-dependent natural language specialization from rewards only, without direct supervision. Through controlled experiments, we show that training a speaker with two listeners that perceive different
    
[^57]: 基于分类重新参数化技巧的回译端到端训练

    End-to-End Training for Back-Translation with Categorical Reparameterization Trick. (arXiv:2202.08465v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2202.08465](http://arxiv.org/abs/2202.08465)

    本文提出了一种基于分类重新参数化技巧的回译端到端训练方法，来有效地减少两个神经机器翻译模型间离散属性的影响，从而实现端到端式的训练，获得了比以前基准测试更好的BLEU分数。

    

    回译是一种在神经机器翻译中有效的半监督学习框架。预先训练的神经机器翻译模型翻译单语句子并生成合成的双语句对以训练另一个神经机器翻译模型，反之亦然。将两个神经机器翻译模型分别理解为推理和生成模型。以往的研究采用了变分自动编码器（VAE）的培训框架。但是，由于翻译句子的离散属性使得梯度信息无法在两个NMT模型之间流动。本文提出了一种分类重新参数化技巧，使得神经机器翻译模型能够生成可微分的句子，使得VAE的训练框架可以以端到端方式工作。我们的实验表明，我们的方法有效地训练了NMT模型，并在WMT翻译任务的数据集上取得比以前基准测试更好的BLEU分数。

    Back-translation is an effective semi-supervised learning framework in neural machine translation (NMT). A pre-trained NMT model translates monolingual sentences and makes synthetic bilingual sentence pairs for the training of the other NMT model, and vice versa. Understanding the two NMT models as inference and generation models, respectively, previous works applied the training framework of variational auto-encoder (VAE). However, the discrete property of translated sentences prevents gradient information from flowing between the two NMT models. In this paper, we propose a categorical reparameterization trick that makes NMT models generate differentiable sentences so that the VAE's training framework can work in the end-to-end fashion. Our experiments demonstrate that our method effectively trains the NMT models and achieves better BLEU scores than the previous baseline on the datasets of the WMT translation task.
    
[^58]: 通过连续声学通道进行多智能体通讯学习听说能力的实现

    Towards Learning to Speak and Hear Through Multi-Agent Communication over a Continuous Acoustic Channel. (arXiv:2111.02827v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2111.02827](http://arxiv.org/abs/2111.02827)

    本研究旨在通过提供一个智能体间的消息传递环境，使得智能体能够通过连续声学通道进行通讯并观察到新兴语言的产生与特点，结果表明：与离散型信号不同，声学讲话者学习使用冗余信息以提高侦听者的连贯性。

    

    多智能体强化学习已成为研究智能体间新兴通讯的有效手段，但对于连续声学通讯却鲜有研究。这更类似于人类获得语言的方式；人类婴儿主要通过与看护者的连续信号交互来习得语言。因此，我们现在的目标是提供一个平台，以开始填补人类和智能体通信之间的差距，让我们能够分析连续信号以及它们产生的方式，它们的特征以及它们与人类语言习得的关系。

    Multi-agent reinforcement learning has been used as an effective means to study emergent communication between agents, yet little focus has been given to continuous acoustic communication. This would be more akin to human language acquisition; human infants acquire language in large part through continuous signalling with their caregivers. We therefore ask: Are we able to observe emergent language between agents with a continuous communication channel? Our goal is to provide a platform to begin bridging the gap between human and agent communication, allowing us to analyse continuous signals, how they emerge, their characteristics, and how they relate to human language acquisition. We propose a messaging environment where a Speaker agent needs to convey a set of attributes to a Listener over a noisy acoustic channel. Using DQN to train our agents, we show that: (1) unlike the discrete case, the acoustic Speaker learns redundancy to improve Listener coherency, (2) the acoustic Speaker de
    
[^59]: 词向量：一项综述

    Word Embeddings: A Survey. (arXiv:1901.09069v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/1901.09069](http://arxiv.org/abs/1901.09069)

    这篇综述介绍了一些主要的词向量构建策略，称为word embeddings，这些策略基于分布假设，编码了语法和语义信息，并被证明在很多NLP任务中是有用的额外特征。

    

    这项工作列出并描述了近期主要的策略，基于分布假设，用于构建单词的固定长度、密集和分布式表示。 这些表示现在通常被称为词向量，并且除了编码出令人惊讶的语法和语义信息外，在许多下游NLP任务中已被证明是有用的额外特征。

    This work lists and describes the main recent strategies for building fixed-length, dense and distributed representations for words, based on the distributional hypothesis. These representations are now commonly called word embeddings and, in addition to encoding surprisingly good syntactic and semantic information, have been proven useful as extra features in many downstream NLP tasks.
    

