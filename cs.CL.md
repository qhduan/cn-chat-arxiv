# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proto-lm: A Prototypical Network-Based Framework for Built-in Interpretability in Large Language Models.](http://arxiv.org/abs/2311.01732) | Proto-lm是一种基于原型网络的大型语言模型（LLM）内置可解释性框架，通过在微调阶段学习可解释的嵌入来提供解释性，同时保持竞争性能。该方法为创建可解释性模型提供了新的可能性。 |
| [^2] | [Divergent Token Metrics: Measuring degradation to prune away LLM components -- and optimize quantization.](http://arxiv.org/abs/2311.01544) | 本研究引入了一种新的方法，即不同的令牌指标（DTM），用于评估压缩后的大型语言模型（LLM）。通过关注令牌的差异性，DTM提供了对模型压缩微妙之处的深入洞察，并且在不损害文本生成质量的情况下可以实现显著的精确度和稀疏度水平。该研究还提出了一种利用DTM进行模型稀疏化和量化的方法，并发现可以修剪掉超过90%的LLM组件和量化超过80%的参数。 |
| [^3] | [AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models.](http://arxiv.org/abs/2311.01305) | AWEQ是一种后训练量化和激活权重均衡方法，能够在大型语言模型中实现超低位量化和8-bit权重和激活量化，并通过改进的均衡方法减小量化偏差误差，提高模型的鲁棒性。 |
| [^4] | [COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances.](http://arxiv.org/abs/2311.01012) | COPAL-ID是一个印度尼西亚语言常识推理数据集，与以前的数据集相比，它融入了印尼本土和文化细微差别，提供了更自然的日常因果推理描绘。该数据集对于现有的多语言语言模型来说是一个更大的挑战，但对人类来说很容易。在测试中，最新的开源多语言模型在COPAL-ID上的准确率较低，仅为65.47%。 |
| [^5] | [OpinSummEval: Revisiting Automated Evaluation for Opinion Summarization.](http://arxiv.org/abs/2310.18122) | 本文提出了一个新的数据集OpinSummEval，对意见摘要进行自动化评估的可靠性进行重新评估。研究发现基于神经网络的度量通常优于非神经网络的度量，但即使是基于强大模型构建的度量也不能在所有维度上始终保持良好的相关性，突出了对意见摘要自动化评估方法的进一步改进的需求。 |
| [^6] | [StyleBART: Decorate Pretrained Model with Style Adapters for Unsupervised Stylistic Headline Generation.](http://arxiv.org/abs/2310.17743) | StyleBART是一种无监督的风格化标题生成方法，通过使用适配器来装饰预训练模型，可以生成具有多样风格的标题。与其他方法不同，StyleBART将风格学习和标题生成任务分离开来，在推理过程中可以自由组合基础模型和风格适配器。经过广泛的评估，StyleBART表现出了优秀的性能。 |
| [^7] | [Managing AI Risks in an Era of Rapid Progress.](http://arxiv.org/abs/2310.17688) | 在人工智能快速进展的时代，我们提出了管理即将到来的先进人工智能系统所带来的风险的优先事项。 |
| [^8] | [On the Interplay between Fairness and Explainability.](http://arxiv.org/abs/2310.16607) | 公平的NLP模型并不总是依赖于更合理的解释，偏见缓解算法并不总是导致更公平的模型。 |
| [^9] | [Contrastive Learning for Inference in Dialogue.](http://arxiv.org/abs/2310.12467) | 本论文分析了推理任务中的信息差异对模型的影响，并提出了一种对比学习方法来缓解这种信息差异。实验证明，负样本有助于模型改进其推理生成能力。 |
| [^10] | [The Curious Case of Hallucinatory Unanswerablity: Finding Truths in the Hidden States of Over-Confident Large Language Models.](http://arxiv.org/abs/2310.11877) | 本研究探讨了大型语言模型(LLMs)在面对无法回答的查询时的行为，发现模型能够编码查询的可回答性，并且第一个解码的标记是一个强有力的指示符。这些发现揭示了LLMs潜在表示中的空间组织，并为改进解码技术提供了新的思路。 |
| [^11] | [Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets.](http://arxiv.org/abs/2310.11715) | 通过利用粗粒度数据集，提出了一种细粒度命名实体识别模型，使用细粒度-粗粒度映射矩阵来显式利用层次结构，并提出了一种不一致性过滤方法，以增强低资源细粒度命名实体识别。 |
| [^12] | [Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs.](http://arxiv.org/abs/2310.11689) | 本研究提出了一种自适应框架，利用自我评估来改进大型语言模型（LLMs）的选择性预测能力。该方法基于参数效率调整，能够适应特定任务并提高其自我评估能力，实验结果表明其优于最先进的选择性预测方法。 |
| [^13] | [Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning.](http://arxiv.org/abs/2310.11670) | 基于原型的超适配器（PHA）框架用于样本高效多任务调整，通过引入实例密集的检索器和样本高效的原型超网络生成条件模块，在多任务学习和少样本迁移学习中取得了可比性能的提升，甚至在数据量较小时也能超过其他强基线方法的性能。 |
| [^14] | [Dont Add, dont Miss: Effective Content Preserving Generation from Pre-Selected Text Spans.](http://arxiv.org/abs/2310.09017) | 本论文介绍了一个高质量的受控文本缩减（CTR）模型，解决了内容保留约束不充分强制执行和次优的银标签训练数据的限制，通过在训练和推理中增强内容保留约束，进一步改进了模型性能。 |
| [^15] | [Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization.](http://arxiv.org/abs/2310.05317) | 该论文提出了一种任务自适应分词的方法，通过优化分词过程来增强在心理健康领域中的长文本生成。实验证明，该方法在减少标记数量的情况下显著提高了生成性能，并且可与大型语言模型结合使用。 |
| [^16] | [An Investigation of LLMs' Inefficacy in Understanding Converse Relations.](http://arxiv.org/abs/2310.05163) | 本论文调查了LLMs在理解反向关系方面的无效性。作者引入了一个名为ConvRe的新基准，专注于逆向关系。通过两个任务Re2Text和Text2Re，作者评估了LLMs确定关系和相关文本之间匹配能力。实验结果揭示了LLMs在此方面的限制。 |
| [^17] | [FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets.](http://arxiv.org/abs/2310.04793) | 本文介绍了一种针对金融环境下开源语言模型的指令调优方法，并提出了端到端训练和测试的基准测试方案，以加强模型在金融数据集上的专业能力。 |
| [^18] | [Large-Scale Korean Text Dataset for Classifying Biased Speech in Real-World Online Services.](http://arxiv.org/abs/2310.04313) | 该论文介绍了一个大规模韩文文本数据集，用于分类有偏见言论。通过使用最先进的语言模型，该方法在多项分类任务中实现了超越人类水平的准确性。 |
| [^19] | [Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference.](http://arxiv.org/abs/2310.03184) | 通过检索增强的生成模型来改进数学问答，在可靠性和人类偏好之间进行权衡 |
| [^20] | [Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies.](http://arxiv.org/abs/2309.00312) | 本研究提出了一种比较话题建模方法，用于分析马克白彦病研究中存在矛盾结果的报告。通过对比不同话题与显著结果的相关性，找到了与黄斑变性研究中显著结果报告相关的八种化合物。 |
| [^21] | [Activation Addition: Steering Language Models Without Optimization.](http://arxiv.org/abs/2308.10248) | 这项研究探讨了一种在推理时通过改变激活来预测性地改变语言模型行为的方法，并且相比于传统方法具有更低的计算和实施成本，并且能够保持模型性能。 |
| [^22] | [Time Travel in LLMs: Tracing Data Contamination in Large Language Models.](http://arxiv.org/abs/2308.08493) | 该论文提出了一种用于识别大型语言模型（LLMs）中数据污染的简单而有效的方法。通过对随机样本中的单个实例进行分析，以及使用“引导指令”来评估整个数据集分区的污染程度，可以准确地识别污染的实例和分区。 |
| [^23] | [Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic.](http://arxiv.org/abs/2308.07336) | 本研究研究了一种从合成语料库中学习演绎推理能力的方法，通过采用基于形式逻辑理论的演绎规则，训练的语言模型具有更泛化的推理能力。 |
| [^24] | [The Imitation Game: Detecting Human and AI-Generated Texts in the Era of Large Language Models.](http://arxiv.org/abs/2307.12166) | 本论文研究了区分人类和AI生成的文本的任务，在不同体裁下进行了比较研究，提出了一个新的数据集，并采用多种机器学习模型进行分类。结果表明这些模型对于区分人类和AI生成的文本具有很高的效力，尽管在区分GPT生成的文本方面存在一定挑战。 |
| [^25] | [AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models.](http://arxiv.org/abs/2307.11772) | AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。 |
| [^26] | [EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus.](http://arxiv.org/abs/2307.11760) | EmotionPrompt是一个基于心理学的方法，通过将情感刺激融入到提示中，提升了大型语言模型在各项任务上的性能，并且同时改善了其真实性和信息量。 |
| [^27] | [A Systematic Evaluation of Federated Learning on Biomedical Natural Language Processing.](http://arxiv.org/abs/2307.11254) | 本研究对医学领域中的联邦学习在生物医学自然语言处理中的应用进行了系统评估，结果显示联邦学习模型优于单独训练的模型，并且在考虑数据隐私的情况下仍能取得良好的效果。 |
| [^28] | [No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models.](http://arxiv.org/abs/2307.06440) | 本论文重新审视了基于Transformer的语言模型的高效训练算法，包括动态架构，批量选择和高效优化器。然而，在使用这些算法预训练时，相对于基线方法，它们的训练、验证和下游收益消失了。同时，论文提出了一个评估协议来进行计算，并释放了代码来促进高效训练的研究。 |
| [^29] | [Style Over Substance: Evaluation Biases for Large Language Models.](http://arxiv.org/abs/2307.03025) | 这项研究调查了人类和基于大型语言模型的评委在比较不同模型输出时的行为，并发现评估过程中存在偏见，即尽管包含事实错误，答案仍然被更高地评分。为了解决这个问题，我们提出了 |
| [^30] | [Learning to Generate Better Than Your LLM.](http://arxiv.org/abs/2306.11816) | 本论文研究了基于强化学习算法 RLGF，用于在 GPT-3 等动态黑匣子指导下微调大型语言模型 LLM 的条件文本生成，相比通用 RL 算法，该算法在 IMDB 和 CommonGen 任务中表现更好。 |
| [^31] | [Data Augmentation Approaches for Source Code Models: A Survey.](http://arxiv.org/abs/2305.19915) | 本文对源代码的数据增强技术进行了全面的调查和综述，介绍了它们的分类法、优化策略和性能结果，并讨论了未来方向和研究挑战。 |
| [^32] | [Goal-Driven Explainable Clustering via Language Descriptions.](http://arxiv.org/abs/2305.13749) | 该研究提出了一种“带解释的基于目标的聚类”（GoalEx）的新任务形式，它将目标和解释都表示为自由形式的语言描述。通过将摘要系统的注释进行分类来说明研究的有效性以及生成的解释。 |
| [^33] | [Cross-Attention is Not Enough: Incongruity-Aware Multimodal Sentiment Analysis and Emotion Recognition.](http://arxiv.org/abs/2305.13583) | 本文提出了一种基于不协调感知的跨模态情感分析方法，通过Hierarchical Crossmodal Transformer with Modality Gating(HCT-MG)模型来确定主要模态并分层融合辅助模态，有效减轻模态之间的不协调感知和信息冗余问题。 |
| [^34] | [Transfer-Free Data-Efficient Multilingual Slot Labeling.](http://arxiv.org/abs/2305.13528) | 本论文提出了一种无需英文数据的多语言数据高效标记方法，结果显示其比跨语言转移基准显着提高（最多提高22%）。 |
| [^35] | [MixPro: Simple yet Effective Data Augmentation for Prompt-based Learning.](http://arxiv.org/abs/2304.09402) | MixPro是一种数据增强方法，通过对原始输入和模板进行混合来提高基于提示的学习性能，平均提高了5.08%的模型性能。 |
| [^36] | [Larger Probes Tell a Different Story: Extending Psycholinguistic Datasets Via In-Context Learning.](http://arxiv.org/abs/2303.16445) | 本文通过上下文学习扩展否定和角色反转数据集，发现过去的结论可能被小型测试集误导。同时，BERT和ALBERT等模型表现出较高的否定敏感度。 |
| [^37] | [From Wide to Deep: Dimension Lifting Network for Parameter-efficient Knowledge Graph Embedding.](http://arxiv.org/abs/2303.12816) | 本文提出了一个用于实现参数高效的知识图谱嵌入的深度网络，通过增加深度克服因采用低维实体表示而导致的模型精度下降和模型参数减少有限的问题。 |
| [^38] | [The Image of the Process Interpretation of Regular Expressions is Not Closed under Bisimulation Collapse.](http://arxiv.org/abs/2303.08553) | 论文探讨了一种存在于正则表达式进程语义中的双模折叠不封闭性问题，并在1-free正则表达式的解释中发现了对这种难题的关键原因，进一步提出了LEE属性的特征，证明了1-free正则表达式的方程证明系统是完备的，并且多项式时间可以解决解释和过程图双模相似性问题。 |
| [^39] | [Interactive Text Generation.](http://arxiv.org/abs/2303.00908) | 本研究提出了一种新的交互式文本生成任务，使用用户模拟器进行交互式训练生成模型，避免了真实用户参与的成本，并通过提供编辑指导模型朝着给定目标前进，从而提高了生成质量。 |
| [^40] | [InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis.](http://arxiv.org/abs/2302.08624) | InstructABSA是一种使用指令学习范式的方面情感分析方法，能够显著提高Aspect Term Extraction、Aspect Term Sentiment Classification、和Joint Task subtasks三个子任务的性能，并且在多个数据集上表现超过之前的最先进方法。 |
| [^41] | [Theory of Mind May Have Spontaneously Emerged in Large Language Models.](http://arxiv.org/abs/2302.02083) | “通过测试多个语言模型在解决40个ToM任务上的表现，研究发现GPT-3和GPT-4能够解决大部分任务，说明类似ToM的能力可能是语言模型自发出现的附带产物。” |
| [^42] | [SciRepEval: A Multi-Format Benchmark for Scientific Document Representations.](http://arxiv.org/abs/2211.13308) | SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。 |
| [^43] | [HyperMixer: An MLP-based Low Cost Alternative to Transformers.](http://arxiv.org/abs/2203.03691) | HyperMixer是一种低成本的基于MLP的Transformer替代方案，通过动态形成标记混合MLP来实现自然语言理解，其性能比替代方案好，并可与Transformer媲美，成本更低。 |

# 详细

[^1]: Proto-lm：一种基于原型网络的大型语言模型内置可解释性框架

    Proto-lm: A Prototypical Network-Based Framework for Built-in Interpretability in Large Language Models. (arXiv:2311.01732v1 [cs.CL])

    [http://arxiv.org/abs/2311.01732](http://arxiv.org/abs/2311.01732)

    Proto-lm是一种基于原型网络的大型语言模型（LLM）内置可解释性框架，通过在微调阶段学习可解释的嵌入来提供解释性，同时保持竞争性能。该方法为创建可解释性模型提供了新的可能性。

    

    大型语言模型（LLM）在自然语言处理（NLP）领域有显著进展，但其缺乏可解释性是一个主要关注点。目前用于解释LLMs的方法是事后的，在推理时间之后应用，并且存在一些限制，比如它们关注低级特征并且在更高级文本单位上缺乏可解释性。在这项工作中，我们引入了proto-lm，这是一个基于原型网络的白盒子框架，允许LLMs在微调阶段学习即时可解释的嵌入，同时保持具有竞争力的性能。通过对各种NLP任务的实验，我们证明了我们方法的适用性和可解释性，并且我们的结果表明了在不牺牲性能的情况下创建可解释性模型的新可能性。这种在LLMs中的新颖解释性方法可以为无需牺牲性能的更可解释性模型铺平道路。

    Large Language Models (LLMs) have significantly advanced the field of Natural Language Processing (NLP), but their lack of interpretability has been a major concern. Current methods for interpreting LLMs are post hoc, applied after inference time, and have limitations such as their focus on low-level features and lack of explainability at higher level text units. In this work, we introduce proto-lm, a prototypical network-based white-box framework that allows LLMs to learn immediately interpretable embeddings during the fine-tuning stage while maintaining competitive performance. Our method's applicability and interpretability are demonstrated through experiments on a wide range of NLP tasks, and our results indicate a new possibility of creating interpretable models without sacrificing performance. This novel approach to interpretability in LLMs can pave the way for more interpretable models without the need to sacrifice performance.
    
[^2]: 不同的令牌指标：通过测量衰减来修剪LLM组件并优化量化

    Divergent Token Metrics: Measuring degradation to prune away LLM components -- and optimize quantization. (arXiv:2311.01544v1 [cs.CL])

    [http://arxiv.org/abs/2311.01544](http://arxiv.org/abs/2311.01544)

    本研究引入了一种新的方法，即不同的令牌指标（DTM），用于评估压缩后的大型语言模型（LLM）。通过关注令牌的差异性，DTM提供了对模型压缩微妙之处的深入洞察，并且在不损害文本生成质量的情况下可以实现显著的精确度和稀疏度水平。该研究还提出了一种利用DTM进行模型稀疏化和量化的方法，并发现可以修剪掉超过90%的LLM组件和量化超过80%的参数。

    

    大型语言模型（LLM）以其强大的能力改变了自然语言处理。然而，它们不断增长的大小引发了关于它们的有效部署和LLM压缩的担忧。本研究介绍了一种新的评估压缩LLM的方法，即不同的令牌指标（DTM），解决了传统指标如困惑度无法准确反映文本生成质量的局限性。DTM关注令牌的差异性，提供了对模型压缩微妙之处的更深入洞察。我们的结果表明，在不损害文本生成质量的情况下，可以达到显著的精确度和稀疏度水平。此外，DTM还可以更精确地评估每个组件的影响。利用第一个不同的令牌指标（FDTM）在模型稀疏化中显示，超过90%的所有组件可以修剪掉。对于量化，FDTM表明超过80%的参数可以进行量化。

    Large Language Models (LLMs) have reshaped natural language processing with their impressive capabilities. Their ever-increasing size, however, raised concerns about their effective deployment and the need for LLM compressions. This study introduces the Divergent Token metrics (DTMs), a novel approach for assessing compressed LLMs, addressing the limitations of traditional measures like perplexity that fail to accurately reflect text generation quality. DTMs focus on token divergence, providing deeper insights into the subtleties of model compression. Our results indicate that significant levels of precision and sparsity can be achieved without compromising text generation quality. Moreover, DTMs offers a more precise evaluation of each component's impact individually. Utilizing the First Divergent Token metric (FDTM) in model sparsification reveals that nearly 20% of all components can be pruned over 90%. In terms of quantization, the FDTM suggests that over 80% of parameters can be s
    
[^3]: AWEQ：用于大型语言模型的后训练量化和激活权重均衡方法

    AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models. (arXiv:2311.01305v1 [cs.LG])

    [http://arxiv.org/abs/2311.01305](http://arxiv.org/abs/2311.01305)

    AWEQ是一种后训练量化和激活权重均衡方法，能够在大型语言模型中实现超低位量化和8-bit权重和激活量化，并通过改进的均衡方法减小量化偏差误差，提高模型的鲁棒性。

    

    大型语言模型(LLMs)在各种任务中表现出色，但其计算和存储成本也相对较高。量化这些模型是缓解这个问题的有效方法。然而，现有方法很难在模型准确性和硬件效率之间取得平衡。因此，我们引入了AWEQ，一种后训练方法，不需要额外的训练开销。AWEQ在超低位量化和8-bit权重和激活(W8A8)量化方面表现出色。观察到权重量化比激活量化更容易。AWEQ通过通道均衡将激活量化的难度转移到权重上，实现了两者量化困难的平衡，从而最大化了性能。我们进一步改进了均衡方法，减小了量化偏差误差，确保模型的鲁棒性。在像LLaMA这样的流行模型上进行了大量实验。

    Large language models(LLMs) exhibit excellent performance across a variety of tasks, but they come with significant computational and storage costs. Quantizing these models is an effective way to alleviate this issue. However, existing methods struggle to strike a balance between model accuracy and hardware efficiency. This is where we introduce AWEQ, a post-training method that requires no additional training overhead. AWEQ excels in both ultra-low-bit quantization and 8-bit weight and activation (W8A8) quantization. There is an observation that weight quantization is less challenging than activation quantization. AWEQ transfers the difficulty of activation quantization to weights using channel equalization, achieving a balance between the quantization difficulties of both, and thereby maximizing performance. We have further refined the equalization method to mitigate quantization bias error, ensuring the robustness of the model. Extensive experiments on popular models such as LLaMA a
    
[^4]: COPAL-ID: 印度尼西亚语言推理与本土文化和细微差别

    COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances. (arXiv:2311.01012v1 [cs.CL])

    [http://arxiv.org/abs/2311.01012](http://arxiv.org/abs/2311.01012)

    COPAL-ID是一个印度尼西亚语言常识推理数据集，与以前的数据集相比，它融入了印尼本土和文化细微差别，提供了更自然的日常因果推理描绘。该数据集对于现有的多语言语言模型来说是一个更大的挑战，但对人类来说很容易。在测试中，最新的开源多语言模型在COPAL-ID上的准确率较低，仅为65.47%。

    

    我们介绍了公开可用的COPAL-ID，这是一个新颖的印度尼西亚语言常识推理数据集。与以前的印尼COPA数据集（XCOPA-ID）不同，COPAL-ID融入了印尼本土和文化细微差别，因此在印尼文化领域内提供了更自然的日常因果推理描绘。COPAL-ID由本土人从头开始专业撰写，更流利，不像XCOPA-ID的翻译存在尴尬的词语。此外，我们以标准印度尼西亚语和雅加达印度尼西亚语（一种在日常对话中常用的方言）呈现COPAL-ID。COPAL-ID对于现有的开源和闭源最先进的多语言语言模型来说，提出了更大的挑战，对于人类来说却是非常容易的。我们的调查结果表明，即使是当前最好的开源多语言模型也很难表现出色，在COPAL-ID上的准确率为65.47%，远低于没有文化背景的XCOPA-ID（79.40%）。

    We present publicly available COPAL-ID, a novel Indonesian language common sense reasoning dataset. Unlike the previous Indonesian COPA dataset (XCOPA-ID), COPAL-ID incorporates Indonesian local and cultural nuances, and therefore, provides a more natural portrayal of day-to-day causal reasoning within the Indonesian cultural sphere. Professionally written by natives from scratch, COPAL-ID is more fluent and free from awkward phrases, unlike the translated XCOPA-ID. In addition, we present COPAL-ID in both standard Indonesian and in Jakartan Indonesian--a dialect commonly used in daily conversation. COPAL-ID poses a greater challenge for existing open-sourced and closed state-of-the-art multilingual language models, yet is trivially easy for humans. Our findings suggest that even the current best open-source, multilingual model struggles to perform well, achieving 65.47% accuracy on COPAL-ID, significantly lower than on the culturally-devoid XCOPA-ID (79.40%). Despite GPT-4's impressiv
    
[^5]: OpinSummEval:再考自动化评估在意见摘要中的应用

    OpinSummEval: Revisiting Automated Evaluation for Opinion Summarization. (arXiv:2310.18122v1 [cs.CL])

    [http://arxiv.org/abs/2310.18122](http://arxiv.org/abs/2310.18122)

    本文提出了一个新的数据集OpinSummEval，对意见摘要进行自动化评估的可靠性进行重新评估。研究发现基于神经网络的度量通常优于非神经网络的度量，但即使是基于强大模型构建的度量也不能在所有维度上始终保持良好的相关性，突出了对意见摘要自动化评估方法的进一步改进的需求。

    

    与其他类型的摘要任务不同，意见摘要专注于观点和情感，因此与众不同。虽然像ROUGE这样的某些自动化评估方法很受欢迎，但我们发现它们对评估意见摘要的质量是不可靠的。在本文中，我们提出了一个数据集OpinSummEval，它包括来自14个意见摘要模型的人工判断和输出。我们进一步探讨了24个自动度量与人工评分之间的相关性，涵盖了四个维度。我们的研究结果表明，基于神经网络的度量通常优于非神经网络的度量。然而，即使是基于强大模型（如BART和GPT-3/3.5）构建的度量也不能在所有维度上始终保持良好的相关性，突出了需要改进意见摘要的自动化评估方法的需求。代码和数据公开可用于https://github.com/A-Chicharito-S/OpinSummEval/tree/main。

    Opinion summarization sets itself apart from other types of summarization tasks due to its distinctive focus on aspects and sentiments. Although certain automated evaluation methods like ROUGE have gained popularity, we have found them to be unreliable measures for assessing the quality of opinion summaries. In this paper, we present OpinSummEval, a dataset comprising human judgments and outputs from 14 opinion summarization models. We further explore the correlation between 24 automatic metrics and human ratings across four dimensions. Our findings indicate that metrics based on neural networks generally outperform non-neural ones. However, even metrics built on powerful backbones, such as BART and GPT-3/3.5, do not consistently correlate well across all dimensions, highlighting the need for advancements in automated evaluation methods for opinion summarization. The code and data are publicly available at https://github.com/A-Chicharito-S/OpinSummEval/tree/main.
    
[^6]: StyleBART: 使用风格适配器装饰预训练模型进行无监督风格化标题生成

    StyleBART: Decorate Pretrained Model with Style Adapters for Unsupervised Stylistic Headline Generation. (arXiv:2310.17743v1 [cs.CL])

    [http://arxiv.org/abs/2310.17743](http://arxiv.org/abs/2310.17743)

    StyleBART是一种无监督的风格化标题生成方法，通过使用适配器来装饰预训练模型，可以生成具有多样风格的标题。与其他方法不同，StyleBART将风格学习和标题生成任务分离开来，在推理过程中可以自由组合基础模型和风格适配器。经过广泛的评估，StyleBART表现出了优秀的性能。

    

    风格化标题生成任务是生成一个既总结文章内容又反映所需风格来吸引用户的标题。由于风格特定的文章-标题对非常稀缺，先前的研究主要关注于使用标准标题生成数据集和单一风格语料库进行无监督方法。在本研究中，我们遵循这一路线，并提出了StyleBART，一种无监督的风格化标题生成方法。我们的方法使用适配器将预训练的BART模型装饰起来，适配器负责不同的风格，通过简单地切换适配器，可以生成具有多样风格的标题。与之前的工作不同，StyleBART将风格学习和标题生成的任务分离开来，在推理过程中可以自由组合基础模型和风格适配器。我们进一步提出了一个逆向改写任务以增强风格适配器的效果。广泛的自动和人工评估结果表明，StyleBART取得了很好的性能。

    Stylistic headline generation is the task to generate a headline that not only summarizes the content of an article, but also reflects a desired style that attracts users. As style-specific article-headline pairs are scarce, previous researches focus on unsupervised approaches with a standard headline generation dataset and mono-style corpora. In this work, we follow this line and propose StyleBART, an unsupervised approach for stylistic headline generation. Our method decorates the pretrained BART model with adapters that are responsible for different styles and allows the generation of headlines with diverse styles by simply switching the adapters. Different from previous works, StyleBART separates the task of style learning and headline generation, making it possible to freely combine the base model and the style adapters during inference. We further propose an inverse paraphrasing task to enhance the style adapters. Extensive automatic and human evaluations show that StyleBART achi
    
[^7]: 在快速发展时代管理人工智能风险

    Managing AI Risks in an Era of Rapid Progress. (arXiv:2310.17688v1 [cs.CY] CROSS LISTED)

    [http://arxiv.org/abs/2310.17688](http://arxiv.org/abs/2310.17688)

    在人工智能快速进展的时代，我们提出了管理即将到来的先进人工智能系统所带来的风险的优先事项。

    

    在这篇简短的共识文中，我们概述了即将到来的先进人工智能系统所带来的风险。我们审查了大规模的社会危害和恶意使用，以及人类对自主人工智能系统失去控制的不可逆转的损失。鉴于人工智能的快速和持续进展，我们提出了人工智能研发和治理的优先事项。

    In this short consensus paper, we outline risks from upcoming, advanced AI systems. We examine large-scale social harms and malicious uses, as well as an irreversible loss of human control over autonomous AI systems. In light of rapid and continuing AI progress, we propose priorities for AI R&D and governance.
    
[^8]: 公平性与可解释性之间的相互作用

    On the Interplay between Fairness and Explainability. (arXiv:2310.16607v1 [cs.CL])

    [http://arxiv.org/abs/2310.16607](http://arxiv.org/abs/2310.16607)

    公平的NLP模型并不总是依赖于更合理的解释，偏见缓解算法并不总是导致更公平的模型。

    

    为了构建可靠和值得信赖的自然语言处理(NLP)应用，模型需要在不同的人口统计数据中既具有公平性又可解释。通常，这两个目标，即公平性和可解释性，会被独立地进行优化和/或研究。相反，我们认为未来可信的NLP系统应该同时考虑两者。在这项工作中，我们进行了首次研究，以了解它们如何相互影响：更公平的模型是否依赖于更合理的解释？反之亦然。为此，我们在两个英语多类文本分类数据集BIOS和ECtHR上进行实验，这些数据集分别提供了有关性别和国籍的信息，以及人工标注的解释。我们使用多种方法对预训练语言模型进行微调，包括(i)偏见缓解，旨在提高公平性；(ii)解释提取，旨在产生合理的解释。我们发现，偏见缓解算法并不总是导致更公平的模型。此外，我们还发现

    In order to build reliable and trustworthy NLP applications, models need to be both fair across different demographics and explainable. Usually these two objectives, fairness and explainability, are optimized and/or examined independently of each other. Instead, we argue that forthcoming, trustworthy NLP systems should consider both. In this work, we perform a first study to understand how they influence each other: do fair(er) models rely on more plausible rationales? and vice versa. To this end, we conduct experiments on two English multi-class text classification datasets, BIOS and ECtHR, that provide information on gender and nationality, respectively, as well as human-annotated rationales. We fine-tune pre-trained language models with several methods for (i) bias mitigation, which aims to improve fairness; (ii) rationale extraction, which aims to produce plausible explanations. We find that bias mitigation algorithms do not always lead to fairer models. Moreover, we discover that 
    
[^9]: 对话中的对比学习推理

    Contrastive Learning for Inference in Dialogue. (arXiv:2310.12467v1 [cs.CL])

    [http://arxiv.org/abs/2310.12467](http://arxiv.org/abs/2310.12467)

    本论文分析了推理任务中的信息差异对模型的影响，并提出了一种对比学习方法来缓解这种信息差异。实验证明，负样本有助于模型改进其推理生成能力。

    

    推理,尤其是那些来自归纳过程的推理,是我们对话中的一个关键组成部分，用于补充由讲话者隐含或明确传达的信息。虽然最近的大型语言模型在推理任务上取得了显著进展，但它们在归纳推理方面的表现远远落后于演绎推理。在本文中，我们根据语义信息差异来定义任务难度，分析了模型的行为，该差异区分了归纳推理和演绎推理（Johnson-Laird, 1988, 1993）。我们的分析揭示了对话上下文和所需推理之间信息差异的差距对归纳推理过程构成了重要挑战。为了缓解这种信息差距，我们研究了一种对比学习方法，通过提供负样本进行训练。我们的实验表明，负样本有助于模型理解错误并改进其推理生成能力。

    Inference, especially those derived from inductive processes, is a crucial component in our conversation to complement the information implicitly or explicitly conveyed by a speaker. While recent large language models show remarkable advances in inference tasks, their performance in inductive reasoning, where not all information is present in the context, is far behind deductive reasoning. In this paper, we analyze the behavior of the models based on the task difficulty defined by the semantic information gap -- which distinguishes inductive and deductive reasoning (Johnson-Laird, 1988, 1993). Our analysis reveals that the disparity in information between dialogue contexts and desired inferences poses a significant challenge to the inductive inference process. To mitigate this information gap, we investigate a contrastive learning approach by feeding negative samples. Our experiments suggest negative samples help models understand what is wrong and improve their inference generations.
    
[^10]: 大型语言模型中幻觉性无法回答性的好奇案例：在过度自信的隐藏状态中寻找真理

    The Curious Case of Hallucinatory Unanswerablity: Finding Truths in the Hidden States of Over-Confident Large Language Models. (arXiv:2310.11877v1 [cs.CL])

    [http://arxiv.org/abs/2310.11877](http://arxiv.org/abs/2310.11877)

    本研究探讨了大型语言模型(LLMs)在面对无法回答的查询时的行为，发现模型能够编码查询的可回答性，并且第一个解码的标记是一个强有力的指示符。这些发现揭示了LLMs潜在表示中的空间组织，并为改进解码技术提供了新的思路。

    

    大型语言模型(LLMs)展示了令人印象深刻的能力，同时也引发了对其回答准确性的关键担忧。在这个背景下出现的一个主要问题是LLMs如何处理无法回答的查询，往往会导致幻觉行为，原因是过度自信。在本文中，我们探讨了LLMs面对无法回答的查询时的行为。我们问：当生成幻觉回答时，模型是否表示问题无法回答的事实？我们的结果强烈表明，这样的模型对输入查询的可回答性进行编码，第一个解码的标记的表示往往是一个强有力的指示符。这些发现揭示了LLMs潜在表示中的空间组织，揭示了这些模型先前未被探索的方面。此外，它们为开发更好地遵守事实生成的改进解码技术铺平了道路。

    Large language models (LLMs) have been shown to possess impressive capabilities, while also raising crucial concerns about the faithfulness of their responses. A primary issue arising in this context is the management of unanswerable queries by LLMs, which often results in hallucinatory behavior, due to overconfidence. In this paper, we explore the behavior of LLMs when presented with unanswerable queries. We ask: do models \textbf{represent} the fact that the question is unanswerable when generating a hallucinatory answer? Our results show strong indications that such models encode the answerability of an input query, with the representation of the first decoded token often being a strong indicator. These findings shed new light on the spatial organization within the latent representations of LLMs, unveiling previously unexplored facets of these models. Moreover, they pave the way for the development of improved decoding techniques with better adherence to factual generation, particul
    
[^11]: 利用粗粒度数据集增强低资源细粒度命名实体识别

    Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets. (arXiv:2310.11715v1 [cs.CL])

    [http://arxiv.org/abs/2310.11715](http://arxiv.org/abs/2310.11715)

    通过利用粗粒度数据集，提出了一种细粒度命名实体识别模型，使用细粒度-粗粒度映射矩阵来显式利用层次结构，并提出了一种不一致性过滤方法，以增强低资源细粒度命名实体识别。

    

    命名实体识别（NER）在细粒度NER场景下常常面临标注数据不足的问题。虽然可以应用K-shot学习技术，但当注释数量超过几十个标签时，性能往往达到饱和。为了解决这个问题，我们利用现有的粗粒度数据集，其中包含大量的标注。一种直接解决这个问题的方法是预训练，它利用粗粒度数据进行表示学习。然而，它无法直接利用细粒度和粗粒度实体之间的关系，尽管细粒度实体类型很可能是粗粒度实体类型的子类别。我们提出了一个带有细粒度-粗粒度（F2C）映射矩阵的细粒度NER模型，以显式地利用层次结构。此外，我们提出了一种不一致性过滤方法，以消除与细粒度不一致的粗粒度实体。

    Named Entity Recognition (NER) frequently suffers from the problem of insufficient labeled data, particularly in fine-grained NER scenarios. Although $K$-shot learning techniques can be applied, their performance tends to saturate when the number of annotations exceeds several tens of labels. To overcome this problem, we utilize existing coarse-grained datasets that offer a large number of annotations. A straightforward approach to address this problem is pre-finetuning, which employs coarse-grained data for representation learning. However, it cannot directly utilize the relationships between fine-grained and coarse-grained entities, although a fine-grained entity type is likely to be a subcategory of a coarse-grained entity type. We propose a fine-grained NER model with a Fine-to-Coarse(F2C) mapping matrix to leverage the hierarchical structure explicitly. In addition, we present an inconsistency filtering method to eliminate coarse-grained entities that are inconsistent with fine-gr
    
[^12]: 自我评估的自适应改进LLMs中的选择性预测

    Adaptation with Self-Evaluation to Improve Selective Prediction in LLMs. (arXiv:2310.11689v1 [cs.CL])

    [http://arxiv.org/abs/2310.11689](http://arxiv.org/abs/2310.11689)

    本研究提出了一种自适应框架，利用自我评估来改进大型语言模型（LLMs）的选择性预测能力。该方法基于参数效率调整，能够适应特定任务并提高其自我评估能力，实验结果表明其优于最先进的选择性预测方法。

    

    大型语言模型（LLMs）在自然语言理解和生成等多种任务中取得了巨大进展。然而，在高风险决策场景中仍然限于其潜在的错误。选择性预测是一种可以通过在LLMs不确定时使其避免预测而提高其可靠性的技术。在本文中，我们提出了一种新颖的自我评估的自适应框架，以提高LLMs的选择性预测性能。我们的框架基于使用参数效率调整来适应特定任务并改进其自我评估能力的思想。我们在各种问答（QA）数据集上评估了我们的方法，并展示其优于最先进的选择性预测方法。例如，在CoQA基准测试中，我们的方法将AUACC从91.23%提高到92.63%，并将AURO

    Large language models (LLMs) have recently shown great advances in a variety of tasks, including natural language understanding and generation. However, their use in high-stakes decision-making scenarios is still limited due to the potential for errors. Selective prediction is a technique that can be used to improve the reliability of the LLMs by allowing them to abstain from making predictions when they are unsure of the answer. In this work, we propose a novel framework for adaptation with self-evaluation to improve the selective prediction performance of LLMs. Our framework is based on the idea of using parameter-efficient tuning to adapt the LLM to the specific task at hand while improving its ability to perform self-evaluation. We evaluate our method on a variety of question-answering (QA) datasets and show that it outperforms state-of-the-art selective prediction methods. For example, on the CoQA benchmark, our method improves the AUACC from 91.23% to 92.63% and improves the AURO
    
[^13]: 基于原型的超适配器用于样本高效多任务调整

    Prototype-based HyperAdapter for Sample-Efficient Multi-task Tuning. (arXiv:2310.11670v1 [cs.CL])

    [http://arxiv.org/abs/2310.11670](http://arxiv.org/abs/2310.11670)

    基于原型的超适配器（PHA）框架用于样本高效多任务调整，通过引入实例密集的检索器和样本高效的原型超网络生成条件模块，在多任务学习和少样本迁移学习中取得了可比性能的提升，甚至在数据量较小时也能超过其他强基线方法的性能。

    

    参数高效微调（PEFT）已经证明在适应预训练语言模型到下游任务时有效，同时只更新了少量参数。尽管取得了成功，大多数现有方法独立地适应每个任务，没有考虑任务之间的知识传输，并且受限于低数据情景。为了克服这个问题，我们提出了一种基于原型的超适配器（PHA）框架，该框架建立在适配器调整和超网络基础上。它引入了一个实例密集的检索器和一个样本高效的原型超网络来生成条件模块。这导致与现有PEFT方法在多任务学习和少样本迁移学习上相当的性能改进。更重要的是，当可用数据量变小时，我们的方法比其他强基线方法有很大的优势。基于我们在各种数据集上的广泛实证实验，我们证明了PHA在权衡方面取得了更好的结果。

    Parameter-efficient fine-tuning (PEFT) has shown its effectiveness in adapting the pre-trained language models to downstream tasks while only updating a small number of parameters. Despite the success, most existing methods independently adapt to each task without considering knowledge transfer between tasks and are limited to low-data regimes. To overcome this issue, we propose Prototype-based HyperAdapter (PHA), a novel framework built on the adapter-tuning and hypernetwork. It introduces an instance-dense retriever and a prototypical hypernetwork to generate the conditional modules in a sample-efficient manner. This leads to comparable performance improvements against existing PEFT methods on multi-task learning and few-shot transfer learning. More importantly, when the available data size gets smaller, our method outperforms other strong baselines by a large margin. Based on our extensive empirical experiments across various datasets, we demonstrate that PHA strikes a better trade-
    
[^14]: 不添加，不错过：从预选文本段生成有效的内容保留生成模型

    Dont Add, dont Miss: Effective Content Preserving Generation from Pre-Selected Text Spans. (arXiv:2310.09017v1 [cs.CL])

    [http://arxiv.org/abs/2310.09017](http://arxiv.org/abs/2310.09017)

    本论文介绍了一个高质量的受控文本缩减（CTR）模型，解决了内容保留约束不充分强制执行和次优的银标签训练数据的限制，通过在训练和推理中增强内容保留约束，进一步改进了模型性能。

    

    最近引入的受控文本缩减（CTR）任务在典型的摘要任务中将文本生成步骤隔离出来。它通过挑战模型在输入文本的预选内容（"高亮"）中生成连贯的文本来实现。这种框架在类似摘要的任务中增加了模块化能力，允许将单个CTR模型与各种内容选择设置和模块配对使用。然而，目前还没有可靠的CTR模型，而且现有任务基线的性能中等，无法实际使用。为了填补这个空白，我们引入了一个高质量的开源CTR模型，解决了两个先前的关键限制：不充分强制执行内容保留约束和次优的银标签训练数据。通过在训练中通过强化学习和推理中通过受控解码策略来增强内容保留约束。此外，我们还大幅改进了银标签训练数据。

    The recently introduced Controlled Text Reduction (CTR) task isolates the text generation step within typical summarization-style tasks. It does so by challenging models to generate coherent text conforming to pre-selected content within the input text ("highlights").  This framing enables increased modularity in summarization-like tasks, allowing to couple a single CTR model with various content-selection setups and modules.  However, there are currently no reliable CTR models, while the performance of the existing baseline for the task is mediocre, falling short of practical utility.  Here, we address this gap by introducing a high-quality, open-source CTR model that tackles two prior key limitations: inadequate enforcement of the content-preservation constraint, and suboptimal silver training data.  Addressing these, we amplify the content-preservation constraint in both training, via RL, and inference, via a controlled decoding strategy.  Further, we substantially improve the silve
    
[^15]: 在心理健康领域中通过任务自适应分词来增强长文本生成

    Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization. (arXiv:2310.05317v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05317](http://arxiv.org/abs/2310.05317)

    该论文提出了一种任务自适应分词的方法，通过优化分词过程来增强在心理健康领域中的长文本生成。实验证明，该方法在减少标记数量的情况下显著提高了生成性能，并且可与大型语言模型结合使用。

    

    我们提出了任务自适应分词作为一种方式，将生成流水线适应于下游任务的特定要求，并增强在心理健康领域的长文本生成。受认知科学的启发，我们的任务自适应分词器从多个结果中采样可变的分段，采样概率基于任务特定的数据进行优化。我们引入了一种构建专用词汇的策略，并介绍了一种词汇合并协议，可以将任务特定的标记整合到预训练模型的分词步骤中。通过对中英文心理问答任务进行广泛实验，我们发现我们的任务自适应分词方法在使用更少的标记的情况下带来了显著的生成性能提升，最高可达60%。初步实验表明，使用我们的分词方法与非常大的语言模型结合能够得到有希望的结果。

    We propose task-adaptive tokenization as a way to adapt the generation pipeline to the specifics of a downstream task and enhance long-form generation in mental health. Inspired by insights from cognitive science, our task-adaptive tokenizer samples variable segmentations from multiple outcomes, with sampling probabilities optimized based on task-specific data. We introduce a strategy for building a specialized vocabulary and introduce a vocabulary merging protocol that allows for the integration of task-specific tokens into the pre-trained model's tokenization step. Through extensive experiments on psychological question-answering tasks in both Chinese and English, we find that our task-adaptive tokenization approach brings a significant improvement in generation performance while using up to 60% fewer tokens. Preliminary experiments point to promising results when using our tokenization approach with very large language models.
    
[^16]: LLMs在理解反向关系中的无效性的调查

    An Investigation of LLMs' Inefficacy in Understanding Converse Relations. (arXiv:2310.05163v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05163](http://arxiv.org/abs/2310.05163)

    本论文调查了LLMs在理解反向关系方面的无效性。作者引入了一个名为ConvRe的新基准，专注于逆向关系。通过两个任务Re2Text和Text2Re，作者评估了LLMs确定关系和相关文本之间匹配能力。实验结果揭示了LLMs在此方面的限制。

    

    大型语言模型（LLMs）在许多形式语言导向的任务中取得了显着的成功，如结构化数据到文本和语义解析。然而，当前的基准大多遵循LLMs的预训练数据的数据分布。因此，一个自然的问题是，LLMs真正理解形式语言的结构化语义吗？本文在特殊情况下，即逆向二进制关系上进行了调查。我们引入了一个名为ConvRe的新基准，专注于逆向关系，其中包含来自知识图谱完成数据集的17个关系和1240个三元组。我们的ConvRE包括两个任务，Re2Text和Text2Re，这些任务被制定为多项选择题，用于评估LLMs确定关系和相关文本之间匹配能力。在评估协议方面，除了不同的提示方法，我们还引入了测试文本和少样本示例文本的变体。我们在三个实验上进行实验。

    Large Language Models (LLMs) have achieved remarkable success in many formal language oriented tasks, such as structural data-to-text and semantic parsing. However current benchmarks mostly follow the data distribution of the pre-training data of LLMs. Therefore, a natural question rises that do LLMs really understand the structured semantics of formal languages. In this paper, we investigate this problem on a special case, converse binary relation. We introduce a new benchmark ConvRe focusing on converse relations, which contains 17 relations and 1240 triples extracted from popular knowledge graph completion datasets. Our ConvRE features two tasks, Re2Text and Text2Re, which are formulated as multi-choice question answering to evaluate LLMs' ability to determine the matching between relations and associated text. For the evaluation protocol, apart from different prompting methods, we further introduce variants to the test text and few-shot example text. We conduct experiments on three
    
[^17]: FinGPT: 在金融数据集中自然语言处理的指令调优基准测试

    FinGPT: Instruction Tuning Benchmark for Open-Source Large Language Models in Financial Datasets. (arXiv:2310.04793v1 [cs.CL])

    [http://arxiv.org/abs/2310.04793](http://arxiv.org/abs/2310.04793)

    本文介绍了一种针对金融环境下开源语言模型的指令调优方法，并提出了端到端训练和测试的基准测试方案，以加强模型在金融数据集上的专业能力。

    

    在自然语言处理领域迅速扩展的背景下，GPT模型在金融领域的潜力日益明显。然而，将这些模型与金融数据集集成在一起存在挑战，特别是在确定其熟练程度和相关性方面。本文介绍了一种基于指令调优范式的独特方法，专门用于金融环境下的开源大型语言模型。通过这种方法，我们利用开源模型的互操作性，确保了无缝透明的集成。我们首先解释了指令调优范式，并强调其对于立即集成的有效性。本文提出了一个用于端到端训练和测试的基准测试方案，采用成本效益的逐步推进。首先，我们评估了基本的能力和基础任务，比如命名实体识别（NER）和情感分析，以增强专业化。

    In the swiftly expanding domain of Natural Language Processing (NLP), the potential of GPT-based models for the financial sector is increasingly evident. However, the integration of these models with financial datasets presents challenges, notably in determining their adeptness and relevance. This paper introduces a distinctive approach anchored in the Instruction Tuning paradigm for open-source large language models, specifically adapted for financial contexts. Through this methodology, we capitalize on the interoperability of open-source models, ensuring a seamless and transparent integration. We begin by explaining the Instruction Tuning paradigm, highlighting its effectiveness for immediate integration. The paper presents a benchmarking scheme designed for end-to-end training and testing, employing a cost-effective progression. Firstly, we assess basic competencies and fundamental tasks, such as Named Entity Recognition (NER) and sentiment analysis to enhance specialization. Next, 
    
[^18]: 用于在真实世界在线服务中分类有偏见言论的大规模韩文文本数据集

    Large-Scale Korean Text Dataset for Classifying Biased Speech in Real-World Online Services. (arXiv:2310.04313v1 [cs.CL])

    [http://arxiv.org/abs/2310.04313](http://arxiv.org/abs/2310.04313)

    该论文介绍了一个大规模韩文文本数据集，用于分类有偏见言论。通过使用最先进的语言模型，该方法在多项分类任务中实现了超越人类水平的准确性。

    

    随着在线服务的增长，对高级文本分类算法（如情感分析和有偏文本检测）的需求越来越明显。在线服务的匿名性常常导致有偏见和有害言语的存在，对维护在线社区的健康带来挑战。这种现象在韩国尤其相关，目前尚未广泛研究大规模仇恨言论检测算法。在本文中，我们介绍了一个新的综合、大规模数据集，该数据集是从一个知名的韩国社交网络平台收集而来的。我们提供了包括(1)偏好、(2)低俗语言和(3)九种偏见类型的文本样本的注释，使得能够同时对用户生成的文本进行多任务学习的分类。利用最先进的基于BERT的语言模型，我们的方法在各种指标下超过了人类水平的准确性。

    With the growth of online services, the need for advanced text classification algorithms, such as sentiment analysis and biased text detection, has become increasingly evident. The anonymous nature of online services often leads to the presence of biased and harmful language, posing challenges to maintaining the health of online communities. This phenomenon is especially relevant in South Korea, where large-scale hate speech detection algorithms have not yet been broadly explored. In this paper, we introduce a new comprehensive, large-scale dataset collected from a well-known South Korean SNS platform. Our proposed dataset provides annotations including (1) Preferences, (2) Profanities, and (3) Nine types of Bias for the text samples, enabling multi-task learning for simultaneous classification of user-generated texts. Leveraging state-of-the-art BERT-based language models, our approach surpasses human-level accuracy across diverse classification tasks, as measured by various metrics. 
    
[^19]: 使用检索增强的生成模型改进数学问答：在可靠性和人类偏好之间的权衡

    Retrieval-augmented Generation to Improve Math Question-Answering: Trade-offs Between Groundedness and Human Preference. (arXiv:2310.03184v1 [cs.CL])

    [http://arxiv.org/abs/2310.03184](http://arxiv.org/abs/2310.03184)

    通过检索增强的生成模型来改进数学问答，在可靠性和人类偏好之间进行权衡

    

    对于中学数学学生来说，与导师进行互动问答是一种有效的学习方式。生成式大型语言模型的灵活性和新兴能力导致人们对自动化部分辅导过程的兴趣增加，包括支持数学概念的概念讨论的互动问答。然而，生成模型对数学问题的回答可能是错误的，或者与教育背景不匹配，例如与学校的课程不一致。检索增强的生成模型是其中一个潜在的解决方案，它通过在生成模型提示中加入经验证的外部知识资源来提高回答质量。在本文中，我们设计了提示来检索并使用高质量的开源数学教科书中的内容，以回答真实学生提出的问题。我们通过进行一项多条件调查来评估这种检索增强的生成模型在中学代数和几何问答中的效果，并发现人类偏好。

    For middle-school math students, interactive question-answering (QA) with tutors is an effective way to learn. The flexibility and emergent capabilities of generative large language models (LLMs) has led to a surge of interest in automating portions of the tutoring process - including interactive QA to support conceptual discussion of mathematical concepts. However, LLM responses to math questions can be incorrect or mismatched to the educational context such as being misaligned with a school's curriculum. One potential solution is retrieval-augmented generation (RAG), which involves incorporating a vetted external knowledge source in the LLM prompt to increase response quality. In this paper, we designed prompts that retrieve and use content from a high-quality open-source math textbook to generate responses to real student questions. We evaluate the efficacy of this RAG system for middle-school algebra and geometry QA by administering a multi-condition survey, finding that humans p
    
[^20]: 用于马克白彦病研究的不同报告结果的比较话题建模

    Comparative Topic Modeling for Determinants of Divergent Report Results Applied to Macular Degeneration Studies. (arXiv:2309.00312v1 [cs.CL])

    [http://arxiv.org/abs/2309.00312](http://arxiv.org/abs/2309.00312)

    本研究提出了一种比较话题建模方法，用于分析马克白彦病研究中存在矛盾结果的报告。通过对比不同话题与显著结果的相关性，找到了与黄斑变性研究中显著结果报告相关的八种化合物。

    

    话题建模和文本挖掘是自然语言处理的子集，适用于进行元分析和系统审查。对于证据综述，上述NLP方法通常用于特定主题的文献搜索或从报告中提取值以自动化SR和MA的关键阶段。相反，本文提出了一种比较话题建模方法，用于分析同一广义研究问题上存在矛盾结果的报告。具体而言，目标是通过根据其比例发生和在显著结果报告中的一致性分布对其进行排名，找到与感兴趣的结果显著相关的话题。该方法在涉及补充营养化合物是否显著有益于黄斑变性(MD)的广泛范围的研究中进行了测试。确定了八种化合物与显著结果报告的特定相关性。

    Topic modeling and text mining are subsets of Natural Language Processing with relevance for conducting meta-analysis (MA) and systematic review (SR). For evidence synthesis, the above NLP methods are conventionally used for topic-specific literature searches or extracting values from reports to automate essential phases of SR and MA. Instead, this work proposes a comparative topic modeling approach to analyze reports of contradictory results on the same general research question. Specifically, the objective is to find topics exhibiting distinct associations with significant results for an outcome of interest by ranking them according to their proportional occurrence and consistency of distribution across reports of significant results. The proposed method was tested on broad-scope studies addressing whether supplemental nutritional compounds significantly benefit macular degeneration (MD). Eight compounds were identified as having a particular association with reports of significant r
    
[^21]: 激活添加: 无需优化即可操纵语言模型

    Activation Addition: Steering Language Models Without Optimization. (arXiv:2308.10248v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.10248](http://arxiv.org/abs/2308.10248)

    这项研究探讨了一种在推理时通过改变激活来预测性地改变语言模型行为的方法，并且相比于传统方法具有更低的计算和实施成本，并且能够保持模型性能。

    

    可靠地控制大型语言模型的行为是一个紧迫的开放性问题。现有的方法包括有监督微调、根据人类反馈进行强化学习、提示工程和引导解码。我们相反，研究了激活工程：在推理时修改激活以可预测地改变模型行为。特别地，我们通过自然语言隐式指定了一个添加的“导向向量”来偏置前向传播。与以前学习这些导向向量的工作不同，我们的激活添加（ActAdd）方法通过计算来自提示对的激活差异来计算它们。我们在OpenWebText和ConceptNet上展示了ActAdd在GPT-2上的应用。我们的推理时方法控制了输出的高级属性并保持了非目标模型的性能。它所需的计算和实施工作比微调要少得多，允许用户提供自然语言的规范，并且其开销与模型规模自然地扩展。

    Reliably controlling the behavior of large language models is a pressing open problem. Existing methods include supervised finetuning, reinforcement learning from human feedback, prompt engineering, and guided decoding. We instead investigate activation engineering: modifying activations at inference time to predictably alter model behavior. In particular, we bias the forward pass with an added 'steering vector' implicitly specified through natural language.  Unlike past work which learned these steering vectors, our Activation Addition (ActAdd) method computes them by taking the activation differences that result from pairs of prompts. We demonstrate ActAdd on GPT-2 on OpenWebText and ConceptNet. Our inference-time approach yields control over high-level properties of output and preserves off-target model performance. It involves far less compute and implementation effort than finetuning, allows users to provide natural language specifications, and its overhead scales naturally with m
    
[^22]: LLM中的时间旅行：追踪大型语言模型中的数据污染

    Time Travel in LLMs: Tracing Data Contamination in Large Language Models. (arXiv:2308.08493v1 [cs.CL])

    [http://arxiv.org/abs/2308.08493](http://arxiv.org/abs/2308.08493)

    该论文提出了一种用于识别大型语言模型（LLMs）中数据污染的简单而有效的方法。通过对随机样本中的单个实例进行分析，以及使用“引导指令”来评估整个数据集分区的污染程度，可以准确地识别污染的实例和分区。

    

    数据污染是指大型语言模型（LLMs）的训练数据中存在来自下游任务的测试数据，这可能是理解LLMs在其他任务上有效性的一个重要问题。我们提出了一种简单而有效的方法来识别LLMs中的数据污染。我们的方法核心是通过识别从小的随机样本中抽取的单个实例中的潜在污染，然后评估整个数据集分区是否受到污染。为了估计单个实例的污染程度，我们使用了“引导指令”：即一个由数据集名称、分区类型和参考实例的初始部分组成的提示，要求LLM完成它。如果LLM的输出与参考实例的后一部分完全或接近匹配，那么该实例被标记为受到污染。为了了解整个分区是否受到污染，我们提出了两个想法。第一个想法是标记一个数据集的分区，该分区中的实例大多数都被判断为受到污染。

    Data contamination, i.e., the presence of test data from downstream tasks in the training data of large language models (LLMs), is a potential major issue in understanding LLMs' effectiveness on other tasks. We propose a straightforward yet effective method for identifying data contamination within LLMs. At its core, our approach starts by identifying potential contamination in individual instances that are drawn from a small random sample; using this information, our approach then assesses if an entire dataset partition is contaminated. To estimate contamination of individual instances, we employ "guided instruction:" a prompt consisting of the dataset name, partition type, and the initial segment of a reference instance, asking the LLM to complete it. An instance is flagged as contaminated if the LLM's output either exactly or closely matches the latter segment of the reference. To understand if an entire partition is contaminated, we propose two ideas. The first idea marks a dataset
    
[^23]: 从合成语料库和形式逻辑学习演绎推理

    Learning Deductive Reasoning from Synthetic Corpus based on Formal Logic. (arXiv:2308.07336v1 [cs.AI])

    [http://arxiv.org/abs/2308.07336](http://arxiv.org/abs/2308.07336)

    本研究研究了一种从合成语料库中学习演绎推理能力的方法，通过采用基于形式逻辑理论的演绎规则，训练的语言模型具有更泛化的推理能力。

    

    我们研究了一种从合成语料库中学习演绎推理能力的语言模型（LMs）方法。之前的研究使用了具体的演绎规则来生成演绎示例，但这些规则受限或者是任意的。这可能限制了所获得演绎推理能力的泛化能力。我们重新思考并采用基于形式逻辑理论的一组良好基础的演绎规则，当这些规则以多步方式组合时，可以推导出任何其他演绎规则。我们通过实验证明，在提出的语料库上训练的LMs，即$\textbf{FLD}$（$\textbf{F}$ormal $\textbf{L}$ogic $\textbf{D}$eduction），获得了更具泛化性的演绎推理能力。此外，我们确定了演绎推理语料库可以增强LMs的推理能力的方面，以及不同方面无法增强的方面。最后，基于这些结果，我们讨论了将演绎语料库或其他方法应用于每个方面的未来方向。

    We study a synthetic corpus-based approach for language models (LMs) to acquire logical deductive reasoning ability. The previous studies generated deduction examples using specific sets of deduction rules. However, these rules were limited or otherwise arbitrary. This can limit the generalizability of acquired deductive reasoning ability. We rethink this and adopt a well-grounded set of deduction rules based on formal logic theory, which can derive any other deduction rules when combined in a multistep way. We empirically verify that LMs trained on the proposed corpora, which we name $\textbf{FLD}$ ($\textbf{F}$ormal $\textbf{L}$ogic $\textbf{D}$eduction), acquire more generalizable deductive reasoning ability. Furthermore, we identify the aspects of deductive reasoning ability on which deduction corpora can enhance LMs and those on which they cannot. Finally, on the basis of these results, we discuss the future directions for applying deduction corpora or other approaches for each as
    
[^24]: 模仿游戏：在大型语言模型时代检测人类和AI生成的文本

    The Imitation Game: Detecting Human and AI-Generated Texts in the Era of Large Language Models. (arXiv:2307.12166v1 [cs.CL])

    [http://arxiv.org/abs/2307.12166](http://arxiv.org/abs/2307.12166)

    本论文研究了区分人类和AI生成的文本的任务，在不同体裁下进行了比较研究，提出了一个新的数据集，并采用多种机器学习模型进行分类。结果表明这些模型对于区分人类和AI生成的文本具有很高的效力，尽管在区分GPT生成的文本方面存在一定挑战。

    

    基于人工智能的大型语言模型（LLM）具有革新教育、研究和实践的巨大潜力。然而，区分人类写作和AI生成的文本已经成为一项重要任务。本文介绍了一项比较研究，提出了一个新颖的数据集，包含不同体裁的人类写作和LLM生成的文本：论文、故事、诗歌和Python代码。我们采用了几种机器学习模型来对这些文本进行分类。结果表明，尽管数据集的样本数量有限，这些模型在区分人类和AI生成的文本方面表现出了很高的效力。然而，当分类GPT生成的文本时，任务变得更具挑战性，特别是在故事写作方面。结果表明，与更复杂的多类别任务相比，这些模型在二元分类任务（如区分人类生成文本和特定LLM）方面表现出了更优越的性能。

    The potential of artificial intelligence (AI)-based large language models (LLMs) holds considerable promise in revolutionizing education, research, and practice. However, distinguishing between human-written and AI-generated text has become a significant task. This paper presents a comparative study, introducing a novel dataset of human-written and LLM-generated texts in different genres: essays, stories, poetry, and Python code. We employ several machine learning models to classify the texts. Results demonstrate the efficacy of these models in discerning between human and AI-generated text, despite the dataset's limited sample size. However, the task becomes more challenging when classifying GPT-generated text, particularly in story writing. The results indicate that the models exhibit superior performance in binary classification tasks, such as distinguishing human-generated text from a specific LLM, compared to the more complex multiclass tasks that involve discerning among human-ge
    
[^25]: AutoAlign：基于大型语言模型的全自动有效知识图谱对齐方法

    AutoAlign: Fully Automatic and Effective Knowledge Graph Alignment enabled by Large Language Models. (arXiv:2307.11772v1 [cs.IR])

    [http://arxiv.org/abs/2307.11772](http://arxiv.org/abs/2307.11772)

    AutoAlign是一种全自动的知识图谱对齐方法，不需要手工制作的种子对齐。它利用大型语言模型自动捕捉谓词相似性，并使用TransE计算实体嵌入来实现实体对齐。

    

    知识图谱间的实体对齐任务旨在识别出两个不同知识图谱中表示相同实体的每对实体。许多基于机器学习的方法已被提出用于这个任务。然而，据我们所知，现有的方法都需要手工制作的种子对齐，这是非常昂贵的。在本文中，我们提出了第一个名为AutoAlign的完全自动对齐方法，它不需要任何手工制作的种子对齐。具体而言，对于谓词嵌入，AutoAlign使用大型语言模型构建谓词近邻图，自动捕捉两个知识图谱中谓词的相似性。对于实体嵌入，AutoAlign首先使用TransE独立计算每个知识图谱的实体嵌入，然后通过计算基于实体属性的实体相似性，将两个知识图谱的实体嵌入移动到相同的向量空间中。因此，AutoAlign实现了谓词对齐和实体对齐。

    The task of entity alignment between knowledge graphs (KGs) aims to identify every pair of entities from two different KGs that represent the same entity. Many machine learning-based methods have been proposed for this task. However, to our best knowledge, existing methods all require manually crafted seed alignments, which are expensive to obtain. In this paper, we propose the first fully automatic alignment method named AutoAlign, which does not require any manually crafted seed alignments. Specifically, for predicate embeddings, AutoAlign constructs a predicate-proximity-graph with the help of large language models to automatically capture the similarity between predicates across two KGs. For entity embeddings, AutoAlign first computes the entity embeddings of each KG independently using TransE, and then shifts the two KGs' entity embeddings into the same vector space by computing the similarity between entities based on their attributes. Thus, both predicate alignment and entity al
    
[^26]: EmotionPrompt: 通过情感刺激提升大型语言模型的关键心理学方法

    EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus. (arXiv:2307.11760v1 [cs.CL])

    [http://arxiv.org/abs/2307.11760](http://arxiv.org/abs/2307.11760)

    EmotionPrompt是一个基于心理学的方法，通过将情感刺激融入到提示中，提升了大型语言模型在各项任务上的性能，并且同时改善了其真实性和信息量。

    

    大型语言模型（LLMs）在推理、语言理解和数学问题解决等许多领域取得了显著的性能，并被视为人工通用智能（AGI）的关键步骤。然而，LLMs对提示的敏感性仍然是其日常应用的主要瓶颈。本文从心理学中汲取灵感，提出了EmotionPrompt来探索情感智能以提升LLMs的性能。EmotionPrompt基于一个非常简单明了的原则：将情感刺激融入到提示中。实验结果表明，我们的方法在相同的单一提示模板上，与原始的零样本提示和Zero-shot-CoT相比，在8个任务上都显著优于多种模型：ChatGPT、Vicuna-13b、Bloom和T5。此外，观察到EmotionPrompt能够提高真实性和信息量。我们相信EmotionPrompt为探索跨学科知识开辟了一条新的道路。

    Large language models (LLMs) have achieved significant performance in many fields such as reasoning, language understanding, and math problem-solving, and are regarded as a crucial step to artificial general intelligence (AGI). However, the sensitivity of LLMs to prompts remains a major bottleneck for their daily adoption. In this paper, we take inspiration from psychology and propose EmotionPrompt to explore emotional intelligence to enhance the performance of LLMs. EmotionPrompt operates on a remarkably straightforward principle: the incorporation of emotional stimulus into prompts. Experimental results demonstrate that our \method, using the same single prompt templates, significantly outperforms original zero-shot prompt and Zero-shot-CoT on 8 tasks with diverse models: ChatGPT, Vicuna-13b, Bloom, and T5. Further, EmotionPrompt was observed to improve both truthfulness and informativeness. We believe that EmotionPrompt heralds a novel avenue for exploring interdisciplinary knowledg
    
[^27]: 对生物医学自然语言处理中的联邦学习进行系统评估

    A Systematic Evaluation of Federated Learning on Biomedical Natural Language Processing. (arXiv:2307.11254v1 [cs.CL])

    [http://arxiv.org/abs/2307.11254](http://arxiv.org/abs/2307.11254)

    本研究对医学领域中的联邦学习在生物医学自然语言处理中的应用进行了系统评估，结果显示联邦学习模型优于单独训练的模型，并且在考虑数据隐私的情况下仍能取得良好的效果。

    

    语言模型（LM）如BERT和GPT已经改变了自然语言处理（NLP）。然而，隐私敏感的领域，特别是医疗领域，由于有限的数据访问和由《健康保险便携性和责任法案》（HIPPA）和《通用数据保护条例》（GDPR）等法规的隐私约束，面临着训练LM的挑战。联邦学习（FL）提供了一种分散的解决方案，既能够实现协同学习，又能够确保数据隐私的保护。在本研究中，我们对医学中的FL进行了系统评估，涵盖了六个生物医学NLP任务，使用了八个语料库和六个LM。我们的结果表明：1）FL模型始终优于单个客户端数据训练的LM，并且有时能够与使用汇总数据训练的模型匹配；2）在总数据量固定的情况下，使用更多客户端进行FL训练的LM表现出较差的性能，但基于预训练的转换器模型表现出更强的鲁棒性；3）LM们

    Language models (LMs) like BERT and GPT have revolutionized natural language processing (NLP). However, privacy-sensitive domains, particularly the medical field, face challenges to train LMs due to limited data access and privacy constraints imposed by regulations like the Health Insurance Portability and Accountability Act (HIPPA) and the General Data Protection Regulation (GDPR). Federated learning (FL) offers a decentralized solution that enables collaborative learning while ensuring the preservation of data privacy. In this study, we systematically evaluate FL in medicine across $2$ biomedical NLP tasks using $6$ LMs encompassing $8$ corpora. Our results showed that: 1) FL models consistently outperform LMs trained on individual client's data and sometimes match the model trained with polled data; 2) With the fixed number of total data, LMs trained using FL with more clients exhibit inferior performance, but pre-trained transformer-based models exhibited greater resilience. 3) LMs
    
[^28]: 没有训练就没有收益：重新审视基于Transformer的语言模型的高效训练算法

    No Train No Gain: Revisiting Efficient Training Algorithms For Transformer-based Language Models. (arXiv:2307.06440v1 [cs.LG])

    [http://arxiv.org/abs/2307.06440](http://arxiv.org/abs/2307.06440)

    本论文重新审视了基于Transformer的语言模型的高效训练算法，包括动态架构，批量选择和高效优化器。然而，在使用这些算法预训练时，相对于基线方法，它们的训练、验证和下游收益消失了。同时，论文提出了一个评估协议来进行计算，并释放了代码来促进高效训练的研究。

    

    近年来，训练Transformer-based语言模型所需的计算量急剧增加。这一趋势促使研究者们开展了针对高效训练算法的研究，旨在比标准训练更快地改善训练、验证和下游性能。在这项工作中，我们重新审视了三类这样的算法：动态架构（层叠、层丢弃）、批量选择（选择性反向传播、RHO损失）和高效优化器（Lion、Sophia）。当使用这些方法在固定计算预算下对BERT和T5进行预训练时，我们发现它们的训练、验证和下游收益相对于一个具有完全衰减学习率的基线而言会消失。我们定义了一个评估协议，可以通过将所有计算时间映射到一个称为参考系统时间的参考机器上，在任意机器上进行计算。我们讨论了我们提出的协议的局限性，并发布了我们的代码，以鼓励对高效训练的严格研究。

    The computation necessary for training Transformer-based language models has skyrocketed in recent years. This trend has motivated research on efficient training algorithms designed to improve training, validation, and downstream performance faster than standard training. In this work, we revisit three categories of such algorithms: dynamic architectures (layer stacking, layer dropping), batch selection (selective backprop, RHO loss), and efficient optimizers (Lion, Sophia). When pre-training BERT and T5 with a fixed computation budget using such methods, we find that their training, validation, and downstream gains vanish compared to a baseline with a fully-decayed learning rate. We define an evaluation protocol that enables computation to be done on arbitrary machines by mapping all computation time to a reference machine which we call reference system time. We discuss the limitations of our proposed protocol and release our code to encourage rigorous research in efficient training p
    
[^29]: 风格胜过实质：大型语言模型的评估偏见

    Style Over Substance: Evaluation Biases for Large Language Models. (arXiv:2307.03025v1 [cs.CL])

    [http://arxiv.org/abs/2307.03025](http://arxiv.org/abs/2307.03025)

    这项研究调查了人类和基于大型语言模型的评委在比较不同模型输出时的行为，并发现评估过程中存在偏见，即尽管包含事实错误，答案仍然被更高地评分。为了解决这个问题，我们提出了

    

    随着大型语言模型（LLMs）的不断进步，准确和全面评估它们的性能变得越来越具有挑战性。传统上，人类评估被认为是自然语言生成的黄金标准。最近的进展将最先进的LLMs纳入评估过程中，作为人类评委的代理。然而，人类和LLMs作为评估者的能力程度仍然不确定。本研究旨在研究众包人类评委和基于LLMs的评委在比较不同模型的输出时的行为。为了实现这一目标，我们收集了一个包含故意有缺陷的机器生成答案的数据集。我们的研究结果表明，尽管事实上的错误可能带来更大的危险，但带有事实错误的答案仍然比长度过短或包含语法错误的答案评分更高。这突显了评估过程中存在的令人担忧的偏见。为了解决这个问题，我们提出了

    As large language models (LLMs) continue to advance, accurately and comprehensively evaluating their performance becomes increasingly challenging. Conventionally, human evaluations are considered the gold standard in natural language generation. Recent advancements incorporate state-of-the-art LLMs as proxies for human judges in evaluation processes. Nonetheless, the extent to which humans and LLMs are capable evaluators remains uncertain. This study aims to investigate the behavior of both crowd-sourced human and LLM-based judges when comparing outputs from different models. To accomplish this, we curate a dataset comprising intentionally flawed machine-generated answers. Our findings indicate that despite the potentially greater danger posed by factual errors, answers with factual errors were still rated more favorably compared to answers that were too short or contained grammatical errors. This highlights a concerning bias in the evaluation process. To address this issue, we propose
    
[^30]: 学会生成比你的LMM更好的文本

    Learning to Generate Better Than Your LLM. (arXiv:2306.11816v1 [cs.LG])

    [http://arxiv.org/abs/2306.11816](http://arxiv.org/abs/2306.11816)

    本论文研究了基于强化学习算法 RLGF，用于在 GPT-3 等动态黑匣子指导下微调大型语言模型 LLM 的条件文本生成，相比通用 RL 算法，该算法在 IMDB 和 CommonGen 任务中表现更好。

    

    强化学习(RL)已经成为一种强大的范例，用于优化大型语言模型 (LLM) 条件文本生成。特别地，最近的LLM，如ChatGPT和GPT - 4能够与用户进行流畅的对话，并融合了RL和人类反馈。本研究受到学习搜索算法的启发，并利用文本生成的关键特性，探索了超出通用RL算法如PPO之外的强化学习算法。特别地，我们扩展了RL算法，使其能够与动态黑匣子的指导LLM如GPT-3进行交互，并提出了具有引导反馈的RL(RLGF)，这是一套用于LLM微调的RL算法。我们在GRUE基准测试的IMDB正向评论和CommonGen文本生成任务上进行了实验。我们展示了我们的RL算法比监督学习(SL)和默认PPO基线表现更高，证明了与指导LLM互动的好处。

    Reinforcement learning (RL) has emerged as a powerful paradigm for fine-tuning Large Language Models (LLMs) for conditional text generation. In particular, recent LLMs such as ChatGPT and GPT-4 can engage in fluent conversations with users by incorporating RL and feedback from humans. Inspired by learning-to-search algorithms and capitalizing on key properties of text generation, we seek to investigate reinforcement learning algorithms beyond general purpose algorithms such as Proximal policy optimization (PPO). In particular, we extend RL algorithms to allow them to interact with a dynamic black-box guide LLM such as GPT-3 and propose RL with guided feedback (RLGF), a suite of RL algorithms for LLM fine-tuning. We experiment on the IMDB positive review and CommonGen text generation task from the GRUE benchmark. We show that our RL algorithms achieve higher performance than supervised learning (SL) and default PPO baselines, demonstrating the benefit of interaction with the guide LLM. 
    
[^31]: 源代码模型的数据增强方法：一份综述

    Data Augmentation Approaches for Source Code Models: A Survey. (arXiv:2305.19915v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.19915](http://arxiv.org/abs/2305.19915)

    本文对源代码的数据增强技术进行了全面的调查和综述，介绍了它们的分类法、优化策略和性能结果，并讨论了未来方向和研究挑战。

    

    源代码在许多关键任务中的广泛应用促进了数据增强（DA）技术的发展，以增强训练数据并提高这些模型的各种能力（例如健壮性和可泛化性）。虽然已经提出并针对源代码模型进行了一系列DA方法的调整，但缺乏综合性的调查和审查以理解它们的有效性和含义。本文通过对源代码的数据增强进行全面而综合的调查，填补这一空白，我们系统地整理和概述现有文献，以提供该领域的全面概述。我们首先构建了适用于源代码模型的数据增强的分类法，然后讨论了著名的、方法上具有说明性的方法。接下来，我们强调了优化DA质量的一般策略和技术。随后，我们强调了在被广泛接受的基准测试中发挥作用的技术，并呈现了它们的性能结果。最后，我们讨论了DA用于源代码模型的潜在未来方向和开放研究挑战。

    The increasingly popular adoption of source code in many critical tasks motivates the development of data augmentation (DA) techniques to enhance training data and improve various capabilities (e.g., robustness and generalizability) of these models. Although a series of DA methods have been proposed and tailored for source code models, there lacks a comprehensive survey and examination to understand their effectiveness and implications. This paper fills this gap by conducting a comprehensive and integrative survey of data augmentation for source code, wherein we systematically compile and encapsulate existing literature to provide a comprehensive overview of the field. We start by constructing a taxonomy of DA for source code models model approaches, followed by a discussion on prominent, methodologically illustrative approaches. Next, we highlight the general strategies and techniques to optimize the DA quality. Subsequently, we underscore techniques that find utility in widely-accept
    
[^32]: 基于目标的可解释聚类在语言描述中的应用

    Goal-Driven Explainable Clustering via Language Descriptions. (arXiv:2305.13749v1 [cs.CL])

    [http://arxiv.org/abs/2305.13749](http://arxiv.org/abs/2305.13749)

    该研究提出了一种“带解释的基于目标的聚类”（GoalEx）的新任务形式，它将目标和解释都表示为自由形式的语言描述。通过将摘要系统的注释进行分类来说明研究的有效性以及生成的解释。

    

    无监督聚类广泛用于探索大型语料库，但现有表述既不考虑用户的目标，也不解释聚类的含义。我们提出了一个新的任务形式——带解释的基于目标的聚类（GoalEx），它将目标和解释都表示为自由形式的语言描述。对于一个总结系统所犯的错误进行分类，GoalEx的输入是一个注释者为系统生成的摘要撰写的注释语料库和目标描述“根据注释者认为摘要不完美的原因对注释进行分类”;输出是每个具有解释的文本聚类(“此聚类提到摘要缺少重要的上下文信息。“)，这些聚类与目标相关，并准确解释哪些注释应该(不应该)属于一个聚类。为了解决GoalEx，我们使用一个语言模型提示“ [数据集子集]+[目标]+头脑风暴一个代表聚类的解释列表”，然后分类哪些解释属于每个聚类。实验在五个数据集上进行，包括汇总反馈、新闻文章、维基百科页面、科学文章和批评评论，展示了我们方法的有效性和生成的解释。

    Unsupervised clustering is widely used to explore large corpora, but existing formulations neither consider the users' goals nor explain clusters' meanings. We propose a new task formulation, "Goal-Driven Clustering with Explanations" (GoalEx), which represents both the goal and the explanations as free-form language descriptions. For example, to categorize the errors made by a summarization system, the input to GoalEx is a corpus of annotator-written comments for system-generated summaries and a goal description "cluster the comments based on why the annotators think the summary is imperfect.''; the outputs are text clusters each with an explanation ("this cluster mentions that the summary misses important context information."), which relates to the goal and precisely explain which comments should (not) belong to a cluster. To tackle GoalEx, we prompt a language model with "[corpus subset] + [goal] + Brainstorm a list of explanations each representing a cluster."; then we classify wh
    
[^33]: 跨模态注意力不足：基于不协调感知的多模态情感分析与识别

    Cross-Attention is Not Enough: Incongruity-Aware Multimodal Sentiment Analysis and Emotion Recognition. (arXiv:2305.13583v1 [cs.CL])

    [http://arxiv.org/abs/2305.13583](http://arxiv.org/abs/2305.13583)

    本文提出了一种基于不协调感知的跨模态情感分析方法，通过Hierarchical Crossmodal Transformer with Modality Gating(HCT-MG)模型来确定主要模态并分层融合辅助模态，有效减轻模态之间的不协调感知和信息冗余问题。

    

    多模态融合在情感计算任务中的应用对性能的提升已被证明是有效的。然而，多模态融合的机理尚不清楚，在现实世界中使用它通常会导致大型模型的问题。本文在情感分析的基础上，首先分析了跨模态注意力中一个模态中突出的情感信息如何受到另一个模态的影响。我们发现，由于跨模态的关注，模态之间存在潜在的不协调感知。基于这一发现，我们提出了一种轻量级模型(HCT-MG)，该模型通过分层交叉模态Transformer与模态门控制来确定主要的模态，并分层地将辅助模态纳入其中，以减轻模态之间的不协调感知并减少信息冗余。在三个基准数据集CMU-MOSI、CMU-MOSEI和IEMOCAP上的实验评估验证了我们方法的有效性，表明：1）其优于当前最先进的多模态模型；2）它仅使用少量的超参数和参数；3）它的计算成本较低。

    Fusing multiple modalities for affective computing tasks has proven effective for performance improvement. However, how multimodal fusion works is not well understood, and its use in the real world usually results in large model sizes. In this work, on sentiment and emotion analysis, we first analyze how the salient affective information in one modality can be affected by the other in crossmodal attention. We find that inter-modal incongruity exists at the latent level due to crossmodal attention. Based on this finding, we propose a lightweight model via Hierarchical Crossmodal Transformer with Modality Gating (HCT-MG), which determines a primary modality according to its contribution to the target task and then hierarchically incorporates auxiliary modalities to alleviate inter-modal incongruity and reduce information redundancy. The experimental evaluation on three benchmark datasets: CMU-MOSI, CMU-MOSEI, and IEMOCAP verifies the efficacy of our approach, showing that it: 1) outperfo
    
[^34]: 无需转移数据的多语言短语标记方法

    Transfer-Free Data-Efficient Multilingual Slot Labeling. (arXiv:2305.13528v1 [cs.CL])

    [http://arxiv.org/abs/2305.13528](http://arxiv.org/abs/2305.13528)

    本论文提出了一种无需英文数据的多语言数据高效标记方法，结果显示其比跨语言转移基准显着提高（最多提高22%）。

    

    短语标记（SL）是任务导向型对话（ToD）系统的核心组件，而其中的短语和相应的值通常是特定于语言、任务和领域的。因此，将系统扩展到任何新的语言-领域-任务配置需要重新运行昂贵而资源密集型的数据标注流程。为了减轻固有的数据稀缺问题，当前多语言ToD研究假设特定任务和领域的足够英语注释数据始终可用，因此在标准的跨语言传输设置中运行。在这项工作中，我们摆脱这种常常不现实的假设。我们研究挑战性场景，即无法保证具有传输功能的英文注释数据，并专注于在目标语言中直接进行多语言数据高效的标记。我们提出了一种两阶段短语标记方法（称为TWOSL），将标准的多语言SL转化为无需转移数据的高效设置。在第一阶段，我们利用一个小型平行语料库来对齐不同语言之间的短语集，并利用这种对齐来通过一个无监督的多语言短语感应框架从高资源语言传递注释。在第二阶段，我们应用主动学习来通过包含来自目标语言的少量监督数据来迭代更新和改进多语言短语分类器。实验结果表明，在多个语言和领域中，TWOSL比最先进的跨语言转移基线显着提高了短语F1分数（最多提高22%）。

    Slot labeling (SL) is a core component of task-oriented dialogue (ToD) systems, where slots and corresponding values are usually language-, task- and domain-specific. Therefore, extending the system to any new language-domain-task configuration requires (re)running an expensive and resource-intensive data annotation process. To mitigate the inherent data scarcity issue, current research on multilingual ToD assumes that sufficient English-language annotated data are always available for particular tasks and domains, and thus operates in a standard cross-lingual transfer setup. In this work, we depart from this often unrealistic assumption. We examine challenging scenarios where such transfer-enabling English annotated data cannot be guaranteed, and focus on bootstrapping multilingual data-efficient slot labelers in transfer-free scenarios directly in the target languages without any English-ready data. We propose a two-stage slot labeling approach (termed TWOSL) which transforms standar
    
[^35]: MixPro：基于提示学习的简单有效的数据增强方式

    MixPro: Simple yet Effective Data Augmentation for Prompt-based Learning. (arXiv:2304.09402v1 [cs.CL])

    [http://arxiv.org/abs/2304.09402](http://arxiv.org/abs/2304.09402)

    MixPro是一种数据增强方法，通过对原始输入和模板进行混合来提高基于提示的学习性能，平均提高了5.08%的模型性能。

    

    基于提示的学习通过将输入与模板组合起来，将下游任务重构为填空问题。这种技术在少样本学习中特别有用，然而，使用有限的模板和文本仍然存在显着的性能改进空间。此外，现有的使用模型集成的方法可以限制模型的效率。为解决这些问题，我们提出了一种称为MixPro的增强方法，它通过标记级、句子级和时代级的混合策略来增强原始输入文本和模板。我们在五个少样本数据集上进行了实验，结果表明MixPro优于其他增强基线，相比增强前，平均提高了5.08%的模型性能。

    Prompt-based learning reformulates downstream tasks as cloze problems by combining the original input with a template. This technique is particularly useful in few-shot learning, where a model is trained on a limited amount of data. However, the limited templates and text used in few-shot prompt-based learning still leave significant room for performance improvement. Additionally, existing methods using model ensembles can constrain the model efficiency. To address these issues, we propose an augmentation method called MixPro, which augments both the vanilla input text and the templates through token-level, sentence-level, and epoch-level Mixup strategies. We conduct experiments on five few-shot datasets, and the results show that MixPro outperforms other augmentation baselines, improving model performance by an average of 5.08% compared to before augmentation.
    
[^36]: 更大的探针讲述不同的故事: 通过上下文学习扩展心理语言学数据集

    Larger Probes Tell a Different Story: Extending Psycholinguistic Datasets Via In-Context Learning. (arXiv:2303.16445v1 [cs.CL])

    [http://arxiv.org/abs/2303.16445](http://arxiv.org/abs/2303.16445)

    本文通过上下文学习扩展否定和角色反转数据集，发现过去的结论可能被小型测试集误导。同时，BERT和ALBERT等模型表现出较高的否定敏感度。

    

    语言模型探测通常用来测试这些模型的特定能力。然而，当探测基准小且缺乏统计功效时，这类研究的结论可能受到限制。在这项工作中，我们介绍了受心理语言学研究启发的否定（NEG-1500-SIMP）和角色反转（ROLE-1500）的新的、更大的数据集。我们使用GPT3将现有的NEG-136和ROLE-88基准进行了大幅扩展，将它们的规模从18和44个句对分别增加到了750个。我们还创建了另一个使用基于模板的生成创建的扩展否定数据集(NEG-1500-SIMP-TEMP)，它由770个句对组成。我们在扩展数据集上评估了22个模型，发现模型性能与原始较小基准相比下降了20-57%。我们观察到BERT和ALBERT等模型具有较高的否定敏感性，这表明以前的研究结果可能由于较小的测试集而存在误差。最后，我们观察到，虽然GPT3生成了所有的实例，但句子的语法质量受到一些限制。

    Language model probing is often used to test specific capabilities of these models. However, conclusions from such studies may be limited when the probing benchmarks are small and lack statistical power. In this work, we introduce new, larger datasets for negation (NEG-1500-SIMP) and role reversal (ROLE-1500) inspired by psycholinguistic studies. We dramatically extend existing NEG-136 and ROLE-88 benchmarks using GPT3, increasing their size from 18 and 44 sentence pairs to 750 each. We also create another version of extended negation dataset (NEG-1500-SIMP-TEMP), created using template-based generation. It consists of 770 sentence pairs. We evaluate 22 models on the extended datasets, seeing model performance dip 20-57% compared to the original smaller benchmarks. We observe high levels of negation sensitivity in models like BERT and ALBERT demonstrating that previous findings might have been skewed due to smaller test sets. Finally, we observe that while GPT3 has generated all the ex
    
[^37]: 从宽到深：维度提升网络用于参数高效的知识图谱嵌入

    From Wide to Deep: Dimension Lifting Network for Parameter-efficient Knowledge Graph Embedding. (arXiv:2303.12816v1 [cs.LG])

    [http://arxiv.org/abs/2303.12816](http://arxiv.org/abs/2303.12816)

    本文提出了一个用于实现参数高效的知识图谱嵌入的深度网络，通过增加深度克服因采用低维实体表示而导致的模型精度下降和模型参数减少有限的问题。

    

    知识图谱嵌入（KGE）将实体和关系映射到向量表示对于下游任务非常重要。传统的KGE方法需要相对高维的实体表示来保留知识图谱的结构信息，但会导致庞大的模型参数。最近的方法通过采用低维实体表示来降低模型参数，同时开发技术（例如知识蒸馏）来补偿降维。然而，这样的操作会导致模型精度下降和模型参数减少有限。具体来说，我们将所有实体表示的级联视为嵌入层，那么采用高维实体表示的传统KGE方法等同于扩展嵌入层的宽度以获得表现力。为了在不牺牲准确度的情况下实现参数效率，我们相反地增加深度，并提出一个更深的实体嵌入网络。

    Knowledge graph embedding (KGE) that maps entities and relations into vector representations is essential for downstream tasks. Conventional KGE methods require relatively high-dimensional entity representations to preserve the structural information of knowledge graph, but lead to oversized model parameters. Recent methods reduce model parameters by adopting low-dimensional entity representations, while developing techniques (e.g., knowledge distillation) to compensate for the reduced dimension. However, such operations produce degraded model accuracy and limited reduction of model parameters. Specifically, we view the concatenation of all entity representations as an embedding layer, and then conventional KGE methods that adopt high-dimensional entity representations equal to enlarging the width of the embedding layer to gain expressiveness. To achieve parameter efficiency without sacrificing accuracy, we instead increase the depth and propose a deeper embedding network for entity re
    
[^38]: 关于正则表达式进程语义的双模折叠不封闭性问题

    The Image of the Process Interpretation of Regular Expressions is Not Closed under Bisimulation Collapse. (arXiv:2303.08553v1 [cs.LO])

    [http://arxiv.org/abs/2303.08553](http://arxiv.org/abs/2303.08553)

    论文探讨了一种存在于正则表达式进程语义中的双模折叠不封闭性问题，并在1-free正则表达式的解释中发现了对这种难题的关键原因，进一步提出了LEE属性的特征，证明了1-free正则表达式的方程证明系统是完备的，并且多项式时间可以解决解释和过程图双模相似性问题。

    

    Milner提出的正则表达式进程语义的公理化和表达问题在考虑死锁0和空步长1的完整表达式类中变得困难起来。我们报告了当0存在时添加1会产生的现象，这将这个困难的关键原因带到了聚焦点。即，虽然1-free正则表达式的解释在双模折叠下是封闭的，但任意正则表达式的解释不是这种情况。1-free正则表达式的过程图解释满足循环存在和消除属性LEE，该属性在双模折叠下得以保留。LEE的这些特征被用于证明1-free正则表达式的方程证明系统是完备的，并且用于判断一个1-free正则表达式的解释是否与一个过程图双模相似是一个多项式时间可判定的问题。

    Axiomatization and expressibility problems for Milner's process semantics (1984) of regular expressions modulo bisimilarity have turned out to be difficult for the full class of expressions with deadlock 0 and empty step~1. We report on a phenomenon that arises from the added presence of 1 when 0 is available, and that brings a crucial reason for this difficulty into focus. To wit, while interpretations of 1-free regular expressions are closed under bisimulation collapse, this is not the case for the interpretations of arbitrary regular expressions.  Process graph interpretations of 1-free regular expressions satisfy the loop existence and elimination property LEE, which is preserved under bisimulation collapse. These features of LEE were applied for showing that an equational proof system for 1-free regular expressions modulo bisimilarity is complete, and that it is decidable in polynomial time whether a process graph is bisimilar to the interpretation of a 1-free regular expression. 
    
[^39]: 交互式文本生成

    Interactive Text Generation. (arXiv:2303.00908v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.00908](http://arxiv.org/abs/2303.00908)

    本研究提出了一种新的交互式文本生成任务，使用用户模拟器进行交互式训练生成模型，避免了真实用户参与的成本，并通过提供编辑指导模型朝着给定目标前进，从而提高了生成质量。

    

    用户每天都要与文本、图片、代码或其他编辑器互动。然而，机器学习模型很少在反映用户与编辑器之间互动的设置中进行训练。这是可以理解的，因为使用真实用户进行AI模型的训练不仅速度慢且成本高，而且这些模型所学习的内容可能特定于用户界面设计选择。不幸的是，这意味着大多数文本、代码和图像生成的研究都集中在非交互设置上，即模型被期望在没有考虑任何来自愿意帮助的用户输入的情况下完成所有任务。我们引入了一项新的交互式文本生成任务，允许使用用户模拟器进行交互式训练生成模型，无需涉及真实用户的成本，模拟器通过提供编辑指导模型朝着给定的目标文本前进。我们使用模仿学习训练我们的交互式模型，并与具有竞争力的非交互式生成模型进行了实验比较。

    Users interact with text, image, code, or other editors on a daily basis. However, machine learning models are rarely trained in the settings that reflect the interactivity between users and their editor. This is understandable as training AI models with real users is not only slow and costly, but what these models learn may be specific to user interface design choices. Unfortunately, this means most of the research on text, code, and image generation has focused on non-interactive settings, whereby the model is expected to get everything right without accounting for any input from a user who may be willing to help.  We introduce a new Interactive Text Generation task that allows training generation models interactively without the costs of involving real users, by using user simulators that provide edits that guide the model towards a given target text. We train our interactive models using Imitation Learning, and our experiments against competitive non-interactive generation models s
    
[^40]: InstructABSA: 基于指令学习的方面情感分析

    InstructABSA: Instruction Learning for Aspect Based Sentiment Analysis. (arXiv:2302.08624v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.08624](http://arxiv.org/abs/2302.08624)

    InstructABSA是一种使用指令学习范式的方面情感分析方法，能够显著提高Aspect Term Extraction、Aspect Term Sentiment Classification、和Joint Task subtasks三个子任务的性能，并且在多个数据集上表现超过之前的最先进方法。

    

    本文介绍了InstructABSA，一种使用指令学习范式进行Aspect Based Sentiment Analysis (ABSA) 所有子任务（Aspect Term Extraction (ATE)，Aspect Term Sentiment Classification (ATSC)，以及Joint Task modeling）的方法。我们的方法对每个训练样本引入了正面、负面、和中性的例子，并使用指令来调整每个ABSA子任务的模型（Tk-Instruct），从而显著提高了性能。在Sem Eval 2014、2015和2016数据集上的实验结果表明，在所有三个ABSA子任务（ATE、ATSC和Joint Task）上，InstructABSA在性能上都比之前的最先进方法（SOTA）表现出了显著的优势，并且表现超过了7倍大的模型。特别是，在Rest14 ATE子任务上，InstructABSA超过了SOTA 7.31%的得分，Rest15 ATSC子任务上也有提升，并且在Lapt14 Joint Task上的表现提升了8.63%点。我们的结果还表明，对于所有三个子任务，InstructABSA具有强大的新领域泛化能力。

    In this paper, we present InstructABSA, Aspect Based Sentiment Analysis (ABSA) using the instruction learning paradigm for all ABSA subtasks: Aspect Term Extraction (ATE), Aspect Term Sentiment Classification (ATSC), and Joint Task modeling. Our method introduces positive, negative, and neutral examples to each training sample, and instruction tunes the model (Tk-Instruct) for each ABSA subtask, yielding significant performance improvements. Experimental results on the Sem Eval 2014, 15, and 16 datasets demonstrate that InstructABSA outperforms the previous state-of-the-art (SOTA) approaches on all three ABSA subtasks (ATE, ATSC, and Joint Task) by a significant margin, outperforming 7x larger models. In particular, InstructABSA surpasses the SOTA on the Rest14 ATE subtask by 7.31% points, Rest15 ATSC subtask by and on the Lapt14 Joint Task by 8.63% points. Our results also suggest a strong generalization ability to new domains across all three subtasks
    
[^41]: “大型语言模型可能会自发出现心智理论”

    Theory of Mind May Have Spontaneously Emerged in Large Language Models. (arXiv:2302.02083v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.02083](http://arxiv.org/abs/2302.02083)

    “通过测试多个语言模型在解决40个ToM任务上的表现，研究发现GPT-3和GPT-4能够解决大部分任务，说明类似ToM的能力可能是语言模型自发出现的附带产物。”

    

    “心智理论（ToM）指能够推理他人内心的不可观察状态，对于人类社交互动、交流、移情、自我意识和道德至关重要。我们使用了40个广泛用于测试人类ToM的经典虚假信念任务来测试几个语言模型。2020年之前发布的模型在解决ToM任务方面几乎没有能力。然而，2020年5月发布的第一个GPT-3版本（“davinci-001”）解决了约40％的虚假信念任务，与3.5岁的儿童的表现相当。它的第二个版本（“davinci-002”，2022年1月）解决了70％的虚假信念任务，与6岁儿童的表现相当。最新版本的GPT-3.5（“davinci-003”，2022年11月）解决了90％的虚假信念任务，达到了7岁儿童水平。于2023年3月发布的GPT-4解决了几乎所有的任务（95％）。这些发现表明，类似ToM的能力（迄今被认为是人类独有的）可能是语言的附带产物。”

    Theory of mind (ToM), or the ability to impute unobservable mental states to others, is central to human social interactions, communication, empathy, self-consciousness, and morality. We tested several language models using 40 classic false-belief tasks widely used to test ToM in humans. The models published before 2020 showed virtually no ability to solve ToM tasks. Yet, the first version of GPT-3 ("davinci-001"), published in May 2020, solved about 40% of false-belief tasks-performance comparable with 3.5-year-old children. Its second version ("davinci-002"; January 2022) solved 70% of false-belief tasks, performance comparable with six-year-olds. Its most recent version, GPT-3.5 ("davinci-003"; November 2022), solved 90% of false-belief tasks, at the level of seven-year-olds. GPT-4 published in March 2023 solved nearly all the tasks (95%). These findings suggest that ToM-like ability (thus far considered to be uniquely human) may have spontaneously emerged as a byproduct of language
    
[^42]: SciRepEval：一个用于科学文献表示的多格式基准

    SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. (arXiv:2211.13308v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.13308](http://arxiv.org/abs/2211.13308)

    SciRepEval是第一个综合评估科学文献表示的全面基准，其中包括四种格式的 25 个任务。通过使用格式特定的控制代码和适配器，可以改进科学文献表示模型的泛化能力。

    

    学习的科学文献表示可以作为下游任务的有价值输入特征，无需进一步微调。然而，用于评估这些表示的现有基准未能捕捉到相关任务的多样性。为此，我们介绍了 SciRepEval，第一个用于训练和评估科学文献表示的全面基准。它包括四种格式的 25 个具有挑战性和现实性的任务，其中 11 个是新任务：分类、回归、排名和搜索。我们使用该基准来研究和改进科学文档表示模型的泛化能力。我们展示了最先进的模型如何在任务格式方面缺乏泛化性能，简单的多任务训练也不能改进它们。然而，一种新的方法，学习每个文档的多个嵌入，每个嵌入专门针对不同的格式，可以提高性能。我们尝试使用任务格式特定的控制代码和适配器。

    Learned representations of scientific documents can serve as valuable input features for downstream tasks, without the need for further fine-tuning. However, existing benchmarks for evaluating these representations fail to capture the diversity of relevant tasks. In response, we introduce SciRepEval, the first comprehensive benchmark for training and evaluating scientific document representations. It includes 25 challenging and realistic tasks, 11 of which are new, across four formats: classification, regression, ranking and search. We then use the benchmark to study and improve the generalization ability of scientific document representation models. We show how state-of-the-art models struggle to generalize across task formats, and that simple multi-task training fails to improve them. However, a new approach that learns multiple embeddings per document, each tailored to a different format, can improve performance. We experiment with task-format-specific control codes and adapters in 
    
[^43]: HyperMixer：一种基于MLP的低成本Transformer替代方案

    HyperMixer: An MLP-based Low Cost Alternative to Transformers. (arXiv:2203.03691v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.03691](http://arxiv.org/abs/2203.03691)

    HyperMixer是一种低成本的基于MLP的Transformer替代方案，通过动态形成标记混合MLP来实现自然语言理解，其性能比替代方案好，并可与Transformer媲美，成本更低。

    

    Transformer架构是自然语言理解的首选模型，但它们的成本相当高，因为它们在输入长度方面具有二次复杂度，需要大量的训练数据，并且可能难以调整。为了降低成本，我们研究了简单的基于MLP的架构。我们发现现有的架构（例如MLPMixer）通过静态的MLP独立地应用于每个特征，而过于脱离自然语言理解所需的归纳偏差。在本文中，我们提出了一种简单的改进，即HyperMixer，它使用超网络动态地形成标记混合MLP。实验上，我们证明了我们的模型表现优于替代的基于MLP的模型，并与Transformer媲美。与Transformer不同，HyperMixer在处理时间、训练数据和超参数调整方面具有大大降低的成本。

    Transformer-based architectures are the model of choice for natural language understanding, but they come at a significant cost, as they have quadratic complexity in the input length, require a lot of training data, and can be difficult to tune. In the pursuit of lower costs, we investigate simple MLP-based architectures. We find that existing architectures such as MLPMixer, which achieves token mixing through a static MLP applied to each feature independently, are too detached from the inductive biases required for natural language understanding. In this paper, we propose a simple variant, HyperMixer, which forms the token mixing MLP dynamically using hypernetworks. Empirically, we demonstrate that our model performs better than alternative MLP-based models, and on par with Transformers. In contrast to Transformers, HyperMixer achieves these results at substantially lower costs in terms of processing time, training data, and hyperparameter tuning.
    

