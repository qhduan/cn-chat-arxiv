# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CBQ: Cross-Block Quantization for Large Language Models](https://rss.arxiv.org/abs/2312.07950) | CBQ是一种用于大型语言模型的跨块重构型后训练量化方法。CBQ通过使用同源重构方案来建立块间的长程依赖关系，最小化误差积累。CBQ还采用了粗到精的预处理策略和自适应的取整技术，使其能够有效处理极端异常值并提高整体量化精度。 |
| [^2] | [Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs](https://rss.arxiv.org/abs/2312.05356) | 这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。 |
| [^3] | [Toward Informal Language Processing: Knowledge of Slang in Large Language Models](https://arxiv.org/abs/2404.02323) | 大型语言模型通过建立支持多任务评估的俚语数据集，有效实现了俚语检测和识别地区与历史俚语来源，并通过探索LLMs输出分布提供了解释性洞见。 |
| [^4] | [HyperCLOVA X Technical Report](https://arxiv.org/abs/2404.01954) | HyperCLOVA X 是针对韩国语言和文化定制的大型语言模型，同时具有竞争能力的英语、数学和编码能力，其推理能力强大且具有跨语言的通用能力。 |
| [^5] | [Octopus v2: On-device language model for super agent](https://arxiv.org/abs/2404.01744) | 该研究提出了一种新方法，通过将具有20亿参数的设备上模型赋予超越GPT-4的准确性和延迟性能，并将上下文长度缩减95％，从而解决了函数调用中的延迟和准确性问题。 |
| [^6] | [CHOPS: CHat with custOmer Profile Systems for Customer Service with LLMs](https://arxiv.org/abs/2404.01343) | CHOPS提出了一个名为CHOPS的LLM代理，旨在更高效地利用现有数据库或系统来访问用户信息，提供准确合理的响应或执行所需操作，同时避免有害操作。 |
| [^7] | [MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation](https://arxiv.org/abs/2403.19305) | 提出了MATEval框架，利用多个类GPT-4的LLMs作为评估Agent，模拟人类合作讨论方法，以评估开放性文本，结合自我反思和思维链策略，并加入反馈机制，提升评估深度和广度。 |
| [^8] | [Language Models for Text Classification: Is In-Context Learning Enough?](https://arxiv.org/abs/2403.17661) | 本研究通过对16个文本分类数据集的大规模评估研究，填补了现有研究缺乏对文本生成模型与提示技术与更传统的文本分类方法之间比较的理解。 |
| [^9] | [ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models](https://arxiv.org/abs/2403.16187) | ALoRA创新地提出了分配低秩适应性（ALoRA）方法，通过动态调整适应过程中的固有秩，从而解决了在微调大型语言模型时固定固有秩带来的问题。 |
| [^10] | [On the Fragility of Active Learners](https://arxiv.org/abs/2403.15744) | 本研究发现主动学习技术只在特定情境下有效，对文本分类从业者的建议是选择适当的文本表示和分类器同样重要。 |
| [^11] | [Detoxifying Large Language Models via Knowledge Editing](https://arxiv.org/abs/2403.14472) | 本文研究了使用知识编辑技术对大型语言模型进行去毒化，在构建了SafeEdit基准的基础上，提出了一种简单而有效的方法 DINM，可以通过少量调整步骤减少模型的毒性，同时对各种去毒方法的内部机制进行了深入分析。 |
| [^12] | [AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models](https://arxiv.org/abs/2403.13269) | AFLoRA是一种自适应冻结低秩调整方法，通过逐步冻结投影矩阵来提高性能，减少计算量，并提供对GLUE基准测试的最先进表现。 |
| [^13] | [A Novel Paradigm Boosting Translation Capabilities of Large Language Models](https://arxiv.org/abs/2403.11430) | 新范式关注在LLMs的预训练阶段增强跨语言对齐能力，强调使用高质量的小型双语数据，在传统机器翻译方法无法胜任的任务上取得了良好的效果。 |
| [^14] | [Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models](https://arxiv.org/abs/2403.09792) | 论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。 |
| [^15] | [Rectifying Demonstration Shortcut in In-Context Learning](https://arxiv.org/abs/2403.09488) | 本研究旨在纠正大型语言模型在上下文学习中的演示快捷方式，并引入了一种新的明示意识校准方法。 |
| [^16] | [Rethinking ASTE: A Minimalist Tagging Scheme Alongside Contrastive Learning](https://arxiv.org/abs/2403.07342) | 提出一种新颖的标记方案，并采用对比学习方法来重新思考ASTE，该方法在性能上优于最先进技术，同时具有更紧凑的设计和降低的计算开销，尤其在少样本学习情景下展现出优越效果。 |
| [^17] | [PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck](https://arxiv.org/abs/2403.05297) | PEEB是一种基于部分的图像分类器，通过将类别名称转换为描述视觉部分的文本描述符，并将检测到的部分的嵌入与文本描述符匹配，从而在零样本设置中表现出色，并且不仅在监督学习中表现出色，而且还首次实现用户编辑类定义形成新分类器无需重新训练。 |
| [^18] | [Do Large Language Model Understand Multi-Intent Spoken Language ?](https://arxiv.org/abs/2403.04481) | 该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。 |
| [^19] | [IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators](https://arxiv.org/abs/2403.03894) | 通过利用编译器中间表示来改进代码-LMs的多语言能力和促进跨语言转移。 |
| [^20] | [Not all Layers of LLMs are Necessary during Inference](https://arxiv.org/abs/2403.02181) | 推理过程中，根据输入实例的不同难易程度，本文提出了一种名为AdaInfer的算法，可以自适应地使用浅层和深层，从而节省了计算资源。 |
| [^21] | [Evaluating and Mitigating Number Hallucinations in Large Vision-Language Models: A Consistency Perspective](https://arxiv.org/abs/2403.01373) | 本文评估了大型视觉语言模型中的数字幻觉问题，并提出一种一致性训练方法，可使数字幻觉得到显著改善 |
| [^22] | [Entity-Aware Multimodal Alignment Framework for News Image Captioning](https://arxiv.org/abs/2402.19404) | 设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。 |
| [^23] | [Z-AGI Labs at ClimateActivism 2024: Stance and Hate Event Detection on Social Media](https://arxiv.org/abs/2402.17014) | 研究旨在在社交媒体上帮助气候活动人士识别和应对仇恨言论，团队评估了多种模型，在仇恨言论检测、仇恨言论目标识别和立场检测方面取得了有趣的结果。 |
| [^24] | [Triad: A Framework Leveraging a Multi-Role LLM-based Agent to Solve Knowledge Base Question Answering](https://arxiv.org/abs/2402.14320) | Triad框架利用了基于多角色LLM代理来解决知识库问答问题，通过代理的不同角色分别处理KBQA子任务，合作完成KBQA任务，并在多个基准数据集上表现出色。 |
| [^25] | [Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations](https://arxiv.org/abs/2402.12038) | 本研究提出了Self-AMPLIFY方法，通过将事后解释方法应用于小型语言模型（SLMs），自动生成基于原因的解释，以提高它们自身的性能。 |
| [^26] | [Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning](https://arxiv.org/abs/2402.06025) | 本论文建立了一个计算模型来模拟人们通过实验主动推断隐藏规则的过程，并发现显式假设、概率规则和在线更新的组合可以解释人们在类似Zendo任务上的表现。 |
| [^27] | [L-TUNING: Synchronized Label Tuning for Prompt and Prefix in LLMs](https://arxiv.org/abs/2402.01643) | 本文介绍了L-Tuning，一种在自然语言推理（NLI）框架内的高效微调方法，通过对预训练的LLM中的标签标记进行微调，提高了训练准确性和效率，并增强了模型的训练细微差别。 |
| [^28] | [Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?](https://arxiv.org/abs/2402.00841) | 本文研究了在实际环境中的会议摘要任务，通过比较小型紧凑LLMs和零射击较大LLMs的性能，发现大多数小型LLMs无法超越较大的零射击LLMs，但FLAN-T5是一个例外，它的性能与许多零射击LLMs持平甚至更好。 |
| [^29] | [Learning Planning-based Reasoning by Trajectories Collection and Process Reward Synthesizing](https://arxiv.org/abs/2402.00658) | 本文提出了一种通过直接偏好优化在收集到的轨迹上学习基于规划的推理的框架，以解决大型语言模型在复杂推理任务中的虚幻和缺陷问题。 |
| [^30] | [PILOT: Legal Case Outcome Prediction with Case Law](https://arxiv.org/abs/2401.15770) | 提出了新的PILOT框架，旨在解决使用案例法预测法律案例结果的挑战，包括相关案例检索和时间模式处理两个模块。 |
| [^31] | [Tuning Language Models by Proxy](https://arxiv.org/abs/2401.08565) | 介绍了一种代理调整的轻量级解码时算法，可以通过对小型调整后的LM的预测与未调整LM的预测之间的差异来调整大型预训练LM的预测，从而实现资源节约和保留更大规模预训练的好处。 |
| [^32] | [MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization](https://arxiv.org/abs/2401.06838) | MAPO提出了一个多语言对齐作为偏好优化框架，可以显著提高各种模型在多语言推理中的表现。 |
| [^33] | [Machine Translation for Ge'ez Language](https://arxiv.org/abs/2311.14530) | 该研究探讨了改善盖兹语机器翻译的方法，包括从相关语言进行迁移学习、优化共享词汇和标记分割方法、大型预训练模型的微调，以及在少样本情况下使用大型语言模型进行翻译，其中一种基于语言相关性的多语言神经机器翻译模型相比标准双语模型有4 BLEU的平均性能提升。 |
| [^34] | [Psychometric Predictive Power of Large Language Models](https://arxiv.org/abs/2311.07484) | 指令调整和提示在大型语言模型中无法提供比基础模型更好的认知建模估计。 |
| [^35] | [Flames: Benchmarking Value Alignment of Chinese Large Language Models](https://arxiv.org/abs/2311.06899) | 中国的大型语言模型的价值观契合性需要更加全面的评估，该研究提出了一个名为"火焰"（Flames）的基准测试，涵盖了常见的无害原则和特定中国价值观，以及复杂场景和隐含恶意的提示方法。 |
| [^36] | [Sequence-Level Certainty Reduces Hallucination In Knowledge-Grounded Dialogue Generation](https://arxiv.org/abs/2310.18794) | 序列级确定性是减少基于知识的对话生成中幻觉的关键，这项工作提出了基于概率确定性和语义确定性的序列级确定性，结果表明更高水平的确定性对应更低水平的幻觉，进一步提出了基于确定性的响应排序方法 |
| [^37] | [SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation](https://arxiv.org/abs/2310.18376) | SQLformer是一个用于文本到SQL翻译的深度自回归查询图生成模型，采用了特定的Transformer架构，并通过结构归纳偏差解决领域泛化和自然语言与SQL查询对齐的难题。 |
| [^38] | [Evaluating Spatial Understanding of Large Language Models](https://arxiv.org/abs/2310.14540) | 本研究评估了大型语言模型对空间结构的理解能力，发现LLMs在表示和推理空间结构时的表现存在显著差异，具有捕捉空间结构隐含特征的潜力。 |
| [^39] | [Can LLM-Generated Misinformation Be Detected?](https://arxiv.org/abs/2309.13788) | LLM生成的虚假信息可能比人类撰写的虚假信息更难以检测，具有更具欺骗性的风格，可能造成更多危害。 |
| [^40] | [Contextual Label Projection for Cross-Lingual Structure Extraction](https://arxiv.org/abs/2309.08943) | 本文提出了一种新的标签投影方法CLaP，利用上下文翻译标签，确保更准确的翻译，并在跨语言结构预测任务中取得显著表现。 |
| [^41] | [H2O-Danube-1.8B Technical Report.](http://arxiv.org/abs/2401.16818) | H2O-Danube-1.8B 是一个在 1T 个标记上训练的 18 亿语言模型，具有高度竞争力的指标。同时，他们还发布了一个经过微调和优化训练的聊天模型，进一步推动语言模型的经济民主化。 |
| [^42] | [Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding.](http://arxiv.org/abs/2401.12798) | 这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。 |
| [^43] | [Adapting LLM Agents Through Communication.](http://arxiv.org/abs/2310.01444) | 这项研究提出了一种名为学习通信（LTC）的训练方法，使用该方法可使LLM代理通过与环境和其他代理的交互不断改进，以适应新任务，而无需过多人类监督。 |
| [^44] | [BooookScore: A systematic exploration of book-length summarization in the era of LLMs.](http://arxiv.org/abs/2310.00785) | 本文对LLM模型进行了系统探索，以解决对超过上下文窗口大小的书籍进行摘要的问题，并通过两种提示工作流实施了基于LLM的书籍长度摘要器的连贯性研究。通过对100本书的GPT-4生成摘要的人工注释，发现了八种常见的连贯性错误。 |
| [^45] | [Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling.](http://arxiv.org/abs/2309.10435) | 本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。 |
| [^46] | [Understanding Catastrophic Forgetting in Language Models via Implicit Inference.](http://arxiv.org/abs/2309.10105) | 本研究通过在语言模型上进行实验，发现微调对模型在微调数据分布任务上的表现有正面影响，但会抑制模型在其他任务上的能力，特别是与微调分布最接近的任务。作者假设语言模型会隐式推理任务，并且微调过程偏向于微调数据分布中的任务。作者进一步提出了共轭提示方法，以尝试恢复模型在预训练阶段的能力。 |
| [^47] | [Measuring Catastrophic Forgetting in Cross-Lingual Transfer Paradigms: Exploring Tuning Strategies.](http://arxiv.org/abs/2309.06089) | 该研究比较了不同的微调和跨语言转移策略在解决跨语言任务时的表现，评估了灾难性遗忘的程度和转移的成功程度。 |
| [^48] | [Large Language Models as Optimizers.](http://arxiv.org/abs/2309.03409) | 本论文提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，通过自然语言描述优化任务。经过实验证明，该方法在线性回归和旅行推销员问题上表现出色，并且优化的最佳提示超过了人为设计的提示。 |
| [^49] | [Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm.](http://arxiv.org/abs/2308.11767) | 本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。 |
| [^50] | [DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue.](http://arxiv.org/abs/2308.08043) | DiagGPT将大型语言模型(LLMs)扩展到任务导向的对话场景，提供了在复杂诊断场景中主动提问和引导用户完成任务的能力。 |
| [^51] | [CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition.](http://arxiv.org/abs/2307.15432) | 本文提出了一种具有情绪转移感知的跨模态融合网络（CFN-ESA）用于对话情绪识别，通过将文本模态作为主要情感信息的来源，视觉和声学模态作为次要信息的来源，并引入情绪转移模块来解决情绪转移场景下情感识别的问题。 |
| [^52] | [FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets.](http://arxiv.org/abs/2307.10928) | FLASK是一种基于对齐技能集的细粒度语言模型评估协议，通过将粗级评分分解为每个指令的技能集级评分，实现了对模型性能的全面视角和提高评估的可靠性。 |
| [^53] | [Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps.](http://arxiv.org/abs/2307.05052) | 本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。 |
| [^54] | [Recommender Systems in the Era of Large Language Models (LLMs).](http://arxiv.org/abs/2307.02046) | 大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。 |
| [^55] | [In-Context Learning through the Bayesian Prism.](http://arxiv.org/abs/2306.04891) | 这篇论文研究了大型语言模型中的上下文学习现象，并通过实验证据展示了Transformer模型在多种设置下表现出贝叶斯预测器的行为。作者还探讨了上下文学习与贝叶斯学习框架之间的联系，并提出了一个线性回归任务来验证这种联系。 |
| [^56] | [The Curse of Recursion: Training on Generated Data Makes Models Forget.](http://arxiv.org/abs/2305.17493) | 使用生成数据进行训练会导致模型不可逆的缺陷并且使得原始内容分布的尾部消失，这种效应称为模型折叠。我们证明了这种现象在所有学习生成模型中都存在，必须认真对待。 |
| [^57] | [Eliciting the Translation Ability of Large Language Models via Multilingual Finetuning with Translation Instructions.](http://arxiv.org/abs/2305.15083) | 本文通过对多语言预训练语言模型进行微调，研究了它们如何通过翻译指令执行多语言翻译任务。研究发现多语言LLMs具有较强的翻译能力，这取决于语言与英语的相似性和预训练阶段使用的数据量。此外，执行翻译指令的能力依赖于对指令的理解和不同语言之间的对齐。 |
| [^58] | [EE-TTS: Emphatic Expressive TTS with Linguistic Information.](http://arxiv.org/abs/2305.12107) | 该论文提出了一种利用语言信息和强调预测器实现表现力强且自然的TTS系统，并在实验中表现出良好的表现和泛化能力。 |
| [^59] | [Out-of-distribution Evidence-aware Fake News Detection via Dual Adversarial Debiasing.](http://arxiv.org/abs/2304.12888) | 该论文提出了一种新颖的双重对抗学习方法，通过在模型中加入去偏置鉴别器，旨在训练模型更好地进行越界证据感知假新闻检测，有效减轻新闻和证据内容偏差的影响。 |
| [^60] | [Analyzing Feed-Forward Blocks in Transformers through the Lens of Attention Map.](http://arxiv.org/abs/2302.00456) | 通过关注图解析Transformer中的前馈模块，揭示了其修改输入语境化以强调特定类型语言组合的作用，并暗示了Transformer层处理中的潜在冗余。 |
| [^61] | [ComCLIP: Training-Free Compositional Image and Text Matching.](http://arxiv.org/abs/2211.13854) | 本文提出了一个无需训练的组合图像与文本匹配模型 ComCLIP，通过将输入图像分解为主体、对象和动作子图像，并结合视觉编码器和文本编码器进行逐步匹配，以解决组合图像与文本匹配中的伪匹配问题。 |

# 详细

[^1]: 跨块量化：用于大型语言模型的跨块量化方法

    CBQ: Cross-Block Quantization for Large Language Models

    [https://rss.arxiv.org/abs/2312.07950](https://rss.arxiv.org/abs/2312.07950)

    CBQ是一种用于大型语言模型的跨块重构型后训练量化方法。CBQ通过使用同源重构方案来建立块间的长程依赖关系，最小化误差积累。CBQ还采用了粗到精的预处理策略和自适应的取整技术，使其能够有效处理极端异常值并提高整体量化精度。

    

    后训练量化（PTQ）在以极低成本压缩大型语言模型（LLM）方面起着重要作用。然而，现有的PTQ方法只关注处理单个层或单个块内的异常值，忽略了块之间的依赖关系，在低位设置中导致严重的性能下降。本文提出了一种基于块间重构的跨块PTQ方法CBQ。CBQ采用了一种同源重构方案来实现块间的长程依赖关系，以最小化误差积累。此外，CBQ还结合了一种粗到精的预处理策略（CFP）来抑制权重和激活值的异常值，并配合一种自适应的LoRA取整技术实现精确的权重量化。这些创新使CBQ不仅能够有效处理极端异常值，还能提高整体量化精度。广泛的实验证明，CBQ在低位量化（W4A4，W4A8等）方面具有优越性能。

    Post-training quantization (PTQ) has played a key role in compressing large language models (LLMs) with ultra-low costs. However, existing PTQ methods only focus on handling the outliers within one layer or one block, which ignores the dependency of blocks and leads to severe performance degradation in low-bit settings. In this paper, we propose CBQ, a cross-block reconstruction-based PTQ method for LLMs. CBQ employs a cross-block dependency using a homologous reconstruction scheme, establishing long-range dependencies across multiple blocks to minimize error accumulation. Furthermore, CBQ incorporates a coarse-to-fine preprocessing (CFP) strategy for suppressing weight and activation outliers, coupled with an adaptive LoRA-Rounding technique for precise weight quantization. These innovations enable CBQ to not only handle extreme outliers effectively but also improve overall quantization accuracy. Extensive experiments show that CBQ achieves superior low-bit quantization (W4A4, W4A8, W
    
[^2]: Neuron Patching: 神经元层面的模型编辑与代码生成

    Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs

    [https://rss.arxiv.org/abs/2312.05356](https://rss.arxiv.org/abs/2312.05356)

    这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。

    

    大型语言模型在软件工程中得到了成功应用，特别是在代码生成方面。更新这些模型的新知识非常昂贵，通常需要全面实现其价值。在本文中，我们提出了一种新颖有效的模型编辑方法MENT，用于在编码任务中修补LLM模型。基于生成式LLM的机制，MENT可以在预测下一个令牌时进行模型编辑，并进一步支持常见的编码任务。MENT具有高效、有效和可靠的特点。它可以通过修补1或2个神经元来纠正神经模型。作为神经元层面上生成模型编辑的先驱工作，我们规范了编辑过程并介绍了相关概念。此外，我们还引入了新的衡量方法来评估其泛化能力，并建立了一个用于进一步研究的基准。我们的方法在三个编码任务上进行了评估，包括API序列推荐、行级代码生成和伪代码到代码转换。

    Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction
    
[^3]: 迈向非正式语言处理：大型语言模型对俚语的认知

    Toward Informal Language Processing: Knowledge of Slang in Large Language Models

    [https://arxiv.org/abs/2404.02323](https://arxiv.org/abs/2404.02323)

    大型语言模型通过建立支持多任务评估的俚语数据集，有效实现了俚语检测和识别地区与历史俚语来源，并通过探索LLMs输出分布提供了解释性洞见。

    

    大型语言模型（LLMs）的最新进展为自然语言系统处理非正式语言提供了强大潜力。非正式语言的一个代表形式是俚语，在日常对话和在线社交媒体中常用。迄今为止，由于缺乏一个精心设计且可公开访问的基准，俚语在LLMs中尚未得到全面评估。我们利用电影字幕构建了一个数据集，支持在涉及俚语自动处理的多样化任务上进行评估。我们展示了我们的数据集在两个核心应用中（俚语检测和识别自然语句中俚语的地区和历史来源）的评估和微调的有效性。我们还展示了如何使用我们的数据集来探测LLMs的输出分布以获得解释性洞见。我们发现，尽管 GPT-4 等LLMs 在零次尝试方面表现良好，但在俚语处理方面仍有改进的空间。

    arXiv:2404.02323v1 Announce Type: new  Abstract: Recent advancement in large language models (LLMs) has offered a strong potential for natural language systems to process informal language. A representative form of informal language is slang, used commonly in daily conversations and online social media. To date, slang has not been comprehensively evaluated in LLMs due partly to the absence of a carefully designed and publicly accessible benchmark. Using movie subtitles, we construct a dataset that supports evaluation on a diverse set of tasks pertaining to automatic processing of slang. For both evaluation and finetuning, we show the effectiveness of our dataset on two core applications: 1) slang detection, and 2) identification of regional and historical sources of slang from natural sentences. We also show how our dataset can be used to probe the output distributions of LLMs for interpretive insights. We find that while LLMs such as GPT-4 achieve good performance in a zero-shot setti
    
[^4]: HyperCLOVA X 技术报告

    HyperCLOVA X Technical Report

    [https://arxiv.org/abs/2404.01954](https://arxiv.org/abs/2404.01954)

    HyperCLOVA X 是针对韩国语言和文化定制的大型语言模型，同时具有竞争能力的英语、数学和编码能力，其推理能力强大且具有跨语言的通用能力。

    

    我们介绍了 HyperCLOVA X，这是一系列针对韩国语言和文化定制的大型语言模型（LLMs），同时具有在英语、数学和编码方面的竞争能力。HyperCLOVA X 在平衡混合的韩语、英语和代码数据上进行训练，然后通过高质量的人工注释数据进行指导微调，同时遵守严格的安全准则，体现了我们对负责任人工智能的承诺。该模型在包括综合推理、知识、常识、真实性、编码、数学、聊天、遵循指令和无害性在内的各种基准测试中进行了评估，涵盖韩语和英语。HyperCLOVA X 在韩语方面表现出很强的推理能力，得益于对语言和文化细微差异的深刻理解。对固有的双语特性的进一步分析及其扩展到多语言的研究突显出该模型的跨语言熟练性和强大的泛化能力，以适应未定向的目标。

    arXiv:2404.01954v1 Announce Type: cross  Abstract: We introduce HyperCLOVA X, a family of large language models (LLMs) tailored to the Korean language and culture, along with competitive capabilities in English, math, and coding. HyperCLOVA X was trained on a balanced mix of Korean, English, and code data, followed by instruction-tuning with high-quality human-annotated datasets while abiding by strict safety guidelines reflecting our commitment to responsible AI. The model is evaluated across various benchmarks, including comprehensive reasoning, knowledge, commonsense, factuality, coding, math, chatting, instruction-following, and harmlessness, in both Korean and English. HyperCLOVA X exhibits strong reasoning capabilities in Korean backed by a deep understanding of the language and cultural nuances. Further analysis of the inherent bilingual nature and its extension to multilingualism highlights the model's cross-lingual proficiency and strong generalization ability to untargeted la
    
[^5]: Octopus v2：用于超级代理的设备上语言模型

    Octopus v2: On-device language model for super agent

    [https://arxiv.org/abs/2404.01744](https://arxiv.org/abs/2404.01744)

    该研究提出了一种新方法，通过将具有20亿参数的设备上模型赋予超越GPT-4的准确性和延迟性能，并将上下文长度缩减95％，从而解决了函数调用中的延迟和准确性问题。

    

    语言模型在各种软件应用中展现出了高效性，特别是与自动工作流相关的任务。这些模型具有调用函数的关键能力，在创建AI代理时至关重要。尽管大规模语言模型在云环境中表现出色，但往往存在着隐私和成本方面的担忧。当前用于函数调用的设备上模型面临延迟和准确性问题。我们的研究提出了一种新方法，使具有20亿参数的设备上模型在准确性和延迟方面超越了GPT-4，并将上下文长度缩减了95%。与基于RAG的函数调用机制的Llama-7B相比，我们的方法将延迟提高了35倍。这种方法将延迟降低到适合在生产环境中的各种边缘设备上部署的水平上，符合性能要求。

    arXiv:2404.01744v1 Announce Type: new  Abstract: Language models have shown effectiveness in a variety of software applications, particularly in tasks related to automatic workflow. These models possess the crucial ability to call functions, which is essential in creating AI agents. Despite the high performance of large-scale language models in cloud environments, they are often associated with concerns over privacy and cost. Current on-device models for function calling face issues with latency and accuracy. Our research presents a new method that empowers an on-device model with 2 billion parameters to surpass the performance of GPT-4 in both accuracy and latency, and decrease the context length by 95\%. When compared to Llama-7B with a RAG-based function calling mechanism, our method enhances latency by 35-fold. This method reduces the latency to levels deemed suitable for deployment across a variety of edge devices in production environments, aligning with the performance requisite
    
[^6]: CHOPS: CHat with custOmer Profile Systems for Customer Service with LLMs

    CHOPS: CHat with custOmer Profile Systems for Customer Service with LLMs

    [https://arxiv.org/abs/2404.01343](https://arxiv.org/abs/2404.01343)

    CHOPS提出了一个名为CHOPS的LLM代理，旨在更高效地利用现有数据库或系统来访问用户信息，提供准确合理的响应或执行所需操作，同时避免有害操作。

    

    商业和软件平台越来越倾向于使用像GPT-3.5、GPT-4、GLM-3和LLaMa-2这样的大型语言模型（LLMs）作为客户服务的聊天辅助或推理代理。然而，当前基于LLM的客户服务模型在与客户配置文件的集成方面存在局限，并且缺乏有效服务所需的操作能力。为了解决这些问题，我们提出了一个名为CHOPS（CHat with custOmer Profile in existing System）的LLM代理，旨在：（1）高效利用现有数据库或系统以访问用户信息或按照现有指南与这些系统交互；（2）提供准确合理的响应或在系统中执行所需操作，同时避免有害操作；（3）利用

    arXiv:2404.01343v1 Announce Type: cross  Abstract: Businesses and software platforms are increasingly turning to Large Language Models (LLMs) such as GPT-3.5, GPT-4, GLM-3, and LLaMa-2 for chat assistance with file access or as reasoning agents for customer service. However, current LLM-based customer service models have limited integration with customer profiles and lack the operational capabilities necessary for effective service. Moreover, existing API integrations emphasize diversity over the precision and error avoidance essential in real-world customer service scenarios. To address these issues, we propose an LLM agent named CHOPS (CHat with custOmer Profile in existing System), designed to: (1) efficiently utilize existing databases or systems for accessing user information or interacting with these systems following existing guidelines; (2) provide accurate and reasonable responses or carry out required operations in the system while avoiding harmful operations; and (3) leverag
    
[^7]: MATEval：用于推进开放性文本评估的多Agent讨论框架

    MATEval: A Multi-Agent Discussion Framework for Advancing Open-Ended Text Evaluation

    [https://arxiv.org/abs/2403.19305](https://arxiv.org/abs/2403.19305)

    提出了MATEval框架，利用多个类GPT-4的LLMs作为评估Agent，模拟人类合作讨论方法，以评估开放性文本，结合自我反思和思维链策略，并加入反馈机制，提升评估深度和广度。

    

    最近生成式大型语言模型（LLMs）的进展令人瞩目，然而，这些模型生成的文本质量经常暴露出持续存在的问题。评估这些模型生成的文本质量，特别是在开放性文本中，一直是一个重大挑战。为解决这一问题，最近的研究探讨了使用LLMs作为评估者的可能性。虽然使用单个LLM作为评估Agent表现出潜力，但却存在显著的不确定性和不稳定性。为了解决这些问题，我们提出了 MATEval：一种“多Agent文本评估框架”，其中所有Agent都由像GPT-4的LLMs扮演。MATEval框架模拟人类协作讨论方法，整合多个Agent的互动来评估开放性文本。我们的框架结合了自我反思和“思维链”策略，以及反馈机制，增强了评估的深度和广度。

    arXiv:2403.19305v1 Announce Type: cross  Abstract: Recent advancements in generative Large Language Models(LLMs) have been remarkable, however, the quality of the text generated by these models often reveals persistent issues. Evaluating the quality of text generated by these models, especially in open-ended text, has consistently presented a significant challenge. Addressing this, recent work has explored the possibility of using LLMs as evaluators. While using a single LLM as an evaluation agent shows potential, it is filled with significant uncertainty and instability. To address these issues, we propose the MATEval: A "Multi-Agent Text Evaluation framework" where all agents are played by LLMs like GPT-4. The MATEval framework emulates human collaborative discussion methods, integrating multiple agents' interactions to evaluate open-ended text. Our framework incorporates self-reflection and Chain-of-Thought (CoT) strategies, along with feedback mechanisms, enhancing the depth and br
    
[^8]: 文本分类的语言模型：仅仅上下文学习就足够了吗？

    Language Models for Text Classification: Is In-Context Learning Enough?

    [https://arxiv.org/abs/2403.17661](https://arxiv.org/abs/2403.17661)

    本研究通过对16个文本分类数据集的大规模评估研究，填补了现有研究缺乏对文本生成模型与提示技术与更传统的文本分类方法之间比较的理解。

    

    最近的基础语言模型在零次和少次标记设置中展示了在许多自然语言处理任务中的最先进性能。这些模型相对于基于微调的更标准的方法的优势在于能够理解用自然语言编写的指令（提示），这有助于它们更好地推广到不同的任务和领域，而无需特定的训练数据。这使它们适合解决具有有限标注实例数量的领域的文本分类问题。但是，现有研究在规模上有限，并缺乏对文本生成模型与提示技术相结合与更传统的文本分类方法（如微调掩码语言模型）的比较的理解。在本文中，我们通过对涵盖二元、多类和多标签问题的16个文本分类数据集进行大规模评估研究来填补这一研究空白。

    arXiv:2403.17661v1 Announce Type: cross  Abstract: Recent foundational language models have shown state-of-the-art performance in many NLP tasks in zero- and few-shot settings. An advantage of these models over more standard approaches based on fine-tuning is the ability to understand instructions written in natural language (prompts), which helps them generalise better to different tasks and domains without the need for specific training data. This makes them suitable for addressing text classification problems for domains with limited amounts of annotated instances. However, existing research is limited in scale and lacks understanding of how text generation models combined with prompting techniques compare to more established methods for text classification such as fine-tuning masked language models. In this paper, we address this research gap by performing a large-scale evaluation study for 16 text classification datasets covering binary, multiclass, and multilabel problems. In par
    
[^9]: ALoRA: 为大型语言模型微调分配低秩适应性

    ALoRA: Allocating Low-Rank Adaptation for Fine-tuning Large Language Models

    [https://arxiv.org/abs/2403.16187](https://arxiv.org/abs/2403.16187)

    ALoRA创新地提出了分配低秩适应性（ALoRA）方法，通过动态调整适应过程中的固有秩，从而解决了在微调大型语言模型时固定固有秩带来的问题。

    

    参数高效的微调（PEFT）因其在大型语言模型时代的有效性和效率而被广泛研究。低秩适应性（LoRA）已经展示出作为一种流行和代表性方法的令人钦佩的性能。然而，它是使用固定的固有秩来实现的，这可能不是下游任务的理想设置。认识到需要更灵活的下游任务适应性，我们将LoRA的方法学扩展到一种创新的方法，我们将其称为分配低秩适应性（ALoRA），这样可以在适应过程中动态调整固有秩。首先，我们提出了一种新颖的方法，AB-LoRA，可以有效地估计每个LoRA等级的重要性得分。其次，受AB-LoRA的指导，我们逐渐修剪了过多且产生负面影响的LoRA等级，并将被修剪的LoRA预算分配给需要更高等级的重要Transformer模块。我们已经在各种实验上进行了研究。

    arXiv:2403.16187v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning (PEFT) is widely studied for its effectiveness and efficiency in the era of large language models. Low-rank adaptation (LoRA) has demonstrated commendable performance as a popular and representative method. However, it is implemented with a fixed intrinsic rank that might not be the ideal setting for the downstream tasks. Recognizing the need for more flexible downstream task adaptation, we extend the methodology of LoRA to an innovative approach we call allocating low-rank adaptation (ALoRA) that enables dynamic adjustments to the intrinsic rank during the adaptation process. First, we propose a novel method, AB-LoRA, that can effectively estimate the importance score of each LoRA rank. Second, guided by AB-LoRA, we gradually prune abundant and negatively impacting LoRA ranks and allocate the pruned LoRA budgets to important Transformer modules needing higher ranks. We have conducted experiments on variou
    
[^10]: 论主动学习者的脆弱性

    On the Fragility of Active Learners

    [https://arxiv.org/abs/2403.15744](https://arxiv.org/abs/2403.15744)

    本研究发现主动学习技术只在特定情境下有效，对文本分类从业者的建议是选择适当的文本表示和分类器同样重要。

    

    主动学习（AL）技术旨在通过迭代选择最有可能提高预测准确性的实例，最大程度地利用标注预算。然而，与随机抽样相比，在不同设置下（例如不同数据集，分类器），它们的益处并不一致。在这项实证研究中，我们研究了不同因素的组合如何可能掩盖主动学习技术的任何收益。专注于文本分类，我们在大约1000个实验中严格评估了进行分类，我们在大约1000个实验中严格评估了AL技术，这些实验在数据集、批大小、文本表示和分类器方面变化。我们表明，AL只在一组有限的情境中有效。我们还解决了使用与现实世界期望更好对齐的度量的问题。这项研究的影响在于对从业者的洞察：(a) 文本表示和分类器的选择与AL技术的选择一样重要，(b) 选择的

    arXiv:2403.15744v1 Announce Type: cross  Abstract: Active learning (AL) techniques aim to maximally utilize a labeling budget by iteratively selecting instances that are most likely to improve prediction accuracy. However, their benefit compared to random sampling has not been consistent across various setups, e.g., different datasets, classifiers. In this empirical study, we examine how a combination of different factors might obscure any gains from an AL technique.   Focusing on text classification, we rigorously evaluate AL techniques over around 1000 experiments that vary wrt the dataset, batch size, text representation and the classifier. We show that AL is only effective in a narrow set of circumstances. We also address the problem of using metrics that are better aligned with real world expectations.   The impact of this study is in its insights for a practitioner: (a) the choice of text representation and classifier is as important as that of an AL technique, (b) choice of the 
    
[^11]: 通过知识编辑实现对大型语言模型的去毒化

    Detoxifying Large Language Models via Knowledge Editing

    [https://arxiv.org/abs/2403.14472](https://arxiv.org/abs/2403.14472)

    本文研究了使用知识编辑技术对大型语言模型进行去毒化，在构建了SafeEdit基准的基础上，提出了一种简单而有效的方法 DINM，可以通过少量调整步骤减少模型的毒性，同时对各种去毒方法的内部机制进行了深入分析。

    

    本文研究了使用知识编辑技术来对大型语言模型（LLMs）进行去毒化。我们构建了一个名为SafeEdit的基准，涵盖了九种不安全类别，具有各种强大的攻击提示，并配备了全面的度量标准进行系统评估。我们进行了实验，比较了知识编辑方法与之前的基准线，结果表明知识编辑有潜力在对LLMs进行去毒化时，在对一般性能的影响相对有限。然后，我们提出了一个简单但有效的基准线，称为通过术中神经监测去毒化（DINM），通过仅一次实例的少量调整步骤减少LLMs的毒性。我们进一步对各种去毒方法的内部机制进行了深入分析，表明先前的方法如SFT和DPO可能仅抑制有毒参数的激活，而DINM则减轻有毒参数的毒性。

    arXiv:2403.14472v1 Announce Type: cross  Abstract: This paper investigates using knowledge editing techniques to detoxify Large Language Models (LLMs). We construct a benchmark, SafeEdit, which covers nine unsafe categories with various powerful attack prompts and equips comprehensive metrics for systematic evaluation. We conduct experiments to compare knowledge editing approaches with previous baselines, indicating that knowledge editing has the potential to efficiently detoxify LLMs with limited impact on general performance. Then, we propose a simple yet effective baseline, dubbed Detoxifying with Intraoperative Neural Monitoring (DINM), to diminish the toxicity of LLMs within a few tuning steps via only one instance. We further provide an in-depth analysis of the internal mechanism for various detoxify approaches, demonstrating that previous methods like SFT and DPO may merely suppress the activations of toxic parameters, while DINM mitigates the toxicity of the toxic parameters to
    
[^12]: AFLoRA: 自适应冻结低秩调整在大型模型参数高效微调中的应用

    AFLoRA: Adaptive Freezing of Low Rank Adaptation in Parameter Efficient Fine-Tuning of Large Models

    [https://arxiv.org/abs/2403.13269](https://arxiv.org/abs/2403.13269)

    AFLoRA是一种自适应冻结低秩调整方法，通过逐步冻结投影矩阵来提高性能，减少计算量，并提供对GLUE基准测试的最先进表现。

    

    我们提出了一种新颖的参数高效微调（PEFT）方法，称为自适应冻结低秩调整（AFLoRA）。具体地，对于每个预训练的冻结权重张量，我们添加一个可训练的低秩矩阵并行路径，即下投影和上投影矩阵，每个矩阵后面跟着一个特征变换向量。基于一种新颖的冻结分数，我们在微调过程中逐步冻结这些投影矩阵，以减少计算量并减轻过拟合。我们的实验结果表明，我们可以在GLUE基准测试中获得最先进的性能，平均改善高达0.85％，同时可减少高达9.5倍的平均可训练参数。在运行时间方面，与类似的PEFT备选方案相比，AFLoRA可以提供高达1.86倍的改进。除了我们方法的实际效用之外，我们还提供了关于训练

    arXiv:2403.13269v1 Announce Type: new  Abstract: We present a novel Parameter-Efficient Fine-Tuning (PEFT) method, dubbed as Adaptive Freezing of Low Rank Adaptation (AFLoRA). Specifically, for each pre-trained frozen weight tensor, we add a parallel path of trainable low-rank matrices, namely a down-projection and an up-projection matrix, each of which is followed by a feature transformation vector. Based on a novel freezing score, we the incrementally freeze these projection matrices during fine-tuning to reduce the computation and alleviate over-fitting. Our experimental results demonstrate that we can achieve state-of-the-art performance with an average improvement of up to $0.85\%$ as evaluated on GLUE benchmark while yeilding up to $9.5\times$ fewer average trainable parameters. While compared in terms of runtime, AFLoRA can yield up to $1.86\times$ improvement as opposed to similar PEFT alternatives. Besides the practical utility of our approach, we provide insights on the train
    
[^13]: 一种提升大型语言模型翻译能力的新范式

    A Novel Paradigm Boosting Translation Capabilities of Large Language Models

    [https://arxiv.org/abs/2403.11430](https://arxiv.org/abs/2403.11430)

    新范式关注在LLMs的预训练阶段增强跨语言对齐能力，强调使用高质量的小型双语数据，在传统机器翻译方法无法胜任的任务上取得了良好的效果。

    

    这篇论文研究了在机器翻译任务中增强大型语言模型（LLMs）翻译能力的策略。论文提出了一个包括三个阶段的新范式：使用大量单语数据的次级预训练，使用互文格式文档进行持续预训练，以及利用与源语言一致的指导进行监督微调。我们认为，重点应该放在增强LLMs在预训练期间的跨语言对齐能力，而不仅仅依靠SFT期间大量的双语数据。实验结果证实了我们的方法在不依赖大量双语数据的情况下，在传统机器翻译方法无法胜任的任务上取得了良好的效果。

    arXiv:2403.11430v1 Announce Type: new  Abstract: This paper presents a study on strategies to enhance the translation capabilities of large language models (LLMs) in the context of machine translation (MT) tasks. The paper proposes a novel paradigm consisting of three stages: Secondary Pre-training using Extensive Monolingual Data, Continual Pre-training with Interlinear Text Format Documents, and Leveraging Source-Language Consistent Instruction for Supervised Fine-Tuning. Previous research on LLMs focused on various strategies for supervised fine-tuning (SFT), but their effectiveness has been limited. While traditional machine translation approaches rely on vast amounts of parallel bilingual data, our paradigm highlights the importance of using smaller sets of high-quality bilingual data. We argue that the focus should be on augmenting LLMs' cross-lingual alignment abilities during pre-training rather than solely relying on extensive bilingual data during SFT. Experimental results co
    
[^14]: 图像是对齐的软肋：利用视觉漏洞破解多模态大语言模型

    Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models

    [https://arxiv.org/abs/2403.09792](https://arxiv.org/abs/2403.09792)

    论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。

    

    在这篇论文中，我们研究了多模态大语言模型（MLLMs）的无害对齐问题。我们对代表性的MLLMs的无害性能进行了系统的实证分析，并揭示了图像输入对MLLMs造成的对齐漏洞。受此启发，我们提出了一种名为HADES的新型破解方法，利用精心制作的图像隐藏和放大文本输入中的恶意意图的有害性。实验结果表明，HADES可以有效地破解现有的MLLMs，为LLaVA-1.5实现了90.26％的平均攻击成功率（ASR），为Gemini Pro Vision实现了71.60％的ASR。我们的代码和数据将公开发布。

    arXiv:2403.09792v1 Announce Type: cross  Abstract: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.
    
[^15]: 在上下文学习中纠正演示快捷方式

    Rectifying Demonstration Shortcut in In-Context Learning

    [https://arxiv.org/abs/2403.09488](https://arxiv.org/abs/2403.09488)

    本研究旨在纠正大型语言模型在上下文学习中的演示快捷方式，并引入了一种新的明示意识校准方法。

    

    大型语言模型（LLMs）能够利用它们的上下文学习（ICL）能力，仅凭少量演示便能解决各种任务。然而，LLMs常常依赖于它们对演示的预先训练的语义先验，而不是根据输入-标签关系继续进行ICL预测。本文将这一现象称为“演示快捷方式”。尽管先前的研究主要集中于改进预定义任务的ICL预测结果，我们的目标是纠正演示快捷方式，从而使LLM能够有效地从演示中学习新的输入-标签关系。为实现此目标，我们引入了一种明示意识的校准方法：In-Context Calibration。我们在两个设置中评估了所提出方法的有效性：（1）使用标准标签空间的原始ICL任务以及（2）任务学习设置，其中标签空间被语义无关的标记替换。

    arXiv:2403.09488v1 Announce Type: cross  Abstract: Large language models (LLMs) are able to solve various tasks with only a few demonstrations utilizing their in-context learning (ICL) abilities. However, LLMs often rely on their pre-trained semantic priors of demonstrations rather than on the input-label relationships to proceed with ICL prediction. In this work, we term this phenomenon as the `Demonstration Shortcut'. While previous works have primarily focused on improving ICL prediction results for predefined tasks, we aim to rectify the Demonstration Shortcut, thereby enabling the LLM to effectively learn new input-label relationships from demonstrations. To achieve this, we introduce In-Context Calibration, a demonstration-aware calibration method. We evaluate the effectiveness of the proposed method in two settings: (1) the Original ICL Task using the standard label space and (2) the Task Learning setting, where the label space is replaced with semantically unrelated tokens. In 
    
[^16]: 重新思考ASTE:一种极简的标记方案与对比学习

    Rethinking ASTE: A Minimalist Tagging Scheme Alongside Contrastive Learning

    [https://arxiv.org/abs/2403.07342](https://arxiv.org/abs/2403.07342)

    提出一种新颖的标记方案，并采用对比学习方法来重新思考ASTE，该方法在性能上优于最先进技术，同时具有更紧凑的设计和降低的计算开销，尤其在少样本学习情景下展现出优越效果。

    

    Aspect Sentiment Triplet Extraction (ASTE) 是细粒度情感分析的一个新兴子任务，旨在从非结构化文本数据中提取结构化的情感三元组。现有的ASTE方法通常通过额外的结构或外部数据来复杂化任务。在这项研究中，我们提出了一种新颖的标记方案，并采用对比学习方法来缓解这些挑战。所提出的方法在与最先进技术的比较中展示出可比甚至更优越的性能，同时具有更紧凑的设计和降低的计算开销。值得注意的是，在大语言模型(LLMs)时代，我们的方法在少样本学习情景下展现出优于GPT 3.5和GPT 4的效果。本研究还为在大语言模型范式下推进ASTE技术提供了宝贵的见解。

    arXiv:2403.07342v1 Announce Type: cross  Abstract: Aspect Sentiment Triplet Extraction (ASTE) is a burgeoning subtask of fine-grained sentiment analysis, aiming to extract structured sentiment triplets from unstructured textual data. Existing approaches to ASTE often complicate the task with additional structures or external data. In this research, we propose a novel tagging scheme and employ a contrastive learning approach to mitigate these challenges. The proposed approach demonstrates comparable or superior performance in comparison to state-of-the-art techniques, while featuring a more compact design and reduced computational overhead. Notably, even in the era of Large Language Models (LLMs), our method exhibits superior efficacy compared to GPT 3.5 and GPT 4 in a few-shot learning scenarios. This study also provides valuable insights for the advancement of ASTE techniques within the paradigm of large language models.
    
[^17]: PEEB：具有可解释和可编辑语言瓶颈的基于部分的图像分类器

    PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck

    [https://arxiv.org/abs/2403.05297](https://arxiv.org/abs/2403.05297)

    PEEB是一种基于部分的图像分类器，通过将类别名称转换为描述视觉部分的文本描述符，并将检测到的部分的嵌入与文本描述符匹配，从而在零样本设置中表现出色，并且不仅在监督学习中表现出色，而且还首次实现用户编辑类定义形成新分类器无需重新训练。

    

    基于CLIP的分类器依赖于包含{text encoder已知的类名称}的提示。也就是说，CLIP在新类别或其名称很少在互联网上出现的类别（例如鸟类的学名）上表现不佳。针对细粒度分类，我们提出了PEEB - 一种可解释和可编辑的分类器，用于（1）将类别名称表达为一组预定义的描述视觉部分的文本描述符；和（2）将检测到的部分的嵌入与每个类别中的文本描述符进行匹配，以计算用于分类的逻辑分数。在一个零样本设置中，其中类别名称是未知的，PEEB在准确性上大幅优于CLIP（约为10倍）。与基于部分的分类器相比，PEEB不仅在监督学习设置上是最先进的（88.80%准确率），而且还是第一个能够让用户编辑类定义以形成新的分类器而无需重新训练的分类器。

    arXiv:2403.05297v1 Announce Type: cross  Abstract: CLIP-based classifiers rely on the prompt containing a {class name} that is known to the text encoder. That is, CLIP performs poorly on new classes or the classes whose names rarely appear on the Internet (e.g., scientific names of birds). For fine-grained classification, we propose PEEB - an explainable and editable classifier to (1) express the class name into a set of pre-defined text descriptors that describe the visual parts of that class; and (2) match the embeddings of the detected parts to their textual descriptors in each class to compute a logit score for classification. In a zero-shot setting where the class names are unknown, PEEB outperforms CLIP by a large margin (~10x in accuracy). Compared to part-based classifiers, PEEB is not only the state-of-the-art on the supervised-learning setting (88.80% accuracy) but also the first to enable users to edit the class definitions to form a new classifier without retraining. Compar
    
[^18]: 大型语言模型能理解多目标口语语言吗？

    Do Large Language Model Understand Multi-Intent Spoken Language ?

    [https://arxiv.org/abs/2403.04481](https://arxiv.org/abs/2403.04481)

    该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。

    

    这项研究通过利用大型语言模型（LLMs）进行多目标口语语言理解（SLU）取得了重大进展，提出了一种在SLU环境中利用LLMs生成能力的独特方法。我们的创新技术重新配置了实体槽，专门用于LLMs在多目标SLU环境中的应用，并引入了子目标指令（SII）的概念，增强了对不同领域内复杂多目标交流的解剖和解释。由此产生的数据集，被称为LM-MixATIS和LM-MixSNIPS，是从现有基准中精心制作的。我们的研究表明，LLMs可以匹配并潜在地超越当前最先进的多目标SLU模型的能力。它进一步探讨了LLMs在各种意图配置和数据集比例下的有效性。此外，我们介绍了两个开创性的度量标准，即实体槽准确性（ESA）和Com

    arXiv:2403.04481v1 Announce Type: cross  Abstract: This study marks a significant advancement by harnessing Large Language Models (LLMs) for multi-intent spoken language understanding (SLU), proposing a unique methodology that capitalizes on the generative power of LLMs within an SLU context. Our innovative technique reconfigures entity slots specifically for LLM application in multi-intent SLU environments and introduces the concept of Sub-Intent Instruction (SII), enhancing the dissection and interpretation of intricate, multi-intent communication within varied domains. The resultant datasets, dubbed LM-MixATIS and LM-MixSNIPS, are crafted from pre-existing benchmarks. Our research illustrates that LLMs can match and potentially excel beyond the capabilities of current state-of-the-art multi-intent SLU models. It further explores LLM efficacy across various intent configurations and dataset proportions. Moreover, we introduce two pioneering metrics, Entity Slot Accuracy (ESA) and Com
    
[^19]: IRCoder: 中间表示使语言模型成为稳健的多语言代码生成器

    IRCoder: Intermediate Representations Make Language Models Robust Multilingual Code Generators

    [https://arxiv.org/abs/2403.03894](https://arxiv.org/abs/2403.03894)

    通过利用编译器中间表示来改进代码-LMs的多语言能力和促进跨语言转移。

    

    arXiv:2403.03894v1 公告类型: 新的 摘要: 代码理解和生成已迅速成为语言模型（LMs）最受欢迎的应用之一。然而，与自然语言LM的研究相比，对代码-LMs（即用于代码生成的LMs）的多语言方面的研究，如不同编程语言之间的跨语言转移，特定于语言的数据增强以及事后LM调整，以及利用原始文本内容之外的数据源，要稀少得多。特别是，大多数主流代码-LMs仅在源代码文件上进行了预训练。在这项工作中，我们研究了利用现成的编译器中间表示（跨编程语言共享）来改进代码-LMs的多语言能力并促进跨语言转移的前景。为此，我们首先编制了SLTrans，一个由近400万个自包含源代码文件组成的并行数据集。

    arXiv:2403.03894v1 Announce Type: new  Abstract: Code understanding and generation have fast become some of the most popular applications of language models (LMs). Nonetheless, research on multilingual aspects of Code-LMs (i.e., LMs for code generation) such as cross-lingual transfer between different programming languages, language-specific data augmentation, and post-hoc LM adaptation, alongside exploitation of data sources other than the original textual content, has been much sparser than for their natural language counterparts. In particular, most mainstream Code-LMs have been pre-trained on source code files alone. In this work, we investigate the prospect of leveraging readily available compiler intermediate representations - shared across programming languages - to improve the multilingual capabilities of Code-LMs and facilitate cross-lingual transfer.   To this end, we first compile SLTrans, a parallel dataset consisting of nearly 4M self-contained source code files coupled wi
    
[^20]: 推理过程中不是所有LLMs的层都是必要的

    Not all Layers of LLMs are Necessary during Inference

    [https://arxiv.org/abs/2403.02181](https://arxiv.org/abs/2403.02181)

    推理过程中，根据输入实例的不同难易程度，本文提出了一种名为AdaInfer的算法，可以自适应地使用浅层和深层，从而节省了计算资源。

    

    大型语言模型（LLMs）的推理阶段非常昂贵。理想的LLMs推理阶段可以利用更少的计算资源，同时仍保持其能力（例如泛化和上下文学习能力）。本文尝试回答一个问题：“在LLMs推理过程中，我们可以为简单实例使用浅层，并为难以处理的实例使用深层吗？”为了回答这个问题，我们首先通过统计分析跨任务激活的层来指出并非所有层在推理过程中都是必要的。然后，我们提出了一种简单的算法，名为AdaInfer，根据输入实例自适应地确定推理终止时刻。更重要的是，AdaInfer不改变LLMs参数，并在任务之间保持泛化能力。对知名LLMs（即Llama2系列和OPT）的实验证明，AdaInfer节省了平均14.8%的计算资源，甚至在情感方面高达50%。

    arXiv:2403.02181v1 Announce Type: cross  Abstract: The inference phase of Large Language Models (LLMs) is very expensive. An ideal inference stage of LLMs could utilize fewer computational resources while still maintaining its capabilities (e.g., generalization and in-context learning ability). In this paper, we try to answer the question, "During LLM inference, can we use shallow layers for easy instances; and deep layers for hard ones?" To answer this question, we first indicate that Not all Layers are Necessary during Inference by statistically analyzing the activated layers across tasks. Then, we propose a simple algorithm named AdaInfer to determine the inference termination moment based on the input instance adaptively. More importantly, AdaInfer does not alter LLM parameters and maintains generalizability across tasks. Experiments on well-known LLMs (i.e., Llama2 series and OPT) show that AdaInfer saves an average of 14.8% of computational resources, even up to 50% on sentiment 
    
[^21]: 评估和减少大规模视觉语言模型中的数字幻觉：一种一致性视角

    Evaluating and Mitigating Number Hallucinations in Large Vision-Language Models: A Consistency Perspective

    [https://arxiv.org/abs/2403.01373](https://arxiv.org/abs/2403.01373)

    本文评估了大型视觉语言模型中的数字幻觉问题，并提出一种一致性训练方法，可使数字幻觉得到显著改善

    

    大型视觉语言模型已经展示出在处理文本和视觉内容相关挑战方面的显著功效。然而，这些模型容易出现各种幻觉。本文关注一种新形式的幻觉，称为数字幻觉，指的是模型未能准确识别图像中物体的数量的情况。我们建立了一个数据集，并采用评估指标评估数字幻觉，揭示了这一问题在主流大型视觉语言模型(LVLMs)中的明显普遍性。此外，我们从两个相关视角深入分析了数字幻觉，考察了内在和外在的不一致问题。我们认为这种不一致性是数字幻觉的一个原因，并提出了一种一致性训练方法作为减轻此类幻觉的手段，该方法取得了8%的平均改进。

    arXiv:2403.01373v1 Announce Type: new  Abstract: Large vision language models have demonstrated remarkable efficacy in addressing challenges related to both textual and visual content. Nevertheless, these models are susceptible to various hallucinations. In this paper, we focus on a new form of hallucination, specifically termed as number hallucination, which denotes instances where models fail to accurately identify the quantity of objects in an image. We establish a dataset and employ evaluation metrics to assess number hallucination, revealing a pronounced prevalence of this issue across mainstream large vision language models (LVLMs). Additionally, we delve into a thorough analysis of number hallucination, examining inner and outer inconsistency problem from two related perspectives. We assert that this inconsistency is one cause of number hallucination and propose a consistency training method as a means to alleviate such hallucination, which achieves an average improvement of 8\%
    
[^22]: 面向实体的多模态对齐框架用于新闻图像字幕生成

    Entity-Aware Multimodal Alignment Framework for News Image Captioning

    [https://arxiv.org/abs/2402.19404](https://arxiv.org/abs/2402.19404)

    设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。

    

    新闻图像字幕生成任务是图像字幕生成任务的一个变体，要求模型生成一个更具信息性的字幕，其中包含新闻图像和相关新闻文章。近年来，多模态大型语言模型发展迅速，并在新闻图像字幕生成任务中表现出前景。然而，根据我们的实验，常见的多模态大型语言模型在零样本设定下生成实体方面表现不佳。即使在新闻图像字幕生成数据集上进行简单微调，它们处理实体信息的能力仍然有限。为了获得一个更强大的模型来处理多模态实体信息，我们设计了两个多模态实体感知对齐任务和一个对齐框架，以对齐模型并生成新闻图像字幕。我们的方法在GoodNews数据集上将CIDEr分数提高到86.29（从72.33），在NYTimes800k数据集上将其提高到85.61（从70.83），优于先前的最先进模型。

    arXiv:2402.19404v1 Announce Type: cross  Abstract: News image captioning task is a variant of image captioning task which requires model to generate a more informative caption with news image and the associated news article. Multimodal Large Language models have developed rapidly in recent years and is promising in news image captioning task. However, according to our experiments, common MLLMs are not good at generating the entities in zero-shot setting. Their abilities to deal with the entities information are still limited after simply fine-tuned on news image captioning dataset. To obtain a more powerful model to handle the multimodal entity information, we design two multimodal entity-aware alignment tasks and an alignment framework to align the model and generate the news image captions. Our method achieves better results than previous state-of-the-art models in CIDEr score (72.33 -> 86.29) on GoodNews dataset and (70.83 -> 85.61) on NYTimes800k dataset.
    
[^23]: Z-AGI实验室参与ClimateActivism 2024：社交媒体上立场和仇恨事件的检测

    Z-AGI Labs at ClimateActivism 2024: Stance and Hate Event Detection on Social Media

    [https://arxiv.org/abs/2402.17014](https://arxiv.org/abs/2402.17014)

    研究旨在在社交媒体上帮助气候活动人士识别和应对仇恨言论，团队评估了多种模型，在仇恨言论检测、仇恨言论目标识别和立场检测方面取得了有趣的结果。

    

    在数字领域，丰富的数据是洞察社会、政治和经济格局复杂性的关键信息来源。针对对事件高质量信息的日益增长需求以及打击仇恨言论的必要性，本研究发起了在CASE 2024举办的Climate Activism立场和仇恨事件检测共享任务的建立。聚焦气候活动人士在社交媒体上应对仇恨言论的情况，我们的研究为从Twitter推文中识别仇恨言论做出了贡献。通过分析三个子任务-仇恨言论检测（子任务A）、仇恨言论目标识别（子任务B）和立场检测（子任务C）- Z-AGI实验室团队评估了基于Tf-Idf的各种模型，包括LSTM、Xgboost和LGBM。结果显示出有趣的变化，Catboost在子任务B（F1：0.5604）和子任务C（F1：0.7081）中表现出色，而LGBM成为子任务A（F1：0.8684）中表现最佳的模型。

    arXiv:2402.17014v1 Announce Type: new  Abstract: In the digital realm, rich data serves as a crucial source of insights into the complexities of social, political, and economic landscapes. Addressing the growing need for high-quality information on events and the imperative to combat hate speech, this research led to the establishment of the Shared Task on Climate Activism Stance and Hate Event Detection at CASE 2024. Focused on climate activists contending with hate speech on social media, our study contributes to hate speech identification from tweets. Analyzing three sub-tasks - Hate Speech Detection (Sub-task A), Targets of Hate Speech Identification (Sub-task B), and Stance Detection (Sub-task C) - Team Z-AGI Labs evaluated various models, including LSTM, Xgboost, and LGBM based on Tf-Idf. Results unveiled intriguing variations, with Catboost excelling in Subtask-B (F1: 0.5604) and Subtask-C (F1: 0.7081), while LGBM emerged as the top-performing model for Subtask-A (F1: 0.8684). T
    
[^24]: Triad: 一个利用基于多角色LLM代理的框架来解决知识库问答问题

    Triad: A Framework Leveraging a Multi-Role LLM-based Agent to Solve Knowledge Base Question Answering

    [https://arxiv.org/abs/2402.14320](https://arxiv.org/abs/2402.14320)

    Triad框架利用了基于多角色LLM代理来解决知识库问答问题，通过代理的不同角色分别处理KBQA子任务，合作完成KBQA任务，并在多个基准数据集上表现出色。

    

    最近基于LLM代理的进展在各种任务中展现出了令人期待的结果。然而，它们在回答知识库中问题的运用仍然鲜为人知。使用传统方法来实现KBQA系统具有挑战性，因为缺乏特定任务训练数据以及创建以任务为中心的模型结构的复杂性。在本文中，我们提出了Triad，一个利用具有三个角色的LLM代理的统一框架来进行KBQA任务。代理被分配三个角色来处理不同的KBQA子任务：作为掌握各种子任务的通才，作为选择候选者的决策者，以及作为回答带有知识的问题的顾问。我们的KBQA框架在四个阶段中执行，涉及代理的多重角色的协作。我们使用三个基准数据集评估了我们框架的性能，结果显示我们的框架胜过了

    arXiv:2402.14320v1 Announce Type: cross  Abstract: Recent progress with LLM-based agents has shown promising results across various tasks. However, their use in answering questions from knowledge bases remains largely unexplored. Implementing a KBQA system using traditional methods is challenging due to the shortage of task-specific training data and the complexity of creating task-focused model structures. In this paper, we present Triad, a unified framework that utilizes an LLM-based agent with three roles for KBQA tasks. The agent is assigned three roles to tackle different KBQA subtasks: agent as a generalist for mastering various subtasks, as a decision maker for the selection of candidates, and as an advisor for answering questions with knowledge. Our KBQA framework is executed in four phases, involving the collaboration of the agent's multiple roles. We evaluated the performance of our framework using three benchmark datasets, and the results show that our framework outperforms 
    
[^25]: Self-AMPLIFY：通过自我事后解释改进小型语言模型

    Self-AMPLIFY: Improving Small Language Models with Self Post Hoc Explanations

    [https://arxiv.org/abs/2402.12038](https://arxiv.org/abs/2402.12038)

    本研究提出了Self-AMPLIFY方法，通过将事后解释方法应用于小型语言模型（SLMs），自动生成基于原因的解释，以提高它们自身的性能。

    

    在本论文中，我们提出了Self-AMPLIFY方法，该方法通过应用于小型语言模型（SLMs）的事后解释方法自动生成基于原因的解释，从而提高它们自身的性能。Self-AMPLIFY是一个3步骤的方法，用于选择样本、生成理由和构建最终提示以利用上下文学习（ICL）。我们在两个需要推理能力的SLMs和两个数据集上评估了Self-AMPLIFY的性能：这些实验表明Self-AMPLIFY在与竞争对手相比表现出色。Self-AMPLIFY是第一个将事后解释方法应用于SLMs的方法，以生成解释并提高它们自身性能的方法。

    arXiv:2402.12038v1 Announce Type: new  Abstract: Incorporating natural language rationales in the prompt and In-Context Learning (ICL) has led to a significant improvement of Large Language Models (LLMs) performance. However, rationales currently require human-annotation or the use of auxiliary proxy models to target promising samples or generate high-quality rationales. In this work, we propose Self-AMPLIFY to generate automatically rationales from post hoc explanation methods applied to Small Language Models (SLMs) to improve their own performance. Self-AMPLIFY is a 3-step method that targets samples, generates rationales and builds a final prompt to leverage ICL. Self-AMPLIFY performance is evaluated on two SLMs and two datasets requiring reasoning abilities: these experiments show that Self-AMPLIFY achieves good results against competitors. Self-AMPLIFY is the first method to apply post hoc explanation methods to SLM to generate rationales to improve their own performance in a full
    
[^26]: 用自然语言和概率推理进行实验与修订规则

    Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning

    [https://arxiv.org/abs/2402.06025](https://arxiv.org/abs/2402.06025)

    本论文建立了一个计算模型来模拟人们通过实验主动推断隐藏规则的过程，并发现显式假设、概率规则和在线更新的组合可以解释人们在类似Zendo任务上的表现。

    

    我们建立了一个计算模型，模拟人们通过实验主动推断隐藏规则的过程。该模型的基本原理是，即使规则是确定性的，学习者也会考虑更广泛的模糊概率规则，并用自然语言表示，根据近似贝叶斯原则在每次实验后在线更新自己的假设。在同一框架下，我们还根据信息论准则建立了实验设计模型。我们发现，这三个原则的组合——显式假设、概率规则和在线更新——可以解释人们在类似Zendo任务上的表现，而去掉其中任何一个组件都使得模型无法解释数据。

    We build a computational model of how humans actively infer hidden rules by doing experiments. The basic principles behind the model is that, even if the rule is deterministic, the learner considers a broader space of fuzzy probabilistic rules, which it represents in natural language, and updates its hypotheses online after each experiment according to approximately Bayesian principles. In the same framework we also model experiment design according to information-theoretic criteria. We find that the combination of these three principles -- explicit hypotheses, probabilistic rules, and online updates -- can explain human performance on a Zendo-style task, and that removing any of these components leaves the model unable to account for the data.
    
[^27]: L-TUNING：用于LLMs中的提示和前缀的同步标签调整

    L-TUNING: Synchronized Label Tuning for Prompt and Prefix in LLMs

    [https://arxiv.org/abs/2402.01643](https://arxiv.org/abs/2402.01643)

    本文介绍了L-Tuning，一种在自然语言推理（NLI）框架内的高效微调方法，通过对预训练的LLM中的标签标记进行微调，提高了训练准确性和效率，并增强了模型的训练细微差别。

    

    高效地针对特定任务对大型语言模型（LLMs）进行微调在自然语言处理中面临着重大挑战。传统方法，如提示或前缀调整，通常依赖于任意标记进行训练，从而导致训练时间延长并且通用标记在各种类别标签中使用。为了解决这些问题，本文引入了L-Tuning，这是一种在自然语言推理（NLI）框架内设计的用于分类任务的高效微调方法。与传统方法不同，L-Tuning专注于通过预训练的LLM处理的标签标记的微调，从而利用其预先存在的语义知识。这种技术不仅提高了微调的准确性和效率，还促进了为每个类别生成不同的标签嵌入，增强了模型的训练细微差别。我们的实验结果表明，使用L-Tuning可以显著提高训练效率和分类准确性。

    Efficiently fine-tuning Large Language Models (LLMs) for specific tasks presents a considerable challenge in natural language processing. Traditional methods, like prompt or prefix tuning, typically rely on arbitrary tokens for training, leading to prolonged training times and generalized token use across various class labels. To address these issues, this paper introduces L-Tuning, an efficient fine-tuning approach designed for classification tasks within the Natural Language Inference (NLI) framework. Diverging from conventional methods, L-Tuning focuses on the fine-tuning of label tokens processed through a pre-trained LLM, thereby harnessing its pre-existing semantic knowledge. This technique not only improves the fine-tuning accuracy and efficiency but also facilitates the generation of distinct label embeddings for each class, enhancing the model's training nuance. Our experimental results indicate a significant improvement in training efficiency and classification accuracy with 
    
[^28]: 小型大型语言模型在实际环境中的会议摘要中是否能够超越其体量？

    Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?

    [https://arxiv.org/abs/2402.00841](https://arxiv.org/abs/2402.00841)

    本文研究了在实际环境中的会议摘要任务，通过比较小型紧凑LLMs和零射击较大LLMs的性能，发现大多数小型LLMs无法超越较大的零射击LLMs，但FLAN-T5是一个例外，它的性能与许多零射击LLMs持平甚至更好。

    

    大型语言模型（LLMs）在没有明确针对任务特定数据集进行微调的情况下，展示了解决各种任务的出色能力。然而，在实际环境中部署LLMs并非易事，因为需要大量的计算资源。本文研究了在实际工业环境中的会议摘要任务，并通过比较经过微调的小型紧凑LLMs（如FLAN-T5、TinyLLaMA、LiteLLaMA）与零射击较大LLMs（如LLaMA-2、GPT-3.5、PaLM-2）的性能，进行了大量实验以解决利用LLMs在实际环境中的巨大成本问题。我们观察到，大多数小型LLMs，即使经过微调，也无法在会议摘要数据集中超越较大的零射击LLMs。然而，一个值得注意的例外是FLAN-T5（780M个参数），它的性能与许多零射击LLMs持平甚至更好。

    Large Language Models (LLMs) have demonstrated impressive capabilities to solve a wide range of tasks without being explicitly fine-tuned on task-specific datasets. However, deploying LLMs in the real world is not trivial, as it requires substantial computing resources. In this paper, we investigate whether smaller, compact LLMs are a good alternative to the comparatively Larger LLMs2 to address significant costs associated with utilizing LLMs in the real world. In this regard, we study the meeting summarization task in a real-world industrial environment and conduct extensive experiments by comparing the performance of fine-tuned compact LLMs (e.g., FLAN-T5, TinyLLaMA, LiteLLaMA) with zero-shot larger LLMs (e.g., LLaMA-2, GPT-3.5, PaLM-2). We observe that most smaller LLMs, even after fine-tuning, fail to outperform larger zero-shot LLMs in meeting summarization datasets. However, a notable exception is FLAN-T5 (780M parameters), which performs on par or even better than many zero-sho
    
[^29]: 通过收集轨迹和合成奖励来学习基于规划的推理

    Learning Planning-based Reasoning by Trajectories Collection and Process Reward Synthesizing

    [https://arxiv.org/abs/2402.00658](https://arxiv.org/abs/2402.00658)

    本文提出了一种通过直接偏好优化在收集到的轨迹上学习基于规划的推理的框架，以解决大型语言模型在复杂推理任务中的虚幻和缺陷问题。

    

    大型语言模型（LLM）通过逐步合理化生成，展示了处理复杂推理任务的重要潜力。然而，最近的研究对它们的推理过程中的虚幻和缺陷提出了担忧。为了提高生成合理化的可靠性和忠实性，正在进行大量工作。有些方法将推理建模为规划，而其他方法则专注于注释的过程监督。然而，基于规划的搜索过程往往由于频繁评估中间推理状态和广泛的探索空间而导致高延迟。此外，使用人工注释监督推理过程对于LLM训练来说是昂贵且具有挑战性的。为了解决这些问题，我们在本文中提出了一种通过直接偏好优化（DPO）来学习基于规划的推理的框架，其中轨迹直接根据合成的过程奖励进行排名。

    Large Language Models (LLMs) have demonstrated significant potential in handling complex reasoning tasks through step-by-step rationale generation. However, recent studies have raised concerns regarding the hallucination and flaws in their reasoning process. Substantial efforts are being made to improve the reliability and faithfulness of the generated rationales. Some approaches model reasoning as planning, while others focus on annotating for process supervision. Nevertheless, the planning-based search process often results in high latency due to the frequent assessment of intermediate reasoning states and the extensive exploration space. Additionally, supervising the reasoning process with human annotation is costly and challenging to scale for LLM training. To address these issues, in this paper, we propose a framework to learn planning-based reasoning through direct preference optimization (DPO) on collected trajectories, which are ranked according to synthesized process rewards. 
    
[^30]: PILOT: 使用法律案例预测案例结果

    PILOT: Legal Case Outcome Prediction with Case Law

    [https://arxiv.org/abs/2401.15770](https://arxiv.org/abs/2401.15770)

    提出了新的PILOT框架，旨在解决使用案例法预测法律案例结果的挑战，包括相关案例检索和时间模式处理两个模块。

    

    机器学习在预测法律案例结果方面表现出有希望的前景，但大多数研究集中在民事法律案例而非案例法系统上。我们确定了在使用案例法进行法律案例结果预测时的两个独特挑战。首先，识别作为法官在决策过程中基本证据的相关先例案例至关重要。其次，有必要考虑法律原则随时间演变的情况，因为早期案例可能遵循不同的法律背景。在本文中，我们提出了一个名为PILOT（法律案例结果预测）的新框架。它包括用于相关案例检索和时间模式处理的两个模块。为了衡量现有法律案例结果预测模型的性能，我们从大规模案例法数据库中策划了一个数据集。我们证明了准确识别先例案例的重要性以及缓解...

    arXiv:2401.15770v2 Announce Type: replace  Abstract: Machine learning shows promise in predicting the outcome of legal cases, but most research has concentrated on civil law cases rather than case law systems. We identified two unique challenges in making legal case outcome predictions with case law. First, it is crucial to identify relevant precedent cases that serve as fundamental evidence for judges during decision-making. Second, it is necessary to consider the evolution of legal principles over time, as early cases may adhere to different legal contexts. In this paper, we proposed a new framework named PILOT (PredictIng Legal case OuTcome) for case outcome prediction. It comprises two modules for relevant case retrieval and temporal pattern handling, respectively. To benchmark the performance of existing legal case outcome prediction models, we curated a dataset from a large-scale case law database. We demonstrate the importance of accurately identifying precedent cases and mitiga
    
[^31]: 通过代理调整语言模型

    Tuning Language Models by Proxy

    [https://arxiv.org/abs/2401.08565](https://arxiv.org/abs/2401.08565)

    介绍了一种代理调整的轻量级解码时算法，可以通过对小型调整后的LM的预测与未调整LM的预测之间的差异来调整大型预训练LM的预测，从而实现资源节约和保留更大规模预训练的好处。

    

    尽管大型预训练语言模型具有一般的能力，但它们始终受益于进一步调整以更好地实现所需的行为。然而，调整这些模型变得越来越消耗资源，或者在模型权重是私有的情况下是不可能的。我们引入了代理调整，这是一种轻量级的解码时算法，它在黑盒语言模型的基础上运行，以实现与直接调整相同的目的，但只访问其在输出词汇上的预测，而不是其参数。我们的方法调整了一个较小的语言模型，然后将经过调整和未经调整的小模型的预测之间的差异应用于将更大的未调整模型的原始预测转移到调整方向，同时保留较大规模预训练的好处。在实验中，当我们使用仅为7B大小的代理对Llama2-70B应用代理调整时，我们可以关闭88% Llama2-70B 与其真正调整过的聊天版本之间的差距，

    arXiv:2401.08565v2 Announce Type: replace  Abstract: Despite the general capabilities of large pretrained language models, they consistently benefit from further adaptation to better achieve desired behaviors. However, tuning these models has become increasingly resource-intensive, or impossible when model weights are private. We introduce proxy-tuning, a lightweight decoding-time algorithm that operates on top of black-box LMs to achieve the same end as direct tuning, but by accessing only its predictions over the output vocabulary, not its parameters. Our method tunes a smaller LM, then applies the difference between the predictions of the small tuned and untuned LMs to shift the original predictions of the larger untuned model in the direction of tuning, while retaining the benefits of larger-scale pretraining. In experiments, when we apply proxy-tuning to Llama2-70B using proxies of only 7B size, we can close 88% of the gap between Llama2-70B and its truly-tuned chat version, when 
    
[^32]: MAPO：通过多语言对齐作为偏好优化推进多语言推理

    MAPO: Advancing Multilingual Reasoning through Multilingual Alignment-as-Preference Optimization

    [https://arxiv.org/abs/2401.06838](https://arxiv.org/abs/2401.06838)

    MAPO提出了一个多语言对齐作为偏好优化框架，可以显著提高各种模型在多语言推理中的表现。

    

    尽管推理能力被认为与语言无关，但现有的多语言语言模型在不同语言中表现出的推理能力不一致，例如，对主导语言（如英语）的推理能力优于其他语言，这是由于多语言训练数据的不平衡造成的。为了增强非主导语言的推理能力，我们提出了一个名为多语言对齐作为偏好优化（MAPO）的框架，旨在将其他语言中的推理过程与主导语言对齐。具体来说，我们利用现成的翻译模型来保证非主导语言和主导语言之间答案的一致性，这被采纳为优化的偏好，例如，直接偏好优化（DPO）或临近策略优化（PPO）。实验表明，MAPO在所有三个基准测试中（MSVAMP +16.2％，MGSM +6.1％）稳定地实现了各种模型的多语言推理显著改进。

    arXiv:2401.06838v2 Announce Type: replace  Abstract: Though reasoning abilities are considered language-agnostic, existing LLMs exhibit inconsistent reasoning abilities across different languages, e.g., reasoning in the dominant language like English is superior to other languages due to the imbalance of multilingual training data. To enhance reasoning abilities in non-dominant languages, we propose a Multilingual-Alignment-as-Preference Optimization framework (MAPO), aiming to align the reasoning processes in other languages with the dominant language. Specifically, we harness an off-the-shelf translation model for the consistency between answers in non-dominant and dominant languages, which we adopt as the preference for optimization, e.g., Direct Preference Optimization (DPO) or Proximal Policy Optimization (PPO). Experiments show that MAPO stably achieves significant improvements in the multilingual reasoning of various models on all three benchmarks (MSVAMP +16.2%, MGSM +6.1%, and
    
[^33]: 用于盖兹语的机器翻译

    Machine Translation for Ge'ez Language

    [https://arxiv.org/abs/2311.14530](https://arxiv.org/abs/2311.14530)

    该研究探讨了改善盖兹语机器翻译的方法，包括从相关语言进行迁移学习、优化共享词汇和标记分割方法、大型预训练模型的微调，以及在少样本情况下使用大型语言模型进行翻译，其中一种基于语言相关性的多语言神经机器翻译模型相比标准双语模型有4 BLEU的平均性能提升。

    

    arXiv:2311.14530v2 公告类型: 替换 摘要: 机器翻译(MT)用于低资源语言，如盖兹语，一种古老的语言，不再是任何社区的母语，面临诸如词汇外生、领域不匹配和缺乏足够标记的训练数据等挑战。在这项工作中，我们探讨了改善盖兹语MT的各种方法，包括从相关语言进行迁移学习、优化共享词汇和标记分割方法、微调大型预训练模型，以及使用大型语言模型(LLMs)进行少样本翻译配合模糊匹配。我们基于语言相关性开发了一种多语言神经机器翻译(MNMT)模型，使标准双语模型的平均性能提高约4 BLEU。我们还尝试微调NLLB-200模型，这是当今可用的最先进的翻译模型之一，但发现只有4k训练样本的盖兹语表现不佳。

    arXiv:2311.14530v2 Announce Type: replace  Abstract: Machine translation (MT) for low-resource languages such as Ge'ez, an ancient language that is no longer the native language of any community, faces challenges such as out-of-vocabulary words, domain mismatches, and lack of sufficient labeled training data. In this work, we explore various methods to improve Ge'ez MT, including transfer-learning from related languages, optimizing shared vocabulary and token segmentation approaches, finetuning large pre-trained models, and using large language models (LLMs) for few-shot translation with fuzzy matches. We develop a multilingual neural machine translation (MNMT) model based on languages relatedness, which brings an average performance improvement of about 4 BLEU compared to standard bilingual models. We also attempt to finetune the NLLB-200 model, one of the most advanced translation models available today, but find that it performs poorly with only 4k training samples for Ge'ez. Furthe
    
[^34]: 大型语言模型的心理测量预测能力

    Psychometric Predictive Power of Large Language Models

    [https://arxiv.org/abs/2311.07484](https://arxiv.org/abs/2311.07484)

    指令调整和提示在大型语言模型中无法提供比基础模型更好的认知建模估计。

    

    arXiv:2311.07484v2 公告类型: 替换-交叉 摘要:指令调整使大型语言模型（LLMs）的响应与人类偏好一致。尽管在人-LLM对齐方面进行了努力，我们报告了一个有趣的发现，即指令调整并不总是使LLMs从认知建模的角度看起来更像人类。更具体地说，由指令调整的LLM估计的下一个词概率往往比基础LLM估计的概率更糟糕，无法模拟人类阅读行为。此外，我们探讨了使用LLMs模拟人类阅读行为的提示方法。我们的结果表明，反映特定语言假设的提示可以提高PPP，但仍不及小型基础模型的PPP。这些发现突显出LLMs最近的进展，即指令调整和提示，并不能提供比基础LLMs直接概率测量更好的认知建模估计。换句话说，我们的实验表明，纯粹的下一个词概率

    arXiv:2311.07484v2 Announce Type: replace-cross  Abstract: Instruction tuning aligns the response of large language models (LLMs) with human preferences. Despite such efforts in human--LLM alignment, we report that, interestingly, instruction tuning does not always make LLMs human-like from a cognitive modeling perspective. More specifically, next-word probabilities estimated by instruction-tuned LLMs are often worse at simulating human reading behavior than those estimated by base LLMs. In addition, we explore prompting methodologies in simulating human reading behavior with LLMs. Our results show that prompts reflecting a particular linguistic hypothesis improve PPP but are still inferior to PPP from small base models. These findings highlight that recent advancements in LLMs, i.e., instruction tuning and prompting, do not offer better estimates than direct probability measurements from base LLMs in cognitive modeling. In other words, our experiments highlight that pure next-word pro
    
[^35]: 火焰: 评估中国大型语言模型与人类价值观的契合性的基准测试

    Flames: Benchmarking Value Alignment of Chinese Large Language Models

    [https://arxiv.org/abs/2311.06899](https://arxiv.org/abs/2311.06899)

    中国的大型语言模型的价值观契合性需要更加全面的评估，该研究提出了一个名为"火焰"（Flames）的基准测试，涵盖了常见的无害原则和特定中国价值观，以及复杂场景和隐含恶意的提示方法。

    

    大型语言模型（LLMs）在各个地区的广泛应用强调了评估它们与人类价值观契合性的迫切性。然而，当前的基准测试未能有效地揭示LLMs中的安全漏洞。尽管许多模型在这些评估中得分很高，且“名列前茅”，但在LLMs与人类价值观的深层契合性和实现真正无害方面仍存在重大差距。为此，本文提出了一个名为"火焰"（Flames）的价值观契合性基准测试，该测试涵盖了常见的无害原则，以及一个整合了特定中国价值观如和谐的独特道德维度。因此，我们精心设计了包含复杂情境和大多带有隐含恶意的破解方法的对抗性提示。通过对17个主流LLMs进行提示，我们获得了模型的回应，并对其进行了详细评估。

    arXiv:2311.06899v2 Announce Type: replace-cross  Abstract: The widespread adoption of large language models (LLMs) across various regions underscores the urgent need to evaluate their alignment with human values. Current benchmarks, however, fall short of effectively uncovering safety vulnerabilities in LLMs. Despite numerous models achieving high scores and 'topping the chart' in these evaluations, there is still a significant gap in LLMs' deeper alignment with human values and achieving genuine harmlessness. To this end, this paper proposes a value alignment benchmark named Flames, which encompasses both common harmlessness principles and a unique morality dimension that integrates specific Chinese values such as harmony. Accordingly, we carefully design adversarial prompts that incorporate complex scenarios and jailbreaking methods, mostly with implicit malice. By prompting 17 mainstream LLMs, we obtain model responses and rigorously annotate them for detailed evaluation. Our findin
    
[^36]: 序列级确定性降低基于知识的对话生成中的虚构情况

    Sequence-Level Certainty Reduces Hallucination In Knowledge-Grounded Dialogue Generation

    [https://arxiv.org/abs/2310.18794](https://arxiv.org/abs/2310.18794)

    序列级确定性是减少基于知识的对话生成中幻觉的关键，这项工作提出了基于概率确定性和语义确定性的序列级确定性，结果表明更高水平的确定性对应更低水平的幻觉，进一步提出了基于确定性的响应排序方法

    

    在这项工作中，我们提出了序列级确定性作为基于知识的对话生成中虚构情况的共同主题。我们探讨了幻觉水平与两种序列级确定性之间的相关性：概率确定性和语义确定性。实证结果表明，模型响应中两种序列级确定性水平的提高与虚构水平的降低相关。我们进一步提出了基于确定性的响应排序（CRR），这是一种解码时的幻觉缓解方法，根据它们的序列级确定性对响应候选进行排序，并输出具有最高确定性水平的答案。与我们关于序列级确定性的定义相一致，我们设计了两种CRR方法：概率CRR（P-CRR）和语义CRR（S-CRR）。P-CRR使用整个序列的算术平均对各个单独抽样的模型响应进行排名

    arXiv:2310.18794v2 Announce Type: replace-cross  Abstract: In this work, we propose sequence-level certainty as a common theme over hallucination in Knowledge Grounded Dialogue Generation (KGDG). We explore the correlation between the level of hallucination and two types of sequence-level certainty: probabilistic certainty and semantic certainty. Empirical results reveal that a higher level of both types of sequence-level certainty in model responses is correlated with a lower level of hallucination. We further propose Certainty-based Response Ranking (CRR), a decoding-time hallucination mitigation method that ranks response candidates based on their sequence-level certainty and outputs the answer with the highest certainty level. Aligning with our definitions of sequence-level certainty, we design 2 types of CRR approaches: Probabilistic CRR (P-CRR) and Semantic CRR (S-CRR). P-CRR ranks individually sampled model responses using the arithmetic mean log-probability of the entire sequen
    
[^37]: SQLformer：深度自回归查询图生成用于文本到SQL翻译

    SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation

    [https://arxiv.org/abs/2310.18376](https://arxiv.org/abs/2310.18376)

    SQLformer是一个用于文本到SQL翻译的深度自回归查询图生成模型，采用了特定的Transformer架构，并通过结构归纳偏差解决领域泛化和自然语言与SQL查询对齐的难题。

    

    近年来，对于文本到SQL翻译的兴趣日益增长，这是将自然语言问题转化为可执行SQL查询的任务。这项技术具有潜在的潜力，可以使数据库中的数据提取民主化。然而，其中一些主要障碍包括领域泛化，即适应以前未见到的数据库，并且将自然语言问题与相应的SQL查询对齐。为了克服这些挑战，我们引入了SQLformer，这是一种针对执行文本到SQL翻译任务而设计的新型Transformer体系结构。我们的模型以自回归的方式预测SQL查询，并在编码器和解码器层中结合结构归纳偏差。这种偏差是由数据库表和列选择引导的，有助于解码器以广度优先搜索的规范顺序生成SQL查询的图形表示。全面的实验说明了现阶段的技术水平

    In recent years, there has been growing interest in text-to-SQL translation, which is the task of converting natural language questions into executable SQL queries. This technology is important for its potential to democratize data extraction from databases. However, some of its key hurdles include domain generalisation, which is the ability to adapt to previously unseen databases, and alignment of natural language questions with the corresponding SQL queries. To overcome these challenges, we introduce SQLformer, a novel Transformer architecture specifically crafted to perform text-to-SQL translation tasks. Our model predicts SQL queries as abstract syntax trees (ASTs) in an autoregressive way, incorporating structural inductive bias in the encoder and decoder layers. This bias, guided by database table and column selection, aids the decoder in generating SQL query ASTs represented as graphs in a Breadth-First Search canonical order. Comprehensive experiments illustrate the state-of-th
    
[^38]: 评估大型语言模型的空间理解能力

    Evaluating Spatial Understanding of Large Language Models

    [https://arxiv.org/abs/2310.14540](https://arxiv.org/abs/2310.14540)

    本研究评估了大型语言模型对空间结构的理解能力，发现LLMs在表示和推理空间结构时的表现存在显著差异，具有捕捉空间结构隐含特征的潜力。

    

    大型语言模型 (LLMs) 在各种任务中展现出卓越的能力。尽管这些模型在训练中只看到文本，但一些最近的研究表明，LLM表示隐含地捕捉了地面概念的几个方面。在这里，我们探讨了LLM表示对一种特别显著的基础知识 -- 空间关系的表现。我们设计了自然语言导航任务，评估了LLMs，特别是GPT-3.5-turbo、GPT-4 和 Llama2 系列模型，表示和推理空间结构的能力。这些任务揭示了LLM在不同空间结构 (包括正方形、六边形和三角形格、环和树) 上的性能差异。在广泛的错误分析中，我们发现LLMs的错误反映了空间和非空间因素。这些发现表明，LLMs似乎隐含地捕捉了空间结构的某些方面，但仍有提升的空间。

    arXiv:2310.14540v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) show remarkable capabilities across a variety of tasks. Despite the models only seeing text in training, several recent studies suggest that LLM representations implicitly capture aspects of the underlying grounded concepts. Here, we explore LLM representations of a particularly salient kind of grounded knowledge -- spatial relationships. We design natural-language navigation tasks and evaluate the ability of LLMs, in particular GPT-3.5-turbo, GPT-4, and Llama2 series models, to represent and reason about spatial structures. These tasks reveal substantial variability in LLM performance across different spatial structures, including square, hexagonal, and triangular grids, rings, and trees. In extensive error analysis, we find that LLMs' mistakes reflect both spatial and non-spatial factors. These findings suggest that LLMs appear to capture certain aspects of spatial structure implicitly, but room f
    
[^39]: 能够检测到LLM生成的虚假信息吗?

    Can LLM-Generated Misinformation Be Detected?

    [https://arxiv.org/abs/2309.13788](https://arxiv.org/abs/2309.13788)

    LLM生成的虚假信息可能比人类撰写的虚假信息更难以检测，具有更具欺骗性的风格，可能造成更多危害。

    

    大型语言模型（LLMs）的出现产生了深远影响。然而，LLMs（如ChatGPT）可能被利用来生成虚假信息，这给在线安全和公众信任带来了严重关切。一个基本的研究问题是：LLM生成的虚假信息是否会比人类撰写的虚假信息造成更大危害?我们提出从检测难度的角度来探讨这个问题。我们首先建立了一个LLM生成的虚假信息分类法。然后，我们对利用LLMs生成虚假信息的潜在真实世界方法进行分类和验证。通过广泛的实证调查，我们发现与具有相同语义的人类撰写的虚假信息相比，LLM生成的虚假信息对人类和检测器来说更难检测，这表明它可能具有更具欺骗性的风格，潜在地造成更多危害。我们还讨论了我们发现的影响。

    arXiv:2309.13788v3 Announce Type: replace-cross  Abstract: The advent of Large Language Models (LLMs) has made a transformative impact. However, the potential that LLMs such as ChatGPT can be exploited to generate misinformation has posed a serious concern to online safety and public trust. A fundamental research question is: will LLM-generated misinformation cause more harm than human-written misinformation? We propose to tackle this question from the perspective of detection difficulty. We first build a taxonomy of LLM-generated misinformation. Then we categorize and validate the potential real-world methods for generating misinformation with LLMs. Then, through extensive empirical investigation, we discover that LLM-generated misinformation can be harder to detect for humans and detectors compared to human-written misinformation with the same semantics, which suggests it can have more deceptive styles and potentially cause more harm. We also discuss the implications of our discovery
    
[^40]: 跨语言结构提取的情境标签投影

    Contextual Label Projection for Cross-Lingual Structure Extraction

    [https://arxiv.org/abs/2309.08943](https://arxiv.org/abs/2309.08943)

    本文提出了一种新的标签投影方法CLaP，利用上下文翻译标签，确保更准确的翻译，并在跨语言结构预测任务中取得显著表现。

    

    标签投影涉及同时获取翻译标签和文本，对于利用机器翻译促进结构预测任务中的跨语言转移至关重要。先前的研究探索标签投影时通常通过偏爱简化标签翻译或仅依赖于单词级别对齐来牺牲翻译准确性。本文介绍了一种新颖的标签投影方法CLaP，将文本翻译到目标语言，并利用已翻译文本作为上下文对标签进行情境翻译，确保翻译标签的更好准确性。我们利用具有多语言能力的指导调校语言模型作为我们的情境翻译器，通过指导在翻译文本中存在翻译标签的约束。我们在39种语言上通过零-shot跨语言转移对CLaP与其他标签投影技术进行基准测试。

    arXiv:2309.08943v2 Announce Type: replace  Abstract: Label projection, which involves obtaining translated labels and texts jointly, is essential for leveraging machine translation to facilitate cross-lingual transfer in structured prediction tasks. Prior research exploring label projection often compromise translation accuracy by favoring simplified label translation or relying solely on word-level alignments. In this paper, we introduce a novel label projection approach, CLaP, which translates text to the target language and performs contextual translation on the labels using the translated text as the context, ensuring better accuracy for the translated labels. We leverage instruction-tuned language models with multilingual capabilities as our contextual translator, imposing the constraint of the presence of translated labels in the translated text via instructions. We benchmark CLaP with other label projection techniques on zero-shot cross-lingual transfer across 39 languages on tw
    
[^41]: H2O-Danube-1.8B 技术报告

    H2O-Danube-1.8B Technical Report. (arXiv:2401.16818v1 [cs.CL])

    [http://arxiv.org/abs/2401.16818](http://arxiv.org/abs/2401.16818)

    H2O-Danube-1.8B 是一个在 1T 个标记上训练的 18 亿语言模型，具有高度竞争力的指标。同时，他们还发布了一个经过微调和优化训练的聊天模型，进一步推动语言模型的经济民主化。

    

    我们介绍了 H2O-Danube-1.8B，这是一个在 1T 个标记上训练的 18 亿语言模型，遵循 LLama 2 和 Mistral 的核心原则。我们利用和改进了各种大规模语言模型预训练的技术。尽管我们的模型训练所使用的总标记数量明显少于相似规模的参考模型，但它在众多基准测试中展现出了高度竞争力的指标。我们还发布了一个经过监督微调和直接偏好优化训练的聊天模型。我们以 Apache 2.0 许可证将 H2O-Danube-1.8B 开放，进一步推动 LLMs 的经济民主化，让更广泛的受众受益。

    We present H2O-Danube-1.8B, a 1.8B language model trained on 1T tokens following the core principles of LLama 2 and Mistral. We leverage and refine various techniques for pre-training large language models. Although our model is trained on significantly fewer total tokens compared to reference models of similar size, it exhibits highly competitive metrics across a multitude of benchmarks. We additionally release a chat model trained with supervised fine-tuning followed by direct preference optimization. We make H2O-Danube-1.8B openly available under Apache 2.0 license further democratizing LLMs to a wider audience economically.
    
[^42]: 能量的梯度流：实体对齐解码的通用高效方法

    Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding. (arXiv:2401.12798v1 [cs.IR])

    [http://arxiv.org/abs/2401.12798](http://arxiv.org/abs/2401.12798)

    这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。

    

    实体对齐（EA）是在集成多源知识图谱（KGs）中的关键过程，旨在识别这些图谱中的等价实体对。大多数现有方法将EA视为图表示学习任务，专注于增强图编码器。然而，在EA中解码过程-对于有效的操作和对齐准确性至关重要-得到了有限的关注，并且仍然针对特定数据集和模型架构进行定制，需要实体和额外的显式关系嵌入。这种特殊性限制了它的适用性，尤其是在基于GNN的模型中。为了填补这一空白，我们引入了一种新颖、通用和高效的EA解码方法，仅依赖于实体嵌入。我们的方法通过最小化狄利克雷能量来优化解码过程，在图内引导梯度流，以促进图同质性。梯度流的离散化产生了一种快速可扩展的方法，称为三元特征。

    Entity alignment (EA), a pivotal process in integrating multi-source Knowledge Graphs (KGs), seeks to identify equivalent entity pairs across these graphs. Most existing approaches regard EA as a graph representation learning task, concentrating on enhancing graph encoders. However, the decoding process in EA - essential for effective operation and alignment accuracy - has received limited attention and remains tailored to specific datasets and model architectures, necessitating both entity and additional explicit relation embeddings. This specificity limits its applicability, particularly in GNN-based models. To address this gap, we introduce a novel, generalized, and efficient decoding approach for EA, relying solely on entity embeddings. Our method optimizes the decoding process by minimizing Dirichlet energy, leading to the gradient flow within the graph, to promote graph homophily. The discretization of the gradient flow produces a fast and scalable approach, termed Triple Feature
    
[^43]: 通过交流使LLM代理适应环境的方法

    Adapting LLM Agents Through Communication. (arXiv:2310.01444v1 [cs.CL])

    [http://arxiv.org/abs/2310.01444](http://arxiv.org/abs/2310.01444)

    这项研究提出了一种名为学习通信（LTC）的训练方法，使用该方法可使LLM代理通过与环境和其他代理的交互不断改进，以适应新任务，而无需过多人类监督。

    

    最近大语言模型(LLM)的进展显示出了人类化代理的潜力。为了帮助这些代理在没有广泛人类监督的情况下适应新任务，我们提出了学习通信（LTC）范式，这是一种新颖的训练方法，使LLM代理能够通过与环境和其他代理的交互不断改进。通过迭代探索和PPO训练，LTC使代理能够将短期经验融入长期记忆。为了优化特定任务的代理交互，我们引入了三种结构化的通信模式：独白，对话，

    Recent advancements in large language models (LLMs) have shown potential for human-like agents. To help these agents adapt to new tasks without extensive human supervision, we propose the Learning through Communication (LTC) paradigm, a novel training approach enabling LLM agents to improve continuously through interactions with their environments and other agents. Recent advancements in large language models (LLMs) have shown potential for human-like agents. To help these agents adapt to new tasks without extensive human supervision, we propose the Learning through Communication (LTC) paradigm, a novel training approach enabling LLM agents to improve continuously through interactions with their environments and other agents. Through iterative exploration and PPO training, LTC empowers the agent to assimilate short-term experiences into long-term memory. To optimize agent interactions for task-specific learning, we introduce three structured communication patterns: Monologue, Dialogue,
    
[^44]: BooookScore: LLM时代中对书籍长度摘要的系统探索

    BooookScore: A systematic exploration of book-length summarization in the era of LLMs. (arXiv:2310.00785v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00785](http://arxiv.org/abs/2310.00785)

    本文对LLM模型进行了系统探索，以解决对超过上下文窗口大小的书籍进行摘要的问题，并通过两种提示工作流实施了基于LLM的书籍长度摘要器的连贯性研究。通过对100本书的GPT-4生成摘要的人工注释，发现了八种常见的连贯性错误。

    

    对于超过大型语言模型（LLMs）上下文窗口大小的书籍长度文档（>100K标记）进行摘要需要首先将输入文档分成较小的块，然后提示LLM合并、更新和压缩块级摘要。尽管这个任务的复杂性和重要性，但由于评估的困难，它尚未得到有意义的研究：现有的书籍长度摘要数据集（例如BookSum）在大多数公共LLM的预训练数据中，而现有的评估方法难以捕捉现代LLM摘要器的错误。在本文中，我们首次研究通过两种提示工作流实施的基于LLM的书籍长度摘要器的连贯性：（1）分层合并块级摘要，（2）逐步更新一个运行摘要。我们对100本最近出版的书籍的GPT-4生成摘要获得了1193个细粒度的人工注释，并确定了LLMs产生的八种常见的连贯性错误。

    Summarizing book-length documents (>100K tokens) that exceed the context window size of large language models (LLMs) requires first breaking the input document into smaller chunks and then prompting an LLM to merge, update, and compress chunk-level summaries. Despite the complexity and importance of this task, it has yet to be meaningfully studied due to the challenges of evaluation: existing book-length summarization datasets (e.g., BookSum) are in the pretraining data of most public LLMs, and existing evaluation methods struggle to capture errors made by modern LLM summarizers. In this paper, we present the first study of the coherence of LLM-based book-length summarizers implemented via two prompting workflows: (1) hierarchically merging chunk-level summaries, and (2) incrementally updating a running summary. We obtain 1193 fine-grained human annotations on GPT-4 generated summaries of 100 recently-published books and identify eight common types of coherence errors made by LLMs. Bec
    
[^45]: 重塑顺序推荐系统：利用内容增强语言建模学习动态用户兴趣

    Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling. (arXiv:2309.10435v1 [cs.IR])

    [http://arxiv.org/abs/2309.10435](http://arxiv.org/abs/2309.10435)

    本研究提出了一个新的顺序推荐范式 LANCER，利用预训练语言模型的语义理解能力生成更加人性化的个性化推荐。在多个基准数据集上的实验结果表明，该方法有效且有希望，并为了解顺序推荐的影响提供了有价值的见解。

    

    推荐系统对在线应用至关重要，而顺序推荐由于其表达能力强大，能够捕捉到动态用户兴趣而广泛使用。然而，先前的顺序建模方法在捕捉上下文信息方面仍存在局限性。主要的原因是语言模型常常缺乏对领域特定知识和物品相关文本内容的理解。为了解决这个问题，我们采用了一种新的顺序推荐范式，并提出了LANCER，它利用预训练语言模型的语义理解能力生成个性化推荐。我们的方法弥合了语言模型与推荐系统之间的差距，产生了更加人性化的推荐。通过对多个基准数据集上的实验，我们验证了我们的方法的有效性，展示了有希望的结果，并提供了对我们模型对顺序推荐的影响的有价值的见解。

    Recommender systems are essential for online applications, and sequential recommendation has enjoyed significant prevalence due to its expressive ability to capture dynamic user interests. However, previous sequential modeling methods still have limitations in capturing contextual information. The primary reason for this issue is that language models often lack an understanding of domain-specific knowledge and item-related textual content. To address this issue, we adopt a new sequential recommendation paradigm and propose LANCER, which leverages the semantic understanding capabilities of pre-trained language models to generate personalized recommendations. Our approach bridges the gap between language models and recommender systems, resulting in more human-like recommendations. We demonstrate the effectiveness of our approach through experiments on several benchmark datasets, showing promising results and providing valuable insights into the influence of our model on sequential recomm
    
[^46]: 通过隐式推理理解语言模型中的灾难性遗忘

    Understanding Catastrophic Forgetting in Language Models via Implicit Inference. (arXiv:2309.10105v1 [cs.CL])

    [http://arxiv.org/abs/2309.10105](http://arxiv.org/abs/2309.10105)

    本研究通过在语言模型上进行实验，发现微调对模型在微调数据分布任务上的表现有正面影响，但会抑制模型在其他任务上的能力，特别是与微调分布最接近的任务。作者假设语言模型会隐式推理任务，并且微调过程偏向于微调数据分布中的任务。作者进一步提出了共轭提示方法，以尝试恢复模型在预训练阶段的能力。

    

    微调（通过指令微调或从人类反馈进行强化学习等方法）是训练语言模型以鲁棒地执行所需任务的关键步骤。然而，我们缺乏对微调的影响的系统理解，特别是在狭窄的微调分布之外的任务上。在一个简化的场景中，我们证明，在微调数据分布内提高任务表现的同时，会抑制模型在其他任务上的能力。这种退化在与微调分布“最接近”的任务中尤为显著。我们假设语言模型会隐式推理出与提示相对应的任务，并且微调过程主要偏向于微调分布中的任务，以测试这个假设，我们提出了共轭提示以查看是否可以恢复预训练的能力。共轭提示会人为地使任务看起来与微调分布较远。

    Fine-tuning (via methods such as instruction-tuning or reinforcement learning from human feedback) is a crucial step in training language models to robustly carry out tasks of interest. However, we lack a systematic understanding of the effects of fine-tuning, particularly on tasks outside the narrow fine-tuning distribution. In a simplified scenario, we demonstrate that improving performance on tasks within the fine-tuning data distribution comes at the expense of suppressing model capabilities on other tasks. This degradation is especially pronounced for tasks "closest" to the fine-tuning distribution. We hypothesize that language models implicitly infer the task of the prompt corresponds, and the fine-tuning process predominantly skews this task inference towards tasks in the fine-tuning distribution. To test this hypothesis, we propose Conjugate Prompting to see if we can recover pretrained capabilities. Conjugate prompting artificially makes the task look farther from the fine-tun
    
[^47]: 在跨语言转移范式中测量灾难性遗忘：探索调优策略

    Measuring Catastrophic Forgetting in Cross-Lingual Transfer Paradigms: Exploring Tuning Strategies. (arXiv:2309.06089v1 [cs.CL])

    [http://arxiv.org/abs/2309.06089](http://arxiv.org/abs/2309.06089)

    该研究比较了不同的微调和跨语言转移策略在解决跨语言任务时的表现，评估了灾难性遗忘的程度和转移的成功程度。

    

    跨语言转移是一种解决资源匮乏语言任务的有希望的技术。在这个实证研究中，我们比较了两种与零射和全射学习方法相结合的大型语言模型在跨语言设置下的微调方法。作为微调策略，我们比较了参数效率适配器方法与所有参数微调。作为跨语言转移策略，我们比较了使用每个语言依次的中间训练（IT）和在微调的验证阶段已经使用目标语言的跨语言验证（CLV）。我们评估了转移的成功程度以及源语言中由于跨语言转移而导致的灾难性遗忘的程度，即在学习不同语言中的新信息时之前获得的知识损失了多少。在两个不同的分类问题上，包括仇恨言论检测和产品评论，分别包含了多个语种数据集的结果。

    The cross-lingual transfer is a promising technique to solve tasks in less-resourced languages. In this empirical study, we compare two fine-tuning approaches combined with zero-shot and full-shot learning approaches for large language models in a cross-lingual setting. As fine-tuning strategies, we compare parameter-efficient adapter methods with fine-tuning of all parameters. As cross-lingual transfer strategies, we compare the intermediate-training (\textit{IT}) that uses each language sequentially and cross-lingual validation (\textit{CLV}) that uses a target language already in the validation phase of fine-tuning. We assess the success of transfer and the extent of catastrophic forgetting in a source language due to cross-lingual transfer, i.e., how much previously acquired knowledge is lost when we learn new information in a different language. The results on two different classification problems, hate speech detection and product reviews, each containing datasets in several lang
    
[^48]: 大型语言模型作为优化器

    Large Language Models as Optimizers. (arXiv:2309.03409v1 [cs.LG])

    [http://arxiv.org/abs/2309.03409](http://arxiv.org/abs/2309.03409)

    本论文提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，通过自然语言描述优化任务。经过实验证明，该方法在线性回归和旅行推销员问题上表现出色，并且优化的最佳提示超过了人为设计的提示。

    

    优化是无处不在的。虽然基于导数的算法在各种问题上是强大的工具，但是没有梯度对许多实际应用提出了挑战。在这项工作中，我们提出了一种简单有效的方法，利用大型语言模型(LLMs)作为优化器，其中优化任务以自然语言形式描述。在每一次优化步骤中，LLM从包含先前生成的解与其值的提示中生成新的解，然后对新的解进行评估并添加到提示中，用于下一次优化步骤。我们首先展示了OPRO在线性回归和旅行推销员问题上的应用，然后转向提示优化，目标是找到能最大化任务准确性的指令。通过使用各种LLM，我们证明了OPRO优化的最佳提示在GSM8K上击败了人为设计的提示高达8%，在Big-Bench Hard任务上击败了人为设计的提示高达50%。

    Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to prompt optimization where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks.
    
[^49]: 提高ChatGPT生成的假科学检测的方法：引入xFakeBibs监督学习网络算法

    Improving Detection of ChatGPT-Generated Fake Science Using Real Publication Text: Introducing xFakeBibs a Supervised-Learning Network Algorithm. (arXiv:2308.11767v1 [cs.CL])

    [http://arxiv.org/abs/2308.11767](http://arxiv.org/abs/2308.11767)

    本文介绍了一种能够提高对ChatGPT生成的假科学进行检测的算法。通过使用一种新设计的监督机器学习算法，该算法能够准确地将机器生成的出版物与科学家生成的出版物区分开来。结果表明，ChatGPT在技术术语方面与真实科学存在显著差异。算法在分类过程中取得了较高的准确率。

    

    ChatGPT正在成为现实。本文展示了如何区分ChatGPT生成的出版物与科学家生成的出版物。通过使用一种新设计的监督机器学习算法，我们演示了如何检测机器生成的出版物和科学家生成的出版物。该算法使用100个真实出版物摘要进行训练，然后采用10倍交叉验证方法建立了一个接受范围的下限和上限。与ChatGPT内容进行比较，明显可见ChatGPT仅贡献了23\%的二元组内容，这比其他10个交叉验证中的任何一个都少50\%。这个分析凸显了ChatGPT在技术术语上与真实科学的明显差异。在对每篇文章进行分类时，xFakeBibs算法准确地将98篇出版物识别为假的，有2篇文献错误地分类为真实出版物。尽管这项工作引入了一种算法应用

    ChatGPT is becoming a new reality. In this paper, we show how to distinguish ChatGPT-generated publications from counterparts produced by scientists. Using a newly designed supervised Machine Learning algorithm, we demonstrate how to detect machine-generated publications from those produced by scientists. The algorithm was trained using 100 real publication abstracts, followed by a 10-fold calibration approach to establish a lower-upper bound range of acceptance. In the comparison with ChatGPT content, it was evident that ChatGPT contributed merely 23\% of the bigram content, which is less than 50\% of any of the other 10 calibrating folds. This analysis highlights a significant disparity in technical terms where ChatGPT fell short of matching real science. When categorizing the individual articles, the xFakeBibs algorithm accurately identified 98 out of 100 publications as fake, with 2 articles incorrectly classified as real publications. Though this work introduced an algorithmic app
    
[^50]: DiagGPT:一种基于LLM的任务导向对话的聊天机器人

    DiagGPT: An LLM-based Chatbot with Automatic Topic Management for Task-Oriented Dialogue. (arXiv:2308.08043v1 [cs.CL])

    [http://arxiv.org/abs/2308.08043](http://arxiv.org/abs/2308.08043)

    DiagGPT将大型语言模型(LLMs)扩展到任务导向的对话场景，提供了在复杂诊断场景中主动提问和引导用户完成任务的能力。

    

    大型语言模型(LLMs)如ChatGPT正变得越来越复杂，展示出与人类相似的能力。这些AI模型在日常生活中辅助人类完成各种任务方面发挥着重要作用。AI作为聊天代理人的重要应用是回答人类在各个领域的问题。目前的LLMs在回答一般问题方面已经显示出熟练的能力。然而，在复杂的诊断场景(如法律或医疗咨询)中，基本的问答对话往往表现不佳。这些场景通常需要任务导向对话(TOD)，其中AI聊天代理需要主动提问并引导用户完成特定任务。以前的微调模型在TOD方面表现不佳，而当前的LLMs并未固有这种能力。在本文中，我们介绍了一种名为DiagGPT (Diagnosis GPT)的创新方法，它将LLMs推广到TOD场景中。

    Large Language Models (LLMs), such as ChatGPT, are becoming increasingly sophisticated, demonstrating capabilities that closely resemble those of humans. These AI models are playing an essential role in assisting humans with a wide array of tasks in daily life. A significant application of AI is its use as a chat agent, responding to human inquiries across various domains. Current LLMs have shown proficiency in answering general questions. However, basic question-answering dialogue often falls short in complex diagnostic scenarios, such as legal or medical consultations. These scenarios typically necessitate Task-Oriented Dialogue (TOD), wherein an AI chat agent needs to proactively pose questions and guide users towards specific task completion. Previous fine-tuning models have underperformed in TOD, and current LLMs do not inherently possess this capability. In this paper, we introduce DiagGPT (Dialogue in Diagnosis GPT), an innovative method that extends LLMs to TOD scenarios. Our e
    
[^51]: CFN-ESA：一种具有情绪转移感知的跨模态融合网络用于对话情绪识别

    CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition. (arXiv:2307.15432v1 [cs.CL])

    [http://arxiv.org/abs/2307.15432](http://arxiv.org/abs/2307.15432)

    本文提出了一种具有情绪转移感知的跨模态融合网络（CFN-ESA）用于对话情绪识别，通过将文本模态作为主要情感信息的来源，视觉和声学模态作为次要信息的来源，并引入情绪转移模块来解决情绪转移场景下情感识别的问题。

    

    在对话情绪识别方面，多模态情感识别受到了各领域研究界的越来越多的关注。本文提出了一种具有情绪转移感知的跨模态融合网络（CFN-ESA）用于对话情绪识别。现有方法均平等地使用每个模态而无法区分情感信息的多少，从而难以充分提取多模态数据中的互补和关联信息。为了解决这个问题，在CFN-ESA中，文本模态被视为情感信息的主要来源，而视觉和声学模态则被视为次要来源。此外，大多数多模态情感识别模型忽视了情绪转移信息，过度关注上下文信息，导致在情绪转移场景下情感识别失败。我们设计了一个情绪转移模块来应对这一挑战。CFN-ESA主要包括单模态编码器（RUME）、跨模态编码器（ACME）和情绪转移模块（LESM）。

    Multimodal Emotion Recognition in Conversation (ERC) has garnered growing attention from research communities in various fields. In this paper, we propose a cross-modal fusion network with emotion-shift awareness (CFN-ESA) for ERC. Extant approaches employ each modality equally without distinguishing the amount of emotional information, rendering it hard to adequately extract complementary and associative information from multimodal data. To cope with this problem, in CFN-ESA, textual modalities are treated as the primary source of emotional information, while visual and acoustic modalities are taken as the secondary sources. Besides, most multimodal ERC models ignore emotion-shift information and overfocus on contextual information, leading to the failure of emotion recognition under emotion-shift scenario. We elaborate an emotion-shift module to address this challenge. CFN-ESA mainly consists of the unimodal encoder (RUME), cross-modal encoder (ACME), and emotion-shift module (LESM).
    
[^52]: FLASK: 基于对齐技能集的细粒度语言模型评估

    FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. (arXiv:2307.10928v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2307.10928](http://arxiv.org/abs/2307.10928)

    FLASK是一种基于对齐技能集的细粒度语言模型评估协议，通过将粗级评分分解为每个指令的技能集级评分，实现了对模型性能的全面视角和提高评估的可靠性。

    

    由于指令需要与人类的价值观进行对齐，并且所需的技能集根据指令而异，因此对大型语言模型（LLMs）进行评估具有挑战性。然而，先前的研究主要集中在粗粒度评估（即基于整体偏好的评估），这限制了可解释性，因为它未考虑需要实例级技能组合的用户指令的特性。在本文中，我们介绍了FLASK（基于对齐技能集的细粒度语言模型评估），这是一种细粒度评估协议，用于人类和模型的评估，它将粗级评分分解为每个指令的技能集级评分。通过实验证明，评估的细粒度对于获得对模型性能的全面视角和提高评估的可靠性至关重要。利用FLASK，我们比较了多个开源和专有LLMs，并观察到高度相关性。

    Evaluation of Large Language Models (LLMs) is challenging because instruction-following necessitates alignment with human values and the required set of skills varies depending on the instruction. However, previous studies have mainly focused on coarse-grained evaluation (i.e. overall preference-based evaluation), which limits interpretability since it does not consider the nature of user instructions that require instance-wise skill composition. In this paper, we introduce FLASK (Fine-grained Language Model Evaluation based on Alignment Skill Sets), a fine-grained evaluation protocol for both human-based and model-based evaluation which decomposes coarse-level scoring to a skill set-level scoring for each instruction. We experimentally observe that the fine-graininess of evaluation is crucial for attaining a holistic view of model performance and increasing the reliability of the evaluation. Using FLASK, we compare multiple open-source and proprietary LLMs and observe a high correlati
    
[^53]: 探索对比演示和显著性图在上下文学习中的作用

    Towards Understanding In-Context Learning with Contrastive Demonstrations and Saliency Maps. (arXiv:2307.05052v1 [cs.CL])

    [http://arxiv.org/abs/2307.05052](http://arxiv.org/abs/2307.05052)

    本研究探索了对比演示和显著性图在上下文学习中的作用，并发现改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。另外，补充解释在提高上下文学习方面是有效的。

    

    本文研究了在大型语言模型的上下文学习(ICL)性能中，各种演示组件的作用。具体而言，我们探讨了标签、输入分布和补充解释等因素的影响，特别是在这些因素被修改或扰动时的影响。我们基于之前的工作，这些工作对于这些元素如何影响ICL给出了不一致的结果。为了探究这些问题，我们采用了可解释的自然语言处理(XNLP)方法，并利用对比演示的显著性图进行定性和定量分析。我们的研究结果表明，改变标签对显著性有显著影响，尤其对于更大的语言模型更为明显。我们对输入分布进行了粒度级别的分析，发现在情感分析任务中，将表达情感的术语改为中性词并不像改变标签那样具有显著影响。最后，我们发现补充解释在提高ICL方面的效果是存在的。

    We investigate the role of various demonstration components in the in-context learning (ICL) performance of large language models (LLMs). Specifically, we explore the impacts of ground-truth labels, input distribution, and complementary explanations, particularly when these are altered or perturbed. We build on previous work, which offers mixed findings on how these elements influence ICL. To probe these questions, we employ explainable NLP (XNLP) methods and utilize saliency maps of contrastive demonstrations for both qualitative and quantitative analysis. Our findings reveal that flipping ground-truth labels significantly affects the saliency, though it's more noticeable in larger LLMs. Our analysis of the input distribution at a granular level reveals that changing sentiment-indicative terms in a sentiment analysis task to neutral ones does not have as substantial an impact as altering ground-truth labels. Finally, we find that the effectiveness of complementary explanations in boos
    
[^54]: 大语言模型时代的推荐系统 (LLMs)

    Recommender Systems in the Era of Large Language Models (LLMs). (arXiv:2307.02046v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.02046](http://arxiv.org/abs/2307.02046)

    大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。

    

    随着电子商务和网络应用的繁荣，推荐系统（RecSys）已经成为我们日常生活中重要的组成部分，为用户提供个性化建议以满足其喜好。尽管深度神经网络（DNN）通过模拟用户-物品交互和整合文本侧信息在提升推荐系统方面取得了重要进展，但是DNN方法仍然存在一些限制，例如理解用户兴趣、捕捉文本侧信息的困难，以及在不同推荐场景中泛化和推理能力的不足等。与此同时，大型语言模型（LLMs）的出现（例如ChatGPT和GPT4）在自然语言处理（NLP）和人工智能（AI）领域引起了革命，因为它们在语言理解和生成的基本职责上有着卓越的能力，同时具有令人印象深刻的泛化和推理能力。

    With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, 
    
[^55]: 基于贝叶斯原理的上下文学习

    In-Context Learning through the Bayesian Prism. (arXiv:2306.04891v1 [cs.LG])

    [http://arxiv.org/abs/2306.04891](http://arxiv.org/abs/2306.04891)

    这篇论文研究了大型语言模型中的上下文学习现象，并通过实验证据展示了Transformer模型在多种设置下表现出贝叶斯预测器的行为。作者还探讨了上下文学习与贝叶斯学习框架之间的联系，并提出了一个线性回归任务来验证这种联系。

    

    上下文学习是大型语言模型中令人惊讶且有用的特性之一。它的工作原理是一个活跃的研究领域。近期，人们设计了一些风格化的类元学习的设置，它们使用语言建模损失函数对来自函数类的输入输出对$(x, f(x))$ 进行训练，并观察模型对同一类中未见过的函数的泛化能力。这一研究线路中的一个主要发现是，对于诸如线性回归等几个问题，训练好的 Transformer 学习了上下文学习算法。然而，导致这种行为的归纳偏差并不清楚。拥有无限的训练数据和计算能力的模型是贝叶斯预测器：它学习了预训练分布。已经证明，高容量的 Transformer 模型在线性回归任务上模拟贝叶斯预测器的行为。在本文中，我们展示了Transformer在多种设置下表现出理想学习者的行为的经验证据，包括外推和求解微分方程。我们探讨了上下文学习和贝叶斯学习框架之间的联系，认为这些模型学习了合理函数的先验概率，而不仅仅是最小化语言建模损失。最后，我们提出了一个简单的线性回归任务来进一步探究这种联系，证明使用真实的贝叶斯先验进行训练的模型比使用固定先验或没有先验训练的模型表现更好。

    In-context learning is one of the surprising and useful features of large language models. How it works is an active area of research. Recently, stylized meta-learning-like setups have been devised that train these models on a sequence of input-output pairs $(x, f(x))$ from a function class using the language modeling loss and observe generalization to unseen functions from the same class. One of the main discoveries in this line of research has been that for several problems such as linear regression, trained transformers learn algorithms for learning functions in context. However, the inductive biases of these models resulting in this behavior are not clearly understood. A model with unlimited training data and compute is a Bayesian predictor: it learns the pretraining distribution. It has been shown that high-capacity transformers mimic the Bayesian predictor for linear regression. In this paper, we show empirical evidence of transformers exhibiting the behavior of this ideal learne
    
[^56]: 递归的诅咒：使用生成数据进行训练会让模型忘记

    The Curse of Recursion: Training on Generated Data Makes Models Forget. (arXiv:2305.17493v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.17493](http://arxiv.org/abs/2305.17493)

    使用生成数据进行训练会导致模型不可逆的缺陷并且使得原始内容分布的尾部消失，这种效应称为模型折叠。我们证明了这种现象在所有学习生成模型中都存在，必须认真对待。

    

    稳定扩散技术革命性地改变了从描述性文本中生成图像的方法。GPT-2、GPT-3(.5)和GPT-4在各种语言任务中表现惊人。ChatGPT将这些语言模型引入了大众视野。大语言模型(LLMs)已经不可避免并将彻底改变在线文本和图像的整个生态系统。本文考虑了未来可能发生的事情。当LLMs占据了在线语言的大部分时，GPT-{n}会发生什么？我们发现，在训练中使用模型生成的内容会导致所得模型中不可逆缺陷，原始内容分布的尾部消失。我们将这种效应称为模型折叠，并显示它可以发生在变分自编码器、高斯混合模型和LLMs中。我们建立了现象背后的理论直觉，并展示了这种现象在所有学习生成模型中的普遍性。我们证明，如果我们要在实践中使用生成数据进行训练，就必须认真对待这一问题。

    Stable Diffusion revolutionised image creation from descriptive text. GPT-2, GPT-3(.5) and GPT-4 demonstrated astonishing performance across a variety of language tasks. ChatGPT introduced such language models to the general public. It is now clear that large language models (LLMs) are here to stay, and will bring about drastic change in the whole ecosystem of online text and images. In this paper we consider what the future might hold. What will happen to GPT-{n} once LLMs contribute much of the language found online? We find that use of model-generated content in training causes irreversible defects in the resulting models, where tails of the original content distribution disappear. We refer to this effect as Model Collapse and show that it can occur in Variational Autoencoders, Gaussian Mixture Models and LLMs. We build theoretical intuition behind the phenomenon and portray its ubiquity amongst all learned generative models. We demonstrate that it has to be taken seriously if we ar
    
[^57]: 通过多语言微调和翻译指令诱发大规模语言模型的翻译能力

    Eliciting the Translation Ability of Large Language Models via Multilingual Finetuning with Translation Instructions. (arXiv:2305.15083v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15083](http://arxiv.org/abs/2305.15083)

    本文通过对多语言预训练语言模型进行微调，研究了它们如何通过翻译指令执行多语言翻译任务。研究发现多语言LLMs具有较强的翻译能力，这取决于语言与英语的相似性和预训练阶段使用的数据量。此外，执行翻译指令的能力依赖于对指令的理解和不同语言之间的对齐。

    

    大规模预训练语言模型（LLMs），例如ChatGPT和GPT4，展现出在多语言翻译方面的强大能力，而无需明确训练并行语料库。本文研究了LLMs如何获得其对不同语言进行翻译指令的能力。我们通过对多语言预训练语言模型XGLM-7B进行微调来执行多语言翻译任务，并进行了详细分析。首先，我们展示了多语言LLMs具有比先前展示的更强的翻译能力。对于某种语言，其表现取决于其与英语的相似性和预训练阶段使用的数据量。其次，我们发现LLMs执行翻译指令的能力依赖于对翻译指令的理解以及不同语言之间的对齐。通过多语言微调，LLMs能够学习并在翻译任务中表现出良好的能力，即使对于那些语言间平行语料较少的情况。

    Large-scale Pretrained Language Models (LLMs), such as ChatGPT and GPT4, have shown strong abilities in multilingual translations, without being explicitly trained on parallel corpora. It is interesting how the LLMs obtain their ability to carry out translation instructions for different languages. In this paper, we present a detailed analysis by finetuning a multilingual pretrained language model, XGLM-7B, to perform multilingual translation following given instructions. Firstly, we show that multilingual LLMs have stronger translation abilities than previously demonstrated. For a certain language, the performance depends on its similarity to English and the amount of data used in the pretraining phase. Secondly, we find that LLMs' ability to carry out translation instructions relies on the understanding of translation instructions and the alignment among different languages. With multilingual finetuning, LLMs could learn to perform the translation task well even for those language pa
    
[^58]: 用语言信息实现强烈感情表现和语调的TTS系统

    EE-TTS: Emphatic Expressive TTS with Linguistic Information. (arXiv:2305.12107v1 [cs.SD])

    [http://arxiv.org/abs/2305.12107](http://arxiv.org/abs/2305.12107)

    该论文提出了一种利用语言信息和强调预测器实现表现力强且自然的TTS系统，并在实验中表现出良好的表现和泛化能力。

    

    目前的TTS系统能够合成高质量的语音，但要产生高度表现力的语音仍然是一个挑战。强调是决定语音表现力的关键因素，近年来受到了更多的关注。以前的研究通常通过添加中间特征来增强强调，但不能保证语音的整体表现力。为了解决这个问题，我们提出了一种名为EE-TTS的新方法，它利用句法和语义上的多层语言信息。EE-TTS包含一个强调位置预测器，可以从文本中识别出适当的强调位置，并具有一个条件声学模型，可以合成带强调和语言信息的表现力语音。实验结果表明，EE-TTS在表现力和自然度方面比基线有0.49和0.67的MOS提升。EE-TTS还表现出强大的泛化能力，根据AB测试结果在不同数据集上都表现良好。

    While Current TTS systems perform well in synthesizing high-quality speech, producing highly expressive speech remains a challenge. Emphasis, as a critical factor in determining the expressiveness of speech, has attracted more attention nowadays. Previous works usually enhance the emphasis by adding intermediate features, but they can not guarantee the overall expressiveness of the speech. To resolve this matter, we propose Emphatic Expressive TTS (EE-TTS), which leverages multi-level linguistic information from syntax and semantics. EE-TTS contains an emphasis predictor that can identify appropriate emphasis positions from text and a conditioned acoustic model to synthesize expressive speech with emphasis and linguistic information. Experimental results indicate that EE-TTS outperforms baseline with MOS improvements of 0.49 and 0.67 in expressiveness and naturalness. EE-TTS also shows strong generalization across different datasets according to AB test results.
    
[^59]: 通过双重对抗去偏置实现的越界证据感知假新闻检测

    Out-of-distribution Evidence-aware Fake News Detection via Dual Adversarial Debiasing. (arXiv:2304.12888v1 [cs.CL])

    [http://arxiv.org/abs/2304.12888](http://arxiv.org/abs/2304.12888)

    该论文提出了一种新颖的双重对抗学习方法，通过在模型中加入去偏置鉴别器，旨在训练模型更好地进行越界证据感知假新闻检测，有效减轻新闻和证据内容偏差的影响。

    

    越界证据感知假新闻检测旨在对新闻和基于新闻内容检索的证据进行推理，以查找统一性或不一致性。然而，我们发现，证据感知检测模型会受到偏差的影响，即新闻/证据内容和真实/假新闻标签之间的虚假相关性，并且很难推广到越界情况。为了应对这个问题，我们提出了一种新颖的双重对抗学习方法。我们在DAL中加入了新闻方面和证据方面去偏置鉴别器，它们的目标都是真假新闻标签。然后，DAL会逆向优化新闻方面和证据方面去偏置鉴别器，以减轻新闻和证据内容偏差的影响。同时，DAL还优化主要的假新闻预测器，让新闻-证据交互模块能够被学习。这个过程能够帮助我们教新闻检测模型更好地进行新闻证据推理，并将检测假新闻的负面影响降至最低。

    Evidence-aware fake news detection aims to conduct reasoning between news and evidence, which is retrieved based on news content, to find uniformity or inconsistency. However, we find evidence-aware detection models suffer from biases, i.e., spurious correlations between news/evidence contents and true/fake news labels, and are hard to be generalized to Out-Of-Distribution (OOD) situations. To deal with this, we propose a novel Dual Adversarial Learning (DAL) approach. We incorporate news-aspect and evidence-aspect debiasing discriminators, whose targets are both true/fake news labels, in DAL. Then, DAL reversely optimizes news-aspect and evidence-aspect debiasing discriminators to mitigate the impact of news and evidence content biases. At the same time, DAL also optimizes the main fake news predictor, so that the news-evidence interaction module can be learned. This process allows us to teach evidence-aware fake news detection models to better conduct news-evidence reasoning, and min
    
[^60]: 通过关注图解析Transformer中的前馈模块

    Analyzing Feed-Forward Blocks in Transformers through the Lens of Attention Map. (arXiv:2302.00456v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.00456](http://arxiv.org/abs/2302.00456)

    通过关注图解析Transformer中的前馈模块，揭示了其修改输入语境化以强调特定类型语言组合的作用，并暗示了Transformer层处理中的潜在冗余。

    

    鉴于Transformer在广泛的任务中无处不在，解释它们的内部机制是一个关键问题。然而，它们的特定组件，前馈(FF)模块，尽管它们有大量的参数，但通常被分析得较少。我们通过将FF模块在关注图中渲染出来作为一种易于理解的可视化方案，来分析FF模块的输入语境效果。我们对有屏蔽和因果语言模型进行的实验表明，FF网络修改了输入的语境化以强调特定类型的语言组合。此外，FF模块及其周围的组件往往会互相抵消效果，表明Transformer层的处理中可能存在潜在的冗余。

    Given that Transformers are ubiquitous in wide tasks, interpreting their internals is a pivotal issue. Still, their particular components, feed-forward (FF) blocks, have typically been less analyzed despite their substantial parameter amounts. We analyze the input contextualization effects of FF blocks by rendering them in the attention maps as a human-friendly visualization scheme. Our experiments with both masked- and causal-language models reveal that FF networks modify the input contextualization to emphasize specific types of linguistic compositions. In addition, FF and its surrounding components tend to cancel out each other's effects, suggesting potential redundancy in the processing of the Transformer layer.
    
[^61]: ComCLIP: 无需训练的组合图像与文本匹配

    ComCLIP: Training-Free Compositional Image and Text Matching. (arXiv:2211.13854v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.13854](http://arxiv.org/abs/2211.13854)

    本文提出了一个无需训练的组合图像与文本匹配模型 ComCLIP，通过将输入图像分解为主体、对象和动作子图像，并结合视觉编码器和文本编码器进行逐步匹配，以解决组合图像与文本匹配中的伪匹配问题。

    

    对比语言-图像预训练（CLIP）已经展示了在图像与文本匹配方面的很好的零样本性能。然而，将 CLIP 这样的视觉-语言预训练模型适应于更具挑战性的组合图像与文本匹配仍然具有挑战性，这需要模型理解组合词概念和视觉组件。为了实现更好的零样本图像与文本匹配中的组合泛化能力，本文从因果关系的角度研究了该问题：单个实体的错误语义本质上是导致匹配失败的混淆因素。因此，我们提出了一种新颖的“无需训练”的组合 CLIP 模型（ComCLIP）。ComCLIP将输入图像分解为主体、对象和动作子图像，并组合 CLIP 的视觉编码器和文本编码器，以在组合文本嵌入和子图像嵌入之上进行逐步匹配。通过这种方式，ComCLIP 可以减轻伪匹配问题。

    Contrastive Language-Image Pretraining (CLIP) has demonstrated great zero-shot performance for matching images and text. However, it is still challenging to adapt vision-lanaguage pretrained models like CLIP to compositional image and text matching -- a more challenging image and text matching task requiring the model understanding of compositional word concepts and visual components. Towards better compositional generalization in zero-shot image and text matching, in this paper, we study the problem from a causal perspective: the erroneous semantics of individual entities are essentially confounders that cause the matching failure. Therefore, we propose a novel \textbf{\textit{training-free}} compositional CLIP model (ComCLIP). ComCLIP disentangles input images into subjects, objects, and action sub-images and composes CLIP's vision encoder and text encoder to perform evolving matching over compositional text embedding and sub-image embeddings. In this way, ComCLIP can mitigate spurio
    

