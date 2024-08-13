# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution](https://rss.arxiv.org/abs/2402.01586) | 本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。 |
| [^2] | [Decoding Speculative Decoding](https://rss.arxiv.org/abs/2402.01528) | 推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。 |
| [^3] | [Prompt-prompted Mixture of Experts for Efficient LLM Generation](https://arxiv.org/abs/2404.01365) | 提出了一种名为GRIFFIN的训练-free MoE，能够在各种LLM模型中选择唯一的FF专家以实现高效生成。 |
| [^4] | [LITE: Modeling Environmental Ecosystems with Multimodal Large Language Models](https://arxiv.org/abs/2404.01165) | 使用LITE模型，可以将不同的环境变量转换成自然语言描述和折线图像，并利用统一编码器来捕捉空间-时间关系，从而更好地预测环境变量。 |
| [^5] | [Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs](https://arxiv.org/abs/2403.19827) | 语言模型通过从相关结构（例如“a few days”）进行泛化学习，能够更好地学习AANN结构。 |
| [^6] | [Moderating Illicit Online Image Promotion for Unsafe User-Generated Content Games Using Large Vision-Language Models](https://arxiv.org/abs/2403.18957) | 该研究旨在调查不安全用户生成内容游戏中的违法推广威胁，收集了一组包含性暴力和暴力内容的真实图像数据集。 |
| [^7] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^8] | [Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models](https://arxiv.org/abs/2403.15740) | 通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。 |
| [^9] | [XLAVS-R: Cross-Lingual Audio-Visual Speech Representation Learning for Noise-Robust Speech Perception](https://arxiv.org/abs/2403.14402) | XLAVS-R是一个跨语言视听语音表示学习模型，通过利用有限的多语言AV预训练数据，简化预训练方案，以提高对噪声的鲁棒性，在下游音频-视觉语音识别和翻译任务中比先前最先进技术提升了高达18.5% WER和4.7 BLEU。 |
| [^10] | [LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968) | 该论文提出了一种数据精炼的方法，通过从LLM中提取知识来实现Prompt的压缩，确保压缩后的提示保持对原始提示的忠实性。 |
| [^11] | [RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems](https://arxiv.org/abs/2403.09040) | RAGGED框架分析和优化了检索增强生成系统，揭示了不同模型适合不同RAG设置的事实，编码器-解码器模型随文档数量增加而改善，而仅解码器模型只能有效利用少量文档。 |
| [^12] | [ConspEmoLLM: Conspiracy Theory Detection Using an Emotion-Based Large Language Model](https://arxiv.org/abs/2403.06765) | 本研究提出了ConspEmoLLM，这是第一个集成情感信息的大型语言模型，通过对阴谋理论文本的情感特征进行综合分析，能够执行多项任务，包括阴谋理论检测、理论类型分类和相关文本检测。 |
| [^13] | [LIEDER: Linguistically-Informed Evaluation for Discourse Entity Recognition](https://arxiv.org/abs/2403.06301) | 大型语言模型在语篇实体识别上具有基本的能力，但在新颖性方面仍未达到人类水平 |
| [^14] | [Learning or Self-aligning? Rethinking Instruction Fine-tuning](https://arxiv.org/abs/2402.18243) | 本研究揭示了指导微调的潜在机制，发现尝试通过指导微调学习额外世界知识往往难以产生积极影响，重点在于保持内部知识一致性。 |
| [^15] | [Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities](https://arxiv.org/abs/2402.10835) | 本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。 |
| [^16] | [LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset](https://arxiv.org/abs/2402.09391) | 本文介绍了LlaSMol，它是一种推进化学领域大规模语言模型的方法。通过使用一个大规模、全面、高质量的指令调优数据集来训练模型，LlaSMol在化学任务中表现出强大的性能，超过了GPT-4并接近于任务特定模型。 |
| [^17] | [Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents](https://arxiv.org/abs/2402.00798) | 本文提出了一种将自然语言和形式语言整合的“正式-LLM”框架，用于解决现有LLM智能体无法控制的计划生成问题。实验证明，该框架在提高生成计划性能和确保可控性方面取得了显著改进。 |
| [^18] | [LLsM: Generative Linguistic Steganography with Large Language Model](https://arxiv.org/abs/2401.15656) | 本研究提出了LLsM，一种基于大型语言模型的生成式语言隐写术。通过对大规模数据集进行微调，LLM能够以可控的方式生成具有特定话语特征的隐写文本，提高了隐蔽通信的效果。 |
| [^19] | [Digital Socrates: Evaluating LLMs through Explanation Critiques](https://arxiv.org/abs/2311.09613) | 通过定义新的解释批评任务、创建人工验证过的数据集并训练开源自动批评模型，数字苏格拉底有助于揭示学生模型的见解。 |
| [^20] | [MambaByte: Token-free Selective State Space Model.](http://arxiv.org/abs/2401.13660) | MambaByte是一种无标记的选择性状态空间模型，通过在字节级别上进行自回归训练，解决了标准自回归Transformer在处理长序列时的性能问题，并展现了与最先进的子词Transformer相媲美甚至更优的性能，从而证明了MambaByte在无标记语言建模方面的有效性。 |
| [^21] | [Private Fine-tuning of Large Language Models with Zeroth-order Optimization.](http://arxiv.org/abs/2401.04343) | 引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。 |
| [^22] | [Benchmarking Cognitive Biases in Large Language Models as Evaluators.](http://arxiv.org/abs/2309.17012) | 本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。 |
| [^23] | [BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks.](http://arxiv.org/abs/2305.17100) | BiomedGPT是一种面向视觉、语言和多模态任务的通用生物医学生成预训练Transformer，在多个临床任务中取得了16个最新的最优结果，包括超过了OpenAI的GPT-4V和Google的Med-PaLM M（12B）。同时，BiomedGPT还支持零-shot迁移学习。 |
| [^24] | [SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations.](http://arxiv.org/abs/2305.13235) | 这篇论文介绍了SparseFit，一种少样本刺激的稀疏微调策略，用于联合生成预测和自然语言解释。该方法可以在只有少量自然语言解释可用时生成高质量的自然语言解释。 |
| [^25] | [Bot or Human? Detecting ChatGPT Imposters with A Single Question.](http://arxiv.org/abs/2305.06424) | 本文提出了一个名为FLAIR的框架，通过一个问题和回答来检测ChatGPT中的聊天机器人真实性，可以分类人和机器人。单问题分为对于人类而言容易但对于机器人很难和对于机器人而言容易但对于人类很难两个类别，分别进行检测。 在多个数据集上实现了最先进的性能。 |
| [^26] | [Inspecting and Editing Knowledge Representations in Language Models.](http://arxiv.org/abs/2304.00740) | REMEDI是一种将自然语言语句映射到LM内部表示系统中的事实编码的学习方法。 REMEDI编码可用作知识编辑器，也可以用作探针，揭示了LM已经将哪些属性归因于提到的实体，并可以预测LM会生成输出的情况。 |
| [^27] | [Large language models can rate news outlet credibility.](http://arxiv.org/abs/2304.00228) | 本文评估了 ChatGPT 是否能够评估新闻机构的可信度，结果表明 ChatGPT 可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明，并且这些评级与人类专家的评级相关。LLMs可以成为事实检查应用程序中可信度评级的经济参考。 |
| [^28] | [PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue.](http://arxiv.org/abs/2302.06674) | PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。 |
| [^29] | [How would Stance Detection Techniques Evolve after the Launch of ChatGPT?.](http://arxiv.org/abs/2212.14548) | ChatGPT是一种新的预训练语言模型，可以用于解决立场检测问题，并提供了其预测的解释能力。 |

# 详细

[^1]: TrustAgent: 通过代理构成实现安全可信赖的LLM代理

    TrustAgent: Towards Safe and Trustworthy LLM-based Agents through Agent Constitution

    [https://rss.arxiv.org/abs/2402.01586](https://rss.arxiv.org/abs/2402.01586)

    本文介绍了一种基于代理构成的代理框架TrustAgent，该框架通过预先规划、规划过程中和计划后检查三种策略来提高LLM代理的安全性。实验结果表明，这些方法可以有效识别和预防潜在危险。此外，还研究了安全性与使用者满意度以及模型推理能力与效率之间的关系。

    

    近年来，基于LLM的代理引起了广泛关注，但其可信度仍未得到深入探索。由于代理可以直接与物理环境交互，其可靠性和安全性至关重要。本文提出了一种基于代理构成的代理框架TrustAgent，对LLM代理的安全性维度进行了初步研究。该框架包括三种策略：预先规划策略，在生成计划之前向模型注入安全知识；规划过程中策略，在生成计划时增强安全性；计划后检查策略，通过计划后检查确保安全性。通过实验分析，我们展示了这些方法如何通过识别和预防潜在危险有效提高LLM代理的安全性。此外，我们还探讨了安全性与使用者满意度之间的复杂关系，以及模型的推理能力与其效率之间的关联。

    The emergence of LLM-based agents has garnered considerable attention, yet their trustworthiness remains an under-explored area. As agents can directly interact with the physical environment, their reliability and safety is critical. This paper presents an Agent-Constitution-based agent framework, TrustAgent, an initial investigation into improving the safety dimension of trustworthiness in LLM-based agents. This framework consists of threefold strategies: pre-planning strategy which injects safety knowledge to the model prior to plan generation, in-planning strategy which bolsters safety during plan generation, and post-planning strategy which ensures safety by post-planning inspection. Through experimental analysis, we demonstrate how these approaches can effectively elevate an LLM agent's safety by identifying and preventing potential dangers. Furthermore, we explore the intricate relationships between safety and helpfulness, and between the model's reasoning ability and its efficac
    
[^2]: 解码推测解码

    Decoding Speculative Decoding

    [https://rss.arxiv.org/abs/2402.01528](https://rss.arxiv.org/abs/2402.01528)

    推测解码是一种用于加速大型语言模型推断的技术，但我们的实验表明，选择的草稿模型生成的令牌被目标模型接受的概率越高，吞吐量越低。我们通过大量实验，分析了各种因素对推测解码效果的影响，并提出了一个分析模型来提高效率。

    

    推测解码是一种常用的技术，用于加速大型语言模型（LLM）的推断，而不修改其结果。在对LLM进行推断时，推测解码使用较小的草稿模型生成推测令牌，然后使用目标LLM验证这些草稿令牌。推测解码提供的加速取决于草稿模型的选择。普遍建议选择一个草稿模型，该模型生成的令牌被LLM接受的概率很高，以实现最高吞吐量。然而，我们的实验结果与之相反，随着生成的令牌被目标模型接受的概率增加，吞吐量减少。为了理解这一现象，我们进行了大量实验，对影响推测解码的不同因素进行了表征，并研究了这些因素如何相互作用和影响加速效果。基于我们的实验结果，我们描述了一个分析模型，可以使用该模型来进行决策，提高推测解码的效率。

    Speculative Decoding is a widely used technique to speed up inference for Large Language Models (LLMs) without modifying its outcome. When performing inference on an LLM, speculative decoding uses a smaller draft model which generates speculative tokens and then uses the target LLM to verify those draft tokens. The speedup provided by speculative decoding heavily depends on the choice of the draft model. It has been widely suggested to select a draft model that provides a high probability of the generated token being accepted by the LLM to achieve the highest throughput. However, our experiments indicate the contrary with throughput diminishing as the probability of generated tokens to be accepted by the target model increases. To understand this phenomenon, we perform extensive experiments to characterize the different factors that affect speculative decoding and how those factors interact and affect the speedups. Based on our experiments we describe an analytical model which can be u
    
[^3]: 基于提示的混合专家模型用于高效生成LLM

    Prompt-prompted Mixture of Experts for Efficient LLM Generation

    [https://arxiv.org/abs/2404.01365](https://arxiv.org/abs/2404.01365)

    提出了一种名为GRIFFIN的训练-free MoE，能够在各种LLM模型中选择唯一的FF专家以实现高效生成。

    

    随着基于transformer的大规模语言模型（LLMs）的发展，由于其出色的实用性，它们已被应用于许多领域，但在部署时存在相当大的计算成本。幸运的是，一些方法，如修剪或构建混合专家（MoE），旨在利用transformer前馈（FF）块中的稀疏性，以提高速度并降低内存需求。但是，这些技术在实践中可能非常昂贵和不灵活，因为它们通常需要训练或仅限于特定类型的架构。为了解决这个问题，我们引入了GRIFFIN，一种新颖的无需训练的MoE，它在序列级别为不同非ReLU激活函数的大量LLMs选择独特的FF专家以实现高效生成。这是可能的，因为我们关键观察到，许多经过训练的LLMs在序列中自然产生高度结构化的FF激活模式，这

    arXiv:2404.01365v1 Announce Type: cross  Abstract: With the development of transformer-based large language models (LLMs), they have been applied to many fields due to their remarkable utility, but this comes at a considerable computational cost at deployment. Fortunately, some methods such as pruning or constructing a mixture of experts (MoE) aim at exploiting sparsity in transformer feedforward (FF) blocks to gain boosts in speed and reduction in memory requirements. However, these techniques can be very costly and inflexible in practice, as they often require training or are restricted to specific types of architectures. To address this, we introduce GRIFFIN, a novel training-free MoE that selects unique FF experts at the sequence level for efficient generation across a plethora of LLMs with different non-ReLU activation functions. This is possible due to a critical observation that many trained LLMs naturally produce highly structured FF activation patterns within a sequence, which
    
[^4]: 用多模态大型语言模型对环境生态系统进行建模

    LITE: Modeling Environmental Ecosystems with Multimodal Large Language Models

    [https://arxiv.org/abs/2404.01165](https://arxiv.org/abs/2404.01165)

    使用LITE模型，可以将不同的环境变量转换成自然语言描述和折线图像，并利用统一编码器来捕捉空间-时间关系，从而更好地预测环境变量。

    

    对环境生态系统进行建模在可持续管理地球的过程中发挥关键作用。精确预测空间和时间上的关键环境变量可以帮助制定明智的政策和决策，从而改善人们的生活。最近，基于深度学习的方法在建模空间-时间关系以预测环境变量方面显示出潜力。然而，这些方法通常在处理不完整特征和分布变化方面表现不佳，环境数据中常见这些问题是由于数据收集成本高昂和测量仪器失灵造成的。为了解决这些问题，我们提出了LITE——用于环境生态系统建模的多模态大型语言模型。具体来说，LITE通过将不同环境变量转换成自然语言描述和折线图像来统一它们。然后，LITE利用统一编码器来捕捉

    arXiv:2404.01165v1 Announce Type: new  Abstract: The modeling of environmental ecosystems plays a pivotal role in the sustainable management of our planet. Accurate prediction of key environmental variables over space and time can aid in informed policy and decision-making, thus improving people's livelihood. Recently, deep learning-based methods have shown promise in modeling the spatial-temporal relationships for predicting environmental variables. However, these approaches often fall short in handling incomplete features and distribution shifts, which are commonly observed in environmental data due to the substantial cost of data collection and malfunctions in measuring instruments. To address these issues, we propose LITE -- a multimodal large language model for environmental ecosystems modeling. Specifically, LITE unifies different environmental variables by transforming them into natural language descriptions and line graph images. Then, LITE utilizes unified encoders to capture 
    
[^5]: 语言模型从不常见的现象中学习：缺失AANN的情况

    Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs

    [https://arxiv.org/abs/2403.19827](https://arxiv.org/abs/2403.19827)

    语言模型通过从相关结构（例如“a few days”）进行泛化学习，能够更好地学习AANN结构。

    

    语言模型学习罕见的句法现象，但有人认为它们依赖于死记硬背，而不是语法概括。我们在规模为人类规模的语料库（1亿字）上进行训练，迭代训练变压器语言模型，然后评估它们对特定罕见语法现象的学习：英语的冠词+形容词+数字+名词（AANN）结构（“a beautiful five days”）。

    arXiv:2403.19827v1 Announce Type: new  Abstract: Language models learn rare syntactic phenomena, but it has been argued that they rely on rote memorization, as opposed to grammatical generalization. Training on a corpus of human-scale in size (100M words), we iteratively trained transformer language models on systematically manipulated corpora and then evaluated their learning of a particular rare grammatical phenomenon: the English Article+Adjective+Numeral+Noun (AANN) construction (``a beautiful five days''). We first compared how well this construction was learned on the default corpus relative to a counterfactual corpus in which the AANN sentences were removed. AANNs were still learned better than systematically perturbed variants of the construction. Using additional counterfactual corpora, we suggest that this learning occurs through generalization from related constructions (e.g., ``a few days''). An additional experiment showed that this learning is enhanced when there is more 
    
[^6]: 利用大规模视觉语言模型调节不安全用户生成内容游戏中的违法在线图片推广

    Moderating Illicit Online Image Promotion for Unsafe User-Generated Content Games Using Large Vision-Language Models

    [https://arxiv.org/abs/2403.18957](https://arxiv.org/abs/2403.18957)

    该研究旨在调查不安全用户生成内容游戏中的违法推广威胁，收集了一组包含性暴力和暴力内容的真实图像数据集。

    

    在线用户生成内容游戏（UGCGs）在儿童和青少年中越来越受欢迎，用于社交互动和更有创意的在线娱乐。然而，它们存在着更高的暴露不良内容的风险，引发了人们对儿童和青少年在线安全的日益关注。我们采取了第一步研究对不安全UGCGs的违法推广进行威胁性分析。我们收集了一组现实世界数据集，包括2,924张展示不同性暴力和暴力内容的图像，这些内容被游戏创建者用于推广UGCGs。

    arXiv:2403.18957v1 Announce Type: cross  Abstract: Online user-generated content games (UGCGs) are increasingly popular among children and adolescents for social interaction and more creative online entertainment. However, they pose a heightened risk of exposure to explicit content, raising growing concerns for the online safety of children and adolescents. Despite these concerns, few studies have addressed the issue of illicit image-based promotions of unsafe UGCGs on social media, which can inadvertently attract young users. This challenge arises from the difficulty of obtaining comprehensive training data for UGCG images and the unique nature of these images, which differ from traditional unsafe content. In this work, we take the first step towards studying the threat of illicit promotions of unsafe UGCGs. We collect a real-world dataset comprising 2,924 images that display diverse sexually explicit and violent content used to promote UGCGs by their game creators. Our in-depth studi
    
[^7]: 与人类判断相一致：大型语言模型评估中成对偏好的作用

    Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

    [https://arxiv.org/abs/2403.16950](https://arxiv.org/abs/2403.16950)

    在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。

    

    大型语言模型（LLMs）作为自动评估器在评估生成的自然语言质量方面表现出有希望的能力。然而，LLMs在评估中仍存在偏见，常常难以生成与人类评估一致的连贯评估。在这项工作中，我们首先对LLM评估器与人类判断之间的不一致进行系统研究，揭示现有旨在减轻偏见的校准方法不足以有效将LLM评估器对齐。受到RLHF中对偏好数据的使用的启发，我们将评估形式化为一个排序问题，并引入Pairwise-preference Search（PAIRS），这是一种以LLMs进行成对比较并有效对候选文本进行排序的基于不确定性引导的搜索方法。PAIRS在代表性评估任务上实现了最先进的性能，并且显示出比直接打分有显著改进。

    arXiv:2403.16950v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated promising capabilities as automatic evaluators in assessing the quality of generated natural language. However, LLMs still exhibit biases in evaluation and often struggle to generate coherent evaluations that align with human assessments. In this work, we first conduct a systematic study of the misalignment between LLM evaluators and human judgement, revealing that existing calibration methods aimed at mitigating biases are insufficient for effectively aligning LLM evaluators. Inspired by the use of preference data in RLHF, we formulate the evaluation as a ranking problem and introduce Pairwise-preference Search (PAIRS), an uncertainty-guided search method that employs LLMs to conduct pairwise comparisons and efficiently ranks candidate texts. PAIRS achieves state-of-the-art performance on representative evaluation tasks and demonstrates significant improvements over direct scoring. Furthe
    
[^8]: Ghost Sentence：一种供普通用户使用的工具，用于对大型语言模型中的数据进行版权保护

    Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models

    [https://arxiv.org/abs/2403.15740](https://arxiv.org/abs/2403.15740)

    通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。

    

    Web用户数据在预训练大型语言模型（LLMs）及其微调变种的生态系统中起着核心作用。本文提出了一种方法，建议用户在其文档中反复插入个人密码，使LLMs能够记忆这些密码。这些用户文档中隐藏的密码，被称为“幽灵句子”，一旦它们出现在LLMs生成的内容中，用户就可以确信他们的数据被用于训练。为了探索这种版权工具的有效性和用法，我们利用幽灵句子定义了“用户训练数据识别”任务。我们创建了来自不同来源、不同规模的多个数据集，并使用不同规模的LLMs进行测试。为了评估，我们引入了一个最后$k$个单词验证的方式。

    arXiv:2403.15740v1 Announce Type: new  Abstract: Web user data plays a central role in the ecosystem of pre-trained large language models (LLMs) and their fine-tuned variants. Billions of data are crawled from the web and fed to LLMs. How can \textit{\textbf{everyday web users}} confirm if LLMs misuse their data without permission? In this work, we suggest that users repeatedly insert personal passphrases into their documents, enabling LLMs to memorize them. These concealed passphrases in user documents, referred to as \textit{ghost sentences}, once they are identified in the generated content of LLMs, users can be sure that their data is used for training. To explore the effectiveness and usage of this copyrighting tool, we define the \textit{user training data identification} task with ghost sentences. Multiple datasets from various sources at different scales are created and tested with LLMs of different sizes. For evaluation, we introduce a last $k$ words verification manner along 
    
[^9]: XLAVS-R: 跨语言视听语音表示学习用于噪声鲁棒语音知觉

    XLAVS-R: Cross-Lingual Audio-Visual Speech Representation Learning for Noise-Robust Speech Perception

    [https://arxiv.org/abs/2403.14402](https://arxiv.org/abs/2403.14402)

    XLAVS-R是一个跨语言视听语音表示学习模型，通过利用有限的多语言AV预训练数据，简化预训练方案，以提高对噪声的鲁棒性，在下游音频-视觉语音识别和翻译任务中比先前最先进技术提升了高达18.5% WER和4.7 BLEU。

    

    语音识别和翻译系统对嘈杂的输入表现不佳，在现实环境中经常出现。通过视觉信号增强这些系统有潜力提高对噪声的鲁棒性。然而，视听（AV）数据仅有限可用，并且比仅有音频资源的语言更少。为填补这一空白，我们提出XLAVS-R，一个跨语言视听语音表示模型，用于超过100种语言的噪声鲁棒语音识别和翻译。它旨在最大程度利用有限的多语言AV预训练数据的益处，通过在音频-仅多语言预训练的基础上构建，并简化现有的预训练方案。在MuAViC基准评估上对XLAVS-R进行了广泛评估，显示了其在下游音频-视觉语音识别和翻译任务上的优势，在给出嘈杂的AV输入时，其优于先前最先进技术最高达到18.5% WER和4.7 BLEU。

    arXiv:2403.14402v1 Announce Type: cross  Abstract: Speech recognition and translation systems perform poorly on noisy inputs, which are frequent in realistic environments. Augmenting these systems with visual signals has the potential to improve robustness to noise. However, audio-visual (AV) data is only available in limited amounts and for fewer languages than audio-only resources. To address this gap, we present XLAVS-R, a cross-lingual audio-visual speech representation model for noise-robust speech recognition and translation in over 100 languages. It is designed to maximize the benefits of limited multilingual AV pre-training data, by building on top of audio-only multilingual pre-training and simplifying existing pre-training schemes. Extensive evaluation on the MuAViC benchmark shows the strength of XLAVS-R on downstream audio-visual speech recognition and translation tasks, where it outperforms the previous state of the art by up to 18.5% WER and 4.7 BLEU given noisy AV inputs
    
[^10]: LLMLingua-2: 高效且忠实的无任务Prompt压缩的数据精炼

    LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression

    [https://arxiv.org/abs/2403.12968](https://arxiv.org/abs/2403.12968)

    该论文提出了一种数据精炼的方法，通过从LLM中提取知识来实现Prompt的压缩，确保压缩后的提示保持对原始提示的忠实性。

    

    这篇论文关注于无任务的Prompt压缩，以提高泛化能力和效率。考虑到自然语言中的冗余性，现有方法通过根据从因果语言模型（如LLaMa-7B）获得的信息熵来删除token或词汇单位来压缩prompt。挑战在于信息熵可能是一个次优的压缩度量：(i)它仅利用单向上下文，可能无法捕获所有用于prompt压缩的关键信息；(ii)它与prompt压缩目标不一致。为了解决这些问题，我们提出了一种数据精炼过程，从LLM中获得知识以压缩prompt而不丢失关键信息，并同时引入了一个抽取式文本压缩数据集。我们将prompt压缩格式化为一个token分类问题，以确保压缩后的prompt与原始prompt的一致性。

    arXiv:2403.12968v1 Announce Type: new  Abstract: This paper focuses on task-agnostic prompt compression for better generalizability and efficiency. Considering the redundancy in natural language, existing approaches compress prompts by removing tokens or lexical units according to their information entropy obtained from a causal language model such as LLaMa-7B. The challenge is that information entropy may be a suboptimal compression metric: (i) it only leverages unidirectional context and may fail to capture all essential information needed for prompt compression; (ii) it is not aligned with the prompt compression objective.   To address these issues, we propose a data distillation procedure to derive knowledge from an LLM to compress prompts without losing crucial information, and meantime, introduce an extractive text compression dataset. We formulate prompt compression as a token classification problem to guarantee the faithfulness of the compressed prompt to the original one, and 
    
[^11]: RAGGED:朝着基于检索增强生成系统的知情设计

    RAGGED: Towards Informed Design of Retrieval Augmented Generation Systems

    [https://arxiv.org/abs/2403.09040](https://arxiv.org/abs/2403.09040)

    RAGGED框架分析和优化了检索增强生成系统，揭示了不同模型适合不同RAG设置的事实，编码器-解码器模型随文档数量增加而改善，而仅解码器模型只能有效利用少量文档。

    

    arXiv:2403.09040v1 声明类型: 新的 摘要: 检索增强生成（RAG）通过为文档型问答等任务提供附加上下文，极大地提升了语言模型（LMs）的性能。尽管具有潜力，但RAG的效力高度依赖于其配置，从而引发一个问题：什么是最佳RAG配置？为了回答这个问题，我们引入了RAGGED框架来分析和优化RAG系统。在一组代表性的文档型问答任务上，我们研究了两种经典的稀疏和密集检索器，以及四种在编码器-解码器和仅解码器结构中表现优异的LMs。通过RAGGED，我们发现不同模型适合完全不同的RAG设置。虽然编码器-解码器模型随着更多文档的增加而单调提升，但我们发现仅解码器模型只能有效地使用<5个文档，尽管通常具有更长的上下文窗口。RAGGED进一步揭示了LMs的上下文利用习惯，我们发现编码器-解码器模型...

    arXiv:2403.09040v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) greatly benefits language models (LMs) by providing additional context for tasks such as document-based question answering (DBQA). Despite its potential, the power of RAG is highly dependent on its configuration, raising the question: What is the optimal RAG configuration? To answer this, we introduce the RAGGED framework to analyze and optimize RAG systems. On a set of representative DBQA tasks, we study two classic sparse and dense retrievers, and four top-performing LMs in encoder-decoder and decoder-only architectures. Through RAGGED, we uncover that different models suit substantially varied RAG setups. While encoder-decoder models monotonically improve with more documents, we find decoder-only models can only effectively use < 5 documents, despite often having a longer context window. RAGGED offers further insights into LMs' context utilization habits, where we find that encoder-decoder models r
    
[^12]: 使用基于情感的大型语言模型检测阴谋理论

    ConspEmoLLM: Conspiracy Theory Detection Using an Emotion-Based Large Language Model

    [https://arxiv.org/abs/2403.06765](https://arxiv.org/abs/2403.06765)

    本研究提出了ConspEmoLLM，这是第一个集成情感信息的大型语言模型，通过对阴谋理论文本的情感特征进行综合分析，能够执行多项任务，包括阴谋理论检测、理论类型分类和相关文本检测。

    

    互联网给社会带来了好处和伤害。后者的一个主要例子是误导信息，包括充斥网络的阴谋理论。 自然语言处理的最新进展，特别是大型语言模型（LLMs）的出现，已经提高了准确检测误导信息的前景。然而，大多数基于LLM的阴谋理论检测方法仅专注于二元分类，并未考虑误导信息与情感特征（即情感和情绪）之间的重要关系。通过对揭示其独特情感特征的阴谋文本的全面分析，我们提出了ConspEmoLLM，这是第一个集成情感信息且能够执行涉及阴谋理论的多样任务的开源LLM。 这些任务不仅包括阴谋理论检测，还包括理论类型分类和相关文本检测。

    arXiv:2403.06765v1 Announce Type: new  Abstract: The internet has brought both benefits and harms to society. A prime example of the latter is misinformation, including conspiracy theories, which flood the web. Recent advances in natural language processing, particularly the emergence of large language models (LLMs), have improved the prospects of accurate misinformation detection. However, most LLM-based approaches to conspiracy theory detection focus only on binary classification and fail to account for the important relationship between misinformation and affective features (i.e., sentiment and emotions). Driven by a comprehensive analysis of conspiracy text that reveals its distinctive affective features, we propose ConspEmoLLM, the first open-source LLM that integrates affective information and is able to perform diverse tasks relating to conspiracy theories. These tasks include not only conspiracy theory detection, but also classification of theory type and detection of related d
    
[^13]: LIEDER: 用于语篇实体识别的语言学评估

    LIEDER: Linguistically-Informed Evaluation for Discourse Entity Recognition

    [https://arxiv.org/abs/2403.06301](https://arxiv.org/abs/2403.06301)

    大型语言模型在语篇实体识别上具有基本的能力，但在新颖性方面仍未达到人类水平

    

    论文提出了一种称为 LIEDER 的数据集，允许详细检查语言模型对存在性、唯一性、复数性和新颖性等四个关键语义属性的认知水平。研究发现，最先进的大型语言模型对所有这些属性都表现出敏感性，除了新颖性，这表明它们尚未达到人类水平的语言理解能力。

    arXiv:2403.06301v1 Announce Type: new  Abstract: Discourse Entity (DE) recognition is the task of identifying novel and known entities introduced within a text. While previous work has found that large language models have basic, if imperfect, DE recognition abilities (Schuster and Linzen, 2022), it remains largely unassessed which of the fundamental semantic properties that govern the introduction and subsequent reference to DEs they have knowledge of. We propose the Linguistically-Informed Evaluation for Discourse Entity Recognition (LIEDER) dataset that allows for a detailed examination of language models' knowledge of four crucial semantic properties: existence, uniqueness, plurality, and novelty. We find evidence that state-of-the-art large language models exhibit sensitivity to all of these properties except novelty, which demonstrates that they have yet to reach human-level language understanding abilities.
    
[^14]: 学习还是自我调整？重新思考指导微调

    Learning or Self-aligning? Rethinking Instruction Fine-tuning

    [https://arxiv.org/abs/2402.18243](https://arxiv.org/abs/2402.18243)

    本研究揭示了指导微调的潜在机制，发现尝试通过指导微调学习额外世界知识往往难以产生积极影响，重点在于保持内部知识一致性。

    

    指导微调（IFT）是构建大型语言模型（LLM）中至关重要的阶段。先前的研究主要关注IFT在行为规范传递和额外世界知识学习中的作用。然而，对IFT潜在机制的理解仍然相当有限。本文设计了一个知识干预框架，以解耦IFT的潜在因素，从而实现对不同因素的个体分析。令人惊讶的是，我们的实验揭示，通过IFT试图学习额外的世界知识往往难以产生积极影响，甚至可能导致明显负面影响。此外，我们发现在IFT之前和之后保持内部知识一致性是实现成功IFT的关键因素。我们的研究结果揭示了IFT的潜在机制，并为最新和潜在未来的研究提供了有力支持。

    arXiv:2402.18243v1 Announce Type: new  Abstract: Instruction Fine-tuning~(IFT) is a critical phase in building large language models~(LLMs). Previous works mainly focus on the IFT's role in the transfer of behavioral norms and the learning of additional world knowledge. However, the understanding of the underlying mechanisms of IFT remains significantly limited. In this paper, we design a knowledge intervention framework to decouple the potential underlying factors of IFT, thereby enabling individual analysis of different factors. Surprisingly, our experiments reveal that attempting to learn additional world knowledge through IFT often struggles to yield positive impacts and can even lead to markedly negative effects. Further, we discover that maintaining internal knowledge consistency before and after IFT is a critical factor for achieving successful IFT. Our findings reveal the underlying mechanisms of IFT and provide robust support for some very recent and potential future works.
    
[^15]: LLMs下的时间序列预测：理解和增强模型能力

    Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities

    [https://arxiv.org/abs/2402.10835](https://arxiv.org/abs/2402.10835)

    本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。

    

    大语言模型(LLMs)近年来在许多领域得到迅速发展。作为一种经典的机器学习任务，时间序列预测最近从LLMs中获得了推动。然而，在这一领域，LLMs的偏好存在研究空白。通过将LLMs与传统模型进行比较，发现了LLMs在时间序列预测中的许多特性。例如，我们的研究表明，LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战。我们通过设计提示要求LLMs告知数据集的周期来解释我们的发现。此外，本文还研究了输入策略，发现融入外部知识和采用自然语言释义积极影响了LLMs在时间序列预测中的预测性能。总的来说，这项研究有助于洞察LLMs在时间序列预测中的优势和局限性。

    arXiv:2402.10835v1 Announce Type: new  Abstract: Large language models (LLMs) have been applied in many fields with rapid development in recent years. As a classic machine learning task, time series forecasting has recently received a boost from LLMs. However, there is a research gap in the LLMs' preferences in this field. In this paper, by comparing LLMs with traditional models, many properties of LLMs in time series prediction are found. For example, our study shows that LLMs excel in predicting time series with clear patterns and trends but face challenges with datasets lacking periodicity. We explain our findings through designing prompts to require LLMs to tell the period of the datasets. In addition, the input strategy is investigated, and it is found that incorporating external knowledge and adopting natural language paraphrases positively affects the predictive performance of LLMs for time series. Overall, this study contributes to insight into the advantages and limitations of
    
[^16]: LlaSMol:利用大规模、全面、高质量的指令调优数据集推进化学的大规模语言模型

    LlaSMol: Advancing Large Language Models for Chemistry with a Large-Scale, Comprehensive, High-Quality Instruction Tuning Dataset

    [https://arxiv.org/abs/2402.09391](https://arxiv.org/abs/2402.09391)

    本文介绍了LlaSMol，它是一种推进化学领域大规模语言模型的方法。通过使用一个大规模、全面、高质量的指令调优数据集来训练模型，LlaSMol在化学任务中表现出强大的性能，超过了GPT-4并接近于任务特定模型。

    

    化学在药物研发和材料科学等许多领域中起着至关重要的作用。尽管诸如GPT-4之类的大型语言模型（LLM）在自然语言处理任务上展现出了非凡的能力，但现有工作表明它们在化学任务上的性能令人失望。然而，在本文中，我们展示了我们开发的LLM在一系列化学任务上可以取得非常强大的结果，在所有任务上都显著优于最先进的GPT-4，并接近SoTA任务特定模型。我们取得成功的关键是一个名为SMolInstruct的大规模、全面、高质量的指令调优数据集。它包含了14个经过精心挑选的化学任务和超过三百万个高质量样本，为训练和评估化学LLM奠定了坚实基础。基于SMolInstruct，我们对一组开源LLM进行了微调，其中，我们发现Mistral ser是最佳性能的模型。

    arXiv:2402.09391v1 Announce Type: new Abstract: Chemistry plays a crucial role in many domains, such as drug discovery and material science. While large language models (LLMs) such as GPT-4 exhibit remarkable capabilities on natural language processing tasks, existing work shows their performance on chemistry tasks is discouragingly low. In this paper, however, we demonstrate that our developed LLMs can achieve very strong results on a comprehensive set of chemistry tasks, outperforming the most advanced GPT-4 across all the tasks by a substantial margin and approaching the SoTA task-specific models. The key to our success is a large-scale, comprehensive, high-quality dataset for instruction tuning named SMolInstruct. It contains 14 meticulously selected chemistry tasks and over three million high-quality samples, laying a solid foundation for training and evaluating LLMs for chemistry. Based on SMolInstruct, we fine-tune a set of open-source LLMs, among which, we find that Mistral ser
    
[^17]: 正式-LLM：将形式语言和自然语言集成于可控的LLM智能体中

    Formal-LLM: Integrating Formal Language and Natural Language for Controllable LLM-based Agents

    [https://arxiv.org/abs/2402.00798](https://arxiv.org/abs/2402.00798)

    本文提出了一种将自然语言和形式语言整合的“正式-LLM”框架，用于解决现有LLM智能体无法控制的计划生成问题。实验证明，该框架在提高生成计划性能和确保可控性方面取得了显著改进。

    

    最近，对于大型语言模型（LLMs）的进展使得人工智能智能体能够自动生成和执行解决复杂任务的多步计划。然而，由于LLM的内容生成过程几乎无法控制，当前的LLM智能体经常生成无效或不可执行的计划，这损害了生成计划的性能并破坏了用户对LLM智能体的信任。为应对这个问题，本文提出了一种新颖的“正式-LLM”框架，用于LLM智能体，通过将自然语言的表达力和形式语言的精确性进行整合。具体而言，该框架允许人类用户将他们对计划过程的要求或约束表达为自动机。然后，在自动机的监督下，使用基于堆栈的LLM计划生成过程来确保生成的计划满足约束条件，从而使计划过程可控。我们在基准任务和实际的真实任务上进行了实验，并且obtained significant improvements over existing LLM-based agents, demonstrating the effectiveness and controllability of the proposed Formal-LLM framework.

    Recent advancements on Large Language Models (LLMs) enable AI Agents to automatically generate and execute multi-step plans to solve complex tasks. However, since LLM's content generation process is hardly controllable, current LLM-based agents frequently generate invalid or non-executable plans, which jeopardizes the performance of the generated plans and corrupts users' trust in LLM-based agents. In response, this paper proposes a novel ``Formal-LLM'' framework for LLM-based agents by integrating the expressiveness of natural language and the precision of formal language. Specifically, the framework allows human users to express their requirements or constraints for the planning process as an automaton. A stack-based LLM plan generation process is then conducted under the supervision of the automaton to ensure that the generated plan satisfies the constraints, making the planning process controllable. We conduct experiments on both benchmark tasks and practical real-life tasks, and o
    
[^18]: LLsM: 基于大型语言模型的生成式语言隐写术

    LLsM: Generative Linguistic Steganography with Large Language Model

    [https://arxiv.org/abs/2401.15656](https://arxiv.org/abs/2401.15656)

    本研究提出了LLsM，一种基于大型语言模型的生成式语言隐写术。通过对大规模数据集进行微调，LLM能够以可控的方式生成具有特定话语特征的隐写文本，提高了隐蔽通信的效果。

    

    语言隐写术（LS）旨在根据秘密信息生成隐写文本（stego）。只有授权接收者才能察觉文本中秘密的存在并提取出来，从而保护隐私。然而，现有方案生成的隐写文本可控性较差，很难包含特定的话语特征，如风格。结果，隐写文本容易被检测出来，危及隐蔽通信。为解决这些问题，本文提出了LLsM，第一个基于大型语言模型（LLM）的LS方法。我们使用一个包含丰富话语特征的大规模构建数据集对LLaMA2进行微调，使得微调后的LLM能够以可控的方式生成具有特定话语特征的文本。然后将话语作为引导信息和秘密一起输入给微调后的LLM，形式为“Prompt”。在此基础上，构建的候选池将进行范围编码。

    Linguistic Steganography (LS) tasks aim to generate steganographic text (stego) based on secret information. Only authorized recipients can perceive the existence of secrets in the texts and extract them, thereby preserving privacy. However, the controllability of the stego generated by existing schemes is poor, and the stego is difficult to contain specific discourse characteristics such as style. As a result, the stego is easily detectable, compromising covert communication. To address these problems, this paper proposes LLsM, the first LS with the Large Language Model (LLM). We fine-tuned the LLaMA2 with a large-scale constructed dataset encompassing rich discourse characteristics, which enables the fine-tuned LLM to generate texts with specific discourse in a controllable manner. Then the discourse is used as guiding information and inputted into the fine-tuned LLM in the form of the Prompt together with secret. On this basis, the constructed candidate pool will be range encoded an
    
[^19]: 数字苏格拉底：通过解释批评评估LLM

    Digital Socrates: Evaluating LLMs through Explanation Critiques

    [https://arxiv.org/abs/2311.09613](https://arxiv.org/abs/2311.09613)

    通过定义新的解释批评任务、创建人工验证过的数据集并训练开源自动批评模型，数字苏格拉底有助于揭示学生模型的见解。

    

    虽然LLMs可以提供有理有据的解释以及答案，但这些解释的性质和质量仍然知之甚少。作为回应，我们的目标是定义一种详细的方式来表征现代模型的解释能力，创建一个细致且可解释的解释评估工具，该工具可以自动生成这种表征，而无需依赖昂贵的API调用或人类注释。我们的方法是：(a)定义解释批评的新任务——识别和分类解释中的任何主要缺陷，并提供建议来解决这些缺陷；(b)为此任务创建一个规模可观且经过人工验证的数据集；(c)使用这些数据训练一个开源的自动批评模型（称为数字苏格拉底）。通过定量和定性分析，我们展示了数字苏格拉底如何有助于通过检查其理由来揭示有关学生模型的见解。

    arXiv:2311.09613v2 Announce Type: replace-cross  Abstract: While LLMs can provide reasoned explanations along with their answers, the nature and quality of those explanations are still poorly understood. In response, our goal is to define a detailed way of characterizing the explanation capabilities of modern models and to create a nuanced, interpretable explanation evaluation tool that can generate such characterizations automatically, without relying on expensive API calls or human annotations. Our approach is to (a) define the new task of explanation critiquing - identifying and categorizing any main flaw in an explanation and providing suggestions to address the flaw, (b) create a sizeable, human-verified dataset for this task, and (c) train an open-source, automatic critique model (called Digital Socrates) using this data. Through quantitative and qualitative analysis, we demonstrate how Digital Socrates is useful for revealing insights about student models by examining their reas
    
[^20]: MambaByte: 无标记选择性状态空间模型

    MambaByte: Token-free Selective State Space Model. (arXiv:2401.13660v1 [cs.CL])

    [http://arxiv.org/abs/2401.13660](http://arxiv.org/abs/2401.13660)

    MambaByte是一种无标记的选择性状态空间模型，通过在字节级别上进行自回归训练，解决了标准自回归Transformer在处理长序列时的性能问题，并展现了与最先进的子词Transformer相媲美甚至更优的性能，从而证明了MambaByte在无标记语言建模方面的有效性。

    

    无标记语言模型直接从原始字节学习，消除了子词标记化的偏差。然而，操作字节会导致序列长度显著增加，在这种情况下，标准自回归Transformer的扩展性较差。我们尝试了MambaByte，它是基于字节序列自回归训练的无标记适应Mamba状态空间模型。我们的实验表明，与其他字节级模型相比，MambaByte具有计算效率。我们还发现，MambaByte在性能上与甚至胜过最先进的子词Transformer。此外，由于长度的线性扩展，MambaByte在推理过程中获得了快速性能，相比之下，Transformer则没有。我们的研究结果证实了MambaByte在实现无标记语言建模方面的可行性。

    Token-free language models learn directly from raw bytes and remove the bias of subword tokenization. Operating on bytes, however, results in significantly longer sequences, and standard autoregressive Transformers scale poorly in such settings. We experiment with MambaByte, a token-free adaptation of the Mamba state space model, trained autoregressively on byte sequences. Our experiments indicate the computational efficiency of MambaByte compared to other byte-level models. We also find MambaByte to be competitive with and even outperform state-of-the-art subword Transformers. Furthermore, owing to linear scaling in length, MambaByte benefits from fast inference compared to Transformers. Our findings establish the viability of MambaByte in enabling token-free language modeling.
    
[^21]: 私有零阶优化的大型语言模型的私有微调

    Private Fine-tuning of Large Language Models with Zeroth-order Optimization. (arXiv:2401.04343v1 [cs.LG])

    [http://arxiv.org/abs/2401.04343](http://arxiv.org/abs/2401.04343)

    引入了DP-ZO，一种通过私有化零阶优化来保护大型语言模型训练数据隐私的方法。

    

    在私有数据集上对大型预训练模型进行微调可能会存在违反隐私的风险。差分隐私是一种通过强制算法稳定性来减轻隐私风险的框架。DP-SGD可以以保护隐私的方式训练具有私有数据的模型，但会带来性能损失和重大工程挑战。我们引入了DP-ZO，一种通过私有化零阶优化来保护训练数据隐私的大型语言模型微调方法。我们的方法设计的一个关键见解是，我们使用的零阶算法SPSA中的梯度方向始终是随机的，而仅依赖于私有数据的信息是步长，即一个标量。因此，我们只需要对标量步长进行隐私处理，这是存储效率高的方法。DP-ZO可以使用拉普拉斯噪声或高斯噪声来实现，在不同任务之间提供了隐私和效用之间的强大权衡。

    Fine-tuning large pretrained models on private datasets may run the risk of violating privacy. Differential privacy is a framework for mitigating privacy risks by enforcing algorithmic stability. DP-SGD enables training models with private data in a privacy-preserving manner, but raises new obstacles in the form of performance loss and significant engineering challenges. We introduce DP-ZO, a new method for fine-tuning large language models that preserves the privacy of training data by privatizing zeroth-order optimization. A key insight into the design of our method is that the direction of the gradient in SPSA, the zeroth-order algorithm we use, is always random and the only information that depends on private data is the step size, i.e., a scalar. Therefore, we only need to privatize the scalar step size, which is memory-efficient. DP-ZO, which can be instantiated with either Laplace or Gaussian noise, provides a strong privacy-utility trade-off across different tasks, and model si
    
[^22]: 作为评估器的大型语言模型中认知偏差的基准测试

    Benchmarking Cognitive Biases in Large Language Models as Evaluators. (arXiv:2309.17012v1 [cs.CL])

    [http://arxiv.org/abs/2309.17012](http://arxiv.org/abs/2309.17012)

    本研究对15个不同大小的大型语言模型进行了评估，发现它们作为评估器存在认知偏差，尤其在文本质量评估中表现出较强的偏见，这对其鲁棒性提出了质疑。同时，研究还发现了人类和机器偏好之间的相关性。

    

    最近的研究表明，大型语言模型（LLMs）通过简单的提示和上下文学习作为自动评估器非常有效。本研究组装了15个大小不同的LLMs，并通过其他LLMs的偏好排名来评估它们的输出响应，例如System Star比System Square更好。然后，我们引入了用于评估LLMs输出中六种不同认知偏差的认知偏差基准测试（CoBBLEr），如自我中心偏差，即模型更喜欢将自己的输出在评估中排名较高。我们发现LLMs是有偏见的文本质量评估器，在每个评估中都表现出对我们偏见基准的强烈迹象（在所有模型上的平均比较约为40%），这对它们作为评估器的鲁棒性提出了质疑。此外，我们还研究了人类和机器偏好之间的相关性，并计算了平均的Rank-Biased O值。

    Large Language Models (LLMs) have recently been shown to be effective as automatic evaluators with simple prompting and in-context learning. In this work, we assemble 15 LLMs of four different size ranges and evaluate their output responses by preference ranking from the other LLMs as evaluators, such as System Star is better than System Square. We then evaluate the quality of ranking outputs introducing the Cognitive Bias Benchmark for LLMs as Evaluators (CoBBLEr), a benchmark to measure six different cognitive biases in LLM evaluation outputs, such as the Egocentric bias where a model prefers to rank its own outputs highly in evaluation. We find that LLMs are biased text quality evaluators, exhibiting strong indications on our bias benchmark (average of 40% of comparisons across all models) within each of their evaluations that question their robustness as evaluators. Furthermore, we examine the correlation between human and machine preferences and calculate the average Rank-Biased O
    
[^23]: BiomedGPT：一种面向视觉、语言和多模态任务的统一且通用的生物医学生成预训练Transformer

    BiomedGPT: A Unified and Generalist Biomedical Generative Pre-trained Transformer for Vision, Language, and Multimodal Tasks. (arXiv:2305.17100v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17100](http://arxiv.org/abs/2305.17100)

    BiomedGPT是一种面向视觉、语言和多模态任务的通用生物医学生成预训练Transformer，在多个临床任务中取得了16个最新的最优结果，包括超过了OpenAI的GPT-4V和Google的Med-PaLM M（12B）。同时，BiomedGPT还支持零-shot迁移学习。

    

    传统的任务和模态特定的人工智能模型在生物医学领域的实际应用和维护中不够灵活。与此同时，生物医学数据的不断增加，结合现代多模态多任务人工智能技术的进展，为通用的生物医学人工智能解决方案的出现铺平了道路。这些解决方案有潜力解释不同的医疗模态，并产生如自由文本报告或疾病诊断等表达性输出。本文提出了BiomedGPT，这是第一个面向多样化生物医学任务的开源通用视觉语言人工智能模型。BiomedGPT在26个数据集的五个临床重要任务中实现了16个最新的结果。值得注意的是，在放射学人员评估中，它超越了OpenAI的GPT-4 with vision（GPT-4V），并在乳腺癌诊断和医学视觉问题回答方面超过了Google的Med-PaLM M（12B）。此外，BiomedGPT还支持零-shot迁移学习。

    Conventional task- and modality-specific artificial intelligence (AI) models are inflexible in real-world deployment and maintenance for biomedicine. At the same time, the growing availability of biomedical data, coupled with the advancements in modern multi-modal multi-task AI techniques, has paved the way for the emergence of generalist biomedical AI solutions. These solutions hold the potential to interpret different medical modalities and produce expressive outputs such as free-text reports or disease diagnosis. Here, we propose BiomedGPT, the first open-source and generalist visual language AI for diverse biomedical tasks. BiomedGPT achieved 16 state-of-the-art results across five clinically significant tasks on 26 datasets. Notably, it outperformed OpenAI's GPT-4 with vision (GPT-4V) in radiology human evaluation and surpassed Google's Med-PaLM M (12B) in breast cancer diagnosis and medical visual question answering. Moreover, BiomedGPT facilitates zero-shot transfer learning, gr
    
[^24]: SPARSEFIT：少样本刺激的稀疏微调，联合生成预测和自然语言解释

    SPARSEFIT: Few-shot Prompting with Sparse Fine-tuning for Jointly Generating Predictions and Natural Language Explanations. (arXiv:2305.13235v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13235](http://arxiv.org/abs/2305.13235)

    这篇论文介绍了SparseFit，一种少样本刺激的稀疏微调策略，用于联合生成预测和自然语言解释。该方法可以在只有少量自然语言解释可用时生成高质量的自然语言解释。

    

    解释神经模型的决策对于确保这些模型在部署时的可信度很关键。最近，使用自然语言解释来证明模型的预测越来越受到关注。然而，这种方法通常需要大量的人工编写的自然语言解释作为真实答案的数据集，这些数据集既昂贵又可能对于某些应用程序来说不可行。为了使模型在只有少量自然语言解释可用时生成高质量的自然语言解释，最近提出了基于刺激学习的预训练语言模型微调方法。然而，预训练语言模型通常具有数十亿个参数，使得微调十分昂贵。我们提出了SparseFit，一种稀疏的少样本微调策略，利用离散刺激来联合生成预测和自然语言解释。我们在T5模型和四个数据集上使用SparseFit，并将其与现有的参数高效微调技术进行比较。我们进行了自动和人工评估。

    Explaining the decisions of neural models is crucial for ensuring their trustworthiness at deployment time. Using Natural Language Explanations (NLEs) to justify a model's predictions has recently gained increasing interest. However, this approach usually demands large datasets of human-written NLEs for the ground-truth answers, which are expensive and potentially infeasible for some applications. For models to generate high-quality NLEs when only a few NLEs are available, the fine-tuning of Pre-trained Language Models (PLMs) in conjunction with prompt-based learning recently emerged. However, PLMs typically have billions of parameters, making fine-tuning expensive. We propose SparseFit, a sparse few-shot fine-tuning strategy that leverages discrete prompts to jointly generate predictions and NLEs. We experiment with SparseFit on the T5 model and four datasets and compare it against state-of-the-art parameter-efficient fine-tuning techniques. We perform automatic and human evaluations 
    
[^25]: 机器人还是人类？用一个问题检测ChatGPT冒名顶替者

    Bot or Human? Detecting ChatGPT Imposters with A Single Question. (arXiv:2305.06424v1 [cs.CL])

    [http://arxiv.org/abs/2305.06424](http://arxiv.org/abs/2305.06424)

    本文提出了一个名为FLAIR的框架，通过一个问题和回答来检测ChatGPT中的聊天机器人真实性，可以分类人和机器人。单问题分为对于人类而言容易但对于机器人很难和对于机器人而言容易但对于人类很难两个类别，分别进行检测。 在多个数据集上实现了最先进的性能。

    

    大型语言模型如ChatGPT最近展示了令人瞩目的自然语言理解和生成能力，使得翻译、写作和闲聊等各种应用成为可能。然而，人们担心它们可能被滥用于欺诈或拒绝服务攻击等恶意用途。因此，开发检测聊天中涉及的另一方是机器人还是人类的方法至关重要。本文提出了一个名为FLAIR的框架，即通过单个问题和回答来查找大型语言模型的真实性，以在线方式检测会话中的对话机器人。具体而言，我们针对一个单一问题场景，该场景可以有效地区分人类用户和机器人。这些问题分为两类：对于人类而言容易但对于机器人很难（例如计数、替换、定位、噪音过滤和ASCII艺术），以及对于机器人而言容易但对于人类很难（例如机器生成文本识别）。我们在多个基准数据集上评估了FLAIR，并在机器人检测方面实现了最先进的性能。

    Large language models like ChatGPT have recently demonstrated impressive capabilities in natural language understanding and generation, enabling various applications including translation, essay writing, and chit-chatting. However, there is a concern that they can be misused for malicious purposes, such as fraud or denial-of-service attacks. Therefore, it is crucial to develop methods for detecting whether the party involved in a conversation is a bot or a human. In this paper, we propose a framework named FLAIR, Finding Large language model Authenticity via a single Inquiry and Response, to detect conversational bots in an online manner. Specifically, we target a single question scenario that can effectively differentiate human users from bots. The questions are divided into two categories: those that are easy for humans but difficult for bots (e.g., counting, substitution, positioning, noise filtering, and ASCII art), and those that are easy for bots but difficult for humans (e.g., m
    
[^26]: 检查和编辑语言模型中的知识表示

    Inspecting and Editing Knowledge Representations in Language Models. (arXiv:2304.00740v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.00740](http://arxiv.org/abs/2304.00740)

    REMEDI是一种将自然语言语句映射到LM内部表示系统中的事实编码的学习方法。 REMEDI编码可用作知识编辑器，也可以用作探针，揭示了LM已经将哪些属性归因于提到的实体，并可以预测LM会生成输出的情况。

    

    神经语言模型（LMs）表示有关文本所描述世界的事实。有时这些事实来自训练数据（在大多数LMs中，“香蕉”一词的表示表示香蕉是水果的事实）。有时事实来自输入文本本身（“我倒出了瓶子”这个句子的表示表示瓶子变空了的事实）。我们描述了REMEDI，一种学习将自然语言中的语句映射到LM的内部表示系统中的事实编码的方法。 REMEDI编码可用作知识编辑器：当添加到LM隐藏表示时，它们会修改下游生成，使其与新事实一致。 REMEDI编码也可以用作探针：与LM表示进行比较时，它们揭示了LM已经将哪些属性归因于提到的实体，在某些情况下，这使得可以预测LM将生成与背景知识或输入文本冲突的输出时。因此，REMEDI链接了有关探测，PR

    Neural language models (LMs) represent facts about the world described by text. Sometimes these facts derive from training data (in most LMs, a representation of the word "banana" encodes the fact that bananas are fruits). Sometimes facts derive from input text itself (a representation of the sentence "I poured out the bottle" encodes the fact that the bottle became empty). We describe REMEDI, a method for learning to map statements in natural language to fact encodings in an LM's internal representation system. REMEDI encodings can be used as knowledge editors: when added to LM hidden representations, they modify downstream generation to be consistent with new facts. REMEDI encodings may also be used as probes: when compared to LM representations, they reveal which properties LMs already attribute to mentioned entities, in some cases making it possible to predict when LMs will generate outputs that conflict with background knowledge or input text. REMEDI thus links work on probing, pr
    
[^27]: 大型语言模型可评估新闻机构的可信度。

    Large language models can rate news outlet credibility. (arXiv:2304.00228v1 [cs.CL])

    [http://arxiv.org/abs/2304.00228](http://arxiv.org/abs/2304.00228)

    本文评估了 ChatGPT 是否能够评估新闻机构的可信度，结果表明 ChatGPT 可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明，并且这些评级与人类专家的评级相关。LLMs可以成为事实检查应用程序中可信度评级的经济参考。

    

    虽然大型语言模型（LLMs）在各种自然语言处理任务中表现出色，但它们容易产生幻象。现代最先进的聊天机器人，如新的 Bing，尝试通过直接从互联网收集信息来解决这个问题。在这种情况下，区分值得信赖的信息源对于向用户提供适当的准确性背景至关重要。本文评估了知名的LLM ChatGPT是否能够评估新闻机构的可信度。在适当的指导下，ChatGPT可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明。我们的结果表明，这些评级与人类专家的评级相关（Spearmam's $\rho=0.54, p<0.001$）。这些发现表明，LLMs可以成为事实检查应用程序中可信度评级的经济参考。未来的LLMs应增强它们的对齐性。

    Although large language models (LLMs) have shown exceptional performance in various natural language processing tasks, they are prone to hallucinations. State-of-the-art chatbots, such as the new Bing, attempt to mitigate this issue by gathering information directly from the internet to ground their answers. In this setting, the capacity to distinguish trustworthy sources is critical for providing appropriate accuracy contexts to users. Here we assess whether ChatGPT, a prominent LLM, can evaluate the credibility of news outlets. With appropriate instructions, ChatGPT can provide ratings for a diverse set of news outlets, including those in non-English languages and satirical sources, along with contextual explanations. Our results show that these ratings correlate with those from human experts (Spearmam's $\rho=0.54, p<0.001$). These findings suggest that LLMs could be an affordable reference for credibility ratings in fact-checking applications. Future LLMs should enhance their align
    
[^28]: PK-ICR: 基于角色和知识的互动上下文检索进行基于场景对话

    PK-ICR: Persona-Knowledge Interactive Context Retrieval for Grounded Dialogue. (arXiv:2302.06674v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.06674](http://arxiv.org/abs/2302.06674)

    PK-ICR是一种基于角色和知识的互动上下文检索方法，可以在复杂的多场景对话中同时识别角色和知识。通过利用神经问答检索模型，该方法可以在较少的计算资源下实现检索，并且通过引入空-正向排名测试方法来提高排名性能。

    

    鉴别与对话系统相关的角色和知识对于基于场景的对话应答生成至关重要。然而，目前每个对话基本上都是孤立研究的，而最近的工作中引入了更实际的多场景对话任务。我们将角色和知识双上下文识别定义为为给定的对话同时识别角色和知识的任务，在复杂的多场景对话设置中可能具有提升重要性。我们开发了一种新的基于检索的检索方法，可以同时利用对话的所有上下文信息。我们的方法通过使用神经问答检索模型，需要较少的计算资源。我们进一步介绍了一种新的空-正向排名测试方法，用于衡量与数据增强相关的语义差异样本（即困难负样本）的排名性能。

    Identifying relevant persona or knowledge for conversational systems is critical to grounded dialogue response generation. However, each grounding has been mostly researched in isolation with more practical multi-context dialogue tasks introduced in recent works. We define Persona and Knowledge Dual Context Identification as the task to identify persona and knowledge jointly for a given dialogue, which could be of elevated importance in complex multi-context dialogue settings. We develop a novel grounding retrieval method that utilizes all contexts of dialogue simultaneously. Our method requires less computational power via utilizing neural QA retrieval models. We further introduce our novel null-positive rank test which measures ranking performance on semantically dissimilar samples (i.e. hard negatives) in relation to data augmentation.
    
[^29]: ChatGPT发布后，立场检测技术会如何发展？

    How would Stance Detection Techniques Evolve after the Launch of ChatGPT?. (arXiv:2212.14548v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.14548](http://arxiv.org/abs/2212.14548)

    ChatGPT是一种新的预训练语言模型，可以用于解决立场检测问题，并提供了其预测的解释能力。

    

    立场检测是指从给定文本中提取对目标的立场（支持、反对或中立）的任务。随着社交媒体内容的大量增加，这方面的研究越来越受到关注。传统的处理立场检测的框架是将其转化为文本分类任务。深度学习模型已经取代了基于规则的模型和传统的机器学习模型来解决此类问题。目前的深度神经网络面临两个主要挑战，即标记数据和社交媒体帖子中的信息不足，以及深度学习模型的不可解释性。ChatGPT是一种新的预训练语言模型，于2022年11月30日发布。针对立场检测任务，我们的实验表明，ChatGPT可以在常用数据集（包括SemEval-2016和P-Stance）上实现SOTA或类似的性能。同时，ChatGPT可以为其自身的预测提供解释，这超出了任何现有模型的能力。

    Stance detection refers to the task of extracting the standpoint (Favor, Against or Neither) towards a target in given texts. Such research gains increasing attention with the proliferation of social media contents. The conventional framework of handling stance detection is converting it into text classification tasks. Deep learning models have already replaced rule-based models and traditional machine learning models in solving such problems. Current deep neural networks are facing two main challenges which are insufficient labeled data and information in social media posts and the unexplainable nature of deep learning models. A new pre-trained language model chatGPT was launched on Nov 30, 2022. For the stance detection tasks, our experiments show that ChatGPT can achieve SOTA or similar performance for commonly used datasets including SemEval-2016 and P-Stance. At the same time, ChatGPT can provide explanation for its own prediction, which is beyond the capability of any existing mo
    

