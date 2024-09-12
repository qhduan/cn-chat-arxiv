# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation](https://arxiv.org/abs/2402.18334) | Bonito是一种用于生成指令调优训练数据集的模型，通过将未注释的文本转换为特定任务训练数据，实现大型语言模型对用户专属数据的零shot任务适应，并显著提高了预训练和指令调整模型的平均性能。 |
| [^3] | [MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs](https://arxiv.org/abs/2402.16352) | MathGenie通过问题反向翻译生成合成数据，用于增强LLMs的数学推理能力，并创造了一个家族化的模型系列MathGenieLM。 |
| [^4] | [CriticBench: Evaluating Large Language Models as Critic](https://arxiv.org/abs/2402.13764) | CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。 |
| [^5] | [WebLINX: Real-World Website Navigation with Multi-Turn Dialogue](https://arxiv.org/abs/2402.05930) | 本论文提出了对话式网站导航的问题，并设计了一个 WEBLINX 基准测试，用于训练和评估代理。为了解决大量信息的处理瓶颈，文中提出了一个受检索启发的模型。实验结果表明，该模型能够在多种场景下复制人类行为的能力。 |
| [^6] | [Explaining Text Classifiers with Counterfactual Representations](https://arxiv.org/abs/2402.00711) | 本论文提出了一种使用反事实表示解释文本分类器的方法，通过干预文本表示来生成反事实，并通过实验证实了方法的有效性。 |
| [^7] | [With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation](https://arxiv.org/abs/2401.11504) | Temp-Lora方法通过在长文本生成过程中逐步训练临时Lora模块，有效保留上下文知识并避免对模型参数的永久性改变。 |
| [^8] | [Explainable Identification of Hate Speech towards Islam using Graph Neural Networks](https://arxiv.org/abs/2311.04916) | 使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。 |
| [^9] | [Extreme Compression of Large Language Models via Additive Quantization.](http://arxiv.org/abs/2401.06118) | 本文提出的算法在大规模语言模型的极端压缩方面取得了较好的性能，相比最新技术，在给定的压缩预算下准确性更高。 |
| [^10] | [Re-parameterized Low-rank Prompt: Generalize a Vision-Language Model within 0.5K Parameters.](http://arxiv.org/abs/2312.10813) | 该论文提出了一种新型的提示方法，重新参数化低秩提示（RLP），用于在大型预训练视觉语言模型的适应过程中实现高效和有效的知识转移。该方法能够显著减少可调参数和存储开销。 |
| [^11] | [Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach.](http://arxiv.org/abs/2310.11616) | 本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。 |
| [^12] | [Testing the Predictions of Surprisal Theory in 11 Languages.](http://arxiv.org/abs/2307.03667) | 本研究填补了现有文献中的空白，通过研究11种不同语言之间的surprisal与阅读时间之间的关系，测试了Surprisal理论的三个预测，并发现了其他语言特征对阅读时间的影响。 |
| [^13] | [Exploring Spoken Named Entity Recognition: A Cross-Lingual Perspective.](http://arxiv.org/abs/2307.01310) | 本研究探索了口述命名实体识别（NER）的跨语言视角。通过使用荷兰语、英语和德语之间的迁移学习，以及管道和端到端方案，利用自定义的伪标注数据集和Wav2Vec2-XLS-R模型，研究了几种适应跨语言系统的架构。结果显示，端到端口述NER在有限的标注数据上表现出优于管道系统的性能。值得注意的是，从德语到荷兰语的迁移学习取得了较好的效果，超过了荷兰语系统的性能。 |
| [^14] | [RRWKV: Capturing Long-range Dependencies in RWKV.](http://arxiv.org/abs/2306.05176) | 本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。 |
| [^15] | [Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors.](http://arxiv.org/abs/2305.14450) | 本研究从性能、度量标准、鲁棒性和错误类型四个方面评估ChatGPT的信息抽取能力，发现了ChatGPT与SOTA结果之间存在巨大的性能差距，同时提出了一种软匹配策略以更准确地反映ChatGPT的性能。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 学习生成用于零shot任务适应的指令调优数据集

    Learning to Generate Instruction Tuning Datasets for Zero-Shot Task Adaptation

    [https://arxiv.org/abs/2402.18334](https://arxiv.org/abs/2402.18334)

    Bonito是一种用于生成指令调优训练数据集的模型，通过将未注释的文本转换为特定任务训练数据，实现大型语言模型对用户专属数据的零shot任务适应，并显著提高了预训练和指令调整模型的平均性能。

    

    我们介绍了Bonito，这是一个开源模型，用于条件任务生成：将未注释的文本转换为用于指令调优的特定任务训练数据集。我们的目标是在用户专门的私人数据上实现大型语言模型的零shot任务适应。我们使用1.65M个示例的新大规模数据集训练Bonito，该数据集是通过将现有的指令调优数据集重新混合成元模板而创建的。数据集的元模板产生训练示例，其中输入是未注释的文本和任务属性，输出包括指令和响应。我们使用Bonito为七个专业领域的数据集生成合成任务，跨三种任务类型 -- 是非问答、抽取式问答和自然语言推理 -- 并调整语言模型。我们展示了Bonito显著改善了预训练和指令调整模型的平均性能。

    arXiv:2402.18334v1 Announce Type: new  Abstract: We introduce Bonito, an open-source model for conditional task generation: the task of converting unannotated text into task-specific training datasets for instruction tuning. Our goal is to enable zero-shot task adaptation of large language models on users' specialized, private data. We train Bonito on a new large-scale dataset with 1.65M examples created by remixing existing instruction tuning datasets into meta-templates. The meta-templates for a dataset produce training examples where the input is the unannotated text and the task attribute and the output consists of the instruction and the response. We use Bonito to generate synthetic tasks for seven datasets from specialized domains across three task types -- yes-no question answering, extractive question answering, and natural language inference -- and adapt language models. We show that Bonito significantly improves the average performance of pretrained and instruction tuned mode
    
[^3]: MathGenie: 使用问题反向翻译生成合成数据，以增强LLMs的数学推理能力

    MathGenie: Generating Synthetic Data with Question Back-translation for Enhancing Mathematical Reasoning of LLMs

    [https://arxiv.org/abs/2402.16352](https://arxiv.org/abs/2402.16352)

    MathGenie通过问题反向翻译生成合成数据，用于增强LLMs的数学推理能力，并创造了一个家族化的模型系列MathGenieLM。

    

    大型语言模型(LLMs)在数学推理方面展现出巨大潜力。然而，目前开源模型和GPT-4等闭源模型之间在这一领域仍存在性能差距。本文介绍了一种新颖的方法MathGenie，用于从小规模问题-解决方案数据集（称为种子数据）中生成多样且可靠的数学问题。我们扩充了种子数据的真实解决方案，并训练了一个反向翻译模型，将扩充的解决方案翻译回新问题。随后，我们为新问题生成了集成代码解决方案。为确保集成代码解决方案的正确性，我们采用了基于原理的解决方案验证策略。我们在新筛选的数据上对从7B到70B不等的各种预训练模型进行训练，以测试所提出的增强技术的有效性，从而产生了一个称为MathGenieLM的模型系列。

    arXiv:2402.16352v1 Announce Type: cross  Abstract: Large language models (LLMs) have exhibited great potential in mathematical reasoning. However, there remains a performance gap in this area between existing open-source models and closed-source models such as GPT-4. In this paper, we introduce MathGenie, a novel method for generating diverse and reliable math problems from a small-scale problem-solution dataset (denoted as seed data). We augment the ground-truth solutions of our seed data and train a back-translation model to translate the augmented solutions back into new questions. Subsequently, we generate code-integrated solutions for the new questions. To ensure the correctness of the code-integrated solutions, we employ rationale-based strategy for solution verification. Various pretrained models, ranging from 7B to 70B, are trained on the newly curated data to test the effectiveness of the proposed augmentation technique, resulting in a family of models known as MathGenieLM. Th
    
[^4]: CriticBench: 将大型语言模型作为评论家进行评估

    CriticBench: Evaluating Large Language Models as Critic

    [https://arxiv.org/abs/2402.13764](https://arxiv.org/abs/2402.13764)

    CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。

    

    论文提出了 CriticBench，这是一个旨在全面和可靠地评估大型语言模型（LLMs）的四个关键评论能力维度（反馈、比较、改进和元反馈）的新型基准。CriticBench包含九个不同的任务，每个任务评估LLMs在不同质量细粒度水平上评论响应的能力。对开源和闭源LLMs进行的广泛评估揭示了评论能力与任务、响应质量和模型规模之间有趣的关系。CriticBench的数据集、资源和评估工具包将在https://github.com/gmftbyGMFTBY/Cri上公开发布。

    arXiv:2402.13764v1 Announce Type: cross  Abstract: Critique ability are crucial in the scalable oversight and self-improvement of Large Language Models (LLMs). While many recent studies explore the critique ability of LLMs to judge and refine flaws in generations, how to comprehensively and reliably measure the critique abilities of LLMs is under-explored. This paper introduces \shortname, a novel benchmark designed to comprehensively and reliably evaluate four key critique ability dimensions of LLMs: feedback, comparison, refinement and meta-feedback. \shortname~encompasses nine diverse tasks, each assessing the LLMs' ability to critique responses at varying levels of quality granularity. Our extensive evaluations of open-source and closed-source LLMs reveal intriguing relationships between the critique ability and tasks, response qualities, and model scales. Datasets, resources and evaluation toolkit for \shortname~will be publicly released at \url{https://github.com/gmftbyGMFTBY/Cri
    
[^5]: WebLINX: 多轮对话下的真实世界网站导航

    WebLINX: Real-World Website Navigation with Multi-Turn Dialogue

    [https://arxiv.org/abs/2402.05930](https://arxiv.org/abs/2402.05930)

    本论文提出了对话式网站导航的问题，并设计了一个 WEBLINX 基准测试，用于训练和评估代理。为了解决大量信息的处理瓶颈，文中提出了一个受检索启发的模型。实验结果表明，该模型能够在多种场景下复制人类行为的能力。

    

    我们提出了对话式网站导航的问题，其中数字代理控制着一个网页浏览器，并按照用户的指令以多轮对话的方式解决真实世界任务。为了支持这个问题，我们引入了 WEBLINX - 一个100K交互的大规模基准测试，在2300个专家演示中进行了对话式网站导航的测试。我们的基准涵盖了150多个真实世界网站上的广泛模式，可以用于在不同场景下训练和评估代理。由于存在大量信息，大型语言模型 (LLMs) 无法实时处理整个网页。为了解决这个瓶颈，我们设计了一个受检索启发的模型，通过排名相关元素来高效地修剪 HTML 页面。我们使用选定的元素，以及屏幕截图和操作历史记录，评估各种模型在导航网页时复制人类行为的能力。我们的实验从小型纯文本模型到专有的多模态 LLMs 进行了测试。

    We propose the problem of conversational web navigation, where a digital agent controls a web browser and follows user instructions to solve real-world tasks in a multi-turn dialogue fashion. To support this problem, we introduce WEBLINX - a large-scale benchmark of 100K interactions across 2300 expert demonstrations of conversational web navigation. Our benchmark covers a broad range of patterns on over 150 real-world websites and can be used to train and evaluate agents in diverse scenarios. Due to the magnitude of information present, Large Language Models (LLMs) cannot process entire web pages in real-time. To solve this bottleneck, we design a retrieval-inspired model that efficiently prunes HTML pages by ranking relevant elements. We use the selected elements, along with screenshots and action history, to assess a variety of models for their ability to replicate human behavior when navigating the web. Our experiments span from small text-only to proprietary multimodal LLMs. We fi
    
[^6]: 使用反事实表示解释文本分类器

    Explaining Text Classifiers with Counterfactual Representations

    [https://arxiv.org/abs/2402.00711](https://arxiv.org/abs/2402.00711)

    本论文提出了一种使用反事实表示解释文本分类器的方法，通过干预文本表示来生成反事实，并通过实验证实了方法的有效性。

    

    一种基于反事实的解释方法可以为分类器提供合理的解释，其中反事实是指除了一个分类特征之外，与真实观察完全相同的假设事件。然而，在文本领域构建这种反事实存在特定挑战，因为某些属性值可能与现实世界的事件不一致。在这篇论文中，我们提出了一种简单的方法，通过对文本表示进行干预来生成反事实，从而克服了这个限制。我们认为我们的干预方法是最小程度的干扰，并且在理论上是可靠的，因为它们与Pearl的因果推断框架中定义的反事实是一致的。为了验证我们的方法，我们首先在合成数据集上进行实验，比较了基于真实反事实（通过明确的文本干预获得）和我们的反事实（通过对文本表示的干预得到）的分类器预测。

    One well motivated explanation method for classifiers leverages counterfactuals which are hypothetical events identical to real observations in all aspects except for one categorical feature. Constructing such counterfactual poses specific challenges for texts, however, as some attribute values may not necessarily align with plausible real-world events. In this paper we propose a simple method for generating counterfactuals by intervening in the space of text representations which bypasses this limitation. We argue that our interventions are minimally disruptive and that they are theoretically sound as they align with counterfactuals as defined in Pearl's causal inference framework. To validate our method, we first conduct experiments on a synthetic dataset of counterfactuals, allowing for a direct comparison between classifier predictions based on ground truth counterfactuals (obtained through explicit text interventions) and our counterfactuals, derived through interventions in the r
    
[^7]: 随着文本量增加，推断训练有助于长文本生成

    With Greater Text Comes Greater Necessity: Inference-Time Training Helps Long Text Generation

    [https://arxiv.org/abs/2401.11504](https://arxiv.org/abs/2401.11504)

    Temp-Lora方法通过在长文本生成过程中逐步训练临时Lora模块，有效保留上下文知识并避免对模型参数的永久性改变。

    

    长文本生成，如小说创作和具有极长上下文的篇章级翻译，对当前的语言模型提出了重大挑战。现有方法主要集中在通过长度外推等策略扩展模型的上下文窗口。然而，这些方法在训练和/或推断阶段要求大量硬件资源。我们提出的方法Temp-Lora引入了一个替代概念。我们不依赖于KV缓存存储所有上下文信息，而是将这些信息直接嵌入临时Lora模块中。在长文本生成过程中，这个模块会随着先前生成的文本逐渐进行训练。这种方法不仅有效地保留上下文知识，还防止了对模型参数的任何永久性改变，因为模块在生成后被丢弃。在PG19语言建模上进行了大量实验。

    arXiv:2401.11504v2 Announce Type: replace-cross  Abstract: Long text generation, such as novel writing and discourse-level translation with extremely long contexts, presents significant challenges to current language models. Existing methods mainly focus on extending the model's context window through strategies like length extrapolation. However, these approaches demand substantial hardware resources during the training and/or inference phases. Our proposed method, Temp-Lora, introduces an alternative concept. Instead of relying on the KV cache to store all context information, we embeds this information directly into a temporary Lora module. In the process of long text generation, this module is progressively trained with text generated previously. This approach not only efficiently preserves contextual knowledge but also prevents any permanent alteration to the model's parameters given that the module is discarded post-generation. Extensive experiments on the PG19 language modeling 
    
[^8]: 使用图神经网络解释伊斯兰教仇恨言论的研究

    Explainable Identification of Hate Speech towards Islam using Graph Neural Networks

    [https://arxiv.org/abs/2311.04916](https://arxiv.org/abs/2311.04916)

    使用图神经网络解释和识别伊斯兰教仇恨言论，模型在保持出色性能的同时能够解释相关性和因果关系。

    

    伊斯兰教仇恨言论在在线社交互动平台上是一个普遍存在的挑战。识别和消除这种仇恨是迈向和谐与和平未来的关键一步。本研究提出了一种新的范例，利用图神经网络来识别和解释针对伊斯兰教的仇恨言论。利用图神经网络发现、提取并利用不同数据点之间的关系的内在能力，我们的模型始终能够在保持出色性能的同时提供对潜在相关性和因果关系的解释。

    arXiv:2311.04916v2 Announce Type: cross  Abstract: Islamophobic language is a prevalent challenge on online social interaction platforms. Identifying and eliminating such hatred is a crucial step towards a future of harmony and peace. This study presents a novel paradigm for identifying and explaining hate speech towards Islam using graph neural networks. Utilizing the intrinsic ability of graph neural networks to find, extract, and use relationships across disparate data points, our model consistently achieves outstanding performance while offering explanations for the underlying correlations and causation.
    
[^9]: 大规模语言模型的极端压缩通过加性量化

    Extreme Compression of Large Language Models via Additive Quantization. (arXiv:2401.06118v1 [cs.LG])

    [http://arxiv.org/abs/2401.06118](http://arxiv.org/abs/2401.06118)

    本文提出的算法在大规模语言模型的极端压缩方面取得了较好的性能，相比最新技术，在给定的压缩预算下准确性更高。

    

    准确的开源大规模语言模型(LLMs)的出现引发了对这些模型进行量化技术的竞赛，从而使其能够在最终用户设备上执行。在本文中，我们从多码本量化(MCQ)的经典方法角度重新思考了“极端”LLM压缩的问题，即针对非常低的位数，例如每个参数2到3位。我们的工作建立在加性量化这一经典算法之上，并将其适应于语言模型的量化。由此得到的算法在LLM压缩方面推进了最新技术，以给定压缩预算的准确性而言，优于所有最近提出的技术。例如，当将Llama 2模型压缩到每个参数2位时，我们的算法将7B模型量化为6.93困惑度(相对于之前最佳工作改进1.29，相对于FP16改进1.81)，13B模型量化为5.70困惑度(改进0.36)，70B模型量化为3.94困惑度。

    The emergence of accurate open large language models (LLMs) has led to a race towards quantization techniques for such models enabling execution on end-user devices. In this paper, we revisit the problem of "extreme" LLM compression--defined as targeting extremely low bit counts, such as 2 to 3 bits per parameter, from the point of view of classic methods in Multi-Codebook Quantization (MCQ). Our work builds on top of Additive Quantization, a classic algorithm from the MCQ family, and adapts it to the quantization of language models. The resulting algorithm advances the state-of-the-art in LLM compression, outperforming all recently-proposed techniques in terms of accuracy at a given compression budget. For instance, when compressing Llama 2 models to 2 bits per parameter, our algorithm quantizes the 7B model to 6.93 perplexity (a 1.29 improvement relative to the best prior work, and 1.81 points from FP16), the 13B model to 5.70 perplexity (a .36 improvement) and the 70B model to 3.94 
    
[^10]: 重新参数化低秩提示：在0.5K参数内推广视觉语言模型

    Re-parameterized Low-rank Prompt: Generalize a Vision-Language Model within 0.5K Parameters. (arXiv:2312.10813v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.10813](http://arxiv.org/abs/2312.10813)

    该论文提出了一种新型的提示方法，重新参数化低秩提示（RLP），用于在大型预训练视觉语言模型的适应过程中实现高效和有效的知识转移。该方法能够显著减少可调参数和存储开销。

    

    随着大型预训练视觉语言模型的发展，如何有效地将这些基础模型的知识转移到下游任务中成为一个热门话题，尤其是在数据不足的情况下。最近，提示调优已成为一种流行的解决方案。在调整视觉语言模型时，研究人员冻结骨干部分的参数，只设计和调整提示。一方面，提示调优的精心设计展现出强大的性能。另一方面，复杂的结构和更新规则大大增加了计算和存储成本。受到观察到的视觉语言模型中泛化能力的演变模式与适应过程中提示矩阵秩变化趋势的调和一致性的启发，我们设计了一种新型提示，重新参数化低秩提示（RLP），用于高效和有效的适应。我们的方法能大大减少可调参数和存储开销。

    With the development of large pre-trained vision-language models, how to effectively transfer the knowledge of such foundational models to downstream tasks becomes a hot topic, especially in a data-deficient scenario. Recently, prompt tuning has become a popular solution. When adapting the vision-language models, researchers freeze the parameters in the backbone and only design and tune the prompts. On the one hand, the delicate design of prompt tuning exhibits strong performance. On the other hand, complicated structures and update rules largely increase the computation and storage cost. Motivated by the observation that the evolution pattern of the generalization capability in visual-language models aligns harmoniously with the trend of rank variations in the prompt matrix during adaptation, we design a new type of prompt, Re-parameterized Low-rank Prompt (RLP), for both efficient and effective adaptation. Our method could largely reduce the number of tunable parameters and storage s
    
[^11]: 揭示语言模型中的普遍智能因子：一种心理测量方法

    Unveiling the General Intelligence Factor in Language Models: A Psychometric Approach. (arXiv:2310.11616v1 [cs.CL])

    [http://arxiv.org/abs/2310.11616](http://arxiv.org/abs/2310.11616)

    本研究利用心理测量理论揭示了语言模型中的普遍智能因子g的存在，并发现了该因子解释模型性能方差的85%，为模型评估和开发提供了统一的指标。

    

    本研究采用心理测量理论，揭示了语言模型中普遍智能因子g的存在，并扩展了该理论在人类和某些动物物种中的应用。通过对两个大型数据集Open LLM Leaderboard（包含1,232个模型）和General Language Understanding Evaluation（GLUE）Leaderboard（包含88个模型）进行因子分析，我们发现了一个具有一维性和高度稳定性的g因子，可以解释模型性能方差的85%。研究还发现模型大小和g之间的中度相关性为0.48。在语言模型中发现g因子为模型评估提供了统一的指标，为更强大、基于g因子的模型能力评估开辟了新的途径。这些发现为从心理测量的角度理解和未来研究人工智能提供了基础，并对模型评估和开发具有实际意义。

    This study uncovers the factor of general intelligence, or g, in language models, extending the psychometric theory traditionally applied to humans and certain animal species. Utilizing factor analysis on two extensive datasets Open LLM Leaderboard with 1,232 models and General Language Understanding Evaluation (GLUE) Leaderboard with 88 models - we find compelling evidence for a unidimensional, highly stable g factor that accounts for 85% of the variance in model performance. The study also finds a moderate correlation of .48 between model size and g. The discovery of g in language models offers a unified metric for model evaluation and opens new avenues for more robust, g-based model ability assessment. These findings lay the foundation for understanding and future research on artificial general intelligence from a psychometric perspective and have practical implications for model evaluation and development.
    
[^12]: 在11种语言中测试Surprisal理论的预测

    Testing the Predictions of Surprisal Theory in 11 Languages. (arXiv:2307.03667v1 [cs.CL])

    [http://arxiv.org/abs/2307.03667](http://arxiv.org/abs/2307.03667)

    本研究填补了现有文献中的空白，通过研究11种不同语言之间的surprisal与阅读时间之间的关系，测试了Surprisal理论的三个预测，并发现了其他语言特征对阅读时间的影响。

    

    心理语言学的一个基本结果是，可预测性较低的词语需要更长时间来处理。Surprisal理论（Hale, 2001; Levy, 2008）是对这一发现的一个理论解释，它将一个词的可预测性量化为其surprisal，即在给定上下文的情况下，其负对数概率。虽然有大量的证据支持Surprisal理论的预测，但大多数研究都集中在一个非常有限的数据范围内，即以英语为母语的人阅读英语文本。事实上，目前还没有全面的多语言分析。我们通过研究在五个语言家族中分布的十一种不同语言中surprisal与阅读时间之间的关系来填补当前文献中的这一空白。通过从单语和多语语料库训练的语言模型中推导估计值，我们测试了与surprisal理论相关的三个预测：(i) surprisal是否能够预测阅读时间；(ii) 预期surprisal，即上下文熵，是否影响阅读时间；(iii) 与surprisal相关的其他语言特征是否可以解释阅读时间。

    A fundamental result in psycholinguistics is that less predictable words take a longer time to process. One theoretical explanation for this finding is Surprisal Theory (Hale, 2001; Levy, 2008), which quantifies a word's predictability as its surprisal, i.e. its negative log-probability given a context. While evidence supporting the predictions of Surprisal Theory have been replicated widely, most have focused on a very narrow slice of data: native English speakers reading English texts. Indeed, no comprehensive multilingual analysis exists. We address this gap in the current literature by investigating the relationship between surprisal and reading times in eleven different languages, distributed across five language families. Deriving estimates from language models trained on monolingual and multilingual corpora, we test three predictions associated with surprisal theory: (i) whether surprisal is predictive of reading times; (ii) whether expected surprisal, i.e. contextual entropy, i
    
[^13]: 探索口述命名实体识别：跨语言视角

    Exploring Spoken Named Entity Recognition: A Cross-Lingual Perspective. (arXiv:2307.01310v1 [cs.CL])

    [http://arxiv.org/abs/2307.01310](http://arxiv.org/abs/2307.01310)

    本研究探索了口述命名实体识别（NER）的跨语言视角。通过使用荷兰语、英语和德语之间的迁移学习，以及管道和端到端方案，利用自定义的伪标注数据集和Wav2Vec2-XLS-R模型，研究了几种适应跨语言系统的架构。结果显示，端到端口述NER在有限的标注数据上表现出优于管道系统的性能。值得注意的是，从德语到荷兰语的迁移学习取得了较好的效果，超过了荷兰语系统的性能。

    

    最近在命名实体识别（NER）方面取得了显著进展，提高了文本数据中实体的识别能力。然而，口述NER作为口述文档检索的专门领域，由于研究有限和数据稀缺而滞后。此外，口述NER中的跨语言迁移学习仍未被探索。本文利用荷兰语、英语和德语之间的迁移学习，使用管道和端到端（E2E）方案。我们利用自定义的伪标注数据集使用Wav2Vec2-XLS-R模型，并研究了几种适应跨语言系统的架构。我们的结果表明，端到端口述NER在我们有限的标注数据上优于基于管道的替代方案。值得注意的是，从德语到荷兰语的迁移学习超过了荷兰语E2E系统7%和荷兰语管道系统4%。这项研究不仅突出了口述NER中迁移学习的可行性，而且为未来的评估提供了有希望的结果。

    Recent advancements in Named Entity Recognition (NER) have significantly improved the identification of entities in textual data. However, spoken NER, a specialized field of spoken document retrieval, lags behind due to its limited research and scarce datasets. Moreover, cross-lingual transfer learning in spoken NER has remained unexplored. This paper utilizes transfer learning across Dutch, English, and German using pipeline and End-to-End (E2E) schemes. We employ Wav2Vec2-XLS-R models on custom pseudo-annotated datasets and investigate several architectures for the adaptability of cross-lingual systems. Our results demonstrate that End-to-End spoken NER outperforms pipeline-based alternatives over our limited annotations. Notably, transfer learning from German to Dutch surpasses the Dutch E2E system by 7% and the Dutch pipeline system by 4%. This study not only underscores the feasibility of transfer learning in spoken NER but also sets promising outcomes for future evaluations, hint
    
[^14]: RRWKV：在RWKV中捕捉长距离依赖关系

    RRWKV: Capturing Long-range Dependencies in RWKV. (arXiv:2306.05176v1 [cs.CL])

    [http://arxiv.org/abs/2306.05176](http://arxiv.org/abs/2306.05176)

    本文介绍了一种新的RRWKV架构，它在保持记忆和计算效率的同时，通过加入回顾能力有效地捕捉长距离依赖关系。

    

    由于Transformer惊人的点积注意力，它已经成为各种自然语言处理（NLP）任务中的主要架构。最近，Receptance Weighted Key Value（RWKV）架构遵循非Transformer架构，消除了点积注意力的缺点，其中存储和计算复杂度随着序列长度呈二次扩展。尽管RWKV利用了线性张量积注意机制并通过部署时间序列模式实现了并行计算，但与标准Transformer中直接交互获得的完整信息相比，它无法捕捉长距离依赖关系，因为其受限于向后查看先前信息的能力。因此，本文通过将回顾能力纳入RWKV中来设计Retrospected Receptance Weighted Key Value（RRWKV）架构，以有效地吸收信息，同时保持记忆和计算效率。

    Owing to the impressive dot-product attention, the Transformers have been the dominant architectures in various natural language processing (NLP) tasks. Recently, the Receptance Weighted Key Value (RWKV) architecture follows a non-transformer architecture to eliminate the drawbacks of dot-product attention, where memory and computational complexity exhibits quadratic scaling with sequence length. Although RWKV has exploited a linearly tensor-product attention mechanism and achieved parallelized computations by deploying the time-sequential mode, it fails to capture long-range dependencies because of its limitation on looking back at previous information, compared with full information obtained by direct interactions in the standard transformer. Therefore, the paper devises the Retrospected Receptance Weighted Key Value (RRWKV) architecture via incorporating the retrospecting ability into the RWKV to effectively absorb information, which maintains memory and computational efficiency as 
    
[^15]: ChatGPT是否解决了信息抽取问题？性能、度量标准、鲁棒性和错误分析

    Is Information Extraction Solved by ChatGPT? An Analysis of Performance, Evaluation Criteria, Robustness and Errors. (arXiv:2305.14450v1 [cs.CL])

    [http://arxiv.org/abs/2305.14450](http://arxiv.org/abs/2305.14450)

    本研究从性能、度量标准、鲁棒性和错误类型四个方面评估ChatGPT的信息抽取能力，发现了ChatGPT与SOTA结果之间存在巨大的性能差距，同时提出了一种软匹配策略以更准确地反映ChatGPT的性能。

    

    ChatGPT激发了大型语言模型领域的研究热潮。本文从性能、度量标准、鲁棒性和错误类型四个方面评估了ChatGPT的能力。具体而言，我们在零样本、小样本和思考串联等场景下对17个数据集的14个IE子任务评估了ChatGPT的性能，并发现ChatGPT与SOTA结果之间存在巨大的性能差距。接下来，我们重新思考这种差距，并提出一种软匹配策略以更准确地反映ChatGPT的性能。然后，我们分析了ChatGPT在14个IE子任务上的鲁棒性，并发现：1）ChatGPT很少输出无效的响应；2）不相关的上下文和长尾目标类型极大地影响了ChatGPT的性能；3）ChatGPT无法很好地理解RE任务中的主客体关系。最后，我们分析了ChatGPT的错误，并发现“未注释的跨度”是最主要的错误类型。这引起了有关现实场景下信息提取性能的担忧。

    ChatGPT has stimulated the research boom in the field of large language models. In this paper, we assess the capabilities of ChatGPT from four perspectives including Performance, Evaluation Criteria, Robustness and Error Types. Specifically, we first evaluate ChatGPT's performance on 17 datasets with 14 IE sub-tasks under the zero-shot, few-shot and chain-of-thought scenarios, and find a huge performance gap between ChatGPT and SOTA results. Next, we rethink this gap and propose a soft-matching strategy for evaluation to more accurately reflect ChatGPT's performance. Then, we analyze the robustness of ChatGPT on 14 IE sub-tasks, and find that: 1) ChatGPT rarely outputs invalid responses; 2) Irrelevant context and long-tail target types greatly affect ChatGPT's performance; 3) ChatGPT cannot understand well the subject-object relationships in RE task. Finally, we analyze the errors of ChatGPT, and find that "unannotated spans" is the most dominant error type. This raises concerns about 
    

