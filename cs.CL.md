# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PairEval: Open-domain Dialogue Evaluation with Pairwise Comparison](https://arxiv.org/abs/2404.01015) | 提出了PairEval，一种新颖的对话评估指标，通过将回复的质量与不同对话中的回复进行比较来评估，与人类判断具有更高的相关性。 |
| [^2] | [Learning From Correctness Without Prompting Makes LLM Efficient Reasoner](https://arxiv.org/abs/2403.19094) | 本文介绍了一种用于大型语言模型的内在自我修正推理框架LeCo，无需人类反馈、外部工具或手动提示，通过学习正确的推理步骤并基于生成logits来提高推理性能。 |
| [^3] | [BIMCV-R: A Landmark Dataset for 3D CT Text-Image Retrieval](https://arxiv.org/abs/2403.15992) | 提出了一个里程碑数据集BIMCV-R，包含8,069个3D CT体积和其放射学报告，同时开发了检索策略MedFinder，为3D医学文本图像检索领域提供了重要贡献 |
| [^4] | [Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models](https://arxiv.org/abs/2403.09635) | 提出了一个统一的信号传播理论，提供了控制transformer模型信号传播的公式，提出了DeepScaleLM初始化和缩放方案，使得可以训练非常深的模型，并发现深层模型在多个任务和数据集上胜过浅层模型。 |
| [^5] | [Evaluating the Elementary Multilingual Capabilities of Large Language Models with MultiQ](https://arxiv.org/abs/2403.03814) | 本研究通过引入MultiQ基准，调查了最先进的开放LLMs在其预期使用范围之外的基本多语能力，发现这些模型对于至少某些语言能够忠实和准确地进行回答。 |
| [^6] | [Survey in Characterization of Semantic Change](https://arxiv.org/abs/2402.19088) | 语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。 |
| [^7] | [Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/abs/2402.13950) | 本文研究了大型语言模型的推理过程中的忠实性问题，引入了FRODO框架来改进生成推理步骤和坚固推理的方法 |
| [^8] | [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) | QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。 |
| [^9] | [Common Sense Reasoning for Deep Fake Detection](https://arxiv.org/abs/2402.00126) | 该论文提出使用常识推理来建模深度伪造检测，通过扩展到Deepfake Detection VQA任务来模拟人类直觉，解释标记图像为真实或伪造的原因。 |
| [^10] | [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) | 多Agent辩论（MAD）作为增强大型语言模型（LLMs）真实性的策略，对于解决确保生成代理提供准确可靠答案的挑战具有潜力，但当前形式下的多Agent辩论系统在可靠性上不一定优于其他提示策略。 |
| [^11] | [Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation.](http://arxiv.org/abs/2401.10005) | 本文提出了一种新的方法，通过显性推理和问题生成，将大型多模态模型(LMM)赋予了显性推理能力，从而提高了推理过程的鲁棒性和可解释性。 |
| [^12] | [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.](http://arxiv.org/abs/2306.00978) | AWQ是一种激活感知的权重量化方法，通过保护少量显著权重来降低量化误差，不依赖于反向传播或重构，并在语言建模和领域特定任务上优于现有方法。 |
| [^13] | [SIFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency.](http://arxiv.org/abs/2303.11525) | 本研究提出了一种名为SIFT的方法，用于提高深度神经网络的训练效率、准确性和表示能力，通过稀疏等FLOP转换，缩短训练时间。 |
| [^14] | [Language models show human-like content effects on reasoning tasks.](http://arxiv.org/abs/2207.07051) | 本研究探讨了语言模型在逻辑推理任务中是否像人类一样通过混入内容来影响答案，结果发现大型语言模型的先验期望能够捕捉到这种特征。 |

# 详细

[^1]: PairEval：使用两两比较进行开放域对话评估

    PairEval: Open-domain Dialogue Evaluation with Pairwise Comparison

    [https://arxiv.org/abs/2404.01015](https://arxiv.org/abs/2404.01015)

    提出了PairEval，一种新颖的对话评估指标，通过将回复的质量与不同对话中的回复进行比较来评估，与人类判断具有更高的相关性。

    

    arXiv:2404.01015v1 公告类型：新 建立可靠且自动化的评估指标是开放域对话系统中必不可少但具有挑战性的问题。最近的研究提出了评估指标，通过考虑生成的回复与之前的对话历史的相关性来评估这些回复。尽管有效，但这些指标直接评估单个回复，而未考虑其相对质量与其他回复相比的情况。为了解决这个问题，我们提出了PairEval，一种新颖的对话评估指标，通过将回复的质量与不同对话中的回复进行比较来评估。PairEval建立在开源和中等规模的语言模型之上，并使其专门化于对话回复之间的两两比较。在多个基准测试上进行了大量实验证明，我们的指标与人类判断呈现出更高的相关性超过基线指标。我们还发现，所提出的比较性指标在

    arXiv:2404.01015v1 Announce Type: new  Abstract: Building a reliable and automated evaluation metric is a necessary but challenging problem for open-domain dialogue systems. Recent studies proposed evaluation metrics that assess generated responses by considering their relevance to previous dialogue histories. Although effective, these metrics evaluate individual responses directly rather than considering their relative quality compared to other responses. To handle this, we propose PairEval, a novel dialogue evaluation metric for assessing responses by comparing their quality against responses in different conversations. PairEval is built on top of open-sourced and moderate-size language models, and we make them specialized in pairwise comparison between dialogue responses. Extensive experiments on multiple benchmarks demonstrate that our metric exhibits a higher correlation with human judgments than baseline metrics. We also find that the proposed comparative metric is more robust in
    
[^2]: 没有提示的情况下学习正确性使LLM成为高效推理者

    Learning From Correctness Without Prompting Makes LLM Efficient Reasoner

    [https://arxiv.org/abs/2403.19094](https://arxiv.org/abs/2403.19094)

    本文介绍了一种用于大型语言模型的内在自我修正推理框架LeCo，无需人类反馈、外部工具或手动提示，通过学习正确的推理步骤并基于生成logits来提高推理性能。

    

    大型语言模型（LLMs）在各种任务中表现出色，但仍然存在幻觉、不忠实的推理和有毒内容等局限性。缓解这些问题的一个潜在方法是从人类或外部反馈（例如工具）中学习。本文介绍了一种用于LLMs的内在自我修正推理框架，消除了人类反馈、外部工具和手工提示的需求。提出的框架基于一种多步推理范式Learning from Correctness (LeCo)，在不需要从错误中学习的情况下提高了推理性能。该范式优先学习正确的推理步骤，并基于生成logits来衡量每个推理步骤的置信度。在各种多步推理任务上的实验结果表明，该框架在改善推理方面的有效性。

    arXiv:2403.19094v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated outstanding performance across various tasks, yet they still exhibit limitations such as hallucination, unfaithful reasoning, and toxic content. One potential approach to mitigate these issues is learning from human or external feedback (e.g. tools). In this paper, we introduce an intrinsic self-correct reasoning framework for LLMs that eliminates the need for human feedback, external tools, and handcraft prompts. The proposed framework, based on a multi-step reasoning paradigm \textbf{Le}arning from \textbf{Co}rrectness (\textsc{LeCo}), improves reasoning performance without needing to learn from errors. This paradigm prioritizes learning from correct reasoning steps, and a unique method to measure confidence for each reasoning step based on generation logits. Experimental results across various multi-step reasoning tasks demonstrate the effectiveness of the framework in improving reasoning
    
[^3]: BIMCV-R：用于3D CT文本图像检索的里程碑数据集

    BIMCV-R: A Landmark Dataset for 3D CT Text-Image Retrieval

    [https://arxiv.org/abs/2403.15992](https://arxiv.org/abs/2403.15992)

    提出了一个里程碑数据集BIMCV-R，包含8,069个3D CT体积和其放射学报告，同时开发了检索策略MedFinder，为3D医学文本图像检索领域提供了重要贡献

    

    arXiv:2403.15992v1 发布类型: 跨越  摘要: 三维医学图像与医疗保健的融合不断增加了医疗专业人员的工作量。为了帮助临床医生在诊断过程中，减轻其工作量，开发一个可靠的检索相似病例研究的系统是一个可行的解决方案。尽管这一概念有很大的潜力，但是目前3D医学文本图像检索领域受限于缺乏健全的评估基准和精心策划的数据集。为了解决这一问题，我们的研究提出了一种开创性的数据集，BIMCV-R（此数据集将在接受后发布。），其中包含了8,069个3D CT体积的广泛收集，包括超过200万张切片，以及它们各自的放射学报告。在我们数据集的基础上，我们拓展了一种检索策略，MedFinder。该方法采用双流网络架构，利用大

    arXiv:2403.15992v1 Announce Type: cross  Abstract: The burgeoning integration of 3D medical imaging into healthcare has led to a substantial increase in the workload of medical professionals. To assist clinicians in their diagnostic processes and alleviate their workload, the development of a robust system for retrieving similar case studies presents a viable solution. While the concept holds great promise, the field of 3D medical text-image retrieval is currently limited by the absence of robust evaluation benchmarks and curated datasets. To remedy this, our study presents a groundbreaking dataset, BIMCV-R (This dataset will be released upon acceptance.), which includes an extensive collection of 8,069 3D CT volumes, encompassing over 2 million slices, paired with their respective radiological reports. Expanding upon the foundational work of our dataset, we craft a retrieval strategy, MedFinder. This approach employs a dual-stream network architecture, harnessing the potential of larg
    
[^4]: Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models

    Transformers Get Stable: An End-to-End Signal Propagation Theory for Language Models

    [https://arxiv.org/abs/2403.09635](https://arxiv.org/abs/2403.09635)

    提出了一个统一的信号传播理论，提供了控制transformer模型信号传播的公式，提出了DeepScaleLM初始化和缩放方案，使得可以训练非常深的模型，并发现深层模型在多个任务和数据集上胜过浅层模型。

    

    尽管transformer模型取得了巨大的成功，但在深度方面仍然很难扩展。本研究提出了一个统一的信号传播理论，并提供了控制transformer模型前向和反向信号矩的公式。我们的框架可以用于理解和缓解与高注意力分数相关的梯度消失/爆炸、秩坍缩和不稳定性。我们还提出了DeepScaleLM，一种初始化和缩放方案，通过该方案能够在模型中保持单位输出/梯度矩，从而使训练具有100多层的非常深模型成为可能。我们发现，transformer模型可以更深 - 我们的深层模型在语言建模、语音翻译和图像分类方面表现优异，包括仅编码器、仅解码器和编码器-解码器变体，适用于Pre-LN和Post-LN transformers，适用于多个数据集和模型大小。

    arXiv:2403.09635v1 Announce Type: cross  Abstract: In spite of their huge success, transformer models remain difficult to scale in depth. In this work, we develop a unified signal propagation theory and provide formulae that govern the moments of the forward and backward signal through the transformer model. Our framework can be used to understand and mitigate vanishing/exploding gradients, rank collapse, and instability associated with high attention scores. We also propose DeepScaleLM, an initialization and scaling scheme that conserves unit output/gradient moments throughout the model, enabling the training of very deep models with 100s of layers. We find that transformer models could be much deeper - our deep models with fewer parameters outperform shallow models in Language Modeling, Speech Translation, and Image Classification, across Encoder-only, Decoder-only and Encoder-Decoder variants, for both Pre-LN and Post-LN transformers, for multiple datasets and model sizes. These imp
    
[^5]: 用MultiQ评估大型语言模型的基本多语能力

    Evaluating the Elementary Multilingual Capabilities of Large Language Models with MultiQ

    [https://arxiv.org/abs/2403.03814](https://arxiv.org/abs/2403.03814)

    本研究通过引入MultiQ基准，调查了最先进的开放LLMs在其预期使用范围之外的基本多语能力，发现这些模型对于至少某些语言能够忠实和准确地进行回答。

    

    大型语言模型（LLMs）需要为全球大多数非英语使用者提供服务。然而，大多数LLMs今天，特别是开放的LLMs，通常仅意为在英语（例如Llama2、Mistral）或少数几种高资源语言（例如Mixtral、Qwen）中使用。最近的研究表明，尽管存在使用上的限制，人们会用许多不同的语言提示LLMs。因此，在本文中，我们调查了最先进的开放LLMs在其预期使用范围之外的基本多语能力。为此，我们引入了MultiQ，一个新的用于基本开放式问答的银标准基准，涵盖137种语言的27.4k个测试问题。通过MultiQ，我们评估了语言忠实度，即模型是否以提示的语言回复，以及问题回答准确性。我们测试的所有LLMs对至少某些语言响应得忠实和/或准确。

    arXiv:2403.03814v1 Announce Type: cross  Abstract: Large language models (LLMs) need to serve everyone, including a global majority of non-English speakers. However, most LLMs today, and open LLMs in particular, are often intended for use in just English (e.g. Llama2, Mistral) or a small handful of high-resource languages (e.g. Mixtral, Qwen). Recent research shows that, despite limits in their intended use, people prompt LLMs in many different languages. Therefore, in this paper, we investigate the basic multilingual capabilities of state-of-the-art open LLMs beyond their intended use. For this purpose, we introduce MultiQ, a new silver standard benchmark for basic open-ended question answering with 27.4k test questions across a typologically diverse set of 137 languages. With MultiQ, we evaluate language fidelity, i.e.\ whether models respond in the prompted language, and question answering accuracy. All LLMs we test respond faithfully and/or accurately for at least some languages be
    
[^6]: 对语义变化特征的调查

    Survey in Characterization of Semantic Change

    [https://arxiv.org/abs/2402.19088](https://arxiv.org/abs/2402.19088)

    语义变化对计算语言学算法的结果质量可能会产生影响，因此重要性日益凸显。

    

    活语言不断发展，以吸纳人类社会的文化变化。这种演变通过新词语（新单词）或单词的语义变化（赋予已有单词新的含义）来体现。理解单词的含义对解释来自不同文化（地方用语或俚语）、领域（例如技术术语）或时代的文本至关重要。在计算机科学中，这些单词与计算语言学算法相关，例如翻译、信息检索、问答等。语义变化可能会影响这些算法的结果质量。因此，了解和形式化表征这些变化是很重要的。研究这种影响是计算语言学界近期引起关注的问题。几种方法提出了检测语义变化的方法，具有较高的精度，但需要更多努力来对其进行表征。

    arXiv:2402.19088v1 Announce Type: cross  Abstract: Live languages continuously evolve to integrate the cultural change of human societies. This evolution manifests through neologisms (new words) or \textbf{semantic changes} of words (new meaning to existing words). Understanding the meaning of words is vital for interpreting texts coming from different cultures (regionalism or slang), domains (e.g., technical terms), or periods. In computer science, these words are relevant to computational linguistics algorithms such as translation, information retrieval, question answering, etc. Semantic changes can potentially impact the quality of the outcomes of these algorithms. Therefore, it is important to understand and characterize these changes formally. The study of this impact is a recent problem that has attracted the attention of the computational linguistics community. Several approaches propose methods to detect semantic changes with good precision, but more effort is needed to charact
    
[^7]: 使推理变得重要：衡量和提高链式思维推理的忠实性

    Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2402.13950](https://arxiv.org/abs/2402.13950)

    本文研究了大型语言模型的推理过程中的忠实性问题，引入了FRODO框架来改进生成推理步骤和坚固推理的方法

    

    大型语言模型(LLMs)在回答问题之前经过逐步推理已被证明表现更好。然而，模型最终答案与所述推理步骤的忠实程度尚不明确。本文对十二个LLMs进行因果中介分析，以检验LLM生成的中间推理步骤如何影响最终结果，并发现LLMs在生成答案时并不可靠地使用其中间推理步骤。为了解决这个问题，我们介绍了FRODO，一个旨在定制小型LM以生成正确推理步骤并在这些步骤上进行坚固推理的框架。FRODO包括一个推断模块，通过学习使用隐式因果奖励函数生成正确推理步骤，并且一个推理模块，通过学习使用反事实和因果偏好目标在这些中间推理上忠实推理。我们的实验证明F

    arXiv:2402.13950v1 Announce Type: new  Abstract: Large language models (LLMs) have been shown to perform better when asked to reason step-by-step before answering a question. However, it is unclear to what degree the model's final answer is faithful to the stated reasoning steps. In this paper, we perform a causal mediation analysis on twelve LLMs to examine how intermediate reasoning steps generated by the LLM influence the final outcome and find that LLMs do not reliably use their intermediate reasoning steps when generating an answer. To address this issue, we introduce FRODO, a framework to tailor small-sized LMs to generate correct reasoning steps and robustly reason over these steps. FRODO consists of an inference module that learns to generate correct reasoning steps using an implicit causal reward function and a reasoning module that learns to faithfully reason over these intermediate inferences using a counterfactual and causal preference objective. Our experiments show that F
    
[^8]: 选择高质量数据用于训练语言模型的QuRating方法

    QuRating: Selecting High-Quality Data for Training Language Models

    [https://arxiv.org/abs/2402.09739](https://arxiv.org/abs/2402.09739)

    QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。

    

    选择高质量的预训练数据对于创建能力强的语言模型很重要，但现有方法依赖简单的启发式方法。我们介绍了一种名为QuRating的方法，用于选择能够捕捉人类直观感知的文本的抽象特征的预训练文本数据。在本文中，我们研究了四个特征 - 写作风格、所需专业知识、事实和琐事以及教育价值。我们发现，语言模型能够辨别这些特征，并观察到它们在进行文本的配对判断方面比直接评估文本质量更好。我们训练了一个QuRater模型，从配对判断中学习标量评分，并使用它为260B的训练语料库中的每个标准进行质量评级注释。在实验中，我们根据不同的质量评级选择了30B个令牌，并在所选数据上训练了13亿参数的语言模型。我们发现在质量和多样性之间保持平衡是很重要的。

    arXiv:2402.09739v1 Announce Type: new  Abstract: Selecting high-quality pre-training data is important for creating capable language models, but existing methods rely on simple heuristics. We introduce QuRating, a method for selecting pre-training data that captures the abstract qualities of texts which humans intuitively perceive. In this paper, we investigate four qualities - writing style, required expertise, facts & trivia, and educational value. We find that LLMs are able to discern these qualities and observe that they are better at making pairwise judgments of texts than at rating the quality of a text directly. We train a QuRater model to learn scalar ratings from pairwise judgments, and use it to annotate a 260B training corpus with quality ratings for each of the four criteria. In our experiments, we select 30B tokens according to the different quality ratings and train 1.3B-parameter language models on the selected data. We find that it is important to balance quality and di
    
[^9]: 深度伪造检测的常识推理

    Common Sense Reasoning for Deep Fake Detection

    [https://arxiv.org/abs/2402.00126](https://arxiv.org/abs/2402.00126)

    该论文提出使用常识推理来建模深度伪造检测，通过扩展到Deepfake Detection VQA任务来模拟人类直觉，解释标记图像为真实或伪造的原因。

    

    最先进的方法依赖于通过神经网络提取的基于图像的特征进行深度伪造检测二分类。虽然这些方法在监督训练下提取了可能的伪造特征，但它们可能无法有效表示不自然的“非物理”语义面部属性 - 模糊的发际线、双眉毛、僵硬的瞳孔或不自然的皮肤着色。然而，这类面部属性通常通过常识推理对人类来说很容易感知。此外，通过显著性图提供视觉解释的基于图像的特征提取方法可能很难被人类解释。为了解决这些挑战，我们建议使用常识推理来建模深度伪造检测，并将其扩展到Deepfake Detection VQA（DD-VQA）任务，目的是模拟人类直觉来解释标记图像为真实或伪造的原因。为此，我们引入了一个新的数据集，为与深度伪造检测相关的问题提供答案。

    State-of-the-art approaches rely on image-based features extracted via neural networks for the deepfake detection binary classification. While these approaches trained in the supervised sense extract likely fake features, they may fall short in representing unnatural `non-physical' semantic facial attributes -- blurry hairlines, double eyebrows, rigid eye pupils, or unnatural skin shading. However, such facial attributes are generally easily perceived by humans via common sense reasoning. Furthermore, image-based feature extraction methods that provide visual explanation via saliency maps can be hard to be interpreted by humans. To address these challenges, we propose the use of common sense reasoning to model deepfake detection, and extend it to the Deepfake Detection VQA (DD-VQA) task with the aim to model human intuition in explaining the reason behind labeling an image as either real or fake. To this end, we introduce a new dataset that provides answers to the questions related to 
    
[^10]: 我们应该疯狂吗？多Agent辩论策略对LLMs的影响

    Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs

    [https://arxiv.org/abs/2311.17371](https://arxiv.org/abs/2311.17371)

    多Agent辩论（MAD）作为增强大型语言模型（LLMs）真实性的策略，对于解决确保生成代理提供准确可靠答案的挑战具有潜力，但当前形式下的多Agent辩论系统在可靠性上不一定优于其他提示策略。

    

    最近大型语言模型（LLMs）的发展突显了它们在回答各种领域问题方面的潜力。然而，确保生成代理提供准确可靠的答案仍然是一个持续挑战。在这种背景下，多Agent辩论（MAD）已成为增强LLMs真实性的一种有前途的策略。我们对一系列辩论和提示策略进行基准测试，探讨成本、时间和准确性之间的权衡。重要的是，我们发现，目前形式下的多Agent辩论系统在可靠性上不一定优于其他建议的提示策略，如自一致性和使用多个推理路径进行集成。但是，在执行超参数调整时，一些MAD系统，如Multi-Persona，表现更好。这表明MAD协议可能并不会比其他方法天然更差，而是更容易受到不同超参数的影响。

    arXiv:2311.17371v2 Announce Type: replace-cross  Abstract: Recent advancements in large language models (LLMs) underscore their potential for responding to inquiries in various domains. However, ensuring that generative agents provide accurate and reliable answers remains an ongoing challenge. In this context, multi-agent debate (MAD) has emerged as a promising strategy for enhancing the truthfulness of LLMs. We benchmark a range of debating and prompting strategies to explore the trade-offs between cost, time, and accuracy. Importantly, we find that multi-agent debating systems, in their current form, do not reliably outperform other proposed prompting strategies, such as self-consistency and ensembling using multiple reasoning paths. However, when performing hyperparameter tuning, several MAD systems, such as Multi-Persona, perform better. This suggests that MAD protocols might not be inherently worse than other approaches, but that they are more sensitive to different hyperparameter
    
[^11]: 以显性推理链和视觉问题生成推进大型多模态模型

    Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation. (arXiv:2401.10005v1 [cs.CV])

    [http://arxiv.org/abs/2401.10005](http://arxiv.org/abs/2401.10005)

    本文提出了一种新的方法，通过显性推理和问题生成，将大型多模态模型(LMM)赋予了显性推理能力，从而提高了推理过程的鲁棒性和可解释性。

    

    随着对能够解释和推理视觉内容的智能系统需求越来越高，需要开发不仅准确而且具有显性推理能力的大型多模态模型（LMMs）。本文提出了一种新颖的方法，将显性推理能力赋予LMMs，基于视觉内容和文本指导进行显性推理。我们引入了一个系统，可以提问以获取必要的知识，从而增强推理过程的鲁棒性和可解释性。我们的方法包括通过一个大型语言模型（LLM）生成的新颖数据集的开发，旨在促进思维链推理与提问机制的结合。我们设计了一个高度具有区域意识的LMM，以解决图像-文本对齐的复杂需求。该模型经历了三个阶段的训练，首先是使用大规模数据集进行大规模图像-文本对齐，接下来是通过显式推理的问题生成阶段。

    The increasing demand for intelligent systems capable of interpreting and reasoning about visual content requires the development of Large Multi-Modal Models (LMMs) that are not only accurate but also have explicit reasoning capabilities. This paper presents a novel approach to imbue an LMM with the ability to conduct explicit reasoning based on visual content and textual instructions. We introduce a system that can ask a question to acquire necessary knowledge, thereby enhancing the robustness and explicability of the reasoning process. Our method comprises the development of a novel dataset generated by a Large Language Model (LLM), designed to promote chain-of-thought reasoning combined with a question-asking mechanism. We designed an LMM, which has high capabilities on region awareness to address the intricate requirements of image-text alignment. The model undergoes a three-stage training phase, starting with large-scale image-text alignment using a large-scale datasets, followed 
    
[^12]: AWQ：LLM压缩与加速的激活感知权重量化方法

    AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. (arXiv:2306.00978v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.00978](http://arxiv.org/abs/2306.00978)

    AWQ是一种激活感知的权重量化方法，通过保护少量显著权重来降低量化误差，不依赖于反向传播或重构，并在语言建模和领域特定任务上优于现有方法。

    

    大语言模型(LLM)在各种任务上展现出出色的性能，但巨大的模型大小提高了为服务(内存大小)带来的硬件障碍，并降低了令牌生成速度(内存带宽)。本文提出了一种名为激活感知权重量化(AWQ)的硬件友好方法，用于LLM低比特权重量化。我们的方法基于一个观察：权重并不是等重要的；仅保护1%的显著权重就能大大降低量化误差。我们提出寻找通过观察激活值而不是权重来保护显著权重的最佳按通道缩放方法。AWQ不依赖于任何反向传播或重构，因此可以很好地保持LLM在不同领域和模式下的泛化能力，而不会过度拟合校准集。AWQ在各种语言建模和领域特定基准测试上优于现有方法。由于更好的泛化能力，它实现了优秀的量化效果。

    Large language models (LLMs) have shown excellent performance on various tasks, but the astronomical model size raises the hardware barrier for serving (memory size) and slows down token generation (memory bandwidth). In this paper, we propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization. Our method is based on the observation that weights are not equally important: protecting only 1% of salient weights can greatly reduce quantization error. We then propose to search for the optimal per-channel scaling that protects the salient weights by observing the activation, not weights. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs' generalization ability on different domains and modalities, without overfitting to the calibration set. AWQ outperforms existing work on various language modeling and domain-specific benchmarks. Thanks to better generalization, it achieves excellent quantiz
    
[^13]: SIFT: 稀疏等FLOP转换以最大限度提高训练效率

    SIFT: Sparse Iso-FLOP Transformations for Maximizing Training Efficiency. (arXiv:2303.11525v1 [cs.LG])

    [http://arxiv.org/abs/2303.11525](http://arxiv.org/abs/2303.11525)

    本研究提出了一种名为SIFT的方法，用于提高深度神经网络的训练效率、准确性和表示能力，通过稀疏等FLOP转换，缩短训练时间。

    

    最近的研究探索了使用权重稀疏性来改善深度神经网络（DNN）的训练效率（与训练FLOPS相关的测试准确性）。 这些工作旨在减少训练FLOP，但使用稀疏权重进行训练通常会导致准确性损失或需要更长的训练周期，使得结果的训练效率不够清晰。 相比之下，我们专注于使用稀疏性提高准确性，同时使用与密集模型相同的FLOPS，并通过更高的准确性展示训练效率提高。 在本文中，我们介绍了SIFT，一组用作密集层的即插即用替代品来提高其表示能力和FLOP效率的稀疏等FLOP转换。 每个转换都由一个单一参数（稀疏级别）参数化，并提供更大的搜索空间以找到最佳的稀疏掩膜。

    Recent works have explored the use of weight sparsity to improve the training efficiency (test accuracy w.r.t training FLOPs) of deep neural networks (DNNs). These works aim to reduce training FLOPs but training with sparse weights often leads to accuracy loss or requires longer train schedules, making the resulting training efficiency less clear. In contrast, we focus on using sparsity to increase accuracy while using the same FLOPS as the dense model and show training efficiency gains through higher accuracy. In this work, we introduce SIFT, a family of Sparse Iso-FLOP Transformations which are used as drop-in replacements for dense layers to improve their representational capacity and FLOP efficiency. Each transformation is parameterized by a single parameter (sparsity level) and provides a larger search space to find optimal sparse masks. Without changing any training hyperparameters, replacing dense layers with SIFT leads to significant improvements across computer vision (CV) and
    
[^14]: 语言模型显示对推理任务具有类似人类的内容效应

    Language models show human-like content effects on reasoning tasks. (arXiv:2207.07051v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.07051](http://arxiv.org/abs/2207.07051)

    本研究探讨了语言模型在逻辑推理任务中是否像人类一样通过混入内容来影响答案，结果发现大型语言模型的先验期望能够捕捉到这种特征。

    

    抽象推理是智能系统的关键能力。大型语言模型在抽象推理任务上实现了高于随机的性能，但存在许多不完善之处。然而，人类的抽象推理也是不完美的。例如，人类推理受到我们对真实世界的知识和信念的影响，并表现出显著的“内容效应”；当问题的语义内容支持正确的逻辑推理时，人类更可靠地进行推理。这些内容纠缠的推理模式在关于人类智能基本性质的争论中起着核心作用。在这里，我们研究了语言模型是否以类似的方式混入内容来回答逻辑问题，这些语言模型的先验期望捕捉了一些人类知识的特征。我们在三个逻辑推理任务上探索了这个问题：自然语言推理、判断三段论的逻辑有效性和Wason选择任务。我们评估了最先进的大型语言模型的性能。

    Abstract reasoning is a key ability for an intelligent system. Large language models (LMs) achieve above-chance performance on abstract reasoning tasks, but exhibit many imperfections. However, human abstract reasoning is also imperfect. For example, human reasoning is affected by our real-world knowledge and beliefs, and shows notable "content effects"; humans reason more reliably when the semantic content of a problem supports the correct logical inferences. These content-entangled reasoning patterns play a central role in debates about the fundamental nature of human intelligence. Here, we investigate whether language models $\unicode{x2014}$ whose prior expectations capture some aspects of human knowledge $\unicode{x2014}$ similarly mix content into their answers to logical problems. We explored this question across three logical reasoning tasks: natural language inference, judging the logical validity of syllogisms, and the Wason selection task. We evaluate state of the art large 
    

