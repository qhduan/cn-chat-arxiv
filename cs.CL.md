# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness](https://rss.arxiv.org/abs/2401.15963) | 本论文提出了一个新的基准测试 NoFunEval，用于评估代码语言模型在非功能性要求和简单分类实例方面的表现。研究发现，目前的代码语言模型在处理这些要求时存在根本性的盲点。 |
| [^2] | [FABLES: Evaluating faithfulness and content selection in book-length summarization](https://arxiv.org/abs/2404.01261) | 本文首次对LLM生成的虚构书籍摘要进行了忠实性和内容选择的大规模人类评估，建立了FABLES数据集，通过对26本书的3158个声明进行了注释，成功对LLM摘要进行了基于忠实性的排名 |
| [^3] | [Tastle: Distract Large Language Models for Automatic Jailbreak Attack](https://arxiv.org/abs/2403.08424) | Tastle是一种新颖的黑盒越狱框架，采用恶意内容隐藏和内存重构以及迭代优化算法，用于自动对大型语言模型进行红队攻击。 |
| [^4] | [VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models](https://arxiv.org/abs/2403.06098) | VidProM是一个包含167万个独特文本到视频提示的大规模数据集，对于文本到视频扩散模型带来了新的研究进展，揭示了真实用户提示对视频生成的重要性。 |
| [^5] | [WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off](https://arxiv.org/abs/2403.04808) | WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。 |
| [^6] | [Negating Negatives: Alignment without Human Positive Samples via Distributional Dispreference Optimization](https://arxiv.org/abs/2403.03419) | 通过提出Distributional Dispreference Optimization (D$^2$O)方法，在不需要人类正样本的情况下实现了对齐，减少了有害信息的传播。 |
| [^7] | [Cause and Effect: Can Large Language Models Truly Understand Causality?](https://arxiv.org/abs/2402.18139) | 本研究提出了一种名为CARE CA的新型架构，通过结合显式因果检测模块和反事实陈述、以及隐含因果检测模块，旨在增强大型语言模型对因果关系的理解能力。 |
| [^8] | [AmbigNLG: Addressing Task Ambiguity in Instruction for NLG](https://arxiv.org/abs/2402.17717) | AmbigNLG是一个旨在解决自然语言生成任务中指令模糊性挑战的新任务，通过识别和减轻指令中的模糊性，改进了文本生成质量，并突出了清晰和具体指令在提升LLM在NLG任务中表现的关键作用。 |
| [^9] | [Chain-of-Discussion: A Multi-Model Framework for Complex Evidence-Based Question Answering](https://arxiv.org/abs/2402.16313) | 提出了一种Chain-of-Discussion框架，通过多个开源语言模型的协同作用，提高了复杂问题回答的质量 |
| [^10] | [PANDA (Pedantic ANswer-correctness Determination and Adjudication):Improving Automatic Evaluation for Question Answering and Text Generation](https://arxiv.org/abs/2402.11161) | 提出了PANDA方法，引入了更精确的答案正确性评测方式，解决了当前自动评估问答和文本生成过程中的挑战。 |
| [^11] | [Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models](https://arxiv.org/abs/2402.10612) | 本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。 |
| [^12] | [Policy Improvement using Language Feedback Models](https://arxiv.org/abs/2402.07876) | 本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。 |
| [^13] | [Systematic Biases in LLM Simulations of Debates](https://arxiv.org/abs/2402.04049) | 本研究揭示了LLMs在模拟政治辩论中存在的系统性偏差，尽管被指定从特定的政治观点进行辩论，LLMs代理机构倾向于遵循模型固有的社会偏见。通过自动自我优化方法，我们进一步证实了这些观察结果。 |
| [^14] | [OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking](https://arxiv.org/abs/2311.09758) | 本研究提出了一种新颖的SLM/LLM路由框架，以提高计算效率和增强任务性能，通过利用结构化知识提取任务中SLMs和LLMs的互补优势，从而降低成本而不牺牲性能。 |
| [^15] | [TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data.](http://arxiv.org/abs/2401.13223) | TAT-LLM是一种专门用于离散推理的语言模型，针对混合表格和文本数据上的问答任务。该模型通过分步流水线的方式，包括提取器、推理器和执行器，利用LLMs的强大能力来解决问题。而为了应对成本、延迟和数据安全风险等挑战，我们开发了TAT-LLM，一个专门针对此任务的较小LLM。 |
| [^16] | [Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities.](http://arxiv.org/abs/2401.11143) | 该论文提出了一个名为GAAM的多头高斯自适应注意力机制，用于增强跨多个模态的信息聚合。通过将可学习的均值和方差纳入注意力机制中，GAAM能够动态地重新调整特征的重要性，从而在处理非平稳数据时取得了显著的性能提升，超过了目前现有的注意力技术。该方法的适应性强且参数数量较少，具有改进现有注意力框架的潜力。 |
| [^17] | [Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs.](http://arxiv.org/abs/2401.10065) | 本论文研究了在大型语言模型（LLMs）中触发条件推理能力的方法，通过使用代码提示将自然语言问题转化为代码，从而在多个数据集上实现了显著的性能提升。 |
| [^18] | [A Joint-Reasoning based Disease Q&A System.](http://arxiv.org/abs/2401.03181) | 这项研究提出了一种基于联合推理的疾病问答系统，通过结合语言模型和知识图谱的方法，旨在回答普通用户的健康相关问题并减轻医疗保健专业人员的负担。 |
| [^19] | [Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition.](http://arxiv.org/abs/2309.10524) | 本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。 |
| [^20] | [Interpretable Stereotype Identification through Reasoning.](http://arxiv.org/abs/2308.00071) | 本研究通过使用推理方法，在零射击刻板印象识别中取得了重要的进展，并发现推理的性能增益远远超过模型规模扩展的增益。推理不仅提高了准确性，还提高了决策的可解释性。 |
| [^21] | [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations.](http://arxiv.org/abs/2305.16326) | 本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。 |
| [^22] | [Whisper-KDQ: A Lightweight Whisper via Guided Knowledge Distillation and Quantization for Efficient ASR.](http://arxiv.org/abs/2305.10788) | 本文提出了一种通过引导知识蒸馏和量化，实现对大型预训练语音识别模型Whisper进行压缩优化的方法，可以将模型大小缩小并提高性能。 |

# 详细

[^1]: NoFunEval: 有趣的是，代码语言模型在超出功能正确性的要求上遇到困难

    NoFunEval: Funny How Code LMs Falter on Requirements Beyond Functional Correctness

    [https://rss.arxiv.org/abs/2401.15963](https://rss.arxiv.org/abs/2401.15963)

    本论文提出了一个新的基准测试 NoFunEval，用于评估代码语言模型在非功能性要求和简单分类实例方面的表现。研究发现，目前的代码语言模型在处理这些要求时存在根本性的盲点。

    

    现有的代码语言模型（code LMs）的评估基准几乎完全集中在LMs是否能够生成功能正确的代码上。在实际的软件工程中，开发人员会考虑超出功能正确性的要求。他们对于“如何”实现功能有着对整体系统设计目标（如效率、安全性和可维护性）的要求。如果LMs能够展示对要求和代码语义的强大理解能力，他们也会更加信任这些LMs。我们提出了一个新的基准测试NoFunEval来评估代码LMs在非功能性要求和简单分类实例方面的表现。我们提出了一个提示方法Coding Concepts (CoCo)，可以用于开发人员向LMs传达领域知识。我们对22个代码LMs进行了广泛评估，发现它们在我们的基准测试中普遍表现不佳，暗示着它们在处理这些问题时存在根本性的盲点。

    Existing evaluation benchmarks of language models of code (code LMs) focus almost exclusively on whether the LMs can generate functionally-correct code. In real-world software engineering, developers think beyond functional correctness. They have requirements on "how" a functionality should be implemented to meet overall system design objectives like efficiency, security, and maintainability. They would also trust the code LMs more if the LMs demonstrate robust understanding of requirements and code semantics.   We propose a new benchmark NoFunEval to evaluate code LMs on non-functional requirements and simple classification instances for both functional and non-functional requirements. We propose a prompting method, Coding Concepts (CoCo), as a way for a developer to communicate the domain knowledge to the LMs. We conduct an extensive evaluation of twenty-two code LMs. Our finding is that they generally falter when tested on our benchmark, hinting at fundamental blindspots in their tr
    
[^2]: FABLES：评估书籍摘要中的忠实性和内容选择

    FABLES: Evaluating faithfulness and content selection in book-length summarization

    [https://arxiv.org/abs/2404.01261](https://arxiv.org/abs/2404.01261)

    本文首次对LLM生成的虚构书籍摘要进行了忠实性和内容选择的大规模人类评估，建立了FABLES数据集，通过对26本书的3158个声明进行了注释，成功对LLM摘要进行了基于忠实性的排名

    

    虽然长文本大语言模型（LLMs）在技术上可以总结长达100K个标记的书籍，但迄今为止，文档的长度和复杂性阻碍了对忠实性等输入相关方面的评估。本文在虚构书籍的LLM生成摘要上进行了首次大规模人类评估，通过专注于2023或2024年出版的书籍摘要，雇佣在进行注释任务之前已完全阅读每本书的注释者来减少成本和认知负担，从而缓解了数据污染问题。我们收集了FABLES数据集，对26本书的LLM生成摘要中的3158个声明进行了注释，花费了5200美元，这使我们能够基于忠实性对LLM摘要进行排名：Claude-3-Opus在忠实性方面明显优于所有闭源LLMs，而开源的Mixtral与GPT-3.5-Turbo持平。

    arXiv:2404.01261v1 Announce Type: cross  Abstract: While long-context large language models (LLMs) can technically summarize book-length documents (>100K tokens), the length and complexity of the documents have so far prohibited evaluations of input-dependent aspects like faithfulness. In this paper, we conduct the first large-scale human evaluation of faithfulness and content selection on LLM-generated summaries of fictional books. Our study mitigates the issue of data contamination by focusing on summaries of books published in 2023 or 2024, and we hire annotators who have fully read each book prior to the annotation task to minimize cost and cognitive burden. We collect FABLES, a dataset of annotations on 3,158 claims made in LLM-generated summaries of 26 books, at a cost of $5.2K USD, which allows us to rank LLM summarizers based on faithfulness: Claude-3-Opus significantly outperforms all closed-source LLMs, while the open-source Mixtral is on par with GPT-3.5-Turbo. An analysis o
    
[^3]: Tastle: 为自动越狱攻击干扰大型语言模型

    Tastle: Distract Large Language Models for Automatic Jailbreak Attack

    [https://arxiv.org/abs/2403.08424](https://arxiv.org/abs/2403.08424)

    Tastle是一种新颖的黑盒越狱框架，采用恶意内容隐藏和内存重构以及迭代优化算法，用于自动对大型语言模型进行红队攻击。

    

    大型语言模型（LLMs）近年来取得了重要进展。在LLMs公开发布之前，人们已经做出了大量努力来将它们的行为与人类价值观保持一致。对齐的主要目标是确保它们的有益性、诚实性和无害性。然而，即使经过细致对齐的LLMs仍然容易受到恶意操纵，如越狱，导致意外的行为。越狱是有意开发恶意提示，从LLM安全限制中逃脱以生成未经审查的有害内容。以前的工作探索了不同的越狱方法来对LLMs进行红队攻击，但它们在效果和可伸缩性方面遇到了挑战。在这项工作中，我们提出了Tastle，一种新颖的黑盒越狱框架，用于自动对LLMs进行红队攻击。我们设计了恶意内容隐藏和内存重构，并结合迭代优化算法来越狱LLMs。

    arXiv:2403.08424v1 Announce Type: cross  Abstract: Large language models (LLMs) have achieved significant advances in recent days. Extensive efforts have been made before the public release of LLMs to align their behaviors with human values. The primary goal of alignment is to ensure their helpfulness, honesty and harmlessness. However, even meticulously aligned LLMs remain vulnerable to malicious manipulations such as jailbreaking, leading to unintended behaviors. The jailbreak is to intentionally develop a malicious prompt that escapes from the LLM security restrictions to produce uncensored detrimental contents. Previous works explore different jailbreak methods for red teaming LLMs, yet they encounter challenges regarding to effectiveness and scalability. In this work, we propose Tastle, a novel black-box jailbreak framework for automated red teaming of LLMs. We designed malicious content concealing and memory reframing with an iterative optimization algorithm to jailbreak LLMs, mo
    
[^4]: VidProM：一个百万规模的真实即时图库数据集，用于文本到视频扩散模型

    VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models

    [https://arxiv.org/abs/2403.06098](https://arxiv.org/abs/2403.06098)

    VidProM是一个包含167万个独特文本到视频提示的大规模数据集，对于文本到视频扩散模型带来了新的研究进展，揭示了真实用户提示对视频生成的重要性。

    

    Sora的到来标志着文本到视频扩散模型的新时代的到来，带来了视频生成和潜在应用方面的显著进步。然而，Sora以及其他文本到视频扩散模型高度依赖提示，但目前尚没有公开可用的包含文本到视频提示研究的数据集。本文介绍了VidProM，这是第一个由167万个来自真实用户的独特文本到视频提示组成的大规模数据集。此外，该数据集包括由四种最先进的扩散模型生成的669万个视频以及一些相关数据。我们首先展示了这一大规模数据集的策展过程，这是一个耗时且昂贵的过程。随后，我们展示了所提出的VidProM与DiffusionDB之间的区别，后者是一个用于图像生成的大规模提示图库数据集。通过对这些提示的分析，我们确定了一个专门的新提示数据集的必要性。

    arXiv:2403.06098v1 Announce Type: cross  Abstract: The arrival of Sora marks a new era for text-to-video diffusion models, bringing significant advancements in video generation and potential applications. However, Sora, as well as other text-to-video diffusion models, highly relies on the prompts, and there is no publicly available dataset featuring a study of text-to-video prompts. In this paper, we introduce VidProM, the first large-scale dataset comprising 1.67 million unique text-to-video prompts from real users. Additionally, the dataset includes 6.69 million videos generated by four state-of-the-art diffusion models and some related data. We initially demonstrate the curation of this large-scale dataset, which is a time-consuming and costly process. Subsequently, we show how the proposed VidProM differs from DiffusionDB, a large-scale prompt-gallery dataset for image generation. Based on the analysis of these prompts, we identify the necessity for a new prompt dataset specificall
    
[^5]: WaterMax: 打破LLM水印可检测性-稳健性-质量的平衡

    WaterMax: breaking the LLM watermark detectability-robustness-quality trade-off

    [https://arxiv.org/abs/2403.04808](https://arxiv.org/abs/2403.04808)

    WaterMax提出了一种新的水印方案，能够在保持生成文本质量的同时实现高检测性能，打破了水印技术中质量和稳健性之间的传统平衡。

    

    水印是阻止大型语言模型被恶意使用的技术手段。本文提出了一种称为WaterMax的新颖水印方案，具有高检测性能，同时保持原始LLM生成文本的质量。其新设计不会对LLM进行任何修改（不调整权重、对数、温度或采样技术）。WaterMax平衡了稳健性和复杂性，与文献中的水印技术相反，从根本上引发了质量和稳健性之间的平衡。其性能在理论上得到证明并经过实验证实。在最全面的基准测试套件下，它胜过所有的最先进技术。

    arXiv:2403.04808v1 Announce Type: cross  Abstract: Watermarking is a technical means to dissuade malfeasant usage of Large Language Models. This paper proposes a novel watermarking scheme, so-called WaterMax, that enjoys high detectability while sustaining the quality of the generated text of the original LLM. Its new design leaves the LLM untouched (no modification of the weights, logits, temperature, or sampling technique). WaterMax balances robustness and complexity contrary to the watermarking techniques of the literature inherently provoking a trade-off between quality and robustness. Its performance is both theoretically proven and experimentally validated. It outperforms all the SotA techniques under the most complete benchmark suite.
    
[^6]: 否定否定：通过分布式反喜好优化实现对齐而无需人类正样本

    Negating Negatives: Alignment without Human Positive Samples via Distributional Dispreference Optimization

    [https://arxiv.org/abs/2403.03419](https://arxiv.org/abs/2403.03419)

    通过提出Distributional Dispreference Optimization (D$^2$O)方法，在不需要人类正样本的情况下实现了对齐，减少了有害信息的传播。

    

    大型语言模型（LLM）改变了人工智能的角色，但也可能存在传播不道德内容的潜在风险。对齐技术被引入以引导LLM朝着人类偏好方向发展，并受到越来越多的关注。尽管在这个方向上取得了显著突破，但现有方法严重依赖于高质量的正负训练对，受到嘈杂标签和首选和非首选响应数据之间的边缘区别的困扰。鉴于最近LLM在生成有用响应方面的高水平，本文将研究重点转向一个新的方向：仅使用人工注释的负样本来实现对齐，保留有用性的同时降低有害性。为此，我们提出了分布式反喜好优化（D$^2$O），通过最大化生成的响应与非首选响应之间的差异，有效地排除有害信息。我们在理论上证明

    arXiv:2403.03419v1 Announce Type: cross  Abstract: Large language models (LLMs) have revolutionized the role of AI, yet also pose potential risks of propagating unethical content. Alignment technologies have been introduced to steer LLMs towards human preference, gaining increasing attention. Despite notable breakthroughs in this direction, existing methods heavily rely on high-quality positive-negative training pairs, suffering from noisy labels and the marginal distinction between preferred and dispreferred response data. Given recent LLMs' proficiency in generating helpful responses, this work pivots towards a new research focus: achieving alignment using solely human-annotated negative samples, preserving helpfulness while reducing harmfulness. For this purpose, we propose Distributional Dispreference Optimization (D$^2$O), which maximizes the discrepancy between the generated responses and the dispreferred ones to effectively eschew harmful information. We theoretically demonstrat
    
[^7]: 因果关系：大型语言模型真正理解因果关系吗？

    Cause and Effect: Can Large Language Models Truly Understand Causality?

    [https://arxiv.org/abs/2402.18139](https://arxiv.org/abs/2402.18139)

    本研究提出了一种名为CARE CA的新型架构，通过结合显式因果检测模块和反事实陈述、以及隐含因果检测模块，旨在增强大型语言模型对因果关系的理解能力。

    

    随着大型语言模型（LLMs）的兴起，理解它们在解读和解释语言所涉及的复杂因果关系的能力和局限性变得至关重要。目前的方法使用明确或隐含的因果推理，然而迫切需要一种统一的方法，将两者结合起来更有效地处理各种因果关系。本研究提出了一种新颖的架构，称为具有反事实分析的上下文感知推理增强（CARE CA）框架，以增强因果推理和可解释性。所提出的框架将 ConceptNet 和反事实陈述中的明确因果检测模块以及通过LLMs进行的隐含因果检测相结合。我们的框架通过一层反事实解释进一步突出LLMs对因果关系的理解。ConceptNet 中的知识提高了多

    arXiv:2402.18139v1 Announce Type: cross  Abstract: With the rise of Large Language Models(LLMs), it has become crucial to understand their capabilities and limitations in deciphering and explaining the complex web of causal relationships that language entails. Current methods use either explicit or implicit causal reasoning, yet there is a strong need for a unified approach combining both to tackle a wide array of causal relationships more effectively. This research proposes a novel architecture called Context Aware Reasoning Enhancement with Counterfactual Analysis(CARE CA) framework to enhance causal reasoning and explainability. The proposed framework incorporates an explicit causal detection module with ConceptNet and counterfactual statements, as well as implicit causal detection through LLMs. Our framework goes one step further with a layer of counterfactual explanations to accentuate LLMs understanding of causality. The knowledge from ConceptNet enhances the performance of multi
    
[^8]: AmbigNLG: 解决NLG指令中的任务模糊性问题

    AmbigNLG: Addressing Task Ambiguity in Instruction for NLG

    [https://arxiv.org/abs/2402.17717](https://arxiv.org/abs/2402.17717)

    AmbigNLG是一个旨在解决自然语言生成任务中指令模糊性挑战的新任务，通过识别和减轻指令中的模糊性，改进了文本生成质量，并突出了清晰和具体指令在提升LLM在NLG任务中表现的关键作用。

    

    在这项研究中，我们介绍了AmbigNLG，这是一个旨在解决自然语言生成（NLG）任务中指令模糊性挑战的新任务。尽管大语言模型（LLMs）在理解和执行各种任务方面具有令人印象深刻的能力，但它们的性能受到现实指令中的模糊性的显著限制。为了解决这个问题，AmbigNLG试图识别并减轻这种模糊性，旨在精细化指令以更好地匹配用户期望。我们介绍了一个包含2,500个实例的数据集AmbigSNI-NLG，并开发了一个模糊性分类法，用于对指令中的模糊性进行分类和注释。我们的方法在文本生成质量方面取得了显著改进，突出了清晰和具体指令在增强LLM在NLG任务中表现方面的关键作用。

    arXiv:2402.17717v1 Announce Type: new  Abstract: In this study, we introduce AmbigNLG, a new task designed to tackle the challenge of task ambiguity in instructions for Natural Language Generation (NLG) tasks. Despite the impressive capabilities of Large Language Models (LLMs) in understanding and executing a wide range of tasks through natural language interaction, their performance is significantly hindered by the ambiguity present in real-world instructions. To address this, AmbigNLG seeks to identify and mitigate such ambiguities, aiming to refine instructions to match user expectations better. We introduce a dataset, AmbigSNI-NLG, consisting of 2,500 instances, and develop an ambiguity taxonomy for categorizing and annotating instruction ambiguities. Our approach demonstrates substantial improvements in text generation quality, highlighting the critical role of clear and specific instructions in enhancing LLM performance in NLG tasks.
    
[^9]: Chain-of-Discussion：复杂证据问题回答的多模型框架

    Chain-of-Discussion: A Multi-Model Framework for Complex Evidence-Based Question Answering

    [https://arxiv.org/abs/2402.16313](https://arxiv.org/abs/2402.16313)

    提出了一种Chain-of-Discussion框架，通过多个开源语言模型的协同作用，提高了复杂问题回答的质量

    

    开放式问题回答需要模型找到适当的证据来形成合理、全面和有帮助的答案。在实际应用中，模型还需要参与对与问题密切相关的潜在场景进行深入讨论。在检索模块的增强下，开源大型语言模型（LLMs）通常能够产生一致的答案，但在可靠证据选择和深入问题分析方面仍不够理想。本文提出了一种新颖的Chain-of-Discussion框架，旨在利用多个开源LLMs之间的协同作用，为开放式QA提供更正确、更全面的答案，尽管它们在个体上还不够强大。我们的实验证明，多个LLMs之间的讨论对提高答案质量起着至关重要的作用。我们在\url{https://github.com/kobaya}上发布了我们的数据和代码。

    arXiv:2402.16313v1 Announce Type: cross  Abstract: Open-ended question answering requires models to find appropriate evidence to form well-reasoned, comprehensive and helpful answers. In practical applications, models also need to engage in extended discussions on potential scenarios closely relevant to the question. With augmentation of retrieval module, open-source Large Language Models (LLMs) can produce coherent answers often with different focuses, but are still sub-optimal in terms of reliable evidence selection and in-depth question analysis. In this paper, we propose a novel Chain-of-Discussion framework to leverage the synergy among multiple open-source LLMs aiming to provide \textbf{more correct} and \textbf{more comprehensive} answers for open-ended QA, although they are not strong enough individually. Our experiments show that discussions among multiple LLMs play a vital role in enhancing the quality of answers. We release our data and code at \url{https://github.com/kobaya
    
[^10]: PANDA（Pedantic ANswer-correctness Determination and Adjudication）：改进问答和文本生成的自动评估

    PANDA (Pedantic ANswer-correctness Determination and Adjudication):Improving Automatic Evaluation for Question Answering and Text Generation

    [https://arxiv.org/abs/2402.11161](https://arxiv.org/abs/2402.11161)

    提出了PANDA方法，引入了更精确的答案正确性评测方式，解决了当前自动评估问答和文本生成过程中的挑战。

    

    问答（QA）只有在我们知道答案是否正确时才能取得进展，但对于许多最具挑战性和有趣的QA示例，当前的答案正确性（AC）指标与人类判断不一致，特别是来自大型语言模型（LLM）的冗长、自由格式答案。我们提出了两个挑战：缺乏数据和模型过大。基于LLM的评分器与人类更好地相关，但这项昂贵的任务仅在有限的QA数据集上进行了测试。我们通过提供清晰的指南来评估从人类QA比赛中采纳的机器QA，解决了这些问题。我们还引入了精确的答案正确性确定和裁决（Precise ANswer correctness Determination and Adjudication，PANDA），这是一个小巧、高效、确定性的AC分类器（812 KB），更准确地评估答案的正确性。

    arXiv:2402.11161v1 Announce Type: cross  Abstract: Question answering (QA) can only make progress if we know if an answer is correct, but for many of the most challenging and interesting QA examples, current answer correctness (AC) metrics do not align with human judgments, particularly verbose, free form answers from large language models (LLM). There are two challenges: a lack of data and that models are too big. LLM based scorers correlate better with humans, but this expensive task has only been tested on limited QA datasets. We rectify these issues by providing clear guidelines for evaluating machine QA adopted from human QA contests. We also introduce Precise ANswer correctness Determination and Adjudication (PANDA), a small, efficient, deterministic AC classifier (812 KB) that more accurately evaluates answer correctness.
    
[^11]: 仅在需要时检索：大型语言模型中的适应性检索增强以减轻幻觉

    Retrieve Only When It Needs: Adaptive Retrieval Augmentation for Hallucination Mitigation in Large Language Models

    [https://arxiv.org/abs/2402.10612](https://arxiv.org/abs/2402.10612)

    本研究提出了一种新方法Rowen，通过选择性检索增强过程，采用多语义感知检测模块来平衡参数化知识和外部信息，以减轻大型语言模型中的幻觉问题。

    

    幻觉对于大型语言模型（LLMs）的实际实施构成了显著挑战。生成事实内容时利用参数化知识受到LLMs有限知识的限制，可能导致内部幻觉。虽然整合外部信息可以填补知识空白，但也会引入无关信息的风险，从而增加外部幻觉的可能性。在LLMs内部平衡地整合参数化知识和外部信息对缓解幻觉至关重要。本研究中，我们提出Rowen，一种增强LLMs的新方法，其中包括一种选择性检索增强过程，旨在解决幻觉输出。该过程由一个多语义感知检测模块管理，该模块评估了对相同查询在不同语言中的扰动响应的一致性。

    arXiv:2402.10612v1 Announce Type: new  Abstract: Hallucinations pose a significant challenge for the practical implementation of large language models (LLMs). The utilization of parametric knowledge in generating factual content is constrained by the limited knowledge of LLMs, potentially resulting in internal hallucinations. While incorporating external information can help fill knowledge gaps, it also introduces the risk of irrelevant information, thereby increasing the likelihood of external hallucinations. A careful and balanced integration of the parametric knowledge within LLMs with external information is crucial to alleviate hallucinations. In this study, we present Rowen, a novel approach that enhances LLMs with a selective retrieval augmentation process tailored to address hallucinated outputs. This process is governed by a multilingual semantic-aware detection module, which evaluates the consistency of the perturbed responses across various languages for the same queries. Up
    
[^12]: 使用语言反馈模型来改进政策

    Policy Improvement using Language Feedback Models

    [https://arxiv.org/abs/2402.07876](https://arxiv.org/abs/2402.07876)

    本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。

    

    我们引入了语言反馈模型（LFMs），用于在指令遵循中识别期望的行为-有助于实现指令中指定任务的行动-以进行模仿学习。为了训练LFMs，我们从大型语言模型（LLMs）获取对视觉轨迹进行语言描述的反馈。首先，通过使用LFMs识别期望模仿的行为，我们在三种不同的语言基础环境（Touchdown，ScienceWorld和ALFWorld）上，在任务完成率上改善了强行为克隆的基线方法。其次，与LLMs直接预测行动相比，使用LFMs在LLM输出标记的数量相同的情况下表现更好。第三，LFMs适应未见环境，通过一轮适应使任务完成率提高了3.5-12.0％。最后，可以修改LFM以提供人类可解释的反馈，无需性能损失，从而允许人类验证模仿学习的期望行为。

    We introduce Language Feedback Models (LFMs) that identify desirable behaviour - actions that help achieve tasks specified in the instruction - for imitation learning in instruction following. To train LFMs, we obtain feedback from Large Language Models (LLMs) on visual trajectories verbalized to language descriptions. First, by using LFMs to identify desirable behaviour to imitate, we improve in task-completion rate over strong behavioural cloning baselines on three distinct language grounding environments (Touchdown, ScienceWorld, and ALFWorld). Second, LFMs outperform using LLMs as experts to directly predict actions, when controlling for the number of LLM output tokens. Third, LFMs generalize to unseen environments, improving task-completion rate by 3.5-12.0% through one round of adaptation. Finally, LFM can be modified to provide human-interpretable feedback without performance loss, allowing human verification of desirable behaviour for imitation learning.
    
[^13]: 论语料库模拟辩论中的系统性偏差

    Systematic Biases in LLM Simulations of Debates

    [https://arxiv.org/abs/2402.04049](https://arxiv.org/abs/2402.04049)

    本研究揭示了LLMs在模拟政治辩论中存在的系统性偏差，尽管被指定从特定的政治观点进行辩论，LLMs代理机构倾向于遵循模型固有的社会偏见。通过自动自我优化方法，我们进一步证实了这些观察结果。

    

    最近自然语言处理的进展，特别是大型语言模型（LLMs）的出现，为构建能够准确复制人类行为的计算机模拟提供了令人兴奋的可能性。然而，LLMs是复杂的统计学习器，没有直接的演绎规则，使其容易出现意外行为。在本研究中，我们重点介绍了LLMs在模拟人类互动中的限制，特别关注LLMs在模拟政治辩论方面的能力。我们的发现表明，尽管被指定从特定的政治观点进行辩论，LLMs代理机构倾向于遵循模型固有的社会偏见。这种倾向导致出现行为模式，似乎偏离了人类之间已经确立的社会动态。我们使用自动自我优化方法加强了这些观察结果，该方法使我们能够操纵LLMs内部的偏见，并证明代理随后与这些调整保持一致。

    Recent advancements in natural language processing, especially the emergence of Large Language Models (LLMs), have opened exciting possibilities for constructing computational simulations designed to replicate human behavior accurately. However, LLMs are complex statistical learners without straightforward deductive rules, making them prone to unexpected behaviors. In this study, we highlight the limitations of LLMs in simulating human interactions, particularly focusing on LLMs' ability to simulate political debates. Our findings indicate a tendency for LLM agents to conform to the model's inherent social biases despite being directed to debate from certain political perspectives. This tendency results in behavioral patterns that seem to deviate from well-established social dynamics among humans. We reinforce these observations using an automatic self-fine-tuning method, which enables us to manipulate the biases within the LLM and demonstrate that agents subsequently align with the al
    
[^14]: OrchestraLLM：用于对话状态跟踪的语言模型高效编排

    OrchestraLLM: Efficient Orchestration of Language Models for Dialogue State Tracking

    [https://arxiv.org/abs/2311.09758](https://arxiv.org/abs/2311.09758)

    本研究提出了一种新颖的SLM/LLM路由框架，以提高计算效率和增强任务性能，通过利用结构化知识提取任务中SLMs和LLMs的互补优势，从而降低成本而不牺牲性能。

    

    大型语言模型（LLMs）已经彻底改变了自然语言处理系统的格局，但计算成本昂贵。为了降低成本而不损害性能，先前的研究探索了各种方法来利用小型语言模型（SLMs）作为其更大型对应物的经济有效替代品。受到SLMs和LLMs在结构化知识提取任务中显示出互补优势的发现驱动，本文提出了一种新颖的SLM/LLM路由框架，旨在提高计算效率并增强任务性能。首先，创建示范池以表示每个LM提供更可靠答案的上下文类型，利用句子嵌入进行微调，使上下文相似性接近对话状态相似性。然后，在推理过程中，检索到测试实例的k个最近示范，并根据情况路由实例。

    arXiv:2311.09758v2 Announce Type: replace  Abstract: Large language models (LLMs) have revolutionized the landscape of Natural Language Processing systems, but are computationally expensive. To reduce the cost without sacrificing performance, previous studies have explored various approaches to harness the potential of Small Language Models (SLMs) as cost-effective alternatives to their larger counterparts. Driven by findings that SLMs and LLMs exhibit complementary strengths in a structured knowledge extraction task, this work presents a novel SLM/LLM routing framework designed to improve computational efficiency and enhance task performance. First, exemplar pools are created to represent the types of contexts where each LM provides a more reliable answer, leveraging a sentence embedding fine-tuned so that context similarity is close to dialogue state similarity. Then, during inference, the k-nearest exemplars to the testing instance are retrieved, and the instance is routed according
    
[^15]: TAT-LLM: 一种针对表格和文本数据的专用语言模型用于离散推理

    TAT-LLM: A Specialized Language Model for Discrete Reasoning over Tabular and Textual Data. (arXiv:2401.13223v1 [cs.CL])

    [http://arxiv.org/abs/2401.13223](http://arxiv.org/abs/2401.13223)

    TAT-LLM是一种专门用于离散推理的语言模型，针对混合表格和文本数据上的问答任务。该模型通过分步流水线的方式，包括提取器、推理器和执行器，利用LLMs的强大能力来解决问题。而为了应对成本、延迟和数据安全风险等挑战，我们开发了TAT-LLM，一个专门针对此任务的较小LLM。

    

    在这项工作中，我们解决了在混合表格和文本数据上进行问答的问题，这在Web上非常常见（如SEC文件），通常需要离散推理能力。最近，像GPT-4这样的大型语言模型展示了强大的多步骤推理能力。我们考虑利用LLMs的强大能力来解决我们的任务。我们提出了面向表格和文本问答的分步流水线的抽象，包括提取器、推理器和执行器三个关键步骤，并首先设计了一份指令来实例化该流水线并验证GPT-4优于所有现有方法。然而，利用像GPT-4这样的在线LLM存在成本、延迟和数据安全风险等各种挑战，这促使我们专门针对此任务开发较小的LLM。我们通过对现有专家标注数据集自动生成的训练数据对LLaMA 2进行微调，开发了TAT-LLM语言模型。

    In this work, we address question answering (QA) over a hybrid of tabular and textual data that are very common content on the Web (e.g. SEC filings), where discrete reasoning capabilities are often required. Recently, large language models (LLMs) like GPT-4 have demonstrated strong multi-step reasoning capabilities. We then consider harnessing the amazing power of LLMs to solve our task. We abstract a Step-wise Pipeline for tabular and textual QA, which consists of three key steps, including Extractor, Reasoner and Executor, and initially design an instruction to instantiate the pipeline and validate that GPT-4 outperforms all existing methods. However, utilizing an online LLM like GPT-4 holds various challenges in terms of cost, latency, and data security risk, which motivates us to specialize smaller LLMs in this task. We develop a TAT-LLM language model by fine-tuning LLaMA 2 with the training data generated automatically from existing expert-annotated datasets following the Step-w
    
[^16]: 高斯自适应注意力是唯一所需的：跨多个模态的健壮上下文表示

    Gaussian Adaptive Attention is All You Need: Robust Contextual Representations Across Multiple Modalities. (arXiv:2401.11143v1 [cs.LG])

    [http://arxiv.org/abs/2401.11143](http://arxiv.org/abs/2401.11143)

    该论文提出了一个名为GAAM的多头高斯自适应注意力机制，用于增强跨多个模态的信息聚合。通过将可学习的均值和方差纳入注意力机制中，GAAM能够动态地重新调整特征的重要性，从而在处理非平稳数据时取得了显著的性能提升，超过了目前现有的注意力技术。该方法的适应性强且参数数量较少，具有改进现有注意力框架的潜力。

    

    我们提出了多头高斯自适应注意力机制（GAAM），一种新颖的概率注意力框架，并设计了高斯自适应变压器（GAT），旨在增强跨多个模态（包括语音、文本和视觉）的信息聚合。GAAM将可学习的均值和方差融入其注意力机制中，采用多头框架实现，使其能够集体建模任何概率分布，以动态重新调整特征重要性。该方法在处理高度非平稳数据时表现出显著改进，通过识别特征空间中的关键元素，超越了现有的注意力技术在模型性能上的状态（精度增加约20%）。GAAM与基于点积的注意力模型兼容，并具有相对较低的参数数量，展示了其适应性和提升现有注意力框架的潜力。在实证方面，GAAM表现出卓越的适应性和功效。

    We propose the Multi-Head Gaussian Adaptive Attention Mechanism (GAAM), a novel probabilistic attention framework, and the Gaussian Adaptive Transformer (GAT), designed to enhance information aggregation across multiple modalities, including Speech, Text and Vision. GAAM integrates learnable mean and variance into its attention mechanism, implemented in a Multi-Headed framework enabling it to collectively model any Probability Distribution for dynamic recalibration of feature significance. This method demonstrates significant improvements, especially with highly non-stationary data, surpassing the state-of-the-art attention techniques in model performance (up to approximately +20% in accuracy) by identifying key elements within the feature space. GAAM's compatibility with dot-product-based attention models and relatively low number of parameters showcases its adaptability and potential to boost existing attention frameworks. Empirically, GAAM exhibits superior adaptability and efficacy
    
[^17]: 代码提示在文本+代码LLMs中引发了条件推理能力

    Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs. (arXiv:2401.10065v1 [cs.CL])

    [http://arxiv.org/abs/2401.10065](http://arxiv.org/abs/2401.10065)

    本论文研究了在大型语言模型（LLMs）中触发条件推理能力的方法，通过使用代码提示将自然语言问题转化为代码，从而在多个数据集上实现了显著的性能提升。

    

    推理是实现语言理解的基本组成部分。在多种推理类型中，条件推理是一种在某些条件下得出不同结论的能力，在大型语言模型（LLMs）中一直没有得到充分研究。最近的提示方法，如思维链，显著改进了在推理任务上的LLMs性能。然而，我们对于什么触发了LLMs中的推理能力仍然知之甚少。我们假设代码提示能够触发在文本和代码上训练的LLMs中的条件推理。我们提出了一系列的提示，将自然语言问题转化为代码，并用生成的代码提示LLMs。我们的实验发现，在需要条件推理的多个数据集上，代码提示使得GPT 3.5的性能提升了2.6到7.7个百分点。接着，我们进行了实验，探索了代码提示如何引发条件推理能力以及通过哪些特征进行。我们观察到，提示的形式和内容对于引发条件推理能力起到了重要作用。

    Reasoning is a fundamental component for achieving language understanding. Among the multiple types of reasoning, conditional reasoning, the ability to draw different conclusions depending on some condition, has been understudied in large language models (LLMs). Recent prompting methods, such as chain of thought, have significantly improved LLMs on reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs. We hypothesize that code prompts can trigger conditional reasoning in LLMs trained on text and code. We propose a chain of prompts that transforms a natural language problem into code and prompts the LLM with the generated code. Our experiments find that code prompts exhibit a performance boost between 2.6 and 7.7 points on GPT 3.5 across multiple datasets requiring conditional reasoning. We then conduct experiments to discover how code prompts elicit conditional reasoning abilities and through which features. We observe that prom
    
[^18]: 一种基于联合推理的疾病问答系统

    A Joint-Reasoning based Disease Q&A System. (arXiv:2401.03181v1 [cs.CL])

    [http://arxiv.org/abs/2401.03181](http://arxiv.org/abs/2401.03181)

    这项研究提出了一种基于联合推理的疾病问答系统，通过结合语言模型和知识图谱的方法，旨在回答普通用户的健康相关问题并减轻医疗保健专业人员的负担。

    

    医学问答（QA）助手通过使用自然语言处理和相关技术从多个信息源合成信息来回答普通用户的健康相关问题。它们可以作为重要工具来缓解误导、信息过载和医学术语复杂性问题，从而满足普通用户的信息需求并减轻医疗保健专业人员的负担。QA系统通常使用语言模型（LM）或知识图谱（KG），尽管这两种方法可以互补。基于LM的QA系统擅长理解复杂问题并提供合适的答案，但易于出现事实错误。基于KG的QA系统能够很好地表示事实，但大多数仅限于回答已预先创建模板的简短问题。虽然一些研究已经联合使用了LM和KG方法来进行基于文本的QA，但这是用于回答多项选择题。现有的QA系统也存在一些问题。

    Medical question answer (QA) assistants respond to lay users' health-related queries by synthesizing information from multiple sources using natural language processing and related techniques. They can serve as vital tools to alleviate issues of misinformation, information overload, and complexity of medical language, thus addressing lay users' information needs while reducing the burden on healthcare professionals. QA systems, the engines of such assistants, have typically used either language models (LMs) or knowledge graphs (KG), though the approaches could be complementary. LM-based QA systems excel at understanding complex questions and providing well-formed answers, but are prone to factual mistakes. KG-based QA systems, which represent facts well, are mostly limited to answering short-answer questions with pre-created templates. While a few studies have jointly used LM and KG approaches for text-based QA, this was done to answer multiple-choice questions. Extant QA systems also 
    
[^19]: 发挥指导调整的大语言模型在端到端语音识别中的零-shot能力

    Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition. (arXiv:2309.10524v1 [eess.AS])

    [http://arxiv.org/abs/2309.10524](http://arxiv.org/abs/2309.10524)

    本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。

    

    我们提出了一种将指导调整的大语言模型和端到端自动语音识别相结合的新方法。现代大语言模型在零-shot学习中可以执行各种语言任务，只要提供明确的指导或提示来指导文本生成过程。我们探索使用这种零-shot能力的大语言模型来提取语言信息，以改善语音识别性能。具体来说，我们将大语言模型引导去纠正语音识别假设中的语法错误，并利用嵌入的语言知识进行端到端语音识别。所提出的模型基于混合连接主义时间分类和注意力架构，其中指导调整的大语言模型（即Llama2）被用作解码器的前端。通过CTC解码从编码器获得一个需要纠正的语音识别假设，然后将其与指导一起输入大语言模型。解码器随后采取...

    We present a novel integration of an instruction-tuned large language model (LLM) and end-to-end automatic speech recognition (ASR). Modern LLMs can perform a wide range of linguistic tasks within zero-shot learning when provided with a precise instruction or a prompt to guide the text generation process towards the desired task. We explore using this zero-shot capability of LLMs to extract linguistic information that can contribute to improving ASR performance. Specifically, we direct an LLM to correct grammatical errors in an ASR hypothesis and harness the embedded linguistic knowledge to conduct end-to-end ASR. The proposed model is built on the hybrid connectionist temporal classification (CTC) and attention architecture, where an instruction-tuned LLM (i.e., Llama2) is employed as a front-end of the decoder. An ASR hypothesis, subject to correction, is obtained from the encoder via CTC decoding, which is then fed into the LLM along with an instruction. The decoder subsequently tak
    
[^20]: 可解释的推理方法用于刻板印象识别

    Interpretable Stereotype Identification through Reasoning. (arXiv:2308.00071v1 [cs.CL])

    [http://arxiv.org/abs/2308.00071](http://arxiv.org/abs/2308.00071)

    本研究通过使用推理方法，在零射击刻板印象识别中取得了重要的进展，并发现推理的性能增益远远超过模型规模扩展的增益。推理不仅提高了准确性，还提高了决策的可解释性。

    

    鉴于语言模型训练使用了包含固有偏见的大量数据集，可能会不经意地持续系统性歧视，因此，审查和解决语言模型中的偏见变得至关重要，将公平性整合到它们的发展中，以确保这些模型具有公正和无偏的特性。在这项工作中，我们展示了基于Vicuna-13B-v1.3的零射击刻板印象识别中推理的重要性。尽管我们观察到从13B到33B的规模扩展会提高准确性，但我们表明推理的性能增益远远超过规模扩展的增益。我们的研究结果表明，推理可能是使LLMs在刻板印象等领域任务上超越规模定律的关键因素。此外，通过对选定的推理追踪进行定性分析，我们突出显示了推理不仅提高了准确性，还提高了决策的可解释性。

    Given that language models are trained on vast datasets that may contain inherent biases, there is a potential danger of inadvertently perpetuating systemic discrimination. Consequently, it becomes essential to examine and address biases in language models, integrating fairness into their development to ensure these models are equitable and free from bias. In this work, we demonstrate the importance of reasoning in zero-shot stereotype identification based on Vicuna-13B-v1.3. While we do observe improved accuracy by scaling from 13B to 33B, we show that the performance gain from reasoning significantly exceeds the gain from scaling up. Our findings suggest that reasoning could be a key factor that enables LLMs to trescend the scaling law on out-of-domain tasks such as stereotype identification. Additionally, through a qualitative analysis of select reasoning traces, we highlight how reasoning enhances not just accuracy but also the interpretability of the decision.
    
[^21]: 生物医学自然语言处理中的大型语言模型: 基准、基线和建议

    Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations. (arXiv:2305.16326v1 [cs.CL])

    [http://arxiv.org/abs/2305.16326](http://arxiv.org/abs/2305.16326)

    本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。

    

    生物医学文献呈指数级增长，手动筛选和提取知识变得困难。自动从生物医学文献中提取信息的生物医学自然语言处理（BioNLP）技术有助于减轻这种负担。近年来，如GPT-3和GPT-4等大型语言模型（LLMs）因其卓越的性能而受到重视。但是，它们在BioNLP任务中的有效性以及对方法开发和下游用户的影响仍未得到研究。本研究（1）在四个应用程序中在八个BioNLP数据集中建立了GPT-3和GPT-4在零-shot和一-shot设置下的基准表现，包括命名实体识别，关系提取，多标签文档分类和语义相似性和推理；（2）审查了LLMs产生的错误，并将错误分为三种类型：缺失，不一致和不需要的人工内容；（3）提出了使用LLMs的建议。

    Biomedical literature is growing rapidly, making it challenging to curate and extract knowledge manually. Biomedical natural language processing (BioNLP) techniques that can automatically extract information from biomedical literature help alleviate this burden. Recently, large Language Models (LLMs), such as GPT-3 and GPT-4, have gained significant attention for their impressive performance. However, their effectiveness in BioNLP tasks and impact on method development and downstream users remain understudied. This pilot study (1) establishes the baseline performance of GPT-3 and GPT-4 at both zero-shot and one-shot settings in eight BioNLP datasets across four applications: named entity recognition, relation extraction, multi-label document classification, and semantic similarity and reasoning, (2) examines the errors produced by the LLMs and categorized the errors into three types: missingness, inconsistencies, and unwanted artificial content, and (3) provides suggestions for using L
    
[^22]: Whisper-KDQ: 通过引导知识蒸馏和量化实现高效ASR的轻型Whisper

    Whisper-KDQ: A Lightweight Whisper via Guided Knowledge Distillation and Quantization for Efficient ASR. (arXiv:2305.10788v1 [cs.SD])

    [http://arxiv.org/abs/2305.10788](http://arxiv.org/abs/2305.10788)

    本文提出了一种通过引导知识蒸馏和量化，实现对大型预训练语音识别模型Whisper进行压缩优化的方法，可以将模型大小缩小并提高性能。

    

    随着计算硬件资源的快速发展和数据的显著增长，预训练模型在语音识别等任务中的应用显著提高了性能。然而，这些模型通常具有很高的计算开销，使其难以在资源受限的设备上有效执行。为了加速推理、减少模型大小，并保持性能，我们提出了一种新颖的引导知识蒸馏和量化方法，用于大型预训练模型Whisper。学生模型基于量化损失和蒸馏损失选择蒸馏和量化层。我们将$\text{Whisper}_\text{small}$压缩到$\text{Whisper}_\text{base}$和$\text{Whisper}_\text{tiny}$级别，使$\text{Whisper}_\text{small}$分别小5.18x/10.48x。此外，与原始$\text{Whisper}_\text{base}$和$\text{Whisper}_\text{tiny}$相比，还有相对字符错误率降低.

    Due to the rapid development of computing hardware resources and the dramatic growth of data, pre-trained models in speech recognition, such as Whisper, have significantly improved the performance of speech recognition tasks. However, these models usually have a high computational overhead, making it difficult to execute effectively on resource-constrained devices. To speed up inference and reduce model size while maintaining performance, we propose a novel guided knowledge distillation and quantization for large pre-trained model Whisper. The student model selects distillation and quantization layers based on quantization loss and distillation loss, respectively. We compressed $\text{Whisper}_\text{small}$ to $\text{Whisper}_\text{base}$ and $\text{Whisper}_\text{tiny}$ levels, making $\text{Whisper}_\text{small}$ 5.18x/10.48x smaller, respectively. Moreover, compared to the original $\text{Whisper}_\text{base}$ and $\text{Whisper}_\text{tiny}$, there is also a relative character erro
    

