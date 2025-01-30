# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment](https://arxiv.org/abs/2404.01054) | 提出了Regularized Best-of-N (RBoN)，通过引入接近性项来减轻奖励欺骗，提高了算法在解码时与人类偏好对齐的效果。 |
| [^2] | [FlexCap: Generating Rich, Localized, and Flexible Captions in Images](https://arxiv.org/abs/2403.12026) | FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。 |
| [^3] | [Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics](https://arxiv.org/abs/2403.11821) | 评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标 |
| [^4] | [Algorithmic syntactic causal identification](https://arxiv.org/abs/2403.09580) | 通过替换传统概率论为对称单调范畴的替代基础，可以扩展因果识别技术到更多因果设置中。 |
| [^5] | [What is different between these datasets?](https://arxiv.org/abs/2403.05652) | 这里是中文总结出的一句话要点 |
| [^6] | [Homeostatic motion planning with innate physics knowledge](https://arxiv.org/abs/2402.15384) | 通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。 |
| [^7] | [API Pack: A Massive Multilingual Dataset for API Call Generation](https://arxiv.org/abs/2402.09615) | 这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成 |
| [^8] | [A Survey on Context-Aware Multi-Agent Systems: Techniques, Challenges and Future Directions](https://arxiv.org/abs/2402.01968) | 这篇论文调查了上下文感知多代理系统的技术、挑战和未来发展方向，并提供了一个综合的概述。它介绍了上下文感知系统和多代理系统的特性，以及集成这些系统的通用过程。 |
| [^9] | [Learning Concepts Definable in First-Order Logic with Counting](https://arxiv.org/abs/1909.03820) | 该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。 |
| [^10] | [How well can large language models explain business processes?.](http://arxiv.org/abs/2401.12846) | 该论文介绍了一个用于生成业务流程解释的SAX4BPM框架，该框架通过与大型语言模型（LLM）集成来综合各种输入要素，以提供更好的情境感知解释（SAX）。 |
| [^11] | [netFound: Foundation Model for Network Security.](http://arxiv.org/abs/2310.17025) | netFound是一个基于自我监督算法的基础模型，用于网络安全领域。该模型通过预训练捕捉网络流量的层次化和多模态属性，并能够在质量低、有限和嘈杂的数据情况下进行微调。 |
| [^12] | [Computing the gradients with respect to all parameters of a quantum neural network using a single circuit.](http://arxiv.org/abs/2307.08167) | 该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。 |

# 详细

[^1]: 正则化的最佳-N采样以减轻语言模型对齐中的奖励欺骗问题

    Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment

    [https://arxiv.org/abs/2404.01054](https://arxiv.org/abs/2404.01054)

    提出了Regularized Best-of-N (RBoN)，通过引入接近性项来减轻奖励欺骗，提高了算法在解码时与人类偏好对齐的效果。

    

    Best-of-N (BoN)采样与奖励模型已被证明是一种有效的策略，用于在解码时将大型语言模型(LLMs)与人类偏好对齐。然而，BoN采样容易受到奖励欺骗问题的影响。为了防止奖励欺骗，我们提出了一种名为Regularized Best-of-N (RBoN)的变体，通过在响应选择中结合接近性项来减轻奖励欺骗，类似于偏好学习技术。

    arXiv:2404.01054v1 Announce Type: cross  Abstract: Best-of-N (BoN) sampling with a reward model has been shown to be an effective strategy for aligning Large Language Models (LLMs) to human preferences at the time of decoding. BoN sampling is susceptible to a problem known as reward hacking. Because the reward model is an imperfect proxy for the true objective, over-optimizing its value can compromise its performance on the true objective. A common solution to prevent reward hacking in preference learning techniques is to optimize a reward using proximity regularization (e.g., KL regularization), which ensures that the language model remains close to the reference model. In this research, we propose Regularized Best-of-N (RBoN), a variant of BoN that aims to mitigate reward hacking by incorporating a proximity term in response selection, similar to preference learning techniques. We evaluate two variants of RBoN on the AlpacaFarm dataset and find that they outperform BoN, especially wh
    
[^2]: FlexCap：在图像中生成丰富、本地化和灵活的标题

    FlexCap: Generating Rich, Localized, and Flexible Captions in Images

    [https://arxiv.org/abs/2403.12026](https://arxiv.org/abs/2403.12026)

    FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。

    

    我们介绍了一种多功能的$\textit{灵活字幕}$视觉-语言模型（VLM），能够生成长度不同的特定区域描述。该模型FlexCap经过训练，可为输入的边界框生成长度条件的字幕，从而可以控制其输出的信息密度，描述范围从简洁的对象标签到详细的字幕。为了实现这一点，我们从带字幕的图像开始创建了大规模的图像区域描述训练数据集。这种灵活的字幕功能有几个宝贵的应用。首先，FlexCap在Visual Genome数据集上的密集字幕任务中表现出优越性能。其次，可以通过采用FlexCap生成本地化描述作为大型语言模型的输入来构建视觉问答（VQA）系统。由此产生的系统在许多VQ上实现了最新技术的零样本性能。

    arXiv:2403.12026v1 Announce Type: cross  Abstract: We introduce a versatile $\textit{flexible-captioning}$ vision-language model (VLM) capable of generating region-specific descriptions of varying lengths. The model, FlexCap, is trained to produce length-conditioned captions for input bounding boxes, and this allows control over the information density of its output, with descriptions ranging from concise object labels to detailed captions. To achieve this we create large-scale training datasets of image region descriptions of varying length, starting from captioned images. This flexible-captioning capability has several valuable applications.   First, FlexCap demonstrates superior performance in dense captioning tasks on the Visual Genome dataset. Second, a visual question answering (VQA) system can be built by employing FlexCap to generate localized descriptions as inputs to a large language model. The resulting system achieves state-of-the-art zero-shot performance on a number of VQ
    
[^3]: 评估文本到图像合成：图像质量度量的调查与分类

    Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics

    [https://arxiv.org/abs/2403.11821](https://arxiv.org/abs/2403.11821)

    评估文本到图像合成中，提出了针对图像质量的新评估指标，以确保文本和图像内容的对齐，并提出了新的分类法来归纳这些指标

    

    最近，通过利用语言和视觉结合的基础模型，推动了文本到图像合成方面的进展。这些模型在互联网或其他大规模数据库中的海量文本-图像对上进行了预训练。随着对高质量图像生成的需求转向确保文本与图像之间的内容对齐，已开发了新颖的评估度量标准，旨在模拟人类判断。因此，研究人员开始收集具有越来越复杂注释的数据集，以研究视觉语言模型的组成性及其作为文本与图像内容组成对齐质量度量的其纳入。在这项工作中，我们全面介绍了现有的文本到图像评估指标，并提出了一个新的分类法来对这些指标进行分类。我们还审查了经常采用的文本-图像基准数据集

    arXiv:2403.11821v1 Announce Type: cross  Abstract: Recent advances in text-to-image synthesis have been enabled by exploiting a combination of language and vision through foundation models. These models are pre-trained on tremendous amounts of text-image pairs sourced from the World Wide Web or other large-scale databases. As the demand for high-quality image generation shifts towards ensuring content alignment between text and image, novel evaluation metrics have been developed with the aim of mimicking human judgments. Thus, researchers have started to collect datasets with increasingly complex annotations to study the compositionality of vision-language models and their incorporation as a quality measure of compositional alignment between text and image contents. In this work, we provide a comprehensive overview of existing text-to-image evaluation metrics and propose a new taxonomy for categorizing these metrics. We also review frequently adopted text-image benchmark datasets befor
    
[^4]: 算法句法因果识别

    Algorithmic syntactic causal identification

    [https://arxiv.org/abs/2403.09580](https://arxiv.org/abs/2403.09580)

    通过替换传统概率论为对称单调范畴的替代基础，可以扩展因果识别技术到更多因果设置中。

    

    在因果贝叶斯网络（CBN）中进行因果识别是因果推断中的一项重要工具，允许从理论上可能的情况下的观测分布推导干预分布。然而，大多数现有的因果识别形式，如使用d分离和do-演算的技术都是在CBN上利用经典概率论的数学语言表达的。然而，在许多因果设置中，概率论和因此目前的因果识别技术不适用，如关系数据库、数据流程序（例如硬件描述语言）、分布式系统和大多数现代机器学习算法。我们表明，可以通过用对称单调范畴的替代公理基础来消除这种限制。在这种替代公理化中，我们展示了如何获得一个明确且清晰的

    arXiv:2403.09580v1 Announce Type: new  Abstract: Causal identification in causal Bayes nets (CBNs) is an important tool in causal inference allowing the derivation of interventional distributions from observational distributions where this is possible in principle. However, most existing formulations of causal identification using techniques such as d-separation and do-calculus are expressed within the mathematical language of classical probability theory on CBNs. However, there are many causal settings where probability theory and hence current causal identification techniques are inapplicable such as relational databases, dataflow programs such as hardware description languages, distributed systems and most modern machine learning algorithms. We show that this restriction can be lifted by replacing the use of classical probability theory with the alternative axiomatic foundation of symmetric monoidal categories. In this alternative axiomatization, we show how an unambiguous and clean
    
[^5]: 这里是翻译过的论文标题

    What is different between these datasets?

    [https://arxiv.org/abs/2403.05652](https://arxiv.org/abs/2403.05652)

    这里是中文总结出的一句话要点

    

    这里是翻译过的论文摘要

    arXiv:2403.05652v1 Announce Type: cross  Abstract: The performance of machine learning models heavily depends on the quality of input data, yet real-world applications often encounter various data-related challenges. One such challenge could arise when curating training data or deploying the model in the real world - two comparable datasets in the same domain may have different distributions. While numerous techniques exist for detecting distribution shifts, the literature lacks comprehensive approaches for explaining dataset differences in a human-understandable manner. To address this gap, we propose a suite of interpretable methods (toolbox) for comparing two datasets. We demonstrate the versatility of our approach across diverse data modalities, including tabular data, language, images, and signals in both low and high-dimensional settings. Our methods not only outperform comparable and related approaches in terms of explanation quality and correctness, but also provide actionable,
    
[^6]: 具有固有物理知识的稳态运动规划

    Homeostatic motion planning with innate physics knowledge

    [https://arxiv.org/abs/2402.15384](https://arxiv.org/abs/2402.15384)

    通过定义"任务"的方式和引入具有物理和因果关系理解的监督模块，我们提出了一种具有固有物理知识的稳态运动规划框架，可以在机器人上实现复杂计划。

    

    生物体以闭环方式与周围环境进行互动，其中感官输入决定行为的启动和终止。即使是简单的动物也能制定并执行复杂计划，但纯闭环输入控制的机器人尚未复制这一点。我们提出通过定义一组离散临时闭环控制器，称为“任务”，每个任务代表一个闭环行为，来解决这个问题。我们进一步引入了一个具有固有物理和因果关系理解的监督模块，通过该模块可以模拟随时间执行任务序列并将结果存储在环境模型中。基于这个模型，可以通过链接临时闭环控制器进行制定计划。所提出的框架已在实际机器人中实施，并在两种场景下作为概念验证进行了测试。

    arXiv:2402.15384v1 Announce Type: cross  Abstract: Living organisms interact with their surroundings in a closed-loop fashion, where sensory inputs dictate the initiation and termination of behaviours. Even simple animals are able to develop and execute complex plans, which has not yet been replicated in robotics using pure closed-loop input control. We propose a solution to this problem by defining a set of discrete and temporary closed-loop controllers, called "tasks", each representing a closed-loop behaviour. We further introduce a supervisory module which has an innate understanding of physics and causality, through which it can simulate the execution of task sequences over time and store the results in a model of the environment. On the basis of this model, plans can be made by chaining temporary closed-loop controllers. The proposed framework was implemented for a real robot and tested in two scenarios as proof of concept.
    
[^7]: API Pack：一个用于API调用生成的大规模多语言数据集

    API Pack: A Massive Multilingual Dataset for API Call Generation

    [https://arxiv.org/abs/2402.09615](https://arxiv.org/abs/2402.09615)

    这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成

    

    我们介绍了API Pack，一个包含超过一百万个指令-API调用对的多语言数据集，旨在提高大型语言模型的API调用生成能力。通过实验，我们证明了API Pack在提升模型在这一特定任务上的效果的同时，保持其在一般编码方面的整体熟练程度。仅在20,000个Python实例上对CodeLlama-13B进行微调，其生成未见过的API调用的准确率比GPT-3.5和GPT-4分别高出10%和5%。扩展到100k个例子可以提高对训练期间未见过的新API的泛化能力。此外，实现了跨语言的API调用生成，而无需大量语言特定的数据。数据集、经过微调的模型和整体代码库可在https://github.com/anonymous_url上公开获取。

    arXiv:2402.09615v1 Announce Type: cross  Abstract: We introduce API Pack, a multilingual dataset featuring over one million instruction-API call pairs aimed at advancing large language models' API call generation capabilities. Through experiments, we demonstrate API Pack's efficacy in enhancing models for this specialized task while maintaining their overall proficiency at general coding. Fine-tuning CodeLlama-13B on just 20,000 Python instances yields over 10% and 5% higher accuracy than GPT-3.5 and GPT-4 respectively in generating unseen API calls. Scaling to 100k examples improves generalization to new APIs not seen during training. In addition, cross-lingual API call generation is achieved without needing extensive data per language. The dataset, fine-tuned models, and overall code base are publicly available at https://github.com/anonymous_url.
    
[^8]: 对上下文感知多agent系统的调查：技术、挑战和未来发展方向

    A Survey on Context-Aware Multi-Agent Systems: Techniques, Challenges and Future Directions

    [https://arxiv.org/abs/2402.01968](https://arxiv.org/abs/2402.01968)

    这篇论文调查了上下文感知多代理系统的技术、挑战和未来发展方向，并提供了一个综合的概述。它介绍了上下文感知系统和多代理系统的特性，以及集成这些系统的通用过程。

    

    随着新兴主题的兴起，自主代理的研究兴趣正在增加。大型语言模型的显著成就已经展示了在自主代理中达到人类智能的巨大潜力。然而，挑战在于使这些代理能够在动态环境中学习、推理和导航不确定性。当处理动态情况时，上下文意识成为强化多agent系统的关键因素。尽管现有的研究专注于上下文感知系统和多agent系统，但缺乏全面概述如何将上下文感知系统与多agent系统集成的综合调查。为了填补这个空白，本调查提供了对最先进的上下文感知多agent系统的全面概述。首先，我们概述了促进这些系统之间集成的上下文感知系统和多 agent 系统的特性。随后，我们提出了一个通用的过程来建模上下文感知和多agent系统的集成。

    Research interest in autonomous agents is on the rise as an emerging topic. The notable achievements of Large Language Models (LLMs) have demonstrated the considerable potential to attain human-like intelligence in autonomous agents. However, the challenge lies in enabling these agents to learn, reason, and navigate uncertainties in dynamic environments. Context awareness emerges as a pivotal element in fortifying multi-agent systems when dealing with dynamic situations. Despite existing research focusing on both context-aware systems and multi-agent systems, there is a lack of comprehensive surveys outlining techniques for integrating context-aware systems with multi-agent systems. To address this gap, this survey provides a comprehensive overview of state-of-the-art context-aware multi-agent systems. First, we outline the properties of both context-aware systems and multi-agent systems that facilitate integration between these systems. Subsequently, we propose a general process for c
    
[^9]: 用计数符号的一阶逻辑定义的概念的学习

    Learning Concepts Definable in First-Order Logic with Counting

    [https://arxiv.org/abs/1909.03820](https://arxiv.org/abs/1909.03820)

    该研究将一阶逻辑与计数符号相结合，证明了可以在多对数度结构下以次线性时间一致学习可定义的分类器，为包含数值方面的机器学习扩展学习框架迈出了第一步。

    

    我们研究了在Grohe和Tur\'an引入的逻辑框架下的关系背景结构上的布尔分类问题。众所周知(Grohe和Ritzert, LICS 2017)，在多对数度结构上的一阶逻辑可定义的分类器可以在次线性时间内学习，其中结构的度和运行时间是以结构的大小为单位来衡量的。我们将结果推广到了由Kuske和Schweikardt(LICS 2017)引入的带计数的一阶逻辑FOCN，它作为一个广泛推广各种计数逻辑的表现逻辑。具体来说，我们证明了可以在多对数度结构类上定义的FOCN中的分类器可以在次线性时间内一致地学习。这可以看作是将学习框架扩展以包含机器学习的数值方面的第一步。我们将这一结果扩展到了无视的概率

    arXiv:1909.03820v2 Announce Type: replace-cross  Abstract: We study Boolean classification problems over relational background structures in the logical framework introduced by Grohe and Tur\'an (TOCS 2004). It is known (Grohe and Ritzert, LICS 2017) that classifiers definable in first-order logic over structures of polylogarithmic degree can be learned in sublinear time, where the degree of the structure and the running time are measured in terms of the size of the structure. We generalise the results to the first-order logic with counting FOCN, which was introduced by Kuske and Schweikardt (LICS 2017) as an expressive logic generalising various other counting logics. Specifically, we prove that classifiers definable in FOCN over classes of structures of polylogarithmic degree can be consistently learned in sublinear time. This can be seen as a first step towards extending the learning framework to include numerical aspects of machine learning. We extend the result to agnostic probabl
    
[^10]: 大型语言模型能够如何解释业务流程？

    How well can large language models explain business processes?. (arXiv:2401.12846v1 [cs.AI])

    [http://arxiv.org/abs/2401.12846](http://arxiv.org/abs/2401.12846)

    该论文介绍了一个用于生成业务流程解释的SAX4BPM框架，该框架通过与大型语言模型（LLM）集成来综合各种输入要素，以提供更好的情境感知解释（SAX）。

    

    大型语言模型（LLMs）可能在未来的AI辅助业务流程管理系统（ABPMSs）中发挥重要作用，其功能涵盖系统生命周期的各个阶段。其中一个系统功能是情境感知解释（SAX），它涉及生成在考虑所解释条件出现的流程上下文的前提下既符合因果关系又可人类解读的解释。在本文中，我们介绍了开发用于生成SAX解释的SAX4BPM框架。SAX4BPM套件包括一组服务和一个中央知识库。这些服务的功能是获取构成SAX解释的各种知识要素。其中一个创新性的关键组成部分是因果过程执行视图。在本工作中，我们将该框架与LLM集成，以利用其综合各种输入要素的能力，从而改进SAX解释的质量。

    Large Language Models (LLMs) are likely to play a prominent role in future AI-augmented business process management systems (ABPMSs) catering functionalities across all system lifecycle stages. One such system's functionality is Situation-Aware eXplainability (SAX), which relates to generating causally sound and yet human-interpretable explanations that take into account the process context in which the explained condition occurred. In this paper, we present the SAX4BPM framework developed to generate SAX explanations. The SAX4BPM suite consists of a set of services and a central knowledge repository. The functionality of these services is to elicit the various knowledge ingredients that underlie SAX explanations. A key innovative component among these ingredients is the causal process execution view. In this work, we integrate the framework with an LLM to leverage its power to synthesize the various input ingredients for the sake of improved SAX explanations. Since the use of LLMs for
    
[^11]: netFound: 网络安全的基础模型

    netFound: Foundation Model for Network Security. (arXiv:2310.17025v1 [cs.NI])

    [http://arxiv.org/abs/2310.17025](http://arxiv.org/abs/2310.17025)

    netFound是一个基于自我监督算法的基础模型，用于网络安全领域。该模型通过预训练捕捉网络流量的层次化和多模态属性，并能够在质量低、有限和嘈杂的数据情况下进行微调。

    

    在网络安全的机器学习领域，传统工作流依赖于高质量标记数据和手动特征工程，但有限的数据集和人类专业知识阻碍了特征选择，导致模型难以捕捉关键关系和有效泛化。受到GPT-4和Vision Transformers等机器学习应用领域的最新进展的启发，我们开发了netFound，一个网络安全的基础模型。该模型利用自我监督算法对现有的未标记网络数据包进行预训练。netFound的设计融合了网络流量的层次化和多模态属性，有效捕捉了隐藏的网络上下文，包括应用逻辑、通信协议和网络条件。有了这个预训练基础，即使处理质量低、有限和嘈杂的标记数据，我们也可以对netFound进行微调，适用于各种下游任务。我们的实验证明了netFound的效果。

    In ML for network security, traditional workflows rely on high-quality labeled data and manual feature engineering, but limited datasets and human expertise hinder feature selection, leading to models struggling to capture crucial relationships and generalize effectively. Inspired by recent advancements in ML application domains like GPT-4 and Vision Transformers, we have developed netFound, a foundational model for network security. This model undergoes pre-training using self-supervised algorithms applied to readily available unlabeled network packet traces. netFound's design incorporates hierarchical and multi-modal attributes of network traffic, effectively capturing hidden networking contexts, including application logic, communication protocols, and network conditions.  With this pre-trained foundation in place, we can fine-tune netFound for a wide array of downstream tasks, even when dealing with low-quality, limited, and noisy labeled data. Our experiments demonstrate netFound'
    
[^12]: 使用单个电路计算量子神经网络所有参数的梯度

    Computing the gradients with respect to all parameters of a quantum neural network using a single circuit. (arXiv:2307.08167v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2307.08167](http://arxiv.org/abs/2307.08167)

    该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。

    

    在使用参数平移规则计算量子神经网络的梯度时，需要对网络的单个可调参数计算两次代价函数。当参数总数较高时，需要调整和运行多次用于计算的量子电路。在这里，我们提出了一种仅使用一个电路计算所有梯度的方法，它具有较低的电路深度和较少的经典寄存器。我们还在真实量子硬件和模拟器上进行了实验证明，我们的方法具有电路编译时间明显缩短的优势，从而加速了总体运行时间。

    When computing the gradients of a quantum neural network using the parameter-shift rule, the cost function needs to be calculated twice for the gradient with respect to a single adjustable parameter of the network. When the total number of parameters is high, the quantum circuit for the computation has to be adjusted and run for many times. Here we propose an approach to compute all the gradients using a single circuit only, with a much reduced circuit depth and less classical registers. We also demonstrate experimentally, on both real quantum hardware and simulator, that our approach has the advantages that the circuit takes a significantly shorter time to compile than the conventional approach, resulting in a speedup on the total runtime.
    

