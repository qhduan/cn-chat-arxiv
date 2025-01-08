# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unity by Diversity: Improved Representation Learning in Multimodal VAEs](https://arxiv.org/abs/2403.05300) | 通过软约束取代硬约束，提出了一种新的专家混合先验，改善了多模态VAEs中的表示学习。 |
| [^2] | [Predicting Single-cell Drug Sensitivity by Adaptive Weighted Feature for Adversarial Multi-source Domain Adaptation](https://arxiv.org/abs/2403.05260) | 本文提出了一种名为scAdaDrug的多源自适应加权模型，通过对抗性领域自适应从多个源领域中提取药物敏感性相关的域不变特征，并引入自适应权重生成器来自适应地调节每个样本的嵌入，以预测单细胞药物敏感性。 |
| [^3] | [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/abs/2402.14658) | OpenCodeInterpreter是一种开源代码系统，集成了执行、人类反馈和动态代码细化的功能，并在关键基准测试中表现出色，甚至与GPT-4相媲美。 |
| [^4] | [Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models](https://arxiv.org/abs/2402.11140) | 本文提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，通过迭代地探索和自我评估多个思维树，获得一系列试错推理经验，作为解决复杂问题的新形式的提示。 |
| [^5] | [Multi: Multimodal Understanding Leaderboard with Text and Images](https://arxiv.org/abs/2402.03173) | Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。 |
| [^6] | [AllSpark: A Multimodal Spatio-Temporal General Intelligence Model with Thirteen Modalities](https://arxiv.org/abs/2401.00546) | 提出了一个名为AllSpark的多模态时空智能通用人工智能模型，集成了十三种不同的模态，旨在解决多模态时空数据联合解释的挑战。 |
| [^7] | [Human-Readable Fingerprint for Large Language Models](https://arxiv.org/abs/2312.04828) | 这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。 |
| [^8] | [Predictable Artificial Intelligence.](http://arxiv.org/abs/2310.06167) | 可预测人工智能是一个新兴研究领域，旨在预测人工智能生态系统的关键指标，并强调可预测性对于提高信任、责任、控制、协调和安全的重要性。 |
| [^9] | [Towards Mitigating Architecture Overfitting in Dataset Distillation.](http://arxiv.org/abs/2309.04195) | 本文解决了数据集蒸馏中的架构过度拟合问题，提出了一系列方法来提高不同网络架构在蒸馏训练数据上的泛化性能。 |

# 详细

[^1]: 多模态VAEs中的统一多样性：改进的表示学习

    Unity by Diversity: Improved Representation Learning in Multimodal VAEs

    [https://arxiv.org/abs/2403.05300](https://arxiv.org/abs/2403.05300)

    通过软约束取代硬约束，提出了一种新的专家混合先验，改善了多模态VAEs中的表示学习。

    

    多模态数据的变分自编码器在数据分析的许多任务中表现出潜力，如表示学习、有条件生成和填补。目前的架构要么跨模态共享编码器输出、解码器输入，要么两者都要学习共享表示。这样的架构对模型施加了严格约束。在这项工作中，我们展示了通过用软约束取代这些硬约束可以获得更好的潜在表示。我们提出了一种新的专家混合先验，软性地引导每个模态的潜在表示朝着共享的后验。这种方法导致了优秀的潜在表示，并允许每个编码保留来自其未压缩原始特征更好的信息。通过对多个基准数据集和一个具有挑战性的现实世界神经科学数据集进行的广泛实验，我们展示了改进的学习潜在表示和填补。

    arXiv:2403.05300v1 Announce Type: cross  Abstract: Variational Autoencoders for multimodal data hold promise for many tasks in data analysis, such as representation learning, conditional generation, and imputation. Current architectures either share the encoder output, decoder input, or both across modalities to learn a shared representation. Such architectures impose hard constraints on the model. In this work, we show that a better latent representation can be obtained by replacing these hard constraints with a soft constraint. We propose a new mixture-of-experts prior, softly guiding each modality's latent representation towards a shared aggregate posterior. This approach results in a superior latent representation and allows each encoding to preserve information from its uncompressed original features better. In extensive experiments on multiple benchmark datasets and a challenging real-world neuroscience data set, we show improved learned latent representations and imputation of m
    
[^2]: 通过自适应加权特征进行对抗性多源领域自适应预测单细胞药物敏感性

    Predicting Single-cell Drug Sensitivity by Adaptive Weighted Feature for Adversarial Multi-source Domain Adaptation

    [https://arxiv.org/abs/2403.05260](https://arxiv.org/abs/2403.05260)

    本文提出了一种名为scAdaDrug的多源自适应加权模型，通过对抗性领域自适应从多个源领域中提取药物敏感性相关的域不变特征，并引入自适应权重生成器来自适应地调节每个样本的嵌入，以预测单细胞药物敏感性。

    

    单细胞测序技术的发展推动了大量单细胞转录谱的生成，为探索肿瘤中耐药细胞亚群提供了宝贵机会。然而，迄今为止，单细胞水平的药物敏感性数据仍然稀缺，迫切需要对个体细胞的药物敏感性进行计算预测，这是一项紧迫且具有挑战性的任务。本文提出了一种名为scAdaDrug的多源自适应加权模型，用于预测单细胞药物敏感性。我们利用对抗领域自适应从多个源领域中提取与药物敏感性相关的域不变特征。特别地，我们引入了一种自适应权重生成器，用于产生重要性感知和相互独立的权重，可以自适应地调节每个样本在源和目标领域中的维度级别嵌入。大量实验证明了我们方法的有效性。

    arXiv:2403.05260v1 Announce Type: new  Abstract: The development of single-cell sequencing technology had promoted the generation of a large amount of single-cell transcriptional profiles, providing valuable opportunities to explore drug-resistant cell subpopulations in a tumor. However, the drug sensitivity data in single-cell level is still scarce to date, pressing an urgent and highly challenging task for computational prediction of the drug sensitivity to individual cells. This paper proposed scAdaDrug, a multi-source adaptive weighting model to predict single-cell drug sensitivity. We used an autoencoder to extract domain-invariant features related to drug sensitivity from multiple source domains by exploiting adversarial domain adaptation. Especially, we introduced an adaptive weight generator to produce importance-aware and mutual independent weights, which could adaptively modulate the embedding of each sample in dimension-level for both source and target domains. Extensive exp
    
[^3]: OpenCodeInterpreter：集成代码生成、执行和细化

    OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement

    [https://arxiv.org/abs/2402.14658](https://arxiv.org/abs/2402.14658)

    OpenCodeInterpreter是一种开源代码系统，集成了执行、人类反馈和动态代码细化的功能，并在关键基准测试中表现出色，甚至与GPT-4相媲美。

    

    大型语言模型的引入显著推动了代码生成的发展。然而，开源模型通常缺乏类似GPT-4 Code Interpreter这样的高级系统的执行能力和迭代细化能力。为了解决这一问题，我们介绍了OpenCodeInterpreter，这是一族旨在生成、执行和迭代细化代码的开源代码系统。通过Code-Feedback支持，该系统集成了执行和人类反馈，用于动态代码细化。我们对OpenCodeInterpreter在诸如HumanEval、MBPP以及它们来自EvalPlus的增强版本等关键基准上进行了全面评估，证实了其出色的性能。值得注意的是，OpenCodeInterpreter-33B在HumanEval和MBPP的平均值（以及其增强版本）上取得了83.2（76.4）的准确率，与GPT-4的84.2（76.2）紧密匹敌，并且通过合成hum

    arXiv:2402.14658v1 Announce Type: cross  Abstract: The introduction of large language models has significantly advanced code generation. However, open-source models often lack the execution capabilities and iterative refinement of advanced systems like the GPT-4 Code Interpreter. To address this, we introduce OpenCodeInterpreter, a family of open-source code systems designed for generating, executing, and iteratively refining code. Supported by Code-Feedback, a dataset featuring 68K multi-turn interactions, OpenCodeInterpreter integrates execution and human feedback for dynamic code refinement. Our comprehensive evaluation of OpenCodeInterpreter across key benchmarks such as HumanEval, MBPP, and their enhanced versions from EvalPlus reveals its exceptional performance. Notably, OpenCodeInterpreter-33B achieves an accuracy of 83.2 (76.4) on the average (and plus versions) of HumanEval and MBPP, closely rivaling GPT-4's 84.2 (76.2) and further elevates to 91.6 (84.6) with synthesized hum
    
[^4]: 思维的提升：使用大型语言模型进行试错问题解决

    Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models

    [https://arxiv.org/abs/2402.11140](https://arxiv.org/abs/2402.11140)

    本文提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，通过迭代地探索和自我评估多个思维树，获得一系列试错推理经验，作为解决复杂问题的新形式的提示。

    

    大型语言模型（LLMs）在各种问题上的推理性能关键取决于思维链提示，其中包括在提示中提供一些思维链示范作为示例。最近的工作（例如Thought Tree）指出了在复杂问题解决的推理步骤选择中，探索和自我评估的重要性。在本文中，我们提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，用于通过迭代地探索和自我评估许多思维树来获得一系列试错推理经验，这将作为解决复杂问题的新形式的提示。BoT从一个简单提示开始，无需示例，迭代地探索和评估大量的推理步骤，更重要的是，利用LLM获得的错误分析来明确修改提示。

    arXiv:2402.11140v1 Announce Type: new  Abstract: The reasoning performance of Large Language Models (LLMs) on a wide range of problems critically relies on chain-of-thought prompting, which involves providing a few chain of thought demonstrations as exemplars in prompts. Recent work, e.g., Tree of Thoughts, has pointed out the importance of exploration and self-evaluation in reasoning step selection for complex problem solving. In this paper, we present Boosting of Thoughts (BoT), an automated prompting framework for problem solving with LLMs by iteratively exploring and self-evaluating many trees of thoughts in order to acquire an ensemble of trial-and-error reasoning experiences, which will serve as a new form of prompting to solve the complex problem. Starting from a simple prompt without requiring examples, BoT iteratively explores and evaluates a large collection of reasoning steps, and more importantly, uses error analysis obtained from the LLM on them to explicitly revise prompt
    
[^5]: 多模态：文本和图像的多模态理解排行榜

    Multi: Multimodal Understanding Leaderboard with Text and Images

    [https://arxiv.org/abs/2402.03173](https://arxiv.org/abs/2402.03173)

    Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。

    

    多模态大型语言模型（MLLM）的快速进展强调了向学术界引入具有挑战性而又真实的基准的需求。现有的基准主要关注简单的自然图像理解，但Multi成为了MLLM的尖端基准，提供了一个综合性的数据集，用于评估MLLM对理解复杂图表和科学问题的能力。该基准反映了当前真实的考试风格，提供多模态的输入，并要求准确或开放式的回答，类似于现实中的学校考试。它通过各种任务挑战MLLM，从公式推导到图像细节分析，以及跨模态推理。Multi包括超过18,000个问题，重点关注不同格式的基于科学的问答。我们还引入了Multi-Elite，一个包含500个问题的子集，用于测试MLLM的极端情况，以及Multi-Extend，通过超过4..。

    Rapid progress in multimodal large language models (MLLMs) highlights the need to introduce challenging yet realistic benchmarks to the academic community. Existing benchmarks primarily focus on simple natural image understanding, but Multi emerges as a cutting-edge benchmark for MLLMs, offering a comprehensive dataset for evaluating MLLMs against understanding complex figures and tables, and scientific questions. This benchmark, reflecting current realistic examination styles, provides multimodal inputs and requires responses that are either precise or open-ended, similar to real-life school tests. It challenges MLLMs with a variety of tasks, ranging from formula derivation to image detail analysis, and cross-modality reasoning. Multi includes over 18,000 questions, with a focus on science-based QA in diverse formats. We also introduce Multi-Elite, a 500-question subset for testing the extremities of MLLMs, and Multi-Extend, which enhances In-Context Learning research with more than 4
    
[^6]: AllSpark: 一个具有十三种模态的多模态时空智能模型

    AllSpark: A Multimodal Spatio-Temporal General Intelligence Model with Thirteen Modalities

    [https://arxiv.org/abs/2401.00546](https://arxiv.org/abs/2401.00546)

    提出了一个名为AllSpark的多模态时空智能通用人工智能模型，集成了十三种不同的模态，旨在解决多模态时空数据联合解释的挑战。

    

    长期以来，由于各种时空模态数据之间结构和语义的高度异质性，多模态时空数据的联合解释一直是一个极具挑战性的问题。主要挑战在于在不同模态之间的凝聚力和自治性之间取得平衡，而随着模态数量的增加，这种平衡表现出逐渐非线性的特性。我们引入了语言作为参考框架（LaRF），这是构建多模态统一模型的基本原则，旨在在不同模态之间取得凝聚力和自治性之间的平衡。我们提出了一个名为AllSpark的多模态时空智能通用人工智能模型。我们的模型将十三种不同的模态集成到一个统一框架中，包括1D（文本，代码），2D（RGB，红外线，SAR，多光谱，高光谱，表格，图表，轨迹，斜角摄影）。

    arXiv:2401.00546v2 Announce Type: replace  Abstract: For a long time, due to the high heterogeneity in structure and semantics among various spatiotemporal modal data, the joint interpretation of multimodal spatiotemporal data has been an extremely challenging problem. The primary challenge resides in striking a trade-off between the cohesion and autonomy of diverse modalities, and this trade-off exhibits a progressively nonlinear nature as the number of modalities expands. We introduce the Language as Reference Framework (LaRF), a fundamental principle for constructing a multimodal unified model, aiming to strike a trade-off between the cohesion and autonomy among different modalities. We propose a multimodal spatiotemporal general artificial intelligence model, called AllSpark. Our model integrates thirteen different modalities into a unified framework, including 1D (text, code), 2D (RGB, infrared, SAR, multispectral, hyperspectral, tables, graphs, trajectory, oblique photography), a
    
[^7]: 大型语言模型的人类可读指纹

    Human-Readable Fingerprint for Large Language Models

    [https://arxiv.org/abs/2312.04828](https://arxiv.org/abs/2312.04828)

    这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。

    

    由于大型语言模型（LLM）的资源密集型训练和配套的精心设计的许可证，保护LLM的版权变得至关重要。然而，由于可能的参数修改，确定LLM的原始基本模型是具有挑战性的。在本研究中，我们介绍了一种用于LLM的人类可读指纹，可以唯一地识别基本模型，而不暴露模型参数或干扰训练。我们首先观察到，在预训练期间模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括持续预训练、监督微调和RLHF，几乎没有扰动，这使得它成为识别基本模型的足够条件。通过继续训练LLM并添加一个额外的项来推开模型参数的方向，验证了这种必要性，结果使得模型受损。然而，这个方向容易受到简单攻击的影响，如维度...

    Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension 
    
[^8]: 可预测人工智能

    Predictable Artificial Intelligence. (arXiv:2310.06167v1 [cs.AI])

    [http://arxiv.org/abs/2310.06167](http://arxiv.org/abs/2310.06167)

    可预测人工智能是一个新兴研究领域，旨在预测人工智能生态系统的关键指标，并强调可预测性对于提高信任、责任、控制、协调和安全的重要性。

    

    我们介绍了可预测人工智能的基本思想和挑战，这是一个探索如何预测现有和未来人工智能生态系统关键指标的新兴研究领域。我们认为，实现可预测性对于促进人工智能生态系统的信任、责任、控制、协调和安全至关重要，因此应优先考虑而非性能。尽管与其他技术和非技术的人工智能研究领域有所不同，但与可预测人工智能相关的问题、假设和挑战尚未被清楚描述。本文旨在阐明这些问题，呼吁找到通向人工智能可预测性的路径，并概述这一新兴领域的潜在影响。

    We introduce the fundamental ideas and challenges of Predictable AI, a nascent research area that explores the ways in which we can anticipate key indicators of present and future AI ecosystems. We argue that achieving predictability is crucial for fostering trust, liability, control, alignment and safety of AI ecosystems, and thus should be prioritised over performance. While distinctive from other areas of technical and non-technical AI research, the questions, hypotheses and challenges relevant to Predictable AI were yet to be clearly described. This paper aims to elucidate them, calls for identifying paths towards AI predictability and outlines the potential impact of this emergent field.
    
[^9]: 在数据集蒸馏中缓解架构过度拟合的方法

    Towards Mitigating Architecture Overfitting in Dataset Distillation. (arXiv:2309.04195v1 [cs.LG])

    [http://arxiv.org/abs/2309.04195](http://arxiv.org/abs/2309.04195)

    本文解决了数据集蒸馏中的架构过度拟合问题，提出了一系列方法来提高不同网络架构在蒸馏训练数据上的泛化性能。

    

    数据集蒸馏方法在使用极少训练数据进行神经网络训练时表现出了显著的性能。然而，一个重要的挑战是架构过度拟合：由特定网络架构（即训练网络）合成的蒸馏训练数据在其他网络架构（即测试网络）训练时表现出较差的性能。本文解决了这个问题，提出了一系列架构设计和训练方案的方法，可以共同提高不同网络架构在蒸馏训练数据上的泛化性能。我们进行了大量实验证明了我们方法的有效性和普适性。特别地，在不同大小的蒸馏数据涉及的各种场景中，我们的方法在使用容量更大的网络对蒸馏数据进行训练时实现了与现有方法相当或更好的性能。

    Dataset distillation methods have demonstrated remarkable performance for neural networks trained with very limited training data. However, a significant challenge arises in the form of architecture overfitting: the distilled training data synthesized by a specific network architecture (i.e., training network) generates poor performance when trained by other network architectures (i.e., test networks). This paper addresses this issue and proposes a series of approaches in both architecture designs and training schemes which can be adopted together to boost the generalization performance across different network architectures on the distilled training data. We conduct extensive experiments to demonstrate the effectiveness and generality of our methods. Particularly, across various scenarios involving different sizes of distilled data, our approaches achieve comparable or superior performance to existing methods when training on the distilled data using networks with larger capacities.
    

