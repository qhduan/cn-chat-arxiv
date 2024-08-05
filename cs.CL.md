# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do language models plan ahead for future tokens?](https://arxiv.org/abs/2404.00859) | 语言模型在推理过程中会提前准备未来标记所需的信息，可能是通过预缓存或面包屑的方式实现。 |
| [^2] | [A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing](https://arxiv.org/abs/2403.02504) | 预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。 |
| [^3] | [A Survey on Data Selection for Language Models](https://arxiv.org/abs/2402.16827) | 大型语言模型成功的关键在于使用大规模的文本数据集进行无监督预训练，但如何优化选择数据以降低碳足迹和财务成本仍是一个挑战。 |
| [^4] | [Improving Sentence Embeddings with an Automatically Generated NLI Dataset](https://arxiv.org/abs/2402.15132) | 通过自动生成的NLI数据集改进句子嵌入，实验结果表明该方法在STS任务中表现出色，优于现有方法。 |
| [^5] | [SymBa: Symbolic Backward Chaining for Multi-step Natural Language Reasoning](https://arxiv.org/abs/2402.12806) | SymBa提出了一种符号化向后推理方法，在多步自然语言推理中取得了显著的性能和效率提升，能够生成可解释的结构化证明。 |
| [^6] | [A Reparameterized Discrete Diffusion Model for Text Generation](https://arxiv.org/abs/2302.05737) | 本文提出了一种重新参数化离散扩散模型，该模型在文本生成方面表现出更好的灵活性、训练技术和生成效果，实验证明其较现有的扩散模型有显著的改进。 |
| [^7] | [Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration.](http://arxiv.org/abs/2401.13979) | 本研究提出了Leeroo编排器的架构，通过集成多个训练过的LLMs模型，实现了一个新的最先进模型。该编排器在性能上与Mixtral模型相当，并且成本只有其三分之二。当允许更高的成本时，Leeroo编排器的准确性超过了Mixtral模型，并且当集成GPT4时进一步提升。 |
| [^8] | [MERA: A Comprehensive LLM Evaluation in Russian.](http://arxiv.org/abs/2401.04531) | 这项研究提出了MERA，一个多模态俄语基础模型评估指标。该指标包括21个评估任务，涵盖了11个技能领域中生成模型的评估。研究还提出了一种在零样本和少样本固定指令设置下评估FM和LM的方法。 |
| [^9] | [RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models.](http://arxiv.org/abs/2310.16340) | RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。 |
| [^10] | [A Family of Pretrained Transformer Language Models for Russian.](http://arxiv.org/abs/2309.10931) | 本文介绍了一组专门针对俄语的预训练Transformer语言模型，包括编码器、解码器和编码器-解码器模型。这些模型在俄语自然语言理解和生成方面展现了良好的泛化能力，希望能够推动俄语领域的NLP研究和工业应用的发展。 |
| [^11] | [Arithmetic with Language Models: from Memorization to Computation.](http://arxiv.org/abs/2308.01154) | 本研究探索了使用语言模型进行算术计算的能力，发现语言模型可以通过内部的值空间进行计算，并取得了成功的实验结果。 |
| [^12] | [Answering Questions by Meta-Reasoning over Multiple Chains of Thought.](http://arxiv.org/abs/2304.13007) | 本论文提出了基于元推理的Multi-Chain Reasoning (MCR)方法，该方法检查多个推理链，混合它们之间的信息并选择最相关的事实，从而超越多链思维，解决多跳QA问题。 实验结果表明MCR胜过多个强基线，解释质量高。 |

# 详细

[^1]: 语言模型是否提前为未来标记进行规划？

    Do language models plan ahead for future tokens?

    [https://arxiv.org/abs/2404.00859](https://arxiv.org/abs/2404.00859)

    语言模型在推理过程中会提前准备未来标记所需的信息，可能是通过预缓存或面包屑的方式实现。

    

    arXiv:2404.00859v1 公告类型：跨领域 摘要：在给定位置的推理过程中，变压器是否会“提前思考”？已知变压器在$t$的前向传递的隐藏状态中准备信息，然后在未来的前向传递$t+\tau$中使用。我们提出了两种解释这种现象的可能性：预缓存，即训练中存在的非对角梯度项导致模型在$t$计算与当前推理任务无关但对未来有用的特征，以及面包屑，即与时间步长$t$最相关的特征已经与那些将最有利于时间步长$t+\tau$的特征相同。我们通过训练不将梯度传播到过去时间步的语言模型来测试这些假设，这种方案我们正式称为短视训练。在合成数据设置中，我们发现了预缓存的明确证据。在自回归语言建模设置中，我们的实验更多地支持了面包屑假设。

    arXiv:2404.00859v1 Announce Type: cross  Abstract: Do transformers "think ahead" during inference at a given position? It is known transformers prepare information in the hidden states of the forward pass at $t$ that is then used in future forward passes $t+\tau$. We posit two explanations for this phenomenon: pre-caching, in which off-diagonal gradient terms present in training result in the model computing features at $t$ irrelevant to the present inference task but useful for the future, and breadcrumbs, in which features most relevant to time step $t$ are already the same as those that would most benefit inference at time $t+\tau$. We test these hypotheses by training language models without propagating gradients to past timesteps, a scheme we formalize as myopic training. In a synthetic data setting, we find clear evidence for pre-caching. In the autoregressive language modeling setting, our experiments are more suggestive of the breadcrumbs hypothesis.
    
[^2]: 自然语言处理中的预训练-微调范式教程

    A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing

    [https://arxiv.org/abs/2403.02504](https://arxiv.org/abs/2403.02504)

    预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。

    

    预训练-微调范式代表了自然语言处理中的一种变革性方法。该范式通过使用大型预训练语言模型区别于众，展示了在微调任务中即使训练数据有限也具有显著的效率。这种效率对社会科学研究特别有益，因为注释样本的数量通常非常有限。我们的教程全面介绍了预训练-微调范式。我们首先深入探讨了预训练和微调的基本概念，然后进行了实际应用的案例练习。我们展示了该范式在各种任务中的应用，包括多类别分类和回归。强调其高效性和用户友好性，该教程旨在鼓励更广泛地采纳这种范式。为此，我们提供了所有代码和数据集的开放访问。

    arXiv:2403.02504v1 Announce Type: cross  Abstract: The pretrain-finetune paradigm represents a transformative approach in natural language processing (NLP). This paradigm distinguishes itself through the use of large pretrained language models, demonstrating remarkable efficiency in finetuning tasks, even with limited training data. This efficiency is especially beneficial for research in social sciences, where the number of annotated samples is often quite limited. Our tutorial offers a comprehensive introduction to the pretrain-finetune paradigm. We first delve into the fundamental concepts of pretraining and finetuning, followed by practical exercises using real-world applications. We demonstrate the application of the paradigm across various tasks, including multi-class classification and regression. Emphasizing its efficacy and user-friendliness, the tutorial aims to encourage broader adoption of this paradigm. To this end, we have provided open access to all our code and datasets
    
[^3]: 语言模型数据选择概述

    A Survey on Data Selection for Language Models

    [https://arxiv.org/abs/2402.16827](https://arxiv.org/abs/2402.16827)

    大型语言模型成功的关键在于使用大规模的文本数据集进行无监督预训练，但如何优化选择数据以降低碳足迹和财务成本仍是一个挑战。

    

    最近大型语言模型取得成功的一个主要因素是利用巨大且不断增长的文本数据集进行无监督预训练。然而，简单地在所有可用数据上训练模型可能并不是最佳选择（或不可行），因为可用文本数据的质量可能有所不同。数据过滤也可以通过减少所需的训练量来降低训练模型的碳足迹和财务成本。数据选择方法旨在确定要包括在训练数据集中的哪些候选数据点，以及如何从所选数据点中适当采样。改进的数据选择方法的前景已经导致该领域的研究量迅速扩大。然而，由于深度学习主要受实证证据驱动，对大规模数据进行实验成本昂贵，很少有组织拥有资源进行广泛的数据选择研究。因此，有效数据选择的知识可能大多局限于大型技术公司或研究机构内部。

    arXiv:2402.16827v1 Announce Type: new  Abstract: A major factor in the recent success of large language models is the use of enormous and ever-growing text datasets for unsupervised pre-training. However, naively training a model on all available data may not be optimal (or feasible), as the quality of available text data can vary. Filtering out data can also decrease the carbon footprint and financial costs of training models by reducing the amount of training required.   Data selection methods aim to determine which candidate data points to include in the training dataset and how to appropriately sample from the selected data points. The promise of improved data selection methods has caused the volume of research in the area to rapidly expand. However, because deep learning is mostly driven by empirical evidence and experimentation on large-scale data is expensive, few organizations have the resources for extensive data selection research. Consequently, knowledge of effective data se
    
[^4]: 通过自动生成的NLI数据集改进句子嵌入

    Improving Sentence Embeddings with an Automatically Generated NLI Dataset

    [https://arxiv.org/abs/2402.15132](https://arxiv.org/abs/2402.15132)

    通过自动生成的NLI数据集改进句子嵌入，实验结果表明该方法在STS任务中表现出色，优于现有方法。

    

    基于解码器的大型语言模型在自然语言处理的许多任务中表现出了很高的性能。这在句子嵌入学习中同样成立，其中基于解码器的模型PromptEOL 在语义文本相似性（STS）任务中取得了最佳表现。然而，PromptEOL 在很大程度上利用了对自然语言推理（NLI）数据集的手动标注进行微调。我们旨在通过使用LLM自动生成的NLI数据集来改进在无监督设置下学习的句子嵌入，并将其用于微调PromptEOL。在STS任务的实验中，提出的方法在人类评估方面达到了82.21的平均Spearman等级相关系数，从而优于现有方法而无需使用大规模手动注释的数据集。

    arXiv:2402.15132v1 Announce Type: new  Abstract: Decoder-based large language models (LLMs) have shown high performance on many tasks in natural language processing. This is also true for sentence embedding learning, where a decoder-based model, PromptEOL, has achieved the best performance on semantic textual similarity (STS) tasks. However, PromptEOL makes great use of fine-tuning with a manually annotated natural language inference (NLI) dataset. We aim to improve sentence embeddings learned in an unsupervised setting by automatically generating an NLI dataset with an LLM and using it to fine-tune PromptEOL. In experiments on STS tasks, the proposed method achieved an average Spearman's rank correlation coefficient of 82.21 with respect to human evaluation, thus outperforming existing methods without using large, manually annotated datasets.
    
[^5]: SymBa：符号化向后推理用于多步自然语言推理

    SymBa: Symbolic Backward Chaining for Multi-step Natural Language Reasoning

    [https://arxiv.org/abs/2402.12806](https://arxiv.org/abs/2402.12806)

    SymBa提出了一种符号化向后推理方法，在多步自然语言推理中取得了显著的性能和效率提升，能够生成可解释的结构化证明。

    

    最近大型语言模型（LLMs）展示了在一系列思维提示中出色的推理能力，但忠实的多步推理依然是一个挑战。我们专注于向后推理，即通过逻辑规则递归地分解查询，直到证明为止。为了解决当前向后推理实现的局限性，我们提出了SymBa（符号化向后推理）。在SymBa中，符号化自顶向下求解器控制整个证明过程，当求解器遇到死胡同时，才调用LLM生成单个推理步骤。通过这种新颖的求解器-LLM集成，SymBa在各种多步推理基准（ProofWriter，Birds-Electricity，GSM8k，CLUTRR-TF，ECtHR Article 6）中相比向后推理基线取得了性能、证明忠实性和效率显著提高，能够生成可解释的结构化证明。

    arXiv:2402.12806v1 Announce Type: new  Abstract: Large Language Models (LLMs) have recently demonstrated remarkable reasoning ability as in Chain-of-thought prompting, but faithful multi-step reasoning remains a challenge. We specifically focus on backward chaining, where the query is recursively decomposed using logical rules until proven. To address the limitations of current backward chaining implementations, we propose SymBa (Symbolic Backward Chaining). In SymBa, the symbolic top-down solver controls the entire proof process and the LLM is called to generate a single reasoning step only when the solver encounters a dead end. By this novel solver-LLM integration, while being able to produce an interpretable, structured proof, SymBa achieves significant improvement in performance, proof faithfulness, and efficiency in diverse multi-step reasoning benchmarks (ProofWriter, Birds-Electricity, GSM8k, CLUTRR-TF, ECtHR Article 6) compared to backward chaining baselines.
    
[^6]: 一种用于文本生成的重新参数化离散扩散模型的研究

    A Reparameterized Discrete Diffusion Model for Text Generation

    [https://arxiv.org/abs/2302.05737](https://arxiv.org/abs/2302.05737)

    本文提出了一种重新参数化离散扩散模型，该模型在文本生成方面表现出更好的灵活性、训练技术和生成效果，实验证明其较现有的扩散模型有显著的改进。

    

    本文研究了应用于自然语言生成的离散扩散概率模型。我们推导出了从离散扩散过程中采样的另一种等价形式，并利用这一洞见开发了一族重新参数化离散扩散模型。这个派生的通用框架非常灵活，为离散扩散模型中的生成过程提供了新的视角，并具备更有效的训练和解码技术。我们进行了大量实验证明我们模型的文本生成能力，在现有的扩散模型上取得了显著的改进。

    This work studies discrete diffusion probabilistic models with applications to natural language generation. We derive an alternative yet equivalent formulation of the sampling from discrete diffusion processes and leverage this insight to develop a family of reparameterized discrete diffusion models. The derived generic framework is highly flexible, offers a fresh perspective of the generation process in discrete diffusion models, and features more effective training and decoding techniques. We conduct extensive experiments to evaluate the text generation capability of our model, demonstrating significant improvements over existing diffusion models.
    
[^7]: Leeroo Orchestrator: 通过模型集成提高LLMs的性能

    Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration. (arXiv:2401.13979v1 [cs.CL])

    [http://arxiv.org/abs/2401.13979](http://arxiv.org/abs/2401.13979)

    本研究提出了Leeroo编排器的架构，通过集成多个训练过的LLMs模型，实现了一个新的最先进模型。该编排器在性能上与Mixtral模型相当，并且成本只有其三分之二。当允许更高的成本时，Leeroo编排器的准确性超过了Mixtral模型，并且当集成GPT4时进一步提升。

    

    本文提出了一种架构，利用多个训练过的LLMs的集体知识，创建一个新的最先进模型。该框架的核心是一个基于LLM的编排器，能够选择最佳的底层LLM专家进行任务执行。受到强化学习中的自我对弈的启发，我们创建了一个查询生成、编排和评估的循环，为编排器生成训练数据。我们的评估主要针对MMLU基准，在Hugging Face上使用了具有7B、13B和34B参数的模型。结果显示我们的Leeroo编排器实现了与Mixtral模型相当的性能，但只产生了其成本的三分之二。此外，增加允许的成本超过了Mixtral的准确性，达到了75.9%的准确性。当将GPT4集成到底层模型池中时，进一步提升也得到了观察。

    In this paper, we propose an architecture to harness the collective knowledge of multiple trained LLMs to create a new state-of-the-art. At the core of this framework is a LLM-based orchestrator that is adept at picking the right underlying LLM experts for optimal task execution. Inspired by self-play in reinforcement learning, we created a loop of query generation, orchestration, and evaluation to generate training data for the orchestrator. Our evaluation focused on the MMLU benchmark, employing models with 7B, 13B, and 34B parameters available on Hugging Face. The results demonstrate new state-of-the-art open-source models: Our Leeroo orchestrator achieves performance on par with the Mixtral model while incurring only two-thirds of its cost. Moreover, increasing the allowed cost surpasses Mixtral's accuracy by over 5% at the same cost level, reaching an accuracy of 75.9%. Further enhancements were observed when integrating GPT4 into the underlying model pool. The Leeroo orchestrator
    
[^8]: MERA: 俄语LLM综合评估的研究

    MERA: A Comprehensive LLM Evaluation in Russian. (arXiv:2401.04531v1 [cs.CL])

    [http://arxiv.org/abs/2401.04531](http://arxiv.org/abs/2401.04531)

    这项研究提出了MERA，一个多模态俄语基础模型评估指标。该指标包括21个评估任务，涵盖了11个技能领域中生成模型的评估。研究还提出了一种在零样本和少样本固定指令设置下评估FM和LM的方法。

    

    在过去几年中，人工智能研究中最显著的进展之一是基础模型（FM）的发展，其中语言模型（LM）的崛起引人注目。随着模型的规模增大，LM在可衡量的方面展示了提升，并且发展出了新的定性特征。然而，尽管研究人员的关注和LM应用的快速增长，LM的能力、限制和相关风险仍需更好地理解。为了解决这些问题，我们介绍了一种开放的俄语多模态架构评估（MERA）指导基准，用于评估以俄语为导向的基础模型。该基准涵盖了11个技能领域中生成模型的21个评估任务，并被设计为黑盒测试，以确保排除数据泄漏。论文介绍了一种在零样本和少样本固定指令设置下评估FM和LM的方法，并可扩展到其他模态。

    Over the past few years, one of the most notable advancements in AI research has been in foundation models (FMs), headlined by the rise of language models (LMs). As the models' size increases, LMs demonstrate enhancements in measurable aspects and the development of new qualitative features. However, despite researchers' attention and the rapid growth in LM application, the capabilities, limitations, and associated risks still need to be better understood. To address these issues, we introduce an open Multimodal Evaluation of Russian-language Architectures (MERA), a new instruction benchmark for evaluating foundation models oriented towards the Russian language. The benchmark encompasses 21 evaluation tasks for generative models in 11 skill domains and is designed as a black-box test to ensure the exclusion of data leakage. The paper introduces a methodology to evaluate FMs and LMs in zeroand few-shot fixed instruction settings that can be extended to other modalities. We propose an 
    
[^9]: RCAgent：基于自主代理和增强的大型语言模型的云根本原因分析

    RCAgent: Cloud Root Cause Analysis by Autonomous Agents with Tool-Augmented Large Language Models. (arXiv:2310.16340v1 [cs.SE])

    [http://arxiv.org/abs/2310.16340](http://arxiv.org/abs/2310.16340)

    RCAgent是一个工具增强的LLM自主代理框架，用于云根本原因分析，能够实现自由格式的数据收集和全面的分析，并在各个方面优于当前方法。

    

    最近，云根本原因分析中的大型语言模型（LLM）应用受到了积极的关注。然而，当前方法仍然依赖于手动工作流设置，并没有充分发挥LLMs的决策和环境交互能力。我们提出了RCAgent，这是一个实用和注重隐私的工具增强LLM自主代理框架，用于实际的工业RCA使用。RCAgent在内部部署的模型上运行，而不是GPT系列，能够进行自由格式的数据收集和全面的分析，并结合各种增强功能，包括独特的行动轨迹自一致性和一套用于上下文管理、稳定化和导入领域知识的方法。我们的实验证明RCAgent在RCA的各个方面（预测根本原因、解决方案、证据和责任）以及当前规则未涵盖的任务上都明显优于ReAct，得到了自动化和人工验证的确认。

    Large language model (LLM) applications in cloud root cause analysis (RCA) have been actively explored recently. However, current methods are still reliant on manual workflow settings and do not unleash LLMs' decision-making and environment interaction capabilities. We present RCAgent, a tool-augmented LLM autonomous agent framework for practical and privacy-aware industrial RCA usage. Running on an internally deployed model rather than GPT families, RCAgent is capable of free-form data collection and comprehensive analysis with tools. Our framework combines a variety of enhancements, including a unique Self-Consistency for action trajectories, and a suite of methods for context management, stabilization, and importing domain knowledge. Our experiments show RCAgent's evident and consistent superiority over ReAct across all aspects of RCA -- predicting root causes, solutions, evidence, and responsibilities -- and tasks covered or uncovered by current rules, as validated by both automate
    
[^10]: 一种针对俄语的预训练Transformer语言模型家族

    A Family of Pretrained Transformer Language Models for Russian. (arXiv:2309.10931v1 [cs.CL])

    [http://arxiv.org/abs/2309.10931](http://arxiv.org/abs/2309.10931)

    本文介绍了一组专门针对俄语的预训练Transformer语言模型，包括编码器、解码器和编码器-解码器模型。这些模型在俄语自然语言理解和生成方面展现了良好的泛化能力，希望能够推动俄语领域的NLP研究和工业应用的发展。

    

    如今，Transformer语言模型（LMs）是自然语言处理（NLP）研究方法和应用的基本组成部分。然而，专门针对俄语的这种模型的发展却受到了较少的关注。本文介绍了一组基于编码器（ruBERT, ruRoBERTa, ruELECTRA）、解码器（ruGPT-3）和编码器-解码器（ruT5, FRED-T5）模型的13个俄语Transformer LMs，具有多种尺寸。这些模型可通过HuggingFace平台轻松获取。我们提供了模型架构设计和预训练的报告，并评估了它们在俄语自然语言理解和生成数据集以及基准测试中的泛化能力。通过预训练和发布这些专门的Transformer LMs，我们希望拓宽NLP研究方向的范围，并促进针对俄语的工业解决方案的开发。

    Nowadays, Transformer language models (LMs) represent a fundamental component of the NLP research methodologies and applications. However, the development of such models specifically for the Russian language has received little attention. This paper presents a collection of 13 Russian Transformer LMs based on the encoder (ruBERT, ruRoBERTa, ruELECTRA), decoder (ruGPT-3), and encoder-decoder (ruT5, FRED-T5) models in multiple sizes. Access to these models is readily available via the HuggingFace platform. We provide a report of the model architecture design and pretraining, and the results of evaluating their generalization abilities on Russian natural language understanding and generation datasets and benchmarks. By pretraining and releasing these specialized Transformer LMs, we hope to broaden the scope of the NLP research directions and enable the development of industrial solutions for the Russian language.
    
[^11]: 使用语言模型进行算术运算：从记忆到计算

    Arithmetic with Language Models: from Memorization to Computation. (arXiv:2308.01154v1 [cs.AI])

    [http://arxiv.org/abs/2308.01154](http://arxiv.org/abs/2308.01154)

    本研究探索了使用语言模型进行算术计算的能力，发现语言模型可以通过内部的值空间进行计算，并取得了成功的实验结果。

    

    更好地理解最近的大型语言模型的出现性计算和问题解决能力对于进一步改进它们并拓宽其适用性至关重要。本研究探讨了一个训练用于预测下一个标记的语言模型如何在训练数据之外执行算术计算。二进制加法和乘法是一个很好的测试基础，因为它们需要一个非常小的词汇表，并且在输入/输出上展示了相关的不连续性，使得对新数据进行平滑的输入插值无效。我们成功地训练了一个轻量级的语言模型来学习这些任务，并进行了一系列实验证明其外推能力和内部信息处理。我们的研究结果支持这样一个假设，即语言模型作为一个编码-回归-解码机器，一旦将输入标记表示映射到合适的内部值空间，计算就在值空间中进行。

    A better understanding of the emergent computation and problem-solving capabilities of recent large language models is of paramount importance to further improve them and broaden their applicability. This work investigates how a language model, trained to predict the next token, can perform arithmetic computations generalizing beyond training data. Binary addition and multiplication constitute a good testbed for this purpose, since they require a very small vocabulary and exhibit relevant input/output discontinuities making smooth input interpolation ineffective for novel data. We successfully trained a light language model to learn these tasks and ran a number of experiments to investigate the extrapolation capabilities and internal information processing. Our findings support the hypotheses that the language model works as an Encoding-Regression-Decoding machine where the computation takes place in the value space once the input token representation is mapped to an appropriate intern
    
[^12]: 超越多链思维：基于元推理的问题解答方法

    Answering Questions by Meta-Reasoning over Multiple Chains of Thought. (arXiv:2304.13007v1 [cs.CL])

    [http://arxiv.org/abs/2304.13007](http://arxiv.org/abs/2304.13007)

    本论文提出了基于元推理的Multi-Chain Reasoning (MCR)方法，该方法检查多个推理链，混合它们之间的信息并选择最相关的事实，从而超越多链思维，解决多跳QA问题。 实验结果表明MCR胜过多个强基线，解释质量高。

    

    现代多跳问题解答（QA）系统通常将问题分解为一系列思考步骤（CoT），然后才得出最终答案。通常来说，多个链条被抽样并通过最终答案的投票机制进行聚合，但中间步骤本身被丢弃。虽然这种方法提高了性能，但它们并不考虑链之间的中间步骤之间的关系，并且不提供预测答案的统一解释。我们引入了基于元推理的 Multi-Chain Reasoning (MCR) 方法，该方法利用大型语言模型来超越多个思考链，而不是聚合回答。MCR检查不同的推理链，混合它们之间的信息并选择在生成解释和预测答案时最相关的事实。MCR在7个多跳QA数据集上胜过强基线。此外，我们的分析表明MCR的解释具有高质量。

    Modern systems for multi-hop question answering (QA) typically break questions into a sequence of reasoning steps, termed chain-of-thought (CoT), before arriving at a final answer. Often, multiple chains are sampled and aggregated through a voting mechanism over the final answers, but the intermediate steps themselves are discarded. While such approaches improve performance, they do not consider the relations between intermediate steps across chains and do not provide a unified explanation for the predicted answer. We introduce Multi-Chain Reasoning (MCR), an approach which prompts large language models to meta-reason over multiple chains of thought, rather than aggregating their answers. MCR examines different reasoning chains, mixes information between them and selects the most relevant facts in generating an explanation and predicting the answer. MCR outperforms strong baselines on 7 multi-hop QA datasets. Moreover, our analysis reveals that MCR explanations exhibit high quality, en
    

