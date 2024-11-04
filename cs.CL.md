# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text](https://arxiv.org/abs/2403.09131) | ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。 |
| [^2] | [Metric-aware LLM inference](https://arxiv.org/abs/2403.04182) | 提出了度量感知的LLM推断方法，通过优化自定义指标来改进推断性能 |
| [^3] | [Gradient-Free Adaptive Global Pruning for Pre-trained Language Models](https://arxiv.org/abs/2402.17946) | 提出了自适应全局剪枝（AdaGP）框架，通过重新定义全局剪枝过程为可管理的协调子问题，实现对大型语言模型的资源高效优化，显著提高性能。 |
| [^4] | [Can Large Language Model Agents Simulate Human Trust Behaviors?](https://arxiv.org/abs/2402.04559) | 大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。 |
| [^5] | [When Large Language Models Meet Vector Databases: A Survey](https://arxiv.org/abs/2402.01763) | 本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。 |
| [^6] | [ChOiRe: Characterizing and Predicting Human Opinions with Chain of Opinion Reasoning](https://arxiv.org/abs/2311.08385) | ChOiRe是一个通过观点链推理表征和预测人类观点的框架，结合用户明确和隐式的个人角色特征，实现了对人类观点的预测。 |
| [^7] | [MAPLE: Mobile App Prediction Leveraging Large Language model Embeddings.](http://arxiv.org/abs/2309.08648) | MAPLE是一个利用大型语言模型嵌入进行移动应用预测的模型，通过严格测试验证了其在解密复杂模式和理解用户环境方面的能力，并强调了语言模型在不同领域中的广泛适用性。 |
| [^8] | [$FPDM$: Domain-Specific Fast Pre-training Technique using Document-Level Metadata.](http://arxiv.org/abs/2306.06190) | 本文提出了$FPDM$，使用文档元数据和领域特定分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。$FPDM$通过句子级别的输入预训练开放领域的编码器，在微调时使用词汇级别的输入，性能优于其他基于transformer的模型。 |

# 详细

[^1]: ProSwitch：知识引导的语言模型微调，生成专业和非专业风格的文本

    ProSwitch: Knowledge-Guided Language Model Fine-Tuning to Generate Professional and Non-Professional Styled Text

    [https://arxiv.org/abs/2403.09131](https://arxiv.org/abs/2403.09131)

    ProSwitch通过知识引导的指令微调，在专业和非专业风格之间生成文本，并在专业性评估和质量评估方面表现出优越性。

    

    大语言模型（LLMs）在各种语言应用中表现出有效性，包括文本摘要和可控文本生成。然而，关于它们通过微调在不同风格间切换的能力的研究仍未被充分探讨。本研究聚焦于文本专业性，并引入了一种新颖的方法，名为ProSwitch，通过知识引导的指令微调，使语言模型具备生成专业和非专业回复的能力。ProSwitch分为三个阶段：数据准备，用于收集领域知识和训练语料库；指令微调，用于优化带有多种指令格式的语言模型；全面评估，用于评估生成文本的专业性区分能力和基于参考的质量。 ProSwitch相对于通用和专门语言模型的比较分析显示了我们的方法的优越性。

    arXiv:2403.09131v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated efficacy in various linguistic applications, including text summarization and controlled text generation. However, studies into their capacity of switching between styles via fine-tuning remain underexplored. This study concentrates on textual professionalism and introduces a novel methodology, named ProSwitch, which equips a language model with the ability to produce both professional and non-professional responses through knowledge-guided instruction tuning. ProSwitch unfolds across three phases: data preparation for gathering domain knowledge and training corpus; instruction tuning for optimizing language models with multiple levels of instruction formats; and comprehensive evaluation for assessing the professionalism discrimination and reference-based quality of generated text. Comparative analysis of ProSwitch against both general and specialized language models reveals that our appro
    
[^2]: 度量感知的LLM推断

    Metric-aware LLM inference

    [https://arxiv.org/abs/2403.04182](https://arxiv.org/abs/2403.04182)

    提出了度量感知的LLM推断方法，通过优化自定义指标来改进推断性能

    

    大型语言模型（LLMs）已经在各种NLP任务中展示出强大的结果。通常，输出是通过从LLM的基础分布中进行自回归采样获得的。我们表明，这种推断策略对于一系列任务和相关的评估指标可能是次优的。为此，我们提出了度量感知的LLM推断：一种在推断时针对自定义指标进行优化的决策理论方法。我们在学术基准数据集和公开可用模型上报告了相对基线的改进。

    arXiv:2403.04182v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong results on a range of NLP tasks. Typically, outputs are obtained via autoregressive sampling from the LLM's underlying distribution. We show that this inference strategy can be suboptimal for a range of tasks and associated evaluation metrics. As a remedy, we propose metric aware LLM inference: a decision theoretic approach optimizing for custom metrics at inference time. We report improvements over baselines on academic benchmarks and publicly available models.
    
[^3]: 无梯度自适应全局剪枝用于预训练语言模型

    Gradient-Free Adaptive Global Pruning for Pre-trained Language Models

    [https://arxiv.org/abs/2402.17946](https://arxiv.org/abs/2402.17946)

    提出了自适应全局剪枝（AdaGP）框架，通过重新定义全局剪枝过程为可管理的协调子问题，实现对大型语言模型的资源高效优化，显著提高性能。

    

    大型语言模型（LLMs）如LLaMA和GPT在自然语言处理中的转变性影响受到它们计算需求过高的限制。剪枝作为一种关键的压缩策略出现，引入稀疏性以增强内存和计算效率。然而，传统的全局剪枝对LLMs来说由于可扩展性问题而不实用，而本地剪枝，尽管效率高，却导致次优解决方案。为解决这些挑战，我们提出了自适应全局剪枝（AdaGP），这是一个重新定义全局剪枝处理为可管理的协调子问题的新框架，可以实现资源有效的全局最优化优化。AdaGP的方法将LLMs概念化为一系列模块化函数，并利用辅助变量进行问题分解，不仅便于在LLMs上实现实际应用，而且显示出显著的性能改进。

    arXiv:2402.17946v1 Announce Type: new  Abstract: The transformative impact of large language models (LLMs) like LLaMA and GPT on natural language processing is countered by their prohibitive computational demands. Pruning has emerged as a pivotal compression strategy, introducing sparsity to enhance both memory and computational efficiency. Yet, traditional global pruning is impractical for LLMs due to scalability issues, while local pruning, despite its efficiency, leads to suboptimal solutions. Addressing these challenges, we propose Adaptive Global Pruning (AdaGP), a novel framework that redefines the global pruning process into manageable, coordinated subproblems, allowing for resource-efficient optimization with global optimality. AdaGP's approach, which conceptualizes LLMs as a chain of modular functions and leverages auxiliary variables for problem decomposition, not only facilitates a pragmatic application on LLMs but also demonstrates significant performance improvements, part
    
[^4]: 大语言模型代理能够模拟人类的信任行为吗？

    Can Large Language Model Agents Simulate Human Trust Behaviors?

    [https://arxiv.org/abs/2402.04559](https://arxiv.org/abs/2402.04559)

    大语言模型代理能够模拟人类的信任行为，表现出在信任游戏中的信任行为，并且与人类行为具有高度一致性，但存在一些偏见和对代理与人类的差异。

    

    大语言模型（LLM）代理已经越来越多地被采用作为模拟工具，用于模拟人类在社会科学等领域中的行为。然而，一个基本的问题仍然存在：LLM代理是否真的能够模拟人类行为？在本文中，我们专注于人类互动中最关键的行为之一，信任，旨在调查LLM代理是否能够模拟人类的信任行为。我们首先发现，在被行为经济学广泛接受的信任游戏框架下，LLM代理通常表现出信任行为，称为代理信任。然后，我们发现LLM代理在信任行为方面与人类具有较高的行为一致性，表明使用LLM代理模拟人类的信任行为是可行的。此外，我们还探索了代理信任中的偏见以及代理信任在对代理和人类之间的差异方面的内在特性。我们还探讨了包括高级推理策略在内的条件下代理信任的内在特性。

    Large Language Model (LLM) agents have been increasingly adopted as simulation tools to model humans in applications such as social science. However, one fundamental question remains: can LLM agents really simulate human behaviors? In this paper, we focus on one of the most critical behaviors in human interactions, trust, and aim to investigate whether or not LLM agents can simulate human trust behaviors. We first find that LLM agents generally exhibit trust behaviors, referred to as agent trust, under the framework of Trust Games, which are widely recognized in behavioral economics. Then, we discover that LLM agents can have high behavioral alignment with humans regarding trust behaviors, indicating the feasibility to simulate human trust behaviors with LLM agents. In addition, we probe into the biases in agent trust and the differences in agent trust towards agents and humans. We also explore the intrinsic properties of agent trust under conditions including advanced reasoning strate
    
[^5]: 当大型语言模型遇上向量数据库：一项综述

    When Large Language Models Meet Vector Databases: A Survey

    [https://arxiv.org/abs/2402.01763](https://arxiv.org/abs/2402.01763)

    本综述论文深入分析了大型语言模型和向量数据库之间的交叉点，大型语言模型的突破带来了新的挑战，而向量数据库提供了潜在的解决方案，可以显著增强人工智能系统管理和利用多样数据的能力。

    

    最近大型语言模型的突破在人类文字处理和生成方面开启了新的领域。然而，随着它们的显著增长，大型语言模型面临着包括幻觉、偏见、实时知识更新以及在商业环境中实施和维护的高成本等重要挑战。而另一种日益流行的工具，向量数据库则为这些挑战提供了潜在的解决方案。这些数据库擅长处理高维数据，并且对于高效的信息检索和语义搜索等任务至关重要。通过与大型语言模型的整合，它们显著增强了人工智能系统管理和更有效地利用多样数据的能力。本综述论文对大型语言模型和向量数据库之间的交叉点进行了深入而独特的分析。

    The recent burst in Large Language Models has opened new frontiers in human-like text processing and generation. However, alongside their remarkable growth, Large Language Models have encountered critical challenges including issues of hallucination, bias, real-time knowledge updates, and the high costs of implementation and maintenance in commercial settings. Vector Databases, another increasingly popular tool, offer potential solutions to these challenges. These databases are adept at handling high-dimensional data and are crucial for tasks such as efficient information retrieval and semantic search. By integrating with Large Language Models, they significantly enhance AI systems' ability to manage and utilize diverse data more effectively. This survey paper provides an in-depth and unique analysis of the intersection between Large Language Models and Vector Databases.
    
[^6]: ChOiRe：通过观点链推理表征和预测人类观点

    ChOiRe: Characterizing and Predicting Human Opinions with Chain of Opinion Reasoning

    [https://arxiv.org/abs/2311.08385](https://arxiv.org/abs/2311.08385)

    ChOiRe是一个通过观点链推理表征和预测人类观点的框架，结合用户明确和隐式的个人角色特征，实现了对人类观点的预测。

    

    将语言模型与人类观点对齐对于增强它们把握人类价值观、喜好和信仰至关重要。我们提出了ChOiRe，一个四步框架，用于预测人类观点，该框架不同地对待用户明确声明的个人角色（即人口统计或意识形态属性）和从用户历史观点推断出的隐式个人角色。ChOiRe包括：（i）一个语言模型分析用户明确的个人角色，以过滤出不相关的属性；（ii）语言模型将隐式人物观点排名成优先列表；（iii）观点链推理（CoO），其中语言模型顺序地分析明确的个人角色和最相关的隐式个人角色以执行观点预测；（iv）以及ChOiRe执行第（iii）步CoO多次，随着隐式个人角色列表不断增加来克服个人角色信息不足以推断最终结果。ChOiRe取得了新的成果。

    arXiv:2311.08385v3 Announce Type: replace  Abstract: Aligning language models (LMs) with human opinion is challenging yet vital to enhance their grasp of human values, preferences, and beliefs. We present ChOiRe, a four-step framework to predict human opinion which differentially models the user explicit personae (i.e. demographic or ideological attributes) that are manually declared, and implicit personae inferred from user historical opinions. ChOiRe consists of (i) an LM analyzing the user explicit personae to filter out irrelevant attributes; (ii) the LM ranking the implicit persona opinions into a preferential list; (iii) Chain-of-Opinion (CoO) reasoning, where the LM sequentially analyzes the explicit personae and the most relevant implicit personae to perform opinion prediction; (iv) and where ChOiRe executes Step (iii) CoO multiple times with increasingly larger lists of implicit personae to overcome insufficient personae information to infer a final result. ChOiRe achieves new
    
[^7]: MAPLE: 基于大型语言模型嵌入的移动应用预测

    MAPLE: Mobile App Prediction Leveraging Large Language model Embeddings. (arXiv:2309.08648v1 [cs.CL])

    [http://arxiv.org/abs/2309.08648](http://arxiv.org/abs/2309.08648)

    MAPLE是一个利用大型语言模型嵌入进行移动应用预测的模型，通过严格测试验证了其在解密复杂模式和理解用户环境方面的能力，并强调了语言模型在不同领域中的广泛适用性。

    

    尽管移动应用的发展迅速，但由于复杂的用户行为和不断演变的环境，预测应用的使用仍然是一个严峻的挑战。为了解决这些问题，本文介绍了Mobile App Prediction Leveraging Large Language Model Embeddings (MAPLE)模型。这种创新的方法利用大型语言模型(LLM)来准确预测应用的使用情况。通过对两个公开数据集进行严格测试，MAPLE的能力在解密复杂模式和理解用户环境方面得到了验证。这些强大的结果证实了MAPLE在不同场景中的多功能性和弹性。尽管其主要设计面向应用预测，但结果也强调了LLM在不同领域中的广泛适用性。通过这项研究，我们强调了LLM在应用使用预测中的潜力，并建议在建模各种领域中的人类行为方面，它们具有变革能力。

    Despite the rapid advancement of mobile applications, predicting app usage remains a formidable challenge due to intricate user behaviours and ever-evolving contexts. To address these issues, this paper introduces the Mobile App Prediction Leveraging Large Language Model Embeddings (MAPLE) model. This innovative approach utilizes Large Language Models (LLMs) to predict app usage accurately. Rigorous testing on two public datasets highlights MAPLE's capability to decipher intricate patterns and comprehend user contexts. These robust results confirm MAPLE's versatility and resilience across various scenarios. While its primary design caters to app prediction, the outcomes also emphasize the broader applicability of LLMs in different domains. Through this research, we emphasize the potential of LLMs in app usage prediction and suggest their transformative capacity in modelling human behaviours across diverse fields.
    
[^8]: 使用文档级元数据的领域特定快速预训练技术$FPDM$

    $FPDM$: Domain-Specific Fast Pre-training Technique using Document-Level Metadata. (arXiv:2306.06190v1 [cs.CL])

    [http://arxiv.org/abs/2306.06190](http://arxiv.org/abs/2306.06190)

    本文提出了$FPDM$，使用文档元数据和领域特定分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。$FPDM$通过句子级别的输入预训练开放领域的编码器，在微调时使用词汇级别的输入，性能优于其他基于transformer的模型。

    

    在各种领域的预训练已显示出在开放领域和领域特定下游任务上具有良好的结果。然而，最先进的transformers需要大量的预训练数据和计算资源。在本文中，我们提出了$FPDM$（Fast Pre-training Technique using Document Level Metadata），这是一个新颖、计算效率高的框架，利用文档元数据和领域特定的分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。最主要的创新在于，在领域特定的预训练过程中，使用句子级别的嵌入作为输入，持续对开放领域的编码器进行预训练（以适应长文档），但在对该编码器进行微调时，则使用词汇级别嵌入作为输入。实验表明，$FPDM$在客户支持、科学和法律等领域的字符级F1分数和其他自动化指标方面优于几种基于transformer的基准，且在下游任务微调后性能下降可以忽略不计。

    Pre-training Transformers has shown promising results on open-domain and domain-specific downstream tasks. However, state-of-the-art Transformers require an unreasonably large amount of pre-training data and compute. In this paper, we propose $FPDM$ (Fast Pre-training Technique using Document Level Metadata), a novel, compute-efficient framework that utilizes Document metadata and Domain-Specific Taxonomy as supervision signals to pre-train transformer encoder on a domain-specific corpus. The main innovation is that during domain-specific pretraining, an open-domain encoder is continually pre-trained using sentence-level embeddings as inputs (to accommodate long documents), however, fine-tuning is done with token-level embeddings as inputs to this encoder. We show that $FPDM$ outperforms several transformer-based baselines in terms of character-level F1 scores and other automated metrics in the Customer Support, Scientific, and Legal Domains, and shows a negligible drop in performance 
    

