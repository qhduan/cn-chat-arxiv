# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Comparing Bad Apples to Good Oranges: Aligning Large Language Models via Joint Preference Optimization](https://arxiv.org/abs/2404.00530) | 本研究提出了一种新的偏好获取方法，通过DOVE协议对指令-响应对的联合概率进行优化，以对齐大型语言模型。 |
| [^2] | [Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators](https://arxiv.org/abs/2403.16950) | 在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。 |
| [^3] | [Efficient Tool Use with Chain-of-Abstraction Reasoning](https://arxiv.org/abs/2401.17464) | 该方法通过让LLM首先解码抽象推理链，然后调用领域工具填充具体知识，使得LLM能够更好地利用工具进行多步推理，并且具有通用性和鲁棒性。 |
| [^4] | [Preference-grounded Token-level Guidance for Language Model Fine-tuning.](http://arxiv.org/abs/2306.00398) | 本论文通过设计一个交替训练过程，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。实验证明该方法在多个文本生成任务中表现良好。 |

# 详细

[^1]: 将坏苹果与好橘子进行比较：通过联合优化偏好对齐大型语言模型

    Comparing Bad Apples to Good Oranges: Aligning Large Language Models via Joint Preference Optimization

    [https://arxiv.org/abs/2404.00530](https://arxiv.org/abs/2404.00530)

    本研究提出了一种新的偏好获取方法，通过DOVE协议对指令-响应对的联合概率进行优化，以对齐大型语言模型。

    

    一种常见的对齐大型语言模型（LLMs）的技术依赖于通过比较在固定上下文中条件生成的多个生成的人类偏好。然而，当这些生成放置在相同的上下文中时，这仅利用了成对比较。然而，这种条件排名通常无法捕获人类偏好的复杂和多维方面。在这项工作中，我们重新审视偏好获取的传统范式，并提出了一个基于在指令-响应对上联合引发偏好的新轴。虽然先前的偏好优化是针对条件排名协议（例如，DPO）设计的，但我们提出的偏好获取协议引入了DOVE，这是一个新的偏好优化目标，通过提升所选指令-响应对的联合概率来降低所拒绝指令-响应对的概率。

    arXiv:2404.00530v1 Announce Type: cross  Abstract: A common technique for aligning large language models (LLMs) relies on acquiring human preferences by comparing multiple generations conditioned on a fixed context. This only leverages the pairwise comparisons when the generations are placed in an identical context. However, such conditional rankings often fail to capture the complex and multidimensional aspects of human preferences. In this work, we revisit the traditional paradigm of preference acquisition and propose a new axis that is based on eliciting preferences jointly over the instruction-response pairs. While prior preference optimizations are designed for conditional ranking protocols (e.g., DPO), our proposed preference acquisition protocol introduces DOVE, a new preference optimization objective that upweights the joint probability of the chosen instruction-response pair over the rejected instruction-response pair. Interestingly, we find that the LLM trained with joint ins
    
[^2]: 与人类判断相一致：大型语言模型评估中成对偏好的作用

    Aligning with Human Judgement: The Role of Pairwise Preference in Large Language Model Evaluators

    [https://arxiv.org/abs/2403.16950](https://arxiv.org/abs/2403.16950)

    在大型语言模型评估中，通过引入成对偏好搜索方法PAIRS，成功解决了LLMs与人类判断不一致的问题，并取得了优于直接打分的最先进性能。

    

    大型语言模型（LLMs）作为自动评估器在评估生成的自然语言质量方面表现出有希望的能力。然而，LLMs在评估中仍存在偏见，常常难以生成与人类评估一致的连贯评估。在这项工作中，我们首先对LLM评估器与人类判断之间的不一致进行系统研究，揭示现有旨在减轻偏见的校准方法不足以有效将LLM评估器对齐。受到RLHF中对偏好数据的使用的启发，我们将评估形式化为一个排序问题，并引入Pairwise-preference Search（PAIRS），这是一种以LLMs进行成对比较并有效对候选文本进行排序的基于不确定性引导的搜索方法。PAIRS在代表性评估任务上实现了最先进的性能，并且显示出比直接打分有显著改进。

    arXiv:2403.16950v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated promising capabilities as automatic evaluators in assessing the quality of generated natural language. However, LLMs still exhibit biases in evaluation and often struggle to generate coherent evaluations that align with human assessments. In this work, we first conduct a systematic study of the misalignment between LLM evaluators and human judgement, revealing that existing calibration methods aimed at mitigating biases are insufficient for effectively aligning LLM evaluators. Inspired by the use of preference data in RLHF, we formulate the evaluation as a ranking problem and introduce Pairwise-preference Search (PAIRS), an uncertainty-guided search method that employs LLMs to conduct pairwise comparisons and efficiently ranks candidate texts. PAIRS achieves state-of-the-art performance on representative evaluation tasks and demonstrates significant improvements over direct scoring. Furthe
    
[^3]: 使用抽象链推理的高效工具使用

    Efficient Tool Use with Chain-of-Abstraction Reasoning

    [https://arxiv.org/abs/2401.17464](https://arxiv.org/abs/2401.17464)

    该方法通过让LLM首先解码抽象推理链，然后调用领域工具填充具体知识，使得LLM能够更好地利用工具进行多步推理，并且具有通用性和鲁棒性。

    

    为了实现与人类期望一致的准确推理，大型语言模型（LLM）需要将推理与现实世界的知识（例如网络事实、数学和物理规则）联系起来。工具可以帮助LLM获取这些外部知识，但是在多步推理问题中仍然存在挑战，即如何精细调整LLM代理（例如Toolformer）以调用工具，其中相互连接的工具调用需要整体化和高效的工具使用规划。在这项工作中，我们提出了一种新的方法，让LLM在多步推理中更好地利用工具。我们的方法是通过训练LLM首先用抽象占位符解码推理链，然后调用领域工具以填充具体知识来实现每个推理链。抽象链的规划使LLM能够学习更通用的推理策略，对于与不同推理问题相关的领域知识（例如数学结果）的变化具有鲁棒性。它还允许LLM执行解码操作

    To achieve faithful reasoning that aligns with human expectations, large language models (LLMs) need to ground their reasoning to real-world knowledge (e.g., web facts, math and physical rules). Tools help LLMs access this external knowledge, but there remains challenges for fine-tuning LLM agents (e.g., Toolformer) to invoke tools in multi-step reasoning problems, where inter-connected tool calls require holistic and efficient tool usage planning.   In this work, we propose a new method for LLMs to better leverage tools in multi-step reasoning. Our method, Chain-of-Abstraction (CoA), trains LLMs to first decode reasoning chains with abstract placeholders, and then call domain tools to reify each reasoning chain by filling in specific knowledge. This planning with abstract chains enables LLMs to learn more general reasoning strategies, which are robust to shifts of domain knowledge (e.g., math results) relevant to different reasoning questions. It also allows LLMs to perform decoding a
    
[^4]: 基于偏好的语言模型微调中的标记级指导

    Preference-grounded Token-level Guidance for Language Model Fine-tuning. (arXiv:2306.00398v1 [cs.CL])

    [http://arxiv.org/abs/2306.00398](http://arxiv.org/abs/2306.00398)

    本论文通过设计一个交替训练过程，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。实验证明该方法在多个文本生成任务中表现良好。

    

    将偏好与语言模型（LMs）相匹配是自然语言生成中的一个重要问题。关键挑战是偏好通常在序列级别上提供，而LM训练和生成都发生在标记级别上。因此，偏好和LM训练损失之间存在颗粒度不匹配，这可能会复杂化学习问题。本文通过开发一种交替训练过程解决了这个问题，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。为了学习指导，我们设计了一个框架，将模仿学习中的成对偏好学习扩展到可变长度LM生成和利用多个生成之间的偏好。对于LM训练，基于监督数据量，我们提出了两种利用学习到的指导的极简主义学习目标。在实验中，我们的方法在多个文本生成任务中表现良好，表明我们的指导提高了生成序列的质量，允许更精细的控制。

    Aligning language models (LMs) with preferences is an important problem in natural language generation. A key challenge is that preferences are typically provided at the sequence level while LM training and generation both occur at the token level. There is, therefore, a granularity mismatch between the preference and the LM training losses, which may complicate the learning problem. In this paper, we address this issue by developing an alternate training process, where we iterate between grounding the sequence-level preference into token-level training guidance, and improving the LM with the learned guidance. For guidance learning, we design a framework that extends the pairwise-preference learning in imitation learning to both variable-length LM generation and utilizing the preference among multiple generations. For LM training, based on the amount of supervised data, we present two minimalist learning objectives that utilize the learned guidance. In experiments, our method performs 
    

