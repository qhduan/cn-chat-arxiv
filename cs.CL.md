# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Neural Prompting with Large Language Models.](http://arxiv.org/abs/2309.15427) | 本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。 |
| [^2] | [A Survey on Evaluation of Large Language Models.](http://arxiv.org/abs/2307.03109) | 本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。 |
| [^3] | [Solving Math Word Problems via Cooperative Reasoning induced Language Models.](http://arxiv.org/abs/2210.16257) | 该论文提出了一种基于合作推理诱导的语言模型——CoRe，可以高效地解决数学应用题。实验表明，CoRe 在准确性和效率方面优于现有最先进的方法。 |

# 详细

[^1]: 使用大型语言模型的图神经提示

    Graph Neural Prompting with Large Language Models. (arXiv:2309.15427v1 [cs.CL])

    [http://arxiv.org/abs/2309.15427](http://arxiv.org/abs/2309.15427)

    本文提出了一种名为图神经提示（GNP）的方法，可以帮助大型语言模型从知识图中学习有益的知识，以弥补它们在准确捕捉和返回基于知识的信息方面的固有限制。

    

    大型语言模型（LLMs）在各种语言建模任务中表现出了卓越的泛化能力和出色的性能，但它们在准确捕捉和返回基于知识的信息方面仍存在固有限制。现有的研究已经探索了利用知识图来通过联合训练和定制模型架构增强语言建模，但是将此应用于LLMs存在参数数量庞大和计算成本高的问题。此外，如何利用预训练的LLMs并避免从头开始训练自定义模型仍然是一个开放的问题。在这项工作中，我们提出了图神经提示（GNP），一种新颖的即插即用方法，可以帮助预训练的LLMs从知识图中学习有益的知识。GNP包括各种设计，包括标准的图神经网络编码器、跨模态汇聚模块、域投影器和自监督链接预测目标。在多个实验中展示了GNP的有效性。

    Large Language Models (LLMs) have shown remarkable generalization capability with exceptional performance in various language modeling tasks. However, they still exhibit inherent limitations in precisely capturing and returning grounded knowledge. While existing work has explored utilizing knowledge graphs to enhance language modeling via joint training and customized model architectures, applying this to LLMs is problematic owing to their large number of parameters and high computational cost. In addition, how to leverage the pre-trained LLMs and avoid training a customized model from scratch remains an open question. In this work, we propose Graph Neural Prompting (GNP), a novel plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from KGs. GNP encompasses various designs, including a standard graph neural network encoder, a cross-modality pooling module, a domain projector, and a self-supervised link prediction objective. Extensive experiments on multiple
    
[^2]: 对大型语言模型评估的调查

    A Survey on Evaluation of Large Language Models. (arXiv:2307.03109v1 [cs.CL])

    [http://arxiv.org/abs/2307.03109](http://arxiv.org/abs/2307.03109)

    本文综述了大型语言模型（LLMs）的评估方法，关注三个关键维度：评估什么、在哪里评估以及如何评估。评估任务包括自然语言处理、推理、医学应用、伦理学、教育、自然和社会科学、代理应用等多个领域。本文为社会层面对LLMs潜在风险的理解提供了重要参考。

    

    大型语言模型（LLMs）由于在各种应用中表现出的前所未有的性能而在学术界和工业界越来越受欢迎。随着LLMs在研究和日常使用中继续发挥着重要作用，它们的评估变得越来越关键，不仅在任务水平上，而且在社会层面上，以更好地了解它们的潜在风险。在过去的几年里，已经做出了相当大的努力来从不同的角度来研究LLMs。本文综述了LLMs的这些评估方法，重点关注三个关键维度：评估什么、在哪里评估以及如何评估。首先，我们从评估任务的角度提供了一个概述，涵盖了一般的自然语言处理任务、推理、医学应用、伦理学、教育、自然科学和社会科学、代理应用和其他领域。其次，我们通过深入探讨评估方法和基准答案来回答“在哪里”和“如何”这两个问题。

    Large language models (LLMs) are gaining increasing popularity in both academia and industry, owing to their unprecedented performance in various applications. As LLMs continue to play a vital role in both research and daily use, their evaluation becomes increasingly critical, not only at the task level, but also at the society level for better understanding of their potential risks. Over the past years, significant efforts have been made to examine LLMs from various perspectives. This paper presents a comprehensive review of these evaluation methods for LLMs, focusing on three key dimensions: what to evaluate, where to evaluate, and how to evaluate. Firstly, we provide an overview from the perspective of evaluation tasks, encompassing general natural language processing tasks, reasoning, medical usage, ethics, educations, natural and social sciences, agent applications, and other areas. Secondly, we answer the `where' and `how' questions by diving into the evaluation methods and bench
    
[^3]: 通过合作推理诱导的语言模型解决数学应用题

    Solving Math Word Problems via Cooperative Reasoning induced Language Models. (arXiv:2210.16257v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.16257](http://arxiv.org/abs/2210.16257)

    该论文提出了一种基于合作推理诱导的语言模型——CoRe，可以高效地解决数学应用题。实验表明，CoRe 在准确性和效率方面优于现有最先进的方法。

    

    大规模预训练语言模型 (PLMs) 为需要高水平智能的挑战性问题（如数学应用题）带来了新的机遇。然而，直接应用现有的 PLMs 到数学应用题上会失败，因为其生成的过程缺乏足够的监督，缺乏像人类一样的快速适应性。我们注意到人类的推理过程有一个双重推理框架，包括一个即时反应系统 (system 1) 和一个精细推理系统 (system 2)，整个推理过程由它们的交互决定。这启发我们开发了一种合作推理诱导的 PLM 模型，称为 Cooperative Reasoning (CoRe)，得到了一个像人类推理结构的架构，其中 system 1 作为生成器，system 2 作为验证器。在我们的方法中，生成器负责产生推理路径，验证器用于监督评估以获取可靠的反馈信息。我们在基准数据集上评估了我们的模型，实验结果表明，CoRe 在准确性和效率方面优于现有最先进的方法。

    Large-scale pre-trained language models (PLMs) bring new opportunities to challenging problems, especially those that need high-level intelligence, such as the math word problem (MWPs). However, directly applying existing PLMs to MWPs can fail as the generation process lacks sufficient supervision and thus lacks fast adaptivity as humans. We notice that human reasoning has a dual reasoning framework that consists of an immediate reaction system (system 1) and a delicate reasoning system (system 2), where the entire reasoning is determined by their interaction. This inspires us to develop a cooperative reasoning-induced PLM for solving MWPs, called Cooperative Reasoning (CoRe), resulting in a human-like reasoning architecture with system 1 as the generator and system 2 as the verifier. In our approach, the generator is responsible for generating reasoning paths, and the verifiers are used to supervise the evaluation in order to obtain reliable feedback for the generator. We evaluate our
    

