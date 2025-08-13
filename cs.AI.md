# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLM Agent Operating System](https://arxiv.org/abs/2403.16971) | 提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。 |
| [^2] | [BELLA: Black box model Explanations by Local Linear Approximations.](http://arxiv.org/abs/2305.11311) | 本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。 |

# 详细

[^1]: LLM Agent Operating System

    LLM Agent Operating System

    [https://arxiv.org/abs/2403.16971](https://arxiv.org/abs/2403.16971)

    提出了一种将大型语言模型嵌入操作系统中的LLM代理操作系统，旨在优化资源分配、促进代理间上下文切换、实现并发执行以及为代理提供工具服务。

    

    arXiv:2403.16971v1 公告类型: 跨领域 摘要: 部署大型语言模型（LLM）智能代理存在诸多挑战，会损害它们的效率和功效。其中包括代理请求在LLM上的次优调度和资源分配、在代理和LLM之间交互时保持上下文的困难，以及将具有不同能力和专业化的异构代理集成在一起的复杂性。代理数量和复杂性的快速增加进一步加剧了这些问题，通常会导致资源瓶颈和次优资源利用。受到这些挑战的启发，本文提出了AIOS，一种LLM代理操作系统，它将大型语言模型嵌入操作系统（OS）中。具体地，AIOS旨在优化资源分配，促进代理之间的上下文切换，实现代理的并发执行，为代理提供工具服务。

    arXiv:2403.16971v1 Announce Type: cross  Abstract: The integration and deployment of large language model (LLM)-based intelligent agents have been fraught with challenges that compromise their efficiency and efficacy. Among these issues are sub-optimal scheduling and resource allocation of agent requests over the LLM, the difficulties in maintaining context during interactions between agent and LLM, and the complexities inherent in integrating heterogeneous agents with different capabilities and specializations. The rapid increase of agent quantity and complexity further exacerbates these issues, often leading to bottlenecks and sub-optimal utilization of resources. Inspired by these challenges, this paper presents AIOS, an LLM agent operating system, which embeds large language model into operating systems (OS). Specifically, AIOS is designed to optimize resource allocation, facilitate context switch across agents, enable concurrent execution of agents, provide tool service for agents
    
[^2]: BELLA: 通过本地线性逼近进行黑盒模型解释

    BELLA: Black box model Explanations by Local Linear Approximations. (arXiv:2305.11311v1 [cs.LG])

    [http://arxiv.org/abs/2305.11311](http://arxiv.org/abs/2305.11311)

    本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。

    

    近年来，理解黑盒模型的决策过程不仅成为法律要求，也成为评估其性能的另一种方式。然而，现有的事后解释方法依赖于合成数据生成，这引入了不确定性并可能损害解释的可靠性，并且它们 tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a

    In recent years, understanding the decision-making process of black-box models has become not only a legal requirement but also an additional way to assess their performance. However, the state of the art post-hoc interpretation approaches rely on synthetic data generation. This introduces uncertainty and can hurt the reliability of the interpretations. Furthermore, they tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a
    

