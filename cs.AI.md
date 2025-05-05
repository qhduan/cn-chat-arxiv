# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning](https://arxiv.org/abs/2403.07404) | 早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点 |
| [^2] | [Wiki-TabNER:Advancing Table Interpretation Through Named Entity Recognition](https://arxiv.org/abs/2403.04577) | 本文提出了一个新的挑战性数据集，并介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别，并提出了一个提示框架用于评估大型语言模型在这一新任务上的效果。 |
| [^3] | [AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL](https://arxiv.org/abs/2311.09830) | AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。 |

# 详细

[^1]: 提高推理速度和减少遗忘：早期退出网络在持续学习中的双重好处

    Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning

    [https://arxiv.org/abs/2403.07404](https://arxiv.org/abs/2403.07404)

    早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点

    

    arXiv:2403.07404v1 公告类型: 跨界 摘要: 受深度神经网络能源高效利用需求驱动，早期退出方法备受关注。这些策略通过在网络早期做出决定，实现快速预测，从而节省计算时间和资源。然而，迄今为止，早期退出网络仅针对静态数据分布进行了开发，限制了它们在具有持续非静态数据的实际场景中的应用。本研究旨在探讨早期退出网络的持续学习。我们改编现有的持续学习方法以适应早期退出架构，并研究它们在持续设置中的行为。我们注意到，早期网络层表现出减少遗忘，即使使用的资源显著更少，也能胜过标准网络。此外，我们分析任务最近性偏差对早期退出推理的影响，并提出任务...

    arXiv:2403.07404v1 Announce Type: cross  Abstract: Driven by the demand for energy-efficient employment of deep neural networks, early-exit methods have experienced a notable increase in research attention. These strategies allow for swift predictions by making decisions early in the network, thereby conserving computation time and resources. However, so far the early-exit networks have only been developed for stationary data distributions, which restricts their application in real-world scenarios with continuous non-stationary data. This study aims to explore the continual learning of the early-exit networks. We adapt existing continual learning methods to fit with early-exit architectures and investigate their behavior in the continual setting. We notice that early network layers exhibit reduced forgetting and can outperform standard networks even when using significantly fewer resources. Furthermore, we analyze the impact of task-recency bias on early-exit inference and propose Task
    
[^2]: Wiki-TabNER:通过命名实体识别推进表格解释

    Wiki-TabNER:Advancing Table Interpretation Through Named Entity Recognition

    [https://arxiv.org/abs/2403.04577](https://arxiv.org/abs/2403.04577)

    本文提出了一个新的挑战性数据集，并介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别，并提出了一个提示框架用于评估大型语言模型在这一新任务上的效果。

    

    arXiv:2403.04577v1 发布类型：新摘要：网络表格包含大量宝贵知识，激发了旨在解决表格解释（TI）任务的表格语言模型。本文分析了用于评估TI任务的广泛使用的基准数据集，特别关注实体链接任务。我们的分析显示，该数据集过于简化，可能降低其用于全面评估的有效性，并未准确代表表格在现实世界中的外观。为克服这一缺点，我们构建并注释了一个更具挑战性的新数据集。除了介绍新数据集外，我们还介绍了一个旨在解决实体链接任务的新问题：单元格内的命名实体识别。最后，我们提出了一个提示框架，用于评估新开发的大型语言模型（LLMs）在这一新的TI任务上。我们在各种设置下对提示LLMs进行实验证明，其中我们同时使用了随机

    arXiv:2403.04577v1 Announce Type: new  Abstract: Web tables contain a large amount of valuable knowledge and have inspired tabular language models aimed at tackling table interpretation (TI) tasks. In this paper, we analyse a widely used benchmark dataset for evaluation of TI tasks, particularly focusing on the entity linking task. Our analysis reveals that this dataset is overly simplified, potentially reducing its effectiveness for thorough evaluation and failing to accurately represent tables as they appear in the real-world. To overcome this drawback, we construct and annotate a new more challenging dataset. In addition to introducing the new dataset, we also introduce a novel problem aimed at addressing the entity linking task: named entity recognition within cells. Finally, we propose a prompting framework for evaluating the newly developed large language models (LLMs) on this novel TI task. We conduct experiments on prompting LLMs under various settings, where we use both random
    
[^3]: AutoPlanBench: 从PDDL自动生成LLM规划器的基准测试

    AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL

    [https://arxiv.org/abs/2311.09830](https://arxiv.org/abs/2311.09830)

    AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。

    

    LLMs（逻辑-概率模型）在规划任务中的应用越来越广泛，但是它们在规划和推理方面的能力尚不明确。我们提出了AutoPlanBench，一种将PDDL中的规划基准测试自动转换为文本描述的新方法，并提供了使用我们方法创建的基准测试数据集。我们展示了最好的LLM规划器在某些规划任务上表现良好，但其他任务仍然超出了当前方法的能力范围。

    LLMs are being increasingly used for planning-style tasks, but their capabilities for planning and reasoning are poorly understood. We present AutoPlanBench, a novel method for automatically converting planning benchmarks written in PDDL into textual descriptions and offer a benchmark dataset created with our method. We show that while the best LLM planners do well on some planning tasks, others remain out of reach of current methods.
    

