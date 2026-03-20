# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CADGL: Context-Aware Deep Graph Learning for Predicting Drug-Drug Interactions](https://arxiv.org/abs/2403.17210) | 通过CADGL框架，利用上下文感知深度图学习来预测药物-药物相互作用，解决了现有DDI预测模型在泛化、特征提取和现实应用方面的挑战 |
| [^2] | [Benchmarking the Text-to-SQL Capability of Large Language Models: A Comprehensive Evaluation](https://arxiv.org/abs/2403.02951) | 大型语言模型在文本生成SQL任务中表现出色，但对于最佳提示模板和设计框架仍无共识，新数据集和评估任务有助于全面评估各种方法的表现，并提出了优化解决方案。 |
| [^3] | [MORBDD: Multiobjective Restricted Binary Decision Diagrams by Learning to Sparsify](https://arxiv.org/abs/2403.02482) | 本文提出了一种基于机器学习的BDD稀疏化器MORBDD，通过训练二元分类器来消除不太可能贡献于帕累托解的BDD节点。 |

# 详细

[^1]: CADGL: 上下文感知深度图学习用于预测药物-药物相互作用

    CADGL: Context-Aware Deep Graph Learning for Predicting Drug-Drug Interactions

    [https://arxiv.org/abs/2403.17210](https://arxiv.org/abs/2403.17210)

    通过CADGL框架，利用上下文感知深度图学习来预测药物-药物相互作用，解决了现有DDI预测模型在泛化、特征提取和现实应用方面的挑战

    

    药物-药物相互作用（DDIs）的研究是药物开发过程中的一个关键元素。DDIs发生在一个药物的性质受其他药物包含的影响时。检测有利的DDIs有可能为在实际设置中应用的创新药物的创造和推进铺平道路。然而，现有的DDI预测模型在极端情况下的泛化、稳健特征提取以及现实应用可能性方面持续面临挑战。我们旨在通过利用上下文感知深度图学习的有效性，引入一种名为CADGL的新颖框架来应对这些挑战。基于定制的变分图自编码器（VGAE），我们利用两个上下文预处理器从两个不同视角：局部邻域和分子上下文，在异质图结构中提取特征，捕获关键的结构和生理化学信息。

    arXiv:2403.17210v1 Announce Type: cross  Abstract: Examining Drug-Drug Interactions (DDIs) is a pivotal element in the process of drug development. DDIs occur when one drug's properties are affected by the inclusion of other drugs. Detecting favorable DDIs has the potential to pave the way for creating and advancing innovative medications applicable in practical settings. However, existing DDI prediction models continue to face challenges related to generalization in extreme cases, robust feature extraction, and real-life application possibilities. We aim to address these challenges by leveraging the effectiveness of context-aware deep graph learning by introducing a novel framework named CADGL. Based on a customized variational graph autoencoder (VGAE), we capture critical structural and physio-chemical information using two context preprocessors for feature extraction from two different perspectives: local neighborhood and molecular context, in a heterogeneous graphical structure. Ou
    
[^2]: 评估大型语言模型的文本生成SQL能力：全面评估

    Benchmarking the Text-to-SQL Capability of Large Language Models: A Comprehensive Evaluation

    [https://arxiv.org/abs/2403.02951](https://arxiv.org/abs/2403.02951)

    大型语言模型在文本生成SQL任务中表现出色，但对于最佳提示模板和设计框架仍无共识，新数据集和评估任务有助于全面评估各种方法的表现，并提出了优化解决方案。

    

    大型语言模型（LLMs）已经成为推动文本生成SQL任务的强大工具，明显优于传统方法。然而，作为一个新兴的研究领域，对于最佳提示模板和设计框架仍然没有达成共识。

    arXiv:2403.02951v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have emerged as a powerful tool in advancing the Text-to-SQL task, significantly outperforming traditional methods. Nevertheless, as a nascent research field, there is still no consensus on the optimal prompt templates and design frameworks. Additionally, existing benchmarks inadequately explore the performance of LLMs across the various sub-tasks of the Text-to-SQL process, which hinders the assessment of LLMs' cognitive capabilities and the optimization of LLM-based solutions.To address the aforementioned issues, we firstly construct a new dataset designed to mitigate the risk of overfitting in LLMs. Then we formulate five evaluation tasks to comprehensively assess the performance of diverse methods across various LLMs throughout the Text-to-SQL process.Our study highlights the performance disparities among LLMs and proposes optimal in-context learning solutions tailored to each task. These findings offer
    
[^3]: MORBDD：通过学习稀疏化实现的多目标约束二进制决策图

    MORBDD: Multiobjective Restricted Binary Decision Diagrams by Learning to Sparsify

    [https://arxiv.org/abs/2403.02482](https://arxiv.org/abs/2403.02482)

    本文提出了一种基于机器学习的BDD稀疏化器MORBDD，通过训练二元分类器来消除不太可能贡献于帕累托解的BDD节点。

    

    在多标准决策中，用户寻求约束多目标优化问题的非支配解集，即所谓的帕累托前沿。本文旨在将一种精确的多目标整数线性规划方法引入到启发式领域。我们着眼于二进制决策图（BDDs），首先构建一个代表问题所有可行解的图形，然后遍历该图以提取帕累托前沿。由于帕累托前沿可能呈指数增长，BDD上的枚举可能耗时。我们探讨了如何通过机器学习（ML）将已被证明对单目标问题有效的受限BDDs调整为多目标优化。我们的基于ML的BDD稀疏化器MORBDD首先训练一个二元分类器，以消除不太可能贡献于帕累托解的BDD节点。

    arXiv:2403.02482v1 Announce Type: new  Abstract: In multicriteria decision-making, a user seeks a set of non-dominated solutions to a (constrained) multiobjective optimization problem, the so-called Pareto frontier. In this work, we seek to bring a state-of-the-art method for exact multiobjective integer linear programming into the heuristic realm. We focus on binary decision diagrams (BDDs) which first construct a graph that represents all feasible solutions to the problem and then traverse the graph to extract the Pareto frontier. Because the Pareto frontier may be exponentially large, enumerating it over the BDD can be time-consuming. We explore how restricted BDDs, which have already been shown to be effective as heuristics for single-objective problems, can be adapted to multiobjective optimization through the use of machine learning (ML). MORBDD, our ML-based BDD sparsifier, first trains a binary classifier to eliminate BDD nodes that are unlikely to contribute to Pareto solution
    

