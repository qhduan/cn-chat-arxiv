# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Multidimensional Scaling](https://arxiv.org/abs/2402.04436) | 连续多维标度是关于将距离信息嵌入欧几里得空间的过程，并探讨了在对象集不断增加的情况下，将整个嵌入问题序列视为一个固定空间中的一系列优化问题的方法和结论。 |
| [^2] | [Dual-Directed Algorithm Design for Efficient Pure Exploration.](http://arxiv.org/abs/2310.19319) | 该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。 |
| [^3] | [Multiple Physics Pretraining for Physical Surrogate Models.](http://arxiv.org/abs/2310.02994) | 多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。 |
| [^4] | [Causal thinking for decision making on Electronic Health Records: why and how.](http://arxiv.org/abs/2308.01605) | 本文介绍了在电子健康记录中使用因果思维进行决策的必要性和方法。通过模拟随机试验来个性化决策，以减少数据中的偏见。这对于分析电子健康记录或索赔数据以得出因果结论的最重要陷阱和考虑因素进行了重点强调。 |

# 详细

[^1]: 连续多维标度

    Continuous Multidimensional Scaling

    [https://arxiv.org/abs/2402.04436](https://arxiv.org/abs/2402.04436)

    连续多维标度是关于将距离信息嵌入欧几里得空间的过程，并探讨了在对象集不断增加的情况下，将整个嵌入问题序列视为一个固定空间中的一系列优化问题的方法和结论。

    

    多维标度(MDS)是将关于一组$n$个对象的距离信息嵌入到$d$维欧几里得空间中的过程。最初由心理测量学界构思，MDS关注的是嵌入到一组固定对象上的一组固定距离。现代关注的问题更常涉及到研究与一组不断增加的对象相关联的一系列距离的极限行为，如在随机图的统计推断的渐近理论中出现的问题。点到集合映射理论中的标准结果表明，若$n$固定，则嵌入结构的极限是极限距离的嵌入结构。但如果$n$增加怎么办呢？那么就需要重新制定MDS，以便将整个嵌入问题序列视为一个固定空间中的一系列优化问题。我们提出了这样一种重新制定，并推导出一些结论。

    Multidimensional scaling (MDS) is the act of embedding proximity information about a set of $n$ objects in $d$-dimensional Euclidean space. As originally conceived by the psychometric community, MDS was concerned with embedding a fixed set of proximities associated with a fixed set of objects. Modern concerns, e.g., that arise in developing asymptotic theories for statistical inference on random graphs, more typically involve studying the limiting behavior of a sequence of proximities associated with an increasing set of objects. Standard results from the theory of point-to-set maps imply that, if $n$ is fixed, then the limit of the embedded structures is the embedded structure of the limiting proximities. But what if $n$ increases? It then becomes necessary to reformulate MDS so that the entire sequence of embedding problems can be viewed as a sequence of optimization problems in a fixed space. We present such a reformulation and derive some consequences.
    
[^2]: 高效纯探索的双向算法设计

    Dual-Directed Algorithm Design for Efficient Pure Exploration. (arXiv:2310.19319v1 [stat.ML])

    [http://arxiv.org/abs/2310.19319](http://arxiv.org/abs/2310.19319)

    该论文研究了在有限备选方案集合中的纯探索问题。通过使用对偶变量，提出了一种新的算法设计原则，能够避免组合结构的复杂性，实现高效纯探索，从而准确回答查询问题。

    

    我们考虑在有限的备选方案集合中的随机顺序自适应实验的纯探索问题。决策者的目标是通过最小的测量工作以高置信度准确回答与备选方案相关的查询问题。一个典型的查询问题是确定表现最佳的备选方案，这在排名和选择问题以及机器学习文献中称为最佳臂识别问题。我们专注于固定精度的设定，并导出了一个与样本最优分配有强收敛性概念相关的优化条件的充分条件。使用对偶变量，我们刻画了一个分配是否最优的必要和充分条件。对偶变量的使用使我们能够绕过完全依赖于原始变量的最优条件的组合结构。值得注意的是，这些最优条件使得双向算法设计原则的扩展成为可能。

    We consider pure-exploration problems in the context of stochastic sequential adaptive experiments with a finite set of alternative options. The goal of the decision-maker is to accurately answer a query question regarding the alternatives with high confidence with minimal measurement efforts. A typical query question is to identify the alternative with the best performance, leading to ranking and selection problems, or best-arm identification in the machine learning literature. We focus on the fixed-precision setting and derive a sufficient condition for optimality in terms of a notion of strong convergence to the optimal allocation of samples. Using dual variables, we characterize the necessary and sufficient conditions for an allocation to be optimal. The use of dual variables allow us to bypass the combinatorial structure of the optimality conditions that relies solely on primal variables. Remarkably, these optimality conditions enable an extension of top-two algorithm design princ
    
[^3]: 多物理学预训练用于物理代理模型

    Multiple Physics Pretraining for Physical Surrogate Models. (arXiv:2310.02994v1 [cs.LG])

    [http://arxiv.org/abs/2310.02994](http://arxiv.org/abs/2310.02994)

    多物理学预训练是一种用于物理代理建模的自回归预训练方法，通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。实验证明，单个MPP预训练的变换器可以在所有预训练子任务上与或超过特定任务的基准结果，无需微调，并且在下游任务中，微调MPP训练的模型相较于从头训练的模型，对新物理的预测结果更准确。

    

    我们引入了一种多物理学预训练（MPP）的方法，这是一种自回归任务不可知的预训练方法，用于物理代理建模。MPP通过训练大型代理模型同时预测多个异构物理系统的动力学，学习在不同物理任务中广泛适用的特征。为了有效学习，在这种设置中，我们引入了一种共享嵌入和归一化策略，将多个系统的字段投影到一个共享嵌入空间中。我们在一个涉及流体力学的广泛基准测试中验证了我们方法的有效性。我们表明，单个MPP预训练的变换器能够在所有预训练子任务上与或超过特定任务的基准结果，而无需微调。对于下游任务，我们证明微调MPP训练的模型相较于从头训练的模型，在多个时间步骤上对新物理的预测结果更准确。

    We introduce multiple physics pretraining (MPP), an autoregressive task-agnostic pretraining approach for physical surrogate modeling. MPP involves training large surrogate models to predict the dynamics of multiple heterogeneous physical systems simultaneously by learning features that are broadly useful across diverse physical tasks. In order to learn effectively in this setting, we introduce a shared embedding and normalization strategy that projects the fields of multiple systems into a single shared embedding space. We validate the efficacy of our approach on both pretraining and downstream tasks over a broad fluid mechanics-oriented benchmark. We show that a single MPP-pretrained transformer is able to match or outperform task-specific baselines on all pretraining sub-tasks without the need for finetuning. For downstream tasks, we demonstrate that finetuning MPP-trained models results in more accurate predictions across multiple time-steps on new physics compared to training from
    
[^4]: 用于决策的因果思维在电子健康记录中的应用：为什么以及如何

    Causal thinking for decision making on Electronic Health Records: why and how. (arXiv:2308.01605v1 [stat.ME])

    [http://arxiv.org/abs/2308.01605](http://arxiv.org/abs/2308.01605)

    本文介绍了在电子健康记录中使用因果思维进行决策的必要性和方法。通过模拟随机试验来个性化决策，以减少数据中的偏见。这对于分析电子健康记录或索赔数据以得出因果结论的最重要陷阱和考虑因素进行了重点强调。

    

    准确的预测，如同机器学习一样，可能无法为每个患者提供最佳医疗保健。确实，预测可能受到数据中的捷径（如种族偏见）的驱动。为数据驱动的决策需要因果思维。在这里，我们介绍关键要素，重点关注常规收集的数据，即电子健康记录（EHRs）和索赔数据。使用这些数据评估干预的价值需要谨慎：时间依赖性和现有实践很容易混淆因果效应。我们提供了一个逐步框架，帮助从真实患者记录中构建有效的决策，通过模拟随机试验来个性化决策，例如使用机器学习。我们的框架强调了分析EHRs或索赔数据以得出因果结论时最重要的陷阱和考虑因素。我们在用于重症医学信息市场中的肌酐对败血症死亡率的影响的研究中说明了各种选择。

    Accurate predictions, as with machine learning, may not suffice to provide optimal healthcare for every patient. Indeed, prediction can be driven by shortcuts in the data, such as racial biases. Causal thinking is needed for data-driven decisions. Here, we give an introduction to the key elements, focusing on routinely-collected data, electronic health records (EHRs) and claims data. Using such data to assess the value of an intervention requires care: temporal dependencies and existing practices easily confound the causal effect. We present a step-by-step framework to help build valid decision making from real-life patient records by emulating a randomized trial before individualizing decisions, eg with machine learning. Our framework highlights the most important pitfalls and considerations in analysing EHRs or claims data to draw causal conclusions. We illustrate the various choices in studying the effect of albumin on sepsis mortality in the Medical Information Mart for Intensive C
    

