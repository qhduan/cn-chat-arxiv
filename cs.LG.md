# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [What is different between these datasets?](https://arxiv.org/abs/2403.05652) | 这里是中文总结出的一句话要点 |
| [^2] | [DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup](https://arxiv.org/abs/2306.03897) | 在无监督学习设置中，提出了一种名为DANSE的基于数据驱动的非线性状态估计方法，利用数据驱动的循环神经网络捕捉模型无关过程中的潜在非线性动态。 |
| [^3] | [Is Pre-training Truly Better Than Meta-Learning?.](http://arxiv.org/abs/2306.13841) | 在少样本学习中，当数据集的正式多样性较低时，预训练模型（PT）胜过模型无关元学习（MAML）。当正式多样性较高时，MAML更好。 |
| [^4] | [Packed-Ensembles for Efficient Uncertainty Estimation.](http://arxiv.org/abs/2210.09184) | Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。 |

# 详细

[^1]: 这里是翻译过的论文标题

    What is different between these datasets?

    [https://arxiv.org/abs/2403.05652](https://arxiv.org/abs/2403.05652)

    这里是中文总结出的一句话要点

    

    这里是翻译过的论文摘要

    arXiv:2403.05652v1 Announce Type: cross  Abstract: The performance of machine learning models heavily depends on the quality of input data, yet real-world applications often encounter various data-related challenges. One such challenge could arise when curating training data or deploying the model in the real world - two comparable datasets in the same domain may have different distributions. While numerous techniques exist for detecting distribution shifts, the literature lacks comprehensive approaches for explaining dataset differences in a human-understandable manner. To address this gap, we propose a suite of interpretable methods (toolbox) for comparing two datasets. We demonstrate the versatility of our approach across diverse data modalities, including tabular data, language, images, and signals in both low and high-dimensional settings. Our methods not only outperform comparable and related approaches in terms of explanation quality and correctness, but also provide actionable,
    
[^2]: DANSE: 无监督学习设置中模型无关过程的基于数据驱动的非线性状态估计

    DANSE: Data-driven Non-linear State Estimation of Model-free Process in Unsupervised Learning Setup

    [https://arxiv.org/abs/2306.03897](https://arxiv.org/abs/2306.03897)

    在无监督学习设置中，提出了一种名为DANSE的基于数据驱动的非线性状态估计方法，利用数据驱动的循环神经网络捕捉模型无关过程中的潜在非线性动态。

    

    我们解决了在无监督学习设置中针对模型无关过程的贝叶斯状态估计和预测任务。对于模型无关过程，我们没有任何关于过程动态的先验知识。在文章中，我们提出了DANSE——一种基于数据驱动的非线性状态估计方法。DANSE提供了给定状态的线性测量的封闭形式后验概率。此外，它还提供了预测的封闭形式后验概率。DANSE中使用数据驱动的循环神经网络（RNN）来提供状态先验的参数。先验依赖于过去的测量作为输入，然后使用当前测量作为输入找到状态的封闭形式后验概率。数据驱动的RNN捕捉模型无关过程的潜在非线性动态。DANSE的训练，主要是学习RNN的参数，是使用无监督的方法进行的。

    arXiv:2306.03897v2 Announce Type: replace-cross  Abstract: We address the tasks of Bayesian state estimation and forecasting for a model-free process in an unsupervised learning setup. For a model-free process, we do not have any a-priori knowledge of the process dynamics. In the article, we propose DANSE -- a Data-driven Nonlinear State Estimation method. DANSE provides a closed-form posterior of the state of the model-free process, given linear measurements of the state. In addition, it provides a closed-form posterior for forecasting. A data-driven recurrent neural network (RNN) is used in DANSE to provide the parameters of a prior of the state. The prior depends on the past measurements as input, and then we find the closed-form posterior of the state using the current measurement as input. The data-driven RNN captures the underlying non-linear dynamics of the model-free process. The training of DANSE, mainly learning the parameters of the RNN, is executed using an unsupervised lea
    
[^3]: 预训练真的比元学习更好吗？

    Is Pre-training Truly Better Than Meta-Learning?. (arXiv:2306.13841v1 [cs.LG])

    [http://arxiv.org/abs/2306.13841](http://arxiv.org/abs/2306.13841)

    在少样本学习中，当数据集的正式多样性较低时，预训练模型（PT）胜过模型无关元学习（MAML）。当正式多样性较高时，MAML更好。

    

    在少样本学习的背景下，目前普遍认为固定的预训练模型（PT）加上在评价时微调最后一层，胜过标准的元学习算法。我们通过深入的实证研究和广泛的数据集比较PT和模型无关元学习（MAML）这些说法。与以前的工作不同，我们强调使用相同的体系结构、相同的优化器，以及所有模型都训练到收敛。关键地，我们使用一个更严格的统计工具——效应量（Cohen's d）——来确定使用PT与使用MAML之间的模型差异的实际意义。然后使用一个预先提出的度量——多样性系数——来计算数据集的平均正式多样性。使用这种分析，我们证明了以下事实：1. 当数据集的正式多样性较低时，PT在平均意义上胜过MAML；2. 当正式多样性较高时，MAML胜过PT。

    In the context of few-shot learning, it is currently believed that a fixed pre-trained (PT) model, along with fine-tuning the final layer during evaluation, outperforms standard meta-learning algorithms. We re-evaluate these claims under an in-depth empirical examination of an extensive set of formally diverse datasets and compare PT to Model Agnostic Meta-Learning (MAML). Unlike previous work, we emphasize a fair comparison by using: the same architecture, the same optimizer, and all models trained to convergence. Crucially, we use a more rigorous statistical tool -- the effect size (Cohen's d) -- to determine the practical significance of the difference between a model trained with PT vs. a MAML. We then use a previously proposed metric -- the diversity coefficient -- to compute the average formal diversity of a dataset. Using this analysis, we demonstrate the following: 1. when the formal diversity of a data set is low, PT beats MAML on average and 2. when the formal diversity is hi
    
[^4]: 紧凑集成用于高效的不确定性估计

    Packed-Ensembles for Efficient Uncertainty Estimation. (arXiv:2210.09184v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.09184](http://arxiv.org/abs/2210.09184)

    Packed-Ensembles是一种能够在标准神经网络内运行的轻量级结构化集合，它通过精心调节编码空间的维度来设计。该方法在不损失效果的情况下提高了训练和推理速度。

    

    深度集成是实现关键指标（如准确性、校准、不确定性估计和超出分布检测）卓越性能的突出方法。但是，现实系统的硬件限制限制了更小的集合和较低容量的网络，严重损害了它们的性能和属性。我们引入了一种称为Packed-Ensembles（PE）的策略，通过精心调节其编码空间的维度来设计和训练轻量级结构化集合。我们利用组卷积将集合并行化为单个共享骨干，并进行前向传递以提高训练和推理速度。PE旨在在标准神经网络的内存限制内运行。

    Deep Ensembles (DE) are a prominent approach for achieving excellent performance on key metrics such as accuracy, calibration, uncertainty estimation, and out-of-distribution detection. However, hardware limitations of real-world systems constrain to smaller ensembles and lower-capacity networks, significantly deteriorating their performance and properties. We introduce Packed-Ensembles (PE), a strategy to design and train lightweight structured ensembles by carefully modulating the dimension of their encoding space. We leverage grouped convolutions to parallelize the ensemble into a single shared backbone and forward pass to improve training and inference speeds. PE is designed to operate within the memory limits of a standard neural network. Our extensive research indicates that PE accurately preserves the properties of DE, such as diversity, and performs equally well in terms of accuracy, calibration, out-of-distribution detection, and robustness to distribution shift. We make our c
    

