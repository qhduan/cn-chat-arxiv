# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey of Computerized Adaptive Testing: A Machine Learning Perspective](https://arxiv.org/abs/2404.00712) | 本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。 |
| [^2] | [Machine Unlearning by Suppressing Sample Contribution](https://arxiv.org/abs/2402.15109) | 本文提出了一种机器遗忘方法，通过最小化输入敏感度来抑制遗忘数据的贡献，并在实验中表现出优异的性能。 |
| [^3] | [Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?](https://arxiv.org/abs/2212.14511) | 该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。 |

# 详细

[^1]: 计算机自适应测试综述：机器学习视角

    Survey of Computerized Adaptive Testing: A Machine Learning Perspective

    [https://arxiv.org/abs/2404.00712](https://arxiv.org/abs/2404.00712)

    本文以机器学习视角综述了计算机自适应测试（CAT），重点解析其测试问题选择算法和如何优化认知诊断模型、题库构建和测试控制。

    

    计算机自适应测试（CAT）提供了一种高效、量身定制的评估考生熟练程度的方法，通过根据他们的表现动态调整测试问题。CAT广泛应用于教育、医疗、体育和社会学等多个领域，彻底改变了测试实践。然而，随着大规模测试的增加复杂性，CAT已经融合了机器学习技术。本文旨在提供一个以机器学习为重点的CAT综述，从新的角度解读这种自适应测试方法。通过研究CAT适应性核心的测试问题选择算法，我们揭示了其功能。此外，我们探讨了认知诊断模型、题库构建和CAT中的测试控制，探索了机器学习如何优化这些组成部分。通过对当前情况的分析，

    arXiv:2404.00712v1 Announce Type: cross  Abstract: Computerized Adaptive Testing (CAT) provides an efficient and tailored method for assessing the proficiency of examinees, by dynamically adjusting test questions based on their performance. Widely adopted across diverse fields like education, healthcare, sports, and sociology, CAT has revolutionized testing practices. While traditional methods rely on psychometrics and statistics, the increasing complexity of large-scale testing has spurred the integration of machine learning techniques. This paper aims to provide a machine learning-focused survey on CAT, presenting a fresh perspective on this adaptive testing method. By examining the test question selection algorithm at the heart of CAT's adaptivity, we shed light on its functionality. Furthermore, we delve into cognitive diagnosis models, question bank construction, and test control within CAT, exploring how machine learning can optimize these components. Through an analysis of curre
    
[^2]: 抑制样本贡献的机器遗忘

    Machine Unlearning by Suppressing Sample Contribution

    [https://arxiv.org/abs/2402.15109](https://arxiv.org/abs/2402.15109)

    本文提出了一种机器遗忘方法，通过最小化输入敏感度来抑制遗忘数据的贡献，并在实验中表现出优异的性能。

    

    机器遗忘（MU）是指从经过良好训练的模型中删除数据，这在实践中非常重要，因为涉及“被遗忘的权利”。本文从训练数据和未见数据对模型贡献的基本区别入手：训练数据对最终模型有贡献，而未见数据没有。我们理论上发现输入敏感度可以近似衡量贡献，并实际设计了一种算法，称为MU-Mis（通过最小化输入敏感度进行机器遗忘），来抑制遗忘数据的贡献。实验结果表明，MU-Mis明显优于最先进的MU方法。此外，MU-Mis与MU的应用更加密切，因为它不需要使用剩余数据。

    arXiv:2402.15109v1 Announce Type: new  Abstract: Machine Unlearning (MU) is to forget data from a well-trained model, which is practically important due to the "right to be forgotten". In this paper, we start from the fundamental distinction between training data and unseen data on their contribution to the model: the training data contributes to the final model while the unseen data does not. We theoretically discover that the input sensitivity can approximately measure the contribution and practically design an algorithm, called MU-Mis (machine unlearning via minimizing input sensitivity), to suppress the contribution of the forgetting data. Experimental results demonstrate that MU-Mis outperforms state-of-the-art MU methods significantly. Additionally, MU-Mis aligns more closely with the application of MU as it does not require the use of remaining data.
    
[^3]: 直接潜在模型学习能够解决线性二次高斯控制问题吗？

    Can Direct Latent Model Learning Solve Linear Quadratic Gaussian Control?

    [https://arxiv.org/abs/2212.14511](https://arxiv.org/abs/2212.14511)

    该论文提出了直接潜在模型学习的方法，用于解决线性二次高斯控制问题，能够在有限样本下找到近似最优状态表示函数和控制器。

    

    我们研究了从潜在高维观测中学习状态表示的任务，目标是控制未知的部分可观察系统。我们采用直接潜在模型学习方法，通过预测与规划直接相关的数量（例如成本）来学习潜在状态空间中的动态模型，而无需重建观测。具体来说，我们专注于一种直观的基于成本驱动的状态表示学习方法，用于解决线性二次高斯（LQG）控制问题，这是最基本的部分可观察控制问题之一。作为我们的主要结果，我们建立了在有限样本下找到近似最优状态表示函数和使用直接学习的潜在模型找到近似最优控制器的保证。据我们所知，尽管以前的相关工作取得了各种经验成功，但在这项工作之前，尚不清楚这种基于成本驱动的潜在模型学习方法是否具有有限样本保证。

    arXiv:2212.14511v2 Announce Type: replace  Abstract: We study the task of learning state representations from potentially high-dimensional observations, with the goal of controlling an unknown partially observable system. We pursue a direct latent model learning approach, where a dynamic model in some latent state space is learned by predicting quantities directly related to planning (e.g., costs) without reconstructing the observations. In particular, we focus on an intuitive cost-driven state representation learning method for solving Linear Quadratic Gaussian (LQG) control, one of the most fundamental partially observable control problems. As our main results, we establish finite-sample guarantees of finding a near-optimal state representation function and a near-optimal controller using the directly learned latent model. To the best of our knowledge, despite various empirical successes, prior to this work it was unclear if such a cost-driven latent model learner enjoys finite-sampl
    

