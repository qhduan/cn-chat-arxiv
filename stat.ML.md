# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self Supervised Correlation-based Permutations for Multi-View Clustering](https://arxiv.org/abs/2402.16383) | 提出了一种基于深度学习的多视图聚类框架，利用新颖的基于置换的规范相关性目标学习融合数据表示，并通过识别多个视图的一致伪标签来学习聚类分配，实验结果表明模型有效性，理论上证明逼近监督线性判别分析（LDA）表示，提供了由错误伪标签注释引起的误差界限。 |
| [^2] | [Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects](https://arxiv.org/abs/2402.04906) | 本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。 |
| [^3] | [Nonlinear Meta-Learning Can Guarantee Faster Rates.](http://arxiv.org/abs/2307.10870) | 非线性元学习可以保证更快的收敛速度。 |
| [^4] | [Sequential Kernelized Independence Testing.](http://arxiv.org/abs/2212.07383) | 该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。 |

# 详细

[^1]: 自监督基于相关性的多视图聚类排序

    Self Supervised Correlation-based Permutations for Multi-View Clustering

    [https://arxiv.org/abs/2402.16383](https://arxiv.org/abs/2402.16383)

    提出了一种基于深度学习的多视图聚类框架，利用新颖的基于置换的规范相关性目标学习融合数据表示，并通过识别多个视图的一致伪标签来学习聚类分配，实验结果表明模型有效性，理论上证明逼近监督线性判别分析（LDA）表示，提供了由错误伪标签注释引起的误差界限。

    

    融合来自不同模态的信息可以增强数据分析任务，包括聚类。然而，现有的多视图聚类（MVC）解决方案仅限于特定领域，或者依赖于次优的且计算需求高的表示和聚类两阶段程序。我们提出了一个基于端到端深度学习的通用数据（图像、表格等）的MVC框架。我们的方法涉及使用基于新颖置换的规范相关性目标来学习有意义的融合数据表示。同时，我们通过识别跨多个视图的一致伪标签来学习聚类分配。我们使用十个MVC基准数据集展示了我们模型的有效性。在理论上，我们证明了我们的模型逼近了监督线性判别分析（LDA）表示。另外，我们提供了由错误伪标签注释引起的误差界限。

    arXiv:2402.16383v1 Announce Type: new  Abstract: Fusing information from different modalities can enhance data analysis tasks, including clustering. However, existing multi-view clustering (MVC) solutions are limited to specific domains or rely on a suboptimal and computationally demanding two-stage procedure of representation and clustering. We propose an end-to-end deep learning-based MVC framework for general data (image, tabular, etc.). Our approach involves learning meaningful fused data representations with a novel permutation-based canonical correlation objective. Concurrently, we learn cluster assignments by identifying consistent pseudo-labels across multiple views. We demonstrate the effectiveness of our model using ten MVC benchmark datasets. Theoretically, we show that our model approximates the supervised linear discrimination analysis (LDA) representation. Additionally, we provide an error bound induced by false-pseudo label annotations.
    
[^2]: 预测个体治疗效果的一致性蒙特卡洛元学习模型

    Conformal Monte Carlo Meta-learners for Predictive Inference of Individual Treatment Effects

    [https://arxiv.org/abs/2402.04906](https://arxiv.org/abs/2402.04906)

    本研究提出了一种新方法，即一致性蒙特卡洛元学习模型，用于预测个体治疗效果。通过利用一致性预测系统、蒙特卡洛采样和CATE元学习模型，该方法生成可用于个性化决策的预测分布。实验结果显示，该方法在保持较小区间宽度的情况下具有强大的实验覆盖范围，可以提供真实个体治疗效果的估计。

    

    认识干预效果，即治疗效果，对于决策至关重要。用条件平均治疗效果 (CATE) 估计等方法通常只提供治疗效果的点估计，而常常需要额外的不确定性量化。因此，我们提出了一个新方法，即一致性蒙特卡洛 (CMC) 元学习模型，利用一致性预测系统、蒙特卡洛采样和 CATE 元学习模型，来产生可用于个性化决策的预测分布。此外，我们展示了结果噪声分布的特定假设如何严重影响这些不确定性预测。尽管如此，CMC框架展示了强大的实验覆盖范围，同时保持较小的区间宽度，以提供真实个体治疗效果的估计。

    Knowledge of the effect of interventions, called the treatment effect, is paramount for decision-making. Approaches to estimating this treatment effect, e.g. by using Conditional Average Treatment Effect (CATE) estimators, often only provide a point estimate of this treatment effect, while additional uncertainty quantification is frequently desired instead. Therefore, we present a novel method, the Conformal Monte Carlo (CMC) meta-learners, leveraging conformal predictive systems, Monte Carlo sampling, and CATE meta-learners, to instead produce a predictive distribution usable in individualized decision-making. Furthermore, we show how specific assumptions on the noise distribution of the outcome heavily affect these uncertainty predictions. Nonetheless, the CMC framework shows strong experimental coverage while retaining small interval widths to provide estimates of the true individual treatment effect.
    
[^3]: 非线性元学习可以保证更快的收敛速度

    Nonlinear Meta-Learning Can Guarantee Faster Rates. (arXiv:2307.10870v1 [stat.ML])

    [http://arxiv.org/abs/2307.10870](http://arxiv.org/abs/2307.10870)

    非线性元学习可以保证更快的收敛速度。

    

    最近许多关于元学习的理论研究旨在利用相关任务中的相似表示结构来简化目标任务，并实现收敛速率的保证。然而，在实践中，表示往往是高度非线性的，引入了每个任务中不可简单平均的非平凡偏差。本研究通过非线性表示推导出元学习的理论保证。

    Many recent theoretical works on \emph{meta-learning} aim to achieve guarantees in leveraging similar representational structures from related tasks towards simplifying a target task. Importantly, the main aim in theory works on the subject is to understand the extent to which convergence rates -- in learning a common representation -- \emph{may scale with the number $N$ of tasks} (as well as the number of samples per task). First steps in this setting demonstrate this property when both the shared representation amongst tasks, and task-specific regression functions, are linear. This linear setting readily reveals the benefits of aggregating tasks, e.g., via averaging arguments. In practice, however, the representation is often highly nonlinear, introducing nontrivial biases in each task that cannot easily be averaged out as in the linear case. In the present work, we derive theoretical guarantees for meta-learning with nonlinear representations. In particular, assuming the shared nonl
    
[^4]: 顺序核独立性测试

    Sequential Kernelized Independence Testing. (arXiv:2212.07383v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.07383](http://arxiv.org/abs/2212.07383)

    该论文介绍了顺序核独立性测试的方法，以解决传统批量测试在流数据上的问题，实现了根据任务复杂性自适应调整样本大小，并在收集新数据后持续监测和控制误报率。

    

    独立性测试是一个经典的统计问题，在固定采集数据之前的批量设置中得到了广泛研究。然而，实践者们往往更喜欢能够根据问题的复杂性进行自适应的程序，而不是事先设定样本大小。理想情况下，这样的程序应该（a）在简单任务上尽早停止（在困难任务上稍后停止），因此更好地利用可用资源，以及（b）在收集新数据之后，持续监测数据并高效地整合统计证据，同时控制误报率。经典的批量测试不适用于流数据：在数据观察后进行有效推断需要对多重测试进行校正，这导致了低功率。遵循通过投注进行测试的原则，我们设计了顺序核独立性测试，克服了这些缺点。我们通过采用由核相关性测度（如Hilbert-）启发的投注来说明我们的广泛框架。

    Independence testing is a classical statistical problem that has been extensively studied in the batch setting when one fixes the sample size before collecting data. However, practitioners often prefer procedures that adapt to the complexity of a problem at hand instead of setting sample size in advance. Ideally, such procedures should (a) stop earlier on easy tasks (and later on harder tasks), hence making better use of available resources, and (b) continuously monitor the data and efficiently incorporate statistical evidence after collecting new data, while controlling the false alarm rate. Classical batch tests are not tailored for streaming data: valid inference after data peeking requires correcting for multiple testing which results in low power. Following the principle of testing by betting, we design sequential kernelized independence tests that overcome such shortcomings. We exemplify our broad framework using bets inspired by kernelized dependence measures, e.g., the Hilbert-
    

