# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Empirical investigation of multi-source cross-validation in clinical machine learning](https://arxiv.org/abs/2403.15012) | 本研究在多源环境中系统地评估了标准K折交叉验证和留出源交叉验证方法，为实现更全面和真实的精确度评估提供了新的机会 |
| [^2] | [CASPER: Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal Time Series Imputation](https://arxiv.org/abs/2403.11960) | CASPER提出了一种因果关系感知的方法来处理时空时间序列数据插补问题，避免过度利用非因果关系，提高数据分析的准确性。 |
| [^3] | [Statistical Efficiency of Distributional Temporal Difference](https://arxiv.org/abs/2403.05811) | 该论文分析了分布式时间差分的统计效率和有限样本性能。 |
| [^4] | [Predictors from causal features do not generalize better to new domains](https://arxiv.org/abs/2402.09891) | 因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。 |
| [^5] | [A general theory for robust clustering via trimmed mean.](http://arxiv.org/abs/2401.05574) | 本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。 |
| [^6] | [Stable generative modeling using diffusion maps.](http://arxiv.org/abs/2401.04372) | 本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。 |
| [^7] | [Coefficient Shape Alignment in Multivariate Functional Regression.](http://arxiv.org/abs/2312.01925) | 该论文提出了一种新的分组多元函数回归模型，其中采用了一种新的正则化方法来解决不同函数协变量的潜在同质性问题。 |
| [^8] | [Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs.](http://arxiv.org/abs/2310.10107) | 本文分析了后验采样学习算法在序列化POMDPs中的遗憾性能，并在一定条件下提供了改进的多项式贝叶斯遗憾界。 |
| [^9] | [Bayes optimal learning in high-dimensional linear regression with network side information.](http://arxiv.org/abs/2306.05679) | 本文首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题，引入了Reg-Graph模型并提出了基于AMP的迭代算法，在实验中优于现有的几种网络辅助回归方法。 |
| [^10] | [Utility Theory of Synthetic Data Generation.](http://arxiv.org/abs/2305.10015) | 本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。 |
| [^11] | [Transport map unadjusted Langevin algorithms: learning and discretizing perturbed samplers.](http://arxiv.org/abs/2302.07227) | 本研究提出了交通图未调整的 Langevin 算法 (ULA) 和 Riemann 流形 Langevin 动力学 (RMLD)，通过应用交通图可以加速 Langevin 动力学的收敛，并提供了学习度量和扰动的新思路。 |

# 详细

[^1]: 临床机器学习中多源交叉验证的实证研究

    Empirical investigation of multi-source cross-validation in clinical machine learning

    [https://arxiv.org/abs/2403.15012](https://arxiv.org/abs/2403.15012)

    本研究在多源环境中系统地评估了标准K折交叉验证和留出源交叉验证方法，为实现更全面和真实的精确度评估提供了新的机会

    

    传统上，基于机器学习的临床预测模型是在来自单一来源（如医院）的患者数据上进行训练和评估的。交叉验证方法可通过重复随机拆分数据来估计这些模型在来自同一来源的新患者上的精确度。然而，与部署模型到数据集中未代表的源头（如新医院）获得的精确度相比，这些估计往往过于乐观。多源医疗数据集的不断增加为通过基于源头的交叉验证设计获得更全面和真实的预期精确度评估提供了新机会。

    arXiv:2403.15012v1 Announce Type: new  Abstract: Traditionally, machine learning-based clinical prediction models have been trained and evaluated on patient data from a single source, such as a hospital. Cross-validation methods can be used to estimate the accuracy of such models on new patients originating from the same source, by repeated random splitting of the data. However, such estimates tend to be highly overoptimistic when compared to accuracy obtained from deploying models to sources not represented in the dataset, such as a new hospital. The increasing availability of multi-source medical datasets provides new opportunities for obtaining more comprehensive and realistic evaluations of expected accuracy through source-level cross-validation designs.   In this study, we present a systematic empirical evaluation of standard K-fold cross-validation and leave-source-out cross-validation methods in a multi-source setting. We consider the task of electrocardiogram based cardiovascul
    
[^2]: CASPER：因果关系感知时空图神经网络用于时空时间序列插补

    CASPER: Causality-Aware Spatiotemporal Graph Neural Networks for Spatiotemporal Time Series Imputation

    [https://arxiv.org/abs/2403.11960](https://arxiv.org/abs/2403.11960)

    CASPER提出了一种因果关系感知的方法来处理时空时间序列数据插补问题，避免过度利用非因果关系，提高数据分析的准确性。

    

    arXiv:2403.11960v1 公告类型：新 提要：时空时间序列是理解人类活动及其影响的基础，通常通过放置在不同位置的监测传感器收集。收集到的数据通常包含由于各种故障而导致的缺失值，这对数据分析有重要影响。为了填补缺失值，已经提出了许多方法。在恢复特定数据点时，大多数现有方法倾向于考虑与该点相关的所有信息，无论它们是否具有因果关系。在数据收集过程中，包括一些未知混杂因素是不可避免的，例如时间序列中的背景噪声和构建的传感器网络中的非因果快捷边。这些混杂因素可能在输入和输出之间开辟反向路径，换句话说，它们建立了输入和输出之间的非因果相关性。

    arXiv:2403.11960v1 Announce Type: new  Abstract: Spatiotemporal time series is the foundation of understanding human activities and their impacts, which is usually collected via monitoring sensors placed at different locations. The collected data usually contains missing values due to various failures, which have significant impact on data analysis. To impute the missing values, a lot of methods have been introduced. When recovering a specific data point, most existing methods tend to take into consideration all the information relevant to that point regardless of whether they have a cause-and-effect relationship. During data collection, it is inevitable that some unknown confounders are included, e.g., background noise in time series and non-causal shortcut edges in the constructed sensor network. These confounders could open backdoor paths between the input and output, in other words, they establish non-causal correlations between the input and output. Over-exploiting these non-causa
    
[^3]: 分布式时间差分的统计效率

    Statistical Efficiency of Distributional Temporal Difference

    [https://arxiv.org/abs/2403.05811](https://arxiv.org/abs/2403.05811)

    该论文分析了分布式时间差分的统计效率和有限样本性能。

    

    分布式强化学习(DRL)关注的是返回的完整分布，而不仅仅是均值，在各个领域取得了经验成功。领域DRL中的核心任务之一是分布式策略评估，涉及估计给定策略pi的返回分布η^pi。相应地提出了分布时间差分(TD)算法，这是经典RL文献中时间差分算法的延伸。在表格案例中，citet{rowland2018analysis}和citet{rowland2023analysis}分别证明了两个分布式TD实例即分类时间差分算法(CTD)和分位数时间差分算法(QTD)的渐近收敛。在这篇论文中，我们进一步分析了分布式TD的有限样本性能。为了促进理论分析，我们提出了一个非参数的 dis

    arXiv:2403.05811v1 Announce Type: cross  Abstract: Distributional reinforcement learning (DRL), which cares about the full distribution of returns instead of just the mean, has achieved empirical success in various domains. One of the core tasks in the field of DRL is distributional policy evaluation, which involves estimating the return distribution $\eta^\pi$ for a given policy $\pi$. A distributional temporal difference (TD) algorithm has been accordingly proposed, which is an extension of the temporal difference algorithm in the classic RL literature. In the tabular case, \citet{rowland2018analysis} and \citet{rowland2023analysis} proved the asymptotic convergence of two instances of distributional TD, namely categorical temporal difference algorithm (CTD) and quantile temporal difference algorithm (QTD), respectively. In this paper, we go a step further and analyze the finite-sample performance of distributional TD. To facilitate theoretical analysis, we propose non-parametric dis
    
[^4]: 预测因果特征不能更好地推广到新领域

    Predictors from causal features do not generalize better to new domains

    [https://arxiv.org/abs/2402.09891](https://arxiv.org/abs/2402.09891)

    因果特征不能更好地推广到新领域，预测器使用所有特征的效果更好。

    

    我们研究了在不同领域中，基于因果特征训练的机器学习模型的泛化效果。我们考虑了涵盖健康、就业、教育、社会福利和政治等应用的16个表格数据集的预测任务。每个数据集都有多个领域，我们可以测试一个在一个领域训练的模型在另一个领域的表现。对于每个预测任务，我们选择对预测目标有因果影响的特征。我们的目标是测试基于因果特征训练的模型是否在不同领域中更好地泛化。我们发现，无论是否具有因果关系，使用所有可用特征的预测器都比使用因果特征的预测器在领域内外的准确性更高。而且，即使是从一个领域到另一个领域的准确性绝对下降对于因果预测器来说也不比使用所有特征的模型更好。如果目标是在新领域中泛化，实践中使用所有特征的预测器效果更好。

    arXiv:2402.09891v1 Announce Type: new  Abstract: We study how well machine learning models trained on causal features generalize across domains. We consider 16 prediction tasks on tabular datasets covering applications in health, employment, education, social benefits, and politics. Each dataset comes with multiple domains, allowing us to test how well a model trained in one domain performs in another. For each prediction task, we select features that have a causal influence on the target of prediction. Our goal is to test the hypothesis that models trained on causal features generalize better across domains. Without exception, we find that predictors using all available features, regardless of causality, have better in-domain and out-of-domain accuracy than predictors using causal features. Moreover, even the absolute drop in accuracy from one domain to the other is no better for causal predictors than for models that use all features. If the goal is to generalize to new domains, prac
    
[^5]: 通过修剪均值的鲁棒聚类的一般理论

    A general theory for robust clustering via trimmed mean. (arXiv:2401.05574v1 [math.ST])

    [http://arxiv.org/abs/2401.05574](http://arxiv.org/abs/2401.05574)

    本文提出了一种通过使用修剪均值类型的中心点估计的混合聚类技术，用于在存在次高斯误差的中心点周围分布的弱初始化条件下产生最优错误标记保证，并且在存在敌对异常值的情况下仍然有效。

    

    在存在异质数据的统计机器学习中，聚类是一种基本工具。许多最近的结果主要关注在数据围绕带有次高斯误差的中心点分布时的最优错误标记保证。然而，限制性的次高斯模型在实践中常常无效，因为各种实际应用展示了围绕中心点的重尾分布或受到可能的敌对攻击，需要具有鲁棒数据驱动初始化的鲁棒聚类。在本文中，我们引入一种混合聚类技术，利用一种新颖的多变量修剪均值类型的中心点估计，在中心点周围的误差分布的弱初始化条件下产生错误标记保证。我们还给出了一个相匹配的下界，上界依赖于聚类的数量。此外，我们的方法即使在存在敌对异常值的情况下也能产生最优错误标记。我们的结果简化为亚高斯模型的情况。

    Clustering is a fundamental tool in statistical machine learning in the presence of heterogeneous data. Many recent results focus primarily on optimal mislabeling guarantees, when data are distributed around centroids with sub-Gaussian errors. Yet, the restrictive sub-Gaussian model is often invalid in practice, since various real-world applications exhibit heavy tail distributions around the centroids or suffer from possible adversarial attacks that call for robust clustering with a robust data-driven initialization. In this paper, we introduce a hybrid clustering technique with a novel multivariate trimmed mean type centroid estimate to produce mislabeling guarantees under a weak initialization condition for general error distributions around the centroids. A matching lower bound is derived, up to factors depending on the number of clusters. In addition, our approach also produces the optimal mislabeling even in the presence of adversarial outliers. Our results reduce to the sub-Gaus
    
[^6]: 使用扩散映射进行稳定的生成建模

    Stable generative modeling using diffusion maps. (arXiv:2401.04372v1 [stat.ML])

    [http://arxiv.org/abs/2401.04372](http://arxiv.org/abs/2401.04372)

    本文提出了一种稳定的生成建模方法，通过将扩散映射与朗之万动力学相结合，在仅有有限数量的训练样本的情况下生成新样本，并解决了时间步长僵硬随机微分方程中的稳定性问题。

    

    我们考虑从仅有足够数量的训练样本可得到的未知分布中抽样的问题。在生成建模的背景下，这样的设置最近引起了相当大的关注。本文中，我们提出了一种将扩散映射和朗之万动力学相结合的生成模型。扩散映射用于从可用的训练样本中近似得到漂移项，然后在离散时间的朗之万采样器中实现生成新样本。通过将核带宽设置为与未调整的朗之万算法中使用的时间步长匹配，我们的方法可以有效地避免通常与时间步长僵硬随机微分方程相关的稳定性问题。更准确地说，我们引入了一种新颖的分裂步骤方案，确保生成的样本保持在训练样本的凸包内。我们的框架可以自然地扩展为生成条件样本。我们展示了性能。

    We consider the problem of sampling from an unknown distribution for which only a sufficiently large number of training samples are available. Such settings have recently drawn considerable interest in the context of generative modelling. In this paper, we propose a generative model combining diffusion maps and Langevin dynamics. Diffusion maps are used to approximate the drift term from the available training samples, which is then implemented in a discrete-time Langevin sampler to generate new samples. By setting the kernel bandwidth to match the time step size used in the unadjusted Langevin algorithm, our method effectively circumvents any stability issues typically associated with time-stepping stiff stochastic differential equations. More precisely, we introduce a novel split-step scheme, ensuring that the generated samples remain within the convex hull of the training samples. Our framework can be naturally extended to generate conditional samples. We demonstrate the performance
    
[^7]: 在多元函数回归中的系数形状对齐

    Coefficient Shape Alignment in Multivariate Functional Regression. (arXiv:2312.01925v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2312.01925](http://arxiv.org/abs/2312.01925)

    该论文提出了一种新的分组多元函数回归模型，其中采用了一种新的正则化方法来解决不同函数协变量的潜在同质性问题。

    

    在多元函数数据分析中，不同的函数协变量可能具有同质性。隐藏的同质性结构对于不同协变量的连接或关联具有信息价值。具有明显同质性的协变量可以在同一群组中进行联合分析，从而产生一种简化建模多元函数数据的方法。本文提出了一种新颖的分组多元函数回归模型，采用称为“系数形状对齐”的新正则化方法来解决不同函数协变量的潜在同质性问题。建模过程包括两个主要步骤：首先，使用新的正则化方法检测未知分组结构，将协变量聚合到不相交的群组中；然后，基于检测到的分组结构建立分组多元函数回归模型。在这个新的分组模型中，同一同质群组中的系数函数应对齐。

    In multivariate functional data analysis, different functional covariates can be homogeneous. The hidden homogeneity structure is informative about the connectivity or association of different covariates. The covariates with pronounced homogeneity can be analyzed jointly within the same group, which gives rise to a way of parsimoniously modeling multivariate functional data. In this paper, a novel grouped multivariate functional regression model with a new regularization approach termed "coefficient shape alignment" is developed to tackle the potential homogeneity of different functional covariates. The modeling procedure includes two main steps: first detect the unknown grouping structure with the new regularization approach to aggregate covariates into disjoint groups; and then the grouped multivariate functional regression model is established based on the detected grouping structure. In this new grouped model, the coefficient functions of covariates in the same homogeneous group sh
    
[^8]: 后验采样学习算法在序列化POMDPs中的遗憾分析

    Regret Analysis of the Posterior Sampling-based Learning Algorithm for Episodic POMDPs. (arXiv:2310.10107v1 [cs.LG])

    [http://arxiv.org/abs/2310.10107](http://arxiv.org/abs/2310.10107)

    本文分析了后验采样学习算法在序列化POMDPs中的遗憾性能，并在一定条件下提供了改进的多项式贝叶斯遗憾界。

    

    相比于马尔科夫决策过程（MDPs），部分可观察马尔科夫决策过程（POMDPs）的学习由于观察数据难以解读而变得更加困难。在本文中，我们考虑了具有未知转移和观测模型的POMDPs中的序列化学习问题。我们考虑了基于后验采样的强化学习算法（PSRL）在POMDPs中的应用，并证明其贝叶斯遗憾随着序列的数量的平方根而缩小。一般来说，遗憾随着时间长度$H$呈指数级增长，并通过提供一个下界证明了这一点。然而，在POMDP是欠完备且弱可识别的条件下，我们建立了一个多项式贝叶斯遗憾界，相比于arXiv:2204.08967的最新结果，改进了遗憾界约$\Omega(H^2\sqrt{SA})$倍。

    Compared to Markov Decision Processes (MDPs), learning in Partially Observable Markov Decision Processes (POMDPs) can be significantly harder due to the difficulty of interpreting observations. In this paper, we consider episodic learning problems in POMDPs with unknown transition and observation models. We consider the Posterior Sampling-based Reinforcement Learning (PSRL) algorithm for POMDPs and show that its Bayesian regret scales as the square root of the number of episodes. In general, the regret scales exponentially with the horizon length $H$, and we show that this is inevitable by providing a lower bound. However, under the condition that the POMDP is undercomplete and weakly revealing, we establish a polynomial Bayesian regret bound that improves the regret bound by a factor of $\Omega(H^2\sqrt{SA})$ over the recent result by arXiv:2204.08967.
    
[^9]: 具有网络辅助信息的高维线性回归中的贝叶斯最优学习

    Bayes optimal learning in high-dimensional linear regression with network side information. (arXiv:2306.05679v1 [math.ST])

    [http://arxiv.org/abs/2306.05679](http://arxiv.org/abs/2306.05679)

    本文首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题，引入了Reg-Graph模型并提出了基于AMP的迭代算法，在实验中优于现有的几种网络辅助回归方法。

    

    在基因组学、蛋白质组学和神经科学等应用中，具有网络辅助信息的监督学习问题经常出现。本文中，我们首次研究了具有网络辅助信息的高维线性回归中的贝叶斯最优学习问题。为此，我们首先引入了一个简单的生成模型（称为Reg-Graph模型），通过一组共同的潜在参数为监督数据和观测到的网络设定了一个联合分布。接下来，我们介绍了一种基于近似消息传递（AMP）的迭代算法，在非常一般的条件下可证明是贝叶斯最优的。此外，我们对潜在信号和观测到的数据之间的极限互信息进行了表征，从而精确量化了网络辅助信息在回归问题中的统计影响。我们对模拟数据和实际数据的实验表明，我们的方法优于现有的几种网络辅助回归方法。

    Supervised learning problems with side information in the form of a network arise frequently in applications in genomics, proteomics and neuroscience. For example, in genetic applications, the network side information can accurately capture background biological information on the intricate relations among the relevant genes. In this paper, we initiate a study of Bayes optimal learning in high-dimensional linear regression with network side information. To this end, we first introduce a simple generative model (called the Reg-Graph model) which posits a joint distribution for the supervised data and the observed network through a common set of latent parameters. Next, we introduce an iterative algorithm based on Approximate Message Passing (AMP) which is provably Bayes optimal under very general conditions. In addition, we characterize the limiting mutual information between the latent signal and the data observed, and thus precisely quantify the statistical impact of the network side 
    
[^10]: 合成数据生成的效用理论

    Utility Theory of Synthetic Data Generation. (arXiv:2305.10015v1 [stat.ML])

    [http://arxiv.org/abs/2305.10015](http://arxiv.org/abs/2305.10015)

    本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。

    

    评估合成数据的效用对于衡量合成算法的有效性和效率至关重要。现有的结果侧重于对合成数据效用的经验评估，而针对合成数据算法如何影响效用的理论理解仍然未被充分探索。本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用。该指标定义为在合成和原始数据集上训练的模型之间泛化的绝对差异。我们建立了该效用指标的分析界限来研究指标收敛的关键条件。一个有趣的结果是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，则该效用指标会收敛。另一个重要的效用指标基于合成和原始数据之间潜在的因果机制一致性。该理论使用几种合成算法进行说明，并分析了它们的效用属性。

    Evaluating the utility of synthetic data is critical for measuring the effectiveness and efficiency of synthetic algorithms. Existing results focus on empirical evaluations of the utility of synthetic data, whereas the theoretical understanding of how utility is affected by synthetic data algorithms remains largely unexplored. This paper establishes utility theory from a statistical perspective, aiming to quantitatively assess the utility of synthetic algorithms based on a general metric. The metric is defined as the absolute difference in generalization between models trained on synthetic and original datasets. We establish analytical bounds for this utility metric to investigate critical conditions for the metric to converge. An intriguing result is that the synthetic feature distribution is not necessarily identical to the original one for the convergence of the utility metric as long as the model specification in downstream learning tasks is correct. Another important utility metri
    
[^11]: 交通图未调整的 Langevin 算法：学习和离散化扰动采样器

    Transport map unadjusted Langevin algorithms: learning and discretizing perturbed samplers. (arXiv:2302.07227v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.07227](http://arxiv.org/abs/2302.07227)

    本研究提出了交通图未调整的 Langevin 算法 (ULA) 和 Riemann 流形 Langevin 动力学 (RMLD)，通过应用交通图可以加速 Langevin 动力学的收敛，并提供了学习度量和扰动的新思路。

    

    Langevin 动力学被广泛用于抽样高维、非高斯分布，其密度已知但常数未知。特别感兴趣的是未修正的 Langevin 算法 (ULA)，它直接离散化 Langevin 动力学以估计目标分布上的期望。我们研究了使用交通图来近似标准化目标分布的方法，以预处理和加速 Langevin 动力学的收敛。我们展示了在连续时间下，当将交通图应用于 Langevin 动力学时，结果是具有由交通图定义的度量的 Riemann 流形 Langevin 动力学（RMLD）。我们还展示了将交通图应用于不可逆扰动的 ULA 会产生原动力学的几何信息不可逆扰动 （GiIrr）。这些联系表明了学习度量和扰动的更系统的方法，并提供了描述 RMLD 的替代离散化方法。

    Langevin dynamics are widely used in sampling high-dimensional, non-Gaussian distributions whose densities are known up to a normalizing constant. In particular, there is strong interest in unadjusted Langevin algorithms (ULA), which directly discretize Langevin dynamics to estimate expectations over the target distribution. We study the use of transport maps that approximately normalize a target distribution as a way to precondition and accelerate the convergence of Langevin dynamics. We show that in continuous time, when a transport map is applied to Langevin dynamics, the result is a Riemannian manifold Langevin dynamics (RMLD) with metric defined by the transport map. We also show that applying a transport map to an irreversibly-perturbed ULA results in a geometry-informed irreversible perturbation (GiIrr) of the original dynamics. These connections suggest more systematic ways of learning metrics and perturbations, and also yield alternative discretizations of the RMLD described b
    

