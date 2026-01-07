# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Bayesian inference for the generalized linear mixed model](https://arxiv.org/abs/2403.03007) | 该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。 |
| [^2] | [A Method For Bounding Tail Probabilities](https://arxiv.org/abs/2402.13662) | 提出了一种界定连续随机变量右尾和左尾概率上下界的方法，通过设置特定的函数，得到了新的上下界限，并与马尔可夫不等式建立了联系 |
| [^3] | [Meta-learning the mirror map in policy mirror descent](https://arxiv.org/abs/2402.05187) | 该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。 |
| [^4] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |
| [^5] | [Bayesian score calibration for approximate models.](http://arxiv.org/abs/2211.05357) | 本文提出了一种用于减小偏差和产生更准确不确定性量化的近似后验调整方法，通过优化近似后验的变换来最大化得分规则。这种方法只需要进行少量复杂模型模拟，且具有数值稳定性。 |

# 详细

[^1]: 通用线性混合模型的可扩展贝叶斯推断

    Scalable Bayesian inference for the generalized linear mixed model

    [https://arxiv.org/abs/2403.03007](https://arxiv.org/abs/2403.03007)

    该论文提出了一种针对通用线性混合模型的可扩展贝叶斯推断算法，解决了在大数据环境中进行统计推断时的计算难题。

    

    通用线性混合模型（GLMM）是处理相关数据的一种流行统计方法，在包括生物医学数据等大数据常见的应用领域被广泛使用。本文的重点是针对GLMM的可扩展统计推断，我们将统计推断定义为：（i）对总体参数的估计以及（ii）在存在不确定性的情况下评估科学假设。人工智能（AI）学习算法擅长可扩展的统计估计，但很少包括不确定性量化。相比之下，贝叶斯推断提供完整的统计推断，因为不确定性量化自动来自后验分布。不幸的是，包括马尔可夫链蒙特卡洛（MCMC）在内的贝叶斯推断算法在大数据环境中变得难以计算。在本文中，我们介绍了一个统计推断算法

    arXiv:2403.03007v1 Announce Type: cross  Abstract: The generalized linear mixed model (GLMM) is a popular statistical approach for handling correlated data, and is used extensively in applications areas where big data is common, including biomedical data settings. The focus of this paper is scalable statistical inference for the GLMM, where we define statistical inference as: (i) estimation of population parameters, and (ii) evaluation of scientific hypotheses in the presence of uncertainty. Artificial intelligence (AI) learning algorithms excel at scalable statistical estimation, but rarely include uncertainty quantification. In contrast, Bayesian inference provides full statistical inference, since uncertainty quantification results automatically from the posterior distribution. Unfortunately, Bayesian inference algorithms, including Markov Chain Monte Carlo (MCMC), become computationally intractable in big data settings. In this paper, we introduce a statistical inference algorithm 
    
[^2]: 一种界定尾部概率的方法

    A Method For Bounding Tail Probabilities

    [https://arxiv.org/abs/2402.13662](https://arxiv.org/abs/2402.13662)

    提出了一种界定连续随机变量右尾和左尾概率上下界的方法，通过设置特定的函数，得到了新的上下界限，并与马尔可夫不等式建立了联系

    

    我们提出了一种方法，用于上下界定连续随机变量（RVs）的右尾和左尾概率。对于具有概率密度函数$f_X(x)$的RV $X$的右尾概率，该方法首先要求设置一个连续的、正的、严格递减的函数$g_X(x)$，使得$-f_X(x)/g'_X(x)$是一个递减且递增的函数，$\forall x>x_0$，分别给出形式为$-f_X(x) g_X(x)/g'_X(x)$的上界和下界，$\forall x>x_0$，其中$x_0$是某个点。类似地，对于$X$的左尾概率的上下界，该方法首先要求设置一个连续的、正的、严格递增的函数$g_X(x)$，使得$f_X(x)/g'_X(x)$是一个增加且递减的函数，$\forall x<x_0$。我们提供了一些函数$g_X(x)$的良好候选示例。我们还建立了新界限与马尔可夫不等式的联系。

    arXiv:2402.13662v1 Announce Type: cross  Abstract: We present a method for upper and lower bounding the right and the left tail probabilities of continuous random variables (RVs). For the right tail probability of RV $X$ with probability density function $f_X(x)$, this method requires first setting a continuous, positive, and strictly decreasing function $g_X(x)$ such that $-f_X(x)/g'_X(x)$ is a decreasing and increasing function, $\forall x>x_0$, which results in upper and lower bounds, respectively, given in the form $-f_X(x) g_X(x)/g'_X(x)$, $\forall x>x_0$, where $x_0$ is some point. Similarly, for the upper and lower bounds on the left tail probability of $X$, this method requires first setting a continuous, positive, and strictly increasing function $g_X(x)$ such that $f_X(x)/g'_X(x)$ is an increasing and decreasing function, $\forall x<x_0$. We provide some examples of good candidates for the function $g_X(x)$. We also establish connections between the new bounds and Markov's in
    
[^3]: 在策略镜像下降中元学习镜像映射

    Meta-learning the mirror map in policy mirror descent

    [https://arxiv.org/abs/2402.05187](https://arxiv.org/abs/2402.05187)

    该论文通过实证研究发现，传统的镜像映射选择（NPG）在标准基准环境中常常导致不理想的结果。通过元学习方法，找到了更高效的镜像映射，提升了性能。

    

    策略镜像下降（PMD）是强化学习中的一种流行框架，作为一种统一视角，它包含了许多算法。这些算法是通过选择一个镜像映射而导出的，并且具有有限时间的收敛保证。尽管它很受欢迎，但对PMD的全面潜力的探索是有限的，大部分研究集中在一个特定的镜像映射上，即负熵，从而产生了著名的自然策略梯度（NPG）方法。目前的理论研究还不确定镜像映射的选择是否会对PMD的有效性产生重大影响。在我们的工作中，我们进行了实证研究，证明了传统的镜像映射选择（NPG）在几个标准基准环境中经常产生不理想的结果。通过应用元学习方法，我们确定了更高效的镜像映射，提高了性能，无论是平均性能还是最佳性能。

    Policy Mirror Descent (PMD) is a popular framework in reinforcement learning, serving as a unifying perspective that encompasses numerous algorithms. These algorithms are derived through the selection of a mirror map and enjoy finite-time convergence guarantees. Despite its popularity, the exploration of PMD's full potential is limited, with the majority of research focusing on a particular mirror map -- namely, the negative entropy -- which gives rise to the renowned Natural Policy Gradient (NPG) method. It remains uncertain from existing theoretical studies whether the choice of mirror map significantly influences PMD's efficacy. In our work, we conduct empirical investigations to show that the conventional mirror map choice (NPG) often yields less-than-optimal outcomes across several standard benchmark environments. By applying a meta-learning approach, we identify more efficient mirror maps that enhance performance, both on average and in terms of best performance achieved along th
    
[^4]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    
[^5]: 适用于近似模型的贝叶斯得分校准

    Bayesian score calibration for approximate models. (arXiv:2211.05357v4 [stat.CO] UPDATED)

    [http://arxiv.org/abs/2211.05357](http://arxiv.org/abs/2211.05357)

    本文提出了一种用于减小偏差和产生更准确不确定性量化的近似后验调整方法，通过优化近似后验的变换来最大化得分规则。这种方法只需要进行少量复杂模型模拟，且具有数值稳定性。

    

    科学家们不断发展越来越复杂的机械模型，以更真实地反映他们的知识。使用这些模型进行统计推断可能具有挑战性，因为相应的似然函数通常难以处理，并且模型模拟可能带来计算负担。幸运的是，在许多情况下，可以采用替代模型或近似似然函数。直接使用替代似然函数进行贝叶斯推断可能很方便，但可能导致偏差和不准确的不确定性量化。在本文中，我们提出了一种新的方法，通过优化近似后验的变换来最大化得分规则，从而减小偏差并产生更准确的不确定性量化。我们的方法只需要进行（固定的）少量复杂模型模拟，且具有数值稳定性。我们在几个不断增加的示例上展示了新方法的良好性能。

    Scientists continue to develop increasingly complex mechanistic models to reflect their knowledge more realistically. Statistical inference using these models can be challenging since the corresponding likelihood function is often intractable and model simulation may be computationally burdensome. Fortunately, in many of these situations, it is possible to adopt a surrogate model or approximate likelihood function. It may be convenient to conduct Bayesian inference directly with the surrogate, but this can result in bias and poor uncertainty quantification. In this paper we propose a new method for adjusting approximate posterior samples to reduce bias and produce more accurate uncertainty quantification. We do this by optimizing a transform of the approximate posterior that maximizes a scoring rule. Our approach requires only a (fixed) small number of complex model simulations and is numerically stable. We demonstrate good performance of the new method on several examples of increasin
    

