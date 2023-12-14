# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-task learning of convex combinations of forecasting models.](http://arxiv.org/abs/2310.20545) | 本文提出了一种多任务学习方法，通过深度神经网络同时解决了预测模型选择和凸组合权重学习的问题。通过回归分支学习权重和分类分支选择具有多样性的预测方法，提高了基于特征的预测的精确度。 |
| [^2] | [On the Stability of Iterative Retraining of Generative Models on their own Data.](http://arxiv.org/abs/2310.00429) | 本文研究了生成模型在混合数据集上训练对稳定性的影响，通过证明初始生成模型足够接近数据分布并且数据比例适当，迭代训练是稳定的。 |
| [^3] | [On Estimating the Gradient of the Expected Information Gain in Bayesian Experimental Design.](http://arxiv.org/abs/2308.09888) | 该论文提出了一种估计贝叶斯实验设计中期望信息增益梯度的方法，通过结合随机梯度下降算法，实现了高效优化。具体而言，通过后验期望表示来估计与设计变量相关的梯度，并提出了UEEG-MCMC和BEEG-AP两种估计方法。这些方法在不同的实验设计问题上都表现出良好的性能。 |
| [^4] | [Discretization-Induced Dirichlet Posterior for Robust Uncertainty Quantification on Regression.](http://arxiv.org/abs/2308.09065) | 本文提出了一种用于回归任务的广义AuxUE方案，目的是实现更鲁棒的不确定性量化。具体而言，该方案通过考虑不同的分布假设，选择Laplace分布来近似p，以实现更鲁棒的本质不确定性估计。 |
| [^5] | [Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques.](http://arxiv.org/abs/2307.12971) | 本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。 |
| [^6] | [Fit Like You Sample: Sample-Efficient Generalized Score Matching from Fast Mixing Markov Chains.](http://arxiv.org/abs/2306.09332) | 本文提出了一种从快速混合马尔可夫链中实现样本高效的广义得分匹配方法，解决了得分匹配算法在具有较差等周性质的分布上的统计代价问题。 |
| [^7] | [The Blessing of Heterogeneity in Federated Q-learning: Linear Speedup and Beyond.](http://arxiv.org/abs/2305.10697) | 本文提出了异构群体强化学习中联邦Q学习的样本复杂度保证，讨论了同步和异步版本的线性加速，同时探究了等权重平均本地Q估计的缺陷。 |
| [^8] | [Mixed moving average field guided learning for spatio-temporal data.](http://arxiv.org/abs/2301.00736) | 本论文提出了一种理论引导机器学习方法，采用广义贝叶斯算法进行混合移动平均场引导的时空数据建模，可以进行因果未来预测。 |
| [^9] | [On the fast convergence of minibatch heavy ball momentum.](http://arxiv.org/abs/2206.07553) | 本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。 |

# 详细

[^1]: 多任务学习凸组合预测模型

    Multi-task learning of convex combinations of forecasting models. (arXiv:2310.20545v1 [cs.LG])

    [http://arxiv.org/abs/2310.20545](http://arxiv.org/abs/2310.20545)

    本文提出了一种多任务学习方法，通过深度神经网络同时解决了预测模型选择和凸组合权重学习的问题。通过回归分支学习权重和分类分支选择具有多样性的预测方法，提高了基于特征的预测的精确度。

    

    预测组合涉及使用多个预测来创建单一、更精确的预测。最近，基于特征的预测已被用于选择最合适的预测模型或学习它们的凸组合权重。在本文中，我们提出了一种同时解决这两个问题的多任务学习方法。该方法通过深度神经网络实现，其中包括两个分支：回归分支通过最小化组合预测误差来学习各种预测方法的权重，分类分支则重点选择多样性的预测方法。为了为分类任务生成训练标签，我们引入了一种优化驱动的方法，用于确定给定时间序列的最合适的方法。所提出的方法揭示了基于特征的预测中多样性的重要作用，并凸显了模型组合和选择之间的相互作用。

    Forecast combination involves using multiple forecasts to create a single, more accurate prediction. Recently, feature-based forecasting has been employed to either select the most appropriate forecasting models or to learn the weights of their convex combination. In this paper, we present a multi-task learning methodology that simultaneously addresses both problems. This approach is implemented through a deep neural network with two branches: the regression branch, which learns the weights of various forecasting methods by minimizing the error of combined forecasts, and the classification branch, which selects forecasting methods with an emphasis on their diversity. To generate training labels for the classification task, we introduce an optimization-driven approach that identifies the most appropriate methods for a given time series. The proposed approach elicits the essential role of diversity in feature-based forecasting and highlights the interplay between model combination and mo
    
[^2]: 关于生成模型在其自己的数据上迭代训练的稳定性研究

    On the Stability of Iterative Retraining of Generative Models on their own Data. (arXiv:2310.00429v1 [cs.LG])

    [http://arxiv.org/abs/2310.00429](http://arxiv.org/abs/2310.00429)

    本文研究了生成模型在混合数据集上训练对稳定性的影响，通过证明初始生成模型足够接近数据分布并且数据比例适当，迭代训练是稳定的。

    

    深度生成模型在建模复杂数据方面取得了巨大的进展，往往展现出超过典型人类能力的样本真实性辨别能力。这一成功的关键驱动力无疑是这些模型消耗海量网络规模数据的结果。由于这些模型惊人的性能和易得性，网络上将不可避免地出现越来越多的合成内容。这个事实直接意味着生成模型的未来迭代必须面对一个现实：它们的训练数据由清洁数据和先前模型生成的人工数据组成。在本文中，我们开发了一个框架来对混合数据集（包括真实数据和合成数据）上训练生成模型对稳定性的影响进行严格研究。我们首先证明了在初始生成模型足够好地近似数据分布并且真实数据与合成数据的比例适当的情况下，迭代训练的稳定性。

    Deep generative models have made tremendous progress in modeling complex data, often exhibiting generation quality that surpasses a typical human's ability to discern the authenticity of samples. Undeniably, a key driver of this success is enabled by the massive amounts of web-scale data consumed by these models. Due to these models' striking performance and ease of availability, the web will inevitably be increasingly populated with synthetic content. Such a fact directly implies that future iterations of generative models must contend with the reality that their training is curated from both clean data and artificially generated data from past models. In this paper, we develop a framework to rigorously study the impact of training generative models on mixed datasets (of real and synthetic data) on their stability. We first prove the stability of iterative training under the condition that the initial generative models approximate the data distribution well enough and the proportion o
    
[^3]: 关于贝叶斯实验设计中期望信息增益梯度的估计

    On Estimating the Gradient of the Expected Information Gain in Bayesian Experimental Design. (arXiv:2308.09888v1 [stat.ML])

    [http://arxiv.org/abs/2308.09888](http://arxiv.org/abs/2308.09888)

    该论文提出了一种估计贝叶斯实验设计中期望信息增益梯度的方法，通过结合随机梯度下降算法，实现了高效优化。具体而言，通过后验期望表示来估计与设计变量相关的梯度，并提出了UEEG-MCMC和BEEG-AP两种估计方法。这些方法在不同的实验设计问题上都表现出良好的性能。

    

    贝叶斯实验设计旨在找到贝叶斯推断的最佳实验条件，通常被描述为优化期望信息增益（EIG）。为了高效地优化EIG，往往需要梯度信息，因此估计EIG的梯度能力对于贝叶斯实验设计问题至关重要。该工作的主要目标是开发估计EIG梯度的方法，结合随机梯度下降算法，实现EIG的高效优化。具体而言，我们首先介绍了与设计变量相关的EIG梯度的后验期望表示。基于此，我们提出了两种估计EIG梯度的方法，UEEG-MCMC利用通过马尔科夫链蒙特卡洛（MCMC）生成的后验样本来估计EIG梯度，而BEEG-AP则专注于通过反复使用参数样本来实现高模拟效率。理论分析和数值实验表明，我们的方法在不同的实验设计问题上都能获得较好的性能。

    Bayesian Experimental Design (BED), which aims to find the optimal experimental conditions for Bayesian inference, is usually posed as to optimize the expected information gain (EIG). The gradient information is often needed for efficient EIG optimization, and as a result the ability to estimate the gradient of EIG is essential for BED problems. The primary goal of this work is to develop methods for estimating the gradient of EIG, which, combined with the stochastic gradient descent algorithms, result in efficient optimization of EIG. Specifically, we first introduce a posterior expected representation of the EIG gradient with respect to the design variables. Based on this, we propose two methods for estimating the EIG gradient, UEEG-MCMC that leverages posterior samples generated through Markov Chain Monte Carlo (MCMC) to estimate the EIG gradient, and BEEG-AP that focuses on achieving high simulation efficiency by repeatedly using parameter samples. Theoretical analysis and numerica
    
[^4]: 通过离散化引发的Dirichlet后验用于回归问题的鲁棒性不确定性量化

    Discretization-Induced Dirichlet Posterior for Robust Uncertainty Quantification on Regression. (arXiv:2308.09065v1 [cs.CV])

    [http://arxiv.org/abs/2308.09065](http://arxiv.org/abs/2308.09065)

    本文提出了一种用于回归任务的广义AuxUE方案，目的是实现更鲁棒的不确定性量化。具体而言，该方案通过考虑不同的分布假设，选择Laplace分布来近似p，以实现更鲁棒的本质不确定性估计。

    

    在实际应用中，不确定性量化对于部署深度神经网络（DNNs）至关重要。辅助不确定性估计器（AuxUE）是一种在不修改主任务模型的情况下估计主任务预测不确定性的最有效手段之一。为了被认为是鲁棒的，AuxUE必须能够在遇到超出分布范围的输入时保持性能并引发更高的不确定性，即提供鲁棒的本质不确定性和认识不确定性。然而，对于视觉回归任务，当前的AuxUE设计主要用于本质不确定性估计，并且尚未探索AuxUE的鲁棒性。在这项工作中，我们提出了一种用于回归任务的更鲁棒不确定性量化的广义AuxUE方案。具体而言，为了实现更鲁棒的本质不确定性估计，在异方差噪声方面考虑了不同的分布假设，并最终选择Laplace分布来近似p

    Uncertainty quantification is critical for deploying deep neural networks (DNNs) in real-world applications. An Auxiliary Uncertainty Estimator (AuxUE) is one of the most effective means to estimate the uncertainty of the main task prediction without modifying the main task model. To be considered robust, an AuxUE must be capable of maintaining its performance and triggering higher uncertainties while encountering Out-of-Distribution (OOD) inputs, i.e., to provide robust aleatoric and epistemic uncertainty. However, for vision regression tasks, current AuxUE designs are mainly adopted for aleatoric uncertainty estimates, and AuxUE robustness has not been explored. In this work, we propose a generalized AuxUE scheme for more robust uncertainty quantification on regression tasks. Concretely, to achieve a more robust aleatoric uncertainty estimation, different distribution assumptions are considered for heteroscedastic noise, and Laplace distribution is finally chosen to approximate the p
    
[^5]: 大数据-供应链管理框架的预测：数据预处理和机器学习技术

    Big Data - Supply Chain Management Framework for Forecasting: Data Preprocessing and Machine Learning Techniques. (arXiv:2307.12971v1 [cs.LG])

    [http://arxiv.org/abs/2307.12971](http://arxiv.org/abs/2307.12971)

    本文介绍了一种新的大数据-供应链管理框架，通过数据预处理和机器学习技术实现供应链预测，优化操作管理、透明度，并讨论了幻影库存对预测的不利影响。

    

    本文旨在系统地识别和比较分析最先进的供应链预测策略和技术。提出了一个新的框架，将大数据分析应用于供应链管理中，包括问题识别、数据来源、探索性数据分析、机器学习模型训练、超参数调优、性能评估和优化，以及预测对人力、库存和整个供应链的影响。首先讨论了根据供应链策略收集数据的需求以及如何收集数据。文章讨论了根据周期或供应链目标需要不同类型的预测。推荐使用供应链绩效指标和误差测量系统来优化表现最佳的模型。还讨论了幻影库存对预测的不利影响以及管理决策依赖供应链绩效指标来确定模型性能参数和改进运营管理、透明度的问题。

    This article intends to systematically identify and comparatively analyze state-of-the-art supply chain (SC) forecasting strategies and technologies. A novel framework has been proposed incorporating Big Data Analytics in SC Management (problem identification, data sources, exploratory data analysis, machine-learning model training, hyperparameter tuning, performance evaluation, and optimization), forecasting effects on human-workforce, inventory, and overall SC. Initially, the need to collect data according to SC strategy and how to collect them has been discussed. The article discusses the need for different types of forecasting according to the period or SC objective. The SC KPIs and the error-measurement systems have been recommended to optimize the top-performing model. The adverse effects of phantom inventory on forecasting and the dependence of managerial decisions on the SC KPIs for determining model performance parameters and improving operations management, transparency, and 
    
[^6]: Fit Like You Sample: 从快速混合马尔可夫链中实现样本高效的广义得分匹配

    Fit Like You Sample: Sample-Efficient Generalized Score Matching from Fast Mixing Markov Chains. (arXiv:2306.09332v1 [cs.DS])

    [http://arxiv.org/abs/2306.09332](http://arxiv.org/abs/2306.09332)

    本文提出了一种从快速混合马尔可夫链中实现样本高效的广义得分匹配方法，解决了得分匹配算法在具有较差等周性质的分布上的统计代价问题。

    

    得分匹配是一种学习概率分布的方法，其参数化为比例常数（例如，能量基模型）。其思想是拟合分布的得分，而不是似然函数，从而避免评估比例常数的需求。虽然这具有明显的算法优势，但统计代价可能很高：Koehler等人的最新工作表明，对于具有较差等周性质（较大的Poincare或对数Sobolev常数）的分布，得分匹配的统计效率明显低于极大似然估计。然而，许多自然实际的分布，例如一维中的两个高斯分布混合物等多峰分布，具有较差的Poincaré常数。在本文中，我们展示了任意马尔可夫过程的混合时间与试图拟合$\frac{\mathcal{O} p}{p}$的广义得分匹配损失之间的密切关系。如果$\mathcal{L}$的特征向量不依赖于$p$，我们展示了一种基于随机梯度下降的算法，从而实现的样本高效广义得分匹配。

    Score matching is an approach to learning probability distributions parametrized up to a constant of proportionality (e.g. Energy-Based Models). The idea is to fit the score of the distribution, rather than the likelihood, thus avoiding the need to evaluate the constant of proportionality. While there's a clear algorithmic benefit, the statistical "cost'' can be steep: recent work by Koehler et al. 2022 showed that for distributions that have poor isoperimetric properties (a large Poincar\'e or log-Sobolev constant), score matching is substantially statistically less efficient than maximum likelihood. However, many natural realistic distributions, e.g. multimodal distributions as simple as a mixture of two Gaussians in one dimension -- have a poor Poincar\'e constant.  In this paper, we show a close connection between the mixing time of an arbitrary Markov process with generator $\mathcal{L}$ and a generalized score matching loss that tries to fit $\frac{\mathcal{O} p}{p}$. If $\mathca
    
[^7]: 异构群体强化学习中的福音：线性加速和更多可能

    The Blessing of Heterogeneity in Federated Q-learning: Linear Speedup and Beyond. (arXiv:2305.10697v1 [cs.LG])

    [http://arxiv.org/abs/2305.10697](http://arxiv.org/abs/2305.10697)

    本文提出了异构群体强化学习中联邦Q学习的样本复杂度保证，讨论了同步和异步版本的线性加速，同时探究了等权重平均本地Q估计的缺陷。

    

    当强化学习（RL）的数据由多个代理以分布式方式收集时，联邦RL算法允许协作学习，无需共享本地数据。本文考虑联邦Q学习，其目的是通过定期聚合仅在本地数据上训练的本地Q估计来学习最优Q函数。针对无限时间蒸馏标记决策过程，我们为同步和异步版本的联邦Q学习提供了样本复杂度保证。在两种情况下，我们的界限展示了与代理数量成线性加速以及其他显著问题参数的更尖锐的依赖关系。此外，现有的联邦Q学习方法采用等权重平均本地Q估计，这在异步设置中可能会高度次优，因为由于不同的本地行为策略，本地轨迹可能高度异构。现有的样本最优化策略在异步设置中存在巨大缺陷。

    When the data used for reinforcement learning (RL) are collected by multiple agents in a distributed manner, federated versions of RL algorithms allow collaborative learning without the need of sharing local data. In this paper, we consider federated Q-learning, which aims to learn an optimal Q-function by periodically aggregating local Q-estimates trained on local data alone. Focusing on infinite-horizon tabular Markov decision processes, we provide sample complexity guarantees for both the synchronous and asynchronous variants of federated Q-learning. In both cases, our bounds exhibit a linear speedup with respect to the number of agents and sharper dependencies on other salient problem parameters. Moreover, existing approaches to federated Q-learning adopt an equally-weighted averaging of local Q-estimates, which can be highly sub-optimal in the asynchronous setting since the local trajectories can be highly heterogeneous due to different local behavior policies. Existing sample com
    
[^8]: 混合移动平均场引导的时空数据学习

    Mixed moving average field guided learning for spatio-temporal data. (arXiv:2301.00736v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.00736](http://arxiv.org/abs/2301.00736)

    本论文提出了一种理论引导机器学习方法，采用广义贝叶斯算法进行混合移动平均场引导的时空数据建模，可以进行因果未来预测。

    

    受到混合移动平均场的影响，时空数据的建模是一个多功能的技巧。但是，它们的预测分布通常不可访问。在这个建模假设下，我们定义了一种新的理论引导机器学习方法，采用广义贝叶斯算法进行预测。我们采用Lipschitz预测器（例如线性模型或前馈神经网络），并通过最小化沿空间和时间维度串行相关的数据的新型PAC贝叶斯界限来确定一个随机估计值。进行因果未来预测是我们方法的一个亮点，因为它适用于具有短期和长期相关性的数据。最后，我们通过展示线性预测器和模拟STOU过程的时空数据的示例来展示学习方法的性能。

    Influenced mixed moving average fields are a versatile modeling class for spatio-temporal data. However, their predictive distribution is not generally accessible. Under this modeling assumption, we define a novel theory-guided machine learning approach that employs a generalized Bayesian algorithm to make predictions. We employ a Lipschitz predictor, for example, a linear model or a feed-forward neural network, and determine a randomized estimator by minimizing a novel PAC Bayesian bound for data serially correlated along a spatial and temporal dimension. Performing causal future predictions is a highlight of our methodology as its potential application to data with short and long-range dependence. We conclude by showing the performance of the learning methodology in an example with linear predictors and simulated spatio-temporal data from an STOU process.
    
[^9]: 论小批量重球动量法的快速收敛性

    On the fast convergence of minibatch heavy ball momentum. (arXiv:2206.07553v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.07553](http://arxiv.org/abs/2206.07553)

    本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。

    

    简单的随机动量方法被广泛用于机器学习优化中，但由于还没有加速的理论保证，这与它们在实践中的良好性能并不相符。本文旨在通过展示，随机重球动量在二次最优化问题中保持（确定性）重球动量的快速线性率，至少在使用足够大的批量大小进行小批量处理时。我们所研究的算法可以被解释为带小批量处理和重球动量的加速随机Kaczmarz算法。该分析依赖于仔细分解动量转移矩阵，并使用新的独立随机矩阵乘积的谱范围集中界限。我们提供了数值演示，证明了我们的界限相当尖锐。

    Simple stochastic momentum methods are widely used in machine learning optimization, but their good practical performance is at odds with an absence of theoretical guarantees of acceleration in the literature. In this work, we aim to close the gap between theory and practice by showing that stochastic heavy ball momentum retains the fast linear rate of (deterministic) heavy ball momentum on quadratic optimization problems, at least when minibatching with a sufficiently large batch size. The algorithm we study can be interpreted as an accelerated randomized Kaczmarz algorithm with minibatching and heavy ball momentum. The analysis relies on carefully decomposing the momentum transition matrix, and using new spectral norm concentration bounds for products of independent random matrices. We provide numerical illustrations demonstrating that our bounds are reasonably sharp.
    

