# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data.](http://arxiv.org/abs/2401.15610) | 本论文提出了一种预验证的岭回归模型，该模型在高维数据中与逻辑回归非常接近，但具有更高的计算效率和几乎没有超参数。它通过利用在拟合过程中计算得到的数量来缩放模型系数，并最小化一组预验证预测的对数损失。 |
| [^2] | [Utility Theory of Synthetic Data Generation.](http://arxiv.org/abs/2305.10015) | 本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。 |
| [^3] | [Convex optimization over a probability simplex.](http://arxiv.org/abs/2305.09046) | 这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。 |
| [^4] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |

# 详细

[^1]: 预验证的岭回归是高维数据中逻辑回归的高效替代方法

    Prevalidated ridge regression is a highly-efficient drop-in replacement for logistic regression for high-dimensional data. (arXiv:2401.15610v1 [cs.LG])

    [http://arxiv.org/abs/2401.15610](http://arxiv.org/abs/2401.15610)

    本论文提出了一种预验证的岭回归模型，该模型在高维数据中与逻辑回归非常接近，但具有更高的计算效率和几乎没有超参数。它通过利用在拟合过程中计算得到的数量来缩放模型系数，并最小化一组预验证预测的对数损失。

    

    逻辑回归是一种常见的概率分类方法。然而，逻辑回归的有效性取决于仔细且相对计算密集的调优，尤其是对于正则化超参数，并且尤其在高维数据的背景下。我们提出了一种预验证的岭回归模型，该模型在分类错误和对数损失方面与逻辑回归非常接近，特别适用于高维数据，同时在计算效率上明显更高，并且除了正则化之外没有超参数。我们通过缩放模型的系数来最小化由估计的留一交叉验证误差推导出的一组预验证预测的对数损失。这利用了在拟合岭回归模型过程中已经计算的数量，以找到具有名义附加计算开销的缩放参数。

    Logistic regression is a ubiquitous method for probabilistic classification. However, the effectiveness of logistic regression depends upon careful and relatively computationally expensive tuning, especially for the regularisation hyperparameter, and especially in the context of high-dimensional data. We present a prevalidated ridge regression model that closely matches logistic regression in terms of classification error and log-loss, particularly for high-dimensional data, while being significantly more computationally efficient and having effectively no hyperparameters beyond regularisation. We scale the coefficients of the model so as to minimise log-loss for a set of prevalidated predictions derived from the estimated leave-one-out cross-validation error. This exploits quantities already computed in the course of fitting the ridge regression model in order to find the scaling parameter with nominal additional computational expense.
    
[^2]: 合成数据生成的效用理论

    Utility Theory of Synthetic Data Generation. (arXiv:2305.10015v1 [stat.ML])

    [http://arxiv.org/abs/2305.10015](http://arxiv.org/abs/2305.10015)

    本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用，效用指标的分析界限揭示了指标收敛的关键条件，令人惊讶的是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，效用指标会收敛。

    

    评估合成数据的效用对于衡量合成算法的有效性和效率至关重要。现有的结果侧重于对合成数据效用的经验评估，而针对合成数据算法如何影响效用的理论理解仍然未被充分探索。本文从统计学角度建立效用理论，旨在基于一般性指标定量评估合成算法的效用。该指标定义为在合成和原始数据集上训练的模型之间泛化的绝对差异。我们建立了该效用指标的分析界限来研究指标收敛的关键条件。一个有趣的结果是，只要下游学习任务中的模型规范是正确的，合成特征分布不一定与原始特征分布相同，则该效用指标会收敛。另一个重要的效用指标基于合成和原始数据之间潜在的因果机制一致性。该理论使用几种合成算法进行说明，并分析了它们的效用属性。

    Evaluating the utility of synthetic data is critical for measuring the effectiveness and efficiency of synthetic algorithms. Existing results focus on empirical evaluations of the utility of synthetic data, whereas the theoretical understanding of how utility is affected by synthetic data algorithms remains largely unexplored. This paper establishes utility theory from a statistical perspective, aiming to quantitatively assess the utility of synthetic algorithms based on a general metric. The metric is defined as the absolute difference in generalization between models trained on synthetic and original datasets. We establish analytical bounds for this utility metric to investigate critical conditions for the metric to converge. An intriguing result is that the synthetic feature distribution is not necessarily identical to the original one for the convergence of the utility metric as long as the model specification in downstream learning tasks is correct. Another important utility metri
    
[^3]: 概率单纯形上的凸优化

    Convex optimization over a probability simplex. (arXiv:2305.09046v1 [math.OC])

    [http://arxiv.org/abs/2305.09046](http://arxiv.org/abs/2305.09046)

    这篇论文提出了一种新的迭代方案，用于求解概率单纯形上的凸优化问题。该方法具有收敛速度快且简单易行的特点。

    

    我们提出了一种新的迭代方案——柯西单纯形来优化凸问题，使其满足概率单纯形上的限制条件，即$w\in\mathbb{R}^n$中$\sum_i w_i=1$，$w_i\geq0$。我们将单纯形映射到单位球的正四面体，通过梯度下降获得隐变量的解，并将结果映射回原始变量。该方法适用于高维问题，每次迭代由简单的操作组成，且针对凸函数证明了收敛速度为${O}(1/T)$。同时本文关注了信息理论（如交叉熵和KL散度）的应用。

    We propose a new iteration scheme, the Cauchy-Simplex, to optimize convex problems over the probability simplex $\{w\in\mathbb{R}^n\ |\ \sum_i w_i=1\ \textrm{and}\ w_i\geq0\}$. Other works have taken steps to enforce positivity or unit normalization automatically but never simultaneously within a unified setting. This paper presents a natural framework for manifestly requiring the probability condition. Specifically, we map the simplex to the positive quadrant of a unit sphere, envisage gradient descent in latent variables, and map the result back in a way that only depends on the simplex variable. Moreover, proving rigorous convergence results in this formulation leads inherently to tools from information theory (e.g. cross entropy and KL divergence). Each iteration of the Cauchy-Simplex consists of simple operations, making it well-suited for high-dimensional problems. We prove that it has a convergence rate of ${O}(1/T)$ for convex functions, and numerical experiments of projection 
    
[^4]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    

