# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting).](http://arxiv.org/abs/2305.08857) | 本文提出了一种新的比例代表方法，COWPEA，可根据候选人不同的权重进行最优选举，并可转换为分数或分级投票方法。 |
| [^2] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |
| [^3] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |

# 详细

[^1]: COWPEA（候选人按比例使用赞成投票进行最优加权）：一种新的比例代表方法

    COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting). (arXiv:2305.08857v1 [econ.TH])

    [http://arxiv.org/abs/2305.08857](http://arxiv.org/abs/2305.08857)

    本文提出了一种新的比例代表方法，COWPEA，可根据候选人不同的权重进行最优选举，并可转换为分数或分级投票方法。

    

    本文描述了一种使用赞成投票的比例代表新方法，称为COWPEA（候选人按比例使用赞成投票进行最优加权）。COWPEA在选择一定数量的候选人时，可以根据其不同的权重进行最优选举，而不是只给予固定数量相同的权重。COWPEA Lottery 是一个不确定性的版本，可以选择一定数量的候选人，并使它们拥有相等的权重。COWPEA是唯一已知通过单调性、与无关选票和普遍喜欢的候选人标准的比例方法。同时，也有方法可以将COWPEA和COWPEA Lottery转换为分数或分级投票方法。

    This paper describes a new method of proportional representation that uses approval voting, known as COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting). COWPEA optimally elects an unlimited number of candidates with potentially different weights to a body, rather than giving a fixed number equal weight. A version that elects a fixed a number of candidates with equal weight does exist, but it is non-deterministic, and is known as COWPEA Lottery. This is the only proportional method known to pass monotonicity, Independence of Irrelevant Ballots, and the Universally Liked Candidate criterion. There are also ways to convert COWPEA and COWPEA Lottery to a score or graded voting method.
    
[^2]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    
[^3]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    

