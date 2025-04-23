# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Boundary conditions at infinity for Black-Scholes equations.](http://arxiv.org/abs/2401.05549) | 提出了一种计算Markov局部鞅情况下远期合约价格的数值方法，通过有限网格和边界条件获得上下界，并且在无穷大时逼近准确值。 |
| [^2] | [The Quadratic Local Variance Gamma Model: an arbitrage-free interpolation of class $\mathcal{C}^3$ for option prices.](http://arxiv.org/abs/2305.13791) | 本研究推广了局部波动伽马模型，使用分段二次局部波动函数实现了$\mathcal{C}^3$级别的无套利类插值，并能够在插值时减少节点数量来降低计算成本。 |
| [^3] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |
| [^4] | [Pitman's Theorem, Black-Scholes Equation, and Derivative Pricing for Fundraisers.](http://arxiv.org/abs/2303.13956) | 本文提出了一种包括筹款人和普通投资者的金融市场模型，并通过Pitman定理导出了筹款人的欧式期权价格，并使用数值方案计算了具有泡沫市场中的看涨期权价格。 |

# 详细

[^1]: 无穷远处的边界条件对于Black-Scholes方程的影响

    Boundary conditions at infinity for Black-Scholes equations. (arXiv:2401.05549v1 [q-fin.MF])

    [http://arxiv.org/abs/2401.05549](http://arxiv.org/abs/2401.05549)

    提出了一种计算Markov局部鞅情况下远期合约价格的数值方法，通过有限网格和边界条件获得上下界，并且在无穷大时逼近准确值。

    

    我们提出了一种用于计算基础资产价格为Markov局部鞅的远期合约价格的数值方法。如果基础过程是严格的局部鞅，相应的Black-Scholes方程存在多个解，并且衍生品价格被表征为最小解。我们使用数值方法在有限网格上根据相应的边界条件获得上下界。随着基础价格趋于无穷大，这些界和边界值逼近准确值。我们通过数值测试验证了所提出的方法。

    We propose numerical procedures for computing the prices of forward contracts where the underlying asset price is a Markovian local martingale. If the underlying process is a strict local martingale, multiple solutions exist for the corresponding Black-Scholes equations, and the derivative prices are characterized as the minimal solutions. Our prices are upper and lower bounds obtained using numerical methods on a finite grid under the respective boundary conditions. These bounds and the boundary values converge to the exact value as the underlying price approaches infinity. The proposed procedures are demonstrated through numerical tests.
    
[^2]: 二次局部波动伽马模型：用于期权价格的无套利类$\mathcal{C}^3$插值的推广

    The Quadratic Local Variance Gamma Model: an arbitrage-free interpolation of class $\mathcal{C}^3$ for option prices. (arXiv:2305.13791v1 [q-fin.CP])

    [http://arxiv.org/abs/2305.13791](http://arxiv.org/abs/2305.13791)

    本研究推广了局部波动伽马模型，使用分段二次局部波动函数实现了$\mathcal{C}^3$级别的无套利类插值，并能够在插值时减少节点数量来降低计算成本。

    

    本文将Carr和Nadtochiy的局部波动伽马模型推广到分段二次局部波动函数。该公式包括分段线性Bachelier和分段线性Black局部波动伽马模型。二次局部波动函数导致了$\mathcal{C}^3$级别的无套利类插值。相较于分段常数和分段线性表示，增加的平滑度在插值原始市场报价时允许减少节点数量，从而提供了一个有趣的正则化替代方案同时降低计算成本。

    This paper generalizes the local variance gamma model of Carr and Nadtochiy, to a piecewise quadratic local variance function. The formulation encompasses the piecewise linear Bachelier and piecewise linear Black local variance gamma models. The quadratic local variance function results in an arbitrage-free interpolation of class $\mathcal{C}^3$. The increased smoothness over the piecewise-constant and piecewise-linear representation allows to reduce the number of knots when interpolating raw market quotes, thus providing an interesting alternative to regularization while reducing the computational cost.
    
[^3]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    
[^4]: Pitman定理，Black-Scholes方程和筹款的衍生品定价

    Pitman's Theorem, Black-Scholes Equation, and Derivative Pricing for Fundraisers. (arXiv:2303.13956v1 [q-fin.MF])

    [http://arxiv.org/abs/2303.13956](http://arxiv.org/abs/2303.13956)

    本文提出了一种包括筹款人和普通投资者的金融市场模型，并通过Pitman定理导出了筹款人的欧式期权价格，并使用数值方案计算了具有泡沫市场中的看涨期权价格。

    

    我们提出了一个金融市场模型，包括储蓄账户和股票，其中股票价格过程建模为一维扩散。在这个模型中存在两种类型的交易者：普通投资者和筹款人。尽管投资者只能观测到扩散自然滤波，但筹款人拥有额外的筹款信息，并因为筹款而获得额外的现金流。这个概念被应用到了Pitman定理的三维Bessel过程。我们提出了两个贡献：第一，导出了筹款人的欧式期权价格。第二，提出了一种数值方案来计算具有泡沫市场中的看涨期权价格，其中Black-Scholes方程存在多个解，衍生品价格被确定为最小的非负超级解。更准确地说，这种市场中的看涨期权价格被从下面近似。

    We propose a financial market model that comprises a savings account and a stock, where the stock price process is modeled as a one-dimensional diffusion, wherein two types of agents exist: an ordinary investor and a fundraiser who buys or sells stocks as funding activities. Although the investor information is the natural filtration of the diffusion, the fundraiser possesses extra information regarding the funding, as well as additional cash flows as a result of the funding. This concept is modeled using Pitman's theorem for the three-dimensional Bessel process. Two contributions are presented: First, the prices of European options for the fundraiser are derived. Second, a numerical scheme is proposed for call option prices in a market with a bubble, where multiple solutions exist for the Black-Scholes equation and the derivative prices are characterized as the smallest nonnegative supersolution. More precisely, the call option price in such a market is approximated from below by the 
    

