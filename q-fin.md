# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Asset-Liability Management.](http://arxiv.org/abs/2310.00553) | 本研究提出了一种鲁棒的资产负债管理方法，通过无模型的债券组合选择方法来对冲利率风险，同时兼顾任意的负债结构、组合约束和利率波动。数值评估结果表明，该方法相对于现有方法具有可行性和准确性。 |
| [^2] | [Machine learning for option pricing: an empirical investigation of network architectures.](http://arxiv.org/abs/2307.07657) | 广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。 |
| [^3] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |

# 详细

[^1]: 鲁棒的资产负债管理

    Robust Asset-Liability Management. (arXiv:2310.00553v1 [q-fin.RM])

    [http://arxiv.org/abs/2310.00553](http://arxiv.org/abs/2310.00553)

    本研究提出了一种鲁棒的资产负债管理方法，通过无模型的债券组合选择方法来对冲利率风险，同时兼顾任意的负债结构、组合约束和利率波动。数值评估结果表明，该方法相对于现有方法具有可行性和准确性。

    

    当金融机构拥有长期资产和负债时，他们应该如何对冲利率风险？我们从泛函和数值分析的角度出发，提出了一种无模型的债券组合选择方法，该方法推广了经典的免疫化，并适应了任意的负债结构、组合约束和利率波动。我们证明了存在一个最大化最坏情况下股权价值的免疫化组合，并提供了一个解决算法。使用来自无套利期限结构模型的经验和模拟收益曲线进行数值评估，支持我们的方法相对于现有方法的可行性和准确性。

    How should financial institutions hedge their balance sheets against interest rate risk when they have long-term assets and liabilities? Using the perspective of functional and numerical analysis, we propose a model-free bond portfolio selection method that generalizes classical immunization and accommodates arbitrary liability structure, portfolio constraints, and perturbations in interest rates. We prove the generic existence of an immunizing portfolio that maximizes the worst-case equity with a tight error estimate and provide a solution algorithm. Numerical evaluations using empirical and simulated yield curves from a no-arbitrage term structure model support the feasibility and accuracy of our approach relative to existing methods.
    
[^2]: 机器学习用于期权定价：对网络结构的实证研究

    Machine learning for option pricing: an empirical investigation of network architectures. (arXiv:2307.07657v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.07657](http://arxiv.org/abs/2307.07657)

    广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。

    

    本文考虑了使用适当的输入数据（模型参数）和相应输出数据（期权价格或隐含波动率）来学习期权价格或隐含波动率的监督学习问题。大部分相关文献都使用（普通的）前馈神经网络结构来连接用于学习将输入映射到输出的神经元。在本文中，受到图像分类方法和用于偏微分方程机器学习方法的最新进展的启发，我们通过实证研究来探究网络结构的选择如何影响机器学习算法的精确度和训练时间。我们发现，在期权定价问题中，我们主要关注Black-Scholes和Heston模型，广义高速公路网络结构相较于其他变体在均方误差和训练时间方面表现更好。此外，在计算隐含波动率方面，

    We consider the supervised learning problem of learning the price of an option or the implied volatility given appropriate input data (model parameters) and corresponding output data (option prices or implied volatilities). The majority of articles in this literature considers a (plain) feed forward neural network architecture in order to connect the neurons used for learning the function mapping inputs to outputs. In this article, motivated by methods in image classification and recent advances in machine learning methods for PDEs, we investigate empirically whether and how the choice of network architecture affects the accuracy and training time of a machine learning algorithm. We find that for option pricing problems, where we focus on the Black--Scholes and the Heston model, the generalized highway network architecture outperforms all other variants, when considering the mean squared error and the training time as criteria. Moreover, for the computation of the implied volatility, a
    
[^3]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    

