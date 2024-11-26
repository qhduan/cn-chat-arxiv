# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Estimating the roughness exponent of stochastic volatility from discrete observations of the realized variance.](http://arxiv.org/abs/2307.02582) | 本文提出了一种新的估计器，用于从离散观测中测量连续轨迹的粗糙指数，该估计器适用于随机波动模型的估计，并在大多数分数布朗运动样本路径中收敛。 |
| [^2] | [Multi-Modal Deep Learning for Credit Rating Prediction Using Text and Numerical Data Streams.](http://arxiv.org/abs/2304.10740) | 本文研究了基于多模态的深度学习融合技术在信用评级预测中的应用，通过比较不同融合策略和深度学习模型的组合，证明了一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术，同时在比较简单和复杂的模型中发现，更复杂的模型并不一定表现更好。 |
| [^3] | [A cross-border market model with limited transmission capacities.](http://arxiv.org/abs/2207.01939) | 本论文提出了一个跨境市场模型，该模型的传输容量是有限的，在该模型中，允许来自外国市场的委托与固定的国内市场数量相匹配，从而产生跨境交易。 |

# 详细

[^1]: 从离散实现方差的观测中估计随机波动的粗糙指数

    Estimating the roughness exponent of stochastic volatility from discrete observations of the realized variance. (arXiv:2307.02582v1 [q-fin.ST])

    [http://arxiv.org/abs/2307.02582](http://arxiv.org/abs/2307.02582)

    本文提出了一种新的估计器，用于从离散观测中测量连续轨迹的粗糙指数，该估计器适用于随机波动模型的估计，并在大多数分数布朗运动样本路径中收敛。

    

    本文考虑在随机波动模型中估计波动性的粗糙度，该模型是作为分数布朗运动（带漂移）的非线性函数而产生的。为此，我们引入一个新的估计量，该估计量测量连续轨迹的所谓粗糙指数，基于其原函数的离散观测。我们给出了对于基础轨迹的条件，在这些条件下，我们的估计器以严格路径方式收敛。然后我们验证了这些条件在几乎每个分数布朗运动（带漂移）样本路径中都得到满足。作为结果，在大类粗波动模型的背景下，我们得到了强一致性定理。数值模拟结果表明，在经过我们估计器的尺度不变修改后，我们的估计程序表现良好。

    We consider the problem of estimating the roughness of the volatility in a stochastic volatility model that arises as a nonlinear function of fractional Brownian motion with drift. To this end, we introduce a new estimator that measures the so-called roughness exponent of a continuous trajectory, based on discrete observations of its antiderivative. We provide conditions on the underlying trajectory under which our estimator converges in a strictly pathwise sense. Then we verify that these conditions are satisfied by almost every sample path of fractional Brownian motion (with drift). As a consequence, we obtain strong consistency theorems in the context of a large class of rough volatility models. Numerical simulations show that our estimation procedure performs well after passing to a scale-invariant modification of our estimator.
    
[^2]: 基于多模态深度学习的信用评级预测方法研究——以文本和数字数据流为例

    Multi-Modal Deep Learning for Credit Rating Prediction Using Text and Numerical Data Streams. (arXiv:2304.10740v1 [q-fin.GN])

    [http://arxiv.org/abs/2304.10740](http://arxiv.org/abs/2304.10740)

    本文研究了基于多模态的深度学习融合技术在信用评级预测中的应用，通过比较不同融合策略和深度学习模型的组合，证明了一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术，同时在比较简单和复杂的模型中发现，更复杂的模型并不一定表现更好。

    

    了解信用评级分配中哪些因素是重要的可以帮助做出更好的决策。然而，目前文献的重点大多集中在结构化数据上，较少研究非结构化或多模态数据集。本文提出了一种分析结构化和非结构化不同类型数据集的深度学习模型融合的有效架构，以预测公司信用评级标准。在模型中，我们测试了不同的深度学习模型及融合策略的组合，包括CNN，LSTM，GRU和BERT。我们研究了数据融合策略（包括早期和中间融合）以及技术（包括串联和交叉注意）等方面。结果表明，一个基于CNN的多模态模型通过两种融合策略优于其他多模态技术。此外，通过比较简单的架构与更复杂的架构，我们发现，更复杂的模型并不一定能在信用评级预测中发挥更好的性能。

    Knowing which factors are significant in credit rating assignment leads to better decision-making. However, the focus of the literature thus far has been mostly on structured data, and fewer studies have addressed unstructured or multi-modal datasets. In this paper, we present an analysis of the most effective architectures for the fusion of deep learning models for the prediction of company credit rating classes, by using structured and unstructured datasets of different types. In these models, we tested different combinations of fusion strategies with different deep learning models, including CNN, LSTM, GRU, and BERT. We studied data fusion strategies in terms of level (including early and intermediate fusion) and techniques (including concatenation and cross-attention). Our results show that a CNN-based multi-modal model with two fusion strategies outperformed other multi-modal techniques. In addition, by comparing simple architectures with more complex ones, we found that more soph
    
[^3]: 一种有限传输容量的跨境市场模型

    A cross-border market model with limited transmission capacities. (arXiv:2207.01939v3 [math.PR] UPDATED)

    [http://arxiv.org/abs/2207.01939](http://arxiv.org/abs/2207.01939)

    本论文提出了一个跨境市场模型，该模型的传输容量是有限的，在该模型中，允许来自外国市场的委托与固定的国内市场数量相匹配，从而产生跨境交易。

    

    我们开发了一个跨越两个国家的市场模型，在该模型中，连接不同国家的市场参与者的传输容量是有限的。从两个国家限价委托簿动态的简化形式出发，我们允许来自外国市场的委托与固定的法国市场数量相匹配，从而产生跨境交易。由于我们模型中的传输容量是有限的，所以我们的模型在具有跨境交易可能性和仅能将来自同一地方的市场订单匹配的模式之间交替切换。我们导出了我们微观模型的高频近似，假设单个订单的大小趋于零，订单到达率趋于无穷大。如果有传输容量，极限过程行为如下：正半轴内的四维线性布朗运动中的体积动态。

    We develop a cross-border market model between two countries in which the transmission capacities that enable transactions between market participants of different countries are limited. Starting from two reduced-form representations of national limit order book dynamics, we allow incoming market orders to be matched with standing volumes of the foreign market, resulting in cross-border trades. Since the transmission capacities in our model are limited, our model alternates between regimes in which cross-border trades are possible and regimes in which incoming market orders can only be matched against limit orders of the same origin. We derive a high-frequency approximation of our microscopic model, assuming that the size of an individual order converges to zero while the order arrival rate tends to infinity. If transmission capacities are available, the limit process behaves as follows: the volume dynamics is a four-dimensional linear Brownian motion in the positive orthant with obliq
    

