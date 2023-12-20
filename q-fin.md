# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Futures Quantitative Investment with Heterogeneous Continual Graph Neural Network.](http://arxiv.org/abs/2303.16532) | 为了预测期货价格趋势，本文提出了一种基于异构任务设计和连续训练的时空图神经网络模型，可以捕捉长期和短期特征。 |

# 详细

[^1]: 异构时空图神经网络在期货量化投资中的应用研究

    Futures Quantitative Investment with Heterogeneous Continual Graph Neural Network. (arXiv:2303.16532v1 [cs.LG])

    [http://arxiv.org/abs/2303.16532](http://arxiv.org/abs/2303.16532)

    为了预测期货价格趋势，本文提出了一种基于异构任务设计和连续训练的时空图神经网络模型，可以捕捉长期和短期特征。

    

    传统计量模型预测期货价格趋势是一个具有挑战性的问题，因为需要考虑到期货历史数据以及不同期货之间的关联。时空图神经网络在处理此类空间时间数据方面具有很大的优势。本研究通过设计四个异构任务来捕捉长期和短期特征：价格回归、移动平均价格回归、短时间内的价格差回归和变化点检测。为了充分利用这些标签，我们采用连续训练的方式对模型进行训练。

    It is a challenging problem to predict trends of futures prices with traditional econometric models as one needs to consider not only futures' historical data but also correlations among different futures. Spatial-temporal graph neural networks (STGNNs) have great advantages in dealing with such kind of spatial-temporal data. However, we cannot directly apply STGNNs to high-frequency future data because future investors have to consider both the long-term and short-term characteristics when doing decision-making. To capture both the long-term and short-term features, we exploit more label information by designing four heterogeneous tasks: price regression, price moving average regression, price gap regression (within a short interval), and change-point detection, which involve both long-term and short-term scenes. To make full use of these labels, we train our model in a continual manner. Traditional continual GNNs define the gradient of prices as the parameter important to overcome ca
    

