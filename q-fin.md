# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A solution to walrasian auctions for many tokens with AMMs available.](http://arxiv.org/abs/2310.12255) | 本研究提出了适用于多代币的Walrasian拍卖问题的解决方案，该方案基于Brouwer的不动点定理，能够执行所有订单并进行最优的AMMs交换。 |
| [^2] | [Deep Stock: training and trading scheme using deep learning.](http://arxiv.org/abs/2304.14870) | 本文提出了一种使用深度学习进行训练和交易的方案，DeepStock通过查看股票价格的过去数据，并使用Resnet和logits来预测股票价格在未来D天内是否会升降一定百分比，并在韩国和美国市场上取得了超过市场回报的利润。 |

# 详细

[^1]: 适用于多代币的Walrasian拍卖问题的解决方案与AMMs

    A solution to walrasian auctions for many tokens with AMMs available. (arXiv:2310.12255v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.12255](http://arxiv.org/abs/2310.12255)

    本研究提出了适用于多代币的Walrasian拍卖问题的解决方案，该方案基于Brouwer的不动点定理，能够执行所有订单并进行最优的AMMs交换。

    

    考虑某一状态下有限数量的交易订单和自动市场制造商（AMMs）。我们提出了一个解决方案，用于找到一个均衡价格向量，以便与相应的最优AMMs交换一起执行所有订单。该解决方案基于Brouwer的不动点定理。我们讨论了与公共区块链活动中的实际情况相关的计算方面问题。

    Consider a finite set of trade orders and automated market makers (AMMs) at some state. We propose a solution to the problem of finding an equilibrium price vector to execute all the orders jointly with corresponding optimal AMMs swaps. The solution is based on Brouwer's fixed-point theorem. We discuss computational aspects relevant for realistic situations in public blockchain activity.
    
[^2]: Deep Stock: 使用深度学习进行训练和交易的方案

    Deep Stock: training and trading scheme using deep learning. (arXiv:2304.14870v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.14870](http://arxiv.org/abs/2304.14870)

    本文提出了一种使用深度学习进行训练和交易的方案，DeepStock通过查看股票价格的过去数据，并使用Resnet和logits来预测股票价格在未来D天内是否会升降一定百分比，并在韩国和美国市场上取得了超过市场回报的利润。

    

    尽管有效市场假说存在，但许多研究表明股票市场存在失灵现象，导致出现了一些能够获得超过市场回报的技术，即alpha。近几十年来，系统性交易已经取得了重大进展，深度学习作为分析和预测市场行为的强大工具已经开始崭露头角。本文中，我们提出了一种受专业交易员启发的模型，该模型查看先前的600天的股票价格，并预测股票价格在接下来D天内是否会升降一定百分比。我们的模型称为DeepStock，使用Resnet的跳跃连接和logits来增加模型在交易方案中的概率。我们在韩国和美国股票市场上测试了我们的模型，并在韩国市场上获得了N％的利润，超过市场回报M％，并在美国市场上获得了A％的利润，超过市场回报B％。

    Despite the efficient market hypothesis, many studies suggest the existence of inefficiencies in the stock market, leading to the development of techniques to gain above-market returns, known as alpha. Systematic trading has undergone significant advances in recent decades, with deep learning emerging as a powerful tool for analyzing and predicting market behavior. In this paper, we propose a model inspired by professional traders that look at stock prices of the previous 600 days and predicts whether the stock price rises or falls by a certain percentage within the next D days. Our model, called DeepStock, uses Resnet's skip connections and logits to increase the probability of a model in a trading scheme. We test our model on both the Korean and US stock markets and achieve a profit of N\% on Korea market, which is M\% above the market return, and profit of A\% on US market, which is B\% above the market return.
    

