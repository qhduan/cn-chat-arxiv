# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Transformer approach for Electricity Price Forecasting](https://arxiv.org/abs/2403.16108) | 这种独特的Transformer模型在电力价格预测中取得了更好的表现，为可靠和可持续的电力系统运行提供了有前景的解决方案。 |
| [^2] | [FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models](https://arxiv.org/abs/2403.09904) | FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。 |
| [^3] | [PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation](https://arxiv.org/abs/2402.04355) | PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。 |

# 详细

[^1]: 一种用于电力价格预测的Transformer方法

    A Transformer approach for Electricity Price Forecasting

    [https://arxiv.org/abs/2403.16108](https://arxiv.org/abs/2403.16108)

    这种独特的Transformer模型在电力价格预测中取得了更好的表现，为可靠和可持续的电力系统运行提供了有前景的解决方案。

    

    本文提出了一种使用纯Transformer模型进行电力价格预测（EPF）的新方法。与其他方法不同，没有使用其他递归网络结合注意力机制。因此，表明注意力层足以捕捉时间模式。该论文还通过使用开源EPF工具进行了对模型的公平比较，并提供了代码以增强EPF研究的可再现性和透明度。结果表明，Transformer模型优于传统方法，为可靠和可持续的电力系统运行提供了一种有希望的解决方案。

    arXiv:2403.16108v1 Announce Type: cross  Abstract: This paper presents a novel approach to electricity price forecasting (EPF) using a pure Transformer model. As opposed to other alternatives, no other recurrent network is used in combination to the attention mechanism. Hence, showing that the attention layer is enough for capturing the temporal patterns. The paper also provides fair comparison of the models using the open-source EPF toolbox and provide the code to enhance reproducibility and transparency in EPF research. The results show that the Transformer model outperforms traditional methods, offering a promising solution for reliable and sustainable power system operation.
    
[^2]: FedComLoc: 稀疏和量化模型的通信高效分布式训练

    FedComLoc: Communication-Efficient Distributed Training of Sparse and Quantized Models

    [https://arxiv.org/abs/2403.09904](https://arxiv.org/abs/2403.09904)

    FedComLoc利用Scaffnew算法的基础，引入了压缩和本地训练，显著降低了分布式训练中的通信开销。

    

    联邦学习（FL）由于其允许异构客户端在本地处理其私有数据并与中央服务器互动，同时尊重隐私的独特特点而受到越来越多的关注。我们的工作受到了创新的Scaffnew算法的启发，该算法在FL中大大推动了通信复杂性的降低。我们引入了FedComLoc（联邦压缩和本地训练），将实用且有效的压缩集成到Scaffnew中，以进一步增强通信效率。广泛的实验证明，使用流行的TopK压缩器和量化，它在大幅减少异构中的通信开销方面具有卓越的性能。

    arXiv:2403.09904v1 Announce Type: cross  Abstract: Federated Learning (FL) has garnered increasing attention due to its unique characteristic of allowing heterogeneous clients to process their private data locally and interact with a central server, while being respectful of privacy. A critical bottleneck in FL is the communication cost. A pivotal strategy to mitigate this burden is \emph{Local Training}, which involves running multiple local stochastic gradient descent iterations between communication phases. Our work is inspired by the innovative \emph{Scaffnew} algorithm, which has considerably advanced the reduction of communication complexity in FL. We introduce FedComLoc (Federated Compressed and Local Training), integrating practical and effective compression into \emph{Scaffnew} to further enhance communication efficiency. Extensive experiments, using the popular TopK compressor and quantization, demonstrate its prowess in substantially reducing communication overheads in heter
    
[^3]: PQMass: 使用概率质量估计的生成模型质量的概率评估

    PQMass: Probabilistic Assessment of the Quality of Generative Models using Probability Mass Estimation

    [https://arxiv.org/abs/2402.04355](https://arxiv.org/abs/2402.04355)

    PQMass是一种使用概率质量估计来评估生成模型质量的全面方法，能够直接处理高维数据，不依赖于假设或训练其他模型。

    

    我们提出了一种全面的基于样本的方法来评估生成模型的质量。所提出的方法能够估计两个样本集合来自同一分布的概率，为评估单个生成模型的性能或比较在同一数据集上训练的多个竞争模型提供了一个统计上严格的方法。该比较可以通过将空间划分为非重叠的区域并比较每个区域中的数据样本数量来进行。该方法仅需要生成模型和测试数据的样本。它能够直接处理高维数据，无需降维。显著的是，该方法不依赖于关于真实分布密度的假设，并且不依赖于训练或拟合任何辅助模型。相反，它着重于近似计算密度的积分（概率质量）。

    We propose a comprehensive sample-based method for assessing the quality of generative models. The proposed approach enables the estimation of the probability that two sets of samples are drawn from the same distribution, providing a statistically rigorous method for assessing the performance of a single generative model or the comparison of multiple competing models trained on the same dataset. This comparison can be conducted by dividing the space into non-overlapping regions and comparing the number of data samples in each region. The method only requires samples from the generative model and the test data. It is capable of functioning directly on high-dimensional data, obviating the need for dimensionality reduction. Significantly, the proposed method does not depend on assumptions regarding the density of the true distribution, and it does not rely on training or fitting any auxiliary models. Instead, it focuses on approximating the integral of the density (probability mass) acros
    

