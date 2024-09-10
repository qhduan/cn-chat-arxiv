# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rolling Diffusion Models](https://arxiv.org/abs/2402.09470) | 本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。 |
| [^2] | [Online Graph Topology Learning from Matrix-valued Time Series](https://arxiv.org/abs/2107.08020) | 本文通过研究矩阵值时间序列的统计分析，提出了在线图拓扑学习的方法。首先，将VAR模型扩展为矩阵变量模型以适用于图形学习。其次，提出了两种在线过程，针对低维和高维情况快速更新系数的估计。这些方法在高维情况下引入了一种新的Lasso-type进行拓扑处理。 |
| [^3] | [CA-PCA: Manifold Dimension Estimation, Adapted for Curvature.](http://arxiv.org/abs/2309.13478) | 本文提出了CA-PCA算法，它基于曲率校准的局部PCA版本，通过考虑底层流形的曲率，改进了维度估计器的性能。 |
| [^4] | [Estimation and inference for the Wasserstein distance between mixing measures in topic models.](http://arxiv.org/abs/2206.12768) | 本文提出了对混合模型中混合测度的Wasserstein距离的新的规范解释，并提供了在主题模型中进行此距离推断的工具。 |
| [^5] | [Graph Neural Network Sensitivity Under Probabilistic Error Model.](http://arxiv.org/abs/2203.07831) | 本文研究了概率误差模型对图卷积网络（GCN）性能的影响，并证明了误差模型下邻接矩阵的受限性。通过实验验证了这种误差界限，并研究了GCN在这种概率误差模型下的准确性敏感性。 |

# 详细

[^1]: 滚动扩散模型

    Rolling Diffusion Models

    [https://arxiv.org/abs/2402.09470](https://arxiv.org/abs/2402.09470)

    本文介绍了一种滚动扩散模型，用于处理时间数据，通过滑动窗口去噪并根据帧在序列中的时间先后分配不同的噪声量，更好地捕捉到复杂的时间动态。通过实验证明，在视频预测和混沌流体动力学预测任务中，该模型优于传统扩散方法。

    

    最近，扩散模型越来越多地应用于时间数据，如视频、流体力学模拟或气候数据。这些方法通常将后续帧在扩散过程中的噪声量视为相等。本文探讨了滚动扩散：一种使用滑动窗口去噪的新方法。它确保扩散过程逐渐通过时间进行破坏，通过将更多的噪声分配给序列中出现较晚的帧，反映出随着生成过程的展开，对未来的不确定性越来越大。通过实证研究，我们表明当时间动态复杂时，滚动扩散优于标准扩散。特别是在使用Kinetics-600视频数据集进行视频预测任务和混沌流体动力学预测实验中证明了这一结果。

    arXiv:2402.09470v1 Announce Type: new  Abstract: Diffusion models have recently been increasingly applied to temporal data such as video, fluid mechanics simulations, or climate data. These methods generally treat subsequent frames equally regarding the amount of noise in the diffusion process. This paper explores Rolling Diffusion: a new approach that uses a sliding window denoising process. It ensures that the diffusion process progressively corrupts through time by assigning more noise to frames that appear later in a sequence, reflecting greater uncertainty about the future as the generation process unfolds. Empirically, we show that when the temporal dynamics are complex, Rolling Diffusion is superior to standard diffusion. In particular, this result is demonstrated in a video prediction task using the Kinetics-600 video dataset and in a chaotic fluid dynamics forecasting experiment.
    
[^2]: 基于矩阵值时间序列的在线图拓扑学习

    Online Graph Topology Learning from Matrix-valued Time Series

    [https://arxiv.org/abs/2107.08020](https://arxiv.org/abs/2107.08020)

    本文通过研究矩阵值时间序列的统计分析，提出了在线图拓扑学习的方法。首先，将VAR模型扩展为矩阵变量模型以适用于图形学习。其次，提出了两种在线过程，针对低维和高维情况快速更新系数的估计。这些方法在高维情况下引入了一种新的Lasso-type进行拓扑处理。

    

    本文研究了矩阵值时间序列的统计分析。这些数据是在一个传感器网络上收集的（通常是一组空间位置），观测到每个传感器的每个时间点的特征向量。因此，每个传感器由一个向量时序列来描述。我们希望识别这些传感器之间的依赖结构，并用图形来表示它。当每个传感器只有一个特征时，矢量自回归模型已被广泛应用于推断格兰杰因果关系的结构。所得到的图被称为因果图。我们的第一个贡献是将VAR模型扩展为矩阵变量模型，以用于图形学习的目的。其次，我们提出了两种在线过程，分别适用于低维和高维情况，在新样本到达时可以快速更新系数的估计。特别是在高维情况下，引入了一种新的Lasso-type，并对其进行了拓扑处理。

    This paper is concerned with the statistical analysis of matrix-valued time series. These are data collected over a network of sensors (typically a set of spatial locations) along time, where a vector of features is observed per time instant per sensor. Thus each sensor is characterized by a vectorial time series. We would like to identify the dependency structure among these sensors and represent it by a graph. When there is only one feature per sensor, the vector auto-regressive models have been widely adapted to infer the structure of Granger causality. The resulting graph is referred to as causal graph. Our first contribution is then extending VAR models to matrix-variate models to serve the purpose of graph learning. Secondly, we propose two online procedures respectively in low and high dimensions, which can update quickly the estimates of coefficients when new samples arrive. In particular in high dimensional regime, a novel Lasso-type is introduced and we develop its homotopy a
    
[^3]: CA-PCA: 测量曲率的流形维度估计

    CA-PCA: Manifold Dimension Estimation, Adapted for Curvature. (arXiv:2309.13478v1 [stat.ML])

    [http://arxiv.org/abs/2309.13478](http://arxiv.org/abs/2309.13478)

    本文提出了CA-PCA算法，它基于曲率校准的局部PCA版本，通过考虑底层流形的曲率，改进了维度估计器的性能。

    

    高维数据分析算法的成功常归因于流形假设，即假设数据分布在或接近低维流形上。在进行维度约简之前，确定或估计该流形的维度通常是有用的。现有的维度估计方法使用平坦单位球进行校准。本文提出了CA-PCA，一种基于二次嵌入校准的局部PCA版本，以考虑底层流形的曲率。大量的精心实验表明，这种适应性改进了估计器在各种设置下的性能。

    The success of algorithms in the analysis of high-dimensional data is often attributed to the manifold hypothesis, which supposes that this data lie on or near a manifold of much lower dimension. It is often useful to determine or estimate the dimension of this manifold before performing dimension reduction, for instance. Existing methods for dimension estimation are calibrated using a flat unit ball. In this paper, we develop CA-PCA, a version of local PCA based instead on a calibration of a quadratic embedding, acknowledging the curvature of the underlying manifold. Numerous careful experiments show that this adaptation improves the estimator in a wide range of settings.
    
[^4]: 主题模型中混合测度的Wasserstein距离的估计和推断

    Estimation and inference for the Wasserstein distance between mixing measures in topic models. (arXiv:2206.12768v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2206.12768](http://arxiv.org/abs/2206.12768)

    本文提出了对混合模型中混合测度的Wasserstein距离的新的规范解释，并提供了在主题模型中进行此距离推断的工具。

    

    在混合模型的统计分析中，混合测度的Wasserstein距离已经成为了一个核心问题。本研究提出了这种距离的新的规范解释，并提供了在主题模型中进行混合测度的Wasserstein距离推断的工具。我们考虑了一般可识别混合模型的情况，其中包括了多个来自集合$\mathcal{A}$内带有任意度量$d$的分布的混合，我们证明了混合测度的Wasserstein距离是唯一地表征出混合元素集合$\mathcal{A}$上度量$d$的最有区分性的凸扩展。虽然Wasserstein距离在混合模型的研究中已被广泛使用，但缺乏公理证明。我们的结果确立了这个度量作为一个规范选择。特准化这个度量到主题模型，我们考虑了这个距离的估计和推断。虽然$i$

    The Wasserstein distance between mixing measures has come to occupy a central place in the statistical analysis of mixture models. This work proposes a new canonical interpretation of this distance and provides tools to perform inference on the Wasserstein distance between mixing measures in topic models.  We consider the general setting of an identifiable mixture model consisting of mixtures of distributions from a set $\mathcal{A}$ equipped with an arbitrary metric $d$, and show that the Wasserstein distance between mixing measures is uniquely characterized as the most discriminative convex extension of the metric $d$ to the set of mixtures of elements of $\mathcal{A}$. The Wasserstein distance between mixing measures has been widely used in the study of such models, but without axiomatic justification. Our results establish this metric to be a canonical choice.  Specializing our results to topic models, we consider estimation and inference of this distance. Though upper bounds for i
    
[^5]: 图形神经网络在概率误差模型下的敏感性

    Graph Neural Network Sensitivity Under Probabilistic Error Model. (arXiv:2203.07831v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.07831](http://arxiv.org/abs/2203.07831)

    本文研究了概率误差模型对图卷积网络（GCN）性能的影响，并证明了误差模型下邻接矩阵的受限性。通过实验验证了这种误差界限，并研究了GCN在这种概率误差模型下的准确性敏感性。

    

    图卷积网络（GCN）可以通过图卷积成功学习图信号表示。图卷积依赖于图滤波器，其中包含数据的拓扑依赖关系并传播数据特征。然而，在传播矩阵（例如邻接矩阵）中的估计误差可能对图滤波器和GCNs产生重大影响。本文研究概率图误差模型对GCN性能的影响。我们证明了在误差模型下的邻接矩阵受到图大小和误差概率函数的限制。我们进一步分析了带有自循环的归一化邻接矩阵的上界。最后，我们通过在合成数据集上运行实验来说明误差界限，并研究简单GCN在这种概率误差模型下的准确性敏感性。

    Graph convolutional networks (GCNs) can successfully learn the graph signal representation by graph convolution. The graph convolution depends on the graph filter, which contains the topological dependency of data and propagates data features. However, the estimation errors in the propagation matrix (e.g., the adjacency matrix) can have a significant impact on graph filters and GCNs. In this paper, we study the effect of a probabilistic graph error model on the performance of the GCNs. We prove that the adjacency matrix under the error model is bounded by a function of graph size and error probability. We further analytically specify the upper bound of a normalized adjacency matrix with self-loop added. Finally, we illustrate the error bounds by running experiments on a synthetic dataset and study the sensitivity of a simple GCN under this probabilistic error model on accuracy.
    

