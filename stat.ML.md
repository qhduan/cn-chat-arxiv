# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dataset Size Dependence of Rate-Distortion Curve and Threshold of Posterior Collapse in Linear VAE.](http://arxiv.org/abs/2309.07663) | 本文通过分析在高维限制下的最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。结果显示，随着beta的增加，产生较大的广义误差平台，并且选择一个小于特定阈值的beta值可以提高模型性能。 |
| [^2] | [Privacy Amplification via Importance Sampling.](http://arxiv.org/abs/2307.10187) | 通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。 |

# 详细

[^1]: 线性变分自编码器中数据集大小对率失真曲线和后验坍缩阈值的影响

    Dataset Size Dependence of Rate-Distortion Curve and Threshold of Posterior Collapse in Linear VAE. (arXiv:2309.07663v1 [stat.ML])

    [http://arxiv.org/abs/2309.07663](http://arxiv.org/abs/2309.07663)

    本文通过分析在高维限制下的最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。结果显示，随着beta的增加，产生较大的广义误差平台，并且选择一个小于特定阈值的beta值可以提高模型性能。

    

    在变分自编码器（VAE）中，变分后验经常与先验密切吻合，这被称为后验坍缩，影响了表示学习的质量。为了缓解这个问题，VAE中引入了一个可调节的超参数beta。本文通过在高维限制下分析最简化的VAE，提出了一个闭式表达式，评估了beta与VAE中数据集大小、后验坍缩和率失真曲线之间的关系。这些结果表明，一个较大的beta会产生一个长的广义误差平台。随着beta的增加，平台的长度延长，超过一定的阈值后变为无穷。这意味着与通常的正则化参数不同，beta的选择可能会导致后验坍缩，而与数据集大小无关。因此，beta是一个需要谨慎调整的风险参数。此外，考虑到数据集大小对率失真曲线的依赖性，我们发现存在一个与数据集大小相关的阈值，选择小于这个阈值的beta值可以提高模型的性能。

    In the Variational Autoencoder (VAE), the variational posterior often aligns closely with the prior, which is known as posterior collapse and hinders the quality of representation learning. To mitigate this problem, an adjustable hyperparameter beta has been introduced in the VAE. This paper presents a closed-form expression to assess the relationship between the beta in VAE, the dataset size, the posterior collapse, and the rate-distortion curve by analyzing a minimal VAE in a high-dimensional limit. These results clarify that a long plateau in the generalization error emerges with a relatively larger beta. As the beta increases, the length of the plateau extends and then becomes infinite beyond a certain beta threshold. This implies that the choice of beta, unlike the usual regularization parameters, can induce posterior collapse regardless of the dataset size. Thus, beta is a risky parameter that requires careful tuning. Furthermore, considering the dataset-size dependence on the ra
    
[^2]: 隐私放大通过重要性采样

    Privacy Amplification via Importance Sampling. (arXiv:2307.10187v1 [cs.CR])

    [http://arxiv.org/abs/2307.10187](http://arxiv.org/abs/2307.10187)

    通过重要性采样进行隐私放大，可以同时增强隐私保护和提高效用。我们提供了一个一般的结果来量化选择概率权重对隐私放大的影响，并展示了异质采样概率可以在保持子采样大小不变的情况下获得更好的隐私和效用。

    

    我们研究了通过重要性采样对数据集进行子采样作为差分隐私机制的预处理步骤来增强隐私保护的性质。这扩展了已有的通过子采样进行隐私放大的结果到重要性采样，其中每个数据点的权重为其被选择概率的倒数。每个点的选择概率的权重对隐私的影响并不明显。一方面，较低的选择概率会导致更强的隐私放大。另一方面，权重越高，在点被选择时，点对机制输出的影响就越强。我们提供了一个一般的结果来量化这两个影响之间的权衡。我们展示了异质采样概率可以同时比均匀子采样具有更强的隐私和更好的效用，并保持子采样大小不变。特别地，我们制定和解决了隐私优化采样的问题，即寻找...

    We examine the privacy-enhancing properties of subsampling a data set via importance sampling as a pre-processing step for differentially private mechanisms. This extends the established privacy amplification by subsampling result to importance sampling where each data point is weighted by the reciprocal of its selection probability. The implications for privacy of weighting each point are not obvious. On the one hand, a lower selection probability leads to a stronger privacy amplification. On the other hand, the higher the weight, the stronger the influence of the point on the output of the mechanism in the event that the point does get selected. We provide a general result that quantifies the trade-off between these two effects. We show that heterogeneous sampling probabilities can lead to both stronger privacy and better utility than uniform subsampling while retaining the subsample size. In particular, we formulate and solve the problem of privacy-optimal sampling, that is, finding
    

