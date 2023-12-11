# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Temporal graph models fail to capture global temporal dynamics.](http://arxiv.org/abs/2309.15730) | 时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。 |

# 详细

[^1]: 时间图模型无法捕捉全局时间动态

    Temporal graph models fail to capture global temporal dynamics. (arXiv:2309.15730v1 [cs.IR])

    [http://arxiv.org/abs/2309.15730](http://arxiv.org/abs/2309.15730)

    时间图模型无法捕捉全局时间动态，我们提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了两个基于Wasserstein距离的度量来量化全局动态。我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用，我们还展示了简单的负采样方法可能导致模型退化。我们提出了改进的负采样方案，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。

    

    在动态链接属性预测的背景下，我们分析了最近发布的时间图基准，并提出了一种"最近流行节点"的基线方法，在时间图基准的中等和大规模数据集上胜过其他方法。我们提出了基于Wasserstein距离的两个度量，可以量化数据集的短期和长期全局动态的强度。通过分析我们出乎意料的强大基线，我们展示了标准的负采样评估方法在具有强烈时间动态的数据集上可能不适用。我们还展示了简单的负采样方法在训练过程中可能导致模型退化，导致无法对时间图网络进行排序的预测完全饱和。我们提出了改进的负采样方案用于训练和评估，并证明了它们的有效性。我们还将其与无负采样的非对比训练模型进行了比较。我们的结果表明...

    A recently released Temporal Graph Benchmark is analyzed in the context of Dynamic Link Property Prediction. We outline our observations and propose a trivial optimization-free baseline of "recently popular nodes" outperforming other methods on all medium and large-size datasets in the Temporal Graph Benchmark. We propose two measures based on Wasserstein distance which can quantify the strength of short-term and long-term global dynamics of datasets. By analyzing our unexpectedly strong baseline, we show how standard negative sampling evaluation can be unsuitable for datasets with strong temporal dynamics. We also show how simple negative-sampling can lead to model degeneration during training, resulting in impossible to rank, fully saturated predictions of temporal graph networks. We propose improved negative sampling schemes for both training and evaluation and prove their usefulness. We conduct a comparison with a model trained non-contrastively without negative sampling. Our resul
    

