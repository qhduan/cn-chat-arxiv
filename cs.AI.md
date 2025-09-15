# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential.](http://arxiv.org/abs/2304.11631) | TSGCNeXt是一个用于基于骨架的动作识别的模型，具有长期学习潜力，它采用动静态多图卷积来汇集多个独立拓扑图的特征，以及构建了一个图形卷积训练加速机制。 |
| [^2] | [Sufficient Invariant Learning for Distribution Shift.](http://arxiv.org/abs/2210.13533) | 本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。 |

# 详细

[^1]: TSGCNeXt：具备长期学习潜力的高效基于骨架的动作识别的动静态多图卷积

    TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential. (arXiv:2304.11631v1 [cs.CV])

    [http://arxiv.org/abs/2304.11631](http://arxiv.org/abs/2304.11631)

    TSGCNeXt是一个用于基于骨架的动作识别的模型，具有长期学习潜力，它采用动静态多图卷积来汇集多个独立拓扑图的特征，以及构建了一个图形卷积训练加速机制。

    

    随着图卷积网络（GCN）的发展，基于骨架的动作识别在人类动作识别方面取得了显著的成果。然而，最近的研究趋向于构建具有冗余训练的复杂学习机制，并存在长时间序列的瓶颈。为了解决这些问题，我们提出了Temporal-Spatio Graph ConvNeXt（TSGCNeXt）来探索长时间骨骼序列的高效学习机制。首先，我们提出了一个新的图形学习机制，动静分离多图卷积（DS-SMG），以汇集多个独立拓扑图的特征并避免节点信息在动态卷积期间被忽略。接下来，我们构建了一个图形卷积训练加速机制，以55.08％的速度提高动态图形学习的反向传播计算速度。最后，TSGCNeXt通过三个时空学习模块重新构建了GCN的整体结构，实现了更加高效的基于骨架的动作识别。

    Skeleton-based action recognition has achieved remarkable results in human action recognition with the development of graph convolutional networks (GCNs). However, the recent works tend to construct complex learning mechanisms with redundant training and exist a bottleneck for long time-series. To solve these problems, we propose the Temporal-Spatio Graph ConvNeXt (TSGCNeXt) to explore efficient learning mechanism of long temporal skeleton sequences. Firstly, a new graph learning mechanism with simple structure, Dynamic-Static Separate Multi-graph Convolution (DS-SMG) is proposed to aggregate features of multiple independent topological graphs and avoid the node information being ignored during dynamic convolution. Next, we construct a graph convolution training acceleration mechanism to optimize the back-propagation computing of dynamic graph learning with 55.08\% speed-up. Finally, the TSGCNeXt restructure the overall structure of GCN with three Spatio-temporal learning modules,effic
    
[^2]: 分布转移的充分不变学习

    Sufficient Invariant Learning for Distribution Shift. (arXiv:2210.13533v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13533](http://arxiv.org/abs/2210.13533)

    本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。

    

    机器学习算法在各种应用中展现出了卓越的性能。然而，在训练集和测试集的分布不同的情况下，保证性能仍然具有挑战性。为了改善分布转移情况下的性能，已经提出了一些方法，通过学习跨组或领域的不变特征来提高性能。然而，我们观察到之前的工作只部分地学习了不变特征。虽然先前的工作侧重于有限的不变特征，但我们首次提出了充分不变特征的重要性。由于只有训练集是经验性的，从训练集中学习得到的部分不变特征可能不存在于分布转移时的测试集中。因此，分布转移情况下的性能提高可能受到限制。本文认为从训练集中学习充分的不变特征对于分布转移情况至关重要。

    Machine learning algorithms have shown remarkable performance in diverse applications. However, it is still challenging to guarantee performance in distribution shifts when distributions of training and test datasets are different. There have been several approaches to improve the performance in distribution shift cases by learning invariant features across groups or domains. However, we observe that the previous works only learn invariant features partially. While the prior works focus on the limited invariant features, we first raise the importance of the sufficient invariant features. Since only training sets are given empirically, the learned partial invariant features from training sets might not be present in the test sets under distribution shift. Therefore, the performance improvement on distribution shifts might be limited. In this paper, we argue that learning sufficient invariant features from the training set is crucial for the distribution shift case. Concretely, we newly 
    

