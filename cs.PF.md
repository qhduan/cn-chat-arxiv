# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Novel Tensor-Expert Hybrid Parallelism Approach to Scale Mixture-of-Experts Training.](http://arxiv.org/abs/2303.06318) | 本文提出了一种新的混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。 |
| [^2] | [Exploiting Sparsity in Pruned Neural Networks to Optimize Large Model Training.](http://arxiv.org/abs/2302.05045) | 本文提出了一种新的方法，利用稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％。 |
| [^3] | [A comprehensive review of Binary Neural Network.](http://arxiv.org/abs/2110.06804) | 本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。 |

# 详细

[^1]: 一种新的张量专家混合并行方法来扩展混合专家训练

    A Novel Tensor-Expert Hybrid Parallelism Approach to Scale Mixture-of-Experts Training. (arXiv:2303.06318v1 [cs.LG])

    [http://arxiv.org/abs/2303.06318](http://arxiv.org/abs/2303.06318)

    本文提出了一种新的混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。

    This paper proposes a novel hybrid parallel algorithm that combines tensor, expert, and data parallelism to enable the training of MoE models with 4-8x larger base models than the current state-of-the-art -- DeepSpeed-MoE.

    最近提出了一种名为Mixture-of-Experts（MoE）的新型神经网络架构，通过添加稀疏激活的专家块来增加神经网络（基本模型）的参数，而不改变训练或推理的总浮点操作数。理论上，这种架构允许我们训练任意大的模型，同时保持计算成本与基本模型相同。然而，在64到128个专家块之外，先前的工作观察到这些MoE模型的测试准确性递减。因此，训练高质量的MoE模型需要我们扩展基本模型的大小以及专家块的数量。在这项工作中，我们提出了一种新颖的三维混合并行算法，结合了张量、专家和数据并行，以实现MoE模型的训练，其基本模型比当前最先进的DeepSpeed-MoE大4-8倍。我们在优化器步骤中提出了内存优化。

    A new neural network architecture called Mixture-of-Experts (MoE) has been proposed recently that increases the parameters of a neural network (the base model) by adding sparsely activated expert blocks, without changing the total number of floating point operations for training or inference. In theory, this architecture allows us to train arbitrarily large models while keeping the computational costs same as that of the base model. However, beyond 64 to 128 experts blocks, prior work has observed diminishing returns in the test accuracies of these MoE models. Thus, training high quality MoE models requires us to scale the size of the base models, along with the number of expert blocks. In this work, we propose a novel, three-dimensional, hybrid parallel algorithm that combines tensor, expert, and data parallelism to enable the training of MoE models with 4-8x larger base models than the current state-of-the-art -- DeepSpeed-MoE. We propose memory optimizations in the optimizer step, a
    
[^2]: 利用剪枝神经网络中的稀疏性来优化大型模型训练

    Exploiting Sparsity in Pruned Neural Networks to Optimize Large Model Training. (arXiv:2302.05045v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05045](http://arxiv.org/abs/2302.05045)

    本文提出了一种新的方法，利用稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％。

    This paper proposes a novel approach that exploits sparse subnetworks to optimize memory utilization and communication in two popular algorithms for parallel deep learning, and demonstrates significant reductions in memory consumption and communication time on a 2.7 billion parameter model using 512 NVIDIA V100 GPUs.

    由于通信开销的显著增加，规模化神经网络的并行训练具有挑战性。最近，深度学习研究人员开发了各种剪枝算法，能够剪枝（即将神经网络中的参数设置为零）80-90％的参数，以产生与未剪枝父网络相等的稀疏子网络。在本文中，我们提出了一种新的方法，利用这些稀疏子网络来优化两种流行的深度学习并行算法 - 数据并行和层间并行的内存利用和通信。我们将我们的方法集成到AxoNN中，这是一个高度可扩展的并行深度学习框架，依赖于数据和层间并行，并展示了通信时间和内存利用的减少。在512个NVIDIA V100 GPU上，我们的优化将27亿参数模型的内存消耗减少了74％，总通信时间减少了40％，从而提供了

    Parallel training of neural networks at scale is challenging due to significant overheads arising from communication. Recently, deep learning researchers have developed a variety of pruning algorithms that are capable of pruning (i.e. setting to zero) 80-90% of the parameters in a neural network to yield sparse subnetworks that equal the accuracy of the unpruned parent network. In this work, we propose a novel approach that exploits these sparse subnetworks to optimize the memory utilization and communication in two popular algorithms for parallel deep learning namely -- data and inter-layer parallelism. We integrate our approach into AxoNN, a highly scalable framework for parallel deep learning that relies on data and inter-layer parallelism, and demonstrate the reduction in communication time and memory utilization. On 512 NVIDIA V100 GPUs, our optimizations reduce the memory consumption of a 2.7 billion parameter model by 74%, and the total communication time by 40%, thus providing 
    
[^3]: 二进制神经网络的全面综述

    A comprehensive review of Binary Neural Network. (arXiv:2110.06804v4 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2110.06804](http://arxiv.org/abs/2110.06804)

    本文全面综述了二进制神经网络的最新发展，重点关注1位激活和1位卷积网络的权重，这些网络可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。

    This article provides a comprehensive overview of recent developments in Binary Neural Networks (BNN), with a focus on 1-bit activations and 1-bit convolution networks. These networks can be implemented and embedded on tiny restricted devices, saving significant storage, computation cost, and energy consumption.

    深度学习（DL）最近改变了智能系统的发展，并被广泛应用于许多实际应用中。尽管DL具有各种好处和潜力，但在不同的计算受限和能量受限设备中需要进行DL处理。研究二进制神经网络（BNN）等具有改变游戏规则的技术以增加深度学习能力是很自然的。最近在BNN方面取得了显着进展，因为它们可以在微小的受限设备上实现和嵌入，并节省大量存储、计算成本和能量消耗。然而，几乎所有的BNN行为都会带来额外的内存、计算成本和更高的性能。本文提供了BNN最近发展的完整概述。本文专门关注1位激活和1位卷积网络的权重，与以前的调查混合使用低位作品相反。它对BNN的开发进行了全面调查。

    Deep learning (DL) has recently changed the development of intelligent systems and is widely adopted in many real-life applications. Despite their various benefits and potentials, there is a high demand for DL processing in different computationally limited and energy-constrained devices. It is natural to study game-changing technologies such as Binary Neural Networks (BNN) to increase deep learning capabilities. Recently remarkable progress has been made in BNN since they can be implemented and embedded on tiny restricted devices and save a significant amount of storage, computation cost, and energy consumption. However, nearly all BNN acts trade with extra memory, computation cost, and higher performance. This article provides a complete overview of recent developments in BNN. This article focuses exclusively on 1-bit activations and weights 1-bit convolution networks, contrary to previous surveys in which low-bit works are mixed in. It conducted a complete investigation of BNN's dev
    

