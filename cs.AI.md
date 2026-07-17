# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Short Review on Novel Approaches for Maximum Clique Problem: from Classical algorithms to Graph Neural Networks and Quantum algorithms](https://arxiv.org/abs/2403.09742) | 该综述回顾了解决最大团问题的经典算法，同时也涵盖了图神经网络和量子算法的最新进展，并提出了用于测试这些算法的基准。 |
| [^2] | [TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions](https://arxiv.org/abs/2403.01977) | TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。 |

# 详细

[^1]: 最大团问题的新方法简要回顾：从经典算法到图神经网络和量子算法

    A Short Review on Novel Approaches for Maximum Clique Problem: from Classical algorithms to Graph Neural Networks and Quantum algorithms

    [https://arxiv.org/abs/2403.09742](https://arxiv.org/abs/2403.09742)

    该综述回顾了解决最大团问题的经典算法，同时也涵盖了图神经网络和量子算法的最新进展，并提出了用于测试这些算法的基准。

    

    这篇手稿全面回顾了最大团问题，这是一个涉及在图中找到所有两两相邻的顶点子集的计算问题。手稿以简单的方式涵盖了解决该问题的经典算法，并包括了对图神经网络和量子算法最近发展的审查。该综述以基准测试来评估经典以及新的学习和量子算法。

    arXiv:2403.09742v1 Announce Type: new  Abstract: This manuscript provides a comprehensive review of the Maximum Clique Problem, a computational problem that involves finding subsets of vertices in a graph that are all pairwise adjacent to each other. The manuscript covers in a simple way classical algorithms for solving the problem and includes a review of recent developments in graph neural networks and quantum algorithms. The review concludes with benchmarks for testing classical as well as new learning, and quantum algorithms.
    
[^2]: TTA-Nav: 测试时自适应重建用于视觉损坏下的点目标导航

    TTA-Nav: Test-time Adaptive Reconstruction for Point-Goal Navigation under Visual Corruptions

    [https://arxiv.org/abs/2403.01977](https://arxiv.org/abs/2403.01977)

    TTA-Nav提出了一种测试时自适应方法，通过引入自顶向下解码器，从损坏图像中重建出更清晰的图像，显著增强了点目标导航性能。

    

    arXiv:2403.01977v1 公告类型: 跨  摘要: 在视觉损坏下的机器人导航是一个巨大的挑战。为了解决这一问题，我们提出了一种名为TTA-Nav的测试时自适应（TTA）方法，用于在视觉损坏下的点目标导航。我们的“即插即用”方法将自顶向下的解码器与预训练的导航模型相结合。首先，预训练的导航模型接收一个损坏的图像并提取特征。其次，自顶向下的解码器根据预训练模型提取的高级特征生成重建图像。然后，将损坏图像的重建图像馈送回预训练模型。最后，预训练模型再次进行前向传播以输出动作。尽管仅在清晰图像上训练，自顶向下的解码器可以从损坏图像中重建出更清晰的图像，无需基于梯度的自适应。具有我们自顶向下解码器的预训练导航模型显著提高了导航性能。

    arXiv:2403.01977v1 Announce Type: cross  Abstract: Robot navigation under visual corruption presents a formidable challenge. To address this, we propose a Test-time Adaptation (TTA) method, named as TTA-Nav, for point-goal navigation under visual corruptions. Our "plug-and-play" method incorporates a top-down decoder to a pre-trained navigation model. Firstly, the pre-trained navigation model gets a corrupted image and extracts features. Secondly, the top-down decoder produces the reconstruction given the high-level features extracted by the pre-trained model. Then, it feeds the reconstruction of a corrupted image back to the pre-trained model. Finally, the pre-trained model does forward pass again to output action. Despite being trained solely on clean images, the top-down decoder can reconstruct cleaner images from corrupted ones without the need for gradient-based adaptation. The pre-trained navigation model with our top-down decoder significantly enhances navigation performance acr
    

