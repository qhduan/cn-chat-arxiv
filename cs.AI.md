# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CURATRON: Complete Robust Preference Data for Robust Alignment of Large Language Models](https://arxiv.org/abs/2403.02745) | 本论文提出了一种新方法，通过彻底重校准偏好数据集中的价值观，以增强大型语言模型对问题的韧性。 |
| [^2] | [Quantum linear algebra is all you need for Transformer architectures](https://arxiv.org/abs/2402.16714) | 本文研究了在容错性量子计算的视角下Transformer架构，展示了如何利用量子线性代数构建Transformer的关键组件。 |
| [^3] | [Brain-Inspired Computational Intelligence via Predictive Coding.](http://arxiv.org/abs/2308.07870) | 这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。 |

# 详细

[^1]: CURATRON：完整健壮偏好数据用于大型语言模型的健壮对齐

    CURATRON: Complete Robust Preference Data for Robust Alignment of Large Language Models

    [https://arxiv.org/abs/2403.02745](https://arxiv.org/abs/2403.02745)

    本论文提出了一种新方法，通过彻底重校准偏好数据集中的价值观，以增强大型语言模型对问题的韧性。

    

    这篇论文解决了通过偏好学习（PL）将大型语言模型（LLMs）与人类价值观对齐的挑战，重点关注偏好数据集中不完整和损坏的问题。我们提出了一种新方法，通过彻底和完全地重新校准这些数据集中的价值观，以增强LLMs对问题的韧性。特别是，我们设计了一个有保证的多项式时间排名算法，可以增强几种现有模型的健壮性，比如经典的Bradley–Terry–Luce（BTL）（Bradley和Terry，1952）模型以及对其某些推广。据我们所知，我们的工作是第一个提出一种可证明在高概率下恢复{\epsilon}-最优排序的算法，同时允许每个模型响应多达O(n)扰动的成对比较结果。此外，我们展示了在部分观察设置下的健壮恢复结果。我们的实验证实了我们的算法

    arXiv:2403.02745v1 Announce Type: new  Abstract: This paper addresses the challenges of aligning large language models (LLMs) with human values via preference learning (PL), with a focus on the issues of incomplete and corrupted data in preference datasets. We propose a novel method for robustly and completely recalibrating values within these datasets to enhance LLMs resilience against the issues. In particular, we devise a guaranteed polynomial time ranking algorithm that robustifies several existing models, such as the classic Bradley--Terry--Luce (BTL) (Bradley and Terry, 1952) model and certain generalizations of it. To the best of our knowledge, our present work is the first to propose an algorithm that provably recovers an {\epsilon}-optimal ranking with high probability while allowing as large as O(n) perturbed pairwise comparison results per model response. Furthermore, we show robust recovery results in the partially observed setting. Our experiments confirm that our algorith
    
[^2]: 量子线性代数是Transformer架构所需的一切

    Quantum linear algebra is all you need for Transformer architectures

    [https://arxiv.org/abs/2402.16714](https://arxiv.org/abs/2402.16714)

    本文研究了在容错性量子计算的视角下Transformer架构，展示了如何利用量子线性代数构建Transformer的关键组件。

    

    生成式机器学习方法如大型语言模型正在彻底改变文本和图像的创作。本文通过容错性量子计算的视角研究了Transformer架构。我们展示了如何准备self-attention矩阵的块编码，并结合量子子程序构建了Transformer中的重要组成部分。

    arXiv:2402.16714v1 Announce Type: cross  Abstract: Generative machine learning methods such as large-language models are revolutionizing the creation of text and images. While these models are powerful they also harness a large amount of computational resources. The transformer is a key component in large language models that aims to generate a suitable completion of a given partial sequence. In this work, we investigate transformer architectures under the lens of fault-tolerant quantum computing. The input model is one where pre-trained weight matrices are given as block encodings to construct the query, key, and value matrices for the transformer. As a first step, we show how to prepare a block encoding of the self-attention matrix, with a row-wise application of the softmax function using the Hadamard product. In addition, we combine quantum subroutines to construct important building blocks in the transformer, the residual connection, layer normalization, and the feed-forward neura
    
[^3]: 通过预测编码实现脑启发式计算智能

    Brain-Inspired Computational Intelligence via Predictive Coding. (arXiv:2308.07870v1 [cs.AI])

    [http://arxiv.org/abs/2308.07870](http://arxiv.org/abs/2308.07870)

    这项研究介绍了一种通过预测编码的脑启发式计算智能方法，它可以解决现有人工智能方法的一些重要限制，并具有在机器学习领域有希望的应用潜力。

    

    人工智能（AI）正在迅速成为本世纪的关键技术之一。到目前为止，在AI领域取得的大部分成果都是使用误差反向传播学习算法训练的深度神经网络所实现的。然而，这种方法的普及应用已经凸显出了一些重要的局限性，例如计算成本高、难以量化不确定性、缺乏鲁棒性、不可靠性和生物学上的不合理性。解决这些限制可能需要受到神经科学理论的启发和指导的方案。其中一种理论称为预测编码（PC），在机器智能任务中表现出有希望的性能，具有令人兴奋的特性，使其在机器学习社区中具有潜在的价值：PC可以模拟不同脑区的信息处理，可以用于认知控制和机器人技术，并在变分推理方面具有坚实的数学基础，提供了一个强大的工具。

    Artificial intelligence (AI) is rapidly becoming one of the key technologies of this century. The majority of results in AI thus far have been achieved using deep neural networks trained with the error backpropagation learning algorithm. However, the ubiquitous adoption of this approach has highlighted some important limitations such as substantial computational cost, difficulty in quantifying uncertainty, lack of robustness, unreliability, and biological implausibility. It is possible that addressing these limitations may require schemes that are inspired and guided by neuroscience theories. One such theory, called predictive coding (PC), has shown promising performance in machine intelligence tasks, exhibiting exciting properties that make it potentially valuable for the machine learning community: PC can model information processing in different brain areas, can be used in cognitive control and robotics, and has a solid mathematical grounding in variational inference, offering a pow
    

