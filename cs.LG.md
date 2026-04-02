# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Differentially Private Functional Summaries via the Independent Component Laplace Process.](http://arxiv.org/abs/2309.00125) | 本论文提出了一种新的差分隐私函数性摘要机制，通过使用独立分量拉普拉斯过程对无限维的函数性摘要进行扰动，放宽了对数据轨迹的假设，并相对于传统的有限维子空间嵌入方法保留了更高的效用。实验证明了该机制的可行性和有效性。 |
| [^2] | [MCMC-Correction of Score-Based Diffusion Models for Model Composition.](http://arxiv.org/abs/2307.14012) | 本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。 |
| [^3] | [A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware.](http://arxiv.org/abs/2306.14052) | 本文综述了如何加速图神经网络处理图结构数据的方法，包括智能算法、高效系统和自定义硬件等，提出了GNN加速的分类，并为未来的研究提供了方向。 |
| [^4] | [Learning to Grow Artificial Hippocampi in Vision Transformers for Resilient Lifelong Learning.](http://arxiv.org/abs/2303.08250) | 本文提出了一种在Vision Transformer中学习成长人工海马的方法，以实现弹性终身学习。通过神经架构搜索进行维护，选取多头自注意力块中的最终线性投影层进行ArtiHippo的实现和成长。 |

# 详细

[^1]: 通过独立分量拉普拉斯过程实现差分隐私的函数性摘要

    Differentially Private Functional Summaries via the Independent Component Laplace Process. (arXiv:2309.00125v1 [stat.ML])

    [http://arxiv.org/abs/2309.00125](http://arxiv.org/abs/2309.00125)

    本论文提出了一种新的差分隐私函数性摘要机制，通过使用独立分量拉普拉斯过程对无限维的函数性摘要进行扰动，放宽了对数据轨迹的假设，并相对于传统的有限维子空间嵌入方法保留了更高的效用。实验证明了该机制的可行性和有效性。

    

    在这项工作中，我们提出了一种称为独立分量拉普拉斯过程（ICLP）机制的差分隐私函数性摘要的新机制。通过将感兴趣的函数性摘要视为真正无限维对象，并使用ICLP噪声来扰动它们，该新机制放宽了关于数据轨迹的假设，并相对于文献中的经典有限维子空间嵌入方法保留了更高的效用。我们在多个函数空间中验证了所提出机制的可行性。我们考虑了几个统计估计问题，并通过轻微过平滑摘要来证明隐私成本不会主导统计误差，并且在渐近情况下可以忽略。对合成和真实数据集的数值实验证明了所提出机制的有效性。

    In this work, we propose a new mechanism for releasing differentially private functional summaries called the Independent Component Laplace Process, or ICLP, mechanism. By treating the functional summaries of interest as truly infinite-dimensional objects and perturbing them with the ICLP noise, this new mechanism relaxes assumptions on data trajectories and preserves higher utility compared to classical finite-dimensional subspace embedding approaches in the literature. We establish the feasibility of the proposed mechanism in multiple function spaces. Several statistical estimation problems are considered, and we demonstrate by slightly over-smoothing the summary, the privacy cost will not dominate the statistical error and is asymptotically negligible. Numerical experiments on synthetic and real datasets demonstrate the efficacy of the proposed mechanism.
    
[^2]: MCMC-修正基于得分的扩散模型用于模型组合

    MCMC-Correction of Score-Based Diffusion Models for Model Composition. (arXiv:2307.14012v1 [stat.ML])

    [http://arxiv.org/abs/2307.14012](http://arxiv.org/abs/2307.14012)

    本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。

    

    扩散模型可以用得分或能量函数来参数化。能量参数化具有更好的理论特性，主要是它可以通过在提议样本中总能量的变化基于Metropolis-Hastings修正步骤来进行扩展采样过程。然而，它似乎产生了稍微较差的性能，更重要的是，由于基于得分的扩散模型的普遍流行，现有的预训练能量参数化模型的可用性受到限制。这种限制削弱了模型组合的目的，即将预训练模型组合起来从新分布中进行采样。然而，我们的提议建议保留得分参数化，而是通过对得分函数进行线积分来计算基于能量的接受概率。这使我们能够重用现有的扩散模型，并将反向过程与各种马尔可夫链蒙特卡罗（MCMC）方法组合起来。

    Diffusion models can be parameterised in terms of either a score or an energy function. The energy parameterisation has better theoretical properties, mainly that it enables an extended sampling procedure with a Metropolis--Hastings correction step, based on the change in total energy in the proposed samples. However, it seems to yield slightly worse performance, and more importantly, due to the widespread popularity of score-based diffusion, there are limited availability of off-the-shelf pre-trained energy-based ones. This limitation undermines the purpose of model composition, which aims to combine pre-trained models to sample from new distributions. Our proposal, however, suggests retaining the score parameterization and instead computing the energy-based acceptance probability through line integration of the score function. This allows us to re-use existing diffusion models and still combine the reverse process with various Markov-Chain Monte Carlo (MCMC) methods. We evaluate our 
    
[^3]: 一项图神经网络加速的综述：算法、系统和自定义硬件

    A Survey on Graph Neural Network Acceleration: Algorithms, Systems, and Customized Hardware. (arXiv:2306.14052v1 [cs.LG])

    [http://arxiv.org/abs/2306.14052](http://arxiv.org/abs/2306.14052)

    本文综述了如何加速图神经网络处理图结构数据的方法，包括智能算法、高效系统和自定义硬件等，提出了GNN加速的分类，并为未来的研究提供了方向。

    

    图神经网络（GNN）正在成为处理图结构数据的机器学习研究中的新兴领域。尽管GNN在许多任务上取得了最先进的性能，但在涉及许多数据和严格延迟要求的实际应用中，它们面临着可扩展性挑战。为了解决这些挑战，许多研究已经在如何加速GNN方面进行了。这些加速技术涉及GNN流程的各个方面，从智能训练和推理算法到高效的系统和自定义硬件。随着对GNN加速的研究量快速增长，缺乏一个系统化的处理来提供一个统一的视图，并解决相关工作的复杂性问题。在这篇综述中，我们提供了GNN加速的分类，审查了现有方法，并提出了未来的研究方向。我们对GNN加速的分类处理连接了现有的工作，并为这一领域的进一步发展铺平了道路。

    Graph neural networks (GNNs) are emerging for machine learning research on graph-structured data. GNNs achieve state-of-the-art performance on many tasks, but they face scalability challenges when it comes to real-world applications that have numerous data and strict latency requirements. Many studies have been conducted on how to accelerate GNNs in an effort to address these challenges. These acceleration techniques touch on various aspects of the GNN pipeline, from smart training and inference algorithms to efficient systems and customized hardware. As the amount of research on GNN acceleration has grown rapidly, there lacks a systematic treatment to provide a unified view and address the complexity of relevant works. In this survey, we provide a taxonomy of GNN acceleration, review the existing approaches, and suggest future research directions. Our taxonomic treatment of GNN acceleration connects the existing works and sets the stage for further development in this area.
    
[^4]: 在视觉Transformer中学习成长人工海马，实现弹性终身学习

    Learning to Grow Artificial Hippocampi in Vision Transformers for Resilient Lifelong Learning. (arXiv:2303.08250v1 [cs.CV])

    [http://arxiv.org/abs/2303.08250](http://arxiv.org/abs/2303.08250)

    本文提出了一种在Vision Transformer中学习成长人工海马的方法，以实现弹性终身学习。通过神经架构搜索进行维护，选取多头自注意力块中的最终线性投影层进行ArtiHippo的实现和成长。

    

    终身学习需要拥有人类智能的韧性，即不存在灾难性遗忘，这种韧性与大脑中复杂的记忆机制，尤其是海马维护的长期记忆（LM）紧密相关。Transformer已经成为人工智能“大脑”的对应体，但LM组件在终身学习中尚未充分探索。本文提出了一种在Vision Transformer中学习成长人工海马（ArtiHippo）以实现弹性终身学习的方法。通过全面消融实验，选定多头自注意力（MHSA）块中的最终线性投影层来实现和成长ArtiHippo。ArtiHippo由专家混合（MoEs）表示。每个专家组件是线性投影层的现场变体，通过神经架构搜索（NAS）进行维护，搜索空间由四个基本成长操作（跳过、重用、适应和新）定义。

    Lifelong learning without catastrophic forgetting (i.e., resiliency) possessed by human intelligence is entangled with sophisticated memory mechanisms in the brain, especially the long-term memory (LM) maintained by Hippocampi. To a certain extent, Transformers have emerged as the counterpart ``Brain" of Artificial Intelligence (AI), and yet leave the LM component under-explored for lifelong learning settings. This paper presents a method of learning to grow Artificial Hippocampi (ArtiHippo) in Vision Transformers (ViTs) for resilient lifelong learning. With a comprehensive ablation study, the final linear projection layer in the multi-head self-attention (MHSA) block is selected in realizing and growing ArtiHippo. ArtiHippo is represented by a mixture of experts (MoEs). Each expert component is an on-site variant of the linear projection layer, maintained via neural architecture search (NAS) with the search space defined by four basic growing operations -- skip, reuse, adapt, and new 
    

