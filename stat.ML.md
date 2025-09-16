# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Early alignment in two-layer networks training is a two-edged sword.](http://arxiv.org/abs/2401.10791) | 本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。 |
| [^2] | [Piecewise Deterministic Markov Processes for Bayesian Neural Networks.](http://arxiv.org/abs/2302.08724) | 本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。 |

# 详细

[^1]: 两层网络训练中的早期对齐是一把双刃剑

    Early alignment in two-layer networks training is a two-edged sword. (arXiv:2401.10791v1 [cs.LG])

    [http://arxiv.org/abs/2401.10791](http://arxiv.org/abs/2401.10791)

    本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。

    

    使用一阶优化方法训练神经网络是深度学习成功的核心。初始化的规模是一个关键因素，因为小的初始化通常与特征学习模式相关，在这种模式下，梯度下降对简单解隐含偏好。本文提供了早期对齐阶段的普遍和量化描述，最初由Maennel等人提出。对于小初始化和一个隐藏的ReLU层网络，训练动态的早期阶段导致神经元向关键方向进行对齐。这种对齐引发了网络的稀疏表示，这与梯度流在收敛时的隐含偏好直接相关。然而，这种稀疏诱导的对齐是以在最小化训练目标方面遇到困难为代价的：我们还提供了一个简单的数据示例，其中超参数网络无法收敛到全局最小值。

    Training neural networks with first order optimisation methods is at the core of the empirical success of deep learning. The scale of initialisation is a crucial factor, as small initialisations are generally associated to a feature learning regime, for which gradient descent is implicitly biased towards simple solutions. This work provides a general and quantitative description of the early alignment phase, originally introduced by Maennel et al. (2018) . For small initialisation and one hidden ReLU layer networks, the early stage of the training dynamics leads to an alignment of the neurons towards key directions. This alignment induces a sparse representation of the network, which is directly related to the implicit bias of gradient flow at convergence. This sparsity inducing alignment however comes at the expense of difficulties in minimising the training objective: we also provide a simple data example for which overparameterised networks fail to converge towards global minima and
    
[^2]: 基于分段确定性马尔可夫过程的贝叶斯神经网络研究

    Piecewise Deterministic Markov Processes for Bayesian Neural Networks. (arXiv:2302.08724v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08724](http://arxiv.org/abs/2302.08724)

    本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。

    

    现代贝叶斯神经网络（BNNs）的推理通常依赖于变分推断处理，这要求违反了独立性和后验形式的假设。传统的MCMC方法避免了这些假设，但由于无法适应似然的子采样，导致计算量增加。新的分段确定性马尔可夫过程（PDMP）采样器允许子采样，但引入了模型特定的不均匀泊松过程（IPPs），从中采样困难。本研究引入了一种新的通用自适应稀疏方案，用于从这些IPPs中进行采样，并展示了如何加速将PDMPs应用于BNNs推理。实验表明，使用这些方法进行推理在计算上是可行的，可以提高预测准确性、MCMC混合性能，并与其他近似推理方案相比，提供更有信息量的不确定性测量。

    Inference on modern Bayesian Neural Networks (BNNs) often relies on a variational inference treatment, imposing violated assumptions of independence and the form of the posterior. Traditional MCMC approaches avoid these assumptions at the cost of increased computation due to its incompatibility to subsampling of the likelihood. New Piecewise Deterministic Markov Process (PDMP) samplers permit subsampling, though introduce a model specific inhomogenous Poisson Process (IPPs) which is difficult to sample from. This work introduces a new generic and adaptive thinning scheme for sampling from these IPPs, and demonstrates how this approach can accelerate the application of PDMPs for inference in BNNs. Experimentation illustrates how inference with these methods is computationally feasible, can improve predictive accuracy, MCMC mixing performance, and provide informative uncertainty measurements when compared against other approximate inference schemes.
    

