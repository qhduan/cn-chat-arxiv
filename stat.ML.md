# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual Fairness through Transforming Data Orthogonal to Bias](https://arxiv.org/abs/2403.17852) | 提出了一种新颖的数据预处理算法，正交于偏见（OB），通过确保数据与敏感变量不相关，实现机器学习应用中的反事实公平性。 |
| [^2] | [Flexible infinite-width graph convolutional networks and the importance of representation learning](https://arxiv.org/abs/2402.06525) | 本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。 |
| [^3] | [SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models.](http://arxiv.org/abs/2309.05019) | 本文提出了一种改进的高效随机亚当方法SA-Solver，用于解扩散随机微分方程以生成高质量的数据，实验结果显示它在少步采样中相较于现有最先进的方法有改进或可比的性能，并达到了SOTA FID分数。 |
| [^4] | [Efficient uniform approximation using Random Vector Functional Link networks.](http://arxiv.org/abs/2306.17501) | 本文研究了使用随机向量功能连接网络进行高效统一逼近的方法，证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是隐藏层相对于输入维度是指数级宽度的。这是第一个证明了$L_\infty$逼近误差和高斯内部权重条件下的结果，给出了非渐进性的隐藏层节点数量下界。 |

# 详细

[^1]: 通过将数据转化为与偏见正交的方式实现反事实公平性

    Counterfactual Fairness through Transforming Data Orthogonal to Bias

    [https://arxiv.org/abs/2403.17852](https://arxiv.org/abs/2403.17852)

    提出了一种新颖的数据预处理算法，正交于偏见（OB），通过确保数据与敏感变量不相关，实现机器学习应用中的反事实公平性。

    

    机器学习模型在解决各个领域的复杂问题中展现出了卓越的能力。然而，这些模型有时可能表现出有偏见的决策，导致不同群体之间的待遇不平等。尽管公平性方面的研究已经很广泛，但多元连续敏感变量对决策结果的微妙影响尚未得到充分研究。我们引入了一种新颖的数据预处理算法，即正交于偏见（OB），旨在消除连续敏感变量的影响，从而促进机器学习应用中的反事实公平性。我们的方法基于结构因果模型（SCM）中联合正态分布的假设，证明了通过确保数据与敏感变量不相关即可实现反事实公平性。OB算法与模型无关，适用于多种机器学习应用。

    arXiv:2403.17852v1 Announce Type: new  Abstract: Machine learning models have shown exceptional prowess in solving complex issues across various domains. Nonetheless, these models can sometimes exhibit biased decision-making, leading to disparities in treatment across different groups. Despite the extensive research on fairness, the nuanced effects of multivariate and continuous sensitive variables on decision-making outcomes remain insufficiently studied. We introduce a novel data pre-processing algorithm, Orthogonal to Bias (OB), designed to remove the influence of a group of continuous sensitive variables, thereby facilitating counterfactual fairness in machine learning applications. Our approach is grounded in the assumption of a jointly normal distribution within a structural causal model (SCM), proving that counterfactual fairness can be achieved by ensuring the data is uncorrelated with sensitive variables. The OB algorithm is model-agnostic, catering to a wide array of machine 
    
[^2]: 灵活的无限宽图卷积网络及表示学习的重要性

    Flexible infinite-width graph convolutional networks and the importance of representation learning

    [https://arxiv.org/abs/2402.06525](https://arxiv.org/abs/2402.06525)

    本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。

    

    理解神经网络的一种常见理论方法是进行无限宽度限制，此时输出成为高斯过程（GP）分布。这被称为神经网络高斯过程（NNGP）。然而，NNGP内核是固定的，只能通过少量超参数进行调节，消除了任何表示学习的可能性。这与有限宽度的神经网络形成对比，后者通常被认为能够表现良好，正是因为它们能够学习表示。因此，简化神经网络以使其在理论上可处理的同时，NNGP可能会消除使其工作良好的因素（表示学习）。这激发了我们对一系列图分类任务中表示学习是否必要的理解。我们开发了一个精确的工具来完成这个任务，即图卷积深度内核机（graph convolutional deep kernel machine）。这与NNGP非常相似，因为它是无限宽度限制并使用内核，但它带有一个“旋钮”来控制表示学习的程度。

    A common theoretical approach to understanding neural networks is to take an infinite-width limit, at which point the outputs become Gaussian process (GP) distributed. This is known as a neural network Gaussian process (NNGP). However, the NNGP kernel is fixed, and tunable only through a small number of hyperparameters, eliminating any possibility of representation learning. This contrasts with finite-width NNs, which are often believed to perform well precisely because they are able to learn representations. Thus in simplifying NNs to make them theoretically tractable, NNGPs may eliminate precisely what makes them work well (representation learning). This motivated us to understand whether representation learning is necessary in a range of graph classification tasks. We develop a precise tool for this task, the graph convolutional deep kernel machine. This is very similar to an NNGP, in that it is an infinite width limit and uses kernels, but comes with a `knob' to control the amount 
    
[^3]: SA-Solver：用于快速采样扩散模型的随机亚当求解器

    SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models. (arXiv:2309.05019v1 [cs.LG])

    [http://arxiv.org/abs/2309.05019](http://arxiv.org/abs/2309.05019)

    本文提出了一种改进的高效随机亚当方法SA-Solver，用于解扩散随机微分方程以生成高质量的数据，实验结果显示它在少步采样中相较于现有最先进的方法有改进或可比的性能，并达到了SOTA FID分数。

    

    扩散概率模型在生成任务中取得了相当大的成功。由于从扩散概率模型中进行采样相当于解扩散随机微分方程或常微分方程，这是一项耗时的工作，因此提出了许多基于改进的微分方程求解器的快速采样方法。这些技术中的大部分方法都考虑解扩散常微分方程，因为它具有更好的效率。然而，随机采样可以在生成多样化和高质量数据方面提供额外的优势。在这项工作中，我们从两个方面进行了对随机采样的综合分析：方差控制的扩散随机微分方程和线性多步扩散随机微分方程求解器。基于我们的分析，我们提出了SA-Solver，它是一种改进的高效随机亚当方法，用于解扩散随机微分方程以生成高质量的数据。我们的实验结果显示，SA-Solver实现了：1）在少步采样中与现有最先进的采样方法相比，有改进或可比性能；2）SOTA FID分数。

    Diffusion Probabilistic Models (DPMs) have achieved considerable success in generation tasks. As sampling from DPMs is equivalent to solving diffusion SDE or ODE which is time-consuming, numerous fast sampling methods built upon improved differential equation solvers are proposed. The majority of such techniques consider solving the diffusion ODE due to its superior efficiency. However, stochastic sampling could offer additional advantages in generating diverse and high-quality data. In this work, we engage in a comprehensive analysis of stochastic sampling from two aspects: variance-controlled diffusion SDE and linear multi-step SDE solver. Based on our analysis, we propose SA-Solver, which is an improved efficient stochastic Adams method for solving diffusion SDE to generate data with high quality. Our experiments show that SA-Solver achieves: 1) improved or comparable performance compared with the existing state-of-the-art sampling methods for few-step sampling; 2) SOTA FID scores o
    
[^4]: 使用随机向量功能连接网络进行高效统一逼近

    Efficient uniform approximation using Random Vector Functional Link networks. (arXiv:2306.17501v1 [stat.ML])

    [http://arxiv.org/abs/2306.17501](http://arxiv.org/abs/2306.17501)

    本文研究了使用随机向量功能连接网络进行高效统一逼近的方法，证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是隐藏层相对于输入维度是指数级宽度的。这是第一个证明了$L_\infty$逼近误差和高斯内部权重条件下的结果，给出了非渐进性的隐藏层节点数量下界。

    

    随机向量功能连接(RVFL)网络是一个具有随机内部权重和偏置的二层神经网络。由于这种架构只需要学习外部权重，学习过程可以简化为线性优化任务，从而避免了非凸优化问题的困扰。在本文中，我们证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是其隐藏层相对于输入维度是指数级宽度的。尽管之前已经证明了以$L_2$方式可以实现这样的逼近，但我们证明了在$L_\infty$逼近误差和高斯内部权重情况下的可行性。据我们所知，这是第一个这样的结果。我们给出了非渐进性的隐藏层节点数量的下界，取决于目标函数的利普希茨常数、期望的准确度和输入维度等因素。我们的证明方法根植于概率论。

    A Random Vector Functional Link (RVFL) network is a depth-2 neural network with random inner weights and biases. As only the outer weights of such architectures need to be learned, the learning process boils down to a linear optimization task, allowing one to sidestep the pitfalls of nonconvex optimization problems. In this paper, we prove that an RVFL with ReLU activation functions can approximate Lipschitz continuous functions provided its hidden layer is exponentially wide in the input dimension. Although it has been established before that such approximation can be achieved in $L_2$ sense, we prove it for $L_\infty$ approximation error and Gaussian inner weights. To the best of our knowledge, our result is the first of this kind. We give a nonasymptotic lower bound for the number of hidden layer nodes, depending on, among other things, the Lipschitz constant of the target function, the desired accuracy, and the input dimension. Our method of proof is rooted in probability theory an
    

