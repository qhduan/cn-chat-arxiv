# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Gradient Descent for Additive Nonparametric Regression](https://arxiv.org/abs/2401.00691) | 本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。 |
| [^2] | [Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences.](http://arxiv.org/abs/2308.14555) | 本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。 |
| [^3] | [A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy.](http://arxiv.org/abs/2308.14048) | 本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。 |
| [^4] | [Structural restrictions in local causal discovery: identifying direct causes of a target variable.](http://arxiv.org/abs/2307.16048) | 这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。 |
| [^5] | [On the tightness of information-theoretic bounds on generalization error of learning algorithms.](http://arxiv.org/abs/2303.14658) | 本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。 |
| [^6] | [Indeterminate Probability Neural Network.](http://arxiv.org/abs/2303.11536) | 本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。 |
| [^7] | [On the fast convergence of minibatch heavy ball momentum.](http://arxiv.org/abs/2206.07553) | 本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。 |
| [^8] | [Stable and consistent density-based clustering via multiparameter persistence.](http://arxiv.org/abs/2005.09048) | 这篇论文通过引入一种度量层次聚类的对应交错距离，研究了一种稳定一致的密度-based聚类算法，提供了一个从一参数层次聚类中提取单个聚类的算法，并证明了该算法的一致性和稳定性。 |

# 详细

[^1]: 添加非参数回归的随机梯度下降

    Stochastic Gradient Descent for Additive Nonparametric Regression

    [https://arxiv.org/abs/2401.00691](https://arxiv.org/abs/2401.00691)

    本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。

    

    本文介绍了一种用于训练加性模型的迭代算法，该算法具有良好的内存存储和计算要求。该算法可以看作是对组件函数的截断基扩展的系数应用随机梯度下降的函数对应物。我们证明了得到的估计量满足一个奥拉克不等式，允许模型错误规范。在规范很好的情况下，通过在训练的三个不同阶段仔细选择学习率，我们证明了其风险在数据维度和训练样本大小的依赖方面是最小和最优的。通过在两个实际数据集上将该方法与传统的反向拟合进行比较，我们进一步说明了计算优势。

    This paper introduces an iterative algorithm for training additive models that enjoys favorable memory storage and computational requirements. The algorithm can be viewed as the functional counterpart of stochastic gradient descent, applied to the coefficients of a truncated basis expansion of the component functions. We show that the resulting estimator satisfies an oracle inequality that allows for model mis-specification. In the well-specified setting, by choosing the learning rate carefully across three distinct stages of training, we demonstrate that its risk is minimax optimal in terms of the dependence on the dimensionality of the data and the size of the training sample. We further illustrate the computational benefits by comparing the approach with traditional backfitting on two real-world datasets.
    
[^2]: 循环神经网络在遍历数据序列上训练的核极限

    Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences. (arXiv:2308.14555v1 [cs.LG])

    [http://arxiv.org/abs/2308.14555](http://arxiv.org/abs/2308.14555)

    本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。

    

    本文开发了数学方法来描述循环神经网络（RNN）的渐近特性，其中隐藏单元的数量、序列中的数据样本、隐藏状态的更新和训练步骤同时趋于无穷大。对于具有简化权重矩阵的RNN，我们证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。分析需要解决RNN所特有的几个挑战。在典型的均场应用中（例如前馈神经网络），离散的更新量为$\mathcal{O}(\frac{1}{N})$，更新的次数为$\mathcal{O}(N)$。因此，系统可以表示为适当ODE/PDE的Euler逼近，当$N \rightarrow \infty$时收敛到该ODE/PDE。然而，RNN的隐藏层更新为$\mathcal{O}(1)$。因此，RNN不能表示为ODE/PDE的离散化和标准均场技术。

    Mathematical methods are developed to characterize the asymptotics of recurrent neural networks (RNN) as the number of hidden units, data samples in the sequence, hidden state updates, and training steps simultaneously grow to infinity. In the case of an RNN with a simplified weight matrix, we prove the convergence of the RNN to the solution of an infinite-dimensional ODE coupled with the fixed point of a random algebraic equation. The analysis requires addressing several challenges which are unique to RNNs. In typical mean-field applications (e.g., feedforward neural networks), discrete updates are of magnitude $\mathcal{O}(\frac{1}{N})$ and the number of updates is $\mathcal{O}(N)$. Therefore, the system can be represented as an Euler approximation of an appropriate ODE/PDE, which it will converge to as $N \rightarrow \infty$. However, the RNN hidden layer updates are $\mathcal{O}(1)$. Therefore, RNNs cannot be represented as a discretization of an ODE/PDE and standard mean-field tec
    
[^3]: 一种贝叶斯非参数方法用于生成模型：使用Wasserstein和最大均值差异度量集成变分自编码器和生成对抗网络

    A Bayesian Non-parametric Approach to Generative Models: Integrating Variational Autoencoder and Generative Adversarial Networks using Wasserstein and Maximum Mean Discrepancy. (arXiv:2308.14048v1 [stat.ML])

    [http://arxiv.org/abs/2308.14048](http://arxiv.org/abs/2308.14048)

    本研究提出了一种融合生成对抗网络和变分自编码器的贝叶斯非参数方法，通过在损失函数中使用Wasserstein和最大均值差异度量，实现了对潜在空间的有效学习，并能够生成多样且高质量的样本。

    

    生成模型已成为一种产生与真实图像难以区分的高质量图像的有前途的技术。生成对抗网络（GAN）和变分自编码器（VAE）是最为重要且被广泛研究的两种生成模型。GAN在生成逼真图像方面表现出色，而VAE则能够生成多样的图像。然而，GAN忽视了大部分可能的输出空间，这导致不能完全体现目标分布的多样性，而VAE则常常生成模糊图像。为了充分发挥两种模型的优点并减轻它们的弱点，我们采用了贝叶斯非参数方法将GAN和VAE相结合。我们的方法在损失函数中同时使用了Wasserstein和最大均值差异度量，以有效学习潜在空间并生成多样且高质量的样本。

    Generative models have emerged as a promising technique for producing high-quality images that are indistinguishable from real images. Generative adversarial networks (GANs) and variational autoencoders (VAEs) are two of the most prominent and widely studied generative models. GANs have demonstrated excellent performance in generating sharp realistic images and VAEs have shown strong abilities to generate diverse images. However, GANs suffer from ignoring a large portion of the possible output space which does not represent the full diversity of the target distribution, and VAEs tend to produce blurry images. To fully capitalize on the strengths of both models while mitigating their weaknesses, we employ a Bayesian non-parametric (BNP) approach to merge GANs and VAEs. Our procedure incorporates both Wasserstein and maximum mean discrepancy (MMD) measures in the loss function to enable effective learning of the latent space and generate diverse and high-quality samples. By fusing the di
    
[^4]: 局部因果发现中的结构限制: 识别目标变量的直接原因

    Structural restrictions in local causal discovery: identifying direct causes of a target variable. (arXiv:2307.16048v1 [stat.ME])

    [http://arxiv.org/abs/2307.16048](http://arxiv.org/abs/2307.16048)

    这项研究的目标是从观测数据中识别目标变量的直接原因，通过不对其他变量做太多假设，研究者提出了可识别性结果和两种实用算法。

    

    我们考虑从观察联合分布中学习目标变量的一组直接原因的问题。学习表示因果结构的有向无环图(DAG)是科学中的一个基本问题。当完整的DAG从分布中可识别时，已知有一些结果，例如假设非线性高斯数据生成过程。通常，我们只对识别一个目标变量的直接原因（局部因果结构），而不是完整的DAG感兴趣。在本文中，我们讨论了对目标变量的数据生成过程的不同假设，该假设下直接原因集合可以从分布中识别出来。在这样做的过程中，我们对除目标变量之外的变量基本上没有任何假设。除了新的可识别性结果，我们还提供了两种从有限随机样本估计直接原因的实用算法，并在几个基准数据集上证明了它们的有效性。

    We consider the problem of learning a set of direct causes of a target variable from an observational joint distribution. Learning directed acyclic graphs (DAGs) that represent the causal structure is a fundamental problem in science. Several results are known when the full DAG is identifiable from the distribution, such as assuming a nonlinear Gaussian data-generating process. Often, we are only interested in identifying the direct causes of one target variable (local causal structure), not the full DAG. In this paper, we discuss different assumptions for the data-generating process of the target variable under which the set of direct causes is identifiable from the distribution. While doing so, we put essentially no assumptions on the variables other than the target variable. In addition to the novel identifiability results, we provide two practical algorithms for estimating the direct causes from a finite random sample and demonstrate their effectiveness on several benchmark dataset
    
[^5]: 关于学习算法泛化误差信息理论界限的紧密性研究

    On the tightness of information-theoretic bounds on generalization error of learning algorithms. (arXiv:2303.14658v1 [cs.IT])

    [http://arxiv.org/abs/2303.14658](http://arxiv.org/abs/2303.14658)

    本文研究了学习算法泛化误差信息理论界限的紧密性。研究表明，通过适当的假设，可以在快速收敛速度下使用信息理论量$O(\lambda/n)$来上界估计泛化误差。

    

    Russo和Xu提出了一种方法来证明学习算法的泛化误差可以通过信息度量进行上界估计。然而，这种收敛速度通常被认为是“慢”的，因为它的期望收敛速度的形式为$O(\sqrt{\lambda/n})$，其中$\lambda$是一些信息理论量。在本文中我们证明了根号并不一定意味着收敛速度慢，可以在适当的假设下使用这个界限来得到$O(\lambda/n)$的快速收敛速度。此外，我们确定了达到快速收敛速度的关键条件，即所谓的$(\eta,c)$-中心条件。在这个条件下，我们给出了学习算法泛化误差的信息理论界限。

    A recent line of works, initiated by Russo and Xu, has shown that the generalization error of a learning algorithm can be upper bounded by information measures. In most of the relevant works, the convergence rate of the expected generalization error is in the form of $O(\sqrt{\lambda/n})$ where $\lambda$ is some information-theoretic quantities such as the mutual information or conditional mutual information between the data and the learned hypothesis. However, such a learning rate is typically considered to be ``slow", compared to a ``fast rate" of $O(\lambda/n)$ in many learning scenarios. In this work, we first show that the square root does not necessarily imply a slow rate, and a fast rate result can still be obtained using this bound under appropriate assumptions. Furthermore, we identify the critical conditions needed for the fast rate generalization error, which we call the $(\eta,c)$-central condition. Under this condition, we give information-theoretic bounds on the generaliz
    
[^6]: 不定概率神经网络

    Indeterminate Probability Neural Network. (arXiv:2303.11536v1 [cs.LG])

    [http://arxiv.org/abs/2303.11536](http://arxiv.org/abs/2303.11536)

    本文提出了一种新型通用模型——不定概率神经网络；它可以进行无监督聚类和使用很小的神经网络处理大规模分类，其理论优势体现在新的概率理论和神经网络框架中。

    

    本文提出了一个称为IPNN的新型通用模型，它将神经网络和概率论结合在一起。在传统概率论中，概率的计算是基于事件的发生，而这在当前的神经网络中几乎不使用。因此，我们提出了一种新的概率理论，它是经典概率论的扩展，并使经典概率论成为我们理论的一种特殊情况。此外，对于我们提出的神经网络框架，神经网络的输出被定义为概率事件，并基于这些事件的统计分析，推导出分类任务的推理模型。IPNN展现了新的特性：它在进行分类的同时可以执行无监督聚类。此外，IPNN能够使用非常小的神经网络进行非常大的分类，例如100个输出节点的模型可以分类10亿类别。理论优势体现在新的概率理论和神经网络框架中，并且实验结果展示了IPNN在各种应用中的潜力。

    We propose a new general model called IPNN - Indeterminate Probability Neural Network, which combines neural network and probability theory together. In the classical probability theory, the calculation of probability is based on the occurrence of events, which is hardly used in current neural networks. In this paper, we propose a new general probability theory, which is an extension of classical probability theory, and makes classical probability theory a special case to our theory. Besides, for our proposed neural network framework, the output of neural network is defined as probability events, and based on the statistical analysis of these events, the inference model for classification task is deduced. IPNN shows new property: It can perform unsupervised clustering while doing classification. Besides, IPNN is capable of making very large classification with very small neural network, e.g. model with 100 output nodes can classify 10 billion categories. Theoretical advantages are refl
    
[^7]: 论小批量重球动量法的快速收敛性

    On the fast convergence of minibatch heavy ball momentum. (arXiv:2206.07553v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.07553](http://arxiv.org/abs/2206.07553)

    本文研究了一种随机Kaczmarz算法，使用小批量和重球动量进行加速，在二次优化问题中保持快速收敛率。

    

    简单的随机动量方法被广泛用于机器学习优化中，但由于还没有加速的理论保证，这与它们在实践中的良好性能并不相符。本文旨在通过展示，随机重球动量在二次最优化问题中保持（确定性）重球动量的快速线性率，至少在使用足够大的批量大小进行小批量处理时。我们所研究的算法可以被解释为带小批量处理和重球动量的加速随机Kaczmarz算法。该分析依赖于仔细分解动量转移矩阵，并使用新的独立随机矩阵乘积的谱范围集中界限。我们提供了数值演示，证明了我们的界限相当尖锐。

    Simple stochastic momentum methods are widely used in machine learning optimization, but their good practical performance is at odds with an absence of theoretical guarantees of acceleration in the literature. In this work, we aim to close the gap between theory and practice by showing that stochastic heavy ball momentum retains the fast linear rate of (deterministic) heavy ball momentum on quadratic optimization problems, at least when minibatching with a sufficiently large batch size. The algorithm we study can be interpreted as an accelerated randomized Kaczmarz algorithm with minibatching and heavy ball momentum. The analysis relies on carefully decomposing the momentum transition matrix, and using new spectral norm concentration bounds for products of independent random matrices. We provide numerical illustrations demonstrating that our bounds are reasonably sharp.
    
[^8]: 稳定一致的密度-based聚类算法通过多参数持续性

    Stable and consistent density-based clustering via multiparameter persistence. (arXiv:2005.09048v3 [math.ST] UPDATED)

    [http://arxiv.org/abs/2005.09048](http://arxiv.org/abs/2005.09048)

    这篇论文通过引入一种度量层次聚类的对应交错距离，研究了一种稳定一致的密度-based聚类算法，提供了一个从一参数层次聚类中提取单个聚类的算法，并证明了该算法的一致性和稳定性。

    

    我们考虑了拓扑数据分析中的度-Rips构造，它提供了一种密度敏感的多参数层次聚类算法。我们使用我们引入的一种度量层次聚类的对应交错距离，分析了它对输入数据的扰动的稳定性。从度-Rips中取某些一参数切片可以恢复出已知的基于密度的聚类方法，但我们证明了这些方法是不稳定的。然而，我们证明了作为多参数对象的度-Rips是稳定的，并提出了一种从度-Rips中取切片的替代方法，该方法产生一个具有更好稳定性属性的一参数层次聚类算法。我们使用对应交错距离证明了该算法的一致性。我们提供了从一参数层次聚类中提取单个聚类的算法，该算法在对应交错距离方面是稳定的。

    We consider the degree-Rips construction from topological data analysis, which provides a density-sensitive, multiparameter hierarchical clustering algorithm. We analyze its stability to perturbations of the input data using the correspondence-interleaving distance, a metric for hierarchical clusterings that we introduce. Taking certain one-parameter slices of degree-Rips recovers well-known methods for density-based clustering, but we show that these methods are unstable. However, we prove that degree-Rips, as a multiparameter object, is stable, and we propose an alternative approach for taking slices of degree-Rips, which yields a one-parameter hierarchical clustering algorithm with better stability properties. We prove that this algorithm is consistent, using the correspondence-interleaving distance. We provide an algorithm for extracting a single clustering from one-parameter hierarchical clusterings, which is stable with respect to the correspondence-interleaving distance. And, we
    

