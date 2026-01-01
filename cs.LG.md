# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Maxwell's Demon at Work: Efficient Pruning by Leveraging Saturation of Neurons](https://arxiv.org/abs/2403.07688) | 重新评估深度神经网络中的死亡神经元现象，提出了Demon Pruning（DemP）方法，通过控制死亡神经元的产生，动态实现网络稀疏化。 |
| [^2] | [Orchid: Flexible and Data-Dependent Convolution for Sequence Modeling](https://arxiv.org/abs/2402.18508) | 兰花引入了一种新的数据相关卷积机制，通过动态调整卷积核，实现了高表达能力和计算效率的平衡。 |
| [^3] | [Stochastic Gradient Descent for Additive Nonparametric Regression](https://arxiv.org/abs/2401.00691) | 本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。 |
| [^4] | [Are Ensembles Getting Better all the Time?](https://arxiv.org/abs/2311.17885) | 只有当考虑的损失函数为凸函数时，集成模型一直在变得更好，当损失函数为非凸函数时，好模型的集成变得更好，坏模型的集成变得更糟。 |
| [^5] | [Content-based Recommendation Engine for Video Streaming Platform.](http://arxiv.org/abs/2308.08406) | 本文提出了一种基于内容的推荐引擎，通过计算文档中单词的相关性和使用余弦相似度方法来推荐视频给用户。同时，还通过计算精确率、召回率和F1得分来评估引擎的性能。 |
| [^6] | [Generative Modelling of L\'{e}vy Area for High Order SDE Simulation.](http://arxiv.org/abs/2308.02452) | 本文提出了一种基于深度学习的模型LévyGAN，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩，解决了非高斯性质下的抽样困难问题。 |
| [^7] | [Machine learning for option pricing: an empirical investigation of network architectures.](http://arxiv.org/abs/2307.07657) | 广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。 |
| [^8] | [HiGen: Hierarchical Graph Generative Networks.](http://arxiv.org/abs/2305.19337) | HiGen是一种新颖的图形生成网络，能够以粗到细的方式捕捉图形的层次结构，并使用多项式分布来生成具有整数值的子图的边权。它能够有效地捕捉图形的局部和全局属性，并实现了最先进的性能。 |
| [^9] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |
| [^10] | [The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing.](http://arxiv.org/abs/2302.01186) | 该研究提出了ScaledGD(𝜆)方法，相较于传统梯度下降法更加鲁棒，并且在处理低秩矩阵感知问题时具有很好的表现。 |

# 详细

[^1]: Maxwell的恶魔之工作：通过利用神经元饱和实现有效修剪

    Maxwell's Demon at Work: Efficient Pruning by Leveraging Saturation of Neurons

    [https://arxiv.org/abs/2403.07688](https://arxiv.org/abs/2403.07688)

    重新评估深度神经网络中的死亡神经元现象，提出了Demon Pruning（DemP）方法，通过控制死亡神经元的产生，动态实现网络稀疏化。

    

    在训练深度神经网络时，$\textit{死亡神经元}$现象——在训练期间变得不活跃或饱和，输出为零的单元—传统上被视为不可取的，与优化挑战有关，并导致在不断学习的情况下丧失可塑性。本文重新评估了这一现象，专注于稀疏性和修剪。通过系统地探索各种超参数配置对死亡神经元的影响，我们揭示了它们有助于促进简单而有效的结构化修剪算法的潜力。我们提出了$\textit{Demon Pruning}$（DemP），一种控制死亡神经元扩张，动态导致网络稀疏性的方法。通过在活跃单元上注入噪声和采用单周期调度正则化策略的组合，DemP因其简单性和广泛适用性而脱颖而出。在CIFAR10上的实验中...

    arXiv:2403.07688v1 Announce Type: cross  Abstract: When training deep neural networks, the phenomenon of $\textit{dying neurons}$ $\unicode{x2013}$units that become inactive or saturated, output zero during training$\unicode{x2013}$ has traditionally been viewed as undesirable, linked with optimization challenges, and contributing to plasticity loss in continual learning scenarios. In this paper, we reassess this phenomenon, focusing on sparsity and pruning. By systematically exploring the impact of various hyperparameter configurations on dying neurons, we unveil their potential to facilitate simple yet effective structured pruning algorithms. We introduce $\textit{Demon Pruning}$ (DemP), a method that controls the proliferation of dead neurons, dynamically leading to network sparsity. Achieved through a combination of noise injection on active units and a one-cycled schedule regularization strategy, DemP stands out for its simplicity and broad applicability. Experiments on CIFAR10 an
    
[^2]: 兰花：灵活且数据相关的卷积用于序列建模

    Orchid: Flexible and Data-Dependent Convolution for Sequence Modeling

    [https://arxiv.org/abs/2402.18508](https://arxiv.org/abs/2402.18508)

    兰花引入了一种新的数据相关卷积机制，通过动态调整卷积核，实现了高表达能力和计算效率的平衡。

    

    在深度学习不断发展的格局中，平衡表达能力与计算效率的模型已经变得至关重要。本文介绍了一种名为兰花（Orchid）的新型架构，通过包含一种新的数据相关卷积机制来重新构想序列建模。兰花旨在解决传统注意力机制固有的限制，特别是它们的二次复杂性，同时不影响捕捉远程依赖性和上下文学习的能力。兰花的核心是数据相关卷积层，它利用专门的条件化神经网络根据输入数据动态调整其卷积核。我们设计了两个简单的条件化网络，以在自适应卷积操作中维持平移等变性。数据相关卷积核的动态特性，加上门控操作，赋予了兰花高表达能力，同时维持了计算效率。

    arXiv:2402.18508v1 Announce Type: new  Abstract: In the rapidly evolving landscape of deep learning, the quest for models that balance expressivity with computational efficiency has never been more critical. This paper introduces Orchid, a novel architecture that reimagines sequence modeling by incorporating a new data-dependent convolution mechanism. Orchid is designed to address the inherent limitations of traditional attention mechanisms, particularly their quadratic complexity, without compromising the ability to capture long-range dependencies and in-context learning. At the core of Orchid lies the data-dependent convolution layer, which dynamically adjusts its kernel conditioned on input data using a dedicated conditioning neural network. We design two simple conditioning networks that maintain shift equivariance in the adaptive convolution operation. The dynamic nature of data-dependent convolution kernel, coupled with gating operations, grants Orchid high expressivity while mai
    
[^3]: 添加非参数回归的随机梯度下降

    Stochastic Gradient Descent for Additive Nonparametric Regression

    [https://arxiv.org/abs/2401.00691](https://arxiv.org/abs/2401.00691)

    本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。

    

    本文介绍了一种用于训练加性模型的迭代算法，该算法具有良好的内存存储和计算要求。该算法可以看作是对组件函数的截断基扩展的系数应用随机梯度下降的函数对应物。我们证明了得到的估计量满足一个奥拉克不等式，允许模型错误规范。在规范很好的情况下，通过在训练的三个不同阶段仔细选择学习率，我们证明了其风险在数据维度和训练样本大小的依赖方面是最小和最优的。通过在两个实际数据集上将该方法与传统的反向拟合进行比较，我们进一步说明了计算优势。

    This paper introduces an iterative algorithm for training additive models that enjoys favorable memory storage and computational requirements. The algorithm can be viewed as the functional counterpart of stochastic gradient descent, applied to the coefficients of a truncated basis expansion of the component functions. We show that the resulting estimator satisfies an oracle inequality that allows for model mis-specification. In the well-specified setting, by choosing the learning rate carefully across three distinct stages of training, we demonstrate that its risk is minimax optimal in terms of the dependence on the dimensionality of the data and the size of the training sample. We further illustrate the computational benefits by comparing the approach with traditional backfitting on two real-world datasets.
    
[^4]: 集成模型是否一直在不断进步？

    Are Ensembles Getting Better all the Time?

    [https://arxiv.org/abs/2311.17885](https://arxiv.org/abs/2311.17885)

    只有当考虑的损失函数为凸函数时，集成模型一直在变得更好，当损失函数为非凸函数时，好模型的集成变得更好，坏模型的集成变得更糟。

    

    集成方法结合了几个基础模型的预测。本研究探讨了是否始终将更多模型纳入集成会提升其平均性能。这个问题取决于所考虑的集成类型，以及选择的预测度量。我们专注于所有集成成员被预期表现相同的情况，这是几种流行方法（如随机森林或深度集成）的情况。在这种设定下，我们表明，只有当考虑的损失函数为凸函数时，集成才会一直变得更好。更具体地说，在这种情况下，集成的平均损失是模型数量的减函数。当损失函数为非凸函数时，我们展示了一系列结果，可以总结为：好模型的集成会变得更好，坏模型的集成会变得更糟。为此，我们证明了关于尾概率单调性的新结果。

    arXiv:2311.17885v2 Announce Type: replace-cross  Abstract: Ensemble methods combine the predictions of several base models. We study whether or not including more models always improves their average performance. This question depends on the kind of ensemble considered, as well as the predictive metric chosen. We focus on situations where all members of the ensemble are a priori expected to perform as well, which is the case of several popular methods such as random forests or deep ensembles. In this setting, we show that ensembles are getting better all the time if, and only if, the considered loss function is convex. More precisely, in that case, the average loss of the ensemble is a decreasing function of the number of models. When the loss function is nonconvex, we show a series of results that can be summarised as: ensembles of good models keep getting better, and ensembles of bad models keep getting worse. To this end, we prove a new result on the monotonicity of tail probabiliti
    
[^5]: 基于内容的视频流媒体平台推荐引擎

    Content-based Recommendation Engine for Video Streaming Platform. (arXiv:2308.08406v1 [cs.IR])

    [http://arxiv.org/abs/2308.08406](http://arxiv.org/abs/2308.08406)

    本文提出了一种基于内容的推荐引擎，通过计算文档中单词的相关性和使用余弦相似度方法来推荐视频给用户。同时，还通过计算精确率、召回率和F1得分来评估引擎的性能。

    

    推荐引擎使用机器学习算法向用户建议内容、产品或服务。本文提出了一种基于内容的推荐引擎，根据用户之前的兴趣和选择，向用户提供视频推荐。我们将使用TF-IDF文本向量化方法来确定文档中单词的相关性。然后通过计算它们之间的余弦相似度，找出每个内容之间的相似性。最后，根据得到的相似度分数值，引擎将向用户推荐视频。另外，我们将通过计算精确率、召回率和F1得分来衡量引擎的性能。

    Recommendation engine suggest content, product or services to the user by using machine learning algorithm. This paper proposed a content-based recommendation engine for providing video suggestion to the user based on their previous interests and choices. We will use TF-IDF text vectorization method to determine the relevance of words in a document. Then we will find out the similarity between each content by calculating cosine similarity between them. Finally, engine will recommend videos to the users based on the obtained similarity score value. In addition, we will measure the engine's performance by computing precision, recall, and F1 core of the proposed system.
    
[^6]: 对高阶SDE模拟的Lévy区域进行生成建模

    Generative Modelling of L\'{e}vy Area for High Order SDE Simulation. (arXiv:2308.02452v1 [stat.ML])

    [http://arxiv.org/abs/2308.02452](http://arxiv.org/abs/2308.02452)

    本文提出了一种基于深度学习的模型LévyGAN，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩，解决了非高斯性质下的抽样困难问题。

    

    众所周知，当数值模拟SDE的解时，要实现强收敛速率超过O(\sqrt{h})（其中h为步长），需要使用某些布朗运动的迭代积分，通常称为其“Lévy区域”。然而，由于其非高斯性质，对于d维布朗运动（d>2），目前没有快速近似抽样算法。本文提出了LévyGAN，一种基于深度学习的模型，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩。我们的生成器采用经过量身定制的GNN-inspired架构，强制输出分布与条件变量之间的正确依赖结构。此外，我们还结合了基于特征函数的数学原理的判别性归一化操作。

    It is well known that, when numerically simulating solutions to SDEs, achieving a strong convergence rate better than O(\sqrt{h}) (where h is the step size) requires the use of certain iterated integrals of Brownian motion, commonly referred to as its "L\'{e}vy areas". However, these stochastic integrals are difficult to simulate due to their non-Gaussian nature and for a d-dimensional Brownian motion with d > 2, no fast almost-exact sampling algorithm is known.  In this paper, we propose L\'{e}vyGAN, a deep-learning-based model for generating approximate samples of L\'{e}vy area conditional on a Brownian increment. Due to our "Bridge-flipping" operation, the output samples match all joint and conditional odd moments exactly. Our generator employs a tailored GNN-inspired architecture, which enforces the correct dependency structure between the output distribution and the conditioning variable. Furthermore, we incorporate a mathematically principled characteristic-function based discrim
    
[^7]: 机器学习用于期权定价：对网络结构的实证研究

    Machine learning for option pricing: an empirical investigation of network architectures. (arXiv:2307.07657v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.07657](http://arxiv.org/abs/2307.07657)

    广义高速公路网络结构在期权定价问题中的应用表现出更高的准确性和更短的训练时间。

    

    本文考虑了使用适当的输入数据（模型参数）和相应输出数据（期权价格或隐含波动率）来学习期权价格或隐含波动率的监督学习问题。大部分相关文献都使用（普通的）前馈神经网络结构来连接用于学习将输入映射到输出的神经元。在本文中，受到图像分类方法和用于偏微分方程机器学习方法的最新进展的启发，我们通过实证研究来探究网络结构的选择如何影响机器学习算法的精确度和训练时间。我们发现，在期权定价问题中，我们主要关注Black-Scholes和Heston模型，广义高速公路网络结构相较于其他变体在均方误差和训练时间方面表现更好。此外，在计算隐含波动率方面，

    We consider the supervised learning problem of learning the price of an option or the implied volatility given appropriate input data (model parameters) and corresponding output data (option prices or implied volatilities). The majority of articles in this literature considers a (plain) feed forward neural network architecture in order to connect the neurons used for learning the function mapping inputs to outputs. In this article, motivated by methods in image classification and recent advances in machine learning methods for PDEs, we investigate empirically whether and how the choice of network architecture affects the accuracy and training time of a machine learning algorithm. We find that for option pricing problems, where we focus on the Black--Scholes and the Heston model, the generalized highway network architecture outperforms all other variants, when considering the mean squared error and the training time as criteria. Moreover, for the computation of the implied volatility, a
    
[^8]: HiGen：层次图生成网络

    HiGen: Hierarchical Graph Generative Networks. (arXiv:2305.19337v1 [cs.LG])

    [http://arxiv.org/abs/2305.19337](http://arxiv.org/abs/2305.19337)

    HiGen是一种新颖的图形生成网络，能够以粗到细的方式捕捉图形的层次结构，并使用多项式分布来生成具有整数值的子图的边权。它能够有效地捕捉图形的局部和全局属性，并实现了最先进的性能。

    

    大多数真实世界的图表现出层次结构，这通常被现有的图形生成方法所忽视。为了解决这个限制，我们提出了一种新颖的图形生成网络，能够以粗到细的方式捕捉图形的层次结构并成功地生成图形子结构。在每个层次上，该模型并行生成社区，使用独立的模型预测社区之间的跨边。这种模块化方法使生成的图形网络高度可扩展。此外，我们用多项式分布建模层次图形的输出分布，并针对此分布推导了递归分解，使我们能够以自回归的方式生成具有整数值的子图的边权。实证研究证明，所提出的生成模型能够有效地捕捉图形的局部和全局属性，并实现了最先进的性能。

    Most real-world graphs exhibit a hierarchical structure, which is often overlooked by existing graph generation methods. To address this limitation, we propose a novel graph generative network that captures the hierarchical nature of graphs and successively generates the graph sub-structures in a coarse-to-fine fashion. At each level of hierarchy, this model generates communities in parallel, followed by the prediction of cross-edges between communities using a separate model. This modular approach results in a highly scalable graph generative network. Moreover, we model the output distribution of edges in the hierarchical graph with a multinomial distribution and derive a recursive factorization for this distribution, enabling us to generate sub-graphs with integer-valued edge weights in an autoregressive approach. Empirical studies demonstrate that the proposed generative model can effectively capture both local and global properties of graphs and achieves state-of-the-art performanc
    
[^9]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    
[^10]: 预条件对超参化低秩矩阵感知的影响

    The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing. (arXiv:2302.01186v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01186](http://arxiv.org/abs/2302.01186)

    该研究提出了ScaledGD(𝜆)方法，相较于传统梯度下降法更加鲁棒，并且在处理低秩矩阵感知问题时具有很好的表现。

    

    本文提出了ScaledGD(𝜆)方法来解决低秩矩阵感知中矩阵可能病态以及真实秩未知的问题。该方法使用超参式表示，从一个小的随机初始化开始，通过使用特定形式的阻尼预条件梯度下降来对抗超参化和病态曲率的影响。与基准梯度下降（GD）相比，尽管预处理需要轻微的计算开销，但ScaledGD（𝜆）在面对病态问题时表现出了出色的鲁棒性。在高斯设计下，ScaledGD($\lambda$) 会在仅迭代数对数级别的情况下，以线性速率收敛到真实的低秩矩阵。

    We propose $\textsf{ScaledGD($\lambda$)}$, a preconditioned gradient descent method to tackle the low-rank matrix sensing problem when the true rank is unknown, and when the matrix is possibly ill-conditioned. Using overparametrized factor representations, $\textsf{ScaledGD($\lambda$)}$ starts from a small random initialization, and proceeds by gradient descent with a specific form of damped preconditioning to combat bad curvatures induced by overparameterization and ill-conditioning. At the expense of light computational overhead incurred by preconditioners, $\textsf{ScaledGD($\lambda$)}$ is remarkably robust to ill-conditioning compared to vanilla gradient descent ($\textsf{GD}$) even with overprameterization. Specifically, we show that, under the Gaussian design, $\textsf{ScaledGD($\lambda$)}$ converges to the true low-rank matrix at a constant linear rate after a small number of iterations that scales only logarithmically with respect to the condition number and the problem dimensi
    

