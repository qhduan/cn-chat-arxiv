# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SteinGen: Generating Fidelitous and Diverse Graph Samples](https://arxiv.org/abs/2403.18578) | SteinGen是一种生成高质量图样本的新方法，结合了Stein方法和MCMC动力学，适用于只有一次观察到的图形，避免了参数估计的需求。 |
| [^2] | [Federated Bayesian Deep Learning: The Application of Statistical Aggregation Methods to Bayesian Models](https://arxiv.org/abs/2403.15263) | 该论文研究了联邦贝叶斯深度学习的方法，旨在解决在现代深度学习模型中传达认识不确定性的挑战。 |
| [^3] | [An Evaluation of Real-time Adaptive Sampling Change Point Detection Algorithm using KCUSUM](https://arxiv.org/abs/2402.10291) | KCUSUM算法是一种非参数扩展算法，用于在高容量数据情景下实时检测突变变化，相比于现有算法，其能够更灵活地在在线环境中进行变点检测。 |
| [^4] | [Self-Correcting Self-Consuming Loops for Generative Model Training](https://arxiv.org/abs/2402.07087) | 本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。 |
| [^5] | [On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis](https://arxiv.org/abs/2402.04520) | 通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。 |
| [^6] | [Plug-and-Play image restoration with Stochastic deNOising REgularization](https://arxiv.org/abs/2402.01779) | 本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。 |
| [^7] | [Early warning via transitions in latent stochastic dynamical systems.](http://arxiv.org/abs/2309.03842) | 本研究提出了一种基于定向异性扩散图的方法，通过捕捉低维流形中的潜在演化动态，能够有效提取早期警报信号来检测复杂系统或高维观测数据中的动力学转变，并在真实的脑电图数据上得到了验证。 |
| [^8] | [The Underlying Scaling Laws and Universal Statistical Structure of Complex Datasets.](http://arxiv.org/abs/2306.14975) | 本文研究了复杂数据集中的底层缩放定律和普适统计结构。通过将数据类比为物理系统，并应用统计物理学和随机矩阵理论的方法，揭示了特征-特征协方差矩阵的局部和全局特征值统计量的规律。研究发现，在无关随机数据和真实数据之间存在显著差异，并且可以通过引入长程相关性完全恢复缩放行为。同时，生成的数据和真实世界数据都属于混沌系统，并在较小的数据集大小上即可体现随机矩阵理论的统计行为。 |
| [^9] | [Maximally Machine-Learnable Portfolios.](http://arxiv.org/abs/2306.05568) | 本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。 |
| [^10] | [Difference of Submodular Minimization via DC Programming.](http://arxiv.org/abs/2305.11046) | 本文介绍了一种新的算法，利用DC规划算法来解决子模最小化问题，并证明收敛性质比现有算法更全面，同时在语音特征选择和文档摘要等应用中取得更好的性能。 |
| [^11] | [Bipartite Mixed Membership Distribution-Free Model. A novel model for community detection in overlapping bipartite weighted networks.](http://arxiv.org/abs/2211.00912) | 提出了一种无分布双分块混合成员模型（BiMMDF），可用于重叠双分块加权网络的社区发现，并可以模拟重叠双分块符号网络。该模型的估计具有一致性保证和理论分离条件，并可提高在合成网络和现实网络应用中的性能。 |
| [^12] | [Mixed Membership Distribution-Free Model.](http://arxiv.org/abs/2112.04389) | 本文提出了一种混合成员无分布模型，用于重叠加权网络的社群检测，支持节点所属多个社群和有限实数权值。提出的模型可以推广到之前的模型，包括混合成员随机块模型，并支持具有潜在社群结构的重叠符号网络的生成。我们使用高效谱算法估计模型下的社群成员资格，并提出了模糊加权模块度来评估重叠加权网络的社群检测质量并确定加权网络社群数量。 |
| [^13] | [Finding Outliers in Gaussian Model-Based Clustering.](http://arxiv.org/abs/1907.01136) | 研究提出了一种修剪异常值的算法，该算法删除最不可能出现的数据点，然后用符合参考分布的对数似然度进行修剪，从而固有估计异常值的数量。 |

# 详细

[^1]: SteinGen: 生成忠实和多样化的图样本

    SteinGen: Generating Fidelitous and Diverse Graph Samples

    [https://arxiv.org/abs/2403.18578](https://arxiv.org/abs/2403.18578)

    SteinGen是一种生成高质量图样本的新方法，结合了Stein方法和MCMC动力学，适用于只有一次观察到的图形，避免了参数估计的需求。

    

    生成保留特征结构并促进样本多样性的图形可能具有挑战性，特别是当图形观察数量较少时。在这里，我们解决了仅从一个观察到的图形生成图形的问题。通过在图形的设置中以指数随机图形模型的形式表达，我们提出的生成过程SteinGen结合了Stein方法和基于MCMC的马尔可夫动力学的思想，该动力学基于目标模型的Stein算子。SteinGen使用与e相关联的Glauber动力学

    arXiv:2403.18578v1 Announce Type: cross  Abstract: Generating graphs that preserve characteristic structures while promoting sample diversity can be challenging, especially when the number of graph observations is small. Here, we tackle the problem of graph generation from only one observed graph. The classical approach of graph generation from parametric models relies on the estimation of parameters, which can be inconsistent or expensive to compute due to intractable normalisation constants. Generative modelling based on machine learning techniques to generate high-quality graph samples avoids parameter estimation but usually requires abundant training samples. Our proposed generating procedure, SteinGen, which is phrased in the setting of graphs as realisations of exponential random graph models, combines ideas from Stein's method and MCMC by employing Markovian dynamics which are based on a Stein operator for the target model. SteinGen uses the Glauber dynamics associated with an e
    
[^2]: 联邦贝叶斯深度学习：统计聚合方法应用于贝叶斯模型

    Federated Bayesian Deep Learning: The Application of Statistical Aggregation Methods to Bayesian Models

    [https://arxiv.org/abs/2403.15263](https://arxiv.org/abs/2403.15263)

    该论文研究了联邦贝叶斯深度学习的方法，旨在解决在现代深度学习模型中传达认识不确定性的挑战。

    

    联邦学习(FL)是一种训练机器学习模型的方法，利用多个分布式数据集，同时保持数据隐私和减少与共享本地数据集相关的通信成本。已经开发了聚合策略，用于整合或融合分布式确定性模型的权重和偏差；然而，现代确定性深度学习（DL）模型通常校准不佳，缺乏在预测中传达一种认识不确定性的能力，这对遥感平台和安全关键应用是理想的。相反，贝叶斯DL模型通常校准良好，能够量化和传达一种认识不确定性的能力以及具有竞争力的预测准确性。不幸的是，因为贝叶斯DL模型中的权重和偏差由概率分布定义，所以简单应用聚合方法是困难的。

    arXiv:2403.15263v1 Announce Type: new  Abstract: Federated learning (FL) is an approach to training machine learning models that takes advantage of multiple distributed datasets while maintaining data privacy and reducing communication costs associated with sharing local datasets. Aggregation strategies have been developed to pool or fuse the weights and biases of distributed deterministic models; however, modern deterministic deep learning (DL) models are often poorly calibrated and lack the ability to communicate a measure of epistemic uncertainty in prediction, which is desirable for remote sensing platforms and safety-critical applications. Conversely, Bayesian DL models are often well calibrated and capable of quantifying and communicating a measure of epistemic uncertainty along with a competitive prediction accuracy. Unfortunately, because the weights and biases in Bayesian DL models are defined by a probability distribution, simple application of the aggregation methods associa
    
[^3]: 使用KCUSUM算法评估实时自适应采样变点检测

    An Evaluation of Real-time Adaptive Sampling Change Point Detection Algorithm using KCUSUM

    [https://arxiv.org/abs/2402.10291](https://arxiv.org/abs/2402.10291)

    KCUSUM算法是一种非参数扩展算法，用于在高容量数据情景下实时检测突变变化，相比于现有算法，其能够更灵活地在在线环境中进行变点检测。

    

    从科学模拟数据流中实时检测突变变化是一项具有挑战性的任务，要求部署准确和高效的算法。本研究引入了基于核的累积和（KCUSUM）算法，一种传统累积和（CUSUM）方法的非参数扩展，以其在较少限制条件下在线变点检测方面的有效性而备受关注。

    arXiv:2402.10291v1 Announce Type: new  Abstract: Detecting abrupt changes in real-time data streams from scientific simulations presents a challenging task, demanding the deployment of accurate and efficient algorithms. Identifying change points in live data stream involves continuous scrutiny of incoming observations for deviations in their statistical characteristics, particularly in high-volume data scenarios. Maintaining a balance between sudden change detection and minimizing false alarms is vital. Many existing algorithms for this purpose rely on known probability distributions, limiting their feasibility. In this study, we introduce the Kernel-based Cumulative Sum (KCUSUM) algorithm, a non-parametric extension of the traditional Cumulative Sum (CUSUM) method, which has gained prominence for its efficacy in online change point detection under less restrictive conditions. KCUSUM splits itself by comparing incoming samples directly with reference samples and computes a statistic gr
    
[^4]: 自我纠正自我消耗循环用于生成模型训练

    Self-Correcting Self-Consuming Loops for Generative Model Training

    [https://arxiv.org/abs/2402.07087](https://arxiv.org/abs/2402.07087)

    本论文研究了使用合成数据进行生成模型训练时可能出现的自我消耗循环问题，并提出了一种通过引入理想的修正函数来稳定训练的方法。同时，我们还提出了自我修正函数来近似理想的修正函数，并通过实验证实了其有效性。

    

    随着合成数据在互联网上的质量越来越高以及数量不断增加，机器学习模型越来越多地在人工和机器生成的数据的混合上进行训练。尽管使用合成数据进行表征学习的成功案例有很多，但是在生成模型训练中使用合成数据会产生"自我消耗循环"，这可能导致训练不稳定甚至崩溃，除非满足某些条件。我们的论文旨在稳定自我消耗的生成模型训练。我们的理论结果表明，通过引入一个理想的修正函数，将数据点映射为更有可能来自真实数据分布的样本，可以使自我消耗循环的稳定性呈指数增加。然后，我们提出了自我修正函数，它依赖于专家知识（例如，编程在模拟器中的物理定律），并且旨在自动且大规模地近似理想的修正函数。我们通过实验证实了自我纠正自我消耗循环在生成模型训练中的有效性。

    As synthetic data becomes higher quality and proliferates on the internet, machine learning models are increasingly trained on a mix of human- and machine-generated data. Despite the successful stories of using synthetic data for representation learning, using synthetic data for generative model training creates "self-consuming loops" which may lead to training instability or even collapse, unless certain conditions are met. Our paper aims to stabilize self-consuming generative model training. Our theoretical results demonstrate that by introducing an idealized correction function, which maps a data point to be more likely under the true data distribution, self-consuming loops can be made exponentially more stable. We then propose self-correction functions, which rely on expert knowledge (e.g. the laws of physics programmed in a simulator), and aim to approximate the idealized corrector automatically and at scale. We empirically validate the effectiveness of self-correcting self-consum
    
[^5]: 关于现代Hopfield模型计算限制的一个细粒度复杂性分析

    On Computational Limits of Modern Hopfield Models: A Fine-Grained Complexity Analysis

    [https://arxiv.org/abs/2402.04520](https://arxiv.org/abs/2402.04520)

    通过细粒度复杂性分析，我们研究了现代Hopfield模型的记忆检索计算限制，发现了一种基于模式范数的相变行为，并且建立了有效变体的上界条件。使用低秩逼近的方法，我们提供了有效构造的示例，同时证明了计算时间下界、记忆检索误差界和指数记忆容量。

    

    我们从细粒度复杂性分析的角度研究了现代Hopfield模型的记忆检索动力学的计算限制。我们的主要贡献是基于模式的范数对所有可能的现代Hopfield模型的效率进行相变行为的刻画。具体来说，我们建立了对输入查询模式和记忆模式的范数的上界标准。仅在这个标准之下，假设满足Strong Exponential Time Hypothesis (SETH)，存在子二次（高效）变体的现代Hopfield模型。为了展示我们的理论，当有效标准成立时，我们提供了现代Hopfield模型使用低秩逼近的有效构造的正式示例。这包括一个计算时间的下界导出，与$\Max\{$存储的记忆模式数量，输入查询序列的长度$\}$线性缩放。此外，我们证明了记忆检索误差界和指数记忆容量。

    We investigate the computational limits of the memory retrieval dynamics of modern Hopfield models from the fine-grained complexity analysis. Our key contribution is the characterization of a phase transition behavior in the efficiency of all possible modern Hopfield models based on the norm of patterns. Specifically, we establish an upper bound criterion for the norm of input query patterns and memory patterns. Only below this criterion, sub-quadratic (efficient) variants of the modern Hopfield model exist, assuming the Strong Exponential Time Hypothesis (SETH). To showcase our theory, we provide a formal example of efficient constructions of modern Hopfield models using low-rank approximation when the efficient criterion holds. This includes a derivation of a lower bound on the computational time, scaling linearly with $\Max\{$# of stored memory patterns, length of input query sequence$\}$. In addition, we prove its memory retrieval error bound and exponential memory capacity.
    
[^6]: 带有随机去噪正则化的即插即用图像恢复

    Plug-and-Play image restoration with Stochastic deNOising REgularization

    [https://arxiv.org/abs/2402.01779](https://arxiv.org/abs/2402.01779)

    本论文提出了一种新的即插即用图像恢复框架，称为随机去噪正则化（SNORE）。该框架在恰当噪声水平的图像上应用去噪器，并基于随机正则化提供了解决病态逆问题的随机梯度下降算法。实验结果表明，SNORE在去模糊和修复任务中与最先进的方法具有竞争力。

    

    即插即用（PnP）算法是一类迭代算法，通过结合物理模型和深度神经网络进行正则化来解决图像反演问题。尽管这些算法能够产生令人印象深刻的图像恢复结果，但它们依赖于在迭代过程中越来越少噪音的图像上的一种非标准的去噪器使用方法，这与基于扩散模型（DM）的最新算法相矛盾，在这些算法中，去噪器仅应用于重新加噪的图像上。我们提出了一种新的PnP框架，称为随机去噪正则化（SNORE），它仅在噪声水平适当的图像上应用去噪器。它基于显式的随机正则化，从而导致了一种解决病态逆问题的随机梯度下降算法。我们提供了该算法及其退火扩展的收敛分析。在实验上，我们证明SNORE在去模糊和修复任务上与最先进的方法相竞争。

    Plug-and-Play (PnP) algorithms are a class of iterative algorithms that address image inverse problems by combining a physical model and a deep neural network for regularization. Even if they produce impressive image restoration results, these algorithms rely on a non-standard use of a denoiser on images that are less and less noisy along the iterations, which contrasts with recent algorithms based on Diffusion Models (DM), where the denoiser is applied only on re-noised images. We propose a new PnP framework, called Stochastic deNOising REgularization (SNORE), which applies the denoiser only on images with noise of the adequate level. It is based on an explicit stochastic regularization, which leads to a stochastic gradient descent algorithm to solve ill-posed inverse problems. A convergence analysis of this algorithm and its annealing extension is provided. Experimentally, we prove that SNORE is competitive with respect to state-of-the-art methods on deblurring and inpainting tasks, 
    
[^7]: 隐性随机动力学系统中的转向预警

    Early warning via transitions in latent stochastic dynamical systems. (arXiv:2309.03842v1 [stat.ML])

    [http://arxiv.org/abs/2309.03842](http://arxiv.org/abs/2309.03842)

    本研究提出了一种基于定向异性扩散图的方法，通过捕捉低维流形中的潜在演化动态，能够有效提取早期警报信号来检测复杂系统或高维观测数据中的动力学转变，并在真实的脑电图数据上得到了验证。

    

    在许多实际应用中，如基因突变、脑疾病、自然灾害、金融危机和工程可靠性，对复杂系统或高维观测数据中的动力学转变进行早期警报是至关重要的。为了有效提取早期警报信号，我们开发了一种新方法：定向异性扩散图，它捕捉了低维流形中的潜在演化动态。将该方法应用于真实的脑电图（EEG）数据，我们成功找到了适当的有效坐标，并推导出能够检测状态转变中临界点的早期警报信号。我们的方法将潜在动态与原始数据集联系起来。通过数值实验证明了该框架在密度和转变概率等方面的准确性和有效性。结果表明，第二个坐标在各种评估指标中保持有意义的信息。

    Early warnings for dynamical transitions in complex systems or high-dimensional observation data are essential in many real world applications, such as gene mutation, brain diseases, natural disasters, financial crises, and engineering reliability. To effectively extract early warning signals, we develop a novel approach: the directed anisotropic diffusion map that captures the latent evolutionary dynamics in low-dimensional manifold. Applying the methodology to authentic electroencephalogram (EEG) data, we successfully find the appropriate effective coordinates, and derive early warning signals capable of detecting the tipping point during the state transition. Our method bridges the latent dynamics with the original dataset. The framework is validated to be accurate and effective through numerical experiments, in terms of density and transition probability. It is shown that the second coordinate holds meaningful information for critical transition in various evaluation metrics.
    
[^8]: 复杂数据集的底层缩放定律和普适统计结构

    The Underlying Scaling Laws and Universal Statistical Structure of Complex Datasets. (arXiv:2306.14975v1 [cs.LG])

    [http://arxiv.org/abs/2306.14975](http://arxiv.org/abs/2306.14975)

    本文研究了复杂数据集中的底层缩放定律和普适统计结构。通过将数据类比为物理系统，并应用统计物理学和随机矩阵理论的方法，揭示了特征-特征协方差矩阵的局部和全局特征值统计量的规律。研究发现，在无关随机数据和真实数据之间存在显著差异，并且可以通过引入长程相关性完全恢复缩放行为。同时，生成的数据和真实世界数据都属于混沌系统，并在较小的数据集大小上即可体现随机矩阵理论的统计行为。

    

    我们研究了在真实世界的复杂数据集和人工生成的数据集中都出现的普遍特征。我们将数据类比为物理系统，并利用统计物理学和随机矩阵理论的工具揭示其底层结构。我们重点分析了特征-特征协方差矩阵，分析了其局部和全局特征值统计量。我们的主要观察结果是：(i) 大部分特征值呈现的幂律缩放在无相关随机数据和真实数据之间存在显著差异，(ii) 通过简单地引入长程相关性，可以完全恢复这种缩放行为到合成数据中，(iii) 从随机矩阵理论的角度看，生成的数据集和真实世界数据集属于同一个普适性类别，都是混沌系统而非可积系统，(iv) 预期的随机矩阵理论统计行为在相对较小的数据集大小上就已经在经验协方差矩阵中得到体现。

    We study universal traits which emerge both in real-world complex datasets, as well as in artificially generated ones. Our approach is to analogize data to a physical system and employ tools from statistical physics and Random Matrix Theory (RMT) to reveal their underlying structure. We focus on the feature-feature covariance matrix, analyzing both its local and global eigenvalue statistics. Our main observations are: (i) The power-law scalings that the bulk of its eigenvalues exhibit are vastly different for uncorrelated random data compared to real-world data, (ii) this scaling behavior can be completely recovered by introducing long range correlations in a simple way to the synthetic data, (iii) both generated and real-world datasets lie in the same universality class from the RMT perspective, as chaotic rather than integrable systems, (iv) the expected RMT statistical behavior already manifests for empirical covariance matrices at dataset sizes significantly smaller than those conv
    
[^9]: 最大机器学习组合的构建方法

    Maximally Machine-Learnable Portfolios. (arXiv:2306.05568v1 [econ.EM])

    [http://arxiv.org/abs/2306.05568](http://arxiv.org/abs/2306.05568)

    本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。

    

    对于股票回报，任何形式的可预测性都可以增强调整风险后的盈利能力。本文开发了一种协作机器学习算法，优化组合权重，以使得合成证券最大程度的可预测。具体来说，我们引入了MACE，Alternating Conditional Expectations的多元扩展，通过在方程的一侧使用随机森林和受限岭回归在另一侧实现了上述目标。相较于Lo和MacKinlay的最大可预测组合方法，本文有两个关键改进。第一，它适用于任何（非线性）预测算法和预测器集。第二，它可以处理大型组合。我们进行了日频和月频的实验，并发现在使用很少的条件信息时，可预测性和盈利能力显著增加。有趣的是，可预测性在好时和坏时都存在，并且MACE成功地导航了两者。

    When it comes to stock returns, any form of predictability can bolster risk-adjusted profitability. We develop a collaborative machine learning algorithm that optimizes portfolio weights so that the resulting synthetic security is maximally predictable. Precisely, we introduce MACE, a multivariate extension of Alternating Conditional Expectations that achieves the aforementioned goal by wielding a Random Forest on one side of the equation, and a constrained Ridge Regression on the other. There are two key improvements with respect to Lo and MacKinlay's original maximally predictable portfolio approach. First, it accommodates for any (nonlinear) forecasting algorithm and predictor set. Second, it handles large portfolios. We conduct exercises at the daily and monthly frequency and report significant increases in predictability and profitability using very little conditioning information. Interestingly, predictability is found in bad as well as good times, and MACE successfully navigates
    
[^10]: DC规划算法在子模最小化问题上的应用

    Difference of Submodular Minimization via DC Programming. (arXiv:2305.11046v1 [cs.LG])

    [http://arxiv.org/abs/2305.11046](http://arxiv.org/abs/2305.11046)

    本文介绍了一种新的算法，利用DC规划算法来解决子模最小化问题，并证明收敛性质比现有算法更全面，同时在语音特征选择和文档摘要等应用中取得更好的性能。

    

    在各种机器学习问题中，最小化两个子模（DS）函数的差异是一个自然产生的问题。虽然已经有人知道DS问题可以等价地转化为两个凸（DC）函数的差异最小化问题，但现有算法并没有充分利用这种联系。对于DC问题，一个经典的算法叫做DC算法（DCA）。我们介绍了DCA及其完整形式（CDCA）的变体，并将其应用于对应于DS最小化的DC程序中。我们扩展了DCA的现有收敛性质，并将它们与DS问题的收敛性质联系起来。我们的DCA结果与现有的DS算法满足相同的理论保证，同时提供了更完整的收敛性质描述。对于CDCA的情况，我们获得了更强的局部最小保证。我们的数字实验结果表明，我们提出的算法在两个应用——语音语料库选择特征优化和文档摘要中均优于现有的基线算法。

    Minimizing the difference of two submodular (DS) functions is a problem that naturally occurs in various machine learning problems. Although it is well known that a DS problem can be equivalently formulated as the minimization of the difference of two convex (DC) functions, existing algorithms do not fully exploit this connection. A classical algorithm for DC problems is called the DC algorithm (DCA). We introduce variants of DCA and its complete form (CDCA) that we apply to the DC program corresponding to DS minimization. We extend existing convergence properties of DCA, and connect them to convergence properties on the DS problem. Our results on DCA match the theoretical guarantees satisfied by existing DS algorithms, while providing a more complete characterization of convergence properties. In the case of CDCA, we obtain a stronger local minimality guarantee. Our numerical results show that our proposed algorithms outperform existing baselines on two applications: speech corpus sel
    
[^11]: 无分布双分块混合成员模型：一种新的用于重叠双分块加权网络社区发现的模型

    Bipartite Mixed Membership Distribution-Free Model. A novel model for community detection in overlapping bipartite weighted networks. (arXiv:2211.00912v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2211.00912](http://arxiv.org/abs/2211.00912)

    提出了一种无分布双分块混合成员模型（BiMMDF），可用于重叠双分块加权网络的社区发现，并可以模拟重叠双分块符号网络。该模型的估计具有一致性保证和理论分离条件，并可提高在合成网络和现实网络应用中的性能。

    

    近年来，对于重叠单分块无权重网络的混合成员建模和估计已经得到了广泛的研究。然而，据我们所知，没有模型适用于更一般的情况，即重叠双分块加权网络。为了填补这一空白，我们引入了一种新的模型，即无分布双分块混合成员模型（BiMMDF）。我们的模型允许邻接矩阵遵循任何分布，只要其期望具有与节点成员有关的块结构即可。特别地，BiMMDF可以模拟重叠双分块符号网络，并且是许多先前模型的扩展，包括流行的混合成员随机块模型。我们应用具有一致估计理论保证的高效算法来拟合BiMMDF。我们进一步探讨了不同分布的BiMMDF的分离条件。此外，我们还考虑了稀疏网络的缺失边缘。BiMMDF的优势在广泛的合成网络和现实网络应用中得到了展示。

    Modeling and estimating mixed memberships for overlapping unipartite un-weighted networks has been well studied in recent years. However, to our knowledge, there is no model for a more general case, the overlapping bipartite weighted networks. To close this gap, we introduce a novel model, the Bipartite Mixed Membership Distribution-Free (BiMMDF) model. Our model allows an adjacency matrix to follow any distribution as long as its expectation has a block structure related to node membership. In particular, BiMMDF can model overlapping bipartite signed networks and it is an extension of many previous models, including the popular mixed membership stochastic blcokmodels. An efficient algorithm with a theoretical guarantee of consistent estimation is applied to fit BiMMDF. We then obtain the separation conditions of BiMMDF for different distributions. Furthermore, we also consider missing edges for sparse networks. The advantage of BiMMDF is demonstrated in extensive synthetic networks an
    
[^12]: 混合成员无分布模型

    Mixed Membership Distribution-Free Model. (arXiv:2112.04389v4 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2112.04389](http://arxiv.org/abs/2112.04389)

    本文提出了一种混合成员无分布模型，用于重叠加权网络的社群检测，支持节点所属多个社群和有限实数权值。提出的模型可以推广到之前的模型，包括混合成员随机块模型，并支持具有潜在社群结构的重叠符号网络的生成。我们使用高效谱算法估计模型下的社群成员资格，并提出了模糊加权模块度来评估重叠加权网络的社群检测质量并确定加权网络社群数量。

    

    本文考虑在具有重叠加权网络中进行社群检测的问题，其中节点可以属于多个社群，边权可以是有限实数。为了对这样的复杂网络进行建模，我们提出了一个通用框架——混合成员无分布（MMDF）模型。MMDF没有对边权的分布约束，可以被视为一些先前模型的推广，包括著名的混合成员随机块模型。特别地，具有潜在社群结构的重叠符号网络也可以从我们的模型中生成。我们使用具有收敛率理论保证的高效谱算法来估计模型下的社群成员资格。我们还提出了模糊加权模块度来评估具有正负边权的重叠加权网络的社群检测质量。然后，我们提供了一种利用我们的fuzzy weighted modularity来确定加权网络社群数量的方法。

    We consider the problem of community detection in overlapping weighted networks, where nodes can belong to multiple communities and edge weights can be finite real numbers. To model such complex networks, we propose a general framework - the mixed membership distribution-free (MMDF) model. MMDF has no distribution constraints of edge weights and can be viewed as generalizations of some previous models, including the well-known mixed membership stochastic blockmodels. Especially, overlapping signed networks with latent community structures can also be generated from our model. We use an efficient spectral algorithm with a theoretical guarantee of convergence rate to estimate community memberships under the model. We also propose fuzzy weighted modularity to evaluate the quality of community detection for overlapping weighted networks with positive and negative edge weights. We then provide a method to determine the number of communities for weighted networks by taking advantage of our f
    
[^13]: 基于高斯模型的聚类中异常值的发现

    Finding Outliers in Gaussian Model-Based Clustering. (arXiv:1907.01136v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/1907.01136](http://arxiv.org/abs/1907.01136)

    研究提出了一种修剪异常值的算法，该算法删除最不可能出现的数据点，然后用符合参考分布的对数似然度进行修剪，从而固有估计异常值的数量。

    

    无监督分类或聚类常常受到异常值的影响。然而，在无监督分类中处理异常值的研究较少。目前，异常值算法可分为两大类：异常点包含方法和修剪方法，这些方法通常需要预先指定要删除的数据点的数量。本文利用样本马氏距离的贝塔分布导出了一个近似分布，用于有限高斯混合模型子集的对数似然度。提出了一种算法，该算法删除最不可能出现的数据点，即判定为异常值，直到对数似然度符合参考分布。这导致了一种固有估计异常值数量的修剪方法。

    Unsupervised classification, or clustering, is a problem often plagued by outliers, yet there is a paucity of work on handling outliers in unsupervised classification. Outlier algorithms tend to fall into two broad categories: outlier inclusion methods and trimming methods, which often require pre-specification of the number of points to remove. The fact that sample Mahalanobis distance is beta-distributed is used to derive an approximate distribution for the log-likelihoods of subset finite Gaussian mixture models. An algorithm is proposed that removes the least likely points, which are deemed outliers, until the log-likelihoods adhere to the reference distribution. This results in a trimming method which inherently estimates the number of outliers present.
    

