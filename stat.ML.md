# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sliced-Wasserstein Estimation with Spherical Harmonics as Control Variates](https://rss.arxiv.org/abs/2402.01493) | 这种方法提出了一种新的蒙特卡罗方法，使用球谐函数作为控制变量来近似计算切片华瑟斯坦距离。与蒙特卡罗相比，该方法具有更好的收敛速度和理论性质。 |
| [^2] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^3] | [Conformalized Adaptive Forecasting of Heterogeneous Trajectories](https://arxiv.org/abs/2402.09623) | 本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。 |
| [^4] | [Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization](https://arxiv.org/abs/2402.02746) | 标准 Gaussian 过程在高维贝叶斯优化中表现优秀，经验证据显示其在函数估计和协方差建模中克服了高维输入困难，比专门为高维优化设计的方法表现更好。 |
| [^5] | [Optimal Multi-Distribution Learning.](http://arxiv.org/abs/2312.05134) | 本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。 |
| [^6] | [MINDE: Mutual Information Neural Diffusion Estimation.](http://arxiv.org/abs/2310.09031) | MINDE是一种基于得分函数扩散模型的新方法，用于估计随机变量之间的互信息。该方法在准确性和自一致性测试方面优于其他文献中的主要替代方法。 |
| [^7] | [Stationarity without mean reversion: Improper Gaussian process regression and improper kernels.](http://arxiv.org/abs/2310.02877) | 本论文展示了使用具有无限方差的不恰当高斯过程先验来定义静止但不均值回归过程的可能性，并引入了一类特殊的不恰当核函数来实现此目的。 |
| [^8] | [Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences.](http://arxiv.org/abs/2308.14555) | 本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。 |
| [^9] | [Nonparametric regression using over-parameterized shallow ReLU neural networks.](http://arxiv.org/abs/2306.08321) | 本文表明，对于学习某些光滑函数类，具有适当权重限制或正则化的过参数化神经网络可以达到最小最优收敛速率。作者通过对过度参数化的神经网络进行了实验，成功证明在浅层ReLU神经网络中使用最小二乘估计值是最小化最优的。同时作者还得出了神经网络的本地Rademacher复杂度的一个新的上界。 |
| [^10] | [Some Fundamental Aspects about Lipschitz Continuity of Neural Network Functions.](http://arxiv.org/abs/2302.10886) | 本文深入研究和描述神经网络实现的函数的Lipschitz行为，在多种设置下进行实证研究，并揭示了神经网络函数Lipschitz连续性的基本和有趣的特性，其中最引人注目的是在Lipschitz常数的上限和下限中识别出了明显的双下降趋势。 |

# 详细

[^1]: 使用球谐函数作为控制变量的切片华瑟斯坦估计

    Sliced-Wasserstein Estimation with Spherical Harmonics as Control Variates

    [https://rss.arxiv.org/abs/2402.01493](https://rss.arxiv.org/abs/2402.01493)

    这种方法提出了一种新的蒙特卡罗方法，使用球谐函数作为控制变量来近似计算切片华瑟斯坦距离。与蒙特卡罗相比，该方法具有更好的收敛速度和理论性质。

    

    切片华瑟斯坦（SW）距离是概率测度之间的华瑟斯坦距离的平均值，结果为相关的一维投影的华瑟斯坦距离。因此，SW距离可以写成对球面上均匀测度的积分，并且可以使用蒙特卡罗框架来计算SW距离。球谐函数是球面上的多项式，它们构成了球面上可积函数集合的正交基。将这两个事实结合在一起，提出了一种新的蒙特卡罗方法，称为球谐控制变量（SHCV），用于使用球谐函数作为控制变量近似计算SW距离。结果表明，该方法具有良好的理论性质，例如在变量之间存在一定形式的线性依赖时，混合高斯测度的无误差特性。此外，与蒙特卡罗相比，得到了更快的收敛速度。

    The Sliced-Wasserstein (SW) distance between probability measures is defined as the average of the Wasserstein distances resulting for the associated one-dimensional projections. As a consequence, the SW distance can be written as an integral with respect to the uniform measure on the sphere and the Monte Carlo framework can be employed for calculating the SW distance. Spherical harmonics are polynomials on the sphere that form an orthonormal basis of the set of square-integrable functions on the sphere. Putting these two facts together, a new Monte Carlo method, hereby referred to as Spherical Harmonics Control Variates (SHCV), is proposed for approximating the SW distance using spherical harmonics as control variates. The resulting approach is shown to have good theoretical properties, e.g., a no-error property for Gaussian measures under a certain form of linear dependency between the variables. Moreover, an improved rate of convergence, compared to Monte Carlo, is established for g
    
[^2]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^3]: 多元轨迹的符合性自适应预测方法

    Conformalized Adaptive Forecasting of Heterogeneous Trajectories

    [https://arxiv.org/abs/2402.09623](https://arxiv.org/abs/2402.09623)

    本研究提出了一种新的符合性方法，通过结合在线符合性预测技术和解决回归中异方差性的方法，生成了同时预测边界，并能够可靠地覆盖新随机轨迹的整个路径。这种方法不仅有精确的有限样本保证，而且往往比之前的方法具有更丰富的预测结果。

    

    本文提出了一种新的符合性方法，用于生成同时预测边界，以具有足够高的概率覆盖新随机轨迹的整个路径。鉴于在运动规划应用中需要可靠的不确定性估计，其中不同物体的行为可能更或更少可预测，我们将来自单个和多个时间序列的在线符合性预测技术，以及解决回归中的异方差性的方法进行了融合。该解决方案既有原则性，提供了精确的有限样本保证，又有效，通常比先前的方法具有更丰富的预测结果。

    arXiv:2402.09623v1 Announce Type: cross  Abstract: This paper presents a new conformal method for generating simultaneous forecasting bands guaranteed to cover the entire path of a new random trajectory with sufficiently high probability. Prompted by the need for dependable uncertainty estimates in motion planning applications where the behavior of diverse objects may be more or less unpredictable, we blend different techniques from online conformal prediction of single and multiple time series, as well as ideas for addressing heteroscedasticity in regression. This solution is both principled, providing precise finite-sample guarantees, and effective, often leading to more informative predictions than prior methods.
    
[^4]: 标准 Gaussian 过程在高维贝叶斯优化中足以应对

    Standard Gaussian Process is All You Need for High-Dimensional Bayesian Optimization

    [https://arxiv.org/abs/2402.02746](https://arxiv.org/abs/2402.02746)

    标准 Gaussian 过程在高维贝叶斯优化中表现优秀，经验证据显示其在函数估计和协方差建模中克服了高维输入困难，比专门为高维优化设计的方法表现更好。

    

    长期以来，人们普遍认为使用标准 Gaussian 过程（GP）进行贝叶斯优化（BO），即标准 BO，在高维优化问题中效果不佳。这种观念可以部分归因于 Gaussian 过程在协方差建模和函数估计中对高维输入的困难。虽然这些担忧看起来合理，但缺乏支持这种观点的经验证据。本文系统地研究了在各种合成和真实世界基准问题上，使用标准 GP 回归进行高维优化的贝叶斯优化。令人惊讶的是，标准 GP 的表现始终位于最佳范围内，往往比专门为高维优化设计的现有 BO 方法表现更好。与刻板印象相反，我们发现标准 GP 可以作为学习高维目标函数的能力强大的代理。在没有强结构假设的情况下，使用标准 GP 进行 BO 可以获得非常好的性能。

    There has been a long-standing and widespread belief that Bayesian Optimization (BO) with standard Gaussian process (GP), referred to as standard BO, is ineffective in high-dimensional optimization problems. This perception may partly stem from the intuition that GPs struggle with high-dimensional inputs for covariance modeling and function estimation. While these concerns seem reasonable, empirical evidence supporting this belief is lacking. In this paper, we systematically investigated BO with standard GP regression across a variety of synthetic and real-world benchmark problems for high-dimensional optimization. Surprisingly, the performance with standard GP consistently ranks among the best, often outperforming existing BO methods specifically designed for high-dimensional optimization by a large margin. Contrary to the stereotype, we found that standard GP can serve as a capable surrogate for learning high-dimensional target functions. Without strong structural assumptions, BO wit
    
[^5]: 最优化多分布学习

    Optimal Multi-Distribution Learning. (arXiv:2312.05134v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.05134](http://arxiv.org/abs/2312.05134)

    本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。

    

    多分布学习（MDL）旨在学习一个共享模型，使得在k个不同的数据分布下，最小化最坏情况风险，已成为适应健壮性、公平性、多组合作等需求的统一框架。实现数据高效的MDL需要在学习过程中进行自适应采样，也称为按需采样。然而，最优样本复杂度的上下界之间存在较大差距。针对Vapnik-Chervonenkis（VC）维数为d的假设类，我们提出了一种新颖的算法，可生成一个ε-最优随机假设，其样本复杂度接近于（d+k）/ε^2（在某些对数因子中），与已知的最佳下界匹配。我们的算法思想和理论被进一步扩展，以适应Rademacher类。提出的算法是奥拉克尔高效的，仅仅访问假设类

    Multi-distribution learning (MDL), which seeks to learn a shared model that minimizes the worst-case risk across $k$ distinct data distributions, has emerged as a unified framework in response to the evolving demand for robustness, fairness, multi-group collaboration, etc. Achieving data-efficient MDL necessitates adaptive sampling, also called on-demand sampling, throughout the learning process. However, there exist substantial gaps between the state-of-the-art upper and lower bounds on the optimal sample complexity. Focusing on a hypothesis class of Vapnik-Chervonenkis (VC) dimension $d$, we propose a novel algorithm that yields an $varepsilon$-optimal randomized hypothesis with a sample complexity on the order of $(d+k)/\varepsilon^2$ (modulo some logarithmic factor), matching the best-known lower bound. Our algorithmic ideas and theory have been further extended to accommodate Rademacher classes. The proposed algorithms are oracle-efficient, which access the hypothesis class solely
    
[^6]: MINDE: 互信息神经扩散估计

    MINDE: Mutual Information Neural Diffusion Estimation. (arXiv:2310.09031v1 [cs.LG])

    [http://arxiv.org/abs/2310.09031](http://arxiv.org/abs/2310.09031)

    MINDE是一种基于得分函数扩散模型的新方法，用于估计随机变量之间的互信息。该方法在准确性和自一致性测试方面优于其他文献中的主要替代方法。

    

    在这项工作中，我们提出了一种估计随机变量之间互信息（MI）的新方法。我们的方法基于Girsanov定理的原创解释，允许我们使用基于得分的扩散模型来估计两个密度函数之间的Kullback-Leibler散度，该估计是它们得分函数之间的差异。作为副产品，我们的方法还能够估计随机变量的熵。借助这样的构建模块，我们提出了一种通用的测量MI的方法，该方法分为两个方向展开：一个使用条件扩散过程，另一个使用联合扩散过程，可以同时对两个随机变量进行建模。我们的结果来自于对我们方法的各种变体进行彻底的实验协议，表明我们的方法比文献中的主要替代方法更准确，尤其是对于具有挑战性的分布。此外，我们的方法通过了MI自一致性测试，包括...

    In this work we present a new method for the estimation of Mutual Information (MI) between random variables. Our approach is based on an original interpretation of the Girsanov theorem, which allows us to use score-based diffusion models to estimate the Kullback Leibler divergence between two densities as a difference between their score functions. As a by-product, our method also enables the estimation of the entropy of random variables. Armed with such building blocks, we present a general recipe to measure MI, which unfolds in two directions: one uses conditional diffusion process, whereas the other uses joint diffusion processes that allow simultaneous modelling of two random variables. Our results, which derive from a thorough experimental protocol over all the variants of our approach, indicate that our method is more accurate than the main alternatives from the literature, especially for challenging distributions. Furthermore, our methods pass MI self-consistency tests, includin
    
[^7]: 无均值回归：不恰当高斯过程回归和不恰当核

    Stationarity without mean reversion: Improper Gaussian process regression and improper kernels. (arXiv:2310.02877v1 [stat.ML])

    [http://arxiv.org/abs/2310.02877](http://arxiv.org/abs/2310.02877)

    本论文展示了使用具有无限方差的不恰当高斯过程先验来定义静止但不均值回归过程的可能性，并引入了一类特殊的不恰当核函数来实现此目的。

    

    高斯过程（GP）回归在机器学习应用中已经广泛流行。GP回归的行为取决于协方差函数的选择。在机器学习应用中，静止协方差函数是首选。然而，（非周期性的）静止协方差函数总是均值回归的，因此在应用于不通过到固定全局均值值的数据时可能表现出病态行为。在本文中，我们展示了使用具有无限方差的不恰当GP先验来定义静止但不均值回归过程是可能的。为此，我们引入了一大类只能在这种不恰当的范围内定义的不恰当核函数。具体地，我们引入了平滑行走核，它产生无限平滑的样本，以及一类不恰当的Matern核，它可以被定义为任意整数j倍可微。所得到的后验分布可以用解析的方式计算出来。

    Gaussian processes (GP) regression has gained substantial popularity in machine learning applications. The behavior of a GP regression depends on the choice of covariance function. Stationary covariance functions are favorite in machine learning applications. However, (non-periodic) stationary covariance functions are always mean reverting and can therefore exhibit pathological behavior when applied to data that does not relax to a fixed global mean value. In this paper, we show that it is possible to use improper GP prior with infinite variance to define processes that are stationary but not mean reverting. To this aim, we introduce a large class of improper kernels that can only be defined in this improper regime. Specifically, we introduce the Smooth Walk kernel, which produces infinitely smooth samples, and a family of improper Mat\'ern kernels, which can be defined to be $j$-times differentiable for any integer $j$. The resulting posterior distributions can be computed analyticall
    
[^8]: 循环神经网络在遍历数据序列上训练的核极限

    Kernel Limit of Recurrent Neural Networks Trained on Ergodic Data Sequences. (arXiv:2308.14555v1 [cs.LG])

    [http://arxiv.org/abs/2308.14555](http://arxiv.org/abs/2308.14555)

    本文研究了循环神经网络在遍历数据序列上训练时的核极限，利用数学方法对其渐近特性进行了描述，并证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。这对于理解和改进循环神经网络具有重要意义。

    

    本文开发了数学方法来描述循环神经网络（RNN）的渐近特性，其中隐藏单元的数量、序列中的数据样本、隐藏状态的更新和训练步骤同时趋于无穷大。对于具有简化权重矩阵的RNN，我们证明了RNN收敛到与随机代数方程的不动点耦合的无穷维ODE的解。分析需要解决RNN所特有的几个挑战。在典型的均场应用中（例如前馈神经网络），离散的更新量为$\mathcal{O}(\frac{1}{N})$，更新的次数为$\mathcal{O}(N)$。因此，系统可以表示为适当ODE/PDE的Euler逼近，当$N \rightarrow \infty$时收敛到该ODE/PDE。然而，RNN的隐藏层更新为$\mathcal{O}(1)$。因此，RNN不能表示为ODE/PDE的离散化和标准均场技术。

    Mathematical methods are developed to characterize the asymptotics of recurrent neural networks (RNN) as the number of hidden units, data samples in the sequence, hidden state updates, and training steps simultaneously grow to infinity. In the case of an RNN with a simplified weight matrix, we prove the convergence of the RNN to the solution of an infinite-dimensional ODE coupled with the fixed point of a random algebraic equation. The analysis requires addressing several challenges which are unique to RNNs. In typical mean-field applications (e.g., feedforward neural networks), discrete updates are of magnitude $\mathcal{O}(\frac{1}{N})$ and the number of updates is $\mathcal{O}(N)$. Therefore, the system can be represented as an Euler approximation of an appropriate ODE/PDE, which it will converge to as $N \rightarrow \infty$. However, the RNN hidden layer updates are $\mathcal{O}(1)$. Therefore, RNNs cannot be represented as a discretization of an ODE/PDE and standard mean-field tec
    
[^9]: 利用过参数化的浅层ReLU神经网络的非参数回归

    Nonparametric regression using over-parameterized shallow ReLU neural networks. (arXiv:2306.08321v1 [stat.ML])

    [http://arxiv.org/abs/2306.08321](http://arxiv.org/abs/2306.08321)

    本文表明，对于学习某些光滑函数类，具有适当权重限制或正则化的过参数化神经网络可以达到最小最优收敛速率。作者通过对过度参数化的神经网络进行了实验，成功证明在浅层ReLU神经网络中使用最小二乘估计值是最小化最优的。同时作者还得出了神经网络的本地Rademacher复杂度的一个新的上界。

    

    如果权重得到合适的限制或正则化，那么可以证明过度参数化的神经网络可以达到某些光滑函数类的学习最小最优收敛速率（最多对数因数）。具体地，我们考虑使用浅层ReLU神经网络的非参数回归来估计未知的$d$变量函数。假设回归函数是从具有光滑度$\alpha < (d+3)/2$的Holder空间或对应于浅层神经网络的变化空间中学习的，后者可以视为无限宽的神经网络。在这种情况下，我们证明了基于具有权重某些范数约束的浅层神经网络的最小二乘估计值是最小化最优的，如果网络宽度足够大。作为副产品，我们得到了一个新的和神经网络的本地Rademacher复杂度无关的上界，这可能是有独立兴趣的。

    It is shown that over-parameterized neural networks can achieve minimax optimal rates of convergence (up to logarithmic factors) for learning functions from certain smooth function classes, if the weights are suitably constrained or regularized. Specifically, we consider the nonparametric regression of estimating an unknown $d$-variate function by using shallow ReLU neural networks. It is assumed that the regression function is from the H\"older space with smoothness $\alpha<(d+3)/2$ or a variation space corresponding to shallow neural networks, which can be viewed as an infinitely wide neural network. In this setting, we prove that least squares estimators based on shallow neural networks with certain norm constraints on the weights are minimax optimal, if the network width is sufficiently large. As a byproduct, we derive a new size-independent bound for the local Rademacher complexity of shallow ReLU neural networks, which may be of independent interest.
    
[^10]: 神经网络函数的Lipschitz连续性的一些基本方面

    Some Fundamental Aspects about Lipschitz Continuity of Neural Network Functions. (arXiv:2302.10886v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10886](http://arxiv.org/abs/2302.10886)

    本文深入研究和描述神经网络实现的函数的Lipschitz行为，在多种设置下进行实证研究，并揭示了神经网络函数Lipschitz连续性的基本和有趣的特性，其中最引人注目的是在Lipschitz常数的上限和下限中识别出了明显的双下降趋势。

    

    Lipschitz连续性是任何预测模型的一个简单但关键的功能性质，它处于模型的稳健性、泛化性和对抗性脆弱性的核心。本文旨在深入研究和描述神经网络实现的函数的Lipschitz行为。因此，我们通过耗尽最简单和最一般的下限和上限的极限，在各种不同设置下进行实证研究（即，体系结构、损失、优化器、标签噪音等），虽然这一选择主要是受计算难度结果的驱动，但它也非常丰富，并揭示了神经网络函数Lipschitz连续性的几个基本和有趣的特性，我们还补充了适当的理论论证。

    Lipschitz continuity is a simple yet crucial functional property of any predictive model for it lies at the core of the model's robustness, generalisation, as well as adversarial vulnerability. Our aim is to thoroughly investigate and characterise the Lipschitz behaviour of the functions realised by neural networks. Thus, we carry out an empirical investigation in a range of different settings (namely, architectures, losses, optimisers, label noise, and more) by exhausting the limits of the simplest and the most general lower and upper bounds. Although motivated primarily by computational hardness results, this choice nevertheless turns out to be rather resourceful and sheds light on several fundamental and intriguing traits of the Lipschitz continuity of neural network functions, which we also supplement with suitable theoretical arguments. As a highlight of this investigation, we identify a striking double descent trend in both upper and lower bounds to the Lipschitz constant with in
    

