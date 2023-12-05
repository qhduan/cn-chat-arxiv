# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Locally Optimal Best Arm Identification with a Fixed Budget.](http://arxiv.org/abs/2310.19788) | 该研究解决了识别具有最高预期效果的治疗方案的问题，并提出了具有固定预算的局部最优算法来降低错误识别的概率。 |
| [^2] | [Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization.](http://arxiv.org/abs/2310.03234) | 本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。 |
| [^3] | [Supervised Machine Learning and Physics based Machine Learning approach for prediction of peak temperature distribution in Additive Friction Stir Deposition of Aluminium Alloy.](http://arxiv.org/abs/2309.06838) | 本研究提出了监督机器学习和基于物理的机器学习相结合的方法，用于预测搅拌摩擦增材制造中的峰值温度分布。实验结果表明，集成的机器学习方法在预测中表现出了较好的性能，最佳的SML方法为梯度提升法，最低的均方误差为165.78。 |
| [^4] | [The Effect of Intrinsic Dimension on Metric Learning under Compression.](http://arxiv.org/abs/2309.05751) | 本论文研究了内在维度对压缩下的度量学习的影响，提出了在对数据进行随机压缩后在低维空间内训练全秩度量的方法。理论保证了在不依赖环境维度的情况下，度量学习的误差可以被控制，并且在存在良性几何结构时效果更好。 |
| [^5] | [NLP-based detection of systematic anomalies among the narratives of consumer complaints.](http://arxiv.org/abs/2308.11138) | 本文开发了一种基于自然语言处理的方法，用于检测消费者投诉叙述中的系统异常。这种方法可以解决分类算法对于较小且频繁出现的系统异常检测的问题，并将投诉叙述转化为定量数据进行分析。 |
| [^6] | [Telematics Combined Actuarial Neural Networks for Cross-Sectional and Longitudinal Claim Count Data.](http://arxiv.org/abs/2308.01729) | 本论文提出了一种基于联合精算神经网络框架的横断面和纵向索赔计数模型，通过结合传统精算模型和神经网络，充分利用了两个模型的优势。 |
| [^7] | [Minimal Random Code Learning with Mean-KL Parameterization.](http://arxiv.org/abs/2307.07816) | 本文研究了最小随机编码学习（MIRACLE）的两个变体，提出了一种新的参数化方法Mean-KL，在压缩变分贝叶斯神经网络中实现了更快的收敛和良好的预测性能。 |
| [^8] | [How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model.](http://arxiv.org/abs/2307.02129) | 本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。 |
| [^9] | [Interpreting and Improving Diffusion Models Using the Euclidean Distance Function.](http://arxiv.org/abs/2306.04848) | 本文利用欧几里得距离函数解释去噪扩散模型，并提出了一种新的采样器。采样器表现出了最先进的FID得分，并能够生成高质量的样本。 |
| [^10] | [Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL.](http://arxiv.org/abs/2305.11032) | 本文提出了一种乐观自然策略梯度的在线强化学习策略优化框架，采用乐观策略评估子程序以鼓励探索，适用于线性MDP，样本复杂度具有最优维度依赖关系。 |
| [^11] | [Debiasing Conditional Stochastic Optimization.](http://arxiv.org/abs/2304.10613) | 本文提出了一种通用的随机外推技术，用于降低条件随机优化问题中的偏差，并证明在非凸光滑目标函数中，将外推与方差缩减技术相结合可以显著改善样本复杂度。 |
| [^12] | [Long-term Forecasting with TiDE: Time-series Dense Encoder.](http://arxiv.org/abs/2304.08424) | TiDE是一种基于MLP的编码器-解码器模型，用于长期时间序列预测。它既具备线性模型的简单性和速度，又能处理协变量和非线性依赖，相较于最佳的Transformer模型，速度快5-10倍。 |
| [^13] | [Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers.](http://arxiv.org/abs/2212.12474) | 本文使用物理学知识指导的高斯过程回归方法，解决线性偏微分方程求解器无法量化近似误差的问题。 |

# 详细

[^1]: 具有固定预算的局部最优最佳臂识别算法

    Locally Optimal Best Arm Identification with a Fixed Budget. (arXiv:2310.19788v1 [math.ST])

    [http://arxiv.org/abs/2310.19788](http://arxiv.org/abs/2310.19788)

    该研究解决了识别具有最高预期效果的治疗方案的问题，并提出了具有固定预算的局部最优算法来降低错误识别的概率。

    

    本研究探讨了识别最佳治疗方案的问题，即具有最高预期效果的治疗方案。我们旨在通过降低错误识别的概率来确定最佳治疗方案，这一问题在许多研究领域中已被探索，包括最佳臂识别（Best Arm Identification，BAI）和序列优化。在我们的实验中，治疗分配的轮数是固定的。在每一轮中，决策者将一种治疗方案分配给一个实验单元，并观察相应的结果，该结果遵循不同治疗方案之间方差不同的高斯分布。在实验结束时，我们根据观察结果推荐一种治疗方案作为最佳治疗方案的估计值。决策者的目标是设计一个实验，使错误识别最佳治疗方案的概率最小化。基于这一目标，我们开发了误识别概率的下界。

    This study investigates the problem of identifying the best treatment arm, a treatment arm with the highest expected outcome. We aim to identify the best treatment arm with a lower probability of misidentification, which has been explored under various names across numerous research fields, including \emph{best arm identification} (BAI) and ordinal optimization. In our experiments, the number of treatment-allocation rounds is fixed. In each round, a decision-maker allocates a treatment arm to an experimental unit and observes a corresponding outcome, which follows a Gaussian distribution with a variance different among treatment arms. At the end of the experiment, we recommend one of the treatment arms as an estimate of the best treatment arm based on the observations. The objective of the decision-maker is to design an experiment that minimizes the probability of misidentifying the best treatment arm. With this objective in mind, we develop lower bounds for the probability of misident
    
[^2]: 非光滑弱凸有限和耦合组合优化

    Non-Smooth Weakly-Convex Finite-sum Coupled Compositional Optimization. (arXiv:2310.03234v1 [math.OC])

    [http://arxiv.org/abs/2310.03234](http://arxiv.org/abs/2310.03234)

    本文研究了一种新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)，通过扩展已有的研究，我们研究了非光滑弱凸FCCO的问题，并提出了一种单循环算法来找到Moreau环的ε-稳定点。

    

    本文研究了一类新的组合优化问题，称为非光滑弱凸有限和耦合组合优化(NSWC FCCO)。由于其在机器学习和人工智能领域的广泛应用以及其解决基于经验风险最小化的随机算法的局限性，FCCO引起了越来越多的关注。然而，目前对于FCCO的研究假设内外函数都是光滑的，限制了其能够解决更多种类的问题的潜力。我们的研究从非光滑弱凸FCCO的角度进行了扩展，其中外函数是弱凸且非递减的，内函数是弱凸的。我们分析了一种单循环算法，并确定其在找到Moreau环的ε-稳定点的复杂度。

    This paper investigates new families of compositional optimization problems, called $\underline{\bf n}$on-$\underline{\bf s}$mooth $\underline{\bf w}$eakly-$\underline{\bf c}$onvex $\underline{\bf f}$inite-sum $\underline{\bf c}$oupled $\underline{\bf c}$ompositional $\underline{\bf o}$ptimization (NSWC FCCO). There has been a growing interest in FCCO due to its wide-ranging applications in machine learning and AI, as well as its ability to address the shortcomings of stochastic algorithms based on empirical risk minimization. However, current research on FCCO presumes that both the inner and outer functions are smooth, limiting their potential to tackle a more diverse set of problems. Our research expands on this area by examining non-smooth weakly-convex FCCO, where the outer function is weakly convex and non-decreasing, and the inner function is weakly-convex. We analyze a single-loop algorithm and establish its complexity for finding an $\epsilon$-stationary point of the Moreau env
    
[^3]: 监督机器学习和基于物理的机器学习方法用于预测铝合金搅拌摩擦增材制造中的峰值温度分布

    Supervised Machine Learning and Physics based Machine Learning approach for prediction of peak temperature distribution in Additive Friction Stir Deposition of Aluminium Alloy. (arXiv:2309.06838v1 [cs.LG])

    [http://arxiv.org/abs/2309.06838](http://arxiv.org/abs/2309.06838)

    本研究提出了监督机器学习和基于物理的机器学习相结合的方法，用于预测搅拌摩擦增材制造中的峰值温度分布。实验结果表明，集成的机器学习方法在预测中表现出了较好的性能，最佳的SML方法为梯度提升法，最低的均方误差为165.78。

    

    增材搅拌摩擦沉积（AFSD）是一种新型的固态增材制造技术，它解决了传统粉末床熔炼和定向能量沉积方法中存在的孔隙率、开裂和性能各向异性等问题。然而，AFSD中的工艺参数、热量分布和得到的显微结构之间的相关性仍然不够清楚，这妨碍了性能的工艺优化。本研究运用了一种先进的框架，将监督机器学习（SML）和基于物理的神经网络（PINNs）相结合，以从工艺参数预测AFSD中的峰值温度分布。对于SML建模，使用了八种回归算法，而对于PINNs，使用了运输、波传播、热传导和量子力学的控制方程。在多个统计指标上，集成的机器学习方法表现出了较好的性能，梯度提升法是最佳的SML方法，最低的均方误差为165.78。

    Additive friction stir deposition (AFSD) is a novel solid-state additive manufacturing technique that circumvents issues of porosity, cracking, and properties anisotropy that plague traditional powder bed fusion and directed energy deposition approaches. However, correlations between process parameters, thermal profiles, and resulting microstructure in AFSD remain poorly understood. This hinders process optimization for properties. This work employs a cutting-edge framework combining supervised machine learning (SML) and physics-informed neural networks (PINNs) to predict peak temperature distribution in AFSD from process parameters. Eight regression algorithms were implemented for SML modeling, while four PINNs leveraged governing equations for transport, wave propagation, heat transfer, and quantum mechanics. Across multiple statistical measures, ensemble techniques like gradient boosting proved superior for SML, with lowest MSE of 165.78. The integrated ML approach was also applied 
    
[^4]: 内在维度对压缩下的度量学习的影响

    The Effect of Intrinsic Dimension on Metric Learning under Compression. (arXiv:2309.05751v1 [cs.LG])

    [http://arxiv.org/abs/2309.05751](http://arxiv.org/abs/2309.05751)

    本论文研究了内在维度对压缩下的度量学习的影响，提出了在对数据进行随机压缩后在低维空间内训练全秩度量的方法。理论保证了在不依赖环境维度的情况下，度量学习的误差可以被控制，并且在存在良性几何结构时效果更好。

    

    度量学习旨在在输入空间中找到适当的距离度量，以改善基于距离的学习算法的性能。在高维环境中，度量学习还可以作为降维的手段，通过对学习的度量施加一个低秩约束。本文中，我们考虑的是对数据的一个随机压缩版本，然后在其中训练一个全秩的度量。我们给出了关于距离度量学习的误差的理论保证，这些保证不依赖于环境维度。我们的边界除了对来自有界支持的独立同分布数据没有显式的假设之外，并且在存在良性几何结构时自动收敛。在合成和真实数据集上的实验结果支持我们在高维环境中的理论发现。

    Metric learning aims at finding a suitable distance metric over the input space, to improve the performance of distance-based learning algorithms. In high-dimensional settings, metric learning can also play the role of dimensionality reduction, by imposing a low-rank restriction to the learnt metric. In this paper, instead of training a low-rank metric on high-dimensional data, we consider a randomly compressed version of the data, and train a full-rank metric there. We give theoretical guarantees on the error of distance-based metric learning, with respect to the random compression, which do not depend on the ambient dimension. Our bounds do not make any explicit assumptions, aside from i.i.d. data from a bounded support, and automatically tighten when benign geometrical structures are present. Experimental results on both synthetic and real data sets support our theoretical findings in high-dimensional settings.
    
[^5]: 基于自然语言处理的消费者投诉叙述中系统异常的检测方法

    NLP-based detection of systematic anomalies among the narratives of consumer complaints. (arXiv:2308.11138v1 [stat.ME])

    [http://arxiv.org/abs/2308.11138](http://arxiv.org/abs/2308.11138)

    本文开发了一种基于自然语言处理的方法，用于检测消费者投诉叙述中的系统异常。这种方法可以解决分类算法对于较小且频繁出现的系统异常检测的问题，并将投诉叙述转化为定量数据进行分析。

    

    我们开发了一种基于自然语言处理的方法，用于检测投诉叙述中的系统异常，简称为系统异常。尽管分类算法被用于检测明显的异常，但在较小且频繁出现的系统异常情况下，算法可能会因为各种原因而失效，包括技术原因和人工分析师的自然限制。因此，在分类之后的下一步中，我们将投诉叙述转化为定量数据，然后使用一种算法来检测系统异常。我们使用消费者金融保护局的消费者投诉数据库中的投诉叙述来说明整个过程。

    We develop an NLP-based procedure for detecting systematic nonmeritorious consumer complaints, simply called systematic anomalies, among complaint narratives. While classification algorithms are used to detect pronounced anomalies, in the case of smaller and frequent systematic anomalies, the algorithms may falter due to a variety of reasons, including technical ones as well as natural limitations of human analysts. Therefore, as the next step after classification, we convert the complaint narratives into quantitative data, which are then analyzed using an algorithm for detecting systematic anomalies. We illustrate the entire procedure using complaint narratives from the Consumer Complaint Database of the Consumer Financial Protection Bureau.
    
[^6]: 基于联合精算神经网络的横断面和纵向索赔计数数据的车载通信技术

    Telematics Combined Actuarial Neural Networks for Cross-Sectional and Longitudinal Claim Count Data. (arXiv:2308.01729v1 [stat.ML])

    [http://arxiv.org/abs/2308.01729](http://arxiv.org/abs/2308.01729)

    本论文提出了一种基于联合精算神经网络框架的横断面和纵向索赔计数模型，通过结合传统精算模型和神经网络，充分利用了两个模型的优势。

    

    我们提出了一种基于Mario W\"uthrich和Michael Merz提出的联合精算神经网络（CANN）框架的横断面和纵向索赔计数模型。CANN方法将传统的精算模型（如广义线性模型）与神经网络相结合，形成了一个包含经典回归模型和神经网络部分的双组件模型。CANN模型充分利用了两个模型的优势，既可以提供经典模型的可靠性和可解释性，又可以利用神经网络的灵活性和对复杂关系和交互作用的捕捉能力。在我们提出的模型中，我们使用了广为人知的对数线性索赔计数回归模型作为经典回归部分，使用了多层感知器（MLP）作为神经网络部分。MLP部分用于处理以向量形式表示的车辆驾驶行为的车载通信数据。

    We present novel cross-sectional and longitudinal claim count models for vehicle insurance built upon the Combined Actuarial Neural Network (CANN) framework proposed by Mario W\"uthrich and Michael Merz. The CANN approach combines a classical actuarial model, such as a generalized linear model, with a neural network. This blending of models results in a two-component model comprising a classical regression model and a neural network part. The CANN model leverages the strengths of both components, providing a solid foundation and interpretability from the classical model while harnessing the flexibility and capacity to capture intricate relationships and interactions offered by the neural network. In our proposed models, we use well-known log-linear claim count regression models for the classical regression part and a multilayer perceptron (MLP) for the neural network part. The MLP part is used to process telematics car driving data given as a vector characterizing the driving behavior 
    
[^7]: 带有Mean-KL参数化的最小随机编码学习

    Minimal Random Code Learning with Mean-KL Parameterization. (arXiv:2307.07816v1 [cs.LG])

    [http://arxiv.org/abs/2307.07816](http://arxiv.org/abs/2307.07816)

    本文研究了最小随机编码学习（MIRACLE）的两个变体，提出了一种新的参数化方法Mean-KL，在压缩变分贝叶斯神经网络中实现了更快的收敛和良好的预测性能。

    

    本文研究了最小随机编码学习（MIRACLE）的两个变体在压缩变分贝叶斯神经网络中的定性行为和鲁棒性。MIRACLE实现了强大的条件高斯变分近似权重后验$Q_{\mathbf{w}}$，并使用相对熵编码来压缩从后验中抽样的权重，使用高斯编码分布$P_{\mathbf{w}}$。为了达到所需的压缩率，必须对$Q_{\mathbf{w}} \Vert P_{\mathbf{w}}$进行约束，这需要在传统的均值-方差（Mean-Var）参数化下进行计算上昂贵的退火过程。相反，我们通过其平均值和KL散度来参数化$Q_{\mathbf{w}}$，以通过构造将压缩成本约束为所需值。我们证明了使用Mean-KL参数化的变分训练收敛速度是传统方法的两倍，并且在训练后保持了预测性能。

    This paper studies the qualitative behavior and robustness of two variants of Minimal Random Code Learning (MIRACLE) used to compress variational Bayesian neural networks. MIRACLE implements a powerful, conditionally Gaussian variational approximation for the weight posterior $Q_{\mathbf{w}}$ and uses relative entropy coding to compress a weight sample from the posterior using a Gaussian coding distribution $P_{\mathbf{w}}$. To achieve the desired compression rate, $D_{\mathrm{KL}}[Q_{\mathbf{w}} \Vert P_{\mathbf{w}}]$ must be constrained, which requires a computationally expensive annealing procedure under the conventional mean-variance (Mean-Var) parameterization for $Q_{\mathbf{w}}$. Instead, we parameterize $Q_{\mathbf{w}}$ by its mean and KL divergence from $P_{\mathbf{w}}$ to constrain the compression cost to the desired value by construction. We demonstrate that variational training with Mean-KL parameterization converges twice as fast and maintains predictive performance after 
    
[^8]: 深度神经网络如何学习组合性数据：随机层次模型

    How Deep Neural Networks Learn Compositional Data: The Random Hierarchy Model. (arXiv:2307.02129v1 [cs.LG])

    [http://arxiv.org/abs/2307.02129](http://arxiv.org/abs/2307.02129)

    本文研究了深度神经网络学习组合性数据的问题，通过对随机层次模型进行分类任务，发现深度CNN学习这个任务所需的训练数据数量随着类别数、组合数和迭代次数的增加而渐进增加。

    

    学习一般高维任务是非常困难的，因为它需要与维度成指数增长的训练数据数量。然而，深度卷积神经网络（CNN）在克服这一挑战方面显示出了卓越的成功。一种普遍的假设是可学习任务具有高度结构化，CNN利用这种结构建立了数据的低维表示。然而，我们对它们需要多少训练数据以及这个数字如何取决于数据结构知之甚少。本文回答了针对一个简单的分类任务的这个问题，该任务旨在捕捉真实数据的相关方面：随机层次模型。在这个模型中，$n_c$个类别中的每一个对应于$m$个同义组合的高层次特征，并且这些特征又通过一个重复$L$次的迭代过程由子特征组成。我们发现，需要深度CNN学习这个任务的训练数据数量$P^*$（i）随着$n_c m^L$的增长而渐进地增长，这只有...

    Learning generic high-dimensional tasks is notably hard, as it requires a number of training data exponential in the dimension. Yet, deep convolutional neural networks (CNNs) have shown remarkable success in overcoming this challenge. A popular hypothesis is that learnable tasks are highly structured and that CNNs leverage this structure to build a low-dimensional representation of the data. However, little is known about how much training data they require, and how this number depends on the data structure. This paper answers this question for a simple classification task that seeks to capture relevant aspects of real data: the Random Hierarchy Model. In this model, each of the $n_c$ classes corresponds to $m$ synonymic compositions of high-level features, which are in turn composed of sub-features through an iterative process repeated $L$ times. We find that the number of training data $P^*$ required by deep CNNs to learn this task (i) grows asymptotically as $n_c m^L$, which is only
    
[^9]: 利用欧几里得距离函数解释和改进扩散模型

    Interpreting and Improving Diffusion Models Using the Euclidean Distance Function. (arXiv:2306.04848v1 [cs.LG])

    [http://arxiv.org/abs/2306.04848](http://arxiv.org/abs/2306.04848)

    本文利用欧几里得距离函数解释去噪扩散模型，并提出了一种新的采样器。采样器表现出了最先进的FID得分，并能够生成高质量的样本。

    

    去噪直觉上与投影有关。事实上，在流形假设下，添加随机噪声近似等价于正交扰动。因此，学习去噪近似于学习投影。本文利用这一观察结果，将去噪扩散模型解释为应用于欧几里得距离函数的近似梯度下降。随后，我们基于对去噪器投影误差的简单假设，提供DDIM（Denoising Diffusion Implicit Models）采样器的简单收敛分析。最后，我们基于理论结果的洞见提出一种基于对DDIM的两个简单修改的新采样器。仅需要5-10个函数评估，我们的采样器就能在预训练的CIFAR-10和CelebA模型上达到最先进的FID得分，并且可以在潜在扩散模型上生成高质量的样本。

    Denoising is intuitively related to projection. Indeed, under the manifold hypothesis, adding random noise is approximately equivalent to orthogonal perturbation. Hence, learning to denoise is approximately learning to project. In this paper, we use this observation to reinterpret denoising diffusion models as approximate gradient descent applied to the Euclidean distance function. We then provide straight-forward convergence analysis of the DDIM sampler under simple assumptions on the projection-error of the denoiser. Finally, we propose a new sampler based on two simple modifications to DDIM using insights from our theoretical results. In as few as 5-10 function evaluations, our sampler achieves state-of-the-art FID scores on pretrained CIFAR-10 and CelebA models and can generate high quality samples on latent diffusion models.
    
[^10]: 乐观自然策略梯度：一种简单高效的在线强化学习策略优化框架

    Optimistic Natural Policy Gradient: a Simple Efficient Policy Optimization Framework for Online RL. (arXiv:2305.11032v1 [cs.LG])

    [http://arxiv.org/abs/2305.11032](http://arxiv.org/abs/2305.11032)

    本文提出了一种乐观自然策略梯度的在线强化学习策略优化框架，采用乐观策略评估子程序以鼓励探索，适用于线性MDP，样本复杂度具有最优维度依赖关系。

    

    尽管策略优化算法对于近期强化学习的实证成功发挥了重要作用，但策略优化的现有理论理解仍然相当有限 - 它们要么局限于表格MDP，要么在在线强化学习中存在高度亚最优的样本复杂度问题。本文提出了一种简单高效的在线强化学习策略优化框架 - 乐观自然策略梯度。乐观自然策略梯度可以看作是将经典自然策略梯度算法[Kakade，2001]与乐观策略评估子程序简单组合以鼓励探索。对于$d$-维线性MDP，乐观自然策略梯度具有计算效率，并且在$\tilde{O}(d^2/\varepsilon^3)$ 次采样内学习 $\varepsilon$ -最优策略，这是第一个具有最优维度依赖关系$\tilde {\Theta}(d^2)$样本复杂度的计算高效算法。它也超越了目前领先的一些状态of-the-art算法。

    While policy optimization algorithms have played an important role in recent empirical success of Reinforcement Learning (RL), the existing theoretical understanding of policy optimization remains rather limited -- they are either restricted to tabular MDPs or suffer from highly suboptimal sample complexity, especial in online RL where exploration is necessary. This paper proposes a simple efficient policy optimization framework -- Optimistic NPG for online RL. Optimistic NPG can be viewed as simply combining of the classic natural policy gradient (NPG) algorithm [Kakade, 2001] with optimistic policy evaluation subroutines to encourage exploration. For $d$-dimensional linear MDPs, Optimistic NPG is computationally efficient, and learns an $\varepsilon$-optimal policy within $\tilde{O}(d^2/\varepsilon^3)$ samples, which is the first computationally efficient algorithm whose sample complexity has the optimal dimension dependence $\tilde{\Theta}(d^2)$. It also improves over state-of-the-a
    
[^11]: 消除条件随机优化偏差

    Debiasing Conditional Stochastic Optimization. (arXiv:2304.10613v1 [cs.LG])

    [http://arxiv.org/abs/2304.10613](http://arxiv.org/abs/2304.10613)

    本文提出了一种通用的随机外推技术，用于降低条件随机优化问题中的偏差，并证明在非凸光滑目标函数中，将外推与方差缩减技术相结合可以显著改善样本复杂度。

    

    本文研究了覆盖了多个应用领域，包括投资组合选择、强化学习、鲁棒学习、因果推断等的条件随机优化（CSO）问题。由于其嵌套结构，CSO目标的样本平均梯度存在偏差，因此需要较高的样本复杂度才能达到收敛。我们引入了一种有效降低偏差的通用随机外推技术。我们证明，在非凸光滑目标函数中，将这种外推与方差缩减技术相结合，可以达到比现有界限更好的样本复杂度。我们还开发了用于有限和变量的CSO的新算法，也显著改进了现有结果。最后，我们认为我们的去偏技术也可能是适用于其他随机优化问题的有趣工具。

    In this paper, we study the conditional stochastic optimization (CSO) problem which covers a variety of applications including portfolio selection, reinforcement learning, robust learning, causal inference, etc. The sample-averaged gradient of the CSO objective is biased due to its nested structure and therefore requires a high sample complexity to reach convergence. We introduce a general stochastic extrapolation technique that effectively reduces the bias. We show that for nonconvex smooth objectives, combining this extrapolation with variance reduction techniques can achieve a significantly better sample complexity than existing bounds. We also develop new algorithms for the finite-sum variant of CSO that also significantly improve upon existing results. Finally, we believe that our debiasing technique could be an interesting tool applicable to other stochastic optimization problems too.
    
[^12]: 用TiDE进行长期预测：时间序列稠密编码器

    Long-term Forecasting with TiDE: Time-series Dense Encoder. (arXiv:2304.08424v1 [stat.ML])

    [http://arxiv.org/abs/2304.08424](http://arxiv.org/abs/2304.08424)

    TiDE是一种基于MLP的编码器-解码器模型，用于长期时间序列预测。它既具备线性模型的简单性和速度，又能处理协变量和非线性依赖，相较于最佳的Transformer模型，速度快5-10倍。

    

    最近的研究表明，相比于基于Transformer的方法，简单的线性模型在长期时间序列预测中表现更好。鉴于此，我们提出了一种基于多层感知机(MLP)的编码器-解码器模型，即时间序列稠密编码器(TiDE)，用于长期时间序列预测。它既享有线性模型的简单性和速度，又能处理协变量和非线性依赖。从理论上讲，我们证明了我们模型的最简线性类比在一些假设下可以达到线性动态系统(LDS)的近乎最优误差率。实证上，我们表明，我们的方法可以在流行的长期时间序列预测基准测试中匹配或胜过以前的方法，同时比最佳的基于Transformer的模型快5-10倍。

    Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.
    
[^13]: 物理学知识指导的高斯过程回归应用于解决线性偏微分方程

    Physics-Informed Gaussian Process Regression Generalizes Linear PDE Solvers. (arXiv:2212.12474v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.12474](http://arxiv.org/abs/2212.12474)

    本文使用物理学知识指导的高斯过程回归方法，解决线性偏微分方程求解器无法量化近似误差的问题。

    

    线性偏微分方程是一类重要且广泛应用的机械模型，描述了物理过程，例如热传导、电磁学和波传播等。实践中，通常使用基于离散化的专门数值方法来解决偏微分方程。这些求解器通常使用未知模型参数的估计值以及如果可用的话，物理测量值用于初始化。这些求解器经常嵌入到具有下游应用的更大的科学模型中，因此误差量化起着关键作用。然而，经典的偏微分方程求解器忽略参数和测量不确定性，可能无法产生一致性的估计值，以用于计算其固有的逼近误差。本文通过将求解线性偏微分方程解释为物理学知识指导的高斯过程回归来解决这个问题。我们的框架基于高斯过程推理定理的一个关键推广，该定理适用于通过任意界面进行观察的情况。

    Linear partial differential equations (PDEs) are an important, widely applied class of mechanistic models, describing physical processes such as heat transfer, electromagnetism, and wave propagation. In practice, specialized numerical methods based on discretization are used to solve PDEs. They generally use an estimate of the unknown model parameters and, if available, physical measurements for initialization. Such solvers are often embedded into larger scientific models with a downstream application and thus error quantification plays a key role. However, by ignoring parameter and measurement uncertainty, classical PDE solvers may fail to produce consistent estimates of their inherent approximation error. In this work, we approach this problem in a principled fashion by interpreting solving linear PDEs as physics-informed Gaussian process (GP) regression. Our framework is based on a key generalization of the Gaussian process inference theorem to observations made via an arbitrary bou
    

