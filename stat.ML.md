# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Improved Relaxation for Oracle-Efficient Adversarial Contextual Bandits.](http://arxiv.org/abs/2310.19025) | 该论文提出了一种用于对抗性上下文赌博问题的面向Oracle高效的放松方法，通过调用离线优化Oracle来降低遗憾界限，并且在界限方面取得了显著的改进，达到了先前最佳界限，并与原始界限相匹配。 |
| [^2] | [Stabilizing Estimates of Shapley Values with Control Variates.](http://arxiv.org/abs/2310.07672) | 使用控制变量的方法稳定Shapley值的估计，减少了模型解释的不确定性，适用于任何机器学习模型。 |
| [^3] | [Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory.](http://arxiv.org/abs/2310.04935) | 这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。 |
| [^4] | [Solving Kernel Ridge Regression with Gradient-Based Optimization Methods.](http://arxiv.org/abs/2306.16838) | 本研究提出了一种新的方法来解决核岭回归问题，通过等价的目标函数形式和基于梯度的优化方法，我们不仅可以使用其他惩罚方法，还能够从梯度下降的角度研究核岭回归。通过提前停止的正则化，我们推导出了一个闭合解，即核梯度流（KGF），并证明了KGF和KRR之间的差异。我们还将KRR泛化，使用$\ell_1$和$\ell_\infty$惩罚方法，并发现使用这些方法得到的解与前向分步回归和符号梯度下降结合提前停止得到的解非常相似。因此，我们减少了计算复杂度重的近端梯度下降算法的需求。 |
| [^5] | [Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks.](http://arxiv.org/abs/2306.05674) | 本论文基于神经切向核理论，提出了一种高效的方法以减少超参数化神经网络中的过程不确定性，并只需要使用一种辅助网络就可以消除这种不确定性。 |
| [^6] | [Adapting Fairness Interventions to Missing Values.](http://arxiv.org/abs/2305.19429) | 本文研究了如何在缺失值的情况下实现公平的分类。传统方法会加剧歧视。本文证明从插补数据训练分类器会恶化组公平性和平均准确性。作者提出可扩展和适应性的算法，可以与其他公平干预算法结合使用，以处理所有可能的缺失模式。 |
| [^7] | [Deep quantum neural networks form Gaussian processes.](http://arxiv.org/abs/2305.09957) | 本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。 |
| [^8] | [PriorCVAE: scalable MCMC parameter inference with Bayesian deep generative modelling.](http://arxiv.org/abs/2304.04307) | PriorCVAE 提出了一种处理高斯过程先验 MCMC 参数推断的贝叶斯深度生成建模新方法，可通过将 VAE 建模条件化于随机过程超参数处理超参数推断与学习先验之间的信息流断裂问题。 |

# 详细

[^1]: 《一种改进的面向Oracle高效的对抗性上下文赌博问题的放松方法》

    An Improved Relaxation for Oracle-Efficient Adversarial Contextual Bandits. (arXiv:2310.19025v1 [cs.LG])

    [http://arxiv.org/abs/2310.19025](http://arxiv.org/abs/2310.19025)

    该论文提出了一种用于对抗性上下文赌博问题的面向Oracle高效的放松方法，通过调用离线优化Oracle来降低遗憾界限，并且在界限方面取得了显著的改进，达到了先前最佳界限，并与原始界限相匹配。

    

    我们提出了一种面向Oracle高效的放松方法，用于处理对抗性上下文赌博问题，其中上下文是从已知分布中顺序独立抽取的，而成本序列则由在线对手选择。我们的算法具有一个$O(T^{\frac{2}{3}}(K\log(|\Pi|))^{\frac{1}{3}})$的遗憾界限，并且每轮最多调用$O(K)$次离线优化Oracle，其中$K$表示动作的数量，$T$表示轮数，$\Pi$表示策略集。这是第一个改进Syrgkanis等人在NeurIPS 2016中获得的$O((TK)^{\frac{2}{3}}(\log(|\Pi|))^{\frac{1}{3}})$界限的结果，并且也是与Langford和Zhang在NeurIPS 2007中为随机情况提出的原始界限相匹配的结果。

    We present an oracle-efficient relaxation for the adversarial contextual bandits problem, where the contexts are sequentially drawn i.i.d from a known distribution and the cost sequence is chosen by an online adversary. Our algorithm has a regret bound of $O(T^{\frac{2}{3}}(K\log(|\Pi|))^{\frac{1}{3}})$ and makes at most $O(K)$ calls per round to an offline optimization oracle, where $K$ denotes the number of actions, $T$ denotes the number of rounds and $\Pi$ denotes the set of policies. This is the first result to improve the prior best bound of $O((TK)^{\frac{2}{3}}(\log(|\Pi|))^{\frac{1}{3}})$ as obtained by Syrgkanis et al. at NeurIPS 2016, and the first to match the original bound of Langford and Zhang at NeurIPS 2007 which was obtained for the stochastic case.
    
[^2]: 用控制变量稳定Shapley值的估计

    Stabilizing Estimates of Shapley Values with Control Variates. (arXiv:2310.07672v1 [stat.ML])

    [http://arxiv.org/abs/2310.07672](http://arxiv.org/abs/2310.07672)

    使用控制变量的方法稳定Shapley值的估计，减少了模型解释的不确定性，适用于任何机器学习模型。

    

    Shapley值是解释黑盒机器学习模型预测最流行的工具之一。然而，它们的计算成本很高，因此采用抽样近似来减少不确定性。为了稳定这些模型解释，我们提出了一种基于控制变量的蒙特卡洛技术的方法，称为ControlSHAP。我们的方法适用于任何机器学习模型，并且几乎不需要额外的计算或建模工作。在多个高维数据集上，我们发现它可以显著减少Shapley估计的蒙特卡洛变异性。

    Shapley values are among the most popular tools for explaining predictions of blackbox machine learning models. However, their high computational cost motivates the use of sampling approximations, inducing a considerable degree of uncertainty. To stabilize these model explanations, we propose ControlSHAP, an approach based on the Monte Carlo technique of control variates. Our methodology is applicable to any machine learning model and requires virtually no extra computation or modeling effort. On several high-dimensional datasets, we find it can produce dramatic reductions in the Monte Carlo variability of Shapley estimates.
    
[^3]: 使用PAC-Bayesian理论给变分自动编码器提供统计保证

    Statistical Guarantees for Variational Autoencoders using PAC-Bayesian Theory. (arXiv:2310.04935v1 [cs.LG])

    [http://arxiv.org/abs/2310.04935](http://arxiv.org/abs/2310.04935)

    这项工作利用PAC-Bayesian理论为变分自动编码器提供了统计保证，包括对后验分布、重构损失和输入与生成分布之间距离的上界。

    

    自从它们的问世以来，变分自动编码器（VAEs）在机器学习中变得非常重要。尽管它们被广泛使用，关于它们的理论性质仍存在许多问题。本文利用PAC-Bayesian理论为VAEs提供统计保证。首先，我们推导出了基于独立样本的后验分布的首个PAC-Bayesian界限。然后，利用这一结果为VAE的重构损失提供了泛化保证，同时提供了输入分布与VAE生成模型定义的分布之间距离的上界。更重要的是，我们提供了输入分布与VAE生成模型定义的分布之间Wasserstein距离的上界。

    Since their inception, Variational Autoencoders (VAEs) have become central in machine learning. Despite their widespread use, numerous questions regarding their theoretical properties remain open. Using PAC-Bayesian theory, this work develops statistical guarantees for VAEs. First, we derive the first PAC-Bayesian bound for posterior distributions conditioned on individual samples from the data-generating distribution. Then, we utilize this result to develop generalization guarantees for the VAE's reconstruction loss, as well as upper bounds on the distance between the input and the regenerated distributions. More importantly, we provide upper bounds on the Wasserstein distance between the input distribution and the distribution defined by the VAE's generative model.
    
[^4]: 用基于梯度的优化方法解决核岭回归问题

    Solving Kernel Ridge Regression with Gradient-Based Optimization Methods. (arXiv:2306.16838v1 [stat.ML])

    [http://arxiv.org/abs/2306.16838](http://arxiv.org/abs/2306.16838)

    本研究提出了一种新的方法来解决核岭回归问题，通过等价的目标函数形式和基于梯度的优化方法，我们不仅可以使用其他惩罚方法，还能够从梯度下降的角度研究核岭回归。通过提前停止的正则化，我们推导出了一个闭合解，即核梯度流（KGF），并证明了KGF和KRR之间的差异。我们还将KRR泛化，使用$\ell_1$和$\ell_\infty$惩罚方法，并发现使用这些方法得到的解与前向分步回归和符号梯度下降结合提前停止得到的解非常相似。因此，我们减少了计算复杂度重的近端梯度下降算法的需求。

    

    核岭回归（KRR）是线性岭回归的非线性推广。在这里，我们引入了KRR目标函数的等价形式，为使用其他惩罚方法和从梯度下降的角度研究核岭回归打开了可能。通过连续时间的视角，我们推导出了一个闭合解——核梯度流（KGF），通过提前停止的正则化，让我们能够在KGF和KRR之间理论上界定差异。我们用$\ell_1$和$\ell_\infty$惩罚方法将KRR泛化，并利用类似KGF和KRR之间的相似性，使用这些惩罚方法得到的解与使用前向分步回归（也称为坐标下降）和符号梯度下降结合提前停止得到的解非常相似。因此，减少了计算复杂度重的近端梯度下降算法的需求。

    Kernel ridge regression, KRR, is a non-linear generalization of linear ridge regression. Here, we introduce an equivalent formulation of the objective function of KRR, opening up both for using other penalties than the ridge penalty and for studying kernel ridge regression from the perspective of gradient descent. Using a continuous-time perspective, we derive a closed-form solution, kernel gradient flow, KGF, with regularization through early stopping, which allows us to theoretically bound the differences between KGF and KRR. We generalize KRR by replacing the ridge penalty with the $\ell_1$ and $\ell_\infty$ penalties and utilize the fact that analogously to the similarities between KGF and KRR, the solutions obtained when using these penalties are very similar to those obtained from forward stagewise regression (also known as coordinate descent) and sign gradient descent in combination with early stopping. Thus the need for computationally heavy proximal gradient descent algorithms
    
[^5]: 面向超参数化神经网络的高效不确定性量化和减少方法

    Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks. (arXiv:2306.05674v1 [stat.ML])

    [http://arxiv.org/abs/2306.05674](http://arxiv.org/abs/2306.05674)

    本论文基于神经切向核理论，提出了一种高效的方法以减少超参数化神经网络中的过程不确定性，并只需要使用一种辅助网络就可以消除这种不确定性。

    

    不确定性量化（UQ）对于机器学习模型的可靠性评估和改进至关重要。在深度学习中，不确定性不仅来自数据，还来自训练过程中注入的大量噪声和偏差。这些噪声和偏差妨碍了统计保证的实现，并且由于需要重复的网络重新训练，对UQ提出了计算挑战。基于最近的神经切向核理论，我们创建了具有统计保证的方案，以通过非常低的计算量量化和减少超参数化神经网络的过程不确定性。特别地，我们的方法基于我们称为过程噪声校正（PNC）预测器，通过只使用一种适当标记数据集上训练的辅助网络来消除过程不确定性，而不是使用深层集成中的许多重新训练的网络。此外，通过将我们的PNC预测器与所提出的先验模型结合起来，我们可以显著减少所需的网络正则化。

    Uncertainty quantification (UQ) is important for reliability assessment and enhancement of machine learning models. In deep learning, uncertainties arise not only from data, but also from the training procedure that often injects substantial noises and biases. These hinder the attainment of statistical guarantees and, moreover, impose computational challenges on UQ due to the need for repeated network retraining. Building upon the recent neural tangent kernel theory, we create statistically guaranteed schemes to principally \emph{quantify}, and \emph{remove}, the procedural uncertainty of over-parameterized neural networks with very low computation effort. In particular, our approach, based on what we call a procedural-noise-correcting (PNC) predictor, removes the procedural uncertainty by using only \emph{one} auxiliary network that is trained on a suitably labeled data set, instead of many retrained networks employed in deep ensembles. Moreover, by combining our PNC predictor with su
    
[^6]: 缺失值下的公平性干预措施的适应性研究

    Adapting Fairness Interventions to Missing Values. (arXiv:2305.19429v1 [cs.LG])

    [http://arxiv.org/abs/2305.19429](http://arxiv.org/abs/2305.19429)

    本文研究了如何在缺失值的情况下实现公平的分类。传统方法会加剧歧视。本文证明从插补数据训练分类器会恶化组公平性和平均准确性。作者提出可扩展和适应性的算法，可以与其他公平干预算法结合使用，以处理所有可能的缺失模式。

    

    真实世界中数据的缺失值对算法公平性提出了显著而独特的挑战。不同的族群可能不会同等地受到缺失数据的影响，而处理缺失值的标准程序，即先对数据进行插补，然后使用插补的数据进行分类，这个过程被称为“插补再分类”，会加剧歧视。本文分析了缺失值如何影响算法公平性。我们首先证明了从插补数据训练分类器会显著恶化可以实现的组公平性和平均准确性的值。这是因为插补数据会导致数据缺失模式的丢失，数据缺失模式通常会传达有关预测标签的信息。我们提出了可扩展和适应性的算法，用于处理缺失值的公平分类。这些算法可以与任何现有的公平干预算法结合使用，以处理所有可能的缺失模式，并保留信息。

    Missing values in real-world data pose a significant and unique challenge to algorithmic fairness. Different demographic groups may be unequally affected by missing data, and the standard procedure for handling missing values where first data is imputed, then the imputed data is used for classification -- a procedure referred to as "impute-then-classify" -- can exacerbate discrimination. In this paper, we analyze how missing values affect algorithmic fairness. We first prove that training a classifier from imputed data can significantly worsen the achievable values of group fairness and average accuracy. This is because imputing data results in the loss of the missing pattern of the data, which often conveys information about the predictive label. We present scalable and adaptive algorithms for fair classification with missing values. These algorithms can be combined with any preexisting fairness-intervention algorithm to handle all possible missing patterns while preserving informatio
    
[^7]: 深度量子神经网络对应高斯过程

    Deep quantum neural networks form Gaussian processes. (arXiv:2305.09957v1 [quant-ph])

    [http://arxiv.org/abs/2305.09957](http://arxiv.org/abs/2305.09957)

    本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。

    

    众所周知，从独立同分布的先验条件开始初始化的人工神经网络在隐藏层神经元数目足够大的极限下收敛到高斯过程。本文证明了量子神经网络（QNNs）也存在类似的结果。特别地，我们证明了基于Haar随机酉或正交深QNNs的某些模型的输出在希尔伯特空间维度$d$足够大时会收敛于高斯过程。由于输入状态、测量的可观测量以及酉矩阵的元素不独立等因素的作用，本文对这一结果的推导比经典情形更加微妙。我们分析的一个重要后果是，这个结果得到的高斯过程不能通过贝叶斯统计学来有效地预测QNN的输出。此外，我们的定理表明，Haar随机QNNs中的测量现象比以前认为的要更严重，我们证明了演员的集中现象。

    It is well known that artificial neural networks initialized from independent and identically distributed priors converge to Gaussian processes in the limit of large number of neurons per hidden layer. In this work we prove an analogous result for Quantum Neural Networks (QNNs). Namely, we show that the outputs of certain models based on Haar random unitary or orthogonal deep QNNs converge to Gaussian processes in the limit of large Hilbert space dimension $d$. The derivation of this result is more nuanced than in the classical case due the role played by the input states, the measurement observable, and the fact that the entries of unitary matrices are not independent. An important consequence of our analysis is that the ensuing Gaussian processes cannot be used to efficiently predict the outputs of the QNN via Bayesian statistics. Furthermore, our theorems imply that the concentration of measure phenomenon in Haar random QNNs is much worse than previously thought, as we prove that ex
    
[^8]: PriorCVAE: 基于贝叶斯深度生成建模的可扩展 MCMC 参数推断

    PriorCVAE: scalable MCMC parameter inference with Bayesian deep generative modelling. (arXiv:2304.04307v1 [stat.ML])

    [http://arxiv.org/abs/2304.04307](http://arxiv.org/abs/2304.04307)

    PriorCVAE 提出了一种处理高斯过程先验 MCMC 参数推断的贝叶斯深度生成建模新方法，可通过将 VAE 建模条件化于随机过程超参数处理超参数推断与学习先验之间的信息流断裂问题。

    

    在应用场景中，推理速度和模型灵活性至关重要，贝叶斯推断在具有随机过程先验的模型中（如高斯过程）被广泛应用。最近的研究表明，使用变分自动编码器（VAE）等深度生成模型可以编码由 GP 先验或其有限实现引起的计算瓶颈，并且所学生成器可以代替 MCMC 推断中的原始先验。虽然此方法实现了快速而高效的推理，但它丢失了关于随机过程超参数的信息，导致超参数推断不可能和学到的先验模糊不清。我们建议解决上述问题，通过将 VAE 建模条件化于随机过程超参数，以便超参数与 GP 实现一起进行编码。

    In applied fields where the speed of inference and model flexibility are crucial, the use of Bayesian inference for models with a stochastic process as their prior, e.g. Gaussian processes (GPs) is ubiquitous. Recent literature has demonstrated that the computational bottleneck caused by GP priors or their finite realizations can be encoded using deep generative models such as variational autoencoders (VAEs), and the learned generators can then be used instead of the original priors during Markov chain Monte Carlo (MCMC) inference in a drop-in manner. While this approach enables fast and highly efficient inference, it loses information about the stochastic process hyperparameters, and, as a consequence, makes inference over hyperparameters impossible and the learned priors indistinct. We propose to resolve the aforementioned issue and disentangle the learned priors by conditioning the VAE on stochastic process hyperparameters. This way, the hyperparameters are encoded alongside GP real
    

