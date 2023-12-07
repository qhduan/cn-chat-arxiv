# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Banach Space Optimality of Neural Architectures With Multivariate Nonlinearities.](http://arxiv.org/abs/2310.03696) | 本文研究了具有多变量非线性激活函数的神经网络架构在Banach空间的优化性，并构建了一类新的Banach空间家族。结果表明，学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优架构具有跳跃连接，并与正交权重归一化和多索引模型密切相关。 |
| [^2] | [Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback.](http://arxiv.org/abs/2307.04749) | 本文提出了一种划分、评估和细化的方法来改善文本到图像对齐。通过分解复杂的提示并使用VQA模型进行测量，最终得到文本到图像的对齐分数。 |
| [^3] | [Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models.](http://arxiv.org/abs/2306.13255) | 本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。 |
| [^4] | [Functional Flow Matching.](http://arxiv.org/abs/2305.17209) | 本文介绍了一种名为功能流匹配（FFM）的函数空间生成模型，该模型利用概率测度插值和学习底层函数空间上生成测度的向量场来生成数据分布。这种无需似然或模拟的方法在合成和真实世界基准数据集上表现优异，优于最近提出的几种函数空间生成模型。 |
| [^5] | [Improved Convergence of Score-Based Diffusion Models via Prediction-Correction.](http://arxiv.org/abs/2305.14164) | 本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。 |
| [^6] | [A unified framework for information-theoretic generalization bounds.](http://arxiv.org/abs/2305.11042) | 该论文提出了一种基于概率去相关引理和概率测度空间中一些其他技术的通用方法，可以得到新的学习算法的信息论泛化上限，并且能够恢复许多现有的泛化界，如基于互信息、条件互信息、随机chaining和PAC-Bayes不等式的界。 |
| [^7] | [Microcanonical Langevin Monte Carlo.](http://arxiv.org/abs/2303.18221) | 我们提出的微正则 Langevin Monte Carlo 方法能够高效地采样 $\exp[-S(\x)]$ 分布，同时具有无偏性。 |
| [^8] | [Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US.](http://arxiv.org/abs/2302.02560) | 本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。 |
| [^9] | [Targeted Separation and Convergence with Kernel Discrepancies.](http://arxiv.org/abs/2209.12835) | 通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。 |
| [^10] | [Optimal Variable Clustering for High-Dimensional Matrix Valued Data.](http://arxiv.org/abs/2112.12909) | 提出了一种新的针对高维矩阵数据的特征潜变量模型，使用加权协方差矩阵的差异作为不相似度测量的层次聚类算法，理论上实现了聚类一致性，在模拟和真实数据示例中证明了方法的优越性。 |

# 详细

[^1]: 具有多变量非线性的神经网络架构的Banach空间优化性研究

    Banach Space Optimality of Neural Architectures With Multivariate Nonlinearities. (arXiv:2310.03696v1 [stat.ML])

    [http://arxiv.org/abs/2310.03696](http://arxiv.org/abs/2310.03696)

    本文研究了具有多变量非线性激活函数的神经网络架构在Banach空间的优化性，并构建了一类新的Banach空间家族。结果表明，学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优架构具有跳跃连接，并与正交权重归一化和多索引模型密切相关。

    

    本文研究了一大类具有多变量非线性/激活函数的神经网络架构的变分优化性（具体而言，是Banach空间优化性）。为此，我们通过正则化算子和k-平面变换构建了一类新的Banach空间家族。我们证明了一个表示定理，该定理说明在这些Banach空间上提出的学习问题的解集完全由具有多变量非线性的神经网络架构来描述。这些最优的架构具有跳跃连接，并与正交权重归一化和多索引模型息息相关，这两个模型在神经网络界引起了相当大的兴趣。我们的框架适用于包括修正线性单元（ReLU）激活函数、范数激活函数以及在薄板/多次谐波样条理论中找到的径向基函数在内的多种经典非线性函数。

    We investigate the variational optimality (specifically, the Banach space optimality) of a large class of neural architectures with multivariate nonlinearities/activation functions. To that end, we construct a new family of Banach spaces defined via a regularization operator and the $k$-plane transform. We prove a representer theorem that states that the solution sets to learning problems posed over these Banach spaces are completely characterized by neural architectures with multivariate nonlinearities. These optimal architectures have skip connections and are tightly connected to orthogonal weight normalization and multi-index models, both of which have received considerable interest in the neural network community. Our framework is compatible with a number of classical nonlinearities including the rectified linear unit (ReLU) activation function, the norm activation function, and the radial basis functions found in the theory of thin-plate/polyharmonic splines. We also show that the
    
[^2]: 划分、评估和细化：通过迭代VQA反馈评估和改善文本到图像对齐

    Divide, Evaluate, and Refine: Evaluating and Improving Text-to-Image Alignment with Iterative VQA Feedback. (arXiv:2307.04749v1 [cs.CV])

    [http://arxiv.org/abs/2307.04749](http://arxiv.org/abs/2307.04749)

    本文提出了一种划分、评估和细化的方法来改善文本到图像对齐。通过分解复杂的提示并使用VQA模型进行测量，最终得到文本到图像的对齐分数。

    

    随着潜在扩散模型的最新出现，以文本为条件的图像生成领域取得了前所未有的进展。然而，尽管具有显著性，但是随着文本输入的复杂性增加，最先进的扩散模型仍可能无法生成准确传达给定提示语义的图像。此外，观察到这种不对齐往往被预训练的多模型（如CLIP）未能检测到。为了解决这些问题，在本文中，我们探索了一种简单且有效的分解方法来评估和改善文本到图像对齐。具体而言，我们首先引入了一种分解对齐分数，它将复杂提示分解为一组不相交的断言。然后，使用VQA模型来测量每个断言与生成的图像的对齐情况。最后，将不同断言的对齐分数合并后，得到最终的文本到图像对齐分数。

    The field of text-conditioned image generation has made unparalleled progress with the recent advent of latent diffusion models. While remarkable, as the complexity of given text input increases, the state-of-the-art diffusion models may still fail in generating images which accurately convey the semantics of the given prompt. Furthermore, it has been observed that such misalignments are often left undetected by pretrained multi-modal models such as CLIP. To address these problems, in this paper we explore a simple yet effective decompositional approach towards both evaluation and improvement of text-to-image alignment. In particular, we first introduce a Decompositional-Alignment-Score which given a complex prompt decomposes it into a set of disjoint assertions. The alignment of each assertion with generated images is then measured using a VQA model. Finally, alignment scores for different assertions are combined aposteriori to give the final text-to-image alignment score. Experimenta
    
[^3]: 过参数化线性模型下多类分类的渐进泛化精度研究

    Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models. (arXiv:2306.13255v1 [cs.LG])

    [http://arxiv.org/abs/2306.13255](http://arxiv.org/abs/2306.13255)

    本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。

    

    本文研究了在具有高斯协变量双层模型下，过参数化线性模型在多类分类中的渐进泛化问题，其中数据点数、特征和类别数都同时增长。我们完全解决了Subramanian等人在'22年所提出的猜想，与预测的泛化区间相匹配。此外，我们的新的下界类似于信息论中的强对偶定理：它们能够确立误分类率逐渐趋近于0或1.我们紧密的结果的一个令人惊讶的结果是，最小范数插值分类器在最小范数插值回归器最优的范围内，可以在渐进上次优。我们分析的关键在于一种新的Hanson-Wright不等式变体，该变体在具有稀疏标签的多类问题中具有广泛的适用性。作为应用，我们展示了相同类型分析在几种不同类型的分类模型上的结果。

    We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al.~'22, where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al.~'22, matching the predicted regimes for generalization. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal.  The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis 
    
[^4]: 功能流匹配

    Functional Flow Matching. (arXiv:2305.17209v1 [cs.LG])

    [http://arxiv.org/abs/2305.17209](http://arxiv.org/abs/2305.17209)

    本文介绍了一种名为功能流匹配（FFM）的函数空间生成模型，该模型利用概率测度插值和学习底层函数空间上生成测度的向量场来生成数据分布。这种无需似然或模拟的方法在合成和真实世界基准数据集上表现优异，优于最近提出的几种函数空间生成模型。

    

    本文提出了一种名为功能流匹配（Functional Flow Matching, FFM）的函数空间生成模型，该模型将最近引入的流匹配（Flow Matching）直接推广到无限维空间中进行。我们的方法首先定义了一组概率测度路径，在固定的高斯测度和数据分布之间进行插值，然后学习函数的底层空间上生成此测度路径的向量场。我们的方法不依赖于似然或模拟，因此非常适合函数空间的设置。我们不仅提供构建这种模型的理论框架，还对我们的技术进行了经验评估。通过对合成和真实世界基准数据集的实验，我们证明了我们提出的FFM方法优于最近提出的几种函数空间生成模型。

    In this work, we propose Functional Flow Matching (FFM), a function-space generative model that generalizes the recently-introduced Flow Matching model to operate directly in infinite-dimensional spaces. Our approach works by first defining a path of probability measures that interpolates between a fixed Gaussian measure and the data distribution, followed by learning a vector field on the underlying space of functions that generates this path of measures. Our method does not rely on likelihoods or simulations, making it well-suited to the function space setting. We provide both a theoretical framework for building such models and an empirical evaluation of our techniques. We demonstrate through experiments on synthetic and real-world benchmarks that our proposed FFM method outperforms several recently proposed function-space generative models.
    
[^5]: 通过预测修正提高基于得分扩散模型的收敛性

    Improved Convergence of Score-Based Diffusion Models via Prediction-Correction. (arXiv:2305.14164v1 [cs.LG])

    [http://arxiv.org/abs/2305.14164](http://arxiv.org/abs/2305.14164)

    本文通过使用预测校正方案，提高了基于得分扩散模型的收敛性。

    

    基于得分的生成模型（SGM）是从复杂数据分布中进行采样的强大工具。其基本思想是（i）通过向数据添加噪声运行时间为$T_1$的正向过程，（ii）估计其得分函数，并（iii）使用此估计值运行反向过程。由于反向过程以正向过程的平稳分布作为初始值，因此现有的分析范式要求$T_1\to\infty$。然而，从理论角度来看，对于给定的分数逼近精度，当$T_1$发散时，收敛保证将失败；从实际角度来看，$T_1$越大，计算成本就越高，并且会导致误差传播。本文通过考虑流行的预测器校正方案的一个版本来解决这个问题：在运行正向过程之后，我们首先通过不精确的 Langevin 动力学估计最终分布，然后恢复该过程。我们的关键技术贡献是提供了收敛保证。

    Score-based generative models (SGMs) are powerful tools to sample from complex data distributions. Their underlying idea is to (i) run a forward process for time $T_1$ by adding noise to the data, (ii) estimate its score function, and (iii) use such estimate to run a reverse process. As the reverse process is initialized with the stationary distribution of the forward one, the existing analysis paradigm requires $T_1\to\infty$. This is however problematic: from a theoretical viewpoint, for a given precision of the score approximation, the convergence guarantee fails as $T_1$ diverges; from a practical viewpoint, a large $T_1$ increases computational costs and leads to error propagation. This paper addresses the issue by considering a version of the popular predictor-corrector scheme: after running the forward process, we first estimate the final distribution via an inexact Langevin dynamics and then revert the process. Our key technical contribution is to provide convergence guarantees
    
[^6]: 一种信息论通用泛化界统一框架

    A unified framework for information-theoretic generalization bounds. (arXiv:2305.11042v1 [cs.LG])

    [http://arxiv.org/abs/2305.11042](http://arxiv.org/abs/2305.11042)

    该论文提出了一种基于概率去相关引理和概率测度空间中一些其他技术的通用方法，可以得到新的学习算法的信息论泛化上限，并且能够恢复许多现有的泛化界，如基于互信息、条件互信息、随机chaining和PAC-Bayes不等式的界。

    

    本文提出了一种通用的方法来导出学习算法的信息论泛化界。主要的技术工具是基于改变测度和松弛Young不等式在$L_{\psi_p}$Orlicz空间中的概率去相关性引理。采用去相关性引理与其他技术，如对称化、耦合和概率测度空间中的chaining，我们得到了新的泛化误差上限，包括期望和高概率，同时，我们也恢复了许多现有的泛化界，包括基于互信息、条件互信息、随机chaining和PAC-Bayes不等式的界。此外，Fernique-Talagrand上界也作为一种特殊情况呈现出来。

    This paper presents a general methodology for deriving information-theoretic generalization bounds for learning algorithms. The main technical tool is a probabilistic decorrelation lemma based on a change of measure and a relaxation of Young's inequality in $L_{\psi_p}$ Orlicz spaces. Using the decorrelation lemma in combination with other techniques, such as symmetrization, couplings, and chaining in the space of probability measures, we obtain new upper bounds on the generalization error, both in expectation and in high probability, and recover as special cases many of the existing generalization bounds, including the ones based on mutual information, conditional mutual information, stochastic chaining, and PAC-Bayes inequalities. In addition, the Fernique-Talagrand upper bound on the expected supremum of a subgaussian process emerges as a special case.
    
[^7]: 微正则 Langevin Monte Carlo

    Microcanonical Langevin Monte Carlo. (arXiv:2303.18221v1 [hep-lat])

    [http://arxiv.org/abs/2303.18221](http://arxiv.org/abs/2303.18221)

    我们提出的微正则 Langevin Monte Carlo 方法能够高效地采样 $\exp[-S(\x)]$ 分布，同时具有无偏性。

    

    我们提出了一种方法，用于以可用渐变 $ \nabla S(\x)$ 的形式采样自一任意分布 $ \exp[-S(\x)]$，该方法被制定为保持能量的随机微分方程（SDE）。我们推导出 Fokker-Planck 方程，并证明确定性漂移和随机扩散分别保持平稳分布。这意味着漂移扩散离散化方案无偏，而标准 Langevin 动力学则不是。我们将该方法应用于 $\phi^4$ 晶格场论，展示了结果与标准采样方法一致，但比当前最先进的采样器效率显著提高。

    We propose a method for sampling from an arbitrary distribution $\exp[-S(\x)]$ with an available gradient $\nabla S(\x)$, formulated as an energy-preserving stochastic differential equation (SDE). We derive the Fokker-Planck equation and show that both the deterministic drift and the stochastic diffusion separately preserve the stationary distribution. This implies that the drift-diffusion discretization schemes are bias-free, in contrast to the standard Langevin dynamics. We apply the method to the $\phi^4$ lattice field theory, showing the results agree with the standard sampling methods but with significantly higher efficiency compared to the current state-of-the-art samplers.
    
[^8]: 神经网络在因果估计中的应用: 在美国评估更严格的空气质量标准的健康效益

    Causal Estimation of Exposure Shifts with Neural Networks: Evaluating the Health Benefits of Stricter Air Quality Standards in the US. (arXiv:2302.02560v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.02560](http://arxiv.org/abs/2302.02560)

    本研究提出了一种神经网络方法，利用其理论基础和实施的可行性，从而估计连续暴露/治疗的分布对政策相关结果的因果效应。我们将此方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，通过评估美国国家环境保护局（EPA）对PM2.5的国家环境空气质量标准（NAAQS）进行修订后的健康效益。

    

    在政策研究中，估计连续性暴露/治疗的分布对感兴趣的结果的因果效应是最关键的分析任务之一。我们称之为偏移-响应函数（SRF）估计问题。现有的涉及强健因果效应估计器的神经网络方法缺乏理论保证和实际实现，用于SRF估计。受公共卫生中的关键政策问题的启发，我们开发了一种神经网络方法及其理论基础，以提供具有强健性和效率保证的SRF估计。然后，我们将我们的方法应用于包含6800万个个体和2700万个美国境内死亡事件的数据中，以估计将美国国家环境保护局（EPA）最近提议从12 μg/m³改为9 μg/m³的PM2.5的美国国家环境空气质量标准（NAAQS）的修订对结果的因果效应。我们的目标是首次估计

    In policy research, one of the most critical analytic tasks is to estimate the causal effect of a policy-relevant shift to the distribution of a continuous exposure/treatment on an outcome of interest. We call this problem shift-response function (SRF) estimation. Existing neural network methods involving robust causal-effect estimators lack theoretical guarantees and practical implementations for SRF estimation. Motivated by a key policy-relevant question in public health, we develop a neural network method and its theoretical underpinnings to estimate SRFs with robustness and efficiency guarantees. We then apply our method to data consisting of 68 million individuals and 27 million deaths across the U.S. to estimate the causal effect from revising the US National Ambient Air Quality Standards (NAAQS) for PM 2.5 from 12 $\mu g/m^3$ to 9 $\mu g/m^3$. This change has been recently proposed by the US Environmental Protection Agency (EPA). Our goal is to estimate, for the first time, the 
    
[^9]: 通过核差异实现有针对性的分离与收敛

    Targeted Separation and Convergence with Kernel Discrepancies. (arXiv:2209.12835v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.12835](http://arxiv.org/abs/2209.12835)

    通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。

    

    最大均值差异（MMDs）如核Stein差异（KSD）已经成为广泛应用的中心，包括假设检验、采样器选择、分布近似和变分推断。在每个设置中，这些基于核的差异度量需要实现（i）将目标P与其他概率测度分离，甚至（ii）控制对P的弱收敛。在本文中，我们推导了确保（i）和（ii）的新的充分必要条件。对于可分的度量空间上的MMDs，我们描述了分离Bochner可嵌入测度的核，并引入简单的条件来分离所有具有无界核的测度和用有界核来控制收敛。我们利用这些结果在$\mathbb{R}^d$上大大扩展了KSD分离和收敛控制的已知条件，并开发了首个能够精确度量对P的弱收敛的KSDs。在这个过程中，我们强调了我们的结果的影响。

    Maximum mean discrepancies (MMDs) like the kernel Stein discrepancy (KSD) have grown central to a wide range of applications, including hypothesis testing, sampler selection, distribution approximation, and variational inference. In each setting, these kernel-based discrepancy measures are required to (i) separate a target P from other probability measures or even (ii) control weak convergence to P. In this article we derive new sufficient and necessary conditions to ensure (i) and (ii). For MMDs on separable metric spaces, we characterize those kernels that separate Bochner embeddable measures and introduce simple conditions for separating all measures with unbounded kernels and for controlling convergence with bounded kernels. We use these results on $\mathbb{R}^d$ to substantially broaden the known conditions for KSD separation and convergence control and to develop the first KSDs known to exactly metrize weak convergence to P. Along the way, we highlight the implications of our res
    
[^10]: 针对高维矩阵数据的最优变量聚类

    Optimal Variable Clustering for High-Dimensional Matrix Valued Data. (arXiv:2112.12909v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2112.12909](http://arxiv.org/abs/2112.12909)

    提出了一种新的针对高维矩阵数据的特征潜变量模型，使用加权协方差矩阵的差异作为不相似度测量的层次聚类算法，理论上实现了聚类一致性，在模拟和真实数据示例中证明了方法的优越性。

    

    矩阵值数据在许多应用中日益普及。大多数现有的这种类型数据的聚类方法是针对平均模型设计的，不考虑特征的依赖结构，而该结构在高维情况下尤为重要。为了从依赖结构中提取信息进行聚类，我们提出了一种新的特征潜变量模型，该模型将特征排列成矩阵形式，并使用一些未知的成员矩阵表示行和列的聚类。在此模型下，我们进一步提出了一类使用加权协方差矩阵的差异作为不相似度测量的层次聚类算法。在理论上，我们证明了在温和的条件下，我们的算法可以在高维情况下实现聚类一致性。虽然这种一致性结果适用于我们的算法和广泛的加权协方差矩阵类别，但这个结果的条件依赖于协方差函数和加权机制的选择。通过模拟和真实数据示例，我们证明了我们的方法相对于现有方法提供了更好的聚类性能和特征选择准确性。

    Matrix valued data has become increasingly prevalent in many applications. Most of the existing clustering methods for this type of data are tailored to the mean model and do not account for the dependence structure of the features, which can be very informative, especially in high-dimensional settings. To extract the information from the dependence structure for clustering, we propose a new latent variable model for the features arranged in matrix form, with some unknown membership matrices representing the clusters for the rows and columns. Under this model, we further propose a class of hierarchical clustering algorithms using the difference of a weighted covariance matrix as the dissimilarity measure. Theoretically, we show that under mild conditions, our algorithm attains clustering consistency in the high-dimensional setting. While this consistency result holds for our algorithm with a broad class of weighted covariance matrices, the conditions for this result depend on the choic
    

