# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^2] | [Logistic-beta processes for modeling dependent random probabilities with beta marginals](https://arxiv.org/abs/2402.07048) | 本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。 |
| [^3] | [Learning from Time Series under Temporal Label Noise](https://arxiv.org/abs/2402.04398) | 该论文研究了在时间序列下处理时间标签噪声的问题，提出了一种可以从数据中直接估计时间标签噪声函数并训练出噪声容忍分类器的方法，并在实验中展示了该方法在各种时间标签噪声函数下都取得了最先进的性能。 |
| [^4] | [$\alpha$-Divergence Loss Function for Neural Density Ratio Estimation](https://arxiv.org/abs/2402.02041) | 本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。 |
| [^5] | [Analyzing Sharpness-aware Minimization under Overparameterization](https://arxiv.org/abs/2311.17539) | 本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。 |
| [^6] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^7] | [Entropic Matching for Expectation Propagation of Markov Jump Processes.](http://arxiv.org/abs/2309.15604) | 本文提出了一个基于熵匹配框架的新的可处理的推断方案，可以嵌入到期望传播算法中，对于描述离散状态空间过程的Markov跳跃过程的统计推断问题具有重要意义。我们展示了我们方法的有效性，并通过提供一类近似分布的闭式结果以及应用于化学反应网络的一般类别来加以论证。此外，我们通过一个近似的期望最大化程序导出了潜在参数的点估计的闭式表达式，并在各种化学反应网络示例中评估了我们的方法的性能。我们还讨论了该方法的局限性和未来的潜力。 |
| [^8] | [Simultaneous inference for generalized linear models with unmeasured confounders.](http://arxiv.org/abs/2309.07261) | 本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。 |
| [^9] | [Nystr\"om $M$-Hilbert-Schmidt Independence Criterion.](http://arxiv.org/abs/2302.09930) | 这项研究提出了Nystr\"om $M$-Hilbert-Schmidt独立准则，针对大规模应用的二次计算瓶颈问题进行了解决，并兼顾了多个随机变量的推广情况和理论保证。 |
| [^10] | [Data Augmentation in the Underparameterized and Overparameterized Regimes.](http://arxiv.org/abs/2202.09134) | 这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。 |
| [^11] | [Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves.](http://arxiv.org/abs/2111.03950) | 本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。 |

# 详细

[^1]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^2]: 用于建模具有beta边际分布的相关随机概率的logistic-beta过程

    Logistic-beta processes for modeling dependent random probabilities with beta marginals

    [https://arxiv.org/abs/2402.07048](https://arxiv.org/abs/2402.07048)

    本文提出了一种新颖的logistic-beta过程用于建模具有beta边际分布的相关随机概率。该过程具有灵活的相关结构和计算优势，并通过非参数二分类回归模拟研究进行了验证。

    

    beta分布被广泛应用于概率建模，并在统计学和机器学习中被广泛使用，尤其在贝叶斯非参数领域。尽管其被广泛使用，但在建模相关随机概率的灵活和计算方便的随机过程扩展方面，相关工作有限。我们提出了一种新颖的随机过程，称为logistic-beta过程，其logistic变换生成具有常见beta边际分布的随机过程。类似于高斯过程，logistic-beta过程可以建模离散和连续域（例如空间或时间）上的相关性，并通过相关核函数具有高度灵活的相关结构。此外，它的正态方差-均值混合表示导致了高效的后验推理算法。通过非参数二分类回归模拟研究，展示了logistic-beta过程的灵活性和计算优势。

    The beta distribution serves as a canonical tool for modeling probabilities and is extensively used in statistics and machine learning, especially in the field of Bayesian nonparametrics. Despite its widespread use, there is limited work on flexible and computationally convenient stochastic process extensions for modeling dependent random probabilities. We propose a novel stochastic process called the logistic-beta process, whose logistic transformation yields a stochastic process with common beta marginals. Similar to the Gaussian process, the logistic-beta process can model dependence on both discrete and continuous domains, such as space or time, and has a highly flexible dependence structure through correlation kernels. Moreover, its normal variance-mean mixture representation leads to highly effective posterior inference algorithms. The flexibility and computational benefits of logistic-beta processes are demonstrated through nonparametric binary regression simulation studies. Fur
    
[^3]: 学习在时间序列下处理时间标签噪声

    Learning from Time Series under Temporal Label Noise

    [https://arxiv.org/abs/2402.04398](https://arxiv.org/abs/2402.04398)

    该论文研究了在时间序列下处理时间标签噪声的问题，提出了一种可以从数据中直接估计时间标签噪声函数并训练出噪声容忍分类器的方法，并在实验中展示了该方法在各种时间标签噪声函数下都取得了最先进的性能。

    

    许多顺序分类任务受到随时间变化的标签噪声的影响。这种噪声可能会导致标签质量随时间改善、恶化或周期性变化。我们首先提出和系统化了时间标签噪声的概念，这是关于时间序列顺序分类的一个未经研究的问题。在这种设置下，多个标签连续记录，同时受到一个与时间相关的噪声函数的干扰。我们首先展示了建模时间标签噪声函数的重要性，以及现有方法的持续低效。然后，我们提出了一种直接从数据中估计时间标签噪声函数的方法，可以训练出对噪声具有容忍性的分类器。我们展示了我们的方法在各种各样的时间标签噪声函数下，使用真实和合成数据在性能上达到了最先进水平。

    Many sequential classification tasks are affected by label noise that varies over time. Such noise can cause label quality to improve, worsen, or periodically change over time. We first propose and formalize temporal label noise, an unstudied problem for sequential classification of time series. In this setting, multiple labels are recorded in sequence while being corrupted by a time-dependent noise function. We first demonstrate the importance of modelling the temporal nature of the label noise function and how existing methods will consistently underperform. We then propose methods that can train noise-tolerant classifiers by estimating the temporal label noise function directly from data. We show that our methods lead to state-of-the-art performance in the presence of diverse temporal label noise functions using real and synthetic data.
    
[^4]: 用于神经密度比估计的$\alpha$-散度损失函数

    $\alpha$-Divergence Loss Function for Neural Density Ratio Estimation

    [https://arxiv.org/abs/2402.02041](https://arxiv.org/abs/2402.02041)

    本文提出了一种应用于神经密度比估计的$\alpha$-散度损失函数($\alpha$-Div)，通过简洁实现和稳定优化解决了现有方法中存在的优化问题。实验证明了这种损失函数的稳定性，并提出了对DRE任务的估计准确性的研究，同时给出了样本要求的解决方案。

    

    最近，神经网络在机器学习中的基础技术密度比估计(DRE)方面取得了最先进的结果。然而，现有方法因DRE的损失函数而出现了优化问题：KL散度需要大样本，训练损失梯度消失，损失函数梯度有偏。因此，本文提出了一种提供简洁实现和稳定优化的$\alpha$-散度损失函数($\alpha$-Div)。此外，还给出了对所提出的损失函数的技术验证。实验证明了所提出的损失函数的稳定性，并研究了DRE任务的估计准确性。此外，本研究还提出了使用所提出的损失函数进行DRE的样本要求，以$L_1$误差的上界联系起来，该上界将高维度DRE任务中的维度诅咒作为一个共同问题。

    Recently, neural networks have produced state-of-the-art results for density-ratio estimation (DRE), a fundamental technique in machine learning. However, existing methods bear optimization issues that arise from the loss functions of DRE: a large sample requirement of Kullback--Leibler (KL)-divergence, vanishing of train loss gradients, and biased gradients of the loss functions. Thus, an $\alpha$-divergence loss function ($\alpha$-Div) that offers concise implementation and stable optimization is proposed in this paper. Furthermore, technical justifications for the proposed loss function are presented. The stability of the proposed loss function is empirically demonstrated and the estimation accuracy of DRE tasks is investigated. Additionally, this study presents a sample requirement for DRE using the proposed loss function in terms of the upper bound of $L_1$ error, which connects a curse of dimensionality as a common problem in high-dimensional DRE tasks.
    
[^5]: 在过参数化下分析锐度感知最小化

    Analyzing Sharpness-aware Minimization under Overparameterization

    [https://arxiv.org/abs/2311.17539](https://arxiv.org/abs/2311.17539)

    本文分析了在过参数化条件下的锐度感知最小化方法。通过实证和理论结果，发现过参数化对锐度感知最小化具有重要影响，并且在过参数化增加的情况下，锐度感知最小化仍然受益。

    

    在训练过参数化的神经网络时，尽管训练损失相同，但可以得到具有不同泛化能力的极小值。有证据表明，极小值的锐度与其泛化误差之间存在相关性，因此已经做出了更多努力开发一种优化方法，以显式地找到扁平极小值作为更具有泛化能力的解。然而，至今为止，关于过参数化对锐度感知最小化（SAM）策略的影响的研究还不多。在这项工作中，我们分析了在不同程度的过参数化下的SAM，并提出了实证和理论结果，表明过参数化对SAM具有重要影响。具体而言，我们进行了广泛的数值实验，涵盖了各个领域，并表明存在一种一致的趋势，即SAM在过参数化增加的情况下仍然受益。我们还发现了一些令人信服的案例，说明了过参数化的影响。

    Training an overparameterized neural network can yield minimizers of different generalization capabilities despite the same level of training loss. With evidence that suggests a correlation between sharpness of minima and their generalization errors, increasing efforts have been made to develop an optimization method to explicitly find flat minima as more generalizable solutions. However, this sharpness-aware minimization (SAM) strategy has not been studied much yet as to whether and how it is affected by overparameterization.   In this work, we analyze SAM under overparameterization of varying degrees and present both empirical and theoretical results that indicate a critical influence of overparameterization on SAM. Specifically, we conduct extensive numerical experiments across various domains, and show that there exists a consistent trend that SAM continues to benefit from increasing overparameterization. We also discover compelling cases where the effect of overparameterization is
    
[^6]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^7]: Entropic Matching用于Markov跳跃过程的期望传播的熵匹配

    Entropic Matching for Expectation Propagation of Markov Jump Processes. (arXiv:2309.15604v1 [cs.LG])

    [http://arxiv.org/abs/2309.15604](http://arxiv.org/abs/2309.15604)

    本文提出了一个基于熵匹配框架的新的可处理的推断方案，可以嵌入到期望传播算法中，对于描述离散状态空间过程的Markov跳跃过程的统计推断问题具有重要意义。我们展示了我们方法的有效性，并通过提供一类近似分布的闭式结果以及应用于化学反应网络的一般类别来加以论证。此外，我们通过一个近似的期望最大化程序导出了潜在参数的点估计的闭式表达式，并在各种化学反应网络示例中评估了我们的方法的性能。我们还讨论了该方法的局限性和未来的潜力。

    

    本文解决了潜在连续时间随机过程的统计推断问题，该问题通常难以处理，特别是对于由Markov跳跃过程描述的离散状态空间过程。为了克服这个问题，我们提出了一种新的可处理的推断方案，基于熵匹配框架，可以嵌入到众所周知的期望传播算法中。我们通过为一类简单的近似分布提供闭式结果，并将其应用于化学反应网络的一般类别，该类别是系统生物学建模的重要工具，来证明我们方法的有效性。此外，我们使用近似的期望最大化程序导出了潜在参数的点估计的闭式表达式。我们评估了我们方法在各种化学反应网络示例中的性能，包括随机的Lotka-Voltera示例，并讨论了它的局限性和未来的潜力。

    This paper addresses the problem of statistical inference for latent continuous-time stochastic processes, which is often intractable, particularly for discrete state space processes described by Markov jump processes. To overcome this issue, we propose a new tractable inference scheme based on an entropic matching framework that can be embedded into the well-known expectation propagation algorithm. We demonstrate the effectiveness of our method by providing closed-form results for a simple family of approximate distributions and apply it to the general class of chemical reaction networks, which are a crucial tool for modeling in systems biology. Moreover, we derive closed form expressions for point estimation of the underlying parameters using an approximate expectation maximization procedure. We evaluate the performance of our method on various chemical reaction network instantiations, including a stochastic Lotka-Voltera example, and discuss its limitations and potential for future 
    
[^8]: 具有未测混淆因素的广义线性模型的同时推断

    Simultaneous inference for generalized linear models with unmeasured confounders. (arXiv:2309.07261v1 [stat.ME])

    [http://arxiv.org/abs/2309.07261](http://arxiv.org/abs/2309.07261)

    本文研究了存在混淆效应时的广义线性模型的大规模假设检验问题，并提出了一种利用正交结构和线性投影的统计估计和推断框架，解决了由于未测混淆因素引起的偏差问题。

    

    在基因组研究中，常常进行成千上万个同时假设检验，以确定差异表达的基因。然而，由于存在未测混淆因素，许多标准统计方法可能存在严重的偏差。本文研究了存在混淆效应时的多元广义线性模型的大规模假设检验问题。在任意混淆机制下，我们提出了一个统一的统计估计和推断方法，利用正交结构并将线性投影整合到三个关键阶段中。首先，利用多元响应变量分离边际和不相关的混淆效应，恢复混淆系数的列空间。随后，利用$\ell_1$正则化进行稀疏性估计，并强加正交性限制于混淆系数，联合估计潜在因子和主要效应。最后，我们结合投影和加权偏差校正步骤。

    Tens of thousands of simultaneous hypothesis tests are routinely performed in genomic studies to identify differentially expressed genes. However, due to unmeasured confounders, many standard statistical approaches may be substantially biased. This paper investigates the large-scale hypothesis testing problem for multivariate generalized linear models in the presence of confounding effects. Under arbitrary confounding mechanisms, we propose a unified statistical estimation and inference framework that harnesses orthogonal structures and integrates linear projections into three key stages. It first leverages multivariate responses to separate marginal and uncorrelated confounding effects, recovering the confounding coefficients' column space. Subsequently, latent factors and primary effects are jointly estimated, utilizing $\ell_1$-regularization for sparsity while imposing orthogonality onto confounding coefficients. Finally, we incorporate projected and weighted bias-correction steps 
    
[^9]: Nystr\"om $M$-Hilbert-Schmidt独立准则

    Nystr\"om $M$-Hilbert-Schmidt Independence Criterion. (arXiv:2302.09930v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.09930](http://arxiv.org/abs/2302.09930)

    这项研究提出了Nystr\"om $M$-Hilbert-Schmidt独立准则，针对大规模应用的二次计算瓶颈问题进行了解决，并兼顾了多个随机变量的推广情况和理论保证。

    

    核技术是数据科学中最受欢迎和强大的方法之一。核的广泛应用的关键特性包括：(i) 它们针对的领域数量多，(ii) 与核相关的函数类具有Hilbert结构，便于统计分析，以及(iii) 它们能够以不丢失信息的方式表示概率分布。这些特性导致了Hilbert-Schmidt独立准则(HSIC)的巨大成功，该准则能够在温和条件下捕捉随机变量的联合独立性，并允许具有二次计算复杂性的闭式估计器(相对于样本大小)。为了解决大规模应用中的二次计算瓶颈问题，已经提出了多个HSIC近似估计器，然而这些估计器限制于$M=2$个随机变量，不能自然地推广到$M \geq 2$的情况，并且缺乏理论保证。在这项工作中，我们提出了一个Nystr\"om $M$-Hilbert-Schmidt独立准则来解决这个问题。

    Kernel techniques are among the most popular and powerful approaches of data science. Among the key features that make kernels ubiquitous are (i) the number of domains they have been designed for, (ii) the Hilbert structure of the function class associated to kernels facilitating their statistical analysis, and (iii) their ability to represent probability distributions without loss of information. These properties give rise to the immense success of Hilbert-Schmidt independence criterion (HSIC) which is able to capture joint independence of random variables under mild conditions, and permits closed-form estimators with quadratic computational complexity (w.r.t. the sample size). In order to alleviate the quadratic computational bottleneck in large-scale applications, multiple HSIC approximations have been proposed, however these estimators are restricted to $M=2$ random variables, do not extend naturally to the $M\ge 2$ case, and lack theoretical guarantees. In this work, we propose an
    
[^10]: 在欠参数化和过参数化的模式中的数据增强

    Data Augmentation in the Underparameterized and Overparameterized Regimes. (arXiv:2202.09134v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.09134](http://arxiv.org/abs/2202.09134)

    这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。

    

    我们提供了确切量化数据增强如何影响估计的方差和极限分布的结果，并详细分析了几个具体模型。结果证实了机器学习实践中的一些观察，但也得出了意外的发现：数据增强可能会增加而不是减少估计的不确定性，比如经验预测风险。它可以充当正则化器，但在某些高维问题中却无法实现，并且可能会改变经验风险的双重下降峰值。总的来说，分析表明数据增强被赋予的几个属性要么是真的，要么是假的，而是取决于多个因素的组合-特别是数据分布，估计器的属性以及样本大小，增强数量和维数的相互作用。我们的主要理论工具是随机转换的高维随机向量的函数的极限定理。

    We provide results that exactly quantify how data augmentation affects the variance and limiting distribution of estimates, and analyze several specific models in detail. The results confirm some observations made in machine learning practice, but also lead to unexpected findings: Data augmentation may increase rather than decrease the uncertainty of estimates, such as the empirical prediction risk. It can act as a regularizer, but fails to do so in certain high-dimensional problems, and it may shift the double-descent peak of an empirical risk. Overall, the analysis shows that several properties data augmentation has been attributed with are not either true or false, but rather depend on a combination of factors -- notably the data distribution, the properties of the estimator, and the interplay of sample size, number of augmentations, and dimension. Our main theoretical tool is a limit theorem for functions of randomly transformed, high-dimensional random vectors. The proof draws on 
    
[^11]: 序贯核嵌入用于介导和时变剂量响应曲线

    Sequential Kernel Embedding for Mediated and Time-Varying Dose Response Curves. (arXiv:2111.03950v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.03950](http://arxiv.org/abs/2111.03950)

    本论文提出了一种基于核岭回归的简单非参数估计方法，可以用于估计介导和时变剂量响应曲线。通过引入序贯核嵌入技术，我们实现了对复杂因果估计的简化。通过模拟实验和真实数据的估计结果，证明了该方法的强大性能和普适性。

    

    我们提出了基于核岭回归的介导和时变剂量响应曲线的简单非参数估计器。通过嵌入Pearl的介导公式和Robins的g公式与核函数，我们允许处理、介导者和协变量在一般空间中连续变化，也允许非线性的处理-混淆因素反馈。我们的关键创新是一种称为序贯核嵌入的再生核希尔伯特空间技术，我们使用它来构建复杂因果估计的简单估计器。我们的估计器保留了经典识别的普适性，同时实现了非渐进均匀收敛速度。在具有许多协变量的非线性模拟中，我们展示了强大的性能。我们估计了美国职业训练团的介导和时变剂量响应曲线，并清洁可能成为未来工作基准的数据。我们将我们的结果推广到介导和时变处理效应以及反事实分布，验证了半参数效率。

    We propose simple nonparametric estimators for mediated and time-varying dose response curves based on kernel ridge regression. By embedding Pearl's mediation formula and Robins' g-formula with kernels, we allow treatments, mediators, and covariates to be continuous in general spaces, and also allow for nonlinear treatment-confounder feedback. Our key innovation is a reproducing kernel Hilbert space technique called sequential kernel embedding, which we use to construct simple estimators for complex causal estimands. Our estimators preserve the generality of classic identification while also achieving nonasymptotic uniform rates. In nonlinear simulations with many covariates, we demonstrate strong performance. We estimate mediated and time-varying dose response curves of the US Job Corps, and clean data that may serve as a benchmark in future work. We extend our results to mediated and time-varying treatment effects and counterfactual distributions, verifying semiparametric efficiency 
    

