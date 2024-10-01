# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal and Adaptive Non-Stationary Dueling Bandits Under a Generalized Borda Criterion](https://arxiv.org/abs/2403.12950) | 该研究建立了首个在对抗性多臂老虎机中优化且自适应的波达动态遗憾上界，揭示了在Condorcet和Borda之间严重非平稳性可学习性的基本差异 |
| [^2] | [Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional Systems](https://arxiv.org/abs/2402.08193) | 高斯模型集成置信传播算法（GEnBP）是一种用于高维系统中高效推断的方法，通过集成卡尔曼滤波器和高斯置信传播等技术相结合，能有效处理高维状态、参数和复杂的依赖结构。 |
| [^3] | [Estimating the Local Learning Coefficient at Scale](https://arxiv.org/abs/2402.03698) | 本文提出了一种方法，可以在深度线性网络中准确地测量高达1亿参数的局部学习系数(LLC)，并证明了估计得到的LLC具有重缩放不变性。 |
| [^4] | [Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations.](http://arxiv.org/abs/2401.14142) | 基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。 |
| [^5] | [Gromov-Wassertein-like Distances in the Gaussian Mixture Models Space.](http://arxiv.org/abs/2310.11256) | 本文介绍了两种在高斯混合模型空间中的Gromov-Wasserstein类型距离，分别用于评估分布之间的距离和推导最优的点分配方案。 |
| [^6] | [It begins with a boundary: A geometric view on probabilistically robust learning.](http://arxiv.org/abs/2305.18779) | 本文探讨了深度神经网络对于对抗生成的示例缺乏鲁棒性的问题，并提出了一种从几何角度出发的新颖视角，介绍一族概率非局部周长函数来优化概率鲁棒学习（PRL）的原始表述，以提高其鲁棒性。 |
| [^7] | [Combinatorial Causal Bandits without Graph Skeleton.](http://arxiv.org/abs/2301.13392) | 本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题，提出了可在BGLMs上实现的无需图骨架的遗憾最小化算法，达到了与依赖于图结构的最先进算法相同的渐进遗憾率$O(\sqrt{T}\ln T)$。 |
| [^8] | [Controlling Moments with Kernel Stein Discrepancies.](http://arxiv.org/abs/2211.05408) | 本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。 |
| [^9] | [Approximate Cross-validated Mean Estimates for Bayesian Hierarchical Regression Models.](http://arxiv.org/abs/2011.14238) | 我们提出了一种用于贝叶斯层次回归模型的近似交叉验证均值估计的新方法，通过在方差-协方差参数上进行条件，将交叉验证问题转化为简单的优化问题，从而提高了大型BHRMs的可行性。 |

# 详细

[^1]: 优化的自适应非平稳对抗性多臂老虎机在广义波达准则下

    Optimal and Adaptive Non-Stationary Dueling Bandits Under a Generalized Borda Criterion

    [https://arxiv.org/abs/2403.12950](https://arxiv.org/abs/2403.12950)

    该研究建立了首个在对抗性多臂老虎机中优化且自适应的波达动态遗憾上界，揭示了在Condorcet和Borda之间严重非平稳性可学习性的基本差异

    

    在对抗性多臂老虎机中，学习者接收臂之间的偏好反馈，并将某个臂的遗憾定义为其相对于优胜臂的次优性。更具挑战性和实践动机的非平稳对抗性多臂老虎机变体，在这种变体中，偏好随时间变化，已经成为近期多项工作的焦点。目标是设计出算法，而无需提前了解变化量。已知结果的大部分研究了孔多塞优胜者设置，其中优先于其他任何臂的臂在任何时候都存在。然而，这样的优胜者可能并不存在，为了对比，此问题的波达版本（始终有明确定义）却受到了很少关注。在这项工作中，我们建立了第一个最优和自适应的波达动态遗憾上界，突显了在孔多塞和波达之间的严重非平稳性可学习性的基本差异。

    arXiv:2403.12950v1 Announce Type: new  Abstract: In dueling bandits, the learner receives preference feedback between arms, and the regret of an arm is defined in terms of its suboptimality to a winner arm. The more challenging and practically motivated non-stationary variant of dueling bandits, where preferences change over time, has been the focus of several recent works (Saha and Gupta, 2022; Buening and Saha, 2023; Suk and Agarwal, 2023). The goal is to design algorithms without foreknowledge of the amount of change.   The bulk of known results here studies the Condorcet winner setting, where an arm preferred over any other exists at all times. Yet, such a winner may not exist and, to contrast, the Borda version of this problem (which is always well-defined) has received little attention. In this work, we establish the first optimal and adaptive Borda dynamic regret upper bound, which highlights fundamental differences in the learnability of severe non-stationarity between Condorce
    
[^2]: 高斯模型集成置信传播用于高维系统中的高效推断

    Gaussian Ensemble Belief Propagation for Efficient Inference in High-Dimensional Systems

    [https://arxiv.org/abs/2402.08193](https://arxiv.org/abs/2402.08193)

    高斯模型集成置信传播算法（GEnBP）是一种用于高维系统中高效推断的方法，通过集成卡尔曼滤波器和高斯置信传播等技术相结合，能有效处理高维状态、参数和复杂的依赖结构。

    

    高维模型中的高效推断仍然是机器学习中的一个核心挑战。本文介绍了一种名为高斯模型集成置信传播（GEnBP）算法的方法，该方法是集成卡尔曼滤波器和高斯置信传播（GaBP）方法的结合。GEnBP通过在图模型结构中传递低秩本地信息来更新集成模型。这种组合继承了每种方法的有利特性。集成技术使得GEnBP能够处理高维状态、参数和复杂的、嘈杂的黑箱生成过程。在图模型结构中使用本地信息确保了该方法适用于分布式计算，并能高效地处理复杂的依赖结构。当集成大小远小于推断维度时，GEnBP特别有优势。这种情况在空时建模、图像处理和物理模型反演等领域经常出现。GEnBP可以应用于一般性问题。

    Efficient inference in high-dimensional models remains a central challenge in machine learning. This paper introduces the Gaussian Ensemble Belief Propagation (GEnBP) algorithm, a fusion of the Ensemble Kalman filter and Gaussian belief propagation (GaBP) methods. GEnBP updates ensembles by passing low-rank local messages in a graphical model structure. This combination inherits favourable qualities from each method. Ensemble techniques allow GEnBP to handle high-dimensional states, parameters and intricate, noisy, black-box generation processes. The use of local messages in a graphical model structure ensures that the approach is suited to distributed computing and can efficiently handle complex dependence structures. GEnBP is particularly advantageous when the ensemble size is considerably smaller than the inference dimension. This scenario often arises in fields such as spatiotemporal modelling, image processing and physical model inversion. GEnBP can be applied to general problem s
    
[^3]: 在大规模情况下估计局部学习系数

    Estimating the Local Learning Coefficient at Scale

    [https://arxiv.org/abs/2402.03698](https://arxiv.org/abs/2402.03698)

    本文提出了一种方法，可以在深度线性网络中准确地测量高达1亿参数的局部学习系数(LLC)，并证明了估计得到的LLC具有重缩放不变性。

    

    局部学习系数(LLC)是一种量化模型复杂性的原则性方法，最初是在贝叶斯统计中使用奇异学习理论(SLT)推导出来的。已知有几种数值估计局部学习系数的方法，但迄今为止这些方法尚未扩展到现代深度学习架构或数据集的规模。通过在arXiv:2308.12108 [stat.ML]中开发的一种方法，我们经验证明可以准确和自洽地测量深度线性网络(DLN)中高达1亿参数的局部学习系数(LLC)。我们还证明了估计得到的LLC具有理论数量所具备的重缩放不变性。

    The \textit{local learning coefficient} (LLC) is a principled way of quantifying model complexity, originally derived in the context of Bayesian statistics using singular learning theory (SLT). Several methods are known for numerically estimating the local learning coefficient, but so far these methods have not been extended to the scale of modern deep learning architectures or data sets. Using a method developed in {\tt arXiv:2308.12108 [stat.ML]} we empirically show how the LLC may be measured accurately and self-consistently for deep linear networks (DLNs) up to 100M parameters. We also show that the estimated LLC has the rescaling invariance that holds for the theoretical quantity.
    
[^4]: 基于能量的概念瓶颈模型：统一预测、概念干预和条件解释

    Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Conditional Interpretations. (arXiv:2401.14142v1 [cs.CV])

    [http://arxiv.org/abs/2401.14142](http://arxiv.org/abs/2401.14142)

    基于能量的概念瓶颈模型统一了预测、概念干预和条件解释的功能，解决了现有方法在高阶非线性相互作用和复杂条件依赖关系上的限制。

    

    现有方法，如概念瓶颈模型 (CBM)，在为黑盒深度学习模型提供基于概念的解释方面取得了成功。它们通常通过在给定输入的情况下预测概念，然后在给定预测的概念的情况下预测最终的类别标签。然而，它们经常无法捕捉到概念之间的高阶非线性相互作用，例如纠正一个预测的概念（例如“黄色胸部”）无法帮助纠正高度相关的概念（例如“黄色腹部”），导致最终准确率不理想；它们无法自然地量化不同概念和类别标签之间的复杂条件依赖关系（例如对于一个带有类别标签“Kentucky Warbler”和概念“黑色嘴巴”的图像，模型能够正确预测另一个概念“黑色冠”的概率是多少），因此无法提供关于黑盒模型工作原理更深层次的洞察。针对这些限制，我们提出了基于能量的概念瓶颈模型（Energy-based Concept Bottleneck Models）。

    Existing methods, such as concept bottleneck models (CBMs), have been successful in providing concept-based interpretations for black-box deep learning models. They typically work by predicting concepts given the input and then predicting the final class label given the predicted concepts. However, (1) they often fail to capture the high-order, nonlinear interaction between concepts, e.g., correcting a predicted concept (e.g., "yellow breast") does not help correct highly correlated concepts (e.g., "yellow belly"), leading to suboptimal final accuracy; (2) they cannot naturally quantify the complex conditional dependencies between different concepts and class labels (e.g., for an image with the class label "Kentucky Warbler" and a concept "black bill", what is the probability that the model correctly predicts another concept "black crown"), therefore failing to provide deeper insight into how a black-box model works. In response to these limitations, we propose Energy-based Concept Bot
    
[^5]: 在高斯混合模型空间中引入了类似于Gromov-Wassertein的距离

    Gromov-Wassertein-like Distances in the Gaussian Mixture Models Space. (arXiv:2310.11256v1 [stat.ML])

    [http://arxiv.org/abs/2310.11256](http://arxiv.org/abs/2310.11256)

    本文介绍了两种在高斯混合模型空间中的Gromov-Wasserstein类型距离，分别用于评估分布之间的距离和推导最优的点分配方案。

    

    本文介绍了两种在高斯混合模型集合上的Gromov-Wasserstein类型距离。第一种距离是在高斯测度空间上两个离散分布的Gromov-Wasserstein距离。该距离可以作为Gromov-Wasserstein的替代，用于评估分布之间的距离，但不能直接推导出最优的运输方案。为了设计出这样的运输方案，我们引入了另一种在不可比较的空间中的测度之间的距离，该距离与Gromov-Wasserstein密切相关。当将允许的运输耦合限制为高斯混合模型时，这定义了另一种高斯混合模型之间的距离，可以作为Gromov-Wasserstein的另一种替代，并允许推导出最优的点分配方案。

    In this paper, we introduce two Gromov-Wasserstein-type distances on the set of Gaussian mixture models. The first one takes the form of a Gromov-Wasserstein distance between two discrete distributionson the space of Gaussian measures. This distance can be used as an alternative to Gromov-Wasserstein for applications which only require to evaluate how far the distributions are from each other but does not allow to derive directly an optimal transportation plan between clouds of points. To design a way to define such a transportation plan, we introduce another distance between measures living in incomparable spaces that turns out to be closely related to Gromov-Wasserstein. When restricting the set of admissible transportation couplings to be themselves Gaussian mixture models in this latter, this defines another distance between Gaussian mixture models that can be used as another alternative to Gromov-Wasserstein and which allows to derive an optimal assignment between points. Finally,
    
[^6]: 从几何角度看待概率鲁棒学习中的边界问题

    It begins with a boundary: A geometric view on probabilistically robust learning. (arXiv:2305.18779v1 [cs.LG])

    [http://arxiv.org/abs/2305.18779](http://arxiv.org/abs/2305.18779)

    本文探讨了深度神经网络对于对抗生成的示例缺乏鲁棒性的问题，并提出了一种从几何角度出发的新颖视角，介绍一族概率非局部周长函数来优化概率鲁棒学习（PRL）的原始表述，以提高其鲁棒性。

    

    尽管深度神经网络在许多分类任务上已经实现了超人类的表现，但它们往往对于对抗生成的示例缺乏鲁棒性，因此需要将经验风险最小化（ERM）重构为对抗性鲁棒的框架。最近，关注点已经转向了介于对抗性训练提供的鲁棒性和ERM提供的更高干净准确性和更快训练时间之间的方法。本文从几何角度出发，对一种这样的方法——概率鲁棒学习（PRL）（Robey等人，ICML，2022）进行了新颖的几何视角的探讨。我们提出了一个几何框架来理解PRL，这使我们能够确定其原始表述中的微妙缺陷，并介绍了一族概率非局部周长函数来解决这一问题。我们使用新颖的松弛方法证明了解的存在，并研究了引入的非局部周长函数的特性以及局部极限。

    Although deep neural networks have achieved super-human performance on many classification tasks, they often exhibit a worrying lack of robustness towards adversarially generated examples. Thus, considerable effort has been invested into reformulating Empirical Risk Minimization (ERM) into an adversarially robust framework. Recently, attention has shifted towards approaches which interpolate between the robustness offered by adversarial training and the higher clean accuracy and faster training times of ERM. In this paper, we take a fresh and geometric view on one such method -- Probabilistically Robust Learning (PRL) (Robey et al., ICML, 2022). We propose a geometric framework for understanding PRL, which allows us to identify a subtle flaw in its original formulation and to introduce a family of probabilistic nonlocal perimeter functionals to address this. We prove existence of solutions using novel relaxation methods and study properties as well as local limits of the introduced per
    
[^7]: 不考虑图骨架的组合因果赌博机

    Combinatorial Causal Bandits without Graph Skeleton. (arXiv:2301.13392v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13392](http://arxiv.org/abs/2301.13392)

    本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题，提出了可在BGLMs上实现的无需图骨架的遗憾最小化算法，达到了与依赖于图结构的最先进算法相同的渐进遗憾率$O(\sqrt{T}\ln T)$。

    

    在组合因果赌博机问题中，学习代理在每一轮选择一组变量进行干预，收集观测变量的反馈以最小化期望遗憾或样本复杂度。先前的工作研究了一般因果模型和二值广义线性模型（BGLMs）中的问题。但是，它们都需要先验知识来构建因果关系图。本文研究了在二值一般因果模型和BGLMs上不考虑图骨架的组合因果赌博机问题。我们首先在一般因果模型上提供了累积遗憾的指数下限。然后，我们设计了一种无需图骨架来实现BGLMs的遗憾最小化算法，表明它仍然达到$O(\sqrt{T}\ln T)$的期望遗憾。这个渐进的遗憾率与依赖于图结构的最先进算法相同。

    In combinatorial causal bandits (CCB), the learning agent chooses a subset of variables in each round to intervene and collects feedback from the observed variables to minimize expected regret or sample complexity. Previous works study this problem in both general causal models and binary generalized linear models (BGLMs). However, all of them require prior knowledge of causal graph structure. This paper studies the CCB problem without the graph structure on binary general causal models and BGLMs. We first provide an exponential lower bound of cumulative regrets for the CCB problem on general causal models. To overcome the exponentially large space of parameters, we then consider the CCB problem on BGLMs. We design a regret minimization algorithm for BGLMs even without the graph skeleton and show that it still achieves $O(\sqrt{T}\ln T)$ expected regret. This asymptotic regret is the same as the state-of-art algorithms relying on the graph structure. Moreover, we sacrifice the regret t
    
[^8]: 用核斯坦离差控制矩

    Controlling Moments with Kernel Stein Discrepancies. (arXiv:2211.05408v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.05408](http://arxiv.org/abs/2211.05408)

    本研究分析了核斯坦离差（KSD）控制性质，发现标准KSD无法控制矩的收敛，提出了可控制矩和弱收敛的下游扩散KSD，并且发展了可以准确描述$q$-Wasserstein收敛的KSD。

    

    核斯坦离差（KSD）用于衡量分布逼近的质量，并且可以在目标密度具有不可计算的归一化常数时计算。显著的应用包括诊断近似MCMC采样器和非归一化统计模型的适配度检验。本文分析了KSD的收敛控制性质。我们首先证明了用于弱收敛控制的标准KSD无法控制矩的收敛。为了解决这个限制，我们提供了一组充分条件，下游扩散KSD可以同时控制矩和弱收敛。作为一个直接的结果，我们发展了对于每个$q>0$，第一组已知可以准确描述$q$-Wasserstein收敛的KSD。

    Kernel Stein discrepancies (KSDs) measure the quality of a distributional approximation and can be computed even when the target density has an intractable normalizing constant. Notable applications include the diagnosis of approximate MCMC samplers and goodness-of-fit tests for unnormalized statistical models. The present work analyzes the convergence control properties of KSDs. We first show that standard KSDs used for weak convergence control fail to control moment convergence. To address this limitation, we next provide sufficient conditions under which alternative diffusion KSDs control both moment and weak convergence. As an immediate consequence we develop, for each $q > 0$, the first KSDs known to exactly characterize $q$-Wasserstein convergence.
    
[^9]: Bayesian层次回归模型的近似交叉验证均值估计

    Approximate Cross-validated Mean Estimates for Bayesian Hierarchical Regression Models. (arXiv:2011.14238v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2011.14238](http://arxiv.org/abs/2011.14238)

    我们提出了一种用于贝叶斯层次回归模型的近似交叉验证均值估计的新方法，通过在方差-协方差参数上进行条件，将交叉验证问题转化为简单的优化问题，从而提高了大型BHRMs的可行性。

    

    我们引入了一种新的方法，用于获取贝叶斯层次回归模型(BHRMs)的交叉验证预测估计。贝叶斯层次模型以其能够建模复杂的依赖结构并提供概率不确定性估计而受到欢迎，但运行的计算开销很大。因此，交叉验证(CV)不是评估BHRMs预测性能的常见实践。我们的方法避免了为每个交叉验证折叠重新运行计算开销昂贵的估计方法的需要，使CV在大型BHRMs中更可行。通过在方差-协方差参数上进行条件，将CV问题从基于概率的抽样转化为简单熟悉的优化问题。在许多情况下，这产生的估计与完整的CV等效。我们提供理论结果，并在公开可用的数据和模拟中证明其有效性。

    We introduce a novel procedure for obtaining cross-validated predictive estimates for Bayesian hierarchical regression models (BHRMs). Bayesian hierarchical models are popular for their ability to model complex dependence structures and provide probabilistic uncertainty estimates, but can be computationally expensive to run. Cross-validation (CV) is therefore not a common practice to evaluate the predictive performance of BHRMs. Our method circumvents the need to re-run computationally costly estimation methods for each cross-validation fold and makes CV more feasible for large BHRMs. By conditioning on the variance-covariance parameters, we shift the CV problem from probability-based sampling to a simple and familiar optimization problem. In many cases, this produces estimates which are equivalent to full CV. We provide theoretical results and demonstrate its efficacy on publicly available data and in simulations.
    

