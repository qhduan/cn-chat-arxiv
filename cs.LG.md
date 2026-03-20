# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CADGL: Context-Aware Deep Graph Learning for Predicting Drug-Drug Interactions](https://arxiv.org/abs/2403.17210) | 通过CADGL框架，利用上下文感知深度图学习来预测药物-药物相互作用，解决了现有DDI预测模型在泛化、特征提取和现实应用方面的挑战 |
| [^2] | [On Minimal Depth in Neural Networks](https://arxiv.org/abs/2402.15315) | 本研究研究了神经网络中关于最小深度的问题，特别关注了ReLU神经网络的表达能力和最小深度与CPWL函数的关系。 |
| [^3] | [Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts](https://arxiv.org/abs/2402.01972) | 这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。 |
| [^4] | [Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods](https://arxiv.org/abs/2312.08531) | 研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。 |
| [^5] | [Hidden yet quantifiable: A lower bound for confounding strength using randomized trials](https://arxiv.org/abs/2312.03871) | 利用随机试验设计了一种统计检验，能够量化未观察到的混淆强度，并估计其下界，有效应用于现实世界中识别混淆。 |
| [^6] | [Inverse classification with logistic and softmax classifiers: efficient optimization.](http://arxiv.org/abs/2309.08945) | 本文研究了逻辑回归和softmax分类器中的逆向分类问题，并提出了高效的解决方法，可以在交互式或实时应用中获得准确解。 |

# 详细

[^1]: CADGL: 上下文感知深度图学习用于预测药物-药物相互作用

    CADGL: Context-Aware Deep Graph Learning for Predicting Drug-Drug Interactions

    [https://arxiv.org/abs/2403.17210](https://arxiv.org/abs/2403.17210)

    通过CADGL框架，利用上下文感知深度图学习来预测药物-药物相互作用，解决了现有DDI预测模型在泛化、特征提取和现实应用方面的挑战

    

    药物-药物相互作用（DDIs）的研究是药物开发过程中的一个关键元素。DDIs发生在一个药物的性质受其他药物包含的影响时。检测有利的DDIs有可能为在实际设置中应用的创新药物的创造和推进铺平道路。然而，现有的DDI预测模型在极端情况下的泛化、稳健特征提取以及现实应用可能性方面持续面临挑战。我们旨在通过利用上下文感知深度图学习的有效性，引入一种名为CADGL的新颖框架来应对这些挑战。基于定制的变分图自编码器（VGAE），我们利用两个上下文预处理器从两个不同视角：局部邻域和分子上下文，在异质图结构中提取特征，捕获关键的结构和生理化学信息。

    arXiv:2403.17210v1 Announce Type: cross  Abstract: Examining Drug-Drug Interactions (DDIs) is a pivotal element in the process of drug development. DDIs occur when one drug's properties are affected by the inclusion of other drugs. Detecting favorable DDIs has the potential to pave the way for creating and advancing innovative medications applicable in practical settings. However, existing DDI prediction models continue to face challenges related to generalization in extreme cases, robust feature extraction, and real-life application possibilities. We aim to address these challenges by leveraging the effectiveness of context-aware deep graph learning by introducing a novel framework named CADGL. Based on a customized variational graph autoencoder (VGAE), we capture critical structural and physio-chemical information using two context preprocessors for feature extraction from two different perspectives: local neighborhood and molecular context, in a heterogeneous graphical structure. Ou
    
[^2]: 关于神经网络中的最小深度

    On Minimal Depth in Neural Networks

    [https://arxiv.org/abs/2402.15315](https://arxiv.org/abs/2402.15315)

    本研究研究了神经网络中关于最小深度的问题，特别关注了ReLU神经网络的表达能力和最小深度与CPWL函数的关系。

    

    通过对ReLU神经网络表达能力以及与表示任何连续分段线性函数（CPWL）所需的最小深度相关的猜想的关系进行研究，本研究探讨了神经网络的表达能力特性。研究重点包括对求和和最大运算的最小深度表示，以及对多面体神经网络的探索。实验结果表明，对于求和运算，我们建立了关于操作数最小深度的充分条件以找到运算的最小深度。相反，关于最大运算，我们提供了全面的例子，证明仅依赖于操作数深度的充分条件，并不会暗示运算的最小深度。研究还考察了凸CPWL函数之间的最小深度关系。

    arXiv:2402.15315v1 Announce Type: new  Abstract: A characterization of the representability of neural networks is relevant to comprehend their success in artificial intelligence. This study investigate two topics on ReLU neural network expressivity and their connection with a conjecture related to the minimum depth required for representing any continuous piecewise linear function (CPWL). The topics are the minimal depth representation of the sum and max operations, as well as the exploration of polytope neural networks. For the sum operation, we establish a sufficient condition on the minimal depth of the operands to find the minimal depth of the operation. In contrast, regarding the max operation, a comprehensive set of examples is presented, demonstrating that no sufficient conditions, depending solely on the depth of the operands, would imply a minimal depth for the operation. The study also examine the minimal depth relationship between convex CPWL functions. On polytope neural ne
    
[^3]: 组合T-learning和DR-learning：一个用于高效估计因果对比的框架

    Combining T-learning and DR-learning: a framework for oracle-efficient estimation of causal contrasts

    [https://arxiv.org/abs/2402.01972](https://arxiv.org/abs/2402.01972)

    这篇论文介绍了高效插件学习的框架，能够有效估计异质因果对比，并解决了其他学习策略的一些缺点。该框架构建了人口风险函数的高效插件估计器，具有稳定性和鲁棒性。

    

    我们引入了高效插件（EP）学习，这是一种用于估计异质因果对比的新框架，例如条件平均处理效应和条件相对风险。 EP学习框架享有与Neyman正交学习策略（如DR-learning和R-learning）相同的oracle效率，同时解决了它们的一些主要缺点，包括（i）实际适用性可能受到损失函数非凸性的阻碍； （ii）它们可能因违反界限的倒数概率加权和伪结果而导致性能和稳定性差。为了避免这些缺点，EP学习者构建了因果对比的人口风险函数的高效插件估计器，从而继承了T-learning等插件估计策略的稳定性和鲁棒性特性。在合理条件下，基于经验风险最小化的EP学习者具有oracle效率，表现出渐近等价的性质。

    We introduce efficient plug-in (EP) learning, a novel framework for the estimation of heterogeneous causal contrasts, such as the conditional average treatment effect and conditional relative risk. The EP-learning framework enjoys the same oracle-efficiency as Neyman-orthogonal learning strategies, such as DR-learning and R-learning, while addressing some of their primary drawbacks, including that (i) their practical applicability can be hindered by loss function non-convexity; and (ii) they may suffer from poor performance and instability due to inverse probability weighting and pseudo-outcomes that violate bounds. To avoid these drawbacks, EP-learner constructs an efficient plug-in estimator of the population risk function for the causal contrast, thereby inheriting the stability and robustness properties of plug-in estimation strategies like T-learning. Under reasonable conditions, EP-learners based on empirical risk minimization are oracle-efficient, exhibiting asymptotic equivalen
    
[^4]: 重新审视随机梯度方法的最终迭代收敛性

    Revisiting the Last-Iterate Convergence of Stochastic Gradient Methods

    [https://arxiv.org/abs/2312.08531](https://arxiv.org/abs/2312.08531)

    研究了随机梯度方法的最终迭代收敛性，并提出了不需要限制性假设的最优收敛速率问题。

    

    在过去几年里，随机梯度下降（SGD）算法的最终迭代收敛引起了人们的兴趣，因为它在实践中表现良好但缺乏理论理解。对于Lipschitz凸函数，不同的研究建立了最佳的$O(\log(1/\delta)\log T/\sqrt{T})$或$O(\sqrt{\log(1/\delta)/T})$最终迭代的高概率收敛速率，其中$T$是时间跨度，$\delta$是失败概率。然而，为了证明这些界限，所有现有的工作要么局限于紧致域，要么需要几乎肯定有界的噪声。很自然地会问，不需要这两个限制性假设的情况下，SGD的最终迭代是否仍然可以保证最佳的收敛速率。除了这个重要问题外，还有很多理论问题仍然没有答案。

    arXiv:2312.08531v2 Announce Type: replace  Abstract: In the past several years, the last-iterate convergence of the Stochastic Gradient Descent (SGD) algorithm has triggered people's interest due to its good performance in practice but lack of theoretical understanding. For Lipschitz convex functions, different works have established the optimal $O(\log(1/\delta)\log T/\sqrt{T})$ or $O(\sqrt{\log(1/\delta)/T})$ high-probability convergence rates for the final iterate, where $T$ is the time horizon and $\delta$ is the failure probability. However, to prove these bounds, all the existing works are either limited to compact domains or require almost surely bounded noises. It is natural to ask whether the last iterate of SGD can still guarantee the optimal convergence rate but without these two restrictive assumptions. Besides this important question, there are still lots of theoretical problems lacking an answer. For example, compared with the last-iterate convergence of SGD for non-smoot
    
[^5]: 隐蔽而可量化：使用随机试验的混淆强度下界

    Hidden yet quantifiable: A lower bound for confounding strength using randomized trials

    [https://arxiv.org/abs/2312.03871](https://arxiv.org/abs/2312.03871)

    利用随机试验设计了一种统计检验，能够量化未观察到的混淆强度，并估计其下界，有效应用于现实世界中识别混淆。

    

    在快节奏精准医学时代，观察性研究在正确评估临床实践中新疗法方面发挥着重要作用。然而，未观察到的混淆可能严重损害从非随机数据中得出的因果结论。我们提出了一种利用随机试验来量化未观察到的混淆的新策略。首先，我们设计了一种统计检验来检测强度超过给定阈值的未观察到的混淆。然后，我们使用该检验来估计未观察到的混淆强度的渐近有效下界。我们在几个合成和半合成数据集上评估了我们的统计检验的功效和有效性。此外，我们展示了我们的下界如何能够在真实环境中正确识别未观察到的混淆的存在和不存在。

    arXiv:2312.03871v2 Announce Type: replace-cross  Abstract: In the era of fast-paced precision medicine, observational studies play a major role in properly evaluating new treatments in clinical practice. Yet, unobserved confounding can significantly compromise causal conclusions drawn from non-randomized data. We propose a novel strategy that leverages randomized trials to quantify unobserved confounding. First, we design a statistical test to detect unobserved confounding with strength above a given threshold. Then, we use the test to estimate an asymptotically valid lower bound on the unobserved confounding strength. We evaluate the power and validity of our statistical test on several synthetic and semi-synthetic datasets. Further, we show how our lower bound can correctly identify the absence and presence of unobserved confounding in a real-world setting.
    
[^6]: 逻辑回归和softmax分类器的逆向分类：高效优化

    Inverse classification with logistic and softmax classifiers: efficient optimization. (arXiv:2309.08945v1 [cs.LG] CROSS LISTED)

    [http://arxiv.org/abs/2309.08945](http://arxiv.org/abs/2309.08945)

    本文研究了逻辑回归和softmax分类器中的逆向分类问题，并提出了高效的解决方法，可以在交互式或实时应用中获得准确解。

    

    近年来，一种特定类型的问题引起了人们的兴趣，即在训练好的分类器上进行查询。具体而言，我们希望找到与给定输入实例最接近的实例，以使分类器的预测标签以所需的方式改变。这类问题包括反事实解释，对抗性示例和模型反演。所有这些问题实质上都是涉及输入实例向量上的固定分类器的优化问题，我们希望能够快速解决以用于交互式或实时应用。本文重点在于对逻辑回归和softmax分类器这两种广泛使用的分类器进行高效解决这一问题。由于这些模型的特殊性质，我们证明了对于逻辑回归问题，优化问题可以用闭式解求解，对于softmax分类器，可以通过迭代但非常快速地求解。这使我们能够精确地解决任一情况（接近机器精度）。

    In recent years, a certain type of problems have become of interest where one wants to query a trained classifier. Specifically, one wants to find the closest instance to a given input instance such that the classifier's predicted label is changed in a desired way. Examples of these ``inverse classification'' problems are counterfactual explanations, adversarial examples and model inversion. All of them are fundamentally optimization problems over the input instance vector involving a fixed classifier, and it is of interest to achieve a fast solution for interactive or real-time applications. We focus on solving this problem efficiently for two of the most widely used classifiers: logistic regression and softmax classifiers. Owing to special properties of these models, we show that the optimization can be solved in closed form for logistic regression, and iteratively but extremely fast for the softmax classifier. This allows us to solve either case exactly (to nearly machine precision)
    

