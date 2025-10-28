# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [kNN Algorithm for Conditional Mean and Variance Estimation with Automated Uncertainty Quantification and Variable Selection](https://rss.arxiv.org/abs/2402.01635) | 本文介绍了一种利用kNN算法进行条件均值和方差估计的方法，该方法采用了自动不确定性量化和变量选择技术，提高了估计的准确性和性能。 |
| [^2] | [Training-time Neuron Alignment through Permutation Subspace for Improving Linear Mode Connectivity and Model Fusion](https://rss.arxiv.org/abs/2402.01342) | 本文提出了一个在训练过程中进行神经元对齐的方法，通过置换子空间减少了线性模块连通性的局限性，为模型融合算法的改进提供了可能性。 |
| [^3] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^4] | [Optimal and Fair Encouragement Policy Evaluation and Learning.](http://arxiv.org/abs/2309.07176) | 本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。 |
| [^5] | [Representer Theorems for Metric and Preference Learning: A Geometric Perspective.](http://arxiv.org/abs/2304.03720) | 该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。 |

# 详细

[^1]: kNN算法用于条件均值和方差估计，具有自动不确定性量化和变量选择

    kNN Algorithm for Conditional Mean and Variance Estimation with Automated Uncertainty Quantification and Variable Selection

    [https://rss.arxiv.org/abs/2402.01635](https://rss.arxiv.org/abs/2402.01635)

    本文介绍了一种利用kNN算法进行条件均值和方差估计的方法，该方法采用了自动不确定性量化和变量选择技术，提高了估计的准确性和性能。

    

    本文介绍了一种基于kNN的回归方法，将传统的非参数kNN模型的可扩展性和适应性与一种新的变量选择技术相结合。该方法主要目标是准确估计随机响应变量的条件均值和方差，从而有效地描述各种情景下的条件分布。我们的方法包含了一个健壮的不确定性量化机制，利用我们之前关于条件均值和方差的估计工作。 kNN的应用确保了在预测区间时可扩展的计算效率和与最优非参数速率相一致的统计准确性。此外，我们引入了一种新的kNN半参数算法来估计考虑协变量的ROC曲线。对于选择平滑参数k，我们提出了一个具有理论保证的算法。变量选择的引入显著提高了该方法相对于传统方法的性能。

    In this paper, we introduce a kNN-based regression method that synergizes the scalability and adaptability of traditional non-parametric kNN models with a novel variable selection technique. This method focuses on accurately estimating the conditional mean and variance of random response variables, thereby effectively characterizing conditional distributions across diverse scenarios.Our approach incorporates a robust uncertainty quantification mechanism, leveraging our prior estimation work on conditional mean and variance. The employment of kNN ensures scalable computational efficiency in predicting intervals and statistical accuracy in line with optimal non-parametric rates. Additionally, we introduce a new kNN semi-parametric algorithm for estimating ROC curves, accounting for covariates. For selecting the smoothing parameter k, we propose an algorithm with theoretical guarantees.Incorporation of variable selection enhances the performance of the method significantly over convention
    
[^2]: 通过置换子空间在训练过程中对神经元进行对齐，以改进线性模块连通性和模型融合

    Training-time Neuron Alignment through Permutation Subspace for Improving Linear Mode Connectivity and Model Fusion

    [https://rss.arxiv.org/abs/2402.01342](https://rss.arxiv.org/abs/2402.01342)

    本文提出了一个在训练过程中进行神经元对齐的方法，通过置换子空间减少了线性模块连通性的局限性，为模型融合算法的改进提供了可能性。

    

    在深度学习中，即使在相同初始化条件下，随机梯度下降算法经常产生具有功能相似但在权重空间中分散的解，这导致了线性模块连通性（LMC）的局限性。克服这些局限性对于理解深度学习动态和提高模型融合算法至关重要。以前的研究强调置换对称性在通过网络置换减少训练后的局限性方面的作用。然而，这些事后的方法需要额外的计算，在更大、更复杂的模型（如ViT，LLM）上效果较差，因为存在大量的置换矩阵。因此，在本文中，我们研究了训练过程中神经元的对齐。我们的假设是，在训练过程中的置换子空间可以免费减少LMC的局限性。我们发现，初始化时进行修剪可以支持这一假设。除了修剪之外，我们引入了TNA-PFN，一种简单而无损的算法，在训练过程中使用部分梯度掩码。TNA-PFN在理论上和实验上都得到了支持。

    In deep learning, stochastic gradient descent often yields functionally similar yet widely scattered solutions in the weight space even under the same initialization, causing barriers in the Linear Mode Connectivity (LMC) landscape. Overcoming these barriers is crucial for understanding deep learning dynamics and enhancing model-fusion algorithms. Previous studies highlight the role of permutation symmetry in reducing post-training barriers through network permutation. However, these post-hoc methods, demanding extra computations, are less effective for larger, complex models (e.g., ViT, LLM) due to numerous permutation matrices. Thus, in this paper, we study training-time neuron alignment. Our hypothesis suggests that training-time permutation subspace can reduce LMC barriers for free. We find that pruning at initialization supports this. Beyond pruning, we introduce TNA-PFN, a simple yet lossless algorithm using a partial gradient mask during training. TNA-PFN is theoretically and em
    
[^3]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^4]: 最优和公平的鼓励政策评估与学习

    Optimal and Fair Encouragement Policy Evaluation and Learning. (arXiv:2309.07176v1 [cs.LG])

    [http://arxiv.org/abs/2309.07176](http://arxiv.org/abs/2309.07176)

    本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。

    

    在关键领域中，强制个体接受治疗通常是不可能的，因此在人类不遵循治疗建议的情况下，最优策略规则只是建议。在这些领域中，接受治疗的个体可能存在异质性，治疗效果也可能存在异质性。虽然最优治疗规则可以最大化整个人群的因果结果，但在鼓励的情况下，对于访问平等限制或其他公平考虑因素可能是相关的。例如，在社会服务领域，一个持久的难题是那些最有可能从中受益的人中那些获益服务的使用差距。当决策者对访问和平均结果都有分配偏好时，最优决策规则会发生变化。我们研究了因果识别、统计方差减少估计和稳健估计的最优治疗规则，包括在违反阳性条件的情况下。

    In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. While optimal treatment rules can maximize causal outcomes across the population, access parity constraints or other fairness considerations can be relevant in the case of encouragement. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study causal identification, statistical variance-reduced estimation, and robust estimation of optimal treatment rules, including under potential violations of positivity. We c
    
[^5]: 度量学习与偏好学习的表现定理：基于几何的视角

    Representer Theorems for Metric and Preference Learning: A Geometric Perspective. (arXiv:2304.03720v1 [cs.LG])

    [http://arxiv.org/abs/2304.03720](http://arxiv.org/abs/2304.03720)

    该论文提出了度量学习和偏好学习的新的表现定理，解决了度量学习任务以三元组比较为基础的表现定理问题。这种表现定理可以用内积诱导的范数来表示。

    

    我们探讨了希尔伯特空间中的度量学习和偏好学习问题，并获得了一种新的度量学习和偏好学习的表现定理。我们的关键观察是，表现定理可以根据问题结构内在的内积所诱导的范数来表示。此外，我们展示了如何将我们的框架应用于三元组比较的度量学习任务，并展示它导致了一个简单且自包含的该任务的表现定理。在再生核希尔伯特空间(RKHS)的情况下，我们展示了学习问题的解可以使用类似于经典表现定理的核术语表示。

    We explore the metric and preference learning problem in Hilbert spaces. We obtain a novel representer theorem for the simultaneous task of metric and preference learning. Our key observation is that the representer theorem can be formulated with respect to the norm induced by the inner product inherent in the problem structure. Additionally, we demonstrate how our framework can be applied to the task of metric learning from triplet comparisons and show that it leads to a simple and self-contained representer theorem for this task. In the case of Reproducing Kernel Hilbert Spaces (RKHS), we demonstrate that the solution to the learning problem can be expressed using kernel terms, akin to classical representer theorems.
    

