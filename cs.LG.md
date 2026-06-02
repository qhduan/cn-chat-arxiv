# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness](https://arxiv.org/abs/2404.01356) | 该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。 |
| [^2] | [AutoEval Done Right: Using Synthetic Data for Model Evaluation](https://arxiv.org/abs/2403.07008) | 提出了用合成数据进行模型评估的方法，通过高效和统计上合理的算法，在GPT-4实验中有效的人工标记样本大小增加了50%。 |
| [^3] | [Autoencoder with Ordered Variance for Nonlinear Model Identification](https://arxiv.org/abs/2402.14031) | 提出了一种具有有序方差的自编码器，通过添加方差正则化项来保持潜空间的顺序，并且在无监督设置中展示了其在提取非线性关系方面的有效性。 |
| [^4] | [Incentivized Collaboration in Active Learning.](http://arxiv.org/abs/2311.00260) | 该论文提出了一种激励合作的主动学习框架，旨在最小化标签复杂性，保证智能体无法通过个人行为减少其预期的标签复杂性，并提供了可实现且在标签复杂性方面与最佳可计算近似算法相媲美的合作协议。 |
| [^5] | [Multi-task learning of convex combinations of forecasting models.](http://arxiv.org/abs/2310.20545) | 本文提出了一种多任务学习方法，通过深度神经网络同时解决了预测模型选择和凸组合权重学习的问题。通过回归分支学习权重和分类分支选择具有多样性的预测方法，提高了基于特征的预测的精确度。 |
| [^6] | [Score Function Gradient Estimation to Widen the Applicability of Decision-Focused Learning.](http://arxiv.org/abs/2307.05213) | 本研究采用评分函数梯度估计方法，通过预测参数分布来计算决策焦点模型的更新，以扩大决策焦点学习的适用性。 |
| [^7] | [Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance.](http://arxiv.org/abs/2304.11127) | 该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。 |
| [^8] | [c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization.](http://arxiv.org/abs/2211.14411) | 本文提出了约束TPE（c-TPE）方法，是树形Parzen估计器（TPE）的扩展，可有效处理在性能要求之上施加的约束限制，实验证明在81个昂贵的HPO设置中表现出最佳性能排名。 |

# 详细

[^1]: 输入扰动对鲁棒准确公平性的双刃剑

    The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness

    [https://arxiv.org/abs/2404.01356](https://arxiv.org/abs/2404.01356)

    该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。

    

    深度神经网络(DNNs)被认为对敌对输入扰动敏感，导致预测的准确性或个体公平性降低。为了共同表征预测准确性和个体公平性对敌对扰动的敏感性，我们引入了一个名为鲁棒准确公平性的新定义。鲁棒准确公平性要求当实例及其相似对应物受到输入扰动时，预测与地面事实一致。我们提出一种敌对攻击方法RAFair，以暴露DNN中的虚假或偏见敌对缺陷，这些缺陷会欺骗准确性或损害个体公平性。然后，我们展示这样的敌对实例可以通过精心设计的良性扰动有效地解决，从而使它们的预测准确而公平。我们的工作探讨了输入对准确公平性的双刃剑。

    arXiv:2404.01356v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input 
    
[^2]: 自动评价正确: 使用合成数据进行模型评估

    AutoEval Done Right: Using Synthetic Data for Model Evaluation

    [https://arxiv.org/abs/2403.07008](https://arxiv.org/abs/2403.07008)

    提出了用合成数据进行模型评估的方法，通过高效和统计上合理的算法，在GPT-4实验中有效的人工标记样本大小增加了50%。

    

    机器学习模型的评估使用人工标记的验证数据可能既昂贵又耗时。可以使用AI标记的合成数据来减少此类目的人工注释数量，这一过程称为自动评估。我们提出了用于此目的的高效和统计上合理的算法，可以提高样本效率，同时保持不偏。这些算法在与GPT-4进行的实验中将有效的人工标记样本大小增加了高达50%。

    arXiv:2403.07008v1 Announce Type: cross  Abstract: The evaluation of machine learning models using human-labeled validation data can be expensive and time-consuming. AI-labeled synthetic data can be used to decrease the number of human annotations required for this purpose in a process called autoevaluation. We suggest efficient and statistically principled algorithms for this purpose that improve sample efficiency while remaining unbiased. These algorithms increase the effective human-labeled sample size by up to 50% on experiments with GPT-4.
    
[^3]: 具有有序方差的自编码器用于非线性模型识别

    Autoencoder with Ordered Variance for Nonlinear Model Identification

    [https://arxiv.org/abs/2402.14031](https://arxiv.org/abs/2402.14031)

    提出了一种具有有序方差的自编码器，通过添加方差正则化项来保持潜空间的顺序，并且在无监督设置中展示了其在提取非线性关系方面的有效性。

    

    本文提出了一种新颖的具有有序方差（AEO）的自编码器，其中通过修改损失函数，添加方差正则化项以强制在潜空间中保持顺序。此外，通过使用ResNets对自编码器进行修改，得到了一个ResNet AEO（RAEO）。该论文还展示了AEO和RAEO在无监督设置下提取输入变量之间的非线性关系的有效性。

    arXiv:2402.14031v1 Announce Type: cross  Abstract: This paper presents a novel autoencoder with ordered variance (AEO) in which the loss function is modified with a variance regularization term to enforce order in the latent space. Further, the autoencoder is modified using ResNets, which results in a ResNet AEO (RAEO). The paper also illustrates the effectiveness of AEO and RAEO in extracting nonlinear relationships among input variables in an unsupervised setting.
    
[^4]: 激励合作的主动学习

    Incentivized Collaboration in Active Learning. (arXiv:2311.00260v1 [cs.GT])

    [http://arxiv.org/abs/2311.00260](http://arxiv.org/abs/2311.00260)

    该论文提出了一种激励合作的主动学习框架，旨在最小化标签复杂性，保证智能体无法通过个人行为减少其预期的标签复杂性，并提供了可实现且在标签复杂性方面与最佳可计算近似算法相媲美的合作协议。

    

    在协作的主动学习中，多个智能体试图从共同的假设中学习标签，我们提出了一种创新的激励合作框架。在这里，理性的智能体旨在获得他们数据集的标签，同时尽量减少标签复杂性。我们着重设计（严格的）个体合理（IR）合作协议，确保智能体无法通过个人行为减少其预期的标签复杂性。我们首先证明，对于任何最优的主动学习算法，运行该算法直至整个数据集的合作协议已经是IR的。然而，计算最优算法是NP难的。因此，我们提供了可以实现（严格的）IR且在标签复杂性方面与最佳已知可计算近似算法相媲美的合作协议。

    In collaborative active learning, where multiple agents try to learn labels from a common hypothesis, we introduce an innovative framework for incentivized collaboration. Here, rational agents aim to obtain labels for their data sets while keeping label complexity at a minimum. We focus on designing (strict) individually rational (IR) collaboration protocols, ensuring that agents cannot reduce their expected label complexity by acting individually. We first show that given any optimal active learning algorithm, the collaboration protocol that runs the algorithm as is over the entire data is already IR. However, computing the optimal algorithm is NP-hard. We therefore provide collaboration protocols that achieve (strict) IR and are comparable with the best known tractable approximation algorithm in terms of label complexity.
    
[^5]: 多任务学习凸组合预测模型

    Multi-task learning of convex combinations of forecasting models. (arXiv:2310.20545v1 [cs.LG])

    [http://arxiv.org/abs/2310.20545](http://arxiv.org/abs/2310.20545)

    本文提出了一种多任务学习方法，通过深度神经网络同时解决了预测模型选择和凸组合权重学习的问题。通过回归分支学习权重和分类分支选择具有多样性的预测方法，提高了基于特征的预测的精确度。

    

    预测组合涉及使用多个预测来创建单一、更精确的预测。最近，基于特征的预测已被用于选择最合适的预测模型或学习它们的凸组合权重。在本文中，我们提出了一种同时解决这两个问题的多任务学习方法。该方法通过深度神经网络实现，其中包括两个分支：回归分支通过最小化组合预测误差来学习各种预测方法的权重，分类分支则重点选择多样性的预测方法。为了为分类任务生成训练标签，我们引入了一种优化驱动的方法，用于确定给定时间序列的最合适的方法。所提出的方法揭示了基于特征的预测中多样性的重要作用，并凸显了模型组合和选择之间的相互作用。

    Forecast combination involves using multiple forecasts to create a single, more accurate prediction. Recently, feature-based forecasting has been employed to either select the most appropriate forecasting models or to learn the weights of their convex combination. In this paper, we present a multi-task learning methodology that simultaneously addresses both problems. This approach is implemented through a deep neural network with two branches: the regression branch, which learns the weights of various forecasting methods by minimizing the error of combined forecasts, and the classification branch, which selects forecasting methods with an emphasis on their diversity. To generate training labels for the classification task, we introduce an optimization-driven approach that identifies the most appropriate methods for a given time series. The proposed approach elicits the essential role of diversity in feature-based forecasting and highlights the interplay between model combination and mo
    
[^6]: 评分函数梯度估计以扩大决策焦点学习的适用性

    Score Function Gradient Estimation to Widen the Applicability of Decision-Focused Learning. (arXiv:2307.05213v1 [cs.LG])

    [http://arxiv.org/abs/2307.05213](http://arxiv.org/abs/2307.05213)

    本研究采用评分函数梯度估计方法，通过预测参数分布来计算决策焦点模型的更新，以扩大决策焦点学习的适用性。

    

    许多现实世界的优化问题都包含需要在解决之前进行预测的未知参数。为了训练涉及的预测机器学习（ML）模型，通常采用的方法是专注于最大化预测准确性。然而，这种方法并不总是导致下游任务损失的最小化。决策焦点学习（DFL）是一种最近提出的范式，其目标是通过直接最小化任务损失来训练ML模型。然而，最先进的DFL方法受到它们对优化问题结构的假设（例如，问题是线性的）以及只能预测出现在目标函数中的参数的限制。在这项工作中，我们通过相反地预测参数的分布，并采用评分函数梯度估计（SFGE）来计算决策焦点模型的更新，从而扩大DFL的适用性。我们的实验...

    Many real-world optimization problems contain unknown parameters that must be predicted prior to solving. To train the predictive machine learning (ML) models involved, the commonly adopted approach focuses on maximizing predictive accuracy. However, this approach does not always lead to the minimization of the downstream task loss. Decision-focused learning (DFL) is a recently proposed paradigm whose goal is to train the ML model by directly minimizing the task loss. However, state-of-the-art DFL methods are limited by the assumptions they make about the structure of the optimization problem (e.g., that the problem is linear) and by the fact that can only predict parameters that appear in the objective function. In this work, we address these limitations by instead predicting \textit{distributions} over parameters and adopting score function gradient estimation (SFGE) to compute decision-focused updates to the predictive model, thereby widening the applicability of DFL. Our experiment
    
[^7]: 树状Parzen估计器：理解其算法组成部分及其在提高实证表现中的作用

    Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance. (arXiv:2304.11127v1 [cs.LG])

    [http://arxiv.org/abs/2304.11127](http://arxiv.org/abs/2304.11127)

    该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。

    

    许多领域中最近的进展要求更加复杂的实验设计。这种复杂的实验通常有许多参数，需要参数调整。Tree-structured Parzen estimator (TPE) 是一种贝叶斯优化方法，在最近的参数调整框架中被广泛使用。尽管它很受欢迎，但控制参数的角色和算法直觉尚未得到讨论。在本教程中，我们将确定每个控制参数的作用以及它们对超参数优化的影响，使用多种基准测试。我们将从剖析研究中得出的推荐设置与基准方法进行比较，并证明我们的推荐设置提高了TPE的性能。我们的TPE实现可在https://github.com/nabenabe0928/tpe/tree/single-opt中获得。

    Recent advances in many domains require more and more complicated experiment design. Such complicated experiments often have many parameters, which necessitate parameter tuning. Tree-structured Parzen estimator (TPE), a Bayesian optimization method, is widely used in recent parameter tuning frameworks. Despite its popularity, the roles of each control parameter and the algorithm intuition have not been discussed so far. In this tutorial, we will identify the roles of each control parameter and their impacts on hyperparameter optimization using a diverse set of benchmarks. We compare our recommended setting drawn from the ablation study with baseline methods and demonstrate that our recommended setting improves the performance of TPE. Our TPE implementation is available at https://github.com/nabenabe0928/tpe/tree/single-opt.
    
[^8]: c-TPE:基于树形结构的带不等式约束的帕捷斯特估计器用于昂贵超参数优化

    c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization. (arXiv:2211.14411v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.14411](http://arxiv.org/abs/2211.14411)

    本文提出了约束TPE（c-TPE）方法，是树形Parzen估计器（TPE）的扩展，可有效处理在性能要求之上施加的约束限制，实验证明在81个昂贵的HPO设置中表现出最佳性能排名。

    

    超参数优化（HPO）对于深度学习算法的强大性能至关重要，实际应用通常会在性能要求之上施加一些限制，例如内存使用或延迟等。在本文中，我们提出了约束TPE（c-TPE），这是广泛使用的多功能贝叶斯优化方法——树形Parzen估计器（TPE）的扩展，以处理这些约束。我们提出的扩展不仅是简单地将现有收益函数和原始TPE组合起来，而是包括修改来解决导致性能不佳的问题。我们从经验和理论上深入分析这些修改，提供了有关它们如何有效地克服这些挑战的见解。在实验中，我们证明了c-TPE在81个昂贵的HPO设置中表现出最佳的平均排名性能，具有统计显着性。

    Hyperparameter optimization (HPO) is crucial for strong performance of deep learning algorithms and real-world applications often impose some constraints, such as memory usage, or latency on top of the performance requirement. In this work, we propose constrained TPE (c-TPE), an extension of the widely-used versatile Bayesian optimization method, tree-structured Parzen estimator (TPE), to handle these constraints. Our proposed extension goes beyond a simple combination of an existing acquisition function and the original TPE, and instead includes modifications that address issues that cause poor performance. We thoroughly analyze these modifications both empirically and theoretically, providing insights into how they effectively overcome these challenges. In the experiments, we demonstrate that c-TPE exhibits the best average rank performance among existing methods with statistical significance on 81 expensive HPO settings.
    

