# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness](https://arxiv.org/abs/2404.01356) | 该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。 |
| [^2] | [AutoEval Done Right: Using Synthetic Data for Model Evaluation](https://arxiv.org/abs/2403.07008) | 提出了用合成数据进行模型评估的方法，通过高效和统计上合理的算法，在GPT-4实验中有效的人工标记样本大小增加了50%。 |
| [^3] | [Recent Advances in Multi-modal 3D Scene Understanding: A Comprehensive Survey and Evaluation.](http://arxiv.org/abs/2310.15676) | 多模态3D场景理解在自动驾驶和人机交互等领域中应用广泛。本文从理论的角度对多模态3D方法的进展进行了全面调查和评估。 |
| [^4] | [DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle.](http://arxiv.org/abs/2307.06647) | DeepIPCv2是一种利用LiDAR传感器感知环境的自动驾驶模型，通过使用点云作为感知输入，在各种条件下实现了更强大的驾驶性能。 |
| [^5] | [Score Function Gradient Estimation to Widen the Applicability of Decision-Focused Learning.](http://arxiv.org/abs/2307.05213) | 本研究采用评分函数梯度估计方法，通过预测参数分布来计算决策焦点模型的更新，以扩大决策焦点学习的适用性。 |
| [^6] | [Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance.](http://arxiv.org/abs/2304.11127) | 该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。 |
| [^7] | [c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization.](http://arxiv.org/abs/2211.14411) | 本文提出了约束TPE（c-TPE）方法，是树形Parzen估计器（TPE）的扩展，可有效处理在性能要求之上施加的约束限制，实验证明在81个昂贵的HPO设置中表现出最佳性能排名。 |

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
    
[^3]: 多模态3D场景理解的最新进展：一项全面调查和评估

    Recent Advances in Multi-modal 3D Scene Understanding: A Comprehensive Survey and Evaluation. (arXiv:2310.15676v1 [cs.CV])

    [http://arxiv.org/abs/2310.15676](http://arxiv.org/abs/2310.15676)

    多模态3D场景理解在自动驾驶和人机交互等领域中应用广泛。本文从理论的角度对多模态3D方法的进展进行了全面调查和评估。

    

    多模态3D场景理解因其在自动驾驶和人机交互等许多领域中的广泛应用而受到了广泛关注。与传统的单模态3D理解相比，引入额外的模态不仅提升了场景解释的丰富性和准确性，还确保了更稳健和弹性的理解。这在多样化和具有挑战性的环境中尤为关键，仅依靠3D数据可能是不足够的。尽管在过去三年中出现了许多多模态3D方法的发展，特别是那些整合多摄像头图像（3D+2D）和文本描述（3D+语言）的方法，但缺乏全面深入的评估。在本文中，我们对最近的进展进行了系统的调查，以填补这一空白。我们首先简要介绍了一个背景，正式定义了各种3D多模态任务，并总结了它们固有的挑战。

    Multi-modal 3D scene understanding has gained considerable attention due to its wide applications in many areas, such as autonomous driving and human-computer interaction. Compared to conventional single-modal 3D understanding, introducing an additional modality not only elevates the richness and precision of scene interpretation but also ensures a more robust and resilient understanding. This becomes especially crucial in varied and challenging environments where solely relying on 3D data might be inadequate. While there has been a surge in the development of multi-modal 3D methods over past three years, especially those integrating multi-camera images (3D+2D) and textual descriptions (3D+language), a comprehensive and in-depth review is notably absent. In this article, we present a systematic survey of recent progress to bridge this gap. We begin by briefly introducing a background that formally defines various 3D multi-modal tasks and summarizes their inherent challenges. After that
    
[^4]: DeepIPCv2：利用LiDAR强化自动驾驶环境感知与导航控制

    DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle. (arXiv:2307.06647v1 [cs.RO])

    [http://arxiv.org/abs/2307.06647](http://arxiv.org/abs/2307.06647)

    DeepIPCv2是一种利用LiDAR传感器感知环境的自动驾驶模型，通过使用点云作为感知输入，在各种条件下实现了更强大的驾驶性能。

    

    我们提出了DeepIPCv2，一种利用LiDAR传感器感知环境的自动驾驶模型，以实现更强大的驾驶性能，特别是在光照条件较差的情况下。DeepIPCv2使用一组LiDAR点云作为其主要感知输入。由于点云不受光照变化的影响，它们可以提供清晰的环境观察，无论条件如何。这使得感知模块能够提供更好的场景理解和稳定的特征，从而支持控制模块准确估计导航控制。为了评估其性能，我们通过部署该模型来预测一组驾驶记录并在三种不同条件下进行真实自动驾驶的测试。我们还进行了消融和比较研究，以证明其性能。基于实验结果，DeepIPCv2在所有条件下均显示出强大的驾驶性能。

    We present DeepIPCv2, an autonomous driving model that perceives the environment using a LiDAR sensor for more robust drivability, especially when driving under poor illumination conditions. DeepIPCv2 takes a set of LiDAR point clouds for its main perception input. As point clouds are not affected by illumination changes, they can provide a clear observation of the surroundings no matter what the condition is. This results in a better scene understanding and stable features provided by the perception module to support the controller module in estimating navigational control properly. To evaluate its performance, we conduct several tests by deploying the model to predict a set of driving records and perform real automated driving under three different conditions. We also conduct ablation and comparative studies with some recent models to justify its performance. Based on the experimental results, DeepIPCv2 shows a robust performance by achieving the best drivability in all conditions. C
    
[^5]: 评分函数梯度估计以扩大决策焦点学习的适用性

    Score Function Gradient Estimation to Widen the Applicability of Decision-Focused Learning. (arXiv:2307.05213v1 [cs.LG])

    [http://arxiv.org/abs/2307.05213](http://arxiv.org/abs/2307.05213)

    本研究采用评分函数梯度估计方法，通过预测参数分布来计算决策焦点模型的更新，以扩大决策焦点学习的适用性。

    

    许多现实世界的优化问题都包含需要在解决之前进行预测的未知参数。为了训练涉及的预测机器学习（ML）模型，通常采用的方法是专注于最大化预测准确性。然而，这种方法并不总是导致下游任务损失的最小化。决策焦点学习（DFL）是一种最近提出的范式，其目标是通过直接最小化任务损失来训练ML模型。然而，最先进的DFL方法受到它们对优化问题结构的假设（例如，问题是线性的）以及只能预测出现在目标函数中的参数的限制。在这项工作中，我们通过相反地预测参数的分布，并采用评分函数梯度估计（SFGE）来计算决策焦点模型的更新，从而扩大DFL的适用性。我们的实验...

    Many real-world optimization problems contain unknown parameters that must be predicted prior to solving. To train the predictive machine learning (ML) models involved, the commonly adopted approach focuses on maximizing predictive accuracy. However, this approach does not always lead to the minimization of the downstream task loss. Decision-focused learning (DFL) is a recently proposed paradigm whose goal is to train the ML model by directly minimizing the task loss. However, state-of-the-art DFL methods are limited by the assumptions they make about the structure of the optimization problem (e.g., that the problem is linear) and by the fact that can only predict parameters that appear in the objective function. In this work, we address these limitations by instead predicting \textit{distributions} over parameters and adopting score function gradient estimation (SFGE) to compute decision-focused updates to the predictive model, thereby widening the applicability of DFL. Our experiment
    
[^6]: 树状Parzen估计器：理解其算法组成部分及其在提高实证表现中的作用

    Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance. (arXiv:2304.11127v1 [cs.LG])

    [http://arxiv.org/abs/2304.11127](http://arxiv.org/abs/2304.11127)

    该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。

    

    许多领域中最近的进展要求更加复杂的实验设计。这种复杂的实验通常有许多参数，需要参数调整。Tree-structured Parzen estimator (TPE) 是一种贝叶斯优化方法，在最近的参数调整框架中被广泛使用。尽管它很受欢迎，但控制参数的角色和算法直觉尚未得到讨论。在本教程中，我们将确定每个控制参数的作用以及它们对超参数优化的影响，使用多种基准测试。我们将从剖析研究中得出的推荐设置与基准方法进行比较，并证明我们的推荐设置提高了TPE的性能。我们的TPE实现可在https://github.com/nabenabe0928/tpe/tree/single-opt中获得。

    Recent advances in many domains require more and more complicated experiment design. Such complicated experiments often have many parameters, which necessitate parameter tuning. Tree-structured Parzen estimator (TPE), a Bayesian optimization method, is widely used in recent parameter tuning frameworks. Despite its popularity, the roles of each control parameter and the algorithm intuition have not been discussed so far. In this tutorial, we will identify the roles of each control parameter and their impacts on hyperparameter optimization using a diverse set of benchmarks. We compare our recommended setting drawn from the ablation study with baseline methods and demonstrate that our recommended setting improves the performance of TPE. Our TPE implementation is available at https://github.com/nabenabe0928/tpe/tree/single-opt.
    
[^7]: c-TPE:基于树形结构的带不等式约束的帕捷斯特估计器用于昂贵超参数优化

    c-TPE: Tree-structured Parzen Estimator with Inequality Constraints for Expensive Hyperparameter Optimization. (arXiv:2211.14411v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.14411](http://arxiv.org/abs/2211.14411)

    本文提出了约束TPE（c-TPE）方法，是树形Parzen估计器（TPE）的扩展，可有效处理在性能要求之上施加的约束限制，实验证明在81个昂贵的HPO设置中表现出最佳性能排名。

    

    超参数优化（HPO）对于深度学习算法的强大性能至关重要，实际应用通常会在性能要求之上施加一些限制，例如内存使用或延迟等。在本文中，我们提出了约束TPE（c-TPE），这是广泛使用的多功能贝叶斯优化方法——树形Parzen估计器（TPE）的扩展，以处理这些约束。我们提出的扩展不仅是简单地将现有收益函数和原始TPE组合起来，而是包括修改来解决导致性能不佳的问题。我们从经验和理论上深入分析这些修改，提供了有关它们如何有效地克服这些挑战的见解。在实验中，我们证明了c-TPE在81个昂贵的HPO设置中表现出最佳的平均排名性能，具有统计显着性。

    Hyperparameter optimization (HPO) is crucial for strong performance of deep learning algorithms and real-world applications often impose some constraints, such as memory usage, or latency on top of the performance requirement. In this work, we propose constrained TPE (c-TPE), an extension of the widely-used versatile Bayesian optimization method, tree-structured Parzen estimator (TPE), to handle these constraints. Our proposed extension goes beyond a simple combination of an existing acquisition function and the original TPE, and instead includes modifications that address issues that cause poor performance. We thoroughly analyze these modifications both empirically and theoretically, providing insights into how they effectively overcome these challenges. In the experiments, we demonstrate that c-TPE exhibits the best average rank performance among existing methods with statistical significance on 81 expensive HPO settings.
    

