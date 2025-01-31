# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improving Model's Interpretability and Reliability using Biomarkers](https://arxiv.org/abs/2402.12394) | 利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。 |
| [^2] | [Computing the gradients with respect to all parameters of a quantum neural network using a single circuit.](http://arxiv.org/abs/2307.08167) | 该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。 |
| [^3] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^4] | [Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming.](http://arxiv.org/abs/2305.19706) | 本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。 |

# 详细

[^1]: 利用生物标志物提高模型的解释性和可靠性

    Improving Model's Interpretability and Reliability using Biomarkers

    [https://arxiv.org/abs/2402.12394](https://arxiv.org/abs/2402.12394)

    利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。

    

    准确且具有解释性的诊断模型在医学这个安全关键领域至关重要。我们研究了我们提出的基于生物标志物的肺部超声诊断流程的可解释性，以增强临床医生的诊断能力。本研究的目标是评估决策树分类器利用生物标志物提供的解释是否能够改善用户识别模型不准确预测能力，与传统的显著性图相比。我们的研究发现表明，基于临床建立的生物标志物的决策树解释能够帮助临床医生检测到假阳性，从而提高医学诊断模型的可靠性。

    arXiv:2402.12394v1 Announce Type: cross  Abstract: Accurate and interpretable diagnostic models are crucial in the safety-critical field of medicine. We investigate the interpretability of our proposed biomarker-based lung ultrasound diagnostic pipeline to enhance clinicians' diagnostic capabilities. The objective of this study is to assess whether explanations from a decision tree classifier, utilizing biomarkers, can improve users' ability to identify inaccurate model predictions compared to conventional saliency maps. Our findings demonstrate that decision tree explanations, based on clinically established biomarkers, can assist clinicians in detecting false positives, thus improving the reliability of diagnostic models in medicine.
    
[^2]: 使用单个电路计算量子神经网络所有参数的梯度

    Computing the gradients with respect to all parameters of a quantum neural network using a single circuit. (arXiv:2307.08167v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2307.08167](http://arxiv.org/abs/2307.08167)

    该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。

    

    在使用参数平移规则计算量子神经网络的梯度时，需要对网络的单个可调参数计算两次代价函数。当参数总数较高时，需要调整和运行多次用于计算的量子电路。在这里，我们提出了一种仅使用一个电路计算所有梯度的方法，它具有较低的电路深度和较少的经典寄存器。我们还在真实量子硬件和模拟器上进行了实验证明，我们的方法具有电路编译时间明显缩短的优势，从而加速了总体运行时间。

    When computing the gradients of a quantum neural network using the parameter-shift rule, the cost function needs to be calculated twice for the gradient with respect to a single adjustable parameter of the network. When the total number of parameters is high, the quantum circuit for the computation has to be adjusted and run for many times. Here we propose an approach to compute all the gradients using a single circuit only, with a much reduced circuit depth and less classical registers. We also demonstrate experimentally, on both real quantum hardware and simulator, that our approach has the advantages that the circuit takes a significantly shorter time to compile than the conventional approach, resulting in a speedup on the total runtime.
    
[^3]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^4]: 可分目标的最优决策树：推动动态规划的极限

    Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming. (arXiv:2305.19706v1 [cs.LG])

    [http://arxiv.org/abs/2305.19706](http://arxiv.org/abs/2305.19706)

    本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。

    

    决策树的全局优化在准确性，大小和人类可理解性方面表现出良好的前景。然而，许多方法仍然依赖于通用求解器，可扩展性仍然是一个问题。动态规划方法已被证明具有更好的可扩展性，因为它们通过将子树作为独立的子问题解决来利用树结构。然而，这仅适用于可以分别优化子树的任务。我们详细研究了这种关系，并展示了实现这种可分离约束和目标任意组合的动态规划方法。在四个应用领域的实验表明了这种方法的普适性，同时也比通用求解器具有更好的可扩展性。

    Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue. Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees. We explore this relationship in detail and show necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints. Experiments on four application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.
    

