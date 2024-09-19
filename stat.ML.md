# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient reductions between some statistical models](https://arxiv.org/abs/2402.07717) | 本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。 |
| [^2] | [Deep Neural Network Benchmarks for Selective Classification.](http://arxiv.org/abs/2401.12708) | 本论文研究了用于选择性分类的深度神经网络，目的是设计一种选择机制来平衡被拒绝的预测比例和所选预测的预测性能改进。 |
| [^3] | [Initial Guessing Bias: How Untrained Networks Favor Some Classes.](http://arxiv.org/abs/2306.00809) | 本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。 |
| [^4] | [It begins with a boundary: A geometric view on probabilistically robust learning.](http://arxiv.org/abs/2305.18779) | 本文探讨了深度神经网络对于对抗生成的示例缺乏鲁棒性的问题，并提出了一种从几何角度出发的新颖视角，介绍一族概率非局部周长函数来优化概率鲁棒学习（PRL）的原始表述，以提高其鲁棒性。 |
| [^5] | [Canonical foliations of neural networks: application to robustness.](http://arxiv.org/abs/2203.00922) | 本文探讨了利用黎曼几何和叶面理论创新应用于神经网络鲁棒性的新视角，提出了一种适用于数据空间的以曲率为考量因素的 two-step spectral 对抗攻击方法。 |

# 详细

[^1]: 一些统计模型之间的高效归约

    Efficient reductions between some statistical models

    [https://arxiv.org/abs/2402.07717](https://arxiv.org/abs/2402.07717)

    本研究提出了一种在不知道源统计模型参数的情况下，高效地将样本从源模型转换为目标模型的方法，并构造了几个归约方法。这些归约方法能适应不同的问题，例如专家混合模型、相位恢复和信号降噪等，并且可以处理缺失数据。此外，该研究还指出了一个潜在的应用，即将一个差分隐私机制转换为另一个机制。

    

    我们研究了在不知道源模型参数的情况下，近似地将来自源统计模型的样本转换为目标统计模型的样本的问题，并构造了几个计算上高效的这种统计实验之间的归约。具体而言，我们提供了计算上高效的程序，可以近似将均匀分布、Erlang分布和拉普拉斯分布的位置模型归约到一般的目标族。我们通过建立一些经典的高维问题之间的非渐近归约来说明我们的方法，包括专家混合模型、相位恢复和信号降噪等。值得注意的是，这些归约保持了结构，并可以适应缺失数据。我们还指出了将一个差分隐私机制转换为另一个机制的可能应用。

    We study the problem of approximately transforming a sample from a source statistical model to a sample from a target statistical model without knowing the parameters of the source model, and construct several computationally efficient such reductions between statistical experiments. In particular, we provide computationally efficient procedures that approximately reduce uniform, Erlang, and Laplace location models to general target families. We illustrate our methodology by establishing nonasymptotic reductions between some canonical high-dimensional problems, spanning mixtures of experts, phase retrieval, and signal denoising. Notably, the reductions are structure preserving and can accommodate missing data. We also point to a possible application in transforming one differentially private mechanism to another.
    
[^2]: 用于选择性分类的深度神经网络基准

    Deep Neural Network Benchmarks for Selective Classification. (arXiv:2401.12708v1 [cs.LG])

    [http://arxiv.org/abs/2401.12708](http://arxiv.org/abs/2401.12708)

    本论文研究了用于选择性分类的深度神经网络，目的是设计一种选择机制来平衡被拒绝的预测比例和所选预测的预测性能改进。

    

    随着机器学习模型在许多具有社会敏感性的任务中的部署增加，对可靠和可信预测的需求也日益增长。实现这些要求的一种方法是允许模型在存在高错误风险时放弃进行预测。这需要为模型添加选择机制，该机制选择模型将提供预测的例子。选择性分类框架旨在设计一个平衡被拒绝预测比例（即模型不进行预测的例子比例）与在所选预测上的预测性能改进之间的机制。存在多个选择性分类框架，其中大多数依赖于深度神经网络架构。然而，现有方法的实证评估仍局限于部分方法和设置之间的比较，给实践者提供了很少的见解。

    With the increasing deployment of machine learning models in many socially-sensitive tasks, there is a growing demand for reliable and trustworthy predictions. One way to accomplish these requirements is to allow a model to abstain from making a prediction when there is a high risk of making an error. This requires adding a selection mechanism to the model, which selects those examples for which the model will provide a prediction. The selective classification framework aims to design a mechanism that balances the fraction of rejected predictions (i.e., the proportion of examples for which the model does not make a prediction) versus the improvement in predictive performance on the selected predictions. Multiple selective classification frameworks exist, most of which rely on deep neural network architectures. However, the empirical evaluation of the existing approaches is still limited to partial comparisons among methods and settings, providing practitioners with little insight into 
    
[^3]: 初始猜测偏差：未经过训练的神经网络倾向于某些类别

    Initial Guessing Bias: How Untrained Networks Favor Some Classes. (arXiv:2306.00809v1 [cs.LG])

    [http://arxiv.org/abs/2306.00809](http://arxiv.org/abs/2306.00809)

    本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。

    

    神经网络的初始状态在调节后续的训练过程中扮演重要角色。在分类问题的背景下，我们提供了理论分析，证明神经网络的结构可以在训练之前，甚至在不存在显式偏差的情况下，使模型将所有预测都指向同一个类别。我们展示了这种现象的存在，称为“初始猜测偏差”（Initial Guessing Bias，IGB），这取决于架构选择，例如激活函数、最大池化层和网络深度。我们对IGB进行的分析具有实际意义，可以指导架构的选择和初始化。我们还强调理论后果，例如节点置换对称性的崩溃、自平均的破坏、某些均场近似的有效性以及深度带来的非平凡差异。

    The initial state of neural networks plays a central role in conditioning the subsequent training dynamics. In the context of classification problems, we provide a theoretical analysis demonstrating that the structure of a neural network can condition the model to assign all predictions to the same class, even before the beginning of training, and in the absence of explicit biases. We show that the presence of this phenomenon, which we call "Initial Guessing Bias" (IGB), depends on architectural choices such as activation functions, max-pooling layers, and network depth. Our analysis of IGB has practical consequences, in that it guides architecture selection and initialization. We also highlight theoretical consequences, such as the breakdown of node-permutation symmetry, the violation of self-averaging, the validity of some mean-field approximations, and the non-trivial differences arising with depth.
    
[^4]: 从几何角度看待概率鲁棒学习中的边界问题

    It begins with a boundary: A geometric view on probabilistically robust learning. (arXiv:2305.18779v1 [cs.LG])

    [http://arxiv.org/abs/2305.18779](http://arxiv.org/abs/2305.18779)

    本文探讨了深度神经网络对于对抗生成的示例缺乏鲁棒性的问题，并提出了一种从几何角度出发的新颖视角，介绍一族概率非局部周长函数来优化概率鲁棒学习（PRL）的原始表述，以提高其鲁棒性。

    

    尽管深度神经网络在许多分类任务上已经实现了超人类的表现，但它们往往对于对抗生成的示例缺乏鲁棒性，因此需要将经验风险最小化（ERM）重构为对抗性鲁棒的框架。最近，关注点已经转向了介于对抗性训练提供的鲁棒性和ERM提供的更高干净准确性和更快训练时间之间的方法。本文从几何角度出发，对一种这样的方法——概率鲁棒学习（PRL）（Robey等人，ICML，2022）进行了新颖的几何视角的探讨。我们提出了一个几何框架来理解PRL，这使我们能够确定其原始表述中的微妙缺陷，并介绍了一族概率非局部周长函数来解决这一问题。我们使用新颖的松弛方法证明了解的存在，并研究了引入的非局部周长函数的特性以及局部极限。

    Although deep neural networks have achieved super-human performance on many classification tasks, they often exhibit a worrying lack of robustness towards adversarially generated examples. Thus, considerable effort has been invested into reformulating Empirical Risk Minimization (ERM) into an adversarially robust framework. Recently, attention has shifted towards approaches which interpolate between the robustness offered by adversarial training and the higher clean accuracy and faster training times of ERM. In this paper, we take a fresh and geometric view on one such method -- Probabilistically Robust Learning (PRL) (Robey et al., ICML, 2022). We propose a geometric framework for understanding PRL, which allows us to identify a subtle flaw in its original formulation and to introduce a family of probabilistic nonlocal perimeter functionals to address this. We prove existence of solutions using novel relaxation methods and study properties as well as local limits of the introduced per
    
[^5]: 神经网络的规范叶面：鲁棒性应用研究

    Canonical foliations of neural networks: application to robustness. (arXiv:2203.00922v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2203.00922](http://arxiv.org/abs/2203.00922)

    本文探讨了利用黎曼几何和叶面理论创新应用于神经网络鲁棒性的新视角，提出了一种适用于数据空间的以曲率为考量因素的 two-step spectral 对抗攻击方法。

    

    深度学习模型易受到对抗攻击。而对抗学习正在变得至关重要。本文提出了一种新的神经网络鲁棒性视角，采用黎曼几何和叶面理论。通过创建考虑数据空间曲率的新对抗攻击，即 two-step spectral attack，来说明这个想法。数据空间被视为一个配备了神经网络的 Fisher 信息度量（FIM）拉回的（退化的）黎曼流形。大多数情况下，该度量仅为半正定，其内核成为研究的核心对象。从该核中导出一个规范叶面。横向叶的曲率给出了适当的修正，从而得到了两步近似的测地线和一种新的高效对抗攻击。该方法首先在一个 2D 玩具示例中进行演示。

    Deep learning models are known to be vulnerable to adversarial attacks. Adversarial learning is therefore becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory. The idea is illustrated by creating a new adversarial attack that takes into account the curvature of the data space. This new adversarial attack called the two-step spectral attack is a piece-wise linear approximation of a geodesic in the data space. The data space is treated as a (degenerate) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of transverse leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. The method is first illustrated on a 2D toy example
    

