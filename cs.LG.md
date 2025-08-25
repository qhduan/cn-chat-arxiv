# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Challenges and Opportunities in Generative AI](https://arxiv.org/abs/2403.00025) | 现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。 |
| [^2] | [A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection](https://arxiv.org/abs/2402.17018) | 通过在神经模型中引入不同iable和完全卷积的前端模型，并结合跳跃连接，成功实现对梯度攻击的显著韧性，并通过将模型组合成随机集合，有效对抗黑盒攻击。 |
| [^3] | [Explainable Bayesian Optimization.](http://arxiv.org/abs/2401.13334) | 本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。 |
| [^4] | [Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression.](http://arxiv.org/abs/2310.00369) | 本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。 |
| [^5] | [Asymmetric matrix sensing by gradient descent with small random initialization.](http://arxiv.org/abs/2309.01796) | 本论文研究了矩阵感知问题，通过小的随机初始化应用因式梯度下降算法来重建低秩矩阵。特别地，引入了一个连续微分方程，称为“扰动梯度流”，并证明了在扰动被限制在一定范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。 |
| [^6] | [Interpretable Anomaly Detection via Discrete Optimization.](http://arxiv.org/abs/2303.14111) | 该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。 |

# 详细

[^1]: 关于生成人工智能中的挑战与机遇

    On the Challenges and Opportunities in Generative AI

    [https://arxiv.org/abs/2403.00025](https://arxiv.org/abs/2403.00025)

    现代生成人工智能范例中存在关键的未解决挑战，如何解决这些挑战将进一步增强它们的能力、多功能性和可靠性，并为研究方向提供有价值的见解。

    

    深度生成建模领域近年来增长迅速而稳定。随着海量训练数据的可用性以及可扩展的无监督学习范式的进步，最近的大规模生成模型展现出合成高分辨率图像和文本以及结构化数据（如视频和分子）的巨大潜力。然而，我们认为当前大规模生成人工智能模型没有充分解决若干基本问题，限制了它们在各个领域的广泛应用。在本工作中，我们旨在确定现代生成人工智能范例中的关键未解决挑战，以进一步增强它们的能力、多功能性和可靠性。通过识别这些挑战，我们旨在为研究人员提供有价值的见解，探索有益的研究方向，从而促进更加强大和可访问的生成人工智能的发展。

    arXiv:2403.00025v1 Announce Type: cross  Abstract: The field of deep generative modeling has grown rapidly and consistently over the years. With the availability of massive amounts of training data coupled with advances in scalable unsupervised learning paradigms, recent large-scale generative models show tremendous promise in synthesizing high-resolution images and text, as well as structured data such as videos and molecules. However, we argue that current large-scale generative AI models do not sufficiently address several fundamental issues that hinder their widespread adoption across domains. In this work, we aim to identify key unresolved challenges in modern generative AI paradigms that should be tackled to further enhance their capabilities, versatility, and reliability. By identifying these challenges, we aim to provide researchers with valuable insights for exploring fruitful research directions, thereby fostering the development of more robust and accessible generative AI so
    
[^2]: 通过完全卷积和可微的前端与跳跃连接对梯度攻击表现出显著韧性的耐人寻味案例

    A Curious Case of Remarkable Resilience to Gradient Attacks via Fully Convolutional and Differentiable Front End with a Skip Connection

    [https://arxiv.org/abs/2402.17018](https://arxiv.org/abs/2402.17018)

    通过在神经模型中引入不同iable和完全卷积的前端模型，并结合跳跃连接，成功实现对梯度攻击的显著韧性，并通过将模型组合成随机集合，有效对抗黑盒攻击。

    

    我们测试了通过在一个冻结的分类器之前增加一个可微且完全卷积的模型，并具有跳跃连接的前端增强神经模型。通过使用较小的学习率进行大约一个epoch的训练，我们获得了一些模型，这些模型在保持骨干分类器准确性的同时，对包括AutoAttack软件包中的APGD和FAB-T攻击在内的梯度攻击具有异常的抵抗力，这归因于梯度掩盖。梯度掩盖现象并不新鲜，但对于这些没有梯度破坏部分（如JPEG压缩或预计导致梯度减小的部分）的完全可微模型来说，掩盖的程度相当显著。尽管黑盒攻击对梯度掩盖可能部分有效，但通过将模型组合成随机集合，可以轻松击败它们。我们估计这样的集合在CIFAR10和CIF等上实现了几乎SOTA级别的AutoAttack准确性。

    arXiv:2402.17018v1 Announce Type: cross  Abstract: We tested front-end enhanced neural models where a frozen classifier was prepended by a differentiable and fully convolutional model with a skip connection. By training them using a small learning rate for about one epoch, we obtained models that retained the accuracy of the backbone classifier while being unusually resistant to gradient attacks including APGD and FAB-T attacks from the AutoAttack package, which we attributed to gradient masking. The gradient masking phenomenon is not new, but the degree of masking was quite remarkable for fully differentiable models that did not have gradient-shattering components such as JPEG compression or components that are expected to cause diminishing gradients.   Though black box attacks can be partially effective against gradient masking, they are easily defeated by combining models into randomized ensembles. We estimate that such ensembles achieve near-SOTA AutoAttack accuracy on CIFAR10, CIF
    
[^3]: 可解释性贝叶斯优化

    Explainable Bayesian Optimization. (arXiv:2401.13334v1 [cs.LG])

    [http://arxiv.org/abs/2401.13334](http://arxiv.org/abs/2401.13334)

    本论文介绍了一种可解释性贝叶斯优化的方法，通过TNTRules生成高质量的解释，填补了贝叶斯优化和可解释人工智能之间的间隙。

    

    在工业领域，贝叶斯优化（BO）被广泛应用于人工智能协作参数调优的控制系统中。然而，由于近似误差和简化目标，BO的解决方案可能偏离人类专家的真实目标，需要后续调整。BO的黑盒特性限制了协作调优过程，因为专家不信任BO的建议。目前的可解释人工智能（XAI）方法不适用于优化问题，因此无法解决此间隙。为了填补这一间隙，我们提出了TNTRules（TUNE-NOTUNE规则），一种事后基于规则的可解释性方法，通过多目标优化生成高质量的解释。我们对基准优化问题和实际超参数优化任务的评估表明，TNTRules在生成高质量解释方面优于最先进的XAI方法。这项工作对BO和XAI的交叉领域做出了贡献，提供了可解释的优化方法。

    In industry, Bayesian optimization (BO) is widely applied in the human-AI collaborative parameter tuning of cyber-physical systems. However, BO's solutions may deviate from human experts' actual goal due to approximation errors and simplified objectives, requiring subsequent tuning. The black-box nature of BO limits the collaborative tuning process because the expert does not trust the BO recommendations. Current explainable AI (XAI) methods are not tailored for optimization and thus fall short of addressing this gap. To bridge this gap, we propose TNTRules (TUNE-NOTUNE Rules), a post-hoc, rule-based explainability method that produces high quality explanations through multiobjective optimization. Our evaluation of benchmark optimization problems and real-world hyperparameter optimization tasks demonstrates TNTRules' superiority over state-of-the-art XAI methods in generating high quality explanations. This work contributes to the intersection of BO and XAI, providing interpretable opt
    
[^4]: 提炼归纳偏差：超越模型压缩的知识蒸馏

    Distilling Inductive Bias: Knowledge Distillation Beyond Model Compression. (arXiv:2310.00369v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.00369](http://arxiv.org/abs/2310.00369)

    本文提出了一种超越模型压缩的知识蒸馏方法，通过从轻量级教师模型中提取归纳偏差，使Vision Transformers (ViTs) 的应用成为可能。这种方法包括使用一组不同架构的教师模型来指导学生Transformer，从而有效提高学生的性能。

    

    随着计算机视觉的快速发展，Vision Transformers (ViTs) 提供了在视觉和文本领域中实现统一信息处理的诱人前景。但是由于ViTs缺乏固有的归纳偏差，它们需要大量的训练数据。为了使它们的应用实际可行，我们引入了一种创新的基于集成的蒸馏方法，从轻量级的教师模型中提取归纳偏差。以前的系统仅依靠基于卷积的教学方法。然而，这种方法将一组具有不同架构倾向的轻量级教师模型（例如卷积和非线性卷积）同时用于指导学生Transformer。由于这些独特的归纳偏差，教师模型可以从各种存储数据集中获得广泛的知识，从而提高学生的性能。我们提出的框架还涉及预先计算和存储logits，从根本上实现了非归一化的状态匹配。

    With the rapid development of computer vision, Vision Transformers (ViTs) offer the tantalizing prospect of unified information processing across visual and textual domains. But due to the lack of inherent inductive biases in ViTs, they require enormous amount of data for training. To make their applications practical, we introduce an innovative ensemble-based distillation approach distilling inductive bias from complementary lightweight teacher models. Prior systems relied solely on convolution-based teaching. However, this method incorporates an ensemble of light teachers with different architectural tendencies, such as convolution and involution, to instruct the student transformer jointly. Because of these unique inductive biases, instructors can accumulate a wide range of knowledge, even from readily identifiable stored datasets, which leads to enhanced student performance. Our proposed framework also involves precomputing and storing logits in advance, essentially the unnormalize
    
[^5]: 用小的随机初始化梯度下降进行非对称矩阵感知

    Asymmetric matrix sensing by gradient descent with small random initialization. (arXiv:2309.01796v1 [cs.LG])

    [http://arxiv.org/abs/2309.01796](http://arxiv.org/abs/2309.01796)

    本论文研究了矩阵感知问题，通过小的随机初始化应用因式梯度下降算法来重建低秩矩阵。特别地，引入了一个连续微分方程，称为“扰动梯度流”，并证明了在扰动被限制在一定范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。

    

    我们研究了矩阵感知，即从少量线性测量中重建低秩矩阵的问题。它可以被形式化为一个过参数化回归问题，可以通过因式分解的梯度下降解决，当从一个小的随机初始化开始。线性神经网络，特别是通过因式梯度下降进行矩阵感知，作为现代机器学习中非凸问题的典型模型，可以将复杂现象解开并详细研究。许多研究致力于研究非对称矩阵感知的特殊情况，例如非对称矩阵因式分解和对称半正定矩阵感知。我们的关键贡献是引入了一个连续微分方程，我们称之为“扰动梯度流”。我们证明了当扰动被限制在足够范围内时，扰动梯度流能够快速收敛到真实的目标矩阵。梯度下降对矩阵的动态

    We study matrix sensing, which is the problem of reconstructing a low-rank matrix from a few linear measurements. It can be formulated as an overparameterized regression problem, which can be solved by factorized gradient descent when starting from a small random initialization.  Linear neural networks, and in particular matrix sensing by factorized gradient descent, serve as prototypical models of non-convex problems in modern machine learning, where complex phenomena can be disentangled and studied in detail. Much research has been devoted to studying special cases of asymmetric matrix sensing, such as asymmetric matrix factorization and symmetric positive semi-definite matrix sensing.  Our key contribution is introducing a continuous differential equation that we call the $\textit{perturbed gradient flow}$. We prove that the perturbed gradient flow converges quickly to the true target matrix whenever the perturbation is sufficiently bounded. The dynamics of gradient descent for matr
    
[^6]: 通过离散优化实现可解释性异常检测

    Interpretable Anomaly Detection via Discrete Optimization. (arXiv:2303.14111v1 [cs.LG])

    [http://arxiv.org/abs/2303.14111](http://arxiv.org/abs/2303.14111)

    该论文提出了一个通过学习有限自动机进行异常检测的框架，并通过约束优化算法和新的正则化方案提高了可解释性。

    

    异常检测在许多应用领域中都是必不可少的，例如网络安全、执法、医学和欺诈保护。然而，目前深度学习方法的决策过程往往难以理解，这通常限制了它们的实际应用性。为了克服这个限制，我们提出了一个学习框架，可以从序列数据中学习可解释性的异常检测器。具体来说，我们考虑从给定的未标记序列多重集中学习确定性有限自动机 （DFA）的任务。我们证明了这个问题是计算难题，并基于约束优化开发了两个学习算法。此外，我们为优化问题引入了新的正则化方案，以提高我们的DFA的整体可解释性。通过原型实现，我们证明我们的方法在准确性和F1分数方面表现出有望的结果。

    Anomaly detection is essential in many application domains, such as cyber security, law enforcement, medicine, and fraud protection. However, the decision-making of current deep learning approaches is notoriously hard to understand, which often limits their practical applicability. To overcome this limitation, we propose a framework for learning inherently interpretable anomaly detectors from sequential data. More specifically, we consider the task of learning a deterministic finite automaton (DFA) from a given multi-set of unlabeled sequences. We show that this problem is computationally hard and develop two learning algorithms based on constraint optimization. Moreover, we introduce novel regularization schemes for our optimization problems that improve the overall interpretability of our DFAs. Using a prototype implementation, we demonstrate that our approach shows promising results in terms of accuracy and F1 score.
    

