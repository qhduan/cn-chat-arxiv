# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling.](http://arxiv.org/abs/2310.06389) | 本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。 |
| [^2] | [SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.](http://arxiv.org/abs/2310.03684) | SmoothLLM是第一个用于减轻大型语言模型上越狱攻击的算法，通过在输入提示上随机扰动并汇总预测结果来检测对抗性输入，将攻击成功率降低至不到一个百分点，并提供了可证明的保证。 |
| [^3] | [CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra.](http://arxiv.org/abs/2309.03060) | CoLA是一个用于机器学习中大规模线性代数问题的简单但通用的框架，通过组合调度规则和线性操作符抽象，自动构建了内存和运行时高效的数值算法，提供了内存高效的自动微分、低精度计算和GPU加速，同时可以适应下游软件包中的新对象、操作和规则。 |
| [^4] | [Corruption-Robust Lipschitz Contextual Search.](http://arxiv.org/abs/2307.13903) | 该论文研究了学习具有被篡改的二进制信号的Lipschitz函数的问题，提出了一种腐败鲁棒算法。该算法在不同损失函数下实现了不同程度的后悔。 |
| [^5] | [Learning sources of variability from high-dimensional observational studies.](http://arxiv.org/abs/2307.13868) | 本研究提出了一种针对高维观测研究的方法，将因果估计泛化到任意维度或可测空间的结果，并提出了一种用于名义变量的因果偏差测试。实验证明该方法相比现有策略在有限样本有效性和功率方面有改进。 |
| [^6] | [Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms.](http://arxiv.org/abs/2305.18409) | 本文提出了一种新的面向方向的多目标问题，并给出了两种随机算法以解决这个问题，理论上收敛到帕累托稳定点。 |
| [^7] | [LegendreTron: Uprising Proper Multiclass Loss Learning.](http://arxiv.org/abs/2301.11695) | 本文提出了一种新颖和实用的方法{\sc LegendreTron}，用于联合学习多类别问题的正确标准损失和概率。这种方法在基准测试中经常优于其他方法。 |
| [^8] | [Beyond Invariance: Test-Time Label-Shift Adaptation for Distributions with "Spurious" Correlations.](http://arxiv.org/abs/2211.15646) | 本文提出了一种测试时标签转移校正方法，通过适应分布的变化来提升预测模型性能，该方法可以处理类别标签和噪声因素的依赖关系随域变化的问题。 |
| [^9] | [Rigorous dynamical mean field theory for stochastic gradient descent methods.](http://arxiv.org/abs/2210.06591) | 本研究通过证明的闭式方程，描述了一类基于梯度的方法在高维情况下的精确渐进性能，为随机梯度下降等算法提供了理论支持，并提供了数值实现。 |

# 详细

[^1]: 学习可堆叠和可跳过的乐高积木以实现高效、可重构和可变分辨率的扩散建模

    Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling. (arXiv:2310.06389v1 [cs.CV])

    [http://arxiv.org/abs/2310.06389](http://arxiv.org/abs/2310.06389)

    本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。

    

    扩散模型在生成真实感图像方面表现出色，但在训练和采样方面具有显著的计算成本。尽管有各种技术来解决这些计算挑战，但一个较少探索的问题是设计一个高效且适应性强的网络骨干，用于迭代细化。当前的选项如U-Net和Vision Transformer通常依赖于资源密集型的深度网络，缺乏在变量分辨率下生成图像或使用比训练中更小的网络所需的灵活性。本研究引入了乐高积木，它们无缝集成了局部特征丰富和全局内容协调。这些积木可以堆叠在一起，创建一个测试时可重构的扩散骨干，允许选择性跳过积木以减少采样成本，并生成比训练数据更高分辨率的图像。乐高积木通过MLP对局部区域进行丰富，并使用Transformer块进行变换，同时保持一致的全分辨率

    Diffusion models excel at generating photo-realistic images but come with significant computational costs in both training and sampling. While various techniques address these computational challenges, a less-explored issue is designing an efficient and adaptable network backbone for iterative refinement. Current options like U-Net and Vision Transformer often rely on resource-intensive deep networks and lack the flexibility needed for generating images at variable resolutions or with a smaller network than used in training. This study introduces LEGO bricks, which seamlessly integrate Local-feature Enrichment and Global-content Orchestration. These bricks can be stacked to create a test-time reconfigurable diffusion backbone, allowing selective skipping of bricks to reduce sampling costs and generate higher-resolution images than the training data. LEGO bricks enrich local regions with an MLP and transform them using a Transformer block while maintaining a consistent full-resolution i
    
[^2]: SmoothLLM：防御大型语言模型免受越狱攻击

    SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. (arXiv:2310.03684v1 [cs.LG])

    [http://arxiv.org/abs/2310.03684](http://arxiv.org/abs/2310.03684)

    SmoothLLM是第一个用于减轻大型语言模型上越狱攻击的算法，通过在输入提示上随机扰动并汇总预测结果来检测对抗性输入，将攻击成功率降低至不到一个百分点，并提供了可证明的保证。

    

    尽管努力将大型语言模型（LLM）与人类价值观保持一致，但广泛使用的LLM（如GPT、Llama、Claude和PaLM）仍然容易受到越狱攻击，即对目标LLM进行欺骗，以生成不合适的内容。为了解决这个漏洞，我们提出了SmoothLLM，这是第一个旨在减轻LLM上的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的改变很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后汇总相应的预测结果来检测对抗性输入。SmoothLLM将众多热门LLM的攻击成功率降低至不到一个百分点，避免了不必要的保守性，并对攻击缓解提供了可证明的保证。此外，我们的防御使用的查询数量比现有的攻击方法少得多，并且与任何LLM兼容。

    Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.
    
[^3]: CoLA: 深入利用组合结构实现自动和高效的数值线性代数

    CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra. (arXiv:2309.03060v1 [cs.LG])

    [http://arxiv.org/abs/2309.03060](http://arxiv.org/abs/2309.03060)

    CoLA是一个用于机器学习中大规模线性代数问题的简单但通用的框架，通过组合调度规则和线性操作符抽象，自动构建了内存和运行时高效的数值算法，提供了内存高效的自动微分、低精度计算和GPU加速，同时可以适应下游软件包中的新对象、操作和规则。

    

    许多机器学习和科学领域涉及到大规模的线性代数问题，如特征分解、解线性系统、计算矩阵指数和迹估计等。涉及的矩阵通常具有Krondor、卷积、块对角、求和或乘积等结构。在本文中，我们提出了一个简单但通用的机器学习大规模线性代数问题的框架，名为CoLA（组合线性代数）。通过将线性操作符抽象与组合调度规则相结合，CoLA能够自动构建内存和运行时高效的数值算法。此外，CoLA还提供内存高效的自动微分、低精度计算和JAX和PyTorch中的GPU加速，同时还能够通过多重调度适应下游软件包中的新对象、操作和规则。CoLA可以加速许多代数操作，同时也便于原型化矩阵结构和算法，提供了可行性的降低-

    Many areas of machine learning and science involve large linear algebra problems, such as eigendecompositions, solving linear systems, computing matrix exponentials, and trace estimation. The matrices involved often have Kronecker, convolutional, block diagonal, sum, or product structure. In this paper, we propose a simple but general framework for large-scale linear algebra problems in machine learning, named CoLA (Compositional Linear Algebra). By combining a linear operator abstraction with compositional dispatch rules, CoLA automatically constructs memory and runtime efficient numerical algorithms. Moreover, CoLA provides memory efficient automatic differentiation, low precision computation, and GPU acceleration in both JAX and PyTorch, while also accommodating new objects, operations, and rules in downstream packages via multiple dispatch. CoLA can accelerate many algebraic operations, while making it easy to prototype matrix structures and algorithms, providing an appealing drop-
    
[^4]: 腐败鲁棒的Lipschitz上下文搜索

    Corruption-Robust Lipschitz Contextual Search. (arXiv:2307.13903v1 [cs.LG])

    [http://arxiv.org/abs/2307.13903](http://arxiv.org/abs/2307.13903)

    该论文研究了学习具有被篡改的二进制信号的Lipschitz函数的问题，提出了一种腐败鲁棒算法。该算法在不同损失函数下实现了不同程度的后悔。

    

    我研究了学习具有被篡改的二进制信号的Lipschitz函数的问题。学习者试图学习一个由对手选择的Lipschitz函数$f$。在每一轮中，对手在输入空间中选择一个上下文向量$x_t$，学习者对真实函数值$f(x_t)$进行猜测，并接收一个指示猜测是高还是低的二进制信号。在总共$C$轮中，信号可能被篡改，但学习者不知道$C$的值。学习者的目标是造成小的累积损失。我提出了一个自然而强大的技术验证，对设计腐败鲁棒算法非常有用。我设计了一些算法（将Lipschitz参数$L$视为常数）：对于对称损失，学习者在$d=1$时达到后悔$O(C\log T)$，在$d>1$时达到后悔$O_d(C\log T + T^{(d-1)/d})$；对于计价损失，学习者在$d/(d+1)$时达到后悔$\widetilde{O}(T^{d/(d+1)} + C\cdot T^{1/(d+1)})$。

    I study the problem of learning a Lipschitz function with corrupted binary signals. The learner tries to learn a Lipschitz function $f$ that the adversary chooses. In each round, the adversary selects a context vector $x_t$ in the input space, and the learner makes a guess to the true function value $f(x_t)$ and receives a binary signal indicating whether the guess was high or low. In a total of $C$ rounds, the signal may be corrupted, though the value of $C$ is unknown to the learner. The learner's goal is to incur a small cumulative loss. I present a natural yet powerful technique sanity check, which proves useful in designing corruption-robust algorithms. I design algorithms which (treating the Lipschitz parameter $L$ as constant): for the symmetric loss, the learner achieves regret $O(C\log T)$ with $d = 1$ and $O_d(C\log T + T^{(d-1)/d})$ with $d > 1$; for the pricing loss the learner achieves regret $\widetilde{O} (T^{d/(d+1)} + C\cdot T^{1/(d+1)})$.
    
[^5]: 从高维观测研究中学习变异源

    Learning sources of variability from high-dimensional observational studies. (arXiv:2307.13868v1 [stat.ME])

    [http://arxiv.org/abs/2307.13868](http://arxiv.org/abs/2307.13868)

    本研究提出了一种针对高维观测研究的方法，将因果估计泛化到任意维度或可测空间的结果，并提出了一种用于名义变量的因果偏差测试。实验证明该方法相比现有策略在有限样本有效性和功率方面有改进。

    

    因果推断研究是否存在一个变量影响观测结果。通过诸如“平均治疗效果”等量化指标，这一范式在许多生物领域中被采用，从疫苗和药物开发到政策干预。不幸的是，大多数方法通常仅限于单变量结果。我们的工作将因果估计泛化到任意维度或可测空间的结果，并将传统的因果估计形式化为名义变量的因果偏差测试。我们提出了一种简单的技术来调整一致性条件独立性测试，并证明了这些测试是一致性因果偏差测试。数值实验表明，与现有策略相比，我们的方法Causal CDcorr在有限样本有效性和功率方面均有改进。我们的方法都是开源的，可在github.com/ebridge2/cdcorr上获得。

    Causal inference studies whether the presence of a variable influences an observed outcome. As measured by quantities such as the "average treatment effect," this paradigm is employed across numerous biological fields, from vaccine and drug development to policy interventions. Unfortunately, the majority of these methods are often limited to univariate outcomes. Our work generalizes causal estimands to outcomes with any number of dimensions or any measurable space, and formulates traditional causal estimands for nominal variables as causal discrepancy tests. We propose a simple technique for adjusting universally consistent conditional independence tests and prove that these tests are universally consistent causal discrepancy tests. Numerical experiments illustrate that our method, Causal CDcorr, leads to improvements in both finite sample validity and power when compared to existing strategies. Our methods are all open source and available at github.com/ebridge2/cdcorr.
    
[^6]: 面向方向的多目标学习：简单且可证明的随机算法

    Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms. (arXiv:2305.18409v1 [cs.LG])

    [http://arxiv.org/abs/2305.18409](http://arxiv.org/abs/2305.18409)

    本文提出了一种新的面向方向的多目标问题，并给出了两种随机算法以解决这个问题，理论上收敛到帕累托稳定点。

    

    多目标优化（MOO）已成为许多与多个目标相关的机器学习问题（如多标准学习和多任务学习（MTL））中一个有影响力的框架。本文提出了一种新的面向方向的多目标问题，通过在一个方向的邻域内限制公共下降方向来规范线性组合目标的最优方向，例如MTL中的平均损失。 这个公式包括GD和MGDA作为特殊情况，享受像CAGrad中的面向方向的好处，以及有利于随机算法的设计。为了解决这个问题，我们提出了随机方向导向多目标梯度下降（SDMGrad），它使用简单的SGD类型的更新算法，以及在目标数量较多的情况下，使用高效的目标采样的SDMGrad-OS算法。 对于恒定的正则化参数λ，我们证明SDMGrad和SDMGrad-OS确实收敛到帕累托稳定点。

    Multi-objective optimization (MOO) has become an influential framework in many machine learning problems with multiple objectives such as learning with multiple criteria and multi-task learning (MTL). In this paper, we propose a new direction-oriented multi-objective problem by regularizing the common descent direction within a neighborhood of a direction that optimizes a linear combination of objectives such as the average loss in MTL. This formulation includes GD and MGDA as special cases, enjoys the direction-oriented benefit as in CAGrad, and facilitates the design of stochastic algorithms. To solve this problem, we propose Stochastic Direction-oriented Multi-objective Gradient descent (SDMGrad) with simple SGD type of updates, and its variant SDMGrad-OS with an efficient objective sampling in the setting where the number of objectives is large. For a constant-level regularization parameter $\lambda$, we show that SDMGrad and SDMGrad-OS provably converge to a Pareto stationary poin
    
[^7]: LegendreTron：升级版多类别正确多项损失学习

    LegendreTron: Uprising Proper Multiclass Loss Learning. (arXiv:2301.11695v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.11695](http://arxiv.org/abs/2301.11695)

    本文提出了一种新颖和实用的方法{\sc LegendreTron}，用于联合学习多类别问题的正确标准损失和概率。这种方法在基准测试中经常优于其他方法。

    

    损失函数是监督学习的基础，通常在模型开发之前选择。为避免选择损失函数可能出现的特定选择，统计决策理论描述了损失的一种理想属性，称为“正确性”，它断言贝叶斯规则是最优的。最近的研究尝试联合学习损失和模型。现有方法通过拟合一个将$\mathbb{R}$单调映射到$[0,1]$的反解标准链接函数来估计二元问题的概率。本文通过使用凸函数梯度的单调性将单调性扩展到$\mathbb{R}^{C-1}$到概率的正投影$\tilde{\Delta}^{C-1}$的映射上。我们提出了一种新颖而实用的方法{\sc LegendreTron}，用于联合学习多类别问题的正确标准损失和概率。在最多1,000种类别的领域基准测试中，我们的实验结果表明，我们的方法始终优于其他基准方法。

    Loss functions serve as the foundation of supervised learning and are often chosen prior to model development. To avoid potentially ad hoc choices of losses, statistical decision theory describes a desirable property for losses known as \emph{properness}, which asserts that Bayes' rule is optimal. Recent works have sought to \emph{learn losses} and models jointly. Existing methods do this by fitting an inverse canonical link function which monotonically maps $\mathbb{R}$ to $[0,1]$ to estimate probabilities for binary problems. In this paper, we extend monotonicity to maps between $\mathbb{R}^{C-1}$ and the projected probability simplex $\tilde{\Delta}^{C-1}$ by using monotonicity of gradients of convex functions. We present {\sc LegendreTron} as a novel and practical method that jointly learns \emph{proper canonical losses} and probabilities for multiclass problems. Tested on a benchmark of domains with up to 1,000 classes, our experimental results show that our method consistently ou
    
[^8]: 超越不变性：针对具有“虚假”相关性的分布的测试时标签转移适应性

    Beyond Invariance: Test-Time Label-Shift Adaptation for Distributions with "Spurious" Correlations. (arXiv:2211.15646v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.15646](http://arxiv.org/abs/2211.15646)

    本文提出了一种测试时标签转移校正方法，通过适应分布的变化来提升预测模型性能，该方法可以处理类别标签和噪声因素的依赖关系随域变化的问题。

    

    测试时数据分布的变化可能对预测模型p(y|x)的性能产生不良影响。我们考虑存在附加元数据标签（例如组标签）z的情况，该标签可以说明分布的变化。特别是，我们假设描述类别标签y和“噪声”因素z之间依赖关系的先验分布p(y, z)可能会随着域的变化而改变，要么是由于这些项之间的相关性的变化，要么是由于其中一个变量的边际分布的变化。但是，我们假设特征的生成模型p(x|y, z)在域间是不变的。我们注意到这相当于广泛使用的“标签转移”假设的扩展版本，其中标签现在也包括噪声因素z。基于此观察，我们提出了一种测试时标签转移校正方法，通过对未标记样本应用期望最大化算法来适应p(y, z)的变化。

    Changes in the data distribution at test time can have deleterious effects on the performance of predictive models $p(y|x)$. We consider situations where there are additional meta-data labels (such as group labels), denoted by $z$, that can account for such changes in the distribution. In particular, we assume that the prior distribution $p(y, z)$, which models the dependence between the class label $y$ and the "nuisance" factors $z$, may change across domains, either due to a change in the correlation between these terms, or a change in one of their marginals. However, we assume that the generative model for features $p(x|y, z)$ is invariant across domains. We note that this corresponds to an expanded version of the widely used "label shift" assumption, where the labels now also include the nuisance factors $z$. Based on this observation, we propose a test-time label shift correction that adapts to changes in the joint distribution $p(y, z)$ using EM applied to unlabeled samples from 
    
[^9]: 严格的动力学均场理论用于随机梯度下降方法

    Rigorous dynamical mean field theory for stochastic gradient descent methods. (arXiv:2210.06591v2 [math-ph] UPDATED)

    [http://arxiv.org/abs/2210.06591](http://arxiv.org/abs/2210.06591)

    本研究通过证明的闭式方程，描述了一类基于梯度的方法在高维情况下的精确渐进性能，为随机梯度下降等算法提供了理论支持，并提供了数值实现。

    

    我们证明了一类基于梯度的方法在高维情况下的精确渐进性能闭式方程，该方法从高斯数据的经验风险最小化学习估计器（例如M-估计器，浅层神经网络...）。这包括了广泛使用的算法，如随机梯度下降（SGD）或Nesterov加速。得到的方程与将动力学均场理论（DMFT）方程离散化后应用于梯度流时产生的方程相匹配。我们的证明方法允许我们明确描述记忆核在有效动力学中如何构建，并且包括非可分离的更新函数，允许具有非单位协方差矩阵的数据集。最后，我们提供了具有通用批处理大小和恒定学习率的SGD方程的数值实现。

    We prove closed-form equations for the exact high-dimensional asymptotics of a family of first order gradient-based methods, learning an estimator (e.g. M-estimator, shallow neural network, ...) from observations on Gaussian data with empirical risk minimization. This includes widely used algorithms such as stochastic gradient descent (SGD) or Nesterov acceleration. The obtained equations match those resulting from the discretization of dynamical mean-field theory (DMFT) equations from statistical physics when applied to gradient flow. Our proof method allows us to give an explicit description of how memory kernels build up in the effective dynamics, and to include non-separable update functions, allowing datasets with non-identity covariance matrices. Finally, we provide numerical implementations of the equations for SGD with generic extensive batch-size and with constant learning rates.
    

