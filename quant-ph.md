# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Quantum Machine Learning Implementations: Proposals and Experiments.](http://arxiv.org/abs/2303.06263) | 本文概述了量子机器学习领域中最近的理论提议及其实验实现，重点回顾了量子强化学习、量子自编码器和量子记忆电阻器等高影响主题，并强调了推动这项技术的初步量子实现的必要性。 |
| [^2] | [Enabling Non-Linear Quantum Operations through Variational Quantum Splines.](http://arxiv.org/abs/2303.04788) | 本文提出了一种新方法——广义QSplines，使用混合量子-经典计算来近似非线性量子激活函数，克服了原始QSplines在量子硬件方面的高要求，并适合嵌入现有的量子神经网络架构中。 |
| [^3] | [Lightsolver challenges a leading deep learning solver for Max-2-SAT problems.](http://arxiv.org/abs/2302.06926) | 本文比较了LightSolver的量子启发式算法和领先的深度学习求解器在MAX-2-SAT问题上的表现，实验结果表明LightSolver实现了显著更小的最优解时间。 |
| [^4] | [Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality.](http://arxiv.org/abs/2301.09241) | 本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。 |

# 详细

[^1]: 量子机器学习实现：提议和实验

    Quantum Machine Learning Implementations: Proposals and Experiments. (arXiv:2303.06263v1 [quant-ph])

    [http://arxiv.org/abs/2303.06263](http://arxiv.org/abs/2303.06263)

    本文概述了量子机器学习领域中最近的理论提议及其实验实现，重点回顾了量子强化学习、量子自编码器和量子记忆电阻器等高影响主题，并强调了推动这项技术的初步量子实现的必要性。

    This article provides an overview of recent theoretical proposals and experimental implementations in the field of quantum machine learning, with a focus on high-impact topics such as quantum reinforcement learning, quantum autoencoders, and quantum memristors. The article emphasizes the necessity of pushing forward initial quantum implementations of this technology to achieve better machine learning calculations than any current or future computing paradigm.

    本文概述了量子机器学习领域中最近的理论提议及其实验实现，并回顾了特定的高影响主题，如量子强化学习、量子自编码器和量子记忆电阻器，以及它们在量子光子学和超导电路平台上的实验实现。量子机器学习领域可能是首批为工业和社会带来益处的量子技术之一。因此，有必要推动这项技术的初步量子实现，以在嘈杂的中间规模量子计算机上实现比任何当前或未来计算范式更好的机器学习计算。

    This article gives an overview and a perspective of recent theoretical proposals and their experimental implementations in the field of quantum machine learning. Without an aim to being exhaustive, the article reviews specific high-impact topics such as quantum reinforcement learning, quantum autoencoders, and quantum memristors, and their experimental realizations in the platforms of quantum photonics and superconducting circuits. The field of quantum machine learning could be among the first quantum technologies producing results that are beneficial for industry and, in turn, to society. Therefore, it is necessary to push forward initial quantum implementations of this technology, in Noisy Intermediate-Scale Quantum Computers, aiming for achieving fruitful calculations in machine learning that are better than with any other current or future computing paradigm.
    
[^2]: 通过变分量子样条使非线性量子操作成为可能

    Enabling Non-Linear Quantum Operations through Variational Quantum Splines. (arXiv:2303.04788v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2303.04788](http://arxiv.org/abs/2303.04788)

    本文提出了一种新方法——广义QSplines，使用混合量子-经典计算来近似非线性量子激活函数，克服了原始QSplines在量子硬件方面的高要求，并适合嵌入现有的量子神经网络架构中。

    This paper proposes a novel method, Generalised QSplines (GQSplines), for approximating non-linear quantum activation functions using hybrid quantum-classical computation, which overcomes the highly demanding requirements of the original QSplines in terms of quantum hardware and is suitable to be embedded in existing quantum neural network architectures.

    量子力学的假设仅对量子状态施加幺正变换，这对于量子机器学习算法来说是一个严重的限制。最近提出了量子样条（QSplines）来近似量子激活函数，以在量子算法中引入非线性。然而，QSplines使用HHL作为子程序，并需要一个容错的量子计算机才能正确实现。本文提出了广义QSplines（GQSplines），一种使用混合量子-经典计算来近似非线性量子激活函数的新方法。GQSplines克服了原始QSplines在量子硬件方面的高要求，并可以使用近期的量子计算机来实现。此外，所提出的方法依赖于灵活的问题表示，适合嵌入现有的量子神经网络架构中。

    The postulates of quantum mechanics impose only unitary transformations on quantum states, which is a severe limitation for quantum machine learning algorithms. Quantum Splines (QSplines) have recently been proposed to approximate quantum activation functions to introduce non-linearity in quantum algorithms. However, QSplines make use of the HHL as a subroutine and require a fault-tolerant quantum computer to be correctly implemented. This work proposes the Generalised QSplines (GQSplines), a novel method for approximating non-linear quantum activation functions using hybrid quantum-classical computation. The GQSplines overcome the highly demanding requirements of the original QSplines in terms of quantum hardware and can be implemented using near-term quantum computers. Furthermore, the proposed method relies on a flexible problem representation for non-linear approximation and it is suitable to be embedded in existing quantum neural network architectures. In addition, we provide a pr
    
[^3]: Lightsolver挑战领先的深度学习求解器解决Max-2-SAT问题

    Lightsolver challenges a leading deep learning solver for Max-2-SAT problems. (arXiv:2302.06926v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2302.06926](http://arxiv.org/abs/2302.06926)

    本文比较了LightSolver的量子启发式算法和领先的深度学习求解器在MAX-2-SAT问题上的表现，实验结果表明LightSolver实现了显著更小的最优解时间。

    This paper compares LightSolver's quantum-inspired algorithm to a leading deep-learning solver for the MAX-2-SAT problem, and shows that LightSolver achieves significantly smaller time-to-optimal-solution compared to a state-of-the-art deep-learning algorithm.

    最大2-SAT问题（MAX-2-SAT）是一种已知为NP难的组合决策问题。本文比较了LightSolver的量子启发式算法和领先的深度学习求解器在MAX-2-SAT问题上的表现。基准数据集上的实验表明，与最先进的深度学习算法相比，LightSolver实现了显著更小的最优解时间，其中性能提升的增益往往随着问题规模的增加而增加。

    Maximum 2-satisfiability (MAX-2-SAT) is a type of combinatorial decision problem that is known to be NP-hard. In this paper, we compare LightSolver's quantum-inspired algorithm to a leading deep-learning solver for the MAX-2-SAT problem. Experiments on benchmark data sets show that LightSolver achieves significantly smaller time-to-optimal-solution compared to a state-of-the-art deep-learning algorithm, where the gain in performance tends to increase with the problem size.
    
[^4]: 量子蒙特卡罗算法在金融中高维期权定价中解决Black-Scholes PDE及其克服维度诅咒的证明

    Quantum Monte Carlo algorithm for solving Black-Scholes PDEs for high-dimensional option pricing in finance and its proof of overcoming the curse of dimensionality. (arXiv:2301.09241v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2301.09241](http://arxiv.org/abs/2301.09241)

    本文提出了一种量子蒙特卡罗算法，用于解决高维Black-Scholes PDE的高维期权定价，其复杂度被多项式地限制，克服了维度诅咒。

    This paper proposes a quantum Monte Carlo algorithm for high-dimensional option pricing by solving high-dimensional Black-Scholes PDEs with correlation, and proves that its computational complexity is polynomially bounded in the space dimension and the reciprocal of the prescribed accuracy, overcoming the curse of dimensionality.

    本文提供了一种量子蒙特卡罗算法，用于解决具有相关性的高维Black-Scholes PDE的高维期权定价。期权的支付函数为一般形式，只需要连续且分段仿射（CPWA），涵盖了金融中使用的大多数相关支付函数。我们提供了算法的严格误差分析和复杂度分析。特别地，我们证明了我们的算法的计算复杂度在PDE的空间维度$d$和所需精度$\varepsilon$的倒数中被多项式地限制，从而证明了我们的量子蒙特卡罗算法不会受到维度诅咒的影响。

    In this paper we provide a quantum Monte Carlo algorithm to solve high-dimensional Black-Scholes PDEs with correlation for high-dimensional option pricing. The payoff function of the option is of general form and is only required to be continuous and piece-wise affine (CPWA), which covers most of the relevant payoff functions used in finance. We provide a rigorous error analysis and complexity analysis of our algorithm. In particular, we prove that the computational complexity of our algorithm is bounded polynomially in the space dimension $d$ of the PDE and the reciprocal of the prescribed accuracy $\varepsilon$ and so demonstrate that our quantum Monte Carlo algorithm does not suffer from the curse of dimensionality.
    

