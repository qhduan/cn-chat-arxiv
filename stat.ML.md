# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Interpretable Evaluation of Entropy-based Novelty of Generative Models](https://arxiv.org/abs/2402.17287) | 提出了一种用于评估生成模型新颖性的基于核的熵新颖性 (KEN) 分数 |
| [^2] | [HyperAgent: A Simple, Scalable, Efficient and Provable Reinforcement Learning Framework for Complex Environments](https://arxiv.org/abs/2402.10228) | HyperAgent提出了一种简单、高效、可扩展的强化学习框架，在复杂环境下能够实现高效的计算和数据选择，是首个达到可证明可扩展的每步计算复杂度以及次线性后悔的方法。 |
| [^3] | [Future Directions in Foundations of Graph Machine Learning](https://arxiv.org/abs/2402.02287) | 图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。 |
| [^4] | [Prepare Non-classical Collective Spin State by Reinforcement Learning.](http://arxiv.org/abs/2401.16320) | 通过强化学习设计控制场的方案成功生成了非经典态，以应用于自旋压缩态的产生。该方法在保持压缩和纠缠的同时提供了不同的控制序列，并观察到控制脉冲密集应用可以提高结果的性能。 |
| [^5] | [Adam through a Second-Order Lens.](http://arxiv.org/abs/2310.14963) | 该论文提出了AdamQLR，它是一个通过将K-FAC中的技术与Adam的更新方法相结合的优化器，通过考虑二阶数据上的Adam行为而得到启发。在回归和分类任务上进行了评估，结果显示AdamQLR在运行时间和推广性能方面表现出良好的竞争力。 |
| [^6] | [SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks.](http://arxiv.org/abs/2310.03684) | SmoothLLM是第一个用于减轻大型语言模型上越狱攻击的算法，通过在输入提示上随机扰动并汇总预测结果来检测对抗性输入，将攻击成功率降低至不到一个百分点，并提供了可证明的保证。 |
| [^7] | [Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks.](http://arxiv.org/abs/2310.02600) | 通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。 |
| [^8] | [Initial Guessing Bias: How Untrained Networks Favor Some Classes.](http://arxiv.org/abs/2306.00809) | 本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。 |

# 详细

[^1]: 一个可解释的生成模型熵值新颖性评估

    An Interpretable Evaluation of Entropy-based Novelty of Generative Models

    [https://arxiv.org/abs/2402.17287](https://arxiv.org/abs/2402.17287)

    提出了一种用于评估生成模型新颖性的基于核的熵新颖性 (KEN) 分数

    

    生成模型框架和架构的巨大发展需要有原则的方法来评估模型相对于参考数据集或基线生成模型的新颖性。 虽然最近的文献已广泛研究了生成模型的质量、多样性和泛化能力的评估，但与基线模型相比的模型新颖性评估在机器学习社区中尚未得到充分研究。在这项工作中，我们关注多模态生成模型下的新颖性评估，并尝试回答以下问题：给定生成模型 $\mathcal{G}$ 的样本和参考数据集 $\mathcal{S}$，我们如何发现并计算 $\mathcal{G}$ 比 $\mathcal{S}$ 中更频繁地表达的模式。 我们介绍了一种谱方法来描述这一任务，并提出了基于核的熵新颖性 (KEN) 分数来量化基于模式的新颖性

    arXiv:2402.17287v1 Announce Type: new  Abstract: The massive developments of generative model frameworks and architectures require principled methods for the evaluation of a model's novelty compared to a reference dataset or baseline generative models. While the recent literature has extensively studied the evaluation of the quality, diversity, and generalizability of generative models, the assessment of a model's novelty compared to a baseline model has not been adequately studied in the machine learning community. In this work, we focus on the novelty assessment under multi-modal generative models and attempt to answer the following question: Given the samples of a generative model $\mathcal{G}$ and a reference dataset $\mathcal{S}$, how can we discover and count the modes expressed by $\mathcal{G}$ more frequently than in $\mathcal{S}$. We introduce a spectral approach to the described task and propose the Kernel-based Entropic Novelty (KEN) score to quantify the mode-based novelty 
    
[^2]: HyperAgent：一种简单、可扩展、高效且可证明用于复杂环境的强化学习框架

    HyperAgent: A Simple, Scalable, Efficient and Provable Reinforcement Learning Framework for Complex Environments

    [https://arxiv.org/abs/2402.10228](https://arxiv.org/abs/2402.10228)

    HyperAgent提出了一种简单、高效、可扩展的强化学习框架，在复杂环境下能够实现高效的计算和数据选择，是首个达到可证明可扩展的每步计算复杂度以及次线性后悔的方法。

    

    为了在资源约束下解决复杂任务，强化学习（RL）代理需要简单、高效、可扩展、具有大状态空间和不断积累的交互数据。我们提出了HyperAgent，这是一个具有超模型、索引抽样方案和增量更新机制的RL框架，可以在一般价值函数逼近中进行计算高效的顺序后验逼近和数据高效的动作选择，超越了共轭性。HyperAgent的实现简单，只需要在DDQN中添加一个模块和一行额外代码。在实践中，HyperAgent在大规模深度RL基准测试中表现出稳健的性能，无论是在数据还是计算方面都获得了显着的效率提升。在理论上，在实际可扩展的算法中，HyperAgent是第一个能够实现可证明可扩展的每步计算复杂度以及次线性后悔的方法。

    arXiv:2402.10228v1 Announce Type: cross  Abstract: To solve complex tasks under resource constraints, reinforcement learning (RL) agents need to be simple, efficient, and scalable with (1) large state space and (2) increasingly accumulated data of interactions. We propose the HyperAgent, a RL framework with hypermodel, index sampling schemes and incremental update mechanism, enabling computation-efficient sequential posterior approximation and data-efficient action selection under general value function approximation beyond conjugacy. The implementation of \HyperAgent is simple as it only adds one module and one line of code additional to DDQN. Practically, HyperAgent demonstrates its robust performance in large-scale deep RL benchmarks with significant efficiency gain in terms of both data and computation. Theoretically, among the practically scalable algorithms, HyperAgent is the first method to achieve provably scalable per-step computational complexity as well as sublinear regret u
    
[^3]: 图机器学习基础的未来方向

    Future Directions in Foundations of Graph Machine Learning

    [https://arxiv.org/abs/2402.02287](https://arxiv.org/abs/2402.02287)

    图机器学习领域的未来方向应该是发展一个更加均衡的理论，从更完整的角度探究图神经网络的表达能力、泛化和优化之间的相互关系。

    

    随着图数据在不同学科（从生命科学到社会科学和工程科学）上的广泛应用，图机器学习，尤其是使用图神经网络（GNNs），引起了人们浓厚的兴趣。尽管在实际应用中取得了成功，但我们对GNNs性质的理论理解仍然非常不完整。最近的理论发展主要集中在阐明GNNs粗粒度表达能力方面，主要采用组合技巧。然而，这些研究与实践并不完全一致，特别是在使用随机一阶优化技术训练GNNs时，对GNNs的泛化行为的理解。在这篇定位论文中，我们认为图机器学习领域需要将注意力转移到发展一个更加均衡的图机器学习理论上来，重点关注表达能力、泛化和优化的相互关系的更全面的理解。

    Machine learning on graphs, especially using graph neural networks (GNNs), has seen a surge in interest due to the wide availability of graph data across a broad spectrum of disciplines, from life to social and engineering sciences. Despite their practical success, our theoretical understanding of the properties of GNNs remains highly incomplete. Recent theoretical advancements primarily focus on elucidating the coarse-grained expressive power of GNNs, predominantly employing combinatorial techniques. However, these studies do not perfectly align with practice, particularly in understanding the generalization behavior of GNNs when trained with stochastic first-order optimization techniques. In this position paper, we argue that the graph machine learning community needs to shift its attention to developing a more balanced theory of graph machine learning, focusing on a more thorough understanding of the interplay of expressive power, generalization, and optimization.
    
[^4]: 利用强化学习生成非经典集合自旋态的方案

    Prepare Non-classical Collective Spin State by Reinforcement Learning. (arXiv:2401.16320v1 [quant-ph])

    [http://arxiv.org/abs/2401.16320](http://arxiv.org/abs/2401.16320)

    通过强化学习设计控制场的方案成功生成了非经典态，以应用于自旋压缩态的产生。该方法在保持压缩和纠缠的同时提供了不同的控制序列，并观察到控制脉冲密集应用可以提高结果的性能。

    

    我们提出了一种利用强化学习来设计控制场的方案，用于生成非经典态。该方案以应用于开放集体自旋模型中的自旋压缩态为例，其中设计了一个线性控制项来控制动力学。强化学习代理根据以耗散和去相干为特征的环境中的相干自旋态开始，确定了控制脉冲的时间序列。与恒定控制方案相比，这种方法提供了多种控制序列，保持了集体自旋压缩和纠缠。观察到控制脉冲的密集应用可以增强结果的性能。此外，通过添加控制操作，性能得到了轻微增强。所提出的策略在较大系统中展现了更高的效果。对储备热激发对控制结果有不利影响。应该确认这一点。

    We propose a scheme leveraging reinforcement learning to engineer control fields for generating non-classical states. It is exemplified by the application to prepare spin squeezed state for an open collective spin model where a linear control term is designed to govern the dynamics. The reinforcement learning agent determines the temporal sequence of control pulses, commencing from coherent spin state in an environment characterized by dissipation and dephasing. When compared to constant control scenarios, this approach provides various control sequences maintaining collective spin squeezing and entanglement. It is observed that denser application of the control pulses enhances the performance of the outcomes. Furthermore, there is a minor enhancement in the performance by adding control actions. The proposed strategy demonstrates increased effectiveness for larger systems. And thermal excitations of the reservoir are detrimental to the control outcomes. It should be confirmed that thi
    
[^5]: 通过二阶透镜看Adam

    Adam through a Second-Order Lens. (arXiv:2310.14963v1 [cs.LG])

    [http://arxiv.org/abs/2310.14963](http://arxiv.org/abs/2310.14963)

    该论文提出了AdamQLR，它是一个通过将K-FAC中的技术与Adam的更新方法相结合的优化器，通过考虑二阶数据上的Adam行为而得到启发。在回归和分类任务上进行了评估，结果显示AdamQLR在运行时间和推广性能方面表现出良好的竞争力。

    

    深度学习优化研究存在一种紧张状态，即第一阶梯度法（如SGD和Adam）的计算效率与第二阶曲率法（如拟牛顿方法和K-FAC）的理论效率之间的紧张关系。我们试图将这两种方法的优点结合到一个计算上高效的算法中。注意到二阶方法通常依赖于稳定的启发式方法（如Levenberg-Marquardt阻尼），我们提出AdamQLR：一个将K-FAC中的阻尼和学习率选择技术与Adam提出的更新方向相结合的优化器，通过考虑Adam在二阶数据上的表现而得到启发。我们在各种规模的回归和分类任务上评估了AdamQLR，在运行时间与竞争性推广性能之间取得了竞争性的结果。

    Research into optimisation for deep learning is characterised by a tension between the computational efficiency of first-order, gradient-based methods (such as SGD and Adam) and the theoretical efficiency of second-order, curvature-based methods (such as quasi-Newton methods and K-FAC). We seek to combine the benefits of both approaches into a single computationally-efficient algorithm. Noting that second-order methods often depend on stabilising heuristics (such as Levenberg-Marquardt damping), we propose AdamQLR: an optimiser combining damping and learning rate selection techniques from K-FAC (Martens and Grosse, 2015) with the update directions proposed by Adam, inspired by considering Adam through a second-order lens. We evaluate AdamQLR on a range of regression and classification tasks at various scales, achieving competitive generalisation performance vs runtime.
    
[^6]: SmoothLLM：防御大型语言模型免受越狱攻击

    SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks. (arXiv:2310.03684v1 [cs.LG])

    [http://arxiv.org/abs/2310.03684](http://arxiv.org/abs/2310.03684)

    SmoothLLM是第一个用于减轻大型语言模型上越狱攻击的算法，通过在输入提示上随机扰动并汇总预测结果来检测对抗性输入，将攻击成功率降低至不到一个百分点，并提供了可证明的保证。

    

    尽管努力将大型语言模型（LLM）与人类价值观保持一致，但广泛使用的LLM（如GPT、Llama、Claude和PaLM）仍然容易受到越狱攻击，即对目标LLM进行欺骗，以生成不合适的内容。为了解决这个漏洞，我们提出了SmoothLLM，这是第一个旨在减轻LLM上的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的改变很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后汇总相应的预测结果来检测对抗性输入。SmoothLLM将众多热门LLM的攻击成功率降低至不到一个百分点，避免了不必要的保守性，并对攻击缓解提供了可证明的保证。此外，我们的防御使用的查询数量比现有的攻击方法少得多，并且与任何LLM兼容。

    Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.
    
[^7]: 使用图神经网络的非规则空间数据的神经贝叶斯估计器

    Neural Bayes Estimators for Irregular Spatial Data using Graph Neural Networks. (arXiv:2310.02600v1 [stat.ME])

    [http://arxiv.org/abs/2310.02600](http://arxiv.org/abs/2310.02600)

    通过使用图神经网络，该论文提出了一种解决非规则空间数据的参数估计问题的方法，扩展了神经贝叶斯估计器的应用范围，并带来了显著的计算优势。

    

    神经贝叶斯估计器是一种以快速和免似然方式逼近贝叶斯估计器的神经网络。它们在空间模型和数据中的使用非常吸引人，因为估计经常是计算上的瓶颈。然而，到目前为止，空间应用中的神经贝叶斯估计器仅限于在规则的网格上收集的数据。这些估计器目前还依赖于预先规定的空间位置，这意味着神经网络需要重新训练以适应新的数据集；这使它们在许多应用中变得不实用，并阻碍了它们的广泛应用。在本研究中，我们采用图神经网络来解决从任意空间位置收集的数据进行参数估计的重要问题。除了将神经贝叶斯估计扩展到非规则空间数据之外，我们的架构还带来了显着的计算优势，因为该估计器可以用于任何排列或数量的位置和独立的重复实验中。

    Neural Bayes estimators are neural networks that approximate Bayes estimators in a fast and likelihood-free manner. They are appealing to use with spatial models and data, where estimation is often a computational bottleneck. However, neural Bayes estimators in spatial applications have, to date, been restricted to data collected over a regular grid. These estimators are also currently dependent on a prescribed set of spatial locations, which means that the neural network needs to be re-trained for new data sets; this renders them impractical in many applications and impedes their widespread adoption. In this work, we employ graph neural networks to tackle the important problem of parameter estimation from data collected over arbitrary spatial locations. In addition to extending neural Bayes estimation to irregular spatial data, our architecture leads to substantial computational benefits, since the estimator can be used with any arrangement or number of locations and independent repli
    
[^8]: 初始猜测偏差：未经过训练的神经网络倾向于某些类别

    Initial Guessing Bias: How Untrained Networks Favor Some Classes. (arXiv:2306.00809v1 [cs.LG])

    [http://arxiv.org/abs/2306.00809](http://arxiv.org/abs/2306.00809)

    本文提出了“初始猜测偏差”现象，即在未经过训练的神经网络中，由于架构选择的影响，模型往往会将所有预测指向同一个类别。该现象对架构选择和初始化有实际指导意义，并具有理论后果，例如节点置换对称性的崩溃和深度带来的非平凡差异。

    

    神经网络的初始状态在调节后续的训练过程中扮演重要角色。在分类问题的背景下，我们提供了理论分析，证明神经网络的结构可以在训练之前，甚至在不存在显式偏差的情况下，使模型将所有预测都指向同一个类别。我们展示了这种现象的存在，称为“初始猜测偏差”（Initial Guessing Bias，IGB），这取决于架构选择，例如激活函数、最大池化层和网络深度。我们对IGB进行的分析具有实际意义，可以指导架构的选择和初始化。我们还强调理论后果，例如节点置换对称性的崩溃、自平均的破坏、某些均场近似的有效性以及深度带来的非平凡差异。

    The initial state of neural networks plays a central role in conditioning the subsequent training dynamics. In the context of classification problems, we provide a theoretical analysis demonstrating that the structure of a neural network can condition the model to assign all predictions to the same class, even before the beginning of training, and in the absence of explicit biases. We show that the presence of this phenomenon, which we call "Initial Guessing Bias" (IGB), depends on architectural choices such as activation functions, max-pooling layers, and network depth. Our analysis of IGB has practical consequences, in that it guides architecture selection and initialization. We also highlight theoretical consequences, such as the breakdown of node-permutation symmetry, the violation of self-averaging, the validity of some mean-field approximations, and the non-trivial differences arising with depth.
    

