# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type](https://rss.arxiv.org/abs/2402.01632) | 这篇论文提出了一种新的贝叶斯优化算法，可以处理具有任意类型未知超参数的情况，并具有无遗憾特性。 |
| [^2] | [Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic](https://arxiv.org/abs/2402.09469) | 本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。 |
| [^3] | [Scalable Structure Learning for Sparse Context-Specific Causal Systems](https://arxiv.org/abs/2402.07762) | 提出了一种可扩展的混合算法，用于学习特定背景模型，通过结合基于顺序的MCMC算法和稀疏性假设实现可扩展学习，该方法在准确性和可扩展性方面表现良好。 |
| [^4] | [Understanding What Affects Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence](https://arxiv.org/abs/2402.02701) | 本文通过理论和实证研究，揭示了在测试环境具有干扰因素时影响视觉强化学习中泛化差距的关键因素。结果表明，最小化训练和测试环境之间的表示距离是减少泛化差距最关键的因素。 |
| [^5] | [TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability.](http://arxiv.org/abs/2312.06499) | 本论文提出了一种新颖的方法，通过对NLP模型的嵌入层级进行操作，借鉴了最新的解释性人工智能技术，通过嵌入转换来消除隐含的敏感信息，从而实现模型的公平性。 |

# 详细

[^1]: 超越尺度：具有任意类型未知超参数的无遗憾贝叶斯优化

    Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type

    [https://rss.arxiv.org/abs/2402.01632](https://rss.arxiv.org/abs/2402.01632)

    这篇论文提出了一种新的贝叶斯优化算法，可以处理具有任意类型未知超参数的情况，并具有无遗憾特性。

    

    贝叶斯优化需要拟合高斯过程模型，而拟合高斯过程模型需要指定超参数 - 大部分理论文献假设这些超参数是已知的。之前的理论研究通常假设数据在空间中均匀填充，而常用的高斯过程超参数的最大似然估计器只有在这种情况下才是一致的。然而，在贝叶斯优化中，数据不一定满足这种均匀填充的条件。由于无法保证超参数估计的正确性，并且这些超参数可以显著影响高斯过程拟合，因此对具有未知超参数的贝叶斯优化进行理论分析非常具有挑战性。之前提出的具有无遗憾特性的算法仅能处理特殊情况下的未知长度尺度、再生核希尔伯特空间范数，并且仅适用于频率派的情况。我们提出了一种新的算法，命名为HE-GP-UCB，它是第一个具有无遗憾特性的算法，在具有未知超参数的情况下实现了贝叶斯优化。

    Bayesian optimisation requires fitting a Gaussian process model, which in turn requires specifying hyperparameters - most of the theoretical literature assumes those hyperparameters are known. The commonly used maximum likelihood estimator for hyperparameters of the Gaussian process is consistent only if the data fills the space uniformly, which does not have to be the case in Bayesian optimisation. Since no guarantees exist regarding the correctness of hyperparameter estimation, and those hyperparameters can significantly affect the Gaussian process fit, theoretical analysis of Bayesian optimisation with unknown hyperparameters is very challenging. Previously proposed algorithms with the no-regret property were only able to handle the special case of unknown lengthscales, reproducing kernel Hilbert space norm and applied only to the frequentist case. We propose a novel algorithm, HE-GP-UCB, which is the first algorithm enjoying the no-regret property in the case of unknown hyperparame
    
[^2]: 神经网络中的傅立叶电路：解锁大规模语言模型在数学推理和模运算中的潜力

    Fourier Circuits in Neural Networks: Unlocking the Potential of Large Language Models in Mathematical Reasoning and Modular Arithmetic

    [https://arxiv.org/abs/2402.09469](https://arxiv.org/abs/2402.09469)

    本研究探索了神经网络和Transformer在数学推理和模运算中的潜力。我们分析了单隐藏层神经网络和单层Transformer在解决复杂代数学习任务中的特征。阐明了边缘最大化原则对单隐藏层神经网络的影响。

    

    在机器学习不断发展的背景下，理解神经网络和Transformer所利用的内部表示是一个关键挑战。本研究在近期的研究基础上，对网络采用特定计算策略背后的原因进行了探索。我们的研究聚焦于涉及k个输入的复杂代数学习任务，即模运算的加法。我们对单隐藏层神经网络和单层Transformer在解决这一任务中学到的特征进行了深入的分析。我们理论框架的一个关键是阐明边缘最大化原则对单隐藏层神经网络采用的特征的影响。其中，p表示模数，Dp表示k个输入的模运算数据集，m表示网络输出。

    arXiv:2402.09469v1 Announce Type: new  Abstract: In the evolving landscape of machine learning, a pivotal challenge lies in deciphering the internal representations harnessed by neural networks and Transformers. Building on recent progress toward comprehending how networks execute distinct target functions, our study embarks on an exploration of the underlying reasons behind networks adopting specific computational strategies. We direct our focus to the complex algebraic learning task of modular addition involving $k$ inputs. Our research presents a thorough analytical characterization of the features learned by stylized one-hidden layer neural networks and one-layer Transformers in addressing this task.   A cornerstone of our theoretical framework is the elucidation of how the principle of margin maximization shapes the features adopted by one-hidden layer neural networks. Let $p$ denote the modulus, $D_p$ denote the dataset of modular arithmetic with $k$ inputs and $m$ denote the net
    
[^3]: 可扩展的稀疏特定背景下因果系统的结构学习

    Scalable Structure Learning for Sparse Context-Specific Causal Systems

    [https://arxiv.org/abs/2402.07762](https://arxiv.org/abs/2402.07762)

    提出了一种可扩展的混合算法，用于学习特定背景模型，通过结合基于顺序的MCMC算法和稀疏性假设实现可扩展学习，该方法在准确性和可扩展性方面表现良好。

    

    已经提出了几种表示共同分布分类变量之间特定背景下关系的方法，并且提出了结构学习算法。然而，由于大量特定背景模型的存在，现有的基于优化的方法在可扩展性方面受到限制，而基于约束的方法比约束DAG学习算法更容易出错，因为必须测试更多关系。我们提出了一种混合算法来学习特定背景模型，能够扩展到数百个变量，并且测试的约束不多于标准DAG学习算法。通过结合基于顺序的MCMC算法和类似于DAG模型常用的稀疏性假设，实现了可扩展的学习。为了实现这种方法，我们解决了Alon和Balogh最近提出的一个开放问题的特殊情况。经过在合成数据和真实世界示例上的实验证明，该方法在准确性和可扩展性方面表现良好。

    Several approaches to graphically representing context-specific relations among jointly distributed categorical variables have been proposed, along with structure learning algorithms. While existing optimization-based methods have limited scalability due to the large number of context-specific models, the constraint-based methods are more prone to error than even constraint-based DAG learning algorithms since more relations must be tested. We present a hybrid algorithm for learning context-specific models that scales to hundreds of variables while testing no more constraints than standard DAG learning algorithms. Scalable learning is achieved through a combination of an order-based MCMC algorithm and sparsity assumptions analogous to those typically invoked for DAG models. To implement the method, we solve a special case of an open problem recently posed by Alon and Balogh. The method is shown to perform well on synthetic data and real world examples, in terms of both accuracy and scal
    
[^4]: 理解影响视觉强化学习中泛化差距的因素：理论和实证证据

    Understanding What Affects Generalization Gap in Visual Reinforcement Learning: Theory and Empirical Evidence

    [https://arxiv.org/abs/2402.02701](https://arxiv.org/abs/2402.02701)

    本文通过理论和实证研究，揭示了在测试环境具有干扰因素时影响视觉强化学习中泛化差距的关键因素。结果表明，最小化训练和测试环境之间的表示距离是减少泛化差距最关键的因素。

    

    最近，有许多努力致力于在视觉强化学习中学习对连续控制有用的策略。在这种场景下，学习一个具有泛化能力的策略非常重要，因为测试环境可能与训练环境不同，例如在部署过程中存在干扰因素。许多实际算法被提出来解决这个问题。然而，据我们所知，它们中没有一种算法能够从理论上解释泛化差距的影响因素以及为什么他们的方法有效。在本文中，我们通过在测试环境具有干扰因素时理论上回答影响泛化差距的关键因素来解决这个问题。我们的理论表明，最小化训练和测试环境之间的表示距离（与人类直觉一致）对于减少泛化差距的效益至关重要。我们的理论结果得到了DM数据的实证证据的支持。

    Recently, there are many efforts attempting to learn useful policies for continuous control in visual reinforcement learning (RL). In this scenario, it is important to learn a generalizable policy, as the testing environment may differ from the training environment, e.g., there exist distractors during deployment. Many practical algorithms are proposed to handle this problem. However, to the best of our knowledge, none of them provide a theoretical understanding of what affects the generalization gap and why their proposed methods work. In this paper, we bridge this issue by theoretically answering the key factors that contribute to the generalization gap when the testing environment has distractors. Our theories indicate that minimizing the representation distance between training and testing environments, which aligns with human intuition, is the most critical for the benefit of reducing the generalization gap. Our theoretical results are supported by the empirical evidence in the DM
    
[^5]: TaCo：通过信息论和可解释性在NLP中的输出嵌入中实现有针对性的概念去除

    TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability. (arXiv:2312.06499v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.06499](http://arxiv.org/abs/2312.06499)

    本论文提出了一种新颖的方法，通过对NLP模型的嵌入层级进行操作，借鉴了最新的解释性人工智能技术，通过嵌入转换来消除隐含的敏感信息，从而实现模型的公平性。

    

    自然语言处理（NLP）模型的公平性已成为一个关键问题。信息论表明，为了实现公平性，模型不应能够预测敏感变量，如性别、种族和年龄。然而，与这些变量相关的信息通常以隐式的方式出现在语言中，这给识别和减少偏见带来了挑战。为了解决这个问题，我们提出了一种新颖的方法，在NLP模型的嵌入层级上操作，独立于具体的架构。我们的方法借鉴了最近解释性人工智能技术的进展，并采用嵌入转换来消除选定变量中的隐式信息。通过直接操纵最后一层的嵌入，我们的方法能够无缝集成到现有模型中，而无需进行重大修改或重训练。在评估中，我们展示了该后处理方法显著降低了与性别相关的关联性。

    The fairness of Natural Language Processing (NLP) models has emerged as a crucial concern. Information theory indicates that to achieve fairness, a model should not be able to predict sensitive variables, such as gender, ethnicity, and age. However, information related to these variables often appears implicitly in language, posing a challenge in identifying and mitigating biases effectively. To tackle this issue, we present a novel approach that operates at the embedding level of an NLP model, independent of the specific architecture. Our method leverages insights from recent advances in XAI techniques and employs an embedding transformation to eliminate implicit information from a selected variable. By directly manipulating the embeddings in the final layer, our approach enables a seamless integration into existing models without requiring significant modifications or retraining. In evaluation, we show that the proposed post-hoc approach significantly reduces gender-related associati
    

