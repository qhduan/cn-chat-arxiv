# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^2] | [Field-based Molecule Generation](https://arxiv.org/abs/2402.15864) | 介绍了一种基于场的模型用于生成拟药分子，相比基于点云的方法具有灵活性和竞争性，能够解决对映异构体问题，进而考虑所有分子几何方面。 |
| [^3] | [Foundational theories of hesitant fuzzy sets and hesitant fuzzy information systems and their applications for multi-strength intelligent classifiers](https://arxiv.org/abs/2311.04256) | 本文提出了基于犹豫模糊集的多种包含关系定义、犹豫模糊信息系统的基础命题和基于多强度智能分类器的健康状态诊断方法。 |
| [^4] | [Deep Huber quantile regression networks.](http://arxiv.org/abs/2306.10306) | DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。 |
| [^5] | [Approximation-Generalization Trade-offs under (Approximate) Group Equivariance.](http://arxiv.org/abs/2305.17592) | 本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。 |
| [^6] | [Cooperation Is All You Need.](http://arxiv.org/abs/2305.10449) | 引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。 |
| [^7] | [Stochastic noise can be helpful for variational quantum algorithms.](http://arxiv.org/abs/2210.06723) | 本文证明随机性可以自然地避免变分量子算法中的严格鞍点问题，这一认识可以帮助我们更好地理解近期变分量子算法的概念。 |
| [^8] | [Energy-Latency Attacks via Sponge Poisoning.](http://arxiv.org/abs/2203.08147) | 本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。 |

# 详细

[^1]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^2]: 基于场的分子生成

    Field-based Molecule Generation

    [https://arxiv.org/abs/2402.15864](https://arxiv.org/abs/2402.15864)

    介绍了一种基于场的模型用于生成拟药分子，相比基于点云的方法具有灵活性和竞争性，能够解决对映异构体问题，进而考虑所有分子几何方面。

    

    这项工作介绍了FMG，一种用于生成拟药分子的基于场的模型。我们展示了这种方法的灵活性如何相比普遍使用的基于点云的方法提供了重要优势，并实现了有竞争力的分子稳定性生成。我们解决了光学异构体（对映异构体），这是一个先前被忽略的对于药物安全性和有效性至关重要的分子属性，并因此考虑了所有分子几何方面。我们展示了先前的方法是对一组变换不变的，其中包括对映异构体成对存在，导致它们对分子的R和S构型保持不变，而我们的基于场的生成模型捕捉了这一属性。

    arXiv:2402.15864v1 Announce Type: new  Abstract: This work introduces FMG, a field-based model for drug-like molecule generation. We show how the flexibility of this method provides crucial advantages over the prevalent, point-cloud based methods, and achieves competitive molecular stability generation. We tackle optical isomerism (enantiomers), a previously omitted molecular property that is crucial for drug safety and effectiveness, and thus account for all molecular geometry aspects. We demonstrate how previous methods are invariant to a group of transformations that includes enantiomer pairs, leading them invariant to the molecular R and S configurations, while our field-based generative model captures this property.
    
[^3]: 犹豫模糊集及其应用于多强度智能分类器的基础理论

    Foundational theories of hesitant fuzzy sets and hesitant fuzzy information systems and their applications for multi-strength intelligent classifiers

    [https://arxiv.org/abs/2311.04256](https://arxiv.org/abs/2311.04256)

    本文提出了基于犹豫模糊集的多种包含关系定义、犹豫模糊信息系统的基础命题和基于多强度智能分类器的健康状态诊断方法。

    

    犹豫模糊集在某些不确定和犹豫的情况下被广泛使用。在集合中，包含关系是一个重要且基础的定义。因此，作为一种集合，犹豫模糊集需要一个明确的包含关系定义。基于离散形式的犹豫模糊隶属度，本文提出了几种适用于犹豫模糊集的包含关系。随后，介绍了一些犹豫模糊集的基础命题，以及犹豫模糊集族的命题。针对参数减少，提出了犹豫模糊信息系统的一些基础命题，并给出了一个示例和算法来说明参数减少的过程。最后，提出了一种多强度智能分类器，用于对复杂系统进行健康状态诊断。

    arXiv:2311.04256v3 Announce Type: replace  Abstract: Hesitant fuzzy sets are widely used in certain instances of uncertainty and hesitation. In sets, the inclusion relationship is an important and foundational definition. Thus, as a kind of set, hesitant fuzzy sets require an explicit definition of inclusion relationship. Based on the hesitant fuzzy membership degree of discrete form, several kinds of inclusion relationships for hesitant fuzzy sets are proposed in this work. Then, some foundational propositions of hesitant fuzzy sets are presented, along with propositions of families of hesitant fuzzy sets. Some foundational propositions of hesitant fuzzy information systems are proposed with respect to parameter reductions and an example and an algorithm are given to illustrate the processes of parameter reduction. Finally, a multi-strength intelligent classifier is proposed to make health state diagnoses for complex systems.
    
[^4]: 深度Huber分位数回归网络

    Deep Huber quantile regression networks. (arXiv:2306.10306v1 [stat.ML])

    [http://arxiv.org/abs/2306.10306](http://arxiv.org/abs/2306.10306)

    DHQRN可以预测更一般的Huber分位数，并且在预测分布的尾部提供更好的预测。

    

    典型的机器学习回归应用旨在通过使用平方误差或绝对误差评分函数来报告预测概率分布的均值或中位数。发出更多预测概率分布的函数（分位数和期望值）的重要性已被认为是量化预测不确定性的手段。在深度学习（DL）应用程序中，通过分位数和期望值回归神经网络（QRNN和ERNN）可以实现这一点。在这里，我们介绍了深度Huber分位数回归网络（DHQRN），它将QRNN和ERNN嵌套为边缘情况。 DHQRN可以预测Huber分位数，这是更一般的函数，因为它们将分位数和期望值作为极限情况嵌套起来。主要思想是使用Huber分位数回归函数训练深度学习算法，这与Huber分位数功能一致。作为概念验证，DHQRN被应用于预测房价的真实数据集，并与其他回归技术进行比较。我们观察到，在几个误差指标中，DHQRN胜过其他技术，在预测分布的尾部提供更好的预测。

    Typical machine learning regression applications aim to report the mean or the median of the predictive probability distribution, via training with a squared or an absolute error scoring function. The importance of issuing predictions of more functionals of the predictive probability distribution (quantiles and expectiles) has been recognized as a means to quantify the uncertainty of the prediction. In deep learning (DL) applications, that is possible through quantile and expectile regression neural networks (QRNN and ERNN respectively). Here we introduce deep Huber quantile regression networks (DHQRN) that nest QRNNs and ERNNs as edge cases. DHQRN can predict Huber quantiles, which are more general functionals in the sense that they nest quantiles and expectiles as limiting cases. The main idea is to train a deep learning algorithm with the Huber quantile regression function, which is consistent for the Huber quantile functional. As a proof of concept, DHQRN are applied to predict hou
    
[^5]: (近似)群等变性下的逼近-泛化权衡

    Approximation-Generalization Trade-offs under (Approximate) Group Equivariance. (arXiv:2305.17592v1 [cs.LG])

    [http://arxiv.org/abs/2305.17592](http://arxiv.org/abs/2305.17592)

    本论文详细研究了通过对称性明确地引入任务特定的归纳偏差所导致的逼近-泛化权衡，并且证明了这种模型在捕获任务特定对称性的同时会改进泛化。这一结果对于提高机器学习领域的性能具有非常大的帮助。

    

    通过对称性明确地引入任务特定的归纳偏差已成为高性能机器学习模型开发中的常规设计准则。例如，群等变神经网络在蛋白质和药物设计等各个领域和应用中展现了卓越的性能。这种模型的普遍感觉是，将相关对称性整合到模型中会增强泛化能力。此外，有人认为，当数据和/或模型只能表现出$\textit{近似}$或$\textit{部分}$对称性时，最优或最好性能的模型是一个模型对齐于数据对称性的模型。在本文中，我们对这些直觉进行了正式的统一研究。首先，我们提出一般的数量界限，证明捕获任务特定对称性的模型将导致改进的泛化。事实上，我们的结果不要求变换是有限的，甚至不需要形成完整的....

    The explicit incorporation of task-specific inductive biases through symmetry has emerged as a general design precept in the development of high-performance machine learning models. For example, group equivariant neural networks have demonstrated impressive performance across various domains and applications such as protein and drug design. A prevalent intuition about such models is that the integration of relevant symmetry results in enhanced generalization. Moreover, it is posited that when the data and/or the model may only exhibit $\textit{approximate}$ or $\textit{partial}$ symmetry, the optimal or best-performing model is one where the model symmetry aligns with the data symmetry. In this paper, we conduct a formal unified investigation of these intuitions. To begin, we present general quantitative bounds that demonstrate how models capturing task-specific symmetries lead to improved generalization. In fact, our results do not require the transformations to be finite or even form
    
[^6]: 合作是你所需要的。 （arXiv:2305.10449v1 [cs.LG]）

    Cooperation Is All You Need. (arXiv:2305.10449v1 [cs.LG])

    [http://arxiv.org/abs/2305.10449](http://arxiv.org/abs/2305.10449)

    引入了一种基于“本地处理器民主”的算法Cooperator，该算法在强化学习中表现比Transformer算法更好。

    

    在超越“树突民主”之上，我们引入了一个名为Cooperator的“本地处理器民主”。在这里，我们将它们与基于Transformers的机器学习算法（例如ChatGPT）在置换不变神经网络强化学习（RL）中的功能进行比较。 Transformers基于长期以来的“积分-发射”“点”神经元的概念，而Cooperator则受到最近神经生物学突破的启示，这些突破表明，精神生活的细胞基础取决于新皮层中具有两个功能上不同点的上皮神经元。我们表明，当用于RL时，基于Cooperator的算法学习速度比基于Transformer的算法快得多，即使它们具有相同数量的参数。

    Going beyond 'dendritic democracy', we introduce a 'democracy of local processors', termed Cooperator. Here we compare their capabilities when used in permutation-invariant neural networks for reinforcement learning (RL), with machine learning algorithms based on Transformers, such as ChatGPT. Transformers are based on the long-standing conception of integrate-and-fire 'point' neurons, whereas Cooperator is inspired by recent neurobiological breakthroughs suggesting that the cellular foundations of mental life depend on context-sensitive pyramidal neurons in the neocortex which have two functionally distinct points. We show that when used for RL, an algorithm based on Cooperator learns far quicker than that based on Transformer, even while having the same number of parameters.
    
[^7]: 随机噪声对变分量子算法有帮助。

    Stochastic noise can be helpful for variational quantum algorithms. (arXiv:2210.06723v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2210.06723](http://arxiv.org/abs/2210.06723)

    本文证明随机性可以自然地避免变分量子算法中的严格鞍点问题，这一认识可以帮助我们更好地理解近期变分量子算法的概念。

    

    鞍点是对于一阶梯度下降算法的一个重要挑战。在经典机器学习的概念中，可以通过随机梯度下降方法避免鞍点。本文提出了证据表明，可以通过利用随机性来自然地避免变分量子算法中的鞍点问题。我们证明了收敛保证，并在数值模拟和量子硬件上提供了实际的例子。我们认为，变分算法的自然随机性可以有助于避免严格的鞍点，即至少具有一个负Hessian特征值的鞍点。这个见解表明一定程度的随机噪声可以帮助我们更好地理解近期变分量子算法的概念。

    Saddle points constitute a crucial challenge for first-order gradient descent algorithms. In notions of classical machine learning, they are avoided for example by means of stochastic gradient descent methods. In this work, we provide evidence that the saddle points problem can be naturally avoided in variational quantum algorithms by exploiting the presence of stochasticity. We prove convergence guarantees and present practical examples in numerical simulations and on quantum hardware. We argue that the natural stochasticity of variational algorithms can be beneficial for avoiding strict saddle points, i.e., those saddle points with at least one negative Hessian eigenvalue. This insight that some levels of shot noise could help is expected to add a new perspective to notions of near-term variational quantum algorithms.
    
[^8]: 基于海绵毒化的能耗延迟攻击。

    Energy-Latency Attacks via Sponge Poisoning. (arXiv:2203.08147v4 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2203.08147](http://arxiv.org/abs/2203.08147)

    本文探讨了一种名为“海绵毒化”的攻击方法，首次证明了在训练时注入海绵样本可以在测试时提高机器学习模型在每个输入上的能耗和延迟，并且即使攻击者只控制了一些模型更新也可以进行此攻击，海绵毒化几乎完全消除了硬件加速器的效果。

    

    海绵样本是在测试时精心优化的输入，可在硬件加速器上部署时增加神经网络的能量消耗和延迟。本文首次证明了海绵样本也可通过一种名为海绵毒化的攻击注入到训练中。该攻击允许在每个测试时输入中不加区分地提高机器学习模型的能量消耗和延迟。我们提出了一种新的海绵毒化形式化方法，克服了与优化测试时海绵样本相关的限制，并表明即使攻击者仅控制几个模型更新，例如模型训练被外包给不受信任的第三方或通过联邦学习分布式进行，也可以进行这种攻击。我们进行了广泛的实验分析，表明海绵毒化几乎完全消除了硬件加速器的效果。同时，我们还分析了毒化模型的激活，确定了哪些计算对导致能量消耗和延迟增加起重要作用。

    Sponge examples are test-time inputs carefully optimized to increase energy consumption and latency of neural networks when deployed on hardware accelerators. In this work, we are the first to demonstrate that sponge examples can also be injected at training time, via an attack that we call sponge poisoning. This attack allows one to increase the energy consumption and latency of machine-learning models indiscriminately on each test-time input. We present a novel formalization for sponge poisoning, overcoming the limitations related to the optimization of test-time sponge examples, and show that this attack is possible even if the attacker only controls a few model updates; for instance, if model training is outsourced to an untrusted third-party or distributed via federated learning. Our extensive experimental analysis shows that sponge poisoning can almost completely vanish the effect of hardware accelerators. We also analyze the activations of poisoned models, identifying which comp
    

