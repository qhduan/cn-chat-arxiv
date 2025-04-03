# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IR2: Information Regularization for Information Retrieval](https://arxiv.org/abs/2402.16200) | 介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。 |
| [^2] | [Identifiable Latent Causal Content for Domain Adaptation under Latent Covariate Shift](https://arxiv.org/abs/2208.14161) | 提出了一种新的隐含协变量转移（LCS）范式，增加了领域间的可变性和适应性，并提供了恢复标签变量潜在原因的理论保证。 |
| [^3] | [Active teacher selection for reinforcement learning from human feedback.](http://arxiv.org/abs/2310.15288) | 本论文提出了一个用于强化学习中的主动教师选择模型以解决多教师的学习问题，研究表明该模型在论文推荐系统和COVID-19疫苗测试领域具有优越性能，并揭示了利用教师间差异来学习准确奖励模型的重要性。 |
| [^4] | [Limits to Reservoir Learning.](http://arxiv.org/abs/2307.14474) | 这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。 |
| [^5] | [Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks.](http://arxiv.org/abs/2305.06986) | 本文研究了三层神经网络的特征学习能力，相比之下，它具有比两层网络更丰富的可证的特征学习能力，并提出了一个通用定理，限制了目标结构的样本复杂度和宽度，以实现低测试误差。 |
| [^6] | [E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty.](http://arxiv.org/abs/2210.13455) | 本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。 |

# 详细

[^1]: IR2：信息正则化用于信息检索

    IR2: Information Regularization for Information Retrieval

    [https://arxiv.org/abs/2402.16200](https://arxiv.org/abs/2402.16200)

    介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。

    

    有效地在训练数据有限的情况下进行信息检索（IR），特别是对于复杂查询，仍然是一项具有挑战性的任务。本文介绍了IR2，即信息检索的信息正则化，一种用于在合成数据生成过程中减少过拟合的技术。该方法在具有复杂查询特征的三个最近的IR任务上进行了测试：DORIS-MAE、ArguAna和WhatsThatBook。实验结果表明，我们的正则化技术不仅在所考虑的任务上优于先前的合成查询生成方法，而且还能将成本降低高达50％。此外，本文将不同阶段的三种正则化方法——输入、提示和输出进行了分类和探索，每种方法相对于没有正则化的模型均提供了不同程度的性能改进。

    arXiv:2402.16200v1 Announce Type: cross  Abstract: Effective information retrieval (IR) in settings with limited training data, particularly for complex queries, remains a challenging task. This paper introduces IR2, Information Regularization for Information Retrieval, a technique for reducing overfitting during synthetic data generation. This approach, representing a novel application of regularization techniques in synthetic data creation for IR, is tested on three recent IR tasks characterized by complex queries: DORIS-MAE, ArguAna, and WhatsThatBook. Experimental results indicate that our regularization techniques not only outperform previous synthetic query generation methods on the tasks considered but also reduce cost by up to 50%. Furthermore, this paper categorizes and explores three regularization methods at different stages of the query synthesis pipeline-input, prompt, and output-each offering varying degrees of performance improvement compared to models where no regulariz
    
[^2]: 可识别的潜在因果内容用于隐含协变量转移下的领域自适应

    Identifiable Latent Causal Content for Domain Adaptation under Latent Covariate Shift

    [https://arxiv.org/abs/2208.14161](https://arxiv.org/abs/2208.14161)

    提出了一种新的隐含协变量转移（LCS）范式，增加了领域间的可变性和适应性，并提供了恢复标签变量潜在原因的理论保证。

    

    多源领域自适应（MSDA）解决了利用来自多个源域的标记数据和来自目标域的未标记数据来学习针对未标记目标领域的标签预测函数的挑战。我们提出了一种称为潜在协变量转移（LCS）的新范式，它引入了更大的领域间可变性和适应性。值得注意的是，它为恢复标签变量的潜在原因提供了理论保证。

    arXiv:2208.14161v3 Announce Type: replace  Abstract: Multi-source domain adaptation (MSDA) addresses the challenge of learning a label prediction function for an unlabeled target domain by leveraging both the labeled data from multiple source domains and the unlabeled data from the target domain. Conventional MSDA approaches often rely on covariate shift or conditional shift paradigms, which assume a consistent label distribution across domains. However, this assumption proves limiting in practical scenarios where label distributions do vary across domains, diminishing its applicability in real-world settings. For example, animals from different regions exhibit diverse characteristics due to varying diets and genetics.   Motivated by this, we propose a novel paradigm called latent covariate shift (LCS), which introduces significantly greater variability and adaptability across domains. Notably, it provides a theoretical assurance for recovering the latent cause of the label variable, w
    
[^3]: 强化学习中基于人类反馈的主动教师选择

    Active teacher selection for reinforcement learning from human feedback. (arXiv:2310.15288v1 [cs.AI])

    [http://arxiv.org/abs/2310.15288](http://arxiv.org/abs/2310.15288)

    本论文提出了一个用于强化学习中的主动教师选择模型以解决多教师的学习问题，研究表明该模型在论文推荐系统和COVID-19疫苗测试领域具有优越性能，并揭示了利用教师间差异来学习准确奖励模型的重要性。

    

    从人类反馈中进行强化学习（RLHF）使得机器学习系统能够从人类反馈中学习目标。这些系统的一个核心限制是它们假设所有反馈都来自一个单一的人类教师，尽管需要询问不同教师的意见。我们提出了"Hidden Utility Bandit"（HUB）框架来建模教师在理性、专业知识和成本方面的差异，从而形式化了从多个教师学习的问题。我们开发了多种解决算法，并将它们应用于两个现实世界的领域：论文推荐系统和COVID-19疫苗测试。我们发现，"Active Teacher Selection"（ATS）算法通过主动选择何时以及选择哪个教师来查询，优于基准算法。HUB框架和ATS算法展示了利用教师之间的差异来学习准确的奖励模型的重要性，为鲁棒奖励建模的主动教师选择的未来研究提供了基础。

    Reinforcement learning from human feedback (RLHF) enables machine learning systems to learn objectives from human feedback. A core limitation of these systems is their assumption that all feedback comes from a single human teacher, despite querying a range of distinct teachers. We propose the Hidden Utility Bandit (HUB) framework to model differences in teacher rationality, expertise, and costliness, formalizing the problem of learning from multiple teachers. We develop a variety of solution algorithms and apply them to two real-world domains: paper recommendation systems and COVID-19 vaccine testing. We find that the Active Teacher Selection (ATS) algorithm outperforms baseline algorithms by actively selecting when and which teacher to query. The HUB framework and ATS algorithm demonstrate the importance of leveraging differences between teachers to learn accurate reward models, facilitating future research on active teacher selection for robust reward modeling.
    
[^4]: 河川学习的限制。

    Limits to Reservoir Learning. (arXiv:2307.14474v1 [cs.LG])

    [http://arxiv.org/abs/2307.14474](http://arxiv.org/abs/2307.14474)

    这项工作限制了机器学习的能力，基于物理学所暗示的计算限制。储水库计算机在噪声下的性能下降意味着需要指数数量的样本来学习函数族，并讨论了没有噪声时的性能。

    

    在这项工作中，我们根据物理学所暗示的计算限制来限制机器学习的能力。我们首先考虑信息处理能力（IPC），这是一个对信号集合到完整函数基的期望平方误差进行归一化的指标。我们使用IPC来衡量噪声下储水库计算机（一种特殊的循环网络）的性能降低。首先，我们证明IPC在系统尺寸n上是一个多项式，即使考虑到n个输出信号的$2^n$个可能的逐点乘积。接下来，我们认为这种退化意味着在储水库噪声存在的情况下，储水库所表示的函数族需要指数数量的样本来进行学习。最后，我们讨论了在没有噪声的情况下，同一集合的$2^n$个函数在进行二元分类时的性能。

    In this work, we bound a machine's ability to learn based on computational limitations implied by physicality. We start by considering the information processing capacity (IPC), a normalized measure of the expected squared error of a collection of signals to a complete basis of functions. We use the IPC to measure the degradation under noise of the performance of reservoir computers, a particular kind of recurrent network, when constrained by physical considerations. First, we show that the IPC is at most a polynomial in the system size $n$, even when considering the collection of $2^n$ possible pointwise products of the $n$ output signals. Next, we argue that this degradation implies that the family of functions represented by the reservoir requires an exponential number of samples to learn in the presence of the reservoir's noise. Finally, we conclude with a discussion of the performance of the same collection of $2^n$ functions without noise when being used for binary classification
    
[^5]: 三层神经网络中非线性特征学习的可证保证

    Provable Guarantees for Nonlinear Feature Learning in Three-Layer Neural Networks. (arXiv:2305.06986v1 [cs.LG])

    [http://arxiv.org/abs/2305.06986](http://arxiv.org/abs/2305.06986)

    本文研究了三层神经网络的特征学习能力，相比之下，它具有比两层网络更丰富的可证的特征学习能力，并提出了一个通用定理，限制了目标结构的样本复杂度和宽度，以实现低测试误差。

    

    深度学习理论中的一个核心问题是理解神经网络如何学习分层特征。深度网络提取显著特征的能力对其卓越的泛化能力和现代深度学习范式的预训练和微调至关重要。然而，从理论角度来看，这种特征学习过程仍然不够清晰，现有的分析主要局限于两层网络。在本文中，我们展示了三层神经网络具有证明的比两层网络更丰富的特征学习能力。我们分析了通过逐层梯度下降训练的三层网络学习的特征，并提出了一个通用定理，它上界了目标具有特定层次结构时实现低测试错误所需的样本复杂度和宽度。我们将我们的框架实例化到特定的统计学学习设置中——单指数模型和二次函数。

    One of the central questions in the theory of deep learning is to understand how neural networks learn hierarchical features. The ability of deep networks to extract salient features is crucial to both their outstanding generalization ability and the modern deep learning paradigm of pretraining and finetuneing. However, this feature learning process remains poorly understood from a theoretical perspective, with existing analyses largely restricted to two-layer networks. In this work we show that three-layer neural networks have provably richer feature learning capabilities than two-layer networks. We analyze the features learned by a three-layer network trained with layer-wise gradient descent, and present a general purpose theorem which upper bounds the sample complexity and width needed to achieve low test error when the target has specific hierarchical structure. We instantiate our framework in specific statistical learning settings -- single-index models and functions of quadratic 
    
[^6]: E-MCTS：通过规划表观不确定性进行深度探索的模型基强化学习

    E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty. (arXiv:2210.13455v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13455](http://arxiv.org/abs/2210.13455)

    本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。

    

    模拟退火树搜索（MCTS）是模型基强化学习中应用最广泛、性能最优秀的规划方法之一。MCTS的关键挑战在于深度探索和面对未知时的可靠性，这两个挑战可以通过在MCTS预测中使用原则性的表观不确定性估计来缓解。本文提出了两个主要贡献：首先，我们开发了一种在MCTS中传播表观不确定性的方法，使智能体能够估计其预测的表观不确定性。其次，我们利用传播的不确定性提出了一种新的深度探索算法，通过明确规划探索策略。我们将这种方法应用于基于MCTS的模型基强化学习方法中，包括使用学习和提供的模型，通过实验证明了我们的方法实现了成功的表观不确定性估计并进行了深度探索。我们将其与基于非规划的深度探索基线进行了比较，并表明...

    One of the most well-studied and highly performing planning approaches used in Model-Based Reinforcement Learning (MBRL) is Monte-Carlo Tree Search (MCTS). Key challenges of MCTS-based MBRL methods remain dedicated deep exploration and reliability in the face of the unknown, and both challenges can be alleviated through principled epistemic uncertainty estimation in the predictions of MCTS. We present two main contributions: First, we develop methodology to propagate epistemic uncertainty in MCTS, enabling agents to estimate the epistemic uncertainty in their predictions. Second, we utilize the propagated uncertainty for a novel deep exploration algorithm by explicitly planning to explore. We incorporate our approach into variations of MCTS-based MBRL approaches with learned and provided models, and empirically show deep exploration through successful epistemic uncertainty estimation achieved by our approach. We compare to a non-planning-based deep-exploration baseline, and demonstrate
    

