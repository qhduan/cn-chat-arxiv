# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance](https://arxiv.org/abs/2403.05842) | Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。 |
| [^2] | [IR2: Information Regularization for Information Retrieval](https://arxiv.org/abs/2402.16200) | 介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。 |
| [^3] | [Active teacher selection for reinforcement learning from human feedback.](http://arxiv.org/abs/2310.15288) | 本论文提出了一个用于强化学习中的主动教师选择模型以解决多教师的学习问题，研究表明该模型在论文推荐系统和COVID-19疫苗测试领域具有优越性能，并揭示了利用教师间差异来学习准确奖励模型的重要性。 |
| [^4] | [E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty.](http://arxiv.org/abs/2210.13455) | 本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。 |

# 详细

[^1]: Hufu：一种通过置换等变性对预训练的Transformer进行水印处理的模态不可知水印系统

    Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance

    [https://arxiv.org/abs/2403.05842](https://arxiv.org/abs/2403.05842)

    Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。

    

    随着深度学习模型和服务的蓬勃发展，保护宝贵的模型参数免受盗窃已成为一项迫切关注的问题。水印技术被认为是所有权验证的重要工具。然而，当前的水印方案针对不同的模型和任务定制，难以作为集成的知识产权保护服务。我们提出了Hufu，这是一种针对预训练的基于Transformer的模型的模态不可知水印系统，依赖于Transformer的置换等变性质。Hufu通过微调预训练模型在特定置换的一组数据样本上嵌入水印，嵌入的模型基本上包含两组权重 -- 一组用于正常使用，另一组用于水印提取，触发条件是经过置换的输入。置换等变性确保这两组模型权重之间的最小干扰，从而在水印提取时具有高保真度。

    arXiv:2403.05842v1 Announce Type: cross  Abstract: With the blossom of deep learning models and services, it has become an imperative concern to safeguard the valuable model parameters from being stolen. Watermarking is considered an important tool for ownership verification. However, current watermarking schemes are customized for different models and tasks, hard to be integrated as an integrated intellectual protection service. We propose Hufu, a modality-agnostic watermarking system for pre-trained Transformer-based models, relying on the permutation equivariance property of Transformers. Hufu embeds watermark by fine-tuning the pre-trained model on a set of data samples specifically permuted, and the embedded model essentially contains two sets of weights -- one for normal use and the other for watermark extraction which is triggered on permuted inputs. The permutation equivariance ensures minimal interference between these two sets of model weights and thus high fidelity on downst
    
[^2]: IR2：信息正则化用于信息检索

    IR2: Information Regularization for Information Retrieval

    [https://arxiv.org/abs/2402.16200](https://arxiv.org/abs/2402.16200)

    介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。

    

    有效地在训练数据有限的情况下进行信息检索（IR），特别是对于复杂查询，仍然是一项具有挑战性的任务。本文介绍了IR2，即信息检索的信息正则化，一种用于在合成数据生成过程中减少过拟合的技术。该方法在具有复杂查询特征的三个最近的IR任务上进行了测试：DORIS-MAE、ArguAna和WhatsThatBook。实验结果表明，我们的正则化技术不仅在所考虑的任务上优于先前的合成查询生成方法，而且还能将成本降低高达50％。此外，本文将不同阶段的三种正则化方法——输入、提示和输出进行了分类和探索，每种方法相对于没有正则化的模型均提供了不同程度的性能改进。

    arXiv:2402.16200v1 Announce Type: cross  Abstract: Effective information retrieval (IR) in settings with limited training data, particularly for complex queries, remains a challenging task. This paper introduces IR2, Information Regularization for Information Retrieval, a technique for reducing overfitting during synthetic data generation. This approach, representing a novel application of regularization techniques in synthetic data creation for IR, is tested on three recent IR tasks characterized by complex queries: DORIS-MAE, ArguAna, and WhatsThatBook. Experimental results indicate that our regularization techniques not only outperform previous synthetic query generation methods on the tasks considered but also reduce cost by up to 50%. Furthermore, this paper categorizes and explores three regularization methods at different stages of the query synthesis pipeline-input, prompt, and output-each offering varying degrees of performance improvement compared to models where no regulariz
    
[^3]: 强化学习中基于人类反馈的主动教师选择

    Active teacher selection for reinforcement learning from human feedback. (arXiv:2310.15288v1 [cs.AI])

    [http://arxiv.org/abs/2310.15288](http://arxiv.org/abs/2310.15288)

    本论文提出了一个用于强化学习中的主动教师选择模型以解决多教师的学习问题，研究表明该模型在论文推荐系统和COVID-19疫苗测试领域具有优越性能，并揭示了利用教师间差异来学习准确奖励模型的重要性。

    

    从人类反馈中进行强化学习（RLHF）使得机器学习系统能够从人类反馈中学习目标。这些系统的一个核心限制是它们假设所有反馈都来自一个单一的人类教师，尽管需要询问不同教师的意见。我们提出了"Hidden Utility Bandit"（HUB）框架来建模教师在理性、专业知识和成本方面的差异，从而形式化了从多个教师学习的问题。我们开发了多种解决算法，并将它们应用于两个现实世界的领域：论文推荐系统和COVID-19疫苗测试。我们发现，"Active Teacher Selection"（ATS）算法通过主动选择何时以及选择哪个教师来查询，优于基准算法。HUB框架和ATS算法展示了利用教师之间的差异来学习准确的奖励模型的重要性，为鲁棒奖励建模的主动教师选择的未来研究提供了基础。

    Reinforcement learning from human feedback (RLHF) enables machine learning systems to learn objectives from human feedback. A core limitation of these systems is their assumption that all feedback comes from a single human teacher, despite querying a range of distinct teachers. We propose the Hidden Utility Bandit (HUB) framework to model differences in teacher rationality, expertise, and costliness, formalizing the problem of learning from multiple teachers. We develop a variety of solution algorithms and apply them to two real-world domains: paper recommendation systems and COVID-19 vaccine testing. We find that the Active Teacher Selection (ATS) algorithm outperforms baseline algorithms by actively selecting when and which teacher to query. The HUB framework and ATS algorithm demonstrate the importance of leveraging differences between teachers to learn accurate reward models, facilitating future research on active teacher selection for robust reward modeling.
    
[^4]: E-MCTS：通过规划表观不确定性进行深度探索的模型基强化学习

    E-MCTS: Deep Exploration in Model-Based Reinforcement Learning by Planning with Epistemic Uncertainty. (arXiv:2210.13455v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13455](http://arxiv.org/abs/2210.13455)

    本文提出了一种新的方法E-MCTS，通过在MCTS预测中应用表观不确定性估计，实现了模型基强化学习中的深度探索，以及规划探索策略。通过实验证明这种方法在成功的表观不确定性估计和深度探索方面表现优异。

    

    模拟退火树搜索（MCTS）是模型基强化学习中应用最广泛、性能最优秀的规划方法之一。MCTS的关键挑战在于深度探索和面对未知时的可靠性，这两个挑战可以通过在MCTS预测中使用原则性的表观不确定性估计来缓解。本文提出了两个主要贡献：首先，我们开发了一种在MCTS中传播表观不确定性的方法，使智能体能够估计其预测的表观不确定性。其次，我们利用传播的不确定性提出了一种新的深度探索算法，通过明确规划探索策略。我们将这种方法应用于基于MCTS的模型基强化学习方法中，包括使用学习和提供的模型，通过实验证明了我们的方法实现了成功的表观不确定性估计并进行了深度探索。我们将其与基于非规划的深度探索基线进行了比较，并表明...

    One of the most well-studied and highly performing planning approaches used in Model-Based Reinforcement Learning (MBRL) is Monte-Carlo Tree Search (MCTS). Key challenges of MCTS-based MBRL methods remain dedicated deep exploration and reliability in the face of the unknown, and both challenges can be alleviated through principled epistemic uncertainty estimation in the predictions of MCTS. We present two main contributions: First, we develop methodology to propagate epistemic uncertainty in MCTS, enabling agents to estimate the epistemic uncertainty in their predictions. Second, we utilize the propagated uncertainty for a novel deep exploration algorithm by explicitly planning to explore. We incorporate our approach into variations of MCTS-based MBRL approaches with learned and provided models, and empirically show deep exploration through successful epistemic uncertainty estimation achieved by our approach. We compare to a non-planning-based deep-exploration baseline, and demonstrate
    

