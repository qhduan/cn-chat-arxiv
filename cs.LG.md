# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Contrastive Learning on Multimodal Analysis of Electronic Health Records](https://arxiv.org/abs/2403.14926) | 该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。 |
| [^3] | [Active Learning of Mealy Machines with Timers](https://arxiv.org/abs/2403.02019) | 这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。 |
| [^4] | [Joint Problems in Learning Multiple Dynamical Systems](https://arxiv.org/abs/2311.02181) | 聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差 |
| [^5] | [Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder.](http://arxiv.org/abs/2303.15564) | 本文提出了利用掩码自编码器的盲目防御框架（BDMAE），可以在测试时防御盲目后门攻击，不需要验证数据和模型参数，通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。 |
| [^6] | [Finite Expression Method for Solving High-Dimensional Partial Differential Equations.](http://arxiv.org/abs/2206.10121) | 本文介绍了一种名为有限表达方法（FEX）的新方法，用于在具有有限个解析表达式的函数空间中寻找高维偏微分方程的近似解。通过使用深度强化学习方法将FEX应用于各种高维偏微分方程，可以实现高度准确的求解，并且避免了维度灾难。这种有限解析表达式的近似解还可以提供对真实偏微分方程解的可解释洞察。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 电子健康记录的多模态分析上的对比学习

    Contrastive Learning on Multimodal Analysis of Electronic Health Records

    [https://arxiv.org/abs/2403.14926](https://arxiv.org/abs/2403.14926)

    该论文研究了电子健康记录的多模态分析，强调了结构化和非结构化数据之间的协同作用，并尝试将多模态对比学习方法应用于提高患者医疗历史的完整性。

    

    电子健康记录（EHR）系统包含大量的多模态临床数据，包括结构化数据如临床编码和非结构化数据如临床笔记。然而，许多现有的针对EHR的研究传统上要么集中于个别模态，要么以一种相当粗糙的方式合并不同的模态。这种方法通常会导致将结构化和非结构化数据视为单独实体，忽略它们之间固有的协同作用。具体来说，这两个重要的模态包含临床相关、密切相关和互补的健康信息。通过联合分析这两种数据模态可以捕捉到患者医疗历史的更完整画面。尽管多模态对比学习在视觉语言领域取得了巨大成功，但在多模态EHR领域，尤其是在理论理解方面，其潜力仍未充分挖掘。

    arXiv:2403.14926v1 Announce Type: cross  Abstract: Electronic health record (EHR) systems contain a wealth of multimodal clinical data including structured data like clinical codes and unstructured data such as clinical notes. However, many existing EHR-focused studies has traditionally either concentrated on an individual modality or merged different modalities in a rather rudimentary fashion. This approach often results in the perception of structured and unstructured data as separate entities, neglecting the inherent synergy between them. Specifically, the two important modalities contain clinically relevant, inextricably linked and complementary health information. A more complete picture of a patient's medical history is captured by the joint analysis of the two modalities of data. Despite the great success of multimodal contrastive learning on vision-language, its potential remains under-explored in the realm of multimodal EHR, particularly in terms of its theoretical understandi
    
[^3]: 具有定时器的Mealy机器的主动学习

    Active Learning of Mealy Machines with Timers

    [https://arxiv.org/abs/2403.02019](https://arxiv.org/abs/2403.02019)

    这篇论文提出了一种用于查询学习具有定时器的Mealy机器的算法，在实现上明显比已有算法更有效率。

    

    我们在黑盒环境中提出了第一个用于查询学习一般类别的具有定时器的Mealy机器（MMTs）的算法。我们的算法是Vaandrager等人的L＃算法对定时设置的扩展。类似于Waga提出的用于学习定时自动机的算法，我们的算法受到Maler＆Pnueli思想的启发。我们的算法和Waga的算法都使用符号查询进行基础语言学习，然后使用有限数量的具体查询进行实现。然而，Waga需要指数级的具体查询来实现单个符号查询，而我们只需要多项式数量。这是因为要学习定时自动机，学习者需要确定每个转换的确切卫兵和重置（有指数多种可能性），而要学习MMT，学习者只需要弄清楚哪些先前的转换导致超时。正如我们之前的工作所示，

    arXiv:2403.02019v1 Announce Type: cross  Abstract: We present the first algorithm for query learning of a general class of Mealy machines with timers (MMTs) in a black-box context. Our algorithm is an extension of the L# algorithm of Vaandrager et al. to a timed setting. Like the algorithm for learning timed automata proposed by Waga, our algorithm is inspired by ideas of Maler & Pnueli. Based on the elementary languages of, both Waga's and our algorithm use symbolic queries, which are then implemented using finitely many concrete queries. However, whereas Waga needs exponentially many concrete queries to implement a single symbolic query, we only need a polynomial number. This is because in order to learn a timed automaton, a learner needs to determine the exact guard and reset for each transition (out of exponentially many possibilities), whereas for learning an MMT a learner only needs to figure out which of the preceding transitions caused a timeout. As shown in our previous work, 
    
[^4]: 学习多个动态系统中的联合问题

    Joint Problems in Learning Multiple Dynamical Systems

    [https://arxiv.org/abs/2311.02181](https://arxiv.org/abs/2311.02181)

    聚类时间序列的新问题，提出联合划分轨迹集并学习每个部分的线性动态系统模型，以最小化所有模型的最大误差

    

    时间序列的聚类是一个经过充分研究的问题，其应用范围从通过代谢产物浓度获得的定量个性化代谢模型到量子信息理论中的状态判别。我们考虑了一个变种，即给定一组轨迹和一些部分，我们联合划分轨迹集并学习每个部分的线性动态系统（LDS）模型，以使得所有模型的最大误差最小化。我们提出了全局收敛的方法和EM启发式算法，并附上了有前景的计算结果。

    arXiv:2311.02181v2 Announce Type: replace-cross  Abstract: Clustering of time series is a well-studied problem, with applications ranging from quantitative, personalized models of metabolism obtained from metabolite concentrations to state discrimination in quantum information theory. We consider a variant, where given a set of trajectories and a number of parts, we jointly partition the set of trajectories and learn linear dynamical system (LDS) models for each part, so as to minimize the maximum error across all the models. We present globally convergent methods and EM heuristics, accompanied by promising computational results.
    
[^5]: 掩码还原技术：利用掩码自编码器在测试时防御盲目后门攻击

    Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder. (arXiv:2303.15564v1 [cs.LG])

    [http://arxiv.org/abs/2303.15564](http://arxiv.org/abs/2303.15564)

    本文提出了利用掩码自编码器的盲目防御框架（BDMAE），可以在测试时防御盲目后门攻击，不需要验证数据和模型参数，通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。

    

    深度神经网络容易受到恶意攻击，攻击者会通过在图像上叠加特殊的触发器来恶意操纵模型行为，这称为后门攻击。现有的后门防御方法通常需要访问一些验证数据和模型参数，这在许多实际应用中是不切实际的，例如当模型作为云服务提供时。为了解决这个问题，本文致力于测试时的盲目后门防御实践，特别是针对黑盒模型。每个测试图像的真实标签需要从可疑模型的硬标签预测中恢复。然而，在图像空间中启发式触发器搜索不适用于复杂触发器或高分辨率的图片。我们通过利用通用图像生成模型，提出了一种利用掩码自编码器的盲目防御框架（BDMAE），通过测试图像和 MAE 还原之间的结构相似性和标签一致性来检测后门攻击。

    Deep neural networks are vulnerable to backdoor attacks, where an adversary maliciously manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which are impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for black-box models. The true label of every test image needs to be recovered on the fly from the hard label predictions of a suspicious model. The heuristic trigger search in image space, however, is not scalable to complex triggers or high image resolution. We circumvent such barrier by leveraging generic image generation models, and propose a framework of Blind Defense with Masked AutoEncoder (BDMAE). It uses the image structural similarity and label consistency between the test image and MAE restorations to detec
    
[^6]: 用于求解高维偏微分方程的有限表达方法

    Finite Expression Method for Solving High-Dimensional Partial Differential Equations. (arXiv:2206.10121v3 [math.NA] UPDATED)

    [http://arxiv.org/abs/2206.10121](http://arxiv.org/abs/2206.10121)

    本文介绍了一种名为有限表达方法（FEX）的新方法，用于在具有有限个解析表达式的函数空间中寻找高维偏微分方程的近似解。通过使用深度强化学习方法将FEX应用于各种高维偏微分方程，可以实现高度准确的求解，并且避免了维度灾难。这种有限解析表达式的近似解还可以提供对真实偏微分方程解的可解释洞察。

    

    设计高效准确的高维偏微分方程数值求解器仍然是计算科学和工程中一个具有挑战性和重要性的课题，主要是由于在设计能够按维数进行扩展的数值方案中存在“维度灾难”。本文介绍了一种新的方法，该方法在具有有限个解析表达式的函数空间中寻找近似的偏微分方程解，因此将该方法命名为有限表达方法（FEX）。在近似理论中证明了FEX可以避免维度灾难。作为概念证明，本文提出了一种深度强化学习方法，用于在不同维度上实现FEX求解各种高维偏微分方程，实现了高甚至机器精度，并具有多项式维度的记忆复杂度和可操作的时间复杂度。具有有限解析表达式的近似解还为地面真实偏微分方程解提供了可解释的洞察。

    Designing efficient and accurate numerical solvers for high-dimensional partial differential equations (PDEs) remains a challenging and important topic in computational science and engineering, mainly due to the "curse of dimensionality" in designing numerical schemes that scale in dimension. This paper introduces a new methodology that seeks an approximate PDE solution in the space of functions with finitely many analytic expressions and, hence, this methodology is named the finite expression method (FEX). It is proved in approximation theory that FEX can avoid the curse of dimensionality. As a proof of concept, a deep reinforcement learning method is proposed to implement FEX for various high-dimensional PDEs in different dimensions, achieving high and even machine accuracy with a memory complexity polynomial in dimension and an amenable time complexity. An approximate solution with finite analytic expressions also provides interpretable insights into the ground truth PDE solution, w
    

