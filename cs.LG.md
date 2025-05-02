# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression](https://arxiv.org/abs/2403.16677) | FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。 |
| [^2] | [Fairness Risks for Group-conditionally Missing Demographics](https://arxiv.org/abs/2402.13393) | 通过概率填充敏感特征，联合学习群体条件性缺失概率，增强一般公平风险，实现准确性和公平性之间的改进平衡 |
| [^3] | [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | 本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。 |
| [^4] | [Omitted Labels in Causality: A Study of Paradoxes](https://arxiv.org/abs/2311.06840) | 本研究探讨了所谓的“遗漏标签上下文”，即训练数据仅限于可能标签的一个子集，并利用悖论展示了在这种上下文中因果推断面临的困难。研究发现在某些情况下，必须使用非可交换的处理组和对照组进行正确的校正。此外，研究还发现了结论网络与社会选择理论之间的有趣联系。 |
| [^5] | [EyePreserve: Identity-Preserving Iris Synthesis.](http://arxiv.org/abs/2312.12028) | 本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。 |
| [^6] | [ALEXR: An Optimal Single-Loop Algorithm for Convex Finite-Sum Coupled Compositional Stochastic Optimization.](http://arxiv.org/abs/2312.02277) | 本文提出了一种名为ALEXR的高效算法，用于解决凸有限和耦合组成随机优化问题。此算法在解决平滑和非平滑问题时具有优越的收敛速度，并且可应用于多个领域，包括组分布鲁棒优化、不平衡数据学习、强化学习和排序学习。 |
| [^7] | [Accelerated First-Order Optimization under Nonlinear Constraints.](http://arxiv.org/abs/2302.00316) | 设计了一种新的加速非线性约束下的一阶优化算法，其优点是避免了在整个可行集上进行优化，而且利用速度来表达约束，使得算法在决策变量数量和约束数量上的复杂度增长适度，适用于机器学习应用。 |
| [^8] | [Introduction to Online Nonstochastic Control.](http://arxiv.org/abs/2211.09619) | 介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。 |

# 详细

[^1]: FOOL: 用神经特征压缩解决卫星计算中的下行瓶颈问题

    FOOL: Addressing the Downlink Bottleneck in Satellite Computing with Neural Feature Compression

    [https://arxiv.org/abs/2403.16677](https://arxiv.org/abs/2403.16677)

    FOOL是一种OEC本地和任务不可知的特征压缩方法，通过最大化吞吐量、嵌入上下文和利用瓷砖间的依赖关系，降低传输成本，同时保持预测性能。

    

    具有传感器的纳卫星星座捕获大范围地理区域，为地球观测提供了前所未有的机会。随着星座规模的增加，网络争用形成了下行瓶颈。轨道边缘计算（OEC）利用有限的机载计算资源通过在源头处理原始捕获来减少传输成本。然而，由于依赖粗糙的过滤方法或过分优先考虑特定下游任务，目前的解决方案具有有限的实用性。本文提出了FOOL，一种OEC本地和任务不可知的特征压缩方法，可保留预测性能。FOOL将高分辨率卫星图像进行分区，以最大化吞吐量。此外，它嵌入上下文并利用瓷砖间的依赖关系，以较低的开销降低传输成本。虽然FOOL是一种特征压缩器，但它可以在低

    arXiv:2403.16677v1 Announce Type: new  Abstract: Nanosatellite constellations equipped with sensors capturing large geographic regions provide unprecedented opportunities for Earth observation. As constellation sizes increase, network contention poses a downlink bottleneck. Orbital Edge Computing (OEC) leverages limited onboard compute resources to reduce transfer costs by processing the raw captures at the source. However, current solutions have limited practicability due to reliance on crude filtering methods or over-prioritizing particular downstream tasks.   This work presents FOOL, an OEC-native and task-agnostic feature compression method that preserves prediction performance. FOOL partitions high-resolution satellite imagery to maximize throughput. Further, it embeds context and leverages inter-tile dependencies to lower transfer costs with negligible overhead. While FOOL is a feature compressor, it can recover images with competitive scores on perceptual quality measures at low
    
[^2]: 针对群体条件性缺失人口统计数据的公平风险

    Fairness Risks for Group-conditionally Missing Demographics

    [https://arxiv.org/abs/2402.13393](https://arxiv.org/abs/2402.13393)

    通过概率填充敏感特征，联合学习群体条件性缺失概率，增强一般公平风险，实现准确性和公平性之间的改进平衡

    

    具有公平意识的分类模型近年来越来越受到关注，因为对某些人口统计群体的歧视问题日益引起担忧。大多数现有模型要求完全了解敏感特征，这可能由于隐私、法律问题和个人对歧视的恐惧而不切实际。我们将解决的关键挑战是不可用性的群体依赖性，例如，某些年龄范围的人可能更不愿透露他们的年龄。我们的解决方案通过对敏感特征进行概率填充，同时在变分自动编码器中联合学习群体条件性缺失的概率，将一般公平风险与之增强。我们的模型在图像和表格数据集上表现出了有效性，实现了准确性和公平性之间的改进平衡。

    arXiv:2402.13393v1 Announce Type: new  Abstract: Fairness-aware classification models have gained increasing attention in recent years as concerns grow on discrimination against some demographic groups. Most existing models require full knowledge of the sensitive features, which can be impractical due to privacy, legal issues, and an individual's fear of discrimination. The key challenge we will address is the group dependency of the unavailability, e.g., people of some age range may be more reluctant to reveal their age. Our solution augments general fairness risks with probabilistic imputations of the sensitive features, while jointly learning the group-conditionally missing probabilities in a variational auto-encoder. Our model is demonstrated effective on both image and tabular datasets, achieving an improved balance between accuracy and fairness.
    
[^3]: Fiddler：用于Mixture-of-Experts模型快速推断的CPU-GPU编排

    Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

    [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)

    本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。

    

    基于Mixture-of-Experts（MoE）架构的大型语言模型（LLM）在各种任务上表现出了很好的性能。然而，在资源受限的环境下运行这些模型，即GPU内存资源不丰富的情况下，由于模型规模庞大，存在挑战。现有的将模型权重卸载到CPU内存的系统，由于频繁地在CPU和GPU之间移动数据而导致显著的开销。在本文中，我们提出了Fiddler，一种用于MoE模型的资源高效推断引擎，实现了CPU-GPU编排。Fiddler的核心思想是利用CPU的计算能力来最小化CPU和GPU之间的数据传输。我们的评估结果表明，Fiddler能够在单个具有24GB内存的GPU上运行未压缩的Mixtral-8x7B模型（参数超过90GB），每秒生成超过3个token，相比现有方法提高一个数量级。Fiddler的代码可以公开访问，网址为\url{https://github.com/efeslab/fiddler}

    Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architecture are showing promising performance on various tasks. However, running them on resource-constrained settings, where GPU memory resources are not abundant, is challenging due to huge model sizes. Existing systems that offload model weights to CPU memory suffer from the significant overhead of frequently moving data between CPU and GPU. In this paper, we propose Fiddler, a resource-efficient inference engine with CPU-GPU orchestration for MoE models. The key idea of Fiddler is to use the computation ability of the CPU to minimize the data movement between the CPU and GPU. Our evaluation shows that Fiddler can run the uncompressed Mixtral-8x7B model, which exceeds 90GB in parameters, to generate over $3$ tokens per second on a single GPU with 24GB memory, showing an order of magnitude improvement over existing methods. The code of Fiddler is publicly available at \url{https://github.com/efeslab/fiddler}
    
[^4]: 在因果推断中的遗漏标签: 一项关于悖论的研究

    Omitted Labels in Causality: A Study of Paradoxes

    [https://arxiv.org/abs/2311.06840](https://arxiv.org/abs/2311.06840)

    本研究探讨了所谓的“遗漏标签上下文”，即训练数据仅限于可能标签的一个子集，并利用悖论展示了在这种上下文中因果推断面临的困难。研究发现在某些情况下，必须使用非可交换的处理组和对照组进行正确的校正。此外，研究还发现了结论网络与社会选择理论之间的有趣联系。

    

    我们探讨了我们所称之为“遗漏标签上下文”的概念，即训练数据仅限于可能标签的一个子集。这种设置在专业人士或特定的专注研究中非常普遍。我们利用已广泛研究的悖论（辛普森悖论和康多塞悖论）来说明在遗漏标签上下文中因果推断面临的更普遍困难。与因果推断基本原理相反，我们展示了“正确”的校正有时需要非可交换的处理组和对照组。这些陷阱引导我们研究不同上下文中得出的结论网络和其形成的结构，从而证明了这些网络与社会选择理论的有趣联系。

    We explore what we call ``omitted label contexts,'' in which training data is limited to a subset of the possible labels. This setting is common among specialized human experts or specific focused studies. We lean on well-studied paradoxes (Simpson's and Condorcet) to illustrate the more general difficulties of causal inference in omitted label contexts. Contrary to the fundamental principles on which much of causal inference is built, we show that ``correct'' adjustments sometimes require non-exchangeable treatment and control groups. These pitfalls lead us to the study networks of conclusions drawn from different contexts and the structures the form, proving an interesting connection between these networks and social choice theory.
    
[^5]: EyePreserve: 保持身份的虹膜合成

    EyePreserve: Identity-Preserving Iris Synthesis. (arXiv:2312.12028v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.12028](http://arxiv.org/abs/2312.12028)

    本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法，能够合成不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在保持身份的同时进行非线性纹理变形。

    

    在广泛的瞳孔尺寸范围内保持身份的同身份生物特征虹膜图像的合成是复杂的，因为它涉及到虹膜肌肉收缩机制，需要将虹膜非线性纹理变形模型嵌入到合成流程中。本论文提出了一种完全数据驱动的、保持身份的、瞳孔尺寸变化的虹膜图像合成方法。这种方法能够合成具有不同瞳孔尺寸的虹膜图像，代表不存在的身份，并能够在给定目标虹膜图像的分割掩膜下非线性地变形现有主体的虹膜图像纹理。虹膜识别实验表明，所提出的变形模型不仅在改变瞳孔尺寸时保持身份，而且在瞳孔尺寸有显著差异的同身份虹膜样本之间提供更好的相似度，与最先进的线性方法相比。

    Synthesis of same-identity biometric iris images, both for existing and non-existing identities while preserving the identity across a wide range of pupil sizes, is complex due to intricate iris muscle constriction mechanism, requiring a precise model of iris non-linear texture deformations to be embedded into the synthesis pipeline. This paper presents the first method of fully data-driven, identity-preserving, pupil size-varying s ynthesis of iris images. This approach is capable of synthesizing images of irises with different pupil sizes representing non-existing identities as well as non-linearly deforming the texture of iris images of existing subjects given the segmentation mask of the target iris image. Iris recognition experiments suggest that the proposed deformation model not only preserves the identity when changing the pupil size but offers better similarity between same-identity iris samples with significant differences in pupil size, compared to state-of-the-art linear an
    
[^6]: ALEXR:一种用于凸有限和耦合组成随机优化的最优单循环算法

    ALEXR: An Optimal Single-Loop Algorithm for Convex Finite-Sum Coupled Compositional Stochastic Optimization. (arXiv:2312.02277v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2312.02277](http://arxiv.org/abs/2312.02277)

    本文提出了一种名为ALEXR的高效算法，用于解决凸有限和耦合组成随机优化问题。此算法在解决平滑和非平滑问题时具有优越的收敛速度，并且可应用于多个领域，包括组分布鲁棒优化、不平衡数据学习、强化学习和排序学习。

    

    本文重新审视了一类具有多个应用的凸有限和耦合组成随机优化（cFCCO）问题，包括组分布鲁棒优化（GDRO），不平衡数据学习，强化学习和排序学习。为了更好地解决这些问题，我们引入了一个高效的单循环原始-对偶块坐标近端算法，称为ALEXR。该算法利用块坐标随机镜像上升更新对偶变量和随机近端梯度下降更新原始变量。我们在平滑和非平滑函数条件下建立了ALEXR在凸和强凸情况下的收敛速度，这不仅改进了以前在平滑cFCCO问题上的最佳速度，还扩展了cFCCO的范围，用于解决更具挑战性的非平滑问题，如GDRO的对偶形式。最后，我们提供了较低的复杂性下界，以证明算法具有很强的效率。

    This paper revisits a class of convex Finite-Sum Coupled Compositional Stochastic Optimization (cFCCO) problems with many applications, including group distributionally robust optimization (GDRO), learning with imbalanced data, reinforcement learning, and learning to rank. To better solve these problems, we introduce an efficient single-loop primal-dual block-coordinate proximal algorithm, dubbed ALEXR. This algorithm leverages block-coordinate stochastic mirror ascent updates for the dual variable and stochastic proximal gradient descent updates for the primal variable. We establish the convergence rates of ALEXR in both convex and strongly convex cases under smoothness and non-smoothness conditions of involved functions, which not only improve the best rates in previous works on smooth cFCCO problems but also expand the realm of cFCCO for solving more challenging non-smooth problems such as the dual form of GDRO. Finally, we present lower complexity bounds to demonstrate that the con
    
[^7]: 加速非线性约束下的一阶优化

    Accelerated First-Order Optimization under Nonlinear Constraints. (arXiv:2302.00316v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2302.00316](http://arxiv.org/abs/2302.00316)

    设计了一种新的加速非线性约束下的一阶优化算法，其优点是避免了在整个可行集上进行优化，而且利用速度来表达约束，使得算法在决策变量数量和约束数量上的复杂度增长适度，适用于机器学习应用。

    

    我们利用约束优化和非光滑动力系统之间的类比，设计了一种新的加速非线性约束下的一阶优化算法。与Frank-Wolfe或投影梯度不同，这些算法避免了每次迭代在整个可行集上进行优化。我们证明了在非凸设置中的收敛性，并推导了在连续时间和离散时间中的凸设置的加速率。这些算法的一个重要特性是使用速度而不是位置来表达约束，这自然地导致可行集的稀疏、局部和凸近似（即使可行集是非凸的）。因此，复杂度在决策变量数量和约束数量上适度增长，使得该算法适用于机器学习应用。我们将算法应用于压缩感知和稀疏···

    We exploit analogies between first-order algorithms for constrained optimization and non-smooth dynamical systems to design a new class of accelerated first-order algorithms for constrained optimization. Unlike Frank-Wolfe or projected gradients, these algorithms avoid optimization over the entire feasible set at each iteration. We prove convergence to stationary points even in a nonconvex setting and we derive accelerated rates for the convex setting both in continuous time, as well as in discrete time. An important property of these algorithms is that constraints are expressed in terms of velocities instead of positions, which naturally leads to sparse, local and convex approximations of the feasible set (even if the feasible set is nonconvex). Thus, the complexity tends to grow mildly in the number of decision variables and in the number of constraints, which makes the algorithms suitable for machine learning applications. We apply our algorithms to a compressed sensing and a sparse
    
[^8]: 在线非随机控制简介

    Introduction to Online Nonstochastic Control. (arXiv:2211.09619v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.09619](http://arxiv.org/abs/2211.09619)

    介绍了一种新兴的在线非随机控制方法，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    

    本文介绍了一种新兴的动态系统控制与可微强化学习范式——在线非随机控制，并应用在线凸优化和凸松弛技术得到了具有可证明保证的新方法，在最佳和鲁棒控制方面取得了显著成果。与其他框架不同，该方法的目标是对抗性攻击，在无法预测扰动模型的情况下，通过在一组策略中寻找低后悔，获得对最优策略的近似。

    This text presents an introduction to an emerging paradigm in control of dynamical systems and differentiable reinforcement learning called online nonstochastic control. The new approach applies techniques from online convex optimization and convex relaxations to obtain new methods with provable guarantees for classical settings in optimal and robust control.  The primary distinction between online nonstochastic control and other frameworks is the objective. In optimal control, robust control, and other control methodologies that assume stochastic noise, the goal is to perform comparably to an offline optimal strategy. In online nonstochastic control, both the cost functions as well as the perturbations from the assumed dynamical model are chosen by an adversary. Thus the optimal policy is not defined a priori. Rather, the target is to attain low regret against the best policy in hindsight from a benchmark class of policies.  This objective suggests the use of the decision making frame
    

