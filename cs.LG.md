# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Meta-training Really Necessary for Molecular Few-Shot Learning ?](https://arxiv.org/abs/2404.02314) | 本文重新审视了分子数据微调方法，提出了基于马氏距离的正则化二次探针损失，并设计了块坐标下降优化器，使得在黑匣子设置下，简单微调方法在少样本学习中获得了竞争性表现，同时消除了特定预训练策略的需要。 |
| [^2] | [On the Inclusion of Charge and Spin States in Cartesian Tensor Neural Network Potentials](https://arxiv.org/abs/2403.15073) | TensorNet扩展了其能力，可以处理带电分子和自旋状态，提高了模型在各种化学系统中的预测准确性。 |
| [^3] | [ADAPT to Robustify Prompt Tuning Vision Transformers](https://arxiv.org/abs/2403.13196) | 本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。 |
| [^4] | [Learning Adversarial MDPs with Stochastic Hard Constraints](https://arxiv.org/abs/2403.03672) | 本论文首次研究了涉及对抗损失和硬约束的CMDP，在两种不同情形下设计了具有次线性遗憾的算法，填补了先前研究中对这一问题的空白。 |
| [^5] | [Addressing Distribution Shift in Time Series Forecasting with Instance Normalization Flows.](http://arxiv.org/abs/2401.16777) | 本文提出了一种通过实例规范化流解决时间序列预测中的分布偏移问题的方法，该方法不依赖于固定统计数据，也不限制于预测架构。通过双层优化问题实现转换和预测的联合学习，并提出了实例规范化流作为一种新颖的可逆网络用于时间序列转换。实验证明该方法在合成数据和真实数据上优于最先进的基线模型。 |
| [^6] | [Exploring Counterfactual Alignment Loss towards Human-centered AI.](http://arxiv.org/abs/2310.01766) | 该论文提出了一个基于反事实生成的以人为中心的框架，并引入了一种新的损失函数，用于保证反事实生成归因的特征与人类专家对齐。 |
| [^7] | [A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining.](http://arxiv.org/abs/2305.18407) | MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。 |

# 详细

[^1]: 分子少样本学习是否真的需要元训练？

    Is Meta-training Really Necessary for Molecular Few-Shot Learning ?

    [https://arxiv.org/abs/2404.02314](https://arxiv.org/abs/2404.02314)

    本文重新审视了分子数据微调方法，提出了基于马氏距离的正则化二次探针损失，并设计了块坐标下降优化器，使得在黑匣子设置下，简单微调方法在少样本学习中获得了竞争性表现，同时消除了特定预训练策略的需要。

    

    最近，少样本学习在药物发现领域引起了极大关注，而最近快速增长的文献大多涉及复杂的元学习策略。本文重新审视了更为直接的分子数据微调方法，并提出了基于马氏距离的正则化二次探针损失。我们设计了一个专门的块坐标下降优化器，避免了我们损失函数的退化解。有趣的是，我们的简单微调方法在与最先进方法的比较中获得了极具竞争力的表现，同时适用于黑匣子设置，并消除了特定情节预训练策略的需要。此外，我们引入了一个新的基准来评估竞争方法对领域转移的稳健性。在这个设置下，我们的微调基线始终比元学习方法取得更好的结果。

    arXiv:2404.02314v1 Announce Type: cross  Abstract: Few-shot learning has recently attracted significant interest in drug discovery, with a recent, fast-growing literature mostly involving convoluted meta-learning strategies. We revisit the more straightforward fine-tuning approach for molecular data, and propose a regularized quadratic-probe loss based on the the Mahalanobis distance. We design a dedicated block-coordinate descent optimizer, which avoid the degenerate solutions of our loss. Interestingly, our simple fine-tuning approach achieves highly competitive performances in comparison to state-of-the-art methods, while being applicable to black-box settings and removing the need for specific episodic pre-training strategies. Furthermore, we introduce a new benchmark to assess the robustness of the competing methods to domain shifts. In this setting, our fine-tuning baseline obtains consistently better results than meta-learning methods.
    
[^2]: 论笛卡尔张量神经网络势中包含电荷和自旋状态的研究

    On the Inclusion of Charge and Spin States in Cartesian Tensor Neural Network Potentials

    [https://arxiv.org/abs/2403.15073](https://arxiv.org/abs/2403.15073)

    TensorNet扩展了其能力，可以处理带电分子和自旋状态，提高了模型在各种化学系统中的预测准确性。

    

    在这封信中，我们提出了一种扩展TensorNet的方法，这是一种最先进的等变笛卡尔张量神经网络势，使其能够处理带电分子和自旋状态，而无需进行架构更改或增加成本。通过合并这些属性，我们解决了输入退化问题，增强了该模型在各种化学系统中的预测准确性。这一进展显著拓宽了TensorNet的适用范围，同时保持其效率和准确性。

    arXiv:2403.15073v1 Announce Type: new  Abstract: In this letter, we present an extension to TensorNet, a state-of-the-art equivariant Cartesian tensor neural network potential, allowing it to handle charged molecules and spin states without architectural changes or increased costs. By incorporating these attributes, we address input degeneracy issues, enhancing the model's predictive accuracy across diverse chemical systems. This advancement significantly broadens TensorNet's applicability, maintaining its efficiency and accuracy.
    
[^3]: 使Prompt调优视觉Transformer更为健壮的ADAPT

    ADAPT to Robustify Prompt Tuning Vision Transformers

    [https://arxiv.org/abs/2403.13196](https://arxiv.org/abs/2403.13196)

    本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。

    

    深度模型的性能，包括视觉Transformer，已知容易受到对抗性攻击的影响。许多现有对抗性防御方法，如对抗性训练，依赖于对整个模型进行全面微调以增加模型的稳健性。这些防御方法需要为每个任务存储整个模型的副本，而模型可能包含数十亿个参数。与此同时，参数高效的prompt调优被用来适应大型基于Transformer的模型到下游任务，无需保存大型副本。本文从稳健性的角度研究了对视觉Transformer进行下游任务的参数高效prompt调优。我们发现，之前的对抗性防御方法在应用到prompt调优范式时，存在梯度模糊并容易受到自适应攻击的影响。我们引入了ADAPT，一种在prompt调优范式中执行自适应对抗训练的新框架。

    arXiv:2403.13196v1 Announce Type: new  Abstract: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our meth
    
[^4]: 在具有随机硬约束的对抗MDP中学习

    Learning Adversarial MDPs with Stochastic Hard Constraints

    [https://arxiv.org/abs/2403.03672](https://arxiv.org/abs/2403.03672)

    本论文首次研究了涉及对抗损失和硬约束的CMDP，在两种不同情形下设计了具有次线性遗憾的算法，填补了先前研究中对这一问题的空白。

    

    我们研究带有对抗损失和随机硬约束的受限马尔可夫决策过程（CMDP）中的在线学习问题。我们考虑两种不同的情形。在第一种情形中，我们解决了一般CMDP问题，设计了一个算法，实现了次线性遗憾和累积正约束违反。在第二种情形中，在一个政策严格满足约束存在且为学习者所了解的温和假设下，我们设计了一个算法，实现了次线性遗憾，同时确保在每一轮中约束以高概率得到满足。据我们所知，我们的工作是第一个研究既涉及对抗损失又涉及硬约束的CMDP的工作。实际上，先前的研究要么集中在更弱的软约束上--允许正违反来抵消负违反--要么局限于随机损失。因此，我们的算法可以处理一般的非统计

    arXiv:2403.03672v1 Announce Type: new  Abstract: We study online learning problems in constrained Markov decision processes (CMDPs) with adversarial losses and stochastic hard constraints. We consider two different scenarios. In the first one, we address general CMDPs, where we design an algorithm that attains sublinear regret and cumulative positive constraints violation. In the second scenario, under the mild assumption that a policy strictly satisfying the constraints exists and is known to the learner, we design an algorithm that achieves sublinear regret while ensuring that the constraints are satisfied at every episode with high probability. To the best of our knowledge, our work is the first to study CMDPs involving both adversarial losses and hard constraints. Indeed, previous works either focus on much weaker soft constraints--allowing for positive violation to cancel out negative ones--or are restricted to stochastic losses. Thus, our algorithms can deal with general non-stat
    
[^5]: 通过实例规范化流解决时间序列预测中的分布偏移问题

    Addressing Distribution Shift in Time Series Forecasting with Instance Normalization Flows. (arXiv:2401.16777v1 [cs.LG])

    [http://arxiv.org/abs/2401.16777](http://arxiv.org/abs/2401.16777)

    本文提出了一种通过实例规范化流解决时间序列预测中的分布偏移问题的方法，该方法不依赖于固定统计数据，也不限制于预测架构。通过双层优化问题实现转换和预测的联合学习，并提出了实例规范化流作为一种新颖的可逆网络用于时间序列转换。实验证明该方法在合成数据和真实数据上优于最先进的基线模型。

    

    由于时间序列的非平稳性，分布偏移问题很大程度上阻碍了时间序列预测的性能。现有的解决方案要么无法处理超出简单统计的偏移，要么与预测模型的兼容性有限。在本文中，我们提出了一种针对时间序列预测的通用解耦公式，不依赖于固定统计数据，也不限制于预测架构。然后，我们将这种公式形式化为一个双层优化问题，以实现转换（外循环）和预测（内循环）的联合学习。此外，对于转换而言，对表达能力和双向性的特殊要求促使我们提出了实例规范化流（IN-Flow），一种新颖的可逆网络用于时间序列转换。大量实验证明我们的方法在合成数据和真实数据上始终优于最先进的基线模型。

    Due to non-stationarity of time series, the distribution shift problem largely hinders the performance of time series forecasting. Existing solutions either fail for the shifts beyond simple statistics or the limited compatibility with forecasting models. In this paper, we propose a general decoupled formulation for time series forecasting, with no reliance on fixed statistics and no restriction on forecasting architectures. Then, we make such a formulation formalized into a bi-level optimization problem, to enable the joint learning of the transformation (outer loop) and forecasting (inner loop). Moreover, the special requirements of expressiveness and bi-direction for the transformation motivate us to propose instance normalization flows (IN-Flow), a novel invertible network for time series transformation. Extensive experiments demonstrate our method consistently outperforms state-of-the-art baselines on both synthetic and real-world data.
    
[^6]: 探索针对以人为中心的人工智能的反事实对齐损失

    Exploring Counterfactual Alignment Loss towards Human-centered AI. (arXiv:2310.01766v1 [cs.LG])

    [http://arxiv.org/abs/2310.01766](http://arxiv.org/abs/2310.01766)

    该论文提出了一个基于反事实生成的以人为中心的框架，并引入了一种新的损失函数，用于保证反事实生成归因的特征与人类专家对齐。

    

    深度神经网络在监督学习任务中具有令人印象深刻的准确性。然而，它们缺乏透明度，使得人们难以信任它们的结果，特别是在安全-批评领域如医疗保健中。为了解决这个问题，最近的解释引导学习方法提出了将基于梯度的注意力映射与人类专家标注的图像区域对齐的方法，从而获得一个本质上以人为中心的模型。然而，这些方法所基于的注意力映射可能无法因果地归因于模型预测，从而损害了对其对齐的有效性。为了解决这个问题，我们提出了一个基于反事实生成的新型以人为中心的框架。具体而言，我们利用反事实生成的因果归因能力引入了一种新的损失，称为反事实对齐损失（CF-Align）。这个损失保证了分类器由反事实生成归因的特征与人类专家对齐。

    Deep neural networks have demonstrated impressive accuracy in supervised learning tasks. However, their lack of transparency makes it hard for humans to trust their results, especially in safe-critic domains such as healthcare. To address this issue, recent explanation-guided learning approaches proposed to align the gradient-based attention map to image regions annotated by human experts, thereby obtaining an intrinsically human-centered model. However, the attention map these methods are based on may fail to causally attribute the model predictions, thus compromising their validity for alignment. To address this issue, we propose a novel human-centered framework based on counterfactual generation. In particular, we utilize the counterfactual generation's ability for causal attribution to introduce a novel loss called the CounterFactual Alignment (CF-Align) loss. This loss guarantees that the features attributed by the counterfactual generation for the classifier align with the human 
    
[^7]: 一种用于分子多模态预训练的群对称随机微分方程模型。

    A Group Symmetric Stochastic Differential Equation Model for Molecule Multi-modal Pretraining. (arXiv:2305.18407v1 [cs.LG])

    [http://arxiv.org/abs/2305.18407](http://arxiv.org/abs/2305.18407)

    MoleculeSDE是用于分子多模态预训练的群对称随机微分方程模型，通过在输入空间中直接生成3D几何与2D拓扑之间的转换，它能够更有效地保存分子结构信息。

    

    分子预训练已经成为提高基于 AI 的药物发现性能的主流方法。然而，大部分现有的方法只关注单一的模态。最近的研究表明，最大化两种模态之间的互信息（MI）可以增强分子表示能力。而现有的分子多模态预训练方法基于从拓扑和几何编码的表示空间来估计 MI，因此丢失了分子的关键结构信息。为解决这一问题，我们提出了 MoleculeSDE。MoleculeSDE利用群对称（如 SE（3）-等变和反射-反对称）随机微分方程模型在输入空间中直接生成 3D 几何形状与 2D 拓扑之间的转换。它不仅获得更紧的MI界限，而且还能够有效地保存分子结构信息。

    Molecule pretraining has quickly become the go-to schema to boost the performance of AI-based drug discovery. Naturally, molecules can be represented as 2D topological graphs or 3D geometric point clouds. Although most existing pertaining methods focus on merely the single modality, recent research has shown that maximizing the mutual information (MI) between such two modalities enhances the molecule representation ability. Meanwhile, existing molecule multi-modal pretraining approaches approximate MI based on the representation space encoded from the topology and geometry, thus resulting in the loss of critical structural information of molecules. To address this issue, we propose MoleculeSDE. MoleculeSDE leverages group symmetric (e.g., SE(3)-equivariant and reflection-antisymmetric) stochastic differential equation models to generate the 3D geometries from 2D topologies, and vice versa, directly in the input space. It not only obtains tighter MI bound but also enables prosperous dow
    

