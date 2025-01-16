# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synapse: Learning Preferential Concepts from Visual Demonstrations](https://arxiv.org/abs/2403.16689) | Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。 |
| [^2] | [Understanding Emergent Abilities of Language Models from the Loss Perspective](https://arxiv.org/abs/2403.15796) | 本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。 |
| [^3] | [SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection](https://arxiv.org/abs/2402.16705) | SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。 |
| [^4] | [Learning Optimal Tax Design in Nonatomic Congestion Games](https://arxiv.org/abs/2402.07437) | 本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。 |
| [^5] | [Metric Space Magnitude for Evaluating the Diversity of Latent Representations](https://arxiv.org/abs/2311.16054) | 基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。 |
| [^6] | [Ensemble sampling for linear bandits: small ensembles suffice](https://arxiv.org/abs/2311.08376) | 该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。 |
| [^7] | [SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.](http://arxiv.org/abs/2401.15299) | SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。 |
| [^8] | [Interpreting Equivariant Representations.](http://arxiv.org/abs/2401.12588) | 本文研究了潜在表示的等变性以及在使用中考虑等变模型的归纳偏差的重要性，提出了选择不变投影的原则，并展示了两个实例的影响。 |
| [^9] | [Machine unlearning through fine-grained model parameters perturbation.](http://arxiv.org/abs/2401.04385) | 本文提出了一种精细的机器去学习策略，通过细粒度模型参数的扰动来实现用户隐私保护，同时保持可控的计算成本。采用遗忘率和记忆保留率等新的指标来评估去学习效果和模型泛化能力。 |
| [^10] | [Volterra Accentuated Non-Linear Dynamical Admittance (VANYA) to model Deforestation: An Exemplification from the Amazon Rainforest.](http://arxiv.org/abs/2308.06471) | 本文研究了利用Volterra强调非线性动力学可通量性 (VANYA) 模型来模拟森林砍伐，该模型结合了捕食者-被捕食者动力学，并在对亚马逊雨林数据进行了预测，并与其他预测方法进行了比较。 |
| [^11] | [Clarify Confused Nodes Through Separated Learning.](http://arxiv.org/abs/2306.02285) | 本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。 |
| [^12] | [Finite-Sample Bounds for Adaptive Inverse Reinforcement Learning using Passive Langevin Dynamics.](http://arxiv.org/abs/2304.09123) | 本文提供了有限时间界限，用于被动随机梯度 Langevin 动力学算法，该算法可用于逆强化学习。该算法充当随机采样器，恢复用外部过程优化而来的成本函数。 |
| [^13] | [Expressive Text-to-Image Generation with Rich Text.](http://arxiv.org/abs/2304.06720) | 本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。 |
| [^14] | [Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models.](http://arxiv.org/abs/2304.03271) | 本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。 |

# 详细

[^1]: Synapse: 从视觉演示中学习优先概念

    Synapse: Learning Preferential Concepts from Visual Demonstrations

    [https://arxiv.org/abs/2403.16689](https://arxiv.org/abs/2403.16689)

    Synapse是一种神经符号化方法，旨在从有限演示中高效学习偏好概念，通过将偏好表示为神经符号程序并利用视觉解析、大型语言模型和程序合成相结合的方式来学习个人偏好。

    

    本文解决了偏好学习问题，旨在从视觉输入中学习用户特定偏好（例如，“好停车位”，“方便的下车位置”）。尽管与学习事实概念（例如，“红色立方体”）相似，但偏好学习是一个基本更加困难的问题，因为它涉及主观性质和个人特定训练数据的缺乏。我们使用一种名为Synapse的新框架来解决这个问题，这是一种神经符号化方法，旨在有效地从有限演示中学习偏好概念。Synapse将偏好表示为在图像上运作的领域特定语言（DSL）中的神经符号程序，并利用视觉解析、大型语言模型和程序合成的新组合来学习代表个人偏好的程序。我们通过广泛的实验评估了Synapse，包括一个关注与移动相关的用户案例研究。

    arXiv:2403.16689v1 Announce Type: cross  Abstract: This paper addresses the problem of preference learning, which aims to learn user-specific preferences (e.g., "good parking spot", "convenient drop-off location") from visual input. Despite its similarity to learning factual concepts (e.g., "red cube"), preference learning is a fundamentally harder problem due to its subjective nature and the paucity of person-specific training data. We address this problem using a new framework called Synapse, which is a neuro-symbolic approach designed to efficiently learn preferential concepts from limited demonstrations. Synapse represents preferences as neuro-symbolic programs in a domain-specific language (DSL) that operates over images, and leverages a novel combination of visual parsing, large language models, and program synthesis to learn programs representing individual preferences. We evaluate Synapse through extensive experimentation including a user case study focusing on mobility-related
    
[^2]: 从损失角度理解语言模型的突现能力

    Understanding Emergent Abilities of Language Models from the Loss Perspective

    [https://arxiv.org/abs/2403.15796](https://arxiv.org/abs/2403.15796)

    本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。

    

    近期研究质疑了传统认为语言模型的突现能力仅存在于大模型中的观点。这种怀疑源自两点观察：1）较小的模型也能展现出对突现能力的高性能；2）质疑用于测量这些能力的不连续性指标。本文提议从预训练损失的角度研究突现能力，而非模型大小或训练计算。我们展示了具有相同预训练损失但不同模型和数据大小的模型，在各种下游任务上表现相同。我们还发现，当某一模型的预训练损失低于特定阈值时，在某些任务上表现出突现能力，而不论指标的连续性如何；而在达到该阈值之前，其性能仍保持在随机猜测水平。这启发我们重新定义突现能力为那些......

    arXiv:2403.15796v1 Announce Type: cross  Abstract: Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhibit high performance on emergent abilities and 2) there is doubt on the discontinuous metrics used to measure these abilities. In this paper, we propose to study emergent abilities in the lens of pre-training loss, instead of model size or training compute. We demonstrate that the models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks. We also discover that a model exhibits emergent abilities on certain tasks -- regardless of the continuity of metrics -- when its pre-training loss falls below a specific threshold. Before reaching this threshold, its performance remains at the level of random guessing. This inspires us to redefine emergent abilities as those that
    
[^3]: SelectIT: 通过基于不确定性的自我反思实现大型语言模型的选择性指导调整

    SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection

    [https://arxiv.org/abs/2402.16705](https://arxiv.org/abs/2402.16705)

    SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。

    

    指导调整（IT）对于调整大型语言模型（LLMs）以适应人类中心交互至关重要。最近的进展表明，精心选择一小部分高质量的IT数据可以显着提高LLMs的性能。尽管如此，常见方法通常依赖于额外的模型或数据集，这增加了成本并限制了广泛采用。在这项工作中，我们提出了一种新颖的方法，称为SelectIT，它利用LLM本身的基本能力。具体来说，我们利用LLMs中固有的不确定性，更有效地选择高质量的IT数据，而无需额外资源。此外，我们介绍了一种新颖的IT数据集，名为选择性羊驼（Selective Alpaca），通过将SelectIT应用于Alpaca-GPT4数据集而创建。实证结果表明，使用选择性羊驼进行IT可以极大地提升模型性能。SelectIT的稳健性也得到了验证。

    arXiv:2402.16705v1 Announce Type: new  Abstract: Instruction tuning (IT) is crucial to tailoring large language models (LLMs) towards human-centric interactions. Recent advancements have shown that the careful selection of a small, high-quality subset of IT data can significantly enhance the performance of LLMs. Despite this, common approaches often rely on additional models or data sets, which increases costs and limits widespread adoption. In this work, we propose a novel approach, termed SelectIT, that capitalizes on the foundational capabilities of the LLM itself. Specifically, we exploit the intrinsic uncertainty present in LLMs to more effectively select high-quality IT data, without the need for extra resources. Furthermore, we introduce a novel IT dataset, the Selective Alpaca, created by applying SelectIT to the Alpaca-GPT4 dataset. Empirical results demonstrate that IT using Selective Alpaca leads to substantial model ability enhancement. The robustness of SelectIT has also b
    
[^4]: 非原子拥堵博弈中学习最优税收设计

    Learning Optimal Tax Design in Nonatomic Congestion Games

    [https://arxiv.org/abs/2402.07437](https://arxiv.org/abs/2402.07437)

    本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。

    

    本研究探讨了如何学习最优税收设计，以在非原子拥堵博弈中最大化效率。众所周知，玩家之间的自利行为可能会破坏系统的效率。税务机制是缓解此问题并引导社会最优行为的常见方法。在这项工作中，我们首次采取了学习最优税收的初始步骤，该最优税收可以通过平衡反馈来最小化社会成本，即税务设计者只能观察到强制税收下的均衡状态。由于指数级的税收函数空间，梯度不存在和目标函数的非凸性，现有算法不适用。为了解决这些挑战，我们的算法利用了几个新颖的组成部分：（1）分段线性税收来近似最优税收；（2）额外的线性项来保证强凸潜力函数；（3）有效的子例程来找到“边界”税收。该算法可以找到一个$\epsilon$-最优税收，时间复杂度为$O(\bet

    We study how to learn the optimal tax design to maximize the efficiency in nonatomic congestion games. It is known that self-interested behavior among the players can damage the system's efficiency. Tax mechanisms is a common method to alleviate this issue and induce socially optimal behavior. In this work, we take the initial step for learning the optimal tax that can minimize the social cost with \emph{equilibrium feedback}, i.e., the tax designer can only observe the equilibrium state under the enforced tax. Existing algorithms are not applicable due to the exponentially large tax function space, nonexistence of the gradient, and nonconvexity of the objective. To tackle these challenges, our algorithm leverages several novel components: (1) piece-wise linear tax to approximate the optimal tax; (2) an extra linear term to guarantee a strongly convex potential function; (3) efficient subroutine to find the ``boundary'' tax. The algorithm can find an $\epsilon$-optimal tax with $O(\bet
    
[^5]: 用于评估潜在表示多样性的度量空间大小

    Metric Space Magnitude for Evaluating the Diversity of Latent Representations

    [https://arxiv.org/abs/2311.16054](https://arxiv.org/abs/2311.16054)

    基于度量空间大小的潜在表示多样性度量，可稳定计算，能够进行多尺度比较，在多个领域和任务中展现出优越性能。

    

    度量空间的大小是一种近期建立的不变性，能够在多个尺度上提供空间的“有效大小”的衡量，并捕捉到许多几何属性。我们发展了一系列基于大小的潜在表示内在多样性度量，形式化了有限度量空间大小函数之间的新颖不相似性概念。我们的度量在数据扰动下保证稳定，可以高效计算，并且能够对潜在表示进行严格的多尺度比较。我们展示了我们的度量在实验套件中的实用性和卓越性能，包括不同领域和任务的多样性评估、模式崩溃检测以及用于文本、图像和图形数据的生成模型评估。

    The magnitude of a metric space is a recently-established invariant, providing a measure of the 'effective size' of a space across multiple scales while also capturing numerous geometrical properties. We develop a family of magnitude-based measures of the intrinsic diversity of latent representations, formalising a novel notion of dissimilarity between magnitude functions of finite metric spaces. Our measures are provably stable under perturbations of the data, can be efficiently calculated, and enable a rigorous multi-scale comparison of latent representations. We show the utility and superior performance of our measures in an experimental suite that comprises different domains and tasks, including the evaluation of diversity, the detection of mode collapse, and the evaluation of generative models for text, image, and graph data.
    
[^6]: 线性赌臂的集成抽样：小集成足矣

    Ensemble sampling for linear bandits: small ensembles suffice

    [https://arxiv.org/abs/2311.08376](https://arxiv.org/abs/2311.08376)

    该论文对随机线性赌臂环境中的集成抽样进行了首次实用和严格的分析，展示了在标准假设下，采用规模为$d \log T$的集成抽样可以获得接近$\sqrt{T}$阶的后悔，而不需要集成大小与$T$线性扩展。

    

    我们首次对随机线性赌臂设定下的集成抽样进行了有用且严谨的分析。特别地，我们展示了在标准假设下，对于一个具有交互作用时间跨度$T$的$d$维随机线性赌臂，采用集成大小为$\smash{d \log T}$的集成抽样，遭受的后悔最多为$\smash{(d \log T)^{5/2} \sqrt{T}}$阶。我们的结果是在任何结构化环境中第一个不要求集成大小与$T$线性扩展的结果，这使得集成抽样失去意义，同时获得了接近$\smash{\sqrt{T}}$阶的后悔。我们的结果也是第一个允许无限动作集的结果。

    arXiv:2311.08376v2 Announce Type: replace-cross  Abstract: We provide the first useful and rigorous analysis of ensemble sampling for the stochastic linear bandit setting. In particular, we show that, under standard assumptions, for a $d$-dimensional stochastic linear bandit with an interaction horizon $T$, ensemble sampling with an ensemble of size of order $\smash{d \log T}$ incurs regret at most of the order $\smash{(d \log T)^{5/2} \sqrt{T}}$. Ours is the first result in any structured setting not to require the size of the ensemble to scale linearly with $T$ -- which defeats the purpose of ensemble sampling -- while obtaining near $\smash{\sqrt{T}}$ order regret. Ours is also the first result that allows infinite action sets.
    
[^7]: SupplyGraph: 使用图神经网络进行供应链规划的基准数据集

    SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks. (arXiv:2401.15299v1 [cs.LG])

    [http://arxiv.org/abs/2401.15299](http://arxiv.org/abs/2401.15299)

    SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。

    

    图神经网络（GNNs）在不同领域如运输、生物信息学、语言处理和计算机视觉中取得了重要进展。然而，在将GNNs应用于供应链网络方面，目前尚缺乏研究。供应链网络在结构上类似于图形，使其成为应用GNN方法的理想选择。这为优化、预测和解决供应链问题开辟了无限可能。然而，此方法的一个主要障碍在于缺乏真实世界的基准数据集以促进使用GNN来研究和解决供应链问题。为了解决这个问题，我们提供了一个来自孟加拉国一家领先的快速消费品公司的实际基准数据集，该数据集侧重于用于生产目的的供应链规划的时间任务。该数据集包括时间数据作为节点特征，以实现销售预测、生产计划和故障识别。

    Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of fa
    
[^8]: 解读等变表示

    Interpreting Equivariant Representations. (arXiv:2401.12588v1 [cs.LG])

    [http://arxiv.org/abs/2401.12588](http://arxiv.org/abs/2401.12588)

    本文研究了潜在表示的等变性以及在使用中考虑等变模型的归纳偏差的重要性，提出了选择不变投影的原则，并展示了两个实例的影响。

    

    对于深度学习模型的可视化、插值或特征提取等下游任务，潜在表示被广泛使用。不变和等变神经网络是用于强制执行归纳偏差的强大且已建立的模型。本文表明，在使用潜在表示时，必须同时考虑等变模型施加的归纳偏差。我们展示了不考虑归纳偏差会导致下游任务性能下降，相反，通过使用潜在表示的不变投影可以有效地考虑归纳偏差。我们提出了选择这样一个投影的原则，并展示了在两个常见例子中使用这些原则的影响：首先，我们研究了一种用于分子图生成的置换等变变分自动编码器；在这里，我们展示了可以设计出不产生信息损失的不变投影。

    Latent representations are used extensively for downstream tasks, such as visualization, interpolation or feature extraction of deep learning models. Invariant and equivariant neural networks are powerful and well-established models for enforcing inductive biases. In this paper, we demonstrate that the inductive bias imposed on the by an equivariant model must also be taken into account when using latent representations. We show how not accounting for the inductive biases leads to decreased performance on downstream tasks, and vice versa, how accounting for inductive biases can be done effectively by using an invariant projection of the latent representations. We propose principles for how to choose such a projection, and show the impact of using these principles in two common examples: First, we study a permutation equivariant variational auto-encoder trained for molecule graph generation; here we show that invariant projections can be designed that incur no loss of information in the
    
[^9]: 通过细粒度模型参数扰动实现机器去学习

    Machine unlearning through fine-grained model parameters perturbation. (arXiv:2401.04385v1 [cs.LG])

    [http://arxiv.org/abs/2401.04385](http://arxiv.org/abs/2401.04385)

    本文提出了一种精细的机器去学习策略，通过细粒度模型参数的扰动来实现用户隐私保护，同时保持可控的计算成本。采用遗忘率和记忆保留率等新的指标来评估去学习效果和模型泛化能力。

    

    机器去学习技术涉及到撤销数据记录和减小该数据对训练模型的影响，从而帮助实现用户隐私保护目标，但会带来显著的计算成本。基于参数扰动的权重去学习是一种通用方法，但通常涉及到全局修改参数。我们提出了精细的Top-K和Random-k参数扰动不精确机器去学习策略，以满足隐私需求同时保持计算成本可控。为了展示我们策略的有效性，我们还解决了评估机器去学习效果的挑战，考虑了模型在去学习和剩余数据上的广义性能。为了更好地评估去学习效果和模型泛化能力，我们提出了新的指标，即遗忘率和记忆保留率。然而，对于不精确的机器去学习，现有的指标无法对去学习程度进行准确量化。

    Machine unlearning techniques, which involve retracting data records and reducing influence of said data on trained models, help with the user privacy protection objective but incur significant computational costs. Weight perturbation-based unlearning is a general approach, but it typically involves globally modifying the parameters. We propose fine-grained Top-K and Random-k parameters perturbed inexact machine unlearning strategies that address the privacy needs while keeping the computational costs tractable.  In order to demonstrate the efficacy of our strategies we also tackle the challenge of evaluating the effectiveness of machine unlearning by considering the model's generalization performance across both unlearning and remaining data. To better assess the unlearning effect and model generalization, we propose novel metrics, namely, the forgetting rate and memory retention rate. However, for inexact machine unlearning, current metrics are inadequate in quantifying the degree of
    
[^10]: Volterra强调非线性动力学可通量性 (VANYA) 在森林砍伐模拟中的应用：以亚马逊雨林为例

    Volterra Accentuated Non-Linear Dynamical Admittance (VANYA) to model Deforestation: An Exemplification from the Amazon Rainforest. (arXiv:2308.06471v1 [cs.LG])

    [http://arxiv.org/abs/2308.06471](http://arxiv.org/abs/2308.06471)

    本文研究了利用Volterra强调非线性动力学可通量性 (VANYA) 模型来模拟森林砍伐，该模型结合了捕食者-被捕食者动力学，并在对亚马逊雨林数据进行了预测，并与其他预测方法进行了比较。

    

    智能自动化技术通过其最新的技术进展，在抵御飓风、干旱和地震等方面给予了我们支持。算法学习已经推动了神经科学、遗传学和人机交互等领域的发展。时间序列数据对进展起到了促进作用。在传统领域中采用这些方法仍存在挑战。神经网络面临理解和偏见问题。人工智能在科学领域的扩展是由于其可适应的简单描述符和组合论证。本文侧重于利用VANYA模型预测林地损失，并结合捕食者-被捕食者动力学。VANYA模型对亚马逊雨林的数据进行了预测，并与其他预测方法（如长短期记忆、N-BEATS和RCN）进行了对比。

    Intelligent automation supports us against cyclones, droughts, and seismic events with recent technology advancements. Algorithmic learning has advanced fields like neuroscience, genetics, and human-computer interaction. Time-series data boosts progress. Challenges persist in adopting these approaches in traditional fields. Neural networks face comprehension and bias issues. AI's expansion across scientific areas is due to adaptable descriptors and combinatorial argumentation. This article focuses on modeling Forest loss using the VANYA Model, incorporating Prey Predator Dynamics. VANYA predicts forest cover, demonstrated on Amazon Rainforest data against other forecasters like Long Short-Term Memory, N-BEATS, RCN.
    
[^11]: 通过分离学习解决混淆节点问题

    Clarify Confused Nodes Through Separated Learning. (arXiv:2306.02285v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02285](http://arxiv.org/abs/2306.02285)

    本文提出了使用邻域混淆度量来分离学习解决图神经网络中混淆节点的问题。这种方法可以更可靠地区分异质节点和同质节点，并改善性能。

    

    图神经网络（GNN）在图导向任务中取得了显著的进展。然而，现实世界的图中不可避免地包含一定比例的异质节点，这挑战了经典GNN的同质性假设，并阻碍了其性能。现有研究大多数仍设计了具有异质节点和同质节点间共享权重的通用模型。尽管这些努力中包含了高阶信息和多通道架构，但往往效果不佳。少数研究尝试训练不同节点组的分离学习，但受到了不合适的分离度量和低效率的影响。本文首先提出了一种新的度量指标，称为邻域混淆（NC），以便更可靠地分离节点。我们观察到具有不同NC值的节点组在组内准确度和可视化嵌入上存在一定差异。这为基于邻域混淆的图卷积网络（NC-GCN）铺平了道路。

    Graph neural networks (GNNs) have achieved remarkable advances in graph-oriented tasks. However, real-world graphs invariably contain a certain proportion of heterophilous nodes, challenging the homophily assumption of classical GNNs and hindering their performance. Most existing studies continue to design generic models with shared weights between heterophilous and homophilous nodes. Despite the incorporation of high-order messages or multi-channel architectures, these efforts often fall short. A minority of studies attempt to train different node groups separately but suffer from inappropriate separation metrics and low efficiency. In this paper, we first propose a new metric, termed Neighborhood Confusion (NC), to facilitate a more reliable separation of nodes. We observe that node groups with different levels of NC values exhibit certain differences in intra-group accuracy and visualized embeddings. These pave the way for Neighborhood Confusion-guided Graph Convolutional Network (N
    
[^12]: 使用被动 Langevin 动力学的自适应逆强化学习的有限样本界限

    Finite-Sample Bounds for Adaptive Inverse Reinforcement Learning using Passive Langevin Dynamics. (arXiv:2304.09123v1 [cs.LG])

    [http://arxiv.org/abs/2304.09123](http://arxiv.org/abs/2304.09123)

    本文提供了有限时间界限，用于被动随机梯度 Langevin 动力学算法，该算法可用于逆强化学习。该算法充当随机采样器，恢复用外部过程优化而来的成本函数。

    

    随机梯度 Langevin 动力学 (SGLD) 是从概率分布采样的有用方法。本文提供了一个被动随机梯度 Langevin 动力学算法 (PSGLD) 的有限样本分析，旨在实现逆强化学习。此处的“被动”是指 PSGLD 算法(逆学习过程)可用的噪声渐变是由外部随机梯度算法(正向学习器)在随机选择的点上评估的。PSGLD 算法因此充当一个随机采样器，可恢复正在被此外部过程优化的成本函数。以前的工作使用随机逼近技术分析了这个被动算法的渐近性能；在本文中，我们分析了它的有限时间性能。具体而言，我们提供了在被动算法和其稳定测度之间的 2-Wasserstein 距离上的有限时间界限，从中可以获得重建的成本函数。

    Stochastic gradient Langevin dynamics (SGLD) are a useful methodology for sampling from probability distributions. This paper provides a finite sample analysis of a passive stochastic gradient Langevin dynamics algorithm (PSGLD) designed to achieve inverse reinforcement learning. By "passive", we mean that the noisy gradients available to the PSGLD algorithm (inverse learning process) are evaluated at randomly chosen points by an external stochastic gradient algorithm (forward learner). The PSGLD algorithm thus acts as a randomized sampler which recovers the cost function being optimized by this external process. Previous work has analyzed the asymptotic performance of this passive algorithm using stochastic approximation techniques; in this work we analyze the non-asymptotic performance. Specifically, we provide finite-time bounds on the 2-Wasserstein distance between the passive algorithm and its stationary measure, from which the reconstructed cost function is obtained.
    
[^13]: 富文本生成表达性文本图像

    Expressive Text-to-Image Generation with Rich Text. (arXiv:2304.06720v1 [cs.CV])

    [http://arxiv.org/abs/2304.06720](http://arxiv.org/abs/2304.06720)

    本文提出了一种使用富文本编辑器生成表达性文本图像的方法，可以通过局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成，生成高质量且多样化的图像。

    

    纯文本已经成为文字到图像合成的流行界面。但是，它的定制选项有限，阻碍了用户精确描述所需的输出。为了解决这些挑战，我们提出使用支持字体样式、大小、颜色和脚注等格式的富文本编辑器。我们从富文本中提取每个字的属性，以启用局部样式控制、明确的标记重新加权、精确的颜色渲染和详细的区域合成。我们通过基于区域的扩散过程实现了这些功能。我们的实验表明，我们的方法可以比现有的文本到图像方法更好地生成高质量和多样化的图像。

    Plain text has become a prevalent interface for text-to-image synthesis. However, its limited customization options hinder users from accurately describing desired outputs. For example, plain text makes it hard to specify continuous quantities, such as the precise RGB color value or importance of each word. Furthermore, creating detailed text prompts for complex scenes is tedious for humans to write and challenging for text encoders to interpret. To address these challenges, we propose using a rich-text editor supporting formats such as font style, size, color, and footnote. We extract each word's attributes from rich text to enable local style control, explicit token reweighting, precise color rendering, and detailed region synthesis. We achieve these capabilities through a region-based diffusion process. We first obtain each word's region based on cross-attention maps of a vanilla diffusion process using plain text. For each region, we enforce its text attributes by creating region-s
    
[^14]: 使AI“口渴”减少的方法：揭示和解决AI模型的秘密水消耗

    Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models. (arXiv:2304.03271v1 [cs.LG])

    [http://arxiv.org/abs/2304.03271](http://arxiv.org/abs/2304.03271)

    本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。

    

    人工智能（AI）模型的碳足迹不断增长，特别是像GPT-3和GPT-4这样的大型模型，已经受到公众的关注。然而，同等重要且巨大的AI模型水印尚未引起人们的注意。例如，在微软最先进的美国数据中心中训练GPT-3可以直接消耗70万升清洁淡水（相当于生产370辆宝马汽车或320辆特斯拉电动汽车），如果在微软的亚洲数据中心进行训练，这个水消耗量将增加三倍，但这样的信息一直被保密。这极其令人担忧，因为淡水短缺已成为在人口迅速增长、水资源减少和老化的水基础设施的背景下，我们所有人面临的最紧迫的挑战之一。为了应对全球水资源的挑战，人工智能模型可以，而且应该，承担社会责任，以身作则解决自己的问题。

    The growing carbon footprint of artificial intelligence (AI) models, especially large ones such as GPT-3 and GPT-4, has been undergoing public scrutiny. Unfortunately, however, the equally important and enormous water footprint of AI models has remained under the radar. For example, training GPT-3 in Microsoft's state-of-the-art U.S. data centers can directly consume 700,000 liters of clean freshwater (enough for producing 370 BMW cars or 320 Tesla electric vehicles) and the water consumption would have been tripled if training were done in Microsoft's Asian data centers, but such information has been kept as a secret. This is extremely concerning, as freshwater scarcity has become one of the most pressing challenges shared by all of us in the wake of the rapidly growing population, depleting water resources, and aging water infrastructures. To respond to the global water challenges, AI models can, and also should, take social responsibility and lead by example by addressing their own 
    

