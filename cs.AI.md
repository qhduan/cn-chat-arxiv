# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^2] | [Meta-Tasks: An alternative view on Meta-Learning Regularization](https://arxiv.org/abs/2402.18599) | 该论文提出了一种新颖的解决方案，通过使用meta-tasks作为元学习正则化的视角，实现了对训练和新颖任务的泛化，避免标记数据稀缺的困扰，并在实验中表现优越，相较于原型网络提高了3.9%的性能。 |
| [^3] | [One-Shot Graph Representation Learning Using Hyperdimensional Computing](https://arxiv.org/abs/2402.17073) | 该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。 |
| [^4] | [Universal Physics Transformers](https://arxiv.org/abs/2402.12365) | 提出了通用物理变压器（UPTs）这一新颖学习范式，能够模拟广泛的时空问题，同时适用于拉格朗日和欧拉离散化方案，有效地传播动态并允许查询潜在空间 |
| [^5] | [Rethink Model Re-Basin and the Linear Mode Connectivity](https://arxiv.org/abs/2402.05966) | 本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。 |

# 详细

[^1]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^2]: Meta-Tasks: 元学习正则化的另一种视角

    Meta-Tasks: An alternative view on Meta-Learning Regularization

    [https://arxiv.org/abs/2402.18599](https://arxiv.org/abs/2402.18599)

    该论文提出了一种新颖的解决方案，通过使用meta-tasks作为元学习正则化的视角，实现了对训练和新颖任务的泛化，避免标记数据稀缺的困扰，并在实验中表现优越，相较于原型网络提高了3.9%的性能。

    

    Few-shot learning (FSL)是一个具有挑战性的机器学习问题，因为标记数据稀缺。这篇论文提出了一种新颖的解决方案，可以泛化到训练和新颖的任务，同时利用未标记样本。该方法在更新外层循环之前，使用无监督技术对嵌入模型进行了细化，将其作为“元任务”。实验结果表明，我们提出的方法在新颖和训练任务上表现良好，收敛更快、更好，泛化误差和标准差更低，表明其在FSL中的实际应用潜力。实验结果表明，所提出的方法的表现比原型网络高出3.9%。

    arXiv:2402.18599v1 Announce Type: cross  Abstract: Few-shot learning (FSL) is a challenging machine learning problem due to a scarcity of labeled data. The ability to generalize effectively on both novel and training tasks is a significant barrier to FSL. This paper proposes a novel solution that can generalize to both training and novel tasks while also utilizing unlabeled samples. The method refines the embedding model before updating the outer loop using unsupervised techniques as ``meta-tasks''. The experimental results show that our proposed method performs well on novel and training tasks, with faster and better convergence, lower generalization, and standard deviation error, indicating its potential for practical applications in FSL. The experimental results show that the proposed method outperforms prototypical networks by 3.9%.
    
[^3]: 使用超高维计算进行单次图表示学习

    One-Shot Graph Representation Learning Using Hyperdimensional Computing

    [https://arxiv.org/abs/2402.17073](https://arxiv.org/abs/2402.17073)

    该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    

    我们提出了一种新颖、简单、快速、高效的半监督图学习方法。所提方法利用超高维计算，将数据样本使用随机投影编码到高维空间（简称HD空间）。具体来说，我们提出了一种利用图神经网络节点表示的单射性质的超高维图学习（HDGL）算法。HDGL将节点特征映射到HD空间，然后使用HD运算符（如捆绑和绑定）来聚合每个节点的局部邻域信息。对广泛使用的基准数据集进行的实验结果显示，HDGL实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    arXiv:2402.17073v1 Announce Type: cross  Abstract: We present a novel, simple, fast, and efficient approach for semi-supervised learning on graphs. The proposed approach takes advantage of hyper-dimensional computing which encodes data samples using random projections into a high dimensional space (HD space for short). Specifically, we propose a Hyper-dimensional Graph Learning (HDGL) algorithm that leverages the injectivity property of the node representations of a family of graph neural networks. HDGL maps node features to the HD space and then uses HD operators such as bundling and binding to aggregate information from the local neighborhood of each node. Results of experiments with widely used benchmark data sets show that HDGL achieves predictive performance that is competitive with the state-of-the-art deep learning methods, without the need for computationally expensive training.
    
[^4]: 通用物理变压器

    Universal Physics Transformers

    [https://arxiv.org/abs/2402.12365](https://arxiv.org/abs/2402.12365)

    提出了通用物理变压器（UPTs）这一新颖学习范式，能够模拟广泛的时空问题，同时适用于拉格朗日和欧拉离散化方案，有效地传播动态并允许查询潜在空间

    

    基于深度神经网络的偏微分方程替代者近来引起了越来越多的关注。然而，类似于它们的数值对应物，在不同应用中使用不同的技术，即使系统的基础动态相似。一个著名的例子是在计算流体动力学中的拉格朗日和欧拉表述，这为神经网络有效地建模基于粒子而不是网格的动态构成了挑战。我们引入了通用物理变压器（UPTs），这是一种新颖的学习范式，它模拟了一系列时空问题 - 对拉格朗日和欧拉离散化方案。UPTs在没有基于网格或基于粒子的潜在结构的情况下运行，从而在网格和粒子之间实现了灵活性。UPTs在潜在空间中高效传播动态，强调了逆编码和解码技术。最后，UPTs允许查询潜在空间表现

    arXiv:2402.12365v1 Announce Type: cross  Abstract: Deep neural network based surrogates for partial differential equations have recently gained increased interest. However, akin to their numerical counterparts, different techniques are used across applications, even if the underlying dynamics of the systems are similar. A prominent example is the Lagrangian and Eulerian specification in computational fluid dynamics, posing a challenge for neural networks to effectively model particle- as opposed to grid-based dynamics. We introduce Universal Physics Transformers (UPTs), a novel learning paradigm which models a wide range of spatio-temporal problems - both for Lagrangian and Eulerian discretization schemes. UPTs operate without grid- or particle-based latent structures, enabling flexibility across meshes and particles. UPTs efficiently propagate dynamics in the latent space, emphasized by inverse encoding and decoding techniques. Finally, UPTs allow for queries of the latent space repre
    
[^5]: 重新思考模型重新基底和线性模态连接性

    Rethink Model Re-Basin and the Linear Mode Connectivity

    [https://arxiv.org/abs/2402.05966](https://arxiv.org/abs/2402.05966)

    本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。

    

    最近的研究表明，对于足够宽的模型来说，大部分随机梯度下降（SGD）的解可以收敛到相同的基底，只是顺序可能不同。这种现象被称为模型重新基底的阶段，对于模型平均化有重要影响。然而，当前的重新基底策略在效果上存在局限性，因为对底层机制的理解不够全面。为了填补这一空白，我们的研究重新审视了标准做法，并揭示了现有匹配算法的频繁不足之处，我们通过适当的重归一化来缓解这些问题。通过引入更直接的分析方法，我们揭示了匹配算法与重归一化过程之间的相互作用。这种观点不仅澄清和改进了以前的研究结果，还促进了新的洞见。例如，它将线性模态连接性与剪枝联系起来，从而激发了一种轻量且有效的后剪枝插件，可以直接与任何现有的剪枝技术合并。

    Recent studies suggest that with sufficiently wide models, most SGD solutions can, up to permutation, converge into the same basin. This phenomenon, known as the model re-basin regime, has significant implications for model averaging. However, current re-basin strategies are limited in effectiveness due to a lack of comprehensive understanding of underlying mechanisms. Addressing this gap, our work revisits standard practices and uncovers the frequent inadequacies of existing matching algorithms, which we show can be mitigated through proper re-normalization. By introducing a more direct analytical approach, we expose the interaction between matching algorithms and re-normalization processes. This perspective not only clarifies and refines previous findings but also facilitates novel insights. For instance, it connects the linear mode connectivity to pruning, motivating a lightweight yet effective post-pruning plug-in that can be directly merged with any existing pruning techniques. Ou
    

