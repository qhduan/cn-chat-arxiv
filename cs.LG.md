# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transfer Learning for T-Cell Response Prediction](https://arxiv.org/abs/2403.12117) | 使用转换器模型进行T细胞响应预测，研究多域结构中的转移学习技术，提出领域感知评估方案。 |
| [^2] | [ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image](https://arxiv.org/abs/2403.09871) | ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。 |
| [^3] | [Meta-Tasks: An alternative view on Meta-Learning Regularization](https://arxiv.org/abs/2402.18599) | 该论文提出了一种新颖的解决方案，通过使用meta-tasks作为元学习正则化的视角，实现了对训练和新颖任务的泛化，避免标记数据稀缺的困扰，并在实验中表现优越，相较于原型网络提高了3.9%的性能。 |
| [^4] | [One-Shot Graph Representation Learning Using Hyperdimensional Computing](https://arxiv.org/abs/2402.17073) | 该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。 |
| [^5] | [Advancing Low-Rank and Local Low-Rank Matrix Approximation in Medical Imaging: A Systematic Literature Review and Future Directions](https://arxiv.org/abs/2402.14045) | 本文系统综述了在医学成像中应用低秩矩阵逼近（LRMA）和其派生物局部LRMA（LLRMA）的作品，并指出自2015年以来医学成像领域开始偏向于使用LLRMA，显示其在捕获医学数据中复杂结构方面的潜力和有效性。 |
| [^6] | [Universal Physics Transformers](https://arxiv.org/abs/2402.12365) | 提出了通用物理变压器（UPTs）这一新颖学习范式，能够模拟广泛的时空问题，同时适用于拉格朗日和欧拉离散化方案，有效地传播动态并允许查询潜在空间 |
| [^7] | [Rethink Model Re-Basin and the Linear Mode Connectivity](https://arxiv.org/abs/2402.05966) | 本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。 |
| [^8] | [Elucidating the solution space of extended reverse-time SDE for diffusion models.](http://arxiv.org/abs/2309.06169) | 这项工作介绍了扩展反向时间随机微分方程（ER SDE）用于解决扩散模型中的采样问题，并提供了精确解和高阶近似解，并解释了在快速采样方面ODE求解器优于SDE求解器的数学洞察力。 |
| [^9] | [Near-Linear Time Projection onto the $\ell_{1,\infty}$ Ball; Application to Sparse Autoencoders.](http://arxiv.org/abs/2307.09836) | 本文提出了一种投影算法，能够在几乎线性时间内将矩阵投影到 $\ell_{1,\infty}$ 球面。该算法易于实现，能够在有限时间内收敛到精确解。同时，将该算法应用于自编码器训练中可以实现特征选择和权重的稀疏化。 |
| [^10] | [Efficient and Accurate Optimal Transport with Mirror Descent and Conjugate Gradients.](http://arxiv.org/abs/2307.08507) | 本文设计了一种借鉴了多个文献的算法，通过使用镜像下降和共轭梯度的技术，能够高效准确地计算Wasserstein距离，并且在高维问题上比其他算法具有快速收敛的优势。 |
| [^11] | [FedICT: Federated Multi-task Distillation for Multi-access Edge Computing.](http://arxiv.org/abs/2301.00389) | FedICT是一种用于多接入边缘计算的联邦多任务蒸馏方法，可以在个性化服务和异构机器学习模型的同时实现高效通信和模型异构性。 |

# 详细

[^1]: 转移学习用于T细胞响应预测

    Transfer Learning for T-Cell Response Prediction

    [https://arxiv.org/abs/2403.12117](https://arxiv.org/abs/2403.12117)

    使用转换器模型进行T细胞响应预测，研究多域结构中的转移学习技术，提出领域感知评估方案。

    

    我们研究特定给定肽段的T细胞响应预测，这可以是向个性化癌症疫苗发展迈出重要一步的关键。

    arXiv:2403.12117v1 Announce Type: cross  Abstract: We study the prediction of T-cell response for specific given peptides, which could, among other applications, be a crucial step towards the development of personalized cancer vaccines. It is a challenging task due to limited, heterogeneous training data featuring a multi-domain structure; such data entail the danger of shortcut learning, where models learn general characteristics of peptide sources, such as the source organism, rather than specific peptide characteristics associated with T-cell response.   Using a transformer model for T-cell response prediction, we show that the danger of inflated predictive performance is not merely theoretical but occurs in practice. Consequently, we propose a domain-aware evaluation scheme. We then study different transfer learning techniques to deal with the multi-domain structure and shortcut learning. We demonstrate a per-source fine tuning approach to be effective across a wide range of peptid
    
[^2]: ThermoHands：一种用于从主观视角热图中估计3D手部姿势的基准

    ThermoHands: A Benchmark for 3D Hand Pose Estimation from Egocentric Thermal Image

    [https://arxiv.org/abs/2403.09871](https://arxiv.org/abs/2403.09871)

    ThermoHands提出了一个新的基准ThermoHands，旨在解决热图中主观视角3D手部姿势估计的挑战，介绍了一个具有双transformer模块的定制基线方法TheFormer，表明热成像在恶劣条件下实现稳健的3D手部姿势估计的有效性。

    

    在这项工作中，我们提出了ThermoHands，这是一个针对基于热图的主观视角3D手部姿势估计的新基准，旨在克服诸如光照变化和遮挡（例如手部穿戴物）等挑战。该基准包括来自28名主体进行手-物体和手-虚拟交互的多样数据集，经过自动化过程准确标注了3D手部姿势。我们引入了一个定制的基线方法TheFormer，利用双transformer模块在热图中实现有效的主观视角3D手部姿势估计。我们的实验结果突显了TheFormer的领先性能，并确认了热成像在实现恶劣条件下稳健的3D手部姿势估计方面的有效性。

    arXiv:2403.09871v1 Announce Type: cross  Abstract: In this work, we present ThermoHands, a new benchmark for thermal image-based egocentric 3D hand pose estimation, aimed at overcoming challenges like varying lighting and obstructions (e.g., handwear). The benchmark includes a diverse dataset from 28 subjects performing hand-object and hand-virtual interactions, accurately annotated with 3D hand poses through an automated process. We introduce a bespoken baseline method, TheFormer, utilizing dual transformer modules for effective egocentric 3D hand pose estimation in thermal imagery. Our experimental results highlight TheFormer's leading performance and affirm thermal imaging's effectiveness in enabling robust 3D hand pose estimation in adverse conditions.
    
[^3]: Meta-Tasks: 元学习正则化的另一种视角

    Meta-Tasks: An alternative view on Meta-Learning Regularization

    [https://arxiv.org/abs/2402.18599](https://arxiv.org/abs/2402.18599)

    该论文提出了一种新颖的解决方案，通过使用meta-tasks作为元学习正则化的视角，实现了对训练和新颖任务的泛化，避免标记数据稀缺的困扰，并在实验中表现优越，相较于原型网络提高了3.9%的性能。

    

    Few-shot learning (FSL)是一个具有挑战性的机器学习问题，因为标记数据稀缺。这篇论文提出了一种新颖的解决方案，可以泛化到训练和新颖的任务，同时利用未标记样本。该方法在更新外层循环之前，使用无监督技术对嵌入模型进行了细化，将其作为“元任务”。实验结果表明，我们提出的方法在新颖和训练任务上表现良好，收敛更快、更好，泛化误差和标准差更低，表明其在FSL中的实际应用潜力。实验结果表明，所提出的方法的表现比原型网络高出3.9%。

    arXiv:2402.18599v1 Announce Type: cross  Abstract: Few-shot learning (FSL) is a challenging machine learning problem due to a scarcity of labeled data. The ability to generalize effectively on both novel and training tasks is a significant barrier to FSL. This paper proposes a novel solution that can generalize to both training and novel tasks while also utilizing unlabeled samples. The method refines the embedding model before updating the outer loop using unsupervised techniques as ``meta-tasks''. The experimental results show that our proposed method performs well on novel and training tasks, with faster and better convergence, lower generalization, and standard deviation error, indicating its potential for practical applications in FSL. The experimental results show that the proposed method outperforms prototypical networks by 3.9%.
    
[^4]: 使用超高维计算进行单次图表示学习

    One-Shot Graph Representation Learning Using Hyperdimensional Computing

    [https://arxiv.org/abs/2402.17073](https://arxiv.org/abs/2402.17073)

    该方法提出了一种使用超高维计算进行单次图表示学习的方法，通过将数据投影到高维空间并利用HD运算符进行信息聚合，实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    

    我们提出了一种新颖、简单、快速、高效的半监督图学习方法。所提方法利用超高维计算，将数据样本使用随机投影编码到高维空间（简称HD空间）。具体来说，我们提出了一种利用图神经网络节点表示的单射性质的超高维图学习（HDGL）算法。HDGL将节点特征映射到HD空间，然后使用HD运算符（如捆绑和绑定）来聚合每个节点的局部邻域信息。对广泛使用的基准数据集进行的实验结果显示，HDGL实现了与最先进深度学习方法相竞争的预测性能，而无需进行计算昂贵的训练。

    arXiv:2402.17073v1 Announce Type: cross  Abstract: We present a novel, simple, fast, and efficient approach for semi-supervised learning on graphs. The proposed approach takes advantage of hyper-dimensional computing which encodes data samples using random projections into a high dimensional space (HD space for short). Specifically, we propose a Hyper-dimensional Graph Learning (HDGL) algorithm that leverages the injectivity property of the node representations of a family of graph neural networks. HDGL maps node features to the HD space and then uses HD operators such as bundling and binding to aggregate information from the local neighborhood of each node. Results of experiments with widely used benchmark data sets show that HDGL achieves predictive performance that is competitive with the state-of-the-art deep learning methods, without the need for computationally expensive training.
    
[^5]: 在医学成像中推进低秩和局部低秩矩阵逼近：系统文献综述与未来方向

    Advancing Low-Rank and Local Low-Rank Matrix Approximation in Medical Imaging: A Systematic Literature Review and Future Directions

    [https://arxiv.org/abs/2402.14045](https://arxiv.org/abs/2402.14045)

    本文系统综述了在医学成像中应用低秩矩阵逼近（LRMA）和其派生物局部LRMA（LLRMA）的作品，并指出自2015年以来医学成像领域开始偏向于使用LLRMA，显示其在捕获医学数据中复杂结构方面的潜力和有效性。

    

    医学成像数据集的大容量和复杂性是存储、传输和处理的瓶颈。为解决这些挑战，低秩矩阵逼近（LRMA）及其派生物局部LRMA（LLRMA）的应用已显示出潜力。本文进行了系统文献综述，展示了在医学成像中应用LRMA和LLRMA的作品。文献的详细分析确认了应用于各种成像模态的LRMA和LLRMA方法。本文解决了现有LRMA和LLRMA方法所面临的挑战和限制。我们注意到，自2015年以来，医学成像领域明显偏向于LLRMA，显示了相对于LRMA在捕获医学数据中复杂结构方面的潜力和有效性。鉴于LLRMA所使用的浅层相似性方法的限制，我们建议使用先进语义图像分割来处理相似性。

    arXiv:2402.14045v1 Announce Type: cross  Abstract: The large volume and complexity of medical imaging datasets are bottlenecks for storage, transmission, and processing. To tackle these challenges, the application of low-rank matrix approximation (LRMA) and its derivative, local LRMA (LLRMA) has demonstrated potential.   This paper conducts a systematic literature review to showcase works applying LRMA and LLRMA in medical imaging. A detailed analysis of the literature identifies LRMA and LLRMA methods applied to various imaging modalities. This paper addresses the challenges and limitations associated with existing LRMA and LLRMA methods.   We note a significant shift towards a preference for LLRMA in the medical imaging field since 2015, demonstrating its potential and effectiveness in capturing complex structures in medical data compared to LRMA. Acknowledging the limitations of shallow similarity methods used with LLRMA, we suggest advanced semantic image segmentation for similarit
    
[^6]: 通用物理变压器

    Universal Physics Transformers

    [https://arxiv.org/abs/2402.12365](https://arxiv.org/abs/2402.12365)

    提出了通用物理变压器（UPTs）这一新颖学习范式，能够模拟广泛的时空问题，同时适用于拉格朗日和欧拉离散化方案，有效地传播动态并允许查询潜在空间

    

    基于深度神经网络的偏微分方程替代者近来引起了越来越多的关注。然而，类似于它们的数值对应物，在不同应用中使用不同的技术，即使系统的基础动态相似。一个著名的例子是在计算流体动力学中的拉格朗日和欧拉表述，这为神经网络有效地建模基于粒子而不是网格的动态构成了挑战。我们引入了通用物理变压器（UPTs），这是一种新颖的学习范式，它模拟了一系列时空问题 - 对拉格朗日和欧拉离散化方案。UPTs在没有基于网格或基于粒子的潜在结构的情况下运行，从而在网格和粒子之间实现了灵活性。UPTs在潜在空间中高效传播动态，强调了逆编码和解码技术。最后，UPTs允许查询潜在空间表现

    arXiv:2402.12365v1 Announce Type: cross  Abstract: Deep neural network based surrogates for partial differential equations have recently gained increased interest. However, akin to their numerical counterparts, different techniques are used across applications, even if the underlying dynamics of the systems are similar. A prominent example is the Lagrangian and Eulerian specification in computational fluid dynamics, posing a challenge for neural networks to effectively model particle- as opposed to grid-based dynamics. We introduce Universal Physics Transformers (UPTs), a novel learning paradigm which models a wide range of spatio-temporal problems - both for Lagrangian and Eulerian discretization schemes. UPTs operate without grid- or particle-based latent structures, enabling flexibility across meshes and particles. UPTs efficiently propagate dynamics in the latent space, emphasized by inverse encoding and decoding techniques. Finally, UPTs allow for queries of the latent space repre
    
[^7]: 重新思考模型重新基底和线性模态连接性

    Rethink Model Re-Basin and the Linear Mode Connectivity

    [https://arxiv.org/abs/2402.05966](https://arxiv.org/abs/2402.05966)

    本论文重新审视了模型重新基底的现象，并发现了现有匹配算法的不足。通过适当的重归一化，我们改进了匹配算法，并揭示了它与重归一化过程的相互作用。这为剪枝提供了新的理解，推动了一种轻量且有效的后剪枝插件的开发。

    

    最近的研究表明，对于足够宽的模型来说，大部分随机梯度下降（SGD）的解可以收敛到相同的基底，只是顺序可能不同。这种现象被称为模型重新基底的阶段，对于模型平均化有重要影响。然而，当前的重新基底策略在效果上存在局限性，因为对底层机制的理解不够全面。为了填补这一空白，我们的研究重新审视了标准做法，并揭示了现有匹配算法的频繁不足之处，我们通过适当的重归一化来缓解这些问题。通过引入更直接的分析方法，我们揭示了匹配算法与重归一化过程之间的相互作用。这种观点不仅澄清和改进了以前的研究结果，还促进了新的洞见。例如，它将线性模态连接性与剪枝联系起来，从而激发了一种轻量且有效的后剪枝插件，可以直接与任何现有的剪枝技术合并。

    Recent studies suggest that with sufficiently wide models, most SGD solutions can, up to permutation, converge into the same basin. This phenomenon, known as the model re-basin regime, has significant implications for model averaging. However, current re-basin strategies are limited in effectiveness due to a lack of comprehensive understanding of underlying mechanisms. Addressing this gap, our work revisits standard practices and uncovers the frequent inadequacies of existing matching algorithms, which we show can be mitigated through proper re-normalization. By introducing a more direct analytical approach, we expose the interaction between matching algorithms and re-normalization processes. This perspective not only clarifies and refines previous findings but also facilitates novel insights. For instance, it connects the linear mode connectivity to pruning, motivating a lightweight yet effective post-pruning plug-in that can be directly merged with any existing pruning techniques. Ou
    
[^8]: 阐明扩展反向时间随机微分方程在扩散模型中的解空间

    Elucidating the solution space of extended reverse-time SDE for diffusion models. (arXiv:2309.06169v1 [cs.LG])

    [http://arxiv.org/abs/2309.06169](http://arxiv.org/abs/2309.06169)

    这项工作介绍了扩展反向时间随机微分方程（ER SDE）用于解决扩散模型中的采样问题，并提供了精确解和高阶近似解，并解释了在快速采样方面ODE求解器优于SDE求解器的数学洞察力。

    

    扩散模型在各种生成建模任务中展示出强大的图像生成能力。然而，它们的主要限制在于采样速度较慢，需要通过大型神经网络进行数百或数千次连续函数评估才能生成高质量的图像。从扩散模型中采样可以看作是解相应的随机微分方程（SDE）或常微分方程（ODE）。在这项工作中，我们将采样过程形式化为扩展反向时间 SDE（ER SDE），将之前对ODE和SDE的探索统一起来。利用ER SDE解的半线性结构，我们为VP SDE提供了精确解和任意高阶近似解，为VE SDE提供了高阶近似解。基于ER SDE的解空间，我们揭示了ODE求解器在快速采样方面优于SDE求解器的数学洞察力。此外，我们还揭示了VP SDE求解器与其VE SDE求解器在性能上相当。

    Diffusion models (DMs) demonstrate potent image generation capabilities in various generative modeling tasks. Nevertheless, their primary limitation lies in slow sampling speed, requiring hundreds or thousands of sequential function evaluations through large neural networks to generate high-quality images. Sampling from DMs can be seen as solving corresponding stochastic differential equations (SDEs) or ordinary differential equations (ODEs). In this work, we formulate the sampling process as an extended reverse-time SDE (ER SDE), unifying prior explorations into ODEs and SDEs. Leveraging the semi-linear structure of ER SDE solutions, we offer exact solutions and arbitrarily high-order approximate solutions for VP SDE and VE SDE, respectively. Based on the solution space of the ER SDE, we yield mathematical insights elucidating the superior performance of ODE solvers over SDE solvers in terms of fast sampling. Additionally, we unveil that VP SDE solvers stand on par with their VE SDE c
    
[^9]: 在几乎线性时间内投影到 $\ell_{1,\infty}$ 球面；稀疏自编码器的应用

    Near-Linear Time Projection onto the $\ell_{1,\infty}$ Ball; Application to Sparse Autoencoders. (arXiv:2307.09836v1 [cs.LG])

    [http://arxiv.org/abs/2307.09836](http://arxiv.org/abs/2307.09836)

    本文提出了一种投影算法，能够在几乎线性时间内将矩阵投影到 $\ell_{1,\infty}$ 球面。该算法易于实现，能够在有限时间内收敛到精确解。同时，将该算法应用于自编码器训练中可以实现特征选择和权重的稀疏化。

    

    现在寻找稀疏性对于加速大规模神经网络的训练至关重要。投影到 $\ell_{1,2}$ 和 $\ell_{1,\infty}$ 是稀疏化和降低神经网络整体成本的最高效技术之一。本文介绍了一种新的 $\ell_{1,\infty}$ 范数球面的投影算法。该算法的最坏时间复杂度为 $\mathcal{O}\big(nm+J\log(nm)\big)$，其中矩阵为 $\mathbb{R}^{n\times m}$。$J$ 是一个在稀疏性高时趋近于0，在稀疏性低时趋近于 $nm$ 的项。该算法易于实现，并保证在有限时间内收敛到精确解。此外，我们提出在训练自编码器时将 $\ell_{1,\infty}$ 球面投影纳入其中，以强制进行特征选择和权重的稀疏化。在我们的生物学应用中，稀疏化主要出现在编码器中，以实现特征选择，因为只有非常小的一部分数据（<2%）是相关的。

    Looking for sparsity is nowadays crucial to speed up the training of large-scale neural networks. Projections onto the $\ell_{1,2}$ and $\ell_{1,\infty}$ are among the most efficient techniques to sparsify and reduce the overall cost of neural networks. In this paper, we introduce a new projection algorithm for the $\ell_{1,\infty}$ norm ball. The worst-case time complexity of this algorithm is $\mathcal{O}\big(nm+J\log(nm)\big)$ for a matrix in $\mathbb{R}^{n\times m}$. $J$ is a term that tends to 0 when the sparsity is high, and to $nm$ when the sparsity is low. Its implementation is easy and it is guaranteed to converge to the exact solution in a finite time. Moreover, we propose to incorporate the $\ell_{1,\infty}$ ball projection while training an autoencoder to enforce feature selection and sparsity of the weights. Sparsification appears in the encoder to primarily do feature selection due to our application in biology, where only a very small part ($<2\%$) of the data is relevan
    
[^10]: 借鉴熵最优输运、镜像下降和共轭梯度的文献，我们设计了一种高效准确的最优输运算法

    Efficient and Accurate Optimal Transport with Mirror Descent and Conjugate Gradients. (arXiv:2307.08507v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.08507](http://arxiv.org/abs/2307.08507)

    本文设计了一种借鉴了多个文献的算法，通过使用镜像下降和共轭梯度的技术，能够高效准确地计算Wasserstein距离，并且在高维问题上比其他算法具有快速收敛的优势。

    

    本文设计了一种新颖的最优输运算法，通过借鉴熵最优输运、镜像下降和共轭梯度的文献。我们的算法可扩展且可在GPU上并行计算，能够以极高的精度计算Wasserstein距离，使相对误差达到$10^{-8}$，并且没有数值稳定性问题。实证上，与包括对数域稳定Sinkhorn算法在内的多种算法相比，我们的算法能够更快地达到高精度解，具有更短的墙钟时间。我们详细地分析了算法和问题参数，并在MNIST图像上进行了基准测试，与各种最新的高维问题算法进行了比较。结果表明我们的算法可以成为从业人员最优输运工具包中有用的补充。

    We design a novel algorithm for optimal transport by drawing from the entropic optimal transport, mirror descent and conjugate gradients literatures. Our scalable and GPU parallelizable algorithm is able to compute the Wasserstein distance with extreme precision, reaching relative error rates of $10^{-8}$ without numerical stability issues. Empirically, the algorithm converges to high precision solutions more quickly in terms of wall-clock time than a variety of algorithms including log-domain stabilized Sinkhorn's Algorithm. We provide careful ablations with respect to algorithm and problem parameters, and present benchmarking over upsampled MNIST images, comparing to various recent algorithms over high-dimensional problems. The results suggest that our algorithm can be a useful addition to the practitioner's optimal transport toolkit.
    
[^11]: FedICT:用于多接入边缘计算的联邦多任务蒸馏

    FedICT: Federated Multi-task Distillation for Multi-access Edge Computing. (arXiv:2301.00389v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.00389](http://arxiv.org/abs/2301.00389)

    FedICT是一种用于多接入边缘计算的联邦多任务蒸馏方法，可以在个性化服务和异构机器学习模型的同时实现高效通信和模型异构性。

    

    对于移动设备智能服务和隐私保护的日益关注，联邦学习在多接入边缘计算（MEC）中得到了广泛应用。多样的用户行为要求在不同设备上使用个性化服务和异构机器学习（ML）模型。提出了联邦多任务学习（FMTL）来为不同设备训练相关但个性化的ML模型，然而之前的工作在训练过程中存在过多的通信开销，并忽视了MEC中设备之间的模型异构性。将知识蒸馏引入FMTL可以同时实现高效的通信和客户端之间的模型异构性，而现有方法依赖于公共数据集，这在实际中是不切实际的。为了解决这个困境，提出了用于多接入边缘计算的联邦多任务蒸馏（FedICT）。

    The growing interest in intelligent services and privacy protection for mobile devices has given rise to the widespread application of federated learning in Multi-access Edge Computing (MEC). Diverse user behaviors call for personalized services with heterogeneous Machine Learning (ML) models on different devices. Federated Multi-task Learning (FMTL) is proposed to train related but personalized ML models for different devices, whereas previous works suffer from excessive communication overhead during training and neglect the model heterogeneity among devices in MEC. Introducing knowledge distillation into FMTL can simultaneously enable efficient communication and model heterogeneity among clients, whereas existing methods rely on a public dataset, which is impractical in reality. To tackle this dilemma, Federated MultI-task Distillation for Multi-access Edge CompuTing (FedICT) is proposed. FedICT direct local-global knowledge aloof during bi-directional distillation processes between 
    

