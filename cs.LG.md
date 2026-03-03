# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion-Driven Domain Adaptation for Generating 3D Molecules](https://arxiv.org/abs/2404.00962) | 该研究提出了一种基于扩散的领域自适应分子生成方法 GADM，可以在不需要收集数据的情况下将生成模型迁移到新领域，通过利用等变蒙板自编码器和各种掩蔽策略来捕获结构变化所带来的领域差异，并能够泛化到目标领域中看不见的结构变化。 |
| [^2] | [Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models](https://arxiv.org/abs/2402.06223) | 通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。 |
| [^3] | [Topologically-Regularized Multiple Instance Learning for Red Blood Cell Disease Classification.](http://arxiv.org/abs/2307.14025) | 本论文提出一种基于拓扑正则化的多实例学习方法，用于罕见贫血疾病的红细胞分类。通过从单个红细胞图像中提取多尺度的拓扑特征来进行模型正则化，以保持数据的特征拓扑属性。实验结果表明，该方法是有效的。 |
| [^4] | [NUBO: A Transparent Python Package for Bayesian Optimisation.](http://arxiv.org/abs/2305.06709) | NUBO是一个透明的Python包，用于优化昂贵的黑盒函数，它利用高斯过程做代理模型以及获取函数来指导选择候选点，专注于透明度和用户体验。 |
| [^5] | [FedHB: Hierarchical Bayesian Federated Learning.](http://arxiv.org/abs/2305.04979) | 该论文提出了一种层次贝叶斯联邦学习方法，通过块坐标下降分布式算法实现对客户端私有数据不透露的学习，在收敛速度上与正则化相同。 |
| [^6] | [Impossibility of Depth Reduction in Explainable Clustering.](http://arxiv.org/abs/2305.02850) | 可解释聚类中，决策树深度是无法减少的固有复杂度度量之一，减少深度会显著降低聚类质量。 |

# 详细

[^1]: 基于扩散驱动的领域自适应生成3D分子

    Diffusion-Driven Domain Adaptation for Generating 3D Molecules

    [https://arxiv.org/abs/2404.00962](https://arxiv.org/abs/2404.00962)

    该研究提出了一种基于扩散的领域自适应分子生成方法 GADM，可以在不需要收集数据的情况下将生成模型迁移到新领域，通过利用等变蒙板自编码器和各种掩蔽策略来捕获结构变化所带来的领域差异，并能够泛化到目标领域中看不见的结构变化。

    

    我们能否训练一个分子生成器，可以生成来自新领域的3D分子，从而避免收集数据的需求？这个问题可以被视为领域自适应分子生成的问题。本文提出了一种新颖且基于原则的基于扩散的方法，称为GADM，它允许将生成模型移至所需的新领域，而无需收集任何一个分子。由于领域转移通常由分子的结构变化引起，例如骨架变化，我们利用指定的等变蒙板自编码器（MAE）以及各种掩蔽策略来捕获领域内变体的结构精细表示。特别是，通过一个不对称的编码器-解码器模块，MAE可以泛化到目标领域中看不见的结构变化。这些结构变化被编码为等变编码器，并被视为领域监督员来控制去噪。

    arXiv:2404.00962v1 Announce Type: new  Abstract: Can we train a molecule generator that can generate 3D molecules from a new domain, circumventing the need to collect data? This problem can be cast as the problem of domain adaptive molecule generation. This work presents a novel and principled diffusion-based approach, called GADM, that allows shifting a generative model to desired new domains without the need to collect even a single molecule. As the domain shift is typically caused by the structure variations of molecules, e.g., scaffold variations, we leverage a designated equivariant masked autoencoder (MAE) along with various masking strategies to capture the structural-grained representations of the in-domain varieties. In particular, with an asymmetric encoder-decoder module, the MAE can generalize to unseen structure variations from the target domains. These structure variations are encoded with an equivariant encoder and treated as domain supervisors to control denoising. We s
    
[^2]: 通过潜在部分因果模型揭示多模式对比表示学习

    Revealing Multimodal Contrastive Representation Learning through Latent Partial Causal Models

    [https://arxiv.org/abs/2402.06223](https://arxiv.org/abs/2402.06223)

    通过潜在部分因果模型，我们展示了多模式对比表示学习在识别潜在耦合变量方面的优秀能力，并揭示了预训练的多模态模型通过线性独立分量分析学习分离表示的潜力。

    

    多模式对比表示学习方法在各个领域取得了成功，部分原因是由于它们能够生成复杂现象的有意义的共享表示。为了增强对这些获得的表示的深度分析和理解，我们引入了一种特别针对多模态数据设计的统一因果模型。通过研究这个模型，我们展示了多模式对比表示学习在识别在提出的统一模型中的潜在耦合变量方面的优秀能力，即使在不同假设下导致的线性或置换变换。我们的发现揭示了预训练的多模态模型（如CLIP）通过线性独立分量分析这一令人惊讶的简单而高效的工具学习分离表示的潜力。实验证明了我们发现的鲁棒性，即使在被违反假设的情况下，也验证了所提出方法在学习疾病方面的有效性。

    Multimodal contrastive representation learning methods have proven successful across a range of domains, partly due to their ability to generate meaningful shared representations of complex phenomena. To enhance the depth of analysis and understanding of these acquired representations, we introduce a unified causal model specifically designed for multimodal data. By examining this model, we show that multimodal contrastive representation learning excels at identifying latent coupled variables within the proposed unified model, up to linear or permutation transformations resulting from different assumptions. Our findings illuminate the potential of pre-trained multimodal models, eg, CLIP, in learning disentangled representations through a surprisingly simple yet highly effective tool: linear independent component analysis. Experiments demonstrate the robustness of our findings, even when the assumptions are violated, and validate the effectiveness of the proposed method in learning dise
    
[^3]: 基于拓扑正则化的多实例学习用于红细胞疾病分类

    Topologically-Regularized Multiple Instance Learning for Red Blood Cell Disease Classification. (arXiv:2307.14025v1 [cs.LG])

    [http://arxiv.org/abs/2307.14025](http://arxiv.org/abs/2307.14025)

    本论文提出一种基于拓扑正则化的多实例学习方法，用于罕见贫血疾病的红细胞分类。通过从单个红细胞图像中提取多尺度的拓扑特征来进行模型正则化，以保持数据的特征拓扑属性。实验结果表明，该方法是有效的。

    

    使用显微图像诊断罕见的贫血疾病对于熟练的专家和机器学习方法来说都具有挑战性。由于在单个血样中有数千个与疾病相关的细胞，这构成了一个复杂的多实例学习（MIL）问题。虽然红细胞的空间邻域本身并不重要，但整个血样的拓扑结构，即数据的几何性质，包含了有益的特征，以解决典型的MIL问题，如梯度消失和在有限数据上训练时的过拟合。因此，我们开发了一种基于拓扑的方法，从单个红细胞图像的包中提取多尺度的拓扑特征。这些拓扑特征被用来对模型进行正则化，强制保持数据的特征拓扑属性。在包含71个罕见贫血疾病患者的数据集上，包括521张红细胞显微图像，我们的实验表明拓扑正则化是一个有效的方法。

    Diagnosing rare anemia disorders using microscopic images is challenging for skilled specialists and machine-learning methods alike. Due to thousands of disease-relevant cells in a single blood sample, this constitutes a complex multiple-instance learning (MIL) problem. While the spatial neighborhood of red blood cells is not meaningful per se, the topology, i.e., the geometry of blood samples as a whole, contains informative features to remedy typical MIL issues, such as vanishing gradients and overfitting when training on limited data. We thus develop a topology-based approach that extracts multi-scale topological features from bags of single red blood cell images. The topological features are used to regularize the model, enforcing the preservation of characteristic topological properties of the data. Applied to a dataset of 71 patients suffering from rare anemia disorders with 521 microscopic images of red blood cells, our experiments show that topological regularization is an effe
    
[^4]: NUBO：一个透明的 Python 包用于贝叶斯优化

    NUBO: A Transparent Python Package for Bayesian Optimisation. (arXiv:2305.06709v1 [cs.LG])

    [http://arxiv.org/abs/2305.06709](http://arxiv.org/abs/2305.06709)

    NUBO是一个透明的Python包，用于优化昂贵的黑盒函数，它利用高斯过程做代理模型以及获取函数来指导选择候选点，专注于透明度和用户体验。

    

    NUBO（Newcastle University Bayesian Optimisation）是一个贝叶斯优化框架，用于优化昂贵的黑盒函数，比如物理实验和计算机模拟器。它利用高斯过程做代理模型、并通过获取函数来选择用于全局最优化的候选点。NUBO专注于透明度和用户体验，以便让不同领域的研究人员更容易使用贝叶斯优化。

    NUBO, short for Newcastle University Bayesian Optimisation, is a Bayesian optimisation framework for the optimisation of expensive-to-evaluate black-box functions, such as physical experiments and computer simulators. Bayesian optimisation is a cost-efficient optimisation strategy that uses surrogate modelling via Gaussian processes to represent an objective function and acquisition functions to guide the selection of candidate points to approximate the global optimum of the objective function. NUBO itself focuses on transparency and user experience to make Bayesian optimisation easily accessible to researchers from all disciplines. Clean and understandable code, precise references, and thorough documentation ensure transparency, while user experience is ensured by a modular and flexible design, easy-to-write syntax, and careful selection of Bayesian optimisation algorithms. NUBO allows users to tailor Bayesian optimisation to their specific problem by writing the optimisation loop the
    
[^5]: FedHB: 层次贝叶斯联邦学习

    FedHB: Hierarchical Bayesian Federated Learning. (arXiv:2305.04979v1 [cs.LG])

    [http://arxiv.org/abs/2305.04979](http://arxiv.org/abs/2305.04979)

    该论文提出了一种层次贝叶斯联邦学习方法，通过块坐标下降分布式算法实现对客户端私有数据不透露的学习，在收敛速度上与正则化相同。

    

    本文提出了一种新的层次贝叶斯联邦学习方法，通过层次贝叶斯建模合理地描述了客户端本地数据的生成过程：构成客户端本地模型的随机变量，由更高水平的全局变量进行控制。有趣的是，我们贝叶斯模型中的变分推断导致了一个优化问题，其块坐标下降求解成为一个可分客户端的分布式算法，这使得客户端完全不需要透露自己的私有数据，因此与联邦学习完全兼容。我们还强调，我们的块坐标算法具有特定形式，包括Fed-Avg和Fed-Prox在内的众所周知的FL算法都可以作为其特例进行子归。除了引入新的建模和导出之外，我们还提供了收敛性分析，表明我们的块坐标FL算法以$O(1/\sqrt{t})$的速度收敛到目标的（本地）最优解，这与正则化具有相同的速率。

    We propose a novel hierarchical Bayesian approach to Federated Learning (FL), where our model reasonably describes the generative process of clients' local data via hierarchical Bayesian modeling: constituting random variables of local models for clients that are governed by a higher-level global variate. Interestingly, the variational inference in our Bayesian model leads to an optimisation problem whose block-coordinate descent solution becomes a distributed algorithm that is separable over clients and allows them not to reveal their own private data at all, thus fully compatible with FL. We also highlight that our block-coordinate algorithm has particular forms that subsume the well-known FL algorithms including Fed-Avg and Fed-Prox as special cases. Beyond introducing novel modeling and derivations, we also offer convergence analysis showing that our block-coordinate FL algorithm converges to an (local) optimum of the objective at the rate of $O(1/\sqrt{t})$, the same rate as regul
    
[^6]: 可解释聚类中深度减少的不可能性证明

    Impossibility of Depth Reduction in Explainable Clustering. (arXiv:2305.02850v1 [cs.LG])

    [http://arxiv.org/abs/2305.02850](http://arxiv.org/abs/2305.02850)

    可解释聚类中，决策树深度是无法减少的固有复杂度度量之一，减少深度会显著降低聚类质量。

    

    近年来，可解释聚类引起了许多关注。本论文在Euclidean平面中证明，对于可解释的k-means和k-median聚类问题，决策树的深度是不可避免的复杂度度量之一，无法减少而不显著降低聚类质量。我们证明了对于任何在Euclidean平面上的数据X，深度为k-1的决策树的k-means/k-median聚类代价与X的优化聚类代价相同，但是对于深度小于k-1的决策树，其聚类代价相对于最优聚类代价而言是不可接受的。我们还将结果扩展到了k-center目标。

    Over the last few years Explainable Clustering has gathered a lot of attention. Dasgupta et al. [ICML'20] initiated the study of explainable k-means and k-median clustering problems where the explanation is captured by a threshold decision tree which partitions the space at each node using axis parallel hyperplanes. Recently, Laber et al. [Pattern Recognition'23] made a case to consider the depth of the decision tree as an additional complexity measure of interest.  In this work, we prove that even when the input points are in the Euclidean plane, then any depth reduction in the explanation incurs unbounded loss in the k-means and k-median cost. Formally, we show that there exists a data set X in the Euclidean plane, for which there is a decision tree of depth k-1 whose k-means/k-median cost matches the optimal clustering cost of X, but every decision tree of depth less than k-1 has unbounded cost w.r.t. the optimal cost of clustering. We extend our results to the k-center objective as
    

