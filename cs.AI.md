# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection](https://arxiv.org/abs/2403.17465) | LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。 |
| [^2] | [Model Lakes](https://arxiv.org/abs/2403.02327) | 提出了模型湖的概念，在解决大型模型管理中的基础研究挑战方面具有重要意义。 |
| [^3] | [MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations](https://arxiv.org/abs/2402.10093) | MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。 |
| [^4] | [Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment](https://arxiv.org/abs/2402.01830) | 本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。 |
| [^5] | [Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering](https://arxiv.org/abs/2401.06824) | 通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。 |
| [^6] | [GraphFM: Graph Factorization Machines for Feature Interaction Modeling](https://arxiv.org/abs/2105.11866) | 提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。 |
| [^7] | [Invertible Solution of Neural Differential Equations for Analysis of Irregularly-Sampled Time Series.](http://arxiv.org/abs/2401.04979) | 我们提出了一种可逆解决非规则采样时间序列的神经微分方程分析方法，通过引入神经流的概念，我们的方法既保证了可逆性又降低了计算负担，并且在分类和插值任务中表现出了优异的性能。 |
| [^8] | [Towards Robust Probabilistic Modeling on SO(3) via Rotation Laplace Distribution.](http://arxiv.org/abs/2305.10465) | 本文提出了一种新的基于旋转拉普拉斯分布的SO(3)稳健概率建模方法，对异常值具有鲁棒性，并可以容忍不完美的注释。 |
| [^9] | [Graph Neural Diffusion Networks for Semi-supervised Learning.](http://arxiv.org/abs/2201.09698) | 提出了一种名为 GND-Nets 的图神经网络，利用浅层网络和局部、全局邻域信息来解决图半监督学习中的过度平滑和欠平滑问题。 |

# 详细

[^1]: LaRE^2: 基于潜在重构误差的扩散生成图像检测方法

    LaRE^2: Latent Reconstruction Error Based Method for Diffusion-Generated Image Detection

    [https://arxiv.org/abs/2403.17465](https://arxiv.org/abs/2403.17465)

    LaRE^2 提出了一种基于潜在重构误差的方法用于检测扩散生成的图像，通过引入潜在重构误差（LaRE）和误差引导特征细化模块（EGRE）实现了对特征的有效提取和增强，从而区分真实和生成图像。

    

    arXiv:2403.17465v1 类型：交叉 摘要：扩散模型的发展显著提高了图像生成质量，使真实图像和生成图像之间的区分变得越来越困难。尽管这一进展令人印象深刻，但也引发了重要的隐私和安全问题。为了解决这一问题，我们提出了一种新颖的基于潜在重构误差引导特征细化方法（LaRE^2）来检测扩散生成的图像。我们提出了潜在重构误差（LaRE），作为潜在空间中生成图像检测的第一个基于重构误差的特征。LaRE在特征提取效率方面超越了现有方法，同时保留了区分真假所需的关键线索。为了利用LaRE，我们提出了一种误差引导特征细化模块（EGRE），它可以通过LaRE引导的方式细化图像特征，以增强特征的区分能力。

    arXiv:2403.17465v1 Announce Type: cross  Abstract: The evolution of Diffusion Models has dramatically improved image generation quality, making it increasingly difficult to differentiate between real and generated images. This development, while impressive, also raises significant privacy and security concerns. In response to this, we propose a novel Latent REconstruction error guided feature REfinement method (LaRE^2) for detecting the diffusion-generated images. We come up with the Latent Reconstruction Error (LaRE), the first reconstruction-error based feature in the latent space for generated image detection. LaRE surpasses existing methods in terms of feature extraction efficiency while preserving crucial cues required to differentiate between the real and the fake. To exploit LaRE, we propose an Error-Guided feature REfinement module (EGRE), which can refine the image feature guided by LaRE to enhance the discriminativeness of the feature. Our EGRE utilizes an align-then-refine m
    
[^2]: 模型湖

    Model Lakes

    [https://arxiv.org/abs/2403.02327](https://arxiv.org/abs/2403.02327)

    提出了模型湖的概念，在解决大型模型管理中的基础研究挑战方面具有重要意义。

    

    给定一组深度学习模型，寻找适合特定任务的模型、理解这些模型并区分它们之间的差异可能是困难的。目前，从业者依靠手工编写的文档来理解和选择模型。然而，并非所有模型都有完整可靠的文档。随着机器学习模型数量的增加，发现、区分和理解这些模型的问题变得更为重要。受数据湖研究的启发，我们引入并定义了模型湖的概念。我们讨论了在大型模型管理中的基本研究挑战，并探讨了哪些基本的数据管理技术可以应用于大型模型管理的研究中。

    arXiv:2403.02327v1 Announce Type: cross  Abstract: Given a set of deep learning models, it can be hard to find models appropriate to a task, understand the models, and characterize how models are different one from another. Currently, practitioners rely on manually-written documentation to understand and choose models. However, not all models have complete and reliable documentation. As the number of machine learning models increases, this issue of finding, differentiating, and understanding models is becoming more crucial. Inspired from research on data lakes, we introduce and define the concept of model lakes. We discuss fundamental research challenges in the management of large models. And we discuss what principled data management techniques can be brought to bear on the study of large model management.
    
[^3]: MIM-Refiner：一种从中间预训练表示中获得对比学习提升的方法

    MIM-Refiner: A Contrastive Learning Boost from Intermediate Pre-Trained Representations

    [https://arxiv.org/abs/2402.10093](https://arxiv.org/abs/2402.10093)

    MIM-Refiner是一种对比学习提升方法，通过利用MIM模型中的中间层表示和多个对比头，能够将MIM模型的特征从次优的状态提升到最先进的状态，并在ImageNet-1K数据集上取得了新的最先进结果。

    

    我们引入了MIM-Refiner，这是一种用于预训练MIM模型的对比学习提升方法。MIM-Refiner的动机在于MIM模型中的最佳表示通常位于中间层。因此，MIM-Refiner利用连接到不同中间层的多个对比头。在每个头中，修改后的最近邻目标帮助构建相应的语义聚类。此过程短而有效，在几个epochs内，我们将MIM模型的特征从次优的状态提升到最先进的状态。使用data2vec 2.0在ImageNet-1K上预训练的ViT-H经过改进后，在线性探测和低样本分类方面取得了新的最先进结果（分别为84.7%和64.2%），超过了在ImageNet-1K上预训练的其他模型的表现。

    arXiv:2402.10093v1 Announce Type: cross  Abstract: We introduce MIM (Masked Image Modeling)-Refiner, a contrastive learning boost for pre-trained MIM models. The motivation behind MIM-Refiner is rooted in the insight that optimal representations within MIM models generally reside in intermediate layers. Accordingly, MIM-Refiner leverages multiple contrastive heads that are connected to diverse intermediate layers. In each head, a modified nearest neighbor objective helps to construct respective semantic clusters.   The refinement process is short but effective. Within a few epochs, we refine the features of MIM models from subpar to state-of-the-art, off-the-shelf features. Refining a ViT-H, pre-trained with data2vec 2.0 on ImageNet-1K, achieves new state-of-the-art results in linear probing (84.7%) and low-shot classification among models that are pre-trained on ImageNet-1K. In ImageNet-1K 1-shot classification, MIM-Refiner sets a new state-of-the-art of 64.2%, outperforming larger mo
    
[^4]: LLM中的同行评审方法：开放环境下LLMs的自动评估方法

    Peer-review-in-LLMs: Automatic Evaluation Method for LLMs in Open-environment

    [https://arxiv.org/abs/2402.01830](https://arxiv.org/abs/2402.01830)

    本文提出了一种新的无监督评估方法，利用同行评审机制在开放环境中衡量LLMs。通过为每个LLM分配可学习的能力参数，以最大化各个LLM的能力和得分的一致性。结果表明，高层次的LLM能够更准确地评估其他模型的答案，并能够获得更高的响应得分。

    

    现有的大型语言模型（LLMs）评估方法通常集中于在一些有人工注释的封闭环境和特定领域基准上测试性能。本文探索了一种新颖的无监督评估方法，利用同行评审机制自动衡量LLMs。在这个设置中，开源和闭源的LLMs处于同一环境中，能够回答未标记的问题并互相评估，每个LLM的响应得分由其他匿名的LLMs共同决定。为了获取这些模型之间的能力层次结构，我们为每个LLM分配一个可学习的能力参数来调整最终排序结果。我们将其形式化为一个受约束的优化问题，旨在最大化每个LLM的能力和得分的一致性。背后的关键假设是高层次的LLM能够比低层次的LLM更准确地评估其他模型的答案，而高层次的LLM也可以达到较高的响应得分。

    Existing large language models (LLMs) evaluation methods typically focus on testing the performance on some closed-environment and domain-specific benchmarks with human annotations. In this paper, we explore a novel unsupervised evaluation direction, utilizing peer-review mechanisms to measure LLMs automatically. In this setting, both open-source and closed-source LLMs lie in the same environment, capable of answering unlabeled questions and evaluating each other, where each LLM's response score is jointly determined by other anonymous ones. To obtain the ability hierarchy among these models, we assign each LLM a learnable capability parameter to adjust the final ranking. We formalize it as a constrained optimization problem, intending to maximize the consistency of each LLM's capabilities and scores. The key assumption behind is that high-level LLM can evaluate others' answers more accurately than low-level ones, while higher-level LLM can also achieve higher response scores. Moreover
    
[^5]: 打开LLMs的潘多拉魔盒：通过表示工程对LLMs进行越狱

    Open the Pandora's Box of LLMs: Jailbreaking LLMs through Representation Engineering

    [https://arxiv.org/abs/2401.06824](https://arxiv.org/abs/2401.06824)

    通过表示工程对LLMs进行越狱是一种新颖的方法，它利用少量查询对提取“安全模式”，成功规避目标模型的防御，实现了前所未有的越狱性能。

    

    越狱技术旨在通过诱使大型语言模型（LLMs）生成对恶意查询产生有毒响应，来探索LLMs安全性边界，这在LLMs社区内是一个重要关注点。我们提出一种名为通过表示工程对LLMs进行越狱（Jailbreaking LLMs through Representation Engineering，JRE）的新颖越狱方法，其仅需要少量查询对以提取可用于规避目标模型防御的“安全模式”，实现了前所未有的越狱性能。

    arXiv:2401.06824v2 Announce Type: replace-cross  Abstract: Jailbreaking techniques aim to probe the boundaries of safety in large language models (LLMs) by inducing them to generate toxic responses to malicious queries, a significant concern within the LLM community. While existing jailbreaking methods primarily rely on prompt engineering, altering inputs to evade LLM safety mechanisms, they suffer from low attack success rates and significant time overheads, rendering them inflexible. To overcome these limitations, we propose a novel jailbreaking approach, named Jailbreaking LLMs through Representation Engineering (JRE). Our method requires only a small number of query pairs to extract ``safety patterns'' that can be used to circumvent the target model's defenses, achieving unprecedented jailbreaking performance. Building upon these findings, we also introduce a novel defense framework inspired by JRE principles, which demonstrates notable effectiveness. Extensive experimentation conf
    
[^6]: GraphFM：图因子分解机用于特征交互建模

    GraphFM: Graph Factorization Machines for Feature Interaction Modeling

    [https://arxiv.org/abs/2105.11866](https://arxiv.org/abs/2105.11866)

    提出了一种名为GraphFM的图因子分解机方法，通过图结构自然表示特征，并将FM的交互功能集成到GNN的特征聚合策略中，能够模拟任意阶特征交互。

    

    因子分解机（FM）是处理高维稀疏数据时建模成对（二阶）特征交互的一种常见方法。然而，一方面，FM未能捕捉到高阶特征交互，受到组合扩展的影响。另一方面，考虑每对特征之间的交互可能会引入噪声并降低预测准确性。为了解决这些问题，我们提出了一种新方法，称为Graph Factorization Machine（GraphFM），通过将特征自然表示成图结构。具体而言，我们设计了一种机制来选择有益的特征交互，并将其形式化为特征之间的边。然后，所提出的模型将FM的交互功能整合到图神经网络（GNN）的特征聚合策略中，通过堆叠层来模拟图结构特征上的任意阶特征交互。

    arXiv:2105.11866v4 Announce Type: replace-cross  Abstract: Factorization machine (FM) is a prevalent approach to modeling pairwise (second-order) feature interactions when dealing with high-dimensional sparse data. However, on the one hand, FM fails to capture higher-order feature interactions suffering from combinatorial expansion. On the other hand, taking into account interactions between every pair of features may introduce noise and degrade prediction accuracy. To solve the problems, we propose a novel approach, Graph Factorization Machine (GraphFM), by naturally representing features in the graph structure. In particular, we design a mechanism to select the beneficial feature interactions and formulate them as edges between features. Then the proposed model, which integrates the interaction function of FM into the feature aggregation strategy of Graph Neural Network (GNN), can model arbitrary-order feature interactions on the graph-structured features by stacking layers. Experime
    
[^7]: 可逆解决非规则采样时间序列的神经微分方程分析方法

    Invertible Solution of Neural Differential Equations for Analysis of Irregularly-Sampled Time Series. (arXiv:2401.04979v1 [cs.LG])

    [http://arxiv.org/abs/2401.04979](http://arxiv.org/abs/2401.04979)

    我们提出了一种可逆解决非规则采样时间序列的神经微分方程分析方法，通过引入神经流的概念，我们的方法既保证了可逆性又降低了计算负担，并且在分类和插值任务中表现出了优异的性能。

    

    为了处理非规则和不完整的时间序列数据的复杂性，我们提出了一种基于神经微分方程（NDE）的可逆解决方案。虽然基于NDE的方法是分析非规则采样时间序列的一种强大方法，但它们通常不能保证在其标准形式下进行可逆变换。我们的方法建议使用具有神经流的神经控制微分方程（Neural CDEs）的变种，该方法在保持较低的计算负担的同时确保了可逆性。此外，它还可以训练双重潜在空间，增强了对动态时间动力学的建模能力。我们的研究提出了一个先进的框架，在分类和插值任务中都表现出色。我们方法的核心是一个经过精心设计的增强型双重潜在状态架构，用于在各种时间序列任务中提高精度。实证分析表明，我们的方法明显优于现有模型。

    To handle the complexities of irregular and incomplete time series data, we propose an invertible solution of Neural Differential Equations (NDE)-based method. While NDE-based methods are a powerful method for analyzing irregularly-sampled time series, they typically do not guarantee reversible transformations in their standard form. Our method suggests the variation of Neural Controlled Differential Equations (Neural CDEs) with Neural Flow, which ensures invertibility while maintaining a lower computational burden. Additionally, it enables the training of a dual latent space, enhancing the modeling of dynamic temporal dynamics. Our research presents an advanced framework that excels in both classification and interpolation tasks. At the core of our approach is an enhanced dual latent states architecture, carefully designed for high precision across various time series tasks. Empirical analysis demonstrates that our method significantly outperforms existing models. This work significan
    
[^8]: 基于旋转拉普拉斯分布的SO(3)稳健概率建模研究

    Towards Robust Probabilistic Modeling on SO(3) via Rotation Laplace Distribution. (arXiv:2305.10465v1 [cs.CV])

    [http://arxiv.org/abs/2305.10465](http://arxiv.org/abs/2305.10465)

    本文提出了一种新的基于旋转拉普拉斯分布的SO(3)稳健概率建模方法，对异常值具有鲁棒性，并可以容忍不完美的注释。

    

    从单张RGB图像估计三维自由旋转是一项重要且具有挑战性的任务。概率旋转建模是一种流行的方法，相对于单预测旋转回归可以额外提供预测不确定性信息。对于SO(3)上的概率分布建模，使用类似于高斯的Bingham分布和矩阵Fisher分布是自然的，但是它们对异常预测很敏感，例如180度误差，因此不太可能以最佳性能收敛。本文从多元拉普拉斯分布中汲取灵感，提出了一种新的SO(3)旋转拉普拉斯分布。我们的旋转拉普拉斯分布对异常值的干扰具有鲁棒性，并强制施加梯度到低误差区域，以改进性能。此外，我们还证明了我们的方法对小噪声具有鲁棒性，因此可以容忍不完美的注释。利用这个优势，我们展示了在半监督回归任务上的优势。

    Estimating the 3DoF rotation from a single RGB image is an important yet challenging problem. As a popular approach, probabilistic rotation modeling additionally carries prediction uncertainty information, compared to single-prediction rotation regression. For modeling probabilistic distribution over SO(3), it is natural to use Gaussian-like Bingham distribution and matrix Fisher, however they are shown to be sensitive to outlier predictions, e.g. $180^\circ$ error and thus are unlikely to converge with optimal performance. In this paper, we draw inspiration from multivariate Laplace distribution and propose a novel rotation Laplace distribution on SO(3). Our rotation Laplace distribution is robust to the disturbance of outliers and enforces much gradient to the low-error region that it can improve. In addition, we show that our method also exhibits robustness to small noises and thus tolerates imperfect annotations. With this benefit, we demonstrate its advantages in semi-supervised r
    
[^9]: 图神经扩散网络用于半监督学习

    Graph Neural Diffusion Networks for Semi-supervised Learning. (arXiv:2201.09698v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2201.09698](http://arxiv.org/abs/2201.09698)

    提出了一种名为 GND-Nets 的图神经网络，利用浅层网络和局部、全局邻域信息来解决图半监督学习中的过度平滑和欠平滑问题。

    

    图卷积网络 (GCN) 是用于基于图的半监督学习的先驱模型。然而，GCN 在标记稀疏的图上表现不佳。其两层版本不能有效地将标签信息传播到整个图结构（即欠平滑问题），而其深层版本则过度平滑且难以训练（即过度平滑问题）。为了解决这两个问题，我们提出了一种新的图神经网络，称为 GND-Nets（图神经扩散网络），它在单层中利用了顶点的局部和全局邻域信息。利用浅层网络可以缓解过度平滑问题，而利用局部和全局邻域信息可以缓解欠平滑问题。顶点的局部和全局邻域信息的利用是通过一种称为神经扩散的新图扩散方法实现的，该方法将神经网络融入传统的线性和非线性图扩散中。

    Graph Convolutional Networks (GCN) is a pioneering model for graph-based semi-supervised learning. However, GCN does not perform well on sparsely-labeled graphs. Its two-layer version cannot effectively propagate the label information to the whole graph structure (i.e., the under-smoothing problem) while its deep version over-smoothens and is hard to train (i.e., the over-smoothing problem). To solve these two issues, we propose a new graph neural network called GND-Nets (for Graph Neural Diffusion Networks) that exploits the local and global neighborhood information of a vertex in a single layer. Exploiting the shallow network mitigates the over-smoothing problem while exploiting the local and global neighborhood information mitigates the under-smoothing problem. The utilization of the local and global neighborhood information of a vertex is achieved by a new graph diffusion method called neural diffusions, which integrate neural networks into the conventional linear and nonlinear gra
    

