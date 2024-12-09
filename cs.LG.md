# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Colour and Brush Stroke Pattern Recognition in Abstract Art using Modified Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/2403.18397) | 本文通过引入改进的深度卷积生成对抗网络(mDCGAN)，针对高质量艺术品生成进行了研究，解决了普遍训练问题，有效探索抽象绘画中的颜色和笔触模式。 |
| [^2] | [Machine Unlearning by Suppressing Sample Contribution](https://arxiv.org/abs/2402.15109) | 本文提出了一种机器遗忘方法，通过最小化输入敏感度来抑制遗忘数据的贡献，并在实验中表现出优异的性能。 |
| [^3] | [Mixup Barcodes: Quantifying Geometric-Topological Interactions between Point Clouds](https://arxiv.org/abs/2402.15058) | 提出了一种名为混合条形码的新方法，利用标准持久同调与图像持久同调结合，可以量化任意维度两个点集之间的几何-拓扑相互作用，以及引入简单的统计量来量化这种相互作用的复杂性。 |
| [^4] | [Masked Attention is All You Need for Graphs](https://arxiv.org/abs/2402.10793) | 提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。 |
| [^5] | [Graph Inference Acceleration by Learning MLPs on Graphs without Supervision](https://arxiv.org/abs/2402.08918) | 该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。 |
| [^6] | [Voronoi Candidates for Bayesian Optimization](https://arxiv.org/abs/2402.04922) | 使用Voronoi候选点边界可以在贝叶斯优化中有效地优化黑盒函数，提高了多起始连续搜索的执行时间。 |
| [^7] | [PAC Privacy Preserving Diffusion Models](https://arxiv.org/abs/2312.01201) | 提出了一种PAC隐私保护扩散模型，通过将私有分类器指导集成到采样过程中增强隐私保护，并发展了一种新的度量标准来衡量隐私水平，在保护性能方面表现出卓越表现。 |
| [^8] | [Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models.](http://arxiv.org/abs/2310.12000) | 这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。 |
| [^9] | [Memorization with neural nets: going beyond the worst case.](http://arxiv.org/abs/2310.00327) | 本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。 |
| [^10] | [What can we learn from quantum convolutional neural networks?.](http://arxiv.org/abs/2308.16664) | 通过分析量子卷积神经网络（QCNNs），我们发现它们通过隐藏特征映射嵌入物理系统参数，并且利用量子临界性生成适合的基函数集，池化层选择能够形成高性能决策边界的基函数，而模型的泛化性能依赖于嵌入类型。 |
| [^11] | [A Simple Data Augmentation for Feature Distribution Skewed Federated Learning.](http://arxiv.org/abs/2306.09363) | 本文针对特征分布偏斜的联邦学习提出了FedRDN方法，在输入层级上实现了数据增强，将整个联邦数据集的统计信息注入到本地客户端数据中，以缓解特征漂移问题。 |
| [^12] | [The Score-Difference Flow for Implicit Generative Modeling.](http://arxiv.org/abs/2304.12906) | 本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。 |

# 详细

[^1]: 使用改进的深度卷积生成对抗网络在抽象艺术中进行颜色和笔触模式识别

    Colour and Brush Stroke Pattern Recognition in Abstract Art using Modified Deep Convolutional Generative Adversarial Networks

    [https://arxiv.org/abs/2403.18397](https://arxiv.org/abs/2403.18397)

    本文通过引入改进的深度卷积生成对抗网络(mDCGAN)，针对高质量艺术品生成进行了研究，解决了普遍训练问题，有效探索抽象绘画中的颜色和笔触模式。

    

    抽象艺术是一种广受欢迎、被广泛讨论的艺术形式，通常能够描绘出艺术家的情感。许多研究人员尝试使用机器学习和深度学习的边缘检测、笔触和情感识别算法来研究抽象艺术。本文描述了使用生成对抗神经网络(GAN)对广泛分布的抽象绘画进行研究。 GAN具有学习和再现分布的能力，使研究人员能够有效地探索和研究生成的图像空间。然而，挑战在于开发一种能够克服常见训练问题的高效GAN架构。本文通过引入专门设计用于高质量艺术品生成的改进DCGAN(mDCGAN)来解决这一挑战。该方法涉及对所做修改的深入探讨，深入研究DCGAN的复杂工作。

    arXiv:2403.18397v1 Announce Type: cross  Abstract: Abstract Art is an immensely popular, discussed form of art that often has the ability to depict the emotions of an artist. Many researchers have made attempts to study abstract art in the form of edge detection, brush stroke and emotion recognition algorithms using machine and deep learning. This papers describes the study of a wide distribution of abstract paintings using Generative Adversarial Neural Networks(GAN). GANs have the ability to learn and reproduce a distribution enabling researchers and scientists to effectively explore and study the generated image space. However, the challenge lies in developing an efficient GAN architecture that overcomes common training pitfalls. This paper addresses this challenge by introducing a modified-DCGAN (mDCGAN) specifically designed for high-quality artwork generation. The approach involves a thorough exploration of the modifications made, delving into the intricate workings of DCGANs, opt
    
[^2]: 抑制样本贡献的机器遗忘

    Machine Unlearning by Suppressing Sample Contribution

    [https://arxiv.org/abs/2402.15109](https://arxiv.org/abs/2402.15109)

    本文提出了一种机器遗忘方法，通过最小化输入敏感度来抑制遗忘数据的贡献，并在实验中表现出优异的性能。

    

    机器遗忘（MU）是指从经过良好训练的模型中删除数据，这在实践中非常重要，因为涉及“被遗忘的权利”。本文从训练数据和未见数据对模型贡献的基本区别入手：训练数据对最终模型有贡献，而未见数据没有。我们理论上发现输入敏感度可以近似衡量贡献，并实际设计了一种算法，称为MU-Mis（通过最小化输入敏感度进行机器遗忘），来抑制遗忘数据的贡献。实验结果表明，MU-Mis明显优于最先进的MU方法。此外，MU-Mis与MU的应用更加密切，因为它不需要使用剩余数据。

    arXiv:2402.15109v1 Announce Type: new  Abstract: Machine Unlearning (MU) is to forget data from a well-trained model, which is practically important due to the "right to be forgotten". In this paper, we start from the fundamental distinction between training data and unseen data on their contribution to the model: the training data contributes to the final model while the unseen data does not. We theoretically discover that the input sensitivity can approximately measure the contribution and practically design an algorithm, called MU-Mis (machine unlearning via minimizing input sensitivity), to suppress the contribution of the forgetting data. Experimental results demonstrate that MU-Mis outperforms state-of-the-art MU methods significantly. Additionally, MU-Mis aligns more closely with the application of MU as it does not require the use of remaining data.
    
[^3]: 混合条形码：量化点云之间的几何-拓扑相互作用

    Mixup Barcodes: Quantifying Geometric-Topological Interactions between Point Clouds

    [https://arxiv.org/abs/2402.15058](https://arxiv.org/abs/2402.15058)

    提出了一种名为混合条形码的新方法，利用标准持久同调与图像持久同调结合，可以量化任意维度两个点集之间的几何-拓扑相互作用，以及引入简单的统计量来量化这种相互作用的复杂性。

    

    我们将标准持久同调与图像持久同调相结合，定义了一种新颖的表征形状和它们之间相互作用的方法。具体而言，我们介绍了：（1）混合条形码，捕捉任意维度两个点集之间的几何-拓扑相互作用（混合）；（2）简单的总混合和总百分比混合统计量，作为一个单一数字来量化相互作用的复杂性；（3）一个用于操作上述工具的软件工具。作为一个概念验证，我们将该工具应用到一个源自机器学习的问题上。具体地，我们研究了不同类别嵌入的可分离性。结果表明，拓扑混合是一种用于表征低维和高维数据交互的有效方法。与持久同调的典型用法相比，这个新工具对于拓扑特征的几何位置更为敏感，这通常是可取的。

    arXiv:2402.15058v1 Announce Type: cross  Abstract: We combine standard persistent homology with image persistent homology to define a novel way of characterizing shapes and interactions between them. In particular, we introduce: (1) a mixup barcode, which captures geometric-topological interactions (mixup) between two point sets in arbitrary dimension; (2) simple summary statistics, total mixup and total percentage mixup, which quantify the complexity of the interactions as a single number; (3) a software tool for playing with the above.   As a proof of concept, we apply this tool to a problem arising from machine learning. In particular, we study the disentanglement in embeddings of different classes. The results suggest that topological mixup is a useful method for characterizing interactions for low and high-dimensional data. Compared to the typical usage of persistent homology, the new tool is sensitive to the geometric locations of the topological features, which is often desirabl
    
[^4]: 掩码注意力是图的关键

    Masked Attention is All You Need for Graphs

    [https://arxiv.org/abs/2402.10793](https://arxiv.org/abs/2402.10793)

    提出了一种在图上学习的简单替代方法，称为掩码注意力（MAG），其利用注意力矩阵来创建定制的注意力模式，在长距离任务上表现出色并胜过其他方法。

    

    图神经网络（GNNs）和消息传递算法的变种主要用于在图上学习，这在很大程度上归功于它们的灵活性、速度和令人满意的性能。然而，设计强大而通用的GNNs需要大量的研究工作，通常依赖于精心选择的手工制作的消息传递操作符。受此启发，我们提出了一种在图上学习的非常简单的替代方法，它完全依赖于注意力。图被表示为节点或边集，并通过掩码注意权重矩阵来强制它们的连接，有效地为每个图创建定制的注意力模式。尽管其简单性，用于图的掩码注意力（MAG）在长距离任务上表现出色，并在55多个节点和图级任务上优于强消息传递基线和更复杂的基于注意力的方法。

    arXiv:2402.10793v1 Announce Type: cross  Abstract: Graph neural networks (GNNs) and variations of the message passing algorithm are the predominant means for learning on graphs, largely due to their flexibility, speed, and satisfactory performance. The design of powerful and general purpose GNNs, however, requires significant research efforts and often relies on handcrafted, carefully-chosen message passing operators. Motivated by this, we propose a remarkably simple alternative for learning on graphs that relies exclusively on attention. Graphs are represented as node or edge sets and their connectivity is enforced by masking the attention weight matrix, effectively creating custom attention patterns for each graph. Despite its simplicity, masked attention for graphs (MAG) has state-of-the-art performance on long-range tasks and outperforms strong message passing baselines and much more involved attention-based methods on over 55 node and graph-level tasks. We also show significantly 
    
[^5]: 通过无监督在图上学习多层感知机（MLP）加速图推理

    Graph Inference Acceleration by Learning MLPs on Graphs without Supervision

    [https://arxiv.org/abs/2402.08918](https://arxiv.org/abs/2402.08918)

    该论文提出了一个简单而有效的框架SimMLP，通过在图上无监督学习MLPs，提高了在延迟敏感的应用中的泛化能力。

    

    图神经网络（GNNs）已经在各种图学习任务中展示出了有效性，但是它们对消息传递的依赖限制了它们在延迟敏感的应用中的部署，比如金融欺诈检测。最近的研究探索了从GNNs中提取知识到多层感知机（MLPs）来加速推理。然而，这种任务特定的有监督蒸馏限制了对未见节点的泛化，而在延迟敏感的应用中这种情况很常见。为此，我们提出了一种简单而有效的框架SimMLP，用于在图上无监督学习MLPs，以增强泛化能力。SimMLP利用自监督对齐GNNs和MLPs之间的节点特征和图结构之间的精细和泛化的相关性，并提出了两种策略来减轻平凡解的风险。从理论上讲，

    arXiv:2402.08918v1 Announce Type: cross Abstract: Graph Neural Networks (GNNs) have demonstrated effectiveness in various graph learning tasks, yet their reliance on message-passing constraints their deployment in latency-sensitive applications such as financial fraud detection. Recent works have explored distilling knowledge from GNNs to Multi-Layer Perceptrons (MLPs) to accelerate inference. However, this task-specific supervised distillation limits generalization to unseen nodes, which are prevalent in latency-sensitive applications. To this end, we present \textbf{\textsc{SimMLP}}, a \textbf{\textsc{Sim}}ple yet effective framework for learning \textbf{\textsc{MLP}}s on graphs without supervision, to enhance generalization. \textsc{SimMLP} employs self-supervised alignment between GNNs and MLPs to capture the fine-grained and generalizable correlation between node features and graph structures, and proposes two strategies to alleviate the risk of trivial solutions. Theoretically, w
    
[^6]: Voronoi Candidates用于贝叶斯优化

    Voronoi Candidates for Bayesian Optimization

    [https://arxiv.org/abs/2402.04922](https://arxiv.org/abs/2402.04922)

    使用Voronoi候选点边界可以在贝叶斯优化中有效地优化黑盒函数，提高了多起始连续搜索的执行时间。

    

    贝叶斯优化（BO）为高效优化黑盒函数提供了一种优雅的方法。然而，采集准则需要进行具有挑战性的内部优化，这可能引起很大的开销。许多实际的BO方法，尤其是在高维情况下，不采用对采集函数进行形式化连续优化，而是在有限的空间填充候选集上进行离散搜索。在这里，我们提议使用候选点，其位于当前设计点的Voronoi镶嵌边界上，因此它们与两个或多个设计点等距离。我们讨论了通过直接采样Voronoi边界而不明确生成镶嵌的策略，从而适应高维度中的大设计。通过使用高斯过程和期望改进来对一组测试问题进行优化，我们的方法在不损失准确性的情况下显著提高了多起始连续搜索的执行时间。

    Bayesian optimization (BO) offers an elegant approach for efficiently optimizing black-box functions. However, acquisition criteria demand their own challenging inner-optimization, which can induce significant overhead. Many practical BO methods, particularly in high dimension, eschew a formal, continuous optimization of the acquisition function and instead search discretely over a finite set of space-filling candidates. Here, we propose to use candidates which lie on the boundary of the Voronoi tessellation of the current design points, so they are equidistant to two or more of them. We discuss strategies for efficient implementation by directly sampling the Voronoi boundary without explicitly generating the tessellation, thus accommodating large designs in high dimension. On a battery of test problems optimized via Gaussian processes with expected improvement, our proposed approach significantly improves the execution time of a multi-start continuous search without a loss in accuracy
    
[^7]: PAC隐私保护扩散模型

    PAC Privacy Preserving Diffusion Models

    [https://arxiv.org/abs/2312.01201](https://arxiv.org/abs/2312.01201)

    提出了一种PAC隐私保护扩散模型，通过将私有分类器指导集成到采样过程中增强隐私保护，并发展了一种新的度量标准来衡量隐私水平，在保护性能方面表现出卓越表现。

    

    数据隐私保护正在引起研究人员的越来越多的关注。扩散模型（DMs），尤其是具有严格的差分隐私，有可能生成既具有高隐私性又具有良好视觉质量的图像。然而，挑战在于确保在私有化特定数据属性时的强大保护，当前模型在这些方面经常存在不足。为了解决这些挑战，我们引入了PAC隐私保护扩散模型，这是一种利用扩散原理并确保“可能大致正确（PAC）”隐私性的模型。我们通过将私有分类器指导集成到Langevin采样过程中来增强隐私保护。此外，认识到在衡量模型隐私性方面存在差距，我们开发了一种新的度量标准来衡量隐私水平。我们的模型通过这个新度量标准评估，并通过高斯矩阵计算支持PAC界限，表现出更优异的隐私性能。

    arXiv:2312.01201v2 Announce Type: replace-cross  Abstract: Data privacy protection is garnering increased attention among researchers. Diffusion models (DMs), particularly with strict differential privacy, can potentially produce images with both high privacy and visual quality. However, challenges arise such as in ensuring robust protection in privatizing specific data attributes, areas where current models often fall short. To address these challenges, we introduce the PAC Privacy Preserving Diffusion Model, a model leverages diffusion principles and ensure Probably Approximately Correct (PAC) privacy. We enhance privacy protection by integrating a private classifier guidance into the Langevin Sampling Process. Additionally, recognizing the gap in measuring the privacy of models, we have developed a novel metric to gauge privacy levels. Our model, assessed with this new metric and supported by Gaussian matrix computations for the PAC bound, has shown superior performance in privacy p
    
[^8]: Vecchia-Laplace近似法在潜在高斯过程模型中的迭代方法

    Iterative Methods for Vecchia-Laplace Approximations for Latent Gaussian Process Models. (arXiv:2310.12000v1 [stat.ME])

    [http://arxiv.org/abs/2310.12000](http://arxiv.org/abs/2310.12000)

    这篇文章介绍了用于潜在高斯过程模型中的Vecchia-Laplace近似法的迭代方法，相比于传统的Cholesky分解方法，可以显著加快计算速度。

    

    潜在高斯过程（GP）模型是灵活的概率非参数函数模型。Vecchia近似是用于克服大数据计算瓶颈的准确近似方法，Laplace近似是一种快速方法，可以近似非高斯似然函数的边缘似然和后验预测分布，并具有渐近收敛保证。然而，当与直接求解方法（如Cholesky分解）结合使用时，Vecchia-Laplace近似的计算复杂度增长超线性地随样本大小增加。因此，与Vecchia-Laplace近似计算相关的运算在通常情况下是最准确的大型数据集时会变得非常缓慢。在本文中，我们提出了几种用于Vecchia-Laplace近似推断的迭代方法，相比于基于Cholesky的计算，可以大大加快计算速度。我们对我们的方法进行了分析。

    Latent Gaussian process (GP) models are flexible probabilistic non-parametric function models. Vecchia approximations are accurate approximations for GPs to overcome computational bottlenecks for large data, and the Laplace approximation is a fast method with asymptotic convergence guarantees to approximate marginal likelihoods and posterior predictive distributions for non-Gaussian likelihoods. Unfortunately, the computational complexity of combined Vecchia-Laplace approximations grows faster than linearly in the sample size when used in combination with direct solver methods such as the Cholesky decomposition. Computations with Vecchia-Laplace approximations thus become prohibitively slow precisely when the approximations are usually the most accurate, i.e., on large data sets. In this article, we present several iterative methods for inference with Vecchia-Laplace approximations which make computations considerably faster compared to Cholesky-based calculations. We analyze our propo
    
[^9]: 神经网络的记忆化：超越最坏情况

    Memorization with neural nets: going beyond the worst case. (arXiv:2310.00327v1 [stat.ML])

    [http://arxiv.org/abs/2310.00327](http://arxiv.org/abs/2310.00327)

    本文研究了神经网络的插值问题，提出了一种简单的随机算法，在给定的数据集和两个类的情况下，能够以很高的概率构建一个插值的神经网络。这些结果与训练数据规模无关。

    

    在实践中，深度神经网络通常能够轻松地插值其训练数据。为了理解这一现象，许多研究都旨在量化神经网络架构的记忆能力：即在任意放置这些点并任意分配标签的情况下，架构能够插值的最大点数。然而，对于实际数据，人们直觉地期望存在一种良性结构，使得插值在比记忆能力建议的较小网络尺寸上已经发生。在本文中，我们通过采用实例特定的观点来研究插值。我们引入了一个简单的随机算法，它可以在多项式时间内给定一个固定的有限数据集和两个类的情况下，以很高的概率构建出一个插值三层神经网络。所需的参数数量与这两个类的几何特性及其相互排列有关。因此，我们获得了与训练数据规模无关的保证。

    In practice, deep neural networks are often able to easily interpolate their training data. To understand this phenomenon, many works have aimed to quantify the memorization capacity of a neural network architecture: the largest number of points such that the architecture can interpolate any placement of these points with any assignment of labels. For real-world data, however, one intuitively expects the presence of a benign structure so that interpolation already occurs at a smaller network size than suggested by memorization capacity. In this paper, we investigate interpolation by adopting an instance-specific viewpoint. We introduce a simple randomized algorithm that, given a fixed finite dataset with two classes, with high probability constructs an interpolating three-layer neural network in polynomial time. The required number of parameters is linked to geometric properties of the two classes and their mutual arrangement. As a result, we obtain guarantees that are independent of t
    
[^10]: 我们可以从量子卷积神经网络中学到什么？

    What can we learn from quantum convolutional neural networks?. (arXiv:2308.16664v1 [quant-ph])

    [http://arxiv.org/abs/2308.16664](http://arxiv.org/abs/2308.16664)

    通过分析量子卷积神经网络（QCNNs），我们发现它们通过隐藏特征映射嵌入物理系统参数，并且利用量子临界性生成适合的基函数集，池化层选择能够形成高性能决策边界的基函数，而模型的泛化性能依赖于嵌入类型。

    

    通过分析量子卷积神经网络（QCNNs），我们可以得出以下结论：1）通过隐藏特征映射，工作于量子数据可以被视为嵌入物理系统参数；2）对于量子相位识别，其高性能可以归因于在基态嵌入期间生成非常适合的基函数集，其中自旋模型的量子临界性导致具有快速变化特征的基函数；3）QCNN的池化层负责选择那些能够有助于形成高性能决策边界的基函数，学习过程对应于适应性测量，使得少量量子比特算符映射到整个寄存器可观测量；4）QCNN模型的泛化强烈依赖于嵌入类型，基于傅里叶基的旋转特征映射需要仔细的特征工程；5）基于有限数量的测量次数的读出的QCNN的准确性和泛化能力倾向于地面态。

    We can learn from analyzing quantum convolutional neural networks (QCNNs) that: 1) working with quantum data can be perceived as embedding physical system parameters through a hidden feature map; 2) their high performance for quantum phase recognition can be attributed to generation of a very suitable basis set during the ground state embedding, where quantum criticality of spin models leads to basis functions with rapidly changing features; 3) pooling layers of QCNNs are responsible for picking those basis functions that can contribute to forming a high-performing decision boundary, and the learning process corresponds to adapting the measurement such that few-qubit operators are mapped to full-register observables; 4) generalization of QCNN models strongly depends on the embedding type, and that rotation-based feature maps with the Fourier basis require careful feature engineering; 5) accuracy and generalization of QCNNs with readout based on a limited number of shots favor the groun
    
[^11]: 一种简单的面向特征分布偏斜联邦学习的数据增强方法

    A Simple Data Augmentation for Feature Distribution Skewed Federated Learning. (arXiv:2306.09363v1 [cs.LG])

    [http://arxiv.org/abs/2306.09363](http://arxiv.org/abs/2306.09363)

    本文针对特征分布偏斜的联邦学习提出了FedRDN方法，在输入层级上实现了数据增强，将整个联邦数据集的统计信息注入到本地客户端数据中，以缓解特征漂移问题。

    

    联邦学习（FL）是一种分布式协作学习方法，可以确保隐私保护。然而，由于数据异构性（即非独立同分布数据），它的性能必然受到影响。本文针对特征分布偏斜的FL场景展开研究，提出了一种通用的数据增强方法，以减轻由本地数据集之间潜在分布不同导致的特征漂移问题。

    Federated learning (FL) facilitates collaborative learning among multiple clients in a distributed manner, while ensuring privacy protection. However, its performance is inevitably degraded as suffering data heterogeneity, i.e., non-IID data. In this paper, we focus on the feature distribution skewed FL scenario, which is widespread in real-world applications. The main challenge lies in the feature shift caused by the different underlying distributions of local datasets. While the previous attempts achieved progress, few studies pay attention to the data itself, the root of this issue. Therefore, the primary goal of this paper is to develop a general data augmentation technique at the input level, to mitigate the feature shift. To achieve this goal, we propose FedRDN, a simple yet remarkably effective data augmentation method for feature distribution skewed FL, which randomly injects the statistics of the dataset from the entire federation into the client's data. By this, our method ca
    
[^12]: 评分差值流模型用于隐式生成建模

    The Score-Difference Flow for Implicit Generative Modeling. (arXiv:2304.12906v1 [cs.LG])

    [http://arxiv.org/abs/2304.12906](http://arxiv.org/abs/2304.12906)

    本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。

    

    隐式生成建模(IGM)旨在生成符合目标数据分布特征的合成数据样本。最近的研究(例如评分匹配网络、扩散模型)从通过环境空间中的动态扰动或流将合成源数据推向目标分布的角度解决了IGM问题。我们引入了任意目标和源分布之间的评分差异(SD)作为流，它可以最优地减少它们之间的Kullback-Leibler散度，同时解决Schr​​ödinger桥问题。我们将SD流应用于方便的代理分布，当且仅当原始分布对齐时，它们是对齐的。我们在某些条件下展示了这种公式与去噪扩散模型的形式一致性。然而，与扩散模型不同，SD流没有对先验分布施加任何限制。我们还表明，在无限辨别器能力的极限下，生成对抗网络的训练包含SD流。我们的实验表明，SD流在几个基准数据集上优于先前的最新技术。

    Implicit generative modeling (IGM) aims to produce samples of synthetic data matching the characteristics of a target data distribution. Recent work (e.g. score-matching networks, diffusion models) has approached the IGM problem from the perspective of pushing synthetic source data toward the target distribution via dynamical perturbations or flows in the ambient space. We introduce the score difference (SD) between arbitrary target and source distributions as a flow that optimally reduces the Kullback-Leibler divergence between them while also solving the Schr\"odinger bridge problem. We apply the SD flow to convenient proxy distributions, which are aligned if and only if the original distributions are aligned. We demonstrate the formal equivalence of this formulation to denoising diffusion models under certain conditions. However, unlike diffusion models, SD flow places no restrictions on the prior distribution. We also show that the training of generative adversarial networks includ
    

