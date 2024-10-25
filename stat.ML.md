# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Credal Learning Theory](https://rss.arxiv.org/abs/2402.00957) | 本文提出了一种信任学习理论，通过使用凸集的概率来建模数据生成分布的变异性，从有限样本的训练集中推断出信任集，并推导出bounds。 |
| [^2] | [Classification Using Global and Local Mahalanobis Distances](https://arxiv.org/abs/2402.08283) | 本论文提出了一种使用全局和局部马氏距离的分类方法，适用于椭圆形分布的竞争类别，该方法相比流行的参数化和非参数化分类器具有更好的灵活性和性能。 |
| [^3] | [Gradient Sketches for Training Data Attribution and Studying the Loss Landscape](https://arxiv.org/abs/2402.03994) | 本论文提出了一种可扩展的渐变草图算法，用于训练数据归因和损失地貌研究。作者在三个应用中展示了该方法的有效性。 |
| [^4] | [Equivalence of the Empirical Risk Minimization to Regularization on the Family of f-Divergences](https://arxiv.org/abs/2402.00501) | 经验风险最小化与f-分布族的正则化的解决方案在特定条件下是唯一的，并且可以通过使用不同的f-分布正则化等效地表示。 |
| [^5] | [Tail-adaptive Bayesian shrinkage](https://arxiv.org/abs/2007.02192) | 提出了一种在多样的稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法，通过新的全局-局部-尾部高斯混合分布实现，能够根据稀疏程度自适应调整先验的尾部重量以适应更多或更少信号。 |
| [^6] | [FairWASP: Fast and Optimal Fair Wasserstein Pre-processing.](http://arxiv.org/abs/2311.00109) | FairWASP是一种快速和最优的公平Wasserstein预处理方法，通过重新加权数据集来减少分类数据集中的不平等性，同时满足人口平等性准则。这种方法可以用于构建可以输入任何分类方法的数据集。 |
| [^7] | [Convolutions Through the Lens of Tensor Networks.](http://arxiv.org/abs/2307.02275) | 该论文提供了一种通过张量网络理解和演化卷积的新视角，可以通过绘制和操作张量网络来进行函数转换、子张量访问和融合。研究人员还演示了卷积图表的导出以及各种自动微分操作和二阶信息逼近图表的生成，同时还提供了特定于卷积的图表转换，以优化计算性能。 |
| [^8] | [The Representation Jensen-Shannon Divergence.](http://arxiv.org/abs/2305.16446) | 本文提出了一种基于表示的新型散度——表示Jensen-Shannon散度，通过将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱，实现对数据分布的估计，并提供了具有灵活性，可扩展性，可微分性的经验协方差矩阵估计函数和基于核矩阵的估计函数。 |
| [^9] | [Neural incomplete factorization: learning preconditioners for the conjugate gradient method.](http://arxiv.org/abs/2305.16368) | 本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。 |

# 详细

[^1]: 信任学习理论

    Credal Learning Theory

    [https://rss.arxiv.org/abs/2402.00957](https://rss.arxiv.org/abs/2402.00957)

    本文提出了一种信任学习理论，通过使用凸集的概率来建模数据生成分布的变异性，从有限样本的训练集中推断出信任集，并推导出bounds。

    

    统计学习理论是机器学习的基础，为从未知概率分布中学习到的模型的风险提供理论边界。然而，在实际部署中，数据分布可能会变化，导致领域适应/泛化问题。在本文中，我们建立了一个“信任”学习理论的基础，使用概率的凸集（信任集）来建模数据生成分布的变异性。我们认为，这样的信任集可以从有限样本的训练集中推断出来。对于有限假设空间（无论是否可实现）和无限模型空间，推导出界限，这直接推广了经典结果。

    Statistical learning theory is the foundation of machine learning, providing theoretical bounds for the risk of models learnt from a (single) training set, assumed to issue from an unknown probability distribution. In actual deployment, however, the data distribution may (and often does) vary, causing domain adaptation/generalization issues. In this paper we lay the foundations for a `credal' theory of learning, using convex sets of probabilities (credal sets) to model the variability in the data-generating distribution. Such credal sets, we argue, may be inferred from a finite sample of training sets. Bounds are derived for the case of finite hypotheses spaces (both assuming realizability or not) as well as infinite model spaces, which directly generalize classical results.
    
[^2]: 使用全局和局部马氏距离的分类方法

    Classification Using Global and Local Mahalanobis Distances

    [https://arxiv.org/abs/2402.08283](https://arxiv.org/abs/2402.08283)

    本论文提出了一种使用全局和局部马氏距离的分类方法，适用于椭圆形分布的竞争类别，该方法相比流行的参数化和非参数化分类器具有更好的灵活性和性能。

    

    我们提出了一种基于来自不同类别的观察值的马氏距离的新型半参数分类器。我们的工具是一个具有逻辑链接函数的广义加性模型，它使用这些距离作为特征来估计不同类别的后验概率。尽管流行的参数化分类器如线性和二次判别分析主要基于基础分布的正态性，但所提出的分类器更加灵活，不受此类参数化假设的限制。由于椭圆分布的密度是马氏距离的函数，当竞争类别是（几乎）椭圆形时，该分类器的效果很好。在这种情况下，它经常胜过流行的非参数化分类器，特别是当样本量相对于数据维数较小时。为了应对非椭圆和可能多峰的分布，我们提出了马氏距离的局部版本。随后，我们提出了

    We propose a novel semi-parametric classifier based on Mahalanobis distances of an observation from the competing classes. Our tool is a generalized additive model with the logistic link function that uses these distances as features to estimate the posterior probabilities of the different classes. While popular parametric classifiers like linear and quadratic discriminant analyses are mainly motivated by the normality of the underlying distributions, the proposed classifier is more flexible and free from such parametric assumptions. Since the densities of elliptic distributions are functions of Mahalanobis distances, this classifier works well when the competing classes are (nearly) elliptic. In such cases, it often outperforms popular nonparametric classifiers, especially when the sample size is small compared to the dimension of the data. To cope with non-elliptic and possibly multimodal distributions, we propose a local version of the Mahalanobis distance. Subsequently, we propose 
    
[^3]: 使用渐变草图进行训练数据归因和损失地貌研究

    Gradient Sketches for Training Data Attribution and Studying the Loss Landscape

    [https://arxiv.org/abs/2402.03994](https://arxiv.org/abs/2402.03994)

    本论文提出了一种可扩展的渐变草图算法，用于训练数据归因和损失地貌研究。作者在三个应用中展示了该方法的有效性。

    

    随机投影或渐变和Hessian向量乘积的草图在需要存储许多这些向量并保留关于它们的相对几何信息的应用中起着重要作用。两个重要场景是训练数据归因（跟踪模型对训练数据的行为），其中需要存储每个训练示例的渐变，以及Hessian的频谱研究（分析训练动态），其中需要存储多个Hessian向量乘积。虽然使用密集矩阵的草图易于实现，但它们受存储限制，不能扩展到现代神经网络。在神经网络内在维度的研究工作的推动下，我们提出并研究了可伸缩草图算法的设计空间。我们在三个应用中展示了我们方法的有效性：训练数据归因，Hessian谱分析和微调预先训练时的内在维度计算。

    Random projections or sketches of gradients and Hessian vector products play an essential role in applications where one needs to store many such vectors while retaining accurate information about their relative geometry. Two important scenarios are training data attribution (tracing a model's behavior to the training data), where one needs to store a gradient for each training example, and the study of the spectrum of the Hessian (to analyze the training dynamics), where one needs to store multiple Hessian vector products. While sketches that use dense matrices are easy to implement, they are memory bound and cannot be scaled to modern neural networks. Motivated by work on the intrinsic dimension of neural networks, we propose and study a design space for scalable sketching algorithms. We demonstrate the efficacy of our approach in three applications: training data attribution, the analysis of the Hessian spectrum and the computation of the intrinsic dimension when fine-tuning pre-tra
    
[^4]: 经验风险最小化与f-分布族正则化的等价性

    Equivalence of the Empirical Risk Minimization to Regularization on the Family of f-Divergences

    [https://arxiv.org/abs/2402.00501](https://arxiv.org/abs/2402.00501)

    经验风险最小化与f-分布族的正则化的解决方案在特定条件下是唯一的，并且可以通过使用不同的f-分布正则化等效地表示。

    

    在对f中的温和条件下，给出了经验风险最小化与f-分布的正则化（ERM-$f$DR）的解法。在这些条件下，最优测度被证明是唯一的。并给出了特定选择函数f的解决方案的示例。通过利用f-分布族的灵活性，获得了先前对常见正则化选择的已知解决方案，包括相对熵正则化的唯一解（Type-I和Type-II）。对解的分析揭示了在ERM-$f$DR问题中使用f-分布时的以下属性：$i)$ f-分布正则化强制将解的支持与参考测度的支持重合，引入了在训练数据提供的证据中占主导地位的强归纳偏差；$ii)$ 任何f-分布的正则化都等价于另一种f-分布的正则化。

    The solution to empirical risk minimization with $f$-divergence regularization (ERM-$f$DR) is presented under mild conditions on $f$. Under such conditions, the optimal measure is shown to be unique. Examples of the solution for particular choices of the function $f$ are presented. Previously known solutions to common regularization choices are obtained by leveraging the flexibility of the family of $f$-divergences. These include the unique solutions to empirical risk minimization with relative entropy regularization (Type-I and Type-II). The analysis of the solution unveils the following properties of $f$-divergences when used in the ERM-$f$DR problem: $i\bigl)$ $f$-divergence regularization forces the support of the solution to coincide with the support of the reference measure, which introduces a strong inductive bias that dominates the evidence provided by the training data; and $ii\bigl)$ any $f$-divergence regularization is equivalent to a different $f$-divergence regularization 
    
[^5]: 尾部自适应贝叶斯收缩

    Tail-adaptive Bayesian shrinkage

    [https://arxiv.org/abs/2007.02192](https://arxiv.org/abs/2007.02192)

    提出了一种在多样的稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法，通过新的全局-局部-尾部高斯混合分布实现，能够根据稀疏程度自适应调整先验的尾部重量以适应更多或更少信号。

    

    本文研究了高维回归问题下多样的稀疏情况下的鲁棒贝叶斯方法。传统的收缩先验主要设计用于在所谓的超稀疏领域从成千上万个预测变量中检测少数信号。然而，当稀疏程度适中时，它们可能表现不尽人意。在本文中，我们提出了一种在多样稀疏情况下具有尾部自适应收缩特性的鲁棒稀疏估计方法。在这种特性中，先验的尾部重量会自适应调整，随着稀疏水平的增加或减少变得更大或更小，以适应先验地更多或更少的信号。我们提出了一个全局局部尾部（GLT）高斯混合分布以确保这种属性。我们考察了先验的尾部指数与基础稀疏水平之间的关系，并证明GLT后验会在...

    arXiv:2007.02192v4 Announce Type: replace-cross  Abstract: Robust Bayesian methods for high-dimensional regression problems under diverse sparse regimes are studied. Traditional shrinkage priors are primarily designed to detect a handful of signals from tens of thousands of predictors in the so-called ultra-sparsity domain. However, they may not perform desirably when the degree of sparsity is moderate. In this paper, we propose a robust sparse estimation method under diverse sparsity regimes, which has a tail-adaptive shrinkage property. In this property, the tail-heaviness of the prior adjusts adaptively, becoming larger or smaller as the sparsity level increases or decreases, respectively, to accommodate more or fewer signals, a posteriori. We propose a global-local-tail (GLT) Gaussian mixture distribution that ensures this property. We examine the role of the tail-index of the prior in relation to the underlying sparsity level and demonstrate that the GLT posterior contracts at the
    
[^6]: FairWASP：快速和最优的公平Wasserstein预处理

    FairWASP: Fast and Optimal Fair Wasserstein Pre-processing. (arXiv:2311.00109v1 [cs.LG])

    [http://arxiv.org/abs/2311.00109](http://arxiv.org/abs/2311.00109)

    FairWASP是一种快速和最优的公平Wasserstein预处理方法，通过重新加权数据集来减少分类数据集中的不平等性，同时满足人口平等性准则。这种方法可以用于构建可以输入任何分类方法的数据集。

    

    近年来，机器学习方法的快速发展旨在减少不同子群体之间模型输出的不平等性。在许多情况下，训练数据可能会被不同用户在多个下游应用中使用，这意味着对训练数据本身进行干预可能是最有效的。在这项工作中，我们提出了一种新的预处理方法FairWASP，旨在减少分类数据集中的不平等性，而不会修改原始数据。FairWASP返回样本级权重，使重新加权的数据集最小化与原始数据集的Wasserstein距离，同时满足（经验版本的）人口平等性，这是一种常用的公平性准则。我们从理论上证明了整数权重的最优性，这意味着我们的方法可以等同地理解为复制或删除样本。因此，FairWASP可用于构建可以输入任何分类方法的数据集，而不仅仅是接受样本权重的方法。

    Recent years have seen a surge of machine learning approaches aimed at reducing disparities in model outputs across different subgroups. In many settings, training data may be used in multiple downstream applications by different users, which means it may be most effective to intervene on the training data itself. In this work, we present FairWASP, a novel pre-processing approach designed to reduce disparities in classification datasets without modifying the original data. FairWASP returns sample-level weights such that the reweighted dataset minimizes the Wasserstein distance to the original dataset while satisfying (an empirical version of) demographic parity, a popular fairness criterion. We show theoretically that integer weights are optimal, which means our method can be equivalently understood as duplicating or eliminating samples. FairWASP can therefore be used to construct datasets which can be fed into any classification method, not just methods which accept sample weights. Ou
    
[^7]: 透过张量网络的视角解析卷积

    Convolutions Through the Lens of Tensor Networks. (arXiv:2307.02275v1 [cs.LG])

    [http://arxiv.org/abs/2307.02275](http://arxiv.org/abs/2307.02275)

    该论文提供了一种通过张量网络理解和演化卷积的新视角，可以通过绘制和操作张量网络来进行函数转换、子张量访问和融合。研究人员还演示了卷积图表的导出以及各种自动微分操作和二阶信息逼近图表的生成，同时还提供了特定于卷积的图表转换，以优化计算性能。

    

    尽管卷积的直观概念简单，但其分析比稠密层更加复杂，这使得理论和算法的推广变得困难。我们通过张量网络（TN）提供了对卷积的新视角，通过绘制图表、操作图表进行函数转换、子张量访问和融合来推理底层张量乘法。我们通过推导各种自动微分操作的图表以及具有完整超参数支持、批处理、通道组和任意卷积维度泛化的流行的二阶信息逼近的图表来展示这种表达能力。此外，我们基于连接模式提供了特定于卷积的转换，允许在评估之前重新连接和简化图表。最后，我们通过依赖于高效TN缩并的已建立机制来探究计算性能。我们的TN实现加速了最近提出的

    Despite their simple intuition, convolutions are more tedious to analyze than dense layers, which complicates the generalization of theoretical and algorithmic ideas. We provide a new perspective onto convolutions through tensor networks (TNs) which allow reasoning about the underlying tensor multiplications by drawing diagrams, and manipulating them to perform function transformations, sub-tensor access, and fusion. We demonstrate this expressive power by deriving the diagrams of various autodiff operations and popular approximations of second-order information with full hyper-parameter support, batching, channel groups, and generalization to arbitrary convolution dimensions. Further, we provide convolution-specific transformations based on the connectivity pattern which allow to re-wire and simplify diagrams before evaluation. Finally, we probe computational performance, relying on established machinery for efficient TN contraction. Our TN implementation speeds up a recently-proposed
    
[^8]: 基于表示的Jensen-Shannon散度

    The Representation Jensen-Shannon Divergence. (arXiv:2305.16446v1 [cs.LG])

    [http://arxiv.org/abs/2305.16446](http://arxiv.org/abs/2305.16446)

    本文提出了一种基于表示的新型散度——表示Jensen-Shannon散度，通过将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱，实现对数据分布的估计，并提供了具有灵活性，可扩展性，可微分性的经验协方差矩阵估计函数和基于核矩阵的估计函数。

    

    统计散度量化概率分布之间的差异，是机器学习中的一种重要方法。但是，由于数据的底层分布通常未知，从经验样本中估计散度是一个基本难题。本文提出了一种基于再生核希尔伯特空间(RKHS)中协方差算子的新型散度——表示Jensen-Shannon散度。我们的方法将数据分布嵌入到RKHS中，并利用表示的协方差算子的频谱。我们提供了一个从经验协方差矩阵估计的估计函数，它通过使用Fourier特征将数据映射到RKHS中。此估计函数是灵活、可扩展、可微分的，并且适用于小批量优化问题。此外，我们还提供了一种基于核矩阵的估计函数，而不需要对RKHS进行显式映射。我们证明这个量是Jensen-Shannon散度的一个下界。

    Statistical divergences quantify the difference between probability distributions finding multiple uses in machine-learning. However, a fundamental challenge is to estimate divergence from empirical samples since the underlying distributions of the data are usually unknown. In this work, we propose the representation Jensen-Shannon Divergence, a novel divergence based on covariance operators in reproducing kernel Hilbert spaces (RKHS). Our approach embeds the data distributions in an RKHS and exploits the spectrum of the covariance operators of the representations. We provide an estimator from empirical covariance matrices by explicitly mapping the data to an RKHS using Fourier features. This estimator is flexible, scalable, differentiable, and suitable for minibatch-based optimization problems. Additionally, we provide an estimator based on kernel matrices without having an explicit mapping to the RKHS. We show that this quantity is a lower bound on the Jensen-Shannon divergence, and 
    
[^9]: 神经不完全分解：学习共轭梯度法的预处理器

    Neural incomplete factorization: learning preconditioners for the conjugate gradient method. (arXiv:2305.16368v1 [math.OC])

    [http://arxiv.org/abs/2305.16368](http://arxiv.org/abs/2305.16368)

    本文提出了一种名为神经不完全分解的新方法，利用自监督训练的图神经网络生成适用于特定问题域的有效预处理器。其通过替换传统手工预处理器显着提高了收敛和计算效率，在合成和真实问题上进行的实验均表现出竞争力。

    

    本文提出了一种新型的数据驱动方法，用于加速科学计算和优化中遇到的大规模线性方程组求解。我们的方法利用自监督训练图神经网络，生成适用于特定问题域的有效预处理器。通过替换与共轭梯度法一起使用的传统手工预处理器，我们的方法（称为神经不完全分解）显着加速了收敛和计算效率。我们的方法的核心是一种受稀疏矩阵理论启发的新型消息传递块，它与寻找矩阵的稀疏分解的目标相一致。我们在合成问题和来自科学计算的真实问题上评估了我们的方法。我们的结果表明，神经不完全分解始终优于最常见的通用预处理器，包括不完全的Cholesky方法，在收敛速度和计算效率方面表现出竞争力。

    In this paper, we develop a novel data-driven approach to accelerate solving large-scale linear equation systems encountered in scientific computing and optimization. Our method utilizes self-supervised training of a graph neural network to generate an effective preconditioner tailored to the specific problem domain. By replacing conventional hand-crafted preconditioners used with the conjugate gradient method, our approach, named neural incomplete factorization (NeuralIF), significantly speeds-up convergence and computational efficiency. At the core of our method is a novel message-passing block, inspired by sparse matrix theory, that aligns with the objective to find a sparse factorization of the matrix. We evaluate our proposed method on both a synthetic and a real-world problem arising from scientific computing. Our results demonstrate that NeuralIF consistently outperforms the most common general-purpose preconditioners, including the incomplete Cholesky method, achieving competit
    

