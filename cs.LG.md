# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout](https://arxiv.org/abs/2404.00412) | SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。 |
| [^2] | [CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control](https://arxiv.org/abs/2403.07728) | CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间 |
| [^3] | [GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation](https://arxiv.org/abs/2403.07247) | 该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。 |
| [^4] | [Koopman operators with intrinsic observables in rigged reproducing kernel Hilbert spaces](https://arxiv.org/abs/2403.02524) | 本文提出了一种基于装配再生核希尔伯特空间内在结构和jets几何概念的估计Koopman算子的新方法JetDMD，通过明确的误差界和收敛率证明其优越性，为Koopman算子的数值估计提供了更精确的方法，同时在装配希尔伯特空间框架内提出了扩展Koopman算子的概念，有助于深入理解估计的Koopman特征函数。 |
| [^5] | [Conformer: Embedding Continuous Attention in Vision Transformer for Weather Forecasting](https://arxiv.org/abs/2402.17966) | Conformer是一种用于天气预测的时空连续视觉Transformer，通过在多头注意力机制中实现连续性来学习时间上的连续天气演变。 |
| [^6] | [LCEN: A Novel Feature Selection Algorithm for Nonlinear, Interpretable Machine Learning Models](https://arxiv.org/abs/2402.17120) | LCEN算法是一种用于创建非线性、可解释机器学习模型的新型特征选择算法，能够更准确、更稀疏地生成模型，并具有鲁棒性。 |
| [^7] | [From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection](https://arxiv.org/abs/2402.14332) | 通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。 |
| [^8] | [Families of costs with zero and nonnegative MTW tensor in optimal transport.](http://arxiv.org/abs/2401.00953) | 这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。 |
| [^9] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^10] | [Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data.](http://arxiv.org/abs/2309.15039) | 该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。 |
| [^11] | [Improving Expressivity of Graph Neural Networks using Localization.](http://arxiv.org/abs/2305.19659) | 本文提出了Weisfeiler-Leman (WL)算法的局部版本，用于解决子图计数问题并提高图神经网络的表达能力，同时，也给出了一些时间和空间效率更高的$k-$WL变体和分裂技术。 |
| [^12] | [RPLKG: Robust Prompt Learning with Knowledge Graph.](http://arxiv.org/abs/2304.10805) | 本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。 |
| [^13] | [Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks.](http://arxiv.org/abs/2212.07356) | 本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。 |
| [^14] | [Sparse PCA With Multiple Components.](http://arxiv.org/abs/2209.14790) | 本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。 |

# 详细

[^1]: SVGCraft:超越单个目标文字到SVG综合画布布局

    SVGCraft: Beyond Single Object Text-to-SVG Synthesis with Comprehensive Canvas Layout

    [https://arxiv.org/abs/2404.00412](https://arxiv.org/abs/2404.00412)

    SVGCraft引入了一种端到端框架，可以从文本描述中生成描绘整个场景的矢量图，其中包括利用预训练的LLM进行布局生成、产生遮罩潜变量以进行准确对象放置、融合注意力图以及使用扩散U-Net进行合成，同时通过预训练的编码器和LPIPS损失进行优化。

    

    生成从文本提示到矢量图的VectorArt是一项具有挑战性的视觉任务，需要对已知和未知实体进行多样化而真实的描述。然而，现有研究主要局限于生成单个对象，而不是由多个元素组成的场景。为此，本文介绍了SVGCraft，这是一个新颖的端到端框架，用于从文本描述中生成描绘整个场景的矢量图。该框架利用预训练的LLM从文本提示生成布局，并引入了一种技术，通过生产特定边界框中的掩膜潜变量实现准确的对象放置。它引入了一个融合机制，用于集成注意力图，并使用扩散U-Net进行连贯的合成，加快绘图过程。生成的SVG使用预训练的编码器和LPIPS损失进行优化，通过透明度调制来最大程度地增加相似性。

    arXiv:2404.00412v1 Announce Type: cross  Abstract: Generating VectorArt from text prompts is a challenging vision task, requiring diverse yet realistic depictions of the seen as well as unseen entities. However, existing research has been mostly limited to the generation of single objects, rather than comprehensive scenes comprising multiple elements. In response, this work introduces SVGCraft, a novel end-to-end framework for the creation of vector graphics depicting entire scenes from textual descriptions. Utilizing a pre-trained LLM for layout generation from text prompts, this framework introduces a technique for producing masked latents in specified bounding boxes for accurate object placement. It introduces a fusion mechanism for integrating attention maps and employs a diffusion U-Net for coherent composition, speeding up the drawing process. The resulting SVG is optimized using a pre-trained encoder and LPIPS loss with opacity modulation to maximize similarity. Additionally, th
    
[^2]: CAS: 一种具有FCR控制的在线选择性符合预测的通用算法

    CAS: A General Algorithm for Online Selective Conformal Prediction with FCR Control

    [https://arxiv.org/abs/2403.07728](https://arxiv.org/abs/2403.07728)

    CAS框架允许在在线选择性预测中控制FCR，通过自适应选择和校准集构造输出符合预测区间

    

    我们研究了在线方式下后选择预测推断的问题。为了避免将资源耗费在不重要的单位上，在报告其预测区间之前对当前个体进行初步选择在在线预测任务中是常见且有意义的。由于在线选择导致所选预测区间中存在时间多重性，因此控制实时误覆盖陈述率（FCR）来测量平均误覆盖误差是重要的。我们开发了一个名为CAS（适应性选择后校准）的通用框架，可以包裹任何预测模型和在线选择规则，以输出后选择的预测区间。如果选择了当前个体，我们首先对历史数据进行自适应选择来构建校准集，然后为未观察到的标签输出符合预测区间。我们为校准集提供了可行的构造方式

    arXiv:2403.07728v1 Announce Type: cross  Abstract: We study the problem of post-selection predictive inference in an online fashion. To avoid devoting resources to unimportant units, a preliminary selection of the current individual before reporting its prediction interval is common and meaningful in online predictive tasks. Since the online selection causes a temporal multiplicity in the selected prediction intervals, it is important to control the real-time false coverage-statement rate (FCR) to measure the averaged miscoverage error. We develop a general framework named CAS (Calibration after Adaptive Selection) that can wrap around any prediction model and online selection rule to output post-selection prediction intervals. If the current individual is selected, we first perform an adaptive selection on historical data to construct a calibration set, then output a conformal prediction interval for the unobserved label. We provide tractable constructions for the calibration set for 
    
[^3]: GuideGen：一种用于联合CT体积和解剖结构生成的文本引导框架

    GuideGen: A Text-guided Framework for Joint CT Volume and Anatomical structure Generation

    [https://arxiv.org/abs/2403.07247](https://arxiv.org/abs/2403.07247)

    该论文提出了一种名为GuideGen的框架，可以根据文本提示联合生成CT图像和腹部器官以及结直肠癌组织掩膜，为医学图像分析领域提供了一种生成数据集的新途径。

    

    arXiv:2403.07247v1 公告类型：交叉 摘要：为了收集带有图像和相应标签的大型医学数据集而进行的注释负担和大量工作很少是划算且令人望而生畏的。这导致了缺乏丰富的训练数据，削弱了下游任务，并在一定程度上加剧了医学领域面临的图像分析挑战。作为一种权宜之计，鉴于生成性神经模型的最近成功，现在可以在外部约束的引导下以高保真度合成图像数据集。本文探讨了这种可能性，并提出了GuideGen：一种联合生成腹部器官和结直肠癌CT图像和组织掩膜的管线，其受文本提示条件约束。首先，我们介绍了体积掩膜采样器，以适应掩膜标签的离散分布并生成低分辨率3D组织掩膜。其次，我们的条件图像生成器会在收到相应文本提示的情况下自回归生成CT切片。

    arXiv:2403.07247v1 Announce Type: cross  Abstract: The annotation burden and extensive labor for gathering a large medical dataset with images and corresponding labels are rarely cost-effective and highly intimidating. This results in a lack of abundant training data that undermines downstream tasks and partially contributes to the challenge image analysis faces in the medical field. As a workaround, given the recent success of generative neural models, it is now possible to synthesize image datasets at a high fidelity guided by external constraints. This paper explores this possibility and presents \textbf{GuideGen}: a pipeline that jointly generates CT images and tissue masks for abdominal organs and colorectal cancer conditioned on a text prompt. Firstly, we introduce Volumetric Mask Sampler to fit the discrete distribution of mask labels and generate low-resolution 3D tissue masks. Secondly, our Conditional Image Generator autoregressively generates CT slices conditioned on a corre
    
[^4]: 在装配再生核希尔伯特空间中具有内在可观测性的Koopman算子

    Koopman operators with intrinsic observables in rigged reproducing kernel Hilbert spaces

    [https://arxiv.org/abs/2403.02524](https://arxiv.org/abs/2403.02524)

    本文提出了一种基于装配再生核希尔伯特空间内在结构和jets几何概念的估计Koopman算子的新方法JetDMD，通过明确的误差界和收敛率证明其优越性，为Koopman算子的数值估计提供了更精确的方法，同时在装配希尔伯特空间框架内提出了扩展Koopman算子的概念，有助于深入理解估计的Koopman特征函数。

    

    本文提出了一种新颖的方法，用于估计装配再生核希尔伯特空间（RKHS）上定义的Koopman算子及其谱。我们提出了一种估计方法，称为Jet Dynamic Mode Decomposition（JetDMD），利用RKHS的内在结构和称为jets的几何概念来增强Koopman算子的估计。该方法在精确度上优化了传统的扩展动态模态分解（EDMD），特别是在特征值的数值估计方面。本文通过明确的误差界和特殊正定内核的收敛率证明了JetDMD的优越性，为其性能提供了坚实的理论基础。我们还深入探讨了Koopman算子的谱分析，在装配希尔伯特空间框架内提出了扩展Koopman算子的概念。这个概念有助于更深入地理解估计的Koopman特征函数并捕捉

    arXiv:2403.02524v1 Announce Type: cross  Abstract: This paper presents a novel approach for estimating the Koopman operator defined on a reproducing kernel Hilbert space (RKHS) and its spectra. We propose an estimation method, what we call Jet Dynamic Mode Decomposition (JetDMD), leveraging the intrinsic structure of RKHS and the geometric notion known as jets to enhance the estimation of the Koopman operator. This method refines the traditional Extended Dynamic Mode Decomposition (EDMD) in accuracy, especially in the numerical estimation of eigenvalues. This paper proves JetDMD's superiority through explicit error bounds and convergence rate for special positive definite kernels, offering a solid theoretical foundation for its performance. We also delve into the spectral analysis of the Koopman operator, proposing the notion of extended Koopman operator within a framework of rigged Hilbert space. This notion leads to a deeper understanding of estimated Koopman eigenfunctions and captu
    
[^5]: Conformer：将连续注意力嵌入视觉Transformer用于天气预测

    Conformer: Embedding Continuous Attention in Vision Transformer for Weather Forecasting

    [https://arxiv.org/abs/2402.17966](https://arxiv.org/abs/2402.17966)

    Conformer是一种用于天气预测的时空连续视觉Transformer，通过在多头注意力机制中实现连续性来学习时间上的连续天气演变。

    

    操作性天气预报系统依赖于计算昂贵的基于物理的模型。尽管基于Transformer的模型在天气预测中显示出了显著潜力，但Transformers是离散模型，限制了其学习动态天气系统连续时空特征的能力。我们通过Conformer解决了这个问题，这是一种用于天气预测的时空连续视觉Transformer。Conformer旨在通过在多头注意力机制中实现连续性来学习时间上的连续天气演变。注意力机制被编码为Transformer架构中的可微分函数，以建模复杂的天气动态。我们将Conformer与最先进的数值天气预报（NWP）模型和几种基于深度学习的天气预测模型进行了评估。Conformer在所有前导时间上优于一些现有的数据驱动模型

    arXiv:2402.17966v1 Announce Type: new  Abstract: Operational weather forecasting system relies on computationally expensive physics-based models. Although Transformers-based models have shown remarkable potential in weather forecasting, Transformers are discrete models which limit their ability to learn the continuous spatio-temporal features of the dynamical weather system. We address this issue with Conformer, a spatio-temporal Continuous Vision Transformer for weather forecasting. Conformer is designed to learn the continuous weather evolution over time by implementing continuity in the multi-head attention mechanism. The attention mechanism is encoded as a differentiable function in the transformer architecture to model the complex weather dynamics. We evaluate Conformer against a state-of-the-art Numerical Weather Prediction (NWP) model and several deep learning based weather forecasting models. Conformer outperforms some of the existing data-driven models at all lead times while 
    
[^6]: LCEN：一种新型特征选择算法，用于非线性的可解释机器学习模型

    LCEN: A Novel Feature Selection Algorithm for Nonlinear, Interpretable Machine Learning Models

    [https://arxiv.org/abs/2402.17120](https://arxiv.org/abs/2402.17120)

    LCEN算法是一种用于创建非线性、可解释机器学习模型的新型特征选择算法，能够更准确、更稀疏地生成模型，并具有鲁棒性。

    

    可解释的架构相对于黑盒架构具有优势，在关键领域如航空或医学中，可解释性对机器学习应用至关重要。然而，最简单、最常用的可解释架构（如LASSO或EN）仅限于线性预测，并且特征选择能力较差。在这项工作中，我们引入了LASSO-Clip-EN（LCEN）算法，用于创建非线性、可解释的机器学习模型。LCEN在多种人工和实证数据集上进行了测试，生成比其他常用架构更准确、更稀疏的模型。这些实验表明，LCEN对数据集和建模中通常存在的许多问题具有鲁棒性，包括噪声、多重共线性、数据稀缺和超参数方差。LCEN还能够从实证数据中重新发现多个物理定律，

    arXiv:2402.17120v1 Announce Type: new  Abstract: Interpretable architectures can have advantages over black-box architectures, and interpretability is essential for the application of machine learning in critical settings, such as aviation or medicine. However, the simplest, most commonly used interpretable architectures (such as LASSO or EN) are limited to linear predictions and have poor feature selection capabilities. In this work, we introduce the LASSO-Clip-EN (LCEN) algorithm for the creation of nonlinear, interpretable machine learning models. LCEN is tested on a wide variety of artificial and empirical datasets, creating more accurate, sparser models than other commonly used architectures. These experiments reveal that LCEN is robust against many issues typically present in datasets and modeling, including noise, multicollinearity, data scarcity, and hyperparameter variance. LCEN is also able to rediscover multiple physical laws from empirical data and, for processes with no kn
    
[^7]: 从大规模到小规模数据集：用于聚类算法选择的尺寸泛化

    From Large to Small Datasets: Size Generalization for Clustering Algorithm Selection

    [https://arxiv.org/abs/2402.14332](https://arxiv.org/abs/2402.14332)

    通过引入尺寸泛化概念，研究了在半监督设置下的聚类算法选择问题，提出了能够在小实例上保证准确度最高的算法也将在原始大实例上拥有最高准确度的条件。

    

    在聚类算法选择中，我们会得到一个大规模数据集，并要有效地选择要使用的聚类算法。我们在半监督设置下研究了这个问题，其中有一个未知的基准聚类，我们只能通过昂贵的oracle查询来访问。理想情况下，聚类算法的输出将与基本事实结构上接近。我们通过引入一种聚类算法准确性的尺寸泛化概念来解决这个问题。我们确定在哪些条件下我们可以（1）对大规模聚类实例进行子采样，（2）在较小实例上评估一组候选算法，（3）保证在小实例上准确度最高的算法将在原始大实例上拥有最高的准确度。我们为三种经典聚类算法提供了理论尺寸泛化保证：单链接、k-means++和Gonzalez的k中心启发式（一种平滑的变种）。

    arXiv:2402.14332v1 Announce Type: new  Abstract: In clustering algorithm selection, we are given a massive dataset and must efficiently select which clustering algorithm to use. We study this problem in a semi-supervised setting, with an unknown ground-truth clustering that we can only access through expensive oracle queries. Ideally, the clustering algorithm's output will be structurally close to the ground truth. We approach this problem by introducing a notion of size generalization for clustering algorithm accuracy. We identify conditions under which we can (1) subsample the massive clustering instance, (2) evaluate a set of candidate algorithms on the smaller instance, and (3) guarantee that the algorithm with the best accuracy on the small instance will have the best accuracy on the original big instance. We provide theoretical size generalization guarantees for three classic clustering algorithms: single-linkage, k-means++, and (a smoothed variant of) Gonzalez's k-centers heuris
    
[^8]: 拥有零和非负MTW张量的费用族在最优传送中的应用

    Families of costs with zero and nonnegative MTW tensor in optimal transport. (arXiv:2401.00953v1 [math.AP])

    [http://arxiv.org/abs/2401.00953](http://arxiv.org/abs/2401.00953)

    这篇论文介绍了在最优传送中使用形式为$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数时的零和非负MTW张量的计算方法，并提供了MTW张量在零向量上为零的条件以及相应的线性ODE的简化方法。此外，还给出了逆函数的解析表达式以及一些具体的应用情况。

    

    我们计算了在$\mathbb{R}^n$上具有形式$\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$的费用函数的最优传送问题的MTW张量（或交叉曲率）。其中，$\mathsf{u}$是一个具有逆函数$\mathsf{s}$的标量函数，$x^{\ft}y$是属于$\mathbb{R}^n$开子集的向量$x，y$的非退化双线性配对。MTW张量在Kim-McCann度量下对于零向量的条件是一个四阶非线性ODE，可以被简化为具有常数系数$P$和$S$的形式为$\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$的线性ODE。最终得到的逆函数包括Lambert和广义反双曲/三角函数。平方欧氏度量和$\log$型费用是这些解的实例。这个家族的最优映射也是显式的。

    We compute explicitly the MTW tensor (or cross curvature) for the optimal transport problem on $\mathbb{R}^n$ with a cost function of form $\mathsf{c}(x, y) = \mathsf{u}(x^{\mathfrak{t}}y)$, where $\mathsf{u}$ is a scalar function with inverse $\mathsf{s}$, $x^{\ft}y$ is a nondegenerate bilinear pairing of vectors $x, y$ belonging to an open subset of $\mathbb{R}^n$. The condition that the MTW-tensor vanishes on null vectors under the Kim-McCann metric is a fourth-order nonlinear ODE, which could be reduced to a linear ODE of the form $\mathsf{s}^{(2)} - S\mathsf{s}^{(1)} + P\mathsf{s} = 0$ with constant coefficients $P$ and $S$. The resulting inverse functions include {\it Lambert} and {\it generalized inverse hyperbolic\slash trigonometric} functions. The square Euclidean metric and $\log$-type costs are equivalent to instances of these solutions. The optimal map for the family is also explicit. For cost functions of a similar form on a hyperboloid model of the hyperbolic space and u
    
[^9]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^10]: 结合存活分析和机器学习利用电子健康记录数据进行肿瘤风险预测

    Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data. (arXiv:2309.15039v1 [cs.LG])

    [http://arxiv.org/abs/2309.15039](http://arxiv.org/abs/2309.15039)

    该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。

    

    纯粹的医学肿瘤筛查方法通常费用高昂、耗时长，并且仅适用于大规模应用。先进的人工智能（AI）方法在癌症检测方面发挥了巨大作用，但需要特定或深入的医学数据。这些方面影响了癌症筛查方法的大规模实施。因此，基于已有的电子健康记录（EHR）数据对患者进行大规模个性化癌症风险评估应用AI方法是一种颠覆性的改变。本文提出了一种利用EHR数据进行大规模肿瘤风险预测的新方法。与其他方法相比，我们的方法通过最小的数据贪婪策略脱颖而出，仅需要来自EHR的医疗服务代码和诊断历史。我们将问题形式化为二分类问题。该数据集包含了175441名不记名的患者（其中2861名被诊断为癌症）。作为基准，我们实现了一个基于循环神经网络（RNN）的解决方案。我们提出了一种方法，将存活分析和机器学习相结合，

    Purely medical cancer screening methods are often costly, time-consuming, and weakly applicable on a large scale. Advanced Artificial Intelligence (AI) methods greatly help cancer detection but require specific or deep medical data. These aspects affect the mass implementation of cancer screening methods. For these reasons, it is a disruptive change for healthcare to apply AI methods for mass personalized assessment of the cancer risk among patients based on the existing Electronic Health Records (EHR) volume.  This paper presents a novel method for mass cancer risk prediction using EHR data. Among other methods, our one stands out by the minimum data greedy policy, requiring only a history of medical service codes and diagnoses from EHR. We formulate the problem as a binary classification. This dataset contains 175 441 de-identified patients (2 861 diagnosed with cancer). As a baseline, we implement a solution based on a recurrent neural network (RNN). We propose a method that combine
    
[^11]: 利用局部化提高图神经网络的表达能力

    Improving Expressivity of Graph Neural Networks using Localization. (arXiv:2305.19659v1 [cs.LG])

    [http://arxiv.org/abs/2305.19659](http://arxiv.org/abs/2305.19659)

    本文提出了Weisfeiler-Leman (WL)算法的局部版本，用于解决子图计数问题并提高图神经网络的表达能力，同时，也给出了一些时间和空间效率更高的$k-$WL变体和分裂技术。

    

    本文提出了Weisfeiler-Leman (WL)算法的局部版本，旨在增加表达能力并减少计算负担。我们专注于子图计数问题，并为任意$k$给出$k-$WL的局部版本。我们分析了Local $k-$WL的作用，并证明其比$k-$WL更具表现力，并且至多与$(k+1)-$WL一样具有表现力。我们给出了一些模式的特征，如果两个图是Local $k-$WL等价的，则它们的子图和诱导子图的计数是不变的。我们还介绍了$k-$WL的两个变体：层$k-$WL和递归$k-$WL。这些方法的时间和空间效率比在整个图上应用$k-$WL更高。我们还提出了一种分裂技术，使用$1-$WL即可保证所有大小不超过4的诱导子图的准确计数。相同的方法可以使用$k>1$进一步扩展到更大的模式。我们还将Local $k-$WL的表现力与其他GNN层次结构进行了比较。

    In this paper, we propose localized versions of Weisfeiler-Leman (WL) algorithms in an effort to both increase the expressivity, as well as decrease the computational overhead. We focus on the specific problem of subgraph counting and give localized versions of $k-$WL for any $k$. We analyze the power of Local $k-$WL and prove that it is more expressive than $k-$WL and at most as expressive as $(k+1)-$WL. We give a characterization of patterns whose count as a subgraph and induced subgraph are invariant if two graphs are Local $k-$WL equivalent. We also introduce two variants of $k-$WL: Layer $k-$WL and recursive $k-$WL. These methods are more time and space efficient than applying $k-$WL on the whole graph. We also propose a fragmentation technique that guarantees the exact count of all induced subgraphs of size at most 4 using just $1-$WL. The same idea can be extended further for larger patterns using $k>1$. We also compare the expressive power of Local $k-$WL with other GNN hierarc
    
[^12]: RPLKG: 基于知识图谱的鲁棒提示学习

    RPLKG: Robust Prompt Learning with Knowledge Graph. (arXiv:2304.10805v1 [cs.AI])

    [http://arxiv.org/abs/2304.10805](http://arxiv.org/abs/2304.10805)

    本研究提出了一种基于知识图谱的鲁棒提示学习方法，通过自动设计有意义和可解释的提示集，提高小样本学习的泛化性能。

    

    大规模预训练模型已经被证明是可迁移的，并且对未知数据集具有很好的泛化性能。最近，诸如CLIP之类的多模态预训练模型在各种实验中表现出显着的性能提升。然而，当标记数据集有限时，新数据集或领域的泛化仍然具有挑战性。为了提高小样本学习的泛化性能，已经进行了各种努力，如提示学习和适配器。然而，当前的少样本自适应方法不具备可解释性，并且需要高计算成本来进行自适应。在本研究中，我们提出了一种新的方法，即基于知识图谱的鲁棒提示学习（RPLKG）。基于知识图谱，我们自动设计出各种可解释和有意义的提示集。我们的模型在大型预训练模型的一次正向传递后获得提示集的缓存嵌入。之后，模型使用GumbelSoftmax优化提示选择过程。

    Large-scale pre-trained models have been known that they are transferable, and they generalize well on the unseen dataset. Recently, multimodal pre-trained models such as CLIP show significant performance improvement in diverse experiments. However, when the labeled dataset is limited, the generalization of a new dataset or domain is still challenging. To improve the generalization performance on few-shot learning, there have been diverse efforts, such as prompt learning and adapter. However, the current few-shot adaptation methods are not interpretable, and they require a high computation cost for adaptation. In this study, we propose a new method, robust prompt learning with knowledge graph (RPLKG). Based on the knowledge graph, we automatically design diverse interpretable and meaningful prompt sets. Our model obtains cached embeddings of prompt sets after one forwarding from a large pre-trained model. After that, model optimizes the prompt selection processes with GumbelSoftmax. In
    
[^13]: 异步联邦学习在无线网络中的调度和聚合设计

    Scheduling and Aggregation Design for Asynchronous Federated Learning over Wireless Networks. (arXiv:2212.07356v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.07356](http://arxiv.org/abs/2212.07356)

    本文提出了一种异步联邦学习的调度策略和聚合加权设计，通过采用基于信道感知数据重要性的调度策略和“年龄感知”的聚合加权设计来解决FL系统中的“拖沓”问题，并通过仿真证实了其有效性。

    

    联邦学习（FL）是一种协作的机器学习（ML）框架，它结合了设备上的训练和基于服务器的聚合来在分布式代理间训练通用的ML模型。本文中，我们提出了一种异步FL设计，采用周期性的聚合来解决FL系统中的“拖沓”问题。考虑到有限的无线通信资源，我们研究了不同调度策略和聚合设计对收敛性能的影响。基于降低聚合模型更新的偏差和方差的重要性，我们提出了一个调度策略，它同时考虑了用户设备的信道质量和训练数据表示。通过仿真验证了我们的基于信道感知数据重要性的调度策略相对于同步联邦学习提出的现有最新方法的有效性。此外，我们还展示了一种“年龄感知”的聚合加权设计可以显著提高学习性能。

    Federated Learning (FL) is a collaborative machine learning (ML) framework that combines on-device training and server-based aggregation to train a common ML model among distributed agents. In this work, we propose an asynchronous FL design with periodic aggregation to tackle the straggler issue in FL systems. Considering limited wireless communication resources, we investigate the effect of different scheduling policies and aggregation designs on the convergence performance. Driven by the importance of reducing the bias and variance of the aggregated model updates, we propose a scheduling policy that jointly considers the channel quality and training data representation of user devices. The effectiveness of our channel-aware data-importance-based scheduling policy, compared with state-of-the-art methods proposed for synchronous FL, is validated through simulations. Moreover, we show that an ``age-aware'' aggregation weighting design can significantly improve the learning performance i
    
[^14]: 多组分的稀疏主成分分析

    Sparse PCA With Multiple Components. (arXiv:2209.14790v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.14790](http://arxiv.org/abs/2209.14790)

    本研究提出了一种新的方法来解决稀疏主成分分析问题，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。

    

    稀疏主成分分析是一种用于以可解释的方式解释高维数据集方差的基本技术。这涉及解决一个稀疏性和正交性约束的凸最大化问题，其计算复杂度非常高。大多数现有的方法通过迭代计算一个稀疏主成分并缩减协方差矩阵来解决稀疏主成分分析，但在寻找多个相互正交的主成分时，这些方法不能保证所得解的正交性和最优性。我们挑战这种现状，通过将正交性条件重新表述为秩约束，并同时对稀疏性和秩约束进行优化。我们设计了紧凑的半正定松弛来提供高质量的上界，当每个主成分的个体稀疏性被指定时，我们通过额外的二阶锥不等式加强上界。此外，我们采用另一种方法来加强上界，我们使用额外的二阶锥不等式来加强上界。

    Sparse Principal Component Analysis (sPCA) is a cardinal technique for obtaining combinations of features, or principal components (PCs), that explain the variance of high-dimensional datasets in an interpretable manner. This involves solving a sparsity and orthogonality constrained convex maximization problem, which is extremely computationally challenging. Most existing works address sparse PCA via methods-such as iteratively computing one sparse PC and deflating the covariance matrix-that do not guarantee the orthogonality, let alone the optimality, of the resulting solution when we seek multiple mutually orthogonal PCs. We challenge this status by reformulating the orthogonality conditions as rank constraints and optimizing over the sparsity and rank constraints simultaneously. We design tight semidefinite relaxations to supply high-quality upper bounds, which we strengthen via additional second-order cone inequalities when each PC's individual sparsity is specified. Further, we de
    

