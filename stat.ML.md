# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Grid Monitoring and Protection with Continuous Point-on-Wave Measurements and Generative AI](https://arxiv.org/abs/2403.06942) | 提出了基于连续时序测量和生成人工智能的电网监测和控制系统，通过数据压缩和故障检测，实现了对传统监控系统的进步。 |
| [^2] | [Universal Generalization Guarantees for Wasserstein Distributionally Robust Models](https://arxiv.org/abs/2402.11981) | 本文建立了涵盖所有实际情况的Wasserstein分布鲁棒模型确切泛化保证，不需要限制性假设，适用于各种传输成本函数和损失函数，包括深度学习。 |
| [^3] | [An analysis of the noise schedule for score-based generative models](https://arxiv.org/abs/2402.04650) | 本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。 |
| [^4] | [MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage.](http://arxiv.org/abs/2401.12824) | 本文提出了一种使用有限敏感信息泄露的去偏置图神经网络进行公平节点分类的方法，该方法克服了非独立同分布图结构中的拓扑依赖问题，并构建了一个模型无关的去偏置框架，以防止下游误用并提高训练的可靠性。 |
| [^5] | [Closed-Form Diffusion Models.](http://arxiv.org/abs/2310.12395) | 本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。 |
| [^6] | [Effect of hyperparameters on variable selection in random forests.](http://arxiv.org/abs/2309.06943) | 这项研究评估了随机森林中超参数对变量选择的影响，在高维组学研究中，适当设置RF超参数对选择重要变量具有重要意义。 |
| [^7] | [Improved learning theory for kernel distribution regression with two-stage sampling.](http://arxiv.org/abs/2308.14335) | 本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。 |
| [^8] | [PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models.](http://arxiv.org/abs/2307.09254) | 本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。 |
| [^9] | [Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning.](http://arxiv.org/abs/2307.05772) | 这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。 |
| [^10] | [Sources of Uncertainty in Machine Learning -- A Statisticians' View.](http://arxiv.org/abs/2305.16703) | 本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。 |
| [^11] | [Randomized Block-Coordinate Optimistic Gradient Algorithms for Root-Finding Problems.](http://arxiv.org/abs/2301.03113) | 本文提出了两种新的随机乐观梯度算法来解决大规模情况下的根查找问题。第一种算法在底层算子满足一定条件时可以达到较好的收敛速度，第二种算法是一种加速算法，可以更快地收敛。算法的收敛性和解的存在性也得到了证明。 |

# 详细

[^1]: 使用连续时序测量和生成人工智能进行电网监测和保护

    Grid Monitoring and Protection with Continuous Point-on-Wave Measurements and Generative AI

    [https://arxiv.org/abs/2403.06942](https://arxiv.org/abs/2403.06942)

    提出了基于连续时序测量和生成人工智能的电网监测和控制系统，通过数据压缩和故障检测，实现了对传统监控系统的进步。

    

    本文提出了一个下一代电网监测和控制系统的案例，利用生成人工智能（AI）、机器学习和统计推断方面的最新进展。我们提出了一种基于连续时序测量和AI支持的数据压缩和故障检测的监测和控制框架，超越了先前基于SCADA和同步相量技术构建的广域监测系统的发展。

    arXiv:2403.06942v1 Announce Type: cross  Abstract: Purpose This article presents a case for a next-generation grid monitoring and control system, leveraging recent advances in generative artificial intelligence (AI), machine learning, and statistical inference. Advancing beyond earlier generations of wide-area monitoring systems built upon supervisory control and data acquisition (SCADA) and synchrophasor technologies, we argue for a monitoring and control framework based on the streaming of continuous point-on-wave (CPOW) measurements with AI-powered data compression and fault detection.   Methods and Results: The architecture of the proposed design originates from the Wiener-Kallianpur innovation representation of a random process that transforms causally a stationary random process into an innovation sequence with independent and identically distributed random variables. This work presents a generative AI approach that (i) learns an innovation autoencoder that extracts innovation se
    
[^2]: Wasserstein分布鲁棒模型的通用泛化保证

    Universal Generalization Guarantees for Wasserstein Distributionally Robust Models

    [https://arxiv.org/abs/2402.11981](https://arxiv.org/abs/2402.11981)

    本文建立了涵盖所有实际情况的Wasserstein分布鲁棒模型确切泛化保证，不需要限制性假设，适用于各种传输成本函数和损失函数，包括深度学习。

    

    分布稳健优化已经成为一种训练鲁棒机器学习模型的吸引人方式，能够捕捉数据的不确定性和分布的变化。最近的统计分析证明，基于Wasserstein模糊集构建的鲁棒模型具有很好的泛化保证，打破了维度灾难。然而，这些结果是在特定情况下获得的，以近似代价获得，或者在实践中难以验证的假设下获得的。相反，我们在本文中建立了涵盖所有实际情况的确切泛化保证，包括任何传输成本函数和任何损失函数，可能是非凸和非平滑的情况。例如，我们的结果适用于深度学习，而不需要限制性假设。我们通过一种将非平滑分析理论与经典集中结果相结合的新颖证明技术来实现这一结果。我们的方法足够通用，可以拓展至

    arXiv:2402.11981v1 Announce Type: cross  Abstract: Distributionally robust optimization has emerged as an attractive way to train robust machine learning models, capturing data uncertainty and distribution shifts. Recent statistical analyses have proved that robust models built from Wasserstein ambiguity sets have nice generalization guarantees, breaking the curse of dimensionality. However, these results are obtained in specific cases, at the cost of approximations, or under assumptions difficult to verify in practice. In contrast, we establish, in this article, exact generalization guarantees that cover all practical cases, including any transport cost function and any loss function, potentially non-convex and nonsmooth. For instance, our result applies to deep learning, without requiring restrictive assumptions. We achieve this result through a novel proof technique that combines nonsmooth analysis rationale with classical concentration results. Our approach is general enough to ext
    
[^3]: 基于得分的生成模型噪声调度分析

    An analysis of the noise schedule for score-based generative models

    [https://arxiv.org/abs/2402.04650](https://arxiv.org/abs/2402.04650)

    本研究针对基于得分的生成模型噪声调度进行了分析，提出了目标分布和估计分布之间KL散度的上界以及Wasserstein距离的改进误差界限，同时提出了自动调节噪声调度的算法，并通过实验证明了算法的性能。

    

    基于得分的生成模型（SGMs）旨在通过仅使用目标数据的噪声扰动样本来学习得分函数，从而估计目标数据分布。最近的文献主要关注评估目标分布和估计分布之间的误差，通过KL散度和Wasserstein距离来衡量生成质量。至今为止，所有现有结果都是针对时间均匀变化的噪声调度得到的。在对数据分布进行温和假设的前提下，我们建立了目标分布和估计分布之间KL散度的上界，明确依赖于任何时间相关的噪声调度。假设得分是利普希茨连续的情况下，我们提供了更好的Wasserstein距离误差界限，利用了有利的收缩机制。我们还提出了一种使用所提出的上界自动调节噪声调度的算法。我们通过实验证明了算法的性能。

    Score-based generative models (SGMs) aim at estimating a target data distribution by learning score functions using only noise-perturbed samples from the target. Recent literature has focused extensively on assessing the error between the target and estimated distributions, gauging the generative quality through the Kullback-Leibler (KL) divergence and Wasserstein distances.  All existing results  have been obtained so far for time-homogeneous speed of the noise schedule.  Under mild assumptions on the data distribution, we establish an upper bound for the KL divergence between the target and the estimated distributions, explicitly depending on any time-dependent noise schedule. Assuming that the score is Lipschitz continuous, we provide an improved error bound in Wasserstein distance, taking advantage of favourable underlying contraction mechanisms. We also propose an algorithm to automatically tune the noise schedule using the proposed upper bound. We illustrate empirically the perfo
    
[^4]: MAPPING: 使用有限敏感信息泄露的去偏置图神经网络进行公平节点分类

    MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage. (arXiv:2401.12824v1 [cs.LG])

    [http://arxiv.org/abs/2401.12824](http://arxiv.org/abs/2401.12824)

    本文提出了一种使用有限敏感信息泄露的去偏置图神经网络进行公平节点分类的方法，该方法克服了非独立同分布图结构中的拓扑依赖问题，并构建了一个模型无关的去偏置框架，以防止下游误用并提高训练的可靠性。

    

    尽管在各种基于网络的应用中取得了显著的成功，但图神经网络（GNN）继承并进一步加剧了历史上的偏见和社会刻板印象，这严重阻碍了它们在在线临床诊断、金融信贷等高风险领域的部署。然而，当前的公平性研究主要集中在独立同分布数据上，并不能简单地复制到具有拓扑依赖的非独立同分布图结构中。现有的公平图学习通常偏好于使用成对约束来实现公平性，但无法克服维度限制并将其推广到多个敏感属性；此外，大多数研究集中在处理技术上来强制并调整公平性，在预处理阶段构建一个模型无关的去偏置GNN框架，以防止下游误用并提高训练的可靠性在先前的工作中，GNN往往倾向于增强公平性或增加预测性能，因此在二者之间进行全面权衡仍然是一个挑战。

    Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or 
    
[^5]: 闭式扩散模型

    Closed-Form Diffusion Models. (arXiv:2310.12395v1 [cs.LG])

    [http://arxiv.org/abs/2310.12395](http://arxiv.org/abs/2310.12395)

    本研究提出了一种闭式扩散模型，通过显式平滑的闭式得分函数来生成新样本，无需训练，且在消费级CPU上能够实现与神经SGMs相竞争的采样速度。

    

    基于得分的生成模型(SGMs)通过迭代地使用扰动目标函数的得分函数来从目标分布中采样。对于任何有限的训练集，可以闭式地评估这个得分函数，但由此得到的SGMs会记忆其训练数据，不能生成新样本。在实践中，可以通过训练神经网络来近似得分函数，但这种近似的误差有助于推广，然而神经SGMs的训练和采样代价高，而且对于这种误差提供的有效正则化方法在理论上尚不清楚。因此，在这项工作中，我们采用显式平滑的闭式得分来获得一个生成新样本的SGMs，而无需训练。我们分析了我们的模型，并提出了一个基于最近邻的高效得分函数估计器。利用这个估计器，我们的方法在消费级CPU上运行时能够达到与神经SGMs相竞争的采样速度。

    Score-based generative models (SGMs) sample from a target distribution by iteratively transforming noise using the score function of the perturbed target. For any finite training set, this score function can be evaluated in closed form, but the resulting SGM memorizes its training data and does not generate novel samples. In practice, one approximates the score by training a neural network via score-matching. The error in this approximation promotes generalization, but neural SGMs are costly to train and sample, and the effective regularization this error provides is not well-understood theoretically. In this work, we instead explicitly smooth the closed-form score to obtain an SGM that generates novel samples without training. We analyze our model and propose an efficient nearest-neighbor-based estimator of its score function. Using this estimator, our method achieves sampling times competitive with neural SGMs while running on consumer-grade CPUs.
    
[^6]: 随机森林中超参数对变量选择的影响

    Effect of hyperparameters on variable selection in random forests. (arXiv:2309.06943v1 [stat.ML])

    [http://arxiv.org/abs/2309.06943](http://arxiv.org/abs/2309.06943)

    这项研究评估了随机森林中超参数对变量选择的影响，在高维组学研究中，适当设置RF超参数对选择重要变量具有重要意义。

    

    随机森林（RF）在高维组学研究中适用于预测建模和变量选择。先前研究了RF算法的超参数对预测性能和变量重要性估计的影响，但超参数对基于RF的变量选择的影响尚不清楚。我们利用理论分布和实证基因表达数据进行了两个模拟研究，评估了Vita和Boruta变量选择 procedures 在选择重要变量（敏感性）的同时控制虚警率（FDR）的能力。我们的结果表明，在训练数据集中，要比训练数据集的抽取策略和最小终端节点大小更能影响选择 procedures。RF超参数的合适设置取决于

    Random forests (RFs) are well suited for prediction modeling and variable selection in high-dimensional omics studies. The effect of hyperparameters of the RF algorithm on prediction performance and variable importance estimation have previously been investigated. However, how hyperparameters impact RF-based variable selection remains unclear. We evaluate the effects on the Vita and the Boruta variable selection procedures based on two simulation studies utilizing theoretical distributions and empirical gene expression data. We assess the ability of the procedures to select important variables (sensitivity) while controlling the false discovery rate (FDR). Our results show that the proportion of splitting candidate variables (mtry.prop) and the sample fraction (sample.fraction) for the training dataset influence the selection procedures more than the drawing strategy of the training datasets and the minimal terminal node size. A suitable setting of the RF hyperparameters depends on the
    
[^7]: 改进的核分布回归学习理论与两阶段采样

    Improved learning theory for kernel distribution regression with two-stage sampling. (arXiv:2308.14335v1 [math.ST])

    [http://arxiv.org/abs/2308.14335](http://arxiv.org/abs/2308.14335)

    本文改进了核分布回归的学习理论，引入了新的近无偏条件，并提供了关于两阶段采样效果的新误差界。

    

    分布回归问题涵盖了许多重要的统计和机器学习任务，在各种应用中都有出现。在解决这个问题的各种现有方法中，核方法已经成为首选的方法。事实上，核分布回归在计算上是有利的，并且得到了最近的学习理论的支持。该理论还解决了两阶段采样的设置，其中只有输入分布的样本可用。在本文中，我们改进了核分布回归的学习理论。我们研究了基于希尔伯特嵌入的核，这些核包含了大多数（如果不是全部）现有方法。我们引入了嵌入的新近无偏条件，使我们能够通过新的分析提供关于两阶段采样效果的新误差界。我们证明了这种新近无偏条件对三个重要的核类别成立，这些核基于最优输运和平均嵌入。

    The distribution regression problem encompasses many important statistics and machine learning tasks, and arises in a large range of applications. Among various existing approaches to tackle this problem, kernel methods have become a method of choice. Indeed, kernel distribution regression is both computationally favorable, and supported by a recent learning theory. This theory also tackles the two-stage sampling setting, where only samples from the input distributions are available. In this paper, we improve the learning theory of kernel distribution regression. We address kernels based on Hilbertian embeddings, that encompass most, if not all, of the existing approaches. We introduce the novel near-unbiased condition on the Hilbertian embeddings, that enables us to provide new error bounds on the effect of the two-stage sampling, thanks to a new analysis. We show that this near-unbiased condition holds for three important classes of kernels, based on optimal transport and mean embedd
    
[^8]: 用于量化生成式语言模型不确定性的PAC神经预测集学习

    PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models. (arXiv:2307.09254v1 [cs.LG])

    [http://arxiv.org/abs/2307.09254](http://arxiv.org/abs/2307.09254)

    本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。

    

    学习和量化模型的不确定性是增强模型可信度的关键任务。由于对生成虚构事实的担忧，最近兴起的生成式语言模型（GLM）特别强调可靠的不确定性量化的需求。本文提出了一种学习神经预测集模型的方法，该方法能够以可能近似正确（PAC）的方式量化GLM的不确定性。与现有的预测集模型通过标量值参数化不同，我们提出通过神经网络参数化预测集，实现更精确的不确定性量化，但仍满足PAC保证。通过在四种类型的语言数据集和六种类型的模型上展示，我们的方法相比标准基准方法平均提高了63％的量化不确定性。

    Uncertainty learning and quantification of models are crucial tasks to enhance the trustworthiness of the models. Importantly, the recent surge of generative language models (GLMs) emphasizes the need for reliable uncertainty quantification due to the concerns on generating hallucinated facts. In this paper, we propose to learn neural prediction set models that comes with the probably approximately correct (PAC) guarantee for quantifying the uncertainty of GLMs. Unlike existing prediction set models, which are parameterized by a scalar value, we propose to parameterize prediction sets via neural networks, which achieves more precise uncertainty quantification but still satisfies the PAC guarantee. We demonstrate the efficacy of our method on four types of language datasets and six types of models by showing that our method improves the quantified uncertainty by $63\%$ on average, compared to a standard baseline method.
    
[^9]: 随机集合卷积神经网络（RS-CNN）用于认识论深度学习

    Random-Set Convolutional Neural Network (RS-CNN) for Epistemic Deep Learning. (arXiv:2307.05772v1 [cs.LG])

    [http://arxiv.org/abs/2307.05772](http://arxiv.org/abs/2307.05772)

    这篇论文提出了一种新的随机集合卷积神经网络（RS-CNN）用于分类，通过预测信念函数而不是概率矢量集合，以表示模型的置信度和认识不确定性。基于认识论深度学习方法，该模型能够估计由有限训练集引起的认识不确定性。

    

    机器学习越来越多地应用于安全关键领域，对抗攻击的鲁棒性至关重要，错误的预测可能导致潜在的灾难性后果。这突出了学习系统需要能够确定模型对其预测的置信度以及与之相关联的认识不确定性的手段，“知道一个模型不知道”。在本文中，我们提出了一种新颖的用于分类的随机集合卷积神经网络（RS-CNN），其预测信念函数而不是概率矢量集合，使用随机集合的数学，即对样本空间的幂集的分布。基于认识论深度学习方法，随机集模型能够表示机器学习中由有限训练集引起的“认识性”不确定性。我们通过近似预测信念函数相关联的置信集的大小来估计认识不确定性。

    Machine learning is increasingly deployed in safety-critical domains where robustness against adversarial attacks is crucial and erroneous predictions could lead to potentially catastrophic consequences. This highlights the need for learning systems to be equipped with the means to determine a model's confidence in its prediction and the epistemic uncertainty associated with it, 'to know when a model does not know'. In this paper, we propose a novel Random-Set Convolutional Neural Network (RS-CNN) for classification which predicts belief functions rather than probability vectors over the set of classes, using the mathematics of random sets, i.e., distributions over the power set of the sample space. Based on the epistemic deep learning approach, random-set models are capable of representing the 'epistemic' uncertainty induced in machine learning by limited training sets. We estimate epistemic uncertainty by approximating the size of credal sets associated with the predicted belief func
    
[^10]: 机器学习中的不确定性来源 -- 一个统计学家的视角

    Sources of Uncertainty in Machine Learning -- A Statisticians' View. (arXiv:2305.16703v1 [stat.ML])

    [http://arxiv.org/abs/2305.16703](http://arxiv.org/abs/2305.16703)

    本文讨论了机器学习中不确定性的来源和类型，从统计学家的视角出发，分类别介绍了随机性和认知性不确定性的概念，证明了不确定性来源各异，不可简单归为两类。同时，与统计学概念进行类比，探讨不确定性在机器学习中的作用。

    

    机器学习和深度学习已经取得了令人瞩目的成就，使我们能够回答几年前难以想象的问题。除了这些成功之外，越来越清晰的是，在纯预测之外，量化不确定性也是相关和必要的。虽然近年来已经出现了这方面的第一批概念和思想，但本文采用了一个概念性的视角，并探讨了可能的不确定性来源。通过采用统计学家的视角，我们讨论了与机器学习更常见相关的随机性和认知性不确定性的概念。本文旨在规范这两种类型的不确定性，并证明不确定性的来源各异，并且不总是可以分解为随机性和认知性。通过将统计概念与机器学习中的不确定性进行类比，我们也展示了统计学概念和机器学习中不确定性的作用。

    Machine Learning and Deep Learning have achieved an impressive standard today, enabling us to answer questions that were inconceivable a few years ago. Besides these successes, it becomes clear, that beyond pure prediction, which is the primary strength of most supervised machine learning algorithms, the quantification of uncertainty is relevant and necessary as well. While first concepts and ideas in this direction have emerged in recent years, this paper adopts a conceptual perspective and examines possible sources of uncertainty. By adopting the viewpoint of a statistician, we discuss the concepts of aleatoric and epistemic uncertainty, which are more commonly associated with machine learning. The paper aims to formalize the two types of uncertainty and demonstrates that sources of uncertainty are miscellaneous and can not always be decomposed into aleatoric and epistemic. Drawing parallels between statistical concepts and uncertainty in machine learning, we also demonstrate the rol
    
[^11]: 随机块坐标乐观梯度算法解决根查找问题

    Randomized Block-Coordinate Optimistic Gradient Algorithms for Root-Finding Problems. (arXiv:2301.03113v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2301.03113](http://arxiv.org/abs/2301.03113)

    本文提出了两种新的随机乐观梯度算法来解决大规模情况下的根查找问题。第一种算法在底层算子满足一定条件时可以达到较好的收敛速度，第二种算法是一种加速算法，可以更快地收敛。算法的收敛性和解的存在性也得到了证明。

    

    本文提出了两种新的随机块坐标乐观梯度算法，用于在大规模情况下近似求解非线性方程，也称为根查找问题。第一种算法使用恒定的步长，非加速算法，在底层算子G满足Lipschitz连续性和弱Minty解条件时，它在数学期望E[||Gx^k||^2]上达到O(1/k)的最优迭代收敛速度，其中E[·]表示期望，k为迭代计数器。第二种方法是一种新的加速随机块坐标乐观梯度算法，在G的夹逼性条件下，该算法在E[||Gx^k||^2]和E[||x^{k+1} x^{k}||^2]上分别达到O(1/k^2)和o(1/k^2)的迭代收敛速度。此外，我们证明迭代序列{x^k}几乎必然收敛到一个解，以及在此解处Gx^k的模的平方在...

    In this paper, we develop two new randomized block-coordinate optimistic gradient algorithms to approximate a solution of nonlinear equations in large-scale settings, which are called root-finding problems. Our first algorithm is non-accelerated with constant stepsizes, and achieves $\mathcal{O}(1/k)$ best-iterate convergence rate on $\mathbb{E}[ \Vert Gx^k\Vert^2]$ when the underlying operator $G$ is Lipschitz continuous and satisfies a weak Minty solution condition, where $\mathbb{E}[\cdot]$ is the expectation and $k$ is the iteration counter. Our second method is a new accelerated randomized block-coordinate optimistic gradient algorithm. We establish both $\mathcal{O}(1/k^2)$ and $o(1/k^2)$ last-iterate convergence rates on both $\mathbb{E}[ \Vert Gx^k\Vert^2]$ and $\mathbb{E}[ \Vert x^{k+1} x^{k}\Vert^2]$ for this algorithm under the co-coerciveness of $G$. In addition, we prove that the iterate sequence $\{x^k\}$ converges to a solution almost surely, and $\Vert Gx^k\Vert^2$ at
    

