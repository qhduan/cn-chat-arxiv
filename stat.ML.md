# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent variable model for high-dimensional point process with structured missingness](https://arxiv.org/abs/2402.05758) | 本文提出了一种针对高维点过程的带有结构缺失的灵活高效的潜变量模型，利用高斯过程捕获时间相关性，并开发了可扩展的变分推理方法进行训练。 |
| [^2] | [Scaling laws for learning with real and surrogate data](https://arxiv.org/abs/2402.04376) | 本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。 |
| [^3] | [Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling.](http://arxiv.org/abs/2310.06389) | 本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。 |
| [^4] | [Fishnets: Information-Optimal, Scalable Aggregation for Sets and Graphs.](http://arxiv.org/abs/2310.03812) | Fishnets是一种用于学习信息最优的集合和图聚合的方法，在规模上可以优化到任意数量的数据对象，具有鲁棒性，能够饱和贝叶斯信息内容，并可用于GNNs中的消息传递。 |
| [^5] | [Optimal Rate of Kernel Regression in Large Dimensions.](http://arxiv.org/abs/2309.04268) | 该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。 |
| [^6] | [MALIBO: Meta-learning for Likelihood-free Bayesian Optimization.](http://arxiv.org/abs/2307.03565) | MALIBO是一种元学习贝叶斯优化方法，通过直接学习跨任务的查询效用，并引入辅助模型以实现对新任务的稳健适应，克服了现有方法的可伸缩性和不确定性的限制。 |
| [^7] | [An Empirical Study on Challenging Math Problem Solving with GPT-4.](http://arxiv.org/abs/2306.01337) | 本研究探索使用GPT-4解决更复杂和有挑战性的数学问题，提出了一种名为MathChat的对话式问题求解框架，并在困难高中竞赛问题上进行了评估。 |

# 详细

[^1]: 高维点过程的带结构缺失的潜变量模型

    Latent variable model for high-dimensional point process with structured missingness

    [https://arxiv.org/abs/2402.05758](https://arxiv.org/abs/2402.05758)

    本文提出了一种针对高维点过程的带有结构缺失的灵活高效的潜变量模型，利用高斯过程捕获时间相关性，并开发了可扩展的变分推理方法进行训练。

    

    纵向数据在医疗保健、社会学和地震学等许多领域中具有重要意义，但是真实世界的数据集对从业人员来说存在明显的挑战，因为它们可能是高维的，包含有结构化的缺失模式，并且测量时间点可能受到未知随机过程的控制。尽管已经提出了各种解决方案，但其中大多数仅考虑了这些挑战中的一个。在这项工作中，我们提出了一种灵活高效的潜变量模型，能够应对所有这些限制。我们的方法利用高斯过程来捕获样本与其关联的缺失模式之间的时间相关性，同时也用于建模底层的点过程。我们将我们的模型构建为一个变分自动编码器，同时使用深度神经网络参数化的编码器和解码器模型，并开发了一个可扩展的变分推理方法来进行高效的模型训练。我们展示了这个模型在各个领域的竞争性能。

    Longitudinal data are important in numerous fields, such as healthcare, sociology and seismology, but real-world datasets present notable challenges for practitioners because they can be high-dimensional, contain structured missingness patterns, and measurement time points can be governed by an unknown stochastic process. While various solutions have been suggested, the majority of them have been designed to account for only one of these challenges. In this work, we propose a flexible and efficient latent-variable model that is capable of addressing all these limitations. Our approach utilizes Gaussian processes to capture temporal correlations between samples and their associated missingness masks as well as to model the underlying point process. We construct our model as a variational autoencoder together with deep neural network parameterised encoder and decoder models, and develop a scalable amortised variational inference approach for efficient model training. We demonstrate compe
    
[^2]: 使用真实数据和替代数据进行学习的扩展规律

    Scaling laws for learning with real and surrogate data

    [https://arxiv.org/abs/2402.04376](https://arxiv.org/abs/2402.04376)

    本研究探讨了将替代数据与真实数据整合以进行训练的方案，发现整合替代数据能够显著降低测试误差，并提出了一个扩展规律来描述混合模型的测试误差，可以用于预测最优加权和收益。

    

    收集大量高质量的数据通常被限制在成本昂贵或不切实际的范围内, 这是机器学习中的一个关键瓶颈。相反地, 可以将来自目标分布的小规模数据集与来自公共数据集、不同情况下收集的数据或由生成模型合成的数据相结合, 作为替代数据。我们提出了一种简单的方案来将替代数据整合到训练中, 并使用理论模型和实证研究探索其行为。我们的主要发现是：(i) 整合替代数据可以显著降低原始分布的测试误差；(ii) 为了获得这种效益, 使用最优加权经验风险最小化非常关键；(iii) 在混合使用真实数据和替代数据训练的模型的测试误差可以很好地用一个扩展规律来描述。这可以用来预测最优加权和收益。

    Collecting large quantities of high-quality data is often prohibitively expensive or impractical, and a crucial bottleneck in machine learning. One may instead augment a small set of $n$ data points from the target distribution with data from more accessible sources like public datasets, data collected under different circumstances, or synthesized by generative models. Blurring distinctions, we refer to such data as `surrogate data'.   We define a simple scheme for integrating surrogate data into training and use both theoretical models and empirical studies to explore its behavior. Our main findings are: $(i)$ Integrating surrogate data can significantly reduce the test error on the original distribution; $(ii)$ In order to reap this benefit, it is crucial to use optimally weighted empirical risk minimization; $(iii)$ The test error of models trained on mixtures of real and surrogate data is well described by a scaling law. This can be used to predict the optimal weighting and the gai
    
[^3]: 学习可堆叠和可跳过的乐高积木以实现高效、可重构和可变分辨率的扩散建模

    Learning Stackable and Skippable LEGO Bricks for Efficient, Reconfigurable, and Variable-Resolution Diffusion Modeling. (arXiv:2310.06389v1 [cs.CV])

    [http://arxiv.org/abs/2310.06389](http://arxiv.org/abs/2310.06389)

    本研究提出了乐高积木，通过集成局部特征丰富和全局内容协调，实现了高效且可自适应的迭代细化扩散建模。这些积木可以堆叠在一起，用于在测试时根据需要进行重构，从而减少采样成本并生成高分辨率图像。

    

    扩散模型在生成真实感图像方面表现出色，但在训练和采样方面具有显著的计算成本。尽管有各种技术来解决这些计算挑战，但一个较少探索的问题是设计一个高效且适应性强的网络骨干，用于迭代细化。当前的选项如U-Net和Vision Transformer通常依赖于资源密集型的深度网络，缺乏在变量分辨率下生成图像或使用比训练中更小的网络所需的灵活性。本研究引入了乐高积木，它们无缝集成了局部特征丰富和全局内容协调。这些积木可以堆叠在一起，创建一个测试时可重构的扩散骨干，允许选择性跳过积木以减少采样成本，并生成比训练数据更高分辨率的图像。乐高积木通过MLP对局部区域进行丰富，并使用Transformer块进行变换，同时保持一致的全分辨率

    Diffusion models excel at generating photo-realistic images but come with significant computational costs in both training and sampling. While various techniques address these computational challenges, a less-explored issue is designing an efficient and adaptable network backbone for iterative refinement. Current options like U-Net and Vision Transformer often rely on resource-intensive deep networks and lack the flexibility needed for generating images at variable resolutions or with a smaller network than used in training. This study introduces LEGO bricks, which seamlessly integrate Local-feature Enrichment and Global-content Orchestration. These bricks can be stacked to create a test-time reconfigurable diffusion backbone, allowing selective skipping of bricks to reduce sampling costs and generate higher-resolution images than the training data. LEGO bricks enrich local regions with an MLP and transform them using a Transformer block while maintaining a consistent full-resolution i
    
[^4]: 鱼网：信息最优，可扩展的集合和图聚合

    Fishnets: Information-Optimal, Scalable Aggregation for Sets and Graphs. (arXiv:2310.03812v1 [cs.LG])

    [http://arxiv.org/abs/2310.03812](http://arxiv.org/abs/2310.03812)

    Fishnets是一种用于学习信息最优的集合和图聚合的方法，在规模上可以优化到任意数量的数据对象，具有鲁棒性，能够饱和贝叶斯信息内容，并可用于GNNs中的消息传递。

    

    基于集合的学习是现代深度学习和网络科学的重要组成部分。图神经网络（GNNs）及其不含边的对应物Deepsets在不规则和拓扑复杂的数据集上被证明非常有用。为了学习集合成员的信息丰富的嵌入，关键是指定一个聚合函数，通常是求和、最大值或均值。我们提出了Fishnets，一种用于学习集合数据和图聚合的信息最优嵌入策略，适用于贝叶斯推理。我们证明了：i）Fishnets神经摘要可以最优地扩展到任意数量的数据对象；ii）Fishnets聚合对数据分布的改变具有鲁棒性，而标准的Deepsets不具备这种特性；iii）Fishnets饱和贝叶斯信息内容，并扩展到MCMC技术失败的领域；iv）Fishnets可以作为GNN中的一个插入式聚合方案。我们展示了通过采用Fishnets聚合方案进行消息传递，GNNs可以实现 达到

    Set-based learning is an essential component of modern deep learning and network science. Graph Neural Networks (GNNs) and their edge-free counterparts Deepsets have proven remarkably useful on ragged and topologically challenging datasets. The key to learning informative embeddings for set members is a specified aggregation function, usually a sum, max, or mean. We propose Fishnets, an aggregation strategy for learning information-optimal embeddings for sets of data for both Bayesian inference and graph aggregation. We demonstrate that i) Fishnets neural summaries can be scaled optimally to an arbitrary number of data objects, ii) Fishnets aggregations are robust to changes in data distribution, unlike standard deepsets, iii) Fishnets saturate Bayesian information content and extend to regimes where MCMC techniques fail and iv) Fishnets can be used as a drop-in aggregation scheme within GNNs. We show that by adopting a Fishnets aggregation scheme for message passing, GNNs can achieve 
    
[^5]: 大维度情况下核回归的最优比率

    Optimal Rate of Kernel Regression in Large Dimensions. (arXiv:2309.04268v1 [stat.ML])

    [http://arxiv.org/abs/2309.04268](http://arxiv.org/abs/2309.04268)

    该论文提出了一种针对大维度数据的核回归的最优比率，通过使用Mendelson复杂性和度量熵来刻画其上界和最小化下界。此外，研究还发现最优比率随着维度与样本大小关系的变化呈现出多次下降的行为。

    

    我们对大维度数据（样本大小$n$与样本维度$d$的关系为多项式，即$n\asymp d^{\gamma}$，其中$\gamma>0$）的核回归进行了研究。我们首先通过Mendelson复杂性$\varepsilon_{n}^{2}$和度量熵$\bar{\varepsilon}_{n}^{2}$来建立一个通用工具，用于刻画大维度数据的核回归的上界和最小化下界。当目标函数属于与$\mathbb{S}^{d}$上定义的（一般）内积模型相关联的RKHS时，我们利用这个新工具来展示核回归的过量风险的最小化率是$n^{-1/2}$，当$n\asymp d^{\gamma}$，其中$\gamma=2, 4, 6, 8, \cdots$。然后我们进一步确定了对于所有$\gamma>0$，核回归过量风险的最优比率，并发现随着$\gamma$的变化，最优比率的曲线展现出几个新现象，包括多次下降行为。

    We perform a study on kernel regression for large-dimensional data (where the sample size $n$ is polynomially depending on the dimension $d$ of the samples, i.e., $n\asymp d^{\gamma}$ for some $\gamma >0$ ). We first build a general tool to characterize the upper bound and the minimax lower bound of kernel regression for large dimensional data through the Mendelson complexity $\varepsilon_{n}^{2}$ and the metric entropy $\bar{\varepsilon}_{n}^{2}$ respectively. When the target function falls into the RKHS associated with a (general) inner product model defined on $\mathbb{S}^{d}$, we utilize the new tool to show that the minimax rate of the excess risk of kernel regression is $n^{-1/2}$ when $n\asymp d^{\gamma}$ for $\gamma =2, 4, 6, 8, \cdots$. We then further determine the optimal rate of the excess risk of kernel regression for all the $\gamma>0$ and find that the curve of optimal rate varying along $\gamma$ exhibits several new phenomena including the {\it multiple descent behavior
    
[^6]: MALIBO: 元学习应用于无似然贝叶斯优化

    MALIBO: Meta-learning for Likelihood-free Bayesian Optimization. (arXiv:2307.03565v1 [cs.LG])

    [http://arxiv.org/abs/2307.03565](http://arxiv.org/abs/2307.03565)

    MALIBO是一种元学习贝叶斯优化方法，通过直接学习跨任务的查询效用，并引入辅助模型以实现对新任务的稳健适应，克服了现有方法的可伸缩性和不确定性的限制。

    

    贝叶斯优化是一种优化昂贵黑盒函数的流行方法。传统的贝叶斯优化会从头开始优化每个新的目标任务，而元学习则是利用相关任务的知识来更快地优化新任务的一种方式。然而，现有的元学习贝叶斯优化方法依赖于标准模型，这些模型存在可伸缩性问题，并且对不同任务之间观察数据的尺度和噪声类型非常敏感。此外，它们常常忽视与任务相似性相关的不确定性，这导致在仅有有限观察数据或新任务与相关任务差异显著时，任务适应性不可靠。为了解决这些限制，我们提出了一种新颖的元学习贝叶斯优化方法，旨在绕开标准模型，直接学习跨任务的查询效用。我们的方法明确建模任务的不确定性，并引入了一个辅助模型，使其能够对新任务进行稳健适应。大量实验证明了我们方法的有效性。

    Bayesian optimization (BO) is a popular method to optimize costly black-box functions. While traditional BO optimizes each new target task from scratch, meta-learning has emerged as a way to leverage knowledge from related tasks to optimize new tasks faster. However, existing meta-learning BO methods rely on surrogate models that suffer from scalability issues and are sensitive to observations with different scales and noise types across tasks. Moreover, they often overlook the uncertainty associated with task similarity. This leads to unreliable task adaptation when only limited observations are obtained or when the new tasks differ significantly from the related tasks. To address these limitations, we propose a novel meta-learning BO approach that bypasses the surrogate model and directly learns the utility of queries across tasks. Our method explicitly models task uncertainty and includes an auxiliary model to enable robust adaptation to new tasks. Extensive experiments show that ou
    
[^7]: 基于GPT-4的复杂数学问题求解的实证研究

    An Empirical Study on Challenging Math Problem Solving with GPT-4. (arXiv:2306.01337v1 [cs.CL])

    [http://arxiv.org/abs/2306.01337](http://arxiv.org/abs/2306.01337)

    本研究探索使用GPT-4解决更复杂和有挑战性的数学问题，提出了一种名为MathChat的对话式问题求解框架，并在困难高中竞赛问题上进行了评估。

    

    使用大型语言模型（LLM）来解决数学问题是一项有趣的研究，考虑到在各种科学和工程领域中用自然语言表达的数学问题的丰富性。虽然之前有几项工作研究了使用LLM解决初等数学问题，但本研究探索了使用GPT-4解决更复杂和有挑战性的数学问题的前沿。我们评估了使用GPT-4的各种方法。其中一些是从现有工作中改编而来的，其中一个是MathChat，这是本研究新提出的一种对话式问题求解框架。我们在来自MATH数据集的困难高中竞赛问题上进行评估，表明了所提出的对话式方法的优势。

    Employing Large Language Models (LLMs) to address mathematical problems is an intriguing research endeavor, considering the abundance of math problems expressed in natural language across numerous science and engineering fields. While several prior works have investigated solving elementary mathematics using LLMs, this work explores the frontier of using GPT-4 for solving more complex and challenging math problems. We evaluate various ways of using GPT-4. Some of them are adapted from existing work, and one is \MathChat, a conversational problem-solving framework newly proposed in this work. We perform the evaluation on difficult high school competition problems from the MATH dataset, which shows the advantage of the proposed conversational approach.
    

