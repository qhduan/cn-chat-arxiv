# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reconstruction for Sparse View Tomography of Long Objects Applied to Imaging in the Wood Industry](https://arxiv.org/abs/2403.02820) | 本研究提出了一种基于学习原始-对偶神经网络的迭代重建方法，适用于顺序扫描几何形状，可以在稀疏视角下进行原木的三维层析成像重建。 |
| [^2] | [Unlearnable Algorithms for In-context Learning](https://arxiv.org/abs/2402.00751) | 本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。 |
| [^3] | [Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance.](http://arxiv.org/abs/2304.11127) | 该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。 |
| [^4] | [Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge.](http://arxiv.org/abs/2208.10300) | 该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。 |

# 详细

[^1]: 针对木材工业中应用于稀疏视角层析成像的长物体的重建

    Reconstruction for Sparse View Tomography of Long Objects Applied to Imaging in the Wood Industry

    [https://arxiv.org/abs/2403.02820](https://arxiv.org/abs/2403.02820)

    本研究提出了一种基于学习原始-对偶神经网络的迭代重建方法，适用于顺序扫描几何形状，可以在稀疏视角下进行原木的三维层析成像重建。

    

    在木材工业中，通过在移动传送带上从几个源位置进行离散X射线扫描来对原木进行常规质量筛查。通常，通过顺序扫描几何形状获得二维（2D）切片测量。每个2D切片单独不包含足够信息进行三维层析重建，在其中感兴趣的原木生物特征得以很好保留。在本研究中，我们提出了一种基于学习原始-对偶神经网络的迭代重建方法，适用于顺序扫描几何形状。我们的方法在重建过程中积累了相邻切片之间的信息，而不是仅在重建期间考虑单个切片。我们的定量和定性评价结果显示，我们的方法在仅使用五个源位置的情况下产生的原木重建足够准确，以识别像节（分支）、心材等生物特征。

    arXiv:2403.02820v1 Announce Type: new  Abstract: In the wood industry, logs are commonly quality screened by discrete X-ray scans on a moving conveyor belt from a few source positions. Typically, two-dimensional (2D) slice-wise measurements are obtained by a sequential scanning geometry. Each 2D slice alone does not carry sufficient information for a three-dimensional tomographic reconstruction in which biological features of interest in the log are well preserved. In the present work, we propose a learned iterative reconstruction method based on the Learned Primal-Dual neural network, suited for sequential scanning geometries. Our method accumulates information between neighbouring slices, instead of only accounting for single slices during reconstruction. Our quantitative and qualitative evaluations with as few as five source positions show that our method yields reconstructions of logs that are sufficiently accurate to identify biological features like knots (branches), heartwood an
    
[^2]: 无法学习的算法用于上下文学习

    Unlearnable Algorithms for In-context Learning

    [https://arxiv.org/abs/2402.00751](https://arxiv.org/abs/2402.00751)

    本文提出了一种针对预先训练的大型语言模型的高效去学习方法，通过选择少量训练示例来实现任务适应训练数据的精确去学习，并与微调方法进行了比较和讨论。

    

    随着模型被越来越多地部署在未知来源的数据上，机器去学习变得越来越受欢迎。然而，要实现精确的去学习——在没有使用要遗忘的数据的情况下获得与模型分布匹配的模型——是具有挑战性或低效的，通常需要大量的重新训练。在本文中，我们专注于预先训练的大型语言模型（LLM）的任务适应阶段的高效去学习方法。我们观察到LLM进行任务适应的上下文学习能力可以实现任务适应训练数据的高效精确去学习。我们提供了一种算法，用于选择少量训练示例加到LLM的提示前面（用于任务适应），名为ERASE，它的去学习操作成本与模型和数据集的大小无关，意味着它适用于大型模型和数据集。我们还将我们的方法与微调方法进行了比较，并讨论了两种方法之间的权衡。这使我们得到了以下结论：

    Machine unlearning is a desirable operation as models get increasingly deployed on data with unknown provenance. However, achieving exact unlearning -- obtaining a model that matches the model distribution when the data to be forgotten was never used -- is challenging or inefficient, often requiring significant retraining. In this paper, we focus on efficient unlearning methods for the task adaptation phase of a pretrained large language model (LLM). We observe that an LLM's ability to do in-context learning for task adaptation allows for efficient exact unlearning of task adaptation training data. We provide an algorithm for selecting few-shot training examples to prepend to the prompt given to an LLM (for task adaptation), ERASE, whose unlearning operation cost is independent of model and dataset size, meaning it scales to large models and datasets. We additionally compare our approach to fine-tuning approaches and discuss the trade-offs between the two approaches. This leads us to p
    
[^3]: 树状Parzen估计器：理解其算法组成部分及其在提高实证表现中的作用

    Tree-structured Parzen estimator: Understanding its algorithm components and their roles for better empirical performance. (arXiv:2304.11127v1 [cs.LG])

    [http://arxiv.org/abs/2304.11127](http://arxiv.org/abs/2304.11127)

    该论文介绍了一种广泛使用的贝叶斯优化方法 Tree-structured Parzen estimator (TPE)，并对其控制参数的作用和算法直觉进行了讨论和分析，提供了一组推荐设置并证明其能够提高TPE的性能表现。

    

    许多领域中最近的进展要求更加复杂的实验设计。这种复杂的实验通常有许多参数，需要参数调整。Tree-structured Parzen estimator (TPE) 是一种贝叶斯优化方法，在最近的参数调整框架中被广泛使用。尽管它很受欢迎，但控制参数的角色和算法直觉尚未得到讨论。在本教程中，我们将确定每个控制参数的作用以及它们对超参数优化的影响，使用多种基准测试。我们将从剖析研究中得出的推荐设置与基准方法进行比较，并证明我们的推荐设置提高了TPE的性能。我们的TPE实现可在https://github.com/nabenabe0928/tpe/tree/single-opt中获得。

    Recent advances in many domains require more and more complicated experiment design. Such complicated experiments often have many parameters, which necessitate parameter tuning. Tree-structured Parzen estimator (TPE), a Bayesian optimization method, is widely used in recent parameter tuning frameworks. Despite its popularity, the roles of each control parameter and the algorithm intuition have not been discussed so far. In this tutorial, we will identify the roles of each control parameter and their impacts on hyperparameter optimization using a diverse set of benchmarks. We compare our recommended setting drawn from the ablation study with baseline methods and demonstrate that our recommended setting improves the performance of TPE. Our TPE implementation is available at https://github.com/nabenabe0928/tpe/tree/single-opt.
    
[^4]: 多目标参数优化中的有效效用函数学习与先验知识

    Efficient Utility Function Learning for Multi-Objective Parameter Optimization with Prior Knowledge. (arXiv:2208.10300v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.10300](http://arxiv.org/abs/2208.10300)

    该论文提出了一种利用偏好学习离线学习效用函数的方法，以应对真实世界问题中用专家知识定义效用函数困难且与专家反复互动昂贵的问题。使用效用函数空间的粗略信息，能够在使用很少结果时提高效用函数估计，并通过整个优化链中传递效用函数学习任务中出现的不确定性。

    

    目前的多目标优化技术通常假定已有效用函数、通过互动学习效用函数或尝试确定完整的Pareto前沿来进行。然而，在真实世界的问题中，结果往往基于隐含和显性的专家知识，难以定义一个效用函数，而互动学习或后续启发式需要反复并且昂贵地专家参与。为了缓解这种情况，我们使用偏好学习离线学习效用函数，利用专家知识。与其他工作不同的是，我们不仅使用（成对的）结果偏好，而且使用效用函数空间的粗略信息。这使我们能够提高效用函数估计，特别是在使用很少的结果时。此外，我们对效用函数学习任务中出现的不确定性进行建模，并将其传递到整个优化链中。

    The current state-of-the-art in multi-objective optimization assumes either a given utility function, learns a utility function interactively or tries to determine the complete Pareto front, requiring a post elicitation of the preferred result. However, result elicitation in real world problems is often based on implicit and explicit expert knowledge, making it difficult to define a utility function, whereas interactive learning or post elicitation requires repeated and expensive expert involvement. To mitigate this, we learn a utility function offline, using expert knowledge by means of preference learning. In contrast to other works, we do not only use (pairwise) result preferences, but also coarse information about the utility function space. This enables us to improve the utility function estimate, especially when using very few results. Additionally, we model the occurring uncertainties in the utility function learning task and propagate them through the whole optimization chain. 
    

