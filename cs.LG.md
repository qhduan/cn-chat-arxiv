# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-Excitation Projective Simulation with a Many-Body Physics Inspired Inductive Bias](https://arxiv.org/abs/2402.10192) | 该论文引入了多激发投影模拟（mePS），通过在超图上多个粒子的随机游走，解决了投影模拟（PS）无法模拟同时结合多个概念的思维的问题。 |
| [^2] | [P-tensors: a General Formalism for Constructing Higher Order Message Passing Networks.](http://arxiv.org/abs/2306.10767) | P-tensors 提供了构建高阶消息传递网络的通用形式，其在分子等高度结构化的图形中具有优异的性能表现。 |
| [^3] | [Dictionary Learning under Symmetries via Group Representations.](http://arxiv.org/abs/2305.19557) | 本文研究在预定变换群下学习不变的字典问题。利用非阿贝尔傅里叶分析，提供了算法，建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。 |
| [^4] | [Vicarious Offense and Noise Audit of Offensive Speech Classifiers: Unifying Human and Machine Disagreement on What is Offensive.](http://arxiv.org/abs/2301.12534) | 本文通过人工和机器审核员的共情冒犯和噪声审计研究了冒犯性言论检测中的差异性。结果表明，审核员之间存在广泛的分歧，并且人工审核员和大型语言模型分类器无法预测其他审核员的回应。这对于内容审核具有重要意义。 |

# 详细

[^1]: 借鉴多体物理的归纳偏置的多激发投影模拟

    Multi-Excitation Projective Simulation with a Many-Body Physics Inspired Inductive Bias

    [https://arxiv.org/abs/2402.10192](https://arxiv.org/abs/2402.10192)

    该论文引入了多激发投影模拟（mePS），通过在超图上多个粒子的随机游走，解决了投影模拟（PS）无法模拟同时结合多个概念的思维的问题。

    

    随着深度学习的进步，依赖于机器学习的应用正在越来越多地融入日常生活。然而，大多数深度学习模型具有不透明的、类似于神谕般的特性，使得解释和理解它们的决策变得困难。这个问题导致了被称为可解释人工智能（XAI）的领域的发展。该领域中的一种方法称为投影模拟（PS），将思维过程建模为一个在具有概念附加的顶点的图上的粒子的随机游走。虽然这种描述具有各种好处，包括量化的可能性，但不能自然地用来模拟同时结合多个概念的思维。为了克服这个限制，我们引入了一种称为多激发投影模拟（mePS）的推广，它将思维过程视为超图上多个粒子的随机游走。

    arXiv:2402.10192v1 Announce Type: cross  Abstract: With the impressive progress of deep learning, applications relying on machine learning are increasingly being integrated into daily life. However, most deep learning models have an opaque, oracle-like nature making it difficult to interpret and understand their decisions. This problem led to the development of the field known as eXplainable Artificial Intelligence (XAI). One method in this field known as Projective Simulation (PS) models a chain-of-thought as a random walk of a particle on a graph with vertices that have concepts attached to them. While this description has various benefits, including the possibility of quantization, it cannot be naturally used to model thoughts that combine several concepts simultaneously. To overcome this limitation, we introduce Multi-Excitation Projective Simulation (mePS), a generalization that considers a chain-of-thought to be a random walk of several particles on a hypergraph. A definition for
    
[^2]: P张量：构建高阶消息传递网络的通用形式。

    P-tensors: a General Formalism for Constructing Higher Order Message Passing Networks. (arXiv:2306.10767v1 [stat.ML])

    [http://arxiv.org/abs/2306.10767](http://arxiv.org/abs/2306.10767)

    P-tensors 提供了构建高阶消息传递网络的通用形式，其在分子等高度结构化的图形中具有优异的性能表现。

    

    最近的几篇论文表明，高阶图神经网络在高度结构化的图形如分子中能够比其标准的消息传递对应物获得更好的准确性。这些模型通常通过考虑包含在给定图形中的子图的高阶表示，然后在它们之间执行某些线性映射来工作。我们将这些结构正式化为排列等变张量或P张量，并推导出所有线性映射之间的任意顺序等变P张量的基础。在实验中，我们展示了这种范式在几个基准数据集上达到了现有最先进性能。

    Several recent papers have recently shown that higher order graph neural networks can achieve better accuracy than their standard message passing counterparts, especially on highly structured graphs such as molecules. These models typically work by considering higher order representations of subgraphs contained within a given graph and then perform some linear maps between them. We formalize these structures as permutation equivariant tensors, or P-tensors, and derive a basis for all linear maps between arbitrary order equivariant P-tensors. Experimentally, we demonstrate this paradigm achieves state of the art performance on several benchmark datasets.
    
[^3]: 通过群表示学习对称下的字典学习

    Dictionary Learning under Symmetries via Group Representations. (arXiv:2305.19557v1 [math.OC])

    [http://arxiv.org/abs/2305.19557](http://arxiv.org/abs/2305.19557)

    本文研究在预定变换群下学习不变的字典问题。利用非阿贝尔傅里叶分析，提供了算法，建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。

    

    字典学习问题可以被看作是一个数据驱动的过程，旨在学习一个合适的变换，以便通过示例数据直接表示数据的稀疏性。本文研究了在预定的变换群下学习不变的字典问题。自然的应用领域包括冷冻电镜、多目标跟踪、同步和姿态估计等。我们特别从数学表示理论的角度研究了这个问题。通过利用非阿贝尔傅里叶分析，我们为符合这些不变性的字典学习提供了算法。我们将自然界中的字典学习问题，其自然被建模为无限维度的问题，与相关的计算问题，这必然是有限维度的问题，联系起来。我们建立了字典学习问题可以被有效地理解为某些矩阵优化问题的理论基础。

    The dictionary learning problem can be viewed as a data-driven process to learn a suitable transformation so that data is sparsely represented directly from example data. In this paper, we examine the problem of learning a dictionary that is invariant under a pre-specified group of transformations. Natural settings include Cryo-EM, multi-object tracking, synchronization, pose estimation, etc. We specifically study this problem under the lens of mathematical representation theory. Leveraging the power of non-abelian Fourier analysis for functions over compact groups, we prescribe an algorithmic recipe for learning dictionaries that obey such invariances. We relate the dictionary learning problem in the physical domain, which is naturally modelled as being infinite dimensional, with the associated computational problem, which is necessarily finite dimensional. We establish that the dictionary learning problem can be effectively understood as an optimization instance over certain matrix o
    
[^4]: 人工和机器关于什么是冒犯存在较大分歧的共情冒犯和噪声审计：统一主观冒犯的人类和机器差异

    Vicarious Offense and Noise Audit of Offensive Speech Classifiers: Unifying Human and Machine Disagreement on What is Offensive. (arXiv:2301.12534v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.12534](http://arxiv.org/abs/2301.12534)

    本文通过人工和机器审核员的共情冒犯和噪声审计研究了冒犯性言论检测中的差异性。结果表明，审核员之间存在广泛的分歧，并且人工审核员和大型语言模型分类器无法预测其他审核员的回应。这对于内容审核具有重要意义。

    

    冒犯性言论检测是内容审核的一个关键组成部分。然而，什么是冒犯性的可以是高度主观的。本文研究了当涉及到现实世界社交网站政治言论时，人工和机器审核员对于什么是冒犯性的存在分歧。我们发现（1）审核员之间（包括人工和机器）存在广泛分歧；和（2）人工审核员和大型语言模型分类器无法预测其他审核员基于他们的政治倾向如何回应。对于（1），我们进行了一个前所未有规模的噪声审计，结合了机器和人工回答。对于（2），我们介绍了一个首创的共情冒犯的数据集。我们的噪声审计揭示了不同机器审核员之间的审核结果差异很大。我们与人工审核员进行的实验表明，政治倾向结合敏感问题会影响到一对一的冒犯，以及共情冒犯。数据集可通过https://github.com/Homan-Lab/voic获得。

    Offensive speech detection is a key component of content moderation. However, what is offensive can be highly subjective. This paper investigates how machine and human moderators disagree on what is offensive when it comes to real-world social web political discourse. We show that (1) there is extensive disagreement among the moderators (humans and machines); and (2) human and large-language-model classifiers are unable to predict how other human raters will respond, based on their political leanings. For (1), we conduct a noise audit at an unprecedented scale that combines both machine and human responses. For (2), we introduce a first-of-its-kind dataset of vicarious offense. Our noise audit reveals that moderation outcomes vary wildly across different machine moderators. Our experiments with human moderators suggest that political leanings combined with sensitive issues affect both first-person and vicarious offense. The dataset is available through https://github.com/Homan-Lab/voic
    

