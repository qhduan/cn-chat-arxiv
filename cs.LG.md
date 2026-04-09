# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Active Statistical Inference](https://arxiv.org/abs/2403.03208) | 主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。 |
| [^2] | [Don't Label Twice: Quantity Beats Quality when Comparing Binary Classifiers on a Budget](https://arxiv.org/abs/2402.02249) | 在比较两个二元分类器的准确性时，通过收集更多样本的单个标签而不是汇总多个噪声标签能更好地利用预算。 |
| [^3] | [Smoothing the Edges: A General Framework for Smooth Optimization in Sparse Regularization using Hadamard Overparametrization.](http://arxiv.org/abs/2307.03571) | 本文介绍了一种通用框架，可以在稀疏正则化中进行平滑优化，与主流的一阶优化方法兼容，并且能够得到匹配的全局最小值和等价的局部最小值。 |
| [^4] | [A survey of Generative AI Applications.](http://arxiv.org/abs/2306.02781) | 本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。 |

# 详细

[^1]: 主动统计推断

    Active Statistical Inference

    [https://arxiv.org/abs/2403.03208](https://arxiv.org/abs/2403.03208)

    主动推断是一种统计推断方法，通过利用机器学习模型确定最有利于标记的数据点来有效利用预算，实现比现有基线更少样本的相同准确性。

    

    受主动学习概念启发，我们提出了主动推断——一种利用机器学习辅助数据收集进行统计推断的方法。假设对可收集的标签数量有预算限制，该方法利用机器学习模型确定哪些数据点最有利于标记，从而有效利用预算。其运作方式基于一种简单而强大的直觉：优先收集模型表现出不确定性的数据点的标签，并在模型表现出自信时依赖于其预测。主动推断构建了可证明有效的置信区间和假设检验，同时利用任何黑盒机器学习模型并处理任何数据分布。关键点在于，它能以比依赖于非自适应收集数据的现有基线更少的样本达到相同水平的准确性。这意味着对于相同数量的样本，...

    arXiv:2403.03208v1 Announce Type: cross  Abstract: Inspired by the concept of active learning, we propose active inference$\unicode{x2013}$a methodology for statistical inference with machine-learning-assisted data collection. Assuming a budget on the number of labels that can be collected, the methodology uses a machine learning model to identify which data points would be most beneficial to label, thus effectively utilizing the budget. It operates on a simple yet powerful intuition: prioritize the collection of labels for data points where the model exhibits uncertainty, and rely on the model's predictions where it is confident. Active inference constructs provably valid confidence intervals and hypothesis tests while leveraging any black-box machine learning model and handling any data distribution. The key point is that it achieves the same level of accuracy with far fewer samples than existing baselines relying on non-adaptively-collected data. This means that for the same number 
    
[^2]: 不要重复标记：在有限预算下比较二元分类器时，数量胜过质量

    Don't Label Twice: Quantity Beats Quality when Comparing Binary Classifiers on a Budget

    [https://arxiv.org/abs/2402.02249](https://arxiv.org/abs/2402.02249)

    在比较两个二元分类器的准确性时，通过收集更多样本的单个标签而不是汇总多个噪声标签能更好地利用预算。

    

    我们研究了如何更好地利用有限预算来比较两个二元分类器的准确性。通常的做法是通过多次收集和汇总给定数据点的多个噪声标签，通过多数投票形成一个不太噪声的标签。我们证明了一个与常识相反的定理。如果目标是确定两个分类器中的较好者，我们展示了更好的做法是将预算用于收集更多样本的单个标签。我们的结果来自于对Cram\'er定理的非平凡应用，这是大偏差理论中的一个重要工具。我们讨论了我们的工作对机器学习基准设计的影响，其中它们推翻了一些历史上的建议。此外，我们的结果提供了比Hoeffding界更优的样本大小界限。

    We study how to best spend a budget of noisy labels to compare the accuracy of two binary classifiers. It's common practice to collect and aggregate multiple noisy labels for a given data point into a less noisy label via a majority vote. We prove a theorem that runs counter to conventional wisdom. If the goal is to identify the better of two classifiers, we show it's best to spend the budget on collecting a single label for more samples. Our result follows from a non-trivial application of Cram\'er's theorem, a staple in the theory of large deviations. We discuss the implications of our work for the design of machine learning benchmarks, where they overturn some time-honored recommendations. In addition, our results provide sample size bounds superior to what follows from Hoeffding's bound.
    
[^3]: 平滑边缘：利用Hadamard超参数化在稀疏正则化的平滑优化中的一般框架

    Smoothing the Edges: A General Framework for Smooth Optimization in Sparse Regularization using Hadamard Overparametrization. (arXiv:2307.03571v1 [cs.LG])

    [http://arxiv.org/abs/2307.03571](http://arxiv.org/abs/2307.03571)

    本文介绍了一种通用框架，可以在稀疏正则化中进行平滑优化，与主流的一阶优化方法兼容，并且能够得到匹配的全局最小值和等价的局部最小值。

    

    本文介绍了一种用于（结构化）稀疏正则化问题中的$\ell_q$和$\ell_{p,q}$正则化的平滑方法。这些非平滑且可能非凸的问题的优化通常依赖于专门的过程。相比之下，我们的一般框架与主流的一阶优化方法（如随机梯度下降和加速变体）兼容，无需任何修改。这是通过平滑优化转移实现的，其中选定模型参数的超参数化使用Hadamard乘积和惩罚的改变。在超参数问题中，通过用替代参数进行平滑和凸性的$\ell_2$正则化，能够在原始参数化中引入非平滑和非凸性的$\ell_q$或$\ell_{p,q}$正则化。我们证明了我们的方法不仅能够得到匹配的全局最小值，还能得到等价的局部最小值。这在非凸稀疏正则化中尤其有用，因为在这种情况下找到全局最小值非常困难。

    This paper introduces a smooth method for (structured) sparsity in $\ell_q$ and $\ell_{p,q}$ regularized optimization problems. Optimization of these non-smooth and possibly non-convex problems typically relies on specialized procedures. In contrast, our general framework is compatible with prevalent first-order optimization methods like Stochastic Gradient Descent and accelerated variants without any required modifications. This is accomplished through a smooth optimization transfer, comprising an overparametrization of selected model parameters using Hadamard products and a change of penalties. In the overparametrized problem, smooth and convex $\ell_2$ regularization of the surrogate parameters induces non-smooth and non-convex $\ell_q$ or $\ell_{p,q}$ regularization in the original parametrization. We show that our approach yields not only matching global minima but also equivalent local minima. This is particularly useful in non-convex sparse regularization, where finding global m
    
[^4]: 生成式人工智能应用调查

    A survey of Generative AI Applications. (arXiv:2306.02781v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02781](http://arxiv.org/abs/2306.02781)

    本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    

    近年来，生成式人工智能有了显著增长，并在各个领域展示了广泛的应用。本文对350多个生成式人工智能应用进行了全面调查，提供了分类结构和对不同单模和多模生成式人工智能的简洁描述。该调查分成多个部分，覆盖了文本、图像、视频、游戏和脑信息等单模生成式人工智能的广泛应用。我们的调研旨在为研究人员和从业者提供宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    Generative AI has experienced remarkable growth in recent years, leading to a wide array of applications across diverse domains. In this paper, we present a comprehensive survey of more than 350 generative AI applications, providing a structured taxonomy and concise descriptions of various unimodal and even multimodal generative AIs. The survey is organized into sections, covering a wide range of unimodal generative AI applications such as text, images, video, gaming and brain information. Our survey aims to serve as a valuable resource for researchers and practitioners to navigate the rapidly expanding landscape of generative AI, facilitating a better understanding of the current state-of-the-art and fostering further innovation in the field.
    

