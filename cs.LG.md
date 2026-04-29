# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints](https://arxiv.org/abs/2402.09034) | 该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。 |
| [^2] | [NUBO: A Transparent Python Package for Bayesian Optimisation.](http://arxiv.org/abs/2305.06709) | NUBO是一个透明的Python包，用于优化昂贵的黑盒函数，它利用高斯过程做代理模型以及获取函数来指导选择候选点，专注于透明度和用户体验。 |

# 详细

[^1]: 使用平方Sigmoid TanH (SST)激活在数据限制下提高顺序模型性能

    Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints

    [https://arxiv.org/abs/2402.09034](https://arxiv.org/abs/2402.09034)

    该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。

    

    激活函数通过引入非线性来使神经网络能够学习复杂的表示。虽然前馈模型通常使用修正线性单元，但是顺序模型如递归神经网络、长短时记忆（LSTM）和门控循环单元（GRU）仍然依赖于Sigmoid和TanH激活函数。然而，这些传统的激活函数常常在训练在小顺序数据集上时难以建模稀疏模式以有效捕获时间依赖性。为了解决这个限制，我们提出了特别针对在数据限制下增强顺序模型学习能力的平方Sigmoid TanH（SST）激活。SST通过数学平方来放大强激活和弱激活之间的差异，随着信号随时间传播，有助于改善梯度流和信息过滤。我们评估了使用SST的LSTM和GRU模型在不同应用中的性能。

    arXiv:2402.09034v1 Announce Type: cross Abstract: Activation functions enable neural networks to learn complex representations by introducing non-linearities. While feedforward models commonly use rectified linear units, sequential models like recurrent neural networks, long short-term memory (LSTMs) and gated recurrent units (GRUs) still rely on Sigmoid and TanH activation functions. However, these classical activation functions often struggle to model sparse patterns when trained on small sequential datasets to effectively capture temporal dependencies. To address this limitation, we propose squared Sigmoid TanH (SST) activation specifically tailored to enhance the learning capability of sequential models under data constraints. SST applies mathematical squaring to amplify differences between strong and weak activations as signals propagate over time, facilitating improved gradient flow and information filtering. We evaluate SST-powered LSTMs and GRUs for diverse applications, such a
    
[^2]: NUBO：一个透明的 Python 包用于贝叶斯优化

    NUBO: A Transparent Python Package for Bayesian Optimisation. (arXiv:2305.06709v1 [cs.LG])

    [http://arxiv.org/abs/2305.06709](http://arxiv.org/abs/2305.06709)

    NUBO是一个透明的Python包，用于优化昂贵的黑盒函数，它利用高斯过程做代理模型以及获取函数来指导选择候选点，专注于透明度和用户体验。

    

    NUBO（Newcastle University Bayesian Optimisation）是一个贝叶斯优化框架，用于优化昂贵的黑盒函数，比如物理实验和计算机模拟器。它利用高斯过程做代理模型、并通过获取函数来选择用于全局最优化的候选点。NUBO专注于透明度和用户体验，以便让不同领域的研究人员更容易使用贝叶斯优化。

    NUBO, short for Newcastle University Bayesian Optimisation, is a Bayesian optimisation framework for the optimisation of expensive-to-evaluate black-box functions, such as physical experiments and computer simulators. Bayesian optimisation is a cost-efficient optimisation strategy that uses surrogate modelling via Gaussian processes to represent an objective function and acquisition functions to guide the selection of candidate points to approximate the global optimum of the objective function. NUBO itself focuses on transparency and user experience to make Bayesian optimisation easily accessible to researchers from all disciplines. Clean and understandable code, precise references, and thorough documentation ensure transparency, while user experience is ensured by a modular and flexible design, easy-to-write syntax, and careful selection of Bayesian optimisation algorithms. NUBO allows users to tailor Bayesian optimisation to their specific problem by writing the optimisation loop the
    

