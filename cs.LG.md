# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Quantum Algorithm for Attention Computation.](http://arxiv.org/abs/2307.08045) | 这篇论文研究了使用快速量子算法进行注意力计算，以加速大型语言模型 (LLMs) 的计算速度。 |
| [^2] | [An efficient, provably exact algorithm for the 0-1 loss linear classification problem.](http://arxiv.org/abs/2306.12344) | 该研究详细介绍了一种名为增量单元枚举（ICE）的算法，该算法可以精确解决定维度0-1损失线性分类问题。 |
| [^3] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |

# 详细

[^1]: 快速量子算法用于注意力计算

    Fast Quantum Algorithm for Attention Computation. (arXiv:2307.08045v1 [quant-ph])

    [http://arxiv.org/abs/2307.08045](http://arxiv.org/abs/2307.08045)

    这篇论文研究了使用快速量子算法进行注意力计算，以加速大型语言模型 (LLMs) 的计算速度。

    

    大型语言模型 (LLMs) 表现出色，在各种任务中显示出异常的性能。这些模型由先进的深度学习技术驱动，已经在自然语言处理 (NLP) 领域引起了革命，并在各种与语言相关的任务中取得了显著的结果。LLMs 在机器翻译、情感分析、问答、文本生成、文本分类、语言建模等任务中表现出色。它们在捕捉复杂的语言模式、理解背景、生成连贯且相关的文本方面非常有效。注意力计算方案在大型语言模型 (LLMs) 的架构中起着关键作用。它是一个基本组件，使得模型能够在语言处理任务中有效地捕捉和利用上下文信息。加快注意力计算方案的速度是加速LLMs计算的核心问题之一。

    Large language models (LLMs) have demonstrated exceptional performance across a wide range of tasks. These models, powered by advanced deep learning techniques, have revolutionized the field of natural language processing (NLP) and have achieved remarkable results in various language-related tasks.  LLMs have excelled in tasks such as machine translation, sentiment analysis, question answering, text generation, text classification, language modeling, and more. They have proven to be highly effective in capturing complex linguistic patterns, understanding context, and generating coherent and contextually relevant text. The attention scheme plays a crucial role in the architecture of large language models (LLMs). It is a fundamental component that enables the model to capture and utilize contextual information during language processing tasks effectively. Making the attention scheme computation faster is one of the central questions to speed up the LLMs computation. It is well-known that
    
[^2]: 一种有效且可证明精确的0-1损失线性分类问题算法

    An efficient, provably exact algorithm for the 0-1 loss linear classification problem. (arXiv:2306.12344v1 [cs.LG])

    [http://arxiv.org/abs/2306.12344](http://arxiv.org/abs/2306.12344)

    该研究详细介绍了一种名为增量单元枚举（ICE）的算法，该算法可以精确解决定维度0-1损失线性分类问题。

    

    解决线性分类问题的算法具有悠久的历史，至少可以追溯到1936年的线性判别分析。对于线性可分数据，许多算法可以有效地得到相应的0-1损失分类问题的精确解，但对于非线性可分数据，已经证明这个问题在完全范围内是NP难的。所有替代方法都涉及某种形式的近似，包括使用0-1损失的代理（例如hinge或logistic损失）或近似的组合搜索，这些都不能保证完全解决问题。找到解决定维度0-1损失线性分类问题的全局最优解的有效算法仍然是一个未解决的问题。在本研究中，我们详细介绍了一个新算法的构建过程，增量单元枚举（ICE），它可以精确解决0-1损失分类问题。

    Algorithms for solving the linear classification problem have a long history, dating back at least to 1936 with linear discriminant analysis. For linearly separable data, many algorithms can obtain the exact solution to the corresponding 0-1 loss classification problem efficiently, but for data which is not linearly separable, it has been shown that this problem, in full generality, is NP-hard. Alternative approaches all involve approximations of some kind, including the use of surrogates for the 0-1 loss (for example, the hinge or logistic loss) or approximate combinatorial search, none of which can be guaranteed to solve the problem exactly. Finding efficient algorithms to obtain an exact i.e. globally optimal solution for the 0-1 loss linear classification problem with fixed dimension, remains an open problem. In research we report here, we detail the construction of a new algorithm, incremental cell enumeration (ICE), that can solve the 0-1 loss classification problem exactly in po
    
[^3]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    

