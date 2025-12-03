# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Attention for Linear Time Transformers](https://arxiv.org/abs/2402.17512) | 提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。 |
| [^2] | [Data Augmentation in the Underparameterized and Overparameterized Regimes.](http://arxiv.org/abs/2202.09134) | 这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。 |

# 详细

[^1]: Latent Attention for Linear Time Transformers

    Latent Attention for Linear Time Transformers

    [https://arxiv.org/abs/2402.17512](https://arxiv.org/abs/2402.17512)

    提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。

    

    标准transformer中的注意力机制的时间复杂度随着序列长度的增加呈二次方增长。我们引入一种通过定义潜在向量的注意力来将其降低到与时间线性相关的方法。该方法可以轻松作为标准注意力机制的替代品。我们的“Latte Transformer”模型可用于双向和单向任务，因果版本允许一种在推理语言生成任务中内存和时间高效的递归实现。标准transformer的下一个标记预测随着序列长度线性增长，而Latte Transformer计算下一个标记所需的时间是恒定的。我们的方法的实证表现可与标准注意力媲美，但允许将上下文窗口扩展到远远超出标准注意力实际可行的范围。

    arXiv:2402.17512v1 Announce Type: new  Abstract: The time complexity of the standard attention mechanism in a transformer scales quadratically with the length of the sequence. We introduce a method to reduce this to linear scaling with time, based on defining attention via latent vectors. The method is readily usable as a drop-in replacement for the standard attention mechanism. Our "Latte Transformer" model can be implemented for both bidirectional and unidirectional tasks, with the causal version allowing a recurrent implementation which is memory and time-efficient during inference of language generation tasks. Whilst next token prediction scales linearly with the sequence length for a standard transformer, a Latte Transformer requires constant time to compute the next token. The empirical performance of our method is comparable to standard attention, yet allows scaling to context windows much larger than practical in standard attention.
    
[^2]: 在欠参数化和过参数化的模式中的数据增强

    Data Augmentation in the Underparameterized and Overparameterized Regimes. (arXiv:2202.09134v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2202.09134](http://arxiv.org/abs/2202.09134)

    这项研究提供了数据增强如何影响估计的方差和极限分布的确切量化结果，发现数据增强可能会增加估计的不确定性，并且其效果取决于多个因素。同时，该研究还通过随机转换的高维随机向量的函数的极限定理进行了证明。

    

    我们提供了确切量化数据增强如何影响估计的方差和极限分布的结果，并详细分析了几个具体模型。结果证实了机器学习实践中的一些观察，但也得出了意外的发现：数据增强可能会增加而不是减少估计的不确定性，比如经验预测风险。它可以充当正则化器，但在某些高维问题中却无法实现，并且可能会改变经验风险的双重下降峰值。总的来说，分析表明数据增强被赋予的几个属性要么是真的，要么是假的，而是取决于多个因素的组合-特别是数据分布，估计器的属性以及样本大小，增强数量和维数的相互作用。我们的主要理论工具是随机转换的高维随机向量的函数的极限定理。

    We provide results that exactly quantify how data augmentation affects the variance and limiting distribution of estimates, and analyze several specific models in detail. The results confirm some observations made in machine learning practice, but also lead to unexpected findings: Data augmentation may increase rather than decrease the uncertainty of estimates, such as the empirical prediction risk. It can act as a regularizer, but fails to do so in certain high-dimensional problems, and it may shift the double-descent peak of an empirical risk. Overall, the analysis shows that several properties data augmentation has been attributed with are not either true or false, but rather depend on a combination of factors -- notably the data distribution, the properties of the estimator, and the interplay of sample size, number of augmentations, and dimension. Our main theoretical tool is a limit theorem for functions of randomly transformed, high-dimensional random vectors. The proof draws on 
    

