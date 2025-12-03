# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latent Attention for Linear Time Transformers](https://arxiv.org/abs/2402.17512) | 提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。 |

# 详细

[^1]: Latent Attention for Linear Time Transformers

    Latent Attention for Linear Time Transformers

    [https://arxiv.org/abs/2402.17512](https://arxiv.org/abs/2402.17512)

    提出了一种基于潜在向量定义注意力的方法，将标准transformer中的注意力机制的时间复杂度从二次方降低到与时间线性相关，表现与标准注意力媲美，但允许上下文窗口扩展到远远超出标准的范围。

    

    标准transformer中的注意力机制的时间复杂度随着序列长度的增加呈二次方增长。我们引入一种通过定义潜在向量的注意力来将其降低到与时间线性相关的方法。该方法可以轻松作为标准注意力机制的替代品。我们的“Latte Transformer”模型可用于双向和单向任务，因果版本允许一种在推理语言生成任务中内存和时间高效的递归实现。标准transformer的下一个标记预测随着序列长度线性增长，而Latte Transformer计算下一个标记所需的时间是恒定的。我们的方法的实证表现可与标准注意力媲美，但允许将上下文窗口扩展到远远超出标准注意力实际可行的范围。

    arXiv:2402.17512v1 Announce Type: new  Abstract: The time complexity of the standard attention mechanism in a transformer scales quadratically with the length of the sequence. We introduce a method to reduce this to linear scaling with time, based on defining attention via latent vectors. The method is readily usable as a drop-in replacement for the standard attention mechanism. Our "Latte Transformer" model can be implemented for both bidirectional and unidirectional tasks, with the causal version allowing a recurrent implementation which is memory and time-efficient during inference of language generation tasks. Whilst next token prediction scales linearly with the sequence length for a standard transformer, a Latte Transformer requires constant time to compute the next token. The empirical performance of our method is comparable to standard attention, yet allows scaling to context windows much larger than practical in standard attention.
    

