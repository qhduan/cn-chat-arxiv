# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient compilation of expressive problem space specifications to neural network solvers](https://rss.arxiv.org/abs/2402.01353) | 本文提出了一种算法，可以将高级的问题空间规范编译为适合神经网络求解器的满足性查询，以解决神经网络验证中存在的嵌入间隙问题。 |

# 详细

[^1]: 将表达丰富的问题空间规范高效编译为神经网络求解器

    Efficient compilation of expressive problem space specifications to neural network solvers

    [https://rss.arxiv.org/abs/2402.01353](https://rss.arxiv.org/abs/2402.01353)

    本文提出了一种算法，可以将高级的问题空间规范编译为适合神经网络求解器的满足性查询，以解决神经网络验证中存在的嵌入间隙问题。

    

    最近的研究揭示了神经网络验证中存在的嵌入间隙。在间隙的一侧是一个关于网络行为的高级规范，由领域专家根据可解释的问题空间编写。在另一侧是一组逻辑上等价的可满足性查询，以适合神经网络求解器的形式表达在不可理解的嵌入空间中。在本文中，我们描述了一种将前者编译为后者的算法。我们探索和克服了针对神经网络求解器而不是标准SMT求解器所出现的问题。

    Recent work has described the presence of the embedding gap in neural network verification. On one side of the gap is a high-level specification about the network's behaviour, written by a domain expert in terms of the interpretable problem space. On the other side are a logically-equivalent set of satisfiability queries, expressed in the uninterpretable embedding space in a form suitable for neural network solvers. In this paper we describe an algorithm for compiling the former to the latter. We explore and overcome complications that arise from targeting neural network solvers as opposed to standard SMT solvers.
    

