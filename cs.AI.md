# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification.](http://arxiv.org/abs/2305.04228) | 本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。 |

# 详细

[^1]: 基于抽象语法树的异构有向超图神经网络用于代码分类

    Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification. (arXiv:2305.04228v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2305.04228](http://arxiv.org/abs/2305.04228)

    本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。

    

    代码分类是程序理解和自动编码中的一个难题。由于程序的模糊语法和复杂语义，大多数现有研究使用基于抽象语法树（AST）和图神经网络（GNN）的技术创建代码表示用于代码分类。这些技术利用代码的结构和语义信息，但只考虑节点之间的成对关系，忽略了AST中节点之间已经存在的高阶相关性，可能导致代码结构信息的丢失。本研究提出使用异构有向超图（HDHG）表示AST，并使用异构有向超图神经网络（HDHGN）处理图形。HDHG保留了节点之间的高阶相关性，并更全面地编码了AST的语义和结构信息。HDHGN通过聚合不同节点的特征并使用不同的函数对其进行处理来对AST进行建模。在四个数据集上的实验表明，HDHG和HDHGN在代码分类任务中超越了现有方法。

    Code classification is a difficult issue in program understanding and automatic coding. Due to the elusive syntax and complicated semantics in programs, most existing studies use techniques based on abstract syntax tree (AST) and graph neural network (GNN) to create code representations for code classification. These techniques utilize the structure and semantic information of the code, but they only take into account pairwise associations and neglect the high-order correlations that already exist between nodes in the AST, which may result in the loss of code structural information. On the other hand, while a general hypergraph can encode high-order data correlations, it is homogeneous and undirected which will result in a lack of semantic and structural information such as node types, edge types, and directions between child nodes and parent nodes when modeling AST. In this study, we propose to represent AST as a heterogeneous directed hypergraph (HDHG) and process the graph by hetero
    

