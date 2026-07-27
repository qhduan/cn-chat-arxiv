# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks](https://arxiv.org/abs/2606.26212) | 本文提出了一种通用框架，利用图神经网络从有限群的凯莱图中直接学习并区分其代数性质（如阿贝尔性、幂零性和可解性），证明了图表示中蕴含的代数信息可通过GNN有效提取。 |
| [^2] | [Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks](https://arxiv.org/abs/2606.14954) | 本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。 |

# 详细

[^1]: 利用图神经网络从凯莱图学习代数性质的一般框架

    A General Framework for Learning Algebraic Properties from Cayley Graphs using Graph Neural Networks

    [https://arxiv.org/abs/2606.26212](https://arxiv.org/abs/2606.26212)

    本文提出了一种通用框架，利用图神经网络从有限群的凯莱图中直接学习并区分其代数性质（如阿贝尔性、幂零性和可解性），证明了图表示中蕴含的代数信息可通过GNN有效提取。

    

    论文摘要：arXiv:2606.26212v1 公告类型：新 摘要：文献[1]提出了一种图神经网络框架，用于从有限群的凯莱图表示预测其可解性。在本工作中，我们推广了该方法，并开发了一个与性质无关的框架，可直接从凯莱图学习有限群的代数性质。作为代表性案例研究，我们考虑了阿贝尔性、幂零性和可解性。通过使用通用的图神经网络架构和训练流程，我们探究了仅从基于图的表示中能恢复多少代数结构。在来自多个族群的有限群集合上的实验表明，该框架能够成功地从其关联的凯莱图中学习并区分多种代数性质。这些发现表明，图表示中编码了大量代数信息，并且可以通过图神经网络提取出来。更广泛地说，所提出的框架为从图结构数据中自动发现代数性质提供了一种通用方法。

    arXiv:2606.26212v1 Announce Type: new  Abstract: A Graph Neural Network (GNN) framework for predicting the solvability of finite groups from their Cayley graph representations was introduced in [1]. In the present work, we generalize this approach and develop a property-independent framework for learning algebraic properties of finite groups directly from Cayley graphs. As representative case studies, we consider abelianity, nilpotency, and solvability. Using a common GNN architecture and training pipeline, we investigate the extent to which algebraic structure can be recovered from graph-based representations alone. Results on a collection of finite groups drawn from several families demonstrate that the framework successfully learns and distinguishes multiple algebraic properties from their associated Cayley graphs. These findings suggest that substantial algebraic information is encoded in graph representations and can be extracted through GNNs. More broadly, the proposed framework 
    
[^2]: 数据科学中的表示成本：深度神经网络的基础与拟巴拿赫空间

    Representation Costs in Data Science: Foundations and the Quasi-Banach Spaces of Deep Neural Networks

    [https://arxiv.org/abs/2606.14954](https://arxiv.org/abs/2606.14954)

    本文提出了一个统一框架，通过参数空间正则化器分析数据科学中的表示成本，揭示了参数化方法与其原生函数空间之间的联系，并将核方法、小波和神经网络等经典方法统一为特例。

    

    我们开发了一个通用框架，用于通过参数空间正则化器分析参数化数据拟合方法的表示成本。从这一抽象视角出发，我们定义了任意参数化模型的表示成本，并揭示了它们所诱导的（原生）函数空间。这统一了近期关于数据拟合方法的函数空间视角。我们还证明，在该抽象设定下许多自然结论成立，包括参数化方法在其原生空间上的表示定理。该框架还严格地将参数化方法与其在充分过参数化下的等价非参数描述联系起来。经典方法及其原生空间，如核方法/再生核希尔伯特空间、小波/贝索夫空间以及浅层神经网络/变分空间，均作为我们抽象框架的特例出现。将表示成本研究“公理化”是一个副产品。

    arXiv:2606.14954v3 Announce Type: replace-cross  Abstract: We develop a general framework for analyzing representation costs of parametric data-fitting methods through their parameter-space regularizers. From this abstract perspective, we define representation costs for arbitrary parametric models and reveal their induced (native) function spaces. This unifies recent function-space views of data-fitting methods. We also prove that many natural results hold in this abstract setting, including representer theorems for parametric methods on their native spaces. The framework also rigorously connects parametric methods with their equivalent nonparametric descriptions under sufficient overparameterization. Classical methods and their native spaces, such as kernel methods / reproducing kernel Hilbert spaces, wavelets / Besov spaces, and shallow neural networks / variation spaces emerge as special cases of our abstract framework. A byproduct of "axiomatizing" the study of representation costs
    

