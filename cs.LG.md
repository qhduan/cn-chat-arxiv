# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Neural Networks: A Formulation Via Non-Archimedean Analysis](https://arxiv.org/abs/2402.00094) | 该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。 |
| [^2] | [Information Leakage Detection through Approximate Bayes-optimal Prediction.](http://arxiv.org/abs/2401.14283) | 本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。 |

# 详细

[^1]: 深度神经网络: 非阿基米德分析的表述方式

    Deep Neural Networks: A Formulation Via Non-Archimedean Analysis

    [https://arxiv.org/abs/2402.00094](https://arxiv.org/abs/2402.00094)

    该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。

    

    我们引入了一种新的深度神经网络（DNNs），采用多层树状结构的架构。这些架构使用非阿基米德局部域的整数环中的数字进行编码。这些环具有自然的层次结构，类似无限根树。这些环上的自然态射使我们能够构建有限的多层架构。新的DNNs是对在所提到的环上定义的实值函数的稳健的普遍逼近器。我们还证明了DNNs也是对在单位区间上定义的实值平方可积函数的稳健的普遍逼近器。

    We introduce a new class of deep neural networks (DNNs) with multilayered tree-like architectures. The architectures are codified using numbers from the ring of integers of non-Archimdean local fields. These rings have a natural hierarchical organization as infinite rooted trees. Natural morphisms on these rings allow us to construct finite multilayered architectures. The new DNNs are robust universal approximators of real-valued functions defined on the mentioned rings. We also show that the DNNs are robust universal approximators of real-valued square-integrable functions defined in the unit interval.
    
[^2]: 通过近似贝叶斯最优预测检测信息泄漏

    Information Leakage Detection through Approximate Bayes-optimal Prediction. (arXiv:2401.14283v1 [stat.ML])

    [http://arxiv.org/abs/2401.14283](http://arxiv.org/abs/2401.14283)

    本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。

    

    在今天的以数据驱动的世界中，公开可获得的信息的增加加剧了信息泄漏（IL）的挑战，引发了安全问题。IL涉及通过系统的可观察信息无意地将秘密（敏感）信息暴露给未经授权的方，传统的统计方法通过估计可观察信息和秘密信息之间的互信息（MI）来检测IL，面临维度灾难、收敛、计算复杂度和MI估计错误等挑战。此外，虽然新兴的监督机器学习（ML）方法在二进制系统敏感信息的检测上有效，但缺乏一个全面的理论框架。为了解决这些限制，我们使用统计学习理论和信息论建立了一个理论框架来准确量化和检测IL。我们证明了可以通过近似贝叶斯预测的对数损失和准确性来准确估计MI。

    In today's data-driven world, the proliferation of publicly available information intensifies the challenge of information leakage (IL), raising security concerns. IL involves unintentionally exposing secret (sensitive) information to unauthorized parties via systems' observable information. Conventional statistical approaches, which estimate mutual information (MI) between observable and secret information for detecting IL, face challenges such as the curse of dimensionality, convergence, computational complexity, and MI misestimation. Furthermore, emerging supervised machine learning (ML) methods, though effective, are limited to binary system-sensitive information and lack a comprehensive theoretical framework. To address these limitations, we establish a theoretical framework using statistical learning theory and information theory to accurately quantify and detect IL. We demonstrate that MI can be accurately estimated by approximating the log-loss and accuracy of the Bayes predict
    

