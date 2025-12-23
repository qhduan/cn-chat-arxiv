# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs](https://arxiv.org/abs/2403.15676) | 该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。 |
| [^2] | [Structured Language Generation Model for Robust Structure Prediction](https://arxiv.org/abs/2402.08971) | 鲁棒结构预测的结构化语言生成模型通过新的损失函数和推理方法的混合，成功提高了结构化输出的泛化能力，并且可以在没有数据集信息的情况下工作，并且减少了格式错误。 |

# 详细

[^1]: AC4：用于ZKP中电路约束的代数计算检查器

    AC4: Algebraic Computation Checker for Circuit Constraints in ZKPs

    [https://arxiv.org/abs/2403.15676](https://arxiv.org/abs/2403.15676)

    该论文引入了一种新方法，通过将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统，以精确定位ZKP电路中两种不同类型的错误。

    

    ZKP系统已经引起了人们的关注，在当代密码学中发挥着基础性作用。 Zk-SNARK协议主导了ZKP的使用，通常通过算术电路编程范式实现。然而，欠约束或过约束的电路可能导致错误。 欠约束的电路指的是缺乏必要约束的电路，导致电路中出现意外解决方案，并导致验证者接受错误见证。 过约束的电路是指约束过度的电路，导致电路缺乏必要的解决方案，并导致验证者接受没有见证，使电路毫无意义。 本文介绍了一种新方法，用于找出ZKP电路中两种不同类型的错误。 该方法涉及将算术电路约束编码为多项式方程系统，并通过代数计算在有限域上解决多项式方程系统。

    arXiv:2403.15676v1 Announce Type: cross  Abstract: ZKP systems have surged attention and held a fundamental role in contemporary cryptography. Zk-SNARK protocols dominate the ZKP usage, often implemented through arithmetic circuit programming paradigm. However, underconstrained or overconstrained circuits may lead to bugs. Underconstrained circuits refer to circuits that lack the necessary constraints, resulting in unexpected solutions in the circuit and causing the verifier to accept a bogus witness. Overconstrained circuits refer to circuits that are constrained excessively, resulting in the circuit lacking necessary solutions and causing the verifier to accept no witness, rendering the circuit meaningless. This paper introduces a novel approach for pinpointing two distinct types of bugs in ZKP circuits. The method involves encoding the arithmetic circuit constraints to polynomial equation systems and solving polynomial equation systems over a finite field by algebraic computation. T
    
[^2]: 鲁棒结构预测的结构化语言生成模型

    Structured Language Generation Model for Robust Structure Prediction

    [https://arxiv.org/abs/2402.08971](https://arxiv.org/abs/2402.08971)

    鲁棒结构预测的结构化语言生成模型通过新的损失函数和推理方法的混合，成功提高了结构化输出的泛化能力，并且可以在没有数据集信息的情况下工作，并且减少了格式错误。

    

    我们提出了一种结构化语言生成模型（SLGM），通过新的损失函数和推理方法的混合来改善结构化输出的泛化能力。以往的结构预测研究（如NER，RE）利用了显式的数据集信息，这可以提高性能，但可能会对现实世界中的鲁棒泛化性产生挑战。相反，我们的模型间接地提供了有关数据的通用格式信息。利用格式信息，我们可以通过损失校准和格式化解码将序列到序列问题简化为分类问题。我们的实验结果表明，SLGM在没有数据集信息的情况下成功保持了性能，并且显示出较少的格式错误。我们还展示了我们的模型可以像适配器一样在各个数据集上工作，而无需额外的训练。

    arXiv:2402.08971v1 Announce Type: new Abstract: We propose Structured Language Generation Model (SLGM), a mixture of new loss function and inference method for better generalization of structured outputs. Previous studies on structure prediction (e.g. NER, RE) make use of explicit dataset information, which would boost performance, yet it might pose challenges to robust generalization in real-world situations. Instead, our model gives generalized format information about data indirectly. With format information, we could reduce sequence-to-sequence problem into classification problem via loss calibration and formatted decoding. Our experimental results showed SLGM successfully maintain performance without dataset information, and showed much less format errors. We also showed our model can work like adapters on individual dataset, with no additional training.
    

