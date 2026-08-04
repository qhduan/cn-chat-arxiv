# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) | 可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。 |
| [^2] | [Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One.](http://arxiv.org/abs/2209.03282) | 本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。 |

# 详细

[^1]: 可微分编程的要素

    The Elements of Differentiable Programming

    [https://arxiv.org/abs/2403.14606](https://arxiv.org/abs/2403.14606)

    可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。

    

    人工智能最近取得了显著进展，这得益于大型模型、庞大数据集、加速硬件，以及可微分编程的变革性力量。这种新的编程范式使复杂计算机程序（包括具有控制流和数据结构的程序）能够进行端对端的微分，从而实现对程序参数的基于梯度的优化。不仅仅是程序的微分，可微分编程也包括了程序优化、概率等多个领域的概念。本书介绍了可微分编程所需的基本概念，并采用了优化和概率两个主要视角进行阐述。

    arXiv:2403.14606v1 Announce Type: new  Abstract: Artificial intelligence has recently experienced remarkable advances, fueled by large models, vast datasets, accelerated hardware, and, last but not least, the transformative power of differentiable programming. This new programming paradigm enables end-to-end differentiation of complex computer programs (including those with control flows and data structures), making gradient-based optimization of program parameters possible. As an emerging paradigm, differentiable programming builds upon several areas of computer science and applied mathematics, including automatic differentiation, graphical models, optimization and statistics. This book presents a comprehensive review of the fundamental concepts useful for differentiable programming. We adopt two main perspectives, that of optimization and that of probability, with clear analogies between the two. Differentiable programming is not merely the differentiation of programs, but also the t
    
[^2]: 二次梯度：将梯度算法和牛顿法融合为一体

    Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One. (arXiv:2209.03282v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.03282](http://arxiv.org/abs/2209.03282)

    本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。

    

    使用一列与梯度相同大小的列向量，而不是仅使用一个浮点数来加速每个梯度元素的不同速率，可能对牛顿法的线搜索技术不足。此外，使用一个与海森矩阵大小相同的正方形矩阵来纠正海森矩阵可能是有用的。Chiang提出了一种介于列向量和正方形矩阵之间的东西，即对角矩阵，来加速梯度，并进一步提出了一种更快的梯度变体，称为二次梯度。在本文中，我们提出一种构建新版本的二次梯度的新方法。这个新的二次梯度不满足固定海森牛顿法的收敛条件。然而，实验结果显示，它有时比原始方法的收敛速度更快。此外，Chiang推测海森矩阵与学习率f之间可能存在关系。

    It might be inadequate for the line search technique for Newton's method to use only one floating point number. A column vector of the same size as the gradient might be better than a mere float number to accelerate each of the gradient elements with different rates. Moreover, a square matrix of the same order as the Hessian matrix might be helpful to correct the Hessian matrix. Chiang applied something between a column vector and a square matrix, namely a diagonal matrix, to accelerate the gradient and further proposed a faster gradient variant called quadratic gradient. In this paper, we present a new way to build a new version of the quadratic gradient. This new quadratic gradient doesn't satisfy the convergence conditions of the fixed Hessian Newton's method. However, experimental results show that it sometimes has a better performance than the original one in convergence rate. Also, Chiang speculates that there might be a relation between the Hessian matrix and the learning rate f
    

