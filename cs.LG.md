# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Just Say the Name: Online Continual Learning with Category Names Only via Data Generation](https://arxiv.org/abs/2403.10853) | 提出了在线连续学习框架G-NoCL，采用生成数据并利用DIverSity和COmplexity enhancing ensemBlER（DISCOBER）进行数据融合，展示了其在在线连续学习基准测试中的优越性能。 |
| [^2] | [Exploring Prime Number Classification: Achieving High Recall Rate and Rapid Convergence with Sparse Encoding](https://arxiv.org/abs/2402.03363) | 通过稀疏编码和神经网络结构的组合，本文提出了一种在质数和非质数分类中实现高召回率和快速收敛的新方法，取得了令人满意的结果。 |
| [^3] | [Resampling Stochastic Gradient Descent Cheaply for Efficient Uncertainty Quantification.](http://arxiv.org/abs/2310.11065) | 本研究提出了两种低成本重采样的方法，用于构建随机梯度下降解的置信区间，这一方法可以有效减少计算工作量，并绕过现有方法中的混合条件。 |
| [^4] | [Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One.](http://arxiv.org/abs/2209.03282) | 本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。 |

# 详细

[^1]: 只说名称：通过数据生成实现仅利用类别名称进行在线连续学习

    Just Say the Name: Online Continual Learning with Category Names Only via Data Generation

    [https://arxiv.org/abs/2403.10853](https://arxiv.org/abs/2403.10853)

    提出了在线连续学习框架G-NoCL，采用生成数据并利用DIverSity和COmplexity enhancing ensemBlER（DISCOBER）进行数据融合，展示了其在在线连续学习基准测试中的优越性能。

    

    在现实世界的场景中，由于成本过高，对于连续学习进行大量手动注释是不切实际的。虽然之前的研究受到大规模网络监督训练的影响，建议在连续学习中利用网络抓取的数据，但这带来了诸如数据不平衡、使用限制和隐私问题等挑战。为了解决连续网络监督训练的风险，我们提出了一种在线连续学习框架 - 仅使用名称的生成式连续学习（G-NoCL）。所提出的G-NoCL使用一组生成器G以及学习者。当遇到新概念（例如，类别）时，G-NoCL采用新颖的样本复杂性引导数据合成技术DIverSity and COmplexity enhancing ensemBlER（DISCOBER）从生成的数据中最优抽样训练数据。通过大量实验，我们展示了DISCOBER在G-NoCL在线连续学习基准测试中表现出的优越性能，涵盖了In-Distributi。

    arXiv:2403.10853v1 Announce Type: cross  Abstract: In real-world scenarios, extensive manual annotation for continual learning is impractical due to prohibitive costs. Although prior arts, influenced by large-scale webly supervised training, suggest leveraging web-scraped data in continual learning, this poses challenges such as data imbalance, usage restrictions, and privacy concerns. Addressing the risks of continual webly supervised training, we present an online continual learning framework - Generative Name only Continual Learning (G-NoCL). The proposed G-NoCL uses a set of generators G along with the learner. When encountering new concepts (i.e., classes), G-NoCL employs the novel sample complexity-guided data ensembling technique DIverSity and COmplexity enhancing ensemBlER (DISCOBER) to optimally sample training data from generated data. Through extensive experimentation, we demonstrate superior performance of DISCOBER in G-NoCL online CL benchmarks, covering both In-Distributi
    
[^2]: 探索质数分类：使用稀疏编码实现高召回率和快速收敛

    Exploring Prime Number Classification: Achieving High Recall Rate and Rapid Convergence with Sparse Encoding

    [https://arxiv.org/abs/2402.03363](https://arxiv.org/abs/2402.03363)

    通过稀疏编码和神经网络结构的组合，本文提出了一种在质数和非质数分类中实现高召回率和快速收敛的新方法，取得了令人满意的结果。

    

    本文提出了一种新颖的方法，结合机器学习和数论，在质数和非质数分类上进行研究。我们的研究核心是开发一种高度稀疏的编码方法，与传统的神经网络结构相结合。这种组合取得了令人满意的结果，在识别质数时达到了超过99\%的召回率，在识别非质数时达到了79\%的召回率，这些数字是从本质上不平衡的顺序整数序列中得出的，并且在完成单个训练周期之前迅速收敛。我们使用 $10^6$ 个整数进行训练，从指定的整数开始，然后在一个不同范围的 $2 \times 10^6$ 个整数上进行测试，范围从 $10^6$ 到 $3 \times 10^6$，偏移量相同。尽管受限于资源的内存容量，限制我们的分析跨越了 $3\times10^6$，但我们认为我们的研究对机器学习在......的应用做出了贡献

    This paper presents a novel approach at the intersection of machine learning and number theory, focusing on the classification of prime and non-prime numbers. At the core of our research is the development of a highly sparse encoding method, integrated with conventional neural network architectures. This combination has shown promising results, achieving a recall of over 99\% in identifying prime numbers and 79\% for non-prime numbers from an inherently imbalanced sequential series of integers, while exhibiting rapid model convergence before the completion of a single training epoch. We performed training using $10^6$ integers starting from a specified integer and tested on a different range of $2 \times 10^6$ integers extending from $10^6$ to $3 \times 10^6$, offset by the same starting integer. While constrained by the memory capacity of our resources, which limited our analysis to a span of $3\times10^6$, we believe that our study contribute to the application of machine learning in
    
[^3]: 低成本重采样随机梯度下降用于高效不确定性量化

    Resampling Stochastic Gradient Descent Cheaply for Efficient Uncertainty Quantification. (arXiv:2310.11065v1 [stat.ML])

    [http://arxiv.org/abs/2310.11065](http://arxiv.org/abs/2310.11065)

    本研究提出了两种低成本重采样的方法，用于构建随机梯度下降解的置信区间，这一方法可以有效减少计算工作量，并绕过现有方法中的混合条件。

    

    随机梯度下降（SGD）或随机逼近在模型训练和随机优化中被广泛使用。虽然有大量关于其收敛性分析的文献，但对从SGD获得的解进行推断的研究只是最近才开始，但由于对不确定性量化的日益需求而变得重要。我们研究了两种计算上廉价的基于重采样的方法来构建SGD解的置信区间。一个方法通过从数据中进行替换重采样来使用多个但少量的SGD并行进行操作，另一个方法以在线方式进行操作。我们的方法可以被视为对已建立的Bootstrap方案进行增强，以显着减少重采样需求方面的计算工作量，同时绕过现有批处理方法中复杂的混合条件。我们通过最近的所谓低成本bootstrap思想和SGD的Berry-Esseen型边界来实现这些目标。

    Stochastic gradient descent (SGD) or stochastic approximation has been widely used in model training and stochastic optimization. While there is a huge literature on analyzing its convergence, inference on the obtained solutions from SGD has only been recently studied, yet is important due to the growing need for uncertainty quantification. We investigate two computationally cheap resampling-based methods to construct confidence intervals for SGD solutions. One uses multiple, but few, SGDs in parallel via resampling with replacement from the data, and another operates this in an online fashion. Our methods can be regarded as enhancements of established bootstrap schemes to substantially reduce the computation effort in terms of resampling requirements, while at the same time bypassing the intricate mixing conditions in existing batching methods. We achieve these via a recent so-called cheap bootstrap idea and Berry-Esseen-type bound for SGD.
    
[^4]: 二次梯度：将梯度算法和牛顿法融合为一体

    Quadratic Gradient: Combining Gradient Algorithms and Newton's Method as One. (arXiv:2209.03282v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.03282](http://arxiv.org/abs/2209.03282)

    本文提出了一种基于对角矩阵的二次梯度，可以加速梯度的收敛速度，在实验中表现良好。研究者还推测海森矩阵与学习率之间可能存在关系。

    

    使用一列与梯度相同大小的列向量，而不是仅使用一个浮点数来加速每个梯度元素的不同速率，可能对牛顿法的线搜索技术不足。此外，使用一个与海森矩阵大小相同的正方形矩阵来纠正海森矩阵可能是有用的。Chiang提出了一种介于列向量和正方形矩阵之间的东西，即对角矩阵，来加速梯度，并进一步提出了一种更快的梯度变体，称为二次梯度。在本文中，我们提出一种构建新版本的二次梯度的新方法。这个新的二次梯度不满足固定海森牛顿法的收敛条件。然而，实验结果显示，它有时比原始方法的收敛速度更快。此外，Chiang推测海森矩阵与学习率f之间可能存在关系。

    It might be inadequate for the line search technique for Newton's method to use only one floating point number. A column vector of the same size as the gradient might be better than a mere float number to accelerate each of the gradient elements with different rates. Moreover, a square matrix of the same order as the Hessian matrix might be helpful to correct the Hessian matrix. Chiang applied something between a column vector and a square matrix, namely a diagonal matrix, to accelerate the gradient and further proposed a faster gradient variant called quadratic gradient. In this paper, we present a new way to build a new version of the quadratic gradient. This new quadratic gradient doesn't satisfy the convergence conditions of the fixed Hessian Newton's method. However, experimental results show that it sometimes has a better performance than the original one in convergence rate. Also, Chiang speculates that there might be a relation between the Hessian matrix and the learning rate f
    

