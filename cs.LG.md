# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved Hardness Results for Learning Intersections of Halfspaces](https://arxiv.org/abs/2402.15995) | 我们通过展示学习在维度N中的$\omega(\log \log N)$个半空间甚至需要超多项式时间的标准假设，显著缩小了这一差距 |
| [^2] | [Statistical Test for Generated Hypotheses by Diffusion Models](https://arxiv.org/abs/2402.11789) | 本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。 |
| [^3] | [Consistency of Lloyd's Algorithm Under Perturbations.](http://arxiv.org/abs/2309.00578) | 该论文研究了Lloyd算法在扰动样本上的一致性，证明了在适当初始化和扰动相对于亚高斯噪声较小的假设下，算法在O(log(n))次迭代后的错聚类率在指数下界受限。 |

# 详细

[^1]: 改进学习半空间交集的困难性结果

    Improved Hardness Results for Learning Intersections of Halfspaces

    [https://arxiv.org/abs/2402.15995](https://arxiv.org/abs/2402.15995)

    我们通过展示学习在维度N中的$\omega(\log \log N)$个半空间甚至需要超多项式时间的标准假设，显著缩小了这一差距

    

    我们展示了在不正确设置中学习半空间交集的弱学习下界，这些下界非常强大（并且令人惊讶地简单）。关于这个问题知之甚少。例如，甚至不知道是否存在一个多项式时间算法来学习仅两个半空间的交集。另一方面，基于良好建立的假设（如近似最坏情况的格问题或Feige的3SAT假设的变体）的下界仅对超对数个半空间的交集已知（或者由已有结果暗示）。

    arXiv:2402.15995v1 Announce Type: cross  Abstract: We show strong (and surprisingly simple) lower bounds for weakly learning intersections of halfspaces in the improper setting. Strikingly little is known about this problem. For instance, it is not even known if there is a polynomial-time algorithm for learning the intersection of only two halfspaces. On the other hand, lower bounds based on well-established assumptions (such as approximating worst-case lattice problems or variants of Feige's 3SAT hypothesis) are only known (or are implied by existing results) for the intersection of super-logarithmically many halfspaces [KS09,KS06,DSS16]. With intersections of fewer halfspaces being only ruled out under less standard assumptions [DV21] (such as the existence of local pseudo-random generators with large stretch). We significantly narrow this gap by showing that even learning $\omega(\log \log N)$ halfspaces in dimension $N$ takes super-polynomial time under standard assumptions on wors
    
[^2]: 通过扩散模型生成的假设的统计检验

    Statistical Test for Generated Hypotheses by Diffusion Models

    [https://arxiv.org/abs/2402.11789](https://arxiv.org/abs/2402.11789)

    本研究提出了一种统计检验方法，通过选择性推断框架，在考虑生成图像是由训练的扩散模型产生的条件下，量化医学图像诊断结果的可靠性。

    

    AI的增强性能加速了其融入科学研究。特别是，利用生成式AI创建科学假设是很有前途的，并且正在越来越多地应用于各个领域。然而，当使用AI生成的假设进行关键决策（如医学诊断）时，验证它们的可靠性至关重要。在本研究中，我们考虑使用扩散模型生成的图像进行医学诊断任务，并提出了一种统计检验来量化其可靠性。所提出的统计检验的基本思想是使用选择性推断框架，我们考虑在生成的图像是由经过训练的扩散模型产生的这一事实条件下的统计检验。利用所提出的方法，医学图像诊断结果的统计可靠性可以以p值的形式量化，从而实现在控制错误率的情况下进行决策。

    arXiv:2402.11789v1 Announce Type: cross  Abstract: The enhanced performance of AI has accelerated its integration into scientific research. In particular, the use of generative AI to create scientific hypotheses is promising and is increasingly being applied across various fields. However, when employing AI-generated hypotheses for critical decisions, such as medical diagnoses, verifying their reliability is crucial. In this study, we consider a medical diagnostic task using generated images by diffusion models, and propose a statistical test to quantify its reliability. The basic idea behind the proposed statistical test is to employ a selective inference framework, where we consider a statistical test conditional on the fact that the generated images are produced by a trained diffusion model. Using the proposed method, the statistical reliability of medical image diagnostic results can be quantified in the form of a p-value, allowing for decision-making with a controlled error rate. 
    
[^3]: Lloyd算法在扰动下的一致性

    Consistency of Lloyd's Algorithm Under Perturbations. (arXiv:2309.00578v1 [cs.LG])

    [http://arxiv.org/abs/2309.00578](http://arxiv.org/abs/2309.00578)

    该论文研究了Lloyd算法在扰动样本上的一致性，证明了在适当初始化和扰动相对于亚高斯噪声较小的假设下，算法在O(log(n))次迭代后的错聚类率在指数下界受限。

    

    在无监督学习的背景下，Lloyd算法是最常用的聚类算法之一。它启发了大量的工作，研究了算法在不同设置下对地面真实聚类的正确性。特别是在2016年，卢和周表明，在正确初始化算法的前提下，Lloyd算法在从亚高斯混合中独立抽取的n个样本上的错聚类率在O(log(n))次迭代后指数下界受限。然而，在许多应用中，真实样本是未观测到的，需要通过预处理流水线（如合适的数据矩阵上的谱方法）从数据中学习。我们展示了在适当初始化和扰动相对于亚高斯噪声较小的假设下，Lloyd算法在从亚高斯混合中扰动样本上的错聚类率在O(log(n))次迭代后同样指数下界受限。

    In the context of unsupervised learning, Lloyd's algorithm is one of the most widely used clustering algorithms. It has inspired a plethora of work investigating the correctness of the algorithm under various settings with ground truth clusters. In particular, in 2016, Lu and Zhou have shown that the mis-clustering rate of Lloyd's algorithm on $n$ independent samples from a sub-Gaussian mixture is exponentially bounded after $O(\log(n))$ iterations, assuming proper initialization of the algorithm. However, in many applications, the true samples are unobserved and need to be learned from the data via pre-processing pipelines such as spectral methods on appropriate data matrices. We show that the mis-clustering rate of Lloyd's algorithm on perturbed samples from a sub-Gaussian mixture is also exponentially bounded after $O(\log(n))$ iterations under the assumptions of proper initialization and that the perturbation is small relative to the sub-Gaussian noise. In canonical settings with g
    

