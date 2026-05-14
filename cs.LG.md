# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Impact of LoRA on the Emergence of Clusters in Transformers](https://arxiv.org/abs/2402.15415) | 本文利用转换器数学框架探讨了LoRA算法对Token聚类结构动态的影响，发现在不同参数下，修改后的注意力矩阵动态的聚类表现出较长时间的显著差异，但仍在短时间内保持密切相似。 |
| [^2] | [Digital Computers Break the Curse of Dimensionality: Adaptive Bounds via Finite Geometry](https://arxiv.org/abs/2402.05576) | 通过利用离散结构，本论文以真实计算机上的实现为基础，打破了统计学习中的维度诅咒，并给出了无维度率的新的泛化界限。 |
| [^3] | [The Score-Difference Flow for Implicit Generative Modeling.](http://arxiv.org/abs/2304.12906) | 本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。 |

# 详细

[^1]: LoRA对转换器中聚类的影响

    The Impact of LoRA on the Emergence of Clusters in Transformers

    [https://arxiv.org/abs/2402.15415](https://arxiv.org/abs/2402.15415)

    本文利用转换器数学框架探讨了LoRA算法对Token聚类结构动态的影响，发现在不同参数下，修改后的注意力矩阵动态的聚类表现出较长时间的显著差异，但仍在短时间内保持密切相似。

    

    在本文中，我们利用\citet{sander2022sinkformers,geshkovski2023emergence,geshkovski2023mathematical}提出的转换器数学框架，探讨注意力参数和初始标记值的变化如何影响标记聚类的结构动态。我们的分析表明，虽然修改后的注意力矩阵动态中的聚类可能在较长时间内与原始聚类差异显著，但在较短时间间隔内，它们在参数差异的影响下仍保持密切相似。这项工作通过LoRA算法\cite{hu2021lora,peft}的实际应用，为微调领域做出了贡献，增进了我们对LoRA增强的Transformer模型行为的理解。

    arXiv:2402.15415v1 Announce Type: new  Abstract: In this paper, we employ the mathematical framework on Transformers developed by \citet{sander2022sinkformers,geshkovski2023emergence,geshkovski2023mathematical} to explore how variations in attention parameters and initial token values impact the structural dynamics of token clusters. Our analysis demonstrates that while the clusters within a modified attention matrix dynamics can exhibit significant divergence from the original over extended periods, they maintain close similarities over shorter intervals, depending on the parameter differences. This work contributes to the fine-tuning field through practical applications to the LoRA algorithm \cite{hu2021lora,peft}, enhancing our understanding of the behavior of LoRA-enhanced Transformer models.
    
[^2]: 数字计算机打破维度诅咒：通过有限几何的自适应界限

    Digital Computers Break the Curse of Dimensionality: Adaptive Bounds via Finite Geometry

    [https://arxiv.org/abs/2402.05576](https://arxiv.org/abs/2402.05576)

    通过利用离散结构，本论文以真实计算机上的实现为基础，打破了统计学习中的维度诅咒，并给出了无维度率的新的泛化界限。

    

    许多机器学习的基础是建立在理想情况下的前提下，即所有的输入和输出空间都是无穷的，例如$\mathbb{R}^d$。然而，由于有限的机器精度、舍入和有限的存储空间等数字计算机的限制，实际情况下这个核心假设往往被违背。简而言之，数字计算机在$\mathbb{R}^d$上操作的是有限的网格。通过利用这些离散结构，我们展示了在实际计算机上实现模型时，统计学习中的维度诅咒被系统地打破。因此，我们针对在真实世界机器上实现的核函数和深度ReLU MLP回归器获得了新的无维度率的泛化界限。我们的结果应用了一种新的非渐进测度集中性结果，该结果给出了概率测度和其在$N$个独立同分布样本上的经验版本之间的距离为$1$-Wasserstein距离的集中性。

    Many of the foundations of machine learning rely on the idealized premise that all input and output spaces are infinite, e.g.~$\mathbb{R}^d$. This core assumption is systematically violated in practice due to digital computing limitations from finite machine precision, rounding, and limited RAM. In short, digital computers operate on finite grids in $\mathbb{R}^d$. By exploiting these discrete structures, we show the curse of dimensionality in statistical learning is systematically broken when models are implemented on real computers. Consequentially, we obtain new generalization bounds with dimension-free rates for kernel and deep ReLU MLP regressors, which are implemented on real-world machines.   Our results are derived using a new non-asymptotic concentration of measure result between a probability measure over any finite metric space and its empirical version associated with $N$ i.i.d. samples when measured in the $1$-Wasserstein distance. Unlike standard concentration of measure 
    
[^3]: 评分差值流模型用于隐式生成建模

    The Score-Difference Flow for Implicit Generative Modeling. (arXiv:2304.12906v1 [cs.LG])

    [http://arxiv.org/abs/2304.12906](http://arxiv.org/abs/2304.12906)

    本文提出了一种新的评分差异流模型(SD flow)，它可以最优地减少两个分布之间的散度，同时解决Schr​​ödinger桥问题。与去噪扩散模型不同，它没有对先验分布施加任何限制，在一些基准数据集中优于其他方法。

    

    隐式生成建模(IGM)旨在生成符合目标数据分布特征的合成数据样本。最近的研究(例如评分匹配网络、扩散模型)从通过环境空间中的动态扰动或流将合成源数据推向目标分布的角度解决了IGM问题。我们引入了任意目标和源分布之间的评分差异(SD)作为流，它可以最优地减少它们之间的Kullback-Leibler散度，同时解决Schr​​ödinger桥问题。我们将SD流应用于方便的代理分布，当且仅当原始分布对齐时，它们是对齐的。我们在某些条件下展示了这种公式与去噪扩散模型的形式一致性。然而，与扩散模型不同，SD流没有对先验分布施加任何限制。我们还表明，在无限辨别器能力的极限下，生成对抗网络的训练包含SD流。我们的实验表明，SD流在几个基准数据集上优于先前的最新技术。

    Implicit generative modeling (IGM) aims to produce samples of synthetic data matching the characteristics of a target data distribution. Recent work (e.g. score-matching networks, diffusion models) has approached the IGM problem from the perspective of pushing synthetic source data toward the target distribution via dynamical perturbations or flows in the ambient space. We introduce the score difference (SD) between arbitrary target and source distributions as a flow that optimally reduces the Kullback-Leibler divergence between them while also solving the Schr\"odinger bridge problem. We apply the SD flow to convenient proxy distributions, which are aligned if and only if the original distributions are aligned. We demonstrate the formal equivalence of this formulation to denoising diffusion models under certain conditions. However, unlike diffusion models, SD flow places no restrictions on the prior distribution. We also show that the training of generative adversarial networks includ
    

