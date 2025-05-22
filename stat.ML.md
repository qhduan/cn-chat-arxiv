# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://arxiv.org/abs/2403.04082) | 通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断 |
| [^2] | [Uncertainty quantification in fine-tuned LLMs using LoRA ensembles](https://arxiv.org/abs/2402.12264) | 使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。 |
| [^3] | [Learning Memory Kernels in Generalized Langevin Equations](https://arxiv.org/abs/2402.11705) | 提出一种学习广义朗之万方程中记忆核的新方法，通过正则化Prony方法估计相关函数并在Sobolev范数Loss函数和RKHS正则化下实现回归，在指数加权的$L^2$空间内获得改进性能，对比其他回归估计器展示了其优越性。 |

# 详细

[^1]: 通过插值进行推断：对比表示可证明启用规划和推断

    Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference

    [https://arxiv.org/abs/2403.04082](https://arxiv.org/abs/2403.04082)

    通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断

    

    给定时间序列数据，我们如何回答诸如“未来会发生什么？”和“我们是如何到达这里的？”这类概率推断问题在观测值为高维时具有挑战性。本文展示了这些问题如何通过学习表示的紧凑闭式解决方案。关键思想是将对比学习的变体应用于时间序列数据。之前的工作已经表明，通过对比学习学到的表示编码了概率比。通过将之前的工作扩展以表明表示的边际分布是高斯分布，我们随后证明表示的联合分布也是高斯分布。这些结果共同表明，通过时间对比学习学到的表示遵循高斯马尔可夫链，一种图形模型，其中对表示进行的推断（例如预测、规划）对应于反演低维分布。

    arXiv:2403.04082v1 Announce Type: new  Abstract: Given time series data, how can we answer questions like "what will happen in the future?" and "how did we get here?" These sorts of probabilistic inference questions are challenging when observations are high-dimensional. In this paper, we show how these questions can have compact, closed form solutions in terms of learned representations. The key idea is to apply a variant of contrastive learning to time series data. Prior work already shows that the representations learned by contrastive learning encode a probability ratio. By extending prior work to show that the marginal distribution over representations is Gaussian, we can then prove that joint distribution of representations is also Gaussian. Taken together, these results show that representations learned via temporal contrastive learning follow a Gauss-Markov chain, a graphical model where inference (e.g., prediction, planning) over representations corresponds to inverting a low-
    
[^2]: 使用LoRA集成在精调LLMs中的不确定性量化

    Uncertainty quantification in fine-tuned LLMs using LoRA ensembles

    [https://arxiv.org/abs/2402.12264](https://arxiv.org/abs/2402.12264)

    使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。

    

    精调大型语言模型可以提高特定任务的性能，尽管对于精调模型学到了什么、遗忘了什么以及如何信任其预测仍然缺乏一个一般的理解。我们提出了使用计算效率高的低秩适应集成对精调LLMs进行基于后验逼近的原则性不确定性量化。我们使用基于Mistral-7b的低秩适应集成分析了三个常见的多项选择数据集，并对其在精调过程中和之后对不同目标领域的感知复杂性和模型效能进行了定量和定性的结论。具体而言，基于数值实验支持，我们对那些对于给定架构难以学习的数据领域的熵不确定性度量提出了假设。

    arXiv:2402.12264v1 Announce Type: cross  Abstract: Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and model efficacy on the different target domains during and after fine-tuning. In particular, backed by the numerical experiments, we hypothesise about signals from entropic uncertainty measures for data domains that are inherently difficult for a given architecture to learn.
    
[^3]: 在广义朗之万方程中学习记忆核

    Learning Memory Kernels in Generalized Langevin Equations

    [https://arxiv.org/abs/2402.11705](https://arxiv.org/abs/2402.11705)

    提出一种学习广义朗之万方程中记忆核的新方法，通过正则化Prony方法估计相关函数并在Sobolev范数Loss函数和RKHS正则化下实现回归，在指数加权的$L^2$空间内获得改进性能，对比其他回归估计器展示了其优越性。

    

    我们引入了一种新颖的方法来学习广义朗之万方程中的记忆核。该方法最初利用正则化Prony方法从轨迹数据中估计相关函数，然后通过基于Sobolev范数的回归和RKHS正则化来进行回归。我们的方法保证在指数加权的$L^2$空间内获得了改进的性能，核估计误差受控于估计相关函数的误差。我们通过数值示例展示了我们的估计器相对于依赖于$L^2$损失函数的其他回归估计器以及从逆拉普拉斯变换推导出的估计器的优越性，这些示例突显了我们的估计器在各种权重参数选择上的持续优势。此外，我们提供了包括力和漂移项在方程中的应用示例。

    arXiv:2402.11705v1 Announce Type: cross  Abstract: We introduce a novel approach for learning memory kernels in Generalized Langevin Equations. This approach initially utilizes a regularized Prony method to estimate correlation functions from trajectory data, followed by regression over a Sobolev norm-based loss function with RKHS regularization. Our approach guarantees improved performance within an exponentially weighted $L^2$ space, with the kernel estimation error controlled by the error in estimated correlation functions. We demonstrate the superiority of our estimator compared to other regression estimators that rely on $L^2$ loss functions and also an estimator derived from the inverse Laplace transform, using numerical examples that highlight its consistent advantage across various weight parameter selections. Additionally, we provide examples that include the application of force and drift terms in the equation.
    

