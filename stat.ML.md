# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Coreset-based, Tempered Variational Posterior for Accurate and Scalable Stochastic Gaussian Process Inference.](http://arxiv.org/abs/2311.01409) | 这篇论文提出了一种基于核心集的、温和变分后验的高斯过程推理方法，通过利用稀疏的、可解释的数据表示来降低参数大小，并且具有数值稳定性和较低的时间和空间复杂度。 |

# 详细

[^1]: 基于核心集的、温和变分后验的精确和可扩展随机高斯过程推理方法

    A Coreset-based, Tempered Variational Posterior for Accurate and Scalable Stochastic Gaussian Process Inference. (arXiv:2311.01409v1 [cs.LG])

    [http://arxiv.org/abs/2311.01409](http://arxiv.org/abs/2311.01409)

    这篇论文提出了一种基于核心集的、温和变分后验的高斯过程推理方法，通过利用稀疏的、可解释的数据表示来降低参数大小，并且具有数值稳定性和较低的时间和空间复杂度。

    

    我们提出了一种新颖的随机变分高斯过程($\mathcal{GP}$)推理方法，该方法基于可学习的权重伪输入输出点的后验（核心集）。与自由形式的变分族不同，提出的基于核心集的、温和变分的$\mathcal{GP}$（CVTGP）是基于$\mathcal{GP}$先验和数据似然函数来定义的，因此适应了建模的归纳偏差。我们通过对提出的后验进行潜在的$\mathcal{GP}$核心集变量的边缘化，推导出CVTGP的对数边际似然下界，并且证明其适用于随机优化。CVTGP通过利用基于核心集的温和后验来减小可学习参数的大小到$\mathcal{O}(M)$，具有数值稳定性，并且通过提供稀疏且可解释的数据表示来保持$\mathcal{O}(M^3)$时间复杂度和$\mathcal{O}(M^2)$空间复杂度。在模拟和真实回归问题上的实验结果显示了CVTGP的性能优势。

    We present a novel stochastic variational Gaussian process ($\mathcal{GP}$) inference method, based on a posterior over a learnable set of weighted pseudo input-output points (coresets). Instead of a free-form variational family, the proposed coreset-based, variational tempered family for $\mathcal{GP}$s (CVTGP) is defined in terms of the $\mathcal{GP}$ prior and the data-likelihood; hence, accommodating the modeling inductive biases. We derive CVTGP's lower bound for the log-marginal likelihood via marginalization of the proposed posterior over latent $\mathcal{GP}$ coreset variables, and show it is amenable to stochastic optimization. CVTGP reduces the learnable parameter size to $\mathcal{O}(M)$, enjoys numerical stability, and maintains $\mathcal{O}(M^3)$ time- and $\mathcal{O}(M^2)$ space-complexity, by leveraging a coreset-based tempered posterior that, in turn, provides sparse and explainable representations of the data. Results on simulated and real-world regression problems wi
    

