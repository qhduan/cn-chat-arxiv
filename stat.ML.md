# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |
| [^2] | [Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm.](http://arxiv.org/abs/2303.07287) | 本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。 |
| [^3] | [Data-Driven Influence Functions for Optimization-Based Causal Inference.](http://arxiv.org/abs/2208.13701) | 本文提出了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，并研究了从数据中进行概率分布估计的情况下的Gateaux导数估计。研究结果为因果推断和动态治疗方案等问题提供了解决方案。 |

# 详细

[^1]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    
[^2]: 通过子高斯内在矩范实现紧凑的非渐进推断

    Tight Non-asymptotic Inference via Sub-Gaussian Intrinsic Moment Norm. (arXiv:2303.07287v1 [stat.ML])

    [http://arxiv.org/abs/2303.07287](http://arxiv.org/abs/2303.07287)

    本文提出了一种通过最大化一系列归一化矩来使用子高斯内在矩范实现紧凑的非渐进推断的方法，该方法可以导致更紧的Hoeffding子高斯浓度不等式，并且可以通过子高斯图检查具有有限样本大小的子高斯数据。

    This paper proposes a method of achieving tight non-asymptotic inference by using sub-Gaussian intrinsic moment norm through maximizing a series of normalized moments, which can lead to tighter Hoeffding's sub-Gaussian concentration inequalities and can be checked with sub-Gaussian plot for sub-Gaussian data with a finite sample size.

    在非渐进统计推断中，子高斯分布的方差类型参数起着至关重要的作用。然而，基于经验矩生成函数（MGF）的直接估计这些参数是不可行的。为此，我们建议通过最大化一系列归一化矩来使用子高斯内在矩范[Buldygin和Kozachenko（2000），定理1.3]。重要的是，推荐的范数不仅可以恢复相应MGF的指数矩界限，而且还可以导致更紧的Hoeffding子高斯浓度不等式。在实践中，我们提出了一种直观的方法，通过子高斯图检查具有有限样本大小的子高斯数据。可以通过简单的插入方法鲁棒地估计内在矩范数。我们的理论结果应用于非渐进分析，包括多臂赌博机。

    In non-asymptotic statistical inferences, variance-type parameters of sub-Gaussian distributions play a crucial role. However, direct estimation of these parameters based on the empirical moment generating function (MGF) is infeasible. To this end, we recommend using a sub-Gaussian intrinsic moment norm [Buldygin and Kozachenko (2000), Theorem 1.3] through maximizing a series of normalized moments. Importantly, the recommended norm can not only recover the exponential moment bounds for the corresponding MGFs, but also lead to tighter Hoeffding's sub-Gaussian concentration inequalities. In practice, {\color{black} we propose an intuitive way of checking sub-Gaussian data with a finite sample size by the sub-Gaussian plot}. Intrinsic moment norm can be robustly estimated via a simple plug-in approach. Our theoretical results are applied to non-asymptotic analysis, including the multi-armed bandit.
    
[^3]: 基于数据驱动的最优化因果推断影响函数

    Data-Driven Influence Functions for Optimization-Based Causal Inference. (arXiv:2208.13701v4 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.13701](http://arxiv.org/abs/2208.13701)

    本文提出了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，并研究了从数据中进行概率分布估计的情况下的Gateaux导数估计。研究结果为因果推断和动态治疗方案等问题提供了解决方案。

    

    本研究探讨了一种利用有限差分逼近统计泛函Gateaux导数的构造算法，重点研究了在因果推断中出现的泛函。我们研究了概率分布未知但需要从数据中进行估计的情况。这些估计分布引导了经验Gateaux导数，我们研究了经验、数值和解析Gateaux导数之间的关系。从干预均值（平均潜在结果）的案例入手，我们勾勒了有限差分和解析Gateaux导数之间的关系。然后，我们得出了关于扰动和平滑的数值逼近速率要求，以保持单步调整的统计优势，例如速率双重强健性。接下来，我们研究了更复杂的泛函，如动态治疗方案、无限时段Markov决策中策略优化的线性规划形式。

    We study a constructive algorithm that approximates Gateaux derivatives for statistical functionals by finite differencing, with a focus on functionals that arise in  causal inference. We study the case where probability distributions are not known a priori but need to be estimated from data. These estimated distributions lead to empirical Gateaux derivatives, and we study the relationships between empirical, numerical, and analytical Gateaux derivatives. Starting with a case study of the interventional mean (average potential outcome), we delineate the relationship between finite differences and the analytical Gateaux derivative. We then derive requirements on the rates of numerical approximation in perturbation and smoothing that preserve the statistical benefits of one-step adjustments, such as rate double robustness. We then study more complicated functionals such as dynamic treatment regimes, the linear-programming formulation for policy optimization in infinite-horizon Markov dec
    

