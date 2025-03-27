# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Learning Framework with Uncertainty Quantification for Survey Data: Assessing and Predicting Diabetes Mellitus Risk in the American Population](https://arxiv.org/abs/2403.19752) | 该论文提出了一个利用神经网络模型进行回归和分类的预测框架，并引入了适用于复杂调查设计数据的不确定性量化算法，以评估美国人群糖尿病风险。 |
| [^2] | [Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors.](http://arxiv.org/abs/2401.02739) | 本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。 |
| [^3] | [Sinkhorn Distributionally Robust Optimization.](http://arxiv.org/abs/2109.11926) | 本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。 |

# 详细

[^1]: 具有不确定性量化的调查数据深度学习框架: 评估和预测美国人群糖尿病风险

    Deep Learning Framework with Uncertainty Quantification for Survey Data: Assessing and Predicting Diabetes Mellitus Risk in the American Population

    [https://arxiv.org/abs/2403.19752](https://arxiv.org/abs/2403.19752)

    该论文提出了一个利用神经网络模型进行回归和分类的预测框架，并引入了适用于复杂调查设计数据的不确定性量化算法，以评估美国人群糖尿病风险。

    

    多种医学队列中通常采用复杂的调查设计。在这种情况下，开发反映研究设计的独特特征的特定病例预测风险评分模型至关重要。本文的目标是:(i) 提出一个通用的预测框架，利用神经网络(NN)建模进行回归和分类，其将调查权重纳入估计过程中;(ii) 引入一种模型预测的不确定性量化算法，专为来自复杂调查设计的数据量身定制;(iii) 应用这种方法开发健壮的风险评分模型，评估美国人群糖尿病风险，利用NHANES 2011-2014队列中的数据。我们的估计器的理论性质旨在确保最小偏差和统计一致性，从而确保我们的

    arXiv:2403.19752v1 Announce Type: cross  Abstract: Complex survey designs are commonly employed in many medical cohorts. In such scenarios, developing case-specific predictive risk score models that reflect the unique characteristics of the study design is essential. This approach is key to minimizing potential selective biases in results. The objectives of this paper are: (i) To propose a general predictive framework for regression and classification using neural network (NN) modeling, which incorporates survey weights into the estimation process; (ii) To introduce an uncertainty quantification algorithm for model prediction, tailored for data from complex survey designs; (iii) To apply this method in developing robust risk score models to assess the risk of Diabetes Mellitus in the US population, utilizing data from the NHANES 2011-2014 cohort. The theoretical properties of our estimators are designed to ensure minimal bias and the statistical consistency, thereby ensuring that our m
    
[^2]: 扩散变分推断：扩散模型作为表达性变分后验

    Diffusion Variational Inference: Diffusion Models as Expressive Variational Posteriors. (arXiv:2401.02739v1 [cs.LG])

    [http://arxiv.org/abs/2401.02739](http://arxiv.org/abs/2401.02739)

    本文提出了去噪扩散变分推断（DDVI）算法，该算法使用扩散模型作为表达性变分后验，并通过反转加噪过程在潜空间中进行扩散。该方法易于实现，兼容黑盒变分推断，并在深度潜变量模型中的任务中表现优异。

    

    我们提出了去噪扩散变分推断（DDVI），一种用扩散模型作为表达性变分后验的潜变量模型的近似推断算法。我们的方法通过辅助潜变量增加了变分后验，从而得到一个表达性的模型类，通过反转用户指定的加噪过程在潜空间中进行扩散。我们通过优化一个受到觉醒-睡眠算法启发的边际似然新下界来拟合这些模型。我们的方法易于实现（它适配了正则化的ELBO扩展），与黑盒变分推断兼容，并且表现优于基于归一化流或对抗网络的替代近似后验类别。将我们的方法应用于深度潜变量模型时，我们的方法得到了去噪扩散变分自动编码器（DD-VAE）算法。我们将该算法应用于生物学中的一个激励任务 -- 从人类基因组中推断潜在血统 -- 超过了强基线模型。

    We propose denoising diffusion variational inference (DDVI), an approximate inference algorithm for latent variable models which relies on diffusion models as expressive variational posteriors. Our method augments variational posteriors with auxiliary latents, which yields an expressive class of models that perform diffusion in latent space by reversing a user-specified noising process. We fit these models by optimizing a novel lower bound on the marginal likelihood inspired by the wake-sleep algorithm. Our method is easy to implement (it fits a regularized extension of the ELBO), is compatible with black-box variational inference, and outperforms alternative classes of approximate posteriors based on normalizing flows or adversarial networks. When applied to deep latent variable models, our method yields the denoising diffusion VAE (DD-VAE) algorithm. We use this algorithm on a motivating task in biology -- inferring latent ancestry from human genomes -- outperforming strong baselines
    
[^3]: Sinkhorn分布鲁棒优化

    Sinkhorn Distributionally Robust Optimization. (arXiv:2109.11926v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2109.11926](http://arxiv.org/abs/2109.11926)

    本文通过使用Sinkhorn距离进行分布鲁棒优化，推导出更容易处理且在实际中更合理的最坏情况分布，提出了解决方案，并展示了其优越性能。

    

    我们研究了使用Sinkhorn距离 -一种基于熵正则化的Wasserstein距离变体- 的分布鲁棒优化（DRO）。我们为一般名义分布推导了凸规划对偶重构。相比于Wasserstein DRO，对于更大类的损失函数，它在计算上更容易处理，它的最坏情况分布对实际应用更合理。为了解决对偶重构，我们开发了一种使用有偏梯度神经元的随机镜像下降算法，并分析了其收敛速度。最后，我们提供了使用合成和真实数据的数值实例，以证明其优越性能。

    We study distributionally robust optimization (DRO) with Sinkhorn distance -a variant of Wasserstein distance based on entropic regularization. We derive convex programming dual reformulation for a general nominal distribution. Compared with Wasserstein DRO, it is computationally tractable for a larger class of loss functions, and its worst-case distribution is more reasonable for practical applications. To solve the dual reformulation, we develop a stochastic mirror descent algorithm using biased gradient oracles and analyze its convergence rate. Finally, we provide numerical examples using synthetic and real data to demonstrate its superior performance.
    

