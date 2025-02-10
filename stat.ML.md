# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ADAPT to Robustify Prompt Tuning Vision Transformers](https://arxiv.org/abs/2403.13196) | 本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。 |
| [^2] | [Consistent model selection in the spiked Wigner model via AIC-type criteria.](http://arxiv.org/abs/2307.12982) | 该论文介绍了在带尖峰的Wigner模型中使用AIC类型准则进行一致性模型选择。研究发现，对于$\gamma > 2$，该准则是强一致估计的，而对于$\gamma < 2$，它几乎肯定会高估尖峰数量$k$。此外，作者还提出了一个使AIC弱一致估计的方法，并证明了某个软最小化器是强一致估计的。 |

# 详细

[^1]: 使Prompt调优视觉Transformer更为健壮的ADAPT

    ADAPT to Robustify Prompt Tuning Vision Transformers

    [https://arxiv.org/abs/2403.13196](https://arxiv.org/abs/2403.13196)

    本文提出了ADAPT框架，用于在prompt调优范式中进行自适应对抗训练，增强视觉Transformer在下游任务中的稳健性。

    

    深度模型的性能，包括视觉Transformer，已知容易受到对抗性攻击的影响。许多现有对抗性防御方法，如对抗性训练，依赖于对整个模型进行全面微调以增加模型的稳健性。这些防御方法需要为每个任务存储整个模型的副本，而模型可能包含数十亿个参数。与此同时，参数高效的prompt调优被用来适应大型基于Transformer的模型到下游任务，无需保存大型副本。本文从稳健性的角度研究了对视觉Transformer进行下游任务的参数高效prompt调优。我们发现，之前的对抗性防御方法在应用到prompt调优范式时，存在梯度模糊并容易受到自适应攻击的影响。我们引入了ADAPT，一种在prompt调优范式中执行自适应对抗训练的新框架。

    arXiv:2403.13196v1 Announce Type: new  Abstract: The performance of deep models, including Vision Transformers, is known to be vulnerable to adversarial attacks. Many existing defenses against these attacks, such as adversarial training, rely on full-model fine-tuning to induce robustness in the models. These defenses require storing a copy of the entire model, that can have billions of parameters, for each task. At the same time, parameter-efficient prompt tuning is used to adapt large transformer-based models to downstream tasks without the need to save large copies. In this paper, we examine parameter-efficient prompt tuning of Vision Transformers for downstream tasks under the lens of robustness. We show that previous adversarial defense methods, when applied to the prompt tuning paradigm, suffer from gradient obfuscation and are vulnerable to adaptive attacks. We introduce ADAPT, a novel framework for performing adaptive adversarial training in the prompt tuning paradigm. Our meth
    
[^2]: 在带尖峰的Wigner模型中通过AIC类型准则进行一致性模型选择

    Consistent model selection in the spiked Wigner model via AIC-type criteria. (arXiv:2307.12982v1 [math.ST])

    [http://arxiv.org/abs/2307.12982](http://arxiv.org/abs/2307.12982)

    该论文介绍了在带尖峰的Wigner模型中使用AIC类型准则进行一致性模型选择。研究发现，对于$\gamma > 2$，该准则是强一致估计的，而对于$\gamma < 2$，它几乎肯定会高估尖峰数量$k$。此外，作者还提出了一个使AIC弱一致估计的方法，并证明了某个软最小化器是强一致估计的。

    

    考虑带尖峰的Wigner模型\[ X = \sum_{i = 1}^k \lambda_i u_i u_i^\top + \sigma G, \]其中$G$是一个$N \times N$的GOE随机矩阵，而特征值$\lambda_i$都是有尖峰的，即超过了Baik-Ben Arous-P\'ech\'e (BBP)的阈值$\sigma$。我们考虑形式为\[ -2 \, (\text{最大化的对数似然}) + \gamma \, (\text{参数数量}) \]的AIC类型模型选择准则，用于估计尖峰数量$k$。对于$\gamma > 2$，上述准则是强一致估计的，前提是$\lambda_k > \lambda_{\gamma}$，其中$\lambda_{\gamma}$是严格高于BBP阈值的阈值，而对于$\gamma < 2$，它几乎肯定会高估$k$。虽然AIC（对应于$\gamma = 2$）并非强一致估计，但我们证明，取$\gamma = 2 + \delta_N$，其中$\delta_N \to 0$且$\delta_N \gg N^{-2/3}$，会得到$k$的弱一致估计量。我们还证明了AIC的某个软最小化器是强一致估计的。

    Consider the spiked Wigner model \[ X = \sum_{i = 1}^k \lambda_i u_i u_i^\top + \sigma G, \] where $G$ is an $N \times N$ GOE random matrix, and the eigenvalues $\lambda_i$ are all spiked, i.e. above the Baik-Ben Arous-P\'ech\'e (BBP) threshold $\sigma$. We consider AIC-type model selection criteria of the form \[ -2 \, (\text{maximised log-likelihood}) + \gamma \, (\text{number of parameters}) \] for estimating the number $k$ of spikes. For $\gamma > 2$, the above criterion is strongly consistent provided $\lambda_k > \lambda_{\gamma}$, where $\lambda_{\gamma}$ is a threshold strictly above the BBP threshold, whereas for $\gamma < 2$, it almost surely overestimates $k$. Although AIC (which corresponds to $\gamma = 2$) is not strongly consistent, we show that taking $\gamma = 2 + \delta_N$, where $\delta_N \to 0$ and $\delta_N \gg N^{-2/3}$, results in a weakly consistent estimator of $k$. We also show that a certain soft minimiser of AIC is strongly consistent.
    

