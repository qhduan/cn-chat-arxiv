# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nested Nonparametric Instrumental Variable Regression: Long Term, Mediated, and Time Varying Treatment Effects](https://arxiv.org/abs/2112.14249) | 该论文提出了嵌套非参数工具变量回归的对抗估计器，并提供了对因果参数进行有效推断的充分条件，具有限制病态性复合技术、多种适应模型和扩展到因果函数等特征。 |
| [^2] | [Information Leakage Detection through Approximate Bayes-optimal Prediction.](http://arxiv.org/abs/2401.14283) | 本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。 |
| [^3] | [Noise-adaptive (Accelerated) Stochastic Heavy-Ball Momentum.](http://arxiv.org/abs/2401.06738) | 本研究分析了在光滑、强凸环境中随机重力球动量的收敛性，并证明了当批量大小大于某个阈值时，该方法可以实现加速收敛速度。针对强凸二次函数，我们建议了一种噪声自适应的多阶段方法，可以使收敛速度进一步提高。实验结果验证了该方法的有效性。 |
| [^4] | [Federated Learning with Differential Privacy for End-to-End Speech Recognition.](http://arxiv.org/abs/2310.00098) | 本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。 |
| [^5] | [Nonparametric Additive Value Functions: Interpretable Reinforcement Learning with an Application to Surgical Recovery.](http://arxiv.org/abs/2308.13135) | 该论文提出了一种非参数可加模型，用于估计可解释的值函数，并在强化学习中具有应用价值。该方法能够克服传统模型的线性假设限制，同时提供较强的决策建议解释性。 |

# 详细

[^1]: 嵌套非参数工具变量回归：长期、中介和时变治疗效应

    Nested Nonparametric Instrumental Variable Regression: Long Term, Mediated, and Time Varying Treatment Effects

    [https://arxiv.org/abs/2112.14249](https://arxiv.org/abs/2112.14249)

    该论文提出了嵌套非参数工具变量回归的对抗估计器，并提供了对因果参数进行有效推断的充分条件，具有限制病态性复合技术、多种适应模型和扩展到因果函数等特征。

    

    短面板数据模型中的几个因果参数是称为嵌套非参数工具变量回归（nested NPIV）的函数的标量总结。例如，使用代理变量识别出长期、中介和时变治疗效应。然而，似乎不存在关于嵌套NPIV的先前估计量或保证，这样就无法灵活地估计和推断这些因果参数。一个主要挑战是由于嵌套逆问题而导致的复合病态性。我们分析了嵌套NPIV的对抗估计器，并提供了对因果参数进行有效推断的充分条件。我们的非渐近分析具有三个显著特征：（i）引入限制病态性复合的技术；（ii）适应神经网络、随机森林和再生核希尔伯特空间；（iii）扩展到因果函数，例如长期异质治疗效果。

    arXiv:2112.14249v3 Announce Type: replace-cross  Abstract: Several causal parameters in short panel data models are scalar summaries of a function called a nested nonparametric instrumental variable regression (nested NPIV). Examples include long term, mediated, and time varying treatment effects identified using proxy variables. However, it appears that no prior estimators or guarantees for nested NPIV exist, preventing flexible estimation and inference for these causal parameters. A major challenge is compounding ill posedness due to the nested inverse problems. We analyze adversarial estimators of nested NPIV, and provide sufficient conditions for efficient inference on the causal parameter. Our nonasymptotic analysis has three salient features: (i) introducing techniques that limit how ill posedness compounds; (ii) accommodating neural networks, random forests, and reproducing kernel Hilbert spaces; and (iii) extending to causal functions, e.g. long term heterogeneous treatment eff
    
[^2]: 通过近似贝叶斯最优预测检测信息泄漏

    Information Leakage Detection through Approximate Bayes-optimal Prediction. (arXiv:2401.14283v1 [stat.ML])

    [http://arxiv.org/abs/2401.14283](http://arxiv.org/abs/2401.14283)

    本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。

    

    在今天的以数据驱动的世界中，公开可获得的信息的增加加剧了信息泄漏（IL）的挑战，引发了安全问题。IL涉及通过系统的可观察信息无意地将秘密（敏感）信息暴露给未经授权的方，传统的统计方法通过估计可观察信息和秘密信息之间的互信息（MI）来检测IL，面临维度灾难、收敛、计算复杂度和MI估计错误等挑战。此外，虽然新兴的监督机器学习（ML）方法在二进制系统敏感信息的检测上有效，但缺乏一个全面的理论框架。为了解决这些限制，我们使用统计学习理论和信息论建立了一个理论框架来准确量化和检测IL。我们证明了可以通过近似贝叶斯预测的对数损失和准确性来准确估计MI。

    In today's data-driven world, the proliferation of publicly available information intensifies the challenge of information leakage (IL), raising security concerns. IL involves unintentionally exposing secret (sensitive) information to unauthorized parties via systems' observable information. Conventional statistical approaches, which estimate mutual information (MI) between observable and secret information for detecting IL, face challenges such as the curse of dimensionality, convergence, computational complexity, and MI misestimation. Furthermore, emerging supervised machine learning (ML) methods, though effective, are limited to binary system-sensitive information and lack a comprehensive theoretical framework. To address these limitations, we establish a theoretical framework using statistical learning theory and information theory to accurately quantify and detect IL. We demonstrate that MI can be accurately estimated by approximating the log-loss and accuracy of the Bayes predict
    
[^3]: 噪声自适应（加速）随机重力球动量

    Noise-adaptive (Accelerated) Stochastic Heavy-Ball Momentum. (arXiv:2401.06738v1 [math.OC])

    [http://arxiv.org/abs/2401.06738](http://arxiv.org/abs/2401.06738)

    本研究分析了在光滑、强凸环境中随机重力球动量的收敛性，并证明了当批量大小大于某个阈值时，该方法可以实现加速收敛速度。针对强凸二次函数，我们建议了一种噪声自适应的多阶段方法，可以使收敛速度进一步提高。实验结果验证了该方法的有效性。

    

    我们分析了在光滑，强凸环境中随机重力球动量（SHB）的收敛性。Kidambi等人（2018）表明，对于二次函数，SHB（带有小批量）无法达到加速的收敛速度，并猜想SHB的实际收益是小批量的副产品。我们通过展示当批量大小大于一定阈值时，SHB可以获得加速的收敛速度来证实这一观点。特别地，对于条件数为$\kappa$的强凸二次函数，我们证明了使用标准步长和动量参数的SHB具有$O\left(\exp(-\frac{T}{\sqrt{\kappa}}) + \sigma \right)$的收敛速度，其中$T$为迭代次数，$\sigma^2$为随机梯度的方差。为确保收敛到极小值，我们提出了一种多阶段方法，结果是噪声自适应的$O\left(\exp\left(-\frac{T}{\sqrt{\kappa}} \right) + \frac{\sigma}{T}\right)$速度。对于一般的强凸函数，我们在实验中展示了所提方法的有效性。

    We analyze the convergence of stochastic heavy ball (SHB) momentum in the smooth, strongly-convex setting. Kidambi et al. (2018) show that SHB (with small mini-batches) cannot attain an accelerated rate of convergence even for quadratics, and conjecture that the practical gain of SHB is a by-product of mini-batching. We substantiate this claim by showing that SHB can obtain an accelerated rate when the mini-batch size is larger than some threshold. In particular, for strongly-convex quadratics with condition number $\kappa$, we prove that SHB with the standard step-size and momentum parameters results in an $O\left(\exp(-\frac{T}{\sqrt{\kappa}}) + \sigma \right)$ convergence rate, where $T$ is the number of iterations and $\sigma^2$ is the variance in the stochastic gradients. To ensure convergence to the minimizer, we propose a multi-stage approach that results in a noise-adaptive $O\left(\exp\left(-\frac{T}{\sqrt{\kappa}} \right) + \frac{\sigma}{T}\right)$ rate. For general strongly-
    
[^4]: 使用差分隐私的联邦学习进行端到端语音识别

    Federated Learning with Differential Privacy for End-to-End Speech Recognition. (arXiv:2310.00098v1 [cs.LG])

    [http://arxiv.org/abs/2310.00098](http://arxiv.org/abs/2310.00098)

    本文提出了一种基于联邦学习和差分隐私的端到端语音识别方法，探索了大型Transformer模型的不同方面，并建立了基线结果。

    

    联邦学习是一种有前景的训练机器学习模型的方法，但在自动语音识别领域仅限于初步探索。此外，联邦学习不能本质上保证用户隐私，并需要差分隐私来提供稳健的隐私保证。然而，我们还不清楚在自动语音识别中应用差分隐私的先前工作。本文旨在通过为联邦学习提供差分隐私的自动语音识别基准，并建立第一个基线来填补这一研究空白。我们扩展了现有的联邦学习自动语音识别研究，探索了最新的大型端到端Transformer模型的不同方面：架构设计，种子模型，数据异质性，领域转移，以及cohort大小的影响。通过合理的中央聚合数量，我们能够训练出即使在异构数据、来自另一个领域的种子模型或无预先训练的情况下仍然接近最优的联邦学习模型。

    While federated learning (FL) has recently emerged as a promising approach to train machine learning models, it is limited to only preliminary explorations in the domain of automatic speech recognition (ASR). Moreover, FL does not inherently guarantee user privacy and requires the use of differential privacy (DP) for robust privacy guarantees. However, we are not aware of prior work on applying DP to FL for ASR. In this paper, we aim to bridge this research gap by formulating an ASR benchmark for FL with DP and establishing the first baselines. First, we extend the existing research on FL for ASR by exploring different aspects of recent $\textit{large end-to-end transformer models}$: architecture design, seed models, data heterogeneity, domain shift, and impact of cohort size. With a $\textit{practical}$ number of central aggregations we are able to train $\textbf{FL models}$ that are \textbf{nearly optimal} even with heterogeneous data, a seed model from another domain, or no pre-trai
    
[^5]: 非参数可加值函数：具有可解释性的强化学习方法及其在外科手术恢复中的应用

    Nonparametric Additive Value Functions: Interpretable Reinforcement Learning with an Application to Surgical Recovery. (arXiv:2308.13135v1 [stat.ML])

    [http://arxiv.org/abs/2308.13135](http://arxiv.org/abs/2308.13135)

    该论文提出了一种非参数可加模型，用于估计可解释的值函数，并在强化学习中具有应用价值。该方法能够克服传统模型的线性假设限制，同时提供较强的决策建议解释性。

    

    我们提出了一种非参数可加模型，用于在强化学习中估计可解释的值函数。学习依靠数字表型特征的有效自适应临床干预是医务人员重视的问题。在脊柱手术方面，关于患者运动能力恢复的不同术后恢复建议可能会导致患者恢复程度的显著变化。虽然强化学习在游戏等领域取得了广泛成功，但最近的方法严重依赖于黑盒方法，如神经网络。不幸的是，这些方法阻碍了考察每个特征对于产生最终建议决策的贡献。虽然在经典算法（如最小二乘策略迭代）中可以轻松提供这样的解释，但基本的线性假设阻止了学习特征之间的高阶灵活交互作用。在本文中，我们提出了一种新颖的方法，提供了一种灵活的技术来克服这些限制，并能够得到解释性强的决策建议模型。

    We propose a nonparametric additive model for estimating interpretable value functions in reinforcement learning. Learning effective adaptive clinical interventions that rely on digital phenotyping features is a major for concern medical practitioners. With respect to spine surgery, different post-operative recovery recommendations concerning patient mobilization can lead to significant variation in patient recovery. While reinforcement learning has achieved widespread success in domains such as games, recent methods heavily rely on black-box methods, such neural networks. Unfortunately, these methods hinder the ability of examining the contribution each feature makes in producing the final suggested decision. While such interpretations are easily provided in classical algorithms such as Least Squares Policy Iteration, basic linearity assumptions prevent learning higher-order flexible interactions between features. In this paper, we present a novel method that offers a flexible techniq
    

