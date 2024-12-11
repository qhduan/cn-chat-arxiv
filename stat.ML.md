# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference for Regression with Variables Generated from Unstructured Data](https://arxiv.org/abs/2402.15585) | 提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。 |
| [^2] | [What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement](https://arxiv.org/abs/2402.01865) | 本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。 |
| [^3] | [Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures.](http://arxiv.org/abs/2311.03242) | 本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。 |
| [^4] | [Mathematical analysis of singularities in the diffusion model under the submanifold assumption.](http://arxiv.org/abs/2301.07882) | 本文提供了扩散模型中漂移项的数学分析。通过次流形假设，提出一种新的目标函数和相关的损失函数，可处理低维流形上的奇异数据分布，解决了均值漂移函数和得分函数渐近发散的问题。 |

# 详细

[^1]: 使用来自非结构化数据生成的变量进行回归的推断

    Inference for Regression with Variables Generated from Unstructured Data

    [https://arxiv.org/abs/2402.15585](https://arxiv.org/abs/2402.15585)

    提出了一种使用联合上游和下游模型进行有效推断的一步策略，显著减少了偏误，在CEO时间利用数据的应用中产生了重要效果，适合应用研究人员。

    

    分析非结构化数据的主要策略包括两个步骤。首先，使用上游信息检索模型估计感兴趣的潜在经济变量。其次，将估计值视为下游计量经济模型中的“数据”。我们建立了理论论点，解释为什么在实证合理的设置中，这种两步策略会导致偏误的推断。更具建设性的是，我们提出了一个有效推断的一步策略，该策略同时使用上游和下游模型。在模拟中，这一步策略(i) 显著减少了偏误；(ii) 在使用CEO时间利用数据的主要应用中产生了定量重要的效果；(iii) 可以很容易地被应用研究人员采用。

    arXiv:2402.15585v1 Announce Type: new  Abstract: The leading strategy for analyzing unstructured data uses two steps. First, latent variables of economic interest are estimated with an upstream information retrieval model. Second, the estimates are treated as "data" in a downstream econometric model. We establish theoretical arguments for why this two-step strategy leads to biased inference in empirically plausible settings. More constructively, we propose a one-step strategy for valid inference that uses the upstream and downstream models jointly. The one-step strategy (i) substantially reduces bias in simulations; (ii) has quantitatively important effects in a leading application using CEO time-use data; and (iii) can be readily adapted by applied researchers.
    
[^2]: 我的模型会忘记什么？语言模型改进中的被遗忘实例预测

    What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement

    [https://arxiv.org/abs/2402.01865](https://arxiv.org/abs/2402.01865)

    本文研究了语言模型更新中的遗忘现象，提出了一种预测上游实例遗忘的方法，以改进重播过程的可控性和解释性。根据预训练实例的预-softmax对数几率分数变化与在线学习实例的相似性，提出了一种部分可解释的预测模型，在BART模型上表现良好但在T5模型上失败。此外，还展示了基于内积的黑盒分类器。

    

    在实际应用中，语言模型会出现错误。然而，仅仅通过将模型更新为纠正错误实例，会导致灾难性的遗忘，更新后的模型在指导微调或上游训练阶段中学到的实例上出现错误。随机重播上游数据的效果不令人满意，往往伴随着较高的方差和较差的可控性。为了改善重播过程的可控性和解释性，我们试图预测由于模型更新而遗忘的上游实例。我们根据一组在线学习的实例和相应被遗忘的上游预训练实例训练预测模型。我们提出了一种部分可解释的预测模型，该模型基于这样的观察结果：预训练实例的预-softmax对数几率分数的变化类似于在线学习实例的变化，这在BART模型上表现出不错的效果，但在T5模型上失败。我们进一步展示了基于内积的黑盒分类器

    Language models deployed in the wild make errors. However, simply updating the model with the corrected error instances causes catastrophic forgetting -- the updated model makes errors on instances learned during the instruction tuning or upstream training phase. Randomly replaying upstream data yields unsatisfactory performance and often comes with high variance and poor controllability. To this end, we try to forecast upstream examples that will be forgotten due to a model update for improved controllability of the replay process and interpretability. We train forecasting models given a collection of online learned examples and corresponding forgotten upstream pre-training examples. We propose a partially interpretable forecasting model based on the observation that changes in pre-softmax logit scores of pretraining examples resemble that of online learned examples, which performs decently on BART but fails on T5 models. We further show a black-box classifier based on inner products 
    
[^3]: 使用类似ResNet的神经网络架构近似Langevin Monte Carlo

    Approximating Langevin Monte Carlo with ResNet-like Neural Network architectures. (arXiv:2311.03242v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2311.03242](http://arxiv.org/abs/2311.03242)

    本论文提出了一种使用类似ResNet的神经网络架构来近似Langevin Monte Carlo算法，通过将来自简单参考分布的样本映射到目标分布的样本中来进行采样，具有较好的逼近速度和表达性。

    

    我们通过构建一个神经网络，将来自简单参考分布（如标准正态分布）的样本映射到目标分布的样本中，从而从给定的目标分布中进行采样。为此，我们提出使用受Langevin Monte Carlo (LMC)算法启发的神经网络架构。基于LMC扰动结果，在Wasserstein-2距离上，我们展示了该架构对于平滑的对数凹目标分布的逼近速度。分析严重依赖于扰动LMC过程的中间度量的亚高斯性概念。特别地，我们根据不同扰动假设推导出了中间方差代理的增长界限。此外，我们提出了一种类似于深度残差神经网络的架构，并推导出了近似样本与目标分布映射的表达性结果。

    We sample from a given target distribution by constructing a neural network which maps samples from a simple reference, e.g. the standard normal distribution, to samples from the target. To that end, we propose using a neural network architecture inspired by the Langevin Monte Carlo (LMC) algorithm. Based on LMC perturbation results, we show approximation rates of the proposed architecture for smooth, log-concave target distributions measured in the Wasserstein-$2$ distance. The analysis heavily relies on the notion of sub-Gaussianity of the intermediate measures of the perturbed LMC process. In particular, we derive bounds on the growth of the intermediate variance proxies under different assumptions on the perturbations. Moreover, we propose an architecture similar to deep residual neural networks and derive expressivity results for approximating the sample to target distribution map.
    
[^4]: 基于次流形假设下扩散模型奇异性的数学分析

    Mathematical analysis of singularities in the diffusion model under the submanifold assumption. (arXiv:2301.07882v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.07882](http://arxiv.org/abs/2301.07882)

    本文提供了扩散模型中漂移项的数学分析。通过次流形假设，提出一种新的目标函数和相关的损失函数，可处理低维流形上的奇异数据分布，解决了均值漂移函数和得分函数渐近发散的问题。

    

    本文提供了机器学习中扩散模型的数学分析。以条件期望表示反向采样流程的漂移项，其中涉及数据分布和前向扩散。训练过程旨在通过最小化与条件期望相关的均方残差来寻找此类漂移函数。使用前向扩散的Green函数的小时间近似，我们证明了DDPM中的解析均值漂移函数和SGM中的得分函数在采样过程的最后阶段，对于像那些集中在低维流形上的奇异数据分布而言，渐近地发散，因此难以通过网络进行逼近。为了克服这个困难，我们推导出了一个新的目标函数和相关的损失函数，即使在处理奇异数据分布时仍然保持有界。我们通过几个数值实验来说明理论发现。

    This paper provide several mathematical analyses of the diffusion model in machine learning. The drift term of the backwards sampling process is represented as a conditional expectation involving the data distribution and the forward diffusion. The training process aims to find such a drift function by minimizing the mean-squared residue related to the conditional expectation. Using small-time approximations of the Green's function of the forward diffusion, we show that the analytical mean drift function in DDPM and the score function in SGM asymptotically blow up in the final stages of the sampling process for singular data distributions such as those concentrated on lower-dimensional manifolds, and is therefore difficult to approximate by a network. To overcome this difficulty, we derive a new target function and associated loss, which remains bounded even for singular data distributions. We illustrate the theoretical findings with several numerical examples.
    

