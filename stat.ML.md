# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds](https://arxiv.org/abs/2404.02866) | 通过添加噪声到最后层的激活来保护隐私，使用HCR界限可量化保护机密性的可信度 |
| [^2] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^3] | [Supervised Autoencoder MLP for Financial Time Series Forecasting](https://arxiv.org/abs/2404.01866) | 通过监督型自编码器的使用和参数调整，可以显著提升金融时间序列预测的效果，对投资策略性能有重要影响。 |
| [^4] | [Preventing Model Collapse in Gaussian Process Latent Variable Models](https://arxiv.org/abs/2404.01697) | 本文通过理论分析投影方差对高斯过程潜变量模型的影响，以及集成了谱混合（SM）核和可微随机傅立叶特征（RFF）核逼近来解决核灵活性不足问题，从而防止模型崩溃。 |
| [^5] | [Convergence of Kinetic Langevin Monte Carlo on Lie groups](https://arxiv.org/abs/2403.12012) | 提出了一个基于Lie群的动力学Langevin Monte Carlo采样算法，通过添加噪声和精细离散化实现了Lie群结构的保持，并在W2距离下证明了连续动力学和离散采样器的指数收敛性。 |
| [^6] | [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://arxiv.org/abs/2403.04082) | 通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断 |
| [^7] | [Benefits of Transformer: In-Context Learning in Linear Regression Tasks with Unstructured Data](https://arxiv.org/abs/2402.00743) | 本研究通过线性回归任务的实验研究了Transformer在非结构化数据中的上下文学习能力，并解释了其中的关键组件。 |
| [^8] | [Uncertainty Quantification on Clinical Trial Outcome Prediction.](http://arxiv.org/abs/2401.03482) | 本研究将不确定性量化方法应用于临床试验结果预测，提高模型对微妙差异的识别能力，从而改善其整体性能。 |
| [^9] | [Polynomial Chaos Surrogate Construction for Random Fields with Parametric Uncertainty.](http://arxiv.org/abs/2311.00553) | 这项研究介绍了一种针对带有固有噪声和参数不确定性的随机计算模型的多项式混沌代理构建方法。 |
| [^10] | [Optimal Excess Risk Bounds for Empirical Risk Minimization on $p$-norm Linear Regression.](http://arxiv.org/abs/2310.12437) | 对于$p$-范数线性回归问题上的经验风险最小化，我们证明，在可实现的情况下，通过$O(d)$个样本就足够精确地恢复目标值，并且在其他情况下，我们证明了的高概率超出风险界。 |
| [^11] | [Clustering Without an Eigengap.](http://arxiv.org/abs/2308.15642) | 这个论文介绍了在随机块模型中进行图聚类的新算法，能够恢复大聚类，无论其他聚类的大小，并且对中等大小的聚类提出了新的技术挑战。 |
| [^12] | [Likelihood-free neural Bayes estimators for censored peaks-over-threshold models.](http://arxiv.org/abs/2306.15642) | 该论文提出了一种基于神经网络的无似然贝叶斯估计方法，用于构建高效的截尾超阈值模型估计器。该方法挑战了传统的基于截尾似然的空间极值推理，并在计算和统计效率上取得了显著的提升。 |
| [^13] | [DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation.](http://arxiv.org/abs/2306.02071) | 本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。 |
| [^14] | [Embeddings between Barron spaces with higher order activation functions.](http://arxiv.org/abs/2305.15839) | 本文研究了不同激活函数的Barron空间之间的嵌入，并证明了Barron空间的层次结构类似于Sobolev空间$H^m$。其中，修正功率单位激活函数在这个研究中特别重要。 |
| [^15] | [Expressiveness Remarks for Denoising Diffusion Models and Samplers.](http://arxiv.org/abs/2305.09605) | 本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。 |
| [^16] | [Exploring Numerical Priors for Low-Rank Tensor Completion with Generalized CP Decomposition.](http://arxiv.org/abs/2302.05881) | 本文提出了一种新的方法框架GCDTC，利用数值先验和广义CP分解实现了更高的低秩张量补全精度；同时介绍了一个算法SPTC，作为该框架的一个实现。在实验中，该方法表现出比现有技术更好的性能。 |
| [^17] | [Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation.](http://arxiv.org/abs/2106.03907) | 本论文提出了一种深度代理因果学习（PCL）方法，用于在存在混淆因素的情况下估计治疗对结果的因果效应。通过构建治疗和代理之间的模型，并利用该模型在给定代理的情况下学习治疗对结果的影响，PCL可以保证恢复真实的因果效应。作者还提出了一种名为深度特征代理变量方法（DFPV）的新方法，用于处理高维和非线性复杂关系的情况，并表明DFPV在合成基准测试中的性能优于最先进的PCL方法。 |

# 详细

[^1]: 通过Hammersley-Chapman-Robbins界限保证机密性

    Guarantees of confidentiality via Hammersley-Chapman-Robbins bounds

    [https://arxiv.org/abs/2404.02866](https://arxiv.org/abs/2404.02866)

    通过添加噪声到最后层的激活来保护隐私，使用HCR界限可量化保护机密性的可信度

    

    在深度神经网络推断过程中通过向最后几层的激活添加噪声来保护隐私是可能的。这些层中的激活被称为“特征”（少见的称为“嵌入”或“特征嵌入”）。添加的噪声有助于防止从嘈杂的特征中重建输入。通过对所有可能的无偏估计量的方差进行下限估计，量化了由此添加的噪声产生的机密性。经典不等式Hammersley和Chapman以及Robbins提供便利的、可计算的界限-- HCR界限。数值实验表明，对于包含10个类别的图像分类数据集“MNIST”和“CIFAR-10”，HCR界限在小型神经网络上表现良好。HCR界限似乎单独无法保证

    arXiv:2404.02866v1 Announce Type: new  Abstract: Protecting privacy during inference with deep neural networks is possible by adding noise to the activations in the last layers prior to the final classifiers or other task-specific layers. The activations in such layers are known as "features" (or, less commonly, as "embeddings" or "feature embeddings"). The added noise helps prevent reconstruction of the inputs from the noisy features. Lower bounding the variance of every possible unbiased estimator of the inputs quantifies the confidentiality arising from such added noise. Convenient, computationally tractable bounds are available from classic inequalities of Hammersley and of Chapman and Robbins -- the HCR bounds. Numerical experiments indicate that the HCR bounds are on the precipice of being effectual for small neural nets with the data sets, "MNIST" and "CIFAR-10," which contain 10 classes each for image classification. The HCR bounds appear to be insufficient on their own to guar
    
[^2]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^3]: 监督型自编码器多层感知机用于金融时间序列预测

    Supervised Autoencoder MLP for Financial Time Series Forecasting

    [https://arxiv.org/abs/2404.01866](https://arxiv.org/abs/2404.01866)

    通过监督型自编码器的使用和参数调整，可以显著提升金融时间序列预测的效果，对投资策略性能有重要影响。

    

    本文研究了如何通过使用神经网络中的监督型自编码器来增强金融时间序列预测，旨在改善投资策略表现。具体研究了噪声增强和三重障碍标记对风险调整回报的影响，使用夏普比率和信息比率。研究重点关注了从2010年1月1日至2022年4月30日期间作为交易资产的标普500指数，EUR/USD和BTC/USD。研究结果表明，具有平衡噪声增强和瓶颈大小的监督型自编码器显著提升了策略效果。然而，过多的噪声和大的瓶颈大小可能会损害表现，突出了精确参数调整的重要性。本文还提出了一种新的优化指标的推导，可与三重障碍标记一起使用。这项研究的结果对政策具有重要影响，暗示金融市场预测工具的持续改进。

    arXiv:2404.01866v1 Announce Type: new  Abstract: This paper investigates the enhancement of financial time series forecasting with the use of neural networks through supervised autoencoders, aiming to improve investment strategy performance. It specifically examines the impact of noise augmentation and triple barrier labeling on risk-adjusted returns, using the Sharpe and Information Ratios. The study focuses on the S&P 500 index, EUR/USD, and BTC/USD as the traded assets from January 1, 2010, to April 30, 2022. Findings indicate that supervised autoencoders, with balanced noise augmentation and bottleneck size, significantly boost strategy effectiveness. However, excessive noise and large bottleneck sizes can impair performance, highlighting the importance of precise parameter tuning. This paper also presents a derivation of a novel optimization metric that can be used with triple barrier labeling. The results of this study have substantial policy implications, suggesting that financi
    
[^4]: 防止高斯过程潜变量模型中的模型崩溃

    Preventing Model Collapse in Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2404.01697](https://arxiv.org/abs/2404.01697)

    本文通过理论分析投影方差对高斯过程潜变量模型的影响，以及集成了谱混合（SM）核和可微随机傅立叶特征（RFF）核逼近来解决核灵活性不足问题，从而防止模型崩溃。

    

    Gaussian process latent variable models (GPLVMs)是一类多才多艺的无监督学习模型，通常用于降维。然而，用GPLVMs对数据建模时常见的挑战包括核灵活性不足和投影噪声选择不当，导致了一种以模糊潜变量表示为主要特征的模型崩溃，这种表示不反映数据的潜在结构。本文首先从理论上通过线性GPLVM的视角研究了投影方差对模型崩溃的影响。其次，通过集成谱混合（SM）核和可微随机傅立叶特征（RFF）核逼近，解决了由于核灵活性不足导致的模型崩溃问题，从而保证了通过现成的自动微分工具实现学习核参数的计算可扩展性和效率。

    arXiv:2404.01697v1 Announce Type: cross  Abstract: Gaussian process latent variable models (GPLVMs) are a versatile family of unsupervised learning models, commonly used for dimensionality reduction. However, common challenges in modeling data with GPLVMs include inadequate kernel flexibility and improper selection of the projection noise, which leads to a type of model collapse characterized primarily by vague latent representations that do not reflect the underlying structure of the data. This paper addresses these issues by, first, theoretically examining the impact of the projection variance on model collapse through the lens of a linear GPLVM. Second, we address the problem of model collapse due to inadequate kernel flexibility by integrating the spectral mixture (SM) kernel and a differentiable random Fourier feature (RFF) kernel approximation, which ensures computational scalability and efficiency through off-the-shelf automatic differentiation tools for learning the kernel hype
    
[^5]: 基于Lie群的动力学Langevin Monte Carlo算法的收敛性

    Convergence of Kinetic Langevin Monte Carlo on Lie groups

    [https://arxiv.org/abs/2403.12012](https://arxiv.org/abs/2403.12012)

    提出了一个基于Lie群的动力学Langevin Monte Carlo采样算法，通过添加噪声和精细离散化实现了Lie群结构的保持，并在W2距离下证明了连续动力学和离散采样器的指数收敛性。

    

    最近，基于变分优化和左平凡化等技术构建了一个明确的、基于动量的动力学系统，用于优化定义在Lie群上的函数。我们适当地为优化动力学添加可处理的噪声，将其转化为采样动力学，利用动量变量是欧几里得的这一有利特性，尽管潜在函数存在于流形上。然后，我们通过精心离散化导致的动力学采样动力学提出了一个Lie群MCMC采样器。这种离散化完全保持了Lie群结构。在W2距离下，分别对连续动力学和离散采样器证明了指数收敛性，其中只需要Lie群的紧致性和潜在函数的测地L-光滑性。据我们所知，这是对动力学Langevin算法的第一个收敛性结果。

    arXiv:2403.12012v1 Announce Type: cross  Abstract: Explicit, momentum-based dynamics for optimizing functions defined on Lie groups was recently constructed, based on techniques such as variational optimization and left trivialization. We appropriately add tractable noise to the optimization dynamics to turn it into a sampling dynamics, leveraging the advantageous feature that the momentum variable is Euclidean despite that the potential function lives on a manifold. We then propose a Lie-group MCMC sampler, by delicately discretizing the resulting kinetic-Langevin-type sampling dynamics. The Lie group structure is exactly preserved by this discretization. Exponential convergence with explicit convergence rate for both the continuous dynamics and the discrete sampler are then proved under W2 distance. Only compactness of the Lie group and geodesically L-smoothness of the potential function are needed. To the best of our knowledge, this is the first convergence result for kinetic Langev
    
[^6]: 通过插值进行推断：对比表示可证明启用规划和推断

    Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference

    [https://arxiv.org/abs/2403.04082](https://arxiv.org/abs/2403.04082)

    通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断

    

    给定时间序列数据，我们如何回答诸如“未来会发生什么？”和“我们是如何到达这里的？”这类概率推断问题在观测值为高维时具有挑战性。本文展示了这些问题如何通过学习表示的紧凑闭式解决方案。关键思想是将对比学习的变体应用于时间序列数据。之前的工作已经表明，通过对比学习学到的表示编码了概率比。通过将之前的工作扩展以表明表示的边际分布是高斯分布，我们随后证明表示的联合分布也是高斯分布。这些结果共同表明，通过时间对比学习学到的表示遵循高斯马尔可夫链，一种图形模型，其中对表示进行的推断（例如预测、规划）对应于反演低维分布。

    arXiv:2403.04082v1 Announce Type: new  Abstract: Given time series data, how can we answer questions like "what will happen in the future?" and "how did we get here?" These sorts of probabilistic inference questions are challenging when observations are high-dimensional. In this paper, we show how these questions can have compact, closed form solutions in terms of learned representations. The key idea is to apply a variant of contrastive learning to time series data. Prior work already shows that the representations learned by contrastive learning encode a probability ratio. By extending prior work to show that the marginal distribution over representations is Gaussian, we can then prove that joint distribution of representations is also Gaussian. Taken together, these results show that representations learned via temporal contrastive learning follow a Gauss-Markov chain, a graphical model where inference (e.g., prediction, planning) over representations corresponds to inverting a low-
    
[^7]: Transformer的好处：在非结构化数据的线性回归任务中的上下文学习

    Benefits of Transformer: In-Context Learning in Linear Regression Tasks with Unstructured Data

    [https://arxiv.org/abs/2402.00743](https://arxiv.org/abs/2402.00743)

    本研究通过线性回归任务的实验研究了Transformer在非结构化数据中的上下文学习能力，并解释了其中的关键组件。

    

    实践中观察到，基于Transformer的模型在推理阶段能够学习上下文中的概念。现有的文献，例如\citet{zhang2023trained,huang2023context}对这种上下文学习能力提供了理论解释，但是他们假设每个样本的输入$x_i$和输出$y_i$都被嵌入到相同的令牌中（即结构化数据）。然而，在现实中，它们呈现为两个令牌（即非结构化数据\cite{wibisono2023role}）。在这种情况下，本文进行了线性回归任务的实验，研究了Transformer架构的好处，并提供了一些相应的理论直觉，解释了为什么Transformer可以从非结构化数据中学习。我们研究了在Transformer中起到上下文学习作用的确切组件。特别地，我们观察到（1）带有两层softmax（自我）注意力和前瞻性注意力掩码的Transformer可以从提示中学习，如果$y_i$在令牌中。

    In practice, it is observed that transformer-based models can learn concepts in context in the inference stage. While existing literature, e.g., \citet{zhang2023trained,huang2023context}, provide theoretical explanations on this in-context learning ability, they assume the input $x_i$ and the output $y_i$ for each sample are embedded in the same token (i.e., structured data). However, in reality, they are presented in two tokens (i.e., unstructured data \cite{wibisono2023role}). In this case, this paper conducts experiments in linear regression tasks to study the benefits of the architecture of transformers and provides some corresponding theoretical intuitions to explain why the transformer can learn from unstructured data. We study the exact components in a transformer that facilitate the in-context learning. In particular, we observe that (1) a transformer with two layers of softmax (self-)attentions with look-ahead attention mask can learn from the prompt if $y_i$ is in the token n
    
[^8]: 临床试验结果预测中的不确定性量化

    Uncertainty Quantification on Clinical Trial Outcome Prediction. (arXiv:2401.03482v1 [cs.LG])

    [http://arxiv.org/abs/2401.03482](http://arxiv.org/abs/2401.03482)

    本研究将不确定性量化方法应用于临床试验结果预测，提高模型对微妙差异的识别能力，从而改善其整体性能。

    

    不确定性量化在机器学习的不同领域中的重要性日益被认识到。准确评估模型预测的不确定性可以帮助研究人员和从业人员更深入地理解和增加信心。这在医学诊断和药物发现领域尤为重要，因为可靠的预测直接影响研究质量和患者健康。本文提出将不确定性量化纳入临床试验结果预测中。我们的主要目标是提高模型辨别微妙差异的能力，从而显著改善其整体性能。我们采用了一种选择性分类方法来实现我们的目标，并将其与层次交互网络(HINT)无缝集成，HINT是临床试验预测建模的最前沿。选择性分类涵盖了一系列不确定性量化方法，使模型能够保留信息以供进一步分析。

    The importance of uncertainty quantification is increasingly recognized in the diverse field of machine learning. Accurately assessing model prediction uncertainty can help provide deeper understanding and confidence for researchers and practitioners. This is especially critical in medical diagnosis and drug discovery areas, where reliable predictions directly impact research quality and patient health.  In this paper, we proposed incorporating uncertainty quantification into clinical trial outcome predictions. Our main goal is to enhance the model's ability to discern nuanced differences, thereby significantly improving its overall performance.  We have adopted a selective classification approach to fulfill our objective, integrating it seamlessly with the Hierarchical Interaction Network (HINT), which is at the forefront of clinical trial prediction modeling. Selective classification, encompassing a spectrum of methods for uncertainty quantification, empowers the model to withhold de
    
[^9]: 针对带参数不确定性的随机场的多项式混沌代理构建

    Polynomial Chaos Surrogate Construction for Random Fields with Parametric Uncertainty. (arXiv:2311.00553v1 [stat.ME])

    [http://arxiv.org/abs/2311.00553](http://arxiv.org/abs/2311.00553)

    这项研究介绍了一种针对带有固有噪声和参数不确定性的随机计算模型的多项式混沌代理构建方法。

    

    工程和应用科学依靠计算实验来严谨地研究物理系统。用于探究这些系统的数学模型非常复杂，而且采样密集的研究通常需要大量模拟以获得可接受的准确性。代理模型为避免采样这些复杂模型的高计算开销提供了一种方法。尤其是，在参数不确定性为主要不确定因素的确定性模型的不确定性量化研究中，多项式混沌展开（PCEs）已经取得了成功。我们讨论了一种对传统的PCE代理模型的扩展，以实现对具有固有噪声和参数不确定性的随机计算模型的代理构建。我们通过Rosenblatt变换在固有和参数不确定性的联合空间上开发了一个PCE代理，然后通过Karhunen-Loeve展开将其扩展到随机场数据上。

    Engineering and applied science rely on computational experiments to rigorously study physical systems. The mathematical models used to probe these systems are highly complex, and sampling-intensive studies often require prohibitively many simulations for acceptable accuracy. Surrogate models provide a means of circumventing the high computational expense of sampling such complex models. In particular, polynomial chaos expansions (PCEs) have been successfully used for uncertainty quantification studies of deterministic models where the dominant source of uncertainty is parametric. We discuss an extension to conventional PCE surrogate modeling to enable surrogate construction for stochastic computational models that have intrinsic noise in addition to parametric uncertainty. We develop a PCE surrogate on a joint space of intrinsic and parametric uncertainty, enabled by Rosenblatt transformations, and then extend the construction to random field data via the Karhunen-Loeve expansion. We 
    
[^10]: 在$p$-范数线性回归的经验风险最小化中的最优超出风险界

    Optimal Excess Risk Bounds for Empirical Risk Minimization on $p$-norm Linear Regression. (arXiv:2310.12437v1 [math.ST])

    [http://arxiv.org/abs/2310.12437](http://arxiv.org/abs/2310.12437)

    对于$p$-范数线性回归问题上的经验风险最小化，我们证明，在可实现的情况下，通过$O(d)$个样本就足够精确地恢复目标值，并且在其他情况下，我们证明了的高概率超出风险界。

    

    本文研究了在$p$-范数线性回归问题上的经验风险最小化的性能。我们证明，在可实现的情况下，在没有矩的假设下，通过$O(d)$个样本就足够精确地恢复目标值，相应性化常数有关。否则，对于$p \in [2, \infty)$并且对目标和协变量具有较弱的矩假设，我们证明了经验风险最小化的高概率超出风险界，其中主要项与渐近精确率匹配，常数仅依赖于$p$。在保证风险函数在其最小化点上存在Hessian矩阵的情况下，我们将此结果扩展到$p \in (1, 2)$的情况下。

    We study the performance of empirical risk minimization on the $p$-norm linear regression problem for $p \in (1, \infty)$. We show that, in the realizable case, under no moment assumptions, and up to a distribution-dependent constant, $O(d)$ samples are enough to exactly recover the target. Otherwise, for $p \in [2, \infty)$, and under weak moment assumptions on the target and the covariates, we prove a high probability excess risk bound on the empirical risk minimizer whose leading term matches, up to a constant that depends only on $p$, the asymptotically exact rate. We extend this result to the case $p \in (1, 2)$ under mild assumptions that guarantee the existence of the Hessian of the risk at its minimizer.
    
[^11]: 无需特征间隔的聚类方法

    Clustering Without an Eigengap. (arXiv:2308.15642v1 [cs.LG])

    [http://arxiv.org/abs/2308.15642](http://arxiv.org/abs/2308.15642)

    这个论文介绍了在随机块模型中进行图聚类的新算法，能够恢复大聚类，无论其他聚类的大小，并且对中等大小的聚类提出了新的技术挑战。

    

    我们在随机块模型（SBM）中研究了具有大聚类和小不可恢复聚类的图聚类问题。之前的方法要么不允许小于$ o（\sqrt {n}）$大小的小聚类，要么要求最小可恢复聚类和最大不可恢复聚类之间存在大小间隔。我们提供了一个基于半定规划（SDP）的算法，它消除了这些要求，并可以确定地恢复大聚类，而不考虑其他聚类的大小。中等大小的聚类对分析提出了独特的挑战，因为它们接近恢复阈值，非常敏感于小的噪声扰动，不允许闭合形式的候选解决方案。我们开发了新颖的技术，包括leave-one-out风格的论证，即使去掉一行噪声也可能大幅改变SDP解决方案，仍然可以控制SDP解决方案与噪声向量之间的相关性。

    We study graph clustering in the Stochastic Block Model (SBM) in the presence of both large clusters and small, unrecoverable clusters. Previous approaches achieving exact recovery do not allow any small clusters of size $o(\sqrt{n})$, or require a size gap between the smallest recovered cluster and the largest non-recovered cluster. We provide an algorithm based on semidefinite programming (SDP) which removes these requirements and provably recovers large clusters regardless of the remaining cluster sizes. Mid-sized clusters pose unique challenges to the analysis, since their proximity to the recovery threshold makes them highly sensitive to small noise perturbations and precludes a closed-form candidate solution. We develop novel techniques, including a leave-one-out-style argument which controls the correlation between SDP solutions and noise vectors even when the removal of one row of noise can drastically change the SDP solution. We also develop improved eigenvalue perturbation bo
    
[^12]: 无似然神经贝叶斯估计的截尾超阈值模型

    Likelihood-free neural Bayes estimators for censored peaks-over-threshold models. (arXiv:2306.15642v1 [stat.ME])

    [http://arxiv.org/abs/2306.15642](http://arxiv.org/abs/2306.15642)

    该论文提出了一种基于神经网络的无似然贝叶斯估计方法，用于构建高效的截尾超阈值模型估计器。该方法挑战了传统的基于截尾似然的空间极值推理，并在计算和统计效率上取得了显著的提升。

    

    在高维度下，对于空间极值依赖模型的推理往往因其依赖于难以处理的或截尾的似然函数而造成计算负担。利用最近在无似然推理方面的进展，我们通过在神经网络架构中编码截尾信息，为截尾超阈值模型构建了高效的估计器。我们的新方法对于传统的基于截尾似然的空间极值推理提出了挑战。我们的模拟研究表明，在推断流行的极值依赖模型（如最大稳定模型、r-帕累托模型和随机比例混合过程）时，相对于竞争的基于似然的方法，我们的新估计器在计算和统计效率方面提供了显著的提升。

    Inference for spatial extremal dependence models can be computationally burdensome in moderate-to-high dimensions due to their reliance on intractable and/or censored likelihoods. Exploiting recent advances in likelihood-free inference with neural Bayes estimators (that is, neural estimators that target Bayes estimators), we develop a novel approach to construct highly efficient estimators for censored peaks-over-threshold models by encoding censoring information in the neural network architecture. Our new method provides a paradigm shift that challenges traditional censored likelihood-based inference for spatial extremes. Our simulation studies highlight significant gains in both computational and statistical efficiency, relative to competing likelihood-based approaches, when applying our novel estimators for inference of popular extremal dependence models, such as max-stable, $r$-Pareto, and random scale mixture processes. We also illustrate that it is possible to train a single esti
    
[^13]: DU-Shapley: 一种有效的数据集价值评估的Shapley值代理

    DU-Shapley: A Shapley Value Proxy for Efficient Dataset Valuation. (arXiv:2306.02071v1 [cs.AI])

    [http://arxiv.org/abs/2306.02071](http://arxiv.org/abs/2306.02071)

    本论文提出了一种称为DU-Shapley的方法，用于更有效地计算Shapley值，以实现机器学习中的数据集价值评估。

    

    许多机器学习问题需要进行数据集评估，即量化将一个单独的数据集与其他数据集聚合的增量收益，以某些相关预定义公用事业为基础。最近，Shapley值被提出作为实现这一目标的一种基本工具，因为它具有形式公理证明。由于其计算通常需要指数时间，因此考虑基于Monte Carlo积分的标准近似策略。然而，在某些情况下，这种通用近似方法仍然昂贵。本文利用数据集评估问题的结构知识，设计了更有效的Shapley值估计器。我们提出了一种新的Shapley值近似，称为离散均匀Shapley (DU-Shapley)，其表达为期望值

    Many machine learning problems require performing dataset valuation, i.e. to quantify the incremental gain, to some relevant pre-defined utility, of aggregating an individual dataset to others. As seminal examples, dataset valuation has been leveraged in collaborative and federated learning to create incentives for data sharing across several data owners. The Shapley value has recently been proposed as a principled tool to achieve this goal due to formal axiomatic justification. Since its computation often requires exponential time, standard approximation strategies based on Monte Carlo integration have been considered. Such generic approximation methods, however, remain expensive in some cases. In this paper, we exploit the knowledge about the structure of the dataset valuation problem to devise more efficient Shapley value estimators. We propose a novel approximation of the Shapley value, referred to as discrete uniform Shapley (DU-Shapley) which is expressed as an expectation under 
    
[^14]: 具有高阶激活函数的Barron空间之间的嵌入

    Embeddings between Barron spaces with higher order activation functions. (arXiv:2305.15839v1 [stat.ML])

    [http://arxiv.org/abs/2305.15839](http://arxiv.org/abs/2305.15839)

    本文研究了不同激活函数的Barron空间之间的嵌入，并证明了Barron空间的层次结构类似于Sobolev空间$H^m$。其中，修正功率单位激活函数在这个研究中特别重要。

    

    无限宽浅层神经网络的逼近性质很大程度上取决于激活函数的选择。为了了解这种影响，我们研究了具有不同激活函数的Barron空间之间的嵌入。通过提供用于表示函数$f$的测量$\mu$上的推进映射来证明这些嵌入。一种特别感兴趣的激活函数是给定为$\operatorname{RePU}_s(x)=\max(0,x)^s$的修正功率单位($\operatorname{RePU}$)。对于许多常用的激活函数，可以使用众所周知的泰勒余项定理构造推进映射，这使我们能够证明相关Barron空间嵌入到具有$\operatorname{RePU}$作为激活函数的Barron空间中。此外，与$\operatorname{RePU}_s$相关的Barron空间具有类似于Sobolev空间$H^m$的分层结构。

    The approximation properties of infinitely wide shallow neural networks heavily depend on the choice of the activation function. To understand this influence, we study embeddings between Barron spaces with different activation functions. These embeddings are proven by providing push-forward maps on the measures $\mu$ used to represent functions $f$. An activation function of particular interest is the rectified power unit ($\operatorname{RePU}$) given by $\operatorname{RePU}_s(x)=\max(0,x)^s$. For many commonly used activation functions, the well-known Taylor remainder theorem can be used to construct a push-forward map, which allows us to prove the embedding of the associated Barron space into a Barron space with a $\operatorname{RePU}$ as activation function. Moreover, the Barron spaces associated with the $\operatorname{RePU}_s$ have a hierarchical structure similar to the Sobolev spaces $H^m$.
    
[^15]: 漫扩扩散模型和采样器的表达能力研究

    Expressiveness Remarks for Denoising Diffusion Models and Samplers. (arXiv:2305.09605v1 [stat.ML])

    [http://arxiv.org/abs/2305.09605](http://arxiv.org/abs/2305.09605)

    本文在漫扩扩散模型和采样器方面进行了表达能力的研究，通过将已知的神经网络逼近结果扩展到漫扩扩散模型和采样器来实现。

    

    漫扩扩散模型是一类生成模型，在许多领域最近已经取得了最先进的结果。通过漫扩过程逐渐向数据中添加噪声，将数据分布转化为高斯分布。然后，通过模拟该漫扩的时间反演的逼近来获取生成模型的样本，刚开始这个漫扩模拟的初始值是高斯样本。最近的研究探索了将漫扩模型适应于采样和推断任务。本文基于众所周知的与F\"ollmer漂移类似的随机控制联系，将针对F\"ollmer漂移的已知神经网络逼近结果扩展到漫扩扩散模型和采样器。

    Denoising diffusion models are a class of generative models which have recently achieved state-of-the-art results across many domains. Gradual noise is added to the data using a diffusion process, which transforms the data distribution into a Gaussian. Samples from the generative model are then obtained by simulating an approximation of the time reversal of this diffusion initialized by Gaussian samples. Recent research has explored adapting diffusion models for sampling and inference tasks. In this paper, we leverage known connections to stochastic control akin to the F\"ollmer drift to extend established neural network approximation results for the F\"ollmer drift to denoising diffusion models and samplers.
    
[^16]: 探索基于数值先验的广义CP分解低秩张量补全算法

    Exploring Numerical Priors for Low-Rank Tensor Completion with Generalized CP Decomposition. (arXiv:2302.05881v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2302.05881](http://arxiv.org/abs/2302.05881)

    本文提出了一种新的方法框架GCDTC，利用数值先验和广义CP分解实现了更高的低秩张量补全精度；同时介绍了一个算法SPTC，作为该框架的一个实现。在实验中，该方法表现出比现有技术更好的性能。

    

    张量补全在计算机视觉、数据分析和信号处理等领域中具有重要意义。最近，低秩张量补全这一类别的方法得到了广泛研究，对补全张量施加低秩结构。虽然这些方法取得了巨大成功，但尚未考虑到张量元素的数值先验信息。忽略数值先验将导致丢失关于数据的重要信息，因此阻止算法达到最优精度。本研究试图构建一个新的方法框架，名为GCDTC（广义CP分解张量补全），以利用数值先验并实现更高的张量补全精度。在这个新引入的框架中，将广义的CP分解应用于低秩张量补全。本文还提出了一种名为SPTC（平滑泊松张量补全）的算法，用于非负整数张量补全，作为GCDTC框架的一个实现。通过对合成和真实世界数据集的大量实验，证明所提出的方法相比于现有技术具有更优的张量补全性能。

    Tensor completion is important to many areas such as computer vision, data analysis, and signal processing. Enforcing low-rank structures on completed tensors, a category of methods known as low-rank tensor completion has recently been studied extensively. While such methods attained great success, none considered exploiting numerical priors of tensor elements. Ignoring numerical priors causes loss of important information regarding the data, and therefore prevents the algorithms from reaching optimal accuracy. This work attempts to construct a new methodological framework called GCDTC (Generalized CP Decomposition Tensor Completion) for leveraging numerical priors and achieving higher accuracy in tensor completion. In this newly introduced framework, a generalized form of CP Decomposition is applied to low-rank tensor completion. This paper also proposes an algorithm known as SPTC (Smooth Poisson Tensor Completion) for nonnegative integer tensor completion as an instantiation of the G
    
[^17]: 深度代理因果学习及其在混淆赌博策略评估中的应用

    Deep Proxy Causal Learning and its Application to Confounded Bandit Policy Evaluation. (arXiv:2106.03907v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.03907](http://arxiv.org/abs/2106.03907)

    本论文提出了一种深度代理因果学习（PCL）方法，用于在存在混淆因素的情况下估计治疗对结果的因果效应。通过构建治疗和代理之间的模型，并利用该模型在给定代理的情况下学习治疗对结果的影响，PCL可以保证恢复真实的因果效应。作者还提出了一种名为深度特征代理变量方法（DFPV）的新方法，用于处理高维和非线性复杂关系的情况，并表明DFPV在合成基准测试中的性能优于最先进的PCL方法。

    

    代理因果学习（PCL）是一种在存在未观察到的混淆因素时，利用代理（结构化侧面信息）估计治疗对结果的因果效应的方法。这是通过两阶段回归实现的：在第一阶段，我们建模治疗和代理之间的关系；在第二阶段，我们利用这个模型来学习在给定代理提供的上下文下，治疗对结果的影响。PCL在可识别条件下保证恢复真实的因果效应。我们提出了一种新的PCL方法，深度特征代理变量方法（DFPV），以解决代理、治疗和结果为高维且具有非线性复杂关系的情况，如深度神经网络特征表示。我们表明DFPV在具有挑战性的合成基准测试中优于最近的最先进的PCL方法，包括涉及高维图像数据的设置。此外，我们还展示了PCL的应用...

    Proxy causal learning (PCL) is a method for estimating the causal effect of treatments on outcomes in the presence of unobserved confounding, using proxies (structured side information) for the confounder. This is achieved via two-stage regression: in the first stage, we model relations among the treatment and proxies; in the second stage, we use this model to learn the effect of treatment on the outcome, given the context provided by the proxies. PCL guarantees recovery of the true causal effect, subject to identifiability conditions. We propose a novel method for PCL, the deep feature proxy variable method (DFPV), to address the case where the proxies, treatments, and outcomes are high-dimensional and have nonlinear complex relationships, as represented by deep neural network features. We show that DFPV outperforms recent state-of-the-art PCL methods on challenging synthetic benchmarks, including settings involving high dimensional image data. Furthermore, we show that PCL can be app
    

