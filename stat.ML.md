# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters](https://arxiv.org/abs/2404.02270) | 将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。 |
| [^2] | [Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks](https://arxiv.org/abs/2404.02151) | 展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。 |
| [^3] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^4] | [High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media](https://arxiv.org/abs/2403.01318) | 提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。 |
| [^5] | [An Elementary Predictor Obtaining $2\sqrt{T}$ Distance to Calibration](https://arxiv.org/abs/2402.11410) | 给出了一种简单、高效、确定性的算法，该算法的校准距离误差最多为$2\sqrt{T}$ |
| [^6] | [Multiply Robust Causal Mediation Analysis with Continuous Treatments](https://arxiv.org/abs/2105.09254) | 本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。 |
| [^7] | [CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer.](http://arxiv.org/abs/2306.14590) | 本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。 |

# 详细

[^1]: 电力价格概率预测的点预测后处理：多样性至关重要

    Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters

    [https://arxiv.org/abs/2404.02270](https://arxiv.org/abs/2404.02270)

    将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。

    

    依赖于电力价格的预测分布进行操作决策相较于仅基于点预测的决策可以带来显著更高的利润。然而，在学术和工业环境中开发的大多数模型仅提供点预测。为了解决这一问题，我们研究了三种将点预测转换为概率预测的后处理方法：分位数回归平均、一致性预测和最近引入的等温分布回归。我们发现，虽然等温分布回归表现最为多样化，但将其预测分布与另外两种方法结合使用，相较于具有正态分布误差的基准模型，在德国电力市场的4.5年测试期间（涵盖COVID大流行和乌克兰战争），实现了约7.5%的改进。值得注意的是，这种组合的性能与最先进的Dis

    arXiv:2404.02270v1 Announce Type: new  Abstract: Operational decisions relying on predictive distributions of electricity prices can result in significantly higher profits compared to those based solely on point forecasts. However, the majority of models developed in both academic and industrial settings provide only point predictions. To address this, we examine three postprocessing methods for converting point forecasts into probabilistic ones: Quantile Regression Averaging, Conformal Prediction, and the recently introduced Isotonic Distributional Regression. We find that while IDR demonstrates the most varied performance, combining its predictive distributions with those of the other two methods results in an improvement of ca. 7.5% compared to a benchmark model with normally distributed errors, over a 4.5-year test period in the German power market spanning the COVID pandemic and the war in Ukraine. Remarkably, the performance of this combination is at par with state-of-the-art Dis
    
[^2]: 用简单自适应攻击越狱功能对齐的LLM

    Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks

    [https://arxiv.org/abs/2404.02151](https://arxiv.org/abs/2404.02151)

    展示了对齐的LLM对简单自适应越狱攻击不具有鲁棒性，并成功实现了在多个模型上几乎100%的攻击成功率，同时还介绍了对于不公开logprobs的模型如何进行越狱以及如何在受污染的模型中查找木马字符串的方法。

    

    我们展示了即使是最新的安全对齐的LLM也不具有抵抗简单自适应越狱攻击的稳健性。首先，我们展示了如何成功利用对logprobs的访问进行越狱：我们最初设计了一个对抗性提示模板（有时会适应目标LLM），然后我们在后缀上应用随机搜索以最大化目标logprob（例如token“Sure”），可能会进行多次重启。通过这种方式，我们实现了对GPT-3.5/4、Llama-2-Chat-7B/13B/70B、Gemma-7B和针对GCG攻击进行对抗训练的HarmBench上的R2D2等几乎100%的攻击成功率--根据GPT-4的评判。我们还展示了如何通过转移或预填充攻击以100%的成功率对所有不暴露logprobs的Claude模型进行越狱。此外，我们展示了如何在受污染的模型中使用对一组受限制的token执行随机搜索以查找木马字符串的方法--这项任务与许多其他任务共享相同的属性。

    arXiv:2404.02151v1 Announce Type: cross  Abstract: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize the target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve nearly 100\% attack success rate -- according to GPT-4 as a judge -- on GPT-3.5/4, Llama-2-Chat-7B/13B/70B, Gemma-7B, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with 100\% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many s
    
[^3]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^4]: 高维尾指数回归：以社交媒体病毒帖子文本分析为例

    High-Dimensional Tail Index Regression: with An Application to Text Analyses of Viral Posts in Social Media

    [https://arxiv.org/abs/2403.01318](https://arxiv.org/abs/2403.01318)

    提出了高维尾指数回归方法，利用正则化估计和去偏方法进行推断，支持理论的仿真研究，并在社交媒体病毒帖子文本分析中应用。

    

    受社交媒体病毒帖子的点赞分布（如点赞数量）经验性幂律的启发，我们引入了高维尾指数回归及其参数的估计和推断方法。我们提出了一种正则化估计量，证明了它的一致性，并推导了其收敛速度。为了进行推断，我们提出了去偏正则化估计，证明了去偏估计量的渐近正态性。仿真研究支持了我们的理论。这些方法被应用于对涉及 LGBTQ+ 话题的 X（原 Twitter）病毒帖子的文本分析。

    arXiv:2403.01318v1 Announce Type: cross  Abstract: Motivated by the empirical power law of the distributions of credits (e.g., the number of "likes") of viral posts in social media, we introduce the high-dimensional tail index regression and methods of estimation and inference for its parameters. We propose a regularized estimator, establish its consistency, and derive its convergence rate. To conduct inference, we propose to debias the regularized estimate, and establish the asymptotic normality of the debiased estimator. Simulation studies support our theory. These methods are applied to text analyses of viral posts in X (formerly Twitter) concerning LGBTQ+.
    
[^5]: 获得$2\sqrt{T}$到校准的基本预测器

    An Elementary Predictor Obtaining $2\sqrt{T}$ Distance to Calibration

    [https://arxiv.org/abs/2402.11410](https://arxiv.org/abs/2402.11410)

    给出了一种简单、高效、确定性的算法，该算法的校准距离误差最多为$2\sqrt{T}$

    

    Blasiok等人[2023]提出了校准距离作为一种自然的校准误差度量，与预期的校准误差(ECE)不同，它是连续的。最近，Qiao和Zheng [2024]给出了一个非构造性的论证，建立了一种在线预测器的存在，该预测器可以在对抗设置中获得$O(\sqrt{T})$的校准距离，而对于ECE来说是不可能的。他们将找到一种明确的、高效的算法作为一个需要解决的问题。我们解决了这个问题，并给出了一个非常简单、高效、确定性的算法，该算法的校准距离误差最多为$2\sqrt{T}$。

    arXiv:2402.11410v1 Announce Type: new  Abstract: Blasiok et al. [2023] proposed distance to calibration as a natural measure of calibration error that unlike expected calibration error (ECE) is continuous. Recently, Qiao and Zheng [2024] gave a non-constructive argument establishing the existence of an online predictor that can obtain $O(\sqrt{T})$ distance to calibration in the adversarial setting, which is known to be impossible for ECE. They leave as an open problem finding an explicit, efficient algorithm. We resolve this problem and give an extremely simple, efficient, deterministic algorithm that obtains distance to calibration error at most $2\sqrt{T}$.
    
[^6]: 在连续治疗下的多重稳健因果中介分析

    Multiply Robust Causal Mediation Analysis with Continuous Treatments

    [https://arxiv.org/abs/2105.09254](https://arxiv.org/abs/2105.09254)

    本文提出了一种适用于连续治疗环境的多重稳健因果中介分析估计器，采用了核平滑方法，并具有多重稳健性和渐近正态性。

    

    在许多应用中，研究人员对治疗或暴露对感兴趣的结果的直接和间接的因果效应。中介分析为鉴定和估计这些因果效应提供了一个严谨的框架。对于二元治疗，Tchetgen Tchetgen和Shpitser (2012)提出了直接和间接效应的高效估计器，基于参数的影响函数。这些估计器具有良好的性质，如多重稳健性和渐近正态性，同时允许对干扰参数进行低于根号n的收敛速度。然而，在涉及连续治疗的情况下，这些基于影响函数的估计器没有准备好应用，除非进行强参数假设。在这项工作中，我们利用核平滑方法提出了一种适用于连续治疗环境的估计器，受到Tchetgen Tchetgen的影响函数估计器的启发。

    In many applications, researchers are interested in the direct and indirect causal effects of a treatment or exposure on an outcome of interest. Mediation analysis offers a rigorous framework for identifying and estimating these causal effects. For binary treatments, efficient estimators for the direct and indirect effects are presented in Tchetgen Tchetgen and Shpitser (2012) based on the influence function of the parameter of interest. These estimators possess desirable properties, such as multiple-robustness and asymptotic normality, while allowing for slower than root-n rates of convergence for the nuisance parameters. However, in settings involving continuous treatments, these influence function-based estimators are not readily applicable without making strong parametric assumptions. In this work, utilizing a kernel-smoothing approach, we propose an estimator suitable for settings with continuous treatments inspired by the influence function-based estimator of Tchetgen Tchetgen an
    
[^7]: CST-YOLO: 一种基于改进的YOLOv7和CNN-Swin Transformer的血细胞检测新方法

    CST-YOLO: A Novel Method for Blood Cell Detection Based on Improved YOLOv7 and CNN-Swin Transformer. (arXiv:2306.14590v1 [cs.CV])

    [http://arxiv.org/abs/2306.14590](http://arxiv.org/abs/2306.14590)

    本论文提出了一种CST-YOLO算法，基于改进的YOLOv7和CNN-Swin Transformer，引入了几个有用的模型，有效提高了血细胞检测精度，实验结果显示其在三个血细胞数据集上均优于现有最先进算法。

    

    血细胞检测是计算机视觉中典型的小物体检测问题。本文提出了一种CST-YOLO模型，基于YOLOv7结构并使用CNN-Swin Transformer（CST）进行增强，这是一种CNN-Transformer融合的新尝试。同时，我们还引入了三个有用的模型：加权高效层聚合网络（W-ELAN）、多尺度通道分割（MCS）和级联卷积层（CatConv），以提高小物体检测精度。实验结果表明，我们提出的CST-YOLO在三个血细胞数据集上分别达到了92.7、95.6和91.1 mAP@0.5，优于最先进的物体检测器，如YOLOv5和YOLOv7。我们的代码可在https://github.com/mkang315/CST-YOLO上找到。

    Blood cell detection is a typical small-scale object detection problem in computer vision. In this paper, we propose a CST-YOLO model for blood cell detection based on YOLOv7 architecture and enhance it with the CNN-Swin Transformer (CST), which is a new attempt at CNN-Transformer fusion. We also introduce three other useful modules: Weighted Efficient Layer Aggregation Networks (W-ELAN), Multiscale Channel Split (MCS), and Concatenate Convolutional Layers (CatConv) in our CST-YOLO to improve small-scale object detection precision. Experimental results show that the proposed CST-YOLO achieves 92.7, 95.6, and 91.1 mAP@0.5 respectively on three blood cell datasets, outperforming state-of-the-art object detectors, e.g., YOLOv5 and YOLOv7. Our code is available at https://github.com/mkang315/CST-YOLO.
    

