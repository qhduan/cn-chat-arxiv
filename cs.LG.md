# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Time Weaver: A Conditional Time Series Generation Model](https://arxiv.org/abs/2403.02682) | 引入了Time Weaver模型，利用异构元数据改善时间序列生成，并指出标准评估指标的朴素扩展是不够的。 |
| [^2] | [SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems](https://arxiv.org/abs/2402.03145) | SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。 |
| [^3] | [Reward Collapse in Aligning Large Language Models.](http://arxiv.org/abs/2305.17608) | 本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。 |

# 详细

[^1]: 时间编织者：一种条件时间序列生成模型

    Time Weaver: A Conditional Time Series Generation Model

    [https://arxiv.org/abs/2403.02682](https://arxiv.org/abs/2403.02682)

    引入了Time Weaver模型，利用异构元数据改善时间序列生成，并指出标准评估指标的朴素扩展是不够的。

    

    arXiv:2403.02682v1 通告类型：新的 摘要：想象根据天气、电动车存在和位置生成城市的电力需求模式，这可以用于冬季寒冻期间的容量规划。这些真实世界的时间序列常常包含配对的异构背景元数据（天气、位置等）。当前的时间序列生成方法通常忽略了这些配对的元数据，其异构性给将现有的条件生成方法从图像、音频和视频领域适应到时间序列领域中带来了几个实际挑战。为了填补这一空白，我们引入了时间编织者，这是一种基于扩散的新型模型，利用形式各异的元数据（分类、连续甚至时间变量）显著改善时间序列生成。此外，我们证明了从图像到时间序列领域的标准评估指标的朴素扩展是不够的。

    arXiv:2403.02682v1 Announce Type: new  Abstract: Imagine generating a city's electricity demand pattern based on weather, the presence of an electric vehicle, and location, which could be used for capacity planning during a winter freeze. Such real-world time series are often enriched with paired heterogeneous contextual metadata (weather, location, etc.). Current approaches to time series generation often ignore this paired metadata, and its heterogeneity poses several practical challenges in adapting existing conditional generation approaches from the image, audio, and video domains to the time series domain. To address this gap, we introduce Time Weaver, a novel diffusion-based model that leverages the heterogeneous metadata in the form of categorical, continuous, and even time-variant variables to significantly improve time series generation. Additionally, we show that naive extensions of standard evaluation metrics from the image to the time series domain are insufficient. These m
    
[^2]: SafEDMD：一种专为非线性动态系统数据驱动控制而设计的认证学习架构

    SafEDMD: A certified learning architecture tailored to data-driven control of nonlinear dynamical systems

    [https://arxiv.org/abs/2402.03145](https://arxiv.org/abs/2402.03145)

    SafEDMD是一种基于EDMD的学习架构，通过稳定性和认证导向，生成可靠的数据驱动替代模型，并基于半定规划进行认证控制器设计。它在多个基准示例上展示了优于现有方法的优势。

    

    Koopman算子作为机器学习动态控制系统的理论基础，其中算子通过扩展动态模态分解（EDMD）启发式近似。在本文中，我们提出了稳定性和认证导向的EDMD（SafEDMD）：一种新颖的基于EDMD的学习架构，它提供了严格的证书，从而以数据驱动的方式生成可靠的替代模型。为了确保SafEDMD的可靠性，我们推导出比例误差界限，这些界限在原点处消失，并且适用于控制任务，从而基于半定规划进行认证控制器设计。我们通过几个基准示例说明了所开发的机制，并强调其相对于现有方法的优势。

    The Koopman operator serves as the theoretical backbone for machine learning of dynamical control systems, where the operator is heuristically approximated by extended dynamic mode decomposition (EDMD). In this paper, we propose Stability- and certificate-oriented EDMD (SafEDMD): a novel EDMD-based learning architecture which comes along with rigorous certificates, resulting in a reliable surrogate model generated in a data-driven fashion. To ensure trustworthiness of SafEDMD, we derive proportional error bounds, which vanish at the origin and are tailored for control tasks, leading to certified controller design based on semi-definite programming. We illustrate the developed machinery by means of several benchmark examples and highlight the advantages over state-of-the-art methods.
    
[^3]: 对齐大型语言模型中的奖励塌缩现象

    Reward Collapse in Aligning Large Language Models. (arXiv:2305.17608v1 [cs.LG])

    [http://arxiv.org/abs/2305.17608](http://arxiv.org/abs/2305.17608)

    本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。

    

    大型语言模型（LLMs），如ChatGPT和GPT-4，具有非凡的能力，部分原因在于将它们与训练在人类偏好上的奖励模型对齐，这些偏好通常表示为对响应提示的排名。本文记录了奖励塌陷现象，这是一种经验观察，其中基于排名的方法导致在训练的终止阶段生成的完整奖励分布\textit{无论}\textbf{prompt是什么}都是\textit{相同的}。这种结果是不可取的，因为像“写一篇关于你最好的朋友的简短故事”这样的开放式提示应生成完成它们的连续奖励范围，而像“新西兰的首都是什么”这样的特定提示应生成高或低奖励。我们的理论调查表明，奖励塌陷主要是由于基于排名的目标函数在优化过程中未能纳入与提示相关的信息所致。

    The extraordinary capabilities of large language models (LLMs) such as ChatGPT and GPT-4 are in part unleashed by aligning them with reward models that are trained on human preferences, which are often represented as rankings of responses to prompts. In this paper, we document the phenomenon of \textit{reward collapse}, an empirical observation where the prevailing ranking-based approach results in an \textit{identical} reward distribution \textit{regardless} of the prompts during the terminal phase of training. This outcome is undesirable as open-ended prompts like ``write a short story about your best friend'' should yield a continuous range of rewards for their completions, while specific prompts like ``what is the capital of New Zealand'' should generate either high or low rewards. Our theoretical investigation reveals that reward collapse is primarily due to the insufficiency of the ranking-based objective function to incorporate prompt-related information during optimization. Thi
    

