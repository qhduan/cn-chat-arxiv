# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://arxiv.org/abs/2403.20150) | TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。 |
| [^2] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^3] | [Latent Plan Transformer: Planning as Latent Variable Inference](https://arxiv.org/abs/2402.04647) | 潜在计划变换器（LPT）是一种新颖的模型，它通过将Transformer-based轨迹生成器和最终回报连接起来，并利用潜在空间进行规划。在学习中，通过对潜在变量的后验采样形成一致的抽象，在测试时通过推断潜在变量指导自回归策略。实验证明LPT能够从次优解中发现改进的决策。 |
| [^4] | [Explainable Anomaly Detection in Images and Videos: A Survey](https://arxiv.org/abs/2302.06670) | 这项研究提供了针对图像和视频的可解释异常检测方法的首次调研，为机器学习学术界和实际应用提供了重要参考。 |
| [^5] | [TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models.](http://arxiv.org/abs/2311.01301) | TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。 |
| [^6] | [A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays.](http://arxiv.org/abs/2310.17176) | 本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。 |
| [^7] | [A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models.](http://arxiv.org/abs/2309.06230) | 该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。 |
| [^8] | [Learning Zero-Sum Linear Quadratic Games with Improved Sample Complexity.](http://arxiv.org/abs/2309.04272) | 这项研究提出了改进样本复杂性的零和线性二次博弈，并发现了自然策略梯度方法的隐式正则化属性。在无模型参数知识的情况下，他们还提出了第一个多项式样本复杂性算法来达到Nash均衡。 |
| [^9] | [Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach.](http://arxiv.org/abs/2307.01316) | 本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。 |
| [^10] | [Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators.](http://arxiv.org/abs/2305.14561) | 本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。 |
| [^11] | [Kernel Ridge Regression Inference.](http://arxiv.org/abs/2302.06578) | 我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。 |
| [^12] | [Compressing Transformer-based self-supervised models for speech processing.](http://arxiv.org/abs/2211.09949) | 本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。 |

# 详细

[^1]: TFB：面向时间序列预测方法全面且公平的基准比较

    TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

    [https://arxiv.org/abs/2403.20150](https://arxiv.org/abs/2403.20150)

    TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。

    

    时间序列会在经济、交通、健康和能源等不同领域中产生，对未来数值的预测在许多重要应用中起着关键作用。不出所料，许多预测方法被提出。为了确保进展，有必要能够以全面且可靠的方式经验性地研究和比较这些方法。为了实现这一目标，我们提出了TFB，一个自动化的时间序列预测（TSF）方法基准测试。TFB通过解决与数据集、比较方法和评估管道相关的缺点，推动了最新技术的发展：1）数据领域覆盖不足，2）对传统方法的刻板印象，3）不一致和不灵活的流程。为了获得更好的领域覆盖率，我们包括了来自10个不同领域的数据集：交通、电力、能源、环境、自然、经济、股票市场、银行、健康和网络。我们还提供了一个时间序列特性

    arXiv:2403.20150v1 Announce Type: cross  Abstract: Time series are generated in diverse domains such as economic, traffic, health, and energy, where forecasting of future values has numerous important applications. Not surprisingly, many forecasting methods are being proposed. To ensure progress, it is essential to be able to study and compare such methods empirically in a comprehensive and reliable manner. To achieve this, we propose TFB, an automated benchmark for Time Series Forecasting (TSF) methods. TFB advances the state-of-the-art by addressing shortcomings related to datasets, comparison methods, and evaluation pipelines: 1) insufficient coverage of data domains, 2) stereotype bias against traditional methods, and 3) inconsistent and inflexible pipelines. To achieve better domain coverage, we include datasets from 10 different domains: traffic, electricity, energy, the environment, nature, economic, stock markets, banking, health, and the web. We also provide a time series char
    
[^2]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^3]: 潜在计划变换器：规划作为潜在变量推断

    Latent Plan Transformer: Planning as Latent Variable Inference

    [https://arxiv.org/abs/2402.04647](https://arxiv.org/abs/2402.04647)

    潜在计划变换器（LPT）是一种新颖的模型，它通过将Transformer-based轨迹生成器和最终回报连接起来，并利用潜在空间进行规划。在学习中，通过对潜在变量的后验采样形成一致的抽象，在测试时通过推断潜在变量指导自回归策略。实验证明LPT能够从次优解中发现改进的决策。

    

    在追求长期回报的任务中，规划变得必要。我们研究了利用离线强化学习的数据集进行规划的生成建模。具体来说，我们确定了在缺乏逐步奖励的情况下的时间一致性是一个关键的技术挑战。我们引入了潜在计划变换器（LPT），这是一种新颖的模型，它利用了一个潜在空间来连接基于Transformer的轨迹生成器和最终回报。LPT可以通过轨迹-回报对的最大似然估计来学习。在学习中，通过对潜在变量的后验采样，尽管有限的上下文，自然地聚集子轨迹以形成一致的抽象。在测试时，通过预期回报对潜在变量进行推断，实现了规划作为推断的思想。然后，它在整个回合中指导自回归策略，起到一个计划的作用。我们的实验表明，LPT可以从次优解中发现改进的决策。

    In tasks aiming for long-term returns, planning becomes necessary. We study generative modeling for planning with datasets repurposed from offline reinforcement learning. Specifically, we identify temporal consistency in the absence of step-wise rewards as one key technical challenge. We introduce the Latent Plan Transformer (LPT), a novel model that leverages a latent space to connect a Transformer-based trajectory generator and the final return. LPT can be learned with maximum likelihood estimation on trajectory-return pairs. In learning, posterior sampling of the latent variable naturally gathers sub-trajectories to form a consistent abstraction despite the finite context. During test time, the latent variable is inferred from an expected return before policy execution, realizing the idea of planning as inference. It then guides the autoregressive policy throughout the episode, functioning as a plan. Our experiments demonstrate that LPT can discover improved decisions from suboptima
    
[^4]: 图像和视频中可解释的异常检测：一项调研

    Explainable Anomaly Detection in Images and Videos: A Survey

    [https://arxiv.org/abs/2302.06670](https://arxiv.org/abs/2302.06670)

    这项研究提供了针对图像和视频的可解释异常检测方法的首次调研，为机器学习学术界和实际应用提供了重要参考。

    

    异常检测和定位视觉数据（包括图像和视频）在机器学习学术界和应用实际场景中具有重要意义。尽管近年来可视异常检测技术迅速发展，但对于这些黑盒模型的解释以及为何可以区分异常的合理解释却十分稀缺。本文首次提供了一项集中于可解释视觉异常检测方法的调研。我们首先介绍了图像级和视频级异常检测的基本背景。然后，作为本调研的主要内容，我们展示了针对图像和视频的可解释异常检测方法的全面和详尽的文献综述。接下来，我们分析了为什么一些可解释异常检测方法可以应用于图像和视频，而另一些则只能应用于一种模态。此外，我们提供了总结

    arXiv:2302.06670v2 Announce Type: replace-cross  Abstract: Anomaly detection and localization of visual data, including images and videos, are of great significance in both machine learning academia and applied real-world scenarios. Despite the rapid development of visual anomaly detection techniques in recent years, the interpretations of these black-box models and reasonable explanations of why anomalies can be distinguished out are scarce. This paper provides the first survey concentrated on explainable visual anomaly detection methods. We first introduce the basic background of image-level and video-level anomaly detection. Then, as the main content of this survey, a comprehensive and exhaustive literature review of explainable anomaly detection methods for both images and videos is presented. Next, we analyze why some explainable anomaly detection methods can be applied to both images and videos and why others can be only applied to one modality. Additionally, we provide summaries
    
[^5]: TRIALSCOPE：一个统一的因果框架，用于利用生物医学语言模型扩展实际世界证据生成

    TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models. (arXiv:2311.01301v1 [cs.LG])

    [http://arxiv.org/abs/2311.01301](http://arxiv.org/abs/2311.01301)

    TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。

    

    实际世界数据的快速数字化为优化医疗服务和加速生物医学发现提供了前所未有的机会。然而，在实践中，这些数据往往以非结构化形式存在，如电子医疗记录中的临床笔记，并且通常受到混杂因素的困扰。本文介绍了TRIALSCOPE，一个用于从人群级观察数据中提取实际世界证据的统一框架。TRIALSCOPE利用生物医学语言模型来扩展规模化的临床文本，采用先进的概率建模进行去噪和插补，并结合最先进的因果推断技术来应对常见的混杂因素。利用临床试验规范作为通用表示形式，TRIALSCOPE提供了一个一键式解决方案，可使用观察数据生成和推理临床假设。在一个包含超过一百万个癌症患者的大规模实际世界数据集上进行了广泛的实验和分析。

    The rapid digitization of real-world data offers an unprecedented opportunity for optimizing healthcare delivery and accelerating biomedical discovery. In practice, however, such data is most abundantly available in unstructured forms, such as clinical notes in electronic medical records (EMRs), and it is generally plagued by confounders. In this paper, we present TRIALSCOPE, a unifying framework for distilling real-world evidence from population-level observational data. TRIALSCOPE leverages biomedical language models to structure clinical text at scale, employs advanced probabilistic modeling for denoising and imputation, and incorporates state-of-the-art causal inference techniques to combat common confounders. Using clinical trial specification as generic representation, TRIALSCOPE provides a turn-key solution to generate and reason with clinical hypotheses using observational data. In extensive experiments and analyses on a large-scale real-world dataset with over one million canc
    
[^6]: 从全景X射线中进行牙齿分割和定位的深度学习方法

    A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays. (arXiv:2310.17176v1 [cs.CV])

    [http://arxiv.org/abs/2310.17176](http://arxiv.org/abs/2310.17176)

    本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。

    

    准确的牙齿分割和定位在现代口腔保健中是基础，可实现精确诊断、治疗计划和牙齿种植设计。本研究提出了一种综合的方法，利用深度学习技术从全景X射线图像中进行牙齿分割和定位。我们根据FUSegNet构建了我们的模型，这是一种最初用于创面分割的流行模型，并通过将基于网格的注意力门引入跳跃连接进行了修改。我们通过主成分分析（PCA）引入定向边界框（OBB）生成，以实现精确的牙齿定位估计。在公开可获得的DNS数据集上评估我们的方法，该数据集包括543个全景X射线图像，我们在牙齿实例分割中得到了最高的交并比（IoU）得分82.43%，Dice相似系数（DSC）得分90.37%，在OBB分析中，我们获得了旋转的交并比（RIoU）得分82.82%。

    Accurate teeth segmentation and orientation are fundamental in modern oral healthcare, enabling precise diagnosis, treatment planning, and dental implant design. In this study, we present a comprehensive approach to teeth segmentation and orientation from panoramic X-ray images, leveraging deep learning techniques. We build our model based on FUSegNet, a popular model originally developed for wound segmentation, and introduce modifications by incorporating grid-based attention gates into the skip connections. We introduce oriented bounding box (OBB) generation through principal component analysis (PCA) for precise tooth orientation estimation. Evaluating our approach on the publicly available DNS dataset, comprising 543 panoramic X-ray images, we achieve the highest Intersection-over-Union (IoU) score of 82.43% and Dice Similarity Coefficient (DSC) score of 90.37% among compared models in teeth instance segmentation. In OBB analysis, we obtain the Rotated IoU (RIoU) score of 82.82%. We
    
[^7]: 单指数模型中最佳子集选择的一致性和可扩展算法

    A Consistent and Scalable Algorithm for Best Subset Selection in Single Index Models. (arXiv:2309.06230v1 [stat.ML])

    [http://arxiv.org/abs/2309.06230](http://arxiv.org/abs/2309.06230)

    该论文提出了针对高维单指数模型中最佳子集选择的一致性和可扩展算法，通过使用广义信息准则来确定支持的回归系数大小，消除了模型选择的调优需求，并具有子集选择一致性和高概率下的理想属性。

    

    高维数据的分析引发了对单指数模型（SIMs）和最佳子集选择的增加兴趣。SIMs为高维数据提供了一种可解释和灵活的建模框架，而最佳子集选择旨在从大量的预测因子中找到稀疏模型。然而，在高维模型中的最佳子集选择被认为是计算上难以处理的。现有的方法倾向于放宽选择，但不能得到最佳子集解。在本文中，我们通过提出第一个经过证明的针对高维SIMs中最佳子集选择的可扩展算法，直接解决了计算难题。我们的算法解具有子集选择一致性，并且几乎肯定具有用于参数估计的虚拟属性。该算法包括一个广义信息准则来确定回归系数的支持大小，消除模型选择调整。此外，我们的方法不假设误差分布或特定参数。

    Analysis of high-dimensional data has led to increased interest in both single index models (SIMs) and best subset selection. SIMs provide an interpretable and flexible modeling framework for high-dimensional data, while best subset selection aims to find a sparse model from a large set of predictors. However, best subset selection in high-dimensional models is known to be computationally intractable. Existing methods tend to relax the selection, but do not yield the best subset solution. In this paper, we directly tackle the intractability by proposing the first provably scalable algorithm for best subset selection in high-dimensional SIMs. Our algorithmic solution enjoys the subset selection consistency and has the oracle property with a high probability. The algorithm comprises a generalized information criterion to determine the support size of the regression coefficients, eliminating the model selection tuning. Moreover, our method does not assume an error distribution or a specif
    
[^8]: 学习改进样本复杂性的零和线性二次博弈

    Learning Zero-Sum Linear Quadratic Games with Improved Sample Complexity. (arXiv:2309.04272v1 [eess.SY])

    [http://arxiv.org/abs/2309.04272](http://arxiv.org/abs/2309.04272)

    这项研究提出了改进样本复杂性的零和线性二次博弈，并发现了自然策略梯度方法的隐式正则化属性。在无模型参数知识的情况下，他们还提出了第一个多项式样本复杂性算法来达到Nash均衡。

    

    零和线性二次（LQ）博弈在最优控制中是基础性的，可以用于（i）风险敏感或鲁棒控制的动态博弈形式，或者（ii）作为连续状态-控制空间中两个竞争智能体的多智能体强化学习的基准设置。与广泛研究的单智能体线性二次调节器问题不同，零和LQ博弈涉及解决一个具有缺乏强制性的目标函数的具有挑战性的非凸非凹最小-最大问题。最近，张等人发现了自然策略梯度方法的隐式正则化属性，这对于安全关键的控制系统非常重要，因为它在学习过程中保持了控制器的鲁棒性。此外，在没有模型参数知识的模型无关设置中，张等人提出了第一个多项式样本复杂性算法，以达到Nash均衡的ε-邻域，同时保持理想的隐式正则化属性。

    Zero-sum Linear Quadratic (LQ) games are fundamental in optimal control and can be used (i) as a dynamic game formulation for risk-sensitive or robust control, or (ii) as a benchmark setting for multi-agent reinforcement learning with two competing agents in continuous state-control spaces. In contrast to the well-studied single-agent linear quadratic regulator problem, zero-sum LQ games entail solving a challenging nonconvex-nonconcave min-max problem with an objective function that lacks coercivity. Recently, Zhang et al. discovered an implicit regularization property of natural policy gradient methods which is crucial for safety-critical control systems since it preserves the robustness of the controller during learning. Moreover, in the model-free setting where the knowledge of model parameters is not available, Zhang et al. proposed the first polynomial sample complexity algorithm to reach an $\epsilon$-neighborhood of the Nash equilibrium while maintaining the desirable implicit 
    
[^9]: 用神经符号深度强化学习方法实现安全自主驾驶策略的研究

    Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach. (arXiv:2307.01316v1 [cs.RO])

    [http://arxiv.org/abs/2307.01316](http://arxiv.org/abs/2307.01316)

    本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。

    

    自主驾驶中的动态驾驶环境和多样化道路使用者的存在给决策造成了巨大的挑战。深度强化学习(DRL)已成为解决这一问题的一种流行方法。然而，由于安全问题的限制，现有的DRL解决方案的应用主要局限于模拟环境，阻碍了它们在现实世界中的部署。为了克服这一局限，本文引入了一种新颖的神经符号无模型深度强化学习方法，称为带有符号逻辑的DRL(DRLSL)，它将DRL(从经验中学习)和符号一阶逻辑知识驱动的推理相结合，以实现在实际环境下安全学习自主驾驶的实时交互。这种创新的方法提供了一种通过积极与物理环境互动来学习自主驾驶政策并确保安全性的方式。我们使用高维度数据实现了自主驾驶的DRLSL框架。

    The dynamic nature of driving environments and the presence of diverse road users pose significant challenges for decision-making in autonomous driving. Deep reinforcement learning (DRL) has emerged as a popular approach to tackle this problem. However, the application of existing DRL solutions is mainly confined to simulated environments due to safety concerns, impeding their deployment in real-world. To overcome this limitation, this paper introduces a novel neuro-symbolic model-free DRL approach, called DRL with Symbolic Logics (DRLSL) that combines the strengths of DRL (learning from experience) and symbolic first-order logics knowledge-driven reasoning) to enable safe learning in real-time interactions of autonomous driving within real environments. This innovative approach provides a means to learn autonomous driving policies by actively engaging with the physical environment while ensuring safety. We have implemented the DRLSL framework in autonomous driving using the highD data
    
[^10]: 负反馈训练：提高NVCiM DNN加速器鲁棒性的新概念

    Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators. (arXiv:2305.14561v1 [cs.LG])

    [http://arxiv.org/abs/2305.14561](http://arxiv.org/abs/2305.14561)

    本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。

    

    利用非挥发性存储器(NVM)实现的内存计算(CiM)为加速深度神经网络(DNNs)提供了一种高效的方法。 CiM加速器通过在同一电路板结构中存储网络权重和执行矩阵操作，以最小的面积需求和异常的能效，提供DNN推理加速。然而，NVM设备的随机性和内在变化往往导致性能降低，如与预期结果相比减少分类精度。尽管提出了几种方法来减轻设备变异并增强鲁棒性，但大多数方法都依赖于整体调节并缺乏对训练过程的限制。受到负反馈机制的启发，我们引入了一种新的训练方法，使用多出口机制作为负反馈，在设备变异的情况下增强DNN模型的性能。

    Compute-in-Memory (CiM) utilizing non-volatile memory (NVM) devices presents a highly promising and efficient approach for accelerating deep neural networks (DNNs). By concurrently storing network weights and performing matrix operations within the same crossbar structure, CiM accelerators offer DNN inference acceleration with minimal area requirements and exceptional energy efficiency. However, the stochasticity and intrinsic variations of NVM devices often lead to performance degradation, such as reduced classification accuracy, compared to expected outcomes. Although several methods have been proposed to mitigate device variation and enhance robustness, most of them rely on overall modulation and lack constraints on the training process. Drawing inspiration from the negative feedback mechanism, we introduce a novel training approach that uses a multi-exit mechanism as negative feedback to enhance the performance of DNN models in the presence of device variation. Our negative feedbac
    
[^11]: 核岭回归推断

    Kernel Ridge Regression Inference. (arXiv:2302.06578v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.06578](http://arxiv.org/abs/2302.06578)

    我们提供了核岭回归方法的一致推断和置信带，为广泛应用于各种数据类型的非参数回归估计器提供了准确的统计推断方法。

    

    我们提供了核岭回归(KRR)的一致推断和置信带，这是一种广泛应用于包括排名、图像和图表在内的一般数据类型的非参数回归估计器。尽管这些数据的普遍存在，如学校分配中的排序优先级列表，但KRR的推断理论尚未完全知悉，限制了它在经济学和其他科学领域中的作用。我们构建了针对一般回归器的尖锐、一致的置信区间。为了进行推断，我们开发了一种有效的自举程序，通过对称化来消除偏差并限制计算开销。为了证明该程序，我们推导了再生核希尔伯特空间(RKHS)中部分和的有限样本、均匀高斯和自举耦合。这些推导暗示了基于RKHS单位球的经验过程的强逼近，对覆盖数具有对数依赖关系。模拟验证了置信度。

    We provide uniform inference and confidence bands for kernel ridge regression (KRR), a widely-used non-parametric regression estimator for general data types including rankings, images, and graphs. Despite the prevalence of these data -e.g., ranked preference lists in school assignment -- the inferential theory of KRR is not fully known, limiting its role in economics and other scientific domains. We construct sharp, uniform confidence sets for KRR, which shrink at nearly the minimax rate, for general regressors. To conduct inference, we develop an efficient bootstrap procedure that uses symmetrization to cancel bias and limit computational overhead. To justify the procedure, we derive finite-sample, uniform Gaussian and bootstrap couplings for partial sums in a reproducing kernel Hilbert space (RKHS). These imply strong approximation for empirical processes indexed by the RKHS unit ball with logarithmic dependence on the covering number. Simulations verify coverage. We use our proce
    
[^12]: 对基于Transformer的自监督模型在语音处理中进行压缩

    Compressing Transformer-based self-supervised models for speech processing. (arXiv:2211.09949v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09949](http://arxiv.org/abs/2211.09949)

    本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。

    

    尽管Transformer在自监督学习中取得了成功，并应用于各种下游任务，但是训练和推断的计算成本仍然是将这些模型应用于各种设备的主要挑战。目前已有一些孤立的尝试来压缩Transformer，但研究中的设置和指标各不相同。此前的工作很少涉及不同压缩率之间的权衡，这使得比较压缩技术变得困难。在这项工作中，我们旨在为这些孤立结果提供背景，研究几种常用的压缩技术，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。我们报告了在不同压缩率下的权衡，包括墙钟时间、参数数量和乘加操作数量。我们的结果表明，与最近的方法相比，基本的压缩技术是强大的基准。我们进一步提出了几种压缩方法来改进模型的压缩效果。

    Despite the success of Transformers in self- supervised learning with applications to various downstream tasks, the computational cost of training and inference remains a major challenge for applying these models to a wide spectrum of devices. Several isolated attempts have been made to compress Transformers, but the settings and metrics are different across studies. Trade-off at various compression rates are also largely missing in prior work, making it difficult to compare compression techniques. In this work, we aim to provide context for the isolated results, studying several commonly used compression techniques, including weight pruning, head pruning, low-rank approximation, and knowledge distillation. We report trade- off at various compression rate, including wall-clock time, the number of parameters, and the number of multiply-accumulate operations. Our results show that compared to recent approaches, basic compression techniques are strong baselines. We further present several
    

