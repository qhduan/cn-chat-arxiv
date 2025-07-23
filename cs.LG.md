# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Differential Algebraic Equations](https://arxiv.org/abs/2403.12938) | 提出了神经微分代数方程（NDAEs）用于数据驱动的建模，展示了其在系统理论数据驱动建模任务中的适用性，并通过具体示例表明了其在噪声和外部干扰下的鲁棒性。 |
| [^2] | [Quantifying and Mitigating Privacy Risks for Tabular Generative Models](https://arxiv.org/abs/2403.07842) | 该论文研究了量化和减轻表格生成模型的隐私风险，通过对五种最先进的表格合成器进行实证分析，提出了差分隐私表格潜在扩散模型。 |
| [^3] | [Practical Insights into Knowledge Distillation for Pre-Trained Models](https://arxiv.org/abs/2402.14922) | 研究对知识蒸馏在预训练模型中的应用进行了深入比较，包括优化的温度和权重参数的调整，以及数据分区KD，揭示了最有效的知识蒸馏策略。 |
| [^4] | [Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests](https://arxiv.org/abs/2402.12668) | 随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。 |
| [^5] | [Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science](https://arxiv.org/abs/2402.04247) | 本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。 |
| [^6] | [Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment.](http://arxiv.org/abs/2311.01059) | 本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。 |
| [^7] | [Sample-Driven Federated Learning for Energy-Efficient and Real-Time IoT Sensing.](http://arxiv.org/abs/2310.07497) | 本文提出了一种针对具有实时感知能力的IoT网络设计的基于样本驱动的联邦学习方法，通过控制采样过程来减轻过拟合问题，提高整体准确性，并解决能效问题。 |
| [^8] | [Learning from Data Streams: An Overview and Update.](http://arxiv.org/abs/2212.14720) | 这篇文章探讨了在数据流上的机器学习任务的定义和设置存在的问题，针对这些问题，提出了重新构思监督数据流学习的基本定义和设置，并重新考虑了一些关于数据流学习的基本假设。 |

# 详细

[^1]: 神经微分代数方程

    Neural Differential Algebraic Equations

    [https://arxiv.org/abs/2403.12938](https://arxiv.org/abs/2403.12938)

    提出了神经微分代数方程（NDAEs）用于数据驱动的建模，展示了其在系统理论数据驱动建模任务中的适用性，并通过具体示例表明了其在噪声和外部干扰下的鲁棒性。

    

    微分代数方程（DAEs）描述了符合微分和代数约束的系统的时间演化。特别感兴趣的是包含其组件之间隐性关系（如守恒关系）的系统。在这里，我们提出适用于基于数据的DAE建模的神经微分代数方程（NDAEs）。这一方法建立在通用微分方程的概念之上；即，构建为受特定科学领域理论支持的一组神经常微分方程的模型。在这项工作中，我们展示了所提出的NDAEs抽象适用于相关系统理论数据驱动的建模任务。所示示例包括（i）油箱流形动态的逆问题和（ii）泵、油箱和管道网络的差异建模。我们的实验表明了所提方法对噪声和外部干扰的稳健性。

    arXiv:2403.12938v1 Announce Type: new  Abstract: Differential-Algebraic Equations (DAEs) describe the temporal evolution of systems that obey both differential and algebraic constraints. Of particular interest are systems that contain implicit relationships between their components, such as conservation relationships. Here, we present Neural Differential-Algebraic Equations (NDAEs) suitable for data-driven modeling of DAEs. This methodology is built upon the concept of the Universal Differential Equation; that is, a model constructed as a system of Neural Ordinary Differential Equations informed by theory from particular science domains. In this work, we show that the proposed NDAEs abstraction is suitable for relevant system-theoretic data-driven modeling tasks. Presented examples include (i) the inverse problem of tank-manifold dynamics and (ii) discrepancy modeling of a network of pumps, tanks, and pipes. Our experiments demonstrate the proposed method's robustness to noise and extr
    
[^2]: 量化和减轻表格生成模型的隐私风险

    Quantifying and Mitigating Privacy Risks for Tabular Generative Models

    [https://arxiv.org/abs/2403.07842](https://arxiv.org/abs/2403.07842)

    该论文研究了量化和减轻表格生成模型的隐私风险，通过对五种最先进的表格合成器进行实证分析，提出了差分隐私表格潜在扩散模型。

    

    针对合成数据生成模型出现作为保护隐私的数据共享解决方案的情况，该合成数据集应该类似于原始数据，而不会透露可识别的私人信息。表格合成器的核心技术根植于图像生成模型，范围从生成对抗网络（GAN）到最近的扩散模型。最近的先前工作揭示和量化了表格数据上的效用-隐私权衡，揭示了合成数据的隐私风险。我们首先进行了详尽的实证分析，突出了五种最先进的表格合成器针对八种隐私攻击的效用-隐私权衡，特别关注成员推断攻击。在观察到表格扩散中高数据质量但也高隐私风险的情况下，我们提出了DP-TLDM，差分隐私表格潜在扩散模型，由自动编码器网络组成。

    arXiv:2403.07842v1 Announce Type: new  Abstract: Synthetic data from generative models emerges as the privacy-preserving data-sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. The backbone technology of tabular synthesizers is rooted in image generative models, ranging from Generative Adversarial Networks (GANs) to recent diffusion models. Recent prior work sheds light on the utility-privacy tradeoff on tabular data, revealing and quantifying privacy risks on synthetic data. We first conduct an exhaustive empirical analysis, highlighting the utility-privacy tradeoff of five state-of-the-art tabular synthesizers, against eight privacy attacks, with a special focus on membership inference attacks. Motivated by the observation of high data quality but also high privacy risk in tabular diffusion, we propose DP-TLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder networ
    
[^3]: 针对预训练模型的知识蒸馏的实践见解

    Practical Insights into Knowledge Distillation for Pre-Trained Models

    [https://arxiv.org/abs/2402.14922](https://arxiv.org/abs/2402.14922)

    研究对知识蒸馏在预训练模型中的应用进行了深入比较，包括优化的温度和权重参数的调整，以及数据分区KD，揭示了最有效的知识蒸馏策略。

    

    这项研究探讨了在预训练模型中对知识蒸馏（KD）过程的增强，这是知识传输中一个新兴领域，并对分布式训练和联邦学习环境产生重要影响。尽管采用了许多知识蒸馏方法来在预训练模型之间传递知识，但在这些场景中了解知识蒸馏的应用仍然缺乏全面的理解。我们的研究对多种知识蒸馏技术进行了广泛比较，包括标准KD、经过优化温度和权重参数调整的KD、深度相互学习以及数据分区KD。我们评估这些方法在不同数据分布策略下的表现，以确定每种方法最有效的情境。通过详细研究超参数调整，结合广泛的网格搜索评估来获取信息

    arXiv:2402.14922v1 Announce Type: cross  Abstract: This research investigates the enhancement of knowledge distillation (KD) processes in pre-trained models, an emerging field in knowledge transfer with significant implications for distributed training and federated learning environments. These environments benefit from reduced communication demands and accommodate various model architectures. Despite the adoption of numerous KD approaches for transferring knowledge among pre-trained models, a comprehensive understanding of KD's application in these scenarios is lacking. Our study conducts an extensive comparison of multiple KD techniques, including standard KD, tuned KD (via optimized temperature and weight parameters), deep mutual learning, and data partitioning KD. We assess these methods across various data distribution strategies to identify the most effective contexts for each. Through detailed examination of hyperparameter tuning, informed by extensive grid search evaluations, w
    
[^4]: 随机化既可以减少偏差又可以减少方差：随机森林的案例研究

    Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests

    [https://arxiv.org/abs/2402.12668](https://arxiv.org/abs/2402.12668)

    随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。

    

    我们研究了往往被忽视的现象，首次在\cite{breiman2001random}中指出，即随机森林似乎比装袋法减少了偏差。受\cite{mentch2020randomization}一篇有趣的论文的启发，其中作者认为随机森林减少了有效自由度，并且只有在低信噪比（SNR）环境下才能胜过装袋集成，我们探讨了随机森林如何能够揭示被装袋法忽视的数据模式。我们在实证中证明，在存在这种模式的情况下，随机森林不仅可以减小偏差还能减小方差，并且当信噪比高时随机森林的表现愈发好于装袋集成。我们的观察为解释随机森林在各种信噪比情况下的真实世界成功提供了见解，并增进了我们对随机森林与装袋集成在每次分割注入的随机化方面的差异的理解。我们的调查结果还提供了实用见解。

    arXiv:2402.12668v1 Announce Type: cross  Abstract: We study the often overlooked phenomenon, first noted in \cite{breiman2001random}, that random forests appear to reduce bias compared to bagging. Motivated by an interesting paper by \cite{mentch2020randomization}, where the authors argue that random forests reduce effective degrees of freedom and only outperform bagging ensembles in low signal-to-noise ratio (SNR) settings, we explore how random forests can uncover patterns in the data missed by bagging. We empirically demonstrate that in the presence of such patterns, random forests reduce bias along with variance and increasingly outperform bagging ensembles when SNR is high. Our observations offer insights into the real-world success of random forests across a range of SNRs and enhance our understanding of the difference between random forests and bagging ensembles with respect to the randomization injected into each split. Our investigations also yield practical insights into the 
    
[^5]: 优先安全保障而非自治：科学中LLM智能机器人的风险

    Prioritizing Safeguarding Over Autonomy: Risks of LLM Agents for Science

    [https://arxiv.org/abs/2402.04247](https://arxiv.org/abs/2402.04247)

    本文探讨了科学领域中基于LLM的智能机器人的漏洞与风险，并强调了对安全措施的重要性。

    

    由大型语言模型（LLMs）驱动的智能机器人在各个学科中自主进行实验和促进科学发现方面展示了巨大的前景。尽管它们的能力非常有前途，但也引入了一些新的漏洞，需要仔细考虑安全性。然而，文献中存在显著的空白，尚未对这些漏洞进行全面探讨。本文通过对科学领域中基于LLM的机器人的漏洞进行深入研究，揭示了它们误用可能带来的潜在风险，并强调了对安全措施的需求，填补了这一空白。我们首先全面概述了科学LLM机器人固有的潜在风险，考虑了用户意图、特定的科学领域以及它们对外部环境可能造成的影响。然后，我们深入探讨了这些漏洞的起源和提供的解决方案。

    Intelligent agents powered by large language models (LLMs) have demonstrated substantial promise in autonomously conducting experiments and facilitating scientific discoveries across various disciplines. While their capabilities are promising, they also introduce novel vulnerabilities that demand careful consideration for safety. However, there exists a notable gap in the literature, as there has been no comprehensive exploration of these vulnerabilities. This position paper fills this gap by conducting a thorough examination of vulnerabilities in LLM-based agents within scientific domains, shedding light on potential risks associated with their misuse and emphasizing the need for safety measures. We begin by providing a comprehensive overview of the potential risks inherent to scientific LLM agents, taking into account user intent, the specific scientific domain, and their potential impact on the external environment. Then, we delve into the origins of these vulnerabilities and provid
    
[^6]: 在部署时进行实时调节：用于单机器人部署的行为调控

    Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment. (arXiv:2311.01059v1 [cs.RO])

    [http://arxiv.org/abs/2311.01059](http://arxiv.org/abs/2311.01059)

    本研究提出了一种名为ROAM的方法，通过利用先前学习到的行为来实时调节机器人在部署过程中应对未曾见过的情况。在测试中，ROAM可以在单个阶段内实现快速适应，并且在模拟环境和真实场景中取得了成功，具有较高的效率和适应性。

    

    为了在现实世界中取得成功，机器人必须应对训练过程中未曾见过的情况。本研究探讨了在部署过程中针对这些新场景的实时调节问题，通过利用先前学习到的多样化行为库。我们的方法，RObust Autonomous Modulation（ROAM），引入了基于预训练行为的感知价值的机制，以在特定情况下选择和调整预训练行为。关键是，这种调节过程在测试时的单个阶段内完成，无需任何人类监督。我们对选择机制进行了理论分析，并证明了ROAM使得机器人能够在模拟环境和真实的四足动物Go1上快速适应动态变化，甚至在脚上套着滚轮滑鞋的情况下成功前进。与现有方法相比，我们的方法在面对各种分布情况的部署时能够以超过2倍的效率进行调节，通过有效选择来实现适应。

    To succeed in the real world, robots must cope with situations that differ from those seen during training. We study the problem of adapting on-the-fly to such novel scenarios during deployment, by drawing upon a diverse repertoire of previously learned behaviors. Our approach, RObust Autonomous Modulation (ROAM), introduces a mechanism based on the perceived value of pre-trained behaviors to select and adapt pre-trained behaviors to the situation at hand. Crucially, this adaptation process all happens within a single episode at test time, without any human supervision. We provide theoretical analysis of our selection mechanism and demonstrate that ROAM enables a robot to adapt rapidly to changes in dynamics both in simulation and on a real Go1 quadruped, even successfully moving forward with roller skates on its feet. Our approach adapts over 2x as efficiently compared to existing methods when facing a variety of out-of-distribution situations during deployment by effectively choosing
    
[^7]: 基于样本驱动的联邦学习用于能效和实时IoT感知

    Sample-Driven Federated Learning for Energy-Efficient and Real-Time IoT Sensing. (arXiv:2310.07497v1 [cs.LG])

    [http://arxiv.org/abs/2310.07497](http://arxiv.org/abs/2310.07497)

    本文提出了一种针对具有实时感知能力的IoT网络设计的基于样本驱动的联邦学习方法，通过控制采样过程来减轻过拟合问题，提高整体准确性，并解决能效问题。

    

    在联邦学习系统领域，最近的前沿方法在收敛分析中严重依赖于理想条件。特别地，这些方法假设IoT设备上的训练数据具有与全局数据分布相似的属性。然而，在实时感知联邦学习系统中，这种方法无法捕捉到数据特征的全面范围。为了克服这个限制，我们提出了一种针对具有实时感知能力的IoT网络设计的新方法。我们的方法考虑了由用户数据采样过程引起的泛化差距。通过有效地控制这个采样过程，我们可以减轻过拟合问题，并提高整体准确性。特别地，我们首先制定了一个优化问题，利用采样过程同时减少过拟合和最大化准确性。为了达到这个目标，我们的替代优化问题擅长处理能效问题。

    In the domain of Federated Learning (FL) systems, recent cutting-edge methods heavily rely on ideal conditions convergence analysis. Specifically, these approaches assume that the training datasets on IoT devices possess similar attributes to the global data distribution. However, this approach fails to capture the full spectrum of data characteristics in real-time sensing FL systems. In order to overcome this limitation, we suggest a new approach system specifically designed for IoT networks with real-time sensing capabilities. Our approach takes into account the generalization gap due to the user's data sampling process. By effectively controlling this sampling process, we can mitigate the overfitting issue and improve overall accuracy. In particular, We first formulate an optimization problem that harnesses the sampling process to concurrently reduce overfitting while maximizing accuracy. In pursuit of this objective, our surrogate optimization problem is adept at handling energy ef
    
[^8]: 从数据流中学习：概述与更新

    Learning from Data Streams: An Overview and Update. (arXiv:2212.14720v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.14720](http://arxiv.org/abs/2212.14720)

    这篇文章探讨了在数据流上的机器学习任务的定义和设置存在的问题，针对这些问题，提出了重新构思监督数据流学习的基本定义和设置，并重新考虑了一些关于数据流学习的基本假设。

    

    在数据流上的机器学习文献非常广泛且不断增长。然而，关于数据流学习任务的定义假设通常太强，在实践中难以满足或者甚至相互矛盾，尤其在监督学习的背景下。算法的选择和设计基于通常不明确说明的标准，针对不明确定义的问题设置，在不现实的环境中进行测试，并且与更广泛的文献中的相关方法孤立地进行。这对于在这种背景下构思的许多方法产生了真实世界影响的可能性提出了质疑，并且存在传播误导性研究焦点的风险。我们提议通过重新构思监督数据流学习的基本定义和设置，以考虑相关概念漂移和时间依赖的现代思考方式来解决这些问题；同时，我们重新审视了什么构成了监督数据流学习任务，以及重新考虑一些关于数据流学习的基本假设。

    The literature on machine learning in the context of data streams is vast and growing. However, many of the defining assumptions regarding data-stream learning tasks are too strong to hold in practice, or are even contradictory such that they cannot be met in the contexts of supervised learning. Algorithms are chosen and designed based on criteria which are often not clearly stated, for problem settings not clearly defined, tested in unrealistic settings, and/or in isolation from related approaches in the wider literature. This puts into question the potential for real-world impact of many approaches conceived in such contexts, and risks propagating a misguided research focus. We propose to tackle these issues by reformulating the fundamental definitions and settings of supervised data-stream learning with regard to contemporary considerations of concept drift and temporal dependence; and we take a fresh look at what constitutes a supervised data-stream learning task, and a reconsidera
    

