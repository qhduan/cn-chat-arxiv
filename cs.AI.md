# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions](https://rss.arxiv.org/abs/2312.15101) | 本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。 |
| [^2] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^3] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^4] | [Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance](https://arxiv.org/abs/2403.05842) | Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。 |
| [^5] | [Can we forget how we learned? Doxastic redundancy in iterated belief revision](https://arxiv.org/abs/2402.15445) | 在迭代信念修订中，有时候在其他修订存在时会出现信念修订的多余情况，以及给出了导致序列中第一个修订多余的必要和充分条件。 |
| [^6] | [On the $O(\frac{\sqrt{d}}{T^{1/4}})$ Convergence Rate of RMSProp and Its Momentum Extension Measured by $\ell_1$ Norm: Better Dependence on the Dimension](https://arxiv.org/abs/2402.00389) | 这项研究探讨了RMSProp及其动量扩展方法的收敛速度，并发现使用$\ell_1$范数测度时，收敛速度为$O(\frac{\sqrt{d}}{T^{1/4}})$，在维度极大的问题中具有改进依赖性。 |
| [^7] | [End-To-End Planning of Autonomous Driving in Industry and Academia: 2022-2023.](http://arxiv.org/abs/2401.08658) | 这篇论文总结了工业和学术界中的自动驾驶端到端规划方法，包括特斯拉FSD V12、Momenta 2023、Horizon Robotics 2023、Motional RoboTaxi 2022、Woven Planet（丰田）：城市驾驶员和Nvidia，并回顾了最新的学术研究。这篇文章提供了2022-2023年端到端规划的最新结构和快速学习，并适用于初学者和高级研究人员。 |
| [^8] | [OccluTrack: Rethinking Awareness of Occlusion for Enhancing Multiple Pedestrian Tracking.](http://arxiv.org/abs/2309.10360) | 本文提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack，通过引入异常运动抑制机制、姿势导向的再识别模块和全局相关性优化方法，解决了多目标行人跟踪中遮挡导致的问题。 |
| [^9] | [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations.](http://arxiv.org/abs/2305.16326) | 本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。 |
| [^10] | [Can we forget how we learned? Representing states in iterated belief revision}.](http://arxiv.org/abs/2305.09200) | 本文比较了迭代信念修正中的三种状态表示方法，证明了用字典序修订的重写历史是最有效率的，并提供了一个多项式时间算法，用于确定Horn公式是否等价于neg。 |
| [^11] | [NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online.](http://arxiv.org/abs/2303.10430) | 本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。 |
| [^12] | [Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles.](http://arxiv.org/abs/2303.09271) | 本研究提出了一种高效的oracle系统，能够寻找树集成模型预测的最小代价解释，该算法比目前最先进的替代方案的运行表现更好。m-MARCO算法可以计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。 |

# 详细

[^1]: 修复-Con：深度学习模型转换的自动故障定位和修复

    Fix-Con: Automatic Fault Localization and Repair of Deep Learning Model Conversions

    [https://rss.arxiv.org/abs/2312.15101](https://rss.arxiv.org/abs/2312.15101)

    本文提出了一种自动化的故障定位和修复方法Fix-Con，用于在深度学习模型转换过程中修复由转换引入的故障。Fix-Con能够检测和修复模型输入、参数、超参数和模型图方面的故障，提高转换模型的部署和预测正确性。

    

    在不同深度学习框架之间进行模型转换是一种常见的步骤，可以最大程度地增加模型在设备之间的兼容性，并利用可能只在一个深度学习框架中提供的优化功能。然而，这个转换过程可能存在错误，导致转换后的模型无法部署或存在问题，严重降低了其预测的正确性。我们提出了一种自动化的故障定位和修复方法，Fix-Con，在深度学习框架之间进行模型转换时使用。Fix-Con能够检测和修复在转换过程中引入的模型输入、参数、超参数和模型图的故障。Fix-Con使用从调查转换问题中挖掘出的一组故障类型来定位转换模型中潜在的转换故障，并适当修复它们，例如使用源模型的参数替换目标模型的参数。这一过程在数据集中的每个图像上进行迭代执行。

    Converting deep learning models between frameworks is a common step to maximize model compatibility across devices and leverage optimization features that may be exclusively provided in one deep learning framework. However, this conversion process may be riddled with bugs, making the converted models either undeployable or problematic, considerably degrading their prediction correctness.   We propose an automated approach for fault localization and repair, Fix-Con, during model conversion between deep learning frameworks. Fix-Con is capable of detecting and fixing faults introduced in model input, parameters, hyperparameters, and the model graph during conversion.   Fix-Con uses a set of fault types mined from surveying conversion issues raised to localize potential conversion faults in the converted target model, and then repairs them appropriately, e.g. replacing the parameters of the target model with those from the source model. This is done iteratively for every image in the datas
    
[^2]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^3]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^4]: Hufu：一种通过置换等变性对预训练的Transformer进行水印处理的模态不可知水印系统

    Hufu: A Modality-Agnositc Watermarking System for Pre-Trained Transformers via Permutation Equivariance

    [https://arxiv.org/abs/2403.05842](https://arxiv.org/abs/2403.05842)

    Hufu提出了一种适用于预训练Transformer模型的模态不可知水印系统，利用Transformer的置换等变性质，实现了在模型中嵌入水印并保持高保真度。

    

    随着深度学习模型和服务的蓬勃发展，保护宝贵的模型参数免受盗窃已成为一项迫切关注的问题。水印技术被认为是所有权验证的重要工具。然而，当前的水印方案针对不同的模型和任务定制，难以作为集成的知识产权保护服务。我们提出了Hufu，这是一种针对预训练的基于Transformer的模型的模态不可知水印系统，依赖于Transformer的置换等变性质。Hufu通过微调预训练模型在特定置换的一组数据样本上嵌入水印，嵌入的模型基本上包含两组权重 -- 一组用于正常使用，另一组用于水印提取，触发条件是经过置换的输入。置换等变性确保这两组模型权重之间的最小干扰，从而在水印提取时具有高保真度。

    arXiv:2403.05842v1 Announce Type: cross  Abstract: With the blossom of deep learning models and services, it has become an imperative concern to safeguard the valuable model parameters from being stolen. Watermarking is considered an important tool for ownership verification. However, current watermarking schemes are customized for different models and tasks, hard to be integrated as an integrated intellectual protection service. We propose Hufu, a modality-agnostic watermarking system for pre-trained Transformer-based models, relying on the permutation equivariance property of Transformers. Hufu embeds watermark by fine-tuning the pre-trained model on a set of data samples specifically permuted, and the embedded model essentially contains two sets of weights -- one for normal use and the other for watermark extraction which is triggered on permuted inputs. The permutation equivariance ensures minimal interference between these two sets of model weights and thus high fidelity on downst
    
[^5]: 我们能否忘记我们是如何学习的？在迭代信念修订中的信念多余

    Can we forget how we learned? Doxastic redundancy in iterated belief revision

    [https://arxiv.org/abs/2402.15445](https://arxiv.org/abs/2402.15445)

    在迭代信念修订中，有时候在其他修订存在时会出现信念修订的多余情况，以及给出了导致序列中第一个修订多余的必要和充分条件。

    

    信息的获取方式可能变得无关紧要。明显的情况是当某事被多次确认时。在迭代信念修订方面，特定的修订在其他修订存在时可能变得无关紧要。简单的重复是一个例子，但并非唯一的情况。有时，即使没有相等的修订存在，甚至没有暗示它的其他修订，一个修订也会变得多余。给出了词典修订序列中第一个修订多余的一个必要且充分条件。即使只有两个命题修订，该问题的复杂性也是coNP-完全的。在Horn情况下复杂性相同，但只有不受限制的修订数量：在两个修订情况下它变为多项式。词典修订不仅仅因为它们本身是相关的，也因为它们的序列是用于表示迭代修订过程状态的常见机制中最紧凑的。缩短词典修订序列。

    arXiv:2402.15445v1 Announce Type: new  Abstract: How information was acquired may become irrelevant. An obvious case is when something is confirmed many times. In terms of iterated belief revision, a specific revision may become irrelevant in presence of others. Simple repetitions are an example, but not the only case when this happens. Sometimes, a revision becomes redundant even in presence of none equal, or even no else implying it. A necessary and sufficient condition for the redundancy of the first of a sequence of lexicographic revisions is given. The problem is coNP-complete even with two propositional revisions only. Complexity is the same in the Horn case but only with an unbounded number of revisions: it becomes polynomial with two revisions. Lexicographic revisions are not only relevant by themselves, but also because sequences of them are the most compact of the common mechanisms used to represent the state of an iterated revision process. Shortening sequences of lexicograp
    
[^6]: 关于RMSProp及其动量扩展方法的$O(\frac{\sqrt{d}}{T^{1/4}})$收敛速度和对维度的改进依赖性

    On the $O(\frac{\sqrt{d}}{T^{1/4}})$ Convergence Rate of RMSProp and Its Momentum Extension Measured by $\ell_1$ Norm: Better Dependence on the Dimension

    [https://arxiv.org/abs/2402.00389](https://arxiv.org/abs/2402.00389)

    这项研究探讨了RMSProp及其动量扩展方法的收敛速度，并发现使用$\ell_1$范数测度时，收敛速度为$O(\frac{\sqrt{d}}{T^{1/4}})$，在维度极大的问题中具有改进依赖性。

    

    尽管自适应梯度方法在深度学习中被广泛使用，但其收敛速度尚未得到彻底研究，特别是对于其对维度的依赖性。本文考虑了经典的RMSProp及其动量扩展方法，并通过$\ell_1$范数建立了收敛率$\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_1\right]\leq O(\frac{\sqrt{d}}{T^{1/4}})$，无需假设梯度有界，其中$d$是优化变量的维度，$T$是迭代次数。由于对于维度极大的问题，$\|x\|_2\ll\|x\|_1\leq\sqrt{d}\|x\|_2$，因此我们的收敛速度可以类比为SGD的$\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_2\right]\leq O(\frac{1}{T^{1/4}})$，测度为$\ell_1$范数。

    Although adaptive gradient methods have been extensively used in deep learning, their convergence rates have not been thoroughly studied, particularly with respect to their dependence on the dimension. This paper considers the classical RMSProp and its momentum extension and establishes the convergence rate of $\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_1\right]\leq O(\frac{\sqrt{d}}{T^{1/4}})$ measured by $\ell_1$ norm without the bounded gradient assumption, where $d$ is the dimension of the optimization variable and $T$ is the iteration number. Since $\|x\|_2\ll\|x\|_1\leq\sqrt{d}\|x\|_2$ for problems with extremely large $d$, our convergence rate can be considered to be analogous to the $\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_2\right]\leq O(\frac{1}{T^{1/4}})$ one of SGD measured by $\ell_1$ norm.
    
[^7]: 工业和学术界中的自动驾驶端到端规划：2022-2023年的综述

    End-To-End Planning of Autonomous Driving in Industry and Academia: 2022-2023. (arXiv:2401.08658v1 [cs.RO])

    [http://arxiv.org/abs/2401.08658](http://arxiv.org/abs/2401.08658)

    这篇论文总结了工业和学术界中的自动驾驶端到端规划方法，包括特斯拉FSD V12、Momenta 2023、Horizon Robotics 2023、Motional RoboTaxi 2022、Woven Planet（丰田）：城市驾驶员和Nvidia，并回顾了最新的学术研究。这篇文章提供了2022-2023年端到端规划的最新结构和快速学习，并适用于初学者和高级研究人员。

    

    本文旨在对目前在工业和学术界报告的包括详细技术在内的方法进行快速回顾。具体而言，本文回顾了包括特斯拉FSD V12、Momenta 2023、Horizon Robotics 2023、Motional RoboTaxi 2022、Woven Planet（丰田）：城市驾驶员和Nvidia在内的端到端规划。此外，我们还回顾了调查自动驾驶端到端规划的最新学术研究。本文提供了2022-2023年端到端规划的最新结构和快速学习，为初学者提供了入门材料，供其了解工业和学术界中的端到端自动驾驶规划的最新发展，同时也为高级研究人员提供了补充资料。

    This paper aims to provide a quick review of the methods including the technologies in detail that are currently reported in industry and academia. Specifically, this paper reviews the end-to-end planning, including Tesla FSD V12, Momenta 2023, Horizon Robotics 2023, Motional RoboTaxi 2022, Woven Planet (Toyota): Urban Driver, and Nvidia. In addition, we review the state-of-the-art academic studies that investigate end-to-end planning of autonomous driving. This paper provides readers with a concise structure and fast learning of state-of-the-art end-to-end planning for 2022-2023. This article provides a meaningful overview as introductory material for beginners to follow the state-of-the-art end-to-end planning of autonomous driving in industry and academia, as well as supplementary material for advanced researchers.
    
[^8]: OccluTrack: 重新思考增强多目标行人跟踪中对遮挡感知的方法

    OccluTrack: Rethinking Awareness of Occlusion for Enhancing Multiple Pedestrian Tracking. (arXiv:2309.10360v1 [cs.CV])

    [http://arxiv.org/abs/2309.10360](http://arxiv.org/abs/2309.10360)

    本文提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack，通过引入异常运动抑制机制、姿势导向的再识别模块和全局相关性优化方法，解决了多目标行人跟踪中遮挡导致的问题。

    

    多目标行人跟踪在遮挡的情况下面临着挑战。现有方法在遮挡情况下由于不准确的运动估计、外观特征提取和关联而受到影响，导致身份F1得分（IDF1）不够准确，ID切换过多（IDSw），以及关联准确性和召回率（AssA和AssR）不足。我们发现主要原因是部分遮挡引起的异常检测。本文认为明确的运动估计、可靠的外观特征和公平的关联是解决遮挡场景下问题的关键。具体而言，我们提出了一种自适应遮挡感知多目标行人跟踪器OccluTrack。首先，我们将异常运动抑制机制引入到卡尔曼滤波器中，以自适应地检测和抑制部分遮挡引起的异常运动。其次，我们提出了一种姿势导向的再识别模块，用于提取部分遮挡行人的判别性部分特征。最后，我们设计了一个全局相关性优化方法，以提高遮挡场景下的关联准确性。

    Multiple pedestrian tracking faces the challenge of tracking pedestrians in the presence of occlusion. Existing methods suffer from inaccurate motion estimation, appearance feature extraction, and association due to occlusion, leading to inadequate Identification F1-Score (IDF1), excessive ID switches (IDSw), and insufficient association accuracy and recall (AssA and AssR). We found that the main reason is abnormal detections caused by partial occlusion. In this paper, we suggest that the key insight is explicit motion estimation, reliable appearance features, and fair association in occlusion scenes. Specifically, we propose an adaptive occlusion-aware multiple pedestrian tracker, OccluTrack. We first introduce an abnormal motion suppression mechanism into the Kalman Filter to adaptively detect and suppress outlier motions caused by partial occlusion. Second, we propose a pose-guided re-ID module to extract discriminative part features for partially occluded pedestrians. Last, we desi
    
[^9]: 生物医学自然语言处理中的大型语言模型: 基准、基线和建议

    Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations. (arXiv:2305.16326v1 [cs.CL])

    [http://arxiv.org/abs/2305.16326](http://arxiv.org/abs/2305.16326)

    本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。

    

    生物医学文献呈指数级增长，手动筛选和提取知识变得困难。自动从生物医学文献中提取信息的生物医学自然语言处理（BioNLP）技术有助于减轻这种负担。近年来，如GPT-3和GPT-4等大型语言模型（LLMs）因其卓越的性能而受到重视。但是，它们在BioNLP任务中的有效性以及对方法开发和下游用户的影响仍未得到研究。本研究（1）在四个应用程序中在八个BioNLP数据集中建立了GPT-3和GPT-4在零-shot和一-shot设置下的基准表现，包括命名实体识别，关系提取，多标签文档分类和语义相似性和推理；（2）审查了LLMs产生的错误，并将错误分为三种类型：缺失，不一致和不需要的人工内容；（3）提出了使用LLMs的建议。

    Biomedical literature is growing rapidly, making it challenging to curate and extract knowledge manually. Biomedical natural language processing (BioNLP) techniques that can automatically extract information from biomedical literature help alleviate this burden. Recently, large Language Models (LLMs), such as GPT-3 and GPT-4, have gained significant attention for their impressive performance. However, their effectiveness in BioNLP tasks and impact on method development and downstream users remain understudied. This pilot study (1) establishes the baseline performance of GPT-3 and GPT-4 at both zero-shot and one-shot settings in eight BioNLP datasets across four applications: named entity recognition, relation extraction, multi-label document classification, and semantic similarity and reasoning, (2) examines the errors produced by the LLMs and categorized the errors into three types: missingness, inconsistencies, and unwanted artificial content, and (3) provides suggestions for using L
    
[^10]: 我们能否忘记我们的学习方式？比较迭代信念修正中的状态表示(arXiv:2305.09200v1 [cs.AI])

    Can we forget how we learned? Representing states in iterated belief revision}. (arXiv:2305.09200v1 [cs.AI])

    [http://arxiv.org/abs/2305.09200](http://arxiv.org/abs/2305.09200)

    本文比较了迭代信念修正中的三种状态表示方法，证明了用字典序修订的重写历史是最有效率的，并提供了一个多项式时间算法，用于确定Horn公式是否等价于neg。

    

    本文比较了迭代信念修正中三种最常见的状态表示方法：显式表示，按层次表示和按历史表示。前者是模型之间的连通偏序关系，第二种是表示等价类的公式列表，第三种是先前修订的序列。后者取决于修订语义和历史重写，而前者则取决于允许的重写。所有机制都表示所有可能的状态。用字典序修订的重写历史在大小方面比其他考虑的表示方法更有效率。证明了这样一个历史的冗余是一种轻微的重写。在一般情况下，这是一个coNP完全问题，即使在Horn公式的两次修订历史或任意长度的修订历史上，这也是困难的，但在两个Horn公式的历史上，它是多项式的。一个次要的技术结果是一个多项式时间算法，用于确定一个Horn公式是否等价于neg。

    The three most common representations of states in iterated belief revision are compared: explicit, by levels and by history. The first is a connected preorder between models, the second is a list of formulae representing equivalence classes, the third is the sequence of the previous revisions. The latter depends on the revision semantics and on history rewriting, and the latter depends on the allowed rewritings. All mechanisms represent all possible states. A rewritten history of lexicographic revision is more efficient than the other considered representations in terms of size with arbitrary history rewritings. Establishing the redundancy of such a history is a mild rewriting. It is coNP-complete in the general case, and is hard even on histories of two revisions or revisions of arbitrary length of Horn formulae, and is polynomial on histories of two Horn formulae. A minor technical result is a polynomial-time algorithm for establishing whether a Horn formula is equivalent to the neg
    
[^11]: NoisyHate：在人类编写的在线扰动下对内容审核机器学习模型进行基准测试

    NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online. (arXiv:2303.10430v1 [cs.LG])

    [http://arxiv.org/abs/2303.10430](http://arxiv.org/abs/2303.10430)

    本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。

    

    在社交媒体上，具有有害内容的在线文本是一种威胁，可能会引起网络骚扰。尽管许多平台采取了措施，例如基于机器学习的仇恨言论检测系统来减少其影响，但那些有害内容发布者仍然可以通过修改有害词汇的拼写来逃避系统。这些修改后的单词也称为人类编写的文本扰动。许多研究开发了一定的技术来生成对抗样本，以帮助机器学习模型获得识别这些扰动的能力。然而，机器生成的扰动与人类编写的扰动之间仍存在差距。在本文中，我们介绍了一个包含人类编写的在线扰动的基准测试集，用于毒性言论检测模型。我们还招募了一组工人来评估此测试集的质量并删除低质量的样本。同时，为了检查我们的扰动是否可以归一化为其干净版本，我们还创建了一个相关的测试集。

    Online texts with toxic content are a threat in social media that might cause cyber harassment. Although many platforms applied measures, such as machine learning-based hate-speech detection systems, to diminish their effect, those toxic content publishers can still evade the system by modifying the spelling of toxic words. Those modified words are also known as human-written text perturbations. Many research works developed certain techniques to generate adversarial samples to help the machine learning models obtain the ability to recognize those perturbations. However, there is still a gap between those machine-generated perturbations and human-written perturbations. In this paper, we introduce a benchmark test set containing human-written perturbations online for toxic speech detection models. We also recruited a group of workers to evaluate the quality of this test set and dropped low-quality samples. Meanwhile, to check if our perturbation can be normalized to its clean version, w
    
[^12]: 寻找树集成模型预测的最小代价解释

    Finding Minimum-Cost Explanations for Predictions made by Tree Ensembles. (arXiv:2303.09271v1 [cs.LG])

    [http://arxiv.org/abs/2303.09271](http://arxiv.org/abs/2303.09271)

    本研究提出了一种高效的oracle系统，能够寻找树集成模型预测的最小代价解释，该算法比目前最先进的替代方案的运行表现更好。m-MARCO算法可以计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。

    

    当机器学习模型作为关键系统的决策支持时，能够解释为何模型做出特定预测的能力至关重要。提供的解释必须是可证明的，并且最好不包含冗余信息，即最小解释。本文旨在寻找树集成模型预测的解释，这些解释不仅是最小的，而且在成本函数方面也是最小的。为此，我们首先提出了一个高效的“神谕”系统，可以确定解释的正确性，在计算最小解释时超越了当前最先进的替代方案的运行表现数个数量级。其次，我们改编了来自相关工作的叫做MARCO的算法（将其称为m-MARCO），目的是计算每个预测的单个最小解释，并证明相对于枚举所有最小解释的MARCO算法，我们的方法具有两倍的总体加速比。

    The ability to explain why a machine learning model arrives at a particular prediction is crucial when used as decision support by human operators of critical systems. The provided explanations must be provably correct, and preferably without redundant information, called minimal explanations. In this paper, we aim at finding explanations for predictions made by tree ensembles that are not only minimal, but also minimum with respect to a cost function.  To this end, we first present a highly efficient oracle that can determine the correctness of explanations, surpassing the runtime performance of current state-of-the-art alternatives by several orders of magnitude when computing minimal explanations.  Secondly, we adapt an algorithm called MARCO from related works (calling it m-MARCO) for the purpose of computing a single minimum explanation per prediction, and demonstrate an overall speedup factor of two compared to the MARCO algorithm which enumerates all minimal explanations.  Final
    

