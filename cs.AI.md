# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild](https://arxiv.org/abs/2403.14539) | 提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。 |
| [^2] | [Efficient Prompt Tuning of Large Vision-Language Model for Fine-Grained Ship Classification](https://arxiv.org/abs/2403.08271) | 本研究使用大规模预训练的视觉-语言模型，提出了一种高效的提示调整方法，以增强未见船舶类别的分类准确性。 |
| [^3] | [Large language models surpass human experts in predicting neuroscience results](https://arxiv.org/abs/2403.03230) | 大型语言模型通过整合广泛科学文献中的相关发现，能够优于人类专家预测神经科学实验结果，预示着人类与大型语言模型共同进行发现的未来。 |
| [^4] | [Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries](https://arxiv.org/abs/2402.08349) | 本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。 |
| [^5] | [An explainable three dimension framework to uncover learning patterns: A unified look in variable sulci recognition](https://arxiv.org/abs/2309.00903) | 该论文提出了一个针对医学成像中的可解释AI的三维框架，旨在解决神经科学领域中识别大脑沟特征的复杂性问题。 |
| [^6] | [Moving-Horizon Estimators for Hyperbolic and Parabolic PDEs in 1-D.](http://arxiv.org/abs/2401.02516) | 本文介绍了一种用于双曲和抛物型PDE的移动时域估计器，通过PDE反步法将难以解决的观测器PDE转化为可以明确解决的目标观测器PDE，从而实现了在实时环境下消除数值解观测器PDE的需求。 |
| [^7] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |
| [^8] | [Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning.](http://arxiv.org/abs/2305.01834) | 本文提出了一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行了测试，展示其在有限环境中实现自主搜索和覆盖的能力。 |
| [^9] | [DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video.](http://arxiv.org/abs/2303.13397) | 提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。 |

# 详细

[^1]: Object-Centric Domain Randomization用于野外3D形状重建

    Object-Centric Domain Randomization for 3D Shape Reconstruction in the Wild

    [https://arxiv.org/abs/2403.14539](https://arxiv.org/abs/2403.14539)

    提出了ObjectDR，利用对象-centric的域随机化合成单视图3D形状重建中缺乏的配对数据，通过条件生成模型和解耦框架来生成和保留对象轮廓以及广泛变化的数据，从而为培训模型捕捉域不变性几何形状。

    

    单视图3D形状在野外的重建面临的最大挑战之一是来自真实环境中的<3D形状，2D图像>-配对数据的稀缺性。受域随机化引人注目的成就的启发，我们提出了ObjectDR，通过对对象外观和背景的视觉变化进行随机仿真，合成这种配对数据。我们的数据合成框架利用条件生成模型（例如ControlNet）生成符合空间条件（例如2.5D草图）的图像，这些条件可以通过从对象集合（例如Objaverse-XL）的渲染过程获得3D形状。为了模拟多样化的变化同时保留嵌入空间条件中的对象轮廓，我们还引入了一个利用初始对象指导的解耦框架。

    arXiv:2403.14539v1 Announce Type: cross  Abstract: One of the biggest challenges in single-view 3D shape reconstruction in the wild is the scarcity of <3D shape, 2D image>-paired data from real-world environments. Inspired by remarkable achievements via domain randomization, we propose ObjectDR which synthesizes such paired data via a random simulation of visual variations in object appearances and backgrounds. Our data synthesis framework exploits a conditional generative model (e.g., ControlNet) to generate images conforming to spatial conditions such as 2.5D sketches, which are obtainable through a rendering process of 3D shapes from object collections (e.g., Objaverse-XL). To simulate diverse variations while preserving object silhouettes embedded in spatial conditions, we also introduce a disentangled framework which leverages an initial object guidance. After synthesizing a wide range of data, we pre-train a model on them so that it learns to capture a domain-invariant geometry p
    
[^2]: 用于细粒度船舶分类的大规模视觉-语言模型的高效提示调整

    Efficient Prompt Tuning of Large Vision-Language Model for Fine-Grained Ship Classification

    [https://arxiv.org/abs/2403.08271](https://arxiv.org/abs/2403.08271)

    本研究使用大规模预训练的视觉-语言模型，提出了一种高效的提示调整方法，以增强未见船舶类别的分类准确性。

    

    遥感中的细粒度船舶分类 (RS-FGSC) 由于类别之间的高相似性以及有限的标记数据可用性而面临重大挑战，限制了传统监督分类方法的有效性。最近大规模预训练的视觉-语言模型 (VLMs) 在少样本或零样本学习中展现出令人印象深刻的能力，特别是在理解图像内容方面。本研究深入挖掘了VLMs的潜力，以提高未见船舶类别的分类准确性，在由于成本或隐私限制而数据受限的情况下具有重要意义。直接为RS-FGSC微调VLMs通常会遇到过拟合可见类的挑战，导致对未见类的泛化不佳，突出了区分复杂背景和捕捉独特船舶特征的困难。

    arXiv:2403.08271v1 Announce Type: cross  Abstract: Fine-grained ship classification in remote sensing (RS-FGSC) poses a significant challenge due to the high similarity between classes and the limited availability of labeled data, limiting the effectiveness of traditional supervised classification methods. Recent advancements in large pre-trained Vision-Language Models (VLMs) have demonstrated impressive capabilities in few-shot or zero-shot learning, particularly in understanding image content. This study delves into harnessing the potential of VLMs to enhance classification accuracy for unseen ship categories, which holds considerable significance in scenarios with restricted data due to cost or privacy constraints. Directly fine-tuning VLMs for RS-FGSC often encounters the challenge of overfitting the seen classes, resulting in suboptimal generalization to unseen classes, which highlights the difficulty in differentiating complex backgrounds and capturing distinct ship features. To 
    
[^3]: 大型语言模型在预测神经科学结果方面超越人类专家

    Large language models surpass human experts in predicting neuroscience results

    [https://arxiv.org/abs/2403.03230](https://arxiv.org/abs/2403.03230)

    大型语言模型通过整合广泛科学文献中的相关发现，能够优于人类专家预测神经科学实验结果，预示着人类与大型语言模型共同进行发现的未来。

    

    科学发现常常取决于综合几十年的研究，这一任务可能超出人类信息处理能力。大型语言模型（LLMs）提供了一个解决方案。在广泛的科学文献上训练的LLMs可能能够整合嘈杂但相关的发现，以优于人类专家来预测新颖结果。为了评估这种可能性，我们创建了BrainBench，一个前瞻性的基准，用于预测神经科学结果。我们发现LLMs在预测实验结果方面超越了专家。在神经科学文献上调整的一个LLM，BrainGPT表现得更好。与人类专家一样，当LLMs对他们的预测有信心时，他们更有可能是正确的，这预示着未来人类和LLMs将合作进行发现。我们的方法并非特定于神经科学，并且可转移到其他知识密集型事业中。

    arXiv:2403.03230v1 Announce Type: cross  Abstract: Scientific discoveries often hinge on synthesizing decades of research, a task that potentially outstrips human information processing capacities. Large language models (LLMs) offer a solution. LLMs trained on the vast scientific literature could potentially integrate noisy yet interrelated findings to forecast novel results better than human experts. To evaluate this possibility, we created BrainBench, a forward-looking benchmark for predicting neuroscience results. We find that LLMs surpass experts in predicting experimental outcomes. BrainGPT, an LLM we tuned on the neuroscience literature, performed better yet. Like human experts, when LLMs were confident in their predictions, they were more likely to be correct, which presages a future where humans and LLMs team together to make discoveries. Our approach is not neuroscience-specific and is transferable to other knowledge-intensive endeavors.
    
[^4]: 基于真实用户查询评估文本到SQL系统的数据模型鲁棒性

    Evaluating the Data Model Robustness of Text-to-SQL Systems Based on Real User Queries

    [https://arxiv.org/abs/2402.08349](https://arxiv.org/abs/2402.08349)

    本文首次深入评估了在实践中文本到SQL系统的数据模型的鲁棒性，通过基于一个多年的国际项目集中评估，对一个在FIFA World Cup背景下连续运行了9个月的真实部署的FootballDB系统进行了评估。

    

    文本到SQL系统（也称为自然语言到SQL系统）已成为弥合用户能力与基于SQL的数据访问之间差距的越来越流行的解决方案。这些系统将用户的自然语言请求转化为特定数据库的有效SQL语句。最近的基于转换器的语言模型使得文本到SQL系统受益匪浅。然而，虽然这些系统在常常是合成基准数据集上不断取得新的高分，但对于它们在真实世界、现实场景中对不同数据模型的鲁棒性的系统性探索明显缺乏。本文基于一个多年国际项目关于文本到SQL界面的集中评估，提供了对文本到SQL系统在实践中数据模型鲁棒性的首次深度评估。我们的评估基于FootballDB的真实部署，该系统在FIFA World Cup的背景下连续运行了9个月。

    Text-to-SQL systems (also known as NL-to-SQL systems) have become an increasingly popular solution for bridging the gap between user capabilities and SQL-based data access. These systems translate user requests in natural language to valid SQL statements for a specific database. Recent Text-to-SQL systems have benefited from the rapid improvement of transformer-based language models. However, while Text-to-SQL systems that incorporate such models continuously reach new high scores on -- often synthetic -- benchmark datasets, a systematic exploration of their robustness towards different data models in a real-world, realistic scenario is notably missing. This paper provides the first in-depth evaluation of the data model robustness of Text-to-SQL systems in practice based on a multi-year international project focused on Text-to-SQL interfaces. Our evaluation is based on a real-world deployment of FootballDB, a system that was deployed over a 9 month period in the context of the FIFA Wor
    
[^5]: 一种可解释的三维框架揭示学习模式：变量脑沟识别的统一视角

    An explainable three dimension framework to uncover learning patterns: A unified look in variable sulci recognition

    [https://arxiv.org/abs/2309.00903](https://arxiv.org/abs/2309.00903)

    该论文提出了一个针对医学成像中的可解释AI的三维框架，旨在解决神经科学领域中识别大脑沟特征的复杂性问题。

    

    可解释的人工智能在医学成像中至关重要。在挑战性的神经科学领域里，视觉主题在三维空间内表现出高度复杂性。神经科学的应用涉及从MRI中识别大脑沟特征，由于专家之间的标注规程存在差异和大脑复杂的三维功能，我们面临着重大障碍。因此，传统的可解释性方法在有效验证和评估这些网络方面表现不佳。为了解决这个问题，我们首先提出了数学公式，细化了不同计算机视觉任务中解释需求的各种类别，分为自解释、半解释、非解释和基于验证协议可靠性的新模式学习应用。根据这个数学公式，我们提出了一个旨在解释三维的框架。

    arXiv:2309.00903v2 Announce Type: replace-cross  Abstract: Explainable AI is crucial in medical imaging. In the challenging field of neuroscience, visual topics present a high level of complexity, particularly within three-dimensional space. The application of neuroscience, which involves identifying brain sulcal features from MRI, faces significant hurdles due to varying annotation protocols among experts and the intricate three-dimension functionality of the brain. Consequently, traditional explainability approaches fall short in effectively validating and evaluating these networks. To address this, we first present a mathematical formulation delineating various categories of explanation needs across diverse computer vision tasks, categorized into self-explanatory, semi-explanatory, non-explanatory, and new-pattern learning applications based on the reliability of the validation protocol. With respect to this mathematical formulation, we propose a 3D explainability framework aimed at
    
[^6]: 在一维中为双曲和抛物型PDE引入移动时域估计器

    Moving-Horizon Estimators for Hyperbolic and Parabolic PDEs in 1-D. (arXiv:2401.02516v1 [eess.SY])

    [http://arxiv.org/abs/2401.02516](http://arxiv.org/abs/2401.02516)

    本文介绍了一种用于双曲和抛物型PDE的移动时域估计器，通过PDE反步法将难以解决的观测器PDE转化为可以明确解决的目标观测器PDE，从而实现了在实时环境下消除数值解观测器PDE的需求。

    

    对于PDE的观测器本身也是PDE。因此，使用这样的观测器产生实时估计是计算负担很重的。对于有限维和ODE系统，移动时域估计器（MHE）是一种操作符，其输出是状态估计，而输入是时域起始处的初始状态估计以及移动时间域内的测量输出和输入信号。在本文中，我们引入了用于解决PDE的MHE，以消除实时数值解观测器PDE的需求。我们使用PDE反步法实现了这一点，对于某些特定类别的双曲和抛物型PDE，它能够明确地产生移动时域状态估计。具体来说，为了明确地产生状态估计，我们使用了一个难以解决的观测器PDE的反步变换，将其转化为一个可以明确解决的目标观测器PDE。我们提出的MHE并不是新的观测器设计，而只是明确的MHE实现，它能够在移动时域内产生状态估计。

    Observers for PDEs are themselves PDEs. Therefore, producing real time estimates with such observers is computationally burdensome. For both finite-dimensional and ODE systems, moving-horizon estimators (MHE) are operators whose output is the state estimate, while their inputs are the initial state estimate at the beginning of the horizon as well as the measured output and input signals over the moving time horizon. In this paper we introduce MHEs for PDEs which remove the need for a numerical solution of an observer PDE in real time. We accomplish this using the PDE backstepping method which, for certain classes of both hyperbolic and parabolic PDEs, produces moving-horizon state estimates explicitly. Precisely, to explicitly produce the state estimates, we employ a backstepping transformation of a hard-to-solve observer PDE into a target observer PDE, which is explicitly solvable. The MHEs we propose are not new observer designs but simply the explicit MHE realizations, over a moving
    
[^7]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    
[^8]: 基于动态系统路径规划和无监督学习的实时环境自主搜索

    Autonomous search of real-life environments combining dynamical system-based path planning and unsupervised learning. (arXiv:2305.01834v1 [cs.RO])

    [http://arxiv.org/abs/2305.01834](http://arxiv.org/abs/2305.01834)

    本文提出了一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行了测试，展示其在有限环境中实现自主搜索和覆盖的能力。

    

    近年来取得了使用混沌覆盖路径规划器进行有限环境搜索和遍历的进展，但该领域的现状仍处于初级阶段，目前的实验工作尚未开发出可满足混沌覆盖路径规划器需要克服的立即问题的强大方法 。本文旨在提出一种自动生成基于动态系统路径规划器和无监督机器学习技术相结合的算法，以克服混沌覆盖路径规划器的立即问题，并在模拟和实际环境中进行测试，展示其在有限环境中实现自主搜索和覆盖的能力。

    In recent years, advancements have been made towards the goal of using chaotic coverage path planners for autonomous search and traversal of spaces with limited environmental cues. However, the state of this field is still in its infancy as there has been little experimental work done. Current experimental work has not developed robust methods to satisfactorily address the immediate set of problems a chaotic coverage path planner needs to overcome in order to scan realistic environments within reasonable coverage times. These immediate problems are as follows: (1) an obstacle avoidance technique which generally maintains the kinematic efficiency of the robot's motion, (2) a means to spread chaotic trajectories across the environment (especially crucial for large and/or complex-shaped environments) that need to be covered, and (3) a real-time coverage calculation technique that is accurate and independent of cell size. This paper aims to progress the field by proposing algorithms that a
    
[^9]: DDT：一种基于扩散驱动变压器的从视频中恢复人体网格的框架

    DDT: A Diffusion-Driven Transformer-based Framework for Human Mesh Recovery from a Video. (arXiv:2303.13397v1 [cs.CV])

    [http://arxiv.org/abs/2303.13397](http://arxiv.org/abs/2303.13397)

    提出了一种基于扩散驱动变压器的视频 HMR 框架（DDT），它旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性，并输出所有帧的人体网格，使得 DDT 更适用于时间效率至关重要的实际应用。

    

    人体网格恢复（HMR）为各种实际应用提供了丰富的人体信息，例如游戏、人机交互和虚拟现实。与单一图像方法相比，基于视频的方法可以利用时间信息通过融合人体运动先验进一步提高性能。然而，像 VIBE 这样的多对多方法存在运动平滑性和时间一致性的挑战。而像 TCMR 和 MPS-Net 这样的多对一方法则依赖于未来帧，在推理过程中是非因果和时间效率低下的。为了解决这些挑战，提出了一种新的基于扩散驱动变压器的视频 HMR 框架（DDT）。DDT 旨在从输入序列中解码特定的运动模式，增强运动平滑性和时间一致性。作为一种多对多方法，DDT 的解码器输出所有帧的人体网格，使 DDT 更适用于时间效率至关重要的实际应用。

    Human mesh recovery (HMR) provides rich human body information for various real-world applications such as gaming, human-computer interaction, and virtual reality. Compared to single image-based methods, video-based methods can utilize temporal information to further improve performance by incorporating human body motion priors. However, many-to-many approaches such as VIBE suffer from motion smoothness and temporal inconsistency. While many-to-one approaches such as TCMR and MPS-Net rely on the future frames, which is non-causal and time inefficient during inference. To address these challenges, a novel Diffusion-Driven Transformer-based framework (DDT) for video-based HMR is presented. DDT is designed to decode specific motion patterns from the input sequence, enhancing motion smoothness and temporal consistency. As a many-to-many approach, the decoder of our DDT outputs the human mesh of all the frames, making DDT more viable for real-world applications where time efficiency is cruc
    

