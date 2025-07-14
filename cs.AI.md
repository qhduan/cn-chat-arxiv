# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GoalNet: Goal Areas Oriented Pedestrian Trajectory Prediction](https://arxiv.org/abs/2402.19002) | 通过利用场景背景和观察到的轨迹信息，该研究提出了一种基于行人目标区域的轨迹预测神经网络，可以将不确定性限制在几个目标区域内。 |
| [^2] | [Large Language Models in Mental Health Care: a Scoping Review.](http://arxiv.org/abs/2401.02984) | 本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。 |
| [^3] | [SDR-GAIN: A High Real-Time Occluded Pedestrian Pose Completion Method for Autonomous Driving.](http://arxiv.org/abs/2306.03538) | SDR-GAIN是一种用于解决行人姿态中部分遮挡问题的关键点补全方法，它通过对不完整的关键点进行降维，统一特征分布，并使用GAN框架的两种生成模型来完成姿态的补全。该方法的实验表明性能优于基本的GAIN框架。 |

# 详细

[^1]: GoalNet: 面向目标区域的行人轨迹预测

    GoalNet: Goal Areas Oriented Pedestrian Trajectory Prediction

    [https://arxiv.org/abs/2402.19002](https://arxiv.org/abs/2402.19002)

    通过利用场景背景和观察到的轨迹信息，该研究提出了一种基于行人目标区域的轨迹预测神经网络，可以将不确定性限制在几个目标区域内。

    

    预测道路上行人未来的轨迹是自动驾驶中的重要任务。行人轨迹预测受场景路径、行人意图和决策影响，这是一个多模态问题。最近的研究大多使用过去的轨迹来预测各种潜在的未来轨迹分布，这并未考虑场景背景和行人目标。我们提出了一种不直接预测未来轨迹的方法，即首先使用场景背景和观察到的轨迹来预测目标点，然后重复使用目标点来预测未来轨迹。通过利用场景背景和观察到的轨迹信息，我们可以将不确定性限制在几个目标区域内，这些区域代表了行人的“目标”。在本文中，我们提出了GoalNet，一种基于行人目标区域的新轨迹预测神经网络。

    arXiv:2402.19002v1 Announce Type: cross  Abstract: Predicting the future trajectories of pedestrians on the road is an important task for autonomous driving. The pedestrian trajectory prediction is affected by scene paths, pedestrian's intentions and decision-making, which is a multi-modal problem. Most recent studies use past trajectories to predict a variety of potential future trajectory distributions, which do not account for the scene context and pedestrian targets. Instead of predicting the future trajectory directly, we propose to use scene context and observed trajectory to predict the goal points first, and then reuse the goal points to predict the future trajectories. By leveraging the information from scene context and observed trajectory, the uncertainty can be limited to a few target areas, which represent the "goals" of the pedestrians. In this paper, we propose GoalNet, a new trajectory prediction neural network based on the goal areas of a pedestrian. Our network can pr
    
[^2]: 大型语言模型在心理健康护理中的应用：一项综述研究

    Large Language Models in Mental Health Care: a Scoping Review. (arXiv:2401.02984v1 [cs.CL])

    [http://arxiv.org/abs/2401.02984](http://arxiv.org/abs/2401.02984)

    本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。

    

    目的：大型语言模型（LLM）的使用越来越广泛，需要对它们在心理健康护理领域的应用和结果进行全面的综述。本综述研究旨在对LLMs在心理健康护理中的现有发展和应用进行批判性分析，突出它们的成功，并识别这些专业领域中的挑战和限制。材料和方法：2023年11月，在PubMed、Web of Science、Google Scholar、arXiv、medRxiv和PsyArXiv六个数据库中进行了广泛的文献搜索，遵循2020年版的“系统评价和Meta分析的首选报告项目”（PRISMA）指南。最初识别了313篇出版物，按照研究纳入标准，最终选择了34篇出版物进行综述。结果：我们发现了LLMs在心理健康护理中的多种应用，包括诊断、治疗、患者参与增强等。关键挑战和限制方面的发现将被总结和讨论。

    Objective: The growing use of large language models (LLMs) stimulates a need for a comprehensive review of their applications and outcomes in mental health care contexts. This scoping review aims to critically analyze the existing development and applications of LLMs in mental health care, highlighting their successes and identifying their challenges and limitations in these specialized fields. Materials and Methods: A broad literature search was conducted in November 2023 using six databases (PubMed, Web of Science, Google Scholar, arXiv, medRxiv, and PsyArXiv) following the 2020 version of the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. A total of 313 publications were initially identified, and after applying the study inclusion criteria, 34 publications were selected for the final review. Results: We identified diverse applications of LLMs in mental health care, including diagnosis, therapy, patient engagement enhancement, etc. Key challen
    
[^3]: SDR-GAIN：一种用于自动驾驶的高实时遮挡行人姿态完成方法

    SDR-GAIN: A High Real-Time Occluded Pedestrian Pose Completion Method for Autonomous Driving. (arXiv:2306.03538v1 [cs.CV])

    [http://arxiv.org/abs/2306.03538](http://arxiv.org/abs/2306.03538)

    SDR-GAIN是一种用于解决行人姿态中部分遮挡问题的关键点补全方法，它通过对不完整的关键点进行降维，统一特征分布，并使用GAN框架的两种生成模型来完成姿态的补全。该方法的实验表明性能优于基本的GAIN框架。

    

    为了缓解基于人体姿态关键点的行人检测算法中部分遮挡带来的挑战，我们提出了一种称为分离和降维基于生成对抗性补全网络(SDR-GAIN)的新型行人姿势关键点补全方法。首先，我们利用OpenPose在图像中估计行人的姿态。然后，我们对由于遮挡或其他因素而不完整的行人头部和躯干关键点进行维度缩减，以增强特征并进一步统一特征分布。最后，我们引入了基于生成对抗网络(GAN)框架的两种生成模型，这些模型融合了Huber损失、残差结构和L1正则化来生成部分遮挡行人不完整头部和躯干姿态关键点的缺失部分，从而实现了姿态补全。我们在MS COCO和JAAD数据集上的实验表明，SDR-GAIN的性能优于基本的GAIN框架。

    To mitigate the challenges arising from partial occlusion in human pose keypoint based pedestrian detection methods , we present a novel pedestrian pose keypoint completion method called the separation and dimensionality reduction-based generative adversarial imputation networks (SDR-GAIN) . Firstly, we utilize OpenPose to estimate pedestrian poses in images. Then, we isolate the head and torso keypoints of pedestrians with incomplete keypoints due to occlusion or other factors and perform dimensionality reduction to enhance features and further unify feature distribution. Finally, we introduce two generative models based on the generative adversarial networks (GAN) framework, which incorporate Huber loss, residual structure, and L1 regularization to generate missing parts of the incomplete head and torso pose keypoints of partially occluded pedestrians, resulting in pose completion. Our experiments on MS COCO and JAAD datasets demonstrate that SDR-GAIN outperforms basic GAIN framework
    

