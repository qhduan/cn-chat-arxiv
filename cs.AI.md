# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [From attention to profit: quantitative trading strategy based on transformer](https://arxiv.org/abs/2404.00424) | 该研究介绍了一种基于Transformer的量化交易策略，利用改进的模型架构和情感分析的迁移学习，不仅在捕捉长期依赖关系和建模数据关系方面具有优势，而且能够准确预测未来一段时间内的回报。 |
| [^2] | [TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://arxiv.org/abs/2403.20150) | TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。 |
| [^3] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^4] | [GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure](https://arxiv.org/abs/2311.11319) | GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。 |
| [^5] | [Explainable Anomaly Detection in Images and Videos: A Survey](https://arxiv.org/abs/2302.06670) | 这项研究提供了针对图像和视频的可解释异常检测方法的首次调研，为机器学习学术界和实际应用提供了重要参考。 |
| [^6] | [TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models.](http://arxiv.org/abs/2311.01301) | TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。 |
| [^7] | [A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays.](http://arxiv.org/abs/2310.17176) | 本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。 |
| [^8] | [Large language models can replicate cross-cultural differences in personality.](http://arxiv.org/abs/2310.10679) | 大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。 |
| [^9] | [Unravelling Responsibility for AI.](http://arxiv.org/abs/2308.02608) | 本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。 |
| [^10] | [New Interaction Paradigm for Complex EDA Software Leveraging GPT.](http://arxiv.org/abs/2307.14740) | 本研究通过开发SmartonAI插件，基于GPT和BERT等大型语言模型，提供一种新的交互范式来解决EDA软件中初学者面临的复杂命令结构和高学习曲线问题。 |
| [^11] | [Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach.](http://arxiv.org/abs/2307.01316) | 本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。 |
| [^12] | [Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control.](http://arxiv.org/abs/2307.01312) | 本研究提出了一种基于混合Actor-Critic神经结构的自整定PID控制器，用于四旋翼飞行器的姿态和高度控制，通过强化学习的方法调整PID增益，提高了系统的稳健性和可靠性。 |
| [^13] | [Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators.](http://arxiv.org/abs/2305.14561) | 本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。 |
| [^14] | [Interpretable and Robust AI in EEG Systems: A Survey.](http://arxiv.org/abs/2304.10755) | 这篇论文综述了近年来脑电图系统中可解释和鲁棒的AI技术的发展。其中，作者提出了解释性分类法，详细介绍了鲁棒AI的方法，包括对抗攻击和防御、迁移学习和不确定性建模，并讨论了未来的研究方向和挑战。 |

# 详细

[^1]: 从注意力到利润：基于Transformer的量化交易策略

    From attention to profit: quantitative trading strategy based on transformer

    [https://arxiv.org/abs/2404.00424](https://arxiv.org/abs/2404.00424)

    该研究介绍了一种基于Transformer的量化交易策略，利用改进的模型架构和情感分析的迁移学习，不仅在捕捉长期依赖关系和建模数据关系方面具有优势，而且能够准确预测未来一段时间内的回报。

    

    传统量化交易实践中，应对复杂动态的金融市场一直是个持久挑战。先前的机器学习方法往往难以充分捕捉各种市场变量，经常忽视长期信息并且无法捕捉可能带来利润的基本信号。本文引入了改进的Transformer架构，并设计了一个基于该模型的新型因子。通过从情感分析进行迁移学习，所提出的模型不仅发挥了其原有的长距离依赖捕捉和建模复杂数据关系的优势，而且能够处理具有数值输入的任务，并准确预测未来一段时间内的回报。该研究收集了2010年至2019年中国资本市场4,601只股票的5,000,000多条滚动数据。研究结果证明了该模型在预测股票表现方面的卓越性能。

    arXiv:2404.00424v1 Announce Type: cross  Abstract: In traditional quantitative trading practice, navigating the complicated and dynamic financial market presents a persistent challenge. Former machine learning approaches have struggled to fully capture various market variables, often ignore long-term information and fail to catch up with essential signals that may lead the profit. This paper introduces an enhanced transformer architecture and designs a novel factor based on the model. By transfer learning from sentiment analysis, the proposed model not only exploits its original inherent advantages in capturing long-range dependencies and modelling complex data relationships but is also able to solve tasks with numerical inputs and accurately forecast future returns over a period. This work collects more than 5,000,000 rolling data of 4,601 stocks in the Chinese capital market from 2010 to 2019. The results of this study demonstrated the model's superior performance in predicting stock
    
[^2]: TFB：面向时间序列预测方法全面且公平的基准比较

    TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods

    [https://arxiv.org/abs/2403.20150](https://arxiv.org/abs/2403.20150)

    TFB通过解决数据领域覆盖不足、对传统方法的刻板印象以及不一致、不灵活的流程等问题，推动了时间序列预测方法基准比较的最新技术发展。

    

    时间序列会在经济、交通、健康和能源等不同领域中产生，对未来数值的预测在许多重要应用中起着关键作用。不出所料，许多预测方法被提出。为了确保进展，有必要能够以全面且可靠的方式经验性地研究和比较这些方法。为了实现这一目标，我们提出了TFB，一个自动化的时间序列预测（TSF）方法基准测试。TFB通过解决与数据集、比较方法和评估管道相关的缺点，推动了最新技术的发展：1）数据领域覆盖不足，2）对传统方法的刻板印象，3）不一致和不灵活的流程。为了获得更好的领域覆盖率，我们包括了来自10个不同领域的数据集：交通、电力、能源、环境、自然、经济、股票市场、银行、健康和网络。我们还提供了一个时间序列特性

    arXiv:2403.20150v1 Announce Type: cross  Abstract: Time series are generated in diverse domains such as economic, traffic, health, and energy, where forecasting of future values has numerous important applications. Not surprisingly, many forecasting methods are being proposed. To ensure progress, it is essential to be able to study and compare such methods empirically in a comprehensive and reliable manner. To achieve this, we propose TFB, an automated benchmark for Time Series Forecasting (TSF) methods. TFB advances the state-of-the-art by addressing shortcomings related to datasets, comparison methods, and evaluation pipelines: 1) insufficient coverage of data domains, 2) stereotype bias against traditional methods, and 3) inconsistent and inflexible pipelines. To achieve better domain coverage, we include datasets from 10 different domains: traffic, electricity, energy, the environment, nature, economic, stock markets, banking, health, and the web. We also provide a time series char
    
[^3]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^4]: GeoSAM: 使用稀疏和密集的视觉提示对SAM进行改进，实现自动化的移动基础设施分割

    GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure

    [https://arxiv.org/abs/2311.11319](https://arxiv.org/abs/2311.11319)

    GeoSAM是一个基于SAM的新框架，使用了来自零样本学习和预训练CNN分割模型的视觉提示，提高了地理图像分割的性能。

    

    当应用于自然图像分割时，Segment Anything Model (SAM)已经展现出了令人印象深刻的性能。然而，它在地理图像（如航拍和卫星图像）中面临困难，特别是在分割道路、人行道和人行横道等移动基础设施时。这种较差的性能源于这些对象的窄小特征，它们的纹理融入环境中，以及树木、建筑物、车辆和行人等物体的干扰，这些都可能使模型失去定向产生不准确的分割图。为了解决这些挑战，我们提出了地理SAM（GeoSAM），这是一个基于SAM的新框架，它使用来自零样本学习的密集视觉提示和预训练CNN分割模型的稀疏视觉提示实施了细调策略。所提出的GeoSAM在地理图像分割方面优于现有方法，特别是对于道路基础设施、行人基础设施的分割性能提升了26％、7％和17％。

    The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects, their textures blending into the surroundings, and interference from objects like trees, buildings, vehicles, and pedestrians - all of which can disorient the model to produce inaccurate segmentation maps. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning strategy using the dense visual prompt from zero-shot learning, and the sparse visual prompt from a pre-trained CNN segmentation model. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 26%, 7%, and 17% for road infrastructure, pedestrian infrastructur
    
[^5]: 图像和视频中可解释的异常检测：一项调研

    Explainable Anomaly Detection in Images and Videos: A Survey

    [https://arxiv.org/abs/2302.06670](https://arxiv.org/abs/2302.06670)

    这项研究提供了针对图像和视频的可解释异常检测方法的首次调研，为机器学习学术界和实际应用提供了重要参考。

    

    异常检测和定位视觉数据（包括图像和视频）在机器学习学术界和应用实际场景中具有重要意义。尽管近年来可视异常检测技术迅速发展，但对于这些黑盒模型的解释以及为何可以区分异常的合理解释却十分稀缺。本文首次提供了一项集中于可解释视觉异常检测方法的调研。我们首先介绍了图像级和视频级异常检测的基本背景。然后，作为本调研的主要内容，我们展示了针对图像和视频的可解释异常检测方法的全面和详尽的文献综述。接下来，我们分析了为什么一些可解释异常检测方法可以应用于图像和视频，而另一些则只能应用于一种模态。此外，我们提供了总结

    arXiv:2302.06670v2 Announce Type: replace-cross  Abstract: Anomaly detection and localization of visual data, including images and videos, are of great significance in both machine learning academia and applied real-world scenarios. Despite the rapid development of visual anomaly detection techniques in recent years, the interpretations of these black-box models and reasonable explanations of why anomalies can be distinguished out are scarce. This paper provides the first survey concentrated on explainable visual anomaly detection methods. We first introduce the basic background of image-level and video-level anomaly detection. Then, as the main content of this survey, a comprehensive and exhaustive literature review of explainable anomaly detection methods for both images and videos is presented. Next, we analyze why some explainable anomaly detection methods can be applied to both images and videos and why others can be only applied to one modality. Additionally, we provide summaries
    
[^6]: TRIALSCOPE：一个统一的因果框架，用于利用生物医学语言模型扩展实际世界证据生成

    TRIALSCOPE A Unifying Causal Framework for Scaling Real-World Evidence Generation with Biomedical Language Models. (arXiv:2311.01301v1 [cs.LG])

    [http://arxiv.org/abs/2311.01301](http://arxiv.org/abs/2311.01301)

    TRIALSCOPE是一个统一的框架，利用生物医学语言模型将临床文本进行结构化，采用概率建模进行去噪和插补，并应用因果推断技术来应对混杂因素，以从实际世界数据中提取实证证据和推理临床假设。

    

    实际世界数据的快速数字化为优化医疗服务和加速生物医学发现提供了前所未有的机会。然而，在实践中，这些数据往往以非结构化形式存在，如电子医疗记录中的临床笔记，并且通常受到混杂因素的困扰。本文介绍了TRIALSCOPE，一个用于从人群级观察数据中提取实际世界证据的统一框架。TRIALSCOPE利用生物医学语言模型来扩展规模化的临床文本，采用先进的概率建模进行去噪和插补，并结合最先进的因果推断技术来应对常见的混杂因素。利用临床试验规范作为通用表示形式，TRIALSCOPE提供了一个一键式解决方案，可使用观察数据生成和推理临床假设。在一个包含超过一百万个癌症患者的大规模实际世界数据集上进行了广泛的实验和分析。

    The rapid digitization of real-world data offers an unprecedented opportunity for optimizing healthcare delivery and accelerating biomedical discovery. In practice, however, such data is most abundantly available in unstructured forms, such as clinical notes in electronic medical records (EMRs), and it is generally plagued by confounders. In this paper, we present TRIALSCOPE, a unifying framework for distilling real-world evidence from population-level observational data. TRIALSCOPE leverages biomedical language models to structure clinical text at scale, employs advanced probabilistic modeling for denoising and imputation, and incorporates state-of-the-art causal inference techniques to combat common confounders. Using clinical trial specification as generic representation, TRIALSCOPE provides a turn-key solution to generate and reason with clinical hypotheses using observational data. In extensive experiments and analyses on a large-scale real-world dataset with over one million canc
    
[^7]: 从全景X射线中进行牙齿分割和定位的深度学习方法

    A Deep Learning Approach to Teeth Segmentation and Orientation from Panoramic X-rays. (arXiv:2310.17176v1 [cs.CV])

    [http://arxiv.org/abs/2310.17176](http://arxiv.org/abs/2310.17176)

    本研究提出了一个利用深度学习技术从全景X射线图像中进行牙齿分割和定位的方法。我们通过修改已有模型并引入注意力机制，实现了高精度和高性能的牙齿分割和定位。在公开数据集上的评估结果表明，我们的方法在牙齿实例分割和牙齿定位方面取得了优异的性能。

    

    准确的牙齿分割和定位在现代口腔保健中是基础，可实现精确诊断、治疗计划和牙齿种植设计。本研究提出了一种综合的方法，利用深度学习技术从全景X射线图像中进行牙齿分割和定位。我们根据FUSegNet构建了我们的模型，这是一种最初用于创面分割的流行模型，并通过将基于网格的注意力门引入跳跃连接进行了修改。我们通过主成分分析（PCA）引入定向边界框（OBB）生成，以实现精确的牙齿定位估计。在公开可获得的DNS数据集上评估我们的方法，该数据集包括543个全景X射线图像，我们在牙齿实例分割中得到了最高的交并比（IoU）得分82.43%，Dice相似系数（DSC）得分90.37%，在OBB分析中，我们获得了旋转的交并比（RIoU）得分82.82%。

    Accurate teeth segmentation and orientation are fundamental in modern oral healthcare, enabling precise diagnosis, treatment planning, and dental implant design. In this study, we present a comprehensive approach to teeth segmentation and orientation from panoramic X-ray images, leveraging deep learning techniques. We build our model based on FUSegNet, a popular model originally developed for wound segmentation, and introduce modifications by incorporating grid-based attention gates into the skip connections. We introduce oriented bounding box (OBB) generation through principal component analysis (PCA) for precise tooth orientation estimation. Evaluating our approach on the publicly available DNS dataset, comprising 543 panoramic X-ray images, we achieve the highest Intersection-over-Union (IoU) score of 82.43% and Dice Similarity Coefficient (DSC) score of 90.37% among compared models in teeth instance segmentation. In OBB analysis, we obtain the Rotated IoU (RIoU) score of 82.82%. We
    
[^8]: 大型语言模型可以复制跨文化个性差异

    Large language models can replicate cross-cultural differences in personality. (arXiv:2310.10679v1 [cs.CL])

    [http://arxiv.org/abs/2310.10679](http://arxiv.org/abs/2310.10679)

    大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。

    

    我们使用一项大规模实验(N=8000)来确定GPT-4是否可以复制使用十项人格问卷测量的大五人格的跨文化差异。我们选择美国和韩国作为文化对比，因为先前的研究表明这两个国家的人之间存在显著的人格差异。我们操纵了模拟的目标（美国 vs. 韩国），问卷的语言（英语 vs. 韩语）以及语言模型（GPT-4 vs. GPT-3.5）。我们的结果表明，GPT-4复制了每个因子的跨文化差异。然而，平均评级具有上升偏差，并且比人类样本的变异性更低，以及结构效度较低。总的来说，我们提供了初步的证据说明LLMs可以促进跨文化心理研究。

    We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. Overall, we provide preliminary evidence that LLMs can aid cross-cultural psychological research.
    
[^9]: 解构人工智能责任

    Unravelling Responsibility for AI. (arXiv:2308.02608v1 [cs.AI])

    [http://arxiv.org/abs/2308.02608](http://arxiv.org/abs/2308.02608)

    本文旨在解构人工智能责任的概念，提出了一种包含四种责任意义的有效组合，以支持对人工智能责任的实践推理。

    

    为了在涉及人工智能系统的复杂情况下合理思考责任应该放在何处，我们首先需要一个足够清晰和详细的跨学科词汇来谈论责任。责任是一种三元关系，涉及到一个行为者、一个事件和一种责任方式。作为一种有意识的为了支持对人工智能责任进行实践推理的“解构”责任概念的努力，本文采取了“行为者A对事件O负责”的三部分表述，并确定了A、负责、O的子类别的有效组合。这些有效组合我们称之为“责任串”，分为四种责任意义：角色责任、因果责任、法律责任和道德责任。我们通过两个运行示例进行了说明，一个涉及医疗AI系统，另一个涉及AV与行人的致命碰撞。

    To reason about where responsibility does and should lie in complex situations involving AI-enabled systems, we first need a sufficiently clear and detailed cross-disciplinary vocabulary for talking about responsibility. Responsibility is a triadic relation involving an actor, an occurrence, and a way of being responsible. As part of a conscious effort towards 'unravelling' the concept of responsibility to support practical reasoning about responsibility for AI, this paper takes the three-part formulation, 'Actor A is responsible for Occurrence O' and identifies valid combinations of subcategories of A, is responsible for, and O. These valid combinations - which we term "responsibility strings" - are grouped into four senses of responsibility: role-responsibility; causal responsibility; legal liability-responsibility; and moral responsibility. They are illustrated with two running examples, one involving a healthcare AI-based system and another the fatal collision of an AV with a pedes
    
[^10]: 基于GPT的复杂EDA软件新交互范式

    New Interaction Paradigm for Complex EDA Software Leveraging GPT. (arXiv:2307.14740v1 [cs.SE])

    [http://arxiv.org/abs/2307.14740](http://arxiv.org/abs/2307.14740)

    本研究通过开发SmartonAI插件，基于GPT和BERT等大型语言模型，提供一种新的交互范式来解决EDA软件中初学者面临的复杂命令结构和高学习曲线问题。

    

    在电子设计自动化（EDA）领域中，专业软件如KiCad、Cadence和Altium Designer提供越来越广泛的设计功能。然而，复杂的命令结构和较高的学习曲线对于初学者的印刷电路板（PCB）设计师来说造成了障碍。这导致难以选择适合不同设计目的的功能或插件，并且传统文档、视频和在线论坛之外缺乏直观的学习方法。为解决这一挑战，本研究开发了一个名为SmartonAI的EDA软件人工智能交互辅助插件，其中以KiCad作为第一个示例。SmartonAI受到HuggingGPT框架的启发，采用了GPT和BERT等大型语言模型，以促进任务规划和执行。当接收到设计师的请求时，SmartonAI会进行任务分解并高效执行相关的子任务，

    In the rapidly growing field of electronic design automation (EDA), professional software such as KiCad, Cadence , and Altium Designer provide increasingly extensive design functionalities. However, the intricate command structure and high learning curve create a barrier, particularly for novice printed circuit board (PCB) designers. This results in difficulties in selecting appropriate functions or plugins for varying design purposes, compounded by the lack of intuitive learning methods beyond traditional documentation, videos, and online forums. To address this challenge, an artificial intelligence (AI) interaction assist plugin for EDA software named SmartonAl is developed here, also KiCad is taken as the first example. SmartonAI is inspired by the HuggingGPT framework and employs large language models, such as GPT and BERT, to facilitate task planning and execution. On receiving a designer request, SmartonAI conducts a task breakdown and efficiently executes relevant subtasks, such
    
[^11]: 用神经符号深度强化学习方法实现安全自主驾驶策略的研究

    Towards Safe Autonomous Driving Policies using a Neuro-Symbolic Deep Reinforcement Learning Approach. (arXiv:2307.01316v1 [cs.RO])

    [http://arxiv.org/abs/2307.01316](http://arxiv.org/abs/2307.01316)

    本文介绍了一种名为DRL with Symbolic Logics (DRLSL)的新颖神经符号无模型深度强化学习方法，旨在实现在真实环境中安全学习自主驾驶策略。该方法结合了深度强化学习和符号逻辑驱动的推理，允许通过与物理环境的实时交互来学习自主驾驶策略并确保安全性。

    

    自主驾驶中的动态驾驶环境和多样化道路使用者的存在给决策造成了巨大的挑战。深度强化学习(DRL)已成为解决这一问题的一种流行方法。然而，由于安全问题的限制，现有的DRL解决方案的应用主要局限于模拟环境，阻碍了它们在现实世界中的部署。为了克服这一局限，本文引入了一种新颖的神经符号无模型深度强化学习方法，称为带有符号逻辑的DRL(DRLSL)，它将DRL(从经验中学习)和符号一阶逻辑知识驱动的推理相结合，以实现在实际环境下安全学习自主驾驶的实时交互。这种创新的方法提供了一种通过积极与物理环境互动来学习自主驾驶政策并确保安全性的方式。我们使用高维度数据实现了自主驾驶的DRLSL框架。

    The dynamic nature of driving environments and the presence of diverse road users pose significant challenges for decision-making in autonomous driving. Deep reinforcement learning (DRL) has emerged as a popular approach to tackle this problem. However, the application of existing DRL solutions is mainly confined to simulated environments due to safety concerns, impeding their deployment in real-world. To overcome this limitation, this paper introduces a novel neuro-symbolic model-free DRL approach, called DRL with Symbolic Logics (DRLSL) that combines the strengths of DRL (learning from experience) and symbolic first-order logics knowledge-driven reasoning) to enable safe learning in real-time interactions of autonomous driving within real environments. This innovative approach provides a means to learn autonomous driving policies by actively engaging with the physical environment while ensuring safety. We have implemented the DRLSL framework in autonomous driving using the highD data
    
[^12]: 通过基于混合Actor-Critic神经结构的自整定PID控制器，实现四旋翼飞行器控制

    Self-Tuning PID Control via a Hybrid Actor-Critic-Based Neural Structure for Quadcopter Control. (arXiv:2307.01312v1 [eess.SY])

    [http://arxiv.org/abs/2307.01312](http://arxiv.org/abs/2307.01312)

    本研究提出了一种基于混合Actor-Critic神经结构的自整定PID控制器，用于四旋翼飞行器的姿态和高度控制，通过强化学习的方法调整PID增益，提高了系统的稳健性和可靠性。

    

    比例积分微分（PID）控制器被广泛应用于工业和实验过程中，现有的离线方法可以用于调整PID增益。然而，由于模型参数的不确定性和外部干扰的存在，实际系统（如四旋翼飞行器）需要更稳健可靠的PID控制器。本研究探讨了一种使用强化学习的神经网络来实现四旋翼飞行器姿态和高度控制的自整定PID控制器。采用了增量式PID控制器，并仅对可变增益进行了调整。为了调整动态增益，使用了一种基于模型的无模型Actor-Critic混合神经结构，能够适当调整PID增益，同时充当最佳识别器。在调整和识别任务中，使用了一个具有两个隐藏层和Sigmoid激活函数的神经网络，并利用自适应动量（ADAM）优化器和反向传播算法进行学习。

    Proportional-Integrator-Derivative (PID) controller is used in a wide range of industrial and experimental processes. There are a couple of offline methods for tuning PID gains. However, due to the uncertainty of model parameters and external disturbances, real systems such as Quadrotors need more robust and reliable PID controllers. In this research, a self-tuning PID controller using a Reinforcement-Learning-based Neural Network for attitude and altitude control of a Quadrotor has been investigated. An Incremental PID, which contains static and dynamic gains, has been considered and only the variable gains have been tuned. To tune dynamic gains, a model-free actor-critic-based hybrid neural structure was used that was able to properly tune PID gains, and also has done the best as an identifier. In both tunning and identification tasks, a Neural Network with two hidden layers and sigmoid activation functions has been learned using Adaptive Momentum (ADAM) optimizer and Back-Propagatio
    
[^13]: 负反馈训练：提高NVCiM DNN加速器鲁棒性的新概念

    Negative Feedback Training: A Novel Concept to Improve Robustness of NVCiM DNN Accelerators. (arXiv:2305.14561v1 [cs.LG])

    [http://arxiv.org/abs/2305.14561](http://arxiv.org/abs/2305.14561)

    本文介绍了一种新的训练方法，使用负反馈机制来增强DNN模型的鲁棒性，特别是在存在设备变异的情况下。

    

    利用非挥发性存储器(NVM)实现的内存计算(CiM)为加速深度神经网络(DNNs)提供了一种高效的方法。 CiM加速器通过在同一电路板结构中存储网络权重和执行矩阵操作，以最小的面积需求和异常的能效，提供DNN推理加速。然而，NVM设备的随机性和内在变化往往导致性能降低，如与预期结果相比减少分类精度。尽管提出了几种方法来减轻设备变异并增强鲁棒性，但大多数方法都依赖于整体调节并缺乏对训练过程的限制。受到负反馈机制的启发，我们引入了一种新的训练方法，使用多出口机制作为负反馈，在设备变异的情况下增强DNN模型的性能。

    Compute-in-Memory (CiM) utilizing non-volatile memory (NVM) devices presents a highly promising and efficient approach for accelerating deep neural networks (DNNs). By concurrently storing network weights and performing matrix operations within the same crossbar structure, CiM accelerators offer DNN inference acceleration with minimal area requirements and exceptional energy efficiency. However, the stochasticity and intrinsic variations of NVM devices often lead to performance degradation, such as reduced classification accuracy, compared to expected outcomes. Although several methods have been proposed to mitigate device variation and enhance robustness, most of them rely on overall modulation and lack constraints on the training process. Drawing inspiration from the negative feedback mechanism, we introduce a novel training approach that uses a multi-exit mechanism as negative feedback to enhance the performance of DNN models in the presence of device variation. Our negative feedbac
    
[^14]: 可解释和鲁棒的脑电图AI系统综述

    Interpretable and Robust AI in EEG Systems: A Survey. (arXiv:2304.10755v1 [eess.SP])

    [http://arxiv.org/abs/2304.10755](http://arxiv.org/abs/2304.10755)

    这篇论文综述了近年来脑电图系统中可解释和鲁棒的AI技术的发展。其中，作者提出了解释性分类法，详细介绍了鲁棒AI的方法，包括对抗攻击和防御、迁移学习和不确定性建模，并讨论了未来的研究方向和挑战。

    

    在人工智能时代，人工智能（AI）和脑电图（EEG）的密切耦合极大地推动了人机交互（HCI）技术的发展。相较于传统的EEG系统，基于AI的EEG系统的可解释性和鲁棒性变得尤为关键。可解释性能够阐释AI模型的内部工作机制，因此可以获得用户的信任。鲁棒性则反映了AI对抗攻击和扰动的可靠性，这对于敏感和脆弱的EEG信号来说是至关重要的。因此，EEG系统中AI的可解释性和鲁棒性受到越来越多的关注，并且最近的研究取得了巨大进展。然而，关于这一领域的最新进展仍然没有综述。本文首先提出了一种解释性分类法，通过特征化模型、数据和输出解释性，总结了脑电图系统中解释性和鲁棒的AI技术，并详细介绍了鲁棒AI的方法，包括对抗攻击和防御、迁移学习和不确定性建模。最后，我们讨论了这一领域未来的方向和面临的挑战。

    The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it 
    

