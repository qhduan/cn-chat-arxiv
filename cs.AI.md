# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Proprioception Is All You Need: Terrain Classification for Boreal Forests](https://arxiv.org/abs/2403.16877) | 通过引入 BorealTC 数据集，结合现有数据集，我们评估了基于卷积神经网络（CNN）和新颖的状态空间模型（SSM）-Mamba体系结构在北方森林地形分类上的表现。 |
| [^2] | [PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning](https://arxiv.org/abs/2402.12842) | 提出了PromptKD方法，通过提示调整实现了生成语言模型提取学生友好知识的蒸馏，无需微调整整个教师模型。 |
| [^3] | [Computing Voting Rules with Elicited Incomplete Votes](https://arxiv.org/abs/2402.11104) | 本文研究了通过询问选民有关少量候选人的投票规则计算问题，完全表征了可计算的位置评分规则集合，并给出了对于确定性或随机算法确定最大得分候选人必须进行的查询次数的参数化上限和下限。 |
| [^4] | [Fast ODE-based Sampling for Diffusion Models in Around 5 Steps](https://arxiv.org/abs/2312.00094) | 提出了一种基于几何观察的Approximate MEan-Direction Solver（AMED-Solver），能够通过直接学习均方向来消除截断误差，从而实现快速扩散抽样。 |
| [^5] | [A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management.](http://arxiv.org/abs/2401.12393) | 本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。 |
| [^6] | [Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data.](http://arxiv.org/abs/2309.15039) | 该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。 |
| [^7] | [CyberForce: A Federated Reinforcement Learning Framework for Malware Mitigation.](http://arxiv.org/abs/2308.05978) | CyberForce是一个联邦强化学习框架，用于在物联网设备中协同私密地确定适合缓解各种零日攻击的MTD技术。它整合了设备指纹识别和异常检测，并通过奖励或惩罚FRL agent选择的MTD机制来提高网络安全性。 |
| [^8] | [RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models.](http://arxiv.org/abs/2304.10727) | 本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。 |
| [^9] | [A Survey on In-context Learning.](http://arxiv.org/abs/2301.00234) | 本文调查和总结了上下文学习(ICL)的进展和挑战，ICL已成为自然语言处理(NLP)的新范式，探索ICL以评估和推广大型语言模型(LLM)的能力已成为一种新趋势。本文提出了ICL的正式定义，并总结了高级技术，最后讨论了ICL的挑战以及进一步研究的潜在方向。 |

# 详细

[^1]: 感知力就是你所需要的：北方森林的地形分类

    Proprioception Is All You Need: Terrain Classification for Boreal Forests

    [https://arxiv.org/abs/2403.16877](https://arxiv.org/abs/2403.16877)

    通过引入 BorealTC 数据集，结合现有数据集，我们评估了基于卷积神经网络（CNN）和新颖的状态空间模型（SSM）-Mamba体系结构在北方森林地形分类上的表现。

    

    最近的领域机器人学研究强调了抵御不同类型地形的重要性。北方森林特别受到许多限制机动性的地形的影响，这些地形应该在越野自主导航中加以考虑。此外，作为地球上最大的陆地生物群落之一，北方森林是预计自主车辆将日益普及的地区。在本文中，我们通过引入BorealTC来解决这个问题，这是一个用于基于感知力的地形分类（TC）的公开可用数据集。我们的数据集记录了Husky A200的116分钟的惯性测量单元（IMU）、电机电流和轮胎里程数据，重点关注典型的北方森林地形，特别是雪、冰和淤泥壤。结合我们的数据集与另一个来自最新技术的数据集，我们在TC t

    arXiv:2403.16877v1 Announce Type: cross  Abstract: Recent works in field robotics highlighted the importance of resiliency against different types of terrains. Boreal forests, in particular, are home to many mobility-impeding terrains that should be considered for off-road autonomous navigation. Also, being one of the largest land biomes on Earth, boreal forests are an area where autonomous vehicles are expected to become increasingly common. In this paper, we address this issue by introducing BorealTC, a publicly available dataset for proprioceptive-based terrain classification (TC). Recorded with a Husky A200, our dataset contains 116 min of Inertial Measurement Unit (IMU), motor current, and wheel odometry data, focusing on typical boreal forest terrains, notably snow, ice, and silty loam. Combining our dataset with another dataset from the state-of-the-art, we evaluate both a Convolutional Neural Network (CNN) and the novel state space model (SSM)-based Mamba architecture on a TC t
    
[^2]: PromptKD：通过提示调整为生成语言模型提取学生友好知识的蒸馏方法

    PromptKD: Distilling Student-Friendly Knowledge for Generative Language Models via Prompt Tuning

    [https://arxiv.org/abs/2402.12842](https://arxiv.org/abs/2402.12842)

    提出了PromptKD方法，通过提示调整实现了生成语言模型提取学生友好知识的蒸馏，无需微调整整个教师模型。

    

    近期大型语言模型（LLMs）的发展引起了对推理成本的担忧，进一步增加了对模型压缩研究的需求。尽管知识蒸馏（KD）是一种突出的方法，但是针对LLMs这样的生成语言模型的KD研究相对较少，而提取适合学生的知识的方法，在分类模型的KD中表现出了良好性能，在生成语言模型中尚未被探索。为了探索这种方法，我们提出了PromptKD，一种简单而有效的方法，它利用提示调整 - 在KD中首次出现 - 使生成语言模型能够传递适合学生的知识。与先前分类工作不同，先前那些需要微调整整个教师模型以提取适合学生的知识，PromptKD通过添加少量提示标记，并仅通过学生指导调整提示来达到类似效果。

    arXiv:2402.12842v1 Announce Type: cross  Abstract: Recent advancements in large language models (LLMs) have raised concerns about inference costs, increasing the need for research into model compression. While knowledge distillation (KD) is a prominent method for this, research on KD for generative language models like LLMs is relatively sparse, and the approach of distilling student-friendly knowledge, which has shown promising performance in KD for classification models, remains unexplored in generative language models. To explore this approach, we propose PromptKD, a simple yet effective method that utilizes prompt tuning - for the first time in KD - to enable generative language models to transfer student-friendly knowledge. Unlike previous works in classification that require fine-tuning the entire teacher model for extracting student-friendly knowledge, PromptKD achieves similar effects by adding a small number of prompt tokens and tuning only the prompt with student guidance. Ex
    
[^3]: 用询问不完整选票计算投票规则

    Computing Voting Rules with Elicited Incomplete Votes

    [https://arxiv.org/abs/2402.11104](https://arxiv.org/abs/2402.11104)

    本文研究了通过询问选民有关少量候选人的投票规则计算问题，完全表征了可计算的位置评分规则集合，并给出了对于确定性或随机算法确定最大得分候选人必须进行的查询次数的参数化上限和下限。

    

    受到在大型候选人群体中说明完整序数偏好的困难的启发，我们研究了通过询问选民有关 $t < m$ 候选人的投票规则。在推广了关于该问题特定情况的先前研究的基础上，我们的论文完全表征了对于任意 $1 \leq t < m$ 可以计算得出的位置评分规则的集合，值得注意的是其中并不包括多数制。然后，我们扩展这一研究，展示了单次可转移投票（淘汰投票）的类似无法计算的结果。这些负面结果是信息理论的，不关心查询的数量。最后，对于可以使用有限大小查询计算的评分规则，我们给出了参数化的关于确定性或随机算法必须做出的查询数量的上限和下限。虽然我们的确定性算法的界限之间没有差距，但识别

    arXiv:2402.11104v1 Announce Type: cross  Abstract: Motivated by the difficulty of specifying complete ordinal preferences over a large set of $m$ candidates, we study voting rules that are computable by querying voters about $t < m$ candidates. Generalizing prior works that focused on specific instances of this problem, our paper fully characterizes the set of positional scoring rules that can be computed for any $1 \leq t < m$, which notably does not include plurality. We then extend this to show a similar impossibility result for single transferable vote (elimination voting). These negative results are information-theoretic and agnostic to the number of queries. Finally, for scoring rules that are computable with limited-sized queries, we give parameterized upper and lower bounds on the number of such queries a deterministic or randomized algorithm must make to determine the score-maximizing candidate. While there is no gap between our bounds for deterministic algorithms, identifying
    
[^4]: 在大约5个步骤中，用于扩散模型的快速基于ODE的抽样

    Fast ODE-based Sampling for Diffusion Models in Around 5 Steps

    [https://arxiv.org/abs/2312.00094](https://arxiv.org/abs/2312.00094)

    提出了一种基于几何观察的Approximate MEan-Direction Solver（AMED-Solver），能够通过直接学习均方向来消除截断误差，从而实现快速扩散抽样。

    

    从扩散模型中进行抽样可以被视为解决相应的常微分方程（ODE），旨在以尽可能少的函数评估次数（NFE）获得准确解。最近，出现了利用高阶ODE求解器的各种快速抽样器，并且比最初的一阶求解器表现更好。然而，这些数值方法固有地导致某些近似误差，极大地降低了具有极小NFE（例如，约为5）的样本质量。相反，基于几何观察，每个抽样轨迹几乎位于嵌入在环境空间中的二维子空间中，我们提出了用于快速扩散抽样的AME近似均方向求解器（AMED-Solver），通过直接学习均方向来消除截断误差。此外，我们的方法可以轻松作为插件使用，以进一步改进现有的基于ODE的方法。

    arXiv:2312.00094v2 Announce Type: replace-cross  Abstract: Sampling from diffusion models can be treated as solving the corresponding ordinary differential equations (ODEs), with the aim of obtaining an accurate solution with as few number of function evaluations (NFE) as possible. Recently, various fast samplers utilizing higher-order ODE solvers have emerged and achieved better performance than the initial first-order one. However, these numerical methods inherently result in certain approximation errors, which significantly degrades sample quality with extremely small NFE (e.g., around 5). In contrast, based on the geometric observation that each sampling trajectory almost lies in a two-dimensional subspace embedded in the ambient space, we propose Approximate MEan-Direction Solver (AMED-Solver) that eliminates truncation errors by directly learning the mean direction for fast diffusion sampling. Besides, our method can be easily used as a plugin to further improve existing ODE-base
    
[^5]: 基于学习的声明性隐私保护数据联邦管理框架

    A Learning-based Declarative Privacy-Preserving Framework for Federated Data Management. (arXiv:2401.12393v1 [cs.DB])

    [http://arxiv.org/abs/2401.12393](http://arxiv.org/abs/2401.12393)

    本论文提出了一个基于学习的声明性隐私保护框架，通过使用Differentially-Private Stochastic Gradient Descent（DP-SGD）算法训练的深度学习模型替代部分实际数据来回答查询，并允许用户指定要保护的私人信息。此框架还可以自动选择转换计划和超参数，并允许人工专家审核和调整隐私保护机制。

    

    在多个私有数据孤岛上进行联邦查询处理时，平衡隐私和准确性是一项具有挑战性的任务。在这项工作中，我们将演示一种自动化新兴隐私保护技术的端到端工作流，该技术使用使用差分隐私随机梯度下降（DP-SGD）算法训练的深度学习模型替换实际数据的部分来回答查询。我们提出的新颖声明性隐私保护工作流允许用户指定“要保护的私人信息”而不是“如何保护”。在底层，系统自动选择查询-模型转换计划以及超参数。同时，所提出的工作流还允许人工专家审核和调整选择的隐私保护机制，用于审计/合规和优化目的。

    It is challenging to balance the privacy and accuracy for federated query processing over multiple private data silos. In this work, we will demonstrate an end-to-end workflow for automating an emerging privacy-preserving technique that uses a deep learning model trained using the Differentially-Private Stochastic Gradient Descent (DP-SGD) algorithm to replace portions of actual data to answer a query. Our proposed novel declarative privacy-preserving workflow allows users to specify "what private information to protect" rather than "how to protect". Under the hood, the system automatically chooses query-model transformation plans as well as hyper-parameters. At the same time, the proposed workflow also allows human experts to review and tune the selected privacy-preserving mechanism for audit/compliance, and optimization purposes.
    
[^6]: 结合存活分析和机器学习利用电子健康记录数据进行肿瘤风险预测

    Combining Survival Analysis and Machine Learning for Mass Cancer Risk Prediction using EHR data. (arXiv:2309.15039v1 [cs.LG])

    [http://arxiv.org/abs/2309.15039](http://arxiv.org/abs/2309.15039)

    该论文介绍了一种利用 EHR 数据进行大规模肿瘤风险预测的新方法，其创新之处在于只需利用历史的医疗服务代码和诊断信息来实现最小化的数据需求，通过将存活分析和机器学习相结合，可以在大规模应用中实现对患者癌症风险的个性化评估。

    

    纯粹的医学肿瘤筛查方法通常费用高昂、耗时长，并且仅适用于大规模应用。先进的人工智能（AI）方法在癌症检测方面发挥了巨大作用，但需要特定或深入的医学数据。这些方面影响了癌症筛查方法的大规模实施。因此，基于已有的电子健康记录（EHR）数据对患者进行大规模个性化癌症风险评估应用AI方法是一种颠覆性的改变。本文提出了一种利用EHR数据进行大规模肿瘤风险预测的新方法。与其他方法相比，我们的方法通过最小的数据贪婪策略脱颖而出，仅需要来自EHR的医疗服务代码和诊断历史。我们将问题形式化为二分类问题。该数据集包含了175441名不记名的患者（其中2861名被诊断为癌症）。作为基准，我们实现了一个基于循环神经网络（RNN）的解决方案。我们提出了一种方法，将存活分析和机器学习相结合，

    Purely medical cancer screening methods are often costly, time-consuming, and weakly applicable on a large scale. Advanced Artificial Intelligence (AI) methods greatly help cancer detection but require specific or deep medical data. These aspects affect the mass implementation of cancer screening methods. For these reasons, it is a disruptive change for healthcare to apply AI methods for mass personalized assessment of the cancer risk among patients based on the existing Electronic Health Records (EHR) volume.  This paper presents a novel method for mass cancer risk prediction using EHR data. Among other methods, our one stands out by the minimum data greedy policy, requiring only a history of medical service codes and diagnoses from EHR. We formulate the problem as a binary classification. This dataset contains 175 441 de-identified patients (2 861 diagnosed with cancer). As a baseline, we implement a solution based on a recurrent neural network (RNN). We propose a method that combine
    
[^7]: CyberForce: 一个用于恶意软件缓解的联邦强化学习框架

    CyberForce: A Federated Reinforcement Learning Framework for Malware Mitigation. (arXiv:2308.05978v1 [cs.CR])

    [http://arxiv.org/abs/2308.05978](http://arxiv.org/abs/2308.05978)

    CyberForce是一个联邦强化学习框架，用于在物联网设备中协同私密地确定适合缓解各种零日攻击的MTD技术。它整合了设备指纹识别和异常检测，并通过奖励或惩罚FRL agent选择的MTD机制来提高网络安全性。

    

    互联网物联网(IoT)范例的扩展是不可避免的，但是对于IoT设备对恶意软件事件的脆弱性已成为一个越来越关注的问题。最近的研究显示，将强化学习与移动目标防御(MTD)机制相结合，可以增强IoT设备的网络安全性。然而，大量的新恶意软件攻击和代理人学习和选择有效的MTD技术所需的时间使得这种方法在现实世界的IoT场景中不切实际。为解决这个问题，本研究提出了CyberForce，一个采用联邦强化学习(FRL)的框架，用于集体且保密地确定适合缓解各种零日攻击的MTD技术。CyberForce结合了设备指纹识别和异常检测，通过奖励或惩罚FRL agent选择的MTD机制。该框架在一个由十台真实IoT平台设备组成的联邦中进行了评估。通过六个恶意软件样本进行了一系列实验。

    The expansion of the Internet-of-Things (IoT) paradigm is inevitable, but vulnerabilities of IoT devices to malware incidents have become an increasing concern. Recent research has shown that the integration of Reinforcement Learning with Moving Target Defense (MTD) mechanisms can enhance cybersecurity in IoT devices. Nevertheless, the numerous new malware attacks and the time that agents take to learn and select effective MTD techniques make this approach impractical for real-world IoT scenarios. To tackle this issue, this work presents CyberForce, a framework that employs Federated Reinforcement Learning (FRL) to collectively and privately determine suitable MTD techniques for mitigating diverse zero-day attacks. CyberForce integrates device fingerprinting and anomaly detection to reward or penalize MTD mechanisms chosen by an FRL-based agent. The framework has been evaluated in a federation consisting of ten devices of a real IoT platform. A pool of experiments with six malware samp
    
[^8]: RoCOCO：稳健的基准MS-COCO评估图文匹配模型的鲁棒性

    RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models. (arXiv:2304.10727v1 [cs.CV])

    [http://arxiv.org/abs/2304.10727](http://arxiv.org/abs/2304.10727)

    本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。

    

    近年来，大规模的视觉语言预训练模型和视觉语义嵌入方法显著提高了MS COCO 5K测试集上图文匹配（ITM）的准确性。然而，当将这些最先进的模型用于实际应用时，它们的鲁棒性仍不清楚。本文提出了一个新的评估基准来测试ITM模型的鲁棒性。为此，我们将各种“愚弄”的图片和标题添加到检索池中。具体而言，我们通过插入不相关的图像来更改图像，并通过替换名词来更改标题，从而改变句子的含义。我们发现，仅仅将这些新创建的图像和标题添加到测试集中就可以降低各种最先进模型的性能（例如，在BLIP中从81.9％降至64.5％，在VSE∞中从66.1％降至37.5％）。我们希望我们的发现能为提高视觉语言模型的鲁棒性和设计更多样化的压力测试提供启示。

    Recently, large-scale vision-language pre-training models and visual semantic embedding methods have significantly improved image-text matching (ITM) accuracy on MS COCO 5K test set. However, it is unclear how robust these state-of-the-art (SOTA) models are when using them in the wild. In this paper, we propose a novel evaluation benchmark to stress-test the robustness of ITM models. To this end, we add various fooling images and captions to a retrieval pool. Specifically, we change images by inserting unrelated images, and change captions by substituting a noun, which can change the meaning of a sentence. We discover that just adding these newly created images and captions to the test set can degrade performances (i.e., Recall@1) of a wide range of SOTA models (e.g., 81.9% $\rightarrow$ 64.5% in BLIP, 66.1% $\rightarrow$ 37.5% in VSE$\infty$). We expect that our findings can provide insights for improving the robustness of the vision-language models and devising more diverse stress-te
    
[^9]: 关于上下文学习的综述

    A Survey on In-context Learning. (arXiv:2301.00234v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.00234](http://arxiv.org/abs/2301.00234)

    本文调查和总结了上下文学习(ICL)的进展和挑战，ICL已成为自然语言处理(NLP)的新范式，探索ICL以评估和推广大型语言模型(LLM)的能力已成为一种新趋势。本文提出了ICL的正式定义，并总结了高级技术，最后讨论了ICL的挑战以及进一步研究的潜在方向。

    

    随着大型语言模型（LLM）的能力不断增强，上下文学习（ICL）已成为自然语言处理（NLP）的新范式，在其中LLM仅基于加入少量示例的上下文进行预测。探索ICL以评估和推广LLM的能力已成为一种新趋势。本文旨在调查和总结ICL的进展和挑战。我们首先提出ICL的正式定义，并澄清其与相关研究的关系。然后，我们组织和讨论高级技术，包括训练策略、演示设计策略以及相关分析。最后，我们讨论了ICL的挑战，并提供了进一步研究的潜在方向。我们希望我们的工作可以鼓励更多的研究，揭示ICL的工作原理并改进ICL。

    With the increasing ability of large language models (LLMs), in-context learning (ICL) has become a new paradigm for natural language processing (NLP), where LLMs make predictions only based on contexts augmented with a few examples. It has been a new trend to explore ICL to evaluate and extrapolate the ability of LLMs. In this paper, we aim to survey and summarize the progress and challenges of ICL. We first present a formal definition of ICL and clarify its correlation to related studies. Then, we organize and discuss advanced techniques, including training strategies, demonstration designing strategies, as well as related analysis. Finally, we discuss the challenges of ICL and provide potential directions for further research. We hope that our work can encourage more research on uncovering how ICL works and improving ICL.
    

