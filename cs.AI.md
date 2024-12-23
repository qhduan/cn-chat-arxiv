# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Born With a Silver Spoon? Investigating Socioeconomic Bias in Large Language Models](https://arxiv.org/abs/2403.14633) | 本文调查了大型语言模型中是否存在社会经济偏见，引入了一个新的数据集SilverSpoon，并评估了这种偏见的程度以及随着模型大小的变化。 |
| [^2] | [From Algorithms to Outcomes: Reviewing AI's Role in Non-Muscle-Invasive Bladder Cancer Recurrence Prediction](https://arxiv.org/abs/2403.10586) | 机器学习技术在非肌层侵袭性膀胱癌复发预测中具有潜在作用，可以提高准确性，降低治疗成本，并有效规划治疗方案 |
| [^3] | [Enhancing Trust in Autonomous Agents: An Architecture for Accountability and Explainability through Blockchain and Large Language Models](https://arxiv.org/abs/2403.09567) | 通过区块链和大型语言模型实现责任和可解释性的架构，提高自主代理的信任和安全性，增强代理与用户之间的沟通效果。 |
| [^4] | [Generalizing Denoising to Non-Equilibrium Structures Improves Equivariant Force Fields](https://arxiv.org/abs/2403.09549) | 将去噪方法推广到非平衡结构，从而改进等变力场的性能，提高了对原子间相互作用的理解以及在分子动力学和催化剂设计等领域的应用。 |
| [^5] | [Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model](https://arxiv.org/abs/2402.02262) | 本文介绍了一种使用RoBERTa-CNN模型来在社交媒体帖子中检测自杀意图的新方法。RoBERTa-CNN通过在RoBERTa模型中添加卷积神经网络（CNN）层，提高了对重要模式的捕捉能力，并在实验证明在自杀和抑郁检测数据集上表现出良好的准确性。 |
| [^6] | [Synthesizing Moving People with 3D Control.](http://arxiv.org/abs/2401.10889) | 本文提出了一种基于扩散模型的框架，用于从单张图像中生成具有逼真移动的人物动画，并成功处理了人体不可见部分的合成问题。 |
| [^7] | [Variational measurement-based quantum computation for generative modeling.](http://arxiv.org/abs/2310.13524) | 这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。 |
| [^8] | [Large Language Models can Learn Rules.](http://arxiv.org/abs/2310.07064) | 大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。 |
| [^9] | [Learning ECG signal features without backpropagation.](http://arxiv.org/abs/2307.01930) | 该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。 |
| [^10] | [RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models.](http://arxiv.org/abs/2304.10727) | 本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。 |
| [^11] | [EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics.](http://arxiv.org/abs/2209.08996) | EDO-Net是一个学习可变形物体弹性属性的图动力学模型，通过利用可提取的潜在表示，可以推广到未知的物理属性，实现对类似布料的对象未来状态的预测和转移学习。 |

# 详细

[^1]: 出身富贵？探讨大型语言模型中的社会经济偏见

    Born With a Silver Spoon? Investigating Socioeconomic Bias in Large Language Models

    [https://arxiv.org/abs/2403.14633](https://arxiv.org/abs/2403.14633)

    本文调查了大型语言模型中是否存在社会经济偏见，引入了一个新的数据集SilverSpoon，并评估了这种偏见的程度以及随着模型大小的变化。

    

    社会经济偏见在社会中加剧了不公平现象，根据个人经济和社会背景影响获取机会和资源的机会。这一普遍问题持续地延续了系统性的不平等，阻碍了作为一个社会追求包容性进步。在本文中，我们调查了大型语言模型中是否存在社会经济偏见。为此，我们引入了一个新的数据集（SilverSpoon），包含3000个样本，展示了牵涉到弱势群体由于他们的处境而实施道德模糊行为的假设情景，并问这种行为是否在道德上成立。此外，这个数据集具有双重标记方案，并由属于社会经济两端的人进行了注释。使用SilverSpoon，我们评估了大型语言模型中表现出的社会经济偏见程度以及该程度如何随模型大小变化。

    arXiv:2403.14633v1 Announce Type: cross  Abstract: Socioeconomic bias in society exacerbates disparities, influencing access to opportunities and resources based on individuals' economic and social backgrounds. This pervasive issue perpetuates systemic inequalities, hindering the pursuit of inclusive progress as a society. In this paper, we investigate the presence of socioeconomic bias, if any, in large language models. To this end, we introduce a novel dataset (SilverSpoon), consisting of 3000 samples that illustrate hypothetical scenarios that involve underprivileged people performing ethically ambiguous actions due to their circumstances, and ask whether the action is ethically justified. Further, this dataset has a dual-labeling scheme and has been annotated by people belonging to both ends of the socioeconomic spectrum. Using SilverSpoon, we evaluate the degree of socioeconomic bias expressed in large language models and the variation of this degree as a function of model size. W
    
[^2]: 从算法到结果：审视人工智能在非肌层侵袭性膀胱癌复发预测中的作用

    From Algorithms to Outcomes: Reviewing AI's Role in Non-Muscle-Invasive Bladder Cancer Recurrence Prediction

    [https://arxiv.org/abs/2403.10586](https://arxiv.org/abs/2403.10586)

    机器学习技术在非肌层侵袭性膀胱癌复发预测中具有潜在作用，可以提高准确性，降低治疗成本，并有效规划治疗方案

    

    膀胱癌是英国每天造成15人死亡的领先泌尿道癌症。这种癌症主要表现为非肌层侵袭性膀胱癌（NMIBC），其特点是肿瘤还未渗透到膀胱壁的肌肉层。 NMIBC的复发率非常高，达到70-80％，因此治疗成本最高。目前用于预测复发的工具使用评分系统来高估风险，并具有较低的准确性。对复发的不准确和延迟预测显著提高了死亡的可能性。因此，准确预测复发对于成本效益的管理和治疗计划至关重要。这就是机器学习（ML）技术出现的地方，通过利用分子和临床数据预测NMIBC复发，成为一种有前途的方法。本次审查对预测NMIBC复发的ML方法进行了全面分析。我们的系统评估使

    arXiv:2403.10586v1 Announce Type: cross  Abstract: Bladder cancer, the leading urinary tract cancer, is responsible for 15 deaths daily in the UK. This cancer predominantly manifests as non-muscle-invasive bladder cancer (NMIBC), characterised by tumours not yet penetrating the muscle layer of the bladder wall. NMIBC is plagued by a very high recurrence rate of 70-80% and hence the costliest treatments. Current tools for predicting recurrence use scoring systems that overestimate risk and have poor accuracy. Inaccurate and delayed prediction of recurrence significantly elevates the likelihood of mortality. Accurate prediction of recurrence is hence vital for cost-effective management and treatment planning. This is where Machine learning (ML) techniques have emerged as a promising approach for predicting NMIBC recurrence by leveraging molecular and clinical data. This review provides a comprehensive analysis of ML approaches for predicting NMIBC recurrence. Our systematic evaluation de
    
[^3]: 通过区块链和大型语言模型增强自主代理的信任：一种通过区块链和大型语言模型实现责任和可解释性的架构

    Enhancing Trust in Autonomous Agents: An Architecture for Accountability and Explainability through Blockchain and Large Language Models

    [https://arxiv.org/abs/2403.09567](https://arxiv.org/abs/2403.09567)

    通过区块链和大型语言模型实现责任和可解释性的架构，提高自主代理的信任和安全性，增强代理与用户之间的沟通效果。

    

    自主代理在涉及人类互动的环境中的部署日益引起安全关注。因此，了解事件背后的情况变得至关重要，需要开发能够向非专家用户解释其行为的能力。这些解释在提高可信度和安全性方面至关重要，作为防范失败、错误和误解的措施。此外，它们有助于改善沟通，弥合代理和用户之间的差距，从而提高它们相互作用的效果。这项工作提出了一种为基于ROS的移动机器人实施的责任和可解释性架构。所提出的解决方案包括两个主要组件。首先，一个类似黑盒的元素用于提供问责制，具有通过区块链技术实现的防篡改属性。其次，一个负责的组件

    arXiv:2403.09567v1 Announce Type: cross  Abstract: The deployment of autonomous agents in environments involving human interaction has increasingly raised security concerns. Consequently, understanding the circumstances behind an event becomes critical, requiring the development of capabilities to justify their behaviors to non-expert users. Such explanations are essential in enhancing trustworthiness and safety, acting as a preventive measure against failures, errors, and misunderstandings. Additionally, they contribute to improving communication, bridging the gap between the agent and the user, thereby improving the effectiveness of their interactions. This work presents an accountability and explainability architecture implemented for ROS-based mobile robots. The proposed solution consists of two main components. Firstly, a black box-like element to provide accountability, featuring anti-tampering properties achieved through blockchain technology. Secondly, a component in charge of 
    
[^4]: 将去噪推广到非平衡结构以改进等变力场

    Generalizing Denoising to Non-Equilibrium Structures Improves Equivariant Force Fields

    [https://arxiv.org/abs/2403.09549](https://arxiv.org/abs/2403.09549)

    将去噪方法推广到非平衡结构，从而改进等变力场的性能，提高了对原子间相互作用的理解以及在分子动力学和催化剂设计等领域的应用。

    

    理解原子间的相互作用，如3D原子体系中的力，对于许多应用如分子动力学和催化剂设计至关重要。然而，模拟这些相互作用需要计算密集的从头算计算，因此训练神经网络的数据有限。本文提出使用去噪非平衡结构（DeNS）作为辅助任务，以更好地利用训练数据并提高性能。在使用DeNS进行训练时，我们首先通过向其3D坐标添加噪声来破坏3D结构，然后预测噪声。不同于以往仅限于平衡结构的去噪工作，所提出的方法将去噪泛化到更大范围的非平衡结构。主要区别在于非平衡结构不对应于局部能量最小值，具有非零力，因此可能具有许多可能的原子位置。

    arXiv:2403.09549v1 Announce Type: cross  Abstract: Understanding the interactions of atoms such as forces in 3D atomistic systems is fundamental to many applications like molecular dynamics and catalyst design. However, simulating these interactions requires compute-intensive ab initio calculations and thus results in limited data for training neural networks. In this paper, we propose to use denoising non-equilibrium structures (DeNS) as an auxiliary task to better leverage training data and improve performance. For training with DeNS, we first corrupt a 3D structure by adding noise to its 3D coordinates and then predict the noise. Different from previous works on denoising, which are limited to equilibrium structures, the proposed method generalizes denoising to a much larger set of non-equilibrium structures. The main difference is that a non-equilibrium structure does not correspond to local energy minima and has non-zero forces, and therefore it can have many possible atomic posit
    
[^5]: 数据质量很重要：使用RoBERTa-CNN模型在社交媒体帖子中检测自杀意图

    Data Quality Matters: Suicide Intention Detection on Social Media Posts Using a RoBERTa-CNN Model

    [https://arxiv.org/abs/2402.02262](https://arxiv.org/abs/2402.02262)

    本文介绍了一种使用RoBERTa-CNN模型来在社交媒体帖子中检测自杀意图的新方法。RoBERTa-CNN通过在RoBERTa模型中添加卷积神经网络（CNN）层，提高了对重要模式的捕捉能力，并在实验证明在自杀和抑郁检测数据集上表现出良好的准确性。

    

    自杀仍然是全球健康领域的一个关注焦点，急需创新方法进行早期检测和干预。本文着重于识别SuicideWatch Reddit帖子中的自杀意图，并提出了一种使用尖端的RoBERTa-CNN模型进行自杀检测的新方法，RoBERTa-CNN是RoBERTa（鲁棒性优化BERT方法）的一种变体。RoBERTa被用于各种自然语言处理（NLP）任务，包括文本分类和情感分析。RoBERTa的有效性在于它能够捕捉文本信息并形成文本之间的语义关系。通过在原始模型中添加卷积神经网络（CNN）层，RoBERTa增强了从庞大数据集中捕捉重要模式的能力。我们在自杀和抑郁检测数据集上评估了RoBERTa-CNN，并获得了可靠的结果，例如，RoBERTa-CNN在平均准确率上获得了98％，标准差为...

    Suicide remains a global health concern for the field of health, which urgently needs innovative approaches for early detection and intervention. In this paper, we focus on identifying suicidal intentions in SuicideWatch Reddit posts and present a novel approach to suicide detection using the cutting-edge RoBERTa-CNN model, a variant of RoBERTa (Robustly optimized BERT approach). RoBERTa is used for various Natural Language Processing (NLP) tasks, including text classification and sentiment analysis. The effectiveness of the RoBERTa lies in its ability to capture textual information and form semantic relationships within texts. By adding the Convolution Neural Network (CNN) layer to the original model, the RoBERTa enhances its ability to capture important patterns from heavy datasets. To evaluate the RoBERTa-CNN, we experimented on the Suicide and Depression Detection dataset and obtained solid results. For example, RoBERTa-CNN achieves 98% mean accuracy with the standard deviation (ST
    
[^6]: 使用3D控制合成移动人物

    Synthesizing Moving People with 3D Control. (arXiv:2401.10889v1 [cs.CV])

    [http://arxiv.org/abs/2401.10889](http://arxiv.org/abs/2401.10889)

    本文提出了一种基于扩散模型的框架，用于从单张图像中生成具有逼真移动的人物动画，并成功处理了人体不可见部分的合成问题。

    

    本文提出了一种基于扩散模型的框架，用于从单张图像中为给定的目标3D运动序列生成人物动画。我们的方法包含两个核心组成部分：a) 学习关于人体和服装不可见部分的先验知识，b) 以适当的服装和纹理渲染新的人体姿势。对于第一部分，我们学习了一种填充扩散模型，以给定单张图像生成人物的不可见部分。我们在纹理映射空间上训练这个模型，使其对姿势和视角不变，从而提高了样本效率。其次，我们开发了一个基于扩散的渲染流水线，由3D人体姿势控制。这可以产生逼真的人物新姿势的渲染图像，包括服装、头发和未知区域的合理补充。这种分解的方法使我们的方法能够生成一系列图像，既符合3D姿势中的目标运动，也符合视觉上与输入图像的相似性。

    In this paper, we present a diffusion model-based framework for animating people from a single image for a given target 3D motion sequence. Our approach has two core components: a) learning priors about invisible parts of the human body and clothing, and b) rendering novel body poses with proper clothing and texture. For the first part, we learn an in-filling diffusion model to hallucinate unseen parts of a person given a single image. We train this model on texture map space, which makes it more sample-efficient since it is invariant to pose and viewpoint. Second, we develop a diffusion-based rendering pipeline, which is controlled by 3D human poses. This produces realistic renderings of novel poses of the person, including clothing, hair, and plausible in-filling of unseen regions. This disentangled approach allows our method to generate a sequence of images that are faithful to the target motion in the 3D pose and, to the input image in terms of visual similarity. In addition to tha
    
[^7]: 基于测量的变分量子计算用于生成建模

    Variational measurement-based quantum computation for generative modeling. (arXiv:2310.13524v1 [quant-ph])

    [http://arxiv.org/abs/2310.13524](http://arxiv.org/abs/2310.13524)

    这项研究提出了一种基于测量的变分量子计算算法，将量子测量的随机性视为计算资源，并应用于生成建模任务。

    

    基于测量的量子计算（MBQC）提供了一种基本独特的范例来设计量子算法。在MBQC中，由于量子测量的固有随机性，自然的操作不是确定性和幺正的，而是通过概率附带的。然而，到目前为止，MBQC的主要算法应用是完全抵消这种概率性质，以模拟表达在电路模型中的幺正计算。在这项工作中，我们提出了设计MBQC算法的思路，该算法接受这种固有随机性，并将MBQC中的随机附带视为计算资源。我们考虑了随机性有益的自然应用，即生成建模，这是一个以生成复杂概率分布为中心的机器学习任务。为了解决这个任务，我们提出了一个具有控制参数的变分MBQC算法，可以直接调整允许在计算中引入的随机程度。

    Measurement-based quantum computation (MBQC) offers a fundamentally unique paradigm to design quantum algorithms. Indeed, due to the inherent randomness of quantum measurements, the natural operations in MBQC are not deterministic and unitary, but are rather augmented with probabilistic byproducts. Yet, the main algorithmic use of MBQC so far has been to completely counteract this probabilistic nature in order to simulate unitary computations expressed in the circuit model. In this work, we propose designing MBQC algorithms that embrace this inherent randomness and treat the random byproducts in MBQC as a resource for computation. As a natural application where randomness can be beneficial, we consider generative modeling, a task in machine learning centered around generating complex probability distributions. To address this task, we propose a variational MBQC algorithm equipped with control parameters that allow to directly adjust the degree of randomness to be admitted in the comput
    
[^8]: 大型语言模型可以学习规则

    Large Language Models can Learn Rules. (arXiv:2310.07064v1 [cs.AI])

    [http://arxiv.org/abs/2310.07064](http://arxiv.org/abs/2310.07064)

    大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。为了提高提示方法的准确性和一致性，我们提出了Hypotheses-to-Theories (HtT)框架，用于学习LLMs推理的规则库，从而改进了现有的提示方法。

    

    当给出一些示例和中间步骤时，大型语言模型(LLMs)在各种推理任务中展示了令人印象深刻的性能。然而，依赖LLM中的隐式知识的提示方法在隐式知识错误或与任务不一致时往往会产生错误的答案。为解决这个问题，我们提出了"假设到理论" (HtT) 框架，用于学习LLMs推理的规则库。HtT包括两个阶段，归纳阶段和演绎阶段。在归纳阶段，首先要求LLM根据一组训练示例生成和验证规则。出现并导致正确答案的规则将被收集形成一个规则库。在演绎阶段，然后要求LLM使用学习的规则库进行推理以回答测试问题。在数值推理和关系推理问题上的实验证明，HtT改进了现有的提示方法，使其性能提升。

    When prompted with a few examples and intermediate steps, large language models (LLMs) have demonstrated impressive performance in various reasoning tasks. However, prompting methods that rely on implicit knowledge in an LLM often hallucinate incorrect answers when the implicit knowledge is wrong or inconsistent with the task. To tackle this problem, we present Hypotheses-to-Theories (HtT), a framework that learns a rule library for reasoning with LLMs. HtT contains two stages, an induction stage and a deduction stage. In the induction stage, an LLM is first asked to generate and verify rules over a set of training examples. Rules that appear and lead to correct answers sufficiently often are collected to form a rule library. In the deduction stage, the LLM is then prompted to employ the learned rule library to perform reasoning to answer test questions. Experiments on both numerical reasoning and relational reasoning problems show that HtT improves existing prompting methods, with an 
    
[^9]: 学习ECG信号特征的非反向传播方法

    Learning ECG signal features without backpropagation. (arXiv:2307.01930v1 [cs.LG])

    [http://arxiv.org/abs/2307.01930](http://arxiv.org/abs/2307.01930)

    该论文提出了一种用于生成时间序列数据表示的新方法，依靠理论物理的思想以数据驱动的方式构建紧凑的表示。该方法能够捕捉数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性，并可以在广义设置中应用。

    

    表示学习已经成为机器学习领域的一个关键研究领域，它旨在发现用于提高分类和预测等下游任务的原始数据的有效特征的有效方法。在本文中，我们提出了一种用于生成时间序列类型数据表示的新方法。这种方法依靠理论物理的思想以数据驱动的方式构建紧凑的表示，并可以捕捉到数据的基本结构和任务特定信息，同时保持直观、可解释和可验证性。这个新方法旨在识别能够有效捕捉属于特定类别的样本之间共享特征的线性规律。通过随后利用这些规律在前向方式下生成一个与分类器无关的表示，它们可以在广义设置中应用。我们展示了我们方法的有效性。

    Representation learning has become a crucial area of research in machine learning, as it aims to discover efficient ways of representing raw data with useful features to increase the effectiveness, scope and applicability of downstream tasks such as classification and prediction. In this paper, we propose a novel method to generate representations for time series-type data. This method relies on ideas from theoretical physics to construct a compact representation in a data-driven way, and it can capture both the underlying structure of the data and task-specific information while still remaining intuitive, interpretable and verifiable. This novel methodology aims to identify linear laws that can effectively capture a shared characteristic among samples belonging to a specific class. By subsequently utilizing these laws to generate a classifier-agnostic representation in a forward manner, they become applicable in a generalized setting. We demonstrate the effectiveness of our approach o
    
[^10]: RoCOCO：稳健的基准MS-COCO评估图文匹配模型的鲁棒性

    RoCOCO: Robust Benchmark MS-COCO to Stress-test Robustness of Image-Text Matching Models. (arXiv:2304.10727v1 [cs.CV])

    [http://arxiv.org/abs/2304.10727](http://arxiv.org/abs/2304.10727)

    本文提出了一个新的评估基准来测试ITM模型的鲁棒性，通过将一些“愚弄”的图片和标题添加到检索池中，在MS COCO数据集上为各种最先进的模型进行鲁棒性测试，揭示了它们的不足之处。

    

    近年来，大规模的视觉语言预训练模型和视觉语义嵌入方法显著提高了MS COCO 5K测试集上图文匹配（ITM）的准确性。然而，当将这些最先进的模型用于实际应用时，它们的鲁棒性仍不清楚。本文提出了一个新的评估基准来测试ITM模型的鲁棒性。为此，我们将各种“愚弄”的图片和标题添加到检索池中。具体而言，我们通过插入不相关的图像来更改图像，并通过替换名词来更改标题，从而改变句子的含义。我们发现，仅仅将这些新创建的图像和标题添加到测试集中就可以降低各种最先进模型的性能（例如，在BLIP中从81.9％降至64.5％，在VSE∞中从66.1％降至37.5％）。我们希望我们的发现能为提高视觉语言模型的鲁棒性和设计更多样化的压力测试提供启示。

    Recently, large-scale vision-language pre-training models and visual semantic embedding methods have significantly improved image-text matching (ITM) accuracy on MS COCO 5K test set. However, it is unclear how robust these state-of-the-art (SOTA) models are when using them in the wild. In this paper, we propose a novel evaluation benchmark to stress-test the robustness of ITM models. To this end, we add various fooling images and captions to a retrieval pool. Specifically, we change images by inserting unrelated images, and change captions by substituting a noun, which can change the meaning of a sentence. We discover that just adding these newly created images and captions to the test set can degrade performances (i.e., Recall@1) of a wide range of SOTA models (e.g., 81.9% $\rightarrow$ 64.5% in BLIP, 66.1% $\rightarrow$ 37.5% in VSE$\infty$). We expect that our findings can provide insights for improving the robustness of the vision-language models and devising more diverse stress-te
    
[^11]: EDO-Net: 从图动力学中学习可变形物体的弹性属性

    EDO-Net: Learning Elastic Properties of Deformable Objects from Graph Dynamics. (arXiv:2209.08996v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.08996](http://arxiv.org/abs/2209.08996)

    EDO-Net是一个学习可变形物体弹性属性的图动力学模型，通过利用可提取的潜在表示，可以推广到未知的物理属性，实现对类似布料的对象未来状态的预测和转移学习。

    

    我们研究了学习可变形物体的图动力学的问题，该问题可以推广到未知的物理属性。我们的关键洞察力是利用可提取的类似布料的可变形物体的弹性物理属性的潜在表示，例如从拉伸交互中提取。在本文中，我们提出了EDO-Net（弹性可变形对象-Net），这是一个在具有不同弹性属性的大量样本上进行训练的图动力学模型，不依赖于属性的真实标签。EDO-Net共同学习了一个自适应模块和一个前向动力学模块。前者负责提取对象的物理特性的潜在表示，而后者利用潜在表示来预测以图形表示的类似布料的对象的未来状态。我们在仿真和真实世界中评估了EDO-Net的能力：1）推广到未知的物理属性，2）转移学习所学到的表示

    We study the problem of learning graph dynamics of deformable objects that generalizes to unknown physical properties. Our key insight is to leverage a latent representation of elastic physical properties of cloth-like deformable objects that can be extracted, for example, from a pulling interaction. In this paper we propose EDO-Net (Elastic Deformable Object - Net), a model of graph dynamics trained on a large variety of samples with different elastic properties that does not rely on ground-truth labels of the properties. EDO-Net jointly learns an adaptation module, and a forward-dynamics module. The former is responsible for extracting a latent representation of the physical properties of the object, while the latter leverages the latent representation to predict future states of cloth-like objects represented as graphs. We evaluate EDO-Net both in simulation and real world, assessing its capabilities of: 1) generalizing to unknown physical properties, 2) transferring the learned rep
    

