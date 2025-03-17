# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions](https://arxiv.org/abs/2403.17064) | 通过识别CLIP文本嵌入中的语义方向，实现了文本到图像模型中对高级属性的细粒度主题特定控制。 |
| [^2] | [LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving](https://arxiv.org/abs/2403.08215) | 将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。 |
| [^3] | [Few-Shot Learning for Chronic Disease Management: Leveraging Large Language Models and Multi-Prompt Engineering with Medical Knowledge Injection.](http://arxiv.org/abs/2401.12988) | 本研究提出了慢性病管理的少样本学习框架，利用大型语言模型和多Prompt工程进行精神障碍的检测，通过个性化的提示和医疗知识注入来解决数据挑战，实现慢性病管理的目标。 |
| [^4] | [CoPAL: Corrective Planning of Robot Actions with Large Language Models.](http://arxiv.org/abs/2310.07263) | 本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。 |
| [^5] | [Concise and Organized Perception Facilitates Large Language Models for Deductive Reasoning.](http://arxiv.org/abs/2310.03309) | 利用大型语言模型进行演绎推理是一个具有挑战性的问题。这篇论文提出了一个简明有序的方法，将任务分解为子任务并且人类化地组织思维，以提高演绎推理的效果。 |
| [^6] | [Disentanglement Learning via Topology.](http://arxiv.org/abs/2308.12696) | 本文提出了一种通过拓扑损失实现解缠编码的方法，这是第一个提出用于解缠的可微拓扑损失的论文，实验结果表明所提出的方法相对于最新结果改进了解缠得分。 |
| [^7] | [Playing with Words: Comparing the Vocabulary and Lexical Richness of ChatGPT and Humans.](http://arxiv.org/abs/2308.07462) | 这篇论文比较了ChatGPT和人类在词汇和词汇丰富度方面的差异，研究发现使用ChatGPT等工具会对词汇使用和词汇丰富度产生影响，这可能会对语言演变产生影响。 |
| [^8] | [Diverse Projection Ensembles for Distributional Reinforcement Learning.](http://arxiv.org/abs/2306.07124) | 本文研究了分布式强化学习中多样投影集合的理论特性，提出了使用集合差异度量的算法，以促进可靠的不确定性估计。 |
| [^9] | [In Defense of Pure 16-bit Floating-Point Neural Networks.](http://arxiv.org/abs/2305.10947) | 本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。 |
| [^10] | [Device-Robust Acoustic Scene Classification via Impulse Response Augmentation.](http://arxiv.org/abs/2305.07499) | 本篇论文提出了一种基于冲击响应增强的方法，用于解决音频分类模型泛化到未被训练设备上时性能下降的问题。 |
| [^11] | [Virtual Guidance as a Mid-level Representation for Navigation.](http://arxiv.org/abs/2303.02731) | 该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。 |
| [^12] | [Logit-Q Dynamics for Efficient Learning in Stochastic Teams.](http://arxiv.org/abs/2302.09806) | 本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。 |

# 详细

[^1]: 在T2I模型中通过识别语义方向实现连续、主题特定的属性控制

    Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions

    [https://arxiv.org/abs/2403.17064](https://arxiv.org/abs/2403.17064)

    通过识别CLIP文本嵌入中的语义方向，实现了文本到图像模型中对高级属性的细粒度主题特定控制。

    

    近年来，文本到图像（T2I）扩散模型的进展显著提高了生成图像的质量。然而，由于自然语言提示的限制（例如“人”和“老年人”之间不存在连续的中间描述的集合），实现对属性的细粒度控制仍然是一个挑战。尽管引入了许多方法来增强模型或生成过程以实现这种控制，但不需要固定参考图像的方法仅限于启用全局细粒度属性表达控制或仅限于特定主题的粗粒度属性表达控制，而不能同时兼顾两者。我们展示了在常用的基于标记级别的CLIP文本嵌入中存在可实现文本到图像模型中高级属性的细粒度主题特定控制的方向。基于这一观察，我们引入了一种有效的方法。

    arXiv:2403.17064v1 Announce Type: cross  Abstract: In recent years, advances in text-to-image (T2I) diffusion models have substantially elevated the quality of their generated images. However, achieving fine-grained control over attributes remains a challenge due to the limitations of natural language prompts (such as no continuous set of intermediate descriptions existing between ``person'' and ``old person''). Even though many methods were introduced that augment the model or generation process to enable such control, methods that do not require a fixed reference image are limited to either enabling global fine-grained attribute expression control or coarse attribute expression control localized to specific subjects, not both simultaneously. We show that there exist directions in the commonly used token-level CLIP text embeddings that enable fine-grained subject-specific control of high-level attributes in text-to-image models. Based on this observation, we introduce one efficient op
    
[^2]: LIX：将空间几何先验知识隐式注入视觉语义分割，用于自动驾驶

    LIX: Implicitly Infusing Spatial Geometric Prior Knowledge into Visual Semantic Segmentation for Autonomous Driving

    [https://arxiv.org/abs/2403.08215](https://arxiv.org/abs/2403.08215)

    将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型，通过新的logit蒸馏和特征蒸馏方法，解决自动驾驶中的视觉语义分割问题。

    

    尽管数据融合网络在视觉语义分割中表现出色，但当缺乏空间几何数据时，双编码器变得无效。将双编码器教师模型获得的空间几何先验知识隐式注入单编码器学生模型是一个实用但不太探索的研究领域。本文深入探讨了这个主题，并采用知识蒸馏方法来解决这个问题。我们引入了Learning to Infuse "X" (LIX) 框架，在logit蒸馏和特征蒸馏方面进行了新颖贡献。我们提出了一个数学证明，强调在解耦知识蒸馏中使用单一固定权重的局限性，并引入了logit智能动态权重控制器作为解决这个问题的方法。此外，我们开发了一种自适应重新校准的特征蒸馏算法，包括两种技术。

    arXiv:2403.08215v1 Announce Type: cross  Abstract: Despite the impressive performance achieved by data-fusion networks with duplex encoders for visual semantic segmentation, they become ineffective when spatial geometric data are not available. Implicitly infusing the spatial geometric prior knowledge acquired by a duplex-encoder teacher model into a single-encoder student model is a practical, albeit less explored research avenue. This paper delves into this topic and resorts to knowledge distillation approaches to address this problem. We introduce the Learning to Infuse "X" (LIX) framework, with novel contributions in both logit distillation and feature distillation aspects. We present a mathematical proof that underscores the limitation of using a single fixed weight in decoupled knowledge distillation and introduce a logit-wise dynamic weight controller as a solution to this issue. Furthermore, we develop an adaptively-recalibrated feature distillation algorithm, including two tec
    
[^3]: 慢性病管理的少样本学习：利用大型语言模型和多Prompt工程与医疗知识注入

    Few-Shot Learning for Chronic Disease Management: Leveraging Large Language Models and Multi-Prompt Engineering with Medical Knowledge Injection. (arXiv:2401.12988v1 [cs.CL])

    [http://arxiv.org/abs/2401.12988](http://arxiv.org/abs/2401.12988)

    本研究提出了慢性病管理的少样本学习框架，利用大型语言模型和多Prompt工程进行精神障碍的检测，通过个性化的提示和医疗知识注入来解决数据挑战，实现慢性病管理的目标。

    

    本研究利用最先进的人工智能技术进行慢性病管理，特别是通过用户生成的文本内容来检测各种精神障碍。现有研究通常依赖于全监督机器学习，这带来了一些挑战，比如注释庞大的训练数据对于每种疾病的费时费力的手动过程，以及需要为每个问题设计专门的深度学习架构。为了解决这些挑战，我们提出了一个新颖的框架，利用了先进的人工智能技术，包括大型语言模型和多Prompt工程。具体而言，我们解决了数据驱动慢性病管理中的两个关键技术挑战：（1）开发个性化的提示来表示每个用户的独特性，以及（2）将医疗知识融入到提示中，为慢性病检测提供上下文，指导学习目标，并实现预测目标。我们使用四种精神障碍来评估我们的方法。

    This study harnesses state-of-the-art AI technology for chronic disease management, specifically in detecting various mental disorders through user-generated textual content. Existing studies typically rely on fully supervised machine learning, which presents challenges such as the labor-intensive manual process of annotating extensive training data for each disease and the need to design specialized deep learning architectures for each problem. To address such challenges, we propose a novel framework that leverages advanced AI techniques, including large language models and multi-prompt engineering. Specifically, we address two key technical challenges in data-driven chronic disease management: (1) developing personalized prompts to represent each user's uniqueness and (2) incorporating medical knowledge into prompts to provide context for chronic disease detection, instruct learning objectives, and operationalize prediction goals. We evaluate our method using four mental disorders, w
    
[^4]: CoPAL:具有大规模语言模型的机器人动作纠正规划

    CoPAL: Corrective Planning of Robot Actions with Large Language Models. (arXiv:2310.07263v1 [cs.RO])

    [http://arxiv.org/abs/2310.07263](http://arxiv.org/abs/2310.07263)

    本文提出了一个具有大规模语言模型的机器人动作纠正规划系统，通过处理生成计划中的物理基础、逻辑和语义错误的再规划策略，实现了在复杂环境中的任务和动作规划。通过仿真和实际场景的验证，证明了该系统的有效性。

    

    为了实现完全自主的机器人系统能够接管人类传统执行的任务，开放世界环境的复杂性提出了巨大的挑战。在这一背景下，本研究为应用于机器人任务和动作规划的大规模语言模型领域做出了贡献。我们提出了一个系统架构，协调多个认知层次之间的无缝交互，包括推理、规划和动作生成。其核心是一种处理生成的计划中的物理基础、逻辑和语义错误的新型再规划策略。通过在仿真环境和两个复杂的实际场景（方块世界、酒吧和比萨制作）中进行实证评估，我们展示了所提出的反馈架构的有效性，尤其是对可执行性、正确性和时间复杂性的影响。

    In the pursuit of fully autonomous robotic systems capable of taking over tasks traditionally performed by humans, the complexity of open-world environments poses a considerable challenge. Addressing this imperative, this study contributes to the field of Large Language Models (LLMs) applied to task and motion planning for robots. We propose a system architecture that orchestrates a seamless interplay between multiple cognitive levels, encompassing reasoning, planning, and motion generation. At its core lies a novel replanning strategy that handles physically grounded, logical, and semantic errors in the generated plans. We demonstrate the efficacy of the proposed feedback architecture, particularly its impact on executability, correctness, and time complexity via empirical evaluation in the context of a simulation and two intricate real-world scenarios: blocks world, barman and pizza preparation.
    
[^5]: 简明有序的感知有助于大型语言模型进行演绎推理

    Concise and Organized Perception Facilitates Large Language Models for Deductive Reasoning. (arXiv:2310.03309v1 [cs.CL])

    [http://arxiv.org/abs/2310.03309](http://arxiv.org/abs/2310.03309)

    利用大型语言模型进行演绎推理是一个具有挑战性的问题。这篇论文提出了一个简明有序的方法，将任务分解为子任务并且人类化地组织思维，以提高演绎推理的效果。

    

    利用大型语言模型（LLMs）解决演绎推理问题已经引起了越来越多的关注。在复杂的演绎问题中仍然很难取得令人满意的结果，这类问题具有大量前提（即事实或规则），其中涉及实体之间错综复杂的关系，需要进行多跳推理。一种直观的解决方案是将原始任务分解为较小的子任务，然后以前向（例如选择-推理）或反向（例如LAMBADA）方式将多个因果推理步骤连接在一起。然而，这些技术不可避免地需要大量的总体阶段，导致计算开销大，并且有更高的可能性产生误导性的步骤。除了逐阶段分解之外，我们还从人类问题解决的另一个方面获得了启发。人类倾向于提炼出最相关的信息并有序地组织思维（例如创建思维导图），这有助于他们对问题进行有效的推理。

    Exploiting large language models (LLMs) to tackle deductive reasoning has garnered growing attention. It still remains highly challenging to achieve satisfactory results in complex deductive problems, characterized by plenty of premises (i.e., facts or rules) entailing intricate relationships among entities and requiring multi-hop reasoning. One intuitive solution is to decompose the original task into smaller sub-tasks, and then chain the multiple casual reasoning steps together in a forward (e.g., Selection-Inference) or backward (e.g., LAMBADA) direction. However, these techniques inevitably necessitate a large number of overall stages, leading to computationally expensive operations and a higher possibility of making misleading steps. In addition to stage-by-stage decomposition, we draw inspiration from another aspect of human problem-solving. Humans tend to distill the most relevant information and organize their thoughts systematically (e.g., creating mind maps), which assists th
    
[^6]: 通过拓扑学习实现解缠编码

    Disentanglement Learning via Topology. (arXiv:2308.12696v1 [cs.LG])

    [http://arxiv.org/abs/2308.12696](http://arxiv.org/abs/2308.12696)

    本文提出了一种通过拓扑损失实现解缠编码的方法，这是第一个提出用于解缠的可微拓扑损失的论文，实验结果表明所提出的方法相对于最新结果改进了解缠得分。

    

    我们提出了TopDis（拓扑解缠），一种通过增加多尺度拓扑损失项学习解缠表示的方法。解缠是数据表示的关键属性，对深度学习模型的可解释性和鲁棒性以及高级认知的实现都非常重要。基于VAE的最新方法通过最小化潜变量的联合分布的总体相关性来实现解缠。我们从分析数据流形的拓扑属性的角度来看待解缠，特别是优化数据流形遍历的拓扑相似性。据我们所知，我们的论文是第一个提出用于解缠的可微拓扑损失的方法。我们的实验结果表明，所提出的拓扑损失相对于最新结果改进了解缠得分，如MIG、FactorVAE得分、SAP得分和DCI解缠得分。我们的方法以无监督的方式工作。

    We propose TopDis (Topological Disentanglement), a method for learning disentangled representations via adding multi-scale topological loss term. Disentanglement is a crucial property of data representations substantial for the explainability and robustness of deep learning models and a step towards high-level cognition. The state-of-the-art method based on VAE minimizes the total correlation of the joint distribution of latent variables. We take a different perspective on disentanglement by analyzing topological properties of data manifolds. In particular, we optimize the topological similarity for data manifolds traversals. To the best of our knowledge, our paper is the first one to propose a differentiable topological loss for disentanglement. Our experiments have shown that the proposed topological loss improves disentanglement scores such as MIG, FactorVAE score, SAP score and DCI disentanglement score with respect to state-of-the-art results. Our method works in an unsupervised m
    
[^7]: 玩弄文字：比较ChatGPT和人类的词汇和词汇丰富度

    Playing with Words: Comparing the Vocabulary and Lexical Richness of ChatGPT and Humans. (arXiv:2308.07462v1 [cs.CL])

    [http://arxiv.org/abs/2308.07462](http://arxiv.org/abs/2308.07462)

    这篇论文比较了ChatGPT和人类在词汇和词汇丰富度方面的差异，研究发现使用ChatGPT等工具会对词汇使用和词汇丰富度产生影响，这可能会对语言演变产生影响。

    

    人工智能生成语言模型（如GPT）和ChatGPT等工具的引入引发了一场革命，可以改变文本生成的方式。这对读者的语言能力以及新型人工智能工具的培训是否会产生影响具有许多含义？它是否会影响语言的演变？我们关注语言的一个特定方面：词语；在编写给定文本时，使用ChatGPT等工具会增加或减少使用的词汇量或词汇丰富度（理解为书面或口头表达中使用的不同词汇数量）？这对词语有影响，因为未包含在人工智能生成的内容中的词语往往会变得越来越不受欢迎，并最终可能消失。在这项工作中，我们对ChatGPT和人类的词汇和词汇丰富度进行了初步比较。

    The introduction of Artificial Intelligence (AI) generative language models such as GPT (Generative Pre-trained Transformer) and tools such as ChatGPT has triggered a revolution that can transform how text is generated. This has many implications, for example, as AI-generated text becomes a significant fraction of the text in many disciplines, would this have an effect on the language capabilities of readers and also on the training of newer AI tools? Would it affect the evolution of languages? Focusing on one specific aspect of the language: words; will the use of tools such as ChatGPT increase or reduce the vocabulary used or the lexical richness (understood as the number of different words used in a written or oral production) when writing a given text? This has implications for words, as those not included in AI-generated content will tend to be less and less popular and may eventually be lost. In this work, we perform an initial comparison of the vocabulary and lexical richness of
    
[^8]: 分布式强化学习的多样投影集合

    Diverse Projection Ensembles for Distributional Reinforcement Learning. (arXiv:2306.07124v1 [cs.LG])

    [http://arxiv.org/abs/2306.07124](http://arxiv.org/abs/2306.07124)

    本文研究了分布式强化学习中多样投影集合的理论特性，提出了使用集合差异度量的算法，以促进可靠的不确定性估计。

    

    与传统的强化学习不同，分布式强化学习算法旨在学习回报的分布而不是其期望值。由于回报分布的性质通常是未知的或过于复杂，因此通常采用将未约束的分布投影到可表示的参数分布集合中的方法进行逼近。我们认为，当将这种投影步骤与神经网络和梯度下降相结合时，这种投影步骤会产生强烈的归纳偏见，从而深刻影响学习模型的泛化行为。为了通过多样性促进可靠的不确定性估计，本文研究了分布式集合中多个不同的投影和表示的组合。我们建立了这种投影集合的理论特性，并推导出一种使用集合差异度量的算法。

    In contrast to classical reinforcement learning, distributional reinforcement learning algorithms aim to learn the distribution of returns rather than their expected value. Since the nature of the return distribution is generally unknown a priori or arbitrarily complex, a common approach finds approximations within a set of representable, parametric distributions. Typically, this involves a projection of the unconstrained distribution onto the set of simplified distributions. We argue that this projection step entails a strong inductive bias when coupled with neural networks and gradient descent, thereby profoundly impacting the generalization behavior of learned models. In order to facilitate reliable uncertainty estimation through diversity, this work studies the combination of several different projections and representations in a distributional ensemble. We establish theoretical properties of such projection ensembles and derive an algorithm that uses ensemble disagreement, measure
    
[^9]: 关于纯16位浮点神经网络的辩护

    In Defense of Pure 16-bit Floating-Point Neural Networks. (arXiv:2305.10947v1 [cs.LG])

    [http://arxiv.org/abs/2305.10947](http://arxiv.org/abs/2305.10947)

    本文探讨了纯16位浮点神经网络的被忽视的效率，提供了理论分析来探讨16位和32位模型的差异，并可以定量解释16位模型与其32位对应物之间的条件。

    

    减少编码神经网络权重和激活所需的位数是非常可取的，因为它可以加快神经网络的训练和推理时间，同时减少内存消耗。因此，这一领域的研究引起了广泛关注，以开发利用更低精度计算的神经网络，比如混合精度训练。有趣的是，目前不存在纯16位浮点设置的方法。本文揭示了纯16位浮点神经网络被忽视的效率。我们通过提供全面的理论分析来探讨造成16位和32位模型的差异的因素。我们规范化了浮点误差和容忍度的概念，从而可以定量解释16位模型与其32位对应物之间密切逼近结果的条件。这种理论探索提供了新的视角。

    Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. For these reasons, research in this area has attracted significant attention toward developing neural networks that leverage lower-precision computing, such as mixed-precision training. Interestingly, none of the existing approaches has investigated pure 16-bit floating-point settings. In this paper, we shed light on the overlooked efficiency of pure 16-bit floating-point neural networks. As such, we provide a comprehensive theoretical analysis to investigate the factors contributing to the differences observed between 16-bit and 32-bit models. We formalize the concepts of floating-point error and tolerance, enabling us to quantitatively explain the conditions under which a 16-bit model can closely approximate the results of its 32-bit counterpart. This theoretical exploration offers perspect
    
[^10]: 基于冲击响应增强的设备鲁棒声学场景分类

    Device-Robust Acoustic Scene Classification via Impulse Response Augmentation. (arXiv:2305.07499v1 [cs.SD])

    [http://arxiv.org/abs/2305.07499](http://arxiv.org/abs/2305.07499)

    本篇论文提出了一种基于冲击响应增强的方法，用于解决音频分类模型泛化到未被训练设备上时性能下降的问题。

    

    对于音频分类模型而言，广泛适用于各种录音设备是关键性能因素。不同类型的麦克风特性由于其不同的频率响应，会引入数字化音频信号的分布差异。如果在训练期间不考虑此领域偏移，那么当它用于未见过的设备记录音频时，模型的性能可能会严重下降。特别是，在少数不同麦克风上录制音频信号的模型训练可能会使泛化到未被训练的设备困难。为解决这个问题，我们使用预录制设备脉冲响应(DIR)卷积训练集中的音频信号，从而人工增加录音设备的多样性。我们系统地研究了使用CNN和音频光谱变换进行Acoustic Scene Classification任务的DIR增强效果。结果表明，仅使用DIR增强就能提升模型在未见过设备上的性能。

    The ability to generalize to a wide range of recording devices is a crucial performance factor for audio classification models. The characteristics of different types of microphones introduce distributional shifts in the digitized audio signals due to their varying frequency responses. If this domain shift is not taken into account during training, the model's performance could degrade severely when it is applied to signals recorded by unseen devices. In particular, training a model on audio signals recorded with a small number of different microphones can make generalization to unseen devices difficult. To tackle this problem, we convolve audio signals in the training set with pre-recorded device impulse responses (DIRs) to artificially increase the diversity of recording devices. We systematically study the effect of DIR augmentation on the task of Acoustic Scene Classification using CNNs and Audio Spectrogram Transformers. The results show that DIR augmentation in isolation performs
    
[^11]: 导航的中层表示——虚拟导航

    Virtual Guidance as a Mid-level Representation for Navigation. (arXiv:2303.02731v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.02731](http://arxiv.org/abs/2303.02731)

    该论文介绍了一种名为“虚拟导航”的新技术，通过在智能体的相机视图上叠加彩色路径或球的形式的视觉指引，以易于理解的导航指令传达抽象的导航信息。实验结果表明，在模拟和真实环境中，虚拟导航在遵循计划路径和避开障碍物等多个指标上优于现有方法。

    

    在自主导航的背景下，有效地传达抽象的导航指引给动态环境中的智能体存在挑战，特别是当导航信息是多模态的时候。为了解决这个问题，本文引入了一种名为“虚拟导航”的新技术，旨在以视觉方式呈现非视觉指令信号。这些视觉指引以彩色路径或球的形式叠加在智能体的相机视图上，作为易于理解的导航指令。我们通过在模拟和真实环境中进行实验来评估我们提出的方法。在模拟环境中，我们的虚拟导航在多项指标上优于基线混合方法，包括遵循计划路径和避开障碍物。此外，我们将虚拟导航的概念扩展到将基于文本提示的指令转换为用于真实环境实验的直观视觉格式。我们的结果验证了虚拟导航的适应性。

    In the context of autonomous navigation, effectively conveying abstract navigational cues to agents in dynamic environments poses challenges, particularly when the navigation information is multimodal. To address this issue, the paper introduces a novel technique termed "Virtual Guidance," which is designed to visually represent non-visual instructional signals. These visual cues, rendered as colored paths or spheres, are overlaid onto the agent's camera view, serving as easily comprehensible navigational instructions. We evaluate our proposed method through experiments in both simulated and real-world settings. In the simulated environments, our virtual guidance outperforms baseline hybrid approaches in several metrics, including adherence to planned routes and obstacle avoidance. Furthermore, we extend the concept of virtual guidance to transform text-prompt-based instructions into a visually intuitive format for real-world experiments. Our results validate the adaptability of virtua
    
[^12]: Logit-Q动力学对于随机团队中的高效学习

    Logit-Q Dynamics for Efficient Learning in Stochastic Teams. (arXiv:2302.09806v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.09806](http://arxiv.org/abs/2302.09806)

    本文提出了两种Logit-Q学习动力学，通过将经典和独立的对数线性学习更新与在政策上的值迭代更新相结合，实现了在随机博弈中的高效学习。通过对比和量化分析，证明了该动力学在随机团队中可以达到（接近）高效均衡。

    

    我们提出了两种Logit-Q学习动力学，将经典和独立的对数线性学习更新与一个在政策上的值迭代更新相结合，以实现在随机博弈中的高效学习。我们证明所提出的Logit-Q动力学在随机团队中达到（接近）高效均衡。我们量化了近似误差的上界。我们还展示了Logit-Q动力学对纯定态策略的合理性，并证明了动力学在奖励函数导致潜在博弈的随机博弈中的收敛性，然而只有一个智能体控制状态转换超出随机团队。关键思路是将动力学与一个虚构的场景近似，其中Q函数估计仅在有限长度的纪元中是定态的，仅用于分析。然后，我们将主要场景和虚构场景中的动力学耦合起来，以展示这两个场景由于逐步减小的步长而越来越相似。

    We present two logit-Q learning dynamics combining the classical and independent log-linear learning updates with an on-policy value iteration update for efficient learning in stochastic games. We show that the logit-Q dynamics presented reach (near) efficient equilibrium in stochastic teams. We quantify a bound on the approximation error. We also show the rationality of the logit-Q dynamics against agents following pure stationary strategies and the convergence of the dynamics in stochastic games where the reward functions induce potential games, yet only a single agent controls the state transitions beyond stochastic teams. The key idea is to approximate the dynamics with a fictional scenario where the Q-function estimates are stationary over finite-length epochs only for analysis. We then couple the dynamics in the main and fictional scenarios to show that these two scenarios become more and more similar across epochs due to the vanishing step size.
    

