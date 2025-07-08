# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance](https://arxiv.org/abs/2403.17377) | 提出了一种名为扰动注意力引导（PAG）的新型抽样引导技术，通过在扩散 U-Net 中替换自注意力映射来生成结构降级的中间样本，从而在无条件和有条件设置下改善扩散样本质量。 |
| [^2] | [Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem](https://arxiv.org/abs/2403.03593) | 介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入 |
| [^3] | [VGMShield: Mitigating Misuse of Video Generative Models](https://arxiv.org/abs/2402.13126) | VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。 |
| [^4] | [PIP-Net: Pedestrian Intention Prediction in the Wild](https://arxiv.org/abs/2402.12810) | PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。 |
| [^5] | [GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency](https://arxiv.org/abs/2402.08855) | GhostWriter是一个AI增强的写作设计探针，通过个性化和代理增强用户的写作体验。它利用大型语言模型（LLMs）隐式学习用户的写作风格，并允许用户通过手动样式编辑和批注来控制系统的写作风格。 |
| [^6] | [AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection.](http://arxiv.org/abs/2310.13103) | 本文提出了AVTENet框架，该框架是一个基于音频-视觉Transformer的多专家集成网络，用于在视频深度伪造检测中考虑声学和视觉操作。 |
| [^7] | [Recurrent networks recognize patterns with low-dimensional oscillations.](http://arxiv.org/abs/2310.07908) | 本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。 |
| [^8] | [Human-AI Interactions and Societal Pitfalls.](http://arxiv.org/abs/2309.10448) | 本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。 |
| [^9] | [End-to-End Evaluation for Low-Latency Simultaneous Speech Translation.](http://arxiv.org/abs/2308.03415) | 本文提出了一个端到端的评估框架，用于评估低延迟语音翻译的各个方面。通过该框架，我们比较了不同方法的性能，并进行了全面的评估。 |
| [^10] | [Relation-Aware Network with Attention-Based Loss for Few-Shot Knowledge Graph Completion.](http://arxiv.org/abs/2306.09519) | 本文提出了一种新颖的RANA框架，利用有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用负样本并缓解零损失问题，同时设计了一种动态的关系感知实体编码来捕获不同关系下实体的不同表示。 |
| [^11] | [Generative Logic with Time: Beyond Logical Consistency and Statistical Possibility.](http://arxiv.org/abs/2301.08509) | 本文提出了一种将数据和逻辑结合起来进行推理的理论，解决了符号知识的概率推理问题，并在定位问题中展示出机器人可以完全数据驱动地解决这一问题。 |
| [^12] | [Normality-Guided Distributional Reinforcement Learning for Continuous Control.](http://arxiv.org/abs/2208.13125) | 本论文研究了连续控制任务中的值分布，并发现学习的值分布与正态分布非常接近。基于这一观察，提出了一种正态引导的分布式强化学习方法，利用方差网络预测的方差和回报，以及与标准值函数不同的值分布结构特征来更新策略。这种方法在两种在线算法上产生了显著效果。 |

# 详细

[^1]: 具有扰动注意力引导的自矫正扩散抽样

    Self-Rectifying Diffusion Sampling with Perturbed-Attention Guidance

    [https://arxiv.org/abs/2403.17377](https://arxiv.org/abs/2403.17377)

    提出了一种名为扰动注意力引导（PAG）的新型抽样引导技术，通过在扩散 U-Net 中替换自注意力映射来生成结构降级的中间样本，从而在无条件和有条件设置下改善扩散样本质量。

    

    近期研究表明，扩散模型能够生成高质量样本，但其质量很大程度上依赖于抽样引导技术，比如分类器引导（CG）和无分类器引导（CFG）。这些技术通常在无条件生成或各种下游任务如图像恢复中无法应用。本文提出了一种新颖的抽样引导技术，称为扰动注意力引导（PAG），它改进了扩散样本的质量，不管是在无条件还是有条件的设置中，都能实现这一目标，而不需要额外训练或整合外部模块。PAG 旨在通过整个去噪过程逐步增强样本的结构。它涉及通过用恒等矩阵替换扩散 U-Net 中选择的自注意力映射生成结构降级的中间样本，考虑自注意力机制。

    arXiv:2403.17377v1 Announce Type: cross  Abstract: Recent studies have demonstrated that diffusion models are capable of generating high-quality samples, but their quality heavily depends on sampling guidance techniques, such as classifier guidance (CG) and classifier-free guidance (CFG). These techniques are often not applicable in unconditional generation or in various downstream tasks such as image restoration. In this paper, we propose a novel sampling guidance, called Perturbed-Attention Guidance (PAG), which improves diffusion sample quality across both unconditional and conditional settings, achieving this without requiring additional training or the integration of external modules. PAG is designed to progressively enhance the structure of samples throughout the denoising process. It involves generating intermediate samples with degraded structure by substituting selected self-attention maps in diffusion U-Net with an identity matrix, by considering the self-attention mechanisms
    
[^2]: 您信任您的模型吗？深度学习生态系统中新兴的恶意软件威胁

    Do You Trust Your Model? Emerging Malware Threats in the Deep Learning Ecosystem

    [https://arxiv.org/abs/2403.03593](https://arxiv.org/abs/2403.03593)

    介绍了MaleficNet 2.0，一种在神经网络中嵌入恶意软件的新技术，其注入技术具有隐蔽性，不会降低模型性能，并且对神经网络参数中的恶意有效负载进行注入

    

    训练高质量的深度学习模型是一项具有挑战性的任务，这是因为需要计算和技术要求。越来越多的个人、机构和公司越来越多地依赖于在公共代码库中提供的预训练的第三方模型。这些模型通常直接使用或集成到产品管道中而没有特殊的预防措施，因为它们实际上只是以张量形式的数据，被认为是安全的。在本文中，我们提出了一种针对神经网络的新的机器学习供应链威胁。我们介绍了MaleficNet 2.0，一种在神经网络中嵌入自解压自执行恶意软件的新技术。MaleficNet 2.0使用扩频信道编码结合纠错技术在深度神经网络的参数中注入恶意有效载荷。MaleficNet 2.0注入技术具有隐蔽性，不会降低模型的性能，并且对...

    arXiv:2403.03593v1 Announce Type: cross  Abstract: Training high-quality deep learning models is a challenging task due to computational and technical requirements. A growing number of individuals, institutions, and companies increasingly rely on pre-trained, third-party models made available in public repositories. These models are often used directly or integrated in product pipelines with no particular precautions, since they are effectively just data in tensor form and considered safe. In this paper, we raise awareness of a new machine learning supply chain threat targeting neural networks. We introduce MaleficNet 2.0, a novel technique to embed self-extracting, self-executing malware in neural networks. MaleficNet 2.0 uses spread-spectrum channel coding combined with error correction techniques to inject malicious payloads in the parameters of deep neural networks. MaleficNet 2.0 injection technique is stealthy, does not degrade the performance of the model, and is robust against 
    
[^3]: VGMShield：缓解视频生成模型的误用

    VGMShield: Mitigating Misuse of Video Generative Models

    [https://arxiv.org/abs/2402.13126](https://arxiv.org/abs/2402.13126)

    VGMShield提出了三项简单但开创性的措施，通过检测虚假视频、溯源问题和利用预训练的空间-时间动态模型，防范视频生成模型的误用。

    

    随着视频生成技术的快速发展，人们可以方便地利用视频生成模型创建符合其特定需求的视频。然而，人们也越来越担心这些技术被用于创作和传播虚假信息。在这项工作中，我们介绍了VGMShield：一套包含三项直接但开创性的措施，用于防范虚假视频生成过程中可能出现的问题。我们首先从“虚假视频检测”开始，尝试理解生成的视频中是否存在独特性，以及我们是否能够区分它们与真实视频的不同；然后，我们探讨“溯源”问题，即将一段虚假视频追溯回生成它的模型。为此，我们提出利用预训练的关注“时空动态”的模型作为骨干，以识别视频中的不一致性。通过对七个最先进的开源模型进行实验，我们证明了...

    arXiv:2402.13126v1 Announce Type: cross  Abstract: With the rapid advancement in video generation, people can conveniently utilize video generation models to create videos tailored to their specific desires. Nevertheless, there are also growing concerns about their potential misuse in creating and disseminating false information.   In this work, we introduce VGMShield: a set of three straightforward but pioneering mitigations through the lifecycle of fake video generation. We start from \textit{fake video detection} trying to understand whether there is uniqueness in generated videos and whether we can differentiate them from real videos; then, we investigate the \textit{tracing} problem, which maps a fake video back to a model that generates it. Towards these, we propose to leverage pre-trained models that focus on {\it spatial-temporal dynamics} as the backbone to identify inconsistencies in videos. Through experiments on seven state-of-the-art open-source models, we demonstrate that
    
[^4]: PIP-Net：城市中行人意图预测

    PIP-Net: Pedestrian Intention Prediction in the Wild

    [https://arxiv.org/abs/2402.12810](https://arxiv.org/abs/2402.12810)

    PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。

    

    精准的自动驾驶车辆（AVs）对行人意图的预测是当前该领域的一项研究挑战。在本文中，我们介绍了PIP-Net，这是一个新颖的框架，旨在预测AVs在现实世界城市场景中的行人过马路意图。我们提供了两种针对不同摄像头安装和设置设计的PIP-Net变种。利用来自行驶场景的动力学数据和空间特征，所提出的模型采用循环和时间注意力机制的解决方案，性能优于现有技术。为了增强道路用户的视觉表示及其与自车的相关性，我们引入了一个分类深度特征图，结合局部运动流特征，为场景动态提供丰富的洞察。此外，我们探讨了将摄像头的视野从一个扩展到围绕自车的三个摄像头的影响，以提升

    arXiv:2402.12810v1 Announce Type: cross  Abstract: Accurate pedestrian intention prediction (PIP) by Autonomous Vehicles (AVs) is one of the current research challenges in this field. In this article, we introduce PIP-Net, a novel framework designed to predict pedestrian crossing intentions by AVs in real-world urban scenarios. We offer two variants of PIP-Net designed for different camera mounts and setups. Leveraging both kinematic data and spatial features from the driving scene, the proposed model employs a recurrent and temporal attention-based solution, outperforming state-of-the-art performance. To enhance the visual representation of road users and their proximity to the ego vehicle, we introduce a categorical depth feature map, combined with a local motion flow feature, providing rich insights into the scene dynamics. Additionally, we explore the impact of expanding the camera's field of view, from one to three cameras surrounding the ego vehicle, leading to enhancement in the
    
[^5]: GhostWriter:通过个性化和代理增强协作人工智能写作体验

    GhostWriter: Augmenting Collaborative Human-AI Writing Experiences Through Personalization and Agency

    [https://arxiv.org/abs/2402.08855](https://arxiv.org/abs/2402.08855)

    GhostWriter是一个AI增强的写作设计探针，通过个性化和代理增强用户的写作体验。它利用大型语言模型（LLMs）隐式学习用户的写作风格，并允许用户通过手动样式编辑和批注来控制系统的写作风格。

    

    大型语言模型（LLMs）在提供不同形式的写作辅助方面越来越流行，并且具有无处不在的应用。然而，由于个性化和控制能力有限，LLM驱动的写作系统可能会使用户感到沮丧，当用户缺乏提示工程经验时，这种情况可能加剧。我们认为设计可以解决这些挑战之一，并引入GhostWriter，这是一个AI增强的写作设计探针，用户可以通过增强的代理和个性化来进行写作。GhostWriter利用LLMs在用户编写的过程中隐式学习用户所期望的写作风格，同时允许通过手动样式编辑和批注进行显式教学。我们研究了18名参与者在两个不同的写作任务中使用GhostWriter，观察到它帮助用户编写个性化的文本生成，并通过提供多种方式控制系统的写作风格来增强用户的能力。从这项研究中，我们提出了一些见解。

    arXiv:2402.08855v1 Announce Type: cross Abstract: Large language models (LLMs) are becoming more prevalent and have found a ubiquitous use in providing different forms of writing assistance. However, LLM-powered writing systems can frustrate users due to their limited personalization and control, which can be exacerbated when users lack experience with prompt engineering. We see design as one way to address these challenges and introduce GhostWriter, an AI-enhanced writing design probe where users can exercise enhanced agency and personalization. GhostWriter leverages LLMs to learn the user's intended writing style implicitly as they write, while allowing explicit teaching moments through manual style edits and annotations. We study 18 participants who use GhostWriter on two different writing tasks, observing that it helps users craft personalized text generations and empowers them by providing multiple ways to control the system's writing style. From this study, we present insights re
    
[^6]: AVTENet: 基于音频-视觉Transformer的多专家集成网络在视频深度伪造检测中的应用

    AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection. (arXiv:2310.13103v1 [cs.CV])

    [http://arxiv.org/abs/2310.13103](http://arxiv.org/abs/2310.13103)

    本文提出了AVTENet框架，该框架是一个基于音频-视觉Transformer的多专家集成网络，用于在视频深度伪造检测中考虑声学和视觉操作。

    

    在社交媒体平台上广泛分享的伪造内容是一个重大社会问题，要求加强监管并给研究社区带来新的挑战。近年来，超真实的深度伪造视频的普及引起了对音频和视觉伪造威胁的关注。大多数关于检测AI生成的伪造视频的先前工作只利用了视觉模态或音频模态。虽然文献中有一些方法利用音频和视觉模态来检测伪造视频，但它们尚未在涉及声学和视觉操作的多模态深度伪造视频数据集上进行全面评估。此外，这些现有方法大多基于CNN，并且检测准确率较低。受到Transformer在各个领域的最新成功启发，为了解决深度伪造技术带来的挑战，本文提出了一种考虑声学操作的音频-视觉Transformer集成网络（AVTENet）框架。

    Forged content shared widely on social media platforms is a major social problem that requires increased regulation and poses new challenges to the research community. The recent proliferation of hyper-realistic deepfake videos has drawn attention to the threat of audio and visual forgeries. Most previous work on detecting AI-generated fake videos only utilizes visual modality or audio modality. While there are some methods in the literature that exploit audio and visual modalities to detect forged videos, they have not been comprehensively evaluated on multi-modal datasets of deepfake videos involving acoustic and visual manipulations. Moreover, these existing methods are mostly based on CNN and suffer from low detection accuracy. Inspired by the recent success of Transformer in various fields, to address the challenges posed by deepfake technology, in this paper, we propose an Audio-Visual Transformer-based Ensemble Network (AVTENet) framework that considers both acoustic manipulatio
    
[^7]: 循环网络通过低维振荡识别模式

    Recurrent networks recognize patterns with low-dimensional oscillations. (arXiv:2310.07908v1 [q-bio.NC])

    [http://arxiv.org/abs/2310.07908](http://arxiv.org/abs/2310.07908)

    本研究提出了一种通过相位变化识别模式的循环神经网络机制，并通过验证手工制作的振荡模型证实了这一解释。该研究不仅提供了一种潜在的动力学机制用于模式识别，还暗示了有限状态自动机的神经实现方式，并且对深度学习模型的可解释性进行了贡献。

    

    本研究提出了一种通过解释在SET卡牌游戏启发下进行训练的循环神经网络(RNN)在简单任务上的动力学机制来识别模式。我们将训练后的RNN解释为通过低维极限环中的相位变化进行模式识别，类似于有限状态自动机(FSA)中的转换。我们进一步通过手工制作一个简单的振荡模型来验证了这一解释，该模型复制了训练后的RNN的动力学特性。我们的发现不仅暗示了一种潜在的动力学机制能够实现模式识别，还暗示了一种有限状态自动机的潜在神经实现。最重要的是，这项工作有助于关于深度学习模型可解释性的讨论。

    This study proposes a novel dynamical mechanism for pattern recognition discovered by interpreting a recurrent neural network (RNN) trained on a simple task inspired by the SET card game. We interpreted the trained RNN as recognizing patterns via phase shifts in a low-dimensional limit cycle in a manner analogous to transitions in a finite state automaton (FSA). We further validated this interpretation by handcrafting a simple oscillatory model that reproduces the dynamics of the trained RNN. Our findings not only suggest of a potential dynamical mechanism capable of pattern recognition, but also suggest of a potential neural implementation of FSA. Above all, this work contributes to the growing discourse on deep learning model interpretability.
    
[^8]: 人工智能与人类互动以及社会陷阱

    Human-AI Interactions and Societal Pitfalls. (arXiv:2309.10448v1 [cs.AI])

    [http://arxiv.org/abs/2309.10448](http://arxiv.org/abs/2309.10448)

    本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。

    

    当与生成式人工智能（AI）合作时，用户可能会看到生产力的提升，但AI生成的内容可能不完全符合他们的偏好。为了研究这种影响，我们引入了一个贝叶斯框架，其中异质用户选择与AI共享多少信息，面临输出保真度和通信成本之间的权衡。我们展示了这些个体决策与AI训练之间的相互作用可能导致社会挑战。输出可能变得更加同质化，特别是当AI在AI生成的内容上进行训练时。而任何AI的偏见可能成为社会偏见。解决同质化和偏见问题的办法是改进人工智能与人类的互动，实现个性化输出而不牺牲生产力。

    When working with generative artificial intelligence (AI), users may see productivity gains, but the AI-generated content may not match their preferences exactly. To study this effect, we introduce a Bayesian framework in which heterogeneous users choose how much information to share with the AI, facing a trade-off between output fidelity and communication cost. We show that the interplay between these individual-level decisions and AI training may lead to societal challenges. Outputs may become more homogenized, especially when the AI is trained on AI-generated content. And any AI bias may become societal bias. A solution to the homogenization and bias issues is to improve human-AI interactions, enabling personalized outputs without sacrificing productivity.
    
[^9]: 低延迟同时语音翻译的端到端评估

    End-to-End Evaluation for Low-Latency Simultaneous Speech Translation. (arXiv:2308.03415v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.03415](http://arxiv.org/abs/2308.03415)

    本文提出了一个端到端的评估框架，用于评估低延迟语音翻译的各个方面。通过该框架，我们比较了不同方法的性能，并进行了全面的评估。

    

    近年来，低延迟语音翻译的挑战引起了研究界的广泛关注，许多出版物和共享任务也证明了这一点。因此，在实际场景中评估这些不同的方法非常重要。然而，目前只有系统的特定方面被评估，并且往往无法比较不同的方法。在这项工作中，我们提出了第一个在实际条件下执行和评估低延迟语音翻译各个方面的框架。评估是以端到端的方式进行的，包括音频的分段以及不同组成部分的运行时间。其次，我们使用该框架比较了不同的低延迟语音翻译方法。我们评估了具有修订输出选项的模型以及具有固定输出方法。此外，我们直接比较了最先进的级联系统和端到端系统。最后，该框架基于一个统一的度量来评估低延迟语音翻译性能，并提供了一个全面的评估结果。

    The challenge of low-latency speech translation has recently draw significant interest in the research community as shown by several publications and shared tasks. Therefore, it is essential to evaluate these different approaches in realistic scenarios. However, currently only specific aspects of the systems are evaluated and often it is not possible to compare different approaches.  In this work, we propose the first framework to perform and evaluate the various aspects of low-latency speech translation under realistic conditions. The evaluation is carried out in an end-to-end fashion. This includes the segmentation of the audio as well as the run-time of the different components.  Secondly, we compare different approaches to low-latency speech translation using this framework. We evaluate models with the option to revise the output as well as methods with fixed output. Furthermore, we directly compare state-of-the-art cascaded as well as end-to-end systems. Finally, the framework all
    
[^10]: 关系感知网络基于注意力损失的小样本知识图谱补全

    Relation-Aware Network with Attention-Based Loss for Few-Shot Knowledge Graph Completion. (arXiv:2306.09519v1 [cs.CL])

    [http://arxiv.org/abs/2306.09519](http://arxiv.org/abs/2306.09519)

    本文提出了一种新颖的RANA框架，利用有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用负样本并缓解零损失问题，同时设计了一种动态的关系感知实体编码来捕获不同关系下实体的不同表示。

    

    小样本知识图谱补全旨在利用少量参考实体对预测关系的未见事实。现有方法随机选择一个负采样来最小化基于边界的排名损失，但这容易导致零损失问题。此外，实体在不同的上下文中应该具有不同的表征。为了解决这些问题，我们提出了一种新颖的关系感知网络基于注意力损失的框架。具体而言，我们通过有策略地选择相关负样本和设计基于注意力机制的损失函数来更好地利用丰富的负样本并缓解零损失问题。直觉上，与正样本更相似的负样本将对模型贡献更大。此外，我们设计了一种动态的关系感知实体编码来捕捉不同关系下实体的不同表示。三个基准数据集上的实验结果表明，相比最先进的方法，所提出的RANA框架的有效性。

    Few-shot knowledge graph completion (FKGC) task aims to predict unseen facts of a relation with few-shot reference entity pairs. Current approaches randomly select one negative sample for each reference entity pair to minimize a margin-based ranking loss, which easily leads to a zero-loss problem if the negative sample is far away from the positive sample and then out of the margin. Moreover, the entity should have a different representation under a different context. To tackle these issues, we propose a novel Relation-Aware Network with Attention-Based Loss (RANA) framework. Specifically, to better utilize the plentiful negative samples and alleviate the zero-loss issue, we strategically select relevant negative samples and design an attention-based loss function to further differentiate the importance of each negative sample. The intuition is that negative samples more similar to positive samples will contribute more to the model. Further, we design a dynamic relation-aware entity en
    
[^11]: 带有时间的生成逻辑：超越逻辑一致性和统计可能性

    Generative Logic with Time: Beyond Logical Consistency and Statistical Possibility. (arXiv:2301.08509v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2301.08509](http://arxiv.org/abs/2301.08509)

    本文提出了一种将数据和逻辑结合起来进行推理的理论，解决了符号知识的概率推理问题，并在定位问题中展示出机器人可以完全数据驱动地解决这一问题。

    

    本文提出了一种简单的推理理论，可以从数据中完全逻辑推理符号知识。我们采用贝叶斯方法来建模数据如何引起符号知识。符号知识的概率推理被建模为正向和反向过程，分别对应形式逻辑的解释和逆解释。该理论应用于定位问题，展示了一个具有损坏或噪声传感器的机器人可以以完全数据驱动的方式有效地解决该问题。

    This paper gives a simple theory of inference to logically reason symbolic knowledge fully from data over time. We take a Bayesian approach to model how data causes symbolic knowledge. Probabilistic reasoning with symbolic knowledge is modelled as a process of going the causality forwards and backwards. The forward and backward processes correspond to an interpretation and inverse interpretation of formal logic, respectively. The theory is applied to a localisation problem to show a robot with broken or noisy sensors can efficiently solve the problem in a fully data-driven fashion.
    
[^12]: 连续控制的正常引导分布强化学习

    Normality-Guided Distributional Reinforcement Learning for Continuous Control. (arXiv:2208.13125v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.13125](http://arxiv.org/abs/2208.13125)

    本论文研究了连续控制任务中的值分布，并发现学习的值分布与正态分布非常接近。基于这一观察，提出了一种正态引导的分布式强化学习方法，利用方差网络预测的方差和回报，以及与标准值函数不同的值分布结构特征来更新策略。这种方法在两种在线算法上产生了显著效果。

    

    在许多强化学习算法中，学习一个预测回报的均值模型，或价值函数，起着关键作用。分布式强化学习(DRL)通过建模值分布而不仅仅是均值来提高性能。我们研究了几个连续控制任务中的值分布，并发现学习的值分布与正态分布非常接近。我们设计了一种利用这个性质的方法，利用从方差网络预测的方差，以及回报，来分析计算代表我们分布式值函数的正态分布的目标分位栏。此外，我们提出了一种基于值分布的结构特征的正确性来衡量的策略更新方法，这些特征在标准的值函数中不存在。我们概述的方法与许多DRL结构兼容。我们使用两种代表性的在线算法，PPO和TRPO，作为测试平台。我们的方法在统计上产生了显著的效果。

    Learning a predictive model of the mean return, or value function, plays a critical role in many reinforcement learning algorithms. Distributional reinforcement learning (DRL) has been shown to improve performance by modeling the value distribution, not just the mean. We study the value distribution in several continuous control tasks and find that the learned value distribution is empirical quite close to normal. We design a method that exploits this property, employ variances predicted from a variance network, along with returns, to analytically compute target quantile bars representing a normal for our distributional value function. In addition, we propose a policy update strategy based on the correctness as measured by structural characteristics of the value distribution not present in the standard value function. The approach we outline is compatible with many DRL structures. We use two representative on-policy algorithms, PPO and TRPO, as testbeds. Our method yields statistically
    

