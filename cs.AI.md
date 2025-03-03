# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033) | 本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。 |
| [^2] | [Non-Stationary Latent Auto-Regressive Bandits](https://arxiv.org/abs/2402.03110) | 本文提出了非平稳潜在自回归赌博机问题，并提出了一个算法，在这种环境下可以达到较低的遗憾率。 |
| [^3] | [Are Generative AI systems Capable of Supporting Information Needs of Patients?](https://arxiv.org/abs/2402.00234) | 生成式AI系统被应用于支持患者信息需求的研究中，以提高患者对放射学数据的理解和管理能力。通过与患者和医疗专家的对话分析，我们确定了常见的医学信息需求和问题。 |
| [^4] | [GOAT-Bench: Safety Insights to Large Multimodal Models through Meme-Based Social Abuse.](http://arxiv.org/abs/2401.01523) | 通过基于迷因的社交虐待研究对大型多模态模型的安全洞察，我们引入了综合的迷因基准测试集GOAT-Bench，评估各种LMMs在识别和回应迷因中体现的微妙社交虐待方面的能力。 |
| [^5] | [All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment.](http://arxiv.org/abs/2307.03373) | 本文提出了一种一体化视觉-语言跟踪框架，采用统一的Transformer主干网络，实现联合特征提取和交互，提高了在复杂场景下的目标感知能力。 |

# 详细

[^1]: Fiddler：用于Mixture-of-Experts模型快速推断的CPU-GPU编排

    Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

    [https://arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033)

    本文介绍了Fiddler，一种用于Mixture-of-Experts模型的资源高效推断引擎，通过CPU-GPU编排实现最小化数据传输，相比现有方法提高了一个数量级的推断速度。

    

    基于Mixture-of-Experts（MoE）架构的大型语言模型（LLM）在各种任务上表现出了很好的性能。然而，在资源受限的环境下运行这些模型，即GPU内存资源不丰富的情况下，由于模型规模庞大，存在挑战。现有的将模型权重卸载到CPU内存的系统，由于频繁地在CPU和GPU之间移动数据而导致显著的开销。在本文中，我们提出了Fiddler，一种用于MoE模型的资源高效推断引擎，实现了CPU-GPU编排。Fiddler的核心思想是利用CPU的计算能力来最小化CPU和GPU之间的数据传输。我们的评估结果表明，Fiddler能够在单个具有24GB内存的GPU上运行未压缩的Mixtral-8x7B模型（参数超过90GB），每秒生成超过3个token，相比现有方法提高一个数量级。Fiddler的代码可以公开访问，网址为\url{https://github.com/efeslab/fiddler}

    Large Language Models (LLMs) based on Mixture-of-Experts (MoE) architecture are showing promising performance on various tasks. However, running them on resource-constrained settings, where GPU memory resources are not abundant, is challenging due to huge model sizes. Existing systems that offload model weights to CPU memory suffer from the significant overhead of frequently moving data between CPU and GPU. In this paper, we propose Fiddler, a resource-efficient inference engine with CPU-GPU orchestration for MoE models. The key idea of Fiddler is to use the computation ability of the CPU to minimize the data movement between the CPU and GPU. Our evaluation shows that Fiddler can run the uncompressed Mixtral-8x7B model, which exceeds 90GB in parameters, to generate over $3$ tokens per second on a single GPU with 24GB memory, showing an order of magnitude improvement over existing methods. The code of Fiddler is publicly available at \url{https://github.com/efeslab/fiddler}
    
[^2]: 非平稳潜在自回归赌博机

    Non-Stationary Latent Auto-Regressive Bandits

    [https://arxiv.org/abs/2402.03110](https://arxiv.org/abs/2402.03110)

    本文提出了非平稳潜在自回归赌博机问题，并提出了一个算法，在这种环境下可以达到较低的遗憾率。

    

    本文考虑具有非平稳奖励的随机多臂赌博机问题。我们提出了一个新颖的非平稳环境的公式，其中臂的平均奖励随时间变化是由一些未知的潜在自回归(AR)状态的顺序k决定的。我们将这个新的环境称为潜在AR赌博机。潜在AR赌博机的不同形式在许多现实世界的场景中都出现，特别是在行为健康或教育等新兴科学领域中，这里缺乏对环境的机制建模。如果AR顺序k已知，我们提出了一个算法，在这种情况下，算法表现出O(k√T)的遗憾率。实证结果显示，即使k被错误地估计，我们的算法在多个非平稳环境中也胜过标准的UCB算法。

    We consider the stochastic multi-armed bandit problem with non-stationary rewards. We present a novel formulation of non-stationarity in the environment where changes in the mean reward of the arms over time are due to some unknown, latent, auto-regressive (AR) state of order $k$. We call this new environment the latent AR bandit. Different forms of the latent AR bandit appear in many real-world settings, especially in emerging scientific fields such as behavioral health or education where there are few mechanistic models of the environment. If the AR order $k$ is known, we propose an algorithm that achieves $\tilde{O}(k\sqrt{T})$ regret in this setting. Empirically, our algorithm outperforms standard UCB across multiple non-stationary environments, even if $k$ is mis-specified.
    
[^3]: 生成式AI系统能否支持患者的信息需求？

    Are Generative AI systems Capable of Supporting Information Needs of Patients?

    [https://arxiv.org/abs/2402.00234](https://arxiv.org/abs/2402.00234)

    生成式AI系统被应用于支持患者信息需求的研究中，以提高患者对放射学数据的理解和管理能力。通过与患者和医疗专家的对话分析，我们确定了常见的医学信息需求和问题。

    

    患有复杂疾病如癌症的患者面临复杂的信息挑战，他们不仅需要了解他们的疾病，还需要学会如何管理它。与医疗专家（放射科医师、肿瘤科医师）密切互动可以提高患者的学习能力，从而改善疾病预后。然而，这种方法资源密集且占用了专家的时间，使他们无法完成其他关键任务。鉴于生成式AI模型在改进医疗系统方面的最新进展，我们的工作研究了生成式视觉问答系统在放射学成像数据背景下如何负责任地支持患者的信息需求。我们进行了一项形成性需求发现研究，参与者讨论了一个虚构近亲的胸部计算机断层扫描（CT）图像和相关的放射学报告。通过对参与者和医疗专家之间的对话的主题分析，我们确定常见的医学信息需求和问题。

    Patients managing a complex illness such as cancer face a complex information challenge where they not only must learn about their illness but also how to manage it. Close interaction with healthcare experts (radiologists, oncologists) can improve patient learning and thereby, their disease outcome. However, this approach is resource intensive and takes expert time away from other critical tasks. Given the recent advancements in Generative AI models aimed at improving the healthcare system, our work investigates whether and how generative visual question answering systems can responsibly support patient information needs in the context of radiology imaging data. We conducted a formative need-finding study in which participants discussed chest computed tomography (CT) scans and associated radiology reports of a fictitious close relative with a cardiothoracic radiologist. Using thematic analysis of the conversation between participants and medical experts, we identified commonly occurrin
    
[^4]: GOAT-Bench: 通过基于迷因的社交虐待研究对大型多模态模型的安全洞察

    GOAT-Bench: Safety Insights to Large Multimodal Models through Meme-Based Social Abuse. (arXiv:2401.01523v1 [cs.CL])

    [http://arxiv.org/abs/2401.01523](http://arxiv.org/abs/2401.01523)

    通过基于迷因的社交虐待研究对大型多模态模型的安全洞察，我们引入了综合的迷因基准测试集GOAT-Bench，评估各种LMMs在识别和回应迷因中体现的微妙社交虐待方面的能力。

    

    社交媒体的指数级增长深刻改变了信息的创造、传播和吸收方式，在数字时代产生了前所未有的影响。遗憾的是，这个爆炸也导致了网络迷因的滥用数量显著增加。评估迷因的负面影响是相当具有挑战性的，因为它们通常具有微妙和隐晦的含义，这些含义不能直接通过显性的文本和图像传达出来。鉴于此，大型多模态模型(LMMs)作为处理多样化多模态任务的卓越能力的焦点引起了人们的兴趣。针对这一发展，我们的论文旨在深入研究各种LMMs(如GPT-4V)识别和回应迷因中体现的微妙社交虐待方面的能力。我们引入了综合的迷因基准测试集GOAT-Bench，其中包含超过6K个多样的迷因，涵盖的主题包括隐性仇恨言论、性别歧视和网络欺凌等。利用GOAT-Be

    The exponential growth of social media has profoundly transformed how information is created, disseminated, and absorbed, exceeding any precedent in the digital age. Regrettably, this explosion has also spawned a significant increase in the online abuse of memes. Evaluating the negative impact of memes is notably challenging, owing to their often subtle and implicit meanings, which are not directly conveyed through the overt text and imagery. In light of this, large multimodal models (LMMs) have emerged as a focal point of interest due to their remarkable capabilities in handling diverse multimodal tasks. In response to this development, our paper aims to thoroughly examine the capacity of various LMMs (e.g. GPT-4V) to discern and respond to the nuanced aspects of social abuse manifested in memes. We introduce the comprehensive meme benchmark, GOAT-Bench, comprising over 6K varied memes encapsulating themes such as implicit hate speech, sexism, and cyberbullying, etc. Utilizing GOAT-Be
    
[^5]: 一体化视觉-语言跟踪的探索：多模态对齐

    All in One: Exploring Unified Vision-Language Tracking with Multi-Modal Alignment. (arXiv:2307.03373v1 [cs.CV])

    [http://arxiv.org/abs/2307.03373](http://arxiv.org/abs/2307.03373)

    本文提出了一种一体化视觉-语言跟踪框架，采用统一的Transformer主干网络，实现联合特征提取和交互，提高了在复杂场景下的目标感知能力。

    

    当前主流的视觉-语言跟踪框架包括三个部分，即视觉特征提取器、语言特征提取器和融合模型。为了追求更好的性能，视觉-语言跟踪常常使用定制和更重的单模态编码器和多模态融合模型。尽管有效，现有的视觉-语言跟踪器将特征提取和特征集成分开，导致提取的特征缺乏语义引导，在复杂场景下具有有限的目标感知能力，例如相似的干扰物和极端光照。在这项研究中，受到近期在自然语言和计算机视觉任务中统一架构探索的成功启发，我们提出了一种一体化框架，通过采用统一的Transformer主干网络来学习联合特征提取和交互。具体而言，我们混合原始的视觉和语言信号来生成注入语言的视觉单元，然后将它们连接起来。

    Current mainstream vision-language (VL) tracking framework consists of three parts, \ie a visual feature extractor, a language feature extractor, and a fusion model. To pursue better performance, a natural modus operandi for VL tracking is employing customized and heavier unimodal encoders, and multi-modal fusion models. Albeit effective, existing VL trackers separate feature extraction and feature integration, resulting in extracted features that lack semantic guidance and have limited target-aware capability in complex scenarios, \eg similar distractors and extreme illumination. In this work, inspired by the recent success of exploring foundation models with unified architecture for both natural language and computer vision tasks, we propose an All-in-One framework, which learns joint feature extraction and interaction by adopting a unified transformer backbone. Specifically, we mix raw vision and language signals to generate language-injected vision tokens, which we then concatenate
    

