# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SERVAL: Synergy Learning between Vertical Models and LLMs towards Oracle-Level Zero-shot Medical Prediction](https://arxiv.org/abs/2403.01570) | 提出SERVAL，一个协同学习流水线，可以通过相互增强，实现LLMs和小模型的垂直能力无监督开发，从而改善领域特定垂直问题的零-shot预测能力。 |
| [^2] | [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812) | DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。 |
| [^3] | [Are Generative AI systems Capable of Supporting Information Needs of Patients?](https://arxiv.org/abs/2402.00234) | 生成式AI系统被应用于支持患者信息需求的研究中，以提高患者对放射学数据的理解和管理能力。通过与患者和医疗专家的对话分析，我们确定了常见的医学信息需求和问题。 |
| [^4] | [GOAT-Bench: Safety Insights to Large Multimodal Models through Meme-Based Social Abuse.](http://arxiv.org/abs/2401.01523) | 通过基于迷因的社交虐待研究对大型多模态模型的安全洞察，我们引入了综合的迷因基准测试集GOAT-Bench，评估各种LMMs在识别和回应迷因中体现的微妙社交虐待方面的能力。 |

# 详细

[^1]: SERVAL：垂直模型和LLM之间的协同学习，实现零-shot级别的医学预测

    SERVAL: Synergy Learning between Vertical Models and LLMs towards Oracle-Level Zero-shot Medical Prediction

    [https://arxiv.org/abs/2403.01570](https://arxiv.org/abs/2403.01570)

    提出SERVAL，一个协同学习流水线，可以通过相互增强，实现LLMs和小模型的垂直能力无监督开发，从而改善领域特定垂直问题的零-shot预测能力。

    

    近期大型语言模型（LLMs）的发展展示出对通用和常识问题卓越的零-shot能力。然而，LLMs在领域特定垂直问题上的应用仍然落后，主要是由于垂直知识方面的问题和不足。此外，垂直数据注释过程通常需要劳动密集型的专家参与，因此增加了增强模型垂直能力的额外挑战。在本文中，我们提出了SERVAL，一个协同学习流水线，旨在通过相互增强，对LLMs和小模型的垂直能力进行无监督开发。具体来说，SERVAL利用LLMs的零-shot输出作为注释，利用其置信度来从头开始教授一个强大的垂直模型。反过来，训练有素的垂直模型引导LLM微调，以增强其零-shot能力，逐步改进两者。

    arXiv:2403.01570v1 Announce Type: new  Abstract: Recent development of large language models (LLMs) has exhibited impressive zero-shot proficiency on generic and common sense questions. However, LLMs' application on domain-specific vertical questions still lags behind, primarily due to the humiliation problems and deficiencies in vertical knowledge. Furthermore, the vertical data annotation process often requires labor-intensive expert involvement, thereby presenting an additional challenge in enhancing the model's vertical capabilities. In this paper, we propose SERVAL, a synergy learning pipeline designed for unsupervised development of vertical capabilities in both LLMs and small models by mutual enhancement. Specifically, SERVAL utilizes the LLM's zero-shot outputs as annotations, leveraging its confidence to teach a robust vertical model from scratch. Reversely, the trained vertical model guides the LLM fine-tuning to enhance its zero-shot capability, progressively improving both 
    
[^2]: DropBP：通过丢弃反向传播加速大型语言模型的微调

    DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation

    [https://arxiv.org/abs/2402.17812](https://arxiv.org/abs/2402.17812)

    DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。

    

    训练深度神经网络通常涉及正向和反向传播过程中的大量计算成本。传统的层次丢弃技术在训练过程中丢弃某些层以减少计算负担。然而，在正向传播过程中丢弃层会对训练过程产生不利影响，降低准确性。本文提出了DropBP，这是一种旨在减少计算成本同时保持准确性的新方法。DropBP在反向传播过程中随机丢弃层，不影响正向传播。此外，DropBP计算每个层的敏感性以分配适当的丢失率，从而稳定训练过程。DropBP旨在通过反向传播增强训练过程的效率，从而加速使用反向传播进行完全微调和参数高效微调。

    arXiv:2402.17812v1 Announce Type: cross  Abstract: Training deep neural networks typically involves substantial computational costs during both forward and backward propagation. The conventional layer dropping techniques drop certain layers during training for reducing the computations burden. However, dropping layers during forward propagation adversely affects the training process by degrading accuracy. In this paper, we propose Dropping Backward Propagation (DropBP), a novel approach designed to reduce computational costs while maintaining accuracy. DropBP randomly drops layers during the backward propagation, which does not deviate forward propagation. Moreover, DropBP calculates the sensitivity of each layer to assign appropriate drop rate, thereby stabilizing the training process. DropBP is designed to enhance the efficiency of the training process with backpropagation, thereby enabling the acceleration of both full fine-tuning and parameter-efficient fine-tuning using backpropag
    
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
    

