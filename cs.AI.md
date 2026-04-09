# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Teach Large Language Models to Forget Privacy.](http://arxiv.org/abs/2401.00870) | 这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。 |
| [^2] | [DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models.](http://arxiv.org/abs/2306.14685) | 本文介绍了DiffSketcher，一种通过隐式扩散模型实现文本引导的矢量素描合成的创新算法。DiffSketcher通过直接优化贝塞尔曲线和扩散模型损失来生成矢量化的手绘素描，并通过注意力图加快生成过程。实验结果表明DiffSketcher的素描质量高于之前方法。 |
| [^3] | [A survey of Generative AI Applications.](http://arxiv.org/abs/2306.02781) | 本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。 |

# 详细

[^1]: 教导大型语言模型忘记隐私

    Teach Large Language Models to Forget Privacy. (arXiv:2401.00870v1 [cs.CR])

    [http://arxiv.org/abs/2401.00870](http://arxiv.org/abs/2401.00870)

    这项研究提出了Prompt2Forget（P2F）框架，通过教导大型语言模型（LLM）忘记隐私信息，解决了LLM本地隐私挑战。P2F方法将问题分解为片段并生成虚构答案，模糊化模型对原始输入的记忆。实验证明，P2F具有很强的模糊化能力，并且可以在各种应用场景下自适应使用，无需手动设置。

    

    大型语言模型（LLM）已被证明具有强大的能力，但隐私泄露的风险仍然是一个重要问题。传统的保护隐私方法，如差分隐私和同态加密，在只有黑盒API的环境下是不足够的，要求模型透明性或大量计算资源。我们提出了Prompt2Forget（P2F），这是第一个设计用于解决LLM本地隐私挑战的框架，通过教导LLM忘记来实现。该方法涉及将完整问题分解为较小的片段，生成虚构的答案，并使模型对原始输入的记忆模糊化。我们根据不同领域的包含隐私敏感信息的问题创建了基准数据集。P2F实现了零-shot泛化，可以在多种应用场景下自适应，无需手动调整。实验结果表明，P2F具有很强的模糊化LLM记忆的能力，而不会损失任何实用性。

    Large Language Models (LLMs) have proven powerful, but the risk of privacy leakage remains a significant concern. Traditional privacy-preserving methods, such as Differential Privacy and Homomorphic Encryption, are inadequate for black-box API-only settings, demanding either model transparency or heavy computational resources. We propose Prompt2Forget (P2F), the first framework designed to tackle the LLM local privacy challenge by teaching LLM to forget. The method involves decomposing full questions into smaller segments, generating fabricated answers, and obfuscating the model's memory of the original input. A benchmark dataset was crafted with questions containing privacy-sensitive information from diverse fields. P2F achieves zero-shot generalization, allowing adaptability across a wide range of use cases without manual adjustments. Experimental results indicate P2F's robust capability to obfuscate LLM's memory, attaining a forgetfulness score of around 90\% without any utility los
    
[^2]: DiffSketcher: 通过隐式扩散模型实现文本引导的矢量素描合成

    DiffSketcher: Text Guided Vector Sketch Synthesis through Latent Diffusion Models. (arXiv:2306.14685v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.14685](http://arxiv.org/abs/2306.14685)

    本文介绍了DiffSketcher，一种通过隐式扩散模型实现文本引导的矢量素描合成的创新算法。DiffSketcher通过直接优化贝塞尔曲线和扩散模型损失来生成矢量化的手绘素描，并通过注意力图加快生成过程。实验结果表明DiffSketcher的素描质量高于之前方法。

    

    尽管主要训练于图像，但我们发现预训练的扩散模型在引导素描合成方面表现出惊人的能力。本文提出了DiffSketcher，一种创新的算法，利用自然语言输入创建矢量化的手绘素描。DiffSketcher基于预训练的文本到图像扩散模型开发，通过使用扩展版本的得分蒸馏采样（SDS）损失直接优化一组贝塞尔曲线，使得我们可以将栅格级扩散模型作为先验来优化参数化的矢量素描生成器。此外，我们还探索了扩散模型中嵌入的注意力图，在生成过程中实现有效的笔画初始化以加快速度。生成的素描展示了多层次的抽象，同时保持了被绘制主题的可识别性、基本结构和重要的视觉细节。我们的实验证明，DiffSketcher的质量优于之前方法。

    Even though trained mainly on images, we discover that pretrained diffusion models show impressive power in guiding sketch synthesis. In this paper, we present DiffSketcher, an innovative algorithm that creates vectorized free-hand sketches using natural language input. DiffSketcher is developed based on a pre-trained text-to-image diffusion model. It performs the task by directly optimizing a set of Bezier curves with an extended version of the score distillation sampling (SDS) loss, which allows us to use a raster-level diffusion model as a prior for optimizing a parametric vectorized sketch generator. Furthermore, we explore attention maps embedded in the diffusion model for effective stroke initialization to speed up the generation process. The generated sketches demonstrate multiple levels of abstraction while maintaining recognizability, underlying structure, and essential visual details of the subject drawn. Our experiments show that DiffSketcher achieves greater quality than pr
    
[^3]: 生成式人工智能应用调查

    A survey of Generative AI Applications. (arXiv:2306.02781v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.02781](http://arxiv.org/abs/2306.02781)

    本篇论文对350多个生成式人工智能应用进行了全面调查，总结了不同单模和多模生成式人工智能的应用。该调查为研究人员和从业者提供了宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    

    近年来，生成式人工智能有了显著增长，并在各个领域展示了广泛的应用。本文对350多个生成式人工智能应用进行了全面调查，提供了分类结构和对不同单模和多模生成式人工智能的简洁描述。该调查分成多个部分，覆盖了文本、图像、视频、游戏和脑信息等单模生成式人工智能的广泛应用。我们的调研旨在为研究人员和从业者提供宝贵的资源，帮助他们更好地了解生成式人工智能领域目前的最先进技术，并促进该领域的进一步创新。

    Generative AI has experienced remarkable growth in recent years, leading to a wide array of applications across diverse domains. In this paper, we present a comprehensive survey of more than 350 generative AI applications, providing a structured taxonomy and concise descriptions of various unimodal and even multimodal generative AIs. The survey is organized into sections, covering a wide range of unimodal generative AI applications such as text, images, video, gaming and brain information. Our survey aims to serve as a valuable resource for researchers and practitioners to navigate the rapidly expanding landscape of generative AI, facilitating a better understanding of the current state-of-the-art and fostering further innovation in the field.
    

