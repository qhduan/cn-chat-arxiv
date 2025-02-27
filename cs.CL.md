# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NeuroVoz: a Castillian Spanish corpus of parkinsonian speech](https://arxiv.org/abs/2403.02371) | 这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。 |
| [^2] | [Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models](https://arxiv.org/abs/2402.18945) | 论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。 |
| [^3] | [Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs](https://arxiv.org/abs/2402.05864) | 提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。 |
| [^4] | [Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition.](http://arxiv.org/abs/2309.08454) | 本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。 |
| [^5] | [World Models for Math Story Problems.](http://arxiv.org/abs/2306.04347) | 本文介绍了一个基于图形的语义规范MathWorld，可以将世界模型分配给数学故事问题，从而提供比现有方法更好的可解释性和对NLP模型的性能提升。 |

# 详细

[^1]: NeuroVoz：帕金森病患者语音的卡斯蒂利亚语语料库

    NeuroVoz: a Castillian Spanish corpus of parkinsonian speech

    [https://arxiv.org/abs/2403.02371](https://arxiv.org/abs/2403.02371)

    这一研究提出了一个包含108位母语为卡斯蒂利亚语说话者的帕金森病患者语音语料库，涵盖了多种语音任务，通过手动和自动转录确保了数据的准确性和可靠性。

    

    通过语音分析进行帕金森病（PD）诊断的进展受到公开可用、多样化的语言数据集的显著缺乏的阻碍，限制了现有研究结果的可再现性和进一步探索。为了弥补这一空白，我们引入了一个全面的语料库，包括来自108位母语为卡斯蒂利亚语的说话者，包括55名健康对照组和53名被诊断患有PD的个体，所有这些个体都在药物治疗下，并且在药物优化状态下进行记录。 这一独特数据集涵盖了广泛的语音任务，包括持续发音五个西班牙元音、发音测试、16个听后重复的话语以及自由独白。该数据集通过专家手动转录听后重复任务强调准确性和可靠性，并利用Whisper进行自动独白转录，使其成为帕金森病患者语音的最完整的公开语料库。

    arXiv:2403.02371v1 Announce Type: cross  Abstract: The advancement of Parkinson's Disease (PD) diagnosis through speech analysis is hindered by a notable lack of publicly available, diverse language datasets, limiting the reproducibility and further exploration of existing research.   In response to this gap, we introduce a comprehensive corpus from 108 native Castilian Spanish speakers, comprising 55 healthy controls and 53 individuals diagnosed with PD, all of whom were under pharmacological treatment and recorded in their medication-optimized state. This unique dataset features a wide array of speech tasks, including sustained phonation of the five Spanish vowels, diadochokinetic tests, 16 listen-and-repeat utterances, and free monologues. The dataset emphasizes accuracy and reliability through specialist manual transcriptions of the listen-and-repeat tasks and utilizes Whisper for automated monologue transcriptions, making it the most complete public corpus of Parkinsonian speech, 
    
[^2]: Syntactic Ghost：一种对预训练语言模型进行的无感知通用后门攻击

    Syntactic Ghost: An Imperceptible General-purpose Backdoor Attacks on Pre-trained Language Models

    [https://arxiv.org/abs/2402.18945](https://arxiv.org/abs/2402.18945)

    论文提出了一种名为Syntactic Ghost的新方法，实现了对预训练语言模型进行无感知和通用的后门植入。

    

    预训练语言模型（PLMs）被发现容易受到后门攻击，可以将漏洞转移到各种下游任务中。然而，现有的PLM后门攻击采用明显的触发器，在手动对准的情况下进行，因此在效果、隐匿性和通用性方面无法同时满足期望目标。本文提出了一种新方法，实现了不可见和通用的后门植入，称为Syntactic Ghost（简称为synGhost）。具体来说，该方法敌意地使用具有不同预定义句法结构的毒害样本作为隐蔽触发器，然后将后门植入到预训练表示空间，而不会破坏原始知识。毒害样本的输出表示在特征空间中尽可能均匀地分布，通过对比学习形成广泛的后门。此外，在亮

    arXiv:2402.18945v1 Announce Type: cross  Abstract: Pre-trained language models (PLMs) have been found susceptible to backdoor attacks, which can transfer vulnerabilities to various downstream tasks. However, existing PLM backdoors are conducted with explicit triggers under the manually aligned, thus failing to satisfy expectation goals simultaneously in terms of effectiveness, stealthiness, and universality. In this paper, we propose a novel approach to achieve invisible and general backdoor implantation, called \textbf{Syntactic Ghost} (synGhost for short). Specifically, the method hostilely manipulates poisoned samples with different predefined syntactic structures as stealth triggers and then implants the backdoor to pre-trained representation space without disturbing the primitive knowledge. The output representations of poisoned samples are distributed as uniformly as possible in the feature space via contrastive learning, forming a wide range of backdoors. Additionally, in light 
    
[^3]: Permute-and-Flip：一种具有最佳鲁棒性和可加水印的LLMs解码器

    Permute-and-Flip: An optimally robust and watermarkable decoder for LLMs

    [https://arxiv.org/abs/2402.05864](https://arxiv.org/abs/2402.05864)

    提出了一种名为Permute-and-Flip（PF）解码器，其具有最佳的鲁棒性和质量-鲁棒性的 tradeoff，且比采样方法更好。还设计了一种针对PF解码器的水印方案，能够保持样本的分布不变，并实现任意低的假阳性率和高的召回率。实验证明PF解码器在困惑度方面明显优于朴素采样，为LLM解码提供了一种有希望的新方法。

    

    在本文中，我们提出了一种名为Permute-and-Flip（PF）解码器的新解码方法。它具有与标准采样解码器相似的鲁棒性特性，但在质量和鲁棒性的 tradeoff 上证明比采样方法更好，且永远不会差于任何其他解码器。同时，我们还设计了一种类似于Aaronson的Gumbel水印的加密水印方案，但是针对PF解码器而自然量身定制。该水印方案不改变样本的分布，同时允许任意低的假阳性率和高的召回率，只要生成的文本具有高熵。我们的实验证明，PF解码器（及其带有水印的对应物）在困惑度方面明显优于朴素采样（及其带有Gumbel水印的对应物），同时保持相同的鲁棒性（和可检测性），因此为LLM解码提供了一个有希望的新方法。代码可在https://github.com/XuandongZhao/pf-decoding找到。

    In this paper, we propose a new decoding method called Permute-and-Flip (PF) decoder. It enjoys robustness properties similar to the standard sampling decoder, but is provably up to 2x better in its quality-robustness tradeoff than sampling and never worse than any other decoder. We also design a cryptographic watermarking scheme analogous to Aaronson's Gumbel watermark, but naturally tailored for PF decoder. The watermarking scheme does not change the distribution to sample, while allowing arbitrarily low false positive rate and high recall whenever the generated text has high entropy. Our experiments show that the PF decoder (and its watermarked counterpart) significantly outperform(s) naive sampling (and it's Gumbel watermarked counterpart) in terms of perplexity, while retaining the same robustness (and detectability), hence making it a promising new approach for LLM decoding. The code is available at https://github.com/XuandongZhao/pf-decoding
    
[^4]: 混合编码器支持连续语音分离用于会议识别

    Mixture Encoder Supporting Continuous Speech Separation for Meeting Recognition. (arXiv:2309.08454v1 [eess.AS])

    [http://arxiv.org/abs/2309.08454](http://arxiv.org/abs/2309.08454)

    本研究将混合编码器方法从两个说话人情况扩展到了更自然的会议环境，包括任意数量的说话人和动态重叠。实验证明，该方法在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。

    

    自动语音识别（ASR）的许多实际应用需要处理重叠的语音。一种常见的方法是首先将语音分离成无重叠的流，然后对生成的信号进行ASR。最近，提出了在ASR模型中包含混合编码器的方法。该混合编码器利用原始重叠的语音来减轻语音分离引入的伪影效果。然而，先前的方法仅针对两个说话人的情况。在这项工作中，我们将这种方法扩展到更自然的会议环境，包括任意数量的说话人和动态重叠。我们使用不同的语音分离器（包括强大的TF-GridNet模型）评估性能。实验证明，在LibriCSS数据集上达到了最先进的性能，并凸显了混合编码器的优势。此外，实验还展示了TF-GridNet的强大分离能力，大大缩小了先前方法的差距。

    Many real-life applications of automatic speech recognition (ASR) require processing of overlapped speech. A commonmethod involves first separating the speech into overlap-free streams and then performing ASR on the resulting signals. Recently, the inclusion of a mixture encoder in the ASR model has been proposed. This mixture encoder leverages the original overlapped speech to mitigate the effect of artifacts introduced by the speech separation. Previously, however, the method only addressed two-speaker scenarios. In this work, we extend this approach to more natural meeting contexts featuring an arbitrary number of speakers and dynamic overlaps. We evaluate the performance using different speech separators, including the powerful TF-GridNet model. Our experiments show state-of-the-art performance on the LibriCSS dataset and highlight the advantages of the mixture encoder. Furthermore, they demonstrate the strong separation of TF-GridNet which largely closes the gap between previous m
    
[^5]: 数学解题的世界模型研究

    World Models for Math Story Problems. (arXiv:2306.04347v1 [cs.CL])

    [http://arxiv.org/abs/2306.04347](http://arxiv.org/abs/2306.04347)

    本文介绍了一个基于图形的语义规范MathWorld，可以将世界模型分配给数学故事问题，从而提供比现有方法更好的可解释性和对NLP模型的性能提升。

    

    对于学生和自然语言处理模型而言，解决数学故事问题是一项复杂的任务，需要他们理解故事中所描述的世界并对其进行推理，以计算出答案。近年来，大型预训练语言模型和创新技术已经取得了惊人的表现，可以自动解决这些问题。但是，这些模型是否具有数学概念的准确表示仍不清楚。这导致缺乏可解释性和可信度，从而影响它们在各种应用中的有用性。本文将之前的工作整合到分类和表达数学故事问题上，并开发出针对数学故事问题领域的基于图形的语义规范MathWorld。利用MathWorld，我们可以为数学故事问题分配世界模型，它们表示在文本中介绍的情况和行动以及它们的数学关系。我们将来自几个现有数据集的数学故事问题组合在一起，并在我们收集的新数据集Story-Gen Math上评估我们的方法，该数据集包含具有不同难度的具有挑战性的问题。我们的实验表明，MathWorld可以提高自然语言处理模型的性能，并提供比现有方法更好的可解释性。

    Solving math story problems is a complex task for students and NLP models alike, requiring them to understand the world as described in the story and reason over it to compute an answer. Recent years have seen impressive performance on automatically solving these problems with large pre-trained language models and innovative techniques to prompt them. However, it remains unclear if these models possess accurate representations of mathematical concepts. This leads to lack of interpretability and trustworthiness which impedes their usefulness in various applications. In this paper, we consolidate previous work on categorizing and representing math story problems and develop MathWorld, which is a graph-based semantic formalism specific for the domain of math story problems. With MathWorld, we can assign world models to math story problems which represent the situations and actions introduced in the text and their mathematical relationships. We combine math story problems from several exis
    

