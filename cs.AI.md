# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Elements of Differentiable Programming](https://arxiv.org/abs/2403.14606) | 可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。 |
| [^2] | [Align and Distill: Unifying and Improving Domain Adaptive Object Detection](https://arxiv.org/abs/2403.12029) | 引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。 |
| [^3] | [Sum-of-Parts Models: Faithful Attributions for Groups of Features.](http://arxiv.org/abs/2310.16316) | Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。 |
| [^4] | [DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing.](http://arxiv.org/abs/2310.08785) | 本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。 |
| [^5] | [Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages.](http://arxiv.org/abs/2308.16075) | 该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。 |
| [^6] | [DF2: Distribution-Free Decision-Focused Learning.](http://arxiv.org/abs/2308.05889) | DF2是一种无分布的决策焦点学习方法，特别解决了模型不匹配错误、样本平均逼近误差和梯度逼近误差三个瓶颈问题。 |

# 详细

[^1]: 可微分编程的要素

    The Elements of Differentiable Programming

    [https://arxiv.org/abs/2403.14606](https://arxiv.org/abs/2403.14606)

    可微分编程是一个新的编程范式，使得复杂程序能够端对端地进行微分，实现基于梯度的参数优化。

    

    人工智能最近取得了显著进展，这得益于大型模型、庞大数据集、加速硬件，以及可微分编程的变革性力量。这种新的编程范式使复杂计算机程序（包括具有控制流和数据结构的程序）能够进行端对端的微分，从而实现对程序参数的基于梯度的优化。不仅仅是程序的微分，可微分编程也包括了程序优化、概率等多个领域的概念。本书介绍了可微分编程所需的基本概念，并采用了优化和概率两个主要视角进行阐述。

    arXiv:2403.14606v1 Announce Type: new  Abstract: Artificial intelligence has recently experienced remarkable advances, fueled by large models, vast datasets, accelerated hardware, and, last but not least, the transformative power of differentiable programming. This new programming paradigm enables end-to-end differentiation of complex computer programs (including those with control flows and data structures), making gradient-based optimization of program parameters possible. As an emerging paradigm, differentiable programming builds upon several areas of computer science and applied mathematics, including automatic differentiation, graphical models, optimization and statistics. This book presents a comprehensive review of the fundamental concepts useful for differentiable programming. We adopt two main perspectives, that of optimization and that of probability, with clear analogies between the two. Differentiable programming is not merely the differentiation of programs, but also the t
    
[^2]: 对齐与提炼：统一和改进领域自适应目标检测

    Align and Distill: Unifying and Improving Domain Adaptive Object Detection

    [https://arxiv.org/abs/2403.12029](https://arxiv.org/abs/2403.12029)

    引入了统一的基准测试和实现框架ALDI以及新的DAOD基准数据集CFC-DAOD，解决了领域自适应目标检测中的基准问题，并支持未来方法的发展。

    

    目标检测器通常表现不佳于与其训练集不同的数据。最近，领域自适应目标检测（DAOD）方法已经展示了在应对这一挑战上的强大结果。遗憾的是，我们发现了系统化的基准测试陷阱，这些陷阱对过去的结果提出质疑并阻碍了进一步的进展：（a）由于基线不足导致性能高估，（b）不一致的实现实践阻止了方法的透明比较，（c）由于过时的骨干和基准测试缺乏多样性，导致缺乏普遍性。我们通过引入以下问题来解决这些问题：（1）一个统一的基准测试和实现框架，Align and Distill（ALDI），支持DAOD方法的比较并支持未来发展，（2）一个公平且现代的DAOD训练和评估协议，解决了基准测试的陷阱，（3）一个新的DAOD基准数据集，CFC-DAOD，能够在多样化的真实环境中进行评估。

    arXiv:2403.12029v1 Announce Type: cross  Abstract: Object detectors often perform poorly on data that differs from their training set. Domain adaptive object detection (DAOD) methods have recently demonstrated strong results on addressing this challenge. Unfortunately, we identify systemic benchmarking pitfalls that call past results into question and hamper further progress: (a) Overestimation of performance due to underpowered baselines, (b) Inconsistent implementation practices preventing transparent comparisons of methods, and (c) Lack of generality due to outdated backbones and lack of diversity in benchmarks. We address these problems by introducing: (1) A unified benchmarking and implementation framework, Align and Distill (ALDI), enabling comparison of DAOD methods and supporting future development, (2) A fair and modern training and evaluation protocol for DAOD that addresses benchmarking pitfalls, (3) A new DAOD benchmark dataset, CFC-DAOD, enabling evaluation on diverse real
    
[^3]: Sum-of-Parts模型：对特征组的忠实归因

    Sum-of-Parts Models: Faithful Attributions for Groups of Features. (arXiv:2310.16316v1 [cs.LG])

    [http://arxiv.org/abs/2310.16316](http://arxiv.org/abs/2310.16316)

    Sum-of-Parts模型通过构造保证特征组归因的忠实性，将预测分解为可解释的分数之和，帮助天体物理学家发现了关于星系形成的新知识。

    

    如果机器学习模型的解释准确反映了其决策过程，则被认为是“忠实”的解释。然而，例如深度学习的特征归因等解释并不能保证忠实，有可能产生具有误导性的解释。在这项工作中，我们开发了Sum-of-Parts（SOP）模型，它是一类模型，其预测具有通过构造保证忠实的特征组归因。该模型将预测分解为可解释的分数之和，每个分数直接归因于一组稀疏特征。我们使用标准可解释性指标对SOP进行评估，并在一个案例研究中，利用SOP提供的忠实解释帮助天体物理学家发现了关于星系形成的新知识。

    An explanation of a machine learning model is considered "faithful" if it accurately reflects the model's decision-making process. However, explanations such as feature attributions for deep learning are not guaranteed to be faithful, and can produce potentially misleading interpretations. In this work, we develop Sum-of-Parts (SOP), a class of models whose predictions come with grouped feature attributions that are faithful-by-construction. This model decomposes a prediction into an interpretable sum of scores, each of which is directly attributable to a sparse group of features. We evaluate SOP on benchmarks with standard interpretability metrics, and in a case study, we use the faithful explanations from SOP to help astrophysicists discover new knowledge about galaxy formation.
    
[^4]: DeltaSpace:一种用于灵活文本引导图像编辑的语义对齐特征空间

    DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing. (arXiv:2310.08785v1 [cs.CV])

    [http://arxiv.org/abs/2310.08785](http://arxiv.org/abs/2310.08785)

    本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。

    

    文本引导图像编辑面临着训练和推理灵活性的重大挑战。许多文献通过收集大量标注的图像-文本对来从头开始训练文本条件生成模型，这既昂贵又低效。然后，一些利用预训练的视觉语言模型的方法出现了，以避免数据收集，但它们仍然受到基于每个文本提示的优化或推理时的超参数调整的限制。为了解决这些问题，我们调查和确定了一个特定的空间，称为CLIP DeltaSpace，在这个空间中，两个图像的CLIP视觉特征差异与其对应的文本描述的CLIP文本特征差异在语义上是对齐的。基于DeltaSpace，我们提出了一个新颖的框架DeltaEdit，在训练阶段将CLIP视觉特征差异映射到生成模型的潜在空间方向，并从CLIP预测潜在空间方向。

    Text-guided image editing faces significant challenges to training and inference flexibility. Much literature collects large amounts of annotated image-text pairs to train text-conditioned generative models from scratch, which is expensive and not efficient. After that, some approaches that leverage pre-trained vision-language models are put forward to avoid data collection, but they are also limited by either per text-prompt optimization or inference-time hyper-parameters tuning. To address these issues, we investigate and identify a specific space, referred to as CLIP DeltaSpace, where the CLIP visual feature difference of two images is semantically aligned with the CLIP textual feature difference of their corresponding text descriptions. Based on DeltaSpace, we propose a novel framework called DeltaEdit, which maps the CLIP visual feature differences to the latent space directions of a generative model during the training phase, and predicts the latent space directions from the CLIP
    
[^5]: 视觉背景对嘈杂的多模态神经机器翻译的影响：对英印语言的实证研究

    Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages. (arXiv:2308.16075v1 [cs.CL])

    [http://arxiv.org/abs/2308.16075](http://arxiv.org/abs/2308.16075)

    该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。

    

    本研究调查了在神经机器翻译中利用多模态信息的有效性。先前的研究主要关注在资源匮乏的情况下使用多模态数据，而本研究则考察了将图像特征添加到大规模预训练的单模态神经机器翻译系统中的翻译效果。令人惊讶的是，研究发现在这种情况下图像可能是多余的。此外，该研究引入了合成噪声来评估图像是否有助于模型处理文本噪声。在嘈杂的环境中，即使是随机图像，多模态模型在性能上略优于文本模型。实验将英语翻译为印地语、孟加拉语和马拉雅拉姆语，结果显著优于最先进的基准。有趣的是，视觉背景的影响与源文本噪声有所不同：对于非噪声翻译，不使用视觉背景效果最好；对于低噪声，裁剪的图像特征最佳；在高噪声情况下，完整的图像特征效果更好。

    The study investigates the effectiveness of utilizing multimodal information in Neural Machine Translation (NMT). While prior research focused on using multimodal data in low-resource scenarios, this study examines how image features impact translation when added to a large-scale, pre-trained unimodal NMT system. Surprisingly, the study finds that images might be redundant in this context. Additionally, the research introduces synthetic noise to assess whether images help the model deal with textual noise. Multimodal models slightly outperform text-only models in noisy settings, even with random images. The study's experiments translate from English to Hindi, Bengali, and Malayalam, outperforming state-of-the-art benchmarks significantly. Interestingly, the effect of visual context varies with source text noise: no visual context works best for non-noisy translations, cropped image features are optimal for low noise, and full image features work better in high-noise scenarios. This she
    
[^6]: DF2: 无分布的决策焦点学习

    DF2: Distribution-Free Decision-Focused Learning. (arXiv:2308.05889v1 [cs.LG])

    [http://arxiv.org/abs/2308.05889](http://arxiv.org/abs/2308.05889)

    DF2是一种无分布的决策焦点学习方法，特别解决了模型不匹配错误、样本平均逼近误差和梯度逼近误差三个瓶颈问题。

    

    最近决策焦点学习（DFL）作为一种强大的方法在解决预测-优化问题时，通过将预测模型定制到一个下游优化任务。然而，现有的端到端DFL方法受到三个重要瓶颈的制约：模型不匹配错误、样本平均逼近误差和梯度逼近误差。模型不匹配错误源于模型参数化的预测分布与真实概率分布之间的不协调。样本平均逼近误差是使用有限样本来近似期望优化目标时产生的。梯度逼近误差发生在DFL依靠KKT条件进行精确梯度计算时，而大多数方法在非凸目标中近似梯度进行反向传播。在本文中，我们提出DF2 - 第一个明确设计来解决这三个瓶颈的无分布决策焦点学习方法。

    Decision-focused learning (DFL) has recently emerged as a powerful approach for predict-then-optimize problems by customizing a predictive model to a downstream optimization task. However, existing end-to-end DFL methods are hindered by three significant bottlenecks: model mismatch error, sample average approximation error, and gradient approximation error. Model mismatch error stems from the misalignment between the model's parameterized predictive distribution and the true probability distribution. Sample average approximation error arises when using finite samples to approximate the expected optimization objective. Gradient approximation error occurs as DFL relies on the KKT condition for exact gradient computation, while most methods approximate the gradient for backpropagation in non-convex objectives. In this paper, we present DF2 -- the first \textit{distribution-free} decision-focused learning method explicitly designed to address these three bottlenecks. Rather than depending 
    

