# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PATCH -- Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Mathematics Proficiency](https://arxiv.org/abs/2404.01799) | 该论文提出了一种新的框架PATCH，用于将心理测量领域的知识整合到大型语言模型的基准测试中，以解决现有基准测试存在的测量质量、项目级别评估和参考人群等问题。 |
| [^2] | [Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages.](http://arxiv.org/abs/2308.16075) | 该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。 |

# 详细

[^1]: PATCH -- 大型语言模型的心理测量辅助基准测试：数学能力的案例研究

    PATCH -- Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Mathematics Proficiency

    [https://arxiv.org/abs/2404.01799](https://arxiv.org/abs/2404.01799)

    该论文提出了一种新的框架PATCH，用于将心理测量领域的知识整合到大型语言模型的基准测试中，以解决现有基准测试存在的测量质量、项目级别评估和参考人群等问题。

    

    许多现有的大型（多模态）语言模型（LLMs）基准测试着重于衡量LLMs的学术能力，通常也对比较模型性能与人类考试者感兴趣。尽管这些基准测试对LLMs的发展至关重要，但它们存在一些限制，包括有问题的测量质量（例如，它们是否以可靠的方式衡量所需的内容？）、缺乏项目级别的质量评估（例如，有些项目是否比其他更重要或更困难？）以及人类人口参照模糊（例如，模型可以与谁进行比较？）。为了应对这些挑战，我们提出利用心理测量学领域的知识——一门致力于测量潜在变量如学术能力的领域——来进行LLMs基准测试的心理测量辅助方法。我们的主要贡献有三点。首先，我们介绍了PATCH：一种用于大型语言模型的心理测量辅助基准测试的新框架。

    arXiv:2404.01799v1 Announce Type: new  Abstract: Many existing benchmarks of large (multimodal) language models (LLMs) focus on measuring LLMs' academic proficiency, often with also an interest in comparing model performance with human test takers. While these benchmarks have proven key to the development of LLMs, they suffer from several limitations, including questionable measurement quality (e.g., Do they measure what they are supposed to in a reliable way?), lack of quality assessment on the item level (e.g., Are some items more important or difficult than others?) and unclear human population reference (e.g., To whom can the model be compared?). In response to these challenges, we propose leveraging knowledge from psychometrics - a field dedicated to the measurement of latent variables like academic proficiency - into LLM benchmarking. We make three primary contributions. First, we introduce PATCH: a novel framework for Psychometrics-AssisTed benCHmarking of LLMs. PATCH addresses 
    
[^2]: 视觉背景对嘈杂的多模态神经机器翻译的影响：对英印语言的实证研究

    Impact of Visual Context on Noisy Multimodal NMT: An Empirical Study for English to Indian Languages. (arXiv:2308.16075v1 [cs.CL])

    [http://arxiv.org/abs/2308.16075](http://arxiv.org/abs/2308.16075)

    该研究实证研究了神经机器翻译中利用多模态信息的有效性，发现在大规模预训练的单模态系统中添加图像特征可能是多余的。此外，该研究还引入了合成噪声来评估图像对处理文本噪声的帮助。实验结果表明，多模态模型在嘈杂的环境中略优于文本模型，即使是随机图像。研究在英语翻译为印地语、孟加拉语和马拉雅拉姆语时表现出色，且视觉背景对翻译效果的影响与源文本噪声有所不同。

    

    本研究调查了在神经机器翻译中利用多模态信息的有效性。先前的研究主要关注在资源匮乏的情况下使用多模态数据，而本研究则考察了将图像特征添加到大规模预训练的单模态神经机器翻译系统中的翻译效果。令人惊讶的是，研究发现在这种情况下图像可能是多余的。此外，该研究引入了合成噪声来评估图像是否有助于模型处理文本噪声。在嘈杂的环境中，即使是随机图像，多模态模型在性能上略优于文本模型。实验将英语翻译为印地语、孟加拉语和马拉雅拉姆语，结果显著优于最先进的基准。有趣的是，视觉背景的影响与源文本噪声有所不同：对于非噪声翻译，不使用视觉背景效果最好；对于低噪声，裁剪的图像特征最佳；在高噪声情况下，完整的图像特征效果更好。

    The study investigates the effectiveness of utilizing multimodal information in Neural Machine Translation (NMT). While prior research focused on using multimodal data in low-resource scenarios, this study examines how image features impact translation when added to a large-scale, pre-trained unimodal NMT system. Surprisingly, the study finds that images might be redundant in this context. Additionally, the research introduces synthetic noise to assess whether images help the model deal with textual noise. Multimodal models slightly outperform text-only models in noisy settings, even with random images. The study's experiments translate from English to Hindi, Bengali, and Malayalam, outperforming state-of-the-art benchmarks significantly. Interestingly, the effect of visual context varies with source text noise: no visual context works best for non-noisy translations, cropped image features are optimal for low noise, and full image features work better in high-noise scenarios. This she
    

