# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Topic-based Watermarks for LLM-Generated Text](https://arxiv.org/abs/2404.02138) | 提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。 |
| [^2] | [Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews](https://arxiv.org/abs/2403.07183) | 该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。 |

# 详细

[^1]: 基于主题的LLM生成文本的水印

    Topic-based Watermarks for LLM-Generated Text

    [https://arxiv.org/abs/2404.02138](https://arxiv.org/abs/2404.02138)

    提出了一种新的基于主题的水印算法，旨在解决当前水印方案的局限性，为区分LLM生成的文本和人类生成的文本提供了新的思路。

    

    大型语言模型（LLMs）的最新进展导致了生成的文本输出与人类生成的文本相似度难以分辨。水印算法是潜在工具，通过在LLM生成的输出中嵌入可检测的签名，可以区分LLM生成的文本和人类生成的文本。然而，当前的水印方案在已知攻击下缺乏健壮性。此外，考虑到LLM每天生成数万个文本输出，水印算法需要记忆每个输出才能让检测正常工作，这是不切实际的。本文针对当前水印方案的局限性，提出了针对LLMs的“基于主题的水印算法”概念。

    arXiv:2404.02138v1 Announce Type: cross  Abstract: Recent advancements of large language models (LLMs) have resulted in indistinguishable text outputs comparable to human-generated text. Watermarking algorithms are potential tools that offer a way to differentiate between LLM- and human-generated text by embedding detectable signatures within LLM-generated output. However, current watermarking schemes lack robustness against known attacks against watermarking algorithms. In addition, they are impractical considering an LLM generates tens of thousands of text outputs per day and the watermarking algorithm needs to memorize each output it generates for the detection to work. In this work, focusing on the limitations of current watermarking schemes, we propose the concept of a "topic-based watermarking algorithm" for LLMs. The proposed algorithm determines how to generate tokens for the watermarked LLM output based on extracted topics of an input prompt or the output of a non-watermarked 
    
[^2]: 在规模上监测AI修改的内容：AI会议同行评审中ChatGPT影响的案例研究

    Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews

    [https://arxiv.org/abs/2403.07183](https://arxiv.org/abs/2403.07183)

    该研究提出了一种估计大语料库中被大语言模型大幅修改的文本比例的方法，并在AI会议的同行评审中进行了实证分析，发现6.5%至16.9%的文本可能被LLMs大幅修改，揭示了用户行为的一些见解。

    

    我们提出了一种估计大语料库中文本可能被大语言模型（LLM）大幅修改或生成的部分比例的方法。我们的最大似然模型利用专家撰写和AI生成的参考文本，准确高效地检查语料库级别上真实世界LLM使用。我们将这种方法应用于AI会议上科学同行评审的案例研究，该研究发生在ChatGPT发布之后，包括ICLR 2024、NeurIPS 2023、CoRL 2023和EMNLP 2023。我们的研究结果表明，在这些会议提交的同行评审中，6.5%至16.9%的文本可能是由LLMs大幅修改的，即超出拼写检查或小幅更新的范围。生成文本出现的情况为用户行为提供了见解：在报告信心较低、在截止日期前提交的评论以及从评论公司

    arXiv:2403.07183v1 Announce Type: cross  Abstract: We present an approach for estimating the fraction of text in a large corpus which is likely to be substantially modified or produced by a large language model (LLM). Our maximum likelihood model leverages expert-written and AI-generated reference texts to accurately and efficiently examine real-world LLM-use at the corpus level. We apply this approach to a case study of scientific peer review in AI conferences that took place after the release of ChatGPT: ICLR 2024, NeurIPS 2023, CoRL 2023 and EMNLP 2023. Our results suggest that between 6.5% and 16.9% of text submitted as peer reviews to these conferences could have been substantially modified by LLMs, i.e. beyond spell-checking or minor writing updates. The circumstances in which generated text occurs offer insight into user behavior: the estimated fraction of LLM-generated text is higher in reviews which report lower confidence, were submitted close to the deadline, and from review
    

