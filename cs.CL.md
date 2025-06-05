# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with Large Language Models.](http://arxiv.org/abs/2310.12049) | 这项研究开发了一种文本缩放方法，利用生成性大型语言模型的模式识别能力，通过概念导向思维链图和大型语言模型进行文本比较，并使用Bradley-Terry模型来估计评分尺度。该方法在Twitter上对情感言论的缩放效果更好。 |
| [^2] | [Transformers in Speech Processing: A Survey.](http://arxiv.org/abs/2303.11607) | 本文综述了Transformer在语音相关领域的广泛应用，为研究者提供了有价值的资源。同时，指出了在语音处理领域中Transformer所面临的挑战及可能的解决思路。 |

# 详细

[^1]: Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with Large Language Models (使用大型语言模型的概念导向思维链图提示进行文本配对比较缩放)

    Concept-Guided Chain-of-Thought Prompting for Pairwise Comparison Scaling of Texts with Large Language Models. (arXiv:2310.12049v1 [cs.CL])

    [http://arxiv.org/abs/2310.12049](http://arxiv.org/abs/2310.12049)

    这项研究开发了一种文本缩放方法，利用生成性大型语言模型的模式识别能力，通过概念导向思维链图和大型语言模型进行文本比较，并使用Bradley-Terry模型来估计评分尺度。该方法在Twitter上对情感言论的缩放效果更好。

    

    现有的文本缩放方法经常需要大型语料库，难以处理短文本，或需要有标签的数据。我们开发了一种利用生成性大型语言模型（LLM）的模式识别能力来进行文本缩放的方法。具体而言，我们提出了概念导向思维链图（CGCoT），它使用设计用于总结想法并在文本中识别目标方的提示来生成概念特定的细分，类似于人类编码器内容分析的指导。CGCoT将配对文本比较从一个推理问题转变为一个模式识别问题。然后，我们使用LLM对概念特定的细分进行配对比较。我们利用这些配对比较的结果使用Bradley-Terry模型来估计一个评分尺度。我们利用这种方法对Twitter上的情感言论进行缩放。我们的测量值与人类判断的相关性比Wordfish等替代方法更强。除了一小组用于开发CGCoT提示的试验数据之外，...

    Existing text scaling methods often require a large corpus, struggle with short texts, or require labeled data. We develop a text scaling method that leverages the pattern recognition capabilities of generative large language models (LLMs). Specifically, we propose concept-guided chain-of-thought (CGCoT), which uses prompts designed to summarize ideas and identify target parties in texts to generate concept-specific breakdowns, in many ways similar to guidance for human coder content analysis. CGCoT effectively shifts pairwise text comparisons from a reasoning problem to a pattern recognition problem. We then pairwise compare concept-specific breakdowns using an LLM. We use the results of these pairwise comparisons to estimate a scale using the Bradley-Terry model. We use this approach to scale affective speech on Twitter. Our measures correlate more strongly with human judgments than alternative approaches like Wordfish. Besides a small set of pilot data to develop the CGCoT prompts, 
    
[^2]: 论文翻译：语音处理中的Transformer：综述（arXiv:2303.11607v1 [cs.CL]）

    Transformers in Speech Processing: A Survey. (arXiv:2303.11607v1 [cs.CL])

    [http://arxiv.org/abs/2303.11607](http://arxiv.org/abs/2303.11607)

    本文综述了Transformer在语音相关领域的广泛应用，为研究者提供了有价值的资源。同时，指出了在语音处理领域中Transformer所面临的挑战及可能的解决思路。

    

    Transformer 在自然语言处理领域中的显著成功引起了语音处理社区的兴趣，进而探索了其模拟语音序列中长距离依赖关系的潜力。最近，Transformer 在各种涉及语音的领域中名声鹊起，包括自动语音识别、语音合成、语音翻译、语音声调学、语音增强、口语对话系统，以及许多多模态应用。本文提供一份综合性调查报告，旨在桥接语音技术各子领域的研究。通过整合来自语音技术领域的研究结果，我们为希望利用Transformer推进领域发展的研究人员提供了有价值的资源。同时，我们也指出了Transformer在语音处理中遇到的挑战，并提供了解决这些问题的潜在思路。

    The remarkable success of transformers in the field of natural language processing has sparked the interest of the speech-processing community, leading to an exploration of their potential for modeling long-range dependencies within speech sequences. Recently, transformers have gained prominence across various speech-related domains, including automatic speech recognition, speech synthesis, speech translation, speech para-linguistics, speech enhancement, spoken dialogue systems, and numerous multimodal applications. In this paper, we present a comprehensive survey that aims to bridge research studies from diverse subfields within speech technology. By consolidating findings from across the speech technology landscape, we provide a valuable resource for researchers interested in harnessing the power of transformers to advance the field. We identify the challenges encountered by transformers in speech processing while also offering insights into potential solutions to address these issue
    

