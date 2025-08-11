# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking LLMs on the Semantic Overlap Summarization Task](https://arxiv.org/abs/2402.17008) | 该论文对LLMs在Semantic Overlap Summarization任务上进行基准测试，使用TELeR分类法评估了15个流行的LLMs的性能，以评估它们总结多个不同叙述之间重叠信息的能力。 |
| [^2] | [Integrating large language models and active inference to understand eye movements in reading and dyslexia.](http://arxiv.org/abs/2308.04941) | 该论文提出了一种集成大型语言模型和主动推理的计算模型，用于模拟阅读过程中的眼动行为。该模型能够准确地预测和推理不同粒度的文本信息，并能够模拟阅读障碍中不适应推理效果的情况。 |

# 详细

[^1]: 在语义重叠摘要任务上对LLMs进行基准测试

    Benchmarking LLMs on the Semantic Overlap Summarization Task

    [https://arxiv.org/abs/2402.17008](https://arxiv.org/abs/2402.17008)

    该论文对LLMs在Semantic Overlap Summarization任务上进行基准测试，使用TELeR分类法评估了15个流行的LLMs的性能，以评估它们总结多个不同叙述之间重叠信息的能力。

    

    Semantic Overlap Summarization (SOS)是一项受限的多文档摘要任务，其中约束是捕获两个不同叙述之间的共同/重叠信息。虽然最近大型语言模型（LLMs）在许多摘要任务中取得了优越的性能，但尚未进行过使用LLMs进行SOS任务的基准测试研究。由于LLMs的响应对提示设计中的细微变化很敏感，进行这样的基准测试研究的主要挑战是在得出可靠结论之前系统地探索各种提示。幸运的是，最近提出了TELeR分类法，可用于设计和探索LLMs的各种提示。利用这个TELeR分类法和15个流行的LLMs，本文全面评估了LLMs在SOS任务上的表现，评估它们从多个不同叙述中总结重叠信息的能力。

    arXiv:2402.17008v1 Announce Type: new  Abstract: Semantic Overlap Summarization (SOS) is a constrained multi-document summarization task, where the constraint is to capture the common/overlapping information between two alternative narratives. While recent advancements in Large Language Models (LLMs) have achieved superior performance in numerous summarization tasks, a benchmarking study of the SOS task using LLMs is yet to be performed. As LLMs' responses are sensitive to slight variations in prompt design, a major challenge in conducting such a benchmarking study is to systematically explore a variety of prompts before drawing a reliable conclusion. Fortunately, very recently, the TELeR taxonomy has been proposed which can be used to design and explore various prompts for LLMs. Using this TELeR taxonomy and 15 popular LLMs, this paper comprehensively evaluates LLMs on the SOS Task, assessing their ability to summarize overlapping information from multiple alternative narratives. For 
    
[^2]: 集成大型语言模型和主动推理以理解阅读和阅读障碍中的眼动行为

    Integrating large language models and active inference to understand eye movements in reading and dyslexia. (arXiv:2308.04941v1 [q-bio.NC])

    [http://arxiv.org/abs/2308.04941](http://arxiv.org/abs/2308.04941)

    该论文提出了一种集成大型语言模型和主动推理的计算模型，用于模拟阅读过程中的眼动行为。该模型能够准确地预测和推理不同粒度的文本信息，并能够模拟阅读障碍中不适应推理效果的情况。

    

    我们提出了一种新颖的计算模型，采用层次化主动推理来模拟阅读和眼动行为。该模型将语言处理描述为对层次生成模型的推理，从音节到句子的不同粒度实现预测和推理。我们的方法结合了大型语言模型的优势，用于实现逼真的文本预测，以及主动推理用于引导眼动到信息丰富的文本信息，从而使得对预测进行测试成为可能。该模型能够熟练阅读已知和未知的单词和句子，并遵循阅读双路理论中的词汇和非词汇路径的区分。值得注意的是，我们的模型允许模拟阅读过程中对眼动行为产生不适应推理效果的情况，例如阅读障碍。为了模拟这种情况，我们在阅读过程中减弱了先验的贡献，导致不正确的推理和更加断片化的阅读。

    We present a novel computational model employing hierarchical active inference to simulate reading and eye movements. The model characterizes linguistic processing as inference over a hierarchical generative model, facilitating predictions and inferences at various levels of granularity, from syllables to sentences.  Our approach combines the strengths of large language models for realistic textual predictions and active inference for guiding eye movements to informative textual information, enabling the testing of predictions. The model exhibits proficiency in reading both known and unknown words and sentences, adhering to the distinction between lexical and nonlexical routes in dual-route theories of reading. Notably, our model permits the exploration of maladaptive inference effects on eye movements during reading, such as in dyslexia. To simulate this condition, we attenuate the contribution of priors during the reading process, leading to incorrect inferences and a more fragmented
    

