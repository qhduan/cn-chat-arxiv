# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automatic Summarization of Doctor-Patient Encounter Dialogues Using Large Language Model through Prompt Tuning](https://arxiv.org/abs/2403.13089) | 本研究提出了一种使用生成式大型语言模型对医患对话进行总结的方法，并通过提示调整算法指导模型进行临床文本总结，实现了在临床基准数据集上表现最佳的性能。 |
| [^2] | [Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing.](http://arxiv.org/abs/2308.06035) | 这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。 |

# 详细

[^1]: 使用大型语言模型通过提示调整自动总结医患对话

    Automatic Summarization of Doctor-Patient Encounter Dialogues Using Large Language Model through Prompt Tuning

    [https://arxiv.org/abs/2403.13089](https://arxiv.org/abs/2403.13089)

    本研究提出了一种使用生成式大型语言模型对医患对话进行总结的方法，并通过提示调整算法指导模型进行临床文本总结，实现了在临床基准数据集上表现最佳的性能。

    

    自动文本总结（ATS）是一种新兴技术，可以帮助临床医生提供持续和协调的护理。本研究介绍了一种使用生成式大型语言模型（LLMs）对医患对话进行总结的方法。我们开发了提示调整算法来指导生成式LLMs对临床文本进行总结。我们研究了提示调整策略、软提示的大小以及GatorTronGPT的few-short学习能力，该模型是使用2770亿临床和通用英语词汇开发的、拥有高达200亿参数的生成式临床LLM。我们将GatorTronGPT与基于广泛使用的T5模型微调的先前解决方案进行了比较，使用了临床基准数据集MTS-DIALOG。实验结果表明，GatorTronGPT-20B模型在所有评估指标上均取得了最佳性能。所提出的解决方案具有较低的计算成本，因为在提示调整过程中不更新LLM参数。

    arXiv:2403.13089v1 Announce Type: new  Abstract: Automatic text summarization (ATS) is an emerging technology to assist clinicians in providing continuous and coordinated care. This study presents an approach to summarize doctor-patient dialogues using generative large language models (LLMs). We developed prompt-tuning algorithms to instruct generative LLMs to summarize clinical text. We examined the prompt-tuning strategies, the size of soft prompts, and the few-short learning ability of GatorTronGPT, a generative clinical LLM developed using 277 billion clinical and general English words with up to 20 billion parameters. We compared GatorTronGPT with a previous solution based on fine-tuning of a widely used T5 model, using a clinical benchmark dataset MTS-DIALOG. The experimental results show that the GatorTronGPT- 20B model achieved the best performance on all evaluation metrics. The proposed solution has a low computing cost as the LLM parameters are not updated during prompt-tunin
    
[^2]: 多模态大语言模型在预测语言处理期间表现出人类视觉-语言集成的证据

    Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing. (arXiv:2308.06035v1 [cs.AI])

    [http://arxiv.org/abs/2308.06035](http://arxiv.org/abs/2308.06035)

    这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。

    

    大语言模型（LLMs）的先进语言处理能力引发了关于它们是否能够复制人类认知过程的争议。LLMs和人类在语言处理方面的一个区别在于，语言输入通常建立在多个知觉模态上，而大多数LLMs仅处理基于文本的信息。多模态基础使人类能够整合视觉背景与语言信息，从而对即将出现的单词的空间施加限制，减少认知负荷，提高感知和理解能力。最近的多模态LLMs（mLLMs）结合了视觉和语言嵌入空间，并使用变压器类型的注意机制进行下一个单词的预测。在多大程度上，基于多模态输入的预测语言处理在mLLMs和人类中吻合？为了回答这个问题，200名被试观看了短的视听剪辑，并估计了即将出现的动词或名词的可预测性。

    The advanced language processing abilities of large language models (LLMs) have stimulated debate over their capacity to replicate human-like cognitive processes. One differentiating factor between language processing in LLMs and humans is that language input is often grounded in more than one perceptual modality, whereas most LLMs process solely text-based information. Multimodal grounding allows humans to integrate - e.g. visual context with linguistic information and thereby place constraints on the space of upcoming words, reducing cognitive load and improving perception and comprehension. Recent multimodal LLMs (mLLMs) combine visual and linguistic embedding spaces with a transformer type attention mechanism for next-word prediction. To what extent does predictive language processing based on multimodal input align in mLLMs and humans? To answer this question, 200 human participants watched short audio-visual clips and estimated the predictability of an upcoming verb or noun. The 
    

