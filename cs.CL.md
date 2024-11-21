# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs](https://rss.arxiv.org/abs/2312.05356) | 这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。 |
| [^2] | [Rich Semantic Knowledge Enhanced Large Language Models for Few-shot Chinese Spell Checking](https://arxiv.org/abs/2403.08492) | 本文使用富含语义知识的大型语言模型在少样本中文拼写检查任务上取得了比BERT模型更好的性能。 |
| [^3] | [Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks.](http://arxiv.org/abs/2305.01626) | 该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。 |

# 详细

[^1]: Neuron Patching: 神经元层面的模型编辑与代码生成

    Neuron Patching: Neuron-level Model Editing on Code Generation and LLMs

    [https://rss.arxiv.org/abs/2312.05356](https://rss.arxiv.org/abs/2312.05356)

    这项工作介绍了一种神经元层面的模型编辑方法，能够在编码任务中修补LLM模型，并且在API序列推荐、代码生成和伪代码到代码转换等任务中得到了验证和评估。

    

    大型语言模型在软件工程中得到了成功应用，特别是在代码生成方面。更新这些模型的新知识非常昂贵，通常需要全面实现其价值。在本文中，我们提出了一种新颖有效的模型编辑方法MENT，用于在编码任务中修补LLM模型。基于生成式LLM的机制，MENT可以在预测下一个令牌时进行模型编辑，并进一步支持常见的编码任务。MENT具有高效、有效和可靠的特点。它可以通过修补1或2个神经元来纠正神经模型。作为神经元层面上生成模型编辑的先驱工作，我们规范了编辑过程并介绍了相关概念。此外，我们还引入了新的衡量方法来评估其泛化能力，并建立了一个用于进一步研究的基准。我们的方法在三个编码任务上进行了评估，包括API序列推荐、行级代码生成和伪代码到代码转换。

    Large Language Models are successfully adopted in software engineering, especially in code generation. Updating these models with new knowledge is very expensive, and is often required to fully realize their value. In this paper, we propose a novel and effective model editing approach, \textsc{MENT}, to patch LLMs in coding tasks. Based on the mechanism of generative LLMs, \textsc{MENT} enables model editing in next-token predictions, and further supports common coding tasks. \textsc{MENT} is effective, efficient, and reliable. It can correct a neural model by patching 1 or 2 neurons. As the pioneer work on neuron-level model editing of generative models, we formalize the editing process and introduce the involved concepts. Besides, we also introduce new measures to evaluate its generalization ability, and build a benchmark for further study. Our approach is evaluated on three coding tasks, including API-seq recommendation, line-level code generation, and pseudocode-to-code transaction
    
[^2]: 富含语义知识增强的大型语言模型用于少样本中文拼写检查

    Rich Semantic Knowledge Enhanced Large Language Models for Few-shot Chinese Spell Checking

    [https://arxiv.org/abs/2403.08492](https://arxiv.org/abs/2403.08492)

    本文使用富含语义知识的大型语言模型在少样本中文拼写检查任务上取得了比BERT模型更好的性能。

    

    本文探讨了使用一种名为RS-LLM（基于丰富语义的LLMs）的上下文学习方法将大型语言模型（LLMs）引入作为基础模型，以及在我们的框架中引入各种中文丰富语义信息的影响。实验结果表明，通过引入少量特定的中文丰富语义结构，LLMs在少样本中文拼写检查任务上比基于BERT模型表现更好。

    arXiv:2403.08492v1 Announce Type: new  Abstract: Chinese Spell Checking (CSC) is a widely used technology, which plays a vital role in speech to text (STT) and optical character recognition (OCR). Most of the existing CSC approaches relying on BERT architecture achieve excellent performance. However, limited by the scale of the foundation model, BERT-based method does not work well in few-shot scenarios, showing certain limitations in practical applications. In this paper, we explore using an in-context learning method named RS-LLM (Rich Semantic based LLMs) to introduce large language models (LLMs) as the foundation model. Besides, we study the impact of introducing various Chinese rich semantic information in our framework. We found that by introducing a small number of specific Chinese rich semantic structures, LLMs achieve better performance than the BERT-based model on few-shot CSC task. Furthermore, we conduct experiments on multiple datasets, and the experimental results verifie
    
[^3]: 基于语音的基础语法：自发联接的自监督深度神经网络

    Basic syntax from speech: Spontaneous concatenation in unsupervised deep neural networks. (arXiv:2305.01626v1 [cs.CL])

    [http://arxiv.org/abs/2305.01626](http://arxiv.org/abs/2305.01626)

    该论文提出了一种基于语音的完全无监督的方法，可以直接从原始语音中建立基础语法模型。作者发现，在基于声音的单词记录上训练的卷积神经网络可以自发连接两个或三个单词，并且可以学会将单词嵌入到新的未见过的单词组合中，这是之前未报道的属性，这一发现对我们理解神经网络的学习方式和建立从原始声学输入中的语法及其演化的模型都有重要的意义。

    

    语法的计算模型主要基于文本。本文提出了一种完全无监督的方法，可以直接从原始语音中建立基础语法模型。我们重点研究了最普遍和基本的语法特性之一——联接。我们介绍了自发联接现象：卷积神经网络(CNN)在个别单词的声学记录上训练时，开始产生输出，这些输出将两个甚至三个单词连接在一起，而不会接触到具有多个单词的输入数据。此外，训练两个单词的网络可以学习将单词嵌入到新的未见过的单词组合中。据我们所知，这是在生成对抗网络环境下训练的原始语音CNN以前未报道的属性，它不仅对我们理解这些体系结构的学习方式有影响，还对建立从原始声学输入中的语法及其演化的模型有影响。

    Computational models of syntax are predominantly text-based. Here we propose that basic syntax can be modeled directly from raw speech in a fully unsupervised way. We focus on one of the most ubiquitous and basic properties of syntax -- concatenation. We introduce spontaneous concatenation: a phenomenon where convolutional neural networks (CNNs) trained on acoustic recordings of individual words start generating outputs with two or even three words concatenated without ever accessing data with multiple words in the input. Additionally, networks trained on two words learn to embed words into novel unobserved word combinations. To our knowledge, this is a previously unreported property of CNNs trained on raw speech in the Generative Adversarial Network setting and has implications both for our understanding of how these architectures learn as well as for modeling syntax and its evolution from raw acoustic inputs.
    

