# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning](https://arxiv.org/abs/2404.02127) | 本研究提出了一个名为LawInstruct的大型法律指导数据集，证明了领域特定的预训练和指导调整可以改善在LegalBench上的性能，为在法律领域开发具有更强信息处理和决策能力的模型提供了一个资源。 |
| [^2] | [Counter-intuitive: Large Language Models Can Better Understand Knowledge Graphs Than We Thought](https://arxiv.org/abs/2402.11541) | 本文通过对KG知识注入方法进行全面比较，探索为LLMs提供知识图谱知识的最佳方法，以增强它们的理解能力。 |
| [^3] | [Machine Translation Models are Zero-Shot Detectors of Translation Direction.](http://arxiv.org/abs/2401.06769) | 本文探索了一种基于无监督方法的翻译方向检测，并通过实验证实其在高负载语言对上的有效性。论文标题为“Machine Translation Models are Zero-Shot Detectors of Translation Direction”。 |
| [^4] | [Noor-Ghateh: A Benchmark Dataset for Evaluating Arabic Word Segmenters in Hadith Domain.](http://arxiv.org/abs/2307.09630) | 本研究提出了一个用于评估阿拉伯词分割方法的基准数据集，通过分析历史和宗教文本，帮助理解文本中的意义。这个数据集在词汇量和种类上都优于其他现有数据集，并且在阿拉伯哈迪斯领域是首个数据集。通过多种方法对数据集进行了评估并报告了注释质量。 |

# 详细

[^1]: FLawN-T5: 有效指导调整数据混合在法律推理中的实证研究

    FLawN-T5: An Empirical Examination of Effective Instruction-Tuning Data Mixtures for Legal Reasoning

    [https://arxiv.org/abs/2404.02127](https://arxiv.org/abs/2404.02127)

    本研究提出了一个名为LawInstruct的大型法律指导数据集，证明了领域特定的预训练和指导调整可以改善在LegalBench上的性能，为在法律领域开发具有更强信息处理和决策能力的模型提供了一个资源。

    

    arXiv:2404.02127v1  公告类型: 跨领域  摘要: 指导调整是使语言模型对直接用户交互有效的重要步骤。然而，许多法律任务仍然超出了大多数开放式LLMs的范围，而且目前该领域还没有任何大规模的数据集。这严重限制了该应用领域的研究。在这项工作中，我们策划了一个名为LawInstruct的大型法律指导数据集，涵盖了17个司法管辖区、24种语言，总计1200万个示例。我们呈现证据表明，领域特定的预训练和指导调整能够改善在LegalBench上的性能，包括将Flan-T5 XL在基准线上提高8个点或16%。然而，该效应并不适用于所有任务、训练模式、模型大小和其他因素。LawInstruct是一个资源，可以加速在法律领域开发具有更强信息处理和决策能力的模型。

    arXiv:2404.02127v1 Announce Type: cross  Abstract: Instruction tuning is an important step in making language models useful for direct user interaction. However, many legal tasks remain out of reach for most open LLMs and there do not yet exist any large scale instruction datasets for the domain. This critically limits research in this application area. In this work, we curate LawInstruct, a large legal instruction dataset, covering 17 jurisdictions, 24 languages and a total of 12M examples. We present evidence that domain-specific pretraining and instruction tuning improve performance on LegalBench, including improving Flan-T5 XL by 8 points or 16\% over the baseline. However, the effect does not generalize across all tasks, training regimes, model sizes, and other factors. LawInstruct is a resource for accelerating the development of models with stronger information processing and decision making capabilities in the legal domain.
    
[^2]: 逆向认知：大型语言模型比我们想象的更擅长理解知识图谱

    Counter-intuitive: Large Language Models Can Better Understand Knowledge Graphs Than We Thought

    [https://arxiv.org/abs/2402.11541](https://arxiv.org/abs/2402.11541)

    本文通过对KG知识注入方法进行全面比较，探索为LLMs提供知识图谱知识的最佳方法，以增强它们的理解能力。

    

    虽然通过使用知识图谱（KGs）来增强大型语言模型（LLMs）的推理能力并减少它们的幻觉的方法受到了广泛关注，但目前对如何使LLMs能够即时整合KGs中的结构化知识的探索还不足。本文采用复杂问题回答（CQA）作为一项任务，评估LLM理解KG知识的能力。我们对KG知识注入方法进行了全面比较（从三元组到自然语言文本），旨在探索为LLMs提供KG知识的最佳提示方法，从而增强它们的理解能力。

    arXiv:2402.11541v1 Announce Type: cross  Abstract: Although the method of enhancing large language models' (LLMs') reasoning ability and reducing their hallucinations through the use of knowledge graphs (KGs) has received widespread attention, the exploration of how to enable LLMs to integrate the structured knowledge in KGs on-the-fly remains inadequate. Researchers often co-train KG embeddings and LLM parameters to equip LLMs with the ability of comprehending KG knowledge. However, this resource-hungry training paradigm significantly increases the model learning cost and is also unsuitable for non-open-source, black-box LLMs. In this paper, we employ complex question answering (CQA) as a task to assess the LLM's ability of comprehending KG knowledge. We conducted a comprehensive comparison of KG knowledge injection methods (from triples to natural language text), aiming to explore the optimal prompting method for supplying KG knowledge to LLMs, thereby enhancing their comprehension o
    
[^3]: 机器翻译模型是零射击的翻译方向检测器。

    Machine Translation Models are Zero-Shot Detectors of Translation Direction. (arXiv:2401.06769v1 [cs.CL])

    [http://arxiv.org/abs/2401.06769](http://arxiv.org/abs/2401.06769)

    本文探索了一种基于无监督方法的翻译方向检测，并通过实验证实其在高负载语言对上的有效性。论文标题为“Machine Translation Models are Zero-Shot Detectors of Translation Direction”。

    

    检测并行文本的翻译方向对于机器翻译的训练和评估具有应用价值，但也具有法医应用，例如解决剽窃或伪造指控。在这项工作中，我们根据一个简单的假设，即$p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$，以传统上被称为翻译语或机器翻译语中的简化效应为动机，探索了一种无监督的翻译方向检测方法。通过对20个翻译方向进行大规模多语种机器翻译模型的实验，我们验证了该方法在资源丰富的语言对上的有效性，对于NMT生成的翻译，实现了文档级准确率为82-96％，对于人工翻译，根据所使用的模型，实现了60-81％的准确率。代码和演示可在https://github.com/ZurichNLP/translation-direction-detection找到。

    Detecting the translation direction of parallel text has applications for machine translation training and evaluation, but also has forensic applications such as resolving plagiarism or forgery allegations. In this work, we explore an unsupervised approach to translation direction detection based on the simple hypothesis that $p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$, motivated by the well-known simplification effect in translationese or machine-translationese. In experiments with massively multilingual machine translation models across 20 translation directions, we confirm the effectiveness of the approach for high-resource language pairs, achieving document-level accuracies of 82-96% for NMT-produced translations, and 60-81% for human translations, depending on the model used. Code and demo are available at https://github.com/ZurichNLP/translation-direction-detection
    
[^4]: Noor-Ghateh: 一个用于评估哈迪斯领域阿拉伯词分割器的基准数据集

    Noor-Ghateh: A Benchmark Dataset for Evaluating Arabic Word Segmenters in Hadith Domain. (arXiv:2307.09630v1 [cs.CL])

    [http://arxiv.org/abs/2307.09630](http://arxiv.org/abs/2307.09630)

    本研究提出了一个用于评估阿拉伯词分割方法的基准数据集，通过分析历史和宗教文本，帮助理解文本中的意义。这个数据集在词汇量和种类上都优于其他现有数据集，并且在阿拉伯哈迪斯领域是首个数据集。通过多种方法对数据集进行了评估并报告了注释质量。

    

    阿拉伯语具有许多复杂而丰富的形态学细微差别，这在分析传统的阿拉伯文本，特别是在历史和宗教语境中，对于理解文本的含义非常有用。词汇分离意味着将词语分解为诸如词根和词缀等不同部分。在形态学数据集中，标签的多样性和数据样本的数量有助于评估形态学方法。本文提出了一个基准数据集，用于评估分离阿拉伯词汇的方法，该数据集包含来自《伊斯兰教法》的约223,690个词汇，已由专家进行标记。就词汇量和种类而言，该数据集优于其他现有数据集，并且据我们所知，不存在阿拉伯哈迪斯领域的文本。为了评估该数据集，我们对数据集应用了不同的方法，如Farasa、Camel、Madamira和ALP，并通过四个参数报告了注释质量。

    There are many complex and rich morphological subtleties in the Arabic language, which are very useful when analyzing traditional Arabic texts, especially in the historical and religious contexts, and help in understanding the meaning of the texts. Vocabulary separation means separating the word into different parts such as root and affix. In the morphological datasets, the variety of labels and the number of data samples helps to evaluate the morphological methods. In this paper, we present a benchmark data set for evaluating the methods of separating Arabic words which include about 223,690 words from the book of Sharia alIslam, which have been labeled by experts. In terms of the volume and variety of words, this dataset is superior to other existing data sets, and as far as we know, there are no Arabic Hadith Domain texts. To evaluate the dataset, we applied different methods such as Farasa, Camel, Madamira, and ALP to the dataset and we reported the annotation quality through four 
    

