# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic change detection for Slovene language: a novel dataset and an approach based on optimal transport](https://arxiv.org/abs/2402.16596) | 该论文提出了一个新的斯洛文尼亚语数据集，用于评估语义变化检测系统，并提出了一种基于最优输运的全新方法，能够将错误率降低22.8%。 |
| [^2] | [Machine Translation Models are Zero-Shot Detectors of Translation Direction.](http://arxiv.org/abs/2401.06769) | 本文探索了一种基于无监督方法的翻译方向检测，并通过实验证实其在高负载语言对上的有效性。论文标题为“Machine Translation Models are Zero-Shot Detectors of Translation Direction”。 |

# 详细

[^1]: Slovene语义变化检测：一个新的数据集和基于最优输运的方法

    Semantic change detection for Slovene language: a novel dataset and an approach based on optimal transport

    [https://arxiv.org/abs/2402.16596](https://arxiv.org/abs/2402.16596)

    该论文提出了一个新的斯洛文尼亚语数据集，用于评估语义变化检测系统，并提出了一种基于最优输运的全新方法，能够将错误率降低22.8%。

    

    在这篇论文中，我们专注于检测斯洛文尼亚语中的语义变化，这是一种拥有两百万使用者的资源较少的斯拉夫语言。检测和跟踪语义变化可以揭示语言的演变，这是由社会和文化变化引起的。最近，已经提出了几种系统来帮助这项研究，但所有这些系统都依赖于手动注释的黄金标准数据集进行评估。本文提出了第一个斯洛文尼亚语数据集，用于评估语义变化检测系统，其中包含了从3000多个手动注释的句对中获得的104个目标词的汇总语义变化评分。我们在这个数据集上评估了几种已有的语义变化检测方法，并提出了一种基于最优输运的新方法，该方法改进了现有的最先进系统，错误率降低了22.8%。

    arXiv:2402.16596v1 Announce Type: new  Abstract: In this paper, we focus on the detection of semantic changes in Slovene, a less resourced Slavic language with two million speakers. Detecting and tracking semantic changes provides insights into the evolution of the language caused by changes in society and culture. Recently, several systems have been proposed to aid in this study, but all depend on manually annotated gold standard datasets for evaluation. In this paper, we present the first Slovene dataset for evaluating semantic change detection systems, which contains aggregated semantic change scores for 104 target words obtained from more than 3000 manually annotated sentence pairs. We evaluate several existing semantic change detection methods on this dataset and also propose a novel approach based on optimal transport that improves on the existing state-of-the-art systems with an error reduction rate of 22.8%.
    
[^2]: 机器翻译模型是零射击的翻译方向检测器。

    Machine Translation Models are Zero-Shot Detectors of Translation Direction. (arXiv:2401.06769v1 [cs.CL])

    [http://arxiv.org/abs/2401.06769](http://arxiv.org/abs/2401.06769)

    本文探索了一种基于无监督方法的翻译方向检测，并通过实验证实其在高负载语言对上的有效性。论文标题为“Machine Translation Models are Zero-Shot Detectors of Translation Direction”。

    

    检测并行文本的翻译方向对于机器翻译的训练和评估具有应用价值，但也具有法医应用，例如解决剽窃或伪造指控。在这项工作中，我们根据一个简单的假设，即$p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$，以传统上被称为翻译语或机器翻译语中的简化效应为动机，探索了一种无监督的翻译方向检测方法。通过对20个翻译方向进行大规模多语种机器翻译模型的实验，我们验证了该方法在资源丰富的语言对上的有效性，对于NMT生成的翻译，实现了文档级准确率为82-96％，对于人工翻译，根据所使用的模型，实现了60-81％的准确率。代码和演示可在https://github.com/ZurichNLP/translation-direction-detection找到。

    Detecting the translation direction of parallel text has applications for machine translation training and evaluation, but also has forensic applications such as resolving plagiarism or forgery allegations. In this work, we explore an unsupervised approach to translation direction detection based on the simple hypothesis that $p(\text{translation}|\text{original})>p(\text{original}|\text{translation})$, motivated by the well-known simplification effect in translationese or machine-translationese. In experiments with massively multilingual machine translation models across 20 translation directions, we confirm the effectiveness of the approach for high-resource language pairs, achieving document-level accuracies of 82-96% for NMT-produced translations, and 60-81% for human translations, depending on the model used. Code and demo are available at https://github.com/ZurichNLP/translation-direction-detection
    

