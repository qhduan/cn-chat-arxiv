# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Induced Model Matching: How Restricted Models Can Help Larger Ones](https://arxiv.org/abs/2402.12513) | 提出了引导模型匹配（IMM）方法，通过使完整模型的性能与受限模型对齐，将受限模型的知识传递给完整模型，具有广泛的应用性。 |
| [^3] | [CroissantLLM: A Truly Bilingual French-English Language Model](https://arxiv.org/abs/2402.00786) | CroissantLLM是一个1.3B的双语语言模型，通过使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集进行训练，实现了高性能和开源。模型还发布了训练数据集和多个检查点，以及一个法语基准测试 FrenchBench。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: 引导模型匹配：受限模型如何帮助更大的模型

    Induced Model Matching: How Restricted Models Can Help Larger Ones

    [https://arxiv.org/abs/2402.12513](https://arxiv.org/abs/2402.12513)

    提出了引导模型匹配（IMM）方法，通过使完整模型的性能与受限模型对齐，将受限模型的知识传递给完整模型，具有广泛的应用性。

    

    我们考虑在训练更大、具有完整特征的模型时，是否可以利用受限特征的非常准确的预测模型。这个受限模型可以被视为“辅助信息”，可以通过来自辅助源数据集的详尽数据或在相同数据集上通过施加限制来获得。我们提出了一种方法，将受限模型的知识传递给完整模型，通过使完整模型的上下文受限性能与受限模型的性能对齐。我们将这种方法称为引导模型匹配（IMM），首先通过以逻辑回归为玩具示例来说明其普适性。然后我们探讨了IMM在语言建模中的应用，这也是最初的灵感来源，IMM在这里提供了明确的基础，与在技术中隐式使用受限模型的方法相对应，例如添加噪声。

    arXiv:2402.12513v1 Announce Type: new  Abstract: We consider scenarios where a very accurate predictive model using restricted features is available at the time of training of a larger, full-featured, model. This restricted model may be thought of as "side-information", derived either from an auxiliary exhaustive dataset or on the same dataset, by forcing the restriction. How can the restricted model be useful to the full model? We propose an approach for transferring the knowledge of the restricted model to the full model, by aligning the full model's context-restricted performance with that of the restricted model's. We call this methodology Induced Model Matching (IMM) and first illustrate its general applicability by using logistic regression as a toy example. We then explore IMM's use in language modeling, the application that initially inspired it, and where it offers an explicit foundation in contrast to the implicit use of restricted models in techniques such as noising. We dem
    
[^3]: CroissantLLM: 一个真正的双语法语-英语语言模型

    CroissantLLM: A Truly Bilingual French-English Language Model

    [https://arxiv.org/abs/2402.00786](https://arxiv.org/abs/2402.00786)

    CroissantLLM是一个1.3B的双语语言模型，通过使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集进行训练，实现了高性能和开源。模型还发布了训练数据集和多个检查点，以及一个法语基准测试 FrenchBench。

    

    我们介绍了CroissantLLM，这是一个在3T个英语和法语标记上预训练的13亿语言模型，为研究和工业社区带来了一种高性能的、完全开源的双语模型，可以在消费级本地硬件上快速运行。为此，我们首次尝试使用1:1的英语-法语预训练数据比例、自定义的分词器和双语调优数据集来训练一种内在双语的模型。我们发布了训练数据集，其中包含了一个法语分割，其中包含了手工策划、高质量和多样化的数据源。为了评估在英语以外的性能，我们创建了一个新的基准测试 FrenchBench，包括一系列分类和生成任务，涵盖了模型在法语语言中性能的各个方面。此外，为了保持透明度并促进进一步的大规模语言模型研究，我们发布了代码库和各种模型规模、训练数据分布上的几十个检查点。

    We introduce CroissantLLM, a 1.3B language model pretrained on a set of 3T English and French tokens, to bring to the research and industrial community a high-performance, fully open-sourced bilingual model that runs swiftly on consumer-grade local hardware. To that end, we pioneer the approach of training an intrinsically bilingual model with a 1:1 English-to-French pretraining data ratio, a custom tokenizer, and bilingual finetuning datasets. We release the training dataset, notably containing a French split with manually curated, high-quality, and varied data sources. To assess performance outside of English, we craft a novel benchmark, FrenchBench, consisting of an array of classification and generation tasks, covering various orthogonal aspects of model performance in the French Language. Additionally, rooted in transparency and to foster further Large Language Model research, we release codebases, and dozens of checkpoints across various model sizes, training data distributions, 
    

