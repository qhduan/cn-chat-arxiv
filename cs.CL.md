# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models](https://arxiv.org/abs/2403.20331) | 本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。 |
| [^2] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^3] | [RAGAS: Automated Evaluation of Retrieval Augmented Generation.](http://arxiv.org/abs/2309.15217) | RAGAs是一个用于无参考评估检索增强生成（RAG）的框架，能够评估检索系统和生成模块的能力，提供一种加快RAG架构评估周期的方法。 |
| [^4] | [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations.](http://arxiv.org/abs/2305.16326) | 本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。 |
| [^5] | [Generative Meta-Learning for Zero-Shot Relation Triplet Extraction.](http://arxiv.org/abs/2305.01920) | 该论文提出了一种生成式元学习框架，通过任务感知的生成式模型和三种针对典型元学习范畴的方法，提高了零样本关系三元组抽取任务的泛化能力并达到了最佳表现。 |
| [^6] | [NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online.](http://arxiv.org/abs/2303.10430) | 本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。 |

# 详细

[^1]: 不可解问题检测：评估视觉语言模型的可信度

    Unsolvable Problem Detection: Evaluating Trustworthiness of Vision Language Models

    [https://arxiv.org/abs/2403.20331](https://arxiv.org/abs/2403.20331)

    本文提出了一个新颖且重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型在视觉问答任务中能否在面对不可解问题时保持答案的能力，并通过广泛实验发现大多数模型存在改进的空间。

    

    本文介绍了一个新颖而重要的挑战，即Unsolvable Problem Detection（UPD），用于评估视觉语言模型（VLMs）在视觉问答（VQA）任务中面对不可解问题时保持答案的能力。UPD包括三个不同的设置：缺失答案检测（AAD）、不兼容答案集检测（IASD）和不兼容视觉问题检测（IVQD）。通过广泛的实验深入研究UPD问题表明，大多数VLMs，包括GPT-4V和LLaVA-Next-34B，在各种程度上都很难应对我们的基准测试，突显了改进的重要空间。为了解决UPD，我们探索了无需训练和基于训练的解决方案，提供了对其有效性和局限性的新见解。我们希望我们的见解，以及在提议的UPD设置内的未来努力，将增强对VLMs的更广泛理解和发展。

    arXiv:2403.20331v1 Announce Type: cross  Abstract: This paper introduces a novel and significant challenge for Vision Language Models (VLMs), termed Unsolvable Problem Detection (UPD). UPD examines the VLM's ability to withhold answers when faced with unsolvable problems in the context of Visual Question Answering (VQA) tasks. UPD encompasses three distinct settings: Absent Answer Detection (AAD), Incompatible Answer Set Detection (IASD), and Incompatible Visual Question Detection (IVQD). To deeply investigate the UPD problem, extensive experiments indicate that most VLMs, including GPT-4V and LLaVA-Next-34B, struggle with our benchmarks to varying extents, highlighting significant room for the improvements. To address UPD, we explore both training-free and training-based solutions, offering new insights into their effectiveness and limitations. We hope our insights, together with future efforts within the proposed UPD settings, will enhance the broader understanding and development of
    
[^2]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^3]: RAGAS:自动评估检索增强生成

    RAGAS: Automated Evaluation of Retrieval Augmented Generation. (arXiv:2309.15217v1 [cs.CL])

    [http://arxiv.org/abs/2309.15217](http://arxiv.org/abs/2309.15217)

    RAGAs是一个用于无参考评估检索增强生成（RAG）的框架，能够评估检索系统和生成模块的能力，提供一种加快RAG架构评估周期的方法。

    

    我们介绍了RAGAs（检索增强生成评估）框架，用于对检索增强生成（RAG）流水线进行无参考评估。RAG系统由检索模块和基于LLM的生成模块组成，提供来自参考文本数据库的知识给LLMs，使它们能够充当用户和文本数据库之间的自然语言层，减少幻觉的风险。然而，评估RAG架构是具有挑战性的，因为有几个维度需要考虑：检索系统识别相关和有重点的上下文段落的能力，LLM在忠实地利用这些段落的能力，以及生成本身的质量。通过RAGAs，我们提出了一套度量标准，可以用来评估这些不同维度，而无需依赖地面真实的人类注释。我们认为，这样的框架能够对RAG架构的更快评估周期起到至关重要的贡献。

    We introduce RAGAs (Retrieval Augmented Generation Assessment), a framework for reference-free evaluation of Retrieval Augmented Generation (RAG) pipelines. RAG systems are composed of a retrieval and an LLM based generation module, and provide LLMs with knowledge from a reference textual database, which enables them to act as a natural language layer between a user and textual databases, reducing the risk of hallucinations. Evaluating RAG architectures is, however, challenging because there are several dimensions to consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithful way, or the quality of the generation itself. With RAGAs, we put forward a suite of metrics which can be used to evaluate these different dimensions \textit{without having to rely on ground truth human annotations}. We posit that such a framework can crucially contribute to faster evaluation cycles of RAG architectur
    
[^4]: 生物医学自然语言处理中的大型语言模型: 基准、基线和建议

    Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations. (arXiv:2305.16326v1 [cs.CL])

    [http://arxiv.org/abs/2305.16326](http://arxiv.org/abs/2305.16326)

    本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。

    

    生物医学文献呈指数级增长，手动筛选和提取知识变得困难。自动从生物医学文献中提取信息的生物医学自然语言处理（BioNLP）技术有助于减轻这种负担。近年来，如GPT-3和GPT-4等大型语言模型（LLMs）因其卓越的性能而受到重视。但是，它们在BioNLP任务中的有效性以及对方法开发和下游用户的影响仍未得到研究。本研究（1）在四个应用程序中在八个BioNLP数据集中建立了GPT-3和GPT-4在零-shot和一-shot设置下的基准表现，包括命名实体识别，关系提取，多标签文档分类和语义相似性和推理；（2）审查了LLMs产生的错误，并将错误分为三种类型：缺失，不一致和不需要的人工内容；（3）提出了使用LLMs的建议。

    Biomedical literature is growing rapidly, making it challenging to curate and extract knowledge manually. Biomedical natural language processing (BioNLP) techniques that can automatically extract information from biomedical literature help alleviate this burden. Recently, large Language Models (LLMs), such as GPT-3 and GPT-4, have gained significant attention for their impressive performance. However, their effectiveness in BioNLP tasks and impact on method development and downstream users remain understudied. This pilot study (1) establishes the baseline performance of GPT-3 and GPT-4 at both zero-shot and one-shot settings in eight BioNLP datasets across four applications: named entity recognition, relation extraction, multi-label document classification, and semantic similarity and reasoning, (2) examines the errors produced by the LLMs and categorized the errors into three types: missingness, inconsistencies, and unwanted artificial content, and (3) provides suggestions for using L
    
[^5]: 零样本关系三元组抽取的生成式元学习

    Generative Meta-Learning for Zero-Shot Relation Triplet Extraction. (arXiv:2305.01920v1 [cs.CL])

    [http://arxiv.org/abs/2305.01920](http://arxiv.org/abs/2305.01920)

    该论文提出了一种生成式元学习框架，通过任务感知的生成式模型和三种针对典型元学习范畴的方法，提高了零样本关系三元组抽取任务的泛化能力并达到了最佳表现。

    

    零样本关系三元组抽取任务旨在从一个包含未见过关系类型的文本中提取关系三元组。比较有代表性的工作采用预训练的生成式模型为新关系生成合成样本。然而，当前的生成式模型在训练中缺乏对于模型泛化到不同任务的优化过程，因此具有有限的泛化能力。因此，我们提出了一种新颖的生成式元学习框架，利用元学习的“学习如何学习”的能力提高生成式模型的泛化能力。具体而言，我们首先设计了一个任务感知的生成式模型，它可以通过在多个任务上强制进行优化过程来学习一般性知识。基于此，我们提出了三种针对三类典型元学习范畴的生成式元学习方法。广泛的实验结果表明，我们的框架在零样本关系三元组抽取任务上实现了新的最佳表现。

    The zero-shot relation triplet extraction (ZeroRTE) task aims to extract relation triplets from a piece of text with unseen relation types. The seminal work adopts the pre-trained generative model to generate synthetic samples for new relations. However, current generative models lack the optimization process of model generalization on different tasks during training, and thus have limited generalization capability. For this reason, we propose a novel generative meta-learning framework which exploits the `learning-to-learn' ability of meta-learning to boost the generalization capability of generative models. Specifically, we first design a task-aware generative model which can learn the general knowledge by forcing the optimization process to be conducted across multiple tasks. Based on it, we then present three generative meta-learning approaches designated for three typical meta-learning categories. Extensive experimental results demonstrate that our framework achieves a new state-of
    
[^6]: NoisyHate：在人类编写的在线扰动下对内容审核机器学习模型进行基准测试

    NoisyHate: Benchmarking Content Moderation Machine Learning Models with Human-Written Perturbations Online. (arXiv:2303.10430v1 [cs.LG])

    [http://arxiv.org/abs/2303.10430](http://arxiv.org/abs/2303.10430)

    本文提出了一个包含人类编写的在线扰动的测试集，用于毒性言论检测模型的评估。

    

    在社交媒体上，具有有害内容的在线文本是一种威胁，可能会引起网络骚扰。尽管许多平台采取了措施，例如基于机器学习的仇恨言论检测系统来减少其影响，但那些有害内容发布者仍然可以通过修改有害词汇的拼写来逃避系统。这些修改后的单词也称为人类编写的文本扰动。许多研究开发了一定的技术来生成对抗样本，以帮助机器学习模型获得识别这些扰动的能力。然而，机器生成的扰动与人类编写的扰动之间仍存在差距。在本文中，我们介绍了一个包含人类编写的在线扰动的基准测试集，用于毒性言论检测模型。我们还招募了一组工人来评估此测试集的质量并删除低质量的样本。同时，为了检查我们的扰动是否可以归一化为其干净版本，我们还创建了一个相关的测试集。

    Online texts with toxic content are a threat in social media that might cause cyber harassment. Although many platforms applied measures, such as machine learning-based hate-speech detection systems, to diminish their effect, those toxic content publishers can still evade the system by modifying the spelling of toxic words. Those modified words are also known as human-written text perturbations. Many research works developed certain techniques to generate adversarial samples to help the machine learning models obtain the ability to recognize those perturbations. However, there is still a gap between those machine-generated perturbations and human-written perturbations. In this paper, we introduce a benchmark test set containing human-written perturbations online for toxic speech detection models. We also recruited a group of workers to evaluate the quality of this test set and dropped low-quality samples. Meanwhile, to check if our perturbation can be normalized to its clean version, w
    

