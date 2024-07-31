# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Auxiliary task demands mask the capabilities of smaller language models](https://arxiv.org/abs/2404.02418) | 较小语言模型对类比推理、反思推理、单词预测和语法判断的表现受辅助任务需求的影响，评估方法的任务需求越大，性能越低，这种"需求差距"在参数较少、训练数据较少的模型中尤为显著 |
| [^2] | [GigaPevt: Multimodal Medical Assistant](https://arxiv.org/abs/2402.16654) | GigaPevt是第一个结合大型语言模型和专业医疗模型的多模态医疗助手，在对话质量和度量性能方面表现出明显优势，并在问答任务中提高了1.18\%的准确率。 |
| [^3] | [ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic](https://arxiv.org/abs/2402.12840) | ArabicMMLU是针对阿拉伯语的第一个多任务语言理解基准测试，通过学校考试中收集的数据对35个模型进行全面评估，揭示了在阿拉伯语中性能改进的潜力。 |
| [^4] | [C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.03181) | C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。 |
| [^5] | [ScreenQA: Large-Scale Question-Answer Pairs over Mobile App Screenshots](https://arxiv.org/abs/2209.08199) | ScreenQA提出了一个新的任务和数据集，通过86K个问答对在RICO数据集上注释，旨在评估屏幕阅读理解能力。 |
| [^6] | [Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers.](http://arxiv.org/abs/2401.06461) | 本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。 |
| [^7] | [Language-based Valence and Arousal Expressions between the United States and China: a Cross-Cultural Examination.](http://arxiv.org/abs/2401.05254) | 本文从跨文化的角度研究了美国和中国社交媒体上的情感表达之间的差异。研究发现，与美国Twitter用户相比，中国新浪微博用户在情感强度的变化和激动程度上有更明显的差异。 |
| [^8] | [Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences.](http://arxiv.org/abs/2310.11960) | 提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。 |
| [^9] | [Auto-Regressive Next-Token Predictors are Universal Learners.](http://arxiv.org/abs/2309.06979) | 自回归的下一个标记预测器可以有效地近似图灵机计算的任何函数，并且在文本生成和算术任务上展现出非平凡的性能。 |
| [^10] | [A Framework for Responsible Development of Automated Student Feedback with Generative AI.](http://arxiv.org/abs/2308.15334) | 一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。 |
| [^11] | [A Survey on Model Compression for Large Language Models.](http://arxiv.org/abs/2308.07633) | 本论文提供了关于大型语言模型的模型压缩综述，探讨了量化、修剪、知识蒸馏等不同方法，并突出介绍了最新进展和创新方法，为实现高效的部署提供了重要思路。 |

# 详细

[^1]: 辅助任务需求掩盖了较小语言模型的能力

    Auxiliary task demands mask the capabilities of smaller language models

    [https://arxiv.org/abs/2404.02418](https://arxiv.org/abs/2404.02418)

    较小语言模型对类比推理、反思推理、单词预测和语法判断的表现受辅助任务需求的影响，评估方法的任务需求越大，性能越低，这种"需求差距"在参数较少、训练数据较少的模型中尤为显著

    

    发展心理学家们对认知能力如语言理解或心灵理论何时出现进行了争论。这些辩论常常关注"任务需求"的概念--执行特定评估时所伴随的辅助挑战--这些挑战可能掩盖了儿童的潜在能力。当衡量语言模型（LMs）的能力时，同样的问题也会出现：任务表现取决于模型的基本能力，结合了模型解释和执行任务的能力以及其可用资源。在这里，我们展示了对于类比推理、反思推理、单词预测和语法判断，具有更大任务需求的评估方法会比降低需求的评估得到更低的性能。这种"需求差距"在参数较少、训练数据较少的模型中最为显著。我们的结果表明，LM的性能不应被解释为

    arXiv:2404.02418v1 Announce Type: cross  Abstract: Developmental psychologists have argued about when cognitive capacities such as language understanding or theory of mind emerge. These debates often hinge on the concept of "task demands" -- the auxiliary challenges associated with performing a particular evaluation -- that may mask the child's underlying ability. The same issues arise when measuring the capacities of language models (LMs): performance on a task is a function of the model's underlying competence, combined with the model's ability to interpret and perform the task given its available resources. Here, we show that for analogical reasoning, reflective reasoning, word prediction, and grammaticality judgments, evaluation methods with greater task demands yield lower performance than evaluations with reduced demands. This "demand gap" is most pronounced for models with fewer parameters and less training data. Our results illustrate that LM performance should not be interpret
    
[^2]: GigaPevt：多模态医疗助手

    GigaPevt: Multimodal Medical Assistant

    [https://arxiv.org/abs/2402.16654](https://arxiv.org/abs/2402.16654)

    GigaPevt是第一个结合大型语言模型和专业医疗模型的多模态医疗助手，在对话质量和度量性能方面表现出明显优势，并在问答任务中提高了1.18\%的准确率。

    

    建立一个智能高效的医疗助手仍然是一个具有挑战性的人工智能问题。主要限制来自数据模态的稀缺性，降低了全面的患者感知。本演示论文介绍了GigaPevt，这是第一个结合了大型语言模型的对话功能和专业医疗模型的多模态医疗助手。这种方法在对话质量和度量性能方面具有明显优势，使得在问答任务中准确率提高了1.18\%。

    arXiv:2402.16654v1 Announce Type: cross  Abstract: Building an intelligent and efficient medical assistant is still a challenging AI problem. The major limitation comes from the data modality scarceness, which reduces comprehensive patient perception. This demo paper presents the GigaPevt, the first multimodal medical assistant that combines the dialog capabilities of large language models with specialized medical models. Such an approach shows immediate advantages in dialog quality and metric performance, with a 1.18\% accuracy improvement in the question-answering task.
    
[^3]: ArabicMMLU：评估阿拉伯语中的大规模多任务语言理解

    ArabicMMLU: Assessing Massive Multitask Language Understanding in Arabic

    [https://arxiv.org/abs/2402.12840](https://arxiv.org/abs/2402.12840)

    ArabicMMLU是针对阿拉伯语的第一个多任务语言理解基准测试，通过学校考试中收集的数据对35个模型进行全面评估，揭示了在阿拉伯语中性能改进的潜力。

    

    语言模型评估的重点已经转向推理和知识密集型任务，这得益于预训练大型模型的进展。尽管最先进的模型部分在大量阿拉伯文本上进行了训练，但由于相关数据集的有限可用性，评估它们在阿拉伯语中的性能仍然具有挑战性。为了弥合这一差距，我们提出了ArabicMMLU，这是第一个针对阿拉伯语言的多任务语言理解基准测试，其数据来自于跨越北非、黎凡特和海湾地区不同国家教育水平的学校考试。我们的数据包括40个任务和14,575个现代标准阿拉伯语（MSA）的多项选择题，通过与该地区的母语者合作精心构建。我们对35个模型的全面评估显示出相当大的改进空间，特别是在最好的开源模型中。值得注意的是BLOOMZ、mT0、LLama2和Fa。

    arXiv:2402.12840v1 Announce Type: new  Abstract: The focus of language model evaluation has transitioned towards reasoning and knowledge-intensive tasks, driven by advancements in pretraining large models. While state-of-the-art models are partially trained on large Arabic texts, evaluating their performance in Arabic remains challenging due to the limited availability of relevant datasets. To bridge this gap, we present ArabicMMLU, the first multi-task language understanding benchmark for Arabic language, sourced from school exams across diverse educational levels in different countries spanning North Africa, the Levant, and the Gulf regions. Our data comprises 40 tasks and 14,575 multiple-choice questions in Modern Standard Arabic (MSA), and is carefully constructed by collaborating with native speakers in the region. Our comprehensive evaluations of 35 models reveal substantial room for improvement, particularly among the best open-source models. Notably, BLOOMZ, mT0, LLama2, and Fa
    
[^4]: C-RAG: 针对检索增强语言模型的认证生成风险

    C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models

    [https://arxiv.org/abs/2402.03181](https://arxiv.org/abs/2402.03181)

    C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。

    

    尽管大型语言模型（LLMs）在各种应用中具备令人印象深刻的能力，但它们仍然存在可信度问题，如幻觉和错位。检索增强语言模型（RAG）被提出来增强生成结果的可信性，通过引入外部知识。但是，对于RAG模型的生成风险的理论理解尚未被研究。本文回答了以下问题：1）RAG是否确实能够降低生成风险，2）如何对RAG和传统LLM的生成风险提供可证明的保证，以及3）哪些充分条件使得RAG模型能够降低生成风险。我们提出了C-RAG，第一个用于认证RAG模型生成风险的框架。具体而言，我们为RAG模型提供了符合风险分析，并确保了生成风险的上界，我们称之为符合生成风险。我们还对一般有界风险下的符合生成风险提供了理论保证。

    Despite the impressive capabilities of large language models (LLMs) across diverse applications, they still suffer from trustworthiness issues, such as hallucinations and misalignments. Retrieval-augmented language models (RAG) have been proposed to enhance the credibility of generations by grounding external knowledge, but the theoretical understandings of their generation risks remains unexplored. In this paper, we answer: 1) whether RAG can indeed lead to low generation risks, 2) how to provide provable guarantees on the generation risks of RAG and vanilla LLMs, and 3) what sufficient conditions enable RAG models to reduce generation risks. We propose C-RAG, the first framework to certify generation risks for RAG models. Specifically, we provide conformal risk analysis for RAG models and certify an upper confidence bound of generation risks, which we refer to as conformal generation risk. We also provide theoretical guarantees on conformal generation risks for general bounded risk f
    
[^5]: ScreenQA: 移动应用截图上的大规模问答对

    ScreenQA: Large-Scale Question-Answer Pairs over Mobile App Screenshots

    [https://arxiv.org/abs/2209.08199](https://arxiv.org/abs/2209.08199)

    ScreenQA提出了一个新的任务和数据集，通过86K个问答对在RICO数据集上注释，旨在评估屏幕阅读理解能力。

    

    我们提出了一个新的任务和数据集ScreenQA，用于通过问答来理解屏幕内容。现有的屏幕数据集要么侧重于结构和组件级别的理解，要么侧重于像导航和任务完成之类的更高级别的组合任务。我们试图通过在RICO数据集上注释86K个问答对来弥合这两者之间的差距，希望能够基准化屏幕阅读理解能力。

    arXiv:2209.08199v2 Announce Type: replace  Abstract: We present a new task and dataset, ScreenQA, for screen content understanding via question answering. The existing screen datasets are focused either on structure and component-level understanding, or on a much higher-level composite task such as navigation and task completion. We attempt to bridge the gap between these two by annotating 86K question-answer pairs over the RICO dataset in hope to benchmark the screen reading comprehension capacity.
    
[^6]: 代码之间的界限：揭示机器和人类程序员之间不同的模式

    Between Lines of Code: Unraveling the Distinct Patterns of Machine and Human Programmers. (arXiv:2401.06461v1 [cs.SE])

    [http://arxiv.org/abs/2401.06461](http://arxiv.org/abs/2401.06461)

    本文通过分析代码的属性，揭示了机器和人类代码之间的独特模式，尤其是结构分割对于识别代码来源很关键。基于这些发现，我们提出了一种名为DetectCodeGPT的新方法来检测机器生成的代码。

    

    大型语言模型在代码生成方面取得了显著的进展，但它们模糊了机器和人类源代码之间的区别，导致软件产物的完整性和真实性问题。本文通过对代码长度、词汇多样性和自然性等属性的严格分析，揭示了机器和人类代码固有的独特模式。在我们的研究中特别注意到，代码的结构分割是识别其来源的关键因素。基于我们的发现，我们提出了一种名为DetectCodeGPT的新型机器生成代码检测方法，该方法改进了DetectGPT。

    Large language models have catalyzed an unprecedented wave in code generation. While achieving significant advances, they blur the distinctions between machine-and human-authored source code, causing integrity and authenticity issues of software artifacts. Previous methods such as DetectGPT have proven effective in discerning machine-generated texts, but they do not identify and harness the unique patterns of machine-generated code. Thus, its applicability falters when applied to code. In this paper, we carefully study the specific patterns that characterize machine and human-authored code. Through a rigorous analysis of code attributes such as length, lexical diversity, and naturalness, we expose unique pat-terns inherent to each source. We particularly notice that the structural segmentation of code is a critical factor in identifying its provenance. Based on our findings, we propose a novel machine-generated code detection method called DetectCodeGPT, which improves DetectGPT by cap
    
[^7]: 中美两国之间基于语言的情绪表达的价值和激动对比：一个跨文化的研究

    Language-based Valence and Arousal Expressions between the United States and China: a Cross-Cultural Examination. (arXiv:2401.05254v1 [cs.CY])

    [http://arxiv.org/abs/2401.05254](http://arxiv.org/abs/2401.05254)

    本文从跨文化的角度研究了美国和中国社交媒体上的情感表达之间的差异。研究发现，与美国Twitter用户相比，中国新浪微博用户在情感强度的变化和激动程度上有更明显的差异。

    

    尽管社交媒体上个体的情感表达已经得到了广泛研究，但研究主要集中在西方环境中。不同文化之间存在着引发情感表达的重要差异。本文研究了美国Twitter和中国新浪微博上的两个主要情感维度（价值和激动）之间的差异。我们研究了美国和中国个体之间的激动和价值之间的功能关系差异，并探讨了相关内容上的差异。此外，我们还对两个平台上的词语使用和话题进行了相关性分析，以解读它们之间的差异。我们观察到，对于Twitter用户来说，负面情绪和正面情绪之间的情感强度变化不太明显，而对于新浪微博用户来说，伴随着情感的上升，激动程度有更明显的升级。从语言特征中，我们发现情感表达方面的差异。

    Although affective expressions of individuals have been extensively studied using social media, research has primarily focused on the Western context. There are substantial differences among cultures that contribute to their affective expressions. This paper examines the differences between Twitter (X) in the United States and Sina Weibo posts in China on two primary dimensions of affect - valence and arousal. We study the difference in the functional relationship between arousal and valence (so-called V-shaped) among individuals in the US and China and explore the associated content differences. Furthermore, we correlate word usage and topics in both platforms to interpret their differences. We observe that for Twitter users, the variation in emotional intensity is less distinct between negative and positive emotions compared to Weibo users, and there is a sharper escalation in arousal corresponding with heightened emotions. From language features, we discover that affective expressio
    
[^8]: 快速多极化注意力：一种用于长序列的分治注意力机制

    Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences. (arXiv:2310.11960v1 [cs.CL])

    [http://arxiv.org/abs/2310.11960](http://arxiv.org/abs/2310.11960)

    提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。

    

    基于Transformer的模型已在许多领域取得了最先进的性能。然而，自注意力对于输入长度的二次复杂度限制了Transformer模型在长序列上的适用性。为了解决这个问题，我们提出了快速多极化注意力，一种使用分治策略来减少注意力时间和内存复杂度的新型注意力机制，将长度为n的序列的注意力复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。这种分层方法将查询、键和值分为O(log n)级的分辨率，较远距离的组群越来越大，并学习计算组群数量的权重。因此，以高效分层的方式在较低的分辨率中考虑远离彼此的标记之间的相互作用。快速多极化注意力的总体复杂度为O(n)或O(n log n)。

    Transformer-based models have achieved state-of-the-art performance in many areas. However, the quadratic complexity of self-attention with respect to the input length hinders the applicability of Transformer-based models to long sequences. To address this, we present Fast Multipole Attention, a new attention mechanism that uses a divide-and-conquer strategy to reduce the time and memory complexity of attention for sequences of length $n$ from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$ or $O(n)$, while retaining a global receptive field. The hierarchical approach groups queries, keys, and values into $\mathcal{O}( \log n)$ levels of resolution, where groups at greater distances are increasingly larger in size and the weights to compute group quantities are learned. As such, the interaction between tokens far from each other is considered in lower resolution in an efficient hierarchical manner. The overall complexity of Fast Multipole Attention is $\mathcal{O}(n)$ or $\mathcal{O}(n \
    
[^9]: 自回归的下一个标记预测器是通用学习器。

    Auto-Regressive Next-Token Predictors are Universal Learners. (arXiv:2309.06979v1 [cs.LG])

    [http://arxiv.org/abs/2309.06979](http://arxiv.org/abs/2309.06979)

    自回归的下一个标记预测器可以有效地近似图灵机计算的任何函数，并且在文本生成和算术任务上展现出非平凡的性能。

    

    大型语言模型展现出在逻辑和数学推理方面的非凡能力，使其能够解决复杂任务。有趣的是，这些能力在训练于下一个标记预测的简单任务上的网络中出现。在这项工作中，我们提出了一个用于研究自回归下一个标记预测器的理论框架。我们证明了即使是简单的模型，如线性下一个标记预测器，当其在思维链数据上训练时，可以有效地近似图灵机计算的任何函数。我们引入了一个新的复杂度度量——长度复杂度，它衡量了在近似某个目标函数时，思维链序列中所需的中间标记的数量，并分析了长度复杂度和其他复杂性概念之间的相互关系。最后，我们通过实验证明简单的下一个标记预测器，如线性网络和浅层多层感知机（MLP），在文本生成和算术任务上展示出非平凡的性能。

    Large language models display remarkable capabilities in logical and mathematical reasoning, allowing them to solve complex tasks. Interestingly, these abilities emerge in networks trained on the simple task of next-token prediction. In this work, we present a theoretical framework for studying auto-regressive next-token predictors. We demonstrate that even simple models such as linear next-token predictors, trained on Chain-of-Thought (CoT) data, can approximate any function efficiently computed by a Turing machine. We introduce a new complexity measure -- length complexity -- which measures the number of intermediate tokens in a CoT sequence required to approximate some target function, and analyze the interplay between length complexity and other notions of complexity. Finally, we show experimentally that simple next-token predictors, such as linear networks and shallow Multi-Layer Perceptrons (MLPs), display non-trivial performance on text generation and arithmetic tasks. Our resul
    
[^10]: 一种负责任开发基于生成AI的自动学生反馈框架

    A Framework for Responsible Development of Automated Student Feedback with Generative AI. (arXiv:2308.15334v1 [cs.CY])

    [http://arxiv.org/abs/2308.15334](http://arxiv.org/abs/2308.15334)

    一种基于生成AI的自动学生反馈框架可以提供丰富的反馈，但引入了伦理问题，并需要解决“多数人的暴政”和忽视长尾中少数群体需求的挑战。

    

    提供丰富的反馈对于支持学生学习至关重要。最近生成AI尤其是大规模语言模型的进展，为向学生提供可重复、可扩展和即时生成的自动反馈提供了机会，使得之前稀缺且昂贵的学习资源变得丰富起来。从技术角度而言，这种方法是可行的，得益于最近人工智能和自然语言处理的进步；然而，采用这些技术也引入了一系列潜在的伦理问题，需要认真考虑。人工智能系统的吸引力在于它们可以有效地自动化最乏味的任务；但是这也可能导致“多数人的暴政”，即忽视了长尾中少数群体的需求，因为这些需求很难自动化。因此，开发能够产生有价值和真实的机器学习模型变得至关重要。

    Providing rich feedback to students is essential for supporting student learning. Recent advances in generative AI, particularly within large language modelling (LLM), provide the opportunity to deliver repeatable, scalable and instant automatically generated feedback to students, making abundant a previously scarce and expensive learning resource. Such an approach is feasible from a technical perspective due to these recent advances in Artificial Intelligence (AI) and Natural Language Processing (NLP); while the potential upside is a strong motivator, doing so introduces a range of potential ethical issues that must be considered as we apply these technologies. The attractiveness of AI systems is that they can effectively automate the most mundane tasks; but this risks introducing a "tyranny of the majority", where the needs of minorities in the long tail are overlooked because they are difficult to automate.  Developing machine learning models that can generate valuable and authentic
    
[^11]: 关于大型语言模型的模型压缩综述

    A Survey on Model Compression for Large Language Models. (arXiv:2308.07633v1 [cs.CL])

    [http://arxiv.org/abs/2308.07633](http://arxiv.org/abs/2308.07633)

    本论文提供了关于大型语言模型的模型压缩综述，探讨了量化、修剪、知识蒸馏等不同方法，并突出介绍了最新进展和创新方法，为实现高效的部署提供了重要思路。

    

    大型语言模型（LLMs）以惊人的成功彻底改变了自然语言处理任务。然而，它们庞大的体量和计算需求在资源受限环境下的实际部署中带来了重大挑战。随着这些挑战日益紧迫，模型压缩领域已成为一个关键的研究领域，旨在缓解这些限制。本文提供了一份全面的综述，探讨专门针对LLMs的模型压缩技术。我们深入研究了各种方法，包括量化、修剪、知识蒸馏等，以应对高效部署的迫切需求。在每种技术中，我们重点介绍了最新进展和创新方法，为LLM研究的发展提供了贡献。此外，我们还探讨了用于评估效果的基准策略和评估指标的重要性。

    Large Language Models (LLMs) have revolutionized natural language processing tasks with remarkable success. However, their formidable size and computational demands present significant challenges for practical deployment, especially in resource-constrained environments. As these challenges become increasingly pertinent, the field of model compression has emerged as a pivotal research area to alleviate these limitations. This paper presents a comprehensive survey that navigates the landscape of model compression techniques tailored specifically for LLMs. Addressing the imperative need for efficient deployment, we delve into various methodologies, encompassing quantization, pruning, knowledge distillation, and more. Within each of these techniques, we highlight recent advancements and innovative approaches that contribute to the evolving landscape of LLM research. Furthermore, we explore benchmarking strategies and evaluation metrics that are essential for assessing the effectiveness of 
    

