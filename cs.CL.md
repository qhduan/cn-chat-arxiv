# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Developing Safe and Responsible Large Language Models -- A Comprehensive Framework](https://arxiv.org/abs/2404.01399) | 该论文介绍了一种新的模型SR$_{\text{LLM}}$，旨在通过引入全面的安全风险分类法和专家标注数据集来增强大型语言模型（LLM）在语言生成中的安全性，并通过指令和参数高效微调方法有效减少了不安全内容的生成。 |
| [^2] | [From Words to Routes: Applying Large Language Models to Vehicle Routing](https://arxiv.org/abs/2403.10795) | 该研究探索了应用大型语言模型解决车辆路径规划问题的能力，提出了基于自然语言任务描述的基本提示范例并提出一种使模型进行改进的框架。 |
| [^3] | [CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences](https://arxiv.org/abs/2403.09032) | 介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。 |
| [^4] | [AmbigNLG: Addressing Task Ambiguity in Instruction for NLG](https://arxiv.org/abs/2402.17717) | AmbigNLG是一个旨在解决自然语言生成任务中指令模糊性挑战的新任务，通过识别和减轻指令中的模糊性，改进了文本生成质量，并突出了清晰和具体指令在提升LLM在NLG任务中表现的关键作用。 |
| [^5] | [Enhancing Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought](https://arxiv.org/abs/2401.06836) | 该研究提出了一种名为情感思维链（ECoT）的提示方法，通过与人类情感智慧准则对齐，增强大型语言模型在情感生成任务上的性能。 |
| [^6] | [Guiding Language Model Math Reasoning with Planning Tokens](https://arxiv.org/abs/2310.05707) | 本论文介绍了一种通过引入规划标记来引导语言模型进行数学推理的方法。这种方法在保持推理过程中的一致性方面具有显著的准确性提升，而增加的训练参数很少。 |
| [^7] | [I am a Strange Dataset: Metalinguistic Tests for Language Models.](http://arxiv.org/abs/2401.05300) | 本研究提出了一个新的数据集，名为“我是一个奇怪的数据集”，用来测试大型语言模型（LLMs）是否能够处理元语言自指的陈述。实验证明，各种开源和闭源LLMs在生成和验证任务中的表现都接近随机猜测。 |
| [^8] | [A ripple in time: a discontinuity in American history.](http://arxiv.org/abs/2312.01185) | 该论文通过使用向量嵌入和非线性降维方法，发现GPT-2与UMAP的结合可以提供更好的分离和聚类效果。同时，经过微调的DistilBERT模型可用于识别总统和演讲的年份。 |
| [^9] | [CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models.](http://arxiv.org/abs/2307.07705) | CPET提出了一种基于压缩LLM的有效参数优化框架，通过引入知识继承和恢复策略，解决了在参数有效调整中压缩LLM的推理计算瓶颈问题。 |

# 详细

[^1]: 开发安全和负责任的大型语言模型 - 一个全面框架

    Developing Safe and Responsible Large Language Models -- A Comprehensive Framework

    [https://arxiv.org/abs/2404.01399](https://arxiv.org/abs/2404.01399)

    该论文介绍了一种新的模型SR$_{\text{LLM}}$，旨在通过引入全面的安全风险分类法和专家标注数据集来增强大型语言模型（LLM）在语言生成中的安全性，并通过指令和参数高效微调方法有效减少了不安全内容的生成。

    

    鉴于人们对大型语言模型（LLM）的安全性和风险日益关注，发展减轻这些问题的方法至关重要。我们引入了安全和负责任的大型语言模型（SR$_{\text{LLM}}$），这个模型旨在通过使用LLM来增强语言生成的安全性。我们的方法结合了一个全面的LLM安全风险分类法，并利用专家注释的数据集与这种分类法相一致。SR$_{\text{LLM}}$旨在识别潜在的不安全内容并产生良性变化。它采用基于指令的和参数高效的微调方法，使得该模型不仅有效地增强安全性，而且资源高效且易于调整。在我们对五个基准数据集和两个专有数据集进行测试后，我们观察到不安全内容生成的显著减少。此外，在实施安全措施后，出现了...

    arXiv:2404.01399v1 Announce Type: new  Abstract: Given the growing concerns around the safety and risks of Large Language Models (LLMs), it is essential to develop methods for mitigating these issues. We introduce Safe and Responsible Large Language Model (SR$_{\text{LLM}}$) , a model designed to enhance the safety of language generation using LLMs. Our approach incorporates a comprehensive LLM safety risk taxonomy and utilizes a dataset annotated by experts that align with this taxonomy. SR$_{\text{LLM}}$ is designed to identify potentially unsafe content and produce benign variations. It employs instruction-based and parameter-efficient fine-tuning methods, making the model not only effective in enhancing safety but also resource-efficient and straightforward to adjust. Through our testing on five benchmark datasets and two proprietary datasets, we observed notable reductions in the generation of unsafe content. Moreover, following the implementation of safety measures, there was a s
    
[^2]: 从单词到路径：将大型语言模型应用于车辆路径规划

    From Words to Routes: Applying Large Language Models to Vehicle Routing

    [https://arxiv.org/abs/2403.10795](https://arxiv.org/abs/2403.10795)

    该研究探索了应用大型语言模型解决车辆路径规划问题的能力，提出了基于自然语言任务描述的基本提示范例并提出一种使模型进行改进的框架。

    

    LLMs在机器人领域（例如操作和导航）中展示出令人印象深刻的进展，其利用自然语言任务描述。LLMs在这些任务中取得成功让我们思考：LLMs在使用自然语言任务描述解决车辆路径规划问题（VRPs）的能力如何？在这项工作中，我们分三步研究这个问题。首先，我们构建了一个包含21种单车或多车路径问题的数据集。其次，我们评估了LLMs在四种基本提示范例文本到代码生成的性能，每种包括不同类型的文本输入。我们发现，直接从自然语言任务描述生成代码的基本提示范例对于GPT-4效果最佳，实现了56%的可行性，40%的优化性和53%的效率。第三，基于观察到LLMs可能无法在初始尝试中提供正确解决方案的现象，我们提出了一个框架，使LLMs能够进行改进。

    arXiv:2403.10795v1 Announce Type: cross  Abstract: LLMs have shown impressive progress in robotics (e.g., manipulation and navigation) with natural language task descriptions. The success of LLMs in these tasks leads us to wonder: What is the ability of LLMs to solve vehicle routing problems (VRPs) with natural language task descriptions? In this work, we study this question in three steps. First, we construct a dataset with 21 types of single- or multi-vehicle routing problems. Second, we evaluate the performance of LLMs across four basic prompt paradigms of text-to-code generation, each involving different types of text input. We find that the basic prompt paradigm, which generates code directly from natural language task descriptions, performs the best for GPT-4, achieving 56% feasibility, 40% optimality, and 53% efficiency. Third, based on the observation that LLMs may not be able to provide correct solutions at the initial attempt, we propose a framework that enables LLMs to refin
    
[^3]: CodeUltraFeedback：一种用于将大型语言模型与编程偏好对齐的LLM作为法官数据集

    CodeUltraFeedback: An LLM-as-a-Judge Dataset for Aligning Large Language Models to Coding Preferences

    [https://arxiv.org/abs/2403.09032](https://arxiv.org/abs/2403.09032)

    介绍了 CodeUltraFeedback 数据集，通过 AI 反馈使 14 种不同的 LLMs 对 10,000 个复杂指令生成响应，并使用 LLM-as-a-Judge 方法评估它们与五种编程偏好的对齐情况，同时提出了用于评估 LLM 对编程偏好对齐的基准 CODAL-Bench。

    

    评估大型语言模型（LLMs）与用户定义的编程偏好的对齐性是一项具有挑战性的工作，需要评估复杂文本LLMs的输出。现有基准仰赖自动化指标和静态分析工具，未能评估用户指令和LLM输出中的微妙之处，突显了对LLM偏好对齐的大规模数据集和基准的需求。在本文中，我们介绍了CodeUltraFeedback，一个包含10,000个复杂指令的偏好数据集，通过AI反馈来调整和对齐LLMs与编程偏好。我们使用14种不同的LLMs对这些指令生成响应，然后根据它们与五种编程偏好的对齐情况进行注释，使用GPT-3.5的LLM作为法官方法产生数字和文本反馈。我们还提出了CODAL-Bench，一个用于评估LLM与这些编程偏好对齐的基准。我们的结果显示C

    arXiv:2403.09032v1 Announce Type: cross  Abstract: Evaluating the alignment of large language models (LLMs) with user-defined coding preferences is a challenging endeavour that requires assessing intricate textual LLMs' outputs. By relying on automated metrics and static analysis tools, existing benchmarks fail to assess nuances in user instructions and LLM outputs, highlighting the need for large-scale datasets and benchmarks for LLM preference alignment. In this paper, we introduce CodeUltraFeedback, a preference dataset of 10,000 complex instructions to tune and align LLMs to coding preferences through AI feedback. We generate responses to the instructions using a pool of 14 diverse LLMs, which we then annotate according to their alignment with five coding preferences using the LLM-as-a-Judge approach with GPT-3.5, producing both numerical and textual feedback. We also present CODAL-Bench, a benchmark for assessing LLM alignment with these coding preferences. Our results show that C
    
[^4]: AmbigNLG: 解决NLG指令中的任务模糊性问题

    AmbigNLG: Addressing Task Ambiguity in Instruction for NLG

    [https://arxiv.org/abs/2402.17717](https://arxiv.org/abs/2402.17717)

    AmbigNLG是一个旨在解决自然语言生成任务中指令模糊性挑战的新任务，通过识别和减轻指令中的模糊性，改进了文本生成质量，并突出了清晰和具体指令在提升LLM在NLG任务中表现的关键作用。

    

    在这项研究中，我们介绍了AmbigNLG，这是一个旨在解决自然语言生成（NLG）任务中指令模糊性挑战的新任务。尽管大语言模型（LLMs）在理解和执行各种任务方面具有令人印象深刻的能力，但它们的性能受到现实指令中的模糊性的显著限制。为了解决这个问题，AmbigNLG试图识别并减轻这种模糊性，旨在精细化指令以更好地匹配用户期望。我们介绍了一个包含2,500个实例的数据集AmbigSNI-NLG，并开发了一个模糊性分类法，用于对指令中的模糊性进行分类和注释。我们的方法在文本生成质量方面取得了显著改进，突出了清晰和具体指令在增强LLM在NLG任务中表现方面的关键作用。

    arXiv:2402.17717v1 Announce Type: new  Abstract: In this study, we introduce AmbigNLG, a new task designed to tackle the challenge of task ambiguity in instructions for Natural Language Generation (NLG) tasks. Despite the impressive capabilities of Large Language Models (LLMs) in understanding and executing a wide range of tasks through natural language interaction, their performance is significantly hindered by the ambiguity present in real-world instructions. To address this, AmbigNLG seeks to identify and mitigate such ambiguities, aiming to refine instructions to match user expectations better. We introduce a dataset, AmbigSNI-NLG, consisting of 2,500 instances, and develop an ambiguity taxonomy for categorizing and annotating instruction ambiguities. Our approach demonstrates substantial improvements in text generation quality, highlighting the critical role of clear and specific instructions in enhancing LLM performance in NLG tasks.
    
[^5]: 通过情绪思维链增强大型语言模型的情绪生成能力

    Enhancing Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought

    [https://arxiv.org/abs/2401.06836](https://arxiv.org/abs/2401.06836)

    该研究提出了一种名为情感思维链（ECoT）的提示方法，通过与人类情感智慧准则对齐，增强大型语言模型在情感生成任务上的性能。

    

    大型语言模型（LLMs）在各种情绪识别任务中表现出色，引起了研究界对探索它们在情感智能中潜力的好奇心。然而，情感生成任务领域仍存在一些问题，包括人类偏好的对齐和情感生成评估。本文提出了一种名为情感思维链（ECoT）的即插即用提示方法，通过与人类情感智力指南对齐来增强LLMs在各种情感生成任务上的表现。为了评估ECoT的可靠性，我们提出了一种称为情感生成得分（EGS）的自动化基于模型的评估方法。EGS将戈尔曼的情绪智力理论作为人类专家共识，为情感生成任务的评估提供了新的视角。大量实验结果证明...

    arXiv:2401.06836v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown remarkable performance in various emotion recognition tasks, thereby piquing the research community's curiosity for exploring their potential in emotional intelligence. However, several issues in the field of emotional generation tasks remain unresolved, including human preference alignment and emotional generation assessment. In this paper, we propose the Emotional Chain-of-Thought (ECoT), a plug-and-play prompting method that enhances the performance of LLMs on various emotional generation tasks by aligning with human emotional intelligence guidelines. To assess the reliability of ECoT, we propose an automated model-based evaluation method called Emotional Generation Score (EGS). EGS incorporates Goleman's Emotional Intelligence Theory as a consensus of human experts, providing a new perspective on the evaluation of emotional generation tasks. Extensive experimental results demonstrate 
    
[^6]: 用规划标记引导语言模型的数学推理

    Guiding Language Model Math Reasoning with Planning Tokens

    [https://arxiv.org/abs/2310.05707](https://arxiv.org/abs/2310.05707)

    本论文介绍了一种通过引入规划标记来引导语言模型进行数学推理的方法。这种方法在保持推理过程中的一致性方面具有显著的准确性提升，而增加的训练参数很少。

    

    大型语言模型（LLMs）近来因其进行复杂推理任务的能力（如思维链推理）而引起了广泛关注。然而，大多数现有的增强模型推理能力方法过于依赖数据驱动方法，忽视了模型推理能力的结构化方面。我们发现，虽然LLMs可以很好地处理个别推理步骤，但在整个推理链上保持一致性方面却存在困难。为了解决这个问题，我们在每个推理步骤的开始处引入规划标记，作为模型的引导，并将它们的嵌入添加到模型参数中。我们的方法对于可训练参数的增加非常小（仅为0.001%），可以通过完全微调或更高效的参数方案来应用。我们通过将其应用于三种不同的LLMs，在三个数学单词问题数据集上展示了我们方法的有效性，相对于标准方法，准确性显著提高。

    Large language models (LLMs) have recently attracted considerable interest for their ability to perform complex reasoning tasks, such as chain-of-thought reasoning. However, most of the existing approaches to enhance this ability rely heavily on data-driven methods, while neglecting the structural aspects of the model's reasoning capacity. We find that while LLMs can manage individual reasoning steps well, they struggle with maintaining consistency across an entire reasoning chain. To solve this, we introduce planning tokens at the start of each reasoning step, serving as a guide for the model, and add their embeddings to the model parameters. Our approach requires a negligible increase in trainable parameters (just 0.001%) and can be applied through either full fine-tuning or a more parameter-efficient scheme. We demonstrate our method's effectiveness by applying it to three different LLMs, showing notable accuracy improvements across three math word problem datasets w.r.t. standard f
    
[^7]: 我是一个奇怪的数据集：用于语言模型的元语言测试

    I am a Strange Dataset: Metalinguistic Tests for Language Models. (arXiv:2401.05300v1 [cs.CL])

    [http://arxiv.org/abs/2401.05300](http://arxiv.org/abs/2401.05300)

    本研究提出了一个新的数据集，名为“我是一个奇怪的数据集”，用来测试大型语言模型（LLMs）是否能够处理元语言自指的陈述。实验证明，各种开源和闭源LLMs在生成和验证任务中的表现都接近随机猜测。

    

    在许多领域中，涉及元语言自指的陈述（“本论文有六个部分。”）是普遍存在的。大型语言模型（LLMs）能否处理这样的语言？在本文中，我们提出了一个新的数据集“我是一个奇怪的数据集”，用来解决这个问题。它包含两个子任务：生成和验证。在生成任务中，模型会继续类似于“这个句子中倒数第二个词是”的陈述（正确的继续应该是“是”）。在验证任务中，模型会判断类似于“这个句子中倒数第二个词是句子。”的陈述的真实性（是假的）。我们还提供了最小差异的非自指元语言示例，来补充主数据集，以测试模型是否能够处理元语言语言。数据集由专家手工制作，非专家标注员进行验证。我们测试了各种开源LLMs（从7B到70B的参数）以及通过API进行测试的闭源LLMs。所有模型在两个子任务上的表现都接近随机猜测。

    Statements involving metalinguistic self-reference ("This paper has six sections.") are prevalent in many domains. Can large language models (LLMs) handle such language? In this paper, we present "I am a Strange Dataset", a new dataset for addressing this question. There are two subtasks: generation and verification. In generation, models continue statements like "The penultimate word in this sentence is" (where a correct continuation is "is"). In verification, models judge the truth of statements like "The penultimate word in this sentence is sentence." (false). We also provide minimally different metalinguistic non-self-reference examples to complement the main dataset by probing for whether models can handle metalinguistic language at all. The dataset is hand-crafted by experts and validated by non-expert annotators. We test a variety of open-source LLMs (7B to 70B parameters) as well as closed-source LLMs through APIs. All models perform close to chance across both subtasks and eve
    
[^8]: 时间中的涟漪：美国历史中的不连续性

    A ripple in time: a discontinuity in American history. (arXiv:2312.01185v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.01185](http://arxiv.org/abs/2312.01185)

    该论文通过使用向量嵌入和非线性降维方法，发现GPT-2与UMAP的结合可以提供更好的分离和聚类效果。同时，经过微调的DistilBERT模型可用于识别总统和演讲的年份。

    

    在这篇论文中，我们使用来自Kaggle的国情咨文数据集对美国历史的总体时间线及咨文本身的特点和性质进行了一些令人惊讶（也有些不那么令人惊讶）的观察。我们的主要方法是使用向量嵌入，如BERT（DistilBERT）和GPT-2。虽然广泛认为BERT（及其变体）最适合NLP分类任务，但我们发现GPT-2结合UMAP等非线性降维方法可以提供更好的分离和更强的聚类效果。这使得GPT-2 + UMAP成为一个有趣的替代方案。在我们的情况下，不需要对模型进行微调，预训练的GPT-2模型就足够好用。我们还使用了经过微调的DistilBERT模型来检测哪位总统发表了哪篇演讲，并取得了非常好的结果（准确率为93\% - 95\%，具体取决于运行情况）。为了确定写作年份，我们还执行了一个类似的任务。

    In this note we use the State of the Union Address (SOTU) dataset from Kaggle to make some surprising (and some not so surprising) observations pertaining to the general timeline of American history, and the character and nature of the addresses themselves. Our main approach is using vector embeddings, such as BERT (DistilBERT) and GPT-2.  While it is widely believed that BERT (and its variations) is most suitable for NLP classification tasks, we find out that GPT-2 in conjunction with nonlinear dimension reduction methods such as UMAP provide better separation and stronger clustering. This makes GPT-2 + UMAP an interesting alternative. In our case, no model fine-tuning is required, and the pre-trained out-of-the-box GPT-2 model is enough.  We also used a fine-tuned DistilBERT model for classification detecting which President delivered which address, with very good results (accuracy 93\% - 95\% depending on the run). An analogous task was performed to determine the year of writing, an
    
[^9]: CPET: 高效压缩大型语言模型的参数优化

    CPET: Effective Parameter-Efficient Tuning for Compressed Large Language Models. (arXiv:2307.07705v1 [cs.CL])

    [http://arxiv.org/abs/2307.07705](http://arxiv.org/abs/2307.07705)

    CPET提出了一种基于压缩LLM的有效参数优化框架，通过引入知识继承和恢复策略，解决了在参数有效调整中压缩LLM的推理计算瓶颈问题。

    

    近年来，参数有效调整（PET）因为在调整相对较少的参数（PET模块）的同时仍能激活大型语言模型（LLM）的足够知识以用于下游任务而得到广泛研究。此外，当PET用于为多个任务提供服务时，可以在冻结的LLM上构建不同的任务特定PET模块，避免冗余LLM部署。虽然PET显著降低了调优和部署LLM的成本，但其推理仍然受到LLM计算瓶颈的影响。为了解决上述问题，我们提出了一种基于压缩LLM的有效PET框架，称为“CPET”。在CPET中，我们评估了主流LLM压缩技术对PET性能的影响，然后引入了知识继承和恢复策略来恢复由这些压缩技术引起的知识丢失。我们的实验结果表明，由于CPET的恢复策略，合作

    Parameter-efficient tuning (PET) has been widely explored in recent years because it tunes much fewer parameters (PET modules) than full-parameter fine-tuning (FT) while still stimulating sufficient knowledge from large language models (LLMs) for downstream tasks. Moreover, when PET is employed to serve multiple tasks, different task-specific PET modules can be built on a frozen LLM, avoiding redundant LLM deployments. Although PET significantly reduces the cost of tuning and deploying LLMs, its inference still suffers from the computational bottleneck of LLMs. To address the above issue, we propose an effective PET framework based on compressed LLMs, named "CPET". In CPET, we evaluate the impact of mainstream LLM compression techniques on PET performance and then introduce knowledge inheritance and recovery strategies to restore the knowledge loss caused by these compression techniques. Our experimental results demonstrate that, owing to the restoring strategies of CPET, collaborating
    

