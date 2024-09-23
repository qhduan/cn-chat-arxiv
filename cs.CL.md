# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266) | Latxa是一种用于巴斯克语的大型语言模型系列，在语言熟练度和理解能力方面表现出色，优于所有以前的开放模型，并具有多个评估数据集，填补了巴斯克语高质量基准的不足。 |
| [^2] | [Imagination Augmented Generation: Learning to Imagine Richer Context for Question Answering over Large Language Models](https://arxiv.org/abs/2403.15268) | 提出了一种新颖的知识增强框架，即想象增强生成（IAG），通过想象力，而非依赖外部资源，来补充大型语言模型中可能存在的知识缺陷，并提出了一种想象更丰富背景的方法（IMcQA）来解决问题回答中的挑战。 |
| [^3] | [Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic Language Models](https://arxiv.org/abs/2403.11130) | 本文研究了令牌化策略和词汇量对阿拉伯语言模型性能的影响，结果显示字节对编码（BPE）与Farasa在多个任务中表现优异，突显了形态分析在捕捉阿拉伯语言细微差异中的重要性。 |
| [^4] | [Entity-Aware Multimodal Alignment Framework for News Image Captioning](https://arxiv.org/abs/2402.19404) | 设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。 |
| [^5] | [MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages](https://arxiv.org/abs/2402.10691) | MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。 |
| [^6] | [Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models](https://arxiv.org/abs/2402.03877) | 本文调查了大型语言模型（LLMs）在几何推理方面的能力，并发现了它们在目标变量选择和2D空间关系方面存在偏见和困难。通过引入基于LLMs的多代理体系结构，本研究提出了一种通过自我纠正、协作和不同角色专业化来提高LLMs几何推理能力的框架。 |
| [^7] | [Faithful Knowledge Graph Explanations for Commonsense Reasoning](https://arxiv.org/abs/2310.04910) | 本论文提出了两个量化指标来衡量基于知识图谱的解释的可信性，并引入了一种新的训练方法来改善解释的可信度。实验结果表明该方法可以提高解释的一致性和保真度。 |
| [^8] | [ChemDFM: Dialogue Foundation Model for Chemistry.](http://arxiv.org/abs/2401.14818) | ChemDFM是首个面向化学智能的大型语言模型，它通过对化学文献和数据的训练，具备了存储、理解和推理化学知识和语言的能力，并且在化学领域的性能上优于其他开源模型。 |
| [^9] | [Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism.](http://arxiv.org/abs/2311.01041) | 本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。 |
| [^10] | [Knowledge Editing for Large Language Models: A Survey.](http://arxiv.org/abs/2310.16218) | 大型语言模型(LLMs)在学术和工业领域具有巨大潜力。本文综述了LLMs的知识编辑问题，强调了需要开发有效和高效的技术来更新预训练LLMs以纳入新知识的重要性。 |
| [^11] | [Product Attribute Value Extraction using Large Language Models.](http://arxiv.org/abs/2310.12537) | 本文研究使用大型语言模型作为预训练的替代方法，解决了传统属性/值提取技术中需要大量训练数据和对未知属性值的挑战问题。 |
| [^12] | [ChatGPT an ENFJ, Bard an ISTJ: Empirical Study on Personalities of Large Language Models.](http://arxiv.org/abs/2305.19926) | 本研究通过采用特质理论框架，实验证明了ChatGPT始终表现出ENFJ型人格，无论指令或情境如何。研究揭示了LLMs的个性化，有助于促进人与机器之间更好的沟通和协作。 |
| [^13] | [ContraSim -- A Similarity Measure Based on Contrastive Learning.](http://arxiv.org/abs/2303.16992) | 本文提出了一种新的相似度度量方法: ContraSim，该方法利用对比学习学习参数化的度量方法。实验表明，ContraSim在多种基准测试中均获得了比之前相似度量方法更高的准确性。 |

# 详细

[^1]: Latxa: 一种用于巴斯克语的开放语言模型和评估套件

    Latxa: An Open Language Model and Evaluation Suite for Basque

    [https://arxiv.org/abs/2403.20266](https://arxiv.org/abs/2403.20266)

    Latxa是一种用于巴斯克语的大型语言模型系列，在语言熟练度和理解能力方面表现出色，优于所有以前的开放模型，并具有多个评估数据集，填补了巴斯克语高质量基准的不足。

    

    我们介绍了Latxa，这是一个基于Llama 2的大型巴斯克语言模型系列，参数范围从7到700亿。Latxa基于新的巴斯克语语料库预训练，包括430万个文档和42亿个标记。针对巴斯克语高质量基准的稀缺性，我们进一步提出了4个多项选择评估数据集：EusProficiency，包括来自官方语言能力考试的5169个问题；EusReading，包括352个阅读理解问题；EusTrivia，包括来自5个知识领域的1715个琐事问题；以及EusExams，包括来自公共考试的16774个问题。在我们的广泛评估中，Latxa在与我们比较的所有先前开放模型中表现出色。此外，尽管在阅读理解和知识密集型任务方面落后，但在语言熟练度和理解能力方面，它与GPT-4 Turbo具有竞争力。Latxa模型系列，以及

    arXiv:2403.20266v1 Announce Type: cross  Abstract: We introduce Latxa, a family of large language models for Basque ranging from 7 to 70 billion parameters. Latxa is based on Llama 2, which we continue pretraining on a new Basque corpus comprising 4.3M documents and 4.2B tokens. Addressing the scarcity of high-quality benchmarks for Basque, we further introduce 4 multiple choice evaluation datasets: EusProficiency, comprising 5,169 questions from official language proficiency exams; EusReading, comprising 352 reading comprehension questions; EusTrivia, comprising 1,715 trivia questions from 5 knowledge areas; and EusExams, comprising 16,774 questions from public examinations. In our extensive evaluation, Latxa outperforms all previous open models we compare to by a large margin. In addition, it is competitive with GPT-4 Turbo in language proficiency and understanding, despite lagging behind in reading comprehension and knowledge-intensive tasks. Both the Latxa family of models, as well
    
[^2]: 想象增强生成：学习想象更丰富的背景来进行大型语言模型问题回答

    Imagination Augmented Generation: Learning to Imagine Richer Context for Question Answering over Large Language Models

    [https://arxiv.org/abs/2403.15268](https://arxiv.org/abs/2403.15268)

    提出了一种新颖的知识增强框架，即想象增强生成（IAG），通过想象力，而非依赖外部资源，来补充大型语言模型中可能存在的知识缺陷，并提出了一种想象更丰富背景的方法（IMcQA）来解决问题回答中的挑战。

    

    检索增强生成和生成增强生成已被提出来增强大型语言模型（LLMs）上的问题回答所需的知识。然而，前者依赖于外部资源，而且两者都需要将显式文档合并到上下文中，导致更长的上下文，从而消耗更多资源。最近的研究表明，LLMs已经建模了丰富的知识，尽管没有被有效地触发或激活。在此启发下，我们提出了一种新颖的知识增强框架，即想象增强生成（IAG），它模拟了人类通过想象力在仅凭想象回答问题时弥补知识缺陷的能力，而不依赖外部资源。在IAG的指导下，我们提出了一种用于问题回答的想象更丰富背景的方法（IMcQA），通过以下两个模块获得更丰富的背景：通过生成简单的想象实现显式想象

    arXiv:2403.15268v1 Announce Type: new  Abstract: Retrieval-Augmented-Generation and Gener-ation-Augmented-Generation have been proposed to enhance the knowledge required for question answering over Large Language Models (LLMs). However, the former depends on external resources, and both require incorporating the explicit documents into the context, which results in longer contexts that lead to more resource consumption. Recent works indicate that LLMs have modeled rich knowledge, albeit not effectively triggered or activated. Inspired by this, we propose a novel knowledge-augmented framework, Imagination-Augmented-Generation (IAG), which simulates the human capacity to compensate for knowledge deficits while answering questions solely through imagination, without relying on external resources. Guided by IAG, we propose an imagine richer context method for question answering (IMcQA), which obtains richer context through the following two modules: explicit imagination by generating a sho
    
[^3]: 探索令牌化策略和词汇量对增强阿拉伯语言模型的影响

    Exploring Tokenization Strategies and Vocabulary Sizes for Enhanced Arabic Language Models

    [https://arxiv.org/abs/2403.11130](https://arxiv.org/abs/2403.11130)

    本文研究了令牌化策略和词汇量对阿拉伯语言模型性能的影响，结果显示字节对编码（BPE）与Farasa在多个任务中表现优异，突显了形态分析在捕捉阿拉伯语言细微差异中的重要性。

    

    本文全面研究了令牌化策略和词汇量对阿拉伯语言模型在下游自然语言处理任务中性能的影响。我们的研究重点关注了四种令牌化器在各种任务中的有效性，包括新闻分类、仇恨言论检测、情感分析和自然语言推理。利用多样化的词汇量，我们仔细研究了令牌化方法与模型性能之间的复杂相互作用。结果表明，使用Farasa的字节对编码（BPE）在多个任务中优于其他策略，强调了在捕捉阿拉伯语言细微差异方面形态分析的重要性。然而，在情感分析中存在挑战，方言特定的分割问题影响了模型的效率。计算效率分析表明，BPE与Farasa的稳定性较高。

    arXiv:2403.11130v1 Announce Type: new  Abstract: This paper presents a comprehensive examination of the impact of tokenization strategies and vocabulary sizes on the performance of Arabic language models in downstream natural language processing tasks. Our investigation focused on the effectiveness of four tokenizers across various tasks, including News Classification, Hate Speech Detection, Sentiment Analysis, and Natural Language Inference. Leveraging a diverse set of vocabulary sizes, we scrutinize the intricate interplay between tokenization approaches and model performance. The results reveal that Byte Pair Encoding (BPE) with Farasa outperforms other strategies in multiple tasks, underscoring the significance of morphological analysis in capturing the nuances of the Arabic language. However, challenges arise in sentiment analysis, where dialect specific segmentation issues impact model efficiency. Computational efficiency analysis demonstrates the stability of BPE with Farasa, su
    
[^4]: 面向实体的多模态对齐框架用于新闻图像字幕生成

    Entity-Aware Multimodal Alignment Framework for News Image Captioning

    [https://arxiv.org/abs/2402.19404](https://arxiv.org/abs/2402.19404)

    设计了面向实体的多模态对齐任务和对齐框架，提高了新闻图像字幕生成任务的性能表现。

    

    新闻图像字幕生成任务是图像字幕生成任务的一个变体，要求模型生成一个更具信息性的字幕，其中包含新闻图像和相关新闻文章。近年来，多模态大型语言模型发展迅速，并在新闻图像字幕生成任务中表现出前景。然而，根据我们的实验，常见的多模态大型语言模型在零样本设定下生成实体方面表现不佳。即使在新闻图像字幕生成数据集上进行简单微调，它们处理实体信息的能力仍然有限。为了获得一个更强大的模型来处理多模态实体信息，我们设计了两个多模态实体感知对齐任务和一个对齐框架，以对齐模型并生成新闻图像字幕。我们的方法在GoodNews数据集上将CIDEr分数提高到86.29（从72.33），在NYTimes800k数据集上将其提高到85.61（从70.83），优于先前的最先进模型。

    arXiv:2402.19404v1 Announce Type: cross  Abstract: News image captioning task is a variant of image captioning task which requires model to generate a more informative caption with news image and the associated news article. Multimodal Large Language models have developed rapidly in recent years and is promising in news image captioning task. However, according to our experiments, common MLLMs are not good at generating the entities in zero-shot setting. Their abilities to deal with the entities information are still limited after simply fine-tuned on news image captioning dataset. To obtain a more powerful model to handle the multimodal entity information, we design two multimodal entity-aware alignment tasks and an alignment framework to align the model and generate the news image captions. Our method achieves better results than previous state-of-the-art models in CIDEr score (72.33 -> 86.29) on GoodNews dataset and (70.83 -> 85.61) on NYTimes800k dataset.
    
[^5]: MultiPoT: 多语言思维程序利用多种编程语言

    MultiPoT: Multilingual Program of Thoughts Harnesses Multiple Programming Languages

    [https://arxiv.org/abs/2402.10691](https://arxiv.org/abs/2402.10691)

    MultiPoT 提出了一种任务和模型无关的方法，通过利用多种编程语言的优势和多样性，在表现上显著优于 Python 自一致性。

    

    arXiv:2402.10691v1 公告类型：新的 摘要：思维程序（PoT）是一种以其可执行中间步骤为特征的方法，其确保推理过程中数值计算的准确性。目前，PoT主要使用Python。然而，仅依赖单一语言可能导致次优解决方案，忽视其他编程语言的潜在优势。在本文中，我们对PoT中使用的编程语言进行了全面实验，发现没有一种单一语言在所有任务和模型上始终提供最佳性能。每种语言的有效性取决于具体情景。受此启发，我们提出了一种称为MultiPoT的任务和模型无关方法，该方法从各种语言中获取强大和多样性。实验结果显示，MultiPoT 在很大程度上优于Python 自一致性。此外，与最佳模型相比，它实现了可比或更优异的性能。

    arXiv:2402.10691v1 Announce Type: new  Abstract: Program of Thoughts (PoT) is an approach characterized by its executable intermediate steps, which ensure the accuracy of the numerical calculations in the reasoning process. Currently, PoT primarily uses Python. However, relying solely on a single language may result in suboptimal solutions and overlook the potential benefits of other programming languages. In this paper, we conduct comprehensive experiments on the programming languages used in PoT and find that no single language consistently delivers optimal performance across all tasks and models. The effectiveness of each language varies depending on the specific scenarios. Inspired by this, we propose a task and model agnostic approach called MultiPoT, which harnesses strength and diversity from various languages. Experimental results reveal that it significantly outperforms Python Self-Consistency. Furthermore, it achieves comparable or superior performance compared to the best mo
    
[^6]: 超越线条和圆圈：揭示大型语言模型中的几何推理差距

    Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models

    [https://arxiv.org/abs/2402.03877](https://arxiv.org/abs/2402.03877)

    本文调查了大型语言模型（LLMs）在几何推理方面的能力，并发现了它们在目标变量选择和2D空间关系方面存在偏见和困难。通过引入基于LLMs的多代理体系结构，本研究提出了一种通过自我纠正、协作和不同角色专业化来提高LLMs几何推理能力的框架。

    

    大型语言模型（LLMs）在数学和算法任务方面展示了不断增长的能力，然而它们在几何推理方面的技能还未被充分探索。我们调查了LLMs在构造性几何问题解决中的能力，这是人类数学推理发展中最基础的步骤之一。我们的研究揭示了目前最先进的LLMs在这个领域面临的显著挑战，尽管在类似领域取得了许多成功。LLMs在目标变量选择方面存在偏见，并且在2D空间关系方面面临困难，经常会错误地表示和臆造对象及其放置位置。为此，我们引入了一个基于LLMs的多代理体系结构，通过进行内部对话来增强它们现有的推理潜力。这项工作强调了LLMs在几何推理中的现有限制，并通过自我纠正、协作和不同角色专业化来提高几何推理能力。

    Large Language Models (LLMs) demonstrate ever-increasing abilities in mathematical and algorithmic tasks, yet their geometric reasoning skills are underexplored. We investigate LLMs' abilities in constructive geometric problem-solving one of the most fundamental steps in the development of human mathematical reasoning. Our work reveals notable challenges that the state-of-the-art LLMs face in this domain despite many successes in similar areas. LLMs exhibit biases in target variable selection and struggle with 2D spatial relationships, often misrepresenting and hallucinating objects and their placements. To this end, we introduce a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue. This work underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
    
[^7]: 关于常识推理的知识图谱解释的可信性

    Faithful Knowledge Graph Explanations for Commonsense Reasoning

    [https://arxiv.org/abs/2310.04910](https://arxiv.org/abs/2310.04910)

    本论文提出了两个量化指标来衡量基于知识图谱的解释的可信性，并引入了一种新的训练方法来改善解释的可信度。实验结果表明该方法可以提高解释的一致性和保真度。

    

    融合语言模型(LMs)和知识图谱(KGs)已成为常识问答研究中的常见方法，但在这些模型中实现精确的思路链解释仍然是一个未解决的问题。当前基于知识图谱的解释技术的一个主要弱点是在评估过程中忽视了生成解释的可信性。为了弥补这一差距，我们提出并验证了两个量化指标 - 图一致性和图保真度 - 来衡量基于知识图谱的解释的可信性。我们引入一种新的训练方法Consistent GNN (CGNN)，该方法添加了一项一致性正则化项来改善解释的可信度。我们的分析表明，KG的预测经常偏离原始模型的预测。所提出的CGNN方法提高了一致性和保真度，展示了它产生更可信解释的潜力。我们的工作强调了明确评估解释可信性的重要性。

    While fusing language models (LMs) and knowledge graphs (KGs) has become common in commonsense question answering research, enabling faithful chain-of-thought explanations in these models remains an open problem. One major weakness of current KG-based explanation techniques is that they overlook the faithfulness of generated explanations during evaluation. To address this gap, we make two main contributions: (1) We propose and validate two quantitative metrics - graph consistency and graph fidelity - to measure the faithfulness of KG-based explanations. (2) We introduce Consistent GNN (CGNN), a novel training method that adds a consistency regularization term to improve explanation faithfulness. Our analysis shows that predictions from KG often diverge from original model predictions. The proposed CGNN approach boosts consistency and fidelity, demonstrating its potential for producing more faithful explanations. Our work emphasises the importance of explicitly evaluating suggest a path
    
[^8]: ChemDFM: 化学领域对话基础模型

    ChemDFM: Dialogue Foundation Model for Chemistry. (arXiv:2401.14818v1 [cs.CL])

    [http://arxiv.org/abs/2401.14818](http://arxiv.org/abs/2401.14818)

    ChemDFM是首个面向化学智能的大型语言模型，它通过对化学文献和数据的训练，具备了存储、理解和推理化学知识和语言的能力，并且在化学领域的性能上优于其他开源模型。

    

    大型语言模型(LLMs)在自然语言处理的一般领域取得了巨大成功。它们的任务概括和自由对话能力可以极大地帮助设计化学智能(CGI)，以协助化学领域的实际研究。然而，在化学领域中存在专业语言和知识，如高度信息化的SMILES符号表示法，阻碍了一般领域LLMs在化学领域的性能。为此，我们开发了ChemDFM，这是首个面向CGI的LLM。ChemDFM-13B是在化学文献、教科书、说明书以及各种一般领域的数据中训练的34B令牌。因此，它可以存储、理解和推理化学知识和语言，同时具有先进的自由形式语言理解能力。广泛的定量评估表明，ChemDFM可以明显优于代表性的开源LLMs。此外，ChemDFM还可以...

    Large language models (LLMs) have established great success in the general domain of natural language processing. Their emerging task generalization and free-form dialogue capabilities can greatly help to design Chemical General Intelligence (CGI) to assist real-world research in chemistry. However, the existence of specialized language and knowledge in the field of chemistry, such as the highly informative SMILES notation, hinders the performance of general-domain LLMs in chemistry. To this end, we develop ChemDFM, the first LLM towards CGI. ChemDFM-13B is trained on 34B tokens from chemical literature, textbooks, and instructions as well as various data from the general domain. Therefore, it can store, understand, and reason over chemical knowledge and languages while still possessing advanced free-form language comprehension capabilities. Extensive quantitative evaluation shows that ChemDFM can significantly outperform the representative open-sourced LLMs. Moreover, ChemDFM can also
    
[^9]: 学会拒绝：通过知识范围限制和拒绝机制使大型语言模型更可控和可靠

    Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism. (arXiv:2311.01041v1 [cs.CL])

    [http://arxiv.org/abs/2311.01041](http://arxiv.org/abs/2311.01041)

    本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。

    

    大型语言模型（LLMs）展示了令人印象深刻的语言理解和生成能力，使它们能够回答各个领域的广泛问题。然而，这些模型并不完美，经常产生含有错误或错误信息的回答。这些不准确性，通常称为幻觉，使得LLMs在许多场景中不可靠甚至不可用。本文的重点是在LLMs中缓解幻觉问题，特别是在问答环境中。我们探索了一种拒绝机制，指导LLMs拒绝回答具有挑战性的问题以避免错误。我们提出了一个简单而有效的解决方案Learn to Refuse (L2R)，它将拒绝机制纳入到LLMs中，使其能够识别和拒绝那些它们难以回答的问题。为了实现这一点，我们利用结构化知识库来表示所有LLMs所需要的知识。

    Large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, enabling them to answer a wide range of questions across various domains. However, these models are not flawless and often produce responses that contain errors or misinformation. These inaccuracies, commonly referred to as hallucinations, render LLMs unreliable and even unusable in many scenarios. In this paper, our focus is on mitigating the issue of hallucination in LLMs, particularly in the context of question-answering. Instead of attempting to answer all questions, we explore a refusal mechanism that instructs LLMs to refuse to answer challenging questions in order to avoid errors. We then propose a simple yet effective solution called Learn to Refuse (L2R), which incorporates the refusal mechanism to enable LLMs to recognize and refuse to answer questions that they find difficult to address. To achieve this, we utilize a structured knowledge base to represent all the LLM
    
[^10]: 大型语言模型的知识编辑：一项综述

    Knowledge Editing for Large Language Models: A Survey. (arXiv:2310.16218v1 [cs.CL])

    [http://arxiv.org/abs/2310.16218](http://arxiv.org/abs/2310.16218)

    大型语言模型(LLMs)在学术和工业领域具有巨大潜力。本文综述了LLMs的知识编辑问题，强调了需要开发有效和高效的技术来更新预训练LLMs以纳入新知识的重要性。

    

    大型语言模型(LLMs)近期以其出色的理解、分析和生成文本的能力，根据其广博的知识和推理能力，改变了学术和工业领域的格局。然而，LLMs的一个主要缺点是它们在预训练时需要大量计算资源，因为其参数数量前所未有。当需要频繁引入新知识到预训练模型中时，这个缺点更加显著。因此，开发有效和高效的技术来更新预训练LLMs是必不可少的。传统方法是通过直接微调将新知识编码到预训练LLMs中。然而，简单地重新训练LLMs可能计算资源密集，并且存在将与模型更新无关的有价值的预训练知识退化的风险。最近，基于知识的模型编辑(KME)引起了越来越多的关注，旨在精确修改LLMs以纳入特定的知识。

    Large language models (LLMs) have recently transformed both the academic and industrial landscapes due to their remarkable capacity to understand, analyze, and generate texts based on their vast knowledge and reasoning ability. Nevertheless, one major drawback of LLMs is their substantial computational cost for pre-training due to their unprecedented amounts of parameters. The disadvantage is exacerbated when new knowledge frequently needs to be introduced into the pre-trained model. Therefore, it is imperative to develop effective and efficient techniques to update pre-trained LLMs. Traditional methods encode new knowledge in pre-trained LLMs through direct fine-tuning. However, naively re-training LLMs can be computationally intensive and risks degenerating valuable pre-trained knowledge irrelevant to the update in the model. Recently, Knowledge-based Model Editing (KME) has attracted increasing attention, which aims to precisely modify the LLMs to incorporate specific knowledge, wit
    
[^11]: 使用大型语言模型进行产品属性值提取

    Product Attribute Value Extraction using Large Language Models. (arXiv:2310.12537v1 [cs.CL])

    [http://arxiv.org/abs/2310.12537](http://arxiv.org/abs/2310.12537)

    本文研究使用大型语言模型作为预训练的替代方法，解决了传统属性/值提取技术中需要大量训练数据和对未知属性值的挑战问题。

    

    电子商务应用（如面向属性的产品搜索或产品比较）基于结构化的产品描述，如属性/值对。电子商务平台上的供应商不提供结构化的产品描述，而是使用标题或描述来描述产品。为了处理这样的产品，有必要从文本产品属性中提取属性/值对。现有技术中，属性/值提取方法依赖于预训练的语言模型（如BERT）。这些模型在属性/值提取方面存在两个主要缺点：（一）模型需要大量的与任务相关的训练数据；（二）优化后的模型在推广到训练数据中未包含的属性值方面面临挑战。本文探讨了大型语言模型（LLMs）作为训练数据效率高且鲁棒性强的替代方法在属性/值提取中的潜力。我们考虑了托管的LLMs，如GPT-3.5和GPT-4。

    E-commerce applications such as faceted product search or product comparison are based on structured product descriptions like attribute/value pairs. The vendors on e-commerce platforms do not provide structured product descriptions but describe offers using titles or descriptions. To process such offers, it is necessary to extract attribute/value pairs from textual product attributes. State-of-the-art attribute/value extraction techniques rely on pre-trained language models (PLMs), such as BERT. Two major drawbacks of these models for attribute/value extraction are that (i) the models require significant amounts of task-specific training data and (ii) the fine-tuned models face challenges in generalizing to attribute values not included in the training data. This paper explores the potential of large language models (LLMs) as a training data-efficient and robust alternative to PLM-based attribute/value extraction methods. We consider hosted LLMs, such as GPT-3.5 and GPT-4, as well as 
    
[^12]: ChatGPT是ENFJ，Bard是ISTJ：大型语言模型的个性实证研究。

    ChatGPT an ENFJ, Bard an ISTJ: Empirical Study on Personalities of Large Language Models. (arXiv:2305.19926v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.19926](http://arxiv.org/abs/2305.19926)

    本研究通过采用特质理论框架，实验证明了ChatGPT始终表现出ENFJ型人格，无论指令或情境如何。研究揭示了LLMs的个性化，有助于促进人与机器之间更好的沟通和协作。

    

    大型语言模型（LLMs）在人工智能领域取得了显著进展，大大重塑了人机交互。我们不仅关注LLMs的性能，还从心理学角度探索它们的特点，认识到了理解它们行为特征的重要性。本研究采用心理学的一个框架——特质理论研究LLMs所展示的行为模式。我们首先关注评估ChatGPT所展示的人格类型的一致性。此外，实验涉及七种附加语言的跨语言影响，以及六种其他LLMs的研究。此外，该研究还调查了ChatGPT是否能够展示对指令或情境线索的人格变化。研究结果表明，无论指令或情境如何，ChatGPT始终保持其ENFJ人格。通过揭示LLMs的个性化，我们预计我们的解决方案可以促进人与机器之间更好的沟通和协作。

    Large Language Models (LLMs) have made remarkable advancements in the field of artificial intelligence, significantly reshaping the human-computer interaction. We not only focus on the performance of LLMs, but also explore their features from a psychological perspective, acknowledging the importance of understanding their behavioral characteristics. Our study examines the behavioral patterns displayed by LLMs by employing trait theory, a psychological framework. We first focus on evaluating the consistency of personality types exhibited by ChatGPT. Furthermore, experiments include cross-lingual effects on seven additional languages, and the investigation of six other LLMs. Moreover, the study investigates whether ChatGPT can exhibit personality changes in response to instructions or contextual cues. The findings show that ChatGPT consistently maintains its ENFJ personality regardless of instructions or contexts. By shedding light on the personalization of LLMs, we anticipate that our s
    
[^13]: ContraSim -- 基于对比学习的相似度度量方法

    ContraSim -- A Similarity Measure Based on Contrastive Learning. (arXiv:2303.16992v1 [cs.CL])

    [http://arxiv.org/abs/2303.16992](http://arxiv.org/abs/2303.16992)

    本文提出了一种新的相似度度量方法: ContraSim，该方法利用对比学习学习参数化的度量方法。实验表明，ContraSim在多种基准测试中均获得了比之前相似度量方法更高的准确性。

    

    最近有研究通过基于相似性的分析比较神经网络表示，揭示了不同方面（如架构、训练数据等）如何影响模型的内部表示。相似度量的质量通常通过其在预期匹配的表示中分配高分数的成功来评估。然而，现有的相似度量在标准基准测试中表现平庸。本文提出一种新的相似度度量方法，称为ContraSim，基于对比学习，与常见的闭式相似性度量不同，ContraSim使用相似和不相似的示例来学习参数化的度量方法。我们在标准的图层预测基准测试和我们介绍的两个新基准测试中使用语言和视觉模型进行广泛的实验评估：多语言基准测试和图像字幕基准测试。在所有情况下，ContraSim的准确性都比之前的相似度量方法高得多。

    Recent work has compared neural network representations via similarity-based analyses, shedding light on how different aspects (architecture, training data, etc.) affect models' internal representations. The quality of a similarity measure is typically evaluated by its success in assigning a high score to representations that are expected to be matched. However, existing similarity measures perform mediocrely on standard benchmarks. In this work, we develop a new similarity measure, dubbed ContraSim, based on contrastive learning. In contrast to common closed-form similarity measures, ContraSim learns a parameterized measure by using both similar and dissimilar examples. We perform an extensive experimental evaluation of our method, with both language and vision models, on the standard layer prediction benchmark and two new benchmarks that we introduce: the multilingual benchmark and the image-caption benchmark. In all cases, ContraSim achieves much higher accuracy than previous simila
    

