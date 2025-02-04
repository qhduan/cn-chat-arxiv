# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions](https://arxiv.org/abs/2402.18060) | 在回答医学问题方面，大型语言模型在处理具有挑战性的实际临床案例上的表现是关键，因此构建了两个结构化数据集进行评估。 |
| [^2] | [Large Language Models are Advanced Anonymizers](https://arxiv.org/abs/2402.13846) | 大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。 |
| [^3] | [Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement](https://arxiv.org/abs/2402.11060) | 介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。 |
| [^4] | [Limits of Large Language Models in Debating Humans](https://arxiv.org/abs/2402.06049) | 大型语言模型在与人类辩论中的能力有限，尽管它们能够融入和促进人类的工作效率，但在辩论中的说服力较弱。在成为可行的辩手之前，LLMs需要进一步发展。 |
| [^5] | [Do LLMs Dream of Ontologies?.](http://arxiv.org/abs/2401.14931) | 本文研究了通用预训练大型语言模型（LLMs）是否记忆了已知本体论的信息以及记忆的程度，结果显示LLMs部分地了解本体论的概念，记忆程度与其在Web上的流行程度成正比。 |
| [^6] | [Large language models in bioinformatics: applications and perspectives.](http://arxiv.org/abs/2401.04155) | 本综述介绍了在生物信息学中使用的大型语言模型，如BERT和GPT，并重点探讨了它们在基因组学、转录组学、蛋白质组学、药物发现和单细胞分析等方面的应用。大型语言模型在解决生物信息学问题方面具有巨大潜力和前景。 |
| [^7] | [A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises.](http://arxiv.org/abs/2306.04802) | 本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。 |

# 详细

[^1]: 在回答和解释具有挑战性的医学问题上对大型语言模型的基准测试

    Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions

    [https://arxiv.org/abs/2402.18060](https://arxiv.org/abs/2402.18060)

    在回答医学问题方面，大型语言模型在处理具有挑战性的实际临床案例上的表现是关键，因此构建了两个结构化数据集进行评估。

    

    LLMs在回答医学问题方面表现出色，例如通过医学执照考试。然而，大多数现有的基准测试依赖于委员会考试问题或一般医学问题，无法捕捉真实临床案例的复杂性。此外，缺乏答案的参考解释阻碍了对模型解释的评估，这对支持医生做出复杂的医疗决策至关重要。为解决这些挑战，我们构建了两个新数据集：JAMA临床挑战和Medbullets。JAMA临床挑战包含基于具有挑战性的临床案例的问题，而Medbullets包含类似USMLE Step 2&3风格的临床问题。两个数据集均以多项选择问题-回答任务的结构化形式呈现，每个问题都附有专家撰写的解释。我们使用各种提示在这两个数据集上评估了四个LLMs。实验表明

    arXiv:2402.18060v1 Announce Type: new  Abstract: LLMs have demonstrated impressive performance in answering medical questions, such as passing medical licensing examinations. However, most existing benchmarks rely on board exam questions or general medical questions, falling short in capturing the complexity of realistic clinical cases. Moreover, the lack of reference explanations for answers hampers the evaluation of model explanations, which are crucial to supporting doctors in making complex medical decisions. To address these challenges, we construct two new datasets: JAMA Clinical Challenge and Medbullets. JAMA Clinical Challenge consists of questions based on challenging clinical cases, while Medbullets comprises USMLE Step 2&3 style clinical questions. Both datasets are structured as multiple-choice question-answering tasks, where each question is accompanied by an expert-written explanation. We evaluate four LLMs on the two datasets using various prompts. Experiments demonstrat
    
[^2]: 大型语言模型是先进的匿名化工具

    Large Language Models are Advanced Anonymizers

    [https://arxiv.org/abs/2402.13846](https://arxiv.org/abs/2402.13846)

    大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。

    

    最近在隐私研究领域对大型语言模型的研究表明，它们在推断真实世界在线文本中的个人数据方面表现出接近人类水平的性能。随着模型能力的不断增强，现有的文本匿名化方法当前已经落后于监管要求和对抗威胁。这引出了一个问题：个人如何有效地保护他们在分享在线文本时的个人数据。在这项工作中，我们采取了两步来回答这个问题：首先，我们提出了一个新的设置，用于评估面对对抗性LLM的推断时的匿名化效果，从而允许自然地测量匿名化性能，同时纠正了以前指标的一些缺陷。然后，我们提出了基于LLM的对抗性匿名化框架，利用LLM的强大推断能力来指导我们的匿名化过程。在我们的实验评估中，我们展示了在真实世界中的匿名化实践。

    arXiv:2402.13846v1 Announce Type: cross  Abstract: Recent work in privacy research on large language models has shown that they achieve near human-level performance at inferring personal data from real-world online texts. With consistently increasing model capabilities, existing text anonymization methods are currently lacking behind regulatory requirements and adversarial threats. This raises the question of how individuals can effectively protect their personal data in sharing online texts. In this work, we take two steps to answer this question: We first present a new setting for evaluating anonymizations in the face of adversarial LLMs inferences, allowing for a natural measurement of anonymization performance while remedying some of the shortcomings of previous metrics. We then present our LLM-based adversarial anonymization framework leveraging the strong inferential capabilities of LLMs to inform our anonymization procedure. In our experimental evaluation, we show on real-world 
    
[^3]: Persona-DB：用于响应预测的高效大规模语言模型个性化与协同数据优化

    Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement

    [https://arxiv.org/abs/2402.11060](https://arxiv.org/abs/2402.11060)

    介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。

    

    随着对大型语言模型（LLMs）个性化交互需求的增加，需要开发能够准确快速识别用户意见和偏好的方法。检索增强作为一种有效策略出现，因为它可以适应大量用户而无需进行微调的成本。然而，现有研究主要集中在增强检索阶段，并对数据库表示的优化进行了有限的探索，这是个性化等任务的关键方面。在这项工作中，我们从一个新的角度研究了这个问题，着重于如何更有效地表示数据，以便在LLM定制的情境下更有效地进行检索。为了解决这一挑战，我们介绍了Persona-DB，这是一个简单而有效的框架，包括一个分层构建过程，以改善跨任务背景的泛化能力，并进行协同优化。

    arXiv:2402.11060v1 Announce Type: cross  Abstract: The increasing demand for personalized interactions with large language models (LLMs) calls for the development of methodologies capable of accurately and efficiently identifying user opinions and preferences. Retrieval augmentation emerges as an effective strategy, as it can accommodate a vast number of users without the costs from fine-tuning. Existing research, however, has largely focused on enhancing the retrieval stage and devoted limited exploration toward optimizing the representation of the database, a crucial aspect for tasks such as personalization. In this work, we examine the problem from a novel angle, focusing on how data can be better represented for more efficient retrieval in the context of LLM customization. To tackle this challenge, we introduce Persona-DB, a simple yet effective framework consisting of a hierarchical construction process to improve generalization across task contexts and collaborative refinement to
    
[^4]: 大型语言模型在与人类辩论中的局限性

    Limits of Large Language Models in Debating Humans

    [https://arxiv.org/abs/2402.06049](https://arxiv.org/abs/2402.06049)

    大型语言模型在与人类辩论中的能力有限，尽管它们能够融入和促进人类的工作效率，但在辩论中的说服力较弱。在成为可行的辩手之前，LLMs需要进一步发展。

    

    大型语言模型(LLMs)在与人类的互动中展现出了显著的潜力。随后，将它们作为人工代表和替代品进行社会学实验的潜在应用是一个令人激动的前景。但是这个想法有多可行呢？本文试图通过一项预先注册的研究来测试现阶段LLMs的局限性，该研究将真实的人类与扮演人类的LLM代理结合起来。本研究着重探讨辩论为基础的意见共识形成在三种环境下的情况：仅人类、代理和人类、仅代理。我们的目标是理解LLM代理对人类的影响，并评估它们在辩论方面的能力是否与人类相似。我们发现LLMs能够融入并促进人类的工作效率，但在辩论中的说服力较弱，最终行为与人类有所偏离。我们阐明了这些主要缺陷，并预计在成为可行的辩手之前，LLMs必须进一步发展。

    Large Language Models (LLMs) have shown remarkable promise in their ability to interact proficiently with humans. Subsequently, their potential use as artificial confederates and surrogates in sociological experiments involving conversation is an exciting prospect. But how viable is this idea? This paper endeavors to test the limits of current-day LLMs with a pre-registered study integrating real people with LLM agents acting as people. The study focuses on debate-based opinion consensus formation in three environments: humans only, agents and humans, and agents only. Our goal is to understand how LLM agents influence humans, and how capable they are in debating like humans. We find that LLMs can blend in and facilitate human productivity but are less convincing in debate, with their behavior ultimately deviating from human's. We elucidate these primary failings and anticipate that LLMs must evolve further before being viable debaters.
    
[^5]: LLM是否能记忆本体论？

    Do LLMs Dream of Ontologies?. (arXiv:2401.14931v1 [cs.CL])

    [http://arxiv.org/abs/2401.14931](http://arxiv.org/abs/2401.14931)

    本文研究了通用预训练大型语言模型（LLMs）是否记忆了已知本体论的信息以及记忆的程度，结果显示LLMs部分地了解本体论的概念，记忆程度与其在Web上的流行程度成正比。

    

    大型语言模型（LLMs）最近在自动文本理解和生成方面取得了革命性的进展。这些模型的性能依赖于底层神经网络体系结构的参数数量，这使得LLMs能够记忆训练过程中接触到的大量数据的一部分。本文研究了通用预训练LLMs是否记忆了已知本体论的信息以及记忆的程度。我们的结果表明，LLMs部分地了解本体论：它们可以记忆文本中提到的本体论概念，但其对概念的记忆程度似乎与其在Web上的流行程度成比例变化，因为Web是它们训练材料的主要来源。此外，我们提出了新的度量标准，通过测量不同提示重复、查询语言和确定度的输出一致性来估计LLMs对本体论信息的记忆程度。

    Large language models (LLMs) have recently revolutionized automated text understanding and generation. The performance of these models relies on the high number of parameters of the underlying neural architectures, which allows LLMs to memorize part of the vast quantity of data seen during the training. This paper investigates whether and to what extent general-purpose pre-trained LLMs have memorized information from known ontologies. Our results show that LLMs partially know ontologies: they can, and do indeed, memorize concepts from ontologies mentioned in the text, but the level of memorization of their concepts seems to vary proportionally to their popularity on the Web, the primary source of their training material. We additionally propose new metrics to estimate the degree of memorization of ontological information in LLMs by measuring the consistency of the output produced across different prompt repetitions, query languages, and degrees of determinism.
    
[^6]: 生物信息学中的大型语言模型：应用与展望

    Large language models in bioinformatics: applications and perspectives. (arXiv:2401.04155v1 [q-bio.QM])

    [http://arxiv.org/abs/2401.04155](http://arxiv.org/abs/2401.04155)

    本综述介绍了在生物信息学中使用的大型语言模型，如BERT和GPT，并重点探讨了它们在基因组学、转录组学、蛋白质组学、药物发现和单细胞分析等方面的应用。大型语言模型在解决生物信息学问题方面具有巨大潜力和前景。

    

    大型语言模型（LLMs）是一类基于深度学习的人工智能模型，在各种任务中表现出色，尤其在自然语言处理（NLP）中。大型语言模型通常由具有大量参数的人工神经网络组成，通过自监督或半监督学习，在大量无标签输入上进行训练。然而，它们在解决生物信息学问题方面的潜力甚至超过了在模拟人类语言方面的能力。在这篇综述中，我们将介绍在自然语言处理中使用的几个重要的大型语言模型，如BERT和GPT，并重点探讨大型语言模型在生物信息学中不同组学水平的应用，主要包括基因组学、转录组学、蛋白质组学、药物发现和单细胞分析方面的应用。最后，本综述总结了大型语言模型在解决生物信息学问题方面的潜力和前景。

    Large language models (LLMs) are a class of artificial intelligence models based on deep learning, which have great performance in various tasks, especially in natural language processing (NLP). Large language models typically consist of artificial neural networks with numerous parameters, trained on large amounts of unlabeled input using self-supervised or semi-supervised learning. However, their potential for solving bioinformatics problems may even exceed their proficiency in modeling human language. In this review, we will present a summary of the prominent large language models used in natural language processing, such as BERT and GPT, and focus on exploring the applications of large language models at different omics levels in bioinformatics, mainly including applications of large language models in genomics, transcriptomics, proteomics, drug discovery and single cell analysis. Finally, this review summarizes the potential and prospects of large language models in solving bioinfo
    
[^7]: 医疗知识图谱综述：资源、应用和前景

    A Survey on Knowledge Graphs for Healthcare: Resources, Applications, and Promises. (arXiv:2306.04802v1 [cs.AI])

    [http://arxiv.org/abs/2306.04802](http://arxiv.org/abs/2306.04802)

    本论文综述了医疗知识图谱(HKGs)的构建流程、关键技术和利用方法以及现有资源，并深入探讨了HKG在各种医疗领域的变革性影响。

    

    医疗知识图谱(HKGs)已成为组织医学知识的有结构且可解释的有为工具，提供了医学概念及其关系的全面视图。然而，数据异质性和覆盖范围有限等挑战仍然存在，强调了在HKG领域需要进一步研究的必要性。本综述是HKG的第一份综合概述。我们总结了HKG构建的流程和关键技术（即从头开始和通过集成），以及常见的利用方法（即基于模型和非基于模型）。为了为研究人员提供有价值的资源，我们根据它们捕获的数据类型和应用领域（该资源存储于https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase）组织了现有的HKG，并提供了相关的统计信息。在应用部分，我们深入探讨了HKG在各种医疗领域的变革性影响。

    Healthcare knowledge graphs (HKGs) have emerged as a promising tool for organizing medical knowledge in a structured and interpretable way, which provides a comprehensive view of medical concepts and their relationships. However, challenges such as data heterogeneity and limited coverage remain, emphasizing the need for further research in the field of HKGs. This survey paper serves as the first comprehensive overview of HKGs. We summarize the pipeline and key techniques for HKG construction (i.e., from scratch and through integration), as well as the common utilization approaches (i.e., model-free and model-based). To provide researchers with valuable resources, we organize existing HKGs (The resource is available at https://github.com/lujiaying/Awesome-HealthCare-KnowledgeBase) based on the data types they capture and application domains, supplemented with pertinent statistical information. In the application section, we delve into the transformative impact of HKGs across various hea
    

