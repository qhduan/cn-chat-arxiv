# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLMs in the Loop: Leveraging Large Language Model Annotations for Active Learning in Low-Resource Languages](https://arxiv.org/abs/2404.02261) | 在低资源语言中，通过将LLMs集成到主动学习循环中进行数据注释，有效减少所需数据量，并取得接近最先进性能的结果。 |
| [^2] | [Make Large Language Model a Better Ranker](https://arxiv.org/abs/2403.19181) | 本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。 |
| [^3] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |
| [^4] | [Fundus: A Simple-to-Use News Scraper Optimized for High Quality Extractions](https://arxiv.org/abs/2403.15279) | Fundus是一个简单易用的新闻爬虫工具，通过手工定制的内容提取器，针对每个支持的在线报纸格式指南进行优化，实现高质量的新闻文章提取，同时结合爬取和内容提取于一体，为非技术用户提供统一使用界面。 |
| [^5] | [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319) | 这项调查深入分析了LLMs在融合上下文和参数化知识时所面临的知识冲突，探讨了三类知识冲突对其可信度和性能的重要影响，并提出改进LLMs稳健性策略的策略。 |
| [^6] | [EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models](https://arxiv.org/abs/2402.03049) | EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。 |
| [^7] | [EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models](https://arxiv.org/abs/2308.07269) | EasyEdit提出了一种易于使用的知识编辑框架，针对大型语言模型的知识截断或谬误问题，支持各种最新的知识编辑方法，并可应用于多个知名的LLMs。 |
| [^8] | [A Survey on Neural Topic Models: Methods, Applications, and Challenges.](http://arxiv.org/abs/2401.15351) | 这篇综述调研了神经主题模型的方法、应用和挑战，对于短文本和跨语言文档等各种场景提供了系统性的组织和介绍，并讨论了广泛应用的一系列热门应用。 |
| [^9] | [ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction.](http://arxiv.org/abs/2310.09234) | 这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。 |
| [^10] | [Searching COVID-19 clinical research using graphical abstracts.](http://arxiv.org/abs/2310.04094) | 该论文介绍了一种使用图形摘要来搜索COVID-19临床研究的方法。通过将摘要和图形摘要表示为本体术语图并在网络中匹配，可以更有效地对现有文献进行搜索。 |
| [^11] | [ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation.](http://arxiv.org/abs/2308.11131) | 本论文提出了一种名为ReLLa的检索增强大型语言模型框架，用于零样本和小样本推荐任务。通过语义用户行为检索（SUBR）来提取上下文中的有用信息，以改善LLMs的推荐性能。 |

# 详细

[^1]: 在LLMs中循环：利用大型语言模型注释进行低资源语言的主动学习

    LLMs in the Loop: Leveraging Large Language Model Annotations for Active Learning in Low-Resource Languages

    [https://arxiv.org/abs/2404.02261](https://arxiv.org/abs/2404.02261)

    在低资源语言中，通过将LLMs集成到主动学习循环中进行数据注释，有效减少所需数据量，并取得接近最先进性能的结果。

    

    由于语言资源和数据标注专业知识有限，低资源语言在人工智能开发中面临着重大障碍，使它们变得罕见且成本高昂。为了解决这一不足，我们提出利用LLMs的潜力在主动学习环节中进行数据注释。我们首先进行评估以评估注释者之间的一致性，从而选择适当的LLM注释者。然后，选择的注释者被集成到一个分类器的训练循环中，使用主动学习范式，最小化所需的查询数据量。实证评估，特别是使用GPT-4-Turbo，展示了几乎达到最先进性能的结果，同时大大减少了数据需求，由估算的潜在性能指示。

    arXiv:2404.02261v1 Announce Type: cross  Abstract: Low-resource languages face significant barriers in AI development due to limited linguistic resources and expertise for data labeling, rendering them rare and costly. The scarcity of data and the absence of preexisting tools exacerbate these challenges, especially since these languages may not be adequately represented in various NLP datasets. To address this gap, we propose leveraging the potential of LLMs in the active learning loop for data annotation. Initially, we conduct evaluations to assess inter-annotator agreement and consistency, facilitating the selection of a suitable LLM annotator. The chosen annotator is then integrated into a training loop for a classifier using an active learning paradigm, minimizing the amount of queried data required. Empirical evaluations, notably employing GPT-4-Turbo, demonstrate near-state-of-the-art performance with significantly reduced data requirements, as indicated by estimated potential co
    
[^2]: 让大型语言模型成为更好的排名器

    Make Large Language Model a Better Ranker

    [https://arxiv.org/abs/2403.19181](https://arxiv.org/abs/2403.19181)

    本文介绍了一种具有对齐列表排名目标的语言模型框架（ALRO），旨在弥合大型语言模型的能力与推荐系统排名任务的要求之间的差距。

    

    大型语言模型（LLMs）的发展显著增强了各个领域的能力，导致推荐系统（RSs）概念和开发方式发生了转变。然而，现有研究主要集中在点对点和成对推荐范式上。这些方法在基于LLM的推荐器中效率低下，因为利用大型语言模型的计算成本很高。一些研究虽然深入研究了列表型方法，但在排名任务中表现不佳。这一不足归因于排名和语言生成目标之间的不匹配。为此，本文介绍了具有对齐列表排名目标的语言模型框架（ALRO）。ALRO旨在弥合LLMs的能力与推荐系统排名任务的微妙要求之间的差距。ALRO的一个关键特性是引入了软lambda值lo

    arXiv:2403.19181v1 Announce Type: cross  Abstract: The evolution of Large Language Models (LLMs) has significantly enhanced capabilities across various fields, leading to a paradigm shift in how Recommender Systems (RSs) are conceptualized and developed. However, existing research primarily focuses on point-wise and pair-wise recommendation paradigms. These approaches prove inefficient in LLM-based recommenders due to the high computational cost of utilizing Large Language Models. While some studies have delved into list-wise approaches, they fall short in ranking tasks. This shortfall is attributed to the misalignment between the objectives of ranking and language generation. To this end, this paper introduces the Language Model Framework with Aligned Listwise Ranking Objectives (ALRO). ALRO is designed to bridge the gap between the capabilities of LLMs and the nuanced requirements of ranking tasks within recommender systems. A key feature of ALRO is the introduction of soft lambda lo
    
[^3]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    
[^4]: Fundus：一个简单易用的新闻爬虫，优化高质量提取

    Fundus: A Simple-to-Use News Scraper Optimized for High Quality Extractions

    [https://arxiv.org/abs/2403.15279](https://arxiv.org/abs/2403.15279)

    Fundus是一个简单易用的新闻爬虫工具，通过手工定制的内容提取器，针对每个支持的在线报纸格式指南进行优化，实现高质量的新闻文章提取，同时结合爬取和内容提取于一体，为非技术用户提供统一使用界面。

    

    本文介绍了Fundus，一个用户友好的新闻爬虫，使用户可以仅凭几行代码获得数百万高质量的新闻文章。与现有的新闻爬虫不同，我们使用手工定制的、专门针对每个支持的在线报纸的格式指南的内容提取器。这样我们可以优化我们的爬取质量，以确保检索到的新闻文章完整且没有HTML痕迹。此外，我们的框架将爬取（从网络或大型网络归档中检索HTML）和内容提取结合到一个单一的流水线中。通过为预定义的一组报纸提供统一的界面，我们的目标是使Fundus即使对非技术用户也易于使用。本文概述了框架，讨论了我们的设计选择，并针对其他流行的新闻爬虫进行了比较评估。我们的评估表明，Fundus取得了...

    arXiv:2403.15279v1 Announce Type: new  Abstract: This paper introduces Fundus, a user-friendly news scraper that enables users to obtain millions of high-quality news articles with just a few lines of code. Unlike existing news scrapers, we use manually crafted, bespoke content extractors that are specifically tailored to the formatting guidelines of each supported online newspaper. This allows us to optimize our scraping for quality such that retrieved news articles are textually complete and without HTML artifacts. Further, our framework combines both crawling (retrieving HTML from the web or large web archives) and content extraction into a single pipeline. By providing a unified interface for a predefined collection of newspapers, we aim to make Fundus broadly usable even for non-technical users. This paper gives an overview of the framework, discusses our design choices, and presents a comparative evaluation against other popular news scrapers. Our evaluation shows that Fundus yie
    
[^5]: LLMs的知识冲突：一项调查

    Knowledge Conflicts for LLMs: A Survey

    [https://arxiv.org/abs/2403.08319](https://arxiv.org/abs/2403.08319)

    这项调查深入分析了LLMs在融合上下文和参数化知识时所面临的知识冲突，探讨了三类知识冲突对其可信度和性能的重要影响，并提出改进LLMs稳健性策略的策略。

    

    这项调查对大型语言模型（LLMs）的知识冲突进行了深入分析，突出了当它们融合上下文和参数化知识时所遇到的复杂挑战。我们关注三类知识冲突：上下文-记忆冲突、跨上下文冲突和内部记忆冲突。这些冲突可能会显著影响LLMs的可信度和性能，特别是在现实世界应用中，噪音和错误信息很常见。通过对这些冲突进行分类，探讨其原因，研究LLMs在这些冲突下的行为，并回顾可用的解决方案，本调查旨在为改进LLMs的稳健性策略提供启示，从而成为推动这一不断发展领域研究的宝贵资源。

    arXiv:2403.08319v1 Announce Type: cross  Abstract: This survey provides an in-depth analysis of knowledge conflicts for large language models (LLMs), highlighting the complex challenges they encounter when blending contextual and parametric knowledge. Our focus is on three categories of knowledge conflicts: context-memory, inter-context, and intra-memory conflict. These conflicts can significantly impact the trustworthiness and performance of LLMs, especially in real-world applications where noise and misinformation are common. By categorizing these conflicts, exploring the causes, examining the behaviors of LLMs under such conflicts, and reviewing available solutions, this survey aims to shed light on strategies for improving the robustness of LLMs, thereby serving as a valuable resource for advancing research in this evolving area.
    
[^6]: EasyInstruct：一个易于使用的用于大型语言模型的指令处理框架

    EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models

    [https://arxiv.org/abs/2402.03049](https://arxiv.org/abs/2402.03049)

    EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。

    

    近年来，指令调整已经引起了越来越多的关注，并成为增强大型语言模型（LLMs）能力的一种关键技术。为了构建高质量的指令数据集，已经提出了许多指令处理方法，旨在在数据数量和数据质量之间达到精巧的平衡。然而，由于各种指令处理方法之间仍然存在不一致，目前没有标准的开源指令处理实现框架可供社区使用，这使得从业者无法进一步开发和推进。为了促进指令处理的研究和开发，我们提出了EasyInstruct，一个易于使用的用于LLMs的指令处理框架，它将指令生成、选择和提示模块化，并考虑它们的组合和交互。EasyInstruct已经在https://github.com/zjunlp/EasyInstruct上公开发布，并得到了积极维护。

    In recent years, instruction tuning has gained increasing attention and emerged as a crucial technique to enhance the capabilities of Large Language Models (LLMs). To construct high-quality instruction datasets, many instruction processing approaches have been proposed, aiming to achieve a delicate balance between data quantity and data quality. Nevertheless, due to inconsistencies that persist among various instruction processing methods, there is no standard open-source instruction processing implementation framework available for the community, which hinders practitioners from further developing and advancing. To facilitate instruction processing research and development, we present EasyInstruct, an easy-to-use instruction processing framework for LLMs, which modularizes instruction generation, selection, and prompting, while also considering their combination and interaction. EasyInstruct is publicly released and actively maintained at https://github.com/zjunlp/EasyInstruct, along 
    
[^7]: EasyEdit：一种易于使用的大型语言模型知识编辑框架

    EasyEdit: An Easy-to-use Knowledge Editing Framework for Large Language Models

    [https://arxiv.org/abs/2308.07269](https://arxiv.org/abs/2308.07269)

    EasyEdit提出了一种易于使用的知识编辑框架，针对大型语言模型的知识截断或谬误问题，支持各种最新的知识编辑方法，并可应用于多个知名的LLMs。

    

    大型语言模型（LLMs）通常遭受知识截断或谬误问题，这意味着它们对未见事件不知情或生成具有不正确事实的文本，原因是数据过时/嘈杂。为此，出现了许多针对LLMs的知识编辑方法，旨在微妙地注入/编辑更新的知识或调整不良行为，同时将对不相关输入的影响最小化。然而，由于各种知识编辑方法之间存在显著差异，以及任务设置中的变化，社区中没有可用于知识编辑的标准实施框架，这妨碍了从业者将知识编辑应用于应用程序。为解决这些问题，我们提出了EasyEdit，一种易于使用的LLMs知识编辑框架。它支持各种尖端的知识编辑方法，并可以轻松应用于许多著名的LLMs，如T5、GPT-J、LlaMA等。从经验上来看，我们报告了kno

    arXiv:2308.07269v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) usually suffer from knowledge cutoff or fallacy issues, which means they are unaware of unseen events or generate text with incorrect facts owing to outdated/noisy data. To this end, many knowledge editing approaches for LLMs have emerged -- aiming to subtly inject/edit updated knowledge or adjust undesired behavior while minimizing the impact on unrelated inputs. Nevertheless, due to significant differences among various knowledge editing methods and the variations in task setups, there is no standard implementation framework available for the community, which hinders practitioners from applying knowledge editing to applications. To address these issues, we propose EasyEdit, an easy-to-use knowledge editing framework for LLMs. It supports various cutting-edge knowledge editing approaches and can be readily applied to many well-known LLMs such as T5, GPT-J, LlaMA, etc. Empirically, we report the kno
    
[^8]: 关于神经主题模型的综述：方法、应用和挑战

    A Survey on Neural Topic Models: Methods, Applications, and Challenges. (arXiv:2401.15351v1 [cs.CL])

    [http://arxiv.org/abs/2401.15351](http://arxiv.org/abs/2401.15351)

    这篇综述调研了神经主题模型的方法、应用和挑战，对于短文本和跨语言文档等各种场景提供了系统性的组织和介绍，并讨论了广泛应用的一系列热门应用。

    

    主题模型几十年来一直被广泛应用于无监督方式下发现潜在主题和推断文档的主题比例。它们在文本分析和上下文推荐等各种应用中得到广泛应用。近年来，神经网络的崛起促成了一个新的研究领域——神经主题模型(NTMs)的出现。与传统的主题模型不同，NTMs直接优化参数，而不需要模型特定的推导。这使得NTMs具有更好的可扩展性和灵活性，吸引了大量的研究关注并产生了丰富的新方法和应用。在本文中，我们对神经主题模型的方法、应用和挑战进行了全面的调研。具体而言，根据网络结构系统地组织了当前NTM方法，并介绍了针对短文本和跨语言文档等各种场景的NTMs。我们还讨论了广泛应用的一系列热门应用。

    Topic models have been prevalent for decades to discover latent topics and infer topic proportions of documents in an unsupervised fashion. They have been widely used in various applications like text analysis and context recommendation. Recently, the rise of neural networks has facilitated the emergence of a new research field -- Neural Topic Models (NTMs). Different from conventional topic models, NTMs directly optimize parameters without requiring model-specific derivations. This endows NTMs with better scalability and flexibility, resulting in significant research attention and plentiful new methods and applications. In this paper, we present a comprehensive survey on neural topic models concerning methods, applications, and challenges. Specifically, we systematically organize current NTM methods according to their network structures and introduce the NTMs for various scenarios like short texts and cross-lingual documents. We also discuss a wide range of popular applications built 
    
[^9]: ClickPrompt: CTR模型是将语言模型适应为CTR预测的强大提示生成器

    ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction. (arXiv:2310.09234v1 [cs.IR])

    [http://arxiv.org/abs/2310.09234](http://arxiv.org/abs/2310.09234)

    这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。

    

    点击率（CTR）预测已经成为各种互联网应用程序中越来越不可或缺的。传统的CTR模型通过独热编码将多字段分类数据转换为ID特征，并提取特征之间的协同信号。这种范式的问题在于语义信息的丢失。另一方面的研究通过将输入数据转换为文本句子来探索预训练语言模型（PLM）在CTR预测中的潜力。虽然语义信号得到了保留，但它们通常无法捕捉到协同信息（如特征交互、纯ID特征），更不用说由庞大的模型大小带来的无法接受的推理开销了。在本文中，我们旨在为准确的CTR估计建立语义知识和协同知识，并解决推理效率问题。为了从两个领域中受益并弥合它们之间的差距，我们提出了一种新颖的模型-。

    Click-through rate (CTR) prediction has become increasingly indispensable for various Internet applications. Traditional CTR models convert the multi-field categorical data into ID features via one-hot encoding, and extract the collaborative signals among features. Such a paradigm suffers from the problem of semantic information loss. Another line of research explores the potential of pretrained language models (PLMs) for CTR prediction by converting input data into textual sentences through hard prompt templates. Although semantic signals are preserved, they generally fail to capture the collaborative information (e.g., feature interactions, pure ID features), not to mention the unacceptable inference overhead brought by the huge model size. In this paper, we aim to model both the semantic knowledge and collaborative knowledge for accurate CTR estimation, and meanwhile address the inference inefficiency issue. To benefit from both worlds and close their gaps, we propose a novel model-
    
[^10]: 使用图形摘要搜索COVID-19临床研究

    Searching COVID-19 clinical research using graphical abstracts. (arXiv:2310.04094v1 [cs.IR])

    [http://arxiv.org/abs/2310.04094](http://arxiv.org/abs/2310.04094)

    该论文介绍了一种使用图形摘要来搜索COVID-19临床研究的方法。通过将摘要和图形摘要表示为本体术语图并在网络中匹配，可以更有效地对现有文献进行搜索。

    

    目标：图形摘要是对科学文章主要发现进行视觉总结的概念图。虽然图形摘要通常用于科学出版物中预测和总结主要结果，但我们将其作为表达对现有文献进行图形搜索的手段。材料和方法：我们考虑COVID-19开放研究数据集（CORD-19），这是一个包含超过一百万个摘要的语料库；每个摘要被描述为共现本体术语图，这些术语从统一医学语言系统（UMLS）和冠状病毒传染病本体（CIDO）中选择。图形摘要也被表示为本体术语图，可能还包括描述其相互作用的实用术语（例如，“相关”，“增加”，“引发”）。我们构建了一个包含语料库中提及的概念的共现网络；然后我们在网络上识别出图形摘要的最佳匹配项。我们利用图形数据库...

    Objective. Graphical abstracts are small graphs of concepts that visually summarize the main findings of scientific articles. While graphical abstracts are customarily used in scientific publications to anticipate and summarize their main results, we propose them as a means for expressing graph searches over existing literature. Materials and methods. We consider the COVID-19 Open Research Dataset (CORD-19), a corpus of more than one million abstracts; each of them is described as a graph of co-occurring ontological terms, selected from the Unified Medical Language System (UMLS) and the Ontology of Coronavirus Infectious Disease (CIDO). Graphical abstracts are also expressed as graphs of ontological terms, possibly augmented by utility terms describing their interactions (e.g., "associated with", "increases", "induces"). We build a co-occurrence network of concepts mentioned in the corpus; we then identify the best matches of graphical abstracts on the network. We exploit graph databas
    
[^11]: ReLLa: 基于检索增强的大型语言模型的推荐系统中的生命周期序列行为理解

    ReLLa: Retrieval-enhanced Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation. (arXiv:2308.11131v1 [cs.IR])

    [http://arxiv.org/abs/2308.11131](http://arxiv.org/abs/2308.11131)

    本论文提出了一种名为ReLLa的检索增强大型语言模型框架，用于零样本和小样本推荐任务。通过语义用户行为检索（SUBR）来提取上下文中的有用信息，以改善LLMs的推荐性能。

    

    随着大型语言模型（LLMs）在自然语言处理（NLP）领域取得了显著突破，基于LLM的推荐系统引起了广泛关注并被积极探索。本文专注于适应和增强纯大型语言模型以用于零样本和小样本推荐任务。首先，我们针对推荐领域中LLMs无法从长用户行为序列的文本上下文中提取有用信息的问题，提出并定义了生命周期序列行为理解问题。为了解决这个问题并提高LLMs的推荐性能，我们提出了一种新的框架，即检索增强的大型语言模型（ReLLa）。针对零样本推荐，我们执行语义用户行为检索（SUBR）来提高数据的利用率。

    With large language models (LLMs) achieving remarkable breakthroughs in natural language processing (NLP) domains, LLM-enhanced recommender systems have received much attention and have been actively explored currently. In this paper, we focus on adapting and empowering a pure large language model for zero-shot and few-shot recommendation tasks. First and foremost, we identify and formulate the lifelong sequential behavior incomprehension problem for LLMs in recommendation domains, i.e., LLMs fail to extract useful information from a textual context of long user behavior sequence, even if the length of context is far from reaching the context limitation of LLMs. To address such an issue and improve the recommendation performance of LLMs, we propose a novel framework, namely Retrieval-enhanced Large Language models (ReLLa) for recommendation tasks in both zero-shot and few-shot settings. For zero-shot recommendation, we perform semantic user behavior retrieval (SUBR) to improve the data
    

