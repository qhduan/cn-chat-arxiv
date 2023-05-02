# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Blueprint of IR Evaluation Integrating Task and User Characteristics: Test Collection and Evaluation Metrics.](http://arxiv.org/abs/2305.00747) | 本论文探讨了如何在IR评估中处理多维度和多级相关性评估，文档重叠以及如何将文档的可用性属性与相关性评估相结合，并提出了一个正式的模型。 |
| [^2] | [Contextual Response Interpretation for Automated Structured Interviews: A Case Study in Market Research.](http://arxiv.org/abs/2305.00577) | 该文探讨了使用自动对话系统进行市场研究结构化面试的挑战，并通过将多项选择题转化为对话格式并进行用户研究来解决这些挑战。 |
| [^3] | [The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples.](http://arxiv.org/abs/2305.00574) | 这篇论文讨论了解释能力的反面，研究使用反事实例子污染推荐系统。实验显示，他们的策略能够成功干扰推荐系统的性能。 |
| [^4] | [Making Changes in Webpages Discoverable: A Change-Text Search Interface for Web Archives.](http://arxiv.org/abs/2305.00546) | 本文提出了一种改变文本搜索引擎，允许用户找到网页中的更改。使用该引擎可以清楚地显示网页中添加或删除的术语和短语的时间。 |
| [^5] | [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.](http://arxiv.org/abs/2305.00447) | TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。 |
| [^6] | [S2abEL: A Dataset for Entity Linking from Scientific Tables.](http://arxiv.org/abs/2305.00366) | 该论文提供了第一个专注于科学表格的 EL 数据集 S2abEL，用于实体链接任务。由于科学知识库的不完整性和语境影响，科学表格上的 EL 具有挑战性，该数据集专注于机器学习结果表中的 EL，包含手工标记的单元格类型、属性和实体链接，并引入了一种优于其他方法的神经基线方法。 |
| [^7] | [Synthetic Cross-language Information Retrieval Training Data.](http://arxiv.org/abs/2305.00331) | 该论文介绍了JH-POLO CLIR训练集创建方法，该方法通过生成大型语言模型来解决跨语言信息检索系统所面临的训练数据匮乏、固定资源大小和文体话语领域固定等问题。 |
| [^8] | [Evaluation of GPT-3.5 and GPT-4 for supporting real-world information needs in healthcare delivery.](http://arxiv.org/abs/2304.13714) | 本研究评估了在临床环境中使用GPT-3.5和GPT-4解决医学问题的安全性以及与信息技术咨询服务报告的一致性。研究结果表明，两个LLMs都可以以安全和一致的方式满足医生的信息需求。 |
| [^9] | [Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited.](http://arxiv.org/abs/2303.13835) | 推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。 |
| [^10] | [GSim: A Graph Neural Network based Relevance Measure for Heterogeneous Graphs.](http://arxiv.org/abs/2208.06144) | GSim是一种基于图神经网络的异构图关联度量方法，不需要预定义的元路径，能够捕捉异构图的隐含结构，已在多个数据集上得到验证。 |
| [^11] | [Bootstrap Latent Representations for Multi-modal Recommendation.](http://arxiv.org/abs/2207.05969) | 本文提出了一种名为自助潜在表示（BLR）的新型自监督方法，它可以在不涉及辅助图形的情况下增强用户/物品表示并有效地对待正负样本，从而提高了多模态推荐的准确性和效率。 |
| [^12] | [SelfCF: A Simple Framework for Self-supervised Collaborative Filtering.](http://arxiv.org/abs/2107.03019) | SelfCF是一种自监督协同过滤框架，用于推荐场景，通过增强现有的深度学习协同过滤模型中输出的嵌入来简化算法以及避免昂贵的计算和潜在的负样本问题。 |

# 详细

[^1]: 一种融合任务与用户特征的IR评估蓝图：测试集与评估指标

    A Blueprint of IR Evaluation Integrating Task and User Characteristics: Test Collection and Evaluation Metrics. (arXiv:2305.00747v1 [cs.IR])

    [http://arxiv.org/abs/2305.00747](http://arxiv.org/abs/2305.00747)

    本论文探讨了如何在IR评估中处理多维度和多级相关性评估，文档重叠以及如何将文档的可用性属性与相关性评估相结合，并提出了一个正式的模型。

    

    相关性通常被理解为信息需求与信息对象之间的多级和多维关系。然而，传统的IR评估指标过于简单化，假定相关性是单一维度的。本文提出了几个问题：如何处理IR评估中的多维度和多级相关性评估？如何处理文档的重叠和评估？如何将文档的可用性属性与多维度相关性评估相结合？最终，我们探讨如何定义一个正式的模型，以处理多维度分级相关性评估、文档重叠和文档可用性属性。

    Relevance is generally understood as a multi-level and multi-dimensional relationship between an information need and an information object. However, traditional IR evaluation metrics naively assume mono-dimensionality. We ask: How to deal with multidimensional and graded relevance assessments in IR evaluation? Moreover, search result evaluation metrics neglect document overlaps and naively assume gains piling up as the searcher examines the ranked list into greater length. Consequently, we examine: How to deal with document overlap in IR evaluation? The usability of a document for a person-in-need also depends on document usability attributes beyond relevance. Therefore, we ask: How to deal with usability attributes, and how to combine this with multidimensional relevance assessments in IR evaluation? Finally, we ask how to define a formal model, which deals with multidimensional graded relevance assessments, document overlaps, and document usability attributes in a coherent framework
    
[^2]: 自动化结构化面试的语境响应解释：以市场研究为例

    Contextual Response Interpretation for Automated Structured Interviews: A Case Study in Market Research. (arXiv:2305.00577v1 [cs.IR])

    [http://arxiv.org/abs/2305.00577](http://arxiv.org/abs/2305.00577)

    该文探讨了使用自动对话系统进行市场研究结构化面试的挑战，并通过将多项选择题转化为对话格式并进行用户研究来解决这些挑战。

    

    结构化面试在许多场景下被使用，尤其是在品牌感知、客户习惯或偏好等市场研究中。这些面试一般由一系列问题组成，并由熟练的面试官进行解释。使用自动对话系统来进行这些面试可以触达更广泛和多样化的受访者群体，但相应技术挑战尚未被充分探索。该文将市场研究多项选择题转化为对话格式，并进行了用户研究，以更好地理解这些挑战。

    Structured interviews are used in many settings, importantly in market research on topics such as brand perception, customer habits, or preferences, which are critical to product development, marketing, and e-commerce at large. Such interviews generally consist of a series of questions that are asked to a participant. These interviews are typically conducted by skilled interviewers, who interpret the responses from the participants and can adapt the interview accordingly. Using automated conversational agents to conduct such interviews would enable reaching a much larger and potentially more diverse group of participants than currently possible. However, the technical challenges involved in building such a conversational system are relatively unexplored. To learn more about these challenges, we convert a market research multiple-choice questionnaire to a conversational format and conduct a user study. We address the key task of conducting structured interviews, namely interpreting the 
    
[^3]: 解释的黑暗面：用反事实例子污染推荐系统

    The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples. (arXiv:2305.00574v1 [cs.IR])

    [http://arxiv.org/abs/2305.00574](http://arxiv.org/abs/2305.00574)

    这篇论文讨论了解释能力的反面，研究使用反事实例子污染推荐系统。实验显示，他们的策略能够成功干扰推荐系统的性能。

    

    基于深度学习的推荐系统已经成为几个在线平台的重要组成部分。然而，它们的黑匣子本质强调了对可解释人工智能（XAI）方法的需求，以提供人类可以理解的原因，说明为什么向特定用户推荐特定的项目。其中一种方法是反事实说明（CF）。虽然CF对用户和系统设计人员可能会非常有益，但恶意行为者也可能利用这些说明来破坏系统的安全性。在这项工作中，我们提出了一种新的策略H-CARS，通过CF污染推荐系统。具体而言，我们首先在从反事实说明派生的训练数据上训练了一种基于逻辑推理的代理模型。通过颠倒推荐模型的学习过程，我们因此开发出了一个有效的贪婪算法，为上述代理模型生成虚假用户资料及其相关交互记录。我们的实验使用了一个众所周知的CF模型测试了我们的腐败技术，结果显示能够成功干扰推荐系统的性能。

    Deep learning-based recommender systems have become an integral part of several online platforms. However, their black-box nature emphasizes the need for explainable artificial intelligence (XAI) approaches to provide human-understandable reasons why a specific item gets recommended to a given user. One such method is counterfactual explanation (CF). While CFs can be highly beneficial for users and system designers, malicious actors may also exploit these explanations to undermine the system's security. In this work, we propose H-CARS, a novel strategy to poison recommender systems via CFs. Specifically, we first train a logical-reasoning-based surrogate model on training data derived from counterfactual explanations. By reversing the learning process of the recommendation model, we thus develop a proficient greedy algorithm to generate fabricated user profiles and their associated interaction records for the aforementioned surrogate model. Our experiments, which employ a well-known CF
    
[^4]: 使网页变化可发现性:面向Web档案馆的文本变化搜索界面

    Making Changes in Webpages Discoverable: A Change-Text Search Interface for Web Archives. (arXiv:2305.00546v1 [cs.IR])

    [http://arxiv.org/abs/2305.00546](http://arxiv.org/abs/2305.00546)

    本文提出了一种改变文本搜索引擎，允许用户找到网页中的更改。使用该引擎可以清楚地显示网页中添加或删除的术语和短语的时间。

    

    网页随时间改变，而Web档案馆保存着网页的历史版本副本。Web档案馆的用户，如记者，想要找到并查看网页随时间的变化。然而，目前的Web档案馆搜索界面不支持这项任务。我们提出了一种改变文本搜索引擎，允许用户找到网页中的更改。我们描述了搜索引擎后端和前端的实现，包括一个工具，允许用户在上下文中查看两个网页版本之间的更改，作为动画呈现。我们使用2016年至2020年之间更改过的美国联邦环境网页评估了搜索引擎。变化文本搜索结果页面可以清楚地显示从网页中添加或删除的术语和短语的时间。

    Webpages change over time, and web archives hold copies of historical versions of webpages. Users of web archives, such as journalists, want to find and view changes on webpages over time. However, the current search interfaces for web archives do not support this task. For the web archives that include a full-text search feature, multiple versions of the same webpage that match the search query are shown individually without enumerating changes, or are grouped together in a way that hides changes. We present a change text search engine that allows users to find changes in webpages. We describe the implementation of the search engine backend and frontend, including a tool that allows users to view the changes between two webpage versions in context as an animation. We evaluate the search engine with U.S. federal environmental webpages that changed between 2016 and 2020. The change text search results page can clearly show when terms and phrases were added or removed from webpages. The 
    
[^5]: TALLRec: 一种与推荐系统对齐的大型语言模型有效且高效的调整框架

    TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. (arXiv:2305.00447v1 [cs.IR])

    [http://arxiv.org/abs/2305.00447](http://arxiv.org/abs/2305.00447)

    TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。

    

    大型语言模型（LLMs）已经展现了在不同领域的显著性能，因此研究人员开始探索它们在推荐系统中的潜力。虽然初始的尝试已经利用了LLMs的优异能力，比如通过上下文学习中的提示词来丰富知识并进行强化泛化，但是由于LLMs的训练任务与推荐任务之间的巨大差异以及预训练期间的不足的推荐数据，LLMs在推荐任务中的性能仍然不理想。为了填补这一差距，我们考虑使用推荐数据对LLMs进行调整来构建大型推荐语言模型。为此，我们提出了一种名为TALLRec的高效且有效的调整框架，用于将LLMs与推荐系统对齐。我们已经证明了所提出的TALLRec框架可以显著增强LLMs在推荐任务中的能力。

    Large Language Models (LLMs) have demonstrated remarkable performance across diverse domains, thereby prompting researchers to explore their potential for use in recommendation systems. Initial attempts have leveraged the exceptional capabilities of LLMs, such as rich knowledge and strong generalization through In-context Learning, which involves phrasing the recommendation task as prompts. Nevertheless, the performance of LLMs in recommendation tasks remains suboptimal due to a substantial disparity between the training tasks for LLMs and recommendation tasks, as well as inadequate recommendation data during pre-training. To bridge the gap, we consider building a Large Recommendation Language Model by tunning LLMs with recommendation data. To this end, we propose an efficient and effective Tuning framework for Aligning LLMs with Recommendation, namely TALLRec. We have demonstrated that the proposed TALLRec framework can significantly enhance the recommendation capabilities of LLMs in 
    
[^6]: S2abEL：一份用于科学表格实体链接的数据集

    S2abEL: A Dataset for Entity Linking from Scientific Tables. (arXiv:2305.00366v1 [cs.CL])

    [http://arxiv.org/abs/2305.00366](http://arxiv.org/abs/2305.00366)

    该论文提供了第一个专注于科学表格的 EL 数据集 S2abEL，用于实体链接任务。由于科学知识库的不完整性和语境影响，科学表格上的 EL 具有挑战性，该数据集专注于机器学习结果表中的 EL，包含手工标记的单元格类型、属性和实体链接，并引入了一种优于其他方法的神经基线方法。

    

    实体链接（EL）是将文本提及链接到知识库中相应条目的任务，这对于许多知识密集型的自然语言处理应用来说是至关重要的。当应用于科学论文中的表格时，EL是实现大规模科学知识库的一步，这可以实现先进的科学问答和分析。我们提供了第一个针对科学表格中的EL的数据集。科学表格的EL尤其具有挑战性，因为科学知识库可能非常不完整，并且通常需要理解论文中的文本以及表格的上下文来消除歧义。我们的数据集S2abEL专注于机器学习结果表中的EL，并包括来自PaperswithCode分类法的8,429个单元格的手工标记的单元格类型、来源属性和实体链接。我们引入了一种针对科学表格的神经基线方法，该方法包含许多知识库之外提及的实体，并显示它明显优于其他方法。

    Entity linking (EL) is the task of linking a textual mention to its corresponding entry in a knowledge base, and is critical for many knowledge-intensive NLP applications. When applied to tables in scientific papers, EL is a step toward large-scale scientific knowledge bases that could enable advanced scientific question answering and analytics. We present the first dataset for EL in scientific tables. EL for scientific tables is especially challenging because scientific knowledge bases can be very incomplete, and disambiguating table mentions typically requires understanding the papers's tet in addition to the table. Our dataset, S2abEL, focuses on EL in machine learning results tables and includes hand-labeled cell types, attributed sources, and entity links from the PaperswithCode taxonomy for 8,429 cells from 732 tables. We introduce a neural baseline method designed for EL on scientific tables containing many out-of-knowledge-base mentions, and show that it significantly outperfor
    
[^7]: 合成跨语言信息检索训练数据

    Synthetic Cross-language Information Retrieval Training Data. (arXiv:2305.00331v1 [cs.IR])

    [http://arxiv.org/abs/2305.00331](http://arxiv.org/abs/2305.00331)

    该论文介绍了JH-POLO CLIR训练集创建方法，该方法通过生成大型语言模型来解决跨语言信息检索系统所面临的训练数据匮乏、固定资源大小和文体话语领域固定等问题。

    

    神经跨语言信息检索(CLIR)系统的一个关键难点在于缺乏训练数据。MS MARCO单语训练集的出现显著提高了神经单语检索技术的发展水平。通过使用机器翻译将MS MARCO文档翻译成其他语言，这一资源已经被用于跨语言信息检索领域。然而，这种翻译存在多个问题，它是一个固定大小的资源，它的文体和话语领域是固定的，翻译文档不是用母语而是用翻译语言编写的。为了解决这些问题，我们介绍了JH-POLO CLIR训练集创建方法。该方法首先选择一对非英语段落，然后使用生成型大型语言模型来生成一个英语查询，使得第一个段落相关而第二个段落不相关。

    A key stumbling block for neural cross-language information retrieval (CLIR) systems has been the paucity of training data. The appearance of the MS MARCO monolingual training set led to significant advances in the state of the art in neural monolingual retrieval. By translating the MS MARCO documents into other languages using machine translation, this resource has been made useful to the CLIR community. Yet such translation suffers from a number of problems. While MS MARCO is a large resource, it is of fixed size; its genre and domain of discourse are fixed; and the translated documents are not written in the language of a native speaker of the language, but rather in translationese. To address these problems, we introduce the JH-POLO CLIR training set creation methodology. The approach begins by selecting a pair of non-English passages. A generative large language model is then used to produce an English query for which the first passage is relevant and the second passage is not rel
    
[^8]: 评估GPT-3.5和GPT-4在支持医疗保健信息需求方面的实际作用

    Evaluation of GPT-3.5 and GPT-4 for supporting real-world information needs in healthcare delivery. (arXiv:2304.13714v1 [cs.AI])

    [http://arxiv.org/abs/2304.13714](http://arxiv.org/abs/2304.13714)

    本研究评估了在临床环境中使用GPT-3.5和GPT-4解决医学问题的安全性以及与信息技术咨询服务报告的一致性。研究结果表明，两个LLMs都可以以安全和一致的方式满足医生的信息需求。

    

    尽管在医疗保健领域使用大型语言模型(LLMs)越来越受关注，但当前的探索并未评估LLMs在临床环境中的实用性和安全性。我们的目标是确定两个LLM是否可以以安全和一致的方式满足由医生提交的信息需求问题。我们将66个来自信息技术咨询服务的问题通过简单的提示提交给GPT-3.5和GPT-4。12名医生评估了LLM响应对患者造成伤害的可能性以及与信息技术咨询服务的现有报告的一致性。医生的评估基于多数票汇总。对于没有任何问题，大多数医生认为任何一个LLM响应都不会造成伤害。对于GPT-3.5，8个问题的响应与信息技术咨询报告一致，20个不一致，9个无法评估。有29个响应没有多数票表示“同意”、“不同意”和“无法评估”。

    Despite growing interest in using large language models (LLMs) in healthcare, current explorations do not assess the real-world utility and safety of LLMs in clinical settings. Our objective was to determine whether two LLMs can serve information needs submitted by physicians as questions to an informatics consultation service in a safe and concordant manner. Sixty six questions from an informatics consult service were submitted to GPT-3.5 and GPT-4 via simple prompts. 12 physicians assessed the LLM responses' possibility of patient harm and concordance with existing reports from an informatics consultation service. Physician assessments were summarized based on majority vote. For no questions did a majority of physicians deem either LLM response as harmful. For GPT-3.5, responses to 8 questions were concordant with the informatics consult report, 20 discordant, and 9 were unable to be assessed. There were 29 responses with no majority on "Agree", "Disagree", and "Unable to assess". Fo
    
[^9]: 推荐系统何去何从？ID- vs. 基于模态的推荐模型再探讨

    Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited. (arXiv:2303.13835v1 [cs.IR])

    [http://arxiv.org/abs/2303.13835](http://arxiv.org/abs/2303.13835)

    推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。

    

    过去十年，利用唯一标识（ID）来表示不同用户和物品的推荐模型一直是最先进的，并且在推荐系统文献中占主导地位。与此同时，预训练模态编码器（如BERT和ViT）在对物品的原始模态特征（如文本和图像）进行建模方面变得越来越强大。因此，自然而然的问题是：通过用最先进的模态编码器替换物品ID嵌入向量，一个纯粹的基于模态的推荐模型（MoRec）能否胜过或与纯ID基础模型（IDRec）相匹配？实际上，早在十年前，这个问题就被回答了，IDRec在推荐准确性和效率方面都远远胜过MoRec。我们旨在重新审视这个“老问题”，从多个方面对MoRec进行系统研究。具体而言，我们研究了几个子问题：（i）在实际场景中，MoRec或IDRec哪个推荐模式表现更好，特别是在一般情况和......

    Recommendation models that utilize unique identities (IDs) to represent distinct users and items have been state-of-the-art (SOTA) and dominated the recommender systems (RS) literature for over a decade. Meanwhile, the pre-trained modality encoders, such as BERT and ViT, have become increasingly powerful in modeling the raw modality features of an item, such as text and images. Given this, a natural question arises: can a purely modality-based recommendation model (MoRec) outperforms or matches a pure ID-based model (IDRec) by replacing the itemID embedding with a SOTA modality encoder? In fact, this question was answered ten years ago when IDRec beats MoRec by a strong margin in both recommendation accuracy and efficiency. We aim to revisit this `old' question and systematically study MoRec from several aspects. Specifically, we study several sub-questions: (i) which recommendation paradigm, MoRec or IDRec, performs better in practical scenarios, especially in the general setting and 
    
[^10]: GSim ：面向异构图的图神经网络关联度量

    GSim: A Graph Neural Network based Relevance Measure for Heterogeneous Graphs. (arXiv:2208.06144v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.06144](http://arxiv.org/abs/2208.06144)

    GSim是一种基于图神经网络的异构图关联度量方法，不需要预定义的元路径，能够捕捉异构图的隐含结构，已在多个数据集上得到验证。

    

    异构图在各个领域中广泛存在，包含具有多种类型的节点和边缘。关联度量是分析异构图的基本任务之一，目的是计算不同类型的两个对象之间的相关性，已经在许多应用程序中得到了广泛的应用，例如网络搜索、推荐和社区检测。大多数现有的关联度量方法专注于同质网络，但为异构图开发了一些方法，但它们经常需要预定义的元路径。定义有意义的元路径需要大量领域知识，这在基于架构丰富的异构图（如知识图谱）上极大地限制了它们的应用。最近，图神经网络已广泛应用于许多图挖掘任务中，但尚未用于衡量关联性。为解决上述问题，我们提出了GSim，一种基于图神经网络的异构图关联度量方法。GSim能够捕捉异构图的隐含结构，不需要预定义的元路径。我们在三个真实世界的数据集上评估了我们提出的方法，结果表明GSim显著优于几种最先进的方法。

    Heterogeneous graphs, which contain nodes and edges of multiple types, are prevalent in various domains, including bibliographic networks, social media, and knowledge graphs. As a fundamental task in analyzing heterogeneous graphs, relevance measure aims to calculate the relevance between two objects of different types, which has been used in many applications such as web search, recommendation, and community detection. Most of existing relevance measures focus on homogeneous networks where objects are of the same type, and a few measures are developed for heterogeneous graphs, but they often need the pre-defined meta-path. Defining meaningful meta-paths requires much domain knowledge, which largely limits their applications, especially on schema-rich heterogeneous graphs like knowledge graphs. Recently, the Graph Neural Network (GNN) has been widely applied in many graph mining tasks, but it has not been applied for measuring relevance yet. To address the aforementioned problems, we p
    
[^11]: 多模态推荐中的 Bootstrap 潜在表示法

    Bootstrap Latent Representations for Multi-modal Recommendation. (arXiv:2207.05969v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2207.05969](http://arxiv.org/abs/2207.05969)

    本文提出了一种名为自助潜在表示（BLR）的新型自监督方法，它可以在不涉及辅助图形的情况下增强用户/物品表示并有效地对待正负样本，从而提高了多模态推荐的准确性和效率。

    

    本文研究了多模态推荐问题，其中利用物品的多模态信息（例如图像和文本描述）来提高推荐准确性。针对现有的最先进方法通常使用辅助图形（例如用户-用户或物品-物品关系图）来增强用户和/或物品的学习表示，本文提出了一种名为自助潜在表示（BLR）的新型自监督方法，可以结合学习用户/物品表示和图形结构。同时，为了减轻噪声监督信号的问题，我们提出了一种自举采样策略来有效地对待正负样本。在三个真实世界数据集上的大量实验表明，我们提出的方法在推荐精度和效率方面都优于现有最先进的方法。

    This paper studies the multi-modal recommendation problem, where the item multi-modality information (e.g., images and textual descriptions) is exploited to improve the recommendation accuracy. Besides the user-item interaction graph, existing state-of-the-art methods usually use auxiliary graphs (e.g., user-user or item-item relation graph) to augment the learned representations of users and/or items. These representations are often propagated and aggregated on auxiliary graphs using graph convolutional networks, which can be prohibitively expensive in computation and memory, especially for large graphs. Moreover, existing multi-modal recommendation methods usually leverage randomly sampled negative examples in Bayesian Personalized Ranking (BPR) loss to guide the learning of user/item representations, which increases the computational cost on large graphs and may also bring noisy supervision signals into the training process. To tackle the above issues, we propose a novel self-superv
    
[^12]: SelfCF：一种简单的自监督协同过滤框架

    SelfCF: A Simple Framework for Self-supervised Collaborative Filtering. (arXiv:2107.03019v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2107.03019](http://arxiv.org/abs/2107.03019)

    SelfCF是一种自监督协同过滤框架，用于推荐场景，通过增强现有的深度学习协同过滤模型中输出的嵌入来简化算法以及避免昂贵的计算和潜在的负样本问题。

    

    协同过滤（CF）被广泛用于从观察到的交互中学习有用的用户和项目的潜在表示。现有的基于CF的方法通常采用负采样来区分不同的项目。在大型数据集上使用负采样进行训练计算成本很高。此外，必须根据定义的分布谨慎选择负项，以避免在训练数据集中选择观察到的正项。不可避免地，从训练数据集中采样的一些负项在测试集中可能是正项。我们提出了一种专门用于隐式反馈推荐场景的自监督协同过滤框架（SelfCF）。所提出的SelfCF框架简化了连体网络，并可轻松应用于现有的基于深度学习的CF模型，我们称其为骨干网络。SelfCF的主要思想是增强由骨干网络生成的输出嵌入。

    Collaborative filtering (CF) is widely used to learn informative latent representations of users and items from observed interactions. Existing CF-based methods commonly adopt negative sampling to discriminate different items. Training with negative sampling on large datasets is computationally expensive. Further, negative items should be carefully sampled under the defined distribution, in order to avoid selecting an observed positive item in the training dataset. Unavoidably, some negative items sampled from the training dataset could be positive in the test set. In this paper, we propose a self-supervised collaborative filtering framework (SelfCF), that is specially designed for recommender scenario with implicit feedback. The proposed SelfCF framework simplifies the Siamese networks and can be easily applied to existing deep-learning based CF models, which we refer to as backbone networks. The main idea of SelfCF is to augment the output embeddings generated by backbone networks, b
    

