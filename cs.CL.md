# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library](https://arxiv.org/abs/2404.00699) | LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。 |
| [^2] | [Improving Cross-lingual Representation for Semantic Retrieval with Code-switching](https://arxiv.org/abs/2403.01364) | 提出了一种通过代码切换的交替跨语言PTM，首次将代码切换方法应用于跨语言语义检索。 |
| [^3] | [Beyond Hate Speech: NLP's Challenges and Opportunities in Uncovering Dehumanizing Language](https://arxiv.org/abs/2402.13818) | 本文评估了几种最先进的NLP模型在识别贬低性语言方面的性能，发现它们能够以70%的准确率区分贬低性语言和更广泛的仇恨言论，但也存在着偏见。 |
| [^4] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |
| [^5] | [Exploring Value Biases: How LLMs Deviate Towards the Ideal](https://arxiv.org/abs/2402.11005) | 研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。 |
| [^6] | [When Dialects Collide: How Socioeconomic Mixing Affects Language Use.](http://arxiv.org/abs/2307.10016) | 本研究使用地理标记的推特数据和计算方法，在英格兰和威尔士的七千个行政区域上进行了大规模映射，发现社会经济交叉影响了语言使用，混合不同社会经济阶层的人群频率偏离标准语法的程度越高，其收入关联越弱。 |

# 详细

[^1]: LLM受到多少污染？一项全面调查和LLMSanitize库

    How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library

    [https://arxiv.org/abs/2404.00699](https://arxiv.org/abs/2404.00699)

    LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。

    

    随着近年来大型语言模型（LLMs）的崛起，新的机会正在出现，但也带来了新的挑战，污染问题迅速变得至关重要。企业应用和人工智能筹款已经达到一定规模，流行的问答基准提高几个百分点可能意味着数百万美元，对模型的完整性施加了巨大压力。同时，追踪LLMs见过的数据变得越来越困难；对于像GPT-4和Claude-3这样的闭源模型，他们不透露任何有关训练集的信息。因此，污染成为一个关键问题：LLMs的性能可能不再可靠，因为其高性能至少部分归因于其先前接触到的数据。这种局限性危及了自然语言处理领域的整体进展，然而，如何有效解决这一问题仍然缺乏方法。

    arXiv:2404.00699v1 Announce Type: new  Abstract: With the rise of Large Language Models (LLMs) in recent years, new opportunities are emerging, but also new challenges, and contamination is quickly becoming critical. Business applications and fundraising in AI have reached a scale at which a few percentage points gained on popular question-answering benchmarks could translate into dozens of millions of dollars, placing high pressure on model integrity. At the same time, it is becoming harder and harder to keep track of the data that LLMs have seen; if not impossible with closed-source models like GPT-4 and Claude-3 not divulging any information on the training set. As a result, contamination becomes a critical issue: LLMs' performance may not be reliable anymore, as the high performance may be at least partly due to their previous exposure to the data. This limitation jeopardizes the entire progress in the field of NLP, yet, there remains a lack of methods on how to efficiently address
    
[^2]: 通过代码切换改进语义检索的跨语言表示

    Improving Cross-lingual Representation for Semantic Retrieval with Code-switching

    [https://arxiv.org/abs/2403.01364](https://arxiv.org/abs/2403.01364)

    提出了一种通过代码切换的交替跨语言PTM，首次将代码切换方法应用于跨语言语义检索。

    

    arXiv:2403.01364v1 公告类型：新 提要：语义检索（SR）已成为任务导向问答（QA）对话场景中FAQ系统中不可或缺的部分。最近，对于电子商务平台或某些特定业务环境的跨语言智能客户服务系统的需求日益增加。大多数先前的研究直接利用跨语言预训练模型（PTMs）用于多语言知识检索，而其他一些研究也利用持续预训练在对下游任务的PTMs进行微调之前。然而，无论使用哪种模式，先前的工作都忽略了向PTMs告知与SR相关的一些特征，即在不提供与SR相关的任何信号的情况下训练他们的PTMs。为此，在这项工作中，我们提出了一种通过代码切换的交替跨语言PTM用于SR。我们是第一个为跨语言SR使用代码切换方法的研究。此外，我们还介绍了新颖的代码切换持续预训练方法。

    arXiv:2403.01364v1 Announce Type: new  Abstract: Semantic Retrieval (SR) has become an indispensable part of the FAQ system in the task-oriented question-answering (QA) dialogue scenario. The demands for a cross-lingual smart-customer-service system for an e-commerce platform or some particular business conditions have been increasing recently. Most previous studies exploit cross-lingual pre-trained models (PTMs) for multi-lingual knowledge retrieval directly, while some others also leverage the continual pre-training before fine-tuning PTMs on the downstream tasks. However, no matter which schema is used, the previous work ignores to inform PTMs of some features of the downstream task, i.e. train their PTMs without providing any signals related to SR. To this end, in this work, we propose an Alternative Cross-lingual PTM for SR via code-switching. We are the first to utilize the code-switching approach for cross-lingual SR. Besides, we introduce the novel code-switched continual pre-t
    
[^3]: 超越仇恨言论: 自然语言处理在发现贬低性语言中的挑战与机遇

    Beyond Hate Speech: NLP's Challenges and Opportunities in Uncovering Dehumanizing Language

    [https://arxiv.org/abs/2402.13818](https://arxiv.org/abs/2402.13818)

    本文评估了几种最先进的NLP模型在识别贬低性语言方面的性能，发现它们能够以70%的准确率区分贬低性语言和更广泛的仇恨言论，但也存在着偏见。

    

    人身具象化被定义为仇恨言论的一种微妙但有害的表现形式，涉及否认个人的人类特质，通常导致对边缘群体的暴力行为。尽管自然语言处理在各个领域取得了显著进展，但其在检测贬低性言语方面的应用有限，主要是由于这一领域公开可用的带标签数据稀缺。本文评估了最先进的NLP模型（包括GPT-4、GPT-3.5和LLAMA-2）在识别贬低性语言方面的性能。我们的发现显示，虽然这些模型表现出潜力，达到了70%的准确率来区分贬低性言语和更广泛的仇恨言论，但它们也显示出偏见。它们在对其他形式的仇恨言论进行分类时过于敏感，将其误判为特定目标群体的人身具象化，同时更频繁地未能识别明显的人身具象化案例。

    arXiv:2402.13818v1 Announce Type: new  Abstract: Dehumanization, characterized as a subtle yet harmful manifestation of hate speech, involves denying individuals of their human qualities and often results in violence against marginalized groups. Despite significant progress in Natural Language Processing across various domains, its application in detecting dehumanizing language is limited, largely due to the scarcity of publicly available annotated data for this domain. This paper evaluates the performance of cutting-edge NLP models, including GPT-4, GPT-3.5, and LLAMA-2, in identifying dehumanizing language. Our findings reveal that while these models demonstrate potential, achieving a 70\% accuracy rate in distinguishing dehumanizing language from broader hate speech, they also display biases. They are over-sensitive in classifying other forms of hate speech as dehumanization for a specific subset of target groups, while more frequently failing to identify clear cases of dehumanizati
    
[^4]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    
[^5]: 探究价值偏好：LLMs偏向理想状态的偏差

    Exploring Value Biases: How LLMs Deviate Towards the Ideal

    [https://arxiv.org/abs/2402.11005](https://arxiv.org/abs/2402.11005)

    研究发现大型语言模型（LLMs）在给出响应时存在一个价值偏好的机制，倾向于偏向理想状态，这种偏差会对不同应用场景产生重要影响。

    

    大型语言模型（LLMs）被部署在各种应用中，并且它们的响应对社会产生着越来越大的影响。理解LLMs在给出响应时的非故意机制对于解释它们的性能并辨别它们在现实世界应用中的偏差至关重要。这类似于人类研究中，这种无意识的响应被称为抽样。我们研究了LLMs的这种抽样现象，发现LLMs的抽样倾向于偏爱高价值选项。价值偏好对应于从最可能的响应向LLM中代表的理想价值的转变。实际上，即便是通过上下文提示学习到的新实体，这种效果也能够再现。我们表明这种偏差表现在意想不到的地方，并对选择典型实例等相关应用场景产生影响。结果显示，价值偏好在不同分类的LLMs中都很明显。

    arXiv:2402.11005v1 Announce Type: cross  Abstract: Large-Language-Models (LLMs) are deployed in a wide range of applications, and their response has an increasing social impact. Understanding the non-deliberate(ive) mechanism of LLMs in giving responses is essential in explaining their performance and discerning their biases in real-world applications. This is analogous to human studies, where such inadvertent responses are referred to as sampling. We study this sampling of LLMs in light of value bias and show that the sampling of LLMs tends to favour high-value options. Value bias corresponds to this shift of response from the most likely towards an ideal value represented in the LLM. In fact, this effect can be reproduced even with new entities learnt via in-context prompting. We show that this bias manifests in unexpected places and has implications on relevant application scenarios, like choosing exemplars. The results show that value bias is strong in LLMs across different categor
    
[^6]: 方言的碰撞：社会经济交叉对语言使用的影响

    When Dialects Collide: How Socioeconomic Mixing Affects Language Use. (arXiv:2307.10016v1 [physics.soc-ph] CROSS LISTED)

    [http://arxiv.org/abs/2307.10016](http://arxiv.org/abs/2307.10016)

    本研究使用地理标记的推特数据和计算方法，在英格兰和威尔士的七千个行政区域上进行了大规模映射，发现社会经济交叉影响了语言使用，混合不同社会经济阶层的人群频率偏离标准语法的程度越高，其收入关联越弱。

    

    人们的社会经济背景与他们使用标准语言的方式并不独立，这在各种社会语言学研究中已经得到证明。然而，不同社会经济阶层的人们交叉混合可能对这些相关性造成何种影响，在定量角度上尚未得到充分探索。在这项工作中，我们利用带地理标记的推特和可转移的计算方法，在英格兰和威尔士的七千个行政区域上对与标准英语偏离的情况进行大规模映射。我们将这些数据与高分辨率的收入地图结合起来，为居住地用户分配一个代理社会经济指标。令人惊讶的是，在八个大都市区域，我们发现一种一致的模式，表明不同社会经济阶层的人们混合得越多，其偏离标准语法和收入的频率就越不相互依存。

    The socioeconomic background of people and how they use standard forms of language are not independent, as demonstrated in various sociolinguistic studies. However, the extent to which these correlations may be influenced by the mixing of people from different socioeconomic classes remains relatively unexplored from a quantitative perspective. In this work we leverage geotagged tweets and transferable computational methods to map deviations from standard English on a large scale, in seven thousand administrative areas of England and Wales. We combine these data with high-resolution income maps to assign a proxy socioeconomic indicator to home-located users. Strikingly, across eight metropolitan areas we find a consistent pattern suggesting that the more different socioeconomic classes mix, the less interdependent the frequency of their departures from standard grammar and their income become. Further, we propose an agent-based model of linguistic variety adoption that sheds light on th
    

