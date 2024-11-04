# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generate then Retrieve: Conversational Response Retrieval Using LLMs as Answer and Query Generators](https://arxiv.org/abs/2403.19302) | 本文提出了一种利用大型语言模型生成多个查询以增强对话响应检索的方法，并基于不同LLMs实现评估，同时还提出了一个新的TREC iKAT基准。 |
| [^2] | [Multi-Agent Collaboration Framework for Recommender Systems](https://arxiv.org/abs/2402.15235) | MACRec是一个新颖的框架，旨在通过多智能体协作来增强推荐系统，提供了多种专业智能体的协作努力解决推荐任务，并提供了应用示例。 |

# 详细

[^1]: 生成然后检索：使用LLM作为答案和查询生成器的对话响应检索

    Generate then Retrieve: Conversational Response Retrieval Using LLMs as Answer and Query Generators

    [https://arxiv.org/abs/2403.19302](https://arxiv.org/abs/2403.19302)

    本文提出了一种利用大型语言模型生成多个查询以增强对话响应检索的方法，并基于不同LLMs实现评估，同时还提出了一个新的TREC iKAT基准。

    

    CIS是信息检索中的一个重要领域，专注于开发交互式知识助手。这些系统必须能够熟练地理解用户在对话环境中的信息需求，并检索相关信息。为实现这一目标，现有方法使用一个称为重写查询的查询来建模用户的信息需求，并将此查询用于段落检索。在本文中，我们提出了三种用于生成多个查询以增强检索的不同方法。在这些方法中，我们利用大型语言模型（LLMs）的能力来理解用户的信息需求和生成适当的响应，以生成多个查询。我们实现并评估了提出的模型，利用包括GPT-4和Llama-2在零-shot和少-shot设置中的各种LLMs。此外，我们基于gpt 3.5的判断提出了一个针对TREC iKAT的新基准。我们的实验揭示了效果

    arXiv:2403.19302v1 Announce Type: new  Abstract: CIS is a prominent area in IR that focuses on developing interactive knowledge assistants. These systems must adeptly comprehend the user's information requirements within the conversational context and retrieve the relevant information. To this aim, the existing approaches model the user's information needs with one query called rewritten query and use this query for passage retrieval. In this paper, we propose three different methods for generating multiple queries to enhance the retrieval. In these methods, we leverage the capabilities of large language models (LLMs) in understanding the user's information need and generating an appropriate response, to generate multiple queries. We implement and evaluate the proposed models utilizing various LLMs including GPT-4 and Llama-2 chat in zero-shot and few-shot settings. In addition, we propose a new benchmark for TREC iKAT based on gpt 3.5 judgments. Our experiments reveal the effectivenes
    
[^2]: 用于推荐系统的多智能体协作框架

    Multi-Agent Collaboration Framework for Recommender Systems

    [https://arxiv.org/abs/2402.15235](https://arxiv.org/abs/2402.15235)

    MACRec是一个新颖的框架，旨在通过多智能体协作来增强推荐系统，提供了多种专业智能体的协作努力解决推荐任务，并提供了应用示例。

    

    基于LLM的智能体因其决策技能和处理复杂任务的能力而受到广泛关注。鉴于当前在利用智能体协作能力增强推荐系统方面存在的空白，我们介绍了MACRec，这是一个旨在通过多智能体协作增强推荐系统的新颖框架。与现有关于使用智能体进行用户/商品模拟的工作不同，我们旨在部署多智能体直接处理推荐任务。在我们的框架中，通过各种专业智能体的协作努力来解决推荐任务，包括经理、用户/商品分析师、反射器、搜索器和任务解释器，它们具有不同的工作流。此外，我们提供应用示例，说明开发人员如何轻松在各种推荐任务上使用MACRec，包括评分预测、序列推荐、对话推荐和解释生成。

    arXiv:2402.15235v1 Announce Type: new  Abstract: LLM-based agents have gained considerable attention for their decision-making skills and ability to handle complex tasks. Recognizing the current gap in leveraging agent capabilities for multi-agent collaboration in recommendation systems, we introduce MACRec, a novel framework designed to enhance recommendation systems through multi-agent collaboration. Unlike existing work on using agents for user/item simulation, we aim to deploy multi-agents to tackle recommendation tasks directly. In our framework, recommendation tasks are addressed through the collaborative efforts of various specialized agents, including Manager, User/Item Analyst, Reflector, Searcher, and Task Interpreter, with different working flows. Furthermore, we provide application examples of how developers can easily use MACRec on various recommendation tasks, including rating prediction, sequential recommendation, conversational recommendation, and explanation generation
    

