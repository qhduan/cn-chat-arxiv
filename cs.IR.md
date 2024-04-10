# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LLaRA: Aligning Large Language Models with Sequential Recommenders.](http://arxiv.org/abs/2312.02445) | LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。 |
| [^2] | [Multiple Models for Recommending Temporal Aspects of Entities.](http://arxiv.org/abs/1803.07890) | 本研究提出了一种新颖的基于事件中心的集合排名方法，该方法考虑到时间动态性，能够推荐最相关的实体方面，提高搜索体验。 |

# 详细

[^1]: LLaRA: 使用顺序推荐器对齐大型语言模型

    LLaRA: Aligning Large Language Models with Sequential Recommenders. (arXiv:2312.02445v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.02445](http://arxiv.org/abs/2312.02445)

    LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。

    

    顺序推荐旨在根据用户的历史交互预测与用户偏好相匹配的后续项目。随着大型语言模型 (LLMs) 的发展，人们对于将LLMs 应用于顺序推荐并将其视为语言建模任务的潜力越来越感兴趣。之前的工作中，使用ID索引或文本索引来表示文本提示中的项目，并将提示输入LLMs，但无法全面融合世界知识或展示足够的顺序理解能力。为了充分发挥传统推荐器（可以编码用户行为知识）和LLMs（具有项目的世界知识）的互补优势，我们提出了LLaRA - 一种大型语言和推荐助手框架。具体而言，LLaRA使用一种新颖的混合方法，将传统推荐器的基于ID的项目嵌入与文本项目特征整合到LLM的输入提示中。

    Sequential recommendation aims to predict the subsequent items matching user preference based on her/his historical interactions. With the development of Large Language Models (LLMs), there is growing interest in exploring the potential of LLMs for sequential recommendation by framing it as a language modeling task. Prior works represent items in the textual prompts using either ID indexing or text indexing and feed the prompts into LLMs, but falling short of either encapsulating comprehensive world knowledge or exhibiting sufficient sequential understanding. To harness the complementary strengths of traditional recommenders (which encode user behavioral knowledge) and LLMs (which possess world knowledge about items), we propose LLaRA -- a Large Language and Recommendation Assistant framework. Specifically, LLaRA represents items in LLM's input prompts using a novel hybrid approach that integrates ID-based item embeddings from traditional recommenders with textual item features. Viewin
    
[^2]: 推荐实体的时间因素的多模型方法

    Multiple Models for Recommending Temporal Aspects of Entities. (arXiv:1803.07890v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/1803.07890](http://arxiv.org/abs/1803.07890)

    本研究提出了一种新颖的基于事件中心的集合排名方法，该方法考虑到时间动态性，能够推荐最相关的实体方面，提高搜索体验。

    

    实体方面的推荐是语义搜索中的新兴任务，可以帮助用户发现与实体相关的巧合和突出信息，其中显着性（例如流行度）是以前工作中最重要的因素。但是，实体方面是具有时间动态性的，经常受到随时间发生的事件的影响。在这种情况下，仅基于显着性特征的方面建议可能会给出令人不满意的结果，原因有两个。首先，显着性通常在长时间段内累积，并且不考虑最近情况。其次，与事件实体相关的许多方面强烈依赖于时间。在本文中，我们研究了针对给定实体的时间方面推荐任务，旨在推荐最相关的方面，并考虑时间以提高搜索体验。我们提出了一种新颖的基于事件中心的集合排名方法，该方法从多个时间和类型依赖的模型中学习，并动态权衡显着性和最近情况。

    Entity aspect recommendation is an emerging task in semantic search that helps users discover serendipitous and prominent information with respect to an entity, of which salience (e.g., popularity) is the most important factor in previous work. However, entity aspects are temporally dynamic and often driven by events happening over time. For such cases, aspect suggestion based solely on salience features can give unsatisfactory results, for two reasons. First, salience is often accumulated over a long time period and does not account for recency. Second, many aspects related to an event entity are strongly time-dependent. In this paper, we study the task of temporal aspect recommendation for a given entity, which aims at recommending the most relevant aspects and takes into account time in order to improve search experience. We propose a novel event-centric ensemble ranking method that learns from multiple time and type-dependent models and dynamically trades off salience and recency c
    

