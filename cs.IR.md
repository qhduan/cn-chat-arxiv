# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning](https://arxiv.org/abs/2403.00843) | 利用大型语言模型的规划能力来增强长期推荐，使模型在个性化推荐中更有效地理解和应用任务解决原则 |
| [^2] | [A First Look at Information Highlighting in Stack Overflow Answers.](http://arxiv.org/abs/2401.01472) | 本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。 |
| [^3] | [Retrieving Texts based on Abstract Descriptions.](http://arxiv.org/abs/2305.12517) | 本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。 |

# 详细

[^1]: 利用双层可学习大型语言模型规划增强长期推荐

    Enhancing Long-Term Recommendation with Bi-level Learnable Large Language Model Planning

    [https://arxiv.org/abs/2403.00843](https://arxiv.org/abs/2403.00843)

    利用大型语言模型的规划能力来增强长期推荐，使模型在个性化推荐中更有效地理解和应用任务解决原则

    

    传统推荐系统倾向于过分迎合用户的即时兴趣而忽视他们的长期参与。 为了解决这个问题，在推荐决策过程中合并规划能力是至关重要的，以开发能够同时考虑即时兴趣和长期参与的策略。本文提出利用大型语言模型（LLMs）对稀疏数据的显著规划能力用于长期推荐。关键在于使语言模型能够在个性化推荐场景中有效理解和应用任务解决原则，因为模型的预训练可能并未自然包含这些内容。

    arXiv:2403.00843v1 Announce Type: cross  Abstract: Traditional recommendation setting tends to excessively cater to users' immediate interests and neglect their long-term engagement. To address it, it is crucial to incorporate planning capabilities into the recommendation decision-making process to develop policies that take into account both immediate interests and long-term engagement. Despite Reinforcement Learning (RL) can learn planning capacity by maximizing cumulative reward, the scarcity of recommendation data presents challenges such as instability and susceptibility to overfitting when training RL models from scratch.   In this context, we propose to leverage the remarkable planning capabilities over sparse data of Large Language Models (LLMs) for long-term recommendation. The key lies in enabling a language model to understand and apply task-solving principles effectively in personalized recommendation scenarios, as the model's pre-training may not naturally encompass these 
    
[^2]: Stack Overflow回答中信息高亮的初探

    A First Look at Information Highlighting in Stack Overflow Answers. (arXiv:2401.01472v1 [cs.CL])

    [http://arxiv.org/abs/2401.01472](http://arxiv.org/abs/2401.01472)

    本论文进行了首次大规模的探索性研究，研究了Stack Overflow回答中的信息高亮。通过使用神经网络架构，开发了自动推荐突出内容的方法。

    

    背景：浏览Stack Overflow（SO）的知识仍然具有挑战性。为了使帖子对用户更生动，SO允许用户使用Markdown或HTML编写和编辑帖子，以便用户可以利用各种格式化样式（例如粗体、斜体和代码）来突出重要信息。然而，关于突出信息的研究仍然有限。目标：我们在最近的研究中进行了首次大规模的探索性研究，研究了SO回答中的信息高亮。为了扩展我们之前的研究，我们利用最初设计用于命名实体识别任务的神经网络架构，开发了自动推荐带有格式化样式的突出内容的方法。方法：本文研究了Stack Overflow的31,169,429个回答。为了训练推荐模型，我们选择了CNN和BERT模型，针对每种格式化类型（即粗体、斜体、代码和标题）使用我们从SO回答收集的突出信息数据集。

    Context: Navigating the knowledge of Stack Overflow (SO) remains challenging. To make the posts vivid to users, SO allows users to write and edit posts with Markdown or HTML so that users can leverage various formatting styles (e.g., bold, italic, and code) to highlight the important information. Nonetheless, there have been limited studies on the highlighted information. Objective: We carried out the first large-scale exploratory study on the information highlighted in SO answers in our recent study. To extend our previous study, we develop approaches to automatically recommend highlighted content with formatting styles using neural network architectures initially designed for the Named Entity Recognition task. Method: In this paper, we studied 31,169,429 answers of Stack Overflow. For training recommendation models, we choose CNN and BERT models for each type of formatting (i.e., Bold, Italic, Code, and Heading) using the information highlighting dataset we collected from SO answers.
    
[^3]: 基于摘要描述的文本检索

    Retrieving Texts based on Abstract Descriptions. (arXiv:2305.12517v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12517](http://arxiv.org/abs/2305.12517)

    本研究针对语义检索问题，提出了一种基于摘要描述的文本检索模型，通过改进当前的文本嵌入方法，在标准最近邻搜索中取得了显著性能提升。

    

    虽然针对文本的信息提取，指令优化的大型语言模型表现优异，但对于在大规模文档集合中定位符合给定描述的文本（语义检索）并不适用。基于嵌入向量的相似度搜索可以通过查询执行检索，但嵌入中的相似度定义不明确且不一致，并且对于许多用例来说都是次优的。那么，什么是有效检索的好的查询表示？我们确定了根据内容的摘要描述检索句子的明确定义且一致的任务。我们展示了当前文本嵌入的不足，并提出了一种替代模型，在标准最近邻搜索中的表现显著提升。该模型使用通过提示LLM获得的正负样本对进行训练。虽然很容易从LLM中获得训练材料，但LLM无法直接执行检索任务。

    While instruction-tuned Large Language Models (LLMs) excel at extracting information from text, they are not suitable for locating texts conforming to a given description in a large document collection (semantic retrieval). Similarity search over embedding vectors does allow to perform retrieval by query, but the similarity reflected in the embedding is ill-defined and non-consistent, and is sub-optimal for many use cases. What, then, is a good query representation for effective retrieval?  We identify the well defined and consistent task of retrieving sentences based on abstract descriptions of their content. We demonstrate the inadequacy of current text embeddings and propose an alternative model that significantly improves when used in standard nearest neighbor search. The model is trained using positive and negative pairs sourced through prompting a LLM. While it is easy to source the training material from an LLM, the retrieval task cannot be performed by the LLM directly. This de
    

