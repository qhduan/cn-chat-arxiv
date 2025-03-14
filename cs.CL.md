# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Subobject-level Image Tokenization](https://arxiv.org/abs/2402.14327) | 提出一种在子对象级别进行图像标记的方法，通过序列自编码器将子对象段压缩为紧凑的嵌入向量，实现了有效地将图像转换为对象和属性描述的学习。 |
| [^2] | [Punctuation Restoration Improves Structure Understanding without Supervision](https://arxiv.org/abs/2402.08382) | 标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。 |
| [^3] | [Helping the Helper: Supporting Peer Counselors via AI-Empowered Practice and Feedback.](http://arxiv.org/abs/2305.08982) | 本论文介绍了一个基于AI的交互式工具CARE，用于支持同侪辅导员通过自动建议生成来提高他们的能力。利用 Motivational Interviewing 框架，CARE 在实际培训阶段帮助辅导员诊断哪种具体的辅导策略最合适，并提供个性化的响应示例作为建议。 |

# 详细

[^1]: 子对象级图像标记化

    Subobject-level Image Tokenization

    [https://arxiv.org/abs/2402.14327](https://arxiv.org/abs/2402.14327)

    提出一种在子对象级别进行图像标记的方法，通过序列自编码器将子对象段压缩为紧凑的嵌入向量，实现了有效地将图像转换为对象和属性描述的学习。

    

    基于Transformer的视觉模型通常将图像标记为固定大小的方形补丁作为输入单元，这种方法缺乏对图像内容的适应性，并忽略了固有的像素分组结构。受语言模型广泛采用的子词标记化启发，我们提出了一种在子对象级别进行图像标记的方法，其中子对象由通过分割模型（例如，分割任何模型）获得的具有语义意义的图像段表示。为了实现基于子对象标记化的学习系统，我们首先引入了一个序列自编码器（SeqAE），将不同大小和形状的子对象段压缩为紧凑的嵌入向量，然后将子对象嵌入馈送到大型语言模型进行视觉语言学习。实证结果表明，我们的子对象级别标记化显著促进了有效地将图像转换为对象和属性描述的学习。

    arXiv:2402.14327v1 Announce Type: cross  Abstract: Transformer-based vision models typically tokenize images into fixed-size square patches as input units, which lacks the adaptability to image content and overlooks the inherent pixel grouping structure. Inspired by the subword tokenization widely adopted in language models, we propose an image tokenizer at a subobject level, where the subobjects are represented by semantically meaningful image segments obtained by segmentation models (e.g., segment anything models). To implement a learning system based on subobject tokenization, we first introduced a Sequence-to-sequence AutoEncoder (SeqAE) to compress subobject segments of varying sizes and shapes into compact embedding vectors, then fed the subobject embeddings into a large language model for vision language learning. Empirical results demonstrated that our subobject-level tokenization significantly facilitates efficient learning of translating images into object and attribute descr
    
[^2]: 标点符号恢复在没有监督的情况下改善结构理解

    Punctuation Restoration Improves Structure Understanding without Supervision

    [https://arxiv.org/abs/2402.08382](https://arxiv.org/abs/2402.08382)

    标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。

    

    无监督学习目标，如语言建模和去噪等，在生成预训练模型方面起着重要作用，这些预训练模型能够执行从自然语言理解到会话任务的各种下游应用。然而，尽管最近的大型语言模型具有令人印象深刻的对话能力，但它们在捕捉文本的句法或语义结构方面的能力仍然落后。我们假设，语言性能和机器能力之间的不匹配归因于当前流行的预训练目标未能充分传递语言结构知识给计算系统。我们展示了标点符号恢复对结构相关任务的内部和外部表现的改善，如命名实体识别、开放式信息提取、分块和词性标注。标点符号恢复是一个有效的学习目标，可以改善结构理解并产生更加鲁棒的模型。

    Unsupervised learning objectives like language modeling and de-noising constitute a significant part in producing pre-trained models that perform various downstream applications from natural language understanding to conversational tasks. However, despite impressive conversational capabilities of recent large language model, their abilities to capture syntactic or semantic structure within text lag behind. We hypothesize that the mismatch between linguistic performance and competence in machines is attributable to insufficient transfer of linguistic structure knowledge to computational systems with currently popular pre-training objectives. We show that punctuation restoration transfers to improvements in in- and out-of-distribution performance on structure-related tasks like named entity recognition, open information extraction, chunking, and part-of-speech tagging. Punctuation restoration is an effective learning objective that can improve structure understanding and yield a more rob
    
[^3]: 帮助帮助者：通过 AI 强化实践和反馈来支持同侪辅导员。

    Helping the Helper: Supporting Peer Counselors via AI-Empowered Practice and Feedback. (arXiv:2305.08982v1 [cs.HC])

    [http://arxiv.org/abs/2305.08982](http://arxiv.org/abs/2305.08982)

    本论文介绍了一个基于AI的交互式工具CARE，用于支持同侪辅导员通过自动建议生成来提高他们的能力。利用 Motivational Interviewing 框架，CARE 在实际培训阶段帮助辅导员诊断哪种具体的辅导策略最合适，并提供个性化的响应示例作为建议。

    

    数百万用户来到在线同侪辅导平台寻求关于从关系压力到焦虑等多种主题的支持。然而，研究表明，在线同侪支持群体并不总是像预期的那样有效，这主要是由于用户与无用的辅导员产生了负面体验。同侪辅导员是在线同侪辅导平台成功的关键，但他们中的大多数通常没有系统地接收指导或监督的方式。在这项工作中，我们介绍 CARE：一个交互式的基于 AI 的工具，通过自动建议生成增强同侪辅导员的能力。在实际培训阶段，CARE 帮助诊断在给定情境下哪些具体的辅导策略最合适，并提供量身定制的示例响应作为建议。辅导员可以选择在回复求助者之前选择、修改或忽略任何建议。

    Millions of users come to online peer counseling platforms to seek support on diverse topics ranging from relationship stress to anxiety. However, studies show that online peer support groups are not always as effective as expected largely due to users' negative experiences with unhelpful counselors. Peer counselors are key to the success of online peer counseling platforms, but most of them often do not have systematic ways to receive guidelines or supervision. In this work, we introduce CARE: an interactive AI-based tool to empower peer counselors through automatic suggestion generation. During the practical training stage, CARE helps diagnose which specific counseling strategies are most suitable in the given context and provides tailored example responses as suggestions. Counselors can choose to select, modify, or ignore any suggestion before replying to the support seeker. Building upon the Motivational Interviewing framework, CARE utilizes large-scale counseling conversation data
    

