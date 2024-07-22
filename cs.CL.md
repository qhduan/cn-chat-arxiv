# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection](https://arxiv.org/abs/2403.14270) | 提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。 |
| [^2] | [MoralBERT: Detecting Moral Values in Social Discourse](https://arxiv.org/abs/2403.07678) | MoralBERT 是一种专门设计用于捕捉文本中道德微妙之处的语言表示模型，利用来自Twitter、Reddit和Facebook的数据，扩大了模型理解道德的能力。 |
| [^3] | [A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing](https://arxiv.org/abs/2403.02504) | 预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。 |
| [^4] | [Cognitive Bias in High-Stakes Decision-Making with LLMs](https://arxiv.org/abs/2403.00811) | 提出了BiasBuster框架，用于揭示、评估和减轻LLMs中的认知偏见，特别是在高风险决策任务中，通过开发包含16,800个提示的数据集和测试多种偏见缓解策略，并提出一种利用LLMs自身来消除其提示中偏见的新方法。 |
| [^5] | [Query-OPT: Optimizing Inference of Large Language Models via Multi-Query Instructions in Meeting Summarization](https://arxiv.org/abs/2403.00067) | 本研究旨在通过将相同输入上下文的查询组合为单个提示，以最小化重复调用来优化使用大型语言模型在会议摘要中的推理。 |
| [^6] | [Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models](https://arxiv.org/abs/2402.15131) | 提出了一种互动式KBQA框架，通过直接与知识库互动生成逻辑形式，开发了用于KB交互的通用API，并设计了示例来指导大型语言模型进行推理。 |
| [^7] | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。 |
| [^8] | [Transforming and Combining Rewards for Aligning Large Language Models](https://arxiv.org/abs/2402.00742) | 本研究主要研究了对齐大规模语言模型的方法中出现的两个问题：奖励模型的选择以及多个奖励模型的组合。通过引入概率解释，我们提出了一种从Bradley-Terry偏好模型中学习的奖励的自然变换选择，该变换强调改善表现不佳的输出，从而减轻了欠拟合和奖励欺骗。 |
| [^9] | [Comparing Human-Centered Language Modeling: Is it Better to Model Groups, Individual Traits, or Both?.](http://arxiv.org/abs/2401.12492) | 本研究比较了以群体属性、个体用户和组合方法来模拟人的上下文。合并群体和个体特征显著提高了用户级回归任务的性能，而模拟个体用户则显著提高了单个文档级分类任务的性能。 |
| [^10] | [Locating Cross-Task Sequence Continuation Circuits in Transformers.](http://arxiv.org/abs/2311.04131) | 通过分析和比较Transformer模型中类似的序列继续任务的电路，研究发现共享的计算结构可以提高模型的行为预测能力、错误识别能力和编辑过程的安全性。 |
| [^11] | [WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research.](http://arxiv.org/abs/2303.17395) | 本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。 |
| [^12] | [Geolocation Predicting of Tweets Using BERT-Based Models.](http://arxiv.org/abs/2303.07865) | 该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。 |

# 详细

[^1]: 场景图ViT：端到端的开放词汇视觉关系检测

    Scene-Graph ViT: End-to-End Open-Vocabulary Visual Relationship Detection

    [https://arxiv.org/abs/2403.14270](https://arxiv.org/abs/2403.14270)

    提出了一种简单高效的无解码器架构，用于开放词汇的视觉关系检测，通过Transformer-based图像编码器隐式建模对象之间的关系，使用注意力机制提取关系信息，在混合数据上进行端到端训练，实现了最先进的关系检测性能。

    

    视觉关系检测旨在识别图像中的对象及其关系。以往的方法通过在现有目标检测架构中添加单独的关系模块或解码器来处理此任务。这种分离增加了复杂性，阻碍了端到端训练，限制了性能。我们提出了一种简单且高效的无解码器架构，用于开放词汇的视觉关系检测。我们的模型由基于Transformer的图像编码器组成，将对象表示为标记，并隐含地建模它们的关系。为了提取关系信息，我们引入了一个注意力机制，选择可能形成关系的对象对。我们提供了一个单阶段的训练方法，可以在混合对象和关系检测数据上训练此模型。我们的方法在Visual Genome和大词汇GQA基准测试上实现了最先进的关系检测性能，可实现实时性。

    arXiv:2403.14270v1 Announce Type: cross  Abstract: Visual relationship detection aims to identify objects and their relationships in images. Prior methods approach this task by adding separate relationship modules or decoders to existing object detection architectures. This separation increases complexity and hinders end-to-end training, which limits performance. We propose a simple and highly efficient decoder-free architecture for open-vocabulary visual relationship detection. Our model consists of a Transformer-based image encoder that represents objects as tokens and models their relationships implicitly. To extract relationship information, we introduce an attention mechanism that selects object pairs likely to form a relationship. We provide a single-stage recipe to train this model on a mixture of object and relationship detection data. Our approach achieves state-of-the-art relationship detection performance on Visual Genome and on the large-vocabulary GQA benchmark at real-tim
    
[^2]: MoralBERT：检测社会话语中的道德价值

    MoralBERT: Detecting Moral Values in Social Discourse

    [https://arxiv.org/abs/2403.07678](https://arxiv.org/abs/2403.07678)

    MoralBERT 是一种专门设计用于捕捉文本中道德微妙之处的语言表示模型，利用来自Twitter、Reddit和Facebook的数据，扩大了模型理解道德的能力。

    

    道德在我们感知信息、影响决策和判断过程中起着基础性作用。包括疫苗接种、堕胎、种族主义和性取向在内的有争议话题往往引发的意见和态度并非仅基于证据，而更多反映了道德世界观。最近自然语言处理的进展表明，道德价值可以从人类生成的文本内容中得到判断。本文设计了一系列旨在捕捉文本中道德微妙之处的语言表示模型，称为MoralBERT。我们利用来自三个不同来源（Twitter、Reddit和Facebook）的带有注释的道德数据，涵盖各种社会相关主题。这种方法扩大了语言多样性，可能增强模型在不同上下文中理解道德的能力。我们还探讨了一种领域自适应技术，并将其与标准的微调方法进行了比较。

    arXiv:2403.07678v1 Announce Type: new  Abstract: Morality plays a fundamental role in how we perceive information while greatly influencing our decisions and judgements. Controversial topics, including vaccination, abortion, racism, and sexuality, often elicit opinions and attitudes that are not solely based on evidence but rather reflect moral worldviews. Recent advances in natural language processing have demonstrated that moral values can be gauged in human-generated textual content. Here, we design a range of language representation models fine-tuned to capture exactly the moral nuances in text, called MoralBERT. We leverage annotated moral data from three distinct sources: Twitter, Reddit, and Facebook user-generated content covering various socially relevant topics. This approach broadens linguistic diversity and potentially enhances the models' ability to comprehend morality in various contexts. We also explore a domain adaptation technique and compare it to the standard fine-tu
    
[^3]: 自然语言处理中的预训练-微调范式教程

    A Tutorial on the Pretrain-Finetune Paradigm for Natural Language Processing

    [https://arxiv.org/abs/2403.02504](https://arxiv.org/abs/2403.02504)

    预训练-微调范式在自然语言处理中展现了显著的效率，尤其对社会科学研究中数据有限的情况下具有益处。

    

    预训练-微调范式代表了自然语言处理中的一种变革性方法。该范式通过使用大型预训练语言模型区别于众，展示了在微调任务中即使训练数据有限也具有显著的效率。这种效率对社会科学研究特别有益，因为注释样本的数量通常非常有限。我们的教程全面介绍了预训练-微调范式。我们首先深入探讨了预训练和微调的基本概念，然后进行了实际应用的案例练习。我们展示了该范式在各种任务中的应用，包括多类别分类和回归。强调其高效性和用户友好性，该教程旨在鼓励更广泛地采纳这种范式。为此，我们提供了所有代码和数据集的开放访问。

    arXiv:2403.02504v1 Announce Type: cross  Abstract: The pretrain-finetune paradigm represents a transformative approach in natural language processing (NLP). This paradigm distinguishes itself through the use of large pretrained language models, demonstrating remarkable efficiency in finetuning tasks, even with limited training data. This efficiency is especially beneficial for research in social sciences, where the number of annotated samples is often quite limited. Our tutorial offers a comprehensive introduction to the pretrain-finetune paradigm. We first delve into the fundamental concepts of pretraining and finetuning, followed by practical exercises using real-world applications. We demonstrate the application of the paradigm across various tasks, including multi-class classification and regression. Emphasizing its efficacy and user-friendliness, the tutorial aims to encourage broader adoption of this paradigm. To this end, we have provided open access to all our code and datasets
    
[^4]: LLM在高风险决策中的认知偏见

    Cognitive Bias in High-Stakes Decision-Making with LLMs

    [https://arxiv.org/abs/2403.00811](https://arxiv.org/abs/2403.00811)

    提出了BiasBuster框架，用于揭示、评估和减轻LLMs中的认知偏见，特别是在高风险决策任务中，通过开发包含16,800个提示的数据集和测试多种偏见缓解策略，并提出一种利用LLMs自身来消除其提示中偏见的新方法。

    

    大型语言模型(LLMs)在支持日益扩大的决策任务方面具有重要潜力。然而，由于它们在人类(创造的)数据上训练，LLMs可能会继承针对受保护群体的社会偏见，同时也可能受到认知偏见的影响。这种类似于人类的偏见可能会妨碍利用LLM协助做出公平和可解释的决策。我们的工作引入了BiasBuster，一个旨在揭示、评估和减轻LLMs中的认知偏见的框架，特别是在高风险决策任务中。受心理学和认知科学先前研究的启发，我们开发了一个包含16,800个提示的数据集，用于评估不同认知偏见(例如，提示诱导、顺序、固有)。我们测试了各种偏见缓解策略，同时提出了一种新方法，利用LLMs来消除它们自己的提示中的偏见。我们的分析提供了关于不同领域认知偏见存在和影响的全面图景。

    arXiv:2403.00811v1 Announce Type: new  Abstract: Large language models (LLMs) offer significant potential as tools to support an expanding range of decision-making tasks. However, given their training on human (created) data, LLMs can inherit both societal biases against protected groups, as well as be subject to cognitive bias. Such human-like bias can impede fair and explainable decisions made with LLM assistance. Our work introduces BiasBuster, a framework designed to uncover, evaluate, and mitigate cognitive bias in LLMs, particularly in high-stakes decision-making tasks. Inspired by prior research in psychology and cognitive sciences, we develop a dataset containing 16,800 prompts to evaluate different cognitive biases (e.g., prompt-induced, sequential, inherent). We test various bias mitigation strategies, amidst proposing a novel method using LLMs to debias their own prompts. Our analysis provides a comprehensive picture on the presence and effects of cognitive bias across diffe
    
[^5]: Query-OPT：通过多查询指令优化大型语言模型在会议摘要中的推理

    Query-OPT: Optimizing Inference of Large Language Models via Multi-Query Instructions in Meeting Summarization

    [https://arxiv.org/abs/2403.00067](https://arxiv.org/abs/2403.00067)

    本研究旨在通过将相同输入上下文的查询组合为单个提示，以最小化重复调用来优化使用大型语言模型在会议摘要中的推理。

    

    这项工作关注基于查询的会议摘要任务，在此任务中，针对特定查询对上下文（会议记录）生成摘要。使用大型语言模型（LLMs）进行此任务时，即使上下文保持不变，每个新查询也需要对LLM推理端点/API进行一次新调用。然而，反复调用LLM推理端点会显著增加在生产中使用它们的成本，这使得许多实际用例中LLMs都不切实际。为解决这一问题，在本文中，我们研究了是否可以成功地将相同输入上下文的查询组合为单个提示以最小化重复调用，在会议摘要中使用。在这方面，我们通过比较各种流行的LLM（GPT-4、PaLM-2、LLaMA-2、Mistral和FLAN-T5）在单查询和多查询设置中的表现进行了广泛实验。

    arXiv:2403.00067v1 Announce Type: new  Abstract: This work focuses on the task of query-based meeting summarization in which the summary of a context (meeting transcript) is generated in response to a specific query. When using Large Language Models (LLMs) for this task, a new call to the LLM inference endpoint/API is required for each new query even if the context stays the same. However, repeated calls to the LLM inference endpoints would significantly increase the costs of using them in production, making LLMs impractical for many real-world use cases. To address this problem, in this paper, we investigate whether combining the queries for the same input context in a single prompt to minimize repeated calls can be successfully used in meeting summarization. In this regard, we conduct extensive experiments by comparing the performance of various popular LLMs: GPT-4, PaLM-2, LLaMA-2, Mistral, and FLAN-T5 in single-query and multi-query settings. We observe that while most LLMs tend to
    
[^6]: 互动式知识库问答：基于大型语言模型的多轮交互式知识库问答

    Interactive-KBQA: Multi-Turn Interactions for Knowledge Base Question Answering with Large Language Models

    [https://arxiv.org/abs/2402.15131](https://arxiv.org/abs/2402.15131)

    提出了一种互动式KBQA框架，通过直接与知识库互动生成逻辑形式，开发了用于KB交互的通用API，并设计了示例来指导大型语言模型进行推理。

    

    本研究探讨了知识库问答（KBQA）的领域。KBQA被认为是一项具有挑战性的任务，特别是在将复杂问题解析为可执行逻辑形式方面。传统的基于语义解析（SP）的方法需要大量的数据注释，这导致了显著的成本。最近，由大型语言模型（LLM）推动的少样本上下文学习的出现展示了很好的能力。然而，在低资源情景下充分利用LLMs将问题解析为逻辑形式是一个重大挑战。为了应对这些障碍，我们引入了互动式知识库问答（Interactive-KBQA），这是一个旨在通过与知识库（KBs）直接互动来生成逻辑形式的框架。在这个框架内，我们开发了三个用于KB交互的通用API。对于每种复杂问题类别，我们设计了示例来指导LLMs完成推理过程。我们的方法取得了具有竞争力的结果。

    arXiv:2402.15131v1 Announce Type: cross  Abstract: This study explores the realm of knowledge-base question answering (KBQA). KBQA is considered a challenging task, particularly in parsing intricate questions into executable logical forms. Traditional semantic parsing (SP)-based methods require extensive data annotations, which result in significant costs. Recently, the advent of few-shot in-context learning, powered by large language models (LLMs), has showcased promising capabilities. Yet, fully leveraging LLMs to parse questions into logical forms in low-resource scenarios poses a substantial challenge. To tackle these hurdles, we introduce Interactive-KBQA, a framework designed to generate logical forms through direct interaction with knowledge bases (KBs). Within this framework, we have developed three generic APIs for KB interaction. For each category of complex question, we devised exemplars to guide LLMs through the reasoning processes. Our method achieves competitive results o
    
[^7]: SLEB: 通过冗余验证和消除Transformer块优化LLM的流程

    SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks

    [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

    SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。

    

    大型语言模型（LLM）在各种自然语言处理任务中证明了其高效性。然而，它们庞大的参数数量给实际部署带来了重大挑战。精简，一种旨在减小LLM大小和复杂度的技术，通过从网络中删除冗余组件提供了潜在解决方案。尽管精简有希望，但现有方法往往难以实现显著的端到端LLM推理加速。本文中，我们引入了SLEB，一种通过消除冗余的Transformer块来优化LLM流程的新方法。我们选择Transformer块作为精简的基本单位，因为LLM在相邻块的输出之间具有块级别的冗余和高相似性。这个选择使我们能够有效地增强LLM的处理速度。我们的实验证明，SLEB成功加速了LLM的推理过程。

    arXiv:2402.09025v1 Announce Type: new Abstract: Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical deployment. Pruning, a technique aimed at reducing the size and complexity of LLMs, offers a potential solution by removing redundant components from the network. Despite the promise of pruning, existing methods often struggle to achieve substantial end-to-end LLM inference speedup. In this paper, we introduce SLEB, a novel approach designed to streamline LLMs by eliminating redundant transformer blocks. We choose the transformer block as the fundamental unit for pruning, because LLMs exhibit block-level redundancy with high similarity between the outputs of neighboring blocks. This choice allows us to effectively enhance the processing speed of LLMs. Our experimental results demonstrate that SLEB successfully accelerates LLM inference without
    
[^8]: 改变和组合奖励以对齐大规模语言模型

    Transforming and Combining Rewards for Aligning Large Language Models

    [https://arxiv.org/abs/2402.00742](https://arxiv.org/abs/2402.00742)

    本研究主要研究了对齐大规模语言模型的方法中出现的两个问题：奖励模型的选择以及多个奖励模型的组合。通过引入概率解释，我们提出了一种从Bradley-Terry偏好模型中学习的奖励的自然变换选择，该变换强调改善表现不佳的输出，从而减轻了欠拟合和奖励欺骗。

    

    将语言模型与人类偏好对齐的常见方法是首先从偏好数据中学习奖励模型，然后使用该奖励模型来更新语言模型。我们研究了这种方法中出现的两个密切相关的问题。首先，奖励模型的任何单调变换都保持偏好排名；是否有一种比其他选择“更好”的选择？其次，我们经常希望将语言模型与多个特性对齐：我们如何组合多个奖励模型？通过对齐过程的概率解释，我们确定了从Bradley-Terry偏好模型学习的奖励（常见情况）的自然变换选择。这个派生的变换具有两个重要的属性。首先，它强调改进表现不佳的输出，而不是已经得分良好的输出。这既减轻了欠拟合（其中一些提示没有得到改进），又减少了奖励欺骗（模型学习利用错误指定）。

    A common approach for aligning language models to human preferences is to first learn a reward model from preference data, and then use this reward model to update the language model. We study two closely related problems that arise in this approach. First, any monotone transformation of the reward model preserves preference ranking; is there a choice that is ``better'' than others? Second, we often wish to align language models to multiple properties: how should we combine multiple reward models? Using a probabilistic interpretation of the alignment procedure, we identify a natural choice for transformation for (the common case of) rewards learned from Bradley-Terry preference models. This derived transformation has two important properties. First, it emphasizes improving poorly-performing outputs, rather than outputs that already score well. This mitigates both underfitting (where some prompts are not improved) and reward hacking (where the model learns to exploit misspecification of
    
[^9]: 比较以人为中心的语言建模：模拟群体、个体特点还是两者兼顾？

    Comparing Human-Centered Language Modeling: Is it Better to Model Groups, Individual Traits, or Both?. (arXiv:2401.12492v1 [cs.CL])

    [http://arxiv.org/abs/2401.12492](http://arxiv.org/abs/2401.12492)

    本研究比较了以群体属性、个体用户和组合方法来模拟人的上下文。合并群体和个体特征显著提高了用户级回归任务的性能，而模拟个体用户则显著提高了单个文档级分类任务的性能。

    

    自然语言处理在将人的上下文纳入其模型中取得了进展，但使用群体属性（如45岁以上的人群）还是模拟个体人物更有效的问题尚未确定。群体属性在技术上更容易实现，但是过于粗糙：并非所有45岁以上的人都以相同的方式书写。相反，模拟个体人物能够捕捉每个人身份的复杂性，允许更个性化的表示，但我们可能需要模拟无限数量的用户并且需要可能无法获取的数据。我们比较了通过群体属性、个体用户和组合方法来模拟人的上下文。将群体和个体特征结合起来，显著提高了基于用户文档的用户级回归任务（如年龄估计或个性评估）的性能。模拟个体用户显著提高了单个文档级分类任务（如立场和主题检测）的性能。

    Natural language processing has made progress in incorporating human context into its models, but whether it is more effective to use group-wise attributes (e.g., over-45-year-olds) or model individuals remains open. Group attributes are technically easier but coarse: not all 45-year-olds write the same way. In contrast, modeling individuals captures the complexity of each person's identity. It allows for a more personalized representation, but we may have to model an infinite number of users and require data that may be impossible to get. We compare modeling human context via group attributes, individual users, and combined approaches. Combining group and individual features significantly benefits user-level regression tasks like age estimation or personality assessment from a user's documents. Modeling individual users significantly improves the performance of single document-level classification tasks like stance and topic detection. We also find that individual-user modeling does w
    
[^10]: 在Transformer中定位跨任务序列继续电路

    Locating Cross-Task Sequence Continuation Circuits in Transformers. (arXiv:2311.04131v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.04131](http://arxiv.org/abs/2311.04131)

    通过分析和比较Transformer模型中类似的序列继续任务的电路，研究发现共享的计算结构可以提高模型的行为预测能力、错误识别能力和编辑过程的安全性。

    

    虽然Transformer模型在语言任务上展现出强大的能力，但其复杂的架构使其难以解释。最近的研究旨在将Transformer模型还原为可读的电路表示，用于实现算法功能。我们通过分析和比较类似的序列继续任务的电路来扩展这项研究，其中包括数字、数字词和月份的递增序列。通过应用电路分析技术，我们确定了负责检测序列成员和预测序列中下一个成员的关键子电路。我们的分析揭示了语义相关序列依赖于具有类似作用的共享电路子图。总体而言，记录共享的计算结构能够更好地预测模型行为，识别错误，并进行更安全的编辑过程。这种对Transformer的机械理解是构建更健壮、调试和编辑更安全的模型的关键一步。

    While transformer models exhibit strong capabilities on linguistic tasks, their complex architectures make them difficult to interpret. Recent work has aimed to reverse engineer transformer models into human-readable representations called circuits that implement algorithmic functions. We extend this research by analyzing and comparing circuits for similar sequence continuation tasks, which include increasing sequences of digits, number words, and months. Through the application of circuit analysis techniques, we identify key sub-circuits responsible for detecting sequence members and for predicting the next member in a sequence. Our analysis reveals that semantically related sequences rely on shared circuit subgraphs with analogous roles. Overall, documenting shared computational structures enables better prediction of model behaviors, identification of errors, and safer editing procedures. This mechanistic understanding of transformers is a critical step towards building more robust,
    
[^11]: WavCaps: 一种ChatGPT辅助的弱标注音频字幕数据集，用于音频-语言多模态研究

    WavCaps: A ChatGPT-Assisted Weakly-Labelled Audio Captioning Dataset for Audio-Language Multimodal Research. (arXiv:2303.17395v1 [eess.AS])

    [http://arxiv.org/abs/2303.17395](http://arxiv.org/abs/2303.17395)

    本文介绍了第一个大规模的弱标注音频字幕数据集WavCaps，含约40万条带有配对字幕的音频剪辑。为克服噪声标注的问题，提出了基于ChatGPT的三阶段字幕生成流程。

    

    近年来，音频-语言（AL）多模态学习任务的发展非常显著。然而，现有的AL数据集收集过程昂贵费时，规模有限，给研究者带来了挑战。为解决这个数据稀缺问题，我们介绍了WavCaps，这是第一个包含大约40万条带有配对字幕的大规模弱标注音频字幕数据集。我们从Web资源和声音事件检测数据集中获取音频剪辑及原始描述。但是，在线收集到的原始描述非常嘈杂，不适合用于自动化音频字幕等任务。为了克服这个问题，我们提出了一个三阶段的处理流程，以过滤嘈杂数据并生成高质量字幕，在其中利用了ChatGPT，一种大型语言模型，来自动过滤和转换原始描述。我们对WavCaps的特征进行了全面的分析。

    The advancement of audio-language (AL) multimodal learning tasks has been significant in recent years. However, researchers face challenges due to the costly and time-consuming collection process of existing audio-language datasets, which are limited in size. To address this data scarcity issue, we introduce WavCaps, the first large-scale weakly-labelled audio captioning dataset, comprising approximately 400k audio clips with paired captions. We sourced audio clips and their raw descriptions from web sources and a sound event detection dataset. However, the online-harvested raw descriptions are highly noisy and unsuitable for direct use in tasks such as automated audio captioning. To overcome this issue, we propose a three-stage processing pipeline for filtering noisy data and generating high-quality captions, where ChatGPT, a large language model, is leveraged to filter and transform raw descriptions automatically. We conduct a comprehensive analysis of the characteristics of WavCaps 
    
[^12]: 基于BERT模型的推文地理位置预测

    Geolocation Predicting of Tweets Using BERT-Based Models. (arXiv:2303.07865v1 [cs.CL])

    [http://arxiv.org/abs/2303.07865](http://arxiv.org/abs/2303.07865)

    该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。

    

    该研究旨在解决推文/用户地理位置预测任务，并提供了处理文本大数据地理标记的灵活方法。该方法采用基于神经网络的自然语言处理来估计坐标对（经度，纬度）和二维高斯混合模型（GMM）。提出的模型的范围已经在Twitter数据集上使用预训练的BERT模型进行调整。性能指标表明，对于在推文内容和元数据上训练和评估的模型，全球范围内的中位误差小于30公里，美国范围内的中位误差小于15公里。

    This research is aimed to solve the tweet/user geolocation prediction task and provide a flexible methodology for the geotagging of textual big data. The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as coordinate pairs (longitude, latitude) and two-dimensional Gaussian Mixture Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as base models. Performance metrics show a median error of fewer than 30 km on a worldwide-level, and fewer than 15 km on the US-level datasets for the models trained and evaluated on text features of tweets' content and metadata context.
    

