# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Invalsi Benchmark: measuring Language Models Mathematical and Language understanding in Italian](https://arxiv.org/abs/2403.18697) | 该研究提出了两个新的基准，用于评估语言模型在意大利语的数学和语言理解能力，为当前语言模型的性能提供了具有挑战性的评估标准。 |
| [^2] | [Dated Data: Tracing Knowledge Cutoffs in Large Language Models](https://arxiv.org/abs/2403.12958) | 本文提出了在大型语言模型中追踪知识截止日期的概念，通过资源级别的时间对齐性估计有效截止日期，并发现这些截止日期通常与报道的不同。 |
| [^3] | [Data Augmentation is Dead, Long Live Data Augmentation](https://arxiv.org/abs/2402.14895) | 数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能 |
| [^4] | [COBIAS: Contextual Reliability in Bias Assessment](https://arxiv.org/abs/2402.14889) | 我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。 |
| [^5] | [Multi-Hop Table Retrieval for Open-Domain Text-to-SQL](https://arxiv.org/abs/2402.10666) | 提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果 |
| [^6] | [Large Language Models for Mathematical Reasoning: Progresses and Challenges](https://arxiv.org/abs/2402.00157) | 大型语言模型(LLMs)在解决数学问题方面涉及了大量的数学问题类型和不同的数据集和设置。目前仍然存在一些挑战，需要进一步研究和解决。 |
| [^7] | [Towards Goal-oriented Large Language Model Prompting: A Survey.](http://arxiv.org/abs/2401.14043) | 本文调查了大型语言模型(LLM)中目标导向提示工程的重要性。通过对35个代表性研究的回顾，我们发现引导LLM遵循人类的逻辑思维的目标导向提示公式显著提高了LLM的性能。我们还提出了一个新的分类体系，并总结了十个适用任务来展示我们框架的广泛适用性。同时，我们提出了四个未来的方向，以推动目标导向提示工程的进一步发展。 |
| [^8] | [Large language models can replicate cross-cultural differences in personality.](http://arxiv.org/abs/2310.10679) | 大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。 |
| [^9] | [Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models.](http://arxiv.org/abs/2308.16463) | Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。 |
| [^10] | [A new mapping of technological interdependence.](http://arxiv.org/abs/2308.00014) | 本文利用文本挖掘和网络分析的方法，研究了不同部门之间的技术相互依赖关系，并证明了在技术创新中，间接联系和直接联系同等重要。 |

# 详细

[^1]: Invalsi基准：衡量语言模型在意大利语的数学和语言理解能力

    The Invalsi Benchmark: measuring Language Models Mathematical and Language understanding in Italian

    [https://arxiv.org/abs/2403.18697](https://arxiv.org/abs/2403.18697)

    该研究提出了两个新的基准，用于评估语言模型在意大利语的数学和语言理解能力，为当前语言模型的性能提供了具有挑战性的评估标准。

    

    尽管意大利语在所有指标上都是一种高资源语言，但目前并没有一种专门针对该语言进行预训练的语言模型。这导致了可用于评估意大利语语言模型性能的基准数目较少。本文提出了两个新的基准，用于评估模型在意大利语的数学理解和语言理解方面的性能。这些基准基于意大利学校系统内11至18岁学生进行的实际测试，并已由多位教学和教育学专家验证。

    arXiv:2403.18697v1 Announce Type: new  Abstract: While Italian is by all metrics a high resource language, currently, there are isn't a Language Model pre-trained exclusively in this language. This results in a lower number of available benchmarks to evaluate the performance of language models in Italian.   This work presents two new benchmarks to evaluate the models performance on mathematical understanding and language understanding in Italian. These benchmarks are based on real tests that are undertaken by students of age between 11 and 18 within the Italian school system and have therefore been validated by several experts in didactics and pedagogy.   To validate this dataset we evaluate the performance of 9 language models that are the best performing when writing in Italian, including our own fine-tuned models. We show that this is a challenging benchmark where current language models are bound by 60\% accuracy.   We believe that the release of this dataset paves the way for impr
    
[^2]: 数据的时效性：在大型语言模型中追踪知识截止日期

    Dated Data: Tracing Knowledge Cutoffs in Large Language Models

    [https://arxiv.org/abs/2403.12958](https://arxiv.org/abs/2403.12958)

    本文提出了在大型语言模型中追踪知识截止日期的概念，通过资源级别的时间对齐性估计有效截止日期，并发现这些截止日期通常与报道的不同。

    

    发布的大型语言模型通常配有声称的知识截止日期，即获取训练数据的日期。这些信息对于需要语言模型提供最新信息的应用至关重要。然而，这一说法只是表面现象：训练数据中的所有资源是否都具有相同的知识截止日期？模型对这些子集的展示知识是否与它们的截止日期密切相关？在这项工作中，我们定义了有效截止日期的概念。这与语言模型设计者报告的截止日期不同，分别适用于子资源和主题。我们提出了一种简单的方法，通过探测数据版本之间的时间对齐性来估计语言模型在资源级别的有效截止日期。通过这项分析，我们发现有效截止日期通常与报告的截止日期不同。为了了解这一观察结果的根本原因，我们进行了直接的大规模分析。

    arXiv:2403.12958v1 Announce Type: new  Abstract: Released Large Language Models (LLMs) are often paired with a claimed knowledge cutoff date, or the dates at which training data was gathered. Such information is crucial for applications where the LLM must provide up to date information. However, this statement only scratches the surface: do all resources in the training data share the same knowledge cutoff date? Does the model's demonstrated knowledge for these subsets closely align to their cutoff dates? In this work, we define the notion of an effective cutoff. This is distinct from the LLM designer reported cutoff and applies separately to sub-resources and topics. We propose a simple approach to estimate effective cutoffs on the resource-level temporal alignment of an LLM by probing across versions of the data. Using this analysis, we find that effective cutoffs often differ from reported cutoffs. To understand the root cause of this observation, we conduct a direct large-scale ana
    
[^3]: 数据增强已死，数据增强万岁

    Data Augmentation is Dead, Long Live Data Augmentation

    [https://arxiv.org/abs/2402.14895](https://arxiv.org/abs/2402.14895)

    数据增强不过是更好地微调模型，零唁态和少样本数据生成可提高性能

    

    文本数据增强（DA）是一个繁荣的研究领域，不断提出新颖的技术来创建人工数据，已经在小数据环境中表现出很高的效率，至少对于文本分类任务而言。在本文中，我们质疑这些结果，表明经典的数据增强只是一种更好地进行微调的方式，并且在应用数据增强之前花更多时间进行微调会抵消其效果。这是一个重要的贡献，因为它回答了最近几年留下的几个问题，即：哪种DA技术表现最佳（只要它们生成的数据与训练集足够接近，不会损害训练），为什么DA表现出积极的结果（简化网络训练）。此外，我们还展示了通过对话代理（如ChatGPT或LLama2）零唁态和少样本数据生成可以提高性能，从而得出了结论，此法可以提高模型性能。

    arXiv:2402.14895v1 Announce Type: cross  Abstract: Textual data augmentation (DA) is a prolific field of study where novel techniques to create artificial data are regularly proposed, and that has demonstrated great efficiency on small data settings, at least for text classification tasks. In this paper, we challenge those results, showing that classical data augmentation is simply a way of performing better fine-tuning, and that spending more time fine-tuning before applying data augmentation negates its effect. This is a significant contribution as it answers several questions that were left open in recent years, namely~: which DA technique performs best (all of them as long as they generate data close enough to the training set as to not impair training) and why did DA show positive results (facilitates training of network). We furthermore show that zero and few-shot data generation via conversational agents such as ChatGPT or LLama2 can increase performances, concluding that this f
    
[^4]: COBIAS：偏见评估中的情境可靠性

    COBIAS: Contextual Reliability in Bias Assessment

    [https://arxiv.org/abs/2402.14889](https://arxiv.org/abs/2402.14889)

    我们提出了COBIAS，旨在通过考虑多样情境的用户输入内容，衡量语句的情境可靠性，从而培养偏见意识。

    

    大型语言模型（LLMs）是基于固有偏见数据训练的。以往的去偏见模型研究依赖基准数据集来衡量模型性能。然而，这些数据集由于对偏见的极其主观理解而存在多个缺陷，凸显出对情境探索的迫切需求。我们提出考虑输入用户内容的情境，考虑到输入语句可能存在的多种情况。这种方法将允许培养偏见意识的框架，而不是伤害用户参与的防护设施。我们的贡献有两个方面：(i) 我们创建了一个包含2287个陈词滥调语句以及添加情境要点的数据集；(ii) 我们开发了面向情境的偏见指标和评估分数（COBIAS）来评估语句在衡量偏见方面的情境可靠性。我们的度量是衡量偏见基准数据集情境可靠性的重要预测因子。

    arXiv:2402.14889v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are trained on inherently biased data. Previous works on debiasing models rely on benchmark datasets to measure model performance. However, these datasets suffer from several pitfalls due to the extremely subjective understanding of bias, highlighting a critical need for contextual exploration. We propose understanding the context of user inputs with consideration of the diverse situations in which input statements are possible. This approach would allow for frameworks that foster bias awareness rather than guardrails that hurt user engagement. Our contribution is twofold: (i) we create a dataset of 2287 stereotyped statements augmented with points for adding context; (ii) we develop the Context-Oriented Bias Indicator and Assessment Score (COBIAS) to assess statements' contextual reliability in measuring bias. Our metric is a significant predictor of the contextual reliability of bias-benchmark datasets ($
    
[^5]: 开放域文本到SQL的多跳表检索

    Multi-Hop Table Retrieval for Open-Domain Text-to-SQL

    [https://arxiv.org/abs/2402.10666](https://arxiv.org/abs/2402.10666)

    提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果

    

    开放域文本到SQL是一个重要任务，它从庞大的数据库中检索与问题相关的表，然后生成SQL。然而，现有的单跳检索方法并未关注文本到SQL挑战中的模式链接，这涉及到将问题中的实体与表中实体对齐，主要体现在两个方面：相似的无关实体和领域不匹配实体。因此，我们提出了我们的方法，即带重写和波束搜索的多跳表检索（Murre）。为了减少相似的无关实体的影响，我们的方法侧重于每个跳跃中未检索到的实体，并通过波束搜索考虑排名较低的表。为了缓解领域不匹配实体的限制，Murre基于多个跳跃中检索到的表重写问题，减少与相关表的领域差距。我们在SpiderUnion和BirdUnion+上进行实验，取得了新的最先进结果。

    arXiv:2402.10666v1 Announce Type: new  Abstract: Open-domain text-to-SQL is an important task that retrieves question-relevant tables from massive databases and then generates SQL. However, existing retrieval methods that retrieve in a single hop do not pay attention to the text-to-SQL challenge of schema linking, which is aligning the entities in the question with table entities, reflected in two aspects: similar irrelevant entity and domain mismatch entity. Therefore, we propose our method, the multi-hop table retrieval with rewrite and beam search (Murre). To reduce the effect of the similar irrelevant entity, our method focuses on unretrieved entities at each hop and considers the low-ranked tables by beam search. To alleviate the limitation of domain mismatch entity, Murre rewrites the question based on retrieved tables in multiple hops, decreasing the domain gap with relevant tables. We conduct experiments on SpiderUnion and BirdUnion+, reaching new state-of-the-art results with 
    
[^6]: 大型语言模型在数学推理中的应用：进展与挑战

    Large Language Models for Mathematical Reasoning: Progresses and Challenges

    [https://arxiv.org/abs/2402.00157](https://arxiv.org/abs/2402.00157)

    大型语言模型(LLMs)在解决数学问题方面涉及了大量的数学问题类型和不同的数据集和设置。目前仍然存在一些挑战，需要进一步研究和解决。

    

    数学推理是评估人类智能基本认知能力的基石。近年来，大型语言模型（LLMs）的发展引起了人们对自动解决数学问题的重视。然而，数学问题的类型非常广泛，LLM相关技术在不同数据集和设置下进行评估，使得如何判断这一新兴领域中的真正进展和障碍变得困难。本调查研究包括了以下四个关键方面：i）全面探索各种已经研究的数学问题及其相应数据集；ii）研究提出的解决数学问题的LLM技术的范围；iii）概述影响LLM在解决数学问题中的因素和关注点；iv）阐明仍然存在的挑战。

    Mathematical reasoning serves as a cornerstone for assessing the fundamental cognitive capabilities of human intelligence. In recent times, there has been a notable surge in the development of Large Language Models (LLMs) geared towards the automated resolution of mathematical problems. However, the landscape of mathematical problem types is vast and varied, with LLM-oriented techniques undergoing evaluation across diverse datasets and settings. This diversity makes it challenging to discern the true advancements and obstacles within this burgeoning field. This survey endeavors to address four pivotal dimensions: i) a comprehensive exploration of the various mathematical problems and their corresponding datasets that have been investigated; ii) an examination of the spectrum of LLM-oriented techniques that have been proposed for mathematical problem-solving; iii) an overview of factors and concerns affecting LLMs in solving math; and iv) an elucidation of the persisting challenges with
    
[^7]: 朝着目标导向的大型语言模型提示方法：一项调查

    Towards Goal-oriented Large Language Model Prompting: A Survey. (arXiv:2401.14043v1 [cs.CL])

    [http://arxiv.org/abs/2401.14043](http://arxiv.org/abs/2401.14043)

    本文调查了大型语言模型(LLM)中目标导向提示工程的重要性。通过对35个代表性研究的回顾，我们发现引导LLM遵循人类的逻辑思维的目标导向提示公式显著提高了LLM的性能。我们还提出了一个新的分类体系，并总结了十个适用任务来展示我们框架的广泛适用性。同时，我们提出了四个未来的方向，以推动目标导向提示工程的进一步发展。

    

    大型语言模型(LLM)在各种下游任务中显示出卓越的性能，而提示工程在优化LLM性能中起着关键作用。本文旨在强调设计提示的限制，同时保持人类追求LLM像人类思考的人类学假设。通过对35个代表性研究的回顾，我们展示了目标导向提示公式的重要性，该公式指导LLM遵循人类的逻辑思维，显著提高了LLM的性能。此外，我们引入了一个新的分类体系，将目标导向提示方法分为五个相互关联的阶段，并通过总结十个适用任务来展示我们框架的广泛适用性。最后，我们提出了四个未来的方向，希望进一步强调和推动目标导向提示工程。

    Large Language Models (LLMs) have shown prominent performance in various downstream tasks in which prompt engineering plays a pivotal role in optimizing LLMs' performance. This paper, not as an overview of current prompt engineering methods, aims to highlight the limitation of designing prompts while holding an anthropomorphic assumption that expects LLMs to think like humans. From our review of 35 representative studies, we demonstrate that a goal-oriented prompt formulation, which guides LLMs to follow established human logical thinking, significantly improves the performance of LLMs. Furthermore, We introduce a novel taxonomy that categorizes goal-oriented prompting methods into five interconnected stages and we demonstrate the broad applicability of our framework by summarizing ten applicable tasks. With four future directions proposed, we hope to further emphasize and promote goal-oriented prompt engineering.
    
[^8]: 大型语言模型可以复制跨文化个性差异

    Large language models can replicate cross-cultural differences in personality. (arXiv:2310.10679v1 [cs.CL])

    [http://arxiv.org/abs/2310.10679](http://arxiv.org/abs/2310.10679)

    大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。

    

    我们使用一项大规模实验(N=8000)来确定GPT-4是否可以复制使用十项人格问卷测量的大五人格的跨文化差异。我们选择美国和韩国作为文化对比，因为先前的研究表明这两个国家的人之间存在显著的人格差异。我们操纵了模拟的目标（美国 vs. 韩国），问卷的语言（英语 vs. 韩语）以及语言模型（GPT-4 vs. GPT-3.5）。我们的结果表明，GPT-4复制了每个因子的跨文化差异。然而，平均评级具有上升偏差，并且比人类样本的变异性更低，以及结构效度较低。总的来说，我们提供了初步的证据说明LLMs可以促进跨文化心理研究。

    We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. Overall, we provide preliminary evidence that LLMs can aid cross-cultural psychological research.
    
[^9]: Sparkles: 解锁多图聊天以实现多模态指令跟踪模型

    Sparkles: Unlocking Chats Across Multiple Images for Multimodal Instruction-Following Models. (arXiv:2308.16463v1 [cs.CV])

    [http://arxiv.org/abs/2308.16463](http://arxiv.org/abs/2308.16463)

    Sparkles是一个多模态指令跟踪模型，通过整合文本和图像实现多图对话。我们引入了SparklesDialogue数据集和SparklesEval基准来支持训练和评估。实验证实了SparklesChat在理解多图对话方面的有效性。

    

    当使用指令跟踪数据来进行微调时，大型语言模型在各种任务上展现出了强大的零-shot性能。多模态指令跟踪模型通过整合文本和图像进一步扩展了这些能力。然而，现有的模型（如MiniGPT-4）在涉及多个图像的情况下保持对话连贯性面临挑战。一个主要原因是缺乏一个专门针对这一关键应用的数据集。为了弥合这些差距，我们提出了SparklesChat，一个用于多图对话的多模态指令跟踪模型。为了支持训练，我们引入了SparklesDialogue，这是第一个专为单词级交错多图像和文本交互而定制的机器生成对话数据集。此外，我们构建了SparklesEval，一个借助GPT辅助的基准，用于定量评估模型在多个图像和对话轮次中的对话能力。我们的实验验证了SparklesChat在理解多图对话方面的有效性。

    Large language models exhibit enhanced zero-shot performance on various tasks when fine-tuned with instruction-following data. Multimodal instruction-following models extend these capabilities by integrating both text and images. However, existing models such as MiniGPT-4 face challenges in maintaining dialogue coherence in scenarios involving multiple images. A primary reason is the lack of a specialized dataset for this critical application. To bridge these gaps, we present SparklesChat, a multimodal instruction-following model for open-ended dialogues across multiple images. To support the training, we introduce SparklesDialogue, the first machine-generated dialogue dataset tailored for word-level interleaved multi-image and text interactions. Furthermore, we construct SparklesEval, a GPT-assisted benchmark for quantitatively assessing a model's conversational competence across multiple images and dialogue turns. Our experiments validate the effectiveness of SparklesChat in understa
    
[^10]: 一种新的技术相互依赖的映射

    A new mapping of technological interdependence. (arXiv:2308.00014v1 [econ.EM])

    [http://arxiv.org/abs/2308.00014](http://arxiv.org/abs/2308.00014)

    本文利用文本挖掘和网络分析的方法，研究了不同部门之间的技术相互依赖关系，并证明了在技术创新中，间接联系和直接联系同等重要。

    

    哪些技术联系影响了部门的创新能力？这些效应如何通过技术空间传递？本文使用新颖的文本挖掘和网络分析方法回答了这两个关键问题。我们通过分析美国专利商标局（USPTO）授予的650万项专利的文本，并应用网络分析方法，研究了半个世纪（从1976年到2021年）期间不同部门之间的技术相互依赖关系，揭示了存在于技术领域之间的全谱的联系。我们证明专利文本包含了往往无法通过传统的创新指标（例如专利引用）捕捉到的丰富信息。通过使用网络分析，我们记录了间接联系和直接联系同等重要，并且前者大部分使用传统的间接联系度量方法（如Leontief逆矩阵）往往会被隐藏。最后，基于冲击响应分析，我们进行了说明。

    Which technological linkages affect the sector's ability to innovate? How do these effects transmit through the technology space? This paper answers these two key questions using novel methods of text mining and network analysis. We examine technological interdependence across sectors over a period of half a century (from 1976 to 2021) by analyzing the text of 6.5 million patents granted by the United States Patent and Trademark Office (USPTO), and applying network analysis to uncover the full spectrum of linkages existing across technology areas. We demonstrate that patent text contains a wealth of information often not captured by traditional innovation metrics, such as patent citations. By using network analysis, we document that indirect linkages are as important as direct connections and that the former would remain mostly hidden using more traditional measures of indirect linkages, such as the Leontief inverse matrix. Finally, based on an impulse-response analysis, we illustrate 
    

