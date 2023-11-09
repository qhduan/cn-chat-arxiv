# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Large Language Model for Recommender Systems.](http://arxiv.org/abs/2311.01343) | 本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。 |
| [^2] | [Question Answering for Electronic Health Records: A Scoping Review of datasets and models.](http://arxiv.org/abs/2310.08759) | 本文对电子健康记录（EHR）中的问题回答进行了范围回顾。与其他医学QA任务不同，EHR QA通过从患者的医疗记录中获取答案。这项研究为现有的EHR QA作品提供了方法论回顾。 |
| [^3] | [Linear Recurrent Units for Sequential Recommendation.](http://arxiv.org/abs/2310.02367) | 本研究提出了一种用于顺序推荐的线性循环单元（LRURec）。与当前的自注意模型相比，LRURec具有快速的推断速度、能够进行增量推断、更小的模型大小和可并行训练。通过优化架构并引入非线性，LRURec在顺序推荐任务中取得了有效的结果。 |
| [^4] | [Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search.](http://arxiv.org/abs/2308.01098) | 本文提出了一种知识蒸馏框架（KC），通过在严格的低延迟约束下提升在线FastText模型的查询分类性能，在京东广告搜索中取得了显著的性能提升。 |
| [^5] | [Machine Reading Comprehension using Case-based Reasoning.](http://arxiv.org/abs/2305.14815) | 本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。 |

# 详细

[^1]: 协同大型语言模型用于推荐系统

    Collaborative Large Language Model for Recommender Systems. (arXiv:2311.01343v1 [cs.IR])

    [http://arxiv.org/abs/2311.01343](http://arxiv.org/abs/2311.01343)

    本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。

    

    最近，越来越多的人对基于预训练的大型语言模型（LLM）开发下一代推荐系统（RS）产生了兴趣，充分利用其编码知识和推理能力。然而，自然语言与推荐任务之间的语义差距仍未得到很好的解决，导致一些问题，如虚假相关的用户/项目描述符、对用户/项目内容的低效语言建模以及通过自动回归进行低效的推荐等。在本文中，我们提出了CLLM4Rec，这是第一个紧密集成LLM范式和RS的ID范式的生成RS，旨在同时解决上述挑战。我们首先使用用户/项目ID标记扩展了预训练LLM的词汇表，以忠实地模拟用户/项目的协同和内容语义。因此，在预训练阶段，提出了一种新颖的软硬提示策略，通过语言建模有效地学习用户/项目的协同/内容标记嵌入。

    Recently, there is a growing interest in developing next-generation recommender systems (RSs) based on pretrained large language models (LLMs), fully utilizing their encoded knowledge and reasoning ability. However, the semantic gap between natural language and recommendation tasks is still not well addressed, leading to multiple issues such as spuriously-correlated user/item descriptors, ineffective language modeling on user/item contents, and inefficient recommendations via auto-regression, etc. In this paper, we propose CLLM4Rec, the first generative RS that tightly integrates the LLM paradigm and ID paradigm of RS, aiming to address the above challenges simultaneously. We first extend the vocabulary of pretrained LLMs with user/item ID tokens to faithfully model the user/item collaborative and content semantics. Accordingly, in the pretraining stage, a novel soft+hard prompting strategy is proposed to effectively learn user/item collaborative/content token embeddings via language m
    
[^2]: 电子健康记录的问题回答：数据集和模型的范围回顾

    Question Answering for Electronic Health Records: A Scoping Review of datasets and models. (arXiv:2310.08759v1 [cs.LG])

    [http://arxiv.org/abs/2310.08759](http://arxiv.org/abs/2310.08759)

    本文对电子健康记录（EHR）中的问题回答进行了范围回顾。与其他医学QA任务不同，EHR QA通过从患者的医疗记录中获取答案。这项研究为现有的EHR QA作品提供了方法论回顾。

    

    与患者相关的问题回答（QA）系统可以帮助临床医生和患者。它们可以帮助临床医生做决策，并使患者更好地了解他们的病历。大量的患者数据存储在电子健康记录（EHR）中，使得EHR QA成为一个重要的研究领域。在EHR QA中，答案是从患者的医疗记录中获得的。由于数据格式和模式的差异，这与其他使用医学网站或科学论文检索答案的医学QA任务有很大的不同，这使得研究EHR问题回答变得至关重要。本研究旨在对现有关于EHR QA的作品进行方法论回顾。我们在包括Google Scholar、ACL Anthology、ACM Digital Library和PubMed在内的四个数字资源中搜索了从2005年1月1日到2023年9月30日的文章，以收集有关EHR QA的相关出版物。共发现了4111篇论文。

    Question Answering (QA) systems on patient-related data can assist both clinicians and patients. They can, for example, assist clinicians in decision-making and enable patients to have a better understanding of their medical history. Significant amounts of patient data are stored in Electronic Health Records (EHRs), making EHR QA an important research area. In EHR QA, the answer is obtained from the medical record of the patient. Because of the differences in data format and modality, this differs greatly from other medical QA tasks that employ medical websites or scientific papers to retrieve answers, making it critical to research EHR question answering. This study aimed to provide a methodological review of existing works on QA over EHRs. We searched for articles from January 1st, 2005 to September 30th, 2023 in four digital sources including Google Scholar, ACL Anthology, ACM Digital Library, and PubMed to collect relevant publications on EHR QA. 4111 papers were identified for our
    
[^3]: 用于顺序推荐的线性循环单元

    Linear Recurrent Units for Sequential Recommendation. (arXiv:2310.02367v1 [cs.IR])

    [http://arxiv.org/abs/2310.02367](http://arxiv.org/abs/2310.02367)

    本研究提出了一种用于顺序推荐的线性循环单元（LRURec）。与当前的自注意模型相比，LRURec具有快速的推断速度、能够进行增量推断、更小的模型大小和可并行训练。通过优化架构并引入非线性，LRURec在顺序推荐任务中取得了有效的结果。

    

    当前的顺序推荐依赖于基于自注意的推荐模型。然而，这些模型计算代价高，往往对实时推荐来说过于缓慢。此外，自注意操作是在序列层级上进行的，因此对于低成本的增量推断来说具有挑战性。受到高效语言建模的最新进展的启发，我们提出了用于顺序推荐的线性循环单元（LRURec）。类似于循环神经网络，LRURec具有快速的推断速度，并且能够对顺序输入进行增量推断。通过分解线性循环操作并在我们的框架中设计递归并行化，LRURec提供了减小模型大小和可并行训练的额外优势。此外，我们通过实施一系列修改来优化LRURec的架构，以解决缺乏非线性和改善训练动态的问题。为了验证我们提出的LRURec的有效性

    State-of-the-art sequential recommendation relies heavily on self-attention-based recommender models. Yet such models are computationally expensive and often too slow for real-time recommendation. Furthermore, the self-attention operation is performed at a sequence-level, thereby making low-cost incremental inference challenging. Inspired by recent advances in efficient language modeling, we propose linear recurrent units for sequential recommendation (LRURec). Similar to recurrent neural networks, LRURec offers rapid inference and can achieve incremental inference on sequential inputs. By decomposing the linear recurrence operation and designing recursive parallelization in our framework, LRURec provides the additional benefits of reduced model size and parallelizable training. Moreover, we optimize the architecture of LRURec by implementing a series of modifications to address the lack of non-linearity and improve training dynamics. To validate the effectiveness of our proposed LRURe
    
[^4]: 在京东广告搜索中利用多专家知识蒸馏实现更好的查询分类

    Towards Better Query Classification with Multi-Expert Knowledge Condensation in JD Ads Search. (arXiv:2308.01098v1 [cs.IR])

    [http://arxiv.org/abs/2308.01098](http://arxiv.org/abs/2308.01098)

    本文提出了一种知识蒸馏框架（KC），通过在严格的低延迟约束下提升在线FastText模型的查询分类性能，在京东广告搜索中取得了显著的性能提升。

    

    查询分类作为理解用户意图的有效方法，在现实世界的在线广告系统中具有重要意义。为了确保更低的延迟，常使用浅层模型（如FastText）进行高效的在线推断。然而，FastText模型的表征能力不足，导致分类性能较差，特别是在一些低频查询和尾部类别上。使用更深入且更复杂的模型（如BERT）是一种有效的解决方案，但它将导致更高的在线推断延迟和更昂贵的计算成本。因此，如何在推断效率和分类性能之间折衷显然具有重大实际意义。为了克服这个挑战，在本文中，我们提出了知识蒸馏（KC），一个简单而有效的知识蒸馏框架，以在严格的低延迟约束下提升在线FastText模型的分类性能。具体来说，我们提出了训练一个离线模型，通过蒸馏知识来改善在线模型的分类性能。

    Search query classification, as an effective way to understand user intents, is of great importance in real-world online ads systems. To ensure a lower latency, a shallow model (e.g. FastText) is widely used for efficient online inference. However, the representation ability of the FastText model is insufficient, resulting in poor classification performance, especially on some low-frequency queries and tailed categories. Using a deeper and more complex model (e.g. BERT) is an effective solution, but it will cause a higher online inference latency and more expensive computing costs. Thus, how to juggle both inference efficiency and classification performance is obviously of great practical importance. To overcome this challenge, in this paper, we propose knowledge condensation (KC), a simple yet effective knowledge distillation framework to boost the classification performance of the online FastText model under strict low latency constraints. Specifically, we propose to train an offline
    
[^5]: 使用基于案例推理的机器阅读理解

    Machine Reading Comprehension using Case-based Reasoning. (arXiv:2305.14815v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14815](http://arxiv.org/abs/2305.14815)

    本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。

    

    我们提出了一种准确且可解释的方法，用于机器阅读理解中的答案提取，该方法类似于经典人工智能中的基于案例推理（CBR）。我们的方法（CBR-MRC）基于一个假设，即相似问题的上下文化答案彼此之间具有语义相似性。给定一个测试问题，CBR-MRC首先从非参数化存储器中检索一组相似的案例，然后通过选择测试上下文中最类似于检索到的案例中上下文化答案表示的范围来预测答案。我们的方法半参数化的特性使其能够将预测归因于特定的证据案例集，因此在构建可靠且可调试的问答系统时是一个理想的选择。我们展示了CBR-MRC在自然语言问题（NaturalQuestions）和新闻问答（NewsQA）上比大型读者模型提供了高准确性，并且优于基准分别提升了11.5和8.4 EM。此外，我们还展示了CBR-MRC在识别与他人评估员不同的答案方面的能力。

    We present an accurate and interpretable method for answer extraction in machine reading comprehension that is reminiscent of case-based reasoning (CBR) from classical AI. Our method (CBR-MRC) builds upon the hypothesis that contextualized answers to similar questions share semantic similarities with each other. Given a test question, CBR-MRC first retrieves a set of similar cases from a non-parametric memory and then predicts an answer by selecting the span in the test context that is most similar to the contextualized representations of answers in the retrieved cases. The semi-parametric nature of our approach allows it to attribute a prediction to the specific set of evidence cases, making it a desirable choice for building reliable and debuggable QA systems. We show that CBR-MRC provides high accuracy comparable with large reader models and outperforms baselines by 11.5 and 8.4 EM on NaturalQuestions and NewsQA, respectively. Further, we demonstrate the ability of CBR-MRC in identi
    

