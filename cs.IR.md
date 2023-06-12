# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Can Recommender Systems Benefit from Large Language Models: A Survey.](http://arxiv.org/abs/2306.05817) | 本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。 |
| [^2] | [Interactive Explanation with Varying Level of Details in an Explainable Scientific Literature Recommender System.](http://arxiv.org/abs/2306.05809) | 本文旨在采用以用户为中心的交互式解释模型，在推荐系统中为用户提供不同细节级别的解释，赋予用户个性化解释的能力。 |
| [^3] | [RankFormer: Listwise Learning-to-Rank Using Listwide Labels.](http://arxiv.org/abs/2306.05808) | RankFormer是一个可以利用列表标签进行列表学习排序的架构，并在多个最先进的LTR方法上展现了更好的表现。 |
| [^4] | [Customizing General-Purpose Foundation Models for Medical Report Generation.](http://arxiv.org/abs/2306.05642) | 这项工作中，我们提出了一种自定义通用基础模型以用于医疗报告生成的方法，其利用轻量级查询Transformer连接两个FMs，并在三个基准数据集上实现了最先进的结果。 |
| [^5] | [Bayesian Knowledge-driven Critiquing with Indirect Evidence.](http://arxiv.org/abs/2306.05636) | 本文提出了一个基于知识图谱的贝叶斯批判式推荐系统，将直接和间接的物品属性结合，允许用户提供更加复杂的、基于知识的反馈。 |
| [^6] | [Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data.](http://arxiv.org/abs/2306.05535) | 政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。 |
| [^7] | [CLC: Cluster Assignment via Contrastive Representation Learning.](http://arxiv.org/abs/2306.05439) | 本文提出了一种基于对比学习的聚类方法（CLC），它使用对比学习直接学习聚类分配，并在大规模数据集上取得了更好的聚类性能。 |
| [^8] | [Towards Alleviating the Object Bias in Prompt Tuning-based Factual Knowledge Extraction.](http://arxiv.org/abs/2306.03378) | 本文提出了一种名为MeCoD的提示调整方法，通过提示编码器、对象均衡和有偏对象阻塞三个模块，有效减少了对象偏差，提高了事实知识的提取准确性。 |
| [^9] | [Generative Flow Network for Listwise Recommendation.](http://arxiv.org/abs/2306.02239) | 本文提出了生成流网络用于列表化推荐的解决方案GFN4Rec，通过生成流网络和列表变换器的强大建模能力，生成具有高质量和多样性的项目列表，实验证明其在推荐质量和多样性方面优于现有方法。 |
| [^10] | [How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?.](http://arxiv.org/abs/2305.01555) | 本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。 |
| [^11] | [Reinforcement Re-ranking with 2D Grid-based Recommendation Panels.](http://arxiv.org/abs/2204.04954) | 该论文提出了一种名为Panel-MDP的新型模型，通过采用强化学习策略，以用户喜好为导向，能够有效解决网格面板排列物品的问题，提高用户体验。 |
| [^12] | [A Systematic Review of Automated Query Reformulations in Source Code Search.](http://arxiv.org/abs/2108.09646) | 许多研究尝试重新制定即席查询以支持开发人员进行代码搜索，本文针对70个主要查询再制定研究进行了细致筛选和深入的定性分析，提出了八种主要方法。 |

# 详细

[^1]: 推荐系统如何从大型语言模型中受益：一项调查研究

    How Can Recommender Systems Benefit from Large Language Models: A Survey. (arXiv:2306.05817v1 [cs.IR])

    [http://arxiv.org/abs/2306.05817](http://arxiv.org/abs/2306.05817)

    本文对将大型语言模型（LLM）应用于推荐系统进行了全面的调查研究，从两个角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。最后，我们提出了一些潜在的研究方向和挑战。

    

    推荐系统在匹配互联网应用程序用户的信息需求方面发挥着重要作用。在自然语言处理领域中，大型语言模型已经展现出了惊人的新兴能力（例如指令跟踪、推理），从而为将LLM调整到推荐系统中以提高性能和改善用户体验的研究方向带来了希望。在本文中，我们从应用导向的角度对此研究方向进行了全面的调查。我们首先从两个正交的角度总结了现有的研究工作：如何在推荐系统中调整LLM和调整LLM时在哪里调整。对于“在哪里”这个问题，我们讨论了LLM在推荐流程的不同阶段中可能发挥的作用，即特征工程、特征编码器、评分/排名函数和流程控制器。对于“如何”这个问题，我们调查了训练和推理策略，从而得出两个细粒度的分类标准，即是否调整LLM和是否将LLM作为独立模型或混合模型组件使用。最后，我们提出了在将LLM调整到RS中的一些挑战和潜在方向，包括与现有系统的集成、用户反馈、评估度量和知识蒸馏。

    Recommender systems (RS) play important roles to match users' information needs for Internet applications. In natural language processing (NLP) domains, large language model (LLM) has shown astonishing emergent abilities (e.g., instruction following, reasoning), thus giving rise to the promising research direction of adapting LLM to RS for performance enhancements and user experience improvements. In this paper, we conduct a comprehensive survey on this research direction from an application-oriented view. We first summarize existing research works from two orthogonal perspectives: where and how to adapt LLM to RS. For the "WHERE" question, we discuss the roles that LLM could play in different stages of the recommendation pipeline, i.e., feature engineering, feature encoder, scoring/ranking function, and pipeline controller. For the "HOW" question, we investigate the training and inference strategies, resulting in two fine-grained taxonomy criteria, i.e., whether to tune LLMs or not, a
    
[^2]: 在可解释的科学文献推荐系统中采用不同细节级别的交互式解释

    Interactive Explanation with Varying Level of Details in an Explainable Scientific Literature Recommender System. (arXiv:2306.05809v1 [cs.IR])

    [http://arxiv.org/abs/2306.05809](http://arxiv.org/abs/2306.05809)

    本文旨在采用以用户为中心的交互式解释模型，在推荐系统中为用户提供不同细节级别的解释，赋予用户个性化解释的能力。

    

    传统上，可解释的推荐系统采用一种“一刀切”的方法，向每个用户提供相同程度的解释，而不考虑他们的个体需求和目标。此外，推荐系统中的解释大多以静态和非交互方式呈现。为填补这些研究空白，本文旨在采用以用户为中心的交互式解释模型，为用户提供不同细节级别的解释，并赋予用户基于其需求和偏好进行交互、控制和个性化解释的能力。我们采用以用户为中心的方法，设计了三个细节级别的交互式解释（基本、中级和高级），并在透明的推荐和兴趣建模应用（RIMA）中实现了它们。我们进行了一个定性用户研究（N=14），以调查提供不同细节级别的交互式解释对用户对系统可解释性的感知的影响。

    Explainable recommender systems (RS) have traditionally followed a one-size-fits-all approach, delivering the same explanation level of detail to each user, without considering their individual needs and goals. Further, explanations in RS have so far been presented mostly in a static and non-interactive manner. To fill these research gaps, we aim in this paper to adopt a user-centered, interactive explanation model that provides explanations with different levels of detail and empowers users to interact with, control, and personalize the explanations based on their needs and preferences. We followed a user-centered approach to design interactive explanations with three levels of detail (basic, intermediate, and advanced) and implemented them in the transparent Recommendation and Interest Modeling Application (RIMA). We conducted a qualitative user study (N=14) to investigate the impact of providing interactive explanations with varying level of details on the users' perception of the e
    
[^3]: RankFormer：使用列表标签的列表学习排序

    RankFormer: Listwise Learning-to-Rank Using Listwide Labels. (arXiv:2306.05808v1 [cs.IR])

    [http://arxiv.org/abs/2306.05808](http://arxiv.org/abs/2306.05808)

    RankFormer是一个可以利用列表标签进行列表学习排序的架构，并在多个最先进的LTR方法上展现了更好的表现。

    

    网络应用程序常常使用排序模型将最相关的结果排在前面，以呈现给用户有限的选择。通常假定从用户获得的反馈只反映了项目效用的相对评价，例如用户单击项目只暗示它比在同一排序列表中未单击的项目更好。因此，学习排序（LTR）中优化的目标往往是成对或按列表排序。然而，只看到相对反馈，我们忽视了用户对列表整体质量的绝对反馈，例如当选择中没有项目被单击时。因此，我们重新考虑了标准的LTR范式，并论述了从这种列表级信号中学习的好处。为此，我们提出了RankFormer作为一个带有Transformer核心的架构，可以共同优化新的列表评估目标和传统的按列表LTR目标。我们在公共数据集上模拟隐式反馈，并观察到RankFormer比几种最先进的LTR方法表现更好，从而证明了利用列表标签的有效性。

    Web applications where users are presented with a limited selection of items have long employed ranking models to put the most relevant results first. Any feedback received from users is typically assumed to reflect a relative judgement on the utility of items, e.g. a user clicking on an item only implies it is better than items not clicked in the same ranked list. Hence, the objectives optimized in Learning-to-Rank (LTR) tend to be pairwise or listwise.  Yet, by only viewing feedback as relative, we neglect the user's absolute feedback on the list's overall quality, e.g. when no items in the selection are clicked. We thus reconsider the standard LTR paradigm and argue the benefits of learning from this listwide signal. To this end, we propose the RankFormer as an architecture that, with a Transformer at its core, can jointly optimize a novel listwide assessment objective and a traditional listwise LTR objective.  We simulate implicit feedback on public datasets and observe that the Ra
    
[^4]: 面向医疗报告生成的通用基础模型自定义

    Customizing General-Purpose Foundation Models for Medical Report Generation. (arXiv:2306.05642v1 [cs.CV])

    [http://arxiv.org/abs/2306.05642](http://arxiv.org/abs/2306.05642)

    这项工作中，我们提出了一种自定义通用基础模型以用于医疗报告生成的方法，其利用轻量级查询Transformer连接两个FMs，并在三个基准数据集上实现了最先进的结果。

    

    医疗字幕预测，也被视为医疗报告生成（MRG）的任务，需要为给定的医疗图像自动生成连贯准确的字幕。然而，标记的医疗图像-报告对的稀缺性在深度和大规模神经网络的开发中提出了巨大挑战，这些网络可以利用大型语言模型（LLM）这样的人工智能潜力。在这项工作中，我们建议将通用的面向计算机视觉和自然语言处理的基础模型进行定制，特别关注医疗报告生成。具体来说，我们根据BLIP-2提出了基于编码器-解码器的MRG模型，该模型利用轻量级查询Transformer连接两个FMs：巨型视觉Transformer EVA-ViT-g和双语LLM，该LLM被训练用于与人类意图对齐（称为T5-base-CN）。实验结果表明，我们提出的方法在三个医疗报告生成基准数据集上实现了最先进的结果，这表明了将基础模型适应于此任务的有效性。

    Medical caption prediction which can be regarded as a task of medical report generation (MRG), requires the automatic generation of coherent and accurate captions for the given medical images. However, the scarcity of labelled medical image-report pairs presents great challenges in the development of deep and large-scale neural networks capable of harnessing the potential artificial general intelligence power like large language models (LLMs). In this work, we propose customizing off-the-shelf general-purpose large-scale pre-trained models, i.e., foundation models (FMs), in computer vision and natural language processing with a specific focus on medical report generation. Specifically, following BLIP-2, a state-of-the-art vision-language pre-training approach, we introduce our encoder-decoder-based MRG model. This model utilizes a lightweight query Transformer to connect two FMs: the giant vision Transformer EVA-ViT-g and a bilingual LLM trained to align with human intentions (referred
    
[^5]: 基于知识图谱的贝叶斯批判式推荐系统

    Bayesian Knowledge-driven Critiquing with Indirect Evidence. (arXiv:2306.05636v1 [cs.IR])

    [http://arxiv.org/abs/2306.05636](http://arxiv.org/abs/2306.05636)

    本文提出了一个基于知识图谱的贝叶斯批判式推荐系统，将直接和间接的物品属性结合，允许用户提供更加复杂的、基于知识的反馈。

    

    会话式推荐系统增强了推荐的表现力和个性化程度，通过多轮用户-系统交互进行。批判是CRS中广为人知的一个范例，允许用户通过提供有关推荐物品的属性的反馈来逐步完善推荐。本研究利用知识图谱中有关物品的更丰富的背景信息，不仅限于利用物品的直接属性解决用户请求，而是通过采用经典的推理方法将这些信息与直接属性结合起来。因此批判式推荐系统可以实现更加复杂的基于知识的反馈，例如“我喜欢描述退伍军人战争后果的电影”。本文提出了一个基于贝叶斯知识的批判式推荐模型，该模型允许用户提供带有间接信息的反馈，同时仍具有很高的可解释性。实验结果表明，纳入间接信息显着提高了推荐的质量和模型的鲁棒性。

    Conversational recommender systems (CRS) enhance the expressivity and personalization of recommendations through multiple turns of user-system interaction. Critiquing is a well-known paradigm for CRS that allows users to iteratively refine recommendations by providing feedback about attributes of recommended items. While existing critiquing methodologies utilize direct attributes of items to address user requests such as 'I prefer Western movies', the opportunity of incorporating richer contextual and side information about items stored in Knowledge Graphs (KG) into the critiquing paradigm has been overlooked. Employing this substantial knowledge together with a well-established reasoning methodology paves the way for critique-based recommenders to allow for complex knowledge-based feedback (e.g., 'I like movies featuring war side effects on veterans') which may arise in natural user-system conversations. In this work, we aim to increase the flexibility of critique-based recommendation
    
[^6]: 使用音频数据检测政治辩论、演讲和访谈中值得核实的论断

    Detecting Check-Worthy Claims in Political Debates, Speeches, and Interviews Using Audio Data. (arXiv:2306.05535v1 [cs.CL])

    [http://arxiv.org/abs/2306.05535](http://arxiv.org/abs/2306.05535)

    政治辩论、演讲和访谈中的值得核实的论断可以使用音频数据进行检测和确认，这可帮助主持人、记者和事实核查组织进行工作。

    

    社会的一大部分团结在相同的愿景和思想周围，具有巨大的能量。这正是政治人物希望为他们的事业所累积的。为了达到这个目标，他们有时会使用扭曲或隐藏真相的手段，无论是无意的还是有意的，这为错误信息和误导开了大门。自动检测值得核实的论断的工具将对辩论主持人、记者和事实核查组织有很大帮助。虽然以前关于检测值得核实的论断的工作重点是文本，但在这里，我们探讨了音频信号作为额外信息源的实用性。我们创建了一个新的多模态数据集（英语文本和音频），包含48小时的演讲。我们的评估结果表明，在多个演讲者的情况下，音频模态与文本结合使用比仅使用文本具有改进效果。此外，单声道音频模型可以胜过单声道文本模型。

    A large portion of society united around the same vision and ideas carries enormous energy. That is precisely what political figures would like to accumulate for their cause. With this goal in mind, they can sometimes resort to distorting or hiding the truth, unintentionally or on purpose, which opens the door for misinformation and disinformation. Tools for automatic detection of check-worthy claims would be of great help to moderators of debates, journalists, and fact-checking organizations. While previous work on detecting check-worthy claims has focused on text, here we explore the utility of the audio signal as an additional information source. We create a new multimodal dataset (text and audio in English) containing 48 hours of speech. Our evaluation results show that the audio modality together with text yields improvements over text alone in the case of multiple speakers. Moreover, an audio-only model could outperform a text-only one for a single speaker.
    
[^7]: CLC: 基于对比表示学习的聚类分配方法

    CLC: Cluster Assignment via Contrastive Representation Learning. (arXiv:2306.05439v1 [cs.LG])

    [http://arxiv.org/abs/2306.05439](http://arxiv.org/abs/2306.05439)

    本文提出了一种基于对比学习的聚类方法（CLC），它使用对比学习直接学习聚类分配，并在大规模数据集上取得了更好的聚类性能。

    

    聚类是一项重要而具有挑战性的任务，旨在将样本分组，而不需要手动注释。最近的研究通过对自监督学习得到的特征表示进行聚类，在小型数据集上取得了出色的结果。然而，对于包含大量聚类的数据集，如ImageNet，当前的方法仍然无法实现高聚类性能。在本文中，我们提出了基于对比学习的聚类方法（CLC），它使用对比学习直接学习聚类分配。我们将表示分解为两部分：一部分对类别信息进行编码，并采用等分约束，另一部分捕捉实例因素。我们提出了一种对比损失，使用表示的两个部分。我们在理论上分析了所提出的对比损失，并揭示了CLC在学习聚类分配时为负样本设置不同的权重。进一步的梯度分析表明，当使用CLC时，在大规模数据集上取得了更好的聚类性能。

    Clustering remains an important and challenging task of grouping samples into clusters without manual annotations. Recent works have achieved excellent results on small datasets by performing clustering on feature representations learned from self-supervised learning. However, for datasets with a large number of clusters, such as ImageNet, current methods still can not achieve high clustering performance. In this paper, we propose Contrastive Learning-based Clustering (CLC), which uses contrastive learning to directly learn cluster assignment. We decompose the representation into two parts: one encodes the categorical information under an equipartition constraint, and the other captures the instance-wise factors. We propose a contrastive loss using both parts of the representation. We theoretically analyze the proposed contrastive loss and reveal that CLC sets different weights for the negative samples while learning cluster assignments. Further gradient analysis shows that the larger 
    
[^8]: 消除基于提示调整的事实知识提取中的对象偏差

    Towards Alleviating the Object Bias in Prompt Tuning-based Factual Knowledge Extraction. (arXiv:2306.03378v1 [cs.IR])

    [http://arxiv.org/abs/2306.03378](http://arxiv.org/abs/2306.03378)

    本文提出了一种名为MeCoD的提示调整方法，通过提示编码器、对象均衡和有偏对象阻塞三个模块，有效减少了对象偏差，提高了事实知识的提取准确性。

    

    许多工作采用提示调整方法自动优化提示查询并提取预训练语言模型中存储的事实知识。本文观察到，包括离散提示和连续提示在内的优化提示表现出不良的对象偏差。为解决这个问题，我们提出了一个新的提示调整方法MeCoD，由三个模块组成：提示编码器、对象均衡和有偏对象阻塞。实验结果表明，MeCoD可以显著减少对象偏差，同时提高事实知识的提取准确性。

    Many works employed prompt tuning methods to automatically optimize prompt queries and extract the factual knowledge stored in Pretrained Language Models. In this paper, we observe that the optimized prompts, including discrete prompts and continuous prompts, exhibit undesirable object bias. To handle this problem, we propose a novel prompt tuning method called MeCoD. consisting of three modules: Prompt Encoder, Object Equalization and Biased Object Obstruction. Experimental results show that MeCoD can significantly reduce the object bias and at the same time improve accuracy of factual knowledge extraction.
    
[^9]: 生成流网络用于列表化推荐

    Generative Flow Network for Listwise Recommendation. (arXiv:2306.02239v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02239](http://arxiv.org/abs/2306.02239)

    本文提出了生成流网络用于列表化推荐的解决方案GFN4Rec，通过生成流网络和列表变换器的强大建模能力，生成具有高质量和多样性的项目列表，实验证明其在推荐质量和多样性方面优于现有方法。

    

    个性化推荐系统能够满足用户的日常需求并促进在线业务的发展。本研究的目标是学习一种策略，能够生成符合用户需求或兴趣的项目列表。虽然大多数现有方法学习了一种预测每个单独项目排名得分的点积评分模型，但最近的研究表明，列表式方法通过建模同时展示的项目的内部列表相关性，可以进一步提高推荐质量。这激发了最近的列表重排和生成式推荐方法，它们优化整个列表的总体效用。然而，探索列表操作的组合空间是具有挑战性的，现有使用交叉熵损失的方法可能会遭受低多样性问题。本研究旨在学习一种策略，能够生成用户的足够多样性的项目列表，同时保持高推荐质量。提出的解决方案GFN4Rec是一个生成元学习模型，由生成流网络和列表变换器组成，通过利用生成流网络和处理项目的内部列表相互关联性的列表变换器的强大建模能力，生成具有高质量和多样性的项目列表。在真实世界数据集上的综合实验证明，GFN4Rec在推荐质量和多样性方面优于现有的最先进方法。

    Personalized recommender systems fulfill the daily demands of customers and boost online businesses. The goal is to learn a policy that can generate a list of items that matches the user's demand or interest. While most existing methods learn a pointwise scoring model that predicts the ranking score of each individual item, recent research shows that the listwise approach can further improve the recommendation quality by modeling the intra-list correlations of items that are exposed together. This has motivated the recent list reranking and generative recommendation approaches that optimize the overall utility of the entire list. However, it is challenging to explore the combinatorial space of list actions and existing methods that use cross-entropy loss may suffer from low diversity issues. In this work, we aim to learn a policy that can generate sufficiently diverse item lists for users while maintaining high recommendation quality. The proposed solution, GFN4Rec, is a generative met
    
[^10]: 如何发挥大语言模型在少样本关系抽取中的能力？

    How to Unleash the Power of Large Language Models for Few-shot Relation Extraction?. (arXiv:2305.01555v1 [cs.CL])

    [http://arxiv.org/abs/2305.01555](http://arxiv.org/abs/2305.01555)

    本文通过使用GPT-3.5模型在少样本关系抽取中，实现在四个不同数据集上的新的最优性能，并提出了与任务相关的指导说明和约束模式下的数据生成方法。

    

    语言模型的扩展已经彻底改变了广泛的自然语言处理任务，但是使用大型语言模型进行少样本关系抽取还没有得到全面探索。本文通过详细实验，研究了使用GPT-3.5进行少样本关系抽取的基本方法——上下文学习和数据生成。为了增强少样本性能，我们进一步提出了与任务相关的指导说明和约束模式下的数据生成。我们观察到，在上下文学习的情况下，可以实现与以前的提示学习方法相当的性能，而使用大型语言模型的数据生成可以推动以前的解决方案以在四个广泛研究的关系抽取数据集上获得新的最先进的少样本结果。我们希望我们的工作可以激发未来对大型语言模型在少样本关系抽取中的能力的研究。代码可以在 \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm} 中找到。

    Scaling language models have revolutionized widespread NLP tasks, yet little comprehensively explored few-shot relation extraction with large language models. In this paper, we investigate principal methodologies, in-context learning and data generation, for few-shot relation extraction via GPT-3.5 through exhaustive experiments. To enhance few-shot performance, we further propose task-related instructions and schema-constrained data generation. We observe that in-context learning can achieve performance on par with previous prompt learning approaches, and data generation with the large language model can boost previous solutions to obtain new state-of-the-art few-shot results on four widely-studied relation extraction datasets. We hope our work can inspire future research for the capabilities of large language models in few-shot relation extraction. Code is available in \url{https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^11]: 基于2D网格推荐面板的强化再排序

    Reinforcement Re-ranking with 2D Grid-based Recommendation Panels. (arXiv:2204.04954v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.04954](http://arxiv.org/abs/2204.04954)

    该论文提出了一种名为Panel-MDP的新型模型，通过采用强化学习策略，以用户喜好为导向，能够有效解决网格面板排列物品的问题，提高用户体验。

    

    现代推荐系统通常作为一个流式的单维排序列表呈现物品。近年来，在电子商务中有一种趋势，即推荐的物品以二维网格面板的形式组织，用户可以在竖直和水平方向上查看物品。在网格形式的结果面板中呈现物品对于推荐系统提出了新的挑战，因为现有模型都是设计用于输出序列列表，而网格面板中的插槽没有明确的顺序。直接将物品排名转换为网格（例如，预定义插槽的顺序）忽略了网格面板上用户特定的行为模式，并且不可避免地影响用户体验。为了解决这个问题，我们提出了一种新的马尔可夫决策过程（MDP），用于在推荐系统的最终再排序阶段中放置物品到二维网格结果面板中。该模型被称为Panel-MDP，它以早期阶段的初始物品排序为输入。然后，模型将以用户喜好为导向，采用强化学习策略来决定如何排列物品。

    Modern recommender systems usually present items as a streaming, one-dimensional ranking list. Recently there is a trend in e-commerce that the recommended items are organized grid-based panels with two dimensions where users can view the items in both vertical and horizontal directions. Presenting items in grid-based result panels poses new challenges to recommender systems because existing models are all designed to output sequential lists while the slots in a grid-based panel have no explicit order. Directly converting the item rankings into grids (e.g., pre-defining an order on the slots) overlooks the user-specific behavioral patterns on grid-based panels and inevitably hurts the user experiences. To address this issue, we propose a novel Markov decision process (MDP) to place the items in 2D grid-based result panels at the final re-ranking stage of the recommender systems. The model, referred to as Panel-MDP, takes an initial item ranking from the early stages as the input. Then,
    
[^12]: 自动查询再制定在源代码搜索中的系统性研究

    A Systematic Review of Automated Query Reformulations in Source Code Search. (arXiv:2108.09646v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2108.09646](http://arxiv.org/abs/2108.09646)

    许多研究尝试重新制定即席查询以支持开发人员进行代码搜索，本文针对70个主要查询再制定研究进行了细致筛选和深入的定性分析，提出了八种主要方法。

    

    修复软件漏洞和添加新功能是主要的维护任务之二。这些漏洞和功能以更改请求的形式报告。开发人员会从这些请求中选择一些关键词作为即席查询，然后使用搜索引擎执行查询，查找需要更改的软件代码的确切位置。然而，即使是经验丰富的开发人员通常也无法选择适当的查询，这导致在代码搜索期间进行昂贵的试错。多年来，许多研究尝试重新制定开发人员的即席查询以支持他们。本文系统地对70个主要查询再制定研究进行细致筛选（从2,970个候选研究中选择），进行深入的定性分析（如基础理论），并回答七个研究问题并提出主要发现。首先，迄今为止，已采用了八种主要方法（如词项加权，词项共现分析，词库查找）。

    Fixing software bugs and adding new features are two of the major maintenance tasks. Software bugs and features are reported as change requests. Developers consult these requests and often choose a few keywords from them as an ad hoc query. Then they execute the query with a search engine to find the exact locations within software code that need to be changed. Unfortunately, even experienced developers often fail to choose appropriate queries, which leads to costly trials and errors during a code search. Over the years, many studies attempt to reformulate the ad hoc queries from developers to support them. In this systematic literature review, we carefully select 70 primary studies on query reformulations from 2,970 candidate studies, perform an in-depth qualitative analysis (e.g., Grounded Theory), and then answer seven research questions with major findings. First, to date, eight major methodologies (e.g., term weighting, term co-occurrence analysis, thesaurus lookup) have been adop
    

