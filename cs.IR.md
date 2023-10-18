# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MeKB-Rec: Personal Knowledge Graph Learning for Cross-Domain Recommendation.](http://arxiv.org/abs/2310.11088) | 本论文提出了一种名为MeKB-Rec的跨领域推荐方法，在推荐系统中解决了冷启动问题。该方法利用个人知识图谱作为领域不变的用户兴趣表示，通过学习语义表示和注入世界知识，实现了对新用户的零-shot推荐。 |
| [^2] | [Nonet at SemEval-2023 Task 6: Methodologies for Legal Evaluation.](http://arxiv.org/abs/2310.11049) | 这篇论文介绍了我们在SemEval-2023法律评估任务6上的提交，主要集中在法律命名实体识别、法律判决预测和带解释的法院判决预测等子任务上。我们进行了多个实验，并取得了在各个子任务中具有竞争力的排名。 |
| [^3] | [If the Sources Could Talk: Evaluating Large Language Models for Research Assistance in History.](http://arxiv.org/abs/2310.10808) | 本文评估了大型语言模型在历史研究辅助中的应用，通过将高度专业化的学术资源嵌入到模型中，提供了一种对话形式的研究方法，可帮助研究人员检索不同类型的历史文献，并在问答和数据提取等任务中展现出卓越的表现。 |
| [^4] | [ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction.](http://arxiv.org/abs/2310.09234) | 这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。 |
| [^5] | [Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation.](http://arxiv.org/abs/2310.04633) | 提出了一个增强外部注意力的图对比学习框架，能够消除跨领域密度偏差并稳定地捕捉用户的行为模式。 |
| [^6] | [RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models.](http://arxiv.org/abs/2308.09904) | 提出了一个人类中心的推荐框架RAH，利用大型语言模型（LLMs）作为助手，实现用户满意度和个性化反馈，并成功应用于学习用户个性和调整推荐系统。 |
| [^7] | [Efficient High-Resolution Template Matching with Vector Quantized Nearest Neighbour Fields.](http://arxiv.org/abs/2306.15010) | 本研究提出了一种高效的高分辨率模板匹配方法，通过向量量化和滤波来减少计算量和考虑变形，取得了最先进的性能。 |
| [^8] | [Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation.](http://arxiv.org/abs/2305.07609) | 这篇论文介绍了一种新的推荐范式——通过LLM进行推荐，但由于LLMs可能存在社会偏见，需要进一步调查RecLLM所做推荐的公正性。为此，作者提出了一个新的公平性基准——FaiRLLM，并针对音乐和电影推荐场景中的八个敏感属性进行了评估。 |
| [^9] | [TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.](http://arxiv.org/abs/2305.00447) | TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。 |
| [^10] | [Editable User Profiles for Controllable Text Recommendation.](http://arxiv.org/abs/2304.04250) | 本文提出了一种新的概念值瓶颈模型LACE，用于可控文本推荐。该模型基于用户文档学习个性化的概念表示，并通过多种交互方式为用户提供了控制推荐的机制，验证了在离线和在线实验中该模型的推荐质量和有效性。 |
| [^11] | [Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval.](http://arxiv.org/abs/2210.05521) | 本研究提出了一种混合倒排索引(HI$^2$)用于加速稠密检索，通过嵌入聚类和显著词汇的协同作用，构建紧凑的倒排列表并提高检索质量。 |
| [^12] | [Vertical Allocation-based Fair Exposure Amortizing in Ranking.](http://arxiv.org/abs/2204.03046) | 本研究关注排名服务中的曝光公平性问题，证明了现有公平性优化方法在公平性与相关性之间的权衡方面存在不足，并提出了一种新的算法Vertic来解决该问题。 |
| [^13] | [Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems.](http://arxiv.org/abs/2204.01815) | 本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。 |
| [^14] | [BLM-17m: A Large-Scale Dataset for Black Lives Matter Topic Detection on Twitter.](http://arxiv.org/abs/2105.01331) | 本论文提出了一个用于推特上检测黑人生命至关重要话题的大规模数据集BLM-17m，涵盖了乔治·弗洛伊德事件期间的17百万推文。作者提供了两个基线模型TF-IDF和LDA，并对其进行了评估。 |

# 详细

[^1]: MeKB-Rec：个人知识图谱学习用于跨领域推荐

    MeKB-Rec: Personal Knowledge Graph Learning for Cross-Domain Recommendation. (arXiv:2310.11088v1 [cs.IR])

    [http://arxiv.org/abs/2310.11088](http://arxiv.org/abs/2310.11088)

    本论文提出了一种名为MeKB-Rec的跨领域推荐方法，在推荐系统中解决了冷启动问题。该方法利用个人知识图谱作为领域不变的用户兴趣表示，通过学习语义表示和注入世界知识，实现了对新用户的零-shot推荐。

    

    在现代推荐系统中，如何针对新用户有效地进行推荐，即冷启动问题，一直是一个长期存在的挑战。我们提出了个人知识图谱（PKG）作为一个领域不变的兴趣表示，并提出了一种名为MeKB-Rec的新型跨领域推荐范式。我们首先将知识图谱中的用户和实体进行关联，构建了用户兴趣的PKG，即MeKB。然后我们学习了MeKB的语义表示，用于跨领域推荐。为了高效利用CDR中有限的训练数据，MeKB-Rec采用了预训练语言模型将世界知识注入到对用户兴趣的理解中。与大多数现有系统不同，我们的方法在领域之间建立了语义映射，消除了对领域内用户行为的要求，实现了对新用户的零-shot推荐。

    It is a long-standing challenge in modern recommender systems to effectively make recommendations for new users, namely the cold-start problem. Cross-Domain Recommendation (CDR) has been proposed to address this challenge, but current ways to represent users' interests across systems are still severely limited. We introduce Personal Knowledge Graph (PKG) as a domain-invariant interest representation, and propose a novel CDR paradigm named MeKB-Rec. We first link users and entities in a knowledge base to construct a PKG of users' interests, named MeKB. Then we learn a semantic representation of MeKB for the cross-domain recommendation. To efficiently utilize limited training data in CDR, MeKB-Rec employs Pretrained Language Models to inject world knowledge into understanding users' interests. Beyond most existing systems, our approach builds a semantic mapping across domains which breaks the requirement for in-domain user behaviors, enabling zero-shot recommendations for new users in a 
    
[^2]: SemEval-2023任务6中的非纳任务:法律评估方法论。(arXiv:2310.11049v1 [cs.CL])

    Nonet at SemEval-2023 Task 6: Methodologies for Legal Evaluation. (arXiv:2310.11049v1 [cs.CL])

    [http://arxiv.org/abs/2310.11049](http://arxiv.org/abs/2310.11049)

    这篇论文介绍了我们在SemEval-2023法律评估任务6上的提交，主要集中在法律命名实体识别、法律判决预测和带解释的法院判决预测等子任务上。我们进行了多个实验，并取得了在各个子任务中具有竞争力的排名。

    

    本文描述了我们在SemEval-2023法律评估任务6上的提交。我们的提交主要集中在三个子任务上：任务B的法律命名实体识别(L-NER)，任务C1的法律判决预测(LJP)和任务C2的带解释的法院判决预测(CJPE)。我们对这些子任务进行了各种实验，并详细呈现了结果，包括数据统计和方法论。值得注意的是，像本研究中所涉及的法律任务正在因自动化法律分析和支持的需求增加而变得越来越重要。我们的团队在排行榜上报告的任务B、任务C1和任务C2中分别获得了15th、11th和1st的竞争排名。

    This paper describes our submission to the SemEval-2023 for Task 6 on LegalEval: Understanding Legal Texts. Our submission concentrated on three subtasks: Legal Named Entity Recognition (L-NER) for Task-B, Legal Judgment Prediction (LJP) for Task-C1, and Court Judgment Prediction with Explanation (CJPE) for Task-C2. We conducted various experiments on these subtasks and presented the results in detail, including data statistics and methodology. It is worth noting that legal tasks, such as those tackled in this research, have been gaining importance due to the increasing need to automate legal analysis and support. Our team obtained competitive rankings of 15$^{th}$, 11$^{th}$, and 1$^{st}$ in Task-B, Task-C1, and Task-C2, respectively, as reported on the leaderboard.
    
[^3]: 如果资源能够说话：评估大型语言模型在历史研究辅助中的应用

    If the Sources Could Talk: Evaluating Large Language Models for Research Assistance in History. (arXiv:2310.10808v1 [cs.IR])

    [http://arxiv.org/abs/2310.10808](http://arxiv.org/abs/2310.10808)

    本文评估了大型语言模型在历史研究辅助中的应用，通过将高度专业化的学术资源嵌入到模型中，提供了一种对话形式的研究方法，可帮助研究人员检索不同类型的历史文献，并在问答和数据提取等任务中展现出卓越的表现。

    

    强大的大型语言模型(LLM)的出现为历史记忆的对话形式提供了一种新的研究途径。我们通过将高度专业化学术资源的向量嵌入引入到LLM中，使得对话方法可以被历史学家和其他人文学科研究人员使用。具体地，我们评估和展示了LLM在研究人员检查不同类型文档的定制语料库时的辅助能力，包括但不限于：(1).一手资料，(2).由专家撰写的二手资料，以及(3).两者的结合。与传统的数字目录搜索界面（如元数据和全文搜索）相比，我们评估了LLM的更丰富的对话风格对两种主要任务的表现：(1).问答，以及(2).数据的提取和组织。我们展示了LLM的语义检索和推理能力在这些任务中的效果。

    The recent advent of powerful Large-Language Models (LLM) provides a new conversational form of inquiry into historical memory (or, training data, in this case). We show that by augmenting such LLMs with vector embeddings from highly specialized academic sources, a conversational methodology can be made accessible to historians and other researchers in the Humanities. Concretely, we evaluate and demonstrate how LLMs have the ability of assisting researchers while they examine a customized corpora of different types of documents, including, but not exclusive to: (1). primary sources, (2). secondary sources written by experts, and (3). the combination of these two. Compared to established search interfaces for digital catalogues, such as metadata and full-text search, we evaluate the richer conversational style of LLMs on the performance of two main types of tasks: (1). question-answering, and (2). extraction and organization of data. We demonstrate that LLMs semantic retrieval and reaso
    
[^4]: ClickPrompt: CTR模型是将语言模型适应为CTR预测的强大提示生成器

    ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction. (arXiv:2310.09234v1 [cs.IR])

    [http://arxiv.org/abs/2310.09234](http://arxiv.org/abs/2310.09234)

    这篇论文提出了一个新颖的模型，旨在同时模拟语义和协同知识，以实现准确的CTR估计，并解决推理效率问题。

    

    点击率（CTR）预测已经成为各种互联网应用程序中越来越不可或缺的。传统的CTR模型通过独热编码将多字段分类数据转换为ID特征，并提取特征之间的协同信号。这种范式的问题在于语义信息的丢失。另一方面的研究通过将输入数据转换为文本句子来探索预训练语言模型（PLM）在CTR预测中的潜力。虽然语义信号得到了保留，但它们通常无法捕捉到协同信息（如特征交互、纯ID特征），更不用说由庞大的模型大小带来的无法接受的推理开销了。在本文中，我们旨在为准确的CTR估计建立语义知识和协同知识，并解决推理效率问题。为了从两个领域中受益并弥合它们之间的差距，我们提出了一种新颖的模型-。

    Click-through rate (CTR) prediction has become increasingly indispensable for various Internet applications. Traditional CTR models convert the multi-field categorical data into ID features via one-hot encoding, and extract the collaborative signals among features. Such a paradigm suffers from the problem of semantic information loss. Another line of research explores the potential of pretrained language models (PLMs) for CTR prediction by converting input data into textual sentences through hard prompt templates. Although semantic signals are preserved, they generally fail to capture the collaborative information (e.g., feature interactions, pure ID features), not to mention the unacceptable inference overhead brought by the huge model size. In this paper, we aim to model both the semantic knowledge and collaborative knowledge for accurate CTR estimation, and meanwhile address the inference inefficiency issue. To benefit from both worlds and close their gaps, we propose a novel model-
    
[^5]: 无偏和鲁棒性：增强外部注意力的跨领域序列推荐中的图对比学习

    Unbiased and Robust: External Attention-enhanced Graph Contrastive Learning for Cross-domain Sequential Recommendation. (arXiv:2310.04633v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.04633](http://arxiv.org/abs/2310.04633)

    提出了一个增强外部注意力的图对比学习框架，能够消除跨领域密度偏差并稳定地捕捉用户的行为模式。

    

    跨领域序列推荐器（CSRs）因能够利用多个领域的辅助信息捕捉用户的序列偏好而引起了相当大的研究关注。然而，这些研究通常遵循一个理想的设置，即不同的领域遵守相似的数据分布，忽视了由不对称交互密度带来的偏差（即跨领域密度偏差）。此外，序列编码器中经常采用的机制（如自注意网络）只关注局部视图内的交互，忽视了不同训练批次之间的全局相关性。为此，我们提出了一种增强外部注意力的图对比学习框架，即EA-GCL。具体而言，为了消除跨领域密度偏差的影响，在传统图编码器下附加了一个辅助自监督学习（SSL）任务，采用多任务学习方式。为了稳定地捕捉用户的行为模式，我们开发了一个...

    Cross-domain sequential recommenders (CSRs) are gaining considerable research attention as they can capture user sequential preference by leveraging side information from multiple domains. However, these works typically follow an ideal setup, i.e., different domains obey similar data distribution, which ignores the bias brought by asymmetric interaction densities (a.k.a. the inter-domain density bias). Besides, the frequently adopted mechanism (e.g., the self-attention network) in sequence encoder only focuses on the interactions within a local view, which overlooks the global correlations between different training batches. To this end, we propose an External Attention-enhanced Graph Contrastive Learning framework, namely EA-GCL. Specifically, to remove the impact of the inter-domain density bias, an auxiliary Self-Supervised Learning (SSL) task is attached to the traditional graph encoder under a multi-task learning manner. To robustly capture users' behavioral patterns, we develop a
    
[^6]: RAH！RecSys-Assistant-Human：一个具有大型语言模型的人类中心推荐框架

    RAH! RecSys-Assistant-Human: A Human-Central Recommendation Framework with Large Language Models. (arXiv:2308.09904v1 [cs.IR])

    [http://arxiv.org/abs/2308.09904](http://arxiv.org/abs/2308.09904)

    提出了一个人类中心的推荐框架RAH，利用大型语言模型（LLMs）作为助手，实现用户满意度和个性化反馈，并成功应用于学习用户个性和调整推荐系统。

    

    推荐生态系统涉及到推荐系统（计算机）和用户（人类）之间的交互。与推荐系统的角度不同，我们尝试从用户的角度利用大型语言模型（LLMs），并提出一个更加人类中心的推荐框架，命名为RAH。该框架包括推荐系统、助手和人类。助手是一个基于LLMs的个人代理，用于实现用户满意度。助手扮演非侵入性的角色，RAH框架可以适应不同的推荐系统和用户群体。随后，我们实现并评估了RAH框架，用于学习用户个性和代理人类反馈。实验表明：（1）使用学习-行动-评论家和反思机制可以导致更加一致的个性，（2）我们的助手可以有效地代理人类反馈并帮助调整推荐系统。最后，我们讨论了在RAH框架中进一步解决人类中心问题的策略，包括用户``夺权''等问题。

    The recommendation ecosystem involves interactions between recommender systems(Computer) and users(Human). Orthogonal to the perspective of recommender systems, we attempt to utilize LLMs from the perspective of users and propose a more human-central recommendation framework named RAH, which consists of Recommender system, Assistant and Human. The assistant is a LLM-based and personal proxy for a human to achieve user satisfaction. The assistant plays a non-invasion role and the RAH framework can adapt to different recommender systems and user groups. Subsequently, we implement and evaluate the RAH framework for learning user personalities and proxy human feedback. The experiment shows that (1) using learn-action-critic and reflection mechanisms can lead more aligned personality and (2) our assistant can effectively proxy human feedback and help adjust recommender systems. Finally, we discuss further strategies in the RAH framework to address human-central concerns including user contr
    
[^7]: 高分辨率模板匹配中的高效向量量化最近邻场

    Efficient High-Resolution Template Matching with Vector Quantized Nearest Neighbour Fields. (arXiv:2306.15010v1 [cs.CV])

    [http://arxiv.org/abs/2306.15010](http://arxiv.org/abs/2306.15010)

    本研究提出了一种高效的高分辨率模板匹配方法，通过向量量化和滤波来减少计算量和考虑变形，取得了最先进的性能。

    

    模板匹配是计算机视觉中的基础问题，并在物体检测、图像配准和物体跟踪等领域有应用。当前最先进的方法是依赖于最近邻（NN）匹配，在该方法中，将查询特征空间转换为NN空间，其中每个查询像素用模板像素中的最近邻表示。NN匹配在遮挡、外观变化、光照变化和非刚性变换等方面表现出更好的性能。然而，NN匹配在高分辨率数据和高维特征方面的扩展性较差。本文提出了一种基于NN的模板匹配方法，该方法有效地减少了NN计算量，并在NN场中引入滤波以考虑变形。首先，通过向量量化将模板表示为k个特征，然后通过滤波比较模板和查询在k个特征上的分布。我们展示了该方法达到了最先进的性能。

    Template matching is a fundamental problem in computer vision and has applications in various fields, such as object detection, image registration, and object tracking. The current state-of-the-art methods rely on nearest-neighbour (NN) matching in which the query feature space is converted to NN space by representing each query pixel with its NN in the template pixels. The NN-based methods have been shown to perform better in occlusions, changes in appearance, illumination variations, and non-rigid transformations. However, NN matching scales poorly with high-resolution data and high feature dimensions. In this work, we present an NN-based template-matching method which efficiently reduces the NN computations and introduces filtering in the NN fields to consider deformations. A vector quantization step first represents the template with $k$ features, then filtering compares the template and query distributions over the $k$ features. We show that state-of-the-art performance was achiev
    
[^8]: ChatGPT是否公平可靠？评估大型语言模型推荐中的公平性

    Is ChatGPT Fair for Recommendation? Evaluating Fairness in Large Language Model Recommendation. (arXiv:2305.07609v1 [cs.IR])

    [http://arxiv.org/abs/2305.07609](http://arxiv.org/abs/2305.07609)

    这篇论文介绍了一种新的推荐范式——通过LLM进行推荐，但由于LLMs可能存在社会偏见，需要进一步调查RecLLM所做推荐的公正性。为此，作者提出了一个新的公平性基准——FaiRLLM，并针对音乐和电影推荐场景中的八个敏感属性进行了评估。

    

    大型语言模型（LLM）的显着成就导致一种新的推荐范式——通过LLM进行推荐（RecLLM）。然而，需要注意LLMs可能包含社会偏见，因此需要进一步调查RecLLM所做推荐的公正性。为了避免RecLLM的潜在风险，有必要从用户的各种敏感属性角度评估RecLLM的公平性。由于RecLLM范式与传统推荐范式之间存在差异，因此直接使用传统推荐的公平性基准是有问题的。为了解决这个困境，我们提出了一个新的基准，称为“通过LLM的推荐的公平性”（FaiRLLM）。该基准包括精心设计的指标和数据集，涵盖两个推荐场景中的八个敏感属性：音乐和电影。通过利用我们的FaiRLLM基准，我们进行了一项评估。

    The remarkable achievements of Large Language Models (LLMs) have led to the emergence of a novel recommendation paradigm -- Recommendation via LLM (RecLLM). Nevertheless, it is important to note that LLMs may contain social prejudices, and therefore, the fairness of recommendations made by RecLLM requires further investigation. To avoid the potential risks of RecLLM, it is imperative to evaluate the fairness of RecLLM with respect to various sensitive attributes on the user side. Due to the differences between the RecLLM paradigm and the traditional recommendation paradigm, it is problematic to directly use the fairness benchmark of traditional recommendation. To address the dilemma, we propose a novel benchmark called Fairness of Recommendation via LLM (FaiRLLM). This benchmark comprises carefully crafted metrics and a dataset that accounts for eight sensitive attributes1 in two recommendation scenarios: music and movies. By utilizing our FaiRLLM benchmark, we conducted an evaluation 
    
[^9]: TALLRec: 一种与推荐系统对齐的大型语言模型有效且高效的调整框架

    TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation. (arXiv:2305.00447v1 [cs.IR])

    [http://arxiv.org/abs/2305.00447](http://arxiv.org/abs/2305.00447)

    TALLRec是对LLMs进行调整的一种高效且有效的框架，用于将LLMs与推荐系统对齐，从而增强LLMs在推荐任务中的能力。

    

    大型语言模型（LLMs）已经展现了在不同领域的显著性能，因此研究人员开始探索它们在推荐系统中的潜力。虽然初始的尝试已经利用了LLMs的优异能力，比如通过上下文学习中的提示词来丰富知识并进行强化泛化，但是由于LLMs的训练任务与推荐任务之间的巨大差异以及预训练期间的不足的推荐数据，LLMs在推荐任务中的性能仍然不理想。为了填补这一差距，我们考虑使用推荐数据对LLMs进行调整来构建大型推荐语言模型。为此，我们提出了一种名为TALLRec的高效且有效的调整框架，用于将LLMs与推荐系统对齐。我们已经证明了所提出的TALLRec框架可以显著增强LLMs在推荐任务中的能力。

    Large Language Models (LLMs) have demonstrated remarkable performance across diverse domains, thereby prompting researchers to explore their potential for use in recommendation systems. Initial attempts have leveraged the exceptional capabilities of LLMs, such as rich knowledge and strong generalization through In-context Learning, which involves phrasing the recommendation task as prompts. Nevertheless, the performance of LLMs in recommendation tasks remains suboptimal due to a substantial disparity between the training tasks for LLMs and recommendation tasks, as well as inadequate recommendation data during pre-training. To bridge the gap, we consider building a Large Recommendation Language Model by tunning LLMs with recommendation data. To this end, we propose an efficient and effective Tuning framework for Aligning LLMs with Recommendation, namely TALLRec. We have demonstrated that the proposed TALLRec framework can significantly enhance the recommendation capabilities of LLMs in 
    
[^10]: 可编辑用户档案的可控文本推荐方法

    Editable User Profiles for Controllable Text Recommendation. (arXiv:2304.04250v1 [cs.IR])

    [http://arxiv.org/abs/2304.04250](http://arxiv.org/abs/2304.04250)

    本文提出了一种新的概念值瓶颈模型LACE，用于可控文本推荐。该模型基于用户文档学习个性化的概念表示，并通过多种交互方式为用户提供了控制推荐的机制，验证了在离线和在线实验中该模型的推荐质量和有效性。

    

    实现高质量推荐的方法通常依赖于从交互数据中学习潜在表示。然而这些方法没有提供给用户控制所接收的推荐的机制。本文提出了LACE，一种新颖的概念值瓶颈模型，用于可控文本推荐。LACE基于用户交互的文档检索，将每个用户表示为简洁的可读的概念集，并基于用户文档学习概念的个性化表示。该基于概念的用户档案被利用来做出推荐。我们的模型设计通过透明的用户档案，提供了控制推荐的多种直观交互方式。我们首先在三个推荐任务（温启动、冷启动和零样本）的六个数据集上进行了离线评估，验证了从LACE获得的推荐质量。接下来，我们在在线实验中验证了LACE的有效性和用户控制能力。

    Methods for making high-quality recommendations often rely on learning latent representations from interaction data. These methods, while performant, do not provide ready mechanisms for users to control the recommendation they receive. Our work tackles this problem by proposing LACE, a novel concept value bottleneck model for controllable text recommendations. LACE represents each user with a succinct set of human-readable concepts through retrieval given user-interacted documents and learns personalized representations of the concepts based on user documents. This concept based user profile is then leveraged to make recommendations. The design of our model affords control over the recommendations through a number of intuitive interactions with a transparent user profile. We first establish the quality of recommendations obtained from LACE in an offline evaluation on three recommendation tasks spanning six datasets in warm-start, cold-start, and zero-shot setups. Next, we validate the 
    
[^11]: 混合倒排索引是一种强大的稠密检索加速器

    Hybrid Inverted Index Is a Robust Accelerator for Dense Retrieval. (arXiv:2210.05521v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2210.05521](http://arxiv.org/abs/2210.05521)

    本研究提出了一种混合倒排索引(HI$^2$)用于加速稠密检索，通过嵌入聚类和显著词汇的协同作用，构建紧凑的倒排列表并提高检索质量。

    

    倒排文件结构是一种常用的加速稠密检索的技术。它根据嵌入将文档聚类；在搜索过程中，根据输入查询探测附近的聚类，并且仅对其中的文档进行后续的解码，从而避免了穷举遍历的昂贵代价。然而，聚类过程总是有损的，这导致探测到的聚类中缺失了相关的文档，从而降低了检索质量。相反，词汇匹配，如显著词汇的重叠，更容易识别相关文档。在这项工作中，我们提出了混合倒排索引 (HI$^2$)，其中嵌入聚类和显著词汇共同加速稠密检索。为了兼顾效果和效率，我们设计了一个聚类选择器和一个词汇选择器，用于构建紧凑的倒排列表并快速搜索它们。此外，我们利用简单的无监督算法和端到端学习来提高索引质量.

    Inverted file structure is a common technique for accelerating dense retrieval. It clusters documents based on their embeddings; during searching, it probes nearby clusters w.r.t. an input query and only evaluates documents within them by subsequent codecs, thus avoiding the expensive cost of exhaustive traversal. However, the clustering is always lossy, which results in the miss of relevant documents in the probed clusters and hence degrades retrieval quality. In contrast, lexical matching, such as overlaps of salient terms, tends to be strong feature for identifying relevant documents. In this work, we present the Hybrid Inverted Index (HI$^2$), where the embedding clusters and salient terms work collaboratively to accelerate dense retrieval. To make best of both effectiveness and efficiency, we devise a cluster selector and a term selector, to construct compact inverted lists and efficiently searching through them. Moreover, we leverage simple unsupervised algorithms as well as end-
    
[^12]: 基于垂直分配的排名中公平曝光摊销

    Vertical Allocation-based Fair Exposure Amortizing in Ranking. (arXiv:2204.03046v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.03046](http://arxiv.org/abs/2204.03046)

    本研究关注排名服务中的曝光公平性问题，证明了现有公平性优化方法在公平性与相关性之间的权衡方面存在不足，并提出了一种新的算法Vertic来解决该问题。

    

    结果排名经常影响消费者满意度以及排名服务中每个项目的曝光量。仅根据相关性对项目进行排名会导致项目曝光分配不公平，从而为项目生产者/提供者带来不公平的机会和经济收益。这种不公平会导致提供者离开系统，并阻止新的提供者加入。最终，消费者会剩下更少的购买选项，消费者和提供者的效用都会受到损害。因此，对于双方来说，保持排名相关性和公平之间的平衡至关重要。本文聚焦于排名服务中的曝光公平性。我们证明了现有的公平性优化方法在公平性和相关性之间的权衡方面可能不是最优的，因为它们没有充分利用消费者的先验知识。我们进一步提出了一种名为Vertic的新算法。

    Result ranking often affects consumer satisfaction as well as the amount of exposure each item receives in the ranking services. Myopically maximizing customer satisfaction by ranking items only according to relevance will lead to unfair distribution of exposure for items, followed by unfair opportunities and economic gains for item producers/providers. Such unfairness will force providers to leave the system and discourage new providers from coming in. Eventually, fewer purchase options would be left for consumers, and the utilities of both consumers and providers would be harmed. Thus, to maintain a balance between ranking relevance and fairness is crucial for both parties. In this paper, we focus on the exposure fairness in ranking services. We demonstrate that existing methods for amortized fairness optimization could be suboptimal in terms of fairness-relevance tradeoff because they fail to utilize the prior knowledge of consumers. We further propose a novel algorithm named Vertic
    
[^13]: 具有可证明的一致性和公平保证的推荐系统张量补全

    Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems. (arXiv:2204.01815v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2204.01815](http://arxiv.org/abs/2204.01815)

    本文介绍了一种新的一致性方法来解决矩阵和张量补全问题，在推荐系统应用中，我们证明了通过保留单位比例和一致性两个约束条件可以实现解的存在性与唯一性。

    

    我们引入了一种新的基于一致性的方法来定义和解决非负/正矩阵和张量补全问题。该框架的新颖之处在于，我们不是人为地将问题形式化为任意优化问题，例如，最小化一个结构量，如秩或范数，而是展示了一个单一的属性/约束：保留单位比例一致性，保证了解的存在，并在相对较弱的支持假设下保证了解的唯一性。该框架和解算法也直接推广到任意维度的张量中，同时保持了固定维度 d 的问题规模的线性计算复杂性。在推荐系统应用中，我们证明了两个合理的性质，这些性质应该适用于任何 RS 问题的解，足以允许在我们的框架内建立唯一性保证。关键理论贡献是展示了这些约束下解的存在性与唯一性。

    We introduce a new consistency-based approach for defining and solving nonnegative/positive matrix and tensor completion problems. The novelty of the framework is that instead of artificially making the problem well-posed in the form of an application-arbitrary optimization problem, e.g., minimizing a bulk structural measure such as rank or norm, we show that a single property/constraint: preserving unit-scale consistency, guarantees the existence of both a solution and, under relatively weak support assumptions, uniqueness. The framework and solution algorithms also generalize directly to tensors of arbitrary dimensions while maintaining computational complexity that is linear in problem size for fixed dimension d. In the context of recommender system (RS) applications, we prove that two reasonable properties that should be expected to hold for any solution to the RS problem are sufficient to permit uniqueness guarantees to be established within our framework. Key theoretical contribu
    
[^14]: BLM-17m: 一个用于推特上黑人生命至关重要话题检测的大规模数据集

    BLM-17m: A Large-Scale Dataset for Black Lives Matter Topic Detection on Twitter. (arXiv:2105.01331v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2105.01331](http://arxiv.org/abs/2105.01331)

    本论文提出了一个用于推特上检测黑人生命至关重要话题的大规模数据集BLM-17m，涵盖了乔治·弗洛伊德事件期间的17百万推文。作者提供了两个基线模型TF-IDF和LDA，并对其进行了评估。

    

    人权保护是世界上最重要的问题之一。本文旨在提供一个涵盖最近几个月全球影响深远的人权矛盾之一——乔治·弗洛伊德事件的数据集。我们提出了一个带有17百万推文的主题检测标记数据集。这些推文是从2020年5月25日至2020年8月21日收集的，涵盖了这一事件开始后的89天。我们通过监测全球和本地报纸的最热门新闻主题对数据集进行了标记。除此之外，我们还提供了两个基线模型，TF-IDF和LDA。我们使用三个不同的k值对这两种方法的精确度、召回率和F1分数进行了评估。收集到的数据集可以在https://github.com/MeysamAsgariC/BLMT 上找到。

    Protection of human rights is one of the most important problems of our world. In this paper, our aim is to provide a dataset which covers one of the most significant human rights contradiction in recent months affected the whole world, George Floyd incident. We propose a labeled dataset for topic detection that contains 17 million tweets. These Tweets are collected from 25 May 2020 to 21 August 2020 that covers 89 days from start of this incident. We labeled the dataset by monitoring most trending news topics from global and local newspapers. Apart from that, we present two baselines, TF-IDF and LDA. We evaluated the results of these two methods with three different k values for metrics of precision, recall and f1-score. The collected dataset is available at https://github.com/MeysamAsgariC/BLMT.
    

