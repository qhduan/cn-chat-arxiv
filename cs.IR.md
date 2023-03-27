# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited.](http://arxiv.org/abs/2303.13835) | 推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。 |
| [^2] | [GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering.](http://arxiv.org/abs/2303.13284) | 本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。 |
| [^3] | [One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER.](http://arxiv.org/abs/2301.10410) | 本论文提出了基于协作域前缀调整的跨领域实体识别，使用文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务，避免了先前的为每个领域结束一个全新的NER模型的问题。 |
| [^4] | [Dataset vs Reality: Understanding Model Performance from the Perspective of Information Need.](http://arxiv.org/abs/2212.02726) | 研究探讨了模型在解决真实世界问题和基准数据集中表现不同的原因，指出模型受到数据集信息需求影响。 |

# 详细

[^1]: 推荐系统何去何从？ID- vs. 基于模态的推荐模型再探讨

    Where to Go Next for Recommender Systems? ID- vs. Modality-based recommender models revisited. (arXiv:2303.13835v1 [cs.IR])

    [http://arxiv.org/abs/2303.13835](http://arxiv.org/abs/2303.13835)

    推荐系统中，使用唯一标识的IDRec模型相比使用模态的MoRec模型在推荐准确性和效率上表现更好，然而，需要根据具体情况选择适合的推荐模型。

    

    过去十年，利用唯一标识（ID）来表示不同用户和物品的推荐模型一直是最先进的，并且在推荐系统文献中占主导地位。与此同时，预训练模态编码器（如BERT和ViT）在对物品的原始模态特征（如文本和图像）进行建模方面变得越来越强大。因此，自然而然的问题是：通过用最先进的模态编码器替换物品ID嵌入向量，一个纯粹的基于模态的推荐模型（MoRec）能否胜过或与纯ID基础模型（IDRec）相匹配？实际上，早在十年前，这个问题就被回答了，IDRec在推荐准确性和效率方面都远远胜过MoRec。我们旨在重新审视这个“老问题”，从多个方面对MoRec进行系统研究。具体而言，我们研究了几个子问题：（i）在实际场景中，MoRec或IDRec哪个推荐模式表现更好，特别是在一般情况和......

    Recommendation models that utilize unique identities (IDs) to represent distinct users and items have been state-of-the-art (SOTA) and dominated the recommender systems (RS) literature for over a decade. Meanwhile, the pre-trained modality encoders, such as BERT and ViT, have become increasingly powerful in modeling the raw modality features of an item, such as text and images. Given this, a natural question arises: can a purely modality-based recommendation model (MoRec) outperforms or matches a pure ID-based model (IDRec) by replacing the itemID embedding with a SOTA modality encoder? In fact, this question was answered ten years ago when IDRec beats MoRec by a strong margin in both recommendation accuracy and efficiency. We aim to revisit this `old' question and systematically study MoRec from several aspects. Specifically, we study several sub-questions: (i) which recommendation paradigm, MoRec or IDRec, performs better in practical scenarios, especially in the general setting and 
    
[^2]: GETT-QA：基于图嵌入的知识图谱问答中的T2T Transformer

    GETT-QA: Graph Embedding based T2T Transformer for Knowledge Graph Question Answering. (arXiv:2303.13284v1 [cs.CL])

    [http://arxiv.org/abs/2303.13284](http://arxiv.org/abs/2303.13284)

    本论文提出了GETT-QA系统，该系统使用T5对自然语言问题生成简化的SPARQL查询，并使用截断的KG嵌入提高了知识图谱问答的性能。

    

    本文提出了一个名为GETT-QA的端到端知识图谱问答系统。GETT-QA使用了T5，这是一种热门的文本到文本预训练语言模型。该模型以自然语言形式的问题作为输入并生成所需SPARQL查询的简化形式。在简化形式中，模型不直接生成实体和关系ID，而是产生相应的实体和关系标签。标签在随后的步骤中与KG实体和关系ID联系起来。为了进一步改进结果，我们指导模型为每个实体生成KG嵌入的截断版本。截断的KG嵌入使得更精细的搜索从而更有效进行消歧。我们发现，T5能够在不改变损失函数的情况下学习截断的KG嵌入，提高了KGQA的性能。因此，我们在Wikidata的LC-QuAD 2.0和SimpleQuestions-Wikidata数据集上报告了端到端KGQA的强大结果。

    In this work, we present an end-to-end Knowledge Graph Question Answering (KGQA) system named GETT-QA. GETT-QA uses T5, a popular text-to-text pre-trained language model. The model takes a question in natural language as input and produces a simpler form of the intended SPARQL query. In the simpler form, the model does not directly produce entity and relation IDs. Instead, it produces corresponding entity and relation labels. The labels are grounded to KG entity and relation IDs in a subsequent step. To further improve the results, we instruct the model to produce a truncated version of the KG embedding for each entity. The truncated KG embedding enables a finer search for disambiguation purposes. We find that T5 is able to learn the truncated KG embeddings without any change of loss function, improving KGQA performance. As a result, we report strong results for LC-QuAD 2.0 and SimpleQuestions-Wikidata datasets on end-to-end KGQA over Wikidata.
    
[^3]: 适用于所有领域的一个模型：基于协作域前缀调整的跨领域实体识别

    One Model for All Domains: Collaborative Domain-Prefix Tuning for Cross-Domain NER. (arXiv:2301.10410v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.10410](http://arxiv.org/abs/2301.10410)

    本论文提出了基于协作域前缀调整的跨领域实体识别，使用文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务，避免了先前的为每个领域结束一个全新的NER模型的问题。

    

    解决实际场景中低资源问题是跨领域实体识别的一个挑战性任务。先前典型的解决方案主要通过使用来自丰富资源领域的数据进行预训练语言模型(PLMs)获得NER模型并将其适应于目标领域。由于不同领域实体类型之间的不匹配问题，先前的方法通常调整所有PLMs的参数，从而为每个领域结束一个全新的NER模型。此外，当前的模型只关注于利用一个普通来源领域中的知识，而未能成功地将来自多个来源领域的知识转移到目标上。为了解决这些问题，我们基于文本到文本生成的PLM引入了协作域前缀调整跨领域NER(CP-NER)。具体来说，我们呈现了用于文本到文本生成的支撑领域相关指导来将知识转移至新域NER任务而无需结构修改。我们利用冻结的PLMs并进行协作域前缀调整。

    Cross-domain NER is a challenging task to address the low-resource problem in practical scenarios. Previous typical solutions mainly obtain a NER model by pre-trained language models (PLMs) with data from a rich-resource domain and adapt it to the target domain. Owing to the mismatch issue among entity types in different domains, previous approaches normally tune all parameters of PLMs, ending up with an entirely new NER model for each domain. Moreover, current models only focus on leveraging knowledge in one general source domain while failing to successfully transfer knowledge from multiple sources to the target. To address these issues, we introduce Collaborative Domain-Prefix Tuning for cross-domain NER (CP-NER) based on text-to-text generative PLMs. Specifically, we present text-to-text generation grounding domain-related instructors to transfer knowledge to new domain NER tasks without structural modifications. We utilize frozen PLMs and conduct collaborative domain-prefix tuning
    
[^4]: 数据集与现实：从信息需求的角度理解模型性能

    Dataset vs Reality: Understanding Model Performance from the Perspective of Information Need. (arXiv:2212.02726v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2212.02726](http://arxiv.org/abs/2212.02726)

    研究探讨了模型在解决真实世界问题和基准数据集中表现不同的原因，指出模型受到数据集信息需求影响。

    

    深度学习技术带来了许多在一些基准测试中胜过人类的模型。一个有趣的问题是：这些模型能否很好地解决与基准数据集相似的真实世界问题（例如，相同的输入/输出）？我们认为一个模型是被训练来回答创建训练数据集时相同的信息需求的。虽然一些数据集可能具有高度结构相似性，例如问答（QA）任务的问答对和图像字幕（IC）任务的图像字幕对，但它们可能代表不同的研究任务，旨在回答不同的信息需求。为了支持我们的论点，我们使用问答任务和图像字幕任务作为两个案例研究，并比较它们的广泛使用的基准数据集。从信息检索的信息需求角度出发，我们展示了数据集创建过程中的差异以及数据集之间形态和语法属性的差异。

    Deep learning technologies have brought us many models that outperform human beings on a few benchmarks. An interesting question is: can these models well solve real-world problems with similar settings (e.g., identical input/output) to the benchmark datasets? We argue that a model is trained to answer the same information need for which the training dataset is created. Although some datasets may share high structural similarities, e.g., question-answer pairs for the question answering (QA) task and image-caption pairs for the image captioning (IC) task, they may represent different research tasks aiming for answering different information needs. To support our argument, we use the QA task and IC task as two case studies and compare their widely used benchmark datasets. From the perspective of information need in the context of information retrieval, we show the differences in the dataset creation processes, and the differences in morphosyntactic properties between datasets. The differ
    

