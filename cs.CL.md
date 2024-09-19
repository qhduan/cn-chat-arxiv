# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models](https://arxiv.org/abs/2404.02657) | 本研究重新思考了大型语言模型知识蒸馏中对Kullback-Leibler散度的应用，发现逆Kullback-Leibler和正向Kullback-Leibler散度在优化目标上相似，为此提出了一种自适应Kullback-Leiber散度方法。 |
| [^2] | [Prior Constraints-based Reward Model Training for Aligning Large Language Models](https://arxiv.org/abs/2404.00978) | 本文提出了基于先验约束的奖励模型训练方法，有效改善了对齐大型语言模型的性能。 |
| [^3] | [TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa](https://arxiv.org/abs/2404.00297) | TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。 |
| [^4] | [Multi-Hop Table Retrieval for Open-Domain Text-to-SQL](https://arxiv.org/abs/2402.10666) | 提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果 |
| [^5] | [LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education](https://arxiv.org/abs/2402.06264) | 本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。 |
| [^6] | [LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback](https://arxiv.org/abs/2311.09336) | LLMRefine提出了一种细粒度反馈模型来指导大型语言模型定位缺陷并进行优化，在机器翻译、长篇问答和主题总结等任务中取得显著的改进。 |
| [^7] | [Natural Language Processing for Dialects of a Language: A Survey.](http://arxiv.org/abs/2401.05632) | 这项调查研究了自然语言处理中针对方言的方法和问题，强调了方言对于NLP模型性能和语言技术公平性的影响，并提供了关于方言相关任务和语言的全面综述。 |
| [^8] | [Product Attribute Value Extraction using Large Language Models.](http://arxiv.org/abs/2310.12537) | 本文研究使用大型语言模型作为预训练的替代方法，解决了传统属性/值提取技术中需要大量训练数据和对未知属性值的挑战问题。 |
| [^9] | [Image Hijacking: Adversarial Images can Control Generative Models at Runtime.](http://arxiv.org/abs/2309.00236) | 本研究发现对抗性图像能够在运行时控制生成模型，并提出了通用的方法来创建图像劫持。通过研究三种攻击类型，我们发现这些攻击对最新的视觉语言模型具有高达90％以上的成功率。该研究引发了对基础模型安全性的严重担忧。 |
| [^10] | [PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning.](http://arxiv.org/abs/2305.19472) | PlaSma提出了一种使用小型语言模型进行过程知识和计划能力的新方法， |
| [^11] | [Measuring Stereotypes using Entity-Centric Data.](http://arxiv.org/abs/2305.09548) | 本文提出并评估了三种新的以实体为中心的方法，展示了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。 |
| [^12] | [On the Creativity of Large Language Models.](http://arxiv.org/abs/2304.00008) | 这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。 |

# 详细

[^1]: 在大型语言模型知识蒸馏中重新思考Kullback-Leibler散度

    Rethinking Kullback-Leibler Divergence in Knowledge Distillation for Large Language Models

    [https://arxiv.org/abs/2404.02657](https://arxiv.org/abs/2404.02657)

    本研究重新思考了大型语言模型知识蒸馏中对Kullback-Leibler散度的应用，发现逆Kullback-Leibler和正向Kullback-Leibler散度在优化目标上相似，为此提出了一种自适应Kullback-Leiber散度方法。

    

    Kullback-Leibler散度在知识蒸馏中被广泛应用于压缩大型语言模型。本研究从经验和理论上证明了，在LLMs的知识蒸馏中，与之前断言的逆Kullback-Leibler（RKL）散度寻找模式并因此优于寻找平均值的正向Kullback-Leibler（FKL）散度相反，实际上在知识蒸馏中都没有体现出寻找模式或寻找平均值的特性。相反，发现RKL和FKL具有相同的优化目标，并在足够数量的时代之后都会收敛。然而，由于实际约束，LLMs很少被训练如此多的时代。同时，我们进一步发现，RKL在分布的尾部，而FKL在开始时代侧重于分布的头部。因此，我们提出了一种简单而有效的自适应Kullback-Leiber（AKL）散度方法，该方法自适应地分配权重来组合F

    arXiv:2404.02657v1 Announce Type: cross  Abstract: Kullback-Leiber divergence has been widely used in Knowledge Distillation (KD) to compress Large Language Models (LLMs). Contrary to prior assertions that reverse Kullback-Leibler (RKL) divergence is mode-seeking and thus preferable over the mean-seeking forward Kullback-Leibler (FKL) divergence, this study empirically and theoretically demonstrates that neither mode-seeking nor mean-seeking properties manifest in KD for LLMs. Instead, RKL and FKL are found to share the same optimization objective and both converge after a sufficient number of epochs. However, due to practical constraints, LLMs are seldom trained for such an extensive number of epochs. Meanwhile, we further find that RKL focuses on the tail part of the distributions, while FKL focuses on the head part at the beginning epochs. Consequently, we propose a simple yet effective Adaptive Kullback-Leiber (AKL) divergence method, which adaptively allocates weights to combine F
    
[^2]: 基于先验约束的奖励模型训练以对齐大尺寸语言模型

    Prior Constraints-based Reward Model Training for Aligning Large Language Models

    [https://arxiv.org/abs/2404.00978](https://arxiv.org/abs/2404.00978)

    本文提出了基于先验约束的奖励模型训练方法，有效改善了对齐大型语言模型的性能。

    

    使用人类反馈的强化学习方法来对齐大型语言模型（LLMs）通常训练一个奖励模型，该模型使用比较对来计算排名损失。然而，训练过程存在一个固有问题：由于缺乏约束，奖励分数在强化学习过程中呈现不受控制的扩展。本文提出了一种基于先验约束的奖励模型（PCRM）训练方法来缓解这一问题。PCRM在奖励模型训练中融合了先验约束，具体来说是每个比较对输出之间的长度比和余弦相似性，以调节优化幅度并控制得分差距。我们通过检查PCRM与人类偏好的排名相关性以及通过RL对LLMs对齐的有效性来全面评估PCRM。实验结果表明，PCRM通过有效地约束奖励显著提升了对齐性能。

    arXiv:2404.00978v1 Announce Type: new  Abstract: Reinforcement learning with human feedback for aligning large language models (LLMs) trains a reward model typically using ranking loss with comparison pairs.However, the training procedure suffers from an inherent problem: the uncontrolled scaling of reward scores during reinforcement learning due to the lack of constraints while training the reward model.This paper proposes a Prior Constraints-based Reward Model (namely PCRM) training method to mitigate this problem. PCRM incorporates prior constraints, specifically, length ratio and cosine similarity between outputs of each comparison pair, during reward model training to regulate optimization magnitude and control score margins. We comprehensively evaluate PCRM by examining its rank correlation with human preferences and its effectiveness in aligning LLMs via RL. Experimental results demonstrate that PCRM significantly improves alignment performance by effectively constraining reward
    
[^3]: TRABSA：使用基于注意力的BiLSTM和Twitter-RoBERTa进行可解释的推文情感分析

    TRABSA: Interpretable Sentiment Analysis of Tweets using Attention-based BiLSTM and Twitter-RoBERTa

    [https://arxiv.org/abs/2404.00297](https://arxiv.org/abs/2404.00297)

    TRABSA是一个集成了transformer架构、注意力机制和BiLSTM网络的混合框架，利用RoBERTa在大量推特上训练，填补了情感分析领域的差距，实现了94%的准确性和显著的性能提升。

    

    情感分析对于理解公众舆论和消费者行为至关重要。现有模型面临着语言多样性、泛化能力和可解释性方面的挑战。我们提出了TRABSA，这是一个集成了基于transformer的架构、注意力机制和BiLSTM网络的混合框架，旨在解决这些挑战。利用在124M条推文上训练的RoBERTa，我们填补了情感分析基准测试中的差距，确保了最先进的准确性。通过将来自32个国家和美国各州的推文与数据集相结合，我们比较了六种词嵌入技术和三种基于词典的标注技术，并选择了最佳技术以实现最佳情感分析效果。TRABSA以94%的准确性和显著的精确度、召回率和F1得分增益，胜过了传统的机器学习和深度学习模型。在不同数据集上的评估显示了一致的优越性和泛化能力。SHAP和LIME分析提高了可解释性，增强了信心。

    arXiv:2404.00297v1 Announce Type: new  Abstract: Sentiment analysis is crucial for understanding public opinion and consumer behavior. Existing models face challenges with linguistic diversity, generalizability, and explainability. We propose TRABSA, a hybrid framework integrating transformer-based architectures, attention mechanisms, and BiLSTM networks to address this. Leveraging RoBERTa-trained on 124M tweets, we bridge gaps in sentiment analysis benchmarks, ensuring state-of-the-art accuracy. Augmenting datasets with tweets from 32 countries and US states, we compare six word-embedding techniques and three lexicon-based labeling techniques, selecting the best for optimal sentiment analysis. TRABSA outperforms traditional ML and deep learning models with 94% accuracy and significant precision, recall, and F1-score gains. Evaluation across diverse datasets demonstrates consistent superiority and generalizability. SHAP and LIME analyses enhance interpretability, improving confidence i
    
[^4]: 开放域文本到SQL的多跳表检索

    Multi-Hop Table Retrieval for Open-Domain Text-to-SQL

    [https://arxiv.org/abs/2402.10666](https://arxiv.org/abs/2402.10666)

    提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果

    

    开放域文本到SQL是一个重要任务，它从庞大的数据库中检索与问题相关的表，然后生成SQL。然而，现有的单跳检索方法并未关注文本到SQL挑战中的模式链接，这涉及到将问题中的实体与表中实体对齐，主要体现在两个方面：相似的无关实体和领域不匹配实体。因此，我们提出了我们的方法，即带重写和波束搜索的多跳表检索（Murre）。为了减少相似的无关实体的影响，我们的方法侧重于每个跳跃中未检索到的实体，并通过波束搜索考虑排名较低的表。为了缓解领域不匹配实体的限制，Murre基于多个跳跃中检索到的表重写问题，减少与相关表的领域差距。我们在SpiderUnion和BirdUnion+上进行实验，取得了新的最先进结果。

    arXiv:2402.10666v1 Announce Type: new  Abstract: Open-domain text-to-SQL is an important task that retrieves question-relevant tables from massive databases and then generates SQL. However, existing retrieval methods that retrieve in a single hop do not pay attention to the text-to-SQL challenge of schema linking, which is aligning the entities in the question with table entities, reflected in two aspects: similar irrelevant entity and domain mismatch entity. Therefore, we propose our method, the multi-hop table retrieval with rewrite and beam search (Murre). To reduce the effect of the similar irrelevant entity, our method focuses on unretrieved entities at each hop and considers the low-ranked tables by beam search. To alleviate the limitation of domain mismatch entity, Murre rewrites the question based on retrieved tables in multiple hops, decreasing the domain gap with relevant tables. We conduct experiments on SpiderUnion and BirdUnion+, reaching new state-of-the-art results with 
    
[^5]: LLaVA-Docent：利用多模态大型语言模型支持艺术鉴赏教育的教学调优

    LLaVA-Docent: Instruction Tuning with Multimodal Large Language Model to Support Art Appreciation Education

    [https://arxiv.org/abs/2402.06264](https://arxiv.org/abs/2402.06264)

    本研究利用多模态大型语言模型（MLLM）开发了LLaVA-Docent模型，以支持艺术鉴赏教育。通过综述文献和专家咨询，构建了数据框架，并使用该框架生成了虚拟对话数据集用于训练MLLM。该研究对于解决传统艺术鉴赏教育中的资源限制和主流教育中的科学技术工程和数学偏重具有重要意义。

    

    艺术鉴赏对于培养学习者的批判性思维和情感智力至关重要。然而，传统的艺术鉴赏教育常面临艺术资源有限的问题，特别是对于弱势学生，并且在主流教育中过度强调科学技术工程和数学科目。为了应对这些挑战，最近的技术进步为创新解决方案铺平了道路。本研究探索了多模态大型语言模型（MLLM）在艺术鉴赏教育中的应用，重点是开发了LLaVA-Docent模型来利用这些进展。我们的方法包括全面的文献综述和与领域专家的咨询，从而形成了一个强大的数据框架。利用这个框架，我们生成了一个虚拟对话数据集，该数据集被GPT-4利用。这个数据集对于训练MLLM（即LLaVA-Docent）起到了关键作用。六名研究人员进行了定量和定性评估。

    Art appreciation is vital in nurturing critical thinking and emotional intelligence among learners. However, traditional art appreciation education has often been hindered by limited access to art resources, especially for disadvantaged students, and an imbalanced emphasis on STEM subjects in mainstream education. In response to these challenges, recent technological advancements have paved the way for innovative solutions. This study explores the application of multi-modal large language models (MLLMs) in art appreciation education, focusing on developing LLaVA-Docent, a model that leverages these advancements. Our approach involved a comprehensive literature review and consultations with experts in the field, leading to developing a robust data framework. Utilizing this framework, we generated a virtual dialogue dataset that was leveraged by GPT-4. This dataset was instrumental in training the MLLM, named LLaVA-Docent. Six researchers conducted quantitative and qualitative evaluation
    
[^6]: LLMRefine：通过细粒度可操作反馈精确定位和优化大型语言模型

    LLMRefine: Pinpointing and Refining Large Language Models via Fine-Grained Actionable Feedback

    [https://arxiv.org/abs/2311.09336](https://arxiv.org/abs/2311.09336)

    LLMRefine提出了一种细粒度反馈模型来指导大型语言模型定位缺陷并进行优化，在机器翻译、长篇问答和主题总结等任务中取得显著的改进。

    

    最近，大型语言模型（LLM）正在利用人类反馈来提高生成质量。然而，在推断过程中获取人类反馈成本高昂。在这项工作中，我们提出了LLMRefine，一种用于优化推理时间的方法，以改进LLM的输出。其核心思想是利用学习的细粒度反馈模型来准确定位缺陷，并引导LLM进行迭代优化。通过将原始LLM作为编辑建议，LLMRefine通过模拟退火搜索无缺陷文本，权衡探索和开发。我们在三个文本生成任务上进行实验，包括机器翻译，长篇问答（QA）和主题总结。LLMRefine在所有基线方法上一贯表现优异，在翻译任务上取得了高达1.7 MetricX点的改进，在ASQA上为8.1 ROUGE-L，在主题总结上为2.2 ROUGE-L。

    arXiv:2311.09336v2 Announce Type: replace  Abstract: Recent large language models (LLM) are leveraging human feedback to improve their generation quality. However, human feedback is costly to obtain, especially during inference. In this work, we propose LLMRefine, an inference time optimization method to refine LLM's output. The core idea is to use a learned fine-grained feedback model to pinpoint defects and guide LLM to refine them iteratively. Using original LLM as a proposal of edits, LLMRefine searches for defect-less text via simulated annealing, trading off the exploration and exploitation. We conduct experiments on three text generation tasks, including machine translation, long-form question answering (QA), and topical summarization. LLMRefine consistently outperforms all baseline approaches, achieving improvements up to 1.7 MetricX points on translation tasks, 8.1 ROUGE-L on ASQA, 2.2 ROUGE-L on topical summarization.
    
[^7]: 一种针对语言方言的自然语言处理方法：一项调查

    Natural Language Processing for Dialects of a Language: A Survey. (arXiv:2401.05632v1 [cs.CL])

    [http://arxiv.org/abs/2401.05632](http://arxiv.org/abs/2401.05632)

    这项调查研究了自然语言处理中针对方言的方法和问题，强调了方言对于NLP模型性能和语言技术公平性的影响，并提供了关于方言相关任务和语言的全面综述。

    

    最先进的自然语言处理（NLP）模型是在大规模训练语料库上训练的，并在评估数据集上展现出卓越的性能。本调查探讨了这些数据集的一个重要属性：语言方言。考虑到针对方言数据集的NLP模型性能下降及其对语言技术公平性的影响，我们调查了有关方言NLP的过去研究，包括数据集和方法。我们从两个类别的视角描述了各种NLP任务：自然语言理解（NLU）（如方言分类、情感分析、解析和NLU基准测试）和自然语言生成（NLG）（如摘要、机器翻译和对话系统）。这项调查还广泛涵盖了英语、阿拉伯语、德语等多种语言。我们观察到，有关方言的过去NLP工作不止于方言分类，而是...

    State-of-the-art natural language processing (NLP) models are trained on massive training corpora, and report a superlative performance on evaluation datasets. This survey delves into an important attribute of these datasets: the dialect of a language. Motivated by the performance degradation of NLP models for dialectic datasets and its implications for the equity of language technologies, we survey past research in NLP for dialects in terms of datasets, and approaches. We describe a wide range of NLP tasks in terms of two categories: natural language understanding (NLU) (for tasks such as dialect classification, sentiment analysis, parsing, and NLU benchmarks) and natural language generation (NLG) (for summarisation, machine translation, and dialogue systems). The survey is also broad in its coverage of languages which include English, Arabic, German among others. We observe that past work in NLP concerning dialects goes deeper than mere dialect classification, and . This includes ear
    
[^8]: 使用大型语言模型进行产品属性值提取

    Product Attribute Value Extraction using Large Language Models. (arXiv:2310.12537v1 [cs.CL])

    [http://arxiv.org/abs/2310.12537](http://arxiv.org/abs/2310.12537)

    本文研究使用大型语言模型作为预训练的替代方法，解决了传统属性/值提取技术中需要大量训练数据和对未知属性值的挑战问题。

    

    电子商务应用（如面向属性的产品搜索或产品比较）基于结构化的产品描述，如属性/值对。电子商务平台上的供应商不提供结构化的产品描述，而是使用标题或描述来描述产品。为了处理这样的产品，有必要从文本产品属性中提取属性/值对。现有技术中，属性/值提取方法依赖于预训练的语言模型（如BERT）。这些模型在属性/值提取方面存在两个主要缺点：（一）模型需要大量的与任务相关的训练数据；（二）优化后的模型在推广到训练数据中未包含的属性值方面面临挑战。本文探讨了大型语言模型（LLMs）作为训练数据效率高且鲁棒性强的替代方法在属性/值提取中的潜力。我们考虑了托管的LLMs，如GPT-3.5和GPT-4。

    E-commerce applications such as faceted product search or product comparison are based on structured product descriptions like attribute/value pairs. The vendors on e-commerce platforms do not provide structured product descriptions but describe offers using titles or descriptions. To process such offers, it is necessary to extract attribute/value pairs from textual product attributes. State-of-the-art attribute/value extraction techniques rely on pre-trained language models (PLMs), such as BERT. Two major drawbacks of these models for attribute/value extraction are that (i) the models require significant amounts of task-specific training data and (ii) the fine-tuned models face challenges in generalizing to attribute values not included in the training data. This paper explores the potential of large language models (LLMs) as a training data-efficient and robust alternative to PLM-based attribute/value extraction methods. We consider hosted LLMs, such as GPT-3.5 and GPT-4, as well as 
    
[^9]: 图像劫持：对抗性图像能在运行时控制生成模型

    Image Hijacking: Adversarial Images can Control Generative Models at Runtime. (arXiv:2309.00236v1 [cs.LG])

    [http://arxiv.org/abs/2309.00236](http://arxiv.org/abs/2309.00236)

    本研究发现对抗性图像能够在运行时控制生成模型，并提出了通用的方法来创建图像劫持。通过研究三种攻击类型，我们发现这些攻击对最新的视觉语言模型具有高达90％以上的成功率。该研究引发了对基础模型安全性的严重担忧。

    

    基础模型是否能够免受恶意行为者的攻击？本文研究了视觉语言模型（VLM）的图像输入。我们发现了图像劫持，即能够在运行时控制生成模型的对抗性图像。我们引入了一种名为“行为匹配”的通用方法来创建图像劫持，并用它来探索三种类型的攻击：具体字符串攻击可以生成任意被攻击者选择的输出；泄露上下文攻击可以将上下文窗口中的信息泄露到输出中；越狱攻击可以绕过模型的安全训练。我们对基于CLIP和LLaMA-2的最新VLM模型LLaVA-2进行了这些攻击的研究，并发现我们所有的攻击类型成功率均在90％以上。而且，我们的攻击是自动化的，只需要对图像进行小的扰动。这些发现对基础模型的安全性提出了严重的担忧。如果图像劫持与CIFAR-10中的对抗性样本一样难以防御，那么可能需要很多年才能找到解决方案。

    Are foundation models secure from malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control generative models at runtime. We introduce Behavior Matching, a general method for creating image hijacks, and we use it to explore three types of attacks. Specific string attacks generate arbitrary output of the adversary's choosing. Leak context attacks leak information from the context window into the output. Jailbreak attacks circumvent a model's safety training. We study these attacks against LLaVA-2, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all our attack types have above a 90\% success rate. Moreover, our attacks are automated and require only small image perturbations. These findings raise serious concerns about the security of foundation models. If image hijacks are as difficult to defend against as adversarial examples in CIFAR-10, then it might be many years before a s
    
[^10]: PlaSma: 为 (反事实) 计划制定增强过程知识模型的小型语言模型

    PlaSma: Making Small Language Models Better Procedural Knowledge Models for (Counterfactual) Planning. (arXiv:2305.19472v1 [cs.CL])

    [http://arxiv.org/abs/2305.19472](http://arxiv.org/abs/2305.19472)

    PlaSma提出了一种使用小型语言模型进行过程知识和计划能力的新方法，

    

    过程规划是机器的一项重要而又复杂的任务，它将一个高级目标分解为一系列时间顺序的步骤。它需要整合常识知识以推理出常常是反事实的复杂情境，例如 "没有电话时安排医生的约会"。当前的方法使用大型语言模型 (LLM) 取得了令人鼓舞的结果，但受到昂贵的 API 调用和可复现性问题的限制。本文提出使用更小的语言模型来进行规划，我们介绍了 PlaSma，这是一种新的双重方法，使小型语言模型具有过程知识和 (反事实) 计划能力。更具体地说，我们开发了符号过程知识蒸馏来增强小型语言模型中的隐含知识，以及一种推理算法来促进更结构化和准确的推理。此外，我们还引入了一个新的任务，反事实规划。

    Procedural planning, which entails decomposing a high-level goal into a sequence of temporally ordered steps, is an important yet intricate task for machines. It involves integrating common-sense knowledge to reason about complex contextualized situations that are often counterfactual, e.g. "scheduling a doctor's appointment without a phone". While current approaches show encouraging results using large language models (LLMs), they are hindered by drawbacks such as costly API calls and reproducibility issues. In this paper, we advocate planning using smaller language models. We present PlaSma, a novel two-pronged approach to endow small language models with procedural knowledge and (counterfactual) planning capabilities. More concretely, we develop symbolic procedural knowledge distillation to enhance the implicit knowledge in small language models and an inference-time algorithm to facilitate more structured and accurate reasoning. In addition, we introduce a novel task, Counterfactua
    
[^11]: 使用以实体为中心的数据来衡量刻板印象

    Measuring Stereotypes using Entity-Centric Data. (arXiv:2305.09548v1 [cs.CL])

    [http://arxiv.org/abs/2305.09548](http://arxiv.org/abs/2305.09548)

    本文提出并评估了三种新的以实体为中心的方法，展示了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。

    

    刻板印象影响我们如何展示自己和他人，从而影响我们的行为。因此，衡量刻板印象非常重要。最近的研究使用分布语义模型（DSM）（如BERT）中嵌入的投影来进行这些测量。然而，DSMs捕捉到的认知联想不一定与刻板印象的人际性质相关。在这里，我们提出并评估了三种新的以实体为中心的方法，从Twitter和Wikipedia传记中学习刻板印象。通过利用多个短语应用于同一个人的事实来训练模型，扩大了学习联想的人本身中心性。我们证明了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。通过一个案例研究，我们还展示了这些模型对未来计算社会科学问题的实用性。

    Stereotypes inform how we present ourselves and others, and in turn how we behave. They are thus important to measure. Recent work has used projections of embeddings from Distributional Semantic Models (DSMs), such as BERT, to perform these measurements. However, DSMs capture cognitive associations that are not necessarily relevant to the interpersonal nature of stereotyping. Here, we propose and evaluate three novel, entity-centric methods for learning stereotypes from Twitter and Wikipedia biographies. Models are trained by leveraging the fact that multiple phrases are applied to the same person, magnifying the person-centric nature of the learned associations. We show that these models outperform existing approaches to stereotype measurement with respect to 1) predicting which identities people apply to themselves and others, and 2) quantifying stereotypes on salient social dimensions (e.g. gender). Via a case study, we also show the utility of these models for future questions in c
    
[^12]: 关于大型语言模型的创造性研究

    On the Creativity of Large Language Models. (arXiv:2304.00008v1 [cs.AI])

    [http://arxiv.org/abs/2304.00008](http://arxiv.org/abs/2304.00008)

    这篇论文探讨了大型语言模型的创造性问题，分析了与之相关的机器创造性的难点和易点，并重点分析了这些技术在创意产业中的社会影响。

    

    大型语言模型(LLMs)正在颠覆人工智能的多个领域。其中最显著的应用之一是创作，例如诗歌或故事：生成的输出通常具有惊人的质量。但是，一个自然的问题是：LLMs真的可以被认为是创造性的吗？在本文中，我们首先通过创造性理论的角度分析了LLMs的发展，探讨了关键的未解决问题和挑战。然后，我们在与LLMs相关的机器创造性方面确定了一组“易”和“难”问题，并对其进行了讨论。最后，我们分析了这些技术在创意产业中的社会影响。

    Large Language Models (LLMs) are revolutionizing several areas of Artificial Intelligence. One of the most remarkable applications is creative writing, e.g., poetry or storytelling: the generated outputs are often of astonishing quality. However, a natural question arise: can LLMs really be considered creative? In this article we firstly analyze the development of LLMs under the lens of creativity theories, investigating the key open questions and challenges. Then, we identify a set of "easy" and "hard" problems in machine creativity, discussing them in relation to LLMs. Finally, we analyze the societal impact of these technologies with a particular focus on the creative industries.
    

