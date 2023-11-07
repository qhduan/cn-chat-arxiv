# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Large Language Model for Recommender Systems.](http://arxiv.org/abs/2311.01343) | 本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。 |
| [^2] | [LLMRec: Large Language Models with Graph Augmentation for Recommendation.](http://arxiv.org/abs/2311.00423) | LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。 |
| [^3] | [Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language.](http://arxiv.org/abs/2310.13540) | 本研究提出了一种新颖的统一预训练语言模型增强顺序推荐方法（UPSR），旨在构建一个统一的预训练推荐模型用于多领域推荐任务。研究者设计了五个关键指标来指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。 |
| [^4] | [Robust Training for Conversational Question Answering Models with Reinforced Reformulation Generation.](http://arxiv.org/abs/2310.13505) | 这项研究提出了一种新的框架REIGN，通过生成训练问题的改写，并使用深度强化学习来指导对话问答模型，增加模型对表面形式变化的鲁棒性，同时在不同的基准上进行零-shot应用。 |
| [^5] | [Recommender Systems with Generative Retrieval.](http://arxiv.org/abs/2305.05065) | 本文提出了一种新型的生成式检索模型，将检索和生成组合在一起以产生推荐。 |
| [^6] | [Bi-directional Training for Composed Image Retrieval via Text Prompt Learning.](http://arxiv.org/abs/2303.16604) | 本文提出了一种基于文本提示学习和双向训练的组成图像检索方法，可以应用于现有的体系结构，并且在修改文本存在噪声或歧义的情况下特别有效。 |

# 详细

[^1]: 协同大型语言模型用于推荐系统

    Collaborative Large Language Model for Recommender Systems. (arXiv:2311.01343v1 [cs.IR])

    [http://arxiv.org/abs/2311.01343](http://arxiv.org/abs/2311.01343)

    本研究提出了CLLM4Rec，首个将大型语言模型与推荐系统的 ID 模式紧密集成的协同推荐算法，旨在解决语义差距、虚假相关和低效推荐等问题。通过扩展预训练语言模型的词汇表，并引入软硬提示策略，该算法能够准确地模拟用户和项目的协同与内容语义。

    

    最近，越来越多的人对基于预训练的大型语言模型（LLM）开发下一代推荐系统（RS）产生了兴趣，充分利用其编码知识和推理能力。然而，自然语言与推荐任务之间的语义差距仍未得到很好的解决，导致一些问题，如虚假相关的用户/项目描述符、对用户/项目内容的低效语言建模以及通过自动回归进行低效的推荐等。在本文中，我们提出了CLLM4Rec，这是第一个紧密集成LLM范式和RS的ID范式的生成RS，旨在同时解决上述挑战。我们首先使用用户/项目ID标记扩展了预训练LLM的词汇表，以忠实地模拟用户/项目的协同和内容语义。因此，在预训练阶段，提出了一种新颖的软硬提示策略，通过语言建模有效地学习用户/项目的协同/内容标记嵌入。

    Recently, there is a growing interest in developing next-generation recommender systems (RSs) based on pretrained large language models (LLMs), fully utilizing their encoded knowledge and reasoning ability. However, the semantic gap between natural language and recommendation tasks is still not well addressed, leading to multiple issues such as spuriously-correlated user/item descriptors, ineffective language modeling on user/item contents, and inefficient recommendations via auto-regression, etc. In this paper, we propose CLLM4Rec, the first generative RS that tightly integrates the LLM paradigm and ID paradigm of RS, aiming to address the above challenges simultaneously. We first extend the vocabulary of pretrained LLMs with user/item ID tokens to faithfully model the user/item collaborative and content semantics. Accordingly, in the pretraining stage, a novel soft+hard prompting strategy is proposed to effectively learn user/item collaborative/content token embeddings via language m
    
[^2]: LLMRec: 使用图增强的大型语言模型用于推荐系统

    LLMRec: Large Language Models with Graph Augmentation for Recommendation. (arXiv:2311.00423v1 [cs.IR])

    [http://arxiv.org/abs/2311.00423](http://arxiv.org/abs/2311.00423)

    LLMRec是一种利用大型语言模型的图增强策略来改进推荐系统的新方法，它解决了数据稀缺性和附加信息引入副作用的问题，通过加强交互边、增强物品节点属性理解和进行用户节点建模来提高推荐性能。

    

    数据稀疏性一直是推荐系统中的一个挑战，之前的研究尝试通过引入附加信息来解决这个问题。然而，这种方法往往会带来噪声、可用性问题和数据质量低下等副作用，从而影响对用户偏好的准确建模，进而对推荐性能产生不利影响。鉴于大型语言模型（LLM）在知识库和推理能力方面的最新进展，我们提出了一个名为LLMRec的新框架，它通过采用三种简单而有效的基于LLM的图增强策略来增强推荐系统。我们的方法利用在线平台（如Netflix，MovieLens）中丰富的内容，在三个方面增强交互图：（i）加强用户-物品交互边，（ii）增强对物品节点属性的理解，（iii）进行用户节点建模，直观地表示用户特征。

    The problem of data sparsity has long been a challenge in recommendation systems, and previous studies have attempted to address this issue by incorporating side information. However, this approach often introduces side effects such as noise, availability issues, and low data quality, which in turn hinder the accurate modeling of user preferences and adversely impact recommendation performance. In light of the recent advancements in large language models (LLMs), which possess extensive knowledge bases and strong reasoning capabilities, we propose a novel framework called LLMRec that enhances recommender systems by employing three simple yet effective LLM-based graph augmentation strategies. Our approach leverages the rich content available within online platforms (e.g., Netflix, MovieLens) to augment the interaction graph in three ways: (i) reinforcing user-item interaction egde, (ii) enhancing the understanding of item node attributes, and (iii) conducting user node profiling, intuiti
    
[^3]: 全面将多领域预训练推荐建模为语言

    Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language. (arXiv:2310.13540v1 [cs.IR])

    [http://arxiv.org/abs/2310.13540](http://arxiv.org/abs/2310.13540)

    本研究提出了一种新颖的统一预训练语言模型增强顺序推荐方法（UPSR），旨在构建一个统一的预训练推荐模型用于多领域推荐任务。研究者设计了五个关键指标来指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。

    

    随着预训练语言模型（PLM）在各种自然语言处理任务中的广泛应用，先驱性工作试图探索将PLM中的通用文本信息与用户历史行为序列中的个性化行为信息相结合，以增强顺序推荐（SR）。然而，尽管输入格式和任务目标存在共性，行为和文本信息之间存在巨大差距，这阻碍了将SR作为语言建模完全建模。为了填补这一差距，我们提出了一种新颖的统一预训练语言模型增强顺序推荐（UPSR）方法，旨在构建一个统一的预训练推荐模型用于多领域推荐任务。我们正式设计了自然性、领域一致性、信息性、噪声和模糊性以及文本长度等五个关键指标，分别用于指导预训练和微调过程中的文本->物品适应和行为序列->文本序列适应。

    With the thriving of pre-trained language model (PLM) widely verified in various of NLP tasks, pioneer efforts attempt to explore the possible cooperation of the general textual information in PLM with the personalized behavioral information in user historical behavior sequences to enhance sequential recommendation (SR). However, despite the commonalities of input format and task goal, there are huge gaps between the behavioral and textual information, which obstruct thoroughly modeling SR as language modeling via PLM. To bridge the gap, we propose a novel Unified pre-trained language model enhanced sequential recommendation (UPSR), aiming to build a unified pre-trained recommendation model for multi-domain recommendation tasks. We formally design five key indicators, namely naturalness, domain consistency, informativeness, noise & ambiguity, and text length, to guide the text->item adaptation and behavior sequence->text sequence adaptation differently for pre-training and fine-tuning 
    
[^4]: 具有强化改写生成的对话问答模型的鲁棒训练

    Robust Training for Conversational Question Answering Models with Reinforced Reformulation Generation. (arXiv:2310.13505v1 [cs.CL])

    [http://arxiv.org/abs/2310.13505](http://arxiv.org/abs/2310.13505)

    这项研究提出了一种新的框架REIGN，通过生成训练问题的改写，并使用深度强化学习来指导对话问答模型，增加模型对表面形式变化的鲁棒性，同时在不同的基准上进行零-shot应用。

    

    知识图谱（KG）上的对话问答（ConvQA）模型通常在黄金QA对的基准上进行训练和测试。这意味着训练仅限于在相应数据集中见到的表面形式，评估仅针对一小部分问题。通过我们的提出的框架REIGN，我们采取了几个步骤来解决这个受限的学习设置。首先，我们系统地生成训练问题的改写，以提高模型对表面形式变化的鲁棒性。这是一个特别具有挑战性的问题，因为这些问题的不完整性。其次，我们使用深度强化学习将ConvQA模型引导到更高的性能，只提供那些有助于提高回答质量的改写。第三，我们展示了在一个基准上训练主要模型组件并将其零-shot应用于另一个的可行性。最后，为了对训练模型的鲁棒性进行严格评估，我们使用和重新配置初始的改写、测试语料。

    Models for conversational question answering (ConvQA) over knowledge graphs (KGs) are usually trained and tested on benchmarks of gold QA pairs. This implies that training is limited to surface forms seen in the respective datasets, and evaluation is on a small set of held-out questions. Through our proposed framework REIGN, we take several steps to remedy this restricted learning setup. First, we systematically generate reformulations of training questions to increase robustness of models to surface form variations. This is a particularly challenging problem, given the incomplete nature of such questions. Second, we guide ConvQA models towards higher performance by feeding it only those reformulations that help improve their answering quality, using deep reinforcement learning. Third, we demonstrate the viability of training major model components on one benchmark and applying them zero-shot to another. Finally, for a rigorous evaluation of robustness for trained models, we use and re
    
[^5]: 生成式检索推荐系统

    Recommender Systems with Generative Retrieval. (arXiv:2305.05065v1 [cs.IR])

    [http://arxiv.org/abs/2305.05065](http://arxiv.org/abs/2305.05065)

    本文提出了一种新型的生成式检索模型，将检索和生成组合在一起以产生推荐。

    

    现代推荐系统使用大规模检索模型进行推荐，包括两个阶段：训练双编码模型将查询和候选项嵌入到相同的空间中，然后使用近似最近邻搜索来选择给定查询嵌入的顶部候选项。本文提出了一种新的单阶段范例：生成式检索模型，该模型通过自回归方式在一个阶段中解码目标候选项的标识符。为此，我们不是为每个项目分配随机生成的原子ID，而是生成语义ID：每个项目的语义有意义的元组编码词，它作为其唯一标识符。我们使用称为RQ-VAE的分层方法生成这些编码词。一旦我们对所有项目都有了语义ID，就会训练基于Transformer的序列到序列模型来预测下一个项目的语义ID。由于这个模型以自回归的方式直接预测标识下一个项的编码词元组，因此它可以将检索和生成组合在一起以产生推荐。

    Modern recommender systems leverage large-scale retrieval models consisting of two stages: training a dual-encoder model to embed queries and candidates in the same space, followed by an Approximate Nearest Neighbor (ANN) search to select top candidates given a query's embedding. In this paper, we propose a new single-stage paradigm: a generative retrieval model which autoregressively decodes the identifiers for the target candidates in one phase. To do this, instead of assigning randomly generated atomic IDs to each item, we generate Semantic IDs: a semantically meaningful tuple of codewords for each item that serves as its unique identifier. We use a hierarchical method called RQ-VAE to generate these codewords. Once we have the Semantic IDs for all the items, a Transformer based sequence-to-sequence model is trained to predict the Semantic ID of the next item. Since this model predicts the tuple of codewords identifying the next item directly in an autoregressive manner, it can be c
    
[^6]: 基于文本提示学习和双向训练的组成图像检索方法

    Bi-directional Training for Composed Image Retrieval via Text Prompt Learning. (arXiv:2303.16604v1 [cs.CV])

    [http://arxiv.org/abs/2303.16604](http://arxiv.org/abs/2303.16604)

    本文提出了一种基于文本提示学习和双向训练的组成图像检索方法，可以应用于现有的体系结构，并且在修改文本存在噪声或歧义的情况下特别有效。

    

    组成图像检索是根据包含参考图像和描述所需更改的修改文本的多模态用户查询来搜索目标图像的方法。现有的解决这个具有挑战性的任务的方法学习从（参考图像，修改文本）对到图像嵌入的映射，然后将其与大型图像语料库进行匹配。本文提出了一种双向训练方案，利用了这种反向查询，并可应用于现有的组成图像检索体系结构。为了编码双向查询，我们在修改文本前面添加一个可学习的令牌，指定查询的方向，然后微调文本嵌入模块的参数。我们没有对网络架构进行其他更改。在两个标准数据集上的实验表明，双向训练在提高组成图像检索性能方面是有效的，特别是在修改文本存在噪声或歧义的情况下。

    Composed image retrieval searches for a target image based on a multi-modal user query comprised of a reference image and modification text describing the desired changes. Existing approaches to solving this challenging task learn a mapping from the (reference image, modification text)-pair to an image embedding that is then matched against a large image corpus. One area that has not yet been explored is the reverse direction, which asks the question, what reference image when modified as describe by the text would produce the given target image? In this work we propose a bi-directional training scheme that leverages such reversed queries and can be applied to existing composed image retrieval architectures. To encode the bi-directional query we prepend a learnable token to the modification text that designates the direction of the query and then finetune the parameters of the text embedding module. We make no other changes to the network architecture. Experiments on two standard datas
    

