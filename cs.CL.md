# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation](https://arxiv.org/abs/2404.01129) | 将抽象意义表示结合到LLMs中，提出了一个简单有效的框架用于改善开放领域对话评估 |
| [^2] | [Improving Sampling Methods for Fine-tuning SentenceBERT in Text Streams](https://arxiv.org/abs/2403.15455) | 本研究旨在解决概念漂移问题，通过探索七种文本采样方法的有效性，精细调整语言模型，从而减轻性能下降。 |
| [^3] | [No Language is an Island: Unifying Chinese and English in Financial Large Language Models, Instruction Data, and Benchmarks](https://arxiv.org/abs/2403.06249) | ICE-PIXIU模型将中文和英文金融分析统一，通过整合多种任务和数据集提升双语金融建模的效果。 |
| [^4] | [Apollo: Lightweight Multilingual Medical LLMs towards Democratizing Medical AI to 6B People](https://arxiv.org/abs/2403.03640) | Apollo项目开发了多语言医学LLMs，创建了全球人口61亿的医学数据集，并发布了各种尺寸的最佳性能模型，其中Apollo-7B是最先进的多语言医学LLMs，可改善更大模型的多语言医学能力。 |
| [^5] | [What Do Language Models Hear? Probing for Auditory Representations in Language Models](https://arxiv.org/abs/2402.16998) | 通过训练一个线性探针，将语言模型中的文本表示和预训练音频模型中的声音表示联系在一起，研究发现尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识有着基于实质的编码。 |
| [^6] | [ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages](https://arxiv.org/abs/2402.10753) | ToolSword提出了一个专门用于细致调查大型语言模型在工具学习中安全问题的全面框架，揭示了在工具学习中持久存在的安全挑战。 |
| [^7] | [Multi-Hop Table Retrieval for Open-Domain Text-to-SQL](https://arxiv.org/abs/2402.10666) | 提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果 |
| [^8] | [A Data Generation Perspective to the Mechanism of In-Context Learning](https://arxiv.org/abs/2402.02212) | 本文从数据生成的角度重新解释了In-Context Learning（ICL）的机制，并探讨了流行的技术解决方案的潜在应用。对不同解决方案的优劣进行了全面研究，强调了其中的不足之处。 |
| [^9] | [AI-as-exploration: Navigating intelligence space](https://arxiv.org/abs/2401.07964) | 这篇论文阐述了一个被忽视但重要的科学角色，即AI作为探索。它强调通过创建和研究智能系统来揭示可能与人类和动物的智能形式不同的候选构建模块。论文通过讨论人类和大型语言模型在组合新颖和创造性概念方面的能力，说明了AI作为探索的价值。 |
| [^10] | [Transformers and Cortical Waves: Encoders for Pulling In Context Across Time.](http://arxiv.org/abs/2401.14267) | 这项研究探讨了transformer网络和大脑皮层波之间的相似性，并指出了皮层波在提取感觉输入序列中的时间上下文方面的潜在应用。 |
| [^11] | [Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation.](http://arxiv.org/abs/2310.02304) | 本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。 |
| [^12] | [MedEdit: Model Editing for Medical Question Answering with External Knowledge Bases.](http://arxiv.org/abs/2309.16035) | 该论文研究了利用外部知识库进行医学问答的模型编辑方法，通过提取医学事实并将其融入到语言模型的查询提示中，显著提高了医学问答的准确率。 |
| [^13] | [Self-Supervised Multimodal Learning: A Survey.](http://arxiv.org/abs/2304.01008) | 自监督多模态学习是一项旨在解决多模态数据中的自监督学习挑战的研究方向。它通过学习来自原始多模态数据中的表示，并解决了没有标签的多模态数据学习、不同模态的融合和不对齐数据学习等问题。 |

# 详细

[^1]: 结构化信息很重要：将抽象意义表示引入LLMs以改善开放领域对话评估

    Structured Information Matters: Incorporating Abstract Meaning Representation into LLMs for Improved Open-Domain Dialogue Evaluation

    [https://arxiv.org/abs/2404.01129](https://arxiv.org/abs/2404.01129)

    将抽象意义表示结合到LLMs中，提出了一个简单有效的框架用于改善开放领域对话评估

    

    arXiv:2404.01129v1 公告类型：新的 摘要：自动的开放领域对话评估已经引起越来越多的关注。可训练的评估指标通常是通过训练具有真正正例和随机选择的负例回复来训练的，导致它们倾向于将更高内容相似性的回复分配更高的得分给定一个上下文。然而，对抗性的负面回复具有与上下文高内容相似性，同时在语义上不同。因此，现有的评估指标不足以评估这类回复，导致与人类判断之间的相关性较低。虽然最近的研究已经显示出在利用大型语言模型（LLMs）进行开放领域对话评估方面有一定效果，但它们仍然在有效处理对抗性负面示例方面遇到挑战。在本文中，我们提出了一个简单而有效的框架用于开放领域对话评估，它结合了领域特定的语言模型（SLMs）。

    arXiv:2404.01129v1 Announce Type: new  Abstract: Automatic open-domain dialogue evaluation has attracted increasing attention. Trainable evaluation metrics are commonly trained with true positive and randomly selected negative responses, resulting in a tendency for them to assign a higher score to the responses that share higher content similarity with a given context. However, adversarial negative responses possess high content similarity with the contexts whilst being semantically different. Therefore, existing evaluation metrics are not robust enough to evaluate such responses, resulting in low correlations with human judgments. While recent studies have shown some efficacy in utilizing Large Language Models (LLMs) for open-domain dialogue evaluation, they still encounter challenges in effectively handling adversarial negative examples. In this paper, we propose a simple yet effective framework for open-domain dialogue evaluation, which combines domain-specific language models (SLMs
    
[^2]: 改进文本流中用于微调SentenceBERT的采样方法

    Improving Sampling Methods for Fine-tuning SentenceBERT in Text Streams

    [https://arxiv.org/abs/2403.15455](https://arxiv.org/abs/2403.15455)

    本研究旨在解决概念漂移问题，通过探索七种文本采样方法的有效性，精细调整语言模型，从而减轻性能下降。

    

    互联网上文本数据的激增为机构和公司提供了一个独特的机会，可以监测公众对其服务和产品的意见。考虑到这些数据的快速生成，处理依次到达、潜在无限的文本流的文本流挖掘设置通常比传统的批量学习更合适。虽然预训练语言模型通常因其在流式内容中高质量的文本向量化能力而被广泛采用，但它们在适应概念漂移（数据分布随时间发生变化，从而对模型性能产生负面影响的现象）方面面临挑战。本研究解决了概念漂移问题，探讨了七种文本采样方法对精心微调语言模型的效果，从而减轻性能下降。我们准确评估了这些方法对使用四种不同方式进行微调的SBERT模型的影响。

    arXiv:2403.15455v1 Announce Type: new  Abstract: The proliferation of textual data on the Internet presents a unique opportunity for institutions and companies to monitor public opinion about their services and products. Given the rapid generation of such data, the text stream mining setting, which handles sequentially arriving, potentially infinite text streams, is often more suitable than traditional batch learning. While pre-trained language models are commonly employed for their high-quality text vectorization capabilities in streaming contexts, they face challenges adapting to concept drift - the phenomenon where the data distribution changes over time, adversely affecting model performance. Addressing the issue of concept drift, this study explores the efficacy of seven text sampling methods designed to selectively fine-tune language models, thereby mitigating performance degradation. We precisely assess the impact of these methods on fine-tuning the SBERT model using four differ
    
[^3]: 没有孤岛语言:统一中英文金融大型语言模型、指导数据和基准

    No Language is an Island: Unifying Chinese and English in Financial Large Language Models, Instruction Data, and Benchmarks

    [https://arxiv.org/abs/2403.06249](https://arxiv.org/abs/2403.06249)

    ICE-PIXIU模型将中文和英文金融分析统一，通过整合多种任务和数据集提升双语金融建模的效果。

    

    随着大型语言模型（LLM）的发展显著推动了金融分析，但它们的应用主要局限在单一语言领域，未充分开发中英文双语能力的潜力。为弥合这一鸿沟，我们引入 ICE-PIXIU，无缝融合 ICE-INTENT 模型和 ICE-FLARE 双语金融分析基准。ICE-PIXIU 独特地整合了一系列中文任务，以及翻译和原始英文数据集，丰富了双语金融建模的广度和深度。它提供了对多种模型变体的无限访问权限，一个包含多种跨语言和多模态指导数据的大量编译，以及一个具有专家注释的评估基准，包括 10 个 NLP 任务，20 个双语专用任务，共计1,185k 数据集。我们的彻底评估强调了将这些双语数据集纳入的优势，尤其在t

    arXiv:2403.06249v1 Announce Type: cross  Abstract: While the progression of Large Language Models (LLMs) has notably propelled financial analysis, their application has largely been confined to singular language realms, leaving untapped the potential of bilingual Chinese-English capacity. To bridge this chasm, we introduce ICE-PIXIU, seamlessly amalgamating the ICE-INTENT model and ICE-FLARE benchmark for bilingual financial analysis. ICE-PIXIU uniquely integrates a spectrum of Chinese tasks, alongside translated and original English datasets, enriching the breadth and depth of bilingual financial modeling. It provides unrestricted access to diverse model variants, a substantial compilation of diverse cross-lingual and multi-modal instruction data, and an evaluation benchmark with expert annotations, comprising 10 NLP tasks, 20 bilingual specific tasks, totaling 1,185k datasets. Our thorough evaluation emphasizes the advantages of incorporating these bilingual datasets, especially in t
    
[^4]: Apollo：轻量级多语言医学LLMs：让医学人工智能普惠60亿人

    Apollo: Lightweight Multilingual Medical LLMs towards Democratizing Medical AI to 6B People

    [https://arxiv.org/abs/2403.03640](https://arxiv.org/abs/2403.03640)

    Apollo项目开发了多语言医学LLMs，创建了全球人口61亿的医学数据集，并发布了各种尺寸的最佳性能模型，其中Apollo-7B是最先进的多语言医学LLMs，可改善更大模型的多语言医学能力。

    

    尽管全球医学知识的庞大存储库主要是以英语为主，但在传递量身定制医疗服务方面，本地语言对于在医疗资源有限的地区尤为重要。为了将医学人工智能的进展扩展到更广泛的人群，我们旨在开发涵盖全球61亿人口的六种最常用语言的医学LLMs。这一努力最终促成了ApolloCorpora多语言医学数据集和XMedBench基准的创建。在多语言医学基准测试中，发布的Apollo模型，在各种相对较小尺寸（即0.5B、1.8B、2B、6B和7B）上取得了与同等大小模型最佳性能。特别地，Apollo-7B是迄今为止达到70B的最先进的多语言医学LLMs。此外，这些轻量级模型可用于在不需要微调的情况下改进较大模型的多语言医学能力。

    arXiv:2403.03640v1 Announce Type: cross  Abstract: Despite the vast repository of global medical knowledge predominantly being in English, local languages are crucial for delivering tailored healthcare services, particularly in areas with limited medical resources. To extend the reach of medical AI advancements to a broader population, we aim to develop medical LLMs across the six most widely spoken languages, encompassing a global population of 6.1 billion. This effort culminates in the creation of the ApolloCorpora multilingual medical dataset and the XMedBench benchmark. In the multilingual medical benchmark, the released Apollo models, at various relatively-small sizes (i.e., 0.5B, 1.8B, 2B, 6B, and 7B), achieve the best performance among models of equivalent size. Especially, Apollo-7B is the state-of-the-art multilingual medical LLMs up to 70B. Additionally, these lite models could be used to improve the multi-lingual medical capabilities of larger models without fine-tuning in a
    
[^5]: 语言模型听到了什么？探究语言模型中的听觉表征

    What Do Language Models Hear? Probing for Auditory Representations in Language Models

    [https://arxiv.org/abs/2402.16998](https://arxiv.org/abs/2402.16998)

    通过训练一个线性探针，将语言模型中的文本表示和预训练音频模型中的声音表示联系在一起，研究发现尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识有着基于实质的编码。

    

    这项工作探讨了语言模型是否对物体的声音具有含义深刻且基于实质的表征。我们学习了一个线性探针，通过一个预训练的音频模型给出一个对象的声音表示，从而在给定与该对象相关的音频片段的情况下检索出该对象的正确文本表示。这个探针是通过对比损失进行训练的，推动对象的语言表示和声音表示彼此接近。在训练之后，我们测试了探针对于一些在训练中没有见过的对象的泛化能力。在不同的语言模型和音频模型中，我们发现在许多情况下探针的泛化能力超过了随机猜测的水平，这表明尽管仅在原始文本上进行训练，语言模型对于一些对象的声音知识具有基于实质的编码。

    arXiv:2402.16998v1 Announce Type: cross  Abstract: This work explores whether language models encode meaningfully grounded representations of sounds of objects. We learn a linear probe that retrieves the correct text representation of an object given a snippet of audio related to that object, where the sound representation is given by a pretrained audio model. This probe is trained via a contrastive loss that pushes the language representations and sound representations of an object to be close to one another. After training, the probe is tested on its ability to generalize to objects that were not seen during training. Across different language models and audio models, we find that the probe generalization is above chance in many cases, indicating that despite being trained only on raw text, language models encode grounded knowledge of sounds for some objects.
    
[^6]: ToolSword：揭示大型语言模型在工具学习中的安全问题跨三个阶段

    ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages

    [https://arxiv.org/abs/2402.10753](https://arxiv.org/abs/2402.10753)

    ToolSword提出了一个专门用于细致调查大型语言模型在工具学习中安全问题的全面框架，揭示了在工具学习中持久存在的安全挑战。

    

    arXiv:2402.10753v1 公告类型：跨领域 抽象：工具学习被广泛认为是在现实场景中部署大型语言模型（LLMs）的基础方法。尽管当前研究主要强调利用工具来增强LLMs，但它经常忽视与其应用相关的新兴安全考虑。为填补这一空白，我们提出了$ToolSword$，这是一个致力于细致调查LLMs在工具学习中安全问题的全面框架。具体来说，ToolSword勾画了LLMs在工具学习中的六个安全场景，包括输入阶段的$恶意$ $查询$和$越狱$ $攻击$，执行阶段的$噪声$ $误导$和$风险$ $线索$，以及输出阶段的$有害$ $反馈$和$错误$ $冲突$。对11个开源和闭源LLMs进行的实验表明，在工具学习中存在持久的安全挑战，如处理有害查询、使用风险工具和提供有害反馈。

    arXiv:2402.10753v1 Announce Type: cross  Abstract: Tool learning is widely acknowledged as a foundational approach or deploying large language models (LLMs) in real-world scenarios. While current research primarily emphasizes leveraging tools to augment LLMs, it frequently neglects emerging safety considerations tied to their application. To fill this gap, we present $ToolSword$, a comprehensive framework dedicated to meticulously investigating safety issues linked to LLMs in tool learning. Specifically, ToolSword delineates six safety scenarios for LLMs in tool learning, encompassing $malicious$ $queries$ and $jailbreak$ $attacks$ in the input stage, $noisy$ $misdirection$ and $risky$ $cues$ in the execution stage, and $harmful$ $feedback$ and $error$ $conflicts$ in the output stage. Experiments conducted on 11 open-source and closed-source LLMs reveal enduring safety challenges in tool learning, such as handling harmful queries, employing risky tools, and delivering detrimental feedb
    
[^7]: 开放域文本到SQL的多跳表检索

    Multi-Hop Table Retrieval for Open-Domain Text-to-SQL

    [https://arxiv.org/abs/2402.10666](https://arxiv.org/abs/2402.10666)

    提出了一种多跳表检索方法，通过重写问题和波束搜索来减少相似无关实体的影响，并通过多跳检索中重新编写问题来缓解领域不匹配实体的限制，取得了新的最先进结果

    

    开放域文本到SQL是一个重要任务，它从庞大的数据库中检索与问题相关的表，然后生成SQL。然而，现有的单跳检索方法并未关注文本到SQL挑战中的模式链接，这涉及到将问题中的实体与表中实体对齐，主要体现在两个方面：相似的无关实体和领域不匹配实体。因此，我们提出了我们的方法，即带重写和波束搜索的多跳表检索（Murre）。为了减少相似的无关实体的影响，我们的方法侧重于每个跳跃中未检索到的实体，并通过波束搜索考虑排名较低的表。为了缓解领域不匹配实体的限制，Murre基于多个跳跃中检索到的表重写问题，减少与相关表的领域差距。我们在SpiderUnion和BirdUnion+上进行实验，取得了新的最先进结果。

    arXiv:2402.10666v1 Announce Type: new  Abstract: Open-domain text-to-SQL is an important task that retrieves question-relevant tables from massive databases and then generates SQL. However, existing retrieval methods that retrieve in a single hop do not pay attention to the text-to-SQL challenge of schema linking, which is aligning the entities in the question with table entities, reflected in two aspects: similar irrelevant entity and domain mismatch entity. Therefore, we propose our method, the multi-hop table retrieval with rewrite and beam search (Murre). To reduce the effect of the similar irrelevant entity, our method focuses on unretrieved entities at each hop and considers the low-ranked tables by beam search. To alleviate the limitation of domain mismatch entity, Murre rewrites the question based on retrieved tables in multiple hops, decreasing the domain gap with relevant tables. We conduct experiments on SpiderUnion and BirdUnion+, reaching new state-of-the-art results with 
    
[^8]: 从数据生成的角度对In-Context Learning机制的解释

    A Data Generation Perspective to the Mechanism of In-Context Learning

    [https://arxiv.org/abs/2402.02212](https://arxiv.org/abs/2402.02212)

    本文从数据生成的角度重新解释了In-Context Learning（ICL）的机制，并探讨了流行的技术解决方案的潜在应用。对不同解决方案的优劣进行了全面研究，强调了其中的不足之处。

    

    In-Context Learning（ICL）使大型语言模型（LLM）能够在上下文中学习，在只有少量上下文示例的情况下实现下游泛化，而无需梯度更新。尽管有鼓舞人心的实证成功，ICL的基本机制仍然不清楚，现有研究提供了各种不同观点的理解。这些研究提出了基于直觉和临时技术解决方案来解释ICL，呈现出了一条模糊的路线图。在本文中，我们利用数据生成的视角重新解释最近的研究成果，并展示了流行技术解决方案的潜在广泛应用，从而接近一个系统的角度。我们严格采用技能学习和技能识别的概念定义。它们之间的区别在于技能学习可以从上下文数据中学习新的数据生成函数。我们还对不同解决方案的优势和弱点进行了全面的研究，并强调了其中的不足之处。

    In-Context Learning (ICL) empowers Large Language Models (LLMs) with the capacity to learn in context, achieving downstream generalization without gradient updates but with a few in-context examples. Despite the encouraging empirical success, the underlying mechanism of ICL remains unclear, and existing research offers various viewpoints of understanding. These studies propose intuition-driven and ad-hoc technical solutions for interpreting ICL, illustrating an ambiguous road map. In this paper, we leverage a data generation perspective to reinterpret recent efforts and demonstrate the potential broader usage of popular technical solutions, approaching a systematic angle. For a conceptual definition, we rigorously adopt the terms of skill learning and skill recognition. The difference between them is skill learning can learn new data generation functions from in-context data. We also provide a comprehensive study on the merits and weaknesses of different solutions, and highlight the un
    
[^9]: AI作为探索：导航智能空间

    AI-as-exploration: Navigating intelligence space

    [https://arxiv.org/abs/2401.07964](https://arxiv.org/abs/2401.07964)

    这篇论文阐述了一个被忽视但重要的科学角色，即AI作为探索。它强调通过创建和研究智能系统来揭示可能与人类和动物的智能形式不同的候选构建模块。论文通过讨论人类和大型语言模型在组合新颖和创造性概念方面的能力，说明了AI作为探索的价值。

    

    人工智能是一个拥有许多生命的领域，这个术语已经包含了一系列科学和商业努力。在这篇论文中，我阐述了人工智能具有一个被忽视但十分重要的科学角色，即“AI作为探索”。AI作为探索的基本思想是创建和研究能够揭示智能候选构建模块的系统，这些模块可能不同于我们熟悉的人类和动物智能形式。换句话说，我认为人工智能是探索智能空间，即可能的智能系统空间，的最佳工具之一。我通过关注一个具体的案例研究，即人类和大型语言模型在组合新颖和创造性概念方面的能力，来说明AI作为探索的价值。我展示了尽管后者在这样的任务中表现出人类水平的准确性，但很可能以根本不同的方式解决这个问题。

    Artificial Intelligence is a field that lives many lives, and the term has come to encompass a motley collection of scientific and commercial endeavours. In this paper, I articulate the contours of a rather neglected but central scientific role that AI has to play, which I dub `AI-as-exploration'.The basic thrust of AI-as-exploration is that of creating and studying systems that can reveal candidate building blocks of intelligence that may differ from the forms of human and animal intelligence we are familiar with. In other words, I suggest that AI is one of the best tools we have for exploring intelligence space, namely the space of possible intelligent systems. I illustrate the value of AI-as-exploration by focusing on a specific case study, i.e., recent work on the capacity to combine novel and invented concepts in humans and Large Language Models. I show that the latter, despite showing human-level accuracy in such a task, most probably solve it in ways radically different, but no 
    
[^10]: Transformers和大脑皮层波：在时间上传递上下文的编码器

    Transformers and Cortical Waves: Encoders for Pulling In Context Across Time. (arXiv:2401.14267v1 [cs.CL])

    [http://arxiv.org/abs/2401.14267](http://arxiv.org/abs/2401.14267)

    这项研究探讨了transformer网络和大脑皮层波之间的相似性，并指出了皮层波在提取感觉输入序列中的时间上下文方面的潜在应用。

    

    类似ChatGPT和其他大语言模型（LLM）的transformer网络的能力已经引起了世界的关注。它们的性能依赖于将完整的输入序列（例如句子中的所有单词）转化为一个长的“编码向量”，使得transformer能够学习自然序列中的长程时间依赖关系。具体而言，“自注意力”应用于这个编码向量，通过计算输入序列中单词对之间的关联，增强了transformer中的时间上下文。我们认为神经活动在单个皮层区域内或整个大脑范围内传播的波可以实现类似的编码原理。通过在每个时刻将最近的输入历史封装为单个空间模式，皮层波可以从感觉输入序列中提取时间上下文，这与计算原理相同。

    The capabilities of transformer networks such as ChatGPT and other Large Language Models (LLMs) have captured the world's attention. The crucial computational mechanism underlying their performance relies on transforming a complete input sequence - for example, all the words in a sentence into a long "encoding vector" - that allows transformers to learn long-range temporal dependencies in naturalistic sequences. Specifically, "self-attention" applied to this encoding vector enhances temporal context in transformers by computing associations between pairs of words in the input sequence. We suggest that waves of neural activity, traveling across single cortical regions or across multiple regions at the whole-brain scale, could implement a similar encoding principle. By encapsulating recent input history into a single spatial pattern at each moment in time, cortical waves may enable temporal context to be extracted from sequences of sensory inputs, the same computational principle used in
    
[^11]: 自学优化器（STOP）：递归自我改进的代码生成

    Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation. (arXiv:2310.02304v1 [cs.CL])

    [http://arxiv.org/abs/2310.02304](http://arxiv.org/abs/2310.02304)

    本文提出了一种自学优化器（STOP），通过递归自我改进的代码生成，使用融合了语言模型的脚手架程序来改进自身，从而生成性能更好的程序。

    

    最近几年的人工智能系统（例如思维树和程序辅助语言模型）取得了一些重要进展，通过提供一个“脚手架”程序来解决问题，该程序构建了多次调用语言模型以生成更好的输出。脚手架程序通常使用Python等编程语言编写。在这项工作中，我们使用了一个融合了语言模型的脚手架程序来改进自身。我们从一个种子“改进器”开始，通过多次查询语言模型并返回最佳解决方案，根据给定的效用函数来改进输入程序。然后，我们运行这个种子改进器来改进自身。在一系列细分任务中，得到的改进改进器生成的程序在性能上明显优于种子改进器。随后，我们对语言模型提出的各种自我改进策略进行了分析，包括波束搜索、遗传算法和模拟退火。由于语言模型本身没有改变，这并不是一种增长领域。

    Several recent advances in AI systems (e.g., Tree-of-Thoughts and Program-Aided Language Models) solve problems by providing a "scaffolding" program that structures multiple calls to language models to generate better outputs. A scaffolding program is written in a programming language such as Python. In this work, we use a language-model-infused scaffolding program to improve itself. We start with a seed "improver" that improves an input program according to a given utility function by querying a language model several times and returning the best solution. We then run this seed improver to improve itself. Across a small set of downstream tasks, the resulting improved improver generates programs with significantly better performance than its seed improver. Afterward, we analyze the variety of self-improvement strategies proposed by the language model, including beam search, genetic algorithms, and simulated annealing. Since the language models themselves are not altered, this is not fu
    
[^12]: MedEdit：利用外部知识库进行医学问答的模型编辑

    MedEdit: Model Editing for Medical Question Answering with External Knowledge Bases. (arXiv:2309.16035v1 [cs.CL])

    [http://arxiv.org/abs/2309.16035](http://arxiv.org/abs/2309.16035)

    该论文研究了利用外部知识库进行医学问答的模型编辑方法，通过提取医学事实并将其融入到语言模型的查询提示中，显著提高了医学问答的准确率。

    

    大型语言模型（LLM）虽然在一般领域表现强大，但在特定领域的任务，如医学问答（QA）方面往往表现不佳。此外，它们往往作为“黑盒”运作，难以修改其行为。针对这一问题，我们的研究探讨了利用上下文学习的模型编辑，旨在改进LLM的响应，而无需重新微调或重新训练。具体而言，我们提出了一种全面的检索策略，从外部知识库中提取医学事实，然后将它们合并到LLM的查询提示中。通过对MedQA-SMILE数据集进行医学QA的重点研究，我们评估了不同检索模型和向LLM提供的事实数量对其影响。值得注意的是，我们编辑后的Vicuna模型的准确率从44.46％提高到48.54％。这项工作凸显了模型编辑改善LLM性能的潜力，为缓解黑盒LLM的挑战提供了实用的方法。

    Large Language Models (LLMs), although powerful in general domains, often perform poorly on domain-specific tasks like medical question answering (QA). Moreover, they tend to function as "black-boxes," making it challenging to modify their behavior. Addressing this, our study delves into model editing utilizing in-context learning, aiming to improve LLM responses without the need for fine-tuning or retraining. Specifically, we propose a comprehensive retrieval strategy to extract medical facts from an external knowledge base, and then we incorporate them into the query prompt for the LLM. Focusing on medical QA using the MedQA-SMILE dataset, we evaluate the impact of different retrieval models and the number of facts provided to the LLM. Notably, our edited Vicuna model exhibited an accuracy improvement from 44.46% to 48.54%. This work underscores the potential of model editing to enhance LLM performance, offering a practical approach to mitigate the challenges of black-box LLMs.
    
[^13]: 自监督多模态学习：一项综述

    Self-Supervised Multimodal Learning: A Survey. (arXiv:2304.01008v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2304.01008](http://arxiv.org/abs/2304.01008)

    自监督多模态学习是一项旨在解决多模态数据中的自监督学习挑战的研究方向。它通过学习来自原始多模态数据中的表示，并解决了没有标签的多模态数据学习、不同模态的融合和不对齐数据学习等问题。

    

    多模态学习旨在理解和分析来自多种模态的信息，在监督学习范式下取得了重大进展。然而，由于依赖于配对数据和昂贵的人工注释，模型的扩展性受到了限制。与此同时，鉴于野外有大规模未注释的数据可用，自监督学习成为缓解注释瓶颈的一种有吸引力的策略。自监督多模态学习（SSML）建立在这两个方向的基础上，提供了从原始多模态数据中学习的方法。在本综述中，我们全面回顾了SSML的最新进展，阐述了自监督学习在多模态数据中面临的三个主要挑战：（1）在没有标签的多模态数据中学习表示，（2）不同模态的融合，以及（3）与不对齐数据的学习。然后，我们详细介绍了这些挑战的现有解决方案。具体而言，我们考虑了（1）目标

    Multimodal learning, which aims to understand and analyze information from multiple modalities, has achieved substantial progress in the supervised regime in recent years. However, the heavy dependence on data paired with expensive human annotations impedes scaling up models. Meanwhile, given the availability of large-scale unannotated data in the wild, self-supervised learning has become an attractive strategy to alleviate the annotation bottleneck. Building on these two directions, self-supervised multimodal learning (SSML) provides ways to learn from raw multimodal data. In this survey, we provide a comprehensive review of the state-of-the-art in SSML, in which we elucidate three major challenges intrinsic to self-supervised learning with multimodal data: (1) learning representations from multimodal data without labels, (2) fusion of different modalities, and (3) learning with unaligned data. We then detail existing solutions to these challenges. Specifically, we consider (1) object
    

