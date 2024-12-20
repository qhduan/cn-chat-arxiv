# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fairness in Large Language Models: A Taxonomic Survey](https://arxiv.org/abs/2404.01349) | 该调查总结了大型语言模型中公平性的最新进展，包括对偏见因素的分析、公平度量和现有算法分类。 |
| [^2] | [Hands-Free VR](https://arxiv.org/abs/2402.15083) | Hands-Free VR 是一种无需手部操作的虚拟现实系统，通过语音命令实现，具有英语口音鲁棒性，通过深度学习模型和大型语言模型实现对文本的转换和执行。 |
| [^3] | [From Prejudice to Parity: A New Approach to Debiasing Large Language Model Word Embeddings](https://arxiv.org/abs/2402.11512) | 提出了DeepSoftDebias算法，在不同领域数据集、准确度指标和NLP任务中全面评估，发现其在减少性别、种族和宗教偏见方面优于现有最先进方法 |
| [^4] | [Agent-OM: Leveraging LLM Agents for Ontology Matching](https://arxiv.org/abs/2312.00326) | 本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。 |
| [^5] | [LoBaSS: Gauging Learnability in Supervised Fine-tuning Data.](http://arxiv.org/abs/2310.13008) | 本文介绍了一种新的方法LoBaSS，利用数据的可学习性作为选择监督微调数据的主要标准。这种方法可以根据模型的能力将数据选择与模型对齐，确保高效的学习。 |
| [^6] | [UOR: Universal Backdoor Attacks on Pre-trained Language Models.](http://arxiv.org/abs/2305.09574) | 本文介绍了一种新的后门攻击方法UOR，可以自动选择触发器并学习通用输出表示，成功率高达99.3％，能够对多种预训练语言模型和下游任务实施攻击，且可突破最新的防御方法。 |

# 详细

[^1]: 大型语言模型中的公平性：一个分类调查

    Fairness in Large Language Models: A Taxonomic Survey

    [https://arxiv.org/abs/2404.01349](https://arxiv.org/abs/2404.01349)

    该调查总结了大型语言模型中公平性的最新进展，包括对偏见因素的分析、公平度量和现有算法分类。

    

    大型语言模型（LLMs）在各个领域展现了显著的成功。然而，尽管它们在许多实际应用中表现出色，大多数这些算法缺乏公平性考虑。因此，它们可能导致针对某些社区，特别是边缘化人群的歧视性结果，促使对公平的LLMs进行广泛研究。与传统机器学习中的公平相反，在LLMs中的公平性涉及独特的背景、分类法和实现技术。为此，该调查提供了关于公平LLMs的现有文献研究进展的全面概述。具体来说，提供了有关LLMs的简要介绍，接着分析了导致LLMs偏见的因素。此外，分类讨论了LLMs中的公平概念，总结了评估LLMs偏见的指标和现有算法。

    arXiv:2404.01349v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated remarkable success across various domains. However, despite their promising performance in numerous real-world applications, most of these algorithms lack fairness considerations. Consequently, they may lead to discriminatory outcomes against certain communities, particularly marginalized populations, prompting extensive study in fair LLMs. On the other hand, fairness in LLMs, in contrast to fairness in traditional machine learning, entails exclusive backgrounds, taxonomies, and fulfillment techniques. To this end, this survey presents a comprehensive overview of recent advances in the existing literature concerning fair LLMs. Specifically, a brief introduction to LLMs is provided, followed by an analysis of factors contributing to bias in LLMs. Additionally, the concept of fairness in LLMs is discussed categorically, summarizing metrics for evaluating bias in LLMs and existing algorithms 
    
[^2]: 无需手部操作的虚拟现实系统

    Hands-Free VR

    [https://arxiv.org/abs/2402.15083](https://arxiv.org/abs/2402.15083)

    Hands-Free VR 是一种无需手部操作的虚拟现实系统，通过语音命令实现，具有英语口音鲁棒性，通过深度学习模型和大型语言模型实现对文本的转换和执行。

    

    本文介绍了一种名为Hands-Free VR的基于语音的自然语言虚拟现实界面。用户可以通过语音发出命令，其语音音频数据经过一个针对单词音素相似性和英语口音的鲁棒性进行微调的语音识别深度学习模型转换为文本，然后利用一个对自然语言多样性具有鲁棒性的大型语言模型将文本映射为可执行的虚拟现实命令。Hands-Free VR在一个受控的被试研究中（N = 22）进行了评估，要求参与者找到特定物体并以各种配置放置它们。在对照条件下，参与者使用传统的虚拟现实用户界面通过手持控制器抓取、搬运和定位物体。在实验条件下，参与者使用Hands-Free VR。结果表明：（1）Hands-Free VR对英语口音具有鲁棒性，因为在我们的20名参与者中，英语不是他们的首选语言。

    arXiv:2402.15083v1 Announce Type: cross  Abstract: The paper introduces Hands-Free VR, a voice-based natural-language interface for VR. The user gives a command using their voice, the speech audio data is converted to text using a speech-to-text deep learning model that is fine-tuned for robustness to word phonetic similarity and to spoken English accents, and the text is mapped to an executable VR command using a large language model that is robust to natural language diversity. Hands-Free VR was evaluated in a controlled within-subjects study (N = 22) that asked participants to find specific objects and to place them in various configurations. In the control condition participants used a conventional VR user interface to grab, carry, and position the objects using the handheld controllers. In the experimental condition participants used Hands-Free VR. The results confirm that: (1) Hands-Free VR is robust to spoken English accents, as for 20 of our participants English was not their f
    
[^3]: 从偏见到平等：去偏巨型语言模型词嵌入的新方法

    From Prejudice to Parity: A New Approach to Debiasing Large Language Model Word Embeddings

    [https://arxiv.org/abs/2402.11512](https://arxiv.org/abs/2402.11512)

    提出了DeepSoftDebias算法，在不同领域数据集、准确度指标和NLP任务中全面评估，发现其在减少性别、种族和宗教偏见方面优于现有最先进方法

    

    嵌入在巨型语言模型的有效性中扮演着重要角色。它们是这些模型把握上下文关系、促进更细致语言理解以及在许多需要对人类语言有基本理解的复杂任务上表现出色的基石。鉴于这些嵌入往往自身反映或展示偏见，因此这些模型可能也会无意中学习这种偏见。在这项研究中，我们在开创性前人研究基础上提出了DeepSoftDebias，这是一种使用神经网络进行“软去偏”的算法。我们在各类最先进数据集、准确度指标和具有挑战的自然语言处理任务中全面评估了这个算法。我们发现DeepSoftDebias在减少性别、种族和宗教偏见方面优于目前的最先进方法。

    arXiv:2402.11512v1 Announce Type: new  Abstract: Embeddings play a pivotal role in the efficacy of Large Language Models. They are the bedrock on which these models grasp contextual relationships and foster a more nuanced understanding of language and consequently perform remarkably on a plethora of complex tasks that require a fundamental understanding of human language. Given that these embeddings themselves often reflect or exhibit bias, it stands to reason that these models may also inadvertently learn this bias. In this work, we build on the seminal previous work and propose DeepSoftDebias, an algorithm that uses a neural network to perform `soft debiasing'. We exhaustively evaluate this algorithm across a variety of SOTA datasets, accuracy metrics, and challenging NLP tasks. We find that DeepSoftDebias outperforms the current state-of-the-art methods at reducing bias across gender, race, and religion.
    
[^4]: Agent-OM：利用LLM代理进行本体匹配

    Agent-OM: Leveraging LLM Agents for Ontology Matching

    [https://arxiv.org/abs/2312.00326](https://arxiv.org/abs/2312.00326)

    本研究提出了Agent-OM，利用LLM代理为本体匹配系统引入了新的设计范式。

    

    本体匹配（OM）能够实现不同本体之间的语义互操作性，通过对齐相关实体来解决其概念异构性。本研究引入了一种新颖的基于代理的LLM设计范式，命名为Agent-OM，包括两个用于检索和匹配的同体代理以及一组基于提示的简单OM工具。

    arXiv:2312.00326v2 Announce Type: replace  Abstract: Ontology matching (OM) enables semantic interoperability between different ontologies and resolves their conceptual heterogeneity by aligning related entities. OM systems currently have two prevailing design paradigms: conventional knowledge-based expert systems and newer machine learning-based predictive systems. While large language models (LLMs) and LLM agents have revolutionised data engineering and have been applied creatively in many domains, their potential for OM remains underexplored. This study introduces a novel agent-powered LLM-based design paradigm for OM systems. With consideration of several specific challenges in leveraging LLM agents for OM, we propose a generic framework, namely Agent-OM, consisting of two Siamese agents for retrieval and matching, with a set of simple prompt-based OM tools. Our framework is implemented in a proof-of-concept system. Evaluations of three Ontology Alignment Evaluation Initiative (OAE
    
[^5]: LoBaSS：在监督微调数据中测量可学习性

    LoBaSS: Gauging Learnability in Supervised Fine-tuning Data. (arXiv:2310.13008v1 [cs.LG])

    [http://arxiv.org/abs/2310.13008](http://arxiv.org/abs/2310.13008)

    本文介绍了一种新的方法LoBaSS，利用数据的可学习性作为选择监督微调数据的主要标准。这种方法可以根据模型的能力将数据选择与模型对齐，确保高效的学习。

    

    监督微调（SFT）是将大型语言模型（LLM）与特定任务的先决条件对齐的关键阶段。微调数据的选择深刻影响模型的性能，传统上以数据质量和分布为基础。在本文中，我们引入了SFT数据选择的一个新维度：可学习性。这个新维度的动机是由LLM在预训练阶段获得的能力。鉴于不同的预训练模型具有不同的能力，适合一个模型的SFT数据可能不适合另一个模型。因此，我们引入了学习能力这个术语来定义数据对模型进行有效学习的适合性。我们提出了基于损失的SFT数据选择（LoBaSS）方法，利用数据的可学习性作为选择SFT数据的主要标准。这种方法提供了一种细致的方法，允许将数据选择与固有的模型能力对齐，确保高效的学习。

    Supervised Fine-Tuning (SFT) serves as a crucial phase in aligning Large Language Models (LLMs) to specific task prerequisites. The selection of fine-tuning data profoundly influences the model's performance, whose principle is traditionally grounded in data quality and distribution. In this paper, we introduce a new dimension in SFT data selection: learnability. This new dimension is motivated by the intuition that SFT unlocks capabilities acquired by a LLM during the pretraining phase. Given that different pretrained models have disparate capabilities, the SFT data appropriate for one may not suit another. Thus, we introduce the term learnability to define the suitability of data for effective learning by the model. We present the Loss Based SFT Data Selection (LoBaSS) method, utilizing data learnability as the principal criterion for the selection SFT data. This method provides a nuanced approach, allowing the alignment of data selection with inherent model capabilities, ensuring op
    
[^6]: UOR：预训练语言模型的通用后门攻击

    UOR: Universal Backdoor Attacks on Pre-trained Language Models. (arXiv:2305.09574v1 [cs.CL])

    [http://arxiv.org/abs/2305.09574](http://arxiv.org/abs/2305.09574)

    本文介绍了一种新的后门攻击方法UOR，可以自动选择触发器并学习通用输出表示，成功率高达99.3％，能够对多种预训练语言模型和下游任务实施攻击，且可突破最新的防御方法。

    

    在预训练语言模型中植入后门可以传递到各种下游任务，这对安全构成了严重威胁。然而，现有的针对预训练语言模型的后门攻击大都是非目标和特定任务的。很少有针对目标和任务不可知性的方法使用手动预定义的触发器和输出表示，这使得攻击效果不够强大和普适。本文首先总结了一个更具威胁性的预训练语言模型后门攻击应满足的要求，然后提出了一种新的后门攻击方法UOR，通过将手动选择变成自动优化，打破了以往方法的瓶颈。具体来说，我们定义了被污染的监督对比学习，可以自动学习各种预训练语言模型触发器的更加均匀和通用输出表示。此外，我们使用梯度搜索选取适当的触发词，可以适应不同的预训练语言模型和词汇表。实验证明，UOR可以在各种PLMs和下游任务中实现高后门成功率（高达99.3％），优于现有方法。此外，UOR还可以突破对抗后门攻击的最新防御方法。

    Backdoors implanted in pre-trained language models (PLMs) can be transferred to various downstream tasks, which exposes a severe security threat. However, most existing backdoor attacks against PLMs are un-targeted and task-specific. Few targeted and task-agnostic methods use manually pre-defined triggers and output representations, which prevent the attacks from being more effective and general. In this paper, we first summarize the requirements that a more threatening backdoor attack against PLMs should satisfy, and then propose a new backdoor attack method called UOR, which breaks the bottleneck of the previous approach by turning manual selection into automatic optimization. Specifically, we define poisoned supervised contrastive learning which can automatically learn the more uniform and universal output representations of triggers for various PLMs. Moreover, we use gradient search to select appropriate trigger words which can be adaptive to different PLMs and vocabularies. Experi
    

