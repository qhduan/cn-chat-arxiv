# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node](https://arxiv.org/abs/2403.17209) | 通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。 |
| [^2] | [Multi-Agent Collaboration Framework for Recommender Systems](https://arxiv.org/abs/2402.15235) | MACRec是一个新颖的框架，旨在通过多智能体协作来增强推荐系统，提供了多种专业智能体的协作努力解决推荐任务，并提供了应用示例。 |
| [^3] | [A Survey on Data-Centric Recommender Systems](https://arxiv.org/abs/2401.17878) | 数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。 |
| [^4] | [History-Aware Conversational Dense Retrieval.](http://arxiv.org/abs/2401.16659) | 该论文提出了一种历史感知的对话式稠密检索系统，通过上下文去噪的查询重构以及根据历史轮次的实际影响自动挖掘监督信号改进了现有的对话式稠密检索方法。 |
| [^5] | [INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning.](http://arxiv.org/abs/2401.06532) | 本研究探索了指令调优的方法，以增强大型语言模型在信息检索任务中的能力，通过引入一个新的指令调优数据集INTERS，涵盖了21个IR任务，该方法显著提升了性能。 |
| [^6] | [CausalCite: A Causal Formulation of Paper Citations.](http://arxiv.org/abs/2311.02790) | CausalCite是一种以因果推断为基础的论文引用公式化方法，通过对文本进行嵌入和相似样本的提取来评估论文的重要性，并在各个标准上展示了其有效性。 |
| [^7] | [Inference-time Re-ranker Relevance Feedback for Neural Information Retrieval.](http://arxiv.org/abs/2305.11744) | 本文提出了一种利用重排器提供推理时间反馈来改进检索的方法，可以显著提高低召回率@ K下的检索性能。 |

# 详细

[^1]: 利用大型语言模型代理生成资产管理外壳：数字孪生和语义节点中的互操作性

    Generation of Asset Administration Shell with Large Language Model Agents: Interoperability in Digital Twins with Semantic Node

    [https://arxiv.org/abs/2403.17209](https://arxiv.org/abs/2403.17209)

    通过大型语言模型代理生成AAS实例模型，实现了在数字孪生中的互操作性，降低了手动创建成本和时间。

    

    这项研究介绍了一种新颖的方法，用于协助在工业4.0背景下为数字孪生建模创建资产管理外壳（AAS）实例，旨在增强智能制造中的互操作性，减少手动工作。我们构建了一个“语义节点”数据结构来捕捉文本数据的语义要义。然后，设计并实现了一个由大型语言模型驱动的系统，用于处理“语义节点”并从文本技术数据生成AAS实例模型。我们的评估表明，有效生成率为62-79%，表明相当比例的手动创建工作可以转换为更容易的验证工作，从而减少创建AAS实例模型的时间和成本。在我们的评估中，对不同LLM的比较分析以及检索增强生成（RAG）机制的深入消融研究提供了有关LLM有效性的见解。

    arXiv:2403.17209v1 Announce Type: new  Abstract: This research introduces a novel approach for assisting the creation of Asset Administration Shell (AAS) instances for digital twin modeling within the context of Industry 4.0, aiming to enhance interoperability in smart manufacturing and reduce manual effort. We construct a "semantic node" data structure to capture the semantic essence of textual data. Then, a system powered by large language models is designed and implemented to process "semantic node" and generate AAS instance models from textual technical data. Our evaluation demonstrates a 62-79% effective generation rate, indicating a substantial proportion of manual creation effort can be converted into easier validation effort, thereby reducing the time and cost in creating AAS instance models. In our evaluation, a comparative analysis of different LLMs and an in-depth ablation study of Retrieval-Augmented Generation (RAG) mechanisms provide insights into the effectiveness of LLM
    
[^2]: 用于推荐系统的多智能体协作框架

    Multi-Agent Collaboration Framework for Recommender Systems

    [https://arxiv.org/abs/2402.15235](https://arxiv.org/abs/2402.15235)

    MACRec是一个新颖的框架，旨在通过多智能体协作来增强推荐系统，提供了多种专业智能体的协作努力解决推荐任务，并提供了应用示例。

    

    基于LLM的智能体因其决策技能和处理复杂任务的能力而受到广泛关注。鉴于当前在利用智能体协作能力增强推荐系统方面存在的空白，我们介绍了MACRec，这是一个旨在通过多智能体协作增强推荐系统的新颖框架。与现有关于使用智能体进行用户/商品模拟的工作不同，我们旨在部署多智能体直接处理推荐任务。在我们的框架中，通过各种专业智能体的协作努力来解决推荐任务，包括经理、用户/商品分析师、反射器、搜索器和任务解释器，它们具有不同的工作流。此外，我们提供应用示例，说明开发人员如何轻松在各种推荐任务上使用MACRec，包括评分预测、序列推荐、对话推荐和解释生成。

    arXiv:2402.15235v1 Announce Type: new  Abstract: LLM-based agents have gained considerable attention for their decision-making skills and ability to handle complex tasks. Recognizing the current gap in leveraging agent capabilities for multi-agent collaboration in recommendation systems, we introduce MACRec, a novel framework designed to enhance recommendation systems through multi-agent collaboration. Unlike existing work on using agents for user/item simulation, we aim to deploy multi-agents to tackle recommendation tasks directly. In our framework, recommendation tasks are addressed through the collaborative efforts of various specialized agents, including Manager, User/Item Analyst, Reflector, Searcher, and Task Interpreter, with different working flows. Furthermore, we provide application examples of how developers can easily use MACRec on various recommendation tasks, including rating prediction, sequential recommendation, conversational recommendation, and explanation generation
    
[^3]: 数据中心推荐系统综述

    A Survey on Data-Centric Recommender Systems

    [https://arxiv.org/abs/2401.17878](https://arxiv.org/abs/2401.17878)

    数据中心推荐系统综述了推荐系统从模型为中心到数据为中心的转变。这篇综述首次系统概述了数据中心推荐系统的基本概念、推荐数据的主要问题以及最近的研究和未来的发展方向。

    

    推荐系统已成为应对信息过载的重要工具，适用于各种实际场景。最近推荐系统的发展趋势出现了范式转变，从模型为中心的创新转向数据质量和数量的重要性。这一变化引出了数据中心推荐系统（Data-Centric RS）的概念，标志着该领域的重要发展。本综述首次系统地概述了数据中心推荐系统，包括1）推荐数据和数据中心推荐系统的基本概念；2）推荐数据面临的三个主要问题；3）为解决这些问题而开展的最近研究；以及4）数据中心推荐系统可能的未来发展方向。

    Recommender systems (RS) have become essential tools for mitigating information overload in a range of real-world scenarios. Recent trends in RS have seen a paradigm shift, moving the spotlight from model-centric innovations to the importance of data quality and quantity. This evolution has given rise to the concept of data-centric recommender systems (Data-Centric RS), marking a significant development in the field. This survey provides the first systematic overview of Data-Centric RS, covering 1) the foundational concepts of recommendation data and Data-Centric RS; 2) three primary issues in recommendation data; 3) recent research developed to address these issues; and 4) several potential future directions in Data-Centric RS.
    
[^4]: 历史感知的对话式稠密检索

    History-Aware Conversational Dense Retrieval. (arXiv:2401.16659v1 [cs.IR])

    [http://arxiv.org/abs/2401.16659](http://arxiv.org/abs/2401.16659)

    该论文提出了一种历史感知的对话式稠密检索系统，通过上下文去噪的查询重构以及根据历史轮次的实际影响自动挖掘监督信号改进了现有的对话式稠密检索方法。

    

    对话搜索通过实现用户和系统之间的多轮交互，实现了复杂信息检索的便利。支持这种交互需要对对话输入有全面的理解，以便根据历史信息制定良好的搜索查询。特别是，搜索查询应包括来自先前对话回合的相关信息。然而，目前的对话式稠密检索方法主要依赖于对经过精调的预训练专门检索器进行整个对话式搜索会话的优化，这可能会变得冗长和嘈杂。此外，现有方法受现有数据集中手动监督信号数量的限制。为了解决上述问题，我们提出了一种历史感知的对话式稠密检索(HAConvDR)系统，它结合了两个思想：上下文去噪的查询重构和根据历史轮次的实际影响进行自动挖掘监督信号。

    Conversational search facilitates complex information retrieval by enabling multi-turn interactions between users and the system. Supporting such interactions requires a comprehensive understanding of the conversational inputs to formulate a good search query based on historical information. In particular, the search query should include the relevant information from the previous conversation turns. However, current approaches for conversational dense retrieval primarily rely on fine-tuning a pre-trained ad-hoc retriever using the whole conversational search session, which can be lengthy and noisy. Moreover, existing approaches are limited by the amount of manual supervision signals in the existing datasets. To address the aforementioned issues, we propose a History-Aware Conversational Dense Retrieval (HAConvDR) system, which incorporates two ideas: context-denoised query reformulation and automatic mining of supervision signals based on the actual impact of historical turns. Experime
    
[^5]: INTERS: 使用指令调优解锁大型语言模型在搜索中的力量

    INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning. (arXiv:2401.06532v1 [cs.CL])

    [http://arxiv.org/abs/2401.06532](http://arxiv.org/abs/2401.06532)

    本研究探索了指令调优的方法，以增强大型语言模型在信息检索任务中的能力，通过引入一个新的指令调优数据集INTERS，涵盖了21个IR任务，该方法显著提升了性能。

    

    大型语言模型（LLMs）在各种自然语言处理任务中展示了令人印象深刻的能力。然而，由于许多与信息检索（IR）具体概念的不经常出现在自然语言中，它们在信息检索任务中的应用仍然具有挑战性。虽然基于提示的方法可以向LLMs提供任务描述，但它们往往在促进全面理解和执行IR任务方面存在不足，从而限制了LLMs的适用性。为了弥补这一差距，本研究探索了指令调优的潜力，以提高LLMs在IR任务中的熟练程度。我们引入了一个新的指令调优数据集INTERS，涵盖了3个基本IR类别中的21个任务：查询理解、文档理解和查询文档关系理解。数据来自43个不同的由手动编写的模板构成的数据集。我们的实证结果表明，INTERS显著提升了各种公开数据集上的性能。

    Large language models (LLMs) have demonstrated impressive capabilities in various natural language processing tasks. Despite this, their application to information retrieval (IR) tasks is still challenging due to the infrequent occurrence of many IR-specific concepts in natural language. While prompt-based methods can provide task descriptions to LLMs, they often fall short in facilitating comprehensive understanding and execution of IR tasks, thereby limiting LLMs' applicability. To address this gap, in this work, we explore the potential of instruction tuning to enhance LLMs' proficiency in IR tasks. We introduce a novel instruction tuning dataset, INTERS, encompassing 21 tasks across three fundamental IR categories: query understanding, document understanding, and query-document relationship understanding. The data are derived from 43 distinct datasets with manually written templates. Our empirical results reveal that INTERS significantly boosts the performance of various publicly a
    
[^6]: CausalCite：一种论文引用的因果公式化

    CausalCite: A Causal Formulation of Paper Citations. (arXiv:2311.02790v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.02790](http://arxiv.org/abs/2311.02790)

    CausalCite是一种以因果推断为基础的论文引用公式化方法，通过对文本进行嵌入和相似样本的提取来评估论文的重要性，并在各个标准上展示了其有效性。

    

    对于科学界来说，评估一篇论文的重要性至关重要但也具有挑战性。尽管引用次数是最常用的评估指标，但它们被广泛批评为无法准确反映一篇论文的真正影响力。在这项工作中，我们提出了一种因果推断方法，称为TextMatch，它将传统的匹配框架适应于高维文本嵌入。具体而言，我们使用大型语言模型（LLM）对每篇论文进行文本嵌入，通过余弦相似性提取相似样本，并根据相似度值的加权平均合成一个反事实样本。我们将得到的指标称为CausalCite，作为论文引用的因果公式化。我们展示了它在各种标准上的有效性，如与科学专家对1K篇论文的报告的论文影响力的高相关性，过去论文的（经过时间考验的）奖项，以及在各个子领域的稳定性。

    Evaluating the significance of a paper is pivotal yet challenging for the scientific community. While the citation count is the most commonly used proxy for this purpose, they are widely criticized for failing to accurately reflect a paper's true impact. In this work, we propose a causal inference method, TextMatch, which adapts the traditional matching framework to high-dimensional text embeddings. Specifically, we encode each paper using the text embeddings by large language models (LLMs), extract similar samples by cosine similarity, and synthesize a counterfactual sample by the weighted average of similar papers according to their similarity values. We apply the resulting metric, called CausalCite, as a causal formulation of paper citations. We show its effectiveness on various criteria, such as high correlation with paper impact as reported by scientific experts on a previous dataset of 1K papers, (test-of-time) awards for past papers, and its stability across various sub-fields o
    
[^7]: 面向神经信息检索的推理时间重排反馈

    Inference-time Re-ranker Relevance Feedback for Neural Information Retrieval. (arXiv:2305.11744v1 [cs.IR])

    [http://arxiv.org/abs/2305.11744](http://arxiv.org/abs/2305.11744)

    本文提出了一种利用重排器提供推理时间反馈来改进检索的方法，可以显著提高低召回率@ K下的检索性能。

    

    神经信息检索通常采用检索和重排框架：先使用双编码器网络检索K（例如100）个候选项，然后再使用更强大的交叉编码器模型对这些候选项进行重新排序，以使更好的候选项排名更高。重排器通常产生比检索器更好的候选分数，但仅限于查看前K个检索到的候选项，因此无法提高检索性能（以Recall @ K为度量）。在本文中，我们利用重排器通过提供推理时间相关反馈来改进检索。具体而言，我们利用重排器的预测对测试实例的重要信息进行了检索器查询表示的更新。我们的方法可以通过轻量级的推理时间蒸馏来实现，目的是使检索器的候选分数更接近于重排器的分数。然后使用更新后的查询向量执行第二个检索步骤。通过实验证明，我们的方法可以显著提高检索性能，特别是在低召回率@ K下。

    Neural information retrieval often adopts a retrieve-and-rerank framework: a bi-encoder network first retrieves K (e.g., 100) candidates that are then re-ranked using a more powerful cross-encoder model to rank the better candidates higher. The re-ranker generally produces better candidate scores than the retriever, but is limited to seeing only the top K retrieved candidates, thus providing no improvements in retrieval performance as measured by Recall@K. In this work, we leverage the re-ranker to also improve retrieval by providing inference-time relevance feedback to the retriever. Concretely, we update the retriever's query representation for a test instance using a lightweight inference-time distillation of the re-ranker's prediction for that instance. The distillation loss is designed to bring the retriever's candidate scores closer to those of the re-ranker. A second retrieval step is then performed with the updated query vector. We empirically show that our approach, which can 
    

