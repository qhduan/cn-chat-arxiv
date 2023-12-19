# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge Graph Prompting for Multi-Document Question Answering.](http://arxiv.org/abs/2308.11730) | 这篇论文提出了一种知识图谱引导的方法，用于在多文档问答任务中为大型语言模型（LLMs）提示正确的上下文。通过构建多个文档上的知识图谱，并设计基于语言模型的图遍历器，该方法能够帮助LLMs在MD-QA中进行答案预测。 |
| [^2] | [Ranking with Long-Term Constraints.](http://arxiv.org/abs/2307.04923) | 本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。 |
| [^3] | [Learning to Rank in Generative Retrieval.](http://arxiv.org/abs/2306.15222) | 这篇论文介绍了一种将生成式检索和经典的学习排序范例相结合的方法，通过使用段落排序损失来训练自回归模型，直接优化自回归模型朝着最优解优化。 |
| [^4] | [CTRL: Connect Tabular and Language Model for CTR Prediction.](http://arxiv.org/abs/2306.02841) | 提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。 |
| [^5] | [Editing Language Model-based Knowledge Graph Embeddings.](http://arxiv.org/abs/2301.10405) | 本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。 |

# 详细

[^1]: 多文档问答中的知识图谱引导

    Knowledge Graph Prompting for Multi-Document Question Answering. (arXiv:2308.11730v1 [cs.CL])

    [http://arxiv.org/abs/2308.11730](http://arxiv.org/abs/2308.11730)

    这篇论文提出了一种知识图谱引导的方法，用于在多文档问答任务中为大型语言模型（LLMs）提示正确的上下文。通过构建多个文档上的知识图谱，并设计基于语言模型的图遍历器，该方法能够帮助LLMs在MD-QA中进行答案预测。

    

    大型语言模型（LLMs）的“预训练、提示、预测”范式在开放域问答（OD-QA）中取得了显著的成功。然而，很少有工作在多文档问答（MD-QA）场景下探索这个范式，这是一个要求对不同文档的内容和结构之间的逻辑关联有深入理解的任务。为了填补这一重要的空白，我们提出了一种知识图谱引导（KGP）方法，用于在MD-QA中为LLMs提示正确的上下文，该方法包括图构建模块和图遍历模块。对于图构建，我们使用节点来表示文段或文档结构（例如，页面/表格），而使用边来表示文段之间的语义/词汇相似性或者文档内的结构关系。对于图遍历，我们设计了一个基于LM的图遍历器，它在节点之间导航并收集支持性的文段，以帮助LLMs在MD-QA中进行答案预测。

    The 'pre-train, prompt, predict' paradigm of large language models (LLMs) has achieved remarkable success in open-domain question answering (OD-QA). However, few works explore this paradigm in the scenario of multi-document question answering (MD-QA), a task demanding a thorough understanding of the logical associations among the contents and structures of different documents. To fill this crucial gap, we propose a Knowledge Graph Prompting (KGP) method to formulate the right context in prompting LLMs for MD-QA, which consists of a graph construction module and a graph traversal module. For graph construction, we create a knowledge graph (KG) over multiple documents with nodes symbolizing passages or document structures (e.g., pages/tables), and edges denoting the semantic/lexical similarity between passages or intra-document structural relations. For graph traversal, we design an LM-guided graph traverser that navigates across nodes and gathers supporting passages assisting LLMs in MD
    
[^2]: 带有长期约束的排名

    Ranking with Long-Term Constraints. (arXiv:2307.04923v1 [cs.IR])

    [http://arxiv.org/abs/2307.04923](http://arxiv.org/abs/2307.04923)

    本文提出了一个新的框架，使决策者可以表达平台行为的长期目标，并通过新的基于控制的算法实现这些目标，同时最小化对短期参与的影响。

    

    用户通过他们的选择反馈（例如点击，购买）是为训练搜索和推荐算法提供的最常见类型的数据之一。然而，仅基于选择数据进行短视培训的系统可能仅改善短期参与度，而不能改善平台的长期可持续性以及对用户、内容提供者和其他利益相关者的长期利益。因此，本文开发了一个新的框架，其中决策者（例如平台运营商、监管机构、用户）可以表达平台行为的长期目标（例如公平性、收入分配、法律要求）。这些目标采取了超越个体会话的曝光或影响目标的形式，我们提供了新的基于控制的算法来实现这些目标。具体而言，控制器的设计旨在以最小化对短期参与的影响来实现所述的长期目标。除了原则性的理论推导外，

    The feedback that users provide through their choices (e.g., clicks, purchases) is one of the most common types of data readily available for training search and recommendation algorithms. However, myopically training systems based on choice data may only improve short-term engagement, but not the long-term sustainability of the platform and the long-term benefits to its users, content providers, and other stakeholders. In this paper, we thus develop a new framework in which decision makers (e.g., platform operators, regulators, users) can express long-term goals for the behavior of the platform (e.g., fairness, revenue distribution, legal requirements). These goals take the form of exposure or impact targets that go well beyond individual sessions, and we provide new control-based algorithms to achieve these goals. In particular, the controllers are designed to achieve the stated long-term goals with minimum impact on short-term engagement. Beyond the principled theoretical derivation
    
[^3]: 学习在生成式检索中进行排序

    Learning to Rank in Generative Retrieval. (arXiv:2306.15222v1 [cs.CL])

    [http://arxiv.org/abs/2306.15222](http://arxiv.org/abs/2306.15222)

    这篇论文介绍了一种将生成式检索和经典的学习排序范例相结合的方法，通过使用段落排序损失来训练自回归模型，直接优化自回归模型朝着最优解优化。

    

    生成式检索是一种有前景的文本检索范例，它将相关段落的标识符字符串生成为检索目标。这种范例利用强大的生成模型，并代表了与传统的学习排序方法有所不同的新范例。然而，尽管其快速发展，当前的生成式检索方法仍存在局限性。它们通常依赖启发式函数将预测的标识符转换为段落排序列表，这在生成式检索的学习目标与期望的段落排序目标之间产生了差距。此外，文本生成的固有曝光偏差问题在生成式检索中仍然存在。为了解决这些问题，我们提出了一种新颖的框架，称为LTRGR，它将生成式检索与经典的学习排序范例相结合。我们的方法涉及使用段落排序损失训练一个自回归模型，该损失直接优化自回归模型朝着最优解优化。

    Generative retrieval is a promising new paradigm in text retrieval that generates identifier strings of relevant passages as the retrieval target. This paradigm leverages powerful generation models and represents a new paradigm distinct from traditional learning-to-rank methods. However, despite its rapid development, current generative retrieval methods are still limited. They typically rely on a heuristic function to transform predicted identifiers into a passage rank list, which creates a gap between the learning objective of generative retrieval and the desired passage ranking target. Moreover, the inherent exposure bias problem of text generation also persists in generative retrieval. To address these issues, we propose a novel framework, called LTRGR, that combines generative retrieval with the classical learning-to-rank paradigm. Our approach involves training an autoregressive model using a passage rank loss, which directly optimizes the autoregressive model toward the optimal 
    
[^4]: CTRL: 连接表格和语言模型进行CTR预测

    CTRL: Connect Tabular and Language Model for CTR Prediction. (arXiv:2306.02841v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2306.02841](http://arxiv.org/abs/2306.02841)

    提出了CTRL框架，将原始表格数据转换为文本数据，使用协作CTR模型分别对两种数据进行建模，提取关于CTR预测的语义信息，并在真实工业数据集上取得最新的SOTA性能水平。

    

    传统的CTR预测模型将表格数据转换为one-hot向量，并利用特征之间的协作关系来推断用户对项目的偏好。这种建模范式抛弃了基本的语义信息。尽管一些最近的工作（如P5和M6-Rec）已经探索了使用预训练语言模型（PLMs）提取CTR预测的语义信号的潜力，但它们计算成本高，效率低。此外，尚未考虑到有益的协作关系，从而阻碍了推荐的性能。为了解决这些问题，我们提出了一个新的框架CTRL，它是工业友好的和模型不可知的，具有高训练和推理效率。具体而言，原始的表格数据首先被转换为文本数据。两种不同的模态被分别视为两个模态，并分别输入协作CTR模型中以建模它们的交互作用。我们还提出了信息蒸馏机制，从PLMs中提取关于CTR预测的语义信息，进一步提高了模型的性能。在三个真实的工业数据集上，我们的模型在比较其他现有的模型时均达到了最新的SOTA性能水平。

    Traditional click-through rate (CTR) prediction models convert the tabular data into one-hot vectors and leverage the collaborative relations among features for inferring user's preference over items. This modeling paradigm discards the essential semantic information. Though some recent works like P5 and M6-Rec have explored the potential of using Pre-trained Language Models (PLMs) to extract semantic signals for CTR prediction, they are computationally expensive and suffer from low efficiency. Besides, the beneficial collaborative relations are not considered, hindering the recommendation performance. To solve these problems, in this paper, we propose a novel framework \textbf{CTRL}, which is industrial friendly and model-agnostic with high training and inference efficiency. Specifically, the original tabular data is first converted into textual data. Both tabular data and converted textual data are regarded as two different modalities and are separately fed into the collaborative CTR
    
[^5]: 基于语言模型的知识图谱嵌入编辑

    Editing Language Model-based Knowledge Graph Embeddings. (arXiv:2301.10405v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.10405](http://arxiv.org/abs/2301.10405)

    本文提出了一种新的任务——编辑基于语言模型的知识图谱嵌入，旨在实现对KG嵌入的数据高效和快速更新。针对这一任务，提出了一个简单而强大的方案——KGEditor，可以更好地更新特定事实而不影响其余部分的性能。

    

    近几十年来，使用语言模型进行知识图谱（KG）嵌入已经取得了实证成功。但是，基于语言模型的KG嵌入通常作为静态工件部署，修改起来具有挑战性，需要重新训练。为了解决这个问题，本文提出了一种新的任务，即编辑基于语言模型的KG嵌入。该任务旨在实现对KG嵌入的数据高效和快速更新，而不影响其余部分的性能。我们构建了四个新数据集：E-FB15k237、A-FB15k237、E-WN18RR 和 A-WN18RR，并评估了几种知识编辑基线，证明了之前的模型处理该任务的能力有限。我们进一步提出了一个简单但强大的基线——KGEditor，它利用超网络的附加参数层来编辑/添加事实。全面的实验结果表明，当更新特定事实而不影响其余部分的性能时，KGEditor 的表现更好。

    Recently decades have witnessed the empirical success of framing Knowledge Graph (KG) embeddings via language models. However, language model-based KG embeddings are usually deployed as static artifacts, which are challenging to modify without re-training after deployment. To address this issue, we propose a new task of editing language model-based KG embeddings in this paper. The proposed task aims to enable data-efficient and fast updates to KG embeddings without damaging the performance of the rest. We build four new datasets: E-FB15k237, A-FB15k237, E-WN18RR, and A-WN18RR, and evaluate several knowledge editing baselines demonstrating the limited ability of previous models to handle the proposed challenging task. We further propose a simple yet strong baseline dubbed KGEditor, which utilizes additional parametric layers of the hyper network to edit/add facts. Comprehensive experimental results demonstrate that KGEditor can perform better when updating specific facts while not affec
    

