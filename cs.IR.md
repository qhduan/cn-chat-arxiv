# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fairness-aware Differentially Private Collaborative Filtering.](http://arxiv.org/abs/2303.09527) | 本文提出了DP-Fair，一个两阶段的协同过滤算法框架，它结合了差分隐私机制和公平约束，旨在保护用户隐私、确保公平推荐。 |
| [^2] | [Measuring the Impact of Explanation Bias: A Study of Natural Language Justifications for Recommender Systems.](http://arxiv.org/abs/2303.09498) | 本研究提出了一种实验方案，通过积极或消极偏见的自然语言解释来测量解释对用户选择建议的影响，并发现解释可以显著影响用户选择。 |
| [^3] | [LLMSecEval: A Dataset of Natural Language Prompts for Security Evaluations.](http://arxiv.org/abs/2303.09384) | 本文提出了一个 LLMSecEval 数据集，其中包含 150 个自然语言提示，可用于评估大型语言模型在生成容易出现安全漏洞的代码时的安全性能。 |
| [^4] | [Graph Neural Network Surrogates of Fair Graph Filtering.](http://arxiv.org/abs/2303.08157) | 通过引入过滤器感知的通用近似框架，该方法定义了合适的图神经网络在运行时训练以满足统计平等约束，同时最小程度扰动原始后验情况下实现此目标。 |
| [^5] | [Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems.](http://arxiv.org/abs/2302.03735) | 本文系统地研究了如何从不同预训练语言模型中提取和转移知识，提高推荐系统性能。我们提出了一个分类法，分析和总结了基于预训练语言模型的推荐系统的培训策略和目标。 |
| [^6] | [UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text.](http://arxiv.org/abs/2108.08614) | 本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。 |

# 详细

[^1]: 公平感知的差分隐私协同过滤

    Fairness-aware Differentially Private Collaborative Filtering. (arXiv:2303.09527v1 [cs.IR])

    [http://arxiv.org/abs/2303.09527](http://arxiv.org/abs/2303.09527)

    本文提出了DP-Fair，一个两阶段的协同过滤算法框架，它结合了差分隐私机制和公平约束，旨在保护用户隐私、确保公平推荐。

    

    最近，越来越多的隐私保护机器学习任务采用差分隐私引导算法，然而，这样的算法使用在算法公平性方面有折衷，这一点被广泛认可。本文针对差分隐私随机梯度下降（DP-SGD）训练的经典协同过滤方法导致用户群体与不同用户参与水平之间存在不公平影响的问题，提出了一个两阶段框架DP-Fair，它将差分隐私机制与公平限制相结合，从而在保护用户隐私的同时确保公平推荐。

    Recently, there has been an increasing adoption of differential privacy guided algorithms for privacy-preserving machine learning tasks. However, the use of such algorithms comes with trade-offs in terms of algorithmic fairness, which has been widely acknowledged. Specifically, we have empirically observed that the classical collaborative filtering method, trained by differentially private stochastic gradient descent (DP-SGD), results in a disparate impact on user groups with respect to different user engagement levels. This, in turn, causes the original unfair model to become even more biased against inactive users. To address the above issues, we propose \textbf{DP-Fair}, a two-stage framework for collaborative filtering based algorithms. Specifically, it combines differential privacy mechanisms with fairness constraints to protect user privacy while ensuring fair recommendations. The experimental results, based on Amazon datasets, and user history logs collected from Etsy, one of th
    
[^2]: 量化解释偏差的影响：关于推荐系统自然语言解释的研究

    Measuring the Impact of Explanation Bias: A Study of Natural Language Justifications for Recommender Systems. (arXiv:2303.09498v1 [cs.HC])

    [http://arxiv.org/abs/2303.09498](http://arxiv.org/abs/2303.09498)

    本研究提出了一种实验方案，通过积极或消极偏见的自然语言解释来测量解释对用户选择建议的影响，并发现解释可以显著影响用户选择。

    

    尽管解释可能对决策产生影响，但缺乏研究来量化其对用户选择的影响。本文提出了一个实验方案，用于测量积极或消极偏见解释可能导致用户选择次优建议的程度。该方案的关键要素包括偏好引导阶段以允许个性化建议、手动识别和提取评论中的项目要素以及通过将积极和消极要素结合而引入偏见的控制方法。我们研究了两种不同的文本格式的解释：作为项目要素列表的形式和作为流畅自然语言文本的形式。通过对129名参与者进行用户研究，我们证明了解释可以显著影响用户的选择，并且这些发现可以推广到解释格式上。

    Despite the potential impact of explanations on decision making, there is a lack of research on quantifying their effect on users' choices. This paper presents an experimental protocol for measuring the degree to which positively or negatively biased explanations can lead to users choosing suboptimal recommendations. Key elements of this protocol include a preference elicitation stage to allow for personalizing recommendations, manual identification and extraction of item aspects from reviews, and a controlled method for introducing bias through the combination of both positive and negative aspects. We study explanations in two different textual formats: as a list of item aspects and as fluent natural language text. Through a user study with 129 participants, we demonstrate that explanations can significantly affect users' selections and that these findings generalize across explanation formats.
    
[^3]: LLMSecEval: 一个用于安全评估的自然语言提示数据集

    LLMSecEval: A Dataset of Natural Language Prompts for Security Evaluations. (arXiv:2303.09384v1 [cs.SE])

    [http://arxiv.org/abs/2303.09384](http://arxiv.org/abs/2303.09384)

    本文提出了一个 LLMSecEval 数据集，其中包含 150 个自然语言提示，可用于评估大型语言模型在生成容易出现安全漏洞的代码时的安全性能。

    

    大型语言模型（LLM）如 Codex 在代码自动补全和生成任务方面具有强大的能力，因为它们通过公开可用的代码从数十亿行代码中进行训练。此外，这些模型能够通过从公共 GitHub 仓库学习语言和编程实践来生成来自自然语言描述的代码片段。尽管 LLM 承诺实现软件应用的 NL 驱动部署，但是它们生成的代码的安全性尚未得到广泛调查和记录。在这项工作中，我们提出了 LLMSecEval，这是一个包含 150 个 NL 提示的数据集，可用于评估此类模型的安全性能。这些提示是基于MITRE的前25个常见弱点列表中容易出现各种安全漏洞的代码片段的自然语言描述。我们数据集中的每个提示都配有一个安全实现示例，以便与由LLM生成的代码进行比较评估。

    Large Language Models (LLMs) like Codex are powerful tools for performing code completion and code generation tasks as they are trained on billions of lines of code from publicly available sources. Moreover, these models are capable of generating code snippets from Natural Language (NL) descriptions by learning languages and programming practices from public GitHub repositories. Although LLMs promise an effortless NL-driven deployment of software applications, the security of the code they generate has not been extensively investigated nor documented. In this work, we present LLMSecEval, a dataset containing 150 NL prompts that can be leveraged for assessing the security performance of such models. Such prompts are NL descriptions of code snippets prone to various security vulnerabilities listed in MITRE's Top 25 Common Weakness Enumeration (CWE) ranking. Each prompt in our dataset comes with a secure implementation example to facilitate comparative evaluations against code produced by
    
[^4]: 基于图神经网络的公平图过滤替代方法

    Graph Neural Network Surrogates of Fair Graph Filtering. (arXiv:2303.08157v1 [cs.LG])

    [http://arxiv.org/abs/2303.08157](http://arxiv.org/abs/2303.08157)

    通过引入过滤器感知的通用近似框架，该方法定义了合适的图神经网络在运行时训练以满足统计平等约束，同时最小程度扰动原始后验情况下实现此目标。

    

    通过边传播将先前的节点值转换为后来的分数的图滤波器通常支持影响人类的图挖掘任务，例如推荐和排名。因此，重要的是在满足节点组之间的统计平等约束方面使它们公平（例如，按其代表性将分数质量在性别之间均衡分配）。为了在最小程度地扰动原始后验情况下实现此目标，我们引入了一个过滤器感知的通用近似框架，用于后验目标。这定义了适当的图神经网络，其在运行时训练，类似于过滤器，但也在本地优化包括公平感知在内的大类目标。在一组8个过滤器和5个图形的实验中，我们的方法在满足统计平等约束方面表现得不亚于替代品，同时保留基于分数的社区成员推荐的AUC并在传播先前节拍时创建最小实用损失。

    Graph filters that transform prior node values to posterior scores via edge propagation often support graph mining tasks affecting humans, such as recommendation and ranking. Thus, it is important to make them fair in terms of satisfying statistical parity constraints between groups of nodes (e.g., distribute score mass between genders proportionally to their representation). To achieve this while minimally perturbing the original posteriors, we introduce a filter-aware universal approximation framework for posterior objectives. This defines appropriate graph neural networks trained at runtime to be similar to filters but also locally optimize a large class of objectives, including fairness-aware ones. Experiments on a collection of 8 filters and 5 graphs show that our approach performs equally well or better than alternatives in meeting parity constraints while preserving the AUC of score-based community member recommendation and creating minimal utility loss in prior diffusion.
    
[^5]: 预训练、提示和推荐：语言模型范式在推荐系统中的综合调查

    Pre-train, Prompt and Recommendation: A Comprehensive Survey of Language Modelling Paradigm Adaptations in Recommender Systems. (arXiv:2302.03735v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2302.03735](http://arxiv.org/abs/2302.03735)

    本文系统地研究了如何从不同预训练语言模型中提取和转移知识，提高推荐系统性能。我们提出了一个分类法，分析和总结了基于预训练语言模型的推荐系统的培训策略和目标。

    

    预训练语言模型（PLM）的出现，通过自监督方式在大型语料库上学习通用表示，在自然语言处理（NLP）领域取得了巨大成功。预训练模型和学到的表示可受益于一系列下游NLP任务。这种培训范式最近被适用于推荐领域，并被学术界和工业界认为是一种有前途的方法。本文系统地研究了如何从不同PLM相关训练范式学习到的预训练模型中提取和转移知识，从多个角度（如通用性、稀疏性、效率和效果）提高推荐性能。具体而言，我们提出了一个正交分类法来划分现有的基于PLM的推荐系统，针对其培训策略和目标进行分析和总结。

    The emergency of Pre-trained Language Models (PLMs) has achieved tremendous success in the field of Natural Language Processing (NLP) by learning universal representations on large corpora in a self-supervised manner. The pre-trained models and the learned representations can be beneficial to a series of downstream NLP tasks. This training paradigm has recently been adapted to the recommendation domain and is considered a promising approach by both academia and industry. In this paper, we systematically investigate how to extract and transfer knowledge from pre-trained models learned by different PLM-related training paradigms to improve recommendation performance from various perspectives, such as generality, sparsity, efficiency and effectiveness. Specifically, we propose an orthogonal taxonomy to divide existing PLM-based recommender systems w.r.t. their training strategies and objectives. Then, we analyze and summarize the connection between PLM-based training paradigms and differe
    
[^6]: UNIQORN：统一的RDF知识图谱与自然语言文本问答系统

    UNIQORN: Unified Question Answering over RDF Knowledge Graphs and Natural Language Text. (arXiv:2108.08614v5 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2108.08614](http://arxiv.org/abs/2108.08614)

    本文提出了一个名为UNIQORN的问答系统，它能够无缝地处理RDF数据和文本，使用fine-tuned BERT模型为问题构建上下文图，并使用图算法确定与问题相关的子图来回答问题。

    

    问题回答在知识图谱和其他RDF数据上已经取得了巨大的进展，许多优秀的系统可以为自然语言问题或电报查询提供清晰的答案。其中一些系统将文本源作为附加证据纳入回答过程，但不能计算仅存在于文本中的答案。相反，IR和NLP社区的系统已经解决了有关文本的QA问题，但是这些系统几乎不利用语义数据和知识。本文提出了第一个可以无缝操作混合RDF数据集和文本语料库或单个来源的复杂问题的系统，在统一框架中进行操作。我们的方法称为UNIQORN，通过使用经过精细调整的BERT模型从RDF数据和/或文本语料库中检索与问题相关的证据来动态构建上下文图。结果图通常非常丰富但高度嘈杂。UNIQORN通过用于组Steiner树的图算法来处理这个输入，从而确定与问题相关的子图，进而回答问题。

    Question answering over knowledge graphs and other RDF data has been greatly advanced, with a number of good systems providing crisp answers for natural language questions or telegraphic queries. Some of these systems incorporate textual sources as additional evidence for the answering process, but cannot compute answers that are present in text alone. Conversely, systems from the IR and NLP communities have addressed QA over text, but such systems barely utilize semantic data and knowledge. This paper presents the first system for complex questions that can seamlessly operate over a mixture of RDF datasets and text corpora, or individual sources, in a unified framework. Our method, called UNIQORN, builds a context graph on-the-fly, by retrieving question-relevant evidences from the RDF data and/or a text corpus, using fine-tuned BERT models. The resulting graph is typically rich but highly noisy. UNIQORN copes with this input by a graph algorithm for Group Steiner Trees, that identifi
    

