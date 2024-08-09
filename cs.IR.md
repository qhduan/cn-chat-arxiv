# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering](https://arxiv.org/abs/2403.00816) | 该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。 |
| [^2] | [Retrieval-Augmented Thought Process as Sequential Decision Making](https://arxiv.org/abs/2402.07812) | 检索增强思维过程（RATP）通过多步决策和蒙特卡洛树搜索，以及Q值估计器，解决了大型语言模型在隐私、产生幻觉和处理长文本方面的挑战，并在处理私人数据的问答任务中实现了50%的性能提升。 |
| [^3] | [A Purely Regular Approach to Non-Regular Core Spanners](https://arxiv.org/abs/2010.13442) | 本文提出了一种纯粹规则的非规则核心跨度生成方法，通过将字符串相等选择直接纳入底层规则语言中，可以得到具有略微较弱表达能力的核心跨度生成器的片段。 |

# 详细

[^1]: CFRet-DVQA：粗到精检索和高效调优用于文档视觉问答

    CFRet-DVQA: Coarse-to-Fine Retrieval and Efficient Tuning for Document Visual Question Answering

    [https://arxiv.org/abs/2403.00816](https://arxiv.org/abs/2403.00816)

    该研究提出了一种名为CFRet-DVQA的方法，通过检索和高效调优，解决了文档视觉问答中定位信息和限制模型输入的长度等问题，进一步提升了答案的生成性能。

    

    文档视觉问答（DVQA）是一个涉及根据图像内容回答查询的任务。现有工作仅限于定位单页内的信息，不支持跨页面问答交互。此外，对模型输入的标记长度限制可能导致与答案相关的部分被截断。在本研究中，我们引入了一种简单但有效的方法学，称为CFRet-DVQA，重点放在检索和高效调优上，以有效解决这一关键问题。为此，我们首先从文档中检索与所提问题相关的多个片段。随后，我们利用大型语言模型（LLM）的先进推理能力，通过指导调优进一步增强其性能。该方法使得生成的答案与文档标签的风格相符。实验演示了...

    arXiv:2403.00816v1 Announce Type: cross  Abstract: Document Visual Question Answering (DVQA) is a task that involves responding to queries based on the content of images. Existing work is limited to locating information within a single page and does not facilitate cross-page question-and-answer interaction. Furthermore, the token length limitation imposed on inputs to the model may lead to truncation of segments pertinent to the answer. In this study, we introduce a simple but effective methodology called CFRet-DVQA, which focuses on retrieval and efficient tuning to address this critical issue effectively. For that, we initially retrieve multiple segments from the document that correlate with the question at hand. Subsequently, we leverage the advanced reasoning abilities of the large language model (LLM), further augmenting its performance through instruction tuning. This approach enables the generation of answers that align with the style of the document labels. The experiments demo
    
[^2]: 检索增强的思维过程作为序列决策制定

    Retrieval-Augmented Thought Process as Sequential Decision Making

    [https://arxiv.org/abs/2402.07812](https://arxiv.org/abs/2402.07812)

    检索增强思维过程（RATP）通过多步决策和蒙特卡洛树搜索，以及Q值估计器，解决了大型语言模型在隐私、产生幻觉和处理长文本方面的挑战，并在处理私人数据的问答任务中实现了50%的性能提升。

    

    大型语言模型(LLM)展示了其强大的辅助人类并展现出"智能的火花"的能力。然而，几个开放挑战阻碍了它们的广泛应用：如对隐私的关注、倾向于产生幻觉、难以处理长文本。在本研究中，我们通过引入检索增强思维过程(RATP)来解决这些挑战。通过获取外部知识，RATP将LLM的思考生成过程定式为多步决策过程。为了优化这种思考过程，RATP利用蒙特卡洛树搜索，并学习了一个Q值估计器，实现了高效的推理。在处理具有私人数据的问答任务时，LLM训练方法受到伦理和安全问题的限制。RATP在上下文检索增强语言模型的基础上实现了50%的性能提升。

    Large Language Models (LLMs) have demonstrated their strong ability to assist people and show "sparks of intelligence". However, several open challenges hinder their wider application: such as concerns over privacy, tendencies to produce hallucinations, and difficulties in handling long contexts. In this work, we address those challenges by introducing the Retrieval-Augmented Thought Process (RATP). Given access to external knowledge, RATP formulates the thought generation of LLMs as a multiple-step decision process. To optimize such a thought process, RATP leverages Monte-Carlo Tree Search, and learns a Q-value estimator that permits cost-efficient inference. In addressing the task of question-answering with private data, where ethical and security concerns limit LLM training methods, RATP achieves a 50% improvement over existing in-context retrieval-augmented language models.
    
[^3]: 一种纯粹规则的非规则核心跨度生成方法

    A Purely Regular Approach to Non-Regular Core Spanners

    [https://arxiv.org/abs/2010.13442](https://arxiv.org/abs/2010.13442)

    本文提出了一种纯粹规则的非规则核心跨度生成方法，通过将字符串相等选择直接纳入底层规则语言中，可以得到具有略微较弱表达能力的核心跨度生成器的片段。

    

    规则跨度生成器是通过vset-自动机特征化的，它们对并集、连接和投影等代数操作封闭，并具有理想的算法属性。核心跨度生成器作为IBM SystemT中查询语言AQL的核心功能的形式化引入，除了需要字符串相等选择外，还被证明会导致静态分析和查询评估中典型问题的高复杂性甚至不可判定性。我们提出了一种替代性的核心跨度生成方法：将字符串相等选择直接纳入表示底层规则跨度生成器的规则语言中（而不是将其视为在规则跨度生成器提取的表上的代数操作），我们得到了一个具有略微较弱表达能力的核心跨度生成器的片段。

    The regular spanners (characterised by vset-automata) are closed under the algebraic operations of union, join and projection, and have desirable algorithmic properties. The core spanners (introduced by Fagin, Kimelfeld, Reiss, and Vansummeren (PODS 2013, JACM 2015) as a formalisation of the core functionality of the query language AQL used in IBM's SystemT) additionally need string-equality selections and it has been shown by Freydenberger and Holldack (ICDT 2016, Theory of Computing Systems 2018) that this leads to high complexity and even undecidability of the typical problems in static analysis and query evaluation. We propose an alternative approach to core spanners: by incorporating the string-equality selections directly into the regular language that represents the underlying regular spanner (instead of treating it as an algebraic operation on the table extracted by the regular spanner), we obtain a fragment of core spanners that, while having slightly weaker expressive power t
    

