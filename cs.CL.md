# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Weak-to-Strong Jailbreaking on Large Language Models](https://arxiv.org/abs/2401.17256) | 通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。 |
| [^2] | [Quantifying the Uniqueness of Donald Trump in Presidential Discourse.](http://arxiv.org/abs/2401.01405) | 这项研究使用了一种新的度量标准来量化唐纳德·特朗普在总统演讲中的独特性，并发现了他在使用具有分裂性和对抗性的语言、重复强调等方面与其他总统候选人存在显著差异。此外，特朗普比共和党同僚更具独特性。 |
| [^3] | [Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs.](http://arxiv.org/abs/2308.09954) | Eva-KELLM是一个新的用于评估LLMs知识编辑的基准，提供了一个评估框架和数据集。该基准通过使用原始文档进行知识编辑和多角度的评估来解决了现有研究中收集成本高、表达复杂事实困难、评估视角受限等问题。 |

# 详细

[^1]: 大规模语言模型的弱到强破解

    Weak-to-Strong Jailbreaking on Large Language Models

    [https://arxiv.org/abs/2401.17256](https://arxiv.org/abs/2401.17256)

    通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。

    

    尽管已经付出了大量努力来对齐大规模语言模型（LLMs），但红队测试报告表明，这些经过精心对齐的LLMs仍然可以通过对抗性提示、调优或解码进行破解。在调查对齐LLMs的破解漏洞时，我们观察到破解和对齐模型的解码分布仅在初始生成中存在差异。这一观察结果激发了我们提出的弱到强破解攻击，敌对方可以利用较小的不安全/对齐LLMs（例如7B）指导对显著较大的对齐LLMs（例如70B）进行破解。要进行破解，只需额外解码两个较小的LLMs一次，与解码较大的LLMs相比，其计算和延迟成本较小。通过在三个不同组织的五个模型上进行实验，我们证明了该攻击的有效性。我们的研究揭示了一种以前未注意到但高效的破解方式，

    Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, expo
    
[^2]: 量化唐纳德·特朗普在总统演讲中的独特性

    Quantifying the Uniqueness of Donald Trump in Presidential Discourse. (arXiv:2401.01405v1 [cs.CL])

    [http://arxiv.org/abs/2401.01405](http://arxiv.org/abs/2401.01405)

    这项研究使用了一种新的度量标准来量化唐纳德·特朗普在总统演讲中的独特性，并发现了他在使用具有分裂性和对抗性的语言、重复强调等方面与其他总统候选人存在显著差异。此外，特朗普比共和党同僚更具独特性。

    

    唐纳德·特朗普与其他总统在演讲中是否表达出不同的风格？如果是，有哪些方面的不同？这些差异是否局限于任何单一的沟通媒介？为了调查这些问题，本文引入了一种基于大型语言模型的独特性度量标准，开发了一个新的分裂性演讲词库，并提出了比较政治对手词汇特征的框架。将这些工具应用于多种总统演讲语料库，我们发现有相当多的证据表明特朗普的讲话模式与近代历任主要总统候选人不同。一些显著的发现包括特朗普使用特别具有分裂性和对抗性的语言针对他的政治对手，并且他重复强调的模式。此外，特朗普比他的共和党同僚更加独特，而他们的独特性值与民主党相对较接近。这些差异在多种度量方法下保持一致。

    Does Donald Trump speak differently from other presidents? If so, in what ways? Are these differences confined to any single medium of communication? To investigate these questions, this paper introduces a novel metric of uniqueness based on large language models, develops a new lexicon for divisive speech, and presents a framework for comparing the lexical features of political opponents. Applying these tools to a variety of corpora of presidential speeches, we find considerable evidence that Trump's speech patterns diverge from those of all major party nominees for the presidency in recent history. Some notable findings include Trump's employment of particularly divisive and antagonistic language targeting of his political opponents and his patterns of repetition for emphasis. Furthermore, Trump is significantly more distinctive than his fellow Republicans, whose uniqueness values are comparably closer to those of the Democrats. These differences hold across a variety of measurement 
    
[^3]: Eva-KELLM：评估LLMs的知识编辑的新基准

    Eva-KELLM: A New Benchmark for Evaluating Knowledge Editing of LLMs. (arXiv:2308.09954v1 [cs.CL])

    [http://arxiv.org/abs/2308.09954](http://arxiv.org/abs/2308.09954)

    Eva-KELLM是一个新的用于评估LLMs知识编辑的基准，提供了一个评估框架和数据集。该基准通过使用原始文档进行知识编辑和多角度的评估来解决了现有研究中收集成本高、表达复杂事实困难、评估视角受限等问题。

    

    大型语言模型（LLMs）的参数中存储着丰富的知识。然而，这些知识随着时间的推移可能变得过时或不合适。因此，对LLMs进行知识编辑并评估其效果引起了越来越多的关注。现有研究主要集中在使用事实三元组进行知识编辑，这不仅在收集上产生高成本，而且在表达复杂事实时也存在困难。此外，这些研究在评估视角上往往受到限制。在本文中，我们提出了Eva-KELLM，用于评估LLMs的知识编辑的新基准。该基准包括一个评估框架和相应的数据集。在我们的框架下，我们首先要求LLM使用原始文档进行知识编辑，与使用事实三元组相比，这提供了一种更方便、更通用的方法。然后我们从多个角度评估更新后的LLM的表现。

    Large language models (LLMs) possess a wealth of knowledge encoded in their parameters. However, this knowledge may become outdated or unsuitable over time. As a result, there has been a growing interest in knowledge editing for LLMs and evaluating its effectiveness. Existing studies primarily focus on knowledge editing using factual triplets, which not only incur high costs for collection but also struggle to express complex facts. Furthermore, these studies are often limited in their evaluation perspectives. In this paper, we propose Eva-KELLM, a new benchmark for evaluating knowledge editing of LLMs. This benchmark includes an evaluation framework and a corresponding dataset. Under our framework, we first ask the LLM to perform knowledge editing using raw documents, which provides a more convenient and universal approach compared to using factual triplets. We then evaluate the updated LLM from multiple perspectives. In addition to assessing the effectiveness of knowledge editing and
    

