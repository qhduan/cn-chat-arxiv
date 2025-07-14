# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sequence graphs realizations and ambiguity in language models](https://arxiv.org/abs/2402.08830) | 本文研究了语言模型中序列图的实现和歧义问题，通过组合和计算的方法考虑了图的窗口大小、方向性和权重等因素，并提供了多项式时间算法来解决实现和枚举问题。 |
| [^2] | [Weak-to-Strong Jailbreaking on Large Language Models](https://arxiv.org/abs/2401.17256) | 通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。 |
| [^3] | [Large Language Models in Mental Health Care: a Scoping Review.](http://arxiv.org/abs/2401.02984) | 本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。 |
| [^4] | [An Approach to Automatically generating Riddles aiding Concept Attainment.](http://arxiv.org/abs/2310.18290) | 这个论文介绍了一种自动生成概念谜题的方法，以促进在线学习环境中的学习者参与度。通过应用概念达成模型和生成谜题，该方法可以帮助学习者更好地理解概念。 |

# 详细

[^1]: 序列图实现与语言模型中的歧义

    Sequence graphs realizations and ambiguity in language models

    [https://arxiv.org/abs/2402.08830](https://arxiv.org/abs/2402.08830)

    本文研究了语言模型中序列图的实现和歧义问题，通过组合和计算的方法考虑了图的窗口大小、方向性和权重等因素，并提供了多项式时间算法来解决实现和枚举问题。

    

    几种流行的语言模型将输入文本中的局部上下文表示为词袋。这样的表示自然地通过一个序列图来编码，其中顶点是出现在输入文本中的不同词，边表示在大小为w的滑动窗口内两个词的（有序）共现。然而，这种压缩表示通常不是双射的，可能引入一定程度的歧义。一些序列图可能以多种方式实现为一个序列，而其他一些可能无法实现任何序列。在本文中，我们从组合和计算的角度研究了序列图的可实现性和歧义。我们考虑在多种设置下的序列图实现的存在和枚举：窗口大小w、图的方向性的存在/缺失和权重（重复性）的存在/缺失。当w = 2时，我们提供了多项式时间算法来实现和枚举。

    arXiv:2402.08830v1 Announce Type: cross Abstract: Several popular language models represent local contexts in an input text as bags of words. Such representations are naturally encoded by a sequence graph whose vertices are the distinct words occurring in x, with edges representing the (ordered) co-occurrence of two words within a sliding window of size w. However, this compressed representation is not generally bijective, and may introduce some degree of ambiguity. Some sequence graphs may admit several realizations as a sequence, while others may not admit any realization. In this paper, we study the realizability and ambiguity of sequence graphs from a combinatorial and computational point of view. We consider the existence and enumeration of realizations of a sequence graph under multiple settings: window size w, presence/absence of graph orientation, and presence/absence of weights (multiplicities). When w = 2, we provide polynomial time algorithms for realizability and enumeratio
    
[^2]: 大规模语言模型的弱到强破解

    Weak-to-Strong Jailbreaking on Large Language Models

    [https://arxiv.org/abs/2401.17256](https://arxiv.org/abs/2401.17256)

    通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。

    

    尽管已经付出了大量努力来对齐大规模语言模型（LLMs），但红队测试报告表明，这些经过精心对齐的LLMs仍然可以通过对抗性提示、调优或解码进行破解。在调查对齐LLMs的破解漏洞时，我们观察到破解和对齐模型的解码分布仅在初始生成中存在差异。这一观察结果激发了我们提出的弱到强破解攻击，敌对方可以利用较小的不安全/对齐LLMs（例如7B）指导对显著较大的对齐LLMs（例如70B）进行破解。要进行破解，只需额外解码两个较小的LLMs一次，与解码较大的LLMs相比，其计算和延迟成本较小。通过在三个不同组织的五个模型上进行实验，我们证明了该攻击的有效性。我们的研究揭示了一种以前未注意到但高效的破解方式，

    Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, expo
    
[^3]: 大型语言模型在心理健康护理中的应用：一项综述研究

    Large Language Models in Mental Health Care: a Scoping Review. (arXiv:2401.02984v1 [cs.CL])

    [http://arxiv.org/abs/2401.02984](http://arxiv.org/abs/2401.02984)

    本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。

    

    目的：大型语言模型（LLM）的使用越来越广泛，需要对它们在心理健康护理领域的应用和结果进行全面的综述。本综述研究旨在对LLMs在心理健康护理中的现有发展和应用进行批判性分析，突出它们的成功，并识别这些专业领域中的挑战和限制。材料和方法：2023年11月，在PubMed、Web of Science、Google Scholar、arXiv、medRxiv和PsyArXiv六个数据库中进行了广泛的文献搜索，遵循2020年版的“系统评价和Meta分析的首选报告项目”（PRISMA）指南。最初识别了313篇出版物，按照研究纳入标准，最终选择了34篇出版物进行综述。结果：我们发现了LLMs在心理健康护理中的多种应用，包括诊断、治疗、患者参与增强等。关键挑战和限制方面的发现将被总结和讨论。

    Objective: The growing use of large language models (LLMs) stimulates a need for a comprehensive review of their applications and outcomes in mental health care contexts. This scoping review aims to critically analyze the existing development and applications of LLMs in mental health care, highlighting their successes and identifying their challenges and limitations in these specialized fields. Materials and Methods: A broad literature search was conducted in November 2023 using six databases (PubMed, Web of Science, Google Scholar, arXiv, medRxiv, and PsyArXiv) following the 2020 version of the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. A total of 313 publications were initially identified, and after applying the study inclusion criteria, 34 publications were selected for the final review. Results: We identified diverse applications of LLMs in mental health care, including diagnosis, therapy, patient engagement enhancement, etc. Key challen
    
[^4]: 一种自动生成谜题以辅助概念理解的方法

    An Approach to Automatically generating Riddles aiding Concept Attainment. (arXiv:2310.18290v1 [cs.CL])

    [http://arxiv.org/abs/2310.18290](http://arxiv.org/abs/2310.18290)

    这个论文介绍了一种自动生成概念谜题的方法，以促进在线学习环境中的学习者参与度。通过应用概念达成模型和生成谜题，该方法可以帮助学习者更好地理解概念。

    

    在在线学习环境中，保持学习者的积极参与是一个主要的挑战。为增强学习者的参与度，提出了各种不同的教学策略，无论是在线还是离线环境中。概念达成模型就是一种教学策略，它着重于学习者对概念的深入理解，而不仅仅是对概念的字典定义。通过搜索和列举用于区分各种概念的实例和非实例之间的属性，来达到这一目的。我们的工作试图将概念达成模型应用于构建概念谜题，以在在线学习环境中使用。该方法涉及从学习资源中创建事实三元组，根据其对概念的唯一性进行分类为“主题标记”和“共同”，然后根据概念达成模型的格式生成谜题，并捕获这些谜题的所有可能解。从人类评估中获得的结果显示...

    One of the primary challenges in online learning environments, is to retain learner engagement. Several different instructional strategies are proposed both in online and offline environments to enhance learner engagement. The Concept Attainment Model is one such instructional strategy that focuses on learners acquiring a deeper understanding of a concept rather than just its dictionary definition. This is done by searching and listing the properties used to distinguish examples from non-examples of various concepts. Our work attempts to apply the Concept Attainment Model to build conceptual riddles, to deploy over online learning environments. The approach involves creating factual triples from learning resources, classifying them based on their uniqueness to a concept into `Topic Markers' and `Common', followed by generating riddles based on the Concept Attainment Model's format and capturing all possible solutions to those riddles. The results obtained from the human evaluation of r
    

