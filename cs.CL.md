# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report](https://arxiv.org/abs/2606.26529) | 论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。 |
| [^2] | [Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation](https://arxiv.org/abs/2606.26502) | 论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。 |

# 详细

[^1]: 注意盲区：任务条件化的语言与视觉模型会遗漏它们本可报告的安全关键信号

    The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report

    [https://arxiv.org/abs/2606.26529](https://arxiv.org/abs/2606.26529)

    论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。

    

    arXiv:2606.26529v1 公告类型：交叉 摘要：人工智能安全性通常通过模型能否可靠地检测指定危害来评估，但事故往往源于无人指定的危害。我们证明，将语言或视觉模型限定于狭窄任务会抑制其报告同时存在的、本可报告的安全关键信号，这类似于人类注意盲区的机器模拟，但源于不同的机制。在放射学与驾驶文本场景以及胸部X光片视觉任务中，所有测试模型均出现抑制现象，且不随模型规模减小，在推理模型中持续存在，不同模型家族间的差异大于模型大小间的差异，而同一模型在无约束条件下报告这些信号的比率显著更高。我们将这种分离命名为“注意盲区”，并论证它使基准安全性评估与真实世界安全性脱钩：系统可在评估指定的危害上获得近乎完美的分数，却对未指定的危害视而不见。

    arXiv:2606.26529v1 Announce Type: cross  Abstract: AI safety is evaluated by how reliably a model detects the hazards it is told to find, yet accidents often arise from the hazard no one specified. We show that conditioning a language or vision model on a narrow task suppresses its reporting of co-present, safety-critical signals it can otherwise report, a machine analogue of human inattentional blindness arising from a different mechanism. Across radiology and driving text scenarios and chest-radiograph vision tasks, suppression appeared in every model tested, did not diminish with scale, persisted in a reasoning model, and varied more by model family than by size, while the same models reported these signals at substantially higher rates when unconstrained. We name this dissociation the Inattentional Gap and argue that it decouples measured benchmark safety from real-world safety: a system can score near-perfectly on the hazards an evaluation specifies while remaining blind to those 
    
[^2]: 人类放弃，推理模型坚持：区分难度登记与深思分配

    Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration from Deliberation Allocation

    [https://arxiv.org/abs/2606.26502](https://arxiv.org/abs/2606.26502)

    论文发现，在相同问题上，人类答错时花更少时间（放弃），而大型推理模型答错时花更多令牌（坚持），揭示了人类与AI在失败时资源分配策略的根本差异。

    

    arXiv:2606.26502v1 公告类型：新 摘要：大型推理模型（LRMs）在更困难的问题上花费更长时间，就像人类一样。这种表面上的相似性掩盖了项目内部的相反模式。当LRM答错一个问题时，它花费的令牌数比答对同一个问题时更多；而人类则相反，在答错的试验上花费的时间更少。我们将深思的两个层面分开：反应时间如何跨项目追踪难度（登记），以及在项目身份固定的情况下，智能体是在自己的失败还是成功上花费更多（分配）。在一个公开的人-LRM匹配语料库上，人类和所有五种思维LRM都再现了已知的跨项目一致性（登记），但在项目内部分配中出现了分歧：每个LRM都显示出显著的“答错 vs 答对”效应（在H-ARC上Cohen's d = 1.47-3.13），而人类则显示出相反的符号。比较是在每个智能体自身的尺度内进行的；我们从未将秒和令牌放在同一轴上。在项目固定的情况下，这种分离依然成立。

    arXiv:2606.26502v1 Announce Type: new  Abstract: Large reasoning models (LRMs) take longer on harder problems, just as humans do. This surface similarity hides an opposite pattern within items. When an LRM gets a problem wrong, it spends more tokens than when it gets the same problem right; humans do the reverse, spending less time on the trials they get wrong. We separate two levels of deliberation: how response time tracks difficulty across items (registration), and, with item identity held fixed, whether an agent spends more on its own failures or successes (allocation). On a public matched human-LRM corpus, humans and all five thinking LRMs reproduce the known cross-item alignment (registration) but diverge within items (allocation): every LRM shows a large wrong-vs-right effect (Cohen's d = 1.47-3.13 on H-ARC) while humans show the opposite sign. The comparison stays inside each agent's own scale; we never put seconds and tokens on one axis. The dissociation holds under item fixed
    

