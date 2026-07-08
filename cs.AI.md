# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^2] | [Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions](https://arxiv.org/abs/2606.27106) | 本文提出了一种结合跨学科风险模型、开源情报媒体收集和大语言模型的新方法，用于维和任务的威胁评估，验证了其自动结果与人类判断的高度一致性。 |
| [^3] | [The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report](https://arxiv.org/abs/2606.26529) | 论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。 |
| [^4] | [Accelerating Returns and the Qualitative Engine for Science](https://arxiv.org/abs/2606.26359) | 本文指出，即使技术进步呈现加速回报的指数增长趋势，这主要提升执行和基础设施能力，而科学发现的核心——识别框架结构缺陷并进行概念创新——依赖于不同的定性推理能力，人类在这方面仍具优势。 |
| [^5] | [An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models](https://arxiv.org/abs/2606.09843) | 本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。 |

# 详细

[^1]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^2]: 大语言模型在外围维和任务威胁评估中的应用

    Application of LLMs to Threat Assessment of Foreign Peacekeeping Missions

    [https://arxiv.org/abs/2606.27106](https://arxiv.org/abs/2606.27106)

    本文提出了一种结合跨学科风险模型、开源情报媒体收集和大语言模型的新方法，用于维和任务的威胁评估，验证了其自动结果与人类判断的高度一致性。

    

    我们提出了一种新颖的方法，将大语言模型应用于外围维和任务中的威胁评估。基于PINPOINT项目及其用例——欧盟驻格鲁吉亚监测团，我们结合了跨学科风险模型、基于开源情报的媒体收集以及大语言模型支持的威胁提取。所提出的工作流程将媒体内容映射到与任务相关的威胁，提取结构化信息，并应用多个基于大语言模型的额外处理步骤以提高相关性和基础性。对从媒体文档中提取的威胁进行的评估显示，在威胁和任务相关性等核心方面，自动生成的结果与人类判断之间具有高度一致性。这些结果表明，大语言模型为支持维和任务中的分析人员提供了一种有前景的方法。

    arXiv:2606.27106v1 Announce Type: cross  Abstract: We present a novel approach for applying Large Language Models (LLMs) to threat assessment in the context of foreign peacekeeping missions. Building on the PINPOINT project and its use case, the EU Monitoring Mission in Georgia, we combine an interdisciplinary risk-model with OSINT-based media collection and LLM-supported threat extraction. The proposed workflow maps media contents to mission-relevant threats, extracts structured information and applies several additional LLM-based processing steps to improve relevance and grounding. An evaluation of threats extracted from media documents shows high agreement between automatically generated results and human judgment for core aspects such as threat and mission relevance. These results indicate that LLMs provide a promising approach to support analysts in the context of peacekeeping missions.
    
[^3]: 注意盲区：任务条件化的语言与视觉模型会遗漏它们本可报告的安全关键信号

    The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report

    [https://arxiv.org/abs/2606.26529](https://arxiv.org/abs/2606.26529)

    论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。

    

    arXiv:2606.26529v1 公告类型：交叉 摘要：人工智能安全性通常通过模型能否可靠地检测指定危害来评估，但事故往往源于无人指定的危害。我们证明，将语言或视觉模型限定于狭窄任务会抑制其报告同时存在的、本可报告的安全关键信号，这类似于人类注意盲区的机器模拟，但源于不同的机制。在放射学与驾驶文本场景以及胸部X光片视觉任务中，所有测试模型均出现抑制现象，且不随模型规模减小，在推理模型中持续存在，不同模型家族间的差异大于模型大小间的差异，而同一模型在无约束条件下报告这些信号的比率显著更高。我们将这种分离命名为“注意盲区”，并论证它使基准安全性评估与真实世界安全性脱钩：系统可在评估指定的危害上获得近乎完美的分数，却对未指定的危害视而不见。

    arXiv:2606.26529v1 Announce Type: cross  Abstract: AI safety is evaluated by how reliably a model detects the hazards it is told to find, yet accidents often arise from the hazard no one specified. We show that conditioning a language or vision model on a narrow task suppresses its reporting of co-present, safety-critical signals it can otherwise report, a machine analogue of human inattentional blindness arising from a different mechanism. Across radiology and driving text scenarios and chest-radiograph vision tasks, suppression appeared in every model tested, did not diminish with scale, persisted in a reasoning model, and varied more by model family than by size, while the same models reported these signals at substantially higher rates when unconstrained. We name this dissociation the Inattentional Gap and argue that it decouples measured benchmark safety from real-world safety: a system can score near-perfectly on the hazards an evaluation specifies while remaining blind to those 
    
[^4]: 加速回报与科学中的定性引擎

    Accelerating Returns and the Qualitative Engine for Science

    [https://arxiv.org/abs/2606.26359](https://arxiv.org/abs/2606.26359)

    本文指出，即使技术进步呈现加速回报的指数增长趋势，这主要提升执行和基础设施能力，而科学发现的核心——识别框架结构缺陷并进行概念创新——依赖于不同的定性推理能力，人类在这方面仍具优势。

    

    arXiv:2606.26359v1 公告类型：新 摘要：雷·库兹韦尔提出了一种加速回报的论点，这是技术进步讨论中最具影响力的叙述之一。其核心主张是，多个技术领域（尤其是计算、人工智能、脑科学和生物技术）的进步相互影响，使得进步呈现自放大且近似指数增长的趋势。本文对这一主张给出了一个简单的数学解释，然后论证，即使这种加速是真实的，它本身并不能解决科学发现的核心问题。原因是加速回报最自然地适用于执行和基础设施能力，而真正的发现往往依赖于另一种能力：关于当前框架在结构上何时不足以及下一步需要何种概念转变的定性推理。最近的ARC-AGI-3结果进一步凸显了这一区别：人类能够解决该基准测试。

    arXiv:2606.26359v1 Announce Type: new  Abstract: Ray Kurzweil described a thesis of accelerating returns, which is the most influential narratives in discussions of technological progress. Its central claim is that advances in multiple technological fields, especially compute, artificial intelligence, brain science, and biotechnology, interact in such a way that progress becomes self-amplifying and approximately exponential. This paper gives a simple mathematical interpretation of that claim and then argues that, even if such acceleration is real, it does not by itself resolve the central problem of scientific discovery. The reason is that accelerating returns apply most naturally to executional and infrastructural capability, whereas genuine discovery often depends on a different capacity: qualitative reasoning about when a current framework is structurally inadequate and what conceptual move is needed next. Recent ARC-AGI-3 results sharpen this distinction: humans solve the benchmark
    
[^5]: 一种基于大语言模型原生的心理测量工具无法预测大语言模型行为：来自25个模型的证据

    An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models

    [https://arxiv.org/abs/2606.09843](https://arxiv.org/abs/2606.09843)

    本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。

    

    大语言模型（LLMs）对人格问卷给出了稳定的回答，但这些自我报告未能预测模型的实际行为。这种差距是源于将人类特质类别强加给LLMs的人为产物，还是源于LLM自我报告本身的更深层问题？为了探究这一点，我们构建了首个心理测量工具，其维度是从LLM行为中自下而上推导出来的，而非借用人类心理学。我们向来自17个模型家族的25个LLM（每个模型重复30次）施测了300个条目（240个李克特量表+60个情景题），探索性因素分析揭示了五个可复制且高度可靠的因素：响应性、服从性、大胆性、谨慎性和冗长性（所有Tucker $\phi \geq .957$，所有$\alpha \geq .930$）。随后，我们收集了2500个开放式行为样本，并由151名人类和三人LLM评判团进行评分。人类与评判团对模型行为的看法一致（平均相关系数$r = .51$），但自我报告未能预测这些行为。

    arXiv:2606.09843v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models actually behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report itself? To find out, we built the first psychometric instrument whose dimensions are derived bottom-up from LLM behavior rather than borrowed from human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five replicable, highly reliable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $\phi \geq .957$, all $\alpha \geq .930$). We then collected 2,500 open-ended behavioral samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed about model behavior ($\bar{r} = .51$), but self-report predicted nei
    

