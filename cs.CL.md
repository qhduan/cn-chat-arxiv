# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377) | DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。 |
| [^2] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^3] | [Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"](https://arxiv.org/abs/2606.26783) | 本研究复现了AlphaEdit知识编辑方法，发现其原始结果基本可重复，但在流畅性指标上存在差异，且该方法在新型模型架构上的优势不具有普遍性。 |
| [^4] | [The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report](https://arxiv.org/abs/2606.26529) | 论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。 |
| [^5] | [An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models](https://arxiv.org/abs/2606.09843) | 本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。 |

# 详细

[^1]: DanceOPD：在线策略生成场蒸馏

    DanceOPD: On-Policy Generative Field Distillation

    [https://arxiv.org/abs/2606.27377](https://arxiv.org/abs/2606.27377)

    DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。

    

    现代图像生成需要一个统一的模型，能够集成多种能力，包括文生图、局部编辑和全局编辑。然而，这些能力很少自然对齐，且常常相互冲突。例如，编辑往往会降低文生图的性能，而全局编辑与局部编辑也会相互干扰。因此，如何有效组合这些能力已成为图像生成模型训练的核心挑战。为了解决这一问题，我们提出了DanceOPD，一种用于流匹配模型的在线策略生成场蒸馏框架。该框架将每个样本路由至一个能力场，查询一个低噪声的学生诱导状态，并通过简单的速度均方误差目标进行训练。每个能力源被定义为共享流状态空间上的速度场，学生通过在其自身生成状态上查询这些场来学习组合专家能力。该公式还吸收了操作符依赖。

    arXiv:2606.27377v1 Announce Type: cross  Abstract: Modern image generation demands a single model that unifies diverse capabilities, including text-to-image (T2I), local editing, and global editing. However, these capabilities are rarely naturally aligned and often conflict. For instance, editing tends to degrade T2I performance, while global and local editing interfere with each other. Consequently, effectively composing these capabilities has become a central challenge for image generation model training. To tackle this, we introduce DanceOPD, an on-policy generative field distillation framework for flow-matching models that routes each sample to one capability field, queries one low-noise student-induced state, and trains with a simple velocity MSE objective. With each capability source defined as a velocity field over the shared flow state space, the student learns from fields queried on its own rollout states to compose expert capabilities. This formulation also absorbs operator-d
    
[^2]: CARVE：面向分块并行线性注意力的内容感知递归与价值效率模型

    CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention

    [https://arxiv.org/abs/2606.27229](https://arxiv.org/abs/2606.27229)

    CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。

    

    arXiv:2606.27229v1 公告类型：跨领域 摘要：递归模型必须通过遗忘来记住，然而现有技术决定遗忘什么时并不参考已存储的内容——门控机制仅看到当前到达的标记，而非即将修改的记忆。这种记忆盲区门控是当前主流delta规则架构（GDN-2）中三个相互耦合的缺陷之一：价值轴擦除掩码在价值投影的尺度上浪费参数，并且——正如我们所证明的——在数学上阻碍了使递归训练与Transformer相媲美的WY形式三角分块求解器。我们提出CARVE（内容感知递归与价值效率模型），通过一个原则解决所有三个问题：仅在键轴上擦除。这在数学上被证明是WY形式求解器保持有效的必要且充分条件。在此框架下，CARVE复用已写入GPU内存的递归输出张量作为擦除门控的免费内容信号，并替换逐值写入门控。

    arXiv:2606.27229v1 Announce Type: cross  Abstract: Recurrent models must forget in order to remember, yet the state of the art decides what to erase without consulting what is stored -- the gate sees only the arriving token, not the memory it is about to modify. This memory-blind gating is one of three coupled defects in the leading delta-rule architecture (GDN-2): the value-axis erase mask wastes parameters at the scale of the value projection, and -- as we prove -- mathematically prevents the WY-form triangular chunk solver that makes recurrent training competitive with Transformers.   We introduce CARVE (Content-Aware Recurrent with Value Efficiency), which resolves all three problems through one principle: erase only on the key axis. This is provably necessary and sufficient for the WY-form solver to remain valid. Within it, CARVE reuses the recurrent output tensor -- already written to GPU memory -- as a free content signal for the erase gate, and replaces the per-value write-gate
    
[^3]: “AlphaEdit：语言模型的零空间约束知识编辑”的可重复性研究

    Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"

    [https://arxiv.org/abs/2606.26783](https://arxiv.org/abs/2606.26783)

    本研究复现了AlphaEdit知识编辑方法，发现其原始结果基本可重复，但在流畅性指标上存在差异，且该方法在新型模型架构上的优势不具有普遍性。

    

    Fang等人（2025）提出了一种名为AlphaEdit的零空间约束投影方法，用于“定位-编辑”式知识编辑技术，该方法在理论上保证了编辑操作不会破坏先前保存的知识，并在LLaMA3、GPT2-XL和GPT-J上报告了相较于现有编辑方法的显著性能提升。本研究对AlphaEdit进行了可重复性验证，在原始实验设置下复现了其报告的结果，并沿着三个方向扩展了评估：新的模型架构、额外的下游基准测试以及更长的序列编辑范围。我们成功地在原始模型上复现了AlphaEdit报告的指标，但在报告的流畅性和一致性指标上发现了一处差异。将AlphaEdit扩展到更新的模型系列后，我们发现其优势并未普遍适用，这归因于“定位-编辑”范式中的架构假设。

    arXiv:2606.26783v1 Announce Type: cross  Abstract: Fang et al. (2025) introduced a null-space constrained projection, named AlphaEdit, for locate-then-edit knowledge editing methods, theoretically guaranteeing that edits do not disrupt previously preserved knowledge, and reports substantial gains over existing editing methods on LLaMA3, GPT2-XL, and GPT-J. In this work, we present a reproducibility study of AlphaEdit, reproducing its reported results under the original experimental setup and extending the evaluation along three axes: new model architectures, additional downstream benchmarks, and substantially longer sequential editing horizons. We successfully reproduce AlphaEdit's reported metrics across the original models, though we identify a discrepancy in the reported fluency and consistency metric. Extending AlphaEdit to newer model families, we find that its advantage does not generalize uniformly, which we trace to architectural assumptions in the locate-then-edit paradigm tha
    
[^4]: 注意盲区：任务条件化的语言与视觉模型会遗漏它们本可报告的安全关键信号

    The Inattentional Gap: Task-Conditioned Language and Vision Models Omit the Safety-Critical Signals They Can Otherwise Report

    [https://arxiv.org/abs/2606.26529](https://arxiv.org/abs/2606.26529)

    论文发现，当语言或视觉模型被限定于特定任务时，会系统性地抑制报告同时出现的其他安全关键信号，导致基准测试安全性与真实安全性脱钩。

    

    arXiv:2606.26529v1 公告类型：交叉 摘要：人工智能安全性通常通过模型能否可靠地检测指定危害来评估，但事故往往源于无人指定的危害。我们证明，将语言或视觉模型限定于狭窄任务会抑制其报告同时存在的、本可报告的安全关键信号，这类似于人类注意盲区的机器模拟，但源于不同的机制。在放射学与驾驶文本场景以及胸部X光片视觉任务中，所有测试模型均出现抑制现象，且不随模型规模减小，在推理模型中持续存在，不同模型家族间的差异大于模型大小间的差异，而同一模型在无约束条件下报告这些信号的比率显著更高。我们将这种分离命名为“注意盲区”，并论证它使基准安全性评估与真实世界安全性脱钩：系统可在评估指定的危害上获得近乎完美的分数，却对未指定的危害视而不见。

    arXiv:2606.26529v1 Announce Type: cross  Abstract: AI safety is evaluated by how reliably a model detects the hazards it is told to find, yet accidents often arise from the hazard no one specified. We show that conditioning a language or vision model on a narrow task suppresses its reporting of co-present, safety-critical signals it can otherwise report, a machine analogue of human inattentional blindness arising from a different mechanism. Across radiology and driving text scenarios and chest-radiograph vision tasks, suppression appeared in every model tested, did not diminish with scale, persisted in a reasoning model, and varied more by model family than by size, while the same models reported these signals at substantially higher rates when unconstrained. We name this dissociation the Inattentional Gap and argue that it decouples measured benchmark safety from real-world safety: a system can score near-perfectly on the hazards an evaluation specifies while remaining blind to those 
    
[^5]: 一种基于大语言模型原生的心理测量工具无法预测大语言模型行为：来自25个模型的证据

    An LLM-Native Psychometric Instrument Does Not Predict LLM Behavior: Evidence Across 25 Models

    [https://arxiv.org/abs/2606.09843](https://arxiv.org/abs/2606.09843)

    本研究构建了首个从LLM行为中自下而上推导的心理测量工具，发现其维度（响应性、服从性、大胆性、谨慎性和冗长性）高度可靠，但LLM的自我报告仍无法预测其实际行为，表明人类特质类别与LLM行为之间存在根本性差异。

    

    大语言模型（LLMs）对人格问卷给出了稳定的回答，但这些自我报告未能预测模型的实际行为。这种差距是源于将人类特质类别强加给LLMs的人为产物，还是源于LLM自我报告本身的更深层问题？为了探究这一点，我们构建了首个心理测量工具，其维度是从LLM行为中自下而上推导出来的，而非借用人类心理学。我们向来自17个模型家族的25个LLM（每个模型重复30次）施测了300个条目（240个李克特量表+60个情景题），探索性因素分析揭示了五个可复制且高度可靠的因素：响应性、服从性、大胆性、谨慎性和冗长性（所有Tucker $\phi \geq .957$，所有$\alpha \geq .930$）。随后，我们收集了2500个开放式行为样本，并由151名人类和三人LLM评判团进行评分。人类与评判团对模型行为的看法一致（平均相关系数$r = .51$），但自我报告未能预测这些行为。

    arXiv:2606.09843v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) give stable answers to personality questionnaires, yet these self-reports fail to predict how the models actually behave. Is this gap an artifact of forcing human trait categories onto LLMs, or something deeper about LLM self-report itself? To find out, we built the first psychometric instrument whose dimensions are derived bottom-up from LLM behavior rather than borrowed from human psychology. Administering 300 items (240 Likert + 60 scenario) to 25 LLMs across 17 model families, 30 times each, exploratory factor analysis revealed five replicable, highly reliable factors: Responsiveness, Deference, Boldness, Guardedness, and Verbosity (all Tucker $\phi \geq .957$, all $\alpha \geq .930$). We then collected 2,500 open-ended behavioral samples and had them rated by 151 humans and a three-judge LLM ensemble. Humans and judges agreed about model behavior ($\bar{r} = .51$), but self-report predicted nei
    

