# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DanceOPD: On-Policy Generative Field Distillation](https://arxiv.org/abs/2606.27377) | DanceOPD提出了一种在线策略生成场蒸馏框架，通过将不同图像生成能力（文生图、局部编辑、全局编辑）建模为共享空间中的速度场，并利用学生自身状态进行查询和训练，有效解决了多种能力之间的冲突与组合问题。 |
| [^2] | [CARVE: Content-Aware Recurrent with Value Efficiency for Chunk-Parallel Linear Attention](https://arxiv.org/abs/2606.27229) | CARVE通过仅在键轴上擦除的单一原则，解决了递归模型中的记忆盲区门控、参数浪费和WY形式求解器失效三个问题，实现了高效的内容感知递归线性注意力。 |
| [^3] | [Reproducibility Study of "AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models"](https://arxiv.org/abs/2606.26783) | 本研究复现了AlphaEdit知识编辑方法，发现其原始结果基本可重复，但在流畅性指标上存在差异，且该方法在新型模型架构上的优势不具有普遍性。 |
| [^4] | [Label Hierarchy Transition: Delving into Class Hierarchies to Enhance Deep Classifiers.](http://arxiv.org/abs/2112.02353) | 本文提出了Label Hierarchy Transition (LHT)框架，基于深度学习，用于改进层次分类。LHT框架主要包括转换网络和混淆损失两个部分，通过显式学习标签层次转换矩阵和鼓励分类网络处理混淆情况，有效地利用类层次结构的相关性。 |

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
    
[^4]: 标签层级转换：深入研究类层次结构以增强深度分类器

    Label Hierarchy Transition: Delving into Class Hierarchies to Enhance Deep Classifiers. (arXiv:2112.02353v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2112.02353](http://arxiv.org/abs/2112.02353)

    本文提出了Label Hierarchy Transition (LHT)框架，基于深度学习，用于改进层次分类。LHT框架主要包括转换网络和混淆损失两个部分，通过显式学习标签层次转换矩阵和鼓励分类网络处理混淆情况，有效地利用类层次结构的相关性。

    

    层次分类旨在将对象按照类别的层次结构进行排序。现有方法通常通过将其解耦为一系列多类别分类任务来处理层次分类。然而，这种多任务学习策略未能充分利用层次结构不同层级之间各个类别之间的相关性。在本文中，我们提出了一种基于深度学习的统一概率框架Label Hierarchy Transition (LHT)，以应对层次分类的挑战。LHT框架由一个转换网络和一个混淆损失组成。转换网络专注于显式学习标签层次转换矩阵，这有助于有效地编码嵌入在类层次结构中的潜在相关性。混淆损失鼓励分类网络学习更好地处理类别之间的混淆情况。

    Hierarchical classification aims to sort the object into a hierarchical structure of categories. For example, a bird can be categorized according to a three-level hierarchy of order, family, and species. Existing methods commonly address hierarchical classification by decoupling it into a series of multi-class classification tasks. However, such a multi-task learning strategy fails to fully exploit the correlation among various categories across different levels of the hierarchy. In this paper, we propose Label Hierarchy Transition (LHT), a unified probabilistic framework based on deep learning, to address the challenges of hierarchical classification. The LHT framework consists of a transition network and a confusion loss. The transition network focuses on explicitly learning the label hierarchy transition matrices, which has the potential to effectively encode the underlying correlations embedded within class hierarchies. The confusion loss encourages the classification network to le
    

