# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems](https://arxiv.org/abs/2606.27243) | NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。 |
| [^2] | [The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development](https://arxiv.org/abs/2606.27045) | 提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。 |
| [^3] | [A Deterministic Control Plane for LLM Coding Agents](https://arxiv.org/abs/2606.26924) | 针对LLM编码代理配置缺乏管理的问题，提出一个确定性控制平面，通过将代理定义作为受管制的可追踪工件来管理，从而解决配置传播、低修订率和权限声明缺失等关键空白。 |
| [^4] | [Context-Aware Synthesis of Optimization Pipelines for Warehouse Optimization](https://arxiv.org/abs/2606.26852) | 提出CASOP框架，通过模块化算法库和上下文感知机制，自动合成并评估适用于特定仓库场景的优化流水线，解决了现有研究缺乏通用组合与评估方法的问题。 |
| [^5] | [Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs](https://arxiv.org/abs/2606.26492) | 本研究揭示了深度学习程序故障诊断中程序内评估与跨程序评估之间存在显著性能差距（平衡准确率差距达0.190），并发现该差距主要源于特征中的程序级结构。 |
| [^6] | [An Empirical Study of LLM-Generated Specifications for VeriFast](https://arxiv.org/abs/2606.26490) | 本研究系统评估了大型语言模型为分离逻辑验证器VeriFast生成C函数规范的能力，通过多种提示方法和模型对比，揭示了LLM在生成可验证规范方面的表现与局限性。 |
| [^7] | [Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries](https://arxiv.org/abs/2605.19576) | 本文首次系统性地识别、诊断并修复了自我进化型大语言模型技能库中的“库漂移”问题，通过可复现的触发条件、追踪级诊断工具和最小化治理方案，有效解决了无管理技能积累导致的性能停滞。 |
| [^8] | [Hierarchical Fault Detection and Diagnosis for Transformer Architectures](https://arxiv.org/abs/2604.28118) | 提出了一种名为DEFault++的层次化学习方法，能够自动检测Transformer模型中的隐蔽故障、定位受影响的组件并找出根本原因，同时构建了包含5556个标注运行实例的DEFault-bench基准测试集用于训练和评估。 |

# 详细

[^1]: NOVA：面向工业推荐系统架构演进的验证感知智能体框架

    NOVA: A Verification-Aware Agent Harness for Architecture Evolution in Industrial Recommender Systems

    [https://arxiv.org/abs/2606.27243](https://arxiv.org/abs/2606.27243)

    NOVA 通过引入“架构梯度”这一受SGD启发的不可微更新信号，实现了对工业推荐系统架构演进的验证感知自动化，解决了现有方法仅关注代码可运行性而忽视架构有效性的问题。

    

    arXiv:2606.27243v1 公告类型：新文 摘要：工业广告推荐模型通过架构演进持续改进。诸如RankMixer、TokenMixer-Large和MixFormer等升级表明，更好的结构仍然是质量和业务收益的关键来源。然而，在生产环境中开发此类升级需要大量专家投入且难以规模化。现有自动化方法存在不足：AutoML主要调优超参数，而有效收益往往需要在严格约束下进行跨模块更改；通用LLM编码智能体优化可运行代码，但可运行代码并不意味着有效的推荐架构。候选方案可能通过局部测试，但会导致静默故障从而降低性能。我们提出了NOVA，一种用于验证感知架构演进的层级感知智能体框架。NOVA使用架构梯度，这是一种受SGD启发的、不可微的更新信号，它聚合了先前的修改、验证诊断和指标反馈。

    arXiv:2606.27243v1 Announce Type: new  Abstract: Industrial advertising recommender models are continuously improved through architecture evolution. Upgrades such as RankMixer, TokenMixer-Large, and MixFormer show that better structures remain a key source of quality and business gains. Yet developing such upgrades in production is expert-intensive and difficult to scale. Existing automation is insufficient: AutoML mainly tunes hyper-parameters, while effective gains often require cross-module changes under strict constraints; generic LLM coding agents optimize for runnable code, but runnable code does not imply a valid recommender architecture. Candidates may pass local tests while causing silent failures that degrade performance.   We present NOVA, a level-aware agent harness for verification-aware architecture evolution. NOVA uses an architecture gradient, an SGD-inspired, non-differentiable update signal that aggregates prior modifications, verification diagnostics, metric feedback
    
[^2]: 规格增长引擎：面向AI辅助软件开发的规格锚定、代码耦合与漂移强制架构

    The Spec Growth Engine: Spec-Anchored, Code-Coupled, Drift-Enforced Architecture for AI-Assisted Software Development

    [https://arxiv.org/abs/2606.27045](https://arxiv.org/abs/2606.27045)

    提出规格增长引擎，通过机器可读规格图、主干上下文组装器、垂直切片增长协议和漂移门，解决AI编码代理的上下文爆炸和规格-代码漂移问题。

    

    arXiv:2606.27045v1 公告类型：交叉  摘要：AI编码代理极大地加速了实现速度，但引入了两种现有规格驱动方法未能完全解决的结构性失效模式：(1) 上下文爆炸——代理必须同时推理整个代码仓库，随着上下文窗口填满，输出质量下降；以及(2) 静默规格-代码漂移——代码演化，规格未更新，差异变得不可见，直到修复代价高昂才被发现。  我们提出了规格增长引擎，这是一个轻量级框架，通过以下方式解决这两种失效模式：一个机器可读的规格图，其节点携带明确的合同/设计分离；一个主干上下文组装器，将代理上下文限定到所有权路径；一个垂直切片增长协议，强制执行最难优先排序；以及一个漂移门，使规格-代码差异成为阻塞合并条件。  该设计综合了成熟的软件工程原则（Parnas信息隐藏）。

    arXiv:2606.27045v1 Announce Type: cross  Abstract: AI coding agents dramatically accelerate implementation speed but introduce two structural failure modes that existing spec-driven approaches do not fully solve: (1) context explosion -- the agent must reason over an entire repository at once, degrading output quality as the context window fills; and (2) silent spec-code drift -- code evolves, the specification does not, and the divergence becomes invisible until it is costly to repair.   We present the Spec Growth Engine, a lightweight framework that addresses both failure modes through a machine-readable spec graph whose nodes carry explicit contract/design separation, a Spine context assembler that scopes agent context to an ownership path, a vertical-slice growth protocol that enforces hardest-first ordering, and a drift gate that makes spec-code divergence a blocking merge condition.   The design synthesises well-established software engineering principles (Parnas information hidi
    
[^3]: 面向LLM编码代理的确定性控制平面

    A Deterministic Control Plane for LLM Coding Agents

    [https://arxiv.org/abs/2606.26924](https://arxiv.org/abs/2606.26924)

    针对LLM编码代理配置缺乏管理的问题，提出一个确定性控制平面，通过将代理定义作为受管制的可追踪工件来管理，从而解决配置传播、低修订率和权限声明缺失等关键空白。

    

    arXiv:2606.26924v1 公告类型：交叉 摘要：LLM编码工具赋予代理广泛的文件与shell访问权限，然而指导它们的配置层——规则文件、代理定义、IDE特定的markdown——在很大程度上是未受管理的。一项针对10,008个公共GitHub仓库（n=6,145个代理配置文件）的普遍性研究发现，代理配置作为未声明的共享组件传播：10.1%的追踪路径在不同独立仓库之间是SHA-256精确重复（经fork调整、阈值无关），其中75.5%的克隆对跨越组织边界。另外两种模式具有指示性：配置很少被修订（58%为单次提交；与CI/CD工作流按年龄标准化后，每月0.4次对比0.6次提交），并且很少声明权限边界（<1%的代理配置对比33%的Actions工作流，n=31个真阳性）。我们提出在工具之上建立一个确定性控制平面，该平面与这些空白一一对应。Rel(AI)Build将代理定义视为一种受管制的、可追踪的工件，而非临时脚本。

    arXiv:2606.26924v1 Announce Type: cross  Abstract: LLM coding harnesses grant agents broad file and shell access, yet the configuration layer that steers them -- rules files, agent definitions, IDE-specific markdown -- is largely unmanaged. A prevalence study of 10,008 public GitHub repositories (n=6,145 agent config files) finds that agent configurations propagate as undeclared shared components: 10.1% of tracked paths are SHA-256 exact duplicates across independent repositories (fork-adjusted, threshold-independent), with 75.5% of clone pairs crossing organisational boundaries. Two further patterns are indicative: configurations are rarely revised (58% single-commit; 0.4 vs 0.6 commits/month age-normalised against CI/CD workflows), and rarely declare permission boundaries (<1% of agent configs vs 33% of Actions workflows, n=31 true positives).   We propose a deterministic control plane above the harness that maps one-to-one to these gaps. Rel(AI)Build treats agent definitions as a ma
    
[^4]: 面向仓库优化的上下文感知优化流水线合成

    Context-Aware Synthesis of Optimization Pipelines for Warehouse Optimization

    [https://arxiv.org/abs/2606.26852](https://arxiv.org/abs/2606.26852)

    提出CASOP框架，通过模块化算法库和上下文感知机制，自动合成并评估适用于特定仓库场景的优化流水线，解决了现有研究缺乏通用组合与评估方法的问题。

    

    在人工拣货到货位仓库中，订单履行涉及商品分配、订单分批和拣货路径规划等相互关联的决策。虽然集成模型能够捕捉这些决策之间的交互，但由于组织边界、职责分工或数据可用性有限，实际仓库系统通常需要采用分解方法。现有研究主要针对孤立子问题或特定仓库场景下的固定子问题组合评估算法，但缺乏一种通用机制来确定适用的算法配置、将其组合成有效的求解流水线并评估其性能。为此，我们提出上下文感知优化流水线合成（CASOP）框架，用于构建和评估特定上下文下的优化流水线，并将其应用于订单履行。该框架包括：（1）一个针对常见订单履行子问题的模块化算法库。

    arXiv:2606.26852v1 Announce Type: new  Abstract: Order fulfillment in manual picker-to-goods warehouses involves interconnected decisions such as item assignment, order batching, and picker routing. While integrated models capture interactions between these decisions, practical warehouse systems often require decomposed approaches due to organizational boundaries, differing responsibilities, or limited data availability. Existing studies primarily evaluate algorithms for isolated subproblems or fixed subproblem combinations for specific warehouse settings, but lack a general mechanism to determine applicable algorithm configurations, compose them into valid solution pipelines, and assess their performance.   With Context-Aware Synthesis of Optimization Pipelines (CASOP), we propose a framework for constructing and evaluating context-specific optimization pipelines and apply these to order fulfillment. The framework comprises: (1) a modular repository of algorithms for common order fulf
    
[^5]: 深度学习程序故障诊断中的评估策略差距

    Evaluation-Strategy Gap in Fault Diagnosis of Deep Learning Programs

    [https://arxiv.org/abs/2606.26492](https://arxiv.org/abs/2606.26492)

    本研究揭示了深度学习程序故障诊断中程序内评估与跨程序评估之间存在显著性能差距（平衡准确率差距达0.190），并发现该差距主要源于特征中的程序级结构。

    

    深度学习程序在训练过程中可能因多种原因失败，诊断故障原因是一项成本高昂且耗时的维护任务。用于诊断此类故障的技术通常采用程序内交叉验证进行评估，但这可能不足以应对涉及未见过程的部署场景。因此，有必要评估不同场景下的性能差异，并识别现有深度学习故障诊断技术中性能差距的成因。我们利用DynFault数据集（包含38个真实世界深度学习程序的5542个注入故障的训练轨迹）研究了这一差距。我们发现，在程序内评估与保留完整程序评估之间，现有故障诊断技术的平衡准确率存在0.190的差距。我们还发现，这一差距源于特征中的程序级结构，这促使我们研究了两种运行时特征集：曲率特征和优化器特征。

    arXiv:2606.26492v1 Announce Type: cross  Abstract: Deep Learning (DL) programs can fail during training for many reasons, and diagnosing the cause is a costly and time-consuming maintenance task. Techniques for diagnosing such failures are commonly assessed using within-program cross-validation, which may be inadequate for deployment settings involving previously unseen programs. It is therefore necessary to assess how performance differs across these settings and to identify the causes of any performance gap in established fault diagnosis techniques for DL. We investigate this gap using DynFault, a corpus of 5,542 fault-injected training traces from 38 real-world DL programs. We found a gap of 0.190 in balanced accuracy for existing fault diagnosis techniques between within-program evaluation and holding out whole programs. We also found the gap comes from program-level structure in the features, which led us to examine two runtime feature sets, curvature features and optimizer featur
    
[^6]: 大型语言模型生成VeriFast规范的经验研究

    An Empirical Study of LLM-Generated Specifications for VeriFast

    [https://arxiv.org/abs/2606.26490](https://arxiv.org/abs/2606.26490)

    本研究系统评估了大型语言模型为分离逻辑验证器VeriFast生成C函数规范的能力，通过多种提示方法和模型对比，揭示了LLM在生成可验证规范方面的表现与局限性。

    

    arXiv:2606.26490v1 公告类型：交叉 摘要：静态验证工具可以确保工业规模软件的质量，但需要大量人力来编写规范。这一点对于基于分离逻辑的静态验证器（SL验证器）尤为突出，这类验证器擅长验证堆操作程序，但需要许多复杂的辅助规范来推理堆结构。最近的研究将大型语言模型（LLMs）应用于生成代码、测试和证明，包括验证器规范，但主要针对非SL验证器。为填补这一空白，本文全面评估了LLMs在提示生成规范以使用SL验证器VeriFast验证303个C函数时的表现。我们分两个阶段探索了八种提示方法、十种LLM和三种输入类型。通过定量和定性分析，评估了LLM生成的代码和规范在功能行为、可验证性和错误方面的表现。结果表明，L

    arXiv:2606.26490v1 Announce Type: cross  Abstract: Static verification tools can assure industrial scale software, but require significant human labor to write specifications. This is particularly true of static verifiers based on separation logic (SL verifiers), which excel at verifying heapmanipulating programs, but require many complex auxiliary specifications to reason about heap structure. Recent work applies large language models (LLMs) to generate code, tests, and proofs, including specifications for verifiers, but mostly targeting non-SL verifiers. To address this gap, this paper thoroughly evaluates how well LLMs perform when prompted to generate specifications for verifying 303 C functions with the SL verifier VeriFast. We explored eight prompting approaches, ten LLMs, and three input types in two stages. Quantitative and qualitative analyses are used to assess the LLM-generated code and specifications for functional behavior, verifiability and errors. The results show that L
    
[^7]: 库漂移：诊断与修复自我进化型大语言模型技能库中的一种静默失效模式

    Library Drift: Diagnosing and Fixing a Silent Failure Mode in Self-Evolving LLM Skill Libraries

    [https://arxiv.org/abs/2605.19576](https://arxiv.org/abs/2605.19576)

    本文首次系统性地识别、诊断并修复了自我进化型大语言模型技能库中的“库漂移”问题，通过可复现的触发条件、追踪级诊断工具和最小化治理方案，有效解决了无管理技能积累导致的性能停滞。

    

    arXiv:2605.19576v2 公告类型：替换 摘要：自我进化的技能库面临一种我们称之为“库漂移”的静默失效模式：无限制的技能积累缺乏基于结果的生命周期管理，导致检索退化、误报注入以及性能停滞。最近的评估证实了该症状（大语言模型生成的技能带来+0.0%的提升，而人工整理的技能带来+16.2%的提升（SkillsBench）），但其根本机制尚未被分离。我们提供：（1）一个**可复现的触发条件**：通过消融实验隔离漂移：一项禁用技能注入（平坦基线，+0.002），另一项强制过早淘汰（主动损害，-0.019）；（2）**追踪级别的诊断**：一个仅追加的日志，包含每项技能的贡献分数、归属判定和路由器参与指标，使故障在影响最终任务得分之前即可被发现；（3）**经过验证的修复方案**：一个最小化的治理方案（基于结果的淘汰 + 有界活跃缓存）。

    arXiv:2605.19576v2 Announce Type: replace  Abstract: Self-evolving skill libraries face a silent failure mode we term \emph{library drift}: unbounded skill accumulation without outcome-driven lifecycle management causes retrieval degradation, false-positive injections, and performance stagnation. Recent evaluation confirms the symptom (LLM-authored skills deliver +0.0pp gain while human-curated ones deliver +16.2pp (SkillsBench)), yet the underlying mechanism has not been isolated. We provide (1) a \textbf{reproducible trigger}: ablations that isolate drift: one disables skill injection (flat floor, +0.002), one imposes premature retirement (active harm, $-$0.019); (2) \textbf{trace-level diagnostics}: an append-only evidence log with per-skill contribution scores, attribution verdicts, and router engagement metrics that make the failure visible before it reaches end-task scores; and (3) a \textbf{verified fix}: a minimal governance recipe (outcome-driven retirement + bounded active-ca
    
[^8]: 针对Transformer架构的层次化故障检测与诊断方法

    Hierarchical Fault Detection and Diagnosis for Transformer Architectures

    [https://arxiv.org/abs/2604.28118](https://arxiv.org/abs/2604.28118)

    提出了一种名为DEFault++的层次化学习方法，能够自动检测Transformer模型中的隐蔽故障、定位受影响的组件并找出根本原因，同时构建了包含5556个标注运行实例的DEFault-bench基准测试集用于训练和评估。

    

    arXiv:2604.28118v2 公告类型：替换交叉 摘要：Transformer模型如今支撑着工业界和研究领域的众多关键AI系统。然而，其故障可能在不触发运行时错误的情况下悄然改变模型行为，而现有技术几乎无法将这些故障追溯至具体组件及其根本原因。这类故障之所以难以检测，是因为损失函数值和数值指标保持正常，且可见的症状很少能指明具体是哪个组件出了问题。我们提出了DEFault++，一种基于层次化学习的技术，它首先检测故障，然后识别受影响的组件，最后定位组件内的具体原因，从而帮助开发者高效地调试Transformer模型。DEFault++通过故障传播图（FPG）——一种基于架构依赖路径的结构先验——来组织组件级的运行时测量，并报告每项诊断背后的证据。为了训练和评估该方法，我们构建了DEFault-bench基准测试集，该基准包含来自跨模型变异测试的5,556个带标签的运行实例。

    arXiv:2604.28118v2 Announce Type: replace-cross  Abstract: Transformers now underpin critical AI systems across industry and research. Yet their faults can silently alter model behavior without runtime errors, and existing techniques offer little support for tracing these failures to their component and root cause. Such faults evade detection because loss and numerical values stay normal, and the visible symptom rarely identifies the component responsible. We present DEFault++, a hierarchical learning-based technique that first detects a fault, then identifies the affected component, and finally the cause within it, helping developers effectively debug transformer models. DEFault++ organizes component-level runtime measurements with a Fault Propagation Graph (FPG), a structural prior over the architecture's dependency paths, and reports the evidence behind each diagnosis. To train and evaluate it, we construct DEFault-bench, a benchmark of 5,556 labeled runs from mutation testing acros
    

