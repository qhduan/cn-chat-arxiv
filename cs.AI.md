# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds](https://arxiv.org/abs/2606.26964) | 提出一种“先看再动”的摄像机规划框架，通过将观察规范与运动执行分离，使动态3D故事世界中的具身观察者能在叙事引导下主动决定观察内容和注意力转移方式。 |
| [^2] | [A Pipeline for Generating Longitudinal Synthetic Clinical Notes Using Large Language Models](https://arxiv.org/abs/2606.26879) | 本文提出了一种结合结构化生成、半结构化模拟和LLM非结构化笔记生成的模块化流水线，用于生成内部一致且风格多样的纵向合成临床数据，以支持临床AI开发并规避隐私风险。 |
| [^3] | [AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems](https://arxiv.org/abs/2606.26859) | AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。 |
| [^4] | [\textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models](https://arxiv.org/abs/2606.26530) | 本文提出DiARC方法，通过构建正负样本偏好对来提升大型语言模型在类ARC任务中的推理能力，弥补了仅依赖正样本监督的不足。 |
| [^5] | [Boundary-Aware Context Grounding for A Low-Channel EEG Agent](https://arxiv.org/abs/2606.26519) | 本文提出了一种名为NeuraDock Agent的开源架构，通过将确定性本地脑电引擎与硬件感知语言层分离，并仅向大型语言模型提供紧凑许可摘要和版本化上下文包，实现了对低通道脑电数据的边界感知上下文接地，从而避免产生缺乏支持的解读。 |
| [^6] | [Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare](https://arxiv.org/abs/2606.26104) | 研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。 |
| [^7] | [Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training](https://arxiv.org/abs/2606.26102) | 论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。 |
| [^8] | [Joint Reward Modeling: Internalizing Chain-of-Thought for Efficient Visual Reward Models](https://arxiv.org/abs/2602.07533) | 提出联合奖励建模（JRM）方法，通过在共享的视觉-语言骨干网络上联合优化偏好学习和语言建模，将生成模型的语义与推理能力内化到高效的判别式奖励模型中，解决了现有判别式模型推理不足和生成式模型推理成本高的问题。 |
| [^9] | [Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting](https://arxiv.org/abs/2601.16632) | 提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。 |
| [^10] | [SciFig: Towards Automating Editable Figure Generation for Scientific Papers](https://arxiv.org/abs/2601.04390) | SciFig是一个端到端的多智能体框架，通过将图形生成分解为规划、布局合成、组件渲染和迭代优化四个步骤，从科学文本中自动生成视觉丰富且完全可编辑的XML格式方法论图形，解决了现有方法在可编辑性和视觉质量之间的权衡问题。 |
| [^11] | ["Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets.](http://arxiv.org/abs/2308.05201) | 这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。 |

# 详细

[^1]: 先看再动：动态3D故事世界中的叙事引导型视觉注意力机制

    Look-Before-Move: Narrative-Grounded World Visual Attention in Dynamic 3D Story Worlds

    [https://arxiv.org/abs/2606.26964](https://arxiv.org/abs/2606.26964)

    提出一种“先看再动”的摄像机规划框架，通过将观察规范与运动执行分离，使动态3D故事世界中的具身观察者能在叙事引导下主动决定观察内容和注意力转移方式。

    

    随着具身AI和世界模型在动态3D环境中的广泛应用，视觉感知必须从被动解读已有观察，转向主动决定观察什么。我们通过动态3D故事世界中的摄像机规划来研究这一问题，摄像机不仅要生成平滑的运动，还要在移动之前决定应该获取哪些视觉证据。我们将这一能力定义为“叙事引导型世界视觉注意力”，其中摄像机作为一个具身观察者，决定观察什么、如何构图观察内容、以及如何在叙事意图和物理3D约束下随时间调整注意力焦点。为实现这一能力，我们提出了“先看再动”摄像机规划框架，它将观察规范与运动执行分离。该框架首先构建一个“语义观察合约”，将导演意图转化为可执行的视觉约束，然后据此执行运动。

    arXiv:2606.26964v1 Announce Type: new  Abstract: As embodied AI and world models increasingly operate in dynamic 3D environments, visual perception must move beyond passively interpreting given observations toward actively deciding what to observe. We study this problem through camera planning in dynamic 3D story worlds, where the camera must not only generate smooth motion, but also decide what visual evidence should be acquired before it moves. We formulate this capability as Narrative-Grounded World Visual Attention, where the camera acts as an embodied observer that determines what to observe, how to compose the observation, and how to shift attention over time under narrative intent and physical 3D constraints. To realize this capability, we propose Look-Before-Move, a camera planning framework that separates observation specification from motion execution. It first builds a Semantic Observation Contract to convert directorial intent into executable visual constraints, then perfor
    
[^2]: 使用大语言模型生成纵向合成临床笔记的流水线

    A Pipeline for Generating Longitudinal Synthetic Clinical Notes Using Large Language Models

    [https://arxiv.org/abs/2606.26879](https://arxiv.org/abs/2606.26879)

    本文提出了一种结合结构化生成、半结构化模拟和LLM非结构化笔记生成的模块化流水线，用于生成内部一致且风格多样的纵向合成临床数据，以支持临床AI开发并规避隐私风险。

    

    arXiv:2606.26879v1 公告类型：新 摘要：合成数据越来越多地被用于在真实世界数据访问受限的领域中开发和评估人工智能系统。在医疗领域，临床文档因其敏感性而面临特殊挑战。本研究介绍了一种合成临床笔记流水线和数据集，旨在支持临床人工智能工具的开发，同时避免真实患者数据带来的隐私风险。该数据集通过模块化流水线生成，该流水线结合了结构化患者生成、半结构化患者旅程模拟以及使用大语言模型生成非结构化临床笔记。该流水线设计旨在优先确保纵向患者记录的内部一致性，同时捕捉写作风格、笔记结构和临床细节的多样性。额外的机制（包括基于大语言模型的验证和增强步骤）被用于提高忠实度、真实性和完整性。

    arXiv:2606.26879v1 Announce Type: new  Abstract: Synthetic data is increasingly used to enable the development and evaluation of AI systems in domains where access to real-world data is restricted. In healthcare, clinical documentation presents particular challenges due to its sensitivity. This work introduces a synthetic clinical notes pipeline and dataset designed to support the development of clinical AI tools while avoiding the privacy risks associated with real patient data. The dataset is generated using a modular pipeline that combines structured patient generation, semi-structured patient journey simulation, and unstructured clinical note generation using large language models. The pipeline is designed to prioritise internal consistency across longitudinal patient records, while also capturing variation in writing style, note structure, and clinical detail. Additional mechanisms, including LLM-based validation and augmentation steps, are used to improve faithfulness, realism, a
    
[^3]: AgentX：迈向工业推荐系统智能体驱动的自我迭代

    AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems

    [https://arxiv.org/abs/2606.26859](https://arxiv.org/abs/2606.26859)

    AgentX是一个已部署的多智能体系统，通过自主生成、实施、评估和学习推荐实验，实现了工业推荐系统从依赖人工到自我迭代的范式转变，从而打破了创新瓶颈。

    

    arXiv:2606.26859v1 公告类型：新 摘要：推荐算法的迭代正从依赖工程师的手工过程向工业化研究循环转变，但这种转变仍被结构性执行瓶颈所阻碍：从想法到上线的周期仍然依赖人类工程师提出假设、修改生产代码、启动A/B实验并归因线上结果。因此，创新规模与人力成正比，而非与证据、计算资源和积累的实验知识复合增长。我们提出AgentX，一个已投入生产的、从根本上重构这一生产函数的多智能体系统。AgentX作为一个自我进化的开发引擎运作：它能自主生成、实现、评估并学习推荐实验，其规模和速度远超任何人工工作流所能维持的水平。该系统在一个闭环中编排四个紧密耦合的阶段。一个头脑风暴智能体从历史数据中综合证据。

    arXiv:2606.26859v1 Announce Type: new  Abstract: Recommendation algorithm iteration is moving from an artisanal, engineer-bound process toward an industrialized research loop, but this transition remains blocked by a structural execution bottleneck: the idea-to-launch cycle still depends on human engineers to generate hypotheses, modify production code, launch A/B experiments, and attribute online results. Innovation therefore scales linearly with headcount rather than compounding with evidence, compute, and accumulated experimental knowledge. We present AgentX, a production-deployed multi-agent system that fundamentally restructures this production function. AgentX operates as a self-evolving development engine: it autonomously generates, implements, evaluates, and learns from recommendation experiments at a scale and pace that no manual workflow can sustain.   The system orchestrates four tightly coupled stages in a closed loop. A Brainstorm Agent synthesizes evidence from historical
    
[^4]: \textsc{DiARC}：区分正负样本有助于提升大型语言模型的类ARC推理能力

    \textsc{DiARC}: Distinguishing Positive and Negative Samples Helps Improving ARC-like Reasoning Ability of Large Language Models

    [https://arxiv.org/abs/2606.26530](https://arxiv.org/abs/2606.26530)

    本文提出DiARC方法，通过构建正负样本偏好对来提升大型语言模型在类ARC任务中的推理能力，弥补了仅依赖正样本监督的不足。

    

    抽象与推理语料库（ARC；\citealp{chollet2019measure}）包含需要从有限网格样本中总结模式并预测输出网格的任务。近年来，许多基于大型语言模型的方法尝试将其转化为基于文本的推理任务。然而，基于开源模型的方法通常效果不佳，而依赖闭源模型的方法成本过高。当前工作主要集中在数据增强上，即构建类ARC数据以进行更全面的监督微调。在本研究中，我们认为解决类ARC问题不仅需要正样本监督，还需要通过区分负样本来提升模型的推理能力。为此，我们借鉴偏好对齐的思想，提出了\textsc{DiARC}方法，该方法构建偏好对，使模型能够区分正负样本。

    arXiv:2606.26530v1 Announce Type: cross  Abstract: The Abstraction and Reasoning Corpus (ARC;~\citealp{chollet2019measure}) contains tasks that require summarizing patterns from limited grid samples and predicting output grids. Recently, many large language model based approaches have attempted to transform it into a text-based reasoning task. However, methods based on open-source models have generally yielded unsatisfactory results, while those relying on closed-source models are too costly. Current efforts mainly focus on data augmentation, constructing ARC-like data for more comprehensive supervised fine-tuning. In this work, we argue that solving ARC-like problems requires not only \textit{positive} sample supervision but also the ability to improve model reasoning by distinguishing \textit{negative} samples. To this end, we draw on the idea of preference alignment and propose \textsc{DiARC}, a method that constructs preference pairs to enable the model to distinguish between them.
    
[^5]: 面向低通道脑电智能体的边界感知上下文接地

    Boundary-Aware Context Grounding for A Low-Channel EEG Agent

    [https://arxiv.org/abs/2606.26519](https://arxiv.org/abs/2606.26519)

    本文提出了一种名为NeuraDock Agent的开源架构，通过将确定性本地脑电引擎与硬件感知语言层分离，并仅向大型语言模型提供紧凑许可摘要和版本化上下文包，实现了对低通道脑电数据的边界感知上下文接地，从而避免产生缺乏支持的解读。

    

    大型语言模型（LLM）可以使科学软件更易于使用。然而，通用模型并不能自动知道特定传感器支持哪些测量、当前软件实现了哪些算法，或计算出的结果能证明哪些结论。这些区分对于低通道脑电图（EEG）尤为重要，因为稀疏的空间覆盖和可变的信号质量使得看似合理但缺乏支持的解读容易产生。我们提出了NeuraDock Agent，一种开源架构，它将确定性的本地脑电引擎与硬件感知的语言层分离。数值引擎解析记录、执行质量控制、运行经过评审的频谱工作流程，并生成机器可读的产物。LLM仅接收一个紧凑的、经过许可的摘要和一个版本化的上下文包。该上下文描述了七通道硬件、经过评审的工作流程以及结果文件。

    arXiv:2606.26519v1 Announce Type: new  Abstract: Large language models (LLMs) can make scientific software easier to use. However, a general model does not automatically know which measurements a particular sensor can support, which algorithms are implemented in the current software, or which conclusions are justified by a computed result. These distinctions are especially important for low-channel electroencephalography (EEG), where sparse spatial coverage and variable signal quality make plausible but unsupported interpretations easy to produce. We present NeuraDock Agent, an open-source architecture that separates a deterministic local EEG engine from a hardware-aware language layer. The numerical engine parses recordings, performs quality control, executes reviewed spectral workflows, and writes machine-readable artifacts. The LLM receives only a compact, allowlisted summary and a versioned context pack. The context describes the seven-channel hardware, reviewed workflows, result f
    
[^6]: 断言，而非描述：改变大语言模型对动物福利推理的语言特征

    Assert, don't describe: Linguistic features that shift LLM reasoning about animal welfare

    [https://arxiv.org/abs/2606.26104](https://arxiv.org/abs/2606.26104)

    研究发现，在微调语言模型时，使用断言式确定性、道德词汇和情感词汇等特征会显著增强模型对动物福利的支持，而模糊语言和具体感官描述则会削弱这一立场。

    

    arXiv:2606.26104v1 公告类型：交叉 摘要：动物福利倡导者撰写了大量文本，而这些文本越来越多地用于训练语言模型，随后数百万用户会向这些模型询问动物福利问题。通过在一个预留的动物福利基准测试上使用词汇匹配的立场对比探针，我们测量了十种语言特征分别作为微调数据使用时，如何改变Llama-3.2-1B对支持动物福利推理的偏好。其中八种特征产生了统计上显著的转变。七种特征使模型朝向更强的支持动物福利推理方向移动：断言式确定性、明确的道德词汇、情感词汇、评价性主张、叙事结构、描绘的伤害严重程度以及即时时间框架。两种特征使模型朝相反方向移动：模糊语言和具体感官描述都削弱了支持动物福利的立场。第一人称视角没有统计上显著的影响。对于任何撰写旨在影响语言模型的动物福利文本的人来说，实际建议是使用断言式确定性、明确的道德词汇和情感词汇，同时避免模糊语言和具体感官描述。

    arXiv:2606.26104v1 Announce Type: cross  Abstract: Animal-welfare advocates produce a lot of writing, and increasingly that writing trains the language models that millions of people then ask about animal welfare. Using vocabulary-matched stance-contrast probes on a held-out animal-welfare benchmark, we measure how each of ten linguistic features changes Llama-3.2-1B's preference for pro-animal-welfare reasoning when used as fine-tuning data. Eight of the ten features produce statistically significant shifts. Seven move the model toward stronger pro-animal-welfare reasoning: assertive certainty, explicit moral vocabulary, emotion words, evaluative claims, narrative structure, depicted harm severity, and immediate temporal framing. Two move it the other way: hedged language and concrete sensory description both dilute the pro-animal-welfare stance. First-person perspective has no statistically significant effect. The practical recommendation for anyone writing animal-welfare text that m
    
[^7]: 有益性有害：后训练中基于领域的中期训练同情价值观退化

    Helpfulness Hurts: Domain-Dependent Degradation of Mid-Trained Compassion Values Under Post-Training

    [https://arxiv.org/abs/2606.26102](https://arxiv.org/abs/2606.26102)

    论文发现，后训练中的有益性训练（如SFT和GRPO）会显著削弱语言模型在中期训练中获得的动物同情价值观，而编码训练的影响较小，且这一现象在多个数据集和训练范式上得到验证。

    

    摘要：标准后训练流程采用监督微调（SFT）和强化学习（RL）来使语言模型具有有益性，但这些过程可能会无意中削弱预训练期间灌输的价值观。我们研究了后训练数据的领域是否会对基于同情导向的合成数据进行中期训练的Llama 3.1 8B模型中的动物同情价值观保持产生差异化影响，使用了SFT（通过Dolly-15k的有益性与通过Magicoder-110K的编码）和GRPO（通过RLHFlow的有益性与通过Magicoder的编码），并在动物伤害基准（AHB 2.2）和不确定性下的道德推理基准（MORU）上进行评估。与编码训练相比，有益性训练在AHB上显著降低了动物同情（SFT：35.7%对65.2%；GRPO：18.7%对32.0%），这一结果在两个独立的有益性数据集和两种训练范式上得到复现。在英文MORU项目中，有益性训练也降低了通用道德推理。

    arXiv:2606.26102v1 Announce Type: cross  Abstract: Standard post-training pipelines apply supervised fine-tuning (SFT) and reinforcement learning (RL) to make language models helpful, but these processes may inadvertently degrade values instilled during pre-training. We investigate whether the domain of post-training data differentially affects the retention of animal compassion values in a Llama 3.1 8B model mid-trained on compassion-oriented synthetic data, using both SFT (helpfulness via Dolly-15k vs. coding via Magicoder-110K) and GRPO (helpfulness via RLHFlow vs. coding via Magicoder), evaluated on the Animal Harm Benchmark (AHB 2.2) and MORU benchmark (Moral Reasoning Under Uncertainty). Helpfulness training significantly degrades animal compassion relative to coding training on AHB (SFT: 35.7% vs. 65.2%; GRPO: 18.7% vs. 32.0%), replicating across two independent helpfulness datasets and two training paradigms. On English MORU items, helpfulness training degrades general moral re
    
[^8]: 联合奖励建模：将思维链内化以实现高效的视觉奖励模型

    Joint Reward Modeling: Internalizing Chain-of-Thought for Efficient Visual Reward Models

    [https://arxiv.org/abs/2602.07533](https://arxiv.org/abs/2602.07533)

    提出联合奖励建模（JRM）方法，通过在共享的视觉-语言骨干网络上联合优化偏好学习和语言建模，将生成模型的语义与推理能力内化到高效的判别式奖励模型中，解决了现有判别式模型推理不足和生成式模型推理成本高的问题。

    

    arXiv:2602.07533v2 公告类型：替换 摘要：奖励模型对于从人类反馈中进行强化学习至关重要，因为它们决定了生成模型的对齐质量和可靠性。对于图像编辑等复杂任务，奖励模型需要捕捉全局语义一致性和超越局部相似性的隐式逻辑约束。现有的奖励建模方法存在明显局限性。判别式奖励模型与人类偏好对齐良好，但由于推理监督有限，在处理复杂语义时表现不佳。生成式奖励模型提供了更强的语义理解和推理能力，但在推理时成本高昂，且难以直接与人类偏好对齐。为此，我们提出了联合奖励建模（JRM），该方法在共享的视觉-语言骨干网络上联合优化偏好学习和语言建模。这种方法将生成模型的语义和推理能力内化到高效的判别式奖励模型中，无需推理时生成链式思维（CoT）即可实现更优的性能。

    arXiv:2602.07533v2 Announce Type: replace  Abstract: Reward models are critical for reinforcement learning from human feedback, as they determine the alignment quality and reliability of generative models. For complex tasks such as image editing, reward models are required to capture global semantic consistency and implicit logical constraints beyond local similarity. Existing reward modeling approaches have clear limitations. Discriminative reward models align well with human preferences but struggle with complex semantics due to limited reasoning supervision. Generative reward models offer stronger semantic understanding and reasoning, but they are costly at inference time and difficult to align directly with human preferences. To this end, we propose Joint Reward Modeling (JRM), which jointly optimizes preference learning and language modeling on a shared vision-language backbone. This approach internalizes the semantic and reasoning capabilities of generative models into efficient 
    
[^9]: 双原型解耦：一种面向时间序列预测的上下文感知增强框架

    Dual-Prototype Disentanglement: A Context-Aware Enhancement Framework for Time Series Forecasting

    [https://arxiv.org/abs/2601.16632](https://arxiv.org/abs/2601.16632)

    提出一种模型无关的辅助框架DPAD，通过动态双原型库解耦常见与罕见时间模式，使预测模型获得上下文感知的自适应能力。

    

    时间序列预测在深度学习的推动下取得了显著进展。虽然主流方法通过修改架构或引入新颖的增强策略来提升预测性能，但它们往往无法动态解耦并利用时间序列中固有的复杂、交织的时间模式，从而学习到缺乏上下文感知能力的静态平均化表示。为解决这一问题，我们提出了双原型自适应解耦框架（DPAD），这是一种模型无关的辅助方法，使预测模型具备模式解耦和上下文感知自适应能力。具体来说，我们构建了一个动态双原型库（DDP），包含一个具有强时间先验的公共模式库（用于捕获主流趋势或季节模式）和一个动态记忆关键但罕见事件的罕见模式库，然后通过一个双原...

    arXiv:2601.16632v4 Announce Type: replace-cross  Abstract: Time series forecasting has witnessed significant progress with deep learning. While prevailing approaches enhance forecasting performance by modifying architectures or introducing novel enhancement strategies, they often fail to dynamically disentangle and leverage the complex, intertwined temporal patterns inherent in time series, thus resulting in the learning of static, averaged representations that lack context-aware capabilities. To address this, we propose the Dual-Prototype Adaptive Disentanglement framework (DPAD), a model-agnostic auxiliary method that equips forecasting models with the ability of pattern disentanglement and context-aware adaptation. Specifically, we construct a Dynamic Dual-Prototype bank (DDP), comprising a common pattern bank with strong temporal priors to capture prevailing trend or seasonal patterns, and a rare pattern bank dynamically memorizing critical yet infrequent events, and then an Dual-P
    
[^10]: SciFig：迈向科学论文可编辑图形自动生成

    SciFig: Towards Automating Editable Figure Generation for Scientific Papers

    [https://arxiv.org/abs/2601.04390](https://arxiv.org/abs/2601.04390)

    SciFig是一个端到端的多智能体框架，通过将图形生成分解为规划、布局合成、组件渲染和迭代优化四个步骤，从科学文本中自动生成视觉丰富且完全可编辑的XML格式方法论图形，解决了现有方法在可编辑性和视觉质量之间的权衡问题。

    

    arXiv:2601.04390v2 公告类型：替换 摘要：高质量的方法论图形是科学交流的核心，但制作它们仍然困难且耗时。随着论文的演变，这类图形必须将方法的组成部分和信息流提炼成一个清晰、可修改的图表。现有的方法论图形自动化系统通常在可编辑性和视觉质量之间面临权衡：基于TikZ或SVG的方法能生成可编辑的结构化输出，但往往缺乏人类设计图形的丰富性；而图像生成模型能产生精美的光栅输出，却难以修改。我们提出了SciFig，一个端到端的多智能体框架，能从科学文本生成视觉丰富且完全可编辑的方法论图形。SciFig将图形生成分解为规划、布局合成、组件渲染和迭代优化，生成可在标准绘图工具中编辑的XML图形，并能通过人类或VLM反馈进行完善。

    arXiv:2601.04390v2 Announce Type: replace  Abstract: High-quality methodology figures are central to scientific communication, yet they remain difficult and time-consuming to create. Such figures must distill a method's components and information flow into a clear, revisable diagram as the paper evolves. Existing methodology diagram automation systems typically face a trade-off between editability and visual quality: TikZ- or SVG-based methods produce editable structured outputs but often lack the richness of human-designed figures, while image-generation models produce polished raster outputs that are difficult to revise. We introduce SciFig, an end-to-end multi-agent framework for generating visually rich and fully editable methodology figures from scientific text. SciFig decomposes figure generation into planning, layout synthesis, component rendering, and iterative refinement, producing XML figures that can be edited in standard diagramming tools and refined through human or VLM fe
    
[^11]: 通过人工智能"生成"工作：在线劳动市场的经验证据

    "Generate" the Future of Work through AI: Empirical Evidence from Online Labor Markets. (arXiv:2308.05201v1 [cs.AI])

    [http://arxiv.org/abs/2308.05201](http://arxiv.org/abs/2308.05201)

    这项研究通过利用ChatGPT作为外生冲击，揭示了其对在线劳动市场的影响。结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降，但适应新技术并提供增强人工智能的服务的自由职业者仍能获得利益。

    

    随着通用生成式人工智能的出现，对其对劳动市场的影响的兴趣不断增加。为了填补现有的实证空白，我们将ChatGPT的推出解释为一种外生冲击，并采用差异法来量化其对在线劳动市场中与文本相关的工作和自由职业者的影响。我们的结果显示，直接接触ChatGPT的任务和自由职业者的交易量显著下降。此外，这种下降在相对较高的过去交易量或较低的质量标准下尤为显著。然而，并非所有服务提供商都普遍经历了负面影响。随后的分析表明，在这个转型期间，能够适应新进展并提供增强人工智能技术的服务的自由职业者可以获得可观的利益。因此，虽然ChatGPT的出现有可能替代人力劳动

    With the advent of general-purpose Generative AI, the interest in discerning its impact on the labor market escalates. In an attempt to bridge the extant empirical void, we interpret the launch of ChatGPT as an exogenous shock, and implement a Difference-in-Differences (DID) approach to quantify its influence on text-related jobs and freelancers within an online labor marketplace. Our results reveal a significant decrease in transaction volume for gigs and freelancers directly exposed to ChatGPT. Additionally, this decline is particularly marked in units of relatively higher past transaction volume or lower quality standards. Yet, the negative effect is not universally experienced among service providers. Subsequent analyses illustrate that freelancers proficiently adapting to novel advancements and offering services that augment AI technologies can yield substantial benefits amidst this transformative period. Consequently, even though the advent of ChatGPT could conceivably substitute
    

