# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AlphaMapleSAT: An MCTS-based Cube-and-Conquer SAT Solver for Hard Combinatorial Problems.](http://arxiv.org/abs/2401.13770) | AlphaMapleSAT是一种基于MCTS的Cube-and-Conquer SAT求解器，通过推理驱动的先行计算技术来高效解决困难的组合问题。 |
| [^2] | [DiffusionGPT: LLM-Driven Text-to-Image Generation System.](http://arxiv.org/abs/2401.10061) | DiffusionGPT是一个基于LLM的统一文本生成图像系统，能够处理多样化的输入并整合领域专家模型。 |
| [^3] | [Manipulating Feature Visualizations with Gradient Slingshots.](http://arxiv.org/abs/2401.06122) | 本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。 |
| [^4] | [Quadratic Time-Frequency Analysis of Vibration Signals for Diagnosing Bearing Faults.](http://arxiv.org/abs/2401.01172) | 本文提出了一种融合时间频率分析和深度学习技术的方法，用于在实际条件下诊断带有时间变化速度和不同噪声水平的轴承故障。这种方法有效地解析与不同轴承故障相关的独特动态模式。 |
| [^5] | [GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs.](http://arxiv.org/abs/2312.02317) | GNN2R是一种基于图神经网络的两步推理模型，通过弱监督训练，能够在知识图谱问答中提供最终答案以及推理子图的理由。该方法解决了现有方法缺乏解释以及效率低下的问题。 |
| [^6] | [Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand.](http://arxiv.org/abs/2310.20350) | 本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。 |
| [^7] | [Shape Completion with Prediction of Uncertain Regions.](http://arxiv.org/abs/2308.00377) | 该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。 |

# 详细

[^1]: AlphaMapleSAT：一种基于MCTS的Cube-and-Conquer SAT求解器，用于解决困难的组合问题

    AlphaMapleSAT: An MCTS-based Cube-and-Conquer SAT Solver for Hard Combinatorial Problems. (arXiv:2401.13770v1 [cs.AI])

    [http://arxiv.org/abs/2401.13770](http://arxiv.org/abs/2401.13770)

    AlphaMapleSAT是一种基于MCTS的Cube-and-Conquer SAT求解器，通过推理驱动的先行计算技术来高效解决困难的组合问题。

    

    本文介绍了AlphaMapleSAT，一种新颖的基于Monte Carlo Tree Search (MCTS)的Cube-and-Conquer (CnC) SAT求解方法，旨在高效地解决具有挑战性的组合问题。尽管CnC求解器在解决各种困难的组合问题上取得了巨大成功，但多年来，CnC的先行计算技术并没有得到很大发展。其中一个原因是很难提出既低成本又能有效地将输入公式分割为子公式的新型分割技术，从而使整体运行时间最小化。当前最先进的CnC求解器（如March）使用的先行计算技术通过约束搜索最优分割变量来降低计算成本。相比之下，我们的关键创新是一种基于推理驱动的MCTS先行计算技术，通过进行更深入的启发式搜索来寻找有效的分割，同时使计算成本低。我们进行了详细的对比实验

    This paper introduces AlphaMapleSAT, a novel Monte Carlo Tree Search (MCTS) based Cube-and-Conquer (CnC) SAT solving method aimed at efficiently solving challenging combinatorial problems. Despite the tremendous success of CnC solvers in solving a variety of hard combinatorial problems, the lookahead cubing techniques at the heart of CnC have not evolved much for many years. Part of the reason is the sheer difficulty of coming up with new cubing techniques that are both low-cost and effective in partitioning input formulas into sub-formulas, such that the overall runtime is minimized.  Lookahead cubing techniques used by current state-of-the-art CnC solvers, such as March, keep their cubing costs low by constraining the search for the optimal splitting variables. By contrast, our key innovation is a deductively-driven MCTS-based lookahead cubing technique, that performs a deeper heuristic search to find effective cubes, while keeping the cubing cost low. We perform an extensive compari
    
[^2]: DiffusionGPT: 基于LLM的文本生成图像系统

    DiffusionGPT: LLM-Driven Text-to-Image Generation System. (arXiv:2401.10061v1 [cs.CV])

    [http://arxiv.org/abs/2401.10061](http://arxiv.org/abs/2401.10061)

    DiffusionGPT是一个基于LLM的统一文本生成图像系统，能够处理多样化的输入并整合领域专家模型。

    

    扩散模型为图像生成领域打开了新的道路，导致了在开源平台上共享高质量模型的广泛传播。然而，目前的文本生成图像系统存在一个主要挑战，即往往无法处理多样化的输入，或仅限于单一模型的结果。目前的统一尝试通常分为两个正交方面：i）在输入阶段解析多样的提示；ii）激活专家模型进行输出。为了兼顾两者的优点，我们提出了DiffusionGPT，它利用大型语言模型（LLM）提供了一个统一的生成系统，能够无缝地适应各种类型的提示并整合领域专家模型。DiffusionGPT基于先验知识为各种生成模型构建了领域特定的Thought树。当提供输入时，LLM解析提示并利用Thought树来指导选择适当的模型，从而放松输入约束并确保异常的效果。

    Diffusion models have opened up new avenues for the field of image generation, resulting in the proliferation of high-quality models shared on open-source platforms. However, a major challenge persists in current text-to-image systems are often unable to handle diverse inputs, or are limited to single model results. Current unified attempts often fall into two orthogonal aspects: i) parse Diverse Prompts in input stage; ii) activate expert model to output. To combine the best of both worlds, we propose DiffusionGPT, which leverages Large Language Models (LLM) to offer a unified generation system capable of seamlessly accommodating various types of prompts and integrating domain-expert models. DiffusionGPT constructs domain-specific Trees for various generative models based on prior knowledge. When provided with an input, the LLM parses the prompt and employs the Trees-of-Thought to guide the selection of an appropriate model, thereby relaxing input constraints and ensuring exceptional 
    
[^3]: 用梯度弹射操纵特征可视化

    Manipulating Feature Visualizations with Gradient Slingshots. (arXiv:2401.06122v1 [cs.LG])

    [http://arxiv.org/abs/2401.06122](http://arxiv.org/abs/2401.06122)

    本研究探究了激活最大化方法在对抗模型操作中的脆弱性，并提出了一种新的方法来操纵特征可视化，以隐藏特定神经元的功能。

    

    深度神经网络(DNNs)能够学习复杂而多样化的表示，然而，学习到的概念的语义性质仍然未知。解释DNNs学习到的概念的常用方法是激活最大化(AM)，它生成一个合成的输入信号，最大化激活网络中的特定神经元。在本文中，我们研究了这种方法对于对抗模型操作的脆弱性，并引入了一种新的方法来操纵特征可视化，而不改变模型结构或对模型的决策过程产生显著影响。我们评估了我们的方法对几个神经网络模型的效果，并展示了它隐藏特定神经元功能的能力，在模型审核过程中使用选择的目标解释屏蔽了原始解释。作为一种补救措施，我们提出了一种防止这种操纵的防护措施，并提供了定量证据，证明了它的有效性。

    Deep Neural Networks (DNNs) are capable of learning complex and versatile representations, however, the semantic nature of the learned concepts remains unknown. A common method used to explain the concepts learned by DNNs is Activation Maximization (AM), which generates a synthetic input signal that maximally activates a particular neuron in the network. In this paper, we investigate the vulnerability of this approach to adversarial model manipulations and introduce a novel method for manipulating feature visualization without altering the model architecture or significantly impacting the model's decision-making process. We evaluate the effectiveness of our method on several neural network models and demonstrate its capabilities to hide the functionality of specific neurons by masking the original explanations of neurons with chosen target explanations during model auditing. As a remedy, we propose a protective measure against such manipulations and provide quantitative evidence which 
    
[^4]: 振动信号的二次时间频率分析用于诊断轴承故障

    Quadratic Time-Frequency Analysis of Vibration Signals for Diagnosing Bearing Faults. (arXiv:2401.01172v1 [cs.LG])

    [http://arxiv.org/abs/2401.01172](http://arxiv.org/abs/2401.01172)

    本文提出了一种融合时间频率分析和深度学习技术的方法，用于在实际条件下诊断带有时间变化速度和不同噪声水平的轴承故障。这种方法有效地解析与不同轴承故障相关的独特动态模式。

    

    轴承故障的诊断对于降低维修成本和设备停机至关重要。轴承故障是机器振动的主要原因，分析其信号形态可以揭示其健康状况。然而，现有的方法主要针对控制环境进行优化，忽略了实际条件下的时间变化的转速和振动的非平稳性。本文提出了一种时间频率分析和深度学习技术的融合方法，用于在时间变化速度和不同噪声水平下诊断轴承故障。首先，我们制定了轴承故障引起的振动，并讨论了它们的非平稳性与轴承固有和操作参数之间的联系。我们还阐述了二次时间频率分布，并验证了它们解析与不同轴承故障相关的独特动态模式的有效性。基于此，我们设计了一个时间频率卷积神经网络。

    Diagnosis of bearing faults is paramount to reducing maintenance costs and operational breakdowns. Bearing faults are primary contributors to machine vibrations, and analyzing their signal morphology offers insights into their health status. Unfortunately, existing approaches are optimized for controlled environments, neglecting realistic conditions such as time-varying rotational speeds and the vibration's non-stationary nature. This paper presents a fusion of time-frequency analysis and deep learning techniques to diagnose bearing faults under time-varying speeds and varying noise levels. First, we formulate the bearing fault-induced vibrations and discuss the link between their non-stationarity and the bearing's inherent and operational parameters. We also elucidate quadratic time-frequency distributions and validate their effectiveness in resolving distinctive dynamic patterns associated with different bearing faults. Based on this, we design a time-frequency convolutional neural n
    
[^5]: GNN2R: 基于弱监督的知识图谱问答中提供理由的问题回答方法

    GNN2R: Weakly-Supervised Rationale-Providing Question Answering over Knowledge Graphs. (arXiv:2312.02317v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.02317](http://arxiv.org/abs/2312.02317)

    GNN2R是一种基于图神经网络的两步推理模型，通过弱监督训练，能够在知识图谱问答中提供最终答案以及推理子图的理由。该方法解决了现有方法缺乏解释以及效率低下的问题。

    

    目前大多数基于知识图谱的多跳问题回答方法只提供最终的确定答案，而没有解释，对于普通用户难以理解和查看的KG实体集。这严重限制了知识图谱问答在现实场景中的应用。本文提出了一种基于图神经网络的两步推理模型（GNN2R）来解决这个问题。GNN2R能够通过仅有的问题-最终答案对提供最终答案以及作为最终答案背后的推理子图的理由，且仅需要通过弱监督进行训练。我们对GNN2R进行了大量评估，并进行了详细的实验。

    Most current methods for multi-hop question answering (QA) over knowledge graphs (KGs) only provide final conclusive answers without explanations, such as a set of KG entities that is difficult for normal users to review and comprehend. This issue severely limits the application of KG-based QA in real-world scenarios. However, it is non-trivial to solve due to two challenges: First, annotations of reasoning chains of multi-hop questions, which could serve as supervision for explanation generation, are usually lacking. Second, it is difficult to maintain high efficiency when explicit KG triples need to be retrieved to generate explanations. In this paper, we propose a novel Graph Neural Network-based Two-Step Reasoning model (GNN2R) to solve this issue. GNN2R can provide both final answers and reasoning subgraphs as a rationale behind final answers efficiently with only weak supervision that is available through question-final answer pairs. We extensively evaluated GNN2R with detailed a
    
[^6]: 将形状完成和抓取预测结合，实现快速灵活的多指抓取

    Combining Shape Completion and Grasp Prediction for Fast and Versatile Grasping with a Multi-Fingered Hand. (arXiv:2310.20350v1 [cs.RO])

    [http://arxiv.org/abs/2310.20350](http://arxiv.org/abs/2310.20350)

    本文提出了一种结合形状完成和抓取预测的方法，实现了快速灵活的多指抓取。通过使用基于深度图像的形状完成模块和基于预测的抓取预测器，实现了在具有有限或无先验知识的情况下，对物体进行抓取的任务。

    

    在辅助机器人中，对于具有有限或无先验知识的物体进行抓取是一项非常重要的技能。然而，在这种普适情况下，尤其是在观测能力有限和利用多指手进行灵活抓取时，仍然存在一个开放的问题。我们提出了一种新颖、快速和高保真度的深度学习流程，由基于单个深度图像的形状完成模块和基于预测的物体形状的抓取预测器组成。形状完成网络基于VQDIF，在任意查询点上预测空间占用值。作为抓取预测器，我们使用了两阶段架构，首先使用自回归模型生成手姿势，然后回归每个姿势的手指关节配置。关键因素是足够的数据真实性和增强，以及在训练过程中对困难情况的特殊关注。在物理机器人平台上进行的实验表明，成功地实现了抓取。

    Grasping objects with limited or no prior knowledge about them is a highly relevant skill in assistive robotics. Still, in this general setting, it has remained an open problem, especially when it comes to only partial observability and versatile grasping with multi-fingered hands. We present a novel, fast, and high fidelity deep learning pipeline consisting of a shape completion module that is based on a single depth image, and followed by a grasp predictor that is based on the predicted object shape. The shape completion network is based on VQDIF and predicts spatial occupancy values at arbitrary query points. As grasp predictor, we use our two-stage architecture that first generates hand poses using an autoregressive model and then regresses finger joint configurations per pose. Critical factors turn out to be sufficient data realism and augmentation, as well as special attention to difficult cases during training. Experiments on a physical robot platform demonstrate successful gras
    
[^7]: 带有不确定区域预测的形状完成

    Shape Completion with Prediction of Uncertain Regions. (arXiv:2308.00377v1 [cs.CV])

    [http://arxiv.org/abs/2308.00377](http://arxiv.org/abs/2308.00377)

    该论文提出了两种方法来处理在给定模糊物体视图时可能存在的物体部分的不确定区域预测问题。研究表明这些方法可以作为任何预测空间占用的方法的直接扩展，通过后处理占用评分或直接预测不确定性指标来实现。这些方法与已知的概率形状完成方法进行了比较，并使用自动生成的深度图像数据集进行了验证。

    

    形状完成，即从部分观测预测物体的完整几何形状，对于几个下游任务非常重要，尤其是机器人操作。当基于物体形状重建进行规划或实际抓取的预测时，指示严重几何不确定性是必不可少的。特别是在给定模糊的物体视图时，在整个物体部分存在 irreducible uncertainty 的扩展区域。为了处理这种重要情况，我们提出了两种新方法来预测这些不确定区域，这两种方法都可以作为预测局部空间占用的任何方法的直接扩展，一种是通过后处理占用评分，另一种是通过直接预测不确定性指标。我们将这些方法与两种已知的概率形状完成方法进行了比较。此外，我们还生成了一个基于ShapeNet的数据集，其中包含了真实渲染的物体视图深度图像及其带有地面真值标注。

    Shape completion, i.e., predicting the complete geometry of an object from a partial observation, is highly relevant for several downstream tasks, most notably robotic manipulation. When basing planning or prediction of real grasps on object shape reconstruction, an indication of severe geometric uncertainty is indispensable. In particular, there can be an irreducible uncertainty in extended regions about the presence of entire object parts when given ambiguous object views. To treat this important case, we propose two novel methods for predicting such uncertain regions as straightforward extensions of any method for predicting local spatial occupancy, one through postprocessing occupancy scores, the other through direct prediction of an uncertainty indicator. We compare these methods together with two known approaches to probabilistic shape completion. Moreover, we generate a dataset, derived from ShapeNet, of realistically rendered depth images of object views with ground-truth annot
    

