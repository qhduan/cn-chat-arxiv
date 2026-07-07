# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Autodata: An agentic data scientist to create high quality synthetic data](https://arxiv.org/abs/2606.25996) | 本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。 |
| [^2] | [MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models](https://arxiv.org/abs/2606.24155) | MedBench v5通过引入过程可见性、原子技能评估和集成幻觉检测，将临床多模态模型的评估从静态问答转变为动态、过程导向的基准测试，并揭示了模型在信息流压力下的退化模式和隐性幻觉传播。 |
| [^3] | [CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes](https://arxiv.org/abs/2404.01299) | 利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。 |

# 详细

[^1]: Autodata：一个用于创建高质量合成数据的自主数据科学家

    Autodata: An agentic data scientist to create high quality synthetic data

    [https://arxiv.org/abs/2606.25996](https://arxiv.org/abs/2606.25996)

    本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。

    

    arXiv:2606.25996v2 公告类型：替换 摘要：我们介绍了Autodata，一种通用方法，使AI智能体能够充当数据科学家，构建高质量的训练和评估数据。我们展示了如何训练（元优化）这样一个数据科学家智能体，使其学会创建更强大的数据。我们描述了总体框架以及一个具体的实践实现——自主自我指令（Agentic Self-Instruct）。我们在计算机科学研究任务、法律推理任务和数学对象推理任务上进行了实验，与经典的合成数据集创建方法相比，我们获得了更好的结果。此外，对数据科学家智能体本身进行元优化带来了更大的性能提升。自主数据创建提供了一种将增加的推理计算转化为更高质量模型训练的方法。总体而言，我们相信这一方向有潜力改变我们构建AI数据的方式。

    arXiv:2606.25996v2 Announce Type: replace  Abstract: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.
    
[^2]: MedBench v5：面向临床多模态模型的动态、过程导向且具有幻觉感知能力的基准测试

    MedBench v5: A Dynamic, Process-Oriented, and Hallucination-Aware Benchmark for Clinical Multimodal Models

    [https://arxiv.org/abs/2606.24155](https://arxiv.org/abs/2606.24155)

    MedBench v5通过引入过程可见性、原子技能评估和集成幻觉检测，将临床多模态模型的评估从静态问答转变为动态、过程导向的基准测试，并揭示了模型在信息流压力下的退化模式和隐性幻觉传播。

    

    arXiv:2606.24155v3 公告类型：替换 摘要：现有的医学人工智能基准测试缺乏过程可见性、原子技能评估以及集成的幻觉检测能力。我们推出了MedBench v5，这是一个为临床多模态模型（包括语言模型、视觉语言模型和智能体系统）重新设计的基准测试，它将静态问答转变为动态的、过程导向的评估。MedBench v5的特点包括：（1）一个结合了临床认知响应性（14个子维度）和医学原子技能（4个智能体环境）的双维度框架，覆盖63个任务；（2）三种可切换的信息流压力因素（信息缺失、信息矛盾、证据延迟），用于分解式性能退化分析；（3）一个包含五个推理节点的动态过程审计协议，可生成特定模型的失败特征指纹；（4）跨启动、传播、锚定和矛盾交互的幻觉传播监控，以捕捉隐性幻觉。对前沿模型的实验表明，尽管整体性能强大，但...

    arXiv:2606.24155v3 Announce Type: replace  Abstract: Existing medical AI benchmarks lack process visibility, atomic skill evaluation, and integrated hallucination detection. We introduce MedBench v5, a redesigned benchmark for clinical multimodal models (language, vision-language, and agent systems) that moves from static QA to dynamic, process-oriented evaluation. MedBench v5 features: (1) a dual-dimensional framework combining Clinical Cognitive Responsiveness (14 sub-dimensions) and Medical Atomic Skills (4 agent environments), covering 63 tasks; (2) three switchable information-flow stressors (omission, contradiction, evidence delay) for factorized degradation analysis; (3) a dynamic process audit protocol with five reasoning nodes that produces model-specific failure fingerprints; (4) hallucination propagation monitoring across initiation, propagation, anchoring, and contradiction interaction-capturing silent hallucination. Experiments on frontier models show that strong overall t
    
[^3]: CausalChaos!数据集：基于动态视觉场景中更长因果链的全面因果行动问答

    CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes

    [https://arxiv.org/abs/2404.01299](https://arxiv.org/abs/2404.01299)

    利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。

    

    因果视频问答（QA）越来越受到关注，然而现有数据集在因果推理分析方面往往缺乏深度。为了填补这一空白，我们利用卡通的独特属性构建了CausalChaos!，这是一个新颖且具有挑战性的因果问答（Why-QA）数据集，基于标志性的“猫和老鼠”卡通系列。我们的数据集通过周到的问题和多层次答案，包含着嵌入动态互动和视觉中的更长因果链，同时动画原理允许动画师创造定义明确、明了的因果关系。这些因素使模型能够解决更具挑战性但明确定义的因果关系。我们还引入了硬负采样，包括CausalConfusion版本。虽然模型表现良好，但仍有很大改进空间，特别是在开放式答案方面。我们确定了更为先进/明确的因果关系建模和联合建模等改进方向。

    arXiv:2404.01299v1 Announce Type: cross  Abstract: Causal video question answering (QA) has garnered increasing interest, yet existing datasets often lack depth in causal reasoning analysis. To address this gap, we capitalize on the unique properties of cartoons and construct CausalChaos!, a novel, challenging causal Why-QA dataset built upon the iconic "Tom and Jerry" cartoon series. With thoughtful questions and multi-level answers, our dataset contains much longer causal chains embedded in dynamic interactions and visuals, at the same time principles of animation allows animators to create well-defined, unambiguous causal relationships. These factors allow models to solve more challenging, yet well-defined causal relationships. We also introduce hard negative mining, including CausalConfusion version. While models perform well, there is much room for improvement, especially, on open-ended answers. We identify more advanced/explicit causal relationship modeling and joint modeling of 
    

