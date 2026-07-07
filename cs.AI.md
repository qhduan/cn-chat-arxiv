# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heavy-Ball Q-Learning with Residual Weighting Correction](https://arxiv.org/abs/2606.27112) | 本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。 |
| [^2] | [Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting](https://arxiv.org/abs/2606.26520) | 提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。 |
| [^3] | [Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model](https://arxiv.org/abs/2606.26373) | 本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。 |
| [^4] | [Autodata: An agentic data scientist to create high quality synthetic data](https://arxiv.org/abs/2606.25996) | 本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。 |
| [^5] | [To Use AI as Dice of Possibilities with Timing Computation](https://arxiv.org/abs/2605.01134) | 本文提出了一种基于动词范式的因果推理框架，通过时间计算与因果事实定义，使AI能够从数据中自动发现临床轨迹并进行反事实推理，在乳腺癌患者数据上首次实现了纯数据驱动的因果世界模型。 |
| [^6] | [CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes](https://arxiv.org/abs/2404.01299) | 利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。 |
| [^7] | [Graph Unitary Message Passing](https://arxiv.org/abs/2403.11199) | 提出了一种名为GUMP的图单元消息传递方法，通过应用单元邻接矩阵来缓解图神经网络中的过度压缩问题。 |
| [^8] | [Learning to Visually Connect Actions and their Effects.](http://arxiv.org/abs/2401.10805) | 该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。 |

# 详细

[^1]: 带残差加权校正的重球Q学习

    Heavy-Ball Q-Learning with Residual Weighting Correction

    [https://arxiv.org/abs/2606.27112](https://arxiv.org/abs/2606.27112)

    本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。

    

    本文提出了一种用于强化学习的校正重球Q学习方法，并证明了其收敛性。同时，文章识别了该方法在理论上保证比标准Q学习收敛更快的条件。随后，相同的构造被扩展到线性函数逼近的Q学习中，并推导出了类似的收敛性和加速结论。该分析基于Q学习算法的切换线性系统表示以及相关切换族的联合谱半径。这种切换线性系统视角在标准Q学习分析中并不常用，它为理解重球动量如何加速Q学习提供了补充框架和新的见解。

    arXiv:2606.27112v1 Announce Type: cross  Abstract: This paper proposes a corrected heavy-ball Q-learning method for reinforcement learning (RL) and establishes its convergence. It also identifies conditions under which the method is theoretically guaranteed to converge faster than standard Q-learning. The same construction is then extended to Q-learning with linear function approximation, where analogous convergence and acceleration statements are derived. The analysis is based on a switched linear system (SLS) representation of Q-learning algorithms and on the joint spectral radius (JSR) of the associated switching families. This SLS viewpoint is not commonly used in standard analyses of Q-learning, and it provides a complementary framework and new insight into how heavy-ball momentum can accelerate Q-learning.
    
[^2]: 基于多路径自适应门控瓶颈潜在常微分方程与拉曼数据融合的细胞培养过程预测方法

    Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting

    [https://arxiv.org/abs/2606.26520](https://arxiv.org/abs/2606.26520)

    提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。

    

    哺乳动物细胞培养过程是许多生物制药生产的基础，但保持过程稳定运行十分困难：关键工艺参数会随时间漂移，而偏离规格的趋势往往在确认时已为时过晚，无法及时干预。早期的多日预测能够实现对补料、采样和控制的及时调整，但生物过程预测面临诸多挑战：测量数据稀疏且采样时间不规则，不同细胞系和培养基的操作条件存在异质性，且初始行为几乎相同的运行过程可能走向截然不同的未来结果。为此，我们提出了一种自适应框架，将门控瓶颈潜在常微分方程与多路径即时微调相结合。门控瓶颈潜在常微分方程通过可学习的变量级门控和掩码感知瓶颈机制对标准潜在常微分方程进行增强，能够压缩高维稀疏输入，从而在有限数据条件下提升学习效果。

    arXiv:2606.26520v1 Announce Type: cross  Abstract: Mammalian cell-culture processes underpin the manufacture of many biopharmaceuticals, yet keeping a run on track is hard: critical process parameters drift over days, and an off-specification trend is often confirmed too late to intervene. Early-stage, multi-day forecasts could enable timely adjustment of feeding, sampling, and control, but bioprocess forecasting is challenging because measurements are sparse and irregularly sampled, operating conditions are heterogeneous across cell lines and media, and runs with near-identical early behaviour can diverge into different futures. We propose an adaptive framework combining a Gated Bottleneck Latent Ordinary Differential Equation (GB-Latent ODE) with Multi-Path Just-In-Time Fine Tuning (MP-JIT-FT). The GB-Latent ODE augments the stan dard Latent ODE with learnable variable-wise gating and a mask-aware bottleneck that compress high-dimensional sparse inputs, improving learning under limit
    
[^3]: 混合隐私感知语义搜索：受限威胁模型下基于SVD截断文档几何与CKKS加密查询重排序

    Hybrid privacy-aware semantic search: SVD-truncated document geometry and CKKS-encrypted query reranking under a restricted threat model

    [https://arxiv.org/abs/2606.26373](https://arxiv.org/abs/2606.26373)

    本文提出一种混合隐私保护语义搜索方法，通过SVD截断和秘密正交变换保护文档集合，利用CKKS同态加密保护查询，在受限威胁模型下平衡了安全性与效率。

    

    arXiv:2606.26373v1 公告类型：交叉 摘要：稠密嵌入为语义搜索和检索增强生成提供了强大支持，但嵌入反转攻击可以从向量中重建源文本：当向量数据库泄露时，其背后的文档也会随之泄露。教科书式的防御措施是极端方案——对整个搜索进行同态加密是可靠的，但在百万级文档规模下速度过慢，而隐私噪声在提供保护之前就已严重降低排序质量。我们研究了一条中间路径，利用静态集合与动态查询之间的不对称性。集合通过几何方式保护：每个向量被截断到低维SVD子空间，并通过仅由所有者知道的秘密正交变换进行旋转。查询通过密码学方式保护：在CKKS同态加密下进行重排序，因此诚实但好奇的服务器永远无法看到查询或分数。CKKS参数来自一个小型离线基准测试。我们证明了重构的下界紧致性。

    arXiv:2606.26373v1 Announce Type: cross  Abstract: Dense embeddings power semantic search and retrieval-augmented generation, but embedding-inversion attacks can reconstruct source text from a vector: when a vector database leaks, the documents behind it leak too. The textbook defences are extremes - encrypting the whole search homomorphically is sound but too slow at million-document scale, while privacy noise degrades ranking long before it protects. We study a middle path exploiting the asymmetry between the static collection and the dynamic query. The collection is protected geometrically: each vector is truncated onto a lower-dimensional SVD subspace and rotated by a secret orthogonal transform known only to the owner. The query is protected cryptographically: it is reranked under CKKS homomorphic encryption, so an honest-but-curious server never sees the query or the scores. CKKS parameters come from a small offline benchmark.   We prove a tight lower bound on the reconstruction 
    
[^4]: Autodata：一个用于创建高质量合成数据的自主数据科学家

    Autodata: An agentic data scientist to create high quality synthetic data

    [https://arxiv.org/abs/2606.25996](https://arxiv.org/abs/2606.25996)

    本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。

    

    arXiv:2606.25996v2 公告类型：替换 摘要：我们介绍了Autodata，一种通用方法，使AI智能体能够充当数据科学家，构建高质量的训练和评估数据。我们展示了如何训练（元优化）这样一个数据科学家智能体，使其学会创建更强大的数据。我们描述了总体框架以及一个具体的实践实现——自主自我指令（Agentic Self-Instruct）。我们在计算机科学研究任务、法律推理任务和数学对象推理任务上进行了实验，与经典的合成数据集创建方法相比，我们获得了更好的结果。此外，对数据科学家智能体本身进行元优化带来了更大的性能提升。自主数据创建提供了一种将增加的推理计算转化为更高质量模型训练的方法。总体而言，我们相信这一方向有潜力改变我们构建AI数据的方式。

    arXiv:2606.25996v2 Announce Type: replace  Abstract: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.
    
[^5]: 将人工智能用作带有时间计算的“可能性骰子”

    To Use AI as Dice of Possibilities with Timing Computation

    [https://arxiv.org/abs/2605.01134](https://arxiv.org/abs/2605.01134)

    本文提出了一种基于动词范式的因果推理框架，通过时间计算与因果事实定义，使AI能够从数据中自动发现临床轨迹并进行反事实推理，在乳腺癌患者数据上首次实现了纯数据驱动的因果世界模型。

    

    arXiv:2605.01134v3 公告类型：替换 摘要：当前以名词为主的建模范式从根本上限制了人工智能的发展，使其无法充分将未来表征为一个开放的时间维度。本文引入了一种以动词为主的范式，并给出了“时间计算”和“因果事实”的精确定义，从而使人工智能能够作为自发构建因果推理世界模型的工具。将该框架应用于来自3276名乳腺癌患者的纵向电子健康记录数据，实证结果表明：(1) 自动发现具有临床意义的患者轨迹，以及(2) 反事实时间推理，即一种“假设机器”。这两项结果均以纯数据驱动的方式实现，无需借助先验领域知识，据我们所知，这是机器学习文献中首次展示此类成果。

    arXiv:2605.01134v3 Announce Type: replace  Abstract: The dominant noun-based modeling paradigm has fundamentally constrained AI development, precluding any adequate representation of the future as an open temporal dimension. This paper introduces a verb-based paradigm, together with precise definitions of \emph{timing computation} and \emph{causal factum}, that enables AI to function as an instrument for spontaneously constructing a causal-reasoning world model.   Applied to longitudinal EHR data from 3,276 breast cancer patients, the framework empirically demonstrates: (1) automatic discovery of clinically significant patient trajectories, and (2) counterfactual timing deduction, that is, a \emph{What-If Machine}. Both results are achieved in a purely data-driven manner, without recourse to prior domain knowledge, and represent, to our knowledge, the first such demonstrations in the machine learning literature.
    
[^6]: CausalChaos!数据集：基于动态视觉场景中更长因果链的全面因果行动问答

    CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes

    [https://arxiv.org/abs/2404.01299](https://arxiv.org/abs/2404.01299)

    利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。

    

    因果视频问答（QA）越来越受到关注，然而现有数据集在因果推理分析方面往往缺乏深度。为了填补这一空白，我们利用卡通的独特属性构建了CausalChaos!，这是一个新颖且具有挑战性的因果问答（Why-QA）数据集，基于标志性的“猫和老鼠”卡通系列。我们的数据集通过周到的问题和多层次答案，包含着嵌入动态互动和视觉中的更长因果链，同时动画原理允许动画师创造定义明确、明了的因果关系。这些因素使模型能够解决更具挑战性但明确定义的因果关系。我们还引入了硬负采样，包括CausalConfusion版本。虽然模型表现良好，但仍有很大改进空间，特别是在开放式答案方面。我们确定了更为先进/明确的因果关系建模和联合建模等改进方向。

    arXiv:2404.01299v1 Announce Type: cross  Abstract: Causal video question answering (QA) has garnered increasing interest, yet existing datasets often lack depth in causal reasoning analysis. To address this gap, we capitalize on the unique properties of cartoons and construct CausalChaos!, a novel, challenging causal Why-QA dataset built upon the iconic "Tom and Jerry" cartoon series. With thoughtful questions and multi-level answers, our dataset contains much longer causal chains embedded in dynamic interactions and visuals, at the same time principles of animation allows animators to create well-defined, unambiguous causal relationships. These factors allow models to solve more challenging, yet well-defined causal relationships. We also introduce hard negative mining, including CausalConfusion version. While models perform well, there is much room for improvement, especially, on open-ended answers. We identify more advanced/explicit causal relationship modeling and joint modeling of 
    
[^7]: 图单元消息传递

    Graph Unitary Message Passing

    [https://arxiv.org/abs/2403.11199](https://arxiv.org/abs/2403.11199)

    提出了一种名为GUMP的图单元消息传递方法，通过应用单元邻接矩阵来缓解图神经网络中的过度压缩问题。

    

    消息传递机制是图神经网络在各种应用中取得成功的原因，但也带来了过度压缩的问题。最近的研究通过改善图谱的重连技术、破坏图中的结构偏见来抵制过度压缩，然而在过度压缩度量方面对过度压缩的改进有所限制。受到单元RNN的启发，我们提出了图单元消息传递（GUMP），通过应用单元邻接矩阵进行消息传递来缓解图神经网络中的过度压缩问题。为设计GUMP，首先提出了一种转换方法，使普通图具有单元邻接矩阵并保持其结构偏差。然后，通过利用单元邻接矩阵的固有结构实现单位化投影算法获得单元邻接矩阵，并允许GUMP是置换等变的。实验结果表明了GUMP在改善各种应用任务上性能的有效性。

    arXiv:2403.11199v1 Announce Type: cross  Abstract: Message passing mechanism contributes to the success of GNNs in various applications, but also brings the oversquashing problem. Recent works combat oversquashing by improving the graph spectrums with rewiring techniques, disrupting the structural bias in graphs, and having limited improvement on oversquashing in terms of oversquashing measure. Motivated by unitary RNN, we propose Graph Unitary Message Passing (GUMP) to alleviate oversquashing in GNNs by applying unitary adjacency matrix for message passing. To design GUMP, a transformation is first proposed to make general graphs have unitary adjacency matrix and keep its structural bias. Then, unitary adjacency matrix is obtained with a unitary projection algorithm, which is implemented by utilizing the intrinsic structure of unitary adjacency matrix and allows GUMP to be permutation-equivariant. Experimental results show the effectiveness of GUMP in improving the performance on vari
    
[^8]: 学习视觉连接动作和其效果

    Learning to Visually Connect Actions and their Effects. (arXiv:2401.10805v1 [cs.CV])

    [http://arxiv.org/abs/2401.10805](http://arxiv.org/abs/2401.10805)

    该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。

    

    在这项工作中，我们引入了视觉连接动作和其效果（CATE）的新概念，用于视频理解。CATE可以在任务规划和从示范中学习等领域中应用。我们提出了不同基于CATE的任务形式，如动作选择和动作指定，其中视频理解模型以语义和细粒度的方式连接动作和效果。我们观察到不同的形式产生了捕捉直观动作特性的表示。我们还设计了各种基线模型用于动作选择和动作指定。尽管任务具有直观性，但我们观察到模型困难重重，人类表现明显优于它们。本研究旨在为未来的努力奠定基础，展示了连接视频理解中动作和效果的灵活性和多功能性，希望能激发出高级形式和模型的灵感。

    In this work, we introduce the novel concept of visually Connecting Actions and Their Effects (CATE) in video understanding. CATE can have applications in areas like task planning and learning from demonstration. We propose different CATE-based task formulations, such as action selection and action specification, where video understanding models connect actions and effects at semantic and fine-grained levels. We observe that different formulations produce representations capturing intuitive action properties. We also design various baseline models for action selection and action specification. Despite the intuitive nature of the task, we observe that models struggle, and humans outperform them by a large margin. The study aims to establish a foundation for future efforts, showcasing the flexibility and versatility of connecting actions and effects in video understanding, with the hope of inspiring advanced formulations and models.
    

