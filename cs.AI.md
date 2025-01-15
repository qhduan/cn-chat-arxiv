# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment](https://arxiv.org/abs/2403.11075) | GOMA提出了一种面向目标的心智对齐的合作沟通框架，通过最小化智能体心智状态部分之间的不一致性来帮助实现更好的合作。 |
| [^2] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^3] | [A Mixed-Integer Conic Program for the Moving-Target Traveling Salesman Problem based on a Graph of Convex Sets](https://arxiv.org/abs/2403.04917) | 本文提出了一个新的公式，用于解决移动目标旅行推销员问题，该公式基于目标在空间-时间坐标系内成为凸集的概念，通过在凸集图中寻找最短路径来实现，在实验中表现出比当前Mixed Integer Conic Program (MICP)求解器更好的效果。 |
| [^4] | [Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad](https://arxiv.org/abs/2403.02648) | KATE是一种新的优化算法，提出了一种与AdaGrad标度不变的适应方法，并在广义线性模型和一般的非凸问题中证明了其标度不变性。数值实验结果表明，KATE在各种场景中均优于AdaGrad并与Adam性能匹配/超越。 |
| [^5] | [Invariant Test-Time Adaptation for Vision-Language Model Generalization](https://arxiv.org/abs/2403.00376) | 本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。 |
| [^6] | [ACPO: AI-Enabled Compiler-Driven Program Optimization](https://arxiv.org/abs/2312.09982) | 该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。 |
| [^7] | [Divergences between Language Models and Human Brains](https://arxiv.org/abs/2311.09308) | 该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。 |
| [^8] | [FLM-101B: An Open LLM and How to Train It with $100K Budget.](http://arxiv.org/abs/2309.03852) | 本文介绍了一种开放的LLM模型（FLM-101B）以及如何用10万美元的预算来训练它。通过采用增长策略，可以显著降低LLM训练的成本。同时，引入了一种系统的评估方法，以评估LLM的智能能力。 |
| [^9] | [Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words.](http://arxiv.org/abs/2307.09059) | 本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。 |
| [^10] | [Set-based Neural Network Encoding.](http://arxiv.org/abs/2305.16625) | 提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。 |
| [^11] | [PastNet: Introducing Physical Inductive Biases for Spatio-temporal Video Prediction.](http://arxiv.org/abs/2305.11421) | 本文介绍了一种名为PastNet的新颖方法，通过在傅里叶域中引入谱卷积算子，利用内在的物理知识生成高质量的时空视频预测，并通过离散化局部特征降低计算成本。 |

# 详细

[^1]: GOMA：通过面向目标的心智对齐实现主动合作沟通

    GOMA: Proactive Embodied Cooperative Communication via Goal-Oriented Mental Alignment

    [https://arxiv.org/abs/2403.11075](https://arxiv.org/abs/2403.11075)

    GOMA提出了一种面向目标的心智对齐的合作沟通框架，通过最小化智能体心智状态部分之间的不一致性来帮助实现更好的合作。

    

    口头交流在人类合作中起着至关重要的作用，特别是当合作伙伴只对任务、环境和彼此的心理状态具有不完整的信息时。本文提出了一种新颖的合作沟通框架，即面向目标的心智对齐（GOMA）。GOMA将口头交流形式化为一个规划问题，通过最小化与目标相关的智能体心智状态部分之间的不一致性来促进合作。这种方法使得一个具有身体的助手能够推理何时以及如何以自然语言主动开始与人类的口头沟通，从而帮助实现更好的合作。我们在两个具有挑战性的环境，Overcooked（一款多人游戏）和VirtualHome（一个家庭模拟器）中，对我们的方法进行了评估。实验结果表明，大型语言模型在生成基于语境的有意义沟通方面存在困难。

    arXiv:2403.11075v1 Announce Type: cross  Abstract: Verbal communication plays a crucial role in human cooperation, particularly when the partners only have incomplete information about the task, environment, and each other's mental state. In this paper, we propose a novel cooperative communication framework, Goal-Oriented Mental Alignment (GOMA). GOMA formulates verbal communication as a planning problem that minimizes the misalignment between the parts of agents' mental states that are relevant to the goals. This approach enables an embodied assistant to reason about when and how to proactively initialize communication with humans verbally using natural language to help achieve better cooperation. We evaluate our approach against strong baselines in two challenging environments, Overcooked (a multiplayer game) and VirtualHome (a household simulator). Our experimental results demonstrate that large language models struggle with generating meaningful communication that is grounded in th
    
[^2]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^3]: 基于凸集图的移动目标旅行推销员问题的混合整数锥规划

    A Mixed-Integer Conic Program for the Moving-Target Traveling Salesman Problem based on a Graph of Convex Sets

    [https://arxiv.org/abs/2403.04917](https://arxiv.org/abs/2403.04917)

    本文提出了一个新的公式，用于解决移动目标旅行推销员问题，该公式基于目标在空间-时间坐标系内成为凸集的概念，通过在凸集图中寻找最短路径来实现，在实验中表现出比当前Mixed Integer Conic Program (MICP)求解器更好的效果。

    

    本文介绍了一种寻找移动目标旅行推销员问题（MT-TSP）的最佳解决方案的新的公式，该问题旨在找到一个最短路径，使一个从仓库出发的代理访问一组移动目标，并在它们分配的时间窗口内恰好访问一次，然后返回到仓库。该公式依赖于一个关键思想，即当目标沿着线移动时，它们的轨迹在空间-时间坐标系内变为凸集。然后，问题就缩减为在一个凸集图中寻找最短路径，受到一些速度约束的限制。我们将我们的公式与当前最先进的Mixed Integer Conic Program (MICP)求解器进行了比较，结果显示，我们的公式在目标数量最多为20个的情况下性能优于MICP，在运行时间上缩短了两个数量级，并且最优性差距缩小了高达60％。我们还展示了该解法的成本...

    arXiv:2403.04917v1 Announce Type: cross  Abstract: This paper introduces a new formulation that finds the optimum for the Moving-Target Traveling Salesman Problem (MT-TSP), which seeks to find a shortest path for an agent, that starts at a depot, visits a set of moving targets exactly once within their assigned time-windows, and returns to the depot. The formulation relies on the key idea that when the targets move along lines, their trajectories become convex sets within the space-time coordinate system. The problem then reduces to finding the shortest path within a graph of convex sets, subject to some speed constraints. We compare our formulation with the current state-of-the-art Mixed Integer Conic Program (MICP) solver for the MT-TSP. The experimental results show that our formulation outperforms the MICP for instances with up to 20 targets, with up to two orders of magnitude reduction in runtime, and up to a 60\% tighter optimality gap. We also show that the solution cost from th
    
[^4]: 移除平方根：一种新的高效标度不变版本的AdaGrad

    Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad

    [https://arxiv.org/abs/2403.02648](https://arxiv.org/abs/2403.02648)

    KATE是一种新的优化算法，提出了一种与AdaGrad标度不变的适应方法，并在广义线性模型和一般的非凸问题中证明了其标度不变性。数值实验结果表明，KATE在各种场景中均优于AdaGrad并与Adam性能匹配/超越。

    

    自适应方法在机器学习中非常流行，因为它们可以降低学习速率调整的成本。本文引入了一种名为KATE的新型优化算法，它提出了一个著名的AdaGrad算法的标度不变适应。我们证明了KATE在广义线性模型案例中的标度不变性。此外，对于一般的光滑非凸问题，我们为KATE建立了一个收敛速率为$O \left(\frac{\log T}{\sqrt{T}} \right)$，与AdaGrad和Adam的最佳收敛速率相匹配。我们还通过不同问题的数值实验将KATE与其他最先进的自适应算法Adam和AdaGrad进行了比较，包括在真实数据上进行图像分类和文本分类等复杂机器学习任务。结果表明，在所有考虑到的场景中，KATE始终胜过AdaGrad，并且在性能上匹配/超越Adam。

    arXiv:2403.02648v1 Announce Type: cross  Abstract: Adaptive methods are extremely popular in machine learning as they make learning rate tuning less expensive. This paper introduces a novel optimization algorithm named KATE, which presents a scale-invariant adaptation of the well-known AdaGrad algorithm. We prove the scale-invariance of KATE for the case of Generalized Linear Models. Moreover, for general smooth non-convex problems, we establish a convergence rate of $O \left(\frac{\log T}{\sqrt{T}} \right)$ for KATE, matching the best-known ones for AdaGrad and Adam. We also compare KATE to other state-of-the-art adaptive algorithms Adam and AdaGrad in numerical experiments with different problems, including complex machine learning tasks like image classification and text classification on real data. The results indicate that KATE consistently outperforms AdaGrad and matches/surpasses the performance of Adam in all considered scenarios.
    
[^5]: 视觉-语言模型泛化的不变测试时适应性

    Invariant Test-Time Adaptation for Vision-Language Model Generalization

    [https://arxiv.org/abs/2403.00376](https://arxiv.org/abs/2403.00376)

    本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。

    

    arXiv:2403.00376v1 公告类型: 交叉摘要: 视觉-语言基础模型在大量图像-文本配对数据集上的可扩展性使其在众多下游任务中展现出卓越成功。然而，这些模型在应用于长尾任务（如细粒度图像分类）时显示出明显局限，这是由于“决策捷径”导致了它们的泛化能力受限。本文发现CLIP模型具有丰富的特征集，涵盖了既有的\textit{期望不变因果特征}又有的\textit{不希望的决策捷径}。此外，CLIP在下游任务中的表现不佳源自其无法有效利用预训练特征以符合特定任务要求。为解决这一挑战，本文引入一种测试时提示调优范式，优化一个可学习的提示，从而促使模型利用真正的因果不变特征。

    arXiv:2403.00376v1 Announce Type: cross  Abstract: Vision-language foundation models have exhibited remarkable success across a multitude of downstream tasks due to their scalability on extensive image-text paired datasets. However, these models display significant limitations when applied to long-tail tasks, such as fine-grained image classification, as a result of "decision shortcuts" that hinders their generalization capabilities. In this work, we find that the CLIP model possesses a rich set of features, encompassing both \textit{desired invariant causal features} and \textit{undesired decision shortcuts}. Moreover, the underperformance of CLIP on downstream tasks originates from its inability to effectively utilize pre-trained features in accordance with specific task requirements. To address this challenge, this paper introduces a test-time prompt tuning paradigm that optimizes a learnable prompt, thereby compelling the model to exploit genuine causal invariant features while dis
    
[^6]: ACPO: AI-Enabled Compiler-Driven Program Optimization

    ACPO: AI-Enabled Compiler-Driven Program Optimization

    [https://arxiv.org/abs/2312.09982](https://arxiv.org/abs/2312.09982)

    该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。

    

    该论文提出了ACPO：AI-Enabled Compiler-driven Program Optimization，这是一个新颖的框架，为LLVM提供简单全面的工具，以从应用机器学习模型来进行不同的优化通路中获益。首先展示了ACPO的高层视图、类层次结构和功能，然后通过将循环展开和函数内联传递的ML使能化，展示了ACPO的一些用例，描述了ACPO如何发挥作用。

    arXiv:2312.09982v2 Announce Type: replace-cross  Abstract: The key to performance optimization of a program is to decide correctly when a certain transformation should be applied by a compiler. This is an ideal opportunity to apply machine-learning models to speed up the tuning process; while this realization has been around since the late 90s, only recent advancements in ML enabled a practical application of ML to compilers as an end-to-end framework.   This paper presents ACPO: \textbf{\underline{A}}I-Enabled \textbf{\underline{C}}ompiler-driven \textbf{\underline{P}}rogram \textbf{\underline{O}}ptimization; a novel framework to provide LLVM with simple and comprehensive tools to benefit from employing ML models for different optimization passes. We first showcase the high-level view, class hierarchy, and functionalities of ACPO and subsequently, demonstrate a couple of use cases of ACPO by ML-enabling the Loop Unroll and Function Inlining passes and describe how ACPO can be leverage
    
[^7]: 语言模型与人脑的差异

    Divergences between Language Models and Human Brains

    [https://arxiv.org/abs/2311.09308](https://arxiv.org/abs/2311.09308)

    该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。

    

    机器和人类是否以相似的方式处理语言？最近的研究暗示肯定，发现大脑信号可以通过语言模型（LMs）的内部表示有效地进行预测。尽管这样的结果被认为反映了LMs和人类大脑之间的共享计算原理，但LMs和人类在语言表示和使用上也存在明显的差异。在这项工作中，我们通过检查LM表示和人类大脑对语言的响应之间的差异，通过采用两个数据集对受试者阅读和听叙述故事的方式，系统地探索了人类和机器语言处理之间的分歧。通过数据驱动的方法，我们确定了两个领域，即社交/情感智能和物理常识，这些领域在LMs中无法很好地捕捉到。然后，我们使用人类行为实验验证了这些领域，并证明在这些领域对LMs进行微调可以改善其性能。

    Do machines and humans process language in similar ways? Recent research has hinted in the affirmative, finding that brain signals can be effectively predicted using the internal representations of language models (LMs). Although such results are thought to reflect shared computational principles between LMs and human brains, there are also clear differences in how LMs and humans represent and use language. In this work, we systematically explore the divergences between human and machine language processing by examining the differences between LM representations and human brain responses to language as measured by Magnetoencephalography (MEG) across two datasets in which subjects read and listened to narrative stories. Using a data-driven approach, we identify two domains that are not captured well by LMs: social/emotional intelligence and physical commonsense. We then validate these domains with human behavioral experiments and show that fine-tuning LMs on these domains can improve th
    
[^8]: FLM-101B：一种开放的LLM和如何用10万美元预算来训练它

    FLM-101B: An Open LLM and How to Train It with $100K Budget. (arXiv:2309.03852v1 [cs.CL])

    [http://arxiv.org/abs/2309.03852](http://arxiv.org/abs/2309.03852)

    本文介绍了一种开放的LLM模型（FLM-101B）以及如何用10万美元的预算来训练它。通过采用增长策略，可以显著降低LLM训练的成本。同时，引入了一种系统的评估方法，以评估LLM的智能能力。

    

    大型语言模型（LLMs）在自然语言处理和多模态任务中取得了显著的成功。然而，它们的发展面临两个主要挑战：（i）高计算成本；（ii）难以进行公平客观的评估。LLMs的价格昂贵，只有少数几家主要参与者有能力进行训练，从而限制了研究和应用机会。这凸显了成本效益的LLM训练的重要性。在本文中，我们采用了一种增长策略，显著降低LLM训练成本。我们证明了可以在10万美元的预算下训练具有101B参数和0.31TB令牌的LLM。我们还采用了一种系统的评估范式，用于对LLMs进行智能的智商评估，这是针对现有评估更注重知识能力的补充。我们引入了包括符号映射、规则理解、模式挖掘在内的重要智能方面的评估基准。

    Large language models (LLMs) have achieved remarkable success in NLP and multimodal tasks. Despite these successes, their development faces two main challenges: (i) high computational cost; and (ii) difficulty in conducting fair and objective evaluations. LLMs are prohibitively expensive, making it feasible for only a few major players to undertake their training, thereby constraining both research and application opportunities. This underscores the importance of cost-effective LLM training. In this paper, we utilize a growth strategy to significantly reduce LLM training cost. We demonstrate that an LLM with 101B parameters and 0.31TB tokens can be trained on a $100K budget. We also adopt a systematic evaluation paradigm for the IQ evaluation of LLMs, in complement to existing evaluations that focus more on knowledge-oriented abilities. We introduce our benchmark including evaluations on important aspects of intelligence including symbolic mapping, itrule understanding, pattern mining,
    
[^9]: 文字想象的释放：通过探索文字的力量实现文本到图像的人物检索的新框架

    Unleashing the Imagination of Text: A Novel Framework for Text-to-image Person Retrieval via Exploring the Power of Words. (arXiv:2307.09059v1 [cs.CL])

    [http://arxiv.org/abs/2307.09059](http://arxiv.org/abs/2307.09059)

    本研究提出了一个新的框架，通过探索文本中的文字的力量，实现了准确地将抽象的文本描述映射到具体的图像，从而实现了文本到图像的人物检索。

    

    文本到图像的人物检索的目标是从大型图库中检索与给定文本描述相匹配的人物图像。这个任务的主要挑战在于视觉和文本模态之间信息表示的显著差异。文本模态通过词汇和语法结构传递抽象和精确的信息，而视觉模态通过图像传递具体和直观的信息。为了充分利用文字表示的表达力，准确地将抽象的文本描述映射到具体图像是至关重要的。为了解决这个问题，我们提出了一个新的框架，通过探索句子中的文字的力量，释放了文本到图像人物检索中的文字想象力。具体来说，该框架使用预训练的全面CLIP模型作为图像和文本的双编码器，利用先前的跨模态对齐知识。

    The goal of Text-to-image person retrieval is to retrieve person images from a large gallery that match the given textual descriptions. The main challenge of this task lies in the significant differences in information representation between the visual and textual modalities. The textual modality conveys abstract and precise information through vocabulary and grammatical structures, while the visual modality conveys concrete and intuitive information through images. To fully leverage the expressive power of textual representations, it is essential to accurately map abstract textual descriptions to specific images.  To address this issue, we propose a novel framework to Unleash the Imagination of Text (UIT) in text-to-image person retrieval, aiming to fully explore the power of words in sentences. Specifically, the framework employs the pre-trained full CLIP model as a dual encoder for the images and texts , taking advantage of prior cross-modal alignment knowledge. The Text-guided Imag
    
[^10]: 集合化的神经网络编码

    Set-based Neural Network Encoding. (arXiv:2305.16625v1 [cs.LG])

    [http://arxiv.org/abs/2305.16625](http://arxiv.org/abs/2305.16625)

    提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。

    

    我们提出了一种利用集合到集合和集合到向量函数来有效编码神经网络参数，进行泛化性能预测的神经网络权重编码方法。与之前需要对不同架构编写自定义编码模型的方法不同，我们的方法能够对混合架构和不同参数大小的模型动态编码。此外，我们的 SNE（集合化神经网络编码器）通过使用一种逐层编码方案，考虑神经网络的分层计算结构。最终将所有层次编码合并到一起，以获取神经网络编码矢量。我们还引入了“pad-chunk-encode”流水线来有效地编码神经网络层，该流水线可根据计算和内存限制进行调整。我们还引入了两个用于神经网络泛化性能预测的新任务：跨数据集和架构适应性预测。

    We propose an approach to neural network weight encoding for generalization performance prediction that utilizes set-to-set and set-to-vector functions to efficiently encode neural network parameters. Our approach is capable of encoding neural networks in a modelzoo of mixed architecture and different parameter sizes as opposed to previous approaches that require custom encoding models for different architectures. Furthermore, our \textbf{S}et-based \textbf{N}eural network \textbf{E}ncoder (SNE) takes into consideration the hierarchical computational structure of neural networks by utilizing a layer-wise encoding scheme that culminates to encoding all layer-wise encodings to obtain the neural network encoding vector. Additionally, we introduce a \textit{pad-chunk-encode} pipeline to efficiently encode neural network layers that is adjustable to computational and memory constraints. We also introduce two new tasks for neural network generalization performance prediction: cross-dataset a
    
[^11]: PastNet：引入物理归纳偏差用于时空视频预测

    PastNet: Introducing Physical Inductive Biases for Spatio-temporal Video Prediction. (arXiv:2305.11421v1 [cs.CV])

    [http://arxiv.org/abs/2305.11421](http://arxiv.org/abs/2305.11421)

    本文介绍了一种名为PastNet的新颖方法，通过在傅里叶域中引入谱卷积算子，利用内在的物理知识生成高质量的时空视频预测，并通过离散化局部特征降低计算成本。

    

    本文研究了时空视频预测的挑战，其中涉及根据历史数据流生成未来视频。现有方法通常利用语义地图等外部信息增强视频预测，但常常忽视视频内固有的物理知识。此外，它们的高计算需求可能会阻碍对高分辨率视频的应用。为解决这些限制，我们引入了一种新颖的方法，称为物理辅助时空网络（PastNet），用于生成高质量的视频预测。我们的PastNet核心在于在傅里叶域中引入谱卷积算子，从而有效地引入基本物理定律的归纳偏差。此外，我们使用一个内在维度估计的存储器库，在处理复杂的时空信号时离散化局部特征，从而降低计算成本。

    In this paper, we investigate the challenge of spatio-temporal video prediction, which involves generating future videos based on historical data streams. Existing approaches typically utilize external information such as semantic maps to enhance video prediction, which often neglect the inherent physical knowledge embedded within videos. Furthermore, their high computational demands could impede their applications for high-resolution videos. To address these constraints, we introduce a novel approach called Physics-assisted Spatio-temporal Network (PastNet) for generating high-quality video predictions. The core of our PastNet lies in incorporating a spectral convolution operator in the Fourier domain, which efficiently introduces inductive biases from the underlying physical laws. Additionally, we employ a memory bank with the estimated intrinsic dimensionality to discretize local features during the processing of complex spatio-temporal signals, thereby reducing computational costs 
    

