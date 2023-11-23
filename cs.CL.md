# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HARE: Explainable Hate Speech Detection with Step-by-Step Reasoning.](http://arxiv.org/abs/2311.00321) | HARE是一个支持逐步推理的可解释性仇恨言论检测框架，利用大型语言模型填补现有注释方案的推理差距，从而提高了检测模型的监督效果。 |
| [^2] | [On the Representational Capacity of Recurrent Neural Language Models.](http://arxiv.org/abs/2310.12942) | 本文研究了基于循环神经网络的语言模型的计算表达性，扩展了图灵完备性结果到概率情况，并提供了上下界分析。 |
| [^3] | [Lifelong Sequence Generation with Dynamic Module Expansion and Adaptation.](http://arxiv.org/abs/2310.09886) | 这项研究提出了一种动态模块扩展和自适应的方法，旨在解决终身序列生成中的持续学习问题。该方法允许模型根据任务相关性动态决定获取新知识的架构，并选择相似的先前任务来帮助适应新任务。此外，还引入了动态梯度缩放来平衡学习过程，以避免对先前学到的知识的严重遗忘。 |
| [^4] | [Merging Experts into One: Improving Computational Efficiency of Mixture of Experts.](http://arxiv.org/abs/2310.09832) | 本文提出了一种名为“合并专家”的计算高效的方法，通过将计算成本降低到单个专家的水平来改进混合专家方法的计算效率，实验证明该方法显著提高了计算效率。 |
| [^5] | [FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation.](http://arxiv.org/abs/2310.03214) | 本文提出了一种使用搜索引擎增强的方法，刷新大型语言模型。我们通过详细研究LLM生成的文本在回答问题方面的事实性，引入了FreshQA这一动态问答基准。通过人类评估，我们发现这些模型存在局限性，并表明有显著的改进空间。 |
| [^6] | [Pre-training Language Models for Comparative Reasoning.](http://arxiv.org/abs/2305.14457) | 本文提出一种预训练语言模型的新框架，旨在增强其在比较推理方面的能力。通过使用可扩展的基于文本实体比较数据的方法和新的预训练任务，该框架得到了显著的结果。 |
| [^7] | [Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches.](http://arxiv.org/abs/2211.08371) | 本文调查了目前语用学模型的研究现状，提出了建议并分析了语言含义的丰富性。未来的任务设计需要引出语用现象，并关注更广泛的交流上下文和效益的方向。 |
| [^8] | [GraphCFC: A Directed Graph based Cross-modal Feature Complementation Approach for Multimodal Conversational Emotion Recognition.](http://arxiv.org/abs/2207.12261) | 本文提出了一种基于有向图的跨模态特征补充方法，可以提取多模态上下文信息和交互信息，缓解了多模态融合中的异构性差距问题。 |
| [^9] | [Representation Projection Invariance Mitigates Representation Collapse.](http://arxiv.org/abs/2205.11603) | 本文提出了一种新的正则化方法 REPINA，旨在减少表示崩溃问题，结果在 13 个语言理解任务上表现出良好的效果。 |

# 详细

[^1]: HARE: 支持逐步推理的可解释性仇恨言论检测

    HARE: Explainable Hate Speech Detection with Step-by-Step Reasoning. (arXiv:2311.00321v1 [cs.CL])

    [http://arxiv.org/abs/2311.00321](http://arxiv.org/abs/2311.00321)

    HARE是一个支持逐步推理的可解释性仇恨言论检测框架，利用大型语言模型填补现有注释方案的推理差距，从而提高了检测模型的监督效果。

    

    随着社交媒体的普及，准确检测仇恨言论变得至关重要以确保在线安全。为了应对细微的仇恨言论形式，重要的是要识别并详细解释仇恨言论，以帮助用户理解其有害影响。最近的基准测试试图通过训练生成模型来处理仇恨文本中含义的自由文本注释，但我们发现现有注释方案存在重大推理差距，这可能会阻碍检测模型的监督。在本文中，我们引入了一种名为HARE的仇恨言论检测框架，该框架利用大型语言模型（LLM）的推理能力来填补这些关于仇恨言论解释的差距，从而实现有效的检测模型监督。在SBIC和Implicit Hate基准测试上的实验证明，我们的方法使用模型生成的数据始终优于使用现有自由文本注释的基准。分析表明，

    With the proliferation of social media, accurate detection of hate speech has become critical to ensure safety online. To combat nuanced forms of hate speech, it is important to identify and thoroughly explain hate speech to help users understand its harmful effects. Recent benchmarks have attempted to tackle this issue by training generative models on free-text annotations of implications in hateful text. However, we find significant reasoning gaps in the existing annotations schemes, which may hinder the supervision of detection models. In this paper, we introduce a hate speech detection framework, HARE, which harnesses the reasoning capabilities of large language models (LLMs) to fill these gaps in explanations of hate speech, thus enabling effective supervision of detection models. Experiments on SBIC and Implicit Hate benchmarks show that our method, using model-generated data, consistently outperforms baselines, using existing free-text human annotations. Analysis demonstrates th
    
[^2]: 关于循环神经网络语言模型的表示能力的研究

    On the Representational Capacity of Recurrent Neural Language Models. (arXiv:2310.12942v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.12942](http://arxiv.org/abs/2310.12942)

    本文研究了基于循环神经网络的语言模型的计算表达性，扩展了图灵完备性结果到概率情况，并提供了上下界分析。

    

    本研究调查了基于循环神经网络(RNNs)的语言模型(LMs)的计算表达性。Siegelmann和Sontag(1992)曾经展示了具有有理权重和隐藏状态以及无限计算时间的RNNs是图灵完备的。然而，LMs不仅定义了字符串上的加权，还定义了(非加权)语言成员关系，对RNN LMs（RLMs）的计算能力分析应该反映这一点。我们将图灵完备性结果扩展到概率情况，展示了如何使用有理权重的RLM和无限计算时间来模拟任何概率图灵机(PTM)。由于在实践中，RLMs实时工作，每个时间步骤处理一个符号，因此我们将上述结果作为RLMs表达性的上界。我们还通过展示在实时计算限制下，这些模型可以模拟确定性实时有理PTMs来提供下界。

    This work investigates the computational expressivity of language models (LMs) based on recurrent neural networks (RNNs). Siegelmann and Sontag (1992) famously showed that RNNs with rational weights and hidden states and unbounded computation time are Turing complete. However, LMs define weightings over strings in addition to just (unweighted) language membership and the analysis of the computational power of RNN LMs (RLMs) should reflect this. We extend the Turing completeness result to the probabilistic case, showing how a rationally weighted RLM with unbounded computation time can simulate any probabilistic Turing machine (PTM). Since, in practice, RLMs work in real-time, processing a symbol at every time step, we treat the above result as an upper bound on the expressivity of RLMs. We also provide a lower bound by showing that under the restriction to real-time computation, such models can simulate deterministic real-time rational PTMs.
    
[^3]: 动态模块扩展和自适应的终身序列生成

    Lifelong Sequence Generation with Dynamic Module Expansion and Adaptation. (arXiv:2310.09886v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09886](http://arxiv.org/abs/2310.09886)

    这项研究提出了一种动态模块扩展和自适应的方法，旨在解决终身序列生成中的持续学习问题。该方法允许模型根据任务相关性动态决定获取新知识的架构，并选择相似的先前任务来帮助适应新任务。此外，还引入了动态梯度缩放来平衡学习过程，以避免对先前学到的知识的严重遗忘。

    

    终身序列生成（LSG）是持续学习中的一个问题，旨在让模型在一系列生成任务上进行持续训练，以不断学习新的生成模式并避免遗忘先前的知识。现有的LSG方法主要关注维持旧知识，而对跨任务的知识传递关注较少。相比之下，人类可以通过利用先前获取的类似任务的知识更好地学习新任务。受人类学习范式的启发，我们提出了动态模块扩展和自适应（DMEA）的方法，该方法使模型能够根据任务相关性动态确定获取新知识的架构，并选择最相似的先前任务来促进对新任务的适应。此外，由于学习过程很容易偏向于当前任务，这可能导致更严重的遗忘先前学到的知识，因此我们提出了动态梯度缩放来平衡学习过程。

    Lifelong sequence generation (LSG), a problem in continual learning, aims to continually train a model on a sequence of generation tasks to learn constantly emerging new generation patterns while avoiding the forgetting of previous knowledge. Existing LSG methods mainly focus on maintaining old knowledge while paying little attention to knowledge transfer across tasks. In contrast, humans can better learn new tasks by leveraging previously acquired knowledge from similar tasks. Inspired by the learning paradigm of humans, we propose Dynamic Module Expansion and Adaptation (DMEA), which enables the model to dynamically determine the architecture for acquiring new knowledge based on task correlation and select the most similar previous tasks to facilitate adaptation to new tasks. In addition, as the learning process can easily be biased towards the current task which might cause more severe forgetting of previously learned knowledge, we propose dynamic gradient scaling to balance the lea
    
[^4]: 合并专家：改进混合专家方法的计算效率

    Merging Experts into One: Improving Computational Efficiency of Mixture of Experts. (arXiv:2310.09832v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09832](http://arxiv.org/abs/2310.09832)

    本文提出了一种名为“合并专家”的计算高效的方法，通过将计算成本降低到单个专家的水平来改进混合专家方法的计算效率，实验证明该方法显著提高了计算效率。

    

    将语言模型的规模扩大通常会带来NLP任务的显著进展，但往往会伴随着不断增长的计算成本。尽管稀疏的混合专家（MoE）可以通过激活每个输入的一个小子集参数（例如一个专家）来减少成本，但如果增加激活的专家数量，其计算将显著增加，限制了其实际效用。在本文中，我们首先展示了选择多个专家的优越性，然后提出了一种计算高效的方法，称为“合并专家”（MEO），将计算成本降低到单个专家的水平。广泛的实验表明，MEO显着提高了计算效率，例如，FLOPS从普通MoE的72.0G降低到28.6G（MEO）。此外，我们提出了一种基于标记的注意力模块，进一步增强了效率。

    Scaling the size of language models usually leads to remarkable advancements in NLP tasks. But it often comes with a price of growing computational cost. Although a sparse Mixture of Experts (MoE) can reduce the cost by activating a small subset of parameters (e.g., one expert) for each input, its computation escalates significantly if increasing the number of activated experts, limiting its practical utility. Can we retain the advantages of adding more experts without substantially increasing the computational costs? In this paper, we first demonstrate the superiority of selecting multiple experts and then propose a computation-efficient approach called \textbf{\texttt{Merging Experts into One}} (MEO), which reduces the computation cost to that of a single expert. Extensive experiments show that MEO significantly improves computational efficiency, e.g., FLOPS drops from 72.0G of vanilla MoE to 28.6G (MEO). Moreover, we propose a token-level attention block that further enhances the ef
    
[^5]: FreshLLMs: 使用搜索引擎增强的方法刷新大型语言模型

    FreshLLMs: Refreshing Large Language Models with Search Engine Augmentation. (arXiv:2310.03214v1 [cs.CL])

    [http://arxiv.org/abs/2310.03214](http://arxiv.org/abs/2310.03214)

    本文提出了一种使用搜索引擎增强的方法，刷新大型语言模型。我们通过详细研究LLM生成的文本在回答问题方面的事实性，引入了FreshQA这一动态问答基准。通过人类评估，我们发现这些模型存在局限性，并表明有显著的改进空间。

    

    大多数大型语言模型（LLMs）只训练一次，不进行更新，因此缺乏对不断变化的世界动态适应的能力。本研究在测试当前世界知识的问题回答的背景下，对LLM生成的文本的事实性进行了详细研究。具体而言，我们引入了FreshQA，一个新颖的动态问答基准，包括广泛的问题和答案类型，包括需要快速变化的世界知识和需要揭示错误前提的问题。我们在一个双模式评估过程中对多种闭源和开源LLM进行了基准测试，可以同时测量正确性和虚构性。通过涉及超过50K个评判的人类评估，我们揭示了这些模型的局限性，并证明了改进的显著空间：例如，所有模型（无论模型大小如何）在涉及快速变化的知识和错误前提的问题上都面临困难。

    Most large language models (LLMs) are trained once and never updated; thus, they lack the ability to dynamically adapt to our ever-changing world. In this work, we perform a detailed study of the factuality of LLM-generated text in the context of answering questions that test current world knowledge. Specifically, we introduce FreshQA, a novel dynamic QA benchmark encompassing a diverse range of question and answer types, including questions that require fast-changing world knowledge as well as questions with false premises that need to be debunked. We benchmark a diverse array of both closed and open-source LLMs under a two-mode evaluation procedure that allows us to measure both correctness and hallucination. Through human evaluations involving more than 50K judgments, we shed light on limitations of these models and demonstrate significant room for improvement: for instance, all models (regardless of model size) struggle on questions that involve fast-changing knowledge and false pr
    
[^6]: 为比较推理预训练语言模型

    Pre-training Language Models for Comparative Reasoning. (arXiv:2305.14457v1 [cs.CL])

    [http://arxiv.org/abs/2305.14457](http://arxiv.org/abs/2305.14457)

    本文提出一种预训练语言模型的新框架，旨在增强其在比较推理方面的能力。通过使用可扩展的基于文本实体比较数据的方法和新的预训练任务，该框架得到了显著的结果。

    

    本文提出了一种新框架，用于预训练语言模型以增强其在文本比较推理方面的能力。我们的方法涉及可扩展的用于收集基于文本实体比较数据的方法，并设计了三个新的预训练任务。在多个下游任务，包括比较问答、问句生成和摘要生成方面的评估表明，我们的预训练框架大大提高了语言模型的比较推理能力，尤其是在资源匮乏的情况下。此外，本工作还发布了第一个比较推理综合基准。

    In this paper, we propose a novel framework to pre-train language models for enhancing their abilities of comparative reasoning over texts. While recent research has developed models for NLP tasks that require comparative reasoning, they suffer from costly manual data labeling and limited generalizability to different tasks. Our approach involves a scalable method for collecting data for text-based entity comparison, which leverages both structured and unstructured data, and the design of three novel pre-training tasks. Evaluation on a range of downstream tasks including comparative question answering, question generation, and summarization shows that our pre-training framework significantly improves the comparative reasoning abilities of language models, especially under low-resource conditions. This work also releases the first integrated benchmark for comparative reasoning over texts.
    
[^7]: 语言基础中的语用学：现象、任务和建模方法

    Pragmatics in Language Grounding: Phenomena, Tasks, and Modeling Approaches. (arXiv:2211.08371v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.08371](http://arxiv.org/abs/2211.08371)

    本文调查了目前语用学模型的研究现状，提出了建议并分析了语言含义的丰富性。未来的任务设计需要引出语用现象，并关注更广泛的交流上下文和效益的方向。

    

    人们在交流中经常依赖上下文来丰富言外之意，从而实现简明而有效的沟通。为了能够与人类成功地自然交互，面向用户的人工智能系统将需要类似的语用学技能：依靠各种上下文信息——从共享的语言目标和约定到视觉和具身世界，有效地使用语言。我们调查了现有的语境设置和语用建模方法，并分析了每个工作中任务目标、环境上下文和交际效益如何丰富语言含义。我们提出了未来基础任务设计的建议，以自然地引出语用学现象，并建议关注更广泛的交流上下文和效益的方向。

    People rely heavily on context to enrich meaning beyond what is literally said, enabling concise but effective communication. To interact successfully and naturally with people, user-facing artificial intelligence systems will require similar skills in pragmatics: relying on various types of context -from shared linguistic goals and conventions, to the visual and embodied world -- to use language effectively. We survey existing grounded settings and pragmatic modeling approaches and analyze how the task goals, environmental contexts, and communicative affordances in each work enrich linguistic meaning. We present recommendations for future grounded task design to naturally elicit pragmatic phenomena, and suggest directions that focus on a broader range of communicative contexts and affordances.
    
[^8]: 基于有向图的跨模态特征补充方法用于多模态对话情感识别

    GraphCFC: A Directed Graph based Cross-modal Feature Complementation Approach for Multimodal Conversational Emotion Recognition. (arXiv:2207.12261v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.12261](http://arxiv.org/abs/2207.12261)

    本文提出了一种基于有向图的跨模态特征补充方法，可以提取多模态上下文信息和交互信息，缓解了多模态融合中的异构性差距问题。

    

    对话情感识别在人机交互系统中起着重要作用，因为它可以提供有共情心理的服务。多模态对话情感识别可以缓解单模态方法的缺点。最近，由于关系建模方面的卓越性能，图神经网络已被广泛用于各种领域。在多模态对话情感识别中，图神经网络能够提取远距离的上下文信息和跨模态的交互信息。不幸的是，由于现有方法（如MMGCN）直接融合多个模态，可能会产生冗余信息，且可能丢失多样化的信息。在本文中，我们提出了一种基于有向图的跨模态特征补充（GraphCFC）模块，可以有效地模拟上下文和互动信息。GraphCFC通过利用多个子空间提取器和成对跨模态补充（PairCC）策略，缓解了多模态融合中的异构性差距问题。

    Emotion Recognition in Conversation (ERC) plays a significant part in Human-Computer Interaction (HCI) systems since it can provide empathetic services. Multimodal ERC can mitigate the drawbacks of uni-modal approaches. Recently, Graph Neural Networks (GNNs) have been widely used in a variety of fields due to their superior performance in relation modeling. In multimodal ERC, GNNs are capable of extracting both long-distance contextual information and inter-modal interactive information. Unfortunately, since existing methods such as MMGCN directly fuse multiple modalities, redundant information may be generated and diverse information may be lost. In this work, we present a directed Graph based Cross-modal Feature Complementation (GraphCFC) module that can efficiently model contextual and interactive information. GraphCFC alleviates the problem of heterogeneity gap in multimodal fusion by utilizing multiple subspace extractors and Pair-wise Cross-modal Complementary (PairCC) strategy. 
    
[^9]: 表示投影不变性缓解表示崩溃问题

    Representation Projection Invariance Mitigates Representation Collapse. (arXiv:2205.11603v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.11603](http://arxiv.org/abs/2205.11603)

    本文提出了一种新的正则化方法 REPINA，旨在减少表示崩溃问题，结果在 13 个语言理解任务上表现出良好的效果。

    

    对预训练语言模型学习的上下文化表示进行微调在自然语言处理领域中仍然是一种流行的做法。然而，微调可能会导致表示降级（也被称为表示崩溃），这可能会导致不稳定性、次优性能和弱泛化。在本文中，我们提出了“表示投影不变性”（REPINA），这是一种新颖的正则化方法，通过抑制表示中的不良变化来维护表示的信息内容并减少表示崩溃问题。我们研究了所提出的正则化与5个可比较基线在13个语言理解任务（GLUE基准测试和其他六个数据集）中的实证行为。在评估内域性能时，REPINA 在大多数任务（13项中的10项）上始终优于其他基线。我们还证明了它在少样本设置中的有效性和对标签扰动的鲁棒性。作为副产品，我们扩展了已有的先前工作的范围，这些工作通过包括预测任务在内的自监督学习来降低表示崩溃率。

    Fine-tuning contextualized representations learned by pre-trained language models remains a prevalent practice in NLP. However, fine-tuning can lead to representation degradation (also known as representation collapse), which may result in instability, sub-optimal performance, and weak generalization.  In this paper, we propose Representation Projection Invariance (REPINA), a novel regularization method to maintain the information content of representation and reduce representation collapse during fine-tuning by discouraging undesirable changes in the representations. We study the empirical behavior of the proposed regularization in comparison to 5 comparable baselines across 13 language understanding tasks (GLUE benchmark and six additional datasets). When evaluating in-domain performance, REPINA consistently outperforms other baselines on most tasks (10 out of 13). We also demonstrate its effectiveness in few-shot settings and robustness to label perturbation. As a by-product, we ext
    

