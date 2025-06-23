# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Analysis of Switchback Designs in Reinforcement Learning](https://arxiv.org/abs/2403.17285) | 本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。 |
| [^2] | [ChatDBG: An AI-Powered Debugging Assistant](https://arxiv.org/abs/2403.16354) | ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。 |
| [^3] | [ICC: Quantifying Image Caption Concreteness for Multimodal Dataset Curation](https://arxiv.org/abs/2403.01306) | 提出一种新的度量标准，图像描述具体性，用于评估标题文本的具体性和相关性，以帮助在多模态学习中隔离提供最强信号的最具体样本。 |
| [^4] | [Neural optimal controller for stochastic systems via pathwise HJB operator](https://arxiv.org/abs/2402.15592) | 本文提出了一种基于路径HJB算子的随机系统神经最优控制器的算法，通过引入基于物理信息学习的方法解决高维随机控制问题，展示了其在各种应用程序上的性能。 |
| [^5] | [AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability](https://arxiv.org/abs/2402.09404) | AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。 |
| [^6] | [Safe Guaranteed Exploration for Non-linear Systems](https://arxiv.org/abs/2402.06562) | 本文提出了一个新颖的安全保证探索框架，使用最优控制实现了对非线性系统的有限时间样本复杂度界限的保证探索，同时具有证明的安全性和广泛适用性。 |
| [^7] | [Open-Set Graph Anomaly Detection via Normal Structure Regularisation](https://arxiv.org/abs/2311.06835) | 通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力 |
| [^8] | [On the generalization of Tanimoto-type kernels to real valued functions](https://arxiv.org/abs/2007.05943) | 本论文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性，并提供了一种光滑逼近的方法。 |
| [^9] | [Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning.](http://arxiv.org/abs/2401.03756) | 该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。 |
| [^10] | [The Impact of Equal Opportunity on Statistical Discrimination.](http://arxiv.org/abs/2310.04585) | 本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。 |
| [^11] | [Sheaf Hypergraph Networks.](http://arxiv.org/abs/2309.17116) | 这项研究通过引入细胞空间来增强超图的表示能力，开发了线性和非线性的细胞超图拉普拉斯算子，为有效建模复杂数据结构提供了强大的工具。 |
| [^12] | [Layer-wise Feedback Propagation.](http://arxiv.org/abs/2308.12053) | 本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。 |
| [^13] | [Conformal prediction for frequency-severity modeling.](http://arxiv.org/abs/2307.13124) | 这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。 |
| [^14] | [Voices of Her: Analyzing Gender Differences in the AI Publication World.](http://arxiv.org/abs/2305.14597) | 通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。 |

# 详细

[^1]: 对强化学习中的往返设计进行的分析

    An Analysis of Switchback Designs in Reinforcement Learning

    [https://arxiv.org/abs/2403.17285](https://arxiv.org/abs/2403.17285)

    本文通过提出“弱信号分析”框架，研究了强化学习中往返设计对平均处理效应估计准确性的影响，发现在大部分奖励误差为正相关时，往返设计比每日切换策略更有效，增加政策切换频率可以降低平均处理效应估计器的均方误差。

    

    本文提供了对A/B测试中往返设计的详细研究，这些设计随时间在基准和新策略之间交替。我们的目标是全面评估这些设计对其产生的平均处理效应（ATE）估计器准确性的影响。我们提出了一个新颖的“弱信号分析”框架，大大简化了这些ATE的均方误差（MSE）在马尔科夫决策过程环境中的计算。我们的研究结果表明：(i) 当大部分奖励误差呈正相关时，往返设计比每日切换策略的交替设计更有效。此外，增加政策切换的频率往往会降低ATE估计器的MSE。(ii) 然而，当误差不相关时，所有这些设计变得渐近等效。(iii) 在大多数误差为负相关时

    arXiv:2403.17285v1 Announce Type: cross  Abstract: This paper offers a detailed investigation of switchback designs in A/B testing, which alternate between baseline and new policies over time. Our aim is to thoroughly evaluate the effects of these designs on the accuracy of their resulting average treatment effect (ATE) estimators. We propose a novel "weak signal analysis" framework, which substantially simplifies the calculations of the mean squared errors (MSEs) of these ATEs in Markov decision process environments. Our findings suggest that (i) when the majority of reward errors are positively correlated, the switchback design is more efficient than the alternating-day design which switches policies in a daily basis. Additionally, increasing the frequency of policy switches tends to reduce the MSE of the ATE estimator. (ii) When the errors are uncorrelated, however, all these designs become asymptotically equivalent. (iii) In cases where the majority of errors are negative correlate
    
[^2]: ChatDBG: 一种基于人工智能的调试助手

    ChatDBG: An AI-Powered Debugging Assistant

    [https://arxiv.org/abs/2403.16354](https://arxiv.org/abs/2403.16354)

    ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。

    

    本文介绍了ChatDBG，这是第一个基于人工智能的调试助手。ChatDBG集成了大型语言模型(LLMs)，显著增强了传统调试器的功能和用户友好性。ChatDBG允许程序员与调试器进行协作对话，使他们能够提出关于程序状态的复杂问题，对崩溃或断言失败进行根本原因分析，并探索诸如“为什么x为空？”之类的开放性查询。为了处理这些查询，ChatDBG授予LLM自主权，通过发出命令来浏览堆栈和检查程序状态进行调试；然后报告其发现并将控制权交还给程序员。我们的ChatDBG原型与标准调试器集成，包括LLDB、GDB和WinDBG用于本地代码以及用于Python的Pdb。我们在各种代码集合上进行了评估，包括具有已知错误的C/C++代码和一套Python代码。

    arXiv:2403.16354v1 Announce Type: cross  Abstract: This paper presents ChatDBG, the first AI-powered debugging assistant. ChatDBG integrates large language models (LLMs) to significantly enhance the capabilities and user-friendliness of conventional debuggers. ChatDBG lets programmers engage in a collaborative dialogue with the debugger, allowing them to pose complex questions about program state, perform root cause analysis for crashes or assertion failures, and explore open-ended queries like "why is x null?". To handle these queries, ChatDBG grants the LLM autonomy to take the wheel and drive debugging by issuing commands to navigate through stacks and inspect program state; it then reports its findings and yields back control to the programmer. Our ChatDBG prototype integrates with standard debuggers including LLDB, GDB, and WinDBG for native code and Pdb for Python. Our evaluation across a diverse set of code, including C/C++ code with known bugs and a suite of Python code includi
    
[^3]: ICC：用于多模态数据集筛选的图像描述具体性量化

    ICC: Quantifying Image Caption Concreteness for Multimodal Dataset Curation

    [https://arxiv.org/abs/2403.01306](https://arxiv.org/abs/2403.01306)

    提出一种新的度量标准，图像描述具体性，用于评估标题文本的具体性和相关性，以帮助在多模态学习中隔离提供最强信号的最具体样本。

    

    arXiv:2403.01306v1 公告类型：新摘要：针对配对文本-图像数据的Web规模训练在多模态学习中变得越来越重要，但挑战在野外数据集的高噪声特性。标准数据过滤方法成功去除了不匹配的文本-图像对，但允许语义相关但非常抽象或主观的文本。这些方法缺乏细粒度的能力来隔离提供在嘈杂数据集中学习最强信号的最具体样本。在这项工作中，我们提出了一种新的度量标准，图像描述具体性，评估没有图像参考的标题文本以衡量其具体性和相关性，以供在多模态学习中使用。我们的方法利用了衡量视觉-语义信息损失的强基础模型来进行评估。我们证明了这与人类对单词和句子级文本具体性的评估高度相关。此外，我们展示了...

    arXiv:2403.01306v1 Announce Type: new  Abstract: Web-scale training on paired text-image data is becoming increasingly central to multimodal learning, but is challenged by the highly noisy nature of datasets in the wild. Standard data filtering approaches succeed in removing mismatched text-image pairs, but permit semantically related but highly abstract or subjective text. These approaches lack the fine-grained ability to isolate the most concrete samples that provide the strongest signal for learning in a noisy dataset. In this work, we propose a new metric, image caption concreteness, that evaluates caption text without an image reference to measure its concreteness and relevancy for use in multimodal learning. Our approach leverages strong foundation models for measuring visual-semantic information loss in multimodal representations. We demonstrate that this strongly correlates with human evaluation of concreteness in both single-word and sentence-level texts. Moreover, we show tha
    
[^4]: 基于路径HJB算子的随机系统神经最优控制器

    Neural optimal controller for stochastic systems via pathwise HJB operator

    [https://arxiv.org/abs/2402.15592](https://arxiv.org/abs/2402.15592)

    本文提出了一种基于路径HJB算子的随机系统神经最优控制器的算法，通过引入基于物理信息学习的方法解决高维随机控制问题，展示了其在各种应用程序上的性能。

    

    本文旨在基于基于物理信息学习和动态规划，为高维随机控制问题开发基于深度学习的算法。与依赖于Hamilton-Jacobi-Bellman（HJB）方程解的概率表示的经典深度学习方法不同，我们引入了与HJB方程相关的路径算子，从而可以定义一个基于物理信息学习的问题。根据最优控制是否具有显式表示，提出了两种数值方法来解决基于物理信息学习的问题。我们对截断、近似和优化误差如何影响这些方法的准确性进行了误差分析。通过在各种应用程序上展示数值结果，说明了所提算法的性能。

    arXiv:2402.15592v1 Announce Type: cross  Abstract: The aim of this work is to develop deep learning-based algorithms for high-dimensional stochastic control problems based on physics-informed learning and dynamic programming. Unlike classical deep learning-based methods relying on a probabilistic representation of the solution to the Hamilton--Jacobi--Bellman (HJB) equation, we introduce a pathwise operator associated with the HJB equation so that we can define a problem of physics-informed learning. According to whether the optimal control has an explicit representation, two numerical methods are proposed to solve the physics-informed learning problem. We provide an error analysis on how the truncation, approximation and optimization errors affect the accuracy of these methods. Numerical results on various applications are presented to illustrate the performance of the proposed algorithms.
    
[^5]: AQA-Bench：评估LLM顺序推理能力的交互式基准测试

    AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability

    [https://arxiv.org/abs/2402.09404](https://arxiv.org/abs/2402.09404)

    AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。

    

    该论文介绍了一种新的基准测试AQA-Bench，用于评估大型语言模型（LLMs）在算法上下文中，如深度优先搜索（DFS）等的顺序推理能力。我们评估基准测试的关键特点在于其交互式评估协议-例如，在DFS中，每个节点的可用连接边取决于模型对该节点的遍历，因此需要LLM有效地记住已访问节点并策划后续移动的能力。我们使用三种不同的算法构建了AQA-Bench，分别是二分搜索，深度优先搜索和广度优先搜索，并评估了12种不同的LLMs的顺序推理能力。我们的调查揭示了一些有趣的发现：（1）类似GPT-4和Gemini等闭源模型通常显示出强大的顺序推理能力，明显优于开源LLMs。（2）天真地提供互操作性

    arXiv:2402.09404v1 Announce Type: cross Abstract: This paper introduces AQA-Bench, a novel benchmark to assess the sequential reasoning capabilities of large language models (LLMs) in algorithmic contexts, such as depth-first search (DFS). The key feature of our evaluation benchmark lies in its interactive evaluation protocol -- for example, in DFS, the availability of each node's connected edge is contingent upon the model's traversal to that node, thereby necessitating the LLM's ability to effectively remember visited nodes and strategize subsequent moves. We comprehensively build AQA-Bench with three different algorithms, namely binary search, depth-first search, and breadth-first search, and to evaluate the sequential reasoning ability of 12 different LLMs. Our investigations reveal several interesting findings: (1) Closed-source models like GPT-4 and Gemini generally show strong sequential reasoning ability, significantly outperforming open-source LLMs. (2) Naively providing inter
    
[^6]: 非线性系统的安全保证探索

    Safe Guaranteed Exploration for Non-linear Systems

    [https://arxiv.org/abs/2402.06562](https://arxiv.org/abs/2402.06562)

    本文提出了一个新颖的安全保证探索框架，使用最优控制实现了对非线性系统的有限时间样本复杂度界限的保证探索，同时具有证明的安全性和广泛适用性。

    

    在具有先验未知约束的环境中进行安全探索是限制机器人自主性的基本挑战。虽然安全性至关重要，但对足够探索的保证对于确保自主任务完成也至关重要。为了解决这些挑战，我们提出了一个新颖的安全保证探索框架，利用最优控制实现前所未有的结果：具有有限时间样本复杂度界限的非线性系统的保证探索，同时在任意高概率下被证明是安全的。该框架具有广泛的适用性，可适用于具有复杂非线性动力学和未知领域的许多实际场景。基于这个框架，我们提出了一种高效的算法，SageMPC，采用模型预测控制进行安全保证探索。SageMPC通过整合三种技术来提高效率：i) 利用Lipschitz边界，ii) 目标导向探索，和iii) 逐步调整风格的重新规划，同时保持高效性。

    Safely exploring environments with a-priori unknown constraints is a fundamental challenge that restricts the autonomy of robots. While safety is paramount, guarantees on sufficient exploration are also crucial for ensuring autonomous task completion. To address these challenges, we propose a novel safe guaranteed exploration framework using optimal control, which achieves first-of-its-kind results: guaranteed exploration for non-linear systems with finite time sample complexity bounds, while being provably safe with arbitrarily high probability. The framework is general and applicable to many real-world scenarios with complex non-linear dynamics and unknown domains. Based on this framework we propose an efficient algorithm, SageMPC, SAfe Guaranteed Exploration using Model Predictive Control. SageMPC improves efficiency by incorporating three techniques: i) exploiting a Lipschitz bound, ii) goal-directed exploration, and iii) receding horizon style re-planning, all while maintaining th
    
[^7]: 通过正常结构规范化实现开放图异常检测

    Open-Set Graph Anomaly Detection via Normal Structure Regularisation

    [https://arxiv.org/abs/2311.06835](https://arxiv.org/abs/2311.06835)

    通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力

    

    本文考虑了一个重要的图异常检测（GAD）任务，即开放式GAD，旨在使用少量标记的训练正常节点和异常节点（称为已知异常）来检测异常节点，这些节点无法展示所有可能的推理时异常。已标记数据的可用性为GAD模型提供了关键的异常先验知识，可大大降低检测错误。然而，当前方法往往过分强调拟合已知异常，导致对未知异常（即未被标记的异常节点）的弱泛化能力。此外，它们被引入以处理欧几里德数据，未能有效捕捉GAD的重要非欧几里德特征。在这项工作中，我们提出了一种新颖的开放式GAD方法，即正常结构规范化（NSReg），以实现对未知异常的广义检测能力。

    arXiv:2311.06835v2 Announce Type: replace-cross  Abstract: This paper considers an important Graph Anomaly Detection (GAD) task, namely open-set GAD, which aims to detect anomalous nodes using a small number of labelled training normal and anomaly nodes (known as seen anomalies) that cannot illustrate all possible inference-time abnormalities. The availability of that labelled data provides crucial prior knowledge about abnormalities for GAD models, enabling substantially reduced detection errors. However, current methods tend to over-emphasise fitting the seen anomalies, leading to a weak generalisation ability to detect unseen anomalies, i.e., those that are not illustrated by the labelled anomaly nodes. Further, they were introduced to handle Euclidean data, failing to effectively capture important non-Euclidean features for GAD. In this work, we propose a novel open-set GAD approach, namely Normal Structure Regularisation (NSReg), to achieve generalised detection ability to unseen 
    
[^8]: 将Tanimoto类型核泛化到实值函数

    On the generalization of Tanimoto-type kernels to real valued functions

    [https://arxiv.org/abs/2007.05943](https://arxiv.org/abs/2007.05943)

    本论文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性，并提供了一种光滑逼近的方法。

    

    Tanimoto核（Jaccard指数）是描述二值属性集相似性的知名工具。已将其扩展到属性为非负实数值的情况。本文介绍了一种更一般的Tanimoto核公式，允许衡量任意实值函数的相似性。通过通过适当选择的集合统一属性表示来构建此扩展。在推导核的一般形式后，从核函数中提取了显式特征表示，并展示了将一般核包含到Tanimoto核中的简单方法。最后，核也表示为分段线性函数的商，并提供了光滑逼近。

    arXiv:2007.05943v2 Announce Type: replace  Abstract: The Tanimoto kernel (Jaccard index) is a well known tool to describe the similarity between sets of binary attributes. It has been extended to the case when the attributes are nonnegative real values. This paper introduces a more general Tanimoto kernel formulation which allows to measure the similarity of arbitrary real-valued functions. This extension is constructed by unifying the representation of the attributes via properly chosen sets. After deriving the general form of the kernel, explicit feature representation is extracted from the kernel function, and a simply way of including general kernels into the Tanimoto kernel is shown. Finally, the kernel is also expressed as a quotient of piecewise linear functions, and a smooth approximation is provided.
    
[^9]: 上下文固定预算的最佳臂识别：适应性实验设计与策略学习

    Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning. (arXiv:2401.03756v1 [cs.LG])

    [http://arxiv.org/abs/2401.03756](http://arxiv.org/abs/2401.03756)

    该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。

    

    个性化治疗推荐是基于证据的决策中的关键任务。在这项研究中，我们将这个任务作为一个带有上下文信息的固定预算最佳臂识别（Best Arm Identification, BAI）问题来进行建模。在这个设置中，我们考虑了一个给定多个治疗臂的自适应试验。在每一轮中，决策者观察一个刻画实验单位的上下文（协变量），并将该单位分配给其中一个治疗臂。在实验结束时，决策者推荐一个在给定上下文条件下预计产生最高期望结果的治疗臂（最佳治疗臂）。该决策的有效性通过最坏情况下的期望简单遗憾（策略遗憾）来衡量，该遗憾表示在给定上下文条件下，最佳治疗臂和推荐治疗臂的条件期望结果之间的最大差异。我们的初始步骤是推导最坏情况下期望简单遗憾的渐近下界，该下界还暗示着解决该问题的一些思路。

    Individualized treatment recommendation is a crucial task in evidence-based decision-making. In this study, we formulate this task as a fixed-budget best arm identification (BAI) problem with contextual information. In this setting, we consider an adaptive experiment given multiple treatment arms. At each round, a decision-maker observes a context (covariate) that characterizes an experimental unit and assigns the unit to one of the treatment arms. At the end of the experiment, the decision-maker recommends a treatment arm estimated to yield the highest expected outcome conditioned on a context (best treatment arm). The effectiveness of this decision is measured in terms of the worst-case expected simple regret (policy regret), which represents the largest difference between the conditional expected outcomes of the best and recommended treatment arms given a context. Our initial step is to derive asymptotic lower bounds for the worst-case expected simple regret, which also implies idea
    
[^10]: 机会平等对统计性歧视的影响

    The Impact of Equal Opportunity on Statistical Discrimination. (arXiv:2310.04585v1 [econ.TH])

    [http://arxiv.org/abs/2310.04585](http://arxiv.org/abs/2310.04585)

    本文通过修改统计性歧视模型，考虑了由机器学习生成的可合同化信念，给监管者提供了一种超过肯定行动的工具，通过要求公司选取一个平衡不同群体真正阳性率的决策策略，实现机会平等来消除统计性歧视。

    

    本文修改了Coate和Loury（1993）的经典统计性歧视模型，假设公司对个体未观察到的类别的信念是由机器学习生成的，因此是可合同化的。这扩展了监管者的工具箱，超出了像肯定行动这样的无信念规定。可合同化的信念使得要求公司选择一个决策策略，使得不同群体之间的真正阳性率相等（算法公平文献中所称的机会平等）成为可能。尽管肯定行动不一定能消除统计性歧视，但本文表明实施机会平等可以做到。

    I modify the canonical statistical discrimination model of Coate and Loury (1993) by assuming the firm's belief about an individual's unobserved class is machine learning-generated and, therefore, contractible. This expands the toolkit of a regulator beyond belief-free regulations like affirmative action. Contractible beliefs make it feasible to require the firm to select a decision policy that equalizes true positive rates across groups -- what the algorithmic fairness literature calls equal opportunity. While affirmative action does not necessarily end statistical discrimination, I show that imposing equal opportunity does.
    
[^11]: Sheaf Hypergraph Networks. (arXiv:2309.17116v1 [cs.LG])

    Sheaf Hypergraph Networks. (arXiv:2309.17116v1 [cs.LG])

    [http://arxiv.org/abs/2309.17116](http://arxiv.org/abs/2309.17116)

    这项研究通过引入细胞空间来增强超图的表示能力，开发了线性和非线性的细胞超图拉普拉斯算子，为有效建模复杂数据结构提供了强大的工具。

    

    高阶关系在自然界中十分普遍，许多现象涉及复杂的相互作用，超出了简单的两两连接。因此，提升高阶处理能力可以加速各个需要结构化数据的领域的发展。目前的方法通常使用超图来表示这些相互作用。我们通过引入细胞空间对超图进行增强，这是一种数学构造，在维持局部高阶连通性的同时为传统超图添加额外的结构。受现有文献中的拉普拉斯算子的启发，我们分别开发了两种独特的细胞超图拉普拉斯的形式：线性和非线性。我们的理论分析表明，将细胞空间引入超图拉普拉斯比标准的超图扩散提供了更富有表现力的归纳偏见，为有效建模复杂数据结构提供了强大的工具。我们应用这些方法...

    Higher-order relations are widespread in nature, with numerous phenomena involving complex interactions that extend beyond simple pairwise connections. As a result, advancements in higher-order processing can accelerate the growth of various fields requiring structured data. Current approaches typically represent these interactions using hypergraphs. We enhance this representation by introducing cellular sheaves for hypergraphs, a mathematical construction that adds extra structure to the conventional hypergraph while maintaining their local, higherorder connectivity. Drawing inspiration from existing Laplacians in the literature, we develop two unique formulations of sheaf hypergraph Laplacians: linear and non-linear. Our theoretical analysis demonstrates that incorporating sheaves into the hypergraph Laplacian provides a more expressive inductive bias than standard hypergraph diffusion, creating a powerful instrument for effectively modelling complex data structures. We employ these 
    
[^12]: 层级反馈传播

    Layer-wise Feedback Propagation. (arXiv:2308.12053v1 [cs.LG])

    [http://arxiv.org/abs/2308.12053](http://arxiv.org/abs/2308.12053)

    本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。

    

    本文提出了一种称为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，该方法利用可解释性，具体而言是层级相关性传播（LRP），根据每个连接对解决给定任务的贡献独立分配奖励。这与传统的梯度下降方法不同，梯度下降方法是朝向估计的损失最小值更新参数。LFP在模型中传播奖励信号，而无需梯度计算。它增强接收到正反馈的结构，同时降低接收到负反馈的结构的影响。我们从理论和实证的角度证明了LFP的收敛性，并展示了它在各种模型和数据集上实现与梯度下降相当的性能。值得注意的是，LFP克服了梯度方法的某些局限性，例如对有意义的导数的依赖。我们进一步研究了LFP如何解决梯度方法相关问题的限制。

    In this paper, we present Layer-wise Feedback Propagation (LFP), a novel training approach for neural-network-like predictors that utilizes explainability, specifically Layer-wise Relevance Propagation(LRP), to assign rewards to individual connections based on their respective contributions to solving a given task. This differs from traditional gradient descent, which updates parameters towards anestimated loss minimum. LFP distributes a reward signal throughout the model without the need for gradient computations. It then strengthens structures that receive positive feedback while reducingthe influence of structures that receive negative feedback. We establish the convergence of LFP theoretically and empirically, and demonstrate its effectiveness in achieving comparable performance to gradient descent on various models and datasets. Notably, LFP overcomes certain limitations associated with gradient-based methods, such as reliance on meaningful derivatives. We further investigate how 
    
[^13]: 频率-严重性建模的符合性预测

    Conformal prediction for frequency-severity modeling. (arXiv:2307.13124v1 [stat.ME])

    [http://arxiv.org/abs/2307.13124](http://arxiv.org/abs/2307.13124)

    这个论文提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，扩展了split conformal prediction技术到两阶段频率-严重性建模领域，并通过使用随机森林作为严重性模型，利用了袋外机制消除了校准集的需要，并实现了具有自适应宽度的预测区间的生成。

    

    我们提出了一个非参数的模型无关框架，用于建立保险理赔的预测区间，并具有有限样本的统计保证，将分割符合性预测技术扩展到两阶段频率-严重性建模领域。通过模拟和真实数据集展示了该框架的有效性。当基础严重性模型是随机森林时，我们扩展了两阶段分割符合性预测过程，展示了如何利用袋外机制消除校准集的需要，并实现具有自适应宽度的预测区间的生成。

    We present a nonparametric model-agnostic framework for building prediction intervals of insurance claims, with finite sample statistical guarantees, extending the technique of split conformal prediction to the domain of two-stage frequency-severity modeling. The effectiveness of the framework is showcased with simulated and real datasets. When the underlying severity model is a random forest, we extend the two-stage split conformal prediction procedure, showing how the out-of-bag mechanism can be leveraged to eliminate the need for a calibration set and to enable the production of prediction intervals with adaptive width.
    
[^14]: 她们的声音：分析人工智能出版领域的性别差异

    Voices of Her: Analyzing Gender Differences in the AI Publication World. (arXiv:2305.14597v1 [cs.CL])

    [http://arxiv.org/abs/2305.14597](http://arxiv.org/abs/2305.14597)

    通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。

    

    虽然先前的研究已经分析了学术界中的性别偏见，但是我们仍然缺乏一个全面的人工智能社区性别差异的分析，涵盖各种主题和不同的发展趋势。我们使用AI Scholar数据集中的78K位AI领域的研究人员，发现了一些性别差异：（1）虽然女性研究人员的总引用次数比男性少，但这种引用差异并不适用于所有学术年龄组；（2）在AI论文的合著中存在很大的性别同质性；（3）女性第一作者的论文显示出不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题。我们的分析为我们的AI社区现有的人口统计趋势提供了一个窗口，并鼓励在未来实现更多的性别平等和多样性。我们的代码和数据可在https://github.com/causalNLP/ai-scholar-gender找到。

    While several previous studies have analyzed gender bias in research, we are still missing a comprehensive analysis of gender differences in the AI community, covering diverse topics and different development trends. Using the AI Scholar dataset of 78K researchers in the field of AI, we identify several gender differences: (1) Although female researchers tend to have fewer overall citations than males, this citation difference does not hold for all academic-age groups; (2) There exist large gender homophily in co-authorship on AI papers; (3) Female first-authored papers show distinct linguistic styles, such as longer text, more positive emotion words, and more catchy titles than male first-authored papers. Our analysis provides a window into the current demographic trends in our AI community, and encourages more gender equality and diversity in the future. Our code and data are at https://github.com/causalNLP/ai-scholar-gender.
    

