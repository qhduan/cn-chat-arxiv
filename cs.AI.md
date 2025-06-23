# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ChatDBG: An AI-Powered Debugging Assistant](https://arxiv.org/abs/2403.16354) | ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。 |
| [^2] | [A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention](https://arxiv.org/abs/2403.10173) | 提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。 |
| [^3] | [ALTO: An Efficient Network Orchestrator for Compound AI Systems](https://arxiv.org/abs/2403.04311) | ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。 |
| [^4] | [AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability](https://arxiv.org/abs/2402.09404) | AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。 |
| [^5] | [ChatGPT vs LLaMA: Impact, Reliability, and Challenges in Stack Overflow Discussions](https://arxiv.org/abs/2402.08801) | 这篇论文通过对Stack Overflow问题的分析，研究了ChatGPT和LLaMA对于该平台的影响和可靠性，以及它们在长期内取代Stack Overflow的挑战。研究结果表明，LLMs在某些方面失败，并提供了对比LLMs的实证比较。 |
| [^6] | [Open-Set Graph Anomaly Detection via Normal Structure Regularisation](https://arxiv.org/abs/2311.06835) | 通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力 |
| [^7] | [Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning.](http://arxiv.org/abs/2401.03756) | 该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。 |
| [^8] | [Layer-wise Feedback Propagation.](http://arxiv.org/abs/2308.12053) | 本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。 |
| [^9] | [Voices of Her: Analyzing Gender Differences in the AI Publication World.](http://arxiv.org/abs/2305.14597) | 通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。 |

# 详细

[^1]: ChatDBG: 一种基于人工智能的调试助手

    ChatDBG: An AI-Powered Debugging Assistant

    [https://arxiv.org/abs/2403.16354](https://arxiv.org/abs/2403.16354)

    ChatDBG是第一个AI-Powered调试助手，通过将大型语言模型集成到传统调试器中，实现了程序员与调试器之间的协作对话，能够处理复杂问题、执行根本原因分析，并探索开放性查询。

    

    本文介绍了ChatDBG，这是第一个基于人工智能的调试助手。ChatDBG集成了大型语言模型(LLMs)，显著增强了传统调试器的功能和用户友好性。ChatDBG允许程序员与调试器进行协作对话，使他们能够提出关于程序状态的复杂问题，对崩溃或断言失败进行根本原因分析，并探索诸如“为什么x为空？”之类的开放性查询。为了处理这些查询，ChatDBG授予LLM自主权，通过发出命令来浏览堆栈和检查程序状态进行调试；然后报告其发现并将控制权交还给程序员。我们的ChatDBG原型与标准调试器集成，包括LLDB、GDB和WinDBG用于本地代码以及用于Python的Pdb。我们在各种代码集合上进行了评估，包括具有已知错误的C/C++代码和一套Python代码。

    arXiv:2403.16354v1 Announce Type: cross  Abstract: This paper presents ChatDBG, the first AI-powered debugging assistant. ChatDBG integrates large language models (LLMs) to significantly enhance the capabilities and user-friendliness of conventional debuggers. ChatDBG lets programmers engage in a collaborative dialogue with the debugger, allowing them to pose complex questions about program state, perform root cause analysis for crashes or assertion failures, and explore open-ended queries like "why is x null?". To handle these queries, ChatDBG grants the LLM autonomy to take the wheel and drive debugging by issuing commands to navigate through stacks and inspect program state; it then reports its findings and yields back control to the programmer. Our ChatDBG prototype integrates with standard debuggers including LLDB, GDB, and WinDBG for native code and Pdb for Python. Our evaluation across a diverse set of code, including C/C++ code with known bugs and a suite of Python code includi
    
[^2]: 一种用于基于事件的对象检测的混合SNN-ANN网络，具有空间和时间注意力机制

    A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention

    [https://arxiv.org/abs/2403.10173](https://arxiv.org/abs/2403.10173)

    提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。

    

    事件相机提供高时间分辨率和动态范围，几乎没有运动模糊，非常适合对象检测任务。尖峰神经网络（SNN）与事件驱动感知数据天生匹配，在神经形态硬件上能够实现超低功耗和低延迟推断，而人工神经网络（ANN）则展示出更稳定的训练动态和更快的收敛速度，从而具有更好的任务性能。混合SNN-ANN方法是一种有前途的替代方案，能够利用SNN和ANN体系结构的优势。在这项工作中，我们引入了第一个基于混合注意力的SNN-ANN骨干网络，用于使用事件相机进行对象检测。我们提出了一种新颖的基于注意力的SNN-ANN桥接模块，从SNN层中捕捉稀疏的空间和时间关系，并将其转换为密集特征图，供骨干网络的ANN部分使用。实验结果表明，我们提出的m

    arXiv:2403.10173v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and dynamic range with minimal motion blur, making them promising for object detection tasks. While Spiking Neural Networks (SNNs) are a natural match for event-based sensory data and enable ultra-energy efficient and low latency inference on neuromorphic hardware, Artificial Neural Networks (ANNs) tend to display more stable training dynamics and faster convergence resulting in greater task performance. Hybrid SNN-ANN approaches are a promising alternative, enabling to leverage the strengths of both SNN and ANN architectures. In this work, we introduce the first Hybrid Attention-based SNN-ANN backbone for object detection using event cameras. We propose a novel Attention-based SNN-ANN bridge module to capture sparse spatial and temporal relations from the SNN layer and convert them into dense feature maps for the ANN part of the backbone. Experimental results demonstrate that our proposed m
    
[^3]: ALTO：一种用于复合AI系统的高效网络编排器

    ALTO: An Efficient Network Orchestrator for Compound AI Systems

    [https://arxiv.org/abs/2403.04311](https://arxiv.org/abs/2403.04311)

    ALTO是一个网络编排器，针对生成语言模型的优化机会，实现了高吞吐量和低延迟，同时解决了流式中间输出的两个新挑战：正确性和负载平衡。

    

    我们提出了ALTO，一种用于有效为诸如语言模型管道之类的复合AI系统提供服务的网络编排器。ALTO通过利用生成语言模型特有的优化机会：流式中间输出，实现了高吞吐量和低延迟。由于语言模型逐个生成token的输出，ALTO在可能时暴露了在阶段之间流式传输中间输出的机会。我们强调了在跨分布式管道阶段实例之间流式传输中间数据时出现的两个新挑战：正确性和负载平衡。我们还提出了聚合感知路由接口和分布式提示感知调度以应对这些挑战的需求。我们在一个复杂的聊天机器人验证管道上展示了ALTO部分输出流式传输的影响，将吞吐量提高了最多3倍，同时将固定延迟目标设置为4秒/请求，还减少了尾延迟。

    arXiv:2403.04311v1 Announce Type: new  Abstract: We present ALTO, a network orchestrator for efficiently serving compound AI systems such as pipelines of language models. ALTO achieves high throughput and low latency by taking advantage of an optimization opportunity specific to generative language models: streaming intermediate outputs. As language models produce outputs token by token, ALTO exposes opportunities to stream intermediate outputs between stages when possible. We highlight two new challenges of correctness and load balancing which emerge when streaming intermediate data across distributed pipeline stage instances. We also motivate the need for an aggregation-aware routing interface and distributed prompt-aware scheduling to address these challenges. We demonstrate the impact of ALTO's partial output streaming on a complex chatbot verification pipeline, increasing throughput by up to 3x for a fixed latency target of 4 seconds / request while also reducing tail latency by 1
    
[^4]: AQA-Bench：评估LLM顺序推理能力的交互式基准测试

    AQA-Bench: An Interactive Benchmark for Evaluating LLMs' Sequential Reasoning Ability

    [https://arxiv.org/abs/2402.09404](https://arxiv.org/abs/2402.09404)

    AQA-Bench是一个交互式基准测试，用于评估大型语言模型在算法上下文中的顺序推理能力。研究发现闭源模型表现出更强的顺序推理能力，显著优于开源模型。

    

    该论文介绍了一种新的基准测试AQA-Bench，用于评估大型语言模型（LLMs）在算法上下文中，如深度优先搜索（DFS）等的顺序推理能力。我们评估基准测试的关键特点在于其交互式评估协议-例如，在DFS中，每个节点的可用连接边取决于模型对该节点的遍历，因此需要LLM有效地记住已访问节点并策划后续移动的能力。我们使用三种不同的算法构建了AQA-Bench，分别是二分搜索，深度优先搜索和广度优先搜索，并评估了12种不同的LLMs的顺序推理能力。我们的调查揭示了一些有趣的发现：（1）类似GPT-4和Gemini等闭源模型通常显示出强大的顺序推理能力，明显优于开源LLMs。（2）天真地提供互操作性

    arXiv:2402.09404v1 Announce Type: cross Abstract: This paper introduces AQA-Bench, a novel benchmark to assess the sequential reasoning capabilities of large language models (LLMs) in algorithmic contexts, such as depth-first search (DFS). The key feature of our evaluation benchmark lies in its interactive evaluation protocol -- for example, in DFS, the availability of each node's connected edge is contingent upon the model's traversal to that node, thereby necessitating the LLM's ability to effectively remember visited nodes and strategize subsequent moves. We comprehensively build AQA-Bench with three different algorithms, namely binary search, depth-first search, and breadth-first search, and to evaluate the sequential reasoning ability of 12 different LLMs. Our investigations reveal several interesting findings: (1) Closed-source models like GPT-4 and Gemini generally show strong sequential reasoning ability, significantly outperforming open-source LLMs. (2) Naively providing inter
    
[^5]: ChatGPT与LLaMA在Stack Overflow讨论中的影响，可靠性和挑战

    ChatGPT vs LLaMA: Impact, Reliability, and Challenges in Stack Overflow Discussions

    [https://arxiv.org/abs/2402.08801](https://arxiv.org/abs/2402.08801)

    这篇论文通过对Stack Overflow问题的分析，研究了ChatGPT和LLaMA对于该平台的影响和可靠性，以及它们在长期内取代Stack Overflow的挑战。研究结果表明，LLMs在某些方面失败，并提供了对比LLMs的实证比较。

    

    自2022年11月发布以来，ChatGPT已经在Stack Overflow上引起轰动，这是开发人员关于编程和软件开发问题的首选平台。ChatGPT展示了生成即时、人类般回答技术问题的能力，引起了开发者社区对人工智能生成时代下人类驱动平台演变角色的辩论。ChatGPT发布两个月后，Meta发布了它自己的大型语言模型(LLM) LLaMA。为了 (i) 测量用户对Stack Overflow的时间演进下的参与程度；(ii) 量化LLMs回答的可靠性及其在长期内取代Stack Overflow的潜力；(iii) 确定和理解LLMs失败的原因；(iv) 对比LLMs。我们进行了实证研究，分析Stack Overflow上的问题，并使用这些LLMs来回答问题。

    arXiv:2402.08801v1 Announce Type: cross Abstract: Since its release in November 2022, ChatGPT has shaken up Stack Overflow, the premier platform for developers' queries on programming and software development. Demonstrating an ability to generate instant, human-like responses to technical questions, ChatGPT has ignited debates within the developer community about the evolving role of human-driven platforms in the age of generative AI. Two months after ChatGPT's release, Meta released its answer with its own Large Language Model (LLM) called LLaMA: the race was on. We conducted an empirical study analyzing questions from Stack Overflow and using these LLMs to address them. This way, we aim to (ii) measure user engagement evolution with Stack Overflow over time; (ii) quantify the reliability of LLMs' answers and their potential to replace Stack Overflow in the long term; (iii) identify and understand why LLMs fails; and (iv) compare LLMs together. Our empirical results are unequivocal: C
    
[^6]: 通过正常结构规范化实现开放图异常检测

    Open-Set Graph Anomaly Detection via Normal Structure Regularisation

    [https://arxiv.org/abs/2311.06835](https://arxiv.org/abs/2311.06835)

    通过正常结构规范化方法，实现开放图异常检测模型对未知异常的广义检测能力

    

    本文考虑了一个重要的图异常检测（GAD）任务，即开放式GAD，旨在使用少量标记的训练正常节点和异常节点（称为已知异常）来检测异常节点，这些节点无法展示所有可能的推理时异常。已标记数据的可用性为GAD模型提供了关键的异常先验知识，可大大降低检测错误。然而，当前方法往往过分强调拟合已知异常，导致对未知异常（即未被标记的异常节点）的弱泛化能力。此外，它们被引入以处理欧几里德数据，未能有效捕捉GAD的重要非欧几里德特征。在这项工作中，我们提出了一种新颖的开放式GAD方法，即正常结构规范化（NSReg），以实现对未知异常的广义检测能力。

    arXiv:2311.06835v2 Announce Type: replace-cross  Abstract: This paper considers an important Graph Anomaly Detection (GAD) task, namely open-set GAD, which aims to detect anomalous nodes using a small number of labelled training normal and anomaly nodes (known as seen anomalies) that cannot illustrate all possible inference-time abnormalities. The availability of that labelled data provides crucial prior knowledge about abnormalities for GAD models, enabling substantially reduced detection errors. However, current methods tend to over-emphasise fitting the seen anomalies, leading to a weak generalisation ability to detect unseen anomalies, i.e., those that are not illustrated by the labelled anomaly nodes. Further, they were introduced to handle Euclidean data, failing to effectively capture important non-Euclidean features for GAD. In this work, we propose a novel open-set GAD approach, namely Normal Structure Regularisation (NSReg), to achieve generalised detection ability to unseen 
    
[^7]: 上下文固定预算的最佳臂识别：适应性实验设计与策略学习

    Contextual Fixed-Budget Best Arm Identification: Adaptive Experimental Design with Policy Learning. (arXiv:2401.03756v1 [cs.LG])

    [http://arxiv.org/abs/2401.03756](http://arxiv.org/abs/2401.03756)

    该论文研究了个性化治疗推荐的问题，提出了一个上下文固定预算的最佳臂识别模型，通过自适应实验设计和策略学习来推荐最佳治疗方案，并通过最坏情况下的期望简单遗憾来衡量推荐的有效性。

    

    个性化治疗推荐是基于证据的决策中的关键任务。在这项研究中，我们将这个任务作为一个带有上下文信息的固定预算最佳臂识别（Best Arm Identification, BAI）问题来进行建模。在这个设置中，我们考虑了一个给定多个治疗臂的自适应试验。在每一轮中，决策者观察一个刻画实验单位的上下文（协变量），并将该单位分配给其中一个治疗臂。在实验结束时，决策者推荐一个在给定上下文条件下预计产生最高期望结果的治疗臂（最佳治疗臂）。该决策的有效性通过最坏情况下的期望简单遗憾（策略遗憾）来衡量，该遗憾表示在给定上下文条件下，最佳治疗臂和推荐治疗臂的条件期望结果之间的最大差异。我们的初始步骤是推导最坏情况下期望简单遗憾的渐近下界，该下界还暗示着解决该问题的一些思路。

    Individualized treatment recommendation is a crucial task in evidence-based decision-making. In this study, we formulate this task as a fixed-budget best arm identification (BAI) problem with contextual information. In this setting, we consider an adaptive experiment given multiple treatment arms. At each round, a decision-maker observes a context (covariate) that characterizes an experimental unit and assigns the unit to one of the treatment arms. At the end of the experiment, the decision-maker recommends a treatment arm estimated to yield the highest expected outcome conditioned on a context (best treatment arm). The effectiveness of this decision is measured in terms of the worst-case expected simple regret (policy regret), which represents the largest difference between the conditional expected outcomes of the best and recommended treatment arms given a context. Our initial step is to derive asymptotic lower bounds for the worst-case expected simple regret, which also implies idea
    
[^8]: 层级反馈传播

    Layer-wise Feedback Propagation. (arXiv:2308.12053v1 [cs.LG])

    [http://arxiv.org/abs/2308.12053](http://arxiv.org/abs/2308.12053)

    本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。

    

    本文提出了一种称为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，该方法利用可解释性，具体而言是层级相关性传播（LRP），根据每个连接对解决给定任务的贡献独立分配奖励。这与传统的梯度下降方法不同，梯度下降方法是朝向估计的损失最小值更新参数。LFP在模型中传播奖励信号，而无需梯度计算。它增强接收到正反馈的结构，同时降低接收到负反馈的结构的影响。我们从理论和实证的角度证明了LFP的收敛性，并展示了它在各种模型和数据集上实现与梯度下降相当的性能。值得注意的是，LFP克服了梯度方法的某些局限性，例如对有意义的导数的依赖。我们进一步研究了LFP如何解决梯度方法相关问题的限制。

    In this paper, we present Layer-wise Feedback Propagation (LFP), a novel training approach for neural-network-like predictors that utilizes explainability, specifically Layer-wise Relevance Propagation(LRP), to assign rewards to individual connections based on their respective contributions to solving a given task. This differs from traditional gradient descent, which updates parameters towards anestimated loss minimum. LFP distributes a reward signal throughout the model without the need for gradient computations. It then strengthens structures that receive positive feedback while reducingthe influence of structures that receive negative feedback. We establish the convergence of LFP theoretically and empirically, and demonstrate its effectiveness in achieving comparable performance to gradient descent on various models and datasets. Notably, LFP overcomes certain limitations associated with gradient-based methods, such as reliance on meaningful derivatives. We further investigate how 
    
[^9]: 她们的声音：分析人工智能出版领域的性别差异

    Voices of Her: Analyzing Gender Differences in the AI Publication World. (arXiv:2305.14597v1 [cs.CL])

    [http://arxiv.org/abs/2305.14597](http://arxiv.org/abs/2305.14597)

    通过对AI学术界的78K研究人员的分析，研究发现女性第一作者的论文具有不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题；在AI论文的合著中存在很大的性别同质性。我们鼓励未来实现更多的性别平等和多样性。

    

    虽然先前的研究已经分析了学术界中的性别偏见，但是我们仍然缺乏一个全面的人工智能社区性别差异的分析，涵盖各种主题和不同的发展趋势。我们使用AI Scholar数据集中的78K位AI领域的研究人员，发现了一些性别差异：（1）虽然女性研究人员的总引用次数比男性少，但这种引用差异并不适用于所有学术年龄组；（2）在AI论文的合著中存在很大的性别同质性；（3）女性第一作者的论文显示出不同的语言风格，例如更长的文本、更多的正面情感词汇和更引人注目的标题。我们的分析为我们的AI社区现有的人口统计趋势提供了一个窗口，并鼓励在未来实现更多的性别平等和多样性。我们的代码和数据可在https://github.com/causalNLP/ai-scholar-gender找到。

    While several previous studies have analyzed gender bias in research, we are still missing a comprehensive analysis of gender differences in the AI community, covering diverse topics and different development trends. Using the AI Scholar dataset of 78K researchers in the field of AI, we identify several gender differences: (1) Although female researchers tend to have fewer overall citations than males, this citation difference does not hold for all academic-age groups; (2) There exist large gender homophily in co-authorship on AI papers; (3) Female first-authored papers show distinct linguistic styles, such as longer text, more positive emotion words, and more catchy titles than male first-authored papers. Our analysis provides a window into the current demographic trends in our AI community, and encourages more gender equality and diversity in the future. Our code and data are at https://github.com/causalNLP/ai-scholar-gender.
    

