# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference](https://arxiv.org/abs/2404.00242) | DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。 |
| [^2] | [A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention](https://arxiv.org/abs/2403.10173) | 提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。 |
| [^3] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |
| [^4] | [Learning Planning Action Models from State Traces](https://arxiv.org/abs/2402.10726) | 该论文探讨了一种从状态轨迹中学习规划行动模型的方法，尤其关注在参数未提供的情况下进行学习，并提出了两种不同级别的跟踪质量以及相应的算法。 |
| [^5] | [On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities](https://arxiv.org/abs/2402.10340) | 论文突出探讨了在机器人应用中整合大型语言模型和视觉语言模型所带来的安全性和健壮性关键问题，指出这种整合可能容易受到恶意攻击并导致严重后果。 |
| [^6] | [AI, Meet Human: Learning Paradigms for Hybrid Decision Making Systems](https://arxiv.org/abs/2402.06287) | 本调查提出了混合决策系统的分类方法，为理解如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。 |
| [^7] | [On the Completeness of Invariant Geometric Deep Learning Models](https://arxiv.org/abs/2402.04836) | 这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。 |
| [^8] | [On the $O(\frac{\sqrt{d}}{T^{1/4}})$ Convergence Rate of RMSProp and Its Momentum Extension Measured by $\ell_1$ Norm: Better Dependence on the Dimension](https://arxiv.org/abs/2402.00389) | 这项研究探讨了RMSProp及其动量扩展方法的收敛速度，并发现使用$\ell_1$范数测度时，收敛速度为$O(\frac{\sqrt{d}}{T^{1/4}})$，在维度极大的问题中具有改进依赖性。 |
| [^9] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^10] | [A match made in consistency heaven: when large language models meet evolutionary algorithms.](http://arxiv.org/abs/2401.10510) | 大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。 |
| [^11] | [Improving the Accuracy and Interpretability of Random Forests via Forest Pruning.](http://arxiv.org/abs/2401.05535) | 通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。 |
| [^12] | [Decidability of Querying First-Order Theories via Countermodels of Finite Width.](http://arxiv.org/abs/2304.06348) | 通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言 |
| [^13] | [Demystifying Misconceptions in Social Bots Research.](http://arxiv.org/abs/2303.17251) | 这篇文章揭示了关于社交机器人研究的普遍误解，强调需要以严谨、公正和负责任的方式讨论虚假信息研究。 |

# 详细

[^1]: DeFT：带IO意识的Flash Tree-attention用于高效的基于树搜索的LLM推断

    DeFT: Flash Tree-attention with IO-Awareness for Efficient Tree-search-based LLM Inference

    [https://arxiv.org/abs/2404.00242](https://arxiv.org/abs/2404.00242)

    DeFT提出了一种带IO意识的树注意力算法，通过在QKV准备和注意力计算阶段实现内存高效的计算，降低内存印记，以解决当前树解码策略和推断系统不适配的问题。

    

    使用树搜索进行解码可以极大地提高基于变压器的大型语言模型（LLMs）的推断质量。根据引导信号，它通过形成LLM输出从根到叶子的最佳路径来提高可控性、推理能力、对齐等。然而，由于计算冗余、内存占用和内存访问，当前的树解码策略及其推断系统互相不适配，导致推断效率低下。为解决这一问题，我们提出了DeFT，一种IO感知树注意力算法，它在两个阶段中保持内存高效的注意力计算，降低内存印记：（1）QKV准备：我们提出了一种KV引导树分裂策略，为GPU的高利用率和尽可能减少GPU全局内存和芯片上共享内存之间的KV缓存的内存读/写; （2）注意力计算...

    arXiv:2404.00242v1 Announce Type: cross  Abstract: Decoding using tree search can greatly enhance the inference quality for transformer-based Large Language Models (LLMs). Depending on the guidance signal, it searches for the best path from root to leaf in the tree by forming LLM outputs to improve controllability, reasoning ability, alignment, et cetera. However, current tree decoding strategies and their inference systems do not suit each other well due to redundancy in computation, memory footprints, and memory access, resulting in inefficient inference. To address this issue, we propose DeFT, an IO-aware tree attention algorithm that maintains memory-efficient attention calculation with low memory footprints in two stages: (1) QKV Preparation: we propose a KV-Guided Tree Split strategy to group QKV wisely for high utilization of GPUs and reduction of memory reads/writes for the KV cache between GPU global memory and on-chip shared memory as much as possible; (2) Attention Calculati
    
[^2]: 一种用于基于事件的对象检测的混合SNN-ANN网络，具有空间和时间注意力机制

    A Hybrid SNN-ANN Network for Event-based Object Detection with Spatial and Temporal Attention

    [https://arxiv.org/abs/2403.10173](https://arxiv.org/abs/2403.10173)

    提出了一种用于基于事件的对象检测的混合SNN-ANN网络，包括了新颖的基于注意力的桥接模块，能够有效捕捉稀疏的空间和时间关系，以提高任务性能。

    

    事件相机提供高时间分辨率和动态范围，几乎没有运动模糊，非常适合对象检测任务。尖峰神经网络（SNN）与事件驱动感知数据天生匹配，在神经形态硬件上能够实现超低功耗和低延迟推断，而人工神经网络（ANN）则展示出更稳定的训练动态和更快的收敛速度，从而具有更好的任务性能。混合SNN-ANN方法是一种有前途的替代方案，能够利用SNN和ANN体系结构的优势。在这项工作中，我们引入了第一个基于混合注意力的SNN-ANN骨干网络，用于使用事件相机进行对象检测。我们提出了一种新颖的基于注意力的SNN-ANN桥接模块，从SNN层中捕捉稀疏的空间和时间关系，并将其转换为密集特征图，供骨干网络的ANN部分使用。实验结果表明，我们提出的m

    arXiv:2403.10173v1 Announce Type: cross  Abstract: Event cameras offer high temporal resolution and dynamic range with minimal motion blur, making them promising for object detection tasks. While Spiking Neural Networks (SNNs) are a natural match for event-based sensory data and enable ultra-energy efficient and low latency inference on neuromorphic hardware, Artificial Neural Networks (ANNs) tend to display more stable training dynamics and faster convergence resulting in greater task performance. Hybrid SNN-ANN approaches are a promising alternative, enabling to leverage the strengths of both SNN and ANN architectures. In this work, we introduce the first Hybrid Attention-based SNN-ANN backbone for object detection using event cameras. We propose a novel Attention-based SNN-ANN bridge module to capture sparse spatial and temporal relations from the SNN layer and convert them into dense feature maps for the ANN part of the backbone. Experimental results demonstrate that our proposed m
    
[^3]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    
[^4]: 从状态轨迹中学习规划行动模型

    Learning Planning Action Models from State Traces

    [https://arxiv.org/abs/2402.10726](https://arxiv.org/abs/2402.10726)

    该论文探讨了一种从状态轨迹中学习规划行动模型的方法，尤其关注在参数未提供的情况下进行学习，并提出了两种不同级别的跟踪质量以及相应的算法。

    

    先前从状态轨迹中学习STRIPS领域模型的方法通常从要学习的行动的名称和参数开始。因此，它们的唯一任务是推断给定行动的前提条件和效应。在这项工作中，我们探讨了在学习时未提供学习行动的参数的情况。我们根据提供的信息定义了两个级别的跟踪质量，并提出了相应的算法。在一个级别(L1)中，轨迹中的状态被标记为行动名称，因此我们可以推断出行动的数量和名称，但我们仍需要弄清参数的数量和类型。在另一个级别(L2)中，状态还额外标记有构成相应基于对象的行动的参数。在这里，我们仍然需要推断学习行动中参数的类型。我们对所提出的算法进行了实验评估，并将其与其他方法进行了比较。

    arXiv:2402.10726v1 Announce Type: new  Abstract: Previous STRIPS domain model acquisition approaches that learn from state traces start with the names and parameters of the actions to be learned. Therefore their only task is to deduce the preconditions and effects of the given actions. In this work, we explore learning in situations when the parameters of learned actions are not provided. We define two levels of trace quality based on which information is provided and present an algorithm for each. In one level (L1), the states in the traces are labeled with action names, so we can deduce the number and names of the actions, but we still need to work out the number and types of parameters. In the other level (L2), the states are additionally labeled with objects that constitute the parameters of the corresponding grounded actions. Here we still need to deduce the types of the parameters in the learned actions. We experimentally evaluate the proposed algorithms and compare them with the
    
[^5]: 论部署LLMs/VLMs在机器人领域存在的安全问题：突显风险和漏洞

    On the Safety Concerns of Deploying LLMs/VLMs in Robotics: Highlighting the Risks and Vulnerabilities

    [https://arxiv.org/abs/2402.10340](https://arxiv.org/abs/2402.10340)

    论文突出探讨了在机器人应用中整合大型语言模型和视觉语言模型所带来的安全性和健壮性关键问题，指出这种整合可能容易受到恶意攻击并导致严重后果。

    

    在这篇论文中，我们着重讨论了将大型语言模型（LLMs）和视觉语言模型（VLMs）整合到机器人应用中所涉及的健壮性和安全性关键问题。最近的研究着重于利用LLMs和VLMs来提高机器人任务（如操作，导航等）的性能。然而，这种整合可能会引入显着的漏洞，即由于语言模型对恶意攻击的敏感性，可能导致灾难性后果。通过研究LLMs/VLMs与机器人界面的最新进展，我们展示了如何轻松操纵或误导机器人的行为，导致安全隐患。我们定义并提供了几种可能的恶意攻击示例，并对集成了语言模型的三个知名机器人框架（包括KnowNo VIMA和Instruct2Act）进行实验，以评估它们对这些攻击的敏感度。

    arXiv:2402.10340v1 Announce Type: cross  Abstract: In this paper, we highlight the critical issues of robustness and safety associated with integrating large language models (LLMs) and vision-language models (VLMs) into robotics applications. Recent works have focused on using LLMs and VLMs to improve the performance of robotics tasks, such as manipulation, navigation, etc. However, such integration can introduce significant vulnerabilities, in terms of their susceptibility to adversarial attacks due to the language models, potentially leading to catastrophic consequences. By examining recent works at the interface of LLMs/VLMs and robotics, we show that it is easy to manipulate or misguide the robot's actions, leading to safety hazards. We define and provide examples of several plausible adversarial attacks, and conduct experiments on three prominent robot frameworks integrated with a language model, including KnowNo VIMA, and Instruct2Act, to assess their susceptibility to these atta
    
[^6]: AI，与人相遇：混合决策系统的学习范式

    AI, Meet Human: Learning Paradigms for Hybrid Decision Making Systems

    [https://arxiv.org/abs/2402.06287](https://arxiv.org/abs/2402.06287)

    本调查提出了混合决策系统的分类方法，为理解如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。

    

    每天，我们越来越多地依赖机器学习模型来自动化和支持高风险任务和决策。这种日益增长的存在意味着人类现在不断与基于机器学习的系统进行互动，每天进行模型的培训和使用。计算机科学文献中有几种不同的技术来考虑人与机器学习系统的交互，但其分类稀疏且目标各异。本调查提出了混合决策系统的分类方法，为理解当前计算机科学文献如何对人与机器之间的交互进行建模提供了概念性和技术性的框架。

    Everyday we increasingly rely on machine learning models to automate and support high-stake tasks and decisions. This growing presence means that humans are now constantly interacting with machine learning-based systems, training and using models everyday. Several different techniques in computer science literature account for the human interaction with machine learning systems, but their classification is sparse and the goals varied. This survey proposes a taxonomy of Hybrid Decision Making Systems, providing both a conceptual and technical framework for understanding how current computer science literature models interaction between humans and machines.
    
[^7]: 关于不变几何深度学习模型的完备性

    On the Completeness of Invariant Geometric Deep Learning Models

    [https://arxiv.org/abs/2402.04836](https://arxiv.org/abs/2402.04836)

    这项研究集中于不变模型的理论表达能力，通过引入完备的设计GeoNGNN，并利用其作为理论工具，首次证明了E(3)-完备性。

    

    不变模型是一类重要的几何深度学习模型，通过利用信息丰富的几何特征生成有意义的几何表示。这些模型以其简单性、良好的实验结果和计算效率而闻名。然而，它们的理论表达能力仍然不清楚，限制了对这种模型潜力的深入理解。在这项工作中，我们集中讨论不变模型的理论表达能力。我们首先严格限制了最经典的不变模型Vanilla DisGNN（结合距离的消息传递神经网络）的表达能力，将其不可识别的情况仅限于高度对称的几何图形。为了打破这些特殊情况的对称性，我们引入了一个简单而完备的不变设计，即嵌套Vanilla DisGNN的GeoNGNN。利用GeoNGNN作为理论工具，我们首次证明了E(3)-完备性。

    Invariant models, one important class of geometric deep learning models, are capable of generating meaningful geometric representations by leveraging informative geometric features. These models are characterized by their simplicity, good experimental results and computational efficiency. However, their theoretical expressive power still remains unclear, restricting a deeper understanding of the potential of such models. In this work, we concentrate on characterizing the theoretical expressiveness of invariant models. We first rigorously bound the expressiveness of the most classical invariant model, Vanilla DisGNN (message passing neural networks incorporating distance), restricting its unidentifiable cases to be only those highly symmetric geometric graphs. To break these corner cases' symmetry, we introduce a simple yet E(3)-complete invariant design by nesting Vanilla DisGNN, named GeoNGNN. Leveraging GeoNGNN as a theoretical tool, we for the first time prove the E(3)-completeness 
    
[^8]: 关于RMSProp及其动量扩展方法的$O(\frac{\sqrt{d}}{T^{1/4}})$收敛速度和对维度的改进依赖性

    On the $O(\frac{\sqrt{d}}{T^{1/4}})$ Convergence Rate of RMSProp and Its Momentum Extension Measured by $\ell_1$ Norm: Better Dependence on the Dimension

    [https://arxiv.org/abs/2402.00389](https://arxiv.org/abs/2402.00389)

    这项研究探讨了RMSProp及其动量扩展方法的收敛速度，并发现使用$\ell_1$范数测度时，收敛速度为$O(\frac{\sqrt{d}}{T^{1/4}})$，在维度极大的问题中具有改进依赖性。

    

    尽管自适应梯度方法在深度学习中被广泛使用，但其收敛速度尚未得到彻底研究，特别是对于其对维度的依赖性。本文考虑了经典的RMSProp及其动量扩展方法，并通过$\ell_1$范数建立了收敛率$\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_1\right]\leq O(\frac{\sqrt{d}}{T^{1/4}})$，无需假设梯度有界，其中$d$是优化变量的维度，$T$是迭代次数。由于对于维度极大的问题，$\|x\|_2\ll\|x\|_1\leq\sqrt{d}\|x\|_2$，因此我们的收敛速度可以类比为SGD的$\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_2\right]\leq O(\frac{1}{T^{1/4}})$，测度为$\ell_1$范数。

    Although adaptive gradient methods have been extensively used in deep learning, their convergence rates have not been thoroughly studied, particularly with respect to their dependence on the dimension. This paper considers the classical RMSProp and its momentum extension and establishes the convergence rate of $\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_1\right]\leq O(\frac{\sqrt{d}}{T^{1/4}})$ measured by $\ell_1$ norm without the bounded gradient assumption, where $d$ is the dimension of the optimization variable and $T$ is the iteration number. Since $\|x\|_2\ll\|x\|_1\leq\sqrt{d}\|x\|_2$ for problems with extremely large $d$, our convergence rate can be considered to be analogous to the $\frac{1}{T}\sum_{k=1}^TE\left[\|\nabla f(x^k)\|_2\right]\leq O(\frac{1}{T^{1/4}})$ one of SGD measured by $\ell_1$ norm.
    
[^9]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^10]: 天作之合：大型语言模型与进化算法的结合

    A match made in consistency heaven: when large language models meet evolutionary algorithms. (arXiv:2401.10510v1 [cs.NE])

    [http://arxiv.org/abs/2401.10510](http://arxiv.org/abs/2401.10510)

    大型语言模型和进化算法的结合具有强大的一致性，包括标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化等多个核心特征。本文分析了现有的耦合研究，并为未来的研究提供了基本路线和关键挑战。

    

    预训练的大型语言模型（LLMs）在生成创造性的自然文本方面具有强大的能力。进化算法（EAs）可以发现复杂实际问题的多样解决方案。本文通过比较文本序列生成和进化的共同特点和方向性，阐述了LLMs与EAs之间的强大一致性，包括多个一对一的核心特征：标记嵌入和基因型-表现型映射、位置编码和适应性塑造、位置嵌入和选择、注意力和交叉、前馈神经网络和突变、模型训练和参数更新以及多任务学习和多目标优化。在这种一致性视角下，分析了现有的耦合研究，包括进化微调和LLM增强型EAs。借助这些洞见，我们概述了未来在LLMs和EAs耦合方面的基本研究路线，并突出了其中的关键挑战。

    Pre-trained large language models (LLMs) have powerful capabilities for generating creative natural text. Evolutionary algorithms (EAs) can discover diverse solutions to complex real-world problems. Motivated by the common collective and directionality of text sequence generation and evolution, this paper illustrates the strong consistency of LLMs and EAs, which includes multiple one-to-one key characteristics: token embedding and genotype-phenotype mapping, position encoding and fitness shaping, position embedding and selection, attention and crossover, feed-forward neural network and mutation, model training and parameter update, and multi-task learning and multi-objective optimization. Based on this consistency perspective, existing coupling studies are analyzed, including evolutionary fine-tuning and LLM-enhanced EAs. Leveraging these insights, we outline a fundamental roadmap for future research in coupling LLMs and EAs, while highlighting key challenges along the way. The consist
    
[^11]: 通过森林修剪提高随机森林的准确性和可解释性

    Improving the Accuracy and Interpretability of Random Forests via Forest Pruning. (arXiv:2401.05535v1 [stat.ML])

    [http://arxiv.org/abs/2401.05535](http://arxiv.org/abs/2401.05535)

    通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。

    

    接近几十年的发展之后，随机森林仍然在各种学习问题中提供最先进的准确性，在这方面超越了决策树甚至神经网络等替代机器学习算法。然而，作为一种集成方法，随机森林在解释性方面往往比决策树表现不佳。在本研究中，我们提出了一种事后方法，旨在兼顾随机森林的准确性和决策树的可解释性。为此，我们提出了两种森林修剪方法，以在给定的随机森林内找到最佳子森林，然后在适用的情况下将选定的树合并为一棵。我们的第一种方法依赖于约束穷举搜索，而第二种方法基于LASSO方法的改进。在合成和真实世界数据集上进行的大量实验证明，在大多数情景下，这两种方法中至少有一种能够显著提高随机森林的准确性和可解释性。

    Decades after their inception, random forests continue to provide state-of-the-art accuracy in a variety of learning problems, outperforming in this respect alternative machine learning algorithms such as decision trees or even neural networks. However, being an ensemble method, the one aspect where random forests tend to severely underperform decision trees is interpretability. In the present work, we propose a post-hoc approach that aims to have the best of both worlds: the accuracy of random forests and the interpretability of decision trees. To this end, we present two forest-pruning methods to find an optimal sub-forest within a given random forest, and then, when applicable, combine the selected trees into one. Our first method relies on constrained exhaustive search, while our second method is based on an adaptation of the LASSO methodology. Extensive experiments over synthetic and real world datasets show that, in the majority of scenarios, at least one of the two methods propo
    
[^12]: 通过有限宽度的反模型查询一阶理论的可决定性

    Decidability of Querying First-Order Theories via Countermodels of Finite Width. (arXiv:2304.06348v1 [cs.LO])

    [http://arxiv.org/abs/2304.06348](http://arxiv.org/abs/2304.06348)

    通过有限宽度的反模型查询一阶理论的可决定性并提出分割宽度，使其能够捕获实际相关的查询语言

    

    我们提出了一个通用框架，基于具有结构简单的反模型的存在性（通过某些类型的宽度量来衡量，包括树宽和团宽等），为广泛的逻辑蕴含问题（简称查询）的可决定性提供了支持。作为我们框架的一个重要特例，我们确定了展现出宽度有限有限通用模型集的逻辑，保证了各种同态封闭查询的可决定性，包括了各种实际相关的查询语言。作为一个特别强大的宽度量，我们提出了Blumensath的分割宽度，该量包含了各种通常考虑的宽度量，具有非常有利的计算和结构特性。针对普遍展现存在性规则为一个展示案例，我们解释了有限分割宽度规则集包含其他已知的抽象可决定类，但借助现有的分层和受控规则集概念，也使我们能够捕获实际相关的查询语言，例如正则，连接和布尔连接查询。我们以存在规则的形式为重点，补充我们的理论结果，并进行了彻底的实验评估，展示了我们的框架在各种高级知识处理场景中的实际适用性和可伸缩性。

    We propose a generic framework for establishing the decidability of a wide range of logical entailment problems (briefly called querying), based on the existence of countermodels that are structurally simple, gauged by certain types of width measures (with treewidth and cliquewidth as popular examples). As an important special case of our framework, we identify logics exhibiting width-finite finitely universal model sets, warranting decidable entailment for a wide range of homomorphism-closed queries, subsuming a diverse set of practically relevant query languages. As a particularly powerful width measure, we propose Blumensath's partitionwidth, which subsumes various other commonly considered width measures and exhibits highly favorable computational and structural properties. Focusing on the formalism of existential rules as a popular showcase, we explain how finite partitionwidth sets of rules subsume other known abstract decidable classes but -- leveraging existing notions of strat
    
[^13]: 揭开对社交机器人研究的误解

    Demystifying Misconceptions in Social Bots Research. (arXiv:2303.17251v1 [cs.SI])

    [http://arxiv.org/abs/2303.17251](http://arxiv.org/abs/2303.17251)

    这篇文章揭示了关于社交机器人研究的普遍误解，强调需要以严谨、公正和负责任的方式讨论虚假信息研究。

    

    社交机器人科学寻求解决网络虚假信息最受争议的形式之一的知识和解决方案。然而，社交机器人研究受到普遍的偏见、夸大的结果和误解的困扰，这些都为歧义、不切实际的期望和看似无法调和的发现打下了基础。克服这些问题对于确保可靠的解决方案和重申科学方法的有效性至关重要。在这篇文章中，我们修订了社交机器人研究中的一些最新结果，强调和纠正了事实错误以及方法论和概念问题。更重要的是，我们揭开了普遍的误解，解决了有关如何讨论社交机器人研究的基本问题。我们的分析揭示了以严谨、公正和负责任的方式讨论虚假信息研究的必要性。本文通过确定并驳斥社交机器人研究的支持者和反对者常用的谬误论证，支持这种努力。

    The science of social bots seeks knowledge and solutions to one of the most debated forms of online misinformation. Yet, social bots research is plagued by widespread biases, hyped results, and misconceptions that set the stage for ambiguities, unrealistic expectations, and seemingly irreconcilable findings. Overcoming such issues is instrumental towards ensuring reliable solutions and reaffirming the validity of the scientific method. In this contribution we revise some recent results in social bots research, highlighting and correcting factual errors as well as methodological and conceptual issues. More importantly, we demystify common misconceptions, addressing fundamental points on how social bots research is discussed. Our analysis surfaces the need to discuss misinformation research in a rigorous, unbiased, and responsible way. This article bolsters such effort by identifying and refuting common fallacious arguments used by both proponents and opponents of social bots research as
    

