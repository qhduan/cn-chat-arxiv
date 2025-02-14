# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Attention is Naturally Sparse with Gaussian Distributed Input](https://arxiv.org/abs/2404.02690) | 通过对高斯输入下注意力得分稀疏性进行理论分析，揭示了注意力机制中稀疏性的特征及其对计算效率的影响。 |
| [^2] | [The Journey to Trustworthy AI- Part 1: Pursuit of Pragmatic Frameworks](https://arxiv.org/abs/2403.15457) | 本文回顾了值得信赖的人工智能（TAI）和其各种定义，主张不应将“负责任的”或“道德的”人工智能等术语视为TAI的替代，而是提倡以公平性、偏见、风险、安全性、可解释性和可靠性等关键属性为中心的方法，认识到地缘政治和地理原因导致的人工智能监管差异对跨国公司构成挑战。 |
| [^3] | [Personalized Negative Reservoir for Incremental Learning in Recommender Systems](https://arxiv.org/abs/2403.03993) | 推荐系统中的个性化负采样技术在增量学习中的应用，解决了更新推荐系统模型时遇到的遗忘灾难问题。 |
| [^4] | [Evaluating the Performance of ChatGPT for Spam Email Detection](https://arxiv.org/abs/2402.15537) | 该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。 |
| [^5] | [Do Large Code Models Understand Programming Concepts? A Black-box Approach](https://arxiv.org/abs/2402.05980) | 本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。 |
| [^6] | [Deep Learning for Multivariate Time Series Imputation: A Survey](https://arxiv.org/abs/2402.04059) | 本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。 |
| [^7] | [Algorithmic Persuasion Through Simulation](https://arxiv.org/abs/2311.18138) | 通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。 |
| [^8] | [A Stability Principle for Learning under Non-Stationarity.](http://arxiv.org/abs/2310.18304) | 本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。 |
| [^9] | [A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments.](http://arxiv.org/abs/2309.17170) | 提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。 |
| [^10] | [Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs.](http://arxiv.org/abs/2308.11914) | 通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。 |
| [^11] | [Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information.](http://arxiv.org/abs/2210.00116) | 本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。 |

# 详细

[^1]: 注意力机制在高斯分布输入下自然稀疏

    Attention is Naturally Sparse with Gaussian Distributed Input

    [https://arxiv.org/abs/2404.02690](https://arxiv.org/abs/2404.02690)

    通过对高斯输入下注意力得分稀疏性进行理论分析，揭示了注意力机制中稀疏性的特征及其对计算效率的影响。

    

    大型语言模型（LLMs）的计算强度是关键瓶颈，主要是由于transformer架构中注意力机制的$O(n^2)$复杂度。稀疏注意力作为一个关键创新应运而生，旨在减少计算负荷同时保持模型性能。本研究对LLMs内的注意力分数稀疏性进行了严格的理论分析，特别是在高斯输入框架下。通过建立一组基础假设并采用一种系统的理论方法，我们揭示了注意力分数稀疏性的内在特征及其对计算效率的影响。我们的主要贡献在于提供了对注意力机制中稀疏性表现形式的详细理论检查，揭示了在计算节约和模型有效性之间潜在权衡的见解。

    arXiv:2404.02690v1 Announce Type: cross  Abstract: The computational intensity of Large Language Models (LLMs) is a critical bottleneck, primarily due to the $O(n^2)$ complexity of the attention mechanism in transformer architectures. Addressing this, sparse attention emerges as a key innovation, aiming to reduce computational load while maintaining model performance. This study presents a rigorous theoretical analysis of the sparsity in attention scores within LLMs, particularly under the framework of Gaussian inputs. By establishing a set of foundational assumptions and employing a methodical theoretical approach, we unravel the intrinsic characteristics of attention score sparsity and its implications on computational efficiency. Our main contribution lies in providing a detailed theoretical examination of how sparsity manifests in attention mechanisms, offering insights into the potential trade-offs between computational savings and model effectiveness. This work not only advances 
    
[^2]: 走向值得信赖的人工智能之旅-第一部分：追求务实框架

    The Journey to Trustworthy AI- Part 1: Pursuit of Pragmatic Frameworks

    [https://arxiv.org/abs/2403.15457](https://arxiv.org/abs/2403.15457)

    本文回顾了值得信赖的人工智能（TAI）和其各种定义，主张不应将“负责任的”或“道德的”人工智能等术语视为TAI的替代，而是提倡以公平性、偏见、风险、安全性、可解释性和可靠性等关键属性为中心的方法，认识到地缘政治和地理原因导致的人工智能监管差异对跨国公司构成挑战。

    

    本文回顾了值得信赖的人工智能（TAI）及其各种定义。考虑到任何社会中尊重的原则，TAI通常被一些属性所特征，其中一些属性已导致监管或工程背景下的混淆。我们反对使用诸如“负责任的”或“道德的”人工智能等术语来替代TAI。为了帮助澄清任何混乱，我们建议将它们抛在脑后。鉴于TAI固有的主观性和复杂性，开发一个通用框架被认为是不可行的。相反，我们主张采取以公平性、偏见、风险、安全性、可解释性和可靠性等关键属性为中心的方法。我们审视了正在进行的监管环境，重点关注欧盟、中国和美国的倡议。我们认识到，基于地缘政治和地理原因而不同的人工智能监管对跨国公司构成额外挑战。

    arXiv:2403.15457v1 Announce Type: cross  Abstract: This paper reviews Trustworthy Artificial Intelligence (TAI) and its various definitions. Considering the principles respected in any society, TAI is often characterized by a few attributes, some of which have led to confusion in regulatory or engineering contexts. We argue against using terms such as Responsible or Ethical AI as substitutes for TAI. And to help clarify any confusion, we suggest leaving them behind. Given the subjectivity and complexity inherent in TAI, developing a universal framework is deemed infeasible. Instead, we advocate for approaches centered on addressing key attributes and properties such as fairness, bias, risk, security, explainability, and reliability. We examine the ongoing regulatory landscape, with a focus on initiatives in the EU, China, and the USA. We recognize that differences in AI regulations based on geopolitical and geographical reasons pose an additional challenge for multinational companies. 
    
[^3]: 个性化负采样在推荐系统增量学习中的应用

    Personalized Negative Reservoir for Incremental Learning in Recommender Systems

    [https://arxiv.org/abs/2403.03993](https://arxiv.org/abs/2403.03993)

    推荐系统中的个性化负采样技术在增量学习中的应用，解决了更新推荐系统模型时遇到的遗忘灾难问题。

    

    推荐系统已成为在线平台的重要组成部分。每天训练数据量不断扩大，用户互动次数不断增加。探索更大更具表现力的模型已成为改善用户体验的必要追求。然而，这种进展带来了更大的计算负担。在商业环境中，一旦推荐系统模型被训练和部署，通常需要频繁更新以适应新的客户数据。累积起来，数据量的增加必将使得从头开始进行全量重训练变得计算上不可行。仅仅在新数据上进行简单微调会遇到已被广泛记录的遗忘灾难问题。尽管负采样在使用隐式反馈进行训练中是至关重要的一部分，但目前并不存在专门针对增量学习的技术。

    arXiv:2403.03993v1 Announce Type: cross  Abstract: Recommender systems have become an integral part of online platforms. Every day the volume of training data is expanding and the number of user interactions is constantly increasing. The exploration of larger and more expressive models has become a necessary pursuit to improve user experience. However, this progression carries with it an increased computational burden. In commercial settings, once a recommendation system model has been trained and deployed it typically needs to be updated frequently as new client data arrive. Cumulatively, the mounting volume of data is guaranteed to eventually make full batch retraining of the model from scratch computationally infeasible. Naively fine-tuning solely on the new data runs into the well-documented problem of catastrophic forgetting. Despite the fact that negative sampling is a crucial part of training with implicit feedback, no specialized technique exists that is tailored to the increme
    
[^4]: 评估ChatGPT用于垃圾邮件检测的性能

    Evaluating the Performance of ChatGPT for Spam Email Detection

    [https://arxiv.org/abs/2402.15537](https://arxiv.org/abs/2402.15537)

    该研究评估了ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件检测的性能，并探讨了其在这一领域的潜力。

    

    电子邮件继续是专业和商业领域中至关重要且广泛使用的通信媒介。然而，垃圾邮件的普及给用户带来了重大挑战，扰乱了他们的日常工作并降低了生产率。因此，基于内容准确地识别和过滤垃圾邮件对网络安全至关重要。最近自然语言处理领域的发展，特别是大型语言模型如ChatGPT，在诸如问答和文本生成等任务中表现出色。然而，其在垃圾邮件识别方面的潜力尚未得到充分探索。为了填补这一空白，本研究尝试评估ChatGPT在英文和中文电子邮件数据集中用于垃圾邮件识别的能力。我们利用ChatGPT进行垃圾邮件检测，采用上下文学习，需要提示说明和少量示范。

    arXiv:2402.15537v1 Announce Type: cross  Abstract: Email continues to be a pivotal and extensively utilized communication medium within professional and commercial domains. Nonetheless, the prevalence of spam emails poses a significant challenge for users, disrupting their daily routines and diminishing productivity. Consequently, accurately identifying and filtering spam based on content has become crucial for cybersecurity. Recent advancements in natural language processing, particularly with large language models like ChatGPT, have shown remarkable performance in tasks such as question answering and text generation. However, its potential in spam identification remains underexplored. To fill in the gap, this study attempts to evaluate ChatGPT's capabilities for spam identification in both English and Chinese email datasets. We employ ChatGPT for spam email detection using in-context learning, which requires a prompt instruction and a few demonstrations. We also investigate how the t
    
[^5]: 大型代码模型是否理解编程概念？一种黑盒方法探究

    Do Large Code Models Understand Programming Concepts? A Black-box Approach

    [https://arxiv.org/abs/2402.05980](https://arxiv.org/abs/2402.05980)

    本文使用反事实分析框架评估了十个大型代码模型对四种编程概念的理解情况，发现当前模型缺乏对数据流和控制流等概念的理解。

    

    大型语言模型在文本生成方面的成功也使其在代码生成和编码任务方面表现更好。虽然有很多工作展示了它们在代码补全和编辑等任务上的出色性能，但为什么它们能够成功还不清楚。我们通过探索自回归模型对底层程序的逻辑结构理解程度，来填补这一差距。我们提出了用于编程概念谓词的反事实分析（CACP）作为一种反事实测试框架，以评估大型代码模型是否理解编程概念。只通过黑盒访问模型，我们使用CACP评估了十个流行的大型代码模型对四个不同编程概念的理解情况。我们的研究结果表明，当前模型缺乏对数据流和控制流等概念的理解。

    Large Language Models' success on text generation has also made them better at code generation and coding tasks. While a lot of work has demonstrated their remarkable performance on tasks such as code completion and editing, it is still unclear as to why. We help bridge this gap by exploring to what degree auto-regressive models understand the logical constructs of the underlying programs. We propose Counterfactual Analysis for Programming Concept Predicates (CACP) as a counterfactual testing framework to evaluate whether Large Code Models understand programming concepts. With only black-box access to the model, we use CACP to evaluate ten popular Large Code Models for four different programming concepts. Our findings suggest that current models lack understanding of concepts such as data flow and control flow.
    
[^6]: 深度学习在多变量时间序列插补中的应用：一项调查

    Deep Learning for Multivariate Time Series Imputation: A Survey

    [https://arxiv.org/abs/2402.04059](https://arxiv.org/abs/2402.04059)

    本文调查了深度学习在多变量时间序列插补中的应用。通过综述不同的方法以及它们的优点和限制，研究了它们对下游任务性能的改进，并指出了未来研究的开放问题。

    

    普遍存在的缺失值导致多变量时间序列数据部分观测，破坏了时间序列的完整性，阻碍了有效的时间序列数据分析。最近，深度学习插补方法在提高损坏的时间序列数据质量方面取得了显著的成功，进而提高了下游任务的性能。本文对最近提出的深度学习插补方法进行了全面的调查。首先，我们提出了对这些方法进行分类的方法，并通过强调它们的优点和限制来进行了结构化的综述。我们还进行了实证实验，研究了不同方法，并比较了它们对下游任务的改进。最后，我们指出了多变量时间序列插补未来研究的开放问题。本文的所有代码和配置，包括定期维护的多变量时间序列插补论文列表，可以在以下位置找到。

    The ubiquitous missing values cause the multivariate time series data to be partially observed, destroying the integrity of time series and hindering the effective time series data analysis. Recently deep learning imputation methods have demonstrated remarkable success in elevating the quality of corrupted time series data, subsequently enhancing performance in downstream tasks. In this paper, we conduct a comprehensive survey on the recently proposed deep learning imputation methods. First, we propose a taxonomy for the reviewed methods, and then provide a structured review of these methods by highlighting their strengths and limitations. We also conduct empirical experiments to study different methods and compare their enhancement for downstream tasks. Finally, the open issues for future research on multivariate time series imputation are pointed out. All code and configurations of this work, including a regularly maintained multivariate time series imputation paper list, can be foun
    
[^7]: 通过模拟进行算法性劝导

    Algorithmic Persuasion Through Simulation

    [https://arxiv.org/abs/2311.18138](https://arxiv.org/abs/2311.18138)

    通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。

    

    我们研究了一个贝叶斯劝导问题，其中发送者希望说服接收者采取二元行为，例如购买产品。发送者了解世界的（二元）状态，比如产品质量是高还是低，但是对接收者的信念和效用只有有限的信息。受到客户调查、用户研究和生成式人工智能的最新进展的启发，我们允许发送者通过查询模拟接收者的行为来了解更多关于接收者的信息。在固定数量的查询之后，发送者承诺一个消息策略，接收者根据收到的消息来最大化她的预期效用来采取行动。我们对发送者在任何接收者类型分布下的最优消息策略进行了表征。然后，我们设计了一个多项式时间查询算法，优化了这个贝叶斯劝导游戏中发送者的预期效用。

    arXiv:2311.18138v2 Announce Type: replace-cross Abstract: We study a Bayesian persuasion problem where a sender wants to persuade a receiver to take a binary action, such as purchasing a product. The sender is informed about the (binary) state of the world, such as whether the quality of the product is high or low, but only has limited information about the receiver's beliefs and utilities. Motivated by customer surveys, user studies, and recent advances in generative AI, we allow the sender to learn more about the receiver by querying an oracle that simulates the receiver's behavior. After a fixed number of queries, the sender commits to a messaging policy and the receiver takes the action that maximizes her expected utility given the message she receives. We characterize the sender's optimal messaging policy given any distribution over receiver types. We then design a polynomial-time querying algorithm that optimizes the sender's expected utility in this Bayesian persuasion game. We 
    
[^8]: 学习非稳态条件下的稳定性原则

    A Stability Principle for Learning under Non-Stationarity. (arXiv:2310.18304v1 [cs.LG])

    [http://arxiv.org/abs/2310.18304](http://arxiv.org/abs/2310.18304)

    本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。

    

    我们在非稳定环境中开发了一个灵活的统计学习框架。在每个时间段，我们的方法应用稳定性原则来选择一个回溯窗口，最大限度地利用历史数据，同时将累积偏差保持在与随机误差相对可接受的范围内。我们的理论展示了该方法对未知非稳定性的适应性。当人口损失函数强凸或仅满足Lipschitz条件时，遗憾界是极小化的最优解，仅受对数因子的影响。我们的分析核心是两个新颖的组成部分：函数之间的相似度度量和将非稳态数据序列划分为准稳态片段的分割技术。

    We develop a versatile framework for statistical learning in non-stationary environments. In each time period, our approach applies a stability principle to select a look-back window that maximizes the utilization of historical data while keeping the cumulative bias within an acceptable range relative to the stochastic error. Our theory showcases the adaptability of this approach to unknown non-stationarity. The regret bound is minimax optimal up to logarithmic factors when the population losses are strongly convex, or Lipschitz only. At the heart of our analysis lie two novel components: a measure of similarity between functions and a segmentation technique for dividing the non-stationary data sequence into quasi-stationary pieces.
    
[^9]: 用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统

    A Vision-Guided Robotic System for Grasping Harvested Tomato Trusses in Cluttered Environments. (arXiv:2309.17170v1 [cs.RO])

    [http://arxiv.org/abs/2309.17170](http://arxiv.org/abs/2309.17170)

    提出了一种用于在混乱环境中抓取已采摘的西红柿穗果的视觉引导机器人系统。该系统利用基于深度学习的视觉系统来识别穗果并确定适合抓取的位置，通过在线学习来排序抓取姿势，并实现无触觉传感器或几何模型的夹持抓取。实验表明，该系统具有100%的清理率和93%的一次性成功抓取率。

    

    目前，对于西红柿的称重和包装需要大量的人工操作。自动化的主要障碍在于开发一个可靠的用于已采摘的穗果的机器人抓取系统的困难。我们提出了一种方法来抓取堆放在装箱中的穗果，这是它们在采摘后常见的存储和运输方式。该方法包括一个基于深度学习的视觉系统，首先识别出装箱中的单个穗果，然后确定茎部的适合抓取的位置。为此，我们引入了一个具有在线学习能力的抓取姿势排序算法。在选择了最有前景的抓取姿势之后，机器人执行一种无需触觉传感器或几何模型的夹持抓取。实验室实验证明，配备了一个手眼一体的RGB-D相机的机器人操纵器从堆中捡起所有的穗果的清理率达到100%。93%的穗果在第一次尝试时成功抓取。

    Currently, truss tomato weighing and packaging require significant manual work. The main obstacle to automation lies in the difficulty of developing a reliable robotic grasping system for already harvested trusses. We propose a method to grasp trusses that are stacked in a crate with considerable clutter, which is how they are commonly stored and transported after harvest. The method consists of a deep learning-based vision system to first identify the individual trusses in the crate and then determine a suitable grasping location on the stem. To this end, we have introduced a grasp pose ranking algorithm with online learning capabilities. After selecting the most promising grasp pose, the robot executes a pinch grasp without needing touch sensors or geometric models. Lab experiments with a robotic manipulator equipped with an eye-in-hand RGB-D camera showed a 100% clearance rate when tasked to pick all trusses from a pile. 93% of the trusses were successfully grasped on the first try,
    
[^10]: 迈向因果GPT：通过促进LLMs中的因果一致性，基于多智能体的方法实现忠实的知识推理

    Towards CausalGPT: A Multi-Agent Approach for Faithful Knowledge Reasoning via Promoting Causal Consistency in LLMs. (arXiv:2308.11914v1 [cs.AI])

    [http://arxiv.org/abs/2308.11914](http://arxiv.org/abs/2308.11914)

    通过多智能体协作，我们提出了一种框架，旨在提高基于知识的推理的忠实度和因果性，通过推理器和因果评估器的合作来解决推理谬误。

    

    尽管LLMs的发展取得了一些进展，但基于知识的推理仍然是一个长期存在的问题，这是由于知识回忆和推理的脆弱性引起的。现有方法主要通过鼓励LLMs自主计划和解决问题或广泛采样推理链来解决这个问题，但未能解决概念和推理谬误。为了减少推理谬误，我们从多智能体协作中得到启发，提出了一个框架来增加基于知识的推理的忠实度和因果性。具体而言，我们建议使用多个智能体（即推理器和因果评估器）在推理和一致性范式中协作工作，以提高推理的忠实度。推理器专注于提供具有人类因果关系的解决方案，用于解决开放领域的问题。另一方面，因果评估器代理检查解决方案中的答案是否从问题中因果推导出来，反之亦然，并用一个反事实的答案来替代。

    Despite advancements in LLMs, knowledge-based reasoning remains a longstanding issue due to the fragility of knowledge recall and inference. Existing methods primarily encourage LLMs to autonomously plan and solve problems or to extensively sample reasoning chains without addressing the conceptual and inferential fallacies. Attempting to alleviate inferential fallacies and drawing inspiration from multi-agent collaboration, we present a framework to increase faithfulness and causality for knowledge-based reasoning. Specifically, we propose to employ multiple intelligent agents (i.e., reasoner and causal evaluator) to work collaboratively in a reasoning-and-consensus paradigm for elevated reasoning faithfulness. The reasoners focus on providing solutions with human-like causality to solve open-domain problems. On the other hand, the causal evaluator agent scrutinizes if the answer in a solution is causally deducible from the question and vice versa, with a counterfactual answer replacin
    
[^11]: 利用变分因果推断和精细关系信息预测细胞响应

    Predicting Cellular Responses with Variational Causal Inference and Refined Relational Information. (arXiv:2210.00116v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.00116](http://arxiv.org/abs/2210.00116)

    本研究利用基因调控网络信息设计了一种新的因果推断框架，并通过邻接矩阵更新技术预训练图卷积网络以更好地预测细胞在反事实干扰下的基因表达。同时，我们提出了一个鲁棒的估计器来高效估计边缘干扰效应。研究结果展示了该框架的优越性能。

    

    预测细胞在干扰下的响应可能为药物研发和个性化治疗带来重要好处。在本研究中，我们提出了一种新的图形变分贝叶斯因果推断框架，预测细胞在反事实干扰下（即细胞未真实接收的干扰）的基因表达，利用代表生物学知识的基因调控网络（GRN）信息来辅助个性化细胞响应预测。我们还针对数据自适应GRN开发了邻接矩阵更新技术用于图卷积网络的预训练，在模型性能上提供了更多的基因关系洞见。

    Predicting the responses of a cell under perturbations may bring important benefits to drug discovery and personalized therapeutics. In this work, we propose a novel graph variational Bayesian causal inference framework to predict a cell's gene expressions under counterfactual perturbations (perturbations that this cell did not factually receive), leveraging information representing biological knowledge in the form of gene regulatory networks (GRNs) to aid individualized cellular response predictions. Aiming at a data-adaptive GRN, we also developed an adjacency matrix updating technique for graph convolutional networks and used it to refine GRNs during pre-training, which generated more insights on gene relations and enhanced model performance. Additionally, we propose a robust estimator within our framework for the asymptotically efficient estimation of marginal perturbation effect, which is yet to be carried out in previous works. With extensive experiments, we exhibited the advanta
    

