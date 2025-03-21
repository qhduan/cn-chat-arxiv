# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) | 该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。 |
| [^2] | [A Comprehensive Survey on Process-Oriented Automatic Text Summarization with Exploration of LLM-Based Methods](https://arxiv.org/abs/2403.02901) | 本综合调查从“过程导向模式“视角提供了自动文本摘要的全面概述，全面审视了最新的基于LLM的ATS工作，并提供了关于ATS最新的调查，弥补了文献中的两年间隔。 |
| [^3] | [CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion](https://arxiv.org/abs/2402.05889) | 该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。 |
| [^4] | [SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning](https://arxiv.org/abs/2401.16013) | 这个论文介绍了SERL软件套件，它是一个用于样本高效的机器人强化学习的库。该库包含了一个离线深度强化学习方法、计算奖励和重置环境的方法，高质量的机器人控制器，以及一些具有挑战性的示例任务。这个软件套件的目标是解决机器人强化学习的难以使用和获取性的挑战。 |
| [^5] | [Graphical Object-Centric Actor-Critic.](http://arxiv.org/abs/2310.17178) | 这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。 |
| [^6] | [Approximate Computing Survey, Part II: Application-Specific & Architectural Approximation Techniques and Applications.](http://arxiv.org/abs/2307.11128) | 近似计算是一种能够调整系统设计结果质量以提高能源效率和/或性能的新兴解决方案，已吸引学术界和工业界的广泛关注。这篇论文是一个关于应用特定和架构近似技术的调查的第二部分。 |
| [^7] | [BELLA: Black box model Explanations by Local Linear Approximations.](http://arxiv.org/abs/2305.11311) | 本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。 |
| [^8] | [Human Choice Prediction in Non-Cooperative Games: Simulation-based Off-Policy Evaluation.](http://arxiv.org/abs/2305.10361) | 本文研究了语言游戏中的离线策略评估，并提出了一种结合真实和模拟数据的新方法。 |
| [^9] | [A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models.](http://arxiv.org/abs/2303.13773) | 本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。 |

# 详细

[^1]: 数据混合规律：通过预测语言建模性能来优化数据混合

    Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

    [https://arxiv.org/abs/2403.16952](https://arxiv.org/abs/2403.16952)

    该研究发现了数据混合规律，可以量化地预测模型性能与数据混合比例之间的关系，并提出了一种方法来通过拟合函数形式来引导理想的数据混合选择，从而优化大型语言模型的训练混合。

    

    大型语言模型的预训练数据包括多个领域（例如网络文本、学术论文、代码），其混合比例对结果模型的能力至关重要。现有的工作通常依赖于启发式方法或定性策略来调整比例，我们发现了模型性能与混合比例之间的函数形式的定量可预测性，我们称之为数据混合规律。在样本混合上拟合这种函数揭示了未见混合的模型性能，从而引导选择理想的数据混合。此外，我们提出了训练步骤、模型大小和我们的数据混合规律的缩放规律的嵌套使用，以使得仅通过小规模训练就能够预测在各种混合数据下训练的大模型的性能。此外，实验结果验证了我们的方法有效地优化了训练混合。

    arXiv:2403.16952v1 Announce Type: cross  Abstract: Pretraining data of large language models composes multiple domains (e.g., web texts, academic papers, codes), whose mixture proportions crucially impact the competence of outcome models. While existing endeavors rely on heuristics or qualitative strategies to tune the proportions, we discover the quantitative predictability of model performance regarding the mixture proportions in function forms, which we refer to as the data mixing laws. Fitting such functions on sample mixtures unveils model performance on unseen mixtures before actual runs, thus guiding the selection of an ideal data mixture. Furthermore, we propose nested use of the scaling laws of training steps, model sizes, and our data mixing law to enable predicting the performance of large models trained on massive data under various mixtures with only small-scale training. Moreover, experimental results verify that our method effectively optimizes the training mixture of a 
    
[^2]: 关于过程导向自动文本摘要的综合调查，并探讨基于LLM的方法

    A Comprehensive Survey on Process-Oriented Automatic Text Summarization with Exploration of LLM-Based Methods

    [https://arxiv.org/abs/2403.02901](https://arxiv.org/abs/2403.02901)

    本综合调查从“过程导向模式“视角提供了自动文本摘要的全面概述，全面审视了最新的基于LLM的ATS工作，并提供了关于ATS最新的调查，弥补了文献中的两年间隔。

    

    arXiv:2403.02901v1 公告类型: 新的 摘要: 自动文本摘要（ATS）利用自然语言处理（NLP）算法，旨在创建简洁准确的摘要，从而显著减少处理大量文本所需的人力。ATS在学术界和工业界都引起了极大兴趣。过去已进行了许多研究来调查ATS的方法; 但是，它们通常缺乏对实际实施的实用性，因为它们经常从理论的角度对以往的方法进行分类。此外，大型语言模型（LLMs）的出现改变了传统的ATS方法。在这项调查中，我们旨在 1）从“过程导向模式”视角提供ATS的全面概述，最符合实际应用; 2) 全面审视最新的基于LLM的ATS工作; 以及 3）提供关于ATS的最新调查，弥补文献中两年间隔之处。令人感到满意

    arXiv:2403.02901v1 Announce Type: new  Abstract: Automatic Text Summarization (ATS), utilizing Natural Language Processing (NLP) algorithms, aims to create concise and accurate summaries, thereby significantly reducing the human effort required in processing large volumes of text. ATS has drawn considerable interest in both academic and industrial circles. Many studies have been conducted in the past to survey ATS methods; however, they generally lack practicality for real-world implementations, as they often categorize previous methods from a theoretical standpoint. Moreover, the advent of Large Language Models (LLMs) has altered conventional ATS methods. In this survey, we aim to 1) provide a comprehensive overview of ATS from a ``Process-Oriented Schema'' perspective, which is best aligned with real-world implementations; 2) comprehensively review the latest LLM-based ATS works; and 3) deliver an up-to-date survey of ATS, bridging the two-year gap in the literature. To the best of o
    
[^3]: CREMA: 通过有效的模块化适应和融合进行多模态组合视频推理

    CREMA: Multimodal Compositional Video Reasoning via Efficient Modular Adaptation and Fusion

    [https://arxiv.org/abs/2402.05889](https://arxiv.org/abs/2402.05889)

    该论文提出了一种名为CREMA的高效且模块化的模态融合框架，用于将任意新的模态注入视频推理。通过利用预训练模型增强多种信息模态，并引入查询转换器和融合模块，实现了灵活且有效的多模态组合推理。

    

    尽管在多模态组合推理方法方面取得了令人瞩目的进展，但由于处理固定模态输入并更新许多模型参数，仍然存在灵活性和效率方面的限制。本文解决了这些关键挑战，提出了CREMA，一种用于将任何新的模态注入视频推理的高效且模块化的模态融合框架。我们首先利用现有的预训练模型从给定的视频中增强多种信息模态（如光流、3D点云、音频），而无需额外的人工注释。接下来，我们引入了一个查询转换器，该转换器与每个可以访问的模态相关联，并具有多个参数高效的模块。它将多种模态特征投影到LLM令牌嵌入空间，使模型能够整合不同的数据类型以进行响应生成。此外，我们提出了一个融合模块，用于压缩多模态查询，在LLM中保持计算效率的同时进行融合组合。

    Despite impressive advancements in multimodal compositional reasoning approaches, they are still limited in their flexibility and efficiency by processing fixed modality inputs while updating a lot of model parameters. This paper tackles these critical challenges and proposes CREMA, an efficient and modular modality-fusion framework for injecting any new modality into video reasoning. We first augment multiple informative modalities (such as optical flow, 3D point cloud, audio) from given videos without extra human annotation by leveraging existing pre-trained models. Next, we introduce a query transformer with multiple parameter-efficient modules associated with each accessible modality. It projects diverse modality features to the LLM token embedding space, allowing the model to integrate different data types for response generation. Furthermore, we propose a fusion module designed to compress multimodal queries, maintaining computational efficiency in the LLM while combining additio
    
[^4]: SERL: 用于样本高效的机器人强化学习的软件套件

    SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning

    [https://arxiv.org/abs/2401.16013](https://arxiv.org/abs/2401.16013)

    这个论文介绍了SERL软件套件，它是一个用于样本高效的机器人强化学习的库。该库包含了一个离线深度强化学习方法、计算奖励和重置环境的方法，高质量的机器人控制器，以及一些具有挑战性的示例任务。这个软件套件的目标是解决机器人强化学习的难以使用和获取性的挑战。

    

    近年来，在机器人强化学习领域取得了显著进展，使得可以处理复杂的图像观察，实际训练，并结合辅助数据（如示范和先前经验）。然而，尽管取得了这些进展，机器人强化学习仍然难以使用。从实践者中认识到，这些算法的具体实现细节对性能的影响常常与算法选择同样重要（如果不是更重要）。我们认为，机器人强化学习被广泛采用以及进一步发展机器人强化学习方法的一个重要挑战是这些方法的相对难以获取性。为了解决这个挑战，我们开发了一个精心实现的库，其中包含了一种高效样本离线深度强化学习方法，以及计算奖励和重置环境的方法，针对广泛采用的机器人的高质量控制器，以及一些具有挑战性的示例任务。

    In recent years, significant progress has been made in the field of robotic reinforcement learning (RL), enabling methods that handle complex image observations, train in the real world, and incorporate auxiliary data, such as demonstrations and prior experience. However, despite these advances, robotic RL remains hard to use. It is acknowledged among practitioners that the particular implementation details of these algorithms are often just as important (if not more so) for performance as the choice of algorithm. We posit that a significant challenge to widespread adoption of robotic RL, as well as further development of robotic RL methods, is the comparative inaccessibility of such methods. To address this challenge, we developed a carefully implemented library containing a sample efficient off-policy deep RL method, together with methods for computing rewards and resetting the environment, a high-quality controller for a widely-adopted robot, and a number of challenging example task
    
[^5]: 图形化的以对象为中心的Actor-Critic算法

    Graphical Object-Centric Actor-Critic. (arXiv:2310.17178v1 [cs.AI])

    [http://arxiv.org/abs/2310.17178](http://arxiv.org/abs/2310.17178)

    这项研究提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家和基于模型的方法结合起来，利用解耦的对象表示有效地学习策略。该方法填补了以对象为中心的强化学习环境中高效且适用于离散或连续动作空间的世界模型的研究空白。

    

    最近在无监督的以对象为中心的表示学习及其在下游任务中的应用方面取得了重要进展。最新的研究支持这样一个观点，即在基于图像的以对象为中心的强化学习任务中采用解耦的对象表示能够促进策略学习。我们提出了一种新颖的以对象为中心的强化学习算法，将演员-评论家算法和基于模型的方法结合起来，以有效利用这些表示。在我们的方法中，我们使用一个变换器编码器来提取对象表示，并使用图神经网络来近似环境的动力学。所提出的方法填补了开发强化学习环境中可以用于离散或连续动作空间的高效以对象为中心的世界模型的研究空白。我们的算法在一个具有复杂视觉3D机器人环境和一个具有组合结构的2D环境中表现更好。

    There have recently been significant advances in the problem of unsupervised object-centric representation learning and its application to downstream tasks. The latest works support the argument that employing disentangled object representations in image-based object-centric reinforcement learning tasks facilitates policy learning. We propose a novel object-centric reinforcement learning algorithm combining actor-critic and model-based approaches to utilize these representations effectively. In our approach, we use a transformer encoder to extract object representations and graph neural networks to approximate the dynamics of an environment. The proposed method fills a research gap in developing efficient object-centric world models for reinforcement learning settings that can be used for environments with discrete or continuous action spaces. Our algorithm performs better in a visually complex 3D robotic environment and a 2D environment with compositional structure than the state-of-t
    
[^6]: 近似计算调查，第二部分：应用特定和架构近似技术及应用

    Approximate Computing Survey, Part II: Application-Specific & Architectural Approximation Techniques and Applications. (arXiv:2307.11128v1 [cs.AR])

    [http://arxiv.org/abs/2307.11128](http://arxiv.org/abs/2307.11128)

    近似计算是一种能够调整系统设计结果质量以提高能源效率和/或性能的新兴解决方案，已吸引学术界和工业界的广泛关注。这篇论文是一个关于应用特定和架构近似技术的调查的第二部分。

    

    计算密集型应用的挑战性部署，如人工智能（AI）和数字信号处理（DSP），迫使计算系统界探索新的设计方法。近似计算成为一种新兴解决方案，允许在系统设计中调整结果的质量，以提高能源效率和/或性能。近年来，这种根本性的范式转变吸引了学术界和工业界的兴趣，并在不同设计层面（从系统到集成电路）上进行了重要的近似技术和方法的研究。受近似计算在过去10年的广泛吸引力的驱使，我们进行了一个两部分的调查，涵盖了关键方面（如术语和应用）并回顾了传统计算堆栈的各个层面的最新近似技术。在我们的调查第二部分中，我们对应用特定和架构近似技术的技术细节进行分类和介绍。

    The challenging deployment of compute-intensive applications from domains such Artificial Intelligence (AI) and Digital Signal Processing (DSP), forces the community of computing systems to explore new design approaches. Approximate Computing appears as an emerging solution, allowing to tune the quality of results in the design of a system in order to improve the energy efficiency and/or performance. This radical paradigm shift has attracted interest from both academia and industry, resulting in significant research on approximation techniques and methodologies at different design layers (from system down to integrated circuits). Motivated by the wide appeal of Approximate Computing over the last 10 years, we conduct a two-part survey to cover key aspects (e.g., terminology and applications) and review the state-of-the art approximation techniques from all layers of the traditional computing stack. In Part II of our survey, we classify and present the technical details of application-s
    
[^7]: BELLA: 通过本地线性逼近进行黑盒模型解释

    BELLA: Black box model Explanations by Local Linear Approximations. (arXiv:2305.11311v1 [cs.LG])

    [http://arxiv.org/abs/2305.11311](http://arxiv.org/abs/2305.11311)

    本文提出了一种确定性的、与模型无关的事后方法BELLA，用于解释回归黑盒模型的个别预测。该方法通过特征空间中训练的线性模型提供解释，使得该模型的系数可以直接用于计算特征值的预测值。此外，BELLA最大化了线性模型适用的领域范围。

    

    近年来，理解黑盒模型的决策过程不仅成为法律要求，也成为评估其性能的另一种方式。然而，现有的事后解释方法依赖于合成数据生成，这引入了不确定性并可能损害解释的可靠性，并且它们 tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a

    In recent years, understanding the decision-making process of black-box models has become not only a legal requirement but also an additional way to assess their performance. However, the state of the art post-hoc interpretation approaches rely on synthetic data generation. This introduces uncertainty and can hurt the reliability of the interpretations. Furthermore, they tend to produce explanations that apply to only very few data points. This makes the explanations brittle and limited in scope. Finally, they provide scores that have no direct verifiable meaning. In this paper, we present BELLA, a deterministic model-agnostic post-hoc approach for explaining the individual predictions of regression black-box models. BELLA provides explanations in the form of a linear model trained in the feature space. Thus, its coefficients can be used directly to compute the predicted value from the feature values. Furthermore, BELLA maximizes the size of the neighborhood to which the linear model a
    
[^8]: 非合作博弈中的人类选择预测：基于模拟的离线策略评估

    Human Choice Prediction in Non-Cooperative Games: Simulation-based Off-Policy Evaluation. (arXiv:2305.10361v1 [cs.LG])

    [http://arxiv.org/abs/2305.10361](http://arxiv.org/abs/2305.10361)

    本文研究了语言游戏中的离线策略评估，并提出了一种结合真实和模拟数据的新方法。

    

    说服游戏在经济和人工智能研究中具有重要意义并具有重要的实际应用。本文探讨了在基于语言的说服游戏中离线策略评估（OPE）的挑战性问题，提出了一种结合真实和模拟人类 - 机器人交互数据的新方法，并给出了一种深度学习训练算法，该算法有效地整合了真实交互和模拟数据。

    Persuasion games have been fundamental in economics and AI research, and have significant practical applications. Recent works in this area have started to incorporate natural language, moving beyond the traditional stylized message setting. However, previous research has focused on on-policy prediction, where the train and test data have the same distribution, which is not representative of real-life scenarios. In this paper, we tackle the challenging problem of off-policy evaluation (OPE) in language-based persuasion games. To address the inherent difficulty of human data collection in this setup, we propose a novel approach which combines real and simulated human-bot interaction data. Our simulated data is created by an exogenous model assuming decision makers (DMs) start with a mixture of random and decision-theoretic based behaviors and improve over time. We present a deep learning training algorithm that effectively integrates real interaction and simulated data, substantially im
    
[^9]: 基于图神经网络的纳米卫星任务调度方法：学习混合整数模型的洞见

    A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models. (arXiv:2303.13773v1 [cs.LG])

    [http://arxiv.org/abs/2303.13773](http://arxiv.org/abs/2303.13773)

    本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。

    

    本研究探讨如何利用图神经网络（GNN）更有效地调度纳米卫星任务。在离线纳米卫星任务调度（ONTS）问题中，目标是找到在轨道上执行任务的最佳安排，同时考虑服务质量（QoS）方面的考虑因素，如优先级，最小和最大激活事件，执行时间框架，周期和执行窗口，以及卫星电力资源和能量收集和管理的复杂性的约束。ONTS问题已经使用传统的数学公式和精确方法进行了处理，但是它们在问题的挑战性案例中的适用性有限。本研究考察了在这种情况下使用GNN的方法，该方法已经成功应用于许多优化问题，包括旅行商问题，调度问题和设施放置问题。在本文中，我们将ONTS问题的MILP实例完全表示成二分图网络结构来应用GNN。

    This study investigates how to schedule nanosatellite tasks more efficiently using Graph Neural Networks (GNN). In the Offline Nanosatellite Task Scheduling (ONTS) problem, the goal is to find the optimal schedule for tasks to be carried out in orbit while taking into account Quality-of-Service (QoS) considerations such as priority, minimum and maximum activation events, execution time-frames, periods, and execution windows, as well as constraints on the satellite's power resources and the complexity of energy harvesting and management. The ONTS problem has been approached using conventional mathematical formulations and precise methods, but their applicability to challenging cases of the problem is limited. This study examines the use of GNNs in this context, which has been effectively applied to many optimization problems, including traveling salesman problems, scheduling problems, and facility placement problems. Here, we fully represent MILP instances of the ONTS problem in biparti
    

