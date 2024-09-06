# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots](https://arxiv.org/abs/2404.02515) | 提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。 |
| [^2] | [PointCloud-Text Matching: Benchmark Datasets and a Baseline](https://arxiv.org/abs/2403.19386) | 本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。 |
| [^3] | [How to Train your Antivirus: RL-based Hardening through the Problem-Space](https://arxiv.org/abs/2402.19027) | 引入了一种基于强化学习的方法，可在问题空间内构建对抗样本，对抗防病毒软件中的恶意软件攻击。 |
| [^4] | [Aligning Large Language Models to a Domain-specific Graph Database](https://arxiv.org/abs/2402.16567) | 该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。 |
| [^5] | [Language-Guided World Models: A Model-Based Approach to AI Control](https://arxiv.org/abs/2402.01695) | 语言引导的世界模型（LWMs）是一种基于模型的人工智能控制方法，它通过阅读语言描述来捕捉环境动态，提高了代理的沟通效率，并允许人类通过简洁的语言反馈同时改变他们在多个任务上的行为。 |
| [^6] | [Decision Theoretic Foundations for Experiments Evaluating Human Decisions.](http://arxiv.org/abs/2401.15106) | 该论文通过综合统计决策理论和信息经济学，提出了决策问题的广泛适用定义。为了将人类决策的下降归咎于偏见形式，实验必须向参与者提供足够的信息来识别规范决策。然而，根据作者对AI辅助决策的研究的评估，只有17%的研究提供了足够的信息来描述参与者的行为偏离了良好的决策。 |
| [^7] | [Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation.](http://arxiv.org/abs/2401.06477) | Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。 |
| [^8] | [Generate and Pray: Using SALLMS to Evaluate the Security of LLM Generated Code.](http://arxiv.org/abs/2311.00889) | 该论文研究了使用SALLMS评估LLM生成代码的安全性，指出现有数据集和评估指标未能充分考虑到与安全相关的真实软件工程任务，从而导致不安全的代码生成。 |
| [^9] | [Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review.](http://arxiv.org/abs/2310.14735) | 这篇论文解释了提示工程在释放大型语言模型能力方面的关键作用，探讨了不同的提示方法以及外部插件如何协助减少机器幻想，并指出了未来研究方向的重要性。 |
| [^10] | [Implementation of The Future of Drug Discovery: QuantumBased Machine Learning Simulation (QMLS).](http://arxiv.org/abs/2308.08561) | 该论文介绍了一种名为QMLS的新概念，通过结合机器学习和量子模拟的方法，可以缩短药物研发的时间和降低成本。通过生成命中物和优化分子的过程，可以大大提高药物发现的效率。 |
| [^11] | [Large-Batch, Neural Multi-Objective Bayesian Optimization.](http://arxiv.org/abs/2306.01095) | 本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。 |
| [^12] | [PESTS: Persian_English Cross Lingual Corpus for Semantic Textual Similarity.](http://arxiv.org/abs/2305.07893) | 本研究提出了跨语言的语义相似性模型PESTS，并通过波斯语-英语的跨语言语料库来验证模型的准确性。 |
| [^13] | [Hierarchical Generative Adversarial Imitation Learning with Mid-level Input Generation for Autonomous Driving on Urban Environments.](http://arxiv.org/abs/2302.04823) | 本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。 |

# 详细

[^1]: 利用在线校准运动模型的紧耦合LiDAR-IMU-轮里程计算法用于滑移转向机器人

    Tightly-Coupled LiDAR-IMU-Wheel Odometry with Online Calibration of a Kinematic Model for Skid-Steering Robots

    [https://arxiv.org/abs/2404.02515](https://arxiv.org/abs/2404.02515)

    提出了一种紧耦合LiDAR-IMU-轮里程计算法，使用在线校准解决滑移转向机器人在挑战性环境中的点云退化问题。

    

    隧道和长廊是移动机器人具有挑战性的环境，因为在这些环境中LiDAR点云会退化。为了解决点云退化问题，本研究提出了一种用于滑移转向机器人的紧耦合LiDAR-IMU-轮里程计算法，同时还使用在线校准方法。我们提出了一个完整的线性轮子里程计因子，不仅作为运动约束，还可以执行滑移转向机器人运动模型的在线校准。尽管运动模型动态变化（例如由于胎压引起的轮胎半径变化）和地形条件变化，我们的方法能够通过在线校准来解决模型误差。此外，我们的方法能够在退化环境下（如长直廊）通过校准而实现准确定位，同时LiDAR-IMU融合运作良好。此外，我们还估计了轮子里程计的不确定性（即协方差矩阵）。

    arXiv:2404.02515v1 Announce Type: cross  Abstract: Tunnels and long corridors are challenging environments for mobile robots because a LiDAR point cloud should degenerate in these environments. To tackle point cloud degeneration, this study presents a tightly-coupled LiDAR-IMU-wheel odometry algorithm with an online calibration for skid-steering robots. We propose a full linear wheel odometry factor, which not only serves as a motion constraint but also performs the online calibration of kinematic models for skid-steering robots. Despite the dynamically changing kinematic model (e.g., wheel radii changes caused by tire pressures) and terrain conditions, our method can address the model error via online calibration. Moreover, our method enables an accurate localization in cases of degenerated environments, such as long and straight corridors, by calibration while the LiDAR-IMU fusion sufficiently operates. Furthermore, we estimate the uncertainty (i.e., covariance matrix) of the wheel o
    
[^2]: PointCloud-Text匹配：基准数据集和一个基线

    PointCloud-Text Matching: Benchmark Datasets and a Baseline

    [https://arxiv.org/abs/2403.19386](https://arxiv.org/abs/2403.19386)

    本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。

    

    在本文中，我们介绍和研究了一个新的实例级检索任务：PointCloud-Text Matching（PTM），旨在找到与给定的点云查询或文本查询匹配的确切跨模态实例。PTM可应用于各种场景，如室内/城市峡谷定位和场景检索。然而，在实践中尚无适用的、有针对性的PTM数据集。因此，我们构建了三个新的PTM基准数据集，分别为3D2T-SR、3D2T-NR和3D2T-QA。我们观察到数据具有挑战性，由于点云的稀疏、噪声或无序，以及文本的模糊、含糊或不完整，导致存在嘈杂的对应关系，使得现有的跨模态匹配方法对PTM无效。为了解决这些挑战，我们提出了一个PTM基线，命名为Robust PointCloud-Text Matching方法（RoMa）。RoMa包含两个模块：双重注意感知模块（DAP）和鲁棒负对比模块

    arXiv:2403.19386v1 Announce Type: cross  Abstract: In this paper, we present and study a new instance-level retrieval task: PointCloud-Text Matching~(PTM), which aims to find the exact cross-modal instance that matches a given point-cloud query or text query. PTM could be applied to various scenarios, such as indoor/urban-canyon localization and scene retrieval. However, there exists no suitable and targeted dataset for PTM in practice. Therefore, we construct three new PTM benchmark datasets, namely 3D2T-SR, 3D2T-NR, and 3D2T-QA. We observe that the data is challenging and with noisy correspondence due to the sparsity, noise, or disorder of point clouds and the ambiguity, vagueness, or incompleteness of texts, which make existing cross-modal matching methods ineffective for PTM. To tackle these challenges, we propose a PTM baseline, named Robust PointCloud-Text Matching method (RoMa). RoMa consists of two modules: a Dual Attention Perception module (DAP) and a Robust Negative Contrast
    
[^3]: 如何训练您的防病毒软件：基于强化学习的问题空间加固

    How to Train your Antivirus: RL-based Hardening through the Problem-Space

    [https://arxiv.org/abs/2402.19027](https://arxiv.org/abs/2402.19027)

    引入了一种基于强化学习的方法，可在问题空间内构建对抗样本，对抗防病毒软件中的恶意软件攻击。

    

    本文探讨了一种特定的机器学习架构，用于加固一家著名商业防病毒公司流程中的机器学习防御技术，以对抗恶意软件。我们引入了一种新颖的强化学习方法，用于构建对抗样本，这是对抗逃避攻击的模型训练的重要组成部分。

    arXiv:2402.19027v1 Announce Type: cross  Abstract: ML-based malware detection on dynamic analysis reports is vulnerable to both evasion and spurious correlations. In this work, we investigate a specific ML architecture employed in the pipeline of a widely-known commercial antivirus company, with the goal to harden it against adversarial malware. Adversarial training, the sole defensive technique that can confer empirical robustness, is not applicable out of the box in this domain, for the principal reason that gradient-based perturbations rarely map back to feasible problem-space programs. We introduce a novel Reinforcement Learning approach for constructing adversarial examples, a constituent part of adversarially training a model against evasion. Our approach comes with multiple advantages. It performs modifications that are feasible in the problem-space, and only those; thus it circumvents the inverse mapping problem. It also makes possible to provide theoretical guarantees on the r
    
[^4]: 将大型语言模型对齐到特定领域的图数据库

    Aligning Large Language Models to a Domain-specific Graph Database

    [https://arxiv.org/abs/2402.16567](https://arxiv.org/abs/2402.16567)

    该论文提出了一种将大型语言模型对齐到特定领域的图数据库的方法，通过利用ChatGPT生成NL-GQL数据对并微调LLMs，实现了两者之间的对齐。

    

    图数据库（Graph DB）被广泛应用于金融、社交网络和医药等各个领域。然而，将自然语言（NL）转换为图查询语言（GQL），通常称为NL2GQL，由于其固有复杂性和专业化特性而变得具有挑战性。一些方法试图利用大型语言模型（LLMs）来解决类似的任务，如文本转SQL。然而，在特定领域的NL2GQL任务中，缺乏特定领域的NL-GQL数据对使得难以建立LLMs和图数据库之间的对齐。为了解决这一挑战，我们提出了一个明确定义的流水线。具体地，我们利用ChatGPT基于给定的图数据库自我生成NL-GQL数据对。然后，我们使用创建的数据来对LLMs进行微调，从而实现LLMs与图数据库之间的对齐。此外，在推断过程中，我们提出了一种提取相关信息的方法。

    arXiv:2402.16567v1 Announce Type: new  Abstract: Graph Databases (Graph DB) are widely applied in various fields, including finance, social networks, and medicine. However, translating Natural Language (NL) into the Graph Query Language (GQL), commonly known as NL2GQL, proves to be challenging due to its inherent complexity and specialized nature. Some approaches have sought to utilize Large Language Models (LLMs) to address analogous tasks like text2SQL. Nevertheless, when it comes to NL2GQL taskson a particular domain, the absence of domain-specific NL-GQL data pairs makes it difficult to establish alignment between LLMs and the graph DB. To address this challenge, we propose a well-defined pipeline. Specifically, we utilize ChatGPT to create NL-GQL data pairs based on the given graph DB with self-instruct. Then, we use the created data to fine-tune LLMs, thereby achieving alignment between LLMs and the graph DB. Additionally, during inference, we propose a method that extracts relev
    
[^5]: 语言引导的世界模型：一种基于模型的人工智能控制方法

    Language-Guided World Models: A Model-Based Approach to AI Control

    [https://arxiv.org/abs/2402.01695](https://arxiv.org/abs/2402.01695)

    语言引导的世界模型（LWMs）是一种基于模型的人工智能控制方法，它通过阅读语言描述来捕捉环境动态，提高了代理的沟通效率，并允许人类通过简洁的语言反馈同时改变他们在多个任务上的行为。

    

    将概率世界模型安装到人工智能代理中，为人类与这些代理沟通和控制打开了一个高效的渠道。除了更新代理策略，人类还可以修改他们的内部世界模型，以影响代理的决策。然而，当前现有的世界模型难以适应人类，因为它们缺乏自然的通信界面。为了解决这个问题，我们开发了语言引导的世界模型（LWMs），它们可以通过阅读语言描述来捕捉环境动态。这些模型提高了代理的沟通效率，使人类能够通过简洁的语言反馈同时改变他们在多个任务上的行为。它们还使代理能够从最初用于指导人类的文本中进行自我学习。为了促进LWMs的发展，我们设计了一个基于MESSENGER游戏（Hanjie等人，2021）的挑战基准，需要对新场景进行组合泛化。

    Installing probabilistic world models into artificial agents opens an efficient channel for humans to communicate with and control these agents. In addition to updating agent policies, humans can modify their internal world models in order to influence their decisions. The challenge, however, is that currently existing world models are difficult for humans to adapt because they lack a natural communication interface. Aimed at addressing this shortcoming, we develop Language-Guided World Models (LWMs), which can capture environment dynamics by reading language descriptions. These models enhance agent communication efficiency, allowing humans to simultaneously alter their behavior on multiple tasks with concise language feedback. They also enable agents to self-learn from texts originally written to instruct humans. To facilitate the development of LWMs, we design a challenging benchmark based on the game of MESSENGER (Hanjie et al., 2021), requiring compositional generalization to new l
    
[^6]: 决策理论基础对评估人类决策的实验的影响

    Decision Theoretic Foundations for Experiments Evaluating Human Decisions. (arXiv:2401.15106v1 [cs.HC])

    [http://arxiv.org/abs/2401.15106](http://arxiv.org/abs/2401.15106)

    该论文通过综合统计决策理论和信息经济学，提出了决策问题的广泛适用定义。为了将人类决策的下降归咎于偏见形式，实验必须向参与者提供足够的信息来识别规范决策。然而，根据作者对AI辅助决策的研究的评估，只有17%的研究提供了足够的信息来描述参与者的行为偏离了良好的决策。

    

    信息展示的决策是可解释AI、人工智能与人类的合作以及数据可视化等领域研究的重点。然而，决策问题的定义以及实验必须具备的条件以得出人类决策存在缺陷的结论仍然存在争议。我们提出了一个广泛适用的决策问题定义，该定义是从统计决策理论和信息经济学中综合提炼而来的。我们认为，要将人类绩效下降归咎于某种偏见形式，实验必须向参与者提供足够的信息，以便合理的代理能够识别规范决策。我们评估了最近有关AI辅助决策的文献中对决策制定进行的评估在多大程度上达到了这一标准。我们发现，只有35项声称确定了有偏差行为的研究中的6项（17%）向参与者提供了足够信息来描述其行为偏离良好决策

    Decision-making with information displays is a key focus of research in areas like explainable AI, human-AI teaming, and data visualization. However, what constitutes a decision problem, and what is required for an experiment to be capable of concluding that human decisions are flawed in some way, remain open to speculation. We present a widely applicable definition of a decision problem synthesized from statistical decision theory and information economics. We argue that to attribute loss in human performance to forms of bias, an experiment must provide participants with the information that a rational agent would need to identify the normative decision. We evaluate the extent to which recent evaluations of decision-making from the literature on AI-assisted decisions achieve this criteria. We find that only 6 (17\%) of 35 studies that claim to identify biased behavior present participants with sufficient information to characterize their behavior as deviating from good decision-making
    
[^7]: Kun: 使用指令反向翻译的中国自对齐问题的答案优化方法

    Kun: Answer Polishment for Chinese Self-Alignment with Instruction Back-Translation. (arXiv:2401.06477v1 [cs.CL])

    [http://arxiv.org/abs/2401.06477](http://arxiv.org/abs/2401.06477)

    Kun是一种使用指令反向翻译和答案优化的方法，用于创建高质量的指导调整数据集，该方法不依赖于手动注释，通过自我筛选过程来改善和选择最有效的指令-输出对。它的主要创新在于通过算法改进提高数据的保留和清晰度，并通过创新的数据生成方法减少了手动注释的依赖。

    

    在本文中，我们介绍了一种名为Kun的新方法，用于在不依赖手动注释的情况下为大型语言模型（LLMs）创建高质量的指导调整数据集。Kun利用来自吾道、完卷和SkyPile等多个来源的未标记数据，采用基于指令反向翻译和答案优化的自我训练算法，生成了一个超过一百万个中文指导数据点的大规模数据集。该方法通过使用自我筛选过程来完善和选择最有效的指令-输出对，显著偏离传统方法。我们在多个基准测试上对6B参数的Yi模型进行了实验，结果表明Kun具有鲁棒性和可扩展性。我们方法的核心贡献在于算法的改进，增强了数据的保留和清晰度，并且创新的数据生成方法极大地减少了对昂贵和耗时的手动注释的依赖。这种方法ological方法提出了一种解决中文自对齐问题的方法，并提高了数据的准确性和质量。

    In this paper, we introduce Kun, a novel approach for creating high-quality instruction-tuning datasets for large language models (LLMs) without relying on manual annotations. Adapting a self-training algorithm based on instruction back-translation and answer polishment, Kun leverages unlabelled data from diverse sources such as Wudao, Wanjuan, and SkyPile to generate a substantial dataset of over a million Chinese instructional data points. This approach significantly deviates from traditional methods by using a self-curation process to refine and select the most effective instruction-output pairs. Our experiments with the 6B-parameter Yi model across various benchmarks demonstrate Kun's robustness and scalability. Our method's core contributions lie in its algorithmic advancement, which enhances data retention and clarity, and its innovative data generation approach that substantially reduces the reliance on costly and time-consuming manual annotations. This methodology presents a sc
    
[^8]: 生成和验证：使用SALLMS评估LLM生成的代码的安全性

    Generate and Pray: Using SALLMS to Evaluate the Security of LLM Generated Code. (arXiv:2311.00889v1 [cs.SE])

    [http://arxiv.org/abs/2311.00889](http://arxiv.org/abs/2311.00889)

    该论文研究了使用SALLMS评估LLM生成代码的安全性，指出现有数据集和评估指标未能充分考虑到与安全相关的真实软件工程任务，从而导致不安全的代码生成。

    

    随着大型语言模型（例如GitHub Copilot，ChatGPT等）在软件工程师的日常实践中越来越受欢迎，确保这些工具生成的代码不仅功能正确，而且没有漏洞变得非常重要。尽管LLM可以帮助开发人员提高生产力，但之前的实证研究表明LLM可能会生成不安全的代码。存在两个导致不安全代码生成的因素。首先，用于评估大型语言模型（LLM）的现有数据集没有充分地代表与安全相关的真实软件工程任务。相反，它们通常基于竞技编程挑战或以课堂形式为基础的编码任务。在真实世界的应用中，生成的代码将被集成到更大的代码库中，引入潜在的安全风险。目前缺乏专注于评估生成代码安全性的基准。其次，现有的评估指标主要侧重于功能性而忽视安全性。

    With the growing popularity of Large Language Models (e.g. GitHub Copilot, ChatGPT, etc.) in software engineers' daily practices, it is important to ensure that the code generated by these tools is not only functionally correct but also free of vulnerabilities. Although LLMs can help developers to be more productive, prior empirical studies have shown that LLMs can generate insecure code. There are two contributing factors to the insecure code generation. First, existing datasets used to evaluate Large Language Models (LLMs) do not adequately represent genuine software engineering tasks sensitive to security. Instead, they are often based on competitive programming challenges or classroom-type coding tasks. In real-world applications, the code produced is integrated into larger codebases, introducing potential security risks. There's a clear absence of benchmarks that focus on evaluating the security of the generated code. Second, existing evaluation metrics primarily focus on the func
    
[^9]: 激发大型语言模型中的提示工程潜力：一项综述

    Unleashing the potential of prompt engineering in Large Language Models: a comprehensive review. (arXiv:2310.14735v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14735](http://arxiv.org/abs/2310.14735)

    这篇论文解释了提示工程在释放大型语言模型能力方面的关键作用，探讨了不同的提示方法以及外部插件如何协助减少机器幻想，并指出了未来研究方向的重要性。

    

    本文深入探讨了提示工程在释放大型语言模型（LLM）能力方面的关键作用。提示工程是为LLM构建输入文本的过程，是优化LLM有效性的重要技术。本综述阐明了提示工程的基本原理，如角色提示、一次性提示和少量提示，以及更高级的方法，如思维链和思维树提示。本文还阐述了外部插件如何协助此任务，并通过检索外部知识来减少机器幻想。随后，我们勾勒了提示工程研究的前景方向，强调了对结构和代理在人工智能生成内容（AIGC）工具中的作用的深入理解的必要性。我们讨论了如何从不同角度和使用不同的方法评估提示方法的有效性。最后，我们提出了展望未来的研究方向。

    This paper delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). Prompt engineering is the process of structuring input text for LLMs and is a technique integral to optimizing the efficacy of LLMs. This survey elucidates foundational principles of prompt engineering, such as role-prompting, one-shot, and few-shot prompting, as well as more advanced methodologies such as the chain-of-thought and tree-of-thoughts prompting. The paper sheds light on how external assistance in the form of plugins can assist in this task, and reduce machine hallucination by retrieving external knowledge. We subsequently delineate prospective directions in prompt engineering research, emphasizing the need for a deeper understanding of structures and the role of agents in Artificial Intelligence-Generated Content (AIGC) tools. We discuss how to assess the efficacy of prompt methods from different perspectives and using different methods. Finally, we
    
[^10]: 未来药物发现的实施：基于量子的机器学习模拟(QMLS)。

    Implementation of The Future of Drug Discovery: QuantumBased Machine Learning Simulation (QMLS). (arXiv:2308.08561v1 [q-bio.BM])

    [http://arxiv.org/abs/2308.08561](http://arxiv.org/abs/2308.08561)

    该论文介绍了一种名为QMLS的新概念，通过结合机器学习和量子模拟的方法，可以缩短药物研发的时间和降低成本。通过生成命中物和优化分子的过程，可以大大提高药物发现的效率。

    

    药物研发的研究与开发(R&D)阶段是一个漫长而昂贵的过程。为了改革这个过程，我们引入了新概念QMLS，将整个R&D阶段缩短到三到六个月，成本仅为五到八万美元。对于命中产生，机器学习分子生成(MLMG)根据目标蛋白的分子结构生成可能的命中物，而量子模拟(QS)根据与目标蛋白的反应和结合效果过滤原始实验中的分子。然后，对于铅优化，从MLMG和QS生成和过滤的结果分子进行比较，并通过机器学习分子变异(MLMV)将那些出现在两个过程中的分子制成数十种分子变体，而其他分子只制成几种变体。最后，所有优化的分子将经过多轮高标准的QS过滤，以确保反应效果。

    The Research & Development (R&D) phase of drug development is a lengthy and costly process. To revolutionize this process, we introduce our new concept QMLS to shorten the whole R&D phase to three to six months and decrease the cost to merely fifty to eighty thousand USD. For Hit Generation, Machine Learning Molecule Generation (MLMG) generates possible hits according to the molecular structure of the target protein while the Quantum Simulation (QS) filters molecules from the primary essay based on the reaction and binding effectiveness with the target protein. Then, For Lead Optimization, the resultant molecules generated and filtered from MLMG and QS are compared, and molecules that appear as a result of both processes will be made into dozens of molecular variations through Machine Learning Molecule Variation (MLMV), while others will only be made into a few variations. Lastly, all optimized molecules would undergo multiple rounds of QS filtering with a high standard for reaction ef
    
[^11]: 大批量神经多目标贝叶斯优化

    Large-Batch, Neural Multi-Objective Bayesian Optimization. (arXiv:2306.01095v1 [cs.LG])

    [http://arxiv.org/abs/2306.01095](http://arxiv.org/abs/2306.01095)

    本文提出了一种针对数据密集型问题和多目标优化设置的贝叶斯优化框架，该方法利用了贝叶斯神经网络代理建模和可扩展、具有不确定性的收购策略，能够在最少迭代次数的情况下高效地进行优化。

    

    贝叶斯优化在全局优化黑盒高成本函数方面提供了强大的框架。然而，由于默认高斯过程代理的可扩展性差，它在处理数据密集型问题，特别是在多目标设置中的能力有限。本文提出了一种新颖的贝叶斯优化框架，专为解决这些限制而设计。我们的方法利用了贝叶斯神经网络方法进行代理建模。这使得它能够有效地处理大批量数据，建模复杂问题以及产生预测的不确定性。此外，我们的方法结合了一种基于众所周知且易于部署的NSGA-II的可扩展的、具有不确定性的收购策略。这种完全可并行化的策略促进了未勘探区域的有效探索。我们的框架允许在最少迭代次数的情况下在数据密集环境中进行有效的优化。我们展示了我们方法的优越性。

    Bayesian optimization provides a powerful framework for global optimization of black-box, expensive-to-evaluate functions. However, it has a limited capacity in handling data-intensive problems, especially in multi-objective settings, due to the poor scalability of default Gaussian Process surrogates. We present a novel Bayesian optimization framework specifically tailored to address these limitations. Our method leverages a Bayesian neural networks approach for surrogate modeling. This enables efficient handling of large batches of data, modeling complex problems, and generating the uncertainty of the predictions. In addition, our method incorporates a scalable, uncertainty-aware acquisition strategy based on the well-known, easy-to-deploy NSGA-II. This fully parallelizable strategy promotes efficient exploration of uncharted regions. Our framework allows for effective optimization in data-intensive environments with a minimum number of iterations. We demonstrate the superiority of ou
    
[^12]: PESTS: 波斯语-英语跨语言语料库用于语义文本相似度

    PESTS: Persian_English Cross Lingual Corpus for Semantic Textual Similarity. (arXiv:2305.07893v1 [cs.CL])

    [http://arxiv.org/abs/2305.07893](http://arxiv.org/abs/2305.07893)

    本研究提出了跨语言的语义相似性模型PESTS，并通过波斯语-英语的跨语言语料库来验证模型的准确性。

    

    近来，语义文本相似度成为自然语言处理中备受关注的组件。在计算语言学和自然语言处理中，评估单词、短语、段落和文本之间的语义相似性很重要。同时，语义相似性度量要求在源和目标语言中提供具有一定语义相似性的句子对。许多跨语言的语义相似度模型使用机器翻译来弥补跨语言语料库不可用的不足，但机器翻译的误差会降低模型的准确性。然而，在使用语义相似度特征实现机器翻译时，用相同的机器翻译模型可以提高结果的准确性。

    One of the components of natural language processing that has received a lot of investigation recently is semantic textual similarity. In computational linguistics and natural language processing, assessing the semantic similarity of words, phrases, paragraphs, and texts is crucial. Calculating the degree of semantic resemblance between two textual pieces, paragraphs, or phrases provided in both monolingual and cross-lingual versions is known as semantic similarity. Cross lingual semantic similarity requires corpora in which there are sentence pairs in both the source and target languages with a degree of semantic similarity between them. Many existing cross lingual semantic similarity models use a machine translation due to the unavailability of cross lingual semantic similarity dataset, which the propagation of the machine translation error reduces the accuracy of the model. On the other hand, when we want to use semantic similarity features for machine translation the same machine t
    
[^13]: 基于分层生成对抗模拟学习的自动驾驶在城市环境中的应用

    Hierarchical Generative Adversarial Imitation Learning with Mid-level Input Generation for Autonomous Driving on Urban Environments. (arXiv:2302.04823v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04823](http://arxiv.org/abs/2302.04823)

    本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。

    

    对于现实中的城市导航场景，设计健壮的控制策略并不是一项简单的任务。在端到端的方法中，这些策略必须将车辆摄像头获得的高维图像映射到低级动作，如转向和油门。本研究提出了一种名为hGAIL的架构，用于解决车辆的自主导航问题，通过将感知信息直接映射到低级动作的同时，学习车辆环境的中级输入表示。

    Deriving robust control policies for realistic urban navigation scenarios is not a trivial task. In an end-to-end approach, these policies must map high-dimensional images from the vehicle's cameras to low-level actions such as steering and throttle. While pure Reinforcement Learning (RL) approaches are based exclusively on rewards,Generative Adversarial Imitation Learning (GAIL) agents learn from expert demonstrations while interacting with the environment, which favors GAIL on tasks for which a reward signal is difficult to derive. In this work, the hGAIL architecture was proposed to solve the autonomous navigation of a vehicle in an end-to-end approach, mapping sensory perceptions directly to low-level actions, while simultaneously learning mid-level input representations of the agent's environment. The proposed hGAIL consists of an hierarchical Adversarial Imitation Learning architecture composed of two main modules: the GAN (Generative Adversarial Nets) which generates the Bird's-
    

