# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity](https://arxiv.org/abs/2403.13374) | 通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。 |
| [^2] | [Decentralized Federated Unlearning on Blockchain](https://arxiv.org/abs/2402.16294) | 提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。 |
| [^3] | [Overconfident and Unconfident AI Hinder Human-AI Collaboration](https://arxiv.org/abs/2402.07632) | 人工智能的过度自信和缺乏自信会阻碍人机协作，披露信心水平和提供反馈有助于认识到人工智能的信心不一致，但参与者往往因此不信任人工智能的建议，导致协作结果较差。 |
| [^4] | [Ocassionally Secure: A Comparative Analysis of Code Generation Assistants](https://arxiv.org/abs/2402.00689) | 本文通过比较分析四种先进的LLMs在9个任务上的表现，确定和理解了在真实场景中有效且安全地部署LLMs生成优质代码的条件和环境。 |
| [^5] | [Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs.](http://arxiv.org/abs/2310.11246) | Query2Triple提出了一种新的方法，将简单查询和复杂查询的训练解耦，通过预训练神经链接预测器来编码和回答复杂查询，提高了性能和训练效率。 |
| [^6] | [Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies.](http://arxiv.org/abs/2309.16025) | 本文介绍了一种名为符号化模仿学习（SIL）的方法，通过引入归纳逻辑编程（ILP）来学习从现有数据集中获取透明、可解释和泛化的驾驶策略。与传统的基于深度神经网络的模仿学习方法相比，SIL不仅提高了驾驶策略的可解释性，还显著改进了它们在各种驾驶情况下的适用性。 |
| [^7] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^8] | [Continual Dialogue State Tracking via Example-Guided Question Answering.](http://arxiv.org/abs/2305.13721) | 本文建议将对话状态跟踪重构为由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，获得持续的学习效益。通过结合简单的持续学习策略，可以在基准数据集上获得最先进的性能。 |
| [^9] | [Vision Langauge Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation.](http://arxiv.org/abs/2305.04474) | 本文提出了一种跨模态相似性逐步细化的对比学习策略，在视觉语言预训练中优化图像/文本锚点与其负样本文本/图像之间的互信息，有效应对了（部分）误反样本的挑战。 |
| [^10] | [Improving Multi-task Learning via Seeking Task-based Flat Regions.](http://arxiv.org/abs/2211.13723) | 通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。 |

# 详细

[^1]: 具有对数据异构性的自适应的拜占庭弹性联邦学习

    Byzantine-resilient Federated Learning With Adaptivity to Data Heterogeneity

    [https://arxiv.org/abs/2403.13374](https://arxiv.org/abs/2403.13374)

    通过提出新的Robust Average Gradient Algorithm（RAGA），本研究在联邦学习中解决了恶意拜占庭攻击和数据异构性的问题，实现了在非凸损失函数和异构数据集上的收敛性分析，并展示了RAGA的良好收敛性能。

    

    本文处理了在存在恶意拜占庭攻击和数据异构性的情况下的联邦学习（FL）。提出了一种新颖的鲁棒平均梯度算法（RAGA），该算法利用几何中位数进行聚合，并可以自由选择本地更新的轮数。与大多数现有的弹性方法不同，这些方法基于强凸损失函数或均匀分布的数据集进行收敛分析，我们进行了对强凸和非凸损失函数在异构数据集上的收敛分析。根据我们的理论分析，只要恶意用户数据集的比例小于一半，RAGA就可以以$\mathcal{O}({1}/{T^{2/3- \delta}})$的速度实现非凸损失函数的收敛，其中$T$为迭代次数，$\delta \in (0, 2/3)$，对于强凸损失函数则呈线性收敛。此外，稳定点或全局最优解

    arXiv:2403.13374v1 Announce Type: new  Abstract: This paper deals with federated learning (FL) in the presence of malicious Byzantine attacks and data heterogeneity. A novel Robust Average Gradient Algorithm (RAGA) is proposed, which leverages the geometric median for aggregation and can freely select the round number for local updating. Different from most existing resilient approaches, which perform convergence analysis based on strongly-convex loss function or homogeneously distributed dataset, we conduct convergence analysis for not only strongly-convex but also non-convex loss function over heterogeneous dataset. According to our theoretical analysis, as long as the fraction of dataset from malicious users is less than half, RAGA can achieve convergence at rate $\mathcal{O}({1}/{T^{2/3- \delta}})$ where $T$ is the iteration number and $\delta \in (0, 2/3)$ for non-convex loss function, and at linear rate for strongly-convex loss function. Moreover, stationary point or global optim
    
[^2]: 区块链上的去中心化联邦遗忘

    Decentralized Federated Unlearning on Blockchain

    [https://arxiv.org/abs/2402.16294](https://arxiv.org/abs/2402.16294)

    提出了基于区块链的联邦遗忘（BlockFUL），使用Chameleon Hash（CH）技术重新设计区块链结构，减少模型更新的复杂性和成本。

    

    区块链联邦学习（FL）在确保FL过程的完整性和可追溯性方面越来越受到关注。区块链FL涉及参与者在本地训练模型并随后将模型发布到区块链上，形成表示模型关系的类似有向无环图（DAG）的继承结构。然而，这种基于DAG的结构在使用敏感数据更新模型时存在挑战，因为涉及的复杂性和开销较大。为了解决这个问题，我们提出了基于区块链的联邦遗忘（BlockFUL），这是一个通用框架，使用变色龙哈希（CH）技术重新设计区块链结构，以减轻模型更新的复杂性，从而降低遗忘任务的计算和共识成本。此外，BlockFUL支持各种联邦遗忘方法，确保模型更新的完整性和可追溯性。

    arXiv:2402.16294v1 Announce Type: cross  Abstract: Blockchained Federated Learning (FL) has been gaining traction for ensuring the integrity and traceability of FL processes. Blockchained FL involves participants training models locally with their data and subsequently publishing the models on the blockchain, forming a Directed Acyclic Graph (DAG)-like inheritance structure that represents the model relationship. However, this particular DAG-based structure presents challenges in updating models with sensitive data, due to the complexity and overhead involved. To address this, we propose Blockchained Federated Unlearning (BlockFUL), a generic framework that redesigns the blockchain structure using Chameleon Hash (CH) technology to mitigate the complexity of model updating, thereby reducing the computational and consensus costs of unlearning tasks.Furthermore, BlockFUL supports various federated unlearning methods, ensuring the integrity and traceability of model updates, whether conduc
    
[^3]: 过于自信和缺乏自信的人工智能阻碍人机协作

    Overconfident and Unconfident AI Hinder Human-AI Collaboration

    [https://arxiv.org/abs/2402.07632](https://arxiv.org/abs/2402.07632)

    人工智能的过度自信和缺乏自信会阻碍人机协作，披露信心水平和提供反馈有助于认识到人工智能的信心不一致，但参与者往往因此不信任人工智能的建议，导致协作结果较差。

    

    随着人工智能的进步，人机协作在专业和日常场景中越来越普遍。在这种协作中，人工智能可以表达其对自己表现的信心水平，作为人类评估人工智能建议的重要指标。然而，人工智能可能表现出过度自信或缺乏自信，即其表达的信心高于或低于其实际表现，这可能导致人们错误地评估人工智能的建议。我们的研究调查了人工智能过度自信和缺乏自信对人类信任、接受人工智能建议和协作结果的影响。我们的研究发现，披露人工智能的信心水平和表现反馈有助于更好地认识人工智能信心不一致。然而，参与者往往会因为察觉到这种不一致而不信任人工智能的建议，导致拒绝人工智能的建议，并且在协作任务中表现更差。相反，没有这些提示的情况下，参与者更容易信任人工智能并接受其建议，从而在协作任务中表现更好。

    As artificial intelligence (AI) advances, human-AI collaboration has become increasingly prevalent across both professional and everyday settings. In such collaboration, AI can express its confidence level about its performance, serving as a crucial indicator for humans to evaluate AI's suggestions. However, AI may exhibit overconfidence or underconfidence--its expressed confidence is higher or lower than its actual performance--which may lead humans to mistakenly evaluate AI advice. Our study investigates the influences of AI's overconfidence and underconfidence on human trust, their acceptance of AI suggestions, and collaboration outcomes. Our study reveal that disclosing AI confidence levels and performance feedback facilitates better recognition of AI confidence misalignments. However, participants tend to withhold their trust as perceiving such misalignments, leading to a rejection of AI suggestions and subsequently poorer performance in collaborative tasks. Conversely, without su
    
[^4]: 偶尔安全：代码生成辅助工具的比较分析

    Ocassionally Secure: A Comparative Analysis of Code Generation Assistants

    [https://arxiv.org/abs/2402.00689](https://arxiv.org/abs/2402.00689)

    本文通过比较分析四种先进的LLMs在9个任务上的表现，确定和理解了在真实场景中有效且安全地部署LLMs生成优质代码的条件和环境。

    

    大型语言模型(LLMs)在各种应用中的应用越来越广泛，代码生成就是一个显著的例子。以往的研究表明LLMs有能力生成安全和不安全的代码，但文献没有考虑到什么因素有助于生成安全和有效的代码。因此，本文重点是确定和理解在真实场景中LLMs能够有效和安全地部署来生成优质代码的条件和环境。我们对四个先进的LLMs进行了比较分析——使用ChatGPT和Bard的GPT-3.5和GPT-4，以及来自Google的Gemini——使用9个独立任务来评估每个模型的代码生成能力。我们将我们的研究置于一个典型的使用场景中，代表了开发人员在工作中使用LLMs进行日常任务的情况。此外，我们还强调了安全意识，通过使用我们开发的两个不同版本的工具来体现。

    $ $Large Language Models (LLMs) are being increasingly utilized in various applications, with code generations being a notable example. While previous research has shown that LLMs have the capability to generate both secure and insecure code, the literature does not take into account what factors help generate secure and effective code. Therefore in this paper we focus on identifying and understanding the conditions and contexts in which LLMs can be effectively and safely deployed in real-world scenarios to generate quality code. We conducted a comparative analysis of four advanced LLMs--GPT-3.5 and GPT-4 using ChatGPT and Bard and Gemini from Google--using 9 separate tasks to assess each model's code generation capabilities. We contextualized our study to represent the typical use cases of a real-life developer employing LLMs for everyday tasks as work. Additionally, we place an emphasis on security awareness which is represented through the use of two distinct versions of our develop
    
[^5]: Query2Triple: 统一查询编码以回答知识图谱上多样复杂查询的挑战

    Query2Triple: Unified Query Encoding for Answering Diverse Complex Queries over Knowledge Graphs. (arXiv:2310.11246v1 [cs.AI])

    [http://arxiv.org/abs/2310.11246](http://arxiv.org/abs/2310.11246)

    Query2Triple提出了一种新的方法，将简单查询和复杂查询的训练解耦，通过预训练神经链接预测器来编码和回答复杂查询，提高了性能和训练效率。

    

    复杂查询回答（CQA）是知识图谱（KG）的一项挑战任务。由于KG的不完整性，已经提出了查询嵌入（QE）方法，将查询和实体编码到相同的嵌入空间中，并将逻辑运算符视为神经集合运算符，以获得答案。然而，这些方法在同时对简单（一跳）和复杂（多跳和逻辑）查询进行训练时，会导致简单查询性能的下降和训练效率低下。在本文中，我们提出了Query to Triple（Q2T），一种新颖的方法，将简单和复杂查询的训练解耦。Q2T将训练分为两个阶段：（1）在简单查询上预训练神经链接预测器，以基于头实体和关系预测尾实体。（2）在复杂查询上训练查询编码器，将多样的复杂查询编码为统一的三元组形式，可以通过预训练的神经链接预测器高效地解决。

    Complex Query Answering (CQA) is a challenge task of Knowledge Graph (KG). Due to the incompleteness of KGs, query embedding (QE) methods have been proposed to encode queries and entities into the same embedding space, and treat logical operators as neural set operators to obtain answers. However, these methods train KG embeddings and neural set operators concurrently on both simple (one-hop) and complex (multi-hop and logical) queries, which causes performance degradation on simple queries and low training efficiency. In this paper, we propose Query to Triple (Q2T), a novel approach that decouples the training for simple and complex queries. Q2T divides the training into two stages: (1) Pre-training a neural link predictor on simple queries to predict tail entities based on the head entity and relation. (2) Training a query encoder on complex queries to encode diverse complex queries into a unified triple form that can be efficiently solved by the pretrained neural link predictor. Our
    
[^6]: 符号化模仿学习：从黑盒到可解释的驾驶策略

    Symbolic Imitation Learning: From Black-Box to Explainable Driving Policies. (arXiv:2309.16025v1 [cs.LG])

    [http://arxiv.org/abs/2309.16025](http://arxiv.org/abs/2309.16025)

    本文介绍了一种名为符号化模仿学习（SIL）的方法，通过引入归纳逻辑编程（ILP）来学习从现有数据集中获取透明、可解释和泛化的驾驶策略。与传统的基于深度神经网络的模仿学习方法相比，SIL不仅提高了驾驶策略的可解释性，还显著改进了它们在各种驾驶情况下的适用性。

    

    当前的模仿学习方法主要基于深度神经网络，提供了从现实世界数据中获取驾驶策略的有效手段，但在可解释性和泛化性方面存在显著局限性。这些缺点在自动驾驶等安全关键应用中尤为令人担忧。本文通过引入符号化模仿学习（SIL），一种使用归纳逻辑编程（ILP）学习从可用数据集中获取透明、可解释和泛化的驾驶策略的创新方法，来解决这些局限性。利用真实世界的highD数据集，我们对我们的方法进行了严格的比较分析，与当前的基于神经网络的模仿学习方法进行了对比。我们的结果表明，SIL不仅提高了驾驶策略的可解释性，还显著提高了它们在各种驾驶情况下的适用性。因此，这项工作为实现更可靠和可解释的驾驶策略打开了一条新的途径。

    Current methods of imitation learning (IL), primarily based on deep neural networks, offer efficient means for obtaining driving policies from real-world data but suffer from significant limitations in interpretability and generalizability. These shortcomings are particularly concerning in safety-critical applications like autonomous driving. In this paper, we address these limitations by introducing Symbolic Imitation Learning (SIL), a groundbreaking method that employs Inductive Logic Programming (ILP) to learn driving policies which are transparent, explainable and generalisable from available datasets. Utilizing the real-world highD dataset, we subject our method to a rigorous comparative analysis against prevailing neural-network-based IL methods. Our results demonstrate that SIL not only enhances the interpretability of driving policies but also significantly improves their applicability across varied driving situations. Hence, this work offers a novel pathway to more reliable an
    
[^7]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^8]: 基于示例引导问答的持续对话状态跟踪

    Continual Dialogue State Tracking via Example-Guided Question Answering. (arXiv:2305.13721v1 [cs.CL])

    [http://arxiv.org/abs/2305.13721](http://arxiv.org/abs/2305.13721)

    本文建议将对话状态跟踪重构为由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，获得持续的学习效益。通过结合简单的持续学习策略，可以在基准数据集上获得最先进的性能。

    

    对话系统需要不断更新以适应新服务，但是简单地使用新服务的数据进行训练会降低先前学习的服务的性能。本文发现，对话状态跟踪(DST)是一个简单的自然语言理解任务，我们建议将其重构为一组由例子引导的粒度问题回答任务，以最小化服务之间的任务转移，从而获得持续的学习效益。我们的方法可以减轻特定服务的记忆负担，并教会模型将所给问题和示例用于从对话中提取必要信息。我们发现，一个只有6000万个参数的模型可以通过学习从检索器获取的上下文示例获得巨大的提升。将我们的方法与简单的持续学习策略相结合，可以在基准数据集上获得最先进的性能，证明了我们方法的有效性。

    Dialogue systems are frequently updated to accommodate new services, but naively updating them by continually training with data for new services in diminishing performance on previously learnt services. Motivated by the insight that dialogue state tracking (DST), a crucial component of dialogue systems that estimates the user's goal as a conversation proceeds, is a simple natural language understanding task, we propose reformulating it as a bundle of granular example-guided question answering tasks to minimize the task shift between services and thus benefit continual learning. Our approach alleviates service-specific memorization and teaches a model to contextualize the given question and example to extract the necessary information from the conversation. We find that a model with just 60M parameters can achieve a significant boost by learning to learn from in-context examples retrieved by a retriever trained to identify turns with similar dialogue state changes. Combining our method
    
[^9]: 跨模态相似性调节的对比学习在视觉语言预训练中的应用

    Vision Langauge Pre-training by Contrastive Learning with Cross-Modal Similarity Regulation. (arXiv:2305.04474v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2305.04474](http://arxiv.org/abs/2305.04474)

    本文提出了一种跨模态相似性逐步细化的对比学习策略，在视觉语言预训练中优化图像/文本锚点与其负样本文本/图像之间的互信息，有效应对了（部分）误反样本的挑战。

    

    在视觉语言预训练中，跨模态对比学习面临着（部分）误反样本的挑战。本文从 mutual information 优化的角度研究了这个问题。我们理论上证明了在存在噪声的情况下，涉及到负样本的互信息也很重要。我们提出了一种跨模态相似性逐步细化的对比学习策略，以更加精确地优化图像/文本锚点与其负样本文本/图像之间的互信息。我们的方法在四个下游跨模态任务上表现出竞争力，并在理论指导下系统地平衡了（部分）误反样本的有益影响和有害影响。

    Cross-modal contrastive learning in vision language pretraining (VLP) faces the challenge of (partial) false negatives. In this paper, we study this problem from the perspective of Mutual Information (MI) optimization. It is common sense that InfoNCE loss used in contrastive learning will maximize the lower bound of MI between anchors and their positives, while we theoretically prove that MI involving negatives also matters when noises commonly exist. Guided by a more general lower bound form for optimization, we propose a contrastive learning strategy regulated by progressively refined cross-modal similarity, to more accurately optimize MI between an image/text anchor and its negative texts/images instead of improperly minimizing it. Our method performs competitively on four downstream cross-modal tasks and systematically balances the beneficial and harmful effects of (partial) false negative samples under theoretical guidance.
    
[^10]: 通过寻找基于任务的平坦区域来改进多任务学习

    Improving Multi-task Learning via Seeking Task-based Flat Regions. (arXiv:2211.13723v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.13723](http://arxiv.org/abs/2211.13723)

    通过寻找基于任务的平坦区域，可以改进多任务学习并提高模型性能，但需要正确使用正则化技术以避免次优解。

    

    多任务学习（MTL）是一种广泛使用且强大的学习范式，用于训练深度神经网络，可以通过单个骨干学习多个目标。与单独训练任务相比，MTL显着降低了计算成本，提高了数据效率，并通过利用任务之间的知识来潜在地提高模型性能。因此，它已经被应用于各种应用领域，从计算机视觉到自然语言处理和语音识别。其中，MTL的一个新兴研究方向集中在操纵任务梯度以推导出对所有任务有益的最终梯度下降方向。尽管在许多基准测试上取得了令人印象深刻的结果，但是在实际问题上直接应用这些方法而不使用适当的正则化技术可能会导致次优解。特别是，标准训练在训练数据上最小化经验损失，很容易遭受过拟合问题。

    Multi-Task Learning (MTL) is a widely-used and powerful learning paradigm for training deep neural networks that allows learning more than one objective by a single backbone. Compared to training tasks separately, MTL significantly reduces computational costs, improves data efficiency, and potentially enhances model performance by leveraging knowledge across tasks. Hence, it has been adopted in a variety of applications, ranging from computer vision to natural language processing and speech recognition. Among them, there is an emerging line of work in MTL that focuses on manipulating the task gradient to derive an ultimate gradient descent direction to benefit all tasks. Despite achieving impressive results on many benchmarks, directly applying these approaches without using appropriate regularization techniques might lead to suboptimal solutions on real-world problems. In particular, standard training that minimizes the empirical loss on the training data can easily suffer from overfi
    

