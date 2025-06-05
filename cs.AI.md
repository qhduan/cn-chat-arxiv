# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PointCloud-Text Matching: Benchmark Datasets and a Baseline](https://arxiv.org/abs/2403.19386) | 本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。 |
| [^2] | [Large language models for crowd decision making based on prompt design strategies using ChatGPT: models, analysis and challenges](https://arxiv.org/abs/2403.15587) | 本文分析了基于提示设计策略的ChatGPT在群体决策过程中的应用，为提取意见和做出决策提供了新的可能性。 |
| [^3] | [T-TAME: Trainable Attention Mechanism for Explaining Convolutional Networks and Vision Transformers](https://arxiv.org/abs/2403.04523) | 本文提出了T-TAME，一种适用于卷积网络和视觉Transformer的可训练注意机制，为解释深度神经网络在图像分类任务中的应用提供了通用方法。 |
| [^4] | [From Reals to Logic and Back: Inventing Symbolic Vocabularies, Actions and Models for Planning from Raw Data](https://arxiv.org/abs/2402.11871) | 本文提出了一种从未标记高维实值机器人轨迹开始自主学习通用的逻辑相关表示，这些表示构成了自动发明的PDDL-like域模型。 |
| [^5] | [The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test](https://arxiv.org/abs/2402.11089) | 通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。 |
| [^6] | [Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization](https://arxiv.org/abs/2311.18703) | 该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。 |
| [^7] | [Performative Time-Series Forecasting.](http://arxiv.org/abs/2310.06077) | 本论文研究了时间序列预测中的展示性问题，提出了一种新的方法（FPS），通过利用延迟响应的概念来解决展示性引起的分布变化，并实现准确的预测。 |
| [^8] | [Label Deconvolution for Node Representation Learning on Large-scale Attributed Graphs against Learning Bias.](http://arxiv.org/abs/2309.14907) | 本文提出了一种标签解卷积技术(LD)，通过对图神经网络(GNNs)的逆映射进行高效的近似，来解决在大规模属性图上进行节点表示学习时的学习偏差挑战。 |
| [^9] | [FedJETs: Efficient Just-In-Time Personalization with Federated Mixture of Experts.](http://arxiv.org/abs/2306.08586) | 本论文提出了一种名为FedJETs的方法，使用联邦混合专家的框架，在联邦学习中实现高效及时的个性化。该方法通过训练专门的专家，并利用门控函数将输入路由到相关的专家，有效提高了模型的准确性。 |

# 详细

[^1]: PointCloud-Text匹配：基准数据集和一个基线

    PointCloud-Text Matching: Benchmark Datasets and a Baseline

    [https://arxiv.org/abs/2403.19386](https://arxiv.org/abs/2403.19386)

    本文提出了一个新的实例级检索任务：PointCloud-Text匹配（PTM），并构建了三个新的基准数据集以解决数据稀疏、文本模糊等挑战，同时提出了RoMa方法作为PTM的基线模型。

    

    在本文中，我们介绍和研究了一个新的实例级检索任务：PointCloud-Text Matching（PTM），旨在找到与给定的点云查询或文本查询匹配的确切跨模态实例。PTM可应用于各种场景，如室内/城市峡谷定位和场景检索。然而，在实践中尚无适用的、有针对性的PTM数据集。因此，我们构建了三个新的PTM基准数据集，分别为3D2T-SR、3D2T-NR和3D2T-QA。我们观察到数据具有挑战性，由于点云的稀疏、噪声或无序，以及文本的模糊、含糊或不完整，导致存在嘈杂的对应关系，使得现有的跨模态匹配方法对PTM无效。为了解决这些挑战，我们提出了一个PTM基线，命名为Robust PointCloud-Text Matching方法（RoMa）。RoMa包含两个模块：双重注意感知模块（DAP）和鲁棒负对比模块

    arXiv:2403.19386v1 Announce Type: cross  Abstract: In this paper, we present and study a new instance-level retrieval task: PointCloud-Text Matching~(PTM), which aims to find the exact cross-modal instance that matches a given point-cloud query or text query. PTM could be applied to various scenarios, such as indoor/urban-canyon localization and scene retrieval. However, there exists no suitable and targeted dataset for PTM in practice. Therefore, we construct three new PTM benchmark datasets, namely 3D2T-SR, 3D2T-NR, and 3D2T-QA. We observe that the data is challenging and with noisy correspondence due to the sparsity, noise, or disorder of point clouds and the ambiguity, vagueness, or incompleteness of texts, which make existing cross-modal matching methods ineffective for PTM. To tackle these challenges, we propose a PTM baseline, named Robust PointCloud-Text Matching method (RoMa). RoMa consists of two modules: a Dual Attention Perception module (DAP) and a Robust Negative Contrast
    
[^2]: 基于ChatGPT的提示设计策略的大型语言模型用于群体决策：模型、分析和挑战

    Large language models for crowd decision making based on prompt design strategies using ChatGPT: models, analysis and challenges

    [https://arxiv.org/abs/2403.15587](https://arxiv.org/abs/2403.15587)

    本文分析了基于提示设计策略的ChatGPT在群体决策过程中的应用，为提取意见和做出决策提供了新的可能性。

    

    社交媒体和互联网有潜力被利用作为丰富决策解决方案意见的来源。群体决策（CDM）是一种能够通过情感分析从纯文本（如社交媒体平台上发布的评论）中推断意见和决策的方法。本文分析了利用基于提示设计策略的ChatGPT来辅助CDM过程，以提取意见和做出决策。我们将ChatGPT整合到CDM过程中作为一种灵活的工具，推断出文本中表达的意见，并根据提示设计策略制定决策模型。

    arXiv:2403.15587v1 Announce Type: new  Abstract: Social Media and Internet have the potential to be exploited as a source of opinion to enrich Decision Making solutions. Crowd Decision Making (CDM) is a methodology able to infer opinions and decisions from plain texts, such as reviews published in social media platforms, by means of Sentiment Analysis. Currently, the emergence and potential of Large Language Models (LLMs) lead us to explore new scenarios of automatically understand written texts, also known as natural language processing. This paper analyzes the use of ChatGPT based on prompt design strategies to assist in CDM processes to extract opinions and make decisions. We integrate ChatGPT in CDM processes as a flexible tool that infer the opinions expressed in texts, providing numerical or linguistic evaluations where the decision making models are based on the prompt design strategies. We include a multi-criteria decision making scenario with a category ontology for criteria. 
    
[^3]: T-TAME：用于解释卷积网络和视觉Transformer的可训练注意机制

    T-TAME: Trainable Attention Mechanism for Explaining Convolutional Networks and Vision Transformers

    [https://arxiv.org/abs/2403.04523](https://arxiv.org/abs/2403.04523)

    本文提出了T-TAME，一种适用于卷积网络和视觉Transformer的可训练注意机制，为解释深度神经网络在图像分类任务中的应用提供了通用方法。

    

    Vision Transformers和其他用于图像分类任务的深度学习架构的发展和应用快速增长。然而，神经网络的“黑匣子”特性是在需要解释性的应用中采用的障碍。虽然已经提出了一些用于生成解释的技术，主要用于卷积神经网络，但是将这些技术适应到视觉Transformer的新范式是非平凡的。本文提出了T-TAME，Transformer兼容的可训练注意机制用于解释，这是一种说明用于图像分类任务中的深度神经网络的通用方法。所提出的架构和训练技术可以轻松应用于任何卷积或类似Vision Transformer的神经网络，使用精简的训练方法。训练后，解释图可以在单次前向传递中计算出；这些解释图可以与Convolutional Neural Networks中生成的解释图相媲美或者

    arXiv:2403.04523v1 Announce Type: cross  Abstract: The development and adoption of Vision Transformers and other deep-learning architectures for image classification tasks has been rapid. However, the "black box" nature of neural networks is a barrier to adoption in applications where explainability is essential. While some techniques for generating explanations have been proposed, primarily for Convolutional Neural Networks, adapting such techniques to the new paradigm of Vision Transformers is non-trivial. This paper presents T-TAME, Transformer-compatible Trainable Attention Mechanism for Explanations, a general methodology for explaining deep neural networks used in image classification tasks. The proposed architecture and training technique can be easily applied to any convolutional or Vision Transformer-like neural network, using a streamlined training approach. After training, explanation maps can be computed in a single forward pass; these explanation maps are comparable to or 
    
[^4]: 从实际到逻辑再到实际：为规划从原始数据中发明符号词汇、动作和模型

    From Reals to Logic and Back: Inventing Symbolic Vocabularies, Actions and Models for Planning from Raw Data

    [https://arxiv.org/abs/2402.11871](https://arxiv.org/abs/2402.11871)

    本文提出了一种从未标记高维实值机器人轨迹开始自主学习通用的逻辑相关表示，这些表示构成了自动发明的PDDL-like域模型。

    

    手工制作的基于逻辑的状态和动作表示已被广泛用于克服长期人工智能机器人规划问题的计算复杂性，包括任务和动作规划问题。但是，创建这样的表示需要具有强烈直觉和详细知识的专家，他们了解机器人和在特定环境中可能需要完成的任务。消除对人类直觉的依赖是一个极为活跃的研究领域。 本文提出了一种自主学习通用逻辑相关表示的方法，该表示从未标记的高维实值机器人轨迹开始。所学表示构成了自动发明的类PDDL域模型。确定性设置下的实证结果表明，仅从少数机器人轨迹中可以学到强大的抽象表示；所学关系

    arXiv:2402.11871v1 Announce Type: cross  Abstract: Hand-crafted, logic-based state and action representations have been widely used to overcome the intractable computational complexity of long-horizon robot planning problems, including task and motion planning problems. However, creating such representations requires experts with strong intuitions and detailed knowledge about the robot and the tasks it may need to accomplish in a given setting. Removing this dependency on human intuition is a highly active research area.   This paper presents the first approach for autonomously learning generalizable, logic-based relational representations for abstract states and actions starting from unannotated high-dimensional, real-valued robot trajectories. The learned representations constitute auto-invented PDDL-like domain models. Empirical results in deterministic settings show that powerful abstract representations can be learned from just a handful of robot trajectories; the learned relation
    
[^5]: 男性CEO和女性助理：通过成对刻板印象测试探究文本-图像模型中的性别偏见

    The Male CEO and the Female Assistant: Probing Gender Biases in Text-To-Image Models Through Paired Stereotype Test

    [https://arxiv.org/abs/2402.11089](https://arxiv.org/abs/2402.11089)

    通过成对刻板印象测试（PST）框架，在文本-图像模型中探究性别偏见，并评估了DALLE-3在性别职业和组织权力方面的偏见。

    

    最近大规模的文本到图像（T2I）模型（如DALLE-3）展示了在新应用中的巨大潜力，但也面临前所未有的公平挑战。先前的研究揭示了单人图像生成中的性别偏见，但T2I模型应用可能需要同时描绘两个或更多人。该设定中的潜在偏见仍未被探究，导致使用中的公平相关风险。为了研究T2I模型中性别偏见的基本方面，我们提出了一种新颖的成对刻板印象测试（PST）偏见评估框架。PST促使模型生成同一图像中的两个个体，用与相反性别刻板印象相关联的两个社会身份来描述他们。通过生成的图像遵从性别刻板印象的程度来衡量偏见。利用PST，我们从两个角度评估DALLE-3：性别职业中的偏见和组织权力中的偏见。

    arXiv:2402.11089v1 Announce Type: cross  Abstract: Recent large-scale Text-To-Image (T2I) models such as DALLE-3 demonstrate great potential in new applications, but also face unprecedented fairness challenges. Prior studies revealed gender biases in single-person image generation, but T2I model applications might require portraying two or more people simultaneously. Potential biases in this setting remain unexplored, leading to fairness-related risks in usage. To study these underlying facets of gender biases in T2I models, we propose a novel Paired Stereotype Test (PST) bias evaluation framework. PST prompts the model to generate two individuals in the same image. They are described with two social identities that are stereotypically associated with the opposite gender. Biases can then be measured by the level of conformation to gender stereotypes in generated images. Using PST, we evaluate DALLE-3 from 2 perspectives: biases in gendered occupation and biases in organizational power.
    
[^6]: 通过熵率最小化实现可预测的强化学习动态

    Predictable Reinforcement Learning Dynamics through Entropy Rate Minimization

    [https://arxiv.org/abs/2311.18703](https://arxiv.org/abs/2311.18703)

    该论文提出了一种名为PA-RL的方法，通过最小化熵率来引导强化学习智能体展现可预测的行为。研究展示了如何利用平均替代奖励实现确定性策略，并在动态模型的基础上近似计算值函数。

    

    在强化学习中，智能体没有动机展示可预测的行为，通常通过策略熵正则化推动智能体在探索上随机化其行为。从人的角度来看，这使得强化学习智能体很难解释和预测；从安全角度来看，更难以进行形式化验证。我们提出了一种新的方法，称为可预测性感知强化学习（PA-RL），用于引导智能体展现可预测的行为，其利用状态序列熵率作为可预测性度量。我们展示了如何将熵率制定为平均奖励目标，并且由于其熵奖励函数依赖于策略，我们引入了一个动作相关的替代熵，以利用PG方法。我们证明了最小化平均替代奖励的确定性策略存在，并且最小化了实际熵率。我们还展示了如何在学习到的动态模型的基础上近似计算与值函数。

    In Reinforcement Learning (RL), agents have no incentive to exhibit predictable behaviors, and are often pushed (through e.g. policy entropy regularization) to randomize their actions in favor of exploration. From a human perspective, this makes RL agents hard to interpret and predict, and from a safety perspective, even harder to formally verify. We propose a novel method to induce predictable behavior in RL agents, referred to as Predictability-Aware RL (PA-RL), which employs the state sequence entropy rate as a predictability measure. We show how the entropy rate can be formulated as an average reward objective, and since its entropy reward function is policy-dependent, we introduce an action-dependent surrogate entropy enabling the use of PG methods. We prove that deterministic policies minimizing the average surrogate reward exist and also minimize the actual entropy rate, and show how, given a learned dynamical model, we are able to approximate the value function associated to th
    
[^7]: 可展示的时间序列预测

    Performative Time-Series Forecasting. (arXiv:2310.06077v1 [cs.LG])

    [http://arxiv.org/abs/2310.06077](http://arxiv.org/abs/2310.06077)

    本论文研究了时间序列预测中的展示性问题，提出了一种新的方法（FPS），通过利用延迟响应的概念来解决展示性引起的分布变化，并实现准确的预测。

    

    时间序列预测是各个领域中的一个关键挑战，在近年来取得了实质性的进展。许多现实生活场景，如公共卫生、经济和社会应用，涉及到反馈循环，其中预测结果可能会影响到预测的结果，进而改变目标变量的分布。这种现象被称为展示性，引入了可能出现“自我抵消”或“自我实现”的预测的潜力。尽管在各个领域中对分类问题进行了广泛的研究，但展示性在机器学习视角下的时间序列预测问题尚未得到广泛探讨。在这篇论文中，我们对可展示的时间序列预测（PeTS）进行了形式化，解决了当可能存在展示性引起的分布变化时的准确预测挑战。我们提出了一种新颖方法，特征展示性转移（FPS），它利用延迟响应的概念来预测分布的变化和随后的变量。

    Time-series forecasting is a critical challenge in various domains and has witnessed substantial progress in recent years. Many real-life scenarios, such as public health, economics, and social applications, involve feedback loops where predictions can influence the predicted outcome, subsequently altering the target variable's distribution. This phenomenon, known as performativity, introduces the potential for 'self-negating' or 'self-fulfilling' predictions. Despite extensive studies in classification problems across domains, performativity remains largely unexplored in the context of time-series forecasting from a machine-learning perspective.  In this paper, we formalize performative time-series forecasting (PeTS), addressing the challenge of accurate predictions when performativity-induced distribution shifts are possible. We propose a novel approach, Feature Performative-Shifting (FPS), which leverages the concept of delayed response to anticipate distribution shifts and subseque
    
[^8]: 对大规模属性图上的节点表示学习进行标签解卷积以抵抗学习偏差的研究

    Label Deconvolution for Node Representation Learning on Large-scale Attributed Graphs against Learning Bias. (arXiv:2309.14907v1 [cs.LG])

    [http://arxiv.org/abs/2309.14907](http://arxiv.org/abs/2309.14907)

    本文提出了一种标签解卷积技术(LD)，通过对图神经网络(GNNs)的逆映射进行高效的近似，来解决在大规模属性图上进行节点表示学习时的学习偏差挑战。

    

    在带属性的图中，节点表示学习对许多重要的下游任务起着关键作用。为了同时编码属性和图结构，最近的研究将预训练模型与图神经网络(GNNs)进行整合，其中预训练模型作为节点编码器(NEs)来编码属性。由于在大规模图上同时训练大型NEs和GNNs存在严重的可伸缩性问题，许多方法提出了分别训练NEs和GNNs的方法。因此，在NEs的训练阶段中，他们没有考虑到GNNs中的特征卷积，导致了与联合训练相比的显著学习偏差。为了解决这个挑战，我们提出了一种高效的标签正则化技术，即标签解卷积(LD)，通过对GNNs的逆映射进行新颖且高度可伸缩的近似，以减轻学习偏差。

    Node representation learning on attributed graphs -- whose nodes are associated with rich attributes (e.g., texts and protein sequences) -- plays a crucial role in many important downstream tasks. To encode the attributes and graph structures simultaneously, recent studies integrate pre-trained models with graph neural networks (GNNs), where pre-trained models serve as node encoders (NEs) to encode the attributes. As jointly training large NEs and GNNs on large-scale graphs suffers from severe scalability issues, many methods propose to train NEs and GNNs separately. Consequently, they do not take feature convolutions in GNNs into consideration in the training phase of NEs, leading to a significant learning bias from that by the joint training. To address this challenge, we propose an efficient label regularization technique, namely Label Deconvolution (LD), to alleviate the learning bias by a novel and highly scalable approximation to the inverse mapping of GNNs. The inverse mapping l
    
[^9]: FedJETs：具有联邦混合专家的高效及时个性化方法

    FedJETs: Efficient Just-In-Time Personalization with Federated Mixture of Experts. (arXiv:2306.08586v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.08586](http://arxiv.org/abs/2306.08586)

    本论文提出了一种名为FedJETs的方法，使用联邦混合专家的框架，在联邦学习中实现高效及时的个性化。该方法通过训练专门的专家，并利用门控函数将输入路由到相关的专家，有效提高了模型的准确性。

    

    联邦学习（FL）的目标之一是创建能够适应每个参与客户端上下文的个性化模型，同时利用共享全局模型的知识。然而，通常情况下，个性化需要使用客户标记的数据进行微调以实现良好的性能，这在新来的客户端是不可行的，在隐私方面也存在问题。因此，如何在这些场景中实现及时个性化仍然是个未解决的问题。我们提出了FedJETs，这是一个在FL设置中使用“专家混合（MoE）”框架的新颖解决方案。我们的方法利用客户的多样性，在不同的类别子集上训练专门的专家，并利用一个门控函数将输入路由到最相关的专家。我们的门控函数利用预训练模型的共享专家的知识，以增强其即时的路由决策。值得一提的是，我们的方法能够将准确性提高高达18％，达到现有技术水平水平。

    One of the goals in Federated Learning (FL) is to create personalized models that can adapt to the context of each participating client, while utilizing knowledge from a shared global model. Yet, often, personalization requires a fine-tuning step using clients' labeled data in order to achieve good performance. This may not be feasible in scenarios where incoming clients are fresh and/or have privacy concerns. It, then, remains open how one can achieve just-in-time personalization in these scenarios. We propose FedJETs, a novel solution by using a Mixture-of-Experts (MoE) framework within a FL setup. Our method leverages the diversity of the clients to train specialized experts on different subsets of classes, and a gating function to route the input to the most relevant expert(s). Our gating function harnesses the knowledge of a pretrained model common expert to enhance its routing decisions on-the-fly. As a highlight, our approach can improve accuracy up to 18\% in state of the art F
    

