# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding Emergent Abilities of Language Models from the Loss Perspective](https://arxiv.org/abs/2403.15796) | 本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。 |
| [^2] | [Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving](https://arxiv.org/abs/2403.12176) | 自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。 |
| [^3] | [Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation](https://arxiv.org/abs/2403.10700) | 提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性 |
| [^4] | [The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework](https://arxiv.org/abs/2403.06832) | 提出了一种利用噪声掩模的Transformer-based架构SNAG方法，实现了多模态知识图表示中实体嵌入的最先进性能 |
| [^5] | [Do Large Language Models Mirror Cognitive Language Processing?](https://arxiv.org/abs/2402.18023) | 本文提出了一种新颖方法，通过将大型语言模型（LLMs）的表示与人类认知信号联系起来，评估LLMs模拟认知语言处理的效果。 |
| [^6] | [SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection](https://arxiv.org/abs/2402.16705) | SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。 |
| [^7] | [Learning Optimal Tax Design in Nonatomic Congestion Games](https://arxiv.org/abs/2402.07437) | 本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。 |
| [^8] | [SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks.](http://arxiv.org/abs/2401.15299) | SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。 |
| [^9] | [Machine unlearning through fine-grained model parameters perturbation.](http://arxiv.org/abs/2401.04385) | 本文提出了一种精细的机器去学习策略，通过细粒度模型参数的扰动来实现用户隐私保护，同时保持可控的计算成本。采用遗忘率和记忆保留率等新的指标来评估去学习效果和模型泛化能力。 |
| [^10] | [A performance characteristic curve for model evaluation: the application in information diffusion prediction.](http://arxiv.org/abs/2309.09537) | 本研究提出了一种模型的性能特征曲线，用于评估其在不同复杂度任务中的表现。通过使用基于信息熵的度量方法，我们确定了随机性与模型预测准确性之间的关系，并发现不同条件下的数据点都可以合并成一条曲线，捕捉了模型在面对不确定性时的正确预测能力。 |
| [^11] | [NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking.](http://arxiv.org/abs/2304.04640) | NeuroBench是由学术界和工业界成员共同开发的一套协作、公平和代表性的基准测试，可以解决神经形态计算中缺乏清晰标准的问题，推动该领域的发展。 |
| [^12] | [Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models.](http://arxiv.org/abs/2304.03271) | 本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。 |

# 详细

[^1]: 从损失角度理解语言模型的突现能力

    Understanding Emergent Abilities of Language Models from the Loss Perspective

    [https://arxiv.org/abs/2403.15796](https://arxiv.org/abs/2403.15796)

    本文从损失角度重新定义了语言模型的突现能力，发现具有相同预训练损失的模型在不同任务上表现相似，而当预训练损失低于特定阈值时，模型将展现出突现能力。

    

    近期研究质疑了传统认为语言模型的突现能力仅存在于大模型中的观点。这种怀疑源自两点观察：1）较小的模型也能展现出对突现能力的高性能；2）质疑用于测量这些能力的不连续性指标。本文提议从预训练损失的角度研究突现能力，而非模型大小或训练计算。我们展示了具有相同预训练损失但不同模型和数据大小的模型，在各种下游任务上表现相同。我们还发现，当某一模型的预训练损失低于特定阈值时，在某些任务上表现出突现能力，而不论指标的连续性如何；而在达到该阈值之前，其性能仍保持在随机猜测水平。这启发我们重新定义突现能力为那些......

    arXiv:2403.15796v1 Announce Type: cross  Abstract: Recent studies have put into question the belief that emergent abilities in language models are exclusive to large models. This skepticism arises from two observations: 1) smaller models can also exhibit high performance on emergent abilities and 2) there is doubt on the discontinuous metrics used to measure these abilities. In this paper, we propose to study emergent abilities in the lens of pre-training loss, instead of model size or training compute. We demonstrate that the models with the same pre-training loss, but different model and data sizes, generate the same performance on various downstream tasks. We also discover that a model exhibits emergent abilities on certain tasks -- regardless of the continuity of metrics -- when its pre-training loss falls below a specific threshold. Before reaching this threshold, its performance remains at the level of random guessing. This inspires us to redefine emergent abilities as those that
    
[^2]: 自动驾驶中可解释人工智能的安全影响

    Safety Implications of Explainable Artificial Intelligence in End-to-End Autonomous Driving

    [https://arxiv.org/abs/2403.12176](https://arxiv.org/abs/2403.12176)

    自动驾驶中可解释人工智能的安全影响对于确保车辆自动化安全至关重要，但当前研究中安全性和可解释性方面往往被分开砠。

    

    末端到末端学习管道正在逐渐改变高度自主车辆的持续发展，这主要归功于深度学习的进步、大规模训练数据集的可用性以及综合传感器设备的改进。然而，当代学习方法在实时决策中缺乏可解释性，妨碍了用户的信任，并减弱了这类车辆的广泛部署和商业化。此外，当这些汽车参与或导致交通事故时，问题会变得更加严重。这种缺点从社会和法律的角度引起了严重的安全担忧。因此，在末端到末端自动驾驶中解释性是促进车辆自动化安全的关键。然而，当今最先进技术中研究人员通常将自动驾驶的安全性和可解释性方面分开研究。在本文中，我们旨在弥合这一差距

    arXiv:2403.12176v1 Announce Type: cross  Abstract: The end-to-end learning pipeline is gradually creating a paradigm shift in the ongoing development of highly autonomous vehicles, largely due to advances in deep learning, the availability of large-scale training datasets, and improvements in integrated sensor devices. However, a lack of interpretability in real-time decisions with contemporary learning methods impedes user trust and attenuates the widespread deployment and commercialization of such vehicles. Moreover, the issue is exacerbated when these cars are involved in or cause traffic accidents. Such drawback raises serious safety concerns from societal and legal perspectives. Consequently, explainability in end-to-end autonomous driving is essential to enable the safety of vehicular automation. However, the safety and explainability aspects of autonomous driving have generally been investigated disjointly by researchers in today's state of the art. In this paper, we aim to brid
    
[^3]: 注意错误！检测和定位视觉与语言导航中的指令错误

    Mind the Error! Detection and Localization of Instruction Errors in Vision-and-Language Navigation

    [https://arxiv.org/abs/2403.10700](https://arxiv.org/abs/2403.10700)

    提出了一个新的基准数据集，首次引入了各种类型的指令错误，考虑到潜在的人类原因，以评估连续环境中 VLN 系统的健壮性

    

    Vision-and-Language Navigation in Continuous Environments (VLN-CE) 是一项直观且具有挑战性的体验智能任务。代理人被要求通过执行一系列低级动作、遵循一系列自然语言指令来导航到目标目标。所有文献中的 VLN-CE 方法都假设语言指令是准确的。然而，在实践中，人类给出的指令可能由于不准确的记忆或混淆而包含空间环境描述中的错误。当前 VLN-CE 基准没有解决这种情况，使得 VLN-CE 中的最新方法在面对来自人类用户的错误指令时变得脆弱。我们首次提出了一个引入各种类型指令错误考虑潜在人类原因的新型基准数据集。该基准数据集为连续环境中的 VLN 系统的健壮性提供了宝贵的见解。我们观察到 noticeable...

    arXiv:2403.10700v1 Announce Type: cross  Abstract: Vision-and-Language Navigation in Continuous Environments (VLN-CE) is one of the most intuitive yet challenging embodied AI tasks. Agents are tasked to navigate towards a target goal by executing a set of low-level actions, following a series of natural language instructions. All VLN-CE methods in the literature assume that language instructions are exact. However, in practice, instructions given by humans can contain errors when describing a spatial environment due to inaccurate memory or confusion. Current VLN-CE benchmarks do not address this scenario, making the state-of-the-art methods in VLN-CE fragile in the presence of erroneous instructions from human users. For the first time, we propose a novel benchmark dataset that introduces various types of instruction errors considering potential human causes. This benchmark provides valuable insight into the robustness of VLN systems in continuous environments. We observe a noticeable 
    
[^4]: 噪声的力量：朝着统一的多模态知识图表示框架

    The Power of Noise: Toward a Unified Multi-modal Knowledge Graph Representation Framework

    [https://arxiv.org/abs/2403.06832](https://arxiv.org/abs/2403.06832)

    提出了一种利用噪声掩模的Transformer-based架构SNAG方法，实现了多模态知识图表示中实体嵌入的最先进性能

    

    多模态预训练的进展凸显出鲁棒的多模态知识图（MMKG）表示学习框架的必要性。此框架对于在规模上将结构化知识整合到多模态大型语言模型（LLMs）中至关重要，旨在减轻知识误解和多模态幻觉等问题。在这项工作中，为了评估模型准确嵌入MMKG中的实体的能力，我们专注于两个广泛研究的任务：多模态知识图完成（MKGC）和多模态实体对齐（MMEA）。在此基础上，我们提出了一种新颖的SNAG方法，该方法利用基于Transformer的架构，并配备了模态级噪声掩模，以在知识图中鲁棒地集成多模态实体特征。通过为MKGC和MMEA都引入特定的训练目标，我们的方法在总共十个数据集上（三个用于MKGC和...

    arXiv:2403.06832v1 Announce Type: cross  Abstract: The advancement of Multi-modal Pre-training highlights the necessity for a robust Multi-Modal Knowledge Graph (MMKG) representation learning framework. This framework is crucial for integrating structured knowledge into multi-modal Large Language Models (LLMs) at scale, aiming to alleviate issues like knowledge misconceptions and multi-modal hallucinations. In this work, to evaluate models' ability to accurately embed entities within MMKGs, we focus on two widely researched tasks: Multi-modal Knowledge Graph Completion (MKGC) and Multi-modal Entity Alignment (MMEA). Building on this foundation, we propose a novel SNAG method that utilizes a Transformer-based architecture equipped with modality-level noise masking for the robust integration of multi-modal entity features in KGs. By incorporating specific training objectives for both MKGC and MMEA, our approach achieves SOTA performance across a total of ten datasets (three for MKGC and 
    
[^5]: 大型语言模型是否反映认知语言处理？

    Do Large Language Models Mirror Cognitive Language Processing?

    [https://arxiv.org/abs/2402.18023](https://arxiv.org/abs/2402.18023)

    本文提出了一种新颖方法，通过将大型语言模型（LLMs）的表示与人类认知信号联系起来，评估LLMs模拟认知语言处理的效果。

    

    大型语言模型（LLMs）在文本理解和逻辑推理方面展现出卓越能力，甚至在许多认知任务中实现甚至超越人类水平的表现。由于LLMs是从人类语言认知的大量文本产出中训练出来的，自然而然地会问LLMs是否反映认知语言处理，或LLMs在多大程度上类似于认知语言处理。本文提出了一种新颖的方法，用于连接LLMs表征和人类认知信号，以评估LLMs如何有效地模拟认知语言处理。我们采用表征相似性分析（RSA）来衡量16种主流LLMs与大脑fMRI信号之间的对齐程度。我们在实验中探讨了各种因素（例如模型规模、对齐训练、指导附加）对LLM-大脑对齐的影响。实验结果表明，模型规模与正相关

    arXiv:2402.18023v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated remarkable capabilities in text comprehension and logical reasoning, achiving or even surpassing human-level performance in numerous cognition tasks. As LLMs are trained from massive textual outputs of human language cognition, it is natural to ask whether LLMs mirror cognitive language processing. Or to what extend LLMs resemble cognitive language processing? In this paper, we propose a novel method that bridge between LLM representations and human cognition signals to evaluate how effectively LLMs simulate cognitive language processing. We employ Representational Similarity Analysis (RSA) to mearsure the alignment between 16 mainstream LLMs and fMRI signals of the brain. We empirically investigate the impact of a variety of factors (e.g., model scaling, alignment training, instruction appending) on such LLM-brain alignment. Experimental results indicate that model scaling is positively cor
    
[^6]: SelectIT: 通过基于不确定性的自我反思实现大型语言模型的选择性指导调整

    SelectIT: Selective Instruction Tuning for Large Language Models via Uncertainty-Aware Self-Reflection

    [https://arxiv.org/abs/2402.16705](https://arxiv.org/abs/2402.16705)

    SelectIT通过利用大型语言模型本身的能力和基于不确定性的方法，提出了一种无需额外资源的高效选择指导调整数据集的方法，进而提升了模型的能力。

    

    指导调整（IT）对于调整大型语言模型（LLMs）以适应人类中心交互至关重要。最近的进展表明，精心选择一小部分高质量的IT数据可以显着提高LLMs的性能。尽管如此，常见方法通常依赖于额外的模型或数据集，这增加了成本并限制了广泛采用。在这项工作中，我们提出了一种新颖的方法，称为SelectIT，它利用LLM本身的基本能力。具体来说，我们利用LLMs中固有的不确定性，更有效地选择高质量的IT数据，而无需额外资源。此外，我们介绍了一种新颖的IT数据集，名为选择性羊驼（Selective Alpaca），通过将SelectIT应用于Alpaca-GPT4数据集而创建。实证结果表明，使用选择性羊驼进行IT可以极大地提升模型性能。SelectIT的稳健性也得到了验证。

    arXiv:2402.16705v1 Announce Type: new  Abstract: Instruction tuning (IT) is crucial to tailoring large language models (LLMs) towards human-centric interactions. Recent advancements have shown that the careful selection of a small, high-quality subset of IT data can significantly enhance the performance of LLMs. Despite this, common approaches often rely on additional models or data sets, which increases costs and limits widespread adoption. In this work, we propose a novel approach, termed SelectIT, that capitalizes on the foundational capabilities of the LLM itself. Specifically, we exploit the intrinsic uncertainty present in LLMs to more effectively select high-quality IT data, without the need for extra resources. Furthermore, we introduce a novel IT dataset, the Selective Alpaca, created by applying SelectIT to the Alpaca-GPT4 dataset. Empirical results demonstrate that IT using Selective Alpaca leads to substantial model ability enhancement. The robustness of SelectIT has also b
    
[^7]: 非原子拥堵博弈中学习最优税收设计

    Learning Optimal Tax Design in Nonatomic Congestion Games

    [https://arxiv.org/abs/2402.07437](https://arxiv.org/abs/2402.07437)

    本研究致力于学习如何设计最优税收，以在非原子拥堵博弈中提高效率。为了解决指数级的税收函数空间、梯度不存在和目标函数的非凸性等挑战，该算法利用了分段线性税收、额外的线性项和有效的子例程的新颖组成部分。

    

    本研究探讨了如何学习最优税收设计，以在非原子拥堵博弈中最大化效率。众所周知，玩家之间的自利行为可能会破坏系统的效率。税务机制是缓解此问题并引导社会最优行为的常见方法。在这项工作中，我们首次采取了学习最优税收的初始步骤，该最优税收可以通过平衡反馈来最小化社会成本，即税务设计者只能观察到强制税收下的均衡状态。由于指数级的税收函数空间，梯度不存在和目标函数的非凸性，现有算法不适用。为了解决这些挑战，我们的算法利用了几个新颖的组成部分：（1）分段线性税收来近似最优税收；（2）额外的线性项来保证强凸潜力函数；（3）有效的子例程来找到“边界”税收。该算法可以找到一个$\epsilon$-最优税收，时间复杂度为$O(\bet

    We study how to learn the optimal tax design to maximize the efficiency in nonatomic congestion games. It is known that self-interested behavior among the players can damage the system's efficiency. Tax mechanisms is a common method to alleviate this issue and induce socially optimal behavior. In this work, we take the initial step for learning the optimal tax that can minimize the social cost with \emph{equilibrium feedback}, i.e., the tax designer can only observe the equilibrium state under the enforced tax. Existing algorithms are not applicable due to the exponentially large tax function space, nonexistence of the gradient, and nonconvexity of the objective. To tackle these challenges, our algorithm leverages several novel components: (1) piece-wise linear tax to approximate the optimal tax; (2) an extra linear term to guarantee a strongly convex potential function; (3) efficient subroutine to find the ``boundary'' tax. The algorithm can find an $\epsilon$-optimal tax with $O(\bet
    
[^8]: SupplyGraph: 使用图神经网络进行供应链规划的基准数据集

    SupplyGraph: A Benchmark Dataset for Supply Chain Planning using Graph Neural Networks. (arXiv:2401.15299v1 [cs.LG])

    [http://arxiv.org/abs/2401.15299](http://arxiv.org/abs/2401.15299)

    SupplyGraph是一个基准数据集，用于使用图神经网络进行供应链规划。该数据集包含了来自孟加拉国一家领先快速消费品公司的实际数据，用于优化、预测和解决供应链问题。数据集中的时间数据作为节点特征，可用于销售预测、生产计划和故障识别。

    

    图神经网络（GNNs）在不同领域如运输、生物信息学、语言处理和计算机视觉中取得了重要进展。然而，在将GNNs应用于供应链网络方面，目前尚缺乏研究。供应链网络在结构上类似于图形，使其成为应用GNN方法的理想选择。这为优化、预测和解决供应链问题开辟了无限可能。然而，此方法的一个主要障碍在于缺乏真实世界的基准数据集以促进使用GNN来研究和解决供应链问题。为了解决这个问题，我们提供了一个来自孟加拉国一家领先的快速消费品公司的实际基准数据集，该数据集侧重于用于生产目的的供应链规划的时间任务。该数据集包括时间数据作为节点特征，以实现销售预测、生产计划和故障识别。

    Graph Neural Networks (GNNs) have gained traction across different domains such as transportation, bio-informatics, language processing, and computer vision. However, there is a noticeable absence of research on applying GNNs to supply chain networks. Supply chain networks are inherently graph-like in structure, making them prime candidates for applying GNN methodologies. This opens up a world of possibilities for optimizing, predicting, and solving even the most complex supply chain problems. A major setback in this approach lies in the absence of real-world benchmark datasets to facilitate the research and resolution of supply chain problems using GNNs. To address the issue, we present a real-world benchmark dataset for temporal tasks, obtained from one of the leading FMCG companies in Bangladesh, focusing on supply chain planning for production purposes. The dataset includes temporal data as node features to enable sales predictions, production planning, and the identification of fa
    
[^9]: 通过细粒度模型参数扰动实现机器去学习

    Machine unlearning through fine-grained model parameters perturbation. (arXiv:2401.04385v1 [cs.LG])

    [http://arxiv.org/abs/2401.04385](http://arxiv.org/abs/2401.04385)

    本文提出了一种精细的机器去学习策略，通过细粒度模型参数的扰动来实现用户隐私保护，同时保持可控的计算成本。采用遗忘率和记忆保留率等新的指标来评估去学习效果和模型泛化能力。

    

    机器去学习技术涉及到撤销数据记录和减小该数据对训练模型的影响，从而帮助实现用户隐私保护目标，但会带来显著的计算成本。基于参数扰动的权重去学习是一种通用方法，但通常涉及到全局修改参数。我们提出了精细的Top-K和Random-k参数扰动不精确机器去学习策略，以满足隐私需求同时保持计算成本可控。为了展示我们策略的有效性，我们还解决了评估机器去学习效果的挑战，考虑了模型在去学习和剩余数据上的广义性能。为了更好地评估去学习效果和模型泛化能力，我们提出了新的指标，即遗忘率和记忆保留率。然而，对于不精确的机器去学习，现有的指标无法对去学习程度进行准确量化。

    Machine unlearning techniques, which involve retracting data records and reducing influence of said data on trained models, help with the user privacy protection objective but incur significant computational costs. Weight perturbation-based unlearning is a general approach, but it typically involves globally modifying the parameters. We propose fine-grained Top-K and Random-k parameters perturbed inexact machine unlearning strategies that address the privacy needs while keeping the computational costs tractable.  In order to demonstrate the efficacy of our strategies we also tackle the challenge of evaluating the effectiveness of machine unlearning by considering the model's generalization performance across both unlearning and remaining data. To better assess the unlearning effect and model generalization, we propose novel metrics, namely, the forgetting rate and memory retention rate. However, for inexact machine unlearning, current metrics are inadequate in quantifying the degree of
    
[^10]: 一种模型评估的性能特征曲线：在信息扩散预测中的应用

    A performance characteristic curve for model evaluation: the application in information diffusion prediction. (arXiv:2309.09537v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2309.09537](http://arxiv.org/abs/2309.09537)

    本研究提出了一种模型的性能特征曲线，用于评估其在不同复杂度任务中的表现。通过使用基于信息熵的度量方法，我们确定了随机性与模型预测准确性之间的关系，并发现不同条件下的数据点都可以合并成一条曲线，捕捉了模型在面对不确定性时的正确预测能力。

    

    社交网络上的信息扩散预测旨在预测未来消息的接收者，在市场营销和社交媒体等实际应用中具有实用价值。尽管不同的预测模型都声称表现良好，但性能评估的通用框架仍然有限。本文旨在识别模型的性能特征曲线，该曲线捕获了模型在不同复杂度任务上的表现。我们提出了一种基于信息熵的度量方法来量化扩散数据中的随机性，然后确定了随机性与模型预测准确性之间的缩放模式。不同序列长度、系统大小和随机性下的数据点都合并成一条曲线，捕捉了模型在面对增加的不确定性时作出正确预测的内在能力。考虑到这条曲线具有评估模型的重要属性，我们将其定义为模型的性能特征曲线。

    The information diffusion prediction on social networks aims to predict future recipients of a message, with practical applications in marketing and social media. While different prediction models all claim to perform well, general frameworks for performance evaluation remain limited. Here, we aim to identify a performance characteristic curve for a model, which captures its performance on tasks of different complexity. We propose a metric based on information entropy to quantify the randomness in diffusion data, then identify a scaling pattern between the randomness and the prediction accuracy of the model. Data points in the patterns by different sequence lengths, system sizes, and randomness all collapse into a single curve, capturing a model's inherent capability of making correct predictions against increased uncertainty. Given that this curve has such important properties that it can be used to evaluate the model, we define it as the performance characteristic curve of the model.
    
[^11]: NeuroBench：通过合作、公平和代表性基准测试推进神经形态计算

    NeuroBench: Advancing Neuromorphic Computing through Collaborative, Fair and Representative Benchmarking. (arXiv:2304.04640v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2304.04640](http://arxiv.org/abs/2304.04640)

    NeuroBench是由学术界和工业界成员共同开发的一套协作、公平和代表性的基准测试，可以解决神经形态计算中缺乏清晰标准的问题，推动该领域的发展。

    

    神经形态计算领域在遵循仿生学原理的基础上，具有推进计算效率和能力的巨大潜力。然而，神经形态研究中采用的技术多样性导致缺乏清晰的基准测试标准，阻碍了对神经形态方法与传统基于深度学习的方法的优劣势进行有效评估。本文提出了一个协作项目——NeuroBench，将学术界和工业界成员聚集起来为神经形态计算定义基准测试。NeuroBench的目标是成为社区开发的协作、公平和代表性的基准测试套件。本文讨论了基准测试神经形态解决方案面临的挑战，并概述了NeuroBench的关键特性。我们相信，NeuroBench将是定义能够统一神经形态计算目标的标准的重要一步。

    The field of neuromorphic computing holds great promise in terms of advancing computing efficiency and capabilities by following brain-inspired principles. However, the rich diversity of techniques employed in neuromorphic research has resulted in a lack of clear standards for benchmarking, hindering effective evaluation of the advantages and strengths of neuromorphic methods compared to traditional deep-learning-based methods. This paper presents a collaborative effort, bringing together members from academia and the industry, to define benchmarks for neuromorphic computing: NeuroBench. The goals of NeuroBench are to be a collaborative, fair, and representative benchmark suite developed by the community, for the community. In this paper, we discuss the challenges associated with benchmarking neuromorphic solutions, and outline the key features of NeuroBench. We believe that NeuroBench will be a significant step towards defining standards that can unify the goals of neuromorphic comput
    
[^12]: 使AI“口渴”减少的方法：揭示和解决AI模型的秘密水消耗

    Making AI Less "Thirsty": Uncovering and Addressing the Secret Water Footprint of AI Models. (arXiv:2304.03271v1 [cs.LG])

    [http://arxiv.org/abs/2304.03271](http://arxiv.org/abs/2304.03271)

    本论文揭示以及提出了解决人工智能模型巨大水足迹的方法，因为其淡水消耗已经引起国际社会的重视，并且AI模型应该承担社会责任，做出面对水危机的表率。

    

    人工智能（AI）模型的碳足迹不断增长，特别是像GPT-3和GPT-4这样的大型模型，已经受到公众的关注。然而，同等重要且巨大的AI模型水印尚未引起人们的注意。例如，在微软最先进的美国数据中心中训练GPT-3可以直接消耗70万升清洁淡水（相当于生产370辆宝马汽车或320辆特斯拉电动汽车），如果在微软的亚洲数据中心进行训练，这个水消耗量将增加三倍，但这样的信息一直被保密。这极其令人担忧，因为淡水短缺已成为在人口迅速增长、水资源减少和老化的水基础设施的背景下，我们所有人面临的最紧迫的挑战之一。为了应对全球水资源的挑战，人工智能模型可以，而且应该，承担社会责任，以身作则解决自己的问题。

    The growing carbon footprint of artificial intelligence (AI) models, especially large ones such as GPT-3 and GPT-4, has been undergoing public scrutiny. Unfortunately, however, the equally important and enormous water footprint of AI models has remained under the radar. For example, training GPT-3 in Microsoft's state-of-the-art U.S. data centers can directly consume 700,000 liters of clean freshwater (enough for producing 370 BMW cars or 320 Tesla electric vehicles) and the water consumption would have been tripled if training were done in Microsoft's Asian data centers, but such information has been kept as a secret. This is extremely concerning, as freshwater scarcity has become one of the most pressing challenges shared by all of us in the wake of the rapidly growing population, depleting water resources, and aging water infrastructures. To respond to the global water challenges, AI models can, and also should, take social responsibility and lead by example by addressing their own 
    

