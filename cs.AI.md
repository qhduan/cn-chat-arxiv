# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Absolute Policy Optimization.](http://arxiv.org/abs/2310.13230) | 这篇论文提出了绝对策略优化（APO）的方法，通过优化一个新颖的目标函数，在保证性能下界的同时，实现了连续控制任务和Atari游戏中的令人瞩目的结果。 |
| [^2] | [Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets.](http://arxiv.org/abs/2310.01438) | 本文提出了一种灵活、可扩展且机器学习准备的多模态肿瘤学数据集(MINDS)框架，用于融合来自不同来源的数据，并提供了探索关系和构建大规模多模态机器学习模型的界面。 |
| [^3] | [S.T.A.R.-Track: Latent Motion Models for End-to-End 3D Object Tracking with Adaptive Spatio-Temporal Appearance Representations.](http://arxiv.org/abs/2306.17602) | 本文提出了S.T.A.R.-Track，一个采用物体为中心的Transformer框架，用于端到端3D物体跟踪。通过新颖的潜在运动模型和学习型跟踪嵌入，该框架能够准确建模物体的几何运动和变化，并在nuScenes数据集上取得了优秀的性能。 |
| [^4] | [MRFI: An Open Source Multi-Resolution Fault Injection Framework for Neural Network Processing.](http://arxiv.org/abs/2306.11758) | MRFI是一个高度可配置的神经网络故障注入工具，用户可以修改独立的故障配置文件进行注入和漏洞分析。 |
| [^5] | [Large Language Models Can Be Used To Effectively Scale Spear Phishing Campaigns.](http://arxiv.org/abs/2305.06972) | 大型语言模型可用于扩展钓鱼邮件攻击，作者通过实证测试表明高级的语言模型可以显著提高攻击的效率和成本效益。 |
| [^6] | [FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework.](http://arxiv.org/abs/2305.01658) | FlightBERT++提出了一种非自回归的多时域飞行轨迹预测框架，通过引入时域感知上下文生成器解决了误差累积和低效率的问题。 |
| [^7] | [Gradient Sparsification for Efficient Wireless Federated Learning with Differential Privacy.](http://arxiv.org/abs/2304.04164) | 本文提出了一种基于梯度稀疏化和差分隐私的无线联合学习框架，使用随机稀疏化算法缓解DP引起的性能下降，并减少上传的参数数量，提高训练效率而不损失收敛性能。 |
| [^8] | [Future Aware Pricing and Matching for Sustainable On-demand Ride Pooling.](http://arxiv.org/abs/2302.10510) | 本论文提出了一个新的框架，同时处理定价和匹配问题，并考虑商业决策对未来的影响，实验结果表明该框架可以显著提高拼车的效率和效益。 |
| [^9] | [Explaining wall-bounded turbulence through deep learning.](http://arxiv.org/abs/2302.01250) | 本研究采用深度学习预测了壁面边界层湍流中的速度场，并利用SHAP算法评估了相干结构对预测的重要性。这一过程或有助于解决湍流研究中的难题，为湍流模型的发展提供新思路。 |

# 详细

[^1]: 绝对策略优化

    Absolute Policy Optimization. (arXiv:2310.13230v1 [cs.LG])

    [http://arxiv.org/abs/2310.13230](http://arxiv.org/abs/2310.13230)

    这篇论文提出了绝对策略优化（APO）的方法，通过优化一个新颖的目标函数，在保证性能下界的同时，实现了连续控制任务和Atari游戏中的令人瞩目的结果。

    

    近年来，基于信任域的在线策略强化学习在解决复杂控制任务和游戏场景方面取得了令人瞩目的结果。然而，这一类别中现有的最先进算法主要强调对预期性能的改进，缺乏对最坏情况下性能结果的控制能力。为了解决这个限制，我们引入了一个新颖的目标函数；通过优化该函数，可以确保近乎总体性能样本的下界（绝对性能）呈现单调改进。考虑到这一具有突破性的理论进展，我们通过一系列的近似对这个理论基础算法进行了改进，得到了一种实用的解决方案称为绝对策略优化（APO）。我们的实验证明了我们的方法在具有挑战性的连续控制基准任务上的有效性，并将其适用性扩展到掌握Atari游戏。我们的发现表明，APO在提高性能的同时也显著改善了最坏情况下的性能结果。

    In recent years, trust region on-policy reinforcement learning has achieved impressive results in addressing complex control tasks and gaming scenarios. However, contemporary state-of-the-art algorithms within this category primarily emphasize improvement in expected performance, lacking the ability to control over the worst-case performance outcomes. To address this limitation, we introduce a novel objective function; by optimizing which, it will lead to guaranteed monotonic improvement in the lower bound of near-total performance samples (absolute performance). Considering this groundbreaking theoretical advancement, we then refine this theoretically grounded algorithm through a series of approximations, resulting in a practical solution called Absolute Policy Optimization (APO). Our experiments demonstrate the effectiveness of our approach across challenging continuous control benchmark tasks and extend its applicability to mastering Atari games. Our findings reveal that APO signifi
    
[^2]: 构建灵活、可扩展且机器学习准备的多模态肿瘤学数据集

    Building Flexible, Scalable, and Machine Learning-ready Multimodal Oncology Datasets. (arXiv:2310.01438v1 [cs.LG])

    [http://arxiv.org/abs/2310.01438](http://arxiv.org/abs/2310.01438)

    本文提出了一种灵活、可扩展且机器学习准备的多模态肿瘤学数据集(MINDS)框架，用于融合来自不同来源的数据，并提供了探索关系和构建大规模多模态机器学习模型的界面。

    

    数据采集、存储和处理技术的进步导致了异质医学数据的快速增长。将放射学扫描、组织病理学图像和分子信息与临床数据整合是开发对疾病有全面理解和优化治疗至关重要的。在复杂疾病（如癌症）中，将来自多个来源的数据进行整合的需求更加突出，以实现精准医学和个性化治疗。本研究提出了多模态肿瘤数据系统（MINDS）-一种灵活、可扩展且经济高效的元数据框架，用于将来自公共来源（如癌症研究数据共享库）的异构数据有效地融合到一个相互连接且以患者为中心的框架中。MINDS提供了一个可以探索不同数据类型之间关系并构建大规模多模态机器学习模型的界面。通过协调多模态数据，MINDS旨在实现促进研究创新、精准医学和个性化治疗的目标。

    The advancements in data acquisition, storage, and processing techniques have resulted in the rapid growth of heterogeneous medical data. Integrating radiological scans, histopathology images, and molecular information with clinical data is essential for developing a holistic understanding of the disease and optimizing treatment. The need for integrating data from multiple sources is further pronounced in complex diseases such as cancer for enabling precision medicine and personalized treatments. This work proposes Multimodal Integration of Oncology Data System (MINDS) - a flexible, scalable, and cost-effective metadata framework for efficiently fusing disparate data from public sources such as the Cancer Research Data Commons (CRDC) into an interconnected, patient-centric framework. MINDS offers an interface for exploring relationships across data types and building cohorts for developing large-scale multimodal machine learning models. By harmonizing multimodal data, MINDS aims to pot
    
[^3]: S.T.A.R.-Track：自适应时空外貌表示的端到端3D物体跟踪的潜在运动模型

    S.T.A.R.-Track: Latent Motion Models for End-to-End 3D Object Tracking with Adaptive Spatio-Temporal Appearance Representations. (arXiv:2306.17602v1 [cs.CV])

    [http://arxiv.org/abs/2306.17602](http://arxiv.org/abs/2306.17602)

    本文提出了S.T.A.R.-Track，一个采用物体为中心的Transformer框架，用于端到端3D物体跟踪。通过新颖的潜在运动模型和学习型跟踪嵌入，该框架能够准确建模物体的几何运动和变化，并在nuScenes数据集上取得了优秀的性能。

    

    本文基于跟踪-注意力模式，引入了一个以物体为中心的基于Transformer的3D跟踪框架。传统的基于模型的跟踪方法通过几何运动模型融合帧之间的物体和自运动的几何效应。受此启发，我们提出了S.T.A.R.-Track，使用一种新颖的潜在运动模型来调整对象查询，以在潜在空间中直接考虑视角和光照条件的变化，同时明确建模几何运动。结合一种新颖的可学习的跟踪嵌入，有助于建模轨迹的存在概率，这导致了一个通用的跟踪框架，可以与任何基于查询的检测器集成。在nuScenes基准测试上进行了大量实验，证明了我们方法的优势，展示了基于DETR3D的跟踪器的最先进性能，同时大大减少了轨迹的身份转换次数。

    Following the tracking-by-attention paradigm, this paper introduces an object-centric, transformer-based framework for tracking in 3D. Traditional model-based tracking approaches incorporate the geometric effect of object- and ego motion between frames with a geometric motion model. Inspired by this, we propose S.T.A.R.-Track, which uses a novel latent motion model (LMM) to additionally adjust object queries to account for changes in viewing direction and lighting conditions directly in the latent space, while still modeling the geometric motion explicitly. Combined with a novel learnable track embedding that aids in modeling the existence probability of tracks, this results in a generic tracking framework that can be integrated with any query-based detector. Extensive experiments on the nuScenes benchmark demonstrate the benefits of our approach, showing state-of-the-art performance for DETR3D-based trackers while drastically reducing the number of identity switches of tracks at the s
    
[^4]: MRFI：神经网络处理的开源多分辨率故障注入框架

    MRFI: An Open Source Multi-Resolution Fault Injection Framework for Neural Network Processing. (arXiv:2306.11758v1 [cs.LG])

    [http://arxiv.org/abs/2306.11758](http://arxiv.org/abs/2306.11758)

    MRFI是一个高度可配置的神经网络故障注入工具，用户可以修改独立的故障配置文件进行注入和漏洞分析。

    

    为了确保即使在不可靠的硬件上也能进行有弹性的神经网络处理，通常需要在深度神经网络模型部署之前进行各种硬件故障的全面可靠性分析，并且需要高效的错误注入工具。然而，大多数现有的故障注入工具仍然局限于对神经元的基本故障注入，并未提供细粒度漏洞分析能力。此外，许多故障注入工具仍需要更改神经网络模型并使故障注入与正常神经网络处理紧密耦合，这进一步增加了故障注入工具的使用难度并减慢了故障模拟。在这项工作中，我们提出了一个高度可配置的深度神经网络多分辨率故障注入工具MRFI。它使用户能够修改独立的故障配置文件，而不是修改神经网络模型进行故障注入和漏洞分析。

    To ensure resilient neural network processing on even unreliable hardware, comprehensive reliability analysis against various hardware faults is generally required before the deep neural network models are deployed, and efficient error injection tools are highly demanded. However, most existing fault injection tools remain rather limited to basic fault injection to neurons and fail to provide fine-grained vulnerability analysis capability. In addition, many of the fault injection tools still need to change the neural network models and make the fault injection closely coupled with normal neural network processing, which further complicates the use of the fault injection tools and slows down the fault simulation. In this work, we propose MRFI, a highly configurable multi-resolution fault injection tool for deep neural networks. It enables users to modify an independent fault configuration file rather than neural network models for the fault injection and vulnerability analysis. Particul
    
[^5]: 大型语言模型可用于有效扩展钓鱼邮件攻击

    Large Language Models Can Be Used To Effectively Scale Spear Phishing Campaigns. (arXiv:2305.06972v1 [cs.CY])

    [http://arxiv.org/abs/2305.06972](http://arxiv.org/abs/2305.06972)

    大型语言模型可用于扩展钓鱼邮件攻击，作者通过实证测试表明高级的语言模型可以显著提高攻击的效率和成本效益。

    

    人工智能领域的最新进展尤其是大型语言模型的发展，已经产生了功能强大而通用的双重用途系统。本研究调查了如何使用大型语言模型进行钓鱼邮件攻击，这种流行的网络犯罪形式涉及将目标人物诱骗披露敏感信息。作者首先研究了LLMs在成功的钓鱼攻击的侦察和信息生成阶段的能力，发现先进的LLMs能够在这些阶段显着提高网络罪犯的效率。其次，作者使用OpenAI的GPT-3.5和GPT-4模型为超过600名英国议员创建了独特的钓鱼邮件的实证测试。研究结果表明，这些邮件不仅逼真而且成本效益显著，每封电子邮件仅花费几分之一的美分即可产生。

    Recent progress in artificial intelligence (AI), particularly in the domain of large language models (LLMs), has resulted in powerful and versatile dual-use systems. Indeed, cognition can be put towards a wide variety of tasks, some of which can result in harm. This study investigates how LLMs can be used for spear phishing, a prevalent form of cybercrime that involves manipulating targets into divulging sensitive information. I first explore LLMs' ability to assist with the reconnaissance and message generation stages of a successful spear phishing attack, where I find that advanced LLMs are capable of meaningfully improving cybercriminals' efficiency during these stages. Next, I conduct an empirical test by creating unique spear phishing messages for over 600 British Members of Parliament using OpenAI's GPT-3.5 and GPT-4 models. My findings reveal that these messages are not only realistic but also remarkably cost-effective, as each email cost only a fraction of a cent to generate. N
    
[^6]: FlightBERT++：一种非自回归多时域飞行轨迹预测框架

    FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework. (arXiv:2305.01658v1 [cs.LG])

    [http://arxiv.org/abs/2305.01658](http://arxiv.org/abs/2305.01658)

    FlightBERT++提出了一种非自回归的多时域飞行轨迹预测框架，通过引入时域感知上下文生成器解决了误差累积和低效率的问题。

    

    飞行轨迹预测是空中交通管制中的重要任务，可以帮助空管员更安全高效地管理空域。现有方法通常采用自回归方式执行多时域飞行轨迹预测任务，容易出现误差累积和低效率问题。本文提出了一种新的框架，称为FlightBERT++，以i）直接以非自回归方式预测多时域飞行轨迹，和ii）改善FlightBERT框架中二进制编码（BE）表示的限制。具体而言，所提出的框架通过通用的编码器-解码器架构实现，其中编码器从历史观测中学习时空模式，而解码器预测未来时间步的飞行状态。与传统架构相比，额外的时域感知上下文生成器（HACG）专门设计考虑先前的时域。

    Flight Trajectory Prediction (FTP) is an essential task in Air Traffic Control (ATC), which can assist air traffic controllers to manage airspace more safely and efficiently. Existing approaches generally perform multi-horizon FTP tasks in an autoregressive manner, which is prone to suffer from error accumulation and low-efficiency problems. In this paper, a novel framework, called FlightBERT++, is proposed to i) forecast multi-horizon flight trajectories directly in a non-autoregressive way, and ii) improved the limitation of the binary encoding (BE) representation in the FlightBERT framework. Specifically, the proposed framework is implemented by a generalized Encoder-Decoder architecture, in which the encoder learns the temporal-spatial patterns from historical observations and the decoder predicts the flight status for the future time steps. Compared to conventional architecture, an extra horizon-aware contexts generator (HACG) is dedicatedly designed to consider the prior horizon 
    
[^7]: 基于梯度稀疏化和差分隐私的高效无线联合学习

    Gradient Sparsification for Efficient Wireless Federated Learning with Differential Privacy. (arXiv:2304.04164v1 [cs.DC])

    [http://arxiv.org/abs/2304.04164](http://arxiv.org/abs/2304.04164)

    本文提出了一种基于梯度稀疏化和差分隐私的无线联合学习框架，使用随机稀疏化算法缓解DP引起的性能下降，并减少上传的参数数量，提高训练效率而不损失收敛性能。

    

    联合学习使分布式客户端在不共享原始数据的情况下协同训练机器学习模型。但是，由于上传模型而泄漏私有信息。此外，随着模型大小的增加，由于有限的传输带宽，训练延迟增加，同时使用差分隐私（DP）保护时模型性能会下降。在本文中，我们提出了一种基于梯度稀疏化和差分隐私的无线联合学习框架，以提高训练效率而不损失收敛性能。具体而言，我们首先设计了一个随机稀疏化算法，在每个客户端的本地训练中保留一部分梯度元素，从而缓解了DP引起的性能下降，并减少了无线信道上传输的参数数量。然后，我们通过建模非凸FL问题分析了所提出算法的收敛度界。接下来，我们提出了一个分布式联合优化问题，使用Alternating Direction Method of Multipliers（ADMM）解决其优化问题。

    Federated learning (FL) enables distributed clients to collaboratively train a machine learning model without sharing raw data with each other. However, it suffers the leakage of private information from uploading models. In addition, as the model size grows, the training latency increases due to limited transmission bandwidth and the model performance degrades while using differential privacy (DP) protection. In this paper, we propose a gradient sparsification empowered FL framework over wireless channels, in order to improve training efficiency without sacrificing convergence performance. Specifically, we first design a random sparsification algorithm to retain a fraction of the gradient elements in each client's local training, thereby mitigating the performance degradation induced by DP and and reducing the number of transmission parameters over wireless channels. Then, we analyze the convergence bound of the proposed algorithm, by modeling a non-convex FL problem. Next, we formula
    
[^8]: 面向可持续性的即时拼车的未来感知定价和匹配

    Future Aware Pricing and Matching for Sustainable On-demand Ride Pooling. (arXiv:2302.10510v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2302.10510](http://arxiv.org/abs/2302.10510)

    本论文提出了一个新的框架，同时处理定价和匹配问题，并考虑商业决策对未来的影响，实验结果表明该框架可以显著提高拼车的效率和效益。

    

    即时拼车的受欢迎程度在于为顾客（更低的价格）、出租车司机（更高的收入）、环境（由于更少的车辆而减少碳排放量）和Uber等聚合公司（更高的收入）提供了好处。为了实现这些收益，必须有效地解决两个关键而相互关联的挑战：（a）定价——为出租车顾客请求设置价格；和（b）匹配——将接受价格的客户分配给出租车/汽车。传统上，这两个挑战都是单独研究的，并且使用短视的方法（只考虑当前请求），而不考虑当前匹配对解决未来请求的影响。在本文中，我们开发了一个新的框架，处理定价和匹配问题，同时考虑定价和匹配决策对未来影响的影响。在真实出租车数据集上的实验结果表明，我们的框架可以显著改善匹配和定价效果。

    The popularity of on-demand ride pooling is owing to the benefits offered to customers (lower prices), taxi drivers (higher revenue), environment (lower carbon footprint due to fewer vehicles) and aggregation companies like Uber (higher revenue). To achieve these benefits, two key interlinked challenges have to be solved effectively: (a) pricing -- setting prices to customer requests for taxis; and (b) matching -- assignment of customers (that accepted the prices) to taxis/cars. Traditionally, both these challenges have been studied individually and using myopic approaches (considering only current requests), without considering the impact of current matching on addressing future requests. In this paper, we develop a novel framework that handles the pricing and matching problems together, while also considering the future impact of the pricing and matching decisions. In our experimental results on a real-world taxi dataset, we demonstrate that our framework can significantly improve re
    
[^9]: 通过深度学习解释壁面边界层湍流

    Explaining wall-bounded turbulence through deep learning. (arXiv:2302.01250v2 [physics.flu-dyn] UPDATED)

    [http://arxiv.org/abs/2302.01250](http://arxiv.org/abs/2302.01250)

    本研究采用深度学习预测了壁面边界层湍流中的速度场，并利用SHAP算法评估了相干结构对预测的重要性。这一过程或有助于解决湍流研究中的难题，为湍流模型的发展提供新思路。

    

    壁面边界层湍流作为一个具有重大科学和技术意义的问题，需要寻求新的视角来解决。本研究首次采用可解释的深度学习方法研究了流场中相干结构之间的相互作用。通过卷积神经网络，利用湍流通道中的瞬时速度场预测了时间内的速度场，然后利用SHapley Additive exPlanations（SHAP）算法对每个结构预测的重要性进行了评估。本研究结果与先前文献观察结果一致，并通过量化雷诺应力结构的重要性，找到了这些结构与流动动力学之间的联系。采用深度学习可解释性的方法可能有助于揭示壁面边界层湍流的长期问题，并为湍流模型的开发提供新的见解。

    Despite its great scientific and technological importance, wall-bounded turbulence is an unresolved problem that requires new perspectives to be tackled. One of the key strategies has been to study interactions among the coherent structures in the flow. Such interactions are explored in this study for the first time using an explainable deep-learning method. The instantaneous velocity field in a turbulent channel is used to predict the velocity field in time through a convolutional neural network. Based on the predicted flow, we assess the importance of each structure for this prediction using the game-theoretic algorithm of SHapley Additive exPlanations (SHAP). This work provides results in agreement with previous observations in the literature and extends them by quantifying the importance of the Reynolds-stress structures, finding a connection between these structures and the dynamics of the flow. The process, based on deep-learning explainability, has the potential to shed light on
    

