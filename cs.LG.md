# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Facilitating Reinforcement Learning for Process Control Using Transfer Learning: Perspectives](https://arxiv.org/abs/2404.00247) | 本文从迁移学习的角度探讨了如何将其与强化学习相结合，为过程控制带来新的可能性。 |
| [^2] | [Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data](https://arxiv.org/abs/2403.08728) | 提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。 |
| [^3] | [QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs](https://arxiv.org/abs/2402.15929) | 本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。 |
| [^4] | [Revolutionizing Wireless Networks with Federated Learning: A Comprehensive Review.](http://arxiv.org/abs/2308.04404) | 联邦学习是一种新兴的机器学习模型，它可以在无线边缘网络中实现数据获取和计算的分离，对于未来的移动网络尤其是6G及以后具有重要作用。 |
| [^5] | [PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models.](http://arxiv.org/abs/2307.03034) | 本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。 |
| [^6] | [Hedonic Prices and Quality Adjusted Price Indices Powered by AI.](http://arxiv.org/abs/2305.00044) | 本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。 |
| [^7] | [LASER: Neuro-Symbolic Learning of Semantic Video Representations.](http://arxiv.org/abs/2304.07647) | LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。 |
| [^8] | [Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series.](http://arxiv.org/abs/2304.03069) | 本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。 |
| [^9] | [Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization.](http://arxiv.org/abs/2303.03984) | 本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。 |

# 详细

[^1]: 利用迁移学习促进过程控制的强化学习：观点

    Facilitating Reinforcement Learning for Process Control Using Transfer Learning: Perspectives

    [https://arxiv.org/abs/2404.00247](https://arxiv.org/abs/2404.00247)

    本文从迁移学习的角度探讨了如何将其与强化学习相结合，为过程控制带来新的可能性。

    

    本文从迁移学习的角度，为过程控制中的深度强化学习（DRL）提供了深入见解。我们分析了在过程工业领域应用DRL所面临的挑战，以及引入迁移学习的必要性。此外，我们为未来研究方向提供了建议和展望，探讨了如何将迁移学习与DRL结合起来加强过程控制。

    arXiv:2404.00247v1 Announce Type: cross  Abstract: This paper provides insights into deep reinforcement learning (DRL) for process control from the perspective of transfer learning. We analyze the challenges of applying DRL in the field of process industries and the necessity of introducing transfer learning. Furthermore, recommendations and prospects are provided for future research directions on how transfer learning can be integrated with DRL to empower process control.
    
[^2]: 使用环境扩散后验采样：在受损数据上训练的扩散模型解决逆问题

    Ambient Diffusion Posterior Sampling: Solving Inverse Problems with Diffusion Models trained on Corrupted Data

    [https://arxiv.org/abs/2403.08728](https://arxiv.org/abs/2403.08728)

    提出了一种使用环境扩散后验采样解决逆问题的框架，能在受损数据上训练的扩散模型上表现出色，并在图像恢复和MRI模型训练中取得优越性能。

    

    我们提供了一个框架，用于使用从线性受损数据中学习的扩散模型解决逆问题。我们的方法，Ambient Diffusion Posterior Sampling (A-DPS)，利用一个预先在一种类型的损坏数据上进行过训练的生成模型，以在可能来自不同前向过程（例如图像模糊）的测量条件下执行后验采样。我们在标准自然图像数据集（CelebA、FFHQ 和 AFHQ）上测试了我们的方法的有效性，并展示了 A-DPS 有时在速度和性能上都能胜过在清洁数据上训练的模型，用于几个图像恢复任务。我们进一步扩展了环境扩散框架，以仅访问傅里叶子采样的多线圈 MRI 测量数据来训练 MRI 模型，其加速因子为不同的加速因子（R=2、4、6、8）。我们再次观察到，在高度子采样数据上训练的模型更适用于解决高加速 MRI 逆问题。

    arXiv:2403.08728v1 Announce Type: cross  Abstract: We provide a framework for solving inverse problems with diffusion models learned from linearly corrupted data. Our method, Ambient Diffusion Posterior Sampling (A-DPS), leverages a generative model pre-trained on one type of corruption (e.g. image inpainting) to perform posterior sampling conditioned on measurements from a potentially different forward process (e.g. image blurring). We test the efficacy of our approach on standard natural image datasets (CelebA, FFHQ, and AFHQ) and we show that A-DPS can sometimes outperform models trained on clean data for several image restoration tasks in both speed and performance. We further extend the Ambient Diffusion framework to train MRI models with access only to Fourier subsampled multi-coil MRI measurements at various acceleration factors (R=2, 4, 6, 8). We again observe that models trained on highly subsampled data are better priors for solving inverse problems in the high acceleration r
    
[^3]: QuaCer-C：大型语言模型中知识理解的定量认证

    QuaCer-C: Quantitative Certification of Knowledge Comprehension in LLMs

    [https://arxiv.org/abs/2402.15929](https://arxiv.org/abs/2402.15929)

    本文提出了一种新颖的认证框架QuaCer-C，用于正式认证大型语言模型中知识理解的能力，证书定量化且包含高置信度的概率界限，研究发现，随着参数数量的增加，知识理解能力提高，Mistral模型在这一评估中表现不如其他模型。

    

    大型语言模型（LLMs）在多个基准测试中展现出令人印象深刻的表现。然而，传统研究并未对LLMs的表现提供正式的保证。本文提出了一种新颖的LLM认证框架QuaCer-C，我们在此对知名LLMs的知识理解能力进行正式认证。我们的证书是定量的 - 它们包括对目标LLM在任何相关知识理解提示上给出正确答案的概率的高置信度紧密界限。我们针对Llama、Vicuna和Mistral LLMs的证书表明，知识理解能力随参数数量的增加而提高，并且Mistral模型在这一评估中表现不如其他模型。

    arXiv:2402.15929v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated impressive performance on several benchmarks. However, traditional studies do not provide formal guarantees on the performance of LLMs. In this work, we propose a novel certification framework for LLM, QuaCer-C, wherein we formally certify the knowledge-comprehension capabilities of popular LLMs. Our certificates are quantitative - they consist of high-confidence, tight bounds on the probability that the target LLM gives the correct answer on any relevant knowledge comprehension prompt. Our certificates for the Llama, Vicuna, and Mistral LLMs indicate that the knowledge comprehension capability improves with an increase in the number of parameters and that the Mistral model is less performant than the rest in this evaluation.
    
[^4]: 用联邦学习革新无线网络：一项综合评述

    Revolutionizing Wireless Networks with Federated Learning: A Comprehensive Review. (arXiv:2308.04404v1 [cs.LG])

    [http://arxiv.org/abs/2308.04404](http://arxiv.org/abs/2308.04404)

    联邦学习是一种新兴的机器学习模型，它可以在无线边缘网络中实现数据获取和计算的分离，对于未来的移动网络尤其是6G及以后具有重要作用。

    

    随着智能手机、平板电脑和车辆等无线用户设备的计算能力不断提高，以及对共享私人数据的日益关注，一种名为联邦学习（FL）的新型机器学习模型已经出现。FL使得数据获取和计算在中央单元中分离，这与在数据中心中进行的集中式学习不同。FL通常用于无线边缘网络中，其中通信资源有限且不可靠。带宽限制要求在每次迭代中仅安排部分用户设备进行更新，并且由于无线介质是共享的，传输易受干扰且不保证。本文讨论了机器学习在无线通信中的重要性，并强调了联邦学习（FL）作为一种新颖方法，在未来的移动网络中特别是6G及以后扮演重要角色。

    These days with the rising computational capabilities of wireless user equipment such as smart phones, tablets, and vehicles, along with growing concerns about sharing private data, a novel machine learning model called federated learning (FL) has emerged. FL enables the separation of data acquisition and computation at the central unit, which is different from centralized learning that occurs in a data center. FL is typically used in a wireless edge network where communication resources are limited and unreliable. Bandwidth constraints necessitate scheduling only a subset of UEs for updates in each iteration, and because the wireless medium is shared, transmissions are susceptible to interference and are not assured. The article discusses the significance of Machine Learning in wireless communication and highlights Federated Learning (FL) as a novel approach that could play a vital role in future mobile networks, particularly 6G and beyond.
    
[^5]: 带有一般观测模型的不安定赌博机问题的PCL-可索引性和Whittle索引

    PCL-Indexability and Whittle Index for Restless Bandits with General Observation Models. (arXiv:2307.03034v1 [stat.ML])

    [http://arxiv.org/abs/2307.03034](http://arxiv.org/abs/2307.03034)

    本文研究了一种一般观测模型下的不安定多臂赌博机问题，提出了PCL-可索引性和Whittle索引的分析方法，并通过近似过程将问题转化为有限状态问题。数值实验表明算法表现优秀。

    

    本文考虑了一种一般观测模型，用于不安定多臂赌博机问题。由于资源约束或环境或固有噪声，玩家操作需要基于某种有误差的反馈机制。通过建立反馈/观测动力学的一般概率模型，我们将问题表述为一个从任意初始信念（先验信息）开始的具有可数信念状态空间的不安定赌博机问题。我们利用具有部分守恒定律（PCL）的可实现区域方法，分析了无限状态问题的可索引性和优先级索引（Whittle索引）。最后，我们提出了一个近似过程，将问题转化为可以应用Niño-Mora和Bertsimas针对有限状态问题的AG算法的问题。数值实验表明，我们的算法具有出色的性能。

    In this paper, we consider a general observation model for restless multi-armed bandit problems. The operation of the player needs to be based on certain feedback mechanism that is error-prone due to resource constraints or environmental or intrinsic noises. By establishing a general probabilistic model for dynamics of feedback/observation, we formulate the problem as a restless bandit with a countable belief state space starting from an arbitrary initial belief (a priori information). We apply the achievable region method with partial conservation law (PCL) to the infinite-state problem and analyze its indexability and priority index (Whittle index). Finally, we propose an approximation process to transform the problem into which the AG algorithm of Ni\~no-Mora and Bertsimas for finite-state problems can be applied to. Numerical experiments show that our algorithm has an excellent performance.
    
[^6]: 由人工智能驱动的享乐价格和质量调整价格指数

    Hedonic Prices and Quality Adjusted Price Indices Powered by AI. (arXiv:2305.00044v1 [econ.GN])

    [http://arxiv.org/abs/2305.00044](http://arxiv.org/abs/2305.00044)

    本研究提出了一种基于深度神经网络和转换器的经验享乐模型，能够处理大量未结构化的产品数据，准确地估计产品的享乐价格和派生指数。

    

    在当今的经济环境下，使用电子记录准确地实时测量价格指数的变化对于跟踪通胀和生产率至关重要。本文开发了经验享乐模型，能够处理大量未结构化的产品数据（文本、图像、价格和数量），并输出精确的享乐价格估计和派生指数。为实现这一目标，我们使用深度神经网络从文本描述和图像中生成抽象的产品属性或”特征“，然后使用这些属性来估算享乐价格函数。具体地，我们使用基于transformers的大型语言模型将有关产品的文本信息转换为数字特征，使用训练或微调过的产品描述信息，使用残差网络模型将产品图像转换为数字特征。为了产生估计的享乐价格函数，我们再次使用多任务神经网络，训练以在所有时间段同时预测产品的价格。

    Accurate, real-time measurements of price index changes using electronic records are essential for tracking inflation and productivity in today's economic environment. We develop empirical hedonic models that can process large amounts of unstructured product data (text, images, prices, quantities) and output accurate hedonic price estimates and derived indices. To accomplish this, we generate abstract product attributes, or ``features,'' from text descriptions and images using deep neural networks, and then use these attributes to estimate the hedonic price function. Specifically, we convert textual information about the product to numeric features using large language models based on transformers, trained or fine-tuned using product descriptions, and convert the product image to numeric features using a residual network model. To produce the estimated hedonic price function, we again use a multi-task neural network trained to predict a product's price in all time periods simultaneousl
    
[^7]: LASER：神经符号学习语义视频表示

    LASER: Neuro-Symbolic Learning of Semantic Video Representations. (arXiv:2304.07647v1 [cs.CV])

    [http://arxiv.org/abs/2304.07647](http://arxiv.org/abs/2304.07647)

    LASER提出了一种神经符号学习方法来学习语义视频表示，通过逻辑规范捕捉视频数据中的时空属性，能够对齐原始视频和规范，有效地训练低级感知模型以提取符合所需高级规范的视频表示。

    

    现代涉及视频的AI应用（如视频-文本对齐、视频搜索和视频字幕）受益于对视频语义的细致理解。现有的视频理解方法要么需要大量注释，要么基于不可解释的通用嵌入，可能会忽略重要细节。我们提出了LASER，这是一种神经符号方法，通过利用能够捕捉视频数据中丰富的时空属性的逻辑规范来学习语义视频表示。特别地，我们通过原始视频与规范之间的对齐来公式化问题。对齐过程有效地训练了低层感知模型，以提取符合所需高层规范的细粒度视频表示。我们的流程可以端到端地训练，并可纳入从规范导出的对比和语义损失函数。我们在两个具有丰富空间和时间信息的数据集上评估了我们的方法。

    Modern AI applications involving video, such as video-text alignment, video search, and video captioning, benefit from a fine-grained understanding of video semantics. Existing approaches for video understanding are either data-hungry and need low-level annotation, or are based on general embeddings that are uninterpretable and can miss important details. We propose LASER, a neuro-symbolic approach that learns semantic video representations by leveraging logic specifications that can capture rich spatial and temporal properties in video data. In particular, we formulate the problem in terms of alignment between raw videos and specifications. The alignment process efficiently trains low-level perception models to extract a fine-grained video representation that conforms to the desired high-level specification. Our pipeline can be trained end-to-end and can incorporate contrastive and semantic loss functions derived from specifications. We evaluate our method on two datasets with rich sp
    
[^8]: 自适应学生t分布与方法矩移动估计器用于非平稳时间序列

    Adaptive Student's t-distribution with method of moments moving estimator for nonstationary time series. (arXiv:2304.03069v1 [stat.ME])

    [http://arxiv.org/abs/2304.03069](http://arxiv.org/abs/2304.03069)

    本文提出了一种适用于非平稳时间序列的自适应学生t分布方法，基于方法的一般自适应矩可以使用廉价的指数移动平均值（EMA）来估计参数。

    

    真实的时间序列通常是非平稳的，这带来了模型适应的难题。传统方法如GARCH假定任意类型的依赖性。为了避免这种偏差，我们将着眼于最近提出的不可知的移动估计器哲学：在时间$t$找到优化$F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$移动对数似然的参数，随时间演化。例如，它允许使用廉价的指数移动平均值（EMA）来估计参数，例如绝对中心矩$E[|x-\mu|^p]$随$p\in\mathbb{R}^+$的变化而演化$m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$。这种基于方法的一般自适应矩的应用将呈现在学生t分布上，尤其是在经济应用中流行，这里应用于DJIA公司的对数收益率。

    The real life time series are usually nonstationary, bringing a difficult question of model adaptation. Classical approaches like GARCH assume arbitrary type of dependence. To prevent such bias, we will focus on recently proposed agnostic philosophy of moving estimator: in time $t$ finding parameters optimizing e.g. $F_t=\sum_{\tau<t} (1-\eta)^{t-\tau} \ln(\rho_\theta (x_\tau))$ moving log-likelihood, evolving in time. It allows for example to estimate parameters using inexpensive exponential moving averages (EMA), like absolute central moments $E[|x-\mu|^p]$ evolving with $m_{p,t+1} = m_{p,t} + \eta (|x_t-\mu_t|^p-m_{p,t})$ for one or multiple powers $p\in\mathbb{R}^+$. Application of such general adaptive methods of moments will be presented on Student's t-distribution, popular especially in economical applications, here applied to log-returns of DJIA companies.
    
[^9]: 非凸-PL极小极大优化的增强自适应梯度算法

    Enhanced Adaptive Gradient Algorithms for Nonconvex-PL Minimax Optimization. (arXiv:2303.03984v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2303.03984](http://arxiv.org/abs/2303.03984)

    本文提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决非凸-PL极小极大问题，其中AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    This paper proposes a class of enhanced momentum-based gradient descent ascent methods (MSGDA and AdaMSGDA) to solve nonconvex-PL minimax problems, where the AdaMSGDA algorithm can use various adaptive learning rates to update variables x and y without relying on any global and coordinate-wise adaptive learning rates. Theoretical analysis shows that MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of O(ε−3) in finding an ε-stationary solution.

    本文研究了一类非凸非凹的极小极大优化问题（即$\min_x\max_y f(x,y)$），其中$f(x,y)$在$x$上可能是非凸的，在$y$上是非凹的，并满足Polyak-Lojasiewicz（PL）条件。此外，我们提出了一类增强的基于动量的梯度下降上升方法（即MSGDA和AdaMSGDA）来解决这些随机非凸-PL极小极大问题。特别地，我们的AdaMSGDA算法可以使用各种自适应学习率来更新变量$x$和$y$，而不依赖于任何全局和坐标自适应学习率。理论上，我们提出了一种有效的收敛分析框架来解决我们的方法。具体而言，我们证明了我们的MSGDA和AdaMSGDA方法在找到$\epsilon$-稳定解（即$\mathbb{E}\|\nabla F(x)\|\leq \epsilon$，其中$F(x)=\max_y f(x,y)$）时，只需要在每个循环中进行一次采样，就可以获得已知的最佳样本（梯度）复杂度$O(\epsilon^{-3})$。

    In the paper, we study a class of nonconvex nonconcave minimax optimization problems (i.e., $\min_x\max_y f(x,y)$), where $f(x,y)$ is possible nonconvex in $x$, and it is nonconcave and satisfies the Polyak-Lojasiewicz (PL) condition in $y$. Moreover, we propose a class of enhanced momentum-based gradient descent ascent methods (i.e., MSGDA and AdaMSGDA) to solve these stochastic Nonconvex-PL minimax problems. In particular, our AdaMSGDA algorithm can use various adaptive learning rates in updating the variables $x$ and $y$ without relying on any global and coordinate-wise adaptive learning rates. Theoretically, we present an effective convergence analysis framework for our methods. Specifically, we prove that our MSGDA and AdaMSGDA methods have the best known sample (gradient) complexity of $O(\epsilon^{-3})$ only requiring one sample at each loop in finding an $\epsilon$-stationary solution (i.e., $\mathbb{E}\|\nabla F(x)\|\leq \epsilon$, where $F(x)=\max_y f(x,y)$). This manuscript 
    

