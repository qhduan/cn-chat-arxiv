# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Conditional Generative Learning: Model and Error Analysis](https://rss.arxiv.org/abs/2402.01460) | 提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。 |
| [^2] | [Discovery of the Hidden World with Large Language Models](https://arxiv.org/abs/2402.03941) | 通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。 |
| [^3] | [Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning.](http://arxiv.org/abs/2401.06344) | 本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。 |
| [^4] | [Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets.](http://arxiv.org/abs/2309.12032) | 该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。 |
| [^5] | [Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection.](http://arxiv.org/abs/2308.06701) | 该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。 |
| [^6] | [Privacy-aware Gaussian Process Regression.](http://arxiv.org/abs/2305.16541) | 以高斯过程回归为基础，我们提出了实现数据隐私保护的方法。方法将综合噪声添加到数据中，使得高斯过程预测模型达到特定的隐私级别。我们还通过核方法介绍了连续隐私约束下的隐私感知解形式，以及研究了其理论性质。所提出的方法应用于卫星轨迹跟踪模型。 |
| [^7] | [Online Loss Function Learning.](http://arxiv.org/abs/2301.13247) | 在线损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们的方法在多个任务上稳定地优于现有技术。 |
| [^8] | [Speech Enhancement and Dereverberation with Diffusion-based Generative Models.](http://arxiv.org/abs/2208.05830) | 本文基于扩散生成模型，通过混合噪声语音和高斯噪声进行反向过程，仅使用30步就实现高质量的干净语音估计。调整网络架构，可以显著提高语音增强性能，达到了最新方法的竞争水平。 |

# 详细

[^1]: 深度条件生成学习：模型与误差分析

    Deep Conditional Generative Learning: Model and Error Analysis

    [https://rss.arxiv.org/abs/2402.01460](https://rss.arxiv.org/abs/2402.01460)

    提出了一种基于ODE的深度生成方法，通过条件Follmer流来学习条件分布，通过离散化和深度神经网络实现高效转化。同时，通过Wasserstein距离的非渐近收敛速率，提供了第一个端到端误差分析，数值实验证明其在不同场景下的优越性。

    

    我们介绍了一种基于普通微分方程（ODE）的深度生成方法，用于学习条件分布，称为条件Follmer流。从标准高斯分布开始，所提出的流能够以高效的方式将其转化为目标条件分布，在时间1处达到稳定。为了有效实现，我们使用欧拉方法对流进行离散化，使用深度神经网络非参数化估计速度场。此外，我们导出了学习样本的分布与目标分布之间的Wasserstein距离的非渐近收敛速率，在条件分布学习中提供了第一个全面的端到端误差分析。我们的数值实验展示了它在一系列情况下的有效性，从标准的非参数化条件密度估计问题到涉及图像数据的更复杂的挑战，说明它优于各种现有方法。

    We introduce an Ordinary Differential Equation (ODE) based deep generative method for learning a conditional distribution, named the Conditional Follmer Flow. Starting from a standard Gaussian distribution, the proposed flow could efficiently transform it into the target conditional distribution at time 1. For effective implementation, we discretize the flow with Euler's method where we estimate the velocity field nonparametrically using a deep neural network. Furthermore, we derive a non-asymptotic convergence rate in the Wasserstein distance between the distribution of the learned samples and the target distribution, providing the first comprehensive end-to-end error analysis for conditional distribution learning via ODE flow. Our numerical experiments showcase its effectiveness across a range of scenarios, from standard nonparametric conditional density estimation problems to more intricate challenges involving image data, illustrating its superiority over various existing condition
    
[^2]: 用大型语言模型探索隐藏世界

    Discovery of the Hidden World with Large Language Models

    [https://arxiv.org/abs/2402.03941](https://arxiv.org/abs/2402.03941)

    通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。

    

    科学起源于从已知事实和观察中发现新的因果知识。传统的因果发现方法主要依赖于高质量的测量变量，通常由人类专家提供，以找到因果关系。然而，在许多现实世界的应用中，因果变量通常无法获取。大型语言模型（LLMs）的崛起为从原始观测数据中发现高级隐藏变量提供了新的机会。因此，我们介绍了COAT：因果表示助手。COAT将LLMs作为因素提供器引入，提取出来自非结构化数据的潜在因果因子。此外，LLMs还可以被指示提供用于收集数据值（例如注释标准）的额外信息，并将原始非结构化数据进一步解析为结构化数据。注释数据将被输入到...

    Science originates with discovering new causal knowledge from a combination of known facts and observations. Traditional causal discovery approaches mainly rely on high-quality measured variables, usually given by human experts, to find causal relations. However, the causal variables are usually unavailable in a wide range of real-world applications. The rise of large language models (LLMs) that are trained to learn rich knowledge from the massive observations of the world, provides a new opportunity to assist with discovering high-level hidden variables from the raw observational data. Therefore, we introduce COAT: Causal representatiOn AssistanT. COAT incorporates LLMs as a factor proposer that extracts the potential causal factors from unstructured data. Moreover, LLMs can also be instructed to provide additional information used to collect data values (e.g., annotation criteria) and to further parse the raw unstructured data into structured data. The annotated data will be fed to a
    
[^3]: 超级-STTN：社交群体感知的时空转换网络用于人体轨迹预测与超图推理

    Hyper-STTN: Social Group-aware Spatial-Temporal Transformer Network for Human Trajectory Prediction with Hypergraph Reasoning. (arXiv:2401.06344v1 [cs.CV] CROSS LISTED)

    [http://arxiv.org/abs/2401.06344](http://arxiv.org/abs/2401.06344)

    本论文提出了Hyper-STTN，一种基于超图的时空转换网络，用于人群轨迹预测。通过构建多尺度超图来捕捉拥挤场景中的群体间相互作用，并利用空间-时间转换器来捕捉行人的成对潜在相互作用。这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    

    在各种现实世界的应用中，包括服务机器人和自动驾驶汽车，预测拥挤的意图和轨迹是至关重要的。理解环境动态是具有挑战性的，不仅因为对建模成对的空间和时间相互作用的复杂性，还因为群体间相互作用的多样性。为了解码拥挤场景中全面的成对和群体间相互作用，我们引入了Hyper-STTN，这是一种基于超图的时空转换网络，用于人群轨迹预测。在Hyper-STTN中，通过一组多尺度超图构建了拥挤的群体间相关性，这些超图具有不同的群体大小，通过基于随机游走概率的超图谱卷积进行捕捉。此外，还采用了空间-时间转换器来捕捉行人在空间-时间维度上的对照相互作用。然后，这些异构的群体间和成对间相互作用通过一个多模态转换网络进行融合和对准。

    Predicting crowded intents and trajectories is crucial in varouls real-world applications, including service robots and autonomous vehicles. Understanding environmental dynamics is challenging, not only due to the complexities of modeling pair-wise spatial and temporal interactions but also the diverse influence of group-wise interactions. To decode the comprehensive pair-wise and group-wise interactions in crowded scenarios, we introduce Hyper-STTN, a Hypergraph-based Spatial-Temporal Transformer Network for crowd trajectory prediction. In Hyper-STTN, crowded group-wise correlations are constructed using a set of multi-scale hypergraphs with varying group sizes, captured through random-walk robability-based hypergraph spectral convolution. Additionally, a spatial-temporal transformer is adapted to capture pedestrians' pair-wise latent interactions in spatial-temporal dimensions. These heterogeneous group-wise and pair-wise are then fused and aligned though a multimodal transformer net
    
[^4]: 人机协同下使用祖先GFlowNets进行潜在混淆的因果发现

    Human-in-the-Loop Causal Discovery under Latent Confounding using Ancestral GFlowNets. (arXiv:2309.12032v1 [cs.LG])

    [http://arxiv.org/abs/2309.12032](http://arxiv.org/abs/2309.12032)

    该论文提出了一种人机协同的因果发现方法，通过使用生成流网按照基于评分函数的信念分布采样祖先图，并引入最佳实验设计与专家互动，以提供专家可验证的不确定性估计并迭代改进因果推断。

    

    结构学习是因果推断的关键。值得注意的是，当数据稀缺时，因果发现（CD）算法很脆弱，可能推断出与专家知识相矛盾的不准确因果关系，尤其是考虑到潜在混淆因素时更是如此。为了加重这个问题，大多数CD方法并不提供不确定性估计，这使得用户难以解释结果和改进推断过程。令人惊讶的是，尽管CD是一个以人为中心的事务，但没有任何研究专注于构建既能输出专家可验证的不确定性估计又能与专家进行交互迭代改进的方法。为了解决这些问题，我们首先提出使用生成流网，根据基于评分函数（如贝叶斯信息准则）的信念分布，按比例对（因果）祖先图进行采样。然后，我们利用候选图的多样性并引入最佳实验设计，以迭代性地探索实验来与专家互动。

    Structure learning is the crux of causal inference. Notably, causal discovery (CD) algorithms are brittle when data is scarce, possibly inferring imprecise causal relations that contradict expert knowledge -- especially when considering latent confounders. To aggravate the issue, most CD methods do not provide uncertainty estimates, making it hard for users to interpret results and improve the inference process. Surprisingly, while CD is a human-centered affair, no works have focused on building methods that both 1) output uncertainty estimates that can be verified by experts and 2) interact with those experts to iteratively refine CD. To solve these issues, we start by proposing to sample (causal) ancestral graphs proportionally to a belief distribution based on a score function, such as the Bayesian information criterion (BIC), using generative flow networks. Then, we leverage the diversity in candidate graphs and introduce an optimal experimental design to iteratively probe the expe
    
[^5]: 伪装图像合成是提高伪装物体检测的关键

    Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection. (arXiv:2308.06701v1 [cs.CV])

    [http://arxiv.org/abs/2308.06701](http://arxiv.org/abs/2308.06701)

    该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。

    

    融入自然场景的伪装物体给深度学习模型检测和合成带来了重大挑战。伪装物体检测是计算机视觉中一个关键任务，具有广泛的实际应用，然而由于数据有限，该研究课题一直受到限制。我们提出了一个用于合成伪装数据以增强对自然场景中伪装物体检测的框架。我们的方法利用生成模型生成逼真的伪装图像，这些图像可以用来训练现有的物体检测模型。具体而言，我们使用伪装环境生成器，由伪装分布分类器进行监督，合成伪装图像，然后将其输入我们的生成器以扩展数据集。我们的框架在三个数据集（COD10k、CAMO和CHAMELEON）上的效果超过了目前最先进的方法，证明了它在改善伪装物体检测方面的有效性。

    Camouflaged objects that blend into natural scenes pose significant challenges for deep-learning models to detect and synthesize. While camouflaged object detection is a crucial task in computer vision with diverse real-world applications, this research topic has been constrained by limited data availability. We propose a framework for synthesizing camouflage data to enhance the detection of camouflaged objects in natural scenes. Our approach employs a generative model to produce realistic camouflage images, which can be used to train existing object detection models. Specifically, we use a camouflage environment generator supervised by a camouflage distribution classifier to synthesize the camouflage images, which are then fed into our generator to expand the dataset. Our framework outperforms the current state-of-the-art method on three datasets (COD10k, CAMO, and CHAMELEON), demonstrating its effectiveness in improving camouflaged object detection. This approach can serve as a plug-
    
[^6]: 面向隐私的高斯过程回归

    Privacy-aware Gaussian Process Regression. (arXiv:2305.16541v1 [cs.LG])

    [http://arxiv.org/abs/2305.16541](http://arxiv.org/abs/2305.16541)

    以高斯过程回归为基础，我们提出了实现数据隐私保护的方法。方法将综合噪声添加到数据中，使得高斯过程预测模型达到特定的隐私级别。我们还通过核方法介绍了连续隐私约束下的隐私感知解形式，以及研究了其理论性质。所提出的方法应用于卫星轨迹跟踪模型。

    

    我们提出了第一个在隐私约束条件下的高斯过程回归的理论和方法框架。所提出的方法可以在数据所有者因隐私担忧而不愿与公众分享其从其数据构建的高保真监督学习模型时使用。所提出方法的关键思想是通过添加综合噪声来使高斯过程预测模型的预测方差达到预先指定的隐私级别。合成噪声的最优协方差矩阵以半定编程的形式给出。我们还介绍了基于核的方法来研究在连续约束隐私条件下的隐私感知解的形式，并研究了它们的理论属性。所提出的方法使用跟踪卫星轨迹的模型进行了说明。

    We propose the first theoretical and methodological framework for Gaussian process regression subject to privacy constraints. The proposed method can be used when a data owner is unwilling to share a high-fidelity supervised learning model built from their data with the public due to privacy concerns. The key idea of the proposed method is to add synthetic noise to the data until the predictive variance of the Gaussian process model reaches a prespecified privacy level. The optimal covariance matrix of the synthetic noise is formulated in terms of semi-definite programming. We also introduce the formulation of privacy-aware solutions under continuous privacy constraints using kernel-based approaches, and study their theoretical properties. The proposed method is illustrated by considering a model that tracks the trajectories of satellites.
    
[^7]: 在线损失函数学习

    Online Loss Function Learning. (arXiv:2301.13247v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13247](http://arxiv.org/abs/2301.13247)

    在线损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们的方法在多个任务上稳定地优于现有技术。

    

    损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。现有的损失函数学习技术已经显示出有希望的结果，经常改善模型的训练动态和最终推理性能。然而，这些技术的一个重要限制是损失函数以线下方式进行元学习，元目标仅考虑训练的前几个步骤，这与训练深度神经网络通常使用的时间范围相比显著较短。这导致对于在训练开始时表现良好但在训练结束时表现不佳的损失函数存在明显的偏差。为了解决这个问题，我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们提出的方法在多个任务上稳定地优于现有技术。

    Loss function learning is a new meta-learning paradigm that aims to automate the essential task of designing a loss function for a machine learning model. Existing techniques for loss function learning have shown promising results, often improving a model's training dynamics and final inference performance. However, a significant limitation of these techniques is that the loss functions are meta-learned in an offline fashion, where the meta-objective only considers the very first few steps of training, which is a significantly shorter time horizon than the one typically used for training deep neural networks. This causes significant bias towards loss functions that perform well at the very start of training but perform poorly at the end of training. To address this issue we propose a new loss function learning technique for adaptively updating the loss function online after each update to the base model parameters. The experimental results show that our proposed method consistently out
    
[^8]: 基于扩散生成模型的语音增强和去混响

    Speech Enhancement and Dereverberation with Diffusion-based Generative Models. (arXiv:2208.05830v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2208.05830](http://arxiv.org/abs/2208.05830)

    本文基于扩散生成模型，通过混合噪声语音和高斯噪声进行反向过程，仅使用30步就实现高质量的干净语音估计。调整网络架构，可以显著提高语音增强性能，达到了最新方法的竞争水平。

    

    在这项工作中，我们基于我们之前的出版物，并使用基于扩散的生成模型进行语音增强。我们详细介绍了基于随机微分方程的扩散过程，并进行了广泛的理论探讨其含义。与通常的条件生成任务相反，我们不是从纯高斯噪声开始反向过程，而是从噪声语音和高斯噪声的混合物开始。这与我们的正向过程相匹配，该过程通过包括漂移项将干净语音转变成噪声语音。我们表明，这个过程使得使用仅30个扩散步骤来生成高质量的干净语音估计成为可能。通过调整网络架构，我们能够显著提高语音增强性能，这表明网络而不是形式主义是我们原始方法的主要限制。在广泛的跨数据集评估中，我们展示了改进后的方法可以与最新的方法竞争。

    In this work, we build upon our previous publication and use diffusion-based generative models for speech enhancement. We present a detailed overview of the diffusion process that is based on a stochastic differential equation and delve into an extensive theoretical examination of its implications. Opposed to usual conditional generation tasks, we do not start the reverse process from pure Gaussian noise but from a mixture of noisy speech and Gaussian noise. This matches our forward process which moves from clean speech to noisy speech by including a drift term. We show that this procedure enables using only 30 diffusion steps to generate high-quality clean speech estimates. By adapting the network architecture, we are able to significantly improve the speech enhancement performance, indicating that the network, rather than the formalism, was the main limitation of our original approach. In an extensive cross-dataset evaluation, we show that the improved method can compete with recent 
    

