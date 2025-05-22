# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic-Aware Remote Estimation of Multiple Markov Sources Under Constraints](https://arxiv.org/abs/2403.16855) | 研究了具有语义意识的远程多马尔可夫源估计的最优调度策略以及开发了一种新颖的策略搜索算法 Insec-RVI。 |
| [^2] | [Breast Cancer Classification Using Gradient Boosting Algorithms Focusing on Reducing the False Negative and SHAP for Explainability](https://arxiv.org/abs/2403.09548) | 本研究使用梯度提升算法对乳腺癌进行分类，关注提高召回率，以实现更好的检测和预测效果。 |
| [^3] | [Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference](https://arxiv.org/abs/2403.04082) | 通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断 |
| [^4] | [Uncertainty quantification in fine-tuned LLMs using LoRA ensembles](https://arxiv.org/abs/2402.12264) | 使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。 |
| [^5] | [Learning Memory Kernels in Generalized Langevin Equations](https://arxiv.org/abs/2402.11705) | 提出一种学习广义朗之万方程中记忆核的新方法，通过正则化Prony方法估计相关函数并在Sobolev范数Loss函数和RKHS正则化下实现回归，在指数加权的$L^2$空间内获得改进性能，对比其他回归估计器展示了其优越性。 |
| [^6] | [Functional SDE approximation inspired by a deep operator network architecture](https://arxiv.org/abs/2402.03028) | 本文提出了一种受深度算子网络结构启发的函数SDE近似方法，通过深度神经网络和多项式混沌展开实现对随机微分方程解的近似，并通过学习减轻指数级复杂度的问题。 |
| [^7] | [Quantum Architecture Search with Unsupervised Representation Learning](https://arxiv.org/abs/2401.11576) | 通过利用无监督表示学习，量子架构搜索（QAS）的性能可以得以提升，而不需要耗费大量时间进行标记。 |
| [^8] | [Generalizing Medical Image Representations via Quaternion Wavelet Networks.](http://arxiv.org/abs/2310.10224) | 本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。 |
| [^9] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |
| [^10] | [HurriCast: An Automatic Framework Using Machine Learning and Statistical Modeling for Hurricane Forecasting.](http://arxiv.org/abs/2309.07174) | 本研究提出了HurriCast，一种使用机器学习和统计建模的自动化框架，通过组合ARIMA模型和K-MEANS算法以更好地捕捉飓风趋势，并结合Autoencoder进行改进的飓风模拟，从而有效模拟历史飓风行为并提供详细的未来预测。这项研究通过利用全面且有选择性的数据集，丰富了对飓风模式的理解，并为风险管理策略提供了可操作的见解。 |
| [^11] | [Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning.](http://arxiv.org/abs/2308.04964) | 这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。 |
| [^12] | [Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks.](http://arxiv.org/abs/2303.17523) | 本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。 |
| [^13] | [An Empirical Bayes Analysis of Object Trajectory Representation Models.](http://arxiv.org/abs/2211.01696) | 本论文对对象轨迹表示模型的复杂度和拟合误差之间的权衡进行了经验分析，发现简单的线性模型就能够高度重现真实世界的轨迹，通过使用经验贝叶斯方法可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。 |

# 详细

[^1]: 具有语义意识的远程多马尔可夫源估计在约束条件下的研究

    Semantic-Aware Remote Estimation of Multiple Markov Sources Under Constraints

    [https://arxiv.org/abs/2403.16855](https://arxiv.org/abs/2403.16855)

    研究了具有语义意识的远程多马尔可夫源估计的最优调度策略以及开发了一种新颖的策略搜索算法 Insec-RVI。

    

    本文研究了在丢失和受速率限制的通道上对多个马尔可夫源进行语义感知通信的远程估计。与大多数现有研究将所有源状态视为相等不同，我们利用信息的语义并考虑远程执行器对不同状态的估计误差具有不同的容忍度。我们旨在找到一个最优调度策略，以在传输频率约束下最小化估计误差的长期状态相关成本。我们从理论上通过利用平均成本约束马尔可夫决策过程（CMDP）理论和Lagrange动态规划展示了最优策略的结构。通过利用最优结构结果，我们开发了一种新颖的策略搜索算法，称为交叉搜索加相对值迭代（Insec-RVI），可以仅通过少量迭代找到最优策略。为了避免“维度诅咒”...

    arXiv:2403.16855v1 Announce Type: cross  Abstract: This paper studies semantic-aware communication for remote estimation of multiple Markov sources over a lossy and rate-constrained channel. Unlike most existing studies that treat all source states equally, we exploit the semantics of information and consider that the remote actuator has different tolerances for the estimation errors of different states. We aim to find an optimal scheduling policy that minimizes the long-term state-dependent costs of estimation errors under a transmission frequency constraint. We theoretically show the structure of the optimal policy by leveraging the average-cost Constrained Markov Decision Process (CMDP) theory and the Lagrangian dynamic programming. By exploiting the optimal structural results, we develop a novel policy search algorithm, termed intersection search plus relative value iteration (Insec-RVI), that can find the optimal policy using only a few iterations. To avoid the ``curse of dimensio
    
[^2]: 使用梯度提升算法对乳腺癌进行分类，重点减少假阴性和使用 SHAP 进行解释性研究

    Breast Cancer Classification Using Gradient Boosting Algorithms Focusing on Reducing the False Negative and SHAP for Explainability

    [https://arxiv.org/abs/2403.09548](https://arxiv.org/abs/2403.09548)

    本研究使用梯度提升算法对乳腺癌进行分类，关注提高召回率，以实现更好的检测和预测效果。

    

    癌症是世界上夺走最多女性生命的疾病之一，其中乳腺癌占据了癌症病例和死亡人数最高的位置。然而，通过早期检测可以预防乳腺癌，从而进行早期治疗。许多研究关注的是在癌症预测中具有高准确性的模型，但有时仅依靠准确性可能并非始终可靠。本研究对使用提升技术基于不同机器学习算法预测乳腺癌的性能进行了调查性研究，重点关注召回率指标。提升机器学习算法已被证明是检测医学疾病的有效工具。利用加州大学尔湾分校 (UCI) 数据集对训练和测试模型分类器进行训练，其中包含各自属性。

    arXiv:2403.09548v1 Announce Type: new  Abstract: Cancer is one of the diseases that kill the most women in the world, with breast cancer being responsible for the highest number of cancer cases and consequently deaths. However, it can be prevented by early detection and, consequently, early treatment. Any development for detection or perdition this kind of cancer is important for a better healthy life. Many studies focus on a model with high accuracy in cancer prediction, but sometimes accuracy alone may not always be a reliable metric. This study implies an investigative approach to studying the performance of different machine learning algorithms based on boosting to predict breast cancer focusing on the recall metric. Boosting machine learning algorithms has been proven to be an effective tool for detecting medical diseases. The dataset of the University of California, Irvine (UCI) repository has been utilized to train and test the model classifier that contains their attributes. Th
    
[^3]: 通过插值进行推断：对比表示可证明启用规划和推断

    Inference via Interpolation: Contrastive Representations Provably Enable Planning and Inference

    [https://arxiv.org/abs/2403.04082](https://arxiv.org/abs/2403.04082)

    通过对比学习学到的时间序列数据表示遵循高斯马尔可夫链，从而启用规划和推断

    

    给定时间序列数据，我们如何回答诸如“未来会发生什么？”和“我们是如何到达这里的？”这类概率推断问题在观测值为高维时具有挑战性。本文展示了这些问题如何通过学习表示的紧凑闭式解决方案。关键思想是将对比学习的变体应用于时间序列数据。之前的工作已经表明，通过对比学习学到的表示编码了概率比。通过将之前的工作扩展以表明表示的边际分布是高斯分布，我们随后证明表示的联合分布也是高斯分布。这些结果共同表明，通过时间对比学习学到的表示遵循高斯马尔可夫链，一种图形模型，其中对表示进行的推断（例如预测、规划）对应于反演低维分布。

    arXiv:2403.04082v1 Announce Type: new  Abstract: Given time series data, how can we answer questions like "what will happen in the future?" and "how did we get here?" These sorts of probabilistic inference questions are challenging when observations are high-dimensional. In this paper, we show how these questions can have compact, closed form solutions in terms of learned representations. The key idea is to apply a variant of contrastive learning to time series data. Prior work already shows that the representations learned by contrastive learning encode a probability ratio. By extending prior work to show that the marginal distribution over representations is Gaussian, we can then prove that joint distribution of representations is also Gaussian. Taken together, these results show that representations learned via temporal contrastive learning follow a Gauss-Markov chain, a graphical model where inference (e.g., prediction, planning) over representations corresponds to inverting a low-
    
[^4]: 使用LoRA集成在精调LLMs中的不确定性量化

    Uncertainty quantification in fine-tuned LLMs using LoRA ensembles

    [https://arxiv.org/abs/2402.12264](https://arxiv.org/abs/2402.12264)

    使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。

    

    精调大型语言模型可以提高特定任务的性能，尽管对于精调模型学到了什么、遗忘了什么以及如何信任其预测仍然缺乏一个一般的理解。我们提出了使用计算效率高的低秩适应集成对精调LLMs进行基于后验逼近的原则性不确定性量化。我们使用基于Mistral-7b的低秩适应集成分析了三个常见的多项选择数据集，并对其在精调过程中和之后对不同目标领域的感知复杂性和模型效能进行了定量和定性的结论。具体而言，基于数值实验支持，我们对那些对于给定架构难以学习的数据领域的熵不确定性度量提出了假设。

    arXiv:2402.12264v1 Announce Type: cross  Abstract: Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and model efficacy on the different target domains during and after fine-tuning. In particular, backed by the numerical experiments, we hypothesise about signals from entropic uncertainty measures for data domains that are inherently difficult for a given architecture to learn.
    
[^5]: 在广义朗之万方程中学习记忆核

    Learning Memory Kernels in Generalized Langevin Equations

    [https://arxiv.org/abs/2402.11705](https://arxiv.org/abs/2402.11705)

    提出一种学习广义朗之万方程中记忆核的新方法，通过正则化Prony方法估计相关函数并在Sobolev范数Loss函数和RKHS正则化下实现回归，在指数加权的$L^2$空间内获得改进性能，对比其他回归估计器展示了其优越性。

    

    我们引入了一种新颖的方法来学习广义朗之万方程中的记忆核。该方法最初利用正则化Prony方法从轨迹数据中估计相关函数，然后通过基于Sobolev范数的回归和RKHS正则化来进行回归。我们的方法保证在指数加权的$L^2$空间内获得了改进的性能，核估计误差受控于估计相关函数的误差。我们通过数值示例展示了我们的估计器相对于依赖于$L^2$损失函数的其他回归估计器以及从逆拉普拉斯变换推导出的估计器的优越性，这些示例突显了我们的估计器在各种权重参数选择上的持续优势。此外，我们提供了包括力和漂移项在方程中的应用示例。

    arXiv:2402.11705v1 Announce Type: cross  Abstract: We introduce a novel approach for learning memory kernels in Generalized Langevin Equations. This approach initially utilizes a regularized Prony method to estimate correlation functions from trajectory data, followed by regression over a Sobolev norm-based loss function with RKHS regularization. Our approach guarantees improved performance within an exponentially weighted $L^2$ space, with the kernel estimation error controlled by the error in estimated correlation functions. We demonstrate the superiority of our estimator compared to other regression estimators that rely on $L^2$ loss functions and also an estimator derived from the inverse Laplace transform, using numerical examples that highlight its consistent advantage across various weight parameter selections. Additionally, we provide examples that include the application of force and drift terms in the equation.
    
[^6]: 受深度算子网络结构启发的函数SDE近似方法

    Functional SDE approximation inspired by a deep operator network architecture

    [https://arxiv.org/abs/2402.03028](https://arxiv.org/abs/2402.03028)

    本文提出了一种受深度算子网络结构启发的函数SDE近似方法，通过深度神经网络和多项式混沌展开实现对随机微分方程解的近似，并通过学习减轻指数级复杂度的问题。

    

    本文提出并分析了一种通过深度神经网络近似随机微分方程（SDE）解的新方法。该结构灵感来自于深度算子网络（DeepONets）的概念，它基于函数空间中的算子学习，以及在网络中表示的降维基础。在我们的设置中，我们利用了随机过程的多项式混沌展开（PCE），并将相应的架构称为SDEONet。在参数化偏微分方程的不确定性量化（UQ）领域中，PCE被广泛使用。然而，在SDE中并非如此，传统的采样方法占主导地位，而功能性方法很少见。截断的PCE存在一个主要挑战，即随着最大多项式阶数和基函数数量的增加，分量的数量呈指数级增长。所提出的SDEONet结构旨在通过学习来减轻指数级复杂度的问题。

    A novel approach to approximate solutions of Stochastic Differential Equations (SDEs) by Deep Neural Networks is derived and analysed. The architecture is inspired by the notion of Deep Operator Networks (DeepONets), which is based on operator learning in function spaces in terms of a reduced basis also represented in the network. In our setting, we make use of a polynomial chaos expansion (PCE) of stochastic processes and call the corresponding architecture SDEONet. The PCE has been used extensively in the area of uncertainty quantification (UQ) with parametric partial differential equations. This however is not the case with SDE, where classical sampling methods dominate and functional approaches are seen rarely. A main challenge with truncated PCEs occurs due to the drastic growth of the number of components with respect to the maximum polynomial degree and the number of basis elements. The proposed SDEONet architecture aims to alleviate the issue of exponential complexity by learni
    
[^7]: 利用无监督表示学习进行量子架构搜索

    Quantum Architecture Search with Unsupervised Representation Learning

    [https://arxiv.org/abs/2401.11576](https://arxiv.org/abs/2401.11576)

    通过利用无监督表示学习，量子架构搜索（QAS）的性能可以得以提升，而不需要耗费大量时间进行标记。

    

    使用无监督表示学习进行量子架构搜索（QAS）代表了一种前沿方法，有望在嘈杂的中间规模量子（NISQ）设备上实现潜在的量子优势。大多数QAS算法将它们的搜索空间和搜索算法结合在一起，因此通常需要在搜索过程中评估大量的量子电路。基于预测的QAS算法可以通过直接根据电路结构估计电路的性能来缓解这个问题。然而，高性能的预测器通常需要耗费大量时间进行标记，以获得大量带标签的量子电路。最近，一个经典的神经架构搜索算法Arch2vec启发我们，表明架构搜索可以从将无监督表示学习与搜索过程分离中获益。无监督表示学习是否能帮助QAS

    arXiv:2401.11576v2 Announce Type: replace-cross  Abstract: Utilizing unsupervised representation learning for quantum architecture search (QAS) represents a cutting-edge approach poised to realize potential quantum advantage on Noisy Intermediate-Scale Quantum (NISQ) devices. Most QAS algorithms combine their search space and search algorithms together and thus generally require evaluating a large number of quantum circuits during the search process. Predictor-based QAS algorithms can alleviate this problem by directly estimating the performance of circuits according to their structures. However, a high-performance predictor generally requires very time-consuming labeling to obtain a large number of labeled quantum circuits. Recently, a classical neural architecture search algorithm Arch2vec inspires us by showing that architecture search can benefit from decoupling unsupervised representation learning from the search process. Whether unsupervised representation learning can help QAS w
    
[^8]: 通过四元数小波网络推广医学图像表示

    Generalizing Medical Image Representations via Quaternion Wavelet Networks. (arXiv:2310.10224v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2310.10224](http://arxiv.org/abs/2310.10224)

    本文提出了一种名为QUAVE的四元数小波网络，可以从医学图像中提取显著特征。该网络可以与现有的医学图像分析或综合任务结合使用，并推广了对单通道数据的采用。通过四元数小波变换和加权处理，QUAVE能够处理具有较大变化的医学数据。

    

    鉴于来自不同来源和各种任务的数据集日益增加，神经网络的普适性成为一个广泛研究的领域。当处理医学数据时，这个问题尤为广泛，因为缺乏方法论标准导致不同的成像中心或使用不同设备和辅助因素获取的数据存在较大变化。为了克服这些限制，我们引入了一种新颖的、普适的、数据-和任务不可知的框架，能够从医学图像中提取显著特征。所提出的四元数小波网络（QUAVE）可以很容易地与任何现有的医学图像分析或综合任务相结合，并且可以结合实际、四元数或超复值模型，推广它们对单通道数据的采用。QUAVE首先通过四元数小波变换提取不同的子带，得到低频/近似频带和高频/细粒度特征。然后，它对最有代表性的特征进行加权处理，从而减少了特征重要性不均匀性。

    Neural network generalizability is becoming a broad research field due to the increasing availability of datasets from different sources and for various tasks. This issue is even wider when processing medical data, where a lack of methodological standards causes large variations being provided by different imaging centers or acquired with various devices and cofactors. To overcome these limitations, we introduce a novel, generalizable, data- and task-agnostic framework able to extract salient features from medical images. The proposed quaternion wavelet network (QUAVE) can be easily integrated with any pre-existing medical image analysis or synthesis task, and it can be involved with real, quaternion, or hypercomplex-valued models, generalizing their adoption to single-channel data. QUAVE first extracts different sub-bands through the quaternion wavelet transform, resulting in both low-frequency/approximation bands and high-frequency/fine-grained features. Then, it weighs the most repr
    
[^9]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    
[^10]: HurriCast：使用机器学习和统计建模的自动化框架用于飓风预测

    HurriCast: An Automatic Framework Using Machine Learning and Statistical Modeling for Hurricane Forecasting. (arXiv:2309.07174v1 [cs.LG])

    [http://arxiv.org/abs/2309.07174](http://arxiv.org/abs/2309.07174)

    本研究提出了HurriCast，一种使用机器学习和统计建模的自动化框架，通过组合ARIMA模型和K-MEANS算法以更好地捕捉飓风趋势，并结合Autoencoder进行改进的飓风模拟，从而有效模拟历史飓风行为并提供详细的未来预测。这项研究通过利用全面且有选择性的数据集，丰富了对飓风模式的理解，并为风险管理策略提供了可操作的见解。

    

    飓风由于其灾害性影响而在美国面临重大挑战。减轻这些风险很重要，保险业在这方面起着重要作用，使用复杂的统计模型进行风险评估。然而，这些模型常常忽视关键的时间和空间飓风模式，并受到数据稀缺的限制。本研究引入了一种改进的方法，将ARIMA模型和K-MEANS相结合，以更好地捕捉飓风趋势，并使用Autoencoder进行改进的飓风模拟。我们的实验证明，这种混合方法有效地模拟了历史飓风行为，同时提供了潜在未来路径和强度的详细预测。此外，通过利用全面而有选择性的数据集，我们的模拟丰富了对飓风模式的当前理解，并为风险管理策略提供了可操作的见解。

    Hurricanes present major challenges in the U.S. due to their devastating impacts. Mitigating these risks is important, and the insurance industry is central in this effort, using intricate statistical models for risk assessment. However, these models often neglect key temporal and spatial hurricane patterns and are limited by data scarcity. This study introduces a refined approach combining the ARIMA model and K-MEANS to better capture hurricane trends, and an Autoencoder for enhanced hurricane simulations. Our experiments show that this hybrid methodology effectively simulate historical hurricane behaviors while providing detailed projections of potential future trajectories and intensities. Moreover, by leveraging a comprehensive yet selective dataset, our simulations enrich the current understanding of hurricane patterns and offer actionable insights for risk management strategies.
    
[^11]: Adversarial ModSecurity: 使用强大的机器学习对抗SQL注入攻击

    Adversarial ModSecurity: Countering Adversarial SQL Injections with Robust Machine Learning. (arXiv:2308.04964v1 [cs.LG])

    [http://arxiv.org/abs/2308.04964](http://arxiv.org/abs/2308.04964)

    这篇论文介绍了Adversarial ModSecurity，它是一个使用强大的机器学习来对抗SQL注入攻击的防火墙。通过将核心规则集作为输入特征，该模型可以识别并防御对抗性SQL注入攻击。实验结果表明，AdvModSec在训练后能够有效地应对这类攻击。

    

    ModSecurity被广泛认可为标准的开源Web应用防火墙(WAF)，由OWASP基金会维护。它通过与核心规则集进行匹配来检测恶意请求，识别出常见的攻击模式。每个规则在CRS中都被手动分配一个权重，基于相应攻击的严重程度，如果触发规则的权重之和超过给定的阈值，就会被检测为恶意请求。然而，我们的研究表明，这种简单的策略在检测SQL注入攻击方面很不有效，因为它往往会阻止许多合法请求，同时还容易受到对抗性SQL注入攻击的影响，即故意操纵以逃避检测的攻击。为了克服这些问题，我们设计了一个名为AdvModSec的强大机器学习模型，它将CRS规则作为输入特征，并经过训练以检测对抗性SQL注入攻击。我们的实验表明，AdvModSec在针对该攻击的流量上进行训练后表现出色。

    ModSecurity is widely recognized as the standard open-source Web Application Firewall (WAF), maintained by the OWASP Foundation. It detects malicious requests by matching them against the Core Rule Set, identifying well-known attack patterns. Each rule in the CRS is manually assigned a weight, based on the severity of the corresponding attack, and a request is detected as malicious if the sum of the weights of the firing rules exceeds a given threshold. In this work, we show that this simple strategy is largely ineffective for detecting SQL injection (SQLi) attacks, as it tends to block many legitimate requests, while also being vulnerable to adversarial SQLi attacks, i.e., attacks intentionally manipulated to evade detection. To overcome these issues, we design a robust machine learning model, named AdvModSec, which uses the CRS rules as input features, and it is trained to detect adversarial SQLi attacks. Our experiments show that AdvModSec, being trained on the traffic directed towa
    
[^12]: 利用长短期记忆网络提高量子电路保真度

    Quantum Circuit Fidelity Improvement with Long Short-Term Memory Networks. (arXiv:2303.17523v1 [quant-ph])

    [http://arxiv.org/abs/2303.17523](http://arxiv.org/abs/2303.17523)

    本文提出使用长短期记忆网络解决量子计算中的保真度问题，利用时间序列预测方法预测量子电路的保真度。

    

    量子计算已进入噪声中间规模量子（NISQ）时代，目前我们拥有的量子处理器对辐射和温度等环境变量敏感，因此会产生嘈杂的输出。虽然已经有许多算法和应用程序用于NISQ处理器，但我们仍面临着解释其嘈杂结果的不确定性。具体来说，我们对所选择的量子态有多少信心？这种信心很重要，因为NISQ计算机将输出其量子位测量的概率分布，有时很难区分分布是否表示有意义的计算或只是随机噪声。本文提出了一种新方法来解决这个问题，将量子电路保真度预测框架为时间序列预测问题，因此可以利用长短期记忆（LSTM）神经网络的强大能力。一个完整的工作流程来构建训练电路

    Quantum computing has entered the Noisy Intermediate-Scale Quantum (NISQ) era. Currently, the quantum processors we have are sensitive to environmental variables like radiation and temperature, thus producing noisy outputs. Although many proposed algorithms and applications exist for NISQ processors, we still face uncertainties when interpreting their noisy results. Specifically, how much confidence do we have in the quantum states we are picking as the output? This confidence is important since a NISQ computer will output a probability distribution of its qubit measurements, and it is sometimes hard to distinguish whether the distribution represents meaningful computation or just random noise. This paper presents a novel approach to attack this problem by framing quantum circuit fidelity prediction as a Time Series Forecasting problem, therefore making it possible to utilize the power of Long Short-Term Memory (LSTM) neural networks. A complete workflow to build the training circuit d
    
[^13]: 对象轨迹表示模型的经验贝叶斯分析

    An Empirical Bayes Analysis of Object Trajectory Representation Models. (arXiv:2211.01696v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.01696](http://arxiv.org/abs/2211.01696)

    本论文对对象轨迹表示模型的复杂度和拟合误差之间的权衡进行了经验分析，发现简单的线性模型就能够高度重现真实世界的轨迹，通过使用经验贝叶斯方法可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。

    

    我们对对象轨迹建模中的模型复杂度和拟合误差之间的权衡进行了深入的经验分析。通过分析多个大型公共数据集，我们发现简单的线性模型在相关时间范围内使用较少的模型复杂度就能够高度重现真实世界的轨迹。这一发现允许将轨迹跟踪和预测作为贝叶斯过滤问题进行公式化。我们采用经验贝叶斯方法，从数据中估计模型参数的先验分布，这些先验分布可以为轨迹跟踪问题中必要的运动模型提供信息，并可以帮助规范预测模型。我们主张在轨迹预测任务中使用线性轨迹表示模型，因为它们目前并不会限制预测性能。

    We present an in-depth empirical analysis of the trade-off between model complexity and fit error in modelling object trajectories. Analyzing several large public datasets, we show that simple linear models do represent real-world trajectories with high fidelity over relevant time scales at very moderate model complexity. This finding allows the formulation of trajectory tracking and prediction as a Bayesian filtering problem. Using an Empirical Bayes approach, we estimate prior distributions over model parameters from the data. These prior distributions inform the motion models necessary in the trajectory tracking problem and can help regularize prediction models. We argue for the use of linear trajectory representation models in trajectory prediction tasks as they do not limit prediction performance currently.
    

