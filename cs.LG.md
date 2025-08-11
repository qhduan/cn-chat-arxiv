# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Tuning for the Unknown: Revisiting Evaluation Strategies for Lifelong RL](https://arxiv.org/abs/2404.02113) | 提出了一种新方法来调整和评估终身强化学习代理，在此方法中，只有实验数据的一小部分可用于超参数调整，针对终身强化学习的研究进展可能被不当的经验方法所阻碍 |
| [^2] | [Data Collaboration Analysis Over Matrix Manifolds](https://arxiv.org/abs/2403.02780) | 本研究讨论了在矩阵流形上的数据协作分析，探讨了如何通过隐私保护机器学习来处理多来源数据的道德和隐私问题 |
| [^3] | [Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift](https://arxiv.org/abs/2402.10665) | 本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性 |
| [^4] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |
| [^5] | [Diffusion Random Feature Model.](http://arxiv.org/abs/2310.04417) | 本研究提出了一种以扩散模型为灵感的深度随机特征模型，它具有可解释性并可在数量相同的可训练参数下与全连接神经网络提供可比较的数值结果。通过推导得分匹配的属性，我们扩展了现有随机特征结果，并得出了样本数据分布与真实分布之间的泛化边界。 |
| [^6] | [Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning.](http://arxiv.org/abs/2309.09222) | 这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。 |
| [^7] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |
| [^8] | [Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools.](http://arxiv.org/abs/2308.02613) | 本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。 |
| [^9] | [MESAHA-Net: Multi-Encoders based Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation in CT Scan.](http://arxiv.org/abs/2304.01576) | 本文提出了一种名为MESAHA-Net的高效端到端框架，集成了三种类型的输入，通过采用自适应硬注意力机制，逐层2D分割，实现了 CT扫描中精确的肺结节分割。 |

# 详细

[^1]: 针对未知进行调整：重新审视终身强化学习的评估策略

    Tuning for the Unknown: Revisiting Evaluation Strategies for Lifelong RL

    [https://arxiv.org/abs/2404.02113](https://arxiv.org/abs/2404.02113)

    提出了一种新方法来调整和评估终身强化学习代理，在此方法中，只有实验数据的一小部分可用于超参数调整，针对终身强化学习的研究进展可能被不当的经验方法所阻碍

    

    在继续或终身强化学习中，对环境的访问应该是有限的。如果我们希望设计的算法能够长时间运行，并不断适应新的、意想不到的情况，那么我们必须愿意在整个代理的整个生命周期内部署我们的代理而不调整它们的超参数。本文探讨了深度强化学习中 -- 甚至继续强化学习中 -- 具备对代理的部署环境具有无限制访问权的标准做法可能已经阻碍了对终身强化学习研究的进展。在本文中，我们提出了一种新的方法，用于调整和评估终身强化学习代理，其中只有实验数据的百分之一可以用于超参数调整。然后，我们对DQN和Soft Actor Critic在各种持续和非稳定领域进行了实证研究。我们发现这两种方法通常表现较好。

    arXiv:2404.02113v1 Announce Type: new  Abstract: In continual or lifelong reinforcement learning access to the environment should be limited. If we aspire to design algorithms that can run for long-periods of time, continually adapting to new, unexpected situations then we must be willing to deploy our agents without tuning their hyperparameters over the agent's entire lifetime. The standard practice in deep RL -- and even continual RL -- is to assume unfettered access to deployment environment for the full lifetime of the agent. This paper explores the notion that progress in lifelong RL research has been held back by inappropriate empirical methodologies. In this paper we propose a new approach for tuning and evaluating lifelong RL agents where only one percent of the experiment data can be used for hyperparameter tuning. We then conduct an empirical study of DQN and Soft Actor Critic across a variety of continuing and non-stationary domains. We find both methods generally perform po
    
[^2]: 矩阵流形上的数据协作分析

    Data Collaboration Analysis Over Matrix Manifolds

    [https://arxiv.org/abs/2403.02780](https://arxiv.org/abs/2403.02780)

    本研究讨论了在矩阵流形上的数据协作分析，探讨了如何通过隐私保护机器学习来处理多来源数据的道德和隐私问题

    

    机器学习(ML)算法的有效性与其训练数据集的质量和多样性密切相关。改进的数据集，标志着优越的质量，增强了预测的准确性，并扩展了模型在各种场景下的适用性。研究人员经常整合来自多个来源的数据，以减轻单一来源数据集的偏见和限制。然而，这种广泛的数据融合引发了重大的道德关切，特别是关于用户隐私和未经授权的数据披露风险。已建立了各种全球立法框架来解决这些隐私问题。虽然这些法规对保护隐私至关重要，但它们可能会使ML技术的实际部署变得复杂。隐私保护机器学习(PPML)通过保护从健康记录到地理位置数据等敏感信息，同时实现安全使用这些信息，来应对这一挑战。

    arXiv:2403.02780v1 Announce Type: new  Abstract: The effectiveness of machine learning (ML) algorithms is deeply intertwined with the quality and diversity of their training datasets. Improved datasets, marked by superior quality, enhance the predictive accuracy and broaden the applicability of models across varied scenarios. Researchers often integrate data from multiple sources to mitigate biases and limitations of single-source datasets. However, this extensive data amalgamation raises significant ethical concerns, particularly regarding user privacy and the risk of unauthorized data disclosure. Various global legislative frameworks have been established to address these privacy issues. While crucial for safeguarding privacy, these regulations can complicate the practical deployment of ML technologies. Privacy-Preserving Machine Learning (PPML) addresses this challenge by safeguarding sensitive information, from health records to geolocation data, while enabling the secure use of th
    
[^3]: 使用事后置信度估计的选择性预测在语义分割中的性能及其在分布偏移下的表现

    Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift

    [https://arxiv.org/abs/2402.10665](https://arxiv.org/abs/2402.10665)

    本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性

    

    语义分割在各种计算机视觉应用中扮演着重要角色，然而其有效性常常受到高质量标记数据的缺乏所限。为了解决这一挑战，一个常见策略是利用在不同种群上训练的模型，如公开可用的数据集。然而，这种方法导致了分布偏移问题，在兴趣种群上表现出降低的性能。在模型错误可能带来重大后果的情况下，选择性预测方法提供了一种减轻风险、减少对专家监督依赖的手段。本文研究了在资源匮乏环境下语义分割的选择性预测，着重于应用于在分布偏移下运行的预训练模型的事后置信度估计器。我们提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性。

    arXiv:2402.10665v1 Announce Type: new  Abstract: Semantic segmentation plays a crucial role in various computer vision applications, yet its efficacy is often hindered by the lack of high-quality labeled data. To address this challenge, a common strategy is to leverage models trained on data from different populations, such as publicly available datasets. This approach, however, leads to the distribution shift problem, presenting a reduced performance on the population of interest. In scenarios where model errors can have significant consequences, selective prediction methods offer a means to mitigate risks and reduce reliance on expert supervision. This paper investigates selective prediction for semantic segmentation in low-resource settings, thus focusing on post-hoc confidence estimators applied to pre-trained models operating under distribution shift. We propose a novel image-level confidence measure tailored for semantic segmentation and demonstrate its effectiveness through expe
    
[^4]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    
[^5]: 扩散随机特征模型

    Diffusion Random Feature Model. (arXiv:2310.04417v1 [stat.ML])

    [http://arxiv.org/abs/2310.04417](http://arxiv.org/abs/2310.04417)

    本研究提出了一种以扩散模型为灵感的深度随机特征模型，它具有可解释性并可在数量相同的可训练参数下与全连接神经网络提供可比较的数值结果。通过推导得分匹配的属性，我们扩展了现有随机特征结果，并得出了样本数据分布与真实分布之间的泛化边界。

    

    扩散概率模型已成功用于生成从噪声中产生的数据。然而，大多数扩散模型计算成本高昂，难以解释，缺乏理论依据。另一方面，由于其可解释性，随机特征模型变得越来越受欢迎，但其在复杂机器学习任务中的应用仍然有限。在本工作中，我们提出了一种受扩散模型启发的深度随机特征模型，它既具有可解释性，又能给出与具有相同可训练参数数量的全连接神经网络相当的数值结果。具体而言，我们扩展了现有的随机特征结果，利用得分匹配的属性导出了样本数据分布与真实分布之间的泛化边界。我们通过在时尚MNIST数据集和乐器音频数据上生成样本来验证我们的发现。

    Diffusion probabilistic models have been successfully used to generate data from noise. However, most diffusion models are computationally expensive and difficult to interpret with a lack of theoretical justification. Random feature models on the other hand have gained popularity due to their interpretability but their application to complex machine learning tasks remains limited. In this work, we present a diffusion model-inspired deep random feature model that is interpretable and gives comparable numerical results to a fully connected neural network having the same number of trainable parameters. Specifically, we extend existing results for random features and derive generalization bounds between the distribution of sampled data and the true distribution using properties of score matching. We validate our findings by generating samples on the fashion MNIST dataset and instrumental audio data.
    
[^6]: 双重标准化流：灵活的贝叶斯高斯过程ODE学习

    Double Normalizing Flows: Flexible Bayesian Gaussian Process ODEs Learning. (arXiv:2309.09222v1 [cs.LG])

    [http://arxiv.org/abs/2309.09222](http://arxiv.org/abs/2309.09222)

    这项研究将标准化流引入高斯过程常微分方程(ODE)模型，使其具备更灵活和表达性强的先验分布和非高斯的后验推断，从而提高了贝叶斯高斯过程ODE的准确性和不确定性估计。

    

    最近，高斯过程被用来建模连续动力系统的向量场。对于这样的模型，贝叶斯推断已经得到了广泛研究，并应用于时间序列预测等任务，提供不确定性估计。然而，先前的高斯过程常微分方程(ODE)模型在具有非高斯过程先验的数据集上可能表现不佳，因为它们的约束先验和均值场后验可能缺乏灵活性。为了解决这个限制，我们引入了标准化流来重新参数化ODE的向量场，从而得到一个更灵活、更表达性的先验分布。此外，由于标准化流的解析可计算的概率密度函数，我们将它们应用于GP ODE的后验推断，生成一个非高斯的后验。通过这些标准化流的双重应用，我们的模型在贝叶斯高斯过程ODE中提高了准确性和不确定性估计。

    Recently, Gaussian processes have been utilized to model the vector field of continuous dynamical systems. Bayesian inference for such models \cite{hegde2022variational} has been extensively studied and has been applied in tasks such as time series prediction, providing uncertain estimates. However, previous Gaussian Process Ordinary Differential Equation (ODE) models may underperform on datasets with non-Gaussian process priors, as their constrained priors and mean-field posteriors may lack flexibility. To address this limitation, we incorporate normalizing flows to reparameterize the vector field of ODEs, resulting in a more flexible and expressive prior distribution. Additionally, due to the analytically tractable probability density functions of normalizing flows, we apply them to the posterior inference of GP ODEs, generating a non-Gaussian posterior. Through these dual applications of normalizing flows, our model improves accuracy and uncertainty estimates for Bayesian Gaussian P
    
[^7]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    
[^8]: 用SyntHIR实现互操作性合成健康数据，以便开发CDSS工具

    Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools. (arXiv:2308.02613v1 [cs.LG])

    [http://arxiv.org/abs/2308.02613](http://arxiv.org/abs/2308.02613)

    本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。

    

    利用高质量的患者日志和健康登记来开发基于机器学习的临床决策支持系统（CDSS）有很大的机会。为了在临床工作流程中实施CDSS工具，需要将该工具集成、验证和测试在用于存储和管理患者数据的电子健康记录（EHR）系统上。然而，由于合规法规，通常不可能获得对EHR系统的必要访问权限。我们提出了一种用于生成和使用CDSS工具开发的合成EHR数据的体系架构。该体系结构在一个称为SyntHIR的系统中实现。SyntHIR系统使用Fast Healthcare Interoperability Resources (FHIR)标准进行数据互操作性，使用Gretel框架生成合成数据，使用Microsoft Azure FHIR服务器作为基于FHIR的EHR系统，以及使用SMART on FHIR框架进行工具可迁移性。我们通过使用数据开发机器学习基于CDSS工具来展示SyntHIR的实用性。

    There is a great opportunity to use high-quality patient journals and health registers to develop machine learning-based Clinical Decision Support Systems (CDSS). To implement a CDSS tool in a clinical workflow, there is a need to integrate, validate and test this tool on the Electronic Health Record (EHR) systems used to store and manage patient data. However, it is often not possible to get the necessary access to an EHR system due to legal compliance. We propose an architecture for generating and using synthetic EHR data for CDSS tool development. The architecture is implemented in a system called SyntHIR. The SyntHIR system uses the Fast Healthcare Interoperability Resources (FHIR) standards for data interoperability, the Gretel framework for generating synthetic data, the Microsoft Azure FHIR server as the FHIR-based EHR system and SMART on FHIR framework for tool transportability. We demonstrate the usefulness of SyntHIR by developing a machine learning-based CDSS tool using data
    
[^9]: 基于多编码器的最大强度投影自适应硬注意力网络的CT扫描肺结节分割 MESAHA-Net（arXiv：2304.01576v1 [eess.IV]）

    MESAHA-Net: Multi-Encoders based Self-Adaptive Hard Attention Network with Maximum Intensity Projections for Lung Nodule Segmentation in CT Scan. (arXiv:2304.01576v1 [eess.IV])

    [http://arxiv.org/abs/2304.01576](http://arxiv.org/abs/2304.01576)

    本文提出了一种名为MESAHA-Net的高效端到端框架，集成了三种类型的输入，通过采用自适应硬注意力机制，逐层2D分割，实现了 CT扫描中精确的肺结节分割。

    

    准确的肺结节分割对早期肺癌诊断非常重要，因为它可以大大提高患者的生存率。计算机断层扫描（CT）图像被广泛用于肺结节分析的早期诊断。然而，肺结节的异质性，大小多样性以及周围环境的复杂性对开发鲁棒的结节分割方法提出了挑战。在本研究中，我们提出了一个高效的端到端框架，即基于多编码器的自适应硬注意力网络（MESAHA-Net），用于CT扫描中精确的肺结节分割。MESAHA-Net包括三个编码路径，一个注意力块和一个解码器块，有助于集成三种类型的输入：CT切片补丁，前向和后向的最大强度投影（MIP）图像以及包含结节的感兴趣区域（ROI）掩码。通过采用新颖的自适应硬注意力机制，MESAHA-Net逐层执行逐层2D分割。

    Accurate lung nodule segmentation is crucial for early-stage lung cancer diagnosis, as it can substantially enhance patient survival rates. Computed tomography (CT) images are widely employed for early diagnosis in lung nodule analysis. However, the heterogeneity of lung nodules, size diversity, and the complexity of the surrounding environment pose challenges for developing robust nodule segmentation methods. In this study, we propose an efficient end-to-end framework, the multi-encoder-based self-adaptive hard attention network (MESAHA-Net), for precise lung nodule segmentation in CT scans. MESAHA-Net comprises three encoding paths, an attention block, and a decoder block, facilitating the integration of three types of inputs: CT slice patches, forward and backward maximum intensity projection (MIP) images, and region of interest (ROI) masks encompassing the nodule. By employing a novel adaptive hard attention mechanism, MESAHA-Net iteratively performs slice-by-slice 2D segmentation 
    

