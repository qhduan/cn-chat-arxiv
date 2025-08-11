# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Improved DDIM Sampling with Moment Matching Gaussian Mixtures.](http://arxiv.org/abs/2311.04938) | 在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。 |
| [^2] | [Hypergraph Structure Inference From Data Under Smoothness Prior.](http://arxiv.org/abs/2308.14172) | 本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。 |
| [^3] | [Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools.](http://arxiv.org/abs/2308.02613) | 本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。 |

# 详细

[^1]: 使用矩匹配高斯混合模型改进了DDIM采样

    Improved DDIM Sampling with Moment Matching Gaussian Mixtures. (arXiv:2311.04938v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2311.04938](http://arxiv.org/abs/2311.04938)

    在DDIM框架中使用GMM作为反向转移算子，通过矩匹配可以获得质量更高的样本。在无条件模型和类条件模型上进行了实验，并通过FID和IS指标证明了我们的方法的改进效果。

    

    我们提出在Denoising Diffusion Implicit Models (DDIM)框架中使用高斯混合模型（GMM）作为反向转移算子（内核），这是一种从预训练的Denoising Diffusion Probabilistic Models (DDPM)中加速采样的广泛应用方法之一。具体而言，我们通过约束GMM的参数，匹配DDPM前向边际的一阶和二阶中心矩。我们发现，通过矩匹配，可以获得与使用高斯核的原始DDIM相同或更好质量的样本。我们在CelebAHQ和FFHQ的无条件模型以及ImageNet数据集的类条件模型上提供了实验结果。我们的结果表明，在采样步骤较少的情况下，使用GMM内核可以显著改善生成样本的质量，这是通过FID和IS指标衡量的。例如，在ImageNet 256x256上，使用10个采样步骤，我们实现了一个FID值为...

    We propose using a Gaussian Mixture Model (GMM) as reverse transition operator (kernel) within the Denoising Diffusion Implicit Models (DDIM) framework, which is one of the most widely used approaches for accelerated sampling from pre-trained Denoising Diffusion Probabilistic Models (DDPM). Specifically we match the first and second order central moments of the DDPM forward marginals by constraining the parameters of the GMM. We see that moment matching is sufficient to obtain samples with equal or better quality than the original DDIM with Gaussian kernels. We provide experimental results with unconditional models trained on CelebAHQ and FFHQ and class-conditional models trained on ImageNet datasets respectively. Our results suggest that using the GMM kernel leads to significant improvements in the quality of the generated samples when the number of sampling steps is small, as measured by FID and IS metrics. For example on ImageNet 256x256, using 10 sampling steps, we achieve a FID of
    
[^2]: 从数据中基于光滑性先验推断超图结构

    Hypergraph Structure Inference From Data Under Smoothness Prior. (arXiv:2308.14172v1 [cs.LG])

    [http://arxiv.org/abs/2308.14172](http://arxiv.org/abs/2308.14172)

    本文提出了一种光滑性先验方法，用于从节点特征中推断超图的结构，并捕捉数据内在的关系。该方法不需要标记数据作为监督，能够推断出每个潜在超边的概率。

    

    超图在处理涉及多个实体的高阶关系数据中非常重要。在没有明确超图可用的情况下，希望能够从节点特征中推断出有意义的超图结构，以捕捉数据内在的关系。然而，现有的方法要么采用简单预定义的规则，不能精确捕捉潜在超图结构的分布，要么学习超图结构和节点特征之间的映射，但需要大量标记数据（即预先存在的超图结构）进行训练。这两种方法都局限于实际情景中的应用。为了填补这一空白，我们提出了一种新的光滑性先验，使我们能够设计一种方法，在没有标记数据作为监督的情况下推断出每个潜在超边的概率。所提出的先验表示超边中的节点特征与包含该超边的超边的特征高度相关。

    Hypergraphs are important for processing data with higher-order relationships involving more than two entities. In scenarios where explicit hypergraphs are not readily available, it is desirable to infer a meaningful hypergraph structure from the node features to capture the intrinsic relations within the data. However, existing methods either adopt simple pre-defined rules that fail to precisely capture the distribution of the potential hypergraph structure, or learn a mapping between hypergraph structures and node features but require a large amount of labelled data, i.e., pre-existing hypergraph structures, for training. Both restrict their applications in practical scenarios. To fill this gap, we propose a novel smoothness prior that enables us to design a method to infer the probability for each potential hyperedge without labelled data as supervision. The proposed prior indicates features of nodes in a hyperedge are highly correlated by the features of the hyperedge containing th
    
[^3]: 用SyntHIR实现互操作性合成健康数据，以便开发CDSS工具

    Interoperable synthetic health data with SyntHIR to enable the development of CDSS tools. (arXiv:2308.02613v1 [cs.LG])

    [http://arxiv.org/abs/2308.02613](http://arxiv.org/abs/2308.02613)

    本论文提出了一种利用合成EHR数据开发CDSS工具的体系架构，通过使用SyntHIR系统和FHIR标准实现数据互操作性和工具可迁移性。

    

    利用高质量的患者日志和健康登记来开发基于机器学习的临床决策支持系统（CDSS）有很大的机会。为了在临床工作流程中实施CDSS工具，需要将该工具集成、验证和测试在用于存储和管理患者数据的电子健康记录（EHR）系统上。然而，由于合规法规，通常不可能获得对EHR系统的必要访问权限。我们提出了一种用于生成和使用CDSS工具开发的合成EHR数据的体系架构。该体系结构在一个称为SyntHIR的系统中实现。SyntHIR系统使用Fast Healthcare Interoperability Resources (FHIR)标准进行数据互操作性，使用Gretel框架生成合成数据，使用Microsoft Azure FHIR服务器作为基于FHIR的EHR系统，以及使用SMART on FHIR框架进行工具可迁移性。我们通过使用数据开发机器学习基于CDSS工具来展示SyntHIR的实用性。

    There is a great opportunity to use high-quality patient journals and health registers to develop machine learning-based Clinical Decision Support Systems (CDSS). To implement a CDSS tool in a clinical workflow, there is a need to integrate, validate and test this tool on the Electronic Health Record (EHR) systems used to store and manage patient data. However, it is often not possible to get the necessary access to an EHR system due to legal compliance. We propose an architecture for generating and using synthetic EHR data for CDSS tool development. The architecture is implemented in a system called SyntHIR. The SyntHIR system uses the Fast Healthcare Interoperability Resources (FHIR) standards for data interoperability, the Gretel framework for generating synthetic data, the Microsoft Azure FHIR server as the FHIR-based EHR system and SMART on FHIR framework for tool transportability. We demonstrate the usefulness of SyntHIR by developing a machine learning-based CDSS tool using data
    

