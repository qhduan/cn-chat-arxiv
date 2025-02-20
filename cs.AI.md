# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Addressing the Regulatory Gap: Moving Towards an EU AI Audit Ecosystem Beyond the AIA by Including Civil Society](https://arxiv.org/abs/2403.07904) | 提出了一个融合合规和监督的AI审计生态系统，强调了DSA和AIA监管框架中存在的监管空白，并要求AIA为研究人员和社会公民提供数据和模型访问权限 |
| [^2] | [Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior.](http://arxiv.org/abs/2312.11535) | 该论文提出了一种使用主体特定知识先验的两阶段方法，通过考虑阴影模式和纹理增强来生成高质量、有丰富纹理的3D模型，与以前的方法相比具有显著的优势。 |
| [^3] | [Castor: Causal Temporal Regime Structure Learning.](http://arxiv.org/abs/2311.01412) | “Castor”是一个用于学习多元时间序列数据中因果关系的框架，能够综合学习各个区域的因果图。它通过最大化得分函数来推断区域的数量，并学习每个区域中的线性或非线性因果关系。 |
| [^4] | [Finite Element Operator Network for Solving Parametric PDEs.](http://arxiv.org/abs/2308.04690) | 本文提出了一种新方法，通过有限元算子网络（FEONet）解决参数PDE。它结合了深度学习和传统数值方法，展示了在没有输入-输出训练数据的情况下解决参数PDE的有效性，并在准确度、泛化性和计算灵活性方面优于现有方法。 |
| [^5] | [CARSO: Counter-Adversarial Recall of Synthetic Observations.](http://arxiv.org/abs/2306.06081) | 本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。 |
| [^6] | [On the Computational Cost of Stochastic Security.](http://arxiv.org/abs/2305.07973) | 本文探究了使用长期持续蒙特卡罗模拟是否能提高能量模型的质量，并通过增加计算预算改进了模型的校准性和对抗鲁棒性。 |

# 详细

[^1]: 正视监管空白：通过纳入社会公民打造超越AIA的欧盟AI审计生态系统

    Addressing the Regulatory Gap: Moving Towards an EU AI Audit Ecosystem Beyond the AIA by Including Civil Society

    [https://arxiv.org/abs/2403.07904](https://arxiv.org/abs/2403.07904)

    提出了一个融合合规和监督的AI审计生态系统，强调了DSA和AIA监管框架中存在的监管空白，并要求AIA为研究人员和社会公民提供数据和模型访问权限

    

    欧洲立法机构提出了数字服务法案（DSA）和人工智能法案（AIA）来规范平台和人工智能（AI）产品。本文审查了第三方审计在这两项法律中的地位以及在多大程度上提供模型和数据的访问权限。通过考虑审计生态系统中第三方审计和第三方数据访问的价值，我们发现了一个监管空白，即《人工智能法案》没有为研究人员和社会公民提供数据访问权限。我们对文献的贡献包括：（1）定义了一个融合合规和监督的AI审计生态系统。（2）强调了DSA和AIA监管框架中存在的监管空白，阻碍了AI审计生态系统的建立。（3）强调研究和社会公民的第三方审计必须成为该生态系统的一部分，并要求AIA包括数据和模型访问权限。

    arXiv:2403.07904v1 Announce Type: cross  Abstract: The European legislature has proposed the Digital Services Act (DSA) and Artificial Intelligence Act (AIA) to regulate platforms and Artificial Intelligence (AI) products. We review to what extent third-party audits are part of both laws and to what extent access to models and data is provided. By considering the value of third-party audits and third-party data access in an audit ecosystem, we identify a regulatory gap in that the Artificial Intelligence Act does not provide access to data for researchers and civil society. Our contributions to the literature include: (1) Defining an AI audit ecosystem that incorporates compliance and oversight. (2) Highlighting a regulatory gap within the DSA and AIA regulatory framework, preventing the establishment of an AI audit ecosystem. (3) Emphasizing that third-party audits by research and civil society must be part of that ecosystem and demand that the AIA include data and model access for ce
    
[^2]: Customize-It-3D：使用主体特定知识先验从单个图像创建高质量的3D模型

    Customize-It-3D: High-Quality 3D Creation from A Single Image Using Subject-Specific Knowledge Prior. (arXiv:2312.11535v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.11535](http://arxiv.org/abs/2312.11535)

    该论文提出了一种使用主体特定知识先验的两阶段方法，通过考虑阴影模式和纹理增强来生成高质量、有丰富纹理的3D模型，与以前的方法相比具有显著的优势。

    

    在本文中，我们提出了一种新的两阶段方法，充分利用参考图像提供的信息，建立图像到3D生成的自定义知识先验。之前的方法主要依赖于通用的扩散先验，这些方法在与参考图像得到一致结果方面存在困难，我们提出了一种主体特定且多模态的扩散模型。该模型不仅通过考虑阴影模式来改善几何优化和纹理增强的粗略结果，还有助于使3D内容与主题保持一致。大量实验证明了我们的方法的优越性，Customize-It-3D在视觉质量上远远超过了以前的方法，能够产生出色的360度重建结果，非常适合各种应用，包括文本到3D的创建。

    In this paper, we present a novel two-stage approach that fully utilizes the information provided by the reference image to establish a customized knowledge prior for image-to-3D generation. While previous approaches primarily rely on a general diffusion prior, which struggles to yield consistent results with the reference image, we propose a subject-specific and multi-modal diffusion model. This model not only aids NeRF optimization by considering the shading mode for improved geometry but also enhances texture from the coarse results to achieve superior refinement. Both aspects contribute to faithfully aligning the 3D content with the subject. Extensive experiments showcase the superiority of our method, Customize-It-3D, outperforming previous works by a substantial margin. It produces faithful 360-degree reconstructions with impressive visual quality, making it well-suited for various applications, including text-to-3D creation.
    
[^3]: Castor: 因果时序区域结构学习

    Castor: Causal Temporal Regime Structure Learning. (arXiv:2311.01412v1 [cs.LG])

    [http://arxiv.org/abs/2311.01412](http://arxiv.org/abs/2311.01412)

    “Castor”是一个用于学习多元时间序列数据中因果关系的框架，能够综合学习各个区域的因果图。它通过最大化得分函数来推断区域的数量，并学习每个区域中的线性或非线性因果关系。

    

    揭示多元时间序列数据之间的因果关系是一个重要且具有挑战性的目标，涉及到从气候科学到医疗保健等各个学科的广泛范围。这些数据包含线性或非线性关系，并且通常遵循多个先验未知的区域。现有的因果发现方法可以从具有已知区域的异构数据中推断出摘要因果图，但在全面学习区域和相应的因果图方面存在不足。在本文中，我们介绍了CASTOR，这是一个新颖的框架，旨在学习由不同因果图统治的各种异构时间序列数据中的因果关系。通过EM算法通过最大化一个得分函数，CASTOR推断出区域的数量并学习每个区域中的线性或非线性因果关系。我们展示了CASTOR的稳健收敛性质，特别突出了其有效性。

    The task of uncovering causal relationships among multivariate time series data stands as an essential and challenging objective that cuts across a broad array of disciplines ranging from climate science to healthcare. Such data entails linear or non-linear relationships, and usually follow multiple a priori unknown regimes. Existing causal discovery methods can infer summary causal graphs from heterogeneous data with known regimes, but they fall short in comprehensively learning both regimes and the corresponding causal graph. In this paper, we introduce CASTOR, a novel framework designed to learn causal relationships in heterogeneous time series data composed of various regimes, each governed by a distinct causal graph. Through the maximization of a score function via the EM algorithm, CASTOR infers the number of regimes and learns linear or non-linear causal relationships in each regime. We demonstrate the robust convergence properties of CASTOR, specifically highlighting its profic
    
[^4]: 用于解决参数PDE的有限元算子网络

    Finite Element Operator Network for Solving Parametric PDEs. (arXiv:2308.04690v1 [math.NA])

    [http://arxiv.org/abs/2308.04690](http://arxiv.org/abs/2308.04690)

    本文提出了一种新方法，通过有限元算子网络（FEONet）解决参数PDE。它结合了深度学习和传统数值方法，展示了在没有输入-输出训练数据的情况下解决参数PDE的有效性，并在准确度、泛化性和计算灵活性方面优于现有方法。

    

    偏微分方程（PDE）是我们理解和预测物理、工程和金融等众多领域自然现象的基础。然而，解决参数PDE是一项复杂的任务，需要高效的数值方法。在本文中，我们提出了一种通过有限元算子网络（FEONet）解决参数PDE的新方法。我们的方法结合了深度学习和传统数值方法，特别是有限元法，以在没有任何配对的输入-输出训练数据的情况下解决参数PDE。我们在几个基准问题上展示了我们方法的效果，并且表明它在准确度、泛化性和计算灵活性方面优于现有的最先进方法。我们的FEONet框架在模拟具有不同边界条件和复杂域的各种领域中显示出潜力。

    Partial differential equations (PDEs) underlie our understanding and prediction of natural phenomena across numerous fields, including physics, engineering, and finance. However, solving parametric PDEs is a complex task that necessitates efficient numerical methods. In this paper, we propose a novel approach for solving parametric PDEs using a Finite Element Operator Network (FEONet). Our proposed method leverages the power of deep learning in conjunction with traditional numerical methods, specifically the finite element method, to solve parametric PDEs in the absence of any paired input-output training data. We demonstrate the effectiveness of our approach on several benchmark problems and show that it outperforms existing state-of-the-art methods in terms of accuracy, generalization, and computational flexibility. Our FEONet framework shows potential for application in various fields where PDEs play a crucial role in modeling complex domains with diverse boundary conditions and sin
    
[^5]: CARSO: 对抗性合成观测的反对抗性召回

    CARSO: Counter-Adversarial Recall of Synthetic Observations. (arXiv:2306.06081v1 [cs.CV])

    [http://arxiv.org/abs/2306.06081](http://arxiv.org/abs/2306.06081)

    本文提出了一种新的图像分类的对抗性防御机制CARSO，该方法可以比最先进的对抗性训练更好地保护分类器，通过利用生成模型进行对抗净化来进行最终分类，并成功地保护自己免受未预见的威胁和最终攻击。

    

    本文提出了一种新的对抗性防御机制CARSO，用于图像分类，灵感来自认知神经科学的线索。该方法与对抗训练具有协同互补性，并依赖于被攻击分类器的内部表示的知识。通过利用生成模型进行对抗净化，该方法采样输入的重构来进行最终分类。在各种图像数据集和分类器体系结构上进行的实验评估表明，CARSO能够比最先进的对抗性训练更好地保护分类器——同时具有可接受的清洁准确度损失。此外，防御体系结构成功地保护自己免受未预见的威胁和最终攻击。代码和预训练模型可在https://github.com/获得。

    In this paper, we propose a novel adversarial defence mechanism for image classification -- CARSO -- inspired by cues from cognitive neuroscience. The method is synergistically complementary to adversarial training and relies on knowledge of the internal representation of the attacked classifier. Exploiting a generative model for adversarial purification, conditioned on such representation, it samples reconstructions of inputs to be finally classified. Experimental evaluation by a well-established benchmark of varied, strong adaptive attacks, across diverse image datasets and classifier architectures, shows that CARSO is able to defend the classifier significantly better than state-of-the-art adversarial training alone -- with a tolerable clean accuracy toll. Furthermore, the defensive architecture succeeds in effectively shielding itself from unforeseen threats, and end-to-end attacks adapted to fool stochastic defences. Code and pre-trained models are available at https://github.com/
    
[^6]: 论随机安全性的计算成本

    On the Computational Cost of Stochastic Security. (arXiv:2305.07973v1 [cs.LG])

    [http://arxiv.org/abs/2305.07973](http://arxiv.org/abs/2305.07973)

    本文探究了使用长期持续蒙特卡罗模拟是否能提高能量模型的质量，并通过增加计算预算改进了模型的校准性和对抗鲁棒性。

    

    我们探讨了使用朗之万动力学的长期持续蒙特卡罗模拟是否会提高基于能量的模型（EBM）所达到的表征质量。我们考虑一种方案，其中使用训练过的EBM的扩散过程的蒙特卡罗模拟，用于提高独立分类器网络的对抗鲁棒性和校准分数。我们的结果表明，在持续对比散度的计算预算增加吉布斯采样的情况下，改进了模型的校准性和对抗鲁棒性，澄清了实现有效从连续能量势中进行吉布斯采样的新量子和经典硬件和软件的实际价值。

    We investigate whether long-run persistent chain Monte Carlo simulation of Langevin dynamics improves the quality of the representations achieved by energy-based models (EBM). We consider a scheme wherein Monte Carlo simulation of a diffusion process using a trained EBM is used to improve the adversarial robustness and the calibration score of an independent classifier network. Our results show that increasing the computational budget of Gibbs sampling in persistent contrastive divergence improves the calibration and adversarial robustness of the model, elucidating the practical merit of realizing new quantum and classical hardware and software for efficient Gibbs sampling from continuous energy potentials.
    

