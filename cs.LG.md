# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models](https://arxiv.org/abs/2403.15740) | 通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。 |
| [^2] | [Governance of Generative Artificial Intelligence for Companies](https://arxiv.org/abs/2403.08802) | 本综述填补了有关企业中生成式人工智能（GenAI）治理的研究空白，提出了一个框架，旨在利用业务机会并减轻与GenAI整合相关风险。 |
| [^3] | [Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities](https://arxiv.org/abs/2403.02004) | 证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。 |
| [^4] | [Active Deep Kernel Learning of Molecular Functionalities: Realizing Dynamic Structural Embeddings](https://arxiv.org/abs/2403.01234) | 本文提出了一种利用深度核学习（DKL）的活跃学习方法，通过与传统变分自动编码器（VAEs）的对比分析，创造了优先考虑分子功能性的潜在空间，并且通过迭代重新计算嵌入向量实现了更好组织的潜在空间。 |
| [^5] | [On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension.](http://arxiv.org/abs/2401.15801) | 这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。 |
| [^6] | [Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory.](http://arxiv.org/abs/2310.20360) | 本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。 |
| [^7] | [Understanding Pan-Sharpening via Generalized Inverse.](http://arxiv.org/abs/2310.02718) | 通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。 |
| [^8] | [Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms.](http://arxiv.org/abs/2309.10301) | 该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。 |
| [^9] | [Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19.](http://arxiv.org/abs/2308.09730) | 本研究通过使用多样性的临床和虚拟生成的医学图像开发和评估了COVID-19诊断的AI模型，发现数据集特征对于AI性能具有重要影响，容易导致泛化能力较差，最高下降20％。 |

# 详细

[^1]: Ghost Sentence：一种供普通用户使用的工具，用于对大型语言模型中的数据进行版权保护

    Ghost Sentence: A Tool for Everyday Users to Copyright Data from Large Language Models

    [https://arxiv.org/abs/2403.15740](https://arxiv.org/abs/2403.15740)

    通过在文档中插入个人密码并识别生成内容中的“幽灵句子”，普通用户可以确认大型语言模型是否滥用其数据，从而实现数据版权保护。

    

    Web用户数据在预训练大型语言模型（LLMs）及其微调变种的生态系统中起着核心作用。本文提出了一种方法，建议用户在其文档中反复插入个人密码，使LLMs能够记忆这些密码。这些用户文档中隐藏的密码，被称为“幽灵句子”，一旦它们出现在LLMs生成的内容中，用户就可以确信他们的数据被用于训练。为了探索这种版权工具的有效性和用法，我们利用幽灵句子定义了“用户训练数据识别”任务。我们创建了来自不同来源、不同规模的多个数据集，并使用不同规模的LLMs进行测试。为了评估，我们引入了一个最后$k$个单词验证的方式。

    arXiv:2403.15740v1 Announce Type: new  Abstract: Web user data plays a central role in the ecosystem of pre-trained large language models (LLMs) and their fine-tuned variants. Billions of data are crawled from the web and fed to LLMs. How can \textit{\textbf{everyday web users}} confirm if LLMs misuse their data without permission? In this work, we suggest that users repeatedly insert personal passphrases into their documents, enabling LLMs to memorize them. These concealed passphrases in user documents, referred to as \textit{ghost sentences}, once they are identified in the generated content of LLMs, users can be sure that their data is used for training. To explore the effectiveness and usage of this copyrighting tool, we define the \textit{user training data identification} task with ghost sentences. Multiple datasets from various sources at different scales are created and tested with LLMs of different sizes. For evaluation, we introduce a last $k$ words verification manner along 
    
[^2]: 企业中生成式人工智能的治理

    Governance of Generative Artificial Intelligence for Companies

    [https://arxiv.org/abs/2403.08802](https://arxiv.org/abs/2403.08802)

    本综述填补了有关企业中生成式人工智能（GenAI）治理的研究空白，提出了一个框架，旨在利用业务机会并减轻与GenAI整合相关风险。

    

    生成式人工智能（GenAI），特别是像ChatGPT这样的大型语言模型，已迅速进入企业，但缺乏充分的治理，带来机遇和挑战。尽管对GenAI具有变革性质和监管措施的广泛讨论，但有限的研究涉及组织治理，包括技术和业务视角。本综述填补了这一空白，调查了最近的研究。它不仅仅是总结，还通过制定适用于企业内的GenAI治理框架来进行。我们的框架详细描述了范围、目标和治理机制，旨在利用业务机会并减轻与GenAI整合相关风险。该研究提供了一种专注于GenAI治理的方法，为企业在负责任的AI采用挑战中提供了实用见解。对于技术人员来说，也有助于拓宽他们的视角。

    arXiv:2403.08802v1 Announce Type: new  Abstract: Generative Artificial Intelligence (GenAI), specifically large language models like ChatGPT, has swiftly entered organizations without adequate governance, posing both opportunities and risks. Despite extensive debates on GenAI's transformative nature and regulatory measures, limited research addresses organizational governance, encompassing technical and business perspectives. This review paper fills this gap by surveying recent works. It goes beyond mere summarization by developing a framework for GenAI governance within companies. Our framework outlines the scope, objectives, and governance mechanisms tailored to harness business opportunities and mitigate risks associated with GenAI integration. This research contributes a focused approach to GenAI governance, offering practical insights for companies navigating the challenges of responsible AI adoption. It is also valuable for a technical audience to broaden their perspective as inc
    
[^3]: 粒子梯度下降的误差界限，以及log-Sobolev和Talagrand不等式的推广

    Error bounds for particle gradient descent, and extensions of the log-Sobolev and Talagrand inequalities

    [https://arxiv.org/abs/2403.02004](https://arxiv.org/abs/2403.02004)

    证明了粒子梯度下降算法对于一般化的log-Sobolev和Polyak-Lojasiewicz不等式模型的收敛速度，以及推广了Bakry-Emery定理。

    

    我们证明了粒子梯度下降(PGD)~(Kuntz等人，2023)的非渐近误差界限，这是一种最大似然估计的算法，用于离散化自由能梯度流获得的大型潜变量模型。我们首先展示了对于满足一般化log-Sobolev和Polyak-Lojasiewicz不等式（LSI和PLI）的模型，流以指数速度收敛到自由能的极小化集合。我们通过将最优输运文献中众所周知的结果（LSI意味着Talagrand不等式）及其在优化文献中的对应物（PLI意味着所谓的二次增长条件）扩展并应用到我们的新设置，来实现这一点。我们还推广了Bakry-Emery定理，并展示了对于具有强凹对数似然的模型，LSI/PLI的概括成立。

    arXiv:2403.02004v1 Announce Type: new  Abstract: We prove non-asymptotic error bounds for particle gradient descent (PGD)~(Kuntz et al., 2023), a recently introduced algorithm for maximum likelihood estimation of large latent variable models obtained by discretizing a gradient flow of the free energy. We begin by showing that, for models satisfying a condition generalizing both the log-Sobolev and the Polyak--{\L}ojasiewicz inequalities (LSI and P{\L}I, respectively), the flow converges exponentially fast to the set of minimizers of the free energy. We achieve this by extending a result well-known in the optimal transport literature (that the LSI implies the Talagrand inequality) and its counterpart in the optimization literature (that the P{\L}I implies the so-called quadratic growth condition), and applying it to our new setting. We also generalize the Bakry--\'Emery Theorem and show that the LSI/P{\L}I generalization holds for models with strongly concave log-likelihoods. For such m
    
[^4]: 活跃深度核学习分子功能性：实现动态结构嵌入

    Active Deep Kernel Learning of Molecular Functionalities: Realizing Dynamic Structural Embeddings

    [https://arxiv.org/abs/2403.01234](https://arxiv.org/abs/2403.01234)

    本文提出了一种利用深度核学习（DKL）的活跃学习方法，通过与传统变分自动编码器（VAEs）的对比分析，创造了优先考虑分子功能性的潜在空间，并且通过迭代重新计算嵌入向量实现了更好组织的潜在空间。

    

    探索分子空间对于推进我们对化学性质和反应的理解至关重要，从而在材料科学、医学和能源领域取得突破性创新。本文探讨了一种利用深度核学习（DKL）进行分子发现的主动学习方法，这是一种超越传统变分自动编码器（VAEs）限制的新方法。使用QM9数据集，我们将DKL与传统VAEs进行对比，后者基于相似性分析分子结构，揭示了由于潜在空间中的稀疏规律性而存在的局限性。然而，DKL通过将结构与性质相关联，创造了优先考虑分子功能性的潜在空间，提供了更全面的视角。这是通过迭代重新计算嵌入向量来实现的，与目标性质的实验可用性保持一致。由此产生的潜在空间不仅组织更好，而且具有独特特性。

    arXiv:2403.01234v1 Announce Type: new  Abstract: Exploring molecular spaces is crucial for advancing our understanding of chemical properties and reactions, leading to groundbreaking innovations in materials science, medicine, and energy. This paper explores an approach for active learning in molecular discovery using Deep Kernel Learning (DKL), a novel approach surpassing the limits of classical Variational Autoencoders (VAEs). Employing the QM9 dataset, we contrast DKL with traditional VAEs, which analyze molecular structures based on similarity, revealing limitations due to sparse regularities in latent spaces. DKL, however, offers a more holistic perspective by correlating structure with properties, creating latent spaces that prioritize molecular functionality. This is achieved by recalculating embedding vectors iteratively, aligning with the experimental availability of target properties. The resulting latent spaces are not only better organized but also exhibit unique characteri
    
[^5]: 关于用于低固有数据维度的生成对抗模型的统计属性

    On the Statistical Properties of Generative Adversarial Models for Low Intrinsic Data Dimension. (arXiv:2401.15801v1 [stat.ML])

    [http://arxiv.org/abs/2401.15801](http://arxiv.org/abs/2401.15801)

    这篇论文研究了用于低固有数据维度的生成对抗模型的统计属性，提出了关于估计密度的统计保证，涉及数据和潜空间的内在维度，并证明了估计结果与目标的期望Wasserstein-1距离的缩放关系。

    

    尽管生成对抗网络（GANs）取得了显著的实证成功，但其统计准确性的理论保证仍然相对悲观。特别是在应用GANs的数据分布（如自然图像）中，通常假设其在高维特征空间中具有固有的低维结构，但这在现有分析中往往没有得到反映。在本文中，我们试图通过推导关于数据和潜空间的内在维度的统计保证来弥合GANs及其双向变体BiGANs在理论和实践之间的差距。我们分析地证明，如果我们有来自未知目标分布的 n 个样本，并且选择了适当的网络架构，那么从目标中估计得出的期望 Wasserstein-1 距离会按照 $O(n^{-1/d_\mu })$ 缩放。

    Despite the remarkable empirical successes of Generative Adversarial Networks (GANs), the theoretical guarantees for their statistical accuracy remain rather pessimistic. In particular, the data distributions on which GANs are applied, such as natural images, are often hypothesized to have an intrinsic low-dimensional structure in a typically high-dimensional feature space, but this is often not reflected in the derived rates in the state-of-the-art analyses. In this paper, we attempt to bridge the gap between the theory and practice of GANs and their bidirectional variant, Bi-directional GANs (BiGANs), by deriving statistical guarantees on the estimated densities in terms of the intrinsic dimension of the data and the latent space. We analytically show that if one has access to $n$ samples from the unknown target distribution and the network architectures are properly chosen, the expected Wasserstein-1 distance of the estimates from the target scales as $O\left( n^{-1/d_\mu } \right)$
    
[^6]: 深度学习的数学介绍：方法、实现和理论

    Mathematical Introduction to Deep Learning: Methods, Implementations, and Theory. (arXiv:2310.20360v1 [cs.LG])

    [http://arxiv.org/abs/2310.20360](http://arxiv.org/abs/2310.20360)

    本书提供了对深度学习算法的数学介绍，包括不同的神经网络架构和优化算法，并涵盖了深度学习算法的理论方面。此外，还介绍了深度学习逼近偏微分方程的方法。希望对学生和科学家们有所帮助。

    

    本书旨在介绍深度学习算法的主题。我们详细介绍了深度学习算法的基本组成部分，包括不同的人工神经网络架构（如全连接前馈神经网络、卷积神经网络、循环神经网络、残差神经网络和带有批归一化的神经网络）以及不同的优化算法（如基本的随机梯度下降法、加速方法和自适应方法）。我们还涵盖了深度学习算法的几个理论方面，如人工神经网络的逼近能力（包括神经网络的微积分）、优化理论（包括Kurdyka-Lojasiewicz不等式）和泛化误差。在本书的最后一部分，我们还回顾了一些用于偏微分方程的深度学习逼近方法，包括物理信息神经网络（PINNs）和深度Galerkin方法。希望本书能对学生和科学家们有所帮助。

    This book aims to provide an introduction to the topic of deep learning algorithms. We review essential components of deep learning algorithms in full mathematical detail including different artificial neural network (ANN) architectures (such as fully-connected feedforward ANNs, convolutional ANNs, recurrent ANNs, residual ANNs, and ANNs with batch normalization) and different optimization algorithms (such as the basic stochastic gradient descent (SGD) method, accelerated methods, and adaptive methods). We also cover several theoretical aspects of deep learning algorithms such as approximation capacities of ANNs (including a calculus for ANNs), optimization theory (including Kurdyka-{\L}ojasiewicz inequalities), and generalization errors. In the last part of the book some deep learning approximation methods for PDEs are reviewed including physics-informed neural networks (PINNs) and deep Galerkin methods. We hope that this book will be useful for students and scientists who do not yet 
    
[^7]: 通过广义逆理解全色增强算法

    Understanding Pan-Sharpening via Generalized Inverse. (arXiv:2310.02718v1 [cs.LG])

    [http://arxiv.org/abs/2310.02718](http://arxiv.org/abs/2310.02718)

    通过研究广义逆理论，本文提出了一种新的全色增强算法，该算法基于简单矩阵方程描述全色增强问题，并探讨解的条件和光谱、空间分辨率的获取。通过引入降采样增强方法，我们得到了与分量替代和多分辨率分析方法相对应的广义逆矩阵表达式，并提出了一个新的模型先验来解决全色增强中的理论误差问题。

    

    全色增强算法利用全色图像和多光谱图像获取具有高空间和高光谱的图像。然而，这些算法的优化是根据不同的标准设计的。我们采用简单的矩阵方程来描述全色增强问题，并讨论解的存在条件以及光谱和空间分辨率的获取。我们引入了一种降采样增强方法，以更好地获取空间和光谱降采样矩阵。通过广义逆理论，我们推导出了两种形式的广义逆矩阵表达式，可以对应于两个主要的全色增强方法：分量替代和多分辨率分析方法。具体而言，我们证明了Gram Schmidt自适应(GSA)方法遵循分量替代的广义逆矩阵表达式。我们提出了一个在光谱函数的广义逆矩阵之前的模型先验。我们对理论误差进行了分析。

    Pan-sharpening algorithm utilizes panchromatic image and multispectral image to obtain a high spatial and high spectral image. However, the optimizations of the algorithms are designed with different standards. We adopt the simple matrix equation to describe the Pan-sharpening problem. The solution existence condition and the acquirement of spectral and spatial resolution are discussed. A down-sampling enhancement method was introduced for better acquiring the spatial and spectral down-sample matrices. By the generalized inverse theory, we derived two forms of general inverse matrix formulations that can correspond to the two prominent classes of Pan-sharpening methods, that is, component substitution and multi-resolution analysis methods. Specifically, the Gram Schmidt Adaptive(GSA) was proved to follow the general inverse matrix formulation of component substitution. A model prior to the general inverse matrix of the spectral function was rendered. The theoretical errors are analyzed
    
[^8]: 领域自适应中条件不变组件的突出作用：理论和算法

    Prominent Roles of Conditionally Invariant Components in Domain Adaptation: Theory and Algorithms. (arXiv:2309.10301v1 [stat.ML])

    [http://arxiv.org/abs/2309.10301](http://arxiv.org/abs/2309.10301)

    该论文研究了领域自适应中条件不变组件的作用，提出了一种基于条件不变惩罚的新算法，该算法在目标风险保证方面具有优势。

    

    领域自适应是一个统计学习问题，当用于训练模型的源数据分布与用于评估模型的目标数据分布不同时出现。虽然许多领域自适应算法已经证明了相当大的实证成功，但是盲目应用这些算法往往会导致在新的数据集上表现更差。为了解决这个问题，重要的是澄清领域自适应算法在具备良好目标性能的假设下。在这项工作中，我们关注在预测中具备条件不变的组件（CICs）的存在假设，这些组件在源数据和目标数据之间保持条件不变。我们证明了CICs，通过条件不变惩罚（CIP）可以估计，具备在领域自适应中提供目标风险保证的三个突出作用。首先，我们提出了一种基于CICs的新算法，即重要性加权的条件不变惩罚（IW-CIP），它在目标风险保证方面超越了简单的方法。

    Domain adaptation (DA) is a statistical learning problem that arises when the distribution of the source data used to train a model differs from that of the target data used to evaluate the model. While many DA algorithms have demonstrated considerable empirical success, blindly applying these algorithms can often lead to worse performance on new datasets. To address this, it is crucial to clarify the assumptions under which a DA algorithm has good target performance. In this work, we focus on the assumption of the presence of conditionally invariant components (CICs), which are relevant for prediction and remain conditionally invariant across the source and target data. We demonstrate that CICs, which can be estimated through conditional invariant penalty (CIP), play three prominent roles in providing target risk guarantees in DA. First, we propose a new algorithm based on CICs, importance-weighted conditional invariant penalty (IW-CIP), which has target risk guarantees beyond simple 
    
[^9]: 基于COVID-19的数据多样性和虚拟成像的AI诊断：以病例研究为基础

    Data diversity and virtual imaging in AI-based diagnosis: A case study based on COVID-19. (arXiv:2308.09730v1 [eess.IV])

    [http://arxiv.org/abs/2308.09730](http://arxiv.org/abs/2308.09730)

    本研究通过使用多样性的临床和虚拟生成的医学图像开发和评估了COVID-19诊断的AI模型，发现数据集特征对于AI性能具有重要影响，容易导致泛化能力较差，最高下降20％。

    

    许多研究已经调查了基于深度学习的人工智能（AI）模型在新型冠状病毒（COVID-19）的医学影像诊断中的应用，许多报道称其性能几乎完美。然而，性能的变异性和潜在的数据偏差引发了对临床适用性的担忧。本回顾性研究涉及使用临床多样性和虚拟生成的医学图像开发和评估COVID-19诊断的人工智能（AI）模型。此外，我们进行了一次虚拟成像试验，以评估AI性能受疾病范围、辐射剂量和计算机断层扫描（CT）和胸部放射摄影（CXR）成像模态等几个患者和物理性因素的影响。数据集特征（包括数量、多样性和患病率）强烈影响了AI的性能，导致接收者操作特征曲线下面积下降了高达20％，且泛化能力差。

    Many studies have investigated deep-learning-based artificial intelligence (AI) models for medical imaging diagnosis of the novel coronavirus (COVID-19), with many reports of near-perfect performance. However, variability in performance and underlying data biases raise concerns about clinical generalizability. This retrospective study involved the development and evaluation of artificial intelligence (AI) models for COVID-19 diagnosis using both diverse clinical and virtually generated medical images. In addition, we conducted a virtual imaging trial to assess how AI performance is affected by several patient- and physics-based factors, including the extent of disease, radiation dose, and imaging modality of computed tomography (CT) and chest radiography (CXR). AI performance was strongly influenced by dataset characteristics including quantity, diversity, and prevalence, leading to poor generalization with up to 20% drop in receiver operating characteristic area under the curve. Model
    

