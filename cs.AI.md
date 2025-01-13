# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Arcee's MergeKit: A Toolkit for Merging Large Language Models](https://arxiv.org/abs/2403.13257) | 合并不同语言模型的参数，无需额外训练即可创建多任务模型，提升模型性能和多功能性，解决AI中的复杂挑战。 |
| [^2] | [On the Expressive Power of a Variant of the Looped Transformer](https://arxiv.org/abs/2402.13572) | 设计了一种新型Transformer块AlgoFormer，相比标准Transformer和Looped Transformer，AlgoFormer在相同参数数量下能够实现更高的算法表达能力 |
| [^3] | [LitSumm: Large language models for literature summarisation of non-coding RNAs](https://arxiv.org/abs/2311.03056) | 使用大语言模型为非编码RNA文献生成高质量和准确的摘要，帮助减轻生命科学文献整理中缺乏策展人员时间的问题。 |
| [^4] | [Surrogate-based Autotuning for Randomized Sketching Algorithms in Regression Problems.](http://arxiv.org/abs/2308.15720) | 本文介绍了如何使用基于替代模型的自动调优方法解决随机化草图算法中的参数选择问题，在随机数值线性代数中取得了接近最优性能的实证结果。 |
| [^5] | [Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers.](http://arxiv.org/abs/2308.13969) | 本研究展示了如何将人眼追踪集成到视觉Transformer模型中，提高在多种驾驶情况和数据集上的准确性。 |
| [^6] | [A Pre-trained Data Deduplication Model based on Active Learning.](http://arxiv.org/abs/2308.00721) | 提出了一种基于主动学习的预训练去重模型，将Transformer和主动学习集成到端到端架构中，首次解决了语义级别的去重问题，同时采用R-Drop方法对每一轮标记数据进行数据增强。通过选择最有价值的数据进行去重模型训练，不仅降低了手动标记的成本，还提高了模型的泛化能力。 |
| [^7] | [GUTS: Generalized Uncertainty-Aware Thompson Sampling for Multi-Agent Active Search.](http://arxiv.org/abs/2304.02075) | 本文提出了 GUTS 算法，能够在异构多机器人系统中部署，解决大型非结构化环境下的主动搜索问题，考虑了传感器不确定性、遮挡问题、异构搜索团队和硬件、通信故障的鲁棒性等问题，并在仿真中超过现有算法高达80%。 |
| [^8] | [Discriminative Class Tokens for Text-to-Image Diffusion Models.](http://arxiv.org/abs/2303.17155) | 本文提出了一种基于判别性信息和自由文本的非侵入式微调技术，以实现多样性和高准确率的文本到图像生成模型。 |
| [^9] | [Optimal Spatial Deconvolution and Message Reconstruction from a Large Generative Model of Models.](http://arxiv.org/abs/2303.16045) | 本论文提出了一种基于人工通用智能方法原则的通用单变量信号去卷积方法。通过计算“通用分布”的估计来独立于概率分布地构建一个通用模型，并基于信息论和算法概率的多维空间重构，探索非随机数据中关于物理性质的信息编码。该方法在编码理论尤其是零失真压缩方面有应用价值。 |
| [^10] | [Adversarial Detection by Approximation of Ensemble Boundary.](http://arxiv.org/abs/2211.10227) | 本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。 |
| [^11] | [Self-supervised video pretraining yields human-aligned visual representations.](http://arxiv.org/abs/2210.06433) | 本论文通过自监督训练的视频预训练方法VITO得到了具有人类感知特征的视觉表示，该方法在图像理解和视频理解任务上表现出更好的泛化性和鲁棒性。 |

# 详细

[^1]: Arcee的MergeKit：用于合并大型语言模型的工具包

    Arcee's MergeKit: A Toolkit for Merging Large Language Models

    [https://arxiv.org/abs/2403.13257](https://arxiv.org/abs/2403.13257)

    合并不同语言模型的参数，无需额外训练即可创建多任务模型，提升模型性能和多功能性，解决AI中的复杂挑战。

    

    开源语言模型领域的快速扩张为通过合并其参数来结合这些模型检查点的能力提供了机会。迁移学习的进步导致了大量针对特定任务进行微调的模型的开发，这些模型通常专门针对个别任务进行专门化，无法利用彼此的优势。模型合并促进了多任务模型的创建，无需额外的训练，为增强模型性能和多功能性提供了一个有前途的途径。通过保留原始模型的固有能力，模型合并解决了人工智能中的复杂挑战，包括灾难性遗忘和多任务学习的困难。为了支持这一不断扩大的研究领域，我们介绍了MergeKit，这是一个全面的、开源的库，旨在促进模型合并的应用。

    arXiv:2403.13257v1 Announce Type: new  Abstract: The rapid expansion of the open-source language model landscape presents an opportunity to merge the competencies of these model checkpoints by combining their parameters. Advances in transfer learning, the process of fine-tuning pre-trained models for specific tasks, has resulted in the development of vast amounts of task-specific models, typically specialized in individual tasks and unable to utilize each other's strengths. Model merging facilitates the creation of multitask models without the need for additional training, offering a promising avenue for enhancing model performance and versatility. By preserving the intrinsic capabilities of the original models, model merging addresses complex challenges in AI - including the difficulties of catastrophic forgetting and multi-task learning. To support this expanding area of research, we introduce MergeKit, a comprehensive, open-source library designed to facilitate the application of mo
    
[^2]: 论一种变种Looped Transformer的表达能力

    On the Expressive Power of a Variant of the Looped Transformer

    [https://arxiv.org/abs/2402.13572](https://arxiv.org/abs/2402.13572)

    设计了一种新型Transformer块AlgoFormer，相比标准Transformer和Looped Transformer，AlgoFormer在相同参数数量下能够实现更高的算法表达能力

    

    除了自然语言处理，在解决更广泛的应用程序（包括科学计算和计算机视觉）方面，Transformer展现出卓越的性能。先前的工作试图从表达能力和功能性角度解释，标准的Transformer能够执行一些算法。为了赋予Transformer算法能力，并受到最近提出的Looped Transformer的启发，我们设计了一种新颖的Transformer块，名为Algorithm Transformer（简称AlgoFormer）。与标准Transformer和纯粹的Looped Transformer相比，所提出的AlgoFormer在使用相同数量的参数时可以实现更高的算法表示表达能力。特别是，受人类设计的学习算法结构的启发，我们的Transformer块包括一个负责进行ta

    arXiv:2402.13572v1 Announce Type: cross  Abstract: Besides natural language processing, transformers exhibit extraordinary performance in solving broader applications, including scientific computing and computer vision. Previous works try to explain this from the expressive power and capability perspectives that standard transformers are capable of performing some algorithms. To empower transformers with algorithmic capabilities and motivated by the recently proposed looped transformer (Yang et al., 2024; Giannou et al., 2023), we design a novel transformer block, dubbed Algorithm Transformer (abbreviated as AlgoFormer). Compared with the standard transformer and vanilla looped transformer, the proposed AlgoFormer can achieve significantly higher expressiveness in algorithm representation when using the same number of parameters. In particular, inspired by the structure of human-designed learning algorithms, our transformer block consists of a pre-transformer that is responsible for ta
    
[^3]: LitSumm：大语言模型用于非编码RNA文献摘要

    LitSumm: Large language models for literature summarisation of non-coding RNAs

    [https://arxiv.org/abs/2311.03056](https://arxiv.org/abs/2311.03056)

    使用大语言模型为非编码RNA文献生成高质量和准确的摘要，帮助减轻生命科学文献整理中缺乏策展人员时间的问题。

    

    Motivation: 在生命科学文献的整理工作中，面临着日益严峻的挑战。随着发布速度的持续增加，再加上全球固定数量的策展人员，开发生物医学知识库面临重大挑战。很少有知识库有资源可以扩展到所有相关文献，而所有知识库都必须优先考虑自己的努力。 在这项工作中，我们通过使用大语言模型（LLMs）为非编码RNA生成文献摘要，首次减轻了RNA科学中缺乏策展人员时间的问题。我们展示了可以使用商业LLM和一系列提示和检查从文献中自动生成高质量、事实准确的摘要及准确的引用。人工评估针对摘要子集进行，其中大多数被评为非常高质量。我们还应用了最常用的自动化e

    arXiv:2311.03056v2 Announce Type: replace-cross  Abstract: Motivation: Curation of literature in life sciences is a growing challenge. The continued increase in the rate of publication, coupled with the relatively fixed number of curators worldwide presents a major challenge to developers of biomedical knowledgebases. Very few knowledgebases have resources to scale to the whole relevant literature and all have to prioritise their efforts.   Results: In this work, we take a first step to alleviating the lack of curator time in RNA science by generating summaries of literature for non-coding RNAs using large language models (LLMs). We demonstrate that high-quality, factually accurate summaries with accurate references can be automatically generated from the literature using a commercial LLM and a chain of prompts and checks. Manual assessment was carried out for a subset of summaries, with the majority being rated extremely high quality. We also applied the most commonly used automated e
    
[^4]: 基于替代模型的自动调优方法在回归问题中随机化草图算法的应用

    Surrogate-based Autotuning for Randomized Sketching Algorithms in Regression Problems. (arXiv:2308.15720v1 [cs.LG])

    [http://arxiv.org/abs/2308.15720](http://arxiv.org/abs/2308.15720)

    本文介绍了如何使用基于替代模型的自动调优方法解决随机化草图算法中的参数选择问题，在随机数值线性代数中取得了接近最优性能的实证结果。

    

    从随机数值线性代数(RandNLA)中提出的算法在处理高维计算问题方面表现出很好的效果，提供高质量的经验性能以及强大的概率保证。然而，它们的实际应用受到一个问题的复杂性所限制，即用户需要设置各种不同于传统NLA中使用的算法特定调参参数。本文展示了如何使用基于替代模型的自动调优方法来解决RandNLA算法中参数选择的基础性问题。具体而言，我们对基于草图和预处理(SAP)的随机化最小二乘方法进行了替代模型自动调优的详细研究，这在现代RandNLA中是一个巨大的成功案例。实证结果表明，我们的基于替代模型的自动调优方法可以以比随机搜索少约4倍的试验成本实现接近最优的性能。

    Algorithms from Randomized Numerical Linear Algebra (RandNLA) are known to be effective in handling high-dimensional computational problems, providing high-quality empirical performance as well as strong probabilistic guarantees. However, their practical application is complicated by the fact that the user needs to set various algorithm-specific tuning parameters which are different than those used in traditional NLA. This paper demonstrates how a surrogate-based autotuning approach can be used to address fundamental problems of parameter selection in RandNLA algorithms. In particular, we provide a detailed investigation of surrogate-based autotuning for sketch-and-precondition (SAP) based randomized least squares methods, which have been one of the great success stories in modern RandNLA. Empirical results show that our surrogate-based autotuning approach can achieve near-optimal performance with much less tuning cost than a random search (up to about 4x fewer trials of different para
    
[^5]: 注重注意力：将人眼追踪集成到视觉Transformer模型中

    Fixating on Attention: Integrating Human Eye Tracking into Vision Transformers. (arXiv:2308.13969v1 [cs.CV])

    [http://arxiv.org/abs/2308.13969](http://arxiv.org/abs/2308.13969)

    本研究展示了如何将人眼追踪集成到视觉Transformer模型中，提高在多种驾驶情况和数据集上的准确性。

    

    现代基于Transformer的计算机视觉模型在多种视觉任务上表现出超越人类的能力。然而，一些关键任务，如医学图像解释和自动驾驶，仍然需要依赖人类判断。本研究展示了如何将人类视觉输入，特别是通过眼动仪收集到的注视点，集成到Transformer模型中，以提高在多种驾驶情况和数据集上的准确性。首先，我们在人类实验对象和Vision Transformer模型中观察到，注视区域在左右驾驶决策中的重要性。通过比较人类注视图和ViT注意力权重之间的相似性，我们揭示了单个头部和层之间的重叠动态。通过利用这种重叠动态，我们实现了对模型的修剪而不损失准确性。然后，我们将驾驶场景信息与注视数据相结合，采用“联合空间-注视”（JSF）的注意力设置。最后

    Modern transformer-based models designed for computer vision have outperformed humans across a spectrum of visual tasks. However, critical tasks, such as medical image interpretation or autonomous driving, still require reliance on human judgments. This work demonstrates how human visual input, specifically fixations collected from an eye-tracking device, can be integrated into transformer models to improve accuracy across multiple driving situations and datasets. First, we establish the significance of fixation regions in left-right driving decisions, as observed in both human subjects and a Vision Transformer (ViT). By comparing the similarity between human fixation maps and ViT attention weights, we reveal the dynamics of overlap across individual heads and layers. This overlap is exploited for model pruning without compromising accuracy. Thereafter, we incorporate information from the driving scene with fixation data, employing a "joint space-fixation" (JSF) attention setup. Lastly
    
[^6]: 基于主动学习的预训练数据去重模型

    A Pre-trained Data Deduplication Model based on Active Learning. (arXiv:2308.00721v1 [cs.LG])

    [http://arxiv.org/abs/2308.00721](http://arxiv.org/abs/2308.00721)

    提出了一种基于主动学习的预训练去重模型，将Transformer和主动学习集成到端到端架构中，首次解决了语义级别的去重问题，同时采用R-Drop方法对每一轮标记数据进行数据增强。通过选择最有价值的数据进行去重模型训练，不仅降低了手动标记的成本，还提高了模型的泛化能力。

    

    在大数据时代，数据质量问题日益突出。其中一个主要挑战是重复数据问题，这可能是由于数据的重复输入或多个数据源的合并导致的。这些"脏数据"问题严重限制了大数据的有效应用。为了解决数据去重的问题，我们提出了一种基于主动学习的预训练去重模型，这是首次利用主动学习解决语义级别的去重问题的工作。该模型构建在一个预训练的Transformer上，并通过细调将其应用于序列分类任务，首次将Transformer和主动学习集成到端到端架构中，以选择最有价值的数据进行去重模型训练，同时首次采用R-Drop方法对每一轮标记数据进行数据增强，既能降低手动标记的成本，也能提高模型的泛化能力。

    In the era of big data, the issue of data quality has become increasingly prominent. One of the main challenges is the problem of duplicate data, which can arise from repeated entry or the merging of multiple data sources. These "dirty data" problems can significantly limit the effective application of big data. To address the issue of data deduplication, we propose a pre-trained deduplication model based on active learning, which is the first work that utilizes active learning to address the problem of deduplication at the semantic level. The model is built on a pre-trained Transformer and fine-tuned to solve the deduplication problem as a sequence to classification task, which firstly integrate the transformer with active learning into an end-to-end architecture to select the most valuable data for deduplication model training, and also firstly employ the R-Drop method to perform data augmentation on each round of labeled data, which can reduce the cost of manual labeling and improve
    
[^7]: GUTS：多智能体主动搜索的广义不确定性感知 Thompson Sampling算法

    GUTS: Generalized Uncertainty-Aware Thompson Sampling for Multi-Agent Active Search. (arXiv:2304.02075v1 [cs.RO])

    [http://arxiv.org/abs/2304.02075](http://arxiv.org/abs/2304.02075)

    本文提出了 GUTS 算法，能够在异构多机器人系统中部署，解决大型非结构化环境下的主动搜索问题，考虑了传感器不确定性、遮挡问题、异构搜索团队和硬件、通信故障的鲁棒性等问题，并在仿真中超过现有算法高达80%。

    

    快速灾难响应的机器人解决方案对确保最小生命损失至关重要，特别是当搜索区域对于人类救援者而言过于危险或过于广阔时。本文将这个问题建模为一个异步多智能体主动搜索任务，在此任务中，每个机器人旨在高效地在未知环境中寻找感兴趣的对象（OOI）。这种表述解决了搜索任务应该专注于快速恢复OOI而不是对搜索区域进行全覆盖的要求。先前的方法未能准确建模传感器不确定性，考虑到由于植被或地形的遮挡而导致的遮挡问题，或者考虑到异构搜索团队和硬件、通信故障的鲁棒性要求。我们提出了广义不确定性感知Thompson抽样（GUTS）算法，它解决了这些问题，并适用于在大型非结构化环境中部署异构多机器人系统进行主动搜索。我们通过仿真实验表明，GUTS算法在性能上超过现有算法高达80%。

    Robotic solutions for quick disaster response are essential to ensure minimal loss of life, especially when the search area is too dangerous or too vast for human rescuers. We model this problem as an asynchronous multi-agent active-search task where each robot aims to efficiently seek objects of interest (OOIs) in an unknown environment. This formulation addresses the requirement that search missions should focus on quick recovery of OOIs rather than full coverage of the search region. Previous approaches fail to accurately model sensing uncertainty, account for occlusions due to foliage or terrain, or consider the requirement for heterogeneous search teams and robustness to hardware and communication failures. We present the Generalized Uncertainty-aware Thompson Sampling (GUTS) algorithm, which addresses these issues and is suitable for deployment on heterogeneous multi-robot systems for active search in large unstructured environments. We show through simulation experiments that GU
    
[^8]: 基于判别性类标的文本图片扩散模型

    Discriminative Class Tokens for Text-to-Image Diffusion Models. (arXiv:2303.17155v1 [cs.CV])

    [http://arxiv.org/abs/2303.17155](http://arxiv.org/abs/2303.17155)

    本文提出了一种基于判别性信息和自由文本的非侵入式微调技术，以实现多样性和高准确率的文本到图像生成模型。

    

    最近文本到图像扩散模型的进展使得生成多样且高质量图片成为可能。然而，由于输入文本的歧义，生成的图片常常无法描绘出微妙的细节且易于出错。缓解这些问题的方法之一是在有类标注的数据集上训练扩散模型。这种方法的缺点在于：（i）与用于训练文本到图像模型的大规模爬取的文本-图像数据集相比，有类标注的数据集通常较小，因此生成的图片质量和多样性会严重受影响，或（ii）输入是硬编码的标签，而不是自由文本，这限制了对生成的图像的控制。在这项工作中，我们提出了一种非侵入式的微调技术，利用预训练分类器的判别性信号引导生成过程，既发挥了自由文本的表达潜力，又能够实现高准确率。

    Recent advances in text-to-image diffusion models have enabled the generation of diverse and high-quality images. However, generated images often fall short of depicting subtle details and are susceptible to errors due to ambiguity in the input text. One way of alleviating these issues is to train diffusion models on class-labeled datasets. This comes with a downside, doing so limits their expressive power: (i) supervised datasets are generally small compared to large-scale scraped text-image datasets on which text-to-image models are trained, and so the quality and diversity of generated images are severely affected, or (ii) the input is a hard-coded label, as opposed to free-form text, which limits the control over the generated images.  In this work, we propose a non-invasive fine-tuning technique that capitalizes on the expressive potential of free-form text while achieving high accuracy through discriminative signals from a pretrained classifier, which guides the generation. This 
    
[^9]: 大规模生成模型中的最优空间去卷积和信息重建

    Optimal Spatial Deconvolution and Message Reconstruction from a Large Generative Model of Models. (arXiv:2303.16045v1 [cs.IT])

    [http://arxiv.org/abs/2303.16045](http://arxiv.org/abs/2303.16045)

    本论文提出了一种基于人工通用智能方法原则的通用单变量信号去卷积方法。通过计算“通用分布”的估计来独立于概率分布地构建一个通用模型，并基于信息论和算法概率的多维空间重构，探索非随机数据中关于物理性质的信息编码。该方法在编码理论尤其是零失真压缩方面有应用价值。

    

    本论文介绍一种基于人工通用智能方法原则的通用单变量信号去卷积方法，该方法建立了一个模型生成模型，依赖于信息论和算法概率，并计算出“通用分布”的估计，从而独立于概率分布地构建了一个通用模型。该方法基于信息论和算法概率的多维空间重构，可以探究非随机数据如何编码关于物理性质的信息，例如信号或信息的维度和长度尺度。该方法是与可计算或半可计算的近似方法或编码-解码方案相关但不独立的。本文的结果对编码理论尤其是零失真压缩有应用意义。

    We introduce a general-purpose univariate signal deconvolution method based on the principles of an approach to Artificial General Intelligence. This approach is based on a generative model that combines information theory and algorithmic probability that required a large calculation of an estimation of a `universal distribution' to build a general-purpose model of models independent of probability distributions. This was used to investigate how non-random data may encode information about the physical properties such as dimension and length scales in which a signal or message may have been originally encoded, embedded, or generated. This multidimensional space reconstruction method is based on information theory and algorithmic probability, and it is agnostic, but not independent, with respect to the chosen computable or semi-computable approximation method or encoding-decoding scheme. The results presented in this paper are useful for applications in coding theory, particularly in ze
    
[^10]: 使用集成边界逼近的对抗检测方法

    Adversarial Detection by Approximation of Ensemble Boundary. (arXiv:2211.10227v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.10227](http://arxiv.org/abs/2211.10227)

    本论文提出了一种使用Walsh系数逼近决策边界的对抗攻击检测方法，通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实现了对对抗攻击的检测。

    

    本论文提出了一种新的对抗攻击检测方法，针对解决两类模式识别问题的深度神经网络（DNN）集成。该集成使用Walsh系数进行组合，能够逼近布尔函数并控制集成决策边界的复杂性。本文的假设是高曲率的决策边界允许找到对抗扰动，但会改变决策边界的曲率，而与清晰图像相比，使用Walsh系数对其进行逼近的方式也有所不同。通过观察清晰图像和对抗图像之间的Walsh系数逼近差异，实验证明了攻击的可迁移性可用于检测。此外，逼近决策边界可能有助于理解DNN的学习和可迁移性特性。尽管本文的实验使用图像，所提出的方法可以用于建模两类模式识别问题的集成边界逼近。

    A new method of detecting adversarial attacks is proposed for an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The ensemble is combined using Walsh coefficients which are capable of approximating Boolean functions and thereby controlling the complexity of the ensemble decision boundary. The hypothesis in this paper is that decision boundaries with high curvature allow adversarial perturbations to be found, but change the curvature of the decision boundary, which is then approximated in a different way by Walsh coefficients compared to the clean images. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it is shown experimentally that transferability of attack may be used for detection. Furthermore, approximating the decision boundary may aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class en
    
[^11]: 自监督训练的视频预训练产生与人类对齐的视觉表示

    Self-supervised video pretraining yields human-aligned visual representations. (arXiv:2210.06433v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2210.06433](http://arxiv.org/abs/2210.06433)

    本论文通过自监督训练的视频预训练方法VITO得到了具有人类感知特征的视觉表示，该方法在图像理解和视频理解任务上表现出更好的泛化性和鲁棒性。

    

    人类通过观察对象和场景随时间演变的方式学习到了强大的表示。然而，在不需要明确的时间理解的特定任务之外，静态图像预训练仍然是学习视觉基础模型的主流范式。我们对这种不匹配提出了质疑，并且问是否视频预训练可以产生具有人类感知特征的视觉表示：在各种任务中的泛化性、对扰动的鲁棒性和与人类判断的一致性。为此，我们提出了一种用于筛选视频的新颖程序，并开发了一个对比性框架，从其中的复杂转换中学习。这种从视频中提炼知识的简单范式被称为VITO，它产生的一般表示在图像理解任务上远远优于先前的视频预训练方法，并且在视频理解任务上优于图像预训练方法。此外，VITO表示对自然和合成形变的鲁棒性显著提高。

    Humans learn powerful representations of objects and scenes by observing how they evolve over time. Yet, outside of specific tasks that require explicit temporal understanding, static image pretraining remains the dominant paradigm for learning visual foundation models. We question this mismatch, and ask whether video pretraining can yield visual representations that bear the hallmarks of human perception: generalisation across tasks, robustness to perturbations, and consistency with human judgements. To that end we propose a novel procedure for curating videos, and develop a contrastive framework which learns from the complex transformations therein. This simple paradigm for distilling knowledge from videos, called VITO, yields general representations that far outperform prior video pretraining methods on image understanding tasks, and image pretraining methods on video understanding tasks. Moreover, VITO representations are significantly more robust to natural and synthetic deformati
    

