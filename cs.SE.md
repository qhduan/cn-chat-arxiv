# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Software Vulnerability Prediction Knowledge Transferring Between Programming Languages.](http://arxiv.org/abs/2303.06177) | 本研究提出了一种转移学习技术，利用可用数据集生成一个模型，以检测不同编程语言中的常见漏洞。结果表明，所提出的模型以平均召回率为72％检测C和Java代码中的漏洞。 |
| [^2] | [A Study of Variable-Role-based Feature Enrichment in Neural Models of Code.](http://arxiv.org/abs/2303.04942) | 本文研究了一种基于变量角色的无监督特征增强方法对代码神经模型性能的影响，通过在数据集程序中添加单个变量的角色来丰富源代码数据集，并因此对变量角色增强在训练Code2Seq模型中的影响进行了研究。 |

# 详细

[^1]: 编程语言之间的软件漏洞预测知识转移

    Software Vulnerability Prediction Knowledge Transferring Between Programming Languages. (arXiv:2303.06177v1 [cs.SE])

    [http://arxiv.org/abs/2303.06177](http://arxiv.org/abs/2303.06177)

    本研究提出了一种转移学习技术，利用可用数据集生成一个模型，以检测不同编程语言中的常见漏洞。结果表明，所提出的模型以平均召回率为72％检测C和Java代码中的漏洞。

    This study proposes a transfer learning technique to detect common vulnerabilities in different programming languages by leveraging available datasets. The results show that the proposed model detects vulnerabilities in both C and Java codes with an average recall of 72%.

    开发自动化和智能的软件漏洞检测模型一直受到研究和开发社区的关注。这个领域最大的挑战之一是缺乏所有不同编程语言的代码样本。在本研究中，我们通过提出一种转移学习技术来解决这个问题，利用可用数据集生成一个模型，以检测不同编程语言中的常见漏洞。我们使用C源代码样本训练卷积神经网络（CNN）模型，然后使用Java源代码样本来采用和评估学习的模型。我们使用两个基准数据集的代码样本：NIST软件保障参考数据集（SARD）和Draper VDISC数据集。结果表明，所提出的模型以平均召回率为72％检测C和Java代码中的漏洞。此外，我们采用可解释的AI来调查每个特征对知识转移机制的贡献程度。

    Developing automated and smart software vulnerability detection models has been receiving great attention from both research and development communities. One of the biggest challenges in this area is the lack of code samples for all different programming languages. In this study, we address this issue by proposing a transfer learning technique to leverage available datasets and generate a model to detect common vulnerabilities in different programming languages. We use C source code samples to train a Convolutional Neural Network (CNN) model, then, we use Java source code samples to adopt and evaluate the learned model. We use code samples from two benchmark datasets: NIST Software Assurance Reference Dataset (SARD) and Draper VDISC dataset. The results show that proposed model detects vulnerabilities in both C and Java codes with average recall of 72\%. Additionally, we employ explainable AI to investigate how much each feature contributes to the knowledge transfer mechanisms between 
    
[^2]: 代码神经模型中基于变量角色的特征增强研究

    A Study of Variable-Role-based Feature Enrichment in Neural Models of Code. (arXiv:2303.04942v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.04942](http://arxiv.org/abs/2303.04942)

    本文研究了一种基于变量角色的无监督特征增强方法对代码神经模型性能的影响，通过在数据集程序中添加单个变量的角色来丰富源代码数据集，并因此对变量角色增强在训练Code2Seq模型中的影响进行了研究。

    This paper investigates the impact of an unsupervised feature enrichment approach based on variable roles on the performance of neural models of code, and enriches a source code dataset by adding the role of individual variables in the dataset programs, thereby conducting a study on the impact of variable role enrichment in training the Code2Seq model.

    尽管深度神经模型大大减少了特征工程的开销，但输入中可用的特征可能会显著影响模型的训练成本和性能。本文探讨了一种基于变量角色的无监督特征增强方法对代码神经模型性能的影响。变量角色的概念（如Sajaniemi等人的作品中所介绍的）已被发现有助于学生的编程能力。本文研究了这个概念是否会提高代码神经模型的性能。据我们所知，这是第一篇研究Sajaniemi等人的变量角色概念如何影响代码神经模型的工作。具体而言，我们通过在数据集程序中添加单个变量的角色来丰富源代码数据集，并因此对变量角色增强在训练Code2Seq模型中的影响进行了研究。

    Although deep neural models substantially reduce the overhead of feature engineering, the features readily available in the inputs might significantly impact training cost and the performance of the models. In this paper, we explore the impact of an unsuperivsed feature enrichment approach based on variable roles on the performance of neural models of code. The notion of variable roles (as introduced in the works of Sajaniemi et al. [Refs. 1,2]) has been found to help students' abilities in programming. In this paper, we investigate if this notion would improve the performance of neural models of code. To the best of our knowledge, this is the first work to investigate how Sajaniemi et al.'s concept of variable roles can affect neural models of code. In particular, we enrich a source code dataset by adding the role of individual variables in the dataset programs, and thereby conduct a study on the impact of variable role enrichment in training the Code2Seq model. In addition, we shed l
    

