# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation](https://arxiv.org/abs/2403.17214) | 本研究实证分析了11种流行的专门调整的大型语言模型在五种语言上生成的输出，发现其中26.4%到73.7%的代码翻译需要后处理。 |
| [^2] | [DeAL: Decoding-time Alignment for Large Language Models](https://arxiv.org/abs/2402.06147) | DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。 |
| [^3] | [Discovery of the Hidden World with Large Language Models](https://arxiv.org/abs/2402.03941) | 通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。 |
| [^4] | [Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends](https://arxiv.org/abs/2401.07518) | 这篇论文调查了教育领域自然语言处理的最新进展，提出了分类体系，并总结了挑战和未来研究方向。 |
| [^5] | [Leveraging Twitter Data for Sentiment Analysis of Transit User Feedback: An NLP Framework.](http://arxiv.org/abs/2310.07086) | 本论文提出了一个基于自然语言处理的框架，利用推特数据进行交通用户反馈的情感分析。通过少样本学习识别推特中的问题，并采用词典情感分析模型评估推特情感的强度和极性。 |
| [^6] | [Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection.](http://arxiv.org/abs/2308.06701) | 该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。 |
| [^7] | [ChipGPT: How far are we from natural language hardware design.](http://arxiv.org/abs/2305.14019) | 这篇论文介绍了ChipGPT，一个自动化设计环境，它利用大型语言模型从自然语言规范生成硬件逻辑设计，并展示了与人工设计性能相媲美的结果，且可节省超过75％的编码时间。 |
| [^8] | [SANTA: Separate Strategies for Inaccurate and Incomplete Annotation Noise in Distantly-Supervised Named Entity Recognition.](http://arxiv.org/abs/2305.04076) | 本文提出了一种处理Distantly-Supervised Named Entity Recognition中错误和不完整标注噪声的分离策略，使用不同的模型构建来应对两种类型的噪声。 |
| [^9] | [Online Loss Function Learning.](http://arxiv.org/abs/2301.13247) | 在线损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们的方法在多个任务上稳定地优于现有技术。 |

# 详细

[^1]: 探究输出格式对大型语言模型在代码翻译评估中的影响

    Exploring the Impact of the Output Format on the Evaluation of Large Language Models for Code Translation

    [https://arxiv.org/abs/2403.17214](https://arxiv.org/abs/2403.17214)

    本研究实证分析了11种流行的专门调整的大型语言模型在五种语言上生成的输出，发现其中26.4%到73.7%的代码翻译需要后处理。

    

    编程语言之间的代码翻译是软件工程中长期存在且至关重要的任务，有助于现代化遗留系统，确保跨平台兼容性，提升软件性能。随着大型语言模型（LLMs）及其在代码翻译中的应用的最新进展，对这些模型进行全面评估的需求越来越强烈。在本研究中，我们在五种语言（包括C、C++、Go、Java和Python）上，从1B到46.7B的参数范围内对十一种流行的专门调整的LLMs生成的输出进行了实证分析，并涵盖3820个翻译对。我们的分析发现，在我们评估的LLMs中，26.4%到73.7%的代码翻译需要后处理，因为这些翻译通常包含代码、引号和文本的混合，而不仅仅是纯源代码。忽视这些模型的输出格式可能不经意间导致

    arXiv:2403.17214v1 Announce Type: cross  Abstract: Code translation between programming languages is a long-existing and critical task in software engineering, facilitating the modernization of legacy systems, ensuring cross-platform compatibility, and enhancing software performance. With the recent advances in large language models (LLMs) and their applications to code translation, there is an increasing need for comprehensive evaluation of these models. In this study, we empirically analyze the generated outputs of eleven popular instruct-tuned LLMs with parameters ranging from 1B up to 46.7B on 3,820 translation pairs across five languages, including C, C++, Go, Java, and Python. Our analysis found that between 26.4% and 73.7% of code translations produced by our evaluated LLMs necessitate post-processing, as these translations often include a mix of code, quotes, and text rather than being purely source code. Overlooking the output format of these models can inadvertently lead to u
    
[^2]: DeAL：用于大型语言模型的解码时对齐

    DeAL: Decoding-time Alignment for Large Language Models

    [https://arxiv.org/abs/2402.06147](https://arxiv.org/abs/2402.06147)

    DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。

    

    大型语言模型（LLMs）现在期望生成与人类偏好对齐的内容。目前的工作主要集中在模型训练时间对齐上，通过诸如强化学习与人类反馈（RLHF）等技术。然而，目前还不清楚这些方法是否有效地教导模型对齐目标。首先，无法整合多个自定义奖励和依赖模型开发者对通用和静态原则的理解是主要局限。其次，模型训练中的残留差距以及这些方法的可靠性也值得质疑（例如，即使在安全训练后仍然容易被越狱）。为了解决这些问题，我们提出了DeAL，一个允许用户自定义奖励函数并实现解码时对齐LLMs（DeAL）的框架。核心思想在于将解码视为一个启发式引导的搜索过程，并促使使用各种对齐目标。我们的实验以编程约束为例进行了验证。

    Large Language Models (LLMs) are nowadays expected to generate content aligned with human preferences. Current work focuses on alignment at model training time, through techniques such as Reinforcement Learning with Human Feedback (RLHF). However, it is unclear if such methods are an effective choice to teach alignment objectives to the model. First, the inability to incorporate multiple, custom rewards and reliance on a model developer's view of universal and static principles are key limitations. Second, the residual gaps in model training and the reliability of such approaches are also questionable (e.g. susceptibility to jail-breaking even after safety training). To address these, we propose DeAL, a framework that allows the user to customize reward functions and enables Decoding-time Alignment of LLMs (DeAL). At its core, we view decoding as a heuristic-guided search process and facilitate the use of a wide variety of alignment objectives. Our experiments with programmatic constra
    
[^3]: 用大型语言模型探索隐藏世界

    Discovery of the Hidden World with Large Language Models

    [https://arxiv.org/abs/2402.03941](https://arxiv.org/abs/2402.03941)

    通过使用大型语言模型，我们提出了COAT：因果表示助手，该助手从原始观测数据中提取潜在的因果因子，并将其转化为结构化数据，为探索隐藏世界提供了新的机会。

    

    科学起源于从已知事实和观察中发现新的因果知识。传统的因果发现方法主要依赖于高质量的测量变量，通常由人类专家提供，以找到因果关系。然而，在许多现实世界的应用中，因果变量通常无法获取。大型语言模型（LLMs）的崛起为从原始观测数据中发现高级隐藏变量提供了新的机会。因此，我们介绍了COAT：因果表示助手。COAT将LLMs作为因素提供器引入，提取出来自非结构化数据的潜在因果因子。此外，LLMs还可以被指示提供用于收集数据值（例如注释标准）的额外信息，并将原始非结构化数据进一步解析为结构化数据。注释数据将被输入到...

    Science originates with discovering new causal knowledge from a combination of known facts and observations. Traditional causal discovery approaches mainly rely on high-quality measured variables, usually given by human experts, to find causal relations. However, the causal variables are usually unavailable in a wide range of real-world applications. The rise of large language models (LLMs) that are trained to learn rich knowledge from the massive observations of the world, provides a new opportunity to assist with discovering high-level hidden variables from the raw observational data. Therefore, we introduce COAT: Causal representatiOn AssistanT. COAT incorporates LLMs as a factor proposer that extracts the potential causal factors from unstructured data. Moreover, LLMs can also be instructed to provide additional information used to collect data values (e.g., annotation criteria) and to further parse the raw unstructured data into structured data. The annotated data will be fed to a
    
[^4]: 教育领域自然语言处理的调查：分类体系、系统综述和未来趋势

    Survey of Natural Language Processing for Education: Taxonomy, Systematic Review, and Future Trends

    [https://arxiv.org/abs/2401.07518](https://arxiv.org/abs/2401.07518)

    这篇论文调查了教育领域自然语言处理的最新进展，提出了分类体系，并总结了挑战和未来研究方向。

    

    自然语言处理（NLP）旨在通过计算机科学领域的技术分析文本，应用于医疗保健、商业和教育领域。特别是，在教育领域，NLP已经被应用于教学和学习方面的帮助。本调查研究主要关注解决与教育领域相关的问题，并回顾了NLP的最新进展。具体来说，我们从介绍相关背景开始，然后提出教育领域NLP的分类系统。接着，我们根据上述分类系统说明任务定义、挑战和相应的技术。之后，我们展示了该领域中的一些现有演示，并总结了未来的研究方向。

    Natural Language Processing (NLP) aims to analyze the text via techniques in the computer science field. It serves the applications in healthcare, commerce, and education domains. Particularly, NLP has been applied to the education domain to help teaching and learning. In this survey, we review recent advances in NLP with a focus on solving problems related to the education domain. In detail, we begin with introducing the relevant background. Then, we present the taxonomy of NLP in the education domain. Next, we illustrate the task definition, challenges, and corresponding techniques based on the above taxonomy. After that, we showcase some off-the-shelf demonstrations in this domain and conclude with future directions.
    
[^5]: 利用推特数据进行交通用户反馈的情感分析：一个自然语言处理框架

    Leveraging Twitter Data for Sentiment Analysis of Transit User Feedback: An NLP Framework. (arXiv:2310.07086v1 [cs.AI])

    [http://arxiv.org/abs/2310.07086](http://arxiv.org/abs/2310.07086)

    本论文提出了一个基于自然语言处理的框架，利用推特数据进行交通用户反馈的情感分析。通过少样本学习识别推特中的问题，并采用词典情感分析模型评估推特情感的强度和极性。

    

    传统的通过交通调查收集用户反馈的方法往往耗时、资源密集且昂贵。在本论文中，我们提出了一种新颖的基于自然语言处理的框架，利用推特等社交媒体平台上广泛、丰富且廉价的数据，来了解用户对各种服务问题的感知。推特作为一个微博平台，托管了大量实时的用户生成内容，其中经常包含有关各种产品、服务和体验的有价值的反馈和意见。所提出的框架通过两种技术简化了收集和分析用户反馈的过程，无需昂贵且耗时的用户反馈调查。首先，它利用少样本学习进行推特分类，有效地识别推特中描述的问题。然后，它采用基于词典的情感分析模型来评估推特情感的强度和极性。

    Traditional methods of collecting user feedback through transit surveys are often time-consuming, resource intensive, and costly. In this paper, we propose a novel NLP-based framework that harnesses the vast, abundant, and inexpensive data available on social media platforms like Twitter to understand users' perceptions of various service issues. Twitter, being a microblogging platform, hosts a wealth of real-time user-generated content that often includes valuable feedback and opinions on various products, services, and experiences. The proposed framework streamlines the process of gathering and analyzing user feedback without the need for costly and time-consuming user feedback surveys using two techniques. First, it utilizes few-shot learning for tweet classification within predefined categories, allowing effective identification of the issues described in tweets. It then employs a lexicon-based sentiment analysis model to assess the intensity and polarity of the tweet sentiments, d
    
[^6]: 伪装图像合成是提高伪装物体检测的关键

    Camouflaged Image Synthesis Is All You Need to Boost Camouflaged Detection. (arXiv:2308.06701v1 [cs.CV])

    [http://arxiv.org/abs/2308.06701](http://arxiv.org/abs/2308.06701)

    该研究提出了一个用于合成伪装数据以改善对自然场景中伪装物体检测的框架，该方法利用生成模型生成逼真的伪装图像，并在三个数据集上取得了优于目前最先进方法的结果。

    

    融入自然场景的伪装物体给深度学习模型检测和合成带来了重大挑战。伪装物体检测是计算机视觉中一个关键任务，具有广泛的实际应用，然而由于数据有限，该研究课题一直受到限制。我们提出了一个用于合成伪装数据以增强对自然场景中伪装物体检测的框架。我们的方法利用生成模型生成逼真的伪装图像，这些图像可以用来训练现有的物体检测模型。具体而言，我们使用伪装环境生成器，由伪装分布分类器进行监督，合成伪装图像，然后将其输入我们的生成器以扩展数据集。我们的框架在三个数据集（COD10k、CAMO和CHAMELEON）上的效果超过了目前最先进的方法，证明了它在改善伪装物体检测方面的有效性。

    Camouflaged objects that blend into natural scenes pose significant challenges for deep-learning models to detect and synthesize. While camouflaged object detection is a crucial task in computer vision with diverse real-world applications, this research topic has been constrained by limited data availability. We propose a framework for synthesizing camouflage data to enhance the detection of camouflaged objects in natural scenes. Our approach employs a generative model to produce realistic camouflage images, which can be used to train existing object detection models. Specifically, we use a camouflage environment generator supervised by a camouflage distribution classifier to synthesize the camouflage images, which are then fed into our generator to expand the dataset. Our framework outperforms the current state-of-the-art method on three datasets (COD10k, CAMO, and CHAMELEON), demonstrating its effectiveness in improving camouflaged object detection. This approach can serve as a plug-
    
[^7]: ChipGPT: 远离自然语言硬件设计还有多远

    ChipGPT: How far are we from natural language hardware design. (arXiv:2305.14019v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2305.14019](http://arxiv.org/abs/2305.14019)

    这篇论文介绍了ChipGPT，一个自动化设计环境，它利用大型语言模型从自然语言规范生成硬件逻辑设计，并展示了与人工设计性能相媲美的结果，且可节省超过75％的编码时间。

    

    随着大型语言模型（LLMs）如ChatGPT展示了前所未有的机器智能，它在通过自然语言交互来协助硬件工程师实现更高效的逻辑设计方面也表现出极佳的性能。为了评估LLMs协助硬件设计过程的潜力，本文尝试演示一个自动化设计环境，该环境利用LLMs从自然语言规范生成硬件逻辑设计。为了实现更易用且更高效的芯片开发流程，我们提出了一种基于LLMs的可扩展的四阶段零代码逻辑设计框架，无需重新训练或微调。首先，演示版本ChipGPT通过为LLM生成提示开始，然后产生初始Verilog程序。 其次，输出管理器纠正和优化这些程序，然后将它们收集到最终的设计空间中。最后，ChipGPT将在此空间中搜索以选择符合目标指标的最优设计。评估表明，由ChipGPT设计的逻辑电路的性能与人工设计的性能相当，并且整个过程节省了超过75％的编码时间。

    As large language models (LLMs) like ChatGPT exhibited unprecedented machine intelligence, it also shows great performance in assisting hardware engineers to realize higher-efficiency logic design via natural language interaction. To estimate the potential of the hardware design process assisted by LLMs, this work attempts to demonstrate an automated design environment that explores LLMs to generate hardware logic designs from natural language specifications. To realize a more accessible and efficient chip development flow, we present a scalable four-stage zero-code logic design framework based on LLMs without retraining or finetuning. At first, the demo, ChipGPT, begins by generating prompts for the LLM, which then produces initial Verilog programs. Second, an output manager corrects and optimizes these programs before collecting them into the final design space. Eventually, ChipGPT will search through this space to select the optimal design under the target metrics. The evaluation sh
    
[^8]: SANTA：Distantly-Supervised Named Entity Recognition中处理错误和不完整标注噪声的分离策略

    SANTA: Separate Strategies for Inaccurate and Incomplete Annotation Noise in Distantly-Supervised Named Entity Recognition. (arXiv:2305.04076v1 [cs.CL])

    [http://arxiv.org/abs/2305.04076](http://arxiv.org/abs/2305.04076)

    本文提出了一种处理Distantly-Supervised Named Entity Recognition中错误和不完整标注噪声的分离策略，使用不同的模型构建来应对两种类型的噪声。

    

    远程监督命名实体识别有效地减轻了监督设置中耗时且昂贵的注释负担，但是无上下文的匹配过程和知识库的有限覆盖引入了不准确和不完整的标注噪音。本研究提出了使用不同的策略来处理两种类型的噪声的SANTA，以解决由不准确和不完整标注带来的挑战。

    Distantly-Supervised Named Entity Recognition effectively alleviates the burden of time-consuming and expensive annotation in the supervised setting. But the context-free matching process and the limited coverage of knowledge bases introduce inaccurate and incomplete annotation noise respectively. Previous studies either considered only incomplete annotation noise or indiscriminately handle two types of noise with the same strategy. In this paper, we argue that the different causes of two types of noise bring up the requirement of different strategies in model architecture. Therefore, we propose the SANTA to handle these two types of noise separately with (1) Memory-smoothed Focal Loss and Entity-aware KNN to relieve the entity ambiguity problem caused by inaccurate annotation, and (2) Boundary Mixup to alleviate decision boundary shifting problem caused by incomplete annotation and a noise-tolerant loss to improve the robustness. Benefiting from our separate tailored strategies, we co
    
[^9]: 在线损失函数学习

    Online Loss Function Learning. (arXiv:2301.13247v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13247](http://arxiv.org/abs/2301.13247)

    在线损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们的方法在多个任务上稳定地优于现有技术。

    

    损失函数学习是一种新的元学习范例，旨在自动化为机器学习模型设计损失函数的重要任务。现有的损失函数学习技术已经显示出有希望的结果，经常改善模型的训练动态和最终推理性能。然而，这些技术的一个重要限制是损失函数以线下方式进行元学习，元目标仅考虑训练的前几个步骤，这与训练深度神经网络通常使用的时间范围相比显著较短。这导致对于在训练开始时表现良好但在训练结束时表现不佳的损失函数存在明显的偏差。为了解决这个问题，我们提出了一种新的损失函数学习技术，可以在每次更新基本模型参数后自适应地在线更新损失函数。实验结果表明，我们提出的方法在多个任务上稳定地优于现有技术。

    Loss function learning is a new meta-learning paradigm that aims to automate the essential task of designing a loss function for a machine learning model. Existing techniques for loss function learning have shown promising results, often improving a model's training dynamics and final inference performance. However, a significant limitation of these techniques is that the loss functions are meta-learned in an offline fashion, where the meta-objective only considers the very first few steps of training, which is a significantly shorter time horizon than the one typically used for training deep neural networks. This causes significant bias towards loss functions that perform well at the very start of training but perform poorly at the end of training. To address this issue we propose a new loss function learning technique for adaptively updating the loss function online after each update to the base model parameters. The experimental results show that our proposed method consistently out
    

