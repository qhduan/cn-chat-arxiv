# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability](https://arxiv.org/abs/2403.04483) | 该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。 |
| [^2] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^3] | [Arukikata Travelogue Dataset.](http://arxiv.org/abs/2305.11444) | Arukikata旅游游记数据集是一个包含超过3100万个日文单词的数据集，包括4672个日本国内游记和9607个海外游记，为研究人员提供了可重复和透明的研究数据。 |

# 详细

[^1]: 使用图理解和推理功能增强大规模语言模型的GraphInstruct

    GraphInstruct: Empowering Large Language Models with Graph Understanding and Reasoning Capability

    [https://arxiv.org/abs/2403.04483](https://arxiv.org/abs/2403.04483)

    该论文提出了一个名为GraphInstruct的基准，用于评估和增强大规模语言模型的图理解能力，并通过构建GraphLM和提出GraphLM+模型实现了显著的图推理能力增强。

    

    评估和增强大规模语言模型（LLMs）的通用能力一直是一个重要的研究课题。图是现实世界中常见的数据结构，理解图数据对于推进通用智能至关重要。为了评估和增强LLMs的图理解能力，在本文中，我们提出了一个名为GraphInstruct的基准，全面包括21个经典图推理任务，提供多样的图生成流水线和详细的推理步骤。基于GraphInstruct，我们进一步通过高效的指导调整构建了GraphLM，展示出显著的图理解能力。为了增强LLM的图推理能力，我们提出了一种步骤掩码训练策略，并构建了一个名为GraphLM+的模型。作为增强LLMs图理解和推理能力的先驱性努力之一，我们进行了大量实验。

    arXiv:2403.04483v1 Announce Type: new  Abstract: Evaluating and enhancing the general capabilities of large language models (LLMs) has been an important research topic. Graph is a common data structure in the real world, and understanding graph data is a crucial part for advancing general intelligence. To evaluate and enhance the graph understanding abilities of LLMs, in this paper, we propose a benchmark named GraphInstruct, which comprehensively includes 21 classical graph reasoning tasks, providing diverse graph generation pipelines and detailed reasoning steps. Based on GraphInstruct, we further construct GraphLM through efficient instruction-tuning, which shows prominent graph understanding capability. In order to enhance the LLM with graph reasoning capability as well, we propose a step mask training strategy, and construct a model named GraphLM+. As one of the pioneering efforts to enhance the graph understanding and reasoning abilities of LLMs, extensive experiments have demons
    
[^2]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^3]: Arukikata旅游游记数据集 (arXiv:2305.11444v1 [cs.CL])

    Arukikata Travelogue Dataset. (arXiv:2305.11444v1 [cs.CL])

    [http://arxiv.org/abs/2305.11444](http://arxiv.org/abs/2305.11444)

    Arukikata旅游游记数据集是一个包含超过3100万个日文单词的数据集，包括4672个日本国内游记和9607个海外游记，为研究人员提供了可重复和透明的研究数据。

    

    我们创建了Arukikata旅游游记数据集，并免费提供给学术研究使用。该数据集包含超过3100万个日文单词，包括4672个日本国内游记和9607个海外游记。在我们提供数据集之前，很难获得可用于研究的广泛旅游游记数据，每个研究人员都必须准备自己的数据。这阻碍了对现有研究的复制以及对实验结果进行公正比较分析。我们的数据集使得任何研究人员都可以对相同的数据进行研究，并确保研究的透明度和可重复性。 在本文中，我们描述了我们的数据集的学术意义、特点和前景。

    We have constructed Arukikata Travelogue Dataset and released it free of charge for academic research. This dataset is a Japanese text dataset with a total of over 31 million words, comprising 4,672 Japanese domestic travelogues and 9,607 overseas travelogues. Before providing our dataset, there was a scarcity of widely available travelogue data for research purposes, and each researcher had to prepare their own data. This hinders the replication of existing studies and fair comparative analysis of experimental results. Our dataset enables any researchers to conduct investigation on the same data and to ensure transparency and reproducibility in research. In this paper, we describe the academic significance, characteristics, and prospects of our dataset.
    

