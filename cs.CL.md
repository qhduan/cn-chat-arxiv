# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs](https://arxiv.org/abs/2403.19827) | 语言模型通过从相关结构（例如“a few days”）进行泛化学习，能够更好地学习AANN结构。 |
| [^2] | [FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation](https://arxiv.org/abs/2403.08059) | FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。 |

# 详细

[^1]: 语言模型从不常见的现象中学习：缺失AANN的情况

    Language Models Learn Rare Phenomena from Less Rare Phenomena: The Case of the Missing AANNs

    [https://arxiv.org/abs/2403.19827](https://arxiv.org/abs/2403.19827)

    语言模型通过从相关结构（例如“a few days”）进行泛化学习，能够更好地学习AANN结构。

    

    语言模型学习罕见的句法现象，但有人认为它们依赖于死记硬背，而不是语法概括。我们在规模为人类规模的语料库（1亿字）上进行训练，迭代训练变压器语言模型，然后评估它们对特定罕见语法现象的学习：英语的冠词+形容词+数字+名词（AANN）结构（“a beautiful five days”）。

    arXiv:2403.19827v1 Announce Type: new  Abstract: Language models learn rare syntactic phenomena, but it has been argued that they rely on rote memorization, as opposed to grammatical generalization. Training on a corpus of human-scale in size (100M words), we iteratively trained transformer language models on systematically manipulated corpora and then evaluated their learning of a particular rare grammatical phenomenon: the English Article+Adjective+Numeral+Noun (AANN) construction (``a beautiful five days''). We first compared how well this construction was learned on the default corpus relative to a counterfactual corpus in which the AANN sentences were removed. AANNs were still learned better than systematically perturbed variants of the construction. Using additional counterfactual corpora, we suggest that this learning occurs through generalization from related constructions (e.g., ``a few days''). An additional experiment showed that this learning is enhanced when there is more 
    
[^2]: FluoroSAM: 用于X光图像分割的语言对齐基础模型

    FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation

    [https://arxiv.org/abs/2403.08059](https://arxiv.org/abs/2403.08059)

    FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。

    

    自动X光图像分割将加速诊断和介入精准医学领域的研究和发展。先前的研究已经提出了适用于解决特定图像分析问题的特定任务模型，但这些模型的效用受限于特定任务领域，要拓展到更广泛的应用则需要额外的数据、标签和重新训练工作。最近，基础模型（FMs） - 训练在大量高度变化数据上的机器学习模型因此使得广泛适用性成为可能 - 已经成为自动图像分析的有希望的工具。现有的用于医学图像分析的FMs聚焦于对象被明显可见边界清晰定义的场景和模式，如内窥镜手术工具分割。相比之下，X光成像通常没有提供这种清晰的边界或结构先验。在X光图像形成期间，复杂的三维

    arXiv:2403.08059v1 Announce Type: cross  Abstract: Automated X-ray image segmentation would accelerate research and development in diagnostic and interventional precision medicine. Prior efforts have contributed task-specific models capable of solving specific image analysis problems, but the utility of these models is restricted to their particular task domain, and expanding to broader use requires additional data, labels, and retraining efforts. Recently, foundation models (FMs) -- machine learning models trained on large amounts of highly variable data thus enabling broad applicability -- have emerged as promising tools for automated image analysis. Existing FMs for medical image analysis focus on scenarios and modalities where objects are clearly defined by visually apparent boundaries, such as surgical tool segmentation in endoscopy. X-ray imaging, by contrast, does not generally offer such clearly delineated boundaries or structure priors. During X-ray image formation, complex 3D
    

