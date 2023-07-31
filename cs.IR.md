# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Framework to Automatically Determine the Quality of Open Data Catalogs.](http://arxiv.org/abs/2307.15464) | 本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。 |
| [^2] | [Toward Transparent Sequence Models with Model-Based Tree Markov Model.](http://arxiv.org/abs/2307.15367) | 本研究引入了基于模型的树马尔可夫模型（MOB-HSMM），用于解决复杂黑盒机器学习模型应用于序列数据时的可解释性问题。通过从深度神经网络中蒸馏的知识，实现了提高预测性能的同时提供清晰解释的目标。实验结果表明通过将LSTM学习到的顺序模式转移到MOB树中，可以进一步提高MOB树的性能，并利用MOB-HSMM将MOB树与隐马尔可夫模型（HSMM）整合，实现了潜在和可解释的序列的发现。 |
| [^3] | [Staging E-Commerce Products for Online Advertising using Retrieval Assisted Image Generation.](http://arxiv.org/abs/2307.15326) | 提出了使用检索辅助图像生成的方法来为未布置的电子商务产品图像生成吸引人和逼真的舞台背景，以提高在线广告的点击率。 |
| [^4] | [Reconciling the accuracy-diversity trade-off in recommendations.](http://arxiv.org/abs/2307.15142) | 本论文解释了在推荐系统中准确性和多样性之间的权衡，通过考虑用户的消费约束，提出了一种能够产生多样推荐结果的模型。 |
| [^5] | [Mathematical Modeling of BCG-based Bladder Cancer Treatment Using Socio-Demographics.](http://arxiv.org/abs/2307.15084) | 本研究利用患者的社会人口统计数据提供个性化的数学模型，以描述基于BCG的膀胱癌治疗的临床动态。 |
| [^6] | [Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study.](http://arxiv.org/abs/2305.11164) | 本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。 |
| [^7] | [Towards Answering Climate Questionnaires from Unstructured Climate Reports.](http://arxiv.org/abs/2301.04253) | 本研究提出了一种从非结构化的气候报告中回答气候问卷的方法。研究引入两个新的大规模气候问卷数据集，并使用现有结构训练自监督模型，通过实验和人类试验验证了模型的有效性。同时，还引入了一个气候文本分类数据集的基准，以促进气候领域的自然语言处理研究。 |
| [^8] | [Equivariant Contrastive Learning for Sequential Recommendation.](http://arxiv.org/abs/2211.05290) | 本论文提出了序列推荐的等变性对比学习（ECL-SR）方法，通过使用条件判别器来使得学习到的用户行为表示对于侵入性增强敏感并对轻微增强不敏感，从而提高了序列推荐模型的区分能力。 |
| [^9] | [Rethinking Missing Data: Aleatoric Uncertainty-Aware Recommendation.](http://arxiv.org/abs/2209.11679) | 本研究提出了一种基于随机不确定性感知的推荐系统（AUR）框架，通过考虑缺失数据的固有随机性，解决了模型误差问题，提高了对长尾物品的召回效果。 |

# 详细

[^1]: 自动确定开放数据目录质量的框架

    Framework to Automatically Determine the Quality of Open Data Catalogs. (arXiv:2307.15464v1 [cs.IR])

    [http://arxiv.org/abs/2307.15464](http://arxiv.org/abs/2307.15464)

    本文提出了一个框架，用于自动确定开放数据目录的质量，该框架可以分析核心质量维度并提供评估机制，同时也考虑到了非核心质量维度，旨在帮助数据驱动型组织基于可信的数据资产做出明智的决策。

    

    数据目录在现代数据驱动型组织中起着关键作用，通过促进各种数据资产的发现、理解和利用。然而，在开放和大规模数据环境中确保其质量和可靠性是复杂的。本文提出了一个框架，用于自动确定开放数据目录的质量，解决了高效和可靠的质量评估机制的需求。我们的框架可以分析各种核心质量维度，如准确性、完整性、一致性、可扩展性和及时性，提供多种评估兼容性和相似性的替代方案，以及实施一组非核心质量维度，如溯源性、可读性和许可证。其目标是使数据驱动型组织能够基于可信和精心管理的数据资产做出明智的决策。

    Data catalogs play a crucial role in modern data-driven organizations by facilitating the discovery, understanding, and utilization of diverse data assets. However, ensuring their quality and reliability is complex, especially in open and large-scale data environments. This paper proposes a framework to automatically determine the quality of open data catalogs, addressing the need for efficient and reliable quality assessment mechanisms. Our framework can analyze various core quality dimensions, such as accuracy, completeness, consistency, scalability, and timeliness, offer several alternatives for the assessment of compatibility and similarity across such catalogs as well as the implementation of a set of non-core quality dimensions such as provenance, readability, and licensing. The goal is to empower data-driven organizations to make informed decisions based on trustworthy and well-curated data assets. The source code that illustrates our approach can be downloaded from https://www.
    
[^2]: 通过基于模型的树马尔可夫模型，实现透明的序列模型

    Toward Transparent Sequence Models with Model-Based Tree Markov Model. (arXiv:2307.15367v1 [cs.LG])

    [http://arxiv.org/abs/2307.15367](http://arxiv.org/abs/2307.15367)

    本研究引入了基于模型的树马尔可夫模型（MOB-HSMM），用于解决复杂黑盒机器学习模型应用于序列数据时的可解释性问题。通过从深度神经网络中蒸馏的知识，实现了提高预测性能的同时提供清晰解释的目标。实验结果表明通过将LSTM学习到的顺序模式转移到MOB树中，可以进一步提高MOB树的性能，并利用MOB-HSMM将MOB树与隐马尔可夫模型（HSMM）整合，实现了潜在和可解释的序列的发现。

    

    本研究解决了应用于序列数据的复杂、黑盒机器学习模型的可解释性问题。我们引入了基于模型的树隐马尔可夫模型（MOB-HSMM），这是一个固有可解释性的模型，旨在检测高死亡风险事件，并发现与死亡风险相关的隐藏模式。该模型利用从深度神经网络（DNN）中蒸馏的知识，提高预测性能的同时提供清晰的解释。我们的实验结果表明，通过使用LSTM学习顺序模式，进而将其转移给MOB树，可以提高基于模型的树（MOB树）的性能。将MOB树与基于模型的隐马尔可夫模型（HSMM）集成在MOB-HSMM中，可以使用可用信息揭示潜在的和可解释的序列。

    In this study, we address the interpretability issue in complex, black-box Machine Learning models applied to sequence data. We introduce the Model-Based tree Hidden Semi-Markov Model (MOB-HSMM), an inherently interpretable model aimed at detecting high mortality risk events and discovering hidden patterns associated with the mortality risk in Intensive Care Units (ICU). This model leverages knowledge distilled from Deep Neural Networks (DNN) to enhance predictive performance while offering clear explanations. Our experimental results indicate the improved performance of Model-Based trees (MOB trees) via employing LSTM for learning sequential patterns, which are then transferred to MOB trees. Integrating MOB trees with the Hidden Semi-Markov Model (HSMM) in the MOB-HSMM enables uncovering potential and explainable sequences using available information.
    
[^3]: 使用检索辅助图像生成为电子商务产品进行在线广告展示

    Staging E-Commerce Products for Online Advertising using Retrieval Assisted Image Generation. (arXiv:2307.15326v1 [cs.CV])

    [http://arxiv.org/abs/2307.15326](http://arxiv.org/abs/2307.15326)

    提出了使用检索辅助图像生成的方法来为未布置的电子商务产品图像生成吸引人和逼真的舞台背景，以提高在线广告的点击率。

    

    在线广告通常依赖电子商务平台通过目录将产品的图像发送给广告平台。在广告行业中，这样的广告通常被称为动态产品广告(DPA)。 DPA目录通常包含数百万个产品图像（与可以从电子商务平台购买的产品数量相对应）。然而，并非目录中的所有产品图像在直接重新定位为广告图像时都会吸引人，这可能会导致较低的点击率(CTR)。特别地，只放置在纯色背景上的产品可能不如在自然环境中布置的产品吸引人和逼真。为了解决DPA图像在大规模上的这些缺点，我们提出了一种基于生成对抗网络（GAN）的方法来为未布置产品图像生成舞台背景。生成整个布置的背景是一个具有挑战性的任务，容易产生幻觉。为了解决这个问题，我们引入了一个更简单的方法来生成隐含的框架。

    Online ads showing e-commerce products typically rely on the product images in a catalog sent to the advertising platform by an e-commerce platform. In the broader ads industry such ads are called dynamic product ads (DPA). It is common for DPA catalogs to be in the scale of millions (corresponding to the scale of products which can be bought from the e-commerce platform). However, not all product images in the catalog may be appealing when directly re-purposed as an ad image, and this may lead to lower click-through rates (CTRs). In particular, products just placed against a solid background may not be as enticing and realistic as a product staged in a natural environment. To address such shortcomings of DPA images at scale, we propose a generative adversarial network (GAN) based approach to generate staged backgrounds for un-staged product images. Generating the entire staged background is a challenging task susceptible to hallucinations. To get around this, we introduce a simpler ap
    
[^4]: 在推荐系统中权衡准确性和多样性的方法

    Reconciling the accuracy-diversity trade-off in recommendations. (arXiv:2307.15142v1 [cs.IR])

    [http://arxiv.org/abs/2307.15142](http://arxiv.org/abs/2307.15142)

    本论文解释了在推荐系统中准确性和多样性之间的权衡，通过考虑用户的消费约束，提出了一种能够产生多样推荐结果的模型。

    

    在推荐系统中，准确性（推荐用户最有可能想要的物品）和多样性（推荐代表不同类别的物品）之间存在明显的权衡。因此，在实际的推荐系统中，多样性常常被单独考虑，而不是与准确性一起考虑。然而，这种方法没有回答一个基本问题：为什么首先存在这种权衡？我们通过用户的消费约束解释了这种权衡，用户通常只会消费其中几个被推荐的物品。在我们引入的简化模型中，考虑了这种约束的目标可以产生多样的推荐结果，而不考虑这种约束的目标则产生同质的推荐结果。这表明准确性和多样性之间看起来不协调，是因为标准的准确性度量没有考虑消费约束。我们的模型对不同的多样性有精确且可解释的描述。

    In recommendation settings, there is an apparent trade-off between the goals of accuracy (to recommend items a user is most likely to want) and diversity (to recommend items representing a range of categories). As such, real-world recommender systems often explicitly incorporate diversity separately from accuracy. This approach, however, leaves a basic question unanswered: Why is there a trade-off in the first place?  We show how the trade-off can be explained via a user's consumption constraints -- users typically only consume a few of the items they are recommended. In a stylized model we introduce, objectives that account for this constraint induce diverse recommendations, while objectives that do not account for this constraint induce homogeneous recommendations. This suggests that accuracy and diversity appear misaligned because standard accuracy metrics do not consider consumption constraints. Our model yields precise and interpretable characterizations of diversity in different 
    
[^5]: 使用社会人口统计数据对基于BCG的膀胱癌治疗进行数学建模

    Mathematical Modeling of BCG-based Bladder Cancer Treatment Using Socio-Demographics. (arXiv:2307.15084v1 [cs.LG])

    [http://arxiv.org/abs/2307.15084](http://arxiv.org/abs/2307.15084)

    本研究利用患者的社会人口统计数据提供个性化的数学模型，以描述基于BCG的膀胱癌治疗的临床动态。

    

    癌症是世界上最常见的疾病之一，每年都有数百万新患者。膀胱癌是一种最普遍的癌症类型，影响所有人，并没有明显的典型患者。目前，BCG免疫治疗是膀胱癌的标准治疗方法，所有患者都会进行每周例行的BCG治疗。由于免疫系统、治疗和癌细胞之间的生物和临床复杂性，BCG治疗的临床结果在患者之间存在显著差异。在本研究中，我们利用患者的社会人口统计数据提供个性化的数学模型，以描述与基于BCG的治疗相关的临床动态。为此，我们采用了一种成熟的BCG治疗模型，并整合了机器学习组件，以在模型内部即时调整和重新配置关键参数，从而促进其个性化。

    Cancer is one of the most widespread diseases around the world with millions of new patients each year. Bladder cancer is one of the most prevalent types of cancer affecting all individuals alike with no obvious prototypical patient. The current standard treatment for BC follows a routine weekly Bacillus Calmette-Guerin (BCG) immunotherapy-based therapy protocol which is applied to all patients alike. The clinical outcomes associated with BCG treatment vary significantly among patients due to the biological and clinical complexity of the interaction between the immune system, treatments, and cancer cells. In this study, we take advantage of the patient's socio-demographics to offer a personalized mathematical model that describes the clinical dynamics associated with BCG-based treatment. To this end, we adopt a well-established BCG treatment model and integrate a machine learning component to temporally adjust and reconfigure key parameters within the model thus promoting its personali
    
[^6]: 探索抱抱脸ML模型的碳足迹：一项存储库挖掘研究

    Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study. (arXiv:2305.11164v1 [cs.LG])

    [http://arxiv.org/abs/2305.11164](http://arxiv.org/abs/2305.11164)

    本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。

    

    机器学习(ML)系统的崛起加剧了它们的碳足迹，这是由于其增加的能力和模型大小所致。然而，目前对ML模型的碳足迹如何实际测量、报告和评估的认识相对较少。因此，本论文旨在分析在Hugging Face上1,417个ML模型和相关数据集的碳足迹测量情况，Hugging Face是最受欢迎的预训练ML模型的存储库。目标是提供有关如何报告和优化ML模型的碳效率的见解和建议。该研究包括Hugging Face Hub API上有关碳排放的第一项存储库挖掘研究。本研究旨在回答两个研究问题：(1) ML模型的创建者如何在Hugging Face Hub上测量和报告碳排放？(2) 哪些方面影响了训练ML模型的碳排放？该研究得出了几个关键发现。其中包括碳排放报告模式比例的逐步下降等。

    The rise of machine learning (ML) systems has exacerbated their carbon footprint due to increased capabilities and model sizes. However, there is scarce knowledge on how the carbon footprint of ML models is actually measured, reported, and evaluated. In light of this, the paper aims to analyze the measurement of the carbon footprint of 1,417 ML models and associated datasets on Hugging Face, which is the most popular repository for pretrained ML models. The goal is to provide insights and recommendations on how to report and optimize the carbon efficiency of ML models. The study includes the first repository mining study on the Hugging Face Hub API on carbon emissions. This study seeks to answer two research questions: (1) how do ML model creators measure and report carbon emissions on Hugging Face Hub?, and (2) what aspects impact the carbon emissions of training ML models? The study yielded several key findings. These include a decreasing proportion of carbon emissions-reporting mode
    
[^7]: 从非结构化的气候报告中回答气候问卷的方法

    Towards Answering Climate Questionnaires from Unstructured Climate Reports. (arXiv:2301.04253v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.04253](http://arxiv.org/abs/2301.04253)

    本研究提出了一种从非结构化的气候报告中回答气候问卷的方法。研究引入两个新的大规模气候问卷数据集，并使用现有结构训练自监督模型，通过实验和人类试验验证了模型的有效性。同时，还引入了一个气候文本分类数据集的基准，以促进气候领域的自然语言处理研究。

    

    尽管气候变化问题紧迫，但在自然语言处理领域对其的关注有限。行动者和政策制定者需要自然语言处理工具，能够有效地将庞大且快速增长的非结构化文本气候报告转化为结构化形式。为了应对这一挑战，我们引入了两个新的大规模气候问卷数据集，并利用其现有结构来训练自监督模型。我们进行实验表明，这些模型能够学习到对训练过程中未见的不同组织类型的气候披露进行泛化。然后，我们使用这些模型在人类试验中帮助将非结构化气候文档中的文本与半结构化问卷对齐。最后，为了支持气候领域进一步的自然语言处理研究，我们引入了一个现有气候文本分类数据集的基准，以更好地评估和比较现有模型。

    The topic of Climate Change (CC) has received limited attention in NLP despite its urgency. Activists and policymakers need NLP tools to effectively process the vast and rapidly growing unstructured textual climate reports into structured form. To tackle this challenge we introduce two new large-scale climate questionnaire datasets and use their existing structure to train self-supervised models. We conduct experiments to show that these models can learn to generalize to climate disclosures of different organizations types than seen during training. We then use these models to help align texts from unstructured climate documents to the semi-structured questionnaires in a human pilot study. Finally, to support further NLP research in the climate domain we introduce a benchmark of existing climate text classification datasets to better evaluate and compare existing models.
    
[^8]: 序列推荐的等变性对比学习

    Equivariant Contrastive Learning for Sequential Recommendation. (arXiv:2211.05290v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2211.05290](http://arxiv.org/abs/2211.05290)

    本论文提出了序列推荐的等变性对比学习（ECL-SR）方法，通过使用条件判别器来使得学习到的用户行为表示对于侵入性增强敏感并对轻微增强不敏感，从而提高了序列推荐模型的区分能力。

    

    对比学习（CL）通过信息自监督信号有益于训练序列推荐模型。现有的解决方案采用通用的序列数据增强策略生成正样本，并鼓励它们的表示具有不变性。然而，由于用户行为序列的固有特性，一些增强策略（如物品替换）可能导致用户意图的改变。为了避免不选定适所有增强策略的不变表示，我们提出了序列推荐的等变性对比学习（ECL-SR），该方法赋予SR模型强大的区分能力，使学习到的用户行为表示对侵入性增强（如物品替换）敏感并对轻微增强（如特征级丢失遮蔽）不敏感。具体而言，我们使用条件判别器来捕捉由于物品替换而产生的行为差异。

    Contrastive learning (CL) benefits the training of sequential recommendation models with informative self-supervision signals. Existing solutions apply general sequential data augmentation strategies to generate positive pairs and encourage their representations to be invariant. However, due to the inherent properties of user behavior sequences, some augmentation strategies, such as item substitution, can lead to changes in user intent. Learning indiscriminately invariant representations for all augmentation strategies might be suboptimal. Therefore, we propose Equivariant Contrastive Learning for Sequential Recommendation (ECL-SR), which endows SR models with great discriminative power, making the learned user behavior representations sensitive to invasive augmentations (e.g., item substitution) and insensitive to mild augmentations (e.g., featurelevel dropout masking). In detail, we use the conditional discriminator to capture differences in behavior due to item substitution, which e
    
[^9]: 重新思考缺失数据：基于随机不确定性感知的推荐系统

    Rethinking Missing Data: Aleatoric Uncertainty-Aware Recommendation. (arXiv:2209.11679v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2209.11679](http://arxiv.org/abs/2209.11679)

    本研究提出了一种基于随机不确定性感知的推荐系统（AUR）框架，通过考虑缺失数据的固有随机性，解决了模型误差问题，提高了对长尾物品的召回效果。

    

    历史交互是推荐模型训练的默认选择，但通常呈现很高的稀疏性，即大部分用户-物品对是未观察到的缺失数据。标准的做法是将缺失数据视为负样本，并在观察到的交互中估计用户-物品对之间的交互概率。然而，这种训练方式不可避免地会导致一些潜在的交互被错误标记，损害模型的准确性，尤其是对于长尾物品的召回效果。在本文中，我们从随机不确定性的新视角研究了错标问题，这种不确定性描述了缺失数据的固有随机性。这种随机性促使我们超越仅仅考虑交互概率，而是采用随机不确定性建模。为此，我们提出了一种新的基于随机不确定性感知的推荐系统（AUR）框架，其中包括一个新的不确定性估计器和一个普通推荐模型。

    Historical interactions are the default choice for recommender model training, which typically exhibit high sparsity, i.e., most user-item pairs are unobserved missing data. A standard choice is treating the missing data as negative training samples and estimating interaction likelihood between user-item pairs along with the observed interactions. In this way, some potential interactions are inevitably mislabeled during training, which will hurt the model fidelity, hindering the model to recall the mislabeled items, especially the long-tail ones. In this work, we investigate the mislabeling issue from a new perspective of aleatoric uncertainty, which describes the inherent randomness of missing data. The randomness pushes us to go beyond merely the interaction likelihood and embrace aleatoric uncertainty modeling. Towards this end, we propose a new Aleatoric Uncertainty-aware Recommendation (AUR) framework that consists of a new uncertainty estimator along with a normal recommender mod
    

