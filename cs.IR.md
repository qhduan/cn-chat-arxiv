# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Cross-modal Information Retrieval Possible without Training?.](http://arxiv.org/abs/2304.11095) | 本论文研究了不需要训练，基于简单映射的跨模态信息检索方法，利用来自预训练深度学习模型的编码表示。这种方法可以在语义上将不同模态的数据映射到同一空间，并在文本和图像之间达到有竞争力的性能水平。 |
| [^2] | [Novel Intent Detection and Active Learning Based Classification (Student Abstract).](http://arxiv.org/abs/2304.11058) | 本文提出了一个名为NIDAL的人工智能框架，可以自动检测不同语言中出现的新型意图类别并减少人员标注成本，通过在多个基准数据集上的实验证明，该系统可以实现高准确率和宏F1值。 |
| [^3] | [Can Perturbations Help Reduce Investment Risks? Risk-Aware Stock Recommendation via Split Variational Adversarial Training.](http://arxiv.org/abs/2304.11043) | 本文提出了一种基于分离变分对抗训练的风险感知型股票推荐方法，通过对抗性扰动提高模型对于风险的感知能力，通过变分扰动生成器模拟不同的风险因素并生成代表性的风险指标对抗样本。在真实股票数据上进行的实验表明该方法有效降低了投资风险同时保持高预期收益。 |
| [^4] | [CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval.](http://arxiv.org/abs/2304.11029) | CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。 |
| [^5] | [Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study.](http://arxiv.org/abs/2304.10909) | 本文探究了在MIMIC-III和MIMIC-IV上进行自动化医疗编码的最新机器学习模型，并发现了这些模型的局限性和不足之处。我们提出了一种改进方法，该方法可以更好地评估系统性能，并公开了我们的代码和数据集。 |
| [^6] | [Hear Me Out: A Study on the Use of the Voice Modality for Crowdsourced Relevance Assessments.](http://arxiv.org/abs/2304.10881) | 本研究研究了使用语音调制进行众包相关性评估，发现评估员在文本和语音模态下的判断准确度相同，但随着文档长度的增加，在语音构造下做出相关性判断所需的时间显着增加。 |
| [^7] | [EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction.](http://arxiv.org/abs/2304.10711) | 本文提出了一种自适应特征交互学习模型EulerNet，它采用欧拉公式将高阶特征交互映射到复杂向量空间中学习，从而在保持效率的同时提高模型能力。 |
| [^8] | [E Pluribus Unum: Guidelines on Multi-Objective Evaluation of Recommender Systems.](http://arxiv.org/abs/2304.10621) | 本论文介绍了一种基于第一原理的多目标模型选择方法，并提出了一组指南，旨在解决推荐系统多目标评估中的平衡多个绩效指标的挑战。 |
| [^9] | [MATURE-HEALTH: HEALTH Recommender System for MAndatory FeaTURE choices.](http://arxiv.org/abs/2304.09099) | 该论文提出和实施了一个名为MATURE-HEALTH的健康推荐系统，该系统能够预测电解质不平衡并推荐营养平衡的食物，从而增加早期检测疾病的机会并防止健康进一步恶化。 |
| [^10] | [Combat AI With AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media.](http://arxiv.org/abs/2302.07731) | 本文针对机器生成的虚假评论提出了一种用高质量餐厅评论生成虚假评论并微调GPT输出检测器的方法，该方法预测虚假评论的性能优于现有解决方案。同时，我们还探索了预测非精英评论的模型，并在几个维度上对这些评论进行分析，此类机器生成的虚假评论是社交媒体平台面临的持续挑战。 |

# 详细

[^1]: 不需要训练，跨模态信息检索是否可行？

    Is Cross-modal Information Retrieval Possible without Training?. (arXiv:2304.11095v1 [cs.LG])

    [http://arxiv.org/abs/2304.11095](http://arxiv.org/abs/2304.11095)

    本论文研究了不需要训练，基于简单映射的跨模态信息检索方法，利用来自预训练深度学习模型的编码表示。这种方法可以在语义上将不同模态的数据映射到同一空间，并在文本和图像之间达到有竞争力的性能水平。

    

    预训练深度学习模型中编码的表示(例如BERT文本嵌入，图像的倒数第二个卷积神经网络层激活)传递了一组有益的信息检索特征。给定数据模态的嵌入存在自己的高维空间中，但可以通过简单的映射进行语义对齐。在本文中，我们使用来自最小二乘法和奇异值分解 (SVD) 的简单映射作为Procrustes问题的解决方案，从而实现跨模态信息检索的手段。也就是说，给定一个模态中的信息，例如文本，该映射可以帮助我们在另一个模态中找到与其语义相当的数据项，例如图像。使用现成的预训练深度学习模型，我们在文本到图像和图像到文本的检索任务中尝试了上述简单的跨模态映射。尽管简单，我们的映射表现出竞争性的性能，并达到了与最先进方法相当的水平。

    Encoded representations from a pretrained deep learning model (e.g., BERT text embeddings, penultimate CNN layer activations of an image) convey a rich set of features beneficial for information retrieval. Embeddings for a particular modality of data occupy a high-dimensional space of its own, but it can be semantically aligned to another by a simple mapping without training a deep neural net. In this paper, we take a simple mapping computed from the least squares and singular value decomposition (SVD) for a solution to the Procrustes problem to serve a means to cross-modal information retrieval. That is, given information in one modality such as text, the mapping helps us locate a semantically equivalent data item in another modality such as image. Using off-the-shelf pretrained deep learning models, we have experimented the aforementioned simple cross-modal mappings in tasks of text-to-image and image-to-text retrieval. Despite simplicity, our mappings perform reasonably well reachin
    
[^2]: 新型意图检测和基于主动学习的分类（学生摘要）

    Novel Intent Detection and Active Learning Based Classification (Student Abstract). (arXiv:2304.11058v1 [cs.CL])

    [http://arxiv.org/abs/2304.11058](http://arxiv.org/abs/2304.11058)

    本文提出了一个名为NIDAL的人工智能框架，可以自动检测不同语言中出现的新型意图类别并减少人员标注成本，通过在多个基准数据集上的实验证明，该系统可以实现高准确率和宏F1值。

    

    在连续交互的对话代理情境中，新型意图类别检测是一个重要问题。已经进行了许多研究工作，以检测英语为主要文本和图像中的新型意图。但是，当前系统缺乏一种端到端通用框架，以在各种不同语言的同时检测新型意图，同时减少对人类注释的需求以处理被分类错误或系统拒绝的样本。本文提出了NIDAL（Novel Intent Detection and Active Learning based classification），这是一个半监督框架，用于检测新型意图并减少人类注释成本。各种基准数据集上的实验证明，该系统的准确性和宏F1相对于基线方法提高了10%以上，且总注释成本仅为系统可用未标注数据的6-10％。

    Novel intent class detection is an important problem in real world scenario for conversational agents for continuous interaction. Several research works have been done to detect novel intents in a mono-lingual (primarily English) texts and images. But, current systems lack an end-to-end universal framework to detect novel intents across various different languages with less human annotation effort for mis-classified and system rejected samples. This paper proposes NIDAL (Novel Intent Detection and Active Learning based classification), a semi-supervised framework to detect novel intents while reducing human annotation cost. Empirical results on various benchmark datasets demonstrate that this system outperforms the baseline methods by more than 10% margin for accuracy and macro-F1. The system achieves this while maintaining overall annotation cost to be just ~6-10% of the unlabeled data available to the system.
    
[^3]: 扰动有助于降低投资风险吗？ 基于分离变分对抗训练的风险感知型股票推荐方法

    Can Perturbations Help Reduce Investment Risks? Risk-Aware Stock Recommendation via Split Variational Adversarial Training. (arXiv:2304.11043v1 [q-fin.RM])

    [http://arxiv.org/abs/2304.11043](http://arxiv.org/abs/2304.11043)

    本文提出了一种基于分离变分对抗训练的风险感知型股票推荐方法，通过对抗性扰动提高模型对于风险的感知能力，通过变分扰动生成器模拟不同的风险因素并生成代表性的风险指标对抗样本。在真实股票数据上进行的实验表明该方法有效降低了投资风险同时保持高预期收益。

    

    在股票市场，成功的投资需要在利润和风险之间取得良好的平衡。最近，在量化投资中广泛研究了股票推荐，以为投资者选择具有更高收益率的股票。尽管在获利方面取得了成功，但大多数现有的推荐方法仍然在风险控制方面较弱，这可能导致实际股票投资中难以承受的亏损。为了有效降低风险，我们从对抗性扰动中获得启示，并提出了一种新的基于分离变分对抗训练（SVAT）框架的风险感知型股票推荐方法。本质上，SVAT鼓励模型对风险股票样本的对抗性扰动敏感，并通过学习扰动来增强模型的风险意识。为了生成代表性的风险指标对抗样本，我们设计了一个变分扰动生成器来模拟不同的风险因素。特别地，变分结构使我们的方法能够捕捉难以明确量化和建模的各种风险因素。在真实股票数据上的综合实验表明，SVAT在降低投资风险的同时保持高预期收益上非常有效。

    In the stock market, a successful investment requires a good balance between profits and risks. Recently, stock recommendation has been widely studied in quantitative investment to select stocks with higher return ratios for investors. Despite the success in making profits, most existing recommendation approaches are still weak in risk control, which may lead to intolerable paper losses in practical stock investing. To effectively reduce risks, we draw inspiration from adversarial perturbations and propose a novel Split Variational Adversarial Training (SVAT) framework for risk-aware stock recommendation. Essentially, SVAT encourages the model to be sensitive to adversarial perturbations of risky stock examples and enhances the model's risk awareness by learning from perturbations. To generate representative adversarial examples as risk indicators, we devise a variational perturbation generator to model diverse risk factors. Particularly, the variational architecture enables our method
    
[^4]: CLaMP：用于跨模态符号音乐信息检索的对比语言-音乐预训练

    CLaMP: Contrastive Language-Music Pre-training for Cross-Modal Symbolic Music Information Retrieval. (arXiv:2304.11029v1 [cs.SD])

    [http://arxiv.org/abs/2304.11029](http://arxiv.org/abs/2304.11029)

    CLaMP是一种对比语言-音乐预训练技术，能够学习符号音乐和自然语言之间的跨模态表示。通过数据增强和分块处理，它将符号音乐表示成长度不到10％的序列，并使用掩蔽音乐模型预训练目标来增强音乐编码器对音乐上下文和结构的理解。这种技术超越了现有模型的能力，可以实现符号音乐的语义搜索和零样本分类。

    

    我们介绍了CLaMP：对比语言-音乐预训练，它使用音乐编码器和文本编码器通过对比损失函数联合训练来学习自然语言和符号音乐之间的跨模态表示。为了预训练CLaMP，我们收集了140万个音乐-文本对的大型数据集。它使用了文本随机失活来进行数据增强和分块处理以高效地表示音乐数据，从而将序列长度缩短到不到10％。此外，我们开发了一个掩蔽音乐模型预训练目标，以增强音乐编码器对音乐上下文和结构的理解。CLaMP集成了文本信息，以实现符号音乐的语义搜索和零样本分类，超越了先前模型的能力。为支持语义搜索和音乐分类的评估，我们公开发布了WikiMusicText（WikiMT），这是一个包含1010个ABC符号谱的数据集，每个谱都附带有标题、艺术家、流派和描述信息。

    We introduce CLaMP: Contrastive Language-Music Pre-training, which learns cross-modal representations between natural language and symbolic music using a music encoder and a text encoder trained jointly with a contrastive loss. To pre-train CLaMP, we collected a large dataset of 1.4 million music-text pairs. It employed text dropout as a data augmentation technique and bar patching to efficiently represent music data which reduces sequence length to less than 10%. In addition, we developed a masked music model pre-training objective to enhance the music encoder's comprehension of musical context and structure. CLaMP integrates textual information to enable semantic search and zero-shot classification for symbolic music, surpassing the capabilities of previous models. To support the evaluation of semantic search and music classification, we publicly release WikiMusicText (WikiMT), a dataset of 1010 lead sheets in ABC notation, each accompanied by a title, artist, genre, and description.
    
[^5]: MIMIC-III和MIMIC-IV上的自动化医疗编码：一项关键回顾和可复制性研究

    Automated Medical Coding on MIMIC-III and MIMIC-IV: A Critical Review and Replicability Study. (arXiv:2304.10909v1 [cs.LG])

    [http://arxiv.org/abs/2304.10909](http://arxiv.org/abs/2304.10909)

    本文探究了在MIMIC-III和MIMIC-IV上进行自动化医疗编码的最新机器学习模型，并发现了这些模型的局限性和不足之处。我们提出了一种改进方法，该方法可以更好地评估系统性能，并公开了我们的代码和数据集。

    

    医疗编码是将医学代码分配给临床自由文档的任务。医疗专业人士手动分配这些代码以跟踪患者的诊断和治疗。自动化医疗编码可以极大地减轻这种行政负担。本文重现、比较和分析了最先进的自动化医疗编码机器学习模型。我们显示出多个模型表现不佳，原因是配置弱、训练-测试拆分样本不足以及评估不充分。在以往的工作中，宏平均F1分数被计算出亚优的结果，并且我们的修正使其翻倍。我们采用分层抽样和相同的实验设置进行了修订的模型比较，包括超参数和决策边界调整。我们分析预测误差来验证和证伪以前的工作假设。分析证实，所有模型都难以处理稀有的代码，而长文档仅对结果有微不足道的影响。最后，我们提出了一种基于流行病学采样的改进，该方法可以更好地评估系统的性能，并公开了我们的代码和数据集。

    Medical coding is the task of assigning medical codes to clinical free-text documentation. Healthcare professionals manually assign such codes to track patient diagnoses and treatments. Automated medical coding can considerably alleviate this administrative burden. In this paper, we reproduce, compare, and analyze state-of-the-art automated medical coding machine learning models. We show that several models underperform due to weak configurations, poorly sampled train-test splits, and insufficient evaluation. In previous work, the macro F1 score has been calculated sub-optimally, and our correction doubles it. We contribute a revised model comparison using stratified sampling and identical experimental setups, including hyperparameters and decision boundary tuning. We analyze prediction errors to validate and falsify assumptions of previous works. The analysis confirms that all models struggle with rare codes, while long documents only have a negligible impact. Finally, we present the 
    
[^6]: 听我说：使用语音调制进行众包相关性评估的研究

    Hear Me Out: A Study on the Use of the Voice Modality for Crowdsourced Relevance Assessments. (arXiv:2304.10881v1 [cs.IR])

    [http://arxiv.org/abs/2304.10881](http://arxiv.org/abs/2304.10881)

    本研究研究了使用语音调制进行众包相关性评估，发现评估员在文本和语音模态下的判断准确度相同，但随着文档长度的增加，在语音构造下做出相关性判断所需的时间显着增加。

    

    在构建信息检索测试集合时，人工评估员（现今通常是众包工人）创建相关性评估是至关重要的一步。先前的工作调查了评估员的质量和行为，但并没有研究文档的呈现模式对评估员效率和有效性的影响。鉴于语音界面的普及，我们研究了评估员是否能够通过语音界面判断文本文档的相关性，并在众包平台上对 TREC 深度学习语料库中的短文档和长文档进行了用户研究 (n=49)，向参与者展示了文本和语音模态。我们发现：(i)参与者在文本和语音模态下的判断准确度相同；(ii)随着文档长度的增加，参与者在语音构造下做出相关性判断所需的时间显着增加（对于长度>120个单词的文档，所需时间几乎是文本钟的两倍）。

    The creation of relevance assessments by human assessors (often nowadays crowdworkers) is a vital step when building IR test collections. Prior works have investigated assessor quality & behaviour, though into the impact of a document's presentation modality on assessor efficiency and effectiveness. Given the rise of voice-based interfaces, we investigate whether it is feasible for assessors to judge the relevance of text documents via a voice-based interface. We ran a user study (n = 49) on a crowdsourcing platform where participants judged the relevance of short and long documents sampled from the TREC Deep Learning corpus-presented to them either in the text or voice modality. We found that: (i) participants are equally accurate in their judgements across both the text and voice modality; (ii) with increased document length it takes participants significantly longer (for documents of length > 120 words it takes almost twice as much time) to make relevance judgements in the voice con
    
[^7]: EulerNet: 基于欧拉公式的复杂向量空间特征交互学习以实现点击率预测

    EulerNet: Adaptive Feature Interaction Learning via Euler's Formula for CTR Prediction. (arXiv:2304.10711v1 [cs.IR])

    [http://arxiv.org/abs/2304.10711](http://arxiv.org/abs/2304.10711)

    本文提出了一种自适应特征交互学习模型EulerNet，它采用欧拉公式将高阶特征交互映射到复杂向量空间中学习，从而在保持效率的同时提高模型能力。

    

    在点击率预测任务中，学习高阶特征交互是非常关键的。然而，在在线电子商务平台中，由于海量特征的存在，计算高阶特征交互非常耗时。大多数现有方法手动设计最大阶数，并从中过滤出无用的交互。尽管它们减少了高阶特征组合的指数级增长所引起的高计算成本，但由于受到受限的特征阶数的次优学习的影响，它们仍然会受到模型能力下降的影响。保持模型能力并同时保持其效率的解决方案是一个技术挑战，该问题尚未得到充分解决。为了解决这个问题，我们提出了一个自适应特征交互学习模型，名为EulerNet，在该模型中，通过根据欧拉公式进行空间映射在复杂向量空间中学习特征交互。

    Learning effective high-order feature interactions is very crucial in the CTR prediction task. However, it is very time-consuming to calculate high-order feature interactions with massive features in online e-commerce platforms. Most existing methods manually design a maximal order and further filter out the useless interactions from them. Although they reduce the high computational costs caused by the exponential growth of high-order feature combinations, they still suffer from the degradation of model capability due to the suboptimal learning of the restricted feature orders. The solution to maintain the model capability and meanwhile keep it efficient is a technical challenge, which has not been adequately addressed. To address this issue, we propose an adaptive feature interaction learning model, named as EulerNet, in which the feature interactions are learned in a complex vector space by conducting space mapping according to Euler's formula. EulerNet converts the exponential power
    
[^8]: E Pluribus Unum：关于推荐系统多目标评估的指南

    E Pluribus Unum: Guidelines on Multi-Objective Evaluation of Recommender Systems. (arXiv:2304.10621v1 [cs.IR])

    [http://arxiv.org/abs/2304.10621](http://arxiv.org/abs/2304.10621)

    本论文介绍了一种基于第一原理的多目标模型选择方法，并提出了一组指南，旨在解决推荐系统多目标评估中的平衡多个绩效指标的挑战。

    

    目前，推荐系统的评估主要还是以准确性为主，其他方面的因素，比如多样性、长期用户留存和公平性，往往被忽略。而且，协调多个绩效指标本质上是不确定的，这给寻求全面评估推荐系统的人造成了难题。EvalRS 2022是第一个实践性的数据挑战活动，围绕多目标评估而设计，提供了许多对于平衡多个绩效指标的要求和挑战的洞见。本文回顾了EvalRS 2022，并阐述了重要的理解，制定了一种基于第一原理的多目标模型选择方法，并概述了进行多目标评估挑战的一组指南，具有潜在的适用性，能够解决实际部署中对竞争模型进行全面评估的问题。

    Recommender Systems today are still mostly evaluated in terms of accuracy, with other aspects beyond the immediate relevance of recommendations, such as diversity, long-term user retention and fairness, often taking a back seat. Moreover, reconciling multiple performance perspectives is by definition indeterminate, presenting a stumbling block to those in the pursuit of rounded evaluation of Recommender Systems. EvalRS 2022 -- a data challenge designed around Multi-Objective Evaluation -- was a first practical endeavour, providing many insights into the requirements and challenges of balancing multiple objectives in evaluation. In this work, we reflect on EvalRS 2022 and expound upon crucial learnings to formulate a first-principles approach toward Multi-Objective model selection, and outline a set of guidelines for carrying out a Multi-Objective Evaluation challenge, with potential applicability to the problem of rounded evaluation of competing models in real-world deployments.
    
[^9]: MATURE-HEALTH: MAndatory FeaTURE选择的健康推荐系统

    MATURE-HEALTH: HEALTH Recommender System for MAndatory FeaTURE choices. (arXiv:2304.09099v1 [cs.IR])

    [http://arxiv.org/abs/2304.09099](http://arxiv.org/abs/2304.09099)

    该论文提出和实施了一个名为MATURE-HEALTH的健康推荐系统，该系统能够预测电解质不平衡并推荐营养平衡的食物，从而增加早期检测疾病的机会并防止健康进一步恶化。

    

    平衡电解质对于人体器官的适当功能至关重要和必不可少，因为电解质失衡可能是潜在病理生理学发展的指示。高效监测电解质失衡不仅可以增加疾病早期检测的机会，而且可以通过严格遵循营养控制饮食以平衡电解质从而防止健康进一步恶化。本研究提出并实施了一个推荐系统MATURE Health，该系统预测血液中必需电解质和其他物质的不平衡，然后推荐含有平衡营养的食物，以避免电解质不平衡的发生。该模型考虑到用户最近的实验室结果和每日食物摄入量来预测电解质不平衡。MATURE Health依赖于MATURE Food算法推荐食物，后者仅推荐那些

    Balancing electrolytes is utmost important and essential for appropriate functioning of organs in human body as electrolytes imbalance can be an indication of the development of underlying pathophysiology. Efficient monitoring of electrolytes imbalance not only can increase the chances of early detection of disease, but also prevents the further deterioration of the health by strictly following nutrient controlled diet for balancing the electrolytes post disease detection. In this research, a recommender system MATURE Health is proposed and implemented, which predicts the imbalance of mandatory electrolytes and other substances presented in blood and recommends the food items with the balanced nutrients to avoid occurrence of the electrolytes imbalance. The proposed model takes user most recent laboratory results and daily food intake into account to predict the electrolytes imbalance. MATURE Health relies on MATURE Food algorithm to recommend food items as latter recommends only those
    
[^10]: AI对抗AI：在社交媒体上打击机器生成的虚假餐厅评论

    Combat AI With AI: Counteract Machine-Generated Fake Restaurant Reviews on Social Media. (arXiv:2302.07731v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.07731](http://arxiv.org/abs/2302.07731)

    本文针对机器生成的虚假评论提出了一种用高质量餐厅评论生成虚假评论并微调GPT输出检测器的方法，该方法预测虚假评论的性能优于现有解决方案。同时，我们还探索了预测非精英评论的模型，并在几个维度上对这些评论进行分析，此类机器生成的虚假评论是社交媒体平台面临的持续挑战。

    

    最近生成模型（如GPT）的发展使得以更低的成本制造出难以区分的虚假顾客评论，从而对社交媒体平台检测这些机器生成的虚假评论造成挑战。本文提出利用Yelp验证的高质量的精英餐厅评论来生成OpenAI GPT评论生成器的虚假评论，并最终微调GPT输出检测器来预测明显优于现有解决方案的虚假评论。我们进一步将模型应用于预测非精英评论，并在几个维度（如评论、用户和餐厅特征以及写作风格）上识别模式。我们展示了社交媒体平台正在不断面临机器生成的虚假评论的挑战，尽管他们可能实施检测系统以过滤出可疑的评论。

    Recent advances in generative models such as GPT may be used to fabricate indistinguishable fake customer reviews at a much lower cost, thus posing challenges for social media platforms to detect these machine-generated fake reviews. We propose to leverage the high-quality elite restaurant reviews verified by Yelp to generate fake reviews from the OpenAI GPT review creator and ultimately fine-tune a GPT output detector to predict fake reviews that significantly outperform existing solutions. We further apply the model to predict non-elite reviews and identify the patterns across several dimensions, such as review, user and restaurant characteristics, and writing style. We show that social media platforms are continuously challenged by machine-generated fake reviews, although they may implement detection systems to filter out suspicious reviews.
    

