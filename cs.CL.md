# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [IR2: Information Regularization for Information Retrieval](https://arxiv.org/abs/2402.16200) | 介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。 |
| [^2] | [APPLS: Evaluating Evaluation Metrics for Plain Language Summarization](https://arxiv.org/abs/2305.14341) | 本文提出了一个用于评估纯语言摘要的指标测试平台APPLS，并引入了一种新的指标POMME来评估PLS中的文本简化。通过对指标的分析发现，当前的指标未能始终捕捉到简化度。 |
| [^3] | [Will the Prince Get True Love's Kiss? On the Model Sensitivity to Gender Perturbation over Fairytale Texts.](http://arxiv.org/abs/2310.10865) | 该研究旨在通过评估语言模型对性别扰动的鲁棒性，帮助减轻传统童话中的性别偏见，并通过引入反事实性别刻板印象来减轻学习到的偏见。实验结果显示，模型对性别扰动敏感，但在反事实训练后对后续引入的反性别偏见更不敏感。 |

# 详细

[^1]: IR2：信息正则化用于信息检索

    IR2: Information Regularization for Information Retrieval

    [https://arxiv.org/abs/2402.16200](https://arxiv.org/abs/2402.16200)

    介绍了IR2，一种用于在合成数据生成过程中减少过拟合的信息正则化技术，在复杂查询的信息检索任务中表现出优越性能，同时将成本降低高达50%。

    

    有效地在训练数据有限的情况下进行信息检索（IR），特别是对于复杂查询，仍然是一项具有挑战性的任务。本文介绍了IR2，即信息检索的信息正则化，一种用于在合成数据生成过程中减少过拟合的技术。该方法在具有复杂查询特征的三个最近的IR任务上进行了测试：DORIS-MAE、ArguAna和WhatsThatBook。实验结果表明，我们的正则化技术不仅在所考虑的任务上优于先前的合成查询生成方法，而且还能将成本降低高达50％。此外，本文将不同阶段的三种正则化方法——输入、提示和输出进行了分类和探索，每种方法相对于没有正则化的模型均提供了不同程度的性能改进。

    arXiv:2402.16200v1 Announce Type: cross  Abstract: Effective information retrieval (IR) in settings with limited training data, particularly for complex queries, remains a challenging task. This paper introduces IR2, Information Regularization for Information Retrieval, a technique for reducing overfitting during synthetic data generation. This approach, representing a novel application of regularization techniques in synthetic data creation for IR, is tested on three recent IR tasks characterized by complex queries: DORIS-MAE, ArguAna, and WhatsThatBook. Experimental results indicate that our regularization techniques not only outperform previous synthetic query generation methods on the tasks considered but also reduce cost by up to 50%. Furthermore, this paper categorizes and explores three regularization methods at different stages of the query synthesis pipeline-input, prompt, and output-each offering varying degrees of performance improvement compared to models where no regulariz
    
[^2]: APPLS: 评估纯语言摘要的评价指标

    APPLS: Evaluating Evaluation Metrics for Plain Language Summarization

    [https://arxiv.org/abs/2305.14341](https://arxiv.org/abs/2305.14341)

    本文提出了一个用于评估纯语言摘要的指标测试平台APPLS，并引入了一种新的指标POMME来评估PLS中的文本简化。通过对指标的分析发现，当前的指标未能始终捕捉到简化度。

    

    尽管对于纯语言摘要（PLS）的模型有了很大的发展，但评估仍然是一个挑战。PLS缺乏专门的评估指标，由于涉及到独特的转换（例如，添加背景解释，删除专业术语），因此对于文本生成评估指标的适用性尚不清楚。为了解决这些问题，我们的研究提出了一个细致的元评估测试平台APPLS，旨在评估PLS的指标。我们根据先前工作的启发，定义了四个标准上的一组扰动，PLS指标应该捕捉到：信息性、简化度、连贯性和忠实度。使用我们的测试平台对指标进行分析发现，当前的指标未能始终捕捉到简化度。作为回应，我们引入了一种新的指标POMME，旨在评估PLS中文本简化；该指标是根据域内和域外语言模型之间的标准化困惑度差计算得到的。我们演示了POMME的效果，并与其他指标进行了比较。

    While there has been significant development of models for Plain Language Summarization (PLS), evaluation remains a challenge. PLS lacks a dedicated assessment metric, and the suitability of text generation evaluation metrics is unclear due to the unique transformations involved (e.g., adding background explanations, removing specialized terminology). To address these concerns, our study presents a granular meta-evaluation testbed, APPLS, designed to evaluate metrics for PLS. We define a set of perturbations along four criteria inspired by previous work that a PLS metric should capture: informativeness, simplification, coherence, and faithfulness. An analysis of metrics using our testbed reveals that current metrics fail to capture simplification consistently. In response, we introduce POMME, a new metric designed to assess text simplification in PLS; the metric is calculated as the normalized perplexity difference between an in-domain and out-of-domain language model. We demonstrate P
    
[^3]: 王子会得到真爱之吻吗？关于童话文本中性别扰动对模型敏感性的研究

    Will the Prince Get True Love's Kiss? On the Model Sensitivity to Gender Perturbation over Fairytale Texts. (arXiv:2310.10865v1 [cs.CL])

    [http://arxiv.org/abs/2310.10865](http://arxiv.org/abs/2310.10865)

    该研究旨在通过评估语言模型对性别扰动的鲁棒性，帮助减轻传统童话中的性别偏见，并通过引入反事实性别刻板印象来减轻学习到的偏见。实验结果显示，模型对性别扰动敏感，但在反事实训练后对后续引入的反性别偏见更不敏感。

    

    最近的研究显示，传统的童话故事中存在大量有害的性别偏见。为了减轻童话中的性别偏见，本研究旨在评估语言模型学习到的偏见对性别扰动的鲁棒性。具体而言，我们关注童话故事中的问答任务。通过使用反事实数据增强FairytaleQA数据集，我们评估模型对交换性别角色信息的鲁棒性，并在训练时引入反事实性别刻板印象来减轻学习到的偏见。此外，我们还引入了一种新的方法，利用语言模型的庞大词汇量来支持超越童话故事的文本类型。我们的实验结果表明，模型对性别扰动敏感，性能与原始测试集相比显著下降。然而，当首先在反事实的训练数据集上进行微调后，模型对后续引入的反性别偏见更不敏感。

    Recent studies show that traditional fairytales are rife with harmful gender biases. To help mitigate these gender biases in fairytales, this work aims to assess learned biases of language models by evaluating their robustness against gender perturbations. Specifically, we focus on Question Answering (QA) tasks in fairytales. Using counterfactual data augmentation to the FairytaleQA dataset, we evaluate model robustness against swapped gender character information, and then mitigate learned biases by introducing counterfactual gender stereotypes during training time. We additionally introduce a novel approach that utilizes the massive vocabulary of language models to support text genres beyond fairytales. Our experimental results suggest that models are sensitive to gender perturbations, with significant performance drops compared to the original testing set. However, when first fine-tuned on a counterfactual training dataset, models are less sensitive to the later introduced anti-gend
    

