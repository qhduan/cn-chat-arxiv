# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^2] | [Tailoring Mixup to Data using Kernel Warping functions.](http://arxiv.org/abs/2311.01434) | 本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。 |
| [^3] | [From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI.](http://arxiv.org/abs/2310.17721) | 这项研究探索了使用生成型人工智能工具帮助投资者揭示企业风险的价值，通过从收益电话的上下文中生成风险摘要和评估，这些基于GPT的度量具有显著的信息内容，能够预测企业层面波动性和投资创新选择。此外，生成型人工智能还能有效检测新兴风险，并且这些度量在股权市场中起到定价作用。 |
| [^4] | [Bloated Disclosures: Can ChatGPT Help Investors Process Financial Information?.](http://arxiv.org/abs/2306.10224) | 研究发现生成式 AI 工具 ChatGPT 可以更有效地展示股票市场相关信息，提出了信息膨胀指标并证明其与负面的资本市场后果相关，同时展示其在构建针对性总结方面的效果。 |
| [^5] | [MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data.](http://arxiv.org/abs/2304.08247) | MedAlpaca是一个开源的医疗会话式人工智能模型和训练数据集合，旨在通过细化调整预训练语言模型来改善医疗工作流程和医生认证考试的表现。 |
| [^6] | [On the Need and Applicability of Causality for Fair Machine Learning.](http://arxiv.org/abs/2207.04053) | 本论文探讨了因果关系在公平机器学习中的必要性和适用性，强调了非因果预测的社会影响和法律反歧视过程依赖于因果主张。同时讨论了在实际场景中应用因果关系所面临的挑战和限制，并提出了可能的解决方案。 |

# 详细

[^1]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^2]: 通过核扭曲函数定制Mixup数据

    Tailoring Mixup to Data using Kernel Warping functions. (arXiv:2311.01434v1 [cs.LG])

    [http://arxiv.org/abs/2311.01434](http://arxiv.org/abs/2311.01434)

    本研究提出了一种利用核扭曲函数对Mixup数据进行个性化处理的方法，通过动态改变插值系数的概率分布来实现更频繁和更强烈的混合相似数据点。实验证明这种方法不仅提高了模型性能，还提高了模型的校准性。

    

    数据增强是学习高效深度学习模型的重要基础。在所有提出的增强技术中，线性插值训练数据点（也称为Mixup）已被证明在许多应用中非常有效。然而，大多数研究都集中在选择合适的点进行混合，或者应用复杂的非线性插值，而我们则对更相似的点进行更频繁和更强烈的混合感兴趣。为此，我们提出了通过扭曲函数动态改变插值系数的概率分布的方法，取决于要组合的数据点之间的相似性。我们定义了一个高效而灵活的框架来实现这一点，以避免多样性的损失。我们进行了广泛的分类和回归任务实验，结果显示我们提出的方法既提高了模型的性能，又提高了模型的校准性。代码可在https://github.com/ENSTA-U2IS/torch-uncertainty上找到。

    Data augmentation is an essential building block for learning efficient deep learning models. Among all augmentation techniques proposed so far, linear interpolation of training data points, also called mixup, has found to be effective for a large panel of applications. While the majority of works have focused on selecting the right points to mix, or applying complex non-linear interpolation, we are interested in mixing similar points more frequently and strongly than less similar ones. To this end, we propose to dynamically change the underlying distribution of interpolation coefficients through warping functions, depending on the similarity between data points to combine. We define an efficient and flexible framework to do so without losing in diversity. We provide extensive experiments for classification and regression tasks, showing that our proposed method improves both performance and calibration of models. Code available in https://github.com/ENSTA-U2IS/torch-uncertainty
    
[^3]: 从讲话文本到洞察力：利用生成型人工智能揭示企业风险

    From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI. (arXiv:2310.17721v1 [econ.GN])

    [http://arxiv.org/abs/2310.17721](http://arxiv.org/abs/2310.17721)

    这项研究探索了使用生成型人工智能工具帮助投资者揭示企业风险的价值，通过从收益电话的上下文中生成风险摘要和评估，这些基于GPT的度量具有显著的信息内容，能够预测企业层面波动性和投资创新选择。此外，生成型人工智能还能有效检测新兴风险，并且这些度量在股权市场中起到定价作用。

    

    我们探索了使用ChatGPT等生成型人工智能工具帮助投资者揭示企业风险维度的价值。我们开发并验证了政治、气候和人工智能相关风险的企业层面风险敞口度量。使用GPT 3.5模型从收益电话的背景提供的上下文生成风险摘要和评估，我们发现基于GPT的度量具有显著的信息内容，并在预测（异常）企业层面波动性和企业的选择（如投资和创新）方面优于现有的风险度量。重要的是，风险评估中的信息优于风险摘要，这证明了通用人工智能知识的价值。我们还发现，生成型人工智能对于发现新兴风险（如近几个季度飙升的人工智能风险）非常有效。我们的度量在GPT的训练窗口内外表现良好，并且在股权市场中定价。综上所述，基于人工智能的风险测量方法提供了有用的洞察。

    We explore the value of generative AI tools, such as ChatGPT, in helping investors uncover dimensions of corporate risk. We develop and validate firm-level measures of risk exposure to political, climate, and AI-related risks. Using the GPT 3.5 model to generate risk summaries and assessments from the context provided by earnings call transcripts, we show that GPT-based measures possess significant information content and outperform the existing risk measures in predicting (abnormal) firm-level volatility and firms' choices such as investment and innovation. Importantly, information in risk assessments dominates that in risk summaries, establishing the value of general AI knowledge. We also find that generative AI is effective at detecting emerging risks, such as AI risk, which has soared in recent quarters. Our measures perform well both within and outside the GPT's training window and are priced in equity markets. Taken together, an AI-based approach to risk measurement provides usef
    
[^4]: 膨胀的披露：ChatGPT是否能帮助投资者处理财务信息？

    Bloated Disclosures: Can ChatGPT Help Investors Process Financial Information?. (arXiv:2306.10224v1 [econ.GN])

    [http://arxiv.org/abs/2306.10224](http://arxiv.org/abs/2306.10224)

    研究发现生成式 AI 工具 ChatGPT 可以更有效地展示股票市场相关信息，提出了信息膨胀指标并证明其与负面的资本市场后果相关，同时展示其在构建针对性总结方面的效果。

    

    生成式 AI 工具（如 ChatGPT）可以从根本上改变投资者处理信息的方式。我们使用股票市场作为实验室，探究这些工具在总结复杂的公司披露信息时的经济效用。总结摘要明显更短，通常比原始文本缩短超过 70%，而信息内容得到增强。当一份文件具有积极（消极）情感时，其总结变得更积极（消极）。更重要的是，总结对解释股市对披露信息的反应更有效。基于这些发现，我们提出了信息“膨胀”指标。我们显示，膨胀的披露与负面的资本市场后果相关，例如更低的价格有效性和更高的信息不对称性。最后，我们展示了这个模型在构建针对性总结方面的有效性，以确定公司的（非）财务表现和风险。总之，我们的研究结果表明，像 ChatGPT 这样的生成式 AI 工具可以有效地帮助投资者更高效地处理财务信息。

    Generative AI tools such as ChatGPT can fundamentally change the way investors process information. We probe the economic usefulness of these tools in summarizing complex corporate disclosures using the stock market as a laboratory. The unconstrained summaries are dramatically shorter, often by more than 70% compared to the originals, whereas their information content is amplified. When a document has a positive (negative) sentiment, its summary becomes more positive (negative). More importantly, the summaries are more effective at explaining stock market reactions to the disclosed information. Motivated by these findings, we propose a measure of information "bloat." We show that bloated disclosure is associated with adverse capital markets consequences, such as lower price efficiency and higher information asymmetry. Finally, we show that the model is effective at constructing targeted summaries that identify firms' (non-)financial performance and risks. Collectively, our results indi
    
[^5]: MedAlpaca -- 一个开源的医疗会话式人工智能模型和训练数据集合

    MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data. (arXiv:2304.08247v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.08247](http://arxiv.org/abs/2304.08247)

    MedAlpaca是一个开源的医疗会话式人工智能模型和训练数据集合，旨在通过细化调整预训练语言模型来改善医疗工作流程和医生认证考试的表现。

    

    随着OpenAI的GPT系列等大型语言模型的不断发展，我们见证了人工智能在越来越广泛的领域中的应用出现。在医学领域，这些语言模型在改善医疗工作流程、诊断、患者护理和教育方面具有相当大的潜力。然而，迫切需要开源模型，以在本地部署以保护患者隐私。在我们的工作中，我们提出了一个创新的数据集，其中包含超过16万条数据，专门为了对语言模型进行细化调整以实现有效的医疗应用。我们研究了在公开可访问的预训练语言模型上对这些数据集进行细化调整的影响，并随后通过比较仅使用预训练模型与细化调整模型在未来医生必须通过的考试中的表现来展示其性能。

    As large language models (LLMs) like OpenAI's GPT series continue to make strides, we witness the emergence of artificial intelligence applications in an ever-expanding range of fields. In medicine, these LLMs hold considerable promise for improving medical workflows, diagnostics, patient care, and education. Yet, there is an urgent need for open-source models that can be deployed on-premises to safeguard patient privacy. In our work, we present an innovative dataset consisting of over 160,000 entries, specifically crafted to fine-tune LLMs for effective medical applications. We investigate the impact of fine-tuning these datasets on publicly accessible pre-trained LLMs, and subsequently, we juxtapose the performance of pre-trained-only models against the fine-tuned models concerning the examinations that future medical doctors must pass to achieve certification.
    
[^6]: 论公平机器学习中因果关系的必要性和适用性

    On the Need and Applicability of Causality for Fair Machine Learning. (arXiv:2207.04053v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2207.04053](http://arxiv.org/abs/2207.04053)

    本论文探讨了因果关系在公平机器学习中的必要性和适用性，强调了非因果预测的社会影响和法律反歧视过程依赖于因果主张。同时讨论了在实际场景中应用因果关系所面临的挑战和限制，并提出了可能的解决方案。

    

    除了在流行病学、政治和社会科学中的常见应用案例外，事实证明因果关系在评估自动决策的公正性方面十分重要，无论是在法律上还是日常生活中。我们提供了关于为何因果关系对公平性评估尤为重要的论点和示例。特别是，我们指出了非因果预测的社会影响以及依赖因果主张的法律反歧视过程。我们最后讨论了应用因果关系在实际场景中的挑战和局限性，以及可能的解决方案。

    Besides its common use cases in epidemiology, political, and social sciences, causality turns out to be crucial in evaluating the fairness of automated decisions, both in a legal and everyday sense. We provide arguments and examples, of why causality is particularly important for fairness evaluation. In particular, we point out the social impact of non-causal predictions and the legal anti-discrimination process that relies on causal claims. We conclude with a discussion about the challenges and limitations of applying causality in practical scenarios as well as possible solutions.
    

