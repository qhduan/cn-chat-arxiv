# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A](https://arxiv.org/abs/2402.13213) | 多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。 |
| [^2] | [Knowledge Graph Enhanced Aspect-Level Sentiment Analysis.](http://arxiv.org/abs/2312.10048) | 本文提出了一种知识图谱增强的方面级情感分析方法，通过使用BERT模型和知识图谱的同义词数据，在解决上下文特定词义的挑战上取得了优越的性能。 |
| [^3] | [From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI.](http://arxiv.org/abs/2310.17721) | 这项研究探索了使用生成型人工智能工具帮助投资者揭示企业风险的价值，通过从收益电话的上下文中生成风险摘要和评估，这些基于GPT的度量具有显著的信息内容，能够预测企业层面波动性和投资创新选择。此外，生成型人工智能还能有效检测新兴风险，并且这些度量在股权市场中起到定价作用。 |
| [^4] | [MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data.](http://arxiv.org/abs/2304.08247) | MedAlpaca是一个开源的医疗会话式人工智能模型和训练数据集合，旨在通过细化调整预训练语言模型来改善医疗工作流程和医生认证考试的表现。 |

# 详细

[^1]: 软最大概率（大部分时候）在多项选择问答任务中预测大型语言模型的正确性

    Softmax Probabilities (Mostly) Predict Large Language Model Correctness on Multiple-Choice Q&A

    [https://arxiv.org/abs/2402.13213](https://arxiv.org/abs/2402.13213)

    多项选择问答任务中，基于最大softmax概率（MSPs）的模型预测方法有助于提高大型语言模型（LLMs）的正确性，我们提出了一种根据MSP有选择地弃权的策略以提高性能。

    

    尽管大型语言模型（LLMs）在许多任务上表现出色，但过度自信仍然是一个问题。我们假设在多项选择问答任务中，错误答案将与最大softmax概率（MSPs）较小相关，相比之下正确答案较大。我们在十个开源LLMs和五个数据集上全面评估了这一假设，在表现良好的原始问答任务中发现了对我们假设的强有力证据。对于表现最佳的六个LLMs，从MSP导出的AUROC在59/60个实例中都优于随机机会，p < 10^{-4}。在这六个LLMs中，平均AUROC范围在60%至69%之间。利用这些发现，我们提出了一个带有弃权选项的多项选择问答任务，并展示通过根据初始模型响应的MSP有选择地弃权可以提高性能。我们还用预softmax logits而不是softmax进行了相同的实验。

    arXiv:2402.13213v1 Announce Type: cross  Abstract: Although large language models (LLMs) perform impressively on many tasks, overconfidence remains a problem. We hypothesized that on multiple-choice Q&A tasks, wrong answers would be associated with smaller maximum softmax probabilities (MSPs) compared to correct answers. We comprehensively evaluate this hypothesis on ten open-source LLMs and five datasets, and find strong evidence for our hypothesis among models which perform well on the original Q&A task. For the six LLMs with the best Q&A performance, the AUROC derived from the MSP was better than random chance with p < 10^{-4} in 59/60 instances. Among those six LLMs, the average AUROC ranged from 60% to 69%. Leveraging these findings, we propose a multiple-choice Q&A task with an option to abstain and show that performance can be improved by selectively abstaining based on the MSP of the initial model response. We also run the same experiments with pre-softmax logits instead of sof
    
[^2]: 知识图谱增强的方面级情感分析

    Knowledge Graph Enhanced Aspect-Level Sentiment Analysis. (arXiv:2312.10048v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.10048](http://arxiv.org/abs/2312.10048)

    本文提出了一种知识图谱增强的方面级情感分析方法，通过使用BERT模型和知识图谱的同义词数据，在解决上下文特定词义的挑战上取得了优越的性能。

    

    本文提出了一种新颖的方法，通过解决上下文特定词义的挑战，增强情感分析。它将BERT模型的优势与基于知识图谱的同义词数据相结合。这种协同作用利用动态注意机制来构建一个知识驱动的状态向量。为了对特定方面链接的情感进行分类，该方法构建了一个集成了位置数据的记忆库。然后使用DCGRU分析数据，以确定与特定方面术语相关的情感特征。在三个广泛使用的数据集上的实验表明，我们的方法在情感分类中具有卓越的性能。

    In this paper, we propose a novel method to enhance sentiment analysis by addressing the challenge of context-specific word meanings. It combines the advantages of a BERT model with a knowledge graph based synonym data. This synergy leverages a dynamic attention mechanism to develop a knowledge-driven state vector. For classifying sentiments linked to specific aspects, the approach constructs a memory bank integrating positional data. The data are then analyzed using a DCGRU to pinpoint sentiment characteristics related to specific aspect terms. Experiments on three widely used datasets demonstrate the superior performance of our method in sentiment classification.
    
[^3]: 从讲话文本到洞察力：利用生成型人工智能揭示企业风险

    From Transcripts to Insights: Uncovering Corporate Risks Using Generative AI. (arXiv:2310.17721v1 [econ.GN])

    [http://arxiv.org/abs/2310.17721](http://arxiv.org/abs/2310.17721)

    这项研究探索了使用生成型人工智能工具帮助投资者揭示企业风险的价值，通过从收益电话的上下文中生成风险摘要和评估，这些基于GPT的度量具有显著的信息内容，能够预测企业层面波动性和投资创新选择。此外，生成型人工智能还能有效检测新兴风险，并且这些度量在股权市场中起到定价作用。

    

    我们探索了使用ChatGPT等生成型人工智能工具帮助投资者揭示企业风险维度的价值。我们开发并验证了政治、气候和人工智能相关风险的企业层面风险敞口度量。使用GPT 3.5模型从收益电话的背景提供的上下文生成风险摘要和评估，我们发现基于GPT的度量具有显著的信息内容，并在预测（异常）企业层面波动性和企业的选择（如投资和创新）方面优于现有的风险度量。重要的是，风险评估中的信息优于风险摘要，这证明了通用人工智能知识的价值。我们还发现，生成型人工智能对于发现新兴风险（如近几个季度飙升的人工智能风险）非常有效。我们的度量在GPT的训练窗口内外表现良好，并且在股权市场中定价。综上所述，基于人工智能的风险测量方法提供了有用的洞察。

    We explore the value of generative AI tools, such as ChatGPT, in helping investors uncover dimensions of corporate risk. We develop and validate firm-level measures of risk exposure to political, climate, and AI-related risks. Using the GPT 3.5 model to generate risk summaries and assessments from the context provided by earnings call transcripts, we show that GPT-based measures possess significant information content and outperform the existing risk measures in predicting (abnormal) firm-level volatility and firms' choices such as investment and innovation. Importantly, information in risk assessments dominates that in risk summaries, establishing the value of general AI knowledge. We also find that generative AI is effective at detecting emerging risks, such as AI risk, which has soared in recent quarters. Our measures perform well both within and outside the GPT's training window and are priced in equity markets. Taken together, an AI-based approach to risk measurement provides usef
    
[^4]: MedAlpaca -- 一个开源的医疗会话式人工智能模型和训练数据集合

    MedAlpaca -- An Open-Source Collection of Medical Conversational AI Models and Training Data. (arXiv:2304.08247v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.08247](http://arxiv.org/abs/2304.08247)

    MedAlpaca是一个开源的医疗会话式人工智能模型和训练数据集合，旨在通过细化调整预训练语言模型来改善医疗工作流程和医生认证考试的表现。

    

    随着OpenAI的GPT系列等大型语言模型的不断发展，我们见证了人工智能在越来越广泛的领域中的应用出现。在医学领域，这些语言模型在改善医疗工作流程、诊断、患者护理和教育方面具有相当大的潜力。然而，迫切需要开源模型，以在本地部署以保护患者隐私。在我们的工作中，我们提出了一个创新的数据集，其中包含超过16万条数据，专门为了对语言模型进行细化调整以实现有效的医疗应用。我们研究了在公开可访问的预训练语言模型上对这些数据集进行细化调整的影响，并随后通过比较仅使用预训练模型与细化调整模型在未来医生必须通过的考试中的表现来展示其性能。

    As large language models (LLMs) like OpenAI's GPT series continue to make strides, we witness the emergence of artificial intelligence applications in an ever-expanding range of fields. In medicine, these LLMs hold considerable promise for improving medical workflows, diagnostics, patient care, and education. Yet, there is an urgent need for open-source models that can be deployed on-premises to safeguard patient privacy. In our work, we present an innovative dataset consisting of over 160,000 entries, specifically crafted to fine-tune LLMs for effective medical applications. We investigate the impact of fine-tuning these datasets on publicly accessible pre-trained LLMs, and subsequently, we juxtapose the performance of pre-trained-only models against the fine-tuned models concerning the examinations that future medical doctors must pass to achieve certification.
    

