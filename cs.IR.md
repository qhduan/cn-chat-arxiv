# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Repeated Padding as Data Augmentation for Sequential Recommendation](https://arxiv.org/abs/2403.06372) | 本文提出了一种名为"RepPad"的简单而有效的填充方法，旨在充分利用填充空间来提高顺序推荐模型的性能和训练效率。 |
| [^2] | [C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models](https://arxiv.org/abs/2402.03181) | C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。 |

# 详细

[^1]: 重复填充作为顺序推荐的数据增强

    Repeated Padding as Data Augmentation for Sequential Recommendation

    [https://arxiv.org/abs/2403.06372](https://arxiv.org/abs/2403.06372)

    本文提出了一种名为"RepPad"的简单而有效的填充方法，旨在充分利用填充空间来提高顺序推荐模型的性能和训练效率。

    

    顺序推荐旨在根据用户的历史互动提供个性化建议。在训练顺序模型时，填充是一种被广泛采用的技术，主要原因有两个：1）绝大多数模型只能处理固定长度的序列；2）基于批处理的训练需要确保每个批次中的序列具有相同的长度。通常使用特殊值0作为填充内容，不包含实际信息并在模型计算中被忽略。这种常识填充策略引出了一个以前从未探讨过的问题：我们能否通过填充其他内容充分利用这一闲置输入空间，进一步提高模型性能和训练效率？ 在本文中，我们提出了一种简单而有效的填充方法，名为RepPad (重复填充)。

    arXiv:2403.06372v1 Announce Type: new  Abstract: Sequential recommendation aims to provide users with personalized suggestions based on their historical interactions. When training sequential models, padding is a widely adopted technique for two main reasons: 1) The vast majority of models can only handle fixed-length sequences; 2) Batching-based training needs to ensure that the sequences in each batch have the same length. The special value \emph{0} is usually used as the padding content, which does not contain the actual information and is ignored in the model calculations. This common-sense padding strategy leads us to a problem that has never been explored before: \emph{Can we fully utilize this idle input space by padding other content to further improve model performance and training efficiency?}   In this paper, we propose a simple yet effective padding method called \textbf{Rep}eated \textbf{Pad}ding (\textbf{RepPad}). Specifically, we use the original interaction sequences as
    
[^2]: C-RAG: 针对检索增强语言模型的认证生成风险

    C-RAG: Certified Generation Risks for Retrieval-Augmented Language Models

    [https://arxiv.org/abs/2402.03181](https://arxiv.org/abs/2402.03181)

    C-RAG是第一个用于认证检索增强语言模型生成风险的框架，通过提供符合风险分析和生成风险的上界，确保生成结果的可信性。

    

    尽管大型语言模型（LLMs）在各种应用中具备令人印象深刻的能力，但它们仍然存在可信度问题，如幻觉和错位。检索增强语言模型（RAG）被提出来增强生成结果的可信性，通过引入外部知识。但是，对于RAG模型的生成风险的理论理解尚未被研究。本文回答了以下问题：1）RAG是否确实能够降低生成风险，2）如何对RAG和传统LLM的生成风险提供可证明的保证，以及3）哪些充分条件使得RAG模型能够降低生成风险。我们提出了C-RAG，第一个用于认证RAG模型生成风险的框架。具体而言，我们为RAG模型提供了符合风险分析，并确保了生成风险的上界，我们称之为符合生成风险。我们还对一般有界风险下的符合生成风险提供了理论保证。

    Despite the impressive capabilities of large language models (LLMs) across diverse applications, they still suffer from trustworthiness issues, such as hallucinations and misalignments. Retrieval-augmented language models (RAG) have been proposed to enhance the credibility of generations by grounding external knowledge, but the theoretical understandings of their generation risks remains unexplored. In this paper, we answer: 1) whether RAG can indeed lead to low generation risks, 2) how to provide provable guarantees on the generation risks of RAG and vanilla LLMs, and 3) what sufficient conditions enable RAG models to reduce generation risks. We propose C-RAG, the first framework to certify generation risks for RAG models. Specifically, we provide conformal risk analysis for RAG models and certify an upper confidence bound of generation risks, which we refer to as conformal generation risk. We also provide theoretical guarantees on conformal generation risks for general bounded risk f
    

