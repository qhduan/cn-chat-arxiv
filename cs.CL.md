# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models Are Unconscious of Unreasonability in Math Problems](https://arxiv.org/abs/2403.19346) | 本文研究了大型语言模型在解决数学问题中对不合理性的反应，设计了不合理数学问题基准以及关键计算和结论提示模板，提升了它们在错误检测和修正方面的能力。 |
| [^2] | [Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models.](http://arxiv.org/abs/2308.15022) | 递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。 |

# 详细

[^1]: 大型语言模型在数学问题中对不合理性毫无意识

    Large Language Models Are Unconscious of Unreasonability in Math Problems

    [https://arxiv.org/abs/2403.19346](https://arxiv.org/abs/2403.19346)

    本文研究了大型语言模型在解决数学问题中对不合理性的反应，设计了不合理数学问题基准以及关键计算和结论提示模板，提升了它们在错误检测和修正方面的能力。

    

    大型语言模型(LLMs)展示了在解决数学问题方面的巨大能力。然而，当给出包含不合理错误的问题时，它们倾向于产生幻觉。在本文中，我们研究了LLMs在面对不合理数学问题时的行为，并进一步探讨了它们解决这些问题的潜力。首先，我们构建了不合理数学问题(UMP)基准来检查LLMs的错误检测能力。实验证明，LLMs能够检测到不合理错误，但仍然在生成非幻觉内容方面失败。为了改善它们的错误检测和修正能力，我们进一步设计了一种称为关键计算和结论(CCC)的战略提示模板。通过CCC，LLMs可以更好地自我评估并检测数学问题中的不合理错误，使它们在实际应用场景中更可靠和安全。

    arXiv:2403.19346v1 Announce Type: new  Abstract: Large language models (LLMs) demonstrate substantial capabilities in solving math problems. However, they tend to produce hallucinations when given questions containing unreasonable errors. In this paper, we study the behavior of LLMs when faced with unreasonable math problems and further explore their potential to address these problems. First, we construct the Unreasonable Math Problem (UMP) benchmark to examine the error detection ability of LLMs. Experiments show that LLMs are able to detect unreasonable errors, but still fail in generating non-hallucinatory content. In order to improve their ability of error detection and correction, we further design a strategic prompt template called Critical Calculation and Conclusion(CCC). With CCC, LLMs can better self-evaluate and detect unreasonable errors in math questions, making them more reliable and safe in practical application scenarios.
    
[^2]: 递归总结在大型语言模型中实现长期对话记忆

    Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models. (arXiv:2308.15022v1 [cs.CL])

    [http://arxiv.org/abs/2308.15022](http://arxiv.org/abs/2308.15022)

    递归总结在大型语言模型中实现长期对话记忆，可以提高对话系统在长对话中记忆重要信息的能力。

    

    大多数开放领域的对话系统在长期对话中容易遗忘重要信息。现有方法通常训练特定的检索器或总结器从过去获取关键信息，这需要耗费时间且高度依赖标记数据的质量。为了缓解这个问题，我们提出使用大型语言模型（LLMs）递归生成总结/记忆，以增强长期记忆能力。具体而言，我们的方法首先刺激LLMs记住小对话上下文，然后递归地使用之前的记忆和随后的对话内容产生新的记忆。最后，LLM可以在最新记忆的帮助下轻松生成高度一致的响应。我们使用ChatGPT和text-davinci-003进行评估，对广泛使用的公共数据集进行的实验证明我们的方法在长对话中可以生成更一致的响应。值得注意的是，我们的方法是实现LLM建模的潜在解决方案。

    Most open-domain dialogue systems suffer from forgetting important information, especially in a long-term conversation. Existing works usually train the specific retriever or summarizer to obtain key information from the past, which is time-consuming and highly depends on the quality of labeled data. To alleviate this problem, we propose to recursively generate summaries/ memory using large language models (LLMs) to enhance long-term memory ability. Specifically, our method first stimulates LLMs to memorize small dialogue contexts and then recursively produce new memory using previous memory and following contexts. Finally, the LLM can easily generate a highly consistent response with the help of the latest memory. We evaluate our method using ChatGPT and text-davinci-003, and the experiments on the widely-used public dataset show that our method can generate more consistent responses in a long-context conversation. Notably, our method is a potential solution to enable the LLM to model
    

