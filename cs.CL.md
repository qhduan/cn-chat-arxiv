# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789) | FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用 |
| [^2] | [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT).](http://arxiv.org/abs/2307.01225) | 通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。 |

# 详细

[^1]: FlexLLM：一种用于共同提供大型语言模型推理和参数高效微调的系统

    FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2402.18789](https://arxiv.org/abs/2402.18789)

    FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用

    

    Parameter-efficient finetuning（PEFT）是一种广泛使用的技术，用于为不同任务调整大型语言模型。通常，服务提供商会为用户创建单独的系统，以执行PEFT模型微调和推理任务。这是因为现有系统无法处理包含推理和PEFT微调请求混合的工作负载。因此，共享的GPU资源利用不足，导致效率低下。为解决这一问题，我们提出了FlexLLM，这是第一个可以在同一迭代中为推理和参数高效微调请求提供服务的系统。我们的系统利用这两个任务的互补性质，并利用共享的GPU资源来共同运行它们，使用一种称为共同提供的方法。为实现这一目标，FlexLLM引入了一种新颖的标记级微调机制，将序列的微调计算分解为更小的标记级计算，并使用依赖并行化。

    arXiv:2402.18789v1 Announce Type: cross  Abstract: Parameter-efficient finetuning (PEFT) is a widely used technique to adapt large language models for different tasks. Service providers typically create separate systems for users to perform PEFT model finetuning and inference tasks. This is because existing systems cannot handle workloads that include a mix of inference and PEFT finetuning requests. As a result, shared GPU resources are underutilized, leading to inefficiencies. To address this problem, we present FlexLLM, the first system that can serve inference and parameter-efficient finetuning requests in the same iteration. Our system leverages the complementary nature of these two tasks and utilizes shared GPU resources to run them jointly, using a method called co-serving. To achieve this, FlexLLM introduces a novel token-level finetuning mechanism, which breaks down the finetuning computation of a sequence into smaller token-level computations and uses dependent parallelization
    
[^2]: 解释性和透明性驱动的文本对抗示例的检测与转换（IT-DT）

    Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT). (arXiv:2307.01225v1 [cs.CL])

    [http://arxiv.org/abs/2307.01225](http://arxiv.org/abs/2307.01225)

    通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。

    

    基于Transformer的文本分类器如BERT、Roberta、T5和GPT-3在自然语言处理方面展示了令人印象深刻的性能。然而，它们对于对抗性示例的脆弱性提出了安全风险。现有的防御方法缺乏解释性，很难理解对抗性分类并识别模型的漏洞。为了解决这个问题，我们提出了解释性和透明性驱动的检测与转换（IT-DT）框架。它专注于在检测和转换文本对抗示例时的解释性和透明性。IT-DT利用注意力图、集成梯度和模型反馈等技术进行解释性检测。这有助于识别对对抗性分类有贡献的显著特征和扰动词语。在转换阶段，IT-DT利用预训练的嵌入和模型反馈来生成扰动词语的最佳替代。通过找到合适的替换，我们的目标是将对抗性示例转换为正常示例。

    Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into
    

