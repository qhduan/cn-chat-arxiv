# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought](https://arxiv.org/abs/2403.05518) | 引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。 |

# 详细

[^1]: 通过偏差增强一致性训练减少链式思维中的偏见推理

    Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought

    [https://arxiv.org/abs/2403.05518](https://arxiv.org/abs/2403.05518)

    引入偏差增强的一致性训练（BCT）可以显著减少链式思维中的偏见推理问题，尤其是通过训练模型在带有和不带有偏置特征的提示下进行一致的推理。

    

    虽然链式思维提示（CoT）有潜力改善语言模型推理的可解释性，但它可能会系统性地歪曲影响模型行为的因素--比如，合理化答案以符合用户意见而不提及此偏见。为了减轻这一偏见推理问题，我们引入了偏差增强的一致性训练（BCT），这是一种无监督的微调方案，旨在训练模型在带有和不带有偏置特征的提示下进行一致的推理。我们构建了一个测试单元，针对七个问答任务测试了九种形式的有偏推理，发现将BCT应用于带有一种偏见的GPT-3.5-Turbo可以将有偏推理的比例在未知任务上降低86%。此外，这个模型推广到其他形式的偏见，平均将未知偏见上的有偏推理减少了37%。由于BCT将未知偏见泛化并且不需要金标签，这种方法可能会有助于

    arXiv:2403.05518v1 Announce Type: cross  Abstract: While chain-of-thought prompting (CoT) has the potential to improve the explainability of language model reasoning, it can systematically misrepresent the factors influencing models' behavior--for example, rationalizing answers in line with a user's opinion without mentioning this bias. To mitigate this biased reasoning problem, we introduce bias-augmented consistency training (BCT), an unsupervised fine-tuning scheme that trains models to give consistent reasoning across prompts with and without biasing features. We construct a suite testing nine forms of biased reasoning on seven question-answering tasks, and find that applying BCT to GPT-3.5-Turbo with one bias reduces the rate of biased reasoning by 86% on held-out tasks. Moreover, this model generalizes to other forms of bias, reducing biased reasoning on held-out biases by an average of 37%. As BCT generalizes to held-out biases and does not require gold labels, this method may h
    

