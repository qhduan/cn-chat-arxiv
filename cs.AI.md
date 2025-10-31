# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Verification for Object Detection -- IBP IoU](https://arxiv.org/abs/2403.08788) | 介绍了一种针对目标检测模型的新颖区间边界传播（IBP）方法，IBP IoU在确保准确性和稳定性方面表现出色，为更安全和更稳健的机器学习应用做出贡献。 |
| [^2] | [Reward Collapse in Aligning Large Language Models.](http://arxiv.org/abs/2305.17608) | 本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。 |

# 详细

[^1]: 针对目标检测的验证--IBP IoU

    Verification for Object Detection -- IBP IoU

    [https://arxiv.org/abs/2403.08788](https://arxiv.org/abs/2403.08788)

    介绍了一种针对目标检测模型的新颖区间边界传播（IBP）方法，IBP IoU在确保准确性和稳定性方面表现出色，为更安全和更稳健的机器学习应用做出贡献。

    

    我们介绍了一种新颖的区间边界传播（IBP）方法，用于形式验证对象检测模型，特别针对交并比（IoU）度量。该方法已在一个名为IBP IoU的开源代码中实现，与流行的基于抽象解释的验证工具兼容。该验证器在着陆途径跑道检测和手写数字识别案例研究上进行了评估。与基线（Vanilla IBP IoU）的比较突出了IBP IoU在确保准确性和稳定性方面的出色性能，有助于实现更安全和更稳健的机器学习应用。

    arXiv:2403.08788v1 Announce Type: cross  Abstract: We introduce a novel Interval Bound Propagation (IBP) approach for the formal verification of object detection models, specifically targeting the Intersection over Union (IoU) metric. The approach has been implemented in an open source code, named IBP IoU, compatible with popular abstract interpretation based verification tools. The resulting verifier is evaluated on landing approach runway detection and handwritten digit recognition case studies. Comparisons against a baseline (Vanilla IBP IoU) highlight the superior performance of IBP IoU in ensuring accuracy and stability, contributing to more secure and robust machine learning applications.
    
[^2]: 对齐大型语言模型中的奖励塌缩现象

    Reward Collapse in Aligning Large Language Models. (arXiv:2305.17608v1 [cs.LG])

    [http://arxiv.org/abs/2305.17608](http://arxiv.org/abs/2305.17608)

    本文记录了大型语言模型训练中的奖励塌陷现象，导致在训练结束时，不同的提示生成的奖励分布相同。这主要是因为排名的目标函数无法在优化过程中考虑与提示相关的信息。

    

    大型语言模型（LLMs），如ChatGPT和GPT-4，具有非凡的能力，部分原因在于将它们与训练在人类偏好上的奖励模型对齐，这些偏好通常表示为对响应提示的排名。本文记录了奖励塌陷现象，这是一种经验观察，其中基于排名的方法导致在训练的终止阶段生成的完整奖励分布\textit{无论}\textbf{prompt是什么}都是\textit{相同的}。这种结果是不可取的，因为像“写一篇关于你最好的朋友的简短故事”这样的开放式提示应生成完成它们的连续奖励范围，而像“新西兰的首都是什么”这样的特定提示应生成高或低奖励。我们的理论调查表明，奖励塌陷主要是由于基于排名的目标函数在优化过程中未能纳入与提示相关的信息所致。

    The extraordinary capabilities of large language models (LLMs) such as ChatGPT and GPT-4 are in part unleashed by aligning them with reward models that are trained on human preferences, which are often represented as rankings of responses to prompts. In this paper, we document the phenomenon of \textit{reward collapse}, an empirical observation where the prevailing ranking-based approach results in an \textit{identical} reward distribution \textit{regardless} of the prompts during the terminal phase of training. This outcome is undesirable as open-ended prompts like ``write a short story about your best friend'' should yield a continuous range of rewards for their completions, while specific prompts like ``what is the capital of New Zealand'' should generate either high or low rewards. Our theoretical investigation reveals that reward collapse is primarily due to the insufficiency of the ranking-based objective function to incorporate prompt-related information during optimization. Thi
    

