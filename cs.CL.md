# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |
| [^2] | [Theoretical guarantees on the best-of-n alignment policy.](http://arxiv.org/abs/2401.01879) | 该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。 |

# 详细

[^1]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    
[^2]: 关于最佳n对齐策略的理论保证

    Theoretical guarantees on the best-of-n alignment policy. (arXiv:2401.01879v1 [cs.LG])

    [http://arxiv.org/abs/2401.01879](http://arxiv.org/abs/2401.01879)

    该论文研究了对齐生成模型的最佳n对齐策略，并证明了之前文献中的某个分析表达式是错误的。研究者们提出了一个新的KL散度估计方法，并通过实验证明其有效性。

    

    一个简单有效的生成模型对齐方法是最佳n对齐策略，该策略从一个基本策略中抽取n个样本，并根据奖励函数对它们进行排序，选择排名最高的样本。文献中常用的分析表达式声称最佳n对齐策略与基本策略之间的KL散度等于$\log (n) (n-1)/n$。我们证明了该论断的不正确性，并展示了它只是实际KL散度的一个上界。我们还研究了在不同情况下该上界的紧致性。最后，我们提出了一种新的KL散度估计方法，并通过几个例子的实验证明它能提供一个紧致的近似。

    A simple and effective method for the alignment of generative models is the best-of-$n$ policy, where $n$ samples are drawn from a base policy, and ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the base policy is equal to $\log (n) (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes. Finally, we propose a new estimator for the KL divergence and empirically show that it provides a tight approximation through a few examples.
    

