# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PRSA: Prompt Reverse Stealing Attacks against Large Language Models](https://arxiv.org/abs/2402.19200) | 本文提出了针对商业LLMs的提示反窃取攻击框架PRSA，通过分析输入-输出对的关键特征实现攻击。 |
| [^2] | [Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance](https://arxiv.org/abs/2402.08680) | 本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。 |
| [^3] | [Weak-to-Strong Jailbreaking on Large Language Models](https://arxiv.org/abs/2401.17256) | 通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。 |

# 详细

[^1]: PRSA：大型语言模型的提示反盗窃攻击

    PRSA: Prompt Reverse Stealing Attacks against Large Language Models

    [https://arxiv.org/abs/2402.19200](https://arxiv.org/abs/2402.19200)

    本文提出了针对商业LLMs的提示反窃取攻击框架PRSA，通过分析输入-输出对的关键特征实现攻击。

    

    提示作为重要的知识产权，使得大型语言模型（LLMs）能够执行特定任务而无需微调，突显了它们不断增长的重要性。随着基于提示的服务的崛起，如提示市场和LLM应用程序，提供者经常通过输入-输出示例展示提示的能力，以吸引用户。然而，这种范式提出了一个关键的安全问题：暴露输入-输出对是否会对潜在提示泄漏构成风险，侵犯开发者的知识产权？就我们所知，这个问题还没有得到全面探讨。为了弥补这一空白，在本文中，我们进行了首次深入探讨，并提出了一个针对商业LLMs的提示反窃取攻击框架，即PRSA。PRSA的主要思想是通过分析输入-输出对的关键特征，我们模仿并g

    arXiv:2402.19200v1 Announce Type: cross  Abstract: Prompt, recognized as crucial intellectual property, enables large language models (LLMs) to perform specific tasks without the need of fine-tuning, underscoring their escalating importance. With the rise of prompt-based services, such as prompt marketplaces and LLM applications, providers often display prompts' capabilities through input-output examples to attract users. However, this paradigm raises a pivotal security concern: does the exposure of input-output pairs pose the risk of potential prompt leakage, infringing on the intellectual property rights of the developers? To our knowledge, this problem still has not been comprehensively explored yet. To remedy this gap, in this paper, we perform the first in depth exploration and propose a novel attack framework for reverse-stealing prompts against commercial LLMs, namely PRSA. The main idea of PRSA is that by analyzing the critical features of the input-output pairs, we mimic and g
    
[^2]: 通过无分类器引导来减轻大型视觉语言模型中的物体幻觉

    Mitigating Object Hallucination in Large Vision-Language Models via Classifier-Free Guidance

    [https://arxiv.org/abs/2402.08680](https://arxiv.org/abs/2402.08680)

    本文介绍了一种名为MARINE的框架，用于通过无分类器引导来减少大型视觉语言模型的物体幻觉。该框架无需训练或API访问，并通过集成视觉模型和引入额外的物体基础特征来提高模型的生成精确性和效率。

    

    大型视觉语言模型（LVLM）的进展越来越突出了它们在图像中产生虚假物体的严重问题。为了解决这个问题，先前的研究着重于使用特殊策划的数据集或强大的LLM（例如GPT-3.5）来纠正LVLM的输出。然而，这些方法要求昂贵的训练/微调或API访问先进的LLM来在生成后纠正模型的输出。在本文中，我们通过引入一个名为通过无分类器引导缓解幻觉的框架（MARINE）来解决这个挑战，该框架既无需训练也无需API访问，可以在生成过程中有效地减少物体幻觉。具体而言，MARINE通过集成现有的开源视觉模型丰富LVLM的视觉语境，并使用无分类器引导来整合额外的物体基础特征，以提高LVLM生成的精确性。

    The advancement of Large Vision-Language Models (LVLMs) has increasingly highlighted the critical issue of their tendency to hallucinate non-existing objects in the images. To address this issue, previous works focused on using specially curated datasets or powerful LLMs (e.g., GPT-3.5) to rectify the outputs of LVLMs. However, these approaches require either expensive training/fine-tuning or API access to advanced LLMs to correct the model's output post-generation. In this paper, we tackle this challenge by introducing a framework called Mitigating hallucinAtion via classifieR-Free guIdaNcE (MARINE), which is both training-free and API-free, and can effectively and efficiently reduce object hallucinations during the generation process. Specifically, MARINE enriches the visual context of LVLMs by integrating existing open-source vision models, and employs classifier-free guidance to incorporate the additional object grounding features to improve the precision of LVLMs' generations. Thr
    
[^3]: 大规模语言模型的弱到强破解

    Weak-to-Strong Jailbreaking on Large Language Models

    [https://arxiv.org/abs/2401.17256](https://arxiv.org/abs/2401.17256)

    通过弱到强破解攻击，对手可以利用较小的不安全/对齐LLMs指导对显著较大的对齐LLMs进行破解，与解码较大的LLMs相比，其计算和延迟成本较小。

    

    尽管已经付出了大量努力来对齐大规模语言模型（LLMs），但红队测试报告表明，这些经过精心对齐的LLMs仍然可以通过对抗性提示、调优或解码进行破解。在调查对齐LLMs的破解漏洞时，我们观察到破解和对齐模型的解码分布仅在初始生成中存在差异。这一观察结果激发了我们提出的弱到强破解攻击，敌对方可以利用较小的不安全/对齐LLMs（例如7B）指导对显著较大的对齐LLMs（例如70B）进行破解。要进行破解，只需额外解码两个较小的LLMs一次，与解码较大的LLMs相比，其计算和延迟成本较小。通过在三个不同组织的五个模型上进行实验，我们证明了该攻击的有效性。我们的研究揭示了一种以前未注意到但高效的破解方式，

    Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, expo
    

