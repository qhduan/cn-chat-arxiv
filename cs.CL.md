# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks](https://arxiv.org/abs/2402.05948) | DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。 |
| [^2] | [Prompt Injection attack against LLM-integrated Applications.](http://arxiv.org/abs/2306.05499) | 本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。 |

# 详细

[^1]: DE$^3$-BERT: 基于原型网络的增强距离早期停止方法，用于BERT

    DE$^3$-BERT: Distance-Enhanced Early Exiting for BERT based on Prototypical Networks

    [https://arxiv.org/abs/2402.05948](https://arxiv.org/abs/2402.05948)

    DE$^3$-BERT是一种基于原型网络和距离度量的增强距离早期停止框架，用于提高BERT等预训练语言模型的推断速度和准确性。

    

    早期停止方法通过动态调整执行的层数，提高了像BERT这样的预训练语言模型的推断速度。然而，大多数早期停止方法仅考虑了来自单个测试样本的局部信息来确定早期停止的指标，而未利用样本群体提供的全局信息。这导致对预测正确性的估计不够准确，从而产生错误的早期停止决策。为了弥合这个差距，我们探索了有效结合局部和全局信息以确保可靠的早期停止的必要性。为此，我们利用原型网络学习类别原型，并设计了样本和类别原型之间的距离度量。这使我们能够利用全局信息来估计早期预测的正确性。基于此，我们提出了一种新颖的DE$^3$-BERT增强距离早期停止框架。

    Early exiting has demonstrated its effectiveness in accelerating the inference of pre-trained language models like BERT by dynamically adjusting the number of layers executed. However, most existing early exiting methods only consider local information from an individual test sample to determine their exiting indicators, failing to leverage the global information offered by sample population. This leads to suboptimal estimation of prediction correctness, resulting in erroneous exiting decisions. To bridge the gap, we explore the necessity of effectively combining both local and global information to ensure reliable early exiting during inference. Purposefully, we leverage prototypical networks to learn class prototypes and devise a distance metric between samples and class prototypes. This enables us to utilize global information for estimating the correctness of early predictions. On this basis, we propose a novel Distance-Enhanced Early Exiting framework for BERT (DE$^3$-BERT). DE$^3
    
[^2]: LLM集成应用中的提示注入攻击研究

    Prompt Injection attack against LLM-integrated Applications. (arXiv:2306.05499v1 [cs.CR])

    [http://arxiv.org/abs/2306.05499](http://arxiv.org/abs/2306.05499)

    本研究分析了LLM集成应用中的提示注入攻击的复杂性和影响，提出了一种新颖的黑盒提示注入攻击技术HouYi，并揭示了应用程序提示机制中以前未知和严重低估的漏洞。我们的研究呼吁进一步开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    

    大语言模型(LLM)因其卓越的语言理解和生成能力而在它们周围刺激了一个充满活力的应用生态系统。然而，它们在各种服务中的广泛融合带来了重大的安全风险。本研究将解构实际LLM集成应用中的提示注入攻击的复杂性和影响。最初，我们对十个商业应用程序进行了探索性分析，突出了目前攻击策略在实践中的约束条件。受这些限制的启发，我们随后制定了HouYi，一种新颖的黑盒提示注入攻击技术，它借鉴了传统的Web注入攻击。HouYi分为三个关键元素: 一个无缝集成的预构建提示、一个注入提示诱导上下文分区以及一个恶意载荷，旨在实现攻击目标。利用HouYi，我们揭示了应用程序提示机制中以前未知和严重低估的漏洞，并演示了绕过最先进的检测机制的可行性。我们的研究呼吁进一步研究开发全面的防御措施，以抵御LLM集成应用中的提示注入攻击。

    Large Language Models (LLMs), renowned for their superior proficiency in language comprehension and generation, stimulate a vibrant ecosystem of applications around them. However, their extensive assimilation into various services introduces significant security risks. This study deconstructs the complexities and implications of prompt injection attacks on actual LLM-integrated applications. Initially, we conduct an exploratory analysis on ten commercial applications, highlighting the constraints of current attack strategies in practice. Prompted by these limitations, we subsequently formulate HouYi, a novel black-box prompt injection attack technique, which draws inspiration from traditional web injection attacks. HouYi is compartmentalized into three crucial elements: a seamlessly-incorporated pre-constructed prompt, an injection prompt inducing context partition, and a malicious payload designed to fulfill the attack objectives. Leveraging HouYi, we unveil previously unknown and sev
    

