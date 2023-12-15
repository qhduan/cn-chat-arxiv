# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Impedance Leakage Vulnerability and its Utilization in Reverse-engineering Embedded Software.](http://arxiv.org/abs/2310.03175) | 这项研究发现了一种新的安全漏洞——阻抗泄漏，通过利用该漏洞可以从嵌入式设备中提取受保护内存中的软件指令。 |
| [^2] | [Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces.](http://arxiv.org/abs/2308.03443) | 本文提出了一种用于具有大动作空间的离策略评估的双重稳健估计器（MDR）。与现有的基准估计器相比，MDR能够在减小方差的同时保持无偏性，从而提高了估计的准确性。实验结果证实了MDR相对于现有估计器的优越性。 |

# 详细

[^1]: 阻抗泄漏脆弱性及其在逆向工程嵌入式软件中的利用

    Impedance Leakage Vulnerability and its Utilization in Reverse-engineering Embedded Software. (arXiv:2310.03175v1 [cs.CR])

    [http://arxiv.org/abs/2310.03175](http://arxiv.org/abs/2310.03175)

    这项研究发现了一种新的安全漏洞——阻抗泄漏，通过利用该漏洞可以从嵌入式设备中提取受保护内存中的软件指令。

    

    发现新的漏洞和实施安全和隐私措施对于保护系统和数据免受物理攻击至关重要。其中一种漏洞是阻抗，一种设备的固有属性，可以通过意外的侧信道泄露信息，从而带来严重的安全和隐私风险。与传统的漏洞不同，阻抗通常被忽视或仅在研究和设计中以特定频率的固定值来处理。此外，阻抗从未被探索过作为信息泄漏的源头。本文证明了嵌入式设备的阻抗并非恒定，并直接与设备上执行的程序相关。我们将此现象定义为阻抗泄漏，并将其作为一种侧信道从受保护的内存中提取软件指令。我们在ATmega328P微控制器和Artix 7 FPGA上的实验表明，阻抗侧信道

    Discovering new vulnerabilities and implementing security and privacy measures are important to protect systems and data against physical attacks. One such vulnerability is impedance, an inherent property of a device that can be exploited to leak information through an unintended side channel, thereby posing significant security and privacy risks. Unlike traditional vulnerabilities, impedance is often overlooked or narrowly explored, as it is typically treated as a fixed value at a specific frequency in research and design endeavors. Moreover, impedance has never been explored as a source of information leakage. This paper demonstrates that the impedance of an embedded device is not constant and directly relates to the programs executed on the device. We define this phenomenon as impedance leakage and use this as a side channel to extract software instructions from protected memory. Our experiment on the ATmega328P microcontroller and the Artix 7 FPGA indicates that the impedance side 
    
[^2]: 用于具有大动作空间的离策略评估的双重稳健估计器

    Doubly Robust Estimator for Off-Policy Evaluation with Large Action Spaces. (arXiv:2308.03443v1 [stat.ML])

    [http://arxiv.org/abs/2308.03443](http://arxiv.org/abs/2308.03443)

    本文提出了一种用于具有大动作空间的离策略评估的双重稳健估计器（MDR）。与现有的基准估计器相比，MDR能够在减小方差的同时保持无偏性，从而提高了估计的准确性。实验结果证实了MDR相对于现有估计器的优越性。

    

    本文研究了在具有大动作空间的背景下的离策略评估（OPE）。现有的基准估计器存在严重的偏差和方差折衷问题。参数化方法由于很难确定正确的模型而导致偏差，而重要性加权方法由于方差而产生问题。为了克服这些限制，本文提出了基于判别式的不良行为抑制器（MIPS）来通过对动作的嵌入来减小估计器的方差。为了使估计器更准确，我们提出了MIPS的双重稳健估计器——边际化双重稳健（MDR）估计器。理论分析表明，所提出的估计器在比MIPS更弱的假设下是无偏的，同时保持了对IPS的方差减小，这是MIPS的主要优势。经验实验证实了MDR相对于现有估计器的优越性。

    We study Off-Policy Evaluation (OPE) in contextual bandit settings with large action spaces. The benchmark estimators suffer from severe bias and variance tradeoffs. Parametric approaches suffer from bias due to difficulty specifying the correct model, whereas ones with importance weight suffer from variance. To overcome these limitations, Marginalized Inverse Propensity Scoring (MIPS) was proposed to mitigate the estimator's variance via embeddings of an action. To make the estimator more accurate, we propose the doubly robust estimator of MIPS called the Marginalized Doubly Robust (MDR) estimator. Theoretical analysis shows that the proposed estimator is unbiased under weaker assumptions than MIPS while maintaining variance reduction against IPS, which was the main advantage of MIPS. The empirical experiment verifies the supremacy of MDR against existing estimators.
    

