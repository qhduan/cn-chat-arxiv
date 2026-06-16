# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Neural Networks: A Formulation Via Non-Archimedean Analysis](https://arxiv.org/abs/2402.00094) | 该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。 |
| [^2] | [A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions.](http://arxiv.org/abs/2401.15296) | 本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。 |
| [^3] | [Multi-Grade Deep Learning for Partial Differential Equations with Applications to the Burgers Equation.](http://arxiv.org/abs/2309.07401) | 本文提出了一种多级深度学习方法，用于解决非线性偏微分方程。该方法通过将DNN的学习任务分解为多个堆叠的神经网络，以解决随着网络层数增加而导致的非凸优化问题的复杂度增加的挑战。 |
| [^4] | [Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing.](http://arxiv.org/abs/2308.06035) | 这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。 |

# 详细

[^1]: 深度神经网络: 非阿基米德分析的表述方式

    Deep Neural Networks: A Formulation Via Non-Archimedean Analysis

    [https://arxiv.org/abs/2402.00094](https://arxiv.org/abs/2402.00094)

    该论文引入了一种新的深度神经网络（DNNs）类别，其采用多层树状结构的架构并使用非阿基米德局部域的整数环进行编码。这些DNNs是稳健的对实值函数和实值平方可积函数的普遍逼近器。

    

    我们引入了一种新的深度神经网络（DNNs），采用多层树状结构的架构。这些架构使用非阿基米德局部域的整数环中的数字进行编码。这些环具有自然的层次结构，类似无限根树。这些环上的自然态射使我们能够构建有限的多层架构。新的DNNs是对在所提到的环上定义的实值函数的稳健的普遍逼近器。我们还证明了DNNs也是对在单位区间上定义的实值平方可积函数的稳健的普遍逼近器。

    We introduce a new class of deep neural networks (DNNs) with multilayered tree-like architectures. The architectures are codified using numbers from the ring of integers of non-Archimdean local fields. These rings have a natural hierarchical organization as infinite rooted trees. Natural morphisms on these rings allow us to construct finite multilayered architectures. The new DNNs are robust universal approximators of real-valued functions defined on the mentioned rings. We also show that the DNNs are robust universal approximators of real-valued square-integrable functions defined in the unit interval.
    
[^2]: 基于3D骨架的人员再识别：方法、设计、挑战和未来方向的综述

    A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions. (arXiv:2401.15296v1 [cs.CV])

    [http://arxiv.org/abs/2401.15296](http://arxiv.org/abs/2401.15296)

    本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。

    

    通过3D骨架进行人员再识别是一个重要的新兴研究领域，引起了模式识别社区的极大兴趣。近年来，针对骨架建模和特征学习中突出问题，已经提出了许多具有独特优势的基于3D骨架的人员再识别（SRID）方法。尽管近年来取得了一些进展，但据我们所知，目前还没有对这些研究及其挑战进行综合总结。因此，本文通过对当前SRID方法、模型设计、挑战和未来方向的系统调研，试图填补这一空白。具体而言，我们首先定义了SRID问题，并提出了一个SRID研究的分类体系，总结了常用的基准数据集、常用的模型架构，并对不同方法的特点进行了分析评价。然后，我们详细阐述了SRID模型的设计原则。

    Person re-identification via 3D skeletons is an important emerging research area that triggers great interest in the pattern recognition community. With distinctive advantages for many application scenarios, a great diversity of 3D skeleton based person re-identification (SRID) methods have been proposed in recent years, effectively addressing prominent problems in skeleton modeling and feature learning. Despite recent advances, to the best of our knowledge, little effort has been made to comprehensively summarize these studies and their challenges. In this paper, we attempt to fill this gap by providing a systematic survey on current SRID approaches, model designs, challenges, and future directions. Specifically, we first formulate the SRID problem, and propose a taxonomy of SRID research with a summary of benchmark datasets, commonly-used model architectures, and an analytical review of different methods' characteristics. Then, we elaborate on the design principles of SRID models fro
    
[^3]: 多级深度学习在解决偏微分方程中的应用，以Burgers方程为例

    Multi-Grade Deep Learning for Partial Differential Equations with Applications to the Burgers Equation. (arXiv:2309.07401v1 [math.NA])

    [http://arxiv.org/abs/2309.07401](http://arxiv.org/abs/2309.07401)

    本文提出了一种多级深度学习方法，用于解决非线性偏微分方程。该方法通过将DNN的学习任务分解为多个堆叠的神经网络，以解决随着网络层数增加而导致的非凸优化问题的复杂度增加的挑战。

    

    本文提出了一种多级深度学习方法，用于解决非线性偏微分方程（PDEs）。深度神经网络在解决PDEs方面表现出超强的性能，除了在自然语言处理、计算机视觉和机器人等领域的卓越成功。然而，训练一个非常深的网络往往是一项具有挑战性的任务。随着DNN的层数增加，解决由PDEs的DNN求解结果导致的大规模非凸优化问题变得越来越困难，这可能导致预测准确性的降低而不是增加。为了克服这一挑战，我们提出了一种两阶段多级深度学习（TS-MGDL）方法，将学习DNN的任务分解为一系列堆叠在彼此上方的神经网络。这种方法可以减轻解决具有大量参数的非凸优化问题的复杂度，并学习残差。

    We develop in this paper a multi-grade deep learning method for solving nonlinear partial differential equations (PDEs). Deep neural networks (DNNs) have received super performance in solving PDEs in addition to their outstanding success in areas such as natural language processing, computer vision, and robotics. However, training a very deep network is often a challenging task. As the number of layers of a DNN increases, solving a large-scale non-convex optimization problem that results in the DNN solution of PDEs becomes more and more difficult, which may lead to a decrease rather than an increase in predictive accuracy. To overcome this challenge, we propose a two-stage multi-grade deep learning (TS-MGDL) method that breaks down the task of learning a DNN into several neural networks stacked on top of each other in a staircase-like manner. This approach allows us to mitigate the complexity of solving the non-convex optimization problem with large number of parameters and learn resid
    
[^4]: 多模态大语言模型在预测语言处理期间表现出人类视觉-语言集成的证据

    Evidence of Human-Like Visual-Linguistic Integration in Multimodal Large Language Models During Predictive Language Processing. (arXiv:2308.06035v1 [cs.AI])

    [http://arxiv.org/abs/2308.06035](http://arxiv.org/abs/2308.06035)

    这篇论文研究了多模态大语言模型（mLLMs）在预测语言处理过程中与人类的视觉-语言集成能力是否一致的问题，并通过实验验证了mLLMs的多模态输入方法可以减少认知负荷，提高感知和理解能力。

    

    大语言模型（LLMs）的先进语言处理能力引发了关于它们是否能够复制人类认知过程的争议。LLMs和人类在语言处理方面的一个区别在于，语言输入通常建立在多个知觉模态上，而大多数LLMs仅处理基于文本的信息。多模态基础使人类能够整合视觉背景与语言信息，从而对即将出现的单词的空间施加限制，减少认知负荷，提高感知和理解能力。最近的多模态LLMs（mLLMs）结合了视觉和语言嵌入空间，并使用变压器类型的注意机制进行下一个单词的预测。在多大程度上，基于多模态输入的预测语言处理在mLLMs和人类中吻合？为了回答这个问题，200名被试观看了短的视听剪辑，并估计了即将出现的动词或名词的可预测性。

    The advanced language processing abilities of large language models (LLMs) have stimulated debate over their capacity to replicate human-like cognitive processes. One differentiating factor between language processing in LLMs and humans is that language input is often grounded in more than one perceptual modality, whereas most LLMs process solely text-based information. Multimodal grounding allows humans to integrate - e.g. visual context with linguistic information and thereby place constraints on the space of upcoming words, reducing cognitive load and improving perception and comprehension. Recent multimodal LLMs (mLLMs) combine visual and linguistic embedding spaces with a transformer type attention mechanism for next-word prediction. To what extent does predictive language processing based on multimodal input align in mLLMs and humans? To answer this question, 200 human participants watched short audio-visual clips and estimated the predictability of an upcoming verb or noun. The 
    

