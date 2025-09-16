# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^2] | [Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models.](http://arxiv.org/abs/2309.01219) | 本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。 |
| [^3] | [Calibration in Deep Learning: A Survey of the State-of-the-Art.](http://arxiv.org/abs/2308.01222) | 本文回顾了深度学习中的校准方法的最新发展，并提供了对其原理的理解。研究表明，现代深度神经网络在预测能力上表现出色，但校准性较差，导致模型预测不可靠。因此，需要一些新的方法来改善模型的校准性。 |
| [^4] | [Explaining Emergent In-Context Learning as Kernel Regression.](http://arxiv.org/abs/2305.12766) | 本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。 |

# 详细

[^1]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^2]: AI海洋中的妖怪之歌：大型语言模型中的幻觉调查

    Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models. (arXiv:2309.01219v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01219](http://arxiv.org/abs/2309.01219)

    本文调查了大型语言模型中幻觉的检测、解释和缓解的最新研究，提出了幻觉现象和评估基准的分类，并讨论了未来研究的潜在方向。

    

    尽管大型语言模型（LLMs）在各种下游任务中展示出了卓越的能力，但人们对其产生幻觉的倾向表示担忧：LLMs有时会生成与用户输入不符、与先前生成的内容相矛盾或与已建立的世界知识不符的内容。这种现象对LLMs在现实场景中的可靠性构成了重大挑战。本文对关于幻觉检测、解释和缓解的最新研究进行了调查，重点探讨了LLMs所面临的独特挑战。我们提出了LLM幻觉现象和评估基准的分类，分析了现有的旨在缓解LLM幻觉的方法，并讨论了未来研究的潜在方向。

    While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.
    
[^3]: 深度学习中的校准：最新研究综述

    Calibration in Deep Learning: A Survey of the State-of-the-Art. (arXiv:2308.01222v1 [cs.LG])

    [http://arxiv.org/abs/2308.01222](http://arxiv.org/abs/2308.01222)

    本文回顾了深度学习中的校准方法的最新发展，并提供了对其原理的理解。研究表明，现代深度神经网络在预测能力上表现出色，但校准性较差，导致模型预测不可靠。因此，需要一些新的方法来改善模型的校准性。

    

    在构建可靠、鲁棒的安全关键应用的人工智能系统中，深度神经模型的校准起着重要作用。最近的研究表明，具有高预测能力的现代神经网络的校准性较差，产生不可靠的模型预测。尽管深度学习模型在各种基准测试中取得了显著的性能，但对模型的校准性和可靠性的研究相对较少。理想的深度模型不仅应具有高预测性能，还应具有良好的校准性。最近提出了一些使用不同机制进行深度模型校准的方法。在本综述中，我们回顾了最新的校准方法，并解释了它们执行模型校准的原理。首先，我们从模型校准的定义开始，解释了模型校准不准确的根本原因。然后，我们介绍了可以衡量模型校准性的关键指标。接下来，我们总结了一些校准方法的方法和实践。

    Calibrating deep neural models plays an important role in building reliable, robust AI systems in safety-critical applications. Recent work has shown that modern neural networks that possess high predictive capability are poorly calibrated and produce unreliable model predictions. Though deep learning models achieve remarkable performance on various benchmarks, the study of model calibration and reliability is relatively underexplored. Ideal deep models should have not only high predictive performance but also be well calibrated. There have been some recent methods proposed to calibrate deep models by using different mechanisms. In this survey, we review the state-of-the-art calibration methods and provide an understanding of their principles for performing model calibration. First, we start with the definition of model calibration and explain the root causes of model miscalibration. Then we introduce the key metrics that can measure this aspect. It is followed by a summary of calibrat
    
[^4]: 将 Emergent In-Context Learning 解释为核回归

    Explaining Emergent In-Context Learning as Kernel Regression. (arXiv:2305.12766v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12766](http://arxiv.org/abs/2305.12766)

    本文研究了为什么在预训练之后，基于Transformer的语言模型能够实现上下文学习，并提出了一种假设，认为LLMs在面对上下文示例时能够通过内部表示模拟核回归。

    

    大型语言模型（LLMs）在迁移学习中引起了一场范式转变。与经典的预训练-微调过程相比，为了将LLMs用于下游预测任务，只需要提供一些示例，即上下文示例，而无需添加或更新现有的模型参数。LLMs的这种上下文学习能力非常有意思，但目前尚不完全了解预训练LLMs如何获得这种能力。本文通过提出一个假设，即当面临上下文示例时，LLMs能够通过内部表示模拟核回归，来研究为何基于Transformer的语言模型能够在预训练通用语料库之后实现上下文学习。具体来说，我们首先证明了上下文提示的贝叶斯推断在渐近情况下可以被理解为核回归 $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$，

    Large language models (LLMs) have initiated a paradigm shift in transfer learning. In contrast to the classic pretraining-then-finetuning procedure, in order to use LLMs for downstream prediction tasks, one only needs to provide a few demonstrations, known as in-context examples, without adding more or updating existing model parameters. This in-context learning (ICL) capability of LLMs is intriguing, and it is not yet fully understood how pretrained LLMs acquire such capabilities. In this paper, we investigate the reason why a transformer-based language model can accomplish in-context learning after pre-training on a general language corpus by proposing one hypothesis that LLMs can simulate kernel regression with internal representations when faced with in-context examples. More concretely, we first prove that Bayesian inference on in-context prompts can be asymptotically understood as kernel regression $\hat y = \sum_i y_i K(x, x_i)/\sum_i K(x, x_i)$ as the number of in-context demon
    

