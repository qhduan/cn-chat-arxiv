# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Discrete Neural Algorithmic Reasoning](https://arxiv.org/abs/2402.11628) | 这项工作提出了一种强制神经推理器维护执行轨迹作为有限预定义状态组合的方法，通过对算法状态转换的监督训练，使模型能够与原始算法完美对齐，并在基准测试中取得了完美的测试成绩。 |
| [^2] | [Stars Are All You Need: A Distantly Supervised Pyramid Network for Document-Level End-to-End Sentiment Analysis.](http://arxiv.org/abs/2305.01710) | 本文提出了一种文档级端到端情感分析方法，通过星级评分标签，实现方面检测、情感分析和评分预测，具有良好的性能和可解释性。 |

# 详细

[^1]: 离散神经算法推理

    Discrete Neural Algorithmic Reasoning

    [https://arxiv.org/abs/2402.11628](https://arxiv.org/abs/2402.11628)

    这项工作提出了一种强制神经推理器维护执行轨迹作为有限预定义状态组合的方法，通过对算法状态转换的监督训练，使模型能够与原始算法完美对齐，并在基准测试中取得了完美的测试成绩。

    

    神经算法推理旨在通过学习模仿经典算法的执行来捕捉神经网络中的计算。尽管常见的架构足够表达正确的模型在权重空间中，但当前的神经推理器在处理超出分布数据时面临泛化困难。另一方面，经典计算不受分布变化的影响，因为它们可以描述为离散计算状态之间的转换。在这项工作中，我们提出强制神经推理器将执行轨迹作为有限预定义状态的组合进行维护。通过对算法状态转换的监督训练，这种模型能够与原始算法完美对齐。为了证明这一点，我们在SALSA-CLRS基准测试上评估我们的方法，在那里我们为所有任务获得了完美的测试成绩。此外，所提出的架构选择使我们能够证明...

    arXiv:2402.11628v1 Announce Type: new  Abstract: Neural algorithmic reasoning aims to capture computations with neural networks via learning the models to imitate the execution of classical algorithms. While common architectures are expressive enough to contain the correct model in the weights space, current neural reasoners are struggling to generalize well on out-of-distribution data. On the other hand, classical computations are not affected by distribution shifts as they can be described as transitions between discrete computational states. In this work, we propose to force neural reasoners to maintain the execution trajectory as a combination of finite predefined states. Trained with supervision on the algorithm's state transitions, such models are able to perfectly align with the original algorithm. To show this, we evaluate our approach on the SALSA-CLRS benchmark, where we get perfect test scores for all tasks. Moreover, the proposed architectural choice allows us to prove the 
    
[^2]: 星辰即你所需：用远程监督金字塔网络进行文档级端到端情感分析

    Stars Are All You Need: A Distantly Supervised Pyramid Network for Document-Level End-to-End Sentiment Analysis. (arXiv:2305.01710v1 [cs.CL])

    [http://arxiv.org/abs/2305.01710](http://arxiv.org/abs/2305.01710)

    本文提出了一种文档级端到端情感分析方法，通过星级评分标签，实现方面检测、情感分析和评分预测，具有良好的性能和可解释性。

    

    本文提出了文档级端到端情感分析方法，可以通过星级评分标签对在线评论中表达的方面和评论情感进行有效的统一分析。我们假设星级评分标签是评论中各方面评分的“粗粒度综合”。我们提出了一种远程监督的金字塔网络（DSPN），只用文档星级评分标签进行训练，即可有效地执行方面-类别检测、方面-类别情感分析和评分预测。通过以端到端的方式执行这三个相关的情感子任务，DSPN可以提取评论中提到的方面，确定相应的情感，并预测星级评分标签。我们在英文和汉语多方面评论数据集上评估了DSPN，发现仅使用星级评分标签进行监督，DSPN的性能与各种基准模型相当。我们还展示了DSPN在评论上的可解释性输出，以说明金字塔网络的结构。

    In this paper, we propose document-level end-to-end sentiment analysis to efficiently understand aspect and review sentiment expressed in online reviews in a unified manner. In particular, we assume that star rating labels are a "coarse-grained synthesis" of aspect ratings across in the review. We propose a Distantly Supervised Pyramid Network (DSPN) to efficiently perform Aspect-Category Detection, Aspect-Category Sentiment Analysis, and Rating Prediction using only document star rating labels for training. By performing these three related sentiment subtasks in an end-to-end manner, DSPN can extract aspects mentioned in the review, identify the corresponding sentiments, and predict the star rating labels. We evaluate DSPN on multi-aspect review datasets in English and Chinese and find that with only star rating labels for supervision, DSPN can perform comparably well to a variety of benchmark models. We also demonstrate the interpretability of DSPN's outputs on reviews to show the py
    

