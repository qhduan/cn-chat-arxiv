# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DynST: Dynamic Sparse Training for Resource-Constrained Spatio-Temporal Forecasting](https://arxiv.org/abs/2403.02914) | 传统传感器部署方法在地球科学系统中存在困难，本研究提出了一种动态稀疏训练方法，能够有效优化传感器的部署和数据收集过程。 |
| [^2] | [PeFoMed: Parameter Efficient Fine-tuning on Multimodal Large Language Models for Medical Visual Question Answering.](http://arxiv.org/abs/2401.02797) | 本文提出了一种针对医学视觉问答的多模态大型语言模型的参数高效微调框架，并在公共数据集上进行了验证。实验证明，该模型在封闭式问题上比GPT-4v模型的准确率提高了26％。 |
| [^3] | [Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks.](http://arxiv.org/abs/2306.09377) | 语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。 |

# 详细

[^1]: DynST：资源受限时空预测的动态稀疏训练

    DynST: Dynamic Sparse Training for Resource-Constrained Spatio-Temporal Forecasting

    [https://arxiv.org/abs/2403.02914](https://arxiv.org/abs/2403.02914)

    传统传感器部署方法在地球科学系统中存在困难，本研究提出了一种动态稀疏训练方法，能够有效优化传感器的部署和数据收集过程。

    

    随着传感器服务的不断增加，尽管为面向深度学习的地球科学提供了宝贵的路径并提供了大量的地球系统数据，但这也给它们的工业级部署带来了困难。具体来说，地球科学系统在很大程度上依赖于传感器的广泛部署，然而，由于复杂的地理和社会因素，传感器数据的收集受到限制，这使得实现全面覆盖和统一部署具有挑战性。为了减轻这一障碍，传统的传感器部署方法利用特定算法设计和部署传感器。这些方法动态调整传感器的激活时间，以优化对每个子区域的检测过程。遗憾的是，基于历史观测和地理特征制定激活策略，这些方法和生成的模型既不简单也不实用。

    arXiv:2403.02914v1 Announce Type: new  Abstract: The ever-increasing sensor service, though opening a precious path and providing a deluge of earth system data for deep-learning-oriented earth science, sadly introduce a daunting obstacle to their industrial level deployment. Concretely, earth science systems rely heavily on the extensive deployment of sensors, however, the data collection from sensors is constrained by complex geographical and social factors, making it challenging to achieve comprehensive coverage and uniform deployment. To alleviate the obstacle, traditional approaches to sensor deployment utilize specific algorithms to design and deploy sensors. These methods dynamically adjust the activation times of sensors to optimize the detection process across each sub-region. Regrettably, formulating an activation strategy generally based on historical observations and geographic characteristics, which make the methods and resultant models were neither simple nor practical. Wo
    
[^2]: PeFoMed：针对医学视觉问答的多模态大型语言模型的参数高效微调

    PeFoMed: Parameter Efficient Fine-tuning on Multimodal Large Language Models for Medical Visual Question Answering. (arXiv:2401.02797v1 [cs.CL])

    [http://arxiv.org/abs/2401.02797](http://arxiv.org/abs/2401.02797)

    本文提出了一种针对医学视觉问答的多模态大型语言模型的参数高效微调框架，并在公共数据集上进行了验证。实验证明，该模型在封闭式问题上比GPT-4v模型的准确率提高了26％。

    

    多模态大型语言模型（MLLM）在传统大型语言模型的能力上进行了进化扩展，使它们能够应对超越纯文本应用范围的挑战。它利用了先前编码在这些语言模型中的知识，从而增强了它们在多模态环境下的适用性和功能性。最近的研究探讨了将MLLMs适应为生成任务以解决医学视觉问答（Med-VQA）任务的自由形式答案的能力。在本文中，我们提出了一种针对Med-VQA应用特别定制的参数高效微调框架，并在公共基准数据集上进行了实证验证。为了准确衡量性能，我们进行了人工评估，结果显示我们的模型在封闭式问题的整体准确率上达到了81.9％，且其相对于GPT-4v模型的绝对准确率超过了26％。

    Multimodal large language models (MLLMs) represent an evolutionary expansion in the capabilities of traditional large language models, enabling them to tackle challenges that surpass the scope of purely text-based applications. It leverages the knowledge previously encoded within these language models, thereby enhancing their applicability and functionality in the reign of multimodal contexts. Recent works investigate the adaptation of MLLMs to predict free-form answers as a generative task to solve medical visual question answering (Med-VQA) tasks. In this paper, we propose a parameter efficient framework for fine-tuning MLLM specifically tailored to Med-VQA applications, and empirically validate it on a public benchmark dataset. To accurately measure the performance, we employ human evaluation and the results reveal that our model achieves an overall accuracy of 81.9%, and outperforms the GPT-4v model by a significant margin of 26% absolute accuracy on closed-ended questions. The cod
    
[^3]: 对齐语言的视觉表示预测人类在自然学习任务中的行为

    Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks. (arXiv:2306.09377v1 [cs.LG])

    [http://arxiv.org/abs/2306.09377](http://arxiv.org/abs/2306.09377)

    语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。

    

    人类具备识别和概括自然物体相关特征的能力，在各种情境中有所帮助。为了研究这种现象并确定最有效的表示方式以预测人类行为，我们进行了两个涉及类别学习和奖励学习的实验。我们的实验使用逼真的图像作为刺激物，并要求参与者基于所有试验的新型刺激物作出准确的决策，因此需要泛化。在两个任务中，底层规则是使用人类相似性判断提取的刺激维度生成的简单线性函数。值得注意的是，参与者在几次试验内就成功地确定了相关的刺激特征，证明了有效的泛化。我们进行了广泛的模型比较，评估了各种深度学习模型的表示对人类选择的逐次预测准确性。有趣的是，自然语言处理任务（如语言建模和机器翻译）训练的模型表示优于视觉任务训练的模型表示，表明对齐语言的视觉表示可能更有效地预测人类在自然学习任务中的行为。

    Humans possess the ability to identify and generalize relevant features of natural objects, which aids them in various situations. To investigate this phenomenon and determine the most effective representations for predicting human behavior, we conducted two experiments involving category learning and reward learning. Our experiments used realistic images as stimuli, and participants were tasked with making accurate decisions based on novel stimuli for all trials, thereby necessitating generalization. In both tasks, the underlying rules were generated as simple linear functions using stimulus dimensions extracted from human similarity judgments. Notably, participants successfully identified the relevant stimulus features within a few trials, demonstrating effective generalization. We performed an extensive model comparison, evaluating the trial-by-trial predictive accuracy of diverse deep learning models' representations of human choices. Intriguingly, representations from models train
    

