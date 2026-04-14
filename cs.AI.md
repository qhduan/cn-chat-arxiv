# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CROP: Conservative Reward for Model-based Offline Policy Optimization.](http://arxiv.org/abs/2310.17245) | CROP提出了一种保守奖励的模型训练方法用于基于模型的离线策略优化，通过同时最小化估计误差和随机动作奖励来实现保守的奖励估计。 |
| [^2] | [InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4.](http://arxiv.org/abs/2308.12067) | InstructionGPT-4通过仅使用200个例子进行微调，在多模式指令数据质量度量和选择器的帮助下，在各种评估任务中优于原始的MiniGPT-4。 |

# 详细

[^1]: CROP: 保守奖励用于基于模型的离线策略优化

    CROP: Conservative Reward for Model-based Offline Policy Optimization. (arXiv:2310.17245v1 [cs.LG])

    [http://arxiv.org/abs/2310.17245](http://arxiv.org/abs/2310.17245)

    CROP提出了一种保守奖励的模型训练方法用于基于模型的离线策略优化，通过同时最小化估计误差和随机动作奖励来实现保守的奖励估计。

    

    离线强化学习旨在使用收集到的数据进行策略优化，而无需进行在线交互。基于模型的方法在解决离线强化学习挑战方面特别有吸引力，因为它们能够通过使用模型生成数据来缓解离线数据的限制。之前的研究表明，在策略优化过程中将保守性引入模型或Q函数可以有效缓解离线强化学习中普遍存在的分布漂移问题。然而，关于奖励估计中保守性的影响的研究仍然不足。本文提出了一种新颖的基于模型的离线强化学习算法CROP，该算法在模型训练中保守地估计奖励。为了实现保守的奖励估计，CROP同时最小化估计误差和随机动作的奖励。理论分析表明，这种保守的奖励机制导致...（文章摘要未完，下同）

    Offline reinforcement learning (RL) aims to optimize policy using collected data without online interactions. Model-based approaches are particularly appealing for addressing offline RL challenges due to their capability to mitigate the limitations of offline data through data generation using models. Prior research has demonstrated that introducing conservatism into the model or Q-function during policy optimization can effectively alleviate the prevalent distribution drift problem in offline RL. However, the investigation into the impacts of conservatism in reward estimation is still lacking. This paper proposes a novel model-based offline RL algorithm, Conservative Reward for model-based Offline Policy optimization (CROP), which conservatively estimates the reward in model training. To achieve a conservative reward estimation, CROP simultaneously minimizes the estimation error and the reward of random actions. Theoretical analysis shows that this conservative reward mechanism leads 
    
[^2]: InstructionGPT-4: 一个200指令范式用于微调MiniGPT-4

    InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4. (arXiv:2308.12067v1 [cs.LG])

    [http://arxiv.org/abs/2308.12067](http://arxiv.org/abs/2308.12067)

    InstructionGPT-4通过仅使用200个例子进行微调，在多模式指令数据质量度量和选择器的帮助下，在各种评估任务中优于原始的MiniGPT-4。

    

    多模式大型语言模型通过两阶段的训练过程获取其遵循指令的能力：在图像-文本对上进行预训练，然后在监督式视觉-语言指令数据上进行微调。最近的研究表明，即使只有有限量的高质量遵循指令数据，大型语言模型也能取得令人满意的结果。在本文中，我们介绍了InstructionGPT-4，它经过微调的数据集只包含200个例子，约占MiniGPT-4对齐数据集中使用的遵循指令数据的6%。我们首先提出了几个用于评估多模式指令数据质量的度量指标。基于这些度量指标，我们提出了一个简单而有效的数据选择器，自动识别和过滤低质量的视觉-语言数据。通过采用这种方法，InstructionGPT-4在各种评估（如视觉问答、GPT-4偏好）上优于原始的MiniGPT-4。总体而言，我们的研究发现...

    Multimodal large language models acquire their instruction-following capabilities through a two-stage training process: pre-training on image-text pairs and fine-tuning on supervised vision-language instruction data. Recent studies have shown that large language models can achieve satisfactory results even with a limited amount of high-quality instruction-following data. In this paper, we introduce InstructionGPT-4, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. We first propose several metrics to access the quality of multimodal instruction data. Based on these metrics, we present a simple and effective data selector to automatically identify and filter low-quality vision-language data. By employing this method, InstructionGPT-4 outperforms the original MiniGPT-4 on various evaluations (e.g., visual question answering, GPT-4 preference). Overall, our findi
    

