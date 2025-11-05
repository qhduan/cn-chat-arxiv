# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints](https://arxiv.org/abs/2402.09034) | 该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。 |
| [^2] | [Improving Adversarial Attacks on Latent Diffusion Model](https://arxiv.org/abs/2310.04687) | 提出了一种改进 Latent Diffusion Model 的对抗攻击方法 ACE，其通过统一模式的额外误差来促使模型学习特定的偏差，从而胜过了目前最先进的方法 |
| [^3] | [Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations](https://arxiv.org/abs/2310.04566) | 本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。 |
| [^4] | [Complex QA and language models hybrid architectures, Survey.](http://arxiv.org/abs/2302.09051) | 本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。 |

# 详细

[^1]: 使用平方Sigmoid TanH (SST)激活在数据限制下提高顺序模型性能

    Enhancing Sequential Model Performance with Squared Sigmoid TanH (SST) Activation Under Data Constraints

    [https://arxiv.org/abs/2402.09034](https://arxiv.org/abs/2402.09034)

    该论文提出了一种平方Sigmoid TanH（SST）激活函数，用于增强在数据限制下的顺序模型学习能力。通过数学平方放大强激活和弱激活之间的差异，改善梯度流和信息过滤。在多个应用中评估了SST驱动的LSTM和GRU模型的性能。

    

    激活函数通过引入非线性来使神经网络能够学习复杂的表示。虽然前馈模型通常使用修正线性单元，但是顺序模型如递归神经网络、长短时记忆（LSTM）和门控循环单元（GRU）仍然依赖于Sigmoid和TanH激活函数。然而，这些传统的激活函数常常在训练在小顺序数据集上时难以建模稀疏模式以有效捕获时间依赖性。为了解决这个限制，我们提出了特别针对在数据限制下增强顺序模型学习能力的平方Sigmoid TanH（SST）激活。SST通过数学平方来放大强激活和弱激活之间的差异，随着信号随时间传播，有助于改善梯度流和信息过滤。我们评估了使用SST的LSTM和GRU模型在不同应用中的性能。

    arXiv:2402.09034v1 Announce Type: cross Abstract: Activation functions enable neural networks to learn complex representations by introducing non-linearities. While feedforward models commonly use rectified linear units, sequential models like recurrent neural networks, long short-term memory (LSTMs) and gated recurrent units (GRUs) still rely on Sigmoid and TanH activation functions. However, these classical activation functions often struggle to model sparse patterns when trained on small sequential datasets to effectively capture temporal dependencies. To address this limitation, we propose squared Sigmoid TanH (SST) activation specifically tailored to enhance the learning capability of sequential models under data constraints. SST applies mathematical squaring to amplify differences between strong and weak activations as signals propagate over time, facilitating improved gradient flow and information filtering. We evaluate SST-powered LSTMs and GRUs for diverse applications, such a
    
[^2]: 改进潜在扩散模型的对抗攻击

    Improving Adversarial Attacks on Latent Diffusion Model

    [https://arxiv.org/abs/2310.04687](https://arxiv.org/abs/2310.04687)

    提出了一种改进 Latent Diffusion Model 的对抗攻击方法 ACE，其通过统一模式的额外误差来促使模型学习特定的偏差，从而胜过了目前最先进的方法

    

    对 Latent Diffusion Model (LDM)，这种最先进的图像生成模型，进行对抗攻击已经被证明是有效防止 LDM 在未经授权的图像上进行恶意微调的保护手段。我们展示了这些攻击会对 LDM 预测的对抗样本的评分函数添加额外的误差。在这些对抗样本上进行微调的 LDM 学习通过一个偏差降低误差，从而遭受攻击并使用偏差预测评分函数。基于这一动态，我们提出了通过一致得分函数错误进行攻击（ACE）来改进 LDM 的对抗攻击。ACE 统一了添加到预测得分函数的额外误差的模式。这促使微调的 LDM 学习与对评分函数进行预测的偏差学习相同的模式。然后我们引入一个精心设计的模式来改进攻击。我们的方法在对 LDM 的对抗攻击中胜过了最先进的方法。

    arXiv:2310.04687v3 Announce Type: replace-cross  Abstract: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.
    
[^3]: Knolling Bot: 从整洁的示范中学习机器人对象排列

    Knolling Bot: Learning Robotic Object Arrangement from Tidy Demonstrations

    [https://arxiv.org/abs/2310.04566](https://arxiv.org/abs/2310.04566)

    本论文介绍了一种自监督学习框架，利用Transformer神经网络使机器人能够从整齐排列的示范中理解和复制整洁的概念，从而实现整理物品的功能。

    

    地址：arXiv:2310.04566v2  公告类型：replace-cross  摘要：解决家庭空间中散乱物品的整理挑战受到整洁性的多样性和主观性的复杂性影响。正如人类语言的复杂性允许同一理念的多种表达一样，家庭整洁偏好和组织模式变化广泛，因此预设物体位置将限制对新物体和环境的适应性。受自然语言处理（NLP）的进展启发，本文引入一种自监督学习框架，使机器人能够从整洁布局的示范中理解和复制整洁的概念，类似于使用会话数据集训练大语言模型（LLM）。我们利用一个Transformer神经网络来预测后续物体的摆放位置。我们展示了一个“整理”系统，利用机械臂和RGB相机在桌子上组织不同大小和数量的物品。

    arXiv:2310.04566v2 Announce Type: replace-cross  Abstract: Addressing the challenge of organizing scattered items in domestic spaces is complicated by the diversity and subjective nature of tidiness. Just as the complexity of human language allows for multiple expressions of the same idea, household tidiness preferences and organizational patterns vary widely, so presetting object locations would limit the adaptability to new objects and environments. Inspired by advancements in natural language processing (NLP), this paper introduces a self-supervised learning framework that allows robots to understand and replicate the concept of tidiness from demonstrations of well-organized layouts, akin to using conversational datasets to train Large Language Models(LLM). We leverage a transformer neural network to predict the placement of subsequent objects. We demonstrate a ``knolling'' system with a robotic arm and an RGB camera to organize items of varying sizes and quantities on a table. Our 
    
[^4]: 复杂问答和语言模型混合架构综述

    Complex QA and language models hybrid architectures, Survey. (arXiv:2302.09051v4 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09051](http://arxiv.org/abs/2302.09051)

    本文综述了语言模型架构和策略的最新进展，并重点关注混合技术在复杂问题回答中的应用，讨论了该领域的挑战和未来研究方向。

    

    本文回顾了语言模型架构和策略的最新进展，重点关注混合技术在复杂问题回答中的应用。大型语言模型能够在标准问题上利用公共数据，但在解决更具体的复杂问题时（如在不同文化中个人自由概念的变化如何？什么是为减少气候变化而实现的最佳发电方法组合？），需要特定的架构、知识、技能、方法、敏感数据保护、可解释性、人类审批和多功能反馈。最近的项目如ChatGPT和GALACTICA允许非专业人员了解LLM在复杂QA中的巨大潜力以及同等强大的局限性。在本文中，我们首先审查所需的技能和评估技术。然后，我们综述了现有的混合架构，将LLM与基于规则的方法、信息检索、知识图谱和其他AI/ML技术相结合。最后，我们指出这些CQA系统的挑战，并提出未来研究的可能方向。

    This paper reviews the state-of-the-art of language models architectures and strategies for "complex" question-answering (QA, CQA, CPS) with a focus on hybridization. Large Language Models (LLM) are good at leveraging public data on standard problems but once you want to tackle more specific complex questions or problems (e.g. How does the concept of personal freedom vary between different cultures ? What is the best mix of power generation methods to reduce climate change ?) you may need specific architecture, knowledge, skills, methods, sensitive data protection, explainability, human approval and versatile feedback... Recent projects like ChatGPT and GALACTICA have allowed non-specialists to grasp the great potential as well as the equally strong limitations of LLM in complex QA. In this paper, we start by reviewing required skills and evaluation techniques. We integrate findings from the robust community edited research papers BIG, BLOOM and HELM which open source, benchmark and an
    

