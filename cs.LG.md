# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)](https://arxiv.org/abs/2606.27163) | 本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。 |
| [^2] | [FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering.](http://arxiv.org/abs/2310.16152) | 本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。 |

# 详细

[^1]: 学习折叠：LeHome 2026挑战赛获奖方案（在线第一名，线下第二名）

    Learning to Fold: prizewinning solution at LeHome Challenge 2026 (1st place online, 2nd offline)

    [https://arxiv.org/abs/2606.27163](https://arxiv.org/abs/2606.27163)

    本文提出了一种通过强化学习改进的视觉-语言-动作策略，实现了在线仿真和真实世界双臂衣物折叠的高性能，并整合了多种优化技术。

    

    本文描述了我在LeHome 2026挑战赛（ICRA 2026双臂衣物折叠竞赛）中的解决方案。该系统在在线（仿真）环节中于62支队伍中排名第一，并在真实世界决赛中排名第二。它通过强化学习循环改进了视觉-语言-动作（VLA）策略。该策略本身即为价值函数：同一网络不仅预测动作，还预测成功、进度以及若干任务相关的未来量，这些预测用于优势估计、实时故障检测和候选选择。本工作主要将现有强化学习思想与工程和优化贡献相结合，这些贡献可作为整体配方或单独使用：结合AWR和RECAP用于流匹配VLA；通过HuggingFace Hub实现异步分布式训练/部署流水线；通过汤普森采样进行推理时超参数优化；以及包含相机对齐工具和强数据增强的仿真到现实迁移方案。

    arXiv:2606.27163v1 Announce Type: cross  Abstract: I describe my solution to the LeHome Challenge 2026, an ICRA 2026 competition on bimanual garment folding. The system placed 1st of 62 teams in the online (simulation) round and 2nd in the real-world final. It improves a vision-language-action (VLA) policy with a reinforcement-learning loop. The policy is its own value function: the same network that predicts actions also predicts success, progress, and a few task-relevant future quantities, and those predictions drive advantage estimation, live failure detection, and candidate selection. The work mostly recombines existing RL ideas with engineering and optimization contributions that can be used together as one recipe or individually: AWR + RECAP combined for flow-matching VLA; an asynchronous distributed training / rollout pipeline through HuggingFace Hub; inference-time hyperparameters optimization via Thompson sampling; a sim-to-real recipe with camera-alignment tooling, heavy augm
    
[^2]: FLTrojan: 通过选择性权重篡改对联邦语言模型进行隐私泄露攻击

    FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering. (arXiv:2310.16152v1 [cs.CR])

    [http://arxiv.org/abs/2310.16152](http://arxiv.org/abs/2310.16152)

    本文提出了一种FLTrojan攻击方法，通过选择性权重篡改，从联邦语言模型中泄露隐私敏感用户数据。通过观察到FL中中间轮次的模型快照可以引起更大的隐私泄露，并发现隐私泄露可以通过篡改模型的选择性权重来加剧。

    

    联邦学习(Federated learning, FL)正成为许多技术应用中的关键组件，包括语言建模领域，其中个体FL参与者在其本地数据集中往往具有敏感的文本数据。然而，确定联邦语言模型中的隐私泄露程度并不简单，现有的攻击只是试图提取数据，而不考虑数据的敏感性或天真性。为了填补这一空白，在本文中，我们介绍了关于从联邦语言模型中泄露隐私敏感用户数据的两个新发现。首先，我们观察到FL中中间轮次的模型快照比最终训练模型能够造成更大的隐私泄露。其次，我们确定隐私泄露可以通过篡改模型的选择性权重来加剧，这些权重特别负责记忆敏感训练数据。我们展示了恶意客户端如何在FL中泄露其他用户的隐私敏感数据。

    Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even
    

