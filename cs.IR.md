# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation.](http://arxiv.org/abs/2311.01304) | VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。 |
| [^2] | [Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation.](http://arxiv.org/abs/2306.04487) | 本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。 |

# 详细

[^1]: VM-Rec：一种用于冷启动用户推荐的变分映射方法

    VM-Rec: A Variational Mapping Approach for Cold-start User Recommendation. (arXiv:2311.01304v1 [cs.IR])

    [http://arxiv.org/abs/2311.01304](http://arxiv.org/abs/2311.01304)

    VM-Rec是一种用于解决冷启动用户推荐问题的变分映射方法，该方法基于生成表达能力强的嵌入的现有用户的互动，从而模拟生成冷启动用户的嵌入过程。

    

    冷启动问题是大多数推荐系统面临的共同挑战。传统的推荐模型在冷启动用户的互动非常有限时通常难以生成具有足够表达能力的嵌入。此外，缺乏用户的辅助内容信息加剧了挑战的存在，使得大多数冷启动方法难以应用。为了解决这个问题，我们观察到，如果模型能够为相对更多互动的现有用户生成具有表达能力的嵌入，这些用户最初也是冷启动用户，那么我们可以建立一个从少量初始互动到具有表达能力的嵌入的映射，模拟为冷启动用户生成嵌入的过程。基于这个观察，我们提出了一种变分映射方法用于冷启动用户推荐（VM-Rec）。首先，我们根据冷启动用户的初始互动生成个性化的映射函数，并进行参数优化。

    The cold-start problem is a common challenge for most recommender systems. With extremely limited interactions of cold-start users, conventional recommender models often struggle to generate embeddings with sufficient expressivity. Moreover, the absence of auxiliary content information of users exacerbates the presence of challenges, rendering most cold-start methods difficult to apply. To address this issue, our motivation is based on the observation that if a model can generate expressive embeddings for existing users with relatively more interactions, who were also initially cold-start users, then we can establish a mapping from few initial interactions to expressive embeddings, simulating the process of generating embeddings for cold-start users. Based on this motivation, we propose a Variational Mapping approach for cold-start user Recommendation (VM-Rec). Firstly, we generate a personalized mapping function for cold-start users based on their initial interactions, and parameters 
    
[^2]: 接受不确定性：自适应模糊偏好策略学习用于多轮会话推荐

    Embracing Uncertainty: Adaptive Vague Preference Policy Learning for Multi-round Conversational Recommendation. (arXiv:2306.04487v1 [cs.IR])

    [http://arxiv.org/abs/2306.04487](http://arxiv.org/abs/2306.04487)

    本文提出了一种称为“模糊偏好多轮会话推荐”（VPMCR）的新场景，考虑了用户在 CRS 中的模糊和波动的偏好。该方法采用软估计机制避免过滤过度，并通过自适应模糊偏好策略学习框架获得了实验上的良好推荐效果。

    

    会话式推荐系统 (CRS) 通过多轮交互，动态引导用户表达偏好，有效地解决信息不对称问题。现有的 CRS 基本上假设用户有明确的偏好。在这种情况下，代理将完全信任用户反馈，并将接受或拒绝信号视为过滤项目和减少候选空间的强指标，这可能导致过滤过度的问题。然而，在现实中，用户的偏好往往是模糊和波动的，存在不确定性，他们在交互过程中的愿望和决策可能会发生变化。为了解决这个问题，我们引入了一个新颖的场景，称为“模糊偏好多轮会话推荐”（VPMCR），它考虑到用户在 CRS 中的模糊和波动的偏好。VPMCR 采用软估计机制为所有候选项目分配非零置信度分数，自然地避免了过滤过度的问题。在 VPMCR 设置中，我们提出了一种自适应模糊偏好策略学习框架，利用强化学习和偏好引导来学习 CRS 代理的最优策略。在两个真实数据集上的实验结果表明，相较于几种最先进的基准方法，我们提出的 VPMCR 方法具有更好的推荐效果。

    Conversational recommendation systems (CRS) effectively address information asymmetry by dynamically eliciting user preferences through multi-turn interactions. Existing CRS widely assumes that users have clear preferences. Under this assumption, the agent will completely trust the user feedback and treat the accepted or rejected signals as strong indicators to filter items and reduce the candidate space, which may lead to the problem of over-filtering. However, in reality, users' preferences are often vague and volatile, with uncertainty about their desires and changing decisions during interactions.  To address this issue, we introduce a novel scenario called Vague Preference Multi-round Conversational Recommendation (VPMCR), which considers users' vague and volatile preferences in CRS.VPMCR employs a soft estimation mechanism to assign a non-zero confidence score for all candidate items to be displayed, naturally avoiding the over-filtering problem. In the VPMCR setting, we introduc
    

