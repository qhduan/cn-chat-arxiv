# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Defer in Content Moderation: The Human-AI Interplay](https://arxiv.org/abs/2402.12237) | 本文提出了一个模型，捕捉内容审核中人工智能的相互作用。 |

# 详细

[^1]: 学习在内容审核中推迟：人工智能与人类协同作用

    Learning to Defer in Content Moderation: The Human-AI Interplay

    [https://arxiv.org/abs/2402.12237](https://arxiv.org/abs/2402.12237)

    本文提出了一个模型，捕捉内容审核中人工智能的相互作用。

    

    成功的在线平台内容审核依赖于人工智能协同方法。本文介绍了一个模型，捕捉内容审核中人工智能的相互作用。算法观察到即将发布的帖子的背景信息，做出分类和准入决策，并安排帖子进行人工审核。

    arXiv:2402.12237v1 Announce Type: cross  Abstract: Successful content moderation in online platforms relies on a human-AI collaboration approach. A typical heuristic estimates the expected harmfulness of a post and uses fixed thresholds to decide whether to remove it and whether to send it for human review. This disregards the prediction uncertainty, the time-varying element of human review capacity and post arrivals, and the selective sampling in the dataset (humans only review posts filtered by the admission algorithm).   In this paper, we introduce a model to capture the human-AI interplay in content moderation. The algorithm observes contextual information for incoming posts, makes classification and admission decisions, and schedules posts for human review. Only admitted posts receive human reviews on their harmfulness. These reviews help educate the machine-learning algorithms but are delayed due to congestion in the human review system. The classical learning-theoretic way to ca
    

