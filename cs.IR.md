# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Repeated Padding as Data Augmentation for Sequential Recommendation](https://arxiv.org/abs/2403.06372) | 本文提出了一种名为"RepPad"的简单而有效的填充方法，旨在充分利用填充空间来提高顺序推荐模型的性能和训练效率。 |

# 详细

[^1]: 重复填充作为顺序推荐的数据增强

    Repeated Padding as Data Augmentation for Sequential Recommendation

    [https://arxiv.org/abs/2403.06372](https://arxiv.org/abs/2403.06372)

    本文提出了一种名为"RepPad"的简单而有效的填充方法，旨在充分利用填充空间来提高顺序推荐模型的性能和训练效率。

    

    顺序推荐旨在根据用户的历史互动提供个性化建议。在训练顺序模型时，填充是一种被广泛采用的技术，主要原因有两个：1）绝大多数模型只能处理固定长度的序列；2）基于批处理的训练需要确保每个批次中的序列具有相同的长度。通常使用特殊值0作为填充内容，不包含实际信息并在模型计算中被忽略。这种常识填充策略引出了一个以前从未探讨过的问题：我们能否通过填充其他内容充分利用这一闲置输入空间，进一步提高模型性能和训练效率？ 在本文中，我们提出了一种简单而有效的填充方法，名为RepPad (重复填充)。

    arXiv:2403.06372v1 Announce Type: new  Abstract: Sequential recommendation aims to provide users with personalized suggestions based on their historical interactions. When training sequential models, padding is a widely adopted technique for two main reasons: 1) The vast majority of models can only handle fixed-length sequences; 2) Batching-based training needs to ensure that the sequences in each batch have the same length. The special value \emph{0} is usually used as the padding content, which does not contain the actual information and is ignored in the model calculations. This common-sense padding strategy leads us to a problem that has never been explored before: \emph{Can we fully utilize this idle input space by padding other content to further improve model performance and training efficiency?}   In this paper, we propose a simple yet effective padding method called \textbf{Rep}eated \textbf{Pad}ding (\textbf{RepPad}). Specifically, we use the original interaction sequences as
    

