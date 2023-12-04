# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MuseChat: A Conversational Music Recommendation System for Videos.](http://arxiv.org/abs/2310.06282) | MuseChat是一种创新的对话式音乐推荐系统，通过模拟用户和推荐系统之间的对话交互，利用预训练的音乐标签和艺术家信息，为用户提供定制的音乐推荐，使用户可以个性化选择他们喜欢的音乐。 |
| [^2] | [A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance.](http://arxiv.org/abs/2303.16767) | 该研究提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术相似性，并且实验证明该方法优于仅考虑语义相似性的方法。 |

# 详细

[^1]: MuseChat:一种视频对话音乐推荐系统

    MuseChat: A Conversational Music Recommendation System for Videos. (arXiv:2310.06282v1 [cs.LG])

    [http://arxiv.org/abs/2310.06282](http://arxiv.org/abs/2310.06282)

    MuseChat是一种创新的对话式音乐推荐系统，通过模拟用户和推荐系统之间的对话交互，利用预训练的音乐标签和艺术家信息，为用户提供定制的音乐推荐，使用户可以个性化选择他们喜欢的音乐。

    

    我们引入了MuseChat，一种创新的基于对话的音乐推荐系统。这个独特的平台不仅提供互动用户参与，还为输入的视频提供了定制的音乐推荐，使用户可以改进和个性化他们的音乐选择。与之相反，以前的系统主要强调内容的兼容性，往往忽视了用户个体偏好的细微差别。例如，所有的数据集都只提供基本的音乐-视频配对，或者带有音乐描述的配对。为了填补这一空白，我们的研究提供了三个贡献。首先，我们设计了一种对话合成方法，模拟了用户和推荐系统之间的两轮交互，利用预训练的音乐标签和艺术家信息。在这个交互中，用户提交一个视频给系统，系统会提供一个合适的音乐片段，并附带解释。之后，用户会表达他们对音乐的偏好，系统会呈现一个改进后的音乐推荐

    We introduce MuseChat, an innovative dialog-based music recommendation system. This unique platform not only offers interactive user engagement but also suggests music tailored for input videos, so that users can refine and personalize their music selections. In contrast, previous systems predominantly emphasized content compatibility, often overlooking the nuances of users' individual preferences. For example, all the datasets only provide basic music-video pairings or such pairings with textual music descriptions. To address this gap, our research offers three contributions. First, we devise a conversation-synthesis method that simulates a two-turn interaction between a user and a recommendation system, which leverages pre-trained music tags and artist information. In this interaction, users submit a video to the system, which then suggests a suitable music piece with a rationale. Afterwards, users communicate their musical preferences, and the system presents a refined music recomme
    
[^2]: 一种新的专利相似度测量方法：语义距离和技术距离

    A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance. (arXiv:2303.16767v1 [cs.IR])

    [http://arxiv.org/abs/2303.16767](http://arxiv.org/abs/2303.16767)

    该研究提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术相似性，并且实验证明该方法优于仅考虑语义相似性的方法。

    

    测量专利之间的相似性是确保创新的新颖性的关键步骤。然而，目前大多数专利相似度测量方法仍然依赖于专家手动分类专利。另一方面，一些研究提出了自动化方法；然而，大部分自动化方法只关注专利的语义相似性。为了解决这些问题，我们提出了一种混合方法，用于自动测量专利之间的相似性，同时考虑语义和技术的相似性。我们基于专利文本使用BERT测量语义相似性，使用Jaccard相似性计算专利的技术相似性，并通过分配权重来实现混合。我们的评估结果表明，所提出的方法优于仅考虑语义相似度的基准方法。

    Measuring similarity between patents is an essential step to ensure novelty of innovation. However, a large number of methods of measuring the similarity between patents still rely on manual classification of patents by experts. Another body of research has proposed automated methods; nevertheless, most of it solely focuses on the semantic similarity of patents. In order to tackle these limitations, we propose a hybrid method for automatically measuring the similarity between patents, considering both semantic and technological similarities. We measure the semantic similarity based on patent texts using BERT, calculate the technological similarity with IPC codes using Jaccard similarity, and perform hybridization by assigning weights to the two similarity methods. Our evaluation result demonstrates that the proposed method outperforms the baseline that considers the semantic similarity only.
    

