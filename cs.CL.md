# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing](https://arxiv.org/abs/2403.08004) | 本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。 |
| [^2] | [Speech emotion recognition from voice messages recorded in the wild](https://arxiv.org/abs/2403.02167) | 使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。 |
| [^3] | [Prediction-Powered Ranking of Large Language Models](https://arxiv.org/abs/2402.17826) | 该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。 |

# 详细

[^1]: Pix2Pix-OnTheFly: 利用LLMs进行指导图像编辑

    Pix2Pix-OnTheFly: Leveraging LLMs for Instruction-Guided Image Editing

    [https://arxiv.org/abs/2403.08004](https://arxiv.org/abs/2403.08004)

    本文提出了一种新方法，实现了基于自然语言指令的图像编辑，在不需要任何预备工作的情况下，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，表现出有效性和竞争力。

    

    众所周知，最近结合语言处理和图像处理的研究引起了广泛关注，本文提出了一种全新的方法，通过图像字幕和DDIM反演，获取编辑方向嵌入，进行指导图像编辑，而无需预备工作，证明了该方法的有效性和竞争力。

    arXiv:2403.08004v1 Announce Type: cross  Abstract: The combination of language processing and image processing keeps attracting increased interest given recent impressive advances that leverage the combined strengths of both domains of research. Among these advances, the task of editing an image on the basis solely of a natural language instruction stands out as a most challenging endeavour. While recent approaches for this task resort, in one way or other, to some form of preliminary preparation, training or fine-tuning, this paper explores a novel approach: We propose a preparation-free method that permits instruction-guided image editing on the fly. This approach is organized along three steps properly orchestrated that resort to image captioning and DDIM inversion, followed by obtaining the edit direction embedding, followed by image editing proper. While dispensing with preliminary preparation, our approach demonstrates to be effective and competitive, outperforming recent, state 
    
[^2]: 从野外录制的语音消息中识别语音情感

    Speech emotion recognition from voice messages recorded in the wild

    [https://arxiv.org/abs/2403.02167](https://arxiv.org/abs/2403.02167)

    使用Emotional Voice Messages数据库，结合eGeMAPS特征和Transformer模型，实现了在野外录制的语音消息中的语音情感识别，取得了较高的准确度，并比基准模型提高了10%。

    

    用于语音情感识别（SER）的情感数据集通常包含表演或引发的语音，限制了它们在现实场景中的适用性。在这项工作中，我们使用了Emotional Voice Messages（EMOVOME）数据库，其中包括来自100名西班牙语使用者在消息应用中的自发语音消息，由专家和非专家标注者以连续和离散的情感进行标记。我们使用了eGeMAPS特征、基于Transformer的模型以及它们的组合来创建讲话者无关的SER模型。我们将结果与参考数据库进行了比较，并分析了标注者和性别公平性的影响。预训练的Unispeech-L模型及其与eGeMAPS的组合取得了最佳结果，在3类valence和arousal预测中分别获得了61.64%和55.57%的Unweighted Accuracy（UA），比基线模型提高了10%。对于情感类别，获得了42.58%的UA。EMOVOME表现不佳。

    arXiv:2403.02167v1 Announce Type: cross  Abstract: Emotion datasets used for Speech Emotion Recognition (SER) often contain acted or elicited speech, limiting their applicability in real-world scenarios. In this work, we used the Emotional Voice Messages (EMOVOME) database, including spontaneous voice messages from conversations of 100 Spanish speakers on a messaging app, labeled in continuous and discrete emotions by expert and non-expert annotators. We created speaker-independent SER models using the eGeMAPS features, transformer-based models and their combination. We compared the results with reference databases and analyzed the influence of annotators and gender fairness. The pre-trained Unispeech-L model and its combination with eGeMAPS achieved the highest results, with 61.64% and 55.57% Unweighted Accuracy (UA) for 3-class valence and arousal prediction respectively, a 10% improvement over baseline models. For the emotion categories, 42.58% UA was obtained. EMOVOME performed low
    
[^3]: 大型语言模型的预测排名

    Prediction-Powered Ranking of Large Language Models

    [https://arxiv.org/abs/2402.17826](https://arxiv.org/abs/2402.17826)

    该研究提出了一种统计框架，可以衡量人类与模型偏好之间的不确定性，从而进行大型语言模型的预测排名。

    

    大型语言模型通常根据其与人类偏好的一致性水平进行排名--如果一个模型的输出更受人类偏好，那么它就比其他模型更好。本文提出了一种统计框架来弥合人类与模型偏好之间可能引入的不一致性。

    arXiv:2402.17826v1 Announce Type: cross  Abstract: Large language models are often ranked according to their level of alignment with human preferences -- a model is better than other models if its outputs are more frequently preferred by humans. One of the most popular ways to elicit human preferences utilizes pairwise comparisons between the outputs provided by different models to the same inputs. However, since gathering pairwise comparisons by humans is costly and time-consuming, it has become a very common practice to gather pairwise comparisons by a strong large language model -- a model strongly aligned with human preferences. Surprisingly, practitioners cannot currently measure the uncertainty that any mismatch between human and model preferences may introduce in the constructed rankings. In this work, we develop a statistical framework to bridge this gap. Given a small set of pairwise comparisons by humans and a large set of pairwise comparisons by a model, our framework provid
    

