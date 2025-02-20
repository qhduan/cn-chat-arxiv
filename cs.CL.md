# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [NaturalTurn: A Method to Segment Transcripts into Naturalistic Conversational Turns](https://arxiv.org/abs/2403.15615) | NaturalTurn是一种专门设计用于准确捕捉自然对话交流动态的轮次分割算法，通过区分说话者的主要对话轮次和听众的次要话语，能够比现有方法更好地提取转录信息。 |
| [^2] | [A Transfer Attack to Image Watermarks](https://arxiv.org/abs/2403.15365) | 水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。 |
| [^3] | [Image-Text Matching with Multi-View Attention](https://arxiv.org/abs/2402.17237) | 本研究提出了一种使用多视图注意力的双流图像文本匹配方法，以解决单一表示难以全面覆盖复杂内容和缺乏交互的挑战。 |

# 详细

[^1]: NaturalTurn：一种将转录件分割成自然对话转折的方法

    NaturalTurn: A Method to Segment Transcripts into Naturalistic Conversational Turns

    [https://arxiv.org/abs/2403.15615](https://arxiv.org/abs/2403.15615)

    NaturalTurn是一种专门设计用于准确捕捉自然对话交流动态的轮次分割算法，通过区分说话者的主要对话轮次和听众的次要话语，能够比现有方法更好地提取转录信息。

    

    arXiv:2403.15615v1 公告类型: 新的 摘要: 对话是社会、认知和计算科学越来越感兴趣的主题。然而，随着对话数据集的规模和复杂性不断增加，研究人员缺乏可伸缩的方法将语音转录转换为会话轮次——社会互动的基本构建模块。我们介绍了“NaturalTurn”，一种旨在准确捕捉自然交流动态的轮次分割算法。NaturalTurn通过区分说话者的主要对话轮次和听众的次要话语，如背景声、简短插话和其他表现对话特征的平行言语形式，来运作。使用大型对话语料库的数据，我们展示了与现有方法派生的转录相比，NaturalTurn派生的转录表现出有利的统计和推断特性。NaturalTurn算法代表了一种改进。

    arXiv:2403.15615v1 Announce Type: new  Abstract: Conversation is the subject of increasing interest in the social, cognitive, and computational sciences. And yet, as conversational datasets continue to increase in size and complexity, researchers lack scalable methods to segment speech-to-text transcripts into conversational turns--the basic building blocks of social interaction. We introduce "NaturalTurn," a turn segmentation algorithm designed to accurately capture the dynamics of naturalistic exchange. NaturalTurn operates by distinguishing speakers' primary conversational turns from listeners' secondary utterances, such as backchannels, brief interjections, and other forms of parallel speech that characterize conversation. Using data from a large conversation corpus, we show how NaturalTurn-derived transcripts demonstrate favorable statistical and inferential characteristics compared to transcripts derived from existing methods. The NaturalTurn algorithm represents an improvement i
    
[^2]: 一种针对图像水印的转移攻击

    A Transfer Attack to Image Watermarks

    [https://arxiv.org/abs/2403.15365](https://arxiv.org/abs/2403.15365)

    水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。

    

    水印已被广泛应用于工业领域，用于检测由人工智能生成的图像。文献中对这种基于水印的检测器在白盒和黑盒环境下对抗攻击的稳健性有很好的理解。然而，在无盒环境下的稳健性却知之甚少。具体来说，多项研究声称图像水印在这种环境下是稳健的。在这项工作中，我们提出了一种新的转移对抗攻击来针对无盒环境下的图像水印。我们的转移攻击向带水印的图像添加微扰，以躲避被攻击者训练的多个替代水印模型，并且经过扰动的带水印图像也能躲避目标水印模型。我们的主要贡献是理论上和经验上展示了，基于水印的人工智能生成图像检测器即使攻击者没有访问水印模型或检测API，也不具有对抗攻击的稳健性。

    arXiv:2403.15365v1 Announce Type: cross  Abstract: Watermark has been widely deployed by industry to detect AI-generated images. The robustness of such watermark-based detector against evasion attacks in the white-box and black-box settings is well understood in the literature. However, the robustness in the no-box setting is much less understood. In particular, multiple studies claimed that image watermark is robust in such setting. In this work, we propose a new transfer evasion attack to image watermark in the no-box setting. Our transfer attack adds a perturbation to a watermarked image to evade multiple surrogate watermarking models trained by the attacker itself, and the perturbed watermarked image also evades the target watermarking model. Our major contribution is to show that, both theoretically and empirically, watermark-based AI-generated image detector is not robust to evasion attacks even if the attacker does not have access to the watermarking model nor the detection API.
    
[^3]: 使用多视图注意力的图像文本匹配

    Image-Text Matching with Multi-View Attention

    [https://arxiv.org/abs/2402.17237](https://arxiv.org/abs/2402.17237)

    本研究提出了一种使用多视图注意力的双流图像文本匹配方法，以解决单一表示难以全面覆盖复杂内容和缺乏交互的挑战。

    

    现有的用于图像文本匹配的双流模型在确保检索速度的同时表现出良好的性能，并受到工业界和学术界的广泛关注。这些方法使用单一表示来分别编码图像和文本，并使用余弦相似度或向量内积得到匹配分数。然而，双流模型的性能往往不太理想。一方面，单一表示难以全面覆盖复杂内容。另一方面，在这种缺乏交互的框架中，匹配多重含义是具有挑战性的，这导致信息被忽略。为了解决上述问题并促进双流模型的性能，我们提出了一种双流图像文本匹配的多视图注意力方法MVAM（多视图注意力模型）。

    arXiv:2402.17237v1 Announce Type: cross  Abstract: Existing two-stream models for image-text matching show good performance while ensuring retrieval speed and have received extensive attention from industry and academia. These methods use a single representation to encode image and text separately and get a matching score with cosine similarity or the inner product of vectors. However, the performance of the two-stream model is often sub-optimal. On the one hand, a single representation is challenging to cover complex content comprehensively. On the other hand, in this framework of lack of interaction, it is challenging to match multiple meanings which leads to information being ignored. To address the problems mentioned above and facilitate the performance of the two-stream model, we propose a multi-view attention approach for two-stream image-text matching MVAM (\textbf{M}ulti-\textbf{V}iew \textbf{A}ttention \textbf{M}odel). It first learns multiple image and text representations by
    

