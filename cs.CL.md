# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Do Large Language Model Understand Multi-Intent Spoken Language ?](https://arxiv.org/abs/2403.04481) | 该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。 |
| [^2] | [LaunchpadGPT: Language Model as Music Visualization Designer on Launchpad.](http://arxiv.org/abs/2307.04827) | 我们提出了LaunchpadGPT模型，利用语言模型生成音乐可视化设计，并展示出优于随机生成方法的效果，具有广泛的音乐可视化应用潜力。 |

# 详细

[^1]: 大型语言模型能理解多目标口语语言吗？

    Do Large Language Model Understand Multi-Intent Spoken Language ?

    [https://arxiv.org/abs/2403.04481](https://arxiv.org/abs/2403.04481)

    该研究利用大型语言模型进行口语语言多目标理解，提出了改进实体槽和子目标指令的创新技术，并展示了LLMs在多目标SLU模型方面的潜力。

    

    这项研究通过利用大型语言模型（LLMs）进行多目标口语语言理解（SLU）取得了重大进展，提出了一种在SLU环境中利用LLMs生成能力的独特方法。我们的创新技术重新配置了实体槽，专门用于LLMs在多目标SLU环境中的应用，并引入了子目标指令（SII）的概念，增强了对不同领域内复杂多目标交流的解剖和解释。由此产生的数据集，被称为LM-MixATIS和LM-MixSNIPS，是从现有基准中精心制作的。我们的研究表明，LLMs可以匹配并潜在地超越当前最先进的多目标SLU模型的能力。它进一步探讨了LLMs在各种意图配置和数据集比例下的有效性。此外，我们介绍了两个开创性的度量标准，即实体槽准确性（ESA）和Com

    arXiv:2403.04481v1 Announce Type: cross  Abstract: This study marks a significant advancement by harnessing Large Language Models (LLMs) for multi-intent spoken language understanding (SLU), proposing a unique methodology that capitalizes on the generative power of LLMs within an SLU context. Our innovative technique reconfigures entity slots specifically for LLM application in multi-intent SLU environments and introduces the concept of Sub-Intent Instruction (SII), enhancing the dissection and interpretation of intricate, multi-intent communication within varied domains. The resultant datasets, dubbed LM-MixATIS and LM-MixSNIPS, are crafted from pre-existing benchmarks. Our research illustrates that LLMs can match and potentially excel beyond the capabilities of current state-of-the-art multi-intent SLU models. It further explores LLM efficacy across various intent configurations and dataset proportions. Moreover, we introduce two pioneering metrics, Entity Slot Accuracy (ESA) and Com
    
[^2]: LaunchpadGPT: 以语言模型作为音乐可视化设计师在Launchpad上

    LaunchpadGPT: Language Model as Music Visualization Designer on Launchpad. (arXiv:2307.04827v1 [cs.SD])

    [http://arxiv.org/abs/2307.04827](http://arxiv.org/abs/2307.04827)

    我们提出了LaunchpadGPT模型，利用语言模型生成音乐可视化设计，并展示出优于随机生成方法的效果，具有广泛的音乐可视化应用潜力。

    

    Launchpad是一种乐器，用户可以通过按亮的按钮来创作和演奏音乐。为了辅助和启发Launchpad灯光效果的设计，并为初学者提供更易于使用的方法来通过这个乐器创建音乐可视化效果，我们提出了LaunchpadGPT模型，可以自动生成Launchpad上的音乐可视化设计。基于具有出色生成能力的语言模型，我们的LaunchpadGPT模型以音频音乐作为输入，并输出以视频形式表现Launchpad演奏的灯光效果（Launchpad播放视频）。我们收集Launchpad演奏视频并进行处理，以获取音乐和相应的Launchpad演奏视频帧作为提示完成对，用于训练语言模型。实验证明，所提出的方法比随机生成方法可以创造出更好的音乐可视化效果，并具有更广泛的音乐可视化应用潜力。

    Launchpad is a musical instrument that allows users to create and perform music by pressing illuminated buttons. To assist and inspire the design of the Launchpad light effect, and provide a more accessible approach for beginners to create music visualization with this instrument, we proposed the LaunchpadGPT model to generate music visualization designs on Launchpad automatically. Based on the language model with excellent generation ability, our proposed LaunchpadGPT takes an audio piece of music as input and outputs the lighting effects of Launchpad-playing in the form of a video (Launchpad-playing video). We collect Launchpad-playing videos and process them to obtain music and corresponding video frame of Launchpad-playing as prompt-completion pairs, to train the language model. The experiment result shows the proposed method can create better music visualization than random generation methods and hold the potential for a broader range of music visualization applications. Our code 
    

