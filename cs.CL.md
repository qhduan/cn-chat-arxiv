# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [YaRN: Efficient Context Window Extension of Large Language Models.](http://arxiv.org/abs/2309.00071) | YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。 |
| [^2] | [Cross-Modal Retrieval for Motion and Text via MildTriple Loss.](http://arxiv.org/abs/2305.04195) | 本论文提出了一个创新模型，使用MildTriple Loss捕捉长期依赖并模拟跨模态人类动作序列与文本检索任务，具有重要的应用价值。 |

# 详细

[^1]: YaRN: 大型语言模型的高效上下文窗口扩展方法

    YaRN: Efficient Context Window Extension of Large Language Models. (arXiv:2309.00071v1 [cs.CL])

    [http://arxiv.org/abs/2309.00071](http://arxiv.org/abs/2309.00071)

    YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。

    

    旋转位置嵌入（RoPE）已被证明可以有效地编码transformer-based语言模型中的位置信息。然而，这些模型在超过它们训练的序列长度时无法泛化。我们提出了YaRN（Yet another RoPE extensioN method），一种计算高效的方法来扩展这些模型的上下文窗口，需要的tokens数量和训练步骤少于之前的方法的10倍和2.5倍。使用YaRN，我们展示了LLaMA模型可以有效地利用和推断比原始预训练允许的上下文长度更长的上下文，并且在上下文窗口扩展方面超过了之前的最新研究成果。此外，我们还展示了YaRN具有超越微调数据集有限上下文的能力。我们在https://github.com/jquesnelle/yarn上发布了使用64k和128k上下文窗口进行Fine-tuning的Llama 2 7B/13B的检查点。

    Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. We publish the checkpoints of Llama 2 7B/13B fine-tuned using YaRN with 64k and 128k context windows at https://github.com/jquesnelle/yarn
    
[^2]: MildTriple Loss模型下的运动和文本跨模态检索

    Cross-Modal Retrieval for Motion and Text via MildTriple Loss. (arXiv:2305.04195v1 [cs.CV])

    [http://arxiv.org/abs/2305.04195](http://arxiv.org/abs/2305.04195)

    本论文提出了一个创新模型，使用MildTriple Loss捕捉长期依赖并模拟跨模态人类动作序列与文本检索任务，具有重要的应用价值。

    

    跨模态检索已成为计算机视觉和自然语言处理中的重要研究课题，随着图像文本和视频文本检索技术的进步。尽管在虚拟现实等广泛应用中具有重要价值，但人类动作序列与文本之间的跨模态检索尚未引起足够的关注。这个任务存在一些挑战，包括对两种语言的共同建模，要求从文本中理解以人为中心的信息，并从三维人体运动序列中学习行为特征。以往的运动数据建模主要依赖于自回归特征提取器，这可能会遗忘以前的信息，而我们提出了一种创新模型，其中包括简单而强大的基于变换器的运动和文本编码器，可以从两种不同的模态中学习表示并捕捉长期依赖

    Cross-modal retrieval has become a prominent research topic in computer vision and natural language processing with advances made in image-text and video-text retrieval technologies. However, cross-modal retrieval between human motion sequences and text has not garnered sufficient attention despite the extensive application value it holds, such as aiding virtual reality applications in better understanding users' actions and language. This task presents several challenges, including joint modeling of the two modalities, demanding the understanding of person-centered information from text, and learning behavior features from 3D human motion sequences. Previous work on motion data modeling mainly relied on autoregressive feature extractors that may forget previous information, while we propose an innovative model that includes simple yet powerful transformer-based motion and text encoders, which can learn representations from the two different modalities and capture long-term dependencie
    

