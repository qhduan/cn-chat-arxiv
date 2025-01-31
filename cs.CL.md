# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning.](http://arxiv.org/abs/2211.11337) | DreamArtist采用正负prompt-tuning学习策略来生成可控的一次性文本到图像，并解决了传统方法可能会导致模型过度拟合的问题。 |

# 详细

[^1]: DreamArtist: 通过对比prompt-tuning实现可控的一次性文本到图像生成

    DreamArtist: Towards Controllable One-Shot Text-to-Image Generation via Contrastive Prompt-Tuning. (arXiv:2211.11337v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.11337](http://arxiv.org/abs/2211.11337)

    DreamArtist采用正负prompt-tuning学习策略来生成可控的一次性文本到图像，并解决了传统方法可能会导致模型过度拟合的问题。

    

    大规模文本到图像生成模型通过文本指导合成高质量、特征丰富、高分辨率的图像取得了可观的进展。然而，这些模型在处理新概念（例如新风格、物体实体等）时常常面临困难。尽管最近的尝试采用微调或prompt-tuning策略来教授预先训练的扩散模型从参考图像集中学习新概念，但它们存在过度拟合给定的参考图像，特别是在单次应用中，这对于保持生成可控性并产生多样化、高质量的图像是有害的。为了解决这个挑战，我们提出了一种简单而有效的方法DreamArtist，它采用了正负prompt-tuning学习策略。具体而言，DreamArtist结合了正负嵌入并联合训练它们。正嵌入积极地捕捉参考图像的显着特征来驱动图像生成，而负嵌入则强制模型生成多样性图像以降低过度拟合风险。

    Large-scale text-to-image generation models have achieved remarkable progress in synthesizing high-quality, feature-rich images with high resolution guided by texts. However, these models often struggle with novel concepts, eg, new styles, object entities, etc. Although recent attempts have employed fine-tuning or prompt-tuning strategies to teach the pre-trained diffusion model novel concepts from a reference image set,they have the drawback of overfitting to the given reference images, particularly in one-shot applications, which is harmful to generate diverse and high-quality images while maintaining generation controllability.  To tackle this challenge, we present a simple yet effective method called DreamArtist, which employs a positive-negative prompt-tuning learning strategy. Specifically, DreamArtist incorporates both positive and negative embeddings and jointly trains them. The positive embedding aggressively captures the salient characteristics of the reference image to drive
    

