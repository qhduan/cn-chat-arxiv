# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MMoE: Robust Spoiler Detection with Multi-modal Information and Domain-aware Mixture-of-Experts](https://arxiv.org/abs/2403.05265) | 提出了MMoE，一个利用多模态信息进行剧透检测的网络，并采用专家混合技术来增强领域泛化能力。 |
| [^2] | [AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling](https://arxiv.org/abs/2402.12226) | AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。 |
| [^3] | [Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts.](http://arxiv.org/abs/2401.14295) | 这篇论文探讨了结合结构的提示工程在提高大型语言模型推理性能方面的前景，通过思维链、思维树或思维图的设计来引导整体推理过程。通过大量实例，这种范式显著增强了模型在多个任务中的能力。总的来说，论文提供了一个通用蓝图，为未来的发展铺平道路。 |
| [^4] | [MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer.](http://arxiv.org/abs/2303.06298) | MLP-SRGAN是一种单维超分辨率GAN，使用MLP-Mixer和卷积层进行上采样，可用于FLAIR MRI图像的超分辨率重建，提出了新的图像质量度量方法。 |

# 详细

[^1]: MMoE: 多模态信息和领域感知专家混合的鲁棒剧透检测

    MMoE: Robust Spoiler Detection with Multi-modal Information and Domain-aware Mixture-of-Experts

    [https://arxiv.org/abs/2403.05265](https://arxiv.org/abs/2403.05265)

    提出了MMoE，一个利用多模态信息进行剧透检测的网络，并采用专家混合技术来增强领域泛化能力。

    

    在线电影评论网站对于电影信息和讨论是非常有价值的。然而，大量的剧透评论会影响观影体验，因此剧透检测变得非常重要。先前的方法通常只关注评论的文本内容，忽略了平台中信息的异质性。为了解决这个问题，我们提出了MMoE，一个利用多模态信息进行剧透检测的网络，并采用专家混合技术来增强领域泛化能力。MMoE首先从用户-电影网络中提取图表、文本和元数据特征，分别从评论的文本内容和评论的元数据中提取信息。为了处理特定类型电影评论中的剧透语言。

    arXiv:2403.05265v1 Announce Type: new  Abstract: Online movie review websites are valuable for information and discussion about movies. However, the massive spoiler reviews detract from the movie-watching experience, making spoiler detection an important task. Previous methods simply focus on reviews' text content, ignoring the heterogeneity of information in the platform. For instance, the metadata and the corresponding user's information of a review could be helpful. Besides, the spoiler language of movie reviews tends to be genre-specific, thus posing a domain generalization challenge for existing methods. To this end, we propose MMoE, a multi-modal network that utilizes information from multiple modalities to facilitate robust spoiler detection and adopts Mixture-of-Experts to enhance domain generalization. MMoE first extracts graph, text, and meta feature from the user-movie network, the review's textual content, and the review's metadata respectively. To handle genre-specific spo
    
[^2]: AnyGPT：统一的多模式离散序列建模语言模型

    AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling

    [https://arxiv.org/abs/2402.12226](https://arxiv.org/abs/2402.12226)

    AnyGPT是一个统一的多模态语言模型，通过离散表示实现各种模态的统一处理，能够在不改变大型语言模型架构或训练方式的情况下稳定训练，为新模态的无缝整合提供了可能。

    

    我们介绍了 AnyGPT，这是一个任意多模式语言模型，利用离散表示统一处理各种模态，包括语音、文本、图像和音乐。AnyGPT 可以稳定训练，无需对当前大型语言模型（LLM）架构或训练范式进行任何改动。相反，它仅依赖于数据级预处理，促进了新模态的无缝集成到LLM中，类似于新语言的整合。我们构建了一个多模式文本中心的数据集，用于多模式对齐预训练。利用生成模型，我们合成了第一个大规模任意多模式指令数据集。它包括108k个多轮对话示例，精细地交织各种模态，从而使模型能够处理多模态输入和输出的任意组合。实验结果表明，AnyGPT能够促进...

    arXiv:2402.12226v1 Announce Type: cross  Abstract: We introduce AnyGPT, an any-to-any multimodal language model that utilizes discrete representations for the unified processing of various modalities, including speech, text, images, and music. AnyGPT can be trained stably without any alterations to the current large language model (LLM) architecture or training paradigms. Instead, it relies exclusively on data-level preprocessing, facilitating the seamless integration of new modalities into LLMs, akin to the incorporation of new languages. We build a multimodal text-centric dataset for multimodal alignment pre-training. Utilizing generative models, we synthesize the first large-scale any-to-any multimodal instruction dataset. It consists of 108k samples of multi-turn conversations that intricately interweave various modalities, thus equipping the model to handle arbitrary combinations of multimodal inputs and outputs. Experimental results demonstrate that AnyGPT is capable of facilitat
    
[^3]: 推理的拓扑学：揭秘思维链、树和图

    Topologies of Reasoning: Demystifying Chains, Trees, and Graphs of Thoughts. (arXiv:2401.14295v1 [cs.CL])

    [http://arxiv.org/abs/2401.14295](http://arxiv.org/abs/2401.14295)

    这篇论文探讨了结合结构的提示工程在提高大型语言模型推理性能方面的前景，通过思维链、思维树或思维图的设计来引导整体推理过程。通过大量实例，这种范式显著增强了模型在多个任务中的能力。总的来说，论文提供了一个通用蓝图，为未来的发展铺平道路。

    

    自然语言处理（NLP）领域近年来取得了显著进展，特别是在通过创新的提示技术提高大型语言模型（LLM）性能方面。其中，与结构相结合的提示工程被视为一种有前途的范式，其设计如思维链、思维树或思维图等，通过结构指导整体LLM推理过程。通过大量实例的说明，这种范式显著增强了LLM在逻辑或数学推理、规划或创造性写作等各种任务中的能力。为了方便理解这个不断发展的领域并为未来的发展铺平道路，我们设计了一个有效和高效的LLM推理方案的通用蓝图。为此，我们对提示执行流程进行了深入分析，澄清并明确定义了不同的概念。然后我们建立第一个分类系统

    The field of natural language processing (NLP) has witnessed significant progress in recent years, with a notable focus on improving large language models' (LLM) performance through innovative prompting techniques. Among these, prompt engineering coupled with structures has emerged as a promising paradigm, with designs such as Chain-of-Thought, Tree of Thoughts, or Graph of Thoughts, in which the overall LLM reasoning is guided by a structure such as a graph. As illustrated with numerous examples, this paradigm significantly enhances the LLM's capability to solve numerous tasks, ranging from logical or mathematical reasoning to planning or creative writing. To facilitate the understanding of this growing field and pave the way for future developments, we devise a general blueprint for effective and efficient LLM reasoning schemes. For this, we conduct an in-depth analysis of the prompt execution pipeline, clarifying and clearly defining different concepts. We then build the first taxon
    
[^4]: MLP-SRGAN: 使用MLP-Mixer的单维超分辨率GAN

    MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer. (arXiv:2303.06298v1 [cs.CV])

    [http://arxiv.org/abs/2303.06298](http://arxiv.org/abs/2303.06298)

    MLP-SRGAN是一种单维超分辨率GAN，使用MLP-Mixer和卷积层进行上采样，可用于FLAIR MRI图像的超分辨率重建，提出了新的图像质量度量方法。

    MLP-SRGAN is a single-dimension Super Resolution GAN that utilizes MLP-Mixers and convolutional layers for upsampling, and can be used for super-resolution reconstruction of FLAIR MRI images. New image quality metrics were proposed.

    我们提出了一种新的架构，称为MLP-SRGAN，它是一种单维超分辨率生成对抗网络（SRGAN），利用多层感知器混合器（MLP-Mixer）以及卷积层在切片方向上进行上采样。 MLP-SRGAN使用MSSEG2挑战数据集中的高分辨率（HR）FLAIR MRI进行训练和验证。该方法应用于三个低空间分辨率的多中心FLAIR数据集（CAIN，ADNI，CCNA）的图像，以检查在保留（未见）临床数据上的性能。将上采样结果与几种最先进的SR网络进行比较。对于具有高分辨率（HR）基本事实的图像，使用峰值信噪比（PSNR）和结构相似性指数（SSIM）来衡量上采样性能。提出了几种新的结构，无参考图像质量度量，以在缺乏基础事实的情况下量化锐度（边缘强度），噪声（熵）和模糊度（低频信息）。

    We propose a novel architecture called MLP-SRGAN, which is a single-dimension Super Resolution Generative Adversarial Network (SRGAN) that utilizes Multi-Layer Perceptron Mixers (MLP-Mixers) along with convolutional layers to upsample in the slice direction. MLP-SRGAN is trained and validated using high resolution (HR) FLAIR MRI from the MSSEG2 challenge dataset. The method was applied to three multicentre FLAIR datasets (CAIN, ADNI, CCNA) of images with low spatial resolution in the slice dimension to examine performance on held-out (unseen) clinical data. Upsampled results are compared to several state-of-the-art SR networks. For images with high resolution (HR) ground truths, peak-signal-to-noise-ratio (PSNR) and structural similarity index (SSIM) are used to measure upsampling performance. Several new structural, no-reference image quality metrics were proposed to quantify sharpness (edge strength), noise (entropy), and blurriness (low frequency information) in the absence of groun
    

