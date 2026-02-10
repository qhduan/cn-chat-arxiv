# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay](https://arxiv.org/abs/2403.11852) | 本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。 |
| [^2] | [KIX: A Metacognitive Generalization Framework](https://arxiv.org/abs/2402.05346) | 人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。 |
| [^3] | [UDEEP: Edge-based Computer Vision for In-Situ Underwater Crayfish and Plastic Detection.](http://arxiv.org/abs/2401.06157) | UDEEP是一个基于边缘计算机视觉的平台，可以帮助解决入侵信号龙虾和废弃塑料对水生生态系统的挑战。 |
| [^4] | [DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing.](http://arxiv.org/abs/2310.08785) | 本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。 |
| [^5] | [YaRN: Efficient Context Window Extension of Large Language Models.](http://arxiv.org/abs/2309.00071) | YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。 |

# 详细

[^1]: 具有潜在状态推断的强化学习在自动匝道合并中的应用

    Reinforcement Learning with Latent State Inference for Autonomous On-ramp Merging under Observation Delay

    [https://arxiv.org/abs/2403.11852](https://arxiv.org/abs/2403.11852)

    本文提出了一种具有潜在状态推断的强化学习方法，用于解决自动匝道合并问题，在没有详细了解周围车辆意图或驾驶风格的情况下安全执行匝道合并任务，并考虑了观测延迟，以增强代理在动态交通状况中的决策能力。

    

    本文提出了一种解决自动匝道合并问题的新方法，其中自动驾驶车辆需要无缝地融入多车道高速公路上的车流。我们介绍了Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS)代理，旨在在没有关于周围车辆意图或驾驶风格的全面知识的情况下安全执行匝道合并任务。我们还提出了该代理的增强版AL3IS，考虑了观测延迟，使代理能够在具有车辆间通信延迟的现实环境中做出更稳健的决策。通过通过潜在状态建模环境中的不可观察方面，如其他驾驶员的意图，我们的方法增强了代理适应动态交通状况、优化合并操作并确保与其他车辆进行安全互动的能力。

    arXiv:2403.11852v1 Announce Type: cross  Abstract: This paper presents a novel approach to address the challenging problem of autonomous on-ramp merging, where a self-driving vehicle needs to seamlessly integrate into a flow of vehicles on a multi-lane highway. We introduce the Lane-keeping, Lane-changing with Latent-state Inference and Safety Controller (L3IS) agent, designed to perform the on-ramp merging task safely without comprehensive knowledge about surrounding vehicles' intents or driving styles. We also present an augmentation of this agent called AL3IS that accounts for observation delays, allowing the agent to make more robust decisions in real-world environments with vehicle-to-vehicle (V2V) communication delays. By modeling the unobservable aspects of the environment through latent states, such as other drivers' intents, our approach enhances the agent's ability to adapt to dynamic traffic conditions, optimize merging maneuvers, and ensure safe interactions with other vehi
    
[^2]: KIX: 一种元认知泛化框架

    KIX: A Metacognitive Generalization Framework

    [https://arxiv.org/abs/2402.05346](https://arxiv.org/abs/2402.05346)

    人工智能代理缺乏通用行为，需要利用结构化知识表示。该论文提出了一种元认知泛化框架KIX，通过与对象的交互学习可迁移的交互概念和泛化能力，促进了知识与强化学习的融合，为实现人工智能系统的自主和通用行为提供了潜力。

    

    人类和其他动物能够灵活解决各种任务，并且能够通过重复使用和应用长期积累的高级知识来适应新颖情境，这表现了一种泛化智能行为。但是人工智能代理更多地是专家，缺乏这种通用行为。人工智能代理需要理解和利用关键的结构化知识表示。我们提出了一种元认知泛化框架，称为Knowledge-Interaction-eXecution (KIX)，并且认为通过与对象的交互来利用类型空间可以促进学习可迁移的交互概念和泛化能力。这是将知识融入到强化学习中的一种自然方式，并有望成为人工智能系统中实现自主和通用行为的推广者。

    Humans and other animals aptly exhibit general intelligence behaviors in solving a variety of tasks with flexibility and ability to adapt to novel situations by reusing and applying high level knowledge acquired over time. But artificial agents are more of a specialist, lacking such generalist behaviors. Artificial agents will require understanding and exploiting critical structured knowledge representations. We present a metacognitive generalization framework, Knowledge-Interaction-eXecution (KIX), and argue that interactions with objects leveraging type space facilitate the learning of transferable interaction concepts and generalization. It is a natural way of integrating knowledge into reinforcement learning and promising to act as an enabler for autonomous and generalist behaviors in artificial intelligence systems.
    
[^3]: UDEEP: 基于边缘的水下信号龙虾和塑料检测的计算机视觉

    UDEEP: Edge-based Computer Vision for In-Situ Underwater Crayfish and Plastic Detection. (arXiv:2401.06157v1 [cs.CV])

    [http://arxiv.org/abs/2401.06157](http://arxiv.org/abs/2401.06157)

    UDEEP是一个基于边缘计算机视觉的平台，可以帮助解决入侵信号龙虾和废弃塑料对水生生态系统的挑战。

    

    入侵的信号龙虾对生态系统造成了不利影响。它们传播了对英国唯一的本地白爪龙虾致命的真菌型龙虾瘟疫病(Aphanomyces astaci)。入侵的信号龙虾广泛挖掘洞穴，破坏栖息地，侵蚀河岸并对水质产生不利影响，同时竞争本地物种的资源并导致本地种群下降。此外，污染也使白爪龙虾更加容易受到损害，其种群在英国某些地区下降超过90％，使其极易濒临灭绝。为了保护水生生态系统，解决入侵物种和废弃塑料对英国河流生态系统的挑战至关重要。UDEEP平台可以通过实时分类信号龙虾和塑料碎片，充当环境监测的关键角色。

    Invasive signal crayfish have a detrimental impact on ecosystems. They spread the fungal-type crayfish plague disease (Aphanomyces astaci) that is lethal to the native white clawed crayfish, the only native crayfish species in Britain. Invasive signal crayfish extensively burrow, causing habitat destruction, erosion of river banks and adverse changes in water quality, while also competing with native species for resources and leading to declines in native populations. Moreover, pollution exacerbates the vulnerability of White-clawed crayfish, with their populations declining by over 90% in certain English counties, making them highly susceptible to extinction. To safeguard aquatic ecosystems, it is imperative to address the challenges posed by invasive species and discarded plastics in the United Kingdom's river ecosystem's. The UDEEP platform can play a crucial role in environmental monitoring by performing on-the-fly classification of Signal crayfish and plastic debris while leveragi
    
[^4]: DeltaSpace:一种用于灵活文本引导图像编辑的语义对齐特征空间

    DeltaSpace: A Semantic-aligned Feature Space for Flexible Text-guided Image Editing. (arXiv:2310.08785v1 [cs.CV])

    [http://arxiv.org/abs/2310.08785](http://arxiv.org/abs/2310.08785)

    本文提出了一种名为DeltaSpace的特征空间，用于灵活文本引导图像编辑。在DeltaSpace的基础上，通过一种称为DeltaEdit的新颖框架，将CLIP视觉特征差异映射到潜在空间方向，并从CLIP预测潜在空间方向，解决了训练和推理灵活性的挑战。

    

    文本引导图像编辑面临着训练和推理灵活性的重大挑战。许多文献通过收集大量标注的图像-文本对来从头开始训练文本条件生成模型，这既昂贵又低效。然后，一些利用预训练的视觉语言模型的方法出现了，以避免数据收集，但它们仍然受到基于每个文本提示的优化或推理时的超参数调整的限制。为了解决这些问题，我们调查和确定了一个特定的空间，称为CLIP DeltaSpace，在这个空间中，两个图像的CLIP视觉特征差异与其对应的文本描述的CLIP文本特征差异在语义上是对齐的。基于DeltaSpace，我们提出了一个新颖的框架DeltaEdit，在训练阶段将CLIP视觉特征差异映射到生成模型的潜在空间方向，并从CLIP预测潜在空间方向。

    Text-guided image editing faces significant challenges to training and inference flexibility. Much literature collects large amounts of annotated image-text pairs to train text-conditioned generative models from scratch, which is expensive and not efficient. After that, some approaches that leverage pre-trained vision-language models are put forward to avoid data collection, but they are also limited by either per text-prompt optimization or inference-time hyper-parameters tuning. To address these issues, we investigate and identify a specific space, referred to as CLIP DeltaSpace, where the CLIP visual feature difference of two images is semantically aligned with the CLIP textual feature difference of their corresponding text descriptions. Based on DeltaSpace, we propose a novel framework called DeltaEdit, which maps the CLIP visual feature differences to the latent space directions of a generative model during the training phase, and predicts the latent space directions from the CLIP
    
[^5]: YaRN: 大型语言模型的高效上下文窗口扩展方法

    YaRN: Efficient Context Window Extension of Large Language Models. (arXiv:2309.00071v1 [cs.CL])

    [http://arxiv.org/abs/2309.00071](http://arxiv.org/abs/2309.00071)

    YaRN是一种高效的上下文窗口扩展方法，可以在大型语言模型中有效利用和推断比原始预训练允许的上下文长度更长的上下文，同时超越了之前的最新研究成果。

    

    旋转位置嵌入（RoPE）已被证明可以有效地编码transformer-based语言模型中的位置信息。然而，这些模型在超过它们训练的序列长度时无法泛化。我们提出了YaRN（Yet another RoPE extensioN method），一种计算高效的方法来扩展这些模型的上下文窗口，需要的tokens数量和训练步骤少于之前的方法的10倍和2.5倍。使用YaRN，我们展示了LLaMA模型可以有效地利用和推断比原始预训练允许的上下文长度更长的上下文，并且在上下文窗口扩展方面超过了之前的最新研究成果。此外，我们还展示了YaRN具有超越微调数据集有限上下文的能力。我们在https://github.com/jquesnelle/yarn上发布了使用64k和128k上下文窗口进行Fine-tuning的Llama 2 7B/13B的检查点。

    Rotary Position Embeddings (RoPE) have been shown to effectively encode positional information in transformer-based language models. However, these models fail to generalize past the sequence length they were trained on. We present YaRN (Yet another RoPE extensioN method), a compute-efficient method to extend the context window of such models, requiring 10x less tokens and 2.5x less training steps than previous methods. Using YaRN, we show that LLaMA models can effectively utilize and extrapolate to context lengths much longer than their original pre-training would allow, while also surpassing previous the state-of-the-art at context window extension. In addition, we demonstrate that YaRN exhibits the capability to extrapolate beyond the limited context of a fine-tuning dataset. We publish the checkpoints of Llama 2 7B/13B fine-tuned using YaRN with 64k and 128k context windows at https://github.com/jquesnelle/yarn
    

