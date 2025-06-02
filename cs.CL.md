# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers](https://arxiv.org/abs/2403.02839) | 精调评判模型在领域内测试上表现出色，但泛化能力和公平性不及GPT4。 |
| [^2] | [Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space](https://arxiv.org/abs/2303.14537) | 深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。 |
| [^3] | [Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models.](http://arxiv.org/abs/2401.08491) | 这项研究研究了对比学习目标的集成到微调大型语言模型中，以解决其产生不可取内容的问题，并展示了在清洁领域中有效减少有害内容生成的方法。 |
| [^4] | [StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis.](http://arxiv.org/abs/2312.10741) | StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。 |

# 详细

[^1]: 作为评判器的LLM的实证研究：精调评判器模型是特定任务的分类器

    An Empirical Study of LLM-as-a-Judge for LLM Evaluation: Fine-tuned Judge Models are Task-specific Classifiers

    [https://arxiv.org/abs/2403.02839](https://arxiv.org/abs/2403.02839)

    精调评判模型在领域内测试上表现出色，但泛化能力和公平性不及GPT4。

    

    最近，利用大型语言模型（LLM）评估其他LLM质量的趋势日益增长。许多研究采用专有的闭源模型，尤其是GPT4，作为评估器。另外，其他研究利用开源LLM来精调评判模型作为评估器。在本研究中，我们对不同的评判模型进行了实证研究。我们的发现表明，尽管精调的评判模型在领域内测试集上能够达到较高的准确性，甚至超过GPT4，但它们本质上是特定任务的分类器，其泛化能力和公平性远低于GPT4。

    arXiv:2403.02839v1 Announce Type: new  Abstract: Recently, there has been a growing trend of utilizing Large Language Model (LLM) to evaluate the quality of other LLMs. Many studies have employed proprietary close-source models, especially GPT4, as the evaluator. Alternatively, other works have fine-tuned judge models based on open-source LLMs as the evaluator. In this study, we conduct an empirical study of different judge models on their evaluation capability. Our findings indicate that although the fine-tuned judge models achieve high accuracy on in-domain test sets, even surpassing GPT4, they are inherently task-specific classifiers, and their generalizability and fairness severely underperform GPT4.
    
[^2]: 深度增强：在激活空间中使用自监督学习进行数据增强

    Deep Augmentation: Self-Supervised Learning with Transformations in Activation Space

    [https://arxiv.org/abs/2303.14537](https://arxiv.org/abs/2303.14537)

    深度增强是一种利用dropout或PCA在神经网络中转换目标层的方法，有效改善性能和泛化能力。在对比学习任务中，在Transformers、ResNets和图神经网络等基础模型上，通过深度增强实现了显著的性能提升，但在监督问题上效果相反。

    

    我们提出了一种称为深度增强的方法，通过使用辍学或PCA来转换神经网络中的目标层，以提高性能和泛化能力。我们通过在自然语言处理、计算机视觉和图学习中的对比学习任务上进行大量实验来展示深度增强。 我们观察到在对比学习的基础模型中，如Transformers、ResNets和图神经网络上深度增强能够带来显著的性能提升，但在相应的监督问题上观察到相反的效果。 我们的分析表明，深度增强减轻了层之间的相互适应，即"崩溃"形式的问题。 我们利用这一观察结果制定了一种选择目标层的方法；特别是，我们的实验表明，用深度增强定位更深层次的层要优于增强输入数据。 这种方法的简单网络和模态无关性使其

    arXiv:2303.14537v2 Announce Type: replace-cross  Abstract: We introduce Deep Augmentation, an approach to implicit data augmentation using dropout or PCA to transform a targeted layer within a neural network to improve performance and generalization. We demonstrate Deep Augmentation through extensive experiments on contrastive learning tasks in NLP, computer vision, and graph learning. We observe substantial performance gains with Transformers, ResNets, and Graph Neural Networks as the underlying models in contrastive learning, but observe inverse effects on the corresponding supervised problems. Our analysis suggests that Deep Augmentation alleviates co-adaption between layers, a form of "collapse." We use this observation to formulate a method for selecting which layer to target; in particular, our experimentation reveals that targeting deeper layers with Deep Augmentation outperforms augmenting the input data. The simple network- and modality-agnostic nature of this approach enables
    
[^3]: 对比困惑度在受控生成中的应用：清洁大型语言模型

    Contrastive Perplexity for Controlled Generation: An Application in Detoxifying Large Language Models. (arXiv:2401.08491v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.08491](http://arxiv.org/abs/2401.08491)

    这项研究研究了对比学习目标的集成到微调大型语言模型中，以解决其产生不可取内容的问题，并展示了在清洁领域中有效减少有害内容生成的方法。

    

    大型语言模型产生不可取和事实不正确的内容在很大程度上是一个挑战和未解决的问题。本文研究了对比学习目标的集成，用于微调语言模型以进行隐式知识编辑和受控文本生成。通过对比方式优化训练目标，即对齐文本的困惑度。为了以自监督的方式训练模型，我们利用现成的语言模型来生成训练数据。我们展示了在清洁领域的适用性。在此过程中，所提出的方法显著减少了生成有害内容的数量，同时保留了对于常识推理和阅读理解等下游任务的实用性。所提出的方法在概念上简单但经验上强大。

    The generation of undesirable and factually incorrect content of large language models poses a significant challenge and remains largely an unsolved issue. This paper studies the integration of a contrastive learning objective for fine-tuning LLMs for implicit knowledge editing and controlled text generation. Optimizing the training objective entails aligning text perplexities in a contrastive fashion. To facilitate training the model in a self-supervised fashion, we leverage an off-the-shelf LLM for training data generation. We showcase applicability in the domain of detoxification. Herein, the proposed approach leads to a significant decrease in the generation of toxic content while preserving general utility for downstream tasks such as commonsense reasoning and reading comprehension. The proposed approach is conceptually simple but empirically powerful.
    
[^4]: StyleSinger: 针对领域外演唱声音合成的风格转移

    StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis. (arXiv:2312.10741v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2312.10741](http://arxiv.org/abs/2312.10741)

    StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。

    

    针对领域外演唱声音合成（SVS）的风格转移专注于生成高质量的演唱声音，该声音具有从参考演唱声音样本中衍生的未见风格（如音色、情感、发音和发音技巧）。然而，模拟演唱声音风格的精细差异是一项艰巨的任务，因为演唱声音具有非常高的表现力。此外，现有的SVS方法在领域外场景中合成的演唱声音质量下降，因为它们基于训练阶段可辨别出目标声音属性的假设。为了克服这些挑战，我们提出了StyleSinger，这是第一个用于领域外参考演唱声音样本的零样式转移的演唱声音合成模型。StyleSinger采用了两种关键方法以提高效果：1）残差风格适配器（RSA），它使用残差量化模块来捕捉多样的风格特征。

    Style transfer for out-of-domain (OOD) singing voice synthesis (SVS) focuses on generating high-quality singing voices with unseen styles (such as timbre, emotion, pronunciation, and articulation skills) derived from reference singing voice samples. However, the endeavor to model the intricate nuances of singing voice styles is an arduous task, as singing voices possess a remarkable degree of expressiveness. Moreover, existing SVS methods encounter a decline in the quality of synthesized singing voices in OOD scenarios, as they rest upon the assumption that the target vocal attributes are discernible during the training phase. To overcome these challenges, we propose StyleSinger, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference singing voice samples. StyleSinger incorporates two critical approaches for enhanced effectiveness: 1) the Residual Style Adaptor (RSA) which employs a residual quantization module to capture diverse style character
    

