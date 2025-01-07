# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Developing Safe and Responsible Large Language Models -- A Comprehensive Framework](https://arxiv.org/abs/2404.01399) | 该论文介绍了一种新的模型SR$_{\text{LLM}}$，旨在通过引入全面的安全风险分类法和专家标注数据集来增强大型语言模型（LLM）在语言生成中的安全性，并通过指令和参数高效微调方法有效减少了不安全内容的生成。 |
| [^2] | [Scaling Efficient LLMs](https://arxiv.org/abs/2402.14746) | 训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。 |
| [^3] | [Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models](https://arxiv.org/abs/2402.11532) | 提出了一种名为指令链（CoI）的新概念，通过逐步解决每个子任务来处理由多个子任务组成的指令，进而提高了大型语言模型（LLMs）的泛化能力和多语言摘要性能 |
| [^4] | [Sentiment-enhanced Graph-based Sarcasm Explanation in Dialogue](https://arxiv.org/abs/2402.03658) | 本论文提出了一种名为EDGE的新颖的基于图的情感增强多模态讽刺解释框架，旨在为涉及多种模态的讽刺对话生成自然语言解释。该框架克服了话语记号对情感的多样效应、视频音频情感信号与BART嵌入空间之间的差距以及话语、话语情感和视频音频情感之间的不同关系等挑战。 |
| [^5] | [AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL](https://arxiv.org/abs/2311.09830) | AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。 |
| [^6] | [Improving Summarization with Human Edits.](http://arxiv.org/abs/2310.05857) | 本文介绍了一种改进摘要生成的方法，使用人工编辑的反馈数据，并通过序列对齐（不）似然训练(SALT)技术将人工编辑数据与模型生成数据结合起来。实验证明了这种方法在医学领域摘要生成中的有效性。 |
| [^7] | [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.](http://arxiv.org/abs/2308.08747) | 该研究实证评估了大型语言模型在持续微调过程中的灾难性遗忘现象，并发现随着模型规模增加，遗忘的严重程度也加剧。与编码器-解码器模型相比，仅有解码器的模型遗忘较少并保留更多知识。此外，研究还发现LLMs可以减轻语言偏见，并且ALPACA在保留知识和容量方面具有优势。 |
| [^8] | [Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models.](http://arxiv.org/abs/2307.10236) | 本研究从不确定性的角度对大型语言模型进行了探索性研究，通过实验发现不确定性估计方法在探索和抵制大型语言模型的不良行为方面具有潜力。 |
| [^9] | [NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning.](http://arxiv.org/abs/2307.08941) | 该论文通过使用神经切向核近似MLP融合，提出了一种高效的语言模型微调方法。实验证明，这种方法能够在降低计算和存储开销的同时保持较好的模型性能。 |

# 详细

[^1]: 开发安全和负责任的大型语言模型 - 一个全面框架

    Developing Safe and Responsible Large Language Models -- A Comprehensive Framework

    [https://arxiv.org/abs/2404.01399](https://arxiv.org/abs/2404.01399)

    该论文介绍了一种新的模型SR$_{\text{LLM}}$，旨在通过引入全面的安全风险分类法和专家标注数据集来增强大型语言模型（LLM）在语言生成中的安全性，并通过指令和参数高效微调方法有效减少了不安全内容的生成。

    

    鉴于人们对大型语言模型（LLM）的安全性和风险日益关注，发展减轻这些问题的方法至关重要。我们引入了安全和负责任的大型语言模型（SR$_{\text{LLM}}$），这个模型旨在通过使用LLM来增强语言生成的安全性。我们的方法结合了一个全面的LLM安全风险分类法，并利用专家注释的数据集与这种分类法相一致。SR$_{\text{LLM}}$旨在识别潜在的不安全内容并产生良性变化。它采用基于指令的和参数高效的微调方法，使得该模型不仅有效地增强安全性，而且资源高效且易于调整。在我们对五个基准数据集和两个专有数据集进行测试后，我们观察到不安全内容生成的显著减少。此外，在实施安全措施后，出现了...

    arXiv:2404.01399v1 Announce Type: new  Abstract: Given the growing concerns around the safety and risks of Large Language Models (LLMs), it is essential to develop methods for mitigating these issues. We introduce Safe and Responsible Large Language Model (SR$_{\text{LLM}}$) , a model designed to enhance the safety of language generation using LLMs. Our approach incorporates a comprehensive LLM safety risk taxonomy and utilizes a dataset annotated by experts that align with this taxonomy. SR$_{\text{LLM}}$ is designed to identify potentially unsafe content and produce benign variations. It employs instruction-based and parameter-efficient fine-tuning methods, making the model not only effective in enhancing safety but also resource-efficient and straightforward to adjust. Through our testing on five benchmark datasets and two proprietary datasets, we observed notable reductions in the generation of unsafe content. Moreover, following the implementation of safety measures, there was a s
    
[^2]: 扩展高效的LLM模型

    Scaling Efficient LLMs

    [https://arxiv.org/abs/2402.14746](https://arxiv.org/abs/2402.14746)

    训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。

    

    训练得到的LLM模型通常是稀疏的，即大部分参数为零，这引发了关于效率的问题。为此，我们研究了高效的LLM模型，即那些在训练语料上达到所需准确度的参数最少。具体地，我们比较了当前规模下训练损失的理论和实证估计，以获得自然训练语料中独特序列数量上下界的数量。我们的结果暗示：(1)要在训练语料中表示的技能数量翻倍，需要将语料规模大约扩展三到五倍，(2)对于高效的LLM模型，参数数量$N$和自然训练语料规模$D$满足$N \sim D^{0.58}$的关系，(3)如果一个LLM模型的参数数量小于训练语料中的独特序列数量，扩展可以揭示出新的技能。

    arXiv:2402.14746v1 Announce Type: new  Abstract: Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss at current scale to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale roughly between three and five fold (2) for efficient LLMs, the number of parameters $N$ and the size $D$ of a natural training corpus scale as $N \sim D^{0.58}$ (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    
[^3]: 指令链：大型语言模型的组合指令调整

    Chain-of-Instructions: Compositional Instruction Tuning on Large Language Models

    [https://arxiv.org/abs/2402.11532](https://arxiv.org/abs/2402.11532)

    提出了一种名为指令链（CoI）的新概念，通过逐步解决每个子任务来处理由多个子任务组成的指令，进而提高了大型语言模型（LLMs）的泛化能力和多语言摘要性能

    

    使用一系列大型和多样化的指令对大型语言模型（LLMs）进行微调，提高了模型对不同任务的泛化能力，甚至对未曾见过的任务也适用。本研究提出了一种称为指令链（CoI）的新概念，其中一个指令的输出成为下一个指令的输入，就像一条链条。与解决单一指令任务的传统做法不同，我们提出的方法鼓励模型逐步解决每个子任务，直至得出最终答案。CoI调整（即使用CoI指令进行微调）提高了模型处理由多个子任务组成的指令能力。经CoI调整的模型在多语言摘要上也优于基准模型，证明....

    arXiv:2402.11532v1 Announce Type: new  Abstract: Fine-tuning large language models (LLMs) with a collection of large and diverse instructions has improved the model's generalization to different tasks, even for unseen tasks. However, most existing instruction datasets include only single instructions, and they struggle to follow complex instructions composed of multiple subtasks (Wang et al., 2023a). In this work, we propose a novel concept of compositional instructions called chain-of-instructions (CoI), where the output of one instruction becomes an input for the next like a chain. Unlike the conventional practice of solving single instruction tasks, our proposed method encourages a model to solve each subtask step by step until the final answer is reached. CoI-tuning (i.e., fine-tuning with CoI instructions) improves the model's ability to handle instructions composed of multiple subtasks. CoI-tuned models also outperformed baseline models on multilingual summarization, demonstratin
    
[^4]: 在对话中增强情感的基于图的讽刺解释

    Sentiment-enhanced Graph-based Sarcasm Explanation in Dialogue

    [https://arxiv.org/abs/2402.03658](https://arxiv.org/abs/2402.03658)

    本论文提出了一种名为EDGE的新颖的基于图的情感增强多模态讽刺解释框架，旨在为涉及多种模态的讽刺对话生成自然语言解释。该框架克服了话语记号对情感的多样效应、视频音频情感信号与BART嵌入空间之间的差距以及话语、话语情感和视频音频情感之间的不同关系等挑战。

    

    对话中的讽刺解释（SED）是一项新而具有挑战性的任务，旨在为涉及多种模态（即话语、视频和音频）的讽刺对话生成自然语言解释。尽管现有的研究基于生成式预训练语言模型BART取得了巨大成功，但它们忽视了话语、视频和音频中存在的情感，而这些情感是讽刺解释中的重要线索。事实上，由于以下三个主要挑战：1）话语记号对情感的多样效应；2）视频音频情感信号与BART的嵌入空间之间的差距；3）话语、话语情感和视频音频情感之间的不同关系，将情感融入以提升SED性能是一项非常复杂的任务。为了解决这些挑战，我们提出了一种新颖的基于图的增强情感的多模态讽刺解释框架，命名为EDGE。

    Sarcasm Explanation in Dialogue (SED) is a new yet challenging task, which aims to generate a natural language explanation for the given sarcastic dialogue that involves multiple modalities (i.e., utterance, video, and audio). Although existing studies have achieved great success based on the generative pretrained language model BART, they overlook exploiting the sentiments residing in the utterance, video and audio, which are vital clues for sarcasm explanation. In fact, it is non-trivial to incorporate sentiments for boosting SED performance, due to three main challenges: 1) diverse effects of utterance tokens on sentiments; 2) gap between video-audio sentiment signals and the embedding space of BART; and 3) various relations among utterances, utterance sentiments, and video-audio sentiments. To tackle these challenges, we propose a novel sEntiment-enhanceD Graph-based multimodal sarcasm Explanation framework, named EDGE. In particular, we first propose a lexicon-guided utterance sen
    
[^5]: AutoPlanBench: 从PDDL自动生成LLM规划器的基准测试

    AutoPlanBench: Automatically generating benchmarks for LLM planners from PDDL

    [https://arxiv.org/abs/2311.09830](https://arxiv.org/abs/2311.09830)

    AutoPlanBench是一种新方法，可以自动转换PDDL规划基准测试为文本描述，并提供了相应的基准测试数据集。研究表明，当前最好的LLM规划器在某些规划任务上表现优秀，但对于其他任务来说仍存在挑战。

    

    LLMs（逻辑-概率模型）在规划任务中的应用越来越广泛，但是它们在规划和推理方面的能力尚不明确。我们提出了AutoPlanBench，一种将PDDL中的规划基准测试自动转换为文本描述的新方法，并提供了使用我们方法创建的基准测试数据集。我们展示了最好的LLM规划器在某些规划任务上表现良好，但其他任务仍然超出了当前方法的能力范围。

    LLMs are being increasingly used for planning-style tasks, but their capabilities for planning and reasoning are poorly understood. We present AutoPlanBench, a novel method for automatically converting planning benchmarks written in PDDL into textual descriptions and offer a benchmark dataset created with our method. We show that while the best LLM planners do well on some planning tasks, others remain out of reach of current methods.
    
[^6]: 使用人工编辑改进摘要生成

    Improving Summarization with Human Edits. (arXiv:2310.05857v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05857](http://arxiv.org/abs/2310.05857)

    本文介绍了一种改进摘要生成的方法，使用人工编辑的反馈数据，并通过序列对齐（不）似然训练(SALT)技术将人工编辑数据与模型生成数据结合起来。实验证明了这种方法在医学领域摘要生成中的有效性。

    

    最近的研究表明，通过人类反馈范式学习可以产生高质量的文本。现有的工作在通用领域抽象化摘要生成中使用人类反馈来训练大型语言模型(LLMs)，并获得了超越传统似然训练的摘要质量。在本文中，我们关注一种较少探索的人类反馈形式——人工编辑。我们提出了一种新颖的技术——序列对齐（不）似然训练(SALT)，在训练循环中同时使用人工编辑和模型生成的数据。此外，我们还展示了使用现有训练数据中的基准摘要来模拟人工编辑，以及在训练后获取的模型生成摘要，以减少对昂贵的人工编辑数据的需求。在实验中，我们将人类反馈的探索从通用领域摘要生成扩展到医学领域摘要生成。我们的结果表明SALT在改进摘要生成方面的有效性。

    Recent work has shown the promise of learning with human feedback paradigms to produce human-determined high-quality text. Existing works use human feedback to train large language models (LLMs) in general domain abstractive summarization and have obtained summary quality exceeding traditional likelihood training. In this paper, we focus on a less explored form of human feedback -- Human Edits. We propose Sequence Alignment (un)Likelihood Training (SALT), a novel technique to use both the human-edited and model-generated data together in the training loop. In addition, we demonstrate simulating Human Edits with ground truth summaries coming from existing training data -Imitation edits, along with the model-generated summaries obtained after the training, to reduce the need for expensive human-edit data. In our experiments, we extend human feedback exploration from general domain summarization to medical domain summarization. Our results demonstrate the effectiveness of SALT in improv
    
[^7]: 大型语言模型在持续微调过程中的灾难性遗忘的实证研究

    An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning. (arXiv:2308.08747v1 [cs.CL])

    [http://arxiv.org/abs/2308.08747](http://arxiv.org/abs/2308.08747)

    该研究实证评估了大型语言模型在持续微调过程中的灾难性遗忘现象，并发现随着模型规模增加，遗忘的严重程度也加剧。与编码器-解码器模型相比，仅有解码器的模型遗忘较少并保留更多知识。此外，研究还发现LLMs可以减轻语言偏见，并且ALPACA在保留知识和容量方面具有优势。

    

    灾难性遗忘（CF）是机器学习中的一种现象，当模型学习新信息时，它会忘记先前学到的信息。由于大型语言模型（LLMs）显示出了出色的性能，探究LLMs在持续微调中是否存在CF是很有意义的。在这项研究中，我们从领域知识、推理和阅读理解的角度对LLMs的遗忘现象进行了实证评估。实验表明，从1b到7b的范围内，LLMs普遍存在灾难性遗忘现象，并且随着规模的增加，遗忘的严重程度也加剧。与编码器-解码器模型mT0相比，仅有解码器的模型BLOOMZ遗忘较少并保留更多知识。我们还观察到，在持续微调过程中，LLMs可以减轻语言偏见（如性别偏见）。此外，我们发现与LLAMA相比，ALPACA在保留更多知识和容量方面具有优势。

    Catastrophic forgetting (CF) is a phenomenon that occurs in machine learning when a model forgets previously learned information as it learns new information. As large language models (LLMs) have shown excellent performance, it is interesting to uncover whether CF exists in the continual fine-tuning of LLMs. In this study, we empirically evaluate the forgetting phenomenon in LLMs' knowledge, from the perspectives of domain knowledge, reasoning, and reading comprehension. The experiments demonstrate that catastrophic forgetting is generally observed in LLMs ranging from 1b to 7b. Furthermore, as the scale increases, the severity of forgetting also intensifies. Comparing the decoder-only model BLOOMZ with the encoder-decoder model mT0, BLOOMZ suffers less forgetting and maintains more knowledge. We also observe that LLMs can mitigate language bias (e.g. gender bias) during continual fine-tuning. Moreover, we find that ALPACA can maintain more knowledge and capacity compared with LLAMA du
    
[^8]: 三思而后行：大型语言模型不确定性测量的探索性研究

    Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models. (arXiv:2307.10236v1 [cs.SE])

    [http://arxiv.org/abs/2307.10236](http://arxiv.org/abs/2307.10236)

    本研究从不确定性的角度对大型语言模型进行了探索性研究，通过实验发现不确定性估计方法在探索和抵制大型语言模型的不良行为方面具有潜力。

    

    大型语言模型（LLMs）的最近性能突破为众多工业应用和领域提供了新的机遇。然而，LLMs的错误生成，如虚假预测、错误信息和幻觉，也引发了对LLMs可靠性的严重关注，尤其在对安全、可靠性有敏感的场景中，可能阻碍其在实际中的应用。尽管不确定性估计已经显示出其在解释一般机器学习（ML）模型的预测风险方面的潜力，但关于它是否以及在多大程度上有助于探索LLMs的能力和抵制其不良行为方面知之甚少。为了弥合这一差距，本文从不确定性的角度开展了关于LLMs风险评估的探索性研究。具体来说，我们使用12种不确定性估计方法和4个LLMs在4个重要的自然语言处理（NLP）任务上进行实验，以调查不确定性在探索LLMs能力和对抗其不良行为方面的程度。

    The recent performance leap of Large Language Models (LLMs) opens up new opportunities across numerous industrial applications and domains. However, erroneous generations, such as false predictions, misinformation, and hallucination made by LLMs, have also raised severe concerns for the trustworthiness of LLMs', especially in safety-, security- and reliability-sensitive scenarios, potentially hindering real-world adoptions. While uncertainty estimation has shown its potential for interpreting the prediction risks made by general machine learning (ML) models, little is known about whether and to what extent it can help explore an LLM's capabilities and counteract its undesired behavior. To bridge the gap, in this paper, we initiate an exploratory study on the risk assessment of LLMs from the lens of uncertainty. In particular, we experiment with twelve uncertainty estimation methods and four LLMs on four prominent natural language processing (NLP) tasks to investigate to what extent unc
    
[^9]: NTK-近似MLP融合用于高效的语言模型微调

    NTK-approximating MLP Fusion for Efficient Language Model Fine-tuning. (arXiv:2307.08941v1 [cs.LG])

    [http://arxiv.org/abs/2307.08941](http://arxiv.org/abs/2307.08941)

    该论文通过使用神经切向核近似MLP融合，提出了一种高效的语言模型微调方法。实验证明，这种方法能够在降低计算和存储开销的同时保持较好的模型性能。

    

    在许多自然语言处理应用中，微调预训练语言模型(PLM)已成为主要策略。然而，即使是微调PLM和进行推理也是昂贵的，特别是在计算能力较低的边缘设备上。已经广泛研究了一些通用的方法（例如量化和蒸馏）来减少PLM微调的计算/存储开销，但很少有一次性压缩技术被探索。在本文中，我们研究了多层感知器(MLP)模块中预训练语言模型(PLM)的神经切向核(NTK)，并提出通过NTK近似MLP融合来创建一个轻量级的PLM。为实现这一目标，我们将MLP重新视为一束子MLP，并将它们聚类为给定数量的质心，然后将其恢复为压缩的MLP，并意外地显示出对原始PLM的NTK进行良好近似的效果。在自然语言处理数据集上进行了大量实验以验证PLM微调的效果。

    Fine-tuning a pre-trained language model (PLM) emerges as the predominant strategy in many natural language processing applications. However, even fine-tuning the PLMs and doing inference are expensive, especially on edge devices with low computing power. Some general approaches (e.g. quantization and distillation) have been widely studied to reduce the compute/memory of PLM fine-tuning, while very few one-shot compression techniques are explored. In this paper, we investigate the neural tangent kernel (NTK)--which reveals the gradient descent dynamics of neural networks--of the multilayer perceptrons (MLP) modules in a PLM and propose to coin a lightweight PLM through NTK-approximating MLP fusion. To achieve this, we reconsider the MLP as a bundle of sub-MLPs, and cluster them into a given number of centroids, which can then be restored as a compressed MLP and surprisingly shown to well approximate the NTK of the original PLM. Extensive experiments of PLM fine-tuning on both natural l
    

