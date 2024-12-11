# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SciNews: From Scholarly Complexities to Public Narratives -- A Dataset for Scientific News Report Generation](https://arxiv.org/abs/2403.17768) | 科学新闻报道生成的自动化提高了学术见解的可访问性，该研究提出了一个包含学术出版物和相应科学新闻报道的数据集，用于探索自动生成科学新闻报道的可能性。 |
| [^2] | [S+t-SNE - Bringing dimensionality reduction to data streams](https://arxiv.org/abs/2403.17643) | S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。 |
| [^3] | [Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision](https://arxiv.org/abs/2403.09472) | 通过从更简单的任务学习，实现对更难推理任务的有效泛化，提出了一种可扩展对齐方法。 |
| [^4] | [A Benchmark of Domain-Adapted Large Language Models for Generating Brief Hospital Course Summaries](https://arxiv.org/abs/2403.05720) | 介绍了一个新的基准测试，评估了用于生成简要住院病程摘要的大语言模型在健康保健领域中的性能并提出相应的自适应策略 |
| [^5] | [BPDec: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining](https://arxiv.org/abs/2401.15861) | 本文揭示了BPDec（BERT预训练解码器）的潜力，强调增强的掩码语言建模解码器设计及研究在BERT预训练中的重要性。 |
| [^6] | [Corrupting Convolution-based Unlearnable Datasets with Pixel-based Image Transformations](https://arxiv.org/abs/2311.18403) | 研究者提出了一种基于卷积的不可学习数据集，该数据集使得现有的防御方法都失效，提出通过增加特定度量来减轻不可学习效果。 |
| [^7] | [Mitigating Feature Gap for Adversarial Robustness by Feature Disentanglement.](http://arxiv.org/abs/2401.14707) | 这项研究提出了一种通过特征解缠来缓解对抗鲁棒性中特征差距的方法，该方法明确建模和消除导致特征差距的潜在特征，有效提升了鲁棒性。 |
| [^8] | [Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection.](http://arxiv.org/abs/2308.11480) | 这项研究对机器学习中分布外检测方法进行了评估，发现现有方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不稳定。 |
| [^9] | [ChatGPT vs Human-authored Text: Insights into Controllable Text Summarization and Sentence Style Transfer.](http://arxiv.org/abs/2306.07799) | 本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众和写作风格。研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在一些特征上与人类样本有所不同，有时会包含事实错误或幻觉。 |
| [^10] | [Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization.](http://arxiv.org/abs/2306.06805) | 提出了一种名为MACO的简单方法，通过优化相位谱生成图像，同时保持幅度恒定，解决了特征可视化在深度神经网络上的挑战，并实现了高质量和高效的可解释特征可视化。 |
| [^11] | [Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network.](http://arxiv.org/abs/2306.01631) | 本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。 |
| [^12] | [Incorporating Distributions of Discourse Structure for Long Document Abstractive Summarization.](http://arxiv.org/abs/2305.16784) | 本文提出了一种名为'RSTformer'的摘要模型，该模型全面融合了话语关系类型和不确定性，并以修辞结构理论为基础，经过严格评估，表现明显优于现有的模型。 |
| [^13] | [Rethinking Alignment and Uniformity in Unsupervised Image Semantic Segmentation.](http://arxiv.org/abs/2211.14513) | 本文分析了非监督图像语义分割中困扰UISS模型的特征对齐和特征均匀性问题，提出了Semantic Attention Network(SAN) 模型，包含一个新模块 semantic attention（SEAT），以动态生成逐像素和语义特征。实验结果表明，这一非监督分割框架专注于捕捉语义表示，在多个语义分割基准测试中表现优异。 |

# 详细

[^1]: 从学术复杂性到公众叙事：科学新闻报道生成的数据集

    SciNews: From Scholarly Complexities to Public Narratives -- A Dataset for Scientific News Report Generation

    [https://arxiv.org/abs/2403.17768](https://arxiv.org/abs/2403.17768)

    科学新闻报道生成的自动化提高了学术见解的可访问性，该研究提出了一个包含学术出版物和相应科学新闻报道的数据集，用于探索自动生成科学新闻报道的可能性。

    

    科学新闻报道作为一个桥梁，巧妙地将复杂的研究文章翻译成与更广泛的公众 resonant 的报道。这种叙事的自动生成增强了学术见解的可访问性。在本文中，我们提出了一个新的语料库来促进这种范式的发展。我们的语料库包括九个学科领域中学术出版物及其相应科学新闻报道的平行编译。为了证明我们数据集的实用性和可靠性，我们进行了广泛分析，突出了科学新闻叙事和学术文稿之间的可读性和简洁性差异。我们使用最先进的文本生成模型基准测试我们的数据集。评估过程包括自动评估和人工评估，为未来探索自动生成科学新闻报道打下了基础。

    arXiv:2403.17768v1 Announce Type: cross  Abstract: Scientific news reports serve as a bridge, adeptly translating complex research articles into reports that resonate with the broader public. The automated generation of such narratives enhances the accessibility of scholarly insights. In this paper, we present a new corpus to facilitate this paradigm development. Our corpus comprises a parallel compilation of academic publications and their corresponding scientific news reports across nine disciplines. To demonstrate the utility and reliability of our dataset, we conduct an extensive analysis, highlighting the divergences in readability and brevity between scientific news narratives and academic manuscripts. We benchmark our dataset employing state-of-the-art text generation models. The evaluation process involves both automatic and human evaluation, which lays the groundwork for future explorations into the automated generation of scientific news reports. The dataset and code related 
    
[^2]: S+t-SNE - 将降维引入数据流

    S+t-SNE - Bringing dimensionality reduction to data streams

    [https://arxiv.org/abs/2403.17643](https://arxiv.org/abs/2403.17643)

    S+t-SNE是t-SNE算法的改进版本，在处理数据流时具有增量更新和盲目漂移管理的特点，能够实现高效的降维和信息可视化。

    

    我们提出了S+t-SNE，这是t-SNE算法的一种改进，旨在处理无限数据流。S+t-SNE的核心思想是随着新数据的到来逐步更新t-SNE嵌入，确保可扩展性和适应性，以处理流式场景。通过在每一步选择最重要的点，该算法确保可扩展性同时保持信息可视化。采用盲目方法进行漂移管理调整嵌入空间，促进不断可视化不断发展的数据动态。我们的实验评估证明了S+t-SNE的有效性和效率。结果突显了其在流式场景中捕捉模式的能力。我们希望我们的方法为研究人员和从业者提供一个实时工具，用于理解和解释高维数据。

    arXiv:2403.17643v1 Announce Type: new  Abstract: We present S+t-SNE, an adaptation of the t-SNE algorithm designed to handle infinite data streams. The core idea behind S+t-SNE is to update the t-SNE embedding incrementally as new data arrives, ensuring scalability and adaptability to handle streaming scenarios. By selecting the most important points at each step, the algorithm ensures scalability while keeping informative visualisations. Employing a blind method for drift management adjusts the embedding space, facilitating continuous visualisation of evolving data dynamics. Our experimental evaluations demonstrate the effectiveness and efficiency of S+t-SNE. The results highlight its ability to capture patterns in a streaming scenario. We hope our approach offers researchers and practitioners a real-time tool for understanding and interpreting high-dimensional data.
    
[^3]: 易于难的泛化：超越人类监督的可扩展对齐

    Easy-to-Hard Generalization: Scalable Alignment Beyond Human Supervision

    [https://arxiv.org/abs/2403.09472](https://arxiv.org/abs/2403.09472)

    通过从更简单的任务学习，实现对更难推理任务的有效泛化，提出了一种可扩展对齐方法。

    

    当前人工智能对齐方法依赖于人类提供的演示或判断，由于这种方法，AI系统学习到的能力将受到人类能力的上界限制。这就带来了一个具有挑战性的研究问题：当系统的能力超过人类水平时，我们如何继续改进这些系统？本文在解决难度推理任务（如4-5级数学问题）的背景下回答了这个问题，通过从更简单的任务（如1-3级数学问题）中学习人类注释，我们将其称为“易于难的泛化”。我们的关键观点是，一个在更简单任务的监督下训练的评估器（奖励模型）可以有效地用于评分更难任务的候选解决方案，从而促进在不同难度任务间的易于难的泛化。基于这一观点，我们提出了一种新的可扩展对齐方法，首先训练处理督导

    arXiv:2403.09472v1 Announce Type: cross  Abstract: Current AI alignment methodologies rely on human-provided demonstrations or judgments, and the learned capabilities of AI systems would be upper-bounded by human capabilities as a result. This raises a challenging research question: How can we keep improving the systems when their capabilities have surpassed the levels of humans? This paper answers this question in the context of tackling hard reasoning tasks (e.g., level 4-5 MATH problems) via learning from human annotations on easier tasks (e.g., level 1-3 MATH problems), which we term as \textit{easy-to-hard generalization}. Our key insight is that an evaluator (reward model) trained on supervisions for easier tasks can be effectively used for scoring candidate solutions of harder tasks and hence facilitating easy-to-hard generalization over different levels of tasks. Based on this insight, we propose a novel approach to scalable alignment, which firstly trains the process-supervise
    
[^4]: 用于生成简要住院病程摘要的领域自适应大语言模型的基准测试

    A Benchmark of Domain-Adapted Large Language Models for Generating Brief Hospital Course Summaries

    [https://arxiv.org/abs/2403.05720](https://arxiv.org/abs/2403.05720)

    介绍了一个新的基准测试，评估了用于生成简要住院病程摘要的大语言模型在健康保健领域中的性能并提出相应的自适应策略

    

    简要住院病程（BHC）摘要是通过总结临床记录而生成的常见临床文件。虽然大型语言模型（LLMs）在自动化实际任务方面展现出显著能力，但它们在医疗应用（如BHC合成）中的能力尚未得到展示。为了使LLMs能够适应BHC合成，我们引入了一个新颖的基准测试，其中包含从MIMIC-IV记录中提取的经过预处理的数据集，封装了临床记录和简要住院病程（BHC）对。我们评估了两个通用LLMs和三个医疗领域适应的LLMs的性能，以改进从临床记录生成BHC。我们使用临床记录作为输入来生成BHC，采用基于提示的（使用上下文学习）和基于微调的自适应策略来应用于三个开源LLMs（Clinical-T5-Large，Llama2-13B，FLAN-UL2）和两个专有LLMs（GPT-3.5，GPT-4）。我们定量评估了性能。

    arXiv:2403.05720v1 Announce Type: cross  Abstract: Brief hospital course (BHC) summaries are common clinical documents generated by summarizing clinical notes. While large language models (LLMs) depict remarkable capabilities in automating real-world tasks, their capabilities for healthcare applications such as BHC synthesis have not been shown. To enable the adaptation of LLMs for BHC synthesis, we introduce a novel benchmark consisting of a pre-processed dataset extracted from MIMIC-IV notes, encapsulating clinical note, and brief hospital course (BHC) pairs. We assess the performance of two general-purpose LLMs and three healthcare-adapted LLMs to improve BHC synthesis from clinical notes. Using clinical notes as input for generating BHCs, we apply prompting-based (using in-context learning) and fine-tuning-based adaptation strategies to three open-source LLMs (Clinical-T5-Large, Llama2-13B, FLAN-UL2) and two proprietary LLMs (GPT-3.5, GPT-4). We quantitatively evaluate the performa
    
[^5]: BPDec: 揭示BERT预训练中掩码语言建模解码器的潜力

    BPDec: Unveiling the Potential of Masked Language Modeling Decoder in BERT pretraining

    [https://arxiv.org/abs/2401.15861](https://arxiv.org/abs/2401.15861)

    本文揭示了BPDec（BERT预训练解码器）的潜力，强调增强的掩码语言建模解码器设计及研究在BERT预训练中的重要性。

    

    BERT（来自Transformer的双向编码表示）通过其在许多任务上出色的性能彻底改变了自然语言处理领域。然而，大多数研究人员主要集中在与模型结构相关的增强，例如相对位置嵌入和更有效的注意机制。还有一些人深入研究了与掩码语言建模相关的预训练技巧，包括整词掩码。DeBERTa引入了一种针对BERT编码器模型进行预训练的增强解码器，证明效果非常显著。我们认为围绕增强掩码语言建模解码器的设计和研究并未得到应有的重视。在本文中，我们提出了几种增强解码器的设计，并介绍了BPDec（BERT预训练解码器），这是一种用于建模训练的新方法。通常，预训练的BERT模型会针对特定的自然语

    arXiv:2401.15861v2 Announce Type: replace-cross  Abstract: BERT (Bidirectional Encoder Representations from Transformers) has revolutionized the field of natural language processing through its exceptional performance on numerous tasks. Yet, the majority of researchers have mainly concentrated on enhancements related to the model structure, such as relative position embedding and more efficient attention mechanisms. Others have delved into pretraining tricks associated with Masked Language Modeling, including whole word masking. DeBERTa introduced an enhanced decoder adapted for BERT's encoder model for pretraining, proving to be highly effective. We argue that the design and research around enhanced masked language modeling decoders have been underappreciated. In this paper, we propose several designs of enhanced decoders and introduce BPDec (BERT Pretraining Decoder), a novel method for modeling training. Typically, a pretrained BERT model is fine-tuned for specific Natural Language 
    
[^6]: 使用基于像素的图像转换破坏基于卷积的不可学习数据集

    Corrupting Convolution-based Unlearnable Datasets with Pixel-based Image Transformations

    [https://arxiv.org/abs/2311.18403](https://arxiv.org/abs/2311.18403)

    研究者提出了一种基于卷积的不可学习数据集，该数据集使得现有的防御方法都失效，提出通过增加特定度量来减轻不可学习效果。

    

    不可学习的数据集会通过向干净训练集引入精心设计且难以察觉的扰动，导致模型的泛化性能急剧下降。许多现有防御方法，如JPEG压缩和对抗训练，能够有效对抗基于范数约束的附加噪声的不可学习数据集。然而，最新提出的一种基于卷积的不可学习数据集让现有的防御方法无效，给防御者带来更大挑战。为了解决这个问题，我们在简化的情景中将基于卷积的不可学习样本表达为将矩阵乘以干净样本的结果，并将类内矩阵不一致性形式化为$\Theta_{imi}$，将类间矩阵一致性形式化为$\Theta_{imc}$以研究基于卷积的不可学习数据集的工作机制。我们推测增加这两个度量将有助于减轻不可学习效果。

    arXiv:2311.18403v2 Announce Type: replace-cross  Abstract: Unlearnable datasets lead to a drastic drop in the generalization performance of models trained on them by introducing elaborate and imperceptible perturbations into clean training sets. Many existing defenses, e.g., JPEG compression and adversarial training, effectively counter UDs based on norm-constrained additive noise. However, a fire-new type of convolution-based UDs have been proposed and render existing defenses all ineffective, presenting a greater challenge to defenders. To address this, we express the convolution-based unlearnable sample as the result of multiplying a matrix by a clean sample in a simplified scenario, and formalize the intra-class matrix inconsistency as $\Theta_{imi}$, inter-class matrix consistency as $\Theta_{imc}$ to investigate the working mechanism of the convolution-based UDs. We conjecture that increasing both of these metrics will mitigate the unlearnability effect. Through validation experi
    
[^7]: 通过特征解缠来缓解对抗鲁棒性中的特征差距

    Mitigating Feature Gap for Adversarial Robustness by Feature Disentanglement. (arXiv:2401.14707v1 [cs.CV])

    [http://arxiv.org/abs/2401.14707](http://arxiv.org/abs/2401.14707)

    这项研究提出了一种通过特征解缠来缓解对抗鲁棒性中特征差距的方法，该方法明确建模和消除导致特征差距的潜在特征，有效提升了鲁棒性。

    

    深度神经网络对对抗样本很容易受到攻击。对抗微调方法旨在通过对已经在自然情况下进行预训练的模型进行对抗式微调来提升对抗鲁棒性。然而，我们发现对抗样本中的一些潜在特征被对抗扰动所混淆，并导致自然样本和对抗样本在最后一层隐藏层的特征之间出现意外增加的差距。为了解决这个问题，我们提出了一种基于解缠的方法来明确建模和进一步消除导致特征差距的潜在特征。具体而言，我们引入了特征解缠器，将对抗样本的潜在特征与对抗样本的特征分离开来，从而通过消除潜在特征来提升鲁棒性。此外，我们通过将预训练模型中的特征与对抗样本在微调模型中的特征对齐，进一步从自然样本的特征中获益，避免混淆。

    Deep neural networks are vulnerable to adversarial samples. Adversarial fine-tuning methods aim to enhance adversarial robustness through fine-tuning the naturally pre-trained model in an adversarial training manner. However, we identify that some latent features of adversarial samples are confused by adversarial perturbation and lead to an unexpectedly increasing gap between features in the last hidden layer of natural and adversarial samples. To address this issue, we propose a disentanglement-based approach to explicitly model and further remove the latent features that cause the feature gap. Specifically, we introduce a feature disentangler to separate out the latent features from the features of the adversarial samples, thereby boosting robustness by eliminating the latent features. Besides, we align features in the pre-trained model with features of adversarial samples in the fine-tuned model, to further benefit from the features from natural samples without confusion. Empirical 
    
[^8]: 对广泛的分布外检测的期望：期望之外的未知数据

    Expecting The Unexpected: Towards Broad Out-Of-Distribution Detection. (arXiv:2308.11480v1 [cs.LG])

    [http://arxiv.org/abs/2308.11480](http://arxiv.org/abs/2308.11480)

    这项研究对机器学习中分布外检测方法进行了评估，发现现有方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不稳定。

    

    提高部署的机器学习系统的可靠性通常涉及开发方法来检测分布外（OOD）的输入。然而，现有研究常常狭窄地关注训练集中缺失的类别样本，忽略了其他类型的可能分布变化。这种限制降低了这些方法在现实场景中的适用性，因为系统会遇到各种各样的异常输入。在本研究中，我们将五种不同类型的分布变化进行分类，并对最近的OOD检测方法在每一种分布变化上进行了关键评估。我们以BROAD（Benchmarking Resilience Over Anomaly Diversity）的名义公开发布我们的基准。我们的研究发现这些方法在检测未知类别方面表现出色，但在遇到其他类型的分布变化时性能不一致。换句话说，它们只能可靠地检测到它们特别设计来预期的意外输入。

    Improving the reliability of deployed machine learning systems often involves developing methods to detect out-of-distribution (OOD) inputs. However, existing research often narrowly focuses on samples from classes that are absent from the training set, neglecting other types of plausible distribution shifts. This limitation reduces the applicability of these methods in real-world scenarios, where systems encounter a wide variety of anomalous inputs. In this study, we categorize five distinct types of distribution shifts and critically evaluate the performance of recent OOD detection methods on each of them. We publicly release our benchmark under the name BROAD (Benchmarking Resilience Over Anomaly Diversity). Our findings reveal that while these methods excel in detecting unknown classes, their performance is inconsistent when encountering other types of distribution shifts. In other words, they only reliably detect unexpected inputs that they have been specifically designed to expec
    
[^9]: ChatGPT与人工撰写文本：可控文本摘要和句子风格转移的洞察

    ChatGPT vs Human-authored Text: Insights into Controllable Text Summarization and Sentence Style Transfer. (arXiv:2306.07799v1 [cs.CL])

    [http://arxiv.org/abs/2306.07799](http://arxiv.org/abs/2306.07799)

    本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众和写作风格。研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在一些特征上与人类样本有所不同，有时会包含事实错误或幻觉。

    

    大规模语言模型（如ChatGPT）以其出色的能力从简短的自然语言提示生成连贯的文本引起了媒体的重视。本文旨在系统地检查ChatGPT在两个可控生成任务中的表现，即ChatGPT能否适应不同的目标受众（专家与一般人）和写作风格（正式与非正式）。此外，我们评估了生成文本的忠实度，并将模型的表现与人工撰写的文本进行了比较。我们的研究发现，人类产生的文体变化比ChatGPT表现出的更大，而生成的文本在诸如单词类型分布等几个特征上与人类样本有所不同。此外，我们发现当 ChatGPT 将文本适应特定风格时，有时会包含事实错误或幻觉。

    Large-scale language models, like ChatGPT, have garnered significant media attention and stunned the public with their remarkable capacity for generating coherent text from short natural language prompts. In this paper, we aim to conduct a systematic inspection of ChatGPT's performance in two controllable generation tasks, with respect to ChatGPT's ability to adapt its output to different target audiences (expert vs. layman) and writing styles (formal vs. informal). Additionally, we evaluate the faithfulness of the generated text, and compare the model's performance with human-authored texts. Our findings indicate that the stylistic variations produced by humans are considerably larger than those demonstrated by ChatGPT, and the generated texts diverge from human samples in several characteristics, such as the distribution of word types. Moreover, we observe that ChatGPT sometimes incorporates factual errors or hallucinations when adapting the text to suit a specific style.
    
[^10]: 用幅度受限制优化解锁更深层网络的特征可视化

    Unlocking Feature Visualization for Deeper Networks with MAgnitude Constrained Optimization. (arXiv:2306.06805v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.06805](http://arxiv.org/abs/2306.06805)

    提出了一种名为MACO的简单方法，通过优化相位谱生成图像，同时保持幅度恒定，解决了特征可视化在深度神经网络上的挑战，并实现了高质量和高效的可解释特征可视化。

    

    特征可视化在Olah等人2017年的有影响力的工作之后获得了很大的 popularity，将其确立为可解释性的重要工具。然而，由于依赖于生成可解释图像的技巧以及在将其扩展到更深的神经网络时面临的挑战，其广泛应用受到了限制。在这里，我们描述了一种简单的方法MACO来解决这些问题。主要思想是通过优化相位谱生成图像，同时保持幅度恒定，以确保生成的解释位于自然图像空间中。我们的方法在定性和定量上都取得了显着的改进，并为大型最先进神经网络提供了高效且可解释的特征可视化。我们还展示了我们的方法具有一个属性机制，可以增强特征可视化的空间重要性。我们在一个新的基准测试中验证了我们的方法。

    Feature visualization has gained substantial popularity, particularly after the influential work by Olah et al. in 2017, which established it as a crucial tool for explainability. However, its widespread adoption has been limited due to a reliance on tricks to generate interpretable images, and corresponding challenges in scaling it to deeper neural networks. Here, we describe MACO, a simple approach to address these shortcomings. The main idea is to generate images by optimizing the phase spectrum while keeping the magnitude constant to ensure that generated explanations lie in the space of natural images. Our approach yields significantly better results (both qualitatively and quantitatively) and unlocks efficient and interpretable feature visualizations for large state-of-the-art neural networks. We also show that our approach exhibits an attribution mechanism allowing us to augment feature visualizations with spatial importance. We validate our method on a novel benchmark for compa
    
[^11]: Gode -- 将生物化学知识图谱集成到分子图神经网络的预训练中

    Gode -- Integrating Biochemical Knowledge Graph into Pre-training Molecule Graph Neural Network. (arXiv:2306.01631v1 [cs.LG])

    [http://arxiv.org/abs/2306.01631](http://arxiv.org/abs/2306.01631)

    本研究提出了一种新的方法，在分子结构和生物医学知识图谱中集成多个领域信息，通过自我监督策略预先训练更广泛和更强大的表示，并在化学属性预测任务上展示出出色的性能。

    

    分子属性的准确预测对于促进创新治疗方法的发展和理解化学物质和生物系统之间复杂的相互作用至关重要。本研究提出了一种新的方法，将单个分子结构的图表示与生物医学知识图谱 (KG) 的多个领域信息进行集成。通过集成两个级别的信息，我们可以使用自我监督策略预先训练更广泛和更强大的表示，用于分子级和 KG 级预测任务。在性能评估方面，我们在 11 个具有挑战性的化学属性预测任务上微调我们预先训练的模型。我们的框架的结果表明，我们微调的模型优于现有的最先进的模型。

    The precise prediction of molecular properties holds paramount importance in facilitating the development of innovative treatments and comprehending the intricate interplay between chemicals and biological systems. In this study, we propose a novel approach that integrates graph representations of individual molecular structures with multi-domain information from biomedical knowledge graphs (KGs). Integrating information from both levels, we can pre-train a more extensive and robust representation for both molecule-level and KG-level prediction tasks with our novel self-supervision strategy. For performance evaluation, we fine-tune our pre-trained model on 11 challenging chemical property prediction tasks. Results from our framework demonstrate our fine-tuned models outperform existing state-of-the-art models.
    
[^12]: 结合话语结构分布的长文本自动摘要方法

    Incorporating Distributions of Discourse Structure for Long Document Abstractive Summarization. (arXiv:2305.16784v1 [cs.CL])

    [http://arxiv.org/abs/2305.16784](http://arxiv.org/abs/2305.16784)

    本文提出了一种名为'RSTformer'的摘要模型，该模型全面融合了话语关系类型和不确定性，并以修辞结构理论为基础，经过严格评估，表现明显优于现有的模型。

    

    对于文本摘要，话语结构在辨识文本核心内容方面起着关键作用。可惜的是，之前将修辞结构理论（RST）引入基于transformer的自动摘要模型的研究仅考虑了核心部分的注释，从而忽略了各种不同类型的话语关系。本文提出了一种名为'RSTformer'的新型摘要模型，该模型全面融合了话语关系类型和不确定性。我们的RST-attention机制是基于文档级修辞结构的Longformer框架的扩展。经过严格评估，本文提出的模型表现明显优于现有的模型，凸显出其在多个自动评估指标和人工评估上的卓越表现。

    For text summarization, the role of discourse structure is pivotal in discerning the core content of a text. Regrettably, prior studies on incorporating Rhetorical Structure Theory (RST) into transformer-based summarization models only consider the nuclearity annotation, thereby overlooking the variety of discourse relation types. This paper introduces the 'RSTformer', a novel summarization model that comprehensively incorporates both the types and uncertainty of rhetorical relations. Our RST-attention mechanism, rooted in document-level rhetorical structure, is an extension of the recently devised Longformer framework. Through rigorous evaluation, the model proposed herein exhibits significant superiority over state-of-the-art models, as evidenced by its notable performance on several automatic metrics and human evaluation.
    
[^13]: 重新思考非监督图像语义分割中的对齐和均匀性问题

    Rethinking Alignment and Uniformity in Unsupervised Image Semantic Segmentation. (arXiv:2211.14513v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.14513](http://arxiv.org/abs/2211.14513)

    本文分析了非监督图像语义分割中困扰UISS模型的特征对齐和特征均匀性问题，提出了Semantic Attention Network(SAN) 模型，包含一个新模块 semantic attention（SEAT），以动态生成逐像素和语义特征。实验结果表明，这一非监督分割框架专注于捕捉语义表示，在多个语义分割基准测试中表现优异。

    

    非监督图像语义分割(UISS)旨在将低层视觉特征与语义级别的表示匹配，而无需外部监管。本文从特征对齐和特征均匀性的角度探究了UISS模型的关键性质，并将UISS与整幅图像的表示学习进行了比较。基于分析，我们认为UISS中现有的基于互信息的方法存在表示崩溃的问题。因此，我们提出了一种稳健的网络模型——Semantic Attention Network(SAN)，其中提出了一种新模块Semantic Attention(SEAT)，以动态生成逐像素和语义特征。在多个语义分割基准测试中的实验结果表明，我们的非监督分割框架专注于捕捉语义表示，表现优异，超过了所有未预训练的模型，甚至超过了一些预训练模型。

    Unsupervised image semantic segmentation(UISS) aims to match low-level visual features with semantic-level representations without outer supervision. In this paper, we address the critical properties from the view of feature alignments and feature uniformity for UISS models. We also make a comparison between UISS and image-wise representation learning. Based on the analysis, we argue that the existing MI-based methods in UISS suffer from representation collapse. By this, we proposed a robust network called Semantic Attention Network(SAN), in which a new module Semantic Attention(SEAT) is proposed to generate pixel-wise and semantic features dynamically. Experimental results on multiple semantic segmentation benchmarks show that our unsupervised segmentation framework specializes in catching semantic representations, which outperforms all the unpretrained and even several pretrained methods.
    

