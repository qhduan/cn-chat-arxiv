# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment](https://arxiv.org/abs/2404.01054) | 提出了Regularized Best-of-N (RBoN)，通过引入接近性项来减轻奖励欺骗，提高了算法在解码时与人类偏好对齐的效果。 |
| [^2] | [FlexCap: Generating Rich, Localized, and Flexible Captions in Images](https://arxiv.org/abs/2403.12026) | FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。 |
| [^3] | [Diversity-Aware Ensembling of Language Models Based on Topological Data Analysis](https://arxiv.org/abs/2402.14184) | 基于拓扑数据分析的方法，通过估计NLP模型集成的权重，提高了集成模型的质量，提高了文本分类准确性和相关不确定性估计。 |
| [^4] | [API Pack: A Massive Multilingual Dataset for API Call Generation](https://arxiv.org/abs/2402.09615) | 这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成 |
| [^5] | [Continuously Learning New Words in Automatic Speech Recognition.](http://arxiv.org/abs/2401.04482) | 该论文提出了一种自我监督的持续学习方法，用于解决自动语音识别中识别新词的问题。通过对讲座录音进行推理和收集包含新词的话语，然后在自适应数据集上进行持续学习，可以在新词出现频率较高时提高性能，同时保持整体性能。 |
| [^6] | [Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text.](http://arxiv.org/abs/2310.08225) | 本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER），在大数据场景下具有较高的计算效率和性能提升。 |

# 详细

[^1]: 正则化的最佳-N采样以减轻语言模型对齐中的奖励欺骗问题

    Regularized Best-of-N Sampling to Mitigate Reward Hacking for Language Model Alignment

    [https://arxiv.org/abs/2404.01054](https://arxiv.org/abs/2404.01054)

    提出了Regularized Best-of-N (RBoN)，通过引入接近性项来减轻奖励欺骗，提高了算法在解码时与人类偏好对齐的效果。

    

    Best-of-N (BoN)采样与奖励模型已被证明是一种有效的策略，用于在解码时将大型语言模型(LLMs)与人类偏好对齐。然而，BoN采样容易受到奖励欺骗问题的影响。为了防止奖励欺骗，我们提出了一种名为Regularized Best-of-N (RBoN)的变体，通过在响应选择中结合接近性项来减轻奖励欺骗，类似于偏好学习技术。

    arXiv:2404.01054v1 Announce Type: cross  Abstract: Best-of-N (BoN) sampling with a reward model has been shown to be an effective strategy for aligning Large Language Models (LLMs) to human preferences at the time of decoding. BoN sampling is susceptible to a problem known as reward hacking. Because the reward model is an imperfect proxy for the true objective, over-optimizing its value can compromise its performance on the true objective. A common solution to prevent reward hacking in preference learning techniques is to optimize a reward using proximity regularization (e.g., KL regularization), which ensures that the language model remains close to the reference model. In this research, we propose Regularized Best-of-N (RBoN), a variant of BoN that aims to mitigate reward hacking by incorporating a proximity term in response selection, similar to preference learning techniques. We evaluate two variants of RBoN on the AlpacaFarm dataset and find that they outperform BoN, especially wh
    
[^2]: FlexCap：在图像中生成丰富、本地化和灵活的标题

    FlexCap: Generating Rich, Localized, and Flexible Captions in Images

    [https://arxiv.org/abs/2403.12026](https://arxiv.org/abs/2403.12026)

    FlexCap模型能够生成图像中具有不同长度的区域描述，在密集字幕任务和视觉问答系统中表现出优越性能。

    

    我们介绍了一种多功能的$\textit{灵活字幕}$视觉-语言模型（VLM），能够生成长度不同的特定区域描述。该模型FlexCap经过训练，可为输入的边界框生成长度条件的字幕，从而可以控制其输出的信息密度，描述范围从简洁的对象标签到详细的字幕。为了实现这一点，我们从带字幕的图像开始创建了大规模的图像区域描述训练数据集。这种灵活的字幕功能有几个宝贵的应用。首先，FlexCap在Visual Genome数据集上的密集字幕任务中表现出优越性能。其次，可以通过采用FlexCap生成本地化描述作为大型语言模型的输入来构建视觉问答（VQA）系统。由此产生的系统在许多VQ上实现了最新技术的零样本性能。

    arXiv:2403.12026v1 Announce Type: cross  Abstract: We introduce a versatile $\textit{flexible-captioning}$ vision-language model (VLM) capable of generating region-specific descriptions of varying lengths. The model, FlexCap, is trained to produce length-conditioned captions for input bounding boxes, and this allows control over the information density of its output, with descriptions ranging from concise object labels to detailed captions. To achieve this we create large-scale training datasets of image region descriptions of varying length, starting from captioned images. This flexible-captioning capability has several valuable applications.   First, FlexCap demonstrates superior performance in dense captioning tasks on the Visual Genome dataset. Second, a visual question answering (VQA) system can be built by employing FlexCap to generate localized descriptions as inputs to a large language model. The resulting system achieves state-of-the-art zero-shot performance on a number of VQ
    
[^3]: 基于拓扑数据分析的语言模型多样性集成

    Diversity-Aware Ensembling of Language Models Based on Topological Data Analysis

    [https://arxiv.org/abs/2402.14184](https://arxiv.org/abs/2402.14184)

    基于拓扑数据分析的方法，通过估计NLP模型集成的权重，提高了集成模型的质量，提高了文本分类准确性和相关不确定性估计。

    

    集成是提高机器学习模型性能的重要工具。在与自然语言处理相关的情况下，由于开源中存在多个大型模型，集成有助于提升方法的性能。然而，现有方法主要依赖于对集成中每个模型的预测进行简单平均，对每个模型赋予相同权重，忽略了模型质量和一致性的差异。我们提出利用不仅单个模型表现知识，还使用它们之间的相似性来估计NLP模型集成的权重。通过采用基于拓扑数据分析（TDA）的距离度量，我们改进了我们的集成。文本分类准确性和相关不确定性估计的质量得到提高。

    arXiv:2402.14184v1 Announce Type: cross  Abstract: Ensembles are important tools for improving the performance of machine learning models. In cases related to natural language processing, ensembles boost the performance of a method due to multiple large models available in open source. However, existing approaches mostly rely on simple averaging of predictions by ensembles with equal weights for each model, ignoring differences in the quality and conformity of models. We propose to estimate weights for ensembles of NLP models using not only knowledge of their individual performance but also their similarity to each other. By adopting distance measures based on Topological Data Analysis (TDA), we improve our ensemble. The quality improves for both text classification accuracy and relevant uncertainty estimation.
    
[^4]: API Pack：一个用于API调用生成的大规模多语言数据集

    API Pack: A Massive Multilingual Dataset for API Call Generation

    [https://arxiv.org/abs/2402.09615](https://arxiv.org/abs/2402.09615)

    这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成

    

    我们介绍了API Pack，一个包含超过一百万个指令-API调用对的多语言数据集，旨在提高大型语言模型的API调用生成能力。通过实验，我们证明了API Pack在提升模型在这一特定任务上的效果的同时，保持其在一般编码方面的整体熟练程度。仅在20,000个Python实例上对CodeLlama-13B进行微调，其生成未见过的API调用的准确率比GPT-3.5和GPT-4分别高出10%和5%。扩展到100k个例子可以提高对训练期间未见过的新API的泛化能力。此外，实现了跨语言的API调用生成，而无需大量语言特定的数据。数据集、经过微调的模型和整体代码库可在https://github.com/anonymous_url上公开获取。

    arXiv:2402.09615v1 Announce Type: cross  Abstract: We introduce API Pack, a multilingual dataset featuring over one million instruction-API call pairs aimed at advancing large language models' API call generation capabilities. Through experiments, we demonstrate API Pack's efficacy in enhancing models for this specialized task while maintaining their overall proficiency at general coding. Fine-tuning CodeLlama-13B on just 20,000 Python instances yields over 10% and 5% higher accuracy than GPT-3.5 and GPT-4 respectively in generating unseen API calls. Scaling to 100k examples improves generalization to new APIs not seen during training. In addition, cross-lingual API call generation is achieved without needing extensive data per language. The dataset, fine-tuned models, and overall code base are publicly available at https://github.com/anonymous_url.
    
[^5]: 在自动语音识别中持续学习新词

    Continuously Learning New Words in Automatic Speech Recognition. (arXiv:2401.04482v1 [cs.CL])

    [http://arxiv.org/abs/2401.04482](http://arxiv.org/abs/2401.04482)

    该论文提出了一种自我监督的持续学习方法，用于解决自动语音识别中识别新词的问题。通过对讲座录音进行推理和收集包含新词的话语，然后在自适应数据集上进行持续学习，可以在新词出现频率较高时提高性能，同时保持整体性能。

    

    尽管最近取得了进展，但自动语音识别（ASR）系统仍然远未完美。典型的错误包括缩写词、命名实体和领域特定的专用词，这些词几乎没有或没有数据可用来训练。为了解决识别这些词的问题，我们提出了一种自我监督的持续学习方法。给定带有对应幻灯片的讲座录音，我们通过使用先前工作中的记忆增强型ASR模型来将模型偏向于从幻灯片中解码新词。然后，我们对讲座进行推理，将包含检测到的新词的话语收集到自适应数据集中。接着，对这个集合进行持续学习，通过调整添加到模型的每个权重矩阵的低秩矩阵权重。整个过程对多个讲座进行迭代。我们展示了通过这种方法，我们在新词出现频率较高时获得了性能的提升（超过80%的召回率），同时保持了模型的整体性能。

    Despite recent advances, Automatic Speech Recognition (ASR) systems are still far from perfect. Typical errors include acronyms, named entities and domain-specific special words for which little or no data is available. To address the problem of recognizing these words, we propose an self-supervised continual learning approach. Given the audio of a lecture talk with corresponding slides, we bias the model towards decoding new words from the slides by using a memory-enhanced ASR model from previous work. Then, we perform inference on the talk, collecting utterances that contain detected new words into an adaptation dataset. Continual learning is then performed on this set by adapting low-rank matrix weights added to each weight matrix of the model. The whole procedure is iterated for many talks. We show that with this approach, we obtain increasing performance on the new words when they occur more frequently (more than 80% recall) while preserving the general performance of the model.
    
[^6]: 使用自监督表示法对语音和文本进行快速字错率估计

    Fast Word Error Rate Estimation Using Self-Supervised Representations For Speech And Text. (arXiv:2310.08225v1 [eess.AS])

    [http://arxiv.org/abs/2310.08225](http://arxiv.org/abs/2310.08225)

    本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER），在大数据场景下具有较高的计算效率和性能提升。

    

    自动语音识别（ASR）的质量通常通过字错率（WER）来衡量。WER估计是一项任务，旨在预测ASR系统的WER，给定一个语音说话和一个转录。在大量数据上训练先进的ASR系统的同时，这个任务越来越受到关注。在这种情况下，WER估计在许多场景中变得必要，例如选择具有未知转录质量的训练数据，或在没有地面真实转录的情况下估计ASR系统的测试性能。面对大量数据，WER估计仪的运算效率在实际应用中变得至关重要。然而，以前的研究通常未将其视为优先考虑的问题。本文介绍了一种使用自监督学习表示法（SSLR）的快速WER估计器（Fe-WER）。该估计器基于通过平均池聚合的SSLR构建。结果表明，相对于e-WER3基线，Fe-WER的性能提高了19.69％。

    The quality of automatic speech recognition (ASR) is typically measured by word error rate (WER). WER estimation is a task aiming to predict the WER of an ASR system, given a speech utterance and a transcription. This task has gained increasing attention while advanced ASR systems are trained on large amounts of data. In this case, WER estimation becomes necessary in many scenarios, for example, selecting training data with unknown transcription quality or estimating the testing performance of an ASR system without ground truth transcriptions. Facing large amounts of data, the computation efficiency of a WER estimator becomes essential in practical applications. However, previous works usually did not consider it as a priority. In this paper, a Fast WER estimator (Fe-WER) using self-supervised learning representation (SSLR) is introduced. The estimator is built upon SSLR aggregated by average pooling. The results show that Fe-WER outperformed the e-WER3 baseline relatively by 19.69% an
    

