# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Clinical BERTScore: An Improved Measure of Automatic Speech Recognition Performance in Clinical Settings.](http://arxiv.org/abs/2303.05737) | 本文提出了一种临床BERTScore（CBERTScore）度量，它比其他度量更严厉地惩罚临床相关的错误，更接近于临床医生对医学句子的偏好。作者还收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。 |
| [^2] | [Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study.](http://arxiv.org/abs/2303.03857) | 本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。 |
| [^3] | [Heterogeneous Graph Learning for Acoustic Event Classification.](http://arxiv.org/abs/2303.02665) | 本文提出了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘，可以适应各种空间和时间尺度，有效地连接了跨模态的相关节点，在声音事件分类中表现出最先进的性能。 |
| [^4] | [Fine-grained Emotional Control of Text-To-Speech: Learning To Rank Inter- And Intra-Class Emotion Intensities.](http://arxiv.org/abs/2303.01508) | 本文提出了一种细粒度可控情感TTS，考虑了内部和外部类距离，并能够合成具有可识别强度差异的语音。 |
| [^5] | [Perceptual-Neural-Physical Sound Matching.](http://arxiv.org/abs/2301.02886) | 本文提出了一种新的声音匹配算法，称为感知-神经-物理损失（PNP），它是频谱损失的最优二次近似，能够更好地适应不同参数的感知重要性，同时具有快速收敛的特点。 |
| [^6] | [A large-scale and PCR-referenced vocal audio dataset for COVID-19.](http://arxiv.org/abs/2212.07738) | 英国COVID-19 Vocal Audio Dataset是迄今为止最大的SARS-CoV-2 PCR参考音频记录集合，旨在为训练和评估使用声音数据分类SARS-CoV-2感染状态或相关呼吸症状的机器学习模型而设计。 |
| [^7] | [Neural Transducer Training: Reduced Memory Consumption with Sample-wise Computation.](http://arxiv.org/abs/2211.16270) | 本文提出了一种内存高效的神经转录器训练方法，采用逐个样本计算转录器损失和梯度，显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。 |
| [^8] | [Deep Neural Mel-Subband Beamformer for In-car Speech Separation.](http://arxiv.org/abs/2211.12590) | 本文提出了一种基于DL的Mel-Subband时空波束成形器，用于在车载环境中进行语音分离，通过基于Mel尺度的子带选择策略，实现对低频的细粒度处理和对高频的粗粒度处理，降低了计算成本和推理时间。 |
| [^9] | [LA-VocE: Low-SNR Audio-visual Speech Enhancement using Neural Vocoders.](http://arxiv.org/abs/2211.10999) | LA-VocE是一种新的音频视觉语音增强方法，使用神经声码器将从嘈杂的音频视觉语音预测的mel频谱图转换为波形音频，适用于多种语言和不同水平的背景噪声和语音干扰。 |
| [^10] | [Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models.](http://arxiv.org/abs/2211.05103) | 本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。预训练的语音模型在较低层中最优地编码了语言区分信息，从这些层获得的嵌入能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果，且使用的参数量仅为其它模型的五分之一。 |
| [^11] | [Cutting Through the Noise: An Empirical Comparison of Psychoacoustic and Envelope-based Features for Machinery Fault Detection.](http://arxiv.org/abs/2211.01704) | 本文提出了一个自动化和噪声鲁棒的听觉检查系统，用于检测机械部件的健康状况。我们提供了一个基准来比较不同类型的包络特征与心理声学特征。我们是第一个应用时变心理声学特征进行故障检测的人。 |
| [^12] | [Learning Audio Features with Metadata and Contrastive Learning.](http://arxiv.org/abs/2210.16192) | 本研究使用监督对比学习结合可用元数据解决多个前置任务，学习数据的良好表示。在呼吸音分类数据集上，仅使用元数据学习表示可以获得与仅使用类标签的交叉熵相似的性能。在使用多个监督对比学习将类标签与元数据相结合时，获得了最先进的得分。 |
| [^13] | [Articulation GAN: Unsupervised modeling of articulatory learning.](http://arxiv.org/abs/2210.15173) | 本文提出了一种新的无监督生成模型，通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式，从而更好地模拟人类语音产生的过程。 |
| [^14] | [Play It Back: Iterative Attention for Audio Recognition.](http://arxiv.org/abs/2210.11328) | 该论文提出了一种基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注，最终实现了在三个音频分类基准测试中始终实现最先进的性能。 |
| [^15] | [PSVRF: Learning to restore Pitch-Shifted Voice without reference.](http://arxiv.org/abs/2210.02731) | 本文提出了一种无参考方法PSVRF，用于高质量还原变调语音，可以增强ASV系统对音高缩放攻击的鲁棒性，性能甚至超过了最先进的基于参考的方法。 |
| [^16] | [Uconv-Conformer: High Reduction of Input Sequence Length for End-to-End Speech Recognition.](http://arxiv.org/abs/2208.07657) | 本文提出了一种新型Uconv-Conformer架构，可以将输入序列长度缩短16倍，加速中间层的工作，同时通过使用上采样块解决了收敛问题，表现出更好的WER和更快的训练和推理速度。 |
| [^17] | [Alternate Intermediate Conditioning with Syllable-level and Character-level Targets for Japanese ASR.](http://arxiv.org/abs/2204.00175) | 该论文提出了一种基于音节和字符目标的交替中间条件方法，利用字符级和音节级中间预测作为条件特征来处理日语ASR中的多对一和一对多的映射问题，并在实验中取得了优异的表现。 |

# 详细

[^1]: 临床BERTScore：临床环境下自动语音识别性能的改进度量

    Clinical BERTScore: An Improved Measure of Automatic Speech Recognition Performance in Clinical Settings. (arXiv:2303.05737v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2303.05737](http://arxiv.org/abs/2303.05737)

    本文提出了一种临床BERTScore（CBERTScore）度量，它比其他度量更严厉地惩罚临床相关的错误，更接近于临床医生对医学句子的偏好。作者还收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。

    The paper proposes a Clinical BERTScore (CBERTScore) metric for ASR in medical contexts, which penalizes clinically-relevant mistakes more than other metrics and aligns more closely with clinician preferences. The authors also collect a benchmark of clinician preferences on medical sentences and release it for the community to further develop clinically-aware ASR metrics.

    医学环境中的自动语音识别（ASR）有潜力节省时间，降低成本，提高报告准确性并减少医生的疲劳。然而，由于避免医学相关的转录错误的重要性，医疗行业采用这种技术的速度较慢。在这项工作中，我们提出了临床BERTScore（CBERTScore），这是一种ASR度量，它比其他度量（WER、BLUE、METEOR等）更严厉地惩罚临床相关的错误。我们证明了这个度量更接近于临床医生对医学句子的偏好，有时差距很大。我们收集了13个临床医生对149个现实医学句子的偏好基准，称为临床转录偏好基准（CTP），证明CBERTScore更接近于临床医生的偏好，并将基准发布给社区以进一步开发具有临床意识的ASR度量。

    Automatic Speech Recognition (ASR) in medical contexts has the potential to save time, cut costs, increase report accuracy, and reduce physician burnout. However, the healthcare industry has been slower to adopt this technology, in part due to the importance of avoiding medically-relevant transcription mistakes. In this work, we present the Clinical BERTScore (CBERTScore), an ASR metric that penalizes clinically-relevant mistakes more than others. We demonstrate that this metric more closely aligns with clinician preferences on medical sentences as compared to other metrics (WER, BLUE, METEOR, etc), sometimes by wide margins. We collect a benchmark of 13 clinician preferences on 149 realistic medical sentences called the Clinician Transcript Preference benchmark (CTP), demonstrate that CBERTScore more closely matches what clinicians prefer, and release the benchmark for the community to further develop clinically-aware ASR metrics.
    
[^2]: 利用预训练的AudioLDM进行文本到声音生成：基准研究

    Leveraging Pre-trained AudioLDM for Text to Sound Generation: A Benchmark Study. (arXiv:2303.03857v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.03857](http://arxiv.org/abs/2303.03857)

    本文研究了使用预训练的AudioLDM作为声音生成的骨干的优势，证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势，并在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，为未来的研究提供了基础。

    This paper investigates the advantages of using pre-trained AudioLDM as the backbone for sound generation, demonstrates the benefits of using pre-trained models for text-to-sound generation in data-scarcity scenarios, and evaluates various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols to provide a basis for future research.

    深度神经网络最近在文本提示下实现了声音生成的突破。尽管它们的表现很有前途，但当前的文本到声音生成模型在小规模数据集（例如过度拟合）上面临问题，从而显著限制了它们的性能。在本文中，我们研究了使用预训练的AudioLDM作为声音生成的骨干的优势。我们的研究证明了在数据稀缺情况下使用预训练模型进行文本到声音生成的优势。此外，实验表明，不同的训练策略（例如训练条件）可能会影响AudioLDM在不同规模的数据集上的性能。为了促进未来的研究，我们还在几个常用数据集上使用相同的评估协议评估了各种文本到声音生成系统，这些协议允许在共同基础上公平比较和基准测试这些方法。

    Deep neural networks have recently achieved breakthroughs in sound generation with text prompts. Despite their promising performance, current text-to-sound generation models face issues on small-scale datasets (e.g., overfitting), significantly limiting their performance. In this paper, we investigate the use of pre-trained AudioLDM, the state-of-the-art model for text-to-audio generation, as the backbone for sound generation. Our study demonstrates the advantages of using pre-trained models for text-to-sound generation, especially in data-scarcity scenarios. In addition, experiments show that different training strategies (e.g., training conditions) may affect the performance of AudioLDM on datasets of different scales. To facilitate future studies, we also evaluate various text-to-sound generation systems on several frequently used datasets under the same evaluation protocols, which allow fair comparisons and benchmarking of these methods on the common ground.
    
[^3]: 异构图学习在声音事件分类中的应用

    Heterogeneous Graph Learning for Acoustic Event Classification. (arXiv:2303.02665v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.02665](http://arxiv.org/abs/2303.02665)

    本文提出了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘，可以适应各种空间和时间尺度，有效地连接了跨模态的相关节点，在声音事件分类中表现出最先进的性能。

    This paper proposes a new model, Heterogeneous Graph Crossmodal Network (HGCN), which learns crossmodal edges and can adapt to various spatial and temporal scales, effectively connecting relevant nodes across modalities. It achieves state-of-the-art performance in acoustic event classification.

    异构图提供了一种紧凑、高效、可扩展的方式来建模涉及多个不同模态的数据。这使得使用异构图来建模音频视觉数据成为一种有吸引力的选择。然而，图结构在音频视觉数据中并不自然。音频视觉数据的图是手动构建的，这既困难又次优。在这项工作中，我们通过（i）提出一种参数化图构建策略来解决这个问题，以及（ii）学习跨模态边缘。为此，我们开发了一种新模型，异构图跨模态网络（HGCN），它学习跨模态边缘。我们提出的模型可以适应各种空间和时间尺度，因为它是参数化构建的，而可学习的跨模态边缘有效地连接了跨模态的相关节点。在一个大型基准数据集（AudioSet）上的实验表明，我们的模型是最先进的（0.53平均精度），优于transfo。

    Heterogeneous graphs provide a compact, efficient, and scalable way to model data involving multiple disparate modalities. This makes modeling audiovisual data using heterogeneous graphs an attractive option. However, graph structure does not appear naturally in audiovisual data. Graphs for audiovisual data are constructed manually which is both difficult and sub-optimal. In this work, we address this problem by (i) proposing a parametric graph construction strategy for the intra-modal edges, and (ii) learning the crossmodal edges. To this end, we develop a new model, heterogeneous graph crossmodal network (HGCN) that learns the crossmodal edges. Our proposed model can adapt to various spatial and temporal scales owing to its parametric construction, while the learnable crossmodal edges effectively connect the relevant nodes across modalities. Experiments on a large benchmark dataset (AudioSet) show that our model is state-of-the-art (0.53 mean average precision), outperforming transfo
    
[^4]: 文本转语音的细粒度情感控制：学习排名内部和外部类情感强度

    Fine-grained Emotional Control of Text-To-Speech: Learning To Rank Inter- And Intra-Class Emotion Intensities. (arXiv:2303.01508v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2303.01508](http://arxiv.org/abs/2303.01508)

    本文提出了一种细粒度可控情感TTS，考虑了内部和外部类距离，并能够合成具有可识别强度差异的语音。

    This paper proposes a fine-grained controllable emotional TTS, that considers both interand intra-class distances and be able to synthesize speech with recognizable intensity difference.

    最先进的文本转语音（TTS）模型能够产生高质量的语音。然而，生成的语音通常在情感表达上是中性的，而很多时候人们希望对单词或音素进行细粒度的情感控制。虽然仍然具有挑战性，但最近已经提出了第一批TTS模型，能够通过手动分配情感强度来控制语音。不幸的是，由于忽略了内部类距离，强度差异经常无法识别。在本文中，我们提出了一种细粒度可控情感TTS，考虑了内部和外部类距离，并能够合成具有可识别强度差异的语音。我们的主观和客观实验表明，我们的模型在可控性、情感表达和自然度方面超过了两个最先进的可控TTS模型。

    State-of-the-art Text-To-Speech (TTS) models are capable of producing high-quality speech. The generated speech, however, is usually neutral in emotional expression, whereas very often one would want fine-grained emotional control of words or phonemes. Although still challenging, the first TTS models have been recently proposed that are able to control voice by manually assigning emotion intensity. Unfortunately, due to the neglect of intra-class distance, the intensity differences are often unrecognizable. In this paper, we propose a fine-grained controllable emotional TTS, that considers both interand intra-class distances and be able to synthesize speech with recognizable intensity difference. Our subjective and objective experiments demonstrate that our model exceeds two state-of-the-art controllable TTS models for controllability, emotion expressiveness and naturalness.
    
[^5]: 感知-神经-物理声音匹配

    Perceptual-Neural-Physical Sound Matching. (arXiv:2301.02886v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2301.02886](http://arxiv.org/abs/2301.02886)

    本文提出了一种新的声音匹配算法，称为感知-神经-物理损失（PNP），它是频谱损失的最优二次近似，能够更好地适应不同参数的感知重要性，同时具有快速收敛的特点。

    This paper proposes a new sound matching algorithm called Perceptual-Neural-Physical loss (PNP), which is the optimal quadratic approximation of spectral loss and can better accommodate the differing perceptual significance of each parameter while having fast convergence.

    声音匹配算法旨在通过参数化音频合成来近似目标波形。深度神经网络在匹配持续谐波音调方面取得了有希望的结果。然而，当目标是非平稳和非谐波的时候，例如打击乐器，任务就更具挑战性。我们将这个问题归因于损失函数的不足。一方面，参数域中的均方误差，称为“P-loss”，简单快速，但未能适应每个参数的不同感知重要性。另一方面，频谱时间域中的均方误差，称为“频谱损失”，在感知上是有动机的，并在可微分数字信号处理（DDSP）中发挥作用。然而，频谱损失是音高间隔的不良预测因素，其梯度可能计算成本高，因此收敛速度较慢。在这个困境中，我们提出了感知-神经-物理损失（PNP）。PNP是频谱损失的最优二次近似，同时具有快速收敛的特点。

    Sound matching algorithms seek to approximate a target waveform by parametric audio synthesis. Deep neural networks have achieved promising results in matching sustained harmonic tones. However, the task is more challenging when targets are nonstationary and inharmonic, e.g., percussion. We attribute this problem to the inadequacy of loss function. On one hand, mean square error in the parametric domain, known as "P-loss", is simple and fast but fails to accommodate the differing perceptual significance of each parameter. On the other hand, mean square error in the spectrotemporal domain, known as "spectral loss", is perceptually motivated and serves in differentiable digital signal processing (DDSP). Yet, spectral loss is a poor predictor of pitch intervals and its gradient may be computationally expensive; hence a slow convergence. Against this conundrum, we present Perceptual-Neural-Physical loss (PNP). PNP is the optimal quadratic approximation of spectral loss while being as fast 
    
[^6]: 一份大规模的、基于PCR的COVID-19声音数据集

    A large-scale and PCR-referenced vocal audio dataset for COVID-19. (arXiv:2212.07738v3 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2212.07738](http://arxiv.org/abs/2212.07738)

    英国COVID-19 Vocal Audio Dataset是迄今为止最大的SARS-CoV-2 PCR参考音频记录集合，旨在为训练和评估使用声音数据分类SARS-CoV-2感染状态或相关呼吸症状的机器学习模型而设计。

    The UK COVID-19 Vocal Audio Dataset is the largest collection of SARS-CoV-2 PCR-referenced audio recordings to date, designed for the training and evaluation of machine learning models that classify SARS-CoV-2 infection status or associated respiratory symptoms using vocal audio.

    英国COVID-19 Vocal Audio Dataset旨在为训练和评估使用声音数据分类SARS-CoV-2感染状态或相关呼吸症状的机器学习模型而设计。英国卫生安全局通过国家测试和追踪计划和REACT-1调查在2021年3月至2022年3月期间招募了自愿参与者，收集了自愿咳嗽、呼气和语音的音频记录，并将其与SARS-CoV-2检测结果相关联。该数据集是迄今为止最大的SARS-CoV-2 PCR参考音频记录集合。

    The UK COVID-19 Vocal Audio Dataset is designed for the training and evaluation of machine learning models that classify SARS-CoV-2 infection status or associated respiratory symptoms using vocal audio. The UK Health Security Agency recruited voluntary participants through the national Test and Trace programme and the REACT-1 survey in England from March 2021 to March 2022, during dominant transmission of the Alpha and Delta SARS-CoV-2 variants and some Omicron variant sublineages. Audio recordings of volitional coughs, exhalations, and speech were collected in the 'Speak up to help beat coronavirus' digital survey alongside demographic, self-reported symptom and respiratory condition data, and linked to SARS-CoV-2 test results. The UK COVID-19 Vocal Audio Dataset represents the largest collection of SARS-CoV-2 PCR-referenced audio recordings to date. PCR results were linked to 70,794 of 72,999 participants and 24,155 of 25,776 positive cases. Respiratory symptoms were reported by 45.6
    
[^7]: 神经转录器训练：采用逐样本计算减少内存消耗

    Neural Transducer Training: Reduced Memory Consumption with Sample-wise Computation. (arXiv:2211.16270v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.16270](http://arxiv.org/abs/2211.16270)

    本文提出了一种内存高效的神经转录器训练方法，采用逐个样本计算转录器损失和梯度，显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。

    This paper proposes a memory-efficient training method for neural transducer, which computes the transducer loss and gradients sample by sample, significantly reducing memory usage and performing at competitive speed compared to the default batched computation.

    神经转录器是一种用于自动语音识别（ASR）的端到端模型。虽然该模型非常适合流式ASR，但训练过程仍然具有挑战性。在训练过程中，内存需求可能会迅速超过最先进的GPU的容量，限制批量大小和序列长度。在这项工作中，我们分析了典型转录器训练设置的时间和空间复杂度。我们提出了一种内存高效的训练方法，逐个样本计算转录器损失和梯度。我们提出了优化方法，以增加逐样本方法的效率和并行性。在一组彻底的基准测试中，我们展示了我们的逐样本方法显著减少了内存使用量，并在与默认批量计算相比时表现出竞争速度。作为亮点，我们成功地使用仅6 GB的内存计算了批量大小为1024，音频长度为40秒的转录器损失和梯度。

    The neural transducer is an end-to-end model for automatic speech recognition (ASR). While the model is well-suited for streaming ASR, the training process remains challenging. During training, the memory requirements may quickly exceed the capacity of state-of-the-art GPUs, limiting batch size and sequence lengths. In this work, we analyze the time and space complexity of a typical transducer training setup. We propose a memory-efficient training method that computes the transducer loss and gradients sample by sample. We present optimizations to increase the efficiency and parallelism of the sample-wise method. In a set of thorough benchmarks, we show that our sample-wise method significantly reduces memory usage, and performs at competitive speed when compared to the default batched computation. As a highlight, we manage to compute the transducer loss and gradients for a batch size of 1024, and audio length of 40 seconds, using only 6 GB of memory.
    
[^8]: 深度神经Mel-Subband波束成形器用于车载语音分离

    Deep Neural Mel-Subband Beamformer for In-car Speech Separation. (arXiv:2211.12590v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2211.12590](http://arxiv.org/abs/2211.12590)

    本文提出了一种基于DL的Mel-Subband时空波束成形器，用于在车载环境中进行语音分离，通过基于Mel尺度的子带选择策略，实现对低频的细粒度处理和对高频的粗粒度处理，降低了计算成本和推理时间。

    This paper proposes a DL-based Mel-Subband spatio-temporal beamformer for speech separation in a car environment, which reduces computational costs and inference time by using a Mel-scale based subband selection strategy for fine-grained processing of lower frequencies and coarse-grained processing of higher frequencies.

    当前的深度学习（DL）基于波束成形技术已被证明在语音分离中有效，但它们通常被设计为独立处理窄带（NB）频率，这导致更高的计算成本和推理时间，使它们不适合实际应用。在本文中，我们提出了基于DL的Mel-Subband时空波束成形器，以在车载环境中进行语音分离，从而降低计算成本和推理时间。与传统的子带（SB）方法相反，我们的框架使用基于Mel尺度的子带选择策略，确保对大多数语音共振结构存在的低频进行细粒度处理，对高频进行粗粒度处理。以递归方式，从估计的子带语音和噪声协方差矩阵中确定每个扬声器位置/区域的鲁棒帧级波束成形权重。此外，所提出的框架还估计并抑制任何回声。

    While current deep learning (DL)-based beamforming techniques have been proved effective in speech separation, they are often designed to process narrow-band (NB) frequencies independently which results in higher computational costs and inference times, making them unsuitable for real-world use. In this paper, we propose DL-based mel-subband spatio-temporal beamformer to perform speech separation in a car environment with reduced computation cost and inference time. As opposed to conventional subband (SB) approaches, our framework uses a mel-scale based subband selection strategy which ensures a fine-grained processing for lower frequencies where most speech formant structure is present, and coarse-grained processing for higher frequencies. In a recursive way, robust frame-level beamforming weights are determined for each speaker location/zone in a car from the estimated subband speech and noise covariance matrices. Furthermore, proposed framework also estimates and suppresses any echo
    
[^9]: LA-VocE: 使用神经声码器的低信噪比音频视觉语音增强

    LA-VocE: Low-SNR Audio-visual Speech Enhancement using Neural Vocoders. (arXiv:2211.10999v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2211.10999](http://arxiv.org/abs/2211.10999)

    LA-VocE是一种新的音频视觉语音增强方法，使用神经声码器将从嘈杂的音频视觉语音预测的mel频谱图转换为波形音频，适用于多种语言和不同水平的背景噪声和语音干扰。

    LA-VocE is a new audio-visual speech enhancement method that uses a neural vocoder to convert mel-spectrograms predicted from noisy audio-visual speech via a transformer-based architecture into waveform audio, and is applicable to multiple languages and different levels of background noise and speech interference.

    音频视觉语音增强旨在通过利用音频本身以及目标说话者的唇部运动从嘈杂的环境中提取干净的语音。这种方法已经被证明比仅使用音频的语音增强方法更有效，特别是对于消除干扰语音。尽管语音合成方面取得了最近的进展，但大多数音频视觉方法仍然使用频谱映射/掩蔽来重现干净的音频，通常会在现有的语音增强架构中添加视觉骨干。在这项工作中，我们提出了LA-VocE，一种新的两阶段方法，通过基于Transformer的架构从嘈杂的音频视觉语音预测mel频谱图，然后使用神经声码器（HiFi-GAN）将它们转换为波形音频。我们在数千个说话者和11种以上不同的语言上训练和评估我们的框架，并研究我们的模型适应不同水平的背景噪声和语音干扰的能力。我们的实验表明

    Audio-visual speech enhancement aims to extract clean speech from a noisy environment by leveraging not only the audio itself but also the target speaker's lip movements. This approach has been shown to yield improvements over audio-only speech enhancement, particularly for the removal of interfering speech. Despite recent advances in speech synthesis, most audio-visual approaches continue to use spectral mapping/masking to reproduce the clean audio, often resulting in visual backbones added to existing speech enhancement architectures. In this work, we propose LA-VocE, a new two-stage approach that predicts mel-spectrograms from noisy audio-visual speech via a transformer-based architecture, and then converts them into waveform audio using a neural vocoder (HiFi-GAN). We train and evaluate our framework on thousands of speakers and 11+ different languages, and study our model's ability to adapt to different levels of background noise and speech interference. Our experiments show that 
    
[^10]: 意外学习者：自监督多语言模型中的口语语言识别

    Accidental Learners: Spoken Language Identification in Multilingual Self-Supervised Models. (arXiv:2211.05103v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2211.05103](http://arxiv.org/abs/2211.05103)

    本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。预训练的语音模型在较低层中最优地编码了语言区分信息，从这些层获得的嵌入能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果，且使用的参数量仅为其它模型的五分之一。

    This paper extends previous self-supervised approaches for language identification by experimenting with Conformer based architecture in a multilingual pre-training paradigm. The pre-trained speech models optimally encode language discriminatory information in lower layers, and the embeddings obtained from these layers are significantly robust to classify unseen languages and different acoustic environments without additional training. After fine-tuning a pre-trained Conformer model on the VoxLingua107 dataset, the authors achieve results similar to current state-of-the-art systems for language identification, with 5x less parameters. The model is open-sourced through the NVIDIA NeMo toolkit.

    本文通过在多语言预训练范式中尝试Conformer架构，扩展了先前的自监督语言识别方法。我们发现，预训练的语音模型在较低层中最优地编码了语言区分信息。此外，我们证明了从这些层获得的嵌入在没有额外训练的情况下，能够显著地稳健地分类未见过的语言和不同的声学环境。在对预训练的Conformer模型在VoxLingua107数据集上进行微调后，我们实现了与当前最先进的语言识别系统类似的结果。此外，我们的模型使用的参数量仅为其它模型的五分之一。我们通过NVIDIA NeMo工具包开源了该模型。

    In this paper, we extend previous self-supervised approaches for language identification by experimenting with Conformer based architecture in a multilingual pre-training paradigm. We find that pre-trained speech models optimally encode language discriminatory information in lower layers. Further, we demonstrate that the embeddings obtained from these layers are significantly robust to classify unseen languages and different acoustic environments without additional training. After fine-tuning a pre-trained Conformer model on the VoxLingua107 dataset, we achieve results similar to current state-of-the-art systems for language identification. More, our model accomplishes this with 5x less parameters. We open-source the model through the NVIDIA NeMo toolkit.
    
[^11]: 去除噪音：心理声学和基于包络的特征在机械故障检测中的实证比较

    Cutting Through the Noise: An Empirical Comparison of Psychoacoustic and Envelope-based Features for Machinery Fault Detection. (arXiv:2211.01704v2 [eess.SP] UPDATED)

    [http://arxiv.org/abs/2211.01704](http://arxiv.org/abs/2211.01704)

    本文提出了一个自动化和噪声鲁棒的听觉检查系统，用于检测机械部件的健康状况。我们提供了一个基准来比较不同类型的包络特征与心理声学特征。我们是第一个应用时变心理声学特征进行故障检测的人。

    This paper presents an automated and noise-robust auditory inspection system for detecting the health condition of mechanical parts. A benchmark is provided to compare different types of envelope features with psychoacoustic features. The authors are the first to apply time-varying psychoacoustic features for fault detection.

    基于声学的故障检测具有监测机械部件健康状况的高潜力。然而，工业环境的背景噪音可能会对故障检测的性能产生负面影响。目前对于提高故障检测对工业环境噪声的鲁棒性的关注有限。因此，我们提出了Lenze生产背景噪声（LPBN）真实世界数据集和用于齿轮电机末端检查的自动化和噪声鲁棒的听觉检查（ARAI）系统。采用声学阵列从具有轻微故障、重大故障或健康的电机中获取数据。提供了一个基准来比较基于专家对齿轮箱的知识的不同类型的包络特征与心理声学特征。据我们所知，我们是第一个应用时变心理声学特征进行故障检测的人。我们训练了一种最先进的单类分类器，使用来自健康电机的样本进行训练。

    Acoustic-based fault detection has a high potential to monitor the health condition of mechanical parts. However, the background noise of an industrial environment may negatively influence the performance of fault detection. Limited attention has been paid to improving the robustness of fault detection against industrial environmental noise. Therefore, we present the Lenze production background-noise (LPBN) real-world dataset and an automated and noise-robust auditory inspection (ARAI) system for the end-of-line inspection of geared motors. An acoustic array is used to acquire data from motors with a minor fault, major fault, or which are healthy. A benchmark is provided to compare the psychoacoustic features with different types of envelope features based on expert knowledge of the gearbox. To the best of our knowledge, we are the first to apply time-varying psychoacoustic features for fault detection. We train a state-of-the-art one-class-classifier, on samples from healthy motors an
    
[^12]: 利用元数据和对比学习学习音频特征

    Learning Audio Features with Metadata and Contrastive Learning. (arXiv:2210.16192v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.16192](http://arxiv.org/abs/2210.16192)

    本研究使用监督对比学习结合可用元数据解决多个前置任务，学习数据的良好表示。在呼吸音分类数据集上，仅使用元数据学习表示可以获得与仅使用类标签的交叉熵相似的性能。在使用多个监督对比学习将类标签与元数据相结合时，获得了最先进的得分。

    This study uses supervised contrastive learning combined with available metadata to solve multiple pretext tasks that learn a good representation of data. Learning representations using only metadata obtains similar performance as using cross entropy with class labels only. State-of-the-art score is obtained when combining class labels with metadata using multiple supervised contrastive learning.

    基于注释的监督学习方法一直是分类问题的最先进技术，但是在低数据情况下，它们的泛化能力可能受到限制。本研究使用监督对比学习结合可用元数据解决多个前置任务，学习数据的良好表示。我们将我们的方法应用于ICBHI，这是一个适合这种情况的呼吸音分类数据集。我们表明，仅使用元数据学习表示，而不使用类标签，可以获得与仅使用这些标签的交叉熵相似的性能。此外，我们使用多个监督对比学习将类标签与元数据相结合时，获得了最先进的得分。这项工作表明，在监督对比设置中使用多个元数据源的潜力，特别是在类不平衡和少量数据的情况下。我们的代码已发布。

    Methods based on supervised learning using annotations in an end-to-end fashion have been the state-of-the-art for classification problems. However, they may be limited in their generalization capability, especially in the low data regime. In this study, we address this issue using supervised contrastive learning combined with available metadata to solve multiple pretext tasks that learn a good representation of data. We apply our approach on ICBHI, a respiratory sound classification dataset suited for this setting. We show that learning representations using only metadata, without class labels, obtains similar performance as using cross entropy with those labels only. In addition, we obtain state-of-the-art score when combining class labels with metadata using multiple supervised contrastive learning. This work suggests the potential of using multiple metadata sources in supervised contrastive settings, in particular in settings with class imbalance and few data. Our code is released 
    
[^13]: Articulation GAN: 无监督建模关节学习

    Articulation GAN: Unsupervised modeling of articulatory learning. (arXiv:2210.15173v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.15173](http://arxiv.org/abs/2210.15173)

    本文提出了一种新的无监督生成模型，通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式，从而更好地模拟人类语音产生的过程。

    This paper proposes a new unsupervised generative model that learns to generate articulatory representations (electromagnetic articulography or EMA) in a fully unsupervised manner, which more closely mimics human speech production and better simulates the process of human speech production.

    生成式深度神经网络广泛用于语音合成，但大多数现有模型直接生成波形或频谱输出。然而，人类通过控制关节来产生语音，这通过声音传播的物理特性导致语音声音的产生。我们引入了关节生成器到生成对抗网络范例中，这是一种新的无监督生成模型，用于语音产生/合成。关节生成器通过完全无监督的方式学习生成关节表示（电磁关节成像或EMA），更接近于人类语音产生的方式。然后，一个单独的预训练物理模型（ema2wav）将生成的EMA表示转换为语音波形，这些波形被发送到鉴别器进行评估。关节分析表明，网络学习控制关节的方式类似于人类在语音产生过程中的方式。输出的声学分析表明...

    Generative deep neural networks are widely used for speech synthesis, but most existing models directly generate waveforms or spectral outputs. Humans, however, produce speech by controlling articulators, which results in the production of speech sounds through physical properties of sound propagation. We introduce the Articulatory Generator to the Generative Adversarial Network paradigm, a new unsupervised generative model of speech production/synthesis. The Articulatory Generator more closely mimics human speech production by learning to generate articulatory representations (electromagnetic articulography or EMA) in a fully unsupervised manner. A separate pre-trained physical model (ema2wav) then transforms the generated EMA representations to speech waveforms, which get sent to the Discriminator for evaluation. Articulatory analysis suggests that the network learns to control articulators in a similar manner to humans during speech production. Acoustic analysis of the outputs sugge
    
[^14]: 回放：迭代注意力用于音频识别

    Play It Back: Iterative Attention for Audio Recognition. (arXiv:2210.11328v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.11328](http://arxiv.org/abs/2210.11328)

    该论文提出了一种基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注，最终实现了在三个音频分类基准测试中始终实现最先进的性能。

    The paper proposes an end-to-end attention-based architecture that attends over the most discriminative sounds across the audio sequence through selective repetition, achieving consistently state-of-the-art performance across three audio-classification benchmarks.

    听觉认知的一个关键功能是随着时间的推移将特征声音与其相应的语义关联起来。人类试图区分细粒度音频类别时，通常会重播相同的区分性声音以增加其预测置信度。我们提出了一种端到端的基于注意力的架构，通过选择性重复跨越音频序列的最具区分性的声音来进行关注。我们的模型最初使用完整的音频序列，并通过插槽注意力迭代地细化重播的时间段。在每次播放时，所选段使用较小的跳跃长度重播，这代表了这些段内更高分辨率的特征。我们展示了我们的方法可以在三个音频分类基准测试中始终实现最先进的性能：AudioSet、VGG-Sound和EPIC-KITCHENS-100。

    A key function of auditory cognition is the association of characteristic sounds with their corresponding semantics over time. Humans attempting to discriminate between fine-grained audio categories, often replay the same discriminative sounds to increase their prediction confidence. We propose an end-to-end attention-based architecture that through selective repetition attends over the most discriminative sounds across the audio sequence. Our model initially uses the full audio sequence and iteratively refines the temporal segments replayed based on slot attention. At each playback, the selected segments are replayed using a smaller hop length which represents higher resolution features within these segments. We show that our method can consistently achieve state-of-the-art performance across three audio-classification benchmarks: AudioSet, VGG-Sound, and EPIC-KITCHENS-100.
    
[^15]: PSVRF: 无参考学习还原变调语音

    PSVRF: Learning to restore Pitch-Shifted Voice without reference. (arXiv:2210.02731v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2210.02731](http://arxiv.org/abs/2210.02731)

    本文提出了一种无参考方法PSVRF，用于高质量还原变调语音，可以增强ASV系统对音高缩放攻击的鲁棒性，性能甚至超过了最先进的基于参考的方法。

    This paper proposes a no-reference approach called PSVRF for high-quality restoration of pitch-shifted voice, which enhances the robustness of ASV systems to pitch-scaling attacks and even outperforms the state-of-the-art reference-based approach.

    音高缩放算法对自动说话人验证（ASV）系统的安全性有重要影响。虽然已经提出了许多反欺骗算法来识别变调语音并将其恢复到原始版本，但它们要么性能较差，要么需要原始语音作为参考，限制了应用前景。本文提出了一种无参考方法PSVRF，用于高质量还原变调语音。在AISHELL-1和AISHELL-3上的实验表明，PSVRF可以恢复被各种音高缩放技术伪装的语音，显然增强了ASV系统对音高缩放攻击的鲁棒性。此外，PSVRF的性能甚至超过了最先进的基于参考的方法。

    Pitch scaling algorithms have a significant impact on the security of Automatic Speaker Verification (ASV) systems. Although numerous anti-spoofing algorithms have been proposed to identify the pitch-shifted voice and even restore it to the original version, they either have poor performance or require the original voice as a reference, limiting the prospects of applications. In this paper, we propose a no-reference approach termed PSVRF$^1$ for high-quality restoration of pitch-shifted voice. Experiments on AISHELL-1 and AISHELL-3 demonstrate that PSVRF can restore the voice disguised by various pitch-scaling techniques, which obviously enhances the robustness of ASV systems to pitch-scaling attacks. Furthermore, the performance of PSVRF even surpasses that of the state-of-the-art reference-based approach.
    
[^16]: Uconv-Conformer: 针对端到端语音识别的输入序列长度大幅缩减的新型架构

    Uconv-Conformer: High Reduction of Input Sequence Length for End-to-End Speech Recognition. (arXiv:2208.07657v3 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2208.07657](http://arxiv.org/abs/2208.07657)

    本文提出了一种新型Uconv-Conformer架构，可以将输入序列长度缩短16倍，加速中间层的工作，同时通过使用上采样块解决了收敛问题，表现出更好的WER和更快的训练和推理速度。

    The paper proposes a new Uconv-Conformer architecture that reduces the input sequence length by 16 times, speeds up the work of intermediate layers, and solves the convergence issue by using upsampling blocks. The Uconv-Conformer architecture shows better WER and faster training and inference speed.

    优化现代ASR架构是最高优先级的任务之一，因为它可以节省模型训练和推理的许多计算资源。本文提出了一种基于标准Conformer模型的新型Uconv-Conformer架构。它通过16倍的一致性缩短输入序列长度，从而加速了中间层的工作。为了解决与时间维度大幅缩减相关的收敛问题，我们使用了像U-Net架构中的上采样块来确保正确的CTC损失计算和稳定网络训练。Uconv-Conformer架构不仅在训练和推理速度方面更快，而且与基线Conformer相比，表现出更好的WER。我们最好的Uconv-Conformer模型在CPU和GPU上分别显示出47.8％和23.5％的推理加速。相对WER的减少分别为7.3％和9.2％。

    Optimization of modern ASR architectures is among the highest priority tasks since it saves many computational resources for model training and inference. The work proposes a new Uconv-Conformer architecture based on the standard Conformer model. It consistently reduces the input sequence length by 16 times, which results in speeding up the work of the intermediate layers. To solve the convergence issue connected with such a significant reduction of the time dimension, we use upsampling blocks like in the U-Net architecture to ensure the correct CTC loss calculation and stabilize network training. The Uconv-Conformer architecture appears to be not only faster in terms of training and inference speed but also shows better WER compared to the baseline Conformer. Our best Uconv-Conformer model shows 47.8% and 23.5% inference acceleration on the CPU and GPU, respectively. Relative WER reduction is 7.3% and 9.2% on LibriSpeech test_clean and test_other respectively.
    
[^17]: 日语ASR中基于音节和字符目标的交替中间条件

    Alternate Intermediate Conditioning with Syllable-level and Character-level Targets for Japanese ASR. (arXiv:2204.00175v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2204.00175](http://arxiv.org/abs/2204.00175)

    该论文提出了一种基于音节和字符目标的交替中间条件方法，利用字符级和音节级中间预测作为条件特征来处理日语ASR中的多对一和一对多的映射问题，并在实验中取得了优异的表现。

    This paper proposes an alternate intermediate conditioning method with syllable-level and character-level targets to deal with the many-to-one and one-to-many mapping problems in Japanese ASR, and achieves better performance than conventional multi-task and Self-conditioned CTC methods in experiments.

    端到端的自动语音识别直接将输入语音映射到字符。然而，当多个不同的发音应该映射到一个字符或一个发音被多个不同的字符共享时，映射可能会出现问题。由于日语汉字的存在，日语ASR最容易遭受这种多对一和一对多的映射问题。为了缓解这些问题，我们引入了字符和音节之间的显式交互，使用自我条件连接主义时间分类（CTC），其中上层“自我条件”于下层的中间预测。所提出的方法利用字符级和音节级中间预测作为条件特征来处理字符和音节之间的相互依赖关系。在自发日语语料库上的实验结果表明，所提出的方法优于传统的多任务和自我条件CTC方法。

    End-to-end automatic speech recognition directly maps input speech to characters. However, the mapping can be problematic when several different pronunciations should be mapped into one character or when one pronunciation is shared among many different characters. Japanese ASR suffers the most from such many-to-one and one-to-many mapping problems due to Japanese kanji characters. To alleviate the problems, we introduce explicit interaction between characters and syllables using Self-conditioned connectionist temporal classification (CTC), in which the upper layers are ``self-conditioned'' on the intermediate predictions from the lower layers. The proposed method utilizes character-level and syllable-level intermediate predictions as conditioning features to deal with mutual dependency between characters and syllables. Experimental results on Corpus of Spontaneous Japanese show that the proposed method outperformed the conventional multi-task and Self-conditioned CTC methods.
    

