# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652) | Yi模型系列基于强大的多维能力，通过基于6B和34B预训练模型的扩展，包括聊天模型、长上下文模型、深度放大模型和视觉语言模型，取得了优异的性能。 |
| [^2] | [Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization](https://arxiv.org/abs/2402.11414) | 提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。 |
| [^3] | [Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities](https://arxiv.org/abs/2402.10835) | 本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。 |
| [^4] | [In-Context Learning with Iterative Demonstration Selection.](http://arxiv.org/abs/2310.09881) | 这项研究提出了一种基于迭代示范选择的上下文学习方法，通过使用零样本链式思维推理来选择与测试样本不同但仍与之强相关的示范作为学习的上下文。 |
| [^5] | [Demystifying CLIP Data.](http://arxiv.org/abs/2309.16671) | CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。 |
| [^6] | [An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.](http://arxiv.org/abs/2308.08747) | 该研究实证评估了大型语言模型在持续微调过程中的灾难性遗忘现象，并发现随着模型规模增加，遗忘的严重程度也加剧。与编码器-解码器模型相比，仅有解码器的模型遗忘较少并保留更多知识。此外，研究还发现LLMs可以减轻语言偏见，并且ALPACA在保留知识和容量方面具有优势。 |
| [^7] | [Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey).](http://arxiv.org/abs/2307.10246) | 本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。 |
| [^8] | [Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent.](http://arxiv.org/abs/2304.09542) | 本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。 |

# 详细

[^1]: Yi: 由 01.AI 推出的开放基础模型

    Yi: Open Foundation Models by 01.AI

    [https://arxiv.org/abs/2403.04652](https://arxiv.org/abs/2403.04652)

    Yi模型系列基于强大的多维能力，通过基于6B和34B预训练模型的扩展，包括聊天模型、长上下文模型、深度放大模型和视觉语言模型，取得了优异的性能。

    

    我们介绍了Yi模型系列，这是一系列具有强大多维能力的语言和多模态模型。Yi模型系列基于6B和34B的预训练语言模型，然后我们将它们扩展为聊天模型、200K长上下文模型、深度放大模型和视觉语言模型。我们的基础模型在诸如MMLU之类的各种基准测试中表现出色，而我们微调过的聊天模型在AlpacaEval和Chatbot Arena等主要评估平台上具有较高的人类偏好率。通过依赖于我们的可扩展超级计算基础设施和经典的Transformer架构，我们认为Yi模型的性能主要归因于其数据质量，这是由我们的数据工程工作所带来的。对于预训练，我们使用级联的数据去重和质量过滤流水线构建了3100亿个英文和中文语料库的标记。对于微调，我们对小规模模型进行了改进

    arXiv:2403.04652v1 Announce Type: cross  Abstract: We introduce the Yi model family, a series of language and multimodal models that demonstrate strong multi-dimensional capabilities. The Yi model family is based on 6B and 34B pretrained language models, then we extend them to chat models, 200K long context models, depth-upscaled models, and vision-language models. Our base models achieve strong performance on a wide range of benchmarks like MMLU, and our finetuned chat models deliver strong human preference rate on major evaluation platforms like AlpacaEval and Chatbot Arena. Building upon our scalable super-computing infrastructure and the classical transformer architecture, we attribute the performance of Yi models primarily to its data quality resulting from our data-engineering efforts. For pretraining, we construct 3.1 trillion tokens of English and Chinese corpora using a cascaded data deduplication and quality filtering pipeline. For finetuning, we polish a small scale (less th
    
[^2]: 用于多模态摘要的细粒度可解释事实评估

    Fine-grained and Explainable Factuality Evaluation for Multimodal Summarization

    [https://arxiv.org/abs/2402.11414](https://arxiv.org/abs/2402.11414)

    提出两种细粒度和可解释的评估框架，用于评估多模态摘要模型的事实性，其中无参考事实性评估框架具有更广泛的应用场景，实验证实了方法的有效性。

    

    多模态摘要旨在生成基于输入文本和图像的简洁摘要。然而，现有方法可能存在事实性输出的问题。为了评估多模态摘要模型的事实性，我们提出了两种细粒度和可解释的评估框架（FALLACIOUS）用于不同的应用场景，即基于参考的事实性评估框架和无参考的事实性评估框架。值得注意的是，无参考事实性评估框架不需要基准真值，因此具有更广泛的应用场景。为了评估所提出框架的有效性，我们计算了我们的框架与其他指标之间的相关性。实验结果显示了我们提出方法的有效性。我们将通过GitHub发布我们的代码和数据集。

    arXiv:2402.11414v1 Announce Type: new  Abstract: Multimodal summarization aims to generate a concise summary based on the input text and image. However, the existing methods potentially suffer from unfactual output. To evaluate the factuality of multimodal summarization models, we propose two fine-grained and explainable evaluation frameworks (FALLACIOUS) for different application scenarios, i.e. reference-based factuality evaluation framework and reference-free factuality evaluation framework. Notably, the reference-free factuality evaluation framework doesn't need ground truth and hence it has a wider application scenario. To evaluate the effectiveness of the proposed frameworks, we compute the correlation between our frameworks and the other metrics. The experimental results show the effectiveness of our proposed method. We will release our code and dataset via github.
    
[^3]: LLMs下的时间序列预测：理解和增强模型能力

    Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities

    [https://arxiv.org/abs/2402.10835](https://arxiv.org/abs/2402.10835)

    本研究通过比较LLMs与传统模型，发现了LLMs在时间序列预测中的优势和局限性，指出LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战，同时指出融入外部知识和采用自然语言释义有助于提升LLMs在时间序列预测中的性能。

    

    大语言模型(LLMs)近年来在许多领域得到迅速发展。作为一种经典的机器学习任务，时间序列预测最近从LLMs中获得了推动。然而，在这一领域，LLMs的偏好存在研究空白。通过将LLMs与传统模型进行比较，发现了LLMs在时间序列预测中的许多特性。例如，我们的研究表明，LLMs在预测具有明显模式和趋势的时间序列方面表现出色，但在缺乏周期性的数据集方面面临挑战。我们通过设计提示要求LLMs告知数据集的周期来解释我们的发现。此外，本文还研究了输入策略，发现融入外部知识和采用自然语言释义积极影响了LLMs在时间序列预测中的预测性能。总的来说，这项研究有助于洞察LLMs在时间序列预测中的优势和局限性。

    arXiv:2402.10835v1 Announce Type: new  Abstract: Large language models (LLMs) have been applied in many fields with rapid development in recent years. As a classic machine learning task, time series forecasting has recently received a boost from LLMs. However, there is a research gap in the LLMs' preferences in this field. In this paper, by comparing LLMs with traditional models, many properties of LLMs in time series prediction are found. For example, our study shows that LLMs excel in predicting time series with clear patterns and trends but face challenges with datasets lacking periodicity. We explain our findings through designing prompts to require LLMs to tell the period of the datasets. In addition, the input strategy is investigated, and it is found that incorporating external knowledge and adopting natural language paraphrases positively affects the predictive performance of LLMs for time series. Overall, this study contributes to insight into the advantages and limitations of
    
[^4]: 基于迭代示范选择的上下文学习

    In-Context Learning with Iterative Demonstration Selection. (arXiv:2310.09881v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.09881](http://arxiv.org/abs/2310.09881)

    这项研究提出了一种基于迭代示范选择的上下文学习方法，通过使用零样本链式思维推理来选择与测试样本不同但仍与之强相关的示范作为学习的上下文。

    

    受到规模的进展的推动，大型语言模型(LLMs)通过上下文学习(ICL)展示了强大的少样本学习能力。然而，ICL的性能已被证明对于示范的选择非常敏感。选择最合适的示范作为上下文仍然是一个持续挑战和一个未解决的问题。现有文献已经强调了选择那些与测试样本不同或语义相似性的示范的重要性，而忽视了最优示范选择维度是任务特定的事实。借鉴两个维度的优点，我们提出了迭代示范选择(IDS)。使用零样本链式思维推理(Zero-shot-CoT)，IDS迭代地选择那些与测试样本不同但仍与之强相关的示范作为ICL的示范。具体而言，IDS在示范选择之前将Zero-shot-CoT应用于测试样本。输出的推理路径是...

    Spurred by advancements in scale, large language models (LLMs) have demonstrated strong few-shot learning ability via in-context learning (ICL). However, the performance of ICL has been shown to be highly sensitive to the selection of few-shot demonstrations. Selecting the most suitable examples as context remains an ongoing challenge and an open problem. Existing literature has highlighted the importance of selecting examples that are diverse or semantically similar to the test sample while ignoring the fact that the optimal selection dimension, i.e., diversity or similarity, is task-specific. Leveraging the merits of both dimensions, we propose Iterative Demonstration Selection (IDS). Using zero-shot chain-of-thought reasoning (Zero-shot-CoT), IDS iteratively selects examples that are diverse but still strongly correlated with the test sample as ICL demonstrations. Specifically, IDS applies Zero-shot-CoT to the test sample before demonstration selection. The output reasoning path is 
    
[^5]: 揭秘CLIP数据

    Demystifying CLIP Data. (arXiv:2309.16671v1 [cs.CV])

    [http://arxiv.org/abs/2309.16671](http://arxiv.org/abs/2309.16671)

    CLIP的成功主要归功于其数据而非模型架构或预训练目标。我们通过元数据整理方法引入了MetaCLIP，该方法从原始数据池和元数据中生成一个平衡的子集，提供了更加详细的数据信息。在实验中，我们发现MetaCLIP在处理400M个图像-文本数据对时取得了良好的性能。

    

    对比语言-图像预训练（CLIP）是一种推动计算机视觉研究和应用的方法，为现代识别系统和生成模型注入了活力。我们认为，CLIP成功的主要因素是其数据，而不是模型架构或预训练目标。然而，CLIP只提供了关于其数据和如何收集数据的非常有限的信息，导致其他研究努力通过使用模型参数进行过滤来重现CLIP的数据。在这项工作中，我们意在揭示CLIP的数据整理方法，并在公开给社区的过程中引入元数据整理的语言-图像预训练（MetaCLIP）。MetaCLIP通过对元数据分布进行平衡，从原始数据池和元数据（从CLIP的概念中得出）中产生一个平衡的子集。我们的实验研究严格隔离了模型和训练设置，仅专注于数据。MetaCLIP应用于包含400M图像-文本数据对的CommonCrawl，并获得了较好的性能。

    Contrastive Language-Image Pre-training (CLIP) is an approach that has advanced research and applications in computer vision, fueling modern recognition systems and generative models. We believe that the main ingredient to the success of CLIP is its data and not the model architecture or pre-training objective. However, CLIP only provides very limited information about its data and how it has been collected, leading to works that aim to reproduce CLIP's data by filtering with its model parameters. In this work, we intend to reveal CLIP's data curation approach and in our pursuit of making it open to the community introduce Metadata-Curated Language-Image Pre-training (MetaCLIP). MetaCLIP takes a raw data pool and metadata (derived from CLIP's concepts) and yields a balanced subset over the metadata distribution. Our experimental study rigorously isolates the model and training settings, concentrating solely on data. MetaCLIP applied to CommonCrawl with 400M image-text data pairs outper
    
[^6]: 大型语言模型在持续微调过程中的灾难性遗忘的实证研究

    An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning. (arXiv:2308.08747v1 [cs.CL])

    [http://arxiv.org/abs/2308.08747](http://arxiv.org/abs/2308.08747)

    该研究实证评估了大型语言模型在持续微调过程中的灾难性遗忘现象，并发现随着模型规模增加，遗忘的严重程度也加剧。与编码器-解码器模型相比，仅有解码器的模型遗忘较少并保留更多知识。此外，研究还发现LLMs可以减轻语言偏见，并且ALPACA在保留知识和容量方面具有优势。

    

    灾难性遗忘（CF）是机器学习中的一种现象，当模型学习新信息时，它会忘记先前学到的信息。由于大型语言模型（LLMs）显示出了出色的性能，探究LLMs在持续微调中是否存在CF是很有意义的。在这项研究中，我们从领域知识、推理和阅读理解的角度对LLMs的遗忘现象进行了实证评估。实验表明，从1b到7b的范围内，LLMs普遍存在灾难性遗忘现象，并且随着规模的增加，遗忘的严重程度也加剧。与编码器-解码器模型mT0相比，仅有解码器的模型BLOOMZ遗忘较少并保留更多知识。我们还观察到，在持续微调过程中，LLMs可以减轻语言偏见（如性别偏见）。此外，我们发现与LLAMA相比，ALPACA在保留更多知识和容量方面具有优势。

    Catastrophic forgetting (CF) is a phenomenon that occurs in machine learning when a model forgets previously learned information as it learns new information. As large language models (LLMs) have shown excellent performance, it is interesting to uncover whether CF exists in the continual fine-tuning of LLMs. In this study, we empirically evaluate the forgetting phenomenon in LLMs' knowledge, from the perspectives of domain knowledge, reasoning, and reading comprehension. The experiments demonstrate that catastrophic forgetting is generally observed in LLMs ranging from 1b to 7b. Furthermore, as the scale increases, the severity of forgetting also intensifies. Comparing the decoder-only model BLOOMZ with the encoder-decoder model mT0, BLOOMZ suffers less forgetting and maintains more knowledge. We also observe that LLMs can mitigate language bias (e.g. gender bias) during continual fine-tuning. Moreover, we find that ALPACA can maintain more knowledge and capacity compared with LLAMA du
    
[^7]: 深度神经网络和脑对齐：脑编码和解码（综述）

    Deep Neural Networks and Brain Alignment: Brain Encoding and Decoding (Survey). (arXiv:2307.10246v1 [q-bio.NC])

    [http://arxiv.org/abs/2307.10246](http://arxiv.org/abs/2307.10246)

    本文综述了深度神经网络和脑对齐的研究，重点在于脑编码和解码模型的应用。这些模型对于理解大脑的信息处理机制以及设计脑机接口具有重要意义。

    

    大脑如何表示不同的信息模式？我们能否设计出一个可以自动理解用户思考内容的系统？这些问题可以通过研究功能磁共振成像（fMRI）等大脑记录来回答。作为第一步，神经科学界为被动阅读/听觉/观看概念词汇、叙述、图片和电影相关的认知神经科学数据集作出了贡献。过去二十年中，还提出了使用这些数据集的编码和解码模型。这些模型作为基础研究中的额外工具，在认知科学和神经科学领域有着多种实际应用。编码模型旨在自动地生成fMRI大脑表征，给定一个刺激。它们在评估和诊断神经系统疾病以及设计大脑损伤治疗方法方面有着多种实际应用。解码模型解决了根据fMRI重构刺激的逆问题。它们对于理解大脑如何处理信息以及设计脑机接口的发展都有着重要意义。

    How does the brain represent different modes of information? Can we design a system that automatically understands what the user is thinking? Such questions can be answered by studying brain recordings like functional magnetic resonance imaging (fMRI). As a first step, the neuroscience community has contributed several large cognitive neuroscience datasets related to passive reading/listening/viewing of concept words, narratives, pictures and movies. Encoding and decoding models using these datasets have also been proposed in the past two decades. These models serve as additional tools for basic research in cognitive science and neuroscience. Encoding models aim at generating fMRI brain representations given a stimulus automatically. They have several practical applications in evaluating and diagnosing neurological conditions and thus also help design therapies for brain damage. Decoding models solve the inverse problem of reconstructing the stimuli given the fMRI. They are useful for 
    
[^8]: 大型语言模型在信息检索中的排名能力研究——以ChatGPT为例

    Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agent. (arXiv:2304.09542v1 [cs.CL])

    [http://arxiv.org/abs/2304.09542](http://arxiv.org/abs/2304.09542)

    本文研究了生成性LLMs，如ChatGPT和GPT-4在信息检索中的相关性排名能力，实验结果表明，这些模型经适当指导后表现优异，有时甚至优于传统监督学习方法。将ChatGPT的排名能力提炼为专门模型在BEIR上的效果更优。

    

    大型语言模型（LLMs）已经证明具有remarkable能力，能够将一些零样本语言任务推广至其他领域。本文研究了ChatGPT和GPT-4等生成性LLMs的相关性排名在信息检索方面的能力。实验结果显示，经过适当的指导，ChatGPT和GPT-4可以在流行的信息检索基准上取得竞争优势，甚至有时优于监督学习方法。特别地，GPT-4在TREC数据集上的平均nDCG上表现优于完全微调的monoT5-3B，BEIR数据集上的平均nDCG上优于monoT5-3B 2.3个点，低资源语言Mr.TyDi上的平均nDCG上优于monoT5-3B 2.7个点。随后，我们探讨了将ChatGPT的排名能力提炼为专门的模型的潜力。我们训练的小型专门模型（训练于10K个ChatGPT生成的数据）在BEIR上的表现优于在400K个MS MARCO注释数据上训练的monoT5。代码可在www.github.com/sunnwe上复现。

    Large Language Models (LLMs) have demonstrated a remarkable ability to generalize zero-shot to various language-related tasks. This paper focuses on the study of exploring generative LLMs such as ChatGPT and GPT-4 for relevance ranking in Information Retrieval (IR). Surprisingly, our experiments reveal that properly instructed ChatGPT and GPT-4 can deliver competitive, even superior results than supervised methods on popular IR benchmarks. Notably, GPT-4 outperforms the fully fine-tuned monoT5-3B on MS MARCO by an average of 2.7 nDCG on TREC datasets, an average of 2.3 nDCG on eight BEIR datasets, and an average of 2.7 nDCG on ten low-resource languages Mr.TyDi. Subsequently, we delve into the potential for distilling the ranking capabilities of ChatGPT into a specialized model. Our small specialized model that trained on 10K ChatGPT generated data outperforms monoT5 trained on 400K annotated MS MARCO data on BEIR. The code to reproduce our results is available at www.github.com/sunnwe
    

