# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Consideration of AI Openness: Can Good Intent Be Abused?](https://arxiv.org/abs/2403.06537) | 开源技术虽然促进了科技进步，但也存在滥用风险，研究发现开源语言模型可以被调整用于提供有关犯罪活动的不道德且具有信息性的答案。 |
| [^2] | [Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models](https://arxiv.org/abs/2402.16315) | Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。 |
| [^3] | [Scaling Efficient LLMs](https://arxiv.org/abs/2402.14746) | 训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。 |
| [^4] | [OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement](https://arxiv.org/abs/2402.14658) | OpenCodeInterpreter是一种开源代码系统，集成了执行、人类反馈和动态代码细化的功能，并在关键基准测试中表现出色，甚至与GPT-4相媲美。 |
| [^5] | [Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models](https://arxiv.org/abs/2402.11140) | 本文提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，通过迭代地探索和自我评估多个思维树，获得一系列试错推理经验，作为解决复杂问题的新形式的提示。 |
| [^6] | [Multi: Multimodal Understanding Leaderboard with Text and Images](https://arxiv.org/abs/2402.03173) | Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。 |
| [^7] | [Human-Readable Fingerprint for Large Language Models](https://arxiv.org/abs/2312.04828) | 这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。 |
| [^8] | [Continuously Learning New Words in Automatic Speech Recognition.](http://arxiv.org/abs/2401.04482) | 该论文提出了一种自我监督的持续学习方法，用于解决自动语音识别中识别新词的问题。通过对讲座录音进行推理和收集包含新词的话语，然后在自适应数据集上进行持续学习，可以在新词出现频率较高时提高性能，同时保持整体性能。 |
| [^9] | [Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition.](http://arxiv.org/abs/2309.10524) | 本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。 |

# 详细

[^1]: 对AI开放性的考量: 善意能否被滥用？

    On the Consideration of AI Openness: Can Good Intent Be Abused?

    [https://arxiv.org/abs/2403.06537](https://arxiv.org/abs/2403.06537)

    开源技术虽然促进了科技进步，但也存在滥用风险，研究发现开源语言模型可以被调整用于提供有关犯罪活动的不道德且具有信息性的答案。

    

    开放性对科学的进展至关重要。特别是，最近人工智能的快速进展仅可能通过各种开源模型、数据集和库。然而，这种开放性也意味着技术可以被自由地用于社会有害目的。开源模型或数据集可以被用于恶意目的吗？如果是这样，那么技术被用于这些目的会有多容易？本文在法律领域进行了一个案例研究，这是一个个人决策可能产生深远社会后果的领域。为此，我们构建了EVE数据集，包含了200个关于犯罪活动的问题和对应答案，基于200个韩国先例。我们发现一个广泛认可的开源LLM，最初拒绝回答不道德问题，可以很容易地通过EVE进行调整，提供关于犯罪活动的不道德且具有信息性的答案。这意味着，尽管开源技术能够促进科学技术进步，但同时也带来了滥用风险。

    arXiv:2403.06537v1 Announce Type: new  Abstract: Openness is critical for the advancement of science. In particular, recent rapid progress in AI has been made possible only by various open-source models, datasets, and libraries. However, this openness also means that technologies can be freely used for socially harmful purposes. Can open-source models or datasets be used for malicious purposes? If so, how easy is it to adapt technology for such goals? Here, we conduct a case study in the legal domain, a realm where individual decisions can have profound social consequences. To this end, we build EVE, a dataset consisting of 200 examples of questions and corresponding answers about criminal activities based on 200 Korean precedents. We found that a widely accepted open-source LLM, which initially refuses to answer unethical questions, can be easily tuned with EVE to provide unethical and informative answers about criminal activities. This implies that although open-source technologies c
    
[^2]: Finer: 在大型视觉语言模型中研究和增强细粒度视觉概念识别

    Finer: Investigating and Enhancing Fine-Grained Visual Concept Recognition in Large Vision Language Models

    [https://arxiv.org/abs/2402.16315](https://arxiv.org/abs/2402.16315)

    Finer工作揭示了大型视觉语言模型在细粒度视觉分类上的短板，尤其是难以生成准确的细致属性解释，尽管具有生成高水平图像解释的能力。

    

    最近指导调整的大型视觉语言模型（LVLMs）的进展使模型能够轻松生成高水平的基于图像的解释。尽管这种能力主要归因于大型语言模型（LLMs）中包含的丰富世界知识，但我们的工作揭示了它们在六个不同基准设置下的细粒度视觉分类（FGVC）上的缺陷。最近的LVLMs最先进的模型，如LLaVa-1.5，InstructBLIP和GPT-4V，在分类性能方面严重下降，例如，LLaVA-1.5在斯坦福狗的EM平均下降了65.58，而且还难以根据出现在输入图像中的概念生成具有详细属性的准确解释，尽管它们有生成整体图像级描述的能力。深入分析表明，经过指导调整的LVLMs在给定文本时呈现出模态差距，显示出存在不一致性

    arXiv:2402.16315v1 Announce Type: cross  Abstract: Recent advances in instruction-tuned Large Vision-Language Models (LVLMs) have imbued the models with the ability to generate high-level, image-grounded explanations with ease. While such capability is largely attributed to the rich world knowledge contained within the Large Language Models (LLMs), our work reveals their shortcomings in fine-grained visual categorization (FGVC) across six different benchmark settings. Most recent state-of-the-art LVLMs like LLaVa-1.5, InstructBLIP and GPT-4V not only severely deteriorate in terms of classification performance, e.g., average drop of 65.58 in EM for Stanford Dogs for LLaVA-1.5, but also struggle to generate an accurate explanation with detailed attributes based on the concept that appears within an input image despite their capability to generate holistic image-level descriptions. In-depth analyses show that instruction-tuned LVLMs exhibit modality gap, showing discrepancy when given tex
    
[^3]: 扩展高效的LLM模型

    Scaling Efficient LLMs

    [https://arxiv.org/abs/2402.14746](https://arxiv.org/abs/2402.14746)

    训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。

    

    训练得到的LLM模型通常是稀疏的，即大部分参数为零，这引发了关于效率的问题。为此，我们研究了高效的LLM模型，即那些在训练语料上达到所需准确度的参数最少。具体地，我们比较了当前规模下训练损失的理论和实证估计，以获得自然训练语料中独特序列数量上下界的数量。我们的结果暗示：(1)要在训练语料中表示的技能数量翻倍，需要将语料规模大约扩展三到五倍，(2)对于高效的LLM模型，参数数量$N$和自然训练语料规模$D$满足$N \sim D^{0.58}$的关系，(3)如果一个LLM模型的参数数量小于训练语料中的独特序列数量，扩展可以揭示出新的技能。

    arXiv:2402.14746v1 Announce Type: new  Abstract: Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss at current scale to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale roughly between three and five fold (2) for efficient LLMs, the number of parameters $N$ and the size $D$ of a natural training corpus scale as $N \sim D^{0.58}$ (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    
[^4]: OpenCodeInterpreter：集成代码生成、执行和细化

    OpenCodeInterpreter: Integrating Code Generation with Execution and Refinement

    [https://arxiv.org/abs/2402.14658](https://arxiv.org/abs/2402.14658)

    OpenCodeInterpreter是一种开源代码系统，集成了执行、人类反馈和动态代码细化的功能，并在关键基准测试中表现出色，甚至与GPT-4相媲美。

    

    大型语言模型的引入显著推动了代码生成的发展。然而，开源模型通常缺乏类似GPT-4 Code Interpreter这样的高级系统的执行能力和迭代细化能力。为了解决这一问题，我们介绍了OpenCodeInterpreter，这是一族旨在生成、执行和迭代细化代码的开源代码系统。通过Code-Feedback支持，该系统集成了执行和人类反馈，用于动态代码细化。我们对OpenCodeInterpreter在诸如HumanEval、MBPP以及它们来自EvalPlus的增强版本等关键基准上进行了全面评估，证实了其出色的性能。值得注意的是，OpenCodeInterpreter-33B在HumanEval和MBPP的平均值（以及其增强版本）上取得了83.2（76.4）的准确率，与GPT-4的84.2（76.2）紧密匹敌，并且通过合成hum

    arXiv:2402.14658v1 Announce Type: cross  Abstract: The introduction of large language models has significantly advanced code generation. However, open-source models often lack the execution capabilities and iterative refinement of advanced systems like the GPT-4 Code Interpreter. To address this, we introduce OpenCodeInterpreter, a family of open-source code systems designed for generating, executing, and iteratively refining code. Supported by Code-Feedback, a dataset featuring 68K multi-turn interactions, OpenCodeInterpreter integrates execution and human feedback for dynamic code refinement. Our comprehensive evaluation of OpenCodeInterpreter across key benchmarks such as HumanEval, MBPP, and their enhanced versions from EvalPlus reveals its exceptional performance. Notably, OpenCodeInterpreter-33B achieves an accuracy of 83.2 (76.4) on the average (and plus versions) of HumanEval and MBPP, closely rivaling GPT-4's 84.2 (76.2) and further elevates to 91.6 (84.6) with synthesized hum
    
[^5]: 思维的提升：使用大型语言模型进行试错问题解决

    Boosting of Thoughts: Trial-and-Error Problem Solving with Large Language Models

    [https://arxiv.org/abs/2402.11140](https://arxiv.org/abs/2402.11140)

    本文提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，通过迭代地探索和自我评估多个思维树，获得一系列试错推理经验，作为解决复杂问题的新形式的提示。

    

    大型语言模型（LLMs）在各种问题上的推理性能关键取决于思维链提示，其中包括在提示中提供一些思维链示范作为示例。最近的工作（例如Thought Tree）指出了在复杂问题解决的推理步骤选择中，探索和自我评估的重要性。在本文中，我们提出了一种名为Boosting of Thoughts（BoT）的自动提示框架，用于通过迭代地探索和自我评估许多思维树来获得一系列试错推理经验，这将作为解决复杂问题的新形式的提示。BoT从一个简单提示开始，无需示例，迭代地探索和评估大量的推理步骤，更重要的是，利用LLM获得的错误分析来明确修改提示。

    arXiv:2402.11140v1 Announce Type: new  Abstract: The reasoning performance of Large Language Models (LLMs) on a wide range of problems critically relies on chain-of-thought prompting, which involves providing a few chain of thought demonstrations as exemplars in prompts. Recent work, e.g., Tree of Thoughts, has pointed out the importance of exploration and self-evaluation in reasoning step selection for complex problem solving. In this paper, we present Boosting of Thoughts (BoT), an automated prompting framework for problem solving with LLMs by iteratively exploring and self-evaluating many trees of thoughts in order to acquire an ensemble of trial-and-error reasoning experiences, which will serve as a new form of prompting to solve the complex problem. Starting from a simple prompt without requiring examples, BoT iteratively explores and evaluates a large collection of reasoning steps, and more importantly, uses error analysis obtained from the LLM on them to explicitly revise prompt
    
[^6]: 多模态：文本和图像的多模态理解排行榜

    Multi: Multimodal Understanding Leaderboard with Text and Images

    [https://arxiv.org/abs/2402.03173](https://arxiv.org/abs/2402.03173)

    Multi是一个多模态理解的排行榜，提供了一个综合数据集，评估多模态大型语言模型对理解复杂图表和科学问题的能力。它兼具准确和开放式的回答形式，挑战MLLM的各种任务，并包含超过18,000个问题。

    

    多模态大型语言模型（MLLM）的快速进展强调了向学术界引入具有挑战性而又真实的基准的需求。现有的基准主要关注简单的自然图像理解，但Multi成为了MLLM的尖端基准，提供了一个综合性的数据集，用于评估MLLM对理解复杂图表和科学问题的能力。该基准反映了当前真实的考试风格，提供多模态的输入，并要求准确或开放式的回答，类似于现实中的学校考试。它通过各种任务挑战MLLM，从公式推导到图像细节分析，以及跨模态推理。Multi包括超过18,000个问题，重点关注不同格式的基于科学的问答。我们还引入了Multi-Elite，一个包含500个问题的子集，用于测试MLLM的极端情况，以及Multi-Extend，通过超过4..。

    Rapid progress in multimodal large language models (MLLMs) highlights the need to introduce challenging yet realistic benchmarks to the academic community. Existing benchmarks primarily focus on simple natural image understanding, but Multi emerges as a cutting-edge benchmark for MLLMs, offering a comprehensive dataset for evaluating MLLMs against understanding complex figures and tables, and scientific questions. This benchmark, reflecting current realistic examination styles, provides multimodal inputs and requires responses that are either precise or open-ended, similar to real-life school tests. It challenges MLLMs with a variety of tasks, ranging from formula derivation to image detail analysis, and cross-modality reasoning. Multi includes over 18,000 questions, with a focus on science-based QA in diverse formats. We also introduce Multi-Elite, a 500-question subset for testing the extremities of MLLMs, and Multi-Extend, which enhances In-Context Learning research with more than 4
    
[^7]: 大型语言模型的人类可读指纹

    Human-Readable Fingerprint for Large Language Models

    [https://arxiv.org/abs/2312.04828](https://arxiv.org/abs/2312.04828)

    这项研究介绍了一种大型语言模型的人类可读指纹，可以唯一识别出基本模型，并且不暴露模型参数或干扰训练。通过观察和验证，研究发现模型参数的向量方向在预训练后保持稳定，成为识别基本模型的重要条件。

    

    由于大型语言模型（LLM）的资源密集型训练和配套的精心设计的许可证，保护LLM的版权变得至关重要。然而，由于可能的参数修改，确定LLM的原始基本模型是具有挑战性的。在本研究中，我们介绍了一种用于LLM的人类可读指纹，可以唯一地识别基本模型，而不暴露模型参数或干扰训练。我们首先观察到，在预训练期间模型收敛后，LLM参数的向量方向保持稳定，通过后续的训练步骤，包括持续预训练、监督微调和RLHF，几乎没有扰动，这使得它成为识别基本模型的足够条件。通过继续训练LLM并添加一个额外的项来推开模型参数的方向，验证了这种必要性，结果使得模型受损。然而，这个方向容易受到简单攻击的影响，如维度...

    Protecting the copyright of large language models (LLMs) has become crucial due to their resource-intensive training and accompanying carefully designed licenses. However, identifying the original base model of an LLM is challenging due to potential parameter alterations. In this study, we introduce a human-readable fingerprint for LLMs that uniquely identifies the base model without exposing model parameters or interfering with training. We first observe that the vector direction of LLM parameters remains stable after the model has converged during pretraining, showing negligible perturbations through subsequent training steps, including continued pretraining, supervised fine-tuning (SFT), and RLHF, which makes it a sufficient condition to identify the base model. The necessity is validated by continuing to train an LLM with an extra term to drive away the model parameters' direction and the model becomes damaged. However, this direction is vulnerable to simple attacks like dimension 
    
[^8]: 在自动语音识别中持续学习新词

    Continuously Learning New Words in Automatic Speech Recognition. (arXiv:2401.04482v1 [cs.CL])

    [http://arxiv.org/abs/2401.04482](http://arxiv.org/abs/2401.04482)

    该论文提出了一种自我监督的持续学习方法，用于解决自动语音识别中识别新词的问题。通过对讲座录音进行推理和收集包含新词的话语，然后在自适应数据集上进行持续学习，可以在新词出现频率较高时提高性能，同时保持整体性能。

    

    尽管最近取得了进展，但自动语音识别（ASR）系统仍然远未完美。典型的错误包括缩写词、命名实体和领域特定的专用词，这些词几乎没有或没有数据可用来训练。为了解决识别这些词的问题，我们提出了一种自我监督的持续学习方法。给定带有对应幻灯片的讲座录音，我们通过使用先前工作中的记忆增强型ASR模型来将模型偏向于从幻灯片中解码新词。然后，我们对讲座进行推理，将包含检测到的新词的话语收集到自适应数据集中。接着，对这个集合进行持续学习，通过调整添加到模型的每个权重矩阵的低秩矩阵权重。整个过程对多个讲座进行迭代。我们展示了通过这种方法，我们在新词出现频率较高时获得了性能的提升（超过80%的召回率），同时保持了模型的整体性能。

    Despite recent advances, Automatic Speech Recognition (ASR) systems are still far from perfect. Typical errors include acronyms, named entities and domain-specific special words for which little or no data is available. To address the problem of recognizing these words, we propose an self-supervised continual learning approach. Given the audio of a lecture talk with corresponding slides, we bias the model towards decoding new words from the slides by using a memory-enhanced ASR model from previous work. Then, we perform inference on the talk, collecting utterances that contain detected new words into an adaptation dataset. Continual learning is then performed on this set by adapting low-rank matrix weights added to each weight matrix of the model. The whole procedure is iterated for many talks. We show that with this approach, we obtain increasing performance on the new words when they occur more frequently (more than 80% recall) while preserving the general performance of the model.
    
[^9]: 发挥指导调整的大语言模型在端到端语音识别中的零-shot能力

    Harnessing the Zero-Shot Power of Instruction-Tuned Large Language Model in End-to-End Speech Recognition. (arXiv:2309.10524v1 [eess.AS])

    [http://arxiv.org/abs/2309.10524](http://arxiv.org/abs/2309.10524)

    本论文结合指导调整的大语言模型（LLM）和端到端自动语音识别（ASR），利用LLM的零-shot能力来改善语音识别性能。

    

    我们提出了一种将指导调整的大语言模型和端到端自动语音识别相结合的新方法。现代大语言模型在零-shot学习中可以执行各种语言任务，只要提供明确的指导或提示来指导文本生成过程。我们探索使用这种零-shot能力的大语言模型来提取语言信息，以改善语音识别性能。具体来说，我们将大语言模型引导去纠正语音识别假设中的语法错误，并利用嵌入的语言知识进行端到端语音识别。所提出的模型基于混合连接主义时间分类和注意力架构，其中指导调整的大语言模型（即Llama2）被用作解码器的前端。通过CTC解码从编码器获得一个需要纠正的语音识别假设，然后将其与指导一起输入大语言模型。解码器随后采取...

    We present a novel integration of an instruction-tuned large language model (LLM) and end-to-end automatic speech recognition (ASR). Modern LLMs can perform a wide range of linguistic tasks within zero-shot learning when provided with a precise instruction or a prompt to guide the text generation process towards the desired task. We explore using this zero-shot capability of LLMs to extract linguistic information that can contribute to improving ASR performance. Specifically, we direct an LLM to correct grammatical errors in an ASR hypothesis and harness the embedded linguistic knowledge to conduct end-to-end ASR. The proposed model is built on the hybrid connectionist temporal classification (CTC) and attention architecture, where an instruction-tuned LLM (i.e., Llama2) is employed as a front-end of the decoder. An ASR hypothesis, subject to correction, is obtained from the encoder via CTC decoding, which is then fed into the LLM along with an instruction. The decoder subsequently tak
    

