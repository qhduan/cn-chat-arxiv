# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT).](http://arxiv.org/abs/2307.01225) | 通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。 |
| [^2] | [EasyNER: A Customizable Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text.](http://arxiv.org/abs/2304.07805) | EasyNER是一种用于在医学研究文章中识别命名实体的端到端工具。它基于深度学习模型和字典方法，并且易于使用和定制。在COVID-19相关文章数据集上的应用证明了其可以准确地识别所需实体。 |

# 详细

[^1]: 解释性和透明性驱动的文本对抗示例的检测与转换（IT-DT）

    Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT). (arXiv:2307.01225v1 [cs.CL])

    [http://arxiv.org/abs/2307.01225](http://arxiv.org/abs/2307.01225)

    通过提出的解释性和透明性驱动的检测与转换（IT-DT）框架，我们在检测和转换文本对抗示例方面注重解释性和透明性。这个框架利用了注意力图、集成梯度和模型反馈等技术，在检测阶段有助于识别对对抗性分类有贡献的显著特征和扰动词语，并在转换阶段使用预训练的嵌入和模型反馈来生成扰动词语的最佳替代，以将对抗性示例转换为正常示例。

    

    基于Transformer的文本分类器如BERT、Roberta、T5和GPT-3在自然语言处理方面展示了令人印象深刻的性能。然而，它们对于对抗性示例的脆弱性提出了安全风险。现有的防御方法缺乏解释性，很难理解对抗性分类并识别模型的漏洞。为了解决这个问题，我们提出了解释性和透明性驱动的检测与转换（IT-DT）框架。它专注于在检测和转换文本对抗示例时的解释性和透明性。IT-DT利用注意力图、集成梯度和模型反馈等技术进行解释性检测。这有助于识别对对抗性分类有贡献的显著特征和扰动词语。在转换阶段，IT-DT利用预训练的嵌入和模型反馈来生成扰动词语的最佳替代。通过找到合适的替换，我们的目标是将对抗性示例转换为正常示例。

    Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into
    
[^2]: EasyNER：一种可定制的易于使用的医学文本深度学习和基于字典的命名实体识别工具

    EasyNER: A Customizable Easy-to-Use Pipeline for Deep Learning- and Dictionary-based Named Entity Recognition from Medical Text. (arXiv:2304.07805v1 [q-bio.QM])

    [http://arxiv.org/abs/2304.07805](http://arxiv.org/abs/2304.07805)

    EasyNER是一种用于在医学研究文章中识别命名实体的端到端工具。它基于深度学习模型和字典方法，并且易于使用和定制。在COVID-19相关文章数据集上的应用证明了其可以准确地识别所需实体。

    

    医学研究已经产生了大量出版物，PubMed数据库已经收录了超过3,500万篇研究文章。整合这些分散在大量文献中的知识可以提供有关生理机制和导致新型医学干预的疾病过程的关键见解。然而，对于研究人员来说，利用这些信息成为一个巨大挑战，因为数据的规模和复杂性远远超出了人类的处理能力。在COVID-19大流行的紧急情况下，这尤其成为问题。自动化文本挖掘可以帮助从大量医学研究文章中提取和连接信息。文本挖掘的第一步通常是识别特定类别的关键字（例如所有蛋白质或疾病名称），即命名实体识别（NER）。本文提出了一种端到端的NER工具EasyNER，用于识别医学研究文章中的典型实体，包括疾病名称、药物名称和蛋白质名称。EasyNER基于深度学习模型和基于字典的方法，旨在对自然语言处理具有不同经验水平的研究人员易于使用和定制。我们将EasyNER应用于COVID-19相关文章的数据集中并展示它可以准确地识别感兴趣的实体，为下游分析提供有用的信息。

    Medical research generates a large number of publications with the PubMed database already containing >35 million research articles. Integration of the knowledge scattered across this large body of literature could provide key insights into physiological mechanisms and disease processes leading to novel medical interventions. However, it is a great challenge for researchers to utilize this information in full since the scale and complexity of the data greatly surpasses human processing abilities. This becomes especially problematic in cases of extreme urgency like the COVID-19 pandemic. Automated text mining can help extract and connect information from the large body of medical research articles. The first step in text mining is typically the identification of specific classes of keywords (e.g., all protein or disease names), so called Named Entity Recognition (NER). Here we present an end-to-end pipeline for NER of typical entities found in medical research articles, including diseas
    

