# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Evaluating the Generation Capabilities of Large Chinese Language Models.](http://arxiv.org/abs/2308.04823) | 本文首次对大型中文语言模型在多个学科领域的生成能力进行了全面评估，并提出了Gscore作为衡量生成结果质量的综合指数。 |
| [^2] | [Measuring Massive Multitask Chinese Understanding.](http://arxiv.org/abs/2304.12986) | 本研究提出了一项测试，以衡量大型中文语言模型的多任务准确性，测试涵盖医学、法律、心理学和教育四个主要领域，结果表明所有模型在法律领域中表现都很差，建议研究人员应该开发更加多样化和均衡的多任务中文理解模型。 |

# 详细

[^1]: 评估大型中文语言模型的生成能力

    Evaluating the Generation Capabilities of Large Chinese Language Models. (arXiv:2308.04823v1 [cs.CL])

    [http://arxiv.org/abs/2308.04823](http://arxiv.org/abs/2308.04823)

    本文首次对大型中文语言模型在多个学科领域的生成能力进行了全面评估，并提出了Gscore作为衡量生成结果质量的综合指数。

    

    本文介绍了CG-Eval，这是第一个对大型中文语言模型在多个学科领域生成能力进行全面评估的研究。通过在科学工程、人文社科、数学计算、医师资格考试、司法考试和注册会计师考试六个学科中生成准确和相关的回答，评估了这些模型的性能。本文还提出了Gscore，这是一个由多个度量指标加权求和得到的综合指数，用于衡量模型生成结果与参考答案的质量。测试数据和测试结果可在此http URL找到。

    This paper presents CG-Eval, the first comprehensive evaluation of the generation capabilities of large Chinese language models across a wide range of academic disciplines. The models' performance was assessed based on their ability to generate accurate and relevant responses to different types of questions in six disciplines, namely, Science and Engineering, Humanities and Social Sciences, Mathematical Calculations, Medical Practitioner Qualification Examination, Judicial Examination, and Certified Public Accountant Examination. This paper also presents Gscore, a composite index derived from the weighted sum of multiple metrics to measure the quality of model's generation against a reference. The test data and test results can be found at this http URL
    
[^2]: 测量大规模多任务中文理解能力

    Measuring Massive Multitask Chinese Understanding. (arXiv:2304.12986v1 [cs.CL])

    [http://arxiv.org/abs/2304.12986](http://arxiv.org/abs/2304.12986)

    本研究提出了一项测试，以衡量大型中文语言模型的多任务准确性，测试涵盖医学、法律、心理学和教育四个主要领域，结果表明所有模型在法律领域中表现都很差，建议研究人员应该开发更加多样化和均衡的多任务中文理解模型。

    

    大规模中文语言模型的研发正蓬勃发展，但缺乏相应的能力评估。因此，我们提出了一个测试，以衡量大型中文语言模型的多任务准确性。该测试涵盖了医学、法律、心理学和教育四个主要领域，在医学领域有15个子任务，在教育领域有8个子任务。我们发现，在零样本设置下表现最佳的模型平均比表现最差的模型高出近22个百分点。在四个主要领域中，所有模型的平均零样本准确度均未超过0.5。在子领域中，只有GPT-3.5-turbo模型在临床医学中实现了0.703的零样本准确度，这是所有模型在所有子任务中最高的准确度。所有模型在法律领域中表现都很差，最高的零样本准确度仅达到0.259。通过全面评估多个学科的广度和深度的知识，我们建议研究人员应该开发更加多样化和均衡的多任务中文理解模型。

    The development of large-scale Chinese language models is flourishing, yet there is a lack of corresponding capability assessments. Therefore, we propose a test to measure the multitask accuracy of large Chinese language models. This test encompasses four major domains, including medicine, law, psychology, and education, with 15 subtasks in medicine and 8 subtasks in education. We found that the best-performing models in the zero-shot setting outperformed the worst-performing models by nearly 22 percentage points on average. Across the four major domains, the average zero-shot accuracy of all models did not exceed 0.5. In the subdomains, only the GPT-3.5-turbo model achieved a zero-shot accuracy of 0.703 in clinical medicine, which was the highest accuracy among all models across all subtasks. All models performed poorly in the legal domain, with the highest zero-shot accuracy reaching only 0.259. By comprehensively evaluating the breadth and depth of knowledge across multiple discipli
    

