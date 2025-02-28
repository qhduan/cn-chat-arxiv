# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Factual Consistency Evaluation of Summarisation in the Era of Large Language Models](https://arxiv.org/abs/2402.13758) | 在摘要一致性评估方面，该研究通过引入临床文本摘要的数据集TreatFact并对11个大语言模型进行评估，填补了关于大语言模型在摘要事实一致性评估方面的缺口。 |
| [^2] | [Sheffield's Submission to the AmericasNLP Shared Task on Machine Translation into Indigenous Languages.](http://arxiv.org/abs/2306.09830) | 本文介绍了谢菲尔德大学的机器翻译方法，成功在AmericasNLP机器翻译土著语言分享任务中取得了最高的平均chrF，其中在Aymara，Guarani和Quechua方面有显着的改进。 |

# 详细

[^1]: 大语言模型时代中摘要的事实一致性评估

    Factual Consistency Evaluation of Summarisation in the Era of Large Language Models

    [https://arxiv.org/abs/2402.13758](https://arxiv.org/abs/2402.13758)

    在摘要一致性评估方面，该研究通过引入临床文本摘要的数据集TreatFact并对11个大语言模型进行评估，填补了关于大语言模型在摘要事实一致性评估方面的缺口。

    

    自动生成摘要中与源文件的事实不一致可能导致错误信息或带来风险。现有的事实一致性（FC）度量受到其性能、效率和可解释性的限制。大语言模型（LLMs）的最新进展在文本评估方面表现出卓越的潜力，但它们在评估摘要中的FC方面的效果仍未得到充分探讨。先前的研究主要集中在专有LLMs上，未探讨影响它们评估能力的重要因素。此外，当前的FC评估基准仅限于新闻文章，对在其上测试的FC方法的普遍性产生怀疑。在本文中，我们首先通过引入TreatFact数据集解决这一差距，该数据集包含由领域专家注释的临床文本的LLM生成摘要的FC。此外，我们在新闻和临床领域中为FC评估对比了11个LLMs，并分析了

    arXiv:2402.13758v1 Announce Type: new  Abstract: Factual inconsistency with source documents in automatically generated summaries can lead to misinformation or pose risks. Existing factual consistency(FC) metrics are constrained by their performance, efficiency, and explainability. Recent advances in Large language models (LLMs) have demonstrated remarkable potential in text evaluation but their effectiveness in assessing FC in summarisation remains underexplored. Prior research has mostly focused on proprietary LLMs, leaving essential factors that affect their assessment capabilities unexplored. Additionally, current FC evaluation benchmarks are restricted to news articles, casting doubt on the generality of the FC methods tested on them. In this paper, we first address the gap by introducing TreatFact a dataset of LLM-generated summaries of clinical texts, annotated for FC by domain experts. Moreover, we benchmark 11 LLMs for FC evaluation across news and clinical domains and analyse
    
[^2]: 谢菲尔德大学提交给AmericasNLP机器翻译土著语言分享任务的论文（arXiv: 2306.09830v1 [cs.CL]）

    Sheffield's Submission to the AmericasNLP Shared Task on Machine Translation into Indigenous Languages. (arXiv:2306.09830v1 [cs.CL])

    [http://arxiv.org/abs/2306.09830](http://arxiv.org/abs/2306.09830)

    本文介绍了谢菲尔德大学的机器翻译方法，成功在AmericasNLP机器翻译土著语言分享任务中取得了最高的平均chrF，其中在Aymara，Guarani和Quechua方面有显着的改进。

    

    本文描述了谢菲尔德大学提交给AmericasNLP 2023机器翻译土著语言分享任务的方法，该任务包括将西班牙语翻译成十一种土著语言。 我们的方法包括扩展，训练和与不同种类的NLLB-200组合。 我们使用组织者提供的数据以及宪法，手册，新闻文章和从单语数据生成的回译等各种其他来源的数据。 在开发集上，我们的最佳成绩在所有语言的平均chrF上比基线提高了11％，尤其是在Aymara，Guarani和Quechua方面有显着的改进。 在测试集上，我们实现了所有提交中最高的平均chrF，我们在11种语言中排名前四位，并且我们的至少一个提交在所有语言中排名前三位。

    In this paper we describe the University of Sheffield's submission to the AmericasNLP 2023 Shared Task on Machine Translation into Indigenous Languages which comprises the translation from Spanish to eleven indigenous languages. Our approach consists of extending, training, and ensembling different variations of NLLB-200. We use data provided by the organizers and data from various other sources such as constitutions, handbooks, news articles, and backtranslations generated from monolingual data. On the dev set, our best submission outperforms the baseline by 11% average chrF across all languages, with substantial improvements particularly for Aymara, Guarani and Quechua. On the test set, we achieve the highest average chrF of all the submissions, we rank first in four of the eleven languages, and at least one of our submissions ranks in the top 3 for all languages.
    

