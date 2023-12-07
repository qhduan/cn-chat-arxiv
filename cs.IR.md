# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Co-evolving Vector Quantization for ID-based Recommendation.](http://arxiv.org/abs/2308.16761) | 这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。 |
| [^2] | [Machine Reading Comprehension using Case-based Reasoning.](http://arxiv.org/abs/2305.14815) | 本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。 |

# 详细

[^1]: 基于ID的推荐的共同演化向量量化

    Co-evolving Vector Quantization for ID-based Recommendation. (arXiv:2308.16761v1 [cs.IR])

    [http://arxiv.org/abs/2308.16761](http://arxiv.org/abs/2308.16761)

    这项工作提出了一种用于基于ID的推荐的共同演化向量量化框架（COVE），该框架能够自动学习和生成不同粒度级别下的实体分类信息，并在各种推荐任务中展现了有效性。

    

    类别信息对于提高推荐的质量和个性化起着至关重要的作用。然而，在基于ID的推荐中，项目类别信息的可用性并不一致。在这项工作中，我们提出了一种替代方法，以自动学习和生成实体（即用户和项目）在不同粒度级别上的分类信息，特别适用于基于ID的推荐。具体而言，我们设计了一个共同演化向量量化框架，即COVE，它能够同时学习和改进代码表示和实体嵌入，并以从随机初始化状态开始的端到端方式进行。通过其高度适应性，COVE可以轻松集成到现有的推荐模型中。我们验证了COVE在各种推荐任务中的有效性，包括列表完成、协同过滤和点击率预测，涵盖不同的推荐场景。

    Category information plays a crucial role in enhancing the quality and personalization of recommendations. Nevertheless, the availability of item category information is not consistently present, particularly in the context of ID-based recommendations. In this work, we propose an alternative approach to automatically learn and generate entity (i.e., user and item) categorical information at different levels of granularity, specifically for ID-based recommendation. Specifically, we devise a co-evolving vector quantization framework, namely COVE, which enables the simultaneous learning and refinement of code representation and entity embedding in an end-to-end manner, starting from the randomly initialized states. With its high adaptability, COVE can be easily integrated into existing recommendation models. We validate the effectiveness of COVE on various recommendation tasks including list completion, collaborative filtering, and click-through rate prediction, across different recommend
    
[^2]: 使用基于案例推理的机器阅读理解

    Machine Reading Comprehension using Case-based Reasoning. (arXiv:2305.14815v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14815](http://arxiv.org/abs/2305.14815)

    本文提出了一种基于案例推理的机器阅读理解方法（CBR-MRC），通过从存储器中检索相似案例并选择最类似的上下文来预测答案，以达到高准确性。在自然语言问题和新闻问答中，CBR-MRC的准确性超过基准，并且能够识别与其他评估员不同的答案。

    

    我们提出了一种准确且可解释的方法，用于机器阅读理解中的答案提取，该方法类似于经典人工智能中的基于案例推理（CBR）。我们的方法（CBR-MRC）基于一个假设，即相似问题的上下文化答案彼此之间具有语义相似性。给定一个测试问题，CBR-MRC首先从非参数化存储器中检索一组相似的案例，然后通过选择测试上下文中最类似于检索到的案例中上下文化答案表示的范围来预测答案。我们的方法半参数化的特性使其能够将预测归因于特定的证据案例集，因此在构建可靠且可调试的问答系统时是一个理想的选择。我们展示了CBR-MRC在自然语言问题（NaturalQuestions）和新闻问答（NewsQA）上比大型读者模型提供了高准确性，并且优于基准分别提升了11.5和8.4 EM。此外，我们还展示了CBR-MRC在识别与他人评估员不同的答案方面的能力。

    We present an accurate and interpretable method for answer extraction in machine reading comprehension that is reminiscent of case-based reasoning (CBR) from classical AI. Our method (CBR-MRC) builds upon the hypothesis that contextualized answers to similar questions share semantic similarities with each other. Given a test question, CBR-MRC first retrieves a set of similar cases from a non-parametric memory and then predicts an answer by selecting the span in the test context that is most similar to the contextualized representations of answers in the retrieved cases. The semi-parametric nature of our approach allows it to attribute a prediction to the specific set of evidence cases, making it a desirable choice for building reliable and debuggable QA systems. We show that CBR-MRC provides high accuracy comparable with large reader models and outperforms baselines by 11.5 and 8.4 EM on NaturalQuestions and NewsQA, respectively. Further, we demonstrate the ability of CBR-MRC in identi
    

