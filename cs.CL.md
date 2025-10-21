# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference.](http://arxiv.org/abs/2307.05034) | 该论文介绍了一个名为SICCK的合成数据集以及一种新颖的分析方法，用于评估自然语言推理中复杂组合知识的性能。研究发现，在零-shot和微调情况下，神经网络推理模型能够很好地捕捉结构和语义组合的变化。 |

# 详细

[^1]: 用于评估自然语言推理中复杂组合知识的合成数据集

    Synthetic Dataset for Evaluating Complex Compositional Knowledge for Natural Language Inference. (arXiv:2307.05034v1 [cs.CL])

    [http://arxiv.org/abs/2307.05034](http://arxiv.org/abs/2307.05034)

    该论文介绍了一个名为SICCK的合成数据集以及一种新颖的分析方法，用于评估自然语言推理中复杂组合知识的性能。研究发现，在零-shot和微调情况下，神经网络推理模型能够很好地捕捉结构和语义组合的变化。

    

    我们介绍了一个名为Sentences Involving Complex Compositional Knowledge (SICCK)的合成数据集，以及一种新颖的分析方法，用于研究自然语言推理模型对逻辑组成性的性能。我们通过修改SICK数据集中的15个示例，生成了1,304个句子对。为此，我们使用一组短语 - 与自然逻辑中的普遍量词、存在量词、否定和其他概念修饰符相对应的修饰符 - 修改了原始文本。我们使用这些短语修改前提和假设的主语、谓语和宾语部分。最后，我们根据自然逻辑规则为这些修改后的文本标注相应的包含关系标签。我们对神经网络推理模型在零-shot和微调情况下对结构和语义组合变化的捕捉能力进行了初步验证。我们发现在这些情况下，NLI模型的性能表现良好。

    We introduce a synthetic dataset called Sentences Involving Complex Compositional Knowledge (SICCK) and a novel analysis that investigates the performance of Natural Language Inference (NLI) models to understand compositionality in logic. We produce 1,304 sentence pairs by modifying 15 examples from the SICK dataset (Marelli et al., 2014). To this end, we modify the original texts using a set of phrases - modifiers that correspond to universal quantifiers, existential quantifiers, negation, and other concept modifiers in Natural Logic (NL) (MacCartney, 2009). We use these phrases to modify the subject, verb, and object parts of the premise and hypothesis. Lastly, we annotate these modified texts with the corresponding entailment labels following NL rules. We conduct a preliminary verification of how well the change in the structural and semantic composition is captured by neural NLI models, in both zero-shot and fine-tuned scenarios. We found that the performance of NLI models under th
    

