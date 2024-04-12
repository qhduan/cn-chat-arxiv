# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multi-granular Adversarial Attacks against Black-box Neural Ranking Models](https://arxiv.org/abs/2404.01574) | 这项研究聚焦于利用多粒度扰动生成高质量的对抗性示例，通过转化为顺序决策过程来解决组合爆炸问题。 |

# 详细

[^1]: 目标黑盒神经排序模型的多粒度对抗攻击

    Multi-granular Adversarial Attacks against Black-box Neural Ranking Models

    [https://arxiv.org/abs/2404.01574](https://arxiv.org/abs/2404.01574)

    这项研究聚焦于利用多粒度扰动生成高质量的对抗性示例，通过转化为顺序决策过程来解决组合爆炸问题。

    

    对抗排序攻击由于在发现神经排序模型的脆弱性并增强其鲁棒性方面取得成功而受到越来越多的关注。传统的攻击方法仅在单一粒度上进行扰动，例如单词级或句子级，对目标文档进行攻击。然而，将扰动限制在单一粒度上可能会减少创造对抗性示例的灵活性，从而降低攻击的潜在威胁。因此，我们专注于通过结合不同粒度的扰动生成高质量的对抗性示例。实现这一目标涉及解决组合爆炸问题，需要识别出跨所有可能的粒度、位置和文本片段的最佳组合扰动。为了解决这一挑战，我们将多粒度对抗攻击转化为一个顺序决策过程，其中

    arXiv:2404.01574v1 Announce Type: cross  Abstract: Adversarial ranking attacks have gained increasing attention due to their success in probing vulnerabilities, and, hence, enhancing the robustness, of neural ranking models. Conventional attack methods employ perturbations at a single granularity, e.g., word-level or sentence-level, to a target document. However, limiting perturbations to a single level of granularity may reduce the flexibility of creating adversarial examples, thereby diminishing the potential threat of the attack. Therefore, we focus on generating high-quality adversarial examples by incorporating multi-granular perturbations. Achieving this objective involves tackling a combinatorial explosion problem, which requires identifying an optimal combination of perturbations across all possible levels of granularity, positions, and textual pieces. To address this challenge, we transform the multi-granular adversarial attack into a sequential decision-making process, where 
    

