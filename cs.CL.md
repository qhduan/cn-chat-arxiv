# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Round Trip Translation Defence against Large Language Model Jailbreaking Attacks](https://arxiv.org/abs/2402.13517) | 往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。 |
| [^2] | [LLMs and Finetuning: Benchmarking cross-domain performance for hate speech detection](https://arxiv.org/abs/2310.18964) | 本研究调查了预训练和微调的Large Language Models在识别仇恨言论方面的有效性和适应性，揭示了即使没有预训练，LLMs在性能上仍然具有极大优势。 |

# 详细

[^1]: 大型语言模型逆向翻译防御对抗攻击

    Round Trip Translation Defence against Large Language Model Jailbreaking Attacks

    [https://arxiv.org/abs/2402.13517](https://arxiv.org/abs/2402.13517)

    往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。

    

    大型语言模型（LLMs）容易受到社交工程攻击，这些攻击对人类具有可解释性，但需要LLMs具有高水平的理解能力才能抵抗。现有的防御措施最多只能缓解这些攻击的不到一半。为解决这一问题，我们提出了往返翻译（RTT）方法，这是第一个专门设计用于抵御LLMs社交工程攻击的算法。RTT会改写对抗性提示并推广表达的思想，使LLMs更容易检测出诱发有害行为。这种方法灵活、轻量且可转移至不同的LLMs。我们的防御成功地缓解了超过70%的Prompt Automatic Iterative Refinement (PAIR)攻击，这是目前我们所知最有效的防御。我们也是首次尝试缓解MathsAttack，并将其攻击成功率降低了近40%。我们的代码已公开发布。

    arXiv:2402.13517v1 Announce Type: cross  Abstract: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly av
    
[^2]: LLMs与Fine-tuning: 对仇恨言论检测跨领域性能的基准测试

    LLMs and Finetuning: Benchmarking cross-domain performance for hate speech detection

    [https://arxiv.org/abs/2310.18964](https://arxiv.org/abs/2310.18964)

    本研究调查了预训练和微调的Large Language Models在识别仇恨言论方面的有效性和适应性，揭示了即使没有预训练，LLMs在性能上仍然具有极大优势。

    

    在在线交流不断发展的环境中，仇恨言论检测仍然是一个严峻的挑战，数字平台的多样性进一步加剧了这一挑战。本研究调查了预训练和微调的大型语言模型（LLMs）在识别仇恨言论中的有效性和适应性，以解决两个核心问题：（1）模型性能在多大程度上依赖于微调和训练参数？（2）模型在跨领域仇恨言论检测中的泛化程度如何？以及（3）影响泛化潜力的数据集或模型的具体特征是什么？实验证明，即使没有预训练，LLMs也比最先进的模型具有巨大优势。为了回答问题（1），我们分析了36个领域内分类器，涵盖了LLaMA、Vicuna及其不同的预训练和微调状态，跨越了九个公开可用数据集，涵盖了各种平台。

    arXiv:2310.18964v2 Announce Type: replace  Abstract: In the evolving landscape of online communication, hate speech detection remains a formidable challenge, further compounded by the diversity of digital platforms. This study investigates the effectiveness and adaptability of pre-trained and fine-tuned Large Language Models (LLMs) in identifying hate speech, to address two central questions: (1) To what extent does the model performance depend on the fine-tuning and training parameters?, (2) To what extent do models generalize to cross-domain hate speech detection? and (3) What are the specific features of the datasets or models that influence the generalization potential? The experiment shows that LLMs offer a huge advantage over the state-of-the-art even without pretraining. To answer (1) we analyze 36 in-domain classifiers comprising LLaMA, Vicuna, and their variations in pre-trained and fine-tuned states across nine publicly available datasets that span a wide range of platforms a
    

