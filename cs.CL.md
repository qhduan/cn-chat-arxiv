# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rethinking Human-like Translation Strategy: Integrating Drift-Diffusion Model with Large Language Models for Machine Translation](https://arxiv.org/abs/2402.10699) | 将Thinker与漂移扩散模型集成，重新定义漂移扩散过程以模拟人类翻译者的决策制定，实验证明在机器翻译中取得了优异成绩。 |
| [^2] | [You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting.](http://arxiv.org/abs/2306.15933) | 本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。 |

# 详细

[^1]: 重新思考类人翻译策略：将漂移扩散模型与大型语言模型集成用于机器翻译

    Rethinking Human-like Translation Strategy: Integrating Drift-Diffusion Model with Large Language Models for Machine Translation

    [https://arxiv.org/abs/2402.10699](https://arxiv.org/abs/2402.10699)

    将Thinker与漂移扩散模型集成，重新定义漂移扩散过程以模拟人类翻译者的决策制定，实验证明在机器翻译中取得了优异成绩。

    

    大型语言模型（LLMs）在包括机器翻译在内的各种下游任务中展现出了巨大潜力。然而，基于LLM的机器翻译先前的工作主要集中在更好地利用训练数据、演示版本或预定义的普遍知识来提高性能，缺乏对类似人类翻译者的决策制定的考虑。本文将“Thinker”与漂移扩散模型（Thinker-DDM）相结合，以解决这一问题。然后，我们重新定义了漂移扩散过程，以模拟受限资源情况下类人翻译者的动态决策制定。我们在高资源、低资源和常识翻译设置下，使用WMT22和CommonMT数据集进行了大量实验，在前两种场景中，Thinker-DDM的表现优于基准。我们还对常识翻译进行了额外的分析和评估，以说明其高效性。

    arXiv:2402.10699v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated promising potential in various downstream tasks, including machine translation. However, prior work on LLM-based machine translation has mainly focused on better utilizing training data, demonstrations, or pre-defined and universal knowledge to improve performance, with a lack of consideration of decision-making like human translators. In this paper, we incorporate Thinker with the Drift-Diffusion Model (Thinker-DDM) to address this issue. We then redefine the Drift-Diffusion process to emulate human translators' dynamic decision-making under constrained resources. We conduct extensive experiments under the high-resource, low-resource, and commonsense translation settings using the WMT22 and CommonMT datasets, in which Thinker-DDM outperforms baselines in the first two scenarios. We also perform additional analysis and evaluation on commonsense translation to illustrate the high effectivenes
    
[^2]: 通过验证和纠正提示进行数据生成文本生成

    You Can Generate It Again: Data-to-text Generation with Verification and Correction Prompting. (arXiv:2306.15933v1 [cs.CL])

    [http://arxiv.org/abs/2306.15933](http://arxiv.org/abs/2306.15933)

    本文提出了一种多步骤生成、验证和纠正的数据生成文本方法，通过专门的错误指示提示来改善输出质量。

    

    尽管现有模型取得了显著进展，从结构化数据输入生成文本描述（称为数据生成文本）仍然是一个具有挑战性的任务。在本文中，我们提出了一种新的方法，通过引入包括生成、验证和纠正阶段的多步骤过程，超越了传统的一次性生成方法。我们的方法，VCP（验证和纠正提示），从模型生成初始输出开始。然后，我们继续验证所生成文本的不同方面的正确性。验证步骤的观察结果被转化为专门的错误指示提示，该提示指示模型在重新生成输出时考虑已识别的错误。为了增强模型的纠正能力，我们开发了一个经过精心设计的培训过程。该过程使模型能够融入错误指示提示的反馈，从而改善输出生成。

    Despite significant advancements in existing models, generating text descriptions from structured data input, known as data-to-text generation, remains a challenging task. In this paper, we propose a novel approach that goes beyond traditional one-shot generation methods by introducing a multi-step process consisting of generation, verification, and correction stages. Our approach, VCP(Verification and Correction Prompting), begins with the model generating an initial output. We then proceed to verify the correctness of different aspects of the generated text. The observations from the verification step are converted into a specialized error-indication prompt, which instructs the model to regenerate the output while considering the identified errors. To enhance the model's correction ability, we have developed a carefully designed training procedure. This procedure enables the model to incorporate feedback from the error-indication prompt, resulting in improved output generation. Throu
    

