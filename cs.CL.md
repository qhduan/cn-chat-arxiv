# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Transfer Attack to Image Watermarks](https://arxiv.org/abs/2403.15365) | 水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。 |
| [^2] | [StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis.](http://arxiv.org/abs/2312.10741) | StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。 |
| [^3] | [Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models.](http://arxiv.org/abs/2304.07619) | 本研究探究了使用ChatGPT及其他大型语言模型预测股市回报的潜力，发现ChatGPT的预测表现优于传统情感分析方法，而基础模型无法准确预测股票价格变化，表明复杂模型可预测能力的崛起。这表明在投资决策过程中引入先进的语言模型可以提高预测准确性并增强定量交易策略的表现。 |

# 详细

[^1]: 一种针对图像水印的转移攻击

    A Transfer Attack to Image Watermarks

    [https://arxiv.org/abs/2403.15365](https://arxiv.org/abs/2403.15365)

    水印领域的研究表明，即使在攻击者无法访问水印模型或检测API的情况下，水印基础的AI生成图像检测器也无法抵抗对抗攻击。

    

    水印已被广泛应用于工业领域，用于检测由人工智能生成的图像。文献中对这种基于水印的检测器在白盒和黑盒环境下对抗攻击的稳健性有很好的理解。然而，在无盒环境下的稳健性却知之甚少。具体来说，多项研究声称图像水印在这种环境下是稳健的。在这项工作中，我们提出了一种新的转移对抗攻击来针对无盒环境下的图像水印。我们的转移攻击向带水印的图像添加微扰，以躲避被攻击者训练的多个替代水印模型，并且经过扰动的带水印图像也能躲避目标水印模型。我们的主要贡献是理论上和经验上展示了，基于水印的人工智能生成图像检测器即使攻击者没有访问水印模型或检测API，也不具有对抗攻击的稳健性。

    arXiv:2403.15365v1 Announce Type: cross  Abstract: Watermark has been widely deployed by industry to detect AI-generated images. The robustness of such watermark-based detector against evasion attacks in the white-box and black-box settings is well understood in the literature. However, the robustness in the no-box setting is much less understood. In particular, multiple studies claimed that image watermark is robust in such setting. In this work, we propose a new transfer evasion attack to image watermark in the no-box setting. Our transfer attack adds a perturbation to a watermarked image to evade multiple surrogate watermarking models trained by the attacker itself, and the perturbed watermarked image also evades the target watermarking model. Our major contribution is to show that, both theoretically and empirically, watermark-based AI-generated image detector is not robust to evasion attacks even if the attacker does not have access to the watermarking model nor the detection API.
    
[^2]: StyleSinger: 针对领域外演唱声音合成的风格转移

    StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis. (arXiv:2312.10741v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2312.10741](http://arxiv.org/abs/2312.10741)

    StyleSinger是针对领域外演唱声音合成的风格转移模型，通过残差风格适配器（RSA）捕捉多样的风格特征实现高质量的合成演唱声音。

    

    针对领域外演唱声音合成（SVS）的风格转移专注于生成高质量的演唱声音，该声音具有从参考演唱声音样本中衍生的未见风格（如音色、情感、发音和发音技巧）。然而，模拟演唱声音风格的精细差异是一项艰巨的任务，因为演唱声音具有非常高的表现力。此外，现有的SVS方法在领域外场景中合成的演唱声音质量下降，因为它们基于训练阶段可辨别出目标声音属性的假设。为了克服这些挑战，我们提出了StyleSinger，这是第一个用于领域外参考演唱声音样本的零样式转移的演唱声音合成模型。StyleSinger采用了两种关键方法以提高效果：1）残差风格适配器（RSA），它使用残差量化模块来捕捉多样的风格特征。

    Style transfer for out-of-domain (OOD) singing voice synthesis (SVS) focuses on generating high-quality singing voices with unseen styles (such as timbre, emotion, pronunciation, and articulation skills) derived from reference singing voice samples. However, the endeavor to model the intricate nuances of singing voice styles is an arduous task, as singing voices possess a remarkable degree of expressiveness. Moreover, existing SVS methods encounter a decline in the quality of synthesized singing voices in OOD scenarios, as they rest upon the assumption that the target vocal attributes are discernible during the training phase. To overcome these challenges, we propose StyleSinger, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference singing voice samples. StyleSinger incorporates two critical approaches for enhanced effectiveness: 1) the Residual Style Adaptor (RSA) which employs a residual quantization module to capture diverse style character
    
[^3]: ChatGPT是否能够预测股票价格波动？回报可预测性与大语言模型。

    Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models. (arXiv:2304.07619v1 [q-fin.ST])

    [http://arxiv.org/abs/2304.07619](http://arxiv.org/abs/2304.07619)

    本研究探究了使用ChatGPT及其他大型语言模型预测股市回报的潜力，发现ChatGPT的预测表现优于传统情感分析方法，而基础模型无法准确预测股票价格变化，表明复杂模型可预测能力的崛起。这表明在投资决策过程中引入先进的语言模型可以提高预测准确性并增强定量交易策略的表现。

    

    本文研究了使用情感分析预测股市回报的潜力，探讨了使用ChatGPT以及其他大语言模型在预测股市回报方面的表现。我们使用ChatGPT判断新闻标题对公司股票价格是好消息、坏消息或无关消息。通过计算数字分数，我们发现这些"ChatGPT分数"和随后的日常股票市场回报之间存在正相关性。而且，ChatGPT的表现优于传统的情感分析方法。同时，我们发现GPT-1、GPT-2和BERT等基础模型无法准确预测回报，这表明回报可预测性是复杂模型的一种新兴能力。我们的研究结果表明，将先进的语言模型纳入投资决策过程可以产生更准确的预测，并提高定量交易策略的表现。

    We examine the potential of ChatGPT, and other large language models, in predicting stock market returns using sentiment analysis of news headlines. We use ChatGPT to indicate whether a given headline is good, bad, or irrelevant news for firms' stock prices. We then compute a numerical score and document a positive correlation between these ``ChatGPT scores'' and subsequent daily stock market returns. Further, ChatGPT outperforms traditional sentiment analysis methods. We find that more basic models such as GPT-1, GPT-2, and BERT cannot accurately forecast returns, indicating return predictability is an emerging capacity of complex models. Our results suggest that incorporating advanced language models into the investment decision-making process can yield more accurate predictions and enhance the performance of quantitative trading strategies.
    

