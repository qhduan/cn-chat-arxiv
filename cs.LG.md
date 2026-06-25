# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Narrative Feature or Structured Feature? A Study of Large Language Models to Identify Cancer Patients at Risk of Heart Failure](https://arxiv.org/abs/2403.11425) | 使用大型语言模型结合新颖的叙述特征，能够有效识别癌症患者患心力衰竭的风险，表现优于传统机器学习模型和深度学习模型。 |
| [^2] | [Lattice Approximations in Wasserstein Space.](http://arxiv.org/abs/2310.09149) | 本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。 |

# 详细

[^1]: 叙事特征还是结构特征？研究大型语言模型以识别患心力衰竭风险的癌症患者

    Narrative Feature or Structured Feature? A Study of Large Language Models to Identify Cancer Patients at Risk of Heart Failure

    [https://arxiv.org/abs/2403.11425](https://arxiv.org/abs/2403.11425)

    使用大型语言模型结合新颖的叙述特征，能够有效识别癌症患者患心力衰竭的风险，表现优于传统机器学习模型和深度学习模型。

    

    癌症治疗已知会引入心毒性，对预后和生存率产生负面影响。识别患心力衰竭（HF）风险的癌症患者对于改善癌症治疗结果和安全性至关重要。本研究使用来自电子健康记录（EHRs）的机器学习（ML）模型，包括传统ML、时间感知长短期记忆（T-LSTM）和使用从结构化医学代码衍生的新颖叙述特征的大型语言模型（LLMs）来识别患HF风险的癌症患者。我们从佛罗里达大学健康中心识别了一组包括12,806名肺癌、乳腺癌和结直肠癌患者的癌症队列，其中1,602人在癌症后发展为HF。LLM GatorTron-3.9B取得了最佳的F1分数，比传统支持向量机高出39%，比T-LSTM深度学习模型高出7%，比广泛使用的Transformer模型BERT高出5.6%。

    arXiv:2403.11425v1 Announce Type: cross  Abstract: Cancer treatments are known to introduce cardiotoxicity, negatively impacting outcomes and survivorship. Identifying cancer patients at risk of heart failure (HF) is critical to improving cancer treatment outcomes and safety. This study examined machine learning (ML) models to identify cancer patients at risk of HF using electronic health records (EHRs), including traditional ML, Time-Aware long short-term memory (T-LSTM), and large language models (LLMs) using novel narrative features derived from the structured medical codes. We identified a cancer cohort of 12,806 patients from the University of Florida Health, diagnosed with lung, breast, and colorectal cancers, among which 1,602 individuals developed HF after cancer. The LLM, GatorTron-3.9B, achieved the best F1 scores, outperforming the traditional support vector machines by 39%, the T-LSTM deep learning model by 7%, and a widely used transformer model, BERT, by 5.6%. The analysi
    
[^2]: 微分水平空间中的格点逼近

    Lattice Approximations in Wasserstein Space. (arXiv:2310.09149v1 [stat.ML])

    [http://arxiv.org/abs/2310.09149](http://arxiv.org/abs/2310.09149)

    本论文研究了在Wasserstein空间中通过离散和分段常数测度进行的结构逼近方法。结果表明，对于满秩的格点按比例缩放后得到的Voronoi分割逼近的测度误差是$O(h)$，逼近的$N$项误差为$O(N^{-\frac1d})$，并且可以推广到非紧支撑测度。

    

    我们考虑在Wasserstein空间$W_p(\mathbb{R}^d)$中通过离散和分段常数测度来对测度进行结构逼近。我们证明，如果一个满秩的格点$\Lambda$按照$h\in(0,1]$的比例进行缩放，那么基于$h\Lambda$的Voronoi分割得到的测度逼近是$O(h)$，不论$d$或$p$的取值。之后，我们使用覆盖论证证明，对于紧支撑的测度的$N$项逼近是$O(N^{-\frac1d})$，这与最优量化器和经验测度逼近在大多数情况下已知的速率相匹配。最后，我们将这些结果推广到非紧支撑测度，要求其具有足够的衰减性质。

    We consider structured approximation of measures in Wasserstein space $W_p(\mathbb{R}^d)$ for $p\in[1,\infty)$ by discrete and piecewise constant measures based on a scaled Voronoi partition of $\mathbb{R}^d$. We show that if a full rank lattice $\Lambda$ is scaled by a factor of $h\in(0,1]$, then approximation of a measure based on the Voronoi partition of $h\Lambda$ is $O(h)$ regardless of $d$ or $p$. We then use a covering argument to show that $N$-term approximations of compactly supported measures is $O(N^{-\frac1d})$ which matches known rates for optimal quantizers and empirical measure approximation in most instances. Finally, we extend these results to noncompactly supported measures with sufficient decay.
    

