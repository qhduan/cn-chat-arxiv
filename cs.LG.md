# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables](https://arxiv.org/abs/2403.01673) | CATS通过构建辅助时间序列作为外生变量，有效地表示和整合多元时间序列之间的关系，提高了多元时间序列预测的效果，并且相较于之前的模型大幅减少了复杂性和参数。 |
| [^2] | [ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning.](http://arxiv.org/abs/2310.09488) | 本研究提出了ARM，一种多变量的时间-上下文自适应学习方法，用于优化长期时间序列预测。ARM通过采用自适应单变量效应学习、随机丢弃训练策略和多核局部平滑，能更好地处理时间模式和学习系列之间的依赖关系。在多个基准测试中，ARM展示了卓越的性能，而计算成本相对较低。 |
| [^3] | [Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach.](http://arxiv.org/abs/2309.16858) | 我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。 |

# 详细

[^1]: CATS：通过构建辅助时间序列作为外生变量增强多元时间序列预测

    CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables

    [https://arxiv.org/abs/2403.01673](https://arxiv.org/abs/2403.01673)

    CATS通过构建辅助时间序列作为外生变量，有效地表示和整合多元时间序列之间的关系，提高了多元时间序列预测的效果，并且相较于之前的模型大幅减少了复杂性和参数。

    

    对于多元时间序列预测（MTSF），最近的深度学习应用显示，单变量模型经常优于多元模型。为了解决多元模型的不足，我们引入了一种方法，即构建辅助时间序列（CATS），它类似于2D时间上下文关注机制，从原始时间序列（OTS）生成辅助时间序列（ATS），以有效表示和整合系列间关系用于预测。ATS的关键原则-连续性，稀疏性和变异性-通过不同模块进行识别和实现。即使是基本的2层MLP作为核心预测器，CATS也取得了最先进的成果，相对于先前的多元模型，它显著减少了复杂性和参数，使其成为高效且可转移的MTSF解决方案。

    arXiv:2403.01673v1 Announce Type: cross  Abstract: For Multivariate Time Series Forecasting (MTSF), recent deep learning applications show that univariate models frequently outperform multivariate ones. To address the difficiency in multivariate models, we introduce a method to Construct Auxiliary Time Series (CATS) that functions like a 2D temporal-contextual attention mechanism, which generates Auxiliary Time Series (ATS) from Original Time Series (OTS) to effectively represent and incorporate inter-series relationships for forecasting. Key principles of ATS - continuity, sparsity, and variability - are identified and implemented through different modules. Even with a basic 2-layer MLP as core predictor, CATS achieves state-of-the-art, significantly reducing complexity and parameters compared to previous multivariate models, marking it an efficient and transferable MTSF solution.
    
[^2]: 使用自适应时间-上下文学习优化多变量预测

    ARM: Refining Multivariate Forecasting with Adaptive Temporal-Contextual Learning. (arXiv:2310.09488v1 [stat.ML])

    [http://arxiv.org/abs/2310.09488](http://arxiv.org/abs/2310.09488)

    本研究提出了ARM，一种多变量的时间-上下文自适应学习方法，用于优化长期时间序列预测。ARM通过采用自适应单变量效应学习、随机丢弃训练策略和多核局部平滑，能更好地处理时间模式和学习系列之间的依赖关系。在多个基准测试中，ARM展示了卓越的性能，而计算成本相对较低。

    

    长期时间序列预测（LTSF）在各个领域中都很重要，但在处理复杂的时间-上下文关系方面面临挑战。由于多变量输入模型表现不如最近的一些单变量模型，我们认为问题在于现有的多变量LTSF变压器模型无法高效地建模系列之间的关系：往往不能正确地捕捉到系列之间的特征差异。为了解决这个问题，我们引入了ARM：一种多变量的时间-上下文自适应学习方法，它是专门为多变量LTSF建模而设计的增强型架构。ARM采用自适应单变量效应学习（AUEL）、随机丢弃（RD）训练策略和多核局部平滑（MKLS）来更好地处理单个系列的时间模式并正确学习系列之间的依赖关系。ARM在多个基准测试上展示了卓越的性能，而与现有方法相比并没有显著增加计算成本。

    Long-term time series forecasting (LTSF) is important for various domains but is confronted by challenges in handling the complex temporal-contextual relationships. As multivariate input models underperforming some recent univariate counterparts, we posit that the issue lies in the inefficiency of existing multivariate LTSF Transformers to model series-wise relationships: the characteristic differences between series are often captured incorrectly. To address this, we introduce ARM: a multivariate temporal-contextual adaptive learning method, which is an enhanced architecture specifically designed for multivariate LTSF modelling. ARM employs Adaptive Univariate Effect Learning (AUEL), Random Dropping (RD) training strategy, and Multi-kernel Local Smoothing (MKLS), to better handle individual series temporal patterns and correctly learn inter-series dependencies. ARM demonstrates superior performance on multiple benchmarks without significantly increasing computational costs compared to
    
[^3]: Transductive Learning的尖锐泛化：一种Transductive Local Rademacher Complexity方法

    Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach. (arXiv:2309.16858v1 [stat.ML])

    [http://arxiv.org/abs/2309.16858](http://arxiv.org/abs/2309.16858)

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。

    

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们的工作将传统的local rademacher complexity (LRC)的思想扩展到了transductive设置中，相对于典型的LRC方法在归纳设置中的分析有了相当大的变化。我们提出了一种基于Rademacher complex的局部化工具，可以应用于各种transductive learning问题，并在适当条件下得到了尖锐的界限。与LRC的发展类似，我们通过从独立变量的方差信息开始构建TLRC，将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。经过精心设计的...

    We introduce a new tool, Transductive Local Rademacher Complexity (TLRC), to analyze the generalization performance of transductive learning methods and motivate new transductive learning algorithms. Our work extends the idea of the popular Local Rademacher Complexity (LRC) to the transductive setting with considerable changes compared to the analysis of typical LRC methods in the inductive setting. We present a localized version of Rademacher complexity based tool wihch can be applied to various transductive learning problems and gain sharp bounds under proper conditions. Similar to the development of LRC, we build TLRC by starting from a sharp concentration inequality for independent variables with variance information. The prediction function class of a transductive learning model is then divided into pieces with a sub-root function being the upper bound for the Rademacher complexity of each piece, and the variance of all the functions in each piece is limited. A carefully designed 
    

