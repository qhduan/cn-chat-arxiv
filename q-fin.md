# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Faber--Schauder approximation based on discrete observations of an antiderivative.](http://arxiv.org/abs/2211.11907) | 本文提出了从反导数离散观测中重构Faber-Schauder系数的方法，并发现了在估计过程中只有最终产生的系数会受到位置的影响及初始值的依赖，并通过放弃这些系数获取了高质量的估计值。 |

# 详细

[^1]: 基于反导数离散观测的Faber-Schauder逼近的鲁棒性

    Robust Faber--Schauder approximation based on discrete observations of an antiderivative. (arXiv:2211.11907v3 [math.NA] UPDATED)

    [http://arxiv.org/abs/2211.11907](http://arxiv.org/abs/2211.11907)

    本文提出了从反导数离散观测中重构Faber-Schauder系数的方法，并发现了在估计过程中只有最终产生的系数会受到位置的影响及初始值的依赖，并通过放弃这些系数获取了高质量的估计值。

    

    本文研究从连续函数$f$ 的反导数$F$ 的离散观测中重构Faber-Schauder系数的问题。我们的方法是通过分段二次样条插值来描述这个问题。我们提供了一个闭合形式的解和深入的误差分析。这些结果导致了一些令人惊讶的观察，这些观察还在古典二次样条插值这一主题上投下了新的光：它们表明，这种方法的众所周知的不稳定性只能在最终一代的估计Faber-Schauder系数中找到，这些系数遭受非局部性和对初始值和给定数据的强依赖性。相比之下，所有其他的Faber-Schauder系数只依赖于数据的局部性，独立于初始值，且具有统一的误差界。因此，我们得出结论，只需放弃最终生成的系数，我们就可以获得一个鲁棒且良好的我们所研究问题的估计值。

    We study the problem of reconstructing the Faber--Schauder coefficients of a continuous function $f$ from discrete observations of its antiderivative $F$. Our approach starts with formulating this problem through piecewise quadratic spline interpolation. We then provide a closed-form solution and an in-depth error analysis. These results lead to some surprising observations, which also throw new light on the classical topic of quadratic spline interpolation itself: They show that the well-known instabilities of this method can be located exclusively within the final generation of estimated Faber--Schauder coefficients, which suffer from non-locality and strong dependence on the initial value and the given data. By contrast, all other Faber--Schauder coefficients depend only locally on the data, are independent of the initial value, and admit uniform error bounds. We thus conclude that a robust and well-behaved estimator for our problem can be obtained by simply dropping the final-gener
    

