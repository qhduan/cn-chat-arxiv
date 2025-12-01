# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation](https://arxiv.org/abs/2311.09922) | 本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。 |
| [^2] | [Self-concordant Smoothing for Large-Scale Convex Composite Optimization](https://arxiv.org/abs/2309.01781) | 提出了适用于大规模凸组合优化问题的自共轭平滑方法，通过变量度量和步长规则优化了近端牛顿算法，有效处理了非光滑函数的结构，提出了Prox-N-SCORE和Prox-GGN-SCORE算法，后者通过重要近似程序显著减少了逆Hessian计算开销。 |
| [^3] | [Unseen Image Synthesis with Diffusion Models.](http://arxiv.org/abs/2310.09213) | 本论文提出使用预训练的扩散模型在未见过的领域合成图像的方法，并理论上和经验上证明了这种方法的有效性。 |
| [^4] | [Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach.](http://arxiv.org/abs/2309.16858) | 我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。 |
| [^5] | [Enhancing Continual Learning with Global Prototypes: Counteracting Negative Representation Drift.](http://arxiv.org/abs/2205.12186) | 该论文提出了一种基于全局原型的持续学习方法，在自监督信息的正则化下学习数据表示，以缓解负面表示漂移问题，并减少持续学习中的灾难性遗忘。 |
| [^6] | [New-Onset Diabetes Assessment Using Artificial Intelligence-Enhanced Electrocardiography.](http://arxiv.org/abs/2205.02900) | 本研究表明，使用人工智能增强的心电图可以有效地识别新发糖尿病成人患者，相较于传统的ADA风险检测方法，该方法具有更好的准确性和特异性。 |
| [^7] | [A Trio Neural Model for Dynamic Entity Relatedness Ranking.](http://arxiv.org/abs/1808.08316) | 这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。 |

# 详细

[^1]: 通过采用整数列表作为多项式基数2指数的集合来实现快速乘法

    Fast multiplication by two's complement addition of numbers represented as a set of polynomial radix 2 indexes, stored as an integer list for massively parallel computation

    [https://arxiv.org/abs/2311.09922](https://arxiv.org/abs/2311.09922)

    本论文介绍了一种基于多项式基数2指数集合的快速乘法方法，在特定位数范围内比传统方法更快。该方法把数字表示为整数索引列表，并实现了分布式计算。

    

    我们演示了一种基于用整数列表表示的多项式基数2指数集合的乘法方法。该方法采用python代码实现了一组算法。我们展示了该方法在某一位数范围内比数论变换(NTT)和卡拉茨巴(Karatsuba)乘法更快。我们还实现了用python代码进行比较，与多项式基数2整数方法进行比较。我们展示了任何整数或实数都可以表示为整数索引列表，表示二进制中的有限级数。该数字的整数索引有限级数可以存储和分布在多个CPU / GPU上。我们展示了加法和乘法运算可以应用于作为索引整数表示的两个补码加法，并可以完全分布在给定的CPU / GPU架构上。我们展示了完全的分布性能。

    We demonstrate a multiplication method based on numbers represented as set of polynomial radix 2 indices stored as an integer list. The 'polynomial integer index multiplication' method is a set of algorithms implemented in python code. We demonstrate the method to be faster than both the Number Theoretic Transform (NTT) and Karatsuba for multiplication within a certain bit range. Also implemented in python code for comparison purposes with the polynomial radix 2 integer method. We demonstrate that it is possible to express any integer or real number as a list of integer indices, representing a finite series in base two. The finite series of integer index representation of a number can then be stored and distributed across multiple CPUs / GPUs. We show that operations of addition and multiplication can be applied as two's complement additions operating on the index integer representations and can be fully distributed across a given CPU / GPU architecture. We demonstrate fully distribute
    
[^2]: 大规模凸组合优化的自共轭平滑方法

    Self-concordant Smoothing for Large-Scale Convex Composite Optimization

    [https://arxiv.org/abs/2309.01781](https://arxiv.org/abs/2309.01781)

    提出了适用于大规模凸组合优化问题的自共轭平滑方法，通过变量度量和步长规则优化了近端牛顿算法，有效处理了非光滑函数的结构，提出了Prox-N-SCORE和Prox-GGN-SCORE算法，后者通过重要近似程序显著减少了逆Hessian计算开销。

    

    我们引入了一种自共轭平滑的概念，用于最小化两个凸函数的和，其中一个是光滑的，另一个可能是非光滑的。我们方法的关键亮点在于所得问题结构的自然特性，为我们提供了一种变量度量选择方法和一个特别适用于近端牛顿类型算法的步长选择规则。此外，我们高效处理了非光滑函数推动的具体结构，如$\ell_1$正则化和分组Lasso惩罚。我们证明了两个算法的收敛性：Prox-N-SCORE，一种近端牛顿算法，和Prox-GGN-SCORE，一种近端广义高斯-牛顿算法。Prox-GGN-SCORE算法突出了一种重要的近似程序，有助于显著减少逆Hessian相关的大部分计算开销。这种近似在...

    arXiv:2309.01781v2 Announce Type: replace-cross  Abstract: We introduce a notion of self-concordant smoothing for minimizing the sum of two convex functions, one of which is smooth and the other may be nonsmooth. The key highlight of our approach is in a natural property of the resulting problem's structure which provides us with a variable-metric selection method and a step-length selection rule particularly suitable for proximal Newton-type algorithms. In addition, we efficiently handle specific structures promoted by the nonsmooth function, such as $\ell_1$-regularization and group-lasso penalties. We prove the convergence of two resulting algorithms: Prox-N-SCORE, a proximal Newton algorithm and Prox-GGN-SCORE, a proximal generalized Gauss-Newton algorithm. The Prox-GGN-SCORE algorithm highlights an important approximation procedure which helps to significantly reduce most of the computational overhead associated with the inverse Hessian. This approximation is essentially useful fo
    
[^3]: 使用扩散模型合成未见过的图像

    Unseen Image Synthesis with Diffusion Models. (arXiv:2310.09213v1 [cs.LG])

    [http://arxiv.org/abs/2310.09213](http://arxiv.org/abs/2310.09213)

    本论文提出使用预训练的扩散模型在未见过的领域合成图像的方法，并理论上和经验上证明了这种方法的有效性。

    

    当前生成领域的趋势是通过扩大模型规模和增加训练数据来实现通用领域表示，而我们在这项工作中选择相反的方向，通过使用预训练和冻结的去噪扩散概率模型（DDPMs）在单领域数据集上进行潜在采样和几何优化来合成未见过的领域图像。我们的关键观察是，即使是仅在单领域图像上进行预训练的DDPMs已经具备了足够的表示能力，可以通过反转潜在编码，并经过双向确定性扩散和去噪轨迹重构任意图像。这促使我们研究未见过图像领域中的潜在空间中沿去噪链的OOD样本的统计和几何行为。值得注意的是，我们在理论上和经验上都表明，反转的OOD样本也建立了高斯分布。

    While the current trend in the generative field is scaling up towards larger models and more training data for generalized domain representations, we go the opposite direction in this work by synthesizing unseen domain images without additional training. We do so via latent sampling and geometric optimization using pre-trained and frozen Denoising Diffusion Probabilistic Models (DDPMs) on single-domain datasets. Our key observation is that DDPMs pre-trained even just on single-domain images are already equipped with sufficient representation abilities to reconstruct arbitrary images from the inverted latent encoding following bi-directional deterministic diffusion and denoising trajectories. This motivates us to investigate the statistical and geometric behaviors of the Out-Of-Distribution (OOD) samples from unseen image domains in the latent spaces along the denoising chain. Notably, we theoretically and empirically show that the inverted OOD samples also establish Gaussians that are 
    
[^4]: Transductive Learning的尖锐泛化：一种Transductive Local Rademacher Complexity方法

    Sharp Generalization of Transductive Learning: A Transductive Local Rademacher Complexity Approach. (arXiv:2309.16858v1 [stat.ML])

    [http://arxiv.org/abs/2309.16858](http://arxiv.org/abs/2309.16858)

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们利用变量的方差信息构建了TLRC，并将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。

    

    我们引入了一种新的工具，Transductive Local Rademacher Complexity (TLRC)，用于分析transductive learning方法的泛化性能并推动新的transductive learning算法的发展。我们的工作将传统的local rademacher complexity (LRC)的思想扩展到了transductive设置中，相对于典型的LRC方法在归纳设置中的分析有了相当大的变化。我们提出了一种基于Rademacher complex的局部化工具，可以应用于各种transductive learning问题，并在适当条件下得到了尖锐的界限。与LRC的发展类似，我们通过从独立变量的方差信息开始构建TLRC，将transductive learning模型的预测函数类分为多个部分，每个部分的Rademacher complexity上界由一个子根函数给出，并限制了每个部分中所有函数的方差。经过精心设计的...

    We introduce a new tool, Transductive Local Rademacher Complexity (TLRC), to analyze the generalization performance of transductive learning methods and motivate new transductive learning algorithms. Our work extends the idea of the popular Local Rademacher Complexity (LRC) to the transductive setting with considerable changes compared to the analysis of typical LRC methods in the inductive setting. We present a localized version of Rademacher complexity based tool wihch can be applied to various transductive learning problems and gain sharp bounds under proper conditions. Similar to the development of LRC, we build TLRC by starting from a sharp concentration inequality for independent variables with variance information. The prediction function class of a transductive learning model is then divided into pieces with a sub-root function being the upper bound for the Rademacher complexity of each piece, and the variance of all the functions in each piece is limited. A carefully designed 
    
[^5]: 基于全局原型的增强持续学习: 对抗负表示漂移

    Enhancing Continual Learning with Global Prototypes: Counteracting Negative Representation Drift. (arXiv:2205.12186v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12186](http://arxiv.org/abs/2205.12186)

    该论文提出了一种基于全局原型的持续学习方法，在自监督信息的正则化下学习数据表示，以缓解负面表示漂移问题，并减少持续学习中的灾难性遗忘。

    

    持续学习旨在学习一系列任务，其中数据分布从一个任务转移到另一个任务。在训练新任务数据时，旧任务的数据表示可能会漂移。一些负面的表示漂移可能会导致灾难性遗忘，因为会导致从本地学习的类别原型和数据表示在任务之间的相关性较差。为了缓解这种表示漂移，我们提出一种方法，通过全局原型指导学习，用自监督信息的正则化来学习数据表示。具体来说，对于NLP任务，我们将每个任务以屏蔽语言建模的方式进行公式化，并通过预训练的语言模型进行相邻注意机制学习任务。实验结果表明，我们提出的方法可以学习出具有较少表示漂移的相当一致的表示，并在不重新采样过去任务的数据的情况下显著减少持续学习中的灾难性遗忘。

    Continual learning (CL) aims to learn a sequence of tasks over time, with data distributions shifting from one task to another. When training on new task data, data representations from old tasks may drift. Some negative representation drift can result in catastrophic forgetting, by causing the locally learned class prototypes and data representations to correlate poorly across tasks. To mitigate such representation drift, we propose a method that finds global prototypes to guide the learning, and learns data representations with the regularization of the self-supervised information. Specifically, for NLP tasks, we formulate each task in a masked language modeling style, and learn the task via a neighbor attention mechanism over a pre-trained language model. Experimental results show that our proposed method can learn fairly consistent representations with less representation drift, and significantly reduce catastrophic forgetting in CL without resampling data from past tasks.
    
[^6]: 使用人工智能增强的心电图进行新发糖尿病评估

    New-Onset Diabetes Assessment Using Artificial Intelligence-Enhanced Electrocardiography. (arXiv:2205.02900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2205.02900](http://arxiv.org/abs/2205.02900)

    本研究表明，使用人工智能增强的心电图可以有效地识别新发糖尿病成人患者，相较于传统的ADA风险检测方法，该方法具有更好的准确性和特异性。

    

    未诊断的糖尿病在患者中占21.4％，由于筛查率的限制，糖尿病可能潜伏无症状而未被检测。本研究旨在通过使用人工智能（AI）增强的心电图（ECG）来确定新发糖尿病的成人患者。 我们训练了一个神经网络，使用12导联心电图和可用的人口统计学数据来估计HbA1c。 我们回顾性地收集了一组包含有配对的ECG和HbA1c数据的病人数据集。结果显示，相较于传统的ADA风险检测，基于ECG的评估效果更好。AI增强的ECG评估的准确性达到81％，灵敏度为80％，特异性为82％。研究结果表明，人工智能增强的ECG可以成为新发糖尿病成人患者的一个有前景的工具，特别是在传统筛查方法有限的人群中。

    Undiagnosed diabetes is present in 21.4% of adults with diabetes. Diabetes can remain asymptomatic and undetected due to limitations in screening rates. To address this issue, questionnaires, such as the American Diabetes Association (ADA) Risk test, have been recommended for use by physicians and the public. Based on evidence that blood glucose concentration can affect cardiac electrophysiology, we hypothesized that an artificial intelligence (AI)-enhanced electrocardiogram (ECG) could identify adults with new-onset diabetes. We trained a neural network to estimate HbA1c using a 12-lead ECG and readily available demographics. We retrospectively assembled a dataset comprised of patients with paired ECG and HbA1c data. The population of patients who receive both an ECG and HbA1c may a biased sample of the complete outpatient population, so we adjusted the importance placed on each patient to generate a more representative pseudo-population. We found ECG-based assessment outperforms the 
    
[^7]: 一种三元神经模型用于动态实体相关性排名

    A Trio Neural Model for Dynamic Entity Relatedness Ranking. (arXiv:1808.08316v4 [cs.IR] UPDATED)

    [http://arxiv.org/abs/1808.08316](http://arxiv.org/abs/1808.08316)

    这篇论文提出了一种基于神经网络的方法，通过动态评估实体相关性，利用集体注意作为监督，能学习到丰富而不同的实体表示，能在大规模数据集上比竞争基线获得更好的结果。

    

    测量实体相关性是许多自然语言处理和信息检索应用的基本任务。之前的研究通常在静态设置和非监督方式下研究实体相关性。然而，现实世界中的实体往往涉及许多不同的关系，因此实体关系随时间变得非常动态。在这项工作中，我们提出了一种基于神经网络的方法来动态评估实体相关性，利用集体注意力作为监督。我们的模型能够在联合框架中学习丰富而不同的实体表示。通过对大规模数据集的广泛实验，我们证明了我们的方法比竞争基线获得了更好的结果。

    Measuring entity relatedness is a fundamental task for many natural language processing and information retrieval applications. Prior work often studies entity relatedness in static settings and an unsupervised manner. However, entities in real-world are often involved in many different relationships, consequently entity-relations are very dynamic over time. In this work, we propose a neural networkbased approach for dynamic entity relatedness, leveraging the collective attention as supervision. Our model is capable of learning rich and different entity representations in a joint framework. Through extensive experiments on large-scale datasets, we demonstrate that our method achieves better results than competitive baselines.
    

