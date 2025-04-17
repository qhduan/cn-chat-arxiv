# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Comparative Study of Artificial Potential Fields and Safety Filters](https://arxiv.org/abs/2403.15743) | 本文通过将人工势场信息整合到CBF-QP框架中，建立了人工势场与安全滤波器之间的连接，并扩展了CBF-QP安全滤波器的设计以适应更一般的动力学模型，从而提供了一种适用于控制仿射动力学模型的一般APF解决方案。 |
| [^2] | [StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773) | StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。 |
| [^3] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |
| [^4] | [H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps.](http://arxiv.org/abs/2309.12716) | H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。 |

# 详细

[^1]: 人工势场与安全滤波器的比较研究

    A Comparative Study of Artificial Potential Fields and Safety Filters

    [https://arxiv.org/abs/2403.15743](https://arxiv.org/abs/2403.15743)

    本文通过将人工势场信息整合到CBF-QP框架中，建立了人工势场与安全滤波器之间的连接，并扩展了CBF-QP安全滤波器的设计以适应更一般的动力学模型，从而提供了一种适用于控制仿射动力学模型的一般APF解决方案。

    

    在本文中，我们展示了由经典运动规划工具设计的控制器，即人工势场（APFs），可以从最近普及的方法中得到：控制屏障函数二次规划（CBF-QP）安全滤波器。通过将APF信息整合到CBF-QP框架中，我们建立了这两种方法之间的桥梁。具体而言，这是通过将有吸引力的势场作为控制李雅普诺夫函数（CLF）来引导名义控制器的设计，然后将排斥性势场作为相互作用CBF（RCBF）来定义一个CBF-QP安全滤波器。基于这种整合，我们将CBF-QP安全滤波器的设计扩展到适应更一般的包含控制仿射结构的动力学模型类。这种扩展产生了一种特殊的CBF-QP安全滤波器和适用于控制仿射动力学模型的一般APF解决方案。

    arXiv:2403.15743v1 Announce Type: cross  Abstract: In this paper, we have demonstrated that the controllers designed by a classical motion planning tool, namely artificial potential fields (APFs), can be derived from a recently prevalent approach: control barrier function quadratic program (CBF-QP) safety filters. By integrating APF information into the CBF-QP framework, we establish a bridge between these two methodologies. Specifically, this is achieved by employing the attractive potential field as a control Lyapunov function (CLF) to guide the design of the nominal controller, and then the repulsive potential field serves as a reciprocal CBF (RCBF) to define a CBF-QP safety filter. Building on this integration, we extend the design of the CBF-QP safety filter to accommodate a more general class of dynamical models featuring a control-affine structure. This extension yields a special CBF-QP safety filter and a general APF solution suitable for control-affine dynamical models. Throug
    
[^2]: StreamingT2V: 一种一致、动态和可扩展的基于文本的长视频生成方法

    StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

    [https://arxiv.org/abs/2403.14773](https://arxiv.org/abs/2403.14773)

    StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。

    

    arXiv:2403.14773v1 公告类型: 交叉 摘要: 文本到视频的扩散模型可以生成遵循文本指令的高质量视频，使得创建多样化和个性化内容变得更加容易。然而，现有方法大多集中在生成高质量的短视频（通常为16或24帧），当天真地扩展到长视频合成的情况时，通常会出现硬裁剪。为了克服这些限制，我们引入了StreamingT2V，这是一种自回归方法，用于生成80、240、600、1200或更多帧的长视频，具有平滑的过渡。主要组件包括：（i）一种名为条件注意力模块（CAM）的短期记忆块，通过注意机制将当前生成条件设置为先前块提取的特征，实现一致的块过渡，（ii）一种名为外观保存模块的长期记忆块，从第一个视频块中提取高级场景和对象特征，以防止th

    arXiv:2403.14773v1 Announce Type: cross  Abstract: Text-to-video diffusion models enable the generation of high-quality videos that follow text instructions, making it easy to create diverse and individual content. However, existing approaches mostly focus on high-quality short video generation (typically 16 or 24 frames), ending up with hard-cuts when naively extended to the case of long video synthesis. To overcome these limitations, we introduce StreamingT2V, an autoregressive approach for long video generation of 80, 240, 600, 1200 or more frames with smooth transitions. The key components are:(i) a short-term memory block called conditional attention module (CAM), which conditions the current generation on the features extracted from the previous chunk via an attentional mechanism, leading to consistent chunk transitions, (ii) a long-term memory block called appearance preservation module, which extracts high-level scene and object features from the first video chunk to prevent th
    
[^3]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    
[^4]: H2O+: 一种改进的混合离线和在线强化学习框架，用于动力学差距问题

    H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps. (arXiv:2309.12716v1 [cs.LG])

    [http://arxiv.org/abs/2309.12716](http://arxiv.org/abs/2309.12716)

    H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。

    

    在没有高精度模拟环境或大量离线数据的情况下，使用强化学习（RL）解决实际复杂任务可能相当具有挑战性。在非完美模拟环境中训练的在线RL代理可能会受到严重的模拟与现实问题。虽然离线RL方法可以绕过对模拟器的需求，但往往对离线数据集的大小和质量提出了苛刻的要求。最近出现的混合离线和在线RL提供了一个有吸引力的框架，可以同时使用有限的离线数据和不完美的模拟器进行可转移策略学习。本文提出了一种名为H2O+的新算法，该算法在桥接不同的离线和在线学习方法的同时，也考虑了真实和模拟环境之间的动力学差距。通过广泛的仿真和实际机器人实验，我们证明了H2O+在性能和灵活性上优于先进的跨域在线方法

    Solving real-world complex tasks using reinforcement learning (RL) without high-fidelity simulation environments or large amounts of offline data can be quite challenging. Online RL agents trained in imperfect simulation environments can suffer from severe sim-to-real issues. Offline RL approaches although bypass the need for simulators, often pose demanding requirements on the size and quality of the offline datasets. The recently emerged hybrid offline-and-online RL provides an attractive framework that enables joint use of limited offline data and imperfect simulator for transferable policy learning. In this paper, we develop a new algorithm, called H2O+, which offers great flexibility to bridge various choices of offline and online learning methods, while also accounting for dynamics gaps between the real and simulation environment. Through extensive simulation and real-world robotics experiments, we demonstrate superior performance and flexibility over advanced cross-domain online
    

