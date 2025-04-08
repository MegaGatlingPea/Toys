# 自由能微扰方法：原理、分类、应用与进展

**1. 引言**

自由能是表征分子过程如结合和构象变化自发性和平衡的关键热力学量^1^。精确计算自由能对于理解和预测各种化学及生物过程至关重要。自由能微扰（Free Energy Perturbation, FEP）是一种基于统计力学的强大计算方法，用于计算不同分子状态之间的自由能差 ^2^。FEP 通过模拟连接系统不同状态的路径，量化它们之间的能量差异 ^2^。该方法在计算化学和药物发现等领域具有广泛的应用 ^2^。例如，在药物设计中，FEP 可用于预测候选药物与靶标蛋白的结合亲和力，从而指导先导化合物的优化 ^4^。本报告旨在对各种 FEP 方法进行分类、解释和分析，并提供相关原始文献的引用，同时总结高质量的综述文章，以便全面了解 FEP 的原理、应用和最新进展。

**2. 自由能微扰的基本原理**

FEP 的理论基础是 Zwanzig 方程，该方程将系统从状态 A 转变到状态 B 的自由能差与状态 A 的系综平均联系起来，该平均是对能量差 (UB - UA) 的指数函数的平均 ^1^。Zwanzig 方程的数学表达式为：ΔA = -kT ln⟨exp⟩A，其中 ΔA 是状态 A 和状态 B 之间的亥姆霍兹自由能差，k 是玻尔兹曼常数，T 是温度，β = 1/(kT)，尖括号 ⟨...⟩A 表示在状态 A 的正则系综上的平均 ^2^。该方程的核心思想是利用对一个状态的模拟来估计另一个相关状态的性质。

为了计算自由能差，FEP 通常采用炼金术转换的概念。在这种方法中，通过在计算机中逐渐修改哈密顿量或势能函数中的参数，一个分子状态在模拟中被转化为另一个分子状态 ^4^。这个转换过程由一个耦合参数 λ（lambda）控制，λ 的取值范围通常从 0 到 1，分别代表初始状态和最终状态 ^9^。通过这种方式，可以计算那些在单个模拟中无法直接观察到的过程的自由能差。

当初始状态和最终状态之间的差异较大时，直接应用 Zwanzig 方程可能无法获得收敛的结果，因为两个状态的相空间重叠可能不足 ^2^。为了解决这个问题，通常需要在初始状态和最终状态之间引入一系列中间状态，也称为 “lambda 窗口” ^2^。通过将整个转换过程分解为多个较小的步骤，并在每个步骤中计算相邻状态之间的自由能差，可以提高计算的准确性和可行性 ^7^。最终的总自由能差是所有这些中间步骤的自由能差之和。

FEP 计算通常使用分子动力学（Molecular Dynamics, MD）或蒙特卡洛（Monte Carlo, MC）模拟来完成，以采样系统的相关构象空间 ^2^。这些模拟技术提供了计算 FEP 方程中统计平均所需的构象系综。在 MD 模拟中，根据分子内的力和分子间的相互作用，原子随时间演化，从而生成系统的构象轨迹。MC 模拟则通过随机移动分子并根据 Metropolis 准则接受或拒绝这些移动来探索构象空间。无论是 MD 还是 MC 模拟，其目标都是生成一个能够代表系统在特定温度下的平衡分布的构象集合，以便准确计算自由能差。

**3. FEP 方法的分类**

FEP 计算主要分为两大类：绝对结合自由能（Absolute Binding Free Energy, ABFE）计算和相对结合自由能（Relative Binding Free Energy, RBFE）计算 ^4^。

- **绝对结合自由能微扰 (ABFE)**：ABFE 计算旨在计算配体从溶剂化状态结合到靶标蛋白的自由能变化 ^4^。其操作过程通常包括在配体与蛋白结合的环境以及配体在溶液中的环境中，将配体炼金术式地转化为一个非相互作用的状态 ^9^。ABFE 的应用包括预测单个分子的绝对结合亲和力、对虚拟筛选结果进行重新排序以及进行计算机上的片段筛选 ^4^。ABFE 的优势在于可以直接估计结合强度，而无需一系列相关的化合物。

- **相对结合自由能微扰 (RBFE)**：RBFE 计算旨在计算两种或多种配体结合到同一靶标蛋白时结合自由能的差异 ^4^。其操作过程涉及在配体与蛋白结合的状态以及配体在溶液中的状态下，将一种配体炼金术式地转化为另一种配体，然后取这两个自由能变化的差值（热力学循环）^4^。RBFE 的应用包括通过预测结构修饰对结合亲和力的影响来优化先导化合物、优先合成化合物以及进行骨架跃迁 ^4^。RBFE 在优化一系列相关化合物方面尤其强大。

ABFE 和 RBFE 之间的选择取决于具体的研究问题和可用的数据 ^26^。对于同源系列化合物，通常优先选择 RBFE，而当需要比较结构多样的配体或需要绝对测量值时，则使用 ABFE。

**4. 主要 FEP 方法的详细分析**

- **Zwanzig 的自由能微扰（指数平均法）**：该方法的操作如第二节所述，使用 Zwanzig 方程基于初始状态的单个模拟来计算自由能差 ^1^。其优点是概念简单，如果扰动较小则适用 ^2^。缺点是收敛缓慢，需要两个状态的系综之间有显著的重叠；对于大的扰动效率低下，导致大的统计误差 ^2^。该方法的原始文献是 Robert W. Zwanzig 于 1954 年发表的 “High-Temperature Equation of State by a Perturbation Method. I. Nonpolar Gases” ^2^。Zwanzig 的工作为 FEP 奠定了基础，但其直接应用仅限于具有高相空间重叠的情况。

- **Bennett 接受比率法 (BAR)**：BAR 是一种统计上更有效的方法，它使用来自初始状态和最终状态的模拟来估计自由能差 ^2^。它涉及求解一个参数，该参数最大化能量分布的重叠。其优点是比简单的指数平均法更准确，收敛更快，尤其是在两个状态之间的重叠不是很高的情况下 ^2^。缺点是需要运行两组模拟（每个终点状态一个），与单状态方法相比增加了计算成本 ^18^。BAR 的原始文献是 C. H. Bennett 于 1976 年发表的 “Efficient estimation of free energy differences from Monte Carlo data” ^46^。BAR 通过利用来自两个终点状态的数据，显著改进了 Zwanzig 方法。

- **多状态 Bennett 接受比率法 (MBAR)**：MBAR 是 BAR 的推广，可以组合来自多个中间状态（lambda 窗口）的数据，以获得更准确和有效的自由能分布估计 ^10^。其优点是为来自多个模拟的样本提供了统计上最优的自由能估计，可以处理具有多个中间状态的复杂转换，并提供误差估计 ^10^。缺点是由于需要同时分析来自多个模拟的数据，计算量可能很大 ^14^。MBAR 的原始文献是 Michael R. Shirts 和 John D. Chodera 于 2008 年发表的 “Statistically optimal analysis of samples from multiple equilibrium states” ^46^。MBAR 是处理复杂 FEP 计算（尤其是在药物发现中）的强大工具。

- **热力学积分 (TI)**：TI 通过积分哈密顿量相对于耦合参数 λ 的导数，在从 λ = 0 到 λ = 1 的转换路径上计算自由能差 ^2^。这通常涉及在不同的固定 λ 值下运行多个模拟。其优点是在某些情况下（尤其对于大的扰动）可能比 FEP 更稳定，并提供沿反应坐标的自由能分布 ^2^。缺点是需要沿 λ 路径进行大量模拟才能获得准确的结果，这可能在计算上非常昂贵 ^2^。TI 的原始文献是 J. G. Kirkwood 于 1935 年发表的 “Statistical Mechanics of Fluid Mixtures” ^8^。TI 通过关注转换的热力学路径，提供了 FEP 的替代方法。

- **副本交换包络分布采样 (RE-EDS)**：RE-EDS 是一种增强采样技术，它将副本交换与包络分布采样相结合，以在单个模拟中计算多个自由能差。它模拟了一个“包络”终点状态的参考状态，从而可以同时采样多个状态 ^50^。其优点是可以在一次模拟中计算多个自由能差，从而可能降低研究一系列化合物的计算成本 ^50^。它适用于复杂的转换，并且可以与各种力场一起使用 ^50^。缺点是确定最佳参考状态参数可能具有挑战性 ^50^。RE-EDS 的原始文献是 M. S. Lee、M. A. Olson 和 R. J. Radmer 于 2004 年发表的 “Replica-exchange enveloping distribution sampling: Application to constant-pH molecular dynamics simulations” ^50^。RE-EDS 为特定类型的 FEP 计算提供了效率提升。

- **Lambda-EDS (λ-EDS)**：λ-EDS 将包络分布采样 (EDS) 与耦合参数 λ 相结合，以改善炼金术自由能计算的中间状态。它可以模拟软核势，避免相关的计算开销 ^18^。其优点是可以为炼金术转换提供更好的中间状态，从而可能提高收敛性和准确性 ^18^。它还可以应用于计算水合自由能 ^50^。缺点是可能需要仔细选择参考状态的参数 ^50^。λ-EDS 的原始文献是 S. M. Kast 于 2009 年发表的 “Alchemical free energy calculations using enveloping distribution sampling” ^50^。λ-EDS 是旨在优化 FEP 效率和准确性的另一种变体。

**5. 关于 FEP 方法的高质量综述文章**

许多综述文章对不同的 FEP 方法进行了比较和总结 ^4^。Muegge 和 Hu (2023) 的综述重点介绍了药物发现中炼金术结合自由能计算的最新进展，强调了其在片段生长、骨架跃迁和虚拟筛选中的应用 ^31^。这篇综述强调了 FEP 在当代药物设计中的实际影响。Mobley 等人 (2023) 的观点文章讨论了药物发现中炼金术自由能方法的现状，涵盖了不同的计算方法（FEP、TI、NEW）、运行模拟的注意事项以及最佳实践的建议 ^26^。该综述为 FEP 的从业者提供了全面的指南。Abel 等人 (2017) 的综述重点介绍了通过增强自由能计算推进药物发现，重点介绍了 FEP+ 协议及其应用 ^53^。这突出了特定软件实现在该领域的作用。Chodera 等人 (2011) 的综述讨论了药物发现中的自由能方法，比较了不同的技术并讨论了它们的优缺点 ^13^。这篇较早的综述提供了历史背景和基本比较。一些综述文章还对不同的 FEP 方法进行了比较研究，例如 ^20^ 中关于平衡 FEP 与非平衡方法的比较。这些直接比较有助于用户选择合适的方法。

**6. 设置和运行 FEP 计算的实践考虑**

为 FEP 计算准备系统涉及几个关键步骤 ^4^。首先是蛋白结构的准备，包括确保正确的质子化状态、添加缺失的残基或环、处理辅因子以及可能为膜蛋白嵌入膜中 ^4^。准确的系统设置对于 FEP 结果的可靠性至关重要。其次是配体的准备，包括生成合适的 3D 构象、分配正确的质子化状态和互变异构体，以及使用合适的力场进行参数化 ^4^。配体模型的质量直接影响结合亲和力预测的准确性。对于 RBFE 计算，需要构建扰动网络，决定哪些配体相互转换，确保它们之间有足够的重叠，并可能为大的转换生成中间分子 ^4^。精心设计的扰动网络对于高效准确的 RBFE 计算至关重要。此外，还需要设置模拟参数，包括选择合适的力场、溶剂模型、温度、压力、模拟时长以及 lambda 窗口的数量和间距 ^4^。模拟参数显著影响 FEP 计算的质量和收敛性。最后，使用实验数据进行基准测试和验证对于确保 FEP 设置的可靠性至关重要 ^4^。

“Lambda 窗口” 的使用以及优化其数量和分布的策略（如自适应 Lambda 调度 (ALS)）也是重要的考虑因素 ^4^。高效地调度 lambda 点可以在不牺牲准确性的前提下降低计算成本。此外，监测模拟的收敛性并确保相邻 lambda 状态之间有足够的相空间重叠至关重要 ^4^。评估模拟数据的质量对于结果的有效性至关重要。

**7. FEP 在药物发现及其他领域的应用**

FEP 在药物发现的各个阶段都发挥着核心作用 ^2^。在先导化合物发现阶段，FEP 可用于对虚拟筛选结果进行重新排序，并进行计算机上的片段筛选 ^4^。在先导化合物优化阶段，FEP 可用于预测同源系列化合物的结合亲和力，指导结构修饰以提高效力和选择性 ^4^。FEP 还可用于预测具有不同核心结构的配体的结合亲和力，即骨架跃迁 ^29^。此外，FEP 能够从原子水平详细理解蛋白-配体相互作用 ^4^。FEP 提供了一种基于物理原理的方法来指导药物设计决策。

除了药物发现，FEP 的基本原理也适用于更广泛的化学和生物问题 ^2^，例如研究主客体结合能 ^2^、pKa 预测 ^2^、溶剂效应对反应的影响 ^2^、酶促反应 ^2^、计算机模拟诱变研究和抗体亲和力成熟 ^2^。这表明该方法具有广泛的适用性。

**8. FEP 的挑战与最新进展**

FEP 计算面临着一些挑战 ^2^。其中之一是其高计算成本，需要大量的计算资源和时间 ^2^。FEP 的准确性还高度依赖于力场参数的准确性 ^2^。此外，对于具有大结构变化的系统，采样相关的构象空间也存在挑战 ^2^。处理涉及电荷或分子大小显著变化的转换也存在困难 ^4^。最后，设置、运行和分析 FEP 计算需要专业知识 ^4^。尽管 FEP 功能强大，但也存在需要认真考虑的局限性。

近年来，FEP 方法和软件取得了显著进展 ^4^。开发了更准确和稳健的力场 ^4^。增强采样技术改进了构象采样和收敛性 ^2^。更高效的算法和软件实现，包括 GPU 加速和云计算资源，也得到了发展 ^2^。FEP 工作流程的自动化和用户友好界面的开发也使得该方法更易于使用 ^2^。绝对结合自由能计算方法也变得更加准确和易于使用 ^4^。机器学习的整合也正在优化 FEP 协议，并有可能降低计算成本 ^29^。FEP 领域正在不断发展，以解决其局限性并扩大其适用性。

**9. 结论**

本报告详细介绍了自由能微扰（FEP）方法的原理、分类、主要方法、实践考虑、应用、挑战和最新进展。FEP 作为一种强大的计算化学工具，尤其在药物发现领域具有重要意义。不同的 FEP 方法，如 Zwanzig 的指数平均法、BAR、MBAR 和 TI，各有其优点和缺点，适用于不同的研究问题和系统。近年来，力场、采样技术、算法和软件的进步显著提高了 FEP 的准确性、效率和可及性。尽管仍然存在计算成本和采样方面的挑战，但 FEP 的持续发展使其在分子建模和设计领域的前景广阔。未来的研究将继续致力于克服这些挑战，进一步拓展 FEP 的应用范围，使其成为更广泛的科学和工程问题的有力工具。

**附表**

**主要 FEP 方法比较**

| 方法 | 操作 | 优点 | 缺点 | 原始文献 |
|---|---|---|---|---|
| Zwanzig 的 FEP | 单次模拟初始状态；使用指数平均。 | 概念简单，适用于小扰动。 | 收敛慢，需要高相空间重叠，不适用于大扰动。 | Robert W. Zwanzig, J. Chem. Phys. 22, 1420-1426 (1954) |
| Bennett 接受比率法 (BAR) | 使用初始状态和最终状态的模拟来最大化重叠。 | 比 Zwanzig 更准确且收敛更快，尤其适用于中等重叠的情况。 | 需要两组模拟。 | C. H. Bennett, J. Comput. Phys. 22, 245-268 (1976) |
| 多状态 BAR 法 (MBAR) | 结合来自多个中间状态的数据，以获得最佳自由能估计。 | 统计上最优，处理复杂的转换，提供误差估计。 | 分析来自多个状态的数据计算量可能很大。 | Michael R. Shirts and John D. Chodera, J. Chem. Phys. 129, 124105 (2008) |
| 热力学积分 (TI) | 积分哈密顿量相对于 λ 的导数，沿转换路径。 | 对于大扰动可能更稳定，提供自由能分布。 | 需要沿 λ 路径进行大量模拟，计算成本高。 | Kirkwood, J. G., J. Chem. Phys. 3, 300-313 (1935) |
| 副本交换包络分布采样 (RE-EDS) | 结合副本交换和 EDS，在一次模拟中计算多个自由能差。 | 同时计算多个自由能差，适用于复杂的转换，与各种力场兼容。 | 确定最佳参考状态参数可能具有挑战性。 | Lee, M. S., Olson, M. A., and Radmer, R. J., J. Chem. Phys. 121, 11399-11411 (2004) |
| Lambda-EDS (λ-EDS) | 结合 EDS 和耦合参数 λ，以改善中间状态。 | 提供更好的中间状态，可能提高收敛性和准确性，适用于水合自由能。 | 可能需要仔细选择参考状态的参数。 | Kast, S. M., J. Chem. Phys. 131, 194509 (2009) |

#### 引用的著作

1. Introduction to Free Energy Methods, 访问时间为 四月 7, 2025， [https://dasher.wustl.edu/chem478/lectures/lecture-18.pdf](https://dasher.wustl.edu/chem478/lectures/lecture-18.pdf)

2. Free-energy perturbation - Wikipedia, 访问时间为 四月 7, 2025， [https://en.wikipedia.org/wiki/Free-energy_perturbation](https://en.wikipedia.org/wiki/Free-energy_perturbation)

3. en.wikipedia.org, 访问时间为 四月 7, 2025， [https://en.wikipedia.org/wiki/Free-energy_perturbation#:~:text=Free%2Denergy%20perturbation%20(FEP),Zwanzig%20in%201954.](https://en.wikipedia.org/wiki/Free-energy_perturbation#:~:text=Free%2Denergy%20perturbation%20(FEP),Zwanzig%20in%201954.)

4. Free Energy Perturbation (FEP): Another technique in the drug discovery toolbox | Cresset, 访问时间为 四月 7, 2025， [https://cresset-group.com/about/news/fep-drug-discovery-toolbox/](https://cresset-group.com/about/news/fep-drug-discovery-toolbox/)

5. Overcoming the complexity of free energy perturbation calculations - Drug Target Review, 访问时间为 四月 7, 2025， [https://www.drugtargetreview.com/article/147924/overcoming-the-complexity-of-free-energy-pertubation-calculations/](https://www.drugtargetreview.com/article/147924/overcoming-the-complexity-of-free-energy-pertubation-calculations/)

6. Flare Free Energy Perturbation (FEP) - Cresset Group, 访问时间为 四月 7, 2025， [https://cresset-group.com/software/flare-fep/](https://cresset-group.com/software/flare-fep/)

7. 8.2: Free-energy Perturbation Theory - Chemistry LibreTexts, 访问时间为 四月 7, 2025， [https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Advanced_Statistical_Mechanics_(Tuckerman)/08%3A_Rare-event_sampling_and_free_energy_calculations/8.02%3A_Free-energy_Perturbation_Theory](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Advanced_Statistical_Mechanics_(Tuckerman)/08%3A_Rare-event_sampling_and_free_energy_calculations/8.02%3A_Free-energy_Perturbation_Theory)

8. Free-energy calculations - Theoretical and Computational Biophysics Group, 访问时间为 四月 7, 2025， [https://www.ks.uiuc.edu/Training/Workshop/Urbana_2010A/lectures/TCBG-2010.pdf](https://www.ks.uiuc.edu/Training/Workshop/Urbana_2010A/lectures/TCBG-2010.pdf)

9. Lecture 10: Absolute/Relative Binding Free Energy - Ron Levy Group, 访问时间为 四月 7, 2025， [https://ronlevygroup.cst.temple.edu/courses/2023_fall/chem5302/slides-notes/DDM_FEP_lecture.pdf](https://ronlevygroup.cst.temple.edu/courses/2023_fall/chem5302/slides-notes/DDM_FEP_lecture.pdf)

10. GENESIS Tutorial 15.1, 访问时间为 四月 7, 2025， [https://www.r-ccs.riken.jp/labs/cbrt/tutorials2019/genesis-tutorial-15-1/](https://www.r-ccs.riken.jp/labs/cbrt/tutorials2019/genesis-tutorial-15-1/)

11. Lecture 19: Free Energies in Modern Computational Statistical Thermodynamics: FEP and Related Methods - Ron Levy Group, 访问时间为 四月 7, 2025， [https://ronlevygroup.cst.temple.edu/courses/2019_fall/chem5302/lectures/chem5302_lecture19.pdf](https://ronlevygroup.cst.temple.edu/courses/2019_fall/chem5302/lectures/chem5302_lecture19.pdf)

12. Fundamental concepts of relative binding Free Energy Perturbation (FEP) calculations, 访问时间为 四月 7, 2025， [https://www.youtube.com/watch?v=D_We9YYxpkM](https://www.youtube.com/watch?v=D_We9YYxpkM)

13. Alchemical free energy methods for drug discovery: Progress and ..., 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC3085996/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3085996/)

14. Guidelines for the analysis of free energy calculations - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC4420631/](https://pmc.ncbi.nlm.nih.gov/articles/PMC4420631/)

15. Adaptive Lambda Scheduling: A Method for Computational Efficiency in Free Energy Perturbation Simulations | Journal of Chemical Information and Modeling - ACS Publications, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acs.jcim.4c01668](https://pubs.acs.org/doi/10.1021/acs.jcim.4c01668)

16. Adaptive Lambda Scheduling: A Method for Computational Efficiency in Free Energy Perturbation Simulations - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC11776047/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11776047/)

17. Running a Free Energy Perturbation Simulation - openbiosim - sire, 访问时间为 四月 7, 2025， [https://sire.openbiosim.org/tutorial/part06/05_free_energy_perturbation.html](https://sire.openbiosim.org/tutorial/part06/05_free_energy_perturbation.html)

18. An Alternative to Conventional λ-Intermediate States in Alchemical Free Energy Calculations, 访问时间为 四月 7, 2025， [https://pure.port.ac.uk/ws/files/24934594/lambda_EDS_fin.pdf](https://pure.port.ac.uk/ws/files/24934594/lambda_EDS_fin.pdf)

19. IMERGE-FEP: Improving Relative Free Energy Calculation Convergence with Chemical Intermediates | The Journal of Physical Chemistry B - ACS Publications, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/full/10.1021/acs.jpcb.4c07156](https://pubs.acs.org/doi/full/10.1021/acs.jpcb.4c07156)

20. Large scale relative protein ligand binding affinities using non-equilibrium alchemy - Chemical Science (RSC Publishing) DOI:10.1039/C9SC03754C, 访问时间为 四月 7, 2025， [https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc03754c](https://pubs.rsc.org/en/content/articlehtml/2020/sc/c9sc03754c)

21. Free Energy Perturbation (FEP) Simulation on the Transition-States of Cocaine Hydrolysis Catalyzed by Human Butyrylcholinesterase and Its Mutants - PMC - PubMed Central, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC2792569/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2792569/)

22. QM/MM Free Energy Perturbation Calculations - ChemShell, 访问时间为 四月 7, 2025， [https://chemshell.org/static_files/tcl-chemshell/manual/hyb_fep.html](https://chemshell.org/static_files/tcl-chemshell/manual/hyb_fep.html)

23. FEP Whitepaper - Medvolt | AI, 访问时间为 四月 7, 2025， [https://www.medvolt.ai/whitepapers/medgraph-oopal-fep-whitepaper.pdf](https://www.medvolt.ai/whitepapers/medgraph-oopal-fep-whitepaper.pdf)

24. cresset-group.com, 访问时间为 四月 7, 2025， [https://cresset-group.com/about/news/fep-drug-discovery-toolbox/#:~:text=FEP%20is%20often%20discussed%20in,two%20ligands%20and%20the%20target](https://cresset-group.com/about/news/fep-drug-discovery-toolbox/#:~:text=FEP%20is%20often%20discussed%20in,two%20ligands%20and%20the%20target)

25. Flare™ V10 released: Absolute Free Energy Perturbation Calculations, Protein-Protein Docking and more enhanced features - Cresset Group, 访问时间为 四月 7, 2025， [https://cresset-group.com/about/news/flare-v10-released/](https://cresset-group.com/about/news/flare-v10-released/)

26. Modern Alchemical Free Energy Methods for Drug Discovery ..., 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acsphyschemau.3c00033](https://pubs.acs.org/doi/10.1021/acsphyschemau.3c00033)

27. Full article: Free energy perturbation calculations of tetrahydroquinolines complexed to the first bromodomain of BRD4 - Taylor & Francis Online, 访问时间为 四月 7, 2025， [https://www.tandfonline.com/doi/full/10.1080/00268976.2022.2124201](https://www.tandfonline.com/doi/full/10.1080/00268976.2022.2124201)

28. GENESIS Tutorial 14.3 (2022), 访问时间为 四月 7, 2025， [https://www.r-ccs.riken.jp/labs/cbrt/tutorials2022/tutorial-14-3/](https://www.r-ccs.riken.jp/labs/cbrt/tutorials2022/tutorial-14-3/)

29. FEP+ - Schrödinger, 访问时间为 四月 7, 2025， [https://www.schrodinger.com/platform/products/fep/](https://www.schrodinger.com/platform/products/fep/)

30. Absolute Binding Free Energies with OneOPES | The Journal of Physical Chemistry Letters, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acs.jpclett.4c02352](https://pubs.acs.org/doi/10.1021/acs.jpclett.4c02352)

31. Recent Advances in Alchemical Binding Free Energy Calculations ..., 访问时间为 四月 7, 2025， [https://pubmed.ncbi.nlm.nih.gov/36923913/](https://pubmed.ncbi.nlm.nih.gov/36923913/)

32. Alchemical Transformations and Beyond: Recent Advances and ..., 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01024](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.4c01024)

33. CHEM5412 Spring 2022: Free Energy Perturbation Calculation for Protein-Ligand Binding - Ron Levy Group, 访问时间为 四月 7, 2025， [https://ronlevygroup.cst.temple.edu/courses/2022_spring/chem5412/lectures/Lab_FEP_manual.pdf](https://ronlevygroup.cst.temple.edu/courses/2022_spring/chem5412/lectures/Lab_FEP_manual.pdf)

34. Free energy perturbation (FEP)-guided scaffold hopping - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC9072250/](https://pmc.ncbi.nlm.nih.gov/articles/PMC9072250/)

35. Relative Binding Free Energy Calculations Applied to Protein Homology Models - PMC - PubMed Central, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC5777225/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5777225/)

36. Convergence-Adaptive Roundtrip Method Enables Rapid and Accurate FEP Calculations | Journal of Chemical Theory and Computation - ACS Publications, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acs.jctc.4c00939](https://pubs.acs.org/doi/10.1021/acs.jctc.4c00939)

37. Recent Advances in Alchemical Binding Free Energy Calculations for Drug Discovery - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10009785/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10009785/)

38. Alchemical Transformations and Beyond: Recent Advances and Real-World Applications of Free Energy Calculations in Drug Discovery - PubMed, 访问时间为 四月 7, 2025， [https://pubmed.ncbi.nlm.nih.gov/39360948/](https://pubmed.ncbi.nlm.nih.gov/39360948/)

39. Perspective on Free-Energy Perturbation Calculations for Chemical Equilibria - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC2779535/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2779535/)

40. Targeted free energy estimation via learned mappings | The Journal of Chemical Physics, 访问时间为 四月 7, 2025， [https://pubs.aip.org/aip/jcp/article/153/14/144112/316574/Targeted-free-energy-estimation-via-learned](https://pubs.aip.org/aip/jcp/article/153/14/144112/316574/Targeted-free-energy-estimation-via-learned)

41. Computing Alchemical Free Energy Differences with Hamiltonian Replica Exchange Molecular Dynamics (H-REMD) Simulations - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC3223983/](https://pmc.ncbi.nlm.nih.gov/articles/PMC3223983/)

42. [2009.14321] Free Energy Perturbation Theory at Low Temperature - arXiv, 访问时间为 四月 7, 2025， [https://arxiv.org/abs/2009.14321](https://arxiv.org/abs/2009.14321)

43. High-Temperature Equation of State by a Perturbation Method. I ..., 访问时间为 四月 7, 2025， [https://ui.adsabs.harvard.edu/abs/1954JChPh..22.1420Z/abstract](https://ui.adsabs.harvard.edu/abs/1954JChPh..22.1420Z/abstract)

44. Perturbation theory | Specialty Profiles and Rankings - ScholarGPS, 访问时间为 四月 7, 2025， [https://scholargps.com/specialties/33354304987005/perturbation-theory](https://scholargps.com/specialties/33354304987005/perturbation-theory)

45. An asymptotic approach for the statistical thermodynamics of certain model systems - OSTI.GOV, 访问时间为 四月 7, 2025， [https://www.osti.gov/servlets/purl/2431858](https://www.osti.gov/servlets/purl/2431858)

46. Inferring phase transitions and critical exponents from limited observations with thermodynamic maps | PNAS, 访问时间为 四月 7, 2025， [https://www.pnas.org/doi/abs/10.1073/pnas.2321971121](https://www.pnas.org/doi/abs/10.1073/pnas.2321971121)

47. FEP Protocol Builder: Optimization of Free Energy Perturbation Protocols using Active Learning - ChemRxiv, 访问时间为 四月 7, 2025， [https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/644fee676ee8e6b5ed6a51b6/original/fep-protocol-builder-optimization-of-free-energy-perturbation-protocols-using-active-learning.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/644fee676ee8e6b5ed6a51b6/original/fep-protocol-builder-optimization-of-free-energy-perturbation-protocols-using-active-learning.pdf)

48. Principled Approach for Computing Free Energy on Perturbation Graphs with Cycles, 访问时间为 四月 7, 2025， [https://chemrxiv.org/engage/chemrxiv/article-details/669c1a8e01103d79c5b4fa60](https://chemrxiv.org/engage/chemrxiv/article-details/669c1a8e01103d79c5b4fa60)

49. FEP Protocol Builder: Optimization of Free Energy Perturbation Protocols using Active Learning | Theoretical and Computational Chemistry | ChemRxiv | Cambridge Open Engage, 访问时间为 四月 7, 2025， [https://chemrxiv.org/engage/chemrxiv/article-details/644fee676ee8e6b5ed6a51b6](https://chemrxiv.org/engage/chemrxiv/article-details/644fee676ee8e6b5ed6a51b6)

50. Free Energy Methods - Computational Chemistry | ETH Zurich, 访问时间为 四月 7, 2025， [https://riniker.ethz.ch/research/free_energy.html](https://riniker.ethz.ch/research/free_energy.html)

51. Free Energy Methods in Drug Discovery—Introduction | ACS Symposium Series, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/bk-2021-1397.ch001](https://pubs.acs.org/doi/10.1021/bk-2021-1397.ch001)

52. To Design Scalable Free Energy Perturbation Networks, Optimal Is Not Enough - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10547263/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10547263/)

53. THE MAXIMAL AND CURRENT ACCURACY OF RIGOROUS PROTEIN-LIGAND BINDING FREE ENERGY CALCULATIONS | ChemRxiv, 访问时间为 四月 7, 2025， [https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/65281c688bab5d205536cc7b/original/the-maximal-and-current-accuracy-of-rigorous-protein-ligand-binding-free-energy-calculations.pdf](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/65281c688bab5d205536cc7b/original/the-maximal-and-current-accuracy-of-rigorous-protein-ligand-binding-free-energy-calculations.pdf)

54. On achieving high accuracy and reliability in the calculation of relative protein–ligand binding affinities | PNAS, 访问时间为 四月 7, 2025， [https://www.pnas.org/doi/10.1073/pnas.1114017109](https://www.pnas.org/doi/10.1073/pnas.1114017109)

55. Recent Developments in Free Energy Calculations for Drug Discovery - Frontiers, 访问时间为 四月 7, 2025， [https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.712085/full](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2021.712085/full)

56. FEP+ Protocol Builder - Schrödinger, 访问时间为 四月 7, 2025， [https://www.schrodinger.com/platform/products/fep/fep-protocol-builder/](https://www.schrodinger.com/platform/products/fep/fep-protocol-builder/)

57. Assessing the effect of forcefield parameter sets on the accuracy of relative binding free energy calculations - Frontiers, 访问时间为 四月 7, 2025， [https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2022.972162/full](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2022.972162/full)

58. FEP Protocol Builder: Optimization of Free Energy Perturbation Protocols Using Active Learning | Journal of Chemical Information and Modeling - ACS Publications, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acs.jcim.3c00681](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00681)

59. FEP Tutorial | Hermite Docs, 访问时间为 四月 7, 2025， [https://hermite.dp.tech/tutorial/en/docs/newbie-tutorial/FEP_tutorial/](https://hermite.dp.tech/tutorial/en/docs/newbie-tutorial/FEP_tutorial/)

60. Free Energy Perturbation (FEP) Using QUELO: A Simple Tutorial - QSimulate, 访问时间为 四月 7, 2025， [https://qsimulate.com/documentation/fep_tutorial/fep_tutorial.html](https://qsimulate.com/documentation/fep_tutorial/fep_tutorial.html)

61. Are Free Energy Perturbation (FEP) methods suitable for your project? - MassBio, 访问时间为 四月 7, 2025， [https://www.massbio.org/news/member-news/are-free-energy-perturbation-fep-methods-suitable-for-your-project/](https://www.massbio.org/news/member-news/are-free-energy-perturbation-fep-methods-suitable-for-your-project/)

62. Assessing the stability of free-energy perturbation calculations by performing variations in the method - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC5889414/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5889414/)

63. Exploring the Effectiveness of Binding Free Energy Calculations - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC7032235/](https://pmc.ncbi.nlm.nih.gov/articles/PMC7032235/)

64. Promising results for three GPCR benchmarks using Flare™ FEP for accurate binding affinity calculations in membrane proteins - Cresset Group, 访问时间为 四月 7, 2025， [https://cresset-group.com/about/news/promising-results-three-gpcr-benchmarks/](https://cresset-group.com/about/news/promising-results-three-gpcr-benchmarks/)

65. Free Energy Perturbation Calculations of Mutation Effects on SARS-CoV-2 RBD::ACE2 Binding Affinity - PubMed Central, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC10286572/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10286572/)

66. Limits of Free Energy Computation for Protein-Ligand Interactions - PMC, 访问时间为 四月 7, 2025， [https://pmc.ncbi.nlm.nih.gov/articles/PMC2866028/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2866028/)

67. europlas.com.vn, 访问时间为 四月 7, 2025， [https://europlas.com.vn/en-US/blog-1/fep-plastic-definition-and-applications#:~:text=Applications%20of%20FEP%20plastic,-FEP%20plastic%20gives&text=Some%20outstanding%20internal%20advantages%20of,relatively%20wide%20thermal%20operating%20range.](https://europlas.com.vn/en-US/blog-1/fep-plastic-definition-and-applications#:~:text=Applications%20of%20FEP%20plastic,-FEP%20plastic%20gives&text=Some%20outstanding%20internal%20advantages%20of,relatively%20wide%20thermal%20operating%20range.)

68. FEP Plastic: Definition and applications - EuroPlas, 访问时间为 四月 7, 2025， [https://europlas.com.vn/en-US/blog-1/fep-plastic-definition-and-applications](https://europlas.com.vn/en-US/blog-1/fep-plastic-definition-and-applications)

69. FEP material – Meaning, Benefits, and Application - Holscot, 访问时间为 四月 7, 2025， [https://holscot.com/fep-material-meaning-benefits-application/](https://holscot.com/fep-material-meaning-benefits-application/)

70. FEP Extrusions, Heat Shrink, Roll Covers & Fiber | Zeus, 访问时间为 四月 7, 2025， [https://www.zeusinc.com/materials/fep/](https://www.zeusinc.com/materials/fep/)

71. FEP (Fluorinated Ethylene Propylene) Detailed Properties - Row Inc, 访问时间为 四月 7, 2025， [https://row-inc.com/hubfs/FEP%20(Fluorinated%20Ethylene%20Propylene)%20Detailed%20Properties.pdf](https://row-inc.com/hubfs/FEP%20(Fluorinated%20Ethylene%20Propylene)%20Detailed%20Properties.pdf)

72. Benefits of Fluorinated Ethylene Propylene (FEP) Coatings and Comparison to PTFE, 访问时间为 四月 7, 2025， [https://orioncoat.com/blog/benefits-of-fluorinated-ethylene-propylene-fep-coatings-and-comparison-to-ptfe/](https://orioncoat.com/blog/benefits-of-fluorinated-ethylene-propylene-fep-coatings-and-comparison-to-ptfe/)

73. Computational Method for Determining the Excess Chemical Potential Using Liquid–Vapor Phase Coexistence Simulations - ACS Publications, 访问时间为 四月 7, 2025， [https://pubs.acs.org/doi/10.1021/acs.jpcb.4c07206](https://pubs.acs.org/doi/10.1021/acs.jpcb.4c07206)
