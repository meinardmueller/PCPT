---
title: 'PCPT Notebooks: A Preparation Course for PyTorch with a Focus on Signal Processing'
tags:
  - PyTorch
  - Machine Learning
  - Preparation Course
authors:
  - name: Meinard Müller
    orcid: 0000-0001-6062-7524
    affiliation: 1
  - name: Johannes Zeitler
    orcid: 0000-0003-2171-7679
    affiliation: 1
  - name: Sebastian Strahl
    orcid: 0009-0007-9654-7762
    affiliation: 1
affiliations:
 - name: International Audio Laboratories Erlangen, Germany
   index: 1
date: 05 December 2025
bibliography: paper.bib
---

# Summary
PyTorch [^1] [@PaszkeEtAl19_PyTorch_NeurIPS] is among the most widely used machine-learning frameworks in research and education, supported by a vibrant ecosystem of tutorials, textbooks, and courses, as reflected by curated collections such as *The Incredible PyTorch* [^2]. The Preparation Course for PyTorch (**PCPT**) notebooks, which we present here, are not intended to replace these offerings; rather, they provide a complementary, low-barrier route into PyTorch for learners with basic Python experience. PCPT builds directly on our PCP notebooks  [@MuellerR22_PCP_JOSE] and is conceived as a follow-up: assuming familiarity with core Python concepts, it introduces PyTorch and fundamental machine-learning ideas in a structured and accessible way. Built on the interactive Jupyter framework  [@KluyverEtAl16_Jupyter_Elpub], the notebooks combine executable code, textbook-style explanations, mathematical derivations, and visualizations in a single learning environment.

PCPT is more than a programming course. It revisits and deepens essential concepts from both machine learning and signal processing, including gradient descent, linear regression, discrete convolution, recursion, binary and multiclass classification, loss functions, and common evaluation metrics. A distinctive aspect is the tight connection to digital signal processing (DSP) in the audio domain: machine-learning ideas are consistently related to convolutional and recursive filters, low- and high-pass filtering, smoothing, denoising, and onset detection, with an emphasis on one-dimensional sequence data rather than two-dimensional image examples. PCPT is suitable for self-study and teaching, particularly for students at the end of their Bachelor's studies or the beginning of a Master's program. To keep entry barriers low, the notebooks rely on toy problems and synthetic datasets, focus on principles instead of large-scale data, and run efficiently on standard CPUs without specialized hardware.

[^1]: <https://pytorch.org/>
[^2]: <https://github.com/ritchieng/the-incredible-pytorch>

# Structure, Content, and Access
The PCPT course is organized in a modular fashion and consists of ten units, each corresponding to an individual notebook. The structure and content of the course are illustrated in Figure 1, which is also part of the starting notebook of the PCPT course. Broadly speaking, the first part of PCPT introduces central PyTorch concepts and machine-learning fundamentals, while the second part deepens these notions through applications that are closely connected to DSP. In particular, the PCPT notebooks are designed as a follow-up to our PCP notebooks and are deliberately similar in structure, didactic style, and learning goals, now shifting the focus from Python basics to PyTorch and machine learning.

![Overview of the PCPT notebooks' ten units and their main content.](figure_content_PCPT.pdf)

After a short notebook on how to get started (Unit 1), the course begins with a focused recap of Python classes and object-oriented concepts needed to read and build PyTorch models (Unit 2). It then introduces PyTorch tensors as the core data structure (Unit 3), covering creation, shapes, indexing, reshaping, and basic linear-algebra operations. On this basis, automatic differentiation and gradient-based optimization are explained (Unit 4), enabling students to construct and train neural networks in PyTorch, making use of linear layers, activation functions, loss functions, and optimizers (Unit 5).

With the PyTorch toolkit in place, the course explicitly connects machine-learning concepts to DSP. Discrete one-dimensional convolution and convolutional neural networks (CNNs) are introduced as learnable counterparts of classical DSP filtering, demonstrated through smoothing, denoising, and low-pass or high-pass examples (Unit 6). These ideas are then carried into supervised learning for signals (Unit 7), with applications such as distinguishing noisy sines from square waves and classifying waveforms according to predefined frequency bands, and with evaluation via accuracy, confusion matrices, and out-of-distribution tests; exercises cover binary cross-entropy, softmax with cross-entropy, and frequency band classification using frequency-domain features. The course then turns to training dynamics, focusing on overfitting, generalization, and validation strategies to support reliable model development (Unit 8).

Recursion is revisited by starting from intuitive classical recursive digital filters and impulse responses, then showing how recurrent neural networks (RNNs) realize learned recursive systems in PyTorch (Unit 9). This link is made concrete through an onset-detection case study with peak picking and evaluation by precision, recall, and F-measure, complemented by exercises on recursive filtering and class-imbalance handling. The course concludes with further essential PyTorch techniques (Unit 10), including broadcasting, normalization, regularization, gradient-flow considerations, and custom loss functions. Across all units, PyTorch ideas are grounded in DSP perspectives such as convolutional and recursive filters, spectral and temporal smoothing, denoising, and event detection, with a focus on one-dimensional sequence data suited to audio and signal-processing tasks.

Each unit is organized in a similar fashion. After giving an overview of a unit's structure and learning objectives, the actual content is presented using executable code, textbook-like explanations, mathematical formulas, plots, and visualizations. At the end of each unit, one finds short coding exercises. For self-evaluation, the PCPT notebooks provide for each exercise a sample solution in the form of a Python function, contained in one of the modules of the Python package `libpcpt`. Each such function, which can be easily traced back by its function name, is executed after the respective exercise to produce the results in question.

The PCPT notebooks (including the text, code, and figures) are licensed under the open-source MIT License. Our goal is to continuously improve the PCPT notebooks and provide updates on a regular basis (current version: 1.0.0). To keep the initial hurdles as low as possible and to account for different user needs, the PCPT notebooks can be accessed and executed in different ways.

- The primary source of the PCPT material is the GitHub repository [^3]. It contains executed versions of all notebooks and all source code.

- Additionally, exported HTML versions of all PCPT notebooks can be found at the authors' institutional website [^4]. This static version allows users to access all material, including the explanations, figures, and code examples, by just following the HTML links. 

- To execute the Python code cells, one needs to download the notebooks, install all required Python packages, and start a Jupyter server. The necessary steps are explained in detail in the PCPT notebook on how to get started (Unit 1). Within a Jupyter session, one can follow the IPYNB links for navigating between the units.

- As an alternative to running the notebooks locally, one can also use web-based services such as Google Colab [^5] and Binder [^6]. Explanations on how to run the PCPT notebooks using these services can be found in the notebooks' GitHub repository.

[^3]: <https://github.com/meinardmueller/PCPT>
[^4]: <https://audiolabs-erlangen.de/PCPT>
[^5]: <https://colab.research.google.com>
[^6]: <https://mybinder.org>

# Educational Considerations
The immediate educational contribution of the PCPT notebooks is to provide a structured and concept-focused entry into PyTorch and modern machine learning for students who already possess basic Python programming skills, such as those acquired through our PCP notebooks. The PCPT notebooks are not intended to be a comprehensive PyTorch reference, nor do they aim to cover large-scale training or industrial deployment. Instead, they focus on a compact set of core ideas and deepen them through lightweight and transparent examples that run on standard CPUs. This low-barrier setup allows students to concentrate on understanding principles and developing intuition without being distracted by technical overhead.

A central design goal is continuity with the PCP notebooks. While PCP introduces Python basics and fundamental DSP topics, PCPT builds on this foundation and revisits familiar signal-processing concepts through the lens of learning-based models. For example, discrete convolution is treated both as a classical DSP operation and as a building block of CNNs, while recursion is explored in terms of recursive DSP filters and RNNs. Classical tasks such as smoothing, denoising, and onset detection serve as tangible application scenarios for training and evaluation. In this way, PCPT complements core courses in signal processing and in multimedia, audio, and music processing, and is particularly well suited for students at the end of their Bachelor's studies or the beginning of a Master's program.

Pedagogically, the notebooks alternate systematically between concise theory and hands-on practice. Key mathematical background is summarized in dedicated text boxes that highlight definitions, essential formulas, and take-away messages. These compact textbook-style elements provide a clear theoretical anchor, which is then immediately connected to executable code, small experiments, and guided exercises. This structure allows learners to engage with the material at different depths and supports both self-study and classroom teaching. In our own practice, we use PCPT as a basis for practical courses and research internships in international Master's programs spanning Multimedia Engineering, Computer Science, Data Science, and Artificial Intelligence. In these settings, students work through the units at different speeds, while tutors provide individualized support and encourage collaboration.

The PCPT notebooks are designed to complement existing open educational resources in a targeted way. They extend this programming foundation from basic Python to PyTorch and machine learning. With respect to digital signal processing, they pair naturally with McFee's *Digital Signals Theory* [@McFee23_DST_CRC], which provides an accessible theoretical grounding in discrete-time signals, convolution, and filtering. These concepts are revisited in PCPT from a learning-based perspective. For audio and music processing, PCPT bridges to our FMP notebooks [@MuellerZ19_FMP_ISMIR] and the textbook *Fundamentals of Music Processing* [@Mueller21_FMP_SPRINGER], preparing students to apply machine-learning methods to concrete audio and music tasks. In this broader ecosystem, widely used toolboxes such as `librosa` [@McFeeRLEMBN15_librosa_Python] offer standardized reference implementations that students can adopt once they have acquired the necessary programming and machine-learning foundations. More broadly, the PCPT notebooks support the objectives of open and reproducible research, in line with recommendations for transparent and sustainable software practices in music signal processing  [@McFeeKCSBB19_OpenSourcePractices_IEEE-SPM; @MuellerMK21_LearningMusicSP_IEEE-SPM].

Finally, by working within the Jupyter environment, students become familiar with modern workflows for scientific computing and reproducible experimentation. Together with the course's modular organization, integrated solutions via `libpcpt`, and lightweight execution on standard hardware, this makes PCPT a flexible teaching component that can be embedded into signal-processing curricula, machine-learning modules, or research-oriented lab and internship settings.

# Statement of Need
The PCPT notebooks provide open-source educational material for an accessible entry into PyTorch and machine learning, explicitly positioned as a follow-up to our PCP notebooks. Complementing existing tutorials, PCPT focuses on a small set of foundational concepts in machine learning and reinforces them through interactive coding, clear explanations, and principled exercises. 

A key value of PCPT lies in its close connection to one-dimensional signal processing. Several introductory resources use image-based examples that require two-dimensional data handling and extensive preprocessing. In contrast, PCPT emphasizes 1D sequence data and DSP-motivated tasks, which are often more transparent and technically lightweight, especially in audio-related curricula. 

Organized into ten compact units with integrated solutions and the supporting package `libpcpt`, the notebooks can be used flexibly in preparatory courses, internships, or for self-study. In summary, we hope that PCPT guides students from basic Python knowledge to practical PyTorch proficiency, and that it supports their move toward independent research in machine learning.

# Acknowledgements
The PCPT notebooks are based on material developed by the authors and were refined through valuable feedback from students who used the course in practice. We gratefully acknowledge this feedback, which helped improve the notebooks. This work was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Grant No. 500643750 (MU 2686/15-1). The International Audio Laboratories Erlangen are a joint institution of the Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU) and the Fraunhofer Institute for Integrated Circuits IIS.

# References
