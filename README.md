# Preparation Course PyTorch Notebooks 

This repository contains the PCPT Notebooks, which provide open-source learning materials for a gentle introduction to [PyTorch](https://docs.pytorch.org/docs/stable/index.html), one of today's most widely used machine learning frameworks. Built on the interactive Jupyter platform, the notebooks combine code, explanations, formulas, and visualizations in a single source. They are designed for learners with basic Python skills and focus on exploring key ideas of machine learning in a clear and accessible way.
The PCPT notebooks continue the material of the [PCP notebooks](https://github.com/meinardmueller/PCP), which provide foundational Python programming knowledge. Together, the PCP and PCPT notebooks provide a good foundation for doing a research internship in the fields of signal processing, machine learning, and deep learning, as required by FAU master's study programmes such as [Communications and Multimedia Engineering (CME)](https://www.cme.studium.fau.de/), [Data Science](https://www.math-datascience.nat.fau.de/im-studium/masterstudiengaenge/master-data-science/), and [Artificial Intelligence (AI)](https://meinstudium.fau.de/studiengang/artificial-intelligence-msc/).
The notebooks cover fundamental concepts including gradient descent, linear regression, convolution, recursion, binary and multiclass classification, cross-entropy losses, and evaluation metrics. Machine learning topics are linked to digital signal processing, such as low- and high-pass filtering, denoising, and onset detection. Unlike resources focusing on 2D image data, the PCPT notebooks emphasize 1D sequence data to reduce technical overhead.
The PCPT notebooks are designed with low entry barriers: they use toy examples and synthesized datasets, prioritize conceptual understanding over large-scale computation, and run on standard CPUs without specialized hardware.
In summary, the PCPT notebooks aim to help students transition naturally from learning PyTorch fundamentals to developing the skills needed for independent research in machine learning. The notebooks are freely accessible under the [MIT License](https://opensource.org/licenses/MIT).

If a static view of the PCPT notebooks is enough for you, the [exported HTML versions](https://www.audiolabs-erlangen.de/PCPT) can be used right away without any installation. All material including the explanations and the figures can be accessed by just following the **HTML links**. If you want to **execute** the Python code cells, you have to clone/download the notebooks (along with the `libpcpt` library), create an environment, and start a Jupyter server. You then need to follow the **IPYNB links** within the Jupyter session. The necessary steps are explained in detail in the [PCPT notebook on how to get started](https://www.audiolabs-erlangen.de/resources/MIR/PCPT/PCPT_01_getstarted.html).

## Reference

If you use the [PCPT Notebooks](https://www.audiolabs-erlangen.de/PCPT) in your teaching or research, please consider mentioning the following reference.

```bibtex
@article{MuellerZS25_PCPT,
    author    = {Meinard M{\"u}ller and Johannes Zeitler and Sebastian Strahl},
    title     = {{PCPT} Notebooks: {A} Preparation Course for {P}y{T}orch},
    journal   = {},
    volume    = {},
    number    = {},
    year      = {2025},
    pages     = {},
    doi       = {},
    url       = {https://www.audiolabs-erlangen.de/PCPT},
}
```

## Installing Local Environment for Executing PCPT Notebooks
This is the preferred and tested variant for using the PCPT notebooks.

```
conda env create -f environment.yml
conda activate PCPT
jupyter notebook
```

## Using Web-Based Services for Executing PCPT Notebooks

### Google colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meinardmueller/PCPT/blob/master/PCPT.ipynb)

The PCPT notebooks may be executed using [Google colab](https://colab.research.google.com/). However, this needs some preparation. First, you need to be logged in with a Google account. The starting notebook can be accessed via:

https://colab.research.google.com/github/meinardmueller/PCPT/blob/master/PCPT.ipynb

For the other notebooks, clone the PCPT repository to get access to data and the functions in `libpcpt`. To this end, for each colab session, include and execute a code cell at the beginning of the notebook containing the following lines:

```
%%bash
git clone https://github.com/meinardmueller/PCPT.git PCPT_temp
mv PCPT_temp/* .
rm -rd PCPT_temp
```

### Binder
[![Open In Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meinardmueller/PCPT/master)

One can also use [Binder](https://mybinder.org/) to execute the PCPT notebooks. This clones the repository and automatically creates a conda environment. This may take several (maybe even up to ten) minutes when starting binder.

https://mybinder.org/v2/gh/meinardmueller/PCPT/master

## Contributing
We are happy for suggestions and contributions. However, to facilitate the synchronization, we would be grateful for either directly contacting us via email (meinard.mueller@audiolabs-erlangen.de) or for creating [an issue](https://github.com/meinardmueller/PCPT/issues) in our GitHub repository. Please do not submit a pull request without prior consultation with us.

## Acknowledgements
We want to thank the various people who have contributed to the design, implementation, and code examples of the notebooks. We mention the main contributors in alphabetical order: Meinard Müller, Sebastian Strahl, Johannes Zeitler. The [International Audio Laboratories Erlangen](https://www.audiolabs-erlangen.de/) are a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
