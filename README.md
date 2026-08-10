# Mathematical Modelling of Neuronal Action Potentials for Schizophrenia Classification

### Nonlinear Parameter Estimation · Scientific Computing · Feature Engineering · Bayesian Inference

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-Optimisation-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple.svg)

## Overview

This project develops an interpretable mathematical framework for distinguishing between healthy and schizophrenic neuronal action potentials.

Rather than applying a black-box machine-learning classifier directly to voltage-time data, each action potential is represented using a **piecewise mathematical model** derived from a second-order differential equation. The parameters of this model are estimated independently for each observation using the **Nelder-Mead simplex optimisation algorithm**.

The fitted parameters are then transformed into additional features, analysed using polynomial regression, and used to construct interpretable diagnostic boundaries. Finally, **Bayesian inference** is applied to combine the resulting diagnostic tests and calculate posterior probabilities.

The project also includes a separate externally generated test dataset, allowing the final diagnostic framework to be evaluated on previously unseen observations.

---

# Project Pipeline

```text
Electrophysiological Data
          │
          ▼
Waveform Reconstruction
          │
          ▼
Piecewise Mathematical Model
          │
          ▼
Nelder-Mead Parameter Estimation
          │
          ▼
Derived Feature Extraction
          │
          ▼
Correlation Analysis
          │
          ▼
Polynomial Separation Boundaries
          │
          ▼
Bayesian Classification
          │
          ├──────────────► Internal Error Analysis
          │
          ▼
Independent Test Dataset
          │
          ▼
Independent Parameter Fitting
          │
          ▼
External Probability Evaluation
```

---

# Mathematical Model

The action-potential approximation is based on the differential equation

$$
jv'' + kv' + lv = 0
$$

which produces an oscillatory damped solution under the condition

$$
k^2 < 4jl.
$$

The implementation uses a **piecewise function** consisting of:

1. A baseline resting-potential region
2. A linear depolarisation region
3. A damped oscillatory repolarisation/hyperpolarisation region

The six fitted parameters are:

| Parameter | Description                            |
| --------- | -------------------------------------- |
| (mg)      | Gradient of the depolarisation section |
| (b)       | Initial value of the linear section    |
| (j)       | Restorative-force parameter            |
| (k)       | Resistive/damping parameter            |
| (l)       | Proportionality parameter              |
| `base`    | Baseline membrane potential            |

For the oscillatory region, the implementation defines

$$
Q = \sqrt{\frac{4jl-k^2}{4j^2}}
$$

and

$$
C = mg + \frac{bk}{2j}.
$$

The model therefore takes the form

$$
V(t)=
\begin{cases}
\mathrm{base}, & t < -\frac{b}{mg},\\
mg\*t+b+\mathrm{base}, & -\frac{b}{mg}\leq t<0,\\
e^{-\frac{kt}{2j}}
\left(
b\cos(Qt)+\frac{C}{Q}\sin(Qt)
\right)+\mathrm{base}, & t\geq0.
\end{cases}
$$

The complete implementation can be found in `Claude_curve_fit.py`.

---

# 1. Generate the Primary Action-Potential Dataset

Run:

```bash
python Claude_csv_maker.py
```

This script generates the primary healthy and schizophrenic action-potential datasets.

### Healthy population

Healthy electrophysiological features are obtained from the **Allen Brain Cell Types Database** through its REST API when available.

The implementation uses measured features including:

* resting membrane potential
* action-potential peak voltage
* threshold voltage
* fast after-hyperpolarisation trough
* upstroke/downstroke ratio
* input resistance
* membrane time constant

If the API cannot be reached, the script uses stored reference values derived from the same data source.

### Schizophrenia population

A public REST-queryable database containing equivalent raw action-potential time series specifically from schizophrenia patients was not available for this project.

Instead, the schizophrenia population is parameterised using published electrophysiological measurements from the literature, primarily:

> Ahmad W. et al. (2022), *PNAS*, 119(4), e2109395119.

The script also references supporting electrophysiological literature.

### Important distinction

The resulting waveforms are **reconstructed mathematical waveforms based on real measured electrophysiological features**. They should therefore not be described as raw patient recordings.

The generated files are:

```text
healthy_action_potential.csv
schizophrenia_action_potential.csv
```

The generator also produces provenance information and an initial comparison plot.

---

# 2. Fit the Mathematical Model

Run:

```bash
python Param_values_csv_generator.py
```

This is the computationally intensive stage of the project.

For each action potential, `fit_potential()` from `Claude_curve_fit.py` performs nonlinear optimisation using SciPy's **Nelder-Mead simplex algorithm**.

The optimisation minimises the sum of squared differences between the observed signal and the mathematical approximation:


$$
E( \theta) = \sum_i\left(V_{\mathrm{model}}(t_i;\theta) - V_{\mathrm{observed}}(t_i)\right)^2
$$


where


$$\theta =
(mg,b,j,k,l,\text{base}).$$


The optimisation is subject to numerical validity constraints, including

$$
j>0,\qquad k>0,\qquad l>0
$$

and

$$
k^2<4jl.
$$

Invalid parameter combinations are assigned a large penalty rather than being evaluated directly.

The default optimisation configuration uses:

* Nelder-Mead
* adaptive simplex scaling
* up to 100,000 iterations
* (10e-9) absolute convergence tolerances

The output is:

```text
param_values_both.csv
```

---

# 3. Calculate Derived Features

Run:

```bash
python CSV_editor_final.py
```

This script takes the fitted parameters and calculates two additional quantities.

### Decay constant

$$
\text{decay}=\frac{k}{2j}
$$

### Oscillation frequency parameter

$$
Q=
\sqrt{\frac{4jl-k^2}{4j^2}}
$$

These quantities are appended to the fitted parameter dataset.

Output:

```text
param_values_final.csv
```

The resulting feature set is:

```text
mg
b
j
k
l
base
health
decay
frequency
```

---

# 4. Explore Parameter Relationships

Run:

```bash
python Series_plot.py
```

This script provides tools for extracting individual parameters from `param_values_final.csv` and generating scatter plots.

The purpose is to identify pairs of parameters whose relationships provide strong visual separation between the healthy and schizophrenic populations.

The selected correlation plots are stored in:

```text
Correlation graphs/
```

The primary diagnostic tests use:

* (mg) vs. decay
* (mg) vs. (j)

---

# 5. Construct Mathematical Separation Boundaries

Run:

```bash
python Split_line_finder.py
```

For each selected parameter pair, the script:

### Step 1 — Fit each population

A quadratic polynomial is fitted separately to the healthy and schizophrenic populations using NumPy's polynomial regression:

$$
y=a_2x^2+a_1x+a_0.
$$

### Step 2 — Average the coefficients

The corresponding quadratic, linear and constant coefficients from the two population fits are averaged.

### Step 3 — Construct the separation parabola

A new quadratic curve is created from these averaged coefficients.

The resulting parabola acts as an interpretable boundary between the two populations.

The corresponding plots are stored in:

```text
Correlation graphs/
```

This produces a classification rule based directly on the geometry of the fitted parameter distributions.

---

# 6. Bayesian Classification

Run:

```bash
python Probability_calculator.py
```

The two separation boundaries are treated as diagnostic tests.

They are labelled:

* **Test A:** (mg) vs. decay
* **Test B:** (mg) vs. (j)

The resulting outcomes are:

| Outcome | Test A   | Test B   |
| ------- | -------- | -------- |
| (AB)    | Positive | Positive |
| (AB')   | Positive | Negative |
| (A'B)   | Negative | Positive |
| (A'B')  | Negative | Negative |

For each population, the conditional probability of each outcome is estimated from the sample.

Bayes' theorem is then applied:


$P(S \mid C)$
==========

$$ = \frac{P(C \mid S)P(S)}
{P(C \mid S)P(S)+P(C \mid H)P(H)}$$


where:

* (S) = schizophrenic population
* (H) = healthy population
* (C) = a particular combination of diagnostic-test results

The current implementation uses a population prior of

$$
P(S)=0.01.
$$

This is separate from the approximately balanced composition of the computational study sample.

---

# 7. Internal Error Analysis

Run:

```bash
python Error_finder.py
```

This script evaluates the primary dataset against Test A (the more accurate test) and identifies:

* Type I errors — false positives
* Type II errors — false negatives

The classification rule is based on the (mg)-versus-decay separation boundary.

From this, we found that the method of diagnosis has these many errors:

* Type 1 - 0 (false positives)
* Type 2 - 27 (false negatives)

---

# 8. Diagnose a New Action Potential

Run:

```bash
python Diagnosis.py
```

`Diagnosis.py` provides an interface for fitting a supplied action-potential series using the same mathematical model and optimisation procedure.

The pipeline is:

```text
Input Action Potential
        ↓
Nelder-Mead Optimisation
        ↓
Six Fitted Parameters
        ↓
Derived Features
        ↓
Diagnostic Boundaries
        ↓
Bayesian Probability
```

The final output is an estimated probability of belonging to the schizophrenic population.

## Medical disclaimer

This program is **NOT a medical diagnostic tool**.

Its output should not be interpreted as a clinical diagnosis or medical advice.

The resulting probability is conditional on the mathematical model, simulated/reconstructed data, diagnostic boundaries, prior probability and other assumptions used by the project.

---

# 9. Generate an Independent Test Dataset

Run:

```bash
python Claude_test_csv_maker.py
```

This script generates a separate labelled test dataset containing:

* 100 healthy observations
* 100 schizophrenic observations

for a total of **200 test observations**.

The test populations use the same underlying electrophysiological feature sources as the primary generation process but are generated separately using distinct random seeds.

The resulting dataset is:

```text
labeled_action_potentials.csv
```

This dataset is intended to provide an external evaluation of the classification framework rather than simply measuring performance on the observations used to construct the separation boundaries.

---

# 10. Fit the External Test Dataset

Run:

```bash
python Test_data_processing.py
```

Each test action potential is independently passed through the same `fit_potential()` function used for the primary dataset.

The fitted parameters are written to:

```text
test_param_values.csv
```

This preserves the same mathematical parameter-estimation procedure between the primary and external datasets.

---

# 11. Evaluate the External Test Dataset

Run:

```bash
python Test_probabilities.py
```

The previously constructed Test A and Test B boundaries are applied to the external test observations.

The script evaluates:

* individual test accuracy
* joint test outcomes
* $P(\text{schizophrenic}\mid\text{test outcome})$
* $P(\text{correct diagnosis}\mid\text{test outcome})$

for cases:

$$
AB,\quad AB',\quad A'B,\quad A'B'.
$$

Where A = Positive for test A, and A' = Negative for test A, so A'B is a negative result for A and a positive one for B.

Importantly, the decision boundaries are not refitted to the external test data. They are carried over from the primary analysis.

This provides a more meaningful assessment of whether the mathematical classification framework generalises to previously unseen generated observations.

---

# 12. Validate the Mathematical Approximation

Run:

```bash
python Graph_vs_approx.py
```

This script provides a visual check of the action-potential fitting procedure.

It calculates the population-average action potential and compares it with the mathematical approximation.

It also calculates approximately:

$$
\bar{V}(t)+\sigma(t)
$$

and

$$
\bar{V}(t)-\sigma(t)
$$

to visualise the variation around the mean waveform.

These plots provide a qualitative validation that the mathematical model captures the overall structure of the generated action potentials.

---

# Repository Structure

```text
.
├── Instructions.txt
├── .gitignore
│
├── Claude_csv_maker.py
├── Claude_curve_fit.py
├── Param_values_csv_generator.py
├── CSV_editor_final.py
├── Series_plot.py
├── Split_line_finder.py
├── Probability_calculator.py
├── Error_finder.py
├── Diagnosis.py
│
├── Claude_test_csv_maker.py
├── Test_data_processing.py
├── Test_probabilities.py
│
├── Graph_vs_approx.py
├── useful.py
│
├── healthy_action_potential.csv
├── schizophrenia_action_potential.csv
├── param_values_both.csv
├── param_values_final.csv
│
├── labeled_action_potentials.csv
├── test_param_values.csv
│
├── Correlation graphs/
│   ├── mg vs j.png
│   ├── mg vs j line.png
│   ├── mg vs decay.png
│   └── mg vs decay line.png
│
├── Claude output raw/
└── Claude test output/
```

---

# Execution Order

For a complete reproduction of the computational pipeline, run the scripts in the following order:

```text
1.  Claude_csv_maker.py
2.  Param_values_csv_generator.py
3.  CSV_editor_final.py
4.  Series_plot.py
5.  Split_line_finder.py
6.  Probability_calculator.py
7.  Error_finder.py
8.  Diagnosis.py
9.  Claude_test_csv_maker.py
10. Test_data_processing.py
11. Test_probabilities.py
```

`Graph_vs_approx.py` is an independent visual validation tool and can be run separately.

---

# Supporting Code

## `Claude_curve_fit.py`

Contains the core mathematical model and `fit_potential()` optimisation routine.

This is the central numerical component of the project.

## `useful.py`

A collection of reusable mathematical and data-processing functions developed during the project, including functionality for:

* vector operations
* numerical differentiation
* array manipulation
* plotting
* numerical integration
* matrix transposition
* statistical utilities

This file is an ongoing utility library rather than a component specific to schizophrenia classification.

---

# Data Provenance

The project distinguishes between **measured/source-derived electrophysiological features** and **reconstructed waveforms**.

### Healthy data

Healthy electrophysiological reference values are obtained from the Allen Brain Cell Types Database where possible, with stored reference values available as an offline fallback.

### Schizophrenia data

Schizophrenia-related electrophysiological parameters are derived primarily from published literature, including Ahmad et al. (2022).

The final action-potential traces are mathematically reconstructed from these source-derived features rather than being raw patient voltage recordings.

This distinction is important when interpreting the results.

---

# Reproducibility & Computational Design

The project uses deterministic random seeds for population generation:

* Healthy primary population: seed `42`
* Schizophrenia primary population: seed `99`

The external test-generation process similarly uses controlled random generation.

The parameter-estimation stage is computationally expensive because every individual action potential requires a nonlinear optimisation procedure.

The optimisation is derivative-free, making it suitable for the piecewise model used here, where numerical derivatives can become problematic around transitions between sections of the action-potential approximation.

---

# Technical Concepts Demonstrated

This project combines techniques from several areas of quantitative research:

### Numerical Mathematics

* Nonlinear optimisation
* Nelder-Mead simplex optimisation
* Numerical error minimisation
* Differential-equation-based modelling
* Parameter estimation

### Statistical Modelling

* Polynomial regression
* Feature engineering
* Conditional probability
* Bayesian inference
* Prior/posterior probability
* Classification error analysis

### Scientific Computing

* NumPy
* SciPy
* Pandas
* Matplotlib
* CSV-based data pipelines
* Numerical signal processing

### Quantitative Analysis

* High-dimensional parameter extraction
* Statistical separation of populations
* Decision-boundary construction
* Independent test-set evaluation
* Model validation
* Probabilistic classification

---

# Limitations

Several limitations should be considered when interpreting this work.

### Reconstructed rather than raw waveforms

The action-potential time series are reconstructed from source-derived electrophysiological features. They should not be treated as equivalent to raw patient recordings.

### Different biological sources

The healthy and schizophrenia populations are not necessarily drawn from identical experimental protocols or biological preparations. The schizophrenia parameters are derived from published human iPSC-derived neuronal measurements, while the healthy reference data originate from the Allen Brain Cell Types Database.

### Model assumptions

The classification depends on the ability of the chosen piecewise mathematical model to represent the relevant features of the action potential.

### Generated external dataset

Although the test dataset is independent of the observations used to construct the diagnostic boundaries, it is still generated from the same underlying modelling framework. It therefore represents **external computational validation**, not clinical validation on an independent patient cohort.

### Bayesian prior

The final posterior probabilities depend on the assumed prior probability of schizophrenia. The current implementation uses:

$$
P(S)=0.01.
$$

Changing this prior changes the resulting posterior probabilities.

### Clinical interpretation

The output of this project should not be interpreted as a clinically validated diagnostic probability.

---

# AI-Assisted Development

Files beginning with `Claude` were generated directly using Claude or generated using code produced by Claude.

The remaining code was predominantly written manually, with limited AI assistance primarily for debugging.

The project was developed using Cursor.

AI assistance was therefore used as a development tool, while the mathematical modelling, computational methodology and overall research workflow were developed and assembled as part of the project.

---

# Future Work

Potential extensions include:

* Validation against raw experimental action-potential recordings
* Larger and more demographically controlled datasets
* Confidence intervals for fitted parameters
* Bootstrap analysis
* ROC/AUC analysis
* Sensitivity and specificity analysis
* Formal statistical significance testing
* Robustness analysis under measurement noise
* Comparison of Nelder-Mead against alternative optimisation algorithms
* Parameter-identifiability analysis
* Comparison against conventional statistical and machine-learning classifiers
* Analysis of computational complexity and optimisation runtime

---

# Author Contributions

## Advik Sakhare

**Computational Researcher & Quantitative Developer**

* Developed the mathematical and computational methodology
* Implemented the action-potential fitting pipeline
* Implemented nonlinear parameter estimation using Nelder-Mead
* Developed parameter extraction and feature engineering
* Implemented polynomial separation-boundary construction
* Implemented Bayesian probability calculations
* Developed the external testing pipeline
* Developed computational validation and visualisation tools

## Priya Kaur

**Research Analyst & Scientific Interpretation**

* Conducted background research and literature analysis
* Provided biological and scientific interpretation of the computational results
* Contributed to contextualising the mathematical findings
* Assisted with interpretation of implications and limitations

---

# License / Usage

This repository is made available for personal and educational use.

Please credit the original author when reusing the research, mathematical framework or source code, and do not present the work as your own.

---

# Disclaimer

This project is an **educational and research-oriented computational model**.

It is not clinically validated and must not be used to diagnose schizophrenia or make medical decisions.

The probabilities produced by the software describe the behaviour of the implemented mathematical framework under its stated assumptions. They do not represent clinically validated probabilities of schizophrenia.
