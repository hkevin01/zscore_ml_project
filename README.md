<!-- markdownlint-disable MD033 -->

# Z-Score Normalization in Machine Learning - Reproducible Comparative Study

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.2%2B-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?logo=scikitlearn&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-3.8%2B-11557C)
![status](https://img.shields.io/badge/status-reproducible-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

This repository explains what z-score standardization is, where it belongs in modern ML workflow, and how it performs against min-max, robust scaling, and L2 normalization across multiple model families. The project is built to be practical and auditable: leakage-safe preprocessing, deterministic seeds, saved artifacts, and explicit cross-validation summaries.

The code now supports both a holdout benchmark and a repeated stratified cross-validation benchmark over two datasets, with confidence intervals and result tables saved to the results folder.

> [!IMPORTANT]
> Scaling is fit on training data only. Validation, test, and inference data are transformed using the already-fitted scaler.

---

## What Is a Z-Score and How Does It Work

A z-score, also called a standard score, is a way of expressing how far a single data point sits from the mean of its feature, measured in units of standard deviation. If a feature has values spread widely across different numeric ranges, z-scoring brings every feature onto a common scale centered at zero with unit variance. This does not change the shape of the distribution or remove outliers. It changes the reference frame so that values across different features can be compared and processed as if they were operating at a common magnitude.

The formula is straightforward. For each observed value $x$ in a feature column, subtract the column mean $\mu$ and divide by the column standard deviation $\sigma$. The result $z$ tells you how many standard deviations above or below the mean that value falls.

$$z = \frac{x - \mu}{\sigma}$$

A z-score of 0 means the value is exactly at the mean. A z-score of 1.0 means one standard deviation above the mean. A z-score of -2.0 means two standard deviations below. These values are unitless and dimensionally consistent across all features, which is exactly what distance-based and gradient-based models need to function correctly.

In machine learning, the mean and standard deviation are always estimated from the training data only. They are then frozen and reused to transform validation, test, and production data. This prevents the scaler from seeing future data, which would introduce leakage and inflate evaluation metrics.

> [!NOTE]
> Z-score does not make your data Gaussian. It shifts the center and scales the spread. Skewness, heavy tails, and outliers remain unless you apply additional transforms.

### ML Training Pipeline - Where Z-Score Is Applied Step by Step

The table below shows each step in a standard supervised ML workflow and what z-score is doing or not doing at each stage.

| <sub>#</sub> | <sub>Pipeline Step</sub> | <sub>Z-Score Action</sub> | <sub>How It Works</sub> | <sub>What Happens If Skipped or Wrong</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Collect raw data</sub> | <sub>None yet</sub> | <sub>Inspect units and ranges</sub> | <sub>No impact yet</sub> |
| <sub>2</sub> | <sub>Exploratory analysis</sub> | <sub>None yet</sub> | <sub>Check mean, variance, outliers</sub> | <sub>Missed signals about scaling need</sub> |
| <sub>3</sub> | <sub>Train/test split</sub> | <sub>Split boundary created first</sub> | <sub>Decide split before any fitting</sub> | <sub>If scaler fit before split, leakage occurs</sub> |
| <sub>4</sub> | <sub>Fit scaler on train set</sub> | <sub>Compute mu and sigma per feature</sub> | <sub>StandardScaler.fit(X_train)</sub> | <sub>If fit on all data, test stats leak into train</sub> |
| <sub>5</sub> | <sub>Transform train features</sub> | <sub>Apply z=(x-mu)/sigma to train</sub> | <sub>StandardScaler.transform(X_train)</sub> | <sub>Model trains on raw unscaled features</sub> |
| <sub>6</sub> | <sub>Train model</sub> | <sub>Model receives scaled inputs</sub> | <sub>Optimizer and regularization use scaled space</sub> | <sub>Scale-sensitive models converge poorly</sub> |
| <sub>7</sub> | <sub>Transform val/test features</sub> | <sub>Reuse same mu and sigma from step 4</sub> | <sub>StandardScaler.transform(X_test)</sub> | <sub>If refit on test, evaluation is invalid</sub> |
| <sub>8</sub> | <sub>Evaluate model</sub> | <sub>No new scaler action</sub> | <sub>Metrics computed on scaled test predictions</sub> | <sub>Reported metrics do not reflect real performance</sub> |
| <sub>9</sub> | <sub>Deploy to production</sub> | <sub>Save and reload fitted scaler artifact</sub> | <sub>Apply same transform to live inputs</sub> | <sub>Train-serving skew if scaler not versioned</sub> |
| <sub>10</sub> | <sub>Monitor and retrain</sub> | <sub>Refit scaler on new training window</sub> | <sub>New mu and sigma from updated data</sub> | <sub>Stale scaler causes silent quality decay</sub> |

> [!NOTE]
> This table is the core workflow reference. Steps 3 and 4 are the most commonly mishandled. Split first, fit scaler second, transform third. Never reverse that order.

The table below shows what z-score does to a concrete example set of values so the math is tangible.

| <sub>#</sub> | <sub>Raw Value</sub> | <sub>Feature Mean (mu)</sub> | <sub>Feature Std (sigma)</sub> | <sub>Z-Score Result</sub> | <sub>Interpretation</sub> |
| --- | --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>170 cm height</sub> | <sub>170</sub> | <sub>10</sub> | <sub>0.0</sub> | <sub>Exactly at mean</sub> |
| <sub>2</sub> | <sub>180 cm height</sub> | <sub>170</sub> | <sub>10</sub> | <sub>1.0</sub> | <sub>One std above mean</sub> |
| <sub>3</sub> | <sub>150 cm height</sub> | <sub>170</sub> | <sub>10</sub> | <sub>-2.0</sub> | <sub>Two std below mean</sub> |
| <sub>4</sub> | <sub>65000 income</sub> | <sub>65000</sub> | <sub>18000</sub> | <sub>0.0</sub> | <sub>Exactly at mean</sub> |
| <sub>5</sub> | <sub>101000 income</sub> | <sub>65000</sub> | <sub>18000</sub> | <sub>2.0</sub> | <sub>Two std above mean</sub> |

> [!NOTE]
> After z-scoring, height and income both live on the same numeric scale. A KNN model asking "which sample is nearest?" now treats both features fairly instead of letting income dominate by magnitude.

---

## Table of Contents

- Project Scope
- Quick Scaler Guide
- Architecture
- ML Lifecycle Placement
- Logistic Regression and Z-Score
- Transformers and Attention Context
- Drift and Update Policy
- Experimental Protocol
- Formula and Algorithm Deep Dive
- Visual Explanations with PNG Artifacts
- Results Snapshot
- How to Run
- Publishing and Contribution Files
- References

## Project Scope

The core mission is to compare preprocessing choices, not to chase a single leaderboard metric. Z-score often looks simple, but it directly changes optimization geometry, regularization balance, and neighborhood structure depending on the model.

The repository is also structured for repeatability under practical constraints. You can run one command and regenerate tabular outputs, diagnostics, and plots without manual post-processing. This makes the project useful for learning, peer review, and controlled extensions.

Another design goal is interpretability before complexity. Each new element in the workflow is expected to answer a concrete question, such as whether performance changed because of geometry, because of outlier treatment, or because of model sensitivity. That is why the project adds multiple diagnostics rather than a single aggregate metric.

| <sub>#</sub> | <sub>Topic</sub> | <sub>Covered</sub> | <sub>Not Covered</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>Preprocessing</sub> | <sub>Z-score, min-max, robust, L2</sub> | <sub>Advanced transforms</sub> |
| <sub>2</sub> | <sub>Models</sub> | <sub>LR, KNN, SVM, RF, MLP</sub> | <sub>Full model zoo</sub> |
| <sub>3</sub> | <sub>Validation</sub> | <sub>Holdout + repeated CV</sub> | <sub>Nested CV, Bayesian HPO</sub> |
| <sub>4</sub> | <sub>Artifacts</sub> | <sub>CSV, bar chart, PCA, conf. matrices</sub> | <sub>Interactive dashboards</sub> |

> [!NOTE]
> The table above maps project scope: what is covered and what is intentionally excluded to keep the comparison focused.

| <sub>#</sub> | <sub>Topic</sub> | <sub>Why</sub> |
| --- | --- | --- |
| <sub>1</sub> | <sub>Preprocessing</sub> | <sub>Interpretable comparison</sub> |
| <sub>2</sub> | <sub>Models</sub> | <sub>Show scaling sensitivity</sub> |
| <sub>3</sub> | <sub>Validation</sub> | <sub>Baseline rigor, low complexity</sub> |
| <sub>4</sub> | <sub>Artifacts</sub> | <sub>Portable, versionable outputs</sub> |

> [!NOTE]
> This table defines the project boundary so readers know exactly what claims are supported by the current code.

## Quick Scaler Guide

| <sub>#</sub> | <sub>Scaler</sub> | <sub>Formula</sub> | <sub>Primary Use</sub> | <sub>Common Risk</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Z-Score</sub> | <sub>(x-mu)/sigma</sub> | <sub>General tabular ML default</sub> | <sub>Outlier-sensitive moments</sub> |
| <sub>2</sub> | <sub>Min-Max</sub> | <sub>(x-min)/(max-min)</sub> | <sub>Bounded range inputs</sub> | <sub>Extrema instability</sub> |
| <sub>3</sub> | <sub>Robust</sub> | <sub>(x-median)/IQR</sub> | <sub>Outlier-heavy features</sub> | <sub>Less direct variance interpretation</sub> |
| <sub>4</sub> | <sub>L2 Norm</sub> | <sub>x/norm2(x)</sub> | <sub>Directional similarity vectors</sub> | <sub>Weak on many raw tabular tasks</sub> |

> [!NOTE]
> This table is a fast decision view. It maps each transform to the problem it solves and the failure mode to watch.

The model-side sensitivity also differs substantially by algorithm family.

| <sub>#</sub> | <sub>Model</sub> | <sub>Scaling Sensitivity</sub> | <sub>Reason</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>Logistic Regression</sub> | <sub>High</sub> | <sub>Regularization and gradients depend on magnitude</sub> |
| <sub>2</sub> | <sub>KNN</sub> | <sub>Very High</sub> | <sub>Distance defines prediction</sub> |
| <sub>3</sub> | <sub>SVM RBF</sub> | <sub>High</sub> | <sub>Kernel distance geometry</sub> |
| <sub>4</sub> | <sub>MLP</sub> | <sub>High</sub> | <sub>Optimization conditioning</sub> |
| <sub>5</sub> | <sub>Random Forest</sub> | <sub>Low</sub> | <sub>Splits mostly unit-invariant</sub> |

> [!NOTE]
> The table above shows sensitivity level and the core mechanical reason. See the table below for expected impact.

| <sub>#</sub> | <sub>Model</sub> | <sub>Expected Impact of Good Scaling</sub> |
| --- | --- | --- |
| <sub>1</sub> | <sub>Logistic Regression</sub> | <sub>Better convergence, stable coefficients</sub> |
| <sub>2</sub> | <sub>KNN</sub> | <sub>Strong accuracy shifts likely</sub> |
| <sub>3</sub> | <sub>SVM RBF</sub> | <sub>Better class boundary quality</sub> |
| <sub>4</sub> | <sub>MLP</sub> | <sub>Stability and performance gains</sub> |
| <sub>5</sub> | <sub>Random Forest</sub> | <sub>Usually smaller differences</sub> |

> [!NOTE]
> This table explains why a single scaler can look excellent for one model and weak for another.

## Architecture

```mermaid
flowchart LR
    A[src/zscore_demo.py] --> B[data/synthetic_data.csv]
    C[src/compare_scalers.py] --> D[results/scaler_model_scores.csv]
    C --> E[results/scaler_comparison_table.csv]
    C --> F[results/scaler_model_scores_cv.csv]
    C --> G[results/scaler_dataset_summary_cv.csv]
    C --> H[results/model_scores.png]
    C --> I[results/pca_projections.png]
    C --> J[results/confusion_matrices_*.png]
    K[src/utils.py] --> A
    K --> C
```

> [!NOTE]
> This diagram shows responsibility flow only. It avoids implementation details so project navigation remains simple.

| <sub>#</sub> | <sub>Component</sub> | <sub>Role</sub> | <sub>Outputs</sub> | <sub>Reason for Design</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>src/utils.py</sub> | <sub>Core data, scaling, eval, plotting utilities</sub> | <sub>DataFrames, figures, CSV paths</sub> | <sub>Single source of benchmark logic</sub> |
| <sub>2</sub> | <sub>src/zscore_demo.py</sub> | <sub>Synthetic z-score walkthrough</sub> | <sub>data/synthetic_data.csv</sub> | <sub>Formula grounding before benchmarking</sub> |
| <sub>3</sub> | <sub>src/compare_scalers.py</sub> | <sub>Holdout plus repeated CV benchmark runner</sub> | <sub>All results CSV and plots</sub> | <sub>One command for reproducible artifacts</sub> |
| <sub>4</sub> | <sub>results/</sub> | <sub>Persisted experiment artifacts</sub> | <sub>Versionable benchmark evidence</sub> | <sub>Auditability and sharing</sub> |

> [!NOTE]
> This table describes architectural contracts so future contributors can extend the repo without breaking boundaries.

## ML Lifecycle Placement

Z-score is a preprocessing transform step, not a post-training report step. In supervised learning it should be fit after split creation and before model fitting. At inference it must use the exact training-fitted parameters.

In practice, this means the scaler is part of model definition, not an optional notebook cell. The safest implementation pattern is a single pipeline object that includes both preprocessing and estimator steps. That pattern keeps train, validation, test, and production transformations consistent.

In production systems, the scaler should be versioned as an artifact tied to model version and data window. If a team retrains without updating or validating preprocessing artifacts, silent quality decay can happen even when model code has not changed.

| <sub>#</sub> | <sub>Lifecycle Stage</sub> | <sub>Z-Score Action</sub> | <sub>Allowed Operation</sub> | <sub>Failure if Wrong</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Data split</sub> | <sub>Define train/val/test boundaries</sub> | <sub>No fitting yet</sub> | <sub>Leakage risk if fit before split</sub> |
| <sub>2</sub> | <sub>Train preprocessing</sub> | <sub>Fit mu and sigma on train only</sub> | <sub>fit_transform train</sub> | <sub>Invalid benchmark if fit on all data</sub> |
| <sub>3</sub> | <sub>Validation/Test</sub> | <sub>Reuse trained scaler</sub> | <sub>transform only</sub> | <sub>Optimistic metrics if refit</sub> |
| <sub>4</sub> | <sub>Serving</sub> | <sub>Apply same scaler artifact</sub> | <sub>transform only</sub> | <sub>Train-serving skew</sub> |
| <sub>5</sub> | <sub>Retraining</sub> | <sub>Refit scaler on new train window</sub> | <sub>New fit after governance checks</sub> | <sub>Drift mismatch if stale scaler kept</sub> |

> [!NOTE]
> This table answers where z-score sits in ML workflow and when fitting is allowed.

## Logistic Regression and Z-Score

Logistic regression is usually one of the clearest beneficiaries of z-score on numeric tabular features. Scaling improves optimization conditioning and makes L1/L2 penalties operate on more comparable coefficient scales.

Without scaling, one feature with a larger numerical range can dominate gradient updates and regularization behavior. This often causes slower convergence, unstable coefficient interpretation, and suboptimal calibration in probability outputs. Standardization makes optimizer steps more balanced across coordinates.

A second practical benefit is comparability during model inspection. When features are standardized, coefficient magnitudes reflect predictive influence more consistently under regularization constraints. This does not make interpretation trivial, but it reduces unit-driven distortions.

| <sub>#</sub> | <sub>Question</sub> | <sub>Answer</sub> | <sub>Practical Rule</sub> | <sub>Why</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>When to scale for LR</sub> | <sub>Before model fitting</sub> | <sub>Use Pipeline with scaler then LR</sub> | <sub>Prevents leakage and keeps flow consistent</sub> |
| <sub>2</sub> | <sub>During training updates</sub> | <sub>Scaler params fixed per train run</sub> | <sub>Do not refit on validation batches</sub> | <sub>Stable optimization baseline</sub> |
| <sub>3</sub> | <sub>After training</sub> | <sub>No new fitting for evaluation</sub> | <sub>Transform-only on val/test</sub> | <sub>Metric validity</sub> |
| <sub>4</sub> | <sub>Under drift</sub> | <sub>Refit during retraining cycle</sub> | <sub>Version scaler with model</sub> | <sub>Distribution alignment</sub> |

> [!NOTE]
> This table directly addresses LR timing and operations for fit versus transform.

## Transformers and Attention Context

Transformers and z-score normalization operate at completely different layers of a pipeline, and understanding that distinction is critical for avoiding both over-engineering and under-engineering your preprocessing stack. In classical ML, z-score lives at the input data layer, before any model receives features. In transformers, normalization lives inside the model architecture itself, after token embeddings are projected into hidden representations.

The transformer architecture introduced in "Attention Is All You Need" by Vaswani et al. (2017) uses multi-head self-attention to learn relationships between tokens. Each attention head projects input embeddings into query, key, and value subspaces, computes scaled dot-product attention weights, and combines values according to those weights. The scaling factor applied in attention is $1/\sqrt{d_k}$, where $d_k$ is the key dimension. This is a fixed structural stabilizer, not a learned data-driven normalization in the z-score sense.

LayerNorm is applied after attention and feed-forward blocks in most transformer variants. It normalizes each sample across the hidden-state dimension, not across the dataset. This is fundamentally different from z-score, which normalizes each feature column across the training sample population. LayerNorm operates at the representation level per token per layer. Z-score operates at the raw input level per feature column across all training samples.

### When Z-Score is Used in Transformer Contexts

Z-score becomes relevant in transformer pipelines when raw numeric tabular features are used as input before the first embedding or projection layer. If your transformer model ingests 30 numeric tabular columns representing clinical measurements, financial indicators, or sensor readings, those columns still have heterogeneous units and ranges. The model's first linear projection layers do not inherently cancel that scale difference. In that setting, external z-score preprocessing can stabilize early-layer optimization and reduce dependence on learning-rate scheduling to compensate for input scale variation.

Another scenario where z-score matters is preprocessing for hybrid architectures that mix text tokens with structured numeric features. If numeric features are concatenated or added to token embeddings before attention, scale misalignment between the two modalities can distort early attention distributions. Z-scoring the numeric inputs before fusion is a common engineering practice in such architectures.

> [!NOTE]
> Z-score is most relevant in transformers when raw numeric features enter the model without a learned embedding bottleneck that would absorb scale differences.

### When Z-Score is Not the Right Tool in Transformer Contexts

Z-score is generally not the appropriate tool for text token embeddings. Word embeddings from pretrained models like BERT or GPT are already organized in a learned semantic space. Applying z-score standardization across token embedding dimensions would destroy the geometric structure that makes similarity and attention meaningful. The representations would lose their direction-based semantics because z-score imposes a global column-wise normalization that is blind to semantic neighborhood structure.

Z-score is also not needed when your transformer architecture already includes BatchNorm or LayerNorm at every residual block. Those internal layers handle representational normalization as part of the forward pass. Adding redundant external z-scoring on top of a well-architected transformer often adds noise rather than benefit.

Finally, z-score is ineffective for sparse or categorical inputs that are one-hot encoded or positionally embedded before entry into the model. The statistical properties that z-score depends on, finite mean and variance over a continuous distribution, do not apply in the same way to discrete indicator vectors.

| <sub>#</sub> | <sub>Context</sub> | <sub>Z-Score Used?</sub> | <sub>Reason</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>NLP text tokens</sub> | <sub>No</sub> | <sub>Embeddings already in learned semantic space</sub> |
| <sub>2</sub> | <sub>Tabular numeric inputs to transformer</sub> | <sub>Usually yes</sub> | <sub>Input scale varies before projection layer</sub> |
| <sub>3</sub> | <sub>Hybrid text plus numeric inputs</sub> | <sub>Yes for numeric part</sub> | <sub>Prevents scale skew in multi-modal fusion</sub> |
| <sub>4</sub> | <sub>Internal hidden states</sub> | <sub>No</sub> | <sub>LayerNorm handles this internally</sub> |
| <sub>5</sub> | <sub>One-hot or sparse inputs</sub> | <sub>No</sub> | <sub>Discrete vectors lack continuous distribution</sub> |

> [!NOTE]
> This table separates transformer contexts where external z-score applies from those where it would be counterproductive.

| <sub>#</sub> | <sub>Normalization Type</sub> | <sub>Where Applied</sub> | <sub>What It Normalizes</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>Z-Score (StandardScaler)</sub> | <sub>Input pipeline, before model</sub> | <sub>Feature columns across training samples</sub> |
| <sub>2</sub> | <sub>LayerNorm</sub> | <sub>Inside transformer blocks</sub> | <sub>Hidden states per token per layer</sub> |
| <sub>3</sub> | <sub>BatchNorm</sub> | <sub>Inside deep learning layers</sub> | <sub>Activations per channel across batch</sub> |
| <sub>4</sub> | <sub>Attention scaling (1/sqrt dk)</sub> | <sub>Inside attention head</sub> | <sub>Dot-product scores before softmax</sub> |
| <sub>5</sub> | <sub>RMSNorm</sub> | <sub>Inside transformer blocks</sub> | <sub>Root-mean-square of hidden states per token</sub> |

> [!NOTE]
> This table shows how different normalization mechanisms operate at different pipeline positions.

## Drift and Update Policy

Z-score normalization is grounded in a fundamental assumption: the mean and standard deviation estimated from training data remain good approximations of the true distribution at serving time. When that assumption breaks down because of distribution shift, the scaler becomes a silent source of error. The model continues to receive transformed inputs, but those inputs are now centered and scaled around statistics that no longer match the current data distribution.

Drift is not always obvious in production metrics. If the feature distribution shifts gradually, the model's prediction quality may degrade slowly rather than suddenly. A scaler fitted months ago on one population may still produce numerically reasonable outputs, but the standardized values may no longer carry the same geometric meaning they had during training. This type of silent drift is one of the hardest problems in applied ML operations.

The z-score scaler is particularly exposed to mean drift and variance drift. Mean shift in a feature moves the center of the transformed distribution away from zero, which is where the model learned to operate. Variance change compresses or expands the effective input range, altering the scale of optimizer gradients if the model is later fine-tuned, and distorting distance and boundary calculations for deployed models.

> [!IMPORTANT]
> Drift does not always trigger model errors immediately. Silent preprocessing drift is common in deployed systems and is easily missed without explicit artifact monitoring.

### How Drift Connects to Transformers and Attention

In transformer-based systems, drift affects both external preprocessing and internal normalization layers differently. External z-score drift affects the raw input distribution before any learned projection. Internal LayerNorm is more adaptive because it normalizes per-sample per-layer dynamically during the forward pass, so it partially compensates for input-level distributional changes without any explicit retraining.

This architectural difference means that transformer models can sometimes tolerate moderate input drift better than classical models like logistic regression or KNN, where the scaler's statistics are the only normalization mechanism. However, this tolerance is not unlimited. Severe drift still degrades transformer attention quality because the initial projection layers feed corrupted statistics into the attention mechanism.

For models that mix classical preprocessing with transformer components, the weakest link is usually the external scaler. An internal LayerNorm cannot correct for input features that arrive already distorted by an outdated z-score transform. This is why versioning the scaler artifact together with the model weights is a critical practice in any mixed-pipeline deployment.

| <sub>#</sub> | <sub>Drift Signal</sub> | <sub>Likely Cause</sub> | <sub>Action</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>Feature mean shift</sub> | <sub>Covariate drift</sub> | <sub>Retrain and refit scaler</sub> |
| <sub>2</sub> | <sub>Variance regime change</sub> | <sub>Source or sensor change</sub> | <sub>Refit scaler and rerun CI benchmark</sub> |
| <sub>3</sub> | <sub>Serving metric drop</sub> | <sub>Preprocessing mismatch</sub> | <sub>Audit scaler artifact parity</sub> |
| <sub>4</sub> | <sub>Class balance shift</sub> | <sub>Population change</sub> | <sub>Re-split stratified sets and retrain</sub> |

> [!NOTE]
> This table maps observable drift signals to probable causes and corrective actions.

The table below compares how the same drift event affects z-score preprocessing and transformer-internal LayerNorm differently.

| <sub>#</sub> | <sub>Drift Type</sub> | <sub>Z-Score Impact</sub> | <sub>Transformer LayerNorm Response</sub> |
| --- | --- | --- | --- |
| <sub>1</sub> | <sub>Mean shift</sub> | <sub>Center moves away from zero</sub> | <sub>Partial compensation at hidden layer</sub> |
| <sub>2</sub> | <sub>Variance change</sub> | <sub>Input range expands or compresses</sub> | <sub>Some compensation but corrupted input persists</sub> |
| <sub>3</sub> | <sub>Tail extension</sub> | <sub>New extrema fall past scaler training range</sub> | <sub>Attention scores may shift, model still runs</sub> |
| <sub>4</sub> | <sub>New feature categories</sub> | <sub>No learned statistics for new values</sub> | <sub>No internal correction possible</sub> |

> [!NOTE]
> This table shows why z-score drift and LayerNorm behavior diverge under the same distribution shift event.

## Experimental Protocol

The repository executes two evaluation tracks: a holdout track for diagnostics and a repeated stratified CV track for uncertainty-aware comparison.

The holdout track exists because diagnostic figures require a fixed evaluation split for direct visual comparison. Confusion matrices and PCA scatter structure are easier to interpret when all scaler-model pairs are measured on the same held-out partition.

The repeated CV track exists because single-split estimates can be noisy and sensitive to partition luck. Repeating stratified folds across multiple shuffles provides a stronger estimate of central tendency and spread, and enables confidence interval reporting for each scaler-model-dataset combination.

| <sub>#</sub> | <sub>Track</sub> | <sub>Protocol</sub> | <sub>Datasets</sub> | <sub>Main Output</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Holdout</sub> | <sub>Single stratified split</sub> | <sub>breast_cancer</sub> | <sub>per-model scores plus plots</sub> |
| <sub>2</sub> | <sub>Repeated CV</sub> | <sub>5 folds x 5 repeats, stratified</sub> | <sub>breast_cancer and wine</sub> | <sub>mean, std, 95 percent CI</sub> |

> [!NOTE]
> This table states exactly what is computed so reported claims stay aligned with actual code behavior.

The CI formula used in the project is shown below.

$$
\bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{n}}
$$

> [!NOTE]
> Here, $\bar{x}$ is mean CV accuracy, $s$ is sample standard deviation across splits, and $n$ is number of split scores.

## Formula and Algorithm Deep Dive

The formulas below are the operational math behind every preprocessing step in this repository. They are included to make each design choice explicit and to reduce ambiguity when extending the benchmark with additional models or datasets.

The algorithm table links each model family to the optimization or geometry mechanism that scaling influences most. This helps explain why the same scaler can improve one model while barely affecting another.

A practical reading strategy is to combine this section with the visual section below. Use the formulas to interpret transformation intent, then verify behavior using the PNG artifacts and CV summaries.

| <sub>#</sub> | <sub>Method</sub> | <sub>Formula</sub> | <sub>Key Effect</sub> | <sub>Primary Limitation</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Z-Score</sub> | <sub>z=(x-mu)/sigma</sub> | <sub>Centers and variance-normalizes feature columns</sub> | <sub>Moment estimates sensitive to outliers</sub> |
| <sub>2</sub> | <sub>Min-Max</sub> | <sub>x'=(x-min)/(max-min)</sub> | <sub>Bounds each feature into fixed numeric range</sub> | <sub>Extrema can compress inlier spread</sub> |
| <sub>3</sub> | <sub>Robust</sub> | <sub>x'=(x-median)/IQR</sub> | <sub>Reduces outlier leverage using robust statistics</sub> | <sub>Less direct variance interpretation</sub> |
| <sub>4</sub> | <sub>L2 Normalizer</sub> | <sub>x'=x/norm2(x)</sub> | <sub>Normalizes row length for directional comparison</sub> | <sub>Can hurt feature-wise tabular tasks</sub> |
| <sub>5</sub> | <sub>PCA Projection</sub> | <sub>Z=XW</sub> | <sub>Projects to principal variance directions</sub> | <sub>Information loss in low-dimensional view</sub> |

> [!NOTE]
> This formula table maps transformation math to practical effect and failure risk.

| <sub>#</sub> | <sub>Algorithm</sub> | <sub>Core Objective</sub> | <sub>Scaling Interaction</sub> | <sub>Expected Behavior</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Logistic Regression</sub> | <sub>Minimize regularized log loss</sub> | <sub>Gradient and penalty terms are scale-sensitive</sub> | <sub>Strong gains from z-score or robust scaling</sub> |
| <sub>2</sub> | <sub>KNN</sub> | <sub>Distance-based voting</sub> | <sub>Nearest-neighbor geometry depends on scale</sub> | <sub>Very high sensitivity to preprocessing</sub> |
| <sub>3</sub> | <sub>SVM RBF</sub> | <sub>Margin with kernel distance</sub> | <sub>Kernel radius meaning shifts with scale</sub> | <sub>High sensitivity to preprocessing</sub> |
| <sub>4</sub> | <sub>MLP</sub> | <sub>Optimize non-convex loss by gradient descent</sub> | <sub>Input scale affects conditioning and convergence</sub> | <sub>Performance and stability shifts likely</sub> |
| <sub>5</sub> | <sub>Random Forest</sub> | <sub>Ensemble impurity reduction</sub> | <sub>Split thresholds mostly unit-invariant</sub> | <sub>Usually lower sensitivity than distance models</sub> |

> [!NOTE]
> This algorithm table explains model-specific responses to identical preprocessing choices.

## Visual Explanations with PNG Artifacts

The visual outputs below are generated directly by the benchmark pipeline and stored in the results folder. They are embedded here so readers can connect formulas and tables to concrete model behavior.

Each visualization includes a summary with multiple paragraphs so interpretation is explicit rather than implied. The goal is not only to display figures, but also to explain what each figure means for model selection and preprocessing strategy.

### Visualization 1 - Model Accuracy by Scaling Strategy

![Model Accuracy by Scaling Strategy](results/model_scores.png)

The grouped bar chart provides a fast comparative view across all scaler-model pairs in the holdout run. You can quickly see that the relative ranking pattern differs by model family, which is exactly what scaling theory predicts.

Distance-sensitive and gradient-sensitive learners respond more strongly to preprocessing variation than tree ensembles. This supports the engineering rule that preprocessing should be selected jointly with algorithm class rather than as a global default.

The chart is most useful as a first-pass triage view. It tells you where to focus deeper analysis, then confusion matrices and CV intervals explain whether gains are stable and where error tradeoffs moved.

> [!NOTE]
> This figure summarizes holdout ranking structure and helps prioritize deeper diagnostics.

### Visualization 2 - PCA Projection by Scaler

![PCA Projection by Scaler](results/pca_projections.png)

The PCA figure shows how each scaler changes two-dimensional variance structure after transformation. Even when final accuracy values are close, class cloud geometry can shift enough to affect boundary shape and confidence distribution.

This plot is diagnostic rather than decisive. A cleaner class separation in two components does not guarantee a better classifier, but it often signals whether preprocessing is improving representation geometry for downstream learners.

A useful practice is to compare PCA patterns with CV confidence intervals. If geometry looks cleaner and intervals remain consistently higher across folds, the preprocessing choice is more likely to generalize.

> [!NOTE]
> This figure visualizes feature-space geometry after scaling and supports interpretation of downstream metric changes.

### Visualization 3 - Confusion Matrices for Z-Score

![Confusion Matrices for Z-Score](results/confusion_matrices_zscore.png)

This confusion matrix panel shows error decomposition for each model when z-score is used. The key value is not only total accuracy, but where false positives and false negatives concentrate.

Model-level error asymmetry is important in applied settings. Two models can have similar accuracy while exhibiting very different false-negative profiles, which can matter significantly in risk-sensitive domains.

Use this panel together with per-model CV statistics to avoid overinterpreting a single split. If confusion structure and repeated-CV mean move in the same direction, selection confidence increases.

> [!NOTE]
> This figure decomposes z-score performance into class-specific error patterns.

### Visualization 4 - Confusion Matrices for Min-Max

![Confusion Matrices for Min-Max](results/confusion_matrices_minmax.png)

The min-max panel highlights how bounded scaling can alter class boundary behavior relative to z-score. In some models this can improve margin behavior, while in others it may compress informative spread.

This panel is especially informative when comparing MLP and KNN behavior. Both can benefit from controlled numeric ranges, but the exact gain depends on feature distribution shape and outlier placement.

Interpret this panel alongside the formula table. Min-max relies on extrema, so unstable min or max values can explain shifts in confusion structure even when central tendency appears similar.

> [!NOTE]
> This figure helps evaluate whether bounded scaling improves or distorts class discrimination for each model.

### Visualization 5 - Confusion Matrices for Robust Scaling

![Confusion Matrices for Robust Scaling](results/confusion_matrices_robust.png)

Robust scaling uses median and IQR, so this panel is useful when outliers are plausible contributors to metric instability. In this benchmark it often remains highly competitive across model families.

The most important interpretation is error distribution stability. If robust scaling reduces volatility in confusion structure across models, it can be a safer default in noisy feature regimes.

This panel should not be treated as proof that robust scaling is always best. It is evidence that robust moments can improve resilience under certain feature distributions and should be benchmarked, not assumed.

> [!NOTE]
> This figure evaluates outlier-robust preprocessing effects on class-level errors.

### Visualization 6 - Confusion Matrices for L2 Normalization

![Confusion Matrices for L2 Normalization](results/confusion_matrices_l2_norm.png)

The L2 panel shows the behavior of row-wise normalization in a feature-oriented tabular setting. This transformation is often strong for directional similarity tasks, but it can be misaligned with tabular feature semantics.

In this benchmark, L2 normalization is generally weaker on mean accuracy and often shows less favorable confusion structures for several models. That outcome is consistent with the objective mismatch between row normalization and feature-scale balancing.

This visualization is still valuable because it prevents overgeneralization. A method performing worse here is not universally poor, it is simply less aligned with this dataset type and objective structure.

> [!NOTE]
> This figure illustrates why directional normalization may underperform in many raw tabular classification tasks.

## Results Snapshot

Holdout summary from current run:

| <sub>#</sub> | <sub>Scaler</sub> | <sub>Mean Accuracy</sub> | <sub>Rank</sub> | <sub>Interpretation</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>robust</sub> | <sub>0.9684</sub> | <sub>1</sub> | <sub>Strong with mild outlier robustness</sub> |
| <sub>2</sub> | <sub>zscore</sub> | <sub>0.9667</sub> | <sub>2</sub> | <sub>Near-best general baseline</sub> |
| <sub>3</sub> | <sub>minmax</sub> | <sub>0.9632</sub> | <sub>3</sub> | <sub>Competitive but slightly lower average</sub> |
| <sub>4</sub> | <sub>l2_norm</sub> | <sub>0.9018</sub> | <sub>4</sub> | <sub>Weaker for this tabular task setup</sub> |

> [!NOTE]
> This table is the holdout summary only. Use the CV summary for stronger uncertainty-aware comparison.

Repeated CV cross-dataset scaler summary from current run:

| <sub>#</sub> | <sub>Dataset</sub> | <sub>Best Scaler</sub> | <sub>Best Mean Across Models</sub> | <sub>Lowest Scaler</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>breast_cancer</sub> | <sub>zscore</sub> | <sub>0.9705</sub> | <sub>l2_norm at 0.8944</sub> |
| <sub>2</sub> | <sub>wine</sub> | <sub>zscore</sub> | <sub>0.9804</sub> | <sub>l2_norm at 0.7826</sub> |

> [!NOTE]
> This table shows that z-score is the top mean scorer across both datasets in repeated CV summary, while L2 normalization is lowest in this benchmark configuration.

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/zscore_demo.py
python src/compare_scalers.py
```

| <sub>#</sub> | <sub>Command</sub> | <sub>Primary Effect</sub> | <sub>Expected Output Files</sub> | <sub>Run Time Profile</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>python src/zscore_demo.py</sub> | <sub>Generate synthetic z-score demo data</sub> | <sub>data/synthetic_data.csv</sub> | <sub>Short</sub> |
| <sub>2</sub> | <sub>python src/compare_scalers.py</sub> | <sub>Run holdout plus repeated CV benchmark</sub> | <sub>results/*.csv and results/*.png</sub> | <sub>Moderate</sub> |

> [!NOTE]
> Run from project root so relative output paths match repository structure.

## Publishing and Contribution Files

| <sub>#</sub> | <sub>File</sub> | <sub>Purpose</sub> | <sub>Who Uses It</sub> | <sub>Status</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>LICENSE</sub> | <sub>Legal reuse terms</sub> | <sub>All users</sub> | <sub>Added</sub> |
| <sub>2</sub> | <sub>CONTRIBUTING.md</sub> | <sub>Contribution workflow</sub> | <sub>Contributors</sub> | <sub>Added</sub> |
| <sub>3</sub> | <sub>.github/ISSUE_TEMPLATE/bug_report.md</sub> | <sub>Bug intake template</sub> | <sub>Issue reporters</sub> | <sub>Added</sub> |
| <sub>4</sub> | <sub>.github/ISSUE_TEMPLATE/feature_request.md</sub> | <sub>Feature intake template</sub> | <sub>Issue reporters</sub> | <sub>Added</sub> |
| <sub>5</sub> | <sub>.github/pull_request_template.md</sub> | <sub>PR quality checklist</sub> | <sub>Contributors and reviewers</sub> | <sub>Added</sub> |
| <sub>6</sub> | <sub>.github/RELEASE_CHECKLIST.md</sub> | <sub>Release gate checks</sub> | <sub>Maintainers</sub> | <sub>Added</sub> |

> [!NOTE]
> This table confirms repository publishing hygiene is in place for public collaboration.

## References

| <sub>#</sub> | <sub>Type</sub> | <sub>Source</sub> | <sub>Why It Is Included</sub> | <sub>Link</sub> |
| --- | --- | --- | --- | --- |
| <sub>1</sub> | <sub>Documentation</sub> | <sub>scikit-learn preprocessing guide</sub> | <sub>Transformer definitions and API semantics</sub> | <sub>https://scikit-learn.org/stable/modules/preprocessing.html</sub> |
| <sub>2</sub> | <sub>Documentation</sub> | <sub>scikit-learn scaling example</sub> | <sub>Practical effect of scaling on models and PCA</sub> | <sub>https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html</sub> |
| <sub>3</sub> | <sub>arXiv</sub> | <sub>Ioffe and Szegedy 2015</sub> | <sub>Normalization and optimization context</sub> | <sub>https://arxiv.org/abs/1502.03167</sub> |
| <sub>4</sub> | <sub>arXiv</sub> | <sub>Santurkar et al 2018</sub> | <sub>Why normalization can help optimization</sub> | <sub>https://arxiv.org/abs/1805.11604</sub> |
| <sub>5</sub> | <sub>arXiv</sub> | <sub>Kapoor and Narayanan 2022</sub> | <sub>Leakage and reproducibility context</sub> | <sub>https://arxiv.org/abs/2207.07048</sub> |
| <sub>6</sub> | <sub>Article</sub> | <sub>Singh and Singh 2020</sub> | <sub>Comparative normalization impact evidence</sub> | <sub>https://www.sciencedirect.com/science/article/pii/S1568494619302947</sub> |

> [!NOTE]
> These references anchor both implementation choices and methodological claims.

## Final Accuracy Statement

This README is aligned with current code and outputs in this repository, including MLP, PCA projections, confusion matrices, repeated stratified CV, confidence interval tables, second dataset comparison, and GitHub publishing files.

If behavior changes in source scripts, update this README in the same change set to preserve truthfulness and reproducibility.
