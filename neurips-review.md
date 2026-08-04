# Neurips 2026 Evaluations and Datasets Track Reviews

## Intructions to Reviewers

**Overall**: Please provide an "overall score" for this submission. Choices:

    - 6: Strong Accept: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
    - 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
    - 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
    - 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
    - 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
    - 1: Strong Reject: For instance, a paper with well-known results or unaddressed ethical considerations

**Confidence**:  Please provide a "confidence score" for your assessment of this submission to indicate how confident you are in your evaluation.  Choices

    - 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
    - 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
    - 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    - 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
    - 1: Your assessment is an educated guess. The submission is not in your area or the submission was difficult to understand. Math/other details were not carefully checked.

## Paper Summary:

**Title**: Can Time Series Foundation Models Forecast Nocturnal Hypoglycemia? An Evaluation Study in Type 1 Diabetes
**Keywords**: time series forecasting, foundation models, benchmarking, continuous glucose monitoring, nocturnal hypoglycemia, probabilistic forecasting, clinical AI, evaluation metrics
**TL;DR**: We benchmark 12 models (7 TSFMs, 5 deep learning) on 8-hour nocturnal BG forecasting for T1D across 47.5M CGM readings; attention-based architectures lead on RMSE/WQL but no single model dominates across all clinically relevant metrics.
**Abstract**: Nocturnal hypoglycemia in type 1 diabetes (T1D) accounts for 5--6% of T1D mortality and imposes significant psychological burden on patients and caregivers. Continuous glucose monitors (CGMs) have improved real-time glycemic awareness, yet clinically actionable long-horizon forecasting remains unsolved. Recent time series foundation models (TSFMs) have shown strong generalization across temporal domains, but their application to long-horizon nocturnal blood glucose forecasting is critically underexplored. Here we benchmark 12 architectures --- 7 TSFMs and 5 deep learning models spanning MLP-based, attention-based, diffusion, and generative paradigms --- in zero-shot and fine-tuned settings across four public T1D CGM datasets comprising over 47.5M CGM readings, targeting 8-hour nocturnal forecasting horizons. We show that attention-based architectures consistently outperform others on point (RMSE) and probabilistic (WQL) metrics, though rankings diverge on shape-sensitive (DILATE) and classification metrics (AUROC/AUPRC), highlighting that no single architecture dominates across all clinically relevant criteria. We release a fully reproducible codebase and standardized evaluation pipeline that the community can directly extend to new datasets, architectures, and tasks. To our knowledge, this is the largest CGM benchmarking study to date: 47.5M readings evaluated across 12 architectures --- an order of magnitude larger than any prior diabetes forecasting study. Beyond its technical interest to the TSFM community, this benchmark has direct humanitarian impact: better nocturnal forecasting could reduce fear, prevent sleep disruption, and save lives for millions living with T1D.
**Review Mode**: Double-blind (default; anonymized submission)
Code URL: https://anonymous.4open.science/r/nocturnal-hypo-gly-prob-forecast-2EB3/README.md
**Primary Area**: AI for health and biotechnology (e.g., digital medicine, computational biology, public health)
**Contribution Type**: Benchmark design and benchmark analysis, e.g., new benchmarks, benchmark redesign, benchmarking methodologies, benchmark saturation or overfitting studies, analyses of benchmark limitations or failure modes.
**Submission Number**: 3091

## Reviews

### Meta-Review (Area Chair)
The paper presents a large-scale evaluation benchmark investigating whether Time Series Foundation Models (TSFMs) can effectively forecast nocturnal blood glucose dynamics and predict nocturnal hypoglycemia in Type 1 Diabetes (T1D). Evaluating 12 model architectures (7 TSFMs and 5 deep learning baselines across MLP, attention, diffusion, and generative paradigms) across four public T1D continuous glucose monitoring (CGM) datasets comprising over 47.5 million readings, the authors establish an 8-hour nocturnal forecasting benchmark. Key findings indicate that while attention-based architectures lead on standard point-forecasting (RMSE) and probabilistic (WQL) metrics, model rankings diverge significantly on shape-sensitive (DILATE) and classification metrics (AUROC/AUPRC), demonstrating that no single architecture currently dominates across all clinically relevant evaluation dimensions.

The reviewers are in general agreement with the strengths of the work including the high clinical significance and benchmark scale, the broad and diverse model evaluation, and the multi-metric clinical evaluation. However, there are several key weakness that needs to be addressed in the rebuttal:

    - **Exclusively Univariate CGM Modeling**: Relying solely on univariate CGM signals without incorporating essential clinical covariates (e.g., insulin delivery, carbohydrate intake, physical activity, or sleep status) limits the benchmark's alignment with real-world physiological dynamics.
    - **Classification Performance & Alarm Utility**: Model performance on hypoglycemia event classification (AUROC/AUPRC) remains modest across long prediction horizons, leaving questions as to whether long-horizon TSFM predictions can reliably trigger clinical alerts without high false-alarm rates.
    - **Evaluation Setup & Baseline Adaptation**: Questions exist regarding whether zero-shot foundation models and fine-tuned baselines were adapted under strictly comparable context windowing, hyperparameter tuning, and data preprocessing protocols.

Addressing the following prioritized questions will be important considerations as we discuss the paper:

    1. **Univariate Framing vs. Multimodal Clinical Realism**:
        How do you justify framing long-horizon nocturnal forecasting strictly around univariate CGM history, given that overnight glucose trajectories are heavily influenced by evening insulin boluses, meals, and physical activity?
    2. **Clinical Utility of Classification Metrics**:
        Given that model rankings diverge sharply between RMSE and event classification metrics (AUROC/AUPRC), how should researchers interpret model utility for real-world hypoglycemic alarm systems where false positive rates are critical?
    3. **Foundation Model Fine-Tuning & Baseline Alignment**:
        What protocols ensured fair comparison between zero-shot TSFMs and fine-tuned deep learning baselines, particularly regarding input context window lengths, temporal patch sizes, and hyperparameter tuning budgets?
    4. **Data Leakage & Pre-training Contamination**:
        How was potential pre-training data contamination assessed for the zero-shot TSFMs, given that the benchmark relies on widely accessible public T1D CGM datasets?

### Reviewer 1:
**Summary**:

This paper proposes a new benchmark task for forecasting glucose levels and hypoglycemia in patients with type-1 diabetes. I agree with the authors that this paper provides for the first time a clinically relevant prediction task where prior blood glucose forecasting work has focused on shorter timescales and the more distal metric of blood glucose RMSE. Although there is no clear winning model, and prediction accuracy is somewhat poor, the authors show that prediction is possible, and that the metric chosen to evaluate models changes the rankings of best model.

I hope that all future work in this area takes into account the lessons of this paper and reports these clinically-relevant metrics.

#### Strengths:

- **Quality**: The task design is the main strength of this benchmark, and I think an important voice to those forecasting glucose levels for diabetic patients. Previous works have attempted forecasting glucose levels, but to what end? A diabetic patient needs to know if they will become hypoglycemic: RMSE of predicted glucose levels only tells part of the story. Additionally, these authors astutely identify the nighttime windows as an important use case when all diabetic patients could benefit from an accurate long-range glucose forecast.

- **Clarity**: While the provided codebase is messy and rather littered with experiment artifacts, I applaud the authors for including the detail. While I do not think this work would be easy to reproduce, I believe it would be possible with the code provided.

- **Significance**: My hope is that this paper becomes the standard for glucose-level forecasting in the future. RMSE alone is too distal an outcome to make a difference in patients lives; I hope that the long context, long forecast horizon, task-relevant prediction (hypoglycemia as detected by CGM) with uncertainty quantification becomes the standard in T1D forecasting literature.

- **Originality**: Honestly, none of these ideas are "original" in machine learning. We know that error-based metrics hide model quality on downstream decisions, we know getting the appropriate clinical timescale right is important, and we know that uncertainty quantification matters in machine learning for health. However, It seems that none of these

#### Weaknesses

I think the main weaknesses of this paper are its limitations as an actual benchmark for comparing glucose forecasting models. The hyperparameter discussion is woefully lacking. The authors state that the hyperparameters are available in the code. While these are available as a tidy collection of config files, it is impossible to tell how decisions were made about which hyperparameters were searched. Do methods get comparable FLOPs? Clocktime? Configurations?

We see this carry into the results, as we get a rather inconclusive findings. It is ok that different models excel at different tasks, but I would hope that a benchmark work such as this would provide adequate discussion to explain those tradeoffs and inspire future work.

Lastly, there are some contemporaneous works that should be cited and discussed. From Prediction to Practice: A Task-Aware Evaluation Framework for Blood Glucose Forecasting, a May 1st preprint so it fits this conference's definition of contemporaneous applies a similar framework. Benchmarking Hypoglycemia Classification Using Quality-Enhanced DiaData was published late last year and I believe has some significant weaknesses, but is worth mentioning and comparing to.

Despite these weaknesses, I agree that this paper has significant contributions to the field and merits publication as-written. The point of this paper isn't an exhaustive modle benchmark, but a new paradigm for how models should be benchmarked for this task in the future.

**Reproducibility Comments**:

I have reviewed the code provided and reasonably believe I could reproduce this work. Although I wish the hyperparameters were better documented, I was able to locate model-specific parameters.

#### Questions:

I do have one question which, if addressed would improve my perception of the work (but most likely not raise my score):

What sort of mistakes are these models making in hypoglycemia forecasting? Is the precision good, minimizing false-alarms? Is the recall high? The AUPRC numbers somewhat address this, but a confusion matrix or some other discussion of desirable outcomes for clinical use would be appreciated.

I believe this paper adequately answers the questions it raises. The things it makes me curious about are:

    - What model architectures show promise for these tasks and why?
    - Can decision-aware learning techniques aide classification accuracy?
    - What patients are these models successful for and what patients do the models not work for?

But these are all definitely out-of-scope of this work.

**Rating**: 5: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.

**Confidence**: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.


### Reviewer 2:

**Summary**:

The paper presents a large-scale benchmarking study on long-horizon nocturnal hypoglycemia forecasting in type 1 diabetes using time series foundation models and related deep learning baselines. While the topic and experimental effort are meaningful, the paper’s contribution is not sufficiently novel beyond an existing benchmarking exercise and is limited for a high-impact venue.

#### Strengths:

    1. Practical and clinically relevant task formulation. The 8-hour, midnight-anchored forecasting horizon and evaluation of nocturnal hypoglycemia are well motivated by clinical needs.
    2. Evaluating 12 architectures across multiple public datasets and multiple evaluation pillars provides a valuable reference for future work.
    3. The paper makes an effort to go beyond RMSE by incorporating shape- and tail-/event-oriented metrics (e.g., DILATE, AUROC/AUPRC), which is an important methodological message.

#### Weaknesses:

    1. Key conclusions are plausible but not deeply established. Observations (e.g., attention-based models being strong on RMSE/WQL, and other models being better on shape/tail metrics) are intuitive given model output parameterizations and training objectives. The paper does not provide stronger causal analysis or deeper investigation that would elevate the work from benchmarking to scientific advance.
    2. Insufficient clarity on what is “new.” The paper could better articulate how its evaluation protocol differs in principle from existing CGM forecasting benchmarks and why these differences yield fundamentally new insights. Without that, the incremental contribution feels small.

#### Questions:

    Does the paper produce any actionable recommendations for practitioners (e.g., which model + which metric combination) and justify them quantitatively across datasets?
    In setup, the episode-level risk score is defined as the maximum risk score across all 8-hour steps. Does this max aggregation make the evaluation more sensitive to single-step spurious low-probability spikes (i.e., isolated noisy peaks) or is it actually more robust than alternatives such as mean or product aggregation?
    What would be the expected behavior if the nocturnal hypoglycemia is classified via deterministic point forecasts?

**Rating**: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
**Confidence**: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Reviwer 3:

**Summary**:

This work addresses the central challenge of long-horizon nocturnal hypoglycemia forecasting for patients with Type 1 Diabetes (T1D). The paper evaluates 12 forecasting architectures, including seven time-series foundation models (TSFMs) and five deep learning baselines, on four public continuous glucose monitoring datasets comprising approximately 47.5 million glucose measurements. The benchmark focuses on an 8-hour overnight forecasting task and evaluates models using point forecasting metrics (RMSE), probabilistic forecasting metrics (WQL, coverage, sharpness), shape-aware metrics (DILATE), and clinically motivated hypoglycemia detection metrics (AUROC/AUPRC). The claims of the paper are that (1) this is the largest T1D forecasting benchmark to date, (2) attention-based models perform best on traditional forecasting metrics, (3) continuous-output models such as Moirai and Toto perform better on shape-sensitive and hypoglycemia-detection metrics, and (4) no single architecture dominates across all clinically relevant evaluation criteria. The authors additionally release a standardized evaluation pipeline intended to facilitate future research.

#### Strengths:

The paper presents a large-scale and carefully organized benchmarking effort. The inclusion of four public datasets and approximately 47.5 million glucose readings substantially exceeds the scale of most prior diabetes forecasting studies. The evaluation protocol covers multiple clinically meaningful dimensions rather than relying exclusively on RMSE, which is an important methodological contribution.

A major strength is the demonstration that model rankings vary significantly across evaluation metrics. The findings suggest that optimization for point forecasting accuracy may not translate into superior clinical event detection performance. This observation is valuable for both the TSFM and healthcare forecasting communities.

The benchmark is broad in architectural coverage, including foundation models, transformers, diffusion models, recurrent models, and MLP-based architectures. Such diversity increases the usefulness of the benchmark as a reference resource.

The paper is generally well written and motivates the clinical importance of nocturnal hypoglycemia forecasting effectively. The discussion of shape-aware evaluation and uncertainty quantification is particularly relevant for real-world deployment.

#### Weaknesses:

I have several concerns that limit my confidence in the paper's conclusions.

**Questionable validity of the nocturnal evaluation protocol**.

The benchmark is built around midnight-anchored forecasting because sleep annotations are unavailable. However, the paper explicitly acknowledges that patients may still be awake and actively consuming food or insulin during portions of the evaluation horizon. This introduces a potentially substantial mismatch between the benchmark definition and the clinical problem being studied. The resulting label noise may materially affect conclusions regarding nocturnal hypoglycemia forecasting. The limitation is acknowledged but its quantitative impact is not investigated.

**Lack of statistical significance analysis**.

The paper reports many performance differences between models, but no statistical significance tests, confidence intervals, bootstrap analyses, or paired comparisons are provided. Several reported improvements appear numerically small. Without uncertainty estimates, it is difficult to determine whether observed ranking differences are meaningful.

**Potential unfairness in model comparison**.

Some models are evaluated with insulin and carbohydrate covariates while others are effectively restricted to glucose-only settings. As a result, improvements attributed to architecture may partially reflect access to additional information rather than superior modeling capability. The benchmark would benefit from a strictly controlled comparison where all models receive identical inputs whenever possible.

**Insufficient analysis of dataset heterogeneity**.

Performance varies substantially across datasets. However, there is little investigation into why models succeed or fail on different cohorts, devices, sensor generations, or patient populations. Since ED-track contributions should provide actionable evaluation insights, deeper error analysis would strengthen the contribution.

**Missing subgroup and fairness analyses**.

The paper explicitly states that no subgroup analyses were performed. Given the healthcare context, performance variation across age groups, sex, treatment regimens, and disease characteristics is important. The absence of such analysis limits conclusions regarding real-world utility and generalizability.

**Strong clinical claims are not fully supported**.

The paper repeatedly argues that the benchmark may have humanitarian impact and could reduce harm. While the motivation is compelling, the presented results remain retrospective forecasting evaluations. No prospective deployment study, alarm analysis, clinical workflow evaluation, or decision-support assessment is conducted. Therefore some of the broader clinical claims should be tempered.

**Dataset contribution is limited**.

Although the benchmark is large, the datasets themselves are not newly collected. The primary contribution is evaluation rather than dataset creation. Therefore the significance of the dataset-related contribution should be assessed accordingly.

**Limitations**:

The limitations section is thoughtful but does not sufficiently address the consequences of midnight anchoring, missing sleep labels, subgroup fairness analysis, and statistical uncertainty in benchmark rankings.

#### Questions:

1. Can the authors quantify the impact of midnight anchoring by analyzing subsets of nights with stronger evidence of sleep periods?
2. Can the authors provide statistical significance tests or bootstrap confidence intervals for major benchmark conclusions?
3. How much of the observed performance improvement is attributable to access to insulin/carbohydrate covariates rather than architecture design?
4. Can the authors perform subgroup analyses (age, sex, treatment modality, sensor type, etc.) to assess robustness and fairness?
5. Can the authors provide evidence that improvements in AUROC/AUPRC translate into clinically meaningful reductions in missed hypoglycemic events?

**Rating**: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
**Confidence**: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
