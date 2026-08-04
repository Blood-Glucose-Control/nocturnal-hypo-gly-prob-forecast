PROMPT
I just got my reviews back for NeurIPS 2026

My paper was a benchmarking paper, evaluating dozens of models against each other, the reviewer was a big fan of my paper but one of the things they pointed out was the  'lack of statistical significance analysis' of model performance.

"The paper reports many performance differences between models, but no
 statistical significance tests, confidence intervals, bootstrap analyses, or paired comparisons are provided. Several reported improvements appear numerically small. Without uncertainty estimates, it  is difficult to determine whether observed ranking differences are  meaningful."

If I am comparing dozens of models to each other what is the proper way to set up the confidence intervals, bootstrap analyses and paired comparisons?

RESPONSE

Congratulations on the positive feedback for your NeurIPS submission! Getting a reviewer to champion a benchmarking paper is a huge hurdle cleared. Their request for statistical rigor is a classic reviewer point, and honestly, addressing it will make your evaluation bulletproof.

When you are dealing with dozens of models, the standard approach of just running t-tests breaks down completely. If you are comparing $k$ models, you have $\frac{k(k-1)}{2}$ pairs to test. For 20 models, that is 190 separate comparisons. Doing this without correction guarantees a massive inflation of false positives (the Family-Wise Error Rate).

Here is the standard, statistically rigorous pipeline for evaluating dozens of models in a machine learning benchmark.

---

### 1. The Omnibus Test: Friedman Test

Before comparing specific pairs like Chronos-2 versus PatchTST, you must first prove that there is a statistically significant difference somewhere among the entire group of models.

* **Why it works:** The Friedman test is a non-parametric alternative to ANOVA. Instead of looking at raw metric values, it ranks the algorithms for each dataset or time series independently. The best performing model gets rank 1, the second gets rank 2, and so on.
* **The Math:** It calculates the test statistic $\chi_F^2$ based on the average ranks $R_j$ of $k$ models across $N$ datasets. If the resulting p-value is less than your significance level (usually $\alpha = 0.05$), you can reject the null hypothesis that all models perform equally.

### 2. Post-Hoc Paired Comparisons

Once the Friedman test tells you that differences exist, you can zero in on the pairs.

* **Wilcoxon Signed-Rank Test:** Use this to compare the performance of two specific models across all your datasets. It is non-parametric and does not assume your performance differences are normally distributed.
* **Multiple Testing Corrections:** Because you are running many Wilcoxon tests, you must correct the p-values.
* **Bonferroni Correction:** The simplest but most conservative (adjusting your threshold to $\alpha / m$).
* **Holm’s Step-Down Procedure:** More powerful than Bonferroni and highly recommended for ML benchmarks.
* **Benjamini-Hochberg:** Controls the False Discovery Rate (FDR) and is excellent when you have a massive matrix of comparisons.



### 3. Bootstrapping and Confidence Intervals

The reviewer wants to see uncertainty estimates around your point metrics (like your RMSE, WQL, or DILATE scores).

* **The Time-Series Caveat:** Standard bootstrapping assumes your data is independent and identically distributed (i.i.d.). Because you are dealing with sequential forecasting, randomly resampling individual points will destroy the temporal dependencies and autocorrelation.
* **Block Bootstrapping:** You must use block bootstrapping (like Moving Block Bootstrap). This method samples continuous "blocks" of sequential data rather than individual points.
* **Execution:** Create 1,000 to 10,000 bootstrap datasets using the block method. Calculate your metric (e.g., WQL) for your model on each bootstrapped dataset. Sort these scores to find the 2.5th and 97.5th percentiles. This gives you a robust 95% confidence interval to report alongside your mean improvements.

### 4. Visualization: Critical Difference Diagrams

When comparing dozens of models, tables full of p-values are illegible. You need a Critical Difference (CD) diagram (introduced by Demšar in 2006). This is the gold standard for NeurIPS and ICML benchmarking papers.

* A CD diagram plots the average rank of every model (e.g., Moirai, Toto, etc.) on a horizontal axis.
* It calculates a "Critical Difference" threshold.
* Models that are connected by a thick horizontal line are **not** significantly different from each other. Models that do not share a line have a statistically significant difference in performance.
* This instantly shows the reviewer the exact, meaningful ranking hierarchy without them having to squint at a table of fractional improvements.

---

Adding these elements will directly answer the reviewer's concerns and elevate the quality of your research proposal as well. What specific statistical software or Python libraries are you currently using for your evaluation pipeline, so we can figure out the fastest way to implement these tests and plot the CD diagrams before your deadline?

PROMPT

The way the experiment is set up we have midnight anchors for each episode because we only care about our forecasts at a particular time. We have four datasets and the number of evaluation episodes per dataset is from 2000-5000, does that change what you're saying about the bootstrapping?

RESPONSE

Yes, that architectural setup completely changes the bootstrapping strategy, and it actually makes your life significantly easier.

Because your experiment is structured around discrete, midnight-anchored evaluation episodes, the fundamental unit of your statistical analysis shifts. You are no longer evaluating a single, continuous, autocorrelated stream of time steps; you are evaluating 2,000 to 5,000 distinct forecasting *instances* per dataset.

Here is how your specific episodic structure updates the statistical approach.

### 1. Episode-Level Bootstrapping (No Block Bootstrapping Required)

The temporal dependency (the autocorrelation) exists *within* the forecast window of each individual episode. However, the final evaluation scores assigned to those episodes can generally be treated as independent and identically distributed (i.i.d.) samples for the purpose of comparing models.

Because of this, you can abandon the complex Moving Block Bootstrap and use standard empirical bootstrapping.

**The Pipeline:**

* **Step 1:** For a given dataset, compile the individual episode scores for a specific model. For example, you will have a vector of 2,000 distinct DILATE, WQL, or RMSE scores for that model.
* **Step 2:** Randomly sample from this vector of 2,000 scores *with replacement* to create a new, bootstrapped sample of 2,000 scores.
* **Step 3:** Calculate the aggregate metric (e.g., the mean WQL) for this bootstrapped sample.
* **Step 4:** Repeat this process 10,000 times to build a distribution of the mean score.
* **Step 5:** Find the 2.5th and 97.5th percentiles of this distribution to establish your 95% confidence intervals.

*Caveat:* The only exception where you might still need to consider block bootstrapping across episodes is if your midnight anchors are highly clustered (e.g., consecutive days) *and* your forecast horizon heavily overlaps, causing the errors of Episode $n$ to be highly correlated with the errors of Episode $n+1$. If they are sufficiently spaced or independent draws, standard bootstrapping is perfectly mathematically sound.

### 2. The "Large $N$" Problem for Significance Tests

Having 2,000 to 5,000 evaluation episodes per dataset is fantastic for robust benchmarking, but it introduces a common statistical quirk: the "Large $N$" problem.

When your sample size ($N$) is that large, tests like the Wilcoxon Signed-Rank Test have immense statistical power. They will likely return a $p$-value of $< 0.001$ for almost every model comparison, flagging even a $0.0001$ improvement in RMSE as "statistically significant."

* **The Reviewer's Real Concern:** The reviewer noted that "Several reported improvements appear numerically small." A tiny $p$-value won't convince them if the raw improvement is negligible.
* **The Solution:** You must report **Effect Size** alongside your statistical significance. For Wilcoxon tests, you can calculate the $r$ effect size ($r = Z / \sqrt{N}$). This will prove to the reviewer whether the difference between models is not just statistically real, but practically meaningful in the context of your benchmark.

### 3. Adapting the Critical Difference Diagrams

With four distinct datasets, you have two choices for your CD diagrams:

1. **Per-Dataset Diagrams:** Generate one CD diagram for each dataset, ranking the dozen models based on their performance across the 2,000–5,000 episodes.
2. **Global Diagram:** Average the model ranks across all four datasets to create one master CD diagram. Given you have massive intra-dataset samples, the per-dataset diagrams will be much more informative for a NeurIPS audience, showing where certain architectures excel or fail.

Since calculating pairwise tests and plotting CD diagrams for dozens of models across thousands of episodes requires a bit of heavy lifting, are you planning to write a custom script for this, or would you like to leverage existing libraries like `scikit-posthocs` or `autorank` to automate the pipeline?
