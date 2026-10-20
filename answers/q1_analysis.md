# Question 1 — Decision Trees and Bagging

## Q1(a): Single Decision Tree Baseline

A decision tree classifier was trained on the Olivetti Faces dataset using raw flattened grayscale pixels, resulting in 4096 features per image.

The training, validation, and test sets contain 240, 80, and 80 images respectively, with six, two, and two images per identity.

A grid search was performed over:

- `max_depth = [2, 4, 8, 16, None]`
- `min_samples_leaf = [1, 2, 4]`

All 15 combinations were evaluated using validation accuracy. The test set was not used for hyperparameter selection.

### Selected Configuration

| Parameter or Metric | Value |
|---|---:|
| Maximum depth | None (unrestricted) |
| Minimum samples per leaf | 4 |
| Actual tree depth | 31 |
| Number of leaves | 41 |
| Mean samples per leaf | 5.854 |
| Fraction of single-sample leaves | 0.000 |
| Training accuracy | 96.67% |
| Validation accuracy | 60.00% |
| Clean test accuracy | 60.00% |

### Analysis

Increasing the permitted tree depth allows the classifier to create more splits and learn increasingly complex decision boundaries.

At shallow depths, the tree cannot adequately distinguish between 40 different identities, resulting in underfitting.

The unrestricted tree achieves 96.67% training accuracy, while validation accuracy remains at 60.00%. This generalization gap indicates overfitting.

The selected minimum leaf size of four prevents leaves containing only one training example. Consequently, the selected model has a single-sample leaf fraction of zero.

## Q1(b): Bagging Implementation

Bootstrap aggregation was implemented manually using scikit-learn decision trees as base learners.

Each tree was trained using a bootstrap sample of 240 observations drawn with replacement from the original training set.

Predictions were combined using majority voting. When classes received equal maximum votes, the class with the smallest numerical label was selected.

The following ensemble sizes were evaluated:

`B = [1, 5, 10, 25, 50, 100, 200]`

### Bagging Results

| Trees | Train Accuracy | Validation Accuracy | Clean Test Accuracy |
|---:|---:|---:|---:|
| 1 | 63.75% | 41.25% | 33.75% |
| 5 | 92.50% | 63.75% | 52.50% |
| 10 | 98.75% | 76.25% | 70.00% |
| 25 | 100.00% | 83.75% | 78.75% |
| 50 | 100.00% | 87.50% | 80.00% |
| 100 | 100.00% | 90.00% | 91.25% |
| 200 | 100.00% | 90.00% | 91.25% |

### Selected Ensemble

The 100-tree ensemble was selected because it achieved the maximum validation accuracy of 90.00%, matching the 200-tree ensemble while requiring fewer estimators.

Its clean test accuracy was 91.25%, compared with 60.00% for the single decision tree.

This corresponds to an improvement of 31.25 percentage points.

### Bootstrap Diversity

For a training set of size n, the expected number of distinct examples in a bootstrap sample of size n is:

E[U] = n × [1 − (1 − 1/n)^n]

With n = 240:

E[U] = 151.8932

The experimentally observed averages ranged from approximately 149.60 to 153.00 distinct examples.

At 200 trees, the observed average was 151.75, close to the theoretical expectation.

Thus, each bootstrap sample contains roughly 63.3% distinct training examples on average, with the remaining draws being duplicates.

## Q1(c): OOB Accuracy, Disagreement and Vote Margin

Three ensemble statistics were calculated for each ensemble size.

### Out-of-Bag Accuracy

Each training example was classified using only the trees whose bootstrap samples did not contain that example.

OOB accuracy increased as more trees were added.

| Trees | OOB Accuracy |
|---:|---:|
| 1 | 18.89% |
| 5 | 30.91% |
| 10 | 41.18% |
| 25 | 54.17% |
| 50 | 64.58% |
| 100 | 71.25% |
| 200 | 76.25% |

At one tree, only 90 of the 240 training examples had an OOB prediction. By 25 trees, all 240 examples were covered.

OOB accuracy therefore becomes more informative as OOB coverage increases.

### Pairwise Disagreement

Pairwise disagreement measures the fraction of inputs on which two base classifiers produce different predictions.

For B trees, the reported statistic averages disagreement across all distinct pairs.

At 100 trees, the mean pairwise disagreement was 0.7964, indicating substantial diversity among the individual trees.

For B = 1, pairwise disagreement is undefined because no pair of trees exists.

### Vote Margin

The vote margin is the difference between the two largest normalized class vote counts.

At 100 trees, the mean validation vote margin was 0.2726.

Although validation and OOB accuracy generally improved with ensemble size, disagreement and vote margin remained relatively stable for larger ensembles.

Therefore, the three statistics do not tell exactly the same story. OOB accuracy more directly reflects predictive performance, while disagreement and vote margin characterize diversity and voting behavior.

High disagreement can be useful for bagging, but it does not guarantee high accuracy.

## Q1(d): Single Tree versus Bagging under Occlusion

The frozen single decision tree and selected 100-tree bagging model were evaluated on the original clean test images and on images with the central 16 × 16 pixel block zeroed.

The mask was applied only during evaluation, not during training.

### Accuracy and Confidence

| Condition | Tree Accuracy | Bagging Accuracy | Tree Mean Confidence | Bagging Mean Confidence |
|---|---:|---:|---:|---:|
| Clean | 60.00% | 91.25% | 95.80% | 36.91% |
| Masked | 60.00% | 65.00% | 95.80% | 29.85% |

The single tree's confidence is its predicted leaf-class probability, while bagging confidence is its winning vote fraction. These quantities are not necessarily calibrated probabilities and should not be interpreted as directly equivalent.

Bagging accuracy declined by 26.25 percentage points under masking, but it still exceeded the single tree's masked accuracy.

The single tree's aggregate accuracy and mean confidence remained unchanged, although this does not establish that its predictions were identical for every image.

### Cases Where Bagging Corrected a Tree Error

**Example 1 — Test image 1**

- True identity: 0
- Single-tree prediction: 23
- Bagging prediction: 0
- Bagging votes for identity 0: 16
- Bagging votes for identity 16: 16

Bagging correctly classified the face, although its winning class tied with identity 16. The deterministic smallest-label tie-breaking rule selected identity 0.

**Example 2 — Test image 3**

- True identity: 1
- Single-tree prediction: 15
- Bagging prediction: 1
- Bagging votes for identity 1: 41

Here, the correct identity received the largest vote count, demonstrating how aggregation can correct a single tree's error.

### Cases Where Both Models Failed

**Example 3 — Test image 17**

- True identity: 8
- Single-tree prediction: 37
- Bagging prediction: 3
- Votes for class 3: 9
- Votes for true class 8: 8

The true class narrowly lost to an incorrect class, showing that voting cannot guarantee correct recognition when votes are dispersed.

**Example 4 — Test image 18**

- True identity: 9
- Single-tree prediction: 0
- Bagging prediction: 28
- Votes for class 28: 16
- Votes for true class 9: 2

Many base learners favored incorrect identities, so aggregation did not recover the correct class.

### Interpretation

Bagging reduces the sensitivity of classification to the behavior of any particular decision tree.

It can correct individual-tree errors when the ensemble collectively supports the correct identity. However, errors can persist when the ensemble's votes are concentrated on incorrect classes or spread across many identities.

Central occlusion substantially reduces the bagged model's recognition accuracy, indicating that the removed visual information is important for its predictions.

## Question 1 Conclusion

The unrestricted single decision tree achieved high training accuracy but generalized poorly.

Bagging improved clean test accuracy from 60.00% to 91.25% by aggregating predictions from diverse bootstrap-trained trees.

OOB accuracy improved with ensemble size, while disagreement and vote margin provided complementary information about the voting process.

Although the bagged model suffered a substantial accuracy decrease under central occlusion, it remained more accurate than the single tree.