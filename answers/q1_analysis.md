# Question 1. Decision Trees and Bagging

## Q1(a): Single Decision Tree Baseline

A decision tree classifier was trained on the Olivetti Faces dataset using flattened grayscale pixels giving 4096 features per image.

The training set contains 240 images the validation set contains 80 images. The test set contains 80 images. Each identity has six images in the training set two images in the validation set and two images in the test set.

A grid search was performed over:

- `max_depth = [2, 4 8 16 None]`

- `min_samples_leaf = [1 2 4]`

All fifteen combinations were evaluated based on validation accuracy. The test set was not used to select hyperparameters.

### Selected Configuration

| Parameter or Metric | Value |

|---|---:|

Maximum depth | None () |

| Minimum samples per leaf | 4 |

Actual tree depth | 31 |

Number of leaves | 41 |

| Mean samples per leaf | 5.854 |

| Fraction of single-sample leaves | 0.000 |

| Training accuracy | 96.67% |

Validation accuracy | 60.00% |

| Clean test accuracy | 60.00% |

### Analysis

Increasing the allowed depth of the decision tree lets the decision tree create splits and learn more complex decision boundaries.

At depths the decision tree cannot adequately distinguish between 40 different identities leading to underfitting.

The unrestricted decision tree achieves 96.67 % training accuracy while validation accuracy remains at 60.00 %. This gap shows that the decision tree is overfitting.

The chosen minimum leaf size of four prevents leaves containing one training example. Consequently the decision tree has a single-sample leaf fraction of zero.

## Q1(b): Bagging Implementation

Bootstrap aggregation was implemented manually using decision trees as base learners.

Each decision tree was trained on a bootstrap sample of 240 observations drawn with replacement from the training set.

Predictions were combined using majority voting. When classes received maximum votes the class with the smallest numerical label was selected.

The following ensemble sizes were evaluated:

`B = [1 5 10 25 50 100 200]`

### Bagging Results

| Trees | Train Accuracy | Validation Accuracy | Clean Test Accuracy |

|---:|---:|---:|---:|

| 1 | 63.75% | 41.25% | 33.75% |

| 5 | 92.50% | 63.75% | 52.50% |

| 10 98.75% | 76.25% | 70.00% |

| 25 | 100.00% | 83.75% | 78.75% |

| 50 | 100.00% | 87.50% | 80.00% |

| 100 | 100.00% | 90.00% | 91.25% |

| 200 100.00% | 90.00% | 91.25% |

### Selected Ensemble

The 100‑tree ensemble was chosen because it achieved the validation accuracy of 90.00 % matching the 200‑tree ensemble while requiring fewer decision trees.

Its clean test accuracy was 91.25 % compared with 60.00 % for the decision tree.

This represents an improvement of 31.25 percentage points.

### Bootstrap Diversity

For a training set of size \(n\) the expected number of examples in a bootstrap sample of size \(n\) is:

E[U] = n × [1 − (1 − 1/n)^n]

With n = 240:

E[U] = 151.8932

The experimentally observed averages ranged from approximately 149.60 to 153.00 distinct examples.

At 200 trees the observed average was 151.75 close to the expectation.

Thus each bootstrap sample contains 63.3 % distinct training examples on average with the remaining draws being duplicates.

## Q1(c): OOB Accuracy, Disagreement and Vote Margin

Three ensemble statistics were calculated for each size.

### Out-of-Bag Accuracy

Each training example was classified using the decision trees whose bootstrap samples did not contain that example.

Out‑of‑bag accuracy increased as more decision trees were added.

| Trees OOB Accuracy |

|---:|---:|

| 1 | 18.89% |

| 5 | 30.91% |

| 10 | 41.18% |

25 | 54.17% |

| 50 | 64.58% |

| 100 | 71.25% |

| 200 76.25% |

With one decision tree only 90 of the 240 training examples had an OOB prediction. By 25 decision trees all 240 examples were covered.

Out‑of‑bag accuracy therefore becomes more informative, as OOB coverage increases.

### Pairwise Disagreement

Pairwise disagreement measures the fraction of inputs on which two base decision trees produce predictions.

For B trees the reported statistic averages disagreement across all pairs of trees.

At 100 trees the mean pairwise disagreement is 0.7964 showing diversity among the individual trees.

For B equals 1 pairwise disagreement is not defined because no pair of trees exists.

### Vote Margin

The vote margin is the difference between the two normalized class vote counts.

With 100 trees the mean validation vote margin is 0.2726.

Although validation and OOB accuracy generally improve with size, disagreement and vote margin stay fairly stable for larger ensembles.

Thus the three statistics do not tell the same story. OOB accuracy shows performance more directly while disagreement and vote margin describe diversity and voting behavior.

High disagreement can help bagging. It does not guarantee high accuracy.

## Q1(d): Single Tree versus Bagging under Occlusion

The frozen single decision tree and the chosen 100-tree bagging model were tested on the clean test images and on images where the central 16 × 16 pixel block was zeroed.

The mask was applied during evaluation not during training.

### Accuracy and Confidence

| Condition Tree Accuracy | Bagging Accuracy | Tree Mean Confidence | Bagging Mean Confidence |

|---|---:|---:|---:|---:|

| Clean | 60.00% 91.25% | 95.80% | 36.91% |

| Masked | 60.00% | 65.00% | 95.80% | 29.85% |

The single trees confidence equals its predicted leaf-class probability while bagging confidence equals its winning vote fraction. These values are not calibrated probabilities and should not be seen as directly equivalent.

Bagging accuracy fell by 26.25 percentage points under masking. It still stayed higher than the single trees masked accuracy.

The trees overall accuracy and mean confidence stayed the same although that does not prove its predictions were the same for every image.

### Cases Where Bagging Corrected a Tree Error

Example 1. Test image 1

- True identity: 0

- Single-tree prediction: 23

- Bagging prediction: 0

- Bagging votes for identity 0: 16

- Bagging votes for identity 16: 16

Bagging correctly classified the face although its winning class tied with identity 16. The rule that breaks ties by choosing the label picked identity 0.

Example 2. Test image 3

- identity: 1

- Single-tree prediction: 15

- Bagging prediction: 1

- Bagging votes for identity 1: 41

Here the correct identity received the largest vote count showing how aggregation can fix a single trees error.

### Cases Where Both Models Failed

Example 3. Test image 17

- True identity: 8

- Single-tree prediction: 37

- Bagging prediction: 3

- Votes for class 3: 9

- Votes for class 8: 8

The true class narrowly lost to an incorrect class showing that voting cannot guarantee correct recognition when votes are spread.

Example 4. Test image 18

- identity: 9

- Single-tree prediction: 0

- Bagging prediction: 28

- Votes for class 28: 16

- Votes for true class 9: 2

Many base learners favored incorrect identities so aggregation did not recover the correct class.

### Interpretation

Bagging lowers how much classification depends on any decision tree.

It can fix errors from a tree when the ensemble as a whole backs the correct identity.. Errors can stay when the ensemble votes focus on wrong classes or are spread across many identities.

Central occlusion cuts the models recognition accuracy a lot showing that the removed visual information matters a lot for its predictions.

## Question 1

The unrestricted single decision tree reached training accuracy but did not generalize well.

Bagging lifted test accuracy from 60.00% to 91.25% by combining predictions from diverse bootstrap-trained trees.

OOB accuracy grew as the grew while disagreement and vote margin gave extra information about the voting process.

Even though the bagged model lost a lot of accuracy under occlusion it was still more accurate, than the single tree.