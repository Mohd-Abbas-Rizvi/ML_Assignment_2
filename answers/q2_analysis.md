# Question 2 — Multiclass SAMME Boosting

## Q2(a): Manual SAMME Demonstration

Before applying boosting to Olivetti Faces, a three-class example with nine training samples was constructed.

A depth-1 decision tree was fitted using uniform initial sample weights.

The SAMME algorithm assigns a learner weight according to:

alpha_t = log((1 − epsilon_t) / epsilon_t) + log(K − 1)

where epsilon_t is the weighted classification error and K is the number of classes.

Sample weights are updated by multiplying the weight of each incorrectly classified observation by exp(alpha_t), followed by normalization.

### Experimental Results

| Quantity | Value |
|---|---:|
| Samples | 9 |
| Classes | 3 |
| Initial sample weight | 0.111111 |
| Misclassified samples | 3 |
| Weighted error | 0.333333 |
| Learner weight alpha | 1.386294 |
| Updated weight of each correct example | 0.055556 |
| Updated weight of each incorrect example | 0.222222 |
| Sum of updated weights | 1.000000 |

Because exp(1.386294) = 4, each incorrectly classified example receives four times the unnormalized weight of a correctly classified example.

This illustrates how SAMME directs later learners toward previously misclassified examples.

The implementation rejects weak learners when their weighted error is at least 1 − 1/K. Perfect learners trigger early termination.

## Q2(b): SAMME on Olivetti Faces

Multiclass SAMME was implemented manually using weighted decision trees.

No prebuilt AdaBoost classifier was used.

The experiment evaluated:

- Weak learner depths: 1, 2, 3, 4
- Boosting rounds: 10, 25, 50, 100
- Number of identities: 40

All 16 configurations were compared using clean validation accuracy.

### Selected Configuration

| Metric | Result |
|---|---:|
| Weak learner depth | 4 |
| Boosting rounds | 100 |
| Actual rounds | 100 |
| Training accuracy | 97.08% |
| Validation accuracy | 32.50% |
| Clean test accuracy | 31.25% |
| Training time | 20.48 seconds |

The selected model has a large training–validation accuracy gap, indicating that its strong training performance does not translate into similarly strong generalization.

### Round-by-Round Diagnostics

Four diagnostic plots were generated.

**Weighted training error**

The weighted errors varied across boosting rounds, frequently lying around 0.60–0.73 in later rounds. These are errors of individual weak learners under changing sample weights, not cumulative ensemble classification errors.

For 40 classes, the rejection threshold is 1 − 1/40 = 0.975. No learner was rejected in the selected run.

**Learner weight**

Learner weights frequently ranged around 2.7–3.3.

A lower weighted error produces a larger alpha, giving the corresponding learner greater influence in the ensemble's final weighted vote.

**Maximum sample weight**

The maximum individual sample weight occasionally approached 0.08, much larger than the uniform starting weight of 1/240 = 0.004167.

**Weight of the 10 heaviest images**

The combined weight of the ten heaviest samples often reached approximately 0.30–0.40 in later rounds.

These diagnostics show that boosting concentrates its emphasis on a relatively small number of difficult training examples.

## Q2(c): Bagging versus Boosting

Bagging and SAMME were compared using 25, 50 and 100 learners.

For bagging, the selected unrestricted decision-tree configuration with minimum leaf size four was used.

For SAMME, the selected depth-four weak learner was used.

### Experimental Comparison

| Method | Learners | Train Accuracy | Validation Accuracy | Clean Test Accuracy | Training Time |
|---|---:|---:|---:|---:|---:|
| Bagging | 25 | 100.00% | 83.75% | 78.75% | 16.23 s |
| SAMME | 25 | 67.92% | 27.50% | 32.50% | 5.11 s |
| Bagging | 50 | 100.00% | 87.50% | 80.00% | 32.57 s |
| SAMME | 50 | 57.08% | 18.75% | 23.75% | 10.25 s |
| Bagging | 100 | 100.00% | 90.00% | 91.25% | 65.08 s |
| SAMME | 100 | 97.08% | 32.50% | 31.25% | 20.48 s |

Bagging consistently achieved better validation and test accuracy.

SAMME trained faster under the measured configurations, but its recognition accuracy was substantially lower.

Increasing the number of boosting rounds did not guarantee better performance. For example, its clean test accuracy decreased from 32.50% at 25 learners to 23.75% at 50 learners.

### Analysis of a High-Weight Training Face

A face with a particularly high final boosting weight was selected for examination.

| Property | Value |
|---|---:|
| Training index | 34 |
| Original dataset index | 52 |
| True identity | 5 |
| Incorrect weak-learner predictions | 88 out of 100 |
| Final sample weight | 0.058912 |

The face was misclassified in 88 of the 100 boosting rounds.

At round 91, its weight increased from 0.029200 to 0.045090 following an incorrect prediction.

At round 92, the weak learner classified the face correctly, and the normalized weight decreased to 0.003334.

From rounds 93 to 100, the face was misclassified eight consecutive times, causing its weight to increase again to 0.058912.

This example illustrates SAMME's adaptive emphasis on difficult samples.

However, repeatedly increasing a sample's weight does not guarantee that future learners will classify it correctly.

## Q2(d): Clean and Masked Robustness

The three frozen models were evaluated on clean and centrally masked test images.

The masked images were produced by replacing the central 16 × 16 region with zeros.

### Model Comparison

| Model | Clean Accuracy | Masked Accuracy | Accuracy Drop |
|---|---:|---:|---:|
| Single Decision Tree | 60.00% | 60.00% | 0 pp |
| Bagging | 91.25% | 65.00% | 26.25 pp |
| SAMME | 31.25% | 26.25% | 5.00 pp |

Bagging achieved the highest accuracy under both evaluation conditions.

However, bagging also showed the greatest absolute accuracy reduction when the central region was removed.

SAMME's smaller accuracy drop does not imply superior overall robustness, since its original clean accuracy was already much lower.

### Confidence Comparison

| Model | Clean Mean Confidence | Masked Mean Confidence |
|---|---:|---:|
| Single Decision Tree | 95.80% | 95.80% |
| Bagging | 36.91% | 29.85% |
| SAMME | 12.51% | 12.16% |

The confidence measures are defined differently for the three models:

- Single tree: largest leaf-class probability
- Bagging: largest class vote fraction
- SAMME: largest normalized weighted class score

These quantities are useful for analyzing each model's behavior but are not necessarily calibrated probabilities.

### A Face with Different Responses

Test image 4, corresponding to original dataset index 21, was selected.

Its true identity is 2.

| Model | Clean Prediction | Masked Prediction |
|---|---:|---:|
| Single Tree | 27 | 27 |
| Bagging | 2 (correct) | 13 (incorrect) |
| SAMME | 3 (incorrect) | 12 (incorrect) |

**Bagging behavior**

For the clean image, identity 2 received 14 votes and was predicted correctly.

After masking, identity 2 received only six votes, while incorrect identity 13 received 23 votes.

Mean pairwise tree disagreement on this image decreased from 0.9424 to 0.9149.

Masking slightly reduced pairwise disagreement from 0.9424 to 0.9149. However, disagreement remained high, and the incorrect winning identity received only 23 of the 100 votes. This reflects a shift in the vote distribution toward an incorrect identity rather than strong agreement among the ensemble members.

**SAMME behavior**

SAMME misclassified the face in both conditions, but its predicted identity changed from 3 to 12.

The normalized weighted score for the true identity remained approximately 0.04678.

At the end of training, the six training samples belonging to this identity had a combined SAMME sample weight of approximately 0.02090.

These measurements show how bagging's vote distribution and boosting's weighted class scores respond differently to the removal of the central image region.

The training sample weights show which observations received emphasis during boosting, but they do not by themselves establish the cause of an individual test prediction.

## Question 2 Conclusion

The manually implemented SAMME algorithm correctly performed multiclass weighted-error calculations, learner weighting and sample-weight updates.

In the evaluated configurations, SAMME reached high training accuracy but generalized poorly to unseen faces.

Bagging achieved substantially higher validation and test accuracy, although it required more training time and suffered a larger absolute accuracy decline under central occlusion.

These results demonstrate that sequentially emphasizing difficult training observations does not automatically improve generalization, particularly in a small, high-dimensional multiclass dataset.