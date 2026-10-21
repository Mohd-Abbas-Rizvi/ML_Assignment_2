# Question 3 — Bias–Variance Analysis

## Q3(a): Measured Bias and Variance

The experiment used 30 independently resampled training sets, each containing six bootstrap draws from each of the 40 identities. The original validation and test sets were held fixed.

For each refit, the models produced 40-dimensional class-score vectors on the same 80 clean test images.

The resulting squared bias, variance, and accuracy measurements were:

| Model | Bias² | Variance | Mean Test Accuracy | Accuracy Std. Dev. |
|---|---:|---:|---:|---:|
| Single Tree | 0.3550 | 0.6367 | 50.42% | 4.44 pp |
| Bagging | 0.5246 | 0.0257 | 77.83% | 3.92 pp |
| SAMME | 0.8824 | 0.0169 | 33.79% | 8.05 pp |

The single tree exhibited the largest class-vector variance. Bagging reduced variance substantially while achieving the highest mean test accuracy. SAMME exhibited the lowest measured variance but the largest squared bias, illustrating that low variance alone does not imply good generalization.

## Q3(b): Effect of Tree Depth

The experiments compared maximum-depth settings of 4, 16, and None (unrestricted), using min_samples_leaf=4.

All depths used the same 30 resampled training sets, and every bagged ensemble contained 100 trees.

| Depth | Method | Bias² | Variance | Mean Accuracy |
|---|---|---:|---:|---:|
| 4 | Single Tree | 0.8595 | 0.0643 | 9.79% |
| 4 | Bagging | 0.8960 | 0.0132 | 27.04% |
| 16 | Single Tree | 0.6509 | 0.2237 | 27.38% |
| 16 | Bagging | 0.6764 | 0.0217 | 61.29% |
| None | Single Tree | 0.3550 | 0.6367 | 50.42% |
| None | Bagging | 0.5246 | 0.0257 | 77.83% |

Increasing the maximum permitted depth reduced the single tree's squared bias while increasing variance. Bagging reduced variance at all three depths.

The selected baseline had max_depth=None, meaning no explicit depth limit. As no setting can be deeper than unrestricted growth, depths 4 and 16 were compared with this unrestricted setting. This limitation is acknowledged instead of claiming that a depth greater than None was tested.

## Q3(c): Does the Deepest Tree Give the Largest Improvement from Bagging?

**No, not when improvement is measured as the gain in test accuracy.**

| Depth | Variance Reduction | Accuracy Gain |
|---|---:|---:|
| 4 | 79.53% | +17.25 percentage points |
| 16 | 90.31% | +33.92 percentage points |
| None | 95.97% | +27.42 percentage points |

The unrestricted tree achieved the largest absolute and relative variance reduction, but the intermediate-depth tree obtained the largest improvement in test accuracy.

At depth 4, the individual trees were strongly constrained, producing an average of only 5.13 leaves with approximately 47 samples per leaf. Their high bias limited classification performance even after aggregation.

At depth 16, individual trees had approximately 17.6 leaves and 13.66 samples per leaf. Bagging reduced their variance by 90.31%, improving mean accuracy from 27.38% to 61.29%.

For unrestricted trees, the average number of leaves increased to 40, while mean leaf occupancy fell to six. These trees had higher variance but lower squared bias. Bagging reduced their variance by approximately 95.97%, giving a smaller accuracy gain than at depth 16, although unrestricted bagging achieved the highest absolute accuracy of 77.83%.

Mean pairwise disagreement among bagged trees decreased from 0.9482 at depth 4 to 0.8931 at depth 16 and 0.8358 at unrestricted depth.

Thus, neither maximum variance reduction nor maximum disagreement necessarily produces the largest improvement in classification accuracy. The final benefit also depends on the predictive quality and bias of the base learners.

## Q3(d): Final Model Comparison

The following table reports evaluation of the frozen models trained on the original fixed split, alongside squared bias and variance measured across 30 refits.

| Model | Clean Test Accuracy | Masked Test Accuracy | Bias² | Variance |
|---|---:|---:|---:|---:|
| Single Tree | 60.00% | 60.00% | 0.3550 | 0.6367 |
| Bagging | 91.25% | 65.00% | 0.5246 | 0.0257 |
| SAMME | 31.25% | 26.25% | 0.8824 | 0.0169 |

### Which method is most stable?

Under the assignment's class-vector variance definition, SAMME is the most stable, with variance 0.0169. Bagging follows closely with variance 0.0257, whereas the single tree has much larger variance, 0.6367.

However, accuracy stability gives a different picture. Bagging's test-accuracy standard deviation across the 30 refits is 3.92 percentage points, compared with 8.05 percentage points for SAMME. Thus, stability depends on whether we measure changes in the score vectors or changes in classification accuracy.

### Which method is most sensitive to occlusion?

The bagged model experiences the largest absolute reduction in test accuracy, from 91.25% to 65.00%, a decline of 26.25 percentage points.

The single tree remains at 60.00% accuracy on both sets, whereas SAMME decreases from 31.25% to 26.25%.

Although bagging experiences the largest decrease, it remains the most accurate method on masked images.

These results measure sensitivity under one specific central 16×16 occlusion pattern; they do not establish robustness to every possible image corruption.

### Does minimum variance imply maximum accuracy?

No.

SAMME has the lowest class-vector variance at 0.0169 but achieves only 31.25% clean test accuracy. Bagging has slightly higher variance at 0.0257 but reaches 91.25% clean test accuracy.

SAMME also has considerably higher squared bias, indicating that its average score vectors differ substantially from the true one-hot targets.

Therefore, variance must be interpreted together with bias and predictive accuracy. Low variance is not sufficient to guarantee successful generalization.

## Conclusion

The experiments show that individual decision trees are sensitive to training-set resampling, especially when allowed to grow deeply.

Bagging consistently reduces prediction variance, substantially improves recognition accuracy, and performs best among the evaluated models.

SAMME concentrates training weights on difficult examples but exhibits higher squared bias and poorer classification accuracy in the tested configurations.

The results demonstrate that ensemble performance depends on both the stability and predictive quality of individual learners, rather than variance reduction alone.