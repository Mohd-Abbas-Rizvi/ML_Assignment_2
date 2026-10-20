# Question 3. Bias–Variance Analysis

## Q3(a): Bias and Variance

I ran the experiment with thirty resampled training sets.

Each training set was built from six bootstrap draws for every one of the forty identities.

I kept the validation and test sets unchanged.

For every refit each model produced a forty‑dimensional vector on the same eighty clean test images.

| Model | Bias² | Variance | Mean Test Accuracy | Accuracy Std. Dev. |

|---|---:|---:|---:|---:|

| Single Tree | 0.3550 | 0.6367 | 50.42% | 4.44 pp |

| Bagging | 0.5246 | 0.0257 | 77.83% | 3.92 pp|

| SAMME | 0.8824 | 0.0169 | 33.79% 8.05 pp |

The single tree showed the largest class‑vector variance.

Bagging cut variance by a margin and also achieved the highest mean test accuracy.

SAMME had the measured variance but the biggest squared bias showing that low variance alone does not guarantee good generalization.

## Q3(b): Effect of Tree Depth

The experiments tested maximum‑depth settings of four, sixteen and none (unrestricted).

I used min_samples_leaf to four.

All depth levels used the thirty resampled training sets and each bagged ensemble contained one hundred trees.

| Depth | Method | Bias² | Variance | Mean Accuracy |

|---|---|---:|---:|---:|

| 4 | Single Tree | 0.8595 | 0.0643 | 9.79% |

| 4 | Bagging | 0.8960 | 0.0132 | 27.04% |

| 16 | Single Tree | 0.6509 | 0.2237 27.38% |

| 16 | Bagging | 0.6764 | 0.0217 | 61.29% |

| None | Single Tree | 0.3550 | 0.6367 50.42% |

| None | Bagging | 0.5246 | 0.0257 | 77.83% |

Increasing the allowed depth lowered the single tree’s squared bias but raised its variance.

Bagging cut variance at all three depths.

I chose a baseline with max_depth set to none meaning no depth limit.

Because no depth can be deeper than growth I compared depths four and sixteen with this unrestricted setting.

I acknowledge this limitation of claiming that a depth greater than none was tested.

## Q3(c): Does the Deepest Tree Give the Largest Improvement from Bagging?

No not when the improvement is measured by the gain in test accuracy.

Depth | Variance Reduction | Accuracy Gain |

|---|---:|---:|

| 4 | 79.53% | percentage points |

| 16 | 90.31% +33.92 percentage points |

| None | 95.97% | +27.42 percentage points |

The unrestricted tree achieved the largest absolute and relative variance reduction yet the intermediate‑depth tree achieved the largest improvement in test accuracy.

At depth four the individual trees were strongly constrained, producing an average of five‑point‑one‑three leaves with approximately forty‑seven samples per leaf. Their bias limited classification performance even after aggregation.

At depth sixteen individual trees had seventeen‑point‑six leaves and thirteen‑point‑six‑six samples per leaf. Bagging reduced their variance by ninety‑point‑three percent improving accuracy from twenty‑seven‑point‑three‑eight percent to sixty‑one‑point‑two‑nine percent.

For trees the average number of leaves increased to forty while mean leaf occupancy fell to six. These trees had variance but lower squared bias. Bagging reduced their variance by ninety‑five‑point‑nine‑seven percent giving a smaller accuracy gain than at depth sixteen although unrestricted bagging achieved the highest absolute accuracy of seventy‑seven‑point‑eight‑three percent.

Mean pairwise disagreement among trees decreased from 0.9482 at depth four to 0.8931 at depth sixteen and 0.8358 at unrestricted depth.

Thus neither maximum variance reduction nor maximum disagreement necessarily produces the improvement in classification accuracy. The final benefit also depends on the quality and bias of the base learners.

## Q3(d): Final Model Comparison

The following table reports evaluation of the models trained on the original fixed split alongside squared bias and variance measured across thirty refits.

| Model | Clean Test Accuracy Masked Test Accuracy | Bias² | Variance |

|---|---:|---:|---:|---:|

| Single Tree | 60.00% | 60.00% | 0.3550 | 0.6367 |

|Bagging | 91.25% | 65.00% | 0.5246 | 0.0257 |

| SAMME | 31.25% | 26.25% | 0.8824 | 0.0169 |

### Which method is most stable?

Under the assignments class‑vector variance definition SAMME is the stable, with variance 0.0169.

Bagging follows closely with variance 0.0257 whereas the single tree has a larger variance, 0.6367.

However accuracy stability gives a picture. Bagging has a test‑accuracy deviation of 3.92 percentage points over 30 refits while SAMME has 8.05 percentage points. Thus stability depends on whether we measure changes in the score vectors or changes in classification accuracy.

### Which method is most sensitive to occlusion?

The bagged model sees the absolute drop in test accuracy falling from 91.25% to 65.00% a decline of 26.25 percentage points.

The single tree stays at 60.00% accuracy on both sets while SAMME drops from 31.25% to 26.25%.

Although bagging sees the drop bagging remains the most accurate method on masked images.

These results measure sensitivity under one central 16×16 occlusion pattern; they do not prove robustness to every possible image corruption.

### Does minimum variance imply accuracy?

No.

SAMME has the class‑vector variance at 0.0169 but achieves only 31.25% clean test accuracy. Bagging has higher variance at 0.0257 but reaches 91.25% clean test accuracy.

SAMME also has higher squared bias showing that its average score vectors differ a lot from the true one‑hot targets.

Therefore variance must be read together with bias and predictive accuracy. Low variance alone does not guarantee generalization.

##

The experiments show that individual decision trees are sensitive, to training‑set resampling, especially when they grow deeply.

Bagging consistently cuts prediction variance boosts recognition accuracy a lot and comes out on top among the models we looked at.

SAMME focuses training weight on examples but shows higher squared bias and lower classification accuracy in the tests we ran.

The results show that ensemble performance relies on both stability and predictive quality of each learner not on variance reduction.