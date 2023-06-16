## Session-7
<hr>

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-397/) [![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-green.svg)](https://pytorch.org/) [![torchvision 0.15+](https://img.shields.io/badge/torchvision-0.15+-blue.svg)](https://pypi.org/project/torchvision/) [![torch-summary 1.4](https://img.shields.io/badge/torchsummary-1.4+-green.svg)](https://pypi.org/project/torch-summary/)


**CODE-1**

    (Params - 6.3M) Base architecture (no limit on parameters) - No augmentation, BN, regularization, Variable LR
    Target:
    *   Basic code setup with Data Loaders / Sample Architecture / Predictions + Validation
    *   No check of parameters or augumenters taken care at this step

    Results:
    *   Epochs - 15
    *   Parameters - 6.3M
    *   Best Train Acc -  99.86(LAST epoch)
    *   Best Test Acc  - 99.14(LAST epoch)
    *   Train/Test Acc last layer - (Difference - 0.72)

    Analysis:
    *   Training logs shows sign of overfitting.
    *   The model is large in terms of capacity(parameter count) making it more complex.
    *   Since there are no transforms being done trained model becomes biased to train images which might not be actual representation of true conditions.



**CODE-2**

    Target:
    1.   **Restructuring the architecture** with Conv block and transition blocks.
    2.   Make model bit light on trainable parameters as the data is MNIST which is very small.

    Results:
    1.   Parameters: 54416
    2.   Best Train Acc: 99.00
    3.   Best Test Acc: 98.89
    4.   Difference : 0.11
    5.   Epoch - 15

    Analysis:
    1.   Model still overfits with huge gap. Need to introduce regularization.
    2.   The model is less complex with reduction in kernel values.
    3.   After reducing the intermediate kernel values, model still is performing good.

**CODE-3**

    Target:
    1.   Training the model under **less iterations** and restructuring the architecture for **less trainiable parameters** or complexity.


    Results:
    1.   Parameters: 10546
    2.   Best Train Acc: 98.99(15th epoch)
    3.   Best Test Acc: 98.84(in 15 th epoch)
    4.   Difference : More train accuracy - Small overfit scenario
    5.   Epoch - 15


    Analysis:
    1.   Model looks a bit overfit.
    2.   Architecture looks simple overall and can be improved upon to attain good accuracy and prevent overfitting.


**CODE-4**

    Target:
    1.   Introduce **Batch Normalization** to increase model's efficiency and help it to converge faster.
    Also it provides some regularization.

    Results:
    1.   Parameters: 6288
    2.   Best Train Acc: 99.69(last epoch)
    3.   Best Test Acc: 99.20(last epoch)
    4.   Difference : More train accuracy
    5.   Epoch - 15

    Analysis:
    1.   Model again overfits as compared to the last model
    2.   Need to introduce dropout to bridge the gap between the train and test accuracy

**CODE-5**

    Target:
    1.   Introduce **Dropout** to decrease overfitting.
    2.   Apply different dropout rates as per kernels in each conv layer.

    Results:
    1.   Parameters: 6288
    2.   Best Train Acc: 99.22
    3.   Best Test Acc: 99.34
    4.   Difference : More test accuracy
    5.   Epoch - 15

    Analysis:
    1.   Model seems to be stable with no overfitting.
    2.   Model accuracy must be tuned by trying different learning rates or changing architecture with more feature extractors.


**CODE-6**

    Target:
    1.   Improve Model accuracy to reach stable test accuracy over 99.4
    2.   Added GAP to remove the last layer

    Results:
    1.   Parameters: 5328
    2.   Best Train Acc: 99.19(last Epoch)
    3.   Best Test Acc: 99.22(last epoch)
    4.   Difference : No overfitting
    5.   Epoch - 15

    Analysis:
    1.   Model test accuracy is stuck around 99.20 which has to be further improved using adding augumentation

**CODE-7**

    Target:
    1.   Improve Model accuracy to reach stable test accuracy over 99.4
    2.   Use LR Scheduler to add different learning rates to each Epoch.
    3.   Add augmentation to make model learn different test data.
    3.   Change the architecture to make it more complex.

    Results:
    1.   Parameters: 7592
    2.   Best Train Acc: 98.90(last epoch)
    3.   Best Test Acc: 99.47(12th epoch) / 99.45(13th epoch) / 99.41(14th epoch) / 99.44(15th epoch)
    4.   Difference : No overfitting
    5.   Epoch - 15

    Analysis:
    1.   Model gives a consistent test accuracy above 99.40 which seems to be a good sign.
    2.   We see a bit of underfitting of the model as test score is high.