<h4> Session 7 </h4>

<h3><i><b>Analysis of CODE1</b></i></h2>
<I>

---


**Target:**

*   Basic code setup with Data Loaders / Sample Architecture / Predictions + Validation
*   No check of parameters or augumenters taken care at this step


**Results:**<br>

*   Epochs - 15
*   Parameters - 6.3M
*   Best Train Acc -  99.86(LAST epoch)
*   Best Test Acc  - 99.14(LAST epoch)
*   Train/Test Acc last layer - (Difference - 0.72)


**Analysis:**<br>


*   Training logs shows sign of overfitting.
*   The model is large in terms of capacity(parameter count) making it more complex.
*   Since there are no transforms being done trained model becomes biased to train images which might not be actual representation of true conditions.


---
