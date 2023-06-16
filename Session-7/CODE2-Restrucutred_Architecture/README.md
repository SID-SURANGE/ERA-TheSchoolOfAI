<h4> Session 7 </h4>

<h3><i><b>Analysis of CODE2</b></i></h2>
<I>

---


**Target:**
1.   **Restructuring the architecture** with Conv block and transition blocks.
2.   Make model bit light on trainable parameters as the data is MNIST which is very small.


**Results:**
1.   Parameters: 54416
2.   Best Train Acc: 99.00
3.   Best Test Acc: 98.89
4.   Difference : 0.11
5.   Epoch - 15


**Analysis:**
1.   Model still overfits with huge gap. Need to introduce regularization.
2.   The model is less complex with reduction in kernel values.
3.   After reducing the intermediate kernel values, model still is performing good.

---
