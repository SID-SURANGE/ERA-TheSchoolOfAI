# Session 6 Part1 - ERA-TheSchoolOfAI

### **Sample model architecture**

![lrone](./images/sample_model.png)


<p>

### **Weight updation process**

Below steps show the weight updation process using backpropagation

**STEP 1**

        h1 = w1*i1 + w2*i2		            # 1) first we calculate the h1 neuron value using weights w1 w2 and inputs i1, i2
        h2 = w3*i1 + w4*i2		            # 2) then we calculate the h2 neuron value using weights w3 w4 and inputs i1, i2
        a_h1 = σ(h1) = 1/(1 + exp(-h1))	     # 3) next we calculate the activated h1 neuron value which is sigmoid of h1 neuron
        a_h2 = σ(h2)		             # 4) similarly activated h2 neuron is calculated using sigmoid of h2 neuron
        o1 = w5*a_h1 + w6*a_h2		     # 5) we next derive o1 using weights w5, w6 and inputs ah1, ah2
        o2 = w7*a_h1 + w8*a_h2		     # 6) we next derive o2 using weights w7, w8 and inputs ah1, ah2
        a_o1 = σ(o1)		             # 7) next we calculate the activated ao1 neuron value which is sigmoid of o1 neuron
        a_o2 = σ(o2)		             # 8) then we calculate the activated ao2 neuron value which is sigmoid of o2 neuron
        E_total = E1 + E2		     # 9) ao1 and ao2 can help us calculate the total error by the model
        E1 = ½ * (t1 - a_o1)²		     # 10) the E1 in equation above is calculated using sqaure of diff between truth t1 and predicted ao1
        E2 = ½ * (t2 - a_o2)²		     # 11) the E2 in equation above is calculated using sqaure of diff between truth t2 and predicted ao2

**STEP 2**

        ∂E_total/∂w5 = ∂(E1 + E2)/∂w5					
        ∂E_total/∂w5 = ∂E1/∂w5					
        ∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5					
        ∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)					
        ∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)					
        ∂o1/∂w5 = a_h1					

**STEP 3**

        ∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1					
        ∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2					
        ∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1					
        ∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2					


**STEP 4**

        ∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5								
        ∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7								
        ∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7								
        ∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8								


**STEP 5**

        ∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1					
        ∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2					
        ∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3					


**STEP 6**

        ∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1												
        ∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2												
        ∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1												
        ∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2												

</p>


<hr>

### **Initial weights as per excel**

![lrone1](./images/initial_weights.jpg)



<hr>

## **Loss Curve at various LR values**


### **Loss Curve at LR = 0.1**
![lrone11](./images/lr_point1.png)


### **Loss Curve at LR = 0.2**
![lrone](./images/lr_point2.png)


### **Loss Curve at LR = 0.5**
![lrone](./images/lr_point5.png)


### **Loss Curve at LR = 0.8**
![lrone](./images/lr_point8.png)


### **Loss Curve at LR = 1.0**
![lrone](./images/lr_one.png)


### **Loss Curve at LR = 2.0**
![lrone](./images/lr_two.png)

