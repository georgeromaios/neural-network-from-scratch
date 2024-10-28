
inputs = [1,2,3,2.5]

weight1 = [0.2,0.8,-0.5,1.0]
weight2 = [0.5,-0.91,0.26,-0.5]
weight3 = [-0.26,-0.27,0.17,0.87]

bias1 = 2
bias2 = 3
bias3 = 0.5

res1  = 0
res2 = 0
res3 = 0



for i in range(len(inputs)):
    product1 = inputs[i] * weight1[i]
    product2 = inputs[i] * weight2[i]
    product3 = inputs[i] * weight3[i]
    res1 +=product1
    res2 +=product2
    res3 +=product3

output = [res1 + bias1, res2 + bias2,res3 + bias3]
print (output)
