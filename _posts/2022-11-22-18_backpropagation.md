---
title:  "backpropagation"
excerpt: “역전파 학습법 연습”

categories:
  - DL
tags:
  - python
  - tensor flow
  - backpropagation
---

## 역전파 학습법을 이용한 심층 신경망 학습


```python
import time
import numpy as np
```

## 유틸리티 함수


```python
def _t(x):
    return np.transpose(x)

def _m(A, B):
    return np.matmul(A, B)
```

## Sigmoid 구현


```python
class Sigmoid:
    def __init__(self):
        self.last_o = 1

    def __call__(self, x):
        self.last_o = 1 / (1.0 + np.exp(-x))
        return self.last_o

    def grad(self): # sigmoid(x)(1-sigmoid(x))
        return self.last_o * (1 - self.last_o)
```

## Mean Squared Error 구현


```python
class MeanSquaredError:
    def __init__(self):
        # gradient
        self.dh = 1
        self.last_diff = 1        

    def __call__(self, h, y): # 1/2 * mean ((h - y)^2)
        self.last_diff = h - y
        return 1 / 2 * np.mean(np.square(h - y))

    def grad(self): # h - y
        return self.last_diff
```

## 뉴런 구현


```python
class Neuron:
    def __init__(self, W, b, a_obj):
        # Model parameters
        self.W = W
        self.b = b
        self.a = a_obj()
        
        # gradient
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.dh = np.zeros_like(_t(self.W))
        
        self.last_x = np.zeros((self.W.shape[0]))
        self.last_h = np.zeros((self.W.shape[1]))

    def __call__(self, x):
        self.last_x = x
        self.last_h = _m(_t(self.W), x) + self.b
        return self.a(self.last_h)

    def grad(self): # dy/dh = W
        return self.W * self.a.grad()

    def grad_W(self, dh):
        grad = np.ones_like(self.W)
        grad_a = self.a.grad()
        for j in range(grad.shape[1]): # dy/dw = x
            grad[:, j] = dh[j] * grad_a[j] * self.last_x
        return grad

    def grad_b(self, dh): # dy/dh = 1
        return dh * self.a.grad()
```

## 심층신경망 구현


```python
class DNN:
    def __init__(self, hidden_depth, num_neuron, input, output, activation=Sigmoid):
        def init_var(i, o):
            return np.random.normal(0.0, 0.01, (i, o)), np.zeros((o,))

        self.sequence = list()
        # First hidden layer
        W, b = init_var(input, num_neuron)
        self.sequence.append(Neuron(W, b, activation))

        # Hidden Layers
        for index in range(hidden_depth):
            W, b = init_var(num_neuron, num_neuron)
            self.sequence.append(Neuron(W, b, activation))

        # Output Layer
        W, b = init_var(num_neuron, output)
        self.sequence.append(Neuron(W, b, activation))

    def __call__(self, x):
        for layer in self.sequence:
            x = layer(x)
        return x

    def calc_gradient(self, loss_obj):
        loss_obj.dh = loss_obj.grad()
        self.sequence.append(loss_obj)
        
        # back-prop loop
        for i in range(len(self.sequence) - 1, 0, -1):
            l1 = self.sequence[i]
            l0 = self.sequence[i - 1]
            
            l0.dh = _m(l0.grad(), l1.dh)
            l0.dW = l0.grad_W(l1.dh)
            l0.db = l0.grad_b(l1.dh)
        
        self.sequence.remove(loss_obj)
```

## 경사하강 학습법


```python
def gradient_descent(network, x, y, loss_obj, alpha=0.01):
    loss = loss_obj(network(x), y)  # Forward inference
    network.calc_gradient(loss_obj)  # Back-propagation
    for layer in network.sequence:
        layer.W += -alpha * layer.dW
        layer.b += -alpha * layer.db
    return loss
```

## 동작 테스트


```python
x = np.random.normal(0.0, 1.0, (10,))
y = np.random.normal(0.0, 1.0, (2,))

t = time.time()
dnn = DNN(hidden_depth=5, num_neuron=32, input=10, output=2, activation=Sigmoid)
loss_obj = MeanSquaredError()
for epoch in range(100):
    loss = gradient_descent(dnn, x, y, loss_obj, alpha=0.01)
    print('Epoch {}: Test loss {}'.format(epoch, loss))
print('{} seconds elapsed.'.format(time.time() - t))
```

    Epoch 0: Test loss 0.25462107925756766
    Epoch 1: Test loss 0.2517712355982733
    Epoch 2: Test loss 0.24895288775706292
    Epoch 3: Test loss 0.2461664771772237
    Epoch 4: Test loss 0.24341240010360718
    Epoch 5: Test loss 0.2406910082614006
    Epoch 6: Test loss 0.23800260966241685
    Epoch 7: Test loss 0.23534746952833618
    Epoch 8: Test loss 0.23272581132009612
    Epoch 9: Test loss 0.23013781786250886
    Epoch 10: Test loss 0.22758363255318403
    Epoch 11: Test loss 0.22506336064493232
    Epoch 12: Test loss 0.22257707059102344
    Epoch 13: Test loss 0.22012479544294816
    Epoch 14: Test loss 0.21770653429068848
    Epoch 15: Test loss 0.21532225373591485
    Epoch 16: Test loss 0.2129718893889972
    Epoch 17: Test loss 0.21065534738122918
    Epoch 18: Test loss 0.20837250588420608
    Epoch 19: Test loss 0.20612321662886585
    Epoch 20: Test loss 0.2039073064172843
    Epoch 21: Test loss 0.2017245786209046
    Epoch 22: Test loss 0.19957481465947063
    Epoch 23: Test loss 0.1974577754555169
    Epoch 24: Test loss 0.19537320285984094
    Epoch 25: Test loss 0.19332082104393503
    Epoch 26: Test loss 0.19130033785589448
    Epoch 27: Test loss 0.18931144613682466
    Epoch 28: Test loss 0.18735382499525874
    Epoch 29: Test loss 0.18542714103755076
    Epoch 30: Test loss 0.1835310495526379
    Epoch 31: Test loss 0.18166519564996125
    Epoch 32: Test loss 0.17982921534970017
    Epoch 33: Test loss 0.1780227366248097
    Epoch 34: Test loss 0.17624538039465631
    Epoch 35: Test loss 0.17449676147032056
    Epoch 36: Test loss 0.1727764894518813
    Epoch 37: Test loss 0.17108416957821546
    Epoch 38: Test loss 0.16941940353003881
    Epoch 39: Test loss 0.16778179018707864
    Epoch 40: Test loss 0.16617092634041522
    Epoch 41: Test loss 0.1645864073611494
    Epoch 42: Test loss 0.16302782782665404
    Epoch 43: Test loss 0.16149478210575124
    Epoch 44: Test loss 0.15998686490422173
    Epoch 45: Test loss 0.1585036717721018
    Epoch 46: Test loss 0.15704479957426132
    Epoch 47: Test loss 0.15560984692577462
    Epoch 48: Test loss 0.15419841459361328
    Epoch 49: Test loss 0.1528101058661851
    Epoch 50: Test loss 0.15144452689224042
    Epoch 51: Test loss 0.15010128699064926
    Epoch 52: Test loss 0.14877999893253188
    Epoch 53: Test loss 0.1474802791971954
    Epoch 54: Test loss 0.1462017482032993
    Epoch 55: Test loss 0.1449440305166339
    Epoch 56: Test loss 0.14370675503585553
    Epoch 57: Test loss 0.14248955515748085
    Epoch 58: Test loss 0.14129206892139654
    Epoch 59: Test loss 0.1401139391380961
    Epoch 60: Test loss 0.13895481349880487
    Epoch 61: Test loss 0.13781434466961134
    Epoch 62: Test loss 0.13669219037066935
    Epoch 63: Test loss 0.13558801344149232
    Epoch 64: Test loss 0.13450148189331
    Epoch 65: Test loss 0.13343226894941204
    Epoch 66: Test loss 0.13238005307435802
    Epoch 67: Test loss 0.13134451799288452
    Epoch 68: Test loss 0.13032535269930048
    Epoch 69: Test loss 0.1293222514581155
    Epoch 70: Test loss 0.12833491379660622
    Epoch 71: Test loss 0.12736304448998442
    Epoch 72: Test loss 0.1264063535397938
    Epoch 73: Test loss 0.1254645561461224
    Epoch 74: Test loss 0.12453737267418508
    Epoch 75: Test loss 0.12362452861579422
    Epoch 76: Test loss 0.12272575454620444
    Epoch 77: Test loss 0.12184078607678706
    Epoch 78: Test loss 0.12096936380395937
    Epoch 79: Test loss 0.12011123325476622
    Epoch 80: Test loss 0.11926614482948392
    Epoch 81: Test loss 0.11843385374159232
    Epoch 82: Test loss 0.11761411995543616
    Epoch 83: Test loss 0.11680670812187291
    Epoch 84: Test loss 0.1160113875121862
    Epoch 85: Test loss 0.11522793195051964
    Epoch 86: Test loss 0.11445611974507007
    Epoch 87: Test loss 0.11369573361825944
    Epoch 88: Test loss 0.1129465606360879
    Epoch 89: Test loss 0.11220839213685596
    Epoch 90: Test loss 0.11148102365942711
    Epoch 91: Test loss 0.11076425487118914
    Epoch 92: Test loss 0.11005788949586019
    Epoch 93: Test loss 0.10936173524127069
    Epoch 94: Test loss 0.10867560372724394
    Epoch 95: Test loss 0.10799931041368625
    Epoch 96: Test loss 0.10733267452898573
    Epoch 97: Test loss 0.10667551899881286
    Epoch 98: Test loss 0.10602767037540498
    Epoch 99: Test loss 0.10538895876740907
    0.35006213188171387 seconds elapsed.



```python

```
