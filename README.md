# Machine-Learning-1

## MNIST1.py 결과
![image](https://user-images.githubusercontent.com/24853452/103199244-c4db6280-492d-11eb-8258-c49f9bfbb427.png)

255에 가까울수록 흰색
0에 가까울수록 검은색
-> plt.pcolor(255-img)

## MNIST_FNN.py 결과
![image](https://user-images.githubusercontent.com/24853452/103203660-c2323a80-4938-11eb-93fb-5484efb426e4.png)

전체 2000개의 데이터가 있고, epochs=20, batch_size(한번에 네트워크에 넘겨주는 데이터 수, 1회 갱신에 사용하는 데이터 크기)=500일 때, <br/>
1 epoch = 각 데이터의 (size가 500인) batch가 들어간 (4번의 iteration) <br/>
전체 데이터셋에 대해서는 20번 학습한다. <br/>
batch size가 커지면 한번에 많은 양을 학습할 수 있으나, 메오리에 문제가 생긴다. <br/>
batch 를 사용 -> 확률적 경사 하강법 <br/>
기존의 경사하강법에 비해 얕은 곳의 극소치라면 탈출할 가능성이 있다. <br/>
Local Minima는 경사 하강법을 사용할 때 빠질 수 있는 함정이다. 경사가 0이 될 수 있는 최솟점이 여러 개 있으면, 학습을 시작한 위치에 따라 Global Minimum에 도달하지 못할 수 있다. <br/> 


따라서 batch를 활용해서 **확률적 경사 하강법**을 사용!
![image](https://user-images.githubusercontent.com/24853452/103205262-84371580-493c-11eb-9cf2-75bc3c436468.png)

![image](https://user-images.githubusercontent.com/24853452/103201754-040cb200-4934-11eb-97c0-ba199782167b.png)

### ReLU Activation Function
시그모이드 함수는 입력 x가 어느정도 커지면 항상 1에 가까운 값을 출력하기 때문에, 입력의 변화가 출력에 반영되기 어렵다.<br/>
그 결과, 오차함수의 가중치 매개 변수에 대한 편미분이 0에 가까운 값이 되어, 경사하강법이 늦어지는 문제점 발생

![image](https://user-images.githubusercontent.com/24853452/103206259-cc573780-493e-11eb-9205-b7404cb76831.png)

또한 max(0,x)로 쉽게 나타낼 수 있으므로 계산이 빠르다. 중간층의 activation을 sigmoid에서 relu로 바꾸면 다음과 같은 결과가 나온다.<br/>
**정확도 향상!**

![image](https://user-images.githubusercontent.com/24853452/103205419-db3cea80-493c-11eb-8a35-7ffe5ed93d97.png)

![image](https://user-images.githubusercontent.com/24853452/103205357-b47eb400-493c-11eb-8fcf-8fc83d33784f.png)

![image](https://user-images.githubusercontent.com/24853452/103205379-c4969380-493c-11eb-9441-e28929bdafc5.png)
