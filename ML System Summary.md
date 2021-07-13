ML System Summary
======================

# 1. Machine Learning, Deep Learning
## 1.1. What is ML,DL?
<img src="https://t1.daumcdn.net/cfile/tistory/999597495E744F711D" width="40%" height="30%" title="%(비율) 크기 설정" alt="BEN"></img>
### 1.1.1 ML(Machine Learning)
ML(Machine Learning)은 경험을 통해 자동으로 개선하는 컴퓨터 알고리즘의 연구이다. ML은 사람이 디자인한 feature 값들을 추출하고, 그 후 의도한 출력 값이 나오도록 유도하는 것이다. 이를 매핑이라하며, ML은 어떤 feature를 사용하는지, 디자인을 얼마나 구현하는지에 따라서 성능이 확연하게 달라진다.
ML은 Data Mining application에 적합하다.

<img src="https://i2.wp.com/semiengineering.com/wp-content/uploads/2018/01/MLvsDL.png?resize=733%2C405&ssl=1" width="40%" height="30%" title="%(비율) 크기 설정" alt="BEN"></img>

### 1.1.2 DL(Deep Learning)
DL(Deep Learning)은 머신러닝의 일부로 볼 수 있다. DL은 Deep Neural Network를 사용하여 Representation을 배우는 것이다. DL은 ML과 달리 사람이 어떤 feature를 추출하는 디자인 과정이 없다. 즉 input에서 특정한 feature를 선택해서 사용하겠다고 배우고, 그 feature에서 최종적인 output을 도출한다. DL에서 이를 매핑이라하며 feature를 스스로 학습하므로 Representation Learning이라고도 부른다.
DL은 어려운 Vision, speech, language 문제에 적합하다.




## 1.2. Structure of ML
<img width="400" alt="su" src="https://user-images.githubusercontent.com/81912557/125304934-f1eee000-e368-11eb-93f8-39ce21a2ad14.PNG">
ML을 이용하기 위해서는 크게 3가지가 필요하다.
첫째, 당연히 많은 양의 Data가 필요하다. 데이터가 있어야 배울 수 있기 때문이다.
둘째, 어떤 것을 배우는지 표현하는 Model이 필요하다. 이 Model에 따라서 추출할 값들이 달라진다.
셋째, Data와 Model을 이용해서 계산하기위해 Computing 할 수 있는 하드웨어와 소포트웨어가 필요하다. 특정 Model을 Data를 가지고 학습하여 최종적으로 원하는 Model을 얻고자 할 때는 computing은 꼭 들어가야 한다.

****
# 2. ML,DL Applications
## 2.1. ML Applications


* Regression

    Regression은 변수간의 관계를 찾아내는 것이다. 
    
    Ex) 건전지 사용량 x와 건전지 성능 y 사이의 관계를 찾는 것.
* Classification

    Classification은 말 그대로 분류하는 것이다. 
    
    Pass/Non-Pass와 같이 이분법적으로 분류하면 Binary Classification, A~F까지 Grade를 매기는 것은 Multi-label Classification이라고 한다.
    
    Ex) SpamFilter와 같이 수신된 메일이 스팸인지 스팸이 아닌지 검출하는 것.
* Clustering

   Clustering은  여러 데이터 포인트들이 있을 때, 더 근접한 데이터들끼리 모아서 클러스터를 만드는 것이다.
* Topic Modeling

    Topic Modeling은 어떤 Topic에 해당하는 것인지 찾아내는 것이다.
    
    Ex) 많은 Article들이 어떤 Topic에 해당하는 것인지 찾아내는 것.
    
* Collaborative Filtering

    Collaborative Filtering은 추천 시스템에서 많이 사용한다.
    
    Ex) 사용자가 과거에 매겼던 다른 영화 평점을 기반으로 해당영화 평점을 예측하는 것. 
    
* Frequent Pattern Mining

    Frequent Pattern Mining은 빈번히 동시에 발생하는 패턴들을 찾아서 보는 것이다.
    
* Ranking

    Ranking은 여러가지 답의 중요도를 매겨서 순서를 나타내는 것이다.

...

## 2.2. DL Applications


*  Image Classification 

    Image Classification은 특정 이미지가 들어왔을 때 Fixel 데이터를 Input으로 인식하여 해당 이미지가 어떤 분류에 속하는지 파악하는 것이다.
    
*  Speech-To-Text, Text-To-Speech

   Speech-To-Text는 Speech Data를 Input으로 받아 해당하는 Data를 Text로 변환하는 것이다.(Text-To-Speech는 이와 반대.)
    
    
*  Video Understanding 

    Video Understanding은  Video Data를 보고 해당하는 Video Data의 내용을 Text 형태로 요약하는 것이다. 
* Image, Video Style Transfer

    Image, Video Style Transfer는 하나의 Image 또는 Video Style에서 다른 형태의 Style로 변형을 하는 것이다. 
    
* Text Understanding - Dialog

   Text Understanding은 사용자가 Bot이라는 System과 대화를 주고 받을 때 해당 발화를 이해해서 지속적인 대화를 가능하게 하는 것이다.
   
* Machine Translation

    Machine Translation은 하나의 언어에서 다른 언어로 번역하는 것이다.
    


*   Ranking Ads, Feeds, News

    Ranking Ads, Feeds, News는 Ads, Feed, News 등 이와 같은 것에서 사용자가 더욱 좋아할만한 것을 Ranking하는 것이다.
    
*  Robot Control, Game Play

    Robot Control, Game Play는 DL을 이용하여 Robot을 Control하거나, Game을 Play하는 것을 말한다.
    
*   Self-Driving Car

    Self-Driving Car는 DL을 이용하여 주행을 스스로 하게 만든 Self-Driving Car를 말한다.

...

****
# 3. ML WorkFlow
## 3.1. Two Main Steps of ML
### 3.1.1 Training
Training은 학습에 해당한다. Training은 ML Model을 계속 반복적으로 학습시켜서 Execption 값을 최적화하기 위해서 수행하는 단계이다. 이를 통해 최종적으로 우리가 의도하는 Model이 만들어진다. 이때 Training에 사용하는 Data는 상당히 많고, 복잡한 Model을 사용하기 때문에 계산이 많이 들어간다.
### 3.1.2 Inference(Prediction)
Inference 또는 Prediction은 추론, 예측에 해당한다. 전 단계인 Training을 통해 도출한 Model을 사용하는 것을 의미하며 이 과정에서 새로운 Input Data가 들어왔을 때 Prediction을 하게된다. Prediction은 속도가 중요하므로 Latency가 매우 강조된다. 


## ○ 참고문서
* [78 Tools for writing and previewing Markdown](http://mashable.com/2013/06/24/markdown-tools/)
* [SEMICONDUCTOR ENGINEERING](https://semiengineering.com/deep-learning-spreads/)
* [Creative&Smart](https://blog.lgcns.com/2212/)
