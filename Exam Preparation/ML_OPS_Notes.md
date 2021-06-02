# Machine Learning in Production

[TOC]

# Week -1

## ML OPS

Prediction server can be hosted into

- Cloud
- Edge

**Deployment can be subjected to**

- concept drift/data drift (Lets say lighting condition at the factory where deployment there is different and hence now model is not able to detect.)

![img](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/deployment_architecture.JPG?raw=true)

![MLInfra](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/ML_Infratructure.JPG?raw=true)

## Steps of a ML Project

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/ML_Project_lifecycle.JPG?raw=true)

### Case study: speech recognition ML Project

#### Scoping

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/speech_scoping.JPG?raw=true)

#### Data

Now, we need to think about data for the project. **One of the problem with data is that, consistency of data labelling**.

- how we will handle when each of the speaker volumes or pronunciations are different?
- How we will handle labelling consistently

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/speech_data_collection.JPG?raw=true)

#### Modelling

The following slide says that there are two approaches

- **Research driven/Model Driven**
  - Research was driven by researchers working to improve performance on benchmark data set. In that model, researchers might download the data set and just work on that fixed data set.
  - Lot of research work or academic work you tend to hold the data fixed and vary the code and may be vary the hyper parameters in order to try to get good performance
-  **Product teams/Data Driven**
  -  It can be even more effective to hold the code fixed and to instead focus on optimizing the data and maybe the hyper parameters. In order to get a high performing model, A machine learning system includes both codes and data and also hyper parameters that there maybe a bit easier to optimize than the code or data.
  -  Rather than taking a model centric view of trying to optimize the code to your fixed data set for many problems, you can use an open source implementation of something you download of GIT hub and instead *just focus on optimizing the data*.
  - Work on data optimization first based on error analysis. Sometime, we might need to work on model also which is not very common though.

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/speech_modelling.JPG?raw=true)

#### Deployment

> Monitoring is key for such deployment to ensure we can detect data drift. **One classic example is that a speech recognition system tuned with adult voice may start deteriorating when teenager used the application.** 
>
> So, we need to have a appropriate monitoring to detect such problem and then we should know how to detect such problem.
>
> **Concept Drift**
>
> - When relationship between Input and output changes
>
> **Data Drift**
>
> - When data distribution pattern gets changes

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/speech_deployment.JPG?raw=true)

##### Software Engineering issues

*checklist of questions*

- Batch or real-time?
- Cloud vs Edge/Browser
- Compute resource availability (CPU/GPU/Memory)
- Latency and throughput (Query per second - QPS) requirements
- Logging requirements for analysis, review and retraining
- security and privacy requirements

## Deployment patterns

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/commom_deployment.JPG?raw=true)

First, we always run Shadow Mode and then Canary Mode.

### Shadow Mode

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/shadow_mode_deployment.JPG?raw=true)

### Canary Mode

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/canary_deployment.JPG?raw=true)

### Blue Green Deployment

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/blue-green-deployment.JPG?raw=true)

## Monitoring

### Dashboard

- always set threshold for such alarms to teams

- adapt threshold gradually

   

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/monitoring_dashboard.JPG?raw=true)

### Deployment Monitoring

- Manual Retraining

- Automatic Retraining (Difficult and Uncommon)

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/Capture.JPG?raw=true)

## Pipeline monitoring

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/pipeline-monitoring.JPG?raw=true)

## Week 1: Overview of the ML Lifecycle and Deployment

If you wish to dive more deeply into the topics covered this week, feel free to check out these optional references. You won’t have to read these to complete this week’s practice quizzes.

[Concept and Data Drift](https://towardsdatascience.com/machine-learning-in-production-why-you-should-care-about-data-and-concept-drift-d96d0bc907fb)

[Monitoring ML Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)

[A Chat with Andrew on MLOps: From Model-centric to Data-centric](https://youtu.be/06-AZXmwHjo)



**Papers**

Konstantinos, Katsiapis, Karmarkar, A., Altay, A., Zaks, A., Polyzotis, N., … Li, Z. (2020). Towards ML Engineering: A brief history of TensorFlow Extended (TFX). [http://arxiv.org/abs/2010.02013 ](http://arxiv.org/abs/2010.02013)

Paleyes, A., Urma, R.-G., & Lawrence, N. D. (2020). Challenges in deploying machine learning: A survey of case studies. <http://arxiv.org/abs/2011.09926>

Sculley, D., Holt, G., Golovin, D., Davydov, E., & Phillips, T. (n.d.). Hidden technical debt in machine learning systems. Retrieved April 28, 2021, from Nips.c[ https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)

# Week -2



## Modelling Overview

- Model centric dvelopment
- Data centric development

For practical projects, it can be even more useful to take a more data-centric approach, where you focus not just on improving the neural network architecture, but on making sure you are feeding your algorithm high-quality data. That ultimately lets you be more efficient in getting your system to perform well. But the way I engage in data-centric AI development is not to just go and try to collect more data, which can be very time-consuming, but to instead use tools to help me improve the data in the most efficient possible way.

