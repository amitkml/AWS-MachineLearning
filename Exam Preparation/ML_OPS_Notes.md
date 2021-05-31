# Machine Learning in Production

[TOC]

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







