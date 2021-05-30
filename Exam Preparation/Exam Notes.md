# Notes from AWS ML Whitepapers

[TOC]

![IM](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/ML_Exam.JPG?raw=true)



## Managing ML Projects

### Documenting the Data Catalog and Pipeline

A simple method of identifying potential challenges is to clearly diagram the data pipeline used to build the model, showing where all data is from and how it is transformed. The following is an example of how to annotate the pipeline diagram with this information:

-  Major data cleaning operations performed
- Records dropped, either as an actual count or as a percentage
- Major issues found with the data, for example, “duplicate records found and dropped”
- Assumptions made at the time, for example, “data was extracted for US only, other countries are assumed to be similar”

![img](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/pipeline-error.JPG?raw=true)

characteristics of the source data can also be captured in a Data Catalog table, such as Table 1. This catalog documents the current understanding of the data source, communicates to stakeholders the sources to be used and some basic facts about them, and helps identify potential mismatches, concerns, or clarify misunderstandings. For example, in Table 1, the change in storage format of data source #2 potentially adds a conversion or regularization task to the task list.

| Source                    | Contents                                                     | Duration                       | Quantity                | Comments                                                     |
| ------------------------- | ------------------------------------------------------------ | ------------------------------ | ----------------------- | ------------------------------------------------------------ |
| Data source #1: data lake | Clickstream data                                             | Jan 2018–Jan 2019              | 1.6M                    | User IP address only; user name not known                    |
| Data source #2: data lake | Order history                                                | June 1 2016–Oct 3 2018         | 55k orders              | Format stored in changed on Jan 1 2018
Final order only (not change history)
Orders with errors are deleted |
| Sensor data               | Readings from factory sensors. Streaming data is batched and stored | 90 days history retention only | 50/sec; 5k/sec expected | Data cleaning unknown; is perceived outlier data being dropped? |

You should also create diagrams of the pipeline and data catalog to be used in production. **Make note of the similarities and differences between the two pipelines**.

- If they are the same, then they are likely subject to the same errors. Are these errors important?
- If they are not the same, then different sources, different processing has been applied. Do any of these differences impact the model? How do you know?

The **more data sources that are involved**, the more **disparate the data sources** that are to be merged, and the **more data transformation steps** that are involved, the more complex the data quality challenge becomes.

### Estimating Impact of Data Quality

Frequently, there are many individual cleaning and transformation steps performed before the data is used for ML training.

Examples are:

- When merging data, data might be dropped if no direct key match is found.
- Records with null or extreme values might be dropped.

**How do you ensure that the model’s performance in production will be similar**? Here are two approaches that work together: comparing statistics, and validating the model against unclean data inputs.

### Validate Model Against Unclean Data Inputs

A simple but powerful technique to validate your data model is to take a subset of data that was eliminated during every cleaning or transformation step from the raw data, and compare it to the data eventually used to train the model, and send those items to the ML inference endpoint. Then, assess the resulting inferences.

- Does the endpoint **provide reasonable responses in all cases**? 
- Use the results to identify where checks and error handling should be added. **Should error handling be added to the inference endpoint**?
- Or, should the applications that are calling the inference endpoints be required to identify and **remove problematic inputs, or handle problematic outputs**?

### Assessing Economic Value

- An additional aspect of the ML project is assessing the cost of errors. Implicit with the speed and volume that many ML models address, is that human intervention and oversight that might exist today are removed. 
  - What is the **cost of errors**? 
  - If there is a cost for each error, **how much tolerance is there**, before the economic model ceases to be positive?
- If **model drift occurs**, the number of errors might increase. How serious a problem is that?

As shown in Figure 3, calculating error costs can show that an otherwise well-performing model (based on otherwise good metrics) might not be economically feasible.

![error](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/Impact_Error_Costs.JPG?raw=true?)

An example of such an approach is shown in [Training models with unequal economic error costs using Amazon SageMaker](https://aws.amazon.com/blogs/machine-learning/training-models-with-unequal-economic-error-costs-using-amazon-sagemaker/) .19 Here, the differential costs of errors changes the ML model used to predict breast cancer, producing fewer false negatives (undesirable and expensive) at the cost of more false positives, while still producing a cheaper model overall.

### Using Scorecards to Manage and Mitigate Risk

The sample scorecards are separated into five topics:

- **Project context** – Addresses the social, business, and regulatory environment of the project
- **Financial** – Identifies the costs and benefits of the problem you are trying to solve with ML, and of the ML system you are developing
- **Data quality** – Highlights areas that are frequently problematic in ML projects, and that can easily mislead the project—missing a signal that exists in the data or believing a signal exists where there is none—if not identified and addressed
- **Project processes** – Addresses processes in the ML project that are easily overlooked in the excitement of developing and testing the algorithm and identifying promising results
- **Summary** – Captures the key risk areas to bring to executive attention

## Machine Learning Foundations: Evolution of ML and AI

### AWS and Machine Learning

- The first layer shows AI Services, which are intended for builders creating specific
  solutions that require prediction, recommendation, natural language, speech, vision, or
  other capabilities. These intelligent services are created using machine learning, and
  especially deep learning models, but do not require the developer to have any
  knowledge of machine learning to use them.
- 

![svc](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/AWS-ML-Services.JPG?raw=true)

| AWS Service               | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| Amazon Forecast           | Amazon Forecast is a fully managed service that delivers highly accurate forecasts, and is based on the same technology used at Amazon.com. You provide historical data plus
any additional data that you believe impacts your forecasts. Amazon Forecast examines
the data, identifies what is meaningful and produces a forecasting model. |
| Amazon Personalize        | Amazon Personalize makes it easy for developers to create individualized product and content recommendations for customers using their applications. You provide an activity stream from your application, inventory of items you want to recommend and potential demographic information from your users. Amazon Personalize processes and examines the data, identifies what is meaningful, selects the right algorithms, and trains and optimizes a personalization model. |
| Amazon Lex                | Amazon Lex is a service for building conversational interfaces into any application using voice and text. Amazon Lex provides the advanced deep learning functionalities of automatic speech recognition (ASR) for converting speech to text, and natural language understanding (NLU) to recognize the intent of the text, to enable you to build applications with highly engaging user experiences and lifelike conversational
interactions. With Amazon Lex, the same deep learning technologies that power
Amazon Alexa are now available to any developer, enabling you to quickly |
| Amazon Lex                | Amazon Lex is a service for building conversational interfaces into any application using voice and text. Amazon Lex provides the advanced deep learning functionalities of automatic speech recognition (ASR) for converting speech to text, and natural language understanding (NLU) to recognize the intent of the text, to enable you to build applications with highly engaging user experiences and lifelike conversational
interactions. With Amazon Lex, the same deep learning technologies that power
Amazon Alexa are now available to any developer, enabling you to quickly and easily
build sophisticated, natural language, conversational bots (“chatbots”). |
| Amazon Comprehend         | Amazon Comprehend is a natural language processing (NLP) service that uses machine learning to find insights and relationships in text. Amazon Comprehend
identifies the language of the text; extracts key phrases, places, people, brands, or
events; understands how positive or negative the text is and automatically organizes a
collection of text files by topic. |
| Amazon Comprehend Medical | Amazon Comprehend Medical is a natural language processing service that extracts relevant medical information from unstructured text using advanced machine learning
models. You can use the extracted medical information and their relationships to build
or enhance applications |
| Amazon Translate          | Amazon Translate is a neural machine translation service that delivers fast, high-quality, and affordable language translation. Neural machine translation is a form of language translation automation that uses deep learning models to deliver more accurate and more natural sounding translation than traditional statistical and rule-based translation algorithms. Amazon Translate allows you to localize content - such as websites and applications - for international users, and to easily translate large volumes of text efficiently. |
| Amazon Polly              | Amazon Polly is a service that turns text into lifelike speech, allowing you to create applications that talk, and build entirely new categories of speech-enabled products.
Amazon Polly is a Text-to-Speech service that uses advanced deep learning
technologies to synthesize speech that sounds like a human voice. |
| Amazon Transcribe         | Amazon Transcriber is an automatic speech recognition (ASR) service that makes it easy for developers to add speech-to-text capability to their applications. Using the
Amazon Transcribe API, you can analyze audio files stored in Amazon S3 and have the
service return a text file of the transcribed speech. |
| Amazon Rekognition        | Amazon Rekognition makes it easy to add image and video analysis to your applications. You just provide an image or video to the Rekognition API, and the service
can identify the objects, people, text, scenes, and activities, as well as detect any
inappropriate content. Amazon Rekognition also provides highly accurate facial analysis and facial recognition. You can detect, analyze, and compare faces for a wide variety of user verification, cataloging, people counting, and public safety use cases. |
| Amazon Textract           | Amazon Textract automatically extracts text and data from scanned documents and forms, going beyond simple optical character recognition to identify contents of fields in forms and information stored in tables |

### Amazon SageMaker

- Fully-managed machine learning (ML) service that enables developers and data scientists to quickly and easily build, train, and deploy machine learning models at any scale.

- SageMaker sets up and manages environments for training, and provides hyperparameter optimization with Automatic Model Tuning to make the model as accurate as possible
- SageMaker deployments run models spread across availability zones to deliver high performance and high availability

### Amazon SageMaker GroundTruth

- Helps build data sets quickly and accurately using an active learning model to label data combing machine learning and human interaction to make the model progressively better.

### SageMaker Neo

- Sagemaker Neo allows you to deploy the same trained model to  multiple platforms. Using machine learning, Neo optimizes the performance and size of the model and deploys to edge devices containing the Neo runtime. 

### Amazon EMR/EC2 with Spark/Spark ML

- provides a managed Hadoop framework that makes it easy, fast, and cost-effective to process vast amounts of data across dynamically scalable Amazon EC2 instances. 
- Spark and Spark ML can also be run on Amazon EC2 instances to pre-process data, engineer features or run machine learning models

## Augmented AI: The Power of Human and Machine
### Challenges for agencies dealing with benefits programs

- Customer/Beneficiary experience: Legacy benefits systems often provide poor consumer
  experience as they cannot scale to meet the surge during an open enrollment or
  during a crisis situation. AWS cloud offers multiple options to help address these
  challenges.
- Workforce productivity: Benefits application documents can include federal tax forms, pay stubs, SSN, and etc. These documents are in multiple formats. such as PDFs and images, and are submitted from various sources such as the web, mail-in, and contact centers. The work force spends a significant amount of time to review, process, and validate these documents. AWS offers multiple services including Amazon Augmented AI, to address these challenges through process automation.
- Data scale, size, privacy and security: For example, Healthcare.gov alone handled over 10.7 Million applications during the 2019 open enrollment period. AWS provides multiple storage options
  including the Amazon Simple Storage Service (Amazon S3) to address this challenge.
- Large call volumes:  For example, the Social Security Administration (SSA) handled over 50 Million calls5 during FY-2019. AWS has a number of services including Amazon Connect, and Amazon Lex to enable these capabilities.
- Insights into program operations: Having access to data-driven insights enables agencies to build programs and advocate for innovative policy changes to better serve constituents. AWS provides a number of analytics and AI/ML services to address these challenges.

### High level framework to address challenges with benefits enrollment

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/framework_challenge_framework.JPG?raw=true)

| Outcome                                                     | AWS Service                                                  |
| ----------------------------------------------------------- | ------------------------------------------------------------ |
| Improve beneficiary/customer experience                     | Agencies can enable self-service and automated communications on web, mobile, and through contact centers using AWS services such as **Amazon Pinpoint, Amazon Lex and Amazon Polly**. Using these capabilities, beneficiaries can obtain their case status, execute routine tasks, such as a PIN resets, or obtain general information on claims. |
| Provide centralized storage with Data Lakes                 | There are a variety of options to gather the data from the beneficiaries on claims and applications. These options include standard web, mobile communications combined with AWS services such as **Amazon Kinesis Data Streams and Amazon Kinesis Data Firehose for streaming data, or the AWS Transfer Family service for batch data ingestion and storage into data lakes**.The [Data Lake solution](https://aws.amazon.com/solutions/data-lake-solution/) automatically crawls data sources, identifies data formats, and then suggests schemas and transformations, so you don’t have to spend time handcoding data flows. For example, **if you upload a series of claims and application documents to Amazon S3; AWS Glue, a fully managed extract, transform and load (ETL) tool, can scan these documents to identify the schema and data types present in these files**. This metadata is then stored in a catalog to be used in subsequent transforms and queries |
|                                                             | The [AWS Lake Formation](https://aws.amazon.com/lake-formation/) service builds on the existing data lake solution by enabling you to set up a secure data lake within days. Once you define where your lake is located, Lake Formation collects and catalogs this data, moves the data into Amazon S3 for secure access, and cleans and classifies the data using machine learning algorithms.
Additionally, user-defined tags or meta-data about the documents such as SSN cards, bank statements, driver’s licenses, or other claims data is stored in [Amazon DynamoDB](https://aws.amazon.com/dynamodb/), a key-value document database, to add business-relevant context to each dataset. You can browse available datasets or search on dataset attributes and tags to quickly find and access the documents in S3. |
|                                                             | **To summarize, Amazon S3 combined with AWS Glue and AWS Lake Formation act as a centralized data lake for storing documents from multiple sources with disparate data formats. Amazon DynamoDB provides fast access to these documents by storing the document meta-data (e.g. claimant ID, document storage location in S3, etc.).** |
| Extract relevant information from application documents     | Amazon Textract can help **extract text** and dat a from scanned documents and images without the need for any custom coding; Amazon Rekognition can be used for **image analysis for user verification**  authentication purposes.          The extracted information can be stored in databases such as DynamoDB, Amazon Elasticsearch, or Amazon Kendra, to enable the case managers
with query capabilities. |
| Enhance Case worker and Manager’s productivity              | An ideal way to deal with this challenge is to introduce AI and ML into the entire application process and augment the human workforce with process automation and have human intervention only as needed. |
| Build review / approval work-flow automation                | [Amazon Augmented AI (A2I)](https://aws.amazon.com/augmented-ai/) makes it easy to build process automation and workflows required for human review of ML predictions. Many machine learning applications require humans to review low-confidence predictions to ensure the results are correct. Amazon A2I provides built-in human review workflows for common machine learning use cases, such as text extraction from documents. Using this service, **you can allow human reviewers to**
**step in when a model is unable to make a high-confidence prediction or to audit its predictions** on an ongoing basis. **AWS customers are implementing A2I with Textract to improve the efficiency of their document processing by combining the speed, efficiency and cost savings of ML with A2I in order to include human-in-the-loop validation to ensure accuracy** of results. |
| Build Machine Learning models to identify anomalies / fraud | Amazon SageMaker is a fully managed service that provides developers and data scientists with the ability to build, train, and deploy machine learning (ML) models quickly. SageMaker makes it easy to deploy your trained model into production with a single click so that you can start generating predictions on the claims and application data. This is not only useful in training models with accurate vs. inaccurate applications but also in flagging any suspicious or fraudulent application patterns or anomalous activities. |
| Build intelligent contact centers                           | **Deploy scalable Omni channel contact centers**: *Amazon Connect* is an easy to use Omni channel cloud contact center that helps companies provide superior customer service at a lower cost. Amazon Connect provides a seamless experience across voice and chat for your customers and agents. This includes one set of tools for skills-based routing, powerful real-time and historical analytics, and easy-to-use intuitive management tools.              **Provide AI-powered speech analytics**: *Amazon Contact Lens* (currently in preview) is a set of machine learning (ML) capabilities integrated into Amazon Connect. With Contact Lens for Amazon Connect, contact center supervisors can better understand the sentiment, trends, and compliance risks of customer conversations to effectively train agents, replicate successful interactions, and identify crucial feedback on benefits/claimant services. Additionally, Amazon Transcribe and Amazon Transcribe Medical provide speech-to-text capabilities. Recorded speeches can be converted to text and analyzed for further insights.                                                                             **Develop self service capabilities**:  *Amazon Lex* is a service for building conversational interfaces into any application using voice and text. Amazon Lex provides the advanced deep learning functionalities of automatic speech recognition for converting speech to text, and natural language understanding to recognize the intent of the text, to enable you to build applications with highly engaging user experiences and lifelike conversational interactions. **Provide language translation capabilities**: *Amazon Translate* can be used to convert text from one language to another (e.g. Spanish to English). Using Amazon Transcribe and Translate together, calls in one language can be first transcribed and then translated into a different language.                               **Build effective campaign management strategies**: *Amazon Pinpoint* helps the agencies engage with public by sending them personalized, timely and relevant communications via email, SMS and other channels. |
| Improve operational efficiencies                            | Program leadership can get **deep insights via operational dashboards** that can be built using Amazon QuickSight which is a cloud powered business intelligence service. Additionally,                                                                     Amazon Forecast can be used to **forecast enrollment** models and budgets.          Agencies can also proactively **identify fraud**, waste and abuse within the benefits programs using services such as Amazon Fraud Detector, |

### Reference Architecture and Best Practices

![img](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/Reference_Architecture.JPG?raw=true)

Following are the AWS services will be used to achieve this **Reference Architecture**.

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/aws_services.JPG?raw=true)

### Augmented AI Reference Workflow

Amazon A2I helps to **integrate** Amazon Textract, Amazon Rekognition, or a custom ML model into your workflow. When you create a **flow definition** you will be able to **specify conditions, such as confidence thresholds, that will trigger a human review**. 

The following diagram provides a high-level reference workflow to enhance case managers’ / case workers’ productivity; A2I is built into the workflow and human review is only needed when a document’s confidence level falls below a certain threshold.

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/a2i_workflow.JPG?raw=true)

## Machine Learning Lens - AWS Well-Architected Framework

### E2E ML Process

![im](https://github.com/amitkml/AWS-MachineLearning/blob/main/img/E2E_ML_Process.JPG?raw=true)

#### Business Goal Identification

For an ML-based approach to be successful, having an abundance of relevant, high-quality data that is
applicable to the algorithm that you are trying to train is essential. Carefully evaluate the availability
of the data to make sure that the correct data sources are available and accessible.

Apply these best practices:

- Understand business requirements
- Form a business question
- Determine a project’s ML feasibility and data requirements
- Evaluate the cost of data acquisition, training, inference, and wrong predictions
- Review proven or published work in similar domains, if available
- Determine key performance metrics, including acceptable errors
- Define the machine learning task based on the business question
- Identify critical, must have features

#### ML Problem Framing

Apply these best practices:

- Define criteria for a successful outcome of the project
- Establish an observable and quantifiable performance metric for the project, such as accuracy,
  prediction latency, or minimizing inventory value
- Formulate the ML question in terms of inputs, desired outputs, and the performance metric to be
  optimized
- Evaluate whether ML is a feasible and appropriate approach
- Create a data sourcing and data annotation objective, and a strategy to achieve it
- Start with a simple model that is easy to interpret, and which makes debugging more manageable

#### Data Collection

- AWS provides you with a number of ways to ingest data in bulk from static resources, or from new,
  dynamically generated sources, such as websites, mobile apps, and internet-connected devices. 
- For example, you can build a highly scalable data lake using Amazon Simple Storage Service (Amazon S3). To easily set up your data lake, you can use AWS Lake Formation.

- You can physically transfer petabytes of data in batches using AWS Snowball, or, if you have
  exabytes of data, by using AWS Snowmobile.
- use Amazon Kinesis Data Firehose to collect and ingest multiple streaming-data sources.
- Use a centralized approach to store your data, such as a data lake.
- Confirm the availability of the data, both quantity and quality

#### Data Preparation

AWS provides several services that you can use to annotate your data, extract, transfer, and load (ETL)
data at scale. 

- Start with a small, statistically valid set of sample data for data preparation
- Iteratively experiment with different data preparation strategies
- Implement a feedback loop during the data cleaning process that provides alerts for anomalies
  through the data preparation steps
- Enforce data integrity continuously
- Take advantage of managed ETL services

**Data preparation applies not only to the training data used for building a machine learning model,**
**but also to the new business data that is used to make inferences against the model after the model is deployed. Typically, the same sequence of data processing steps that you apply to the training data is also applied to the inference requests.**

- **Amazon SageMaker** is a *fully managed service* that encompasses the *entire ML workflow* to label
  and prepare your data, choose an algorithm, train it, tune and optimize it for deployment, and make
  prediction

- **Amazon SageMake**r Ground Truth offers easy *access to public and private human labelers* and provides *built-in workflows* and user interfaces for common labeling tasks. It uses a machine learning model to *automatically label raw data* to produce high-quality training datasets quickly at a fraction of the cost of manual labeling. Data is *only routed to humans if the active learning model cannot confidently label it*. The service provides *dynamic custom workflows*, job chaining, and job tracking to save time on subsequent ML labeling jobs by using the output from previous labeling jobs as the input to new labeling jobs.
- **AWS Glue** is a fully managed *extract, transform, and load (ETL)* service that can be used to *automate*
  the ETL pipeline. AWS Glue *automatically discovers and profiles your data* with the Glue Data Catalog,
  recommends and generates ETL code to transform your source data into target schemas, and runs the
  ETL jobs on a *fully managed, scale-out Apache Spark environment to load your data to its destination*. It
  also enables you to set up, orchestrate, and monitor complex data flows.

- **Amazon EMR** provides a *managed Hadoop framework* that makes it easy and fast to process *vast*
  *amounts of data across dynamically scalable Amazon EC2 instances*.

- **Amazon SageMaker** Inference Pipeline deploys pipelines so that you can pass raw input data and
  execute pre-processing, predictions, and post-processing on both real-time and batch inference requests. Inference pipelines enable you to *reuse existing data processing functionality*.

### Data Visualization and Analytics

A key aspect to understanding your data is to identify patterns.AWS provides several services that you can use to visualize and analyze data at scale.

Apply these best practices:

- Profile your data (categorical vs. ordinal vs. quantitative visualization)
- Choose the correct tool or combination of tools for your use case (such as data size, data complexity,
  and real-time vs. batch)
- Monitor your data analysis pipeline
- Validate assumptions about your data

- **Amazon SageMaker** provides a hosted Jupyter notebook environment that you can use to visualize and analyze data. Project Jupyter is an open-source web application that allows you to create visualizations and narrative text, as well as perform data cleaning, data transformation, numerical simulation, statistical modeling, and data visualization.
- **Amazon Athena** is a fully managed interactive query service that you can use to query data in Amazon
  S3 using ANSI SQL operators and functions. Amazon Athena is serverless and can scale seamlessly to
  meet your querying demands.
- **Amazon Kinesis Data Analytics** provides real-time analytic capabilities by analyzing streaming data to
  gain actionable insights. The service scales automatically to match the volume and throughput of your
  incoming data.
- **Amazon QuickSight** is a cloud-powered business intelligence (BI) service that provides dashboards and visualizations. The service automatically scales to support hundreds of users and offers secure sharing and collaboration for storyboarding. Additionally, the service has built-in ML capabilities that provide out-of-the-box anomaly detection, forecasting, and what-if analysis.

### Feature Engineering

Every unique attribute of the data is considered a feature. Feature engineering is a process to select and transform variables when creating a predictive model using machine learning or statistical modeling. **Feature engineering typically includes feature creation,feature transformation, feature extraction, and feature selection**.

- **Feature creation** identifies the features in the dataset that are relevant to the problem at hand.
- **Feature transformation** manages replacing missing features or features that are not valid. Some
  techniques include forming Cartesian products of features, non-linear transformations (such as
  binning numeric variables into categories), and creating domain-specific features.
- **Feature extraction** is the process of creating new features from existing features, typically with the
  goal of reducing the dimensionality of the features.
- **Feature selection** is the filtering of irrelevant or redundant features from your dataset. This is usually
  done by observing variance or correlation thresholds to determine which features to remove.

**Amazon SageMaker provides a Jupyter notebook environment with Spark and scikit-learn preprocessors that you can use to engineer features and transform the data. From Amazon SageMaker, you can also run feature extraction and transformation jobs using ETL services, such as AWS Glue or Amazon EMR. In addition, you can use Amazon SageMaker Inference Pipeline to reuse existing data processing functionality**.

Apply these best practices:

- Use domain experts to help evaluate the feasibility and importance of features
- Remove redundant and irrelevant features (to reduce the noise in the data and reduce correlations)
- Start with features that generalize across contexts
- Iterate as you build your model (new features, feature combinations, and new tuning objectives)

### Model Training

- A training algorithm computes several metrics, such as training error and prediction accuracy. 

- A classification algorithm can be measured by a confusion matrix that captures true or false positives and true or false negatives, while a regression algorithm can be measured by root mean square error (RMSE).

- The number and type of hyperparameters in ML algorithms are specific to each model. Some examples of commonly used hyperparameters are: Learning Rate, Number of Epochs, Hidden Layers, Hidden Units, and Activation Functions. Hyperparameter tuning, or optimization, is the process of choosing the optimal model architecture.
- Amazon SageMaker provides several popular built-in algorithms that can be trained with the training
  data that you prepared and stored in Amazon S3.
- Sagemaker allows custom training and the custom algorithm should be containerized using Amazon ECS and Amazon ECR.
- You can choose to train on a single instance or on a distributed cluster of instances. The infrastructure
  management that is needed for the training process is managed by Amazon SageMaker
- **Amazon SageMaker**
  - Automated training
  - hyperparameter tuning job
  - pre-built in algo
  - custom training
- **Amazon SageMaker Debugger**
  - provides visibility into the ML training process by monitoring, recording, and analyzing data that captures the state of a training job at periodic intervals.
  - it can automatically detect and alert you to commonly occurring errors, such as gradient values getting too large or too small.
- **Amazon SageMaker Autopilot**
  - It allows you to build classification and regression models simply by providing training data in tabular format.
  - This capability explores multiple ML solutions with different combinations of data preprocessors, algorithms, and algorithm parameter settings, to find the most accurate model.
  - Amazon SageMaker Autopilot selects the best algorithm from the list of high-performing algorithms that it natively supports. It also automatically tries different parameter settings on those algorithms to get the best model quality. 
  - You can then directly deploy the best model to production, or evaluate multiple candidates to trade off metrics like accuracy, latency, and model size.
- **AWS Deep Learning AMI and AWS Deep Learning Containers**
  - AWS Deep Learning AMI has popular deep learning frameworks and interfaces preinstalled, such as TensorFlow, PyTorch, Apache MXNet, Chainer, Gluon, Horovod, and Keras
- **Amazon EMR**
  - Has distributed cluster capabilities
  - Has an option for running training jobs on the data that is either stored locally on the cluster or in Amazon S3.

### Model Evaluation and Business Evaluation

- Evaluate your model using historical data (offline evaluation) or live data (online evaluation).
- Model validation can be performed using Amazon SageMaker, AWS Deep Learning AMI, or Amazon EMR.

