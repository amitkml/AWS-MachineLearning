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



