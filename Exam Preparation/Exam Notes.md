# Notes from AWS ML Whitepapers

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

