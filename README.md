- After developing a machine learning or neural network model, to deploy the algorithm into production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. Edge deployment is often preferred in the manufacturing domain because it ensures the factory continues operating even when the internet connection goes down. 
- For example, the edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.

  ![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)

__Degree of automation in decision-making using AI & Machine learning:__
- ___Human only:___ No automation is involved; all decisions are made entirely by humans.
- ___Shadow automation:___ Learning algorithms make predictions, but these are not applied in the actual process. For example, a system might predict factory machine failures but the predictions are not acted upon.
- ___AI Assistance:___ In this stage, AI helps humans by providing insights or suggestions. For example, when inspecting a smartphone, an AI might highlight areas with scratches to guide the human inspector to those spots but the final decision is still made by the human.
- ___Partial automation:___ In this case, the learning algorithm makes decisions when it is confident(e.g., determining if a product is defective or not). If the algorithm is unsure, the decision is referred to a human.
- ___Full automation:___ The learning algorithm handles every decision without human intervention.
- In many real-world deployments, we begin with Human-only decisions and gradually shift towards Full automation as confidence grows. The level of automation chosen depends on the performance of the AI and the use-case. For example, in healthcare, you might stop at partial automation where AI assists doctors rather than making all decisions.
- AI Assistance and Partial Automation are both examples of "human-in-the-loop" systems, where humans remain involved in the decision-making process

  ![image](https://github.com/user-attachments/assets/28a4375e-9fb5-4577-ae23-86f2608f53e5)

__Machine learning Production Project lifecycle:__
- ___Scoping:___
  - In this phase, you define the project, identify features (X) and target (Y), and estimate key metrics like accuracy, latency (prediction time), throughput (queries per second), and resource needs (time, compute, budget).
- ___Data:___
  - In this phase, you collect and organize data, define data sources, establish baselines, and label the data. 
- ___Modeling:___
  - In this phase, you select the algorithm, train the model, and perform error analysis. You then adjust the model and perform hyperparameter tuning. Since machine learning is an iterative process, you may need to update the model or decide whether to collect more data or not, followed by further error analysis.
- ___Deployment & Monitoring:___
  - In this phase, the model is deployed into production (e.g., cloud, edge, IoT, web browser) to serve prediction requests. This phase also includes monitoring the system and tracking incoming data.
- ___Maintenance:___
  - After the initial deployment, you may need to retrain the model using newly collected data to continuously improve and update it until a more accurate version is deployed.

  ![image](https://github.com/user-attachments/assets/be8511b5-0b15-43a5-8f6b-e28d49a620f9)

__Deployment patterns:__
- ___New product:___ When launching a new product that hasn’t been offered before, a common pattern is to direct a small portion of traffic to the model, and then gradually increase it as confidence grows. This reduces the risk of impacting a large number of users with an untested system.
- For example, a company releases a new recommendation system. Initially, only 5% of users got recommendations from the new system. As the system proves reliable, traffic is gradually increased until all users are served by it.
- ___Automate or Assist manual tasks:___ This pattern is useful when replacing or supporting a manual process with an algorithm. For example, if factory workers inspect smartphones for scratches, and you want to use machine learning to assist or replace human inspectors.
  - ___Shadow deployment:___ The machine learning algorithm "shadows" the human worker by running in parallel but doesn’t make any actual decisions. The output from the algorithm is compared with the human’s decisions to evaluate its performance.
  - For example, a factory inspector and an algorithm both check for scratches on phones, but only the human inspector's decision matters at first. Over time, the algorithm's performance is analyzed before it’s allowed to make real decisions.
  - ___Canary deployment:___ Once confident, you can let the algorithm start making decisions with a small percentage of traffic (e.g., 5%). This way, mistakes will only impact a small part of the system. As the model performs well, the traffic can be gradually increased.
  - For example, the algorithm starts inspecting only 5% of phones. If it performs well, this is increased to 50%, and eventually, all phones are inspected by the algorithm.
- ___Replacing a previous machine learning system:___ When replacing an older machine learning system with a newer one, it’s common to gradually direct traffic to the new system while monitoring its performance.
  - ___Blue-Green deployment:___ This involves running two versions of a system: an old version (Blue) and a new version (Green). At deployment, traffic is switched entirely from Blue to Green. If there’s an issue with the new system, you can quickly revert traffic back to the Blue version, ensuring minimal disruption.

__Monitoring deployed machine learning systems:__
- The most common way to monitor a machine learning (ML) system is by using a dashboard that tracks its performance over time. Depending on your application, the dashboard may monitor different metrics, such as server load, non-null outputs (where the ML system fails to return an output), and missing input values (which can indicate that something in the input data has changed).
- The metrics used are decided by discussing potential issues that could arise in the system. A good practice is to start with a lot of metrics and gradually remove those that are not useful.
  - ___Software metrics:___ includes memory usage, compute latency, throughput, and server load metrics
  - ___Input and output metrics:___ includes metrics to track performance of the ML model
- Machine learning modelling and deployment are both highly iterative processes. Once your initial deployment is running, you can set up monitoring dashboards and thresholds to track the model's performance on real traffic and trigger alarms. Based on the metrics, you may need to update your deployment.
- When a model needs updating, there are two main approaches:
  - ___Manual retraining:___ In manual retraining, an engineer may retrain the model, perform error analysis on the new model, and deploy the updated version.
  - ___Automatic retraining (MLops):___ In automatic retraining, the retraining process occurs automatically. Based on predefined rules or performance metrics, the new updated version of the model is deployed without manual intervention.
  
    ![image](https://github.com/user-attachments/assets/36140dcf-7597-4223-975d-e918e967ae23)

__Challenges in deploying machine learning models:__
- ___Machine learning or Statistical issues:___
  - Concept drift
  - Data drift
- ___Software engine issues:___
   - Whether you need real-time predictions or batch predictions?
   - Whether your prediction service runs in the cloud, or on edge devices?
   - How many compute resources do you have (CPU/GPU)?
   - Considerations for latency, throughput (queries per second)
   - Logging (to track the behaviour and performance of a deployed machine learning model)
   - Security and privacy, when dealing with sensitive data.

# MlOps (Machine learning Operations) Introduction:
- MLOps (Machine Learning Operations) is a practice that combines machine learning with DevOps principles to automate and streamline the process of developing, deploying, and managing machine learning models in production.
- MLOps ensures that a model can automatically retrain itself when performance degrades or when new data is available, without requiring manual intervention.  This allows the model to stay up-to-date and accurate as conditions change, reducing the need for constant human oversight.

__Degree of MLOps Automation in ML project lifecycle:__
- ___Low degree of MLOps:___ Manual intervention is needed for retraining, monitoring, and deployment.
- ___Medium degree of MLOps:___ Some processes are automated, like monitoring and triggering retraining, but humans still need to make final decisions.
- ___High degree of MLOps:___ Almost everything is automated, including data collection, model retraining, deployment, and monitoring, with minimal human involvement.

## Machine learning pipeline:
- An ML (Machine Learning) pipeline is a step-by-step process or a software architecture that automates the flow of data through a machine learning model. It typically includes stages such as data collection, data preprocessing, feature engineering, model training, model evaluation, and deployment. The pipeline ensures that each step is executed in the correct order and can be reliably repeated, making the entire machine-learning process more efficient and scalable.

  ![image](https://github.com/user-attachments/assets/5fa9de7a-a4e1-4268-9669-68f25eb10010)

- An ML pipeline can vary from project to project depending on the specific requirements, such as the type of data, model, and deployment needs. However, most ML pipelines are structured as Directed Acyclic Graphs (DAGs), meaning the steps in the pipeline are arranged in a directed order, with no cycles, ensuring that each step is executed in sequence without any loops.
- This structure helps maintain a clear flow of data and processing tasks, making it easier to automate, monitor, and debug the pipeline.

__DAG (Directed Acyclic Graph):__
- DAG stands for Directed Acyclic Graph, where ___'Directed'___ indicates a one-way relationship between nodes, ___'Acyclic'___ means you cannot start at one node and follow the directed edges to return to the same node, and ___'Graph'___ refers to the collection of nodes (tasks) that are connected by edges, representing the flow of data or execution in a way that reflects their relationships and dependencies.
- Orchestrator tools are used to execute nodes or tasks in the correct order based on their dependencies in a pipeline. These tools automate the scheduling, execution, and management of tasks to ensure that each task is performed in sequence, according to the dependencies defined within the pipeline.

### TFX (TensorFlow Extended)
- TFX is a widely used open-source framework for creating end-to-end machine learning pipelines.
  
  ![image](https://github.com/user-attachments/assets/35b9ff9a-1a8a-49a2-9eb4-585df266edc5)

__TFX production components or TFX "Hello World":__
- ___ExampleGen:___ Ingests raw data from various sources, such as databases, cloud storage, or local files, into the pipeline.
- ___StatisticsGen:___ Analyzes the ingested data and generates statistical summaries, such as types of features, ranges of numerical features, and distributions.
- ___ExampleValidator:___ Detects issues with the data, such as missing values or anomalies, and helps validate the quality of the input data before further processing.
- ___SchemaGen:___ Defines the expected structure of the data, including data types, feature constraints, and relationships between features.
- ___Transform:___ Performs feature engineering by applying transformations to the raw data, creating meaningful features that are more suitable for model training.
- ___Tuner:___ Optimizes the model's hyperparameters to enhance performance by performing automated hyperparameter tuning.
- ___Trainer:___ Builds and trains the machine learning model using the transformed data, typically utilizing TensorFlow Estimators or Keras models.
- ___Evaluator:___ Assesses the model's performance using various metrics, such as accuracy, precision, recall, F1 score, and others. It also performs model validation and compares the trained model against baseline models.
- ___InfraValidator:___ Verifies that the deployment infrastructure has sufficient resources (e.g., memory, processing power) to run predictions using the trained model. It helps to prevent deployment failures.
- ___Pusher:___ Deploys the model to production environments, making it available for serving predictions.
  - ___TensorFlow Hub:___ for transfer learning and generating embeddings.
  - ___TensorFlow.js:___ to use the model in web browsers or Node.js applications.
  - ___TensorFlow Lite:___ for deployment on mobile devices or IoT devices.
  - ___TensorFlow Serving:___ for serving the model on servers or serving clusters.
- The TFX production components are built on top of various open-source libraries, including TensorFlow Data Validation, TensorFlow Transform, TensorFlow Model Analysis, TensorFlow Data Validation Outcomes, and TensorFlow Serving.

  ![image](https://github.com/user-attachments/assets/bffe0011-cae9-4e73-b84b-23d88c50a467)

  ![image](https://github.com/user-attachments/assets/759ad55c-4dbe-45b2-82e4-cc581b271be6)

> ___Installing libraries:___
```python
# Install TFX
pip install tfx

# For TensorFlow Estimators and Keras models
pip install tensorflow

# TensorFlow Data Validation (TFDV)
pip install tensorflow-data-validation

# TensorFlow Transform (TFT)
pip install tensorflow-transform

# TensorFlow Model Analysis (TFMA)
pip install tensorflow-model-analysis

# TensorFlow Serving (for model serving, often handled separately, but install the TensorFlow Serving Python client)
pip install tensorflow-serving-api
```
__Ensuring Data Quality in Machine Learning Pipeline:__
- 


## Scrap:
- In an ML pipeline, we use a data pipeline comprising a sequence of automated tasks. These processes include data collection, data ingestion, data validation, and data preparation. Additionally, it is essential to monitor the data collection process.
- You need to ensure that your training data adequately covers the same feature space as the prediction requests you will receive once your model is in production. It's important to maintain good data quality not just at the beginning but throughout the application's lifecycle. This involves sourcing data responsibly and addressing concerns related to bias and fairness.

Questions to ask before working on a new project:
- What type of data do you need, and how much of it is required?
- How often will you need new data, and when do you expect changes in the data?
- Is the data labelled or annotated? If not, what methods can be used to label it?
- What are the predictive features?
- What are the labels?
- What are the metrics used to evaluate the model?

Dataset issues:
- Inconsistent formatting
- Bias and fairness concerns
- Monitoring data sources for system issues and outages

Data privacy involves the proper usage, collection, retention, deletion, and storage of data.
How you handle data privacy and security depends on the nature of the data, operating conditions, regulations, and policies.

Sudden Changes:
- Faulty sensors or cameras
- Corrupted log data
- Moved or disabled sensors/cameras
- System Problems:

Faulty software updates
- Loss of network connectivity
- System outages
- Invalid or incorrect credentials

Detecting Problems with Deployed Models:
- Data and concept drift: Changes in the input data distribution or the relationship between features and target variables.
- Changing ground truth: Evolving definitions or changes in the true labels over time.
- Model performance decay: Model accuracy decreases over time due to data drift or concept drift.
- Model retraining: Retraining the model can help improve performance and adapt to new data patterns.
- Feedback and Labeling Processes:

Collect feedback through direct labelling, human labelling, semi-supervised learning, or active learning to enhance the model's performance.






## Data collection:
![image](https://github.com/user-attachments/assets/26fa7ddd-f666-4e40-ae5e-1d7723d9c5f2)





## Validatin and detecting data issues:
- Drift refers to the changes in data over time
- Skew refres to the differences between two static versions or different sources such as trining set and serving set (Data that you are getting for prediction request)
- 


seasanality and trend or unexpected events

Schema skew
training and serving data do not confront to the same schema
Distribution skew:
manifested with VAriate and covariatte shift

skew detection involves continuous evaluation of data coming to your server once you train your model so to detect these kinda changes you need Continuous monitorin and evaluation of the data 

dataset shift: when the the joint probabilty of input features x and labels y is not same duing trainning and serving
covariate shift referes to the change in the distribution of input variables present in training and serving data 
Concept shift refers to the relationship dbetween input and output variabes as opposed to the difference in 

## Workflow to detetct data skew:
- FIsr stage is looking a the training data nd computing baseline satatistics and a referance schema then you do bascially the same with your serving data you are going to generate the descriptive statistics and then yoou compare the two. SO you compare the baseline stat and instances check differences between serving and traing data and you look for skew and drift. Significant change can be anamolies and can then trigger an alert and then this trigger will goes into monitoring system that can either be a human and another system to analyze the change and decide a proper course of action 

![image](https://github.com/user-attachments/assets/9c5c5070-962c-46f0-8c5d-45c9d543c4ea)

TFDV helps developers understand validate and monitor their ml data at scale. TFDV generates data statistics and browser visualization also helps infers the schema for your data performs validity checks against schema also detects training serving skew by validating against the reference schema that you generated from training data 
- Using TFDV you can detect three different types of skew Schema skew feature skew distribution skew
skew for categorical eatures is expressed in chebyshev's distance.  you can set threshold values SO that you can receive warnings when drift is higher than what you think is acceptable  
- Schema skew occurs when the serving and training data dont confront to the same schema. Feature skew happens when feature values are different that the serving feature values. Distribution skew is of distribution of serving and training dataset significantly different
- 

 
![image](https://github.com/user-attachments/assets/8b5217c6-764e-4a78-9601-298410415c51)

__Data Ingestion:__
- Data ingestion is the process of bringing data from various sources, such as databases, files, or websites, into a system where it can be used. Think of it like gathering ingredients in your kitchen before cooking a meal.
- 

__Data validation:__
- 


__Data pipeline:__ 
- A data pipeline refers to the series of steps involved in processing data during both the development and production phases to produce the final output. Ensuring replicability in a data pipeline is key to maintaining consistency in machine learning models across different environments, such as development, testing, and production. 
- In production machine learning, ensuring data quality is crucial because poor data leads to poor model performance. Therefore, detecting data issues and ensuring data validation is essential. During the initial development phase, manual methods for managing the data pipeline may work, but as you move to production, automated tools are required.
- Tools like TensorFlow Transform, Apache Beam, or Airflow can be used to automate and manage the data pipeline, ensuring that the same data processing methods are applied as new data flows in. These tools help maintain model accuracy, reduce errors, and ensure continuous data validation for better performance.
- Additionally, tracking metadata, data provenance, and data lineage is essential for managing complex data pipelines. This involves maintaining logs of data sources, transformations, and processing steps, which helps to understand the flow of data from raw input to final output. Storing metadata such as creation dates, schema information, and versioning allows for better tracking of modifications and aids in debugging.
- Apache Beam can be used to automate the process of cleaning the data, TensorFlow Transform can be used to automate the feature engineering process, and Airflow can be used to schedule regular evaluations of the model as new data comes in.

__Data validation:__
- Data validation is the process of checking your data to ensure it's correct, consistent, and useful before using it in machine learning models
- Tools like TensorFlow Data Validation (TFDV) and Pandas Profiling can be used for data validation. These tools help analyze data, detect data issues and generate reports on data quality.
- TFDV, for example, provides insights into feature distributions, schema skew, and drift detection.
- TFDV itself doesn't trigger alarms for any data issues but you can integrate other tools or custom scripts with TFDV's output to trigger notifications based on predefined conditions or thresholds. 
- You can also use monitoring and alerting systems like Prometheus, Grafana, or cloud-based services like AWS CloudWatch to watch for specific signals from TFDV and send notifications accordingly.

__Types of Data issues:__
- ___Data drift:___ Data drift occurs when the data changes over time. For example, A store sees a sudden increase in online orders during the holiday season, which is different from the data used to train the model earlier in the year.
- ___Concept drift:___ Concept drift occurs when the relationship between the input features and the target feature changes. For example, A model was trained to flag spam emails based on certain keywords, but now spammers use new words, so the model doesn’t work as well.
- ___Schema skew:___ Schema skew occurs when the types of data change between training and serving. For example, The model was trained with age as a whole number, but now it’s receiving age as a decimal, causing errors.
- ___Distribution skew:___ Distribution skew occurs when the spread or range of feature values is different between training and serving. For example, A weather prediction model trained with temperatures ranging from 0-100°F now sees temperatures from -50°F to 150°F, which it wasn’t prepared for.

![image](https://github.com/user-attachments/assets/2d26a1ec-836d-426a-ab0a-2a0d280bb33e)


__Data Transform:__
- Inconsistent data (e.g., data that isn’t scaled) and different feature engineering approaches can negatively impact model performance. This is why scalable data processing tools are important for handling large datasets efficiently.
- The TensorFlow Transform (TFT) framework helps by allowing you to define preprocessing functions, like scaling or encoding, that can be applied to large datasets. To manage large-scale data efficiently, TFT uses Apache Beam to run these preprocessing functions.
- Apache Beam ensures scalability by providing the infrastructure to process data in parallel and distribute the workload across multiple machines. It also offers a consistent API for both batch and stream processing, making it easy to switch between them without altering your code.
- In TFT data transformation, you start with raw data and create a preprocessing function to define how the data should be transformed (e.g., scaling, encoding). After that, you set up an Apache Beam pipeline to apply this preprocessing function to the raw data.

> ___Set up preprocessing function:___
```python
import tensorflow_transform as tft

def preprocessing_fn(inputs):
    outputs = {}
    outputs['size_normalized'] = tft.scale_to_0_1(inputs['size'])
    outputs['bedrooms_normalized'] = tft.scale_to_0_1(inputs['bedrooms'])
    outputs['location_one_hot'] = tft.compute_and_apply_vocabulary(inputs['location'])
    return outputs
```

> ___Set up Apache Beam pipeline:___
```python
import apache_beam as beam

def run_tft_pipeline(raw_data):
    with beam.Pipeline() as pipeline:
        transformed_data, transform_metadata = (
            pipeline
            | 'ReadData' >> beam.Create(raw_data)  # Create a PCollection from raw data
            | 'TransformData' >> tft.beam.AnalyzeAndTransformDataset(preprocessing_fn)  # Apply TFT
        )
        transformed_data | 'WriteData' >> beam.io.WriteToText('output.txt')  # Write transformed data
```

## Important terminology:
- ___Data drift:___ Data drift occurs when the distribution of data changes over time, leading to a decline in model performance. This happens when the data used for predictions differs from the data the model was originally trained on.
- For example, a model predicting house prices may be trained on data from 2010 to 2020. If economic factors or housing trends shift, such as a recession or new developments, the data distribution changes, causing the model's predictions to become less accurate.
- ___Concept drift:___ Concept drift occurs when the relationship between input data and the target output changes over time, leading to a decline in model performance.
- For example, a model predicting customer churn may be trained on patterns like older customers with long subscriptions being more likely to churn. If the business introduces new features that attract younger customers with different behaviours, the model's performance may decline as the underlying customer patterns shift.
- ___Edge device:___ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, and it is often used in IoT, manufacturing, and automation.
- ___Data-centric approach:___ Data-centric approach focuses on improving the quality and quantity of the data used to train models.
- ___Model-centric approach:___ Model-centric approach focuses on improving the algorithms or model parameters regardless of data.
- ___Real-time predictions:___ It involves making predictions instantly as new data comes in.
- ___Batch predictions:___ It involves making predictions on a group of data at once, rather than one at a time.
- ___Metadata:___ Metadata (data about your data) helps you understand where your data comes from(data provenance) and how it has been processed(data lineage: history of all the steps data went through before reaching its final form), which is useful for fixing errors and improving your models.
- ___Orchestration:___ Orchestration refers to the execution and coordination of tasks within an ML pipeline, ensuring that each task is executed in the correct order based on its dependencies.
- ___Orchestrator:___ Orchestrator is a tool or system that automates and manages this process of orchestration, ensuring that the tasks are scheduled, executed, and monitored efficiently.
- ___DAG (Directed Acyclic Graph):___ DAG is a powerful tool for defining ML pipelines
- ___Model (performance) decay:___ Model performance decay refers to the gradual decline in the accuracy or effectiveness of a machine learning model over time.
- ___Data ingestion:___
- ___Data validation:___
 
## Libraries:
- Tensorflow
- Keras
- Pytorch
- TFDV
- TensorFlow Transform
- Apache Beam / Airflow
- TFX (widely used open source framework for creating an end to end ml pipeline)
- Weights and Biases
- Comet
- Prefect
- DVC
- MLflow
- Zenml
- Bentoml
- Flyte
- Kubeflow pipeline
- AWS Sagemaker pipeline
- Azure ML Pipeline
- SageMaker Studio

## Tools: 
- ___Experiment tracking tools:___ Experiment tracking tools, such as text files, shared spreadsheets, or specialized platforms like Weights and Biases, Comet, MLflow, and SageMaker Studio, help in organizing and tracking machine learning experiments like algorithm and code version, dataset used, hyperparameters, performance metrics(accuracy, precision, recall, f1 score).
- ___Data pipeline tools:___ Data pipeline tools like TensorFlow Transform, Apache Beam, or Airflow can be used to automate and manage the data pipeline. These tools ensure that the same data processing methods are applied as new data flows in, helping to maintain accuracy and reduce errors.
- ___Orchestrator tools:___ Celery, Argo, Airflow, Kubeflow, Luigi

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)

![image](https://github.com/user-attachments/assets/c3e03585-80ef-40ca-9507-84db1f2ca1ca)

![image](https://github.com/user-attachments/assets/1974eaa3-e959-4272-b126-ebaf8dce3699)

![image](https://github.com/user-attachments/assets/ba9edf15-99d6-41b1-9923-82ad629c24c6)

![image](https://github.com/user-attachments/assets/5089272c-56d7-4a5e-acd5-a1691ba0f5bf)

![image](https://github.com/user-attachments/assets/3b289c1a-d812-44a9-ae4b-a4cf49a8f706)

![image](https://github.com/user-attachments/assets/96287642-fa65-464c-a97f-de9a1c529869)

![image](https://github.com/user-attachments/assets/0c663e48-c9a4-44dd-a9b7-7845c9db85bd)

![image](https://github.com/user-attachments/assets/d6a4a7c9-f421-4d1a-853f-6a6f125d10be)

![image](https://github.com/user-attachments/assets/8ff01fd5-b000-47ab-9f5d-7a0b4f17777e)

![image](https://github.com/user-attachments/assets/79ae876f-bb95-477c-8089-7b650d490d35)

![image](https://github.com/user-attachments/assets/215ef187-fa14-4676-81f1-171823407036)

![image](https://github.com/user-attachments/assets/4e02f9bf-36fb-4c89-be63-405cee8ee1fc)

![image](https://github.com/user-attachments/assets/193cdd7f-6325-489d-a197-26529e9c7540)


## Scrap:
__Orchestration Tools:__
Role: Execute the pipeline by managing the workflow defined by the DAG. Function: Handle scheduling, task execution, retries, parallelism, and resource management. Output: Actual execution of the ML pipeline as per the defined DAG.

__Monitoring:__
- Downtime
- Errors
- Distribution shifts
- Data failure
- Service failure

## What is mean by artifact?
Data artefacts are created as pipeline components execute. Each time a component produces a result it generates an artifact. Artifacts are created as the components of the ml pipeline. Artifact includes basically everything that is produced by the pipeline.  including data in different stages of transformation often as a result of feature engineering and the model itself and things like schema and the metrics and so forth. Basically every result that is produced is an artifact
- Meta data helps to identify data drift

### Data versioning: DVC, Git-LFS
### Environment versioning: Docker, Terraform
### Code versioning: Github
### 
- Tracking different data versions. Managing a data pipeline is a big challenge as data evolves through the natural life cycle of a project over many different training runs amachine learning when its done properly 
