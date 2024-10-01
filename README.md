# MlOps (Machine learning Operations)
## Basics:
- After developing a machine learning or neural network model, to deploy the algorithm into production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. Edge deployment is often preferred in the manufacturing domain because it ensures the factory continues operating even when the internet connection goes down.
- For example, the edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.

  ![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)

__Machine learning Project lifecycle__
- ___1) Scoping:___ In this phase, you define the project, identify features (X) and target (Y), and estimate key metrics like accuracy, latency (prediction time), throughput (queries per second), and resource needs (time, compute, budget).
- ___2) Data:___ In this phase, you collect and organize data, define data sources, establish baselines, and label the data.
- ___3) Modeling:___ In this phase, you select the algorithm, train the model, and perform error analysis. You then adjust the model and perform hyperparameter tuning. Since machine learning is an iterative process, you may need to update the model or decide whether to collect more data or not, followed by further error analysis.
- ___4) Deployment:___ In this phase, you deploy the model into production which also includes monitoring the system, and tracking the incoming data.
- ___5) Maintenance:___ After the initial deployment, you may need to retrain the model using newly collected data to continuously improve and update it until a more accurate version is deployed.

  ![image](https://github.com/user-attachments/assets/be8511b5-0b15-43a5-8f6b-e28d49a620f9)

__Data pipeline:__ 
- A data pipeline refers to the concept of replicating a series of steps involved in processing data during the development phase and production phase to produce the final output.
- Replicability is key to ensure that the machine learning model behaves consistently across different environments (development, testing, and production).
- During the initial development phase, it's okay to use manual methods and focus on making the system work but when moving to production, tools like TensorFlow Transform, Apache Beam, or Airflow can be used to automate and manage the data pipeline.
- These tools ensure that the same data processing methods are applied as new data flows in, helping to maintain accuracy and reduce errors.
- Tracking metadata, data provenance, and data lineage is crucial for managing complex data pipelines; maintaining a log of data sources, transformations, and processing steps helps understand the flow of data from raw input to final output. Additionally, storing metadata about the data such as creation date, schema, and versioning, allows you to track modifications and aids in debugging any issues.
- Data pipeline tools like Apache Beam can be used to automate the process of cleaning the data, TensorFlow Transform can be used to automate the feature engineering process, and Airflow can be used to schedule regular evaluations of the model as new data comes in.

__Challenges in deploying machine learning models:__
- ___1) Machine learning or Statistical issues:___
  - Concept drift
  - Data drift
- ___2) Software engine issues:___
   - Whether you need real-time predictions or batch predictions?
   - Whether your prediction service runs in the cloud, or on edge devices?
   - How many compute resources do you have(CPU/GPU)?
   - Considerations for latency, throughput(queries per second)
   - Logging(to track the behaviour and performance of a deployed machine learning model)
   - Security and privacy, when dealing with sensitive data.

__Deployment patterns:__
- ___1) New product:___ When launching a new product that hasn’t been offered before, a common pattern is to direct a small portion of traffic to the model, and then gradually increase it as confidence grows. This reduces the risk of impacting a large number of users with an untested system.
- For example, a company releases a new recommendation system. Initially, only 5% of users got recommendations from the new system. As the system proves reliable, traffic is gradually increased until all users are served by it.
- ___2) Automate or Assist manual tasks:___ This pattern is useful when replacing or supporting a manual process with an algorithm. For example, if factory workers inspect smartphones for scratches, and you want to use machine learning to assist or replace human inspectors.
  - ___Shadow deployment:___ The machine learning algorithm "shadows" the human worker by running in parallel but doesn’t make any actual decisions. The output from the algorithm is compared with the human’s decisions to evaluate its performance.
  - For example, a factory inspector and an algorithm both check for scratches on phones, but only the human inspector's decision matters at first. Over time, the algorithm's performance is analyzed before it’s allowed to make real decisions.
  - ___Canary deployment:___ Once confident, you can let the algorithm start making decisions with a small percentage of traffic(e.g., 5%). This way, mistakes will only impact a small part of the system. As the model performs well, the traffic can be gradually increased.
  - For example, the algorithm starts inspecting only 5% of phones. If it performs well, this is increased to 50%, and eventually, all phones are inspected by the algorithm.
- ___3) Replacing a previous machine learning system:___ When replacing an older machine learning system with a newer one, it’s common to gradually direct traffic to the new system while monitoring its performance.
  - ___Blue-Green deployment:___ This involves running two versions of a system: an old version (Blue) and a new version (Green). At deployment, traffic is switched entirely from Blue to Green. If there’s an issue with the new system, you can quickly revert traffic back to the Blue version, ensuring minimal disruption.

__Degree of automation:__
- ___1) Human only:___ No automation is involved; all decisions are made entirely by humans.
- ___2) Shadow automation:___ Learning algorithms make predictions, but these are not applied in the actual process. For example, a system might predict factory machine failures but the predictions are not acted upon.
- ___3) AI Assistance:___ In this stage, AI helps humans by providing insights or suggestions. For example, when inspecting a smartphone, an AI might highlight areas with scratches to guide the human inspector to those spots but the final decision is still made by the human.
- ___4) Partial automation:___ In this case, the learning algorithm makes decisions when it is confident(e.g., determining if a product is defective or not). If the algorithm is unsure, the decision is referred to a human.
- ___5) Full automation:___ The learning algorithm handles every decision without human intervention.
- In many real-world deployments, we begin with Human-only decisions and gradually shift towards Full automation as confidence grows. The level of automation chosen depends on the performance of the AI and the use-case. For example, in healthcare, you might stop at partial automation where AI assists doctors rather than making all decisions.
- AI Assistance and Partial Automation are both examples of "human-in-the-loop" systems, where humans remain involved in the decision-making process

  ![image](https://github.com/user-attachments/assets/28a4375e-9fb5-4577-ae23-86f2608f53e5)

__Monitoring deployed machine learning systems:__
- The most common way to monitor a machine learning (ML) system is by using a dashboard that tracks its performance over time. Depending on your application, the dashboard may monitor different metrics, such as server load, non-null outputs (where the ML system fails to return an output), and missing input values (which can indicate that something in the input data has changed).
- The metrics used are decided by discussing potential issues that could arise in the system. A good practice is to start with a lot of metrics and gradually remove those that are not useful.
  - ___Software metrics:___ includes memory usage, compute latency, throughput, and server load metrics
  - ___Input and output metrics:___ includes metrics to track performance of the ML model
- Machine learning modelling and deployment are both highly iterative processes. Once your initial deployment is running, you can set up monitoring dashboards and thresholds to track the model's performance on real traffic and trigger alarms. Based on the metrics, you may need to update your deployment.
- When a model needs updating, there are two main approaches:
  - ___Manual retraining:___ In manual retraining, an engineer may retrain the model, perform error analysis on the new model, and deploy the updated version.
  - ___Automatic retraining:___ In automatic retraining, the retraining process occurs automatically. Based on predefined rules or performance metrics, the new updated version of the model is deployed without manual intervention.
  
    ![image](https://github.com/user-attachments/assets/36140dcf-7597-4223-975d-e918e967ae23)

## Machine learning pipeline:
- An ML (Machine Learning) pipeline is a step-by-step process that automates the flow of data through a machine learning model. It typically include stages like data collection, data preprocessing, feature engineering, model training, model evaluation, and deployment. The pipeline ensures that each step is executed in order and can be repeated reliably, making the entire machine-learning process more efficient and scalable.

__DAG (Directed Acyclic Graph):__
- DAG is a powerful tool for structuring the ML pipeline.
- Directed meaning the connections between nodes have a direction, indicating a one-way relationship from one node to another.
- Acyclic meaning you cannot start at one node and follow the directed edges to return to the same node.
- Graph meaning a collection of nodes connected by edges.

__DAG key components as Nodes:__
- ___Data ingestion:___ "ExampleGen" ingests raw data from various sources, such as databases, cloud storage, or local files into the pipeline.
- ___Data analysis:___ "StatisticsGen" analyzes the data and generates statistical summaries of the data such as types of features, ranges of numerical features, etc. "SchemaGen" defines the expected structure of the data, including data types, feature constraints, and relationships between features.
- ___3) Data transformation:___ "Transform" performs feature engineering(applies transformations to the raw data to create meaningful features for model training) and data preprocessing(categorical encoding, feature scaling, etc.).
- ___4) Model training:___ "Trainer" builds and trains the machine learning model using the processed data. "Tuner" optimizes the model’s hyperparameters to enhance performance.
- ___5) Model evaluation:___ "Evaluator" assesses the model’s various performance metrics, such as accuracy, precision, recall, F1 score, and others. It also performs model validation and comparison against baseline models
- ___6) Infrastructure validation:___ "InfraValidator" verifies that the infrastructure has sufficient resources (e.g., memory, processing power) to run predictions using the trained model to prevent deployment failures.
-___7) Deployment:___ "Pusher" deploys the model to production environments.

__TFX (Tensorflow Extended):__
- The TFX framework utilizes DAG to define the ML pipeline components and their dependencies.
- TFX deployment options:
  - ___TensorFlow Hub:___ for transfer learning and generating embeddings.
  - ___TensorFlow.js:___ to use the model in web browsers or Node.js applications.
  - ___TensorFlow Lite:___ for deployment on mobile devices or IoT devices.
  - ___TensorFlow Serving:___ for serving the model on servers or serving clusters.

  ![image](https://github.com/user-attachments/assets/4fe2b27a-5147-4d3a-ae32-d5c8c2366dee)

![image](https://github.com/user-attachments/assets/759ad55c-4dbe-45b2-82e4-cc581b271be6)

## Important terminology:
- ___Data drift:___ Data drift occurs when the distribution of data changes over time, leading to a decline in model performance. This happens when the data used for predictions differs from the data the model was originally trained on.
- For example, a model predicting house prices may be trained on data from 2010 to 2020. If economic factors or housing trends shift, such as a recession or new developments, the data distribution changes, causing the model's predictions to become less accurate.
- ___Concept drift:___ Concept drift occurs when the relationship between input data and the target output changes over time, leading to a decline in model performance.
- For example, a model predicting customer churn may be trained on patterns like older customers with long subscriptions being more likely to churn. If the business introduces new features that attract younger customers with different behaviours, the model's performance may decline as the underlying customer patterns shift.
- ___Edge device:___ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, and it is often used in IoT, manufacturing, and automation.
- ___Data-centric approach:___ Data-centric approach focuses on improving the quality and quantity of the data used to train models.
- ___Model-centric approach:___ Model-centric approach focuses on improving the algorithms or model parameters regardless of data.
- ___MlOps:___ MlOps (Machine Learning Operations) is a way to manage and deploy machine learning models quickly and efficiently into production. 
- ___Real-time predictions:___ It involves making predictions instantly as new data comes in.
- ___Batch predictions:___ It involves making predictions on a group of data at once, rather than one at a time.
- ___Metadata:___ Metadata(data about your data) helps you understand where your data comes from(data provenance) and how it has been processed(data lineage: history of all the steps data went through before reaching its final form), which is useful for fixing errors and improving your models.
- ___Orchestration:___ in simple terms means organizing and coordinating different tasks or components so they work together smoothly.
- ___Orchestrator:___ is a tool that manages and schedules these tasks, ensuring they run in the correct order based on their dependencies. In machine learning
- ___DAG(directed acyclic graph):___ is a powerful tool for defining ML pipelines

## Libraries:
- TFX (end-to-end machine learning platform for deploying production ml pipeline)

  ![image](https://github.com/user-attachments/assets/6222935c-ae42-4016-8a58-796c344e7c02)

- Tensorflow
- Keras
- Pytorch
- TensorFlow Transform
- Apache Beam / Airflow
- Weights and Biases
- Comet
- MLflow
- SageMaker Studio

## Tools: 
- ___Experiment tracking tools:___ Experiment tracking tools, such as text files, shared spreadsheets, or specialized platforms like Weights and Biases, Comet, MLflow, and SageMaker Studio, help in organizing and tracking machine learning experiments like algorithm and code version, dataset used, hyperparameters, performance metrics(accuracy, precision, recall, f1 score).
- ___Data pipeline tools:___ Data pipeline tools like TensorFlow Transform, Apache Beam, or Airflow can be used to automate and manage the data pipeline. These tools ensure that the same data processing methods are applied as new data flows in, helping to maintain accuracy and reduce errors.
- ___Orchestrator tools:___ Orchestration tools(orchestrator) like Celery, Argo, Airflow, Kubeflow, and Luigi can be used to schedule tasks, ensuring they run in the correct order based on their dependencies

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)

![image](https://github.com/user-attachments/assets/1974eaa3-e959-4272-b126-ebaf8dce3699)

![image](https://github.com/user-attachments/assets/ba9edf15-99d6-41b1-9923-82ad629c24c6)

![image](https://github.com/user-attachments/assets/5089272c-56d7-4a5e-acd5-a1691ba0f5bf)

![image](https://github.com/user-attachments/assets/3b289c1a-d812-44a9-ae4b-a4cf49a8f706)

![image](https://github.com/user-attachments/assets/96287642-fa65-464c-a97f-de9a1c529869)


__Orchestration Tools:__
Role: Execute the pipeline by managing the workflow defined by the DAG. Function: Handle scheduling, task execution, retries, parallelism, and resource management. Output: Actual execution of the ML pipeline as per the defined DAG.
