# Machine learning Operations (MlOps):
- After developing a machine learning or neural network model, to deploy the algorithm into production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. Edge deployment is often preferred in the manufacturing domain because it ensures the factory continues operating even when the internet connection goes down.
- For example, The edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.

![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)

### Machine learning Project lifecycle
- __1) Scoping:__ In this phase, you define the project, identify features (X) and target (Y), and estimate key metrics like accuracy, latency(prediction time), throughput(queries per second), and resource needs(time, compute, budget).
- __2) Data:__ In this phase, you Collect and organize data, define data sources, establish baselines, and label the data.
- __3) Modeling:__ In this phase, you select the algorithm, train the model, and perform error analysis. You then adjust the model and perform hyperparameter tuning. Since machine learning is an iterative process, you may need to update the model or decide whether to collect more data or not, followed by further error analysis.
- __4) Deployment:__ In this phase, you deploy the model into production which also includes monitoring the system, and tracking the incoming data.
- __5) Maintenance:__ After the initial deployment, you may need to retrain the model using newly collected data to continuously improve and update it until a more accurate version is deployed.

![image](https://github.com/user-attachments/assets/be8511b5-0b15-43a5-8f6b-e28d49a620f9)

### Challenges in deploying machine learning models:
- __1) Machine learning or Statistical issues:__
  - Concept drift
  - Data drift
- __2) Software engine issues:__
   - whether you need real-time predictions or batch predictions?
   - whether your prediction service run in the cloud, or on edge devices?
   - How many compute resources do you have(CPU/GPU)?
   - Considerations for latency, throughput(queries per second)
   - Logging(to track the behaviour and performance of a deployed machine learning model)
   - Security and privacy, when dealing with sensitive data.

### Deployment patterns:
- __1) New product:__ When launching a new product that hasn’t been offered before, a common pattern is to direct a small portion of traffic to the model, and then gradually increase it as confidence grows. This reduces the risk of impacting a large number of users with an untested system.
- For example, a company releases a new recommendation system. Initially, only 5% of users get recommendations from the new system. As the system proves reliable, traffic is gradually increased until all users are served by it.
- __2) Automate or Assist manual tasks:__ This pattern is useful when replacing or supporting a manual process with an algorithm. For example, if factory workers inspect smartphones for scratches, and you want to use machine learning to assist or replace human inspectors.
  - __Shadow deployment:__ The machine learning algorithm "shadows" the human worker by running in parallel but doesn’t make any actual decisions. The output from the algorithm is compared with the human’s decisions to evaluate its performance.
  - For example, a factory inspector and an algorithm both check for scratches on phones, but only the human inspector's decision matters at first. Over time, the algorithm's performance is analyzed before it’s allowed to make real decisions.
  - __Canary deployment:__ Once confident, you can let the algorithm start making decisions with a small percentage of traffic(e.g., 5%). This way, mistakes will only impact a small part of the system. As the model performs well, the traffic can be gradually increased.
  - For example, the algorithm starts inspecting only 5% of phones. If it performs well, this is increased to 50%, and eventually, all phones are inspected by the algorithm.
- __3) Replacing a previous machine learning system:__ When replacing an older machine learning system with a newer one, it’s common to gradually direct traffic to the new system while monitoring its performance.
  - __Blue-Green deployment:__ This involves running two versions of a system: an old version (Blue) and a new version (Green). At deployment, traffic is switched entirely from Blue to Green. If there’s an issue with the new system, you can quickly revert traffic back to the Blue version, ensuring minimal disruption.

### Degree of automation
- __1) Human only:__ No automation is involved; all decisions are made entirely by humans.
- __2) Shadow automation:__ Learning algorithms make predictions, but these are not applied in the actual process. For example, a system might predict factory machine failures but the predictions are not acted upon.
- __3) AI Assistance:__ In this stage, AI helps humans by providing insights or suggestions. For example, when inspecting a smartphone, an AI might highlight areas with scratches to guide the human inspector to those spots but the final decision is still made by the human.
- __4) Partial automation:__ In this case, the learning algorithm makes decisions when it is confident(e.g., determining if a product is defective or not). If the algorithm is unsure, the decision is referred to a human.
- __5) Full automation:__ The learning algorithm handles every decision without human intervention.
- In many real-world deployments, we begin with Human-only decisions and gradually shift towards Full automation as confidence grows. The level of automation chosen depends on the performance of the AI and the usecase. For example, in healthcare, you might stop at partial automation where AI assists doctors rather than making all decisions.
- AI Assistance and Partial Automation are both examples of "human-in-the-loop" systems, where humans remain involved in the decision-making process

![image](https://github.com/user-attachments/assets/28a4375e-9fb5-4577-ae23-86f2608f53e5)

### Monitoring deployed machine learning systems:
- The most common way to monitor a machine learning (ML) system is by using a dashboard that tracks its performance over time. Depending on your application, the dashboard may monitor different metrics, such as server load, non-null outputs (where the ML system fails to return an output), and missing input values (which can indicate that something in the input data has changed).
- The metrics used are decided by discussing potential issues that could arise in the system. A good practice is to start with a lot of metrics and gradually remove those that are not useful.
  - __Software metrics:__ includes memory usage, compute latency, throughput, and server load metrics
  - __Input and output metrics:__ includes metrics to track performance of the ML model
- Machine learning modeling and deployment are both highly iterative processes. Once your initial deployment is running, you can set up monitoring dashboards and thresholds to track the model's performance on real traffic and trigger alarms. Based on the metrics, you may need to update your deployment.
- When a model needs updating, there are two main approaches:
  - __Manual retraining:__ In manual retraining, an engineer may retrain the model, perform error analysis on the new model, and deploy the updated version.
  - __Automatic retraining:__ In automatic retraining, the retraining process occurs automatically. Based on predefined rules or performance metrics, the new updated version of the model is deployed without manual intervention.

![image](https://github.com/user-attachments/assets/36140dcf-7597-4223-975d-e918e967ae23)

### Important terminology:
- __Data drift:__ Data drift occurs when the distribution of data changes over time, leading to a decline in model performance. This happens when the data used for predictions differs from the data the model was originally trained on.
- For example, a model predicting house prices may be trained on data from 2010 to 2020. If economic factors or housing trends shift, such as a recession or new developments, the data distribution changes, causing the model's predictions to become less accurate.
- __Concept drift:__ Concept drift occurs when the relationship between input data and the target output changes over time, leading to a decline in model performance.
- For example, a model predicting customer churn may be trained on patterns like older customers with long subscriptions being more likely to churn. If the business introduces new features that attract younger customers with different behaviors, the model's performance may decline as the underlying customer patterns shift.
- __Edge device:__ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, often used in IoT, manufacturing, and automation.
- __Data-centric approach:__ Data-centric approach focuses on improving the quality and quantity of the data used to train models.
- __Model-centric approach:__ Model-centric approach focuses on improving the algorithms or models parameters regardless of data.
- __MlOps:__ MlOps (Machine Learning Operations) is a way to manage and deploy machine learning models quickly and efficiently into production. 
- __Real time predictions:__ It involves making predictions instantly as new data comes in.
- __Batch predictions:__ It involves making predictions on a group of data at once, rather than one at a time.

### Libraries:
- TFX
- Tensorflow
- Keras
- Pytorch

### Tools: 
- __Experiment tracking tools:__ Experiment tracking tools, such as text files, shared spreadsheets, or specialized platforms like Weights and Biases, Comet, MLflow, and SageMaker Studio, help in organizing and tracking machine learning experiments like algorithm and code version, dataset used, hyperparameters, performance metrics(accuracy, precision, recall, f1 score). 

### Digrams for better understanding

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)
