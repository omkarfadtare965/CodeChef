### MLOps:
- After developing a machine learning or neural network model, to deploy the algorithm to production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. In manufacturing, edge deployment is often preferred because it ensures the factory continues operating even when the internet connection goes down.
- For example, The edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.
- 

![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)

### Machine Learning Project Lifecycle
- __1) Scoping:__ In this phase, you define the project by deciding what to work on and what exactly you want to apply machine learning to. You need to identify the features (X) and the target variable (Y).
  - Gestimate the key matrics, Accuracy, latency (prediction time), throughput (howm many queries per second), resources needed (time, compute, budget).

- __2) Data:__ Next, you collect the data for your algorithm. This includes defining the data sources, establishing a baseline, labeling, and organizing the data.

- __3) Modeling:__ After you have the data, you train the model. This involves selecting an appropriate algorithm, training the model, and performing error analysis. Since machine learning is an iterative process, during error analysis, you may need to update the model or decide to collect more data if necessary.
  - Algorithm/Nueral network architecture code, Hyperparamters

- __4) Deployment:__ In this step, you deploy the model into production. This includes writing the software needed for deployment, monitoring the system, and tracking the incoming data. If the data distribution changes, you will need to update the model.

- __5) Maintenance:__ After the initial deployment, maintenance often involves performing more error analysis and possibly retraining the model. This may also mean taking the feedback from the data you receive and using that data to continuously improve and update the model until a more accurate version is deployed.

### Challenges in deploying machine learning models:
- __1) Machine learning or Statistical issues:__
  - Concept drift
  - Data drift: Gradual change, Sudden change
- __2) SOfttware engine issues:__ Whn you are inplementing a prediction service whose job is to take queries x and output prediction y, yoou have a lot of design choices as how to implement this piece of software that will affect how you implement your software.
   - Do you need real time predictions or Batch predictions
   - Does your prediction service run into clouds or does it run at the edge device
   - How much compute resources you have (CPU/GPU)
   - Latency, Throughpput (Queries per second)
   - Logging
   - Security and Privacy

![image](https://github.com/user-attachments/assets/017ad3f4-8ebb-464c-8dd3-02262da8f067)


### Important terminology:
- __Data drift:__ Data drift refers to changes in the data distribution over time that can negatively impact the performance of a machine learning model. It occurs when, after deployment, the data used for inference differs from the data the model was trained on, causing the model's predictions to become less accurate.
- For example, Imagine you build a machine learning model to predict house prices based on historical data, including features such as square footage, number of bedrooms, location, and year built. The model is trained on data from 2010 to 2020. During training, the model learns patterns in the data, such as the average price per square foot in different neighborhoods and how specific features affect the price. Over time, several factors change in the housing market, such as an economic downturn, changes in local employment rates, or a new housing development that alters demand in certain neighborhoods. As a result, the distribution of house prices shifts.
- __Concept drift:__ Concept drift refers to the phenomenon where the underlying relationship between input data and the target output changes over time. This can lead to a decline in the performance of a machine learning model, as the model may not accurately predict outcomes based on the new patterns in the data that were not present during training.
- For example, Suppose you develop a machine learning model to predict customer churn for a subscription-based service. The model is trained on historical data, which includes features like customer age, subscription length, usage patterns, and customer support interactions. During training, the model learns that older customers with longer subscription lengths and lower usage are more likely to churn. Over time, the service introduces new features that appeal to younger customers, leading to a shift in customer behavior. Younger customers begin subscribing at a higher rate, and usage patterns change. For example, they may prefer shorter subscription plans with flexible cancellation options.
- __Edge device:__ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, often used in IoT, manufacturing, and automation.
- __Data-centric approach:__
- __Model-centric approach:__
- __MlOps:__
- __Real time predictions or Batch predictions:__

### Libraries:
- TFX
- Tensorflow
- Keras
- Pytorch

### Digrams for better understanding

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)
