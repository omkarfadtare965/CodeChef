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

![image](https://github.com/user-attachments/assets/017ad3f4-8ebb-464c-8dd3-02262da8f067)

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

### Deployment patterns:
- When deploying systems there are a number of common use cases or types of use case as well as different patterns for how you deploy depending on your use cases.
- __1) New product:__ Is when you are offereing a new product or capability that you had not previously offered, in this case a common design pattern is to start up small amount of traffic and then gradually ramp it up
- __Automate/Assist with manual task or use shadow deployment:__ Is when somethig thats already being done by a person but we would now like to use a learning algorithm to either automate or assist with the task. For example if you have people in the factory inspecting smartphones scratches but now you would like to use a learning algorithm to either assist or automate this task
- In shadow deployment you will start by having machine  learning algorithm to shadow the human ispector and running parellal with the human inspector. But during this initial phase the leraning algorithm output is not used for any decision in the factory
- The purpose of shadow mode deployemnt is that it allows you to gather data of how the learning algorithm is performing and how that compares to the hhuman judgement. WHich is then used to decide whether or not allow the learning algorithm to make some real decisions in the future
- WHen you are ready to let learning algorithm start making real decisions a canary deployment pattern is used. In a canary deployement you would roll out to small fraction maybe 5% traffic initially and start let the algorithm making real decisions. But by running this on only small percentatge of traffic because if the algorithm makes any mistakes ,it will only affect small fraction of the traffic. After monitoring you can ramp up the percentage of trafficafter you get a good performanc.
- Blue green deployment: The old version is called Blue version and the new version you just implemented is called green version. In blue green deployment it suddenly switch over to the new version. If you would have an old prediction service running so some sort of service You will then spin up a new prediction version servicea dn you will ave a suddenly switch the traffic over from the old one to the new one. The advanctage of blue green deployment  is that there is a easy way to rollback that means if something goes back you can just very quickly have a router go back to  the old version form a new version. You can sent 100% or youcna choose to gradually ramp up.
- MlOps tools help implement these deplpoyment patterns 
- __3) Replace previous ML system__ is when you have already been doing this task with a previous implementation of machine leaning system but now want t replace it with hopefully an even better one. In this scenario you can often want to gradual ramp up with monitoring which means rather than sending tons of traffic to a maybe not fully proven learning algorithm you may send a small amount of traffic and monitor it and then ramp up the percentatge or amount of traffic or rollaback meaning that if for some reason the alsorithmm isnt working its nice if you can revert back to the previous system if needed there was an earlier system.

### Degree of automation
- It is not about deciding whether to deploy or not but it is about the the degree of automation.
- Human only >> Shadow >> AI Assistant >> partial automation >> full automation
- Human only: There is no automation
- SHadow automation: Your learning algorithms output a prediction but its not actually used in the factory
- Ai assistance: In which a peicture of a smartphone you may have a human inspector to make the decisions but an ai sustems can affect the user interface to highlight the regions where there is a scratch to help draw the persons attention to where it may be most useful for them to look. It s a slightly greater degree of automation while still keeping the human in the loop 
- Partial automation: if the learning algorithm is sure it's fine, then that's its decision. It is sure it's defective, then we just go to algorithm's decision. But if the learning algorithm is not sure, in other words, if the learning algorithm prediction is not too confident, 0 or 1, maybe only then do we send this to a human.
- FUll automation: Where a learning algorithm make evry single decision
- In many deployment applications we will start from left that is human only and grdually move to the right that is full automation 
- The degree of automation depends upon the performance and the needs of the application
- AI assistance an partial automation are the examples of human in the loop deployment

![image](https://github.com/user-attachments/assets/28a4375e-9fb5-4577-ae23-86f2608f53e5)

### MOnitoring deployed mcahine learning systems:
- The most common way to monitor a ml system is to use a dashboard totrack how it is doing over time. Depending on your application your dashboard may monitor different metrics such as server load, non null outputs (where ml system does not return an output), missing input values(which will let you know that something is changed about the data)
- To decide metrics sit down with the time and decide what things may possibly go worng 
- Start with a lot of metrics and remove them over time which you do not find useful
  - SOftware metrics: (memory, compute latency throughput server load) to monitor the software health
  - Input mterics, output metrics to monitor the performance of learning algorithm
- machine learning modeling is a highly iterative process, think of deployments as an iterative process as well.   When you get your first deployments up and running and put in place a set of monitoring dashboards. A running system allows you to get real user data or real traffic. It is by seeing how your learning algorithm performs on real data on real traffic that, that allows you to do performance analysis, and this in turn helps you to update your deployment and to keep on monitoring your system.
- After setting of metrics to monitor, common practice would be to set thresholds for alarms. 
 If something goes wrong with your learning algorithm, if is a software issue such as server load is too high, then that may require changing the software implementation, or if it is a performance problem associated with the accuracy of the learning algorithm, then you may need to update your model. Or if it is an issue associated with the accuracy of the learning algorithm, then you may need to go back to fix that that's why many machine learning models will need a little bit of maintenance or retraining over time. When a model needs to be updated, you can either retrain it manually, where in Engineer, maybe you will retrain the model perform error analysis and the new model and make sure it looks okay before you push that to deployment. Or you could use  automatic retraining. 

![image](https://github.com/user-attachments/assets/36140dcf-7597-4223-975d-e918e967ae23)
  
### Pipeline monitooring:
- When building a complex machine learning pipeline based components or non machine elarning based components throughout the pipeline.
- Metrics that can detect changes including both concept drift and data drift or both at multiple stages of the pipline. 

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
