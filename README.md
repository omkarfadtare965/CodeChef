### MLOps:
- After developing a machine learning or neural network model, to deploy the algorithm to production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. In manufacturing, edge deployment is often preferred because it ensures the factory continues operating even when the internet connection goes down.
- For example, The edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.
- 

![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)

### Important terminology:
- __Data drift:__ Data drift refers to changes in the data distribution over time that can negatively impact the performance of a machine learning model. It occurs when, after deployment, the data used for inference differs from the data the model was trained on, causing the model's predictions to become less accurate.
- For example, Imagine you build a machine learning model to predict house prices based on historical data, including features such as square footage, number of bedrooms, location, and year built. The model is trained on data from 2010 to 2020. During training, the model learns patterns in the data, such as the average price per square foot in different neighborhoods and how specific features affect the price. Over time, several factors change in the housing market, such as an economic downturn, changes in local employment rates, or a new housing development that alters demand in certain neighborhoods. As a result, the distribution of house prices shifts.
- __Concept drift:__ Concept drift refers to the phenomenon where the underlying relationship between input data and the target output changes over time. This can lead to a decline in the performance of a machine learning model, as the model may not accurately predict outcomes based on the new patterns in the data that were not present during training.
- For example, Suppose you develop a machine learning model to predict customer churn for a subscription-based service. The model is trained on historical data, which includes features like customer age, subscription length, usage patterns, and customer support interactions. During training, the model learns that older customers with longer subscription lengths and lower usage are more likely to churn. Over time, the service introduces new features that appeal to younger customers, leading to a shift in customer behavior. Younger customers begin subscribing at a higher rate, and usage patterns change. For example, they may prefer shorter subscription plans with flexible cancellation options.
__Edge device:__ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, often used in IoT, manufacturing, and automation.

### Libraries:
- TFX
- Tensorflow
- Keras
- Pytorch

### Digrams for better understanding

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)
