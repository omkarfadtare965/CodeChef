### MLOps:
- After developing a machine learning or neural network model, to deploy the algorithm to production, we need to set up an API on the prediction server using Flask or any other web framework, along with the rest of the software code. The prediction server can be either in the cloud or at the edge. In manufacturing, edge deployment is often preferred because it ensures the factory continues operating even when the internet connection goes down.
- For example, The edge device has inspection software with camera control. It captures images and sends an API request to the prediction server. The prediction server processes the images to predict the output and sends the result back to the edge device as a response to the API request. Based on this prediction (API response), the software on the edge device determines whether to accept or reject the product.

![image](https://github.com/user-attachments/assets/e287eafe-d487-4dd2-9ff4-727aeeae81da)


### Important terminology:
__Data drift:__ Data drift refers to changes in the data distribution over time that can negatively impact the performance of a machine learning model. It occurs when, after deployment, the data used for inference differs from the data the model was trained on, causing the model's predictions to become less accurate.
__Edge device:__ An edge device is a piece of hardware that processes data locally, closer to where it is generated, rather than sending it to a centralized server or cloud. It can perform tasks like collecting data, running AI models, or controlling systems, often used in IoT, manufacturing, and automation.

![image](https://github.com/user-attachments/assets/1f3bf615-0919-4da7-84dc-29d4c720308e)




## Libraries:
- TFX
- Tensorflow
- Keras
- Pytorch
