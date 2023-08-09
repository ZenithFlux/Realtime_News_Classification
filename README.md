# Realtime News Classification
Analyzing and classifying news articles into different categories using text classification.

There are 5 categories in which the model can classify a news article:  
***Tech, Business, Sport, Politics, Entertainment***

This project has been deployed on **AWS BeanStalk**.  
The deployed API is *not currently active*.

![BeanStalk Environment](https://i.ibb.co/sVJH3gD/eb.png)

<details>
<summary> View CD pipeline screenshot (AWS Codepipline) </summary>  

![AWS Codepipeline](https://i.ibb.co/w4C45NJ/codepipeline.png)

</details>

<details>
<summary> View API testing screenshots </summary>  

![Request](https://i.ibb.co/Gn3TDyg/req.png)
![Response](https://i.ibb.co/Hgdz2BB/res.png)

</details>

## Dataset used for training

[BBC News Classification (Kaggle)](https://www.kaggle.com/competitions/learn-ai-bbc/data)

## How to run

1. Install requirements.  
`pip install requirements.txt`

2. Run **application.py** on the Flask server (inbuilt into Flask).  
`flask --app application run`

3. Send a POST request to the server with form data having a field named **article** in which the news article text will reside.

4. You will recieve JSON data from the server as response, with a key **news_type** whose value will be the predicted news article category.