# Hierarchical Attention Networks for Document Classification

An implementation of a Hierarchical Attention Network for Document Classification from the following paper:

![Paper on Hierarchical Attention Networks for Document Classification](img/paper_screenshot_han_for_doc_class.png)


## Content
This repo contains the following:
- Implementation of the Network Architecture with the following technologies
    - Keras
    - Preprocessing with spaCy
    - Embedding Layer with pretrained Glove vectors
    - Tokenizing with Keras tokenizer
- Jupyter Notebook used during initial development of the network
- REST API server developed with FastAPI for predicting on a trained model and display of the sentence and word attentions in html 

## Train the model
Run the training script `train_script.py`

## Start the server
Running ` uvicorn app.app:app` starts the server under `http://127.0.0.1:8000` 

### Access the API docs
After running the server the API docs can be accessed via `http://127.0.0.1:8000/docs`
 
### Make predictions
Send a HTTP request to the running server, e.g.
```
curl --location --request GET 'http://127.0.0.1:8000/predict/?text=<your-text-to-be-classifier>'
```

### Visualize prediction and attentions
Send a HTTP request to the running server, e.g.
```
http://127.0.0.1:8000/visualize/?text=<your-text-to-be-classifier>
```
The server responds with a static html which should look something like the following:
![Prediction and Attention Visualization Response](img/prediction_and_attention_visualization_response.png)
