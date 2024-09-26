# CS6120: Sentiment Analysis in Finance by TEAM103

An LLM based sentiment prediction pipeline for financial news.

This project is an extension of [cs6120-finance-sentment](https://github.com/AnnaBrunkhorst/cs6120-finance-sentiment/tree/develop]). Originally developed at Northeastern University as part the course project for CS6120 - Natural Language Processing by Ashish, Anna & Nader.

### Dependencies
To run this project, the following dependencies need to installed:

- **Python 3.12+**, Jupyter Notebook
- NLP/data: NLTK, SpaCy, Gensim, Pandas
- ML: NumPy, PyTorch, Transformers, Datasets, evaluate
- GPU-acceleration: CUDA-toolkit
- HuggingFace: Huggingface-hub, Transformers, Datasets, Evaluate, Accelerate
- UI: StreamLit
- Plots: Matplotlib, Seaborn

### Project Navigation

````
/                                           # Project root
    /data                                   # All datasets are present here
    /models                                 # All saved models go here
    /utils 
        DataUtils.py                        # Data utility functions 
        GloveUtils.py                       # Glove embedding utility functions 
    alpha_vantage_bert.ipynb                # Fine-tuning of BERT-cased, uncased and DistilBERT 
    alpha_vantage_finbert.ipynb             # Fine-tuning of FinBERT 
    alpha_vantage_glove.ipynb               # Training of GloVE+NN model 
    alpha_vantage_log_reg.ipynb             # Training of Logistic regression model 
    demo.py                                 # Entry point to the app. streamlit run demo.py                             
    EDA_data.py                             # Exploratory Data Analysis 
    env                                     # File to place the Vantage Free API key 
    evaluate_models.ipynb                   # Evaluate transformer and neural network models 
    train_test_prep.ipynb                   # train test split preparation 
    unit_tests.py                           # unit tests
    yahoo_bert.ipynb                        # Fine-tuning of BERT model on Yahoo dataset 
    yahoo_glove.ipynb                       # Training of GloVE+NN model on Yahoo 
    yahoo_log_reg.ipynb                     # Training of Logistic Reg on Yahoo 
    README.md                               # This file
````

### Instructions to run
- Request @ShonenAsh to get a download link to models and place them under `./models`
- Ensure that all the dependencies are installed as mentioned above.
- Run the streamlit application - `streamlit run demo.py`
