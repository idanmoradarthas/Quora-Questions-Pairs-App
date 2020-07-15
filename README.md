# Quora-Qusetions-Pairs-App
This research is based on the toturial [BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/).

# Introduction
In this research I'd like to use BERT with the huggingface PyTorch library to fine-tune a model which will perform 
best in question pairs classification. The app is build using Streamlit.

So firstly let's talk about the model and the dataset:
## Bert
Bidirectional Encoder Representations from Transformers (BERT) was released, and pretrained, in late 2018 by Google 
(see original model code [here](https://github.com/google-research/bert)) for NLP (Natural Language Processing) tasks. 
Bert was created originally by [Jacob Devlin](https://www.linkedin.com/in/jacob-devlin-135ab048) with two corpora 
in pre-training: BookCorpus and English Wikipedia.

BERT consists of 12 Transformer Encoding layers (or 24 for large BERT). If you stack Transformer Decoding layers you'll 
GPT model to generate senetances.

You can more information inthe those videos: 

[Transformer Neural Networks - EXPLAINED! (Attention is all you need)](https://youtu.be/TQQlZhbC5ps) 

[BERT Neural Network - EXPLAINED!](https://youtu.be/xI0HHN5XKDo)

## Quora Question Pairs Dataset
[Quora](https://www.quora.com/) is a question-and-answer website where questions are asked, answered, followed, and 
edited by Internet users, either factually or in the form of opinions. Quora was co-founded by former Facebook 
employees Adam D'Angelo and Charlie Cheever in June 2009. website was made available to the public for the first time 
on June 21, 2010. Today the website is available in many languages.

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. 
Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their 
question, and make writers feel they need to answer multiple versions of the same question.

The goal is to predict which of the provided pairs of questions contain two questions with the same meaning. The 
ground truth is the set of labels that have been supplied by human experts. The dataset itself can be downloaded 
from kaggle: [here](https://www.kaggle.com/c/quora-question-pairs/).

# Application
## How to use it?
see the following video:

![Instructions video](./images/streamlit-app-2020-07-15-08-07-66.webm.gif)

## Install
Clone the repo:
```bash
git clone https://github.com/idanmoradarthas/Quora-Questions-Pairs-App.git
cd Quora-Questions-Pairs-App
```
go to the training folder, install the requirements and run the notebook in order to create the model:
```bash
cd training-bert
pip install -r requirements.txt
jupyter notebook
```
Install the requirements in the main folder:
```bash
cd ..
pip install -r requirements.txt
```
Run Streamlit:
```bash
streamlit run app.py
``` 
