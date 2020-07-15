from pathlib import Path

import numpy
import pandas
import streamlit
import torch
from torch.nn import functional
from transformers import BertForSequenceClassification, BertTokenizerFast


@streamlit.cache(allow_output_mutation=True)
def load_model_tokenizer():
    model = BertForSequenceClassification.from_pretrained(
        Path(__file__).parents[0].absolute().joinpath("training-bert").joinpath("model_save"))
    tokenizer = BertTokenizerFast.from_pretrained(str(
        Path(__file__).parents[0].absolute().joinpath("training-bert").joinpath("model_save").absolute()),
        do_lower_case=True)
    return model, tokenizer


def predict(model, encoded_dict_questions):
    model.eval()
    with torch.no_grad():
        logits = model(encoded_dict_questions["input_ids"],
                       token_type_ids=encoded_dict_questions["token_type_ids"],
                       attention_mask=encoded_dict_questions["attention_mask"])
    label = numpy.argmax(logits[0].numpy(), axis=1).flatten()
    return label, pandas.DataFrame(functional.softmax(logits[0], dim=1).detach().numpy(), columns=["False", "True"])


streamlit.title("Quora Questions Pairs App")
streamlit.text("By Idan Morad")
streamlit.markdown(
    "This is a simple application using Streamlit, huggingface PyTorch library and a fined-tune BERT model to classify "
    "questions pairs as duplication or not from the Quora website.")
streamlit.markdown("## How Does it work?")
streamlit.markdown("This research is based on the toturial "
                   "[BERT Fine-Tuning Tutorial with PyTorch](https://mccormickml.com/2019/07/22/BERT-fine-tuning/).")
streamlit.markdown("Under training-bert folder you can find a Jupyter notebook. There I show how I fined-tune "
                   "base-uncased bert model to solve the classification problem of duplication questions from "
                   "Quora website.")
streamlit.markdown("## How to use the App?")
streamlit.markdown("This very simple. Fill the ``First question`` and ``Second question`` text inputs and click"
                   " the button ``Check if duplicates``.")

question_1 = streamlit.text_input("First question:", max_chars=512)
question_2 = streamlit.text_input("Second question:", max_chars=512)

if streamlit.button("Check if duplicates"):
    if not question_1 and not question_2:
        streamlit.text("empty questions")
    else:
        model_load_state = streamlit.text("Loading model...")
        bert_model, bert_tokenizer_fast = load_model_tokenizer()
        model_load_state.text("Loading model...done!")

        tokenizer_load_state = streamlit.text("Applying tokenizer...")
        streamlit.text("Extracted tokenization:\n")
        encode = bert_tokenizer_fast.encode(question_1, question_2)
        encoded_frame = pandas.DataFrame(encode, columns=["Token ids"])
        encoded_dict = bert_tokenizer_fast.encode_plus(question_1, question_2,
                                                       max_length=310,
                                                       pad_to_max_length=True, return_attention_mask=True,
                                                       return_tensors="pt", truncation=True)
        encoded_frame["Tokens"] = encoded_frame["Token ids"].apply(lambda token: bert_tokenizer_fast.decode([token]))
        encoded_frame["Token Type ids"] = encoded_dict["token_type_ids"].numpy().flatten()[:encoded_frame.shape[0]]
        streamlit.dataframe(encoded_frame.transpose())
        tokenizer_load_state.text("Tokenization ... done!")

        model_apply_state = streamlit.text("Predicting ...")
        y_pred, predict_proba = predict(bert_model, encoded_dict)
        model_apply_state.text(f"Is duplicate: {True if y_pred == 1 else False}")
        streamlit.text("Probabilities:")
        streamlit.dataframe(predict_proba)
