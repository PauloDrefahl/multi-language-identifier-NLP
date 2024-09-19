"""#Training roBERTa with First Dataset"""

model = ClassificationModel(
    "roberta",
    "roberta-base",
    num_labels=7,
    use_cuda=False,
    args={"reprocess_input_data": True, "overwrite_output_dir": True},
)

model.train_model(df, text_column="inputs", label_column="language")

"""# training Bert Uncased"""

model_bert = ClassificationModel(
    "bert",
    "bert-base-uncased",
    num_labels=7,
    use_cuda=False,
    args={"reprocess_input_data": True, "overwrite_output_dir": True},
)

model_bert.train_model(df, text_column="inputs", label_column="language")

"""#Training Distilbert uncased"""

model_distilbert = ClassificationModel(
    "distilbert",
    "distilbert-base-uncased",
    num_labels=7,
    use_cuda=False,
    args={"reprocess_input_data": True, "overwrite_output_dir": True},
)

model_distilbert.train_model(df, text_column="inputs", label_column="language")



"""#Testing trained roBERTa

"""

from sklearn.metrics import accuracy_score, precision_score, recall_score

predictions_roberta, _ = model.predict(df2['inputs'].tolist())
y_true = df2['language']

accuracy_roberta = accuracy_score(y_true, predictions_roberta)
precision_roberta = precision_score(y_true, predictions_roberta, average='weighted')
recall_roberta = recall_score(y_true, predictions_roberta, average='weighted')

print("RoBERTa Metrics:")
print("Accuracy:", accuracy_roberta)
print("Precision:", precision_roberta)
print("Recall:", recall_roberta)
print()

"""#Testing BERT"""

predictions_bert, _ = model_bert.predict(df2['inputs'].tolist())
accuracy_bert = accuracy_score(y_true, predictions_bert)
precision_bert = precision_score(y_true, predictions_bert, average='weighted')
recall_bert = recall_score(y_true, predictions_bert, average='weighted')

print("BERT Metrics:")
print("Accuracy:", accuracy_bert)
print("Precision:", precision_bert)
print("Recall:", recall_bert)
print()

"""#Testing distilBERT"""

predictions_distilbert, _ = model_distilbert.predict(df2['inputs'].tolist())
accuracy_distilbert = accuracy_score(y_true, predictions_distilbert)
precision_distilbert = precision_score(y_true, predictions_distilbert, average='weighted')
recall_distilbert = recall_score(y_true, predictions_distilbert, average='weighted')

print("DistilBERT Metrics:")
print("Accuracy:", accuracy_distilbert)
print("Precision:", precision_distilbert)
print("Recall:", recall_distilbert)
print()
