"""# Metric: Precision, Recall, f1-score, support"""

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Data for the three models
models = ['RoBERTa', 'BERT', 'DistilBERT']
accuracies = [accuracy_roberta, accuracy_bert, accuracy_distilbert]
precisions = [precision_roberta, precision_bert, precision_distilbert]
recalls = [recall_roberta, recall_bert, recall_distilbert]

# Calculate F1 scores
f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precisions, recalls)]

# Plotting the metrics
fig, ax = plt.subplots(1, 4, figsize=(20, 6))

# Accuracy
ax[0].bar(models, accuracies, color='skyblue')
ax[0].set_title('Accuracy')
ax[0].set_ylim(0, 1)

# Precision
ax[1].bar(models, precisions, color='salmon')
ax[1].set_title('Precision')
ax[1].set_ylim(0, 1)

# Recall
ax[2].bar(models, recalls, color='lightgreen')
ax[2].set_title('Recall')
ax[2].set_ylim(0, 1)

# F1 Score
ax[3].bar(models, f1_scores, color='gold')
ax[3].set_title('F1 Score')
ax[3].set_ylim(0, 1)

# Adding labels
for i in range(4):
    ax[i].set_ylabel('Score')
    ax[i].set_xlabel('Model')
    ax[i].grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

"""# Metrics: Confusion Matrix"""

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

inverse_language_mapping = {
    0: "por",
    1: "eng",
    2: "fra",
    3: "jpn",
    4: "spa",
    5: "ita",
    6: "deu"
}

def convert_labels_to_lang(labels):
    return [inverse_language_mapping[label] for label in labels]

y_true_lang = convert_labels_to_lang(y_true)

predictions_roberta_lang = convert_labels_to_lang(predictions_roberta)
predictions_bert_lang = convert_labels_to_lang(predictions_bert)
predictions_distilbert_lang = convert_labels_to_lang(predictions_distilbert)

conf_matrix_roberta = confusion_matrix(y_true_lang, predictions_roberta_lang)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_roberta, annot=True, fmt='', cmap='viridis', xticklabels=sorted(inverse_language_mapping.values()), yticklabels=sorted(inverse_language_mapping.values()))
plt.title('RoBERTa Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

conf_matrix_bert = confusion_matrix(y_true_lang, predictions_bert_lang)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_bert, annot=True, fmt='', cmap='viridis', xticklabels=sorted(inverse_language_mapping.values()), yticklabels=sorted(inverse_language_mapping.values()))
plt.title('BERT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

conf_matrix_distilbert = confusion_matrix(y_true_lang, predictions_distilbert_lang)
plt.figure(figsize=(12, 8))
sns.heatmap(conf_matrix_distilbert, annot=True, fmt='', cmap='viridis', xticklabels=sorted(inverse_language_mapping.values()), yticklabels=sorted(inverse_language_mapping.values()))
plt.title('DistilBERT Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()