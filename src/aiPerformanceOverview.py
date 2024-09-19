#!pip install openai

"""#Analyzing Results with chatgpts API"""

from openai import OpenAI


prompt = f"""
Given the evaluation metrics for three models (RoBERTa, BERT, and DistilBERT) on a language classification task,
let's analyze the performance of each model:

RoBERTa Metrics:
Accuracy: {accuracy_roberta}
Precision: {precision_roberta}
Recall: {recall_roberta}

BERT Metrics:
Accuracy: {accuracy_bert}
Precision: {precision_bert}
Recall: {recall_bert}

DistilBERT Metrics:
Accuracy: {accuracy_distilbert}
Precision: {precision_distilbert}
Recall: {recall_distilbert}

Based on these metrics, which model performed the best overall and why?
"""

client = OpenAI(api_key='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')
#Dr. Koufakou, after a week I will remove this api key from the project. If the key is not there you can let me know and I will add it again so you can run it.

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Given the evaluation metrics for three models (RoBERTa, BERT, and DistilBERT) on a language classification task, let's analyze the performance of each model:"},
    {"role": "user", "content": f"{prompt}"}
  ]
)

print(completion.choices[0].message.content)

"""#Conclusion
RoBERTa performed the best overall, although it struggled slightly with Portuguese and Italian.
DistilBERT performed closely to RoBERTa but had more issues with English and German.
BERT underperformed compared to the other two models.
RoBERTa had confusion primarily between Portuguese and Italian, and to a lesser extent between Spanish and Italian, and Portuguese and Spanish.
DistilBERT had more problems with English and German, in addition to similar issues with Spanish, Italian, and Portuguese as RoBERTa.
Languages with similar origin can be harder to distinguish depending on the model and the language itself. (english -> deutsch, portuguese -> spanish -> italian )

"""