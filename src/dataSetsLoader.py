"""#Loading Libraries and Datasets"""

from simpletransformers.classification import ClassificationModel
import pandas as pd
from datasets import load_dataset

datasetTrain = load_dataset("CohereForAI/aya_collection", "aya_dataset")
datasetTest = load_dataset("papluca/language-identification")

aya_dataset = datasetTrain["train"]
papluca_dataset = datasetTest["test"]

"""#Raw datasets
Printing datasets before formatting them.

"""

fracDataSetTrain = aya_dataset.select(range(20))
fracDataSetTest = papluca_dataset.select(range(20))

print("Raw First database:")
print(fracDataSetTrain.to_pandas())
print("Raw Second database:")
print(fracDataSetTest.to_pandas())

"""#Formatting Data
Adjusting the number of records, collumns of interrest, renaming them to match and change their position for better visualization
"""

df = fracDataSetTrain.to_pandas()[['inputs', 'language']]
df2 = fracDataSetTest.to_pandas()[['text', 'labels']]
df2.rename(columns={'text': 'inputs', 'labels': 'language'}, inplace=True)

print("First database (aya_collection):")
print(df)
print("\nSecond database (language-identification):")
print(df2)

"""#Filtering Languages and Mapping them to be Trained"""

from simpletransformers.classification import ClassificationModel
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report

datasetTrain = load_dataset("CohereForAI/aya_collection", "aya_dataset")
datasetTest = load_dataset("papluca/language-identification")

aya_dataset = datasetTrain["train"]
papluca_dataset = datasetTest["test"]

fracDataSetTrain = aya_dataset.select(range(5000))
fracDataSetTest = papluca_dataset.select(range(5000))

df = fracDataSetTrain.to_pandas()[['inputs', 'language']]
df2 = fracDataSetTest.to_pandas()[['text', 'labels']]
df2.rename(columns={'text': 'inputs', 'labels': 'language'}, inplace=True)

allowed_languages1 = ["por", "eng", "fra", "jpn", "spa", "ita", "deu"]
df = df[df['language'].isin(allowed_languages1)]

allowed_languages1 = ["pt", "en", "fr", "ja", "es", "it", "de"]
df2 = df2[df2['language'].isin(allowed_languages1)]

language_mapping = {
    "por": 0, "eng": 1, "fra": 2, "jpn": 3, "spa": 4, "ita": 5, "deu": 6,
    "pt": 0, "en": 1, "fr": 2, "ja": 3, "es": 4, "it": 5, "de": 6
}

# Map languages to numbers
df['language'] = df['language'].map(language_mapping)
df2['language'] = df2['language'].map(language_mapping)

print(df2.info())

print(df)
print("\n","-"*80,"\n")
print(df2)
"""#Loading Libraries and Datasets"""

from simpletransformers.classification import ClassificationModel
import pandas as pd
from datasets import load_dataset

datasetTrain = load_dataset("CohereForAI/aya_collection", "aya_dataset")
datasetTest = load_dataset("papluca/language-identification")

aya_dataset = datasetTrain["train"]
papluca_dataset = datasetTest["test"]

"""#Raw datasets
Printing datasets before formatting them.

"""

fracDataSetTrain = aya_dataset.select(range(20))
fracDataSetTest = papluca_dataset.select(range(20))

print("Raw First database:")
print(fracDataSetTrain.to_pandas())
print("Raw Second database:")
print(fracDataSetTest.to_pandas())

"""#Formatting Data
Adjusting the number of records, collumns of interrest, renaming them to match and change their position for better visualization
"""

df = fracDataSetTrain.to_pandas()[['inputs', 'language']]
df2 = fracDataSetTest.to_pandas()[['text', 'labels']]
df2.rename(columns={'text': 'inputs', 'labels': 'language'}, inplace=True)

print("First database (aya_collection):")
print(df)
print("\nSecond database (language-identification):")
print(df2)

"""#Filtering Languages and Mapping them to be Trained"""

from simpletransformers.classification import ClassificationModel
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report

datasetTrain = load_dataset("CohereForAI/aya_collection", "aya_dataset")
datasetTest = load_dataset("papluca/language-identification")

aya_dataset = datasetTrain["train"]
papluca_dataset = datasetTest["test"]

fracDataSetTrain = aya_dataset.select(range(5000))
fracDataSetTest = papluca_dataset.select(range(5000))

df = fracDataSetTrain.to_pandas()[['inputs', 'language']]
df2 = fracDataSetTest.to_pandas()[['text', 'labels']]
df2.rename(columns={'text': 'inputs', 'labels': 'language'}, inplace=True)

allowed_languages1 = ["por", "eng", "fra", "jpn", "spa", "ita", "deu"]
df = df[df['language'].isin(allowed_languages1)]

allowed_languages1 = ["pt", "en", "fr", "ja", "es", "it", "de"]
df2 = df2[df2['language'].isin(allowed_languages1)]

language_mapping = {
    "por": 0, "eng": 1, "fra": 2, "jpn": 3, "spa": 4, "ita": 5, "deu": 6,
    "pt": 0, "en": 1, "fr": 2, "ja": 3, "es": 4, "it": 5, "de": 6
}

# Map languages to numbers
df['language'] = df['language'].map(language_mapping)
df2['language'] = df2['language'].map(language_mapping)

print(df2.info())

print(df)
print("\n","-"*80,"\n")
print(df2)
