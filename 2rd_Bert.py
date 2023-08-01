#!/usr/bin/env python
# coding: utf-8




import re
import torch
import numpy as np
import pandas as pd

import warnings
import seaborn as sns
import matplotlib.pyplot as plt





from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.snowball import SnowballStemmer

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', 255)

# settings
plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams['axes.titlesize'] = 20     # title size
plt.rcParams['axes.labelsize'] = 18     # axis label size
plt.rcParams['xtick.labelsize'] = 15    # x-axis tick label size
plt.rcParams['ytick.labelsize'] = 15    # y-axis tick label size





df =pd.read_csv(r'mtsamples.csv', index_col = "Unnamed: 0")





# columns EDA
df.columns





# Removing unnecessary spaces

df.columns = df.columns.str.strip()
df = df.drop_duplicates()
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ','_')
df_obj = df.select_dtypes(['object'])
df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())





df['medical_specialty'].unique()





df.shape





# word count for each medical specialty with their percentage

word_count_list =[]

for medical_specialty in df['medical_specialty'].unique():
    df_filter = df.loc[(df['medical_specialty'] == medical_specialty)]
    word_count_temp = df_filter['transcription'].str.split().str.len().sum()
    word_count_list.append(word_count_temp)

word_count_df = pd.DataFrame({
    'Medical Specialty' : df['medical_specialty'].unique(),
    'Word Count' : word_count_list
})

word_count_df['Word Count'] = word_count_df['Word Count'].astype('int')
word_count_df = word_count_df.sort_values('Word Count', ascending=False)

# Calculate total sum of word count
total_word_count = word_count_df['Word Count'].sum()

# Create new column 'Percentage' with word count percentage
word_count_df['Percentage [%]'] = ((word_count_df['Word Count'] / total_word_count) * 100).round(2)

word_count_df.reset_index(drop=True)





# calculating null values
df.isnull().sum().sort_values(ascending = False)





# removing transcription rows that is empty
df = df[df['transcription'].notna()]
df.info()





# drop redundant columns
df =df.drop(['description','sample_name','keywords'], axis=1)
df.head()





# Normaliztion
# converting into lowercase
def lower(df, attribute):
    df.loc[:,attribute] = df[attribute].apply(lambda x : str.lower(x))
    return df

df = lower(df,'transcription')
df.head()





# removing transcription punctuation and numbers

def remove_punc_num(df, attribute):
    df.loc[:,attribute] = df[attribute].apply(lambda x : " ".join(re.findall('[\w]+',x)))
    df[attribute] = df[attribute].str.replace('\d+', '')
    return df

df = remove_punc_num(df, 'transcription')
df_no_punc =df.copy()
df.head()





# tokenising transcription

tk =WhitespaceTokenizer()
def tokenise(df, attribute):
    df['tokenised'] = df.apply(lambda row: tk.tokenize(str(row[attribute])), axis=1)
    return df

df =tokenise(df, 'transcription')
df_experiment =df.copy()
df.head()




# stemming

def stemming(df, attribute):
    # Use English stemmer.
    stemmer = SnowballStemmer("english")
    df['stemmed'] = df[attribute].apply(lambda x: [stemmer.stem(y) for y in x]) # Stem every word.
    return df

df =stemming(df_experiment, 'tokenised')
df.head()





import nltk
nltk.download('stopwords')





# removing stop words
stop = stopwords.words('english')
print(f"Total stop words = {len(stop)}  \n")
print(stop)





def remove_stop_words(df, attribute):
    stop = stopwords.words('english')
    df['stemmed_without_stop'] = df[attribute].apply(lambda x: ' '.join([word for word in x if word not in (stop)]))
    return df

df = remove_stop_words(df, 'stemmed')
df.head()





df =df.drop(['transcription','stemmed', 'tokenised'], axis=1)
df.head()





# Label Encoding

le = preprocessing.LabelEncoder()
le.fit(df['medical_specialty'])
df['encoded_target'] = le.transform(df['medical_specialty'])
df.head()





def bar_plot(df, column_name, vertical=False, title=None, x_label=None, y_label=None, xticks_rotation=0):
    """
    Function to create a bar plot for a specific DataFrame column with count labels.
    :param df: pandas DataFrame
    :param column_name: String, the column name in the DataFrame
    :param vertical: Boolean, whether to rotate the x-axis labels and make the bar plot horizontal
    :param title: String, the title of the plot
    :param x_label: String, the label for the x-axis
    :param y_label: String, the label for the y-axis
    :param xticks_rotation: Int or float, the rotation angle for x-axis labels
    :return: None
    """
    # Count the unique values in the column
    value_counts = df[column_name].value_counts()

    if vertical:
        # Create the bar plot with horizontal bars
        # plt.figure()
        bars = plt.barh(value_counts.index, value_counts.values, color='skyblue')

        # Add the count labels next to the bars
        for bar in bars:
            plt.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, bar.get_width(), ha='left', va='center', fontsize = 14)

        # Set plot title and labels
        plt.title(title or 'Bar plot for column ' + column_name)
        plt.xlabel(x_label or 'Counts')
        plt.ylabel(y_label or column_name)

    else:
        # Create the bar plot with vertical bars
        # plt.figure()
        bars = plt.bar(value_counts.index, value_counts.values, color='skyblue')

        # Add the count labels above the bars
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, bar.get_height(), ha='center', va='bottom', fontsize = 14)

        # Set plot title and labels
        plt.title(title or 'Bar plot for column ' + column_name)
        plt.xlabel(x_label or column_name)
        plt.ylabel(y_label or 'Counts')

        # Rotate x-axis labels if requested
        plt.xticks(rotation=xticks_rotation)

    # Show the plot
    plt.tight_layout()
    plt.show()





bar_plot(df, 'medical_specialty', vertical=False, title='Medical Specialty Category Counts',
        x_label='Specialties', y_label='Instances', xticks_rotation=90)





# Modeling
# Split the data into temporary train and final test sets
temp_train_texts, test_texts, temp_train_labels, test_labels = train_test_split(
    df['stemmed_without_stop'], df['encoded_target'], test_size=0.2, random_state=42)

# Split the temporary training set into final train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    temp_train_texts, temp_train_labels, test_size=0.25, random_state=42)





# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

# Convert our data into torch Dataset
class MedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MedDataset(train_encodings, train_labels.tolist())
val_dataset = MedDataset(val_encodings, val_labels.tolist())





# Specify the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    no_cuda=True,)






# Load the base BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=df['encoded_target'].nunique())

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()





# Tokenize the test data
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

# Convert the test data to a Dataset object
test_dataset = MedDataset(test_encodings, test_labels.tolist())

# Evaluate the model
eval_results = trainer.evaluate(eval_dataset=test_dataset)

print(f"Evaluation loss: {eval_results['eval_loss']}")
print(f"Evaluation accuracy: {eval_results['eval_accuracy']}")






# Make predictions
predictions = trainer.predict(test_dataset)

# The predictions are in logits (i.e., before the softmax) so you need to apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions), dim=-1)

# Get the predicted classes
predicted_classes = torch.argmax(probabilities, dim=-1)







