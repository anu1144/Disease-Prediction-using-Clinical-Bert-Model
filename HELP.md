# **Table Of Content**
<!-- TOC -->
  * [**PURPOSE**](#purpose)
* [**INTRODUCTION**](#introduction)
  * [**Gathering the Data**](#gathering-the-data)
  * [**Model Building**](#model-building)
  * [**Inference**](#inference)
  * [**Reference Implementation**](#reference-implementation)
    * [**Dataset**](#dataset)
  * [**Model Training**](#model-training)
    * [optional arguments](#optional-arguments)
* [**Model Inference**](#model-inference)
  * [**Model quantization**](#model-quantization)
  * [**optional arguments**](#optional-arguments-1)
  * [**REFERENCE SOLUTION**](#reference-solution)
  * [**Expected Input-Output**](#expected-input-output)
<!-- TOC -->

## **PURPOSE**
The typical hospital generates 50 petabytes of data each year, much of which is unstructured or semi-structured 
and trapped in the notes sections of electronic health records (EHR) systems, not readily available for analysis.
To gain a more comprehensive picture of patient health from unstructured data, healthcare payers routinely resort
to expensive chart reviews in which clinical professionals manually comb through patient records in search of 
nuggets of useful information which can help them assign appropriate risk categories to each patient. 
The difference between an incorrect code and a correct code can cost millions of dollars per year per patient. 
Payers are beginning to use natural language processing (NLP) to understand nuanced language within a body of 
text to improve risk adjustment, reduce costs, and enhance patient care.

In this project we demonstrate one possible reference implementation of a Deep Learning based NLP pipeline which
aims to train a document classifier that takes in notes about a patients symptoms and predicts the diagnosis 
among a set of known diseases. Specifically, we fine tune a pre-trained Clinical BERT embeddings to perform
document classification. Clinical BERT embeddings are a specialization of classical BERT embeddings to 
know about clinical jargon, which is a specific intricacy of medical NLP applications.


# **INTRODUCTION**
![img_2.png](img_2.png)
## **Gathering the Data**

Data preparation is the primary step for any machine learning problem. We will be using a dataset from Kaggle for this problem.
This dataset consists of two CSV files one for training and one for testing. There is a total of 133 columns in the dataset out of which 132 columns
represent the symptoms and the last column is the prognosis. Cleaning the Data:Cleaning is the most important step in a machine learning
project. The quality of our data determines the quality of our machine-learning model. So it is always necessary to clean the data before feeding it to the
model for training. In our dataset all the columns are numerical, the target column i.e. prognosis is a string type and is encoded to numerical form using a
label encoder.

## **Model Building**

After gathering and cleaning the data, the data is ready and can be used to train a machine learning model. We will be using this cleaned
data to train the Clinical Bert model.
Quantization: 
Python program designed to quantize a pre-trained transformer-based model (likely for natural language processing tasks) using post-training
quantization.

## **Inference**

After training the model we will be predicting the disease for the input symptoms by using the saved model. This makes our overall prediction
more robust and accurate.

## **Reference Implementation**

The reference kit implementation is a reference solution to the described use case that includes:
 1.A reference E2E architecture to arrive at an AI solution with PyTorch using ClinicalBERT

### **Dataset**

The dataset used for this demo is a synthetic symptom and diagnosis dataset obtained from https://www.kaggle.com/kaushil268/disease-prediction-using-machine-learning.
In this dataset, each row corresponds to a list of symptom names and the corresponding diagnosis for that particular set of syptoms.  The original dataset consists of 
indicators for the symptom names however for our purposes, we first transform the data from indicators to string descriptions to emulate a
situation where the symptoms come in the form of text.

## **Model Training**

Using the patient symptom descriptions and diagnoses, we train a fine-tuned ClinicalBERT model for text classification, ingesting free-form symptom description
text and outputting the predicted diagnosis probabilities of the provided text.ClinicalBERT is a set of pre-trained embeddings with a focus on medical terminology
and allows our model to be better prepared for medical contexts.

cd src
conda activate disease_pred_stock
python run_training.py --logfile ../logs/stock.log --save_model_dir ../saved_models/stock --data_dir ../disease-prediction/src/utils

usage: run_training.py [-h] --data_dir DATA_DIR [--logfile LOGFILE] [--stock] [--save_model_dir SAVE_MODEL_DIR] [--seq_length SEQ_LENGTH] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--grad_norm GRAD_NORM] [--bert_model BERT_MODEL]

### optional arguments

  -h, --help            show this help message and exit
  --data_dir DATA_DIR   Directory with Training.csv and Testing.csv
  --logfile LOGFILE     Log file to output benchmarking results to.
  --intel               Use intel accelerated technologies where available.
  --save_model_dir SAVE_MODEL_DIR
                        Directory to save model under.
  --seq_length SEQ_LENGTH
                        Sequence length to use when training.
  --batch_size BATCH_SIZE
                        Batch size to use when training.
  --epochs EPOCHS       Number of training epochs.
  --grad_norm GRAD_NORM
                        Gradient clipping cutoff.
  --bert_model BERT_MODEL
                        Bert base model to fine tune.

which will output a saved model at `saved_models/stock` and log timing information to`logs/stock.log`.
The saved model from the model training process can be used to predict the disease
probabilities from a new NLP symptom string.

# **Model Inference**

Similar to model training, the `run_inference.py` script includes a command line 
flag which enables the  optimizations for the passed in trained model.

cd src
conda activate disease_pred_stock
python run_inference.py --saved_model_dir ../saved_models/stock --batch_size 1 --seq_length 64 --logfile ../logs/stock.log --n_runs 100

## **Model quantization**

Model quantization is the practice of converting the FP32 weights in Deep Neural Networks to a lower precision, such as INT8 in order **to accelerate computation time and reduce storage space of trained models**.  This may be useful if latency and throughput are critical.

cd src
conda activate disease_pred_stock
python run_quantize_inc.py --input_file ../src/utils/Testing.csv --batch_size 1 --saved_model_dir ../saved_models/stock --output_dir OUTPUT_DIR --seq_length 64 

The `run_quantize_inc.py` script takes the following arguments:

usage: run_quantize_inc.py [-h] --input_file INPUT_FILE [--batch_size BATCH_SIZE] --saved_model_dir SAVED_MODEL_DIR --output_dir OUTPUT_DIR [--seq_length SEQ_LENGTH] 

## **optional arguments**

  -h, --help            show this help message and exit
  --input_file INPUT_FILE
                        input data to evaluate on.
  --batch_size BATCH_SIZE
                        batch size to use. Defaults to 10.
  --saved_model_dir SAVED_MODEL_DIR
                        saved pretrained model to quantize.
  --output_dir OUTPUT_DIR
                        directory to save quantized model to.
  --seq_length SEQ_LENGTH
                        sequence length to use.


## **REFERENCE SOLUTION**

The implementation for our AI-based NLP Disease Prediction system is based around
fine-tuning and using a ClinicalBERT-based document classifier.  The pipeline
ingests text statements/documents and outputs the probability of diagnoses for 
a known set of diseases.

> Patient Document/Notes => **Classify Document to Predicted Disease** => Measured Patient Level Risk

Given a set of documents, such as doctors notes or a list of patient systems, the implemented AI system must understand the context of document and ultimately map 
this to the disease that the document is most likely describing.  As documents are often written by a human using Natural Language descriptions, a powerful model which
takes these into account is necessary for good performance.  Here, we choose to implement such a classifier using a Deep Learning based NLP model.  Specifically, 
we fine tune a pre-trained ClinicalBERT embeddings to perform document classification. ClinicalBERT embeddings are a specialization of classical BERT embeddings to know about
clinical jargon, which is a specific intricacy of medical NLP applications.


## **Expected Input-Output**


Input : Patient summary Text
Output : [For each disease $d$, the probability [0, 1] that patient suffers from disease $d$]

**Example:**

**_Input_** : Itching. Reported signs of dischromic patches. Patient reports no patches in throat.
        Issues of frequent skin rash. Patient reports no spotting urination. 
        Patient reports no stomach pain. nodal skin eruptions over the last few days.	

**_Output_** : {'Fungal infection' : 0.378, 'Impetigo' : 0.0963, 'Acne' : 0.0337, ...}