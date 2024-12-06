# Kinopoisk reviews classification

This project focuses on analyzing reviews from the "KinoPoisk" platform using various NLP methods. Originally, this project was created as a "Data Mining course" learning exercise to demonstrate my skills in NLP and ML.

The dataset used can be found here:
https://huggingface.co/datasets/blinoff/kinopoisk

## **Dataset Overview**

Number of reviews: **36,591** (from the Top 250 and the 100 worst movies)  
Time period: **07.04.2004** to **26.11.2012**  
Average review length: **292** words  
Target variable: Three-class rating ("Bad/Neutral/Good")  

## **Project Approaches**

+ **Traditional Analysis:** Vectorized text data using the TF-IDF method and gradient boosting classifiers.  
+ **Deep Learning:** Implementation of a pre-trained BERT model for more comprehensive text analysis.   


## **Project Notebook Overview**

  + graphs_and_EDA.ipynb: Notebook with a detailed data overview, visualizations, and exploratory data analysis.  
  + TFIDF_approach.ipynb: Notebook exploring data vectorized with TF-IDF.  
  + BERT_approach.ipynb: Notebook implementing BERT for text analysis.  
  + preprocessing.py: A small module for text preprocessing.  

## **Results**

1) **TF-IDF Performance**  
Due to the sparsity of the data, TF-IDF performed well only in binary classification tasks. On a balanced dataset, metrics for multiclass classification were suboptimal (F1-score around 0.7).  

2) **BERT Performance**  
The large size of reviews necessitated limiting the number of tokens for BERT. However, the F1-score for multiclass classification was significantly higher, reaching 0.85 on a balanced dataset.

## Future updates
+ Data loading and code readability updates
+ Class balancing through data augmentation.  
+ Character-level vectorization for TF-IDF.  
+ Fine-tuning BERT to improve the differentiation between Neutral and Good classes.  
