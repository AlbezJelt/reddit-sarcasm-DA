# Reddit sarcasm analysis

## Primo esperimento bow_keras_1000

1. Tokenizzazione con TweetTokenizer + join_sarcasm_tag
2. Rimozione stopwords
3. Rimozione punteggiatura
4. Stemming
5. CountVectorizer(1000 feature, 1010822 sample)
6. Keras: val_binary_accuracy: 0.6655 - val_precision: 0.6932 - val_recall: 0.5902
