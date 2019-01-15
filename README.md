# ChatBot
Simple chat bot using Seq2seq

Requires tensorflow, tensorlayer, nltk packages

Chat.py - script to start work with trained model
TrainModel.py - script to start training model
Settings for models stored in file ModelSettings.py:
  DataCorpus - you can select "MovieDialog" to use Cornell Movie Dialog Corpus or "Twitter" to use twitter chat history
               Also you can prepare script data.py and use your own data to train model.
  BatchSize, EpochNum, LearningRate, EmbeddingSize and percentage of samples - hyperparameters of nets
  ParaphraseFrequency - frequency of applying rephrasing to bot answers. Paraphraser is very simple and stored in paraphrase.py
  WikiSearch - parameter enables the small module, which try to answer questions using wikipedia summary info.
