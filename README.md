# Frysk Aging TTS
This repository collects all the documents and programs for the implementation of the text-to-speech system I developed as part of my Thesis Project for the MSc Voice Technology.

The implementation uses FastSpeech2 based on [ming024 PyTorch implementation](https://github.com/ming024/FastSpeech2). The system was inpired by the pipeline developed for [ChildTTS](https://github.com/C3Imaging/ChildTTS), which in turn is based on [CorentinJ code](https://github.com/CorentinJ/Real-Time-Voice-Cloning).

The main difference from the basic FastSpeech2 model is the addition of age embeddings in order to model and control age of the synthesised voice.

Python version used: Python 3.9.6
