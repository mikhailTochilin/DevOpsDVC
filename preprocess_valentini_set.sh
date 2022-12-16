#!/bin/bash

mkdir valentini/;
mkdir valentini/tt/;
mkdir valentini/tr/;

unzip raw_data/tt/noisy_testset_wav.zip -d valentini/tt/;
unzip raw_data/tt/clean_testset_wav.zip -d valentini/tt/;
unzip raw_data/tr/noisy_trainset_28spk_wav.zip -d valentini/tr/;
unzip raw_data/tr/clean_trainset_28spk_wav.zip -d valentini/tr/;


mkdir valentini/egs/;
mkdir valentini/egs/tr;
mkdir valentini/egs/tt;

python -m enhancer_toch.audio valentini/tr/noisy_trainset_28spk_wav/ > valentini/egs/tr/noisy.json;
python -m enhancer_toch.audio valentini/tr/clean_trainset_28spk_wav/ > valentini/egs/tr/clean.json;

python -m enhancer_toch.audio valentini/tt/noisy_testset_wav/ > valentini/egs/tt/noisy.json;
python -m enhancer_toch.audio valentini/tt/clean_testset_wav/ > valentini/egs/tt/clean.json;
