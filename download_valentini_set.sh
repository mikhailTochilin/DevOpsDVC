#!/bin/bash


mkdir raw_data/;
mkdir raw_data/tr;
mkdir raw_data/tt;

curl -o raw_data/tt/noisy_testset_wav.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y;
curl -o raw_data/tt/clean_testset_wav.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y;
curl -o raw_data/tr/noisy_trainset_28spk_wav.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y;
curl -o raw_data/tr/clean_trainset_28spk_wav.zip https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y;

