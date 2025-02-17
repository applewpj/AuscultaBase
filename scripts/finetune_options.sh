#! /bin/bash

if [ "$MODEL_TYPE" = "auscultabase" ]
then
    MODEL_PATH=./models/auscultabase/auscultabase_model.ckpt
fi

if [ "$INIT_MODE" = "opera-ct" ]
then
    MODEL_PATH=./models/opera-ct/encoder-operaCT.ckpt
fi

if [ "$INIT_MODE" = "pann" ]
then
    MODEL_PATH="./models/panns/Cnn14_mAP=0.431.pth"
fi

if [ "$INIT_MODE" = "audiomae" ]
then
    MODEL_PATH=./models/audiomae/pretrained.pth
fi

if [ "$INIT_MODE" = "clap" ]
then
    MODEL_PATH="none"
fi


