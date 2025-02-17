

TEST_RATIO_DICT = {
        "icbhi": None,
        "xinhua_report": 0.1,
        "xinhua_extra": 0.1,
        "xinhua_labeled": 0.1,
        "xinhua_unlabeled": 0.1,
        "korean": 0.1,
        "circor2022": 0.1,
        "sprsound": None,
        "hf_lung": None,
        "cinc2016": 0.1,
        "bowel_sound": None,
        "lung_sound": 0.1,
        "RD@TR": 0.1,
    }

DATA_DIR_DICT = {
        "icbhi": "./data/icbhi/ICBHI_final_database",
        "korean": ".data/korean/korea_heartsound_dataset/",
        "circor2022": "./data/circor2022/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data",
        "sprsound": "./sprsound/SPRSound",
        "hf_lung": "./data/hf_lung/hf_lung",
        "cinc2016": "./data/cinc2016/2016heartsound",
        "bowel_sound": "./data/bowel_sound/data",
        "lung_sound": "./data/lung_sound/wav",
        "RD@TR": "./data/RD@TR/RespiratoryDatabase@TR",
    }