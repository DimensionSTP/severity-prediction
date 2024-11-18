# Basic ML pipeline for Tabular dataset classification

## For post-infectious disease severity tabular dataset classification

### Dataset
Post-infectious disease severity tabular dataset

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/severity-prediction.git
cd severity-prediction

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting
```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
USER_NAME={USER_NAME}
```

### Preprocessing(split data)

* end-to-end
```shell
python src/preprocessing/split_data.py
```

### Model Hyper-Parameters Tuning

* end-to-end
```shell
python main.py mode=tune num_trials={num_trials}
```

### Training

* end-to-end
```shell
python main.py mode=train is_tuned={tuned or untuned} num_trials={num_trials}
```

### Test

* end-to-end
```shell
python main.py mode=test is_tuned={tuned or untuned} num_trials={num_trials}
```

### Prediction

* end-to-end
```shell
python main.py mode=predict is_tuned={tuned or untuned} num_trials={num_trials}
```

### Examples of shell scipts

* preprocessing(split data)
```shell
bash scripts/split_data.sh
```

* tune
```shell
bash scripts/tune.sh
```

* train
```shell
bash scripts/train.sh
```

* test
```shell
bash scripts/test.sh
```

* predict
```shell
bash scripts/predict.sh
```

* end to end(tune, train, test, predict)
```shell
bash scripts/end_to_end.sh
```


__If you want to change main config, use --config-name={config_name}.__

__If you want to use main config as huggingface, set modality={modality}, upload_user={upload_user}, model_type={model_type}.__

__Also, you can use --multirun option.__

__You can set additional arguments through the command line.__