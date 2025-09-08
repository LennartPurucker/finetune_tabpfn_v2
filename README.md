# ⚠️ This Code is Outdated and not Maintained

This repository contains code to finetune TabPFNv2 and its package `tabpfn<2.1`. 
It is not actively maintained or updated. Please check for an alternative source to fine-tune the newest version of TabPFN.

The code and scripts can still be used to fine-tune TabPFN, but you may need to address compatibility issues with the TabPFN package. 

# Finetuning TabPFN 

Heyho, this is code I use for finetuning [TabPFN(v2)](https://github.com/PriorLabs/TabPFN) on **ONE** downstream 
tabular dataset.

The code is optimized for usage in a larger data science pipeline (i.e., for the usage in AutoML systems such 
as [AutoGluon](https://github.com/autogluon/autogluon)). As a result, it is surprisingly complex. 

**Key Features:**
* Minimal hyperparameter for tuning required, focuses only on learning rate and batch size.
* Adaptive early stopping based on validation loss & early stopping based on a time limit
* Logging, basic wandb support, and offline training curve plots. 
* Support for binary classification, multiclass classification, and regression.
* Mixed precision training, gradient scaling + clipping, gradient accumulation.
* Semi-well-written and docstring-documented code. 

**Key Limitations:**
* A lot of code overhead for finetuning, simply because I want to use it in larger data science pipelines.
* No general-purpose solution so far. It is often tricky to get better by finetuning, and the implementation made  
a lot of assumptions for design decisions that require more research (e.g., data loader design, learning HPs, ...)
* Generally, the finetuning on one dataset is very sensitive to the learning rate, batch size, and overfitting. 

## Examples
See `examples/toy_example` for an example of how to run the code for classification and regression.

Minimal Example:
```python
from finetuning_scripts.finetune_tabpfn_main import fine_tune_tabpfn
from sklearn.datasets import load_iris
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# Load data
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Finetune
save_path_to_fine_tuned_model = "./fine_tuned_model.ckpt"
fine_tune_tabpfn(
    path_to_base_model="auto",
    save_path_to_fine_tuned_model=save_path_to_fine_tuned_model,
    # Finetuning HPs
    time_limit=60,
    finetuning_config={"learning_rate": 0.00001, "batch_size": 20},
    validation_metric="log_loss",
    # Input Data
    X_train=X_train,
    y_train=y_train,
    categorical_features_index=None,
    device="cuda",  # use "cpu" if you don't have a GPU
    task_type="multiclass",
    # Optional
    show_training_curve=True,  # Shows a final report after finetuning.
    logger_level=0,  # Shows all logs, higher values shows less
    use_wandb=False,  # Init wandb yourself, and set to True
)

# Evaluate on Test Data
clf = TabPFNClassifier(model_path=save_path_to_fine_tuned_model).fit(X_train, y_train)
print("Log Loss (Finetuned):", log_loss(y_test, clf.predict_proba(X_test)))
clf = TabPFNClassifier().fit(X_train, y_train)
print("Log Loss (Default):", log_loss(y_test, clf.predict_proba(X_test)))
```

## Install
After cloning the repo, do the following:

This code base requires at least Python 3.10. 

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install uv
uv pip install -r requirements.txt
```

### Do you want to edit the model (for adapters)?

To edit the model, I suggest using a local installation of the TabPFN package. 
```bash
# Local TabPFN install so we can change the model without reinstalling the package (e.g. for adapters)
cd custom_libs && git clone --branch finetune_tabpfn --single-branch https://github.com/LennartPurucker/TabPFN.git
# Install dependencies
uv pip install -e TabPFN && cd .. && uv pip install -r requirements.txt
```

### Do you want faster training on old GPUs?
Use the following to install flash attention 1 if you have a GPU like RTX2080 or T4.
Unsure if this still works, so be careful. 
```bash
uv pip uninstall torch torchvision torchaudio && uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

To test if flash attention is working, put the following code before you run TabPFN
```python
import torch.backends.cuda
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

## Developer Docs

* Add requirements to `requirements.txt`
* Change mypy and ruff settings in `pyproject.toml`
* Make sure the toy example runs without errors.  
