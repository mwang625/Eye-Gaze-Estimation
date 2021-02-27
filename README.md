# Project Submission for Eye Gaze Estimation 

## Setup

The following two steps will prepare your environment to begin training and evaluating models.

### Necessary datasets

The required training datasets are located on the Leonhard cluster at `/cluster/project/infk/hilliges/lectures/mp20/project3`.

Please create a symbolic link via commands similar to the following:
```
    cd datasets/
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_train.h5
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_validation.h5
    ln -s /cluster/project/infk/hilliges/lectures/mp20/project3/mp20_test_students.h5
```

### Installing dependencies

Run (with `sudo` appended if necessary),
```
pip install -r requirements.txt
```

Note that this can be done within a [virtual environment](https://docs.python.org/3/tutorial/venv.html). In this case, the sequence of commands would be similar to:
```
    mkvirtualenv -p $(which python3) myenv
    pip install -r requirements.txt
```

when using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

## Structure

* `datasets/` - all data sources required for training/validation/testing.
* `outputs/` - any output for a model will be placed here, including logs, summaries, checkpoints, and submission `.txt.gz` files.
* `src/` - all source code.
    * `core/` - base classes
    * `datasources/` - routines for reading and preprocessing entries for training and testing
    * `models/` - Proposed Models
    * `util/` - utility methods
    * `training.py` -  training script

## Commands for Running on Leonhard Cluster

```
bsub -n 5  -W 24:00 -o log -R "rusage[mem=6200, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]"  python src/training.py
```

