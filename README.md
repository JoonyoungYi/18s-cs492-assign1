# Study Tensorflow
* This is to practice tensorflow.

## Forked From multiple sources.
* This repository is mainly forked from [KAIST CS492C course(Introduction to Deep Learning)](https://github.com/hyunwooj/KAIST-CS492C-Spring2018).
  * KAIST CS492C deep learning course tensorflow session.

* Also, I used this [repository](https://github.com/golbin/TensorFlow-Tutorials)


# Environment
* Python 3.5 and TensorFlow 1.4
* local.txt only for development setting. local.txt packages is not necessary for production.
```
    virtualenv .venv -p python3.5
    . .venv/bin/activate
    pip install -r requirements/common.txt
    pip install -r requirements/local.txt
```

* If you don't want to install gpu version of tensorflow, please remove below line in `requirements.txt` or make that line be comment.
```
tensorflow-gpu==1.4
```

# Run
```
    . .venv/bin/activate
    python launcher.py
```
