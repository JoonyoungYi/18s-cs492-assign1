# 18s-cs492-assign1
* 2018년 봄 KAIST CS492C 딥러닝 수업 Assignment1

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
    python run.py
```
