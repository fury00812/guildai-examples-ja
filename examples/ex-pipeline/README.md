# ex-pipeline
複数の操作を組み合わせてパイプライン処理を実行する

## (a) 個別に実行

```
$ guild run preprocess
```

```
You are about to run preprocess
data: preprocess.ad-hoc
Continue? (Y/n)
function: preprocess.main
data: preprocess.ad-hoc
```

```
$ guild run train
```

```
You are about to run train
noise: 0.2
x: 0.2
Continue? (Y/n)
function: train.main
x: 0.200000
noise: 0.200000
loss: 0.702299
```

## 2. パイプライン処理

```
$ guild run end-to-end
```

```
You are about to run end-to-end
Continue? (Y/n)
INFO: [guild] running preprocess: preprocess data=preprocess.end-to-end
function: preprocess.main
data: preprocess.end-to-end
INFO: [guild] running train: train x=0.3
function: train.main
x: 0.300000
noise: 0.200000
loss: 0.696286
```
