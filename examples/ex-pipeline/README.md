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

## (b) パイプライン処理

```
$ guild run end-to-end
```

```
You are about to run end-to-end
Continue? (Y/n)
INFO: [guild] running preprocess: preprocess data=preprocess.end-to-end
function: preprocess.main
data: preprocess.end-to-end
INFO: [guild] running train: train noise=[-0.2, -0.1, 0.0, 0.1, 0.2] x=[0.3, 0.4]
INFO: [guild] Running trial dbf3c877f08e4c05806bd96669668a50: train (noise=-0.2, x=0.3)
function: train.main
x: 0.300000
noise: -0.200000
loss: 1.156103
INFO: [guild] Running trial 4e76f2050c8647fe94b93ecb0f62bd7f: train (noise=-0.2, x=0.4)
function: train.main
x: 0.400000
noise: -0.200000
loss: 0.648886
INFO: [guild] Running trial de71982353bf4727925a289c2566e736: train (noise=-0.1, x=0.3)
:
```
