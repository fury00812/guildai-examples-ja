# pytorch-chatbot
- [PytorchのSeq2Seqチュートリアル](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)をGuild AIで実行・管理

## Quick Start
1. 学習
```
$ guild run train
# guild.ymlに記述した設定で学習を行う
```
2. 結果の確認
```
$ guild runs
# [1:f22c3cb2]  train                       2020-05-30 19:16:22  completed

$ guild compare
# 学習結果が一覧表示される
```

3. グラフ表示
```
$ guild tensorboard
```
