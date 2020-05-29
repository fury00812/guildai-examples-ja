# pytorch-chatbot
- [Pytorchのチャットボットチュートリアル](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)をGuild AIで実行・管理

## Quick Start
1. 学習
```
$ guild run train
# 学習データのダウンロードから学習までを行う. 設定はguild.ymlに記述
```
2. 結果の確認
```
$ guild runs
# [1:c0c6d26d]  train (examples/pytorch-chatbot)  2020-05-29 17:12:31  completed

$ guild compare
# 学習結果が一覧表示される
```

3. グラフ表示
```
$ guild tensorboard
```
![スクリーンショット 2020-05-29 17 34 25](https://user-images.githubusercontent.com/35480446/83239399-c8a9d980-a1d2-11ea-9e2e-c5527f554e96.png)
