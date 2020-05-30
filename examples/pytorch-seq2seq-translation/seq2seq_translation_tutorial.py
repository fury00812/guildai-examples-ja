import unicodedata
import string
import re
import random
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True)
parser.add_argument('--src_lang', default='ja')
parser.add_argument('--tgt_lang', default='en')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--n_iters', type=int, default=75000)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--print_every', type=int, default=1000)
parser.add_argument('--MAX_LENGTH', type=int, default=50)
args = parser.parse_args()

BOS_token = 0
EOS_token = 1
MAX_LENGTH = args.MAX_LENGTH

# 単語とインデックスを相互変換するためのヘルパークラス. ベクトル化はまた別.
# 辞書クラス
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}  # 単語の出現回数を数えておく. 辞書作成用
        self.index2word = {0: "BOS", 1: "EOS"}
        self.n_words = 2  # Count BOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 前処理
# Unicode正規化と, 句点のトリムなど
def normalizeString(s):
    s = unicodedata.normalize("NFKC", s)  # 日本語使うのでNFKCにしてみた
    s = s.lower().strip("\r\n")  # アルファベットを小文字化&改行除去
    s = re.sub(r"([.!?])", r" \1", s)  # punctuationをトリム. 何故読点を切らないのかは分からない
    return s


def readLangs(lang1, lang2, file_name):
    print("Reading lines...")
    
    # ファイルを読み込み各行をリストに放り込む
    lines = open(file_name).read().strip().split("\n")

    # タブで区切って文ペアを作成
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # 各言語の辞書クラスを作成
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def filterPair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
            len(pair[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, file_name):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, file_name)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size  # initHiddenで呼ぶためのメンバ変数化
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):  # 文ペア学習毎に初期化
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)  # 何故ReLU?
        output, hidden = self.rnn(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):  # 使わないけど
        return torch.zeros(1, 1, self.hidden_size, device=device)


# 文→インデックス ['Hello world'] -> [4, 10] 
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]
#    ndarrayを使う場合 
#    return np.array([lang.word2index[word] for word in sentence.split(' ')])


# 文→テンソル
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)  # 文→インデックス
    indexes.append(EOS_token)
    #  この時点でGPUに乗る (device='cuda'としているので)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
#    ndarrayを使う場合
#    return torch.from_numpy(np.array(indexes)).to(device)


# 文ペア→テンソルペア
# *_langはLangオブジェクト
def tensorsFromPair(input_lang, target_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)  # タプルとして返す


# モデルの学習(一対)を行う関数
# * teacher forcingは行なったり行わなかったりできる (teacher_forcing_ratioで割合を選択)
# * -> Scheduled Sampling
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
    criterion, max_length=MAX_LENGTH, teacher_forcing_ratio = 0.5):
   
    # RNN隠れ層の初期化 (Encoder)
    encoder_hidden = encoder.initHidden()

    # 勾配の初期化
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 文長を取得
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

#    source-target Attentionを使う場合必要
#    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
#        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[BOS_token]], device=device)

    # RNN隠れ層の初期化 (Decoder)
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]

    else:
        # Free running: Use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)  # topv: 1-best, topi: インデックス 
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    # 誤差逆伝播
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


# 秒 → 分・秒 
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# 経過時間と完了までの残り時間の見積もり
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


def trainIters(input_lang, output_lang, encoder, decoder, n_iters,
            print_every, learning_rate):
    start = time.time()
#    plot_losses = []
    print_loss_total = 0
#    plot_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    # バッチ学習ではない
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                        for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
#        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Iteration: {}; Average loss: {:.4f}'.format(iter, print_loss_avg))
#            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
#                                                     iter, iter / n_iters * 100, print_loss_avg))
    
#        if iter % plot_every == 0:
#            plot_loss_avg = plot_loss_total / plot_every
#            plot_losses.append(plot_loss_avg)
#            plot_loss_total = 0
#    
#        showPlot(plot_losses)


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData( \
                args.src_lang, args.tgt_lang, args.file_path)
    print(random.choice(pairs))

    hidden_size = args.hidden_size
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)

    trainIters(input_lang, output_lang, encoder1, decoder1, args.n_iters, \
                args.print_every, args.learning_rate)
