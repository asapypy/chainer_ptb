# chainer_ptb

ロボケンのみなさま
=================

LSTM は chainer の公式 GitHub からダウンロードしてください。
こちらには GRU を使ったバージョンをアップしたました。

加えて，学習済のファイルもアップしてあります。評価用にお使いください

rnnlm.model
-----------
公式サンプルの学習済結合係数ファイル

rnnlm.state
-----------
同上で，その状態ファイル

train_ptbGRU.py
---------------
GRU を使った PTB の訓練をする python コード

netGRU.py
---------
上記の補助ファイル

rnnlmGRU.model
--------------
GRU 版の学習済結合係数ファイル

rnnlmGRU.state
--------------
上記の状態ファイル


まず最初にすべきこと
===================
Chainer のインストールです。

> pip install chainer

してください。そうしないと話が始まりません。GRU 版の ptb を開始するには

> python train_ptbGRU.py

です。訓練済のファイルから始めるには

> python tran_ptgGRU.py --initmodel rnnlmGRU.model --resume rnnlmGRU.state

です。ですが，ソースコードのとおりこれはサンプルプログラムです。
このようにしても必ず決まった回数だけ訓練してしまいます。
PFN の提供しているサンプルコードですので，自分で改造する必要があります。
ご了承ください。

