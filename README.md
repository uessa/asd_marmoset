<!-- -*- coding: utf-8 -*- -->

# 仮想環境の準備
1. 学習用ルートディレクトリの名前と置く場所を決める
    - 例: ホームディレクトリ ~/ に chainer_test という名前で置く
1. 置く場所に chainer_template を clone する
   ``` shell
   $ cd ~/
   $ git clone git@github.com:onolab-tmu/Chainer_template.git chainer_test
   $ cd chainer_test
   ```
   - ssh key を登録していない人はこちら
   ``` shell
   $ git clone https://github.com/onolab-tmu/Chainer_template.git chainer_test
   ```

1. 仮想環境の作成
    - 仮想環境名を考える（例: 2021_ICASSP_kyamaoka）
	- 追記: 現状，以下の方法でやるしかないっぽい．有識者は知見を分けてほしい
	    ``` shell
		$ conda create -n 2021_ICASSP_kyamaoka --clone /home/kyamaoka/.conda/envs/chainer_template
		```
    - 方法1: 仮想環境 chainer_template を複製する
	    ``` shell
		$ conda create -n 2021_ICASSP_kyamaoka --clone chainer_template
		```
    - 方法2: 仮想環境 chainer_template を再構成する
	    ``` shell
		$ conda create -n 2021_ICASSP_kyamaoka -f env.yml
		```
	- 方法3: 自力で構成する
	    - chainer_template に手動でインストールしたパッケージ群は yml.txt に記載されている

1. 仮想環境のアクティベート (ログインするたび毎回必要)
    ```
	$ conda activate 2021_ICASSP_kyamaoka
	```


# データセットの準備
- Notes:
	- コードブロックはコマンドの例を示し，以下の表記を用いる
	- 元のデータセット名: database
	- データセット名: dataset1
	- サブデータセット: subset1
	- 引数を取る .py については，`python xxx.py --help` でヘルプが表示される

## データとラベルの準備
- 作業一覧
	- 元のデータセットから，使用するデータをコピーし，生データのデータセットを用意する
	- 必要に応じてラベルも用意する
		- ラベルが特徴量に依存する場合（フレーム毎の推定など），ここでは用意しない
		- データに依存する場合は，ここで用意しても良い

1. データセットの配置
	- root/raw/ 以下にデータセット名のディレクトリを作成し，データを配置する

		``` shell
		$ mkdir raw/dataset1
		$ cp database/*.wav raw/dataset1/.
		```

1. ラベルの配置
	- 同ディレクトリにラベルを配置する
	- 各データとラベルは拡張子を除き，ファイル名が一致している必要がある (例: data1.wavのラベルはdata1.txt)
	- Note: ラベルに関しては自動化できないため，各自でいい感じにすること

- データセットの中身の例

   	``` shell
   	$ ls raw/dataset1
   	data1.wav data1.txt data2.wav data2.txt data3.wav data3.txt ...
   	```

## サブデータセットの作成
- 作業一覧
	- データを training, validation, test に分けたサブデータセットを作成
	- README を用意

1. サブデータセット用のディレクトリ作成

   	``` shell
   	$ mkdir dataset/subset1
   	```

1. データの分割方法などを dataset/README に記載
	1. subset1 の説明を記載
		1. 各データの分割比を記載
			- (train : valid : test) = (7, 1, 2) など
		1. 分割方法を記載
			- ランダム
			- xx が oo になるように分割，など

1. データを分割
	- 方法1: 頑張って分割する
	- 方法2: ランダムに分割する場合は以下を実行
	1. src/dataset/data_randomize.py arg1 arg2 arg3 arg4を実行，データを分割するシェルスクリプトを作成
		- arg1: dataset1 への絶対パス
		- arg2: subset1/raw への絶対パス
		- arg3: データセット分割の比率
		- arg4 dataset1 にラベルがあり，データの分割に合わせてラベルも自動で移動する場合は1
		- 使い方は src/dataset/sample_data_randomize.sh を参照
		- パラメータを変えて，./sample_data_randomize.sh でもよい

    		``` shell
	    	$ cd src/dataset
     		$ emacs -nw sample_data_randomize.sh
    		$ ./sample_data_randomize.sh
    		$ cd ../..
    		```
		- 注意: dataset1 に .wav ではなく，特徴量 .npy を配置している場合は，arg2 を subset1 への絶対パスにする

	1. dataset/subset1/makelink.sh を実行，データを分割する

		``` shell
		$ ./dataset/subset1/makelink.sh
		```

## 特徴量抽出
- 作業一覧
	- 特徴量を抽出
	- 特徴量に対応するラベルを配置

1. 特徴量抽出
	- src/dataset/makedata.py subset1 を実行，特徴量を抽出
	- src/dataset/sample_makedata.sh を参考
	- 主に指定するオプション引数は以下の通り
		- -l fftlen (fftlen: 窓長 [sample], e.g., 1024)
		- -s fftshift (fftshift: シフト幅 [sample], e.g., 512)
		- -w window (window: 窓関数, e.g., hamming)

			``` shell
			$ cd /src/dataset
			$ emacs -nw sample_makedata.sh
    		$ ./sample_makedata.sh
			$ cd ../..
    		```
		- 注意: makedata.py が抽出するのは abs(STFT(data))
		- 変更したい場合は，61行目の wav2spc を任意の特徴量抽出を行う関数に変更する
		- また，101行目（最終行）の wav2spc.py も変更した関数名にする
		- wav2spc.py も参照
	- 確認してみる
	    ``` shell
		$ cat dataset/subset1/log
		```


2. ラベルの配置
	- 特徴量に対応するラベルを配置
	- ただし，raw/dataset1 にラベルを配置し，かつ，data_randomize.py の arg4 に 1 を与えている場合は自動で配置されている


# 学習
- 作業一覧
	- 学習する

1. src/train_network.py を用いて学習
	- sample_train_network_arch{1,2}.sh を参考に学習
	- model は models/subset1/trial* に
	- test に対する推定結果は results/subset1/trial に保存される
