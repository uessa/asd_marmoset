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


# データセットの作成
1. /vad_marmoset/raw/ 配下に2つフォルダを作成し，それぞれに音声データ（wavファイル）と正解データ（txtファイル）を配置
	- フォルダ作成
	```
	$ mkdir marmoset_wav
	$ mkdir marmoset_text
	```
	- データをコピー（/datanet/projects/MarmoCall/音声共有_国立精神/あやぴょん/Annotation_text_files）
	```
	$ cp *.wav /vad_marmoset/raw/marmoset_wav
	$ cp *.txt /vad_marmoset/raw/marmoset_text
	```

1. 正解ラベルの作成
	- /raw/marmoset_text に make_label.py を配置
	- make_label.pyを実行
	- labelフォルダが作成され，その中に正解ラベルが保存

1. 正解ラベルを /raw/marmoset_wav へコピー
	- wavデータと正解ラベルのtextデータが混在する状態へ

1. /vad_marmoset/datasets/ 配下にtrain, valid, testに分けるサブセット用のフォルダ作成
	- フォルダ作成
	```
	$ mkdir subset_marmoset
	```

1. /datasets/subset_marmoset 配下に raw フォルダ作成
	- フォルダ作成
	```
	$ mkdir raw
	```

1. データをランダムに振り分ける
	- src/dataset/sample_data_randomize.sh を実行
	- data_randomize.pyが実行され，/datasets/subset_marmoset/makelink.shが生成
	- makelink.sh を実行
	- /datasets/subset_marmoset/ に train, valid, test のフォルダが生成され，それぞれに正解ラベルが保存
	- /datasets/subset_marmoset/raw/ に train, valid, test のフォルダが生成され，それぞれに音声データが格納

1. スペクトログラムを抽出
	- src/dataset/sample_makedata.sh を実行
	- /datasets/subset_marmoset/ の train, valid, test フォルダそれぞれにスペクトログラムが保存


# 学習
1. 学習
    - /src/train.py を実行（約24時間分のデータに対して学習約10時間）
    ```
    $ nohup python train.py subset_marmoset --batch_size 2 --lr 0.01 > out.log 2> err.log < /dev/null &
    ```
    - model は models/subset_marmoset/trial* に保存
	- 学習に使用するコード
		- dataset.py：学習に入力するデータセットの作成
		- train.py：学習
		- test.py：テストし正答率を出す
		- model.py：ネットワークの構造
		- path.py：パスの管理
		- logger.py：ログの書き出し
		- trainer.py, util.py：学習に必要なコード

1. テスト
	- test.py を実行
	```
    $ python test.py subset_marmoset --model ../models/subset_marmoset/trial*/model.pth --batch_size 2
    ```