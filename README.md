<!-- -*- coding: utf-8 -*- -->

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
    - /src/train.sh を実行（約24時間分のデータに対して学習約10時間）
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
	- test_for_one_data.sh を実行