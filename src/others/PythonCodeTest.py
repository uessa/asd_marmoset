import numpy as np
import pathlib
import itertools
import os
from sklearn.metrics import confusion_matrix

# フレーム配列に対し，非発声ラベル0を平滑化する関数
def smooth_labels(labels, thr):
    smoothed_labels = np.copy(labels) # 出力配列
    
    count = 0 # ラベル0の連続数のカウントを初期化
    buf = 0 # ラベル0の平滑化先を初期化
    flag = "Off" # 平滑化のフラグを初期化
    
    # 先頭から順に捜査
    for i in range(len(smoothed_labels)):
        # いまのidxがラベル0である
        if smoothed_labels[i] == 0:
            # ラベル0列の左端にきたときのみ初期化
            if count == 0:
                buf = smoothed_labels[i-1] # ラベル0列の直前に出現したラベル
                count += 1 # ラベル0をカウント
                flag = "On" # 平滑化フラグオン
            # ラベル0のカウントが閾値を超えたら（※ここは＞でなく≧を用いる）
            elif count >= thr:
                flag = "Off" # 平滑化フラグオフ
            # それ以外の条件
            else:
                count += 1 # ラベル0をカウント
        # いまのidxがラベル0以外
        else:
            # 平滑化フラグオフの場合
            if flag == "Off":
                count = 0 # 初期化
                buf = 0 # 初期化
            # 平滑化フラグオンの場合
            else:
                smoothed_labels[i-thr:i] = buf # 直前までのラベル0の列をラベルbufに平滑化
                flag = "Off" # 平滑化フラグオフ
                count = 0 # 初期化
                buf = 0 # 初期化
        
    return smoothed_labels



# 正解ラベルと推定ラベルに対し，混同行列を返す関数
def calculate_accuracy(label, predict, classes):
        
    cm = confusion_matrix(label, predict, labels=classes) # 混同行列
    recall = np.diag(cm) / np.sum(cm, axis=1) # 再現率
    total_accuracy = np.sum(np.diag(cm)) / np.sum(cm) # 正解率
    precision = np.diag(cm) / np.sum(cm, axis=0) # 適合率
    f_score = (2 * recall * precision) / (recall + precision) # F1-score
    
    # クラスごとに表示
    # for i in range(len(classes)):  
    #     print("class: {},  recall: {:.2f},  precision: {:.2f},  f1-socre: {}".format(
    #         classes[i], 
    #         round(recall[i], 2), round(precision[i], 2), round(f_score[i], 2)
    #         ))
    # print("label:  ",label)
    # print("predict:",predict)
    print("accuracy:",round(total_accuracy, 2))
    # print("confusion: ")
    # print(cm)

    return cm
    
    
            
if __name__ == "__main__":
    
    # labels = np.loadtxt("test.txt", dtype=int)
    # thr = 2
    # print(labels)

    # smoothed_labels = smooth_labels(labels, thr)
    # print(smoothed_labels)
    
    # label = np.array([0,1,2,3,4,4])
    # predict = np.array([0,1,2,3,4,0])
    # cm = calculate_accuracy(label, predict, [0, 1, 2, 3, 4])
    
    call_label = {1: "Phee",2: "Trill", 3: "Twitter", 4: "Other Calls"}
    call_init = {'Phee':0, 'Trill':0, 'Twitter':0, 'Other Calls':0}
    
    # 正解データ，推定データのディレクトリ，結果出力のディレクトリ
    labelpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/test_5class_before")
    predpath = pathlib.Path("/home/muesaka/projects/marmoset/datasets/subset_marmoset_11vpa_muesaka/test/results_5class_before")
    outputpath = pathlib.Path("./LabelRatio/")

    # 正解データ，推定データのファイル名をリスト化
    labels = list(labelpath.glob("*.txt"))
    preds = list(predpath.glob("*.txt"))

    labels = sorted(labels)
    preds = sorted(preds)

    label_classes = 5
    pred_classes = 5
    cm = np.zeros((label_classes, label_classes))
    for i in range(len(labels)):
        
        print("filename:",labels[i])
        # nparray型のフレームデータ
        label = np.loadtxt(labels[i], dtype=int) # 正解
        pred = np.loadtxt(preds[i], dtype=int) # 推定
        
        # 平滑化の閾値ごとに確認
        # for j in range(10,30,1):
        for j in [0, 10, 21]:
            print("thr:",j)
            
            dict_label = call_init.copy()
            dict_pred = call_init.copy()

            pred_thr = smooth_labels(pred, j) # 平滑化
            label_thr = label
            
            Label = [k for k,g in itertools.groupby(label_thr)] # グループ化
            Pred = [k for k,g in itertools.groupby(pred_thr)] # グループ化
            
            for label_ in Label:
                # labelが有声，1,2,3,4であるとき
                if label_ == 1 or label_ == 2 or label_ == 3 or label_ == 4:
                    tmp1 = call_label[label_]
                    dict_label[tmp1] = dict_label.get(tmp1, 0) + 1
            for pred_ in Pred:
                # predが有声，1,2,3,4であるとき
                if pred_ == 1 or pred_ == 2 or pred_ == 3 or pred_ == 4:
                    tmp2 = call_label[pred_]
                    dict_pred[tmp2] = dict_pred.get(tmp2, 0) + 1
                    
            print("label",dict_label)
            print("result",dict_pred)        
            # cm = calculate_accuracy(label=label_thr, predict=pred_thr, 
            #                         classes=[i for i in range(label_classes)])
            print("")
            
            
        # 混同行列
        # cm = calculate_accuracy(label=label, predict=pred,
        #                         classes=[i for i in range(label_classes)])
    os.system("end_report 上坂奏人 PythonCodeTest")
        # break
