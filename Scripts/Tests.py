import pandas as pd
import numpy as np
import os
import math
from sklearn.ensemble import RandomForestClassifier
from scipy import stats

from sklearn.model_selection import train_test_split

import ModelNN as MNN
import Helpers as H
import Sample as S
import FormatDF as FDF


### Feature engineering with multiple samples per feature
def feature_extraction_test(features, model_num_features, target, required_features, source_location, custom_queries=[],
                            sub_samples=30, output_path='../../test.txt', model_output_path='../../testModels.txt',
                            save_seed=False, nn=False):
    folder = '../../models'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

    start_features = [x for x in features]

    if len(required_features) > 0:
        set_features = required_features
    else:
        set_features = []

    print("features: " + str(features))
    print("set_features: " + str(set_features))

    loop = model_num_features - len(set_features)
    for i in range(loop):
        print("Iteration: " + str(i))
        features = [x for x in start_features if x not in set_features]
        print("features: " + str(features))
        feature_set_eval = pd.DataFrame(
            columns=['acc', 'pos_acc', 'neg_acc', 'eval_score', 'min_pred', 'max_pred', 'epochs', 'time_per_epoch',
                     'added_feature', 'b_acc'])
        for j in range(sub_samples):

            sample_size = 1500
            feature_config = '../../feature_config'
            sample_df, seed_directory = S.get_sample(source_location=source_location, features=features, target=target,
                                                     sample_size=sample_size, custom_queries=custom_queries,
                                                     threshold=10, even=True, feature_config=feature_config,
                                                     save_seed=save_seed, seed_directory='../../seeds')

            for feat in features:
                # print("Selected Feature: " + str(feat))
                current_features = [x for x in set_features]
                current_features.append(feat)
                print("current_features: " + str(current_features))

                seed_df = pd.read_csv(seed_directory + '/sample.csv')

                test2, test4, ind_map = FDF.encode_inputs(seed_df, current_features, target,
                                                          feature_config=feature_config,
                                                          one_hot=False)
                test3 = [1 if math.log(x) >= math.log(10) else 0 for x in test4]

                current_features.append(target)

                if nn:
                    acc, wrong, fn, fp, ex_neg, ex_pos, max_pred, min_pred, mean_pred, std_pred, median_pred, epochs, time_per_epoch = MNN.logistic_binary_crossentropy(
                        test2, test3, wipe_model=True, seed_directory=seed_directory, seed_id=feat)
                else:
                    train_features, test_features, train_labels, test_labels = train_test_split(np.array(test2),
                                                                                                np.array(test3),
                                                                                                test_size=0.25)

                    rf = RandomForestClassifier(bootstrap=False, n_estimators=(500), max_depth=50)
                    rf.fit(train_features, train_labels)
                    print("Score: " + str(rf.score(test_features, test_labels)))
                    print("Feature importance: " + str(rf.feature_importances_))

                    ex_pos = 0
                    ex_neg = 0
                    correct = 0
                    wrong = 0
                    fp = 0
                    fn = 0
                    for i in range(len(test_labels)):
                        pr = rf.predict([test_features[i]])
                        if (test_labels[i] == 1):
                            ex_pos += 1
                        else:
                            ex_neg += 1
                        if (pr != test_labels[i]):
                            wrong += 1
                            if pr == 1:
                                fp += 1
                            else:
                                fn += 0
                        else:
                            correct += 1
                    print('pos: ' + str(ex_pos))
                    print('neg: ' + str(ex_neg))

                max_pred = 0
                min_pred = 0
                mean_pred = 0
                std_pred = 0
                median_pred = 0
                epochs = 0
                time_per_epoch = 0

                total = ex_pos + ex_neg

                n_acc = 1 - (wrong / total)
                n_pos_acc = (ex_pos - fp) / ex_pos
                n_neg_acc = (ex_neg - fn) / ex_neg
                # n_eval_score = (1 - (wrong / total)) + (1 - (((ex_pos - fp) / ex_pos) - ((ex_neg - fn) / ex_neg)))
                n_eval_score = n_acc - max(ex_pos / total, ex_neg / total)

                print(n_acc)

                w = [n_acc, n_pos_acc, n_neg_acc, wrong, fn, fp, ex_neg, ex_pos, max_pred, min_pred, mean_pred,
                     std_pred, median_pred, n_eval_score, epochs, time_per_epoch, j,
                     str(seed_directory).replace(',', ';'), str(current_features).replace(',', ';')]
                exists = os.path.isfile(output_path)
                f = open(output_path, "a")
                if not exists:
                    f.write(
                        "acc,pos_acc,neg_acc,wrong,fn,fp,ex_neg,ex_pos,max_pred,min_pred,mean_pred,std_pred,median_pred,eval_score,epochs,time_per_epoch,sample_number,seed_dir,features\n")
                # print(w)
                ws = H.csv_format(w)
                # print(ws)
                f.write(ws)

                b_pos_acc = ex_pos / total
                b_neg_acc = ex_neg / total
                feature_set_eval.loc[len(feature_set_eval)] = [n_acc, n_pos_acc, n_neg_acc, n_eval_score, min_pred,
                                                               max_pred, epochs, time_per_epoch, feat,
                                                               max(b_pos_acc, b_neg_acc)]
        best_feat = ''
        best_score = -math.inf
        for feat in features:
            feat_data = feature_set_eval.query('added_feature == "' + feat + '"')
            acc_mean = np.mean(feat_data['acc'])
            acc_std = np.std(feat_data['acc'])
            pos_acc_mean = np.mean(feat_data['pos_acc'])
            pos_acc_std = np.std(feat_data['pos_acc'])
            neg_acc_mean = np.mean(feat_data['neg_acc'])
            neg_acc_std = np.std((feat_data['neg_acc']))
            min_pred_mean = np.mean(feat_data['min_pred'])
            max_pred_mean = np.mean(feat_data['max_pred'])
            epoch_mean = np.mean(feat_data['epochs'])
            time_per_epoch_mean = np.mean(feat_data['time_per_epoch'])
            added_feature = feat_data['added_feature'].iloc[0]
            feature_set = set_features + [feat]
            eval_score_mean = np.mean(feat_data['eval_score'])
            eval_score_std = np.std(feat_data['eval_score'])

            p_val = stats.ttest_1samp(feat_data['acc'], np.mean(feat_data['b_acc']))
            # print(p_val)

            if eval_score_mean > best_score:
                best_feat = feat
                best_score = eval_score_mean
            ###TODO: Do not rely on order to assign these values correctly
            w = [acc_mean, acc_std, pos_acc_mean, pos_acc_std, neg_acc_mean, neg_acc_std, p_val, eval_score_mean,
                 eval_score_std, min_pred_mean, max_pred_mean, epoch_mean, time_per_epoch_mean, added_feature,
                 str(feature_set).replace(',', ';')]
            exists = os.path.isfile(model_output_path)
            f = open(model_output_path, "a")
            if not exists:
                f.write(
                    "acc_mean,acc_std,pos_acc_mean,pos_acc_std,neg_acc_mean,neg_acc_std,t_stat,p_val,eval_score_mean,eval_score_std,min_pred_mean,max_pred_mean,epoch_mean,time_per_epoch_mean,added_feature,feature_set\n")
            ws = H.csv_format(w)
            print(ws)
            f.write(ws)

        print("Best Feature: " + str(best_feat))
        set_features.append(best_feat)
