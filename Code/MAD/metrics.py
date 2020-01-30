from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def evaluation(anomaly, pred_anomaly):
    confusion = confusion_matrix(anomaly, pred_anomaly)
    precision = precision_score(anomaly, pred_anomaly)
    recall = recall_score(anomaly, pred_anomaly)
    f1 = f1_score(anomaly, pred_anomaly)
    return confusion, precision, recall, f1

def print_metrics(precision,recall,f1): 
    print('Precision = ' + str(precision))  # TP / (TP+FP)
    print('Recall = ' + str(recall))        # TP / (TP+FN)
    print('Max F1 = ' + str(f1))            # 2 * (P*R) / (P+R)