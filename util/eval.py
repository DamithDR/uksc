from sklearn.metrics import recall_score, precision_score, f1_score


def eval_decisions(df, pred_column, real_column):
    predictions = df[pred_column].tolist()
    real_values = df[real_column].tolist()

    w_recall = recall_score(real_values, predictions, average='weighted')
    w_precision = precision_score(real_values, predictions, average='weighted')
    w_f1 = f1_score(real_values, predictions, average='weighted')
    m_f1 = f1_score(real_values, predictions, average='macro')

    print("\nWeighted Recall {}".format(w_recall))
    print("\nWeighted Precision {}".format(w_precision))
    print("\nWeighted F1 Score {}".format(w_f1))
    print("\nMacro F1 Score {}".format(m_f1))

    return w_recall, w_precision, w_f1, m_f1
