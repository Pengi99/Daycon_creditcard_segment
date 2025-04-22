from sklearn.metrics import f1_score, log_loss

METRICS = {
    'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    'log_loss': lambda y_true, y_pred: log_loss(y_true, y_pred),
}