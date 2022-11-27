def calculate_accuracy(model, X, y):
    y_preds = model.predict(X)
    return sum(y_preds == y) / len(X)
