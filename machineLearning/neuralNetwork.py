from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

def neural_network_tuning_cv(X_train, y_train, X_test, y_test, cv1, cv2):

    model = MLPClassifier(random_state=42, max_iter=1000)  # Ajusta los parámetros según tus necesidades

    # Parámetros para la hiperparametrización
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
    }

    # NO CV

    # Entrenamiento del modelo
    model.fit(X_train, y_train)

    # Predicciones en el conjunto de prueba
    y_pred_test_no_cv = model.predict(X_test)

    # Evaluación del rendimiento
    accuracy_test_no_cv = accuracy_score(y_test, y_pred_test_no_cv)
    confusion_mat_test_no_cv = confusion_matrix(y_test, y_pred_test_no_cv)
    classification_rep_test_no_cv = classification_report(y_test, y_pred_test_no_cv)

    print(f'Accuracy en conjunto de prueba sin CV: {accuracy_test_no_cv:.4f}')
    print('\nConfusion Matrix en conjunto de prueba sin CV:')
    print(confusion_mat_test_no_cv)
    print('\nClassification Report en conjunto de prueba sin CV:')
    print(classification_rep_test_no_cv)

    # Curva ROC
    plt.figure()
    lw = 2

    y_prob_no_cv = model.predict_proba(X_test)[:, 1]
    fpr_no_cv, tpr_no_cv, _ = roc_curve(y_test, y_prob_no_cv)
    roc_auc_no_cv = auc(fpr_no_cv, tpr_no_cv)
    plt.plot(fpr_no_cv, tpr_no_cv, color='red', lw=lw, label=f'ROC sin CV (AUC = %0.2f)' % roc_auc_no_cv)

    # PRIMER VALOR DE CV

    # Hiperparametrización con GridSearchCV
    grid_search_cv1 = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv1, scoring='accuracy', verbose=2,
                                   n_jobs=-1)
    grid_search_cv1.fit(X_train, y_train)

    # Obtención del mejor modelo (gracias a la hiperparametrización)
    best_model_cv1 = grid_search_cv1.best_estimator_

    # Predicciones en conjunto de prueba
    y_pred_test_cv1 = best_model_cv1.predict(X_test)

    # Evaluación del rendimiento
    accuracy_test_cv1 = accuracy_score(y_test, y_pred_test_cv1)
    confusion_mat_test_cv1 = confusion_matrix(y_test, y_pred_test_cv1)
    classification_rep_test_cv1 = classification_report(y_test, y_pred_test_cv1)

    print(f'Accuracy en conjunto de prueba con CV ({cv1}-fold): {accuracy_test_cv1:.4f}')
    print('\nConfusion Matrix en conjunto de prueba con CV:')
    print(confusion_mat_test_cv1)
    print('\nClassification Report en conjunto de prueba con CV:')
    print(classification_rep_test_cv1)

    # Curva ROC
    y_prob_test_cv1 = best_model_cv1.predict_proba(X_test)[:, 1]
    fpr_cv1, tpr_cv1, _ = roc_curve(y_test, y_prob_test_cv1)
    roc_auc_cv1 = auc(fpr_cv1, tpr_cv1)
    plt.plot(fpr_cv1, tpr_cv1, color='green', lw=lw, label=f'ROC con CV ({cv1}-fold) (AUC = %0.2f)' % roc_auc_cv1)

    # SEGUNDO VALOR DE CV

    # Hiperparametrización con GridSearchCV
    grid_search_cv2 = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv2, scoring='accuracy', verbose=2,
                                   n_jobs=-1)
    grid_search_cv2.fit(X_train, y_train)

    # Obtención del mejor modelo (gracias a la hiperparametrización)
    best_model_cv2 = grid_search_cv2.best_estimator_

    # Predicciones en el conjunto de prueba
    y_pred_test_cv2 = best_model_cv2.predict(X_test)

    # Evaluación del rendimiento
    accuracy_test_cv2 = accuracy_score(y_test, y_pred_test_cv2)
    confusion_mat_test_cv2 = confusion_matrix(y_test, y_pred_test_cv2)
    classification_rep_test_cv2 = classification_report(y_test, y_pred_test_cv2)

    print(f'Accuracy en conjunto de prueba con CV ({cv2}-fold): {accuracy_test_cv2:.4f}')
    print('\nConfusion Matrix en conjunto de prueba con CV:')
    print(confusion_mat_test_cv2)
    print('\nClassification Report en conjunto de prueba con CV:')
    print(classification_rep_test_cv2)

    # Curva ROC
    y_prob_test_cv2 = best_model_cv2.predict_proba(X_test)[:, 1]
    fpr_cv2, tpr_cv2, _ = roc_curve(y_test, y_prob_test_cv2)
    roc_auc_cv2 = auc(fpr_cv2, tpr_cv2)
    plt.plot(fpr_cv2, tpr_cv2, color='blue', lw=lw, label=f'ROC con CV ({cv2}-fold) (AUC = %0.2f)' % roc_auc_cv2)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de falsos positivos')
    plt.ylabel('Tasa de verdaderos positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve_neural_network.jpg')
    plt.close()

    return model, best_model_cv1, best_model_cv2
