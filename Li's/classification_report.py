from sklearn.metrics import classification_report, confusion_matrix

y_pred = [0,1,0,1,0,0,1] #机器学习后y的预测值
y_true = [0,0,0,1,1,0,1]
print(classification_report(y_true,y_pred))
#混淆矩阵
print(confusion_matrix(y_true,y_pred))
