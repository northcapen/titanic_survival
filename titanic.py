import pandas as pd

from classifier import build_classifier

frame = pd.read_csv('train.csv', delimiter=',')
predict_survival = build_classifier(frame)

print(predict_survival({'Sex' : 'male', 'Pclass': 1}))
print(predict_survival({'Sex' : 'male', 'Pclass': 2}))
print(predict_survival({'Sex' : 'male', 'Pclass': 3}))
print(predict_survival({'Sex' : 'female', 'Pclass': 1}))
print(predict_survival({'Sex' : 'female', 'Pclass': 2}))
print(predict_survival({'Sex' : 'female', 'Pclass': 3}))

frame['Predicted'] = frame.apply(predict_survival, axis=1)
print("Accuracy: %.3f" %(frame[frame['Predicted'] == frame['Survived']].shape[0] / frame.shape[0]))
