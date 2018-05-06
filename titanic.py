import pandas as pd

from classifier import build_classifier

def feature_engineering(frame):
    frame['PreSchool'] = frame.apply(lambda p: p['Age'] <= 6, axis=1)

frame = pd.read_csv('train.csv', delimiter=',')


feature_engineering(frame)
predict_survival = build_classifier(frame)

def calculate_ratio_dataset(frame):
    survived = frame[frame['Survived'] == 1]
    return survived.shape[0] / frame.shape[0]

all = frame
femaleThirdClass = all[(all['Sex'] == 'female') & (all['Pclass'] == 3)]
femaleSecondClass = all[(all['Sex'] == 'female') & (all['Pclass'] == 2)]
femaleFirstClass = all[(all['Sex'] == 'female') & (all['Pclass'] == 1)]
maleFirstClass = all[(all['Sex'] == 'male') & (all['Pclass'] == 1)]
maleSecondClass = all[(all['Sex'] == 'male') & (all['Pclass'] == 2)]
maleThirdClass = all[(all['Sex'] == 'male') & (all['Pclass'] == 3)]
preschool = all[all['PreSchool'] == True]
print('PreSchool', preschool.shape[0], calculate_ratio_dataset(preschool))
print('F3', femaleThirdClass.shape[0], calculate_ratio_dataset(femaleThirdClass))
print('F2', femaleSecondClass.shape[0], calculate_ratio_dataset(femaleSecondClass))
print('F1', femaleFirstClass.shape[0], calculate_ratio_dataset(femaleFirstClass))
print('M1', maleFirstClass.shape[0], calculate_ratio_dataset(maleFirstClass))
print('M2', maleSecondClass.shape[0], calculate_ratio_dataset(maleSecondClass))
print('M3', maleThirdClass.shape[0], calculate_ratio_dataset(maleThirdClass))


#print(predict_survival({'Sex' : 'male', 'Pclass': 1}))
#print(predict_survival({'Sex' : 'male', 'Pclass': 2}))
#print(predict_survival({'Sex' : 'male', 'Pclass': 3}))
#print(predict_survival({'Sex' : 'female', 'Pclass': 1}))
#print(predict_survival({'Sex' : 'female', 'Pclass': 2}))
#print(predict_survival({'Sex' : 'female', 'Pclass': 3}))

frame['Predicted'] = frame.apply(predict_survival, axis=1)
print("Accuracy: %.3f" %(frame[frame['Predicted'] == frame['Survived']].shape[0] / frame.shape[0]))

test_set = pd.read_csv('test.csv', delimiter=',')
feature_engineering(test_set)

test_set['Survived'] = test_set.apply(predict_survival, axis=1)
test_set.to_csv(path_or_buf='test_submission.csv', columns=['PassengerId', 'Survived'], index=False)