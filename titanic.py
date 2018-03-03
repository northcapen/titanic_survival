import pandas as pd
import matplotlib.pyplot as plt

frame = pd.read_csv('train.csv', delimiter=',')

# print(frame['PassengerId'])

# print(frame[0:1])
all = frame
survived = frame[frame['Survived'] == 1]
died = frame[frame['Survived'] == 0]


def count_ratio(field_name, field_value):
    survived_count = survived[survived[field_name] == field_value].shape[0]
    all_count = all[all[field_name] == field_value].shape[0]
    ratio = survived_count / all_count

    # print('Survival ratio for ' + field_name + " == " + str(field_value) + " is " + str(ratio))

    return ratio


ratios = {}
ratios['Pclass'] = {}
ratios['Pclass'][1] = count_ratio('Pclass', 1)
ratios['Pclass'][2] = count_ratio('Pclass', 2)
ratios['Pclass'][3] = count_ratio('Pclass', 3)

ratios['Sex'] = {}
ratios['Sex']['male'] = count_ratio('Sex', 'male')
ratios['Sex']['female'] = count_ratio('Sex', 'female')

print(ratios)

femaleThirdClass = all[(all['Sex'] == 'female') & (all['Pclass'] == 3)]
#print(femaleThirdClass.shape[0])
print(femaleThirdClass[femaleThirdClass['Survived'] == 0].shape[0])
print(femaleThirdClass[femaleThirdClass['Survived'] == 1].shape[0])


def naive_predict_survival(df):
    if df['Pclass'] == 1:
        return 1
    return 0


def predict_survival(passenger):
    #result = ratios['Pclass'][passenger['Pclass']] * ratios['Sex'][passenger['Sex']] > 0.5 * 0.5
    result = ratios['Sex'][passenger['Sex']] > 0.5
    # print(passenger['Pclass'], passenger['Sex'], " is ", 1 if result else 0, "In fact is ", passenger['Survived'])
    return 1 if result else 0


print(predict_survival({'Sex' : 'male', 'Pclass': 1}))
print(predict_survival({'Sex' : 'male', 'Pclass': 2}))
print(predict_survival({'Sex' : 'male', 'Pclass': 3}))
print(predict_survival({'Sex' : 'female', 'Pclass': 1}))
print(predict_survival({'Sex' : 'female', 'Pclass': 2}))
print(predict_survival({'Sex' : 'female', 'Pclass': 3}))



frame['Predicted'] = frame.apply(predict_survival, axis=1)
print(frame[frame['Predicted'] == frame['Survived']].shape[0] / frame.shape[0])

# x = frame[['Pclass', 'Survived']]
# print(type(x[0:1].ix[0]))

# for i in range(1, 10):
#    el = x.ix[i].to_dict()
#    print(el, ' prediction ', predict_survival(el))


# plt.plot(survived['Pclass'], survived['Fare'], 'bo')
# plt.plot(died['Pclass'], died['Fare'], 'r+')
# plt.show()



# print(x[0:1], predict_survival(x[0:1]))
