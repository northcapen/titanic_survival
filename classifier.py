def build_classifier(frame):
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
    # print(ratios)
    femaleThirdClass = all[(all['Sex'] == 'female') & (all['Pclass'] == 3)]
    # print(femaleThirdClass.shape[0])
    print("{0:.3f}".format(femaleThirdClass[femaleThirdClass['Survived'] == 0].shape[0]))

    # print(femaleThirdClass[femaleThirdClass['Survived'] == 1].shape[0])

    def naive_predict_survival(df):
        if df['Pclass'] == 1:
            return 1
        return 0

    def predict_survival(passenger):
        # result = ratios['Pclass'][passenger['Pclass']] * ratios['Sex'][passenger['Sex']] > 0.5 * 0.5
        result = ratios['Sex'][passenger['Sex']] > 0.5
        # print(passenger['Pclass'], passenger['Sex'], " is ", 1 if result else 0, "In fact is ", passenger['Survived'])
        return 1 if result else 0

    return predict_survival
