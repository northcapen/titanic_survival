def build_classifier(frame):
    all = frame
    survived = frame[frame['Survived'] == 1]
    died = frame[frame['Survived'] == 0]
    survival_ratio = survived.shape[0] / died.shape[0]

    def count_ratio(field_name, field_value):
        survived_count = survived[survived[field_name] == field_value].shape[0]
        all_count = all[all[field_name] == field_value].shape[0]
        ratio = survived_count / all_count
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
    # print(femaleThirdClass.shape[0])
    print("{0:.3f}".format(femaleThirdClass[femaleThirdClass['Survived'] == 0].shape[0]))

    # print(femaleThirdClass[femaleThirdClass['Survived'] == 1].shape[0])


    def ratio(passenger, field_name):
        return ratios[field_name][passenger[field_name]]

    def naive_predict_survival(df):
        if df['Pclass'] == 1:
            return 1
        return 0

    def gender_only_survival(passenger):
        result = ratios['Sex'][passenger['Sex']] > 0.5
        return 1 if result else 0

    def gender_and_class_survival(passenger):
        result = ratio(passenger, 'Pclass') * ratio(passenger, 'Sex') > 0.5 * 0.5
        return 1 if result else 0

    def gender_and_class_survival_bayes(passenger):
        survive = survival_ratio * ratio(passenger, 'Pclass') * ratio(passenger, 'Sex')
        die = (1 - survival_ratio) * (1 - ratio(passenger, 'Pclass')) * (1 - ratio(passenger, 'Sex'))
        return 1 if survive >= die else 0


    return gender_and_class_survival_bayes
