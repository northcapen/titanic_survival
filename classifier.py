import functools
from feature_eng import defined_features

def build_classifier(frame):
    all = frame

    features = defined_features()

    survived = frame[frame['Survived'] == 1]
    died = frame[frame['Survived'] == 0]
    survival_ratio = survived.shape[0] / died.shape[0]


    def count_ratio(field_name, field_value):
        survived_count = survived[survived[field_name] == field_value].shape[0]
        all_count = all[all[field_name] == field_value].shape[0]
        ratio = survived_count / all_count
        return ratio

    def train_feature(feature, values):
        ratios[feature] = {}
        for v in values:
            ratios[feature][v] = count_ratio(feature, v)


    ratios = {}
    for feature in features:
        train_feature(feature, features[feature])

    print(ratios)

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
        survival_features_factor = functools.reduce(lambda acc, feature: acc * ratio(passenger, feature), features, 1)
        die_feature_factor = functools.reduce(lambda acc, feature: acc * (1 - ratio(passenger, feature)), features, 1)

        survive = survival_ratio * survival_features_factor
        die = (1 - survival_ratio) * die_feature_factor
        return 1 if survive >= die else 0

    def decision_tree(passenger):
        if passenger['Sex'] == 'male':
            return 0

        if passenger['Pclass'] == 3 and passenger['Embarked'] == 'S':
            return 0

        return 1

    return decision_tree
