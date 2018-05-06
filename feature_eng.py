def defined_features():
   return {'Sex': ['male', 'female'], 'PreSchool': [True, False], 'Pclass': [1, 2, 3]}

def feature_engineering(frame):
    frame['PreSchool'] = frame.apply(lambda p: p['Age'] <= 6, axis=1)
    frame['Senior'] = frame.apply(lambda p: p['Age'] >= 65, axis=1)