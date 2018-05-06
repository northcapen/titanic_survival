def defined_features():
   return {'Pclass': [1, 2, 3], 'Sex': ['male', 'female'], 'PreSchool': [True, False]}

def feature_engineering(frame):
    frame['PreSchool'] = frame.apply(lambda p: p['Age'] <= 6, axis=1)