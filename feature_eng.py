def feature_engineering(frame):
    frame['PreSchool'] = frame.apply(lambda p: p['Age'] <= 6, axis=1)