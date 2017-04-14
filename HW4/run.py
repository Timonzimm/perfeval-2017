import lstm

epochs  = 1
seq_len = 50

print('> Loading data... ')

X_train, y_train, X_test, y_test = lstm.load_data('sinwave.csv', seq_len, True)

print('> Data Loaded. Compiling...')

model = lstm.build_model([1, 50, 100, 1])

model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=epochs,
    validation_split=0.05)

predicted = lstm.predict_point_by_point(model, X_test)