# TODO: Move this into separate notebook / script for initial data evaluation / examination
def plot_history(history):
    import matplotlib.pyplot as plt
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('model mae')
    plt.ylabel('mae')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()