from anomaly_detection import *
from size_classification import load_data
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError('You must enter the parameter nu , it must be an integer in [0, 1]')

    nu = float(sys.argv[1])
    if nu < 0 or nu > 1:
        raise ValueError('You must enter the parameter nu , it must be an integer in [0, 1]')

    df, df_learning = load_data()

    anom_IF, anom_OCSVM, anom_both = detect_anomalies(df_learning, nu=nu)
    print('plotting ...')
    plot_anomalies(df, anom_IF, anom_OCSVM, anom_both)
    plt.show()
