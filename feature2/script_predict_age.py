from script import script
import pandas as pd
import sys

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Usage: python script_predict_age.py <JSON_filename>")

    json_filename = sys.argv[1]

    script(json_filename)
    age_pred = pd.read_json('age_predicted.json')
    print("the predictions are stored in the file age_predicted.json, here are the first 10 predictions:")
    print(age_pred.head(10))
