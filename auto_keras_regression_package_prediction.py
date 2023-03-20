import os
import pandas as pd
from consts import *
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model


def train_model():

    df = pd.read_csv(modified_hotel_records_csv_file_path)
    # df = df.drop(["package_guide", "package_acc", "travel_with"], axis=1)

    train_size = int(df.shape[0] * 0.95)
    df[:train_size].to_csv(hotel_record_train_file_path, index=False)
    df[train_size:].to_csv(hotel_record_test_file_path, index=False)

    column_types = {
        "city": "categorical",
        "age_group": "categorical",
        "travel_with": "categorical",
        "total_female": "numerical",
        "total_male": "numerical",
        "purpose": "categorical",
        "tour_arrangement": "categorical",
        "package_transport": "categorical",
        "package_acc": "categorical",
        "package_guide": "categorical",
        "package_sight_see": "categorical",
        "night": "numerical",
    }

    reg = ak.StructuredDataRegressor(
        overwrite=True,
        column_types=column_types,
        loss="mean_absolute_error",
        max_trials=5,
    )
    reg.fit(
        hotel_record_train_file_path,
        "total_cost",
        epochs=500,
    )

    print("+" * 50)
    print("showing test results ...")
    print(reg.evaluate(hotel_record_test_file_path, "total_cost"))
    print("+" * 50)

    model = reg.export_model()

    try:
        model.save(regression_model_path_name, save_format="tf")
    except Exception:
        model.save(regression_model_path_name + ".h5")

    os.system("rm -rf ./structured_data_regressor")


def predict_model():
    test_data = pd.read_csv(hotel_record_predict_file_path)
    loaded_model = load_model(
        regression_model_path_name, custom_objects=ak.CUSTOM_OBJECTS
    )
    predicted_y = loaded_model.predict(test_data)
    print(predicted_y)
    return predicted_y[0][0]


if __name__ == "__main__":
    train_model()
    # predict_model()
    print("done!")
