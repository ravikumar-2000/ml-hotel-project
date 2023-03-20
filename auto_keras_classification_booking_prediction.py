import os
import pandas as pd
from consts import *
import autokeras as ak
import tensorflow as tf
from tensorflow.keras.models import load_model


def train_model():
    df = pd.read_csv(modified_hotel_reservations_csv_file_path)
    df.columns = df.columns.str.strip()
    train_size = int(df.shape[0] * 0.90)
    df[:train_size].to_csv(hotel_reservations_train_file_path, index=False)
    df[train_size:].to_csv(hotel_reservations_test_file_path, index=False)

    column_types = {
        "no_of_adults": "numerical",
        "no_of_children": "numerical",
        "no_of_weekend_nights": "numerical",
        "no_of_week_nights": "numerical",
        "type_of_meal_plan": "categorical",
        "required_car_parking_space": "numerical",
        "room_type_reserved": "categorical",
        "lead_time": "numerical",
        "arrival_year": "numerical",
        "arrival_month": "numerical",
        "arrival_date": "numerical",
        "market_segment_type": "categorical",
        "repeated_guest": "numerical",
        "no_of_previous_cancellations": "numerical",
        "no_of_previous_bookings_not_canceled": "numerical",
        "avg_price_per_room": "numerical",
        "no_of_special_requests": "numerical",
    }

    clf = ak.StructuredDataClassifier(
        column_types=column_types, overwrite=True, max_trials=5
    )

    clf.fit(
        hotel_reservations_train_file_path,
        "booking_status",
        epochs=50,
    )

    predicted_y = clf.predict(hotel_reservations_test_file_path)
    print(predicted_y)
    print(clf.evaluate(hotel_reservations_test_file_path, "booking_status"))

    model = clf.export_model()

    try:
        model.save(classification_model_path_name, save_format="tf")
    except Exception:
        model.save(classification_model_path_name + ".h5")
    os.system("rm -rf ./structured_data_classifier")


def predict_model():
    loaded_model = load_model(
        classification_model_path_name, custom_objects=ak.CUSTOM_OBJECTS
    )
    x_test = pd.read_csv(hotel_reservations_predict_file_path)

    predicted_y = loaded_model.predict(x_test)
    print(predicted_y)
    return predicted_y[0][0]


if __name__ == "__main__":
    # train_model()
    # predict_model()
    print("done!")
