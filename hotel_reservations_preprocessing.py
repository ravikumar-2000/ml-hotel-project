import pandas as pd
from consts import *
from sklearn.preprocessing import LabelEncoder


def main():
    df = pd.read_csv(hotel_reservations_file_location)
    df.drop(["Booking_ID"], axis=1, inplace=True)

    le_meal_plan = LabelEncoder()
    le_room_type = LabelEncoder()
    le_market_seg_type = LabelEncoder()
    le_booking_status = LabelEncoder()

    df.type_of_meal_plan = le_meal_plan.fit_transform(df.type_of_meal_plan)
    df.room_type_reserved = le_room_type.fit_transform(df.room_type_reserved)
    df.market_segment_type = le_market_seg_type.fit_transform(df.market_segment_type)
    df.booking_status = le_booking_status.fit_transform(df.booking_status)

    print(
        le_meal_plan.classes_,
        le_room_type.classes_,
        le_market_seg_type.classes_,
        le_booking_status.classes_,
    )

    df = df.sample(frac=1.0)
    df.to_csv(modified_hotel_reservations_csv_file_path, index=False)


if __name__ == "__main__":
    main()
    print("done!")
