import pandas as pd
from consts import *
from sklearn.preprocessing import LabelEncoder


def main():

    df = pd.read_excel(hotel_records_file_location)

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.strip()
    print(df.columns)

    df.drop(["id"], axis=1, inplace=True)

    df.rename(
        {
            "cities": "city",
            "agegroup": "age_group",
            "travelwith": "travel_with",
            "totalfemale": "total_female",
            "totalmale": "total_male",
            "tourarrangement": "tour_arrangement",
            "packagetransport": "package_transport",
            "packageacc": "package_acc",
            "packageguide": "package_guide",
            "packagesightsee": "package_sight_see",
            "total cost": "total_cost",
        },
        axis=1,
        inplace=True,
    )

    df.age_group = df.age_group.str.strip()
    df.travel_with = df.travel_with.str.strip()
    df.purpose = df.purpose.str.strip()
    df.tour_arrangement = df.tour_arrangement.str.strip()

    df.total_cost = df.total_cost.astype("float64")

    print(df.columns)

    print(df.city.unique())
    print("=" * 20)
    print(df.city.value_counts())

    print(df.age_group.unique())
    print("=" * 20)
    print(df.age_group.value_counts())

    print(df.travel_with.unique())
    print("=" * 20)
    print(df.travel_with.value_counts())

    print(df.purpose.unique())
    print("=" * 20)
    print(df.purpose.value_counts())

    print(df.tour_arrangement.unique())
    print("=" * 20)
    print(df.tour_arrangement.value_counts())

    df.dropna(inplace=True)

    df.age_group = df.age_group.str.replace("BELOW 24", "0-24")

    le_city = LabelEncoder()
    le_age_group = LabelEncoder()
    le_travel_with = LabelEncoder()
    le_prupose = LabelEncoder()
    le_tour_arrangement = LabelEncoder()
    le_package = LabelEncoder()

    df.city = le_city.fit_transform(df.city)
    df.age_group = le_age_group.fit_transform(df.age_group)
    df.travel_with = le_travel_with.fit_transform(df.travel_with)
    df.purpose = le_prupose.fit_transform(df.purpose)
    df.tour_arrangement = le_tour_arrangement.fit_transform(df.tour_arrangement)
    df.package_transport = le_package.fit_transform(df.package_transport)
    df.package_acc = le_package.fit_transform(df.package_acc)
    df.package_guide = le_package.fit_transform(df.package_guide)
    df.package_sight_see = le_package.fit_transform(df.package_sight_see)

    print(
        le_city.classes_,
        le_age_group.classes_,
        le_travel_with.classes_,
        le_prupose.classes_,
        le_tour_arrangement.classes_,
    )

    df = df.sample(frac=1.0)
    df.to_csv(modified_hotel_records_csv_file_path, index=False)
    mod_df = pd.read_csv(modified_hotel_records_csv_file_path)
    print(mod_df.head(10))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
