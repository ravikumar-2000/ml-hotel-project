import os
import socket
import uvicorn
import pandas as pd
from consts import *
from fastapi import FastAPI
from pydantic import BaseModel
from auto_keras_regression_package_prediction import predict_model as reg_predict_model
from auto_keras_classification_booking_prediction import (
    predict_model as clf_predict_model,
)
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TourRequestBody(BaseModel):
    city: int
    age_group: int
    travel_with: int
    total_female: int
    total_male: int
    purpose: int
    tour_arrangement: int
    package_transport: int
    package_acc: int
    package_guide: int
    package_sight_see: int
    night: int


class BookingRequestBody(BaseModel):
    no_of_adults: int
    no_of_children: int
    no_of_weekend_nights: int
    no_of_week_nights: int
    type_of_meal_plan: int
    required_car_parking_space: int
    room_type_reserved: int
    lead_time: int
    arrival_year: int
    arrival_month: int
    arrival_date: int
    market_segment_type: int
    repeated_guest: int
    no_of_previous_cancellations: int
    no_of_previous_bookings_not_canceled: int
    avg_price_per_room: int
    no_of_special_requests: int


@app.post("/get-tour-package")
def get_tour_package(request_body: TourRequestBody):
    test_data = request_body.dict()
    print(test_data)
    test_csv_df = pd.DataFrame(test_data, index=[0])
    print(test_csv_df.head())
    test_csv_df.to_csv(hotel_record_predict_file_path, index=False)
    package_price = reg_predict_model()
    return JSONResponse({"price": str(package_price)})


@app.post("/get-booking_prediction")
def get_tour_package(request_body: BookingRequestBody):
    test_data = request_body.dict()
    print(test_data)
    test_csv_df = pd.DataFrame(test_data, index=[0])
    print(test_csv_df.head())
    test_csv_df.to_csv(hotel_reservations_predict_file_path, index=False)
    package_status = clf_predict_model()
    package_status = 0 if package_status < 0.65 else 1
    return JSONResponse({"status": package_status})


if __name__ == "__main__":
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
    except Exception as e:
        print("*" * 20)
        print(e)
        print("*" * 20)
    uvicorn.run(app, host=ip_address, port=8001, log_level="info")
