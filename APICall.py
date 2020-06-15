import requests
import json
import time
import math

myapiKeyDS = "51378d9e32a368173805b8dbd0137926"
apiKeyDS = "5f0f502eebe7f48dd8529c07b652c3f5"
apiKeyMO = "a3b0d74b-ff40-4238-af3d-9a6bd2bc0ba0"
latitudes, longitudes = [54.7678+4*0.0141,54.7678+3*0.0141,54.7678+2*0.0141,54.7678+0.0141,54.7678,54.7678-0.0141,54.7678-2*0.0141,54.7678-3*0.0141,54.7678-4*0.0141],[-1.573643+4*0.0141,-1.573643+3*0.0141,-1.573643+2*0.0141,-1.573643+0.0141,-1.573643,-1.573643-0.0141,-1.573643-2*0.0141,-1.573643-3*0.0141,-1.573643-4*0.0141]
MOID = 350525

def apiRequest(URL):
    request = requests.get(URL)
    data = request.json()
    return data

def dataDump(data, target_file):
    with open(target_file, "w") as data_file:
        json.dump(data, data_file)

def apiForecastRequestDS(latitude,longitude):
    """Provides forecast data when initialised from Darksky api, dumps into txt from json with unix timestamp"""
    url_forecast = f"https://api.darksky.net/forecast/{apiKeyDS}/{latitude},{longitude}?units=si"
    #unix_time = f"{time.time():.0f}"
    #target_file = f"DSForecastData{unix_time}.json"
    data = apiRequest(url_forecast)
    return data

def apiForecastGridDS(latitudes, longitudes):
    unix_time = f"{time.time():.0f}"
    target_file = f"DSForecastData{unix_time}.json"
    data = []
    for lat in latitudes:
        for long in longitudes:
            data.append(apiForecastRequestDS(lat, long))
    dataDump(data, target_file)

def apiForecastRequestMO():
    url_forecast = f"http://datapoint.metoffice.gov.uk/public/data/val/wxfcs/all/json/350525?res=3hourly&key={apiKeyMO}"
    unix_time = f"{time.time():0f}"
    target_file = f"MOForecastData{unix_time}.json"
    data = apiRequest(url_forecast)
    dataDump(data, target_file)

def apiPastRequests(start_time, iterations, latitude, longitude):
    """Provides hourly weather data from Darksky api, dumps into json with unix timestamp"""
    for i in range(iterations):
        time = start_time + i * 86400
        target_file = f"PastData{time}.txt"
        url_past = f"https://api.darksky.net/forecast/{apiKeyDS}/{latitude},{longitude},{time}?units=si"
        data = apiRequest(url_past)
        dataDump(data, target_file)
    return time


if __name__ =="__main__":
    apiForecastGridDS(latitudes, longitudes)
    apiForecastRequestMO()
