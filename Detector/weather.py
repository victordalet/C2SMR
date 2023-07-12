import requests


class DetectorWeather:
    def __init__(self, latitude: float, longitude: float):
        self.latitude: float = latitude
        self.longitude: float = longitude
        self.url_api: str = "https://api.open-meteo.com/v1/meteofrance?latitude=" + str(
            self.latitude) + "&longitude=" + str(
            self.longitude) + "&hourly=temperature_2m,precipitation,cloudcover,cloudcover_low,cloudcover_mid,cloudcover_high,windspeed_10m,windspeed_20m,windspeed_50m,windspeed_100m,windspeed_150m,windspeed_200m,is_day"
        self.data = requests.get(self.url_api)
        self.data = self.data.json()
        print(self.data)
        print(self.url_api)

    def is_day(self) -> bool:
        # ---- replace 0 by the good time ---- #
        return self.data["hourly"]["is_day"][0]


def main():
    d = DetectorWeather(52.5243, 13.41)
    print(d.is_day())


main()
