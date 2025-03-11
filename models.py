import math
import logging
import statistics
import requests
import threading
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
from bisect import bisect_left
from dataclasses import dataclass
from functools import total_ordering

logger = logging.getLogger(__name__)

@total_ordering
@dataclass
class BalloonPosition:
    latitude: float
    longitude: float
    altitude: float
    hour: int
    continent: str
    
    def __lt__(self, other):
        if isinstance(other, BalloonPosition):
            return self.latitude < other.latitude
        return NotImplemented

    def __eq__(self, other):
        if not isinstance(other, BalloonPosition):
            return NotImplemented
        return self.latitude == other.latitude

    def distance_to(self, lat: float, lon: float) -> float:
        """Calculate Haversine distance to a point"""
        R = 6371  # Earth's radius in km
        dlat = math.radians(self.latitude - lat)
        dlon = math.radians(self.longitude - lon)
        a = (math.sin(dlat/2) * math.sin(dlat/2) +
             math.cos(math.radians(lat)) * math.cos(math.radians(self.latitude)) *
             math.sin(dlon/2) * math.sin(dlon/2))
        return 2 * R * math.asin(math.sqrt(a))

class FlightData:
    def __init__(self):
        self.sorted_data: Dict[int, Dict[str, List[BalloonPosition]]] = {}
        self.lat_indices: Dict[int, Dict[str, List[float]]] = {}
        self.continent_boundaries = {   
            "Africa": {"lat_range": (-37, 37), "lon_range": (-20, 50)},
            "Asia": {"lat_range": (10, 80), "lon_range": (60, 180)},
            "Europe": {"lat_range": (35, 70), "lon_range": (-25, 40)},
            "North America": {"lat_range": (10, 85), "lon_range": (-170, -60)},
            "South America": {"lat_range": (-60, 15), "lon_range": (-80, -35)},
            "Oceania": {"lat_range": (-55, 5), "lon_range": (110, 180)},
            "Antarctica": {"lat_range": (-90, -60), "lon_range": (-180, 180)}
        }

    def get_continent(self, latitude, longitude):
        for continent, boundaries in self.continent_boundaries.items():
            lat_range = boundaries["lat_range"]
            lon_range = boundaries["lon_range"]
            if lat_range[0] <= latitude <= lat_range[1] and lon_range[0] <= longitude <= lon_range[1]:
                return continent
        return "Unknown"

    def add_balloon_position(self, hour, latitude, longitude, altitude):
        if self.is_valid_balloon_data(latitude, longitude, altitude):
            continent = self.get_continent(latitude, longitude)
            balloon = BalloonPosition(latitude, longitude, altitude, hour, continent)
            
            if hour not in self.sorted_data:
                self.sorted_data[hour] = {}
                self.lat_indices[hour] = {}
            if continent not in self.sorted_data[hour]:
                self.sorted_data[hour][continent] = []
                self.lat_indices[hour][continent] = []
            
            self.sorted_data[hour][continent].append(balloon)
            self.lat_indices[hour][continent].append(balloon.latitude)
            if len(self.sorted_data[hour][continent]) % 1000 == 0:
                self._sort_continent_data(hour, continent)

    def _sort_continent_data(self, hour: int, continent: str):
        balloons = self.sorted_data[hour][continent]
        balloons.sort(key=lambda b: b.latitude)
        self.lat_indices[hour][continent] = [b.latitude for b in balloons]

    def is_valid_balloon_data(self, latitude, longitude, altitude):
        return not (self.is_nan(latitude) or self.is_nan(longitude) or self.is_nan(altitude)) and \
               self.is_valid_latitude(latitude) and self.is_valid_longitude(longitude) and \
               self.is_valid_altitude(altitude)
    
    def is_nan(self, value):
        return isinstance(value, float) and math.isnan(value)
    
    def is_valid_latitude(self, latitude):
        return -90 <= latitude <= 90
    
    def is_valid_longitude(self, longitude):
        return -180 <= longitude <= 180
    
    def is_valid_altitude(self, altitude):
        return altitude >= 0

    def get_most_recent_hour(self) -> Optional[int]:
        if not self.sorted_data:
            return None
        return max(self.sorted_data.keys())

    def get_24h_trends(self, target_lat: float, target_lon: float) -> dict:
        """Get 30-day ensemble forecast with extended agricultural parameters."""
        weather_service = OpenMeteo()
        return weather_service.fetch_ensemble_forecast(target_lat, target_lon)

class OpenMeteo:
    def __init__(self):
        self.ensemble_api = "https://ensemble-api.open-meteo.com/v1/ensemble"
        self.climate_api = "https://climate-api.open-meteo.com/v1/climate"
        self.air_api = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.model = "MRI_AGCM3_2_S"

    def fetch_ensemble_forecast(self, lat: float, lon: float) -> dict:
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "precipitation",
                "et0_fao_evapotranspiration",
                "temperature_120m",
                "soil_moisture_40_to_100cm",
                "direct_radiation",
                "diffuse_radiation"
            ],
            "forecast_days": 30,
            "models": "gfs_seamless"
        }
        
        try:
            response = requests.get(self.ensemble_api, params=params)
            response.raise_for_status()
            data = response.json()
            
            processed_data = {
                'surface_conditions': {
                    'temperature': data['hourly']['temperature_2m'],
                    'humidity': data['hourly']['relative_humidity_2m'],
                    'dew_point': data['hourly']['dew_point_2m']
                },
                'precipitation': {
                    'values': data['hourly']['precipitation'],
                    'total_30day': sum(data['hourly']['precipitation'])
                },
                'soil_conditions': {
                    'deep_moisture': data['hourly']['soil_moisture_40_to_100cm']
                },
                'radiation': {
                    'direct': data['hourly']['direct_radiation'],
                    'diffuse': data['hourly']['diffuse_radiation'],
                    'total': [d + diff for d, diff in zip(
                        data['hourly']['direct_radiation'], 
                        data['hourly']['diffuse_radiation']
                    )]
                },
                'agricultural_metrics': {
                    'evapotranspiration': data['hourly']['et0_fao_evapotranspiration'],
                    'upper_temp': data['hourly']['temperature_120m']
                },
                'time': data['hourly']['time']
            }
            
            return self._calculate_daily_aggregates(processed_data)
            
        except Exception as e:
            logger.error(f"Error fetching ensemble forecast: {str(e)}")
            return None

    def fetch_climate_data(self, lat: float, lon: float) -> dict:
        """Fetch historical climate data"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "models": self.model,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "windspeed_10m_max"
            ],
            "start_date": "1991-01-01",
            "end_date": "2020-12-31"
        }
        
        try:
            response = requests.get(self.climate_api, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching climate data: {str(e)}")
            return None

    def fetch_air_quality(self, lat: float, lon: float) -> dict:
        """Fetch air quality data"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "pm10",
                "pm2_5",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "ozone",
                "european_aqi"
            ]
        }
        
        try:
            response = requests.get(self.air_api, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching air quality data: {str(e)}")
            return None

    def _calculate_daily_aggregates(self, processed_data):
        days = len(processed_data['time']) // 24
        daily_data = {
            'dates': [],
            'avg_temp': [],
            'min_temp': [],
            'max_temp': [],
            'total_precip': [],
            'avg_soil_moisture': [],
            'total_radiation': [],
            'total_evapotranspiration': []
        }
        
        for day in range(days):
            start_idx = day * 24
            end_idx = start_idx + 24
            date = processed_data['time'][start_idx].split('T')[0]
            
            daily_data['dates'].append(date)
            temps = processed_data['surface_conditions']['temperature'][start_idx:end_idx]
            daily_data['avg_temp'].append(sum(temps) / 24)
            daily_data['min_temp'].append(min(temps))
            daily_data['max_temp'].append(max(temps))
            daily_data['total_precip'].append(
                sum(processed_data['precipitation']['values'][start_idx:end_idx])
            )
            daily_data['avg_soil_moisture'].append(
                sum(processed_data['soil_conditions']['deep_moisture'][start_idx:end_idx]) / 24
            )
            daily_data['total_radiation'].append(
                sum(processed_data['radiation']['total'][start_idx:end_idx])
            )
            daily_data['total_evapotranspiration'].append(
                sum(processed_data['agricultural_metrics']['evapotranspiration'][start_idx:end_idx])
            )
        
        return {
            'hourly': processed_data,
            'daily': daily_data,
            'metadata': self._calculate_metadata(daily_data)
        }

    def _calculate_metadata(self, daily_data):
        return {
            'total_precipitation_30d': sum(daily_data['total_precip']),
            'avg_temperature_30d': sum(daily_data['avg_temp']) / len(daily_data['avg_temp']),
            'total_evapotranspiration_30d': sum(daily_data['total_evapotranspiration']),
            'radiation_summary': {
                'avg_daily_total': sum(daily_data['total_radiation']) / len(daily_data['total_radiation'])
            }
        }

class BalloonDataCache:
    def __init__(self):
        self.data = None
        self.last_fetch_time = 0
        self.cache_duration = 24 * 60 * 60  # 24 hours in seconds
        self.error_count = 0
        self.max_retries = 3
        self.last_error_time = 0
        self.error_cooldown = 300  # 5 minutes in seconds
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def get_data(self) -> Optional[FlightData]:
        with self._lock:
            current_time = time.time()
            
            # Check if we need to refresh the cache
            if (self.data is None or 
                current_time - self.last_fetch_time > self.cache_duration):
                
                # Check error cooldown
                if (self.error_count >= self.max_retries and 
                    current_time - self.last_error_time < self.error_cooldown):
                    self.logger.warning("In error cooldown period")
                    return self.data
                
                try:
                    new_data = fetch_balloon_data()
                    if new_data is not None:
                        self.data = new_data
                        self.last_fetch_time = current_time
                        self.error_count = 0
                    else:
                        self.error_count += 1
                        self.last_error_time = current_time
                        self.logger.error(f"Failed to fetch data (attempt {self.error_count})")
                except Exception as e:
                    self.error_count += 1
                    self.last_error_time = current_time
                    self.logger.error(f"Error fetching data: {str(e)}")
            
            return self.data

    def is_cache_valid(self) -> bool:
        return (self.data is not None and 
                time.time() - self.last_fetch_time <= self.cache_duration) 