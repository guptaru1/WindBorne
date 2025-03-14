from flask import Flask, render_template, jsonify, request, send_file, session

from collections import defaultdict
import math
import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from heapq import heappush, heappop
from bisect import bisect_left
from dataclasses import dataclass
from typing import List, Dict, Optional
from functools import total_ordering
import time
import logging
from datetime import datetime, timedelta
import csv
from io import StringIO
import threading
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from chatmodel import ChatModel
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for session management

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

class FlightData:
    def __init__(self):
        self.sorted_data: Dict[int, Dict[str, List[BalloonPosition]]] = {}  # hour -> continent -> sorted balloons
        self.lat_indices: Dict[int, Dict[str, List[float]]] = {}  # hour -> continent -> sorted latitudes for binary search
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
            
            # Initialize dictionaries if needed
            if hour not in self.sorted_data:
                self.sorted_data[hour] = {}
                self.lat_indices[hour] = {}
            if continent not in self.sorted_data[hour]:
                self.sorted_data[hour][continent] = []
                self.lat_indices[hour][continent] = []
            
            # Insert balloon maintaining sorted order by latitude
            self.sorted_data[hour][continent].append(balloon)
            self.lat_indices[hour][continent].append(balloon.latitude)
            # Sort both lists if this is the last balloon for this hour/continent
            if len(self.sorted_data[hour][continent]) % 1000 == 0:  # Sort periodically
                self._sort_continent_data(hour, continent)

    def _sort_continent_data(self, hour: int, continent: str):
        """Sort balloon data and update indices"""
        balloons = self.sorted_data[hour][continent]
        balloons.sort(key=lambda b: b.latitude)  # Sort by latitude
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
    
    def get_representative_balloon(self, hour: int, continent: str = None) -> Optional[BalloonPosition]:
        if hour not in self.sorted_data:
            return None
        
        if continent:
            if continent not in self.sorted_data[hour] or not self.sorted_data[hour][continent]:
                return None
            
            boundaries = self.continent_boundaries.get(continent)
            if not boundaries:
                return None
            
            center_lat = (boundaries["lat_range"][0] + boundaries["lat_range"][1]) / 2
            center_lon = (boundaries["lon_range"][0] + boundaries["lon_range"][1]) / 2
            
            lat_indices = self.lat_indices[hour][continent]
            balloons = self.sorted_data[hour][continent]
            
            if not lat_indices:
                return None
            
            idx = bisect_left(lat_indices, center_lat)
            
            window_size = 5
            start_idx = max(0, idx - window_size)
            end_idx = min(len(balloons), idx + window_size + 1)
            
            closest_balloon = None
            min_distance = float('inf')
            
            for i in range(start_idx, end_idx):
                balloon = balloons[i]
                dist = (balloon.latitude - center_lat) ** 2 + (balloon.longitude - center_lon) ** 2
                if dist < min_distance:
                    min_distance = dist
                    closest_balloon = balloon
                
            return closest_balloon
        else:
            for continent_balloons in self.sorted_data[hour].values():
                if continent_balloons:
                    return continent_balloons[0]
        return None

    def get_balloon_positions_for_hour(self, hour, continent=None):
        if hour not in self.sorted_data:
            return []
        
        if continent:
            return self.sorted_data[hour].get(continent, [])
        
        # If no continent specified, return all balloons
        all_balloons = []
        for continent_balloons in self.sorted_data[hour].values():
            all_balloons.extend(continent_balloons)
        return all_balloons

    def to_dict(self):
        result = {}
        for hour, continent_data in self.sorted_data.items():
            result[hour] = []
            for continent_balloons in continent_data.values():
                result[hour].extend([vars(balloon) for balloon in continent_balloons])
        return result

    def find_nearest_balloon(self, hour: int, target_lat: float, target_lon: float, continent: str = None) -> Optional[BalloonPosition]:
        """Find nearest balloon using binary search"""
        if hour not in self.sorted_data:
            return None

        # Get balloons for specified continent or all continents
        balloons = []
        if continent:
            if continent not in self.sorted_data[hour]:
                return None
            # Binary search in the sorted list for this continent
            idx = bisect_left(self.lat_indices[hour][continent], target_lat)
            balloons = self.sorted_data[hour][continent]
        else:
            # Combine sorted lists from all continents
            for cont in self.sorted_data[hour].values():
                balloons.extend(cont)
            # Sort combined list if needed
            balloons.sort(key=lambda b: b.latitude)

        # Find closest balloon using window around binary search result
        window_size = 5
        start_idx = max(0, idx - window_size)
        end_idx = min(len(balloons), idx + window_size + 1)
        
        nearest = None
        min_dist = float('inf')
        for i in range(start_idx, end_idx):
            balloon = balloons[i]
            dist = (balloon.latitude - target_lat) ** 2 + (balloon.longitude - target_lon) ** 2
            if dist < min_dist:
                min_dist = dist
                nearest = balloon
        
        return nearest

    def get_most_recent_hour(self, max_hour: int = 24, prioritize_first: bool = True) -> Optional[int]:
        if prioritize_first:
            for hour in [0, 1, 2]:
                if hour in self.sorted_data and any(self.sorted_data[hour].values()):
                    return hour
        
        for hour in range(max_hour):
            if hour in self.sorted_data and any(self.sorted_data[hour].values()):
                return hour
        return None

    def get_24h_trends(self, target_lat: float, target_lon: float) -> dict:
        """
        Get extended forecast trends using Ensemble API for agricultural planning.
        
        Args:
            target_lat (float): Target latitude
            target_lon (float): Target longitude
            
        Returns:
            dict: Dictionary containing trends and forecasts for agricultural planning
        """
        # Get extended forecast from Open-Meteo Ensemble API
        ensemble_url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": target_lat,
            "longitude": target_lon,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "dew_point_2m",
                "et0_fao_evapotranspiration",
                "soil_moisture_40_to_100cm"

            ],
            "forecast_days": 30,  # Extended to 30 days
            "models": "gfs_seamless"
        }
        
        try:
            response = requests.get(ensemble_url, params=params)
            ensemble_data = response.json()
            trends = {
                'temperature': [],
                'humidity': [],
                'dew_point': [],
                'evapotranspiration': [],
                'time': ensemble_data['hourly']['time']
            }
            
            # Process extended forecast data
            for i in range(len(ensemble_data['hourly']['time'])):
                trends['temperature'].append(ensemble_data['hourly']['temperature_2m'][i])
                trends['humidity'].append(ensemble_data['hourly']['relative_humidity_2m'][i])
                trends['dew_point'].append(ensemble_data['hourly']['dew_point_2m'][i])
                trends['evapotranspiration'].append(ensemble_data['hourly']['et0_fao_evapotranspiration'][i])
            
            # Calculate daily averages for easier analysis
            daily_trends = {
                'temperature': [],
                'humidity': [],
                'dew_point': [],
                'evapotranspiration': [],
                'dates': []
            }
            
            # Group hourly data into daily averages
            for i in range(0, len(trends['time']), 24):
                date = trends['time'][i].split('T')[0]
                daily_trends['dates'].append(date)
                
                for key in ['temperature', 'humidity', 'dew_point', 'evapotranspiration']:
                    daily_value = sum(trends[key][i:i+24]) / 24
                    daily_trends[key].append(daily_value)
            
            # Calculate trend analysis for LLM
            trends['llm_summary'] = f"""
            30-Day Agricultural Forecast for {target_lat}°N, {target_lon}°E:
            
            Short-term (Next 7 days):
            - Temperature Range: {min(daily_trends['temperature'][:7]):.1f}°C to {max(daily_trends['temperature'][:7]):.1f}°C
            - Average Humidity: {sum(daily_trends['humidity'][:7])/7:.1f}%
            - Daily Evapotranspiration: {sum(daily_trends['evapotranspiration'][:7])/7:.2f} mm/day
            
            Medium-term (Week 2-3):
            - Temperature Trend: {'increasing' if daily_trends['temperature'][13] > daily_trends['temperature'][7] else 'decreasing'}
            - Humidity Trend: {'increasing' if daily_trends['humidity'][13] > daily_trends['humidity'][7] else 'decreasing'}
            
            Long-term (Week 4):
            - Temperature Outlook: {sum(daily_trends['temperature'][21:28])/7:.1f}°C average
            - Expected Evapotranspiration: {sum(daily_trends['evapotranspiration'][21:28])/7:.2f} mm/day
            
            Key Agricultural Indicators:
            1. Frost Risk: {'Yes' if min(trends['temperature']) < 2 else 'No'}
            2. Heat Stress Risk: {'Yes' if max(trends['temperature']) > 30 else 'No'}
            3. Dew Point Conditions: Ranging from {min(daily_trends['dew_point']):.1f}°C to {max(daily_trends['dew_point']):.1f}°C
            4. Water Requirements: {sum(daily_trends['evapotranspiration']):.1f} mm total over 30 days
            """
            
            # Add both hourly and daily trends to the return data
            return {
                'hourly_trends': trends,
                'daily_trends': daily_trends,
                'llm_summary': trends['llm_summary']
            }
            
        except Exception as e:
            logger.error(f"Error fetching ensemble forecast data: {str(e)}")
            return None

class OpenMeteo:
    def __init__(self):
        self.climate_api = "https://climate-api.open-meteo.com/v1/climate"
        self.air_api = "https://air-quality-api.open-meteo.com/v1/air-quality"
        self.model = "MRI_AGCM3_2_S"

    def _request_get(self, url: str, params: dict) -> dict:
        try:
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return {}

    def fetch_climate_data(self, latitude: float, longitude: float) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "models": self.model,
            "daily": [
                "temperature_2m_max",
                "wind_speed_10m_mean",
                "pressure_msl_mean",
                "soil_moisture_0_to_10cm_mean"
            ]
        }
        
        data = self._request_get(self.climate_api, params)
        
        default_data = {
            "temperature": [],
            "wind_speed": [],
            "pressure": [],
            "soil_moisture": [],
            "time": []
        }
        
        if "daily" in data:
            return {
                "temperature": data["daily"].get("temperature_2m_max", []),
                "wind_speed": data["daily"].get("wind_speed_10m_mean", []),
                "pressure": data["daily"].get("pressure_msl_mean", []),
                "soil_moisture": data["daily"].get("soil_moisture_0_to_10cm_mean", []),
                "time": data["daily"].get("time", [])
            }
        return default_data

    def fetch_air_quality(self, latitude: float, longitude: float) -> dict:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": [
                "dust",
                "alder_pollen",
                "carbon_dioxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "ammonia",
                "methane"
            ]
        }
        
        data = self._request_get(self.air_api, params)
        
        default_data = {
            "dust": [],
            "alder_pollen": [],
            "carbon_dioxide": [],
            "nitrogen_dioxide": [],
            "sulphur_dioxide": [],
            "ozone": [],
            "ammonia": [],
            "methane": [],
            "time": []
        }
        
        if "hourly" in data:
            return {
                "dust": data["hourly"].get("dust", []),
                "alder_pollen": data["hourly"].get("alder_pollen", []),
                "carbon_dioxide": data["hourly"].get("carbon_dioxide", []),
                "nitrogen_dioxide": data["hourly"].get("nitrogen_dioxide", []),
                "sulphur_dioxide": data["hourly"].get("sulphur_dioxide", []),
                "ozone": data["hourly"].get("ozone", []),
                "ammonia": data["hourly"].get("ammonia", []),
                "methane": data["hourly"].get("methane", []),
                "time": data["hourly"].get("time", [])
            }
        return default_data

    def fetch_data(self, latitude: float, longitude: float, data_type: str) -> Optional[dict]:
        if data_type == 'weather':
            return self.fetch_climate_data(latitude, longitude)
        elif data_type == 'air_quality':
            return self.fetch_air_quality(latitude, longitude)
        return None

    def fetch_ensemble_forecast(self, lat: float, lon: float) -> dict:
 
        ensemble_url = "https://ensemble-api.open-meteo.com/v1/ensemble"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",          # Surface temperature
                "relative_humidity_2m",    # Surface humidity
                "dew_point_2m",           # Dew point
                "precipitation",           # Precipitation
                "et0_fao_evapotranspiration",  # Evapotranspiration
                "temperature_120m",        # Upper air temperature
                "soil_moisture_40_to_100cm",  # Deep soil moisture
                "direct_radiation",        # Direct solar radiation
                "diffuse_radiation"        # Diffuse solar radiation
            ],
            "forecast_days": 30,
            "models": "gfs_seamless"
        }
        
        try:
            response = requests.get(ensemble_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process and structure the data
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
            
            # Calculate daily aggregates
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
                daily_data['avg_temp'].append(
                    sum(processed_data['surface_conditions']['temperature'][start_idx:end_idx]) / 24
                )
                daily_data['min_temp'].append(
                    min(processed_data['surface_conditions']['temperature'][start_idx:end_idx])
                )
                daily_data['max_temp'].append(
                    max(processed_data['surface_conditions']['temperature'][start_idx:end_idx])
                )
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
                'metadata': {
                    'total_precipitation_30d': sum(daily_data['total_precip']),
                    'avg_temperature_30d': sum(daily_data['avg_temp']) / len(daily_data['avg_temp']),
                    'total_evapotranspiration_30d': sum(daily_data['total_evapotranspiration']),
                    'radiation_summary': {
                        'avg_daily_total': sum(daily_data['total_radiation']) / len(daily_data['total_radiation'])
                    }
                }
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching ensemble forecast: {str(e)}")
            return None

def fetch_balloon_data(hours=24):
    '''
    Fetches ballon data from our windborne API SYSTEM
    '''
    base_url = "https://a.windbornesystems.com/treasure/"
    flight_data = FlightData()
    
    with requests.Session() as session:
        def fetch_hour_data(hour):
            url = f"{base_url}{str(hour).zfill(2)}.json"
            try:
                response = session.get(url, timeout=5)
                response.raise_for_status()
                text = response.text.strip()
                
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    # Quick string parsing without multiple splits
                    coords_list = [
                        [float(x) for x in coords.strip('[]').split(',')]
                        for coords in text.replace('\n', '').replace(' ', '').strip('[]').split('],[')
                        if coords.strip('[]')
                    ]
                    data = [coord for coord in coords_list if len(coord) == 3]
                
                return hour, data
            except Exception as e:
                logger.error(f"Error fetching hour {hour}: {str(e)}")
                return hour, []

        try:
            # Process hours in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_hour = {
                    executor.submit(fetch_hour_data, hour): hour 
                    for hour in range(hours)
                }
                
                for future in as_completed(future_to_hour):
                    hour, data = future.result()
                    if data:
                        for coords in data:
                            if isinstance(coords, list) and len(coords) == 3:
                                latitude, longitude, altitude = coords
                                flight_data.add_balloon_position(hour, latitude, longitude, altitude)

            return flight_data
        except Exception as e:
            logger.error(f"Error in fetch_balloon_data: {str(e)}")
            return None

class BalloonDataCache:
    def __init__(self):
        self.data = None
        self.last_fetch_time = 0
        self.cache_duration = 300 # 1 hour in seconds
        self.error_count = 0
        self.max_retries = 3
        self.last_error_time = 0
        self.error_cooldown = 300 
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()

    def is_cache_valid(self) -> bool:
        """Check if the cached data is still valid."""
        current_time = time.time()
        # Cache is invalid if more than cache_duration has passed
        return (current_time - self.last_fetch_time) < self.cache_duration

    def refresh_data(self) -> bool:
        """Force refresh the cache data."""
        if not self._lock.acquire(blocking=False):  # Non-blocking lock
            self.logger.warning("Another refresh is in progress, skipping...")
            return False
        
        try:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(f"[{current_time}] Refreshing balloon data cache...")
            new_data = fetch_balloon_data(24)
            if new_data is not None:
                self.data = new_data
                self.last_fetch_time = time.time()
                self.error_count = 0
                self.logger.info(f"[{current_time}] Successfully refreshed balloon data cache")
                return True
            return False
        finally:
            self._lock.release()  # Always release the lock

    def get_data(self) -> Optional[object]:
        """Get balloon data with automatic refresh if stale."""
        with self._lock:
            try:
                # Check if cache needs refresh
                if not self.is_cache_valid():
                    self.logger.info("Cache is stale, attempting refresh...")
                    if self.should_retry():
                        self.refresh_data()
                    else:
                        self.logger.warning("Too many errors, using stale cache data")

                return self.data

            except Exception as e:
                self.logger.error(f"Error in get_data: {str(e)}")
                return self.data

    def should_retry(self) -> bool:
        if self.error_count >= self.max_retries:
            # Check if we've waited long enough since the last error
            if time.time() - self.last_error_time > self.error_cooldown:
                self.error_count = 0  # Reset error count after cooldown
                return True
            return False
        return True

balloon_cache = BalloonDataCache()

# Set up scheduler for periodic cache refresh
from apscheduler.schedulers.background import BackgroundScheduler

def init_scheduler():
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        balloon_cache.refresh_data,
        'interval',
        minutes=5,
        id='refresh_balloon_cache'
    )
    scheduler.start()
    return scheduler

# Initialize the scheduler
scheduler = init_scheduler()

# Update initialize_app to include scheduler status
def initialize_app():
    try:
        logger.info("Initializing application cache...")
        data = balloon_cache.get_data()
        if data is not None:
            logger.info("Successfully initialized application cache")
        else:
            logger.warning("Failed to initialize cache with data")
    except Exception as e:
        logger.error(f"Failed to initialize cache: {str(e)}")


initialize_app()
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    cache_status = "valid" if balloon_cache.is_cache_valid() else "invalid"
    cache_age = time.time() - balloon_cache.last_fetch_time if balloon_cache.last_fetch_time > 0 else None
    
    return jsonify({
        "status": "healthy",
        "cache_status": cache_status,
        "cache_age_seconds": cache_age,
        "error_count": balloon_cache.error_count
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_balloon_path')
def get_balloon_path():
    '''
    Returns live location of all balloons using most recent data
    '''
    try:
        flight_data = balloon_cache.get_data()
        if flight_data is None:
            return jsonify({
                "error": "No balloon data available",
                "hour0_balloons": [],
                "continents": {},
                "hour": None
            }), 503

        recent_hour = flight_data.get_most_recent_hour()
        if recent_hour is None:
            return jsonify({
                "hour0_balloons": [],
                "continents": {},
                "hour": None
            })

        parsed_data = flight_data.to_dict()
        continent = request.args.get('continent')
        live_balloon_locations = parsed_data[recent_hour]
        
        clicked_lat = request.args.get('clicked_lat', type=float)
        clicked_lon = request.args.get('clicked_lon', type=float)

        if clicked_lat and clicked_lon:
            # Find nearest balloon using efficient spatial search
            nearest = None
            min_dist = float('inf')
            
            # Filter by continent if specified
            balloon_list = [b for b in live_balloon_locations if continent is None or b['continent'] == continent]
            
            for balloon in balloon_list:
                dist = (balloon['latitude'] - clicked_lat) ** 2 + (balloon['longitude'] - clicked_lon) ** 2
                if dist < min_dist:
                    min_dist = dist
                    nearest = balloon

            print("helllllooooo")
            print(nearest)
            if nearest:
                return jsonify({
                    "nearest_balloon": nearest,
                    "redirect_url": f"/weather-forecast?lat={nearest['latitude']}&lon={nearest['longitude']}&continent={nearest['continent']}"
                })
        
        # If continent is specified, filter balloons
        if continent:
            live_balloon_locations = [b for b in live_balloon_locations if b['continent'] == continent]
            return jsonify({
                "hour0_balloons": live_balloon_locations,
                "hour": recent_hour,
                "continent": continent
            })
        
        # Calculate continent statistics
        continents = {}
        for balloon in live_balloon_locations:
            cont = balloon['continent']
            if cont not in continents:
                continents[cont] = {'count': 0, 'total_altitude': 0}
            continents[cont]['count'] += 1
            continents[cont]['total_altitude'] += balloon['altitude']
        
        # Calculate averages
        for cont in continents:
            if continents[cont]['count'] > 0:
                continents[cont]['avg_altitude'] = continents[cont]['total_altitude'] / continents[cont]['count']
                del continents[cont]['total_altitude']
        
        return jsonify({
            "hour0_balloons": live_balloon_locations,
            "continents": continents,
            "hour": recent_hour,
            "cache_time": balloon_cache.last_fetch_time
        })

    except Exception as e:
        logger.error(f"Error in get_balloon_path: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "hour0_balloons": [],
            "continents": {},
            "hour": None
        }), 500

@app.route('/get_continent_stats')
def get_continent_stats():
    '''
    Returns statistics for balloons in each continent
    '''
    try:
        flight_data = balloon_cache.get_data()
        if flight_data is None:
            return jsonify({
                'continent_stats': {},
                'hour': None
            })

        recent_hour = flight_data.get_most_recent_hour()
        if recent_hour is None:
            return jsonify({
                'continent_stats': {},
                'hour': None
            })

        # Calculate statistics for each continent
        continent_stats = {}
        for continent, balloons in flight_data.sorted_data[recent_hour].items():
            if balloons:  # Only process if there are balloons
                total_altitude = sum(b.altitude for b in balloons)
                total_lat = sum(b.latitude for b in balloons)
                total_lon = sum(b.longitude for b in balloons)
                count = len(balloons)
                
                # Ensure we have valid numbers
                if count > 0 and all(isinstance(x, (int, float)) for x in [total_altitude, total_lat, total_lon]):
                    continent_stats[continent] = {
                        'count': count,
                        'avg_altitude': total_altitude / count,
                        'center_lat': total_lat / count,
                        'center_lon': total_lon / count
                    }

        # Log the stats for debugging
        logger.info(f"Continent stats: {continent_stats}")

        return jsonify({
            'continent_stats': continent_stats,
            'hour': recent_hour,
            'cache_time': balloon_cache.last_fetch_time
        })

    except Exception as e:
        logger.error(f"Error in get_continent_stats: {str(e)}")
        return jsonify({
            'continent_stats': {},
            'hour': None
        }), 500

@app.route('/weather-forecast')
def weather_forecast():
    try:
        continent = request.args.get('continent')
        lat = request.args.get('lat', type=float)
        lon = request.args.get('lon', type=float)
        
        if lat is None or lon is None:
            return "Missing latitude or longitude", 400

        # Fetch both weather and air quality data
        weather_service = OpenMeteo()
        weather_data = weather_service.fetch_climate_data(lat, lon)
        air_quality_data = weather_service.fetch_air_quality(lat, lon)
        print(air_quality_data)
            
        return render_template('weather_forecast.html', 
                             latitude=lat,
                             longitude=lon,
                             weather_data=weather_data,
                             air_quality_data=air_quality_data,
                             balloon_history=[],
                             continent=continent)
                             
    except Exception as e:
        logger.error(f"Error in weather_forecast: {str(e)}")
        return f"Error fetching weather data: {str(e)}", 500

@app.route('/path-trajectory')
def path_trajectory():
    try:
        continent = request.args.get('continent')
        flight_data = balloon_cache.get_data()
        if flight_data is None:
            return jsonify({"error": "No balloon data available"}), 503
        recent_hour = flight_data.get_most_recent_hour()
        if recent_hour is None:
            return jsonify({"error": "No recent balloon data found"}), 404
        balloon = flight_data.get_representative_balloon(recent_hour, continent)
        if not balloon:
            return jsonify({"error": f"No balloons found for continent: {continent}"}), 404
        balloon_history = []
        for hour in range(24):
            if hour in flight_data.sorted_data:
                nearest = flight_data.find_nearest_balloon(hour, balloon.latitude, balloon.longitude, continent)
                if nearest:
                    balloon_history.append({
                        'hour': hour,
                        'latitude': nearest.latitude,
                        'longitude': nearest.longitude,
                        'altitude': nearest.altitude
                    })
        balloon_history.sort(key=lambda x: x['hour'])
            
        return render_template(
            'path_trajectory.html',
            balloon_history=balloon_history,
            continent=continent
        )
            
    except Exception as e:
        logger.error(f"Error in path trajectory: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



@app.route('/download_weather_csv')
def download_weather_csv():
    try:
        flight_data = balloon_cache.get_data()
        if flight_data is None:
            return jsonify({"error": "No balloon data available"}), 503

        parsed_data = flight_data.to_dict()
        
        # Find hour 0 data, or earliest available hour
        hour0_data = None
        for hour in [0, 1, 2]:  # Prioritize first 3 hours
            if hour in parsed_data and parsed_data[hour]:
                hour0_data = parsed_data[hour]
                break
        
        if not hour0_data:
            # If no data in first 3 hours, find first available hour
            for hour in range(3, 24):
                if hour in parsed_data and parsed_data[hour]:
                    hour0_data = parsed_data[hour]
                    break

        if not hour0_data:
            return jsonify({"error": "No balloon data found"}), 404

        # Limit to 50 balloons and sort by continent
        continent = request.args.get('continent')
        if continent:
            hour0_data = [b for b in hour0_data if b['continent'] == continent]
        hour0_data = sorted(hour0_data, key=lambda x: (x['continent'], x['latitude']))[:20]

        # Use BytesIO instead of StringIO for binary data
        from io import BytesIO
        output = BytesIO()
        writer = csv.writer(output)
        writer.writerow([
            'Latitude (°)',
            'Longitude (°)',
            'Altitude (m)',
            'Continent',
            'Temperature (°C)',
            'Wind Speed (m/s)',
            'Pressure (hPa)',
            'Soil Moisture (%)',
            'Time (UTC)'
        ])

        weather_service = OpenMeteo()
        for balloon in hour0_data:
            try:
                weather_data = weather_service.fetch_data(
                    balloon['latitude'], 
                    balloon['longitude'], 
                    'weather'
                )
                
                # Get the first value from each weather parameter if available
                temp = weather_data['temperature'][0] if weather_data and 'temperature' in weather_data else 'N/A'
                wind = weather_data['wind_speed'][0] if weather_data and 'wind_speed' in weather_data else 'N/A'
                pressure = weather_data['pressure'][0] if weather_data and 'pressure' in weather_data else 'N/A'
                soil = weather_data['soil_moisture'][0] if weather_data and 'soil_moisture' in weather_data else 'N/A'
                time = weather_data['time'][0] if weather_data and 'time' in weather_data else 'N/A'
                
                writer.writerow([
                    f"{balloon['latitude']:.4f}",
                    f"{balloon['longitude']:.4f}",
                    f"{balloon['altitude']:.1f}",
                    balloon['continent'],
                    temp,
                    wind,
                    pressure,
                    soil,
                    time
                ])
            except Exception as e:
                logger.error(f"Error fetching weather data for balloon: {str(e)}")
                continue

        # Prepare the binary data
        output.seek(0)
        
        # Generate timestamp and filename
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f'balloon_weather_data_{continent}_{timestamp}.csv' if continent else f'balloon_weather_data_{timestamp}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename,
            max_age=0
        )

    except Exception as e:
        logger.error(f"Error generating CSV: {str(e)}")
        return jsonify({"error": "Failed to generate CSV"}), 500

@app.route('/agriculture-advice', methods=['POST'])
def agriculture_advice():
    try:
        message = request.json.get('message', '').lower()
        
        # Update system prompt to include formatting
        system_prompt = """You are an expert agricultural advisor. Format your responses with:
1. Clear section headers with emojis
2. Bullet points for lists
3. Important values in bold
4. Clear spacing between sections
5. Organized environmental data at the top
6. Practical, actionable advice with specific quantities

Use markdown-style formatting and emojis to make the information clear and engaging."""

        # Check if this is a follow-up question
        follow_up_phrases = ['explain', 'elaborate', 'tell', 'what', 'how', 'why', 'please', 'can you']
        is_follow_up = any(message.startswith(phrase) for phrase in follow_up_phrases) or 'more' in message
        
        if is_follow_up and 'chat_history' in session:
            # Use existing context for follow-up questions
            system_prompt = session.get('system_prompt', '')
            previous_context = session.get('chat_history', '')
            environmental_data = session.get('environmental_data', {})
            current_crop = session.get('current_crop', '')
            location_name = session.get('location_name', '')
            print("coming in here" ,message)
            #user_prompt = f"""Previous context about growing {current_crop} in {location_name}:
#{previous_context}

#Follow-up question: {message}

#Please answer the follow-up question in a concise manner, with no more than 200 words."""
            
            user_prompt = f"""Previous context For growing {current_crop} in {location_name} previous context {previous_context} :

User asks: {message}

Please answer this specific question only, with practical and actionable advice."""
            
      

            # Initialize ChatModel and get response
            chat_model = ChatModel("gemini")
            llm_response = chat_model.chat(system_prompt, user_prompt)
            
            # Update chat history
            session['chat_history'] = f"{previous_context}\nQ: {message}\nA: {llm_response}"
            
            return jsonify({
                "status": "complete",
                "response": llm_response,
                "is_followup": True
            })
            
        else:
            # New conversation about a crop
            parts = message.split(' ', 1)
            if len(parts) < 2:
                return jsonify({"response": "Please provide both a country and a crop (e.g., 'Spain apple')"})
            
            country = parts[0]
            crop = parts[1]

            try:
                # Geocoding API call
                geocoding_api = "https://geocoding-api.open-meteo.com/v1/search"
                params = {
                    "name": country,
                    "count": 1,
                    "language": "en",
                    "format": "json"
                }
                
                response = requests.get(geocoding_api, params=params)
                response.raise_for_status()
                location_data = response.json()
                
                if not location_data.get("results"):
                    return jsonify({"response": f"Could not find country: '{country}'. Please try another country name (e.g., 'Spain apple' or 'France grape')"})
                
                location = location_data["results"][0]
                weather_service = OpenMeteo()
                
                # Fetch data with error handling
                ensemble_data = weather_service.fetch_ensemble_forecast(
                    location["latitude"],
                    location["longitude"]
                ) or {'hourly': {'soil_conditions': {'deep_moisture': [0.3]}}, 'metadata': {'avg_temperature_30d': 20, 'total_evapotranspiration_30d': 2}}
                
                air_quality = weather_service.fetch_air_quality(
                    location["latitude"],
                    location["longitude"]
                ) or {'carbon_dioxide': [400], 'nitrogen_dioxide': [20]}
                
                # Extract data with safe defaults
                hourly_data = ensemble_data.get('hourly', {})
                metadata = ensemble_data.get('metadata', {})
                soil_moisture = hourly_data.get('soil_conditions', {}).get('deep_moisture', [0.3])
                location_name = f"{location['name']}, {location.get('country', '')}"
                
                # Calculate averages safely
                def safe_average(values, default):
                    try:
                        return sum(values)/len(values) if values else default
                    except (TypeError, ZeroDivisionError):
                        return default

                # Store data in session with safe defaults
                session['location_name'] = location_name
                session['current_crop'] = crop
                session['environmental_data'] = {
                    'temperature': metadata.get('avg_temperature_30d', 20),
                    'evapotranspiration': metadata.get('total_evapotranspiration_30d', 2),
                    'soil_moisture': safe_average(soil_moisture, 0.3),
                    'co2': safe_average(air_quality.get('carbon_dioxide', [400]), 400),
                    'no2': safe_average(air_quality.get('nitrogen_dioxide', [20]), 20)
                }
                
                # Format the prompt with safe values
                env_data = session['environmental_data']
                user_prompt = f"""As an agricultural expert, create a detailed farming plan for growing {crop} in {location_name} based on the current environmental data:

ENVIRONMENTAL CONDITIONS:
• Temperature: {env_data['temperature']:.1f}°C
• Evapotranspiration Rate: {env_data['evapotranspiration']:.1f} mm/day
• Soil Moisture: {env_data['soil_moisture']:.2f} m³/m³
• CO2 Levels: {env_data['co2']:.2f} ppm
• NO2 Levels: {env_data['no2']:.2f} μg/m³

Please provide a comprehensive cultivation plan covering:

1. PLANTING TIMELINE
   • Optimal planting months based on the current temperature of {env_data['temperature']:.1f}°C
   • Growth stages timeline
   • Expected harvest period

2. SOIL & IRRIGATION MANAGEMENT
   • Soil preparation techniques for {crop}
   • Irrigation schedule considering the soil moisture of {env_data['soil_moisture']:.2f} m³/m³
   • Water management adjustments for {env_data['evapotranspiration']:.1f} mm/day evapotranspiration

3. FERTILIZATION STRATEGY
   • Recommended fertilizer types and NPK ratios based on soil moisture
   • Application schedule and quantities
   • Organic vs synthetic fertilizer recommendations

4. ENVIRONMENTAL OPTIMIZATION
   • How to optimize growth with CO2 levels of {env_data['co2']:.2f} ppm
   • Managing NO2 exposure of {env_data['no2']:.2f} μg/m³
   • Climate control suggestions if needed

5. CROP CARE PROTOCOL
   • Spacing and support systems
   • Pruning and training methods
   • Pest and disease prevention specific to {location_name}
   • Companion planting recommendations

6. HARVEST & POST-HARVEST
   • Indicators for harvest readiness
   • Proper harvesting techniques
   • Storage recommendations

Please provide specific, actionable advice that a farmer can implement, including quantities and timelines where applicable."""

                # Update the system prompt to be more specific
                system_prompt = """You are an expert agricultural advisor with deep knowledge of crop science and farming practices. 

FORMAT YOUR RESPONSE WITH:
1. Start with a clear title and introduction
2. Use bullet points (•) for lists
3. Number main sections (1., 2., etc.)
4. Use clear section headers in CAPS
5. Indent sub-points properly
6. Add proper spacing between sections
7. Highlight important values in **bold**
8. Use emojis (🌱, 💧, 🌿, etc.) where appropriate

FOCUS YOUR ADVICE ON:
1. Precise recommendations based on the exact environmental measurements
2. Region-specific agricultural practices
3. Sustainable and efficient farming methods
4. Clear, actionable steps with specific quantities and timelines
5. Scientific explanations for your recommendations

Your response should be structured, detailed, and easy to read with proper formatting and spacing. Each section should provide practical, implementable advice backed by scientific understanding."""
                
                session['system_prompt'] = system_prompt
                session['chat_history'] = user_prompt
                
                chat_model = ChatModel("gemini")
                llm_response = chat_model.chat(system_prompt, user_prompt)
                
                session['chat_history'] += f"\n\nAnswer: {llm_response}"
                
                return jsonify({
                    "status": "complete",
                    "response": llm_response,
                    "is_followup": False
                })
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API error: {str(e)}")
                return jsonify({"response": "Sorry, the weather service is temporarily unavailable. Please try again."})
                
    except Exception as e:
        logger.error(f"Error in agriculture advice: {str(e)}")
        return jsonify({"response": "Sorry, I encountered an error. Please try again."}), 500
    
# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port) # Set debug=False for production