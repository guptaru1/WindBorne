import json
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import FlightData, BalloonPosition

logger = logging.getLogger(__name__)

def fetch_balloon_data(hours=24):
    '''
    Fetches balloon data from our windborne API SYSTEM
    '''
    base_url = "https://a.windbornesystems.com/treasure/"
    flight_data = FlightData()
    
    def fetch_hour_data(hour):
        try:
            response = requests.get(f"{base_url}{hour}")
            if response.status_code == 200:
                data = response.json()
                for balloon in data:
                    if flight_data.is_valid_balloon_data(
                        balloon['latitude'],
                        balloon['longitude'],
                        balloon['altitude']
                    ):
                        flight_data.add_balloon_position(
                            hour,
                            balloon['latitude'],
                            balloon['longitude'],
                            balloon['altitude']
                        )
                return True
        except Exception as e:
            logger.error(f"Error fetching data for hour {hour}: {str(e)}")
            return False
    
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_hour_data, hour) for hour in range(hours)]
            results = [future.result() for future in as_completed(futures)]
            
            if not any(results):
                logger.error("Failed to fetch any valid data")
                return None
            
        return flight_data
    except Exception as e:
        logger.error(f"Error in fetch_balloon_data: {str(e)}")
        return None 