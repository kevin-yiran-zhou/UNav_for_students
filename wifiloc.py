import subprocess
import json
import re
import requests
def get_wifi_list():
    cmd = ["nmcli", "-t", "-f", "BSSID,SIGNAL", "device", "wifi", "list"]
    output = subprocess.check_output(cmd).decode('utf-8').strip().split('\n')
    networks = []

    for line in output:
        # Use regex to extract BSSID (MAC address) and SIGNAL
        bssid_match = re.search(r'([0-9A-Fa-f]{2}\\:){5}[0-9A-Fa-f]{2}', line)
        
        signal_match = re.search(r'(?<=:)\d+$', line)
        
        if not bssid_match or not signal_match:
            print(f"Failed to parse line: {line}")
            continue

        bssid = bssid_match.group(0).replace("\\","").lower()
        signal = int(signal_match.group(0))

        networks.append({
            "macAddress": bssid,
            "signalStrength": -signal,  # Assuming that the signal strength is given in negative dBm
            "signalToNoiseRatio": 0     # Placeholder value, adjust if you have real SNR data
        })

    return networks
def get_geolocation(wifi_networks,google_api_key):

    wifi_networks = get_wifi_list()
    #for network in wifi_networks:
        #print(network["macAddress"])

    data = {
        "considerIp":"false",
        "wifiAccessPoints": wifi_networks
    }
    response = requests.post(
        f"https://www.googleapis.com/geolocation/v1/geolocate?key={google_api_key}",
        json=data
    )
    return response.json()["location"]



def reverse_geocode(latitude, longitude, google_api_key):
    GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
    params = {
        'latlng': f"{latitude},{longitude}",
        'sensor': 'false',
        'key': google_api_key
    }
    response = requests.get(GOOGLE_MAPS_API_URL,params=params).json()
    if response.get("results"):
        address = response['results'][0]['formatted_address']
        return address
    return None

def main():    
    wifi_networks = get_wifi_list()    
    google_api_key = "AIzaSyBQv5i--JZw-8f8tI1v-3BqCTSnDq2p-IE"
    location = get_geolocation(wifi_networks,google_api_key)
    
    latitude = location["lat"]
    longitude = location["lng"]
    #latitude = 40.6942977
    #longitude = -73.9865759
    print(f"Latitude:{latitude},Longitude: {longitude}")
    
    if latitude is None or longitude is None:
        print("Failed to obtain latitude and longitude.")
        return
    
    address= reverse_geocode(latitude, longitude, google_api_key)
    if address:
        print(f"Address:{address}")
    else:
        print("Could not get address details.")
        
        
if __name__== "__main__":
    main()