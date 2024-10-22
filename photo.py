import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import time
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

def fetch_camera_images(url):
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.content, 'html.parser')
    cards = soup.find_all("div", class_="card")
    images = []
    for card in cards:
        img_tag = card.find('img')
        if img_tag:
            img_src = img_tag.get('src')
            if img_src:
                if img_src.startswith('//'):
                    img_src = "https:" + img_src
                elif img_src.startswith('/'):
                    img_src = "https://onemotoring.lta.gov.sg" + img_src
                images.append(img_src)
    return images

model = YOLO("yolov8x.pt")

def count_vehicles(image_url):
    try:
        response = requests.get(image_url, verify=False)
        if response.status_code != 200:
            return {'counts': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}, 'img': None}

        img = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {'counts': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}, 'img': None}

        results = model.predict(img)
        vehicle_count = {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}
        
        
        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label in vehicle_count:
                    vehicle_count[label] += 1
                    x1, y1, x2, y2 = box.xyxy[0]  
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Draw rectangle
                    cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Add label

        _, img_encoded = cv2.imencode('.jpg', img)
        img_bytes = img_encoded.tobytes()

        return {'counts': vehicle_count, 'img': img_bytes}
    
    except Exception as e:
        print(f"Error fetching vehicle counts: {e}")
        return {'counts': {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}, 'img': None}

links = {
    "Woodlands": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/woodlands.html#trafficCameras",
    "KJE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/kje.html#trafficCameras",
    "SLE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/sle.html#trafficCameras",
    "BKE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/bke.html#trafficCameras",
    "AYE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/aye.html#trafficCameras",
    "STG": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/stg.html#trafficCameras",
    "TPE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/tpe.html#trafficCameras",
    "KPE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/kpe.html#trafficCameras",
    "CTE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/cte.html#trafficCameras",
    "MCE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/mce.html#trafficCameras",
    "ECP": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/ecp.html#trafficCameras",
    "PIE": "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/traffic_information/traffic-cameras/pie.html#trafficCameras",
}

st.title("Singapore Live Traffic Monitoring")
st.write("Counting vehicles in real-time from traffic camera feeds.")
st.write("Please wait for the first result after which it will automatically update every minute.")

selected_location = st.selectbox("Select Traffic Camera Location", ["All"] + list(links.keys()))
traffic_data = {location: [] for location in links.keys()}

if 'start_monitoring' not in st.session_state:
    st.session_state.start_monitoring = False

if st.button("Start Monitoring"):
    st.session_state.start_monitoring = True

if st.session_state.start_monitoring:
    for location, link in links.items():
        image_urls = fetch_camera_images(link)
        total_counts = {'car': 0, 'bus': 0, 'motorcycle': 0, 'truck': 0}
        
        for img_url in image_urls:
            result = count_vehicles(img_url)
            counts = result['counts']
            img = result['img']

            for vehicle_type in total_counts.keys():
                total_counts[vehicle_type] += counts.get(vehicle_type, 0)

        traffic_data[location].append(total_counts)

    
    if selected_location == "All":
        for location in links.keys():
            if traffic_data[location]:  
                st.write(f"{location} Counts: {traffic_data[location][-1]}")
    else:
        if traffic_data[selected_location]:  
            st.write(f"{selected_location} Counts: {traffic_data[selected_location][-1]}")
            
            
            img_urls = fetch_camera_images(links[selected_location])
            cols = st.columns(3)  
            images_with_results = [count_vehicles(img_url) for img_url in img_urls]

            for i, result in enumerate(images_with_results):
                if result['img'] is not None:
                    with cols[i % 3]:  
                        st.image(result['img'], caption=f"Image {i+1} from {selected_location}", use_column_width=True)
        
        else:
            st.write(f"No data available for {selected_location}.")

    last_refresh_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Data last updated at: {last_refresh_time}")

    
    if selected_location == "All":
        fig = make_subplots(rows=1, cols=1)

        for location in links.keys():
            counts_df = pd.DataFrame(traffic_data[location])
            if not counts_df.empty:
                total_counts = counts_df.sum()
                fig.add_trace(go.Bar(
                    x=total_counts.index,
                    y=total_counts.values,
                    name=location,
                    text=total_counts.values,
                    textposition='auto'
                ))

        if len(fig.data) > 0:  # Ensure there is data to plot
            fig.update_layout(
                title="Comparison of Vehicle Counts Across Locations",
                xaxis_title="Vehicle Types",
                yaxis_title="Count",
                barmode='group'
            )
            st.plotly_chart(fig)
        else:
            st.write("No data available for plotting.")
    else:
        
        counts_df = pd.DataFrame(traffic_data[selected_location])
        if not counts_df.empty:
            total_counts = counts_df.sum()
            fig = go.Figure(data=[
                go.Bar(x=total_counts.index, y=total_counts.values, text=total_counts.values, textposition='auto')
            ])
            fig.update_layout(
                title=f"Vehicle Counts for {selected_location}",
                xaxis_title="Vehicle Types",
                yaxis_title="Count"
            )
            st.plotly_chart(fig)
        else:
            st.write(f"No data available for plotting {selected_location}.")

    time.sleep(50)  
    st.rerun()  
