# netpie_pub.py
import json
import time
import requests
import paho.mqtt.client as mqtt
from datetime import datetime

# ---- NETPIE credentials ----
APPID  = "fea138ea-de9a-49aa-a7a5-4b6cbfb9ddee"
KEY    = "D655uojUKJuz6G5XL8U9aKMUDq672HgG"
SECRET = "jFwiczRebwnxHznbrrBoDcecbZvvLzY1"
BROKER = "mqtt.netpie.io"
PORT   = 1883

# ---- IMGBB ----
IMGBB_KEY = "2ba1067a2834d1571902d4c8fdb38ae5"

# ---- MQTT setup ----
client = mqtt.Client(client_id=APPID)
client.username_pw_set(KEY, SECRET)
client.connect(BROKER, PORT, 60)
client.loop_start()

# ---- Upload picture ----
def upload_to_imgbb(path):
    """Upload image and return URL."""
    with open(path, "rb") as f:
        resp = requests.post(
            "https://api.imgbb.com/1/upload",
            params={"key": IMGBB_KEY},
            files={"image": f}
        )
    j = resp.json()
    if not j.get("success"):
        print("❌ Error uploading picture:", j)
        return ""
    return j["data"]["url"]

# ---- Main publish function ----
def publish_to_netpie(state, color, image_path="", counters=None, extra=None):
    """
    counters = {
        "total_sort": int,
        "success_sort": int,
        "red_sort": int,
        "blue_sort": int,
        "grey_sort": int,
        "white_sort": int
    }
    """
    image_url = ""
    if image_path:
        try:
            image_url = upload_to_imgbb(image_path)
        except Exception as e:
            print("❌ Failed to upload image:", e)

    payload = {
        "state": state,
        "color": color,
        "image_url": image_url,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }

    # Add counters if provided
    if counters:
        payload.update(counters)

    # Add extra (size, confidence)
    if extra:
        payload.update(extra)

    client.publish("@shadow/data/update", json.dumps({"data": payload}), qos=1)

    print("📤 Published:", json.dumps(payload, indent=2))
