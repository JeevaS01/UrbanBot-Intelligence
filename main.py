import os
import boto3
import pymysql
import smtplib
import pickle
import pandas as pd
from flask import Flask, redirect, render_template, request, flash,jsonify, url_for
from streamlit import status
from ultralytics import YOLO
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.sql import SQLTools
from phi.tools.email import EmailTools
import markdown
#from download_models import download_urbanbot_models

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = "traffic_secret_key"

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- GLOBAL MODEL DICTIONARY ---
# Loading all 6 into a dictionary for clean access
intelligence_engines = {}

def bootstrap_intelligence():
    """Ensures all 6 models are downloaded and loaded into memory."""
    global intelligence_engines
    # 1. Trigger the S3 sync (Downloads only if files are missing)
    print("🛰️ Syncing Intelligence Engines from S3...")
    #download_urbanbot_models()
    
    # 2. Map of Model Names to Local Paths
    model_map = {
        'traffic_model': 'models/traffic.pt',
        'crowd_model': 'models/crowd.pt',
        'accident_model': 'models/accident.pt',
        'pothole_model': 'models/pothole.pt',
        'model': 'models/AQI_Model.pkl',
        'scaler': 'models/AQI_scaler.pkl'
    }

    # 3. Load YOLO models
    try:
        for name in ['traffic_model', 'crowd_model', 'accident_model', 'pothole_model']:
            intelligence_engines[name] = YOLO(model_map[name])
            print(f"✅ Loaded YOLO: {name}")

        # 4. Load Pickle models (LSTM/NLP)
        for name in ['model', 'scaler']:
            with open(model_map[name], 'rb') as f:
                intelligence_engines[name] = pickle.load(f)
            print(f"✅ Loaded Pickle: {name}")
            
    except Exception as e:
        print(f"❌ Critical Error Loading Models: {e}")

# --- INITIALIZE BEFORE STARTUP ---
bootstrap_intelligence()



# local Load Model
# traffic_model = YOLO(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\Traffic Model\best.pt')
# accident_model = YOLO(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\Accident Model\best.pt')
# pothole_model = YOLO(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\Pothole Model\best.pt')
# crowd_model = YOLO(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\Crowd Model\best.pt')

# Weights
traffic_weights = {
    'ambulance': 3.0, 'army vehicle': 3.5, 'auto rickshaw': 0.8, 'bicycle': 0.2,
    'bus': 3.0, 'car': 1.0, 'garbagevan': 3.0, 'human hauler': 1.5,
    'minibus': 1.5, 'minivan': 1.0, 'motorbike': 0.5, 'pickup': 1.0,
    'policecar': 3.0, 'rickshaw': 0.6, 'scooter': 0.5, 'suv': 1.0,
    'taxi': 1.0, 'three wheelers -CNG-': 0.8, 'truck': 3.0, 'van': 1.0,
    'wheelbarrow': 0.6
}
accident_weights = {"moderate": 1.0, "severe": 1.5}
pothole_weights = {'pothole': 1.0}
crowd_weights = {'crowd': 1.0}


#------------------------------------------------------------------------------------------------------------------
# EMAIL ALERT FUNCTION FOR TRAFFIC
def traffic_send_email_alert(status, image_url):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASS")
    receiver = os.getenv("RECEIVER")
    
    # Time formatting for the email body
    local_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    formatted_time = local_time.strftime("%B-%d-%Y | %I:%M %p")
    
    # 1. Logic for dynamic Email Subject and Body
    if status == 'LOW TRAFFIC':
        subject = "🟢 Traffic Update: Normal Flow"
        body = f"""
        <html><body>
            <h3 style="color: green;">✅ <b>Traffic Status: {status}</b></h3>
            <p>Traffic is currently flowing normally at <b>Camera 4</b>.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body></html>
        """
    elif status == 'MEDIUM TRAFFIC':
        subject = "🟡 Traffic Alert: Moderate Congestion"
        body = f"""
        <html><body>
            <h3 style="color: orange;">⚠️ <b>Traffic Status: {status}</b></h3>
            <p>Moderate traffic buildup has been detected at <b>Camera 4</b>. Monitoring recommended.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body></html>
        """
    elif status == 'HIGH TRAFFIC':
        subject = "🔴 URGENT: High Traffic Jam Detected"
        body = f"""
        <html><body>
            <h2 style="color: red;">🚨 <b>HIGH TRAFFIC JAM ALERT</b></h2>
            <p>Critical congestion detected at <b>Camera 4</b>. Immediate attention required!</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view snapshot</a></p>
        </body></html>
        """
    else:
        return False # No email for 'EMPTY' or unknown status

    # 2. Construct and Send Email
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email Alert Sent: {status}")
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False
#------------------------------------------------------------------------------------------------------------------
#EMAIL ALERT FUNCTION FOR ACCIDENT
def accident_send_email_alert(status, image_url):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASS")
    receiver = os.getenv("RECEIVER")
    
    # Time formatting for the email body
    local_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    formatted_time = local_time.strftime("%B-%d-%Y | %I:%M %p")
    
    # 1. Logic for dynamic Email Subject and Body
    if status == 'MODERATE ACCIDENT':
        subject = "🟠 Alert: Vehicle Collision Detected"
        body = f"""
        <html>
        <body>
            <h3 style="color: orange;">⚠️ <b>Accident Status: {status}</b></h3>
            <p>A vehicle collision has been detected at <b>Camera 4</b>. Traffic flow may be impacted. Recovery services notified.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body>
        </html>
        """

    elif status == 'SEVERE ACCIDENT':
        subject = "🔴 URGENT: Major Road Accident Detected"
        body = f"""
        <html>
        <body>
            <h2 style="color: red;">🚨 <b>CRITICAL ACCIDENT ALERT</b></h2>
            <p>A severe accident (high-impact or overturned vehicle) detected at <b>Camera 4</b>. <b>Immediate Emergency Response Required!</b></p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view snapshot</a></p>
        </body>
        </html>
        """
    else:
        return False # No email for 'EMPTY' or unknown status

    # 2. Construct and Send Email
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email Alert Sent: {status}")
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False
#------------------------------------------------------------------------------------------------------------------
#EMAIL ALERT FUNCTION FOR POTHOLE
def pothole_send_email_alert(status, image_url):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASS")
    receiver = os.getenv("RECEIVER")
    
    # Time formatting for the email body
    local_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    formatted_time = local_time.strftime("%B-%d-%Y | %I:%M %p")

    if status == 'LOW POTHOLE':
        subject = "🟡 Pothole Spotted: Minor Issue"
        body = f"""
        <html>
        <body>
            <h3 style="color: #D4AC0D;">⚠️ <b>Pothole Status: {status}</b></h3>
            <p>A small pothole has been detected at <b>Camera 4</b>. Surface wear is beginning to show.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body>
        </html>
        """

    elif status == 'MEDIUM POTHOLE':
        subject = "🟠 Maintenance Alert: Pothole Repair Needed"
        body = f"""
        <html>
        <body>
            <h3 style="color: orange;">🚧 <b>Pothole Status: {status}</b></h3>
            <p>A significant pothole has been detected at <b>Camera 4</b>. Repair should be scheduled soon to prevent vehicle damage.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body>
        </html>
        """

    elif status == 'HIGH POTHOLE':
        subject = "🔴 URGENT: Severe Road Hazard Detected"
        body = f"""
        <html>
        <body>
            <h2 style="color: red;">🚨 <b>CRITICAL POTHOLE ALERT</b></h2>
            <p>A deep/dangerous pothole detected at <b>Camera 4</b>. High risk of tire damage or accidents. Immediate intervention required!</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view snapshot</a></p>
        </body>
        </html>
        """
    else:
        return False # No email for 'EMPTY' or unknown status

    # 2. Construct and Send Email
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email Alert Sent: {status}")
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False
#------------------------------------------------------------------------------------------------------------------
#EMAIL ALERT FUNCTION FOR CROWD
def crowd_send_email_alert(status, image_url):

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASS")
    receiver = os.getenv("RECEIVER")

    # Time formatting for the email body
    local_time = datetime.now(ZoneInfo("Asia/Kolkata"))
    formatted_time = local_time.strftime("%B-%d-%Y | %I:%M %p")

    subject = None
    body = None

    if status == 'LOW CROWD':
        subject = "🔵 Crowd Update: Normal Pedestrian Flow"
        body = f"""<html><body>
            <h3 style="color: #2E86C1;">👥 <b>Crowd Status: {status}</b></h3>
            <p>Pedestrian density is currently low at <b>Camera 4</b>. Flow is normal.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body></html>"""

    elif status == 'MODERATE CROWD':
        subject = "🟡 Crowd Alert: Increasing Density"
        body = f"""<html><body>
            <h3 style="color: #D4AC0D;">⚠️ <b>Crowd Status: {status}</b></h3>
            <p>A moderate gathering has been detected at <b>Camera 4</b>.</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view image</a></p>
        </body></html>"""

    elif status == 'HIGH CROWD':
        subject = "🔴 URGENT: Overcrowding Detected"
        body = f"""<html><body>
            <h2 style="color: red;">🚨 <b>CRITICAL CROWD DENSITY ALERT</b></h2>
            <p>High crowd density detected at <b>Camera 4</b>. Immediate management required!</p>
            <p>🕒 <b>Time:</b> {formatted_time}</p>
            <p>🖼️ <b>Evidence:</b> <a href="{image_url}">Click here to view snapshot</a></p>
        </body></html>"""
    else:
        return False # No email for 'EMPTY' or unknown status
    
    # ONLY send the email if a subject and body were actually created
    if subject and body:
        return send_email(subject, body, sender, receiver, password)
    else:
        print(f"No email sent for status: {status}")
        return False

    # 2. Construct and Send Email
def send_email(subject, body, sender, receiver, password):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = receiver
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email Alert Sent: {status}")
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False
#------------------------------------------------------------------------------------------------------------------
# UPLOAD TO S3 AND SAVE TO RDS FUNCTION (DUAL TABLE INSERT)

def upload_to_cloud_and_db(filepath, status, user_inputs,table_name,alert_type,email_function):
    # 1. Upload to S3
    try:
        s3 = boto3.client('s3', 
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_KEY"))
        
        filename = os.path.basename(filepath)
        s3.upload_file(filepath, os.getenv("S3_BUCKET"), filename, ExtraArgs={'ContentType': 'image/jpeg'})
        image_url = f"https://{os.getenv('S3_BUCKET')}.s3.ap-south-1.amazonaws.com/{filename}"
    except Exception as e:
        print(f"S3 Error: {e}")
        return False, False

    # 2. Send Email
    email_sent = email_function(status, image_url)
    # Convert boolean to string "TRUE"/"FALSE" to match your VARCHAR(20) columns
    email_status_str = "TRUE" if email_sent else "FALSE"

    # 3. Save to RDS (Dual Table Insert)
    try:
        conn = pymysql.connect(
            host=os.getenv("RDS_HOST"), 
            user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASS"), 
            database=os.getenv("RDS_NAME")
        )
        
        curr_time = datetime.now()

        with conn.cursor() as cursor:
            # --- Table 1: traffic_logs ---
            # Note: Record_id is AUTO_INCREMENT, so we skip it
            sql1 = f"""INSERT INTO {table_name} 
                     (timestamp, city, area, latitude, longitude, congestion_level, image_url) 
                     VALUES (%s, %s, %s, %s, %s, %s, %s)"""
            
            cursor.execute(sql1, (
                curr_time, 
                user_inputs.get('city'), 
                user_inputs.get('area'), 
                user_inputs.get('lat'), 
                user_inputs.get('lon'), 
                status, 
                image_url
            ))

            # --- Table 2: alerts_log ---
            sql2 = """INSERT INTO alerts_log 
                     (alert_type, generated_at, location, severity, email_sent, resolved) 
                     VALUES (%s, %s, %s, %s, %s, %s)"""
            
            cursor.execute(sql2, (
                alert_type,           # alert_type
                curr_time,           # generated_at
                user_inputs.get('city'), # location
                status,              # severity
                email_status_str,    # email_sent
                "TRUE"               # resolved
            ))

        conn.commit()
        conn.close()
        print("✅ Success: Data stored in traffic_logs and alerts_log")
        return True, email_sent

    except Exception as e:
        print(f"❌ RDS Database Error: {e}")
        return False, email_sent
    


#--------------------------------------------------------------------------------------------------
#INTRO ROUTE

@app.route('/')
def intro2():
    return render_template('intro2.html')

#--------------------------------------------------------------------------------------------------
#TRAFFIC DETECTION ROUTE

@app.route('/traffic', methods=['GET', 'POST'])

def traffic():
    if request.method == 'POST':
        user_inputs = {
            'city': request.form.get('city'),
            'area': request.form.get('area'),
            'lat': request.form.get('lat'),
            'lon': request.form.get('lon')
        }
        
        file = request.files.get('traffic_file')
        
        # Only proceed if a file was actually uploaded
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            #---------------------------------------------------------------------------------------------
            # model from s3
            traffic_model = intelligence_engines.get('traffic_model')
            #--------------------------------------------------------------------------------------------- 
            
            # YOLO Prediction
            results = traffic_model.predict(source=filepath, conf=0.35)
            frame_score = sum([traffic_weights.get(traffic_model.names[int(b.cls[0])], 1.0) for b in results[0].boxes])
            # Determine Traffic Status
            if frame_score == 0:
                status = "NO TRAFFIC"
            elif frame_score < 10:
                status = "LOW TRAFFIC"
            elif frame_score < 20:
                status = "MEDIUM TRAFFIC"
            else:
                status = "HIGH TRAFFIC"
            
            # Calculate mean confidence safely
            conf_val = round(results[0].boxes.conf.mean().item() * 100, 2) if len(results[0].boxes) > 0 else 0

            # Save Annotated Image
            res_filename = 'res_' + file.filename
            res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)
            results[0].save(res_path)

            db_status, email_status = upload_to_cloud_and_db(res_path, status, user_inputs, 'traffic_logs','Traffic',traffic_send_email_alert)

            # Flash Notifications
            if db_status and email_status:
                flash(f"Success: Alert email sent to Goverment@gmail.com & Data stored in RDS  ", "system_sync")
            else:
                flash("⚠️ System Warning: Database or Email sync issue.", "system_error")

            return render_template('traffic.html', 
                                   status=status, 
                                   score=frame_score, 
                                   confidence=conf_val,
                                   result_img='uploads/'+res_filename, 
                                   show_results=True)

    return render_template('traffic.html', score=0, show_results=False)

#------------------------------------------------------------------------------------------------------------------
#ACCIDENT DETECTION ROUTE

@app.route('/accident', methods=['GET', 'POST'])

def accident():
    if request.method == 'POST':
        user_inputs = {
            'city': request.form.get('city'),
            'area': request.form.get('area'),
            'lat': request.form.get('lat'),
            'lon': request.form.get('lon')
        }
        
        file = request.files.get('accident_file')
        
        # Only proceed if a file was actually uploaded
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            #---------------------------------------------------------------------------------------------
            # model from s3
            accident_model = intelligence_engines.get('accident_model')
            #---------------------------------------------------------------------------------------------  
            
            # YOLO Prediction
            results = accident_model.predict(source=filepath, conf=0.35)
            frame_score = sum([accident_weights.get(accident_model.names[int(b.cls[0])], 1.0) for b in results[0].boxes])
            # Determine Accident Status
            if frame_score == 0:
                status = "EMPTY"
            elif frame_score <= 1:
                status = "MODERATE ACCIDENT"
            else:
                status = "SEVERE ACCIDENT"

            # Calculate mean confidence safely
            conf_val = round(results[0].boxes.conf.mean().item() * 100, 2) if len(results[0].boxes) > 0 else 0

            # Save Annotated Image
            res_filename = 'res_' + file.filename
            res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)
            results[0].save(res_path)

            
            db_status, email_status = upload_to_cloud_and_db(res_path, status, user_inputs, 'Accident_logs','Accident',accident_send_email_alert)

            # Flash Notifications
            if db_status and email_status:
                flash(f"Success: Alert email sent to Goverment@gmail.com & Data stored in RDS", "system_sync")
            else:
                flash("⚠️ System Warning: Database or Email sync issue.", "system_error")

            return render_template('accident.html', 
                                   status=status, 
                                   score=frame_score, 
                                   confidence=conf_val,
                                   result_img='uploads/'+res_filename, 
                                   show_results=True)

    return render_template('accident.html', score=0, show_results=False)
#------------------------------------------------------------------------------------------------------------------
#POTHOLE DETECTION ROUTE
@app.route('/pothole', methods=['GET', 'POST'])

def pothole():
    if request.method == 'POST':
        user_inputs = {
            'city': request.form.get('city'),
            'area': request.form.get('area'),
            'lat': request.form.get('lat'),
            'lon': request.form.get('lon')
        }
        
        file = request.files.get('pothole_file')
        
        # Only proceed if a file was actually uploaded
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            #---------------------------------------------------------------------------------------------
            # model from s3
            pothole_model = intelligence_engines.get('pothole_model')
            #---------------------------------------------------------------------------------------------  
            # YOLO Prediction
            results = pothole_model.predict(source=filepath, conf=0.35)
            frame_score = sum([pothole_weights.get(pothole_model.names[int(b.cls[0])], 1.0) for b in results[0].boxes])
            # Determine Pothole Status
            if frame_score == 0: 
                status = "EMPTY"
            elif frame_score < 3: 
                status = "LOW POTHOLE"
            elif frame_score < 6: 
                status = "MEDIUM POTHOLE"
            else:
                status = "HIGH POTHOLE"

            # Calculate mean confidence safely
            conf_val = round(results[0].boxes.conf.mean().item() * 100, 2) if len(results[0].boxes) > 0 else 0

            # Save Annotated Image
            res_filename = 'res_' + file.filename
            res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)
            results[0].save(res_path)

            
            db_status, email_status = upload_to_cloud_and_db(res_path, status, user_inputs, 'pothole_logs','Pothole',pothole_send_email_alert)

            # Flash Notifications
            if db_status and email_status:
                flash(f"Success: Alert email sent to Goverment@gmail.com & Data stored in RDS", "system_sync")
            else:
                flash("⚠️ System Warning: Database or Email sync issue.", "system_error")

            return render_template('pothole.html', 
                                   status=status, 
                                   score=frame_score, 
                                   confidence=conf_val,
                                   result_img='uploads/'+res_filename, 
                                   show_results=True)

    return render_template('pothole.html', score=0, show_results=False)
#------------------------------------------------------------------------------------------------------------------
# CROWD DETECTION ROUTE    

@app.route('/crowd', methods=['GET', 'POST'])
def crowd():
    if request.method == 'POST':
        user_inputs = {
            'city': request.form.get('city'),
            'area': request.form.get('area'),
            'lat': request.form.get('lat'),
            'lon': request.form.get('lon')
        }

        file = request.files.get('crowd_file')

        # Only proceed if a file was actually uploaded
        if file and file.filename != '':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            #---------------------------------------------------------------------------------------------
            # model from s3
            crowd_model = intelligence_engines.get('crowd_model')
            #---------------------------------------------------------------------------------------------                                                                                                     

            # YOLO Prediction
            results = crowd_model.predict(source=filepath, conf=0.35)
            frame_score = sum([crowd_weights.get(crowd_model.names[int(b.cls[0])], 1.0) for b in results[0].boxes])
            # Determine Crowd Status
            if frame_score == 0:
                status = "NO CROWD"
            elif frame_score < 25:
                status = "LOW CROWD"
            elif frame_score < 50:
                status = "MODERATE CROWD"
            else:
                status = "HIGH CROWD"
            
            # Calculate mean confidence safely
            conf_val = round(results[0].boxes.conf.mean().item() * 100, 2) if len(results[0].boxes) > 0 else 0

            # Save Annotated Image
            res_filename = 'res_' + file.filename
            res_path = os.path.join(app.config['UPLOAD_FOLDER'], res_filename)
            annotated_array = results[0].plot(labels=False, conf=False)

            Image.fromarray(annotated_array[..., ::-1]).save(res_path)

            
            db_status, email_status = upload_to_cloud_and_db(res_path, status, user_inputs, 'Crowd_logs','Crowd',crowd_send_email_alert)

            # Flash Notifications
            if db_status and email_status:
                flash(f"Success: Alert email sent to Goverment@gmail.com & Data stored in RDS  ", "system_sync")
            else:
                flash("⚠️ System Warning: Database or Email sync issue.", "system_error")

            return render_template('crowd.html', 
                                   status=status, 
                                   score=frame_score, 
                                   confidence=conf_val,
                                   result_img='uploads/'+res_filename, 
                                   show_results=True)

    return render_template('crowd.html', score=0, show_results=False)

#------------------------------------------------------------------------------------------------------------------
#AIR QUALITY ROUTE

# Local Load AQI Model and Scaler
# with open(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\AQI Model\AQI_Model.pkl', 'rb') as f:
#     model = pickle.load(f)
# with open(r'C:\Users\LOQ\Documents\GUVI DS\Mini-Project\Urban Bot Intelligence F06\Models\AQI Model\AQI_scaler.pkl', 'rb') as f: # Assuming you saved your scaler too
#     scaler = pickle.load(f)

def get_aqi_category(aqi_value):
    if aqi_value <= 50: return "GOOD", "#28a745"
    elif aqi_value <= 100: return "SATISFACTORY", "#ffff00" 
    elif aqi_value <= 200: return "MODERATE", "#ff9900"
    elif aqi_value <= 300: return "POOR", "#ff6600"
    elif aqi_value <= 400: return "VERY POOR", "#ff0000"
    else: return "SEVERE", "#990000"


@app.route('/air_quality', methods=['GET', 'POST'])
def air_quality():
    prediction = None
    status = None
    status_color = None
    input_data = {}

    if request.method == 'POST':
        # Get inputs from form
        input_data = {
            'PM25': float(request.form.get('PM25', 0)),
            'PM10': float(request.form.get('PM10', 0)),
            'NO': float(request.form.get('NO', 0)),
            'NO2': float(request.form.get('NO2', 0)),
            'NOx': float(request.form.get('NOx', 0)),
            'NH3': float(request.form.get('NH3', 0)),
            'CO': float(request.form.get('CO', 0)),
            'SO2': float(request.form.get('SO2', 0)),
            'O3': float(request.form.get('O3', 0)),
            'City': request.form.get('City'),
            'Area': request.form.get('Area')
        }

        # Predict
        features = pd.DataFrame([[input_data['PM25'], input_data['PM10'], input_data['NO'], 
                                 input_data['NO2'], input_data['NOx'], input_data['NH3'], 
                                 input_data['CO'], input_data['SO2'], input_data['O3']]])
        
        #--------------------------------------------------------------------------------------
        #its get model from s3
        scaler = intelligence_engines.get('scaler')
        model = intelligence_engines.get('model')
        #--------------------------------------------------------------------------------------
        
        scaled_data = scaler.transform(features)
        prediction = round(model.predict(scaled_data)[0], 2)
        status, status_color = get_aqi_category(prediction)


        # Log to RDS
        try:
            conn = pymysql.connect(
            host=os.getenv("RDS_HOST"), 
            user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASS"), 
            database=os.getenv("RDS_NAME")
        )
            with conn.cursor() as cursor:
                timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d | %I:%M:%p")
                sql = "INSERT INTO AQI_logs (timestamp, City, Area, PM25, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, AQI, AQI_Status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                cursor.execute(sql, (timestamp, input_data['City'], input_data['Area'], input_data['PM25'], 
                                    input_data['PM10'], input_data['NO'], input_data['NO2'], input_data['NOx'], 
                                    input_data['NH3'], input_data['CO'], input_data['SO2'], input_data['O3'], 
                                    prediction, status))
            conn.commit()
            conn.close()
        except Exception as e: print(f"DB Error: {e}")

    return render_template('air_quality.html', prediction=prediction, status=status, 
                           status_color=status_color, inputs=input_data)

# API for dynamic visuals
@app.route('/api/aqi_history')
def aqi_history():

    conn = pymysql.connect(
        host=os.getenv("RDS_HOST"), 
        user=os.getenv("RDS_USER"),
        password=os.getenv("RDS_PASS"), 
        database=os.getenv("RDS_NAME"),
        cursorclass=pymysql.cursors.DictCursor
    )
    with conn.cursor() as cur:
        cur.execute("SELECT timestamp, AQI, City FROM AQI_logs ORDER BY timestamp DESC LIMIT 20")
        data = cur.fetchall()
    conn.close()
    return jsonify(data)

#--------------------------------------------------------------------------------------------------
#Sentiment Analysis ROUTE
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from email.message import EmailMessage

# Initialize NLTK Sentiment Analyzer
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

from datetime import datetime
from zoneinfo import ZoneInfo

@app.route('/citizen_complaint', methods=['GET', 'POST'])
def citizen_complaint():
    if request.method == 'POST':
        # 1. Collect Data
        city = request.form.get('city')
        category = request.form.get('category')
        priority = request.form.get('priority')
        complaint_text = request.form.get('complaint_text')
        lat_val = request.form.get('latitude') 
        lon_val = request.form.get('longitude')

        # 2. Sentiment Analysis 
        scores = sia.polarity_scores(complaint_text)
        compound = scores['compound']
        if compound >= 0.05:
            sentiment_result = "Positive"
        elif compound <= -0.05:
            sentiment_result = "Negative"
        else:
            sentiment_result = "Neutral"

        # 3. Database Insertion
        try:
            conn = pymysql.connect(
                host=os.getenv("RDS_HOST"), 
                user=os.getenv("RDS_USER"),
                password=os.getenv("RDS_PASS"), 
                database=os.getenv("RDS_NAME")
            )
            with conn.cursor() as cursor:
                # Format timestamp for MySQL DATETIME
                ts = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S")
                
                sql = """INSERT INTO Complaint_logs (timestamp, city, category, lat, lon, 
                         priority, complaint_text, sentiment) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)"""
                
                cursor.execute(sql, (ts, city, category, lat_val, lon_val, priority, complaint_text, sentiment_result))
            
            conn.commit()
            conn.close()
            print("✅ Data stored successfully")

            # 4. Handle Email
            if priority == 'high':
                email_sent = send_alert_email(city, category, complaint_text)
                if email_sent:
                    flash("📧 High priority alert email has been sent to authorities.", "info")
                else:
                    flash("⚠️ Data stored, but email alert failed to send.", "warning")

        except Exception as e:
            print(f"❌ DB Error: {e}")

        return redirect(url_for('citizen_complaint'))

    return render_template('complaint.html')


def send_alert_email(city, category, text):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASS")
    receiver = os.getenv("RECEIVER")

    msg = EmailMessage()
    msg.set_content(f"High Priority Complaint in {city}\nCategory: {category}\nDetails: {text}")
    msg['Subject'] = f"🚨 URGENT: {category} Issue Reported in {city}"
    msg['From'] = sender
    msg['To'] = receiver

    html_content = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #0f172a; line-height: 1.6;">
            <div style="max-width: 600px; margin: auto; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
                <div style="background-color: #ef4444; padding: 20px; text-align: center;">
                    <h1 style="color: white; margin: 0; font-size: 20px; letter-spacing: 1px;">HIGH PRIORITY COMPLAINT</h1>
                </div>
                
                <div style="padding: 30px; background-color: #ffffff;">
                    <p style="font-size: 16px;">The <strong>Urban Data Hive</strong> has detected a critical citizen report requiring immediate attention.</p>
                    
                    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
                        <tr>
                            <td style="padding: 10px 0; color: #64748b; font-weight: 600; width: 100px;">Location:</td>
                            <td style="padding: 10px 0; font-weight: 700;">{city}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px 0; color: #64748b; font-weight: 600;">Category:</td>
                            <td style="padding: 10px 0;">
                                <span style="background-color: #fee2e2; color: #ef4444; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 800; text-transform: uppercase;">
                                    {category}
                                </span>
                            </td>
                        </tr>
                    </table>

                    <div style="background-color: #f8fafc; border-left: 4px solid #ef4444; padding: 15px; margin-top: 10px;">
                        <strong style="display: block; margin-bottom: 5px; color: #1e293b;">Incident Details:</strong>
                        <i style="color: #475569;">"{text}"</i>
                    </div>
                    
                    <div style="margin-top: 30px; text-align: center;">
                        <a href="http://65.0.18.179:5000/citizen_complaint" style="background-color: #4f46e5; color: white; padding: 12px 25px; text-decoration: none; border-radius: 8px; font-weight: 600; font-size: 14px;">View in Dashboard</a>
                    </div>
                </div>

                <div style="background-color: #f1f5f9; padding: 15px; text-align: center; font-size: 12px; color: #94a3b8;">
                    This is an automated alert from <strong>UrbanBot.AI</strong> Intelligence Hub.
                </div>
            </div>
        </body>
        </html>
        """

# Set the HTML content
    msg.add_alternative(html_content, subtype='html')
  
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"✅ Email Alert Sent: {status}")
        return True
    except Exception as e:
        print(f"❌ Email Error: {e}")
        return False
    


@app.route('/api/complaint_history')
def complaint_history():
    try:
        conn = pymysql.connect(
            host=os.getenv("RDS_HOST"), user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASS"), database=os.getenv("RDS_NAME"),
            cursorclass=pymysql.cursors.DictCursor
        )
        with conn.cursor() as cursor:
            
            sql = "SELECT category, sentiment, priority FROM Complaint_logs ORDER BY Complaint_id DESC LIMIT 20"
            cursor.execute(sql)
            data = cursor.fetchall()
        conn.close()
        
        
        print(f"📡 API sending data: {data}") 
        
        return jsonify(data)
    except Exception as e:
        print(f"❌ API Error: {e}")
        return jsonify([])

#--------------------------------------------------------------------------------------------------
# #AI Agent ROUTE

# --- CONFIGURATION ---
api_key = os.getenv('GROQRY_API_KEY')
db_url = "mysql+pymysql://admin:rootpassword@traffic-db-01.c92eus6qgtdp.ap-south-1.rds.amazonaws.com:3306/log_tables"

# --- 1. REPORT AGENT (The SQL Specialist) ---
report_agent = Agent(
    name="reporter",
    model=Groq(id="openai/gpt-oss-120b", api_key=api_key),
    tools=[SQLTools(db_url=db_url)],
    instructions=[
        "You are an expert SQL analyst.",
        "📊 TABLES: traffic_logs, pothole_logs, Accident_logs, Crowd_logs, AQI_logs, Complaint_logs, alerts_log",
        # --- FIX FOR SQL LIMIT ISSUE ---
        "⚠️ SQL RULE: Always write the LIMIT directly inside the SQL query string.",
        "When calling 'run_sql_query', leave the 'limit' parameter as null or empty.",
        "Ensure all parameters match the expected schema (Numbers must not be strings).",
        # ------------------------------------

        "🔍 PROCESS: 1. Identify table | 2. describe_table | 3. SELECT with LIMIT 10.",
        "🎯 ALWAYS provide specific numbers and statistics from the query results."
    ],
)
# --- 1. UPDATED SMTP FUNCTION WITH CONVERSION ---
def send_urban_report(subject: str, body: str):
    sender_email = os.getenv('EMAIL_SENDER')
    sender_password = os.getenv('EMAIL_PASS')
    receiver_email = os.getenv('RECEIVER')
    
    # NEW: Convert the AI's Markdown table into a real HTML table
    # We use extensions=['extra'] to support the pipe-style tables (| --- |)
    formatted_body_html = markdown.markdown(body, extensions=['extra', 'tables'])

    html_content = f"""
    <html>
    <head>
        <style>
            .container {{ font-family: 'Segoe UI', Arial, sans-serif; color: #333; max-width: 600px; border: 1px solid #eee; }}
            .header {{ background-color: #FFC107; padding: 20px; text-align: center; color: #000; }}
            .content {{ padding: 20px; }}
            /* Style the converted Markdown tables */
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; border: 1px solid #ddd; }}
            th {{ background-color: #333; color: white; padding: 12px; text-align: left; }}
            td {{ border: 1px solid #ddd; padding: 10px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .footer {{ font-size: 12px; color: #777; padding: 20px; text-align: center; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header"><h2>Urban Intelligence Report 🐝</h2></div>
            <div class="content">
                <p>Dear Team, here is the assessment from the <b>Urban Data Hive</b>:</p>
                {formatted_body_html}  </div>
            <div class="footer">Sincerely,<br><b>The Hive Manager</b></div>
        </div>
    </body>
    </html>
    """

    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = sender_email, receiver_email, subject
        msg.attach(MIMEText(html_content, 'html'))
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return "✅ Attractive HTML Email sent successfully!"
    except Exception as e:
        return f"❌ SMTP Error: {str(e)}"
    
# --- 2. UPDATED EMAIL AGENT ---
email_agent = Agent(
    name="email_sender",
    model=Groq(id="openai/gpt-oss-120b", api_key=api_key),
    tools=[send_urban_report], 
    instructions=[
        "You are the Hive's Mail Carrier.",
        "When the Team Agent gives you data, use the 'send_urban_report' tool.",
        "The subject should be 'Urban Intelligence Report'.",
        "The body MUST contain the actual data found in the database.",
        "Confirm to the Team Agent once 'send_urban_report' returns a success message."
    ],  
)

# --- 3. TEAM AGENT (The Polite Urban Bee Manager) ---
team_agent = Agent(
    team=[report_agent, email_agent],
    model=Groq(id="openai/gpt-oss-120b", api_key=api_key),
    instructions=[
        "PERSONALITY: You are 'Urban Bee', a polite, kind  urban data manager.",
        
        # --- FIX FOR 400 ERROR ---
        "⚠️ CRITICAL EXECUTION RULE: Execute tools SEQUENTIALLY. Never call more than one tool in a single turn.",
        "Wait for the output of the 'reporter' before even thinking about the 'email_sender'.",
        # -------------------------

        "1. WORKFLOW ORDER: Whenever a request involves data and an email:",
        "   - STEP 1: Always call the 'reporter' first to get the data.",
        "   - STEP 2: Wait for the 'reporter' to finish. If no data is found, stop and tell the user kindly.",
        "   - STEP 3: If data is found, provide a summary in the chat using professional Markdown.",
        "   - STEP 4: Only THEN call the 'email_sender' to send the report if requested.",
        
        "2. FORMATTING STYLE:",
        "   - Use ## for Headings (e.g., ## 🚨 Pothole Summary)",
        "   - Use **Bold** for numbers and > for insights.",
        "   - Use Tables for data logs.",
        
        "3. OFF-TOPIC GUARDRAILS: For health or other topics, reply: 'I apologize, but my wings only fly toward city data! How about a traffic report instead?'",
        
        "4. NO ERRORS: Never mention JSON, 400 errors, or tool names. If something fails, say: 'Bzzz! It seems the data hive is a bit crowded right now. Could you try again in a moment?'"
    ],
    
    show_tool_calls=False,
    markdown=True,
)

# --- GREETING LOGIC ---
def is_greeting(user_input):
    # Convert to lowercase and split into individual words
    input_words = user_input.lower().strip().split()
    greetings = {'hi', 'hello', 'hey', 'greetings', 'morning', 'afternoon', 'evening'}
    
    # Only return True if the message is SHORT and contains ONLY a greeting
    # This prevents "High traffic" from being caught.
    if len(input_words) <= 3:
        return any(word in greetings for word in input_words)
    return False

@app.route('/chat_ai')
def chat_page():
    return render_template('AI Agent.html')

@app.route('/chat', methods=['POST'])
def chat_logic():
    user_input = request.json.get("message", "").strip()
    
    if not user_input:
        return jsonify({"response": "I didn't catch that. How can I help you today?"})

    if is_greeting(user_input):
        return jsonify({"response": "🐝 Bzzz! Hello! I'm the Urban Bee. I can help you with city reports or email analytics. What's on your mind?"})

    try:
        # Run the coordinator
        run_response = team_agent.run(user_input)
        return jsonify({"response": run_response.content})
    
    except Exception as e:
    # This will print the REAL error in your terminal/cmd
        print(f"CRITICAL DATABASE ERROR: {str(e)}") 
        return jsonify({"response": f"🐝 Bzzz! I'm struggling to reach the data hive. (Error: {str(e)})"})
    


#------------------------------------------------------------------------------------------------------------------
#DASHBOARD ROUTE
from flask import Flask, render_template, jsonify
from sqlalchemy import create_engine, text
import pandas as pd
import json



# UPDATED DB CONNECTION STRING
DB_URL = "mysql+pymysql://admin:rootpassword@traffic-db-01.c92eus6qgtdp.ap-south-1.rds.amazonaws.com:3306/log_tables"
engine = create_engine(DB_URL)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/full_dashboard_data')
def get_full_data():
    data_packet = {}
    
    

    table_configs = [
        {"name": "traffic_logs", "col": "congestion_level", "time": "timestamp"},     
        {"name": "pothole_logs", "col": "congestion_level", "time": "timestamp"}, 
        {"name": "Accident_logs", "col": "area", "time": "timestamp"},
        {"name": "Crowd_logs", "col": "area", "time": "timestamp"},
        {"name": "AQI_logs", "col": "AQI_Status", "time": "timestamp", "metric": "AQI"},
        {"name": "Complaint_logs", "col": "sentiment", "time": "timestamp"},
        {"name": "alerts_log", "col": "severity", "time": "generated_at"}
    ]

    try:
        with engine.connect() as conn:
            res_alerts = conn.execute(text("SELECT COUNT(*) FROM alerts_log WHERE resolved='TRUE'")).scalar()
            
            # 2. Average City AQI (Rounded)
            res_aqi = conn.execute(text("SELECT AVG(AQI) FROM AQI_logs")).scalar()
            
            # 3. Critical Traffic Zones (High Congestion Count)
            res_traffic = conn.execute(text("SELECT COUNT(*) FROM traffic_logs WHERE congestion_level='HIGH TRAFFIC'")).scalar()
            
            # 4. Citizen Sentiment Score (% of Positive Complaints)
            total_complaints = conn.execute(text("SELECT COUNT(*) FROM Complaint_logs")).scalar()
            pos_complaints = conn.execute(text("SELECT COUNT(*) FROM Complaint_logs WHERE sentiment='Positive'")).scalar()
            sentiment_score = int((pos_complaints / total_complaints * 100)) if total_complaints > 0 else 0

            summary_metrics = {
                "active_alerts": res_alerts or 0,
                "avg_aqi": int(res_aqi) if res_aqi else 0,
                "critical_traffic": res_traffic or 0,
                "sentiment_score": sentiment_score
            }

            

            # ADD SUMMARY TO RESPONSE
            data_packet["summary"] = summary_metrics 



            for t in table_configs:
                table = t["name"]

                # 1. Fetch Metric Counts (for Charts)
                
                if table == "AQI_logs":
                    query_chart = text(f"SELECT AQI_Status, AVG(AQI) FROM {table} GROUP BY AQI_Status")
                else:
                    query_chart = text(f"SELECT {t['col']}, COUNT(*) FROM {table} GROUP BY {t['col']}")
                
                chart_data = conn.execute(query_chart).fetchall()
                
                # 2. Fetch Raw Data (Last 5 rows for Table)
                query_rows = text(f"SELECT * FROM {table} ORDER BY {t['time']} DESC LIMIT 5")
                raw_rows = conn.execute(query_rows).fetchall()
                
                # 3. Serialize (Convert Row objects to dicts and Dates to Strings)
                chart_labels = [str(row[0]) for row in chart_data]
                chart_values = [float(row[1]) for row in chart_data]
                
                table_rows = []
                for row in raw_rows:
                    row_dict = dict(row._mapping)
                    # Convert all datetime objects to string to prevent JSON errors
                    for k, v in row_dict.items():
                        if hasattr(v, 'strftime'):
                            row_dict[k] = v.strftime('%Y-%m-%d %H:%M')
                    table_rows.append(row_dict)

                data_packet[table] = {
                    "labels": chart_labels,
                    "values": chart_values,
                    "rows": table_rows,
                    "total": sum(chart_values) if table != "AQI_logs" else len(chart_values)
                }

        return jsonify(data_packet)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

#------------------------------------------------------------------------------------------------------------------
#ABOUT ROUTE

# --- NEW ABOUT ME ROUTE ---
@app.route('/about')
def about_me():
    
    return render_template('about me.html')


# if __name__ == '__main__':
#     app.run(debug=True)

#Running on Port 5000
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)




