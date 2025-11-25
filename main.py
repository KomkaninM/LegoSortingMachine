import RPi.GPIO as GPIO
import time

# from Motor_Control import DCMotorController
from LegoClassificationUpdate import detect_lego_from_paths
from NETPIE import publish_to_netpie

from picamera2 import Picamera2
from datetime import datetime
from pathlib import Path    
import time
import pigpio

# --- GLOBAL CAMERA OBJECT ---
picam2 = None

GPIO.setmode(GPIO.BCM)

IN1 = 5
IN2 = 6
ENA = 12
servo_pin = 13

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# --- MOTOR PWM ---
pwm_motor = GPIO.PWM(ENA, 1000)   # 1 kHz for DC motor
pwm_motor.start(0)

# --- SERVO PWM (if you still want to use RPi.GPIO for servo) ---
GPIO.setup(servo_pin, GPIO.OUT)
pwm_servo = GPIO.PWM(servo_pin, 50)   # 50 Hz servo
pwm_servo.start(0)

def init_camera():
    """
    Create and start Picamera2 once.
    """
    global picam2
    if picam2 is not None:
        return  # already initialised

    print("Initializing Camera (one-time)...")
    picam2 = Picamera2()
    config = picam2.create_still_configuration()
    picam2.configure(config)
    picam2.start()
    # small warm-up
    time.sleep(2)


# ---------- HELPER FUNCTIONS ----------
def angle_to_duty(angle):
    """
    Convert angle in degrees (0–180) to duty cycle.
    You may need to fine-tune 2 and 12 for your servo.
    """
    return 2 + (angle / 18.0)   # 0° -> ~2%, 180° -> ~12%



# GPIO Pin Definitions
VIB1_PIN = 19                # Vibration motor 1 (slider 1)
VIB2_PIN = 16                # Vibration motor 2 (slider 2)
IR_PIN = 26

#Variable
sensor_delay = 0.03

GPIO.setmode(GPIO.BCM)

GPIO.setup(VIB1_PIN, GPIO.OUT)
GPIO.setup(VIB2_PIN, GPIO.OUT)
GPIO.setup(IR_PIN, GPIO.IN)

def set_vibration_motor(state) :
    if state == True :
        GPIO.output(VIB1_PIN, GPIO.HIGH)
        GPIO.output(VIB2_PIN, GPIO.HIGH)
    else :
        GPIO.output(VIB1_PIN, GPIO.LOW)
        GPIO.output(VIB2_PIN, GPIO.LOW)

def vibration_test() :
    print("Vibration Motor Test.")
    for i in range(5):
        print("Motor ON")
        # Turn the motor on (HIGH)
        set_vibration_motor(True)
        # Wait for 5 second
        time.sleep(5)
        
        print("Motor OFF")
        # Turn the motor off (LOW)
        set_vibration_motor(False)
        # Wait for 1 second
        time.sleep(1)

def is_object() :
    value = GPIO.input(IR_PIN)  # digitalRead equivalent
    if value == 0:
        print("Object detected (LOW)")
        return True
    else:
        # print("Clear (HIGH)")
        return False

def test_ir() :
    print("Reading IR sensor")
    for i in range(5):
        object_found = is_object()
        if object_found:
            print("Object detected (LOW)")
        else:
            print("Clear (HIGH)")
        time.sleep(1)

def capture_photo():
    """
    Capture a photo with the already-started camera and return the file path.
    """
    global picam2

    # Make sure camera is initialised
    init_camera()

    # 1. Setup Directory
    save_dir = Path.home() / "Pictures"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. Generate Filename
    filename = save_dir / f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    file_path_str = str(filename)

    try:
        print(f"Capturing to: {file_path_str}")
        picam2.capture_file(file_path_str)
        print(f"Image saved: {file_path_str}")
        return file_path_str

    except Exception as e:
        print(f"Camera Error during capture: {e}")
        return None

def set_angle(angle):
    if not pi.connected:
        print("Please run 'sudo pigpiod' in terminal first!")
        return 0

    # Map angle (0-180) to pulse width (500-2500)
    # 0 deg = 500, 180 deg = 2500
    pulse_width = 500 + (angle * 2000 / 180)
    
    # Send the precise pulse to the servo
    pi.set_servo_pulsewidth(servo_pin, pulse_width)

def run_servo() :
    set_angle(0)
    time.sleep(1)
    for angle in range(0, 150, 5): # Smaller steps = smoother
            set_angle(angle)
            time.sleep(0.05)
            
    # Scan down
    for angle in range(150, -1, -5):
            set_angle(angle)
            time.sleep(0.05)

def run_full_system_check():
    print("\n" + "="*40)
    print("STARTING FULL SYSTEM DIAGNOSTIC")
    print("="*40 + "\n")

    # --- 1. Vibration Sensor ---
    print("[STEP 1] Testing Vibration Sensor...")
    vibration_test()
    print("   ✅ Step 1 Complete.\n")
    time.sleep(1)

    # --- 2. IR Sensor ---
    print("[STEP 2] Testing IR Sensor...")
    test_ir()
    print("   ✅ Step 2 Complete.\n")
    time.sleep(1)

    # --- 3. Motor Initialization ---
    print("[STEP 3] Initializing Motor Controller...")
    try:
        motor = DCMotorController()
        print("   Motor object created successfully.")
    except Exception as e:
        print(f"   Error: {e}")
        return
    print("   Step 3 Complete.\n")
    time.sleep(1)

    # --- 4. Encoder Test ---
    print(" [STEP 4] Testing Encoder Feedback...")
    print("   manually rotate the motor wheel gently within 5 seconds.")
    
    # We implement a timeout here so the test doesn't get stuck forever
    start_pos = motor.get_position()
    timeout = time.time() + 5
    encoder_passed = False
    
    while time.time() < timeout:
        current_pos = motor.get_position()
        if current_pos != start_pos:
            print(f"   -> Detected movement! Position is now: {current_pos}")
            encoder_passed = True
            break
        time.sleep(0.1)
        
    if encoder_passed:
        print("   Encoder is working.")
    else:
        print("    Warning: No movement detected (or encoder disconnected).")
    print("    Step 4 Complete.\n")
    time.sleep(1)

    # --- 5. Motor Movement Test ---
    print("[STEP 5] Testing DC Motor Drive...")
    print("   Watch out! Motor will spin CW then CCW.")
    motor.test_dc_motor()
    print("    Step 5 Complete.\n")
    time.sleep(1)

    # --- 6. Origin Reset ---
    print(" [STEP 6] Resetting Origin...")
    motor.reset_origin()
    print(f"   Current Position set to: {motor.get_position()}")
    print("    Step 6 Complete.\n")
    time.sleep(1)

    # --- 7. Position Control ---
    target = 30
    print(f" [STEP 7] Moving to Position {target}...")
    motor.set_position(target)
    print("    Step 7 Complete.\n")
    time.sleep(1)

    # --- 8. Camera Capture ---
    print(" [STEP 8] Testing Camera Capture...")
    print("   Warming up and taking a picture...")
    path = capture_photo()
    
    if path:
        print(f"   Saved to: {path}")
    else:
        print("    Camera Failed.")
    print("    Step 8 Complete.\n")

    print("="*40)
    print(" ALL TESTS FINISHED")
    print("="*40)
    
    motor.stop()

"""
    Usable Functions
        vibration_test()
        test_ir()
        Initialize the controller
        motor = DCMotorController()
        motor.test_encoder_read()
        motor.test_dc_motor()
        motor.reset_origin()
        motor.set_position(30)
        capture_photo()
"""

def main() :
    # STATE_FEED          // เปิดมอเตอร์สั่น ปล่อยเลโก้มาที่จุดตรวจ
    # STATE_STABILIZE     // เลโก้มาถึงแล้ว รอให้มันหยุดไหว
    # STATE_CAPTURE       // ถ่ายรูป
    # STATE_CLASSIFY      // วิเคราะห์สี+ขนาด
    # STATE_ROTATE        // หมุน DC motor ไปช่องที่ถูกต้อง
    # STATE_PUSH          // Servo ผลักเลโก้ลงกล่อง
    # STATE_WAIT_DROP     // รอให้เลโก้ร่วงลงไปจริง ๆ
    # STATE_PUBLISH       // ส่งข้อมูลขึ้น NETPIE
    # STATE_ERROR         // กรณีเกิน timeout หรือมีปัญหา

    # Initialization
    state = "STATE_FEED"
    color = ""
    path = ""
    red_sort = 0
    blue_sort = 0
    grey_sort = 0
    white_sort = 0
    total_sort = 0
    success_sort = 0
    #motor = DCMotorController()
    pi = pigpio.pi()
    set_angle(0)

    while True :
        if state == "STATE_FEED" :
            print("STATE_FEED")
            set_vibration_motor(True)
            time.sleep(sensor_delay)
            if is_object() :
                set_vibration_motor(False)
                state = "STATE_CAPTURE"

        if state == "STATE_CAPTURE" :
            time.sleep(2)
            print("STATE_CAPTURE")
            path = capture_photo()
            state = "STATE_CLASSIFY"

        if state == "STATE_CLASSIFY" :
            print("STATE_CLASSIFY")
            bg_path = " /home/pi/Pictures/photo_20251125_193159.jpg"
            img_path = path
            threshold = 30
            output_dir = "/home/pi/LegoSorter/Sorted-Output"

            color, bbox = detect_lego_from_paths(
                bg_path= bg_path,
                img_path=img_path,
                threshold=threshold,
                output_dir=output_dir,
                roi_x_min=0,   # or None if you want full image
                save=True,
                show=False,
            )

            print("Detected color:", color)
            if color == "red":
                print("Forward")
            elif color == "blue":
                print("bl")
            else:
                pass
        
            if color == "blue":
                print("Forward")
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(50)
                time.sleep(0.1)
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(0)
                time.sleep(0.2)
                success_sort += 1
                red_sort += 1
            elif color == "red":
                GPIO.output(IN1, 0)
                GPIO.output(IN2, 1)
                pwm_motor.ChangeDutyCycle(50)
                time.sleep(0.1)
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(0)
                time.sleep(0.2)
                success_sort += 1
                blue_sort += 1
            else:
                print("Unknown")
                #no move
            size = bbox[2] * bbox[3] if bbox is not None else 0
            conf = 0

            if bbox is not None:
                x, y, w, h = bbox

                x1 = x
                y1 = y
                x2 = x + w
                y2 = y + h

                print("bbox (x, y, w, h):", bbox)
                print("x1, y1, x2, y2:", x1, y1, x2, y2)
#             color = result["color"]
#             size = result["size"]
#             conf = result["confidence"]
            publish_to_netpie("STATE_CLASSIFY", color, path, {
            "size": size,
            "confidence": conf})

            state = "Motor_Rotate"

        if state == "Motor_Rotate" :
            print("Motor_Rotate")
            state = "STATE_PUSH"

        if state == "STATE_PUSH" :
            print("STATE_PUSH")
            time.sleep(0.3)
            # Push the lego down
            time.sleep(1)
            run_servo()
            # Push andWait
            print(f"Push color :{color}")
            if color == "red":
                print("Forward")
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(20)
                time.sleep(0.1)
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(0)
                time.sleep(0.2)
            elif color == "blue":
                GPIO.output(IN1, 0)
                GPIO.output(IN2, 1)
                pwm_motor.ChangeDutyCycle(20)
                time.sleep(0.1)
                GPIO.output(IN1, 1)
                GPIO.output(IN2, 0)
                pwm_motor.ChangeDutyCycle(0)
                time.sleep(0.2)
            else:
                print("Unknown")
                #no move
            time.sleep(0.5)
            state = "PUBLISH"
            #state = "STATE_FEED"

        if state == "PUBLISH" :
            total_sort += 1
            print("STATE_PUBLISH")
            counters = {
                "total_sort": total_sort,
                "success_sort": success_sort,
                "red_sort": red_sort,
                "blue_sort": blue_sort,
                "grey_sort": grey_sort,
                "white_sort": white_sort
                }

            publish_to_netpie(
                state="STATE_PUBLISH",
                color=color,
                image_path=path,
                counters=counters,
                extra={
                    "size": size,
                    "confidence": conf
                }
            )
            state = "STATE_FEED"

if __name__ == "__main__" :
    
    pi = pigpio.pi()
    main()
#    run_full_system_check()
