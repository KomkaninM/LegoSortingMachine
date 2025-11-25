from gpiozero import DigitalInputDevice, PWMOutputDevice
import time

# --- CONFIGURATION ---
# Pins
PIN_ENC_A = 17
PIN_ENC_B = 27
PIN_IN1 = 6
PIN_IN2 = 5
PIN_ENA = 12

# Speed Constraints
MAX_SPEED = 0.15    # Max 20%
MIN_PWM = 0.08     # Min 8% to start moving

# Encoder direction flip (+1 or -1)
# If sign is still wrong, change this to +1
ENCODER_DIR = -1   # we set -1 so that "negative move" gives decreasing pos


class DCMotorController:
    def __init__(self):
        # Initialize Motor Pins
        self.in1 = PWMOutputDevice(PIN_IN1)
        self.in2 = PWMOutputDevice(PIN_IN2)
        self.ena = PWMOutputDevice(PIN_ENA)
        self.ena.value = 1  # Enable always on

        # Initialize Encoder Pins
        self.encoder_a = DigitalInputDevice(PIN_ENC_A, pull_up=True)
        self.encoder_b = DigitalInputDevice(PIN_ENC_B, pull_up=True)

        # Internal State
        self.pos = 0

        # Setup Interrupts (trigger on rising edge of A)
        self.encoder_a.when_activated = self._pulse_callback

    def _pulse_callback(self):
        """Internal function called automatically by hardware interrupt."""
        # Quadrature decoding: direction depends on B state when A rises
        if self.encoder_b.value:
            delta = 1
        else:
            delta = -1

        self.pos += ENCODER_DIR * delta

    def _set_raw_speed(self, speed):
        """Internal function to apply speed limits and write to hardware."""
        # 1. Clamp to MAX
        speed = max(min(speed, MAX_SPEED), -MAX_SPEED)

        # 2. Apply Deadband (Min Speed) if non-zero
        if speed > 0 and speed < MIN_PWM:
            speed = MIN_PWM
        elif speed < 0 and speed > -MIN_PWM:
            speed = -MIN_PWM

        # 3. Write to Pins
        if speed > 0:
            self.in1.value = speed
            self.in2.value = 0
        elif speed < 0:
            self.in1.value = 0
            self.in2.value = abs(speed)
        else:
            self.in1.value = 0
            self.in2.value = 0

    # --- PUBLIC FUNCTIONS ---

    def reset_origin(self):
        """Sets the current physical location as '0'."""
        self.pos = 0
        print("Origin reset to 0.")

    def get_position(self):
        return self.pos

    def stop(self):
        self._set_raw_speed(0)

    def test_dc_motor(self):
        """Move CW 1s, CCW 1s."""
        print("Testing Motor: CW for 1s")
        self._set_raw_speed(MAX_SPEED)  # Move forward
        time.sleep(1)

        print("Testing Motor: Stop for 0.5s")
        self._set_raw_speed(0)
        time.sleep(0.5)

        print("Testing Motor: CCW for 1s")
        self._set_raw_speed(-MAX_SPEED)  # Move backward
        time.sleep(1)

        self._set_raw_speed(0)
        print("Test Complete.")

    def test_encoder_read(self):
        """Print encoder position when changed."""
        print("Rotate motor manually to see values... (Ctrl+C to exit)")
        last_pos = self.pos
        try:
            while True:
                current_pos = self.pos
                if current_pos != last_pos:
                    print(f"Encoder Changed: {current_pos}")
                    last_pos = current_pos
                time.sleep(0.001)
        except KeyboardInterrupt:
            print("\nEncoder Test Ended.")

    # --------- NEW POSITION CONTROL (with fine adjust) ---------

    def set_position(self, target):
        """
        Move to 'target' encoder counts:

        1) Coarse move using proportional control.
        2) Fine-adjust step-by-step until encoder == target.
        """
        # --- Coarse move phase ---
        Kp = 0.01            # proportional gain
        COARSE_EPS = 3       # when |error| <= this, switch to fine phase
        MAX_COARSE_TIME = 5  # safety timeout (seconds)

        print(f"\n--- Moving to target {target} ---")
        start_time = time.time()

        while True:
            error = target - self.pos
            print(f"[COARSE] pos: {self.pos}, error: {error}")

            # If close enough, stop coarse phase
            if abs(error) <= COARSE_EPS:
                break

            # Safety timeout
            if time.time() - start_time > MAX_COARSE_TIME:
                print("Coarse phase timeout reached.")
                break

            speed = Kp * error
            self._set_raw_speed(speed)
            time.sleep(0.01)

        # Stop motor before fine phase
        self._set_raw_speed(0)

        # --- Fine adjust phase (tick-by-tick) ---
        print("Entering fine adjust phase...")

        MAX_FINE_STEPS = 40  # avoid infinite loop if something wrong

        for step in range(MAX_FINE_STEPS):
            error = target - self.pos
            print(f"[FINE] step {step}, pos: {self.pos}, error: {error}")

            if error == 0:
                print(f"Fine adjust done, exact target reached: {self.pos}")
                break

            direction = 1 if error > 0 else -1
            pulse_speed = direction * MIN_PWM

            # Short pulse in correct direction
            self._set_raw_speed(pulse_speed)

            # Wait until encoder changes or small timeout
            start_pos = self.pos
            t0 = time.time()
            while self.pos == start_pos and (time.time() - t0) < 0.1:
                time.sleep(0.001)

            # Stop after the tick
            self._set_raw_speed(0)
            time.sleep(0.02)  # small settle time

        print(f"Target {target}, final pos {self.pos}")

    # --- Test function in range -30 to +30 ---

    def test_position_range_30(self):
        """
        Test motion within [-30, 30]:
        - go to -30
        - return to 0
        - go to +30
        - return to 0
        """
        targets = [-30, 0, 30, 0]
        for t in targets:
            print("\n==============================") 
            print(f"Moving to target: {t}")
            self.set_position(t)
            print(f"Holding at {self.pos}")
            time.sleep(0.5)  # small pause at each point


# --- MAIN TEST CODE ---

if __name__ == "__main__":
    controller = DCMotorController()
    try:
        controller.reset_origin()
        print("Starting position-range test (-30 to +30)...")
        controller.test_position_range_30()
        print("Position-range test completed.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Stopping motor.")
    finally:
        controller.stop()
        print("Motor stopped. Program ended.")
