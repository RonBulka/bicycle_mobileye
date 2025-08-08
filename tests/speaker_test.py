"""
Raspberry Pi Speaker Testing Script

This script tests speaker functionality on Raspberry Pi hardware using GPIO PWM output.
It provides tone generation and melody playback for hardware audio testing.

Key Features:
- GPIO PWM-based tone generation
- Musical scale and melody playback
- Raspberry Pi hardware integration
- Frequency control for different notes
- Duration control for musical timing
- GPIO cleanup and resource management

Hardware Requirements:
- Raspberry Pi with GPIO access
- Speaker connected to GPIO18 (PWM0)
- RPi.GPIO library installation

Audio Functions:
- play_tone(): Generates tones at specific frequencies
- play_scale(): Plays musical scale for testing
- Melody playback with predefined sequences

Usage:
    python speaker_test.py

This script is used for:
- Testing Raspberry Pi speaker hardware
- Verifying GPIO PWM functionality
- Calibrating audio output levels
- Hardware integration testing

The script provides a complete hardware audio testing solution for
Raspberry Pi-based vehicle warning systems.
"""

import RPi.GPIO as GPIO
import time
import math

# Set up GPIO
GPIO.setmode(GPIO.BCM)
SPEAKER_PIN = 18  # Using GPIO18 (PWM0)
GPIO.setup(SPEAKER_PIN, GPIO.OUT)

# Create PWM instance
pwm = GPIO.PWM(SPEAKER_PIN, 440)  # Start with 440 Hz (A4 note)

def play_tone(frequency, duration):
    """
    Play a tone at the specified frequency for the given duration
    """
    pwm.ChangeFrequency(frequency)
    pwm.start(50)  # 50% duty cycle
    time.sleep(duration)
    pwm.stop()

def play_scale():
    """
    Play a simple scale to test the speaker
    """
    notes = [440, 494, 523, 587, 659, 698, 784, 880]  # A4 to A5
    for note in notes:
        play_tone(note, 0.5)
        time.sleep(0.1)

try:
    print("Starting speaker test...")
    print("Playing a scale...")
    play_scale()
    
    print("Playing a simple melody...")
    # Play a simple melody (Twinkle Twinkle Little Star)
    melody = [
        (440, 0.5), (440, 0.5), (523, 0.5), (523, 0.5),  # C C G G
        (587, 0.5), (587, 0.5), (523, 1),               # A A G
        (494, 0.5), (494, 0.5), (440, 0.5), (440, 0.5),  # F F E E
        (523, 0.5), (523, 0.5), (440, 1)                # G G C
    ]
    
    for note, duration in melody:
        play_tone(note, duration)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nTest stopped by user")
finally:
    GPIO.cleanup()
    print("GPIO cleanup completed") 