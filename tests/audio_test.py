"""
Audio Testing Script

This script provides audio testing capabilities for the vehicle warning system, including
tone generation and audio file playback. It's used for testing audio output functionality.

Key Features:
- Sine wave tone generation at specified frequencies
- Melody playback (Twinkle Twinkle Little Star)
- Audio file playback with looping
- Volume control for testing
- Pygame mixer integration
- Error handling for audio device issues

Audio Functions:
- generate_tone(): Creates sine wave tones at specific frequencies
- play_melody(): Plays predefined melody sequences
- play_audio(): Loops audio file playback for testing

Usage:
    python audio_test.py

This script is used for:
- Testing audio hardware functionality
- Verifying warning system audio output
- Calibrating audio volume levels
- Debugging audio-related issues

The script supports both generated tones and pre-recorded audio files
for comprehensive audio system testing.
"""

#!/usr/bin/env python
import pygame
import time
import os
import numpy as np

def generate_tone(frequency, duration, volume=0.5):
    """Generate a sine wave tone at the specified frequency"""
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate mono tone
    tone = np.sin(frequency * t * 2 * np.pi)
    # Convert to stereo by duplicating the channel
    stereo_tone = np.vstack((tone, tone)).T
    # Scale to 16-bit integer range and convert to int16
    stereo_tone = (stereo_tone * volume * 32767).astype(np.int16)
    # Ensure array is C-contiguous
    stereo_tone = np.ascontiguousarray(stereo_tone)
    return pygame.sndarray.make_sound(stereo_tone)

def play_melody(volume=0.5):
    """Play a melody using generated tones"""
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Define the melody (Twinkle Twinkle Little Star)
        melody = [
            (440, 0.5), (440, 0.5), (523, 0.5), (523, 0.5),  # C C G G
            (587, 0.5), (587, 0.5), (523, 1),               # A A G
            (494, 0.5), (494, 0.5), (440, 0.5), (440, 0.5),  # F F E E
            (523, 0.5), (523, 0.5), (440, 1)                # G G C
        ]

        print(f"Playing melody at volume {volume}...")
        for frequency, duration in melody:
            tone = generate_tone(frequency, duration, volume)
            tone.play()
            time.sleep(duration + 0.07)  # Add small pause between notes

    except Exception as e:
        print(f"Error playing melody: {e}")
    finally:
        pygame.mixer.quit()

def play_audio():
    """Play audio through the default audio device"""
    file_path = "audio/clock-alarm-8762.mp3"
    if not os.path.exists(file_path):
        print(f"Audio file {file_path} not found.")
        return
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the audio file repeatly
        pygame.mixer.music.load(file_path)
        while True:
            pygame.mixer.music.play()
            time.sleep(0.8)

            # Wait for the audio to finish playing
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0.44)

    except Exception as e:
        print(f"Error playing audio: {e}")
    finally:
        pygame.mixer.quit()

def main():
    # Play the melody at different volumes
    # play_melody(volume=0.10)

    play_audio()

if __name__ == "__main__":
    main() 