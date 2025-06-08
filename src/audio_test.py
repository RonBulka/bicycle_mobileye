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
    file_path = "audio/Rick-Roll-Sound-Effect.mp3"
    if not os.path.exists(file_path):
        print(f"Audio file {file_path} not found.")
        return
    try:
        # Initialize pygame mixer
        pygame.mixer.init()

        # Load and play the audio file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            time.sleep(1)

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