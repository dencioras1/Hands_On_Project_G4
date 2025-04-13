import time as Time
import os
import cv2
import numpy as np
import pygame
from support_functions import *
import AudioController
import threading
import serial

class ButtonController:

    def __init__(self, com, baud, timeout):

        # Audio Controller object
        self.audio_controller = AudioController.AudioController(44100, 512)

        # Variables for quadrant states (tracks if a quadrant is currently pressed)
        self.quadrant_states = {
            "top_left": False,
            "top_right": False,
            "bottom_left": False,
            "bottom_right": False,
        }

        # Recording state tracking
        self.recording_active = False
        self.recording_timer = None

        # Serial Setup - update COM port and baudrate to match Arduino
        self.ser = serial.Serial(com, baud, timeout=timeout)  # CHANGE COM3 to whatever your Arduino is using
    
        self.in_game = False

    def set_in_game_true(self):
        print('You are in game! Kick and Clap should be working.')
        self.in_game = True

    def set_in_game_false(self):
        print('You are in the main menu! Kick and Clap should be off.')
        self.in_game = False


    def start_recording(self):
        if not self.recording_active:
            print("Recording started.")
            self.recording_timer = threading.Timer(8.0, self.stop_recording)
            self.recording_timer.start()
            self.recording_active = True

    def stop_recording(self):
        if self.recording_active:
            print("Recording stopped.")
            self.audio_controller.stop_recording_and_save(silence_duration=0)
            self.recording_active = False
            self.recording_timer = None

    def handle_serial_input(self):

        # Serial input handling from Arduino
        try:
            serial_input = self.ser.readline().decode('utf-8').strip()
            # print(f"Recording Timer: {recording_timer}")

            if serial_input:

                # If the top left button is pressed
                if serial_input == "TL":
                    # if not self.quadrant_states["top_left"]:
                    self.quadrant_states["top_left"] = True
                    return "TL"
                else:
                    self.quadrant_states["top_left"] = False

                # If the top right button is pressed
                if serial_input == "TR":
                    # if not self.quadrant_states["top_right"]:
                    self.quadrant_states["top_right"] = True
                    return "TR"
                else:
                    self.quadrant_states["top_right"] = False

                # If the bottom left button is pressed
                if serial_input == "BL" and self.in_game:
                    if not self.quadrant_states["bottom_left"]:
                        self.audio_controller.play_kick()
                        self.start_recording()
                    self.quadrant_states["bottom_left"] = True
                else:
                    self.quadrant_states["bottom_left"] = False

                # If the bottom right button is pressed
                if serial_input == "BR" and self.in_game:
                    if not self.quadrant_states["bottom_right"]:
                        self.audio_controller.play_clap()
                        self.start_recording()
                    self.quadrant_states["bottom_right"] = True
                else:
                    self.quadrant_states["bottom_right"] = False

            # # Handle recording timer based on activity
            # print(f"Serial Input: {serial_input} First: {not any_quadrant_active} Second: {recording_active} Third: {recording_timer is None}")
            # if not any_quadrant_active and recording_active and recording_timer is None:
            #     print("No quadrant active. Stopping recording after delay.")

            # elif any_quadrant_active and recording_timer is not None:
            #     print("Activity detected. Cancelling stop timer.")
            #     recording_timer.cancel()
            #     recording_timer = None


            else:
                # Reset all quadrant states if no serial input
                self.quadrant_states["bottom_left"] = False
                self.quadrant_states["bottom_right"] = False
                self.quadrant_states["top_left"] = False
                self.quadrant_states["top_right"] = False

        except Exception as e:
            print(f"Serial error: {e}")