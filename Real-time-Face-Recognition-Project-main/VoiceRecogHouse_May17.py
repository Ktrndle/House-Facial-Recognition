import speech_recognition as sr
import pyttsx3
import serial
import time
from google import genai

# Initialize Serial Communication to Arduino
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=1)  # Change port accordingly
time.sleep(2)  # Arduino init delay

# Initialize TTS Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# Initialize Google Gemini API
api_key = "AIzaSyDMXx-F-5Ig5Z1YjvKz4pbg7-jgXaRcdPo"
genai_client = genai.Client(api_key=api_key)

# Speech Recognition Function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Please speak clearly.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=5)
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            print("Could not understand the audio.")
        except sr.RequestError:
            print("Could not request results.")
        except Exception as e:
            print(f"Unexpected error: {e}")
    return ""

# Text-to-Speech Function
def speak_text(text):
    engine.say(text)
    engine.runAndWait()

# Gemini Chat Fallback
def chat_with_gemini(prompt):
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash-preview-04-17",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error] {e}")
        return "Sorry, I'm having trouble connecting to Gemini API."

# Map user speech to Arduino commands
command_map = {
    "open main door": "open main door",
    "open front door": "open main door",
    "open back door": "open back door",
    "close main door": "close main door",
    "close back door": "close back door",
    "close all doors": "close all doors",
    "turn on lights": "turn on lights",
    "turn off lights": "turn off lights",
    # add more mappings as needed
}

def find_best_command(user_text):
    user_text = user_text.lower()
    for phrase in command_map:
        if phrase in user_text:
            return command_map[phrase]
    return None

def send_to_arduino(command):
    try:
        arduino.write((command + '\n').encode())
        print(f"Sent to Arduino: {command}")
    except Exception as e:
        print(f"Failed to send to Arduino: {e}")

def main():
    while True:
        print("\nSay something or say 'exit' to quit.")
        user_input = recognize_speech()
        if not user_input:
            continue

        print(f"You: {user_input}")

        if user_input.lower() in ['exit', 'quit', 'bye']:
            speak_text("Goodbye!")
            print("Goodbye!")
            break

        matched_command = find_best_command(user_input)

        if matched_command:
            send_to_arduino(matched_command)
            speak_text(f"Executing command: {matched_command}")
        else:
            response = chat_with_gemini(user_input)
            print(f"Chatbot: {response}")
            speak_text(response)

if __name__ == "__main__":
    main()