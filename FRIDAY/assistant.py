import os
import json
import webbrowser
import pyttsx3
import speech_recognition as sr
import ollama
import pyautogui
import time

engine = None

def speak(text):
    global engine
    if engine is None:
        engine = pyttsx3.init()
    print("AI:", text)
    engine.say(text)
    engine.runAndWait()

def listen_once():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            audio = r.listen(source, timeout=5, phrase_time_limit=4)
            text = r.recognize_google(audio)
            return text.lower()
    except Exception:
        return ""

def wait_for_wake_word():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Waiting for wake word...")
        r.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                audio = r.listen(source, timeout=None, phrase_time_limit=3)
                text = r.recognize_google(audio).lower()

                print("Heard:", text)

                if "stop" in text:
                    speak("Shutting down")
                    exit()

                if "hey friday" in text:
                    speak("Yes?")
                    return

            except Exception:
                continue

def ask_ai(prompt):
    system_prompt = """
You control a Windows computer. ALWAYS respond in JSON.

Actions:
- open_app
- close_app
- open_website
- type_text
- press_key
- hotkey
- click
- scroll
- send_whatsapp
- speak

Rules:
- If user says "close", ALWAYS use close_app
- If user says "open", ALWAYS use open_app

Examples:

User: open chrome
{"action":"open_app","value":"chrome"}

User: close chrome
{"action":"close_app","value":"chrome.exe"}

User: close vs code
{"action":"close_app","value":"Code.exe"}

User: type hello
{"action":"type_text","value":"hello"}

User: copy
{"action":"hotkey","keys":["ctrl","c"]}
"""

    response = ollama.chat(
        model='llama3',
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content']

def open_whatsapp():
    os.system('explorer shell:AppsFolder\\5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App')
    time.sleep(5)

def send_whatsapp(name, message):
    open_whatsapp()
    time.sleep(3)

    pyautogui.hotkey("ctrl", "f")
    time.sleep(1)

    pyautogui.write(name)
    time.sleep(2)

    pyautogui.press("enter")
    time.sleep(2)

    pyautogui.write(message)
    time.sleep(1)

    pyautogui.press("enter")

def execute(ai_response):
    try:
        data = json.loads(ai_response)

        action = data.get("action")
        value = data.get("value")

        app_map = {
            "chrome": "chrome.exe",
            "vs code": "Code.exe",
            "code": "Code.exe",
            "notepad": "notepad.exe",
            "calculator": "calc.exe"
        }

        if action == "open_app":
            if value == "whatsapp":
                open_whatsapp()
            else:
                os.system(f'start "" "{value}"')

        elif action == "close_app":
            exe_name = app_map.get(value, value)
            os.system(f'taskkill /f /im {exe_name}')

        elif action == "open_website":
            webbrowser.open(value)

        elif action == "type_text":
            pyautogui.write(value)

        elif action == "press_key":
            pyautogui.press(value)

        elif action == "hotkey":
            pyautogui.hotkey(*data.get("keys"))

        elif action == "click":
            pyautogui.click()

        elif action == "scroll":
            pyautogui.scroll(data.get("value", -300))

        elif action == "send_whatsapp":
            send_whatsapp(data.get("name"), data.get("message"))

        elif action == "speak":
            speak(value)

        else:
            print("Unknown action:", action)

    except Exception as e:
        print("Error:", e)
        print("AI response:", ai_response)

def main():
    while True:
        wait_for_wake_word()

        print("Listening for command...")
        command = listen_once()

        if not command:
            speak("I didn't hear anything")
            continue

        print("You:", command)

        ai_response = ask_ai(command)
        print("AI RAW:", ai_response)

        execute(ai_response)


if __name__ == "__main__":
    main()
