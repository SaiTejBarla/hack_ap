import speech_recognition as sr

def speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="te-IN")
    except Exception as e:
        return f"Error: {str(e)}"
