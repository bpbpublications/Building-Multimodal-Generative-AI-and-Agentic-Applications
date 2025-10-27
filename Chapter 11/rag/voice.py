import speech_recognition as sr
import pyttsx3
import time

def speak_text(text: str):
    """
    Speaks out the provided text using the system's default voice engine.
    """
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def listen_from_microphone(timeout=5) -> str:
    """
    Captures speech from the microphone and returns transcribed text.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening... Speak now.")
        audio = recognizer.listen(source, timeout=timeout)

    try:
        query = recognizer.recognize_google(audio)
        print(f"✅ Recognized: {query}")
        return query
    except sr.UnknownValueError:
        raise ValueError("Speech recognition failed: could not understand audio.")
    except sr.RequestError as e:
        raise ConnectionError(f"Speech recognition service error: {e}")


def stream_transcription(duration=5) -> str:
    """
    Captures speech with a live-like feedback loop for the given duration (in seconds).
    Returns the full transcript as a string.
    """
    recognizer = sr.Recognizer()
    transcript = ""

    with sr.Microphone() as source:
        print("🎤 Streaming mic input...")
        stt_end_time = time.time() + duration
        while time.time() < stt_end_time:
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
                segment = recognizer.recognize_google(audio)
                transcript += f"{segment} "
                print(f"📝 {segment}")
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"API error: {e}")
                break

    return transcript.strip()
