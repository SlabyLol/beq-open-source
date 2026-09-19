import sys
import subprocess
import os

def fix_and_run():
    print("=" * 50)
    print("REPARATUR-SKRIPT: INSTALLIERE REQUISITEN...")
    print("=" * 50)
    
    # 1. Ermittle den aktuellen Interpreter
    current_python = sys.executable
    print(f"Aktiver Python-Interpreter: {current_python}")
    
    # 2. Erpringe die Installation in genau diesem Interpreter
    required_packages = ["transformers", "torch", "safetensors"]
    
    for package in required_packages:
        print(f"\nPrüfe / Installiere {package}...")
        try:
            # Versuche das Paket zu importieren
            __import__(package)
            print(f"[OK] {package} ist bereits vorhanden.")
        except ImportError:
            print(f"[FIX] {package} fehlt. Starte Installation...")
            try:
                subprocess.check_call([current_python, "-m", "pip", "install", package])
                print(f"[OK] {package} wurde erfolgreich installiert!")
            except Exception as e:
                print(f"[FEHLER] Installation von {package} fehlgeschlagen: {e}")
                print("Versuche es mit Administrator-Rechten oder prüfe deine Internetverbindung.")
                return

    print("\n" + "=" * 50)
    print("ALLE MODULE ERFOLGREICH INSTALLIERT!")
    print("=" * 50)
    
    # 3. Vorbereitung für dein Modell
    print("\nDu kannst nun dein Modell ausführen.")
    print("Ersetze im Code unten einfach den Pfad zu deinen Safetensors.")
    
    # Beispielhafter Lade-Code, der jetzt ohne Fehler importiert werden kann
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("[INFO] 'transformers' lässt sich jetzt fehlerfrei importieren!")
        
        # HINWEIS FÜR DEN NUTZER:
        # Hier den Pfad eintragen, z.B.: model_path = "./mein-modell-ordner"
        # model = AutoModelForCausalLM.from_pretrained(model_path)
    except Exception as e:
        print(f"Fehler beim Test-Import: {e}")

if __name__ == "__main__":
    fix_and_run()
