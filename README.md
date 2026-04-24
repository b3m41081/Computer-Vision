# Computer Vision

Dieses Repository enthält Übungen und Implementierungen aus dem Master-Modul Computer Vision. Der Fokus liegt auf klassischer 3D-Computer-Vision mit OpenCV: Kamerakalibrierung, Entzerrung, Pose-Schätzung und Höhenmessung aus Einzelbildern.

## Inhalt

```text
.
├── a1/   Camera Calibration
└── a2/   Single-View Height Measurement
```

### A1: Camera Calibration

Workshop 1 behandelt die Kalibrierung einer Webcam mit einem Schachbrettmuster.

- Bilder über die Webcam aufnehmen
- Kameramatrix und Verzerrungskoeffizienten bestimmen
- Live-Bild entzerren
- Kamera-Pose schätzen und 3D-Achsen ins Bild projizieren

Start aus dem Ordner `a1/`:

```bash
python3 1_capture.py
python3 2_calibrate.py
python3 3_undistort.py
python3 4_pose.py
```

Weitere Details stehen in [`a1/README.md`](a1/README.md).

### A2: Single-View Height Measurement

Workshop 2 implementiert eine interaktive Höhenmessung aus einem einzelnen Bild. Über angeklickte Referenzpunkte, Fluchtpunkte und eine bekannte Referenzhöhe wird die Höhe eines Zielobjekts berechnet.

Start aus dem Projektordner:

```bash
python3 a2/src/main.py
```

Weitere Details zur Bedienung stehen in [`a2/README.md`](a2/README.md).

## Voraussetzungen

- Python 3
- OpenCV
- NumPy
- Webcam für die Live-Beispiele in `a1`

Installation der Python-Abhängigkeiten:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Projektstruktur

```text
a1/
├── 1_capture.py          # Aufnahme von Kalibrierungsbildern
├── 2_calibrate.py        # Berechnung der Kamerakalibrierung
├── 3_undistort.py        # Live-Entzerrung des Kamerabildes
├── 4_pose.py             # Pose-Schaetzung mit eingeblendeten 3D-Achsen
├── calibration.npz       # gespeicherte Kalibrierungsdaten
└── captured_images/      # aufgenommene Schachbrettbilder

a2/
├── img/                  # Eingabe- und Ergebnisbilder
└── src/                  # interaktive Hoehenmessung
```

## Hinweise

- Die Skripte in `a1` arbeiten relativ zum aktuellen Arbeitsverzeichnis. Daher sollten sie aus dem Ordner `a1/` gestartet werden.
- Die Anwendung in `a2` lädt standardmäßig Bilder nach dem Muster `a2/img/table_bottle*.jpeg`.
- Mit `s` kann in `a2` ein Overlay-Bild mit der aktuellen geometrischen Konstruktion gespeichert werden.
