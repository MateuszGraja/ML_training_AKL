# ML_training_AKL

# Ocena poziomu oświetlenia lamp pasa startowego z użyciem CNN

> Klasyfikacja pięciu poziomów oświetlenia (0%, 3%, 10%, 30%, 100%) na podstawie obrazów wyciętego ROI lampy, z interpretacją predykcji metodą **Integrated Gradients** i raportami jakości (krzywe **Precision‑Recall**).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)

---

## Spis treści

* [Opis](#opis)
* [Funkcjonalności](#funkcjonalności)
* [Wymagania i instalacja](#wymagania-i-instalacja)
* [Dane wejściowe i struktura](#dane-wejściowe-i-struktura)
* [Konfiguracja](#konfiguracja)
* [Uruchomienie](#uruchomienie)
* [Jak to działa](#jak-to-działa)

---

## Opis

Projekt trenuje klasyfikator obrazu na bazie wybranej architektury **ResNet18 / ResNet50 / EfficientNet‑B0** (pre‑trenowane na ImageNet). Skrypt:

* przygotowuje dane (podział 70/15/15),
* zamraża ciężary modelu i podmienia warstwę klasyfikacyjną,
* uczy model, loguje metryki i rysuje wykresy przebiegu nauki,
* generuje wieloklasowe krzywe **Precision‑Recall**,
* wizualizuje atrybucje **Integrated Gradients** (5 próbek/klasę).

---

## Funkcjonalności

* ✅ **Transfer learning** z `torchvision.models` (ResNet/EfficientNet)
* ✅ **Podział danych**: train / val / test = 70% / 15% / 15%
* ✅ **Metryki**: strata i accuracy dla train/val/test
* ✅ **Krzywe PR** dla każdej klasy + AP (pole pod krzywą)
* ✅ **Wyjaśnialność**: Integrated Gradients (Captum)
* ✅ **Artefakty** zapisywane w katalogu `outputs/`

---

## Wymagania i instalacja

```bash
# (opcjonalnie) nowy wirtualny environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# instalacja zależności (dobierz do swojej karty/CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas pillow matplotlib scikit-learn captum
```

> **Uwaga:** dobierz właściwe koło PyTorch (CPU/CUDA) zgodnie ze swoim środowiskiem.

---

## Dane wejściowe i struktura

Skrypt oczekuje:

* katalogu z obrazami (np. `light_speed/`)
* pliku CSV (np. `output_labels_speed.csv`) z dwiema kolumnami:
  `relatywna_ścieżka_do_obrazu,etykieta_liczbowa`

**Przykład struktury katalogu:**

```text
light_speed/
├── seq_0001/img_000001.jpg
├── seq_0001/img_000002.jpg
└── ...
```

**Przykładowy CSV:**

```csv
seq_0001/img_000001.jpg,4
seq_0001/img_000002.jpg,2
...
```

Domyślnie etykiety to liczby z zakresu `0..4`. W praktyce często odpowiadają klasom: **0%, 3%, 10%, 30%, 100%** – jeśli używasz innego mapowania, zachowaj spójność.

---

## Konfiguracja

Najważniejsze ustawienia znajdują się na początku skryptu:

```python
DATA_CSV = "output_labels_speed.csv"
DATA_ROOT = "light_speed"
NUM_CLASS = 5
MODEL_NAME = "efficientnet_b0"   # 'resnet18', 'resnet50', 'efficientnet_b0'
IMG_SIZE = (600, 700)              # np. (224, 224) albo (600, 700)
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "outputs"
```

**Wskazówki:**

* `MODEL_NAME`: łatwo przełączysz się między ResNet18 / ResNet50 / EfficientNet‑B0.
* `IMG_SIZE`: dopasuj do przygotowanego ROI; przetestuj 224×224 vs. 600×700.
* `BATCH_SIZE`, `LR`, `EPOCHS`: standardowe dźwignie stabilności/szybkości treningu.

---

## Uruchomienie

```bash
python train.py
```

Log z epok zawiera m.in. straty i accuracy dla train/val. Po zakończeniu treningu skrypt:

* ewaluuję model na teście,
* zapisuje krzywe PR,
* rysuje wykres przebiegu nauki,
* generuje atrybucje Integrated Gradients.

---

## Jak to działa

### Dataset i transformacje

```python
class SimpleImageDS(Dataset):
    """Dataset oczekujący pliku CSV (img_rel_path,label) i folderu ze zdjęciami."""
    def __init__(self, csv, root, tfm=None):
        # wczytuje adnotacje (path,label), pamięta katalog i transform
        ...
    def __len__(self):
        # zwraca rozmiar zbioru
        ...
    def __getitem__(self, idx):
        # otwiera obraz -> RGB, stosuje tfm, zwraca (tensor, label, ścieżka)
        ...
```

Transformacje standaryzują obraz do `IMG_SIZE` i normalizują jak w ImageNet:

```python
transforms.Resize(IMG_SIZE)
transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
```

### Podział na zbiory

* losowy split: **70%** train / **15%** val / **15%** test
* `DataLoader` z `shuffle=True` dla treningu

### Model (transfer learning)

```python
def get_model(name: str):
    """
    Zwraca model z zamrożonymi wagami (feature extractor) i podmienioną
    ostatnią warstwą na 'NUM_CLASS' wyjść.
    """
    ...
```

**Warianty:**

* **ResNet**: `model.fc = nn.Linear(...)`
* **EfficientNet‑B0**: `model.classifier[1] = nn.Linear(...)`

Wszystkie parametry są zamrożone (`requires_grad=False`).

### Trening i walidacja

* Optymalizator: **Adam** (`lr = 1e-4`)
* Funkcja straty: **CrossEntropyLoss**
* Logowanie: strata/accuracy dla train i val po każdej epoce

### Test, PR‑curves i AP

* Ewaluacja na zbiorze testowym (strata + accuracy)
* Krzywe **Precision‑Recall** liczone per klasa (własna implementacja `compute_pr_curve`)
* **AP** liczone przez całkowanie trapezowe (`np.trapz(precisions, recalls)`)

### Wyjaśnialność: Integrated Gradients

* Dla każdej klasy pobieranych jest 5 obrazów z walidacji
* `captum.attr.IntegratedGradients` wylicza atrybucje
* Zapisywany jest podgląd: oryginał + mapa atrybucji (skala szarości)

---

