"""
File delle costanti per il sistema di riconoscimento facciale.
Contiene tutte le configurazioni e i parametri utilizzati nel progetto.
"""

from pathlib import Path


class Config:
    """
    Classe contenente tutte le configurazioni e costanti del sistema di riconoscimento facciale.
    Tutti gli attributi sono pubblici e accessibili come Config.NOME_ATTRIBUTO
    """
    
    # === PERCORSI BASE ===
    PROJECT_ROOT = Path(__file__).parent.parent.parent

    # === PERCORSI E DIRECTORY ===
    DIRVOLTI = f'{PROJECT_ROOT}/data/volti'
    PATHVIDEO = f'{PROJECT_ROOT}/data/video.mp4'
    PATHVIDEOTAGLIATO = f'{PROJECT_ROOT}/data/videoTagliato.mp4'
    TEMPIMG = f'{PROJECT_ROOT}/data/temp_images/temp.jpg'

    # === DIRECTORY PRINCIPALI ===
    DATA_DIR = f'{PROJECT_ROOT}/data'
    TEMP_IMAGES_DIR = f'{PROJECT_ROOT}/data/temp_images'
    CACHE_DIR = f'{PROJECT_ROOT}/data/cache'
    MODELS_DIR = f'{PROJECT_ROOT}/models'
    LOGS_DIR = f'{PROJECT_ROOT}/logs'

    # === CONFIGURAZIONE VIDEO ===
    URLVIDEO = "https://www.youtube.com/watch?v=6U4-KZSoe6g"
    FRAMEDASALTARE = 3
    SECONDOINIZIO = 14
    DURATAVIDEO = 14

    # === URL IMMAGINI PERSONE ===
    URLIMMAGINI = {
        'Jim Carrey': 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fm.media-amazon.com%2Fimages%2FM%2FMV5BMTQwMjAwNzI0M15BMl5BanBnXkFtZTcwOTY1MTMyOQ%40%40._V1_.jpg&f=1&nofb=1&ipt=b9e9bcc32daa1131841e8f37fd79e22018c58dcb9948134a52a2595bea921bf0',
        'Laura Linney': 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fm.media-amazon.com%2Fimages%2FM%2FMV5BMTMyMDc3Mzc2M15BMl5BanBnXkFtZTcwMjc5OTcyMg%40%40._V1_.jpg&f=1&h=400',
        'lukaku': 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.thefamouspeople.com%2Fprofiles%2Fimages%2Fromelu-lukaku-5.jpg&f=1&nofb=1&ipt=cd979c161ffcc8d75cb1be8bb7da25b84693dcfe67f9371e92c8f43be1f83867',
        'conte': 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.baritalianews.it%2Fwp-content%2Fuploads%2F2014%2F05%2Fconte.jpg&f=1&nofb=1&ipt=81a1f0aa719ae9b154d7ee8d45c1ccd2ae975535967f84dd7e8762a636843881',
    }

    # === PARAMETRI RICONOSCIMENTO ===
    SOGLIA_CONFIDENZA_DEFAULT = 0.55
    DIMENSIONE_MINIMA_IMMAGINE = 40
    MARGINE_BOUNDING_BOX = 20

    # === CONFIGURAZIONE YOLO ===
    YOLO_CONFIDENCE = 0.3
    YOLO_IOU = 0.5
    YOLO_MODEL = "yolo11n.pt"

    # === ESTENSIONI FILE ===
    ESTENSIONI_IMMAGINI = ('.jpg', '.jpeg', '.png', '.bmp')

    # === CACHE FILES (con percorsi completi) ===
    CACHE_EMBEDDINGS_VOLTI = f'{CACHE_DIR}/cache_embeddings_volti.pkl'
    CACHE_EMBEDDINGS_AURAFACE = f'{CACHE_DIR}/cache_embeddings_volti_auraface.pkl'
    CACHE_EMBEDDINGS_BUFFALO = f'{CACHE_DIR}/cache_embeddings_volti_buffalo_l.pkl'

    CACHE_FILES = [
        CACHE_EMBEDDINGS_VOLTI,
        CACHE_EMBEDDINGS_AURAFACE,
        CACHE_EMBEDDINGS_BUFFALO
    ]

    # === MODELLI ===
    AURAFACE_DIR = f'{MODELS_DIR}/auraface'
    AURAFACE_MODEL_REPO = "fal/AuraFace-v1"

    # === PROVIDER ONNX ===
    ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # === DIMENSIONI STANDARD ===
    FACE_SIZE_STANDARD = (112, 112)  # Dimensione standard per ArcFace

    # === PARAMETRI NORMALIZZAZIONE ===
    NORMALIZATION_MEAN = 127.5
    NORMALIZATION_STD = 127.5

    # === CLASSI YOLO ===
    CLASSE_PERSONA = 0
    