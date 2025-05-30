import glob
import os
import shutil
import time
import cv2
from ultralytics import YOLO 
import ffmpeg
import wget
import yt_dlp

from src.config.configurazione_attuale import configurazione  
from src.utils.RiconoscitoreFacciale import RiconoscitoreFacciale
from src.utils.Persona import Persona
from src.utils.statistiche_attuali import stats

def crea_cartelle_necessarie():
    os.makedirs(configurazione.DIRVOLTI, exist_ok=True)
    os.makedirs(f'{configurazione.PROJECT_ROOT}data/temp_images', exist_ok=True)

def dowload_immagini():    
    for chiave in configurazione.URLIMMAGINI:
        wget.download(configurazione.URLIMMAGINI[chiave], f'{configurazione.DIRVOLTI}/{chiave}.jpg')

def dowload_YT_video():
    ydl_opts = {
        'format': 'best',
        'outtmpl': configurazione.PATHVIDEO,
        'nocheckcertificate': True,
        'skip_download': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([configurazione.URLVIDEO])

def ritaglia_video():
    ffmpeg.input(configurazione.PATHVIDEO, ss=configurazione.SECONDOINIZIO, t=configurazione.DURATAVIDEO).output(configurazione.PATHVIDEOTAGLIATO).run()
    
def dowload_e_taglia_video():
    dowload_YT_video()
    ritaglia_video()
    
def creazione_e_tracciamento_video_con_YOLO():    
    model = YOLO(configurazione.YOLO_MODEL)
    return model.track(source=configurazione.PATHVIDEOTAGLIATO, conf=0.3, iou=0.5, stream=True)

def creazione_dizionario_nome_Persona():
    dizionario = {}
    face_files = [name for name in os.listdir(configurazione.DIRVOLTI) if name.lower().endswith(configurazione.ESTENSIONI_IMMAGINI)]

    for name in face_files:
        person_name = name.split('.')[0]
        full_path = f'{configurazione.DIRVOLTI}/{name}'

        if os.path.exists(full_path):
            dizionario[person_name] = Persona(person_name, full_path)
        else:
            print(f"Error: File not found: {full_path}")
    return dizionario

def rimuovi_files_cache_e_contenuto_auraface_dir():
    for cache_file in configurazione.CACHE_FILES:
        if os.path.exists(cache_file):
            os.remove(cache_file)

    _rimuovi_file_in_directory(configurazione.AURAFACE_DIR)

def _rimuovi_file_in_directory(directory):
    """Rimuove tutti i file in una directory"""
    if os.path.exists(directory):
        files = glob.glob(os.path.join(directory, '*'))
        for f in files:
            try:
                if os.path.isfile(f) or os.path.islink(f):
                    os.unlink(f)
                elif os.path.isdir(f):
                    shutil.rmtree(f)
            except Exception as e:
                pass

def identifica_persona(percorso_immagine, riconoscitore_volti, soglia_confidenza=0.55):
    """
    Identifica una persona dall'immagine ritagliata con tracciamento delle tempistiche.
    """
    if not os.path.exists(percorso_immagine):
        return '-1', 0.0

    try:
        start_time = time.time()

        nome, confidenza = riconoscitore_volti.identifica_volto(percorso_immagine, soglia_confidenza)

        matching_time = time.time() - start_time
        if nome != '-1':
            stats.aggiungi_tempistiche_matching_volti_all(nome, matching_time)

        return nome, confidenza
    except Exception:
        return '-1', 0.0

def e_persona_conosciuta(id_tracciamento):
    """Verifica se la persona è già stata identificata."""
    return id_tracciamento is not None and id_tracciamento in _id_conosciuti

def e_ritaglio_valido(immagine, dimensione_minima=40):
    """
    Verifica se l'immagine è adatta per il riconoscimento.

    Args:
        immagine: Array numpy dell'immagine
        dimensione_minima: Dimensione minima richiesta

    Returns:
        bool: True se l'immagine è valida
    """
    if immagine is None or immagine.size == 0:
        return False

    altezza, larghezza = immagine.shape[:2]
    return min(altezza, larghezza) >= dimensione_minima

def estrai_coordinate_box(box, larghezza_frame, altezza_frame, margine=20):
    """
    Estrae le coordinate del bounding box con margine di sicurezza.

    Returns:
        tuple: (x1, y1, x2, y2) coordinate corrette
    """
    coordinate_xyxy = box.xyxy[0].cpu().numpy().astype(int)

    x1 = max(0, coordinate_xyxy[0] - margine)
    y1 = max(0, coordinate_xyxy[1] - margine)
    x2 = min(larghezza_frame, coordinate_xyxy[2] + margine)
    y2 = min(altezza_frame, coordinate_xyxy[3] + margine)

    return x1, y1, x2, y2

def aggiorna_tracciamento_persona(nome_identificato, id_tracciamento, confidenza):
    """
    Aggiorna il tracciamento di una persona identificata.

    Args:
        nome_identificato: Nome della persona identificata
        id_tracciamento: ID di tracciamento corrente
        confidenza: Punteggio di confidenza
    """
    if nome_identificato not in dizionario:
        return

    persona = dizionario[nome_identificato]
    vecchio_id = persona.id

    # Aggiorna i dati della persona
    persona.update(id_tracciamento, confidenza)

    # Gestisce il cambio di ID di tracciamento
    if vecchio_id is not None and vecchio_id != id_tracciamento:
        if vecchio_id in _id_conosciuti:
            _id_conosciuti.remove(vecchio_id)

    # Aggiunge il nuovo ID al tracciamento
    if id_tracciamento is not None:
        _id_conosciuti.add(id_tracciamento)

def processa_rilevazioni(results, riconoscitore_volti):
    """
    Processa i risultati delle rilevazioni YOLO per identificare le persone.

    Args:
        results: Risultati delle rilevazioni YOLO
    """
    for indice, risultato in enumerate(results):
        # Salta frame in base al tasso di campionamento
        if indice % configurazione.FRAMEDASALTARE != 0:
            continue

        stats.incrementa_frame()

        # Verifica se ci sono rilevazioni nel frame
        if risultato.boxes is None:
            continue

        altezza_frame, larghezza_frame = risultato.orig_img.shape[:2]

        # Processa ogni rilevazione nel frame
        for box in risultato.boxes:
            processa_singola_rilevazione(box, risultato.orig_img, larghezza_frame, altezza_frame, stats, riconoscitore_volti)

def processa_singola_rilevazione(box, immagine_frame, larghezza_frame, altezza_frame, stats, riconoscitore_volti):
    """
    Processa una singola rilevazione di persona.

    Args:
        box: Bounding box della rilevazione
        immagine_frame: Immagine del frame corrente
        larghezza_frame, altezza_frame: Dimensioni del frame
        stats: Oggetto statistiche per il tracciamento
    """
    id_classe = int(box.cls)
    id_tracciamento = box.id.item() if box.id is not None else None

    # Processa solo rilevazioni di persone (classe 0)
    if id_classe != 0:
        return

    # Salta persone già identificate
    if e_persona_conosciuta(id_tracciamento):
        return

    # Estrae e valida l'immagine ritagliata
    x1, y1, x2, y2 = estrai_coordinate_box(box, larghezza_frame, altezza_frame)
    immagine_ritagliata = immagine_frame[y1:y2, x1:x2]

    if not e_ritaglio_valido(immagine_ritagliata):
        return

    # Tenta l'identificazione della persona
    tentativo_identificazione(immagine_ritagliata, id_tracciamento, stats, riconoscitore_volti)

def tentativo_identificazione(immagine_ritagliata, id_tracciamento, stats, riconoscitore_volti):
    """
    Esegue il tentativo di identificazione di una persona.

    Args:
        immagine_ritagliata: Immagine della persona ritagliata
        id_tracciamento: ID di tracciamento della persona
        stats: Oggetto statistiche
    """
    file_salvato = False

    try:
        # Salva temporaneamente l'immagine
        cv2.imwrite(configurazione.TEMPIMG, immagine_ritagliata)
        file_salvato = True

        # Esegue l'identificazione
        nome_identificato, confidenza = identifica_persona(configurazione.TEMPIMG, riconoscitore_volti)

        if nome_identificato != '-1':
            # Identificazione riuscita
            stats.aggiungi_successo(confidenza)
            aggiorna_tracciamento_persona(nome_identificato, id_tracciamento, confidenza)
        else:
            # Identificazione fallita
            stats.aggiungi_fallimento()

    except Exception:
        stats.aggiungi_fallimento()

    finally:
        # Pulizia del file temporaneo
        if file_salvato and os.path.exists(configurazione.TEMPIMG):
            os.remove(configurazione.TEMPIMG)
            

def inizializza_tutto():
    riconoscitore_volti = RiconoscitoreFacciale()
    riconoscitore_volti.carica_volti_noti(configurazione.DIRVOLTI)

    start_time_video = time.time()
    processa_rilevazioni(creazione_e_tracciamento_video_con_YOLO(), riconoscitore_volti)
    start_time_video = time.time()

    tempo_analisi_video = start_time_video - start_time_video

    stats.set_elaborazione_video(tempo_analisi_video)

    return stats.calcola_statistiche()


def do_all():
    crea_cartelle_necessarie()
    dowload_immagini()
    dowload_e_taglia_video()
    rimuovi_files_cache_e_contenuto_auraface_dir()
    return inizializza_tutto()
            
dizionario = creazione_dizionario_nome_Persona()

_id_conosciuti = id_conosciuti = {
    dizionario[chiave].id
    for chiave in dizionario
    if hasattr(dizionario[chiave], 'id') and dizionario[chiave].id is not None
}