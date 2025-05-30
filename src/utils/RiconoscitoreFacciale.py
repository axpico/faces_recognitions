import os
import pickle
import time
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.statistiche_attuali import stats
from src.utils.utils import *

class RiconoscitoreFacciale:
    """
    Sistema di riconoscimento facciale con supporto per modelli ONNX e AuraFace come fallback.
    Rimossa dipendenza da InsightFace.
    """

    def __init__(self, nome_modello=None, percorso_modello=None):
        """
        Inizializza il riconoscitore facciale.

        Args:
            nome_modello: Nome specifico del modello da usare
            percorso_modello: Percorso personalizzato del modello
        """
        self.nome_modello_richiesto = nome_modello
        self.percorso_modello = percorso_modello

        # Stato interno
        self.modello_attivo = "unknown"
        self.session = None
        self.input_name = None
        self.output_names = None
        self.embeddings_noti = []
        self.nomi_noti = []

        # Inizializza il modello migliore disponibile
        self._inizializza_modello()

        # File cache specifico per modello
        self.file_cache = f'cache_embeddings_{self.modello_attivo.lower()}.pkl'

    def _inizializza_modello(self):
        """Prova a inizializzare i modelli in ordine di priorità."""
        # Lista modelli da provare in ordine
        modelli = []

        if self.nome_modello_richiesto:
            modelli.append(('custom', self.nome_modello_richiesto))

        if self.percorso_modello:
            modelli.append(('onnx_custom', self.percorso_modello))

        # AuraFace come fallback
        modelli.append(('auraface', None))

        # Prova ogni modello fino a trovarne uno funzionante
        for tipo, nome in modelli:
            if self._carica_modello(tipo, nome):
                print(f"Modello '{self.modello_attivo}' caricato con successo")
                return

        raise Exception("Nessun modello di riconoscimento facciale disponibile")

    def _carica_modello(self, tipo, nome=None):
        """
        Carica un tipo specifico di modello.

        Returns:
            bool: True se il caricamento è riuscito
        """
        try:
            if tipo == 'custom' or tipo == 'onnx_custom':
                percorso = nome if tipo == 'custom' else self.percorso_modello
                if percorso and str(percorso).endswith('.onnx'):
                    return self._carica_modello_onnx_diretto(percorso)
            elif tipo == 'auraface':
                return self._carica_auraface_onnx()

        except Exception as e:
            print(f"Errore caricamento {tipo}: {e}")
            return False

    def _carica_auraface_onnx(self):
        """Carica AuraFace come modello ONNX diretto."""
        from huggingface_hub import snapshot_download

        # Scarica modello se necessario
        auraface_dir = "models/auraface"
        os.makedirs(auraface_dir, exist_ok=True)

        try:
            snapshot_download("fal/AuraFace-v1", local_dir=auraface_dir)

            # Cerca file ONNX nella cartella
            onnx_files = [f for f in os.listdir(auraface_dir) if f.endswith('.onnx')]

            if not onnx_files:
                print("Nessun file ONNX trovato in AuraFace")
                return False

            # Usa il primo file ONNX trovato
            percorso_onnx = os.path.join(auraface_dir, onnx_files[0])

            return self._carica_modello_onnx_diretto(percorso_onnx, nome_modello="AuraFace")

        except Exception as e:
            print(f"Errore download/caricamento AuraFace: {e}")
            return False

    def _carica_modello_onnx_diretto(self, percorso_onnx, nome_modello=None):
        """Carica direttamente un file .onnx."""
        import onnxruntime as ort

        if not os.path.exists(percorso_onnx):
            print(f"File ONNX non trovato: {percorso_onnx}")
            return False

        # Configura i provider
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # Carica il modello
        self.session = ort.InferenceSession(percorso_onnx, providers=providers)

        # Ottieni informazioni su input/output
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Imposta il nome del modello
        if nome_modello:
            self.modello_attivo = nome_modello
        else:
            nome_file = os.path.basename(percorso_onnx)
            self.modello_attivo = os.path.splitext(nome_file)[0]

        print(f"Modello ONNX caricato: {self.modello_attivo}")
        print(f"Input: {self.input_name}")
        print(f"Output: {self.output_names}")

        return True

    def _preprocessa_immagine_onnx(self, immagine):
        """Preprocessa l'immagine per modelli ONNX diretti."""
        # Ridimensiona a 112x112 (standard per ArcFace)
        img_resized = cv2.resize(immagine, (112, 112))

        # Normalizza
        img_normalized = (img_resized.astype(np.float32) - 127.5) / 127.5

        # Converti da HWC a CHW e aggiungi batch dimension
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        img_batch = np.expand_dims(img_transposed, axis=0)

        return img_batch

    def _rileva_volti_semplice(self, immagine):
        """Rilevamento volti semplice usando OpenCV per modelli ONNX diretti."""
        # Carica il classificatore Haar per volti
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Converti in scala di grigi
        gray = cv2.cvtColor(immagine, cv2.COLOR_BGR2GRAY)

        # Rileva volti
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        return faces

    def estrai_embedding(self, percorso_immagine):
        """
        Estrae l'embedding facciale da un'immagine usando modelli ONNX.

        Returns:
            numpy.array: Embedding del volto o None se nessun volto trovato
        """
        try:
            # Carica immagine
            immagine = cv2.imread(percorso_immagine)
            if immagine is None:
                print(f"Impossibile caricare l'immagine: {percorso_immagine}")
                return None

            return self._estrai_embedding_onnx(immagine)

        except Exception as e:
            print(f"Errore estrazione embedding: {e}")
            return None

    def _estrai_embedding_onnx(self, immagine):
        """Estrae embedding usando modello ONNX diretto."""
        # Rileva volti con OpenCV
        faces = self._rileva_volti_semplice(immagine)

        if len(faces) == 0:
            return None

        # Prendi il volto più grande
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face

        # Estrai ROI del volto
        face_roi = immagine[y:y+h, x:x+w]

        # Preprocessa per il modello
        input_data = self._preprocessa_immagine_onnx(face_roi)

        # Esegui inferenza
        outputs = self.session.run(self.output_names, {self.input_name: input_data})

        # Restituisci il primo output (embedding)
        embedding = outputs[0][0]

        # Normalizza l'embedding
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def carica_volti_noti(self, cartella_volti):
        """Carica gli embeddings dei volti noti dalla cartella o dalla cache."""
        # Prova a caricare dalla cache
        if self._carica_cache():
            return

        # Elabora le immagini della cartella
        if not os.path.exists(cartella_volti):
            print(f"Cartella non trovata: {cartella_volti}")
            return

        # Trova tutti i file immagine
        estensioni_valide = ('.png', '.jpg', '.jpeg', '.bmp')
        file_immagini = [f for f in os.listdir(cartella_volti) if f.lower().endswith(estensioni_valide)]

        print(f"Elaborazione {len(file_immagini)} immagini...")

        # Processa ogni immagine
        for file_img in file_immagini:
            percorso_completo = os.path.join(cartella_volti, file_img)
            embedding = self.estrai_embedding(percorso_completo)
            if embedding is not None:
                start_time = time.time()
                nome_persona = os.path.splitext(file_img)[0]
                self.embeddings_noti.append(embedding)
                self.nomi_noti.append(nome_persona)
                end_time = time.time()
                stats.aggiungi_tempistiche_embeddings(nome_persona, (end_time - start_time))
                print(f"{nome_persona}")

        # Salva cache se ci sono embeddings
        if self.embeddings_noti:
            self._salva_cache()
            print(f"Cache salvata: {len(self.embeddings_noti)} volti")

    def _carica_cache(self):
        """Carica embeddings dalla cache se compatibile."""
        if not os.path.exists(self.file_cache):
            return False

        try:
            with open(self.file_cache, 'rb') as f:
                dati = pickle.load(f)

            # Verifica compatibilità modello
            if dati.get('modello') == self.modello_attivo:
                self.embeddings_noti = dati['embeddings']
                self.nomi_noti = dati['nomi']
                print(f"Cache caricata ({len(self.embeddings_noti)} volti)")
                return True

        except Exception as e:
            print(f"Errore caricamento cache: {e}")

        return False

    def _salva_cache(self):
        """Salva embeddings nella cache."""
        try:
            dati_cache = {
                'embeddings': self.embeddings_noti,
                'nomi': self.nomi_noti,
                'modello': self.modello_attivo
            }
            with open(self.file_cache, 'wb') as f:
                pickle.dump(dati_cache, f)
        except Exception as e:
            print(f"Errore salvataggio cache: {e}")

    def identifica_volto(self, percorso_immagine, soglia=0.4):
        """
        Identifica un volto confrontandolo con il database.

        Returns:
            tuple: (nome_persona, confidenza) o ('-1', confidenza) se non riconosciuto
        """
        if not self.embeddings_noti:
            return '-1', 0.0

        # Estrai embedding dall'immagine
        embedding = self.estrai_embedding(percorso_immagine)
        if embedding is None:
            return '-1', 0.0

        # Calcola similarità con tutti i volti noti
        embedding = embedding.reshape(1, -1)
        embeddings_db = np.array(self.embeddings_noti)
        similarita = cosine_similarity(embedding, embeddings_db)[0]

        # Trova il match migliore
        indice_migliore = np.argmax(similarita)
        confidenza_migliore = similarita[indice_migliore]

        # Verifica se supera la soglia
        if confidenza_migliore >= soglia:
            return self.nomi_noti[indice_migliore], confidenza_migliore

        return '-1', confidenza_migliore

    def aggiungi_volto(self, percorso_immagine, nome_persona):
        """Aggiunge un nuovo volto al database."""

        start_time = time.time()
        embedding = self.estrai_embedding(percorso_immagine)
        end_time = time.time()
        stats.aggiungi_tempistiche_embeddings(nome_persona, (end_time - start_time))

        if embedding is None:
            print(f"Impossibile estrarre volto da {percorso_immagine}")
            return False

        # Aggiungi al database
        self.embeddings_noti.append(embedding)
        self.nomi_noti.append(nome_persona)

        # Aggiorna cache
        self._salva_cache()
        print(f"{nome_persona} aggiunto al database")
        return True

    def get_info_modello(self):
        """Restituisce informazioni sul modello attivo."""
        return {
            'modello_attivo': self.modello_attivo,
            'tipo_modello': 'onnx_diretto',
            'modello_richiesto': self.nome_modello_richiesto,
            'volti_nel_database': len(self.embeddings_noti)
        }