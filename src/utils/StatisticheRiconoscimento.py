class StatisticheRiconoscimento:
    def __init__(self):
        self.identificazioni_riuscite = 0
        self.tentativi_falliti = 0
        self.punteggi_confidenza = []
        self.frame_processati = 0
        self.tempistiche_embeddings = {}
        self.avg_tempistiche_embeddings = -1
        self.tempo_elaborazione_video = None 
        self.tempo_matching_volti_best = {}
        self.tempo_matching_volti_all = {}

    def aggiungi_tempistiche_embeddings(self, nome, durata):
        """Aggiunge una tempistica per la generazione degli embeddings."""
        self.tempistiche_embeddings.update({nome: durata})

    def set_elaborazione_video(self, durata):
        """Imposta il tempo totale di elaborazione del video."""
        self.tempo_elaborazione_video = durata

    def aggiungi_tempistiche_matching_volti_all(self, nome, durata):
        """Aggiunge una tempistica di matching per un volto specifico."""
        if nome in self.tempo_matching_volti_all:
            self.tempo_matching_volti_all[nome].append(durata)
        else:
            self.tempo_matching_volti_all[nome] = [durata]

    def calcola_avg_tempistiche_embeddings(self):
        """Calcola il tempo medio per la generazione degli embeddings."""
        if len(self.tempistiche_embeddings) > 0:
            self.avg_tempistiche_embeddings = sum(self.tempistiche_embeddings.values()) / len(self.tempistiche_embeddings)
        else:
            self.avg_tempistiche_embeddings = 0

    def calcola_volti_best(self):
        """Calcola i tempi migliori di matching per ogni volto."""
        for key in self.tempo_matching_volti_all:
            self.tempo_matching_volti_best.update({key: min(self.tempo_matching_volti_all[key])})

    def aggiungi_successo(self, confidenza):
        """Registra un'identificazione riuscita con il suo punteggio di confidenza."""
        self.identificazioni_riuscite += 1
        self.punteggi_confidenza.append(confidenza)

    def aggiungi_fallimento(self):
        """Registra un tentativo di identificazione fallito."""
        self.tentativi_falliti += 1

    def incrementa_frame(self):
        """Incrementa il contatore dei frame processati."""
        self.frame_processati += 1

    def calcola_statistiche(self):
        """
        Calcola e restituisce tutte le statistiche finali del sistema.
        
        Returns:
            dict: Dizionario contenente tutte le metriche di performance del sistema
        """
        self.calcola_volti_best()
        self.calcola_avg_tempistiche_embeddings()
        
        tentativi_totali = self.identificazioni_riuscite + self.tentativi_falliti
        tasso_successo = (self.identificazioni_riuscite / tentativi_totali * 100) if tentativi_totali > 0 else 0
        confidenza_media = sum(self.punteggi_confidenza) / len(self.punteggi_confidenza) if self.punteggi_confidenza else 0

        return {
            'frame_processati': self.frame_processati,
            'identificazioni_riuscite': self.identificazioni_riuscite,
            'tentativi_falliti': self.tentativi_falliti,
            'tasso_successo': tasso_successo,
            'confidenza_media': confidenza_media,
            'tempistiche_embeddings_media': self.avg_tempistiche_embeddings,
            'tempistiche_embeddings':self.tempistiche_embeddings,
            'tempo_elaborazione_video': self.tempo_elaborazione_video,
            'tempo_matching_volti_best': self.tempo_matching_volti_best,
            'tempo_matching_volti_all': self.tempo_matching_volti_all
        }
