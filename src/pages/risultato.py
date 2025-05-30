import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

from utils.StatisticheRiconoscimento import StatisticheRiconoscimento
from src.config.configurazione_attuale import configurazione
from src.utils.statistiche_attuali import stats
from src.utils.utils import dizionario
from src.utils.utils import do_all


#self.identificazioni_riuscite = 0 -
#self.tentativi_falliti = 0 -
#self.punteggi_confidenza = [] -
#self.frame_processati = 0 - 
#self.tempistiche_embeddings = {}
#self.avg_tempistiche_embeddings = -1
#self.tempo_elaborazione_video = None
#self.tempo_matching_volti_best = {}
#self.tempo_matching_volti_all = {}

st.title("Resoconto del modello")
st.divider()
st.subheader(f'Frame processati: {stats.frame_processati}')
st.subheader(f'Identificazioni fallite: {stats.tentativi_falliti}')
st.subheader(f'identificazioni riuscite: {stats.identificazioni_riuscite}')
st.subheader("Grafico punteggi confidenza")

df = pd.DataFrame({"Indice": range(len(stats.punteggi_confidenza)), "Valore": stats.punteggi_confidenza})
fig = px.line(df, x="Indice", y="Valore", markers=True)
st.plotly_chart(fig)



