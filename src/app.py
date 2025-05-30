import streamlit as st
import os
import glob

def carica_file(nome_cartella, descrizione, tipi_file, accept_multiple_files=False):
    st.subheader(descrizione)
    # creazione della cartella se non esiste
    if not os.path.exists(nome_cartella):
        os.makedirs(nome_cartella)
    # caricamento file
    files = st.file_uploader(descrizione, type=tipi_file, accept_multiple_files=accept_multiple_files)
    if files:
        if not accept_multiple_files:
            files = [files]  # convertiamo il file singolo in lista per gestione uniforme
        for file in files:
            with open(os.path.join(nome_cartella, file.name), "wb") as f:
                f.write(file.getbuffer())
        st.success(f"File salvati correttamente nella cartella '{nome_cartella}'")

st.title("Modelli Face Recognition")
st.divider()
st.write("In questo sito potremo visualizzare le statistiche del modello di cui siamo interessati e confrontarle con un altro modello")

#usare multiselect per confronto
option = st.multiselect(
    label="Di quali modelli vuoi avere un resoconto?",
    options=["Auraface","Buffalo_l","Custom (ONNX)"],
    placeholder="Scegli un modello",
    max_selections=2,
)

#seleziona modello
if(option.__contains__("Custom (ONNX)")):
    carica_file("data/modello", "Carica il modello custom", "onnx")

#scegli se vuoi importare o usare i file default
custom = st.toggle("Vuoi usare dei file personalizzati?")

#importa file dell'utente
if custom:
    carica_file("data/video", "Carica il video sorgente", ["mp4", "mov", "avi"])
    carica_file("data/voli", "Carica i volti da tracciare", ["jpg", "jpeg", "png"], True)

#vai alla pagina dei risultati o del confronto, se possibile
if st.button("Vedi i risultati"):
    if (option.__contains__("Custom (ONNX)")):
        if glob.glob('data/modello/*.onnx'):  # Controlla se esiste un file .onnx
            if(len(option) == 1):
                st.switch_page("pages/risultato.py")
            else:
                st.switch_page("pages/confronto.py")
        else:
            st.warning("Devi caricare il modello custom prima di continuare")




