class Persona:
    def __init__(self, nome, path_immagine):
        self.nome = nome
        self.pathImmagine = path_immagine
        self.id = None
        self.list_ID = []
        self.confidence_scores = []
        self.identification_count = 0

    def update(self, new_ID, confidence=None):
        self.list_ID.append(new_ID)
        self.id = new_ID
        self.identification_count += 1

        if confidence is not None:
            self.confidence_scores.append(confidence)

    def get_statistics(self):
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        return {
            'name': self.nome,
            'current_id': self.id,
            'total_ids': len(self.list_ID),
            'identification_count': self.identification_count,
            'average_confidence': avg_confidence,
            'all_ids': self.list_ID.copy()
        }

    def __str__(self) -> str:
        id_history = f"[{', '.join(map(str, self.list_ID[-3:]))}]" if len(self.list_ID) > 0 else "[]"
        avg_conf = sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0
        return f'{self.nome} | Current ID: {self.id} | IDs: {id_history} | Identifications: {self.identification_count} | Avg Conf: {avg_conf:.3f}'

    def __repr__(self) -> str:
        return self.__str__()