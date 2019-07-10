from django_app.settings import *

MODEL_PATH = os.path.join(BASE_DIR, "ewc_app/data/models")
DATA_PATH = os.path.join(BASE_DIR, "ewc_app/data/data_sentence_intent/corrected_oversampled_data.pkl")
VOCAB_PATH = os.path.join(BASE_DIR, "ewc_app/data/vocabulary/fasttext_vocab_fr.dat")
EMBEDDING_PATH = os.path.join(BASE_DIR, "ewc_app/data/embeddings/fasttext_embedding_fr.npy")

BATCH_SPLITTING = 25600
ONLINE_SPLITTING = 31328

INTENTS = ['annulation', 'bot_activities_together', 'bot_age',
       'bot_compliment', 'bot_distraction', 'bot_gender', 'bot_hobby',
       'bot_inlove', 'bot_is_bot', 'bot_localization', 'bot_name',
       'bot_ping', 'bot_skills', 'bot_time',
       'caracteristiques_timbre_personnalisation',
       'caracteristiques_timbre_scope', 'caracteristiques_timbre_suivi',
       'collage_comment_coller', 'collage_comment_decouper',
       'disposition_planche_disposition', 'disposition_planche_format',
       'disposition_planche_plusieurs_timbres', 'dymo',
       'emotion_birthday', 'emotion_disinterest', 'emotion_happy',
       'emotion_howzit', 'emotion_negative', 'emotion_thanks',
       'escalade_sc', 'impression_comment_reimprimer',
       'impression_hardware', 'impression_specimen', 'pdf_ou', 'pdf_pb',
       'process_general_demarche', 'process_general_depot',
       'process_general_tarif', 'services_la_poste_colis',
       'services_la_poste_guichet', 'services_la_poste_marianne',
       'services_la_poste_pro', 'services_la_poste_recommande',
       'support_achat', 'support_enveloppe_adresse',
       'support_enveloppe_disposition', 'support_enveloppe_information',
       'support_liste', 'transaction_code_promo',
       'transaction_compte_prepaye', 'transaction_facture',
       'transaction_paiement', 'transaction_probleme_paiement',
       'validite_code_barres', 'validite_duree', 'validite_noir_et_blanc',
       'validite_placement', 'validite_qualite_format_decalage',
       'validite_reimp_valid']

