import time
from matplotlib import pyplot as plt
from sklearn.metrics import auc, classification_report, roc_curve
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd


start_time = time.time()
df = pd.read_csv('df_final_DL.csv')

train_texts, test_texts, train_labels, test_labels = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Tokenización con DistilBERT

# Se crea el tokenizador y se generan los encodings necesarios
""""
Los encodings son representaciones numericas de los tokens en el tweet. Se usan para crear los tensores que seran utilizados
para entrenar y evaluar el modelo
"""
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Crear conjuntos de datos de entrenamiento
""""
train_input_ids contiene los ids de tokens de los tweets y train_attention_mask es una máscara
que indica que tokens son de relleno y cuales son los válidos.
"""
train_input_ids = torch.tensor(train_encodings.input_ids)
train_attention_mask = torch.tensor(train_encodings.attention_mask)
train_labels.reset_index(drop=True, inplace=True) 
train_labels_tensor = torch.tensor(train_labels) # se pasan las etiquetas a tensores
train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels_tensor) # el conjunto de datos de entrenamiento final

# Crear conjuntos de datos de prueba
test_input_ids = torch.tensor(test_encodings.input_ids)
test_attention_mask = torch.tensor(test_encodings.attention_mask)
test_labels.reset_index(drop=True, inplace=True)
test_labels_tensor = torch.tensor(test_labels)

test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels_tensor) # conjunto de datos de prueba final

# Se crean los cargadores de datos
"""
Sirven para manejar eficientemente los datos durante el entrenamiento y la evaluación. Permite el batching, shuffling, 
parallelizing entre otras fucnciones
"""
train_loader = DataLoader(train_dataset, batch_size=8, sampler=RandomSampler(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=8, sampler=SequentialSampler(test_dataset))

# Modelo DistilBERT y optimizador
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # o se usa la GPU o la CPU (GPU es mejor)
model.to(device) # se transfiere el modelo a ese 'dispositivo'

# Entrenamiento del modelo 
for epoch in range(2):
    model.train() # el modelo se pone en modo entrenamiento
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids.to(device), attention_mask.to(device), labels=labels.to(device)) # se envian los datos al device para realizar los cálculos
        loss = outputs.loss
        loss.backward()
        optimizer.step() # se actualizan los params del modelo basándose en los gradientes

# Evaluación del modelo
model.eval() # se pone el modelo en modo evaluación
preds_list = []
labels_list = []
for batch in test_loader:
    with torch.no_grad():
        input_ids, attention_mask, labels = batch # se extraen los datos de entrada
        outputs = model(input_ids.to(device), attention_mask.to(device))
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1) # se determina la clase predicha
        preds_list.extend(preds.cpu().numpy())
        labels_list.extend(labels.cpu().numpy())

# Classification Report
print(classification_report(labels_list, preds_list))

# Se calcula la probabilidad de predecir la label. (Una forma de evaluación como la anterior pero para la curva roc)
preds_proba = [] # lista para almacenar las probabilidades predichas para la clase positiva.
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids.to(device), attention_mask.to(device))
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1) # Se calculan las probabilidades de clase utilizando una función de activación softmax sobre las salidas del modelo.
        preds_proba.extend(probs[:, 1].cpu().numpy())  # Se añaden a la lista


fpr, tpr, thresholds = roc_curve(labels_list, preds_proba)

# Curva ROC
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc(fpr, tpr)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

# Save the ROC curve as a JPG image
plt.savefig('roc_curve_DISTILBERT.jpg')

end_time = time.time()
execution_time = (end_time - start_time)/60
print("Tiempo de ejecución: {:.2f} minutos".format(execution_time))