import time
from sklearn.metrics import classification_report, roc_curve, auc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer

# Definir la arquitectura CNN
class CNNTextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim, dropout):
        super(CNNTextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        dense = nn.functional.relu(self.fc(cat))
        return self.fc_out(dense)

# Función para entrenamiento
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        text, labels = batch
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Función para evaluación
def evaluate(model, iterator, criterion):
    epoch_loss = 0
    preds_list = []
    labels_list = []
    probs_list = []
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            predictions = model(text)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            preds = torch.argmax(predictions, dim=1)
            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())
            probs = nn.functional.softmax(predictions, dim=1)
            probs_list.extend(probs[:, 1].cpu().numpy())  # Solo tomamos la probabilidad de la clase positiva
    return epoch_loss / len(iterator), preds_list, labels_list, probs_list

# Lectura de datos y preprocesamiento
start_time = time.time()
df = pd.read_csv('df_final_DL.csv')
train_texts, test_texts, train_labels, test_labels = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Tokenización con DistilBERT (solo para obtener el tamaño del vocabulario)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
vocab_size = len(tokenizer)

# Definir los hiperparámetros
"""
Este valor representa la dimensión de los vectores de incrustación. Aumentar la dimensión de incrustación puede permitir al modelo capturar 
características más finas de las palabras, pero también aumentará la complejidad del modelo y el tiempo de entrenamiento.
"""
embedding_dim = 1000 
"""
Este valor determina el número de filtros convolucionales en la capa CNN. Más filtros pueden permitir al modelo capturar una mayor variedad
de características de los datos de entrada, lo que puede ser beneficioso si el conjunto de datos es complejo.
"""
num_filters = 200
"""
Estos son los tamaños de los filtros convolucionales. Experimentar con diferentes tamaños de filtro puede ayudar al modelo a 
capturar diferentes longitudes de n-gramas en los datos de entrada.
"""
filter_sizes = [2, 3]
"""
Este valor representa la dimensión de la capa oculta en la red neuronal completamente conectada. Aumentar este valor puede permitir al modelo 
aprender representaciones más complejas de los datos, pero también aumentará la complejidad del modelo y el tiempo de entrenamiento.
"""
hidden_dim = 1024
output_dim = 2
"""
Este valor controla la tasa de abandono en el modelo. El aumento de la tasa de abandono puede ayudar a regularizar el modelo y prevenir 
el sobreajuste, pero demasiado abandono puede afectar negativamente al rendimiento del modelo.
"""
dropout = 0.5
batch_size = 8
epochs = 10
learning_rate = 1e-4

# Crear conjuntos de datos de entrenamiento y prueba
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

train_input_ids = torch.tensor(train_encodings.input_ids)
train_labels.reset_index(drop=True, inplace=True)
train_labels_tensor = torch.tensor(train_labels)
train_dataset = TensorDataset(train_input_ids, train_labels_tensor)

test_input_ids = torch.tensor(test_encodings.input_ids)
test_labels_array = test_labels.to_numpy()
test_labels_tensor = torch.tensor(test_labels_array)
test_dataset = TensorDataset(test_input_ids, test_labels_tensor)

# Crear cargadores de datos
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Instanciar modelo, optimizador y función de pérdida
model = CNNTextClassifier(vocab_size, embedding_dim, num_filters, filter_sizes, hidden_dim, output_dim, dropout)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Entrenamiento del modelo
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}')

# Evaluación del modelo
test_loss, preds_list, labels_list, probs_list = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.3f}')
print(classification_report(labels_list, preds_list))

# Curva ROC
fpr, tpr, _ = roc_curve(labels_list, probs_list)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

end_time = time.time()
execution_time = (end_time - start_time) / 60
print("Tiempo de ejecución: {:.2f} minutos".format(execution_time))
