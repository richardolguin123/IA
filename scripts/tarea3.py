import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# 1. Configuración y Hiperparámetros


BATCH_SIZE = 32
IMG_SIZE = 64
LEARNING_RATE = 0.001
SEED = 42  # Semilla para reproducibilidad (importante para la nota)

# Fijamos la semilla en CPU y GPU
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)


# 2. Preparación de Datos

print("--- Preparando Datos ---")

# Transformaciones: Convertir a Tensor y Normalizar
# Normalizamos con media 0.5 y desv 0.5 para escalar valores entre -1 y 1
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Ruta relativa: Asumimos que estás ejecutando desde 'IA/scripts' o 'IA'
# Buscamos la carpeta Imagenes subiendo un nivel si es necesario
if os.path.exists('../Imagenes'):
    data_dir = '../Imagenes'
elif os.path.exists('IA/Imagenes'): # Por si ejecutas desde fuera
    data_dir = 'IA/Imagenes'
else:
    # Ajusta esto si tu carpeta tiene otro nombre
    data_dir = './Imagenes' 

try:
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    print(f" Dataset encontrado. Clases: {full_dataset.classes}")
except Exception as e:
    print(f" Error: No se encontró el dataset en '{data_dir}'. Verifica la ruta.")
    exit()

# División del Dataset (70% Train, 15% Val, 15% Test)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

print(f" Distribución: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

# Dataloaders: Cargadores que entregan los datos por lotes (batches)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 3. Arquitectura CNN Base

class SatelliteCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SatelliteCNN, self).__init__()
        
        # Bloque 1: Captura características simples (bordes, colores)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduce a 32x32
        
        # Bloque 2: Captura formas más complejas
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduce a 16x16
        
        # Bloque 3: Captura texturas detalladas (bosque vs cultivo)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduce a 8x8
        
        # Clasificador (Fully Connected)
        self.flatten = nn.Flatten()
        # Entrada: 128 canales * 8 * 8 pixeles
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes) # Salida: 5 clases

    def forward(self, x):
        # Pasada por bloques convolucionales
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Aplanado y clasificación
        x = self.flatten(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

# Prueba rápida de que la arquitectura funciona
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SatelliteCNN().to(device)
    print("\n Arquitectura del Modelo Base creada:")
    print(model)
    
    # Simular una imagen para probar dimensiones
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    output = model(dummy_input)
    print(f"\n Prueba de paso hacia adelante (Forward pass) exitosa.")
    print(f"   Salida: {output.shape} (Debe ser [1, 5])")