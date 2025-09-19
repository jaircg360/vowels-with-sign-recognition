Mediapipe Vowels - Windows
==========================

Backend (Python / FastAPI)
--------------------------
1. Abrir PowerShell y navegar a la carpeta backend:
   cd backend

2. Crear y activar entorno virtual:
   venv\Scripts\activate


3. Instalar dependencias:
   pip install -r requirements.txt

4. Ejecutar el servidor:
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Frontend (React)
----------------
1. Abrir otra terminal y navegar a la carpeta frontend:
   cd frontend

2. Instalar dependencias:
   npm install

3. Iniciar la app:
   npm start

Uso
---
- Abre http://localhost:3000 en tu navegador.
- Permite el acceso a la cámara.
- Verás el video con landmarks dibujados.
- Captura muestras seleccionando la etiqueta y pulsando "Capturar muestra".
- Entrena un modelo y luego pruébalo con el botón "Probar" en la lista.
