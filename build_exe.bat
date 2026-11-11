@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv-build\Scripts\python.exe" (
  python -m venv .venv-build
  if errorlevel 1 exit /b 1
)

".venv-build\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

".venv-build\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 exit /b 1

".venv-build\Scripts\pyinstaller.exe" --noconfirm --clean --onedir --name VisionForensics ^
  --add-data "app.py;." ^
  --add-data "modules;modules" ^
  --collect-data streamlit ^
  --collect-data torch ^
  --collect-data torchvision ^
  --collect-data mediapipe ^
  --collect-data matplotlib ^
  --collect-data cv2 ^
  --collect-data certifi ^
  --exclude-module torch.testing ^
  --exclude-module torch._inductor ^
  --exclude-module torch.distributed ^
  --exclude-module torch.onnx ^
  --exclude-module torch.utils.tensorboard ^
  --exclude-module matplotlib.tests ^
  --exclude-module matplotlib.testing ^
  --exclude-module mediapipe.tasks.python.test ^
  --hidden-import streamlit.web.cli ^
  --hidden-import mediapipe.tasks.python ^
  --hidden-import mediapipe.tasks.python.vision ^
  --hidden-import mediapipe.tasks.python.vision.face_landmarker ^
  --hidden-import cv2 ^
  --hidden-import matplotlib.pyplot ^
  --hidden-import matplotlib.backends.backend_agg ^
  --hidden-import transformers.pipelines ^
  --hidden-import transformers.pipelines.image_classification ^
  --hidden-import transformers.models.auto.configuration_auto ^
  --hidden-import transformers.models.auto.modeling_auto ^
  --hidden-import transformers.models.vit.configuration_vit ^
  --hidden-import transformers.models.vit.image_processing_vit ^
  --hidden-import transformers.models.vit.modeling_vit ^
  --hidden-import modules.metadata ^
  --hidden-import modules.fft ^
  --hidden-import modules.noise ^
  --hidden-import modules.jpeg ^
  --hidden-import modules.face ^
  --hidden-import modules.report ^
  --hidden-import modules.watermark ^
  --hidden-import modules.ela ^
  --hidden-import modules.ai_classifier ^
  run_app.py
if errorlevel 1 exit /b 1

echo.
echo Build complete: dist\VisionForensics\VisionForensics.exe