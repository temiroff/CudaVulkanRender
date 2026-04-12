@echo off
setlocal
set "MAYAUSD=C:\Program Files\Autodesk\MayaUSD\Maya2025\0.29.0\mayausd\USD"
set "MAYA=C:\Program Files\Autodesk\Maya2025"
set "PATH=%MAYAUSD%\lib;%MAYAUSD%\bin;%MAYA%\bin;%MAYA%\Python\Lib\site-packages\shiboken6;%MAYA%\Python\Lib\site-packages\PySide6;%PATH%"
set "PYTHONPATH=%MAYAUSD%\lib\python"
set "PXR_PLUGINPATH_NAME=%MAYAUSD%\plugin\usd"
"%MAYA%\bin\mayapy.exe" "%MAYAUSD%\bin\usdview" %*
