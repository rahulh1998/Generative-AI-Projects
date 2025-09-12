@ECHO OFF
@SET PYTHONIOENCODING=utf-8
@SET PYTHONUTF8=1
@FOR /F "tokens=2 delims=:." %%A in ('chcp') do for %%B in (%%A) do set "_CONDA_OLD_CHCP=%%B"
@chcp 65001 > NUL
@CALL "C:\Users\rahul\anaconda3\condabin\conda.bat" activate "F:\Generative-AI-Projects\genai_projects"
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@F:\Generative-AI-Projects\genai_projects\python.exe -Wi -m compileall -q -l -i C:\Users\rahul\AppData\Local\Temp\tmpwx9c004c -j 0
@IF %ERRORLEVEL% NEQ 0 EXIT /b %ERRORLEVEL%
@chcp %_CONDA_OLD_CHCP%>NUL
