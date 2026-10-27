# Informe de verificación del proyecto

## Pruebas efectivamente ejecutadas

Se realizaron estas pruebas en el entorno de preparación:

| Verificación | Resultado |
|---|---|
| Compilación estática de scripts, módulos y tests (`python -m compileall`) | Correcta |
| Tests unitarios con `pytest -q` | **7 pruebas aprobadas** |
| `00_check_environment.py --smoke` | Correcto (exit 0) |
| `01_download_model.py --smoke` | Correcto (exit 0) |
| `02_download_reviews.py --smoke` | Correcto (exit 0) |
| `03_prepare_data.py --smoke` | Correcto (exit 0) |
| `04_evaluate_baseline.py --smoke` | Correcto (exit 0) |
| `05_train_full.py --smoke` | Correcto (exit 0; entrenamiento toy) |
| `06_train_lora.py --smoke` | Correcto (exit 0; entrenamiento toy) |
| `07_evaluate_models.py --smoke` | Correcto (exit 0) |
| `08_compare_models.py --smoke` | Correcto (exit 0; CSV y gráficos generados) |
| `09_predict.py --smoke` | Correcto (exit 0) |
| Descarga del modelo y dataset con `--dry-run` | Correcta (rutas y repositorios, SIN descarga de red) |

Además se comprobó que el adaptador arranca sin alterar el forward de la base, que solo sus parámetros reciben gradiente, que la pérdida respeta la máscara de prompt y que una actualización del optimizador manual coincide con la actualización de referencia en un ejemplo pequeño.

El archivo `smoke_report.json` contiene los tiempos y códigos de salida de las diez etapas. Está disponible en `results/smoke/` dentro del ZIP como evidencia de la prueba integrada.

## Qué NO se ha comprobado

- **No se descargaron los 510 MB reales del modelo GPT-2** ni los JSONL completos: en el entorno de verificación no se pudieron instalar `transformers` ni acceder a Hugging Face desde Python.
- No se ejecutó `AutoModelForCausalLM.from_pretrained` contra los pesos reales; **la compatibilidad final con el checkpoint debe verificarse en el ordenador con Internet**.
- No se midieron tiempos de full fine-tuning de GPT-2 sobre CPU de 16 GB; dependen del hardware del estudiante.
- La precisión `accuracy` sobre cuatro reseñas **sintéticas** no demuestra capacidad de clasificación ni mejora por fine-tuning.

Para realizar estas verificaciones pendientes, consulte las **Fases 1–6 del README**. Empiece por `--steps 2` y `--limit 2` y amplíe solo después de comprobar tiempos y memoria.
