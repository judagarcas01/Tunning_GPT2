# Fine-tuning manual de GPT-2 en español — Doctorado en Inteligencia Artificial

**Propósito:** descargar un GPT-2 preentrenado y reseñas auténticas de Amazon; realizar **full fine-tuning manual** primero; repetir el mismo experimento con **LoRA implementado a mano**; evaluar bajo idéntica partición de prueba. Diseñado para ordenadores de **16 GB RAM** sin exigir GPU.

> **Diferencia esencial:** este proyecto NO entrena GPT-2 desde cero. Los pesos preentrenados se descargan de Hugging Face; lo que se implementa desde cero es **el procedimiento de fine-tuning**, incluida pérdida causal, AdamW, loop de entrenamiento y capa LoRA. `transformers` se usa **solo** para cargar arquitectura/pesos/tokenizador reales; **no se usa** `Trainer`, `SFTTrainer`, `PEFT` ni `torch.optim` para actualizar pesos.
>
> Los resultados `--smoke` usan datos **sintéticos** y un Transformer **aleatorio diminuto**. SOLO comprueban el funcionamiento del software: **no constituyen resultados científicos ni verifican el entrenamiento real de GPT-2**.

## 1. Fuentes externas comprobables

- **Modelo:** [datificate/gpt2-small-spanish](https://huggingface.co/datificate/gpt2-small-spanish) (GPT-2 español, 12 capas, aproximadamente 124 M parámetros, licencia Apache-2.0, pesos PyTorch de unos 510 MB). El repositorio contiene `config.json`, `vocab.json`, `merges.txt` y `pytorch_model.bin`.
- **Dataset:** [SetFit/amazon_reviews_multi_es](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_es/tree/main). Archivos originales `train.jsonl` (200.000 reseñas), `validation.jsonl` (5.000) y `test.jsonl` (5.000). Campos reales `id`, `text`, `label`, `label_text`. **`label=0` significa una estrella; `label=4` significa cinco estrellas.** Solo utilizamos esas clases extremas y las tratamos como aproximación a sentimiento.
- [LoRA (Hu et al., ICLR 2022)](https://arxiv.org/abs/2106.09685); [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314); [Raschka: *Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch); [PyTorch: guardar checkpoints](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html); [Transformers: GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2).

Revisar condiciones de uso del dataset original antes de redistribuir textos. **El ZIP no contiene pesos ni reseñas reales:** se descargan en el equipo del estudiante para evitar distribuir material y cientos de MB.

## 2. Entorno e instalación paso a paso

Recomendado: **Python 3.11 o 3.12** y **PyTorch >=2.6** (requerido por las protecciones actuales de Transformers al cargar pesos `.bin`), CPU con 16 GB RAM, espacio libre **al menos 4 GB** para pesos/checkpoints/cachés. Windows PowerShell o Terminal de Linux/macOS:

```bash
# A. Descomprimir FineTuning_GPT2_Doctorado.zip y entrar a la carpeta
cd FineTuning_GPT2_Doctorado

# B. Crear y activar un entorno virtual
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS, EN LUGAR de la línea anterior:
# source .venv/bin/activate

# C. Instalar dependencias (necesita acceso a Internet)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# D. Comprobar recursos; sin descargar todavía el modelo
python scripts/00_check_environment.py
```

**Si Windows impide activar PowerShell**, abrir CMD y ejecutar `.venv\Scripts\activate.bat` (no hace falta cambiar las políticas del equipo). En Linux el ejecutable puede llamarse `python3`.

Los comandos siguientes deben ejecutarse desde la **raíz del proyecto** y con el entorno activado. Si falta una dependencia, instálela antes de continuar.

## 3. Primer paso RECOMENDADO: probar TODOS los archivos sin descargar GPT-2

```bash
python -m pytest -q tests
python scripts/00_check_environment.py --smoke
python scripts/01_download_model.py --smoke
python scripts/02_download_reviews.py --smoke
python scripts/03_prepare_data.py --smoke
python scripts/04_evaluate_baseline.py --smoke
python scripts/05_train_full.py --smoke
python scripts/06_train_lora.py --smoke
python scripts/07_evaluate_models.py --smoke
python scripts/08_compare_models.py --smoke
python scripts/09_predict.py --smoke --text "Producto de prueba muy malo"
```

El mismo proceso puede ejecutarse con **`python scripts/smoke_all.py`**. Genera datos inventados en `data/raw_smoke/`, modelo aleatorio en `models/base_smoke/` y gráficos en `results/smoke/`. Cada ejecución tarda habitualmente segundos en CPU. Nunca mezclar archivos `smoke` con experimentos reales.

**Pruebas realizadas durante la preparación:** 7 tests unitarios pasaron y los 10 scripts 00–09 terminaron con código de salida 0 en modo `--smoke`. También se probaron `--dry-run` de las descargas. **No se pudo probar con el checkpoint real de 510 MB** porque el entorno de preparación no tenía acceso a la red para instalar `transformers` ni descargar GPT-2. Debe verificarse en el equipo del estudiante después de instalarlo.

## 4. Ejecución REAL, siguiendo el orden pedagógico

### Fase 1. Descargar sin entrenar

```bash
python scripts/01_download_model.py --dry-run
python scripts/02_download_reviews.py --dry-run
python scripts/01_download_model.py
python scripts/02_download_reviews.py
```

La primera descarga necesita conexión. Se descargará exclusivamente la versión PyTorch (~510 MB) y los archivos del tokenizador (no TensorFlow/Flax). Las reseñas originales quedan en `data/raw/`, sin editar. Las descargas reutilizan la caché cuando están disponibles.

### Fase 2. Limpiar, seleccionar y tokenizar

```bash
python scripts/03_prepare_data.py
```

Por defecto: 250 positivas + 250 negativas en train; 50 + 50 en validation; 100 + 100 en test. El script descarta valoraciones 2–4 estrellas, normaliza espacios, deduplica entre particiones y conserva la negación y la puntuación. Genera `data/processed/{train,validation,test}.jsonl` y `data/tokenized/{train,validation,test}.jsonl`, junto con un manifiesto. **No se vuelven a separar aleatoriamente los datos originales:** se respetan train/validation/test de la fuente.

El campo `labels` es `-100` para los tokens del *prompt* y el padding; solo la respuesta (` POSITIVO` o ` NEGATIVO`, más EOS) contribuye a la pérdida supervisada. Se valida que no se trunque ninguna etiqueta.

### Fase 3. Establecer baseline **ANTES** de entrenar

```bash
python scripts/04_evaluate_baseline.py --limit 8
```

Mide el modelo preentrenado usando una tarea binaria generativa: calcula la suma de log-probabilidades condicionales de todos los tokens de cada etiqueta y elige la mayor. GPT-2 **no es un modelo instruction-tuned**, por lo que el baseline puede ser modesto y sensible al formato de instrucción. La puntuación de una etiqueta no debe confundirse con una clasificación probabilísticamente calibrada.

### Fase 4. Full fine-tuning MANUAL (primero)

```bash
# Prueba breve y real de forward/loss/backprop/AdamW/checkpoint
python scripts/05_train_full.py --steps 2

# Si la máquina responde bien: entrenamiento configurado
python scripts/05_train_full.py
```

**Advertencia:** la segunda ejecución comienza **otra vez desde el modelo base**, no continúa de los dos pasos anteriores. `--steps` indica *micro-batches* procesados, no actualizaciones. Con acumulación de gradientes de 4, **20 micro-pasos generan 5 actualizaciones**. La duración en CPU varía mucho, y el full fine-tuning puede tardar bastante; no hay promesa de duración fija.

El algoritmo manual está en `src/training/sft_loss.py`, `src/training/adamw_manual.py`, `src/training/training_loop.py`. Los pesos resultantes se guardan en `models/checkpoints/full/weights.pt`, **sin sobrescribir el GPT-2 preentrenado**. La optimización usa gradientes de autograd (`loss.backward()`), pero no delega el optimizador a `torch.optim`.

### Fase 5. LoRA MANUAL sobre EXACTAMENTE el modelo base

```bash
python scripts/06_train_lora.py --steps 2
python scripts/06_train_lora.py
```

No partimos del GPT-2 ya ajustado por full fine-tuning. Ambos tratamientos empiezan del **mismo checkpoint original**, usan idénticas observaciones y comparten pérdida y loop.

Se congela `W0` y se aprende **`W0 + (alpha/r) A B`** para la `c_attn` conjunta de GPT-2, cuya matriz se almacena `[d_in,3*d_out]`. **`models/checkpoints/lora/weights.pt` guarda solo A y B**; para inferencia hay que cargar también la base y reinyectar el adaptador con la configuración usada.

### Fase 6. Evaluación común y tablas de resultados

```bash
python scripts/07_evaluate_models.py --limit 8
python scripts/08_compare_models.py
python scripts/09_predict.py --text "No funciona, se rompió al segundo día"
```

En `results/real/` quedarán `baseline/`, `full/`, `lora/` y `comparison/`, con los archivos:

- `metrics.json`: accuracy, macro-F1, matrices de confusión y tamaño evaluado.
- `predictions.csv`: predicción por ID, referencia y scores de las dos etiquetas.
- `train_history.csv`: paso, época, entropía cruzada y segundos de entrenamiento.
- `training_summary.json`: tiempo, parámetros entrenables y pérdida de validación.
- `resource_usage.json`: pico de RAM RSS del proceso muestreado por micro-paso, y de VRAM (si procede).
- `run_metadata.json`: configuración y versiones locales de bibliotecas.
- `comparison/metrics_comparison.csv`, `performance.png`, `learning_curves.png`, `confusion_baseline.png`, `confusion_full.png` y `confusion_lora.png`.

**Para un test más amplio**, usar `--limit 0` (se evalúa todo `data/processed/test.jsonl`, 200 reseñas por defecto). Cuidado: la puntuación secuencial en CPU consume tiempo. Las métricas con `--limit 8` corresponden a una submuestra de 8 y **no sustentan conclusiones científicas**.

## 5. ¿Dónde cambiar cada elemento para hacer comparaciones?

**Modifique `configs/base.json`.** Los scripts importan ese archivo cada vez, así que normalmente no hay que cambiar el código. Para un caso especial, puede crear un archivo nuevo y usar `--config configs/mi_experimento.json` en **todos** los pasos relacionados.

| Pregunta experimental | Configuración / archivo | Qué ejecutar de nuevo |
|---|---|---|
| ¿Qué sucede con más reseñas? | `samples_per_class.train` en `configs/base.json` (250→500, 1000) | 03, 04, 05, 06, 07, 08 |
| ¿Y con un test más grande? | `--limit 0` o `eval_limit` | 04, 07, 08 |
| ¿Cómo influye el contexto? | `max_length` (128→192 o 256) | 03, 04, 05, 06, 07, 08 |
| ¿Cómo cambiar learning rate? | `full_learning_rate`, `lora_learning_rate` | 05, 06, 07, 08 |
| ¿Cómo variar iteraciones? | `max_steps` (20→40, 100) o CLI `--steps` | 05, 06, 07, 08 |
| ¿Cómo variar batch efectivo? | `batch_size` y `gradient_accumulation_steps` | 05, 06, 07, 08 |
| ¿Cómo estudiar rango de LoRA? | `lora_rank` (2, 4, 8, 16) | 06, 07, 08 |
| ¿Qué hace alpha y regularización? | `lora_alpha`, `lora_dropout` | 06, 07, 08 |
| ¿Qué capas LoRA se adaptan? | `target_modules`: `['c_attn']` o `['c_attn','c_proj']` | 06, 07, 08 |
| ¿Cómo congelar capas en full FT? | `scripts/05_train_full.py`, bloque `requires_grad_` | 05, 07, 08 |
| ¿Cómo cambiar prompt? | `src/data/preprocessing.py`, `format_prompt` | 03, 04, 05, 06, 07, 08; también misma función usada en inferencia |
| ¿Cómo cambiar las clases? | `src/data/cleaning.py` (`map_label`), `src/evaluation/label_scoring.py` y preparación | Revisar todos los scripts y tests; requiere nuevos experimentos |
| ¿Dónde está la ecuación de SFT? | `src/training/sft_loss.py` | tests, 05, 06, 07 |
| ¿Dónde está AdamW? | `src/training/adamw_manual.py` | test de equivalencia y volver a entrenar |
| ¿Dónde están las matrices A y B? | `src/models/lora_layer.py`, `lora_injection.py` | tests LoRA y volver a entrenar |

Para conservar los resultados de varios experimentos **cambie también `results_dir` y `checkpoints_dir`** a carpetas distintas en cada JSON antes de entrenar. Si no, la segunda ejecución sustituirá los checkpoints y resultados de la primera. Para un estudio serio, fijar las semillas y registrar versiones de PyTorch, Transformers, repositorios y hardware. En vez de `model_revision=main`, usar el **SHA de un commit** para anclar los pesos y evitar cambios futuros. Hacer lo mismo con `dataset_revision`.

## 6. Correspondencia entre teoría y código

**Función objetivo de SFT:** `src/training/sft_loss.py` computa

\[
\mathcal L(\theta)=-\frac{1}{M}\sum_{i,t}m_{it}\log p_\theta(y_{it}\mid x_i,y_{i,<t}),
\]

con `-100` para ignorar prompt/padding y un desplazamiento de un token de logits. `m_{it}` solo es uno para posiciones supervisadas.

**AdamW:** `src/training/adamw_manual.py` implementa `m_t`, `v_t`, corrección del sesgo y *weight decay* desacoplado. `tests/test_finetuning.py` compara numéricamente **una actualización** con la referencia `torch.optim.AdamW` solo dentro del test (nunca para entrenar).

**LoRA:** `src/models/lora_layer.py` realiza `base(x) + (dropout(x) @ A @ B)*(alpha/r)`. La base permanece congelada y B empieza en cero: por ello, la salida inicial equivale a la del modelo original, dentro de tolerancia. Inicialmente, el gradiente de A es cero y el de B puede ser no nulo. `tests/test_lora.py` comprueba esa equivalencia y la transmisión de gradientes.

**Clasificación generativa:** `src/evaluation/label_scoring.py` puntúa **todos** los tokens de una etiqueta; no presupone que `POSITIVO` sea un único token. La suma de log-probabilidades puede favorecer sistemáticamente secuencias de diferente longitud; si se desea contrastar promedios por token, documentar antes el cambio metodológico y mantener fijo el criterio del test. Se utiliza un **mismo prefijo truncado** para ambas etiquetas.

## 7. Restricciones y decisiones de diseño

1. **No hay `Trainer`, `SFTTrainer` ni PEFT.** Solo Hugging Face para descargar y restaurar GPT-2 y el tokenizador BPE. Cálculo de pérdida, AdamW, LoRA y métricas manuales.
2. **Solo CPU por defecto:** `device: "cpu"`. Si existe GPU CUDA, puede cambiar a `"cuda"`, ajustando parámetros para su memoria. No se implementa QLoRA en 4 bits en esta primera práctica.
3. **Checkpoints separados:** full guarda todos los pesos; LoRA guarda A/B. Cada nueva ejecución de entrenamiento arranca desde la **base**, no retoma exactamente momentos de AdamW, ni posición en el dataloader. Para reanudar bit a bit hay que ampliar el formato para incluir RNG, dataloader y estado del optimizador.
4. **Pérdida de entrenamiento no equivale a calidad en test.** Comparar accuracy/macro-F1 con datos no vistos, curvas de validación, semillas múltiples y presupuestos medidos.
5. **Pruebas de humo no demuestran mejora.** Un Transformer de juguete aleatorio puede obtener incluso 100 % sobre cuatro ejemplos sintéticos por azar; esas cifras NO se reportan como resultados empíricos.
6. **Tiempo/costo:** `max_steps` cuenta micro-batches; el batch efectivo es `batch_size * gradient_accumulation_steps` si el último grupo es completo. En CPU, los entrenamientos reales pueden durar mucho más que las pruebas de humo.
7. **No interpretar estrellas como sentimiento anotado:** se usa un proxy (1 vs 5 estrellas), no se deduce tono psicológico ni se hacen inferencias sobre usuarios.

## 8. Organización del repositorio

```text
FineTuning_GPT2_Doctorado/
├── configs/                # base.json y smoke.json; hiperparámetros editables
├── scripts/                # 00..09 y smoke_all.py; ejecutar desde la raíz
├── src/
│   ├── data/               # cleaning.py, sampling.py, preprocessing.py, dataloader.py
│   ├── models/             # load_gpt2.py, toy.py, lora_layer.py, lora_injection.py
│   ├── training/           # sft_loss.py, adamw_manual.py, training_loop.py, checkpoints.py
│   ├── evaluation/         # label_scoring.py, metrics.py, plots.py
│   └── utils/              # common.py, hardware.py, reproducibility.py, execution.py
├── tests/                  # tests matemáticos y de máscaras/LoRA
├── data/                   # se crea al ejecutar; archivos reales NO incluidos
├── models/                 # se crea al ejecutar; pesos reales NO incluidos
├── results/                # se crea al ejecutar; indicadores reales NO incluidos
├── requirements.txt
└── README.md
```

## 9. Mapa de archivos ejecutables

| Archivo | Cuándo usarlo | Comprobación de salida |
|---|---|---|
| `00_check_environment.py` | Al instalar | Python, torch, CPU/RAM, dependencias |
| `01_download_model.py` | Una vez | Archivos en `models/base/` |
| `02_download_reviews.py` | Una vez | 3 JSONL en `data/raw/` |
| `03_prepare_data.py` | Al cambiar muestreo/contexto | JSONL procesados y tokenizados, `data_manifest.json` |
| `04_evaluate_baseline.py` | Antes del entrenamiento | `results/real/baseline/metrics.json` |
| `05_train_full.py` | Primero: actualización total | `models/checkpoints/full/weights.pt` |
| `06_train_lora.py` | Después: adaptación eficiente | `models/checkpoints/lora/weights.pt` |
| `07_evaluate_models.py` | Después de ambos | Métricas y predicciones bajo idéntico test |
| `08_compare_models.py` | Al terminar | CSV, curvas, barras |
| `09_predict.py` | Para casos nuevos | Tres predicciones en consola |
| `smoke_all.py` | Primer diagnóstico rápido | Ejecuta todos en modo de juguete y genera informe |

### Problemas frecuentes

- **`torch.load` exige PyTorch 2.6 para pesos `.bin`**: actualizar PyTorch y Transformers siguiendo `requirements.txt` en lugar de desactivar las comprobaciones de seguridad.
- **`ModuleNotFoundError: transformers`**: `python -m pip install -r requirements.txt` desde el entorno activado. Las pruebas `--smoke` no requieren Transformers.
- **Error de acceso a Hugging Face**: revisar conexión, proxy institucional o descargar desde el repositorio con la web y conservar exactamente las rutas esperadas. No hay datos de ejemplo reales incluidos en el ZIP.
- **`max_length insuficiente`**: aumentar `max_length` en JSON, reejecutar 03; el sistema jamás debe entrenar sobre respuestas cortadas.
- **Memoria insuficiente en full FT**: bajar `max_length`, `batch_size=1` y `max_validation_batches`; mantener LoRA como demostración operativa, dejando full FT como extensión.
- **Checkpoints incompatibles LoRA**: mismo rango, alpha, capas y base usados durante entrenamiento; `07_evaluate_models.py` valida estructura antes de cargar.
- **Se sobrescribió un experimento**: establecer un directorio diferente en `results_dir` y `checkpoints_dir` antes de volver a ejecutar.

## 10. Pruebas pendientes antes de una demostración en clase

Aunque se ejecutaron todos los scripts en `--smoke`, en el equipo con Internet debe comprobarse con GPT-2 auténtico: **(a)** 01–03 reales; **(b)** 04 sobre 1–2 reseñas; **(c)** 05 y 06 con `--steps 2`; **(d)** 07–09 sobre 2 ejemplos; **(e)** tiempos/memoria para decidir cuántos pasos entran en la clase. Estos pasos con pesos reales **no pudieron ejecutarse en el entorno de preparación** y no debe asumirse su éxito hasta realizarlos.
