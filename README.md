# Bot para completar Google Forms con Ollama local

Este script abre el formulario, consulta a tu Ollama local y marca las respuestas automáticamente.
También incluye un modo de simulación local para generar muchas respuestas de prueba sin enviar nada al formulario real.

## Requisitos

- Python 3
- Ollama corriendo en `http://127.0.0.1:11434`
- El modelo `qwen2.5-coder:7b` disponible, o cambia el parámetro `--model`

## Instalación

```bash
cd /Users/rafael/tmp/bot
/opt/homebrew/bin/python3 -m pip install -r requirements.txt
/opt/homebrew/bin/python3 -m playwright install chromium
```

## Uso

Rellenar y dejar una captura local sin enviar:

```bash
/opt/homebrew/bin/python3 bot_formulario.py
```

Rellenar y enviar automáticamente:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --submit
```

Enviar 200 formularios automáticamente:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --submit --submit-count 200
```

Enviar 200 formularios con pausa de 1.5 segundos entre cada envío:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --submit --submit-count 200 --submit-delay 1.5
```

Simular 200 respuestas sin abrir ni enviar el formulario:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --simulate 200
```

Guardar la simulación en otro archivo:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --simulate 200 --output pruebas/simulacion.jsonl
```

Cambiar la persona que responde:

```bash
/opt/homebrew/bin/python3 bot_formulario.py --persona "Soy un estudiante de universidad interesado en tecnología y datos."
```

La IA local puede:

- Generar respuestas de prueba con distintas personas.
- Repetir simulaciones sin tocar el formulario público.
- Rellenar el formulario solo cuando uses `--submit` de forma manual.
- Guardar resultados de prueba en un archivo `.jsonl` para revisarlos luego.

## Nota

No necesitas ejecutar `ollama serve` otra vez si el puerto `11434` ya está ocupado; eso suele significar que Ollama ya está levantado.
