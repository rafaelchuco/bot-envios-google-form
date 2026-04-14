#!/usr/bin/env python3
"""Rellena el formulario de Google Forms usando respuestas generadas por Ollama local.

El flujo es:
1. Abrir el formulario.
2. Pedir a Ollama una respuesta por pregunta.
3. Marcar las opciones y escribir el texto.
4. Opcionalmente enviar el formulario.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSf0zZIKAnrxFQD096ru7ADsosBMUbAgZx8ZwWraG7i1yJ20hQ/viewform?usp=dialog"


@dataclasses.dataclass(frozen=True)
class QuestionSpec:
    key: str
    prompt: str
    kind: str
    options: tuple[str, ...] = ()


QUESTIONS: tuple[QuestionSpec, ...] = (
    QuestionSpec(
        key="q1",
        prompt="¿Has tenido dudas sobre qué carrera estudiar?",
        kind="single",
        options=("Si", "No"),
    ),
    QuestionSpec(
        key="q2",
        prompt="¿Consideras difícil elegir una carrera profesional?",
        kind="single",
        options=("Si", "No", "Tal vez"),
    ),
    QuestionSpec(
        key="q3",
        prompt="¿Has sentido inseguridad o ansiedad sobre tu futuro profesional?",
        kind="single",
        options=("Sí", "No", "Tal vez"),
    ),
    QuestionSpec(
        key="q4",
        prompt="¿Has utilizado herramientas o tests vocacionales anteriormente?",
        kind="single",
        options=("Si", "No"),
    ),
    QuestionSpec(
        key="q5",
        prompt="¿Qué tan interesado estarías en usar una plataforma que te recomiende carreras usando IA y analice tus habilidades?",
        kind="scale",
        options=("1", "2", "3", "4", "5"),
    ),
    QuestionSpec(
        key="q6",
        prompt="¿Qué función te parece más útil?",
        kind="single",
        options=(
            "Test vocacional inteligente",
            "Recomendación de carreras con explicación",
            "Ruta de aprendizaje por ciclos",
            "Seguimiento de progreso",
            "Recursos y cursos sugeridos",
        ),
    ),
    QuestionSpec(
        key="q7",
        prompt="¿Cuánto estarías dispuesto a pagar por esta plataforma?",
        kind="single",
        options=("Gratis", "S/ 5 - S/ 10", "S/ 10 - S/ 20", "Más de S/ 20"),
    ),
    QuestionSpec(
        key="q8",
        prompt="¿Con qué frecuencia usarías una plataforma así?",
        kind="single",
        options=("Una vez", "Semanalmente", "Mensualmente", "Solo cuando tenga dudas"),
    ),
    QuestionSpec(
        key="q9",
        prompt="¿En qué etapa te encuentras?",
        kind="single",
        options=("Colegio", "Preuniversitario", "Instituto", "Universidad", "Otros:"),
    ),
    QuestionSpec(
        key="q10",
        prompt="¿Qué tan seguro estás de tu elección de carrera actual?",
        kind="scale",
        options=("1", "2", "3", "4", "5"),
    ),
    QuestionSpec(
        key="q11",
        prompt="¿Preferirías usar esta plataforma en?",
        kind="single",
        options=("Web", "App móvil", "Ambos"),
    ),
    QuestionSpec(
        key="q12",
        prompt="¿Qué te gustaría que tenga esta plataforma para ayudarte mejor?",
        kind="text",
    ),
)

TEXT_VARIANTS: tuple[str, ...] = (
    "Me gustaría que incluya recomendaciones personalizadas y ejemplos reales de campo laboral.",
    "Sería útil tener una guía clara por etapas y sugerencias según mis fortalezas.",
    "Me ayudaría que muestre carreras afines y rutas de estudio concretas.",
    "Quisiera que integre test dinámicos y seguimiento de avance personal.",
    "Me serviría ver opciones de becas, cursos y proyección salarial por carrera.",
)


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip().casefold()


def ollama_chat(model: str, persona: str) -> dict[str, Any]:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
        "messages": [
            {
                "role": "system",
                "content": (
                    "Responde encuestas en español. Devuelve solo JSON válido sin markdown. "
                    "Las respuestas deben ser realistas y consistentes con la persona dada."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(persona),
            },
        ],
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise SystemExit(
            "No pude conectar con Ollama en http://127.0.0.1:11434. "
            "Revisa que el servidor local esté activo."
        ) from exc

    payload = json.loads(raw)
    content = payload["message"]["content"]
    return parse_json_object(content)


def build_prompt(persona: str) -> str:
    questions = []
    for question in QUESTIONS:
        block = {
            "id": question.key,
            "pregunta": question.prompt,
            "tipo": question.kind,
        }
        if question.options:
            block["opciones"] = list(question.options)
        questions.append(block)

    return (
        "Persona a imitar:\n"
        f"{persona.strip()}\n\n"
        "Responde este formulario con una salida JSON de esta forma exacta:\n"
        '{"answers": {"q1": "...", "q2": "..."}}\n\n'
        "Reglas:\n"
        "- Devuelve una clave por cada pregunta.\n"
        "- Para preguntas de opción única, usa exactamente una de las opciones dadas.\n"
        "- Para preguntas de escala, usa solo un número como cadena, por ejemplo "
        '"4".\n'
        "- Para la pregunta de texto, escribe 1 o 2 frases en español, concretas y naturales.\n"
        "- No añadas explicaciones extra.\n\n"
        "Preguntas:\n"
        f"{json.dumps(questions, ensure_ascii=False, indent=2)}"
    )


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise SystemExit(f"Ollama no devolvió JSON válido: {text}")
        return json.loads(match.group(0))


def choose_answer(question: QuestionSpec, raw_answer: Any) -> str:
    if question.kind == "text":
        answer = str(raw_answer).strip()
        return answer or "Me gustaría una plataforma práctica, clara y personalizada."

    answer = str(raw_answer).strip()
    normalized_answer = normalize_text(answer)

    for option in question.options:
        if normalized_answer == normalize_text(option):
            return option

    if question.kind == "scale":
        if answer in question.options:
            return answer
        raise ValueError(f"Respuesta de escala inválida para {question.key}: {answer}")

    for option in question.options:
        if normalized_answer in normalize_text(option) or normalize_text(option) in normalized_answer:
            return option

    raise ValueError(f"Respuesta inválida para {question.key}: {answer}")


def find_question_block(page, prompt: str):
    blocks = page.locator('div[role="listitem"]')
    count = blocks.count()
    normalized_prompt = normalize_text(prompt)

    for index in range(count):
        block = blocks.nth(index)
        try:
            text = block.inner_text(timeout=1000)
        except PlaywrightTimeoutError:
            continue
        if normalize_text(text).find(normalized_prompt) != -1:
            return block

    raise SystemExit(f"No encontré la pregunta en el formulario: {prompt}")


def click_choice(locator, answer: str) -> bool:
    try:
        locator.get_by_role("radio", name=answer, exact=True).click()
        return True
    except PlaywrightTimeoutError:
        pass

    try:
        locator.get_by_role("checkbox", name=answer, exact=True).click()
        return True
    except PlaywrightTimeoutError:
        pass

    for role in ("radio", "checkbox"):
        options = locator.get_by_role(role)
        for index in range(options.count()):
            option = options.nth(index)
            label = normalize_text(option.get_attribute("aria-label") or option.inner_text())
            if normalize_text(answer) == label or normalize_text(answer) in label or label in normalize_text(answer):
                option.click()
                return True

    return False


def fill_choice(block, answer: str) -> None:
    if click_choice(block, answer):
        # Si se elige "Otros:", Google Forms suele exigir un texto adicional.
        if normalize_text(answer).startswith("otros"):
            other_field = block.locator('input[type="text"]')
            if other_field.count() > 0:
                other_field.first.fill("Otro")
        return

    raise SystemExit(f"No pude seleccionar la opción '{answer}'.")


def fill_scale(block, answer: str) -> None:
    if click_choice(block, answer):
        return

    fill_choice(block, answer)


def fill_text(block, answer: str) -> None:
    field = block.locator('textarea, input[type="text"]')
    if field.count() == 0:
        raise SystemExit("No encontré un campo de texto para la pregunta abierta.")
    field.first.fill(answer)


def fill_form(page, answers: dict[str, Any]) -> None:
    for question in QUESTIONS:
        block = find_question_block(page, question.prompt)
        if question.key not in answers:
            raise SystemExit(f"Falta la respuesta para {question.key}.")

        answer = choose_answer(question, answers[question.key])
        if question.kind == "single":
            fill_choice(block, answer)
        elif question.kind == "scale":
            fill_scale(block, answer)
        elif question.kind == "text":
            fill_text(block, answer)
        else:
            raise SystemExit(f"Tipo de pregunta no soportado: {question.kind}")


def list_button_labels(page) -> list[str]:
    labels: list[str] = []
    buttons = page.get_by_role("button")
    for index in range(buttons.count()):
        button = buttons.nth(index)
        label = (button.get_attribute("aria-label") or "").strip()
        if not label:
            try:
                label = button.inner_text(timeout=800).strip()
            except PlaywrightTimeoutError:
                label = ""
        if label:
            labels.append(label)
    return labels


def click_submit_or_next(page) -> None:
    submit_pattern = re.compile(r"^\s*(Enviar|Submit|Send)\s*$", re.IGNORECASE)
    next_pattern = re.compile(r"^\s*(Siguiente|Next)\s*$", re.IGNORECASE)

    for _ in range(5):
        submit_button = page.get_by_role("button", name=submit_pattern)
        if submit_button.count() > 0:
            submit_button.first.click(timeout=15000)
            return

        next_button = page.get_by_role("button", name=next_pattern)
        if next_button.count() > 0:
            next_button.first.click(timeout=15000)
            page.wait_for_load_state("networkidle")
            continue

        break

    labels = list_button_labels(page)
    labels_preview = ", ".join(sorted(set(labels))) if labels else "ninguno"
    raise SystemExit(
        "No encontré un botón de envío. "
        f"Botones detectados: {labels_preview}"
    )


def wait_submission_confirmation(page) -> None:
    try:
        page.wait_for_url(re.compile(r"/formResponse"), timeout=15000)
        return
    except PlaywrightTimeoutError:
        pass

    confirmation_patterns = (
        re.compile(r"se registr[oó] tu respuesta", re.IGNORECASE),
        re.compile(r"gracias", re.IGNORECASE),
        re.compile(r"your response has been recorded", re.IGNORECASE),
    )
    for pattern in confirmation_patterns:
        locator = page.get_by_text(pattern)
        try:
            locator.first.wait_for(timeout=5000)
            return
        except PlaywrightTimeoutError:
            continue

    raise SystemExit(
        "No pude confirmar que el formulario se haya enviado. "
        "Revisa si el formulario requiere campos adicionales."
    )


def extract_answers(response: dict[str, Any]) -> dict[str, Any]:
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise SystemExit(f"La respuesta de Ollama no tiene la estructura esperada: {response}")
    return answers


def diversify_answers(
    answers: dict[str, Any],
    *,
    majority_ratio: float,
    rng: random.Random,
) -> dict[str, Any]:
    """Mantiene una mayoría de respuestas base y cambia una minoría para variar."""
    diversified: dict[str, Any] = {}

    for question in QUESTIONS:
        raw_answer = answers.get(question.key)
        base_answer = choose_answer(question, raw_answer)

        if rng.random() <= majority_ratio:
            diversified[question.key] = base_answer
            continue

        if question.kind == "text":
            alternatives = [text for text in TEXT_VARIANTS if normalize_text(text) != normalize_text(base_answer)]
        else:
            alternatives = [
                option
                for option in question.options
                if normalize_text(option) != normalize_text(base_answer)
            ]

        diversified[question.key] = rng.choice(alternatives) if alternatives else base_answer

    return diversified


def generate_answers(
    model: str,
    persona: str,
    *,
    majority_ratio: float,
    rng: random.Random,
) -> dict[str, Any]:
    response = ollama_chat(model, persona)
    answers = extract_answers(response)
    return diversify_answers(answers, majority_ratio=majority_ratio, rng=rng)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rellena un Google Form con respuestas generadas por Ollama.")
    parser.add_argument("--url", default=FORM_URL, help="URL del formulario")
    parser.add_argument("--model", default="qwen2.5-coder:7b", help="Modelo local de Ollama")
    parser.add_argument(
        "--persona",
        default=(
            "Eres un estudiante de secundaria/preuniversitario que todavía tiene dudas sobre su carrera. "
            "Quieres una plataforma útil, clara y con recomendaciones personalizadas."
        ),
        help="Descripción de la persona cuyas respuestas generará la IA",
    )
    parser.add_argument("--headless", action="store_true", help="Ejecutar el navegador en modo headless")
    parser.add_argument("--submit", action="store_true", help="Enviar el formulario al final")
    parser.add_argument(
        "--submit-count",
        type=int,
        default=1,
        help="Cantidad de formularios a enviar cuando se usa --submit",
    )
    parser.add_argument(
        "--submit-delay",
        type=float,
        default=0.0,
        help="Pausa en segundos entre envíos (solo con --submit)",
    )
    parser.add_argument(
        "--simulate",
        type=int,
        default=0,
        help="Generar N respuestas de prueba sin abrir el formulario ni enviarlo",
    )
    parser.add_argument(
        "--output",
        default="simulacion_respuestas.jsonl",
        help="Archivo de salida para --simulate",
    )
    parser.add_argument(
        "--majority-ratio",
        type=float,
        default=1.0,
        help=(
            "Porcentaje de respuestas que se mantienen como la respuesta base. "
            "Ejemplo: 0.8 = 80%% base y 20%% alternativas."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla opcional para reproducir la variación aleatoria",
    )
    args = parser.parse_args()

    if args.simulate < 0:
        raise SystemExit("--simulate debe ser 0 o un número positivo.")

    if args.submit_count < 1:
        raise SystemExit("--submit-count debe ser 1 o mayor.")

    if args.submit_delay < 0:
        raise SystemExit("--submit-delay debe ser 0 o mayor.")

    if not (0 <= args.majority_ratio <= 1):
        raise SystemExit("--majority-ratio debe estar entre 0 y 1.")

    if args.simulate and args.submit:
        raise SystemExit("No se puede combinar --simulate con --submit.")

    if not args.submit and args.submit_count != 1:
        raise SystemExit("--submit-count solo se puede usar junto con --submit.")

    if not args.submit and args.submit_delay != 0:
        raise SystemExit("--submit-delay solo se puede usar junto con --submit.")

    rng = random.Random(args.seed)

    if args.simulate:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for iteration in range(1, args.simulate + 1):
                answers = generate_answers(
                    args.model,
                    args.persona,
                    majority_ratio=args.majority_ratio,
                    rng=rng,
                )
                record = {"iteration": iteration, "answers": answers}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"Simulación completada: {args.simulate} iteraciones guardadas en {output_path}")
        return

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=args.headless)
        if args.submit:
            for iteration in range(1, args.submit_count + 1):
                answers = generate_answers(
                    args.model,
                    args.persona,
                    majority_ratio=args.majority_ratio,
                    rng=rng,
                )

                page = browser.new_page(viewport={"width": 1440, "height": 1800})
                page.goto(args.url, wait_until="networkidle")
                fill_form(page, answers)
                click_submit_or_next(page)
                page.wait_for_load_state("networkidle")
                wait_submission_confirmation(page)
                page.close()

                print(f"[{iteration}/{args.submit_count}] formulario enviado")
                if iteration < args.submit_count and args.submit_delay > 0:
                    time.sleep(args.submit_delay)

            print(f"Envío completado: {args.submit_count} formularios enviados.")
        else:
            answers = generate_answers(
                args.model,
                args.persona,
                majority_ratio=args.majority_ratio,
                rng=rng,
            )

            page = browser.new_page(viewport={"width": 1440, "height": 1800})
            page.goto(args.url, wait_until="networkidle")
            fill_form(page, answers)
            page.screenshot(path="formulario_relleno.png", full_page=True)
            print("Formulario rellenado. Revisé una captura en formulario_relleno.png.")
            page.close()

        browser.close()


if __name__ == "__main__":
    main()
