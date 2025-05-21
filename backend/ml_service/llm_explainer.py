# backend/ml_service/llm_explainer.py
import requests
import json
import logging
from typing import Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

class LLMExplainer:
    def __init__(self, ollama_base_url, model_name):
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.model_name = model_name
        self.prompt_template = """Ты - ассистент аналитика в интернет-магазине.

            Твоя задача - объяснить менеджеру простым и понятным языком, почему данный заказ/событие было помечено как аномальное.
            Используй только предоставленную информацию. Не придумывай детали, которых нет во входных данных.
            Объяснение должно быть кратким (2-4 предложения) и указывать на наиболее вероятные причины.

            Детали аномалии:
            {anomaly_details_str}

            Информация от системы обнаружения:
            Детектор: {detector_name} (тип: {detector_type})
            Скор аномальности: {anomaly_score:.2f}
            {technical_explanation_str}

            Задача: Сгенерируй объяснение для менеджера.
            Объяснение:"""

    def _format_technical_explanation(self, shap_values: Optional[Dict[str, float]], detector_specific_info: Optional[Dict[str, Any]]) -> str:
        explanation_parts = []
        if shap_values:
            explanation_parts.append("SHAP Values (Feature Contributions):")
            # Сортировка SHAP значений по убыванию абсолютного значения
            try:
                sorted_shap = sorted(shap_values.items(), key=lambda item: abs(float(item[1])) if isinstance(item[1], (int, float, str)) and str(item[1]).replace('.', '', 1).replace('-','',1).isdigit() else 0, reverse=True)
            except Exception as e:
                logger.warning(f"Ошибка при сортировке SHAP-значений: {e}. Используется исходный порядок.")
                sorted_shap = shap_values.items()
            for feature, value in sorted_shap:
                try:
                    value_float = float(value)
                    explanation_parts.append(f"  - {feature}: {value_float:.4f}")
                except (ValueError, TypeError):
                    explanation_parts.append(f"  - {feature}: {value} (не удалось преобразовать в float)")
        
        if detector_specific_info:
            explanation_parts.append("\nDetector Specific Information:")
            for key, value in detector_specific_info.items():
                explanation_parts.append(f"  - {key}: {value}")
        
        return "\n".join(explanation_parts)

    def generate_explanation(
        self,
        anomaly_data: Dict[str, Any],
        shap_values: Optional[Dict[str, float]] = None,
        detector_specific_info: Optional[Dict[str, Any]] = None,
        anomaly_score: Optional[Union[float, str]] = None
    ) -> str:
        """
        Генерирует текстовое объяснение аномалии с использованием LLM.
        anomaly_data: Словарь с данными об аномалии (order_id, customer_id, etc.)
        shap_values: Словарь SHAP значений {feature: value}
        detector_specific_info: Словарь с дополнительной информацией от детектора
        anomaly_score: Оценка аномалии (0-1)
        """
        
        logger.debug(f"Запрос на генерацию объяснения LLM для данных: {anomaly_data}, SHAP: {shap_values is not None}, Details: {detector_specific_info is not None}")

        technical_explanation = self._format_technical_explanation(shap_values, detector_specific_info)
        
        # Валидация и форматирование anomaly_score
        score_str = "N/A"
        if anomaly_score is not None:
            try:
                score_float = float(str(anomaly_score).replace(",", ".")) # Обработка запятой как десятичного разделителя
                score_str = f"{score_float:.3f}"
            except ValueError:
                logger.warning(f"Не удалось преобразовать anomaly_score '{anomaly_score}' в float.")
                score_str = str(anomaly_score) # Оставляем как есть, если не удалось

        system_prompt = (
            "Ты — эксперт по анализу данных, специализирующийся на выявлении мошенничества и операционных аномалий в электронной коммерции. "
            "Твоя задача — предоставить менеджеру по операционной деятельности ОЧЕНЬ КРАТКОЕ (3-5 предложений) и МАКСИМАЛЬНО ИНФОРМАТИВНОЕ объяснение обнаруженной аномалии для обычного пользователя. "
            "Ключевая цель: четко объяснить, ПОЧЕМУ система считает данное событие аномальным, опираясь на предоставленные числовые данные, "
            "значения SHAP (если есть) и специфическую логику детектора (если предоставлена). "
            "Избегай общих фраз и излишне технического жаргона. Сосредоточься на конкретных фактах из данных."
            "\n\n"
            "ОБЯЗАТЕЛЬНАЯ СТРУКТУРА ОТВЕТА:\n"
            "1. Суть аномалии (1 предложение): Кратко, что это за событие.\n"
            "2. Причины аномальности (1-3 предложения): Конкретные факторы из данных (например, необычно высокая цена, редкая категория для продавца, несоответствие геолокации, высокие значения SHAP для определенных признаков), которые делают это событие подозрительным. ОБЯЗАТЕЛЬНО сошлись на значения, если они есть.\n"
            "3. Возможные последстви/рекомендации (1 предложение): Краткий вывод о потенциальных рисках или что можно проверить.\n"
            "\n"
            "Ответ должен быть на русском языке."
        )

        prompt = f"""Инструкция для системы:
{system_prompt}

Контекст для анализа:
Обнаружена аномалия!
Оценка аномальности: {score_str}

Детали аномалии (JSON):
{json.dumps(anomaly_data, indent=2, ensure_ascii=False)}

Технические детали от системы обнаружения (если доступны):
{technical_explanation}

ЗАДАЧА:
Следуя ИНСТРУКЦИИ ДЛЯ СИСТЕМЫ и ОБЯЗАТЕЛЬНОЙ СТРУКТУРЕ ОТВЕТА, предоставь объяснение указанной аномалии.
Объяснение:
"""
        
        logger.debug(f"Полный промпт для Ollama ({self.model_name}):\\n{prompt}")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False, 
            "options": { 
                "temperature": 0.3
            }
        }
        try:
            api_endpoint = "/api/generate"
            full_url = self.ollama_base_url + api_endpoint 
            response = requests.post(full_url, json=payload, timeout=60) 
            response.raise_for_status()
            response_data = response.json()
            explanation = response_data.get("response", "Не удалось получить объяснение от LLM.").strip()
            # Логируем сырой ответ перед очисткой
            logger.info(f"LLM ({self.model_name}) сгенерировала объяснение (сырое): {explanation}")

            cleaned_explanation = explanation
            think_block_start_marker = "<think>\n"
            end_of_think_block_sequence = "\n</think>\n\n"

            if cleaned_explanation.startswith(think_block_start_marker):
                # Ищем позицию конца блока мыслей
                end_sequence_index = cleaned_explanation.find(end_of_think_block_sequence)
                
                if end_sequence_index != -1:
                    if end_sequence_index >= len(think_block_start_marker) -1: # -1 т.к. find дает начальный индекс
                        # Извлекаем текст после этой последовательности
                        actual_response_start_index = end_sequence_index + len(end_of_think_block_sequence)
                        cleaned_explanation = cleaned_explanation[actual_response_start_index:].strip()
                    else:
                        # Этого не должно произойти при нормальном формате, но для безопасности
                        logger.warning(
                            f"Обнаружена последовательность конца блока мыслей '{end_of_think_block_sequence}', но она находится до или перекрывает начало блока '{think_block_start_marker}'. "
                            "Ответ может быть в неожиданном формате."
                        )
                        cleaned_explanation = "Получен ответ от LLM в неожиданном формате (некорректное расположение блока мыслей)."
                else:
                    logger.warning(
                        f"LLM объяснение началось с '{think_block_start_marker}', но ожидаемая последовательность конца блока мыслей '{end_of_think_block_sequence}' не найдена. "
                        "Ответ может содержать только размышления LLM или быть в неполном формате."
                    )
                    cleaned_explanation = "Получен ответ от LLM в неполном или неожиданном формате (возможно, только блок размышлений)."
            
            if not cleaned_explanation and explanation and explanation.startswith(think_block_start_marker):
                 logger.info("После обработки блока <think> объяснение стало пустым. Исходный текст, вероятно, содержал только мысли или имел неполный формат.")
                 cleaned_explanation = "LLM не предоставила объяснения после блока размышлений, или формат ответа был неполным."

            logger.info(f"LLM ({self.model_name}) сгенерировала объяснение (очищенное): {cleaned_explanation}")
            return cleaned_explanation
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при запросе к Ollama API ({self.ollama_base_url}): {e}")
            return f"Ошибка связи с LLM ({self.model_name}): {e}"
        except json.JSONDecodeError as e_json:
            logger.error(f"Ошибка декодирования JSON ответа от Ollama API: {e_json}. Ответ: {response.text}")
            return f"Ошибка обработки ответа от LLM ({self.model_name}): невалидный JSON."
        except Exception as e_inner:
            logger.error(f"Неожиданная ошибка при генерации объяснения LLM: {e_inner}", exc_info=True)
            return f"Внутренняя ошибка при генерации объяснения LLM ({self.model_name})." 