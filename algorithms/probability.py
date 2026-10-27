"""Punto 1. Distribuciones discretas y decisiones de respuesta."""
import math


def normalize(weights: dict) -> dict:
    """Retorna un diccionario nuevo cuyas probabilidades suman uno.

    Entrada: weights = {resultado: peso}.
    Salida: {resultado: probabilidad}, con las mismas claves.

    Tips:
    - Conserve las claves y su orden; no modifique weights.
    - Rechace pesos negativos, no finitos, diccionarios vacíos y suma cero.
    """
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Se requiere un diccionario de pesos no vacío")

    probabilities = {}
    for outcome, weight in weights.items():
        try:
            valid = math.isfinite(weight) and weight >= 0
        except TypeError as error:
            raise ValueError("Los pesos deben ser números finitos no negativos") from error
        if not valid:
            raise ValueError("Los pesos deben ser números finitos no negativos")
        probabilities[outcome] = weight

    total = sum(probabilities.values())
    if total == 0:
        raise ValueError("La suma de los pesos debe ser positiva")
    return {outcome: weight / total for outcome, weight in probabilities.items()}


def expected_utility(distribution: dict, utilities: dict) -> float:
    """Calcula sum_s P(s) U(s) para una acción.

    Entradas: distribution = {resultado: probabilidad} y
    utilities = {resultado: utilidad} para una sola acción.
    Salida: un número con la utilidad esperada.

    Tips:
    - distribution ya debe estar normalizada; no normalice utilidades.
    - Ambas tablas deben contener los mismos resultados posibles.
    """
    # Validación de las entradas; no modificar.
    if set(distribution) != set(utilities) or not distribution:
        raise ValueError("Probabilidades y utilidades deben tener las mismas claves")
    if any(not math.isfinite(p) or p < 0 for p in distribution.values()):
        raise ValueError("Distribución inválida")
    if not math.isclose(sum(distribution.values()), 1, abs_tol=1e-9):
        raise ValueError("La distribución debe sumar uno")
    if any(not math.isfinite(u) for u in utilities.values()):
        raise ValueError("Utilidades no finitas")
    return sum(distribution[outcome] * utilities[outcome] for outcome in distribution)


def best_action(distribution: dict, actions: dict) -> tuple[str, dict]:
    """Retorna (nombre de acción, utilidades esperadas por acción).

    Entradas: distribution = {resultado: probabilidad} y
    actions = {acción: {resultado: utilidad}}.
    Salida: (acción elegida, {acción: utilidad esperada}).

    Tips:
    - actions asocia cada acción con su tabla de utilidades.
    - Reutilice expected_utility y conserve la primera acción en los empates.
    """
    # Validación de las acciones; no modificar.
    if not actions:
        raise ValueError("Se requiere al menos una acción")
    expected_values = {
        action: expected_utility(distribution, utility_table)
        for action, utility_table in actions.items()
    }
    chosen = next(iter(expected_values))
    for action, value in expected_values.items():
        if value > expected_values[chosen]:
            chosen = action
    return chosen, expected_values
