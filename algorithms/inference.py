"""Punto 2. Inferencia exacta por enumeración."""
from algorithms.probability import normalize


def enumeration_ask(network, query: str, evidence: dict) -> dict:
    """Calcula P(query | evidence) sobre la conjunta factorizada.

    Entradas: network es la red; query es el nombre de la
    variable consultada; evidence es {nombre de variable: 0 o 1}.
    Salida: {0: probabilidad, 1: probabilidad}.

    Tips:
    - Recorra network.variables en orden topológico.
    - Para una variable fijada multiplique su probabilidad; para una oculta
      sume las dos alternativas 0 y 1. Normalice al terminar.
    - Use copias del evento y no modifique evidence.
    - network.probability(variable, valor, evento) consulta la CPT;
      el evento debe contener los valores de todos los padres.
    - Puede usar una función auxiliar recursiva. Al terminar el recorrido,
      el producto sin factores restantes vale 1.
    - Evidencia imposible: ValueError. Consulta observada: distribución puntual
      solo si la evidencia es posible.
    """
    # Valida nombres de variables y dominio de la evidencia.
    network.validate_query(query, evidence)

    def enumerate_all(index: int, event: dict) -> float:
        if index == len(network.variables):
            return 1.0

        variable = network.variables[index]
        if variable in event:
            value = event[variable]
            probability = network.probability(variable, value, event)
            return probability * enumerate_all(index + 1, event)

        total = 0.0
        for value in (0, 1):
            extended_event = dict(event)
            extended_event[variable] = value
            probability = network.probability(variable, value, extended_event)
            total += probability * enumerate_all(index + 1, extended_event)
        return total

    if query in evidence:
        observed = evidence[query]
        evidence_probability = enumerate_all(0, dict(evidence))
        weights = {0: 0.0, 1: 0.0}
        weights[observed] = evidence_probability
        return normalize(weights)

    weights = {}
    for value in (0, 1):
        event = dict(evidence)
        event[query] = value
        weights[value] = enumerate_all(0, event)

    return normalize(weights)
