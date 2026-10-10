from contextlib import contextmanager
from datetime import datetime

@contextmanager
def timeit(file_write=None):
    """Gerenciador de contexto para verificar o tempo de execução (runtime)."""

    # ==========================================
    # BLOCO DE ENTRADA (Executado ao abrir o 'with')
    # ==========================================

    # Marca o momento exato em que o bloco de código começou a ser executado
    start_time = datetime.now()

    # Imprime o tempo de início.
    print(f'Tempo de Inicio (hh:mm:ss.ms) {start_time}', file=file_write)

    yield

    # ==========================================
    # BLOCO DE SAÍDA (Executado ao sair do 'with')
    # ==========================================

    # Marca o momento exato em que o bloco 'with' terminou a sua execução
    end_time = datetime.now()

    # Calcula a diferença matemática entre o término e o início para obter a duração
    time_elapsed = end_time - start_time

    # Imprime os resultados finais (término e tempo total de processamento)
    print(f'Tempo de Termino (hh:mm:ss.ms) {end_time}', file=file_write)
    print(f'Tempo Total (hh:mm:ss.ms) {time_elapsed}', file=file_write)
