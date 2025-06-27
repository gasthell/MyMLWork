import torch
import torch.nn as nn
import numpy as np
import math

# Импортируем базовый класс и необходимые модули
# Убедитесь, что эти файлы находятся в правильной директории (например, models/)
from models.ctm import ContinuousThoughtMachine
from models.modules import Identity # Используем для чистоты кода

class CTM_NLP(ContinuousThoughtMachine):
    """
    Адаптация Continuous Thought Machine для задач обработки естественного языка (NLP).

    Этот класс наследуется от ContinuousThoughtMachine и заменяет специфичный для 
    изображений бэкенд (ResNet) на стандартные для NLP слои:
    1. Эмбеддинги токенов (Token Embeddings)
    2. Позиционные эмбеддинги (Positional Embeddings)

    Внутренняя механика "мышления" CTM (рекурренция, NLM, синхронизация)
    остается без изменений. Модель принимает на вход `input_ids` (индексы токенов)
    и генерирует предсказания на каждом "шаге мысли".

    Args:
        vocab_size (int): Размер словаря для эмбеддингов токенов.
        max_seq_len (int): Максимальная длина последовательности для позиционных эмбеддингов.
        
        # --- Аргументы, унаследованные от ContinuousThoughtMachine ---
        iterations (int): Количество внутренних 'шагов мысли' (T).
        d_model (int): Размерность внутреннего латентного пространства (D).
        d_input (int): Размерность эмбеддингов и выходов внимания. Должна быть равна d_embedding.
        heads (int): Количество голов внимания.
        n_synch_out (int): Количество нейронов для выходной синхронизации.
        n_synch_action (int): Количество нейронов для синхронизации действий/внимания.
        synapse_depth (int): Глубина модели синапсов (U-Net).
        memory_length (int): Длина истории для Neuron-Level Models (M).
        deep_nlms (bool): Использовать ли глубокие (2-слойные) NLM.
        memory_hidden_dims (int): Скрытая размерность для глубоких NLM.
        dropout (float): Dropout.
        ... (и другие аргументы базового класса)
    """
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 # Аргументы базового класса CTM
                 iterations: int,
                 d_model: int,
                 d_input: int,
                 heads: int,
                 n_synch_out: int,
                 n_synch_action: int,
                 synapse_depth: int,
                 memory_length: int,
                 deep_nlms: bool,
                 memory_hidden_dims: int,
                 dropout: float = 0.1,
                 **kwargs):
        
        # --- Шаг 1: Инициализация родительского класса с параметрами для NLP ---
        # Мы "обманываем" родительский класс, говоря ему, что у нас нет бэкенда и
        # позиционных эмбеддингов для изображений. Мы создадим свои собственные ниже.
        super().__init__(
            iterations=iterations,
            d_model=d_model,
            d_input=d_input,
            heads=heads,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            synapse_depth=synapse_depth,
            memory_length=memory_length,
            deep_nlms=deep_nlms,
            memory_hidden_dims=memory_hidden_dims,
            dropout=dropout,
            # --- Жестко задаем параметры, специфичные для изображений ---
            backbone_type='none',
            positional_embedding_type='none',
            out_dims=vocab_size, # Выходной размер равен размеру словаря
            **kwargs
        )
        
        print("Initializing CTM for NLP tasks...")

        # --- Шаг 2: Замена бэкенда на NLP-специфичные слои ---
        
        # Эмбеддинг токенов: преобразует индексы токенов в векторы
        # d_input - это размерность, которую ожидает механизм внимания
        self.token_embedding = nn.Embedding(vocab_size, d_input)
        
        # Позиционный эмбеддинг: добавляет информацию о порядке токенов
        self.position_embedding_nlp = nn.Embedding(max_seq_len, d_input)
        
        # Заменяем "пустышки", созданные родительским классом, на Identity,
        # чтобы избежать путаницы. Наша логика будет использовать новые слои.
        self.backbone = Identity()
        self.positional_embedding = Identity()
        
        print(f"CTM_NLP initialized with vocab_size={vocab_size}, max_seq_len={max_seq_len}")
        print(f"Output projection layer will map to {self.output_projector[0].out_features} logits.")

    def compute_features(self, input_ids: torch.Tensor):
        """
        Переопределяем метод для вычисления признаков из текстовых данных.
        Вместо пропускания изображения через ResNet, мы получаем эмбеддинги токенов.

        Args:
            input_ids (torch.Tensor): Тензор с индексами токенов.
                                      Shape: (batch_size, sequence_length).

        Returns:
            torch.Tensor: Ключи/значения (kv) для механизма внимания.
                          Shape: (batch_size, sequence_length, d_input).
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        # 1. Получаем эмбеддинги токенов
        token_embeds = self.token_embedding(input_ids) # (B, S, D_input)
        
        # 2. Создаем и получаем позиционные эмбеддинги
        positions = torch.arange(0, seq_length, device=device).unsqueeze(0).expand(batch_size, -1) # (B, S)
        position_embeds = self.position_embedding_nlp(positions) # (B, S, D_input)
        
        # 3. Суммируем эмбеддинги
        combined_features = token_embeds + position_embeds # (B, S, D_input)
        
        # 4. Проецируем в key-value пространство для внимания
        # Слой kv_proj уже создан в родительском классе __init__
        kv = self.kv_proj(combined_features) # (B, S, D_input)
        
        return kv

    def forward(self, input_ids: torch.Tensor, track=False):
        """
        Переопределяем forward для явного указания, что на вход подаются input_ids.
        Внутренняя логика вызывает `super().forward`, который выполнит весь 
        рекуррентный цикл CTM.

        Args:
            input_ids (torch.Tensor): Тензор с индексами токенов.
                                      Shape: (batch_size, sequence_length).
            track (bool): Флаг для отслеживания внутренних состояний.

        Returns:
            Tuple: Кортеж с предсказаниями, уверенностью и другими отладочными данными,
                   аналогично родительскому классу.
        """
        # Мы просто вызываем родительский forward, который, в свою очередь,
        # вызовет наш переопределенный метод `compute_features`.
        return super().forward(input_ids, track=track)

# --- Пример использования ---
if __name__ == '__main__':
    # Параметры для NLP модели
    VOCAB_SIZE = 10000
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 4
    SEQ_LEN = 64

    # Параметры CTM
    D_MODEL = 512       # Внутренняя размерность
    D_INPUT = 256       # Размерность эмбеддингов и внимания
    ITERATIONS = 50     # Количество шагов "мысли"
    HEADS = 8
    N_SYNCH_OUT = 256
    N_SYNCH_ACTION = 128
    SYNAPSE_DEPTH = 4
    MEMORY_LENGTH = 25
    MEMORY_HIDDEN = 32

    print("--- Creating CTM_NLP model ---")
    nlp_ctm = CTM_NLP(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        iterations=ITERATIONS,
        d_model=D_MODEL,
        d_input=D_INPUT,
        heads=HEADS,
        n_synch_out=N_SYNCH_OUT,
        n_synch_action=N_SYNCH_ACTION,
        synapse_depth=SYNAPSE_DEPTH,
        memory_length=MEMORY_LENGTH,
        deep_nlms=True,
        memory_hidden_dims=MEMORY_HIDDEN,
        do_layernorm_nlm=False,
        neuron_select_type='random-pairing',
        n_random_pairing_self=16,
    )
    
    # Проверка количества параметров
    num_params = sum(p.numel() for p in nlp_ctm.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params / 1e6:.2f}M")

    # Создание "игрушечных" данных
    dummy_input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    
    print("\n--- Running forward pass ---")
    # Прямой проход
    predictions, certainties, _ = nlp_ctm(dummy_input_ids)
    
    print("\n--- Checking output shapes ---")
    print(f"Input shape: {dummy_input_ids.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Certainties shape: {certainties.shape}")
    
    # Ожидаемые формы
    # Predictions: (batch_size, out_dims, iterations) -> (4, 10000, 50)
    # Certainties: (batch_size, 2, iterations) -> (4, 2, 50)
    
    expected_pred_shape = (BATCH_SIZE, VOCAB_SIZE, ITERATIONS)
    expected_cert_shape = (BATCH_SIZE, 2, ITERATIONS)
    
    assert predictions.shape == expected_pred_shape, f"Prediction shape mismatch! Expected {expected_pred_shape}, got {predictions.shape}"
    assert certainties.shape == expected_cert_shape, f"Certainty shape mismatch! Expected {expected_cert_shape}, got {certainties.shape}"
    
    print("\nOutput shapes are correct. CTM_NLP is ready for training on text data!")