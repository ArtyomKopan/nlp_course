for step in range(max_len - 1):
    # 1) прогоняем все текущие гипотезы через decode_step
    #    состояние нужно расширить по оси beam: [B*beam, ...]
    flat_state = flatten_beam(state)                 # каждая тензор состояния -> [B*beam, ...]
    last_tokens = tokens[:, :, -1].reshape(-1)       # [B*beam]
    new_flat_state, logits = self.decode_step(flat_state, last_tokens)
    # logits: [B*beam, V]

    logprobs = F.log_softmax(logits, dim=-1).reshape(B, -1, len(self.out_voc))
    # logprobs: [B, beam, V]

    # 2) добавляем к текущим скорам
    candidate_scores = scores.unsqueeze(-1) + logprobs      # [B, beam, V]

    # 3) отбираем top-k по всему (beam * V)
    V = candidate_scores.shape[-1]
    candidate_scores = candidate_scores.reshape(B, -1)       # [B, beam*V]
    top_scores, top_indices = candidate_scores.topk(beam_size, dim=-1)
    # top_indices: [B, beam]

    # 4) восстанавливаем, из какой гипотезы и какой токен выбран
    beam_idx = top_indices // V          # [B, beam]
    token_idx = top_indices % V          # [B, beam]

    # 5) обновляем токены: приклеиваем новый токен к выбранной гипотезе
    # tokens: [B, beam, L] -> gather по beam_idx -> [B, beam, L]
    tokens = tokens.gather(
        1,
        beam_idx.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
    )
    tokens = torch.cat([tokens, token_idx.unsqueeze(-1)], dim=-1)

    # 6) обновляем счёт, состояние и флаг завершения
    scores = top_scores
    state = unflatten_beam(new_flat_state, beam_size)

    # 7) гипотезы, закончившиеся на EOS, больше не расширяем
    finished = finished.gather(1, beam_idx) | (token_idx == self.out_voc.eos_ix)

    # ранний выход, если все гипотезы всех примеров завершились
    if finished.all():
        break

best = scores.argmax(dim=-1)                          # [B]
best_tokens = tokens[torch.arange(B, device=device), best]   # [B, L]
return best_tokens, tokens  # или best_tokens, all_tokens для визуализации


def flatten_beam(state, beam_size):
    # state: список тензоров [B, ...]
    # результат: список тензоров [B*beam, ...]
    out = []
    for t in state:
        # [B, ...] -> [B, 1, ...] -> [B, beam, ...] -> [B*beam, ...]
        t = t.unsqueeze(1).expand(-1, beam_size, *t.shape[1:])
        out.append(t.reshape(-1, *t.shape[2:]))
    return out

def unflatten_beam(state, beam_size):
    # state: список тензоров [B*beam, ...] -> [B, beam, ...]
    B = state[0].shape[0] // beam_size
    out = []
    for t in state:
        out.append(t.reshape(B, beam_size, *t.shape[1:]))
    return out