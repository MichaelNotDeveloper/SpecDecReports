# Обучение draft-моделей DFlash и DSpark: от исходных данных до проверки speculative decoding

Проверено по статьям и исходному коду **28 октября 2026 года**.

Практический основной маршрут: **Qwen/Qwen3-4B → регенерация Open-PerfectBlend → кэш target → обучение DSpark в DeepSpec → оценка acceptance**. Для DFlash отдельно приведены оригинальный рецепт из статьи, воспроизведение в DeepSpec и маршрут через SpecForge. Эти рецепты различаются; ниже указано, какие именно результаты позволяет воспроизводить каждый из них.

**Статус проверки.** Обе приложенные статьи прочитаны; схемы обучения дополнительно проверены визуально. Репозитории скачаны, команды и параметры сверены с исходниками, включая наличие использованных CLI-флагов. Python/JSON-примеры проверены синтаксически, bash-блоки прошли `bash -n`, все ссылки на файлы закреплённых checkout проверены на существование. Полный сбор данных, CUDA-окружение и обучение на GPU здесь не запускались. Примеры, добавленные мной, обозначены как адаптация или рекомендация, а не как команды авторов.

## Содержание

1. [Источники и границы воспроизводимости](#1-источники-и-границы-воспроизводимости)
2. [Что именно обучается](#2-что-именно-обучается)
3. [Как устроен один обучающий пример](#3-как-устроен-один-обучающий-пример)
4. [Отдельная часть: как собирались датасеты](#4-отдельная-часть-как-собирались-датасеты)
5. [Окружение и ресурсы](#5-окружение-и-ресурсы)
6. [Пробный запуск DSpark на небольшом наборе](#6-пробный-запуск-dspark-на-небольшом-наборе)
7. [Полное обучение DSpark и baseline DFlash в DeepSpec](#7-полное-обучение-dspark-и-baseline-dflash-в-deepspec)
8. [Что происходит внутри обучения и функции потерь](#8-что-происходит-внутри-обучения-и-функции-потерь)
9. [Оценка модели и confidence](#9-оценка-модели-и-confidence)
10. [DFlash через SpecForge: ближе к оригинальной архитектуре](#10-dflash-через-specforge-ближе-к-оригинальной-архитектуре)
11. [Online-обучение без полного постоянного кэша](#11-online-обучение-без-полного-постоянного-кэша)
12. [Перенос на другой target или собственный домен](#12-перенос-на-другой-target-или-собственный-домен)
13. [Типичные проблемы и проверка воспроизводимости](#13-типичные-проблемы-и-проверка-воспроизводимости)
14. [Карта шагов и источников](#14-карта-шагов-и-источников)
15. [Ссылки на первоисточники](#15-ссылки-на-первоисточники)

## 1. Источники и границы воспроизводимости

### 1.1. На какие материалы опирается туториал

| Обозначение | Источник | Для чего используется |
|---|---|---|
| P1 | Chen et al., **DFlash: Block Diffusion for Flash Speculative Decoding**, arXiv:2602.06036v2, 28.05.2026; приложенный PDF, 13 страниц | Архитектура DFlash, training attention, исходная смесь данных, гиперпараметры |
| P2 | Cheng et al., **DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation**, arXiv:2607.05147v1, 06.07.2026; приложенный PDF, 33 страницы | Markov/RNN head, confidence, CE/L1/BCE, Open-PerfectBlend, сравнение моделей |
| R1 | [deepseek-ai/DeepSpec][DS-ROOT] | Основной исполнимый pipeline подготовки данных, кэша, обучения и оценки |
| R2 | [z-lab/dflash][DF-ROOT] | Официальная реализация инференса DFlash и ссылки на модели |
| R3 | [sgl-project/SpecForge][SF-ROOT] | Дополнительные маршруты offline/online-обучения DFlash и DSpark |
| B1 | [Блог Z Lab о DFlash][DF-BLOG] | Авторское объяснение метода и первоначальные примеры инференса |
| B2 | [Блог LMSYS о SpecForge, 25.07.2025][SF-BLOG-2025] | Идея online/offline-обучения; исторически материал об EAGLE3 |
| B3 | [Блог LMSYS о SpecForge v0.3, 04.08.2026][SF-BLOG-2026] | Разделение capture/доставки признаков/trainer и управление их жизненным циклом |

**Приоритет источников.** Формулы и экспериментальный дизайн берутся из соответствующей версии статьи. CLI, названия файлов, форматы кэша и фактические значения параметров — из зафиксированного кода. Блоги поясняют архитектурные решения, но не заменяют актуальные команды из репозитория. Если рекомендация моя, это прямо подписано.

### 1.2. Зафиксированные версии репозиториев

| Репозиторий | Проверенный commit |
|---|---|
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` |
| z-lab/dflash | `07ebd93db9f472af339b644bb70221ad8428328a` |
| SpecForge | `8b45bab869ef6b9d539a2626629ad31af51491f7` |

Ссылки на файлы этих репозиториев в документе закреплены за commit. Карточки внешних датасетов могут обновляться: для собственного запуска нужно сохранить их `revision`, число строк и хэши выходных файлов.

### 1.3. Три разных смысла «обучить DFlash»

| Рецепт | Данные | Длина блока | Эпохи | Что воспроизводится |
|---|---|---|---|---|
| Оригинальный DFlash, P1 | Около 800K примеров из Nemotron Post-Training Dataset V2 и CodeAlpaca; ответы генерирует target | Обычно 16 входных позиций, из них 15 предсказаний; для LLaMA 3.1 блок 10 | 6 | Дизайн и настройки оригинальной статьи; точная смесь данных полностью не раскрыта |
| DFlash baseline в DeepSpec, P2/R1 | Open-PerfectBlend; ответы генерирует соответствующий target | В проверенном коде 7 позиций дают 7 предсказаний через общий DSpark backbone | 10 | Публичный baseline авторов DSpark, а не исходный checkpoint Z Lab |
| DSpark в DeepSpec, P2/R1 | Та же подготовка Open-PerfectBlend | 7 предсказаний + target bonus при верификации | 10 | Основной публичный эксперимент DSpark |

Различие DFlash baseline подтверждается непосредственно кодом: конфиг `dflash_qwen3_4b.py` выбирает `Qwen3DSparkTrainer`, отключает Markov/confidence и оставляет только CE. Общая модель использует label offsets `1..block_size`, включая предсказание из anchor-позиции. В оригинальном DFlash P1 anchor не является предсказываемой позицией. **Одинаковое название алгоритма не гарантирует одинаковый сдвиг токенов.**

**Источники:** [P1, §4.2, §5, приложение A.1][DF-PAPER]; [P2, §3.1 и §4.1][DS-PAPER]; [baseline-конфиг DeepSpec][DS-DF-CONFIG]; [forward модели DeepSpec][DS-MODEL].

## 2. Что именно обучается

### 2.1. Target и draft

**Target** — исходная авторегрессионная LLM, например Qwen3-4B. Она генерирует обучающие ответы, предоставляет внутренние признаки и затем проверяет предложения draft-модели. Её веса при обучении draft не обновляются.

**Draft** — отдельная небольшая модель, привязанная к конкретному target. Она получает его признаки контекста и за один тяжёлый параллельный forward строит предложения нескольких будущих токенов. DSpark дополнительно корректирует их лёгкой последовательной головой.

Это обучение ускорителя инференса. Модель draft сама по себе не становится полноценной заменой target. Корректная speculative verification сохраняет распределение target; качество draft влияет главным образом на число принятых токенов и затраты времени.

**Источники:** [P1, §3–4][DF-PAPER]; [P2, §2–3][DS-PAPER]; [авторский блог DFlash][DF-BLOG].

### 2.2. Какие веса заморожены, а какие обучаются

| Компонент | DFlash | DSpark |
|---|---|---|
| Полный target | Заморожен | Заморожен |
| Target token embedding | Заморожен | Заморожен |
| Target LM head | Заморожен | Заморожен |
| Fusion/projector target features | Обучается | Обучается |
| Transformer-слои draft и его нормы | Обучаются | Обучаются |
| Markov или RNN head | Нет | Обучается |
| Confidence head | Нет | Обучается |

В DeepSpec embedding и LM head **копируются** из target в draft и затем замораживаются. «Shared» в статье здесь означает одинаковые замороженные веса; это не обещание общего физического тензора в памяти двух процессов. При offline-обучении полный target загружается на CPU для инициализации этих весов и затем удаляется; признаки уже лежат на диске.

**Источники:** [P1, §4.2][DF-PAPER]; [P2, §3.3][DS-PAPER]; [инициализация моделей trainer][DS-TRAINER]; [initialize_embeddings_and_head][DS-MODEL].

### 2.3. Чем DSpark отличается от DFlash

DFlash вычисляет logits блока параллельно. Из-за отсутствия зависимости между выбранными токенами внутри блока возможны несогласованные сочетания продолжений.

DSpark сохраняет параллельный backbone, но добавляет к logits переходную поправку от предыдущего выбранного токена:

\[
z_k(v)=U_k(v)+W_1[x_{k-1}]W_2[:,v].
\]

Матрица переходов не хранится целиком как `V × V`: используются два множителя ранга `r`, по умолчанию `r=256`. Во время обучения предыдущие токены берутся из target-generated последовательности; во время инференса это токены, уже сэмплированные draft внутри текущего блока. Тяжёлый backbone не запускается заново для каждого токена.

Confidence head оценивает условную вероятность принятия следующей позиции при условии принятия предыдущего префикса. Scheduler использует confidence и профиль производительности serving engine, чтобы выбирать длину проверки.

**Источники:** [P2, §3.1–3.2, формулы 4–8][DS-PAPER]; [VanillaMarkov и последовательное сэмплирование][DS-MARKOV]; [teacher-forcing prev_token_ids][DS-MODEL].

## 3. Как устроен один обучающий пример

### 3.1. Четыре уровня данных

| Уровень | Что содержит | Зачем нужен |
|---|---|---|
| Исходный корпус | Чужие инструкции, диалоги, ответы и метаданные | Получить разнообразные запросы |
| Target-generated JSONL | Те же пользовательские запросы, новые ответы конкретного target | Обучаться на поведении ускоряемой модели |
| Target cache | Token IDs, маски, выбранные hidden states, последний hidden state | Обойти повторный forward target на каждой эпохе |
| Случайные anchor-блоки | Anchor, маски, сдвинутые labels и attention mask | Построить задачу обучения draft |

Anchor-блоки в DeepSpec не нужно заранее сериализовать как отдельный огромный датасет: они заново сэмплируются из последовательностей во время обучения. Веса оптимизируются только по разрешённым assistant-позициям.

**Источники:** [P1, §4.2 и A.1][DF-PAPER]; [P2, §3.3][DS-PAPER]; [общие функции построения блоков][DS-COMMON]; [подготовка кэша][DS-CACHE-PREP].

### 3.2. Разница в сдвиге: наглядный пример

Пусть `D` — известный anchor, а правильное продолжение target — `E F G H`.

| Вариант | Вход draft | Какие позиции дают обучаемые предсказания | Labels |
|---|---|---|---|
| Оригинальный DFlash, 4 новых токена | `D M M M M` | Только 4 mask-позиции | `E F G H` |
| DSpark / текущая модель DeepSpec, 4 новых токена | `D M M M` | Все 4 позиции, включая позицию с `D` | `E F G H` |

`M` — специальный mask token. В DSpark на входе стоит `D`, но label первой позиции — **E**, а не D. Для anchor с индексом `a` и нумерации `j=0..B-1`:

\[
\text{draft label}[j]=x_{a+j+1},\qquad
\text{teacher logits}[j]=\operatorname{LMHead}(h^{\mathrm{last}}_{a+j}).
\]

Последний hidden state target с позиции `t` предсказывает токен `t+1`. Поэтому для target-token с индексом `t` нельзя брать teacher logits с этой же позиции `t`: это даст сдвиг на один токен.

**Источники:** [P1, §4.2 и рис. 4, стр. 5][DF-PAPER]; [P2, §3.1 и рис. 1, стр. 5][DS-PAPER]; [label_offsets и target_pred_indices][DS-MODEL].

### 3.3. Почему полный clean target forward не создаёт утечку будущего

Target обрабатывает чистый диалог каузально, но draft не получает бесконтрольный доступ ко всем вычисленным признакам. В DeepSpec для anchor `a` attention разрешает:

1. Смотреть на context features с индексами **строго меньше `a`**.
2. Смотреть двунаправленно на входы собственного anchor-блока.
3. Запрещает смотреть в другие anchor-блоки.

Anchor `x_a` передаётся как чистый embedding. Labels `x_{a+1}...` не вкладываются вместо масок в параллельный backbone. Они используются в loss и как прошлые токены для teacher forcing Markov head.

Важный нюанс: последний hidden state teacher на более поздних позициях используется как **супервизия** для TV/confidence, а не как доступный draft контекст этой позиции. Если переписать attention mask на обычный full attention, обучение перестанет соответствовать инференсу.

**Источник:** [create_dspark_attention_mask, create_noise_embed и build_eval_mask][DS-COMMON]; [forward модели][DS-MODEL].

## 4. Отдельная часть: как собирались датасеты

### 4.1. Общий принцип обеих статей

Обе работы используют корпус инструкций как **источник промптов**, после чего генерируют ответы той моделью, которую собираются ускорять. Затем чистые пары prompt/response пропускают через target для получения признаков.

Почему недостаточно взять исходный `assistant` из датасета: он мог быть сгенерирован другой моделью, в другом reasoning-режиме и с другим стилем продолжения. Для acceptance важна согласованность с распределением конкретного target, а не только содержательная правильность чужого ответа.

**Источники:** [P1, §5, Datasets, стр. 6][DF-PAPER]; [P2, §4.1, Training data, стр. 10][DS-PAPER].

### 4.2. Оригинальный DFlash: что известно о смеси

В P1 заявлены **около 800K примеров** из двух источников:

- NVIDIA Nemotron Post-Training Dataset V2.
- CodeAlpaca.

Для этих запросов ответы заново генерирует соответствующий target. Основная статья не даёт точного количества примеров каждого источника, распределения по Nemotron subsets, seed отбора, перечня выбранных ID, полного фильтра качества и точных generation settings для каждого target. Проверенный официальный репозиторий Z Lab содержит инференс и бенчмарки, но не полный первоначальный data/training pipeline.

Поэтому **нельзя честно восстановить побитно исходные 800K** только из P1 и текущего R2. Реально воспроизвести принцип, архитектуру и опубликованные гиперпараметры; собственную смесь нужно документировать отдельно.

**Источники:** [P1, §5 и A.1][DF-PAPER]; [проверенный состав официального репозитория][DF-ROOT]. Это указание на отсутствующие подробности в проверенных материалах, а не утверждение, что их нигде никогда не публиковали.

### 4.3. Откуда взялся Nemotron V2

Карточка NVIDIA описывает промпты из открытых корпусов и синтетической генерации. Применялись проверки качества и сложности, включая удаление противоречивых, слишком легко угадываемых и синтаксически некорректных запросов. Исходные ответы синтезировались несколькими открытыми моделями; встречаются reasoning-on/off варианты. **Это происхождение Nemotron, а не описание дополнительной фильтрации авторов DFlash.**

В карточке перечислены следующие категории; числа относятся к upstream-релизу и не являются долями 800K DFlash:

| Категория | Число записей |
|---|---:|
| math | 239 467 |
| code | 175 000 |
| stem | 355 000 |
| chat | 627 720 |
| multilingual_ja | 975 202 |
| multilingual_de | 1 015 314 |
| multilingual_it | 1 016 503 |
| multilingual_es | 935 704 |
| multilingual_fr | 1 001 504 |

Пример выбора subset из самой карточки:

```python
from datasets import load_dataset

subsets = load_dataset(
    "nvidia/Nemotron-Post-Training-Dataset-v2",
    "SFT",
    split=["code", "math"],
)
for ds in subsets:
    print(ds.features)
    print(ds[0].keys())
```

Доступ к файлам требует принятия условий на Hugging Face. Перед написанием конвертера проверьте фактическую схему выбранного subset. Не стоит угадывать названия полей по другой версии Nemotron.

**Источник:** [официальная карточка Nemotron V2, Data distribution / Prompts / Responses / Filtering][NEMOTRON].

### 4.4. Откуда взялся CodeAlpaca

CodeAlpaca — примерно 20K инструкционных примеров по программированию. Авторы взяли Self-Instruct/Stanford Alpaca pipeline, заменили seed tasks и генерационный prompt на задачи генерации, редактирования и оптимизации кода. Ответы исходного релиза получены через `text-davinci-003`.

Формат: `instruction`, необязательный `input`, `output`. Для подготовки запросов нужно сохранять и instruction, и непустой input. Например:

```python
def codealpaca_prompt(row):
    instruction = row["instruction"].strip()
    extra = row.get("input", "").strip()
    return instruction if not extra else instruction + "\n\n" + extra
```

Это моя минимальная адаптация сериализации, а не опубликованный шаблон DFlash. Поле `output` может использоваться для проверки схемы, но обучающий ответ для DFlash следует регенерировать своим target. Исходные настройки fine-tuning LLaMA в README CodeAlpaca не являются гиперпараметрами обучения DFlash.

**Источник:** [CodeAlpaca, Data Release / Data Generation Process][CODEALPACA].

### 4.5. Как практически собрать собственную смесь в стиле DFlash

Следующий порядок — **рекомендованная мной реконструкция** недостающего data pipeline, а не точный оригинальный рецепт 800K:

1. Зафиксировать ревизии Nemotron V2 и CodeAlpaca.
2. Выбрать Nemotron subsets под будущую нагрузку: code/math/chat/STEM и нужные языки.
3. Преобразовать промпты в общий `conversations` JSONL; для CodeAlpaca объединять instruction и input.
4. Сохранить `source`, исходный ID и subset в отдельном manifest или дополнительных полях.
5. Удалить точные дубликаты промптов; при необходимости сгруппировать близкие варианты одной задачи до train/validation split.
6. Отложить validation до генерации ответов. Пользовательские ходы одной беседы должны оставаться в одном split.
7. На train-промптах заново сгенерировать ответы target с зафиксированными sampling/thinking settings.
8. Проверить ответы и получить hidden features.
9. Записать реальные размеры после каждого фильтра. Назвать результат собственной смесью в стиле DFlash.

Не нужно механически смешивать все языковые ветки Nemotron и затем утверждать, что воспроизведена статья. Список ID и реальные квоты в P1 не опубликованы. Код регенерации и получения признаков из следующих разделов можно переиспользовать после нормализации JSONL.

**Основание:** [P1, принцип target-generated responses][DF-PAPER]; [схема данных и regeneration в SpecForge][SF-DATA]. Пункты про manifest, группировку и контроль дубликатов — мои рекомендации по воспроизводимости.

### 4.6. DSpark: состав Open-PerfectBlend

P2 использует Open-PerfectBlend и указывает около **1.3M** примеров с распределением: chat 17.6%, math 39.4%, code 38.9%, instruction-following 4.1%. Ответы регенерируются отдельно каждым target; генерация и оценка идут в **non-thinking** режиме.

Upstream Open-PerfectBlend — открытое воспроизведение instruction mixture из *The Perfect Blend: Redefining RLHF with Mixture of Judges*. Карточка перечисляет такие компоненты:

| Источник | Число примеров в таблице смеси до общей дедупликации |
|---|---:|
| `meta-math/MetaMathQA` | 395 000 |
| `openbmb/UltraInteract_sft` | 288 579 |
| `HuggingFaceH4/ultrachat_200k` | 207 865 |
| `microsoft/orca-math-word-problems-200k` | 200 035 |
| `HuggingFaceH4/ultrafeedback_binarized` | 187 405 |
| `theblackcat102/evol-codealpaca-v1` | 111 272 |
| `Post-training-Data-Flywheel/AutoIF-instruct-61k` | 61 492 |
| `mlabonne/lmsys-arena-human-preference-55k-sharegpt` | 57 362 |

Карточка сообщает удаление примерно 88.1K дублей; её machine-readable metadata содержит **1 420 909** строк итогового train split. Instruction-following источник отличается от исходного Perfect Blend, а непубличная harmful-intent часть не включена. Исходный алгоритм дедупликации и точный порядок конвертации всех компонентов эта карточка не раскрывает.

**Источники:** распределение и режим — [P2, §4.1][DS-PAPER]; состав, дедупликация и отличия — [карточка Open-PerfectBlend][PERFECTBLEND]; точный текущий размер — [README metadata][PERFECTBLEND-RAW].

### 4.7. Как создавались компоненты Open-PerfectBlend

Эта таблица описывает происхождение промптов до того, как они попали в DSpark. При подготовке DSpark исходные ответы этих корпусов заменяются ответами target.

| Компонент | Что известно о создании | Первичный источник |
|---|---|---|
| MetaMathQA | Математические задачи и решения, полученные расширением исходных задач; схема различает answer augmentation, переформулировки и другие типы. В данных есть `original_question`, поэтому разные строки могут происходить от одной задачи | [Официальная карточка][METAMATH] |
| UltraInteract SFT | Строится дерево попыток решения: рассуждения, взаимодействие со средой и текстовая критика. Для SFT берутся корректные действия и траектории с корректным завершением. `parent_id` связывает варианты | [Официальная карточка][ULTRAINTERACT] |
| UltraChat 200k | Из исходных примерно 1.4M диалогов ChatGPT отобран и сильно отфильтрован корпус; исправлен регистр текста, удалены характерные неуместные самоописания assistant. В смесь входит 207 865 примеров, совпадающих с размером `train_sft` | [Карточка Hugging Face H4][ULTRACHAT] |
| Orca Math | Seed-набор математических вопросов расширяется Azure GPT-4 Turbo, ответы также генерируются этой моделью; карточка указывает Lila и DMath как источники | [Карточка Microsoft][ORCA] |
| UltraFeedback Binarized | В upstream на prompt приходятся четыре ответа разных моделей с оценками GPT-4. Лучший `overall_score` выбирается как chosen, один из остальных — как rejected; есть отдельные SFT/prefs/gen splits. Точный способ получения 187 405 строк смеси из этих splits карточка Open-PerfectBlend не уточняет | [Карточка Hugging Face H4][ULTRAFEEDBACK] |
| Evol-CodeAlpaca | CodeAlpaca 20K расширяется десятью стратегиями в стиле WizardCoder; augmentation и ответы получены GPT-4-0314/0613. В релизе фильтруются неанглийские инструкции и некоторые шаблонные самоописания | [Карточка автора][EVOL-CODEALPACA] |
| AutoIF-instruct-61k | Используется опубликованный instruct-корпус AutoIF из проекта Post-Training Data Flywheel. Карточка данного релиза почти не документирует точный запуск генерации; нельзя приписывать ему все параметры произвольной версии AutoIF | [Карточка релиза][AUTOIF]; [проект-источник][FLYWHEEL] |
| LMSYS Arena ShareGPT | Опубликован корпус человеческих сравнений ответов в формате conversations. Карточка конвертированного релиза не раскрывает достаточно деталей, чтобы уверенно восстановить правила выбора ответа и обработки ничьих | [Карточка релиза][ARENA-DATA] |

**Практический вывод из происхождения данных:** случайный split по строкам не гарантирует разделение по исходным задачам. Для строгого собственного эксперимента полезно группировать по `original_question`, `parent_id` или нормализованному prompt там, где эти поля доступны. Это моя рекомендация; публичный DeepSpec делает обычный row-level split.

### 4.8. Почему 1.3M в статье и 1.420909M в карточке — не одно число

Проверенный `download_and_split.py` скачивает `train`, присваивает исходным строкам числовой `id` и вызывает `train_test_split(test_size=0.05, seed=42)`. Для 1 420 909 строк это даёт примерно **1 349 863 train** и **71 046 held-out** до дальнейшей обработки. Это согласуется по порядку с 1.3M, но **не доказывает**, что именно этим объясняется округление в статье: её точные raw revision и промежуточные числа не указаны.

Дальше число пригодных записей ещё меняется из-за ошибок генерации, усечения по длине и фильтра по числу assistant-токенов. Поэтому нужно хранить отдельно `raw_rows`, `train_rows`, `generation_success_rows`, `cache_rows`.

**Источники:** [P2, §4.1][DS-PAPER]; [текущая metadata карточки][PERFECTBLEND-RAW]; [split-код][DS-SPLIT]; [ConversationCollator][DS-CACHE-DATA]. Числа после split — расчёт по этим данным.

### 4.9. Точная последовательность обработки данных в DeepSpec

| Этап | Что делает проверенный код | Что важно проверить |
|---|---|---|
| Загрузка | `load_dataset(dataset_name, split="train")` | В CLI нет `--revision`; ревизию надо фиксировать дополнительно |
| Ограничение для пробы | `select(range(sample_size))` до split | Это первые N строк, а не случайная выборка из полного корпуса |
| ID | Индекс строки записывается в `id` | Должен оставаться устойчивым при восстановлении генерации |
| Split | 95/5, seed 42 | Разделяются беседы-строки; не все варианты одной исходной задачи |
| Нормализация | `human → user`; `gpt/chatgpt/bing/bard → assistant` | Неизвестные роли пропускаются, включая system в этом загрузчике |
| Train JSONL | Сохраняются `id`, `conversations` | Поле upstream `source` этим скриптом теряется |
| Held-out JSONL | Сохраняется `{"turns": [user_turn_1, ...]}` | Здесь только запросы, это ещё не teacher-labeled calibration dataset |
| Regeneration | Все старые assistant-ответы пропускаются; после каждого user вызывается target | Следующие user-ходы видят уже новые target-ответы |
| Токенизация | Chat template target, без повторного добавления special tokens | Проверить реальные токены и assistant spans |
| Truncation | До `data.max_length` токенов полного диалога | `max_tokens` генерации и длина обучающего диалога — разные ограничения |
| Фильтр | По умолчанию нужно минимум 14 loss-токенов после токенизации | Число cache records может быть меньше числа успешных диалогов |
| Capture | BF16 hidden features выбранных слоёв и last hidden state | Это признаки чистого target forward; logits восстанавливаются локально |

**Источники:** [download_and_split.py][DS-SPLIT]; [generate_train_data.py][DS-GEN]; [parser.py][DS-PARSER]; [prepare_target_cache.py][DS-CACHE-PREP]; [target_cache_dataset.py][DS-CACHE-DATA].

### 4.10. Как устроена регенерация multi-turn беседы

Для исходной беседы `user A → old assistant A → user B → old assistant B` код делает:

1. Отправляет `user A` target.
2. Сохраняет `new assistant A`.
3. Отправляет `user A → new assistant A → user B` target.
4. Сохраняет `new assistant B`.

Старые assistant-ответы не остаются в истории. При этом user B не переписывается и может ссылаться на детали старого ответа A. **Моя рекомендация:** отдельно проверить несколько десятков таких диалогов; при сильной рассогласованности использовать single-turn задачи либо явно разработанную фильтрацию. Это не заявленный дополнительный фильтр DSpark.

Деталь воспроизводимости: генератор передаёт сохранённые сообщения как есть, а Qwen parser при подготовке кэша добавляет default system prompt `You are a helpful assistant.`, если в беседе нет system. Для максимально согласованного собственного pipeline system policy нужно выровнять между генерацией и capture; это изменит публичный рецепт и должно быть записано как модификация.

**Источники:** [call_sglang][DS-GEN]; [GeneralParser и Qwen template][DS-PARSER]. Вывод о возможном рассогласовании — анализ этих двух файлов.

### 4.11. Настройки генерации для Qwen3-4B в публичном примере

| Параметр | Значение |
|---|---:|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| min_p | 0 |
| max_tokens на один ответ | 4096 |
| thinking | выключен |
| concurrency в примере | 32 на каждый адрес сервера |

Это параметры **получения обучающих ответов**, а не параметры AdamW и не точная настройка всех evaluation distributions. В P2 acceptance оценивается при temperature 1.0. Для другого target параметры нужно выбирать по его документации и сохранять вместе с данными.

`generate_train_data.py` имеет несколько деталей, которые легко неверно прочитать: `context_length` в его статистике считается через `split()` и означает число слов, не токенов; `--num-samples` ограничивает число отправленных бесед; `--repetition-penalty` в проверенном коде отправляется как `presence_penalty`, поэтому переносить значение механически нельзя. В этом туториале этот флаг не используется.

**Источники:** [официальный data README][DS-DATA]; [CLI и build_query_kwargs][DS-GEN]; [P2, §4.1][DS-PAPER].

### 4.12. Проверка JSONL после генерации

Следующая проверка — **мой дополнительный шаг**, поскольку генератор не делает исчерпывающую валидацию содержимого и не сохраняет `finish_reason` для дальнейшего анализа усечения. Она проверяет ID и структуру, но не доказывает содержательную корректность ответов.

```python
import json
from pathlib import Path

source_path = Path("train_datasets/pilot/perfectblend_train.jsonl")
output_path = Path("train_datasets/pilot/perfectblend_qwen3_4b_regen.jsonl")
source = [json.loads(line) for line in source_path.open()]
rows = [json.loads(line) for line in output_path.open()]
source_by_id = {str(row["id"]): row for row in source}
assert len(source_by_id) == len(source), "Duplicate input IDs"
seen = set()
for row in rows:
    key = str(row["id"])
    assert key in source_by_id and key not in seen, key
    seen.add(key)
    assert row.get("status") == "success", key
    conv = row["conversations"]
    assert len(conv) >= 2 and len(conv) % 2 == 0, key
    for i, msg in enumerate(conv):
        assert msg["role"] == ("user" if i % 2 == 0 else "assistant"), key
        assert isinstance(msg["content"], str) and msg["content"].strip(), key
    original_users = [m["content"] for m in source_by_id[key]["conversations"]
                      if m["role"] == "user"]
    generated_users = [m["content"] for m in conv if m["role"] == "user"]
    assert original_users == generated_users, key
print({"input": len(source), "success": len(rows), "missing": len(source)-len(seen)})
```

Проверка рассчитана на нормализованный Open-PerfectBlend без system/tool сообщений. Для собственного корпуса с такими ролями схема должна быть расширена явно. При ошибках генерации сверяйте success и `_error.jsonl` по множествам ID; одних итоговых счётчиков недостаточно.

**Основание:** [выходной формат и генератор DeepSpec][DS-GEN]. Сам код проверки добавлен для туториала.

### 4.13. Почему не стоит слепо полагаться на `--resume`

В проверенном генераторе DeepSpec `--resume` считает число строк успешного и error output и пропускает столько же строк исходного файла. Результаты параллельных запросов могут записываться не в исходном порядке. При аварийном завершении обработанные ID поэтому не обязаны образовывать сплошной префикс входа: восстановление по одному количеству строк может дать пропуски и повторы.

**Моя рекомендация:** генерировать в свежий output; после прерывания строить pending input по успешным ID, а не по line offset. Минимальный вариант:

```python
import json
from pathlib import Path

src = Path("train_datasets/pilot/perfectblend_train.jsonl")
done_path = Path("train_datasets/pilot/perfectblend_qwen3_4b_regen.jsonl")
pending_path = Path("train_datasets/pilot/pending.jsonl")
done = set()
if done_path.exists():
    for line in done_path.open():
        row = json.loads(line)  # повреждённую последнюю строку сначала разберите отдельно
        if row.get("status") == "success":
            done.add(str(row["id"]))
with pending_path.open("x", encoding="utf-8") as out:
    for line in src.open():
        row = json.loads(line)
        if str(row["id"]) not in done:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
```

Запустить тот же генератор на `pending.jsonl` в **новый** output без `--resume`. Затем объединить успешные результаты с проверкой уникальности ID и соответствия входному списку. Error-ID в `done` не включаются, чтобы их можно было повторить. Не менять target/revision/sampling settings посреди восстановления.

**Источник обнаруженного поведения:** [find_resume_offset, main и запись результатов][DS-GEN]. Это анализ кода и предложенный обход, а не исправление, заявленное авторами.

### 4.14. Что лежит в target cache и почему он огромный

| Поле DeepSpec | Тип / размер на один диалог длины T | Назначение |
|---|---|---|
| input_ids | int32, `T` | Чистая последовательность |
| attention_mask | uint8, `T`, хранится в протоколе | Отметить реальные токены |
| loss_mask | uint8, `T` | Assistant-позиции для supervision |
| target_hidden_states | BF16, `T × (L × d)` | Конкатенация L выбранных decoder-выходов |
| target_last_hidden_states | BF16, `T × d` | Восстановить teacher logits для L1/confidence |

Кэш состоит из бинарных shards, `samples.idx` и `manifest.json`; dataset читает его через mmap. Большой `T × V` массив logits на диск не записывается: frozen LM head применяется к last hidden states при обучении на нужных позициях.

Для Qwen3-4B с `d=2560`, `L=5` грубая формула без служебного индекса:

\[
\text{bytes}\approx\sum_i T_i\,[2d(L+1)+6]
=\sum_i T_i\cdot 30726.
\]

Отсюда диалог длиной 4096 токенов занимает около 120 MiB данных тензоров. Даже небольшие text JSONL превращаются в большой feature cache. Авторы README оценивают **полный default cache примерно в 38 TB**; это их оценка конкретного набора, а не постоянный размер для любого запуска.

Уменьшение `num_anchors` не сокращает сохранённые clean hidden states. Диск экономят меньший корпус, меньшая фактическая длина, меньше capture layers или online feature pipeline. Изменение набора слоёв требует согласованного нового кэша и конфигурации модели.

**Источники:** [cache protocol и размеры тензоров][DS-CACHE-DATA]; [storage warning][DS-DATA]; [Qwen3-4B draft dimensions в SpecForge][SF-DS-CONFIG]; [восстановление teacher logits][DS-MODEL]. Формула и оценка MiB рассчитаны по типам кода.

### 4.15. Минимальный manifest собственных данных

**Рекомендация для собственного эксперимента:** рядом с данными хранить такой JSON, заменив описательные строки реальными значениями:

```json
{
  "dataset": "mlabonne/open-perfectblend",
  "dataset_revision": "actual_huggingface_commit_sha",
  "deepspec_commit": "005e03b81cec38b7da6399833d609ee89a2587f2",
  "target": "Qwen/Qwen3-4B",
  "target_revision": "actual_model_commit_sha",
  "tokenizer_revision": "actual_tokenizer_commit_sha",
  "test_fraction": 0.05,
  "split_seed": 42,
  "thinking": false,
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "min_p": 0,
  "max_new_tokens": 4096,
  "capture_max_length": 4096,
  "target_layer_ids": [1, 9, 17, 25, 33],
  "system_prompt_policy": "record_actual_policy",
  "raw_rows": 0,
  "train_rows": 0,
  "generation_success_rows": 0,
  "generation_error_rows": 0,
  "cache_rows": 0,
  "output_sha256": "actual_sha256"
}
```

Нули здесь — места для фактических счётчиков, а не ожидаемая статистика. ID upstream `source` удобно сохранять отдельной таблицей `id → source → subset/original_task_id`, поскольку стандартный downloader DeepSpec эти поля удаляет.

Важное ограничение: публичные DeepSpec-скрипты не принимают dataset/model revision отдельным CLI-флагом. Для строгой фиксации можно добавить `revision=<SHA>` в конкретные `load_dataset`/`from_pretrained` вызовы или заранее получить immutable snapshot и согласованно использовать его локальный путь. Одной записи SHA в manifest недостаточно, если загрузка всё равно идёт с изменяющегося `main`. Для локального target-пути одно и то же значение должно быть в generation, capture, training и evaluation.

**Основание:** интерфейсы [downloader][DS-SPLIT], [capture][DS-CACHE-PREP] и [train cache validation][DS-CACHE-DATA]. Схема manifest — моя рекомендация.

## 5. Окружение и ресурсы

### 5.1. Что потребуется

Для основного маршрута нужен Linux с NVIDIA GPU и совместимым CUDA/PyTorch. Практический исходный стенд авторов DeepSpec — один узел с восемью GPU. Уменьшить число процессов можно через `CUDA_VISIBLE_DEVICES`, но это **не уменьшает память одного экземпляра модели** и не гарантирует запуск на любой одной видеокарте.

Память расходуется по-разному на этапах:

- Генерация: полный target и его KV cache.
- Capture: полный target и активации; batch-size здесь независим от training batch-size.
- Offline training: draft, замороженные embedding/LM head, FP32 master weights и AdamW states; target для инициализации требует CPU RAM.
- Evaluation: target и draft вместе на каждой GPU-реплике.

Моё практическое предложение — сначала использовать доступную карту с большим объёмом памяти и маленький pilot; точный минимум VRAM без измерения конкретной конфигурации не утверждаю. При OOM сначала уменьшать batch и число anchor-позиций; при нехватке диска — корпус/длину или выбирать online-маршрут.

**Источники:** [DeepSpec README][DS-ROOT]; [capture target placement][DS-CACHE-PREP]; [trainer target initialization][DS-TRAINER]; [BF16Optimizer][DS-OPTIM]; [evaluator model placement][DS-EVALUATOR].

### 5.2. Установка DeepSpec

Команды выполнять в отдельном рабочем каталоге. Версия Python 3.11 ниже — мой выбор для воспроизводимого окружения, а не отдельное требование статьи.

```bash
git clone https://github.com/deepseek-ai/DeepSpec.git
cd DeepSpec
git checkout 005e03b81cec38b7da6399833d609ee89a2587f2

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA:", torch.version.cuda)
print("visible GPU count:", torch.cuda.device_count())
assert torch.cuda.is_available()
PY
```

В проверенном requirements: `torch==2.9.1`, `transformers==5.10.2`, `triton==3.5.1`, `datasets==4.8.5`, `openai==2.6.1` и остальные зафиксированные зависимости. Устанавливать нужно весь файл. Если wheel torch не соответствует драйверу, подобрать совместимую CUDA-сборку и записать отклонение.

`PYTHONPATH` добавлен мной, поскольку отдельные скрипты запускаются из `scripts/data/` и импортируют пакет `deepspec` из корня checkout. Все последующие команды DeepSpec выполняются из его корня.

**Источник:** [requirements.txt][DS-REQ]; [официальная инструкция установки][DS-ROOT]. Создание venv, pin commit и PYTHONPATH — моя адаптация.

### 5.3. Serving engine лучше вынести в отдельное окружение

DeepSpec не включает SGLang в requirements. Для регенерации ответов подходит любой сервер с совместимым `/v1/chat/completions`. Пример авторов использует SGLang.

```bash
python3.11 -m venv .venv-serve
.venv-serve/bin/python -m pip install --upgrade pip
.venv-serve/bin/python -m pip install 'sglang[all]'
.venv-serve/bin/python -m pip freeze > serving-requirements.lock.txt
```

Это адаптация установки из data README. Установка последней версии SGLang не является полным lock: после успешной проверки сохраните конкретную версию/wheels и параметры сервера. Отдельный venv предотвращает перезапись версий torch/transformers тренировочного окружения зависимостями serving engine.

Не переносить старые версии `transformers==4.57.3` из первоначального блога DFlash в окружение текущего DeepSpec: это примеры разных code paths и дат публикации.

**Источники:** [DeepSpec data README, Step 2][DS-DATA]; [исторический quick start Z Lab][DF-BLOG]. Разделение окружений — моя рекомендация.

## 6. Пробный запуск DSpark на небольшом наборе

Цель pilot — проверить весь путь и потребление памяти, а не получить качество статьи. Здесь сознательно уменьшены корпус, capture length, число anchors, global batch и число training steps. Эти значения — **моя стартовая конфигурация** поверх авторского кода.

### 6.1. Подготовить 256 исходных строк и split

В активированном тренировочном окружении из корня DeepSpec:

```bash
mkdir -p train_datasets/pilot logs

python scripts/data/download_and_split.py \
  --dataset-name mlabonne/open-perfectblend \
  --sample-size 256 \
  --test-size 0.05 \
  --seed 42 \
  --train-output-path train_datasets/pilot/perfectblend_train.jsonl \
  --test-output-dir train_datasets/pilot/eval \
  --test-output-name perfectblend.jsonl
```

Ожидаются 243 train-беседы и 13 held-out при успешной обработке всех 256 строк. Это **не случайный 256-row subset полного корпуса**: код сначала берёт первые 256, затем перемешивает их при split. Непустые существующие выходы защищены от перезаписи; для нового опыта используйте отдельный каталог.

**Источник CLI и поведения:** [download_and_split.py][DS-SPLIT]. Размер pilot выбран мной.

### 6.2. Запустить один target server

В отдельном терминале, также из корня DeepSpec:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-serve/bin/python -m sglang.launch_server \
  --model-path Qwen/Qwen3-4B \
  --tp-size 1 \
  --dtype bfloat16 \
  --host 127.0.0.1 \
  --port 30000
```

Дождаться готовности `/health`:

```bash
curl --fail http://127.0.0.1:30000/health
```

Это адаптация авторского восьмисерверного launcher к одной GPU. `CUDA_VISIBLE_DEVICES=0` выбирает одну реплику целого target; это не tensor parallel распределение модели по нескольким GPU.

**Источники:** [launch_sglang_server.sh][DS-SERVE]; [SGLang server examples в SpecForge][SF-DATA].

### 6.3. Сгенерировать ответы target

В терминале с тренировочным venv:

```bash
python scripts/data/generate_train_data.py \
  --model Qwen/Qwen3-4B \
  --server-address 127.0.0.1:30000 \
  --concurrency 4 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0 \
  --max-tokens 4096 \
  --disable-thinking \
  --input-file-path train_datasets/pilot/perfectblend_train.jsonl \
  --output-file-path train_datasets/pilot/perfectblend_qwen3_4b_regen.jsonl
```

Уменьшена только concurrency для пробного сервера; sampling settings сохранены из примера DeepSpec. Выход должен быть свежим: без `--resume` генератор открывает его на запись. Проверить структуру и ID по §4.12, отдельно разобрать `_error.jsonl`.

**Источник:** [официальный Step 2][DS-DATA]; [генератор][DS-GEN]. Одна GPU и concurrency 4 — моя адаптация.

### 6.4. Освободить GPU и собрать небольшой cache

Остановить сервер штатно в его терминале (`Ctrl+C`), если capture будет работать на той же GPU. Затем:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --opts data.max_length=1024 \
  --train-data-path train_datasets/pilot/perfectblend_qwen3_4b_regen.jsonl \
  --output-dir "$PWD/cache/pilot_qwen3_4b" \
  --local-batch-size 1
```

Эта команда вычисляет до 1024 токенов полного диалога, а не только ответа. При длинном prompt ответ может не поместиться; записи с менее чем 14 supervised tokens будут исключены. Для 243 записей верхняя грубая оценка тензоров кэша при 1024 токенах — около 7.1 GiB плюс служебные данные; реальный размер определяется длинами и фильтрами.

```bash
du -sh cache/pilot_qwen3_4b
python - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("cache/pilot_qwen3_4b/manifest.json").read_text())
for key in ["num_samples", "hidden_size", "target_layer_ids", "max_length",
            "target_model_name_or_path", "min_loss_tokens"]:
    print(key, manifest.get(key))
assert manifest["num_samples"] >= 16, "Too few records for pilot global batch"
PY
```

Не использовать один выходной каталог одновременно для несовместимых capture settings. Сам скрипт подготовки кэша не является универсальным resumable downloader.

**Источники:** [prepare_target_cache.py][DS-CACHE-PREP]; [ConversationCollator и cache manifest][DS-CACHE-DATA]. Длина 1024 и batch 1 выбраны мной.

### 6.5. Обучить 50 optimizer steps

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --opts exp_name=dspark_qwen3_4b_pilot \
  --opts "data.target_cache_path=$PWD/cache/pilot_qwen3_4b" \
  --opts data.max_length=1024 \
  --opts model.num_anchors=32 \
  --opts train.local_batch_size=1 \
  --opts train.global_batch_size=16 \
  --opts train.max_train_steps=50 \
  --opts train.torch_compile=false \
  --opts logging.logging_steps=1 \
  --opts logging.checkpointing_steps=25
```

`train.py` сам создаёт worker на каждую видимую GPU: дополнительный `torchrun` здесь не нужен. На одной GPU accumulation равен `16 / (1 × 1) = 16` microsteps на optimizer step. За 50 updates будет больше одного прохода по маленькому pilot: это нормально для проверки работоспособности. При заданном `max_train_steps` код пересчитывает необходимое количество эпох.

`torch_compile=false` убирает компиляцию модели как отдельный фактор отладки, но не превращает Flex Attention в CPU-реализацию. Если остаётся OOM, уменьшить `model.num_anchors` и измерить снова. Обещать качество или скорость по pilot нельзя.

Ожидаемый checkpoint:

```text
~/checkpoints/deepspec/dspark_qwen3_4b_pilot/step_latest
```

**Источники:** [train.py][DS-ENTRY]; [конфиг][DS-CONFIG]; [расчёт accumulation и schedule][DS-TRAINER]; [checkpoint manager][DS-CKPT]. Все уменьшения относительно default — мои.

### 6.6. Проверить оценку pipeline

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path "$HOME/checkpoints/deepspec/dspark_qwen3_4b_pilot/step_latest" \
  --temperature 1.0 \
  --max-new-tokens 128 \
  --confidence-threshold 0.0
```

Это проверка запуска с короткими ответами, не воспроизведение paper evaluation: уменьшен output budget. `eval.py` по умолчанию проходит набор задач, зашитый в `TASKS`; отдельного CLI-флага «один пример» у проверенной версии нет. Для совсем короткого smoke test можно явно сделать отдельную копию evaluator entrypoint с сокращённым `TASKS` и сохранить этот diff. Не стоит придумывать несуществующие `--num-samples`/`--dataset` для данного CLI.

**Источник:** [eval.py, TASKS и parse_args][DS-EVAL-ENTRY]. Короткий output budget — моя адаптация.

## 7. Полное обучение DSpark и baseline DFlash в DeepSpec

### 7.1. Подготовить полный train/held-out split

После успешного pilot и оценки ресурсов выполнить загрузку без `--sample-size`:

```bash
python scripts/data/download_and_split.py \
  --dataset-name mlabonne/open-perfectblend \
  --test-size 0.05 \
  --seed 42 \
  --train-output-path train_datasets/perfectblend_train.jsonl \
  --test-output-dir eval_datasets \
  --test-output-name perfectblend.jsonl \
  --skip-existing
```

`--skip-existing` пропускает работу, только когда уже существуют оба ожидаемых output. Он не проверяет, что файлы соответствуют той же ревизии и тем же настройкам. При изменении источника или split использовать новые пути.

**Источник:** [официальный data README, Step 1][DS-DATA]; [проверка существующих путей][DS-SPLIT].

### 7.2. Масштабировать регенерацию

Если доступны восемь GPU и целый Qwen3-4B помещается на каждой, авторский launcher поднимает восемь независимых реплик на портах 30000–30007:

```bash
# В serving-окружении, из корня DeepSpec:
source .venv-serve/bin/activate
bash scripts/data/launch_sglang_server.sh
```

Затем в тренировочном окружении:

```bash
mkdir -p train_datasets/qwen3_4b

python scripts/data/generate_train_data.py \
  --model Qwen/Qwen3-4B \
  --server-address \
    127.0.0.1:30000 127.0.0.1:30001 \
    127.0.0.1:30002 127.0.0.1:30003 \
    127.0.0.1:30004 127.0.0.1:30005 \
    127.0.0.1:30006 127.0.0.1:30007 \
  --concurrency 32 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0 \
  --max-tokens 4096 \
  --disable-thinking \
  --input-file-path train_datasets/perfectblend_train.jsonl \
  --output-file-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl
```

Команда повторяет настройки README, но намеренно не использует его `--resume`: ограничение восстановления разобрано в §4.13. Список адресов можно сократить до реально запущенных серверов. Concurrency умножается на число доступных адресов, так что восемь адресов по 32 дают до 256 worker threads/запросов в работе.

После завершения сравнить ID, сохранить число success/error и проверить non-empty assistant outputs. Сохранить длительность генерации, параметры target и ревизию. Генерация — отдельная существенная часть затрат; ускорение trainer не убирает этот этап.

**Источники:** [launch_sglang_server.sh][DS-SERVE]; [data README, Step 2][DS-DATA]; [пул worker и queues][DS-GEN]. Проверки и отказ от count-based resume — мои дополнения.

### 7.3. Построить полный target cache

Сначала штатно остановить serving workers, если используются те же GPU. Выбрать файловую систему с достаточным местом и измеренной скоростью чтения/записи.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/data/prepare_target_cache.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --train-data-path train_datasets/qwen3_4b/perfectblend_train_regen.jsonl \
  --output-dir "$PWD/cache/qwen3_4b_target_cache" \
  --local-batch-size 16
```

Это авторская команда с локальным путём output вместо `~/.cache/deepspec`. Максимальная длина берётся из конфига: 4096. Batch 16 — пример для соответствующего железа, не универсальное безопасное значение; при OOM уменьшить до 1–4.

Capture использует **data parallelism**: один целый target на каждую видимую GPU, разные диапазоны входных строк у разных workers. Увеличение числа GPU не распределяет одну слишком большую target-модель по памяти этих устройств. После завершения проверить `manifest.json`, число записей и размеры shards.

**Источники:** [официальный Step 3][DS-DATA]; [capture placement и compute_local_sample_range][DS-CACHE-PREP]; [cache protocol][DS-CACHE-DATA].

### 7.4. Конфигурация DSpark Qwen3-4B

| Группа | Параметр | Значение публичного конфига |
|---|---|---|
| Backbone | draft layers | 5 |
| Backbone | target decoder layer IDs | `[1, 9, 17, 25, 33]` |
| Backbone | block_size | 7 |
| Backbone | mask_token_id | 151669 |
| Sampling | num_anchors | 512 на последовательность, с маскированием невалидных |
| Sequential | markov_rank / type | 256 / vanilla |
| Confidence | alpha / with_markov | 1.0 / true |
| Loss | CE alpha | 0.1 |
| Loss | L1 alpha | 0.9 |
| Loss | loss_decay_gamma | 4.0 |
| Optimizer | AdamW learning rate | `6e-4` |
| Optimizer | weight_decay | 0.0 |
| Schedule | warmup_ratio | 0.04 |
| Schedule | После warmup | cosine annealing |
| Training | num_train_epochs | 10 |
| Training | precision | bf16 |
| Training | local_batch_size | 1 на GPU |
| Training | global_batch_size | 512 |
| Training | max_grad_norm | 1.0 |
| Training | sharding_strategy | no_shard |
| Training | torch_compile | true |
| Data | max_length | 4096 |
| Data | num_workers | 4 на worker |
| Output | logging / checkpoint interval | 10 / 3000 optimizer steps |
| Randomness | seed | 42 |

`loss_decay_gamma=4.0` — **фактический параметр этого конфига**. В P2 формула записана через γ, которым также обозначена длина блока. Не надо автоматически ставить decay=7 только потому, что block_size=7: это уже будет изменение проверенного training recipe.

IDs слоёв здесь — индексы decoder layers. В `transformers.output_hidden_states` нулевой элемент обычно соответствует embedding, поэтому при ручной extraction нельзя механически использовать одинаковые array indices. В DeepSpec cache capture использует hooks, а evaluator — собственную extraction-функцию.

**Источники:** [конфиг DSpark Qwen3-4B][DS-CONFIG]; [BF16Optimizer и cosine scheduler][DS-OPTIM]; [P2, §3.3][DS-PAPER]; [extract_context_feature и validation слоёв][DS-COMMON].

### 7.5. Запустить полное обучение

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --opts "data.target_cache_path=$PWD/cache/qwen3_4b_target_cache"
```

Прямой вызов выбран потому, что в проверенном `scripts/train/train.sh` список GPU и config жёстко записаны внутри команды. Простое внешнее `CUDA_VISIBLE_DEVICES=0 bash scripts/train/train.sh` будет переопределено содержимым wrapper.

Формула accumulation:

\[
G=\frac{B_{\mathrm{global}}}{N_{\mathrm{GPU}}B_{\mathrm{local}}}.
\]

Например, 8 GPU × local batch 1 и global batch 512 дают 64 microsteps на update. Для 4 GPU при том же global batch получится 128. Это сохраняет nominal global batch, но не скорость одного узла. `global_batch_size` должен делиться на произведение world size и local batch.

Число optimizer updates при epochs-режиме:

\[
S_{\mathrm{epoch}}=\left\lfloor\frac{N_{\mathrm{cache}}}{B_{\mathrm{global}}}\right\rfloor,
\qquad S_{\mathrm{total}}=E S_{\mathrm{epoch}}.
\]

Остаток до неполного global batch отбрасывается из расписания эпохи. Если records меньше global batch, запуск завершится assertion. Считать steps надо по **реальному** `manifest.num_samples`, а не по raw размеру Open-PerfectBlend.

**Источники:** [train entrypoint][DS-ENTRY]; [wrapper][DS-TRAIN-SH]; [формулы schedule и assertions][DS-TRAINER].

### 7.6. Мониторинг и checkpoints

```bash
tensorboard --logdir "$HOME/tensorboard/deepspec"
```

По умолчанию:

```text
~/checkpoints/deepspec/dspark_block7_qwen3_4b/step_<N>
~/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest
~/tensorboard/deepspec/dspark_block7_qwen3_4b/
```

Смотреть на CE, L1 и confidence loss отдельно, `accept_rate@k`, `tau_probabilistic`, gradient norm и learning rate. Training `tau_probabilistic` вычисляется на teacher-forced блоках; это удобная диагностика, но не замена реальному decoding acceptance на held-out задачах.

Внешняя auto-evaluation по умолчанию не настроена (`auto_eval_command=None`), поэтому появление checkpoint не означает, что уже выполнен benchmark. Evaluation запускается отдельной командой из §9.

**Источники:** [пути в public constants][DS-CONSTANTS]; [loss metrics][DS-LOSS]; [trainer save/eval logic][DS-TRAINER]; [checkpoint manager][DS-CKPT].

### 7.7. Как работает продолжение обучения

DeepSpec автоматически ищет `step_latest` в каталоге, полученном из `project_name/exp_name`. При его наличии загружаются draft weights, optimizer/scheduler, позиция чтения данных и RNG states.

Повторный запуск той же команды — продолжение существующего опыта. Для нового независимого опыта задавать новое имя:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
  --config config/dspark/dspark_qwen3_4b.py \
  --opts exp_name=dspark_block7_qwen3_4b_run02 \
  --opts "data.target_cache_path=$PWD/cache/qwen3_4b_target_cache"
```

При resume код проверяет, в частности, world size, local batch и gradient accumulation. Не считать перенос на иное число GPU эквивалентным точному продолжению: для такого переноса нужен отдельно разработанный warm-start/конвертация состояния. Не подменять cache и sampling contract в существующем опыте.

**Источники:** [discover_latest_checkpoint и load_training_state][DS-CKPT]; [build_models/schedule][DS-TRAINER]; [порядок обработки --opts и finalize_cfg][DS-CONFIG-UTIL].

### 7.8. Обучить DFlash baseline на том же кэше

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py \
  --config config/dflash/dflash_qwen3_4b.py \
  --opts "data.target_cache_path=$PWD/cache/qwen3_4b_target_cache"
```

В проверенных DSpark и DFlash Qwen3-4B конфигах совпадают target и capture layers, поэтому один clean target cache подходит обоим. Markov отключён через `markov_rank=0`, confidence через `confidence_head_alpha=0.0`, используется `ce_loss_alpha=1.0`, `l1_loss_alpha=0.0`.

Нельзя назвать этот запуск точным воспроизведением P1: другие данные, block convention, max length и epochs. Зато это подходящий baseline для сравнения с DSpark при фиксированных данных и target.

**Источники:** [DFlash config DeepSpec][DS-DF-CONFIG]; [DSpark config][DS-CONFIG]; [совместимость кэша][DS-CACHE-DATA]; [P2, §4.1][DS-PAPER].

## 8. Что происходит внутри обучения и функции потерь

### 8.1. Один training microstep

1. Из mmap cache читаются clean token IDs, masks и target features.
2. Из разрешённых assistant-позиций выбираются случайные anchors.
3. Для каждого anchor строится вход `[anchor, mask, ...]`.
4. Target features объединяются линейным projector и RMSNorm.
5. Draft transformer обрабатывает все упакованные блоки с attention isolation.
6. Frozen LM head переводит draft hidden states в base logits.
7. DSpark добавляет Markov/RNN correction, используя прошлые teacher tokens.
8. Frozen LM head применяется к last hidden states target на согласованных позициях.
9. Вычисляются position-weighted CE, L1 и confidence BCE.
10. Backpropagation обновляет trainable draft-параметры; по окончании accumulation выполняются gradient clipping и AdamW step.

В DeepSpec ограничено максимум 512 anchors по умолчанию. Если валидных позиций меньше, недостающие блоки маскируются. Для допустимого anchor требуется loss mask и на anchor, и на первом предсказываемом токене; хвостовые labels учитываются только пока остаются непрерывным разрешённым префиксом. Блок не должен продолжать supervision сквозь следующий user-turn.

**Источники:** [DSpark model forward][DS-MODEL]; [anchor sampling / eval masks][DS-COMMON]; [loss][DS-LOSS]; [training loop][DS-TRAINER].

### 8.2. DFlash loss

Пусть `j=0..K-1` нумерует предсказываемые токены блока, а `g` — **отдельный коэффициент decay**, чтобы не путать его с длиной блока:

\[
w_j=\exp(-j/g),\qquad
\mathcal L_{CE}=-\sum_j w_j\log p^d_j(x_j^*).
\]

Позиции в начале блока сильнее влияют на acceptance: первая ошибка обрывает принимаемый префикс. В P1 для input block 16 используется decay 7, для block 10 — 5, для block 8 — 4. Не применять loss к известному anchor как к label того же токена.

**Источники:** [P1, формула 4 и A.1][DF-PAPER]; для DeepSpec baseline — [CE-only config][DS-DF-CONFIG].

### 8.3. DSpark loss: CE + L1 + confidence

\[
\mathcal L=0.1\mathcal L_{CE}+0.9\mathcal L_{L1}
+1.0\mathcal L_{conf}.
\]

\[
\mathcal L_{L1}=\sum_j w_j\|p^d_j-p^t_j\|_1.
\]

Хотя статья называет этот член TV loss, в формуле и коде штрафуется полная L1-норма. Собственно total variation равна **половине** этой нормы. Если самостоятельно добавить `0.5` внутрь loss, сохранив коэффициент 0.9, эффективный баланс objective изменится.

Soft label confidence:

\[
c_j^*=1-\tfrac12\|p^d_j-p^t_j\|_1.
\]

\[
\mathcal L_{conf}=\sum_j w_j\operatorname{BCE}(c_j,c_j^*).
\]

В коде используется `binary_cross_entropy_with_logits`, а target label `accept_rate.detach()` не передаёт градиент обратно через целевую вероятность. CE и L1 считаются по distribution после последовательной коррекции logits. Соответствующие position weights применяются только в разрешённых позициях.

Код нормирует loss по сумме действительных weights и согласует знаменатели между distributed ranks, поэтому численные значения логов не нужно сравнивать напрямую с ненормированной суммой из формулы статьи.

**Источники:** [P2, §3.3, формулы 9–12][DS-PAPER]; [compute_dspark_loss][DS-LOSS]; [коэффициенты публичного конфига][DS-CONFIG].

### 8.4. Что означает confidence

`c_j` оценивает принятие позиции **при условии выживания предыдущего префикса**. Вероятность принятия всего префикса приблизительно оценивается произведением:

\[
s_k=\prod_{j=1}^{k}c_j,\qquad
\widehat\tau(\ell)=1+\sum_{k=1}^{\ell}s_k.
\]

Единица соответствует token, который обеспечивает target verification. Это не оценка истинности ответа и не probability выбранного draft-token из softmax. Подстановка `max softmax probability` вместо confidence head — другой алгоритм.

**Источники:** [P2, §3.2, формулы 7–8 и scheduler][DS-PAPER]; [confidence metrics][DS-LOSS].

## 9. Оценка модели и confidence

### 9.1. Основной acceptance benchmark

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
  --target_name_or_path Qwen/Qwen3-4B \
  --draft_name_or_path "$HOME/checkpoints/deepspec/dspark_block7_qwen3_4b/step_latest" \
  --temperature 1.0 \
  --max-new-tokens 2048 \
  --confidence-threshold 0.0
```

Это соответствует default output budget evaluator и temperature из P2. Если доступна одна GPU, оставить `CUDA_VISIBLE_DEVICES=0`; каждый evaluator worker держит target и draft целиком. Дополнительный `torchrun` не нужен.

Проверенный набор задач:

| Домен | Задачи |
|---|---|
| Math | GSM8K, MATH500, AIME25 |
| Code | HumanEval, MBPP, LiveCodeBench |
| Chat | MT-Bench, Alpaca, Arena-Hard-v2 |

P2 использует non-thinking. В evaluator DeepSpec Qwen chat prompt также кодируется с `enable_thinking=False`. Проверяйте это отдельно при переносе на другой target/renderer.

**Источники:** [eval.py][DS-EVAL-ENTRY]; [base evaluator prompt rendering][DS-BASE-EVAL]; [P2, §4.1][DS-PAPER].

### 9.2. Как читать метрики

| Метрика | Значение | Что не следует из неё автоматически |
|---|---|---|
| `accept_len` / τ | Среднее число выданных токенов на round, включая target bonus | Что время уменьшилось ровно во столько же раз |
| `#propose` | Средняя предложенная длина, в таблице дополнительно показывается `+1` | Что все предложенные токены были приняты |
| `verify_rate` DeepSpec | Сумма accept lengths / сумма `(proposal_length + 1)` | Что это чистая draft-only доля accepted/proposed |
| `accept_rate@k` | Эмпирическая статистика принятия соответствующей позиции | Что это training soft overlap на teacher prefixes |
| `tau_probabilistic` train | Оценка через произведения distribution overlaps | Что таков будет τ после rollout модели |

Формулу `verify_rate` стоит особенно учитывать при сравнении с чужими таблицами: в числитель и знаменатель включён target token. При максимуме 7 draft proposals τ находится примерно в диапазоне 1–8, за исключением влияния остановки по EOS/бюджету.

**Источники:** [build_dataset_metrics и таблица результатов][DS-BASE-EVAL]; [training probabilistic stats][DS-LOSS]; [сноска P2 про bonus token][DS-PAPER].

### 9.3. Acceptance и end-to-end speedup измеряются отдельно

Средняя задержка на выданный токен:

\[
L\approx\frac{T_{draft}+T_{verify}}{\tau}.
\]

Больший τ может сопровождаться более дорогим draft или проверкой, особенно при высокой concurrency. Поэтому для реального deployment нужно сравнить target-only и target+draft на одинаковых prompts, decoding settings, GPU, precision, concurrency и output budgets. Отдельно измерять TPOT/inter-token latency, tokens/s, TTFT и потребление памяти.

DeepSpec здесь прежде всего даёт алгоритмическую оценку acceptance. Продакшен-ускорения P2 для DeepSeek-V4 не переносятся автоматически на Qwen3-4B, другой serving engine или домашнюю GPU. В P2 production-модель также отличается архитектурой: три MoE-слоя, sliding window 128 и максимальный block 5, а не публичный пятислойный Qwen draft.

**Источники:** [P1, §3, формула 1][DF-PAPER]; [P2, §5.1 и §5.4][DS-PAPER]. Список измерений — моя рекомендация для честного сравнения.

### 9.4. Confidence calibration: что есть в статье и коде

P2 предлагает **Sequential Temperature Scaling (STS)**: на отдельном validation set последовательно подбирать temperature для каждой позиции по ECE накопленного произведения confidence. Температуры предыдущих позиций фиксируются, затем калибруется следующая.

В проверенном DeepSpec доступны recorder и графики cumulative confidence с ECE/AUROC/Brier. При `--confidence-threshold 0.0` статистика собирается; при ненулевом threshold соответствующий recorder отключается. Наличие этих метрик **не означает**, что автоматический STS fitting и полный hardware-aware production scheduler уже запускаются этим CLI. Готовой команды `calibrate.py`/`--sts` в проверенном пути нет.

Практический порядок собственной реализации STS:

1. Отложить отдельные calibration prompts, не использованные в gradient updates.
2. Выполнить реальные draft/target rollouts без confidence-pruning.
3. Собирать confidence logits, фактическое выживание префиксов и условия decoding.
4. Для k слева направо выполнить 1D grid search температуры; score — ECE кумулятивного prefix probability, как в P2.
5. Проверить reliability на другом held-out наборе, сохранив сами температуры и decoding settings.
6. Передавать calibrated confidence scheduler, поддерживающему этот contract.

Это описание реализации **по статье**, не обещание готовой исполнимой команды DeepSpec. Для пробы threshold-режима доступен `--confidence-threshold`, но фиксированный threshold не эквивалентен STS плюс hardware-aware scheduler.

**Источники:** [P2, §3.2.1][DS-PAPER]; [ConfidenceHeadRecorder][DS-CONFIDENCE]; [условия включения recorder][DS-EVALUATOR]; [eval CLI][DS-EVAL-ENTRY].

### 9.5. Сохранение распределения при adaptive verification

Если реализовывать scheduler самостоятельно, выбор длины не должен зависеть от ещё не допущенных будущих draft-кандидатов. P2 специально разбирает non-anticipating constraint: ретроспективный глобальный поиск лучшего префикса с использованием будущих candidate-dependent confidences может нарушить lossless guarantee. Статья использует причинное раннее завершение поиска; production-адаптация отдельно описана в §5.2.

При temperature > 0 корректная verification требует rejection sampling с согласованными draft/target probabilities; простое сравнение с target argmax для этого режима не подходит. При greedy decoding схема другая. Для учебного обучения достаточно штатного evaluator, а serving implementation стоит проверять отдельно.

**Источники:** [P2, §3.2.2, §5.2 и приложение A][DS-PAPER]; [DeepSpec verification][DS-BASE-EVAL].

## 10. DFlash через SpecForge: ближе к оригинальной архитектуре

### 10.1. Когда выбирать этот маршрут

SpecForge содержит самостоятельную архитектуру DFlash, block 16 и CE loss с decay 7 для Qwen3-8B. Это подходящая основа, когда нужно обучать вариант с original mask-position semantics или затем идти к SGLang-serving workflow.

При этом checked-in пример на ShareGPT **не воспроизводит исходные 800K Nemotron+CodeAlpaca**. Даже если выставить все гиперпараметры P1, отличие training data останется. Для точного заявления о reproduction нужно отдельно зафиксировать происхождение данных и неизвестные части исходного рецепта.

**Источники:** [SpecForge DFlash model config][SF-DF-CONFIG]; [offline YAML][SF-DF-OFFLINE]; [P1, §4.2 и §5][DF-PAPER].

### 10.2. Установка в отдельном checkout и venv

```bash
git clone https://github.com/sgl-project/SpecForge.git
cd SpecForge
git checkout 8b45bab869ef6b9d539a2626629ad31af51491f7

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[data]'
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

В этом snapshot pyproject фиксирует другие версии: torch 2.13.0, transformers 5.12.1, SGLang 0.5.18. Не устанавливать их поверх DeepSpec venv. Приведён pip-вариант авторской установки из source; GPU/CUDA совместимость всё равно проверяется на конкретном хосте.

**Источники:** [SpecForge installation][SF-INSTALL]; [pyproject.toml][SF-PYPROJECT]. Pin commit и отдельный venv — адаптация для воспроизводимости.

### 10.3. Подготовить пример данных

```bash
python scripts/prepare_data.py \
  --dataset sharegpt \
  --sample-size 1000 \
  --split-eval
```

Это небольшой **пример проверки pipeline**, не training corpus статьи. Output по умолчанию находится в `cache/dataset/`. Для полного собственного корпуса использовать нормализованный conversation JSONL:

```json
{"id":"example-001","conversations":[{"role":"user","content":"Напиши функцию сортировки."},{"role":"assistant","content":"Ответ вашего target."}]}
```

Можно также взять уже регенерированный Open-PerfectBlend из DeepSpec, предварительно проверив схему, преобразовав ID в непустые строки и записав абсолютный путь в SpecForge config. Однако для P1 нужны данные из соответствующей собственной смеси §4.5.

**Источники:** [канонические presets и JSONL contract][SF-DATA]; [prepare_data.py][SF-PREP-DATA]. Размер 1000 выбран мной для пробы.

### 10.4. Регенерировать ответы именно Qwen3-8B

Запустить Qwen3-8B через serving environment, как в §6.2, заменив model name. Затем в SpecForge venv:

```bash
python scripts/regenerate_train_data.py \
  --model Qwen/Qwen3-8B \
  --server-address localhost:30000 \
  --concurrency 8 \
  --temperature 0.7 \
  --max-tokens 4096 \
  --reasoning disable \
  --input-file-path cache/dataset/sharegpt_train.jsonl \
  --output-file-path cache/dataset/sharegpt_qwen3_8b_regen.jsonl
```

Это **моя адаптация** публичного regeneration utility для небольшого non-thinking опыта. Параметры не выдаются за оригинальные sampling settings P1. Для согласованности с конкретным экспериментом проверить также top-p/top-k и их фактические defaults в выбранном server/utility; полные settings записывать в manifest.

В текущем SpecForge есть более строгие готовые Qwen ShareGPT regeneration wrappers с accounting success/error/skipped; их non-reasoning profile использует temperature 0. Эти profiles полезны как инженерный пример, но отличаются от DSpark regeneration temperature 0.7. Не смешивать corpora без записи их происхождения.

После генерации проверить outputs и штатно остановить server перед offline capture на тех же GPU.

**Источники:** [Regenerate Datasets и Qwen recipes][SF-DATA]; [regenerate_train_data.py][SF-GEN].

### 10.5. Подготовить DFlash hidden states

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/prepare_hidden_states.py \
  --strategy dflash \
  --target-model-path Qwen/Qwen3-8B \
  --draft-model-config configs/qwen3-8b-dflash.json \
  --data-path cache/dataset/sharegpt_qwen3_8b_regen.jsonl \
  --output-path cache/hidden_states/qwen3-8b-dflash-regen \
  --chat-template qwen \
  --max-length 3072 \
  --tp-size 1 \
  --batch-size 1
```

Здесь `torchrun` **нужен**, потому что это другой entrypoint, не DeepSpec. Это адаптация документированного capture example с уменьшенным batch и одной GPU.

Формат SpecForge отличается от DeepSpec binary cache: для DFlash record содержит `input_ids`, `loss_mask`, `hidden_states` в `.ckpt`/`.ckpt.gz`. DSpark record дополнительно требует `target_last_hidden_states`. Передача каталога DeepSpec mmap cache напрямую в `data.hidden_states_path` SpecForge не поддерживается этим рецептом.

**Источники:** [Prepare Hidden States][SF-DATA]; [prepare_hidden_states.py][SF-CAPTURE].

### 10.6. Запустить offline DFlash

Для небольшого опыта:

```bash
specforge train \
  --config examples/configs/offline/colocated/qwen3-8b-dflash.yaml \
  data.hidden_states_path=./cache/hidden_states/qwen3-8b-dflash-regen \
  training.batch_size=1 \
  training.num_anchors=32 \
  training.max_steps=50 \
  run_id=qwen3-8b-dflash-pilot \
  output_dir=./outputs/qwen3-8b-dflash-pilot
```

Для более полного опыта с большим подготовленным корпусом:

```bash
specforge train \
  --config examples/configs/offline/colocated/qwen3-8b-dflash.yaml \
  data.hidden_states_path=./cache/hidden_states/qwen3-8b-dflash-regen \
  training.num_epochs=6 \
  training.max_steps=null \
  run_id=qwen3-8b-dflash-custom6ep \
  output_dir=./outputs/qwen3-8b-dflash-custom6ep
```

В checked-in YAML стоит `max_steps: 10000`: это stop cap. Для намерения пройти все шесть эпох его надо убрать/поставить `null`, как показано выше, иначе можно закончить раньше. Данные для полного опыта нужно подготовить заново без ограничения 1000 rows; эти команды сами корпус не расширяют.

Исходные основные P1 settings: 5 слоёв, block16, 5 feature layers, 512 anchors, AdamW LR6e-4, cosine, warmup4%, clip1, max length3072, 6 epochs. Для Qwen3-Coder в P1 используются 8 draft layers и length4096; для LLaMA 3.1 block10 и decay5. Не переносить Qwen3-8B config на них заменой только имени.

**Источники:** [P1, §5 и A.1][DF-PAPER]; [offline YAML][SF-DF-OFFLINE]; [SpecForge CLI overrides и training schedule][SF-TRAINING]. Уменьшенный pilot — моя адаптация.

### 10.7. DSpark через SpecForge

Для DSpark Qwen3-4B есть отдельные draft config и offline recipe. После соответствующей регенерации Qwen3-4B:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  scripts/prepare_hidden_states.py \
  --strategy dspark \
  --target-model-path Qwen/Qwen3-4B \
  --draft-model-config configs/qwen3-4b-dspark.json \
  --data-path /absolute/path/to/qwen3_4b_regenerated.jsonl \
  --output-path cache/hidden_states/qwen3-4b-dspark-custom \
  --chat-template qwen \
  --max-length 3072 \
  --tp-size 1 \
  --batch-size 1

specforge train \
  --config examples/configs/offline/colocated/qwen3-4b-dspark.yaml \
  data.hidden_states_path=./cache/hidden_states/qwen3-4b-dspark-custom \
  training.num_anchors=32 \
  training.max_steps=50 \
  run_id=qwen3-4b-dspark-pilot \
  output_dir=./outputs/qwen3-4b-dspark-pilot
```

`/absolute/path/to/...` нужно заменить реальным путём. Это альтернативный pipeline SpecForge, а не точный DeepSpec Table 1 recipe: checked-in recipe отличается длиной, эпохами и default stop cap. Для сравнения авторских checkpoints удобнее основной DeepSpec-маршрут §7.

**Источники:** [DSpark config SpecForge][SF-DS-CONFIG]; [offline DSpark recipe][SF-DS-OFFLINE]; [strategy-specific feature contract][SF-DATA].

## 11. Online-обучение без полного постоянного кэша

### 11.1. Два разных значения слова online

Необходимо различать:

- **Online features:** target hidden states вычисляются по мере потребления последовательностей trainer.
- **Online text regeneration:** сами ответы target генерируются заново по мере обучения.

Возможен online feature pipeline на **заранее регенерированных текстах**. Именно поэтому наличие online training не отменяет автоматически сбор target-generated JSONL. В SpecForge выбор источника features и политика получения текста — отдельные решения.

**Источники:** [P1, A.1][DF-PAPER]; [блог SpecForge v0.3, Feature source is not data policy][SF-BLOG-2026]; [disaggregated guide][SF-DISAGG].

### 11.2. Архитектура producer/consumer

В текущем SpecForge online pipeline разделяет:

1. Capture servers: SGLang вычисляет target features.
2. Producer: назначает samples и публикует ссылки на feature objects.
3. Mooncake: хранит/передаёт большие тензоры.
4. Consumer trainer: читает features, обучает draft и подтверждает потреблённые optimizer windows.

Так можно не хранить полный многотерабайтный corpus features постоянно, но остаются память буферов, временное хранение и стоимость повторного capture. Баланс числа server/trainer GPU влияет на throughput обучения.

Блог от августа описывает capture patch для SGLang 0.5.14; проверенный commit уже фиксирует 0.5.18 и содержит свои patch/plugin пути. Старую команду patch из блога нельзя смешивать с новым checkout. Источником команд служат текущие installation/disaggregated docs.

**Источники:** [авторский блог v0.3][SF-BLOG-2026]; [актуальный disaggregated guide][SF-DISAGG]; [pyproject][SF-PYPROJECT]; [capture plugin README][SF-CAPTURE-PLUGIN].

### 11.3. Конкретный managed-local пример для DFlash

Проверенный пример `qwen3-8b-dflash-1server-dp7.yaml` рассчитан на восемь GPU: target server на GPU0, семь trainer ranks на GPU1–7. Требуются совместимый Mooncake и сервер с нужным capture mechanism; управляемый launcher отвечает за lifecycle, но не исправляет несовместимую установку зависимостей.

В окружении SpecForge установить подходящий Mooncake client согласно текущей документации: для CUDA до 13 — `mooncake-transfer-engine`, для CUDA13+ — `mooncake-transfer-engine-cuda13`. Не устанавливать оба варианта без необходимости.

Для **уже регенерированного Qwen3-8B JSONL** сначала посмотреть план:

```bash
specforge train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7.yaml \
  data.train_data_path=./cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
  --plan
```

После проверки соответствия GPU-топологии, путей, server capture setup и fresh output/control directories:

```bash
specforge train \
  -c examples/configs/online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7.yaml \
  data.train_data_path=./cache/dataset/perfectblend_qwen3-8b_regen.jsonl
```

Это конкретная текущая инфраструктурная конфигурация с `num_epochs=10`, `max_steps=10000`, batch2 и decay7. Она не становится оригинальным шестиипоховым P1 recipe только потому, что strategy называется dflash. Для малого оборудования надо изменять и YAML trainer topology, и managed-local GPU lists; одного внешнего `CUDA_VISIBLE_DEVICES` недостаточно.

**Источники:** [сам managed-local YAML][SF-DF-ONLINE]; [планирование и запуск сервисов][SF-DISAGG].

### 11.4. Online DSpark

Для Qwen3-4B есть `online/disaggregated/external/qwen3-4b-dspark.yaml`. Его server URL на localhost означает, что **Mooncake и capture server уже запущены пользователем или scheduler**; слово external обозначает владельца lifecycle, а не обязательное нахождение на другой машине.

```bash
specforge train \
  -c examples/configs/online/disaggregated/external/qwen3-4b-dspark.yaml \
  data.train_data_path=/absolute/path/to/qwen3_4b_regenerated.jsonl \
  --plan
```

Запуск без `--plan` имеет смысл после настройки существующих Mooncake endpoints и совместимого DSpark capture server. Обычный `/v1`-сервер из шага регенерации не заменяет feature capture service. Не приводится выдуманная универсальная команда сервера: в проверенном коде DSpark имеет отдельный capture contract, который зависит от поддержанной версии serving engine.

**Источники:** [online DSpark YAML][SF-DS-ONLINE]; [external service contract и DSpark capture][SF-DISAGG].

## 12. Перенос на другой target или собственный домен

### 12.1. Другой target в DeepSpec

Публичные конфиги есть для Qwen3-4B/8B/14B и Gemma4-12B. Порядок переноса:

1. Выбрать соответствующий config, а не менять произвольное имя в Qwen3-4B config.
2. Перегенерировать ответы новым target с его sampling/thinking settings.
3. Снова построить cache по его tokenizer, hidden dimensions и feature layers.
4. Обучить draft с соответствующими frozen embedding и LM head.
5. Запустить evaluator с тем же target и совместимым draft.

Сохранить чужой cache при смене target нельзя. Даже при одинаковой форме тензоров они принадлежат другой модели. Если нужны строго pinned weights, строка target path должна совпадать на этапах capture/training: validator сравнивает её как строку.

Для Gemma pipeline имеет отдельные parser details, включая маркер non-thinking assistant prefix. Не переносить Qwen special tokens и mask ID вручную.

**Источники:** [каталог конфигов и поддержанные модели][DS-ROOT]; [cache validation][DS-CACHE-DATA]; [parser и Gemma prefix][DS-PARSER].

### 12.2. Собственные prompts

Обычно полезнее адаптировать **prompt distribution** под реальную нагрузку, чем просто увеличивать число случайных public samples. Для кода — включать реальные форматы файлов, языки, длины и продолжения; для чата — реальные типы многоходовых запросов; для reasoning — соответствующий режим и длины трасс.

Практический pipeline:

1. Подготовить разрешённые к использованию user prompts и source IDs.
2. Разделить задачи/беседы на train/validation до генерации.
3. Сгенерировать target responses в нужном режиме.
4. Измерить долю усечённых и слишком коротких диалогов.
5. Повторить feature capture и обучение с новым experiment name.
6. Сравнить acceptance на своём held-out домене и на общем benchmark.

Не нужно автоматически отбрасывать любой математически неверный ответ target: draft обучается согласованности с target, а verifier обеспечивает target distribution. Фильтрация промптов, пустых результатов и разрушенных форматов — другая задача; любое дополнительное quality filtering надо описывать и оценивать как изменение data distribution.

**Источники принципа:** [P1, target alignment][DF-PAPER]; [P2, training data][DS-PAPER]; [DeepSpec recommendation про domain-specific fine-tuning][DS-ROOT]. Конкретный порядок адаптации — моя рекомендация.

### 12.3. Thinking mode

Опубликованные основные DeepSpec checkpoints и эксперименты P2 ориентированы на non-thinking. Для thinking-нагрузки недостаточно просто включить флаг на serving:

- Нужно получить ответы target в thinking mode и сохранить фактические reasoning traces.
- Проверить их сериализацию в chat template, а не только наличие JSON-поля.
- Пересмотреть max length, долю усечения и необходимый объём cache.
- Обучить/дообучить draft и заново оценить acceptance/confidence.

В DeepSpec генератор имеет `--is-reasoning-model` и сохраняет `thinking`, но сам факт наличия этого поля не доказывает, что downstream tokenizer renderer включит его в обучаемый текст. Это нужно проверить по decoded tokens. У SpecForge есть отдельные utilities для structured multi-turn reasoning и превращения беседы в generation-event rows, когда используется last-turn-only supervision.

**Источники:** [DeepSpec README о thinking/domain adaptation][DS-ROOT]; [reasoning field генератора][DS-GEN]; [rendering DeepSpec][DS-PARSER]; [SpecForge structured reasoning section][SF-DATA].

## 13. Типичные проблемы и проверка воспроизводимости

### 13.1. Диагностика по симптомам

| Симптом | Возможная причина | Что проверить | Источник |
|---|---|---|---|
| `ModuleNotFoundError: deepspec` | Скрипт запущен вне настроенного checkout | Корень repo, активный venv, PYTHONPATH | [Структура entrypoints][DS-ENTRY] |
| Training почти не использует GPU | Медленное чтение огромных cache shards | I/O throughput, workers, mmap/cache locality | [CacheDataset][DS-CACHE-DATA] |
| Capture OOM | Слишком большой local batch чистого target | Уменьшить capture batch отдельно от train batch | [Capture script][DS-CACHE-PREP] |
| Training OOM | Много anchor logits, optimizer states, большой local batch | Уменьшать anchors/local batch; оценивать full-vocab tensors | [Model forward][DS-MODEL], [optimizer][DS-OPTIM] |
| Cache значительно меньше train JSONL | Ошибки generation или меньше 14 supervised tokens после truncation | Счётчики этапов, loss_mask, actual token lengths | [Collator][DS-CACHE-DATA] |
| Loss очень хороший, rollout acceptance плохой | Leakage, неверный label shift, mismatch режима или domain | Attention mask, teacher logits t−1, thinking, target revision | [Mask code][DS-COMMON], [model][DS-MODEL] |
| Неожиданно продолжается старый run | Найден существующий step_latest | Новое exp_name для нового опыта | [Checkpoint discovery][DS-CKPT] |
| Resume падает после изменения GPU count | Incompatible saved distributed state | World size/local batch/accumulation | [Resume validation][DS-CKPT] |
| DFlash и DSpark cache «одинаковые», но SpecForge не читает | Разный binary/file protocol | Создать features в ожидаемом формате каждого framework | [DeepSpec protocol][DS-CACHE-DATA], [SpecForge contract][SF-DATA] |
| Увеличенный τ не ускорил сервер | Draft/verification latency и высокая нагрузка | Измерить end-to-end speedup при своей concurrency | [P1, §3][DF-PAPER], [P2, §3.2][DS-PAPER] |
| Online run ждёт features | Не запущены external services или неверный capture contract | Resolved plan, Mooncake, совместимый server capture | [Disaggregated docs][SF-DISAGG] |
| Дубли/пропуски после generation resume | Восстановление по числу строк при параллельной записи | Повторить pending по ID, проверить объединение | [Generator main][DS-GEN] |

Причины в таблице — направления диагностики, а не доказанный диагноз любого конкретного сбоя.

### 13.2. Что сохранить перед сравнением результатов

Это мой минимальный checklist воспроизводимости:

- Commit кода, Python/CUDA/torch/transformers/serving versions.
- Immutable target и tokenizer revisions, веса draft и training config.
- Dataset revision, source map и точные ID train/validation/calibration.
- Политику system prompt, thinking и sampling settings.
- Raw/train/success/error/cache counts и долю truncated records.
- Feature layer IDs, block convention, mask ID и max length.
- Global/local batch, world size, accumulation, epochs/steps, seeds.
- Отдельные acceptance, confidence calibration и реальные speed benchmarks.
- Хэши итоговых JSONL, cache manifest и checkpoints.

Смысл этих записей — дать возможность понять, **что именно изменилось**, когда acceptance стала лучше или хуже. Одной фразы «обучалось по статье DFlash» для этого недостаточно.

### 13.3. Что в этом туториале не выдаётся за воспроизведённое

| Часть | Статус |
|---|---|
| Архитектура, формулы и гиперпараметры приложенных статей | Разобраны по заданным версиям PDF |
| CLI и data pipeline DeepSpec | Проверены по зафиксированному исходному коду |
| DFlash/DSpark маршруты SpecForge | Проверены по конфигам, scripts и docs snapshot |
| Первый 256-row/50-step pilot | Предложен мной; GPU-запуск не выполнялся |
| Исходные 800K DFlash до последнего sample ID | Не восстановлены: точный manifest в проверенных источниках отсутствует |
| Исходная сборка Open-PerfectBlend до всех деталей dedup | Перечислены опубликованные компоненты и ограничения документации |
| STS + production hardware scheduler | Разобран алгоритм; не заявлено наличие готового полного публичного CLI |
| Продакшен-результаты DeepSeek-V4 | Не воспроизводились и не переносятся автоматически на Qwen |

## 14. Карта шагов и источников

| Практический шаг | Откуда взят | Что добавлено/уточнено в туториале |
|---|---|---|
| Target-generated ответы | P1 §5; P2 §4.1 | Разделены prompt corpus и supervision corpus |
| Open-PerfectBlend 95/5 | DeepSpec downloader | Первые N в pilot и потеря source metadata |
| Регенерация всех assistant-ходов | DeepSpec generator | Проверка user history и ID-based восстановление |
| Qwen3-4B sampling 0.7/0.8/20 | DeepSpec data README | Отделено от eval temperature1 |
| Capture пяти feature layers | P1; DeepSpec configs/cache script | Разобрана decoder/hidden_states индексация |
| Хранение last hidden вместо logits | DeepSpec cache/model; P2 §5.1 объясняет сходный принцип для communication | Рассчитан порядок затрат диска |
| Anchor blocks и attention isolation | P1 §4.2; DeepSpec common/model | Показаны точные shifts и запрет leakage |
| CE и positional decay | P1 формула4; DeepSpec DFlash config | Выделено отличие block size и decay constant |
| Markov rank256 | P2 §3.1; DeepSpec Markov/config | Объяснены training teacher forcing и inference sampling |
| CE0.1/L10.9/confidence1 | P2 §3.3; DeepSpec loss/config | Уточнён множитель 1/2 для TV |
| AdamW/cosine/LR6e-4 | P1 A.1; DeepSpec optimizer/config | Проверены конкретные defaults |
| Train launch и accumulation | DeepSpec train.py/base_trainer | Без лишнего torchrun, с формулами batch/steps |
| Checkpoints и resume | DeepSpec ckpt_manager | Отдельное exp_name для независимых runs |
| Acceptance benchmark | P2 §4.1; DeepSpec eval.py | Уточнены bonus token и verify_rate |
| Confidence calibration | P2 §3.2.1 | Отделён алгоритм STS от доступного recorder |
| Offline DFlash в SpecForge | Текущие configs/data/training docs | Отмечено отличие ShareGPT от P1 data mixture |
| Online producer/consumer | Блог LMSYS v0.3; текущие SpecForge docs | Разделены text regeneration и feature capture |
| Исходные Nemotron/CodeAlpaca | Карточка NVIDIA; README CodeAlpaca | Не приписаны DFlash неизвестные subset quotas |
| Источники Open-PerfectBlend | Upstream card и карточки компонентов | Разделены upstream creation и DSpark response regeneration |

## 15. Ссылки на первоисточники

Ссылки ниже являются Markdown references, используемыми прямо возле соответствующих шагов. Для репозиториев ссылки ведут на проверенный commit; для статей — на фиксированную arXiv-версию. Внешние dataset cards приведены в состоянии, доступном на дату проверки.

### Статьи и авторские блоги

- [P1 — DFlash, arXiv:2602.06036v2][DF-PAPER]. Основные места: §4.2, стр. 5; §5, стр. 6; A.1, стр. 12.
- [P2 — DSpark, arXiv:2607.05147v1][DS-PAPER]. Основные места: §3.1, стр. 5–6; §3.2, стр. 7–9; §3.3, стр. 9–10; §4.1, стр. 10; §5.1, стр. 16.
- [Z Lab — DFlash project blog][DF-BLOG].
- [LMSYS — SpecForge, 25 July 2025][SF-BLOG-2025]. Исторический материал об EAGLE3 и online/offline features.
- [LMSYS — SpecForge v0.3, 4 August 2026][SF-BLOG-2026]. Producer/consumer, capture/delivery/lifecycle; версии команд нужно сверять с текущим кодом.

### Код DeepSpec

- [README][DS-ROOT], [requirements][DS-REQ], [public constants][DS-CONSTANTS].
- [Data README][DS-DATA], [download/split][DS-SPLIT], [regeneration][DS-GEN], [server launcher][DS-SERVE].
- [Cache preparation][DS-CACHE-PREP], [cache protocol/dataset/collators][DS-CACHE-DATA], [parser][DS-PARSER].
- [DSpark Qwen3-4B config][DS-CONFIG], [DFlash Qwen3-4B config][DS-DF-CONFIG].
- [Train entrypoint][DS-ENTRY], [train wrapper][DS-TRAIN-SH], [base trainer][DS-TRAINER], [optimizer][DS-OPTIM], [checkpoints][DS-CKPT], [config overrides][DS-CONFIG-UTIL].
- [DSpark Qwen3 model][DS-MODEL], [blocks/masks][DS-COMMON], [Markov/RNN heads][DS-MARKOV], [loss][DS-LOSS].
- [Eval entrypoint][DS-EVAL-ENTRY], [base evaluator][DS-BASE-EVAL], [DSpark evaluator][DS-EVALUATOR], [confidence recorder][DS-CONFIDENCE].

### Код и документация DFlash / SpecForge

- [Официальный DFlash repo][DF-ROOT].
- [SpecForge README][SF-ROOT], [installation][SF-INSTALL], [dependencies][SF-PYPROJECT], [data preparation][SF-DATA], [training][SF-TRAINING], [disaggregated training][SF-DISAGG].
- [SpecForge data conversion][SF-PREP-DATA], [regeneration][SF-GEN], [feature preparation][SF-CAPTURE], [capture plugin][SF-CAPTURE-PLUGIN].
- [DFlash model config][SF-DF-CONFIG], [offline DFlash recipe][SF-DF-OFFLINE], [managed-local DFlash recipe][SF-DF-ONLINE].
- [DSpark model config][SF-DS-CONFIG], [offline DSpark recipe][SF-DS-OFFLINE], [external-online DSpark recipe][SF-DS-ONLINE].

### Датасеты

- [NVIDIA Nemotron Post-Training Dataset V2][NEMOTRON].
- [CodeAlpaca README][CODEALPACA].
- [Open-PerfectBlend card][PERFECTBLEND] и [machine-readable README][PERFECTBLEND-RAW].
- [MetaMathQA][METAMATH], [UltraInteract SFT][ULTRAINTERACT], [UltraChat 200k][ULTRACHAT], [Orca Math][ORCA].
- [UltraFeedback Binarized][ULTRAFEEDBACK], [Evol-CodeAlpaca][EVOL-CODEALPACA], [AutoIF-instruct-61k][AUTOIF], [Post-Training Data Flywheel][FLYWHEEL], [LMSYS Arena ShareGPT][ARENA-DATA].

<!-- Ссылки зафиксированных источников -->

[DF-PAPER]: https://arxiv.org/abs/2602.06036v2
[DS-PAPER]: https://arxiv.org/abs/2607.05147v1
[DF-ROOT]: https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a
[DF-BLOG]: https://z-lab.ai/projects/dflash/
[SF-BLOG-2025]: https://www.lmsys.org/blog/2025-07-25-spec-forge/
[SF-BLOG-2026]: https://www.lmsys.org/blog/2026-08-04-specforge-v0-3/
[PERFECTBLEND]: https://huggingface.co/datasets/mlabonne/open-perfectblend
[PERFECTBLEND-RAW]: https://huggingface.co/datasets/mlabonne/open-perfectblend/raw/main/README.md
[NEMOTRON]: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
[CODEALPACA]: https://github.com/sahil280114/codealpaca#data-generation-process
[METAMATH]: https://huggingface.co/datasets/meta-math/MetaMathQA
[ULTRAINTERACT]: https://huggingface.co/datasets/openbmb/UltraInteract_sft
[ULTRACHAT]: https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
[ORCA]: https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k
[ULTRAFEEDBACK]: https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized
[EVOL-CODEALPACA]: https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1
[AUTOIF]: https://huggingface.co/datasets/Post-training-Data-Flywheel/AutoIF-instruct-61k
[FLYWHEEL]: https://github.com/shizhediao/Post-Training-Data-Flywheel
[ARENA-DATA]: https://huggingface.co/datasets/mlabonne/lmsys-arena-human-preference-55k-sharegpt
[DS-ROOT]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/README.md
[DS-REQ]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/requirements.txt
[DS-CONSTANTS]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/constant/public.py
[DS-DATA]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/README.md
[DS-SPLIT]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/download_and_split.py
[DS-GEN]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/generate_train_data.py
[DS-SERVE]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/launch_sglang_server.sh
[DS-CACHE-PREP]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/prepare_target_cache.py
[DS-CACHE-DATA]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/target_cache_dataset.py
[DS-PARSER]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/data/parser.py
[DS-CONFIG]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py
[DS-DF-CONFIG]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dflash/dflash_qwen3_4b.py
[DS-ENTRY]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/train.py
[DS-TRAIN-SH]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/train/train.sh
[DS-TRAINER]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/base_trainer.py
[DS-OPTIM]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/optim.py
[DS-CKPT]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/trainer/ckpt_manager.py
[DS-CONFIG-UTIL]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/config.py
[DS-MODEL]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/qwen3/modeling.py
[DS-COMMON]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/common.py
[DS-MARKOV]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/markov_head.py
[DS-LOSS]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py
[DS-EVAL-ENTRY]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py
[DS-BASE-EVAL]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/base_evaluator.py
[DS-EVALUATOR]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/evaluator.py
[DS-CONFIDENCE]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/eval/dspark/confidence_head.py
[SF-ROOT]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/README.md
[SF-INSTALL]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/docs/sections/get_started/installation.md
[SF-PYPROJECT]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/pyproject.toml
[SF-DATA]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/docs/sections/basic_usage/data_preparation.md
[SF-TRAINING]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/docs/sections/basic_usage/training.md
[SF-DISAGG]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/docs/sections/basic_usage/disaggregated_training.md
[SF-PREP-DATA]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/scripts/prepare_data.py
[SF-GEN]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/scripts/regenerate_train_data.py
[SF-CAPTURE]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/scripts/prepare_hidden_states.py
[SF-CAPTURE-PLUGIN]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/plugins/sglang-spec-capture/README.md
[SF-DF-CONFIG]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/configs/qwen3-8b-dflash.json
[SF-DF-OFFLINE]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/examples/configs/offline/colocated/qwen3-8b-dflash.yaml
[SF-DF-ONLINE]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/examples/configs/online/disaggregated/managed-local/qwen3-8b-dflash-1server-dp7.yaml
[SF-DS-CONFIG]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/configs/qwen3-4b-dspark.json
[SF-DS-OFFLINE]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/examples/configs/offline/colocated/qwen3-4b-dspark.yaml
[SF-DS-ONLINE]: https://github.com/sgl-project/SpecForge/blob/8b45bab869ef6b9d539a2626629ad31af51491f7/examples/configs/online/disaggregated/external/qwen3-4b-dspark.yaml
