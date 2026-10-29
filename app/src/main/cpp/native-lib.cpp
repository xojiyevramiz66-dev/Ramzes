#include <jni.h>
#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "llama.h"

#define LOG_TAG "RAMZES_AI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

std::mutex g_mutex;

llama_model *g_model = nullptr;
llama_context *g_context = nullptr;
llama_sampler *g_sampler = nullptr;

bool g_backend_initialized = false;

constexpr int32_t DEFAULT_CONTEXT_SIZE = 2048;
constexpr int32_t BATCH_SIZE = 128;
constexpr int32_t MAX_PROMPT_TOKENS = 1536;
constexpr int32_t MAX_GENERATION_TOKENS = 512;
constexpr int32_t THREAD_COUNT = 4;

std::string jstringToString(JNIEnv *env, jstring value) {
    if (value == nullptr) {
        return {};
    }

    const char *chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }

    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

jstring makeJavaString(JNIEnv *env, const std::string &value) {
    return env->NewStringUTF(value.c_str());
}

void releaseModelLocked() {
    if (g_sampler != nullptr) {
        llama_sampler_free(g_sampler);
        g_sampler = nullptr;
    }

    if (g_context != nullptr) {
        llama_free(g_context);
        g_context = nullptr;
    }

    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
}

/*
 * Исправленная токенизация под актуальный API llama.cpp.
 * Возвращает нужное количество токенов (по модулю).
 */
bool tokenizePrompt(
    const llama_vocab *vocab,
    const std::string &prompt,
    std::vector<llama_token> &tokens
) {
    if (vocab == nullptr || prompt.empty()) {
        LOGE("Пустой промпт или словарь модели");
        return false;
    }

    if (prompt.size() > static_cast<size_t>(INT32_MAX)) {
        LOGE("Промпт слишком велик");
        return false;
    }

    const int32_t promptLength = static_cast<int32_t>(prompt.size());

    // Первый вызов: определяем требуемый размер буфера.
    // В новых версиях llama.cpp возвращает отрицательное число.
    int32_t required = llama_tokenize(
        vocab,
        prompt.c_str(),
        promptLength,
        nullptr,
        0,
        true,
        true
    );

    if (required < 0) {
        required = -required;
    }

    if (required <= 0 || required > 100000) {
        LOGE("Недопустимый размер токенизации: %d", required);
        return false;
    }

    tokens.resize(static_cast<size_t>(required));

    // Второй вызов: реальная токенизация.
    const int32_t actual = llama_tokenize(
        vocab,
        prompt.c_str(),
        promptLength,
        tokens.data(),
        required,
        true,
        true
    );

    if (actual < 0) {
        LOGE("Ошибка токенизации: %d", actual);
        tokens.clear();
        return false;
    }

    tokens.resize(static_cast<size_t>(actual));

    if (tokens.empty()) {
        LOGE("После токенизации не осталось токенов");
        return false;
    }

    LOGI("Токенизация успешна: %d токенов", actual);
    return true;
}

bool decodePromptInBatches(
    llama_context *context,
    const std::vector<llama_token> &tokens
) {
    if (context == nullptr || tokens.empty()) {
        return false;
    }

    const int32_t total = static_cast<int32_t>(tokens.size());
    LOGI("Декодирование промпта батчами, всего токенов: %d", total);

    for (int32_t offset = 0; offset < total; offset += BATCH_SIZE) {
        const int32_t count = std::min(BATCH_SIZE, total - offset);

        llama_batch batch = llama_batch_init(count, 0, 1);

        if (batch.token == nullptr || batch.pos == nullptr ||
            batch.n_seq_id == nullptr || batch.seq_id == nullptr ||
            batch.logits == nullptr) {
            LOGE("Не удалось выделить память для батча");
            llama_batch_free(batch);
            return false;
        }

        batch.n_tokens = count;

        for (int32_t j = 0; j < count; ++j) {
            const int32_t index = offset + j;

            batch.token[j] = tokens[static_cast<size_t>(index)];
            batch.pos[j] = static_cast<llama_pos>(index);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            // Логиты нужны только для самого последнего токена промпта
            batch.logits[j] = (index == total - 1);
        }

        const int result = llama_decode(context, batch);
        llama_batch_free(batch);

        if (result != 0) {
            LOGE("llama_decode завершился с ошибкой %d на позиции %d", result, offset);
            return false;
        }
    }

    return true;
}

bool initializeBackend() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
        LOGI("llama.cpp backend инициализирован");
    }
    return true;
}

} // namespace

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeLoadModel(
    JNIEnv *env,
    jobject /* thiz */,
    jstring modelPath,
    jint requestedContext
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    releaseModelLocked();

    const std::string path = jstringToString(env, modelPath);
    if (path.empty()) {
        return makeJavaString(env, "ERROR: Путь к модели пуст.");
    }

    LOGI("Загрузка модели: %s", path.c_str());

    try {
        initializeBackend();

        llama_model_params modelParams = llama_model_default_params();
        modelParams.n_gpu_layers = 0; // CPU-режим

        g_model = llama_model_load_from_file(path.c_str(), modelParams);

        if (g_model == nullptr) {
            LOGE("Не удалось загрузить модель из файла");
            releaseModelLocked();
            return makeJavaString(env, "ERROR: Не удалось загрузить модель. Проверьте файл GGUF.");
        }

        int32_t contextSize = requestedContext;
        if (contextSize <= 0) {
            contextSize = DEFAULT_CONTEXT_SIZE;
        }
        // Ограничиваем контекст разумными рамками
        contextSize = std::max(512, std::min(contextSize, DEFAULT_CONTEXT_SIZE));

        llama_context_params contextParams = llama_context_default_params();
        contextParams.n_ctx = static_cast<uint32_t>(contextSize);
        contextParams.n_batch = BATCH_SIZE;
        contextParams.n_ubatch = BATCH_SIZE;
        contextParams.n_threads = THREAD_COUNT;
        contextParams.n_threads_batch = THREAD_COUNT;

        g_context = llama_init_from_model(g_model, contextParams);

        if (g_context == nullptr) {
            LOGE("Не удалось создать контекст");
            releaseModelLocked();
            return makeJavaString(env, "ERROR: Не удалось создать контекст. Мало памяти?");
        }

        // Используем greedy для предсказуемости (без зацикливаний)
        g_sampler = llama_sampler_init_greedy();

        if (g_sampler == nullptr) {
            LOGE("Не удалось создать sampler");
            releaseModelLocked();
            return makeJavaString(env, "ERROR: Не удалось создать генератор.");
        }

        LOGI("Модель загружена успешно. Контекст=%d, batch=%d", contextSize, BATCH_SIZE);
        return makeJavaString(env, "OK");

    } catch (...) {
        LOGE("Неизвестная ошибка во время загрузки");
        releaseModelLocked();
        return makeJavaString(env, "ERROR: Исключение при загрузке.");
    }
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_ramzes_ai_MainActivity_nativeGenerate(
    JNIEnv *env,
    jobject /* thiz */,
    jstring userPrompt,
    jint maxTokens
) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_model == nullptr || g_context == nullptr || g_sampler == nullptr) {
        return makeJavaString(env, "Ошибка: модель не загружена.");
    }

    const std::string prompt = jstringToString(env, userPrompt);
    if (prompt.empty()) {
        return makeJavaString(env, "Введите сообщение.");
    }

    const auto start = std::chrono::steady_clock::now();
    LOGI("--- Начало генерации ---");

    const llama_vocab *vocab = llama_model_get_vocab(g_model);
    std::vector<llama_token> tokens;

    if (!tokenizePrompt(vocab, prompt, tokens)) {
        return makeJavaString(env, "Ошибка токенизации промпта.");
    }

    if (tokens.size() > MAX_PROMPT_TOKENS) {
        LOGE("Промпт слишком длинный: %zu токенов", tokens.size());
        return makeJavaString(env, "Запрос слишком длинный. Сократите историю.");
    }

    const int32_t generationLimit = std::max(
        1, std::min(static_cast<int32_t>(maxTokens), MAX_GENERATION_TOKENS)
    );

    const uint32_t contextSize = llama_n_ctx(g_context);
    if (tokens.size() + static_cast<size_t>(generationLimit) > contextSize) {
        return makeJavaString(env, "Недостаточно места в контексте. Сократите историю.");
    }

    llama_memory_t memory = llama_get_memory(g_context);
    if (memory != nullptr) {
        llama_memory_clear(memory, true);
    }

    llama_sampler_reset(g_sampler);

    if (!decodePromptInBatches(g_context, tokens)) {
        if (memory != nullptr) llama_memory_clear(memory, true);
        return makeJavaString(env, "Ошибка обработки промпта.");
    }

    std::string answer;
    answer.reserve(static_cast<size_t>(generationLimit) * 8);

    // Начинаем позицию с количества токенов промпта
    llama_pos n_past = static_cast<llama_pos>(tokens.size());
    bool decodeFailed = false;
    int32_t generatedCount = 0;

    for (int32_t i = 0; i < generationLimit; ++i) {
        const llama_token token = llama_sampler_sample(g_sampler, g_context, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            LOGI("Достигнут EOG (конец текста) на шаге %d", i);
            break;
        }

        char piece[4096];
        const int32_t pieceSize = llama_token_to_piece(
            vocab, token, piece, static_cast<int32_t>(sizeof(piece)), 0, true
        );

        if (pieceSize < 0) {
            LOGE("Ошибка преобразования токена: %d", pieceSize);
            decodeFailed = true;
            break;
        }

        if (pieceSize > 0) {
            answer.append(piece, static_cast<size_t>(pieceSize));
        }
        generatedCount++;

        if (i + 1 >= generationLimit) {
            break;
        }

        // Подготавливаем батч для следующего шага
        llama_batch next = llama_batch_init(1, 0, 1);
        if (next.token == nullptr || next.pos == nullptr ||
            next.n_seq_id == nullptr || next.seq_id == nullptr ||
            next.logits == nullptr) {
            LOGE("Не удалось выделить батч");
            llama_batch_free(next);
            decodeFailed = true;
            break;
        }

        next.n_tokens = 1;
        next.token[0] = token;
        next.pos[0] = n_past; // Используем счетчик позиции
        next.n_seq_id[0] = 1;
        next.seq_id[0][0] = 0;
        next.logits[0] = true;

        const int result = llama_decode(g_context, next);
        llama_batch_free(next);

        if (result != 0) {
            LOGE("Ошибка decode на шаге %d: код %d", i, result);
            decodeFailed = true;
            break;
        }

        n_past++; // Инкрементируем позицию для следующего токена
    }

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start
    ).count();

    LOGI("--- Генерация завершена: %d токенов, %zu байт, %lld мс ---", 
         generatedCount, answer.size(), static_cast<long long>(elapsed));

    if (decodeFailed && answer.empty()) {
        if (memory != nullptr) llama_memory_clear(memory, true);
        return makeJavaString(env, "Ошибка генерации. Проверьте Logcat.");
    }

    if (answer.empty()) {
        return makeJavaString(env, "Модель не вернула текст. Возможно, несовместим шаблон чата.");
    }

    return makeJavaString(env, answer);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_ramzes_ai_MainActivity_nativeUnloadModel(
    JNIEnv * /* env */,
    jobject /* thiz */
) {
    std::lock_guard<std::mutex> lock(g_mutex);
    releaseModelLocked();
    LOGI("Модель выгружена из памяти");
}
